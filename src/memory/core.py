"""Core memory store — all database operations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .models import Document, Memory

DB_PATH = Path.home() / ".claude" / "tools" / "memory" / "data" / "sqlite_vec.db"

EMBEDDING_DIM = 384

_F32_STRUCT = struct.Struct(f"<{EMBEDDING_DIM}f")

# --- Confidence decay rates (per day) ---
# Higher = slower decay.  decision/pattern/reference are near-permanent.
DECAY_RATES: dict[str, float] = {
    "decision": 0.999,
    "pattern": 0.999,
    "reference": 0.999,
    "error": 0.99,
    "learning": 0.99,
    "observation": 0.97,
    "note": 0.97,
}
DEFAULT_DECAY_RATE = 0.98

# --- Importance auto-inference rules ---
_IMPORTANCE_KEYWORDS: list[tuple[float, re.Pattern]] = [
    (0.9, re.compile(r"\b(IMPORTANT|CRITICAL|MUST|BREAKING)\b", re.IGNORECASE)),
    (0.8, re.compile(r"\b(NEVER|ALWAYS|WARNING|DANGER)\b", re.IGNORECASE)),
]
_IMPORTANCE_BY_TYPE: dict[str, float] = {
    "decision": 0.8,
    "pattern": 0.7,
    "error": 0.7,
    "reference": 0.6,
    "learning": 0.6,
    "observation": 0.4,
    "note": 0.4,
}

# --- Composite scoring defaults ---
DEFAULT_SCORING_WEIGHTS = (0.6, 0.2, 0.2)  # similarity, importance, recency


def _serialize_f32(vec: object) -> bytes:
    """Serialize embedding to little-endian bytes for sqlite-vec."""
    if hasattr(vec, "tobytes"):
        return vec.tobytes()  # numpy array, already float32
    return _F32_STRUCT.pack(*vec)


def _safe_tags(raw: str | None) -> list:
    """Parse JSON tags string into a list, returning [] on failure."""
    try:
        return json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []


_TIME_DELTA_RE = re.compile(r"(\d+)\s+(day|week|month|year)s?\s+ago")


def _parse_time_expr(expr: str) -> datetime:
    """Parse natural language time expressions into a datetime."""
    expr = expr.lower().strip()
    now = datetime.now(tz=timezone.utc)

    mappings = {
        "today": timedelta(days=0),
        "yesterday": timedelta(days=1),
        "last week": timedelta(weeks=1),
        "last month": timedelta(days=30),
        "last year": timedelta(days=365),
    }
    if expr in mappings:
        dt = now - mappings[expr]
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # "N days/weeks/months ago"
    m = _TIME_DELTA_RE.match(expr)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
            "year": timedelta(days=n * 365),
        }[unit]
        return now - delta

    raise ValueError(f"Cannot parse time expression: {expr!r}")


def _sanitize_fts_query(query: str) -> str:
    """Escape user input for safe use in FTS5 MATCH expressions.

    Wraps each whitespace-delimited token in double quotes so FTS5
    treats hyphens, colons, asterisks, and boolean keywords as literals.
    """
    query = query.strip()
    if not query:
        return ""
    tokens = query.split()
    quoted = []
    for token in tokens:
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            quoted.append(token)
        else:
            quoted.append(f'"{token.replace(chr(34), chr(34)*2)}"')
    return " ".join(quoted)


def infer_importance(content: str, memory_type: str) -> float:
    """Auto-infer importance from content keywords and memory type."""
    # Keyword rules take priority (highest match wins)
    for score, pattern in _IMPORTANCE_KEYWORDS:
        if pattern.search(content):
            return score
    # Fall back to type-based default
    return _IMPORTANCE_BY_TYPE.get(memory_type, 0.5)


def compute_confidence(
    base_confidence: float,
    memory_type: str,
    last_recalled_at: float | None,
    created_at: float,
) -> float:
    """Compute decayed confidence for a memory."""
    rate = DECAY_RATES.get(memory_type, DEFAULT_DECAY_RATE)
    anchor = last_recalled_at if last_recalled_at else created_at
    days = max(0.0, (time.time() - anchor) / 86400)
    return base_confidence * (rate ** days)


def compute_recency(created_at: float) -> float:
    """Recency score: 1/(1 + days_since_creation)."""
    days = max(0.0, (time.time() - created_at) / 86400)
    return 1.0 / (1.0 + days)


def normalize_importance(recall_count: int, max_recall: int) -> float:
    """Normalize recall-based importance to 0-1 range."""
    if max_recall <= 0:
        return 0.0
    return min(1.0, recall_count / max_recall)


class MemoryStore:
    """Synchronous SQLite memory store with sqlite-vec embeddings."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._conn: sqlite3.Connection | None = None

    @staticmethod
    def _sqlite_vec_path() -> str:
        """Resolve the sqlite-vec loadable extension path without importing numpy."""
        from importlib.util import find_spec
        spec = find_spec("sqlite_vec")
        if spec is None or spec.origin is None:
            raise ImportError("sqlite_vec package not found")
        from os.path import dirname, join, normpath
        return normpath(join(dirname(spec.origin), "vec0"))

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
            self._conn.enable_load_extension(True)
            self._conn.load_extension(self._sqlite_vec_path())
            self._conn.enable_load_extension(False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=15000")
            self._conn.execute("PRAGMA cache_size=20000")
            self._migrate_stats_tables()
        return self._conn

    def _begin_immediate(self) -> sqlite3.Connection:
        """Begin an IMMEDIATE transaction (acquires write lock upfront).

        Use for all write operations to prevent TOCTOU races when
        multiple agents share the same database.
        """
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        return conn

    def _migrate_stats_tables(self) -> None:
        """One-time migration: create operation_events table and recall columns.

        Uses executescript which issues its own implicit COMMIT.
        ALTERs are DDL and auto-commit in autocommit mode (isolation_level=None).
        """
        conn = self._conn
        # Create events table (executescript manages its own transactions)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS operation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                operation TEXT NOT NULL,
                duration_ms REAL,
                query TEXT,
                search_mode TEXT,
                result_count INTEGER,
                top_similarity REAL,
                content_hash TEXT,
                dedup_used BOOLEAN DEFAULT 0,
                duplicate_detected BOOLEAN DEFAULT 0,
                duplicate_similarity REAL,
                chars_returned INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON operation_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_op ON operation_events(operation);
            """
        )
        # Add recall columns (idempotent, DDL auto-commits)
        for col_sql in (
            "ALTER TABLE memories ADD COLUMN recall_count INTEGER DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN last_recalled_at REAL DEFAULT NULL",
            "ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0",
            "ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.5",
        ):
            try:
                conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass

        # FTS5 virtual table for BM25 keyword search (zero cold start, no model)
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(content, content='memories', content_rowid='id',
                       tokenize='porter ascii');
            """
        )
        # Backfill existing rows — only if memories table already exists
        memories_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if memories_exists:
            conn.execute(
                "INSERT OR IGNORE INTO memory_fts(rowid, content) "
                "SELECT id, content FROM memories WHERE deleted_at IS NULL"
            )

        self._migrate_documents_tables()

    def _migrate_documents_tables(self) -> None:
        """One-time migration: create documents table, embeddings, and FTS."""
        conn = self._conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash    TEXT NOT NULL,
                title           TEXT NOT NULL,
                body            TEXT NOT NULL,
                summary         TEXT NOT NULL,
                doc_type        TEXT NOT NULL DEFAULT 'document',
                tags            TEXT DEFAULT '[]',
                metadata        TEXT DEFAULT '{}',
                created_at      REAL,
                updated_at      REAL,
                created_at_iso  TEXT,
                updated_at_iso  TEXT,
                deleted_at      REAL DEFAULT NULL,
                version         INTEGER DEFAULT 1,
                recall_count    INTEGER DEFAULT 0,
                last_recalled_at REAL DEFAULT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
            CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
            """
        )

        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS document_embeddings
                USING vec0(summary_embedding FLOAT[384] distance_metric=cosine);
            """
        )

        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                title, body, content='documents', content_rowid='id',
                tokenize='porter ascii'
            );
            """
        )

        # Backfill FTS for existing documents
        docs_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
        if docs_exists:
            conn.execute(
                "INSERT OR IGNORE INTO document_fts(rowid, title, body) "
                "SELECT id, title, body FROM documents WHERE deleted_at IS NULL"
            )

    @staticmethod
    def _rollback_safe(conn: sqlite3.Connection) -> None:
        """Roll back the current transaction, ignoring errors if none is active."""
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Event Tracking ---

    def _track_event(self, conn: sqlite3.Connection, operation: str, **kwargs) -> None:
        """Append an operation event for analytics."""
        conn.execute(
            """INSERT INTO operation_events
               (timestamp, operation, duration_ms, query, search_mode,
                result_count, top_similarity, content_hash,
                dedup_used, duplicate_detected, duplicate_similarity, chars_returned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), operation,
             kwargs.get("duration_ms"),
             kwargs.get("query"),
             kwargs.get("search_mode"),
             kwargs.get("result_count"),
             kwargs.get("top_similarity"),
             kwargs.get("content_hash"),
             kwargs.get("dedup_used", False),
             kwargs.get("duplicate_detected", False),
             kwargs.get("duplicate_similarity"),
             kwargs.get("chars_returned")),
        )

    # --- Store ---

    def store(
        self,
        content: str,
        tags: list[str] | None = None,
        memory_type: str = "note",
        metadata: dict | None = None,
        dedup_threshold: float | None = None,
        importance: float | None = None,
        _embedding: object | None = None,
    ) -> dict:
        """Store a single memory. Returns dict with hash and status.

        If dedup_threshold is set (0.0-1.0), checks for semantically similar
        memories before storing. Embeds content once and reuses for both the
        similarity check and the stored embedding.

        importance: explicit 0-1 score. If None, auto-inferred from content/type.
        """
        start = time.time()
        # Normalize string tags to list (defensive — callers may pass comma-separated strings)
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        imp = importance if importance is not None else infer_importance(content, memory_type)
        mem = Memory(
            content=content,
            tags=tags or [],
            memory_type=memory_type,
            metadata=metadata or {},
            importance=imp,
        )

        conn = self._begin_immediate()
        try:
            # Check for exact duplicate (hash-based) — no embedding needed
            existing = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (mem.content_hash,),
            ).fetchone()
            if existing:
                duration_ms = (time.time() - start) * 1000
                self._track_event(conn, "store",
                    duration_ms=duration_ms,
                    content_hash=mem.content_hash,
                    dedup_used=dedup_threshold is not None,
                    duplicate_detected=True,
                )
                conn.execute("COMMIT")
                return {
                    "content_hash": mem.content_hash,
                    "status": "duplicate",
                    "message": "Memory with this content already exists",
                }

            # Check for soft-deleted entry with same hash — revive it
            tombstone = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL",
                (mem.content_hash,),
            ).fetchone()
            if tombstone:
                tomb_id = tombstone["id"]
                # Compute embedding for the revived entry
                if _embedding is not None:
                    embedding = _embedding
                else:
                    from .embeddings import get_model
                    embedding = get_model().embed(content)
                conn.execute(
                    """UPDATE memories
                       SET content = ?, tags = ?, memory_type = ?, metadata = ?,
                           updated_at = ?, updated_at_iso = ?,
                           confidence = ?, importance = ?,
                           deleted_at = NULL
                       WHERE id = ?""",
                    (mem.content, json.dumps(mem.tags), mem.memory_type,
                     json.dumps(mem.metadata), mem.updated_at, mem.updated_at_iso,
                     mem.confidence, mem.importance, tomb_id),
                )
                # Re-insert embedding
                conn.execute(
                    "INSERT OR REPLACE INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
                    (tomb_id, _serialize_f32(embedding)),
                )
                # Re-insert FTS
                conn.execute(
                    "INSERT OR REPLACE INTO memory_fts(rowid, content) VALUES (?, ?)",
                    (tomb_id, content),
                )
                duration_ms = (time.time() - start) * 1000
                self._track_event(conn, "store",
                    duration_ms=duration_ms,
                    content_hash=mem.content_hash,
                    dedup_used=dedup_threshold is not None,
                    duplicate_detected=False,
                )
                conn.execute("COMMIT")
                return {"content_hash": mem.content_hash, "status": "revived"}

            # Only compute embedding after confirming not an exact duplicate
            if _embedding is not None:
                embedding = _embedding
            else:
                from .embeddings import get_model
                embedding = get_model().embed(content)

            # Similarity-based dedup (scoped to same memory_type)
            if dedup_threshold is not None:
                similar = self._search_semantic(
                    conn, query=None, limit=5, tags=None,
                    time_expr=None, after=None, before=None,
                    _embedding=embedding,
                )
                # Only dedup against same memory_type to avoid false positives
                # across unrelated content (e.g. reference vs decision)
                same_type = [s for s in similar if s.get("memory_type") == memory_type]
                # Further scope by tags: if the new memory has tags, only compare
                # against memories sharing at least one tag (prevents cross-project
                # false positives from shared sentence structure)
                if mem.tags:
                    same_type = [
                        s for s in same_type
                        if any(t in s.get("tags", []) for t in mem.tags)
                    ]
                if same_type and same_type[0].get("similarity", 0) >= dedup_threshold:
                    duration_ms = (time.time() - start) * 1000
                    self._track_event(conn, "store",
                        duration_ms=duration_ms,
                        content_hash=mem.content_hash,
                        dedup_used=True,
                        duplicate_detected=True,
                        duplicate_similarity=same_type[0]["similarity"],
                    )
                    conn.execute("COMMIT")
                    return {
                        "content_hash": mem.content_hash,
                        "status": "duplicate",
                        "message": f"Similar memory exists (similarity={same_type[0]['similarity']:.2f})",
                        "similar_hash": same_type[0]["content_hash"],
                    }

            # Insert memory
            conn.execute(
                """
                INSERT INTO memories
                    (content_hash, content, tags, memory_type, metadata,
                     created_at, updated_at, created_at_iso, updated_at_iso,
                     confidence, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                mem.to_row(),
            )
            mem_id = conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (mem.content_hash,)
            ).fetchone()["id"]

            # Store pre-computed embedding
            conn.execute(
                "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
                (mem_id, _serialize_f32(embedding)),
            )

            # Keep FTS index in sync
            conn.execute(
                "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                (mem_id, content),
            )

            duration_ms = (time.time() - start) * 1000
            self._track_event(conn, "store",
                duration_ms=duration_ms,
                content_hash=mem.content_hash,
                dedup_used=dedup_threshold is not None,
                duplicate_detected=False,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {
            "content_hash": mem.content_hash,
            "status": "stored",
            "message": "Memory stored successfully",
        }

    def store_batch(
        self, items: list[dict], dedup_threshold: float | None = None,
    ) -> list[dict]:
        """Store multiple memories. Each item: {content, tags?, memory_type?, metadata?}.

        Batches embedding computation for all items upfront, then stores individually.
        """
        if not items:
            return []

        # Batch-compute all embeddings once
        from .embeddings import get_model
        contents = [item["content"] for item in items]
        embeddings = get_model().embed_batch(contents)

        results = []
        for item, embedding in zip(items, embeddings):
            # Normalize string tags to list
            raw_tags = item.get("tags", [])
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            result = self.store(
                content=item["content"],
                tags=raw_tags,
                memory_type=item.get("memory_type") or item.get("type") or "note",
                metadata=item.get("metadata", {}),
                dedup_threshold=dedup_threshold,
                importance=item.get("importance"),
                _embedding=embedding,
            )
            results.append(result)
        return results

    # --- Search ---

    def search(
        self,
        query: str | None = None,
        mode: str = "semantic",
        limit: int = 10,
        tags: list[str] | None = None,
        time_expr: str | None = None,
        after: str | None = None,
        before: str | None = None,
        scoring_weights: tuple[float, float, float] | None = None,
    ) -> list[dict]:
        """Search memories. Modes: semantic, exact, hybrid.

        scoring_weights: (similarity_w, importance_w, recency_w) for composite
        scoring. Defaults to (0.6, 0.2, 0.2). Only applies to semantic/hybrid.
        """
        start = time.time()
        conn = self._get_conn()

        # Read phase — no write lock needed yet
        if mode == "exact":
            results = self._search_exact(conn, query, limit, tags, time_expr, after, before)
        elif mode in ("semantic", "hybrid"):
            results = self._search_semantic(
                conn, query, limit, tags, time_expr, after, before,
                scoring_weights=scoring_weights,
            )
        elif mode == "fts":
            results = self._search_fts(
                conn, query, limit, tags, time_expr, after, before,
                scoring_weights=scoring_weights,
            )
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        # Write phase — acquire lock for event + recall tracking
        duration_ms = (time.time() - start) * 1000
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._track_event(conn, "search",
                duration_ms=duration_ms,
                query=query,
                search_mode=mode,
                result_count=len(results),
                top_similarity=results[0].get("similarity") if results else None,
                chars_returned=sum(len(m.get("content", "")) for m in results),
            )
            # Update recall counts and reset confidence (reinforcement) for returned memories
            now = time.time()
            for m in results:
                conn.execute(
                    "UPDATE memories SET recall_count = recall_count + 1, "
                    "last_recalled_at = ?, confidence = 1.0 WHERE content_hash = ?",
                    (now, m["content_hash"]),
                )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return results

    def _build_time_filter(
        self,
        time_expr: str | None,
        after: str | None,
        before: str | None,
    ) -> tuple[str, list]:
        """Build SQL WHERE clause for time filters."""
        clauses = []
        params = []

        if time_expr:
            dt = _parse_time_expr(time_expr)
            clauses.append("m.created_at >= ?")
            params.append(dt.timestamp())

        if after:
            dt = datetime.fromisoformat(after).replace(tzinfo=timezone.utc)
            clauses.append("m.created_at >= ?")
            params.append(dt.timestamp())

        if before:
            dt = datetime.fromisoformat(before).replace(tzinfo=timezone.utc)
            clauses.append("m.created_at <= ?")
            params.append(dt.timestamp())

        return (" AND ".join(clauses), params) if clauses else ("", [])

    def _filter_by_tags(self, memories: list[dict], tags: list[str]) -> list[dict]:
        """Filter memory dicts by tags (any match)."""
        if not tags:
            return memories
        tag_set = {t.lower() for t in tags}
        return [
            m
            for m in memories
            if any(t.lower() in tag_set for t in m.get("tags", []))
        ]

    def _search_semantic(
        self,
        conn: sqlite3.Connection,
        query: str | None,
        limit: int,
        tags: list[str] | None,
        time_expr: str | None,
        after: str | None,
        before: str | None,
        _embedding: object | None = None,
        scoring_weights: tuple[float, float, float] | None = None,
    ) -> list[dict]:
        if _embedding is not None:
            embedding = _embedding
        elif query:
            from .embeddings import get_model
            embedding = get_model().embed(query)
        else:
            return []
        # Fetch more than needed to allow post-filtering and re-ranking
        fetch_limit = max(limit * 5, 50) if (tags or time_expr or after or before) else max(limit * 3, 30)

        rows = conn.execute(
            """
            SELECT e.rowid, e.distance
            FROM memory_embeddings e
            WHERE e.content_embedding MATCH ?
            ORDER BY e.distance
            LIMIT ?
            """,
            (_serialize_f32(embedding), fetch_limit),
        ).fetchall()

        if not rows:
            return []

        rowids = [r["rowid"] for r in rows]
        distances = {r["rowid"]: r["distance"] for r in rows}

        placeholders = ",".join("?" * len(rowids))
        time_clause, time_params = self._build_time_filter(time_expr, after, before)

        sql = f"""
            SELECT * FROM memories m
            WHERE m.id IN ({placeholders})
              AND m.deleted_at IS NULL
        """
        params = list(rowids)
        if time_clause:
            sql += f" AND {time_clause}"
            params.extend(time_params)

        mem_rows = conn.execute(sql, params).fetchall()
        # Build lookup by id, then iterate in original distance order
        row_by_id = {row["id"]: dict(row) for row in mem_rows}

        w_sim, w_imp, w_rec = scoring_weights or DEFAULT_SCORING_WEIGHTS
        memories = []
        for rid in rowids:
            row = row_by_id.get(rid)
            if row is None:
                continue  # filtered out by time/deleted_at
            mem = Memory.from_row(row)
            d = mem.to_dict()
            similarity = round(1.0 - distances.get(rid, 1.0), 4)
            d["similarity"] = similarity

            # Compute decayed confidence
            conf = compute_confidence(
                mem.confidence, mem.memory_type,
                row.get("last_recalled_at"), mem.created_at,
            )
            d["confidence"] = round(conf, 4)

            # Recall tracking fields
            d["recall_count"] = row.get("recall_count", 0) or 0
            d["last_recalled_at"] = row.get("last_recalled_at")

            # Composite score
            recency = compute_recency(mem.created_at)
            d["score"] = round(
                w_sim * similarity + w_imp * mem.importance + w_rec * recency, 4
            )
            memories.append(d)

        if tags:
            memories = self._filter_by_tags(memories, tags)

        # Re-rank by composite score
        memories.sort(key=lambda m: m["score"], reverse=True)

        return memories[:limit]

    def _search_exact(
        self,
        conn: sqlite3.Connection,
        query: str | None,
        limit: int,
        tags: list[str] | None,
        time_expr: str | None,
        after: str | None,
        before: str | None,
    ) -> list[dict]:
        time_clause, time_params = self._build_time_filter(time_expr, after, before)

        sql = "SELECT * FROM memories m WHERE m.deleted_at IS NULL"
        params: list = []

        if query:
            sql += " AND m.content LIKE ?"
            params.append(f"%{query}%")

        if time_clause:
            sql += f" AND {time_clause}"
            params.extend(time_params)

        sql += " ORDER BY m.created_at DESC LIMIT ?"
        params.append(limit * 5 if tags else limit)

        rows = conn.execute(sql, params).fetchall()
        memories = []
        for r in rows:
            row_dict = dict(r)
            d = Memory.from_row(row_dict).to_dict()
            d["recall_count"] = row_dict.get("recall_count", 0) or 0
            d["last_recalled_at"] = row_dict.get("last_recalled_at")
            memories.append(d)

        if tags:
            memories = self._filter_by_tags(memories, tags)

        return memories[:limit]

    def _search_fts(
        self,
        conn: sqlite3.Connection,
        query: str | None,
        limit: int,
        tags: list[str] | None,
        time_expr: str | None,
        after: str | None,
        before: str | None,
        scoring_weights: tuple[float, float, float] | None = None,
    ) -> list[dict]:
        """BM25 full-text search via FTS5. No embedding model required."""
        if not query:
            return []

        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []

        fetch_limit = max(limit * 5, 50) if (tags or time_expr or after or before) else max(limit * 3, 30)

        fts_rows = conn.execute(
            "SELECT rowid, rank FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
            (safe_query, fetch_limit),
        ).fetchall()

        if not fts_rows:
            return []

        # BM25 rank is negative: more negative = better. Normalize to [0, 1].
        ranks = {r["rowid"]: r["rank"] for r in fts_rows}
        rank_values = list(ranks.values())
        min_rank = min(rank_values)
        max_rank = max(rank_values)
        rank_range = max_rank - min_rank or 1.0

        rowids = list(ranks.keys())
        placeholders = ",".join("?" * len(rowids))
        time_clause, time_params = self._build_time_filter(time_expr, after, before)

        sql = f"""
            SELECT * FROM memories m
            WHERE m.id IN ({placeholders})
              AND m.deleted_at IS NULL
        """
        params = list(rowids)
        if time_clause:
            sql += f" AND {time_clause}"
            params.extend(time_params)

        mem_rows = conn.execute(sql, params).fetchall()

        w_sim, w_imp, w_rec = scoring_weights or DEFAULT_SCORING_WEIGHTS
        memories = []
        for row in mem_rows:
            row_dict = dict(row)
            mem = Memory.from_row(row_dict)
            d = mem.to_dict()

            rid = row_dict["id"]
            rank = ranks[rid]
            similarity = round((max_rank - rank) / rank_range, 4)
            d["similarity"] = similarity

            conf = compute_confidence(
                mem.confidence, mem.memory_type,
                row_dict.get("last_recalled_at"), mem.created_at,
            )
            d["confidence"] = round(conf, 4)
            d["recall_count"] = row_dict.get("recall_count", 0) or 0
            d["last_recalled_at"] = row_dict.get("last_recalled_at")

            recency = compute_recency(mem.created_at)
            d["score"] = round(
                w_sim * similarity + w_imp * mem.importance + w_rec * recency, 4
            )
            memories.append(d)

        if tags:
            memories = self._filter_by_tags(memories, tags)

        memories.sort(key=lambda m: m["score"], reverse=True)
        return memories[:limit]

    # --- List ---

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> dict:
        """Paginated listing with optional filters."""
        conn = self._get_conn()
        offset = (page - 1) * page_size

        sql = "SELECT * FROM memories m WHERE m.deleted_at IS NULL"
        count_sql = "SELECT COUNT(*) as cnt FROM memories m WHERE m.deleted_at IS NULL"
        params: list = []
        count_params: list = []

        if memory_type:
            sql += " AND m.memory_type = ?"
            count_sql += " AND m.memory_type = ?"
            params.append(memory_type)
            count_params.append(memory_type)

        total = conn.execute(count_sql, count_params).fetchone()["cnt"]

        sql += " ORDER BY m.created_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size * 5 if tags else page_size, offset])

        rows = conn.execute(sql, params).fetchall()
        memories = [Memory.from_row(dict(r)).to_dict() for r in rows]

        if tags:
            memories = self._filter_by_tags(memories, tags)
            memories = memories[:page_size]

        return {
            "memories": memories,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def list_tags(self) -> dict:
        """Return all unique tags with their frequency counts."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT tags FROM memories WHERE deleted_at IS NULL"
        ).fetchall()
        counter: Counter = Counter()
        for r in rows:
            for t in _safe_tags(r["tags"]):
                counter[t] += 1
        sorted_tags = counter.most_common()
        return {
            "total_unique": len(sorted_tags),
            "tags": [{"tag": t, "count": c} for t, c in sorted_tags],
        }

    # --- Delete ---

    def delete(
        self,
        content_hash: str | None = None,
        tags: list[str] | None = None,
        before: str | None = None,
        after: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Delete memories by hash, tags, or time range."""
        start = time.time()

        if not any([content_hash, tags, before, after]):
            return {"error": "No filter specified — refusing to delete all memories"}

        # Use IMMEDIATE for the entire delete — read+write must be atomic
        # to avoid deleting rows another agent inserted between SELECT and UPDATE
        conn = self._begin_immediate()
        try:
            if content_hash:
                # Exact match first; fall back to prefix match for short hashes
                rows = conn.execute(
                    "SELECT id, content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                    (content_hash,),
                ).fetchall()
                if not rows and len(content_hash) < 64:
                    rows = conn.execute(
                        "SELECT id, content_hash FROM memories WHERE content_hash LIKE ? AND deleted_at IS NULL",
                        (content_hash + "%",),
                    ).fetchall()
                    if len(rows) > 1:
                        conn.execute("ROLLBACK")
                        return {"error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries"}
            else:
                sql = "SELECT id, content_hash, tags, created_at FROM memories WHERE deleted_at IS NULL"
                params: list = []
                time_clause, time_params = self._build_time_filter(None, after, before)
                if time_clause:
                    sql += f" AND {time_clause}"
                    params.extend(time_params)

                rows = conn.execute(sql, params).fetchall()

                if tags:
                    tag_set = {t.lower() for t in tags}
                    filtered = []
                    for r in rows:
                        row_tags = _safe_tags(r["tags"])
                        if any(t.lower() in tag_set for t in row_tags):
                            filtered.append(r)
                    rows = filtered

            if dry_run:
                conn.execute("ROLLBACK")
                return {
                    "dry_run": True,
                    "would_delete": len(rows),
                    "hashes": [r["content_hash"] for r in rows],
                }

            deleted_hashes = []
            for r in rows:
                mem_id = r["id"]
                conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (time.time(), mem_id),
                )
                # Remove embedding and FTS
                conn.execute(
                    "DELETE FROM memory_embeddings WHERE rowid = ?", (mem_id,)
                )
                conn.execute(
                    "DELETE FROM memory_fts WHERE rowid = ?", (mem_id,)
                )
                deleted_hashes.append(r["content_hash"])

            duration_ms = (time.time() - start) * 1000
            self._track_event(conn, "delete",
                duration_ms=duration_ms,
                result_count=len(deleted_hashes),
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"deleted": len(deleted_hashes), "deleted_hashes": deleted_hashes}

    # --- Get ---

    def get(self, content_hash: str) -> dict:
        """Retrieve a single memory by exact content hash or unique prefix."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
            (content_hash,),
        ).fetchone()
        if not row and len(content_hash) < 64:
            rows = conn.execute(
                "SELECT * FROM memories WHERE content_hash LIKE ? AND deleted_at IS NULL",
                (content_hash + "%",),
            ).fetchall()
            if len(rows) == 1:
                row = rows[0]
            elif len(rows) > 1:
                return {"error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries"}
        if not row:
            return {"error": f"Memory not found: {content_hash}"}
        row_dict = dict(row)
        d = Memory.from_row(row_dict).to_dict()
        d["recall_count"] = row_dict.get("recall_count", 0) or 0
        d["last_recalled_at"] = row_dict.get("last_recalled_at")
        return d

    # --- Update ---

    def update(
        self,
        content_hash: str,
        updates: dict,
        preserve_timestamps: bool = True,
    ) -> dict:
        """Update memory metadata without recreating."""
        conn = self._begin_immediate()
        try:
            row = conn.execute(
                "SELECT * FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (content_hash,),
            ).fetchone()
            if not row and len(content_hash) < 64:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE content_hash LIKE ? AND deleted_at IS NULL",
                    (content_hash + "%",),
                ).fetchall()
                if len(rows) == 1:
                    row = rows[0]
                elif len(rows) > 1:
                    conn.execute("ROLLBACK")
                    return {"error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries"}
            if not row:
                conn.execute("ROLLBACK")
                return {"error": f"Memory not found: {content_hash}"}

            sets = []
            params: list = []
            new_hash = None

            if "content" in updates:
                new_content = updates["content"]
                new_hash = hashlib.sha256(new_content.encode()).hexdigest()
                # Recompute embedding
                from .embeddings import get_model
                new_embedding = get_model().embed(new_content)
                sets.append("content = ?")
                params.append(new_content)
                sets.append("content_hash = ?")
                params.append(new_hash)
                # Update embedding and FTS
                mem_id = row["id"]
                conn.execute(
                    "UPDATE memory_embeddings SET content_embedding = ? WHERE rowid = ?",
                    (_serialize_f32(new_embedding), mem_id),
                )
                conn.execute(
                    "UPDATE memory_fts SET content = ? WHERE rowid = ?",
                    (new_content, mem_id),
                )

            if "tags" in updates:
                tags = updates["tags"]
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                sets.append("tags = ?")
                params.append(json.dumps(tags))

            if "memory_type" in updates:
                sets.append("memory_type = ?")
                params.append(updates["memory_type"])

            if "metadata" in updates:
                existing = json.loads(row["metadata"] or "{}")
                existing.update(updates["metadata"])
                sets.append("metadata = ?")
                params.append(json.dumps(existing))

            if "importance" in updates:
                sets.append("importance = ?")
                params.append(float(updates["importance"]))

            if "confidence" in updates:
                sets.append("confidence = ?")
                params.append(float(updates["confidence"]))

            if not preserve_timestamps or sets:
                now = time.time()
                now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
                sets.append("updated_at = ?")
                params.append(now)
                sets.append("updated_at_iso = ?")
                params.append(now_iso)

            if not sets:
                conn.execute("ROLLBACK")
                return {"error": "No valid updates provided"}

            params.append(row["id"])
            conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"status": "updated", "content_hash": new_hash or content_hash}

    # --- Stats ---

    def stats(
        self,
        after: str | None = None,
        before: str | None = None,
        top_recalled: int | None = None,
        never_recalled: bool = False,
        stale: bool = False,
    ) -> dict:
        """Return aggregated usage statistics."""
        conn = self._get_conn()
        conn.execute("BEGIN DEFERRED")
        try:
            return self._stats_inner(conn, after, before, top_recalled, never_recalled, stale)
        finally:
            try:
                conn.execute("ROLLBACK")  # read-only, nothing to commit
            except Exception:
                pass

    def _stats_inner(
        self,
        conn: sqlite3.Connection,
        after: str | None,
        before: str | None,
        top_recalled: int | None,
        never_recalled: bool,
        stale: bool,
    ) -> dict:
        # --- Time filter for events ---
        ev_clauses: list[str] = []
        ev_params: list = []
        if after:
            dt = datetime.fromisoformat(after).replace(tzinfo=timezone.utc)
            ev_clauses.append("timestamp >= ?")
            ev_params.append(dt.timestamp())
        if before:
            dt = datetime.fromisoformat(before).replace(tzinfo=timezone.utc)
            ev_clauses.append("timestamp <= ?")
            ev_params.append(dt.timestamp())
        ev_where = (" AND " + " AND ".join(ev_clauses)) if ev_clauses else ""

        # --- Memory counts ---
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NULL"
        ).fetchone()["cnt"]
        recalled = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NULL AND recall_count > 0"
        ).fetchone()["cnt"]
        never = total - recalled

        result: dict = {
            "total_memories": total,
            "recalled_at_least_once": recalled,
            "never_recalled": never,
        }

        # --- Event aggregates ---
        def _ev_count(op: str, extra: str = "") -> int:
            sql = f"SELECT COUNT(*) as cnt FROM operation_events WHERE operation = ?{ev_where}"
            if extra:
                sql += f" AND {extra}"
            return conn.execute(sql, [op] + ev_params).fetchone()["cnt"]

        def _ev_avg(op: str, col: str) -> float | None:
            sql = f"SELECT AVG({col}) as val FROM operation_events WHERE operation = ? AND {col} IS NOT NULL{ev_where}"
            row = conn.execute(sql, [op] + ev_params).fetchone()
            v = row["val"]
            return round(v, 2) if v is not None else None

        stores_total = _ev_count("store")
        duplicates_total = _ev_count("store", "duplicate_detected = 1")
        dedup_attempts = _ev_count("store", "dedup_used = 1")
        searches_total = _ev_count("search")
        searches_no_results = _ev_count("search", "result_count = 0")
        deletes_total = _ev_count("delete")

        # Days since first event
        first_ts = conn.execute(
            f"SELECT MIN(timestamp) as ts FROM operation_events WHERE 1=1{ev_where}",
            ev_params,
        ).fetchone()["ts"]
        days_active = max(1.0, (time.time() - first_ts) / 86400) if first_ts else 1.0

        result.update({
            "stores_total": stores_total,
            "stores_per_day": round(stores_total / days_active, 1),
            "duplicates_total": duplicates_total,
            "dedup_rate": round(duplicates_total / dedup_attempts, 3) if dedup_attempts else None,
            "searches_total": searches_total,
            "searches_no_results": searches_no_results,
            "search_hit_rate": round((searches_total - searches_no_results) / searches_total, 3) if searches_total else None,
            "avg_top_similarity": _ev_avg("search", "top_similarity"),
            "avg_results_per_search": _ev_avg("search", "result_count"),
            "total_chars_returned": conn.execute(
                f"SELECT COALESCE(SUM(chars_returned), 0) as val FROM operation_events WHERE operation = 'search'{ev_where}",
                ev_params,
            ).fetchone()["val"],
            "deletes_total": deletes_total,
            "avg_store_ms": _ev_avg("store", "duration_ms"),
            "avg_search_ms": _ev_avg("search", "duration_ms"),
        })

        # --- Top recalled memories ---
        if top_recalled:
            rows = conn.execute(
                "SELECT content_hash, content, tags, memory_type, recall_count, last_recalled_at "
                "FROM memories WHERE deleted_at IS NULL AND recall_count > 0 "
                "ORDER BY recall_count DESC LIMIT ?",
                (top_recalled,),
            ).fetchall()
            result["top_recalled"] = [
                {
                    "content_hash": r["content_hash"],
                    "memory_type": r["memory_type"],
                    "recall_count": r["recall_count"],
                    "content_preview": r["content"][:120],
                    "tags": _safe_tags(r["tags"]),
                }
                for r in rows
            ]

        # --- Never recalled memories ---
        if never_recalled:
            rows = conn.execute(
                "SELECT content_hash, content, tags, memory_type, created_at "
                "FROM memories WHERE deleted_at IS NULL AND recall_count = 0 "
                "ORDER BY created_at DESC LIMIT 20",
            ).fetchall()
            result["never_recalled_list"] = [
                {
                    "content_hash": r["content_hash"],
                    "memory_type": r["memory_type"],
                    "content_preview": r["content"][:120],
                    "created_at_iso": datetime.fromtimestamp(r["created_at"], tz=timezone.utc).isoformat() if r["created_at"] else None,
                    "tags": _safe_tags(r["tags"]),
                }
                for r in rows
            ]

        # --- Stale memories (old, never recalled) ---
        if stale:
            rows = conn.execute(
                "SELECT content_hash, content, tags, memory_type, created_at "
                "FROM memories WHERE deleted_at IS NULL AND recall_count = 0 "
                "ORDER BY created_at ASC LIMIT 10",
            ).fetchall()
            result["stale_memories"] = [
                {
                    "content_hash": r["content_hash"],
                    "memory_type": r["memory_type"],
                    "content_preview": r["content"][:120],
                    "created_at_iso": datetime.fromtimestamp(r["created_at"], tz=timezone.utc).isoformat() if r["created_at"] else None,
                    "tags": _safe_tags(r["tags"]),
                }
                for r in rows
            ]

        # --- Top tags across recalled memories ---
        tag_rows = conn.execute(
            "SELECT tags FROM memories WHERE deleted_at IS NULL AND recall_count > 0"
        ).fetchall()
        tags_flat = (t for r in tag_rows for t in _safe_tags(r["tags"]))
        result["top_tags"] = Counter(tags_flat).most_common(15)

        return result

    # --- Health ---

    def health(self) -> dict:
        """Database health and stats."""
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NULL"
        ).fetchone()["cnt"]
        deleted = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NOT NULL"
        ).fetchone()["cnt"]
        embeddings = conn.execute(
            "SELECT COUNT(*) as cnt FROM memory_embeddings_rowids"
        ).fetchone()["cnt"]

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        types = conn.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories "
            "WHERE deleted_at IS NULL GROUP BY memory_type"
        ).fetchall()

        return {
            "status": "healthy",
            "database": str(self.db_path),
            "database_size_bytes": db_size,
            "total_memories": total,
            "deleted_memories": deleted,
            "total_embeddings": embeddings,
            "memory_types": {r["memory_type"]: r["cnt"] for r in types},
        }

    # --- Cleanup ---

    def cleanup(self) -> dict:
        """Remove duplicate entries."""
        conn = self._begin_immediate()
        try:
            # Find content_hash duplicates (keeping lowest id)
            dupes = conn.execute(
                """
                SELECT id, content_hash FROM memories
                WHERE deleted_at IS NULL
                  AND content_hash IN (
                    SELECT content_hash FROM memories
                    WHERE deleted_at IS NULL
                    GROUP BY content_hash HAVING COUNT(*) > 1
                  )
                ORDER BY content_hash, id
                """
            ).fetchall()

            if not dupes:
                conn.execute("ROLLBACK")
                return {"duplicates_removed": 0}

            # Group by hash, keep first
            seen: dict[str, int] = {}
            to_delete: list[int] = []
            for r in dupes:
                h = r["content_hash"]
                if h in seen:
                    to_delete.append(r["id"])
                else:
                    seen[h] = r["id"]

            for mid in to_delete:
                conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (time.time(), mid),
                )
                conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (mid,))
                conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (mid,))

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"duplicates_removed": len(to_delete)}

    def purge(self, retention_days: int = 30, dry_run: bool = False) -> dict:
        """Hard-delete soft-deleted rows older than retention_days.

        Removes rows from memories, memory_embeddings, and memory_fts
        where deleted_at is set and older than the retention period.
        """
        cutoff = time.time() - (retention_days * 86400)
        conn = self._begin_immediate()
        try:
            rows = conn.execute(
                "SELECT id, content_hash FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                (cutoff,),
            ).fetchall()

            if dry_run:
                conn.execute("ROLLBACK")
                return {
                    "dry_run": True,
                    "would_purge": len(rows),
                    "retention_days": retention_days,
                    "hashes": [r["content_hash"] for r in rows],
                }

            for r in rows:
                mid = r["id"]
                conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (mid,))
                conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (mid,))
                conn.execute("DELETE FROM memories WHERE id = ?", (mid,))

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {
            "purged": len(rows),
            "retention_days": retention_days,
            "purged_hashes": [r["content_hash"] for r in rows],
        }

    def consolidate(
        self,
        threshold: float = 0.92,
        dry_run: bool = False,
        exclude_types: list[str] | None = None,
    ) -> dict:
        """Merge near-duplicate memories deterministically.

        Finds memory pairs with cosine similarity > threshold.
        Keeps the one with higher recall_count (ties: older memory wins),
        soft-deletes the other, and unions tags.

        exclude_types: memory types to skip (default: ["reference"]).
        Pass empty list to include all types.
        """
        if exclude_types is None:
            exclude_types = ["reference"]
        start = time.time()
        conn = self._get_conn()

        # Gather all active memory embeddings
        sql = (
            "SELECT m.id, m.content_hash, m.recall_count, m.created_at, m.tags, "
            "m.importance, m.confidence, m.memory_type "
            "FROM memories m WHERE m.deleted_at IS NULL"
        )
        params: list = []
        if exclude_types:
            placeholders = ",".join("?" * len(exclude_types))
            sql += f" AND m.memory_type NOT IN ({placeholders})"
            params.extend(exclude_types)
        rows = conn.execute(sql, params).fetchall()

        if len(rows) < 2:
            return {"consolidated": 0, "pairs": []}

        # Load all embeddings
        id_list = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(id_list))
        emb_rows = conn.execute(
            f"SELECT rowid, content_embedding FROM memory_embeddings WHERE rowid IN ({placeholders})",
            id_list,
        ).fetchall()
        emb_by_id = {r["rowid"]: r["content_embedding"] for r in emb_rows}

        import numpy as np

        # Parse embeddings into numpy for pairwise comparison
        id_to_idx = {}
        vectors = []
        valid_rows = []
        for r in rows:
            emb_bytes = emb_by_id.get(r["id"])
            if emb_bytes is None:
                continue
            vec = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            id_to_idx[r["id"]] = len(vectors)
            vectors.append(vec)
            valid_rows.append(dict(r))

        if len(vectors) < 2:
            return {"consolidated": 0, "pairs": []}

        mat = np.stack(vectors)
        # Cosine similarity matrix (vectors are already L2-normalized from embed())
        sim_matrix = mat @ mat.T

        # Find pairs above threshold (upper triangle only)
        pairs = []
        merged_ids: set[int] = set()
        n = len(valid_rows)
        for i in range(n):
            if valid_rows[i]["id"] in merged_ids:
                continue
            for j in range(i + 1, n):
                if valid_rows[j]["id"] in merged_ids:
                    continue
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    ri, rj = valid_rows[i], valid_rows[j]
                    # Keep the one with higher recall_count; tie-break by older created_at
                    rc_i = ri.get("recall_count", 0) or 0
                    rc_j = rj.get("recall_count", 0) or 0
                    if rc_i > rc_j or (rc_i == rc_j and ri["created_at"] <= rj["created_at"]):
                        keep, remove = ri, rj
                    else:
                        keep, remove = rj, ri
                    pairs.append({
                        "keep_hash": keep["content_hash"],
                        "remove_hash": remove["content_hash"],
                        "similarity": round(sim, 4),
                    })
                    merged_ids.add(remove["id"])

        if dry_run:
            return {
                "dry_run": True,
                "would_consolidate": len(pairs),
                "pairs": pairs,
            }

        if not pairs:
            return {"consolidated": 0, "pairs": []}

        # Execute merges
        conn.execute("BEGIN IMMEDIATE")
        try:
            now = time.time()
            for p in pairs:
                keep_h = p["keep_hash"]
                remove_h = p["remove_hash"]
                # Union tags
                keep_tags = _safe_tags(
                    conn.execute(
                        "SELECT tags FROM memories WHERE content_hash = ?", (keep_h,)
                    ).fetchone()["tags"]
                )
                remove_tags = _safe_tags(
                    conn.execute(
                        "SELECT tags FROM memories WHERE content_hash = ?", (remove_h,)
                    ).fetchone()["tags"]
                )
                merged_tags = list(dict.fromkeys(keep_tags + remove_tags))  # preserve order, dedup
                conn.execute(
                    "UPDATE memories SET tags = ? WHERE content_hash = ?",
                    (json.dumps(merged_tags), keep_h),
                )
                # Soft-delete the removed memory
                remove_row = conn.execute(
                    "SELECT id FROM memories WHERE content_hash = ?", (remove_h,)
                ).fetchone()
                conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (now, remove_row["id"]),
                )
                conn.execute(
                    "DELETE FROM memory_embeddings WHERE rowid = ?", (remove_row["id"],)
                )
                conn.execute(
                    "DELETE FROM memory_fts WHERE rowid = ?", (remove_row["id"],)
                )

            duration_ms = (time.time() - start) * 1000
            self._track_event(conn, "consolidate",
                duration_ms=duration_ms,
                result_count=len(pairs),
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {"consolidated": len(pairs), "pairs": pairs}

    def briefing(self, budget: int = 150) -> dict:
        """Generate a compact markdown briefing of top memories.

        Ranks memories by confidence * importance * recency, groups by type,
        and allocates a line budget per section.
        """
        start = time.time()
        conn = self._get_conn()

        rows = conn.execute(
            "SELECT content_hash, content, memory_type, confidence, importance, "
            "recall_count, last_recalled_at, created_at "
            "FROM memories WHERE deleted_at IS NULL"
        ).fetchall()

        total_memories = len(rows)
        if not total_memories:
            result = {
                "sections": {},
                "total_memories": 0,
                "total_lines": 0,
                "markdown": "No memories stored.",
            }
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._track_event(conn, "briefing",
                    duration_ms=(time.time() - start) * 1000,
                    result_count=0,
                )
                conn.execute("COMMIT")
            except BaseException:
                self._rollback_safe(conn)
                raise
            return result

        now = time.time()
        seven_days_ago = now - 7 * 86400

        # Score each memory
        scored = []
        for r in rows:
            conf = compute_confidence(
                r["confidence"] or 1.0, r["memory_type"],
                r["last_recalled_at"], r["created_at"],
            )
            imp = r["importance"] or 0.5
            days = max(0.0, (now - r["created_at"]) / 86400)
            recency = 1.0 / (1.0 + days)
            score = conf * imp * recency
            scored.append({
                "content": r["content"],
                "memory_type": r["memory_type"],
                "score": score,
                "created_at": r["created_at"],
            })

        # Section budgets
        section_budgets = {
            "decision": 25,
            "pattern": 25,
            "error": 15,
            "learning": 25,
            "reference": 15,
            "recent": 25,
            "other": 20,
        }

        # Scale budgets to fit total budget
        total_budget_raw = sum(section_budgets.values())
        scale = budget / total_budget_raw
        for k in section_budgets:
            section_budgets[k] = max(1, int(section_budgets[k] * scale))

        # Group memories into sections
        _TYPE_PREFIX_RE = re.compile(
            r"^\[(Pattern|Observation|Decision|Learning|Error|Note|Reference)\]\s*"
        )
        known_sections = {"decision", "pattern", "error", "learning", "reference"}
        groups: dict[str, list] = {k: [] for k in section_budgets}

        for m in scored:
            mt = m["memory_type"]
            if mt in known_sections:
                groups[mt].append(m)
            else:
                groups["other"].append(m)
            # Also add to recent if within 7 days
            if m["created_at"] >= seven_days_ago:
                groups["recent"].append(m)

        # Sort each group by score descending, take top-N per budget
        sections: dict[str, list[str]] = {}
        total_lines = 0
        for section, mems in groups.items():
            if not mems:
                continue
            mems.sort(key=lambda x: x["score"], reverse=True)
            line_budget = section_budgets.get(section, 10)
            lines = []
            for m in mems:
                if len(lines) >= line_budget:
                    break
                first_line = m["content"].split("\n")[0]
                first_line = _TYPE_PREFIX_RE.sub("", first_line)
                if len(first_line) > 120:
                    first_line = first_line[:117] + "..."
                lines.append(first_line)
            if lines:
                sections[section] = lines
                total_lines += len(lines)

        # Build markdown
        section_titles = {
            "decision": "Decisions",
            "pattern": "Patterns",
            "error": "Errors & Fixes",
            "learning": "Learnings",
            "reference": "References",
            "recent": "Recent (last 7 days)",
            "other": "Notes & Observations",
        }
        md_parts = [f"# Session Briefing ({total_memories} memories)\n"]
        for section in ("decision", "pattern", "error", "learning", "reference", "recent", "other"):
            lines = sections.get(section)
            if not lines:
                continue
            title = section_titles.get(section, section.title())
            md_parts.append(f"## {title}")
            for line in lines:
                md_parts.append(f"- {line}")
            md_parts.append("")

        markdown = "\n".join(md_parts).rstrip()

        result = {
            "sections": sections,
            "total_memories": total_memories,
            "total_lines": total_lines,
            "markdown": markdown,
        }

        duration_ms = (time.time() - start) * 1000
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._track_event(conn, "briefing",
                duration_ms=duration_ms,
                result_count=total_lines,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return result

    def apply_decay(self, min_confidence: float = 0.0) -> dict:
        """Recompute and persist decayed confidence for all memories.

        If min_confidence > 0, soft-deletes memories that fall below it.
        Returns count of updated and pruned memories.
        """
        conn = self._begin_immediate()
        try:
            rows = conn.execute(
                "SELECT id, content_hash, memory_type, confidence, "
                "last_recalled_at, created_at "
                "FROM memories WHERE deleted_at IS NULL"
            ).fetchall()

            updated = 0
            pruned = 0
            now = time.time()
            for r in rows:
                new_conf = compute_confidence(
                    r["confidence"] or 1.0, r["memory_type"],
                    r["last_recalled_at"], r["created_at"],
                )
                if min_confidence > 0 and new_conf < min_confidence:
                    conn.execute(
                        "UPDATE memories SET deleted_at = ? WHERE id = ?",
                        (now, r["id"]),
                    )
                    conn.execute(
                        "DELETE FROM memory_embeddings WHERE rowid = ?", (r["id"],)
                    )
                    conn.execute(
                        "DELETE FROM memory_fts WHERE rowid = ?", (r["id"],)
                    )
                    pruned += 1
                elif abs(new_conf - (r["confidence"] or 1.0)) > 0.0001:
                    conn.execute(
                        "UPDATE memories SET confidence = ? WHERE id = ?",
                        (new_conf, r["id"]),
                    )
                    updated += 1

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"updated": updated, "pruned": pruned}

    # ------------------------------------------------------------------ #
    #  Document operations                                                #
    # ------------------------------------------------------------------ #

    def store_doc(
        self,
        title: str,
        body: str,
        summary: str,
        doc_type: str = "document",
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Store a document. Returns dict with content_hash and status."""
        start = time.time()
        doc = Document(
            title=title,
            body=body,
            summary=summary,
            doc_type=doc_type,
            tags=tags or [],
            metadata=metadata or {},
        )

        conn = self._begin_immediate()
        try:
            # Check exact duplicate by content_hash (same body text)
            existing = conn.execute(
                "SELECT id FROM documents WHERE content_hash = ? AND deleted_at IS NULL",
                (doc.content_hash,),
            ).fetchone()
            if existing:
                duration_ms = (time.time() - start) * 1000
                self._track_event(conn, "doc_store",
                    duration_ms=duration_ms,
                    content_hash=doc.content_hash,
                    duplicate_detected=True,
                )
                conn.execute("COMMIT")
                return {
                    "content_hash": doc.content_hash,
                    "status": "duplicate",
                    "message": "Document with this body already exists",
                }

            # Compute embedding of summary
            from .embeddings import get_model
            embedding = get_model().embed(summary)

            # Insert document
            conn.execute(
                """
                INSERT INTO documents
                    (content_hash, title, body, summary, doc_type, tags, metadata,
                     created_at, updated_at, created_at_iso, updated_at_iso,
                     version, recall_count, last_recalled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                doc.to_row(),
            )
            doc_id = conn.execute(
                "SELECT id FROM documents WHERE content_hash = ?", (doc.content_hash,)
            ).fetchone()["id"]

            # Store embedding
            conn.execute(
                "INSERT INTO document_embeddings (rowid, summary_embedding) VALUES (?, ?)",
                (doc_id, _serialize_f32(embedding)),
            )

            # Insert into FTS index
            conn.execute(
                "INSERT INTO document_fts(rowid, title, body) VALUES (?, ?, ?)",
                (doc_id, title, body),
            )

            duration_ms = (time.time() - start) * 1000
            self._track_event(conn, "doc_store",
                duration_ms=duration_ms,
                content_hash=doc.content_hash,
                duplicate_detected=False,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {
            "content_hash": doc.content_hash,
            "status": "stored",
            "message": "Document stored successfully",
        }

    def get_doc(self, content_hash: str) -> dict:
        """Retrieve a single document by exact content hash or unique prefix."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash = ? AND deleted_at IS NULL",
            (content_hash,),
        ).fetchone()
        if not row and len(content_hash) < 64:
            rows = conn.execute(
                "SELECT * FROM documents WHERE content_hash LIKE ? AND deleted_at IS NULL",
                (content_hash + "%",),
            ).fetchall()
            if len(rows) == 1:
                row = rows[0]
            elif len(rows) > 1:
                return {"error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries"}
        if not row:
            return {"error": f"Document not found: {content_hash}"}
        row_dict = dict(row)
        d = Document.from_row(row_dict).to_dict()
        d["recall_count"] = row_dict.get("recall_count", 0) or 0
        d["last_recalled_at"] = row_dict.get("last_recalled_at")
        return d

    def list_docs(
        self,
        page: int = 1,
        page_size: int = 20,
        tags: list[str] | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """Paginated listing of documents with optional filters."""
        conn = self._get_conn()
        offset = (page - 1) * page_size

        sql = "SELECT * FROM documents m WHERE m.deleted_at IS NULL"
        count_sql = "SELECT COUNT(*) as cnt FROM documents m WHERE m.deleted_at IS NULL"
        params: list = []
        count_params: list = []

        if doc_type:
            sql += " AND m.doc_type = ?"
            count_sql += " AND m.doc_type = ?"
            params.append(doc_type)
            count_params.append(doc_type)

        total = conn.execute(count_sql, count_params).fetchone()["cnt"]

        sql += " ORDER BY m.created_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size * 5 if tags else page_size, offset])

        rows = conn.execute(sql, params).fetchall()
        documents = [Document.from_row(dict(r)).to_dict() for r in rows]

        if tags:
            documents = self._filter_by_tags(documents, tags)
            documents = documents[:page_size]

        return {
            "documents": documents,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def search_docs(
        self,
        query: str | None = None,
        mode: str = "auto",
        limit: int = 5,
        tags: list[str] | None = None,
        doc_type: str | None = None,
    ) -> list[dict]:
        """Search documents. Modes: semantic, fts, auto (both merged)."""
        if not query:
            return []
        conn = self._get_conn()

        results_by_hash: dict[str, dict] = {}

        if mode in ("semantic", "auto"):
            sem_results = self._search_docs_semantic(
                conn, query, limit=limit * 2 if mode == "auto" else limit,
            )
            for r in sem_results:
                results_by_hash[r["content_hash"]] = r

        if mode in ("fts", "auto"):
            fts_results = self._search_docs_fts(
                conn, query, limit=limit * 2 if mode == "auto" else limit,
            )
            for r in fts_results:
                h = r["content_hash"]
                if h in results_by_hash:
                    # Merge: take max similarity, add scores
                    existing = results_by_hash[h]
                    existing["similarity"] = max(
                        existing.get("similarity", 0), r.get("similarity", 0),
                    )
                    existing["score"] = existing.get("score", 0) + r.get("score", 0)
                else:
                    results_by_hash[h] = r

        results = list(results_by_hash.values())

        # Filter by tags
        if tags:
            results = self._filter_by_tags(results, tags)
        # Filter by doc_type
        if doc_type:
            results = [r for r in results if r.get("doc_type") == doc_type]

        results.sort(key=lambda r: r.get("score", 0), reverse=True)

        # Update recall counts
        conn.execute("BEGIN IMMEDIATE")
        try:
            now = time.time()
            for r in results[:limit]:
                conn.execute(
                    "UPDATE documents SET recall_count = recall_count + 1, "
                    "last_recalled_at = ? WHERE content_hash = ? AND deleted_at IS NULL",
                    (now, r["content_hash"]),
                )
            self._track_event(conn, "doc_search",
                query=query,
                search_mode=mode,
                result_count=len(results[:limit]),
                top_similarity=results[0].get("similarity") if results else None,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return results[:limit]

    def _search_docs_semantic(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Cosine similarity search on document summary embeddings."""
        from .embeddings import get_model
        embedding = get_model().embed(query)

        fetch_limit = max(limit * 3, 30)
        rows = conn.execute(
            """
            SELECT e.rowid, e.distance
            FROM document_embeddings e
            WHERE e.summary_embedding MATCH ?
            ORDER BY e.distance
            LIMIT ?
            """,
            (_serialize_f32(embedding), fetch_limit),
        ).fetchall()

        if not rows:
            return []

        rowids = [r["rowid"] for r in rows]
        distances = {r["rowid"]: r["distance"] for r in rows}

        placeholders = ",".join("?" * len(rowids))
        doc_rows = conn.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders}) AND deleted_at IS NULL",
            rowids,
        ).fetchall()

        row_by_id = {row["id"]: dict(row) for row in doc_rows}
        results = []
        for rid in rowids:
            row = row_by_id.get(rid)
            if row is None:
                continue
            doc = Document.from_row(row)
            d = doc.to_dict()
            similarity = round(1.0 - distances.get(rid, 1.0), 4)
            d["similarity"] = similarity
            d["recall_count"] = row.get("recall_count", 0) or 0
            d["last_recalled_at"] = row.get("last_recalled_at")
            d["score"] = similarity
            results.append(d)

        return results

    def _search_docs_fts(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
    ) -> list[dict]:
        """BM25 full-text search on document title and body via FTS5."""
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []

        fetch_limit = max(limit * 3, 30)

        fts_rows = conn.execute(
            "SELECT rowid, rank FROM document_fts "
            "WHERE document_fts MATCH ? ORDER BY rank LIMIT ?",
            (safe_query, fetch_limit),
        ).fetchall()

        if not fts_rows:
            return []

        # BM25 rank is negative: more negative = better. Normalize to [0, 1].
        ranks = {r["rowid"]: r["rank"] for r in fts_rows}
        rank_values = list(ranks.values())
        min_rank = min(rank_values)
        max_rank = max(rank_values)
        rank_range = max_rank - min_rank or 1.0

        rowids = list(ranks.keys())
        placeholders = ",".join("?" * len(rowids))

        doc_rows = conn.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders}) AND deleted_at IS NULL",
            rowids,
        ).fetchall()

        results = []
        for row in doc_rows:
            row_dict = dict(row)
            doc = Document.from_row(row_dict)
            d = doc.to_dict()

            rid = row_dict["id"]
            rank = ranks[rid]
            similarity = round((max_rank - rank) / rank_range, 4)
            d["similarity"] = similarity
            d["recall_count"] = row_dict.get("recall_count", 0) or 0
            d["last_recalled_at"] = row_dict.get("last_recalled_at")
            d["score"] = similarity
            results.append(d)

        return results

    def update_doc(self, content_hash: str, **kwargs) -> dict:
        """Update a document. Accepted kwargs: title, body, summary, doc_type, tags, metadata."""
        conn = self._begin_immediate()
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? AND deleted_at IS NULL",
                (content_hash,),
            ).fetchone()
            if not row and len(content_hash) < 64:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE content_hash LIKE ? AND deleted_at IS NULL",
                    (content_hash + "%",),
                ).fetchall()
                if len(rows) == 1:
                    row = rows[0]
                elif len(rows) > 1:
                    conn.execute("ROLLBACK")
                    return {"error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries"}
            if not row:
                conn.execute("ROLLBACK")
                return {"error": f"Document not found: {content_hash}"}

            row_dict = dict(row)
            doc_id = row_dict["id"]
            sets: list[str] = []
            params: list = []
            new_hash = row_dict["content_hash"]
            new_version = row_dict.get("version", 1) or 1
            body_changed = False
            summary_changed = False
            title_changed = False

            if "title" in kwargs:
                sets.append("title = ?")
                params.append(kwargs["title"])
                title_changed = True

            if "body" in kwargs:
                new_body = kwargs["body"]
                new_hash = hashlib.sha256(new_body.encode()).hexdigest()
                new_version += 1
                sets.append("body = ?")
                params.append(new_body)
                sets.append("content_hash = ?")
                params.append(new_hash)
                sets.append("version = ?")
                params.append(new_version)
                body_changed = True

            if "summary" in kwargs:
                sets.append("summary = ?")
                params.append(kwargs["summary"])
                summary_changed = True

            if "doc_type" in kwargs:
                sets.append("doc_type = ?")
                params.append(kwargs["doc_type"])

            if "tags" in kwargs:
                tags = kwargs["tags"]
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                sets.append("tags = ?")
                params.append(json.dumps(tags))

            if "metadata" in kwargs:
                existing_meta = json.loads(row_dict["metadata"] or "{}")
                existing_meta.update(kwargs["metadata"])
                sets.append("metadata = ?")
                params.append(json.dumps(existing_meta))

            if not sets:
                conn.execute("ROLLBACK")
                return {"error": "No valid updates provided"}

            # Always update timestamps
            now = time.time()
            now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
            sets.append("updated_at = ?")
            params.append(now)
            sets.append("updated_at_iso = ?")
            params.append(now_iso)

            params.append(doc_id)
            conn.execute(
                f"UPDATE documents SET {', '.join(sets)} WHERE id = ?",
                params,
            )

            # Re-embed summary if it changed
            if summary_changed:
                from .embeddings import get_model
                new_embedding = get_model().embed(kwargs["summary"])
                conn.execute(
                    "DELETE FROM document_embeddings WHERE rowid = ?", (doc_id,)
                )
                conn.execute(
                    "INSERT INTO document_embeddings (rowid, summary_embedding) VALUES (?, ?)",
                    (doc_id, _serialize_f32(new_embedding)),
                )

            # Update FTS if title or body changed
            # External content FTS5 tables require the special delete command
            # with the original values, then re-insert with new values.
            if title_changed or body_changed:
                conn.execute(
                    "INSERT INTO document_fts(document_fts, rowid, title, body) "
                    "VALUES('delete', ?, ?, ?)",
                    (doc_id, row_dict["title"], row_dict["body"]),
                )
                new_title = kwargs.get("title", row_dict["title"])
                new_body_text = kwargs.get("body", row_dict["body"])
                conn.execute(
                    "INSERT INTO document_fts(rowid, title, body) VALUES (?, ?, ?)",
                    (doc_id, new_title, new_body_text),
                )

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {
            "status": "updated",
            "content_hash": new_hash,
            "version": new_version,
        }

    def delete_doc(self, content_hash: str, dry_run: bool = False) -> dict:
        """Soft-delete a document by content hash."""
        start = time.time()
        conn = self._begin_immediate()
        try:
            # Exact match first; fall back to prefix match for short hashes
            rows = conn.execute(
                "SELECT id, content_hash FROM documents WHERE content_hash = ? AND deleted_at IS NULL",
                (content_hash,),
            ).fetchall()
            if not rows and len(content_hash) < 64:
                rows = conn.execute(
                    "SELECT id, content_hash FROM documents WHERE content_hash LIKE ? AND deleted_at IS NULL",
                    (content_hash + "%",),
                ).fetchall()
                if len(rows) > 1:
                    conn.execute("ROLLBACK")
                    return {"error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries"}

            if not rows:
                conn.execute("ROLLBACK")
                return {"error": f"Document not found: {content_hash}"}

            if dry_run:
                conn.execute("ROLLBACK")
                return {
                    "dry_run": True,
                    "would_delete": len(rows),
                    "content_hash": rows[0]["content_hash"],
                }

            doc_id = rows[0]["id"]
            full_hash = rows[0]["content_hash"]

            # Soft-delete
            conn.execute(
                "UPDATE documents SET deleted_at = ? WHERE id = ?",
                (time.time(), doc_id),
            )
            # Hard-delete embedding
            conn.execute(
                "DELETE FROM document_embeddings WHERE rowid = ?", (doc_id,)
            )

            duration_ms = (time.time() - start) * 1000
            self._track_event(conn, "doc_delete",
                duration_ms=duration_ms,
                content_hash=full_hash,
                result_count=1,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"deleted": 1, "content_hash": full_hash}
