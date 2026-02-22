"""Core memory store — all database operations."""

from __future__ import annotations

import json
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

from .models import Memory

DB_PATH = Path.home() / ".claude" / "tools" / "memory" / "data" / "sqlite_vec.db"

EMBEDDING_DIM = 384

_F32_STRUCT = struct.Struct(f"<{EMBEDDING_DIM}f")


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


class MemoryStore:
    """Synchronous SQLite memory store with sqlite-vec embeddings."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            import sqlite_vec

            self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
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
        try:
            conn.execute("ALTER TABLE memories ADD COLUMN recall_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE memories ADD COLUMN last_recalled_at REAL DEFAULT NULL")
        except sqlite3.OperationalError:
            pass

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
        _embedding: object | None = None,
    ) -> dict:
        """Store a single memory. Returns dict with hash and status.

        If dedup_threshold is set (0.0-1.0), checks for semantically similar
        memories before storing. Embeds content once and reuses for both the
        similarity check and the stored embedding.
        """
        start = time.time()
        mem = Memory(
            content=content,
            tags=tags or [],
            memory_type=memory_type,
            metadata=metadata or {},
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

            # Only compute embedding after confirming not an exact duplicate
            if _embedding is not None:
                embedding = _embedding
            else:
                from .embeddings import get_model
                embedding = get_model().embed(content)

            # Similarity-based dedup
            if dedup_threshold is not None:
                similar = self._search_semantic(
                    conn, query=None, limit=1, tags=None,
                    time_expr=None, after=None, before=None,
                    _embedding=embedding,
                )
                if similar and similar[0].get("similarity", 0) >= dedup_threshold:
                    duration_ms = (time.time() - start) * 1000
                    self._track_event(conn, "store",
                        duration_ms=duration_ms,
                        content_hash=mem.content_hash,
                        dedup_used=True,
                        duplicate_detected=True,
                        duplicate_similarity=similar[0]["similarity"],
                    )
                    conn.execute("COMMIT")
                    return {
                        "content_hash": mem.content_hash,
                        "status": "duplicate",
                        "message": f"Similar memory exists (similarity={similar[0]['similarity']:.2f})",
                        "similar_hash": similar[0]["content_hash"],
                    }

            # Insert memory
            conn.execute(
                """
                INSERT INTO memories
                    (content_hash, content, tags, memory_type, metadata,
                     created_at, updated_at, created_at_iso, updated_at_iso)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            result = self.store(
                content=item["content"],
                tags=item.get("tags", []),
                memory_type=item.get("memory_type", "note"),
                metadata=item.get("metadata", {}),
                dedup_threshold=dedup_threshold,
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
    ) -> list[dict]:
        """Search memories. Modes: semantic, exact, hybrid."""
        start = time.time()
        conn = self._get_conn()

        # Read phase — no write lock needed yet
        if mode == "exact":
            results = self._search_exact(conn, query, limit, tags, time_expr, after, before)
        elif mode in ("semantic", "hybrid"):
            results = self._search_semantic(
                conn, query, limit, tags, time_expr, after, before
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
            # Update recall counts for returned memories
            now = time.time()
            for m in results:
                conn.execute(
                    "UPDATE memories SET recall_count = recall_count + 1, last_recalled_at = ? WHERE content_hash = ?",
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
    ) -> list[dict]:
        if _embedding is not None:
            embedding = _embedding
        elif query:
            from .embeddings import get_model
            embedding = get_model().embed(query)
        else:
            return []
        # Fetch more than needed to allow post-filtering
        fetch_limit = limit * 5 if (tags or time_expr or after or before) else limit

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
        row_by_id = {row["id"]: row for row in mem_rows}
        memories = []
        for rid in rowids:
            row = row_by_id.get(rid)
            if row is None:
                continue  # filtered out by time/deleted_at
            mem = Memory.from_row(dict(row))
            d = mem.to_dict()
            d["similarity"] = round(1.0 - distances.get(rid, 1.0), 4)
            memories.append(d)

        if tags:
            memories = self._filter_by_tags(memories, tags)

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
        memories = [Memory.from_row(dict(r)).to_dict() for r in rows]

        if tags:
            memories = self._filter_by_tags(memories, tags)

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
                # Remove embedding
                conn.execute(
                    "DELETE FROM memory_embeddings WHERE rowid = ?", (mem_id,)
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

            params.append(content_hash)
            conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE content_hash = ?",
                params,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"status": "updated", "content_hash": content_hash}

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

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"duplicates_removed": len(to_delete)}
