"""Core memory store — all database operations."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .models import Document, Memory

DB_PATH = Path.home() / "repos" / "memory" / "data" / "sqlite_vec.db"

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
    (0.9, re.compile(r"\b(IMPORTANT|CRITICAL|MUST|BREAKING)\b")),
    (0.8, re.compile(r"\b(NEVER|ALWAYS|WARNING|DANGER)\b")),
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
DEFAULT_SCORING_WEIGHTS = (0.8, 0.1, 0.1)  # similarity, importance, recency
MIN_SIMILARITY_THRESHOLD = 0.45  # filter out semantically irrelevant results

# --- Demotion weight ---
# Penalises over-recalled memories so they don't crowd out genuine matches.
# 0 disables entirely (demotion factor == 1.0 for all memories).
# At 0.1: recall_count=100 (~log1p≈4.6) loses ~32% of score.
# At 0.2: same memory loses ~48%.
# Override via MEMORY_DEMOTION_WEIGHT env var (read once at import time).
DEMOTION_WEIGHT: float = float(os.environ.get("MEMORY_DEMOTION_WEIGHT", "0.1"))

# --- ACT-R-style activation (Phase C) ---
# activation = w_sim*similarity + w_type*type_weight + w_temporal*decay
#            + w_session*distinct_session_score - w_stale*staleness_penalty
# Defaults from HAI 2025 ablation; override via MEMORY_ACTIVATION_WEIGHTS env
# var as comma-separated floats (5 values).
_DEFAULT_ACTIVATION_WEIGHTS = (0.55, 0.10, 0.15, 0.15, 0.05)


def _parse_activation_weights() -> tuple[float, float, float, float, float]:
    raw = os.environ.get("MEMORY_ACTIVATION_WEIGHTS")
    if not raw:
        return _DEFAULT_ACTIVATION_WEIGHTS
    try:
        parts = tuple(float(x.strip()) for x in raw.split(","))
    except ValueError:
        return _DEFAULT_ACTIVATION_WEIGHTS
    if len(parts) != 5:
        return _DEFAULT_ACTIVATION_WEIGHTS
    return parts  # type: ignore[return-value]


ACTIVATION_WEIGHTS: tuple[float, float, float, float, float] = _parse_activation_weights()

# Whether activation replaces the composite/RRF score on hybrid/semantic/fts modes
# when rerank is OFF.  Default off for backwards compat; rerank=True always
# uses activation as the rerank-blend base.
USE_ACTIVATION: bool = os.environ.get("MEMORY_USE_ACTIVATION", "0") == "1"

# Temporal decay characteristic time (days).  Smaller = faster forgetting.
ACTIVATION_TAU_DAYS: float = float(os.environ.get("MEMORY_ACTIVATION_TAU", "30.0"))

# Staleness kicks in once last_recall is older than this (days).
ACTIVATION_STALE_DAYS: float = float(os.environ.get("MEMORY_ACTIVATION_STALE", "60.0"))

# Active-forget threshold.  Memories below this activation become candidates
# for soft-delete in dream pass 6 (only when MEMORY_ACTIVE_FORGET=1).
FORGET_THRESHOLD: float = float(os.environ.get("MEMORY_FORGET_THRESHOLD", "0.05"))

# Type weight in activation: higher = more "permanent" intent.
_ACTIVATION_TYPE_WEIGHT: dict[str, float] = {
    "decision": 1.0,
    "pattern": 0.9,
    "reference": 0.9,
    "error": 0.75,
    "learning": 0.7,
    "observation": 0.4,
    "note": 0.4,
    "todo": 0.6,
}


def compute_activation(
    similarity: float,
    recall_count: int,
    distinct_session_count: int,
    memory_type: str,
    created_at: float,
    last_recalled_at: float | None,
    now: float | None = None,
) -> float:
    """ACT-R-inspired activation score for memory ranking.

    Components:
      similarity         — cosine distance from current query
      type_weight        — intent-permanence prior per memory_type
      temporal_decay     — exp(-Δt / τ) where Δt is days since last touch
      distinct_session   — log(distinct+1) / log(recall+1); penalises hot-cluster bias
      staleness_penalty  — applied only after ACTIVATION_STALE_DAYS

    Returns activation in roughly [0, 1].  Negative components clamped to 0.
    """
    if now is None:
        now = time.time()

    w_sim, w_type, w_temp, w_sess, w_stale = ACTIVATION_WEIGHTS
    sim = max(0.0, float(similarity))
    tw = _ACTIVATION_TYPE_WEIGHT.get(memory_type or "note", 0.4)

    anchor = last_recalled_at if last_recalled_at else created_at
    dt_days = max(0.0, (now - anchor) / 86400.0)
    temporal = math.exp(-dt_days / ACTIVATION_TAU_DAYS)

    rc = max(0, int(recall_count or 0))
    dsc = max(0, int(distinct_session_count or 0))
    # log(distinct+1) / log(recall+1) — 1.0 when every recall is from a fresh
    # session; collapses toward 0 when one session pumps the same memory.
    sess = 1.0 if rc == 0 else math.log1p(dsc) / math.log1p(rc)

    staleness_penalty = max(0.0, (dt_days - ACTIVATION_STALE_DAYS) / ACTIVATION_STALE_DAYS)

    activation = (
        w_sim * sim
        + w_type * tw
        + w_temp * temporal
        + w_sess * sess
        - w_stale * staleness_penalty
    )
    return max(0.0, min(1.0, activation))

# --- Injection-pattern screening ---
# Compiled once at import time.  Case-insensitive.
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|messages?)\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above)\b", re.IGNORECASE),
    re.compile(r"^\s*(system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\s*/?(system|assistant)\s*>", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(system|administrator|root)\b", re.IGNORECASE),
    re.compile(r"\byou\s+(must|will|shall)\s+(now\s+)?(always|never)\b.*\b(ignore|forget|override)\b", re.IGNORECASE),
    re.compile(r"\[\s*INST\s*\]|\[\s*/INST\s*\]", re.IGNORECASE),
]


def _screen_injection(content: str) -> tuple[bool, str | None]:
    """Scan *content* for prompt-injection patterns.

    Returns ``(suspicious, matched_pattern_string)`` where *matched_pattern_string*
    is the regex pattern that first matched, or ``None`` when content is clean.
    """
    for pat in INJECTION_PATTERNS:
        if pat.search(content):
            return True, pat.pattern
    return False, None


def _serialize_f32(vec: object) -> bytes:
    """Serialize embedding to little-endian bytes for sqlite-vec.

    All real call paths pass numpy float32 arrays; tobytes() is zero-copy.
    """
    return vec.tobytes()


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
    now = datetime.now(tz=UTC)

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


def _rrf_fuse(*result_lists: list[dict], limit: int, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over arbitrary ranked result lists.

    For each unique content_hash, score = Σ 1/(k + rank_i) across the lists it
    appears in (rank is 1-based).  Robust to score-scale heterogeneity between
    semantic cosine, BM25, and graph hop-weight.

    Args:
        *result_lists: each list must contain dicts with 'content_hash'.
        limit: max results to return.
        k: RRF dampening constant (60 is the literature standard).

    Returns:
        Merged list sorted by RRF score desc, truncated to `limit`.  Each result
        carries the merged 'rrf_score' key and 'score' (alias of rrf_score) so
        downstream code that sorts by 'score' keeps working.
    """
    fused: dict[str, dict] = {}
    rrf_scores: dict[str, float] = {}
    for lst in result_lists:
        for rank, r in enumerate(lst, start=1):
            h = r["content_hash"]
            rrf_scores[h] = rrf_scores.get(h, 0.0) + 1.0 / (k + rank)
            if h not in fused:
                fused[h] = dict(r)
            else:
                # Preserve highest similarity seen across lists
                fused[h]["similarity"] = max(
                    fused[h].get("similarity", 0) or 0,
                    r.get("similarity", 0) or 0,
                )

    for h, s in rrf_scores.items():
        fused[h]["rrf_score"] = round(s, 6)
        fused[h]["score"] = round(s, 6)

    merged = sorted(fused.values(), key=lambda m: m["score"], reverse=True)
    return merged[:limit]


_FTS5_OPERATORS = {"OR", "AND", "NOT"}


def _quote_fts_token(token: str) -> str:
    """Quote a single token for FTS5, preserving already-quoted tokens."""
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token
    return f'"{token.replace(chr(34), chr(34) * 2)}"'


def _sanitize_fts_query(query: str) -> str:
    """Escape user input for safe use in FTS5 MATCH expressions.

    Wraps each whitespace-delimited token in double quotes so FTS5
    treats hyphens, colons, asterisks, and special characters as literals.
    Recognizes FTS5 boolean operators (OR, AND, NOT) in valid positions
    and passes them through unquoted.
    """
    query = query.strip()
    if not query:
        return ""
    tokens = query.split()

    # Classify each token as operator or term based on position
    parts: list[tuple[str, str]] = []  # (text, "op" | "term")
    for i, token in enumerate(tokens):
        if token in _FTS5_OPERATORS:
            has_next = i + 1 < len(tokens) and tokens[i + 1] not in _FTS5_OPERATORS
            has_prev = parts and parts[-1][1] == "term"
            is_infix = token in ("OR", "AND") and has_prev and has_next
            is_prefix = token == "NOT" and has_next  # noqa: S105  # nosec B105
            if is_infix or is_prefix:
                parts.append((token, "op"))
            else:
                parts.append((_quote_fts_token(token), "term"))
        else:
            parts.append((_quote_fts_token(token), "term"))

    # Trailing operators are invalid — convert to literals
    while parts and parts[-1][1] == "op":
        parts[-1] = (_quote_fts_token(parts[-1][0]), "term")

    return " ".join(p[0] for p in parts)


# --- Entity extraction (regex-based, zero dependencies) ---

_TICKET_RE = re.compile(r"\b([A-Z]{2,10}-\d+)\b")
_PR_RE = re.compile(r"\bPR\s*#?(\d+)\b", re.IGNORECASE)

# Technology keywords (normalized lowercase → display name)
_TECH_KEYWORDS: dict[str, str] = {
    k: k
    for k in (
        "kubernetes",
        "k8s",
        "docker",
        "terraform",
        "ansible",
        "helm",
        "istio",
        "envoy",
        "nginx",
        "postgres",
        "postgresql",
        "mysql",
        "redis",
        "mongodb",
        "dynamodb",
        "elasticsearch",
        "kafka",
        "rabbitmq",
        "graphql",
        "grpc",
        "react",
        "nextjs",
        "vue",
        "angular",
        "fastapi",
        "django",
        "flask",
        "express",
        "node",
        "nodejs",
        "python",
        "golang",
        "rust",
        "java",
        "typescript",
        "javascript",
        "aws",
        "gcp",
        "azure",
        "s3",
        "ec2",
        "rds",
        "ecs",
        "eks",
        "lambda",
        "cloudfront",
        "route53",
        "iam",
        "vpc",
        "alb",
        "sqs",
        "sns",
        "ecr",
        "git",
        "github",
        "gitlab",
        "jenkins",
        "argocd",
        "prometheus",
        "grafana",
        "datadog",
        "linux",
        "ubuntu",
        "centos",
        "zmk",
        "qemu",
        "buildkit",
        "onnx",
        "sqlite",
        "oauth",
        "jwt",
        "openai",
        "anthropic",
        "llm",
    )
}
_TECH_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_TECH_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_SERVICE_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+)*-(?:service|manager|api|worker|controller|operator))\b",
    re.IGNORECASE,
)

# Tag prefixes that map to entity types
_TAG_ENTITY_PREFIXES: dict[str, str] = {
    "project:": "project",
    "svc:": "service",
    "cloud:": "cloud",
    "tool:": "tool",
}


def extract_entities(content: str, tags: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Extract entities from memory content and tags.

    Returns list of (name_normalized, display_name, entity_type), deduplicated.
    """
    seen: set[tuple[str, str]] = set()  # (name_normalized, entity_type)
    results: list[tuple[str, str, str]] = []

    def _add(name: str, display: str, etype: str) -> None:
        key = (name.lower(), etype)
        if key not in seen:
            seen.add(key)
            results.append((name.lower(), display, etype))

    # 1. Structured identifiers: tickets (TICKET-24, PROJ-42)
    for m in _TICKET_RE.finditer(content):
        ticket = m.group(1)
        _add(ticket, ticket, "ticket")

    # 2. PR references
    for m in _PR_RE.finditer(content):
        pr_num = m.group(1)
        _add(f"pr-{pr_num}", f"PR #{pr_num}", "pr")

    # 3. Technology keywords
    for m in _TECH_RE.finditer(content):
        tech = m.group(1).lower()
        _add(tech, _TECH_KEYWORDS.get(tech, tech), "technology")

    # 4. Service-like names
    for m in _SERVICE_RE.finditer(content):
        svc = m.group(1).lower()
        _add(svc, m.group(1), "service")

    # 5. Tag-derived entities
    for tag in tags or []:
        for prefix, etype in _TAG_ENTITY_PREFIXES.items():
            if tag.lower().startswith(prefix):
                value = tag[len(prefix) :]
                if value:
                    _add(value.lower(), value, etype)

    return results


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
    return base_confidence * (rate**days)


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
        # Bootstrap base schema for fresh DBs. Older installations created
        # these tables out-of-band; checking in the canonical CREATE keeps
        # `memory health` / `memory store` working on a fresh sqlite file.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                memory_type TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL,
                created_at_iso TEXT,
                updated_at_iso TEXT,
                deleted_at REAL DEFAULT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);
            CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Typed memory↔memory edges (supersedes, contradicts, refines,
            -- references, merged_into).  Currently created but written only
            -- by the dream/consolidate paths in a future patch.
            CREATE TABLE IF NOT EXISTS memory_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hash TEXT NOT NULL,
                target_hash TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                metadata TEXT,
                created_at REAL,
                UNIQUE(source_hash, target_hash, relationship_type)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_source ON memory_graph(source_hash);
            CREATE INDEX IF NOT EXISTS idx_graph_target ON memory_graph(target_hash);
            CREATE INDEX IF NOT EXISTS idx_graph_relationship ON memory_graph(relationship_type);
            """
        )
        # Vector index for memory embeddings (requires sqlite_vec loaded).
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings
                USING vec0(content_embedding FLOAT[768] distance_metric=cosine);
            """
        )
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
            # Phase C: session-aware recall (hot-cluster bias fix)
            "ALTER TABLE memories ADD COLUMN last_recall_session TEXT DEFAULT NULL",
            "ALTER TABLE memories ADD COLUMN distinct_session_count INTEGER DEFAULT 0",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(col_sql)

        # FTS5 virtual table for BM25 keyword search (zero cold start, no model)
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(content, content='memories', content_rowid='id',
                       tokenize='porter ascii');
            """
        )
        # Backfill existing rows — only if memories table already exists
        memories_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
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
                USING vec0(summary_embedding FLOAT[768] distance_metric=cosine);
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
        docs_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'").fetchone()
        if docs_exists:
            conn.execute(
                "INSERT OR IGNORE INTO document_fts(rowid, title, body) "
                "SELECT id, title, body FROM documents WHERE deleted_at IS NULL"
            )

        self._migrate_graph_tables()

    def _migrate_graph_tables(self) -> None:
        """One-time migration: create knowledge graph tables."""
        conn = self._conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                display_name  TEXT NOT NULL,
                entity_type   TEXT NOT NULL,
                metadata      TEXT DEFAULT '{}',
                created_at    REAL NOT NULL,
                UNIQUE(name, entity_type)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id     INTEGER NOT NULL,
                entity_id     INTEGER NOT NULL,
                created_at    REAL NOT NULL,
                PRIMARY KEY (memory_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);
            CREATE INDEX IF NOT EXISTS idx_me_memory ON memory_entities(memory_id);

            CREATE TABLE IF NOT EXISTS entity_relations (
                source_id      INTEGER NOT NULL,
                target_id      INTEGER NOT NULL,
                relation_type  TEXT NOT NULL,
                weight         REAL DEFAULT 1.0,
                created_at     REAL NOT NULL,
                updated_at     REAL NOT NULL,
                PRIMARY KEY (source_id, target_id, relation_type)
            );
            CREATE INDEX IF NOT EXISTS idx_er_source ON entity_relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_er_target ON entity_relations(target_id);
            """
        )
        # Bi-temporal columns on edges.  Idempotent — older DBs get them via ALTER.
        for col_sql in (
            "ALTER TABLE entity_relations ADD COLUMN valid_from REAL DEFAULT NULL",
            "ALTER TABLE entity_relations ADD COLUMN valid_to REAL DEFAULT NULL",
            "ALTER TABLE memory_graph ADD COLUMN weight REAL DEFAULT 1.0",
            "ALTER TABLE memory_graph ADD COLUMN valid_from REAL DEFAULT NULL",
            "ALTER TABLE memory_graph ADD COLUMN valid_to REAL DEFAULT NULL",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(col_sql)
        # Index for "currently valid" filters on entity edges
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_er_valid_to ON entity_relations(valid_to)")

    @staticmethod
    def _rollback_safe(conn: sqlite3.Connection) -> None:
        """Roll back the current transaction, ignoring errors if none is active."""
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")

    @staticmethod
    def _enrich_with_activation(conn: sqlite3.Connection, results: list[dict]) -> None:
        """Attach 'activation' key to each result via compute_activation().

        One bulk SQL fetch for memory_type / created_at / session counts; cheap
        enough to run unconditionally on every search.
        """
        if not results:
            return
        hashes = [r["content_hash"] for r in results if r.get("content_hash")]
        if not hashes:
            return
        placeholders = ",".join("?" * len(hashes))
        rows = conn.execute(
            "SELECT content_hash, memory_type, created_at, last_recalled_at, "
            "recall_count, distinct_session_count "
            f"FROM memories WHERE content_hash IN ({placeholders})",
            hashes,
        ).fetchall()
        by_hash = {r["content_hash"]: dict(r) for r in rows}
        for r in results:
            row = by_hash.get(r["content_hash"])
            if not row:
                continue
            # Use similarity if present, else fall back to whatever the mode produced.
            sim = r.get("similarity")
            if sim is None:
                sim = r.get("rrf_score") or r.get("score") or 0.0
            r["activation"] = round(
                compute_activation(
                    similarity=float(sim),
                    recall_count=row.get("recall_count") or 0,
                    distinct_session_count=row.get("distinct_session_count") or 0,
                    memory_type=row.get("memory_type") or "note",
                    created_at=row.get("created_at") or 0.0,
                    last_recalled_at=row.get("last_recalled_at"),
                ),
                4,
            )
            r["distinct_session_count"] = row.get("distinct_session_count") or 0

    @staticmethod
    def _record_memory_edge(
        conn: sqlite3.Connection,
        source_hash: str,
        target_hash: str,
        relationship_type: str,
        weight: float = 1.0,
        valid_from: float | None = None,
        valid_to: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Insert a typed memory↔memory edge into memory_graph.

        Idempotent via UNIQUE(source_hash, target_hash, relationship_type).
        Caller is responsible for transaction boundaries.
        """
        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO memory_graph "
            "(source_hash, target_hash, relationship_type, metadata, "
            " created_at, weight, valid_from, valid_to) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_hash,
                target_hash,
                relationship_type,
                json.dumps(metadata or {}),
                now,
                weight,
                valid_from if valid_from is not None else now,
                valid_to,
            ),
        )

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
            (
                time.time(),
                operation,
                kwargs.get("duration_ms"),
                kwargs.get("query"),
                kwargs.get("search_mode"),
                kwargs.get("result_count"),
                kwargs.get("top_similarity"),
                kwargs.get("content_hash"),
                kwargs.get("dedup_used", False),
                kwargs.get("duplicate_detected", False),
                kwargs.get("duplicate_similarity"),
                kwargs.get("chars_returned"),
            ),
        )

    # --- Entity Linking ---

    def _link_entities(self, conn: sqlite3.Connection, mem_id: int, content: str, tags: list[str]) -> None:
        """Extract entities from content/tags and link them to a memory.

        Creates entities, memory↔entity links, and co-occurrence edges.
        Must be called within an active transaction.
        """
        entities = extract_entities(content, tags)
        if not entities:
            return

        now = time.time()
        entity_ids: list[int] = []

        for name, display, etype in entities:
            conn.execute(
                "INSERT OR IGNORE INTO entities (name, display_name, entity_type, created_at) VALUES (?, ?, ?, ?)",
                (name, display, etype, now),
            )
            row = conn.execute(
                "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
                (name, etype),
            ).fetchone()
            eid = row["id"]
            entity_ids.append(eid)

            # Link memory ↔ entity
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id, created_at) VALUES (?, ?, ?)",
                (mem_id, eid, now),
            )

        # Create/increment co-occurrence edges for all entity pairs
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                a, b = min(entity_ids[i], entity_ids[j]), max(entity_ids[i], entity_ids[j])
                conn.execute(
                    "INSERT INTO entity_relations "
                    "(source_id, target_id, relation_type, weight, created_at, updated_at, valid_from) "
                    "VALUES (?, ?, 'co_occurrence', 1.0, ?, ?, ?) "
                    "ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET "
                    "weight = weight + 1.0, updated_at = ?",
                    (a, b, now, now, now, now),
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
        reject_injection: bool | None = None,
    ) -> dict:
        """Store a single memory. Returns dict with hash and status.

        If dedup_threshold is set (0.0-1.0), checks for semantically similar
        memories before storing. Embeds content once and reuses for both the
        similarity check and the stored embedding.

        importance: explicit 0-1 score. If None, auto-inferred from content/type.

        reject_injection: if True, refuse to store content that matches an injection
        pattern.  If None (default), the env var MEMORY_REJECT_INJECTION=1 acts as a
        global override; absent that, the memory is stored but flagged.
        """
        start = time.time()
        # Normalize string tags to list (defensive — callers may pass comma-separated strings)
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        imp = importance if importance is not None else infer_importance(content, memory_type)

        # --- Injection screening ---
        suspicious, matched_pat = _screen_injection(content)
        if suspicious:
            # Resolve effective reject mode: explicit arg > env var > default (flag-only)
            if reject_injection is None:
                reject_injection = os.environ.get("MEMORY_REJECT_INJECTION", "") == "1"
            if reject_injection:
                return {
                    "error": f"rejected: matches injection pattern '{matched_pat}'",
                    "status": "rejected",
                }
            # Flag-only mode: warn and annotate metadata/tags
            sys.stderr.write(f"[memory] WARNING: stored content matches injection pattern '{matched_pat}'\n")
            metadata = dict(metadata or {})
            metadata["injection_suspicious"] = True
            tags = list(tags or [])
            if "flagged:injection" not in tags:
                tags.append("flagged:injection")

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
                self._track_event(
                    conn,
                    "store",
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

                    embedding = get_model().embed_doc(content)
                conn.execute(
                    """UPDATE memories
                       SET content = ?, tags = ?, memory_type = ?, metadata = ?,
                           updated_at = ?, updated_at_iso = ?,
                           confidence = ?, importance = ?,
                           deleted_at = NULL
                       WHERE id = ?""",
                    (
                        mem.content,
                        json.dumps(mem.tags),
                        mem.memory_type,
                        json.dumps(mem.metadata),
                        mem.updated_at,
                        mem.updated_at_iso,
                        mem.confidence,
                        mem.importance,
                        tomb_id,
                    ),
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
                # Link entities from content and tags
                self._link_entities(conn, tomb_id, content, mem.tags)
                duration_ms = (time.time() - start) * 1000
                self._track_event(
                    conn,
                    "store",
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

                embedding = get_model().embed_doc(content)

            # Similarity-based dedup (scoped to same memory_type)
            if dedup_threshold is not None:
                similar = self._search_semantic(
                    conn,
                    query=None,
                    limit=5,
                    tags=None,
                    time_expr=None,
                    after=None,
                    before=None,
                    _embedding=embedding,
                )
                # Only dedup against same memory_type to avoid false positives
                # across unrelated content (e.g. reference vs decision)
                same_type = [s for s in similar if s.get("memory_type") == memory_type]
                # Further scope by tags: if the new memory has tags, only compare
                # against memories sharing at least one tag (prevents cross-project
                # false positives from shared sentence structure)
                if mem.tags:
                    same_type = [s for s in same_type if any(t in s.get("tags", []) for t in mem.tags)]
                if same_type and same_type[0].get("similarity", 0) >= dedup_threshold:
                    duration_ms = (time.time() - start) * 1000
                    self._track_event(
                        conn,
                        "store",
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
            mem_id = conn.execute("SELECT id FROM memories WHERE content_hash = ?", (mem.content_hash,)).fetchone()[
                "id"
            ]

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

            # Link entities from content and tags
            self._link_entities(conn, mem_id, content, mem.tags)

            duration_ms = (time.time() - start) * 1000
            self._track_event(
                conn,
                "store",
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
        self,
        items: list[dict],
        dedup_threshold: float | None = None,
        reject_injection: bool | None = None,
    ) -> list[dict]:
        """Store multiple memories. Each item: {content, tags?, memory_type?, metadata?}.

        Batches embedding computation for all items upfront, then stores individually.
        reject_injection: passed through to each store() call for injection screening.
        """
        if not items:
            return []

        # Batch-compute all embeddings once
        from .embeddings import get_model

        contents = [item["content"] for item in items]
        embeddings = get_model().embed_doc_batch(contents)

        results = []
        for item, embedding in zip(items, embeddings, strict=True):
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
                reject_injection=reject_injection,
            )
            results.append(result)
        return results

    def auto_archive_pending(
        self,
        *,
        cwd: str | None = None,
        min_age_minutes: int = 5,
        max_sessions: int = 10,
        marker_dir: str | Path | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Scan ~/.claude/projects/<cwd-encoded>/*.jsonl for abandoned sessions.

        For sessions modified >= min_age_minutes ago that lack a marker file,
        trims the transcript and stores it as a session-archive document.
        No LLM, no API key required.

        Returns:
            {scanned, processed, skipped_marked, skipped_too_recent,
             skipped_empty, stored_docs, sessions: [...]}
        """
        import hashlib as _hashlib
        import time as _time
        from datetime import UTC, datetime

        from .transcript import trim_transcript

        # Resolve cwd and encode like Claude Code does:
        # every non-alphanumeric/dash char → "-"
        resolved_cwd = cwd or os.getcwd()

        def _encode_cwd(path: str) -> str:
            return re.sub(r"[^a-zA-Z0-9-]", "-", path)

        encoded = _encode_cwd(resolved_cwd)
        projects_dir = Path.home() / ".claude" / "projects" / encoded

        # Resolve marker directory
        if marker_dir is None:
            marker_dir = Path.home() / ".claude" / "memory" / "extracted"
        marker_path = Path(marker_dir)
        if not dry_run:
            marker_path.mkdir(parents=True, exist_ok=True)

        # Collect candidate JSONL files
        if not projects_dir.exists():
            return {
                "scanned": 0,
                "processed": 0,
                "skipped_marked": 0,
                "skipped_too_recent": 0,
                "skipped_empty": 0,
                "stored_docs": 0,
                "sessions": [],
            }

        now = _time.time()
        min_age_seconds = min_age_minutes * 60

        candidates: list[tuple[float, Path]] = []
        skipped_marked = 0
        skipped_too_recent = 0

        for jsonl_file in projects_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            marker_file = marker_path / f"{session_id}.marker"

            # Skip already-processed sessions
            if marker_file.exists():
                skipped_marked += 1
                continue

            # Skip too-recent sessions (still active)
            mtime = jsonl_file.stat().st_mtime
            age_seconds = now - mtime
            if age_seconds < min_age_seconds:
                skipped_too_recent += 1
                continue

            candidates.append((mtime, jsonl_file))

        # Sort oldest-first, cap at max_sessions
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:max_sessions]

        scanned = skipped_marked + skipped_too_recent + len(candidates)
        skipped_empty = 0
        stored_docs = 0
        session_results: list[dict] = []

        for file_mtime, jsonl_file in candidates:
            session_id = jsonl_file.stem
            session_id_short = session_id[:8]
            marker_file = marker_path / f"{session_id}.marker"
            session_info: dict = {"session_id": session_id_short, "file": str(jsonl_file)}

            # Trim transcript
            try:
                text = trim_transcript(jsonl_file)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"auto_archive_pending: failed to read {jsonl_file.name}: {exc}",
                    file=sys.stderr,
                )
                skipped_empty += 1
                if not dry_run:
                    marker_file.write_text("error\n", encoding="utf-8")
                session_info["status"] = "error"
                session_info["error"] = str(exc)
                session_results.append(session_info)
                continue

            if not text.strip():
                skipped_empty += 1
                if not dry_run:
                    marker_file.write_text("archived\n\n", encoding="utf-8")
                session_info["status"] = "skipped_empty"
                session_results.append(session_info)
                continue

            # Derive cwd_basename from any JSONL entry with a cwd field
            cwd_basename = "unknown"
            try:
                import json as _json

                with open(jsonl_file, encoding="utf-8", errors="replace") as fh:
                    for raw_line in fh:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            obj = _json.loads(raw_line)
                        except _json.JSONDecodeError:
                            continue
                        cwd_val = (
                            obj.get("cwd") or obj.get("message", {}).get("cwd")
                            if isinstance(obj.get("message"), dict)
                            else None
                        )
                        if cwd_val:
                            cwd_basename = Path(cwd_val).name or "unknown"
                            break
            except Exception:  # noqa: BLE001
                pass

            created_date = datetime.fromtimestamp(file_mtime, tz=UTC).strftime("%Y-%m-%d")
            title = f"Session {session_id_short} {cwd_basename} {created_date}"

            summary_text = text[:500]
            if len(text) > 500:
                summary_text += "…"

            tags = ["source:auto", "session-archive", f"project:{cwd_basename}"]

            doc_hash = _hashlib.sha256(text.encode("utf-8")).hexdigest()

            if dry_run:
                session_info["status"] = "dry_run"
                session_info["title"] = title
                session_info["tags"] = tags
                session_info["body_size"] = len(text)
                session_info["doc_hash"] = doc_hash[:16]
                session_results.append(session_info)
                continue

            result = self.store_doc(
                title=title,
                body=text,
                summary=summary_text,
                doc_type="session-archive",
                tags=tags,
                metadata={"session_id": session_id_short, "source_jsonl": str(jsonl_file)},
            )
            stored_hash = result.get("content_hash", doc_hash[:16])

            # Write marker with content hash so sessions can be mapped back to docs
            marker_file.write_text(f"archived\n{stored_hash}\n", encoding="utf-8")

            stored_docs += 1
            session_info["status"] = "archived"
            session_info["doc_hash"] = stored_hash
            session_info["body_size"] = len(text)
            session_results.append(session_info)

        return {
            "scanned": scanned,
            "processed": len(candidates),
            "skipped_marked": skipped_marked,
            "skipped_too_recent": skipped_too_recent,
            "skipped_empty": skipped_empty,
            "stored_docs": stored_docs,
            "sessions": session_results,
        }

    # --- Search ---

    def search(
        self,
        query: str | None = None,
        mode: str = "hybrid",
        limit: int = 10,
        tags: list[str] | None = None,
        time_expr: str | None = None,
        after: str | None = None,
        before: str | None = None,
        scoring_weights: tuple[float, float, float] | None = None,
        exclude_tags: list[str] | None = None,
        memory_types: list[str] | None = None,
        min_importance: float | None = None,
        max_hops: int = 2,
        track_recall: bool = True,
        rerank: bool = False,
        rerank_top_n: int | None = None,
        score_fusion: str = "rrf",
        as_of: float | str | None = None,
    ) -> list[dict]:
        """Search memories. Modes: hybrid (default), semantic, exact, fts, graph.

        Set track_recall=False for automated/background searches (e.g. session-start
        hooks) so they don't inflate recall_count or refresh confidence — that
        signal should reflect user-initiated retrievals only.

        rerank: if True and >=2 results, apply cross-encoder rerank. Final score
        blends 0.4 * composite + 0.6 * cross_encoder_score. Recall tracking is
        applied AFTER reranking so only the surviving top-N reinforce.
        rerank_top_n: truncate to N after rerank (defaults to `limit`).

        score_fusion: "rrf" (default) uses Reciprocal Rank Fusion to merge
        hybrid sub-rankers; "weighted" keeps the legacy additive-score path.
        as_of: Unix timestamp or ISO date.  Graph traversal restricts to edges
        valid at this instant.  Defaults to now.
        """
        start = time.time()
        conn = self._get_conn()
        fetch_limit = limit

        # Normalise as_of to a float timestamp
        as_of_ts: float | None
        if as_of is None:
            as_of_ts = None
        elif isinstance(as_of, (int, float)):
            as_of_ts = float(as_of)
        else:
            as_of_ts = datetime.fromisoformat(as_of).replace(tzinfo=UTC).timestamp()

        # Read phase — no write lock needed yet
        if mode == "exact":
            results = self._search_exact(
                conn,
                query,
                limit,
                tags,
                time_expr,
                after,
                before,
                exclude_tags=exclude_tags,
                memory_types=memory_types,
                min_importance=min_importance,
            )
        elif mode == "semantic":
            results = self._search_semantic(
                conn,
                query,
                fetch_limit,
                tags,
                time_expr,
                after,
                before,
                scoring_weights=scoring_weights,
                exclude_tags=exclude_tags,
                memory_types=memory_types,
                min_importance=min_importance,
            )
        elif mode == "hybrid":
            results = self._search_hybrid(
                conn,
                query,
                fetch_limit,
                tags,
                time_expr,
                after,
                before,
                scoring_weights=scoring_weights,
                exclude_tags=exclude_tags,
                memory_types=memory_types,
                min_importance=min_importance,
                score_fusion=score_fusion,
            )
        elif mode == "fts":
            results = self._search_fts(
                conn,
                query,
                fetch_limit,
                tags,
                time_expr,
                after,
                before,
                scoring_weights=scoring_weights,
                exclude_tags=exclude_tags,
                memory_types=memory_types,
                min_importance=min_importance,
            )
        elif mode == "graph":
            results = self._search_graph(conn, query, fetch_limit, max_hops=max_hops, as_of=as_of_ts)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        # Phase C: enrich each result with ACT-R activation.  Cheap — one round
        # trip to fetch session-count + memory_type for the result hashes.
        self._enrich_with_activation(conn, results)

        # Optional cross-encoder rerank.  Skipped when fewer than 2 results or no
        # query text — single-result lists can't be reordered, and rerank without
        # a query is meaningless.
        if rerank and query and len(results) >= 2:
            from .rerank import get_reranker

            reranker = get_reranker()
            reranker.rerank(query, results, top_n=None)
            for r in results:
                base = r.get("activation", r.get("score", 0.0))
                rr = r.get("rerank_score", 0.0)
                r["score"] = round(0.4 * base + 0.6 * rr, 4)
            results.sort(key=lambda m: m.get("score", 0.0), reverse=True)
            results = results[: rerank_top_n or limit]
        elif USE_ACTIVATION and mode in {"hybrid", "semantic", "fts"} and results:
            # When activation is opted-in, replace composite/RRF score with the
            # session-aware activation so hot-cluster bias drops naturally.
            for r in results:
                if "activation" in r:
                    r["score"] = r["activation"]
            results.sort(key=lambda m: m.get("score", 0.0), reverse=True)
            results = results[:limit]

        # Write phase — acquire lock for event + recall tracking
        duration_ms = (time.time() - start) * 1000
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._track_event(
                conn,
                "search",
                duration_ms=duration_ms,
                query=query,
                search_mode=mode,
                result_count=len(results),
                top_similarity=results[0].get("similarity") if results else None,
                chars_returned=sum(len(m.get("content", "")) for m in results),
            )
            # Update recall counts and reset confidence (reinforcement) for returned memories.
            # Skip when track_recall=False (automated hooks) so recall_count reflects
            # only user-initiated retrievals, preventing auto-recall echo chambers.
            #
            # Session-aware: when a session id is available (MEMORY_SESSION_ID env
            # var, set by the SessionStart hook), bump distinct_session_count only
            # on the first hit per session.  Powers the hot-cluster fix in
            # compute_activation().
            if track_recall:
                now = time.time()
                session_id = os.environ.get("MEMORY_SESSION_ID") or None
                for m in results:
                    if session_id:
                        conn.execute(
                            "UPDATE memories SET recall_count = recall_count + 1, "
                            "last_recalled_at = ?, confidence = 1.0, "
                            "distinct_session_count = distinct_session_count + "
                            "  CASE WHEN COALESCE(last_recall_session, '') = ? THEN 0 ELSE 1 END, "
                            "last_recall_session = ? "
                            "WHERE content_hash = ?",
                            (now, session_id, session_id, m["content_hash"]),
                        )
                    else:
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

    def search_batch(
        self,
        queries: list[str],
        limit: int = 10,
        tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        min_importance: float | None = None,
    ) -> dict:
        """Batch semantic search — embed all queries in one pass.

        Returns {query: [results]} dict.
        """
        if not queries:
            return {}

        from .embeddings import get_model

        embeddings = get_model().embed_query_batch(queries)

        conn = self._get_conn()
        all_results: dict[str, list[dict]] = {}
        all_hashes: set[str] = set()

        for i, query in enumerate(queries):
            results = self._search_semantic(
                conn,
                query,
                limit,
                tags,
                None,
                None,
                None,
                _embedding=embeddings[i],
                exclude_tags=exclude_tags,
                min_importance=min_importance,
            )
            all_results[query] = results
            for m in results:
                all_hashes.add(m["content_hash"])

        # Single recall-tracking transaction for all unique hashes
        if all_hashes:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            try:
                for h in all_hashes:
                    conn.execute(
                        "UPDATE memories SET recall_count = recall_count + 1, "
                        "last_recalled_at = ?, confidence = 1.0 WHERE content_hash = ? AND deleted_at IS NULL",
                        (now, h),
                    )
                self._track_event(
                    conn,
                    "search_batch",
                    query=f"batch({len(queries)})",
                    result_count=sum(len(r) for r in all_results.values()),
                )
                conn.execute("COMMIT")
            except BaseException:
                self._rollback_safe(conn)
                raise

        return all_results

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
            dt = datetime.fromisoformat(after).replace(tzinfo=UTC)
            clauses.append("m.created_at >= ?")
            params.append(dt.timestamp())

        if before:
            dt = datetime.fromisoformat(before).replace(tzinfo=UTC)
            clauses.append("m.created_at <= ?")
            params.append(dt.timestamp())

        return (" AND ".join(clauses), params) if clauses else ("", [])

    def _filter_by_tags(
        self,
        memories: list[dict],
        tags: list[str],
        exclude_tags: list[str] | None = None,
    ) -> list[dict]:
        """Filter memory dicts by tags (any match) and optionally exclude tags."""
        result = memories
        if tags:
            tag_set = {t.lower() for t in tags}
            result = [m for m in result if any(t.lower() in tag_set for t in m.get("tags", []))]
        if exclude_tags:
            excl_set = {t.lower() for t in exclude_tags}
            result = [m for m in result if not any(t.lower() in excl_set for t in m.get("tags", []))]
        return result

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
        exclude_tags: list[str] | None = None,
        memory_types: list[str] | None = None,
        min_importance: float | None = None,
    ) -> list[dict]:
        if _embedding is not None:
            embedding = _embedding
        elif query:
            from .embeddings import get_model

            embedding = get_model().embed_query(query)
        else:
            return []
        # Fetch more than needed to allow post-filtering and re-ranking
        has_filters = tags or time_expr or after or before or exclude_tags or memory_types or min_importance
        fetch_limit = max(limit * 5, 50) if has_filters else max(limit * 3, 30)

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
        if memory_types:
            type_ph = ",".join("?" * len(memory_types))
            sql += f" AND m.memory_type IN ({type_ph})"
            params.extend(memory_types)
        if min_importance is not None:
            sql += " AND m.importance >= ?"
            params.append(min_importance)

        mem_rows = conn.execute(sql, params).fetchall()
        # Build lookup by id, then iterate in original distance order
        row_by_id = {row["id"]: dict(row) for row in mem_rows}

        w_sim, w_imp, w_rec = scoring_weights or DEFAULT_SCORING_WEIGHTS
        memories = []
        for rid in rowids:
            row = row_by_id.get(rid)
            if row is None:
                continue  # filtered out by time/deleted_at/type/importance
            mem = Memory.from_row(row)
            d = mem.to_dict()
            similarity = round(1.0 - distances.get(rid, 1.0), 4)
            if similarity < MIN_SIMILARITY_THRESHOLD:
                continue
            d["similarity"] = similarity

            # Compute decayed confidence
            conf = compute_confidence(
                mem.confidence,
                mem.memory_type,
                row.get("last_recalled_at"),
                mem.created_at,
            )
            d["confidence"] = round(conf, 4)

            # Recall tracking fields
            d["recall_count"] = row.get("recall_count", 0) or 0
            d["last_recalled_at"] = row.get("last_recalled_at")

            # Composite score
            recency = compute_recency(mem.created_at)
            d["score"] = round(w_sim * similarity + w_imp * mem.importance + w_rec * recency, 4)
            d["score_breakdown"] = {
                "similarity": similarity,
                "importance": round(mem.importance, 4),
                "recency": round(recency, 4),
            }
            # Demotion: penalise hot memories so they don't crowd out genuine matches.
            recall_count = row.get("recall_count", 0) or 0
            demotion = 1.0 / (1.0 + DEMOTION_WEIGHT * math.log1p(recall_count))
            d["score"] = round(d["score"] * demotion, 4)
            d["score_breakdown"]["demotion"] = round(demotion, 4)
            d["score_breakdown"]["recall_count"] = recall_count
            memories.append(d)

        if tags or exclude_tags:
            memories = self._filter_by_tags(memories, tags or [], exclude_tags=exclude_tags)

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
        exclude_tags: list[str] | None = None,
        memory_types: list[str] | None = None,
        min_importance: float | None = None,
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

        if memory_types:
            type_ph = ",".join("?" * len(memory_types))
            sql += f" AND m.memory_type IN ({type_ph})"
            params.extend(memory_types)

        if min_importance is not None:
            sql += " AND m.importance >= ?"
            params.append(min_importance)

        has_post_filters = tags or exclude_tags
        sql += " ORDER BY m.created_at DESC LIMIT ?"
        params.append(limit * 5 if has_post_filters else limit)

        rows = conn.execute(sql, params).fetchall()
        memories = []
        for r in rows:
            row_dict = dict(r)
            d = Memory.from_row(row_dict).to_dict()
            d["recall_count"] = row_dict.get("recall_count", 0) or 0
            d["last_recalled_at"] = row_dict.get("last_recalled_at")
            memories.append(d)

        if tags or exclude_tags:
            memories = self._filter_by_tags(memories, tags or [], exclude_tags=exclude_tags)

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
        exclude_tags: list[str] | None = None,
        memory_types: list[str] | None = None,
        min_importance: float | None = None,
    ) -> list[dict]:
        """BM25 full-text search via FTS5. No embedding model required."""
        if not query:
            return []

        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []

        has_filters = tags or time_expr or after or before or exclude_tags or memory_types or min_importance
        fetch_limit = max(limit * 5, 50) if has_filters else max(limit * 3, 30)

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
        rank_range = max_rank - min_rank

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
        if memory_types:
            type_ph = ",".join("?" * len(memory_types))
            sql += f" AND m.memory_type IN ({type_ph})"
            params.extend(memory_types)
        if min_importance is not None:
            sql += " AND m.importance >= ?"
            params.append(min_importance)

        mem_rows = conn.execute(sql, params).fetchall()

        w_sim, w_imp, w_rec = scoring_weights or DEFAULT_SCORING_WEIGHTS
        memories = []
        for row in mem_rows:
            row_dict = dict(row)
            mem = Memory.from_row(row_dict)
            d = mem.to_dict()

            rid = row_dict["id"]
            rank = ranks[rid]
            similarity = 1.0 if rank_range == 0 else round((max_rank - rank) / rank_range, 4)
            d["similarity"] = similarity

            conf = compute_confidence(
                mem.confidence,
                mem.memory_type,
                row_dict.get("last_recalled_at"),
                mem.created_at,
            )
            d["confidence"] = round(conf, 4)
            d["recall_count"] = row_dict.get("recall_count", 0) or 0
            d["last_recalled_at"] = row_dict.get("last_recalled_at")

            recency = compute_recency(mem.created_at)
            d["score"] = round(w_sim * similarity + w_imp * mem.importance + w_rec * recency, 4)
            d["score_breakdown"] = {
                "similarity": similarity,
                "importance": round(mem.importance, 4),
                "recency": round(recency, 4),
            }
            # Demotion: penalise hot memories so they don't crowd out genuine matches.
            recall_count = row_dict.get("recall_count", 0) or 0
            demotion = 1.0 / (1.0 + DEMOTION_WEIGHT * math.log1p(recall_count))
            d["score"] = round(d["score"] * demotion, 4)
            d["score_breakdown"]["demotion"] = round(demotion, 4)
            d["score_breakdown"]["recall_count"] = recall_count
            memories.append(d)

        if tags or exclude_tags:
            memories = self._filter_by_tags(memories, tags or [], exclude_tags=exclude_tags)

        memories.sort(key=lambda m: m["score"], reverse=True)
        return memories[:limit]

    def _search_hybrid(
        self,
        conn: sqlite3.Connection,
        query: str | None,
        limit: int,
        tags: list[str] | None,
        time_expr: str | None,
        after: str | None,
        before: str | None,
        scoring_weights: tuple[float, float, float] | None = None,
        exclude_tags: list[str] | None = None,
        memory_types: list[str] | None = None,
        min_importance: float | None = None,
        score_fusion: str = "weighted",
    ) -> list[dict]:
        """Hybrid search: merge semantic + FTS results.

        score_fusion="weighted" (legacy): semantic score + fts score additively.
        score_fusion="rrf": Reciprocal Rank Fusion — robust to score-scale
        mismatch between dense cosine and BM25.  Score = Σ 1/(60+rank_i).
        """
        sem_results = self._search_semantic(
            conn,
            query,
            limit * 2,
            tags,
            time_expr,
            after,
            before,
            scoring_weights=scoring_weights,
            exclude_tags=exclude_tags,
            memory_types=memory_types,
            min_importance=min_importance,
        )
        fts_results = self._search_fts(
            conn,
            query,
            limit * 2,
            tags,
            time_expr,
            after,
            before,
            scoring_weights=scoring_weights,
            exclude_tags=exclude_tags,
            memory_types=memory_types,
            min_importance=min_importance,
        )

        if score_fusion == "rrf":
            return _rrf_fuse(sem_results, fts_results, limit=limit, k=60)

        # Legacy weighted-sum fusion
        results_by_hash: dict[str, dict] = {}
        for r in sem_results:
            results_by_hash[r["content_hash"]] = r
        for r in fts_results:
            h = r["content_hash"]
            if h in results_by_hash:
                existing = results_by_hash[h]
                existing["similarity"] = max(
                    existing.get("similarity", 0),
                    r.get("similarity", 0),
                )
                existing["score"] = existing.get("score", 0) + r.get("score", 0)
            else:
                results_by_hash[h] = r

        merged = list(results_by_hash.values())
        merged.sort(key=lambda m: m["score"], reverse=True)
        return merged[:limit]

    def _search_graph(
        self,
        conn: sqlite3.Connection,
        query: str | None,
        limit: int,
        max_hops: int = 2,
        as_of: float | None = None,
    ) -> list[dict]:
        """Graph traversal search: find memories connected through shared entities.

        as_of: Unix timestamp.  Only edges valid at this instant are traversed
        (valid_from <= as_of < valid_to, NULL endpoints treated as open).
        Defaults to now.
        """
        if not query:
            return []
        if as_of is None:
            as_of = time.time()

        # Extract entities from query using same regex patterns as store()
        extracted = extract_entities(query)
        candidate_names = list({name for name, _, _ in extracted})
        if not candidate_names:
            # Fallback: use query as-is (original behavior)
            candidate_names = [query.lower()]

        # Find seed entities matching ANY extracted entity
        conditions = []
        params: list[str] = []
        for name in candidate_names:
            conditions.append("(name = ? OR name LIKE ?)")
            params.extend([name, f"{name}%"])
        # Also keep display_name substring match on original query for flexibility
        conditions.append("display_name LIKE ?")
        params.append(f"%{query}%")

        where_clause = " OR ".join(conditions)
        seed_rows = conn.execute(
            f"SELECT id, name, display_name, entity_type FROM entities WHERE {where_clause}",
            params,
        ).fetchall()

        if not seed_rows:
            return []

        seed_ids = [r["id"] for r in seed_rows]

        # Recursive CTE: traverse entity_relations up to max_hops, restricted
        # to edges valid at `as_of`.
        placeholders = ",".join("?" * len(seed_ids))
        cte_sql = f"""
            WITH RECURSIVE graph_walk(entity_id, hops, path_weight) AS (
                -- Seed: the matched entities at hop 0
                SELECT id, 0, 1.0
                FROM entities WHERE id IN ({placeholders})

                UNION ALL

                -- Walk edges (both directions), bi-temporal filter
                SELECT
                    CASE WHEN er.source_id = gw.entity_id THEN er.target_id ELSE er.source_id END,
                    gw.hops + 1,
                    gw.path_weight * er.weight
                FROM graph_walk gw
                JOIN entity_relations er
                    ON er.source_id = gw.entity_id OR er.target_id = gw.entity_id
                WHERE gw.hops < ?
                  AND (er.valid_from IS NULL OR er.valid_from <= ?)
                  AND (er.valid_to   IS NULL OR er.valid_to   >  ?)
            )
            SELECT DISTINCT
                m.id, m.content_hash, m.content, m.tags, m.memory_type,
                m.metadata, m.created_at, m.updated_at, m.created_at_iso,
                m.updated_at_iso, m.confidence, m.importance,
                m.recall_count, m.last_recalled_at,
                MIN(gw.hops) as graph_hops,
                MAX(gw.path_weight) as graph_weight
            FROM graph_walk gw
            JOIN memory_entities me ON me.entity_id = gw.entity_id
            JOIN memories m ON m.id = me.memory_id
            WHERE m.deleted_at IS NULL
            GROUP BY m.id
            ORDER BY MIN(gw.hops) ASC, MAX(gw.path_weight) DESC
            LIMIT ?
        """
        params = [*seed_ids, max_hops, as_of, as_of, limit]
        rows = conn.execute(cte_sql, params).fetchall()

        if not rows:
            return []

        memories = []
        for row in rows:
            row_dict = dict(row)
            mem = Memory.from_row(row_dict)
            d = mem.to_dict()

            hops = row_dict["graph_hops"]
            weight = row_dict["graph_weight"]
            recency = compute_recency(mem.created_at)

            # Score: proximity (hops), importance, recency
            d["score"] = round(0.5 / (1 + hops) + 0.3 * mem.importance + 0.2 * recency, 4)
            d["graph_hops"] = hops
            d["graph_weight"] = round(weight, 4)

            conf = compute_confidence(
                mem.confidence,
                mem.memory_type,
                row_dict.get("last_recalled_at"),
                mem.created_at,
            )
            d["confidence"] = round(conf, 4)
            d["recall_count"] = row_dict.get("recall_count", 0) or 0
            d["last_recalled_at"] = row_dict.get("last_recalled_at")

            # Attach entity names for this memory
            entity_rows = conn.execute(
                "SELECT e.display_name, e.entity_type FROM entities e "
                "JOIN memory_entities me ON me.entity_id = e.id "
                "WHERE me.memory_id = ?",
                (row_dict["id"],),
            ).fetchall()
            d["entities"] = [{"name": er["display_name"], "type": er["entity_type"]} for er in entity_rows]

            memories.append(d)

        memories.sort(key=lambda m: m["score"], reverse=True)
        return memories[:limit]

    # --- List ---

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        memory_types: list[str] | None = None,
    ) -> dict:
        """Paginated listing with optional filters."""
        conn = self._get_conn()
        offset = (page - 1) * page_size

        sql = "SELECT * FROM memories m WHERE m.deleted_at IS NULL"
        count_sql = "SELECT COUNT(*) as cnt FROM memories m WHERE m.deleted_at IS NULL"
        params: list = []
        count_params: list = []

        if memory_types:
            type_ph = ",".join("?" * len(memory_types))
            sql += f" AND m.memory_type IN ({type_ph})"
            count_sql += f" AND m.memory_type IN ({type_ph})"
            params.extend(memory_types)
            count_params.extend(memory_types)
        elif memory_type:
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
        rows = conn.execute("SELECT tags FROM memories WHERE deleted_at IS NULL").fetchall()
        counter: Counter = Counter()
        for r in rows:
            for t in _safe_tags(r["tags"]):
                counter[t] += 1
        sorted_tags = counter.most_common()
        return {
            "total_unique": len(sorted_tags),
            "tags": [{"tag": t, "count": c} for t, c in sorted_tags],
        }

    # --- Knowledge Graph ---

    def list_entities(self, entity_type: str | None = None, limit: int = 50) -> dict:
        """List entities with memory counts."""
        conn = self._get_conn()
        sql = """
            SELECT e.id, e.name, e.display_name, e.entity_type,
                   COUNT(me.memory_id) as memory_count
            FROM entities e
            LEFT JOIN memory_entities me ON me.entity_id = e.id
            GROUP BY e.id
        """
        params: list = []
        if entity_type:
            sql = """
                SELECT e.id, e.name, e.display_name, e.entity_type,
                       COUNT(me.memory_id) as memory_count
                FROM entities e
                LEFT JOIN memory_entities me ON me.entity_id = e.id
                WHERE e.entity_type = ?
                GROUP BY e.id
            """
            params.append(entity_type)

        sql += " ORDER BY memory_count DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return {
            "total": len(rows),
            "entities": [
                {
                    "name": r["display_name"],
                    "type": r["entity_type"],
                    "memory_count": r["memory_count"],
                }
                for r in rows
            ],
        }

    def entity_context(self, entity_name: str, limit: int = 20) -> dict:
        """Full context for an entity: info, connected memories, related entities."""
        conn = self._get_conn()

        # Find entity (case-insensitive)
        row = conn.execute(
            "SELECT id, name, display_name, entity_type FROM entities WHERE name = ?",
            (entity_name.lower(),),
        ).fetchone()
        if not row:
            # Try prefix match
            row = conn.execute(
                "SELECT id, name, display_name, entity_type FROM entities WHERE name LIKE ?",
                (entity_name.lower() + "%",),
            ).fetchone()
        if not row:
            return {"error": f"Entity not found: {entity_name}"}

        entity_id = row["id"]
        entity_info = {
            "name": row["display_name"],
            "type": row["entity_type"],
        }

        # Connected memories
        mem_rows = conn.execute(
            "SELECT m.content_hash, m.content, m.memory_type, m.tags, "
            "m.importance, m.created_at, m.recall_count "
            "FROM memories m "
            "JOIN memory_entities me ON me.memory_id = m.id "
            "WHERE me.entity_id = ? AND m.deleted_at IS NULL "
            "ORDER BY m.created_at DESC LIMIT ?",
            (entity_id, limit),
        ).fetchall()
        memories = [
            {
                "content_hash": r["content_hash"],
                "content": r["content"],
                "memory_type": r["memory_type"],
                "tags": _safe_tags(r["tags"]),
                "importance": r["importance"],
                "recall_count": r["recall_count"] or 0,
            }
            for r in mem_rows
        ]

        # Related entities (via edges)
        rel_rows = conn.execute(
            "SELECT e.display_name, e.entity_type, er.weight "
            "FROM entity_relations er "
            "JOIN entities e ON (e.id = CASE WHEN er.source_id = ? THEN er.target_id ELSE er.source_id END) "
            "WHERE er.source_id = ? OR er.target_id = ? "
            "ORDER BY er.weight DESC LIMIT 20",
            (entity_id, entity_id, entity_id),
        ).fetchall()
        related = [
            {
                "name": r["display_name"],
                "type": r["entity_type"],
                "weight": round(r["weight"], 1),
            }
            for r in rel_rows
        ]

        return {
            "entity": entity_info,
            "memories": memories,
            "related_entities": related,
            "total_memories": len(memories),
            "total_related": len(related),
        }

    def build_graph(self, dry_run: bool = False) -> dict:
        """Build/rebuild the knowledge graph from all existing memories.

        Processes all non-deleted memories through entity extraction.
        Safe to run multiple times (uses INSERT OR IGNORE).
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT id, content, tags FROM memories WHERE deleted_at IS NULL").fetchall()

        total = len(rows)
        if dry_run:
            # Estimate entities without writing
            entity_set: set[tuple[str, str]] = set()
            for r in rows:
                entities = extract_entities(r["content"], _safe_tags(r["tags"]))
                for name, _display, etype in entities:
                    entity_set.add((name, etype))
            return {
                "dry_run": True,
                "memories_to_process": total,
                "estimated_entities": len(entity_set),
            }

        conn.execute("BEGIN IMMEDIATE")
        try:
            entities_before = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
            links_before = conn.execute("SELECT COUNT(*) as cnt FROM memory_entities").fetchone()["cnt"]
            edges_before = conn.execute("SELECT COUNT(*) as cnt FROM entity_relations").fetchone()["cnt"]

            for r in rows:
                self._link_entities(conn, r["id"], r["content"], _safe_tags(r["tags"]))

            entities_after = conn.execute("SELECT COUNT(*) as cnt FROM entities").fetchone()["cnt"]
            links_after = conn.execute("SELECT COUNT(*) as cnt FROM memory_entities").fetchone()["cnt"]
            edges_after = conn.execute("SELECT COUNT(*) as cnt FROM entity_relations").fetchone()["cnt"]

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {
            "memories_processed": total,
            "entities": entities_after,
            "entities_new": entities_after - entities_before,
            "links": links_after,
            "links_new": links_after - links_before,
            "edges": edges_after,
            "edges_new": edges_after - edges_before,
        }

    def rename_tag(self, old_tag: str, new_tag: str) -> dict:
        """Rename a tag across all memories and documents."""
        conn = self._begin_immediate()
        try:
            memories_updated = 0
            documents_updated = 0

            # Update memories
            rows = conn.execute("SELECT id, tags FROM memories WHERE deleted_at IS NULL").fetchall()
            for r in rows:
                tags = _safe_tags(r["tags"])
                if old_tag.lower() not in [t.lower() for t in tags]:
                    continue
                new_tags = []
                for t in tags:
                    if t.lower() == old_tag.lower():
                        if new_tag.lower() not in [nt.lower() for nt in new_tags]:
                            new_tags.append(new_tag)
                    else:
                        new_tags.append(t)
                # Dedup case-insensitively
                seen = set()
                deduped = []
                for t in new_tags:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        deduped.append(t)
                conn.execute(
                    "UPDATE memories SET tags = ? WHERE id = ?",
                    (json.dumps(deduped), r["id"]),
                )
                memories_updated += 1

            # Update documents
            rows = conn.execute("SELECT id, tags FROM documents WHERE deleted_at IS NULL").fetchall()
            for r in rows:
                tags = _safe_tags(r["tags"])
                if old_tag.lower() not in [t.lower() for t in tags]:
                    continue
                new_tags = []
                for t in tags:
                    if t.lower() == old_tag.lower():
                        if new_tag.lower() not in [nt.lower() for nt in new_tags]:
                            new_tags.append(new_tag)
                    else:
                        new_tags.append(t)
                seen = set()
                deduped = []
                for t in new_tags:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        deduped.append(t)
                conn.execute(
                    "UPDATE documents SET tags = ? WHERE id = ?",
                    (json.dumps(deduped), r["id"]),
                )
                documents_updated += 1

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {
            "old_tag": old_tag,
            "new_tag": new_tag,
            "memories_updated": memories_updated,
            "documents_updated": documents_updated,
        }

    def merge_tags(self, source_tags: list[str], target_tag: str) -> dict:
        """Merge multiple source tags into a single target tag."""
        conn = self._begin_immediate()
        try:
            source_set = {t.lower() for t in source_tags}
            memories_updated = 0
            documents_updated = 0

            # Update memories
            rows = conn.execute("SELECT id, tags FROM memories WHERE deleted_at IS NULL").fetchall()
            for r in rows:
                tags = _safe_tags(r["tags"])
                if not any(t.lower() in source_set for t in tags):
                    continue
                new_tags = []
                replaced = False
                for t in tags:
                    if t.lower() in source_set:
                        if not replaced:
                            new_tags.append(target_tag)
                            replaced = True
                    else:
                        new_tags.append(t)
                # Dedup
                seen = set()
                deduped = []
                for t in new_tags:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        deduped.append(t)
                conn.execute(
                    "UPDATE memories SET tags = ? WHERE id = ?",
                    (json.dumps(deduped), r["id"]),
                )
                memories_updated += 1

            # Update documents
            rows = conn.execute("SELECT id, tags FROM documents WHERE deleted_at IS NULL").fetchall()
            for r in rows:
                tags = _safe_tags(r["tags"])
                if not any(t.lower() in source_set for t in tags):
                    continue
                new_tags = []
                replaced = False
                for t in tags:
                    if t.lower() in source_set:
                        if not replaced:
                            new_tags.append(target_tag)
                            replaced = True
                    else:
                        new_tags.append(t)
                seen = set()
                deduped = []
                for t in new_tags:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        deduped.append(t)
                conn.execute(
                    "UPDATE documents SET tags = ? WHERE id = ?",
                    (json.dumps(deduped), r["id"]),
                )
                documents_updated += 1

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {
            "source_tags": source_tags,
            "target_tag": target_tag,
            "memories_updated": memories_updated,
            "documents_updated": documents_updated,
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
            return {
                "error": "No filter specified — refusing to delete all memories. "
                "Provide at least one of: content_hash, tags, before, after. "
                "Example: memory delete <hash> | memory delete --tags "
                '"scope:temp" | memory delete --before 2025-01-01 --dry-run'
            }

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
                        return {
                            "error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries. "
                            "Use a longer prefix (or the full 64-char hash) to uniquely identify the entry. "
                            "Run 'memory list' or 'memory search' to find exact hashes."
                        }
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
                conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (mem_id,))
                conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (mem_id,))
                deleted_hashes.append(r["content_hash"])

            duration_ms = (time.time() - start) * 1000
            self._track_event(
                conn,
                "delete",
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
                return {
                    "error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries. "
                    "Use a longer prefix (or the full 64-char hash) to uniquely identify the entry. "
                    "Run 'memory list' or 'memory search' to find exact hashes."
                }
        if not row:
            return {
                "error": f"Memory not found: {content_hash}. The hash may be incorrect, or the memory was deleted. "
                "Use 'memory list' to see existing memories, or 'memory search <query>' to find by content."
            }
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
                    return {
                        "error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries. "
                        "Use a longer prefix (or the full 64-char hash) to uniquely identify the entry. "
                        "Run 'memory list' or 'memory search' to find exact hashes."
                    }
            if not row:
                conn.execute("ROLLBACK")
                return {
                    "error": f"Memory not found: {content_hash}. The hash may be incorrect, or the memory was deleted. "
                    "Use 'memory list' to see existing memories, or 'memory search <query>' to find by content."
                }

            sets = []
            params: list = []
            new_hash = None

            if "content" in updates:
                new_content = updates["content"]
                new_hash = hashlib.sha256(new_content.encode()).hexdigest()
                # Recompute embedding
                from .embeddings import get_model

                new_embedding = get_model().embed_doc(new_content)
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
                now_iso = datetime.fromtimestamp(now, tz=UTC).isoformat()
                sets.append("updated_at = ?")
                params.append(now)
                sets.append("updated_at_iso = ?")
                params.append(now_iso)

            if not sets:
                conn.execute("ROLLBACK")
                return {
                    "error": "No valid updates provided. Supported fields: content, tags, memory_type, metadata, "
                    "importance, confidence. Example: memory update <hash> --content 'new text' --tags 'tag1,tag2'"
                }

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
            with contextlib.suppress(Exception):
                conn.execute("ROLLBACK")  # read-only, nothing to commit

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
            dt = datetime.fromisoformat(after).replace(tzinfo=UTC)
            ev_clauses.append("timestamp >= ?")
            ev_params.append(dt.timestamp())
        if before:
            dt = datetime.fromisoformat(before).replace(tzinfo=UTC)
            ev_clauses.append("timestamp <= ?")
            ev_params.append(dt.timestamp())
        ev_where = (" AND " + " AND ".join(ev_clauses)) if ev_clauses else ""

        # --- Memory counts ---
        total = conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NULL").fetchone()["cnt"]
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
            return conn.execute(sql, [op, *ev_params]).fetchone()["cnt"]

        def _ev_avg(op: str, col: str) -> float | None:
            sql = f"SELECT AVG({col}) as val FROM operation_events WHERE operation = ? AND {col} IS NOT NULL{ev_where}"
            row = conn.execute(sql, [op, *ev_params]).fetchone()
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

        result.update(
            {
                "stores_total": stores_total,
                "stores_per_day": round(stores_total / days_active, 1),
                "duplicates_total": duplicates_total,
                "dedup_rate": round(duplicates_total / dedup_attempts, 3) if dedup_attempts else None,
                "searches_total": searches_total,
                "searches_no_results": searches_no_results,
                "search_hit_rate": round((searches_total - searches_no_results) / searches_total, 3)
                if searches_total
                else None,
                "avg_top_similarity": _ev_avg("search", "top_similarity"),
                "avg_results_per_search": _ev_avg("search", "result_count"),
                "total_chars_returned": conn.execute(
                    f"SELECT COALESCE(SUM(chars_returned), 0) as val FROM operation_events "
                    f"WHERE operation = 'search'{ev_where}",
                    ev_params,
                ).fetchone()["val"],
                "deletes_total": deletes_total,
                "avg_store_ms": _ev_avg("store", "duration_ms"),
                "avg_search_ms": _ev_avg("search", "duration_ms"),
            }
        )

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
                    "created_at_iso": datetime.fromtimestamp(r["created_at"], tz=UTC).isoformat()
                    if r["created_at"]
                    else None,
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
                    "created_at_iso": datetime.fromtimestamp(r["created_at"], tz=UTC).isoformat()
                    if r["created_at"]
                    else None,
                    "tags": _safe_tags(r["tags"]),
                }
                for r in rows
            ]

        # --- Top tags across recalled memories ---
        tag_rows = conn.execute("SELECT tags FROM memories WHERE deleted_at IS NULL AND recall_count > 0").fetchall()
        tags_flat = (t for r in tag_rows for t in _safe_tags(r["tags"]))
        result["top_tags"] = Counter(tags_flat).most_common(15)

        return result

    # --- Health ---

    def health(self) -> dict:
        """Database health and stats."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NULL").fetchone()["cnt"]
        deleted = conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NOT NULL").fetchone()["cnt"]
        embeddings = conn.execute("SELECT COUNT(*) as cnt FROM memory_embeddings_rowids").fetchone()["cnt"]

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        types = conn.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories WHERE deleted_at IS NULL GROUP BY memory_type"
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
                    pairs.append(
                        {
                            "keep_hash": keep["content_hash"],
                            "remove_hash": remove["content_hash"],
                            "similarity": round(sim, 4),
                        }
                    )
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
                    conn.execute("SELECT tags FROM memories WHERE content_hash = ?", (keep_h,)).fetchone()["tags"]
                )
                remove_tags = _safe_tags(
                    conn.execute("SELECT tags FROM memories WHERE content_hash = ?", (remove_h,)).fetchone()["tags"]
                )
                merged_tags = list(dict.fromkeys(keep_tags + remove_tags))  # preserve order, dedup
                conn.execute(
                    "UPDATE memories SET tags = ? WHERE content_hash = ?",
                    (json.dumps(merged_tags), keep_h),
                )
                # Record lineage edge BEFORE soft-deleting so audit survives.
                self._record_memory_edge(
                    conn,
                    source_hash=remove_h,
                    target_hash=keep_h,
                    relationship_type="merged_into",
                    weight=float(p["similarity"]),
                    valid_from=now,
                    metadata={"similarity": p["similarity"]},
                )
                # Soft-delete the removed memory
                remove_row = conn.execute("SELECT id FROM memories WHERE content_hash = ?", (remove_h,)).fetchone()
                conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (now, remove_row["id"]),
                )
                conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (remove_row["id"],))
                conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (remove_row["id"],))

            duration_ms = (time.time() - start) * 1000
            self._track_event(
                conn,
                "consolidate",
                duration_ms=duration_ms,
                result_count=len(pairs),
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {"consolidated": len(pairs), "pairs": pairs}

    # --- Dream pass ---

    # Patterns for relative-date rewrite pass
    _DREAM_YESTERDAY_RE = re.compile(r"\byesterday\b", re.IGNORECASE)
    _DREAM_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
    _DREAM_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
    _DREAM_THIS_MORNING_RE = re.compile(r"\bthis morning\b", re.IGNORECASE)
    _DREAM_N_DAYS_AGO_RE = re.compile(r"\b(\d+)\s+days?\s+ago\b", re.IGNORECASE)
    _DREAM_N_WEEKS_AGO_RE = re.compile(r"\b(\d+)\s+weeks?\s+ago\b", re.IGNORECASE)
    _DREAM_WEEKDAY_RE = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE)

    # Keywords that indicate the newer memory supersedes an older one
    _DREAM_CONTRADICTION_RE = re.compile(r"\b(now|actually|updated|fixed|replaced|instead)\b", re.IGNORECASE)
    _DREAM_SUPERSEDING_TYPES = {"decision", "error"}

    def _dream_rewrite_dates(self, conn: sqlite3.Connection, dry_run: bool) -> int:
        """Pass 1: rewrite relative date phrases to absolute ISO dates.

        Uses each memory's created_at as the anchor.  Only rewrites when at
        least one pattern matches, and only writes back when content actually
        changed.  Does NOT recompute embeddings — the semantic meaning is
        preserved; we only make temporal references concrete.
        """
        rows = conn.execute(
            "SELECT id, content_hash, content, created_at FROM memories WHERE deleted_at IS NULL"
        ).fetchall()

        _weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        rewritten = 0
        for r in rows:
            original = r["content"]
            anchor = datetime.fromtimestamp(r["created_at"], tz=UTC)

            text = original

            # yesterday / today / tomorrow / this morning
            if self._DREAM_YESTERDAY_RE.search(text):
                d = (anchor - timedelta(days=1)).strftime("%Y-%m-%d")
                text = self._DREAM_YESTERDAY_RE.sub(d, text)
            if self._DREAM_TODAY_RE.search(text):
                d = anchor.strftime("%Y-%m-%d")
                text = self._DREAM_TODAY_RE.sub(d, text)
            if self._DREAM_TOMORROW_RE.search(text):
                d = (anchor + timedelta(days=1)).strftime("%Y-%m-%d")
                text = self._DREAM_TOMORROW_RE.sub(d, text)
            if self._DREAM_THIS_MORNING_RE.search(text):
                d = anchor.strftime("%Y-%m-%d")
                text = self._DREAM_THIS_MORNING_RE.sub(d, text)

            # "N days ago"
            def _replace_days(m: re.Match) -> str:
                n = int(m.group(1))
                return (anchor - timedelta(days=n)).strftime("%Y-%m-%d")

            text = self._DREAM_N_DAYS_AGO_RE.sub(_replace_days, text)

            # "N weeks ago"
            def _replace_weeks(m: re.Match) -> str:
                n = int(m.group(1))
                return (anchor - timedelta(weeks=n)).strftime("%Y-%m-%d")

            text = self._DREAM_N_WEEKS_AGO_RE.sub(_replace_weeks, text)

            # weekday names within last 7 days of anchor
            def _replace_weekday(m: re.Match) -> str:
                name = m.group(1).lower()
                target_dow = _weekdays[name]
                anchor_dow = anchor.weekday()
                days_back = (anchor_dow - target_dow) % 7
                if days_back == 0:
                    days_back = 7  # "Monday" from a Monday means last Monday
                candidate = anchor - timedelta(days=days_back)
                return candidate.strftime("%Y-%m-%d")

            text = self._DREAM_WEEKDAY_RE.sub(_replace_weekday, text)

            if text == original:
                continue

            rewritten += 1
            if dry_run:
                continue

            new_hash = hashlib.sha256(text.encode()).hexdigest()
            now_ts = time.time()
            now_iso = datetime.fromtimestamp(now_ts, tz=UTC).isoformat()
            conn.execute(
                "UPDATE memories SET content = ?, content_hash = ?, updated_at = ?, updated_at_iso = ? WHERE id = ?",
                (text, new_hash, now_ts, now_iso, r["id"]),
            )
            # FTS5 content-table update: delete old row, insert new.
            # Plain UPDATE is not supported on content FTS5 tables.
            conn.execute(
                "INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', ?, ?)",
                (r["id"], original),
            )
            conn.execute(
                "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                (r["id"], text),
            )

        return rewritten

    def _dream_supersession(self, conn: sqlite3.Connection, dry_run: bool) -> tuple[int, list[dict]]:
        """Pass 2: find pairs sharing ≥2 tags + cosine similarity ≥0.85 where
        the newer entry contains a contradiction keyword or is a superseding type.
        Soft-delete the older entry and record superseded_by in its metadata.
        """
        import numpy as np

        rows = conn.execute(
            "SELECT m.id, m.content_hash, m.content, m.tags, m.memory_type, "
            "m.created_at, m.metadata "
            "FROM memories m WHERE m.deleted_at IS NULL"
        ).fetchall()

        if len(rows) < 2:
            return 0, []

        id_list = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(id_list))
        emb_rows = conn.execute(
            f"SELECT rowid, content_embedding FROM memory_embeddings WHERE rowid IN ({placeholders})",
            id_list,
        ).fetchall()
        emb_by_id = {r["rowid"]: r["content_embedding"] for r in emb_rows}

        # Build index of (vector, tags, metadata) per valid row
        valid: list[dict] = []
        vectors: list = []
        for r in rows:
            emb_bytes = emb_by_id.get(r["id"])
            if emb_bytes is None:
                continue
            vec = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            vectors.append(vec)
            valid.append(
                {
                    "id": r["id"],
                    "content_hash": r["content_hash"],
                    "content": r["content"],
                    "tags": set(_safe_tags(r["tags"])),
                    "memory_type": r["memory_type"],
                    "created_at": r["created_at"],
                    "metadata": r["metadata"],
                }
            )

        if len(vectors) < 2:
            return 0, []

        mat = np.stack(vectors)
        sim_matrix = mat @ mat.T  # cosine (L2-normalised in store())

        superseded_count = 0
        superseded_pairs: list[dict] = []
        deleted_ids: set[int] = set()
        n = len(valid)
        now = time.time()

        for i in range(n):
            if valid[i]["id"] in deleted_ids:
                continue
            for j in range(i + 1, n):
                if valid[j]["id"] in deleted_ids:
                    continue
                sim = float(sim_matrix[i, j])
                if sim < 0.85:
                    continue

                shared_tags = valid[i]["tags"] & valid[j]["tags"]
                if len(shared_tags) < 2:
                    continue

                # Determine which is newer
                if valid[i]["created_at"] >= valid[j]["created_at"]:
                    newer, older = valid[i], valid[j]
                else:
                    newer, older = valid[j], valid[i]

                # Gate: newer must mention contradiction keyword OR be a superseding type
                is_superseding = (
                    self._DREAM_CONTRADICTION_RE.search(newer["content"]) is not None
                    or newer["memory_type"] in self._DREAM_SUPERSEDING_TYPES
                )
                if not is_superseding:
                    continue

                superseded_pairs.append(
                    {
                        "older": older["content_hash"],
                        "newer": newer["content_hash"],
                        "similarity": round(sim, 4),
                    }
                )
                deleted_ids.add(older["id"])

                if dry_run:
                    continue

                # Record typed memory↔memory edge BEFORE soft-delete so the
                # lineage survives.  Direction: newer 'supersedes' older.
                self._record_memory_edge(
                    conn,
                    source_hash=newer["content_hash"],
                    target_hash=older["content_hash"],
                    relationship_type="supersedes",
                    weight=sim,
                    valid_from=now,
                    metadata={"similarity": round(sim, 4)},
                )

                # Soft-delete older, preserve audit trail in metadata
                old_meta = json.loads(older["metadata"] or "{}")
                old_meta["superseded_by"] = newer["content_hash"]
                conn.execute(
                    "UPDATE memories SET deleted_at = ?, metadata = ? WHERE id = ?",
                    (now, json.dumps(old_meta), older["id"]),
                )
                conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (older["id"],))
                conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (older["id"],))

                superseded_count += 1

        return superseded_count if not dry_run else len(superseded_pairs), superseded_pairs

    def _dream_tag_suggestions(self, conn: sqlite3.Connection) -> list[dict]:
        """Pass 4: suggest tag merges for near-duplicate tag names.

        Detects: case-only differences, trailing punctuation, singular/plural.
        Returns suggestions only — no auto-merge.
        """
        import re as _re

        rows = conn.execute(
            "SELECT DISTINCT value as tag FROM ("
            "  SELECT json_each.value FROM memories, json_each(memories.tags)"
            "  WHERE deleted_at IS NULL"
            ")"
        ).fetchall()
        all_tags: list[str] = [r["tag"] for r in rows]

        # Normalise for comparison: lowercase, strip trailing punctuation
        def _norm(tag: str) -> str:
            t = tag.lower().rstrip(":.,;")
            # singular/plural: strip trailing 's' for short stems (≥4 chars)
            if len(t) > 4 and t.endswith("s"):
                t = t[:-1]
            return t

        norm_to_canonical: dict[str, str] = {}
        suggestions: list[dict] = []
        seen_pairs: set[frozenset] = set()

        for tag in sorted(all_tags):
            n = _norm(tag)
            if n in norm_to_canonical:
                canonical = norm_to_canonical[n]
                if canonical != tag:
                    pair = frozenset({canonical, tag})
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        # Suggest merging the less-canonical into the first-seen
                        suggestions.append({"from": tag, "to": canonical})
            else:
                norm_to_canonical[n] = tag

        return suggestions

    def _dream_demote(self, conn: sqlite3.Connection, dry_run: bool) -> int:
        """Pass 5: mark old never-recalled auto-tagged memories as demoted.

        Criteria: recall_count=0 AND created_at > 30 days ago AND
        tags contains only 'source:auto' entries or is empty.
        Sets metadata.demoted=true. Does NOT soft-delete.
        """
        cutoff = time.time() - 30 * 86400
        rows = conn.execute(
            "SELECT id, tags, metadata FROM memories WHERE deleted_at IS NULL AND recall_count = 0 AND created_at < ?",
            (cutoff,),
        ).fetchall()

        demoted = 0
        now = time.time()
        now_iso = datetime.fromtimestamp(now, tz=UTC).isoformat()
        for r in rows:
            tags = _safe_tags(r["tags"])
            # Only demote if tags are all 'source:auto' or empty
            non_auto = [t for t in tags if t != "source:auto"]
            if non_auto:
                continue
            meta = json.loads(r["metadata"] or "{}")
            if meta.get("demoted"):
                continue  # already demoted — idempotent

            demoted += 1
            if dry_run:
                continue

            meta["demoted"] = True
            conn.execute(
                "UPDATE memories SET metadata = ?, updated_at = ?, updated_at_iso = ? WHERE id = ?",
                (json.dumps(meta), now, now_iso, r["id"]),
            )

        return demoted

    def _dream_active_forget(
        self,
        conn: sqlite3.Connection,
        dry_run: bool,
        max_fraction: float = 0.05,
    ) -> tuple[int, list[dict]]:
        """Pass 6: active budget-aware forgetting.

        Candidate criteria:
          - activation < FORGET_THRESHOLD
          - distinct_session_count <= 1
          - created_at older than 30 days
          - NOT a 'decision' or 'reference' (intent-permanent types are sacred)

        Soft-deletes at most ceil(max_fraction * total_active) per pass.
        Gated on env var MEMORY_ACTIVE_FORGET=1 to make rollout opt-in;
        when unset, this pass is a no-op (returns 0).
        Always safe under dry_run.
        """
        if os.environ.get("MEMORY_ACTIVE_FORGET") != "1":
            return 0, []

        now = time.time()
        cutoff = now - 30 * 86400

        rows = conn.execute(
            "SELECT id, content_hash, memory_type, created_at, last_recalled_at, "
            "recall_count, distinct_session_count "
            "FROM memories WHERE deleted_at IS NULL AND created_at < ? "
            "AND memory_type NOT IN ('decision', 'reference') "
            "AND COALESCE(distinct_session_count, 0) <= 1",
            (cutoff,),
        ).fetchall()

        candidates: list[dict] = []
        for r in rows:
            row = dict(r)
            act = compute_activation(
                similarity=0.0,  # no query context during sleep
                recall_count=row["recall_count"] or 0,
                distinct_session_count=row["distinct_session_count"] or 0,
                memory_type=row["memory_type"] or "note",
                created_at=row["created_at"] or 0.0,
                last_recalled_at=row["last_recalled_at"],
                now=now,
            )
            if act < FORGET_THRESHOLD:
                row["activation"] = round(act, 4)
                candidates.append(row)

        if not candidates:
            return 0, []

        # Bounded budget: never delete more than max_fraction of active corpus.
        total_active = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE deleted_at IS NULL"
        ).fetchone()["n"]
        budget = max(1, int(math.ceil(max_fraction * total_active)))
        # Forget lowest-activation first.
        candidates.sort(key=lambda c: c["activation"])
        candidates = candidates[:budget]

        forgotten_pairs = [
            {"content_hash": c["content_hash"], "activation": c["activation"]}
            for c in candidates
        ]

        if dry_run:
            return len(forgotten_pairs), forgotten_pairs

        for c in candidates:
            conn.execute("UPDATE memories SET deleted_at = ? WHERE id = ?", (now, c["id"]))
            conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (c["id"],))
            conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (c["id"],))

        return len(forgotten_pairs), forgotten_pairs

    def dream(
        self,
        dry_run: bool = False,
        threshold_new: int = 50,
        threshold_age_hours: int = 24,
    ) -> dict:
        """Run a composite maintenance pass ("dream") over the memory store.

        Passes executed in order:
          1. Relative-date rewrite  — anchors temporal phrases to ISO dates
          2. Supersession detection — soft-deletes older memories overridden by newer
          3. Consolidation          — merges near-duplicates (delegates to self.consolidate)
          4. Tag normalisation      — surfaces tag-merge suggestions (no auto-merge)
          5. Index demotion         — marks old never-recalled auto-tagged memories
          6. Active forgetting      — soft-deletes low-activation memories (env-gated)

        Returns a JSON-serialisable summary dict.

        Trigger heuristic (scheduling hook — not enforced here):
          Fire when new_memories_since_last_dream >= threshold_new
          OR hours_since_last_dream >= threshold_age_hours.
          Last run timestamp is persisted in the metadata table under
          key='last_dream_at' as a Unix timestamp string.
        """
        start = time.time()
        conn = self._begin_immediate()

        try:
            # --- Pass 1: relative-date rewrite ---
            dates_rewritten = self._dream_rewrite_dates(conn, dry_run=dry_run)

            # --- Pass 2: supersession detection ---
            superseded, superseded_pairs = self._dream_supersession(conn, dry_run=dry_run)

            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        # --- Pass 3: consolidate (manages its own transaction) ---
        consolidate_result = self.consolidate(dry_run=dry_run)
        consolidated = consolidate_result.get("consolidated", 0) or consolidate_result.get("would_consolidate", 0)

        # --- Passes 4 & 5: read-only pass + demote (needs write) ---
        conn = self._begin_immediate()
        try:
            # Pass 4: tag suggestions (read-only)
            tag_suggestions = self._dream_tag_suggestions(conn)

            # Pass 5: demote old unrecalled auto-tagged memories
            demoted = self._dream_demote(conn, dry_run=dry_run)

            # Pass 6: active forgetting (env-gated)
            forgotten, forgotten_pairs = self._dream_active_forget(conn, dry_run=dry_run)

            # Persist last_dream_at timestamp (skip on dry_run)
            now = time.time()
            if not dry_run:
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_dream_at', ?)",
                    (str(now),),
                )

            duration_ms = (time.time() - start) * 1000
            self._track_event(
                conn,
                "dream",
                duration_ms=duration_ms,
                result_count=dates_rewritten + superseded + consolidated + demoted + forgotten,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return {
            "dry_run": dry_run,
            "dates_rewritten": dates_rewritten,
            "superseded": superseded,
            "superseded_pairs": superseded_pairs,
            "consolidated": consolidated,
            "tag_merge_suggestions": tag_suggestions,
            "demoted": demoted,
            "forgotten": forgotten,
            "forgotten_pairs": forgotten_pairs,
            "duration_ms": round((time.time() - start) * 1000, 2),
        }

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
                self._track_event(
                    conn,
                    "briefing",
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
                r["confidence"] or 1.0,
                r["memory_type"],
                r["last_recalled_at"],
                r["created_at"],
            )
            imp = r["importance"] or 0.5
            days = max(0.0, (now - r["created_at"]) / 86400)
            recency = 1.0 / (1.0 + days)
            score = conf * imp * recency
            scored.append(
                {
                    "content": r["content"],
                    "memory_type": r["memory_type"],
                    "score": score,
                    "created_at": r["created_at"],
                }
            )

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
        type_prefix_re = re.compile(r"^\[(Pattern|Observation|Decision|Learning|Error|Note|Reference)\]\s*")
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
                first_line = type_prefix_re.sub("", first_line)
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
            self._track_event(
                conn,
                "briefing",
                duration_ms=duration_ms,
                result_count=total_lines,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise

        return result

    def demoted(self, limit: int = 50) -> list[dict]:
        """Return memories most penalised by the demotion ranker.

        Computes a representative composite score for each recalled memory
        (using DEFAULT_SCORING_WEIGHTS and recency at query time), then shows
        how much of that score the demotion factor is taking away.

        Each entry: {hash, content_preview, recall_count, demotion_factor,
                     score_loss_pct}

        Sorted by score_loss_pct descending so the most-penalised memories
        appear first — useful for auditing recall-count inflation.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT content_hash, content, importance, created_at, recall_count "
            "FROM memories "
            "WHERE recall_count > 0 AND deleted_at IS NULL"
        ).fetchall()

        if not rows:
            return []

        w_sim, w_imp, w_rec = DEFAULT_SCORING_WEIGHTS
        entries = []
        for row in rows:
            rc = row["recall_count"] or 0
            recency = compute_recency(row["created_at"])
            importance = row["importance"] or 0.5

            # Representative score without demotion (use similarity = 1.0
            # as a canonical upper-bound proxy so the loss is comparable).
            score_without = w_sim * 1.0 + w_imp * importance + w_rec * recency
            demotion = 1.0 / (1.0 + DEMOTION_WEIGHT * math.log1p(rc))
            score_with = score_without * demotion
            score_loss_pct = round((1.0 - demotion) * 100, 2)

            preview = row["content"][:120]

            entries.append(
                {
                    "hash": row["content_hash"],
                    "content_preview": preview,
                    "recall_count": rc,
                    "demotion_factor": round(demotion, 4),
                    "score_loss_pct": score_loss_pct,
                    # Include raw scores for debugging
                    "score_without_demotion": round(score_without, 4),
                    "score_with_demotion": round(score_with, 4),
                }
            )

        entries.sort(key=lambda e: e["score_loss_pct"], reverse=True)
        return entries[:limit]

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
                    r["confidence"] or 1.0,
                    r["memory_type"],
                    r["last_recalled_at"],
                    r["created_at"],
                )
                if min_confidence > 0 and new_conf < min_confidence:
                    conn.execute(
                        "UPDATE memories SET deleted_at = ? WHERE id = ?",
                        (now, r["id"]),
                    )
                    conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (r["id"],))
                    conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (r["id"],))
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
                self._track_event(
                    conn,
                    "doc_store",
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

            embedding = get_model().embed_doc(summary)

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
            doc_id = conn.execute("SELECT id FROM documents WHERE content_hash = ?", (doc.content_hash,)).fetchone()[
                "id"
            ]

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
            self._track_event(
                conn,
                "doc_store",
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
                return {
                    "error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries. "
                    "Use a longer prefix (or the full 64-char hash) to uniquely identify the entry. "
                    "Run 'memory list' or 'memory search' to find exact hashes."
                }
        if not row:
            return {
                "error": f"Document not found: {content_hash}. The hash may be incorrect, "
                "or the document was deleted. Use 'memory doc list' to see existing documents, "
                "or 'memory doc search <query>' to find by content."
            }
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
                conn,
                query,
                limit=limit * 2 if mode == "auto" else limit,
            )
            for r in sem_results:
                results_by_hash[r["content_hash"]] = r

        if mode in ("fts", "auto"):
            fts_results = self._search_docs_fts(
                conn,
                query,
                limit=limit * 2 if mode == "auto" else limit,
            )
            for r in fts_results:
                h = r["content_hash"]
                if h in results_by_hash:
                    # Merge: take max similarity, add scores
                    existing = results_by_hash[h]
                    existing["similarity"] = max(
                        existing.get("similarity", 0),
                        r.get("similarity", 0),
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
            self._track_event(
                conn,
                "doc_search",
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

        embedding = get_model().embed_query(query)

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
            "SELECT rowid, rank FROM document_fts WHERE document_fts MATCH ? ORDER BY rank LIMIT ?",
            (safe_query, fetch_limit),
        ).fetchall()

        if not fts_rows:
            return []

        # BM25 rank is negative: more negative = better. Normalize to [0, 1].
        ranks = {r["rowid"]: r["rank"] for r in fts_rows}
        rank_values = list(ranks.values())
        min_rank = min(rank_values)
        max_rank = max(rank_values)
        rank_range = max_rank - min_rank

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
            similarity = 1.0 if rank_range == 0 else round((max_rank - rank) / rank_range, 4)
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
                    return {
                        "error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries. "
                        "Use a longer prefix (or the full 64-char hash) to uniquely identify the entry. "
                        "Run 'memory list' or 'memory search' to find exact hashes."
                    }
            if not row:
                conn.execute("ROLLBACK")
                return {
                    "error": f"Document not found: {content_hash}. The hash may be incorrect, "
                    "or the document was deleted. Use 'memory doc list' to see existing documents, "
                    "or 'memory doc search <query>' to find by content."
                }

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
                return {
                    "error": "No valid updates provided. Supported fields: content, tags, memory_type, metadata, "
                    "importance, confidence. Example: memory update <hash> --content 'new text' --tags 'tag1,tag2'"
                }

            # Always update timestamps
            now = time.time()
            now_iso = datetime.fromtimestamp(now, tz=UTC).isoformat()
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

                new_embedding = get_model().embed_doc(kwargs["summary"])
                conn.execute("DELETE FROM document_embeddings WHERE rowid = ?", (doc_id,))
                conn.execute(
                    "INSERT INTO document_embeddings (rowid, summary_embedding) VALUES (?, ?)",
                    (doc_id, _serialize_f32(new_embedding)),
                )

            # Update FTS if title or body changed
            # External content FTS5 tables require the special delete command
            # with the original values, then re-insert with new values.
            if title_changed or body_changed:
                conn.execute(
                    "INSERT INTO document_fts(document_fts, rowid, title, body) VALUES('delete', ?, ?, ?)",
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
                    return {
                        "error": f"Ambiguous hash prefix '{content_hash}' matches {len(rows)} entries. "
                        "Use a longer prefix (or the full 64-char hash) to uniquely identify the entry. "
                        "Run 'memory list' or 'memory search' to find exact hashes."
                    }

            if not rows:
                conn.execute("ROLLBACK")
                return {
                    "error": f"Document not found: {content_hash}. The hash may be incorrect, "
                    "or the document was deleted. Use 'memory doc list' to see existing documents, "
                    "or 'memory doc search <query>' to find by content."
                }

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
            conn.execute("DELETE FROM document_embeddings WHERE rowid = ?", (doc_id,))

            duration_ms = (time.time() - start) * 1000
            self._track_event(
                conn,
                "doc_delete",
                duration_ms=duration_ms,
                content_hash=full_hash,
                result_count=1,
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_safe(conn)
            raise
        return {"deleted": 1, "content_hash": full_hash}

    # --- Export / Import ---

    def export_all(self, include_documents: bool = True) -> dict:
        """Export all non-deleted memories (and optionally documents).

        Returns a portable dict suitable for JSON serialization.
        """
        conn = self._get_conn()
        now_iso = datetime.now(UTC).isoformat()

        # Export memories
        mem_rows = conn.execute("SELECT * FROM memories WHERE deleted_at IS NULL ORDER BY created_at").fetchall()
        memories = []
        for r in mem_rows:
            row_dict = dict(r)
            d = Memory.from_row(row_dict).to_dict()
            d["recall_count"] = row_dict.get("recall_count", 0) or 0
            d["last_recalled_at"] = row_dict.get("last_recalled_at")
            memories.append(d)

        result: dict = {
            "version": 1,
            "exported_at": now_iso,
            "memories": memories,
        }

        # Export documents
        if include_documents:
            doc_rows = conn.execute("SELECT * FROM documents WHERE deleted_at IS NULL ORDER BY created_at").fetchall()
            documents = [Document.from_row(dict(r)).to_dict() for r in doc_rows]
            result["documents"] = documents
        else:
            result["documents"] = []

        return result

    def import_all(self, data: dict, force: bool = False) -> dict:
        """Import memories and documents from an export dict.

        force=True skips dedup (sets dedup_threshold=None).
        Returns counts of imported/skipped items.
        """
        memories = data.get("memories", [])
        documents = data.get("documents", [])

        dedup_threshold = None if force else 0.90

        memories_imported = 0
        memories_skipped = 0
        for m in memories:
            result = self.store(
                content=m["content"],
                tags=m.get("tags", []),
                memory_type=m.get("memory_type", "note"),
                metadata=m.get("metadata", {}),
                dedup_threshold=dedup_threshold,
                importance=m.get("importance"),
            )
            if result.get("status") in ("stored", "revived"):
                memories_imported += 1
            else:
                memories_skipped += 1

        documents_imported = 0
        documents_skipped = 0
        for d in documents:
            result = self.store_doc(
                title=d.get("title", ""),
                body=d.get("body", ""),
                summary=d.get("summary", ""),
                doc_type=d.get("doc_type", "document"),
                tags=d.get("tags", []),
                metadata=d.get("metadata", {}),
            )
            if result.get("status") in ("stored", "revived"):
                documents_imported += 1
            else:
                documents_skipped += 1

        return {
            "memories_imported": memories_imported,
            "memories_skipped": memories_skipped,
            "documents_imported": documents_imported,
            "documents_skipped": documents_skipped,
        }

    # ------------------------------------------------------------------ #
    #  Index front door                                                    #
    # ------------------------------------------------------------------ #

    def build_index(self, *, max_lines: int = 200, max_tokens: int = 4000) -> str:
        """Build a curated TOC of the memory store for SessionStart injection.

        Selection rules (priority order):
        1. Exclude: deleted_at NOT NULL, metadata.demoted==true,
           metadata.injection_suspicious==true.
        2. Top tier: memory_type in (decision, reference) AND at least one
           non-'source:auto' tag (human-curated).
        3. Second tier: memory_type in (learning, pattern, error), ranked by
           recency × importance × distinct_recall_proxy / log(recall_count+1),
           with 0.5× multiplier for source:auto entries.
        4. Third tier: memory_type == 'todo', status PENDING or BLOCKED
           (parsed from [TODO:PENDING|DONE|BLOCKED] prefix).
        5. Hard cap: stop when adding next entry would exceed max_lines or
           max_tokens (approximated as len(text)//4).

        Returns markdown string with a Demoted footer.
        """
        conn = self._get_conn()

        # Fetch all active memories
        rows = conn.execute(
            "SELECT content_hash, content, tags, memory_type, metadata, "
            "       importance, created_at, recall_count, last_recalled_at "
            "FROM memories WHERE deleted_at IS NULL"
        ).fetchall()

        total_active = len(rows)
        now = time.time()

        # --- Filter and classify ---
        tier1: list[dict] = []  # decisions + references (human-curated)
        tier2: list[dict] = []  # learning / pattern / error
        tier3: list[dict] = []  # todo (PENDING / BLOCKED)
        excluded_count = 0

        _todo_re = re.compile(r"^\[TODO:(PENDING|DONE|BLOCKED)\]", re.IGNORECASE)

        for row in rows:
            tags = _safe_tags(row["tags"])
            try:
                meta = json.loads(row["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError):
                meta = {}

            # Rule 1: hard exclusions
            if meta.get("demoted") is True or meta.get("injection_suspicious") is True:
                excluded_count += 1
                continue

            mtype = row["memory_type"] or "note"
            content = row["content"] or ""
            imp = row["importance"] or 0.5
            rc = row["recall_count"] or 0
            created_at = row["created_at"] or now
            is_auto = "source:auto" in tags
            auto_mult = 0.5 if is_auto else 1.0

            entry = {
                "hash": row["content_hash"],
                "content": content,
                "memory_type": mtype,
                "tags": tags,
                "is_auto": is_auto,
            }

            if mtype in ("decision", "reference"):
                # Top tier only if has at least one non-source:auto tag
                has_curated_tag = any(t != "source:auto" for t in tags)
                if has_curated_tag:
                    tier1.append(entry)
                else:
                    # Falls to tier 2 scoring (no human curation)
                    recency = compute_recency(created_at)
                    score = auto_mult * recency * imp / math.log1p(rc + 1)
                    entry["score"] = score
                    tier2.append(entry)

            elif mtype in ("learning", "pattern", "error"):
                recency = compute_recency(created_at)
                # distinct_recall_proxy: approximate unique sessions via log
                distinct_proxy = math.log1p(rc) if rc > 0 else 1.0
                score = auto_mult * recency * imp * distinct_proxy / math.log1p(rc + 1)
                entry["score"] = score
                tier2.append(entry)

            elif mtype == "todo":
                m = _todo_re.match(content)
                if m:
                    status = m.group(1).upper()
                    if status in ("PENDING", "BLOCKED"):
                        tier3.append(entry)
                    # DONE todos silently excluded (not demoted)
                else:
                    # No status prefix — include as PENDING
                    tier3.append(entry)

            # Everything else (note, observation, etc.) is excluded from index
            # but not counted as "demoted" (they're tier-4 / not curated)

        # Sort tier2 by score descending
        tier2.sort(key=lambda e: e.get("score", 0.0), reverse=True)

        # --- Build markdown sections ---
        iso_now = datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        def _entry_line(e: dict) -> str:
            h = e["hash"][:12]
            mtype = e["memory_type"]
            tags_str = " ".join(e["tags"][:3]) if e["tags"] else ""
            preview = e["content"][:80].replace("\n", " ")
            return f"- `{h}` {mtype} {tags_str} — {preview}"

        def _approx_tokens(text: str) -> int:
            return len(text) // 4

        # Header placeholder — filled at end
        sections: dict[str, list[str]] = {
            "Decisions": [],
            "References": [],
            "Learnings / Patterns / Errors": [],
            "Active TODOs": [],
        }

        line_count = 0
        token_estimate = 0
        included_count = 0
        demoted_from_cap = 0

        def _would_exceed(line: str) -> bool:
            return (line_count + 1 > max_lines) or (token_estimate + _approx_tokens(line) > max_tokens)

        def _add_to(section: str, entry: dict) -> bool:
            nonlocal line_count, token_estimate, included_count, demoted_from_cap
            line = _entry_line(entry)
            if _would_exceed(line):
                demoted_from_cap += 1
                return False
            sections[section].append(line)
            line_count += 1
            token_estimate += _approx_tokens(line)
            included_count += 1
            return True

        # Tier 1: decisions then references
        for e in tier1:
            if e["memory_type"] == "decision":
                _add_to("Decisions", e)
            else:
                _add_to("References", e)

        # Tier 2: learning/pattern/error
        for e in tier2:
            _add_to("Learnings / Patterns / Errors", e)

        # Tier 3: todos
        for e in tier3:
            _add_to("Active TODOs", e)

        # Total demoted = explicitly excluded + cap-overflow
        total_demoted = excluded_count + demoted_from_cap

        # --- Assemble markdown ---
        lines: list[str] = [
            f"# Memory Index — {iso_now}",
            "",
            f"**Total memories:** {total_active} (active: {total_active}, demoted: {total_demoted})",
            f"**Index lines:** {line_count} / {max_lines}",
            "**Generated by:** memory admin index",
            "",
        ]

        section_order = [
            ("## Decisions", "Decisions"),
            ("## References", "References"),
            ("## Learnings / Patterns / Errors", "Learnings / Patterns / Errors"),
            ("## Active TODOs", "Active TODOs"),
        ]
        for heading, key in section_order:
            lines.append(heading)
            if sections[key]:
                lines.extend(sections[key])
            else:
                lines.append("*none*")
            lines.append("")

        lines.append("## Demoted (search-only; not auto-loaded)")
        lines.append(
            f"*{total_demoted}* entries available via `memory search` or "
            "`memory get <hash>` but excluded from this index."
        )

        return "\n".join(lines) + "\n"
