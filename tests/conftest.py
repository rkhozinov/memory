"""Shared test fixtures for memory service tests."""

import pytest

from memory.core import MemoryStore

# SQL schema matching the production database structure.
# Production relies on a pre-existing DB file; tests create tables from scratch.
_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    memory_type TEXT DEFAULT 'note',
    metadata TEXT DEFAULT '{}',
    created_at REAL,
    updated_at REAL,
    created_at_iso TEXT,
    updated_at_iso TEXT,
    deleted_at REAL DEFAULT NULL,
    confidence REAL DEFAULT 1.0,
    importance REAL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

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


@pytest.fixture
def store(tmp_path):
    """Create a MemoryStore backed by a temporary database with full schema."""
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=db)
    conn = s._get_conn()  # triggers _migrate_stats_tables (creates operation_events + recall columns)

    # Base tables must exist before the ALTER in _migrate_stats_tables,
    # but _get_conn already ran the migration. Re-running is safe (IF NOT EXISTS / try/except).
    # Create base schema first, then re-trigger migration for the recall columns.
    conn.executescript(_BASE_SCHEMA)

    # Re-run migration so recall_count and last_recalled_at are added
    s._migrate_stats_tables()

    # sqlite-vec virtual table for embeddings
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings "
        "USING vec0(content_embedding FLOAT[384] distance_metric=cosine)"
    )
    # FTS5 virtual table for BM25 keyword search
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
        "USING fts5(content, content='memories', content_rowid='id', "
        "tokenize='porter ascii')"
    )

    # document-specific virtual tables
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS document_embeddings "
        "USING vec0(summary_embedding FLOAT[384] distance_metric=cosine)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS document_fts "
        "USING fts5(title, body, content='documents', content_rowid='id', "
        "tokenize='porter ascii')"
    )
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with realistic memories modeled on production data."""
    memories = [
        # Ticket identifiers — the core problem scenario
        (
            "TICKET-24: RDS instance class upgrade from db.t3.medium to db.r5.large in production",
            "decision",
            ["project:infrastructure", "cloud:aws"],
            0.8,
        ),
        (
            "TICKET-75: DNS records are managed manually in <registrar>. After K8s deployment,"
            " must manually create CNAME pointing to ALB hostname",
            "learning",
            ["project:infrastructure", "svc:billing-manager"],
            0.9,
        ),
        (
            "TICKET-198: Eliminate billing-manager Node.js sidecar. Install frontend-templates at Docker build time",
            "decision",
            ["project:infrastructure", "svc:billing-manager"],
            0.8,
        ),
        (
            "TICKET-64: notifications-service PRs: infra repo PR #N covers ECR, IAM policy, SM secret, IRSA",
            "decision",
            ["project:infrastructure", "svc:diagnostics"],
            0.8,
        ),
        (
            "TICKET-109, TICKET-110, TICKET-111 are unassigned In Progress tickets that need owner cleanup",
            "observation",
            ["project:myproject", "tool:linear-cli"],
            0.9,
        ),
        # Kubernetes/deployment content
        (
            "Kubernetes pod crash loop backoff: check container exit code,"
            " OOM kills, and liveness probe misconfiguration",
            "error",
            ["project:infrastructure", "svc:kubernetes"],
            0.7,
        ),
        (
            "Kubernetes node autoscaler scales down aggressively during low traffic windows",
            "learning",
            ["project:infrastructure", "svc:kubernetes"],
            0.6,
        ),
        # Terraform content
        (
            "Terraform S3 backend state locking requires DynamoDB table with LockID partition key",
            "pattern",
            ["tool:terraform", "cloud:aws"],
            0.7,
        ),
        # Unrelated content (noise floor)
        (
            ".NET Dockerfile with BuildKit secrets is incompatible with QEMU cross-compilation",
            "pattern",
            ["tool:docker"],
            0.7,
        ),
        (
            "Go60 ZMK firmware: disabled BLE and RGB underglow, firmware shrunk 70%",
            "decision",
            ["project:go60"],
            0.8,
        ),
    ]
    for content, mtype, tags, imp in memories:
        store.store(content, memory_type=mtype, tags=tags, importance=imp)
    return store
