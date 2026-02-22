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
    yield s
    s.close()
