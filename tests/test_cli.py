"""CLI integration tests using stdout capture."""

import io
import json
import sys
from unittest.mock import patch

import pytest

from memory.cli import main
from memory.core import MemoryStore


@pytest.fixture
def cli_env(tmp_path):
    """Set up a CLI test environment with a temp DB.

    Monkeypatches DB_PATH so the CLI's own MemoryStore() uses the temp DB.
    """
    db = tmp_path / "cli_test.db"
    # Pre-initialize the schema
    s = MemoryStore(db_path=db)
    conn = s._get_conn()
    conn.executescript("""
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
            deleted_at REAL DEFAULT NULL
        );
    """)
    s._migrate_stats_tables()
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings "
        "USING vec0(content_embedding FLOAT[384] distance_metric=cosine)"
    )
    s.close()

    with patch("memory.core.DB_PATH", db):
        yield db


def _invoke(cli_env, args):
    """Invoke CLI command in JSON mode, return captured stdout."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        main(args)
    return buf.getvalue()


def _invoke_text(cli_env, args):
    """Invoke CLI command in text mode, return captured stdout."""
    return _invoke(cli_env, ["-f", "text"] + args)


def test_store_json_output(cli_env):
    """'memory store' outputs valid JSON with status."""
    output = _invoke(cli_env, ["store", "CLI test memory"])
    data = json.loads(output)
    assert data["status"] == "stored"
    assert "content_hash" in data


def test_store_text_output(cli_env):
    """'memory -f text store' outputs 'stored <hash>'."""
    output = _invoke_text(cli_env, ["store", "text format test"])
    assert output.startswith("stored ")


def test_store_duplicate_text(cli_env):
    """Duplicate in text mode shows 'duplicate <hash>'."""
    _invoke(cli_env, ["store", "dup content"])
    output = _invoke_text(cli_env, ["store", "dup content"])
    assert "duplicate" in output


def test_search_json_output(cli_env):
    """'memory search' returns JSON array."""
    _invoke(cli_env, ["store", "searchable CLI content"])
    output = _invoke(cli_env, ["search", "searchable", "--mode", "exact"])
    data = json.loads(output)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_search_text_no_results(cli_env):
    """Text output shows 'no results' when empty."""
    output = _invoke_text(cli_env, ["search", "nonexistent_xyz_123", "--mode", "exact"])
    assert "no results" in output


def test_list_json_output(cli_env):
    """List returns pagination keys."""
    _invoke(cli_env, ["store", "list item 1"])
    _invoke(cli_env, ["store", "list item 2"])
    output = _invoke(cli_env, ["list"])
    data = json.loads(output)
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "memories" in data


def test_delete_dry_run(cli_env):
    """--dry-run output includes would_delete count."""
    store_output = _invoke(cli_env, ["store", "to delete via cli"])
    h = json.loads(store_output)["content_hash"]
    output = _invoke(cli_env, ["delete", "--hash", h, "--dry-run"])
    data = json.loads(output)
    assert data["dry_run"] is True
    assert data["would_delete"] == 1


def test_health_json(cli_env):
    """Health returns status: healthy."""
    _invoke(cli_env, ["store", "health check item"])
    output = _invoke(cli_env, ["health"])
    data = json.loads(output)
    assert data["status"] == "healthy"


def test_stats_json(cli_env):
    """Stats returns expected aggregate keys."""
    _invoke(cli_env, ["store", "stats item"])
    output = _invoke(cli_env, ["stats"])
    data = json.loads(output)
    assert "total_memories" in data
    assert "stores_total" in data


def test_cleanup_json(cli_env):
    """Cleanup returns duplicates_removed."""
    _invoke(cli_env, ["store", "cleanup test"])
    output = _invoke(cli_env, ["cleanup"])
    data = json.loads(output)
    assert "duplicates_removed" in data


def test_store_batch_stdin(cli_env):
    """Pipe JSON array via stdin."""
    batch = json.dumps([
        {"content": "batch stdin 1", "tags": ["test"]},
        {"content": "batch stdin 2"},
    ])
    buf = io.StringIO()
    with patch("sys.stdout", buf), patch("sys.stdin", io.StringIO(batch)):
        main(["store-batch"])
    output = buf.getvalue()
    data = json.loads(output)
    assert isinstance(data, list)
    assert len(data) == 2
    assert all(r["status"] == "stored" for r in data)


def test_update_no_args_error(cli_env):
    """Missing updates exits with error."""
    store_output = _invoke(cli_env, ["store", "update target"])
    h = json.loads(store_output)["content_hash"]
    with pytest.raises(SystemExit) as exc_info:
        main(["update", h])
    assert exc_info.value.code != 0
