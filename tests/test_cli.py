"""CLI integration tests. All output is JSON."""

import io
import json
from unittest.mock import patch

import pytest

from memory.cli import main
from memory.core import MemoryStore


@pytest.fixture
def cli_env(tmp_path):
    """Set up a CLI test environment with a temp DB."""
    db = tmp_path / "cli_test.db"
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
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
        "USING fts5(content, content='memories', content_rowid='id', "
        "tokenize='porter ascii')"
    )
    s.close()
    with patch("memory.core.DB_PATH", db):
        yield db


def _invoke(cli_env, args) -> dict | list:
    """Invoke CLI, parse JSON output."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        main(args)
    return json.loads(buf.getvalue())


def _invoke_raw(cli_env, args) -> str:
    """Invoke CLI, return raw stdout."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        main(args)
    return buf.getvalue()


# --- store ---


def test_store_plain_text(cli_env):
    data = _invoke(cli_env, ["store", "CLI test memory"])
    assert data["status"] == "stored"
    assert "content_hash" in data


def test_store_json_object(cli_env):
    obj = json.dumps({"content": "json store test", "type": "decision", "tags": ["test"]})
    data = _invoke(cli_env, ["store", obj])
    assert data["status"] == "stored"


def test_store_json_array(cli_env):
    arr = json.dumps([{"content": "batch 1"}, {"content": "batch 2"}])
    data = _invoke(cli_env, ["store", arr])
    assert isinstance(data, list)
    assert len(data) == 2
    assert all(r["status"] == "stored" for r in data)


def test_store_stdin(cli_env):
    obj = json.dumps({"content": "stdin test"})
    buf = io.StringIO()
    with patch("sys.stdout", buf), patch("sys.stdin", io.StringIO(obj)):
        main(["store", "-"])
    data = json.loads(buf.getvalue())
    assert data["status"] == "stored"


def test_store_duplicate(cli_env):
    _invoke(cli_env, ["store", "dup content"])
    data = _invoke(cli_env, ["store", "dup content"])
    assert data["status"] == "duplicate"


def test_store_with_flags(cli_env):
    data = _invoke(cli_env, ["store", "flagged content", "--tags", "a,b", "--type", "decision", "--importance", "0.9"])
    assert data["status"] == "stored"


# --- search ---


def test_search_basic(cli_env):
    _invoke(cli_env, ["store", "searchable CLI content"])
    data = _invoke(cli_env, ["search", "searchable", "--mode", "exact"])
    assert isinstance(data, list)
    assert len(data) >= 1


def test_search_with_types(cli_env):
    _invoke(cli_env, ["store", "decision content", "--type", "decision"])
    _invoke(cli_env, ["store", "error content", "--type", "error"])
    data = _invoke(cli_env, ["search", "content", "--mode", "exact", "--types", "decision"])
    contents = [r["content"] for r in data]
    assert any("decision" in c for c in contents)
    assert not any("error" in c for c in contents)


# --- get ---


def test_get_by_hash(cli_env):
    stored = _invoke(cli_env, ["store", "get test content"])
    h = stored["content_hash"]
    data = _invoke(cli_env, ["get", h])
    assert data["content_hash"] == h
    assert data["content"] == "get test content"


def test_get_by_prefix(cli_env):
    stored = _invoke(cli_env, ["store", "prefix get test"])
    h = stored["content_hash"]
    data = _invoke(cli_env, ["get", h[:8]])
    assert data["content_hash"] == h


def test_get_not_found(cli_env):
    with pytest.raises(SystemExit):
        _invoke(cli_env, ["get", "nonexistent_hash_000"])


# --- delete ---


def test_delete_by_hash(cli_env):
    stored = _invoke(cli_env, ["store", "to delete"])
    h = stored["content_hash"]
    data = _invoke(cli_env, ["delete", h])
    assert data["deleted"] == 1


def test_delete_partial_hash(cli_env):
    stored = _invoke(cli_env, ["store", "partial delete test"])
    h = stored["content_hash"]
    data = _invoke(cli_env, ["delete", h[:10]])
    assert data["deleted"] == 1


def test_delete_dry_run(cli_env):
    stored = _invoke(cli_env, ["store", "dry run delete"])
    h = stored["content_hash"]
    data = _invoke(cli_env, ["delete", h, "--dry-run"])
    assert data["dry_run"] is True
    assert data["would_delete"] == 1


def test_delete_not_found(cli_env):
    data = _invoke(cli_env, ["delete", "nonexistent_000"])
    assert data["deleted"] == 0


# --- update ---


def test_update_content(cli_env):
    stored = _invoke(cli_env, ["store", "original content"])
    old_hash = stored["content_hash"]
    data = _invoke(cli_env, ["update", old_hash, "--content", "updated content"])
    assert data["status"] == "updated"
    assert data["content_hash"] != old_hash


def test_update_no_args_error(cli_env):
    stored = _invoke(cli_env, ["store", "update target"])
    h = stored["content_hash"]
    with pytest.raises(SystemExit):
        _invoke(cli_env, ["update", h])


# --- health ---


def test_health(cli_env):
    _invoke(cli_env, ["store", "health check item"])
    data = _invoke(cli_env, ["health"])
    assert data["status"] == "healthy"
    assert "total_memories" in data
