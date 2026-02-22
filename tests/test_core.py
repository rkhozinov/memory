"""Tests for MemoryStore core operations."""

import time

import pytest

from memory.core import MemoryStore, _parse_time_expr


# --- Original tests (fixture now comes from conftest.py) ---


def test_store_and_search(store):
    result = store.store("Terraform uses HCL for configuration", tags=["terraform"])
    assert result["status"] == "stored"
    assert result["content_hash"]

    results = store.search("terraform configuration", limit=5)
    assert len(results) >= 1
    assert "Terraform" in results[0]["content"]


def test_store_duplicate(store):
    store.store("duplicate content")
    result = store.store("duplicate content")
    assert result["status"] == "duplicate"


def test_store_batch(store):
    items = [
        {"content": "batch item 1", "tags": ["test"]},
        {"content": "batch item 2", "tags": ["test"]},
        {"content": "batch item 3"},
    ]
    results = store.store_batch(items)
    assert len(results) == 3
    assert all(r["status"] == "stored" for r in results)


def test_list(store):
    store.store("list test 1", tags=["a"])
    store.store("list test 2", tags=["b"])
    store.store("list test 3", tags=["a"])

    result = store.list(page=1, page_size=10)
    assert result["total"] == 3
    assert len(result["memories"]) == 3

    result = store.list(page=1, page_size=10, tags=["a"])
    assert len(result["memories"]) == 2


def test_delete_by_hash(store):
    result = store.store("to be deleted")
    h = result["content_hash"]

    # Dry run
    dr = store.delete(content_hash=h, dry_run=True)
    assert dr["dry_run"] is True
    assert dr["would_delete"] == 1

    # Actual delete
    d = store.delete(content_hash=h)
    assert d["deleted"] == 1

    # Should be gone from search
    results = store.search("to be deleted", mode="exact")
    assert len(results) == 0


def test_delete_by_tags(store):
    store.store("temp 1", tags=["temporary"])
    store.store("temp 2", tags=["temporary"])
    store.store("keep this", tags=["important"])

    result = store.delete(tags=["temporary"])
    assert result["deleted"] == 2

    listing = store.list()
    assert listing["total"] == 1


def test_update(store):
    result = store.store("update me", tags=["old"], memory_type="note")
    h = result["content_hash"]

    store.update(h, updates={"tags": ["new", "updated"], "memory_type": "fact"})

    listing = store.list()
    mem = listing["memories"][0]
    assert "new" in mem["tags"]
    assert mem["memory_type"] == "fact"


def test_health(store):
    store.store("health check")
    h = store.health()
    assert h["status"] == "healthy"
    assert h["total_memories"] == 1


def test_cleanup_no_dupes(store):
    store.store("unique content")
    result = store.cleanup()
    assert result["duplicates_removed"] == 0


def test_exact_search(store):
    store.store("Python asyncio patterns")
    store.store("JavaScript promises")

    results = store.search("asyncio", mode="exact")
    assert len(results) == 1
    assert "asyncio" in results[0]["content"]


def test_delete_no_filter(store):
    result = store.delete()
    assert "error" in result


def test_search_with_time(store):
    store.store("recent memory")
    results = store.search("recent", time_expr="today")
    assert len(results) >= 1


# --- New tests ---


def test_store_dedup_threshold(store):
    """Similarity-based dedup returns 'duplicate' with similar_hash."""
    store.store("Kubernetes pods run containers")
    # Very similar content with low threshold should trigger dedup
    result = store.store(
        "Kubernetes pods run containers in a cluster",
        dedup_threshold=0.5,
    )
    assert result["status"] == "duplicate"
    assert "similar_hash" in result


def test_store_dedup_below_threshold(store):
    """Content below dedup threshold is stored normally."""
    store.store("Terraform uses HCL for infrastructure")
    # Completely different content should not trigger dedup even with low threshold
    result = store.store(
        "Python asyncio enables concurrent programming",
        dedup_threshold=0.95,
    )
    assert result["status"] == "stored"


def test_search_recall_tracking(store):
    """Search increments recall_count and last_recalled_at."""
    store.store("recall tracking test content")
    store.search("recall tracking", mode="exact")

    conn = store._get_conn()
    row = conn.execute(
        "SELECT recall_count, last_recalled_at FROM memories WHERE deleted_at IS NULL"
    ).fetchone()
    assert row["recall_count"] == 1
    assert row["last_recalled_at"] is not None


def test_stats_basic(store):
    """stats() returns expected keys and counts."""
    store.store("stats test 1")
    store.store("stats test 2")
    result = store.stats()
    assert result["total_memories"] == 2
    assert "stores_total" in result
    assert "searches_total" in result
    assert "never_recalled" in result


def test_stats_top_recalled(store):
    """top_recalled parameter returns ranked list."""
    store.store("frequently recalled")
    store.search("frequently recalled", mode="exact")
    store.search("frequently recalled", mode="exact")

    result = store.stats(top_recalled=5)
    assert "top_recalled" in result
    assert len(result["top_recalled"]) >= 1
    assert result["top_recalled"][0]["recall_count"] >= 2


def test_stats_never_recalled(store):
    """never_recalled=True lists unrecalled memories."""
    store.store("never searched for")
    result = store.stats(never_recalled=True)
    assert "never_recalled_list" in result
    assert len(result["never_recalled_list"]) >= 1


def test_stats_stale(store):
    """stale=True lists oldest unrecalled."""
    store.store("old stale memory")
    result = store.stats(stale=True)
    assert "stale_memories" in result
    assert len(result["stale_memories"]) >= 1


def test_delete_by_hash_prefix(store):
    """Short hash prefix resolves to single match and deletes."""
    result = store.store("prefix delete test")
    full_hash = result["content_hash"]
    prefix = full_hash[:8]

    d = store.delete(content_hash=prefix)
    assert d["deleted"] == 1


def test_delete_ambiguous_prefix(store):
    """Ambiguous prefix returns error (hard to trigger naturally, test the path)."""
    # Store two items - we need two different hashes that share a prefix.
    # Instead, test the code path directly by inserting rows with crafted hashes.
    conn = store._get_conn()
    now = time.time()
    for suffix in ("aaa", "aab"):
        conn.execute(
            "INSERT INTO memories (content_hash, content, tags, memory_type, metadata, created_at, updated_at) "
            "VALUES (?, ?, '[]', 'note', '{}', ?, ?)",
            (f"deadbeef{suffix}", f"content {suffix}", now, now),
        )
    result = store.delete(content_hash="deadbeef")
    assert "error" in result
    assert "Ambiguous" in result["error"]


def test_update_hash_prefix(store):
    """Update via short hash prefix works."""
    result = store.store("prefix update test")
    prefix = result["content_hash"][:8]
    updated = store.update(prefix, updates={"memory_type": "fact"})
    assert updated["status"] == "updated"


def test_update_not_found(store):
    """Update on missing hash returns error."""
    result = store.update("nonexistent_hash_000", updates={"memory_type": "fact"})
    assert "error" in result


def test_update_no_changes(store):
    """Empty updates dict returns error."""
    result = store.store("no change test")
    h = result["content_hash"]
    updated = store.update(h, updates={})
    assert "error" in updated


def test_search_hybrid_mode(store):
    """mode='hybrid' works as alias for semantic."""
    store.store("hybrid search test content")
    results = store.search("hybrid search", mode="hybrid")
    assert isinstance(results, list)


def test_list_by_memory_type(store):
    """memory_type filter works in list()."""
    store.store("note content", memory_type="note")
    store.store("fact content", memory_type="fact")

    notes = store.list(memory_type="note")
    assert all(m["memory_type"] == "note" for m in notes["memories"])

    facts = store.list(memory_type="fact")
    assert all(m["memory_type"] == "fact" for m in facts["memories"])


# --- Time expression parser tests ---


def test_parse_time_expr_named():
    """Named expressions: today, yesterday, last week."""
    for expr in ("today", "yesterday", "last week", "last month", "last year"):
        dt = _parse_time_expr(expr)
        assert dt is not None
        assert dt.hour == 0 and dt.minute == 0 and dt.second == 0


def test_parse_time_expr_relative():
    """Relative expressions: '3 days ago', '2 weeks ago'."""
    dt = _parse_time_expr("3 days ago")
    assert dt is not None
    dt2 = _parse_time_expr("2 weeks ago")
    assert dt2 < dt  # 2 weeks ago is further in the past


def test_parse_time_expr_invalid():
    """Invalid expression raises ValueError."""
    with pytest.raises(ValueError, match="Cannot parse"):
        _parse_time_expr("not a real time")


# --- Rollback safety ---


def test_rollback_safe_no_transaction(store):
    """_rollback_safe doesn't raise when no transaction is active."""
    conn = store._get_conn()
    # Should not raise even though no transaction is open
    MemoryStore._rollback_safe(conn)
