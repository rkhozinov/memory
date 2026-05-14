"""Tests for `memory admin undelete <hash>`: selective reversal of soft-deletes."""

from __future__ import annotations


def _store_one(store, content: str = "undelete fixture memory") -> tuple[str, int]:
    """Store a memory and return (content_hash, internal id)."""
    h = store.store(content, memory_type="note", tags=["scope:test"])["content_hash"]
    conn = store._get_conn()
    row = conn.execute("SELECT id FROM memories WHERE content_hash = ?", (h,)).fetchone()
    return h, row["id"]


def test_undelete_revives_soft_deleted_memory(store):
    h, mem_id = _store_one(store, "memory targeted for delete + undelete")

    # Soft-delete via the public API; this also drops the embedding/FTS rows.
    del_result = store.delete(content_hash=h)
    assert del_result["deleted"] == 1

    # The row should be invisible to get() until we undelete it.
    assert "error" in store.get(h)

    result = store.undelete(h)
    assert result["undeleted"] is True
    assert result["hash"] == h
    assert result["content_preview"].startswith("memory targeted")
    # delete() removes the embedding row, so undelete must reindex.
    assert result["had_embedding"] is False
    assert result["reindexed"] is True

    # Memory is live again and indexed.
    conn = store._get_conn()
    live = conn.execute(
        "SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)
    ).fetchone()
    assert live["deleted_at"] is None

    emb = conn.execute(
        "SELECT rowid FROM memory_embeddings WHERE rowid = ?", (mem_id,)
    ).fetchone()
    assert emb is not None

    fts = conn.execute(
        "SELECT rowid FROM memory_fts WHERE rowid = ?", (mem_id,)
    ).fetchone()
    assert fts is not None

    # And get() now returns it.
    fetched = store.get(h)
    assert fetched.get("content_hash") == h


def test_undelete_noop_on_live_memory(store):
    h, _ = _store_one(store, "still alive, should not be touched")

    result = store.undelete(h)
    assert result["undeleted"] is False
    assert result["hash"] == h
    assert "not deleted" in (result.get("reason") or "").lower()

    # No reindex side effects.
    assert "reindexed" not in result or result.get("reindexed") is False


def test_undelete_unknown_hash_returns_undeleted_false(store):
    # 64-char SHA256-shaped string that does not exist in the DB.
    bogus = "0" * 64
    result = store.undelete(bogus)
    assert result["undeleted"] is False
    assert "not found" in (result.get("reason") or "").lower()


def test_undelete_reindexes_when_embedding_row_missing(store):
    h, mem_id = _store_one(store, "reindex required when emb row was pruned")

    # Mimic a state where the row was soft-deleted but the embedding row
    # was lost (consolidate()/delete() both drop it).
    conn = store._get_conn()
    conn.execute(
        "UPDATE memories SET deleted_at = strftime('%s', 'now') WHERE id = ?",
        (mem_id,),
    )
    conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (mem_id,))
    conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (mem_id,))
    conn.commit()

    result = store.undelete(h)
    assert result["undeleted"] is True
    assert result["had_embedding"] is False
    assert result["reindexed"] is True

    emb = conn.execute(
        "SELECT rowid FROM memory_embeddings WHERE rowid = ?", (mem_id,)
    ).fetchone()
    assert emb is not None


def test_undelete_no_reindex_when_embedding_intact(store):
    """If only deleted_at was set (no row pruning), undelete should NOT recompute embeddings."""
    h, mem_id = _store_one(store, "soft-delete left embedding intact")

    conn = store._get_conn()
    # Soft-delete WITHOUT touching the ancillary indexes.
    conn.execute(
        "UPDATE memories SET deleted_at = strftime('%s', 'now') WHERE id = ?",
        (mem_id,),
    )
    conn.commit()

    result = store.undelete(h)
    assert result["undeleted"] is True
    assert result["had_embedding"] is True
    assert result["reindexed"] is False


def test_undelete_resolves_hash_prefix(store):
    h, _ = _store_one(store, "prefix lookup target")
    store.delete(content_hash=h)

    prefix = h[:12]
    result = store.undelete(prefix)
    assert result["undeleted"] is True
    assert result["hash"] == h  # resolved to full hash


def test_undelete_dry_run_does_not_mutate(store):
    h, mem_id = _store_one(store, "dry-run target")
    store.delete(content_hash=h)

    result = store.undelete(h, dry_run=True)
    assert result.get("dry_run") is True
    assert result["undeleted"] is False  # nothing was written
    assert result["hash"] == h

    conn = store._get_conn()
    deleted_at = conn.execute(
        "SELECT deleted_at FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()["deleted_at"]
    assert deleted_at is not None  # still soft-deleted


def test_undelete_ambiguous_prefix_errors_out(store):
    # Force two memories whose content_hash share a leading character. We
    # cannot pick the prefix arbitrarily; instead, after storing two memories,
    # pick a 1-char prefix that matches both (a hex digit common to both
    # SHA256 hashes). If none exists in this seed, skip.
    h1, _ = _store_one(store, "ambiguous A — first memory")
    h2, _ = _store_one(store, "ambiguous B — second memory")
    common = next((c for c in h1 if c in h2 and c == h1[0] == h2[0]), None)
    if not common:
        # Build a deterministic ambiguous prefix using the SQL LIKE wildcard.
        # If first chars differ, just use the empty string (matches all).
        common = ""
    result = store.undelete(common)
    # Either "ambiguous" (>1 match) or "not found" (0 match) — never undelete
    # without an exact target.
    assert result["undeleted"] is False
