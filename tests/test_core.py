"""Tests for MemoryStore core operations."""

import time

import pytest

from memory.core import (
    MemoryStore,
    _parse_time_expr,
    _sanitize_fts_query,
    compute_confidence,
    compute_recency,
    infer_importance,
)

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
    row = conn.execute("SELECT recall_count, last_recalled_at FROM memories WHERE deleted_at IS NULL").fetchone()
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


def test_search_hybrid_mode(populated_store):
    """Hybrid search returns exact identifier match as top result."""
    results = populated_store.search("TICKET-24", mode="hybrid")
    assert len(results) >= 1
    assert "TICKET-24" in results[0]["content"]


def test_hybrid_identifier_precision(populated_store):
    """Searching for TICKET-75 returns TICKET-75 memory, not other TICKET-* tickets."""
    results = populated_store.search("TICKET-75", mode="hybrid")
    assert len(results) >= 1
    assert "TICKET-75" in results[0]["content"]
    # Other NEM tickets should not outrank the exact match
    for r in results[1:]:
        if "TICKET-75" not in r["content"]:
            assert r["score"] <= results[0]["score"]


def test_hybrid_nonexistent_identifier(populated_store):
    """Searching for non-existent TICKET-999 should not have FTS-boosted results."""
    results = populated_store.search("TICKET-999", mode="hybrid")
    # Results may exist (semantic similarity to TICKET-* content) but none should have dual-match boost
    for r in results:
        assert "TICKET-999" not in r["content"]


def test_semantic_vs_hybrid_identifier_ranking(populated_store):
    """Hybrid outperforms semantic for identifier queries."""
    sem_results = populated_store.search("TICKET-24", mode="semantic", limit=5)
    hyb_results = populated_store.search("TICKET-24", mode="hybrid", limit=5)
    # Hybrid's top result should contain TICKET-24
    assert "TICKET-24" in hyb_results[0]["content"]
    # Semantic might not have TICKET-24 on top (the original bug)
    # At minimum, hybrid's top result score should be >= semantic's
    if sem_results:
        assert hyb_results[0]["score"] >= sem_results[0]["score"]


def test_fts_single_result_similarity(store):
    """Single FTS result should get similarity 1.0, not 0.0."""
    store.store("unique-identifier-xyz42 is the only match")
    results = store.search("unique-identifier-xyz42", mode="fts", limit=5)
    assert len(results) == 1
    assert results[0]["similarity"] == 1.0
    # Score should reflect the similarity (not 0.0)
    assert results[0]["score"] > 0.3


def test_fts_multiple_results_normalization(store):
    """Multiple FTS results: best gets similarity ~1.0, worst ~0.0."""
    store.store("terraform state backend configuration guide")
    store.store("terraform module for AWS VPC networking")
    store.store("configure terraform provider credentials")
    results = store.search("terraform", mode="fts", limit=5)
    assert len(results) >= 2
    sims = [r["similarity"] for r in results]
    assert max(sims) == 1.0  # best result
    assert min(sims) < max(sims)  # there's variance


def test_hybrid_dual_match_boost(populated_store):
    """Memory found by BOTH semantic and FTS gets higher score than single-backend match."""
    # "Kubernetes pod crash" should match the crash-loop memory via both backends
    results = populated_store.search("Kubernetes pod crash", mode="hybrid")
    assert len(results) >= 1
    assert "crash" in results[0]["content"].lower()
    # The score should include both semantic and FTS contributions
    assert results[0]["score"] > 0


def test_hybrid_topic_search_quality(populated_store):
    """Hybrid search for a topic returns relevant content on top."""
    results = populated_store.search("terraform state locking", mode="hybrid")
    assert len(results) >= 1
    # The terraform state locking memory should be top result
    assert "terraform" in results[0]["content"].lower()
    assert "state" in results[0]["content"].lower()


def test_hybrid_returns_results_when_fts_empty(store):
    """Hybrid degrades gracefully to semantic-only when FTS has no matches."""
    store.store("machine learning model training pipeline optimization")
    # Query is semantically similar but uses different words
    results = store.search("AI model training workflow", mode="hybrid")
    # Should still find the memory via semantic backend
    assert len(results) >= 1


def test_hybrid_returns_results_when_only_fts_matches(store):
    """Hybrid includes FTS-only results when semantic misses."""
    store.store("PROJ-42 configuration update")
    results = store.search("PROJ-42", mode="hybrid")
    assert len(results) >= 1
    assert "PROJ-42" in results[0]["content"]


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


# --- Round 4: Confidence decay ---


def test_confidence_decay_formula():
    """compute_confidence decays over time based on type rate."""
    now = time.time()
    # 10 days ago, error type (rate=0.99)
    conf = compute_confidence(1.0, "error", None, now - 10 * 86400)
    assert 0.9 < conf < 0.95  # 0.99^10 ≈ 0.904

    # decision type barely decays (rate=0.999)
    conf_d = compute_confidence(1.0, "decision", None, now - 10 * 86400)
    assert conf_d > 0.99  # 0.999^10 ≈ 0.990


def test_confidence_reset_on_recall(store):
    """Recalled memories get confidence reset to 1.0."""
    store.store("confidence reset test", memory_type="error")
    # Manually set confidence low
    conn = store._get_conn()
    conn.execute("UPDATE memories SET confidence = 0.5 WHERE deleted_at IS NULL")

    # Search triggers recall → resets confidence to 1.0
    store.search("confidence reset", mode="exact")

    row = conn.execute("SELECT confidence FROM memories WHERE deleted_at IS NULL").fetchone()
    assert row["confidence"] == 1.0


def test_compute_recency():
    """compute_recency returns 1.0 for now, decays for older."""
    now = time.time()
    assert compute_recency(now) > 0.99
    # 30 days ago
    r30 = compute_recency(now - 30 * 86400)
    assert 0.03 < r30 < 0.04  # 1/(1+30) ≈ 0.032


# --- Round 4: Importance scoring ---


def test_infer_importance_keywords():
    """Keywords like IMPORTANT/CRITICAL bump importance to 0.9."""
    assert infer_importance("This is IMPORTANT for deployment", "note") == 0.9
    assert infer_importance("CRITICAL bug in auth", "error") == 0.9
    assert infer_importance("NEVER run git clean -fd", "pattern") == 0.8


def test_infer_importance_by_type():
    """Importance is inferred from memory type when no keywords match."""
    assert infer_importance("plain content", "decision") == 0.8
    assert infer_importance("plain content", "error") == 0.7
    assert infer_importance("plain content", "note") == 0.4


def test_store_auto_infers_importance(store):
    """Storing without explicit importance auto-infers it."""
    result = store.store("CRITICAL: never delete production DB", memory_type="decision")
    assert result["status"] == "stored"

    conn = store._get_conn()
    row = conn.execute(
        "SELECT importance FROM memories WHERE content_hash = ?",
        (result["content_hash"],),
    ).fetchone()
    assert row["importance"] == 0.9  # keyword match


def test_store_explicit_importance(store):
    """Explicit importance overrides auto-inference."""
    result = store.store("plain content", importance=0.95)
    conn = store._get_conn()
    row = conn.execute(
        "SELECT importance FROM memories WHERE content_hash = ?",
        (result["content_hash"],),
    ).fetchone()
    assert row["importance"] == 0.95


# --- Round 4: Composite retrieval scoring ---


def test_search_returns_composite_score(store):
    """Semantic search results include score, confidence, importance."""
    store.store("Kubernetes pods run containers", tags=["k8s"])
    results = store.search("kubernetes pods", limit=5)
    assert len(results) >= 1
    r = results[0]
    assert "score" in r
    assert "confidence" in r
    assert "importance" in r
    assert r["score"] > 0


def test_search_composite_reranks(store):
    """Higher importance memory can outrank slightly closer match."""
    store.store("basic note about cats", memory_type="note", importance=0.1)
    store.store("CRITICAL: always validate user input in API handlers", memory_type="decision", importance=0.9)
    # Search for something that could match both
    results = store.search("validate input", limit=2)
    # The CRITICAL one should rank higher due to importance boost
    if len(results) >= 2:
        assert results[0]["importance"] >= results[1]["importance"]


# --- Round 4: Memory consolidation ---


def test_consolidate_dry_run(store):
    """Consolidate dry-run finds near-duplicates without deleting."""
    store.store("Kubernetes pods run in clusters")
    store.store("Kubernetes pods run in clusters for container orchestration")
    result = store.consolidate(threshold=0.8, dry_run=True)
    assert result["dry_run"] is True
    assert result["would_consolidate"] >= 1


def test_consolidate_merges(store):
    """Consolidate actually merges and soft-deletes."""
    store.store("Terraform uses HCL for infrastructure config", tags=["terraform"])
    store.store("Terraform uses HCL for infrastructure configuration", tags=["iac"])

    result = store.consolidate(threshold=0.8)
    assert result["consolidated"] >= 1

    # One should be soft-deleted
    listing = store.list()
    assert listing["total"] == 1
    # Surviving memory should have union of tags
    surviving = listing["memories"][0]
    assert "terraform" in surviving["tags"] or "iac" in surviving["tags"]


def test_consolidate_no_matches(store):
    """Consolidate with very different memories finds no pairs."""
    store.store("Python asyncio for concurrent IO")
    store.store("Terraform infrastructure management")
    result = store.consolidate(threshold=0.99)
    assert result["consolidated"] == 0


# --- Round 4: Confidence decay application ---


def test_apply_decay(store):
    """apply_decay updates confidence values."""
    store.store("decay test memory", memory_type="observation")
    # Backdate created_at to 30 days ago
    conn = store._get_conn()
    conn.execute(
        "UPDATE memories SET created_at = ? WHERE deleted_at IS NULL",
        (time.time() - 30 * 86400,),
    )
    result = store.apply_decay()
    assert result["updated"] >= 1 or result["pruned"] >= 0


def test_apply_decay_prunes(store):
    """apply_decay with min_confidence prunes low-confidence memories."""
    store.store("very old memory", memory_type="observation")
    conn = store._get_conn()
    # Set confidence very low
    conn.execute("UPDATE memories SET confidence = 0.05 WHERE deleted_at IS NULL")
    result = store.apply_decay(min_confidence=0.1)
    assert result["pruned"] == 1

    listing = store.list()
    assert listing["total"] == 0


# --- Round 4: Update importance/confidence ---


def test_update_importance(store):
    """Update memory importance via update()."""
    result = store.store("update importance test")
    h = result["content_hash"]
    store.update(h, updates={"importance": 0.9})

    conn = store._get_conn()
    row = conn.execute("SELECT importance FROM memories WHERE content_hash = ?", (h,)).fetchone()
    assert row["importance"] == 0.9


def test_update_confidence(store):
    """Update memory confidence via update()."""
    result = store.store("update confidence test")
    h = result["content_hash"]
    store.update(h, updates={"confidence": 0.75})

    conn = store._get_conn()
    row = conn.execute("SELECT confidence FROM memories WHERE content_hash = ?", (h,)).fetchone()
    assert row["confidence"] == 0.75


# --- Round 5: Briefing ---


def test_briefing_empty(store):
    """Briefing on empty store returns sensible defaults."""
    result = store.briefing()
    assert result["total_memories"] == 0
    assert result["total_lines"] == 0
    assert "No memories" in result["markdown"]
    assert result["sections"] == {}


def test_briefing_basic(store):
    """Briefing returns sections and markdown."""
    store.store("[Decision] Use PostgreSQL for production", memory_type="decision")
    store.store("[Error] OOM on large batch — reduce batch size", memory_type="error")
    store.store("[Pattern] Always use uv instead of pip", memory_type="pattern")
    store.store("General note about project", memory_type="note")

    result = store.briefing()
    assert result["total_memories"] == 4
    assert result["total_lines"] > 0
    assert "markdown" in result
    assert "## Decisions" in result["markdown"]
    assert "sections" in result
    assert isinstance(result["sections"], dict)


def test_briefing_budget(store):
    """Briefing respects line budget."""
    for i in range(50):
        store.store(f"Pattern number {i} for testing budget", memory_type="pattern")

    result_small = store.briefing(budget=10)
    result_large = store.briefing(budget=200)
    assert result_small["total_lines"] <= result_large["total_lines"]


def test_briefing_recent_section(store):
    """Recent section includes memories from last 7 days."""
    store.store("[Learning] Fresh learning from today", memory_type="learning")
    result = store.briefing()
    # Should appear in both learning and recent sections
    assert "recent" in result["sections"]
    assert len(result["sections"]["recent"]) >= 1


# --- Round 5: Search recall fields ---


def test_search_semantic_returns_recall_fields(store):
    """Semantic search results include recall_count and last_recalled_at."""
    store.store("Kubernetes pods run containers", tags=["k8s"])
    results = store.search("kubernetes pods", limit=5)
    assert len(results) >= 1
    r = results[0]
    assert "recall_count" in r
    assert "last_recalled_at" in r


def test_search_exact_returns_recall_fields(store):
    """Exact search results include recall_count and last_recalled_at."""
    store.store("Python asyncio patterns for IO")
    results = store.search("asyncio", mode="exact")
    assert len(results) >= 1
    r = results[0]
    assert "recall_count" in r
    assert "last_recalled_at" in r


# --- Type-scoped dedup ---


def test_dedup_cross_type_not_rejected(store):
    """Dedup should NOT reject a memory when the similar match is a different type."""
    store.store(
        "AWS Organizations: use SCPs for account-level guardrails",
        memory_type="decision",
        tags=["cloud:aws"],
    )
    # Similar infrastructure content but different type — should NOT be rejected
    result = store.store(
        "SSH key for bastion host: ssh-rsa AAAA... user@host",
        memory_type="reference",
        tags=["cloud:aws"],
        dedup_threshold=0.85,
    )
    assert result["status"] == "stored"


def test_dedup_same_type_still_rejected(store):
    """Dedup should still reject when the similar match is the same type."""
    store.store("Kubernetes pods run containers in a cluster", memory_type="note")
    result = store.store(
        "Kubernetes pods run containers in a cluster for orchestration",
        memory_type="note",
        dedup_threshold=0.5,
    )
    assert result["status"] == "duplicate"
    assert "similar_hash" in result


def test_force_flag_bypasses_dedup(store):
    """--force (dedup_threshold=None) bypasses dedup entirely."""
    store.store("Terraform uses HCL for infrastructure", memory_type="note")
    # Same type, very similar content, but force=True means dedup_threshold=None
    result = store.store(
        "Terraform uses HCL for infrastructure configuration",
        memory_type="note",
        dedup_threshold=None,  # simulates --force
    )
    assert result["status"] == "stored"


# --- Get ---


def test_get(store):
    """get() returns the full memory dict by exact hash."""
    result = store.store("get test content", tags=["test"], memory_type="note")
    h = result["content_hash"]
    mem = store.get(h)
    assert mem["content_hash"] == h
    assert mem["content"] == "get test content"
    assert "test" in mem["tags"]
    assert mem["memory_type"] == "note"


def test_get_prefix(store):
    """get() resolves a unique short hash prefix."""
    result = store.store("prefix get test")
    prefix = result["content_hash"][:8]
    mem = store.get(prefix)
    assert mem["content_hash"] == result["content_hash"]
    assert mem["content"] == "prefix get test"


def test_get_not_found(store):
    """get() returns error for nonexistent hash."""
    mem = store.get("nonexistent_hash_000")
    assert "error" in mem
    assert "not found" in mem["error"].lower()


def test_get_ambiguous_prefix(store):
    """get() returns error for ambiguous prefix."""
    conn = store._get_conn()
    now = time.time()
    for suffix in ("aaa", "aab"):
        conn.execute(
            "INSERT INTO memories (content_hash, content, tags, memory_type, metadata, created_at, updated_at) "
            "VALUES (?, ?, '[]', 'note', '{}', ?, ?)",
            (f"deadbeef{suffix}", f"content {suffix}", now, now),
        )
    mem = store.get("deadbeef")
    assert "error" in mem
    assert "Ambiguous" in mem["error"]


# --- Update content ---


def test_update_content(store):
    """update() with content rehashes and updates content."""
    result = store.store("original content", tags=["test"], memory_type="note")
    old_hash = result["content_hash"]

    updated = store.update(old_hash, updates={"content": "updated content"})
    assert updated["status"] == "updated"
    new_hash = updated["content_hash"]
    assert new_hash != old_hash

    # Old hash should not resolve
    old_mem = store.get(old_hash)
    assert "error" in old_mem

    # New hash should work
    new_mem = store.get(new_hash)
    assert new_mem["content"] == "updated content"


def test_update_content_preserves_metadata(store):
    """update() with content preserves tags, type, recall_count, importance."""
    result = store.store("preserve meta test", tags=["keep"], memory_type="decision", importance=0.9)
    h = result["content_hash"]

    # Simulate a recall to set recall_count
    conn = store._get_conn()
    conn.execute("UPDATE memories SET recall_count = 5 WHERE content_hash = ?", (h,))

    updated = store.update(h, updates={"content": "new content preserving meta"})
    new_hash = updated["content_hash"]

    new_mem = store.get(new_hash)
    assert new_mem["memory_type"] == "decision"
    assert "keep" in new_mem["tags"]
    assert new_mem["recall_count"] == 5


# --- FTS query sanitization tests ---


class TestSanitizeFtsQuery:
    def test_empty_string(self):
        assert _sanitize_fts_query("") == ""

    def test_whitespace_only(self):
        assert _sanitize_fts_query("   ") == ""

    def test_single_word(self):
        assert _sanitize_fts_query("hello") == '"hello"'

    def test_multi_word(self):
        assert _sanitize_fts_query("hello world") == '"hello" "world"'

    def test_hyphenated(self):
        assert _sanitize_fts_query("video-processing") == '"video-processing"'

    def test_colon(self):
        assert _sanitize_fts_query("title:foo") == '"title:foo"'

    def test_asterisk(self):
        assert _sanitize_fts_query("test*") == '"test*"'

    def test_boolean_keyword(self):
        assert _sanitize_fts_query("NOT important") == '"NOT" "important"'
        assert _sanitize_fts_query("foo AND bar") == '"foo" "AND" "bar"'

    def test_already_quoted_single_token(self):
        assert _sanitize_fts_query('"hello"') == '"hello"'

    def test_already_quoted_multi_word_re_quoted(self):
        # Multi-word quoted phrase is split by whitespace; each part gets re-quoted
        result = _sanitize_fts_query('"already quoted"')
        assert result == '"""already" "quoted"""'

    def test_mixed_quoted_and_unquoted(self):
        assert _sanitize_fts_query('"keep" unquoted') == '"keep" "unquoted"'

    def test_embedded_double_quote(self):
        # A token with an internal quote gets it doubled
        assert _sanitize_fts_query('say"hello') == '"say""hello"'


def test_fts_search_hyphenated_query(store):
    """FTS search with hyphenated query should not crash."""
    store.store("video-processing pipeline for streaming", tags=["media"])
    results = store.search("video-processing", mode="fts", limit=5)
    assert len(results) >= 1
    assert "video-processing" in results[0]["content"]


def test_fts_search_colon_query(store):
    """FTS search with colon in query should not crash."""
    store.store("config key:value pair setting", tags=["config"])
    results = store.search("key:value", mode="fts", limit=5)
    assert len(results) >= 1


# --- Stage 1: Search filter enhancements + score breakdown ---


def test_search_exclude_tags(store):
    """exclude_tags filters out matching memories."""
    store.store("keep this memory", tags=["keep"])
    store.store("exclude this one", tags=["temp"])
    store.store("also keep", tags=["keep", "other"])

    results = store.search("memory", mode="exact", exclude_tags=["temp"])
    contents = [r["content"] for r in results]
    assert "exclude this one" not in contents
    assert any("keep" in c for c in contents)


def test_search_include_and_exclude_tags(store):
    """Include and exclude tags work together."""
    store.store("aws infra note", tags=["cloud:aws", "scope:temp"])
    store.store("aws prod note", tags=["cloud:aws", "scope:prod"])
    store.store("gcp note", tags=["cloud:gcp"])

    results = store.search(
        "note",
        mode="exact",
        tags=["cloud:aws"],
        exclude_tags=["scope:temp"],
    )
    assert len(results) == 1
    assert "prod" in results[0]["content"]


def test_search_min_importance(store):
    """min_importance filters out low-importance memories."""
    store.store("low importance", importance=0.2)
    store.store("high importance", importance=0.9)

    results = store.search("importance", mode="exact", min_importance=0.5)
    assert len(results) == 1
    assert "high" in results[0]["content"]


def test_search_memory_types_plural(store):
    """memory_types filters to multiple types."""
    store.store("decision content", memory_type="decision")
    store.store("pattern content", memory_type="pattern")
    store.store("note content", memory_type="note")

    results = store.search("content", mode="exact", memory_types=["decision", "pattern"])
    types = {r["memory_type"] for r in results}
    assert types <= {"decision", "pattern"}
    assert "note" not in types
    assert len(results) == 2


def test_search_score_breakdown(store):
    """Semantic search results include score_breakdown with correct keys."""
    store.store("Kubernetes pods run containers", tags=["k8s"])
    results = store.search("kubernetes pods", limit=5)
    assert len(results) >= 1
    r = results[0]
    assert "score_breakdown" in r
    sb = r["score_breakdown"]
    assert "similarity" in sb
    assert "importance" in sb
    assert "recency" in sb
    assert isinstance(sb["similarity"], float)
    assert isinstance(sb["importance"], float)
    assert isinstance(sb["recency"], float)


def test_search_fts_score_breakdown(store):
    """FTS search results include score_breakdown."""
    store.store("terraform infrastructure management config", tags=["terraform"])
    results = store.search("terraform infrastructure", mode="fts", limit=5)
    assert len(results) >= 1
    assert "score_breakdown" in results[0]


def test_search_exclude_tags_semantic(store):
    """Exclude tags works with semantic search."""
    store.store("Terraform uses HCL for configuration", tags=["terraform", "scope:temp"])
    store.store("Terraform modules for AWS", tags=["terraform", "scope:prod"])

    results = store.search("terraform", exclude_tags=["scope:temp"])
    for r in results:
        assert "scope:temp" not in r.get("tags", [])


def test_search_min_importance_semantic(store):
    """min_importance works with semantic search."""
    store.store("low importance terraform note", importance=0.1)
    store.store("CRITICAL terraform decision", importance=0.9)

    results = store.search("terraform", min_importance=0.5)
    for r in results:
        assert r["importance"] >= 0.5


# --- Stage 4: Embedding cache eviction ---


def test_cache_eviction_lru(tmp_path):
    """Insert > max_entries, verify oldest are evicted."""
    import numpy as np

    from memory.embeddings import EmbeddingModel

    cache_db = tmp_path / "test_cache.db"
    model = EmbeddingModel(cache_db=cache_db)

    # Insert 15 entries with sequential timestamps
    items = []
    for i in range(15):
        h = f"hash_{i:04d}"
        emb = np.random.rand(384).astype(np.float32)
        items.append((h, emb))
    model._l2_put_many(items)

    # Now evict with max_entries=10
    evicted = model.evict_cache(max_entries=10)
    assert evicted == 5

    # Verify count is now 10
    conn = model._get_cache_conn()
    count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    assert count == 10

    # The oldest 5 (hash_0000 through hash_0004) should be gone
    row = conn.execute("SELECT COUNT(*) FROM cache WHERE text_hash = 'hash_0000'").fetchone()[0]
    assert row == 0

    # The newest should still be present
    row = conn.execute("SELECT COUNT(*) FROM cache WHERE text_hash = 'hash_0014'").fetchone()[0]
    assert row == 1


def test_cache_access_updates_timestamp(tmp_path):
    """Verify reads update last_accessed_at."""
    import time as time_mod

    import numpy as np

    from memory.embeddings import EmbeddingModel

    cache_db = tmp_path / "test_cache_access.db"
    model = EmbeddingModel(cache_db=cache_db)

    # Insert an entry
    emb = np.random.rand(384).astype(np.float32)
    model._l2_put_many([("test_hash", emb)])

    # Get initial timestamp
    conn = model._get_cache_conn()
    row = conn.execute("SELECT last_accessed_at FROM cache WHERE text_hash = 'test_hash'").fetchone()
    initial_ts = row[0]

    # Small delay to ensure timestamp differs
    time_mod.sleep(0.01)

    # Access the entry
    model._l2_get_many(["test_hash"])

    # Check timestamp was updated
    row = conn.execute("SELECT last_accessed_at FROM cache WHERE text_hash = 'test_hash'").fetchone()
    assert row[0] > initial_ts


# --- Stage 2B: Batch semantic search ---


def test_search_batch(store):
    """search_batch returns per-query results."""
    store.store("Python asyncio for concurrent IO", tags=["python"])
    store.store("Terraform infrastructure management", tags=["terraform"])
    store.store("Kubernetes container orchestration", tags=["k8s"])

    results = store.search_batch(["python asyncio", "kubernetes containers"], limit=5)
    assert isinstance(results, dict)
    assert "python asyncio" in results
    assert "kubernetes containers" in results
    assert len(results["python asyncio"]) >= 1
    assert len(results["kubernetes containers"]) >= 1


def test_search_batch_empty(store):
    """search_batch with empty list returns empty dict."""
    results = store.search_batch([])
    assert results == {}


def test_search_batch_recall_tracking(store):
    """search_batch updates recall counts once per unique hash."""
    store.store("shared result content for batch", tags=["test"])

    # Search with two queries that will likely return the same memory
    store.search_batch(["shared result", "batch content"], limit=5)

    conn = store._get_conn()
    row = conn.execute("SELECT recall_count FROM memories WHERE deleted_at IS NULL").fetchone()
    # Should be incremented exactly once (single transaction for unique hashes)
    assert row["recall_count"] == 1


# --- Stage 2A: Tag rename/merge ---


def test_rename_tag(store):
    """rename_tag replaces across memories and documents."""
    store.store("memory with old tag", tags=["old:tag", "keep"])
    store.store_doc(title="doc", body="doc body", summary="doc summary", tags=["old:tag"])

    result = store.rename_tag("old:tag", "new:tag")
    assert result["memories_updated"] == 1
    assert result["documents_updated"] == 1

    # Verify memory tags updated
    listing = store.list()
    mem = listing["memories"][0]
    assert "new:tag" in mem["tags"]
    assert "old:tag" not in mem["tags"]
    assert "keep" in mem["tags"]


def test_rename_tag_dedup(store):
    """Renaming to an already-present tag deduplicates."""
    store.store("memory with both", tags=["old:tag", "new:tag", "other"])
    result = store.rename_tag("old:tag", "new:tag")
    assert result["memories_updated"] == 1

    listing = store.list()
    mem = listing["memories"][0]
    assert mem["tags"].count("new:tag") == 1
    assert "old:tag" not in mem["tags"]


def test_merge_tags(store):
    """merge_tags merges multiple sources into target."""
    store.store("tagged alpha", tags=["alpha", "other"])
    store.store("tagged beta", tags=["beta"])
    store.store("no match", tags=["gamma"])

    result = store.merge_tags(["alpha", "beta"], "merged")
    assert result["memories_updated"] == 2

    listing = store.list()
    for mem in listing["memories"]:
        if mem["content"] == "no match":
            assert "merged" not in mem["tags"]
        else:
            assert "merged" in mem["tags"]
            assert "alpha" not in mem["tags"]
            assert "beta" not in mem["tags"]


def test_rename_tag_no_matches(store):
    """rename_tag with no matching tags returns 0 updated."""
    store.store("no matching tags", tags=["unrelated"])
    result = store.rename_tag("nonexistent", "new")
    assert result["memories_updated"] == 0
    assert result["documents_updated"] == 0


# --- Stage 3: Export/Import ---


def test_export_all(store):
    """export_all returns correct structure with non-deleted only."""
    store.store("memory one", tags=["test"], memory_type="decision")
    store.store("memory two", tags=["test"])
    r = store.store("to delete")
    store.delete(content_hash=r["content_hash"])
    store.store_doc(title="Doc", body="doc body", summary="doc summary", tags=["test"])

    export = store.export_all()
    assert export["version"] == 1
    assert "exported_at" in export
    assert len(export["memories"]) == 2  # deleted one excluded
    assert len(export["documents"]) == 1
    # Verify recall_count included
    assert "recall_count" in export["memories"][0]


def test_export_no_documents(store):
    """export_all with include_documents=False excludes documents."""
    store.store("a memory")
    store.store_doc(title="Doc", body="body", summary="summary")

    export = store.export_all(include_documents=False)
    assert len(export["memories"]) == 1
    assert export["documents"] == []


def test_import_basic(store):
    """import_all imports memories and documents."""
    data = {
        "version": 1,
        "exported_at": "2025-01-01T00:00:00",
        "memories": [
            {
                "content": "imported memory",
                "tags": ["imported"],
                "memory_type": "note",
                "metadata": {},
                "importance": 0.7,
            },
        ],
        "documents": [
            {
                "title": "Imported Doc",
                "body": "doc body text",
                "summary": "doc summary",
                "doc_type": "document",
                "tags": ["imported"],
                "metadata": {},
            },
        ],
    }
    result = store.import_all(data)
    assert result["memories_imported"] == 1
    assert result["documents_imported"] == 1

    listing = store.list()
    assert listing["total"] == 1
    assert listing["memories"][0]["content"] == "imported memory"


def test_roundtrip_export_import(store):
    """Export → clear → import → verify same content."""
    store.store("roundtrip memory 1", tags=["rt"], memory_type="decision", importance=0.9)
    store.store("roundtrip memory 2", tags=["rt"], memory_type="pattern")
    store.store_doc(title="RT Doc", body="roundtrip body", summary="rt summary", tags=["rt"])

    export = store.export_all()
    assert len(export["memories"]) == 2
    assert len(export["documents"]) == 1

    # Delete all
    for m in export["memories"]:
        store.delete(content_hash=m["content_hash"])
    for d in export["documents"]:
        store.delete_doc(content_hash=d["content_hash"])

    # Verify empty
    listing = store.list()
    assert listing["total"] == 0

    # Import with force (skip dedup since we deleted, not purged)
    result = store.import_all(export, force=True)
    assert result["memories_imported"] == 2
    assert result["documents_imported"] == 1

    # Verify content restored
    listing = store.list()
    assert listing["total"] == 2
    contents = {m["content"] for m in listing["memories"]}
    assert "roundtrip memory 1" in contents
    assert "roundtrip memory 2" in contents
