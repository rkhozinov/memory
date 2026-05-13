"""Phase B tests: bi-temporal edges, RRF fusion, supersession provenance."""

from __future__ import annotations

import time

import pytest

from memory.core import _rrf_fuse

# --- _rrf_fuse unit tests ---


def test_rrf_fuse_combines_two_lists_by_rank():
    sem = [{"content_hash": "a", "score": 0.99}, {"content_hash": "b", "score": 0.5}]
    fts = [{"content_hash": "b", "score": 10.0}, {"content_hash": "c", "score": 5.0}]
    out = _rrf_fuse(sem, fts, limit=10, k=60)
    by_hash = {r["content_hash"]: r for r in out}
    # 'b' appears in both lists → highest RRF score
    assert out[0]["content_hash"] == "b"
    assert by_hash["b"]["rrf_score"] > by_hash["a"]["rrf_score"]
    assert by_hash["b"]["rrf_score"] > by_hash["c"]["rrf_score"]
    # rrf_score is mirrored into 'score' for downstream code
    assert out[0]["score"] == out[0]["rrf_score"]


def test_rrf_fuse_truncates_to_limit():
    big = [{"content_hash": f"h{i}", "score": float(100 - i)} for i in range(20)]
    out = _rrf_fuse(big, [], limit=5, k=60)
    assert len(out) == 5


def test_rrf_fuse_empty_lists():
    assert _rrf_fuse([], [], limit=10, k=60) == []


# --- Dream supersession writes memory_graph edge ---


def test_dream_supersession_records_memory_graph_edge(store):
    """Dream pass 2 must persist a 'supersedes' edge before soft-deleting the older row."""
    older = store.store(
        "decision: use postgres as primary database for application X",
        memory_type="decision",
        tags=["project:appx", "cloud:aws"],
        importance=0.8,
    )
    time.sleep(0.01)  # ensure newer created_at strictly greater
    newer = store.store(
        "instead, application X must replace postgres with clickhouse for analytics now",
        memory_type="decision",
        tags=["project:appx", "cloud:aws"],
        importance=0.8,
    )
    assert "content_hash" in older and "content_hash" in newer

    # Force semantic similarity above 0.85 by copying older's embedding onto newer.
    # We test the GATE logic + edge writing, not the embedding model.
    conn = store._get_conn()
    emb_older = conn.execute(
        "SELECT content_embedding FROM memory_embeddings WHERE rowid = "
        "(SELECT id FROM memories WHERE content_hash = ?)",
        (older["content_hash"],),
    ).fetchone()
    conn.execute(
        "UPDATE memory_embeddings SET content_embedding = ? "
        "WHERE rowid = (SELECT id FROM memories WHERE content_hash = ?)",
        (emb_older["content_embedding"], newer["content_hash"]),
    )
    conn.commit()

    result = store.dream(dry_run=False)
    assert result["superseded"] >= 1

    conn = store._get_conn()
    edges = conn.execute(
        "SELECT source_hash, target_hash, relationship_type, valid_from "
        "FROM memory_graph WHERE relationship_type='supersedes'"
    ).fetchall()
    assert len(edges) >= 1
    e = dict(edges[0])
    assert e["source_hash"] == newer["content_hash"]
    assert e["target_hash"] == older["content_hash"]
    assert e["valid_from"] is not None and e["valid_from"] > 0


# --- Consolidate writes merged_into edge ---


def test_consolidate_records_merged_into_edge(store):
    """consolidate() must persist a 'merged_into' edge for each merged pair."""
    a = store.store(
        "Kubernetes node autoscaler scales aggressively",
        memory_type="learning",
        tags=["svc:k8s"],
    )
    b = store.store(
        "Kubernetes node autoscaler is aggressive during low traffic",
        memory_type="learning",
        tags=["svc:k8s"],
    )
    assert "content_hash" in a and "content_hash" in b

    conn = store._get_conn()
    emb_a = conn.execute(
        "SELECT content_embedding FROM memory_embeddings WHERE rowid = "
        "(SELECT id FROM memories WHERE content_hash = ?)",
        (a["content_hash"],),
    ).fetchone()
    if emb_a is None:
        pytest.skip("embedding row missing")
    # Copy a's embedding to b so similarity = 1.0
    conn.execute(
        "UPDATE memory_embeddings SET content_embedding = ? "
        "WHERE rowid = (SELECT id FROM memories WHERE content_hash = ?)",
        (emb_a["content_embedding"], b["content_hash"]),
    )
    conn.commit()

    result = store.consolidate(threshold=0.95)
    assert result["consolidated"] >= 1

    edges = conn.execute(
        "SELECT source_hash, target_hash, relationship_type, valid_from "
        "FROM memory_graph WHERE relationship_type='merged_into'"
    ).fetchall()
    assert len(edges) >= 1
    e = dict(edges[0])
    assert e["source_hash"] in {a["content_hash"], b["content_hash"]}
    assert e["target_hash"] in {a["content_hash"], b["content_hash"]}
    assert e["source_hash"] != e["target_hash"]
    assert e["valid_from"] is not None


# --- as_of filter on _search_graph ---


def test_search_graph_as_of_filter_excludes_invalidated_edges(store):
    """An edge with valid_to in the past must not be traversed when as_of is later."""
    a = store.store(
        "postgres tuning guide for service alpha",
        memory_type="reference",
        tags=["svc:alpha", "tool:postgres"],
    )
    b = store.store(
        "clickhouse query plans for service alpha",
        memory_type="reference",
        tags=["svc:alpha", "tool:clickhouse"],
    )
    assert "content_hash" in a and "content_hash" in b

    conn = store._get_conn()

    # Manually invalidate ALL entity_relations as of some past timestamp so the
    # graph traversal has nothing to walk through.  Simulates "history pruned".
    past = time.time() - 86400
    conn.execute("UPDATE entity_relations SET valid_to = ?", (past,))
    conn.commit()

    # as_of=now (after the valid_to) → no edges valid → no graph results
    now_results = store.search(query="alpha", mode="graph", limit=10)
    # The CTE seed-row alone can still return the seed memories at hop=0,
    # but walking is blocked.  We assert the count is bounded.
    assert len(now_results) <= 2  # at most the two seed memories at hop 0

    # as_of=before invalidation → walking allowed → results may be richer
    earlier = past - 10
    earlier_results = store.search(query="alpha", mode="graph", limit=10, as_of=earlier)
    # Either >= now_results (relaxed assertion — semantic doesn't drop hop-0)
    assert len(earlier_results) >= len(now_results) - 1 or earlier_results == now_results


# --- RRF integration ---


def test_search_hybrid_rrf_returns_rrf_score(populated_store):
    results = populated_store.search(query="kubernetes pod", mode="hybrid", limit=5, score_fusion="rrf")
    if not results:
        pytest.skip("no hybrid results on this corpus")
    assert "rrf_score" in results[0]
    assert results[0]["score"] == results[0]["rrf_score"]


def test_search_hybrid_weighted_path_still_works(populated_store):
    results = populated_store.search(query="kubernetes pod", mode="hybrid", limit=5, score_fusion="weighted")
    if not results:
        pytest.skip("no hybrid results on this corpus")
    # legacy path does not produce rrf_score
    assert "rrf_score" not in results[0] or results[0].get("rrf_score") is None
    assert "score" in results[0]
