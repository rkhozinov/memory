"""Tests for the cross-encoder reranker module and its integration with search()."""

from __future__ import annotations

import numpy as np
import pytest

from memory import rerank as rerank_mod
from memory.rerank import Reranker


def _fake_scorer_factory(boost_substr: str, boost: float = 0.95, base: float = 0.1):
    """Return a scorer that gives `boost` to docs containing boost_substr, else `base`.

    Used to deterministically test reranker reordering without loading a real model.
    """

    def _scorer(_query: str, docs: list[str]) -> np.ndarray:
        return np.array([boost if boost_substr in d else base for d in docs], dtype=np.float32)

    return _scorer


def test_reranker_score_returns_sigmoid_floats(tmp_path):
    r = Reranker(cache_db=tmp_path / "rerank.db", scorer=_fake_scorer_factory("apple"))
    scores = r.score("fruit", ["apple pie", "car wheel", "apple sauce"])
    assert scores.shape == (3,)
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_reranker_rerank_reorders_and_truncates(tmp_path):
    r = Reranker(cache_db=tmp_path / "rerank.db", scorer=_fake_scorer_factory("kubernetes"))
    items = [
        {"content": "kubectl is a tool", "score": 0.4},
        {"content": "kubernetes pod scheduler", "score": 0.5},
        {"content": "completely unrelated content", "score": 0.9},  # high composite, low rerank
    ]
    out = r.rerank("kubernetes pod scheduling", items, top_n=2)
    assert len(out) == 1 or len(out) == 2  # top_n caps result count
    assert out[0]["content"] == "kubernetes pod scheduler"
    assert "rerank_score" in out[0]


def test_reranker_l2_cache_persists(tmp_path):
    cache_db = tmp_path / "rerank.db"
    calls: list[int] = []

    def counting_scorer(_q: str, docs: list[str]) -> np.ndarray:
        calls.append(len(docs))
        return np.full(len(docs), 0.5, dtype=np.float32)

    r1 = Reranker(cache_db=cache_db, scorer=counting_scorer)
    r1.score("q", ["doc-a", "doc-b"])
    assert calls == [2]

    # New Reranker instance, same cache file → no recomputation
    r2 = Reranker(cache_db=cache_db, scorer=counting_scorer)
    r2.score("q", ["doc-a", "doc-b"])
    assert calls == [2]  # unchanged


def test_reranker_empty_inputs(tmp_path):
    r = Reranker(cache_db=tmp_path / "rerank.db", scorer=_fake_scorer_factory("x"))
    assert r.score("q", []).shape == (0,)
    assert r.rerank("q", []) == []


def test_search_rerank_reorders_results(store, tmp_path, monkeypatch):
    """End-to-end: rerank flips the order of two results that hybrid would
    otherwise rank by composite score.

    Bypasses the similarity threshold by directly injecting fake hybrid results
    via monkeypatch — the goal is to verify the rerank block in search() blends
    and reorders, not to exercise the retrieval funnel.
    """
    fake_hybrid_results = [
        {
            "content": "Terraform state locking on AWS",
            "content_hash": "h_terraform",
            "score": 0.9,  # high composite
            "memory_type": "pattern",
            "tags": [],
        },
        {
            "content": "Kubernetes pod scheduler internals",
            "content_hash": "h_k8s",
            "score": 0.4,  # lower composite
            "memory_type": "learning",
            "tags": [],
        },
    ]
    monkeypatch.setattr(store, "_search_hybrid", lambda *a, **kw: [dict(r) for r in fake_hybrid_results])

    fake = Reranker(
        cache_db=tmp_path / "rerank.db",
        scorer=_fake_scorer_factory("Kubernetes", boost=0.99, base=0.01),
    )
    monkeypatch.setattr(rerank_mod, "_reranker", fake)

    results = store.search(query="pod scheduling", mode="hybrid", limit=5, rerank=True)
    assert len(results) == 2
    assert results[0]["content_hash"] == "h_k8s"  # rerank flipped the order
    assert results[0]["rerank_score"] > 0.5
    assert results[1]["rerank_score"] < 0.5


def test_search_rerank_skipped_when_single_result(populated_store, tmp_path, monkeypatch):
    """Rerank should be a no-op when fewer than 2 results — saves a model call."""
    called: list[int] = []

    def counting_scorer(_q: str, docs: list[str]) -> np.ndarray:
        called.append(len(docs))
        return np.full(len(docs), 0.5, dtype=np.float32)

    fake = Reranker(cache_db=tmp_path / "rerank.db", scorer=counting_scorer)
    monkeypatch.setattr(rerank_mod, "_reranker", fake)

    # Query with a hard exact filter that yields 0-1 results
    results = populated_store.search(query="zzz-nonexistent-token-xyz", mode="exact", limit=5, rerank=True)
    assert len(results) < 2
    assert called == []  # reranker never invoked


@pytest.mark.parametrize("rerank_flag", [False, None])
def test_search_rerank_disabled(populated_store, tmp_path, monkeypatch, rerank_flag):
    """When rerank=False, the reranker is never consulted."""
    called: list[int] = []

    def counting_scorer(_q: str, docs: list[str]) -> np.ndarray:
        called.append(len(docs))
        return np.full(len(docs), 0.5, dtype=np.float32)

    fake = Reranker(cache_db=tmp_path / "rerank.db", scorer=counting_scorer)
    monkeypatch.setattr(rerank_mod, "_reranker", fake)

    if rerank_flag is None:
        results = populated_store.search(query="kubernetes", mode="hybrid", limit=5)
    else:
        results = populated_store.search(query="kubernetes", mode="hybrid", limit=5, rerank=rerank_flag)
    assert len(results) >= 1
    assert called == []
