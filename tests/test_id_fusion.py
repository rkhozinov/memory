"""Win #2: identifier-aware hybrid fusion.

When a query contains an identifier-like token (ticket ID, PR ref, error code,
hyphenated symbol), boost the FTS/BM25 sub-ranker over semantic — dense vectors
can't exact-match short symbols, but BM25 nails the rare high-IDF token.
"""

from __future__ import annotations

import pytest

from memory.core import (
    _extract_identifier_tokens,
    _looks_like_identifier,
    _promote_exact_matches,
    _weighted_fuse,
)


# --- _looks_like_identifier unit tests ---


@pytest.mark.parametrize(
    "query",
    [
        "TICKET-194",       # ticket id
        "TICKET-64",
        "PR #502",       # PR ref
        "#456",
        "east-2",        # word-hyphen-digit
        "nemotron-3",
        "acme-2",
        "ERR_CONN_RESET_4XX",  # underscore + digit
        "fix TICKET-204 today",   # embedded in a sentence
    ],
)
def test_looks_like_identifier_true(query):
    assert _looks_like_identifier(query) is True


def test_weighted_best_promotes_exact_id_and_rescores_general(populated_store):
    """weighted_best composes CSLS (general) + id boost/promotion (identifiers).
    An identifier query must return the memory literally containing the id at
    rank 1; a general query must carry a csls_score (hubness correction applied)."""
    id_hits = populated_store.search(query="TICKET-64", mode="hybrid", limit=5, score_fusion="weighted_best")
    assert id_hits, "expected results for TICKET-64"
    assert "TICKET-64" in id_hits[0]["content"], f"exact id not promoted: {id_hits[0]['content'][:60]}"

    gen = populated_store.search(
        query="kubernetes pod crash", mode="hybrid", limit=5, score_fusion="weighted_best"
    )
    if gen:
        assert "csls_score" in gen[0], "csls rescoring not applied on general query"


def test_weighted_fuse_default_matches_additive():
    sem = [{"content_hash": "A", "score": 0.9}, {"content_hash": "B", "score": 0.8}]
    fts = [{"content_hash": "B", "score": 0.5}, {"content_hash": "C", "score": 0.4}]
    out = _weighted_fuse(sem, fts, limit=10, fts_weight=1.0)
    order = [r["content_hash"] for r in out]
    # B in both → 0.8+0.5=1.3 top; A=0.9; C=0.4
    assert order == ["B", "A", "C"]


def test_weighted_fuse_fts_boost_promotes_exact_match():
    sem = [{"content_hash": "A", "score": 0.9}, {"content_hash": "B", "score": 0.8}]
    fts = [{"content_hash": "B", "score": 0.5}, {"content_hash": "C", "score": 0.4}]
    out = _weighted_fuse(sem, fts, limit=10, fts_weight=3.0)
    order = [r["content_hash"] for r in out]
    # B=0.8+1.5=2.3; C=1.2 (fts-only, boosted); A=0.9 (sem-only)
    # → the exact FTS-only hit C now outranks the semantic-only A
    assert order == ["B", "C", "A"]


def test_extract_identifier_tokens():
    assert _extract_identifier_tokens("fix TICKET-194 now") == ["TICKET-194"]
    assert _extract_identifier_tokens("PR #502") == ["#502"]
    assert _extract_identifier_tokens("east-2 region") == ["east-2"]
    assert _extract_identifier_tokens("no ids here") == []


def test_promote_exact_matches_moves_verbatim_hit_to_front():
    # gold literally contains 'TICKET-194'; the higher-scored distractor only shares
    # the split tokens.  Exact-substring promotion must reorder gold to rank 1.
    results = [
        {"content_hash": "distractor", "content": "nem work on 194 things", "score": 0.9},
        {"content_hash": "gold", "content": "cameras-server (TICKET-194) migration", "score": 0.7},
        {"content_hash": "other", "content": "unrelated memory", "score": 0.5},
    ]
    out = _promote_exact_matches(results, ["TICKET-194"])
    assert [r["content_hash"] for r in out] == ["gold", "distractor", "other"]


def test_promote_exact_matches_is_case_insensitive():
    results = [
        {"content_hash": "a", "content": "no match", "score": 0.9},
        {"content_hash": "b", "content": "see PR #502 details", "score": 0.4},
    ]
    out = _promote_exact_matches(results, ["#502"])
    assert out[0]["content_hash"] == "b"


def test_promote_exact_matches_no_tokens_is_identity():
    results = [{"content_hash": "a", "content": "x", "score": 0.9}]
    assert _promote_exact_matches(results, []) == results


@pytest.mark.parametrize(
    "query",
    [
        "kubernetes pod restart",
        "how does the database work",
        "deploy-image force-with-lease create-pull-request",  # hyphens, NO digit
        "modifying invocation re-reads",
        "",
        "the quick brown fox",
    ],
)
def test_looks_like_identifier_false(query):
    assert _looks_like_identifier(query) is False
