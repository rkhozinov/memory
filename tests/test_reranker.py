"""Tests for cross-encoder reranker."""

import numpy as np

from memory.reranker import RerankerModel, get_reranker


def test_score_pairs_returns_scores():
    """score_pairs returns sigmoid [0,1] scores; relevant candidate scores highest."""
    model = RerankerModel()
    scores = model.score_pairs(
        "kubernetes deployment",
        [
            "Kubernetes pod crash loop backoff: check container exit code",
            "Go60 ZMK firmware: disabled BLE and RGB underglow",
            "Terraform S3 backend state locking requires DynamoDB table",
        ],
    )
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    # The kubernetes-related candidate should score highest
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_score_pairs_empty():
    """Empty candidates returns empty array."""
    model = RerankerModel()
    scores = model.score_pairs("test query", [])
    assert len(scores) == 0
    assert isinstance(scores, np.ndarray)


def test_sigmoid_stability():
    """Numerically stable sigmoid handles extreme logit values."""
    model = RerankerModel()
    scores = model.score_pairs(
        "test",
        ["a very relevant document about testing"] * 5,
    )
    assert not np.any(np.isnan(scores))
    assert not np.any(np.isinf(scores))
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_singleton_reuses():
    """Same instance returned on repeated calls."""
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2
