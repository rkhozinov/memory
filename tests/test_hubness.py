"""Win #3: hubness (hot-cluster) correction via CSLS.

Short generic memories embed near topic centroids and become "hubs" that appear
in many kNN lists, crowding out specific ("anti-hub") memories.  CSLS subtracts
each doc's mean-neighbor similarity so centroid-huggers get demoted:

    csls(q, d) = 2*cos(q,d) - r_q - r_d
"""

from __future__ import annotations

import numpy as np

from memory.core import _csls_rescore, compute_hubness


def test_csls_rescore_promotes_anti_hub_over_hub():
    # A has higher raw similarity but is a hub (r_d high, close to everything).
    # B is slightly less similar but an anti-hub (r_d low).  CSLS must flip them.
    results = [
        {"content_hash": "A", "similarity": 0.80},
        {"content_hash": "B", "similarity": 0.78},
    ]
    hubness = {"A": 0.70, "B": 0.30}
    out = _csls_rescore(results, r_q=0.5, hubness=hubness)
    assert [r["content_hash"] for r in out] == ["B", "A"]
    # csls score attached for downstream/debug
    assert out[0]["csls_score"] > out[1]["csls_score"]


def test_csls_rescore_missing_hubness_defaults_zero():
    # unknown doc → r_d=0 → pure 2*sim - r_q; must not crash, keeps order by sim
    results = [{"content_hash": "X", "similarity": 0.9}, {"content_hash": "Y", "similarity": 0.4}]
    out = _csls_rescore(results, r_q=0.5, hubness={})
    assert [r["content_hash"] for r in out] == ["X", "Y"]


def test_compute_hubness_tolerates_degenerate_vectors():
    """Production DBs contain occasional zero-norm or non-finite embeddings
    (bad historical writes).  compute_hubness must never emit NaN/Inf — those
    poison the CSLS rescore.  Regression for the matmul overflow on real data."""
    vecs = {
        "good1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "good2": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        "zero": np.zeros(3, dtype=np.float32),  # zero-norm
        "naninf": np.array([np.nan, np.inf, 0.0], dtype=np.float32),
    }
    hub = compute_hubness(vecs, k=2)
    assert set(hub) == set(vecs)
    for h, v in hub.items():
        assert np.isfinite(v), f"hubness for {h} is non-finite: {v}"


def test_compute_hubness_scores_hub_higher_than_anti_hub():
    # 3 near-identical vectors (a cluster/hub) + 1 far outlier (anti-hub).
    # The clustered vectors must get higher mean-neighbor similarity than the outlier.
    vecs = {
        "hub1": np.array([1.0, 0.0, 0.0]),
        "hub2": np.array([0.99, 0.01, 0.0]),
        "hub3": np.array([0.98, 0.02, 0.0]),
        "outlier": np.array([0.0, 0.0, 1.0]),
    }
    hub = compute_hubness(vecs, k=2)
    assert hub["hub1"] > hub["outlier"]
    assert hub["hub2"] > hub["outlier"]
