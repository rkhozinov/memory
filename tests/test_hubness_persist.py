"""Hubness is persisted, not recomputed on every cold start.

compute_hubness is an O(n^2) matmul plus a full row sort. At n=5908 that is
403 ms and a 498 MB RSS spike, and the CLI spawns cold on every invocation, so
the shipping `weighted_best` default paid it every single search: 1.32 s / 688 MB
against 0.85 s / 156 MB for plain `weighted`. The benchmark's 6.2 ms p50 is a
warm-cache artifact of reusing one store instance.

Persisting it turns that into a column read. The values must be identical, or
this is a ranking change wearing a performance change's clothes.
"""

import numpy as np
import pytest

from memory.core import compute_hubness


@pytest.fixture
def hub_store(store):
    for i in range(12):
        store.store(f"memory number {i} about kubernetes deployments and rollouts", memory_type="note")
    store.store("a completely unrelated note about baking sourdough bread", memory_type="note")
    return store


def test_persisted_values_match_the_computed_ones(hub_store):
    """The pre-check: persistence must not change a single score."""
    conn = hub_store._get_conn()
    rows = conn.execute(
        "SELECT m.content_hash, e.content_embedding FROM memories m "
        "JOIN memory_embeddings e ON e.rowid = m.id WHERE m.deleted_at IS NULL"
    ).fetchall()
    expected = compute_hubness(
        {r["content_hash"]: np.frombuffer(r["content_embedding"], dtype=np.float32) for r in rows}, k=10
    )

    written = hub_store.refresh_hubness()
    assert written == len(expected)

    hub_store._hubness_cache = None
    got = hub_store._get_hubness(conn)
    assert set(got) == set(expected)
    for h, v in expected.items():
        assert got[h] == pytest.approx(v, abs=1e-6)


def test_reads_the_column_without_recomputing(hub_store, monkeypatch):
    hub_store.refresh_hubness()
    hub_store._hubness_cache = None

    import memory.core as core_mod

    def _boom(*_a, **_k):
        raise AssertionError("compute_hubness must not run when the column is populated")

    monkeypatch.setattr(core_mod, "compute_hubness", _boom)
    assert hub_store._get_hubness(hub_store._get_conn())


def test_falls_back_to_computing_when_the_column_is_empty(hub_store):
    """An un-backfilled store must still rank correctly, just slower."""
    conn = hub_store._get_conn()
    conn.execute("UPDATE memories SET hubness = NULL")
    conn.commit()
    hub_store._hubness_cache = None
    got = hub_store._get_hubness(conn)
    assert got and any(v != 0.0 for v in got.values())


def test_a_new_memory_is_neutral_until_refreshed(hub_store):
    """NULL hubness means no CSLS penalty, not a random one. A memory written
    between refreshes must not be demoted for lacking a score."""
    hub_store.refresh_hubness()
    res = hub_store.store("a brand new memory stored after the last refresh", memory_type="note")
    hub_store._hubness_cache = None
    got = hub_store._get_hubness(hub_store._get_conn())
    assert got.get(res["content_hash"], 0.0) == 0.0


def test_search_still_works_end_to_end(hub_store):
    hub_store.refresh_hubness()
    hub_store._hubness_cache = None
    hits = hub_store.search("kubernetes deployments", mode="hybrid", limit=5, score_fusion="weighted_best")
    assert hits
