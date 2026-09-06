"""The value-token guard on consolidation.

Cosine >= 0.85 does not mean duplicate.  On the production store, 393 of the
pairs at that threshold carry diverging value tokens — a port, a version, an
agent id — which means merging them deletes a fact.  `differs_by_value` (already
shipped for write-time dedup) is the same discriminator, reused here: it blocks
388 of those 393, and lets genuine restatements through.

These tests pin both directions.  A guard that only blocked would be trivially
"safe" and useless.
"""

from __future__ import annotations

from tests.test_clusters import _seed_similar_cluster

# Same fact, three phrasings, no value token in dispute — must still merge.
RESTATEMENTS = [
    "Postgres connection pooling is handled by PgBouncer.",
    "PgBouncer handles connection pooling for Postgres.",
    "Connection pooling for Postgres goes through PgBouncer.",
]

# Near-identical by cosine, but each names a different port — merging drops facts.
DIVERGING_VALUES = [
    "PgBouncer listens on port 6432 in front of Postgres.",
    "PgBouncer listens on port 7432 in front of Postgres.",
]


def test_pairwise_guard_blocks_diverging_values(store):
    _seed_similar_cluster(store, DIVERGING_VALUES)
    result = store.consolidate(threshold=0.85, project_scoped=False)
    assert result["consolidated"] == 0
    assert result["blocked_by_guard"] == 1


def test_pairwise_guard_allows_restatements(store):
    _seed_similar_cluster(store, RESTATEMENTS)
    result = store.consolidate(threshold=0.85, project_scoped=False)
    assert result["consolidated"] == 2
    assert result["blocked_by_guard"] == 0


def test_cluster_guard_blocks_whole_cluster(store):
    seed = _seed_similar_cluster(store, DIVERGING_VALUES)
    result = store.consolidate(threshold=0.85, cluster=True, project_scoped=False)
    assert result["consolidated"] == 0
    assert result["blocked_by_guard"] == 1
    conn = store._get_conn()
    alive = conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE content_hash IN (?, ?) AND deleted_at IS NULL",
        tuple(seed),
    ).fetchone()["n"]
    assert alive == 2


def test_cluster_guard_allows_restatements(store):
    _seed_similar_cluster(store, RESTATEMENTS)
    result = store.consolidate(threshold=0.85, cluster=True, project_scoped=False)
    assert result["consolidated"] == 2


def test_guard_can_be_disabled(store):
    """Benchmarks deliberately seed unique-fact clusters; they opt out."""
    _seed_similar_cluster(store, DIVERGING_VALUES)
    result = store.consolidate(threshold=0.85, project_scoped=False, value_guard=False)
    assert result["consolidated"] == 1


def test_dry_run_reports_blocked_count(store):
    _seed_similar_cluster(store, DIVERGING_VALUES)
    result = store.consolidate(threshold=0.85, dry_run=True, project_scoped=False)
    assert result["would_consolidate"] == 0
    assert result["blocked_by_guard"] == 1
