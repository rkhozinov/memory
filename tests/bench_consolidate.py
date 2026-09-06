#!/usr/bin/env python3
"""Consolidation quality benchmark: does a merge lose facts?

`consolidate()` decides which memory survives a near-duplicate cluster, then
applies a `content_strategy` to produce the survivor's content. Nothing in the
repo measured whether the surviving text still carries what the deleted members
uniquely knew — a merge that drops a fact is indistinguishable from a merge that
does not, because both report the same "merged N memories".

This harness fixes that. Each fixture cluster holds several memories that are
near-duplicates by cosine but each contributes exactly one unique fact (a port
number, a threshold, a flag). After consolidation we ask: for every unique fact
across the cluster, does the survivor's content still contain it?

  fact retention = (unique facts still present in survivors) / (unique facts stored)

The four strategies make very different promises here, and the numbers should be
read as a shape rather than a score:

  keep_higher_recall  survivor text untouched — retains only its own fact by
                      construction. This is the shipping default.
  keep_longer         retains whichever single member was longest.
  concat              appends a 200-char first-line snippet per member.
  mmr_union           extractive MMR over all members' sentences; the only
                      strategy that even attempts a union.

Usage:
  uv run python tests/bench_consolidate.py
  uv run pytest tests/bench_consolidate.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

STRATEGIES = ("keep_higher_recall", "keep_longer", "concat", "mmr_union")

# The path dream actually takes. dream calls `self.consolidate(dry_run=...)` with
# every default (core.py:4666), which is PAIRWISE mode — and pairwise never reads
# content_strategy at all. The loser is soft-deleted and the survivor's text is
# left untouched, so whatever the loser uniquely knew is simply gone. Measuring
# only the cluster-mode strategies would report on a path production never runs.
PAIRWISE = "pairwise (dream's actual path)"

# Cluster threshold. consolidate()'s pairwise default is 0.92, but cluster mode is
# documented at 0.85 and that is what `memory admin clusters` uses.
CLUSTER_THRESHOLD = 0.85


@dataclass
class ClusterFixture:
    """A group of near-duplicate memories, each carrying one unique fact."""

    name: str
    project: str
    memory_type: str
    # (content, unique_fact_substring)
    members: list[tuple[str, str]] = field(default_factory=list)


# Members within a fixture must be similar enough to land in one connected
# component at CLUSTER_THRESHOLD, which means near-identical phrasing with one
# detail swapped. That is exactly the shape real duplicate memories take when the
# same thing is written down across several sessions.
FIXTURES: list[ClusterFixture] = [
    ClusterFixture(
        "pgbouncer settings",
        "project:benchcons-a",
        "reference",
        [
            (
                "PgBouncer sits in front of Postgres for connection pooling. It listens on port 6432.",
                "6432",
            ),
            (
                "PgBouncer sits in front of Postgres for connection pooling. The pool mode is transaction.",
                "transaction",
            ),
            (
                "PgBouncer sits in front of Postgres for connection pooling. The default pool size is 25 connections.",
                "25 connections",
            ),
        ],
    ),
    ClusterFixture(
        "kubernetes probe config",
        "project:benchcons-b",
        "reference",
        [
            (
                "The service defines a Kubernetes readiness probe on the health endpoint. "
                "Its initial delay is 30 seconds.",
                "30 seconds",
            ),
            (
                "The service defines a Kubernetes readiness probe on the health endpoint. Its failure threshold is 5.",
                "failure threshold is 5",
            ),
            (
                "The service defines a Kubernetes readiness probe on the health endpoint. It polls every 10 seconds.",
                "every 10 seconds",
            ),
        ],
    ),
    ClusterFixture(
        "terraform backend",
        "project:benchcons-c",
        "reference",
        [
            (
                "Terraform keeps remote state in an S3 backend for this stack. The bucket is versioned.",
                "versioned",
            ),
            (
                "Terraform keeps remote state in an S3 backend for this stack. Locking uses a DynamoDB table.",
                "DynamoDB",
            ),
        ],
    ),
]


def _build_store(tmp: Path):
    """Fresh DB with the full production schema, as conftest builds it."""
    import importlib

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from memory.core import MemoryStore

    conftest = importlib.import_module("conftest")

    s = MemoryStore(db_path=tmp / "consolidate.db")
    conn = s._get_conn()
    conn.executescript(conftest._BASE_SCHEMA)
    s._migrate_stats_tables()
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings "
        "USING vec0(content_embedding FLOAT[768] distance_metric=cosine)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
        "USING fts5(content, content='memories', content_rowid='id', tokenize='porter ascii')"
    )
    return s


def _load(store) -> None:
    for fx in FIXTURES:
        for content, _fact in fx.members:
            store.store(content, memory_type=fx.memory_type, tags=[fx.project])


def _surviving(store) -> list[str]:
    """Every live memory's content. A fact counts as retained if it appears in any
    of them — landing on a different survivor still means it is readable."""
    rows = store._get_conn().execute("SELECT content FROM memories WHERE deleted_at IS NULL").fetchall()
    return [r["content"] for r in rows]


def measure(strategy: str) -> dict:
    """Load the fixtures into a fresh DB, consolidate, report fact retention."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        store = _build_store(Path(td))
        _load(store)
        before = len(_surviving(store))

        if strategy == PAIRWISE:
            result = store.consolidate(
                threshold=CLUSTER_THRESHOLD,
                cluster=False,
                exclude_types=[],
                project_scoped=True,
                # Every fixture member carries a distinct value by construction
                # — that is what fact retention measures. The value guard would
                # correctly block all of it, leaving nothing to measure.
                value_guard=False,
            )
        else:
            result = store.consolidate(
                threshold=CLUSTER_THRESHOLD,
                cluster=True,
                content_strategy=strategy,
                exclude_types=[],
                project_scoped=True,
                value_guard=False,
            )

        survivors = _surviving(store)
        text = "\n".join(survivors).lower()
        facts = [(fx.name, fact) for fx in FIXTURES for _c, fact in fx.members]
        kept = [(n, f) for n, f in facts if f.lower() in text]
        lost = [(n, f) for n, f in facts if f.lower() not in text]

        return {
            "strategy": strategy,
            "memories_before": before,
            "memories_after": len(survivors),
            "merged": result.get("merged", result.get("consolidated", 0)),
            "facts_total": len(facts),
            "facts_kept": len(kept),
            "retention": len(kept) / len(facts) if facts else 0.0,
            "lost": lost,
        }


def main() -> int:
    rows = [measure(s) for s in (PAIRWISE, *STRATEGIES)]

    print(
        f"\nFixtures: {len(FIXTURES)} clusters, {sum(len(f.members) for f in FIXTURES)} memories, "
        f"{rows[0]['facts_total']} unique facts, cluster threshold {CLUSTER_THRESHOLD}\n"
    )
    print("| strategy                    | merged | memories after | facts kept | retention |")
    print("|-----------------------------|--------|----------------|------------|-----------|")
    for r in rows:
        print(
            f"| {r['strategy']:<27} | {r['merged']:>6} | {r['memories_after']:>14} "
            f"| {r['facts_kept']:>2}/{r['facts_total']:<8} | {r['retention']:>8.0%} |"
        )

    for r in rows:
        if r["lost"]:
            print(f"\n{r['strategy']} lost:")
            for name, fact in r["lost"]:
                print(f"  [{name}] {fact!r}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Regression guard
# ---------------------------------------------------------------------------


def test_mmr_union_retains_at_least_as_much_as_default():
    """The union strategy must never lose more facts than the do-nothing default.

    Not asserted as 100% retention: MMR is extractive with a sentence budget, so
    perfect retention is not something it promises. What it must not do is come
    out behind `keep_higher_recall`, which keeps one member's text verbatim.
    """
    default = measure("keep_higher_recall")
    union = measure("mmr_union")
    assert union["retention"] >= default["retention"], (
        f"mmr_union retained {union['retention']:.0%} vs "
        f"keep_higher_recall {default['retention']:.0%}; lost {union['lost']}"
    )


def test_pairwise_is_not_worse_than_the_union_strategy():
    """Pins the gap between what dream runs and what it could run.

    dream takes the pairwise path, which cannot preserve a loser's unique content
    because it never rewrites the survivor. If this ever stops being a gap —
    because dream switched to cluster mode, or pairwise learned to merge content —
    this test should fail and be deleted along with the finding it guards.
    """
    pairwise = measure(PAIRWISE)
    union = measure("mmr_union")
    assert union["retention"] >= pairwise["retention"], (
        f"mmr_union {union['retention']:.0%} vs pairwise {pairwise['retention']:.0%}"
    )


def test_consolidation_actually_merges():
    """If nothing merges, the retention numbers above are vacuously perfect."""
    r = measure("keep_higher_recall")
    assert r["merged"] > 0, (
        f"no clusters merged at threshold {CLUSTER_THRESHOLD} — the fixtures are no "
        "longer near-duplicates under the current embedding model, so this benchmark "
        "is measuring nothing. Retune the fixtures or the threshold."
    )


if __name__ == "__main__":
    sys.exit(main())
