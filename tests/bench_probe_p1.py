#!/usr/bin/env python3
"""P1 — probe: temporal retrieval and write-time conflict detection.

Spec: docs/research/consolidation-redesign.md, "Method probes", P1. Tests two
borrowed methods before either is built:

  as_of retrieval    Zep/Graphiti's bi-temporal model. Ask what was true at a
                     timestamp between a fact and its replacement; the correct
                     answer is the OLD fact.
  conflict on store  Mem0's extract -> conflict-detect -> update pipeline. Storing
                     something that contradicts an existing memory should say so.

Both are expected to fail on the current build. The interesting number is not
whether they fail — it is the FALSE-POSITIVE rate of the conflict detector
against real near-duplicate clusters, because a detector that fires on ordinary
duplicates is worse than none: it trains the operator to ignore it.

Usage:
  uv run python tests/bench_probe_p1.py
  uv run python tests/bench_probe_p1.py --db /tmp/snapshot.db   # FP rate on real data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

DAY = 86400.0


@dataclass
class TemporalCase:
    name: str
    old_content: str
    new_content: str
    query: str
    old_age_days: float
    new_age_days: float
    tags: list[str]
    memory_type: str


# Deliberately varied in whether dream's supersession heuristic could ever catch
# them: it needs cosine >= 0.85 AND >= 2 shared tags AND (a contradiction keyword
# OR type in {decision, error}). The reference pair below satisfies none of the
# last clause and is invisible to it permanently.
TEMPORAL_CASES = [
    TemporalCase(
        "embedding model",
        "The memory service embeds with all-MiniLM-L6-v2 at 384 dimensions on ONNX CPU",
        "The memory service embeds with modernbert-embed-base at 768 dimensions on MLX",
        "which embedding model does the memory service use",
        old_age_days=180,
        new_age_days=2,
        tags=["project:probe", "svc:embeddings"],
        memory_type="reference",
    ),
    TemporalCase(
        "CI runner",
        "CI for this repo builds on GitHub Actions ubuntu-22.04 runners",
        "CI for this repo builds on GitHub Actions ubuntu-24.04 runners",
        "which CI runner image does the repo build on",
        old_age_days=120,
        new_age_days=1,
        tags=["project:probe", "tool:github-actions"],
        memory_type="reference",
    ),
    TemporalCase(
        "auth mechanism",
        "The API authenticates callers with static API keys stored in the environment",
        "The API authenticates callers with short-lived OIDC tokens instead of static API keys",
        "how does the API authenticate callers",
        old_age_days=200,
        new_age_days=3,
        tags=["project:probe", "svc:auth"],
        memory_type="decision",
    ),
]


def _build_store(tmp: Path):
    import importlib

    from memory.core import MemoryStore

    conftest = importlib.import_module("conftest")
    s = MemoryStore(db_path=tmp / "p1.db")
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


def _store_aged(store, content, memory_type, tags, age_days):
    res = store.store(content, memory_type=memory_type, tags=tags)
    ts = time.time() - age_days * DAY
    store._get_conn().execute(
        "UPDATE memories SET created_at = ?, updated_at = ? WHERE content_hash = ?",
        (ts, ts, res["content_hash"]),
    )
    return res["content_hash"]


def probe_as_of() -> dict:
    """Can retrieval answer 'what was true then' rather than 'what is true now'?"""
    with TemporaryDirectory() as td:
        store = _build_store(Path(td))
        pairs = []
        for c in TEMPORAL_CASES:
            old_h = _store_aged(store, c.old_content, c.memory_type, c.tags, c.old_age_days)
            new_h = _store_aged(store, c.new_content, c.memory_type, c.tags, c.new_age_days)
            pairs.append((c, old_h, new_h))

        now_correct = 0
        as_of_correct = 0
        detail = []
        for c, old_h, new_h in pairs:
            now = store.search(c.query, mode="hybrid", limit=5, track_recall=False)
            now_top = now[0]["content_hash"] if now else None

            # Ask as of a moment between the two facts.
            midpoint = time.time() - ((c.old_age_days + c.new_age_days) / 2) * DAY
            past = store.search(c.query, mode="hybrid", limit=5, as_of=midpoint, track_recall=False)
            past_top = past[0]["content_hash"] if past else None

            now_ok = now_top == new_h
            as_of_ok = past_top == old_h
            now_correct += now_ok
            as_of_correct += as_of_ok
            detail.append(
                {"case": c.name, "current_query_returns_new_fact": now_ok, "as_of_returns_old_fact": as_of_ok}
            )

        n = len(pairs)
        return {
            "n": n,
            "current_accuracy": now_correct / n,
            "as_of_accuracy": as_of_correct / n,
            "detail": detail,
            "note": "as_of currently only reaches graph traversal (core.py _search_graph); the "
            "semantic and FTS rankers ignore it, so a past-tense query returns the present.",
        }


def probe_conflict_detection(threshold: float = 0.90) -> dict:
    """Does storing a contradiction get flagged, rather than silently coexisting?"""
    with TemporaryDirectory() as td:
        store = _build_store(Path(td))
        flagged = 0
        detail = []
        for c in TEMPORAL_CASES:
            _store_aged(store, c.old_content, c.memory_type, c.tags, c.old_age_days)
            res = store.store(c.new_content, memory_type=c.memory_type, tags=c.tags, dedup_threshold=threshold)
            status = res.get("status")
            # "duplicate" means the store REJECTED the replacement — arguably worse
            # than silence, because the new fact is lost. Neither is a conflict signal.
            is_conflict = status == "conflict"
            flagged += is_conflict
            detail.append({"case": c.name, "status": status, "flagged_as_conflict": is_conflict})
        n = len(TEMPORAL_CASES)
        return {
            "n": n,
            "conflicts_detected": flagged / n,
            "detail": detail,
            "note": "store() has two outcomes: duplicate (rejected) and stored. There is no "
            "third outcome for 'this contradicts something you already know'.",
        }


def probe_false_positive_rate(db: str, threshold: float = 0.90, sample: int = 400) -> dict:
    """The number that decides whether a conflict detector is usable.

    Runs the candidate rule — same memory_type, >=1 shared tag, cosine >= threshold —
    over real memory pairs. Every hit here is a pair a detector WOULD flag. Since
    the corpus is a store of accumulated facts rather than contradictions, the
    overwhelming majority of those would be false alarms.
    """
    import random

    import numpy as np

    from memory.core import MemoryStore

    store = MemoryStore(db_path=Path(db))
    conn = store._get_conn()
    rows = conn.execute("SELECT id, content_hash, memory_type, tags FROM memories WHERE deleted_at IS NULL").fetchall()
    random.seed(7)
    rows = random.sample(list(rows), min(sample, len(rows)))
    ids = [r["id"] for r in rows]
    ph = ",".join("?" * len(ids))
    emb = {
        r["rowid"]: np.frombuffer(r["content_embedding"], dtype=np.float32)
        # ph is a generated run of "?" placeholders, not user input; ids are bound.
        for r in conn.execute(
            f"SELECT rowid, content_embedding FROM memory_embeddings WHERE rowid IN ({ph})",
            ids,
        )
    }
    rows = [r for r in rows if r["id"] in emb]
    mat = np.stack([emb[r["id"]] for r in rows])
    sim = mat @ mat.T

    def tags_of(r):
        try:
            return set(json.loads(r["tags"] or "[]"))
        except (json.JSONDecodeError, TypeError):
            return set()

    n = len(rows)
    would_flag = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if float(sim[i, j]) < threshold:
                continue
            if rows[i]["memory_type"] != rows[j]["memory_type"]:
                continue
            if not (tags_of(rows[i]) & tags_of(rows[j])):
                continue
            would_flag += 1
    return {
        "sampled_memories": n,
        "pairs_compared": total_pairs,
        "would_flag": would_flag,
        "flag_rate_per_pair": would_flag / total_pairs if total_pairs else 0.0,
        "flags_per_100_memories": would_flag / n * 100 if n else 0.0,
        "threshold": threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="Snapshot DB for the false-positive measurement.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = {"as_of": probe_as_of(), "conflict_detection": probe_conflict_detection()}
    if args.db:
        out["false_positive"] = probe_false_positive_rate(args.db)

    a, c = out["as_of"], out["conflict_detection"]
    print("\nP1 — temporal retrieval and write-time conflict detection\n")
    print(f"  as_of accuracy          {a['as_of_accuracy']:.0%}  (threshold for adoption: 90%)")
    print(f"  current-query accuracy  {a['current_accuracy']:.0%}  (control: does 'now' at least work?)")
    for d in a["detail"]:
        print(
            f"    {d['case']:<18} now={'ok' if d['current_query_returns_new_fact'] else 'MISS'}"
            f"  as_of={'ok' if d['as_of_returns_old_fact'] else 'MISS'}"
        )
    print(f"\n  conflicts detected      {c['conflicts_detected']:.0%}")
    for d in c["detail"]:
        print(f"    {d['case']:<18} store() said {d['status']!r}")
    if "false_positive" in out:
        fp = out["false_positive"]
        print(f"\n  candidate rule on real data (cosine>={fp['threshold']}, same type, shared tag):")
        print(f"    {fp['would_flag']} pairs would be flagged out of {fp['pairs_compared']} compared")
        print(f"    = {fp['flags_per_100_memories']:.1f} flags per 100 memories")
    print()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
