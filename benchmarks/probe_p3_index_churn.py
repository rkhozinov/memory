#!/usr/bin/env python3
"""P3 — probe: does rebuilding the index wholesale erode it?

Spec: docs/research/consolidation-redesign.md, "Method probes", P3.

ACE (arXiv:2510.04618) argues that rewriting a context wholesale on every update
erodes detail — "context collapse" — and that incremental itemized updates avoid
it. `build_index` rebuilds wholesale every time, at 60 lines against ~5000
eligible memories: under 2% of the corpus.

Measures retention: of the entries present in build k, how many survive into
build k+1 as memories are added between builds. High retention means the
wholesale rebuild is fine at this budget and ACE's concern does not apply here.
Churn among entries that did not themselves change is context collapse in the
literal sense.

Read-only: snapshots the DB and writes only to the copy.

Usage:
  uv run python benchmarks/probe_p3_index_churn.py --db /tmp/snapshot.db
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MAX_LINES = 60
MAX_TOKENS = 1200


def _entries(index_md: str) -> list[str]:
    """Hashes in index order. Order matters: an entry that survives but drops
    twenty places has still lost the reader's attention."""
    out = []
    for line in index_md.split("\n"):
        if line.startswith("- `"):
            out.append(line.split("`")[1])
    return out


def run(db: str, rounds: int, adds_per_round: int, scope: str | None) -> dict:
    from memory.core import MemoryStore

    with TemporaryDirectory() as td:
        snap = Path(td) / "churn.db"
        shutil.copy2(db, snap)
        store = MemoryStore(db_path=snap)

        tags = [scope] if scope else []
        history = []
        prev: list[str] | None = None
        random.seed(19)

        for r in range(rounds + 1):
            idx = store.build_index(max_lines=MAX_LINES, max_tokens=MAX_TOKENS, tags=tags or None)
            cur = _entries(idx)
            if prev is not None:
                kept = [h for h in prev if h in cur]
                pos_prev = {h: i for i, h in enumerate(prev)}
                pos_cur = {h: i for i, h in enumerate(cur)}
                moves = [abs(pos_cur[h] - pos_prev[h]) for h in kept]
                history.append(
                    {
                        "round": r,
                        "size": len(cur),
                        "retained": len(kept) / len(prev) if prev else 1.0,
                        "dropped": len(prev) - len(kept),
                        "mean_position_shift": sum(moves) / len(moves) if moves else 0.0,
                    }
                )
            prev = cur

            if r == rounds:
                break
            # Add plausible new memories of index-eligible types, as a real
            # session would, so the next rebuild has something to reorder around.
            for i in range(adds_per_round):
                store.store(
                    f"Probe round {r} note {i}: the deployment pipeline validates "
                    f"the manifest before applying it to the cluster, run {r}-{i}",
                    memory_type=random.choice(["learning", "pattern", "error", "decision"]),
                    tags=(tags or []) + ["source:probe-p3"],
                )
            time.sleep(0)

        retentions = [h["retained"] for h in history]
        shifts = [h["mean_position_shift"] for h in history]
        return {
            "rounds": rounds,
            "adds_per_round": adds_per_round,
            "scope": scope,
            "index_size": history[0]["size"] if history else 0,
            "mean_retention": sum(retentions) / len(retentions) if retentions else 1.0,
            "min_retention": min(retentions) if retentions else 1.0,
            "mean_position_shift": sum(shifts) / len(shifts) if shifts else 0.0,
            "per_round": history,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Snapshot to copy and mutate.")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--adds-per-round", type=int, default=25)
    ap.add_argument("--scope", default="project:memory")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = run(args.db, args.rounds, args.adds_per_round, args.scope)
    print(f"\nP3 — index churn over {res['rounds']} rebuilds, +{res['adds_per_round']} memories between each\n")
    print(f"  index size            {res['index_size']} lines (cap {MAX_LINES})")
    print(f"  mean retention        {res['mean_retention']:.1%}")
    print(f"  worst round           {res['min_retention']:.1%}")
    print(f"  mean position shift   {res['mean_position_shift']:.2f} places\n")
    print(f"  {'round':>6} {'size':>6} {'retained':>10} {'dropped':>9} {'shift':>7}")
    for h in res["per_round"]:
        print(
            f"  {h['round']:>6} {h['size']:>6} {h['retained']:>9.1%} {h['dropped']:>9} {h['mean_position_shift']:>7.2f}"
        )
    print()
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
