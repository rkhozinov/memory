#!/usr/bin/env python3
"""Measure the value-token discriminator against real production pairs.

The question is not "does it catch ubuntu-22.04 -> ubuntu-24.04" — it does, by
construction.  It is: over pairs that dedup at 0.90 would call duplicate, what
fraction does the discriminator reclassify as an update?  Near 100% would mean
the guard has simply disabled dedup.

Prints aggregates plus a handful of examples.  The examples quote verbatim
production memory content, so they go to stdout for inspection and must not be
committed — benchmarks/results/supersede-discriminator.json keeps the aggregates
only.

Usage:
  uv run python tests/bench_supersede.py --db /tmp/snapshot.db --sample 4000
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory.core import _value_tokens as value_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--sample", type=int, default=800)
    a = ap.parse_args()

    import numpy as np

    from memory.core import MemoryStore

    store = MemoryStore(db_path=a.db)
    conn = store._get_conn()
    rows = conn.execute(
        "SELECT m.id, m.content_hash, m.content, m.memory_type, m.tags FROM memories m WHERE m.deleted_at IS NULL"
    ).fetchall()
    rows = [dict(r) for r in rows]
    random.seed(7)
    random.shuffle(rows)
    rows = rows[: a.sample]
    ids = [r["id"] for r in rows]
    # qmarks is a generated run of "?" placeholders, not user input; the ids
    # themselves are still bound as parameters.
    qmarks = ",".join("?" * len(ids))
    embs = {}
    for rid, blob in conn.execute(
        f"SELECT rowid, content_embedding FROM memory_embeddings WHERE rowid IN ({qmarks})",  # noqa: S608
        ids,
    ):
        embs[rid] = np.frombuffer(blob, dtype=np.float32)
    rows = [r for r in rows if r["id"] in embs]
    print(f"sample with embeddings: {len(rows)}", file=sys.stderr)

    mat = np.stack([embs[r["id"]] for r in rows])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mat @ mat.T

    n = len(rows)
    dup_pairs, update_pairs, examples = 0, 0, []
    bands = {"0.90-0.95": [0, 0], "0.95-0.99": [0, 0], ">=0.99": [0, 0]}
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] < a.threshold:
                continue
            ri, rj = rows[i], rows[j]
            if ri["memory_type"] != rj["memory_type"]:
                continue
            ti = set(json.loads(ri["tags"] or "[]"))
            tj = set(json.loads(rj["tags"] or "[]"))
            if ti and tj and not (ti & tj):
                continue
            dup_pairs += 1
            sim = float(sims[i, j])
            band = "0.90-0.95" if sim < 0.95 else ("0.95-0.99" if sim < 0.99 else ">=0.99")
            bands[band][0] += 1
            vi, vj = value_tokens(ri["content"]), value_tokens(rj["content"])
            if vi != vj and (vi - vj or vj - vi):
                update_pairs += 1
                bands[band][1] += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "sim": round(float(sims[i, j]), 3),
                            "diff": sorted(vi ^ vj)[:6],
                            "a": ri["content"][:110],
                            "b": rj["content"][:110],
                        }
                    )
    print(
        json.dumps(
            {
                "threshold": a.threshold,
                "sample": len(rows),
                "pairs_current_dedup_calls_duplicate": dup_pairs,
                "reclassified_as_update": update_pairs,
                "reclassify_rate": round(update_pairs / dup_pairs, 3) if dup_pairs else None,
                "by_similarity_band": {
                    k: {"pairs": v[0], "reclassified": v[1], "rate": round(v[1] / v[0], 3) if v[0] else None}
                    for k, v in bands.items()
                },
                "examples": examples,
            },
            indent=2,
        )
    )


main()
