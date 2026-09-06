#!/usr/bin/env python3
"""Do superseded facts outrank their replacements?

Task 34 deliberately stopped updates from overwriting what they supersede: both
facts now live in the store, on the reasoning that a stale hit is recoverable
and a discarded fact is not. That trade has a cost this measures — a
present-tense query can return the old value.

There is no labelled supersession set (one flagged event exists in production),
so the population is reconstructed the same way the discriminator defines it: a
pair above the dedup bar whose value tokens diverge, older memory treated as
superseded, newer as replacement.

The query must not leak the answer. A user asking what a setting is *now* does
not type the new value, so queries are built only from tokens the two memories
SHARE, with every value token removed. A pair is scored only if the search
reaches at least one of the two — an unreachable pair measures the corpus, not
the ranking.

  stale_wins  — old outranks new (the regression this task is about)
  fresh_wins  — new outranks old (working as intended)
  stale_only  — only the old one is retrieved at all (the worst case)

Usage:
  uv run python tests/bench_supersede_ranking.py --db /tmp/snapshot.db
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory.core import MemoryStore, _value_tokens, differs_by_value

TOK = re.compile(r"[a-z][a-z0-9_.-]{4,}")
STOP = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "been",
    "will",
    "they",
    "them",
    "their",
    "there",
    "when",
    "where",
    "which",
    "what",
    "into",
    "over",
    "more",
    "than",
    "also",
    "only",
    "just",
    "about",
    "after",
    "before",
    "should",
    "would",
    "could",
    "because",
    "these",
    "those",
    "using",
    "under",
}


def shared_query(a: str, b: str, k: int = 4) -> str | None:
    """Topic terms both memories share, with values stripped so the query
    cannot name the answer."""
    values = _value_tokens(a) | _value_tokens(b)
    ta = {t for t in TOK.findall(a.lower()) if t not in STOP and t not in values}
    tb = {t for t in TOK.findall(b.lower()) if t not in STOP and t not in values}
    common = sorted(ta & tb, key=len, reverse=True)[:k]
    return " ".join(common) if len(common) >= 2 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np

    store = MemoryStore(db_path=a.db)
    conn = store._get_conn()
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT id, content_hash, content, created_at, recall_count "
            "FROM memories WHERE deleted_at IS NULL AND memory_type NOT IN ('reference')"
        )
    ]
    ids = [r["id"] for r in rows]
    ph = ",".join("?" * len(ids))
    emb = {
        r["rowid"]: r["content_embedding"]
        for r in conn.execute(
            f"SELECT rowid, content_embedding FROM memory_embeddings WHERE rowid IN ({ph})",  # noqa: S608
            ids,
        )
    }
    vecs, keep = [], []
    for r in rows:
        b = emb.get(r["id"])
        if b is None:
            continue
        vecs.append(np.frombuffer(b, dtype=np.float32).copy())
        keep.append(r)
    sim = np.stack(vecs) @ np.stack(vecs).T
    n = len(keep)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) < a.threshold:
                continue
            x, y = keep[i], keep[j]
            if not differs_by_value(x["content"], y["content"]):
                continue
            old, new = sorted((x, y), key=lambda r: r["created_at"])
            pairs.append((old, new, float(sim[i, j])))

    tally = {"stale_wins": 0, "fresh_wins": 0, "stale_only": 0, "fresh_only": 0}
    unreachable = 0
    no_query = 0
    ranks = []
    for old, new, s in pairs:
        q = shared_query(old["content"], new["content"])
        if q is None:
            no_query += 1
            continue
        hits = store.search(q, limit=a.limit, track_recall=False)
        order = {h["content_hash"]: k for k, h in enumerate(hits)}
        ro, rn = order.get(old["content_hash"]), order.get(new["content_hash"])
        if ro is None and rn is None:
            unreachable += 1
            continue
        if ro is not None and rn is not None:
            tally["stale_wins" if ro < rn else "fresh_wins"] += 1
            ranks.append((ro, rn, s))
        elif ro is not None:
            tally["stale_only"] += 1
        else:
            tally["fresh_only"] += 1

    scored = sum(tally.values())
    both = tally["stale_wins"] + tally["fresh_wins"]
    out = {
        "_meta": {
            "probe": "task 40 — do superseded facts outrank their replacements",
            "harness": "tests/bench_supersede_ranking.py",
            "threshold": a.threshold,
            "limit": a.limit,
            "note": "Aggregates only; the pairs quote verbatim production content.",
        },
        "population": {
            "candidate_pairs": len(pairs),
            "no_shared_query": no_query,
            "unreachable_by_either": unreachable,
            "scored": scored,
        },
        "outcomes": tally,
        "rates": {
            "stale_wins_when_both_retrieved": round(tally["stale_wins"] / both, 3) if both else None,
            "stale_only_share_of_scored": round(tally["stale_only"] / scored, 3) if scored else None,
            # The head-to-head arm alone is ~12 pairs and moves between runs, so
            # read this one: over every scored pair, is the older memory ahead of
            # its replacement or retrieved instead of it?
            "stale_ahead_of_scored": (
                round((tally["stale_wins"] + tally["stale_only"]) / scored, 3) if scored else None
            ),
        },
        "reading": (
            "A value-blind ranker lands near 0.50 on stale_wins_when_both_retrieved. "
            "Above that means something systematically favours the older memory — "
            "recall_count and confidence both accrue with age. stale_only is the "
            "case that actually misleads: the replacement is not retrieved at all."
        ),
    }
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
