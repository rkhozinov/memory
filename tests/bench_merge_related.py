#!/usr/bin/env python3
"""Does merging small related memories into bigger ones cost retrieval?

Not the dedup question. These are memories that are topically close but carry
*different* facts — the 802 diverging pairs the value guard blocks. The proposal
is to fold each such group into one richer memory.

The test asks the only thing that matters: after merging, can you still get back
the fact that lived in one member? For every member we build a query from the
tokens that member has and its group-mates do not — the distinctive fact — and
compare the rank of that member before the merge against the rank of the
survivor after.

  before: originals in the store, query targets member M -> rank of M
  after : group folded into one survivor via mmr_union -> rank of survivor

A merge that preserves retrieval leaves MRR flat. A merge that buries facts
drops it.

Usage:
  uv run python tests/bench_merge_related.py --db /tmp/snap.db --groups 80
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory.core import MemoryStore, _mmr_union_merge, _serialize_f32

TOK = re.compile(r"[a-z][a-z0-9_.-]{3,}")
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
    "does",
}


def toks(s: str) -> set[str]:
    return {t for t in TOK.findall((s or "").lower()) if t not in STOP}


def distinctive_query(member: str, siblings: list[str], k: int = 5) -> str | None:
    """Tokens this member has that its group-mates lack — the fact that would
    be lost if the merge buried it."""
    mine = toks(member)
    others: set[str] = set()
    for s in siblings:
        others |= toks(s)
    uniq = sorted(mine - others, key=len, reverse=True)[:k]
    return " ".join(uniq) if len(uniq) >= 2 else None


def crosscut_query(a_txt: str, b_txt: str, k: int = 4) -> str | None:
    """A question that spans two members: half its terms are unique to one, half
    to the other. This is where a merge should WIN — one memory answers it,
    where before you needed two hits.
    """
    ta, tb = toks(a_txt), toks(b_txt)
    ua = sorted(ta - tb, key=len, reverse=True)[: k // 2]
    ub = sorted(tb - ta, key=len, reverse=True)[: k // 2]
    if len(ua) < 2 or len(ub) < 2:
        return None
    return " ".join(ua + ub)


def rank_of(store: MemoryStore, query: str, target: str, limit: int) -> int | None:
    hits = store.search(query, limit=limit, track_recall=False)
    for i, h in enumerate(hits):
        if h["content_hash"] == target:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Snapshot to mutate. Never the live store.")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--groups", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=900, help="Only merge 'small' memories")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np

    store = MemoryStore(db_path=a.db)
    conn = store._get_conn()
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT id, content_hash, content, memory_type, recall_count FROM memories "
            "WHERE deleted_at IS NULL AND memory_type NOT IN ('reference') "
            "AND LENGTH(content) BETWEEN 80 AND ?",
            (a.max_chars,),
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
    np.fill_diagonal(sim, 0.0)
    n = len(keep)

    # Greedy grouping: each memory joins at most one group.
    used: set[int] = set()
    groups: list[list[int]] = []
    for i in range(n):
        if i in used or len(groups) >= a.groups:
            continue
        mates = [j for j in np.where(sim[i] >= a.threshold)[0] if j not in used and j != i]
        if not mates:
            continue
        g = [i, *mates[:2]]  # groups of 2-3
        used.update(g)
        groups.append(g)

    # --- before ---
    probes = []
    for g in groups:
        for idx in g:
            q = distinctive_query(keep[idx]["content"], [keep[o]["content"] for o in g if o != idx])
            if q is None:
                continue
            r = rank_of(store, q, keep[idx]["content_hash"], a.limit)
            probes.append({"group": g, "member": idx, "query": q, "before": r})

    # --- cross-cutting probes: BEFORE, does one search return both members? ---
    cross = []
    for g in groups:
        if len(g) < 2:
            continue
        q = crosscut_query(keep[g[0]]["content"], keep[g[1]]["content"])
        if q is None:
            continue
        hits = store.search(q, limit=a.limit, track_recall=False)
        got = {h["content_hash"] for h in hits}
        cross.append(
            {
                "group": g,
                "query": q,
                "before_both": keep[g[0]]["content_hash"] in got and keep[g[1]]["content_hash"] in got,
                "before_any": bool(got & {keep[g[0]]["content_hash"], keep[g[1]]["content_hash"]}),
            }
        )

    # --- merge each group into its first member, mmr_union, re-embed ---
    from memory.embeddings import get_model

    model = get_model()
    survivors: dict[tuple, str] = {}
    for g in groups:
        contents = [keep[i]["content"] for i in g]
        merged = _mmr_union_merge(contents, max_sentences=40, max_chars=4000, lambda_relevance=0.7)
        surv_id = keep[g[0]]["id"]
        conn.execute("UPDATE memories SET content = ? WHERE id = ?", (merged, surv_id))
        vec = model.embed_doc(merged)
        conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (surv_id,))
        conn.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
            (surv_id, _serialize_f32(vec)),
        )
        conn.execute("INSERT OR REPLACE INTO memory_fts (rowid, content) VALUES (?, ?)", (surv_id, merged))
        for i in g[1:]:
            rid = keep[i]["id"]
            conn.execute("UPDATE memories SET deleted_at = strftime('%s','now') WHERE id = ?", (rid,))
            conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (rid,))
            conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (rid,))
        survivors[tuple(g)] = keep[g[0]]["content_hash"]
    conn.commit()

    # --- after ---
    for p in probes:
        p["after"] = rank_of(store, p["query"], survivors[tuple(p["group"])], a.limit)

    for cpr in cross:
        hits = store.search(cpr["query"], limit=a.limit, track_recall=False)
        cpr["after_hit"] = survivors[tuple(cpr["group"])] in {h["content_hash"] for h in hits}

    def mrr(key):
        vals = [1.0 / (p[key] + 1) if p[key] is not None else 0.0 for p in probes]
        return sum(vals) / len(vals) if vals else 0.0

    def recall(key):
        return sum(1 for p in probes if p[key] is not None) / len(probes) if probes else 0.0

    lost = sum(1 for p in probes if p["before"] is not None and p["after"] is None)
    gained = sum(1 for p in probes if p["before"] is None and p["after"] is not None)

    out = {
        "_meta": {
            "probe": "does merging small related memories cost retrieval",
            "harness": "tests/bench_merge_related.py",
            "threshold": a.threshold,
            "max_chars": a.max_chars,
            "limit": a.limit,
            "note": "Aggregates only; queries derive from verbatim production content.",
        },
        "population": {
            "candidate_memories": n,
            "groups_merged": len(groups),
            "memories_folded_away": sum(len(g) - 1 for g in groups),
            "probes": len(probes),
        },
        "before": {"mrr": round(mrr("before"), 4), "recall_at_limit": round(recall("before"), 4)},
        "after": {"mrr": round(mrr("after"), 4), "recall_at_limit": round(recall("after"), 4)},
        "delta_mrr": round(mrr("after") - mrr("before"), 4),
        "crosscutting": {
            "probes": len(cross),
            "before_both_members_in_topk": sum(1 for c in cross if c["before_both"]),
            "before_at_least_one": sum(1 for c in cross if c["before_any"]),
            "after_merged_memory_in_topk": sum(1 for c in cross if c["after_hit"]),
            "reading": (
                "The case merging is supposed to win: a query spanning two members. "
                "Before, answering it fully needs BOTH in the top-k. After, one merged "
                "memory can carry it. Compare before_both against after_merged."
            ),
        },
        "facts_lost_from_topk": lost,
        "facts_gained": gained,
        "reading": (
            "Each probe asks for the fact that lived in ONE member, using tokens its "
            "group-mates do not have. If the merged memory still answers, retrieval "
            "survived consolidation. facts_lost_from_topk counts members that were "
            "retrievable before and are not after."
        ),
    }
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
