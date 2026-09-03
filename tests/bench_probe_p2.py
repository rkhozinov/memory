#!/usr/bin/env python3
"""P2 — does multi-hop retrieval pay, and does IDF rescue its ranking?

Spec: docs/research/consolidation-redesign.md, "Method probes", P2, plus the
bounded experiment that probe left open.

The population is the one a knowledge graph exists for and vector similarity
structurally cannot serve: memory pairs sharing >= 2 entities but semantically
DISTANT (cosine < 0.45). The query names memory A and the entity it shares with
B; the gold answer is B.

P2 found traversal works (gold inside a 300-deep 2-hop pool in 50% of cases) but
ranking does not (top-10 in 3.3%). Graph score is
0.5/(1+hops) + 0.3*importance + 0.2*recency — no term for how INFORMATIVE the
bridge is. A shared "infrastructure" (4536 memories) is near-zero evidence; a
shared rare entity is decisive. This harness measures whether weighting the
bridge by IDF moves 3.3% toward the 50% ceiling.

Adoption rule, set before running: IDF must move recall@10 appreciably toward
the measured ceiling. If it does not, graph mode is retired and the O(k^2)
per-write cost in _link_entities is reclaimed.

Usage:
  uv run python tests/bench_probe_p2.py --db /tmp/snapshot.db --n 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 11
COSINE_CEILING = 0.45  # "semantically distant" — below MIN_SIMILARITY_THRESHOLD
MIN_SHARED_ENTITIES = 2


def _load(db: str, sample: int):
    import numpy as np

    from memory.core import MemoryStore

    store = MemoryStore(db_path=Path(db))
    conn = store._get_conn()
    rows = [dict(r) for r in conn.execute("SELECT id, content_hash, content FROM memories WHERE deleted_at IS NULL")]
    random.seed(SEED)
    random.shuffle(rows)
    rows = rows[:sample]
    ids = [r["id"] for r in rows]
    ph = ",".join("?" * len(ids))
    emb = {
        r[0]: np.frombuffer(r[1], dtype=np.float32)
        for r in conn.execute(
            f"SELECT rowid, content_embedding FROM memory_embeddings WHERE rowid IN ({ph})",  # noqa: S608
            ids,
        )
    }
    rows = [r for r in rows if r["id"] in emb]
    return store, conn, rows, emb


def _entity_maps(conn):
    """memory_id -> {entity_id}, plus the same keyed by content_hash.

    Graph results do not carry the integer id (see _search_graph -> to_dict), so
    the rescorer has to look entities up by content_hash. Keying it by id gives a
    silent no-op that re-sorts by importance and looks like a real regression —
    which is exactly what the first run of this experiment reported.
    """
    by_mem: dict[int, set[int]] = {}
    df: dict[int, int] = {}
    for mid, eid in conn.execute("SELECT memory_id, entity_id FROM memory_entities"):
        by_mem.setdefault(mid, set()).add(eid)
        df[eid] = df.get(eid, 0) + 1
    names = {r[0]: r[1] for r in conn.execute("SELECT id, display_name FROM entities")}
    id_to_hash = {r[0]: r[1] for r in conn.execute("SELECT id, content_hash FROM memories WHERE deleted_at IS NULL")}
    by_hash = {id_to_hash[mid]: ents for mid, ents in by_mem.items() if mid in id_to_hash}
    return by_mem, by_hash, df, names


def build_population(rows, emb, by_mem, df, names, n_cases: int, max_bridge_df: int | None = None):
    """Pairs sharing >= 2 entities but semantically distant.

    max_bridge_df restricts to cases where the rarest shared entity is genuinely
    rare. Without it the population is dominated by pairs whose only bridge is a
    hub, and IDF has nothing to weight — that is a property of the corpus, not a
    verdict on IDF, so both slices get measured.
    """
    import numpy as np

    mat = np.stack([emb[r["id"]] for r in rows])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mat @ mat.T

    cases = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if sims[i, j] >= COSINE_CEILING:
                continue
            a, b = rows[i], rows[j]
            shared = by_mem.get(a["id"], set()) & by_mem.get(b["id"], set())
            if len(shared) < MIN_SHARED_ENTITIES:
                continue
            # Seed the query with the RAREST shared entity — the bridge a reader
            # would actually name. Using the commonest would be a strawman.
            bridge = min(shared, key=lambda e: df.get(e, 1))
            if max_bridge_df is not None and df.get(bridge, 1) > max_bridge_df:
                continue
            cases.append(
                {
                    "query": f"{a['content'][:120]} {names.get(bridge, '')}",
                    "gold": b["content_hash"],
                    "bridge_df": df.get(bridge, 1),
                    "shared": len(shared),
                    "cosine": round(float(sims[i, j]), 3),
                }
            )
    random.seed(SEED)
    random.shuffle(cases)
    return cases[:n_cases]


def _rank_of(results, gold):
    for k, r in enumerate(results, 1):
        if r["content_hash"] == gold:
            return k
    return None


def _score(ranks, n, cutoff=10):
    mrr = sum(1.0 / r for r in ranks if r and r <= cutoff) / n
    rec = sum(1 for r in ranks if r and r <= cutoff) / n
    return {"mrr": round(mrr, 4), f"recall@{cutoff}": round(rec, 4)}


def run(store, cases, mode_fn, label):
    ranks, lat = [], []
    for c in cases:
        t0 = time.perf_counter()
        try:
            res = mode_fn(c["query"])
        except Exception:
            res = []
        lat.append((time.perf_counter() - t0) * 1000)
        ranks.append(_rank_of(res, c["gold"]))
    out = _score(ranks, len(cases))
    out["median_ms"] = round(sorted(lat)[len(lat) // 2], 1)
    out["label"] = label
    return out


def idf_rescore(by_hash, df, total_memories, results, query_entity_ids):
    """Re-rank graph hits by how informative the bridge entity is.

    IDF = log(N / df). A shared entity present in 4536 memories carries almost no
    evidence; one present in 2 carries a lot. The current score treats both the
    same, which is why a hub floods the top of the list.
    """
    for r in results:
        shared = by_hash.get(r["content_hash"], set()) & query_entity_ids
        best = 0.0
        for eid in shared:
            best = max(best, math.log(total_memories / max(df.get(eid, 1), 1)))
        max_idf = math.log(total_memories)
        r["idf_score"] = round(best / max_idf, 4) if max_idf else 0.0
        # Keep the structural terms, replace the flat proximity term with one
        # that says how much the bridge is worth.
        hops = r.get("graph_hops", 1)
        r["score"] = round(
            0.5 * r["idf_score"] / (1 + hops) * 2 + 0.3 * r.get("importance", 0.5) + 0.2 * r["idf_score"],
            4,
        )
    results.sort(key=lambda m: m["score"], reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11, help="Population shuffle seed; vary it to check stability.")
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument(
        "--max-bridge-df",
        type=int,
        default=None,
        help="Only keep cases whose rarest shared entity appears in <= N memories.",
    )
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    # one knob, set once from argv before any sampling
    global SEED
    SEED = a.seed
    store, conn, rows, emb = _load(a.db, a.sample)
    by_mem, by_hash, df, names = _entity_maps(conn)
    total = conn.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL").fetchone()[0]
    cases = build_population(rows, emb, by_mem, df, names, a.n, a.max_bridge_df)
    if not cases:
        print("no multi-hop cases found in sample", file=sys.stderr)
        return

    from memory.core import extract_entities

    def _query_entity_ids(q):
        wanted = {n for n, _, _ in extract_entities(q)}
        if not wanted:
            return set()
        ph = ",".join("?" * len(wanted))
        return {
            r[0]
            for r in conn.execute(
                f"SELECT id FROM entities WHERE name IN ({ph})",  # noqa: S608
                list(wanted),
            )
        }

    results = {
        "hybrid": run(store, cases, lambda q: store.search(q, mode="hybrid", limit=10, track_recall=False), "hybrid"),
        "graph_hops1": run(
            store, cases, lambda q: store.search(q, mode="graph", limit=10, max_hops=1, track_recall=False), "graph h1"
        ),
        "graph_hops2": run(
            store, cases, lambda q: store.search(q, mode="graph", limit=10, max_hops=2, track_recall=False), "graph h2"
        ),
    }

    def graph_idf(hops):
        def _f(q):
            pool = store.search(q, mode="graph", limit=a.pool, max_hops=hops, track_recall=False)
            return idf_rescore(by_hash, df, total, pool, _query_entity_ids(q))[:10]

        return _f

    # hops=1 + IDF is the cost-matched comparison: if IDF only pays at hops=2 it
    # is paying for the extra hop, not for the weighting.
    results["graph_hops1_idf"] = run(store, cases, graph_idf(1), f"graph h1 + IDF (pool {a.pool})")
    results["graph_hops2_idf"] = run(store, cases, graph_idf(2), f"graph h2 + IDF (pool {a.pool})")

    # Ceiling: is the gold anywhere in the deep pool at all?
    present = 0
    for c in cases:
        pool = store.search(c["query"], mode="graph", limit=a.pool, max_hops=2, track_recall=False)
        present += any(r["content_hash"] == c["gold"] for r in pool)
    ceiling = round(present / len(cases), 4)

    out = {
        "_meta": {
            "generated": time.strftime("%Y-%m-%d"),
            "probe": "P2 rerun — IDF-weighted graph ranking, the one experiment P2 left open",
            "population": f"pairs sharing >={MIN_SHARED_ENTITIES} entities with cosine < {COSINE_CEILING}",
            "n_cases": len(cases),
            "max_bridge_df": a.max_bridge_df,
            "sample_memories": len(rows),
            "seed": a.seed,
        },
        "results": results,
        f"ceiling_gold_present_in_{a.pool}_deep_pool": ceiling,
        "case_stats": {
            "median_bridge_df": sorted(c["bridge_df"] for c in cases)[len(cases) // 2],
            "median_shared_entities": sorted(c["shared"] for c in cases)[len(cases) // 2],
        },
    }
    print(json.dumps(out, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")


main()
