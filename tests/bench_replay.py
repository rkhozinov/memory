#!/usr/bin/env python3
"""A/B replay benchmark for Phase A/B/C uplift.

Replays the same 20-query corpus from benchmarks/corpus.py against
MemoryStore.search() under four configurations:

  baseline   weighted score fusion, no rerank, no activation
  +rrf       Reciprocal Rank Fusion (Phase B default)
  +rerank    + cross-encoder rerank (Phase A)
  +all       + ACT-R activation override (Phase C)

Metrics reported per config:
  MRR@10     mean reciprocal rank, top-10 results
  Recall@5   fraction of queries whose expected match is in top-5
  Top1       fraction with expected match at rank 1
  p50, p95   per-query search latency (ms)

Usage:
  uv run python tests/bench_replay.py
  uv run python tests/bench_replay.py --json /tmp/bench.json
  uv run python tests/bench_replay.py --configs baseline,+all
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from memory import core as core_mod  # noqa: E402
from memory.core import MemoryStore  # noqa: E402

corpus = importlib.import_module("corpus")


CONFIGS: dict[str, dict] = {
    "baseline": dict(score_fusion="weighted", rerank=False, use_activation=False),
    "+rrf": dict(score_fusion="rrf", rerank=False, use_activation=False),
    "+rrsb": dict(score_fusion="rrsb", rerank=False, use_activation=False),
    "+rerank": dict(score_fusion="weighted", rerank=True, use_activation=False),
    "+rrsb+rr": dict(score_fusion="rrsb", rerank=True, use_activation=False),
    "+all": dict(score_fusion="weighted", rerank=True, use_activation=True),
    "+id": dict(score_fusion="weighted_id", rerank=False, use_activation=False),
    "+csls": dict(score_fusion="weighted_csls", rerank=False, use_activation=False),
    "+best": dict(score_fusion="weighted_best", rerank=False, use_activation=False),
}


def _reciprocal_rank(results: list[dict], expected: str, *, match: str = "substring") -> tuple[float, int | None]:
    """match='substring' (default): expected is a content substring (legacy
    synthetic corpus). match='hash': expected is a full content_hash (real DB
    corpus from build_real_corpus.py)."""
    if match == "hash":
        for i, r in enumerate(results):
            if r.get("content_hash") == expected:
                return 1.0 / (i + 1), i + 1
        return 0.0, None
    needle = expected.lower()
    for i, r in enumerate(results):
        if needle in r["content"].lower():
            return 1.0 / (i + 1), i + 1
    return 0.0, None


def _build_store(tmp: Path) -> MemoryStore:
    """Fresh DB, full schema, populated with corpus."""
    db = tmp / "bench.db"
    s = MemoryStore(db_path=db)
    conn = s._get_conn()
    # Reuse the conftest base schema so the DB matches a real install.
    conftest = importlib.import_module("conftest")
    conn.executescript(conftest._BASE_SCHEMA)
    s._migrate_stats_tables()
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings "
        "USING vec0(content_embedding FLOAT[768] distance_metric=cosine)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
        "USING fts5(content, content='memories', content_rowid='id', "
        "tokenize='porter ascii')"
    )
    for entry in corpus.CORPUS:
        s.store(
            entry.content,
            memory_type=entry.memory_type,
            tags=entry.tags,
            importance=entry.importance,
        )
    return s


def _run_config(
    store: MemoryStore,
    cfg: dict,
    queries: list[tuple[str, str, str]],
    *,
    match: str = "substring",
) -> dict:
    """Execute all queries under one config; return aggregate metrics.

    Each query tuple is (query_text, expected, category).  `match` is
    "substring" (legacy corpus) or "hash" (real DB corpus).
    """
    core_mod.USE_ACTIVATION = bool(cfg["use_activation"])

    rrs: list[float] = []
    per_category: dict[str, list[float]] = {}
    recall_at_5_hits = 0
    top1_hits = 0
    latencies: list[float] = []

    for query, expected, cat in queries:
        t0 = time.perf_counter()
        results = store.search(
            query,
            mode="hybrid",
            limit=10,
            score_fusion=cfg["score_fusion"],
            rerank=cfg["rerank"],
            track_recall=False,
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        rr, rank = _reciprocal_rank(results, expected, match=match)
        rrs.append(rr)
        per_category.setdefault(cat, []).append(rr)
        if rank == 1:
            top1_hits += 1
        if rank is not None and rank <= 5:
            recall_at_5_hits += 1

    n = len(queries)
    return {
        "mrr@10": sum(rrs) / n,
        "recall@5": recall_at_5_hits / n,
        "top1": top1_hits / n,
        "p50_ms": statistics.median(latencies),
        "p95_ms": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies),
        "n_queries": n,
        "by_category_mrr": {c: sum(v) / len(v) for c, v in per_category.items()},
    }


def _format_table(rows: dict[str, dict]) -> str:
    """Render a markdown table with deltas vs baseline."""
    base = rows.get("baseline", {})

    def _delta(cur: float, b: float, pp: bool = True) -> str:
        d = cur - b
        if pp:
            return f"{d * 100:+.1f}pp" if abs(d) > 1e-6 else "    —"
        return f"{d:+.1f}ms" if abs(d) > 0.01 else "    —"

    lines = [
        "",
        "| config    | MRR@10 | Recall@5 | Top1 | p50 ms | p95 ms | ΔMRR | ΔRecall@5 |",
        "|-----------|--------|----------|------|--------|--------|------|-----------|",
    ]
    for name, m in rows.items():
        lines.append(
            f"| {name:<9} | {m['mrr@10']:.3f}  | {m['recall@5']:.3f}    "
            f"| {m['top1']:.2f} | {m['p50_ms']:6.1f} | {m['p95_ms']:6.1f} "
            f"| {_delta(m['mrr@10'], base.get('mrr@10', m['mrr@10']))} "
            f"| {_delta(m['recall@5'], base.get('recall@5', m['recall@5']))} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="baseline,+rrf,+rerank,+all",
        help="Comma-separated subset of configs to run.",
    )
    parser.add_argument("--json", default=None, help="Write JSON results to this path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap number of queries (0 = all).  Useful for smoke runs.",
    )
    parser.add_argument(
        "--real-corpus",
        default=None,
        help="Path to JSON corpus built by tests/build_real_corpus.py (matches by content_hash).",
    )
    parser.add_argument(
        "--use-prod-db",
        action="store_true",
        help="Replay against a snapshot copy of the production DB (only when --real-corpus is set).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to an existing SQLite DB to replay against directly (no snapshot). "
        "Useful for before/after comparisons on the same mutated snapshot.",
    )
    args = parser.parse_args()

    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CONFIGS]
    if unknown:
        print(f"unknown configs: {unknown}; choices: {list(CONFIGS)}", file=sys.stderr)
        return 2

    if args.real_corpus:
        payload = json.loads(Path(args.real_corpus).read_text())
        queries = [(q["query"], q["expected_hash"], q.get("category", "real")) for q in payload["queries"]]
        match = "hash"
        print(f"Loaded {len(queries)} real-corpus queries from {args.real_corpus}")
    else:
        queries = [(tc.query, tc.expected_top, getattr(tc, "category", "general")) for tc in corpus.ALL_TEST_CASES if tc.expected_top]
        match = "substring"

    if args.limit:
        queries = queries[: args.limit]

    print(f"Replay over {len(queries)} queries × {len(selected)} configs")

    results: dict[str, dict] = {}
    with TemporaryDirectory() as td:
        if args.db:
            store = MemoryStore(db_path=Path(args.db))
        elif args.use_prod_db and args.real_corpus:
            # Snapshot the live DB to avoid mutating it during replay.
            import shutil
            from memory.core import DB_PATH as _DB
            snap = Path(td) / "snapshot.db"
            shutil.copy2(_DB, snap)
            store = MemoryStore(db_path=snap)
        else:
            store = _build_store(Path(td))
        for name in selected:
            t0 = time.perf_counter()
            results[name] = _run_config(store, CONFIGS[name], queries, match=match)
            results[name]["wall_s"] = time.perf_counter() - t0
            print(
                f"  {name:<9} MRR={results[name]['mrr@10']:.3f} "
                f"R@5={results[name]['recall@5']:.3f} "
                f"top1={results[name]['top1']:.2f} "
                f"p50={results[name]['p50_ms']:.1f}ms "
                f"wall={results[name]['wall_s']:.1f}s"
            )

    print(_format_table(results))

    # Per-category MRR for the +all config gives us a "where does each phase help"
    # view that the aggregate hides.
    if "+all" in results and results["+all"].get("by_category_mrr"):
        print("\nPer-category MRR (config: +all):")
        for cat, m in sorted(results["+all"]["by_category_mrr"].items()):
            base_cat = results.get("baseline", {}).get("by_category_mrr", {}).get(cat)
            delta = f"  Δ={m - base_cat:+.3f}" if base_cat is not None else ""
            print(f"  {cat:<14} MRR={m:.3f}{delta}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
