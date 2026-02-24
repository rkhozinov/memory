"""Benchmark: FTS5/BM25 vs semantic search.

Measures cold-start latency (subprocess), warm in-process latency,
and result overlap (Jaccard similarity of top-10 hits).

Usage:
    uv run python benchmarks/bench_search.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory.core import MemoryStore

QUERIES = [
    "embedding model cold start",
    "sqlite vector search cosine distance",
    "confidence decay rate",
    "memory deduplication threshold",
    "BM25 full text search",
    "session briefing budget",
    "composite scoring weights similarity importance recency",
    "tag filter search results",
    "ONNX inference thread",
    "soft delete deleted_at",
]

MODES = ["semantic", "fts"]
COLD_REPS = 3
WARM_REPS = 5


def _cold_start(query: str, mode: str) -> float:
    """Measure wall time for a single CLI invocation (fresh process)."""
    t0 = time.perf_counter()
    subprocess.run(
        ["memory", "search", query, "--mode", mode, "-n", "10"],
        capture_output=True,
        text=True,
    )
    return time.perf_counter() - t0


def _warm_search(store: MemoryStore, query: str, mode: str) -> tuple[float, list[str]]:
    """Time a single in-process search; return (seconds, list of content_hashes)."""
    t0 = time.perf_counter()
    results = store.search(query, mode=mode, limit=10)
    elapsed = time.perf_counter() - t0
    hashes = [r["content_hash"] for r in results]
    return elapsed, hashes


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def _fmt_table(rows: list[tuple], headers: list[str]) -> str:
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [sep, hdr, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |")
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    store = MemoryStore()
    try:
        _run(store)
    finally:
        store.close()


def _run(store: MemoryStore) -> None:
    print("=" * 60)
    print("memory search benchmark")
    print("=" * 60)
    print(f"Queries: {len(QUERIES)}  |  Cold reps: {COLD_REPS}  |  Warm reps: {WARM_REPS}")
    print()

    # ------------------------------------------------------------------ #
    # A. Cold start (subprocess)                                          #
    # ------------------------------------------------------------------ #
    print("=== A. Cold-start latency (subprocess, avg over reps) ===")
    cold_rows = []
    for query in QUERIES:
        row = [query[:40]]
        for mode in MODES:
            times = [_cold_start(query, mode) for _ in range(COLD_REPS)]
            avg = sum(times) / len(times)
            row.append(f"{avg * 1000:.0f}ms")
        cold_rows.append(tuple(row))

    headers = ["query"] + [f"{m} (cold)" for m in MODES]
    print(_fmt_table(cold_rows, headers))
    print()

    # ------------------------------------------------------------------ #
    # B. Warm in-process latency                                          #
    # ------------------------------------------------------------------ #
    print("=== B. Warm in-process latency (avg over reps) ===")
    warm_rows = []
    overlap_rows = []
    for query in QUERIES:
        warm_row = [query[:40]]
        hashes_by_mode: dict[str, list[str]] = {}
        for mode in MODES:
            times = []
            last_hashes: list[str] = []
            for _ in range(WARM_REPS):
                elapsed, hashes = _warm_search(store, query, mode)
                times.append(elapsed)
                last_hashes = hashes
            avg = sum(times) / len(times)
            warm_row.append(f"{avg * 1000:.1f}ms")
            hashes_by_mode[mode] = last_hashes
        warm_rows.append(tuple(warm_row))

        # Jaccard between semantic and fts
        if len(MODES) >= 2:
            j = _jaccard(hashes_by_mode.get(MODES[0], []), hashes_by_mode.get(MODES[1], []))
            overlap_rows.append((query[:40], f"{j:.2f}"))

    headers = ["query"] + [f"{m} (warm)" for m in MODES]
    print(_fmt_table(warm_rows, headers))
    print()

    # ------------------------------------------------------------------ #
    # C. Result overlap (Jaccard of top-10)                               #
    # ------------------------------------------------------------------ #
    if overlap_rows:
        print(f"=== C. Result overlap — Jaccard(top-10 {MODES[0]} ∩ {MODES[1]}) ===")
        print(_fmt_table(overlap_rows, ["query", "jaccard"]))
        jaccards = [float(r[1]) for r in overlap_rows]
        print(f"\nMean Jaccard: {sum(jaccards) / len(jaccards):.2f}")
    print()


if __name__ == "__main__":
    main()
