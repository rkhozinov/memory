#!/usr/bin/env python3
"""P5 — probe: would a diversity term improve what the hook injects?

Spec: docs/research/consolidation-redesign.md, "Method probes", P5.

The finding this tests: retrieval score does not predict usefulness and inverts
at the top. Over 648 real injected memories, the >=0.70 score band had a 0.0%
observed reuse rate while >=0.60 had 22.2%. The reading is that a memory very
close to the prompt restates what the agent just read.

If that reading is right, selecting the five injected lines by score alone is
the wrong objective, and an MMR-style trade of score against novelty should do
better. This simulates that offline against the injections already on disk.

Honest limit, and it is not a small one: this reuses the same injections that
produced the hypothesis. It can show internal consistency. It cannot validate.

Usage:
  uv run python benchmarks/probe_p5_injection_diversity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import session_analytics as sa  # noqa: E402

TOP_K = 5  # the hook injects five lines
POOL_SIZE = 20  # reconstructed candidate pool per injection


def _mmr_select(items: list[dict], prompt_tokens: set[str], k: int, lam: float) -> list[dict]:
    """Greedy MMR over injected candidates.

    relevance = the retrieval score the hook already computed
    redundancy = token overlap with the prompt, and with everything already picked

    Overlap is lexical rather than embedding-based on purpose: the hook runs under
    a 4-second timeout and cannot afford to embed its own candidates.
    """
    chosen: list[dict] = []
    pool = list(items)
    chosen_tokens: set[str] = set()
    while pool and len(chosen) < k:
        best = None
        best_val = -1e9
        for cand in pool:
            toks = cand["tokens"]
            if not toks:
                red = 0.0
            else:
                vs_prompt = len(toks & prompt_tokens) / len(toks)
                vs_chosen = len(toks & chosen_tokens) / len(toks) if chosen_tokens else 0.0
                red = max(vs_prompt, vs_chosen)
            val = lam * cand["score"] - (1 - lam) * red
            if val > best_val:
                best_val, best = val, cand
        chosen.append(best)
        chosen_tokens |= best["tokens"]
        pool.remove(best)
    return chosen


def run(projects_dir: Path, lambdas: tuple[float, ...], db: str | None = None) -> dict:
    scans = []
    for f in sorted(projects_dir.rglob("*.jsonl")):
        try:
            s = sa.scan_transcript(f)
        except OSError:
            continue
        if s and s.injections:
            scans.append(s)

    # The transcript records only what was INJECTED — at most five lines, because
    # the hook runs with --limit 5. There is no candidate pool in the log, so the
    # first cut of this probe had every selector tie at 100% (including the
    # oracle): with five candidates and five slots there is nothing to select.
    #
    # So the pool has to be reconstructed by re-running retrieval against the DB
    # with the original prompt at a higher limit. That is a reconstruction, not a
    # replay: the corpus has changed since, and any memory written after the
    # injection is one the hook could never have chosen. Results are directional.
    from memory.core import MemoryStore

    store = MemoryStore(db_path=Path(db)) if db else MemoryStore()

    episodes = []
    for scan in scans:
        for inj in scan.injections:
            # What the assistant went on to write, which is the outcome signal.
            later: set[str] = set()
            for ts, toks in scan.assistant_turns:
                if ts > inj.timestamp:
                    later |= toks
            before: set[str] = set()
            for ts, toks in scan.context_turns:
                if ts <= inj.timestamp:
                    before |= toks
            if not inj.prompt:
                continue
            try:
                pool = store.search(
                    inj.prompt[:512], mode="hybrid", limit=POOL_SIZE, track_recall=False, min_score=None
                )
            except TypeError:
                pool = store.search(inj.prompt[:512], mode="hybrid", limit=POOL_SIZE, track_recall=False)
            except Exception:
                continue
            cands = []
            for r in pool:
                toks = sa._tokens((r.get("content") or "")[:200])
                cands.append(
                    {
                        "hash": r.get("content_hash", ""),
                        "type": r.get("memory_type", ""),
                        "score": float(r.get("score") or 0.0),
                        "tokens": toks,
                        "reused": len((toks - before) & later),
                    }
                )
            if len(cands) > TOP_K:
                episodes.append({"candidates": cands, "prompt_tokens": before})

    def evaluate(selector) -> dict:
        picked_reuse = 0
        picked_total = 0
        redundancy = []
        used_episodes = 0
        for ep in episodes:
            sel = selector(ep)
            if not sel:
                continue
            used_episodes += 1
            picked_reuse += sum(1 for c in sel if c["reused"] >= 2)
            picked_total += len(sel)
            # Mean pairwise token overlap among the selected lines.
            pw = []
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    a, b = sel[i]["tokens"], sel[j]["tokens"]
                    u = len(a | b)
                    pw.append(len(a & b) / u if u else 0.0)
            if pw:
                redundancy.append(sum(pw) / len(pw))
        return {
            "episodes": used_episodes,
            "lines_selected": picked_total,
            "useful_line_rate": picked_reuse / picked_total if picked_total else 0.0,
            "mean_redundancy": sum(redundancy) / len(redundancy) if redundancy else 0.0,
        }

    out = {
        "episodes_available": len(episodes),
        "current_top_k_by_score": evaluate(lambda ep: sorted(ep["candidates"], key=lambda c: -c["score"])[:TOP_K]),
    }
    for lam in lambdas:
        out[f"mmr_lambda_{lam}"] = evaluate(
            lambda ep, lam=lam: _mmr_select(ep["candidates"], ep["prompt_tokens"], TOP_K, lam)
        )
    # An upper bound: pick the lines that actually turned out useful. Not
    # achievable, but it says how much headroom any selector could have.
    out["oracle"] = evaluate(lambda ep: sorted(ep["candidates"], key=lambda c: -c["reused"])[:TOP_K])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects-dir", default=str(sa.DEFAULT_PROJECTS_DIR))
    ap.add_argument("--db", default=None, help="Snapshot DB to reconstruct pools against.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = run(Path(args.projects_dir), (0.9, 0.7, 0.5, 0.3), db=args.db)
    print(f"\nP5 — injection selection, simulated over {res['episodes_available']} real injections\n")
    print(f"  {'selector':<26} {'lines':>7} {'useful':>8} {'redundancy':>12}")
    for key, val in res.items():
        if not isinstance(val, dict):
            continue
        print(f"  {key:<26} {val['lines_selected']:>7} {val['useful_line_rate']:>7.1%} {val['mean_redundancy']:>12.3f}")
    print("\n  Adoption threshold: MMR must raise the useful-line rate, not merely cut redundancy.")
    print("  Limit: same injections that produced the hypothesis — consistency, not validation.\n")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
