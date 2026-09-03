# Benchmark results

Committed retrieval-quality baselines. Before this directory existed the repo had
exactly one committed benchmark result (`tests/bench_results.json`, Feb 2026,
embedding *runtime* only) and zero retrieval-quality results — so every ranking
decision made up to that point was unreproducible.

| File | What |
|---|---|
| `replay-synthetic-baseline.json` | `tests/bench_replay.py` over `benchmarks/corpus.py`, 23 queries / 31 memories |
| `replay-proddb-baseline.json` | Same harness over a snapshot of the production DB, 200 queries / 6767 memories |
| `consolidate-baseline.json` | `tests/bench_consolidate.py` — fact retention per `content_strategy` |
| `real-corpus-*.json` | **Gitignored, never committed.** The generated query set behind the prod-db run. It is built from live memory content and a scan found 25 employer references in the 2026-09-03 build, so it stays local. Regenerate it with the commands below. |

Each JSON carries a `_meta` block naming the corpus, the sample size and the known
sampling bias. Read it before quoting a number.

## Reproducing

```
cp data/sqlite_vec.db /tmp/memsnap.db
uv run python tests/build_real_corpus.py --db /tmp/memsnap.db --out /tmp/real_corpus.json --max 200
uv run python tests/bench_replay.py --db /tmp/memsnap.db --real-corpus /tmp/real_corpus.json \
    --configs baseline,+rrf,+rrsb,+id,+csls,+best,+rerank,+rrsb+rr,+all \
    --json benchmarks/results/replay-proddb-baseline.json
```

Always replay against a **copy**. `--use-prod-db` snapshots for you; `--db` does not.

The generated `real_corpus.json` embeds verbatim memory content and query text
from the production DB, which is a mix of personal and work material. Keep it in
`/tmp`. Only the aggregate metrics files in this directory are safe to commit.

Before committing anything generated from the live DB, run the publish-hygiene
scan defined in the repo's local-only rules over `benchmarks/results/` and require
zero hits.

## Headline results, 2026-09-03

| config | MRR@10 | nDCG@10 | R@1 | p50 ms | ΔMRR |
|---|---|---|---|---|---|
| baseline (`weighted`) | 0.926 | 0.944 | 0.88 | 6.3 | — |
| `+rrf` | 0.765 | 0.824 | 0.57 | 6.2 | **-16.1pp** |
| `+rrsb` | 0.777 | 0.830 | 0.60 | 6.2 | -14.9pp |
| `+id` | 0.936 | 0.952 | 0.89 | 6.4 | +1.0pp |
| `+csls` | 0.963 | 0.972 | 0.94 | 6.2 | +3.7pp |
| **`+best` (shipping default)** | **0.970** | **0.978** | **0.94** | 6.2 | **+4.4pp** |
| `+rerank` | 0.939 | 0.953 | 0.90 | 714 cold | +1.3pp |

Three things follow.

1. **Do not swap `weighted_best` for RRF.** RRF is the textbook default and it is
   16 points of MRR worse on this corpus. The additive-weighted path plus CSLS
   hubness correction wins, and it wins by a lot.
2. **The cross-encoder rerank is not worth its cost.** It adds 1.3pp where
   `weighted_best` adds 4.4pp, and the first uncached call costs 714 ms p50
   (1529 ms p95) against a 6 ms non-rerank path. The 6 ms figures shown for
   `+rerank`/`+all` in a repeat run are cache hits, not the real cost. Keeping it
   opt-in is correct; retiring it is worth considering.
3. **The synthetic corpus is exhausted for ranking work.** All six non-rerank
   configs tie at MRR 0.913 on it. It still earns its keep as per-category
   coverage (identifier, importance-trap, boolean, stale-fact), not as a ranker
   discriminator.

## Known bias

`build_real_corpus.py` validates each generated `(query, gold)` pair by checking
the **baseline** hybrid config can already retrieve it, and discards the rest.
That has two consequences worth stating whenever these numbers are quoted:

- Baseline is flattered. Its true recall over an unfiltered query set is lower.
- Hard cases vanish. All 75 generated `hot_cluster` candidates — the family that
  deliberately stresses over-recalled memories crowding out a correct low-recall
  sibling — failed validation and appear nowhere in the 200. The absence is not
  evidence that hot-cluster crowding is solved; it is evidence that it is total.

## Consolidation fact retention, 2026-09-03

`tests/bench_consolidate.py`. Three synthetic clusters, 8 memories, 8 unique facts.

| strategy | merged | memories after | facts kept | retention |
|---|---|---|---|---|
| **`keep_higher_recall` (shipping default)** | 3 | 5 | 5/8 | **62%** |
| `keep_longer` | 3 | 5 | 5/8 | 62% |
| `concat` | 3 | 5 | 8/8 | 100% |
| `mmr_union` | 3 | 5 | 8/8 | 100% |

The default loses 38% of the unique facts in a merged cluster. It keeps the
survivor's text verbatim, so every other member's distinguishing detail — a port
number, a pool mode, a probe delay — is soft-deleted along with the row that held
it. `mmr_union` and `concat` retain all of it at the *same* surviving-memory
count, so this is a choice of default, not a price paid for consolidating.

Note what this does and does not settle. It rules the extractive union strategy
**in**: `mmr_union` already retains 100% here, so there is no measured case for
adding an LLM merge step on fact-retention grounds. It does not tell you whether
`mmr_union`'s stitched-together sentences read well, or whether they re-embed to
a vector that still retrieves — both are open, and both are cheaper to answer
than an LLM.

The 8-fact fixture set is small; treat 62% vs 100% as a clear ordering, not a
precise rate. `test_consolidation_actually_merges` guards the case where the
fixtures stop clustering under a future embedding model and the retention numbers
go vacuously perfect.
