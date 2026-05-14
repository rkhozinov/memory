---
name: status
description: Check memory service health, stats, demoted memories, dream consolidation, index, and session archives.
allowed-tools: Bash
---

# /memory:status

Run all commands and report:
```bash
memory health && memory admin stats --top-recalled 10 --stale && memory doc list --page-size 50
```

All output is JSON. Parse inline for readable output.

## Optional flags

- `--after YYYY-MM-DD` / `--before YYYY-MM-DD` — date range for stats
- `--top-recalled N` — most recalled memories
- `--never-recalled` — memories never retrieved
- `--stale` — old, never-recalled memories

## Briefing

Quick ranked overview:
```bash
memory admin briefing --budget 50
```

Default ranker: `confidence × importance × recency`.
Opt-in activation ranker (v1.7.0+) — rewards type permanence, fresh recall,
and diverse-session usage:
```bash
MEMORY_USE_ACTIVATION=1 memory admin briefing --budget 50
```
Briefing also filters out injection-flagged memories on read.

## Maintenance

### Tags
```bash
memory admin tags list
memory admin tags rename "old:tag" "new:tag"
memory admin tags merge "tag1,tag2" "merged:tag"
```

### Cleanup
```bash
memory admin cleanup                          # remove exact duplicates
memory admin consolidate --dry-run            # preview pairwise near-duplicate merges
memory admin consolidate                      # merge >0.92 similarity (pairwise)
memory admin decay --min-confidence 0.3       # decay + prune
memory admin purge --dry-run                  # preview hard-deletes
memory admin purge --retention-days 7         # hard-delete old entries
```

### Cluster-based dedup (v1.5.0+)

Pairwise consolidate at default 0.92 only catches obvious clones.  Lower
thresholds are unsafe pairwise (chain merges).  Cluster mode uses union-find
on connected components so 0.85 becomes safe:

```bash
memory admin clusters --threshold 0.85                       # one-line per cluster (default summary)
memory admin clusters --threshold 0.85 --depth full          # full JSON dump with all members
memory admin clusters --threshold 0.85 --query cloudfront    # scope to substring
memory admin clusters --threshold 0.85 --tag project:foo     # scope to exact tag
memory admin consolidate --cluster --threshold 0.85 \
  --strategy mmr_union --dry-run                             # preview
memory admin consolidate --cluster --threshold 0.85 \
  --strategy mmr_union                                       # apply (recommended)
```

Strategies for the survivor's content:
- `mmr_union` (v1.6.0+, recommended) — MMR extractive merge across all member
  sentences; preserved ~97% chars in bench vs concat's ~12%
- `keep_higher_recall` (legacy default) — survivor content unchanged
- `keep_longer` — replace with longest member's content
- `concat` — append a `Related (merged):` block with one-line snippets only
  (lossy; use `mmr_union` instead)

Project-scoped by default — only memories sharing a `project:*` tag merge.
Use `--no-project-scope` to allow cross-project merges (rarely wanted).

Provenance: every merge writes a `merged_into` edge to `memory_graph`
before soft-deleting the loser. Lineage is queryable via raw SQL or via
`memory admin undelete <hash>` (see `/forget` skill) to recover individual
rows if a pass goes wrong.

### Export/Import
```bash
memory admin export -o /tmp/backup.json
memory admin import -f /tmp/backup.json
```

### Demoted memories

List memories that have low recall counts and have been ranked down:
```bash
memory admin demoted
```
Use this to identify candidates for deletion or promotion. Demoted entries are retained but ranked lower in search results.

### Dream pass (composite consolidation)

A "dream" pass reads all memories and merges related clusters into composite summaries:
```bash
memory admin dream --dry-run   # preview what would be merged
memory admin dream             # run consolidation
```
Run this periodically (e.g., after large ingestion) to reduce redundancy and improve search quality.

**Auto-runs on SessionEnd** (v1.7.0+): the `memory-session-end.sh` hook
fires dream in the background after each Claude Code session, throttled to
once every 6h. Marker file: `~/.claude/memory/dream/last_run.marker`.
Override interval: `MEMORY_DREAM_INTERVAL_HOURS=N` (set `0` to always run).

The dream pass output includes:
- `superseded` / `superseded_pairs` — older memories replaced by newer; provenance
  edge of type `supersedes` is written to `memory_graph` before soft-delete.
- `consolidated` — near-duplicate merges; `merged_into` edges written.
- `demoted` — old, never-recalled, auto-tagged memories flagged (not deleted).
- `forgotten` / `forgotten_pairs` — Phase C active forgetting (env-gated; see below).

### Active forgetting (Phase C, env-gated)

Pass 6 soft-deletes low-activation memories. Off by default; opt in:
```bash
MEMORY_ACTIVE_FORGET=1 memory admin dream --dry-run    # preview candidates
MEMORY_ACTIVE_FORGET=1 memory admin dream              # apply
```
Criteria: activation < 0.05 AND distinct_session_count ≤ 1 AND age > 30 days AND
memory_type NOT IN (decision, reference). Budget cap: max 5% of active corpus
per pass. Tune via `MEMORY_FORGET_THRESHOLD`, `MEMORY_ACTIVATION_TAU`.

### Activation scoring (Phase C, opt-in)

Every search result carries an `activation` field. To make activation the primary
ranking signal (replaces composite/RRF score on hybrid/semantic/fts):
```bash
MEMORY_USE_ACTIVATION=1 memory search "query"
```
Rerank (`--rerank`) always uses activation as the base score for blending.

### Index regeneration

Rebuild the curated front-door TOC (table of contents) for the memory store:
```bash
memory admin index             # rebuild index
memory admin index --dry-run   # preview index entries
```
The index provides a fast navigable overview of all stored topics and is updated automatically on dream passes.

### Auto-archived sessions

Auto-archived sessions live in the doc store with tag `session-archive`; search them via `memory doc search`.
