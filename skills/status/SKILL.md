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
memory admin consolidate --dry-run            # preview near-duplicate merges
memory admin consolidate                      # merge >0.92 similarity
memory admin decay --min-confidence 0.3       # decay + prune
memory admin purge --dry-run                  # preview hard-deletes
memory admin purge --retention-days 7         # hard-delete old entries
```

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

### Index regeneration

Rebuild the curated front-door TOC (table of contents) for the memory store:
```bash
memory admin index             # rebuild index
memory admin index --dry-run   # preview index entries
```
The index provides a fast navigable overview of all stored topics and is updated automatically on dream passes.

### Auto-archived sessions

Auto-archived sessions live in the doc store with tag `session-archive`; search them via `memory doc search`.
