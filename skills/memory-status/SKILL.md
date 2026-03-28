---
name: memory-status
description: Check memory service health and stats.
allowed-tools: Bash
---

# /memory-status

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
