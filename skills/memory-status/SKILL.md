---
name: memory-status
description: Check memory service health and stats.
allowed-tools: Bash
---

# /memory-status

Run all commands and report the combined output:

```bash
memory -f text health && echo "---" && memory -f text stats --top-recalled 10 --stale && echo "=== Documents ===" && memory -f text doc list --page-size 50
```

The first command (`health`) returns: service status, total count, type breakdown, and database size.

The second command (`stats`) returns: usage analytics, the 10 most frequently recalled memories, and stale memories (old, never recalled — cleanup candidates).

The third command (`doc list`) shows all stored documents (plans, specs, runbooks, session summaries).

## Optional flags for deeper analysis

If the user asks for a specific date range or different limits, use these flags on `memory stats`:

- `--after YYYY-MM-DD` — filter to events after a date
- `--before YYYY-MM-DD` — filter to events before a date
- `--top-recalled N` — show N most recalled memories (default: 10)
- `--never-recalled` — show all memories that were never retrieved
- `--stale` — show old, never-recalled memories

## Briefing

For a quick overview of top memories ranked by importance and recency:

```bash
memory -f text briefing [--budget N]
```

Default budget is 150 lines. Use `--budget 50` for a compact summary.

## Maintenance

### Tag discovery
```bash
memory -f text list-tags
```
Shows all unique tags with frequency counts — useful for finding tag inconsistencies or discovering available filters.

### Tag management
```bash
# Rename a tag across all memories and documents
memory -f text rename-tag "old:tag" "new:tag"

# Merge multiple tags into one
memory -f text merge-tags "tag1,tag2,tag3" "merged:tag"
```

### Export/Import (backup & restore)
```bash
# Export all memories and documents to JSON
memory -f json export --output /tmp/memory-backup.json

# Export without documents
memory -f json export --output /tmp/memory-backup.json --no-documents

# Import from a backup file
memory -f text import --file /tmp/memory-backup.json

# Force import (skip dedup checks)
memory -f text import --file /tmp/memory-backup.json --force
```

### Cleanup duplicate entries
```bash
memory -f text cleanup
```
Removes exact-hash duplicates (keeps the first entry).

### Consolidate near-duplicates
```bash
# Preview what would be merged
memory -f text consolidate --dry-run

# Merge memories with >0.92 similarity (default threshold)
memory -f text consolidate

# Custom threshold, exclude specific types
memory -f text consolidate --threshold 0.90 --exclude-types "reference,decision"
```
Merges near-duplicate memories: keeps the older one, unions tags from both, soft-deletes the duplicate.

### Confidence decay
```bash
# Apply decay (no pruning)
memory -f text decay

# Apply decay and prune memories below threshold
memory -f text decay --min-confidence 0.3
```
Decays confidence scores based on memory type (decisions decay slowest, observations fastest). Optionally prunes below a threshold.

### Purge soft-deleted entries
```bash
# Preview entries deleted >30 days ago
memory -f text purge --dry-run

# Hard-delete with custom retention
memory -f text purge --retention-days 7
```
Permanently removes soft-deleted entries from the database to reclaim space and prevent UNIQUE constraint issues.
