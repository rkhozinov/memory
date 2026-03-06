---
name: forget
description: Find and delete specific memories. Use when user says "forget", "delete memory", or wants to clean up stored info.
argument-hint: "[description of what to forget]"
allowed-tools: Bash, AskUserQuestion
---

# /forget

1. **Search** — search both memories and documents:
   ```bash
   echo "=== Memories ===" && memory -f text search "$ARGUMENTS" --limit 10 && echo "=== Documents ===" && memory -f text doc search "$ARGUMENTS" --limit 5
   ```
2. **Show** — present the results (each line starts with a 16-char hash prefix). Clearly label which are memories vs documents.
3. **Confirm** — ask user which to delete (by number, "all", or "cancel"). **Never delete without confirmation.**
4. **Delete** — use the appropriate command based on type:
   - Memories: `memory -f text delete <hash_prefix>`
   - Documents: `memory -f text doc delete <hash_prefix>`
5. **Report** — what was deleted

## Batch delete

For bulk cleanup by tags or date range, always preview first:

```bash
# Preview what would be deleted
memory -f text delete --tags "project:old-project" --dry-run

# Delete all memories with specific tags
memory -f text delete --tags "project:old-project"

# Delete memories older than a date
memory -f text delete --before 2025-01-01 --dry-run

# Combine filters
memory -f text delete --tags "scope:temp" --before 2025-06-01
```

## Purge (hard-delete)

After soft-deleting, entries remain in the DB (blocking re-insertion with same content). To permanently remove old deleted entries:

```bash
# Preview entries that would be purged (deleted >30 days ago)
memory -f text purge --dry-run

# Purge with custom retention
memory -f text purge --retention-days 7

# Default: purge entries deleted >30 days ago
memory -f text purge
```
