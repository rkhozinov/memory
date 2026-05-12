---
name: forget
description: Find and delete specific memories. Use when user says "forget", "delete memory", or wants to clean up stored info.
argument-hint: "[description of what to forget]"
allowed-tools: Bash, AskUserQuestion
---

# /forget

1. **Search** — search both memories and documents:
   ```bash
   memory search "$ARGUMENTS" --limit 10 && memory doc search "$ARGUMENTS" --limit 5
   ```
2. **Show** — present results with hash prefix. Parse JSON for display.
3. **Confirm** — ask user which to delete (by number, "all", or "cancel"). **Never delete without confirmation.**
4. **Delete** — partial hash prefixes work:
   - Memories: `memory delete <hash_prefix>`
   - Documents: `memory doc delete <hash_prefix>`
5. **Report** — what was deleted

## Batch delete

Preview first, then delete:
```bash
# Preview
memory delete --tags "project:old-project" --dry-run

# Delete by tags
memory delete --tags "project:old-project"

# Delete by date
memory delete --before 2025-01-01 --dry-run
```

## Purge (hard-delete)

Permanently remove old soft-deleted entries:
```bash
memory admin purge --dry-run           # preview
memory admin purge --retention-days 7  # hard-delete
```

## Auto-extracted entries

Memories tagged `source:auto` were ingested automatically (e.g., from session handoffs or `memory admin auto-extract`).
To bulk-preview and delete them:
```bash
# Preview
memory delete --tags "source:auto" --dry-run

# Bulk delete all auto-extracted entries
memory delete --tags "source:auto"
```
Confirm the dry-run count before proceeding — bulk deletes cannot be undone (soft-deleted entries can be purged).
