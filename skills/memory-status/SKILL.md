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
