---
name: recall
description: Retrieve relevant memories from persistent storage. Use when user asks "what do you remember", "recall", or when starting work on a familiar topic.
argument-hint: "[optional: query or topic to search for]"
allowed-tools: Bash
---

# /recall

Set `PROJECT=$(basename "$(pwd)")` before commands.

## With query (`/recall <query>`)

Run as a **single Bash call** (avoids multi-call dispatch overhead):
```bash
echo "=== Semantic ===" && memory -f text search "<query>" --limit 10 --depth titles && echo "=== Documents ===" && memory -f text doc search "<query>" --limit 5 && echo "=== Project ===" && memory -f text search "<query>" --mode exact --tags "project:$PROJECT" --limit 20 --depth titles && echo "=== Global ===" && memory -f text search "<query>" --mode exact --tags "scope:global" --limit 20 --depth titles
```

**IMPORTANT**: Always chain memory commands in a single Bash call using `&&`. Never split into separate parallel Bash tool calls — Claude Code serializes them with high dispatch overhead, turning 200ms of work into minutes of waiting.

Use `--depth titles` for compact output. If the user needs full details on a specific result, re-run with `--depth full` and a narrower query. For documents, use `memory doc get <hash>` to read the full body.

## Without query (bare `/recall`)

Run: `memory -f text briefing --budget 80 && echo "=== Documents ===" && memory -f text doc list --page-size 10`

This generates a ranked markdown summary grouped by type (decisions, patterns, errors, learnings, references, recent), followed by a list of stored documents. Much more useful than a raw listing.

If the user needs project-specific or global scope listing, fall back to:
```bash
echo "=== Project ===" && memory -f text list --tags "project:$PROJECT" --page-size 50 && echo "=== Global ===" && memory -f text list --tags "scope:global" --page-size 50 && echo "=== Project Docs ===" && memory -f text doc list --tags "project:$PROJECT" --page-size 10
```

Present results grouped by scope (Project / Global), then by type. Each result line starts with its full hash (for `/forget` reference).

## TODOs shortcut (`/recall todos`)

List all deferred work items tagged with `todo`:
```bash
memory -f text search "TODO" --tags "todo" --limit 20
```

This returns items stored with the `[TODO]` prefix and `todo` tag. Status markers in content (DONE, BLOCKED, PENDING) indicate progress.
