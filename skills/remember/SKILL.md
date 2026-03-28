---
name: remember
description: Extract and store key facts, decisions, and patterns from conversation into persistent memory. Use when user says "remember", "save this", or important info should persist.
argument-hint: "[what to remember] [--global for cross-project]"
allowed-tools: Bash
---

# /remember

Set `PROJECT=$(basename "$(pwd)")` before commands.

## Scope

- If `$ARGUMENTS` contains "global" or "--global": tag with `scope:global`
- Otherwise: tag with `project:$PROJECT`

## Routing: Memories vs Documents

| Route | When | Command |
|-------|------|---------|
| **Memory** | Short, atomic facts ≤800 chars | `memory store` |
| **Document** | Long-form >800 chars, multi-section | `memory doc store` |

Route automatically — never ask.

## Storing memories

Single:
```bash
memory store "<content>" --tags "project:$PROJECT,<topic>" --type <type> --dedup 0.90
```

JSON (single or batch):
```bash
memory store '[{"content":"fact 1","type":"decision","tags":["project:X"]},{"content":"fact 2","type":"learning","tags":["project:X"]}]'
```

## Types

| Type | Use for |
|------|---------|
| `decision` | Architecture choices, approved approaches |
| `pattern` | Conventions, recurring solutions |
| `error` | Root causes, fix recipes |
| `learning` | Non-obvious discoveries |
| `reference` | Pointers to external resources |
| `observation` | Status snapshots, context |
| `note` | General (default) |

## Tags

Standard taxonomy: `project:<name>`, `scope:global`, `cloud:<provider>`, `svc:<service>`, `tool:<tool>`.

## Storing documents

```bash
memory doc store --title "Title" --summary "Brief summary for search" --body "Full body text" --tags "project:$PROJECT" --type plan
```

Or from stdin:
```bash
cat doc.md | memory doc store --title "Title" --summary "Summary" --body-file - --tags "project:$PROJECT"
```

## Deduplication

Always pass `--dedup 0.90` when storing memories. This skips storage if a near-duplicate (>90% similarity) already exists with the same type and tags. Omit for documents (they use content hash dedup).

## Importance

Set `--importance 0.9` for critical facts. Auto-inferred from keywords (CRITICAL→0.9, IMPORTANT→0.8) and type (decision→0.8, error→0.7) if not set.
