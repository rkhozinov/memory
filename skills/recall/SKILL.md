---
name: recall
description: Retrieve relevant memories from persistent storage. Use when user asks "what do you remember", "recall", or when starting work on a familiar topic.
argument-hint: "[optional: query or topic to search for]"
allowed-tools: Bash
---

# /recall

Set `PROJECT=$(basename "$(pwd)")` before commands. All output is JSON — parse with python/jq inline.

## With query (`/recall <query>`)

Run as a **single Bash call** (avoids multi-call dispatch overhead):
```bash
memory search "<query>" --limit 10 && memory doc search "<query>" --limit 5
```

Parse JSON results inline for readable output:
```bash
memory search "<query>" --limit 10 | python3 -c "
import sys,json
for r in json.load(sys.stdin):
    h=r['content_hash'][:16]; t=r.get('memory_type','?'); s=r.get('score',0)
    print(f'{h} [{t}] score={s:.2f} {r[\"content\"][:200]}')
" && memory doc search "<query>" --limit 5 | python3 -c "
import sys,json
for r in json.load(sys.stdin):
    h=r['content_hash'][:16]; print(f'{h} [doc] {r.get(\"title\",\"\")}')
"
```

## Without query (bare `/recall`)

Run: `memory admin briefing --budget 80 && memory doc list --page-size 10`

## TODOs shortcut (`/recall todos`)

```bash
memory search "TODO" --tags "todo" --limit 20 --mode exact
```

## Search modes

| Mode | When to use |
|------|------------|
| `hybrid` (default) | Best general purpose — combines semantic + keyword |
| `semantic` | Find by meaning |
| `exact` | Substring match |
| `fts` | Full-text keyword search with OR/AND/NOT |
| `graph` | Entity-based traversal via shared entities |

## Cross-encoder reranking

Add `--rerank` to re-score results with cross-encoder (~45ms extra):
```bash
memory search "kubernetes deployment" --rerank
```

## Reading documents

1. Find: `memory doc search "query"`
2. Read full body: `memory doc get <hash>`
