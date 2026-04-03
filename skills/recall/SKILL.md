---
name: recall
description: Retrieve memories and documents from persistent storage. Run the search immediately with your query.
argument-hint: "[query or topic]"
allowed-tools: Bash
---

# /recall

Searches persistent memory and returns results. Set `PROJECT=$(basename "$(pwd)")` before commands.

## Step 1: Execute NOW

### With query (e.g., `/recall embedding model`):

Run this command immediately with `$ARGUMENTS` as your search query:

```bash
PROJECT=$(basename "$(pwd)")
memory search "$ARGUMENTS" --limit 10 | python3 -c "
import sys,json
for r in json.load(sys.stdin):
    h=r['content_hash'][:16]; t=r.get('memory_type','?'); s=r.get('score',0)
    print(f'{h} [{t}] score={s:.2f} {r[\"content\"][:200]}')
" && memory doc search "$ARGUMENTS" --limit 5 | python3 -c "
import sys,json
for r in json.load(sys.stdin):
    h=r['content_hash'][:16]; print(f'{h} [doc] {r.get(\"title\",\"\")}')
"
```

### Without query (bare `/recall`):

Retrieve session briefing and document list:

```bash
PROJECT=$(basename "$(pwd)")
memory admin briefing --budget 80 && memory doc list --page-size 10
```

## Step 2: Interpret Results

Output shows:
- `<hash> [<type>] score=<relevance> <preview...>` — memories (top 10 by relevance)
- `<hash> [doc] <title>` — documents (top 5 by relevance)

Higher score = more relevant.

---

## Search Modes

Default is `hybrid` (semantic + keyword). For advanced searches, add `--mode`:

```bash
memory search "query" --mode semantic   # Semantic similarity only
memory search "query" --mode exact      # Substring match (no fuzzy)
memory search "query" --mode fts        # Full-text keyword search
memory search "query" --mode graph      # Entity-based traversal
```

## Filtering

Add to search command:
```bash
--tags "project:X,tool:Y"   # Filter by tags
--types decision,error       # Filter by memory type
--rerank                     # Re-score with cross-encoder (slower, +45ms)
```

## Reading Full Content

If you find a memory by hash, read it:
```bash
memory get <hash>
```

For documents:
```bash
memory doc get <hash>
```

## TODOs

To see pending todos:
```bash
memory search "TODO" --tags "todo" --limit 20 --mode exact
```
