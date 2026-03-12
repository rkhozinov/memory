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

## Search modes

Use `--mode` to control how search works:

| Mode | When to use | Example |
|------|------------|---------|
| `semantic` (default) | Find by meaning — best for general queries | `memory search "deployment process"` |
| `exact` | Substring match — when you know the exact phrase | `memory search "UNIQUE constraint" --mode exact` |
| `fts` | Full-text keyword search — technical terms, specific identifiers | `memory search "terraform backend" --mode fts` |
| `hybrid` | Combined semantic + exact | `memory search "IAM policy" --mode hybrid` |

For documents, modes are `semantic`, `fts`, and `auto` (default, combines both).

## Advanced search filters

```bash
# Exclude specific tags from results
memory -f text search "query" --exclude-tags "scope:temp,scope:draft"

# Filter by minimum importance threshold
memory -f text search "query" --min-importance 0.5

# Filter to specific memory types (comma-separated)
memory -f text search "query" --types "decision,pattern"

# Combine all filters
memory -f text search "query" --tags "project:infra" --exclude-tags "scope:temp" --min-importance 0.7 --types "decision,error"

# View score breakdown (similarity, importance, recency components)
memory -f text search "query" --depth full
```

## Cross-encoder reranking

Add `--rerank` to re-score results with a TinyBERT cross-encoder for higher precision. The reranker jointly encodes (query, candidate) pairs, capturing fine-grained interactions that bi-encoder similarity misses. Adds ~30ms latency.

When `--rerank` is active, semantic/fts modes are automatically escalated to hybrid (merging both candidate pools), and scoring weights shift to (0.8, 0.1, 0.1) to favor similarity over importance/recency.

```bash
# Basic reranking
memory -f text search "kubernetes deployment" --rerank

# Adjust blend weight (0=ignore reranker, 1=only reranker, default 0.4)
memory -f text search "query" --rerank --rerank-weight 0.6

# View reranker score in breakdown
memory -f text search "query" --rerank --depth full
```

Reranking is ignored for `exact` and `graph` modes.

## Batch search

Search for multiple queries in a single pass (one embedding batch, one recall transaction):

```bash
echo '["query one", "query two", "query three"]' | memory -f text search-batch --limit 5
# Or from a file:
memory -f text search-batch --file queries.json --limit 5 --tags "project:infra"
```

## Time filtering

Filter results by date:
```bash
memory -f text search "query" --time-expr "last 7 days"
memory -f text search "query" --after 2025-01-01 --before 2025-03-01
```

## Tag discovery

List all tags and their frequencies to find the right filter:
```bash
memory -f text list-tags
```

## Reading document content

Search results show only summaries. To read the full body of a document:
1. Find it: `memory -f text doc search "query"`
2. Read it: `memory -f text doc get <hash>`
