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

Analyze each piece of content and **automatically route** to the right store:

| Route | When | Store with |
|-------|------|-----------|
| **Memory** (`memory store`) | Short, atomic facts ≤800 chars: a decision, pattern, error, learning, observation | `memory store` or `store-batch` |
| **Document** (`memory doc store`) | Long-form content >800 chars OR multi-section structured content: plans, specs, runbooks, session summaries, implementation guides, architecture docs | `memory doc store` |

**Always route automatically.** Never ask the user which store to use — just pick the right one based on content length and structure. Both stores are cheap (SQLite + embeddings).

### Signals that content is a document:
- More than ~800 characters
- Has headings, numbered steps, or multiple sections
- Is a plan, spec, runbook, guide, or summary
- Would lose meaning if condensed to a single line
- User says "save this plan", "remember this spec", "store this summary"

### Signals that content is a memory:
- A single fact, decision, or observation
- Can be expressed in 1-3 sentences
- Prefixable with `[Decision]`, `[Pattern]`, `[Error]`, `[Learning]`, `[Observation]`

## Process

1. **Identify content** from `$ARGUMENTS` or conversation.
2. **Route each piece** — apply the routing rules above to decide memory vs document for each.
3. **Store** using the appropriate command(s):

### Storing memories (short facts)

   Single:
   ```bash
   memory -f text store "<content>" --tags "project:$PROJECT,<topic>" --type <type> --dedup 0.85
   ```

   Batch (2+ facts — batches embeddings in one pass, much faster):
   ```bash
   cat <<'ENDJSON' | memory -f text store-batch --dedup 0.85
   [
     {"content": "...", "tags": ["project:X", "topic"], "memory_type": "decision"},
     {"content": "...", "tags": ["project:X"], "memory_type": "pattern"}
   ]
   ENDJSON
   ```

   **IMPORTANT**: Always use `store-batch` for multiple facts — it computes embeddings in a single batch. Never split into separate parallel Bash tool calls; Claude Code serializes them with high dispatch overhead.

   Content should be a self-contained statement prefixed with `[Decision]`, `[Pattern]`, `[Learning]`, `[Error]`, or `[Observation]`.

### Storing documents (long-form content)

   ```bash
   cat <<'ENDBODY' | memory -f text doc store --title "<title>" --summary "<1-2 sentence summary>" --type <doc_type> --tags "project:$PROJECT,<topic>" --body-file -
   <full body content here>
   ENDBODY
   ```

   - **title**: Short descriptive title (e.g., "EKS Migration Plan")
   - **summary**: 1-2 sentences describing what the document covers — this is what gets embedded for semantic search, so make it descriptive
   - **type**: `plan`, `spec`, `runbook`, `session`, `reference`, or `document` (default)
   - **body**: The full content via `--body-file -` (stdin) to avoid shell quoting issues with long text

   #### Writing session documents

   Session documents (type `session`) capture a full working session. They must be **comprehensive** — never summarize from fading context when the full conversation data is available. Include:

   - All raw command outputs and data points collected during the session
   - Tables with exact values (signal readings, benchmark numbers, config parameters)
   - Every A/B comparison and test result with actual numbers
   - Each phase of investigation, what was tried, and what the result was
   - Dead ends and wrong assumptions — these are valuable for future sessions
   - The final configuration and how to reproduce it
   - TODOs and open questions

   Think of session docs as a lab notebook: someone continuing this work next week (or you in a future session) should be able to pick up exactly where things left off without guessing.

### Mixed content (common case)

When the user says "remember all of this", you'll often have both atomic facts AND long-form content. Store them in a **single chained Bash call**:

```bash
cat <<'ENDJSON' | memory -f text store-batch --dedup 0.85
[
  {"content": "[Decision] Use EKS over ECS for the platform", "tags": ["project:X", "cloud:aws"], "memory_type": "decision"},
  {"content": "[Pattern] Terraform modules go in modules/ with per-env tfvars", "tags": ["project:X", "tool:terraform"], "memory_type": "pattern"}
]
ENDJSON
echo "---"
cat <<'ENDBODY' | memory -f text doc store --title "EKS Migration Plan" --summary "Step-by-step plan for migrating services from ECS to EKS" --type plan --tags "project:X,cloud:aws,svc:eks" --body-file -
# EKS Migration Plan

## Phase 1: Infrastructure
...full plan content...
ENDBODY
```

## Tag Taxonomy

| Category | Format | Examples |
|----------|--------|---------|
| Scope | `project:<name>` or `scope:global` | `project:infrastructure` |
| Cloud | `cloud:<provider>` | `cloud:aws`, `cloud:gcp` |
| Service | `svc:<name>` | `svc:eks`, `svc:iam` |
| Tool | `tool:<name>` | `tool:terraform`, `tool:kubectl` |
| Topic | bare word | `networking`, `cicd` |
| Special | `todo` | Deferred work items, actionable tasks |

Always include at least a scope tag.

**TODO items:** When storing actionable or deferred work items, prefix content with `[TODO]` and include the `todo` tag. TODO items can include status markers in their content: `[TODO]`, `[TODO:PENDING]`, `[TODO:BLOCKED]`, `[TODO:DONE]`. Use multiple categories when applicable (e.g., `project:infrastructure,cloud:aws,svc:eks,tool:terraform`).

## CLI Quick Reference

Available commands: `store`, `store-batch`, `get`, `search`, `list`, `delete`, `update`, `health`, `cleanup`, `consolidate`, `decay`, `briefing`, `stats`, `doc`.

### Gotchas

- **`delete` takes hash as positional arg:** `memory delete <hash>` (not `--hash`)
- **Dedup rejections are not errors.** When `store` returns "duplicate", the memory already exists — no action needed.
- **`update --content` rehashes and re-embeds.** The returned `content_hash` changes — use the new hash for subsequent operations.
- **Long content → use `doc store`**: If content is >800 chars or structured, use `memory doc store` with `--body-file -` to pipe via stdin.

### Common Patterns

```bash
# Retrieve a memory by hash (supports prefix)
memory -f text get <hash>

# Retrieve a document by hash
memory -f text doc get <hash>

# Update content in-place (preserves tags, type, recall_count)
memory -f text update <hash> --content "new content here"

# Update only tags/importance (content stays the same)
memory update <hash> --tags "new,tags" --importance 0.9

# Update document body (increments version, rehashes)
memory -f text doc update <hash> --body-file updated.md --summary "updated summary"

# Search across both stores
memory -f text search "query" --limit 10
memory -f text doc search "query" --limit 5
```

Report: count stored (memories + documents), brief list, scope, any skipped duplicates.
