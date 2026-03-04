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

## Process

1. **Identify facts** from `$ARGUMENTS` or conversation: decisions, patterns, observations, learnings, errors.
2. **Store** — use `store` for a single fact, `store-batch` for 2+ facts:

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

3. Content should be a self-contained statement prefixed with `[Decision]`, `[Pattern]`, `[Learning]`, `[Error]`, or `[Observation]`.

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

### Common Patterns

```bash
# Retrieve a memory by hash (supports prefix)
memory -f text get <hash>

# Update content in-place (preserves tags, type, recall_count)
memory -f text update <hash> --content "new content here"

# Update only tags/importance (content stays the same)
memory update <hash> --tags "new,tags" --importance 0.9
```

### Documents (long-form content)

For plans, specs, runbooks, or session summaries that exceed atomic fact size, use `memory doc`:

```bash
# Store a document (body from file or inline)
memory -f text doc store --title "Deploy Plan" --summary "EKS deploy steps" --body-file plan.md --type plan --tags "project:X"

# Search documents (auto = semantic + FTS merged)
memory -f text doc search "EKS deployment" --limit 5

# Get full document by hash
memory -f text doc get <hash>

# Update body (increments version, rehashes)
memory -f text doc update <hash> --body-file updated.md --summary "updated summary"

# List/delete
memory -f text doc list --type plan --tags "project:X"
memory -f text doc delete <hash>
```

Report: count stored, brief list, scope, any skipped duplicates.
