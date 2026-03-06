# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

memory — a lean memory service for Claude Code. Provides an MCP server and CLI for storing, searching, and managing semantic memories using vector embeddings. Python 3.14+, built with Hatchling.

## Commands

```bash
# Full install: pip package + symlink skills & hooks into ~/.claude/
./install.sh

# Install just the Python package (editable, with test deps)
uv pip install -e ".[test]"

# Run tests
pytest tests/

# Run a single test
pytest tests/test_core.py::test_name -v

# CLI entry point
memory --help

# MCP server (stdio transport)
memory-mcp-server
```

## Architecture

The codebase lives entirely in `src/memory/` (~2000 lines across 5 modules):

**`models.py`** — `Memory` and `Document` dataclasses. Content is SHA256-hashed (`content_hash`) for exact dedup. Tags are `list[str]`, metadata is `dict`. Both provide `to_row()`/`from_row()`/`to_dict()` for DB serialization. `Document` has `title`, `body`, `summary`, `doc_type`, `version` — no confidence/importance (reference material, no decay).

**`embeddings.py`** — Wraps the `intfloat/e5-small` ONNX model (12-layer, 384-dim vectors, seq_length=256, 8-thread ONNX). Lazy-loaded singleton via `get_model()`. Downloads from HuggingFace on first use to `~/.claude/tools/memory/data/models/`. Mean pooling + L2 normalization. Returns raw numpy arrays to avoid `.tolist()` overhead.

**`core.py`** — `MemoryStore`, the main logic layer. All operations go through this class:
- `store()` / `store_batch()` — insert with optional semantic dedup (cosine threshold) and importance scoring
- `search()` — semantic (sqlite-vec cosine distance), exact (LIKE), or hybrid mode with composite scoring. Returns `recall_count` and `last_recalled_at` in results.
- `delete()` — soft delete via `deleted_at` timestamp; supports dry-run
- `consolidate()` — deterministic merge of near-duplicate memories (cosine > threshold)
- `apply_decay()` — recompute confidence decay and optionally prune low-confidence memories
- `briefing()` — generates a compact markdown briefing ranked by `confidence * importance * recency`, grouped into sections with line budget allocation
- `stats()` — aggregated analytics from `operation_events` table
- Document operations: `store_doc()`, `get_doc()`, `list_docs()`, `search_docs()`, `update_doc()`, `delete_doc()` — long-form content (plans, specs, runbooks) with summary-based hybrid retrieval (semantic on summary embedding + FTS5 on body)
- Uses IMMEDIATE transactions to prevent TOCTOU races in multi-agent scenarios
- Database at `~/.claude/tools/memory/data/sqlite_vec.db` (WAL mode, 15s busy timeout)

**`cli.py`** — argparse-based CLI. Commands: `store`, `store-batch`, `search`, `search-batch`, `list`, `delete`, `update`, `health`, `cleanup`, `list-tags`, `rename-tag`, `merge-tags`, `export`, `import`, `purge`, `consolidate`, `decay`, `briefing`, `stats`, `doc`. The `doc` subcommand group has: `store`, `get`, `search`, `list`, `update`, `delete`. Output formats: `json` (default), `text`, `hook` (Claude Code hook format). Search supports `--depth titles|summary|full` for progressive disclosure, `--exclude-tags`, `--min-importance`, `--types` for advanced filtering, and returns `score_breakdown` in full depth.

**`mcp_server.py`** — MCP stdio server exposing the same operations as tools. Handles type coercion (string→int/bool/JSON) since MCP clients send everything as strings. Input validation is intentionally disabled.

## Database Schema

Eight tables in SQLite:
- **`memories`** — core storage (content, tags as JSON, soft delete via `deleted_at`, recall tracking via `recall_count`/`last_recalled_at`, `confidence` for decay, `importance` for scoring)
- **`memory_embeddings`** — sqlite-vec virtual table, 384-dim float vectors with cosine distance
- **`memory_fts`** — FTS5 virtual table for BM25 keyword search on memory content
- **`documents`** — long-form content (title, body, summary, doc_type, version tracking, recall tracking, soft delete)
- **`document_embeddings`** — sqlite-vec virtual table, 384-dim float vectors on summary embedding
- **`document_fts`** — FTS5 virtual table for BM25 keyword search on document title + body
- **`operation_events`** — analytics log (operation type, duration, result counts, dedup info)

## Key Patterns

- **Embedding reuse**: When storing with dedup, the embedding is computed once and reused for both similarity check and storage.
- **Dedup scoping**: Semantic dedup is scoped first by `memory_type`, then by tags (if provided). A new memory only deduplicates against existing memories sharing the same type AND at least one tag. This prevents cross-project false positives from shared sentence structure. Use `--dedup 0.90` (not 0.85) as the default threshold — genuine duplicates cluster above 0.92.
- **Soft delete everywhere**: Records are never hard-deleted; `deleted_at IS NULL` filters them out.
- **Natural language time filters**: `search()` and `delete()` accept expressions like `"last week"`, `"3 days ago"`, or ISO dates.
- **Content hash as primary key for API**: External interfaces use `content_hash` (SHA256) to identify memories, not internal row IDs.
- **Confidence decay**: Memories lose confidence over time at per-type rates (decision/pattern=0.999/day, error/learning=0.99/day, note/observation=0.97/day). Recalled memories reset to 1.0.
- **Composite retrieval scoring**: `score = w1*similarity + w2*importance + w3*recency` (default 0.6/0.2/0.2). Search results re-ranked by composite score.
- **Importance auto-inference**: Keywords (IMPORTANT→0.9, NEVER→0.8) override type-based defaults (decision=0.8, error=0.7, note=0.4). Explicit `--importance` overrides both.
- **Deterministic consolidation**: Pairs with cosine similarity > threshold (default 0.92) are merged — higher recall_count wins, tags are unioned. `reference` type is excluded by default (`--exclude-types=reference`) because templated content like TF layer listings produces false-positive high-similarity matches. Pass `--exclude-types=''` to include all types.
- **Session briefing**: `briefing(budget=150)` generates a markdown summary grouped by type (decision/pattern/error/learning/reference/recent/other) with per-section line budgets scaled to fit the total budget. Ranked by `confidence * importance * recency`.
- **Progressive disclosure**: Search `--depth` controls output verbosity: `titles` (one line per result), `summary` (default, current behavior), `full` (all metadata including recall stats, timestamps, tags).
- **Documents layer**: Long-form content (plans, specs, runbooks, session summaries) stored separately from atomic memories. Summary-based hybrid retrieval: semantic search on summary embedding + FTS5 on full body. No confidence decay (reference material). Version tracking on body updates. `content_hash` is SHA256 of body.

## Skills & Hooks

Skills and hooks live in this repo and are symlinked into `~/.claude/` by `install.sh`.

**`skills/`** — Claude Code slash commands (symlinked to `~/.claude/skills/`):
- **`recall/`** — `/recall [query]`: search memories and documents, or generate briefing
- **`remember/`** — `/remember <content>`: store facts with tag taxonomy (includes `doc` subcommand reference)
- **`forget/`** — `/forget <query>`: find and delete memories or documents with confirmation
- **`memory-status/`** — `/memory-status`: health check, stats, document listing

**`hooks/`** — Claude Code event hooks (symlinked to `~/.claude/hooks/`):
- **`memory-session-start.sh`** — SessionStart: health check, daily cleanup, codebase map check
- **`memory-topic-recall.sh`** — UserPromptSubmit: auto-recall on first prompt per session
- **`memory-cleanup.sh`** — Cleanup report (manual): dedup, stats, recommendations

After editing any skill or hook in this repo, changes take effect immediately (symlinks).

## Testing

Tests use a temp SQLite database (created per-test via the `store` fixture in `conftest.py`). The fixture sets up the full schema including the sqlite-vec virtual table. Tests are synchronous — `MemoryStore` methods are synchronous.
