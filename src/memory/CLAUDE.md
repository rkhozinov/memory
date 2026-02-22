# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

memory — a lean memory service for Claude Code. Provides an MCP server and CLI for storing, searching, and managing semantic memories using vector embeddings. Python 3.14+, built with Hatchling.

## Commands

```bash
# Install (editable, with test deps)
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

**`models.py`** — `Memory` dataclass. Content is SHA256-hashed (`content_hash`) for exact dedup. Tags are `list[str]`, metadata is `dict`. Provides `to_row()`/`from_row()`/`to_dict()` for DB serialization.

**`embeddings.py`** — Wraps the `intfloat/e5-small` ONNX model (12-layer, 384-dim vectors, seq_length=256, 8-thread ONNX). Lazy-loaded singleton via `get_model()`. Downloads from HuggingFace on first use to `~/.claude/tools/memory/data/models/`. Mean pooling + L2 normalization. Returns raw numpy arrays to avoid `.tolist()` overhead.

**`core.py`** — `MemoryStore`, the main logic layer. All operations go through this class:
- `store()` / `store_batch()` — insert with optional semantic dedup (cosine threshold) and importance scoring
- `search()` — semantic (sqlite-vec cosine distance), exact (LIKE), or hybrid mode with composite scoring. Returns `recall_count` and `last_recalled_at` in results.
- `delete()` — soft delete via `deleted_at` timestamp; supports dry-run
- `consolidate()` — deterministic merge of near-duplicate memories (cosine > threshold)
- `apply_decay()` — recompute confidence decay and optionally prune low-confidence memories
- `briefing()` — generates a compact markdown briefing ranked by `confidence * importance * recency`, grouped into sections with line budget allocation
- `stats()` — aggregated analytics from `operation_events` table
- Uses IMMEDIATE transactions to prevent TOCTOU races in multi-agent scenarios
- Database at `~/.claude/tools/memory/data/sqlite_vec.db` (WAL mode, 15s busy timeout)

**`cli.py`** — argparse-based CLI. Commands: `store`, `store-batch`, `search`, `list`, `delete`, `update`, `health`, `cleanup`, `consolidate`, `decay`, `briefing`, `stats`. Output formats: `json` (default), `text`, `hook` (Claude Code hook format). Search supports `--depth titles|summary|full` for progressive disclosure.

**`mcp_server.py`** — MCP stdio server exposing the same operations as tools. Handles type coercion (string→int/bool/JSON) since MCP clients send everything as strings. Input validation is intentionally disabled.

## Database Schema

Five tables in SQLite:
- **`memories`** — core storage (content, tags as JSON, soft delete via `deleted_at`, recall tracking via `recall_count`/`last_recalled_at`, `confidence` for decay, `importance` for scoring)
- **`memory_embeddings`** — sqlite-vec virtual table, 384-dim float vectors with cosine distance
- **`memory_graph`** — relationship edges between memories (source_hash ↔ target_hash)
- **`operation_events`** — analytics log (operation type, duration, result counts, dedup info)
- **`metadata`** — generic key-value store

## Key Patterns

- **Embedding reuse**: When storing with dedup, the embedding is computed once and reused for both similarity check and storage.
- **Soft delete everywhere**: Records are never hard-deleted; `deleted_at IS NULL` filters them out.
- **Natural language time filters**: `search()` and `delete()` accept expressions like `"last week"`, `"3 days ago"`, or ISO dates.
- **Content hash as primary key for API**: External interfaces use `content_hash` (SHA256) to identify memories, not internal row IDs.
- **Confidence decay**: Memories lose confidence over time at per-type rates (decision/pattern=0.999/day, error/learning=0.99/day, note/observation=0.97/day). Recalled memories reset to 1.0.
- **Composite retrieval scoring**: `score = w1*similarity + w2*importance + w3*recency` (default 0.6/0.2/0.2). Search results re-ranked by composite score.
- **Importance auto-inference**: Keywords (IMPORTANT→0.9, NEVER→0.8) override type-based defaults (decision=0.8, error=0.7, note=0.4). Explicit `--importance` overrides both.
- **Deterministic consolidation**: Pairs with cosine similarity > threshold (default 0.92) are merged — higher recall_count wins, tags are unioned. `reference` type is excluded by default (`--exclude-types=reference`) because templated content like TF layer listings produces false-positive high-similarity matches. Pass `--exclude-types=''` to include all types.
- **Session briefing**: `briefing(budget=150)` generates a markdown summary grouped by type (decision/pattern/error/learning/reference/recent/other) with per-section line budgets scaled to fit the total budget. Ranked by `confidence * importance * recency`.
- **Progressive disclosure**: Search `--depth` controls output verbosity: `titles` (one line per result), `summary` (default, current behavior), `full` (all metadata including recall stats, timestamps, tags).

## Testing

Tests use a temp SQLite database (created per-test via the `store` fixture in `conftest.py`). The fixture sets up the full schema including the sqlite-vec virtual table. Tests are synchronous — `MemoryStore` methods are synchronous.
