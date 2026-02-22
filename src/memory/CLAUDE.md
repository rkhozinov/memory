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
- `store()` / `store_batch()` — insert with optional semantic dedup (cosine threshold)
- `search()` — semantic (sqlite-vec cosine distance), exact (LIKE), or hybrid mode
- `delete()` — soft delete via `deleted_at` timestamp; supports dry-run
- `stats()` — aggregated analytics from `operation_events` table
- Uses IMMEDIATE transactions to prevent TOCTOU races in multi-agent scenarios
- Database at `~/.claude/tools/memory/data/sqlite_vec.db` (WAL mode, 15s busy timeout)

**`cli.py`** — Click-based CLI. Commands: `store`, `store-batch`, `search`, `list`, `delete`, `update`, `health`, `cleanup`, `stats`. Output formats: `json` (default), `text`, `hook` (Claude Code hook format).

**`mcp_server.py`** — MCP stdio server exposing the same operations as tools. Handles type coercion (string→int/bool/JSON) since MCP clients send everything as strings. Input validation is intentionally disabled.

## Database Schema

Five tables in SQLite:
- **`memories`** — core storage (content, tags as JSON, soft delete via `deleted_at`, recall tracking via `recall_count`/`last_recalled_at`)
- **`memory_embeddings`** — sqlite-vec virtual table, 384-dim float vectors with cosine distance
- **`memory_graph`** — relationship edges between memories (source_hash ↔ target_hash)
- **`operation_events`** — analytics log (operation type, duration, result counts, dedup info)
- **`metadata`** — generic key-value store

## Key Patterns

- **Embedding reuse**: When storing with dedup, the embedding is computed once and reused for both similarity check and storage.
- **Soft delete everywhere**: Records are never hard-deleted; `deleted_at IS NULL` filters them out.
- **Natural language time filters**: `search()` and `delete()` accept expressions like `"last week"`, `"3 days ago"`, or ISO dates.
- **Content hash as primary key for API**: External interfaces use `content_hash` (SHA256) to identify memories, not internal row IDs.

## Testing

Tests use a temp SQLite database (created per-test via the `store` fixture in `conftest.py`). The fixture sets up the full schema including the sqlite-vec virtual table. Tests are synchronous — `MemoryStore` methods are synchronous.
