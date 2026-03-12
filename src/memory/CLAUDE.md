# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

memory — a lean memory service for Claude Code. Provides an MCP server and CLI for storing, searching, and managing semantic memories using vector embeddings. Python 3.14+, built with Hatchling.

## Commands

```bash
# Full install: sync deps, symlink skills/hooks, run all checks
make install

# Run ALL quality checks (lint + format + security + tests) — use this by default
make check

# Individual targets
make sync          # uv sync (deps only)
make link          # symlink skills/hooks into ~/.claude/
make lint          # ruff check src/ tests/
make format        # ruff format (auto-fix)
make format-check  # ruff format --check (dry-run)
make security      # bandit -r src/
make test          # pytest -q

# Run a single test
uv run pytest tests/test_core.py::test_name -v

# CLI entry point
memory --help

# MCP server (stdio transport)
memory-mcp-server

make reinstall
```

**IMPORTANT: After any code change, always run `make reinstall`.** The `memory` and `memory-mcp-server` binaries on PATH are a `uv tool` install in a separate environment (`~/.local/share/uv/tools/memory/`), NOT the project `.venv`. Source edits do NOT take effect until reinstalled. The `--editable` flag is passed so subsequent reinstalls are fast.

## Architecture

The codebase lives entirely in `src/memory/` (~2600 lines across 6 modules):

**`models.py`** — `Memory` and `Document` dataclasses. Content is SHA256-hashed (`content_hash`) for exact dedup. Tags are `list[str]`, metadata is `dict`. Both provide `to_row()`/`from_row()`/`to_dict()` for DB serialization. `Document` has `title`, `body`, `summary`, `doc_type`, `version` — no confidence/importance (reference material, no decay).

**`embeddings.py`** — Wraps the `intfloat/e5-small` ONNX model (12-layer, 384-dim vectors, seq_length=256, 8-thread ONNX). Lazy-loaded singleton via `get_model()`. Downloads from HuggingFace on first use to `~/.claude/tools/memory/data/models/`. Mean pooling + L2 normalization. Returns raw numpy arrays to avoid `.tolist()` overhead.

**`reranker.py`** — ONNX cross-encoder reranker for search result re-scoring. Lazy-loaded singleton via `get_reranker()`. Downloads quantized TinyBERT-L-2 (4.5MB) or MiniLM-L-6 model from HuggingFace on first use. Platform-aware: picks `model_qint8_arm64.onnx` on ARM, `model_quint8_avx2.onnx` on x86. Scores `(query, candidate)` pairs jointly via tokenizer pair encoding → ONNX inference → sigmoid. No caching (query-dependent scores).

**`core.py`** — `MemoryStore`, the main logic layer. All operations go through this class:
- `store()` / `store_batch()` — insert with optional semantic dedup (cosine threshold) and importance scoring
- `search()` — semantic (sqlite-vec cosine distance), exact (LIKE), hybrid, fts, or graph mode with composite scoring. Returns `recall_count` and `last_recalled_at` in results.
- `delete()` — soft delete via `deleted_at` timestamp; supports dry-run
- `consolidate()` — deterministic merge of near-duplicate memories (cosine > threshold)
- `apply_decay()` — recompute confidence decay and optionally prune low-confidence memories
- `briefing()` — generates a compact markdown briefing ranked by `confidence * importance * recency`, grouped into sections with line budget allocation
- `stats()` — aggregated analytics from `operation_events` table
- Document operations: `store_doc()`, `get_doc()`, `list_docs()`, `search_docs()`, `update_doc()`, `delete_doc()` — long-form content (plans, specs, runbooks) with summary-based hybrid retrieval (semantic on summary embedding + FTS5 on body)
- Knowledge graph: `extract_entities()` (regex-based), `_link_entities()` (auto on store), `list_entities()`, `entity_context()`, `build_graph()`, `_search_graph()` — lightweight entity extraction + co-occurrence graph for relationship traversal
- Uses IMMEDIATE transactions to prevent TOCTOU races in multi-agent scenarios
- Database at `~/.claude/tools/memory/data/sqlite_vec.db` (WAL mode, 15s busy timeout)

**`cli.py`** — argparse-based CLI. Commands: `store`, `store-batch`, `search`, `search-batch`, `list`, `delete`, `update`, `health`, `cleanup`, `list-tags`, `rename-tag`, `merge-tags`, `export`, `import`, `purge`, `consolidate`, `decay`, `briefing`, `stats`, `doc`, `graph`. The `doc` subcommand group has: `store`, `get`, `search`, `list`, `update`, `delete`. The `graph` subcommand group has: `build`, `entities`, `context`, `search`. Output formats: `json` (default), `text`, `hook` (Claude Code hook format). Search supports `--depth titles|summary|full` for progressive disclosure, `--exclude-tags`, `--min-importance`, `--types` for advanced filtering, `--mode graph` for graph traversal, `--rerank` for cross-encoder reranking (with `--rerank-weight` and `--rerank-model`), and returns `score_breakdown` in full depth.

**`mcp_server.py`** — MCP stdio server exposing the same operations as tools. Handles type coercion (string→int/bool/JSON) since MCP clients send everything as strings. Input validation is intentionally disabled.

## Database Schema

Eleven tables in SQLite:
- **`memories`** — core storage (content, tags as JSON, soft delete via `deleted_at`, recall tracking via `recall_count`/`last_recalled_at`, `confidence` for decay, `importance` for scoring)
- **`memory_embeddings`** — sqlite-vec virtual table, 384-dim float vectors with cosine distance
- **`memory_fts`** — FTS5 virtual table for BM25 keyword search on memory content
- **`documents`** — long-form content (title, body, summary, doc_type, version tracking, recall tracking, soft delete)
- **`document_embeddings`** — sqlite-vec virtual table, 384-dim float vectors on summary embedding
- **`document_fts`** — FTS5 virtual table for BM25 keyword search on document title + body
- **`operation_events`** — analytics log (operation type, duration, result counts, dedup info)
- **`entities`** — canonical entities (deduplicated by normalized name + type: ticket, service, technology, project, cloud, tool, pr)
- **`memory_entities`** — many-to-many linking memories to entities
- **`entity_relations`** — entity-to-entity co-occurrence edges with weights

## Key Patterns

- **Embedding reuse**: When storing with dedup, the embedding is computed once and reused for both similarity check and storage.
- **Dedup scoping**: Semantic dedup is scoped first by `memory_type`, then by tags (if provided). A new memory only deduplicates against existing memories sharing the same type AND at least one tag. This prevents cross-project false positives from shared sentence structure. Use `--dedup 0.90` (not 0.85) as the default threshold — genuine duplicates cluster above 0.92.
- **Soft delete everywhere**: Records are never hard-deleted; `deleted_at IS NULL` filters them out.
- **Natural language time filters**: `search()` and `delete()` accept expressions like `"last week"`, `"3 days ago"`, or ISO dates.
- **Content hash as primary key for API**: External interfaces use `content_hash` (SHA256) to identify memories, not internal row IDs.
- **Confidence decay**: Memories lose confidence over time at per-type rates (decision/pattern=0.999/day, error/learning=0.99/day, note/observation=0.97/day). Recalled memories reset to 1.0.
- **Composite retrieval scoring**: `score = w1*similarity + w2*importance + w3*recency` (default 0.6/0.2/0.2). Search results re-ranked by composite score.
- **Cross-encoder reranking** (opt-in): `--rerank` re-scores top candidates with a cross-encoder. Overfetches 3x candidates, blends `(1-w)*normalized_composite + w*reranker_score` (default w=0.4). Only applies to semantic/fts/hybrid modes. Models: `tinybert` (4.5MB, default), `minilm6` (23MB, higher quality).
- **Importance auto-inference**: Keywords (IMPORTANT→0.9, NEVER→0.8) override type-based defaults (decision=0.8, error=0.7, note=0.4). Explicit `--importance` overrides both.
- **Deterministic consolidation**: Pairs with cosine similarity > threshold (default 0.92) are merged — higher recall_count wins, tags are unioned. `reference` type is excluded by default (`--exclude-types=reference`) because templated content like TF layer listings produces false-positive high-similarity matches. Pass `--exclude-types=''` to include all types.
- **Session briefing**: `briefing(budget=150)` generates a markdown summary grouped by type (decision/pattern/error/learning/reference/recent/other) with per-section line budgets scaled to fit the total budget. Ranked by `confidence * importance * recency`.
- **Progressive disclosure**: Search `--depth` controls output verbosity: `titles` (one line per result), `summary` (default, current behavior), `full` (all metadata including recall stats, timestamps, tags).
- **Documents layer**: Long-form content (plans, specs, runbooks, session summaries) stored separately from atomic memories. Summary-based hybrid retrieval: semantic search on summary embedding + FTS5 on full body. No confidence decay (reference material). Version tracking on body updates. `content_hash` is SHA256 of body.
- **Knowledge graph**: Regex-based entity extraction on every `store()` — tickets (`[A-Z]{2,10}-\d+`), PR refs, ~80 technology keywords, `*-service/*-manager/*-api` patterns, and tag-derived entities (`project:X`, `svc:X`, `cloud:X`, `tool:X`). Co-occurrence edges connect entities that appear in the same memory. Graph search (`--mode graph`) uses recursive CTE traversal up to N hops. `build_graph()` retroactively populates from existing memories. Zero new dependencies.

## Skills & Hooks

Skills and hooks live in this repo and are symlinked into `~/.claude/` by ``make install``.

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

## Code Quality

Tooling: **ruff** (lint + format), **bandit** (security). Config in `pyproject.toml`, convenience targets in `Makefile`.

**Always run `make check` before considering work done.** It runs lint, format check, security scan, and all tests in one command. Output is quiet on success — only failures produce verbose output.

Known suppressions:
- `S608` (SQL injection) — all flagged queries use `?` parameterized placeholders, not string interpolation of user input. Skipped globally via `make security`.
- `B310` (URL open) — `urlretrieve` in `embeddings.py` uses hardcoded HuggingFace URLs. Suppressed inline with `# nosec B310`.
- `T201` (print) — CLI uses `print()` for output by design.

## Testing

Tests use a temp SQLite database (created per-test via the `store` fixture in `conftest.py`). The fixture sets up the full schema including the sqlite-vec virtual table. Tests are synchronous — `MemoryStore` methods are synchronous.

Run all tests: `make test` (uses `pytest -q` for compact output). Run a single test: `uv run pytest tests/test_core.py::test_name -v`.
