# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

memory — a lean memory service for Claude Code. Provides an MCP server and CLI for storing, searching, and managing semantic memories using vector embeddings. Python 3.14+, built with Hatchling.

## Commands

```bash
# Full install: sync deps, run all checks, put the CLI on PATH
make install

# Run ALL quality checks (lint + format + security + tests) — use this by default
make check

# Individual targets
make sync          # uv sync (deps only)
make link-dev      # point the installed plugin's hooks/ at this checkout
make unlink-dev    # restore the plugin's own hooks/
make verify-runtime # confirm the running plugin matches this checkout
make lint          # ruff check src/ tests/ + shellcheck hooks/
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

**`paths.py`** — Resolves the data dir + DB path so the plugin is clone-location-agnostic. Order: `MEMORY_DATA_DIR`/`MEMORY_DB` env → legacy `~/repos/memory/data` (if a DB exists there) → `~/.local/share/memory` (XDG). core and embeddings both import from here.
**`embeddings.py`** — Wraps the `nomic-ai/modernbert-embed-base` ONNX model (ModernBERT, 768-dim vectors, seq_length=512, 8-thread ONNX). Lazy-loaded singleton via `get_model()`. Downloads from HuggingFace on first use to `<data-dir>/models/` (see `paths.py`). Mean pooling + L2 normalization. Requires `search_query:` prefix for queries and `search_document:` prefix for stored content — use `embed_query()`/`embed_doc()` convenience methods. Returns raw numpy arrays.


**`core.py`** — `MemoryStore`, the main logic layer. All operations go through this class:
- `store()` / `store_batch()` — insert with optional semantic dedup (cosine threshold) and importance scoring
- `search()` — semantic (sqlite-vec cosine distance), exact (LIKE), hybrid, fts, or graph mode with composite scoring. Returns `recall_count` and `last_recalled_at` in results.
- `delete()` — soft delete via `deleted_at` timestamp; supports dry-run
- `consolidate()` — deterministic merge of near-duplicate memories (cosine > threshold)
- `apply_decay()` — recompute confidence decay and optionally prune low-confidence memories
- `briefing()` — generates a compact markdown briefing ranked by `confidence * importance * recency`, grouped into sections with line budget allocation
- `stats()` — aggregated analytics from `operation_events` table, plus
  `by_provenance`: recall rates for machine-written (`source:extract`) vs
  hand-written memories, the readout for whether auto-extraction earns its keep
- Document operations: `store_doc()`, `get_doc()`, `list_docs()`, `search_docs()`, `update_doc()`, `delete_doc()` — long-form content (plans, specs, runbooks) with summary-based hybrid retrieval (semantic on summary embedding + FTS5 on body)
- Knowledge graph: `extract_entities()` (regex-based), `_link_entities()` (auto on store), `list_entities()`, `entity_context()`, `build_graph()`, `_search_graph()` — lightweight entity extraction + co-occurrence graph for relationship traversal
- Uses IMMEDIATE transactions to prevent TOCTOU races in multi-agent scenarios
- Database at the resolved data dir (`MEMORY_DB`/`MEMORY_DATA_DIR` → legacy `~/repos/memory/data` → `~/.local/share/memory`); `sqlite_vec.db`, WAL mode, 15s busy timeout

**`cli.py`** — argparse-based CLI. **JSON output by default**; `search` and `doc search` also accept `--depth summary` for one plain-text line per hit (used by the `/recall` skill and the topic-recall hook so they don't reformat JSON in python). 8 top-level commands: `store` (accepts plain text, JSON object, or JSON array — unified single/batch), `search`, `get` (partial hash prefix supported), `delete` (partial hash prefix supported), `update`, `health`, `doc` (subgroup: store/get/search/list/update/delete), `admin` (subgroup: cleanup/consolidate/decay/purge/export/import/stats/briefing/tags/graph). Search supports `--mode`, `--tags`, `--types`, `--score-fusion`, `--hops`. Maintenance ops moved under `admin` to reduce top-level clutter. Aliases for muscle memory: `add`→`store`, `find`→`search`, `rm`/`forget`→`delete`, `doc ls`→`doc list`. Each alias needs its own key in the dispatch dict — argparse puts the alias *as typed* into `args.command`. `memory --version` prints the installed dist version.

**`extract.py`** — Idle-gated distillation of finished sessions into atomic memories. A free signal gate (turns, length, mutating tool calls, signal tokens) decides whether to spend a model call at all; the transcript is clipped head+tail; the model runs headless (`--print --json-schema --tools "" --no-session-persistence`) and every proposed fact must quote a verbatim span of the transcript or it is dropped. ADD-only storage tagged `source:extract`; `dream()` handles supersession and pruning. Opt-in via `MEMORY_EXTRACT=1`; `--dry-run` proposes without storing. CLI: `memory admin extract-pending`.

**`mcp_server.py`** — MCP stdio server exposing the same operations as tools. Handles type coercion (string→int/bool/JSON) since MCP clients send everything as strings. Input validation is intentionally disabled.

## Database Schema

Eleven tables in SQLite:
- **`memories`** — core storage (content, tags as JSON, soft delete via `deleted_at`, recall tracking via `recall_count`/`last_recalled_at`, `confidence` for decay, `importance` for scoring)
- **`memory_embeddings`** — sqlite-vec virtual table, 768-dim float vectors with cosine distance
- **`memory_fts`** — FTS5 virtual table for BM25 keyword search on memory content
- **`documents`** — long-form content (title, body, summary, doc_type, version tracking, recall tracking, soft delete)
- **`document_embeddings`** — sqlite-vec virtual table, 768-dim float vectors on summary embedding
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
- **Composite retrieval scoring**: `score = w1*similarity + w2*importance + w3*recency` (default 0.8/0.1/0.1, `DEFAULT_SCORING_WEIGHTS` in core.py). Search results re-ranked by composite score.
- **Importance auto-inference**: Keywords (IMPORTANT→0.9, NEVER→0.8) override type-based defaults (decision=0.8, error=0.7, note=0.4). Explicit `--importance` overrides both.
- **Deterministic consolidation**: Pairs with cosine similarity > threshold (default 0.92) are merged — higher recall_count wins, tags are unioned. `reference` type is excluded by default (`--exclude-types=reference`) because templated content like infra layer listings produces false-positive high-similarity matches. Pass `--exclude-types=''` to include all types.
- **Consolidation value guard**: `consolidate()` refuses to merge members that disagree about a concrete value (version, dotted number, kebab/snake symbol, backticked literal, bare count) — the same `differs_by_value()` discriminator write-time dedup uses. Cosine ≥ 0.85 does not mean duplicate: on the production store 391 of 396 pairs at that threshold diverge by value, so merging them deletes facts. In cluster mode one diverging pair condemns the whole component, since a cluster merges as a unit. Both modes return `blocked_by_guard`. Disable with `value_guard=False` / `--no-value-guard` only for benchmarks that deliberately seed unique-fact clusters. Note IDF-weighted lexical overlap was tried first and **refuted** — the shared boilerplate is corpus-*rare* (df 27–35 of 5964), so IDF scores it high; see `benchmarks/results/consolidate-value-guard.json`.
- **Session briefing**: `briefing(budget=150)` generates a markdown summary grouped by type (decision/pattern/error/learning/reference/recent/other) with per-section line budgets scaled to fit the total budget. Ranked by `confidence * importance * recency`.
- **Progressive disclosure**: `memory search` / `memory doc search` take `--depth {summary,full}`. `full` is the default and dumps the complete JSON record; `summary` prints `<hash16> [<type>] score=<n.nn> <preview>` (docs: `<hash16> [doc] <title>`), with a trailing `[stale:N]` when the hit carries `stale_refs`. `search` also takes `--min-score` to drop low-scoring hits (unscored modes like `exact` pass through). Note hybrid fusion scores are not capped at 1.0. `memory admin clusters` has its own `--depth {summary,full}`, defaulting to `summary`. The MCP server uses a separate `depth: titles` projection.
- **Documents layer**: Long-form content (plans, specs, runbooks, session summaries) stored separately from atomic memories. Summary-based hybrid retrieval: semantic search on summary embedding + FTS5 on full body. No confidence decay (reference material). Version tracking on body updates. `content_hash` is SHA256 of body.
- **Knowledge graph**: Regex-based entity extraction on every `store()` — tickets (`[A-Z]{2,10}-\d+`), PR refs, ~80 technology keywords, `*-service/*-manager/*-api` patterns, and tag-derived entities (`project:X`, `svc:X`, `cloud:X`, `tool:X`). Co-occurrence edges connect entities that appear in the same memory. Graph search (`--mode graph`) uses `extract_entities()` to tokenize queries into entity candidates (e.g., `"terraform state locking"` → seeds on "terraform" entity), with fallback to whole-string lookup when no entities are extracted. Recursive CTE traversal up to N hops (`--hops`, default 2). `build_graph()` retroactively populates from existing memories. Zero new dependencies.

## Skills & Hooks

Skills and hooks ship as part of the Claude Code **plugin**. The code that runs
is the plugin cache (`~/.claude/plugins/cache/rkhozinov/memory/<version>/`),
**not** this checkout — see `make link-dev` / `make verify-runtime`.

There is deliberately no `make link` target any more. It used to symlink
`skills/*` into `~/.claude/skills/` and `hooks/*.sh` into `~/.claude/hooks/`;
both are wrong under plugin packaging. The skills copy shadows the plugin's
namespaced `memory:*` skills with un-namespaced duplicates, and the plugin
runtime reads `${CLAUDE_PLUGIN_ROOT}/hooks/` — never `~/.claude/hooks/` — so the
hook symlinks implied the hooks were live when they were not.

**`skills/`** — `/memory:recall`, `/memory:remember`, `/memory:forget`,
`/memory:status`, `/memory:codebase`. All shell out to the `memory` CLI.

**`hooks/`** (registered in `hooks/hooks.json`; `timeout` is in **seconds**):
- **`memory-session-start.sh`** — banner, daily cleanup, and injection of the
  curated index (`--max-lines 60 --max-tokens 1200`)
- **`memory-topic-recall.sh`** — UserPromptSubmit: one recall per session
- **`memory-session-end.sh`** — throttled `dream`, index refresh, and the
  opt-in extractor
- **`hooks/lib/hooklib.sh`** — shared helpers, including `mem_run` (there is no
  `timeout(1)` on stock macOS) and the hook-input parser

Hook state lives under `$MEMORY_HOOK_STATE_DIR` (default
`~/.claude/memory/state/`), never in the store's data dir and never in a
hardcoded `~/repos/memory` path — the plugin has to work for users who cloned
somewhere else, or not at all.

The injected catalog is **scoped per project**: `mem_project_tag()` derives
`project:<basename-of-cwd>` and passes it as `memory admin index --tags`, and
both the catalog (`INDEX-<project>.md`) and its freshness marker are per-scope —
a single shared file would let the last project to rebuild decide what every
other project's session gets injected. Set `MEMORY_INDEX_SCOPE=""` for a global
catalog. Scoping *prioritises* rather than filters, so an unfamiliar project
still gets a useful catalog instead of an empty one.

Session identity comes from `CLAUDE_CODE_SESSION_ID`, which Claude Code exports
into every tool subprocess. Do not reintroduce a hook-side `export
MEMORY_SESSION_ID` — the export dies with the hook process, which is why
`distinct_session_count` sat at 0 until 1.11.3 and the hot-cluster correction in
`compute_activation()` never engaged.

`tests/test_hooks.py` exercises the scripts with the CLI stubbed out, including
a SessionStart run on a PATH with no coreutils, plus static drift tests that
assert the hooks only call real subcommands and only read JSON keys that exist.

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
