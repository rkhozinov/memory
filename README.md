# memory

Lean memory service for Claude Code: SQLite + sqlite-vec embeddings, FTS5
keyword search, cross-encoder rerank, bi-temporal knowledge graph, ACT-R
confidence decay, cluster-based dedup, and a documents layer for long-form
content (plans, specs, runbooks, session archives). MCP server + CLI, JSON
output only.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/rkhozinov/memory
cd memory
make install     # uv sync + symlink skills/hooks into ~/.claude/ + lint/test
make reinstall    # uv tool install --force --editable . — puts `memory` on PATH
```

`make install` and `make reinstall` are separate steps: `install` sets up
the dev environment and Claude Code integration (skills, hooks); `reinstall`
is what actually builds the `memory` / `memory-mcp-server` / `memory-daemon`
binaries onto your PATH via `uv tool install`. Source edits don't take
effect until you rerun `make reinstall` — the installed binaries live in a
separate `uv tool` environment (`~/.local/share/uv/tools/memory/`), not the
project `.venv`.

`make install`'s lint/test step can fail on pre-existing lint debt in the
source without affecting the actual install — `make reinstall` alone doesn't
depend on lint passing, so run it even if `make install` reports a `lint`
error.

### First run

- The data directory (SQLite DB + downloaded models) is created on first
  use — no manual `mkdir` needed. Default location is
  `~/.local/share/memory/` (or `$XDG_DATA_HOME/memory`). If a DB already
  exists at the legacy `~/repos/memory/data/`, that path is kept. Override
  either with `MEMORY_DATA_DIR` (dir) or `MEMORY_DB` (DB file).
- The first `store`/`search` call downloads the embedding model
  (`nomic-ai/modernbert-embed-base`, ONNX + MLX weights, ~100MB) from
  HuggingFace Hub — a one-time cost (roughly a minute), cached afterward
  under `<data-dir>/models/`. No `HF_TOKEN` needed; that's only for HF's rate
  limits on heavy anonymous usage, irrelevant for a single cached pull.
- Everything runs locally — the embedding model is not an LLM and makes no
  API calls. `memory health` confirms the DB is reachable once set up.

## Usage

```bash
memory store "some fact"                    # store a memory
memory search "query"                       # hybrid semantic+FTS search
memory doc store --title T --body B         # store a long-form document
memory doc search "query"                   # search documents
memory admin stats                          # usage analytics
memory health                               # DB health check
```

Full command surface: `memory --help`, `memory doc --help`,
`memory admin --help`.

## Skills & hooks

`make install` symlinks `skills/*` into `~/.claude/skills/` and `hooks/*.sh`
into `~/.claude/hooks/`: `/recall`, `/remember`, `/forget`,
`/memory:status`, plus SessionStart/UserPromptSubmit hooks for health
checks and auto-recall. Edits to this repo take effect immediately (symlinks,
no reinstall needed for skills/hooks — only for the compiled CLI).

See `src/memory/CLAUDE.md` for architecture, database schema, and
implementation details.
