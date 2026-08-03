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
make install     # uv sync + lint/test + put the CLI on PATH
make reinstall    # uv tool install --force --editable . — puts `memory` on PATH
```

`make install` runs `sync`, `check` and `reinstall`. `reinstall` is the step
that actually builds the `memory` / `memory-mcp-server` / `memory-daemon`
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

Skills and hooks ship as a Claude Code **plugin**, installed from the
`rkhozinov` marketplace:

```
/plugin marketplace add rkhozinov/claude-marketplace
/plugin install memory@rkhozinov
```

That gives you `/memory:recall`, `/memory:remember`, `/memory:forget`,
`/memory:status` and `/memory:codebase`, plus three hooks: a SessionStart
banner that injects a curated index of your memories, a UserPromptSubmit hook
that recalls relevant memories once per session, and a SessionEnd hook that
runs consolidation and refreshes the index.

The CLI is a separate install (`make reinstall`, or `uv tool install`) — the
plugin's hooks and skills all shell out to `memory` on your PATH.

**The code that runs is the plugin cache**, not your checkout. To iterate on
hooks without republishing, `make link-dev` points the installed plugin's
`hooks/` at this repo and `make unlink-dev` puts it back; `make verify-runtime`
tells you which is currently live.

### Automatic capture (opt-in)

Whole-transcript archiving is **not** run from any hook. It captured every
session verbatim — automatic and free, but a pile of raw conversation rather
than facts. `memory admin auto-archive-pending` remains for manual use.


`memory admin extract-pending` distils *finished* sessions into atomic
memories. It only looks at transcripts idle for six hours or more, so it never
runs while you are working, and a free non-LLM gate rejects low-signal sessions
before any model call. Every stored fact must quote a verbatim span of the
transcript; secrets and denylisted terms drop the fact entirely.

It is disabled by default. Preview what it would store:

```
memory admin extract-pending --dry-run
```

Enable it (the SessionEnd hook then drains the backlog in the background):

```
export MEMORY_EXTRACT=1
```

Roughly $0.08 and ~100s per qualifying session on Haiku; `MEMORY_EXTRACT_DAILY_BUDGET`
(default $1.00) is a hard stop.

See `src/memory/CLAUDE.md` for architecture, database schema, and
implementation details.
