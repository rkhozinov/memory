# memory TUI design

## Goal

A Textual TUI for browsing and managing both object types the `memory`
service stores — short `Memory` rows and long-form `Document` rows — mirroring
the 2-pane pattern already proven in `handoff/tui.py` (list left, detail
right, vim-style navigation, mutations routed through the data layer instead
of reimplemented in the UI).

Entry point: `memory tui`, lazy-imported so the base `memory` CLI never pays
Textual's import cost.

## Architecture

- New optional dependency group in `pyproject.toml`: `tui = ["textual>=0.60"]`
  (matches handoff's `[tui]` extra).
- New module `src/memory/tui.py`. Lazy-imported by a new `tui` subcommand
  added to `cli.py`'s argparse tree.
- Unlike handoff (which dual-writes brief `.md` files + a DB mirror),
  `MemoryStore` is the single source of truth — pure SQLite, no parallel
  files to reconcile. The TUI opens one `MemoryStore` on mount and calls its
  methods directly for both reads and mutations; no separate db-reader
  module is needed.

## Layout

One `App`, one CSS block, two modes toggled with `Tab`: **Memories** and
**Documents**. Each mode is a self-contained 2-pane view (`ListView` left,
scrollable detail right) — same shape as handoff, panes swap content based on
mode. Header/`sub_title` shows the current mode name and row count.

- **Memories list row**: type icon + first ~60 chars of content, dim line
  below with tags / importance / confidence — mirrors handoff's
  `_item_text` (icon + title, dim meta line).
- **Documents list row**: title, dim line below with doc_type / tags /
  updated date.
- **Detail pane (Memories)**: full content + metadata (tags, type,
  confidence, importance, trust, timestamps).
- **Detail pane (Documents)**: `body` rendered as markdown (toggle raw with
  `m`, same affordance as handoff) with a summary/metadata header above it.

## Filter vs. search

Two distinct actions — conflating them would make either fast filtering
laggy or scored search fake:

- **`/` — instant local fuzzy filter.** Same subsequence-fuzzy algorithm as
  handoff's `_fuzzy_score`, run over cached rows' content/title + tags. No
  DB hit, no embedding inference — matches tuxedo/handoff precedent for `/`.
- **`s` — real semantic search.** Opens a query prompt, calls
  `store.search(query, mode="hybrid", ...)` (Memories mode) or
  `store.search_docs(query, ...)` (Documents mode). Replaces the list with
  ranked results. Each row's dim meta line grows a `sim: 0.xx` segment (and
  `imp:` / `rec:` when present), sourced directly from the result dict's
  existing `similarity` / `importance` / `recency` fields — no new scoring
  logic, just surfacing what `search()` already computes. `esc` drops back
  to the unscored browse list.

## Manual merge

`consolidate()` in `core.py` only does automatic whole-DB threshold sweeps
(finds pairs/clusters above a cosine threshold, auto-picks the survivor).
There's no existing path for "merge these two specific memories I selected
in the UI" — that needs a small new method:

```python
def merge_memories(self, content_hashes: list[str], content_strategy: str = "keep_higher_recall") -> dict:
    """Merge an explicit set of memories into one survivor. Same survivor
    rule and mechanics as consolidate()'s pairwise merge (union tags, record
    merged_into lineage edges, soft-delete losers), scoped to a caller-given
    hash list instead of a threshold sweep."""
```

Refactor: extract consolidate's per-pair merge body (currently inline in the
pairwise loop — tag union, lineage edge, soft-delete) into a shared
`_merge_into(conn, keep_hash, remove_hash, similarity, content_strategy)`
helper. Both `consolidate()`'s loop and the new `merge_memories()` call it,
so merge mechanics live in exactly one place.

TUI flow: `v` enters visual/multi-select mode (Memories mode only, mirrors
tuxedo's `v`/`space` pattern), `M` merges the selection — a confirmation
modal shows the survivor pick (highest `recall_count`, ties broken by older
`created_at`, matching `consolidate()`'s existing rule) before committing.

## Full keymap

| Key | Memories mode | Documents mode |
|---|---|---|
| `j`/`k`, `gg`/`G` | navigate list | same |
| `enter`/`tab`/`l`, `esc`/`h` | focus detail / back to list | same |
| `x` | soft-delete selected (`store.delete`) | soft-delete selected (`store.delete_doc`) |
| `Tab` | switch to Documents mode | switch to Memories mode |
| `/` | fuzzy filter | fuzzy filter |
| `s` | semantic search w/ scores | semantic search w/ scores (title+body) |
| `v` / `space` / `M` | multi-select / merge | — (not applicable) |
| `m` | — | toggle markdown render |
| `y` | copy content to clipboard | copy body to clipboard |
| `t` | edit tags (modal, like handoff's rename modal) | edit tags |
| `q` | quit | quit |

Delete always soft-deletes (`deleted_at` timestamp) — matches `core.py`'s
"never hard-delete" pattern throughout.

## Testing

Following the repo's existing test conventions (`tests/test_*.py`,
`pytest`), add:
- `tests/test_core.py` additions (or a new `test_merge.py`) covering
  `merge_memories()`: correct survivor pick, tag union, lineage edge
  recorded, loser soft-deleted, and that `consolidate()`'s pairwise
  behavior is unchanged after the `_merge_into` extraction.
- A minimal `tests/test_tui.py` smoke test in the style of handoff's
  `test_tui.py`, if one exists to mirror — otherwise a basic
  compose/mount/query smoke test is sufficient; deep Textual interaction
  testing is not a goal of this pass.
