#!/usr/bin/env bash
#
# Shared helpers for the memory hooks.
#
# Source this from a hook with:
#   . "$(dirname "${BASH_SOURCE[0]}")/lib/hooklib.sh"

# Portable replacement for timeout(1), which is NOT installed on stock macOS.
#
# Using `timeout` directly is what killed the SessionStart hook in 1.10.0: the
# binary is absent, the command substitution returned 127, and the caller's
# `if [[ $? -ne 0 ]]` early-exit swallowed the entire rest of the hook. Nothing
# reported an error, so it looked like the hook was simply quiet.
#
# Usage:  out=$(mem_run 2 memory health); rc=$?
# Returns the command's own exit code, or 143 (SIGTERM) if the watchdog fired.
mem_run() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  # Watchdog: TERM, then KILL if it ignores TERM. Its own output is discarded so
  # it never holds the caller's stdout pipe open for the full timeout.
  { sleep "$secs"; kill -TERM "$pid"; sleep 0.3; kill -KILL "$pid"; } >/dev/null 2>&1 &
  local wd=$!
  local rc=0
  wait "$pid" 2>/dev/null || rc=$?
  kill "$wd" >/dev/null 2>&1
  wait "$wd" 2>/dev/null
  return $rc
}

mem_have() { command -v "$1" >/dev/null 2>&1; }

# Hook state (markers, per-session ledgers) lives here — NOT in the store's data
# dir, and NOT in a hardcoded ~/repos/memory path. The plugin is distributed via
# a marketplace; it cannot assume where the user cloned the source, or that they
# cloned it at all.
mem_state_dir() {
  printf '%s' "${MEMORY_HOOK_STATE_DIR:-$HOME/.claude/memory/state}"
}

# The tag the catalog is scoped to. Defaults to `project:<basename-of-cwd>`,
# which is the convention `/remember` and the extractor already tag with.
# Set MEMORY_INDEX_SCOPE="" to build an unscoped, global catalog.
# The tag is passed to `memory admin index --tags`, so it must match what is
# actually stored — verbatim, spaces and all. It is NOT a filename; sanitising
# it here would mean querying `project:my-repo` for memories tagged
# `project:my repo` and quietly matching nothing.
mem_project_tag() {
  if [ -n "${MEMORY_INDEX_SCOPE+x}" ]; then
    printf '%s' "$MEMORY_INDEX_SCOPE"
    return
  fi
  local base
  base=$(basename "$PWD")
  [ -z "$base" ] && return
  printf 'project:%s' "$base"
}

# A filesystem-safe slug for the tag. Separate from the tag itself: this one
# does end up in a path, so it takes the charset discipline.
#
# Until 1.11.5 a single filter did both jobs, so any project whose directory
# name held a space or a non-ASCII character was rejected outright and fell back
# to the shared global catalog — every such project clobbering the others'
# rebuilds, which is the exact failure per-scope files exist to prevent.
mem_scope_slug() {
  local tag slug
  tag="$(mem_project_tag)"
  [ -z "$tag" ] && return
  # Leading dots and dashes are stripped rather than rejected: "." and ".." would
  # resolve against the catalog directory itself or its parent, and a dotfile-
  # style project name is otherwise perfectly usable once they are gone.
  slug=$(printf '%s' "${tag#project:}" \
    | LC_ALL=C tr -c 'A-Za-z0-9._-' '-' \
    | sed 's/^[^A-Za-z0-9]*//')
  case "$slug" in
    "") printf '' ;;
    *) printf '%.64s' "$slug" ;;
  esac
}

# One catalog file per scope. A single shared INDEX.md would mean the last
# project to rebuild decides what every other project's session gets injected.
mem_index_file() {
  if [ -n "${MEMORY_INDEX_FILE:-}" ]; then
    printf '%s' "$MEMORY_INDEX_FILE"
    return
  fi
  local slug
  slug="$(mem_scope_slug)"
  if [ -n "$slug" ]; then
    printf '%s/.claude/memory/INDEX-%s.md' "$HOME" "$slug"
  else
    printf '%s/.claude/memory/INDEX.md' "$HOME"
  fi
}

# Freshness marker, per scope for the same reason as the catalog itself.
mem_index_marker() {
  local slug
  slug="$(mem_scope_slug)"
  if [ -n "$slug" ]; then
    printf '%s/last-index-build-%s' "$(mem_state_dir)" "$slug"
  else
    printf '%s/last-index-build' "$(mem_state_dir)"
  fi
}

# Rebuild the curated memory catalog that SessionStart injects.
#
# The caps matter: the default 200 lines / 4000 tokens produced a 17KB file, so
# every session paid ~4.2k tokens for a catalog that is mostly about other
# projects. 1200 tokens keeps the hashes (and therefore the `memory get <hash>`
# drill-down) at roughly a third of the cost.
#
# Only the TOKEN cap binds. At ~150 chars per entry the 1200-token budget is
# exhausted at ~32 lines, so --max-lines 60 never takes effect; it is a safety
# valve against pathologically short entries, not the operative limit. Change
# --max-tokens to change the size of the index.
mem_rebuild_index() {
  local out marker total tag
  out="$(mem_index_file)"
  marker="$(mem_index_marker)"
  tag="$(mem_project_tag)"
  mkdir -p "$(dirname "$out")" "$(mem_state_dir)" 2>/dev/null || return 0
  # --tags prioritises, it does not filter, so an unknown project still gets a
  # useful global catalog rather than an empty one.
  memory admin index --out "$out" \
    --max-lines "${MEMORY_INDEX_MAX_LINES:-60}" \
    --max-tokens "${MEMORY_INDEX_MAX_TOKENS:-1200}" \
    ${tag:+--tags "$tag"} >/dev/null 2>&1 || return 0
  touch "$marker" 2>/dev/null
  total=$(memory health 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("total_memories",0))' 2>/dev/null)
  [ -n "$total" ] && printf '%s\n' "$total" > "$marker.count" 2>/dev/null
  return 0
}

# Parse the hook's JSON payload into base64 fields, one per line.
#
# Each field is base64-encoded so that a value containing newlines occupies
# exactly one line. The 1.10.0 recall hook printed prompt and session_id raw on
# successive lines and read them back with `sed -n 1p` / `sed -n 2p`, so any
# multi-line prompt shifted every field: the search query became the prompt's
# first line and the "session id" became its second. That is how markers like
# `/tmp/claude-memory-recalled-      import sys,json` were created.
#
# Base64 rather than NUL separation because `$(...)` silently discards NUL
# bytes, which would reintroduce the same class of bug from a different angle.
#
# Field order: session_id, prompt, cwd.
mem_parse_hook_input() {
  printf '%s' "$1" | python3 -c '
import sys, json, base64
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
if not isinstance(d, dict):
    sys.exit(1)
for key in ("session_id", "prompt", "cwd"):
    print(base64.b64encode(str(d.get(key) or "").encode()).decode())
' 2>/dev/null
}

# Decode field N (1-based) of mem_parse_hook_input output.
mem_field() {
  printf '%s' "$1" | sed -n "${2}p" | base64 -d 2>/dev/null
}

# Echo the session id if it is safe to use as a single path component, else "".
#
# Fail closed: an id we cannot validate means no state directory, which means
# the hook declines to inject rather than reading or writing an attacker- (or
# bug-) chosen path.
# The leading-character class is not redundant with the charset check: "." and
# ".." pass a bare [A-Za-z0-9._-] filter and then resolve to the sessions
# directory itself or its parent.
mem_safe_session_id() {
  [ "${#1}" -gt 128 ] && { printf ''; return; }
  case "$1" in
    ""|[!A-Za-z0-9]*|*[!A-Za-z0-9._-]*) printf '' ;;
    *) printf '%s' "$1" ;;
  esac
}
