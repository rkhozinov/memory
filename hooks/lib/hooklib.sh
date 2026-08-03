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

mem_index_file() {
  printf '%s' "${MEMORY_INDEX_FILE:-$HOME/.claude/memory/INDEX.md}"
}

# Rebuild the curated memory catalog that SessionStart injects.
#
# The caps matter: the default 200 lines / 4000 tokens produced a 17KB file, so
# every session paid ~4.2k tokens for a catalog that is mostly about other
# projects. 60 lines / 1200 tokens keeps the hashes (and therefore the
# `memory get <hash>` drill-down) at roughly a third of the cost.
mem_rebuild_index() {
  local out marker total
  out="$(mem_index_file)"
  marker="$(mem_state_dir)/last-index-build"
  mkdir -p "$(dirname "$out")" "$(mem_state_dir)" 2>/dev/null || return 0
  memory admin index --out "$out" \
    --max-lines "${MEMORY_INDEX_MAX_LINES:-60}" \
    --max-tokens "${MEMORY_INDEX_MAX_TOKENS:-1200}" >/dev/null 2>&1 || return 0
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
