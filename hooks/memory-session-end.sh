#!/usr/bin/env bash
#
# Memory session end hook
# Periodically runs `memory admin dream` (consolidation/decay/forget) in the
# background, throttled via a marker file so we don't churn on every session.
#
# Safe by design:
#   - never blocks session end (always exits 0)
#   - all failures are swallowed (logged to a side file)
#   - throttle interval configurable via MEMORY_DREAM_INTERVAL_HOURS (default 6)

set -euo pipefail

# Catch any unexpected error and exit 0 — SessionEnd must never fail.
trap 'exit 0' ERR

[ -n "${MEMORY_EXTRACTOR:-}" ] && exit 0

# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib/hooklib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib/hooklib.sh"

DREAM_DIR="$HOME/.claude/memory/dream"
MARKER_FILE="$DREAM_DIR/last_run.marker"
LOG_FILE="$DREAM_DIR/last_run.log"
INTERVAL_HOURS="${MEMORY_DREAM_INTERVAL_HOURS:-6}"

# Defensive: ensure dream dir exists.
mkdir -p "$DREAM_DIR" 2>/dev/null || exit 0

# Claude Code hands SessionEnd the id of the session that just finished. That is
# the whole point of this hook: no searching, no idle heuristic, no guessing
# which transcript is "done" — the harness already told us.
END_SESSION_ID=""
if [ ! -t 0 ]; then
  HOOK_INPUT=$(cat 2>/dev/null || true)
  if [ -n "$HOOK_INPUT" ]; then
    PARSED=$(mem_parse_hook_input "$HOOK_INPUT" 2>/dev/null || true)
    [ -n "$PARSED" ] && END_SESSION_ID=$(mem_safe_session_id "$(mem_field "$PARSED" 1)")
  fi
fi

# Bail if the memory CLI is not available — nothing to do.
command -v memory >/dev/null 2>&1 || exit 0

# Index refresh, moved here from the Stop hook in 1.11.0. It used to run after
# every assistant turn, which cost a `memory health` subprocess per turn and
# printed a coloured report nobody asked for. Session end is the right moment.
# Placed BEFORE the dream throttle, which exits early.
INDEX_MARKER="$(mem_state_dir)/last-index-build"
INDEX_THRESHOLD="${MEMORY_INDEX_REFRESH_THRESHOLD:-25}"
if [[ ! -f "$INDEX_MARKER" ]]; then
  (mem_rebuild_index >/dev/null 2>&1 &)
else
  TOTAL_NOW=$(memory health 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("total_memories",0))' 2>/dev/null || echo 0)
  TOTAL_AT=$(cat "${INDEX_MARKER}.count" 2>/dev/null || echo 0)
  [[ "$TOTAL_NOW" =~ ^[0-9]+$ ]] || TOTAL_NOW=0
  [[ "$TOTAL_AT"  =~ ^[0-9]+$ ]] || TOTAL_AT=0
  if (( TOTAL_NOW - TOTAL_AT >= INDEX_THRESHOLD )); then
    (mem_rebuild_index >/dev/null 2>&1 &)
  fi
fi

# Archive the session that just ended, right now.
#
# SessionStart also runs this, but only as a sweeper: it skips transcripts
# younger than five minutes, so the session you are leaving is not archived until
# the next time you open Claude Code *in this same directory*. Finish a piece of
# work, never return to that project, and it was never saved. `--min-age-minutes
# 0` closes that: the current transcript is archived at the moment it ends.
#
# The SessionStart sweep stays, and is still load-bearing — SessionEnd does not
# fire on a crash, a `kill -9`, or a closed terminal pane. Markers make the two
# idempotent, so whichever gets there first wins.
#
# Everything below this point must stay ABOVE the dream throttle, which exits 0.
(memory admin auto-archive-pending --cwd "$PWD" --min-age-minutes 0 >/dev/null 2>&1 &) >/dev/null 2>&1

# Distil the session that just ended into atomic memories. Opt-in: does nothing
# unless MEMORY_EXTRACT=1.
#
# `--idle-hours 0` is the point. SessionEnd IS the "this session is finished"
# signal, so there is nothing to wait for: the transcript is complete and the
# work is fresh. The six-hour idle default only makes sense for a manual run
# draining a backlog, and it meant the session you just did was the one session
# never distilled.
#
# Markers make this idempotent, so a session already handled here is skipped.
if [[ "${MEMORY_EXTRACT:-0}" == "1" ]]; then
  EXTRACT_DIR="$HOME/.claude/memory/extract"
  mkdir -p "$EXTRACT_DIR" 2>/dev/null
  if [[ -n "$END_SESSION_ID" ]]; then
    # Exactly the session that just ended. `_pending` returns oldest-first, so
    # without this a capped run distils the backlog and reaches the session you
    # actually just did last, or not at all.
    (memory admin extract-pending --cwd "$PWD" --session "$END_SESSION_ID" \
       >"$EXTRACT_DIR/last_run.log" 2>&1 &) >/dev/null 2>&1
  else
    # No session id in the payload — fall back to the backlog sweep.
    (memory admin extract-pending --cwd "$PWD" --idle-hours 0 \
       >"$EXTRACT_DIR/last_run.log" 2>&1 &) >/dev/null 2>&1
  fi
fi

# Throttle check: skip if marker exists and is younger than INTERVAL_HOURS.
# INTERVAL_HOURS=0 means "always run" (useful for manual testing).
# NOTE: this exits 0 when throttled, so nothing that must run every session may
# be placed after it.
if [[ "$INTERVAL_HOURS" != "0" && -f "$MARKER_FILE" ]]; then
  NOW=$(date +%s)
  LAST=$(cat "$MARKER_FILE" 2>/dev/null || echo 0)
  # Guard against non-numeric marker content.
  if ! [[ "$LAST" =~ ^[0-9]+$ ]]; then
    LAST=0
  fi
  AGE=$(( NOW - LAST ))
  INTERVAL_SECONDS=$(( INTERVAL_HOURS * 3600 ))
  if (( AGE < INTERVAL_SECONDS )); then
    exit 0
  fi
fi

# Run dream in background. Record marker on success.
# We deliberately decouple from the parent shell so session teardown isn't blocked.
(
  if memory admin dream >"$LOG_FILE" 2>&1; then
    date +%s > "$MARKER_FILE" 2>/dev/null || true
  fi
) >/dev/null 2>&1 &

exit 0
