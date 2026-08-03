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

# Throttle check: skip if marker exists and is younger than INTERVAL_HOURS.
# INTERVAL_HOURS=0 means "always run" (useful for manual testing).
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
