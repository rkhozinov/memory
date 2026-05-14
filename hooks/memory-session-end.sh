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

DREAM_DIR="$HOME/.claude/memory/dream"
MARKER_FILE="$DREAM_DIR/last_run.marker"
LOG_FILE="$DREAM_DIR/last_run.log"
INTERVAL_HOURS="${MEMORY_DREAM_INTERVAL_HOURS:-6}"

# Defensive: ensure dream dir exists.
mkdir -p "$DREAM_DIR" 2>/dev/null || exit 0

# Bail if the memory CLI is not available — nothing to do.
command -v memory >/dev/null 2>&1 || exit 0

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
