#!/bin/bash
# Memory system session start hook
# Runs health check, daily cleanup, and displays status

PROJECT_NAME=$(basename "$(pwd)")
MARKER_FILE="$HOME/.claude/tools/memory/data/.last-cleanup"

# 1. Health check with timeout
HEALTH_OUTPUT=$(timeout 2s memory health 2>/dev/null)
HEALTH_STATUS=$?

if [[ $HEALTH_STATUS -ne 0 ]]; then
  # Fallback if health check fails/times out
  echo "Memory system initialized. Project: ${PROJECT_NAME}. Use /recall to browse."
  exit 0
fi

# Extract memory count from JSON output using grep/sed
# Example: {"status": "healthy", "total_memories": 193, ...}
MEMORY_COUNT=$(echo "$HEALTH_OUTPUT" | grep -o '"total_memories":[[:space:]]*[0-9]*' | sed 's/[^0-9]//g')

# Default to 0 if extraction failed
if [[ -z "$MEMORY_COUNT" ]]; then
  MEMORY_COUNT=0
fi

# 2. Daily cleanup deduplication check
# Run cleanup if marker is missing or older than 24 hours
if [[ ! -f "$MARKER_FILE" ]] || [[ -n $(find "$MARKER_FILE" -mtime +0 2>/dev/null) ]]; then
  # Create marker directory if it doesn't exist
  mkdir -p "$(dirname "$MARKER_FILE")"

  # Touch marker first to prevent race conditions
  touch "$MARKER_FILE"

  # Run cleanup in background (non-blocking)
  (memory cleanup >/dev/null 2>&1 &)
fi

# 3. Check if codebase map exists
MAP_COUNT=$(memory -f text list --tags "project:$PROJECT_NAME,scope:codebase-map" --page-size 1 2>/dev/null | head -1 || echo "")

# 4. Output status message
echo "Memory system active. Project: ${PROJECT_NAME}. ${MEMORY_COUNT} memories. Use /recall to browse."

if echo "$MAP_COUNT" | grep -q "^0 "; then
  echo "No codebase map indexed. Run /codebase index to create one."
else
  echo "Codebase map available (auto-loads on first code exploration, or use /codebase to view)."
fi

exit 0
