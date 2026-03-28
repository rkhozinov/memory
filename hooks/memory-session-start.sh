#!/bin/bash
# Memory system session start hook
# Runs health check, daily cleanup, and displays status

PROJECT_NAME=$(basename "$(pwd)")
MARKER_FILE="$HOME/.claude/tools/memory/data/.last-cleanup"

# 1. Health check with timeout
HEALTH_OUTPUT=$(timeout 2s memory health 2>/dev/null)
HEALTH_STATUS=$?

if [[ $HEALTH_STATUS -ne 0 ]]; then
  echo "Memory system initialized. Project: ${PROJECT_NAME}. Use /recall to browse."
  exit 0
fi

# Extract memory count from JSON
MEMORY_COUNT=$(echo "$HEALTH_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_memories',0))" 2>/dev/null)
MEMORY_COUNT=${MEMORY_COUNT:-0}

# 2. Daily cleanup deduplication check
if [[ ! -f "$MARKER_FILE" ]] || [[ -n $(find "$MARKER_FILE" -mtime +0 2>/dev/null) ]]; then
  mkdir -p "$(dirname "$MARKER_FILE")"
  touch "$MARKER_FILE"
  (memory admin cleanup >/dev/null 2>&1 &)
fi

# 3. Check if codebase map exists
MAP_EXISTS=$(memory search "codebase-map" --mode exact --tags "project:$PROJECT_NAME,scope:codebase-map" -n 1 2>/dev/null | python3 -c "import sys,json; r=json.load(sys.stdin); print('yes' if r else 'no')" 2>/dev/null)

# 4. Output status message
echo "Memory system active. Project: ${PROJECT_NAME}. ${MEMORY_COUNT} memories. Use /recall to browse."

if [[ "$MAP_EXISTS" != "yes" ]]; then
  echo "No codebase map indexed. Run /codebase index to create one."
else
  echo "Codebase map available (auto-loads on first code exploration, or use /codebase to view)."
fi

exit 0
