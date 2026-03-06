#!/usr/bin/env bash
#
# UserPromptSubmit hook: Automatically recall relevant memories at session start
#
# Searches memory bank on the first prompt of each session and injects
# relevant context. Subsequent prompts skip via marker file.

INPUT_JSON=$(cat)
[ -z "$INPUT_JSON" ] && exit 0

# Parse JSON once, output tab-separated values
PARSED=$(echo "$INPUT_JSON" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', ''))
    print(d.get('session_id', 'x'))
except Exception:
    sys.exit(1)
" 2>/dev/null) || exit 0

PROMPT=$(echo "$PARSED" | sed -n '1p')
SESSION_ID=$(echo "$PARSED" | sed -n '2p')

[ -z "$PROMPT" ] && exit 0
MARKER="/tmp/claude-memory-recalled-${SESSION_ID}"
[ -f "$MARKER" ] && exit 0
echo "$PROMPT" | grep -qE "^/(remember|recall|forget)" && exit 0

OUTPUT=$(memory -f text search "$PROMPT" --limit 5 --min-similarity 0.45 --depth titles 2>/dev/null)
touch "$MARKER" 2>/dev/null
if [ -n "$OUTPUT" ]; then
  echo "Relevant memories from previous sessions:"
  echo "$OUTPUT"
fi
exit 0
