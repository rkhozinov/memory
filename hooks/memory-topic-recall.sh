#!/usr/bin/env bash
#
# UserPromptSubmit hook: Automatically recall relevant memories at session start
#
# Searches memory bank on the first prompt of each session and injects
# relevant context. Subsequent prompts skip via marker file.

INPUT_JSON=$(cat)
[ -z "$INPUT_JSON" ] && exit 0

# Parse JSON once
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

# Search returns JSON — format to text with python
OUTPUT=$(memory search "$PROMPT" --limit 5 2>/dev/null | python3 -c "
import sys, json
try:
    results = json.load(sys.stdin)
    if not results:
        sys.exit(0)
    for r in results:
        h = r.get('content_hash', '')[:16]
        t = r.get('memory_type', 'note')
        s = r.get('score', 0)
        c = r.get('content', '')[:200].replace('\n', ' ')
        print(f'{h} [{t}] score={s:.2f} {c}')
except Exception:
    sys.exit(0)
" 2>/dev/null)

touch "$MARKER" 2>/dev/null
if [ -n "$OUTPUT" ]; then
  echo "Relevant memories from previous sessions:"
  echo "$OUTPUT"
fi
exit 0
