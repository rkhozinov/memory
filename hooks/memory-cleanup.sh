#!/usr/bin/env bash
#
# Memory Cleanup Report
# Analyzes Claude Code memory for duplicates, staleness, and unused entries
# NEVER auto-deletes — reporting only

set -euo pipefail

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Memory Cleanup Report ===${NC}\n"

# Step 1: Run memory cleanup and capture results
echo -e "${BLUE}[1/3] Running duplicate detection...${NC}"
if ! cleanup_output=$(memory cleanup 2>&1); then
    echo -e "${RED}Error running memory cleanup:${NC}"
    echo "$cleanup_output"
    echo ""
else
    # Parse JSON output if available
    if echo "$cleanup_output" | jq -e . >/dev/null 2>&1; then
        duplicates_removed=$(echo "$cleanup_output" | jq -r '.duplicatesRemoved // 0')
        duplicates_found=$(echo "$cleanup_output" | jq -r '.duplicatesFound // 0')

        if [[ "$duplicates_removed" -gt 0 ]]; then
            echo -e "${GREEN}Removed $duplicates_removed duplicate(s)${NC}"
        else
            echo -e "${GREEN}No duplicates found${NC}"
        fi
    else
        # Fallback for non-JSON output
        echo "$cleanup_output"
    fi
    echo ""
fi

# Step 2: Get memory statistics
echo -e "${BLUE}[2/3] Memory statistics:${NC}"
if ! stats_output=$(memory admin stats 2>&1); then
    echo -e "${RED}Error getting memory stats:${NC}"
    echo "$stats_output"
    echo ""
else
    # Pretty-print JSON stats
    if echo "$stats_output" | jq -e . >/dev/null 2>&1; then
        echo "$stats_output" | jq -r '
            "Total memories: \(.total // "?")",
            "By type: \(.by_type // {} | to_entries | map("\(.key): \(.value)") | join(", "))",
            "Never recalled: \(.never_recalled // "?")",
            "Stale (>30d): \(.stale // "?")"
        ' 2>/dev/null || echo "$stats_output"
    else
        echo "$stats_output"
    fi
    echo ""
fi

# Step 3: Parse stats and provide recommendations
echo -e "${BLUE}[3/3] Recommendations:${NC}"

# Extract key metrics from JSON stats output
if [[ -n "${stats_output:-}" ]] && echo "$stats_output" | jq -e . >/dev/null 2>&1; then
    never_recalled=$(echo "$stats_output" | jq -r '.never_recalled // 0')
    stale_count=$(echo "$stats_output" | jq -r '.stale // 0')
    total_count=$(echo "$stats_output" | jq -r '.total // 0')

    recommendations=()

    if [[ -n "$never_recalled" ]] && [[ "$never_recalled" -gt 0 ]]; then
        recommendations+=("${YELLOW}$never_recalled memories never recalled${NC} — Review with: memory list --filter 'recallCount:0'")
    fi

    if [[ -n "$stale_count" ]] && [[ "$stale_count" -gt 0 ]]; then
        recommendations+=("${YELLOW}$stale_count stale memories (>30 days)${NC} — Review with: memory list --sort 'lastRecalledAt:asc'")
    fi

    if [[ -n "$total_count" ]] && [[ "$total_count" -gt 100 ]]; then
        recommendations+=("${YELLOW}Large memory store ($total_count entries)${NC} — Consider archiving old project-specific memories")
    fi

    if [[ ${#recommendations[@]} -eq 0 ]]; then
        echo -e "${GREEN}Memory store looks healthy!${NC}"
    else
        for rec in "${recommendations[@]}"; do
            echo -e "  - $rec"
        done
    fi
else
    echo -e "${YELLOW}Could not parse stats — run 'memory admin stats' manually for details${NC}"
fi

echo ""
echo -e "${BLUE}=== End of Report ===${NC}"
echo ""
echo "To review specific memories: memory search [--tags <tag>] [--types <type>]"
echo "To delete a memory: memory delete <id>"
echo "Never run automated cleanup without manual review."

# Index refresh trigger
# Count new memories since last index build marker; rebuild in background if >= 25
LAST_INDEX_MARKER="$HOME/repos/memory/data/.last-index-build"
INDEX_FILE="$HOME/.claude/memory/INDEX.md"

INDEX_REFRESH_THRESHOLD=25

if [[ ! -f "$LAST_INDEX_MARKER" ]]; then
  # No marker: always rebuild
  echo ""
  echo -e "${BLUE}[index] No previous index marker — building index in background...${NC}"
  mkdir -p "$(dirname "$INDEX_FILE")"
  (memory admin index --out "$INDEX_FILE" >/dev/null 2>&1 && touch "$LAST_INDEX_MARKER" &)
else
  # Count memories created/updated since marker mtime
  MARKER_TS=$(python3 -c "import os; print(os.path.getmtime('$LAST_INDEX_MARKER'))" 2>/dev/null || echo "0")
  NEW_SINCE=$(memory admin stats 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('stores_total', 0))
except Exception:
    print(0)
" 2>/dev/null || echo "0")

  # Use a simpler approach: check raw DB count via memory health
  TOTAL_NOW=$(memory health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_memories',0))" 2>/dev/null || echo "0")
  TOTAL_AT_MARKER=$(cat "${LAST_INDEX_MARKER}.count" 2>/dev/null || echo "0")
  DIFF=$(( TOTAL_NOW - TOTAL_AT_MARKER ))

  if [[ "$DIFF" -ge "$INDEX_REFRESH_THRESHOLD" ]]; then
    echo ""
    echo -e "${BLUE}[index] ${DIFF} new memories since last index — rebuilding in background...${NC}"
    mkdir -p "$(dirname "$INDEX_FILE")"
    (memory admin index --out "$INDEX_FILE" >/dev/null 2>&1 && touch "$LAST_INDEX_MARKER" && echo "$TOTAL_NOW" > "${LAST_INDEX_MARKER}.count" &)
  fi
fi
