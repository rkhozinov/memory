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
if ! stats_output=$(memory -f text stats 2>&1); then
    echo -e "${RED}Error getting memory stats:${NC}"
    echo "$stats_output"
    echo ""
else
    echo "$stats_output"
    echo ""
fi

# Step 3: Parse stats and provide recommendations
echo -e "${BLUE}[3/3] Recommendations:${NC}"

# Extract key metrics from stats output (if available)
if [[ -n "${stats_output:-}" ]]; then
    # Try to extract never recalled count
    never_recalled=$(echo "$stats_output" | grep -i "never recalled" | grep -oE '[0-9]+' | head -1 || echo "")

    # Try to extract stale count (>30 days)
    stale_count=$(echo "$stats_output" | grep -i "stale" | grep -oE '[0-9]+' | head -1 || echo "")

    # Try to extract total count
    total_count=$(echo "$stats_output" | grep -i "total" | grep -oE '[0-9]+' | head -1 || echo "")

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
    echo -e "${YELLOW}Could not parse stats — run 'memory stats' manually for details${NC}"
fi

echo ""
echo -e "${BLUE}=== End of Report ===${NC}"
echo ""
echo "To review specific memories: memory list [--tags <tag>] [--filter <field>:<value>]"
echo "To delete a memory: memory delete <id>"
echo "Never run automated cleanup without manual review."
