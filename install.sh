#!/usr/bin/env bash
#
# Install memory: pip install + symlink skills and hooks into ~/.claude/
#
# Usage: ./install.sh
#
# Idempotent — safe to re-run after updates.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_DIR="$HOME/.claude"
SKILLS_DIR="$CLAUDE_DIR/skills"
HOOKS_DIR="$CLAUDE_DIR/hooks"

echo "Installing memory from $REPO_DIR"

# 1. Install Python package (editable)
if command -v uv &>/dev/null; then
    echo "  [pip] uv pip install -e ."
    (cd "$REPO_DIR" && uv pip install -e ".[test]")
else
    echo "  [pip] pip install -e ."
    (cd "$REPO_DIR" && pip install -e ".[test]")
fi

# 2. Symlink skills
echo "  [skills] Symlinking into $SKILLS_DIR/"
mkdir -p "$SKILLS_DIR"
for skill_dir in "$REPO_DIR"/skills/*/; do
    skill_name=$(basename "$skill_dir")
    target="$SKILLS_DIR/$skill_name"

    if [ -L "$target" ]; then
        # Already a symlink — update it
        rm "$target"
    elif [ -d "$target" ]; then
        # Real directory exists — back up and replace
        echo "    $skill_name: backing up existing dir to ${target}.bak"
        mv "$target" "${target}.bak"
    fi

    ln -s "$skill_dir" "$target"
    echo "    $skill_name -> $skill_dir"
done

# 3. Symlink hooks
echo "  [hooks] Symlinking into $HOOKS_DIR/"
mkdir -p "$HOOKS_DIR"
for hook_file in "$REPO_DIR"/hooks/*.sh; do
    hook_name=$(basename "$hook_file")
    target="$HOOKS_DIR/$hook_name"

    if [ -L "$target" ]; then
        rm "$target"
    elif [ -f "$target" ]; then
        echo "    $hook_name: backing up existing file to ${target}.bak"
        mv "$target" "${target}.bak"
    fi

    ln -s "$hook_file" "$target"
    echo "    $hook_name -> $hook_file"
done

echo ""
echo "Done. Skills and hooks are symlinked — edits in $REPO_DIR are live immediately."
