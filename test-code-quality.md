# Code Quality Tooling — Verification Checklist

Run these in a fresh Claude Code session from the repo root.

## 1. Install from scratch

```
make install
```

**Expected**: `uv sync` resolves deps, skills/hooks symlinked, then lint + format + security + tests all pass. Final line: `164 passed`.

## 2. Verify `make check` independently

```
make check
```

**Expected output** (4 lines, quiet on success):
```
All checks passed!
13 files already formatted
164 passed in ~1.3s
```

## 3. Verify individual targets

```
make lint
```
**Expected**: `All checks passed!`

```
make format-check
```
**Expected**: `13 files already formatted`

```
make security
```
**Expected**: no output (exit 0)

```
make test
```
**Expected**: `164 passed`

## 4. Verify ruff catches a real issue

Introduce a deliberate lint error and confirm ruff catches it:

```
# Add an unused import to cli.py
sed -i '' '1s/^/import os\n/' src/memory/cli.py

# Lint should fail
make lint
# Expected: F401 `os` imported but unused

# Auto-fix it
uv run ruff check --fix src/memory/cli.py

# Confirm clean again
make lint
```

## 5. Verify formatter catches style issues

```
# Break formatting
echo 'x=1' >> src/memory/models.py

# Format check should fail
make format-check
# Expected: "Would reformat: src/memory/models.py"

# Auto-fix
make format

# Confirm clean
make format-check

# Revert the junk line
git checkout src/memory/models.py
```

## 6. Verify skills are symlinked and live

```
ls -la ~/.claude/skills/{recall,remember,forget,status}
```
**Expected**: all 4 are symlinks pointing into the repo's `skills/`

```
ls -la ~/.claude/hooks/memory-*.sh
```
**Expected**: 3 hook symlinks pointing into the repo's `hooks/`

## 7. Verify CLI works

```
memory --help
memory health
memory admin stats
```
**Expected**: help text, health JSON, stats JSON.

## 8. Verify CLAUDE.md is discoverable

```
cat src/memory/CLAUDE.md | head -35
```
**Expected**: Commands section shows `make install`, `make check`, individual targets.

## Cleanup

Discard any test mutations:
```
git checkout .
```
