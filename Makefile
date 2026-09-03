.PHONY: install sync lint format format-check security test check reinstall \
        link-dev unlink-dev verify-runtime analytics bench-consolidate bench-replay

# The installed plugin lives in Claude Code's plugin cache, NOT in this checkout.
PLUGIN_CACHE := $(HOME)/.claude/plugins/cache/rkhozinov/memory

## Sync deps, run all checks, install the CLI on PATH
install: sync check reinstall

## Sync Python package + test/lint deps
sync:
	uv sync --group test --group lint

## Point the installed plugin's hooks/ at this checkout, for fast iteration.
##
## There is no `link` target any more. It used to symlink skills/* into
## ~/.claude/skills/ and hooks/*.sh into ~/.claude/hooks/, both of which are
## wrong now that this ships as a plugin: the skills copy shadows the plugin's
## namespaced memory:* skills with un-namespaced duplicates, and the plugin
## runtime reads $${CLAUDE_PLUGIN_ROOT}/hooks/, never ~/.claude/hooks/ — so the
## hook symlinks implied the hooks were live when they were not.
link-dev:
	@v=$$(ls -1 $(PLUGIN_CACHE) 2>/dev/null | sort -V | tail -1); \
	if [ -z "$$v" ]; then echo "plugin not installed under $(PLUGIN_CACHE)"; exit 1; fi; \
	d="$(PLUGIN_CACHE)/$$v"; \
	if [ -L "$$d/hooks" ]; then echo "already linked: $$d/hooks"; exit 0; fi; \
	mv "$$d/hooks" "$$d/hooks.orig" && ln -s "$$(pwd)/hooks" "$$d/hooks"; \
	echo "  $$d/hooks -> $$(pwd)/hooks   (run 'make unlink-dev' before publishing)"

## Restore the plugin cache's own hooks/.
unlink-dev:
	@v=$$(ls -1 $(PLUGIN_CACHE) 2>/dev/null | sort -V | tail -1); \
	d="$(PLUGIN_CACHE)/$$v"; \
	if [ ! -L "$$d/hooks" ]; then echo "not linked: $$d/hooks"; exit 0; fi; \
	rm "$$d/hooks" && mv "$$d/hooks.orig" "$$d/hooks"; \
	echo "  restored $$d/hooks"

## Confirm the code that actually runs matches this checkout.
verify-runtime:
	@v=$$(ls -1 $(PLUGIN_CACHE) 2>/dev/null | sort -V | tail -1); \
	echo "plugin cache: $(PLUGIN_CACHE)/$$v"; \
	diff -rq hooks "$(PLUGIN_CACHE)/$$v/hooks" && echo "  hooks:  in sync"; \
	diff -rq skills "$(PLUGIN_CACHE)/$$v/skills" && echo "  skills: in sync"
	@python3 -c "import json,os;d=json.load(open(os.path.expanduser('~/.claude/plugins/installed_plugins.json')));print('installed:', d['plugins']['memory@rkhozinov'])"

lint:
	uv run ruff check src/ tests/
	shellcheck -x hooks/*.sh hooks/lib/*.sh

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

security:
	uv run bandit -r src/ -s B608 -q

test:
	uv run pytest -q

## Reinstall global tool (memory + memory-mcp-server on PATH)
## MUST run after any code change — the PATH binary is a separate uv tool install, not the .venv editable copy.
reinstall:
	uv tool install --force --editable .

## Real-usage analytics: how the memory service performs in actual sessions.
## Read-only — snapshots the DB before opening it, never writes to ~/.claude.
analytics:
	uv run python benchmarks/session_analytics.py --json /tmp/memory-analytics.json

## Consolidation fact-retention per content_strategy
bench-consolidate:
	uv run python tests/bench_consolidate.py

## Retrieval A/B replay over the synthetic corpus.
## For the production-DB run see benchmarks/results/README.md — it needs a
## generated query set that must not be committed.
bench-replay:
	uv run python tests/bench_replay.py \
		--configs baseline,+rrf,+rrsb,+id,+csls,+best,+rerank,+rrsb+rr,+all \
		--json benchmarks/results/replay-synthetic-baseline.json

## Lint + format + security + tests
check: lint format-check security test
