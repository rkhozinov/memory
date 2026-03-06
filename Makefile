.PHONY: install sync link lint format format-check security test check

CLAUDE_DIR := $(HOME)/.claude
SKILLS_DIR := $(CLAUDE_DIR)/skills
HOOKS_DIR  := $(CLAUDE_DIR)/hooks

## Sync deps, symlink skills/hooks, then run all checks
install: sync link check

## Sync Python package + test/lint deps
sync:
	uv sync --group test --group lint

## Symlink skills and hooks into ~/.claude/
link:
	@mkdir -p $(SKILLS_DIR) $(HOOKS_DIR)
	@for d in skills/*/; do \
		name=$$(basename "$$d"); \
		target="$(SKILLS_DIR)/$$name"; \
		rm -f "$$target"; \
		ln -sfn "$$(pwd)/$$d" "$$target"; \
		echo "  $$name -> $$d"; \
	done
	@for f in hooks/*.sh; do \
		name=$$(basename "$$f"); \
		target="$(HOOKS_DIR)/$$name"; \
		rm -f "$$target"; \
		ln -sf "$$(pwd)/$$f" "$$target"; \
		echo "  $$name -> $$f"; \
	done

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

format-check:
	uv run ruff format --check src/ tests/

security:
	uv run bandit -r src/ -s B608 -q

test:
	uv run pytest -q

## Lint + format + security + tests
check: lint format-check security test
