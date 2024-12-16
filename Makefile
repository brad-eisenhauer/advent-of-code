PYTHON := python3


.PHONY: help
help: ## Display this help message.
	@grep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort -k1

.PHONY: format
FMT_PATH ?= test src
format: ## Format source files with ruff.
	ruff format $(FMT_PATH)

.PHONY: test
test: ## Run all tests.
	$(PYTHON) -m pytest --cov=advent_of_code.util test

.PHONY: install
install: ## Install the current package plus dependencies into the current environment.
	$(PYTHON) -m pip install --editable .

.PHONY: clean
clean:
	find ./src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f .coverage
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache

.PHONY: lint
LINT_PATH ?= src test
lint:  ## Run linting checks with ruff.
	ruff check $(LINT_PATH)

.PHONY: autolint
autolint:  ## Try to fix linting issues with ruff.
	ruff check --fix $(LINT_PATH)

.PHONY: venv
venv: install-uv  ## Build a development environment.
	[ -f .venv/bin/activate ] || uv venv .venv
	. .venv/bin/activate && uv lock && uv sync

.PHONY: install-uv
install-uv:
	@command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
