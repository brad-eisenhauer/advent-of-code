.PHONY: help
help: ## Display this help message.
	@grep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort -k1

.PHONY: format
format: ## Format source files with isort and black.
	isort src
	black src

.PHONY: test
test: ## Run all tests.
	pytest --cov=advent_of_code.util

.PHONY: install
install: ## Install the current package plus dependencies into the current environment.
	python -m pip install --editable .

.PHONY: install-no-deps
install-no-deps: ## Install the current package without dependencies into the current environment.
	python -m pip install --no-deps --editable .

.PHONY: clean-cache
clean-cache:
	find ./src -type d -name __pycache__ -prune -exec rm -rf '{}' \;

.PHONY: lint
lint:
	ruff ./src

.PHONY: autolint
autolint:
	ruff --fix ./src
