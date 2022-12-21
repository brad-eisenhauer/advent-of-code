.PHONY: help
help: ## Display this help message.
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: format
format: ## Format source files with isort and black.
	isort src
	black src

.PHONY: test
test: ## Run all tests.
	pytest --cov=advent_of_code

.PHONY: install
install: ## Install the current package plus dependencies into the current environment.
	python -m pip install --editable .

.PHONY: install-no-deps
install-no-deps: ## Install the current package without dependencies into the current environment.
	python -m pip install --no-deps --editable .
