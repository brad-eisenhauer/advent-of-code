.PHONY: format
format:
	isort src
	black src

.PHONY: test
test:
	pytest
