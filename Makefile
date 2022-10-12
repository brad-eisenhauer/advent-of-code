.PHONY: format
format:
	isort src
	black src

.PHONY: test
test:
	pytest --cov=advent_of_code
