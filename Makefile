.PHONY: setup install test lint format clean

# Interpreter used to build the venv. Override if your default python3 is too old:
#   make setup PYTHON=python3.11
PYTHON ?= python3

# Create a virtualenv and install the package (editable) with dev tools.
setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"
	@echo "Done. Activate with: source .venv/bin/activate"

# Install/refresh deps into the current environment.
install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
