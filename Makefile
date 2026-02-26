.PHONY: help install install-dev test test-cov lint format type-check check clean build pre-commit-install pre-commit-run all

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package dependencies
	uv pip install -e .

install-dev: ## Install package with dev dependencies
	uv pip install -e ".[dev]"

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=src/sklearn_other_metrics --cov-report=term-missing --cov-report=html --cov-report=xml

test-fast: ## Run tests without coverage
	pytest --no-cov -v

lint: ## Run ruff linter
	ruff check .

lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix .

format: ## Format code with ruff
	ruff format .

format-check: ## Check code formatting without making changes
	ruff format --check .

type-check: ## Run mypy type checking
	mypy src/

check: lint format-check type-check test ## Run all checks (lint, format, type-check, test)

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

clean: ## Remove build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean ## Build distribution packages
	uv pip install build
	python -m build

all: clean install-dev pre-commit-install check ## Setup dev environment and run all checks
