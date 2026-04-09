# ------------------------------------------------------------------------------
#  Awesome LLM Tools - Makefile
#  Alternative task runner for contributors who prefer make over just.
#  Prerequisites: make (pre-installed on macOS/Linux), uv, lychee (optional)
#
#  Usage:
#    make           Show all available targets
#    make install   Set up environment and pre-commit hooks
#    make check     Run all validation checks
#    make ci        Run full CI suite (lint + check)
# ------------------------------------------------------------------------------

# Declare targets that don't correspond to real files so make never
# confuses them with filesystem artifacts and always runs them.
.PHONY: help install validate duplicates check lint fix-md links ci clean info

# ------------------------------------------------------------------------------
# Default target — show available commands (mirrors `just --list`)
# ------------------------------------------------------------------------------

help:
	@echo ""
	@echo "Awesome LLM Tools — available targets:"
	@echo ""
	@echo "  Setup"
	@echo "    install      Set up the environment and install pre-commit hooks"
	@echo ""
	@echo "  Validation"
	@echo "    validate     Validate tool entry format in README.md"
	@echo "    duplicates   Check for duplicate tool entries in README.md"
	@echo "    check        Run all validation checks (validate + duplicates)"
	@echo ""
	@echo "  Linting"
	@echo "    lint         Run pre-commit hooks on all files"
	@echo "    fix-md       Auto-fix markdownlint issues only"
	@echo ""
	@echo "  Link Checking"
	@echo "    links        Check for dead links in README.md (requires lychee)"
	@echo ""
	@echo "  CI"
	@echo "    ci           Run full CI suite (lint then check)"
	@echo ""
	@echo "  Utilities"
	@echo "    clean        Remove Python cache files"
	@echo "    info         Show project info"
	@echo ""

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

install:
	uv sync
	uv run pre-commit install
	@echo "Environment ready. Run 'make check' to validate."

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

validate:
	uv run python scripts/validate_format.py

duplicates:
	uv run python scripts/check_duplicates.py

# Runs validate first, then duplicates — same dependency chain as just
check: validate duplicates

# ------------------------------------------------------------------------------
# Linting
# ------------------------------------------------------------------------------

lint:
	uv run pre-commit run --all-files

fix-md:
	uv run pre-commit run markdownlint --all-files

# ------------------------------------------------------------------------------
# Link Checking
# ------------------------------------------------------------------------------

links:
	@command -v lychee >/dev/null 2>&1 || { \
		echo "lychee not found. Install with: cargo install lychee"; exit 1; }
	lychee --no-progress README.md

# ------------------------------------------------------------------------------
# CI
# ------------------------------------------------------------------------------

# Mirrors `just ci` — lint must pass before validation runs
ci: lint check

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned up cache files."

info:
	@echo "Project : Awesome LLM Tools"
	@echo "Python  : $$(uv run python --version)"
	@echo "Scripts : scripts/validate_format.py, scripts/check_duplicates.py"
	@echo "Docs    : README.md, CONTRIBUTING.md, docs/tool-categories.md"
