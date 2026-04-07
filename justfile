# ------------------------------------------------------------------------------
#  Awesome LLM Tools - Task Runner
#  Prerequisites: just, uv, lychee (optional)
# ------------------------------------------------------------------------------

# List all available commands (default)
default:
    @just --list

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

# Set up the environment and install pre-commit hooks
install:
    uv sync
    uv run pre-commit install
    @echo "Environment ready. Run 'just check' to validate."

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

# Validate tool entry format in README.md
validate:
    uv run python scripts/validate_format.py

# Check for duplicate tool entries in README.md
duplicates:
    uv run python scripts/check_duplicates.py

# Run all validation checks
check: validate duplicates

# ------------------------------------------------------------------------------
# Linting
# ------------------------------------------------------------------------------

# Run pre-commit hooks on all files
lint:
    uv run pre-commit run --all-files

# Auto-fix markdownlint issues only
fix-md:
    uv run pre-commit run markdownlint --all-files

# ------------------------------------------------------------------------------
# Link Checking
# ------------------------------------------------------------------------------

# Check for dead links in README.md (requires: cargo install lychee)
links:
    @command -v lychee >/dev/null 2>&1 || { \
        echo "lychee not found. Install with: cargo install lychee"; exit 1; }
    lychee --no-progress README.md

# ------------------------------------------------------------------------------
# CI
# ------------------------------------------------------------------------------

# Run full CI suite (lint first, then validation checks)
ci: lint check

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

# Remove Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    @echo "Cleaned up cache files."

# Show project info
info:
    @echo "Project : Awesome LLM Tools"
    @echo "Python  : $(uv run python --version)"
    @echo "Scripts : scripts/validate_format.py, scripts/check_duplicates.py"
    @echo "Docs    : README.md, CONTRIBUTING.md, docs/tool-categories.md"
