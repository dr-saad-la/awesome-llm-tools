#!/usr/bin/env python3
"""
validate_format.py - Validate tool entries in the Awesome LLM Tools README.

Validation rules:
  - Every tool must have a star rating (2-5 stars)
  - Every tool must have a valid URL (http/https)
  - Every tool must have a "Best for" field
  - Every tool must have at least one description field:
    "What it does", "Models", or "Sizes"
  - All other fields are optional but validated if present
  - Badge characters must be from the approved set
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

README_PATH = Path("README.md")

STAR_CHAR = "\u2b50"
VALID_BADGES = {"\U0001f4b0", "\U0001f4b5", "\U0001f504", "\U0001f680", "\U0001f3e2"}

# The one universally required field
REQUIRED_FIELD = "Best for"

# At least one of these must be present per tool entry
DESCRIPTION_FIELDS = {"What it does", "Models", "Sizes"}

# All known optional fields - validated if present, not required
OPTIONAL_FIELDS = {
    "Features",
    "Pricing",
    "Unique",
    "Languages",
    "License",
    "Hardware",
    "Platforms",
    "Performance",
    "Integration",
    "Note",
    "Developer",
    "Community",
    "Focus",
    "Deployment",
    "UI",
    "Use cases",
    "Impact",
    "Free tier",
    "Essential",
    "Quality",
    "Providers",
    "Enterprise",
    "Language",
    "Usage",
    "Documentation",
}

# Sections that contain no tool entries - skipped during parsing
NON_TOOL_SECTIONS = {
    "Table of Contents",
    "Legend",
    "Cost Comparison",
    "Recommended Stacks",
    "Learning Resources",
    "Contributing",
    "Repository Growth",
    "License",
}

# Field value length constraints
MIN_FIELD_LENGTH = 5
MAX_FIELD_LENGTH = 300

# Separator for report output
SEPARATOR = "-" * 78


# ------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------

@dataclass
class ToolEntry:
    """Represents a single parsed tool entry from the README."""

    name: str
    url: str
    badges: str
    fields: dict[str, str]
    section: str


@dataclass
class ValidationIssue:
    """Represents a single validation issue found on a tool entry."""

    tool_name: str
    section: str
    severity: str          # "ERROR" or "WARNING"
    message: str


@dataclass
class ValidationResult:
    """Holds all issues found during a validation run."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, tool: ToolEntry, message: str) -> None:
        self.issues.append(
            ValidationIssue(tool.name, tool.section, "ERROR", message)
        )

    def add_warning(self, tool: ToolEntry, message: str) -> None:
        self.issues.append(
            ValidationIssue(tool.name, tool.section, "WARNING", message)
        )

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def has_errors(self) -> bool:
        return len(self.errors) > 0


# ------------------------------------------------------------------------------
# ReadmeParser
# ------------------------------------------------------------------------------

class ReadmeParser:
    """
    Reads and parses the README into structured ToolEntry objects.

    Responsibilities:
      - Reading the README file from disk
      - Splitting content into sections
      - Extracting tool entries from each section
    """

    SECTION_PATTERN = re.compile(r"\n## ([^#\n]+).*?(?=\n## |\Z)", re.DOTALL)
    TOOL_PATTERN = re.compile(
        r"\*\*\[([^\]]+)\]\(([^)]+)\)\*\*([^\n]*)\n\n((?:- \*\*[^*]+\*\*:.*\n?)*)"
    )
    FIELD_PATTERN = re.compile(r"- \*\*([^*]+)\*\*:\s*(.+)")

    def __init__(self, path: Path = README_PATH):
        self.path = path

    def read(self) -> str:
        """Read the README file and return its contents."""
        if not self.path.exists():
            print(f"ERROR: {self.path} not found.")
            sys.exit(1)
        return self.path.read_text(encoding="utf-8")

    def parse(self) -> list[ToolEntry]:
        """Parse the README and return all tool entries."""
        content = self.read()
        tools = []
        for match in self.SECTION_PATTERN.finditer(content):
            section_title = match.group(1).strip()
            if any(skip in section_title for skip in NON_TOOL_SECTIONS):
                continue
            section_content = match.group(0)
            tools.extend(self._parse_tools(section_content, section_title))
        return tools

    def _parse_tools(self, content: str, section: str) -> list[ToolEntry]:
        """Extract tool entries from a single section."""
        tools = []
        for match in self.TOOL_PATTERN.finditer(content):
            name = match.group(1).strip()
            url = match.group(2).strip()
            badges = match.group(3).strip()
            fields_raw = match.group(4).strip()
            fields = {
                k.strip(): v.strip()
                for k, v in self.FIELD_PATTERN.findall(fields_raw)
            }
            tools.append(ToolEntry(name, url, badges, fields, section))
        return tools


# ------------------------------------------------------------------------------
# ToolValidator
# ------------------------------------------------------------------------------

class ToolValidator:
    """
    Validates a single ToolEntry against the project rules.

    Responsibilities:
      - Checking required fields
      - Checking description field presence
      - Validating star ratings
      - Validating badge characters
      - Validating URL format
      - Validating field value lengths
    """

    STAR_PATTERN = re.compile(r"\u2b50+")

    def validate(self, tool: ToolEntry, result: ValidationResult) -> None:
        """Run all validation checks on a tool entry."""
        self._check_url(tool, result)
        self._check_required_field(tool, result)
        self._check_description_field(tool, result)
        self._check_star_rating(tool, result)
        self._check_badges(tool, result)
        self._check_field_lengths(tool, result)
        self._check_unknown_fields(tool, result)

    def _check_url(self, tool: ToolEntry, result: ValidationResult) -> None:
        if not tool.url.startswith(("http://", "https://")):
            result.add_error(tool, f"Invalid URL: {tool.url}")

    def _check_required_field(self, tool: ToolEntry, result: ValidationResult) -> None:
        if REQUIRED_FIELD not in tool.fields:
            result.add_error(tool, f"Missing required field: '{REQUIRED_FIELD}'")
        elif not tool.fields[REQUIRED_FIELD]:
            result.add_error(tool, f"Empty required field: '{REQUIRED_FIELD}'")

    def _check_description_field(
        self, tool: ToolEntry, result: ValidationResult
    ) -> None:
        if not DESCRIPTION_FIELDS.intersection(tool.fields.keys()):
            result.add_error(
                tool,
                f"Missing description field - must have one of: "
                f"{', '.join(sorted(DESCRIPTION_FIELDS))}",
            )

    def _check_star_rating(self, tool: ToolEntry, result: ValidationResult) -> None:
        star_matches = self.STAR_PATTERN.findall(tool.badges)
        if not star_matches:
            if tool.badges.strip():
                result.add_error(tool, "Missing star rating")
            else:
                result.add_warning(tool, "No star rating - may inherit rating from group header")
        elif len(star_matches) > 1:
            result.add_error(tool, "Multiple star ratings found")
        else:
            count = len(star_matches[0])
            if count < 2 or count > 5:
                result.add_error(tool, f"Invalid star count: {count} (must be 2-5)")

    def _check_badges(self, tool: ToolEntry, result: ValidationResult) -> None:
        badges_no_stars = self.STAR_PATTERN.sub("", tool.badges).strip()
        for char in badges_no_stars:
            if char not in VALID_BADGES and char != " ":
                result.add_warning(tool, f"Unrecognized badge character: '{char}'")

    def _check_field_lengths(self, tool: ToolEntry, result: ValidationResult) -> None:
        for field_name, value in tool.fields.items():
            if len(value) < MIN_FIELD_LENGTH:
                result.add_warning(
                    tool,
                    f"Field '{field_name}' is too short (minimum {MIN_FIELD_LENGTH} chars)",
                )
            elif len(value) > MAX_FIELD_LENGTH:
                result.add_warning(
                    tool,
                    f"Field '{field_name}' is too long (maximum {MAX_FIELD_LENGTH} chars)",
                )

    def _check_unknown_fields(self, tool: ToolEntry, result: ValidationResult) -> None:
        all_known = DESCRIPTION_FIELDS | OPTIONAL_FIELDS | {REQUIRED_FIELD}
        for field_name in tool.fields:
            if field_name not in all_known:
                result.add_warning(tool, f"Unrecognized field: '{field_name}'")


# ------------------------------------------------------------------------------
# ValidationReport
# ------------------------------------------------------------------------------

class ValidationReport:
    """
    Formats and prints the validation results.

    Responsibilities:
      - Grouping issues by tool
      - Printing a clean, readable report
      - Returning an appropriate exit code
    """

    def print(self, tools: list[ToolEntry], result: ValidationResult) -> int:
        """Print the full validation report and return exit code."""
        print("TOOL FORMAT VALIDATION REPORT")
        print(SEPARATOR)
        print(f"Tools validated : {len(tools)}")
        print(f"Errors found    : {len(result.errors)}")
        print(f"Warnings found  : {len(result.warnings)}")
        print(SEPARATOR)

        if not result.issues:
            print(f"\nPASSED: All {len(tools)} tools are correctly formatted.")
            return 0

        self._print_issues(result)
        self._print_summary(result)

        return 1 if result.has_errors() else 0

    def _print_issues(self, result: ValidationResult) -> None:
        # Group issues by (section, tool_name)
        grouped: dict[str, list[ValidationIssue]] = {}
        for issue in result.issues:
            key = f"[{issue.section}] {issue.tool_name}"
            grouped.setdefault(key, []).append(issue)

        for key, issues in grouped.items():
            print(f"\n{key}")
            for issue in issues:
                print(f"  {issue.severity}: {issue.message}")

    def _print_summary(self, result: ValidationResult) -> None:
        print(f"\n{SEPARATOR}")
        print("SUMMARY")
        print(SEPARATOR)
        print(f"  Errors   : {len(result.errors)}")
        print(f"  Warnings : {len(result.warnings)}")
        print()
        print("  Recommended actions:")
        print("    - Fix all ERROR items before merging")
        print("    - Review WARNING items for quality improvements")
        print("    - See CONTRIBUTING.md for field format guidelines")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    print("Starting tool format validation...\n")

    parser = ReadmeParser()
    tools = parser.parse()

    if not tools:
        print("ERROR: No tool entries found in README.md")
        sys.exit(1)

    validator = ToolValidator()
    result = ValidationResult()

    for tool in tools:
        validator.validate(tool, result)

    report = ValidationReport()
    exit_code = report.print(tools, result)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
