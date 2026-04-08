#!/usr/bin/env python3
"""
validate_format.py - Validate tool entries in the Awesome LLM Tools README.

Parsing strategy: block-oriented, not line-oriented.

The document is first split into logical blocks by scanning for boundary
markers (section headers, subsection headers, tool entry headers). Each
tool entry block contains all text from the tool header down to the next
boundary of any kind. Fields are extracted from the entire block at once
using multi-line regex, so wrapped continuation lines are handled correctly
regardless of how the markdown formatter arranges them on disk.

Validation rules:
  - Every tool must have a star rating (2-5 stars)
  - Every tool must have a valid URL (http/https)
  - Every tool must have a "Best for" field
  - Every tool must have at least one description field:
    "What it does", "Models", or "Sizes"
  - All other fields are optional but validated if present
  - Badge characters must be from the approved set

Severity:
  ERROR   - fails CI, must be fixed before merging
  WARNING - reported but does not block CI
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

README_PATH = Path("README.md")

# Unicode code points stored as strings to avoid encoding issues in source
STAR_CHAR = "\u2b50"  # ⭐
VALID_BADGES = {
    "\U0001f4b0",  # 💰 free/open source
    "\U0001f4b5",  # 💵 paid/commercial
    "\U0001f504",  # 🔄 freemium
    "\U0001f680",  # 🚀 trending/new
    "\U0001f3e2",  # 🏢 enterprise
}

# The one field every tool must have
REQUIRED_FIELD = "Best for"

# At least one of these must be present — describes what the tool is
DESCRIPTION_FIELDS = {"What it does", "Models", "Sizes"}

# All known optional fields — validated if present, not required
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

# Sections that contain no tool entries — skipped entirely during parsing
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

SEPARATOR = "-" * 78


# ------------------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------------------


@dataclass
class ToolEntry:
    """A single parsed tool entry extracted from the README."""

    name: str
    url: str
    badges: str
    fields: dict[str, str]
    section: str


@dataclass
class ValidationIssue:
    """A single validation finding on a tool entry."""

    tool_name: str
    section: str
    severity: str  # "ERROR" or "WARNING"
    message: str


@dataclass
class ValidationResult:
    """Accumulates all issues found during a validation run."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, tool: ToolEntry, message: str) -> None:
        self.issues.append(ValidationIssue(tool.name, tool.section, "ERROR", message))

    def add_warning(self, tool: ToolEntry, message: str) -> None:
        self.issues.append(ValidationIssue(tool.name, tool.section, "WARNING", message))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def has_errors(self) -> bool:
        return bool(self.errors)


# ------------------------------------------------------------------------------
# ReadmeParser
# ------------------------------------------------------------------------------


class ReadmeParser:
    """
    Parses the README into ToolEntry objects using a block-oriented strategy.

    The parser works in two passes:

    Pass 1 - Boundary detection:
        Every line is scanned for structural markers:
        - "## Heading"     -> section boundary, updates current section context
        - "### Subheading" -> subsection boundary
        - "**[Name](url)** -> tool entry boundary

        Each boundary records its line index and the current section context.

    Pass 2 - Block assembly and field extraction:
        For each tool boundary, all lines from that boundary to the next
        boundary of any type are joined into a single multi-line string.
        Fields are then extracted from this whole block using a regex that
        searches across the entire string, not line by line.

        This means continuation lines (where a long field value wraps onto
        the next line with leading spaces) are handled transparently --
        the field value simply contains a space where the newline was.
    """

    # Matches the tool header line: **[Name](url)** badges
    TOOL_HEADER_RE = re.compile(r"^\*\*\[([^\]]+)\]\(([^)]+)\)\*\*(.*)$")

    # Matches a field line and captures everything until the next field or
    # end of block. re.DOTALL lets . match newlines so wrapped values are
    # captured as part of the same field.
    FIELD_RE = re.compile(
        r"-\s+\*\*([^*]+)\*\*:\s*(.*?)(?=\n\s*-\s+\*\*[^*]+\*\*:|\Z)",
        re.DOTALL,
    )

    # Matches a line containing only star and badge emoji plus spaces.
    # Used to detect badges that the formatter has wrapped onto a separate line
    # below the tool header instead of keeping them on the same line.
    BADGES_ONLY_RE = re.compile(
        r"^[\u2b50"  # star emoji
        r"\U0001f4b0"  # 💰
        r"\U0001f4b5"  # 💵
        r"\U0001f504"  # 🔄
        r"\U0001f680"  # 🚀
        r"\U0001f3e2"  # 🏢
        r"\s]+$"
    )

    def __init__(self, path: Path = README_PATH):
        self.path = path

    def read(self) -> str:
        """Read the README from disk."""
        if not self.path.exists():
            print(f"ERROR: {self.path} not found.")
            sys.exit(1)
        return self.path.read_text(encoding="utf-8")

    def parse(self) -> list[ToolEntry]:
        """Return all tool entries found in the README."""
        content = self.read()
        lines = content.splitlines()

        # Pass 1: locate every structural boundary and record section context
        boundaries = self._find_boundaries(lines)

        # Pass 2: for every tool boundary, build its block and parse it
        tools = []
        for idx, boundary in enumerate(boundaries):
            if boundary["type"] != "tool":
                continue

            # Skip tools that fall inside non-tool sections
            if any(skip in boundary["section"] for skip in NON_TOOL_SECTIONS):
                continue

            # The block runs from this boundary to the start of the next one
            start = boundary["line_index"]
            end = (
                boundaries[idx + 1]["line_index"]
                if idx + 1 < len(boundaries)
                else len(lines)
            )
            block = "\n".join(lines[start:end])

            tool = self._parse_block(block, boundary["section"])
            if tool is not None:
                tools.append(tool)

        return tools

    def _find_boundaries(self, lines: list[str]) -> list[dict]:
        """
        Scan every line and record the position and type of each boundary.

        A boundary is any line that begins a new logical unit:
        a section header, a subsection header, or a tool entry header.
        """
        boundaries = []
        current_section = "Unknown"

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("## ") and not stripped.startswith("### "):
                # Section header - extract clean title (strip trailing badges)
                raw_title = stripped[3:].strip()
                # Remove leading emoji-like characters to get plain text title
                current_section = raw_title
                boundaries.append(
                    {
                        "type": "section",
                        "line_index": i,
                        "section": current_section,
                    }
                )

            elif stripped.startswith("### "):
                boundaries.append(
                    {
                        "type": "subsection",
                        "line_index": i,
                        "section": current_section,
                    }
                )

            elif self.TOOL_HEADER_RE.match(stripped):
                boundaries.append(
                    {
                        "type": "tool",
                        "line_index": i,
                        "section": current_section,
                    }
                )

        return boundaries

    def _parse_block(self, block: str, section: str) -> ToolEntry | None:
        """
        Extract a ToolEntry from a block of text.

        The first line of the block is the tool header. All subsequent lines
        (including any wrapped continuation lines) are searched together for
        field patterns.
        """
        first_line = block.split("\n")[0].strip()
        header_match = self.TOOL_HEADER_RE.match(first_line)
        if not header_match:
            return None

        name = header_match.group(1).strip()
        url = header_match.group(2).strip()
        badges = header_match.group(3).strip()

        # If badges came out empty, the formatter may have wrapped them onto the
        # line immediately after the tool header — this happens most often when
        # the URL is very long and pushes the line over the formatter's width
        # limit. We look at the second line of the block and, if it contains
        # only
        # badge and star emoji, we treat it as the badges string.
        if not badges:
            block_lines = block.split("\n")
            if len(block_lines) > 1:
                second_line = block_lines[1].strip()
                if second_line and self.BADGES_ONLY_RE.match(second_line):
                    badges = second_line

        # Extract fields from the entire block in one pass.
        # The FIELD_RE captures everything from "- **FieldName**:" up to
        # the next field marker or end of block, so multi-line values are
        # captured correctly. We then collapse internal whitespace so that
        # "some long value\n  continued here" becomes "some long value continued here".
        fields: dict[str, str] = {}
        for m in self.FIELD_RE.finditer(block):
            field_name = m.group(1).strip()
            raw_value = m.group(2)
            # Collapse continuation lines: replace newline + leading spaces
            # with a single space, then strip the whole value
            clean_value = re.sub(r"\n\s+", " ", raw_value).strip()
            fields[field_name] = clean_value

        return ToolEntry(name, url, badges, fields, section)


# ------------------------------------------------------------------------------
# ToolValidator
# ------------------------------------------------------------------------------


class ToolValidator:
    """
    Validates a single ToolEntry against the project rules.

    Each check is a separate method so new rules can be added without
    touching the existing ones.
    """

    STAR_RE = re.compile(r"\u2b50+")

    def validate(self, tool: ToolEntry, result: ValidationResult) -> None:
        """Run all validation checks on one tool entry."""
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
        if not DESCRIPTION_FIELDS.intersection(tool.fields):
            result.add_error(
                tool,
                "Missing description field - must have one of: "
                + ", ".join(sorted(DESCRIPTION_FIELDS)),
            )

    def _check_star_rating(self, tool: ToolEntry, result: ValidationResult) -> None:
        matches = self.STAR_RE.findall(tool.badges)
        if not matches:
            # A completely empty badges string means the tool likely inherits
            # its rating from a group header (e.g. Llama model families).
            # Treat that as a warning rather than a hard error.
            if tool.badges.strip():
                result.add_error(tool, "Missing star rating")
            else:
                result.add_warning(
                    tool, "No star rating - may inherit rating from group header"
                )
        elif len(matches) > 1:
            result.add_error(tool, "Multiple star ratings found")
        else:
            count = len(matches[0])
            if count < 2 or count > 5:
                result.add_error(tool, f"Invalid star count: {count} (must be 2-5)")

    def _check_badges(self, tool: ToolEntry, result: ValidationResult) -> None:
        badges_no_stars = self.STAR_RE.sub("", tool.badges).strip()
        for char in badges_no_stars:
            if char not in VALID_BADGES and char != " ":
                result.add_warning(tool, f"Unrecognized badge: '{char}'")

    def _check_field_lengths(self, tool: ToolEntry, result: ValidationResult) -> None:
        for name, value in tool.fields.items():
            if len(value) < MIN_FIELD_LENGTH:
                result.add_warning(
                    tool,
                    f"Field '{name}' too short (minimum {MIN_FIELD_LENGTH} chars)",
                )
            elif len(value) > MAX_FIELD_LENGTH:
                result.add_warning(
                    tool,
                    f"Field '{name}' too long (maximum {MAX_FIELD_LENGTH} chars)",
                )

    def _check_unknown_fields(self, tool: ToolEntry, result: ValidationResult) -> None:
        known = DESCRIPTION_FIELDS | OPTIONAL_FIELDS | {REQUIRED_FIELD}
        for name in tool.fields:
            if name not in known:
                result.add_warning(tool, f"Unrecognized field: '{name}'")


# ------------------------------------------------------------------------------
# ValidationReport
# ------------------------------------------------------------------------------


class ValidationReport:
    """
    Formats and prints the validation results, then returns an exit code.

    Issues are grouped by (section, tool name) for readability.
    ERRORs fail CI; WARNINGs are informational only.
    """

    def print(self, tools: list[ToolEntry], result: ValidationResult) -> int:
        """Print the full report and return 0 (pass) or 1 (fail)."""
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
        # Group issues by section + tool name for a clean, scannable layout
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
    sys.exit(report.print(tools, result))


if __name__ == "__main__":
    main()
