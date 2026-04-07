#!/usr/bin/env python3
"""
check_duplicates.py - Detect duplicate tool entries in the Awesome LLM Tools README.

Severity rules:
  ERROR   - Exact name match (case-insensitive)
  ERROR   - Exact URL match (after normalization)
  WARNING - Similar names above similarity threshold
  WARNING - Same domain (excluding common hosting platforms)

Only ERROR level issues will cause CI to fail.
"""

import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

README_PATH = Path("README.md")

# Domains that host many unrelated tools - excluded from same-domain checks
SHARED_HOSTING_DOMAINS = {
    "github.com",
    "github.io",
    "huggingface.co",
    "arxiv.org",
    "ai.google.dev",
    "azure.microsoft.com",
}

# Similarity threshold for fuzzy name matching (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.90

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
    normalized_url: str
    domain: str
    section: str


@dataclass
class DuplicateIssue:
    """Represents a duplicate relationship between two tool entries."""

    tool_a: ToolEntry
    tool_b: ToolEntry
    severity: str          # "ERROR" or "WARNING"
    reason: str
    detail: str = ""


@dataclass
class DuplicateResult:
    """Holds all duplicate issues found during a check run."""

    issues: list[DuplicateIssue] = field(default_factory=list)

    def add_error(
        self, tool_a: ToolEntry, tool_b: ToolEntry, reason: str, detail: str = ""
    ) -> None:
        self.issues.append(DuplicateIssue(tool_a, tool_b, "ERROR", reason, detail))

    def add_warning(
        self, tool_a: ToolEntry, tool_b: ToolEntry, reason: str, detail: str = ""
    ) -> None:
        self.issues.append(DuplicateIssue(tool_a, tool_b, "WARNING", reason, detail))

    @property
    def errors(self) -> list[DuplicateIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[DuplicateIssue]:
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
      - Extracting tool entries with their section context
      - Normalizing URLs for comparison
    """

    TOOL_PATTERN = re.compile(
        r"\*\*\[([^\]]+)\]\(([^)]+)\)\*\*([^\n]*)"
    )

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
        for match in self.TOOL_PATTERN.finditer(content):
            name = match.group(1).strip()
            url = match.group(2).strip()
            section = self._find_section(content, match.start())
            normalized = self._normalize_url(url)
            domain = self._extract_domain(url)
            tools.append(ToolEntry(name, url, normalized, domain, section))
        return tools

    def _find_section(self, content: str, position: int) -> str:
        """Find the section heading that contains the given position."""
        section_start = content.rfind("\n## ", 0, position)
        if section_start == -1:
            return "Unknown"
        section_end = content.find("\n", section_start + 1)
        return content[section_start + 4:section_end].strip()

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL for consistent comparison."""
        parsed = urllib.parse.urlparse(url.lower())
        hostname = parsed.hostname or ""
        if hostname.startswith("www."):
            hostname = hostname[4:]
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{hostname}{path}"

    def _extract_domain(self, url: str) -> str:
        """Extract the domain from a URL."""
        try:
            domain = urllib.parse.urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except ValueError:
            return url


# ------------------------------------------------------------------------------
# DuplicateChecker
# ------------------------------------------------------------------------------

class DuplicateChecker:
    """
    Checks a list of ToolEntry objects for duplicates.

    Responsibilities:
      - Exact name duplicate detection (ERROR)
      - Exact URL duplicate detection (ERROR)
      - Similar name detection via fuzzy matching (WARNING)
      - Same domain detection (WARNING)
    """

    def check(self, tools: list[ToolEntry]) -> DuplicateResult:
        """Run all duplicate checks and return the result."""
        result = DuplicateResult()
        self._check_exact_names(tools, result)
        self._check_exact_urls(tools, result)
        self._check_similar_names(tools, result)
        self._check_same_domain(tools, result)
        return result

    def _check_exact_names(
        self, tools: list[ToolEntry], result: DuplicateResult
    ) -> None:
        """Flag tools with identical names (case-insensitive) as errors."""
        seen: dict[str, ToolEntry] = {}
        for tool in tools:
            key = tool.name.lower()
            if key in seen:
                result.add_error(
                    seen[key], tool, "Exact name match (case-insensitive)"
                )
            else:
                seen[key] = tool

    def _check_exact_urls(
        self, tools: list[ToolEntry], result: DuplicateResult
    ) -> None:
        """Flag tools with identical normalized URLs as errors."""
        seen: dict[str, ToolEntry] = {}
        for tool in tools:
            key = tool.normalized_url
            if key in seen:
                result.add_error(seen[key], tool, "Exact URL match")
            else:
                seen[key] = tool

    def _check_similar_names(
        self, tools: list[ToolEntry], result: DuplicateResult
    ) -> None:
        """Flag tools with very similar names as warnings."""
        for i, tool_a in enumerate(tools):
            for tool_b in tools[i + 1:]:
                sim = SequenceMatcher(
                    None, tool_a.name.lower(), tool_b.name.lower()
                ).ratio()
                if sim >= SIMILARITY_THRESHOLD:
                    result.add_warning(
                        tool_a,
                        tool_b,
                        "Similar names",
                        f"similarity: {sim:.0%}",
                    )

    def _check_same_domain(
        self, tools: list[ToolEntry], result: DuplicateResult
    ) -> None:
        """Flag tools sharing a domain as warnings, excluding shared hosts."""
        seen: dict[str, ToolEntry] = {}
        for tool in tools:
            domain = tool.domain
            if domain in SHARED_HOSTING_DOMAINS:
                continue
            if domain in seen:
                result.add_warning(
                    seen[domain],
                    tool,
                    "Same domain",
                    domain,
                )
            else:
                seen[domain] = tool


# ------------------------------------------------------------------------------
# DuplicateReport
# ------------------------------------------------------------------------------

class DuplicateReport:
    """
    Formats and prints the duplicate check results.

    Responsibilities:
      - Grouping and displaying issues by severity
      - Printing a clean summary
      - Returning the appropriate exit code
    """

    def print(self, tools: list[ToolEntry], result: DuplicateResult) -> int:
        """Print the full duplicate report and return exit code."""
        print("DUPLICATE TOOL CHECKER REPORT")
        print(SEPARATOR)
        print(f"Tools checked   : {len(tools)}")
        print(f"Errors found    : {len(result.errors)}")
        print(f"Warnings found  : {len(result.warnings)}")
        print(SEPARATOR)

        if not result.issues:
            print(f"\nPASSED: All {len(tools)} tools are unique.")
            return 0

        if result.errors:
            print("\nERRORS - must be resolved before merging:")
            self._print_issues(result.errors)

        if result.warnings:
            print("\nWARNINGS - review recommended but will not fail CI:")
            self._print_issues(result.warnings)

        self._print_summary(result)

        return 1 if result.has_errors() else 0

    def _print_issues(self, issues: list[DuplicateIssue]) -> None:
        for i, issue in enumerate(issues, 1):
            detail = f" ({issue.detail})" if issue.detail else ""
            print(f"\n  {i}. {issue.reason}{detail}")
            print(f"     a. [{issue.tool_a.name}]({issue.tool_a.url})")
            print(f"        Section: {issue.tool_a.section}")
            print(f"     b. [{issue.tool_b.name}]({issue.tool_b.url})")
            print(f"        Section: {issue.tool_b.section}")

    def _print_summary(self, result: DuplicateResult) -> None:
        print(f"\n{SEPARATOR}")
        print("SUMMARY")
        print(SEPARATOR)
        print(f"  Errors   : {len(result.errors)}")
        print(f"  Warnings : {len(result.warnings)}")
        print()
        print("  Recommended actions:")
        print("    - Remove exact duplicate entries immediately")
        print("    - Review similar names and same-domain warnings")
        print("    - See CONTRIBUTING.md for quality standards")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    print("Starting duplicate tool detection...\n")

    parser = ReadmeParser()
    tools = parser.parse()

    if not tools:
        print("ERROR: No tool entries found in README.md")
        sys.exit(1)

    checker = DuplicateChecker()
    result = checker.check(tools)

    report = DuplicateReport()
    exit_code = report.print(tools, result)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
