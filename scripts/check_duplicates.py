#!/usr/bin/env python3
"""
Check for duplicate tools in the Awesome LLM Tools README.md

This script identifies duplicate tool entries by:
1. Tool names (case-insensitive)
2. URLs (exact match)
3. Similar names (fuzzy matching)
"""

import re
import sys
import urllib.parse
from typing import List, Dict
from difflib import SequenceMatcher


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing common variations."""
    # Parse the URL
    parsed = urllib.parse.urlparse(url.lower())

    # Remove www. prefix
    hostname = parsed.hostname or ""
    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Remove trailing slashes and common suffixes
    path = parsed.path.rstrip("/")

    # Reconstruct normalized URL
    return f"{parsed.scheme}://{hostname}{path}"


def extract_tools_from_readme() -> List[Dict[str, str]]:
    """Extract all tool entries from README.md with their details."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ ERROR: README.md file not found")
        sys.exit(1)

    tools = []

    # Pattern to match tool entries: **[Tool Name](URL)** ⭐⭐⭐⭐ 💰 🚀
    tool_pattern = r"\*\*\[([^\]]+)\]\(([^)]+)\)\*\* ([⭐💰💵🔄🚀🏢\s]+)"

    # Find all tool matches
    matches = re.finditer(tool_pattern, content)

    for match in matches:
        tool_name = match.group(1).strip()
        tool_url = match.group(2).strip()
        tool_badges = match.group(3).strip()

        # Find the section this tool belongs to
        section_start = content.rfind("\n## ", 0, match.start())
        if section_start != -1:
            section_end = content.find("\n", section_start + 1)
            section = content[section_start + 4 : section_end].strip()
        else:
            section = "Unknown"

        tools.append(
            {
                "name": tool_name,
                "url": tool_url,
                "normalized_url": normalize_url(tool_url),
                "badges": tool_badges,
                "section": section,
                "position": match.start(),
            }
        )

    return tools


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_duplicates(tools: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Find all types of duplicates in the tools list."""
    duplicates = {
        "exact_names": [],
        "exact_urls": [],
        "similar_names": [],
        "same_domain": [],
    }

    # Track seen items
    seen_names = {}
    seen_urls = {}
    seen_domains = {}

    for i, tool in enumerate(tools):
        tool_name_lower = tool["name"].lower()
        normalized_url = tool["normalized_url"]

        # Extract domain from normalized URL
        try:
            domain = urllib.parse.urlparse(tool["url"]).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
        except:
            domain = tool["url"]

        # Check for exact name duplicates
        if tool_name_lower in seen_names:
            duplicates["exact_names"].append(
                {
                    "tools": [seen_names[tool_name_lower], tool],
                    "type": "Exact name match",
                }
            )
        else:
            seen_names[tool_name_lower] = tool

        # Check for exact URL duplicates
        if normalized_url in seen_urls:
            duplicates["exact_urls"].append(
                {"tools": [seen_urls[normalized_url], tool], "type": "Exact URL match"}
            )
        else:
            seen_urls[normalized_url] = tool

        # Check for similar names (fuzzy matching)
        for existing_name, existing_tool in seen_names.items():
            if existing_name != tool_name_lower:
                sim = similarity(tool_name_lower, existing_name)
                if sim > 0.85:  # 85% similarity threshold
                    duplicates["similar_names"].append(
                        {
                            "tools": [existing_tool, tool],
                            "type": f"Similar names (similarity: {sim:.2f})",
                            "similarity": sim,
                        }
                    )

        # Check for same domain (potential duplicates)
        if domain in seen_domains and domain not in ["github.com", "github.io"]:
            # Skip common hosting platforms
            duplicates["same_domain"].append(
                {
                    "tools": [seen_domains[domain], tool],
                    "type": f"Same domain ({domain})",
                }
            )
        else:
            seen_domains[domain] = tool

    return duplicates


def print_duplicate_report(duplicates: Dict[str, List[Dict[str, str]]]) -> int:
    """Print a detailed report of found duplicates."""
    total_issues = 0

    print("🔍 DUPLICATE TOOL CHECKER REPORT")
    print("=" * 50)

    for category, issues in duplicates.items():
        if not issues:
            continue

        category_name = {
            "exact_names": "EXACT NAME DUPLICATES",
            "exact_urls": "EXACT URL DUPLICATES",
            "similar_names": "SIMILAR NAME DUPLICATES",
            "same_domain": "SAME DOMAIN TOOLS",
        }.get(category, category.upper())

        print(f"\n❌ {category_name} ({len(issues)} found)")
        print("-" * 40)

        for i, issue in enumerate(issues, 1):
            tools = issue["tools"]
            issue_type = issue["type"]

            print(f"\n{i}. {issue_type}")
            for j, tool in enumerate(tools):
                print(f"   {chr(97+j)}. [{tool['name']}]({tool['url']})")
                print(f"      Section: {tool['section']}")

            # For similar names, show similarity score
            if "similarity" in issue:
                print(f"      Similarity: {issue['similarity']:.1%}")

        total_issues += len(issues)

    # Summary
    print(f"\n{'='*50}")
    if total_issues == 0:
        print("✅ NO DUPLICATES FOUND - All tools are unique!")
        return 0
    else:
        print(f"❌ TOTAL ISSUES FOUND: {total_issues}")
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("   • Review exact duplicates and remove redundant entries")
        print("   • Check similar names - they might be the same tool")
        print("   • Verify same-domain tools aren't duplicates")
        print("   • Update URLs if tools have moved or been renamed")
        return 1


def main():
    """Main function to run duplicate detection."""
    print("🚀 Starting duplicate tool detection...\n")

    # Extract tools from README
    tools = extract_tools_from_readme()
    print(f"📊 Found {len(tools)} tools in README.md")

    if len(tools) == 0:
        print("⚠️  No tools found in README.md - check the format")
        sys.exit(1)

    # Find duplicates
    duplicates = find_duplicates(tools)

    # Print report and exit with appropriate code
    exit_code = print_duplicate_report(duplicates)

    if exit_code == 0:
        print(f"\n🎉 SUCCESS: All {len(tools)} tools are unique!")
    else:
        print(f"\n💡 TIP: Use the contributing guidelines to avoid duplicates")
        print("   See: CONTRIBUTING.md#quality-standards")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
