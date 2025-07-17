#!/usr/bin/env python3
"""
Validate the format and structure of tools in README.md

Checks for:
- Proper tool entry format
- Required fields
- Valid star ratings
- Appropriate badges
- Correct markdown structure
"""

import re
import sys
from typing import List, Dict


def extract_tool_sections() -> List[Dict]:
    """Extract all tool sections and their tools."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ ERROR: README.md file not found")
        sys.exit(1)

    sections = []

    # Split content by main headers (##)
    section_pattern = r"\n## ([^#\n]+).*?(?=\n## |\Z)"
    section_matches = re.finditer(section_pattern, content, re.DOTALL)

    for match in section_matches:
        section_title = match.group(1).strip()
        section_content = match.group(0)

        # Skip non-tool sections
        skip_sections = [
            "Table of Contents",
            "Legend",
            "Cost Comparison",
            "Recommended Stacks",
            "Learning Resources",
            "Contributing",
            "Repository Growth",
            "License",
        ]

        if any(skip in section_title for skip in skip_sections):
            continue

        # Extract tools from this section
        tools = extract_tools_from_section(section_content, section_title)

        if tools:  # Only add sections that have tools
            sections.append(
                {
                    "title": section_title,
                    "content": section_content,
                    "tools": tools,
                    "tool_count": len(tools),
                }
            )

    return sections


def extract_tools_from_section(content: str, section_title: str) -> List[Dict]:
    """Extract tools from a specific section."""
    tools = []

    # Pattern for tool entries
    tool_pattern = r"\*\*\[([^\]]+)\]\(([^)]+)\)\*\* ([⭐💰💵🔄🚀🏢\s]+)\n((?:- \*\*[^*]+\*\*:.*\n?)*)"

    matches = re.finditer(tool_pattern, content)

    for match in matches:
        name = match.group(1).strip()
        url = match.group(2).strip()
        badges = match.group(3).strip()
        fields_content = match.group(4).strip()

        # Parse fields
        fields = {}
        field_pattern = r"- \*\*([^*]+)\*\*:\s*(.+)"
        field_matches = re.findall(field_pattern, fields_content)

        for field_name, field_value in field_matches:
            fields[field_name.strip()] = field_value.strip()

        tools.append(
            {
                "name": name,
                "url": url,
                "badges": badges,
                "fields": fields,
                "section": section_title,
                "raw_content": match.group(0),
            }
        )

    return tools


def validate_tool_format(tool: Dict) -> List[str]:
    """Validate a single tool's format and return list of issues."""
    issues = []

    # Check required fields
    required_fields = ["What it does", "Best for", "Features"]
    for field in required_fields:
        if field not in tool["fields"]:
            issues.append(f"Missing required field: '{field}'")
        elif not tool["fields"][field].strip():
            issues.append(f"Empty required field: '{field}'")

    # Validate star rating
    star_pattern = r"⭐+"
    star_matches = re.findall(star_pattern, tool["badges"])

    if not star_matches:
        issues.append("Missing star rating")
    elif len(star_matches) > 1:
        issues.append("Multiple star ratings found")
    else:
        star_count = len(star_matches[0])
        if star_count < 2 or star_count > 5:
            issues.append(f"Invalid star count: {star_count} (should be 2-5)")

    # Check for valid badges
    valid_badges = ["💰", "💵", "🔄", "🚀", "🏢"]
    badges_text = tool["badges"]

    # Remove stars for badge validation
    badges_no_stars = re.sub(r"⭐+", "", badges_text).strip()

    if badges_no_stars:
        for char in badges_no_stars:
            if char not in valid_badges + [" "]:
                issues.append(f"Invalid badge: '{char}'")

    # Validate URL format
    url = tool["url"]
    if not url.startswith(("http://", "https://")):
        issues.append(f"Invalid URL format: {url}")

    # Check field content quality
    for field_name, field_value in tool["fields"].items():
        if len(field_value) < 10:
            issues.append(f"Field '{field_name}' too short (minimum 10 characters)")
        elif len(field_value) > 200:
            issues.append(f"Field '{field_name}' too long (maximum 200 characters)")

    return issues


def validate_section_structure(section: Dict) -> List[str]:
    """Validate section structure and organization."""
    issues = []

    tools = section["tools"]

    if len(tools) == 0:
        issues.append(f"Section '{section['title']}' has no tools")
        return issues

    # Check if tools are roughly ordered by star rating
    star_counts = []
    for tool in tools:
        star_matches = re.findall(r"⭐+", tool["badges"])
        if star_matches:
            star_counts.append(len(star_matches[0]))
        else:
            star_counts.append(0)

    # Check if generally decreasing (allowing some variation)
    if len(star_counts) > 3:
        # Compare first quarter with last quarter
        first_quarter = star_counts[: len(star_counts) // 4] or star_counts[:1]
        last_quarter = star_counts[-len(star_counts) // 4 :] or star_counts[-1:]

        avg_first = sum(first_quarter) / len(first_quarter)
        avg_last = sum(last_quarter) / len(last_quarter)

        if avg_last > avg_first + 0.5:
            issues.append(
                f"Tools appear to be in wrong order (higher-rated tools should come first)"
            )

    return issues


def print_validation_report(sections: List[Dict], all_issues: Dict) -> int:
    """Print detailed validation report."""
    total_tools = sum(section["tool_count"] for section in sections)
    total_issues = sum(len(issues) for issues in all_issues.values())

    print("🔍 TOOL FORMAT VALIDATION REPORT")
    print("=" * 50)
    print(f"📊 Total sections analyzed: {len(sections)}")
    print(f"📊 Total tools validated: {total_tools}")

    if total_issues == 0:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("🎉 No format issues found - excellent work!")
        return 0

    print(f"\n❌ TOTAL ISSUES FOUND: {total_issues}")
    print("-" * 50)

    # Group issues by severity
    critical_issues = 0
    warning_issues = 0

    for tool_name, issues in all_issues.items():
        if not issues:
            continue

        print(f"\n🔧 {tool_name}")
        for issue in issues:
            if any(
                keyword in issue.lower() for keyword in ["missing", "invalid", "empty"]
            ):
                print(f"   ❌ {issue}")
                critical_issues += 1
            else:
                print(f"   ⚠️  {issue}")
                warning_issues += 1

    # Summary and recommendations
    print(f"\n{'='*50}")
    print("📋 SUMMARY:")
    print(f"   ❌ Critical issues: {critical_issues}")
    print(f"   ⚠️  Warnings: {warning_issues}")

    print(f"\n🔧 RECOMMENDED ACTIONS:")
    print("   • Fix missing required fields immediately")
    print("   • Correct invalid star ratings (2-5 stars only)")
    print("   • Remove invalid badges")
    print("   • Improve short field descriptions")
    print("   • Verify all URLs are accessible")

    print(f"\n💡 HELP:")
    print("   • See CONTRIBUTING.md for format guidelines")
    print("   • Check existing tools for format examples")
    print("   • Use the tool submission template")

    return 1 if critical_issues > 0 else 0


def main():
    """Main validation function."""
    print("🚀 Starting tool format validation...\n")

    # Extract all sections and tools
    sections = extract_tool_sections()

    if not sections:
        print("⚠️  No tool sections found in README.md")
        sys.exit(1)

    # Validate each tool and section
    all_issues = {}
    section_issues = []

    for section in sections:
        # Validate section structure
        sect_issues = validate_section_structure(section)
        if sect_issues:
            section_issues.extend(
                [f"Section '{section['title']}': {issue}" for issue in sect_issues]
            )

        # Validate each tool in the section
        for tool in section["tools"]:
            tool_issues = validate_tool_format(tool)
            if tool_issues:
                all_issues[f"[{section['title']}] {tool['name']}"] = tool_issues

    # Add section issues to the report
    if section_issues:
        all_issues["📂 Section Structure Issues"] = section_issues

    # Print report and return exit code
    exit_code = print_validation_report(sections, all_issues)

    if exit_code == 0:
        total_tools = sum(section["tool_count"] for section in sections)
        print(f"\n🎉 SUCCESS: All {total_tools} tools are properly formatted!")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
