"""
Service layer handling Pull Request (PR) analysis, suggestions, and documentation generation.
"""

import re


def analyze_pr(title: str, description: str, diff: str | None = None) -> dict:
    """
    Perform lightweight PR analysis.
    Later, this will connect with the model for intelligent code/documentation insights.
    """
    suggestions = []

    # Simple heuristics for MVP
    if "fix" in title.lower():
        suggestions.append("Ensure this fix includes regression tests.")
    if "refactor" in title.lower():
        suggestions.append("Verify that refactoring does not change public API behavior.")
    if not description or len(description.split()) < 10:
        suggestions.append("PR description seems short; consider adding more details.")

    # Example: detect large diffs
    if diff and diff.count("\n") > 50:
        suggestions.append("This seems like a large PR; consider splitting into smaller parts.")

    # Example: look for TODOs or FIXMEs in diff
    if diff and re.search(r"(TODO|FIXME)", diff):
        suggestions.append("Found TODO/FIXME comments in diff; consider resolving before merging.")

    return {
        "summary": f"PR '{title}' analyzed successfully.",
        "suggestions": suggestions or ["Looks good overall!"],
        "length_of_diff": len(diff.splitlines()) if diff else 0,
    }
