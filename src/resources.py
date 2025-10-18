# resources.py
from dataclasses import dataclass
from typing import List


@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mimeType: str = "application/json"


def get_resources() -> List[Resource]:
    """Return all resources available to MCP Server"""
    return [
        Resource(
            uri="git://repository/analysis",
            name="Repository Analysis",
            description="Analyze git repository structure and health",
        ),
        Resource(
            uri="project://metrics/overview",
            name="Project Metrics",
            description="Code quality metrics and statistics",
        ),
    ]
