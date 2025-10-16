# resources.py
from dataclasses import dataclass

@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

def get_resources():
    """Return all available resources"""
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
        )
    ]
