# tests/test_resources.py
import pytest
from src.resources import get_resources, Resource


def test_get_resources_structure():
    resources = get_resources()

    # Check that it returns a list
    assert isinstance(resources, list)
    assert len(resources) > 0

    # Check each item is a Resource and has expected attributes
    for r in resources:
        assert isinstance(r, Resource)
        assert isinstance(r.uri, str) and r.uri != ""
        assert isinstance(r.name, str) and r.name != ""
        assert isinstance(r.description, str)
        assert isinstance(r.mimeType, str)
        # Default mimeType should be application/json
        assert r.mimeType == "application/json"
