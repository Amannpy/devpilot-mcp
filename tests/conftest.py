"""Pytest configuration and fixtures"""

import pytest
from src.server import DeveloperWorkflowServer


@pytest.fixture
def server():
    """Create server instance for testing"""
    return DeveloperWorkflowServer()


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing"""
    return """

def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total

class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add(self, item):
        self.items.append(item)
"""


@pytest.fixture
def buggy_code():
    """Sample code with bugs"""
    return """
import sqlite3

def get_user(username):
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    query = "SELECT id FROM users WHERE username = ' " + username + "'"
    cursor.execute(query)
    return cursor.fetchone()

password = "hardcoded123"
"""
