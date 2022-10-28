"""
Pytest fixtures
"""

import pytest


@pytest.fixture(scope="package", name="image")
def image_fixture():
    """
    Image fixture
    """
