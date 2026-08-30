"""Smoke tests for the Python package."""

import predictive_maintenance


def test_package_is_importable() -> None:
    """The editable installation exposes the project package."""
    assert predictive_maintenance.__version__ == "1.0.1"
