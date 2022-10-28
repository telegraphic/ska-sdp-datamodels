"""
Unit tests for Visibility object
"""


def test_visibility_copy(visibility):
    """
    Test deep-copying Visibility
    """
    new_vis = visibility.copy(deep=True)
    visibility["vis"].data[...] = 0.0
    new_vis["vis"].data[...] = 1.0
    assert new_vis["vis"].data[0, 0].real.all() == 1.0
    assert visibility["vis"].data[0, 0].real.all() == 0.0
