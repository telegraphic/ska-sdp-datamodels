"""
Tests for functions in vis_io_ms.py
The tests here assume that msv2 module works.
For tests specific to the module, refer to test_msv2.py
"""

import os
import pytest
import tempfile
from ska_sdp_datamodels.visibility.vis_io_ms import (
    create_visibility_from_ms,
    extend_visibility_to_ms,
    export_visibility_to_ms,
)
from tests.utils import data_model_equals

casacore = pytest.importorskip("casacore")


def test_export_visibility_to_ms(visibility):
    """
    Test for export_visibility_to_ms function
    """
    pass


def test_extend_visibility_to_ms(visibility):
    """
    Test for extend_visibility_to_ms function
    """

    pass


def test_create_visibility_from_ms():
    """
    Test for create_visibility_from_ms function, basic level
    """

    pass


def test_list_ms():
    """
    Test for list_ms
    """

    pass
