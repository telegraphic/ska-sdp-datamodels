"""
Tests for functions in vis_io_ms.py
The tests here assume that msv2 module works.
For tests specific to the module, refer to test_msv2.py
"""

import os
import shutil
import tempfile

import pytest

from ska_sdp_datamodels.visibility.vis_io_ms import (
    create_visibility_from_ms,
    export_visibility_to_ms,
    extend_visibility_to_ms,
    list_ms,
)
from tests.visibility.test_msv2 import write_tables_WGS84

casacore = pytest.importorskip("casacore")


def test_create_visibility_from_ms():
    """
    Test for create_visibility_from_ms function, basic level
    """
    # Set up
    test_path = tempfile.mkdtemp(prefix="test-ms-", suffix=".tmp")
    ms_file = os.path.join(test_path, "test.ms")

    # Generate a temp MS file
    if not os.path.exists(ms_file):
        write_tables_WGS84(ms_file)

    vis = create_visibility_from_ms(ms_file)

    for value in vis:
        assert value.vis.data.shape[-1] == 1
        assert value.visibility_acc.polarisation_frame.type == "stokesI"


def test_extend_visibility_to_ms():
    """
    Test for extend_visibility_to_ms function
    """
    # Set up
    test_path = tempfile.mkdtemp(prefix="test-ms-", suffix=".tmp")
    ms_file = os.path.join(test_path, "test.ms")
    out_ms_file = os.path.join(test_path, "out.ms")

    # Generate a temp MS file
    if not os.path.exists(ms_file):
        write_tables_WGS84(ms_file)

    # remove temp file if exists
    if os.path.exists(out_ms_file):
        shutil.rmtree(out_ms_file, ignore_errors=False)

    # Create vis and extend
    bvis = create_visibility_from_ms(ms_file)[0]
    bvis_list = [bv[1] for bv in bvis.groupby("time", squeeze=False)]
    for bvis in bvis_list:
        extend_visibility_to_ms(out_ms_file, bvis)

    vis = create_visibility_from_ms(out_ms_file)

    for value in vis:
        assert value.vis.data.shape[-1] == 1
        assert value.visibility_acc.polarisation_frame.type == "stokesI"


def test_list_ms():
    """
    Test for list_ms
    """
    # Set up
    test_path = tempfile.mkdtemp(prefix="test-ms-", suffix=".tmp")
    ms_file = os.path.join(test_path, "test.ms")

    # Generate a temp MS file
    if not os.path.exists(ms_file):
        write_tables_WGS84(ms_file)

    ms_list = list_ms(ms_file)

    assert len(ms_list) > 0


def test_export_visibility_to_ms():
    """
    Test for export_visibility_to_ms function
    """
    # Set up
    test_path = tempfile.mkdtemp(prefix="test-ms-", suffix=".tmp")
    ms_file = os.path.join(test_path, "test.ms")

    # Generate a temp MS file
    if not os.path.exists(ms_file):
        write_tables_WGS84(ms_file)

    # Generate visibility list from MS file
    bvis = create_visibility_from_ms(ms_file)[0]
    bvis_list = [bv[1] for bv in bvis.groupby("time", squeeze=False)]

    # Generate MS file from these visibilities
    export_visibility_to_ms(ms_file, bvis_list, source_name=None)
