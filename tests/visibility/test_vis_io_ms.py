"""
Tests for functions in vis_io_ms.py
The tests here assume that msv2 module works.
For tests specific to the module, refer to test_msv2.py
"""

import os
import shutil
import tempfile

import numpy
import pytest
from casacore.tables import table

from ska_sdp_datamodels.visibility.vis_io_ms import (
    create_visibility_from_ms,
    export_visibility_to_ms,
    extend_visibility_to_ms,
    list_ms,
)
from tests.visibility.test_msv2 import write_tables_WGS84

casacore = pytest.importorskip("casacore")


NCHAN_AVE = 16
NCHAN = 192


@pytest.fixture(scope="module", name="msfile")
def fixture_ms():
    """
    Test setup that includes generating a temporary MS file
    """
    test_path = tempfile.mkdtemp(prefix="test-ms-", suffix=".tmp")
    ms_file = os.path.join(test_path, "test.ms")

    if not os.path.exists(ms_file):
        write_tables_WGS84(ms_file)

    yield ms_file

    shutil.rmtree(ms_file)


def test_create_visibility_from_ms(msfile):
    """
    Test for create_visibility_from_ms function, basic level
    """

    vis = create_visibility_from_ms(msfile)
    # Confirm created visibility is the correct shape and type
    for value in vis:
        assert value.vis.data.shape[0] == 5
        assert value.vis.data.shape[-1] == 1
        assert value.visibility_acc.polarisation_frame.type == "stokesI"


def test_extend_visibility_to_ms(msfile):
    """
    Test for extend_visibility_to_ms function
    """
    outmsfile = msfile.replace("test.ms", "out.ms")
    # remove temp file if exists
    if os.path.exists(outmsfile):
        shutil.rmtree(outmsfile, ignore_errors=False)

    # Create vis and extend
    vis = create_visibility_from_ms(msfile)[0]
    vis_list = [vis_item[1] for vis_item in vis.groupby("time", squeeze=False)]

    for vis in vis_list:
        extend_visibility_to_ms(outmsfile, vis)

    # confirm visibility created from outmsfile is the correct shape and type
    vis_out = create_visibility_from_ms(outmsfile)

    for value in vis_out:
        # assert all times are written
        assert value.vis.data.shape[0] == 5
        assert value.vis.data.shape[-1] == 1
        assert value.visibility_acc.polarisation_frame.type == "stokesI"

    # confirm the outmsfile has been extended
    msfile_table = table(msfile, readonly=True)
    outmsfile_table = table(outmsfile, readonly=True)
    msfile_data_col = msfile_table.getcol("DATA")
    outmsfile_data_col = outmsfile_table.getcol("DATA")

    assert msfile_data_col.shape != outmsfile_data_col.shape


def test_list_ms(msfile):
    """
    Test for list_ms
    """
    ms_list = list_ms(msfile)

    # check that it returns a tuple
    assert len(ms_list) == 2
    assert isinstance(ms_list, tuple)

    # check that the first element is a list of strings
    assert isinstance(ms_list[0], list)
    assert all(isinstance(source, str) for source in ms_list[0])

    # check that the second element is a list of ints
    assert isinstance(ms_list[1], list)
    assert all(isinstance(dds, int) for dds in ms_list[1])


def test_export_visibility_to_ms(msfile):
    """
    Test for export_visibility_to_ms function
    """
    outmsfile = msfile.replace("test.ms", "out.ms")
    # remove temp file if exists
    if os.path.exists(outmsfile):
        shutil.rmtree(outmsfile, ignore_errors=False)

    # This splits vis into 5 different Visibilities
    vis = create_visibility_from_ms(msfile)[0]
    vis_list = [vis_item[1] for vis_item in vis.groupby("time", squeeze=False)]
    assert len(vis_list) == 5

    # Generate MS file from these visibilities
    export_visibility_to_ms(outmsfile, vis_list)

    vis = create_visibility_from_ms(outmsfile)

    for value in vis:
        assert value.vis.data.shape[0] == 5
        assert value.vis.data.shape[-1] == 1
        assert value.visibility_acc.polarisation_frame.type == "stokesI"


def test_create_ms_slice_visibility(msfile):
    """
    Test the specifc scenario of create_visibility_from_ms where the
    visibility is sliced over groups of channels and read by slice
    """

    vis_by_channel = []
    for schan in range(0, NCHAN, NCHAN_AVE):
        max_chan = min(NCHAN, schan + NCHAN_AVE)
        vis = create_visibility_from_ms(
            msfile, start_chan=schan, end_chan=max_chan - 1
        )
        assert vis[0].vis.shape[-2] == NCHAN_AVE
        nchannels = len(numpy.unique(vis[0].frequency))
        assert nchannels == NCHAN_AVE
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 12
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"
        # check the relevant quantities are not zero
        assert numpy.max(numpy.abs(vis.vis)) > 0.0
        assert numpy.max(numpy.abs(vis.visibility_acc.flagged_vis)) > 0.0
        assert numpy.sum(vis.weight) > 0.0
        assert numpy.sum(vis.visibility_acc.flagged_weight) > 0.0


def test_create_ms_single(msfile):
    """
    Specifc scenario of create_visibility_from_ms:
    Test that creates a single visibility per channel
    """

    vis_by_channel = []
    nchan_ave = 1
    nchan = 8
    for schan in range(0, nchan, nchan_ave):
        vis = create_visibility_from_ms(
            msfile, start_chan=schan, end_chan=schan
        )
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 8
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"
        # check the relevant quantities are not zero
        assert numpy.max(numpy.abs(vis.vis)) > 0.0
        assert numpy.max(numpy.abs(vis.visibility_acc.flagged_vis)) > 0.0
        assert numpy.sum(vis.weight) > 0.0
        assert numpy.sum(vis.visibility_acc.flagged_weight) > 0.0


def test_create_ms_average_slice_visibility(msfile):
    """
    Specifc scenario of create_visibility_from_ms:
    Similar to test_create_ms_slice_visibility but we
    average the frequency channels per slice
    """

    vis_by_channel = []
    for schan in range(0, NCHAN, NCHAN_AVE):
        max_chan = min(NCHAN, schan + NCHAN_AVE)
        vis = create_visibility_from_ms(
            msfile,
            start_chan=schan,
            end_chan=max_chan - 1,
            average_channels=True,
        )
        nchannels = len(numpy.unique(vis[0].frequency))
        assert nchannels == 1
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 12
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.vis.data.shape[-2] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"
        # check the relevant quantities are not zero
        assert numpy.max(numpy.abs(vis.vis)) > 0.0
        assert numpy.max(numpy.abs(vis.visibility_acc.flagged_vis)) > 0.0
        assert numpy.sum(vis.weight) > 0.0
        assert numpy.sum(vis.visibility_acc.flagged_weight) > 0.0
