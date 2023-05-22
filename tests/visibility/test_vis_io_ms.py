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

    return ms_file


def test_create_visibility_from_ms(msfile):
    """
    Test for create_visibility_from_ms function, basic level
    """

    vis = create_visibility_from_ms(msfile)

    for value in vis:
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
    bvis = create_visibility_from_ms(msfile)[0]
    bvis_list = [bv[1] for bv in bvis.groupby("time", squeeze=False)]
    for bvis in bvis_list:
        extend_visibility_to_ms(outmsfile, bvis)

    vis = create_visibility_from_ms(outmsfile)

    for value in vis:
        assert value.vis.data.shape[-1] == 1
        assert value.visibility_acc.polarisation_frame.type == "stokesI"


def test_list_ms(msfile):
    """
    Test for list_ms
    """
    ms_list = list_ms(msfile)

    assert len(ms_list) > 0


def test_export_visibility_to_ms(msfile):
    """
    Test for export_visibility_to_ms function
    """
    outmsfile = msfile.replace("test.ms", "out.ms")
    # remove temp file if exists
    if os.path.exists(outmsfile):
        shutil.rmtree(outmsfile, ignore_errors=False)

    # Generate visibility list from MS file
    bvis = create_visibility_from_ms(msfile)[0]
    bvis_list = [bv[1] for bv in bvis.groupby("time", squeeze=False)]

    # Generate MS file from these visibilities
    export_visibility_to_ms(outmsfile, bvis_list, source_name=None)

    vis = create_visibility_from_ms(outmsfile)

    for value in vis:
        assert value.vis.data.shape[-1] == 1
        assert value.visibility_acc.polarisation_frame.type == "stokesI"


def test_create_ms_spectral(msfile):
    """
    Specifc scenario of create_visibility_from_ms:
    Test with spectral information.
    """
    vis_by_channel = []
    for schan in range(0, NCHAN, NCHAN_AVE):
        max_chan = min(NCHAN, schan + NCHAN_AVE)
        vis = create_visibility_from_ms(msfile, range(schan, max_chan))
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 12
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"


def test_create_ms_slice(msfile):
    """
    Specifc scenario of create_visibility_from_ms
    Test using slicing
    """
    vis_by_channel = []

    for schan in range(0, NCHAN, NCHAN_AVE):
        max_chan = min(NCHAN, schan + NCHAN_AVE)
        vis = create_visibility_from_ms(
            msfile, start_chan=schan, end_chan=max_chan - 1
        )
        assert vis[0].vis.shape[-2] == NCHAN_AVE
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 12
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"


def test_create_ms_slice_visibility(msfile):
    """
    Specifc scenario of create_visibility_from_ms:
    Test using slicing with multiple channels
    """

    vis_by_channel = []
    for schan in range(0, NCHAN, NCHAN_AVE):
        max_chan = min(NCHAN, schan + NCHAN_AVE)
        vis = create_visibility_from_ms(
            msfile, start_chan=schan, end_chan=max_chan - 1
        )
        nchannels = len(numpy.unique(vis[0].frequency))
        assert nchannels == NCHAN_AVE
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 12
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"
        assert numpy.max(numpy.abs(vis.vis)) > 0.0
        assert numpy.max(numpy.abs(vis.visibility_acc.flagged_vis)) > 0.0
        assert numpy.sum(vis.weight) > 0.0
        assert numpy.sum(vis.visibility_acc.flagged_weight) > 0.0


def test_create_ms_average_slice_visibility(msfile):
    """
    Specifc scenario of create_visibility_from_ms:
    Test using slicing averaging over channels
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
    for ivis, vis in enumerate(vis_by_channel):
        assert vis.vis.data.shape[-1] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"
        assert numpy.max(numpy.abs(vis.vis)) > 0.0, ivis
        assert numpy.max(numpy.abs(vis.visibility_acc.flagged_vis)) > 0.0, ivis
        assert numpy.sum(vis.weight) > 0.0, ivis
        assert numpy.sum(vis.visibility_acc.flagged_weight) > 0.0, ivis


def test_create_ms_spectral_average(msfile):
    """
    Specifc scenario of create_visibility_from_ms:
    Test when averaging over spectral channels
    """

    vis_by_channel = []

    for schan in range(0, NCHAN, NCHAN_AVE):
        max_chan = min(NCHAN, schan + NCHAN_AVE)
        vis = create_visibility_from_ms(
            msfile, range(schan, max_chan), average_channels=True
        )
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 12
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.vis.data.shape[-2] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"
        assert numpy.max(numpy.abs(vis.vis)) > 0.0
        assert numpy.max(numpy.abs(vis.visibility_acc.flagged_vis)) > 0.0


def test_create_ms_single(msfile):
    """
    Specifc scenario of create_visibility_from_ms:
    Test for a single Visibility
    """

    vis_by_channel = []
    nchan_ave = 1
    nchan = 8
    for schan in range(0, nchan, nchan_ave):
        vis = create_visibility_from_ms(
            msfile, start_chan=schan, end_chan=schan
        )
        vis_by_channel.append(vis[0])

    assert len(vis_by_channel) == 8, len(vis_by_channel)
    for vis in vis_by_channel:
        assert vis.vis.data.shape[-1] == 1
        assert vis.visibility_acc.polarisation_frame.type == "stokesI"
