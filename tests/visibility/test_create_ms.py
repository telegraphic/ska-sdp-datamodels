# pylint: disable=invalid-name, too-many-locals
"""
Unit tests for create and extend Visibility
Note we need to refactor these tests into test_vis_io_ms.py
"""
import os
import shutil
import tempfile
import unittest

import numpy

from ska_sdp_datamodels.visibility.vis_io_ms import (
    create_visibility_from_ms,
    extend_visibility_to_ms,
)
from tests.visibility.test_msv2 import write_tables_WGS84


class TestCreateMS(unittest.TestCase):
    """
    Test Setup
    """

    def setUp(self):
        self.casacore_available = True

        numpy.seterr(all="ignore")
        self.testPath = tempfile.mkdtemp(prefix="test-ms-", suffix=".tmp")
        self.ms_file = os.path.join(self.testPath, "test.ms")
        self.out_ms_file = os.path.join(self.testPath, "out.ms")
        # Generate a temp MS file
        if not os.path.exists(self.ms_file):
            write_tables_WGS84(self.ms_file)

        self.vis = None

    def test_create_list(self):
        """
        Test for basic create_visibility function,
            from a list of Visibilities.
        """
        if not self.casacore_available:
            return

        msfile = self.ms_file
        self.vis = create_visibility_from_ms(msfile)

        for v in self.vis:
            assert v.vis.data.shape[-1] == 1
            assert v.visibility_acc.polarisation_frame.type == "stokesI"

    def test_create_list_spectral(self):
        """
        Test with spectral information.
        """
        if not self.casacore_available:
            return

        msfile = self.ms_file

        vis_by_channel = []
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(msfile, range(schan, max_chan))
            vis_by_channel.append(v[0])

        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 1
            assert v.visibility_acc.polarisation_frame.type == "stokesI"

    def test_create_list_slice(self):
        """
        Test using slicing
        """
        if not self.casacore_available:
            return

        msfile = self.ms_file

        vis_by_channel = []
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(
                msfile, start_chan=schan, end_chan=max_chan - 1
            )
            assert v[0].vis.shape[-2] == nchan_ave
            vis_by_channel.append(v[0])

        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 1
            assert v.visibility_acc.polarisation_frame.type == "stokesI"

    def test_create_list_slice_visibility(self):
        """
        Test using slicing with multiple channels
        """
        if not self.casacore_available:
            return

        msfile = self.ms_file

        vis_by_channel = []
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(
                msfile, start_chan=schan, end_chan=max_chan - 1
            )
            nchannels = len(numpy.unique(v[0].frequency))
            assert nchannels == nchan_ave
            vis_by_channel.append(v[0])

        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 1
            assert v.visibility_acc.polarisation_frame.type == "stokesI"
            assert numpy.max(numpy.abs(v.vis)) > 0.0
            assert numpy.max(numpy.abs(v.visibility_acc.flagged_vis)) > 0.0
            assert numpy.sum(v.weight) > 0.0
            assert numpy.sum(v.visibility_acc.flagged_weight) > 0.0

    def test_create_list_average_slice_visibility(self):
        """
        Test using slicing averaging over channels
        """

        if not self.casacore_available:
            return

        msfile = self.ms_file

        vis_by_channel = []
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(
                msfile,
                start_chan=schan,
                end_chan=max_chan - 1,
                average_channels=True,
            )
            nchannels = len(numpy.unique(v[0].frequency))
            assert nchannels == 1
            vis_by_channel.append(v[0])

        assert len(vis_by_channel) == 12
        for ivis, v in enumerate(vis_by_channel):
            assert v.vis.data.shape[-1] == 1
            assert v.visibility_acc.polarisation_frame.type == "stokesI"
            assert numpy.max(numpy.abs(v.vis)) > 0.0, ivis
            assert (
                numpy.max(numpy.abs(v.visibility_acc.flagged_vis)) > 0.0
            ), ivis
            assert numpy.sum(v.weight) > 0.0, ivis
            assert numpy.sum(v.visibility_acc.flagged_weight) > 0.0, ivis

    def test_create_list_single(self):
        """
        Test for a single Visibility
        """
        if not self.casacore_available:
            return

        msfile = self.ms_file

        vis_by_channel = []
        nchan_ave = 1
        nchan = 8
        for schan in range(0, nchan, nchan_ave):
            v = create_visibility_from_ms(
                msfile, start_chan=schan, end_chan=schan
            )
            vis_by_channel.append(v[0])

        assert len(vis_by_channel) == 8, len(vis_by_channel)
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 1
            assert v.visibility_acc.polarisation_frame.type == "stokesI"

    def test_create_list_spectral_average(self):
        """
        Test when averaging over spectral channels
        """
        if not self.casacore_available:
            return

        msfile = self.ms_file

        vis_by_channel = []
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(
                msfile, range(schan, max_chan), average_channels=True
            )
            vis_by_channel.append(v[0])

        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 1
            assert v.vis.data.shape[-2] == 1
            assert v.visibility_acc.polarisation_frame.type == "stokesI"
            assert numpy.max(numpy.abs(v.vis)) > 0.0
            assert numpy.max(numpy.abs(v.visibility_acc.flagged_vis)) > 0.0

    def test_extend_ms(self):
        """
        Test for extend_visibility_to_ms function
        """
        # Reading
        msfile = self.ms_file
        msoutfile = self.out_ms_file
        # remove temp file if exists
        if os.path.exists(msoutfile):
            shutil.rmtree(msoutfile, ignore_errors=False)
        # open an existent file
        bvis = create_visibility_from_ms(msfile)[0]
        bvis_list = [bv[1] for bv in bvis.groupby("time", squeeze=False)]
        for bvis in bvis_list:
            extend_visibility_to_ms(msoutfile, bvis)

    def tearDown(self):
        """Remove the test path directory and its contents"""

        shutil.rmtree(self.testPath, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
