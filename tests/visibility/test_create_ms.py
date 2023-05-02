# pylint: disable=invalid-name, too-many-locals, duplicate-code
"""
Unit tests for create Visibility
Note we need to refactor these tests to have common setups.
"""
import os
import shutil
import tempfile
import time
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility import msv2
from ska_sdp_datamodels.visibility.msv2fund import Antenna, Stand
from ska_sdp_datamodels.visibility.vis_io_ms import create_visibility_from_ms


class TestCreateMS(unittest.TestCase):
    """
    Test Setup
    """

    def setUp(self):

        self.casacore_available = True

        numpy.seterr(all="ignore")
        self.testPath = tempfile.mkdtemp(prefix="test-ms-", suffix=".tmp")
        self.ms_file = os.path.join(self.testPath, "test.ms")
        # Generate a temp MS file
        if not os.path.exists(self.ms_file):
            self.__write_tables_WGS84()

        self.vis = None

    def __initData_WGS84(self):
        """
        Private function to generate
        a random set of data for writing a UVFITS
        file.  The data is returned as a dictionary with keys:
         * freq - frequency array in Hz
         * site - Observatory object
         * stands - array of stand numbers
         * bl - list of baseline pairs in real stand numbers
         * vis - array of visibility data in baseline x freq format
        """

        # Frequency range
        freq = numpy.arange(0, 192) * 20e6 / 192 + 40e6
        channel_width = numpy.full_like(freq, 20e6 / 192.0)

        # Site and stands
        obs = EarthLocation(
            lon=116.76444824 * u.deg, lat=-26.824722084 * u.deg, height=300.0
        )

        mount = numpy.empty(10)
        mount.fill("equat")

        names = numpy.array(
            [
                "ak02",
                "ak04",
                "ak05",
                "ak12",
                "ak13",
                "ak14",
                "ak16",
                "ak24",
                "ak28",
                "ak30",
            ]
        )
        diameter = numpy.array(
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]
        )
        xyz = numpy.array(
            [
                [-2556109.98244348, 5097388.70050131, -2848440.1332423],
                [-2556087.396082, 5097423.589662, -2848396.867933],
                [-2556028.60254059, 5097451.46195695, -2848399.83113161],
                [-2556496.23893101, 5097333.71466669, -2848187.33832738],
                [-2556407.35299627, 5097064.98390756, -2848756.02069474],
                [-2555972.78456557, 5097233.65481756, -2848839.88915184],
                [-2555592.88867802, 5097835.02121109, -2848098.26409648],
                [-2555959.34313275, 5096979.52802882, -2849303.57702486],
                [-2556552.97431815, 5097767.23612874, -2847354.29540396],
                [-2557348.40370367, 5097170.17682775, -2847716.21368966],
            ]
        )

        site_config = Configuration.constructor(
            name="ASKAP",
            location=obs,
            names=names,
            xyz=xyz,
            mount=mount,
            frame="WGS84",
            receptor_frame=ReceptorFrame("linear"),
            diameter=diameter,
        )
        antennas = []
        for i, name in enumerate(names):
            antennas.append(
                Antenna(i, Stand(name, xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            )

        # Set baselines and data
        blList = []
        N = len(antennas)

        antennas2 = antennas

        for i in range(0, N - 1):
            for j in range(i + 1, N):
                blList.append((antennas[i], antennas2[j]))

        visData = numpy.random.rand(len(blList), len(freq))
        visData = visData.astype(numpy.complex64)

        weights = numpy.random.rand(len(blList), len(freq))
        return {
            "freq": freq,
            "channel_width": channel_width,
            "site": site_config,
            "antennas": antennas,
            "bl": blList,
            "vis": visData,
            "weights": weights,
            "xyz": xyz,
            "obs": obs,
        }

    def __write_tables_WGS84(self):
        """Test if the MeasurementSet writer writes all of the tables."""

        testTime = float(86400.0 * Time(time.time(), format="unix").mjd)
        testFile = self.ms_file

        # Get some data
        data = self.__initData_WGS84()

        # Start the table
        tbl = msv2.Ms(
            testFile, ref_time=testTime, frame=data["site"].attrs["frame"]
        )
        tbl.set_stokes(["xx"])
        tbl.set_frequency(data["freq"], data["channel_width"])
        tbl.set_geometry(data["site"], data["antennas"])
        tbl.add_data_set(testTime, 2.0, data["bl"], data["vis"])

        # Judge if the tbl's antenna is correctly positioned
        antxyz_ecef = numpy.zeros((len(data["antennas"]), 3))
        for i, ant in enumerate(tbl.array[0]["ants"]):
            antxyz_ecef[i][0] = ant.x
            antxyz_ecef[i][1] = ant.y
            antxyz_ecef[i][2] = ant.z

        # the antxyz in table should be same as data['xyz']
        assert numpy.allclose(antxyz_ecef, data["xyz"])

        tbl.write()

        # Make sure everyone is there
        self.assertTrue(os.path.exists(testFile))
        for tbl in (
            "ANTENNA",
            "DATA_DESCRIPTION",
            "FEED",
            "FIELD",
            "FLAG_CMD",
            "HISTORY",
            "OBSERVATION",
            "POINTING",
            "POLARIZATION",
            "PROCESSOR",
            "SOURCE",
            "SPECTRAL_WINDOW",
            "STATE",
        ):
            self.assertTrue(os.path.exists(os.path.join(testFile, tbl)))

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

    def tearDown(self):
        """Remove the test path directory and its contents"""

        shutil.rmtree(self.testPath, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
