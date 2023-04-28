# pylint: disable-all
# -*- coding: utf-8 -*-

"""Unit test for the measurementset module."""

import os
import shutil
import tempfile
import time
import unittest

import astropy.units as u
import numpy
from astropy import units
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    ReceptorFrame,
)

try:
    import casacore
    from casacore.tables import table  # pylint: disable=import-error

    from ska_sdp_datamodels.visibility import msv2
    from ska_sdp_datamodels.visibility.msv2fund import Antenna, Stand

    run_ms_tests = True
#            except ModuleNotFoundError:
except:
    run_ms_tests = False


def lla_to_ecef(lat, lon, alt):
    """
    Convert WGS84 spherical coordinates to ECEF cartesian coordinates.

    :param lat: Latitude in radians
    :param lon: Longitude in radians
    :param alt: Altitude in radians
    :result ECEF: Cartesian coordinates (x, y, z)
    """
    WGS84_a = 6378137.00000000
    WGS84_b = 6356752.31424518
    N = WGS84_a**2 / numpy.sqrt(
        WGS84_a**2 * numpy.cos(lat) ** 2 + WGS84_b**2 * numpy.sin(lat) ** 2
    )

    x = (N + alt) * numpy.cos(lat) * numpy.cos(lon)
    y = (N + alt) * numpy.cos(lat) * numpy.sin(lon)
    z = ((WGS84_b**2 / WGS84_a**2) * N + alt) * numpy.sin(lat)

    return x, y, z


def ecef_to_enu(location, xyz):
    """
    Convert ECEF coordinates to ENU coordinates relative to reference location.

    :param location: Current WGS84 coordinate
    :param xyz: ECEF coordinate
    :result: ENU Local xyz coordinate
    """
    # ECEF coordinates of reference point
    lon = location.geodetic[0].to(units.rad).value
    lat = location.geodetic[1].to(units.rad).value
    alt = location.geodetic[2].to(units.m).value

    # pylint: disable=unbalanced-tuple-unpacking
    x, y, z = numpy.hsplit(xyz, 3)

    center_x, center_y, center_z = lla_to_ecef(lat, lon, alt)

    delta_x, delta_y, delta_z = x - center_x, y - center_y, z - center_z
    sin_lat, cos_lat = numpy.sin(lat), numpy.cos(lat)
    sin_lon, cos_lon = numpy.sin(lon), numpy.cos(lon)

    e = -sin_lon * delta_x + cos_lon * delta_y
    n = (
        -sin_lat * cos_lon * delta_x
        - sin_lat * sin_lon * delta_y
        + cos_lat * delta_z
    )
    u = (
        cos_lat * cos_lon * delta_x
        + cos_lat * sin_lon * delta_y
        + sin_lat * delta_z
    )

    return numpy.hstack([e, n, u])


class measurementset_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the lsl.writer.measurementset.Ms
    class."""

    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        numpy.seterr(all="ignore")
        self.testPath = tempfile.mkdtemp(
            prefix="test-measurementset-", suffix=".tmp"
        )

    def __initData_WGS84(self):
        """Private function to generate a random set of data for writing a UVFITS
        file.  The data is returned as a dictionary with keys:
         * freq - frequency array in Hz
         * site - Observatory object
         * stands - array of stand numbers
         * bl - list of baseline pairs in real stand numbers
         * vis - array of visibility data in baseline x freq format
        """
        if run_ms_tests == False:
            return

        # Frequency range
        freq = numpy.arange(0, 512) * 20e6 / 512 + 40e6
        channel_width = numpy.full_like(freq, 20e6 / 512.0)

        # Site and stands
        obs = EarthLocation(
            lon=116.76444824 * u.deg, lat=-26.824722084 * u.deg, height=300.0
        )

        mount = numpy.array(
            [
                "equat",
                "equat",
                "equat",
                "equat",
                "equat",
                "equat",
                "equat",
                "equat",
                "equat",
                "equat",
            ]
        )
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
        for i in range(len(names)):
            antennas.append(
                Antenna(i, Stand(names[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]))
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

    def __initData_ENU(self):
        """Private function to generate a random set of data for writing a Measurements
        file.  The data is returned as a dictionary with keys:
         * freq - frequency array in Hz
         * site - observatory object
         * stands - array of stand numbers
         * bl - list of baseline pairs in real stand numbers
         * vis - array of visibility data in baseline x freq format
        """
        if run_ms_tests == False:
            return

        # Frequency range
        freq = numpy.arange(0, 512) * 20e6 / 512 + 40e6
        channel_width = numpy.full_like(freq, 20e6 / 512.0)

        # Site and stands
        obs = EarthLocation(
            lon=+116.6356824 * u.deg, lat=-26.70130064 * u.deg, height=377.0
        )

        names = numpy.array(["A%02d" % i for i in range(36)])
        mount = numpy.array(["equat" for i in range(36)])
        diameter = numpy.array([12 for i in range(36)])
        xyz = numpy.array(
            [
                [-175.233429, +1673.460938, 0.0000],
                [+261.119019, +796.922119, 0.0000],
                [-29.2005200, +744.432068, 0.0000],
                [-289.355286, +586.936035, 0.0000],
                [-157.031570, +815.570068, 0.0000],
                [-521.311646, +754.674927, 0.0000],
                [-1061.114258, +840.541443, 0.0000],
                [-921.829407, +997.627686, 0.0000],
                [-818.293579, +1142.272095, 0.0000],
                [-531.752808, +850.726257, 0.0000],
                [+81.352448, +790.245117, 0.0000],
                [+131.126358, +1208.831909, 0.0000],
                [-1164.709351, +316.779236, 0.0000],
                [-686.248901, +590.285278, 0.0000],
                [-498.987305, +506.338226, 0.0000],
                [-182.249146, +365.113464, 0.0000],
                [+420.841858, +811.081543, 0.0000],
                [+804.107910, +1273.328369, 0.0000],
                [-462.810394, +236.353790, 0.0000],
                [-449.772339, +15.039755, 0.0000],
                [+13.791821, +110.971809, 0.0000],
                [-425.687317, -181.908752, 0.0000],
                [-333.404053, -503.603394, 0.0000],
                [-1495.472412, +1416.063232, 0.0000],
                [-1038.578857, +1128.367920, 0.0000],
                [-207.151749, +956.312561, 0.0000],
                [-389.051880, +482.405670, 0.0000],
                [-434.000000, +510.000000, 0.0000],
                [-398.000000, +462.000000, 0.0000],
                [-425.000000, +475.000000, 0.0000],
                [-400.000000, +3664.000000, 0.0000],
                [+1796.000000, +1468.000000, 0.0000],
                [+2600.000000, -1532.000000, 0.0000],
                [-400.000000, -2336.000000, 0.0000],
                [-3400.00000, -1532.000000, 0.0000],
                [-2596.000000, +1468.000000, 0.0000],
            ]
        )

        site_config = Configuration.constructor(
            name="ASKAP",
            location=obs,
            names=names,
            xyz=xyz,
            mount=mount,
            frame="ENU",
            receptor_frame=ReceptorFrame("linear"),
            diameter=diameter,
        )
        antennas = []
        for i in range(len(names)):
            antennas.append(
                Antenna(i, Stand(names[i], xyz[i, 0], xyz[i, 1], xyz[i, 2]))
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

    def test_write_tables_ENU(self):
        """Test if the MeasurementSet writer writes all of the tables."""
        if run_ms_tests == False:
            return

        testTime = float(86400.0 * Time(time.time(), format="unix").mjd)
        testFile = os.path.join(self.testPath, "ms-test-W.ms")

        # Get some data
        data = self.__initData_ENU()

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

        # convert back to enu
        antxyz_enu = ecef_to_enu(data["obs"], antxyz_ecef)
        assert numpy.allclose(antxyz_enu, data["xyz"])

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

    def test_write_tables_WGS84(self):
        """Test if the MeasurementSet writer writes all of the tables."""
        if run_ms_tests == False:
            return

        testTime = float(86400.0 * Time(time.time(), format="unix").mjd)
        testFile = os.path.join(self.testPath, "ms-test-WGS.ms")

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

    def test_main_table(self):
        """Test the primary data table."""
        if run_ms_tests == False:
            return

        testTime = float(86400.0 * Time(time.time(), format="unix").mjd)
        testFile = os.path.join(self.testPath, "ms-test-UV.ms")

        # Get some data
        data = self.__initData_WGS84()

        # Start the file
        fits = msv2.Ms(testFile, ref_time=testTime)
        fits.set_stokes(["xx"])
        fits.set_frequency(data["freq"], data["channel_width"])
        fits.set_geometry(data["site"], data["antennas"])
        fits.add_data_set(
            testTime, 6.0, data["bl"], data["vis"], weights=data["weights"]
        )
        fits.write()

        # Open the table and examine
        ms = casacore.tables.table(testFile, ack=False)
        uvw = ms.getcol("UVW")
        ant1 = ms.getcol("ANTENNA1")
        ant2 = ms.getcol("ANTENNA2")
        vis = ms.getcol("DATA")
        weights = ms.getcol("WEIGHT_SPECTRUM")

        ms2 = casacore.tables.table(
            os.path.join(testFile, "ANTENNA"), ack=False
        )
        mapper = ms2.getcol("NAME")

        # Correct number of visibilities
        self.assertEqual(uvw.shape[0], data["vis"].shape[0])
        self.assertEqual(vis.shape[0], data["vis"].shape[0])

        # Correct number of uvw coordinates
        self.assertEqual(uvw.shape[1], 3)

        # Correct number of frequencies
        self.assertEqual(vis.shape[1], data["freq"].size)

        # Correct number of weights
        self.assertEqual(weights.shape[0], data["weights"].shape[0])
        self.assertEqual(weights.shape[1], data["freq"].size)

        # Correct values
        for row in range(uvw.shape[0]):
            stand1 = ant1[row]
            stand2 = ant2[row]
            visData = vis[row, :, 0]
            weightData = weights[row, :, 0]

            # Find out which visibility set in the random data corresponds to the
            # current visibility
            i = 0
            for a1, a2 in data["bl"]:
                if (
                    a1.stand.id == mapper[stand1]
                    and a2.stand.id == mapper[stand2]
                ):
                    break
                else:
                    i = i + 1

            # Run the comparison
            for vd, sd, wd, wsd in zip(
                visData, data["vis"][i, :], weightData, data["weights"][i, :]
            ):
                self.assertAlmostEqual(vd, sd, 8)
                self.assertAlmostEqual(wd, wsd, 6)
            i = i + 1

        ms.close()
        ms2.close()

    def test_multi_if(self):
        """writing more than one spectral window to a MeasurementSet."""
        if run_ms_tests == False:
            return
        testTime = float(86400.0 * Time(time.time(), format="unix").mjd)
        testFile = os.path.join(self.testPath, "ms-test-MultiIF.ms")

        # Get some data
        data = self.__initData_WGS84()

        # Start the file
        fits = msv2.Ms(testFile, ref_time=testTime)
        fits.set_stokes(["xx"])
        fits.set_frequency(data["freq"], data["channel_width"])
        fits.set_frequency(data["freq"] + 10e6, data["channel_width"])

        fits.set_geometry(data["site"], data["antennas"])
        fits.add_data_set(
            testTime,
            6.0,
            data["bl"],
            numpy.concatenate([data["vis"], 10 * data["vis"]], axis=1),
        )
        fits.write()

        # Open the table and examine
        ms = casacore.tables.table(testFile, ack=False)
        uvw = ms.getcol("UVW")
        ant1 = ms.getcol("ANTENNA1")
        ant2 = ms.getcol("ANTENNA2")
        ddsc = ms.getcol("DATA_DESC_ID")
        vis = ms.getcol("DATA")

        ms2 = casacore.tables.table(
            os.path.join(testFile, "ANTENNA"), ack=False
        )
        mapper = ms2.getcol("NAME")

        ms3 = casacore.tables.table(
            os.path.join(testFile, "DATA_DESCRIPTION"), ack=False
        )
        spw = [i for i in ms3.getcol("SPECTRAL_WINDOW_ID")]

        # Correct number of visibilities
        self.assertEqual(uvw.shape[0], 2 * data["vis"].shape[0])
        self.assertEqual(vis.shape[0], 2 * data["vis"].shape[0])

        # Correct number of uvw coordinates
        self.assertEqual(uvw.shape[1], 3)

        # Correct number of frequencies
        self.assertEqual(vis.shape[1], data["freq"].size)

        # Correct values
        for row in range(uvw.shape[0]):
            stand1 = ant1[row]
            stand2 = ant2[row]
            descid = ddsc[row]
            visData = vis[row, :, 0]

            # Find out which visibility set in the random data corresponds to the
            # current visibility
            i = 0
            for a1, a2 in data["bl"]:
                if (
                    a1.stand.id == mapper[stand1]
                    and a2.stand.id == mapper[stand2]
                ):
                    break
                else:
                    i = i + 1

            # Find out which spectral window this corresponds to
            if spw[descid] == 0:
                compData = data["vis"]
            else:
                compData = 10 * data["vis"]

            # Run the comparison
            for vd, sd in zip(visData, compData[i, :]):
                self.assertAlmostEqual(vd, sd, 8)

        ms.close()
        ms2.close()
        ms3.close()

    def tearDown(self):
        """Remove the test path directory and its contents"""

        shutil.rmtree(self.testPath, ignore_errors=True)


class measurementset_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the lsl.reader units
    tests."""

    def __init__(self):
        unittest.TestSuite.__init__(self)

        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(measurementset_tests))


if __name__ == "__main__":
    unittest.main()
