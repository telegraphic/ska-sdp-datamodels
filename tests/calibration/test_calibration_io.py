# pylint: disable=missing-function-docstring, inconsistent-return-statements,
# pylint: disable=too-few-public-methods
"""
Unit Tests to create GainTable
from CASA Tables
"""
from unittest.mock import patch

import numpy
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.units import Quantity

from ska_sdp_datamodels.calibration.calibration_functions import (
    _generate_configuration_from_cal_table,
    _get_phase_centre_from_cal_table,
    import_gaintable_from_casa_cal_table,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.science_data_model import ReceptorFrame

NTIMES = 4
NANTS = 6
NFREQ = 3
TEL_NAME = "MY-SKA"


class MockBaseTable:
    """
    Mock Base Table Class
    """

    def getcol(self, columnname=None):
        if columnname == "TIME":
            return numpy.array(
                [4.35089331e09, 4.35089332e09, 4.35089333e09, 4.35089334e09]
            )

        if columnname == "INTERVAL":
            return numpy.array([10.0])

        if columnname == "CPARAM":
            return numpy.ones((NTIMES, NANTS, NFREQ, 1))

        if columnname == "ANTENNA1":
            return numpy.array([0, 1, 2, 3, 4, 5])

        if columnname == "SPECTRAL_WINDOW_ID":
            return numpy.array([0, 1])


class MockSpectralWindowTable:
    """
    Mock Spectral Window Table Class
    """

    def getcol(self, columnname=None):
        if columnname == "CHAN_FREQ":
            return numpy.array([[8.0e9, 8.1e9, 8.2e9], [8.4e9, 8.5e9, 8.6e9]])

        if columnname == "NUM_CHAN":
            return numpy.array([NFREQ, NFREQ])


class MockAntennaTable:
    """
    Mock Antenna Table Class
    """

    def getcol(self, columnname=None):
        if columnname == "NAME":
            return ["ANT1", "ANT2", "ANT3", "ANT4", "ANT5", "ANT6"]

        if columnname == "MOUNT":
            return ["ALT-AZ", "ALT-AZ", "ALT-AZ", "ALT-AZ", "ALT-AZ", "ALT-AZ"]

        if columnname == "DISH_DIAMETER":
            return numpy.array([25.0, 25.0, 25.0, 25.0, 25.0, 25.0])

        if columnname == "POSITION":
            return numpy.array(
                [
                    [-1601162.0, -5042003.0, 3554915.0],
                    [-1601192.0190292, -5042007.78341262, 3554960.73493029],
                    [-1601147.19047704, -5042040.12425644, 3554894.80919799],
                    [-1601110.11175873, -5041807.16312437, 3554839.91628013],
                    [-1601405.58491341, -5042041.04214758, 3555275.06577525],
                    [-1601093.35329757, -5042182.23690452, 3554815.49897078],
                ]
            )

        if columnname == "OFFSET":
            return numpy.array(
                [
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                    [2.18848512e-03, 0.00000000e00, 0.00000000e00],
                    [-3.02790361e-03, 0.00000000e00, 0.00000000e00],
                    [-9.89315100e-04, 0.00000000e00, 0.00000000e00],
                    [5.09647129e-04, 0.00000000e00, 0.00000000e00],
                    [-2.87800725e-03, 0.00000000e00, 0.00000000e00],
                ]
            )

        if columnname == "STATION":
            return [
                "SKAMID-CORE",
                "SKAMID-CORE",
                "SKAMID-CORE",
                "SKAMID-ARM1",
                "SKAMID-ARM2",
                "SKAMID-ARM3",
            ]


class MockFieldTable:
    """
    Mock Field Table Class
    """

    def getcol(self, columnname=None):
        if columnname == "PHASE_DIR":
            return numpy.array([[[0.0, 0.0]]])


class MockObservationTable:
    """
    Mock Observation Table Class
    """

    def getcol(self, columnname=None):
        if columnname == "TELESCOPE_NAME":
            return TEL_NAME


def test_generate_configuration_from_cal_table():

    result = _generate_configuration_from_cal_table(
        MockAntennaTable(), TEL_NAME
    )

    location = EarthLocation(
        x=Quantity(-1601162.0, "m"),
        y=Quantity(-5042003.0, "m"),
        z=Quantity(3554915.0, "m"),
    )
    assert result.attrs["name"] == TEL_NAME
    assert result.attrs["location"] == location
    assert result.attrs["receptor_frame"] == ReceptorFrame("linear")
    assert result.coords["id"].data.shape == (6,)
    # Optional: check mount, diametre, xyz, offsets, and so on


def test_get_phase_centre_from_cal_table():
    result = _get_phase_centre_from_cal_table(MockFieldTable())
    expected = SkyCoord(
        ra=0.0 * u.rad,
        dec=0.0 * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    assert result == expected


casacore = pytest.importorskip("python-casacore")


@patch("ska_sdp_datamodels.calibration.calibration_create._load_casa_tables")
def test_import_gaintable_from_casa_cal_table(mock_tables):
    mock_tables.return_value = (
        MockAntennaTable(),
        MockBaseTable(),
        MockFieldTable(),
        MockObservationTable(),
        MockSpectralWindowTable(),
    )
    result = import_gaintable_from_casa_cal_table("fake_ms")
    assert isinstance(result, GainTable)
    # Optional: assert specific attributes
