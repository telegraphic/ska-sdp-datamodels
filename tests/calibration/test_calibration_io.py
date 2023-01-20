from unittest.mock import patch

import numpy

from ska_sdp_datamodels.calibration.calibration_create import (
    _generate_configuration_from_cal_table,
    _get_phase_centre_from_cal_table,
    create_gaintable_from_casa_cal_table,
)

NTIMES = 4
NANTS = 6
NFREQ = 3


class MockBaseTable:
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
    def getcol(self, columnname=None):
        if columnname == "CHAN_FREQ":
            return numpy.array([[8.0e9, 8.1e9, 8.2e9], [8.4e9, 8.5e9, 8.6e9]])

        if columnname == "NUM_CHAN":
            return numpy.array([NFREQ, NFREQ])


class MockAntennaTable:
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
    """TODO"""


class MockObservationTable:
    """TODO"""


def test_generate_configuration_from_cal_table():
    tel_name = "MY-SKA"
    result = _generate_configuration_from_cal_table(MockAntennaTable(), tel_name)

    assert result.attrs["name"] == tel_name
    assert result.attrs["location"] == "<expected_location>"
    assert result.attrs["receptor_frame"] == "<expected_receptor_frame>"
    assert result.coords.data == [0, 1, 2, 3, 4, 5]
    # ETC, check mount, diametre, xyz, offsets, and so on
    pass


def test_get_phase_centre_from_cal_table():
    result = _get_phase_centre_from_cal_table()


@patch("ska_sdp_datamodels.calibration.calibration_create._load_casa_tables")
def test_create_gaintable_from_casa_cal_table(mock_tables):
    mock_tables.return_value = (MockAntennaTable(), MockBaseTable(), MockFieldTable(),
                                MockObservationTable(), MockSpectralWindowTable())
    result = create_gaintable_from_casa_cal_table("fake_ms")
