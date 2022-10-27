""" Unit tests for PointingTable Models
"""

import pytest
import numpy


from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.calibration.calibration_model import PointingTable


@pytest.fixture(scope="module", name="result_pointing_table")
def fixture_pointing_table():
    """
    Generate a simple pointing table using PointingTable.constructor
    """
    # Create a Configuration object
    name = "MID"
    location = (5109237.71471275, 2006795.66194638, -3239109.1838011)
    names = "M000"
    xyz = 222
    mount = "altaz"
    frame = None
    receptor_frame = ReceptorFrame("stokesI")
    diameter = 13.5
    offset = 0.0
    stations = 0
    vp_type = "MEERKAT"
    configuration = Configuration.constructor(name, location, names, xyz, mount, frame, receptor_frame, diameter, offset,
                                              stations, vp_type)
    pointing = numpy.ones((1, 1, 1, 1, 2))
    nominal = numpy.ones((1, 1, 1, 1, 2))
    time = numpy.ones(1)
    interval = numpy.ones(1)
    weight = numpy.ones((1, 1, 1, 1, 1))
    residual = numpy.ones((1, 1, 1, 2))
    frequency = numpy.ones(1)
    pointing_frame = "local"
    pointingcentre = (180., -35.)
    pointing_table = PointingTable.constructor(pointing, nominal, time, interval, weight, residual, frequency,
                                               receptor_frame, pointing_frame, pointingcentre, configuration)
    return pointing_table


def test_constructor_coords(result_pointing_table):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["time", "antenna", "frequency", "receptor", "angle"]
    result_coords = result_pointing_table.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
