""" Unit tests for PointingTable Models
"""

import pytest
import numpy


from src.ska_sdp_datamodels.polarisation_data_models import (
    PolarisationFrame,
    ReceptorFrame,
)

from src.ska_sdp_datamodels.memory_data_models import (
    Configuration,
    PointingTable
)
from src.ska_sdp_datamodels.xarray_coordinate_support import (
    conv_func_wcs,
    griddata_wcs,
    image_wcs,
)

#Create a Configuration object

name = "MID"
location = (5109237.71471275, 2006795.66194638, -3239109.1838011)
names = "M000"
xyz = 222
mount = "altaz"
frame=None
receptor_frame=ReceptorFrame("linear")
diameter = 13.5
offset = 0.0
stations = 0
vp_type = "MEERKAT" 
configuration = Configuration.constructor(name, location, 
    names, xyz, mount, frame, receptor_frame, diameter,
    offset, stations, vp_type)

pointing = numpy.ones((1,1,1,1,2))
nominal = numpy.ones((1,1,1,1,2))
time = numpy.ones((1))
interval = numpy.ones((1))
weight = numpy.ones((1,1,1,1,1))
residual = numpy.ones((1,1,1,2))
frequency = numpy.ones((1))
receptor_frame = ReceptorFrame("linear")
pointing_frame = "local"
pointingcentre = (180., -35.)


@pytest.fixture(scope="module", name="result_pointingTable")
def fixture_pointingTable():
    """
    Generate a simple pointing table using PointingTable.constructor
    """

    pointingTable = PointingTable.constructor(pointing, nominal, time, interval, 
    weight, residual, frequency, receptor_frame, pointing_frame, pointingcentre,
    configuration)
    return pointingTable

def test_contrustor_coords(result_pointingTable):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["time", "antenna", "frequency", "receptor", "angle"]
    result_coords = result_pointingTable.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys) 