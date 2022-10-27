""" Unit tests for GainTable Models
"""

import pytest
import numpy
import pandas
from xarray import DataArray


from src.ska_sdp_datamodels.polarisation_data_models import (
    PolarisationFrame,
    ReceptorFrame
)

from src.ska_sdp_datamodels.memory_data_models import (
    Configuration,
    GainTable
)

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

gain = numpy.ones((1,1,1,1,1))
time = numpy.ones((1))
interval = numpy.ones((1))
weight = numpy.ones((1,1,1,1,1))
residual = numpy.ones((1,1,1,1))
frequency = numpy.ones((1))
receptor_frame = ReceptorFrame("linear")
phasecentre = (180., -35.)
jones_type = "T"

@pytest.fixture(scope="module", name="result_gainTable")
def fixture_gainTable():
    """
    Generate a simple gain table using GainTable.constructor
    """

    gainTable = GainTable.constructor(gain, time, interval, 
    weight, residual, frequency, receptor_frame, phasecentre, 
    configuration, jones_type)
    return gainTable

def test_GainTable_constructor_coords(result_gainTable):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["time", "antenna", "frequency", 
        "receptor1", "receptor2"]
    result_coords = result_gainTable.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
