
""" Unit tests for the FlagTable Model
"""

import pytest
import numpy
import pandas
from xarray import DataArray
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning

from src.ska_sdp_datamodels.polarisation_data_models import (
    PolarisationFrame,
    ReceptorFrame
)

from src.ska_sdp_datamodels.memory_data_models import (
    Configuration,
    FlagTable
)
from src.ska_sdp_datamodels.xarray_coordinate_support import (
    conv_func_wcs,
    griddata_wcs,
    image_wcs,
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


baselines=numpy.ones((1))
flags=numpy.ones((1,1,1,1))
frequency=numpy.ones((1))
channel_bandwidth=numpy.ones((1))
time=numpy.ones((1))
integration_time=numpy.ones((1))

@pytest.fixture(scope="module", name="result_flagTable")
def fixture_flagTable():
    """
    Generate a simple image using FlagTable.constructor.
    """
    polarisation_frame = PolarisationFrame("stokesIV")
    configuration = Configuration.constructor(name, location, 
        names, xyz, mount, frame, receptor_frame, diameter,
        offset, stations, vp_type)
    flagTable = FlagTable.constructor(baselines, flags, frequency, channel_bandwidth,
    configuration, time, integration_time, polarisation_frame)
    return flagTable

def test_constructor_coords(result_flagTable):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["frequency", "polarisation", "dv", "du", "w", "v", "u"]
    result_coords = result_flagTable.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)

