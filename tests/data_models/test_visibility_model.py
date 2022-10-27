# pylint: disable=no-name-in-module,import-error

"""
Unit tests for Visibility model
"""

import astropy.units as u
import numpy
import pytest
from astropy.coordinates import SkyCoord

from src.ska_sdp_datamodels.polarisation_data_models import (
    PolarisationFrame,
    ReceptorFrame
)

from src.ska_sdp_datamodels.memory_data_models import (
    Configuration,
    Visibility
)

#Create Configuration object

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

frequency=numpy.ones((1))
channel_bandwidth=numpy.ones((1))
phasecentre = (180., -35.)
uvw=numpy.ones((1,1,1))
time=numpy.ones((1))
vis=numpy.ones((1,1,1,1))
weight=None
integration_time=numpy.ones((1))
flags=numpy.ones((1,1,1,1))
baselines="baseline1"
meta=None

@pytest.fixture(scope="module", name="result_visibility")
def fixture_visibility():
    """
    Generate a simple gain table using Visibilty.constructor
    """
    polarisation_frame=PolarisationFrame("stokesI")
    source="anonymous"
    low_precision="float64"
    visibility = Visibility.constructor(frequency, channel_bandwidth,
        phasecentre, configuration, uvw, time, vis, weight, integration_time,
        flags, baselines, polarisation_frame, source, meta, low_precision 
    )
    return visibility

def test_Visibility_constructor_coords(result_visibility):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["time", "baselines", "frequency", 
        "polarisation", "spatial"]
    result_coords = result_visibility.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)