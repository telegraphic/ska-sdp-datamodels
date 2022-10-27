""" Unit tests for SkyComponent Class
"""

import pytest
import numpy

from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent

direction = (180., -35.)
frequency = numpy.ones(1)
name = "test"
flux = numpy.ones((1, 1))
shape = "Point"
polarisation_frame = PolarisationFrame("stokesI")


@pytest.fixture(scope="module", name="result_skyComponent")
def fixture_skyComponent():
    """
    Generate a simple image using __init__.
    """
    
    skyComponent = SkyComponent(direction, frequency, name,
        flux, shape, polarisation_frame, params = None)
    return skyComponent

def test_chan(result_skyComponent):
    
    nchans = result_skyComponent.nchan
    assert nchans == flux[0]

def test_npol(result_skyComponent):

    npols = result_skyComponent.npol
    assert npols == flux[-1]

def test__str__(result_skyComponent):
    params ={}
    s = "SkyComponent:\n"
    s += f"\tName: {name}\n"
    s += f"\tFlux: {flux}\n"
    s += f"\tFrequency: {frequency}\n"
    s += f"\tDirection: {direction}\n"
    s += f"\tShape: {shape}\n"
    s += f"\tParams: {params}\n"
    s += f"\tPolarisation frame: stokesI\n"
    assert str(result_skyComponent) == s