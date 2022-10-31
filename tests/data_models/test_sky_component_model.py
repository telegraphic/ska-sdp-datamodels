""" Unit tests for SkyComponent Class
"""
# make python-format
# make python lint
import numpy
import pytest

from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent


@pytest.fixture(scope="module", name="result_sky_component")
def fixture_sky_component():
    """
    Generate a simple sky component object using __init__.
    """
    direction = (180.0, -35.0)
    frequency = numpy.ones(1)
    name = "test"
    flux = numpy.ones((1, 1))
    shape = "Point"
    polarisation_frame = PolarisationFrame("stokesI")
    sky_component = SkyComponent(
        direction, frequency, name, flux, shape, polarisation_frame, params=None
    )
    return sky_component


def test_chan(result_sky_component):
    nchans = result_sky_component.nchan
    assert nchans == 1


def test_npol(result_sky_component):
    npols = result_sky_component.npol
    assert npols == 1


def test__str__(result_sky_component):
    params = {}
    s = "SkyComponent:\n"
    s += f"\tName: test\n"
    s += f"\tFlux: [[1.]]\n"
    s += f"\tFrequency: [1.]\n"
    s += f"\tDirection: (180.0, -35.0)\n"
    s += f"\tShape: Point\n"
    s += f"\tParams: {params}\n"
    s += f"\tPolarisation frame: stokesI\n"
    assert str(result_sky_component) == s
