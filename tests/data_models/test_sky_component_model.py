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
        direction,
        frequency,
        name,
        flux,
        shape,
        polarisation_frame,
        params=None,
    )
    return sky_component


def test_nchan(result_sky_component):
    """
    Check nchans retunrs correct data
    """
    nchans = result_sky_component.nchan
    assert nchans == 1


def test_npol(result_sky_component):
    """
    Check npols retunrs correct data
    """
    npols = result_sky_component.npol
    assert npols == 1


def test__str__(result_sky_component):
    """
    Check __str__() returns the correct string
    """
    params = {}
    expected_text = "SkyComponent:\n"
    expected_text += "\tName: test\n"
    expected_text += "\tFlux: [[1.]]\n"
    expected_text += "\tFrequency: [1.]\n"
    expected_text += "\tDirection: (180.0, -35.0)\n"
    expected_text += "\tShape: Point\n"
    expected_text += f"\tParams: {params}\n"
    expected_text += "\tPolarisation frame: stokesI\n"
    assert str(result_sky_component) == expected_text
