"""
Pytest Fixtures
"""

import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.image import create_image
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility


@pytest.fixture(scope="package", name="phase_centre")
def phase_centre_fixture():
    """
    PhaseCentre fixture
    """
    return SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )


@pytest.fixture(scope="package", name="visibility")
def visibility_fixture(phase_centre):
    """
    Visibility fixture
    """
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    frequency = numpy.linspace(0.8e8, 1.2e8, 5)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
    polarisation_frame = PolarisationFrame("linear")

    vis = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=polarisation_frame,
        phasecentre=phase_centre,
        weight=1.0,
    )
    return vis


@pytest.fixture(scope="package", name="image")
def image_fixture(phase_centre):
    """
    Image fixture
    """
    image = create_image(
        npixel=256,
        cellsize=0.000015,
        phasecentre=phase_centre,
        frequency=1.0e8,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    return image
