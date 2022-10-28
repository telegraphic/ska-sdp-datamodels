"""
Pytest Fixtures
"""

import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import FlagTable, create_visibility


@pytest.fixture(scope="package", name="visibility")
def visibility_fixture():
    """
    Visibility fixture
    """
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    frequency = numpy.linspace(0.8e8, 1.2e8, 5)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
    polarisation_frame = PolarisationFrame("linear")

    # The phase centre is absolute and the component
    # is specified relative (for now). This means that the
    # component should end up at the position phasecentre+compredirection
    phasecentre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )

    vis = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=polarisation_frame,
        phasecentre=phasecentre,
        weight=1.0,
    )
    return vis


@pytest.fixture(scope="package", name="flag_table")
def flag_table_fixture(visibility):
    """
    FlagTable fixture
    """
    return FlagTable.constructor(
        flags=visibility.flags,
        frequency=visibility.frequency,
        channel_bandwidth=visibility.channel_bandwidth,
        configuration=visibility.configuration,
        time=visibility.time,
        integration_time=visibility.integration_time,
        polarisation_frame=visibility.visibility_acc.polarisation_frame,
    )
