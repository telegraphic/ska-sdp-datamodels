# pylint: disable=no-name-in-module,import-error

"""
Unit tests for Visibility object
"""

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.science_data_model import PolarisationFrame
from src.processing_components import (
    create_named_configuration,
    create_visibility,
)


def test_visibility_copy():
    """
    TODO: we will need a good suit of input data,
      e.g. in-memory, simple visibility and configuration objects
      to be used for unit tests.
    """
    lowcore = create_named_configuration("LOWBD2-CORE")
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
    frequency = numpy.linspace(1.0e8, 1.1e8, 3)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    vis = create_visibility(
        lowcore,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )

    new_vis = vis.copy(deep=True)
    vis["vis"].data[...] = 0.0
    new_vis["vis"].data[...] = 1.0
    assert new_vis["vis"].data[0, 0].real.all() == 1.0
    assert vis["vis"].data[0, 0].real.all() == 0.0