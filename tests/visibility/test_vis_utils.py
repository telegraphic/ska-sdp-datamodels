"""
Unit tests for visibility utils.
"""

import numpy
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from ska_sdp_datamodels.visibility.vis_utils import calculate_transit_time

LOCATION = EarthLocation(
    lon=116.76444824 * units.deg, lat=-26.824722084 * units.deg, height=300.0
)
UTC_TIME = Time(["2020-01-01T00:00:00"], format="isot", scale="utc")


def test_transit_time(phase_centre):
    """
    Transit time is correctly calculated
    """
    transit_time = calculate_transit_time(LOCATION, UTC_TIME, phase_centre)
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895866)


def test_transit_time_below_horizon():
    """
    Returns transit time when phase_centre is below the horizon.
    """
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=+80.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    transit_time = calculate_transit_time(LOCATION, UTC_TIME, phase_centre)
    numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.895804)
