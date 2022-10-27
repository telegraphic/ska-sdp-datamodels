"""
Pytest fixtures.
"""

import numpy
import pytest
from astropy import units
from astropy.coordinates import EarthLocation

from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    lla_to_ecef,
)


@pytest.fixture(scope="package", name="low_aa05_config")
def config_fixture():
    """
    Configuration object fixture
    """
    location = EarthLocation(
        lon=116.69345390 * units.deg,
        lat=-26.86371635 * units.deg,
        height=300.0,
    )

    nants = 6
    aa05_low_coords = numpy.array(
        [
            [116.69345390, -26.86371635],
            [116.69365770, -26.86334071],
            [116.72963910, -26.85615287],
            [116.73007800, -26.85612864],
            [116.74788540, -26.88080530],
            [116.74733280, -26.88062234],
        ]
    )
    lon, lat = aa05_low_coords[:, 0], aa05_low_coords[:, 1]

    altitude = 300.0
    diameter = 38.0

    # pylint: disable=duplicate-code
    names = [
        "S008‐1",
        "S008‐2",
        "S009‐1",
        "S009‐2",
        "S010‐1",
        "S010‐2",
    ]
    mount = "XY"
    x_coord, y_coord, z_coord = lla_to_ecef(
        lat * units.deg, lon * units.deg, altitude
    )
    ant_xyz = numpy.stack((x_coord, y_coord, z_coord), axis=1)

    config = Configuration.constructor(
        name="LOW-AA0.5",
        location=location,
        names=names,
        mount=numpy.repeat(mount, nants),
        xyz=ant_xyz,
        vp_type=numpy.repeat("LOW", nants),
        diameter=diameter * numpy.ones(nants),
    )
    return config
