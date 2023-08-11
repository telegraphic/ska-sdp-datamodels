# pylint: disable=too-many-locals

"""
Pytest Fixtures
"""
import numpy
import pytest
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord

from ska_sdp_datamodels.calibration import (
    create_gaintable_from_visibility,
    create_pointingtable_from_visibility,
)
from ska_sdp_datamodels.configuration import (
    Configuration,
    create_named_configuration,
)
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    lla_to_ecef,
)
from ska_sdp_datamodels.gridded_visibility import (
    create_convolutionfunction_from_image,
    create_griddata_from_image,
)
from ska_sdp_datamodels.image import create_image
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent, SkyModel
from ska_sdp_datamodels.visibility import (
    create_flagtable_from_visibility,
    create_visibility,
)


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


@pytest.fixture(scope="package", name="flag_table")
def flag_table_fixture(visibility):
    """
    FlagTable fixture
    """
    return create_flagtable_from_visibility(visibility)


@pytest.fixture(scope="package", name="low_aa05_config")
def config_fixture_low():
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
    x_coord, y_coord, z_coord = lla_to_ecef(lat * units.deg, lon * units.deg, altitude)
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


@pytest.fixture(scope="package", name="low_aa05_vis")
def low_aa0_5_vis_fixture(low_aa05_config, phase_centre):
    """
    Low-AA0.5 visibility.
    """
    times = numpy.array([0.0])
    frequency = numpy.array([1.3e9])
    channel_bandwidth = numpy.array([1e8])
    polarisation_frame = PolarisationFrame("linear")

    vis = create_visibility(
        low_aa05_config,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        times=times,
        polarisation_frame=polarisation_frame,
        phasecentre=phase_centre,
        weight=1.0,
    )

    return vis


@pytest.fixture(scope="package", name="gain_table")
def gain_table_fixture(visibility):
    """
    GainTable fixture.
    """
    gain_table = create_gaintable_from_visibility(visibility, jones_type="B")
    return gain_table


@pytest.fixture(scope="package", name="pointing_table")
def pointing_table_fixture(visibility):
    """
    PointingTable fixture.
    """
    pointing_table = create_pointingtable_from_visibility(visibility)
    return pointing_table


@pytest.fixture(scope="package", name="sky_component")
def sky_comp_fixture(phase_centre):
    """
    SkyComponent fixture
    """
    frequency = numpy.linspace(1.0e8, 1.1e8, 3)
    flux_elem = numpy.array([100.0, 20.0, -10.0, 1.0])
    flux = numpy.array([flux_elem, 0.8 * flux_elem, 0.6 * flux_elem])

    comp = SkyComponent(
        direction=phase_centre,
        frequency=frequency,
        flux=flux,
    )
    return comp


@pytest.fixture(scope="package", name="sky_model")
def sky_model_fixture(image, sky_component, gain_table):
    """
    SkyModel fixture
    """
    mask = image.copy(deep=True)
    mask["pixels"].data[...] = image["pixels"].data[...] * 0
    return SkyModel(
        components=[sky_component],
        image=image,
        gaintable=gain_table,
        mask=mask,
    )


@pytest.fixture(scope="package", name="grid_data")
def grid_data_fixture(image):
    """
    GridData fixture
    """
    grid_data = create_griddata_from_image(image)
    return grid_data


@pytest.fixture(scope="package", name="conv_func")
def convolution_function_fixture(image):
    """
    ConvolutionFunction fixture
    """
    conv_func = create_convolutionfunction_from_image(image)
    return conv_func
