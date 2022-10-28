# pylint: disable=too-many-locals

"""
Pytest Fixtures
"""
import copy

import numpy
import pytest
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.wcs import WCS

from ska_sdp_datamodels.calibration import GainTable, PointingTable
from ska_sdp_datamodels.configuration import (
    Configuration,
    create_named_configuration,
)
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    hadec_to_azel,
    lla_to_ecef,
)
from ska_sdp_datamodels.gridded_visibility import ConvolutionFunction, GridData
from ska_sdp_datamodels.image import create_image
from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.sky_model import SkyComponent, SkyModel
from ska_sdp_datamodels.visibility import FlagTable, create_visibility


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
    return FlagTable.constructor(
        flags=visibility.flags,
        frequency=visibility.frequency,
        channel_bandwidth=visibility.channel_bandwidth,
        configuration=visibility.configuration,
        time=visibility.time,
        integration_time=visibility.integration_time,
        polarisation_frame=visibility.visibility_acc.polarisation_frame,
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


@pytest.fixture(scope="package", name="gain_table")
def gain_table_fixture(phase_centre, low_aa05_config):
    """
    GainTable fixture.
    Calculations based on create_gaintable_from_visibility
    """
    n_ants = low_aa05_config.configuration_acc.nants
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)

    gain_interval = numpy.ones_like(times)
    for time_index, _ in enumerate(times):
        if time_index == 0:
            gain_interval[0] = times[1] - times[0]
        else:
            gain_interval[time_index] = (
                times[time_index] - times[time_index - 1]
            )

    jones_type = "B"
    frequency = numpy.linspace(0.8e8, 1.2e8, 5)
    n_freq = len(frequency)
    n_times = len(times)

    receptor_frame = ReceptorFrame("linear")
    n_rec = receptor_frame.nrec

    gain_shape = [n_times, n_ants, n_freq, n_rec, n_rec]
    gain = numpy.ones(gain_shape, dtype="complex")
    gain[..., 0, 1] = 0.0
    gain[..., 1, 0] = 0.0

    gain_weight = numpy.ones(gain_shape)
    gain_residual = numpy.zeros([n_times, n_freq, n_rec, n_rec])

    gain_table = GainTable.constructor(
        gain=gain,
        time=times,
        interval=gain_interval,
        weight=gain_weight,
        residual=gain_residual,
        frequency=frequency,
        receptor_frame=receptor_frame,
        phasecentre=phase_centre,
        configuration=low_aa05_config,
        jones_type=jones_type,
    )
    return gain_table


@pytest.fixture(scope="package", name="pointing_table")
def pointing_table_fixture(phase_centre, low_aa05_config):
    """
    PointingTable fixture.
    Calculations based on create_pointingtable_from_visibility
    """
    n_ants = low_aa05_config.configuration_acc.nants

    times = numpy.unique((numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0))
    n_times = len(times)
    pointing_interval = numpy.array([30.0] * n_times)

    frequency = numpy.unique(numpy.linspace(0.8e8, 1.2e8, 5))
    n_freq = len(frequency)

    receptor_frame = ReceptorFrame("linear")
    n_rec = receptor_frame.nrec

    pointing_shape = [n_times, n_ants, n_freq, n_rec, 2]
    pointing = numpy.zeros(pointing_shape)
    pointing[..., 0, 0] = 0.0
    pointing[..., 1, 0] = 0.0
    pointing[..., 0, 1] = 0.0
    pointing[..., 1, 1] = 0.0

    hour_angle = numpy.array(
        [
            0.00036201,
            0.00254964,
            0.00473728,
            0.00692491,
            0.00911255,
            0.01130018,
            0.01348782,
            0.01567545,
            0.01786309,
            0.02005072,
        ]
    )
    dec = phase_centre.dec.rad
    latitude = low_aa05_config.location.lat.rad
    azimuth, elevation = hadec_to_azel(hour_angle, dec, latitude)

    pointing_nominal = numpy.zeros([n_times, n_ants, n_freq, n_rec, 2])
    pointing_nominal[..., 0] = azimuth[
        :, numpy.newaxis, numpy.newaxis, numpy.newaxis
    ]
    pointing_nominal[..., 1] = elevation[
        :, numpy.newaxis, numpy.newaxis, numpy.newaxis
    ]
    pointing_weight = numpy.ones(pointing_shape)
    pointing_residual = numpy.zeros([n_times, n_freq, n_rec, 2])
    pointing_frame = "azel"

    pointing_table = PointingTable.constructor(
        pointing=pointing,
        nominal=pointing_nominal,
        time=times,
        interval=pointing_interval,
        weight=pointing_weight,
        residual=pointing_residual,
        frequency=frequency,
        receptor_frame=receptor_frame,
        pointing_frame=pointing_frame,
        pointingcentre=phase_centre,
        configuration=low_aa05_config,
    )

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


# pylint: disable=invalid-name
@pytest.fixture(scope="package", name="grid_data")
def grid_data_fixture(image):
    """
    GridData fixture
    Based on create_griddata_from_image
    """
    ft_types = ["UU", "VV"]
    nchan, npol, ny, nx = image["pixels"].shape
    gridshape = (nchan, npol, ny, nx)
    data = numpy.zeros(gridshape, dtype="complex")

    wcs = copy.deepcopy(image.image_acc.wcs)
    crval = wcs.wcs.crval
    crpix = wcs.wcs.crpix
    cdelt = wcs.wcs.cdelt
    ctype = wcs.wcs.ctype
    d2r = numpy.pi / 180.0
    cdelt[0] = 1.0 / (nx * cdelt[0] * d2r)
    cdelt[1] = 1.0 / (ny * cdelt[1] * d2r)

    # The negation in the longitude is needed by definition of RA, DEC
    grid_wcs = WCS(naxis=4)
    grid_wcs.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, crpix[2], crpix[3]]
    grid_wcs.wcs.ctype = [ft_types[0], ft_types[1], ctype[2], ctype[3]]
    grid_wcs.wcs.crval = [0.0, 0.0, crval[2], crval[3]]
    grid_wcs.wcs.cdelt = [cdelt[0], cdelt[1], cdelt[2], cdelt[3]]
    grid_wcs.wcs.radesys = "ICRS"
    grid_wcs.wcs.equinox = 2000.0

    polarisation_frame = image.image_acc.polarisation_frame

    return GridData.constructor(
        data, polarisation_frame=polarisation_frame, grid_wcs=grid_wcs
    )


# pylint: disable=invalid-name
@pytest.fixture(scope="package", name="conv_func")
def convolution_function_fixture(image):
    """
    ConvolutionFunction fixture
    Based on create_convolutionfunction_from_image
    """
    nchan, npol, ny, nx = image["pixels"].data.shape
    support = 16  # Support of final convolution function
    over_sampling = 8

    wcs = copy.deepcopy(image.image_acc.wcs.wcs)
    crval = wcs.crval
    crpix = wcs.crpix
    cdelt = wcs.cdelt
    ctype = wcs.ctype
    d2r = numpy.pi / 180.0
    cdelt[0] = 1.0 / (nx * cdelt[0] * d2r)
    cdelt[1] = 1.0 / (ny * cdelt[1] * d2r)

    cf_wcs = WCS(naxis=7)
    cf_wcs.wcs.crpix = [
        float(support // 2) + 1.0,
        float(support // 2) + 1.0,
        float(over_sampling // 2) + 1.0,
        float(over_sampling // 2) + 1.0,
        float(1 // 2 + 1.0),
        crpix[2],
        crpix[3],
    ]
    cf_wcs.wcs.ctype = ["UU", "VV", "DUU", "DVV", "WW", ctype[2], ctype[3]]
    cf_wcs.wcs.crval = [0.0, 0.0, 0.0, 0.0, 0.0, crval[2], crval[3]]
    cf_wcs.wcs.cdelt = [
        cdelt[0],
        cdelt[1],
        cdelt[0] / over_sampling,
        cdelt[1] / over_sampling,
        1.0e15,
        cdelt[2],
        cdelt[3],
    ]

    cf_wcs.wcs.radesys = "ICRS"
    cf_wcs.wcs.equinox = 2000.0

    cf_data = numpy.zeros(
        [nchan, npol, 1, over_sampling, over_sampling, support, support],
        dtype="complex",
    )

    polarisation_frame = image.image_acc.polarisation_frame

    return ConvolutionFunction.constructor(
        data=cf_data, cf_wcs=cf_wcs, polarisation_frame=polarisation_frame
    )
