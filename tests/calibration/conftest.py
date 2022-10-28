# pylint: disable=too-many-locals

"""
Pytest Fixtures
"""

import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.calibration import GainTable, PointingTable
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    hadec_to_azel,
)
from ska_sdp_datamodels.science_data_model import ReceptorFrame


@pytest.fixture(scope="package", name="gain_table")
def gain_table_fixture():
    """
    GainTable fixture.
    Calculations based on create_gaintable_from_visibility
    """
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    configuration = create_named_configuration("LOWBD2-CORE")

    n_ants = 6
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
        configuration=configuration,
        jones_type=jones_type,
    )
    return gain_table


@pytest.fixture(scope="package", name="pointing_table")
def pointing_table_fixture():
    """
    PointingTable fixture.
    Calculations based on create_pointingtable_from_visibility
    """
    phase_centre = SkyCoord(
        ra=+180.0 * units.deg,
        dec=-35.0 * units.deg,
        frame="icrs",
        equinox="J2000",
    )
    configuration = create_named_configuration("LOWBD2-CORE")

    n_ants = 6

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
    latitude = configuration.location.lat.rad
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
        configuration=configuration,
    )

    return pointing_table
