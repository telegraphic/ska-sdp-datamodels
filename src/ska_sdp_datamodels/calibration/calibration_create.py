# pylint: disable=too-many-locals

"""
Functions to create calibration models
from Visibility
"""

import numpy
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.units import Quantity

from ska_sdp_datamodels.calibration.calibration_model import (
    GainTable,
    PointingTable,
)
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    hadec_to_azel,
)
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model import ReceptorFrame
from ska_sdp_datamodels.visibility import Visibility
from ska_sdp_datamodels.visibility.vis_utils import (
    calculate_visibility_hourangles,
)


def create_gaintable_from_visibility(
    vis: Visibility,
    timeslice=None,
    jones_type="T",
) -> GainTable:
    """
    Create gain table from visibility.

    This makes an empty gain table consistent with the Visibility.

    :param vis: Visibility object
    :param timeslice: Time interval between solutions (s)
    :param jones_type: Type of calibration matrix T or G or B
    :return: GainTable object

    """
    nants = vis.visibility_acc.nants

    # Set up times
    if timeslice == "auto" or timeslice is None or timeslice <= 0.0:
        gain_time = vis.time
    else:
        nbins = max(
            1,
            numpy.ceil(
                (numpy.max(vis.time.data) - numpy.min(vis.time.data))
                / timeslice
            ).astype("int"),
        )

        gain_time = [
            numpy.average(times)
            for time, times in vis.time.groupby_bins(
                "time", nbins, squeeze=False
            )
        ]
        gain_time = numpy.array(gain_time)

    gain_interval = numpy.ones_like(gain_time)
    if len(gain_time) > 1:
        for time_index, _ in enumerate(gain_time):
            if time_index == 0:
                gain_interval[0] = gain_time[1] - gain_time[0]
            else:
                gain_interval[time_index] = (
                    gain_time[time_index] - gain_time[time_index - 1]
                )

    # Set the frequency sampling
    if jones_type == "B":
        gain_frequency = vis.frequency.data
        nfrequency = len(gain_frequency)
    elif jones_type in ("G", "T"):
        gain_frequency = numpy.average(vis.frequency) * numpy.ones([1])
        nfrequency = 1
    else:
        raise ValueError(f"Unknown Jones type {jones_type}")

    ntimes = len(gain_time)

    receptor_frame = ReceptorFrame(vis.visibility_acc.polarisation_frame.type)
    nrec = receptor_frame.nrec

    gain_shape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gain_shape, dtype="complex")
    if nrec > 1:
        gain[..., 0, 1] = 0.0
        gain[..., 1, 0] = 0.0

    gain_weight = numpy.ones(gain_shape)
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])

    gain_table = GainTable.constructor(
        gain=gain,
        time=gain_time,
        interval=gain_interval,
        weight=gain_weight,
        residual=gain_residual,
        frequency=gain_frequency,
        receptor_frame=receptor_frame,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        jones_type=jones_type,
    )

    return gain_table


def create_pointingtable_from_visibility(
    vis: Visibility,
    pointing_frame="azel",
    timeslice=None,
) -> PointingTable:
    """
    Create pointing table from visibility.

    This makes an empty pointing table consistent with the Visibility.

    :param vis: Visibility object
    :param pointing_frame: pointing frame; e.g. "azel", "local", etc.
    :param timeslice: Time interval between solutions (s)
    :return: PointingTable object
    """
    nants = vis.visibility_acc.nants

    if timeslice is None or timeslice == "auto":
        pointing_time = numpy.unique(vis.time)
        pointing_interval = vis.integration_time
    else:
        pointing_time = vis.time.data[0] + timeslice * numpy.unique(
            numpy.round((vis.time.data - vis.time.data[0]) / timeslice)
        )
        pointing_interval = timeslice * numpy.ones_like(pointing_time)

    ntimes = len(pointing_time)
    pointing_frequency = numpy.unique(vis.frequency)
    nfrequency = len(pointing_frequency)

    receptor_frame = ReceptorFrame(vis.visibility_acc.polarisation_frame.type)
    nrec = receptor_frame.nrec

    pointing_shape = [ntimes, nants, nfrequency, nrec, 2]
    pointing = numpy.zeros(pointing_shape)
    if nrec > 1:
        pointing[..., 0, 0] = 0.0
        pointing[..., 1, 0] = 0.0
        pointing[..., 0, 1] = 0.0
        pointing[..., 1, 1] = 0.0

    hour_angle = (
        calculate_visibility_hourangles(vis, time=pointing_time)
        .to("rad")
        .value
    )
    dec = vis.phasecentre.dec.rad
    latitude = vis.configuration.location.lat.rad
    azimuth, elevation = hadec_to_azel(hour_angle, dec, latitude)

    pointing_nominal = numpy.zeros([ntimes, nants, nfrequency, nrec, 2])
    pointing_nominal[..., 0] = azimuth[
        :, numpy.newaxis, numpy.newaxis, numpy.newaxis
    ]
    pointing_nominal[..., 1] = elevation[
        :, numpy.newaxis, numpy.newaxis, numpy.newaxis
    ]
    pointing_weight = numpy.ones(pointing_shape)
    pointing_residual = numpy.zeros([ntimes, nfrequency, nrec, 2])

    pointing_table = PointingTable.constructor(
        pointing=pointing,
        nominal=pointing_nominal,
        time=pointing_time,
        interval=pointing_interval,
        weight=pointing_weight,
        residual=pointing_residual,
        frequency=pointing_frequency,
        receptor_frame=receptor_frame,
        pointing_frame=pointing_frame,
        pointingcentre=vis.phasecentre,
        configuration=vis.configuration,
    )

    return pointing_table


def create_gaintable_from_casa_cal_table(
    msname,
    jones_type="B",
) -> GainTable:
    """
    Create gain table from Calibration table of CASA.

    This makes an empty gain table consistent with the Visibility.

    :param msname: Visibility object
    :param jones_type: Type of calibration matrix T or G or B
    :return: GainTable object

    """
    anttab, base_table, fieldtab, obs, spw = _load_casa_tables(msname)

    # Get times, interval, bandpass solutions
    gain_time = numpy.unique(base_table.getcol(columnname="TIME"))
    gain_interval = numpy.unique(base_table.getcol(columnname="INTERVAL"))
    gains = base_table.getcol(columnname="CPARAM")
    antenna = base_table.getcol(columnname="ANTENNA1")
    spec_wind_id = base_table.getcol(columnname="SPECTRAL_WINDOW_ID")[0]
    # Below are optional columns
    # field_id = numpy.unique(bt.getcol(columnname="FIELD_ID"))
    # gain_residual = bt.getcol(columnname="PARAMERR")
    # gain_weight = numpy.ones(gains.shape)

    # Get the frequency sampling information
    gain_frequency = spw.getcol(columnname="CHAN_FREQ")[spec_wind_id]
    nfrequency = spw.getcol(columnname="NUM_CHAN")[spec_wind_id]

    # pylint: disable=fixme
    # TODO: Need to confirm what receptor frame(s) are used
    receptor_frame = ReceptorFrame("circular")
    nrec = receptor_frame.nrec

    nants = len(numpy.unique(antenna))
    ntimes = len(gain_time)
    gain_shape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = numpy.ones(gain_shape, dtype="complex")
    if nrec > 1:
        gain[..., 0, 1] = gains[..., 0]
        gain[..., 1, 0] = gains[..., 1]

    gain_weight = numpy.ones(gain_shape)
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])

    # Get configuration
    ts_name = obs.getcol(columnname="TELESCOPE_NAME")[0]
    configuration = _generate_configuration_from_cal_table(anttab, ts_name)

    # Get phase_centres
    phase_centre = _get_phase_centre_from_cal_table(fieldtab)

    gain_table = GainTable.constructor(
        gain=gain,
        time=gain_time,
        interval=gain_interval,
        weight=gain_weight,
        residual=gain_residual,
        frequency=gain_frequency,
        receptor_frame=receptor_frame,
        phasecentre=phase_centre,
        configuration=configuration,
        jones_type=jones_type,
    )

    return gain_table


def _load_casa_tables(msname):
    # pylint: disable=import-error,import-outside-toplevel
    from casacore.tables import table
    base_table = table(tablename=msname)
    # spw --> spectral window
    spw = table(tablename=f"{msname}/SPECTRAL_WINDOW")
    obs = table(tablename=f"{msname}/OBSERVATION")
    anttab = table(f"{msname}/ANTENNA", ack=False)
    fieldtab = table(f"{msname}/FIELD", ack=False)
    return anttab, base_table, fieldtab, obs, spw


def _get_phase_centre_from_cal_table(field_table):
    phase_dir = field_table.getcol(columnname="PHASE_DIR")
    phase_centre = SkyCoord(
        ra=phase_dir[0][0][0] * u.rad,
        dec=phase_dir[0][0][1] * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    return phase_centre


def _generate_configuration_from_cal_table(antenna_table, telescope_name):
    names = numpy.array(antenna_table.getcol("NAME"))

    ant_map = []
    actual = 0
    # This assumes that the names are actually filled in!
    for _, name in enumerate(names):
        if name != "":
            ant_map.append(actual)
            actual += 1
        else:
            ant_map.append(-1)
    if actual == 0:
        ant_map = list(range(len(names)))
        names = numpy.repeat("No name", len(names))

    mount = numpy.array(antenna_table.getcol("MOUNT"))[names != ""]
    diameter = numpy.array(antenna_table.getcol("DISH_DIAMETER"))[names != ""]
    xyz = numpy.array(antenna_table.getcol("POSITION"))[names != ""]
    offset = numpy.array(antenna_table.getcol("OFFSET"))[names != ""]
    stations = numpy.array(antenna_table.getcol("STATION"))[names != ""]
    names = numpy.array(antenna_table.getcol("NAME"))[names != ""]

    location = EarthLocation(
        x=Quantity(xyz[0][0], "m"),
        y=Quantity(xyz[0][1], "m"),
        z=Quantity(xyz[0][2], "m"),
    )

    configuration = Configuration.constructor(
        name=telescope_name,
        location=location,
        names=names,
        xyz=xyz,
        mount=mount,
        frame="ITRF",
        receptor_frame=ReceptorFrame("linear"),
        diameter=diameter,
        offset=offset,
        stations=stations,
    )
    return configuration
