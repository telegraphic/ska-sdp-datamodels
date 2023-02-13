# pylint: disable=invalid-name, too-many-locals
"""
Functions to create and initialise calibration models GainTable and PointingTable.
"""
from typing import Literal, Union

import numpy

from ska_sdp_datamodels.calibration.calibration_model import (
    GainTable,
    PointingTable,
)
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    hadec_to_azel,
)
from ska_sdp_datamodels.science_data_model import ReceptorFrame
from ska_sdp_datamodels.visibility import Visibility
from ska_sdp_datamodels.visibility.vis_utils import (
    calculate_visibility_hourangles,
)


def create_gaintable_from_visibility(
    vis: Visibility,
    timeslice: Union[float, Literal["auto"], None] = None,
    jones_type: Literal["T", "G", "B"] = "T",
) -> GainTable:
    """
    Create a unity- or identity-initialised GainTable consistent with the
    given Visibility.

    GainTable either represents:

    - a collection of complex-valued scalar gains, if Visibility carries only
      Stokes I data.

    - a collection of 2x2 complex-valued Jones matrices otherwise.

    In the first case, gains are initialised to unity. In the second case,
    Jones matrices are initialised to the identity matrix.

    :param vis: Visibility object for which we want to create a matching GainTable
    :type vis: Visibility
    :param timeslice: Defines the time scale over which each gain solution is valid.
        This is used to define the time axis of the GainTable. This parameter is
        interpreted as follows depending on its type:

        - float: this is a custom time interval in seconds. Input timestamps
          are grouped by intervals of this duration, and said groups are separately
          averaged to produce the output time axis.

        - "auto" or None: match the time resolution of the input, i.e. copy
          the time axis of the input Visibility

    :type timeslice: float, str or None, optional
    :param jones_type: Type of Jones term, one of {"T", "G", "B"}.
        The frequency axis of the output GainTable depends on the value provided:

        - "B": the output frequency axis is the same as that of the input Visibility.

        - "T" or "G": the solution is assumed to be frequency-independent,
          and the frequency axis of the output contains a single value: the average
          frequency of the input Visibility's channels.

    :type jones_type: str, optional
    :return: GainTable object; its "gain" data variable is always 5-dimensional
        ``[ntimes, nants, nfreqs, nrec, nrec]`` but some axes may have a
        length of 1 as explained above. In particular, ``nrec=1`` when given
        pure Stokes I visibilities.
    :rtype: GainTable
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

    # There is only one receptor frame in Visibility
    # Use it for both receptor1 and receptor2
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
