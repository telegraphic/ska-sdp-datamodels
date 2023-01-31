"""
Functions to create Visibility
"""

import logging
from typing import Optional

import numpy
import pandas
from astropy.coordinates import SkyCoord
from astropy.time import Time
from numpy.typing import NDArray

from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.configuration.config_coordinate_support import (
    hadec_to_azel,
    xyz_at_latitude,
    xyz_to_uvw,
)
from ska_sdp_datamodels.physical_constants import SIDEREAL_DAY_SECONDS
from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    correlate_polarisation,
)
from ska_sdp_datamodels.visibility.vis_model import FlagTable, Visibility
from ska_sdp_datamodels.visibility.vis_utils import (
    calculate_transit_time,
    generate_baselines,
)

log = logging.getLogger("data-models-logger")


# pylint: disable=too-many-arguments,too-many-locals,invalid-name
# pylint: disable=too-many-branches,too-many-statements
def create_visibility(
    config: Configuration,
    times: NDArray,
    frequency: NDArray,
    phasecentre: SkyCoord,
    channel_bandwidth: NDArray,
    weight: float = 1.0,
    polarisation_frame: Optional[PolarisationFrame] = None,
    integration_time: float = 1.0,
    zerow: bool = False,
    elevation_limit: Optional[float] = 15.0 * numpy.pi / 180.0,
    source: str = "unknown",
    meta: Optional[dict] = None,
    utc_time: Optional[Time] = None,
    times_are_ha: bool = True,
) -> Visibility:
    """
    Create a Visibility object with its main data array filled with complex
    double-precision zeros, and its axes and other attributes adequately
    initialised. What 'adequately initialised' means is explained at length in
    the notes section below, as are several important caveats.

    This function caters specifically to visibility simulation purposes.
    In particular it computes approximate (u, v, w) coordinates from scratch
    rather than taking them as an externally precalculated input. See notes.

    Parameters
    ----------
    config : Configuration
        Configuration object describing the interferometer array with which
        the output Visibilities are assumed to have been observed. This is used
        to calculate (u, v, w) coordinates.
    times : ndarray
        One-dimensional numpy array of floating point numbers representing
        an hour angles in radians. Specifically, the hour angles under which
        the phase centre is seen from the centre of the interferometer. These
        get converted to a timestamp stored in the output Visibility, via a
        process explained in the notes.

        NOTE: how the data are interpreted is controlled by the `times_are_ha`
        parameter. `times_are_ha=True` by default and you should *always* use
        this. If `times_are_ha=False`, the data are also interpreted as radians
        and then scaled by a factor SIDEREAL_DAY_SECONDS / SOLAR_DAY_SECONDS.
        That code branch is not understood and likely an incorrect remnant of
        the past. Do NOT use it.

        Parameter should be renamed `hour_angles` in the future, and the
        `times_are_ha=False` branch should be removed.
    frequency : ndarray
        One-dimensional numpy array containing channel centre frequencies
        in Hz.
    phasecentre : astropy.coordinates.SkyCoord
        Phase centre coordinates as a SkyCoord object.
    channel_bandwidth : ndarray
        One-dimensional numpy array containing channel bandwidths in Hz.
        Must have the same length as `frequency`.
    weight : float, optional
        Intended to mean "weight given to valid data points", but not
        correctly implemented.
        Leave to default value of 1.0; should be removed in the future.
    polarisation_frame : PolarisationFrame or None, optional
        Polarisation frame of the output Visibility. If None, select the
        PolarisationFrame that corresponds to the ReceptorFrame of the array,
        which is specified inside 'config'.
    integration_time : float, optional
        Only used in the specific case where `times` only has one element.
        In this case, this will be the integration time associated to this
        unique time sample in the output Visibility, in seconds.
    zerow : bool, optional
        If True, forcibly set the output Visibility "w" coordinate to zero at
        the end of the initialisation process.
    elevation_limit : float or None, optional
        Elevation limit in radians. When creating the output Visibility,
        discard any time samples such that the phase centre would appear to be
        below `elevation_limit` when observed from the centre of the array.
        If `None`, no discarding is performed.
    source : str, optional
        Source name carried in the attrs of the output Visibility.
    meta : dict or None, optional
        Optional user-defined metadata that gets stored inside the attrs of the
        output Visibility.
    utc_time : astropy.time.Time or None, optional
        Used to convert input hour angles to actual timestamps in the output
        Visibility. How this is done is described in the notes. If None, this
        parameter is taken to be `2000-01-01T00:00:00`.

        Default is leaving this to None. The alternative code path is
        unused/untested in practice.
    times_are_ha : bool, optional
        Whether to interpret `times` as hour angles. Leave this to default of
        `True`, code path for `False` suspected incorrect (see `times` above).
        This parameter should be removed in the future.

    Returns
    -------
    Visibility
        Visibility object with its main data array filled with complex
        double-precision zeros. What its other axes / attributes contain
        depends on the options provided (see notes below).

    Notes
    -----
    Input hour angles are converted to a time close to `utc_time`, the latter
    being `2000-01-01T00:00:00` by default. Firstly, the code first finds the
    date of the transit of `phasecentre` that occurs nearest to `utc_time`;
    from there, it finds the times nearest to this transit time such that the
    phasecentre was seen under the input hour angles from the centre of the
    interferometer array. These are the timestamps of the output visibilities.
    Why things have been implemented that way is likely because it makes
    calculating accurate (u, v, w) easier. Setting the date to 1 Jan 2000
    avoids discrepancies between apparent and mean coordinates, since that
    makes effects from the Earth's precession/nutation negligible.
    See: https://en.wikipedia.org/wiki/Apparent_place

    The output Visibility contains *at most* one time sample per entry in the
    `times` array. If may contain fewer time samples if `elevation_limit` was
    specified.

    Integration times in the output Visibility are set to the difference
    between the consecutive timestamps of each visibility time sample, except
    when there is only one time sample in the output; in this case the value of
    `integration_time` is used.

    Data corresponding to autocorrelation baselines (ant1 == ant2) are both
    flagged and zero-weighted in the output Visibility.
    """
    if phasecentre is None:
        raise ValueError("Must specify phase centre")

    if utc_time is None:
        utc_time_zero = Time("2000-01-01T00:00:00", format="isot", scale="utc")
    elif isinstance(utc_time, Time):
        utc_time_zero = utc_time
        utc_time = None

    if polarisation_frame is None:
        polarisation_frame = correlate_polarisation(config.receptor_frame)

    latitude = config.location.geodetic[1].to("rad").value
    ants_xyz = config["xyz"].data
    ants_xyz = xyz_at_latitude(ants_xyz, latitude)
    nants = len(config["names"].data)

    baselines = pandas.MultiIndex.from_tuples(
        generate_baselines(nants), names=("antenna1", "antenna2")
    )
    nbaselines = len(baselines)

    # Find the number of integrations above elevation_limit
    ntimes = 0
    n_flagged = 0

    for itime, time in enumerate(times):
        # Calculate the az/el of the phasecentre as seen from the centre of
        # the array for this hour angle and declination
        if times_are_ha:
            ha = time
        else:
            ha = time * (SIDEREAL_DAY_SECONDS / 86400.0)

        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            # check if at this hour angle, phasecentre is above elevation_limit
            ntimes += 1
        else:
            # if target is below, flag it
            n_flagged += 1

    if ntimes == 0:
        raise ValueError(
            "No targets above elevation_limit; all points are flagged"
        )

    if elevation_limit is not None and n_flagged > 0:
        log.info(
            "create_visibility: flagged %d/%d times "
            "below elevation limit %f (rad)",
            n_flagged,
            ntimes,
            180.0 * elevation_limit / numpy.pi,
        )
    else:
        log.debug("create_visibility: created %d times", ntimes)

    npol = polarisation_frame.npol
    nchan = len(frequency)
    visshape = [ntimes, nbaselines, nchan, npol]
    rvis = numpy.zeros(visshape, dtype="complex")
    rflags = numpy.ones(visshape, dtype="int")
    rweight = numpy.ones(visshape)
    rtimes = numpy.zeros([ntimes])
    rintegrationtime = numpy.zeros([ntimes])
    ruvw = numpy.zeros([ntimes, nbaselines, 3])

    if utc_time is None:
        stime = calculate_transit_time(
            config.location, utc_time_zero, phasecentre
        )
        if stime.masked:
            stime = utc_time_zero

    # Do each time filling in the actual values
    itime = 0
    for _, time in enumerate(times):

        if times_are_ha:
            ha = time
        else:
            ha = time * (SIDEREAL_DAY_SECONDS / 86400.0)

        # Calculate the az/el of the phase centre as seen from the centre of
        # the array for this hour angle and declination
        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            rtimes[itime] = stime.mjd * 86400.0 + time * 86400.0 / (
                2.0 * numpy.pi
            )
            rweight[itime, ...] = 1.0
            rflags[itime, ...] = 1

            # Loop over all pairs of antennas. Note that a2>a1
            ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.rad)

            for ibaseline, (a1, a2) in enumerate(baselines):
                if a1 != a2:
                    rweight[itime, ibaseline, ...] = weight
                    rflags[itime, ibaseline, ...] = 0
                else:
                    rweight[itime, ibaseline, ...] = 0.0
                    rflags[itime, ibaseline, ...] = 1

                ruvw[itime, ibaseline, :] = ant_pos[a2, :] - ant_pos[a1, :]
                rflags[itime, ibaseline, ...] = 0

            if itime > 0:
                rintegrationtime[itime] = rtimes[itime] - rtimes[itime - 1]
            itime += 1

    if itime > 1:
        rintegrationtime[0] = rintegrationtime[1]
    else:
        rintegrationtime[0] = integration_time
    rchannel_bandwidth = channel_bandwidth
    if zerow:
        ruvw[..., 2] = 0.0

    vis = Visibility.constructor(
        uvw=ruvw,
        time=rtimes,
        frequency=frequency,
        vis=rvis,
        weight=rweight,
        baselines=baselines,
        flags=rflags,
        integration_time=rintegrationtime,
        channel_bandwidth=rchannel_bandwidth,
        polarisation_frame=polarisation_frame,
        source=source,
        meta=meta,
        phasecentre=phasecentre,
        configuration=config,
    )

    log.debug(
        "create_visibility: %d rows, %.3f GB",
        vis.visibility_acc.nvis,
        vis.visibility_acc.size(),
    )

    return vis


def create_flagtable_from_visibility(vis: Visibility) -> FlagTable:
    """
    Create FlagTable matching Visibility

    :param vis: Visibility object
    :return: FlagTable object
    """
    return FlagTable.constructor(
        flags=vis.flags,
        frequency=vis.frequency,
        channel_bandwidth=vis.channel_bandwidth,
        configuration=vis.configuration,
        time=vis.time,
        integration_time=vis.integration_time,
        polarisation_frame=vis.visibility_acc.polarisation_frame,
    )
