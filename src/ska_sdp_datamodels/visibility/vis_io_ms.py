# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
# pylint: disable=too-many-nested-blocks,too-many-branches
# pylint: disable=invalid-name, duplicate-code
"""
Base functions to create and export Visibility
from/into Measurement Set files.
They take definitions of columns from msv2.py
and interact with Casacore.
"""

import logging
import os

import numpy
import pandas
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility
from ska_sdp_datamodels.visibility.vis_utils import generate_baselines

log = logging.getLogger("data-models-logger")


def extend_visibility_to_ms(msname, bvis):
    """
    Visibility to MS converter
    If MS doesn't exist, use export;
    while if MS already exists, use extend by row.

    :param msname: File name of MS
    :param bvis: Visibility
    :return:
    """
    # Determine if file exists

    if not os.path.exists(msname):
        if bvis is not None:
            export_visibility_to_ms(msname, [bvis])
    else:
        if bvis is not None:
            extend_visibility_ms_row(msname, bvis)


def extend_visibility_ms_row(msname, vis):
    """Minimal Visibility to MS converter

    The MS format is much more general than the RASCIL Visibility so we
     cut many corners. This requires casacore to be
    installed. If not an exception ModuleNotFoundError is raised.

    Write a list of Visibility's to a MS file, split by field and
    spectral window

    :param msname: File name of MS
    :param vis_list: list of Visibility
    :return:
    """
    # pylint: disable=import-outside-toplevel
    try:
        from casacore.tables import table

    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("casacore is not installed") from exc

    ms_temp = msname + "____"
    export_visibility_to_ms(ms_temp, [vis], source_name=None)

    try:
        tab = table(msname, readonly=False, ack=False)
        log.debug("Open ms table: %s", tab.info())
        tmp = table(ms_temp, readonly=True, ack=False)
        log.debug("Open ms table: %s", tmp.info())
        tmp.copyrows(tab)
        log.debug("Merge data")
        tmp.close()
        tab.flush()
        tab.close()
    finally:
        import shutil

        if os.path.exists(ms_temp):
            shutil.rmtree(ms_temp, ignore_errors=False)


def export_visibility_to_ms(msname, vis_list, source_name=None):
    """Minimal Visibility to MS converter

    The MS format is much more general than the RASCIL Visibility
    so we cut many corners. This requires casacore to be
    installed. If not an exception ModuleNotFoundError is raised.

    Write a list of Visibility's to a MS file, split by field and
    spectral window

    :param msname: File name of MS
    :param vis_list: list of Visibility
    :param source_name: Source name to use
    :param ack: Ask casacore to acknowledge each table operation
    :return:
    """
    # pylint: disable=import-outside-toplevel
    try:
        from ska_sdp_datamodels.visibility.msv2fund import Antenna, Stand
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("casacore is not installed") from exc

    try:
        from ska_sdp_datamodels.visibility import msv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("cannot import msv2") from exc

    # Start the table
    tbl = msv2.Ms(
        msname,
        ref_time=0,
        source_name=source_name,
        frame=vis_list[0].configuration.attrs["frame"],
        if_delete=True,
    )
    for vis in vis_list:
        if source_name is None:
            source_name = vis.source
        # Check polarisation

        if vis.visibility_acc.polarisation_frame.type == "linear":
            polarization = ["XX", "XY", "YX", "YY"]
        elif vis.visibility_acc.polarisation_frame.type == "linearnp":
            polarization = ["XX", "YY"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesI":
            polarization = ["I"]
        elif vis.visibility_acc.polarisation_frame.type == "circular":
            polarization = ["RR", "RL", "LR", "LL"]
        elif vis.visibility_acc.polarisation_frame.type == "circularnp":
            polarization = ["RR", "LL"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesIQUV":
            polarization = ["I", "Q", "U", "V"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesIQ":
            polarization = ["I", "Q"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesIV":
            polarization = ["I", "V"]
        else:
            raise ValueError(
                f"Unknown visibility polarisation"
                f" {vis.visibility_acc.polarisation_frame.type}"
            )

        tbl.set_stokes(polarization)
        tbl.set_frequency(vis["frequency"].data, vis["channel_bandwidth"].data)
        n_ant = len(vis.attrs["configuration"].xyz)

        antennas = []
        names = vis.configuration.names.data
        xyz = vis.configuration.xyz.data
        for i, name in enumerate(names):
            antennas.append(
                Antenna(i, Stand(name, xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            )

        # Set baselines and data
        bl_list = []
        antennas2 = antennas

        for a_1 in range(0, n_ant):
            for a_2 in range(a_1, n_ant):
                bl_list.append((antennas[a_1], antennas2[a_2]))

        tbl.set_geometry(vis.configuration, antennas)

        int_time = vis["integration_time"].data
        assert vis["integration_time"].data.shape == vis["time"].data.shape

        # Now easier since the Visibility is baseline oriented
        for ntime, time in enumerate(vis["time"]):
            for ipol, pol in enumerate(polarization):
                if int_time[ntime] is not None:
                    tbl.add_data_set(
                        time.data,
                        int_time[ntime],
                        bl_list,
                        vis["vis"].data[ntime, ..., ipol],
                        weights=vis["weight"].data[ntime, ..., ipol],
                        pol=pol,
                        source=source_name,
                        phasecentre=vis.phasecentre,
                        uvw=vis["uvw"].data[ntime, :, :],
                    )
                else:
                    tbl.add_data_set(
                        time.data,
                        0,
                        bl_list,
                        vis["vis"].data[ntime, ..., ipol],
                        weights=vis["weight"].data[ntime, ..., ipol],
                        pol=pol,
                        source=source_name,
                        phasecentre=vis.phasecentre,
                        uvw=vis["uvw"].data[ntime, :, :],
                    )
    tbl.write()


def list_ms(msname, ack=False):
    """List sources and data descriptors in a MeasurementSet

    :param msname: File name of MS
    :param ack: Ask casacore to acknowledge each table operation
    :return: sources, data descriptors

    For example::
        print(list_ms('3C277.1_avg.ms'))
        (['1302+5748', '0319+415', '1407+284', '1252+5634', '1331+305'],
          [0, 1, 2, 3])
    """
    # pylint: disable=import-outside-toplevel
    try:
        from casacore.tables import table
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("casacore is not installed") from exc

    tab = table(msname, ack=ack)
    log.debug("list_ms: %s", tab.info())

    fieldtab = table(f"{msname}/FIELD", ack=False)
    sources = fieldtab.getcol("NAME")

    ddtab = table(f"{msname}/DATA_DESCRIPTION", ack=False)
    dds = list(range(ddtab.nrows()))

    return sources, dds


def create_visibility_from_ms(
    msname,
    channum=None,
    start_chan=None,
    end_chan=None,
    ack=False,
    datacolumn="DATA",
    selected_sources=None,
    selected_dds=None,
    average_channels=False,
):
    """Minimal MS to Visibility converter

    The MS format is much more general than the RASCIL Visibility so we cut
    many corners. This requires casacore to be installed. If not an exception
    ModuleNotFoundError is raised.

    Creates a list of Visibility's, split by field and spectral window

    Reading of a subset of channels is possible using either start_chan and
    end_chan or channnum. Using start_chan and end_chan is preferred since
    it only reads the channels required. Channum is more flexible and can
    be used to read a random list of channels.

    :param msname: File name of MS
    :param channum: range of channels e.g. range(17,32), default is None
                    meaning all
    :param start_chan: Starting channel to read
    :param end_chan: End channel to read
    :param ack: Ask casacore to acknowledge each table operation
    :param datacolumn: MS data column to read DATA, CORRECTED_DATA, or
                    MODEL_DATA
    :param selected_sources: Sources to select
    :param selected_dds: Data descriptors to select
    :param average_channels: Average all channels read
    :return: List of Visibility

    For example::

        selected_sources = ['1302+5748', '1252+5634']
        bvis_list = create_visibility_from_ms('../../data/3C277.1_avg.ms',
            datacolumn='CORRECTED_DATA',
            selected_sources=selected_sources)
        sources = numpy.unique([bv.source for bv in bvis_list])
        print(sources)
        ['1252+5634' '1302+5748']

    """
    # pylint: disable=import-outside-toplevel
    try:
        from casacore.tables import table
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("casacore is not installed") from exc

    tab = table(msname, ack=ack)
    log.debug("create_visibility_from_ms: %s", tab.info())

    if selected_sources is None:
        fields = numpy.unique(tab.getcol("FIELD_ID"))
    else:
        fieldtab = table(f"{msname}/FIELD", ack=False)
        sources = fieldtab.getcol("NAME")
        fields = []
        for field, source in enumerate(sources):
            if source in selected_sources:
                fields.append(field)
        assert len(fields) > 0, "No sources selected"

    if selected_dds is None:
        dds = numpy.unique(tab.getcol("DATA_DESC_ID"))
    else:
        dds = selected_dds

    log.info("Reading uni. fields %s, uni. data descs %s", fields, dds)
    vis_list = []
    for field in fields:
        ftab = table(msname, ack=ack).query(f"FIELD_ID=={field}", style="")
        if ftab.nrows() <= 0:
            raise ValueError(f"Empty selection for FIELD_ID={field}")
        for dd in dds:
            # Now get info from the subtables
            ddtab = table(f"{msname}/DATA_DESCRIPTION", ack=False)
            spwid = ddtab.getcol("SPECTRAL_WINDOW_ID")[dd]
            polid = ddtab.getcol("POLARIZATION_ID")[dd]
            ddtab.close()

            meta = {"MSV2": {"FIELD_ID": field, "DATA_DESC_ID": dd}}
            ms = ftab.query(f"DATA_DESC_ID=={dd}", style="")
            if ms.nrows() <= 0:
                raise ValueError(
                    f"Empty selection for FIELD_ID= {field} "
                    f"and DATA_DESC_ID={dd}"
                )
            log.debug("create_visibility_from_ms: Found %s rows", ms.nrows())
            # The TIME column has descriptor:
            # {'valueType': 'double', 'dataManagerType': 'IncrementalStMan',
            #   'dataManagerGroup': 'TIME',
            # 'option': 0, 'maxlen': 0, 'comment': 'Modified Julian Day',
            # 'keywords': {'QuantumUnits': ['s'], 'MEASINFO': {'type': 'epoch',
            #   'Ref': 'UTC'}}}
            otime = ms.getcol("TIME")
            datacol = ms.getcol(datacolumn, nrow=1)
            datacol_shape = list(datacol.shape)
            channels = datacol.shape[-2]
            log.debug("create_visibility_from_ms: Found %s channels", channels)
            if channum is None:
                if start_chan is not None and end_chan is not None:
                    try:
                        log.debug(
                            "Reading channs from %s to %s",
                            start_chan,
                            end_chan,
                        )
                        blc = [start_chan, 0]
                        trc = [end_chan, datacol_shape[-1] - 1]
                        channum = range(start_chan, end_chan + 1)
                        ms_vis = ms.getcolslice(datacolumn, blc=blc, trc=trc)
                        ms_flags = ms.getcolslice("FLAG", blc=blc, trc=trc)
                        ms_weight = ms.getcol("WEIGHT")

                    except IndexError as exc:
                        raise IndexError(
                            "channel number exceeds max. within ms"
                        ) from exc

                else:
                    log.debug("c_v_f_ms: Reading all %s channels", channels)
                    try:
                        channum = range(channels)
                        ms_vis = ms.getcol(datacolumn)[:, channum, :]
                        ms_weight = ms.getcol("WEIGHT")
                        ms_flags = ms.getcol("FLAG")[:, channum, :]
                        channum = range(channels)
                    except IndexError as exc:
                        raise IndexError(
                            "channel number exceeds max. within ms"
                        ) from exc
            else:
                log.debug(
                    "create_visibility_from_ms: Reading channels %s", channum
                )
                channum = range(channels)
                try:
                    ms_vis = ms.getcol(datacolumn)[:, channum, :]
                    ms_flags = ms.getcol("FLAG")[:, channum, :]
                    ms_weight = ms.getcol("WEIGHT")[:, :]
                except IndexError as exc:
                    raise IndexError(
                        "channel number exceeds max. within ms"
                    ) from exc

            if average_channels:
                weight = ms_weight[:, numpy.newaxis, :] * (1.0 - ms_flags)
                ms_vis = numpy.sum(weight * ms_vis, axis=-2)[
                    ..., numpy.newaxis, :
                ]
                sumwt = numpy.sum(weight, axis=-2)[..., numpy.newaxis, :]
                ms_vis[sumwt > 0.0] = ms_vis[sumwt > 0] / sumwt[sumwt > 0.0]
                ms_vis[sumwt <= 0.0] = 0.0 + 0.0j
                ms_flags = sumwt
                ms_flags[ms_flags <= 0.0] = 1.0
                ms_flags[ms_flags > 0.0] = 0.0

            uvw = -1 * ms.getcol("UVW")
            antenna1 = ms.getcol("ANTENNA1")
            antenna2 = ms.getcol("ANTENNA2")
            integration_time = ms.getcol("INTERVAL")

            time = otime - integration_time / 2.0

            start_time = numpy.min(time) / 86400.0
            end_time = numpy.max(time) / 86400.0

            log.debug(
                "create_visibility_from_ms: Observation from %s to %s",
                Time(start_time, format="mjd").iso,
                Time(end_time, format="mjd").iso,
            )

            spwtab = table(f"{msname}/SPECTRAL_WINDOW", ack=False)
            cfrequency = numpy.array(
                spwtab.getcol("CHAN_FREQ")[spwid][channum]
            )
            cchannel_bandwidth = numpy.array(
                spwtab.getcol("CHAN_WIDTH")[spwid][channum]
            )
            nchan = cfrequency.shape[0]
            if average_channels:
                cfrequency = numpy.array([numpy.average(cfrequency)])
                cchannel_bandwidth = numpy.array(
                    [numpy.sum(cchannel_bandwidth)]
                )
                nchan = cfrequency.shape[0]

            # Get polarisation info
            poltab = table(f"{msname}/POLARIZATION", ack=False)
            corr_type = poltab.getcol("CORR_TYPE")[polid]
            corr_type = sorted(corr_type)
            # These correspond to the CASA Stokes enumerations
            if numpy.array_equal(corr_type, [1, 2, 3, 4]):
                polarisation_frame = PolarisationFrame("stokesIQUV")
                npol = 4
            elif numpy.array_equal(corr_type, [1, 2]):
                polarisation_frame = PolarisationFrame("stokesIQ")
                npol = 2
            elif numpy.array_equal(corr_type, [1, 4]):
                polarisation_frame = PolarisationFrame("stokesIV")
                npol = 2
            elif numpy.array_equal(corr_type, [5, 6, 7, 8]):
                polarisation_frame = PolarisationFrame("circular")
                npol = 4
            elif numpy.array_equal(corr_type, [5, 8]):
                polarisation_frame = PolarisationFrame("circularnp")
                npol = 2
            elif numpy.array_equal(corr_type, [9, 10, 11, 12]):
                polarisation_frame = PolarisationFrame("linear")
                npol = 4
            elif numpy.array_equal(corr_type, [9, 12]):
                polarisation_frame = PolarisationFrame("linearnp")
                npol = 2
            elif numpy.array_equal(corr_type, [9]) or numpy.array_equal(
                corr_type, [1]
            ):
                npol = 1
                polarisation_frame = PolarisationFrame("stokesI")
            else:
                raise KeyError(
                    f"Polarisation not understood: {str(corr_type)}"
                )

            # Get configuration
            anttab = table(f"{msname}/ANTENNA", ack=False)
            names = numpy.array(anttab.getcol("NAME"))

            # pylint: disable=cell-var-from-loop
            ant_map = []
            actual = 0
            # This assumes that the names are actually filled in!
            for name in names:
                if name != "":
                    ant_map.append(actual)
                    actual += 1
                else:
                    ant_map.append(-1)

            if actual == 0:
                ant_map = list(range(len(names)))
                names = numpy.repeat("No name", len(names))

            mount = numpy.array(anttab.getcol("MOUNT"))[names != ""]
            # log.info("mount is: %s" % (mount))
            diameter = numpy.array(anttab.getcol("DISH_DIAMETER"))[names != ""]
            xyz = numpy.array(anttab.getcol("POSITION"))[names != ""]
            offset = numpy.array(anttab.getcol("OFFSET"))[names != ""]
            stations = numpy.array(anttab.getcol("STATION"))[names != ""]
            names = numpy.array(anttab.getcol("NAME"))[names != ""]
            nants = len(names)

            antenna1 = list(map(lambda i: ant_map[i], antenna1))
            antenna2 = list(map(lambda i: ant_map[i], antenna2))

            baselines = pandas.MultiIndex.from_tuples(
                generate_baselines(nants), names=("antenna1", "antenna2")
            )
            nbaselines = len(baselines)

            location = EarthLocation(
                x=Quantity(xyz[0][0], "m"),
                y=Quantity(xyz[0][1], "m"),
                z=Quantity(xyz[0][2], "m"),
            )

            configuration = Configuration.constructor(
                name="",
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
            # Get phasecentres
            fieldtab = table(f"{msname}/FIELD", ack=False)
            pc = fieldtab.getcol("PHASE_DIR")[field, 0, :]
            source = fieldtab.getcol("NAME")[field]
            phasecentre = SkyCoord(
                ra=pc[0] * u.rad,
                dec=pc[1] * u.rad,
                frame="icrs",
                equinox="J2000",
            )

            time_index_row = numpy.zeros_like(time, dtype="int")
            time_last = time[0]
            time_index = 0
            for row, _ in enumerate(time):
                if time[row] > time_last + 0.5 * integration_time[row]:
                    assert (
                        time[row] > time_last
                    ), "MS is not time-sorted - cannot convert"
                    time_index += 1
                    time_last = time[row]
                time_index_row[row] = time_index

            ntimes = time_index + 1

            assert ntimes == len(
                numpy.unique(time_index_row)
            ), "Error in finding data times"

            bv_times = numpy.zeros([ntimes])
            bv_vis = numpy.zeros([ntimes, nbaselines, nchan, npol]).astype(
                "complex"
            )
            bv_flags = numpy.zeros([ntimes, nbaselines, nchan, npol]).astype(
                "int"
            )
            bv_weight = numpy.zeros([ntimes, nbaselines, nchan, npol])
            bv_uvw = numpy.zeros([ntimes, nbaselines, 3])
            bv_integration_time = numpy.zeros([ntimes])

            for row, _ in enumerate(time):
                ibaseline = baselines.get_loc((antenna1[row], antenna2[row]))
                time_index = time_index_row[row]
                bv_times[time_index] = time[row]
                bv_vis[time_index, ibaseline, ...] = ms_vis[row, ...]
                bv_flags[time_index, ibaseline, ...][
                    ms_flags[row, ...].astype("bool")
                ] = 1
                bv_weight[time_index, ibaseline, :, ...] = ms_weight[
                    row, numpy.newaxis, ...
                ]
                bv_uvw[time_index, ibaseline, :] = uvw[row, :]
                bv_integration_time[time_index] = integration_time[row]

            vis_list.append(
                Visibility.constructor(
                    uvw=bv_uvw,
                    baselines=baselines,
                    time=bv_times,
                    frequency=cfrequency,
                    channel_bandwidth=cchannel_bandwidth,
                    vis=bv_vis,
                    flags=bv_flags,
                    weight=bv_weight,
                    integration_time=bv_integration_time,
                    configuration=configuration,
                    phasecentre=phasecentre,
                    polarisation_frame=polarisation_frame,
                    source=source,
                    meta=meta,
                )
            )
        tab.close()
    return vis_list
