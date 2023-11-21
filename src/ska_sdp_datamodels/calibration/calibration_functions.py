# pylint: disable=invalid-name, too-many-locals

"""
Functions working with calibration-type
data models.
"""

import collections
import json
import logging
from typing import List, Union

import h5py
import numpy
import xarray
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.units import Quantity

from ska_sdp_datamodels.calibration.calibration_model import (
    GainTable,
    PointingTable,
)
from ska_sdp_datamodels.configuration import (
    convert_configuration_from_hdf,
    convert_configuration_to_hdf,
    convert_configuration_to_json,
    convert_json_to_configuration,
)
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model import ReceptorFrame

log = logging.getLogger("data-models-logger")


def convert_gaintable_to_hdf(gt: GainTable, f):
    """Convert a GainTable to an HDF file

    :param gt: GainTable
    :param f: hdf group
    :return: group with gt added
    """
    if not isinstance(gt, xarray.Dataset):
        raise ValueError(f"gt is not an xarray.Dataset: {gt}")
    if gt.attrs["data_model"] != "GainTable":
        raise ValueError(f"gt is not a GainTable: {GainTable}")

    f.attrs["data_model"] = "GainTable"
    f.attrs["receptor_frame1"] = gt.receptor_frame1.type
    f.attrs["receptor_frame2"] = gt.receptor_frame2.type
    f.attrs["phasecentre_coords"] = gt.phasecentre.to_string()
    f.attrs["phasecentre_frame"] = gt.phasecentre.frame.name
    datavars = ["time", "gain", "weight", "residual", "interval", "frequency"]
    for var in datavars:
        f[f"data_{var}"] = gt[var].data
    return f


def convert_hdf_to_gaintable(f):
    """Convert HDF root to a GainTable

    :param f: hdf group
    :return: GainTable
    """
    assert f.attrs["data_model"] == "GainTable", "Not a GainTable"
    receptor_frame1 = ReceptorFrame(f.attrs["receptor_frame1"])
    receptor_frame2 = ReceptorFrame(f.attrs["receptor_frame2"])
    s = f.attrs["phasecentre_coords"].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    phasecentre = SkyCoord(
        ra=ss[0], dec=ss[1], frame=f.attrs["phasecentre_frame"]
    )

    time = f["data_time"][()]
    frequency = f["data_frequency"][()]
    gain = f["data_gain"][()]
    weight = f["data_weight"][()]
    residual = f["data_residual"][()]
    interval = f["data_interval"][()]
    gt = GainTable.constructor(
        time=time,
        frequency=frequency,
        gain=gain,
        weight=weight,
        residual=residual,
        interval=interval,
        receptor_frame=(receptor_frame1, receptor_frame2),
        phasecentre=phasecentre,
    )
    return gt


def export_gaintable_to_hdf5(gt: Union[GainTable, List[GainTable]], filename):
    """Export a GainTable or list to HDF5 format

    :param gt: GainTable or list
    :param filename: Name of HDF5 file
    :return: None
    """

    if not isinstance(gt, collections.abc.Iterable):
        gt = [gt]
    with h5py.File(filename, "w") as f:
        if isinstance(gt, list):
            f.attrs["number_data_models"] = len(gt)
            for i, g in enumerate(gt):
                gf = f.create_group(f"GainTable{i}")
                convert_gaintable_to_hdf(g, gf)
        else:
            f.attrs["number_data_models"] = 1
            gf = f.create_group("GainTable0")
            convert_gaintable_to_hdf(gt, gf)

        f.flush()


def import_gaintable_from_hdf5(filename):
    """Import GainTable(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: single gaintable or list of gaintables
    """

    with h5py.File(filename, "r") as f:
        ngtlist = f.attrs["number_data_models"]
        gtlist = [
            convert_hdf_to_gaintable(f[f"GainTable{i}"])
            for i in range(ngtlist)
        ]
        if ngtlist == 1:
            return gtlist[0]

        return gtlist


def convert_pointingtable_to_hdf(pt: PointingTable, f):
    """Convert a PointingTable to an HDF file

    :param pt: PointingTable
    :param f: hdf group
    :return: group with pt added
    """
    if not isinstance(pt, xarray.Dataset):
        raise ValueError(f"pt is not an xarray.Dataset: {pt}")
    if pt.attrs["data_model"] != "PointingTable":
        raise ValueError(f"pt is not a PointingTable: {pt}")

    f.attrs["data_model"] = "PointingTable"
    f.attrs["receptor_frame"] = pt.receptor_frame.type
    f.attrs["pointingcentre_coords"] = pt.pointingcentre.to_string()
    f.attrs["pointingcentre_frame"] = pt.pointingcentre.frame.name
    f.attrs["pointing_frame"] = pt.pointing_frame
    datavars = [
        "time",
        "nominal",
        "pointing",
        "weight",
        "residual",
        "interval",
        "frequency",
    ]
    for var in datavars:
        f[f"data_{var}"] = pt[var].data
    f = convert_configuration_to_hdf(pt.configuration, f)
    return f


def convert_hdf_to_pointingtable(f):
    """Convert HDF root to a PointingTable

    :param f: hdf group
    :return: PointingTable
    """
    assert f.attrs["data_model"] == "PointingTable", "Not a PointingTable"
    receptor_frame = ReceptorFrame(f.attrs["receptor_frame"])
    s = f.attrs["pointingcentre_coords"].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    pointingcentre = SkyCoord(
        ra=ss[0], dec=ss[1], frame=f.attrs["pointingcentre_frame"]
    )
    pointing_frame = f.attrs["pointing_frame"]
    configuration = convert_configuration_from_hdf(f)

    time = f["data_time"][()]
    frequency = f["data_frequency"][()]
    pointing = f["data_pointing"][()]
    nominal = f["data_nominal"][()]
    weight = f["data_weight"][()]
    residual = f["data_residual"][()]
    interval = f["data_interval"][()]

    pt = PointingTable.constructor(
        time=time,
        pointing=pointing,
        nominal=nominal,
        weight=weight,
        residual=residual,
        interval=interval,
        frequency=frequency,
        receptor_frame=receptor_frame,
        pointing_frame=pointing_frame,
        pointingcentre=pointingcentre,
        configuration=configuration,
    )
    return pt


def export_pointingtable_to_hdf5(pt: PointingTable, filename):
    """Export a PointingTable or list to HDF5 format

    :param pt: Pointing Table
    :param filename: Name of HDF5 file
    :return: None
    """

    if not isinstance(pt, collections.abc.Iterable):
        pt = [pt]
    with h5py.File(filename, "w") as f:
        if isinstance(pt, list):
            f.attrs["number_data_models"] = len(pt)
            for i, v in enumerate(pt):
                vf = f.create_group(f"PointingTable{i}")
                convert_pointingtable_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("PointingTable0")
            convert_pointingtable_to_hdf(pt, vf)
        f.flush()


def convert_pointingtable_to_json(pt):
    """
    Convert a pointingtable to a json dictionary.
    :param pt: Pointing Table
    :return: json str
    """
    pointing_dict = {}
    pointing_dict["attrs"] = {}
    pointing_dict["attrs"]["data_model"] = "PointingTable"
    pointing_dict["attrs"]["receptor_frame"] = pt.receptor_frame.type
    pointing_dict["attrs"][
        "pointingcentre_coords"
    ] = pt.pointingcentre.to_string()
    pointing_dict["attrs"]["receptor_frame"] = pt.receptor_frame.type
    pointing_dict["attrs"][
        "pointingcentre_frame"
    ] = pt.pointingcentre.frame.name
    pointing_dict["attrs"]["pointing_frame"] = pt.pointing_frame
    pointing_dict["attrs"]["configuration"] = convert_configuration_to_json(
        pt.configuration
    )

    pointing_dict["data_vars"] = {}
    for var in pt.data_vars:
        pointing_dict["data_vars"][var] = pt[var].data.tolist()

    coords = [
        "time",
        "interval",
        "frequency",
    ]
    pointing_dict["coords"] = {}
    for coord in coords:
        pointing_dict["coords"][coord] = pt[coord].data.tolist()

    return json.dumps(pointing_dict)


def convert_json_to_pointingtable(pt_json):
    """
    Convert a JSON to pointingtable.
    :param pt_json: Json
    :return: PointingTable
    """
    pointing_dict = json.loads(pt_json)

    receptor_frame = ReceptorFrame(pointing_dict["attrs"]["receptor_frame"])
    pointing_center_coords = pointing_dict["attrs"][
        "pointingcentre_coords"
    ].split()
    pointing_center_coords_degs = [
        float(pointing_center_coords[0]),
        float(pointing_center_coords[1]),
    ] * u.deg
    pointingcentre = SkyCoord(
        ra=pointing_center_coords_degs[0],
        dec=pointing_center_coords_degs[1],
        frame=pointing_dict["attrs"]["pointingcentre_frame"],
    )
    pointing_frame = pointing_dict["attrs"]["pointing_frame"]
    configuration = convert_json_to_configuration(
        pointing_dict["attrs"]["configuration"]
    )

    datadict = {}
    for var in pointing_dict["data_vars"].keys():
        datadict[var] = numpy.array(pointing_dict["data_vars"][var])

    coorddict = {}
    for coord in pointing_dict["coords"].keys():
        coorddict[coord] = numpy.array(pointing_dict["coords"][coord])

    pt = PointingTable.constructor(
        time=coorddict["time"],
        interval=coorddict["interval"],
        frequency=coorddict["frequency"],
        pointing=datadict["pointing"],
        nominal=datadict.get("nominal", None),
        weight=datadict["weight"],
        residual=datadict["residual"],
        expected_width=datadict.get("expected_width", None),
        fitted_width=datadict.get("fitted_width", None),
        fitted_width_std=datadict.get("fitted_width_std", None),
        fitted_height=datadict.get("fitted_height", None),
        fitted_height_std=datadict.get("fitted_height_std", None),
        receptor_frame=receptor_frame,
        pointing_frame=pointing_frame,
        pointingcentre=pointingcentre,
        configuration=configuration,
    )
    return pt


def import_pointingtable_from_hdf5(filename):
    """Import PointingTable(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: single pointingtable or list of pointingtables
    """

    with h5py.File(filename, "r") as f:
        nptlist = f.attrs["number_data_models"]
        ptlist = [
            convert_hdf_to_pointingtable(f[f"PointingTable{i}"])
            for i in range(nptlist)
        ]
        if nptlist == 1:
            return ptlist[0]

        return ptlist


# Below are helper functions for import_gaintable_from_casa_cal_table
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


def _generate_configuration_from_cal_table(
    antenna_table, telescope_name, receptor_frame
):
    names = numpy.array(antenna_table.getcol("NAME"))
    mount = numpy.array(antenna_table.getcol("MOUNT"))[names != ""]
    diameter = numpy.array(antenna_table.getcol("DISH_DIAMETER"))[names != ""]
    xyz = numpy.array(antenna_table.getcol("POSITION"))[names != ""]
    offset = numpy.array(antenna_table.getcol("OFFSET"))[names != ""]
    stations = numpy.array(antenna_table.getcol("STATION"))[names != ""]

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
        receptor_frame=receptor_frame,
        diameter=diameter,
        offset=offset,
        stations=stations,
    )
    return configuration


def _set_jones_type(base_table, jones_type):
    """
    Obtain the calibration solution type from the table and use this in
    preference to any user-defined value. If the table does not have this
    information, use the user-defined value.

    :param base_table: main CASA table
    :param jones_type: user-defined jones_type string
    :return: reformatted numpy arrays gains, gain_time, gain_interval, antenna
    """
    table_jones_type = ""
    try:
        table_jones_type = base_table.getkeyword("VisCal")
        log.info("Using table Jones type %s", table_jones_type)
    except RuntimeError:
        log.warning("No main-table keyword VisCal")

    if table_jones_type == "":
        log.info("Using user-defined Jones type %s", jones_type)
        table_jones_type = jones_type
    else:
        # update jones_type based on the VisCal keyword
        # this should just be the first character
        #  - "G Jones" -> "G"
        #  - "B Jones" -> "B"
        #  - "Df Jones" -> "D"
        #  - "K Jones" -> "K"
        #  - "Kcross Jones" -> "K"
        jones_type = table_jones_type[0]

    # interpret the VisCal string so we know how to read and stored the data
    # "G Jones", "B Jones", "Df Jones", "K Jones", "Kcross Jones"
    is_delay = False
    if table_jones_type[0] == "K":
        # "K Jones" or "Kcross Jones" delay terms
        is_delay = True
    is_leakage = False
    if (
        # "Df Jones" or "Kcross Jones" leakage terms
        table_jones_type.find("D") == 0
        or table_jones_type.find("Kcross") == 0
    ):
        is_leakage = True

    return jones_type, is_leakage, is_delay


def _reshape_3d_gain_tables(gains, gain_time, gain_interval, antenna):
    """
    reformat casa gain tables with shape [ntimes*nants, nfrequency, nrec] to
    have shape [ntimes, nants, nfrequency, nrec]. Initial rows are assumed to
    cycle through antennas for each time step. Time and antenana arrays are
    also reduced to one row per time or antenna respectively.

    :param gains: numpy array with shape [ntimes*nants, nfrequency, nrec]
    :param gain_time: numpy array with shape [ntimes*nants]
    :param gain_interval: numpy array with shape [ntimes*nants]
    :param antenna: numpy array with shape [ntimes*nants]
    :return: reformatted numpy arrays gains, gain_time, gain_interval, antenna

    """

    if gains.ndim != 3:
        raise ValueError(f"Expect 3d gains array, have {gains.ndim}")

    input_shape = numpy.shape(gains)

    nrow = input_shape[0]
    nfrequency = input_shape[1]
    nrec = input_shape[2]

    # Antenna rows in the same solution interval may have different times,
    # so cannot set ntimes based on unique time tags. Use the fact that we
    # require each antenna to have one row per solution interval to define
    # ntimes
    antenna = numpy.unique(antenna)
    nants = len(antenna)
    if nrow % nants != 0:
        raise ValueError("Require each antenna in each solution interval")
    ntimes = nrow // nants

    gains = numpy.reshape(gains, (ntimes, nants, nfrequency, nrec))

    # GainTable wants time and increment vectors with one value per
    # solution interval, however the main CASA cal table columns have
    # one row for each solution interval and antenna. Need to remove
    # duplicate values. Take the average time value per solution interval
    gain_time = numpy.mean(numpy.reshape(gain_time, (ntimes, nants)), axis=1)

    # check that the times are increasing
    if numpy.any(numpy.diff(gain_time) <= 0):
        raise ValueError(f"Time error {gain_time-gain_time[0]}")

    # take a single soln interval value per time (scan_id)
    if len(gain_interval) == nants * ntimes:
        gain_interval = gain_interval[::nants, ...]
    else:
        raise ValueError(f"interval length error: {len(gain_interval)}")

    return gains, gain_time, gain_interval, antenna


def _gain_tables_to_jones(table, frequency, is_leakage, is_delay):
    """
    Add the two table polarisations into Jones matrices in an appropriate way

    :param table: numpy gains array with shape [ntimes*nants,nfrequency,nrec]
    :param frequency: list of frequencies for converting time delay to phase
    :param is_leakage: list of frequencies for converting time delay to phase
    :return: numpy gains array with shape [ntimes*nants,nfrequency,nrec,nrec]
    """
    table_shape = table.shape
    ntimes = table_shape[0]
    nants = table_shape[1]
    nfrequency = table_shape[2]
    nrec = table_shape[3]

    gain = numpy.ones((ntimes, nants, nfrequency, nrec, nrec), dtype="complex")

    if nrec == 1:
        gain[..., 0, 0] = table[..., 0]
    elif nrec == 2 and not is_delay:
        if not is_leakage:
            # standard table Jones: G or B = [[gx,0],[0,gy]]
            gain[..., 0, 0] = table[..., 0]
            gain[..., 0, 1] = 0.0
            gain[..., 1, 0] = 0.0
            gain[..., 1, 1] = table[..., 1]
        else:
            # standard leakages Jones: D = [[1,dxy],[-dyx,1]]
            # dyx is not defined with a -1 coeff in CASA
            gain[..., 0, 0] = 1.0
            gain[..., 0, 1] = table[..., 0]
            gain[..., 1, 0] = table[..., 1]
            gain[..., 1, 1] = 1.0
    elif nrec == 2 and is_delay:
        # convert ns time delays to phase at the reference frequency
        if frequency is None:
            raise ValueError("Require frequency list for delay conversion")
        ns2rad = 2 * numpy.pi * frequency[0] * 1e-9
        if nfrequency != 1:
            raise ValueError("expect a single channel for delay fits")
        if not is_leakage:
            # standard table Jones: G = [[gx,0],[0,gy]]
            phase = ns2rad * table
            gain[..., 0, 0] = numpy.exp(1j * phase[..., 0])
            gain[..., 0, 1] = 0.0
            gain[..., 1, 0] = 0.0
            gain[..., 1, 1] = numpy.exp(1j * phase[..., 1])
        else:
            # standard leakages Jones: D = [[1,dxy],[-dyx,1]]
            phase = ns2rad * (table[..., 0] - table[..., 1])
            dxy = numpy.exp(1j * phase)
            gain[..., 0, 0] = 1.0
            gain[..., 0, 1] = dxy
            gain[..., 1, 0] = numpy.conj(dxy)
            gain[..., 1, 1] = 1.0
    else:
        raise ValueError(f"Unsure how to import {nrec} polarisations")

    return gain


def import_gaintable_from_casa_cal_table(
    table_name,
    jones_type="B",
    rec_frame=ReceptorFrame("linear"),
) -> GainTable:
    """
    Create gain table from Calibration table of CASA.
    This import gain table form calibration table of CASA.

    :param table_name: Name of CASA table file
    :param jones_type: Type of calibration matrix T or G or B.
        Overwritten if table keyword "VisCal" is set
    :param rec_frame: Receptor Frame for the GainTable
    :return: GainTable object

    """
    anttab, base_table, fieldtab, obs, spw = _load_casa_tables(table_name)

    # Get times, interval, bandpass solutions
    gain_time = base_table.getcol(columnname="TIME")
    gain_interval = base_table.getcol(columnname="INTERVAL")
    antenna = base_table.getcol(columnname="ANTENNA1")
    spec_wind_id = base_table.getcol(columnname="SPECTRAL_WINDOW_ID")[0]

    # Obtain the calibration solution type from the table
    jones_type, is_leakage, is_delay = _set_jones_type(base_table, jones_type)

    # gains and leakages are stored in CPARAM[nrec,nfrequency]
    # delays are stored in FPARAM[nrec,1]
    if is_delay:
        gains = base_table.getcol(columnname="FPARAM")
    else:
        gains = base_table.getcol(columnname="CPARAM")

    # Get the frequency sampling information
    gain_frequency = spw.getcol(columnname="CHAN_FREQ")[spec_wind_id]
    nfrequency = spw.getcol(columnname="NUM_CHAN")[spec_wind_id]

    # Get receptor frame from Measurement set input
    # Currently we use the same for ideal/model and measured
    receptor_frame = rec_frame
    nrec = receptor_frame.nrec

    if gains.ndim == 3:
        gains, gain_time, gain_interval, antenna = _reshape_3d_gain_tables(
            gains, gain_time, gain_interval, antenna
        )

    if gains.ndim != 4:
        raise ValueError(f"Tables have unexpected shape: {gains.ndim}")

    ntimes = len(gain_time)
    nants = len(antenna)

    # final check of the main table shape
    input_shape = numpy.shape(gains)
    if ntimes != input_shape[0]:
        raise ValueError("gain and time columns are inconsistent")
    if nants != input_shape[1]:
        raise ValueError("gain and antenna columns are inconsistent")
    if nfrequency != input_shape[2]:
        raise ValueError(f"tables have wrong number of channels: {nfrequency}")
    if nrec != input_shape[3]:
        raise ValueError(f"Tables have wrong number of receptors: {nrec}")

    gain = _gain_tables_to_jones(gains, gain_frequency, is_leakage, is_delay)

    # Set the gain weight to one and residual to zero
    # This is temporary since in current tables they are not provided.
    gain_weight = numpy.ones(gain.shape)
    gain_residual = numpy.zeros([ntimes, nfrequency, nrec, nrec])

    # Get configuration
    ts_name = obs.getcol(columnname="TELESCOPE_NAME")[0]
    configuration = _generate_configuration_from_cal_table(
        anttab, ts_name, receptor_frame
    )

    # Get phase_centres
    phase_centre = _get_phase_centre_from_cal_table(fieldtab)

    # pylint: disable=duplicate-code
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
