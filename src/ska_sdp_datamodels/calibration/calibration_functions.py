# pylint: disable=invalid-name

"""
Functions working with calibration-type
data models.
"""

import collections
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
)
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model import ReceptorFrame


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
    f.attrs["receptor_frame"] = gt.receptor_frame.type
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
    receptor_frame = ReceptorFrame(f.attrs["receptor_frame"])
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
        receptor_frame=receptor_frame,
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
    for name in names:
        if name != "":
            ant_map.append(actual)
            actual += 1
        else:
            ant_map.append(-1)
    if actual == 0:
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


def import_gaintable_from_casa_cal_table(
    table_name,
    jones_type="B",
) -> GainTable:
    """
    Create gain table from Calibration table of CASA.
    This import gain table form calibration table of CASA.

    :param table_name: Name of CASA table file
    :param jones_type: Type of calibration matrix T or G or B
    :return: GainTable object

    """
    anttab, base_table, fieldtab, obs, spw = _load_casa_tables(table_name)

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
