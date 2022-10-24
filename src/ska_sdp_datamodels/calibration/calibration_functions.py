# pylint: disable=invalid-name

"""
Functions working with calibration-type
data models.
"""

import collections
from typing import List, Union

import h5py
import xarray
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.calibration import GainTable, PointingTable
from ska_sdp_datamodels.configuration import (
    convert_configuration_from_hdf,
    convert_configuration_to_hdf,
)
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
