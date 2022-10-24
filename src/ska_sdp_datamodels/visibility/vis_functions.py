# pylint: disable=invalid-name

"""
Functions working with Visibility and FlagData models.
"""

import ast
import collections

import h5py
import numpy
import pandas
import xarray
from astropy import units as u
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.configuration import (
    convert_configuration_from_hdf,
    convert_configuration_to_hdf,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import FlagTable, Visibility


def _generate_baselines(nant):
    """Generate mapping from antennas to baselines
    Note that we need to include auto-correlations
    since some input measurement sets
    may contain auto-correlations

    :param nant: Number of antennas
    """
    for ant1 in range(0, nant):
        for ant2 in range(ant1, nant):
            yield ant1, ant2


def convert_visibility_to_hdf(vis: Visibility, f):
    """Convert a Visibility to an HDF file

    :param vis: Visibility
    :param f: hdf group
    :return: group with vis added
    """
    if not isinstance(vis, xarray.Dataset):
        raise ValueError("vis is not an xarray.Dataset")

    if vis.attrs["data_model"] != "Visibility":
        raise ValueError(f"vis is not a Visibility: {vis}")

    # We only need to keep the things we need to reconstruct the data_model
    f.attrs["data_model"] = "Visibility"
    f.attrs["nants"] = numpy.max([b[1] for b in vis.baselines.data]) + 1
    f.attrs["nvis"] = vis.visibility_acc.nvis
    f.attrs["npol"] = vis.visibility_acc.npol
    f.attrs["phasecentre_coords"] = vis.phasecentre.to_string()
    f.attrs["phasecentre_frame"] = vis.phasecentre.frame.name
    f.attrs["polarisation_frame"] = vis.visibility_acc.polarisation_frame.type
    f.attrs["source"] = vis.source
    f.attrs["meta"] = str(vis.meta)
    datavars = [
        "time",
        "frequency",
        "channel_bandwidth",
        "vis",
        "weight",
        "flags",
        "uvw",
        "integration_time",
    ]
    for var in datavars:
        f[f"data_{var}"] = vis[var].data
    f = convert_configuration_to_hdf(vis.configuration, f)
    return f


# pylint: disable=too-many-locals
def convert_hdf_to_visibility(f):
    """Convert HDF root to visibility

    :param f: hdf group
    :return: Visibility
    """

    assert f.attrs["data_model"] == "Visibility", "Not a Visibility"
    s = f.attrs["phasecentre_coords"].split()
    ss = [float(s[0]), float(s[1])] * u.deg
    phasecentre = SkyCoord(
        ra=ss[0], dec=ss[1], frame=f.attrs["phasecentre_frame"]
    )
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    source = f.attrs["source"]
    nants = f.attrs["nants"]
    meta = ast.literal_eval(f.attrs["meta"])
    time = f["data_time"][()]
    frequency = f["data_frequency"][()]
    channel_bandwidth = f["data_channel_bandwidth"][()]
    uvw = f["data_uvw"][()]
    integration_time = f["data_integration_time"][()]
    vis = f["data_vis"][()]
    weight = f["data_weight"][()]
    flags = f["data_flags"][()]

    baselines = pandas.MultiIndex.from_tuples(
        _generate_baselines(nants),
        names=("antenna1", "antenna2"),
    )

    vis = Visibility.constructor(
        vis=vis,
        time=time,
        uvw=uvw,
        integration_time=integration_time,
        frequency=frequency,
        weight=weight,
        flags=flags,
        baselines=baselines,
        polarisation_frame=polarisation_frame,
        phasecentre=phasecentre,
        channel_bandwidth=channel_bandwidth,
        source=source,
        meta=meta,
        configuration=convert_configuration_from_hdf(f),
    )
    return vis


def convert_flagtable_to_hdf(ft: FlagTable, f):
    """Convert a FlagTable to an HDF file

    :param ft: FlagTable
    :param f: hdf group
    :return: group with ft added
    """
    if not isinstance(ft, xarray.Dataset):
        raise ValueError(f"ft is not an xarray.Dataset: {ft}")
    if ft.attrs["data_model"] != "FlagTable":
        raise ValueError(f"ft is not a FlagTable: {ft}")

    f.attrs["data_model"] = "FlagTable"
    f.attrs["nants"] = numpy.max([b[1] for b in ft.baselines.data]) + 1
    f.attrs["polarisation_frame"] = ft.flagtable_acc.polarisation_frame.type
    datavars = [
        "time",
        "frequency",
        "flags",
        "integration_time",
        "channel_bandwidth",
    ]
    for var in datavars:
        f[f"data_{var}"] = ft[var].data
    f = convert_configuration_to_hdf(ft.configuration, f)
    return f


def convert_hdf_to_flagtable(f):
    """Convert HDF root to flagtable

    :param f: hdf group
    :return: FlagTable
    """
    assert f.attrs["data_model"] == "FlagTable", "Not a FlagTable"
    nants = f.attrs["nants"]

    baselines = pandas.MultiIndex.from_tuples(
        _generate_baselines(nants), names=("antenna1", "antenna2")
    )
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    frequency = f["data_frequency"][()]
    channel_bandwidth = f["data_channel_bandwidth"][()]
    time = f["data_time"][()]
    flags = f["data_flags"][()]
    integration_time = f["data_integration_time"][()]
    ft = FlagTable.constructor(
        time=time,
        flags=flags,
        frequency=frequency,
        baselines=baselines,
        integration_time=integration_time,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=polarisation_frame,
        configuration=convert_configuration_from_hdf(f),
    )
    return ft


def export_visibility_to_hdf5(vis, filename):
    """Export a Visibility to HDF5 format

    :param vis: Visibility
    :param filename: Name of HDF5 file
    :return: None
    """

    if not isinstance(vis, collections.abc.Iterable):
        vis = [vis]
    with h5py.File(filename, "w") as f:
        if isinstance(vis, list):
            f.attrs["number_data_models"] = len(vis)
            for i, v in enumerate(vis):
                vf = f.create_group(f"Visibility{i}")
                convert_visibility_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("Visibility0")
            convert_visibility_to_hdf(vis, vf)

        f.flush()


def import_visibility_from_hdf5(filename):
    """Import Visibility(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: If only one then a Visibility, otherwise a list of Visibility's
    """

    with h5py.File(filename, "r") as f:
        nvislist = f.attrs["number_data_models"]
        vislist = [
            convert_hdf_to_visibility(f[f"Visibility{i}"])
            for i in range(nvislist)
        ]
        if nvislist == 1:
            return vislist[0]

        return vislist


def export_flagtable_to_hdf5(ft, filename):
    """Export a FlagTable or list to HDF5 format

    :param ft: FlagTable
    :param filename: Name of HDF5 file
    """

    if not isinstance(ft, collections.abc.Iterable):
        ft = [ft]
    with h5py.File(filename, "w") as f:
        if isinstance(ft, list):
            f.attrs["number_data_models"] = len(ft)
            for i, v in enumerate(ft):
                vf = f.create_group(f"FlagTable{i}")
                convert_flagtable_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("FlagTable0")
            convert_flagtable_to_hdf(ft, vf)
        f.flush()


def import_flagtable_from_hdf5(filename):
    """Import FlagTable(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: If only one then a FlagTable, otherwise a list of FlagTable's
    """

    with h5py.File(filename, "r") as f:
        nftlist = f.attrs["number_data_models"]
        ftlist = [
            convert_hdf_to_flagtable(f[f"FlagTable{i}"])
            for i in range(nftlist)
        ]
        if nftlist == 1:
            return ftlist[0]

        return ftlist
