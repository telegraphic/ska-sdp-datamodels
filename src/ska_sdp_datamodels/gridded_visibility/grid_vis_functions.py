# pylint: disable=invalid-name

"""
Functions working with gridding visiblity
data models.
"""

import collections

import h5py
import numpy
import xarray
from astropy.wcs import WCS

from ska_sdp_datamodels.gridded_visibility.grid_vis_model import ConvolutionFunction, GridData
from ska_sdp_datamodels.science_data_model import PolarisationFrame


def convert_griddata_to_hdf(gd: GridData, f):
    """Convert a GridData to an HDF file

    :param gd: GridData
    :param f: hdf group
    :return: group with gd added
    """
    if not isinstance(gd, xarray.Dataset):
        raise ValueError(f"gd is not an xarray.Dataset: {gd}")
    if gd.attrs["data_model"] != "GridData":
        raise ValueError(f"gd is not a GridData: {gd}")

    f.attrs["data_model"] = "GridData"
    f["data"] = gd["pixels"].data

    f.attrs["grid_wcs"] = numpy.string_(
        gd.griddata_acc.griddata_wcs.to_header_string()
    )
    f.attrs["polarisation_frame"] = gd.griddata_acc.polarisation_frame.type
    return f


def convert_hdf_to_griddata(f):
    """Convert HDF root to a GridData

    :param f: hdf group
    :return: GridData
    """
    assert f.attrs["data_model"] == "GridData", "Not a GridData"
    data = numpy.array(f["data"])
    grid_wcs = WCS(f.attrs["grid_wcs"])
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])

    gd = GridData.constructor(
        data=data, polarisation_frame=polarisation_frame, grid_wcs=grid_wcs
    )

    return gd


def export_griddata_to_hdf5(gd, filename):
    """Export a GridData or list to HDF5 format

    :param gd: GridData
    :param filename: Name of HDF5 file
    :return:None
    """

    if not isinstance(gd, collections.abc.Iterable):
        gd = [gd]
    with h5py.File(filename, "w") as f:
        if isinstance(gd, list):
            f.attrs["number_data_models"] = len(gd)
            for i, v in enumerate(gd):
                vf = f.create_group(f"GridData{i}")
                convert_griddata_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("GridData0")
            convert_griddata_to_hdf(gd, vf)
        f.flush()
        f.close()


def import_griddata_from_hdf5(filename):
    """Import GridData(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: single griddata or list of griddatas
    """

    with h5py.File(filename, "r") as f:
        nimlist = f.attrs["number_data_models"]
        gdlist = [
            convert_hdf_to_griddata(f[f"GridData{i}"]) for i in range(nimlist)
        ]
        if nimlist == 1:
            return gdlist[0]

        return gdlist


def convert_convolutionfunction_to_hdf(cf: ConvolutionFunction, f):
    """Convert a ConvolutionFunction to an HDF file

    :param cf: ConvolutionFunction
    :param f: hdf group
    :return: group with cf added
    """
    if not isinstance(cf, xarray.Dataset):
        raise ValueError("cf is not an xarray.Dataset")
    if cf.attrs["data_model"] != "ConvolutionFunction":
        raise ValueError(f"cf is not a ConvolutionFunction: {cf}")

    f.attrs["data_model"] = "ConvolutionFunction"
    f["data"] = cf["pixels"].data
    f.attrs["grid_wcs"] = numpy.string_(
        cf.convolutionfunction_acc.cf_wcs.to_header_string()
    )
    f.attrs[
        "polarisation_frame"
    ] = cf.convolutionfunction_acc.polarisation_frame.type
    return f


def convert_hdf_to_convolutionfunction(f):
    """Convert HDF root to a ConvolutionFunction

    :param f: hdf group
    :return: ConvolutionFunction
    """
    assert f.attrs["data_model"] == "ConvolutionFunction", f.attrs[
        "data_model"
    ]
    data = numpy.array(f["data"])
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    cf_wcs = WCS(f.attrs["grid_wcs"])
    gd = ConvolutionFunction.constructor(
        data, cf_wcs=cf_wcs, polarisation_frame=polarisation_frame
    )
    return gd


def export_convolutionfunction_to_hdf5(cf, filename):
    """Export a ConvolutionFunction to HDF5 format

    :param cf: ConvolutionFunction
    :param filename: Name of HDF5 file
    :return: None
    """

    if not isinstance(cf, collections.abc.Iterable):
        cf = [cf]
    with h5py.File(filename, "w") as f:
        if isinstance(cf, list):
            f.attrs["number_data_models"] = len(cf)
            for i, v in enumerate(cf):
                vf = f.create_group(f"ConvolutionFunction{i}")
                convert_convolutionfunction_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("ConvolutionFunction0")
            convert_convolutionfunction_to_hdf(cf, vf)
        f.flush()
        f.close()


def import_convolutionfunction_from_hdf5(filename):
    """Import ConvolutionFunction(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: single convolution function or list of convolution functions
    """

    with h5py.File(filename, "r") as f:
        nimlist = f.attrs["number_data_models"]
        cflist = [
            convert_hdf_to_convolutionfunction(f[f"ConvolutionFunction{i}"])
            for i in range(nimlist)
        ]
        if nimlist == 1:
            return cflist[0]

        return cflist
