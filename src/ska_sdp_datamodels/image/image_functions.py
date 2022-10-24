# pylint: disable=invalid-name

"""
Functions working with Image model.
"""

import collections

import h5py
import numpy
import xarray
from astropy.wcs import WCS

from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.science_data_model import PolarisationFrame


def convert_image_to_hdf(im: Image, f):
    """Convert an Image to an HDF file

    :param im: Image
    :param f: hdf group
    :return: group with im added
    """
    if not isinstance(im, xarray.Dataset):
        raise ValueError(f"im is not xarray dataset {im}")
    if im.attrs["data_model"] != "Image":
        raise ValueError(f"fim is not an Image: {im}")

    f.attrs["data_model"] = "Image"
    f["data"] = im["pixels"].data
    f.attrs["wcs"] = numpy.string_(im.image_acc.wcs.to_header_string())
    f.attrs["phasecentre_coords"] = im.image_acc.phasecentre.to_string()
    f.attrs["phasecentre_frame"] = im.image_acc.phasecentre.frame.name
    f.attrs["polarisation_frame"] = im.image_acc.polarisation_frame.type
    f.attrs["frequency"] = im.frequency

    return f


def convert_hdf_to_image(f):
    """Convert HDF root to an Image

    :param f: hdf group
    :return: Image
    """
    if "data_model" in f.attrs.keys() and f.attrs["data_model"] == "Image":
        polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
        wcs = WCS(f.attrs["wcs"])
        data = numpy.array(f["data"])
        im = Image.constructor(
            data=data, polarisation_frame=polarisation_frame, wcs=wcs
        )
        return im

    return None


def export_image_to_hdf5(im, filename):
    """Export an Image or list to HDF5 format

    :param im: Image
    :param filename: Name of HDF5 file
    :return: None
    """

    if not isinstance(im, collections.abc.Iterable):
        im = [im]
    with h5py.File(filename, "w") as f:
        if isinstance(im, list):
            f.attrs["number_data_models"] = len(im)
            for i, v in enumerate(im):
                vf = f.create_group(f"Image{i}")
                convert_image_to_hdf(v, vf)
        else:
            f.attrs["number_data_models"] = 1
            vf = f.create_group("Image0")
            convert_image_to_hdf(im, vf)
        f.flush()
        f.close()


def import_image_from_hdf5(filename):
    """Import Image(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: single image or list of images
    """

    with h5py.File(filename, "r") as f:
        nimlist = f.attrs["number_data_models"]
        imlist = [convert_hdf_to_image(f[f"Image{i}"]) for i in range(nimlist)]
        if nimlist == 1:
            return imlist[0]

        return imlist
