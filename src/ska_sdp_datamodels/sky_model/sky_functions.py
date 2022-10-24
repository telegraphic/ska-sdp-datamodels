# pylint: disable=invalid-name

"""
Functions working with Sky-related data models.
"""

import ast
import collections
from typing import Union

import h5py
import numpy
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.calibration import (
    convert_gaintable_to_hdf,
    convert_hdf_to_gaintable,
)
from ska_sdp_datamodels.image import convert_hdf_to_image, convert_image_to_hdf
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent, SkyModel


def convert_skycomponent_to_hdf(sc: SkyComponent, f):
    """Convert SkyComponent to HDF

    :param sc: SkyComponent
    :param f: HDF root
    :return: group with sc added
    """

    def _convert_direction_to_string(d: SkyCoord):
        """Convert SkyCoord to string

        :param d: SkyCoord
        :return: String
        """
        return f"{d.ra.deg}, {d.dec.deg}, icrs"

    if not isinstance(sc, SkyComponent):
        raise ValueError(f"sc is not a SkyComponent: {sc}")

    f.attrs["data_model"] = "SkyComponent"
    f.attrs["direction"] = _convert_direction_to_string(sc.direction)
    f.attrs["frequency"] = sc.frequency
    f.attrs["polarisation_frame"] = sc.polarisation_frame.type
    f.attrs["flux"] = sc.flux
    f.attrs["shape"] = sc.shape
    f.attrs["params"] = str(sc.params)
    f.attrs["name"] = numpy.string_(sc.name)
    return f


def convert_hdf_to_skycomponent(f):
    """Convert HDF root to a SkyComponent

    :param f: hdf group
    :return: SkyComponent
    """

    def _convert_direction_from_string(s: str):
        """Convert direction (SkyCoord) from string

        :param s: String

        :return: astropy SkyCoord object
        """
        ra, dec, frame = s.split(",")
        d = SkyCoord(ra, dec, unit="deg", frame=frame.strip())
        return d

    assert f.attrs["data_model"] == "SkyComponent", "Not a SkyComponent"
    direction = _convert_direction_from_string(f.attrs["direction"])
    frequency = numpy.array(f.attrs["frequency"])
    name = f.attrs["name"]
    polarisation_frame = PolarisationFrame(f.attrs["polarisation_frame"])
    flux = f.attrs["flux"]
    shape = f.attrs["shape"]
    params = ast.literal_eval(f.attrs["params"])
    sc = SkyComponent(
        direction=direction,
        frequency=frequency,
        name=name,
        flux=flux,
        polarisation_frame=polarisation_frame,
        shape=shape,
        params=params,
    )
    return sc


def export_skycomponent_to_hdf5(sc: Union[SkyComponent, list], filename):
    """Export a SkyComponent or list to HDF5 format

    :param sc: SkyComponent
    :param filename: Name of HDF5 file
    :return: None
    """

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]
    with h5py.File(filename, "w") as f:
        f.attrs["number_data_models"] = len(sc)
        for i, s in enumerate(sc):
            sf = f.create_group(f"SkyComponent{i}")
            convert_skycomponent_to_hdf(s, sf)
        f.flush()


def import_skycomponent_from_hdf5(filename):
    """Import SkyComponent(s) from HDF5 format

    :param filename: Name of HDF5 file
    :return: single skycomponent or list of skycomponents
    """

    with h5py.File(filename, "r") as f:
        nsclist = f.attrs["number_data_models"]
        sclist = [
            convert_hdf_to_skycomponent(f[f"SkyComponent{i}"])
            for i in range(nsclist)
        ]
        if nsclist == 1:
            return sclist[0]

        return sclist


def export_skymodel_to_hdf5(sm, filename):
    """Export a Skymodel or list to HDF5 format

    :param sm: SkyModel or list of SkyModels
    :param filename: Name of HDF5 file
    """

    if not isinstance(sm, collections.abc.Iterable):
        sm = [sm]

    with h5py.File(filename, "w") as f:
        f.attrs["number_data_models"] = len(sm)
        for i, s in enumerate(sm):
            sf = f.create_group(f"SkyModel{i}")
            convert_skymodel_to_hdf(s, sf)
        f.flush()
        f.close()


def convert_skymodel_to_hdf(sm, f):
    """Convert a SkyModel to an HDF file

    :param sm: Skymodel
    :param f: hdf group
    :return: group with skymodel added
    """
    if not isinstance(sm, SkyModel):
        raise ValueError(f"sm is not a SkyModel: {sm}")

    f.attrs["data_model"] = "SkyModel"
    f.attrs["fixed"] = sm.fixed
    if sm.components is not None:
        f.attrs["number_skycomponents"] = len(sm.components)
        for i, sc in enumerate(sm.components):
            cf = f.create_group(f"skycomponent{i}")
            convert_skycomponent_to_hdf(sc, cf)
    if sm.image is not None:
        cf = f.create_group("image")
        convert_image_to_hdf(sm.image, cf)
    if sm.mask is not None:
        cf = f.create_group("mask")
        convert_image_to_hdf(sm.mask, cf)
    if sm.gaintable is not None:
        cf = f.create_group("gaintable")
        convert_gaintable_to_hdf(sm.gaintable, cf)
    return f


def import_skymodel_from_hdf5(filename):
    """Import a Skymodel or list from HDF5 format

    :param filename: Name of HDF5 file
    :return: SkyModel
    """

    with h5py.File(filename, "r") as f:
        nsmlist = f.attrs["number_data_models"]
        smlist = [
            convert_hdf_to_skymodel(f[f"SkyModel{i}"]) for i in range(nsmlist)
        ]
        if nsmlist == 1:
            return smlist[0]

        return smlist


def convert_hdf_to_skymodel(f):
    """Convert HDF to SkyModel

    :param f: hdf group
    :return: SkyModel
    """
    assert f.attrs["data_model"] == "SkyModel", f.attrs["data_model"]

    fixed = f.attrs["fixed"]

    ncomponents = f.attrs["number_skycomponents"]
    components = []
    for i in range(ncomponents):
        cf = f[(f"skycomponent{i}")]
        components.append(convert_hdf_to_skycomponent(cf))
    if "image" in f.keys():
        cf = f["image"]
        image = convert_hdf_to_image(cf)
    else:
        image = None
    if "mask" in f.keys():
        cf = f["mask"]
        mask = convert_hdf_to_image(cf)
    else:
        mask = None
    if "gaintable" in f.keys():
        cf = f["gaintable"]
        gaintable = convert_hdf_to_gaintable(cf)
    else:
        gaintable = None

    return SkyModel(
        image=image,
        components=components,
        gaintable=gaintable,
        mask=mask,
        fixed=fixed,
    )
