# pylint: disable=attribute-defined-outside-init,invalid-name,
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=no-name-in-module,import-error,unused-variable
# pylint: disable=too-many-instance-attributes,undefined-variable
# flake8: noqa
"""
Unit tests for functions in data_convert_persist.
The functions facilitate persistence of data models using HDF5
"""

# funcs to migrate:
#   create_gaintable_from_visibility
#   create_pointingtable_from_visibility
#   create_flagtable_from_visibility
#   create_convolutionfunction_from_image
#   create_griddata_from_image
#   create_image

import logging
import os

import astropy.units as u
import numpy
import xarray
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.gridded_visibility import (
    export_convolutionfunction_to_hdf5,
    export_griddata_to_hdf5,
    import_convolutionfunction_from_hdf5,
    import_griddata_from_hdf5,
)
from ska_sdp_datamodels.image import (
    export_image_to_hdf5,
    import_image_from_hdf5,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import (
    SkyComponent,
    SkyModel,
    export_skycomponent_to_hdf5,
    export_skymodel_to_hdf5,
    import_skycomponent_from_hdf5,
    import_skymodel_from_hdf5,
)
from ska_sdp_datamodels.visibility import create_visibility
from src.processing_components.calibration.operations import (
    create_gaintable_from_visibility,
)
from src.processing_components.griddata import (
    create_convolutionfunction_from_image,
)
from src.processing_components.griddata.operations import (
    create_griddata_from_image,
)
from src.processing_components.image import create_image
from src.processing_components.parameters import rascil_path
from tests.utils import data_model_equals

log = logging.getLogger("src-logger")

log.setLevel(logging.INFO)


def setUp():
    results_dir = rascil_path("test_results")

    mid = create_named_configuration("MID", rmax=1000.0)
    times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 100.0)
    frequency = numpy.linspace(1.0e8, 1.1e8, 3)
    channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
    # Define the component and give it some spectral behaviour
    f = numpy.array([100.0, 20.0, -10.0, 1.0])
    flux = numpy.array([f, 0.8 * f, 0.6 * f])

    # The phase centre is absolute and the component
    # is specified relative (for now).
    # This means that the component should end up at
    # the position phasecentre+compredirection
    phasecentre = SkyCoord(
        ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    compabsdirection = SkyCoord(
        ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame="icrs", equinox="J2000"
    )
    comp = SkyComponent(
        direction=compabsdirection,
        frequency=frequency,
        flux=flux,
    )

    verbose = False


def test_readwriteimage():
    im = create_image(
        phasecentre=phasecentre,
        frequency=frequency,
        npixel=256,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    im["pixels"].data[...] = 1.0
    export_image_to_hdf5(
        im, f"{results_dir}/test_data_convert_persist_image.hdf"
    )
    newim = import_image_from_hdf5(
        f"{results_dir}/test_data_convert_persist_image.hdf"
    )
    assert data_model_equals(newim, im)


def test_readwriteimage_zarr():
    """
    Test to see if an image can be written to
    and read from a zarr file
    """
    im = create_image(
        phasecentre=phasecentre,
        frequency=frequency,
        npixel=256,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    rand = numpy.random.random(im["pixels"].shape)
    im["pixels"].data[...] = rand
    if verbose:
        print(im)

    # We cannot save dicts to a netcdf file
    im.attrs["clean_beam"] = ""

    store = os.path.expanduser(
        f"{results_dir}/test_data_convert_persist_image.zarr"
    )
    im.to_zarr(
        store=store,
        chunk_store=store,
        mode="w",
    )
    del im
    newim = xarray.open_zarr(store, chunk_store=store)
    assert newim["pixels"].data.compute().all() == rand.all()


def test_readwriteskycomponent():
    export_skycomponent_to_hdf5(
        comp,
        f"{results_dir}/test_data_convert_persist_skycomponent.hdf",
    )
    newsc = import_skycomponent_from_hdf5(
        f"{results_dir}/test_data_convert_persist_skycomponent.hdf"
    )

    assert newsc.flux.shape == comp.flux.shape
    assert numpy.max(numpy.abs(newsc.flux - comp.flux)) < 1e-15


def test_readwriteskymodel():
    vis = create_visibility(
        mid,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("linear"),
        weight=1.0,
    )
    im = create_image(
        phasecentre=phasecentre,
        frequency=frequency,
        npixel=256,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    gt = create_gaintable_from_visibility(vis, timeslice="auto")
    sm = SkyModel(components=[comp], image=im, gaintable=gt)
    export_skymodel_to_hdf5(
        sm, f"{results_dir}/test_data_convert_persist_skymodel.hdf"
    )
    newsm = import_skymodel_from_hdf5(
        f"{results_dir}/test_data_convert_persist_skymodel.hdf"
    )

    assert newsm.components[0].flux.shape == comp.flux.shape


def test_readwritegriddata():
    # This fails on comparison of the v axis.
    im = create_image(
        phasecentre=phasecentre,
        frequency=frequency,
        npixel=256,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    gd = create_griddata_from_image(im)
    export_griddata_to_hdf5(
        gd, f"{results_dir}/test_data_convert_persist_griddata.hdf"
    )
    newgd = import_griddata_from_hdf5(
        f"{results_dir}/test_data_convert_persist_griddata.hdf"
    )
    assert data_model_equals(newgd, gd)


def test_readwriteconvolutionfunction():
    # This fails on comparison of the v axis.
    im = create_image(
        phasecentre=phasecentre,
        frequency=frequency,
        npixel=256,
        polarisation_frame=PolarisationFrame("stokesIQUV"),
    )
    cf = create_convolutionfunction_from_image(im)
    if verbose:
        print(cf)
    export_convolutionfunction_to_hdf5(
        cf,
        f"{results_dir}/" f"test_data_convert_persist_convolutionfunction.hdf",
    )
    newcf = import_convolutionfunction_from_hdf5(
        f"{results_dir}/" f"test_data_convert_persist_convolutionfunction.hdf"
    )

    assert data_model_equals(newcf, cf)
