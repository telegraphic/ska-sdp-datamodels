"""
Unit tests for IO and convert functions of Image
"""

import os
import tempfile

import h5py
import numpy
import pytest
import xarray

from ska_sdp_datamodels.image import (
    export_image_to_hdf5,
    import_image_from_hdf5,
)
from tests.utils import data_model_equals


def test_export_visibility_to_hdf5(image):
    """
    We read back the file written by export_image_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_image_to_hdf5.hdf5"

        # tested function
        export_image_to_hdf5(image, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_img = result_file["Image0"]
            assert (
                result_img.attrs["phasecentre_coords"]
                == image.image_acc.phasecentre.to_string()
            )
            assert (
                result_img.attrs["polarisation_frame"]
                == image.image_acc.polarisation_frame.type
            )
            assert (
                result_img.attrs["phasecentre_frame"]
                == image.image_acc.phasecentre.frame.name
            )
            assert result_img.attrs["data_model"] == "Image"
            assert (result_img["data"] == image.pixels.data).all()


def test_import_visibility_from_hdf5(image):
    """
    We import a previously written HDF5 file containing
    image data and we get the data we originally
    exported.

    Note: this test assumes that export_image_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # GIVEN
        test_hdf = f"{temp_dir}/test_export_image_to_hdf5.hdf5"
        export_image_to_hdf5(image, test_hdf)

        # WHEN
        result = import_image_from_hdf5(test_hdf)

        # THEN
        data_model_equals(result, image)


def test_write_image_to_zarr(image):
    """
    Test to see if an image can be written to
    and read from a zarr file (via default Dataset methods)
    """
    pytest.importorskip("zarr")
    new_image = image.copy(deep=True)
    new_image["pixels"].data[...] = numpy.random.random(image["pixels"].shape)

    # We cannot save dicts to a netcdf file
    new_image.attrs["clean_beam"] = ""

    with tempfile.TemporaryDirectory() as temp_dir:
        store = os.path.expanduser(f"{temp_dir}/test_image_to_zarr.zarr")
        new_image.to_zarr(
            store=store,
            chunk_store=store,
            mode="w",
        )
        loaded_data = xarray.open_zarr(store, chunk_store=store)
        assert (
            loaded_data["pixels"].data.compute() == new_image["pixels"].data
        ).all()
