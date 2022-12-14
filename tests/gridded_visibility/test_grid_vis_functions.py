"""
Unit tests for gridded visibility data model functions
"""

import tempfile

import h5py

from ska_sdp_datamodels.gridded_visibility import (
    export_convolutionfunction_to_hdf5,
    export_griddata_to_hdf5,
    import_convolutionfunction_from_hdf5,
    import_griddata_from_hdf5,
)
from tests.utils import data_model_equals


def test_export_griddata_to_hdf5(grid_data):
    """
    We read back the file written by export_griddata_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_griddata_to_hdf5.hdf5"

        # tested function
        export_griddata_to_hdf5(grid_data, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_gd = result_file["GridData0"]
            assert result_gd.attrs["data_model"] == "GridData"
            assert (result_gd["data"] == grid_data.pixels.data).all()
            assert (
                result_gd.attrs["polarisation_frame"]
                == grid_data.griddata_acc.polarisation_frame.type
            )


def test_import_griddata_from_hdf5(grid_data):
    """
    We import a previously written HDF5 file containing
    grid data and we get the data we originally
    exported.

    Note: this test assumes that export_griddata_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # GIVEN
        test_hdf = f"{temp_dir}/test_export_griddata_to_hdf5.hdf5"
        export_griddata_to_hdf5(grid_data, test_hdf)

        # WHEN
        result = import_griddata_from_hdf5(test_hdf)

        # THEN
        data_model_equals(result, grid_data)


def test_export_convolutionfunction_to_hdf5(conv_func):
    """
    We read back the file written by
    export_convolutionfunction_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_convfunc_to_hdf5.hdf5"

        # tested function
        export_convolutionfunction_to_hdf5(conv_func, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_conv_func = result_file["ConvolutionFunction0"]
            assert (
                result_conv_func.attrs["data_model"] == "ConvolutionFunction"
            )
            assert (
                result_conv_func.attrs["polarisation_frame"]
                == conv_func.convolutionfunction_acc.polarisation_frame.type
            )
            assert (result_conv_func["data"] == conv_func.pixels.data).all()


def test_import_convolutionfunction_from_hdf5(conv_func):
    """
    We import a previously written HDF5 file containing
    ConvolutionFunction data and we get the data
    we originally exported.

    Note: this test assumes that
    export_convolutionfunction_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # GIVEN
        test_hdf = f"{temp_dir}/test_export_convfunc_to_hdf5.hdf5"
        export_convolutionfunction_to_hdf5(conv_func, test_hdf)

        # WHEN
        result = import_convolutionfunction_from_hdf5(test_hdf)

        # THEN
        data_model_equals(result, conv_func)
