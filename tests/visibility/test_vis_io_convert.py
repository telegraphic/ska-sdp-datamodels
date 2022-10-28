"""
Unit tests for import, export, convert functions of Visibility
"""

import tempfile

import h5py

from ska_sdp_datamodels.visibility import (
    export_visibility_to_hdf5,
    import_visibility_from_hdf5,
)
from tests.utils import data_model_equals


def test_export_visibility_to_hdf5(visibility):
    """
    We read back the file written by export_visibility_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_visibility_to_hdf5.hdf5"

        # tested function
        export_visibility_to_hdf5(visibility, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_vis = result_file["Visibility0"]
            assert result_vis.attrs["npol"] == visibility.visibility_acc.npol
            assert (
                result_vis.attrs["polarisation_frame"]
                == visibility.visibility_acc.polarisation_frame.type
            )
            assert result_vis.attrs["data_model"] == "Visibility"
            assert (result_vis["data_time"] == visibility.time.data).all()
            assert (result_vis["data_vis"] == visibility.vis.data).all()


def test_import_visibility_from_hdf5(visibility):
    """
    We import a previously written HDF5 file containing
    visibility data and we get the data we originally
    exported.

    Note: this test assumes that export_visibility_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # GIVEN
        test_hdf = f"{temp_dir}/test_export_visibility_to_hdf5.hdf5"
        export_visibility_to_hdf5(visibility, test_hdf)

        # WHEN
        result = import_visibility_from_hdf5(test_hdf)

        # THEN
        data_model_equals(result, visibility)
