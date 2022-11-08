"""
Test calibration data model functions.
"""

import tempfile

import h5py

from ska_sdp_datamodels.calibration import (
    export_gaintable_to_hdf5,
    export_pointingtable_to_hdf5,
    import_gaintable_from_hdf5,
    import_pointingtable_from_hdf5,
)
from tests.utils import data_model_equals


def test_export_gaintable_to_hdf5(gain_table):
    """
    We read back the file written by export_gaintable_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_gain_table_to_hdf5.hdf5"

        # tested function
        export_gaintable_to_hdf5(gain_table, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_gt = result_file["GainTable0"]
            assert (
                result_gt.attrs["receptor_frame"]
                == gain_table.receptor_frame.type
            )
            assert (
                result_gt.attrs["phasecentre_coords"]
                == gain_table.phasecentre.to_string()
            )
            assert (
                result_gt.attrs["phasecentre_frame"]
                == gain_table.phasecentre.frame.name
            )
            assert result_gt.attrs["data_model"] == "GainTable"
            assert (result_gt["data_time"] == gain_table.time.data).all()
            assert (result_gt["data_gain"] == gain_table.gain.data).all()


def test_import_gaintable_from_hdf5(gain_table):
    """
    We import a previously written HDF5 file containing
    gain data and we get the data we originally exported.

    Note: this test assumes that export_gaintable_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_gain_table_to_hdf5.hdf5"
        export_gaintable_to_hdf5(gain_table, test_hdf)

        result = import_gaintable_from_hdf5(test_hdf)

        data_model_equals(result, gain_table)


def test_export_pointingtable_to_hdf5(pointing_table):
    """
    We read back the file written by export_pointingtable_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_pointing_to_hdf5.hdf5"

        # tested function
        export_pointingtable_to_hdf5(pointing_table, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_pt = result_file["PointingTable0"]
            assert (
                result_pt.attrs["receptor_frame"]
                == pointing_table.receptor_frame.type
            )
            assert (
                result_pt.attrs["pointingcentre_coords"]
                == pointing_table.pointingcentre.to_string()
            )
            assert (
                result_pt.attrs["pointingcentre_frame"]
                == pointing_table.pointingcentre.frame.name
            )
            assert result_pt.attrs["data_model"] == "PointingTable"
            assert (
                result_pt["data_nominal"] == pointing_table.nominal.data
            ).all()
            assert (
                result_pt["data_frequency"] == pointing_table.frequency.data
            ).all()


def test_import_pointingtable_from_hdf5(pointing_table):
    """
    We import a previously written HDF5 file containing
    pointing data and we get the data we originally
    exported.

    Note: this test assumes that export_pointingtable_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_pointing_to_hdf5.hdf5"
        export_pointingtable_to_hdf5(pointing_table, test_hdf)

        result = import_pointingtable_from_hdf5(test_hdf)

        data_model_equals(result, pointing_table)
