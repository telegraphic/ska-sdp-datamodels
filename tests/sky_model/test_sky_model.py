"""
Unit test for sky model related functiosn
"""

import tempfile

import h5py
import numpy

from ska_sdp_datamodels.sky_model import (
    export_skycomponent_to_hdf5,
    export_skymodel_to_hdf5,
    import_skycomponent_from_hdf5,
    import_skymodel_from_hdf5,
)


def test_export_skycomponent_to_hdf5(sky_component):
    """
    We read back the file written by export_skycomponent_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_sky_component_to_hdf5.hdf5"

        # tested function
        export_skycomponent_to_hdf5(sky_component, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_sc = result_file["SkyComponent0"]
            assert (result_sc.attrs["flux"] == sky_component.flux).all()
            assert result_sc.attrs["data_model"] == "SkyComponent"
            assert (
                result_sc.attrs["frequency"] == sky_component.frequency
            ).all()
            assert (
                result_sc.attrs["polarisation_frame"]
                == sky_component.polarisation_frame.type
            )


def test_import_skycomponent_from_hdf5(sky_component):
    """
    We import a previously written HDF5 file containing
    sky_component data and we get the data we originally
    exported.

    Note: this test assumes that export_skycomponent_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # GIVEN
        test_hdf = f"{temp_dir}/test_export_sky_comp_to_hdf5.hdf5"
        export_skycomponent_to_hdf5(sky_component, test_hdf)

        # WHEN
        result = import_skycomponent_from_hdf5(test_hdf)

        # THEN
        assert result.flux.shape == sky_component.flux.shape
        assert numpy.max(numpy.abs(result.flux - sky_component.flux)) < 1e-15


def test_export_skymodel_to_hdf5(sky_model):
    """
    We read back the file written by export_skymodel_to_hdf5
    and get the data that we used to write the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_hdf = f"{temp_dir}/test_export_sky_model_to_hdf5.hdf5"

        # tested function
        export_skymodel_to_hdf5(sky_model, test_hdf)

        with h5py.File(test_hdf, "r") as result_file:
            assert result_file.attrs["number_data_models"] == 1

            result_sm = result_file["SkyModel0"]
            assert result_sm.attrs["number_skycomponents"] == 1
            for key in ["image", "gaintable", "mask"]:
                assert key in result_sm.keys()


def test_import_skymodel_from_hdf5(sky_model):
    """
    We import a previously written HDF5 file containing
    SkyModel data and we get the data we originally
    exported.

    Note: this test assumes that export_skymodel_to_hdf5
    works correctly, which is tested above.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # GIVEN
        test_hdf = f"{temp_dir}/test_export_sky_model_to_hdf5.hdf5"
        export_skymodel_to_hdf5(sky_model, test_hdf)

        # WHEN
        result = import_skymodel_from_hdf5(test_hdf)

        # THEN
        assert (
            result.components[0].flux.shape
            == sky_model.components[0].flux.shape
        )
        assert (
            numpy.max(
                numpy.abs(
                    result.components[0].flux - sky_model.components[0].flux
                )
            )
            < 1e-15
        ).all()
        assert (result.image["pixels"] == sky_model.image["pixels"]).all()
        assert (result.gaintable["gain"] == sky_model.gaintable["gain"]).all()
