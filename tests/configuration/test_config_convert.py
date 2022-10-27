"""
Test converting Configuration
"""

import tempfile

import h5py

from ska_sdp_datamodels.configuration import (
    Configuration,
    convert_configuration_from_hdf,
    convert_configuration_to_hdf,
)


def test_convert_configuration_to_hdf(low_aa05_config):
    """
    This test checks some of the HDF5 values
    which are written by convert_configuration_to_hdf.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        result_file = f"{temp_dir}/test_config_write.hdf"
        with h5py.File(result_file, "w") as file:
            hdf_group = file.create_group("test_config")
            convert_configuration_to_hdf(low_aa05_config, hdf_group)

        with h5py.File(result_file, "r") as file:
            assert "configuration" in file["test_config"].keys()
            config_group = file["test_config"]["configuration"]

            for attr in [
                "data_model",
                "frame",
                "location",
                "name",
                "receptor_frame",
            ]:
                assert attr in config_group.attrs.keys()

            assert config_group.attrs["data_model"] == "Configuration"
            assert config_group.attrs["name"] == low_aa05_config.name

            config_data = config_group["configuration"]
            assert (
                [
                    str(name, encoding="utf-8")
                    for name in config_data["names"][:]
                ]
                == low_aa05_config.names.data
            ).all()
            assert (config_data["xyz"][:] == low_aa05_config.xyz).all()
            assert (
                [
                    str(stat, encoding="utf-8")
                    for stat in config_data["stations"][:]
                ]
                == low_aa05_config.stations.data
            ).all()


def test_convert_configuration_from_hdf(low_aa05_config):
    """
    Load configuration from HDF5 and test that the data written
    are the data read. This assumes that convert_configuration_to_hdf
    works well and uses that to write the file which then
    is read back in.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        result_file = f"{temp_dir}/test_config_read.hdf"
        with h5py.File(result_file, "w") as file:
            hdf_group = file.create_group("test_config")
            convert_configuration_to_hdf(low_aa05_config, hdf_group)

        with h5py.File(result_file, "r") as file:
            hdf_group = file["test_config"]
            result_config = convert_configuration_from_hdf(hdf_group)

            assert isinstance(result_config, Configuration)
            assert (result_config.names == low_aa05_config.names).all()
            assert (result_config.vp_type == low_aa05_config.vp_type).all()
            assert (result_config.diameter == low_aa05_config.diameter).all()
            assert (
                result_config.attrs["location"]
                == low_aa05_config.attrs["location"]
            )
            assert (
                result_config.attrs["receptor_frame"]
                == low_aa05_config.attrs["receptor_frame"]
            )
