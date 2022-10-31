""" Unit tests for the Configuration model
"""
# make python-format
# make python lint

import pytest
from xarray import DataArray

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    ReceptorFrame,
)


@pytest.fixture(scope="module", name="result_configuration")
def fixture_configuration():
    """
    Generate a configuration object using Configuration.constructor
    """
    name = "MID"
    location = (5109237.71471275, 2006795.66194638, -3239109.1838011)
    names = "M000"
    xyz = 222
    mount = "altaz"
    frame = None
    receptor_frame = ReceptorFrame("linear")
    diameter = 13.5
    offset = 0.0
    stations = 0
    vp_type = "MEERKAT"
    configuration = Configuration.constructor(
        name,
        location,
        names,
        xyz,
        mount,
        frame,
        receptor_frame,
        diameter,
        offset,
        stations,
        vp_type,
    )
    return configuration


def test_constructor_coords(result_configuration):
    """
    Constructor generates correctly generates coordinates
    """
    # Fails when "stations" is a string:
    # TypeError: not all arguments converted during string formatting

    expected_coords_keys = ["id", "spatial"]
    result_coords = result_configuration.coords
    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert len(result_coords["id"]) == 4
    assert (result_coords["spatial"] == ["X", "Y", "Z"]).all()


def test_constructor_data_vars(result_configuration):
    """
    Constructor generates correctly generates data variables
    """

    result_data_vars = result_configuration.data_vars

    assert len(result_data_vars) == 7  # 7 vars in data_vars
    assert "names" in result_data_vars.keys()
    assert isinstance(result_data_vars["names"], DataArray)
    assert (result_data_vars["names"].data == "M000").all()
    assert "xyz" in result_data_vars.keys()
    assert isinstance(result_data_vars["xyz"], DataArray)
    assert (result_data_vars["xyz"].data == 222).all()
    assert "diameter" in result_data_vars.keys()
    assert isinstance(result_data_vars["diameter"], DataArray)
    assert (result_data_vars["diameter"].data == 13.5).all()
    assert "mount" in result_data_vars.keys()
    assert isinstance(result_data_vars["mount"], DataArray)
    assert (result_data_vars["mount"].data == "altaz").all()
    assert "vp_type" in result_data_vars.keys()
    assert isinstance(result_data_vars["vp_type"], DataArray)
    assert (result_data_vars["vp_type"].data == "MEERKAT").all()
    assert "offset" in result_data_vars.keys()
    assert isinstance(result_data_vars["offset"], DataArray)
    assert (result_data_vars["offset"].data == 0.0).all()
    assert "stations" in result_data_vars.keys()
    assert isinstance(result_data_vars["stations"], DataArray)
    assert (result_data_vars["stations"].data == 0).all()


def test_constructor_attrs(result_configuration):
    """
    Constructor correctly generates attributes
    """

    result_attrs = result_configuration.attrs

    assert len(result_attrs) == 5
    assert result_attrs["data_model"] == "Configuration"
    assert result_attrs["name"] == "MID"
    assert result_attrs["location"] == (
        5109237.71471275,
        2006795.66194638,
        -3239109.1838011,
    )
    assert result_attrs["receptor_frame"] == ReceptorFrame("linear")
    assert result_attrs["frame"] is None


def test_property_accessor(result_configuration):
    """
    Configuration.configuration_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_configuration.configuration_acc

    assert accessor_object.nants == 4
