"""
Unit tests for the Configuration Class
"""
import numpy
from astropy import units
from astropy.coordinates import EarthLocation
from xarray import DataArray

from ska_sdp_datamodels.science_data_model.polarisation_model import (
    ReceptorFrame,
)


def test_constructor_coords(low_aa05_config):
    """
    Constructor correctly generates coordinates
    """
    expected_coords_keys = ["id", "spatial"]
    result_coords = low_aa05_config.coords
    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert len(result_coords["id"]) == 6
    assert (result_coords["spatial"] == ["X", "Y", "Z"]).all()


def test_constructor_data_vars(low_aa05_config):
    """
    Constructor correctly generates data variables
    """

    result_data_vars = low_aa05_config.data_vars

    assert len(result_data_vars) == 7

    for _, value in result_data_vars.items():
        assert isinstance(value, DataArray)

    assert (
        result_data_vars["names"].data
        == [
            "S008‐1",
            "S008‐2",
            "S009‐1",
            "S009‐2",
            "S010‐1",
            "S010‐2",
        ]
    ).all()
    assert result_data_vars["xyz"].data.shape == (6, 3)
    assert (result_data_vars["diameter"].data == 38.0).all()
    assert (result_data_vars["mount"].data == "XY").all()
    assert (result_data_vars["vp_type"].data == "LOW").all()
    assert (result_data_vars["offset"].data == 0.0).all()
    assert (
        result_data_vars["stations"].data
        == numpy.array(["0", "1", "2", "3", "4", "5"])
    ).all()


def test_constructor_attrs(low_aa05_config):
    """
    Constructor correctly generates attributes
    """

    result_attrs = low_aa05_config.attrs

    assert len(result_attrs) == 5
    assert result_attrs["data_model"] == "Configuration"
    assert result_attrs["name"] == "LOW-AA0.5"
    assert result_attrs["location"] == EarthLocation(
        lon=116.69345390 * units.deg,
        lat=-26.86371635 * units.deg,
        height=300.0,
    )
    assert result_attrs["receptor_frame"] == ReceptorFrame("linear")
    assert result_attrs["frame"] == ""


def test_property_accessor(low_aa05_config):
    """
    Configuration.configuration_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = low_aa05_config.configuration_acc

    assert accessor_object.nants == 6
