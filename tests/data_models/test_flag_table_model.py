""" Unit tests for the FlagTable Model
"""
# make python-format
# make python lint

import pytest
import numpy

from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)

from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.visibility.vis_model import FlagTable

# Create a Configuration object
NAME = "MID"
LOCATION = (5109237.71471275, 2006795.66194638, -3239109.1838011)
NAMES = "M000"
XYZ = 222
MOUNT = "altaz"
FRAME = None
RECEPTOR_FRAME = ReceptorFrame("linear")
DIAMETER = 13.5
OFFSET = 0.0
STATIONS = 0
VP_TYPE = "MEERKAT"
CONFIGURATION = Configuration.constructor(
    NAME,
    LOCATION,
    NAMES,
    XYZ,
    MOUNT,
    FRAME,
    RECEPTOR_FRAME,
    DIAMETER,
    OFFSET,
    STATIONS,
    VP_TYPE,
)


@pytest.fixture(scope="module", name="result_flag_table")
def fixture_flag_table():
    """
    Generate a simple image using FlagTable.constructor.
    """

    baselines = numpy.ones(1)
    flags = numpy.ones((1, 1, 1, 1))
    frequency = numpy.ones(1)
    channel_bandwidth = numpy.ones(1)
    time = numpy.ones(1)
    integration_time = numpy.ones(1)
    polarisation_frame = PolarisationFrame("stokesI")

    flag_table = FlagTable.constructor(
        baselines,
        flags,
        frequency,
        channel_bandwidth,
        CONFIGURATION,
        time,
        integration_time,
        polarisation_frame,
    )
    return flag_table


def test_constructor_coords(result_flag_table):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["time", "baselines", "frequency", "polarisation"]
    result_coords = result_flag_table.coords
    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["time"] == 1
    assert result_coords["baselines"] == 1
    assert result_coords["frequency"] == 1
    assert result_coords["polarisation"] == "I"


def test_constructor_datavars(result_flag_table):
    """
    Constructor correctly generates data variables
    """

    result_datavars = result_flag_table.data_vars
    assert len(result_datavars) == 4
    assert (result_datavars["flags"] == 1).all()
    assert result_datavars["integration_time"] == 1
    assert result_datavars["channel_bandwidth"] == 1
    # TODO: figure out the time format used for datetime
    # assert result_datavars["datetime"] == 1


def test_constructor_attrs(result_flag_table):
    """
    Constructor correctly generates attributes.
    """
    result_attrs = result_flag_table.attrs

    assert len(result_attrs) == 3
    assert result_attrs["data_model"] == "FlagTable"
    assert result_attrs["_polarisation_frame"] == "stokesI"
    assert result_attrs["configuration"] == CONFIGURATION


def test_copy(result_flag_table):
    """
    Copy accurately copies a flag table
    """
    copied_ft_deep = result_flag_table.copy(True, None, False)
    copied_ft_no_deep = result_flag_table.copy(False, None, False)
    # copied_ft_zero = result_flag_table.copy(True, None, True)

    assert copied_ft_deep == result_flag_table
    assert copied_ft_no_deep == result_flag_table
    # TODO: test when Zero is True (needed data != None)
    # assert copied_ft_zero == result_flag_table


def test_property_accessor(result_flag_table):
    """
    FlagTable.flagtable_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_flag_table.flagtable_acc
    assert accessor_object.nchan == 1
    assert accessor_object.npol == 1
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesI")
    assert accessor_object.nants == 4
    assert accessor_object.nbaselines == 1


def test_qa_flag_table(result_flag_table):
    """
    QualityAssessment of object data values
    are derived correctly.
    """
    expected_data = {
        "maxabs": 1,
        "minabs": 1,
        "mean": 1,
        "sum": 1,
        "medianabs": 1,
    }

    result_qa = result_flag_table.flagtable_acc.qa_flag_table(context="Test")

    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"
