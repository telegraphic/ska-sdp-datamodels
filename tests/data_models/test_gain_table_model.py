""" Unit tests for GainTable Models
"""
# make python-format
# make python lint

import numpy
import pytest
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import ReceptorFrame

NAME = "MID"
LOCATION = (5109237.71471275, 2006795.66194638, -3239109.1838011)
NAMES = "M000"
XYZ = 222
MOUNT = "altaz"
FRAME = None
RECEPTOR_FRAME = ReceptorFrame("stokesI")
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


@pytest.fixture(scope="module", name="result_gain_table")
def fixture_gain_table():
    """
    Generate a simple gain table using GainTable.constructor
    """

    gain = numpy.ones((1, 1, 1, 1, 1))
    time = numpy.ones(1)
    interval = numpy.ones(1)
    weight = numpy.ones((1, 1, 1, 1, 1))
    residual = numpy.ones((1, 1, 1, 1))
    frequency = numpy.ones(1)
    phasecentre = (180.0, -35.0)
    jones_type = "T"
    gain_table = GainTable.constructor(
        gain,
        time,
        interval,
        weight,
        residual,
        frequency,
        RECEPTOR_FRAME,
        phasecentre,
        CONFIGURATION,
        jones_type,
    )
    return gain_table


def test_constructor_coords(result_gain_table):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["time", "antenna", "frequency", "receptor1", "receptor2"]
    result_coords = result_gain_table.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["time"] == 1
    assert result_coords["antenna"] == 0
    assert result_coords["frequency"] == 1
    assert result_coords["receptor1"] == "I"
    assert result_coords["receptor2"] == "I"


def test_constructor_datavars(result_gain_table):
    """
    Constructor correctly generates data variables
    """

    result_datavars = result_gain_table.data_vars
    assert len(result_datavars) == 5
    assert (result_datavars["gain"] == 1).all()
    assert (result_datavars["weight"] == 1).all()
    assert (result_datavars["residual"] == 1).all()
    assert result_datavars["interval"] == 1
    # TODO: figure out the time format used for datetime
    # assert result_datavars["datetime"] == 1


def test_constructor_attrs(result_gain_table):
    """
    Constructor correctly generates attributes.
    """
    result_attrs = result_gain_table.attrs

    assert len(result_attrs) == 5
    assert result_attrs["data_model"] == "GainTable"
    assert result_attrs["receptor_frame"] == RECEPTOR_FRAME
    assert result_attrs["phasecentre"] == (180.0, -35.0)
    assert result_attrs["configuration"] == CONFIGURATION
    assert result_attrs["jones_type"] == "T"


def test_copy(result_gain_table):
    """
    Copy accurately copies a flag table
    """
    copied_gt_deep = result_gain_table.copy(True, None, False)
    copied_gt_no_deep = result_gain_table.copy(False, None, False)
    # copied_gt_zero = result_gain_table.copy(True, None, True)

    assert copied_gt_deep == result_gain_table
    assert copied_gt_no_deep == result_gain_table
    # TODO: test when Zero is True (needed data != None)
    # assert copied_gt_zero == result_gain_table


def test_property_accessor(result_gain_table):
    """
    GainTable.gaintable_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_gain_table.gaintable_acc
    assert accessor_object.ntimes == 1
    assert accessor_object.nants == 1
    assert accessor_object.nchan == 1
    assert accessor_object.nrec == 1
    assert accessor_object.receptors == "I"


def test_qa_gain_table(result_gain_table):
    """
    QualityAssessment of object data values
    are derived correctly.
    """
    accessor_object = result_gain_table.gaintable_acc
    expected_data = {
        "shape": (1, 1, 1, 1, 1),
        "maxabs-amp": 1,
        "minabs-amp": 1,
        "rms-amp": 0,
        "medianabs-amp": 1,
        "maxabs-phase": 0,
        "minabs-phase": 0,
        "rms-phase": 0,
        "medianabs-phase": 0,
        "residual": 1,
    }

    result_qa = accessor_object.qa_gain_table(context="Test")

    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"
