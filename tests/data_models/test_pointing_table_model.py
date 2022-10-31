""" Unit tests for PointingTable Models
"""
# make python-format
# make python lint

import numpy
import pytest
from astropy.time import Time

from ska_sdp_datamodels.calibration.calibration_model import PointingTable
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    ReceptorFrame,
)

#  Create a  configuration object

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


@pytest.fixture(scope="module", name="result_pointing_table")
def fixture_pointing_table():
    """
    Generate a simple pointing table using PointingTable.constructor
    """

    pointing = numpy.array([[[[[1, 1]]]]])
    nominal = numpy.array([[[[[1, 1]]]]])
    time = numpy.ones(1)
    interval = numpy.ones(1)
    weight = numpy.array([[[[[1, 1]]]]])
    residual = numpy.array([[[[1, 1]]]])
    frequency = numpy.ones(1)
    pointing_frame = "local"
    pointingcentre = (180.0, -35.0)
    pointing_table = PointingTable.constructor(
        pointing,
        nominal,
        time,
        interval,
        weight,
        residual,
        frequency,
        RECEPTOR_FRAME,
        pointing_frame,
        pointingcentre,
        CONFIGURATION,
    )
    return pointing_table


def test_constructor_coords(result_pointing_table):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = [
        "time",
        "antenna",
        "frequency",
        "receptor",
        "angle",
    ]
    result_coords = result_pointing_table.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["time"] == 1
    assert result_coords["antenna"] == 0
    assert result_coords["frequency"] == 1
    assert result_coords["receptor"] == "I"
    (result_coords["angle"] == ["az", "el"]).all()


def test_constructor_datavars(result_pointing_table):
    """
    Constructor correctly generates data variables
    """

    result_data_vars = result_pointing_table.data_vars
    assert len(result_data_vars) == 6
    assert (result_data_vars["pointing"] == 1).all()
    assert (result_data_vars["nominal"] == 1).all()
    assert (result_data_vars["weight"] == 1).all()
    assert (result_data_vars["residual"] == 1).all()
    assert result_data_vars["interval"] == 1
    assert (
        result_data_vars["datetime"]
        == Time(1 / 86400.0, format="mjd", scale="utc").datetime64
    )


def test_constructor_attrs(result_pointing_table):
    """
    Constructor correctly generates attributes.
    """
    result_attrs = result_pointing_table.attrs

    assert len(result_attrs) == 5
    assert result_attrs["data_model"] == "PointingTable"
    assert result_attrs["receptor_frame"] == RECEPTOR_FRAME
    assert result_attrs["pointing_frame"] == "local"
    assert result_attrs["pointingcentre"] == (180.0, -35.0)
    assert result_attrs["configuration"] == CONFIGURATION


def test_copy(result_pointing_table):
    """
    Copy accurately copies a pointing table
    """
    copied_pt_deep = result_pointing_table.copy(True, None, False)
    copied_pt_no_deep = result_pointing_table.copy(False, None, False)
    # copied_pt_zero = result_gain_table.copy(True, None, True)

    assert copied_pt_deep == result_pointing_table
    assert copied_pt_no_deep == result_pointing_table
    # TODO: test when Zero is True (needed data != None)
    # assert copied_gt_zero == result_pointing_table


def test_property_accessor(result_pointing_table):
    """
    PointingTable.pointingtable_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_pointing_table.pointingtable_acc
    assert accessor_object.nants == 1
    assert accessor_object.nchan == 1
    # TODO: current setup of nrec in calibration_model.py does give access to relevant info
    # assert accessor_object.nrec == 1


def test_qa_pointing_table(result_pointing_table):
    """
    QualityAssessment of object data values
    are derived correctly.
    """
    accessor_object = result_pointing_table.pointingtable_acc
    expected_data = {
        "shape": (1, 1, 1, 1, 2),
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

    result_qa = accessor_object.qa_pointing_table(context="Test")

    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"
