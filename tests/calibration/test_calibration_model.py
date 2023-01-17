"""
Unit tests for the Calibration Models
"""

import numpy
import pytest
from astropy.time import Time

from ska_sdp_datamodels.calibration.calibration_model import (
    GainTable,
    PointingTable,
)
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    ReceptorFrame,
)

RECEPTOR_FRAME = ReceptorFrame("stokesI")


@pytest.fixture(scope="module", name="result_gain_table")
def fixture_gain_table(low_aa05_config, phase_centre):
    """
    Generate a simple gain table using GainTable.constructor
    """
    gain = numpy.ones((1, 1, 1, 1, 1))
    time = numpy.ones(1)
    interval = numpy.ones(1)
    weight = numpy.ones((1, 1, 1, 1, 1))
    residual = numpy.ones((1, 1, 1, 1))
    frequency = numpy.ones(1)
    jones_type = "T"
    gain_table = GainTable.constructor(
        gain,
        time,
        interval,
        weight,
        residual,
        frequency,
        (RECEPTOR_FRAME, RECEPTOR_FRAME),
        phase_centre,
        low_aa05_config,
        jones_type,
    )
    return gain_table


@pytest.fixture(scope="module", name="result_pointing_table")
def fixture_pointing_table(low_aa05_config, phase_centre):
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
        phase_centre,
        low_aa05_config,
    )
    return pointing_table


def test_gain_table_constructor_coords(result_gain_table):
    """
    Constructor correctly generates coordinates
    """
    expected_coords_keys = [
        "time",
        "antenna",
        "frequency",
        "receptor1",
        "receptor2",
    ]
    result_coords = result_gain_table.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["time"] == 1
    assert result_coords["antenna"] == 0
    assert result_coords["frequency"] == 1
    assert result_coords["receptor1"] == "I"
    assert result_coords["receptor2"] == "I"


def test_gain_table_constructor_datavars(result_gain_table):
    """
    Constructor correctly generates data variables
    """
    result_data_vars = result_gain_table.data_vars
    assert len(result_data_vars) == 5
    assert (result_data_vars["gain"] == 1).all()
    assert (result_data_vars["weight"] == 1).all()
    assert (result_data_vars["residual"] == 1).all()
    assert result_data_vars["interval"] == 1
    assert (
        result_data_vars["datetime"]
        == Time(1 / 86400.0, format="mjd", scale="utc").datetime64
    )


def test_gain_table_constructor_attrs(
    result_gain_table, low_aa05_config, phase_centre
):
    """
    Constructor correctly generates attributes.
    """
    result_attrs = result_gain_table.attrs

    assert len(result_attrs) == 6
    assert result_attrs["data_model"] == "GainTable"
    assert result_attrs["receptor_frame_in"] == RECEPTOR_FRAME
    assert result_attrs["receptor_frame_out"] == RECEPTOR_FRAME
    assert result_attrs["phasecentre"] == phase_centre
    assert result_attrs["configuration"] == low_aa05_config
    assert result_attrs["jones_type"] == "T"


def test_gain_table_copy(result_gain_table):
    """
    Test deep-copying GainTable
    """
    original_gain = result_gain_table.gain.data
    new_gain_data = result_gain_table.copy(deep=True)
    new_gain_data["gain"].data[...] = 100.0

    assert (result_gain_table["gain"].data == original_gain).all()
    assert (new_gain_data["gain"].data == 100.0).all()


def test_gain_table_property_accessor(result_gain_table):
    """
    GainTable.gaintable_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_gain_table.gaintable_acc
    assert accessor_object.ntimes == 1
    assert accessor_object.nants == 1
    assert accessor_object.nchan == 1
    assert accessor_object.nrec == 1
    assert accessor_object.receptor_in == "I"
    assert accessor_object.receptor_out == "I"


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


def test_pointing_table_constructor_coords(result_pointing_table):
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


def test_pointing_table_constructor_datavars(result_pointing_table):
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


def test_pointing_table_constructor_attrs(
    result_pointing_table, low_aa05_config, phase_centre
):
    """
    Constructor correctly generates attributes.
    """
    result_attrs = result_pointing_table.attrs

    assert len(result_attrs) == 5
    assert result_attrs["data_model"] == "PointingTable"
    assert result_attrs["receptor_frame"] == RECEPTOR_FRAME
    assert result_attrs["pointing_frame"] == "local"
    assert result_attrs["pointingcentre"] == phase_centre
    assert result_attrs["configuration"] == low_aa05_config


def test_pointing_table_copy(result_pointing_table):
    """
    Test deep copy of PointingTable
    """
    original_pointing = result_pointing_table.pointing.data
    new_pointing_data = result_pointing_table.copy(deep=True)
    new_pointing_data["pointing"].data[...] = 100.0

    assert (result_pointing_table["pointing"].data == original_pointing).all()
    assert (new_pointing_data["pointing"].data == 100.0).all()


def test_pointing_table_property_accessor(result_pointing_table):
    """
    PointingTable.pointingtable_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_pointing_table.pointingtable_acc
    assert accessor_object.nants == 1
    assert accessor_object.nchan == 1
    assert accessor_object.nrec == 1


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
