# pylint: disable=no-name-in-module,import-error
# make python-format
# make python lint
"""
Unit tests for Visibility model
"""

import numpy
import pytest
from astropy import constants as const
from astropy.time import Time
from numpy.random import default_rng

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility

#  Create a configuration object

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


@pytest.fixture(scope="module", name="result_visibility")
def fixture_visibility():
    """
    Generate a visibility object using Visibilty.constructor
    """

    frequency = numpy.array([1])
    channel_bandwidth = numpy.array([1])
    phasecentre = (180.0, -35.0)
    uvw = [[[1, 1, 1]]]
    time = numpy.array([1])
    vis = numpy.array([[[[1]]]])
    weight = None
    integration_time = numpy.array([1])
    flags = numpy.array([[[[0]]]])
    baselines = numpy.array([1])
    meta = None
    polarisation_frame = PolarisationFrame("stokesI")
    source = "anonymous"
    low_precision = "float64"
    visibility = Visibility.constructor(
        frequency,
        channel_bandwidth,
        phasecentre,
        CONFIGURATION,
        uvw,
        time,
        vis,
        weight,
        integration_time,
        flags,
        baselines,
        polarisation_frame,
        source,
        meta,
        low_precision,
    )
    return visibility


def test_constructor_coords(result_visibility):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["time", "baselines", "frequency", "polarisation", "spatial"]
    result_coords = result_visibility.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["time"] == 1
    assert result_coords["baselines"] == 1
    assert result_coords["frequency"] == 1
    assert result_coords["polarisation"] == "I"
    assert (result_coords["spatial"] == ["u", "v", "w"]).all()


def test_constructor_data_vars(result_visibility):
    """
    Constructor generates correctly generates data variables
    """

    result_data_vars = result_visibility.data_vars

    assert len(result_data_vars) == 7  # 7 vars in data_vars
    assert result_data_vars["integration_time"] == 1
    assert result_data_vars["datetime"] == Time(1 / 86400.0, format="mjd", scale="utc").datetime64
    assert (result_data_vars["vis"] == 1).all()
    assert (result_data_vars["weight"] == 1).all()
    assert (result_data_vars["flags"] == 0).all()
    assert (result_data_vars["uvw"] == 1).all()
    assert result_data_vars["channel_bandwidth"] == 1


def test_constructor_attrs(result_visibility):
    """
    Constructor correctly generates attributes
    """

    result_attrs = result_visibility.attrs

    assert len(result_attrs) == 6
    assert result_attrs["data_model"] == "Visibility"
    assert result_attrs["configuration"] == CONFIGURATION
    assert result_attrs["source"] == "anonymous"
    assert result_attrs["phasecentre"] == (180.0, -35.0)
    assert result_attrs["_polarisation_frame"] == "stokesI"
    assert result_attrs["meta"] is None


def test_copy(result_visibility):
    """
    Test deep-copying Visibility
    """
    new_vis = result_visibility.copy(deep=True)
    result_visibility["vis"].data[...] = 0.0
    new_vis["vis"].data[...] = 1.0
    assert result_visibility["vis"].data[0, 0].real.all() == 0.0
    result_visibility["vis"].data[...] = 1.0  # reset for following tests
    assert new_vis["vis"].data[0, 0].real.all() == 1.0


def test_property_accessor(result_visibility):
    """
    Visibility.visibility_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_visibility.visibility_acc
    assert accessor_object.rows == range(0, 1)
    assert accessor_object.ntimes == 1
    assert accessor_object.nchan == 1
    assert accessor_object.npol == 1
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesI")
    assert accessor_object.nbaselines == 1
    assert (accessor_object.uvw_lambda == 1/const.c.value).all()
    assert accessor_object.u == 1
    assert accessor_object.v == 1
    assert accessor_object.w == 1
    assert accessor_object.flagged_vis == 1
    assert accessor_object.flagged_weight == 1
    assert accessor_object.nvis == 1


def test_select_uv_range(result_visibility):
    """
    Check that flags are set to 1 if out of the given range
        """

    result_flags = result_visibility.data_vars["flags"]
    uvmin = 2
    uvmax = 100
    assert result_flags.sum() == 0
    result_visibility.visibility_acc.select_uv_range(uvmin, uvmax)
    assert result_flags.sum() == 1


def test_select_r_range_none(result_visibility):
    """
    Check there are no changes to set parameters if rmin and rmax are set to None
    """
    result_range = result_visibility.visibility_acc.select_r_range(None, None)
    expected_sub_bvis = {
        "baselines": 1,
        "frequency": 1,
        "integration_time": 1,
    }
    for key, value in expected_sub_bvis.items():
        assert result_range[key] == value, f"{key} mismatch"

# TODO: add tests for select_r_range "with" a range


def test_group_by_time(result_visibility):
    """
    Check that group_by("time") retunrs the correct array
    """
    times = numpy.array([result[0] for result in result_visibility.groupby("time")])
    assert times.all() == result_visibility.time.all()
# TODO: add more tests for groupby() and groupbybins() functions


def test_performance_visibility(result_visibility):
    """
        Check info about visibility object is correct

        """
    expected_bv_info = {  # except "size"
        "number_times": 1,
        "number_baselines": 1,
        "nchan": 1,
        "npol": 1,
        "polarisation_frame": "stokesI",
        "nvis": 1,
    }
    result_perf = result_visibility.visibility_acc.performance_visibility()
    del result_perf[
        "size"
    ]  # we are not testing the size determined from __sizeof__
    for key, value in expected_bv_info.items():
        assert result_perf[key] == value, f"{key} mismatch"


def test_qa_visibility(result_visibility):
    """
        QualityAssessment of object data values
        are derived correctly.
        """
    expected_data = {
        "maxabs": 1,
        "minabs": 1,
        "rms": 0.0,
        "medianabs": 1,
    }

    result_qa = result_visibility.visibility_acc.qa_visibility(context="Test")

    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"
