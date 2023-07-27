"""
Unit tests for the Visibility model
"""

import numpy
import pytest
from numpy.testing import assert_almost_equal

from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from ska_sdp_datamodels.visibility.vis_model import FlagTable, Visibility


@pytest.fixture(scope="module", name="result_visibility")
def fixture_visibility(low_aa05_config, phase_centre):
    """
    Generate a visibility object using Visibility.constructor
    """
    frequency = numpy.array([1.0e8, 1.01e8])
    channel_bandwidth = numpy.array([1.0e6, 1.0e6])
    uvw = numpy.array(
        [[[1.1, 1.2, 1.3]], [[1.1, 1.2, 1.3]], [[1.1, 1.2, 1.3]]]
    )
    time = numpy.array([10.0, 20.0, 30.0])
    vis = numpy.ones(shape=(len(time), 1, len(frequency), 1))
    integration_time = numpy.array([10.0, 10.0, 10.0])
    flags = numpy.zeros(shape=vis.shape)
    baselines = numpy.array([1])
    polarisation_frame = PolarisationFrame("stokesI")
    source = "anonymous"
    low_precision = "float64"
    scan_id = 1

    visibility = Visibility.constructor(
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        phasecentre=phase_centre,
        configuration=low_aa05_config,
        uvw=uvw,
        time=time,
        vis=vis,
        weight=None,
        integration_time=integration_time,
        flags=flags,
        baselines=baselines,
        polarisation_frame=polarisation_frame,
        source=source,
        scan_id=scan_id,
        meta=None,
        low_precision=low_precision,
    )
    return visibility


def test_visibility_constructor_coords(result_visibility):
    """
    Constructor correctly generates coordinates
    """
    expected_coords_keys = [
        "time",
        "baselines",
        "frequency",
        "polarisation",
        "spatial",
    ]
    result_coords = result_visibility.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert (result_coords["time"] == [10.0, 20.0, 30.0]).all()
    assert (result_coords["baselines"] == 1).all()
    assert (result_coords["frequency"] == [1.0e8, 1.01e8]).all()
    assert result_coords["polarisation"] == "I"
    assert (result_coords["spatial"] == ["u", "v", "w"]).all()


def test_visibility_constructor_data_vars(result_visibility):
    """
    Constructor correctly generates data variables
    """
    result_data_vars = result_visibility.data_vars

    assert len(result_data_vars) == 7  # 7 vars in data_vars
    assert (result_data_vars["integration_time"] == [10.0, 10.0, 10.0]).all()
    assert (
        result_data_vars["datetime"]
        == numpy.array(
            [
                "1858-11-17T00:00:10.000000000",
                "1858-11-17T00:00:20.000000000",
                "1858-11-17T00:00:30.000000000",
            ],
            dtype="datetime64",
        )
    ).all()
    assert (result_data_vars["vis"] == 1).all()
    assert (result_data_vars["weight"] == 1).all()
    assert (result_data_vars["flags"] == 0).all()
    assert (result_data_vars["uvw"] == [1.1, 1.2, 1.3]).all()
    assert (result_data_vars["channel_bandwidth"] == [1.0e6, 1.0e6]).all()


def test_visibility_constructor_attrs(
    result_visibility, low_aa05_config, phase_centre
):
    """
    Constructor correctly generates attributes
    """
    result_attrs = result_visibility.attrs

    assert len(result_attrs) == 7
    assert result_attrs["data_model"] == "Visibility"
    assert result_attrs["configuration"] == low_aa05_config
    assert result_attrs["source"] == "anonymous"
    assert result_attrs["phasecentre"] == phase_centre
    assert result_attrs["_polarisation_frame"] == "stokesI"
    assert result_attrs["meta"] is None
    assert result_attrs["scan_id"] == 1


def test_visibility_copy(result_visibility):
    """
    Test deep-copying Visibility
    """
    original_visibility = result_visibility.vis.data
    new_vis = result_visibility.copy(deep=True)
    new_vis["vis"].data[...] = 100.0

    # make sure we don't accidentally update the copied
    # vis to the same original one
    assert (original_visibility != 100.0).all()

    assert (result_visibility["vis"].data == original_visibility).all()
    assert (new_vis["vis"].data == 100.0).all()


def test_visibility_property_accessor(result_visibility):
    """
    Visibility.visibility_acc (xarray accessor) returns
    properties correctly.
    """
    expected_uvw_lambda_per_time = [
        [
            [0.3669205, 0.40027691, 0.43363332],
            [0.37058971, 0.40427968, 0.43796966],
        ]
    ]
    accessor_object = result_visibility.visibility_acc
    assert accessor_object.rows == range(0, 3)  # == number of times
    assert accessor_object.ntimes == 3
    assert accessor_object.nchan == 2
    assert accessor_object.npol == 1
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesI")
    assert accessor_object.nbaselines == 1
    assert (accessor_object.u.data == [1.1, 1.1, 1.1]).all()
    assert (accessor_object.v.data == [1.2, 1.2, 1.2]).all()
    assert (accessor_object.w.data == [1.3, 1.3, 1.3]).all()
    assert (accessor_object.flagged_vis == 1).all()
    assert (accessor_object.flagged_weight == 1).all()
    assert accessor_object.nvis == 6

    for i in range(3):
        # three time samples
        assert_almost_equal(
            accessor_object.uvw_lambda[i], expected_uvw_lambda_per_time
        )


def test_visibility_select_uv_range(result_visibility):
    """
    Check that flags are set to 1 if out of the given range
    """
    result_flags = result_visibility.data_vars["flags"]
    uvmin = 2
    uvmax = 100
    assert result_flags.sum() == 0
    result_visibility.visibility_acc.select_uv_range(uvmin, uvmax)
    assert result_flags.sum() == 6  # all of the visibilities are flagged


def test_visibility_select_r_range_none(result_visibility):
    """
    Check no changes to set parameters if rmin and rmax are set to None
    """
    result_range = result_visibility.visibility_acc.select_r_range(None, None)
    expected_sub_bvis = {
        "baselines": result_visibility.coords["baselines"],
        "frequency": result_visibility.coords["frequency"],
        "integration_time": result_visibility.data_vars["integration_time"],
    }
    for key, value in expected_sub_bvis.items():
        assert (result_range[key] == value).all(), f"{key} mismatch"


def test_visibility_group_by_time(result_visibility):
    """
    Check that group_by("time") returns the correct array
    """
    times = numpy.array(
        [result[0] for result in result_visibility.groupby("time")]
    )
    assert times.all() == result_visibility.time.all()


def test_visibility_performance_visibility(result_visibility):
    """
    Check info about visibility object is correct
    """
    expected_bv_info = {  # except "size"
        "number_times": 3,
        "number_baselines": 1,
        "nchan": 2,
        "npol": 1,
        "polarisation_frame": "stokesI",
        "nvis": 6,
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


@pytest.fixture(scope="module", name="result_flag_table")
def fixture_flag_table(low_aa05_config):
    """
    Generate a simple flag table using FlagTable.constructor.
    """
    baselines = numpy.ones(1)
    # flags shape: time, baselines, frequency, polarisation
    flags = numpy.ones((5, 1, 3, 1))
    frequency = numpy.ones(3)
    channel_bandwidth = numpy.ones(3)
    time = numpy.ones(5)
    integration_time = numpy.ones(5)
    polarisation_frame = PolarisationFrame("stokesI")

    flag_table = FlagTable.constructor(
        baselines,
        flags,
        frequency,
        channel_bandwidth,
        low_aa05_config,
        time,
        integration_time,
        polarisation_frame,
    )
    return flag_table


def test_flag_table_constructor_coords(result_flag_table):
    """
    Constructor correctly generates coordinates
    """
    expected_coords_keys = ["time", "baselines", "frequency", "polarisation"]
    result_coords = result_flag_table.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert (result_coords["time"] == 1).all()
    assert (result_coords["baselines"] == 1).all()
    assert (result_coords["frequency"] == 1).all()
    assert (result_coords["polarisation"] == "I").all()


def test_flag_table_constructor_data_vars(result_flag_table):
    """
    Constructor correctly generates data variables
    """
    result_data_vars = result_flag_table.data_vars
    assert len(result_data_vars) == 4
    assert (result_data_vars["flags"] == 1).all()
    assert (result_data_vars["integration_time"] == 1).all()
    assert (result_data_vars["channel_bandwidth"] == 1).all()
    assert (
        result_data_vars["datetime"]
        == numpy.array(
            ["1858-11-17T00:00:01.000000000"] * 5, dtype="datetime64"
        )
    ).all()


def test_flag_table_constructor_attrs(result_flag_table, low_aa05_config):
    """
    Constructor correctly generates attributes.
    """
    result_attrs = result_flag_table.attrs

    assert len(result_attrs) == 3
    assert result_attrs["data_model"] == "FlagTable"
    assert result_attrs["_polarisation_frame"] == "stokesI"
    assert result_attrs["configuration"] == low_aa05_config


def test_flag_table_copy(result_flag_table):
    """
    Test deep-copying Visibility
    """
    original_flags = result_flag_table.flags.data
    new_flag = result_flag_table.copy(deep=True)
    # make sure we don't set new flags to what the originals were
    assert (original_flags != 0).all()

    new_flag["flags"].data[...] = 0
    assert (result_flag_table["flags"].data == original_flags).all()
    assert (new_flag["flags"].data == 0).all()


def test_flag_table_property_accessor(result_flag_table):
    """
    FlagTable.flagtable_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_flag_table.flagtable_acc
    assert accessor_object.nchan == 3
    assert accessor_object.npol == 1
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesI")
    assert accessor_object.nants == 6  # from configuration
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
        "sum": 15,
        "medianabs": 1,
    }

    result_qa = result_flag_table.flagtable_acc.qa_flag_table(context="Test")

    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"


def test_flagtable_groupby_time(result_flag_table, visibility):
    """
    Test FlagTable groupby.
    """
    times = numpy.array(
        [result[0] for result in result_flag_table.groupby("time")]
    )
    assert times.all() == visibility.time.all()


def test_flagtable_select_time(flag_table):
    """
    Test FlagTable select by.
    """
    times = flag_table.time
    selected_ft = flag_table.sel({"time": slice(times[1], times[2])})
    assert len(selected_ft.time) == 2
