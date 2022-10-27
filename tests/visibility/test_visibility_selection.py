"""
Unit tests for visibility selectors
"""

import logging

import numpy

log = logging.getLogger("data-models-logger")

log.setLevel(logging.WARNING)


def test_visibility_groupby_time(visibility):
    """
    Time groups created by groupby match
    the times of the Visibility object
    """
    times = numpy.array([result[0] for result in visibility.groupby("time")])
    assert (times == visibility.time).all()


def test_visibility_groupby_bins_time(visibility):
    """
    TODO
    """
    for result in visibility.groupby_bins("time", 3):
        log.info(result[0])


def test_visibility_iselect_time(visibility):
    """
    Sub-select Visibility object using Dataset.isel,
    using indexes
    """
    selected_vis = visibility.isel({"time": slice(5, 7)})

    assert len(selected_vis.time) == 2
    assert (selected_vis.time == visibility.time[5:7]).all()
    assert len(selected_vis.channel_bandwidth.shape) == 1
    assert len(selected_vis.integration_time.shape) == 1


def test_visibility_select_time(visibility):
    """
    Sub-select Visibility object by passing a slice
    to Dataset.isel, using indexes
    """
    time_slice = visibility.time.data[1:4]
    selected_vis = visibility.sel({"time": time_slice})

    assert (selected_vis.time == visibility.time[5:7]).all()
    assert len(selected_vis.time) == 3
    assert len(selected_vis.channel_bandwidth.shape) == 1
    assert len(selected_vis.integration_time.shape) == 1


def test_visibility_select_frequency(visibility):
    """
    Sub-select Visibility object using Dataset.sel,
    using actual values to define the range
    """
    selected_vis = visibility.sel({"frequency": slice(0.9e8, 1.2e8)})

    assert selected_vis.frequency.data[0] >= 0.9e8
    assert selected_vis.frequency.data[-1] <= 1.2e8
    assert len(selected_vis.frequency) == 4
    assert len(selected_vis.channel_bandwidth.shape) == 1
    assert len(selected_vis.integration_time.shape) == 1


def test_visibility_select_frequency_polarisation(visibility):
    """
    Sub-select Visibility object based on
    two different parameters' values (freq and pol).
    """
    selected_vis = visibility.sel(
        {"frequency": slice(0.9e8, 1.2e8), "polarisation": ["XX", "YY"]}
    ).dropna(dim="frequency", how="all")

    assert selected_vis.frequency.data[0] >= 0.9e8
    assert selected_vis.frequency.data[-1] <= 1.2e8
    assert (selected_vis.polarisation.data == ["XX", "YY"]).all()
    assert len(selected_vis.frequency) == 4
    assert len(selected_vis.polarisation) == 2
    assert len(selected_vis.channel_bandwidth.shape) == 1
    assert len(selected_vis.integration_time.shape) == 1


def test_visibility_flag_auto(visibility):
    """
    Update flags data based on slice.
    """
    flags_shape = visibility.flags.shape
    original_flags = visibility["flags"].sum()

    uvdist_lambda = numpy.hypot(
        visibility.visibility_acc.uvw_lambda[..., 0],
        visibility.visibility_acc.uvw_lambda[..., 1],
    )

    new_vis = visibility.copy(deep=True)
    new_vis["flags"].data[numpy.where(uvdist_lambda <= 20000.0)] = 1
    assert new_vis.flags.shape == flags_shape

    updated_flags = new_vis["flags"].sum()
    assert updated_flags > original_flags


def test_visibility_select_uvrange(visibility):
    """
    Visibility flags are updated based on selected
    uv-range. Where data are un-selected, flags
    are set to 1.
    """
    new_vis = visibility.copy(deep=True)
    uvmin = 100.0
    uvmax = 20000.0

    assert new_vis["flags"].sum() == 0
    new_vis.visibility_acc.select_uv_range(uvmin, uvmax)
    assert new_vis["flags"].sum() == 1185464
    assert new_vis.frequency.shape == (5,)


def test_visibility_select_r_range(visibility):
    """
    Expected number of baselines was calculated from a
    manual inspection of the configuration file.
    """
    rmin = 100.0
    rmax = 20000.0

    sub_vis = visibility.visibility_acc.select_r_range(rmin, rmax)
    assert len(sub_vis.baselines) == 11781
    assert len(sub_vis.configuration.names) == 166
    assert sub_vis.frequency.shape == (5,)
    assert sub_vis.integration_time.shape == visibility.integration_time.shape
