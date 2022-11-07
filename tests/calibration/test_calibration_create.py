"""
Unit tests for functions that
create calibration model objects.
"""
import numpy
import pytest
from numpy.testing import assert_almost_equal

from ska_sdp_datamodels.calibration import (
    create_gaintable_from_visibility,
    create_pointingtable_from_visibility,
)


def test_create_gaintable_from_visibility(visibility):
    """
    GainTable correctly created with default
    function arguments.
    """
    result = create_gaintable_from_visibility(visibility)

    assert (result.coords["time"] == visibility.time).all()
    assert (result.coords["frequency"] == 1.0e8).all()
    assert (result.gain.data[..., 0, 1] == 0.0).all()
    assert (result.gain.data[..., 1, 0] == 0.0).all()
    assert (result.gain.data[..., 0, 0] == 1.0).all()
    assert (result.gain.data[..., 1:, 1:] == 1.0).all()


@pytest.mark.parametrize(
    "jones_type, expected_freq",
    [("B", [8.0e7, 9.0e7, 1.0e8, 1.1e8, 1.2e8]), ("T", 1.0e8), ("G", 1.0e8)],
)
def test_create_gaintable_from_visibility_jones_type(
    jones_type, expected_freq, visibility
):
    """
    GainTable frequency correctly determined using
    input jones_type.
    """
    result = create_gaintable_from_visibility(
        visibility, jones_type=jones_type
    )

    assert (result.coords["frequency"] == expected_freq).all()
    assert result.attrs["jones_type"] == jones_type


def test_create_gaintable_from_visibility_timeslice(visibility):
    """
    GainTable time correctly determined using
    input timeslice.
    """
    time_slice = 100  # seconds
    expected_times = numpy.array([4.45347902, 4.45347913, 4.45347922])
    result = create_gaintable_from_visibility(visibility, timeslice=time_slice)

    assert_almost_equal(result.coords["time"].data / 1.0e9, expected_times)


def test_create_pointingtable_from_visibility(visibility):
    """
    PointingTable correctly created with default
    function arguments.
    """
    result = create_pointingtable_from_visibility(visibility)

    assert (result.coords["time"] == visibility.time).all()
    assert (result.pointing.data[..., 0, :] == 0.0).all()
    assert (result.pointing.data[..., 1, :] == 0.0).all()
    assert (result.pointing.data[..., 2:, :] == 1.0).all()
    assert result.attrs["pointing_frame"] == "azel"


def test_create_pointingtable_from_visibility_timeslice(visibility):
    """
    PointingTable time correctly determined using
    input timeslice.
    """
    time_slice = 100  # seconds
    expected_times = numpy.array([4.45347902, 4.45347913, 4.45347922])
    result = create_gaintable_from_visibility(visibility, timeslice=time_slice)

    assert_almost_equal(result.coords["time"].data / 1.0e9, expected_times)
