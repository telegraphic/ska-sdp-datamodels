# pylint: disable=no-name-in-module,import-error
# make python-format
# make python lint
"""
Unit tests for Visibility model
"""

import numpy
import pytest

from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.visibility.vis_model import Visibility

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
    Generate a simple gain table using Visibilty.constructor
    """

    frequency = numpy.ones(1)
    channel_bandwidth = numpy.ones(1)
    phasecentre = (180.0, -35.0)
    uvw = numpy.ones((1, 1, 1))
    time = numpy.ones(1)
    vis = numpy.ones((1, 1, 1, 1))
    weight = None
    integration_time = numpy.ones(1)
    flags = numpy.ones((1, 1, 1, 1))
    baselines = numpy.ones(1)
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
    assert (result_coords["spatial"] == [{"u", "v", "w"}]).all()


def test_constructor_data_vars(result_visibility):
    """
    Constructor generates correctly generates data variables
    """

    result_data_vars = result_visibility.data_vars

    assert len(result_data_vars) == 7  # 7 vars in data_vars
    assert result_data_vars["integration_time"] == 1
    # assert result_data_vars["datetime"] == 1
    assert (result_data_vars["vis"] == 1).all()
    assert (result_data_vars["weight"] == 1).all()
    assert (result_data_vars["flags"] == 1).all()
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
    Copy accurately copies a visibility
    """
    copied_vis_deep = result_visibility.copy(True, None, False)
    copied_vis_no_deep = result_visibility.copy(False, None, False)
    # copied_ft_zero = result_flag_table.copy(True, None, True)
    assert copied_vis_deep == result_visibility
    assert copied_vis_no_deep == result_visibility
    # TODO: test when Zero is True (needed data != None)
    # assert copied_ft_zero == result_flag_table


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
    # assert accessor_object.uvw_lambda == 1
    # assert accessor_object.u == 1
    # assert accessor_object.v == 1
    # assert accessor_object.w == 1
    # assert accessor_object.flagged_vis == 1
    # assert accessor_object.flagged_weight == 1
    # assert accessor_object.flagged_imaging_weight == 1
    # assert accessor_object.nvis == 1
