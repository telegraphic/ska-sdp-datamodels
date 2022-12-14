# pylint: disable=fixme

"""
Unit tests to creating Visibilities
"""

import numpy
import pytest

from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import (
    create_flagtable_from_visibility,
    create_visibility,
)

TIMES = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
FREQUENCY = numpy.linspace(0.8e8, 1.2e8, 5)
CHANNEL_BANDWIDTH = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
POLARISATION_FRAME = PolarisationFrame("linear")

# TODO: in the future, add tests for time and flagging related
#  calculations in create_visibility


def test_create_visibility(low_aa05_config, phase_centre):
    """
    Visibility is created with minimum input to function.
    """
    vis = create_visibility(
        low_aa05_config,
        TIMES,
        FREQUENCY,
        channel_bandwidth=CHANNEL_BANDWIDTH,
        phasecentre=phase_centre,
    )
    # shape: (time, baseline, frequency, polarisation)
    # polarisation is 4, because the code defaults to using the frame
    # that matches the configuration receptor frame, which, by default
    # is linear
    assert vis.vis.shape == (len(TIMES), 21, len(FREQUENCY), 4)
    # default polarisation is linear; coming from defailt receptor frame
    # of configuration
    assert (vis.polarisation.data == ["XX", "XY", "YX", "YY"]).all()


def test_create_visibility_no_phase_centre(low_aa05_config):
    """
    Function raises ValueError if
    phasecentre is None.
    """
    with pytest.raises(ValueError) as error:
        create_visibility(
            low_aa05_config,
            TIMES,
            FREQUENCY,
            channel_bandwidth=CHANNEL_BANDWIDTH,
            phasecentre=None,
            weight=1.0,
        )
    assert str(error.value) == "Must specify phase centre"


def test_create_visibility_polarisation(low_aa05_config, phase_centre):
    """
    The created visibility hase the
    specified polarisation frame.
    """
    vis = create_visibility(
        low_aa05_config,
        TIMES,
        FREQUENCY,
        channel_bandwidth=CHANNEL_BANDWIDTH,
        phasecentre=phase_centre,
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    assert vis.vis.shape[-1] == 1  # polarisation dimension is 1
    assert (vis.polarisation.data == ["I"]).all()


def test_create_flagtable_from_visibility(visibility):
    """
    FlagTable is correctly created from input Visibility.
    """
    new_vis = visibility.copy(deep=True)
    new_vis["flags"].data = new_vis["flags"].data + 1
    result = create_flagtable_from_visibility(new_vis)

    assert (result.flags.data == 1).all()
    assert (result.coords["frequency"] == new_vis.frequency).all()
    assert (result.channel_bandwidth == new_vis.channel_bandwidth).all()
    assert result.configuration == new_vis.configuration
    assert (result.coords["time"] == new_vis.time).all()
    assert (result.integration_time == new_vis.integration_time).all()
    assert (
        result.coords["polarisation"]
        == new_vis.visibility_acc.polarisation_frame.names
    ).all()
