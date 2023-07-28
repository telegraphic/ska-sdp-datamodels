"""Test functions to test the msgpack encode/decode functions."""

# pylint: disable=duplicate-code

import xarray

from ska_sdp_datamodels.utilities import decode, encode
from ska_sdp_datamodels.visibility import create_visibility

from .visibility.test_vis_create import CHANNEL_BANDWIDTH, FREQUENCY, TIMES


def test_encode_decode(low_aa05_config, phase_centre):
    """Test both encode and decode, but only that they work."""
    vis = create_visibility(
        low_aa05_config,
        TIMES,
        FREQUENCY,
        channel_bandwidth=CHANNEL_BANDWIDTH,
        phasecentre=phase_centre,
    )

    encoded = encode(vis)
    assert isinstance(encoded, bytes)

    decoded = decode(encoded)
    assert isinstance(decoded, xarray.Dataset)
