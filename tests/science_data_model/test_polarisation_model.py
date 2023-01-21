# pylint: disable=too-many-locals

"""
Polarisation-type data model unit tests.
"""

import pytest

from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    ReceptorFrame,
)


@pytest.mark.parametrize(
    "frame_type, n_pol",
    [
        ["circular", 4],
        ["circularnp", 2],
        ["linear", 4],
        ["linearnp", 2],
        ["stokesIQUV", 4],
        ["stokesIV", 2],
        ["stokesIQ", 2],
        ["stokesI", 1],
    ],
)
def test_polarisation_frame(frame_type, n_pol):
    """
    PolarisationFrame correctly created from type,
    with correct dimensions
    """
    polarisation_frame = PolarisationFrame(frame_type)
    assert polarisation_frame.type == frame_type
    assert polarisation_frame.npol == n_pol


def test_invalid_polarisation_frame_name():
    """
    Raise ValueError when unknown frame type is given
    """
    with pytest.raises(ValueError):
        PolarisationFrame("circuloid")


def test_polarisation_frame_differences():
    """
    Test that different polarisation frames are indeed
    different
    """
    assert PolarisationFrame("linear") != PolarisationFrame("stokesI")
    assert PolarisationFrame("linear") != PolarisationFrame("circular")
    assert PolarisationFrame("circular") != PolarisationFrame("stokesI")


@pytest.mark.parametrize(
    "frame_type, n_rec", [["linear", 2], ["circular", 2], ["stokesI", 1]]
)
def test_rec_frame(frame_type, n_rec):
    """
    ReceptorFrame correctly created from type,
    with correct dimensions
    """
    rec_frame = ReceptorFrame(frame_type)
    assert rec_frame.nrec == n_rec


def test_invalid_receptor_frame_name():
    """
    Raise ValueError when unknown frame type is given
    """
    with pytest.raises(ValueError):
        ReceptorFrame("circuloid")
