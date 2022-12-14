"""
Tests of functions that create an Image object
"""

import numpy
import pytest

from ska_sdp_datamodels.image.image_create import create_image
from ska_sdp_datamodels.science_data_model import PolarisationFrame

N_PIXELS = 512
CELL_SIZE = 0.000015


def test_create_image(phase_centre):
    """
    Test creating Image with default inputs and
    minimal required arguments.
    """
    result = create_image(N_PIXELS, CELL_SIZE, phase_centre)

    assert result.image_acc.npol == 1  # default PolFrame is StokesI
    # nchan=3 (default), npol=1 (default)
    assert result.image_acc.shape == (3, 1, N_PIXELS, N_PIXELS)
    assert (result["pixels"].data == 0.0).all()
    assert sorted(result.image_acc.wcs.wcs.ctype) == sorted(
        ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    )
    assert (
        result.image_acc.wcs.wcs.crval
        == [phase_centre.ra.deg, phase_centre.dec.deg, 1.0, 1.0e8]
    ).all()
    assert (
        result.image_acc.wcs.wcs.cdelt.round(10)
        == numpy.array([-8.59436693e-04, 8.59436693e-04, 1.0, 1.0e6]).round(10)
    ).all()  # this uses the input cell_size and channel_bandwidth


def test_create_image_phase_centre_none():
    """
    If phasecenter is provided as None,
    raise ValueError.
    """
    with pytest.raises(ValueError):
        create_image(N_PIXELS, CELL_SIZE, None)


def test_creat_image_with_frequency(phase_centre):
    """
    Frequency of Image is determined using all three input
    values: frequency, channel_bandwidth and nchan
    """
    frequency = 1.3e8
    bandwidth = 1.0e7
    nchan = 6
    expected_freq = numpy.arange(
        frequency, nchan * bandwidth + frequency, bandwidth
    )

    result = create_image(
        N_PIXELS,
        CELL_SIZE,
        phase_centre,
        frequency=frequency,
        channel_bandwidth=bandwidth,
        nchan=nchan,
    )

    assert (result.frequency.data == expected_freq).all()


def test_creat_image_with_default_frequency(phase_centre):
    """
    Frequency of Image is determined using all three input
    values: frequency, channel_bandwidth and nchan.

    Testing default values.
    """
    expected_freq = numpy.array([1.0e8, 1.01e8, 1.02e8])

    result = create_image(N_PIXELS, CELL_SIZE, phase_centre)

    assert (result.frequency.data == expected_freq).all()


def test_create_image_with_clean_beam(phase_centre):
    """
    Input clean_beam is correctly added to Image.
    """
    clean_beam = {"bmaj": 0.1, "bmin": 0.05, "bpa": -60.0}
    result = create_image(
        N_PIXELS, CELL_SIZE, phase_centre, clean_beam=clean_beam
    )
    assert result.attrs["clean_beam"] == clean_beam


def test_create_image_with_polarisation_frame(phase_centre):
    """
    Input PolarisationFrame is correctly added to Image.
    """
    pol_frame = PolarisationFrame("linear")
    result = create_image(
        N_PIXELS, CELL_SIZE, phase_centre, polarisation_frame=pol_frame
    )

    assert result.image_acc.npol == 4
    assert (result.coords["polarisation"] == pol_frame.names).all()
