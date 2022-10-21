"""
Unit tests for Image()
"""

import numpy
import pytest
from astropy.wcs import WCS
from xarray import DataArray

from ska_sdp_datamodels.image_model import Image
from ska_sdp_datamodels.polarisation_data_models import PolarisationFrame

N_CHAN = 100
N_POL = 2
Y = 512
X = 256

WCS_HEADER = {
    "CTYPE1": "RA---SIN",
    "CTYPE2": "DEC--SIN",
    "CTYPE3": "STOKES",
    "CTYPE4": "FREQ",
    "CUNIT1": "deg",
    "CUNIT2": "deg",
    "CUNIT4": "Hz",
    "CRPIX1": 1,
    "CRPIX2": 1,
    "CRPIX3": 1,
    "CRPIX4": 1,
    "CRVAL1": 40.0,
    "CRVAL2": 0.0,
    "CRVAL3": 80.0,
    "CRVAL4": 20.0,
    "CDELT1": -0.1,
    "CDELT2": 0.1,
    "CDELT3": -0.1,
    "CDELT4": 0.1,
}


@pytest.fixture(scope="module", name="result_image")
def fixture_image():
    """
    Generate a simple image using Image.constructor
    """
    data = numpy.ones((N_CHAN, N_POL, Y, X))
    pol_frame = PolarisationFrame("stokesIV")
    wcs = WCS(naxis=4)
    image = Image.constructor(data, pol_frame, wcs, clean_beam=None)
    return image


def test_constructor_coords(result_image):
    """
    Constructor correctly generates coordinates
    from simple wcs input
    """
    expected_coords_keys = ["polarisation", "frequency", "x", "y"]

    result_coords = result_image.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["frequency"].shape == (N_CHAN,)
    assert (result_coords["polarisation"].data == ["I", "V"]).all()
    assert result_coords["x"].shape == (X,)
    assert result_coords["y"].shape == (Y,)
    assert result_coords["x"][0] != result_coords["x"][-1]
    assert result_coords["y"][0] != result_coords["y"][-1]


def test_constructor_attrs(result_image):
    """
    Constructor correctly generates attributes
    """
    result_attrs = result_image.attrs

    assert result_attrs["data_model"] == "Image"
    assert result_attrs["_polarisation_frame"] == "stokesIV"
    assert result_attrs["clean_beam"] is None
    assert result_attrs["channel_bandwidth"] == 1.0
    assert result_attrs["ra"] == 0.0
    assert result_attrs["dec"] == 0.0


def test_constructor_data_vars(result_image):
    """
    Constructor correctly generates data variables
    """
    result_data_vars = result_image.data_vars

    assert len(result_data_vars) == 1  # only var is pixels
    assert "pixels" in result_data_vars.keys()
    assert isinstance(result_data_vars["pixels"], DataArray)
    assert (result_data_vars["pixels"].data == 1.0).all()


def test_constructor_attr_ra_dec():
    """
    Test realistic RA, DEC calculations,
    or at least that they're affected by input
    """


def test_constructor_attr_clean_beam():
    """
    Test clean_beam attribute when clean beam is
    provided as input
    """


def test_constructor_freq_chan_bandwidth():
    """
    Test frequency and channel bandwidth changes
    based on input.
    """


def test_is_canonical_true():
    """
    WCS Header contains canonical information
    --> image is canonical
    """
    data = numpy.ones((N_CHAN, N_POL, Y, X))
    pol_frame = PolarisationFrame("stokesIV")
    wcs = WCS(header=WCS_HEADER, naxis=4)
    image = Image.constructor(data, pol_frame, wcs, clean_beam=None)

    assert image.is_canonical() is True


def test_is_canonical_false():
    """
    WCS Header contains non-canonical information
    --> image is not canonical
    """
    new_header = WCS_HEADER.copy()
    new_header["CTYPE1"] = "GLON-CAR"
    new_header["CTYPE2"] = "GLAT-CAR"
    del new_header["CTYPE3"]
    del new_header["CTYPE4"]
    del new_header["CUNIT4"]

    data = numpy.ones((N_CHAN, N_POL, Y, X))
    pol_frame = PolarisationFrame("stokesIV")
    wcs = WCS(header=new_header, naxis=4)
    image = Image.constructor(data, pol_frame, wcs, clean_beam=None)

    assert image.is_canonical() is False
