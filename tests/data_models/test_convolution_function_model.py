""" Unit tests for the Convolution Function model
"""
# make python-format
# make python lint

import numpy
import pytest
from astropy.wcs import WCS
from ska_sdp_datamodels.gridded_visibility.grid_vis_model import ConvolutionFunction
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from xarray import DataArray

N_CHAN = 100
N_POL = 2
NW = 4
OVERSAMPLING = 1
SUPPORT = 3
WCS_HEADER = {
    "CTYPE1": "UU",
    "CTYPE2": "VV",
    "CTYPE3": "DUU",
    "CTYPE4": "DVV",
    "CTYPE5": "WW",
    "CTYPE6": "STOKES",  # no units, so no CUNIT6
    "CTYPE7": "FREQ",
    "CRPIX1": 2,  # CRPIX1-7 are reference pixels
    "CRPIX2": 2,
    "CRPIX3": 1,
    "CRPIX4": 1,
    "CRPIX5": 3,
    "CRPIX6": 1,
    "CRPIX7": 1,
    # TODO: find why these values seem to differ by 0.05 with every test (i.e. never match)
    # "CRVAL1": 40.1,  # UU in deg
    # "CRVAL2": 1.0,  # VV in deg
    # "CRVAL5": 0.0,  # WW in deg
    "CDELT1": -0.1,
    "CDELT2": 0.1,  # abs(CDELT2) = cellsize in deg
    "CDELT3": 1.0,
    "CDELT4": 1.0,
    "CDELT5": 1.0,
    "CDELT6": 3,  # delta between polarisation values (I=0, V=4)
    "CDELT7": 1,  # delta between frequency values
}


@pytest.fixture(scope="module", name="result_convolution_function")
def fixture_convolution_function():
    """
    Generate a simple image using ConvolutionFunction.constructor.
    """
    data = numpy.ones((N_CHAN, N_POL, NW, OVERSAMPLING, OVERSAMPLING, SUPPORT, SUPPORT))
    polarisation_frame = PolarisationFrame("stokesIV")
    cf_wcs = WCS(header=WCS_HEADER, naxis=7)
    convolution_function = ConvolutionFunction.constructor(
        data, cf_wcs, polarisation_frame
    )
    return convolution_function


def test_constructor_coords(result_convolution_function):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["frequency", "polarisation", "dv", "du", "w", "v", "u"]
    result_coords = result_convolution_function.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["frequency"].shape == (N_CHAN,)
    assert (result_coords["polarisation"].data == ["I", "V"]).all()
    assert result_coords["w"].shape == (NW,)


def test_constructor_data_vars(result_convolution_function):
    """
    Constructor generates correctly generates data variables
    """

    result_data_vars = result_convolution_function.data_vars

    assert len(result_data_vars) == 1  # only var is pixels
    assert "pixels" in result_data_vars.keys()
    assert isinstance(result_data_vars["pixels"], DataArray)
    assert (result_data_vars["pixels"].data == 1.0).all()


def test_constructor_attrs(result_convolution_function):
    """
    Constructor correctly generates attributes
    """

    result_attrs = result_convolution_function.attrs

    assert len(result_attrs) == 2
    assert result_attrs["data_model"] == "ConvolutionFunction"
    assert result_attrs["_polarisation_frame"] == "stokesIV"


def test_qa_convolution_function(result_convolution_function):
    """
    QualityAssessment of object data values
    are derived correctly.
    """
    expected_data = {
        "shape": f"({N_CHAN}, {N_POL}, {NW}, {OVERSAMPLING}, {OVERSAMPLING}, {SUPPORT}, {SUPPORT})",
        "max": 1.0,
        "min": 1.0,
        "rms": 0.0,
        "sum": 7200.0,
        "medianabs": 1.0,
        "median": 1.0,
    }
    result_qa = (
        result_convolution_function.convolutionfunction_acc.qa_convolution_function(
            context="Test"
        )
    )
    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"


def test_property_accessor(result_convolution_function):
    """
    ConvolutionFunction.convolutionfunction_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_convolution_function.convolutionfunction_acc

    assert accessor_object.nchan == N_CHAN
    assert accessor_object.npol == N_POL
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesIV")
    assert accessor_object.shape == (
        N_CHAN,
        N_POL,
        NW,
        OVERSAMPLING,
        OVERSAMPLING,
        SUPPORT,
        SUPPORT,
    )
    for key, value in WCS_HEADER.items():
        assert accessor_object.cf_wcs.to_header()[key] == value, f"{key} mismatch"
