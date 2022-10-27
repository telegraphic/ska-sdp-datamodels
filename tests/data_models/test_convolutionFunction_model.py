
""" Unit tests for Memory Data Models
"""

import pytest
import numpy
import pandas
from xarray import DataArray
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning

from src.ska_sdp_datamodels.polarisation_data_models import (
    PolarisationFrame,
    ReceptorFrame,
)

from src.ska_sdp_datamodels.memory_data_models import (
    ConvolutionFunction
)
from src.ska_sdp_datamodels.xarray_coordinate_support import (
    conv_func_wcs,
    griddata_wcs,
    image_wcs,
)

N_CHAN = 100
N_POL = 2
NW = 1
oversampling = 1
support = 1
WCS_HEADER = {
    "CTYPE1": "RA---SIN",
    "CTYPE2": "DEC--SIN",
    "CTYPE3": "STOKES",  # no units, so no CUNIT3
    "CTYPE4": "STEP",
    "CUNIT1": "deg",
    "CUNIT2": "deg",
    "CUNIT4": "Hz",
    "CRPIX1": 120,  # CRPIX1-5 are reference pixels
    "CRPIX2": 120,
    "CRPIX3": 1,
    "CRPIX4": 1,
    "CRVAL1": 40.0,  # RA in deg
    "CRVAL2": 0.0,  # DEC in deg
    "CDELT1": -0.1,
    "CDELT2": 0.1,  # abs(CDELT2) = cellsize in deg
    "CDELT3": 3,  # delta between polarisation values (I=0, V=4)
    "CDELT4": 1,  # delta between frequency values
}

@pytest.fixture(scope="module", name="result_convolutionFunction")
def fixture_convolutionFunction():
    """
    Generate a simple image using ConvolutionFunction.constructor.
    """
    data = numpy.ones((N_CHAN, N_POL, NW, oversampling, oversampling , support, support))
    polarisation_frame = PolarisationFrame("stokesIV")
    cf_wcs = WCS(header=WCS_HEADER, naxis=4)
    convolutionFunction = ConvolutionFunction.constructor(data, cf_wcs,
        polarisation_frame)
    return convolutionFunction

def test_constructor_coords(result_convolutionFunction):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["frequency", "polarisation", "dv", "du", "w", "v", "u"]
    result_coords = result_convolutionFunction.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    # assert result_coords["frequency"].shape == (N_CHAN,)
    # assert (result_coords["polarisation"].data == ["I", "V"]).all()
    # assert result_coords["w"].shape == (NW,)

# def test_constructor_data_vars(result_convolutionFunction):
#     """
#     Constructor generates correctly generates data variables
#     """

#     result_data_vars = result_convolutionFunctionv.data_vars

#     assert len(result_data_vars) == 1  # only var is pixels
#     assert "pixels" in result_data_vars.keys()
#     assert isinstance(result_data_vars["pixels"], DataArray)
#     assert (result_data_vars["pixels"].data == 1.0).all()
    

# def test_constructor_attrs(result_convolutionFunction):
#     """
#     Constructor correctly generates attributes
#     """

#     result_attrs = result_convolutionFunction.attrs

#     assert len(result_attrs) == 2
#     assert result_attrs["data_model"] == "ConvolutionFunction"
#     assert result_attrs["_polarisation_frame"] == "stokesIV"

# def test_qa_grid_data(result_convolutionFunction):
#     """
#     QualityAssessment of object data values
#     are derived correctly.

#     Note: input "result_convolutionFunction" contains 1.0 for
#     every pixel value, which is used to determine most
#     of QA data values.
#     """
#     expected_data = {  # except "size"
#         "shape": f"({N_CHAN}, {N_POL}, {NV}, {NU})",
#         "max": 1.0,
#         "min": 1.0,
#         "rms": 0.0,
#         "sum": 26214400.0,
#         "medianabs": 1.0,
#         "median": 1.0,
#     }

#     result_qa = result_convolutionFunction.qa_grid_data(context="Test")

#     assert result_qa.context == "Test"
#     for key, value in expected_data.items():
#         assert result_qa.data[key] == value, f"{key} mismatch"