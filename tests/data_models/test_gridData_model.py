""" Unit tests for Memory Data Models
"""

import pytest
import numpy
import pandas
from xarray import DataArray
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyDeprecationWarning

from src.ska_sdp_datamodels.polarisation_data_models import (
    PolarisationFrame,
)

from src.ska_sdp_datamodels.memory_data_models import (
    GridData
)

N_CHAN = 100
N_POL = 2
NV = 256
NU = 256
WCS_HEADER = {
    "CTYPE1": "UU",
    "CTYPE2": "VV",
    "CTYPE3": "STOKES",  # no units, so no CUNIT3
    "CTYPE4": "FREQ",
    "CRVAL1": 40.0,  # RA in deg
    "CRVAL2": 0.0,  # DEC in deg
    "CDELT1": -0.1,
    "CDELT2": 0.1,  # abs(CDELT2) = cellsize in deg
    "CDELT3": 3,  # delta between polarisation values (I=0, V=4)
    "CDELT4": 10.0,  # delta between channel_bandwidth values
}

@pytest.fixture(scope="module", name="result_gridData")
def fixture_griddata():
    """
    Generate a simple image using GridData.constructor.
    """
    data = numpy.ones((N_CHAN, N_POL, NV, NU))
    pol_frame = PolarisationFrame("stokesIV")
    grid_wcs = WCS(header=WCS_HEADER, naxis=4)
    gridData = GridData.constructor(data, pol_frame, grid_wcs)
    return gridData

def test_GridData_constructor_coords(result_gridData):
    """
    Constructor correctly generates coordinates
    """

    expected_coords_keys = ["frequency", "polarisation", "v", "u"]
    result_coords = result_gridData.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["frequency"].shape == (N_CHAN,)
    assert (result_coords["polarisation"].data == ["I", "V"]).all()
    assert result_coords["v"].shape == (NV,)
    assert result_coords["u"].shape == (NU,)
    assert result_coords["v"][0] != result_coords["v"][-1]
    assert result_coords["u"][0] != result_coords["u"][-1]

def test_GridData_constructor_data_vars(result_gridData):
    """
    Constructor generates correctly generates data variables
    """

    result_data_vars = result_gridData.data_vars

    assert len(result_data_vars) == 1  # only var is pixels
    assert "pixels" in result_data_vars.keys()
    assert isinstance(result_data_vars["pixels"], DataArray)
    assert (result_data_vars["pixels"].data == 1.0).all()
    

def test_GridData_constructor_attrs(result_gridData):
    """
    Constructor correctly generates attributes
    """

    result_attrs = result_gridData.attrs

    assert len(result_attrs) == 2
    assert result_attrs["data_model"] == "GridData"
    assert result_attrs["_polarisation_frame"] == "stokesIV"

def test_qa_grid_data(result_gridData):
    """
    QualityAssessment of object data values
    are derived correctly.

    Note: input "result_image" contains 1.0 for
    every pixel value, which is used to determine most
    of QA data values.
    """
    expected_data = {  # except "size"
        "shape": f"({N_CHAN}, {N_POL}, {NV}, {NU})",
        "max": 1.0,
        "min": 1.0,
        "rms": 0.0,
        "sum": 13107200.0,
        "medianabs": 1.0,
        "median": 1.0,
    }

    result_qa = result_gridData.qa_grid_data(context="Test")

    assert result_qa.context == "Test"
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"

def test_property_accessor(result_gridData):
    """
    GridData.image_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_gridData.griddata_acc

    assert accessor_object.nchan == 100
    assert accessor_object.npol == 2
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesIV")
    assert accessor_object.shape == (N_CHAN, N_POL, NV, NU)
    for key, value in WCS_HEADER.items():
        assert accessor_object.griddata_wcs.to_header()[key] == value, f"{key} mismatch"
