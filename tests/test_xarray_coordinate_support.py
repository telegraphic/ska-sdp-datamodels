"""
Unit tests for xarray coordinate support functions
"""

from ska_sdp_datamodels.xarray_coordinate_support import (
    conv_func_wcs,
    griddata_wcs,
    image_wcs,
)


def test_image_wcs(image):
    """
    Function creates image WCS from Image object correctly
    """
    result = image_wcs(image)
    assert sorted(result.wcs.ctype) == sorted(
        ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    )
    assert (result.wcs.crval == [180.0, -35.0, 1.0, 100000000.0]).all()
    assert (result.wcs.crpix == [129.0, 129.0, 1.0, 1.0]).all()


def test_griddata_wcs(grid_data):
    """
    Function creates grid data WCS from GridData object correctly
    """
    result = griddata_wcs(grid_data)
    assert sorted(result.wcs.ctype) == sorted(["UU", "VV", "STOKES", "FREQ"])
    assert (result.wcs.crval == [0.0, 0.0, 1.0, 100000000.0]).all()
    assert (result.wcs.crpix == [129.0, 129.0, 1.0, 1.0]).all()


def test_conv_func_wcs(conv_func):
    """
    Function creates convolution function WCS from
    ConvolutionFunction object correctly
    """
    result = conv_func_wcs(conv_func)
    assert sorted(result.wcs.ctype) == sorted(
        ["UU", "VV", "DUU", "DVV", "WW", "STOKES", "FREQ"]
    )
    assert (
        result.wcs.cdelt
        == [
            -260.41666666663866,
            260.41666666666134,
            -32.55208333332983,
            32.55208333333267,
            1000000000000000.0,
            1.0,
            1000000.0,
        ]
    ).all()
