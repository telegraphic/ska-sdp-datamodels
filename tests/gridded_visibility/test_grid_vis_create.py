"""
Unit tests of functions that create
gridded visibility models
"""
import numpy
import pytest

from ska_sdp_datamodels.gridded_visibility import (
    create_convolutionfunction_from_image,
    create_griddata_from_image,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame


def test_create_griddata_from_image(image):
    """
    GridData correctly created with default
    function arguments.
    """
    result = create_griddata_from_image(image)

    assert (
        result.coords["frequency"] == numpy.array([1.00e08, 1.01e08, 1.02e08])
    ).all()
    assert (
        result.coords["polarisation"]
        == image.image_acc.polarisation_frame.names
    ).all()
    assert (result.pixels.data == 0.0 + 0.0j).all()
    assert result.pixels.data.shape == image["pixels"].shape


def test_create_griddata_from_image_pol_frame(image):
    """
    GridData created with input polarisation frame.
    """

    pol_frame = PolarisationFrame("circular")
    result = create_griddata_from_image(image, polarisation_frame=pol_frame)
    assert (result.coords["polarisation"] == ["RR", "RL", "LR", "LL"]).all()


def test_create_griddata_from_image_pol_frame_wrong(image):
    """
    Wrong dimensions of polarisation frame raises ValueError
    """

    pol_frame = PolarisationFrame("stokesI")
    with pytest.raises(ValueError) as error:
        create_griddata_from_image(image, polarisation_frame=pol_frame)

    assert (
        str(error.value)
        == "Polarisation dimensions of input PolarisationFrame "
        "does not mach that of data polarisation dimensions: 1 != 4"
    )


# pylint: disable=invalid-name
@pytest.mark.parametrize(
    "nw, oversampling, support, wstep",
    [(1, 8, 16, 1e15), (2, 4, 32, 1.0e13)],  # function defaults
)
def test_create_convolutionfunction_from_image(
    nw, oversampling, support, wstep, image
):
    """
    ConvolutionFunction correctly created
    using input arguments.
    """
    expected_data_shape = (
        image.image_acc.nchan,
        image.image_acc.npol,
        nw,
        oversampling,
        oversampling,
        support,
        support,
    )

    result = create_convolutionfunction_from_image(
        image, nw=nw, oversampling=oversampling, support=support, wstep=wstep
    )

    assert (
        result.coords["polarisation"]
        == image.image_acc.polarisation_frame.names
    ).all()
    assert (result.pixels.data == 0.0 + 0.0j).all()
    assert result.pixels.data.shape == expected_data_shape
    assert result.convolutionfunction_acc.cf_wcs.wcs.ctype[4] == "WW"
    assert result.convolutionfunction_acc.cf_wcs.wcs.cdelt[4] == wstep
