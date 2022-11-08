# pylint: disable=invalid-name,too-many-locals,too-many-arguments

"""
Functions to create gridded visibility models
from Image
"""

import copy

import numpy
from astropy.wcs import WCS

from ska_sdp_datamodels.gridded_visibility.grid_vis_model import (
    ConvolutionFunction,
    GridData,
)


def create_griddata_from_image(im, polarisation_frame=None, ft_types=None):
    """
    Create a GridData from an image

    :param im: Template Image
    :param polarisation_frame: PolarisationFrame
    :param ft_types: grid projection type
                     e.g. ["UU", "VV"], ["RA---SIN", "DEC--SIN"]
    :return: GridData
    """

    if ft_types is None:
        ft_types = ["UU", "VV"]
    nchan, npol, ny, nx = im["pixels"].shape
    gridshape = (nchan, npol, ny, nx)
    data = numpy.zeros(gridshape, dtype="complex")

    wcs = copy.deepcopy(im.image_acc.wcs)
    crval = wcs.wcs.crval
    crpix = wcs.wcs.crpix
    cdelt = wcs.wcs.cdelt
    ctype = wcs.wcs.ctype
    d2r = numpy.pi / 180.0
    cdelt[0] = 1.0 / (nx * cdelt[0] * d2r)
    cdelt[1] = 1.0 / (ny * cdelt[1] * d2r)

    # The negation in the longitude is needed by definition of RA, DEC
    grid_wcs = WCS(naxis=4)
    grid_wcs.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, crpix[2], crpix[3]]
    grid_wcs.wcs.ctype = [ft_types[0], ft_types[1], ctype[2], ctype[3]]
    grid_wcs.wcs.crval = [0.0, 0.0, crval[2], crval[3]]
    grid_wcs.wcs.cdelt = [cdelt[0], cdelt[1], cdelt[2], cdelt[3]]
    grid_wcs.wcs.radesys = "ICRS"
    grid_wcs.wcs.equinox = 2000.0

    if polarisation_frame is None:
        polarisation_frame = im.image_acc.polarisation_frame

    elif not npol == polarisation_frame.npol:
        raise ValueError(
            "Polarisation dimensions of input PolarisationFrame "
            "does not mach that of data polarisation dimensions: "
            f"{polarisation_frame.npol} != {npol}"
        )

    return GridData.constructor(
        data, polarisation_frame=polarisation_frame, grid_wcs=grid_wcs
    )


def create_convolutionfunction_from_image(
    im,
    nw=1,
    wstep=1e15,
    oversampling=8,
    support=16,
    polarisation_frame=None,
):
    """
    Create a convolution function from an image

    The convolution function has axes [chan, pol, z, dy, dx, y, x]
    where z, y, x are spatial axes in either sky or Fourier plane.
    The order in the WCS is reversed so the conv_func_WCS describes
    UU, VV, WW, STOKES, FREQ axes

    The axes UU,VV have the same physical stride as the image.
    The axes DUU, DVV are sub-sampled.

    Convolution function holds the original sky plane
    projection in the projection_wcs.

    :param im: Template Image
    :param nw: Number of z axes, usually z is W
    :param wstep: Step in z, usually z is W
    :param oversampling: Oversampling (size of dy, dx axes)
    :param support: Support of final convolution function (size of y, x axes)
    :param polarisation_frame: PolarisationFrame object
    :return: Convolution Function

    """
    if not len(im["pixels"].data.shape) == 4:
        raise ValueError(
            "Image pixel shape is not 4; shape has to"
            "follow: (nchan, npol, x, y)"
        )

    if (
        not im.image_acc.wcs.wcs.ctype[0] == "RA---SIN"
        or not im.image_acc.wcs.wcs.ctype[1] == "DEC--SIN"
    ):
        raise ValueError(
            "Image WCS projection has be ['RA---SIN', 'DEC--SIN']. "
            f"Instead, it is [{im.image_acc.wcs.wcs.ctype[0]}, "
            f"{im.image_acc.wcs.wcs.ctype[1]}]"
        )

    # Array Coords are [chan, pol, z, dy, dx, y, x]
    # where x, y, z are spatial axes in real space or Fourier space
    nchan, npol, ny, nx = im["pixels"].data.shape

    # WCS Coords are [x, y, dy, dx, z, pol, chan]
    # where x, y, z are spatial axes in real space or Fourier space
    wcs = copy.deepcopy(im.image_acc.wcs.wcs)
    crval = wcs.crval
    crpix = wcs.crpix
    cdelt = wcs.cdelt
    ctype = wcs.ctype
    d2r = numpy.pi / 180.0
    cdelt[0] = 1.0 / (nx * cdelt[0] * d2r)
    cdelt[1] = 1.0 / (ny * cdelt[1] * d2r)

    cf_wcs = WCS(naxis=7)
    cf_wcs.wcs.crpix = [
        float(support // 2) + 1.0,
        float(support // 2) + 1.0,
        float(oversampling // 2) + 1.0,
        float(oversampling // 2) + 1.0,
        float(nw // 2 + 1.0),
        crpix[2],
        crpix[3],
    ]
    cf_wcs.wcs.ctype = ["UU", "VV", "DUU", "DVV", "WW", ctype[2], ctype[3]]
    cf_wcs.wcs.crval = [0.0, 0.0, 0.0, 0.0, 0.0, crval[2], crval[3]]
    cf_wcs.wcs.cdelt = [
        cdelt[0],
        cdelt[1],
        cdelt[0] / oversampling,
        cdelt[1] / oversampling,
        wstep,
        cdelt[2],
        cdelt[3],
    ]

    cf_wcs.wcs.radesys = "ICRS"
    cf_wcs.wcs.equinox = 2000.0

    cf_data = numpy.zeros(
        [nchan, npol, nw, oversampling, oversampling, support, support],
        dtype="complex",
    )

    if polarisation_frame is None:
        polarisation_frame = im.image_acc.polarisation_frame

    return ConvolutionFunction.constructor(
        data=cf_data, cf_wcs=cf_wcs, polarisation_frame=polarisation_frame
    )
