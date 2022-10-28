"""
Tools to create Image
"""

import logging

import numpy
from astropy.wcs import WCS

from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.science_data_model import PolarisationFrame

log = logging.getLogger("data-models-logger")


# pylint: disable=too-many-arguments
def create_image(
    npixel,
    cellsize,
    phasecentre,
    polarisation_frame=PolarisationFrame("stokesI"),
    frequency=1.0e8,
    channel_bandwidth=1.0e6,
    nchan=3,
    dtype="float",
    clean_beam=None,
) -> Image:
    """
    Create an empty  image consistent with the inputs.

    :param npixel: Number of pixels; e.g. 512
    :param cellsize: cellsize in radians; e.g. 0.000015
    :param phasecentre: phasecentre (SkyCoord)
    :param polarisation_frame: Polarisation frame;
                    default: PolarisationFrame("stokesI")
    :param frequency: Start frequency; default: 1.e8
    :param channel_bandwidth: Channel width (Hz); default: 1.e6
    :param nchan: Number of channels in image
    :param dtype: Python data type for array
    :param clean_beam: dict holding clean beam
                       e.g {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}
    :return: Image
    """
    if phasecentre is None:
        raise ValueError("phasecentre must be specified")

    if polarisation_frame is None:
        log.warning("PolarisationFrame is not specified, using 'stokesI'.")
        polarisation_frame = PolarisationFrame("stokesI")

    npol = polarisation_frame.npol
    pol = PolarisationFrame.fits_codes[polarisation_frame.type]
    if npol > 1:
        dpol = pol[1] - pol[0]
    else:
        dpol = 1.0

    if nchan is None:
        log.warning(
            "Number of channels (nchan) is not provided, using 3 as default"
        )
        nchan = 3

    shape = [nchan, npol, npixel, npixel]

    wcs_obj = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    wcs_obj.wcs.cdelt = [
        -cellsize * 180.0 / numpy.pi,
        cellsize * 180.0 / numpy.pi,
        dpol,
        channel_bandwidth,
    ]
    wcs_obj.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, pol[0], 1.0]
    wcs_obj.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    wcs_obj.wcs.crval = [
        phasecentre.ra.deg,
        phasecentre.dec.deg,
        1.0,
        frequency,
    ]
    wcs_obj.naxis = 4
    wcs_obj.wcs.radesys = "ICRS"
    wcs_obj.wcs.equinox = 2000.0

    return Image.constructor(
        numpy.zeros(shape, dtype=dtype),
        wcs=wcs_obj,
        polarisation_frame=polarisation_frame,
        clean_beam=clean_beam,
    )
