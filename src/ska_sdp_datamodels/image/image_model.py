# pylint: disable=too-many-ancestors,too-many-locals,invalid-name
# pylint: disable=too-many-arguments

"""
Image and FlagTable data models.
"""

import logging

import numpy
import xarray
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    QualityAssessment,
)
from ska_sdp_datamodels.xarray_accessor import XarrayAccessorMixin
from ska_sdp_datamodels.xarray_coordinate_support import image_wcs

log = logging.getLogger("data-models-logger")


class Image(xarray.Dataset):
    """
    Image class with pixels as an
    xarray.DataArray and the AstroPy`implementation of
    a World Coordinate System
    <http://docs.astropy.org/en/stable/wcs>`_

    The actual image values are kept in a data_var
    of the xarray.Dataset called "pixels".

    Many operations can be done conveniently using
    xarray processing_components on Image or on
    numpy operations on Image["pixels"].data.
    If the "pixels" data variable is chunked then
    Dask is automatically used wherever
    possible to distribute processing.

    Here is an example::

        <xarray.Image>
        Dimensions:       (chan: 3, pol: 4, x: 256, y: 256)
        Coordinates:
            frequency     (chan) float64 1e+08 1.01e+08 1.02e+08
            polarisation  (pol) <U1 'I' 'Q' 'U' 'V'
          * y             (y) float64 -35.11 -35.11  ... -34.89 -34.89
          * x             (x) float64 179.9 179.9  ... 180.1 180.1
            ra            (x, y) float64 180.1 180.1  ... 179.9 179.9
            dec           (x, y) float64 -35.11 -35.11  ... -34.89 -34.89
        Dimensions without coordinates: chan, pol
        Data variables:
            pixels        (chan, pol, y, x) float64 0.0 0.0  ... 0.0 0.0
        Attributes:
            data_model:     Image
            frame:          icrs
    """

    __slots__ = ()

    def __init__(
        self,
        data_vars=None,
        coords=None,
        attrs=None,
    ):
        super().__init__(data_vars, coords=coords, attrs=attrs)

    @classmethod
    def constructor(cls, data, polarisation_frame, wcs, clean_beam=None):
        """
        Create an Image

        Note that the spatial coordinates x, y are linear.
        ra, dec coordinates can be added later.

        The addition of ra, dec grid enables selections such as:

        .. math::
            secd = 1.0 / numpy.cos(numpy.deg2rad(im.dec_grid))
            r = numpy.hypot(
                (im.ra_grid - im.ra) * secd,
                im.dec_grid - im.image.dec,
            )
            show_image(im.where(r < 0.3, 0.0))
            plt.show()

        :param data: pixel values; dims = [nchan, npol, ny, nx]
        :param polarisation_frame: as a PolarisationFrame object
        :param wcs: WCS object (with naxis=4 to match dims of data)
        :param clean_beam: dict e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}.
                Units are deg, deg, deg
        :return: Image (i.e. xarray.Dataset)
        """
        nchan, _, ny, nx = data.shape

        frequency = wcs.sub([4]).wcs_pix2world(range(nchan), 0)[0]
        cellsize = numpy.deg2rad(numpy.abs(wcs.wcs.cdelt[1]))
        ra = numpy.deg2rad(wcs.wcs.crval[0])
        dec = numpy.deg2rad(wcs.wcs.crval[1])

        # Define the dimensions
        dims = ["frequency", "polarisation", "y", "x"]

        # Define the coordinates on these dimensions
        coords = {
            "frequency": ("frequency", frequency),
            "polarisation": ("polarisation", polarisation_frame.names),
            "y": numpy.linspace(
                dec - cellsize * ny / 2,
                dec + cellsize * ny / 2,
                ny,
                endpoint=False,
            ),
            "x": numpy.linspace(
                ra - cellsize * nx / 2,
                ra + cellsize * nx / 2,
                nx,
                endpoint=False,
            ),
        }

        data_vars = {}
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)

        if isinstance(clean_beam, dict):
            missing_keys = []
            for key in ["bmaj", "bmin", "bpa"]:
                if key not in clean_beam.keys():
                    missing_keys.append(key)
            if missing_keys:
                raise KeyError(
                    f"Image: clean_beam must have key(s): {missing_keys}"
                )

        attrs = {
            "data_model": "Image",
            "_polarisation_frame": polarisation_frame.type,
            "_projection": (wcs.wcs.ctype[0], wcs.wcs.ctype[1]),
            "spectral_type": wcs.wcs.ctype[3],
            "clean_beam": clean_beam,
            "refpixel": (wcs.wcs.crpix),
            "channel_bandwidth": wcs.wcs.cdelt[3],
            "ra": ra,
            "dec": dec,
        }

        return cls(data_vars, coords=coords, attrs=attrs)

    def __sizeof__(self):
        """Override default method to return size of dataset
        :return: int
        """
        # Dask uses sizeof() class to get memory occupied by various data
        # objects. For custom data objects like this one, dask falls back to
        # sys.getsizeof() function to get memory usage. sys.getsizeof() in
        # turns calls __sizeof__() magic method to get memory size. Here we
        # override the default method (which gives size of reference table)
        # to return size of Dataset.
        return int(self.nbytes)


@xarray.register_dataset_accessor("image_acc")
class ImageAccessor(XarrayAccessorMixin):
    """
    Image property accessor
    """

    @property
    def shape(self):
        """Shape of array"""
        return self._obj["pixels"].data.shape

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj.frequency)

    @property
    def npol(self):
        """Number of polarisations"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"]).npol

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def projection(self):
        """Projection (from coords)"""
        return self._obj.attrs["_projection"]

    @property
    def phasecentre(self):
        """Return the phasecentre as a SkyCoord"""
        return SkyCoord(
            numpy.rad2deg(self._obj.attrs["ra"]) * u.deg,
            numpy.rad2deg(self._obj.attrs["dec"]) * u.deg,
            frame="icrs",
            equinox="J2000",
        )

    @property
    def wcs(self):
        """Return the equivalent WCS"""
        return image_wcs(self._obj)

    def is_canonical(self):
        """Is this Image canonical format?"""
        wcs = self.wcs

        canonical = True
        canonical = canonical and len(self._obj["pixels"].data.shape) == 4
        canonical = (
            canonical
            and wcs.wcs.ctype[0] == "RA---SIN"
            and wcs.wcs.ctype[1] == "DEC--SIN"
        )
        canonical = canonical and wcs.wcs.ctype[2] == "STOKES"
        canonical = canonical and (
            wcs.wcs.ctype[3] == "FREQ" or wcs.wcs.ctype[3] == "MOMENT"
        )

        if not canonical:
            log.debug(
                "Image: is_canonical: Image is not canonical 4D image "
                "with axes RA---SIN, DEC--SIN, STOKES, FREQ"
            )
            log.debug("Image: is_canonical: axes are: %s", wcs.wcs.ctype)

        return canonical

    def export_to_fits(self, fits_file: str = "imaging.fits"):
        """
        Write an image to fits

        :param fits_file: Name of output FITS file in storage
        """
        header = self.wcs.to_header()
        clean_beam = self._obj.attrs["clean_beam"]

        if clean_beam is not None and not isinstance(clean_beam, dict):
            raise ValueError(f"clean_beam is not a dict or None: {clean_beam}")

        if isinstance(clean_beam, dict):
            if (
                "bmaj" in clean_beam.keys()
                and "bmin" in clean_beam.keys()
                and "bpa" in clean_beam.keys()
            ):
                header.append(
                    fits.Card(
                        "BMAJ",
                        clean_beam["bmaj"],
                        "[deg] CLEAN beam major axis",
                    )
                )
                header.append(
                    fits.Card(
                        "BMIN",
                        clean_beam["bmin"],
                        "[deg] CLEAN beam minor axis",
                    )
                )
                header.append(
                    fits.Card(
                        "BPA",
                        clean_beam["bpa"],
                        "[deg] CLEAN beam position angle",
                    )
                )
            else:
                log.warning(
                    "export_to_fits: clean_beam is incompletely "
                    "specified: %s, not writing",
                    clean_beam,
                )
        if self._obj["pixels"].data.dtype == "complex":
            fits.writeto(
                filename=fits_file,
                data=numpy.real(self._obj["pixels"].data),
                header=header,
                overwrite=True,
            )
        else:
            fits.writeto(
                filename=fits_file,
                data=self._obj["pixels"].data,
                header=header,
                overwrite=True,
            )

    def qa_image(self, context="") -> QualityAssessment:
        """
        Assess the quality of an image

        QualityAssessment is a standard set of statistics of an image;
        max, min, maxabs, rms, sum, medianabs, medianabsdevmedian, median

        :return: QualityAssessment
        """
        im_data = self._obj["pixels"].data
        data = {
            "shape": str(im_data.shape),
            "size": self._obj.nbytes,
            "max": numpy.max(im_data),
            "min": numpy.min(im_data),
            "maxabs": numpy.max(numpy.abs(im_data)),
            "rms": numpy.std(im_data),
            "sum": numpy.sum(im_data),
            "medianabs": numpy.median(numpy.abs(im_data)),
            "medianabsdevmedian": numpy.median(
                numpy.abs(im_data - numpy.median(im_data))
            ),
            "median": numpy.median(im_data),
        }

        qa = QualityAssessment(origin="qa_image", data=data, context=context)
        return qa
