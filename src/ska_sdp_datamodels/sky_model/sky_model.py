# pylint: disable=invalid-name,too-many-arguments

"""
Sky-related data models
"""

import numpy

from ska_sdp_datamodels.science_data_model import PolarisationFrame


class SkyComponent:
    """
    SkyComponents are used to represent compact
    sources on the sky. They possess direction,
    flux as a function of frequency and polarisation,
    shape (with params), and polarisation frame

    For example, the following creates and predicts
    the visibility from a collection of point sources
    drawn from the GLEAM catalog::

        sc = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                            polarisation_frame=PolarisationFrame("stokesIQUV"),
                                            frequency=frequency, kind='cubic',
                                            phasecentre=phasecentre,
                                            radius=0.1)
        model = create_image_from_visibility(vis, cellsize=0.001, npixel=512, frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesIQUV'))

        bm = create_low_test_beam(model=model)
        sc = apply_beam_to_skycomponent(sc, bm)
        vis = dft_skycomponent_visibility(vis, sc)
    """  # noqa: E501

    def __init__(
        self,
        direction=None,
        frequency=None,
        name=None,
        flux=None,
        shape="Point",
        polarisation_frame=PolarisationFrame("stokesIQUV"),
        params=None,
    ):
        """Define the required structure

        :param direction: SkyCoord
        :param frequency: numpy.array [nchan]
        :param name: user friendly name
        :param flux: numpy.array [nchan, npol]
        :param shape: str e.g. 'Point' 'Gaussian'
        :param polarisation_frame: Polarisation_frame
                e.g. PolarisationFrame('stokesIQUV')
        :param params: numpy.array shape dependent parameters
        """

        self.direction = direction
        self.frequency = numpy.array(frequency)
        self.name = name
        self.flux = numpy.array(flux)
        self.shape = shape
        if params is None:
            params = {}
        self.params = params
        self.polarisation_frame = polarisation_frame

        assert len(self.frequency.shape) == 1, frequency
        assert len(self.flux.shape) == 2, flux
        assert self.frequency.shape[0] == self.flux.shape[0], (
            f"Frequency shape {self.frequency.shape}, "
            f"flux shape {self.flux.shape}"
        )
        assert polarisation_frame.npol == self.flux.shape[1], (
            f"Polarisation is {polarisation_frame.type}, "
            f"flux shape {self.flux.shape}"
        )

    @property
    def nchan(self):
        """Number of channels"""
        return self.flux.shape[0]

    @property
    def npol(self):
        """Number of polarisations"""
        return self.flux.shape[1]

    def __str__(self):
        """Default printer for SkyComponent"""
        s = "SkyComponent:\n"
        s += f"\tName: {self.name}\n"
        s += f"\tFlux: {self.flux}\n"
        s += f"\tFrequency: {self.frequency}\n"
        s += f"\tDirection: {self.direction}\n"
        s += f"\tShape: {self.shape}\n"

        s += f"\tParams: {self.params}\n"
        s += f"\tPolarisation frame: {str(self.polarisation_frame.type)}\n"
        return s


class SkyModel:
    """
    A model for the sky, including an image,
    components, gaintable and a mask
    """

    def __init__(
        self,
        image=None,
        components=None,
        gaintable=None,
        mask=None,
        fixed=False,
    ):
        """A model of the sky as an image, components, gaintable and a mask

        Use copy_skymodel to make a proper copy of skymodel
        :param image: Image
        :param components: List of components
        :param gaintable: Gaintable for this skymodel
        :param mask: Mask for the image
        :param fixed: Is this model fixed?
        """
        if components is None:
            components = []
        if not isinstance(components, (list, tuple)):
            components = [components]

        self.image = image

        self.components = components
        self.gaintable = gaintable

        self.mask = mask

        self.fixed = fixed

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

        # Get size of reference tables
        obj_size = int(super().__sizeof__())

        # Add size of image data object
        if self.image is not None:
            obj_size += int(self.image.nbytes)

        # Add size of gaintable data object
        if self.gaintable is not None:
            obj_size += int(self.gaintable.nbytes)

        # Add size of gaintable data object
        if self.mask is not None:
            obj_size += int(self.mask.nbytes)

        return obj_size

    def __str__(self):
        """Default printer for SkyModel"""
        s = f"SkyModel: fixed: {self.fixed}\n"
        for _, sc in enumerate(self.components):
            s += str(sc)
        s += "\n"

        s += str(self.image)
        s += "\n"

        s += str(self.mask)
        s += "\n"

        s += str(self.gaintable)

        return s
