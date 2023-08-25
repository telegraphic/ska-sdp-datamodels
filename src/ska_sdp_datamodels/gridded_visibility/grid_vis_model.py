# pylint: disable=too-many-ancestors,too-many-locals,invalid-name
"""
Gridded visibility models.
"""

import numpy
import xarray

from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    QualityAssessment,
)
from ska_sdp_datamodels.xarray_accessor import XarrayAccessorMixin
from ska_sdp_datamodels.xarray_coordinate_support import (
    conv_func_wcs,
    griddata_wcs,
    image_wcs,
)


class GridData(xarray.Dataset):
    """Class to hold Gridded data for Fourier processing

    - Has four or more coordinates: [chan, pol, z, y, x]
        where x can be u, l; y can be v, m; z can be w, n.
        Note: current implementation only uses 4 coordinates:
        [nchan, npol, v, u]

    The conventions for indexing in WCS and numpy are opposite:

    - In astropy.wcs, the order is
        (longitude, latitude, polarisation, frequency);

    - in numpy, the order is
        (frequency, polarisation, depth, latitude, longitude).

    .. warning::
        The polarisation_frame is kept in two places, the
        WCS and the polarisation_frame variable.
        The latter should be considered definitive.

    Here is an example::

        <xarray.GridData>
        Dimensions:       (frequency: 3, polarisation: 4, v: 256, u: 256)
        Coordinates:
          * frequency     (frequency) float64 1e+08 1.01e+08 1.02e+08
          * polarisation  (polarisation) <U1 'I' 'Q' 'U' 'V'
          * v             (v) float64 -3.333e+04 -3.307e+04 ...  3.307e+04
          * u             (u) float64 3.333e+04 3.307e+04 ...  -3.307e+04
        Data variables:
            pixels        (frequency, polarisation, v, u)
                            complex128 0j 0j 0j ... 0j 0j
        Attributes:
            data_model:           GridData
            _polarisation_frame:  stokesIQUV

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
    def constructor(cls, data, polarisation_frame=None, grid_wcs=None):
        """
        Create a GridData

        :param data: pixel data array;
                     dims: ["frequency", "polarisation", "v", "u"]
        :param polarisation_frame: PolarisationFrame object
        :param grid_wcs: astropy WCS object
        :return: GridData
        """
        nchan, npol, nv, nu = data.shape

        frequency = grid_wcs.sub([4]).wcs_pix2world(range(nchan), 0)[0]

        if not npol == polarisation_frame.npol:
            raise ValueError(
                "Polarisation dimensions of input PolarisationFrame "
                "does not mach that of data polarisation dimensions: "
                f"{polarisation_frame.npol} != {npol}"
            )

        cu = grid_wcs.wcs.crval[0]
        cv = grid_wcs.wcs.crval[1]
        du = grid_wcs.wcs.cdelt[0]
        dv = grid_wcs.wcs.cdelt[1]

        dims = ["frequency", "polarisation", "v", "u"]

        # Define the coordinates on these dimensions
        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "v": numpy.linspace(
                cv - dv * nv / 2, cv + dv * nv / 2, nv, endpoint=False
            ),
            "u": numpy.linspace(
                cu - du * nu / 2, cu + du * nu / 2, nu, endpoint=False
            ),
        }

        attrs = {}

        attrs["data_model"] = "GridData"
        attrs["_polarisation_frame"] = polarisation_frame.type

        data_vars = {}
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)

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


@xarray.register_dataset_accessor("griddata_acc")
class GridDataAccessor(XarrayAccessorMixin):
    """
    GridDataAccessor property accessor
    """

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
    def shape(self):
        """Shape of data array"""
        return self._obj["pixels"].data.shape

    @property
    def griddata_wcs(self):
        """Return the equivalent WCS coordinates"""
        return griddata_wcs(self._obj)

    @property
    def projection_wcs(self):
        """Return the projected WCS coordinates on image"""
        return image_wcs(self._obj)

    def qa_grid_data(self, context="") -> QualityAssessment:
        """Assess the quality of a griddata

        :return: QualityAssessment
        """
        grid_data = self._obj["pixels"].data
        data = {
            "shape": str(self._obj["pixels"].data.shape),
            "max": numpy.max(grid_data),
            "min": numpy.min(grid_data),
            "rms": numpy.std(grid_data),
            "sum": numpy.sum(grid_data),
            "medianabs": numpy.median(numpy.abs(grid_data)),
            "median": numpy.median(grid_data),
        }

        qa = QualityAssessment(
            origin="qa_grid_data", data=data, context=context
        )
        return qa


class ConvolutionFunction(xarray.Dataset):
    """
    Class to hold Convolution function for Fourier processing
    - Has four or more coordinates: [chan, pol, z, y, x]
    where x can be u, l; y can be v, m; z can be w, n.

    The cf has axes [chan, pol, dy, dx, y, x] where
    z, y, x are spatial axes in either sky or Fourier plane.
    The order in the WCS is reversed so the grid_WCS describes
    UU, VV, WW, STOKES, FREQ axes.

    The axes UU,VV have the same physical stride as the Image.
    The axes DUU, DVV are sub-sampled.

    Convolution function holds the original sky
    plane projection in the projection_wcs.

    Here is an example::

        <xarray.ConvolutionFunction>
        Dimensions:       (du: 8, dv: 8, frequency: 1,
                            polarisation: 1, u: 16, v: 16, w: 1)
        Coordinates:
          * frequency     (frequency) float64 1e+08
          * polarisation  (polarisation) <U1 'I'
          * w             (w) float64 0.0
          * dv            (dv) float64 -1.031e+05 -7.735e+04 ... 7.735e+04
          * du            (du) float64 -1.031e+05 -7.735e+04 ... 7.735e+04
          * v             (v) float64 -1.65e+06 -1.444e+06 ... 1.444e+06
          * u             (u) float64 -1.65e+06 -1.444e+06 ... 1.444e+06
        Data variables:
            pixels        (frequency, polarisation, w, dv, du, v, u)
                           complex128 0j ......
        Attributes:
            data_model:   ConvolutionFunction
            grid_wcs:            WCS Keywords
                                 Number of WCS axes: 7
                                 CTYPE : 'UU' ...
            projection_wcs:      WCS Keywords
                                 Number of WCS axes: 4
                                 CTYPE : 'RA--...
            polarisation_frame:  stokesI
    """  # noqa:E501

    __slots__ = ()

    def __init__(
        self,
        data_vars=None,
        coords=None,
        attrs=None,
    ):
        super().__init__(data_vars, coords=coords, attrs=attrs)

    @classmethod
    def constructor(cls, data, cf_wcs=None, polarisation_frame=None):
        """
        Create ConvolutionFunction

        :param data: Data for cf
        :param cf_wcs: Astropy WCS object for the grid
        :param polarisation_frame: Polarisation_frame
                e.g. PolarisationFrame('linear')
        """
        nchan, npol, nw, oversampling, _, support, _ = data.shape
        frequency = cf_wcs.sub(["spectral"]).wcs_pix2world(range(nchan), 0)[0]

        assert npol == polarisation_frame.npol, (
            "Mismatch between requested image polarisation "
            "and actual visibility polarisation"
        )

        du = cf_wcs.wcs.cdelt[0]
        dv = cf_wcs.wcs.cdelt[1]
        ddu = cf_wcs.wcs.cdelt[0] / oversampling
        ddv = cf_wcs.wcs.cdelt[1] / oversampling
        cu = cf_wcs.wcs.crval[0]
        cv = cf_wcs.wcs.crval[1]
        cdu = oversampling // 2
        cdv = oversampling // 2

        wstep = numpy.abs(cf_wcs.wcs.cdelt[4])

        coords = {
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "dv": numpy.linspace(
                cdv - ddv * oversampling / 2,
                cdv + ddv * oversampling / 2,
                oversampling,
                endpoint=False,
            ),
            "du": numpy.linspace(
                cdu - ddu * oversampling / 2,
                cdu + ddu * oversampling / 2,
                oversampling,
                endpoint=False,
            ),
            "w": numpy.linspace(
                -wstep * nw / 2, wstep * nw / 2, nw, endpoint=False
            ),
            "v": numpy.linspace(
                cv - dv * support / 2,
                cv + dv * support / 2,
                support,
                endpoint=False,
            ),
            "u": numpy.linspace(
                cu - du * support / 2,
                cu + du * support / 2,
                support,
                endpoint=False,
            ),
        }

        if nw == 1:
            coords["w"]: numpy.zeros([1])

        dims = ["frequency", "polarisation", "w", "dv", "du", "u", "v"]

        assert coords["u"][0] != coords["u"][-1]
        assert coords["v"][0] != coords["v"][-1]

        attrs = {}
        attrs["data_model"] = "ConvolutionFunction"
        attrs["_polarisation_frame"] = polarisation_frame.type

        nchan = len(frequency)
        npol = polarisation_frame.npol
        if data is None:
            data = numpy.zeros(
                [
                    nchan,
                    npol,
                    nw,
                    oversampling,
                    oversampling,
                    support,
                    support,
                ],
                dtype="complex",
            )
        else:
            assert data.shape == (
                nchan,
                npol,
                nw,
                oversampling,
                oversampling,
                support,
                support,
            ), (
                f"Polarisation frame {polarisation_frame.type} and data shape "
                f"{data.shape} are incompatible"
            )
        data_vars = {}
        data_vars["pixels"] = xarray.DataArray(data, dims=dims, coords=coords)

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


@xarray.register_dataset_accessor("convolutionfunction_acc")
class ConvolutionFunctionAccessor(XarrayAccessorMixin):
    """
    ConvolutionFunction property accessor
    """

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj.frequency)

    @property
    def npol(self):
        """Number of polarisations"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"]).npol

    @property
    def cf_wcs(self):
        """Return the equivalent WCS coordinates"""
        return conv_func_wcs(self._obj)

    @property
    def shape(self):
        """Shape of data array"""
        return self._obj["pixels"].data.shape

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    def qa_convolution_function(self, context="") -> QualityAssessment:
        """Assess the quality of a ConvolutionFunction

        :return: QualityAssessment
        """
        conv_func_data = self._obj["pixels"].data
        data = {
            "shape": str(self._obj["pixels"].data.shape),
            "max": numpy.max(conv_func_data),
            "min": numpy.min(conv_func_data),
            "rms": numpy.std(conv_func_data),
            "sum": numpy.sum(conv_func_data),
            "medianabs": numpy.median(numpy.abs(conv_func_data)),
            "median": numpy.median(conv_func_data),
        }

        qa = QualityAssessment(
            origin="qa_convolution_function", data=data, context=context
        )
        return qa
