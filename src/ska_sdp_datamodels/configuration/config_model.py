# pylint: disable=too-many-ancestors,too-many-arguments,too-many-locals

"""
Telescope configuration model.
"""

import numpy
import xarray

from ska_sdp_datamodels.science_data_model import ReceptorFrame
from ska_sdp_datamodels.xarray_accessor import XarrayAccessorMixin


class Configuration(xarray.Dataset):
    """
    A Configuration describes an array configuration

    Here is an example::

        <xarray.Configuration>
        Dimensions:   (id: 115, spatial: 3)
        Coordinates:
          * id        (id) int64 0 1 2 3 4 5 6 7 8 ...  113 114
          * spatial   (spatial) <U1 'X' 'Y' 'Z'
        Data variables:
            names     (id) <U6 'M000' 'M001' 'M002' ... 'SKA103' 'SKA104'
            xyz       (id, spatial) float64 -0.0 9e-05 1.053e+03 ... -810.3 1.053e+03
            diameter  (id) float64 13.5 13.5 13.5 13.5 13.5 ...  15.0 15.0
            mount     (id) <U4 'azel' 'azel' 'azel' 'azel' ...  'azel'
            vp_type   (id) <U7 'MEERKAT' 'MEERKAT' 'MEERKAT' ...  'MID'
            offset    (id, spatial) float64 0.0 0.0 0.0 0.0 0.0 ...  0.0 0.0
            stations  (id) <U3 '0' '1' '2' '3' '4' '5' ...  '112' '113' '114'
        Attributes:
            data_model:  Configuration
            name:               MID
            location:           (5109237.71471275, 2006795.66194638, -3239109.1838011..
            receptor_frame:     <src.ska_sdp_datamodels.polarisation.ReceptorFrame...
            frame:
    """  # noqa: E501

    __slots__ = ()

    def __init__(
        self,
        data_vars=None,
        coords=None,
        attrs=None,
    ):
        super().__init__(data_vars, coords=coords, attrs=attrs)

    @classmethod
    def constructor(
        cls,
        name="",
        location=None,
        names=None,
        xyz=None,
        mount="alt-az",
        frame="",
        receptor_frame=ReceptorFrame("linear"),
        diameter=None,
        offset=None,
        stations="%s",
        vp_type=None,
    ):

        """
        Configuration object describing data for processing

        :param name: Name of configuration e.g. 'LOWR3'
        :param location: Location of array as an astropy EarthLocation
        :param names: Names of the dishes/stations
        :param xyz: Geocentric coordinates of dishes/stations
        :param mount: Mount types of dishes/stations
                      'altaz' | 'xy' | 'equatorial'
        :param frame: Reference frame of locations
        :param receptor_frame: Receptor frame
        :param diameter: Diameters of dishes/stations (m)
        :param offset: Axis offset (m)
        :param stations: Identifiers of the dishes/stations
        :param vp_type: Type of voltage pattern (string)
        """
        nants = len(names)
        if isinstance(stations, str):
            stations = [stations % ant for ant in range(nants)]
            if isinstance(names, str):
                names = [names % ant for ant in range(nants)]
            if isinstance(mount, str):
                mount = numpy.repeat(mount, nants)
        if offset is None:
            offset = numpy.zeros([nants, 3])
        if vp_type is None:
            vp_type = numpy.repeat("", nants)

        coords = {"id": list(range(nants)), "spatial": ["X", "Y", "Z"]}

        datavars = {}
        datavars["names"] = xarray.DataArray(
            names, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["xyz"] = xarray.DataArray(
            xyz, coords=coords, dims=["id", "spatial"]
        )
        datavars["diameter"] = xarray.DataArray(
            diameter, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["mount"] = xarray.DataArray(
            mount, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["vp_type"] = xarray.DataArray(
            vp_type, coords={"id": list(range(nants))}, dims=["id"]
        )
        datavars["offset"] = xarray.DataArray(
            offset, coords=coords, dims=["id", "spatial"]
        )
        datavars["stations"] = xarray.DataArray(
            stations, coords={"id": list(range(nants))}, dims=["id"]
        )

        attrs = {}
        attrs["data_model"] = "Configuration"
        attrs["name"] = name  # Name of configuration
        attrs["location"] = location  # EarthLocation
        attrs["receptor_frame"] = receptor_frame
        attrs["frame"] = frame

        return cls(datavars, coords=coords, attrs=attrs)

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


@xarray.register_dataset_accessor("configuration_acc")
class ConfigurationAccessor(XarrayAccessorMixin):
    """Convenience methods to access the fields of the Configuration"""

    @property
    def nants(self):
        """Names of the dishes/stations"""
        return len(self._obj["names"])
