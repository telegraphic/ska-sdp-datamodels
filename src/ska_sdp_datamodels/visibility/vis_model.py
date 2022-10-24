# pylint: disable=too-many-ancestors,too-many-arguments,too-many-locals
# pylint: disable=invalid-name

"""
Visibility data model.
"""

import warnings

import numpy
import pandas
import xarray
from astropy import constants as const
from astropy.time import Time

from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    QualityAssessment,
)
from ska_sdp_datamodels.xarray_accessor import XarrayAccessorMixin


class Visibility(xarray.Dataset):
    """
    Visibility xarray.Dataset class

    Visibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0.
    If uvw are rotated then this should be updated with the
    new delay tracking centre.

    Polarisation frame is the same for the entire data set and can be
    stokesI, circular, circularnp, linear, linearnp.

    The configuration is stored as an attribute.

    Here is an example::

        <xarray.Visibility>
        Dimensions:            (baselines: 6670, frequency: 3, polarisation: 4, time: 3, uvw_index: 3)
        Coordinates:
          * time               (time) float64 5.085e+09 5.085e+09 5.085e+09
          * baselines          (baselines) MultiIndex
          - antenna1           (baselines) int64 0 0 0 0 0 0 ... 112 112 112 113 113 114
          - antenna2           (baselines) int64 0 1 2 3 4 5 ... 112 113 114 113 114 114
          * frequency          (frequency) float64 1e+08 1.05e+08 1.1e+08
          * polarisation       (polarisation) <U2 'XX' 'XY' 'YX' 'YY'
          * uvw_index          (uvw_index) <U1 'u' 'v' 'w'
        Data variables:
            integration_time   (time) float32 99.72697 99.72697 99.72697
            datetime           (time) datetime64[ns] 2000-01-01T03:54:07.843184299 .....
            vis                (time, baselines, frequency, polarisation) complex128 ...
            weight             (time, baselines, frequency, polarisation) float32 0.0...
            flags              (time, baselines, frequency, polarisation) int32 0.0...
            uvw                (time, baselines, uvw_index) float64 0.0 0.0 ... 0.0 0.0
            channel_bandwidth  (frequency) float64 1e+07 1e+07 1e+07
        Attributes:
            phasecentre:         <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:       <xarray.Configuration>Dimensions:   (id: 115, spat...
            polarisation_frame:  linear
            source:              unknown
            meta:                None
    """  # noqa:E501 pylint: disable=line-too-long

    __slots__ = ("_imaging_weight",)

    def __init__(
        self,
        data_vars=None,
        coords=None,
        attrs=None,
    ):
        super().__init__(data_vars, coords=coords, attrs=attrs)
        self._imaging_weight = None

    @classmethod
    def constructor(
        cls,
        frequency=None,
        channel_bandwidth=None,
        phasecentre=None,
        configuration=None,
        uvw=None,
        time=None,
        vis=None,
        weight=None,
        integration_time=None,
        flags=None,
        baselines=None,
        polarisation_frame=PolarisationFrame("stokesI"),
        source="anonymous",
        meta=None,
        low_precision="float64",
    ):
        """Visibility

        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        :param uvw: UVW coordinates (m) [:, nant, nant, 3]
        :param time: Time (UTC) [:]
        :param baselines: List of baselines
        :param flags: Flags [:, nant, nant, nchan]
        :param weight: [:, nant, nant, nchan, npol]
        :param integration_time: Integration time [:]
        :param polarisation_frame: Polarisation_Frame
                e.g. Polarisation_Frame("linear")
        :param source: Source name
        :param meta: Meta info
        """
        if weight is None:
            weight = numpy.ones(vis.shape)
        else:
            assert weight.shape == vis.shape

        if integration_time is None:
            integration_time = numpy.ones_like(time)
        else:
            assert len(integration_time) == len(time)

        # Define the names of the dimensions
        coords = {  # pylint: disable=duplicate-code
            "time": time,
            "baselines": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "spatial": ["u", "v", "w"],
        }

        datavars = {}
        datavars["integration_time"] = xarray.DataArray(
            integration_time.astype(low_precision),
            dims=["time"],
            attrs={"units": "s"},
        )
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims=["time"],
            attrs={"units": "s"},
        )
        datavars["vis"] = xarray.DataArray(
            vis,
            dims=["time", "baselines", "frequency", "polarisation"],
            attrs={"units": "Jy"},
        )
        datavars["weight"] = xarray.DataArray(
            weight.astype(low_precision),
            dims=["time", "baselines", "frequency", "polarisation"],
        )
        datavars["flags"] = xarray.DataArray(
            flags.astype(int),
            dims=["time", "baselines", "frequency", "polarisation"],
        )
        datavars["uvw"] = xarray.DataArray(
            uvw, dims=["time", "baselines", "spatial"], attrs={"units": "m"}
        )

        datavars["channel_bandwidth"] = xarray.DataArray(
            channel_bandwidth, dims=["frequency"], attrs={"units": "Hz"}
        )

        attrs = {}
        attrs["data_model"] = "Visibility"
        attrs["configuration"] = configuration  # Antenna/station configuration
        attrs["source"] = source
        attrs["phasecentre"] = phasecentre
        attrs["_polarisation_frame"] = polarisation_frame.type
        attrs["meta"] = meta

        return cls(datavars, coords=coords, attrs=attrs)

    @property
    def imaging_weight(self):
        """
        Legacy data attribute. Deprecated.
        """
        warnings.warn(
            "imaging_weight is deprecated, please use weight instead",
            DeprecationWarning,
        )
        if self._imaging_weight is None:
            self._imaging_weight = xarray.DataArray(
                self.weight.data.astype(self.weight.data.dtype),
                dims=["time", "baselines", "frequency", "polarisation"],
            )
        return self._imaging_weight

    @imaging_weight.setter
    def imaging_weight(self, new_img_weight):
        warnings.warn(
            "imaging_weight is deprecated, please use weight instead",
            DeprecationWarning,
        )
        if not new_img_weight.shape == self.weight.data.shape:
            raise ValueError(
                "New imaging weight does not match shape of weight"
            )

        self._imaging_weight = xarray.DataArray(
            new_img_weight.astype(self.weight.data.dtype),
            dims=["time", "baselines", "frequency", "polarisation"],
        )

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

    def copy(self, deep=False, data=None, zero=False):
        """
        Copy Visibility

        :param deep: perform deep-copy
        :param data: data to use in new object; see docstring of
                     xarray.core.dataset.Dataset.copy
        :param zero: if True, set visibility data to zero in copied object
        """
        new_vis = super().copy(deep=deep, data=data)
        if zero:
            new_vis["vis"].data[...] = 0.0

        setattr(new_vis, "_imaging_weight", self._imaging_weight)
        return new_vis

    def qa_visibility(self, context=None) -> QualityAssessment:
        """Assess the quality of Visibility"""

        avis = numpy.abs(self["vis"].data)
        data = {
            "maxabs": numpy.max(avis),
            "minabs": numpy.min(avis),
            "rms": numpy.std(avis),
            "medianabs": numpy.median(avis),
        }
        qa = QualityAssessment(
            origin="qa_visibility", data=data, context=context
        )
        return qa

    def performance_visibility(self):
        """Get info about the visibility

        This works on a single visibility because we
        probably want to send this function to
        the cluster instead of bringing the data back
        :return: bvis info as a dictionary
        """
        bv_info = {
            "number_times": self.visibility_acc.ntimes,
            "number_baselines": len(self.baselines),
            "nchan": self.visibility_acc.nchan,
            "npol": self.visibility_acc.npol,
            "polarisation_frame": self.visibility_acc.polarisation_frame.type,
            "nvis": self.visibility_acc.ntimes
            * len(self.baselines)
            * self.visibility_acc.nchan
            * self.visibility_acc.npol,
            "size": self.nbytes,
        }
        return bv_info

    def select_uv_range(self, uvmin=0.0, uvmax=1.0e15):
        """Visibility selection functions

        To select by row number::
            selected_bvis = bvis.isel({"time": slice(5, 7)})
        To select by frequency channel::
            selected_bvis = bvis.isel({"frequency": slice(1, 3)})
        To select by frequency::
            selected_bvis = bvis.sel({"frequency": slice(0.9e8, 1.2e8)})
        To select by frequency and polarisation::
            selected_bvis = bvis.sel(
              {"frequency": slice(0.9e8, 1.2e8), "polarisation": ["XX", "YY"]}
            ).dropna(dim="frequency", how="all")

        Select uv range: flag in-place all visibility data
        outside uvrange uvmin, uvmax (wavelengths)
        The flags are set to 1 for all data outside the specified uvrange

        :param uvmin: Minimum uv to flag
        :param uvmax: Maximum uv to flag
        :return: Visibility (with flags applied)
        """
        uvdist_lambda = numpy.hypot(
            self.visibility_acc.uvw_lambda[..., 0],
            self.visibility_acc.uvw_lambda[..., 1],
        )
        if uvmax is not None:
            self["flags"].data[numpy.where(uvdist_lambda >= uvmax)] = 1
        if uvmin is not None:
            self["flags"].data[numpy.where(uvdist_lambda <= uvmin)] = 1

    def select_r_range(self, rmin=None, rmax=None):
        """
        Select a visibility with stations in a range
        of distance from the array centre
        r is the distance from the array centre in metres

        :param rmax: Maximum r
        :param rmin: Minimum r
        :return: Selected Visibility
        """
        if rmin is None and rmax is None:
            return self

        # Calculate radius from array centre (in 3D)
        # and set it as a data variable
        xyz0 = self.configuration.xyz - self.configuration.xyz.mean("id")
        r = numpy.sqrt(xarray.dot(xyz0, xyz0, dims="spatial"))
        config = self.configuration.assign(radius=r)
        # Now use it for selection
        if rmax is None:
            sub_config = config.where(config["radius"] > rmin, drop=True)
        elif rmin is None:
            sub_config = config.where(config["radius"] < rmax, drop=True)
        else:
            sub_config = config.where(
                config["radius"] > rmin, drop=True
            ).where(config["radius"] < rmax, drop=True)

        ids = list(sub_config.id.data)
        baselines = self.baselines.where(
            self.baselines.antenna1.isin(ids), drop=True
        ).where(self.baselines.antenna2.isin(ids), drop=True)
        sub_bvis = self.sel({"baselines": baselines}, drop=True)
        setattr(sub_bvis, "_imaging_weight", self._imaging_weight)

        # The baselines coord now is missing the antenna1, antenna2 keys
        # so we add those back
        def generate_baselines(baseline_id):
            for a1 in baseline_id:
                for a2 in baseline_id:
                    if a2 >= a1:
                        yield a1, a2

        sub_bvis["baselines"] = pandas.MultiIndex.from_tuples(
            generate_baselines(ids),
            names=("antenna1", "antenna2"),
        )
        return sub_bvis

    def groupby(
        self, group, squeeze: bool = True, restore_coord_dims: bool = None
    ):
        """Override default method to group _imaging_weight"""
        grouped_dataset = super().groupby(
            group, squeeze=squeeze, restore_coord_dims=restore_coord_dims
        )

        if self._imaging_weight is not None:
            group_imaging_weight = self._imaging_weight.groupby(
                group, squeeze=squeeze, restore_coord_dims=restore_coord_dims
            )

            for (dimension, vis_slice), (_, imaging_weight_slice) in zip(
                grouped_dataset, group_imaging_weight
            ):
                setattr(vis_slice, "_imaging_weight", imaging_weight_slice)
                yield dimension, vis_slice
        else:
            for dimension, vis_slice in grouped_dataset:
                setattr(vis_slice, "_imaging_weight", None)
                yield dimension, vis_slice

    def groupbybins(
        self,
        group,
        bins,
        right=True,
        labels=None,
        precision=3,
        include_lowest=False,
        squeeze=True,
        restore_coord_dims=False,
    ):
        """
        Overwriting groupbybins method.
        See docstring of Dataset.groupbybins
        """
        grouped_dataset = super().groupby_bins(
            group,
            bins,
            right=right,
            labels=labels,
            precision=precision,
            include_lowest=include_lowest,
            squeeze=squeeze,
            restore_coord_dims=restore_coord_dims,
        )

        if self._imaging_weight is not None:
            group_imaging_weight = self._imaging_weight.groupby_bins(
                group,
                squeeze=squeeze,
                bins=bins,
                restore_coord_dims=restore_coord_dims,
                cut_kwargs={
                    "right": right,
                    "labels": labels,
                    "precision": precision,
                    "include_lowest": include_lowest,
                },
            )

            for (dimension, vis_slice), (_, imaging_weight_slice) in zip(
                grouped_dataset, group_imaging_weight
            ):
                setattr(vis_slice, "_imaging_weight", imaging_weight_slice)
                yield dimension, vis_slice
        else:
            for dimension, vis_slice in grouped_dataset:
                setattr(vis_slice, "_imaging_weight", None)
                yield dimension, vis_slice


@xarray.register_dataset_accessor("visibility_acc")
class VisibilityAccessor(XarrayAccessorMixin):
    """
    Visibility property accessor
    """

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        self._uvw_lambda = None

    @property
    def rows(self):
        """Rows"""
        return range(len(self._obj.time))

    @property
    def ntimes(self):
        """Number of times (i.e. rows) in this table"""
        return len(self._obj["time"])

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["frequency"])

    @property
    def npol(self):
        """Number of polarisations"""
        return len(self._obj.polarisation)

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)"""
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def nants(self):
        """Number of antennas"""
        return self._obj.configuration.configuration_acc.nants

    @property
    def nbaselines(self):
        """Number of Baselines"""
        return len(self._obj["baselines"])

    @property
    def uvw_lambda(self):
        """
        Calculate and set uvw_lambda
        dims=[ntimes, nbaselines, nchan, spatial(3)]
        Note: We omit the frequency and polarisation
            dependency of uvw for the calculation
        """
        if self._uvw_lambda is None:
            k = (
                self._obj["frequency"].data
                / const.c  # pylint: disable=no-member
            ).value
            uvw = self._obj["uvw"].data
            if self.nchan == 1:
                self._uvw_lambda = (uvw * k)[..., numpy.newaxis, :]
            else:
                self._uvw_lambda = numpy.einsum("tbs,k->tbks", uvw, k)

        return self._uvw_lambda

    @uvw_lambda.setter
    def uvw_lambda(self, new_value):
        """
        Re-set uvw_lambda to a given value if it has been recalculated
        """

        if not new_value.shape == (
            self.ntimes,
            self.nbaselines,
            self.nchan,
            3,
        ):
            raise ValueError(
                "Data shape of new uvw_lambda "
                "incompatible with visibility setup"
            )

        self._uvw_lambda = new_value

    @property
    def u(self):
        """u coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 0]

    @property
    def v(self):
        """v coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 1]

    @property
    def w(self):
        """w coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 2]

    @property
    def flagged_vis(self):
        """Flagged complex visibility [nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["vis"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_weight(self):
        """Weight [: npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["weight"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_imaging_weight(self):
        """Flagged Imaging_weight[nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        warnings.warn(
            "flagged_imaging_weight is deprecated, "
            "please use flagged_weight instead",
            DeprecationWarning,
        )
        return self._obj.imaging_weight.data * (1 - self._obj["flags"].data)

    @property
    def nvis(self):
        """Number of visibilities (in total)"""
        return numpy.product(self._obj.vis.shape)
