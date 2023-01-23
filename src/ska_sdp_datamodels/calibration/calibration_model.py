# pylint: disable=too-many-ancestors,too-many-arguments,too-many-locals
# pylint: disable=invalid-name

"""
Calibration-related data models.
"""

import numpy
import xarray
from astropy.time import Time

from ska_sdp_datamodels.science_data_model import (
    QualityAssessment,
    ReceptorFrame,
)
from ska_sdp_datamodels.xarray_accessor import XarrayAccessorMixin


class GainTable(xarray.Dataset):
    """
    Gain table with: time, antenna, weight columns

    The weight is usually that output from gain solvers.

    Here is an example::

        <xarray.GainTable>
        Dimensions:    (antenna: 115, frequency: 3, receptor1: 2, receptor2: 2, time: 3)
        Coordinates:
          * time       (time) float64 5.085e+09 5.085e+09 5.085e+09
          * antenna    (antenna) int64 0 1 2 3 4 5 6 7 ... 108 109 110 111 112 113 114
          * frequency  (frequency) float64 1e+08 1.05e+08 1.1e+08
          * receptor1  (receptor1) <U1 'X' 'Y'
          * receptor2  (receptor2) <U1 'X' 'Y'
        Data variables:
            gain       (time, antenna, frequency, receptor1, receptor2) complex128 (0...
            weight     (time, antenna, frequency, receptor1, receptor2) float64 1.0 ....
            residual   (time, frequency, receptor1, receptor2) float64 0.0 0.0 ... 0.0
            interval   (time) float32 99.72697 99.72697 99.72697
            datetime   (time) datetime64[ns] 2000-01-01T03:54:07.843184299 ... 2000-0...
        Attributes:
            data_model:  GainTable
            receptor_frame1:     <src.ska_sdp_datamodels.polarisation.ReceptorFrame object...
            receptor_frame2:     <src.ska_sdp_datamodels.polarisation.ReceptorFrame object...
            phasecentre:        <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:      <xarray.Configuration> Dimensions:   (id: 115, spati...
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
        gain: numpy.array = None,
        time: numpy.array = None,
        interval=None,
        weight: numpy.array = None,
        residual: numpy.array = None,
        frequency: numpy.array = None,
        receptor_frame: ReceptorFrame = ReceptorFrame("linear"),
        phasecentre=None,
        configuration=None,
        jones_type="T",
    ):
        """
        Create a gaintable from arrays

        The definition of gain is:

            Vobs = g_i g_j^* Vmodel

        :param gain: Complex gain [nrows, nants, nchan, nrec, nrec]
        :param time: Centroid of solution [nrows]
        :param interval: Interval of validity
        :param weight: Weight of gain [nrows, nchan, nrec, nrec]
        :param residual: Residual of fit [nchan, nrec, nrec]
        :param frequency: Frequency [nchan]
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        :param jones_type: Type of gain: T, G, B, etc
        :param receptor_frame: Measured and ideal receptor frames
                equivalent to two sides of the Jones matrix
                Receptor1 stands for measured/output polarisation
                Receptor2 stands for ideal/input polarisation
                If a single frame, use it for both receptor1 and receptor2
                If a tuple, it stands for [receptor1, receptor2]
        """
        nants = gain.shape[1]
        antennas = range(nants)
        # If this doesn't work it will automatically raise a ValueError
        if isinstance(receptor_frame, (list, tuple)):
            receptor1, receptor2 = receptor_frame
            if len(receptor1) != len(receptor2):
                raise ValueError(
                    "Polarisation of the two receptors "
                    "need to have the same length "
                    "in order to return square matrix"
                )
        if isinstance(receptor_frame, ReceptorFrame):
            receptor1 = receptor_frame
            receptor2 = receptor_frame

        coords = {
            "time": time,
            "antenna": antennas,
            "frequency": frequency,
            "receptor1": receptor1.names,
            "receptor2": receptor2.names,
        }

        datavars = {}
        datavars["gain"] = xarray.DataArray(
            gain,
            dims=[
                "time",
                "antenna",
                "frequency",
                "receptor1",
                "receptor2",
            ],
        )
        datavars["weight"] = xarray.DataArray(
            weight,
            dims=[
                "time",
                "antenna",
                "frequency",
                "receptor1",
                "receptor2",
            ],
        )
        datavars["residual"] = xarray.DataArray(
            residual, dims=["time", "frequency", "receptor1", "receptor2"]
        )
        datavars["interval"] = xarray.DataArray(interval, dims=["time"])
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims="time",
        )
        attrs = {}
        attrs["data_model"] = "GainTable"
        attrs["receptor_frame1"] = receptor1
        attrs["receptor_frame2"] = receptor2
        attrs["phasecentre"] = phasecentre
        attrs["configuration"] = configuration
        attrs["jones_type"] = jones_type

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

    def copy(self, deep=False, data=None, zero=False):
        """
        Copy GainTable

        :param deep: perform deep-copy
        :param data: data to use in new object; see docstring of
                     xarray.core.dataset.Dataset.copy
        :param zero: if True, set gain data to zero in copied object
        """
        new_gt = super().copy(deep=deep, data=data)
        if zero:
            new_gt["gain"].data[...] = 0.0
        return new_gt


@xarray.register_dataset_accessor("gaintable_acc")
class GainTableAccessor(XarrayAccessorMixin):
    """
    GainTable property accessor
    """

    @property
    def ntimes(self):
        """Number of times (i.e. rows) in this table"""
        return self._obj["gain"].shape[0]

    @property
    def nants(self):
        """Number of dishes/stations"""
        return self._obj["gain"].shape[1]

    @property
    def nchan(self):
        """Number of channels"""
        return self._obj["gain"].shape[2]

    @property
    def nrec(self):
        """Number of polarisation in receptors
        Note that receptor1 and receptor2 need to have the same length"""
        return len(self._obj["receptor1"])

    @property
    def receptor1(self):
        """Measured Receptor Frame"""
        return self._obj["receptor1"]

    @property
    def receptor2(self):
        """Ideal Receptor Frame"""
        return self._obj["receptor2"]

    def qa_gain_table(self, context=None) -> QualityAssessment:
        """Assess the quality of a gaintable

        :return: QualityAssessment
        """
        weight_data = self._obj.weight.data
        if numpy.max(weight_data) <= 0.0:
            raise ValueError("qa_gain_table: All gaintable weights are zero")

        gain_data = self._obj.gain.data
        agt = numpy.abs(gain_data[weight_data > 0.0])
        pgt = numpy.angle(gain_data[weight_data > 0.0])
        rgt = self._obj.residual.data[numpy.sum(weight_data, axis=1) > 0.0]
        data = {
            "shape": self._obj.gain.shape,
            "maxabs-amp": numpy.max(agt),
            "minabs-amp": numpy.min(agt),
            "rms-amp": numpy.std(agt),
            "medianabs-amp": numpy.median(agt),
            "maxabs-phase": numpy.max(pgt),
            "minabs-phase": numpy.min(pgt),
            "rms-phase": numpy.std(pgt),
            "medianabs-phase": numpy.median(pgt),
            "residual": numpy.max(rgt),
        }
        qa = QualityAssessment(
            origin="qa_gain_table", data=data, context=context
        )
        return qa


class PointingTable(xarray.Dataset):
    """
    Pointing table with ska_sdp_datamodels:
    time, antenna, offset[:, chan, rec, 2], weight columns

    Here is an example::

        <xarray.PointingTable>
        Dimensions:    (angle: 2, antenna: 115, frequency: 3, receptor: 2, time: 3)
        Coordinates:
          * time       (time) float64 5.085e+09 5.085e+09 5.085e+09
          * antenna    (antenna) int64 0 1 2 3 4 5 6 7 ... 108 109 110 111 112 113 114
          * frequency  (frequency) float64 1e+08 1.05e+08 1.1e+08
          * receptor   (receptor) <U1 'X' 'Y'
          * angle      (angle) <U2 'az' 'el'
        Data variables:
            pointing   (time, antenna, frequency, receptor, angle) float64 -0.0002627...
            nominal    (time, antenna, frequency, receptor, angle) float64 -3.142 ......
            weight     (time, antenna, frequency, receptor, angle) float64 1.0 ... 1.0
            residual   (time, frequency, receptor, angle) float64 0.0 0.0 ... 0.0 0.0
            interval   (time) float64 99.73 99.73 99.73
            datetime   (time) datetime64[ns] 2000-01-01T03:54:07.843184299 ... 2000-0...
        Attributes:
            data_model:  PointingTable
            receptor_frame:     <src.ska_sdp_datamodels.polarisation.ReceptorFrame object...
            pointing_frame:     azel
            pointingcentre:     <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:      <xarray.Configuration> Dimensions:   (id: 115, spati...
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
        pointing: numpy.array = None,
        nominal: numpy.array = None,
        time: numpy.array = None,
        interval=None,
        weight: numpy.array = None,
        residual: numpy.array = None,
        frequency: numpy.array = None,
        receptor_frame: ReceptorFrame = ReceptorFrame("linear"),
        pointing_frame: str = "local",
        pointingcentre=None,
        configuration=None,
    ):
        """Create a pointing table from arrays

        :param pointing: Pointing (rad) [:, nants, nchan, nrec, 2]
        :param nominal: Nominal pointing (rad) [:, nants, nchan, nrec, 2]
        :param time: Centroid of solution [:]
        :param interval: Interval of validity
        :param weight: Weight [: nants, nchan, nrec]
        :param residual: Residual [: nants, nchan, nrec, 2]
        :param frequency: [nchan]
        :param receptor_frame: e.g. Receptor_frame("linear")
        :param pointing_frame: Pointing frame
        :param pointingcentre: SkyCoord
        :param configuration: Configuration
        """
        nants = pointing.shape[1]
        antennas = range(nants)

        coords = {
            "time": time,
            "antenna": antennas,
            "frequency": frequency,
            "receptor": receptor_frame.names,
            "angle": ["az", "el"],
        }

        datavars = {}
        datavars["pointing"] = xarray.DataArray(
            pointing,
            dims=["time", "antenna", "frequency", "receptor", "angle"],
        )
        datavars["nominal"] = xarray.DataArray(
            nominal, dims=["time", "antenna", "frequency", "receptor", "angle"]
        )
        datavars["weight"] = xarray.DataArray(
            weight, dims=["time", "antenna", "frequency", "receptor", "angle"]
        )
        datavars["residual"] = xarray.DataArray(
            residual, dims=["time", "frequency", "receptor", "angle"]
        )
        datavars["interval"] = xarray.DataArray(interval, dims=["time"])
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims="time",
        )

        attrs = {}
        attrs["data_model"] = "PointingTable"
        attrs["receptor_frame"] = receptor_frame
        attrs["pointing_frame"] = pointing_frame
        attrs["pointingcentre"] = pointingcentre
        attrs["configuration"] = configuration

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

    def copy(self, deep=False, data=None, zero=False):
        """
        Copy PointingTable

        :param deep: perform deep-copy
        :param data: data to use in new object; see docstring of
                     xarray.core.dataset.Dataset.copy
        :param zero: if True, set pointing data to zero in copied object
        """
        new_pointing_table = super().copy(deep=deep, data=data)
        if zero:
            new_pointing_table.data["pt"][...] = 0.0
        return new_pointing_table


@xarray.register_dataset_accessor("pointingtable_acc")
class PointingTableAccessor(XarrayAccessorMixin):
    """
    PointingTable property accessor
    """

    @property
    def nants(self):
        """Number of dishes/stations"""
        return self._obj["pointing"].shape[1]

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["pointing"].frequency)

    @property
    def nrec(self):
        """Number of receptors"""
        return self._obj.attrs["receptor_frame"].nrec

    def qa_pointing_table(self, context=None) -> QualityAssessment:
        """Assess the quality of a PointingTable

        :return: QualityAssessment
        """
        weight = self._obj.weight.data
        pointing = self._obj.pointing.data
        apt = numpy.abs(pointing[weight > 0.0])
        ppt = numpy.angle(pointing[weight > 0.0])
        data = {
            "shape": pointing.shape,
            "maxabs-amp": numpy.max(apt),
            "minabs-amp": numpy.min(apt),
            "rms-amp": numpy.std(apt),
            "medianabs-amp": numpy.median(apt),
            "maxabs-phase": numpy.max(ppt),
            "minabs-phase": numpy.min(ppt),
            "rms-phase": numpy.std(ppt),
            "medianabs-phase": numpy.median(ppt),
            "residual": numpy.max(self._obj.residual),
        }
        qa = QualityAssessment(
            origin="qa_pointingtable", data=data, context=context
        )
        return qa
