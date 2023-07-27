"""This module contains an encode function to convert xarray.Dataset based
objects into a msgpack bytes object. Note that currently the decode does not
convert directly back to what the original object was."""

import msgpack
import msgpack_numpy

try:
    import pyarrow

    PYARROW_AVAILABLE = True
except ModuleNotFoundError:
    PYARROW_AVAILABLE = False
import xarray
from astropy.coordinates import EarthLocation, SkyCoord

from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.science_data_model import ReceptorFrame


def _dataset_encoder(obj):
    """Custom encoder for datamodel specific code."""

    # Dataset to dict conversion:
    if isinstance(obj, xarray.Dataset):
        if "configuration" in obj.attrs and isinstance(
            obj.attrs["configuration"], Configuration
        ):
            obj.attrs["configuration"].attrs["location"] = obj.attrs[
                "configuration"
            ].location.value.tolist()
        out_dict = obj.to_dict(data="array")

        # we need to remove the datetime key, as this cannot be encoded
        if "datetime" in out_dict["data_vars"]:
            del out_dict["data_vars"]["datetime"]

        return out_dict

    # A method to get the ReceptorFrame object to encode
    if isinstance(obj, ReceptorFrame):
        return obj.names

    # A method to get the SkyCoord to convert
    if isinstance(obj, SkyCoord):
        return obj.to_string()

    # A method to get the EarthLocation to convert
    if isinstance(obj, EarthLocation):
        return obj.value.tolist()

    # Convert the Table type to a list
    if PYARROW_AVAILABLE and isinstance(obj, pyarrow.lib.Table):
        return obj.to_pylist()

    # Default to attempting to convert assuming numpy data
    return msgpack_numpy.encode(obj)


def encode(dataset: xarray.Dataset) -> bytes:
    """Encode a Dataset object into a msgpack bytes."""
    return msgpack.packb(dataset, default=_dataset_encoder)


def decode(bytes_dataset: bytes) -> xarray.Dataset:
    """Decode a msgpack bytes to a Dataset object."""
    data_raw = msgpack.unpackb(bytes_dataset, object_hook=msgpack_numpy.decode)
    return xarray.Dataset.from_dict(data_raw)
