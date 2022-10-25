# pylint: disable=invalid-name

"""
Functions working with Configuration model.
"""

import numpy
from astropy.coordinates import EarthLocation
from astropy.units import Quantity

from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model import ReceptorFrame


def convert_configuration_to_hdf(config: Configuration, f):
    """Convert a Configuration to an HDF file

    :param config: Configuration
    :param f: hdf group
    :return: group with config added
    """

    def _convert_earthlocation_to_string(el: EarthLocation):
        """Convert Earth Location to string

        :param el: Earth Location
        :return: String
        """
        return f"{el.x}, {el.y}, {el.z}"

    if not isinstance(config, Configuration):
        raise ValueError(f"config is not a Configuration: {config}")
    if config.attrs["data_model"] != "Configuration":
        raise ValueError(f"config is not a Configuration: {config}")

    cf = f.create_group("configuration")
    cf.attrs["data_model"] = "Configuration"
    cf.attrs["name"] = config.name
    cf.attrs["location"] = _convert_earthlocation_to_string(config.location)
    cf.attrs["frame"] = config.frame
    cf.attrs["receptor_frame"] = config.receptor_frame.type

    cf["configuration/xyz"] = config.xyz
    cf["configuration/diameter"] = config.diameter
    cf["configuration/names"] = [numpy.string_(name) for name in config.names]
    cf["configuration/mount"] = [
        numpy.string_(mount) for mount in config.mount
    ]
    cf["configuration/offset"] = config.offset
    cf["configuration/stations"] = [
        numpy.string_(station) for station in config.stations
    ]
    cf["configuration/vp_type"] = [
        numpy.string_(vpt) for vpt in config.vp_type
    ]
    return f


def convert_configuration_from_hdf(f):
    """Extract configuration from HDF

    :param f: hdf group
    :return: Configuration
    """

    def _convert_earthlocation_from_string(s: str):
        """Convert Earth Location from string

        :param s: String
        :return: Earth Location
        """
        x, y, z = s.split(",")
        el = EarthLocation(x=Quantity(x), y=Quantity(y), z=Quantity(z))
        return el

    cf = f["configuration"]

    assert (
        cf.attrs["data_model"] == "Configuration"
    ), f"{cf.attrs['data_model']} is not a Configuration"

    name = cf.attrs["name"]
    location = _convert_earthlocation_from_string(cf.attrs["location"])
    receptor_frame = ReceptorFrame(cf.attrs["receptor_frame"])
    frame = cf.attrs["frame"]

    xyz = cf["configuration/xyz"]
    diameter = cf["configuration/diameter"]
    names = [str(n) for n in cf["configuration/names"]]
    mount = [str(m) for m in cf["configuration/mount"]]
    stations = [str(p) for p in cf["configuration/stations"]]
    vp_type = [str(p) for p in cf["configuration/vp_type"]]
    offset = cf["configuration/offset"]

    return Configuration.constructor(
        name=name,
        location=location,
        receptor_frame=receptor_frame,
        xyz=xyz,
        frame=frame,
        diameter=diameter,
        names=names,
        mount=mount,
        offset=offset,
        stations=stations,
        vp_type=vp_type,
    )
