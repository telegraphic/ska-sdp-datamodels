# pylint: disable=invalid-name

"""
Functions converting from and to Configuration data model.
"""

import json

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
    cf["configuration/names"] = [
        numpy.string_(name, encoding="utf-8") for name in config.names.data
    ]
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


def convert_configuration_to_json(config: Configuration):
    """Convert configuration to JSON
    :param config: Configuration
    :return: JSON
    """
    cf_dict = {}
    cf_dict["data_model"] = "Configuration"
    cf_dict["name"] = config.name
    cf_dict[
        "location"
    ] = f"{config.location.x}, {config.location.y}, {config.location.z}"
    cf_dict["frame"] = config.frame
    cf_dict["receptor_frame"] = config.receptor_frame.type

    cf_dict["configuration"] = {}
    cf_dict["configuration"]["xyz"] = config.xyz.data.tolist()
    cf_dict["configuration"]["diameter"] = config.diameter.data.tolist()
    cf_dict["configuration"]["names"] = config.names.data.tolist()
    cf_dict["configuration"]["mount"] = config.mount.data.tolist()
    cf_dict["configuration"]["offset"] = config.offset.data.tolist()
    cf_dict["configuration"]["stations"] = config.stations.data.tolist()
    cf_dict["configuration"]["vp_type"] = config.vp_type.data.tolist()
    return json.dumps(cf_dict)


def convert_json_to_configuration(json_str):
    """Convert configuration from json string to Configuration
    :param json_str: json string
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

    cf_dict = json.loads(json_str)
    name = cf_dict["name"]
    location = _convert_earthlocation_from_string(cf_dict["location"])
    receptor_frame = ReceptorFrame(cf_dict["receptor_frame"])
    frame = cf_dict["frame"]

    cf_config_dict = cf_dict["configuration"]
    xyz = numpy.array(cf_config_dict["xyz"])
    diameter = numpy.array(cf_config_dict["diameter"])
    names = cf_config_dict["names"]
    mount = cf_config_dict["mount"]
    stations = cf_config_dict["stations"]
    vp_type = cf_config_dict["vp_type"]
    offset = cf_config_dict["offset"]
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


def convert_configuration_from_hdf(f):
    """Extract configuration from HDF

    :param f: hdf group
    :return: Configuration
    """

    cf = f["configuration"]

    assert (
        cf.attrs["data_model"] == "Configuration"
    ), f"{cf.attrs['data_model']} is not a Configuration"

    name = cf.attrs["name"]
    x, y, z = cf.attrs["location"].split(",")
    location = EarthLocation(x=Quantity(x), y=Quantity(y), z=Quantity(z))
    receptor_frame = ReceptorFrame(cf.attrs["receptor_frame"])
    frame = cf.attrs["frame"]

    xyz = cf["configuration/xyz"]
    diameter = cf["configuration/diameter"]
    names = [str(n, encoding="utf-8") for n in cf["configuration/names"]]
    mount = [str(m, encoding="utf-8") for m in cf["configuration/mount"]]
    stations = [str(p, encoding="utf-8") for p in cf["configuration/stations"]]
    vp_type = [str(p, encoding="utf-8") for p in cf["configuration/vp_type"]]
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
