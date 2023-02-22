"""
Unit tests for creating Configuration and related functions
"""

import numpy
import pytest
from astropy.coordinates import EarthLocation

from ska_sdp_datamodels.configuration.config_create import (
    ANTENNA_FILES,
    _find_vptype_from_name,
    _limit_rmax,
    create_configuration_from_file,
    create_named_configuration,
    decimate_configuration,
    select_configuration,
)

# pylint: disable=duplicate-code
LOW_AA05_NAMES = [
    "S008‐1",
    "S008‐2",
    "S009‐1",
    "S009‐2",
    "S010‐1",
    "S010‐2",
]


@pytest.mark.parametrize("rmax", [200.0, None])
def test_limit_rmax(rmax):
    """
    Two tested cases:
        1. In this tested case, out of the 6 antennas
        there are 2 which are within 200 m (rmax) from
        the centre of the array composed of the 6 antennas.
        2. When rmax is None, the original data are returned
    """
    xyz_coords = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [30.0, 50.0, -0.5],
            [130.0, 76.0, -10.2],
            [350.0, 200.0, -100.2],
            [440.0, 250.0, -163.2],
            [550.0, 340.0, -230.0],
        ]
    )
    diameters = numpy.array([38.0] * 6)
    names = numpy.array(["SKA1", "SKA2", "SKA3", "SKA4", "SKA5", "SKA6"])
    mounts = numpy.array(["XY"] * 6)

    result = _limit_rmax(xyz_coords, diameters, names, mounts, rmax)

    if rmax is not None:
        assert (
            result[0]
            == numpy.array([[130.0, 76.0, -10.2], [350.0, 200.0, -100.2]])
        ).all()  # xyz
        assert (result[1] == [38.0, 38.0]).all()  # diameters
        assert (result[2] == ["SKA3", "SKA4"]).all()  # names
        assert (result[3] == ["XY", "XY"]).all()  # mounts

    else:
        assert (result[0] == xyz_coords).all()
        assert (result[1] == diameters).all()
        assert (result[2] == names).all()
        assert (result[3] == mounts).all()


def test_find_vptype_from_name():
    """
    vp_type determined from a name-vp_type mapping dictionary.
    """
    names = ["M0001", "M0002", "SKA1", "SKA2", "M0003"]
    vp_map = {"M0": "MEERKAT", "SK": "MID"}
    result = _find_vptype_from_name(names, vp_map)

    assert (result == ["MEERKAT", "MEERKAT", "MID", "MID", "MEERKAT"]).all()


def test_find_vptype_from_name_str_match():
    """
    The returned vp_type for each name is the
    privded "match" value, which is a string.
    """
    names = ["M0001", "M0002", "SKA1", "SKA2", "M0003"]
    vp_map = "TEST"
    result = _find_vptype_from_name(names, vp_map)

    assert (result == ["TEST"] * 5).all()


def test_create_configuration_from_file():
    """
    Configuration correctly generated based on
    input values.
    """
    ant_file = ANTENNA_FILES["LOWBD2"]
    result = create_configuration_from_file(
        ant_file, name="LOWBD2", ecef=False
    )

    assert result.name == "LOWBD2"
    assert result.attrs["location"] is None
    assert (
        result.vp_type == "Unknown"
    ).all()  # we didn't specify vp_type on input


def test_create_configuration_from_file_location():
    """
    Location is correctly added, when provided;
    Also, if ecef=True, it only succeeds if location is
    provided as EarthLocation.
    """
    ant_file = ANTENNA_FILES["LOWBD2"]
    location = EarthLocation(lon=60.0, lat=-48.0, height=0.0)
    result = create_configuration_from_file(
        ant_file, name="LOWBD2", ecef=True, location=location
    )

    assert result.attrs["location"] == location


def test_create_configuration_from_file_vp_type():
    """
    When voltage pattern type is set, that is what the
    Configration will be returned with.
    """
    ant_file = ANTENNA_FILES["LOWBD2"]
    result = create_configuration_from_file(
        ant_file, name="LOWBD2", ecef=False, vp_type="LOW"
    )

    assert (result.vp_type == "LOW").all()


def test_create_configuration_from_file_ecef_no_loc():
    """
    When location is not set, but ecef is set to True,
    raise ValueError.
    """
    ant_file = ANTENNA_FILES["LOWBD2"]
    with pytest.raises(ValueError):
        create_configuration_from_file(ant_file, name="LOWBD2", ecef=True)


@pytest.mark.parametrize(
    "config_name, exp_vp_type, exp_name, exp_mount",
    [
        ["LOWBD2", "LOW", "LOWBD2_", "XY"],
        ["LOWBD2-CORE", "LOW", "LOWBD2_", "XY"],
        ["LOW", "LOW", "SKA", "XY"],
        ["LOWR3", "LOW", "SKA", "XY"],
        ["LOWR4", "LOW", "SKA", "XY"],
        ["LOW-AA0.5", "LOW", "S0", "XY"],
        [
            "MID",
            {"M0": "MEERKAT", "SK": "MID"},
            {"M0": "MEERKAT", "SK": "MID"},
            "azel",
        ],
        [
            "MIDR5",
            {"M0": "MEERKAT", "SK": "MID"},
            {"M0": "MEERKAT", "SK": "MID"},
            "azel",
        ],
        [
            "MID-AA0.5",
            {"M0": "MEERKAT", "SK": "MID"},
            {"M0": "MEERKAT", "SKA": "MID"},
            "azel",
        ],
        [
            "MEERKAT+",
            {"m0": "MEERKAT", "s0": "MID"},
            {"m0": "MEERKAT", "s0": "MID"},
            "ALT-AZ",
        ],
        ["ASKAP", "ASKAP", "antennas.A27", "equatorial"],
        [
            "LOFAR",
            "LOFAR",
            "",
            "XY",
        ],  # too many variations of antenna names, not testing it
        ["VLAA", "VLA", "VLA_", "AZEL"],
        ["VLAA_north", "VLA", "VLA_", "AZEL"],
    ],
)
def test_create_named_configuration(
    config_name, exp_vp_type, exp_name, exp_mount
):
    """
    Test that the correct antenna configuration is loaded and
    check some of the values that the Config should have.

    Apart from the LOW and VLA configurations, the names
    are taken from the loaded cfg files.

    :param config_name: input configuration name
    :param exp_vp_type: expected voltage pattern type
    :param exp_name: expected antenna name pattern
    :param exp_mount: expected mounts
    """
    result = create_named_configuration(config_name)

    assert result.name == config_name
    assert (result.mount == exp_mount).all()

    if "MID" in config_name or "MEERKAT" in config_name:
        for i, name in enumerate(result.names.data):
            assert any(name.startswith(key) for key in exp_name.keys())
            assert result.vp_type[i] == exp_vp_type[name[:2]]

    else:
        assert (result.vp_type == exp_vp_type).all()
        for name in result.names.data:
            assert exp_name in name


def test_select_configuration(low_aa05_config):
    """
    Configuration is sub-selected based on names.
    """
    assert (low_aa05_config.names.data == LOW_AA05_NAMES).all()

    result = select_configuration(low_aa05_config, names=["S009‐2", "S008‐2"])
    assert (result.names.data == ["S008‐2", "S009‐2"]).all()


def test_select_configuration_no_select(low_aa05_config):
    """
    If no names to sub-select based on, then
    the function returns the original configuration.
    """
    assert (low_aa05_config.names.data == LOW_AA05_NAMES).all()

    result = select_configuration(low_aa05_config, names=None)
    assert (result.names.data == low_aa05_config.names.data).all()


@pytest.mark.parametrize(
    "start, stop, skip, expected_names",
    [
        [
            0,
            -1,
            1,
            LOW_AA05_NAMES[:-1],
        ],
        [0, -1, 2, ["S008‐1", "S009‐1", "S010‐1"]],
        [2, -1, 1, ["S009‐1", "S009‐2", "S010‐1"]],
        [1, None, 3, ["S008‐2", "S010‐1"]],
        [3, None, 1, ["S009‐2", "S010‐1", "S010‐2"]],
    ],
)
def test_decimate_configuration(
    low_aa05_config, start, stop, skip, expected_names
):
    """
    Select configuration based on name indexes and
    whether some need to be skipped or not
    """
    result = decimate_configuration(
        low_aa05_config, start=start, stop=stop, skip=skip
    )
    assert (result.names.data == expected_names).all()
