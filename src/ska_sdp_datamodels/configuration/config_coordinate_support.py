# pylint: disable=invalid-name

"""
Coordinate support functions.
"""

import numpy
from astropy import units


def xyz_at_latitude(local_xyz, lat):
    """
    Rotate local XYZ coordinates into celestial XYZ coordinates. These
    coordinate systems are very similar, with X pointing towards the
    geographical east in both cases. However, before the rotation Z
    points towards the zenith, whereas afterwards it will point towards
    celestial north (parallel to the earth axis).

    :param lat: target latitude (radians or astropy quantity)
    :param local_xyz: Array of local XYZ coordinates
    :return: Celestial XYZ coordinates
    """
    # pylint: disable=unbalanced-tuple-unpacking
    x, y, z = numpy.hsplit(local_xyz, 3)

    lat2 = numpy.pi / 2 - lat
    y2 = -z * numpy.sin(lat2) + y * numpy.cos(lat2)
    z2 = z * numpy.cos(lat2) + y * numpy.sin(lat2)

    return numpy.hstack([x, y2, z2])


def xyz_to_uvw(xyz, ha, dec):
    """
    Rotate :math:`(x,y,z)` positions in earth coordinates to
    :math:`(u,v,w)` coordinates relative to astronomical source
    position :math:`(ha, dec)`. Can be used for both antenna positions
    as well as for baselines.

    Hour angle and declination can be given as single values or arrays
    of the same length. Angles can be given as radians or astropy
    quantities with a valid conversion.

    :param xyz: :math:`(x,y,z)` co-ordinates of antennas in array
    :param ha: hour angle of phase tracking centre (:math:`ha = ra - lst`)
    :param dec: declination of phase tracking centre.
    """

    # pylint: disable=unbalanced-tuple-unpacking
    x, y, z = numpy.hsplit(xyz, 3)

    # Two rotations:
    #  1. by 'ha' along the z axis
    #  2. by '90-dec' along the u axis
    u = x * numpy.cos(ha) - y * numpy.sin(ha)
    v0 = x * numpy.sin(ha) + y * numpy.cos(ha)
    w = z * numpy.sin(dec) - v0 * numpy.cos(dec)
    v = z * numpy.cos(dec) + v0 * numpy.sin(dec)

    return numpy.hstack([u, v, w])


def hadec_to_azel(ha, dec, latitude):
    """
    Convert HA Dec to Az El

    TMS Appendix 4.1

    sinel = sinlat sindec + coslat cosdec cosha
    cosel cosaz = coslat sindec - sinlat cosdec cosha
    cosel sinaz = - cosdec sinha

    :param ha: hour angle
    :param dec: declination
    :param latitude: latitude
    :return: az, el (azimuth, elevation)
    """
    coslat = numpy.cos(latitude)
    sinlat = numpy.sin(latitude)
    cosdec = numpy.cos(dec)
    sindec = numpy.sin(dec)
    cosha = numpy.cos(ha)
    sinha = numpy.sin(ha)

    az = numpy.arctan2(
        -cosdec * sinha, (coslat * sindec - sinlat * cosdec * cosha)
    )
    el = numpy.arcsin(sinlat * sindec + coslat * cosdec * cosha)
    return az, el


# pylint: disable=too-many-locals
def ecef_to_enu(location, xyz):
    """Convert ECEF coordinates to ENU coordinates
        relative to reference location.
    :param location: Current WGS84 coordinate
    :param xyz: ECEF coordinate
    :result : enu
    """
    # ECEF coordinates of reference point
    lon = location.geodetic[0].to(units.rad).value
    lat = location.geodetic[1].to(units.rad).value
    alt = location.geodetic[2].to(units.m).value
    # pylint: disable=unbalanced-tuple-unpacking
    x, y, z = numpy.hsplit(xyz, 3)

    center_x, center_y, center_z = lla_to_ecef(lat, lon, alt)

    delta_x, delta_y, delta_z = x - center_x, y - center_y, z - center_z
    sin_lat, cos_lat = numpy.sin(lat), numpy.cos(lat)
    sin_lon, cos_lon = numpy.sin(lon), numpy.cos(lon)

    e = -sin_lon * delta_x + cos_lon * delta_y
    n = (
        -sin_lat * cos_lon * delta_x
        - sin_lat * sin_lon * delta_y
        + cos_lat * delta_z
    )
    u = (
        cos_lat * cos_lon * delta_x
        + cos_lat * sin_lon * delta_y
        + sin_lat * delta_z
    )

    return numpy.hstack([e, n, u])


def lla_to_ecef(lat, lon, alt):
    """Convert WGS84 spherical coordinates to ECEF cartesian coordinates.
    :param lat: latitude
    :param lon: longitude
    :param alt: altitude
    :result: ecef coordinates
    """
    WGS84_a = 6378137.00000000
    WGS84_b = 6356752.31424518
    N = WGS84_a**2 / numpy.sqrt(
        WGS84_a**2 * numpy.cos(lat) ** 2 + WGS84_b**2 * numpy.sin(lat) ** 2
    )

    x = (N + alt) * numpy.cos(lat) * numpy.cos(lon)
    y = (N + alt) * numpy.cos(lat) * numpy.sin(lon)
    z = ((WGS84_b**2 / WGS84_a**2) * N + alt) * numpy.sin(lat)

    return x, y, z
