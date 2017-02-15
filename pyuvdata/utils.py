"""Commonly used utility functions."""
import numpy as np

# parameters for transforming between xyz & lat/lon/alt
gps_b = 6356752.31424518
gps_a = 6378137
e_squared = 6.69437999014e-3
e_prime_squared = 6.73949674228e-3


def LatLonAlt_from_XYZ(xyz):
    """
    Calculate lat/lon/alt from topocentric x,y,z.

    Args:
        xyz: numpy array, shape (3,), with topocentric x,y,z coordinates

    Returns:
        tuple of latitude, longitude, altitude values in radians & meters
    """
    # checking for acceptable values
    if np.linalg.norm(xyz) < 6.35e6 or np.linalg.norm(xyz) > 6.39e6:
        raise ValueError('xyz values should be topocentric x, y, z coordinates in meters')

    # see wikipedia geodetic_datum and Datum transformations of
    # GPS positions PDF in docs/references folder
    gps_p = np.sqrt(xyz[0]**2 + xyz[1]**2)
    gps_theta = np.arctan2(xyz[2] * gps_a, gps_p * gps_b)
    latitude = np.arctan2(xyz[2] + e_prime_squared * gps_b *
                          np.sin(gps_theta)**3, gps_p - e_squared * gps_a *
                          np.cos(gps_theta)**3)

    longitude = np.arctan2(xyz[1], xyz[0])
    gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
    altitude = ((gps_p / np.cos(latitude)) - gps_N)
    return latitude, longitude, altitude


def XYZ_from_LatLonAlt(latitude, longitude, altitude):
    """
    Calculate topocentric x,y,z from lat/lon/alt values.

    Args:
        latitude: latitude in radians
        longitude: longitude in radians
        altitude: altitude in meters

    Returns:
        numpy array, shape (3,), with topocentric x,y,z coordinates
    """
    # see wikipedia geodetic_datum and Datum transformations of
    # GPS positions PDF in docs/references folder
    gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
    xyz = np.zeros(3)
    xyz[0] = ((gps_N + altitude) * np.cos(latitude) * np.cos(longitude))
    xyz[1] = ((gps_N + altitude) * np.cos(latitude) * np.sin(longitude))
    xyz[2] = ((gps_b**2 / gps_a**2 * gps_N + altitude) * np.sin(latitude))

    return xyz
