# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for coordinate transforms."""
import warnings

import numpy as np
from astropy.coordinates import EarthLocation

from . import _coordinates

try:
    from lunarsky import MoonLocation

    hasmoon = True
except ImportError:
    hasmoon = False

__all__ = [
    "LatLonAlt_from_XYZ",
    "XYZ_from_LatLonAlt",
    "rotECEF_from_ECEF",
    "ECEF_from_rotECEF",
    "ENU_from_ECEF",
    "ECEF_from_ENU",
]

allowed_location_types = [EarthLocation]
if hasmoon:
    selenoids = {
        "SPHERE": _coordinates.Body.Moon_sphere,
        "GSFC": _coordinates.Body.Moon_gsfc,
        "GRAIL23": _coordinates.Body.Moon_grail23,
        "CE-1-LAM-GEO": _coordinates.Body.Moon_ce1lamgeo,
    }
    allowed_location_types.append(MoonLocation)


_range_dict = {
    "itrs": (6.35e6, 6.39e6, "Earth"),
    "mcmf": (1717100.0, 1757100.0, "Moon"),
}


def LatLonAlt_from_XYZ(xyz, *, frame="ITRS", ellipsoid=None, check_acceptability=True):
    """
    Calculate lat/lon/alt from ECEF x,y,z.

    Parameters
    ----------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.
    frame : str
        Coordinate frame of xyz.
        Valid options are ITRS (default) or MCMF.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is MCMF.
    check_acceptability : bool
        Flag to check XYZ coordinates are reasonable.

    Returns
    -------
    latitude :  ndarray or float
        latitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    longitude :  ndarray or float
        longitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    altitude :  ndarray or float
        altitude, numpy array (if Npts > 1) or value (if Npts = 1) in meters

    """
    frame = frame.upper()
    if not hasmoon and frame == "MCMF":
        raise ValueError("Need to install `lunarsky` package to work with MCMF frame.")

    if frame == "ITRS":
        accept_bounds = (6.35e6, 6.39e6)
    elif frame == "MCMF":
        accept_bounds = (1.71e6, 1.75e6)
        if ellipsoid is None:
            ellipsoid = "SPHERE"

    # convert to a numpy array
    xyz = np.asarray(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        raise ValueError("The expected shape of ECEF xyz array is (Npts, 3).")

    squeeze = xyz.ndim == 1

    if squeeze:
        xyz = xyz[np.newaxis, :]

    xyz = np.ascontiguousarray(xyz.T, dtype=np.float64)

    # checking for acceptable values
    if check_acceptability:
        if frame not in ["ITRS", "MCMF"]:
            raise ValueError(f'Cannot check acceptability for unknown frame "{frame}".')
        norms = np.linalg.norm(xyz, axis=0)
        if not all(
            np.logical_and(norms >= accept_bounds[0], norms <= accept_bounds[1])
        ):
            raise ValueError(
                f"xyz values should be {frame} x, y, z coordinates in meters"
            )
    # this helper function returns one 2D array because it is less overhead for cython
    if frame == "ITRS":
        lla = _coordinates._lla_from_xyz(xyz, _coordinates.Body.Earth.value)
    elif frame == "MCMF":
        lla = _coordinates._lla_from_xyz(xyz, selenoids[ellipsoid].value)
    else:
        raise ValueError(
            f'No spherical to cartesian transform defined for frame "{frame}".'
        )

    if squeeze:
        return lla[0, 0], lla[1, 0], lla[2, 0]
    return lla[0], lla[1], lla[2]


def XYZ_from_LatLonAlt(latitude, longitude, altitude, *, frame="ITRS", ellipsoid=None):
    """
    Calculate ECEF x,y,z from lat/lon/alt values.

    Parameters
    ----------
    latitude :  ndarray or float
        latitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    longitude :  ndarray or float
        longitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    altitude :  ndarray or float
        altitude, numpy array (if Npts > 1) or value (if Npts = 1) in meters
    frame : str
        Coordinate frame of xyz.
        Valid options are ITRS (default) or MCMF.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is MCMF.

    Returns
    -------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.

    """
    latitude = np.ascontiguousarray(latitude, dtype=np.float64)
    longitude = np.ascontiguousarray(longitude, dtype=np.float64)
    altitude = np.ascontiguousarray(altitude, dtype=np.float64)

    n_pts = latitude.size

    frame = frame.upper()
    if not hasmoon and frame == "MCMF":
        raise ValueError("Need to install `lunarsky` package to work with MCMF frame.")

    if longitude.size != n_pts:
        raise ValueError(
            "latitude, longitude and altitude must all have the same length"
        )

    if altitude.size != n_pts:
        raise ValueError(
            "latitude, longitude and altitude must all have the same length"
        )

    if frame == "ITRS":
        xyz = _coordinates._xyz_from_latlonalt(
            latitude, longitude, altitude, _coordinates.Body.Earth.value
        )
    elif frame == "MCMF":
        if ellipsoid is None:
            ellipsoid = "SPHERE"

        xyz = _coordinates._xyz_from_latlonalt(
            latitude, longitude, altitude, selenoids[ellipsoid].value
        )
    else:
        raise ValueError(
            f'No cartesian to spherical transform defined for frame "{frame}".'
        )

    xyz = xyz.T
    if n_pts == 1:
        return xyz[0]

    return xyz


def rotECEF_from_ECEF(xyz, longitude):
    """
    Get rotated ECEF positions such that the x-axis goes through the longitude.

    Miriad and uvfits expect antenna positions in this frame
    (with longitude of the array center/telescope location)

    Parameters
    ----------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.
    longitude : float
        longitude in radians to rotate coordinates to
        (usually the array center/telescope location).

    Returns
    -------
    ndarray of float
        Rotated ECEF coordinates, shape (Npts, 3).

    """
    angle = -1 * longitude
    rot_matrix = np.array(
        [
            [np.cos(angle), -1 * np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return rot_matrix.dot(xyz.T).T


def ECEF_from_rotECEF(xyz, longitude):
    """
    Calculate ECEF from a rotated ECEF (Inverse of rotECEF_from_ECEF).

    Parameters
    ----------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with rotated ECEF x,y,z coordinates.
    longitude : float
        longitude in radians giving the x direction of the rotated coordinates
        (usually the array center/telescope location).

    Returns
    -------
    ndarray of float
        ECEF coordinates, shape (Npts, 3).

    """
    angle = longitude
    rot_matrix = np.array(
        [
            [np.cos(angle), -1 * np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return rot_matrix.dot(xyz.T).T


def ENU_from_ECEF(
    xyz,
    *,
    center_loc=None,
    latitude=None,
    longitude=None,
    altitude=None,
    frame="ITRS",
    ellipsoid=None,
):
    """
    Calculate local ENU (east, north, up) coordinates from ECEF coordinates.

    Parameters
    ----------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.
    center_loc : EarthLocation or MoonLocation object
        An EarthLocation or MoonLocation object giving the center of the ENU
        coordinates. Either `center_loc` or all of `latitude`, `longitude`,
        `altitude` must be passed.
    latitude : float
        Latitude of center of ENU coordinates in radians.
        Not used if `center_loc` is passed.
    longitude : float
        Longitude of center of ENU coordinates in radians.
        Not used if `center_loc` is passed.
    altitude : float
        Altitude of center of ENU coordinates in radians.
        Not used if `center_loc` is passed.
    frame : str
        Coordinate frame of xyz and center of ENU coordinates. Valid options are
        ITRS (default) or MCMF. Not used if `center_loc` is passed.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is MCMF. Not used if `center_loc` is passed.

    Returns
    -------
    ndarray of float
        numpy array, shape (Npts, 3), with local ENU coordinates

    """
    if center_loc is not None:
        if not isinstance(center_loc, tuple(allowed_location_types)):
            raise ValueError(
                "center_loc is not a supported type. It must be one of "
                f"{allowed_location_types}"
            )
        latitude = center_loc.lat.rad
        longitude = center_loc.lon.rad
        altitude = center_loc.height.to_value("m")
        if isinstance(center_loc, EarthLocation):
            frame = "ITRS"
        else:
            frame = "MCMF"
            ellipsoid = center_loc.ellipsoid
    else:
        if latitude is None or longitude is None or altitude is None:
            raise ValueError(
                "Either center_loc or all of latitude, longitude and altitude "
                "must be passed."
            )
        frame = frame.upper()
        if not hasmoon and frame == "MCMF":
            raise ValueError(
                "Need to install `lunarsky` package to work with MCMF frame."
            )

    if frame == "ITRS":
        sensible_radius_range = (6.35e6, 6.39e6)
        world = "earth"
    elif frame == "MCMF":
        world = "moon"
        sensible_radius_range = (1.71e6, 1.75e6)
        if ellipsoid is None:
            ellipsoid = "SPHERE"
    else:
        raise ValueError(f'No ENU_from_ECEF transform defined for frame "{frame}".')

    xyz = np.asarray(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        raise ValueError("The expected shape of ECEF xyz array is (Npts, 3).")

    squeeze = False
    if xyz.ndim == 1:
        squeeze = True
        xyz = xyz[np.newaxis, :]
    xyz = np.ascontiguousarray(xyz.T, dtype=np.float64)

    # check that these are sensible ECEF values -- their magnitudes need to be
    # on the order of Earth's radius
    ecef_magnitudes = np.linalg.norm(xyz, axis=0)
    if np.any(ecef_magnitudes <= sensible_radius_range[0]) or np.any(
        ecef_magnitudes >= sensible_radius_range[1]
    ):
        raise ValueError(
            f"{frame} vector magnitudes must be on the order"
            f" of the radius of the {world}"
        )

    # the cython utility expects (3, Npts) for faster manipulation
    # transpose after we get the array back to match the expected shape
    enu = _coordinates._ENU_from_ECEF(
        xyz,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
        # we have already forced the frame to conform to our options
        # and if we  don't have moon we have already errored.
        (
            _coordinates.Body.Earth.value
            if frame == "ITRS"
            else selenoids[ellipsoid].value
        ),
    )
    enu = enu.T

    if squeeze:
        enu = np.squeeze(enu)

    return enu


def ECEF_from_ENU(
    enu,
    center_loc=None,
    latitude=None,
    longitude=None,
    altitude=None,
    frame="ITRS",
    ellipsoid=None,
):
    """
    Calculate ECEF coordinates from local ENU (east, north, up) coordinates.

    Parameters
    ----------
    enu : ndarray of float
        numpy array, shape (Npts, 3), with local ENU coordinates.
    center_loc : EarthLocation or MoonLocation object
        An EarthLocation or MoonLocation object giving the center of the ENU
        coordinates. Either `center_loc` or all of `latitude`, `longitude`,
        `altitude` must be passed.
    latitude : float
        Latitude of center of ENU coordinates in radians.
        Not used if `center_loc` is passed.
    longitude : float
        Longitude of center of ENU coordinates in radians.
        Not used if `center_loc` is passed.
    altitude : float
        Altitude of center of ENU coordinates in radians.
        Not used if `center_loc` is passed.
    frame : str
        Coordinate frame of xyz and center of ENU coordinates. Valid options are
        ITRS (default) or MCMF. Not used if `center_loc` is passed.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is MCMF. Not used if `center_loc` is passed.

    Returns
    -------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.

    """
    if center_loc is not None:
        if not isinstance(center_loc, tuple(allowed_location_types)):
            raise ValueError(
                "center_loc is not a supported type. It must be one of "
                f"{allowed_location_types}"
            )
        latitude = center_loc.lat.rad
        longitude = center_loc.lon.rad
        altitude = center_loc.height.to_value("m")
        if isinstance(center_loc, EarthLocation):
            frame = "ITRS"
        else:
            frame = "MCMF"
            ellipsoid = center_loc.ellipsoid
    else:
        if latitude is None or longitude is None or altitude is None:
            raise ValueError(
                "Either center_loc or all of latitude, longitude and altitude "
                "must be passed."
            )
        frame = frame.upper()
        if not hasmoon and frame == "MCMF":
            raise ValueError(
                "Need to install `lunarsky` package to work with MCMF frame."
            )

        if frame not in ["ITRS", "MCMF"]:
            raise ValueError(f'No ECEF_from_ENU transform defined for frame "{frame}".')

        if frame == "MCMF" and ellipsoid is None:
            ellipsoid = "SPHERE"

    enu = np.asarray(enu)
    if enu.ndim > 1 and enu.shape[1] != 3:
        raise ValueError("The expected shape of the ENU array is (Npts, 3).")
    squeeze = False

    if enu.ndim == 1:
        squeeze = True
        enu = enu[np.newaxis, :]
    enu = np.ascontiguousarray(enu.T, dtype=np.float64)

    # the cython utility expects (3, Npts) for faster manipulation
    # transpose after we get the array back to match the expected shape
    xyz = _coordinates._ECEF_from_ENU(
        enu,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
        # we have already forced the frame to conform to our options
        # and if we  don't have moon we have already errored.
        (
            _coordinates.Body.Earth.value
            if frame == "ITRS"
            else selenoids[ellipsoid].value
        ),
    )
    xyz = xyz.T

    if squeeze:
        xyz = np.squeeze(xyz)

    return xyz


def check_surface_based_positions(
    *,
    telescope_loc=None,
    telescope_frame="itrs",
    antenna_positions=None,
    raise_error=True,
    raise_warning=True,
):
    """
    Check that antenna positions are consistent with ground-based values.

    Check that the antenna position, telescope location, or combination of both produces
    locations that are consistent with surface-based positions. If supplying both
    antenna position and telescope location, the check will be run against the sum total
    of both. For the Earth, the permitted range of values is betwen 6350 and 6390 km,
    whereas for theMoon the range is 1717.1 to 1757.1 km.

    telescope_loc : tuple or EarthLocation or MoonLocation
        Telescope location, specified as a 3-element tuple (specifying geo/selenocentric
        position in meters) or as an astropy EarthLocation (or lunarsky MoonLocation).
    telescope_frame : str, optional
        Reference frame for latitude/longitude/altitude. Options are itrs (default) or
        mcmf. Only used if telescope_loc is not an EarthLocation or MoonLocation.
    antenna_positions : ndarray of float
        List of antenna positions relative to array center in ECEF coordinates,
        required if not providing `uvw_array`. Shape is (Nants, 3). If no telescope_loc
        is specified, these values will be assumed to be relative to geocenter.
    raise_error : bool
        If True, an error is raised if telescope_loc and/or telescope_loc do not conform
        to expectations for a surface-based telescope. Default is True.
    raise_warning : bool
        If True, a warning is raised if telescope_loc and/or telescope_loc do not
        conform to expectations for a surface-based telescope. Default is True, only
        used if `raise_error` is set to False.

    Returns
    -------
    valid : bool
        If True, the antenna_positions and/or telescope_loc conform to expectations for
        a surface-based telescope. Otherwise returns false.

    """
    if antenna_positions is None:
        antenna_positions = np.zeros((1, 3))

    if isinstance(telescope_loc, EarthLocation) or (
        hasmoon and isinstance(telescope_loc, MoonLocation)
    ):
        antenna_positions = antenna_positions + (
            telescope_loc.x.to_value("m"),
            telescope_loc.y.to_value("m"),
            telescope_loc.z.to_value("m"),
        )
        if isinstance(telescope_loc, EarthLocation):
            telescope_frame = "itrs"
        else:
            telescope_frame = "mcmf"
    elif telescope_loc is not None:
        antenna_positions = antenna_positions + telescope_loc

    low_lim, hi_lim, world = _range_dict[telescope_frame]

    err_type = None
    if np.any(np.sum(antenna_positions**2.0, axis=1) < low_lim**2.0):
        err_type = "below"
    elif np.any(np.sum(antenna_positions**2.0, axis=1) > hi_lim**2.0):
        err_type = "above"

    if err_type is None:
        return True

    err_msg = (
        f"{telescope_frame} position vector magnitudes must be on the order of "
        f"the radius of {world} -- they appear to lie well {err_type} this."
    )

    # If desired, raise an error
    if raise_error:
        raise ValueError(err_msg)

    # Otherwise, if desired, raise a warning instead
    if raise_warning:
        warnings.warn(err_msg)

    return False
