# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for coordinate transforms."""

import warnings
from functools import cache

import numpy as np
from astropy import units
from astropy.coordinates import Angle, EarthLocation

from . import _coordinates

__all__ = [
    "LatLonAlt_from_XYZ",
    "XYZ_from_LatLonAlt",
    "rotECEF_from_ECEF",
    "ECEF_from_rotECEF",
    "ENU_from_ECEF",
    "ECEF_from_ENU",
]

_range_dict = {
    "itrs": (6.35e6, 6.39e6, "Earth"),
    "mcmf": (1717100.0, 1757100.0, "Moon"),
}


@cache
def get_selenoids():
    try:
        from lunarsky.moon import SELENOIDS

        return {
            key: _coordinates.Ellipsoid(
                SELENOIDS[key]._equatorial_radius.to_value("m"),
                SELENOIDS[key]._equatorial_radius.to_value("m")
                * (1 - SELENOIDS[key]._flattening),
            )
            for key in ["SPHERE", "GSFC", "GRAIL23", "CE-1-LAM-GEO"]
        }
    except ImportError as ie:
        raise ImportError(
            "Need to install `lunarsky` package to work with selenoids or MCMF frame."
        ) from ie


def get_loc_obj(
    telescope_loc,
    telescope_frame="itrs",
    ellipsoid=None,
    angle_units=units.rad,
    return_moon=False,
):
    """
    Check if telescope is on the moon.

    Parameters
    ----------
    telescope_loc : array-like of floats or EarthLocation or MoonLocation
        ITRS latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, or a tuple
        of shape (3,) containing (in order) the latitude, longitude, and altitude,
        in units of radians, radians, and meters, respectively.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.

    """
    on_moon = False
    loc_obj = None
    if isinstance(telescope_loc, EarthLocation):
        loc_obj = telescope_loc
    elif telescope_frame.upper() == "ITRS" and (
        isinstance(telescope_loc, tuple | list)
        or (isinstance(telescope_loc, np.ndarray) and telescope_loc.size > 1)
    ):
        # Moon Locations are np.ndarray instances but they have size 1
        loc_obj = EarthLocation.from_geodetic(
            lon=Angle(telescope_loc[1] * angle_units),
            lat=Angle(telescope_loc[0] * angle_units),
            height=telescope_loc[2],
        )
    else:
        try:
            from lunarsky import MoonLocation

            if isinstance(telescope_loc, MoonLocation):
                on_moon = True
                loc_obj = telescope_loc
            elif telescope_frame.upper() == "MCMF":
                on_moon = True
                if ellipsoid is None:
                    ellipsoid = "SPHERE"
                loc_obj = MoonLocation.from_selenodetic(
                    lon=Angle(telescope_loc[1] * angle_units),
                    lat=Angle(telescope_loc[0] * angle_units),
                    height=telescope_loc[2],
                    ellipsoid=ellipsoid,
                )
        except ImportError as ie:
            if telescope_frame.upper() == "MCMF":
                raise ImportError(
                    "Need to install `lunarsky` package to work with MCMF frame."
                ) from ie

    if return_moon:
        return loc_obj, on_moon
    else:
        return loc_obj


def get_frame_ellipsoid_loc_obj(loc_obj, loc_ob_parname="loc_obj"):
    if isinstance(loc_obj, EarthLocation):
        frame = "ITRS"
        ellipsoid = None
    else:
        allowed_location_types = [EarthLocation]
        loc_error = False
        try:
            from lunarsky import MoonLocation

            if isinstance(loc_obj, MoonLocation):
                frame = "MCMF"
                ellipsoid = loc_obj.ellipsoid
            else:
                allowed_location_types.append(MoonLocation)
                loc_error = True
        except ImportError:
            loc_error = True
        if loc_error:
            raise ValueError(
                f"{loc_ob_parname} is not a supported type. It must be one of "
                f"{allowed_location_types}"
            )
    return frame, ellipsoid


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
    if frame == "ITRS":
        accept_bounds = (6.35e6, 6.39e6)
    elif frame == "MCMF":
        accept_bounds = (1.71e6, 1.75e6)
        if ellipsoid is None:
            ellipsoid = "SPHERE"
        selenoids = get_selenoids()

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
        lla = _coordinates._lla_from_xyz(xyz, _coordinates.Earth)
    elif frame == "MCMF":
        lla = _coordinates._lla_from_xyz(xyz, selenoids[ellipsoid])
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
            latitude, longitude, altitude, _coordinates.Earth
        )
    elif frame == "MCMF":
        if ellipsoid is None:
            ellipsoid = "SPHERE"
        selenoids = get_selenoids()
        xyz = _coordinates._xyz_from_latlonalt(
            latitude, longitude, altitude, selenoids[ellipsoid]
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
        frame, ellipsoid = get_frame_ellipsoid_loc_obj(center_loc, "center_loc")

        latitude = center_loc.lat.rad
        longitude = center_loc.lon.rad
        altitude = center_loc.height.to_value("m")
    else:
        if latitude is None or longitude is None or altitude is None:
            raise ValueError(
                "Either center_loc or all of latitude, longitude and altitude "
                "must be passed."
            )
        frame = frame.upper()

    if frame.lower() in _range_dict:
        if frame == "MCMF" and ellipsoid is None:
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
    check_surface_based_positions(
        telescope_loc=None,
        telescope_frame=frame.lower(),
        antenna_positions=xyz[np.newaxis, :],
    )

    # the cython utility expects (3, Npts) for faster manipulation
    # transpose after we get the array back to match the expected shape
    if frame == "ITRS":
        body = _coordinates.Earth
    else:
        # we have already forced the frame to conform to our options
        # and if we  don't have moon we have already errored.
        selenoids = get_selenoids()
        body = selenoids[ellipsoid]
    enu = _coordinates._ENU_from_ECEF(
        xyz,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
        body,
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
        frame, ellipsoid = get_frame_ellipsoid_loc_obj(center_loc, "center_loc")

        latitude = center_loc.lat.rad
        longitude = center_loc.lon.rad
        altitude = center_loc.height.to_value("m")
    else:
        if latitude is None or longitude is None or altitude is None:
            raise ValueError(
                "Either center_loc or all of latitude, longitude and altitude "
                "must be passed."
            )
        frame = frame.upper()

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
    if frame == "ITRS":
        body = _coordinates.Earth
    else:
        # we have already forced the frame to conform to our options
        # and if we  don't have moon we have already errored.
        selenoids = get_selenoids()
        body = selenoids[ellipsoid]
    xyz = _coordinates._ECEF_from_ENU(
        enu,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
        body,
    )
    xyz = xyz.T

    if squeeze:
        xyz = np.squeeze(xyz)

    return xyz


def hpx_latlon_to_zenithangle_azimuth(hpx_lat, hpx_lon):
    """
    Convert from healpix lat/lon to UVBeam za/az convention.

    Note that this is different (unfortunately) from the conversion between
    the UVBeam Zenith Angle, Azimuth coordinate system and the astropy Alt/Az
    coordinate system. The astropy Azimuth runs the opposite direction and
    has a different starting angle than UVBeam's Azimuth because they are both
    right handed coordinate systems but Altitude moves the opposite direction
    than Zenith Angle does.

    The conversion used in this code sets the Healpix latitude to 90-zenith angle
    but it does not change the origin or direction for the azimuthal angle. This
    convention was set early in the development of UVBeam and we preserve it to
    preserve backwards compatibility.

    Parameters
    ----------
    hpx_lat: float or array of float
        Healpix latiudinal coordinate in radians.
    hpx_lon: float or array of float
        Healpix longitudinal coordinate in radians.

    Returns
    -------
    zenith_angle: float or array of float
        In radians
    azimuth: float or array of float
        In radians in uvbeam convention: North of East(East=0, North=pi/2)

    """
    input_alt = np.asarray(hpx_lat)
    input_az = np.asarray(hpx_lon)
    if input_alt.shape != input_az.shape:
        raise ValueError("shapes of hpx_lat and hpx_lon values must match.")

    zenith_angle = np.pi / 2 - hpx_lat
    azimuth = hpx_lon

    return zenith_angle, azimuth


def zenithangle_azimuth_to_hpx_latlon(zenith_angle, azimuth):
    """
    Convert from UVBeam az/za convention to healpix lat/lon.

    Note that this is different (unfortunately) from the conversion between
    the UVBeam Zenith Angle, Azimuth coordinate system and the astropy Alt/Az
    coordinate system. The astropy Azimuth runs the opposite direction and
    has a different starting angle than UVBeam's Azimuth because they are both
    right handed coordinate systems but Altitude moves the opposite direction
    than Zenith Angle does.

    The conversion used in this code sets the Healpix latitude to 90-zenith angle
    but it does not change the origin or direction for the azimuthal angle. This
    convention was set early in the development of UVBeam and we preserve it to
    preserve backwards compatibility.

    Parameters
    ----------
    zenith_angle: float, array_like of float
        Zenith angle in radians
    azimuth: float, array_like of float
        Azimuth in radians in uvbeam convention: North of East(East=0, North=pi/2)

    Returns
    -------
    hpx_lat: float or array of float
        Healpix latiudinal coordinate in radians.
    hpx_lon: float or array of float
        Healpix longitudinal coordinate in radians.

    """
    input_za = np.array(zenith_angle)
    input_az = np.array(azimuth)
    if input_za.shape != input_az.shape:
        raise ValueError("shapes of zenith_angle and azimuth values must match.")

    lat_array = np.pi / 2 - zenith_angle
    lon_array = azimuth

    return lat_array, lon_array


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

    telescope_loc : array-like of float or EarthLocation or MoonLocation
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

    if telescope_loc is not None:
        tel_object = False
        if isinstance(telescope_loc, EarthLocation):
            tel_object = True
            telescope_frame = "itrs"
        elif isinstance(telescope_loc, tuple | list) or (
            isinstance(telescope_loc, np.ndarray) and telescope_loc.size > 1
        ):
            # Moon Locations are np.ndarray instances but they have size 1

            antenna_positions = antenna_positions + telescope_loc
        else:
            try:
                from lunarsky import MoonLocation

                if isinstance(telescope_loc, MoonLocation):
                    tel_object = True
                    telescope_frame = "mcmf"
            except ImportError as ie:
                raise ImportError(
                    "Need to install `lunarsky` package to work with MoonLocations."
                ) from ie

        if tel_object:
            antenna_positions = antenna_positions + (
                telescope_loc.x.to_value("m"),
                telescope_loc.y.to_value("m"),
                telescope_loc.z.to_value("m"),
            )

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
