# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Commonly used utility functions.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import collections
import six
import warnings
import copy
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.utils import iers

# parameters for transforming between xyz & lat/lon/alt
gps_b = 6356752.31424518
gps_a = 6378137
e_squared = 6.69437999014e-3
e_prime_squared = 6.73949674228e-3


if six.PY2:
    def _str_to_bytes(s):
        return s

    def _bytes_to_str(b):
        return b
else:
    def _str_to_bytes(s):
        return s.encode('utf8')

    def _bytes_to_str(b):
        return b.decode('utf8')

# polarization constants
# maps polarization strings to polarization integers
POL_STR2NUM_DICT = {'pI': 1, 'pQ': 2, 'pU': 3, 'pV': 4,
                    'I': 1, 'Q': 2, 'U': 3, 'V': 4,  # support straight stokes names
                    'rr': -1, 'll': -2, 'rl': -3, 'lr': -4,
                    'xx': -5, 'yy': -6, 'xy': -7, 'yx': -8}
# maps polarization integers to polarization strings
POL_NUM2STR_DICT = {1: 'pI', 2: 'pQ', 3: 'pU', 4: 'pV',
                    -1: 'rr', -2: 'll', -3: 'rl', -4: 'lr',
                    -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'}

# maps how polarizations change when antennas are swapped
CONJ_POL_DICT = {'xx': 'xx', 'yy': 'yy', 'xy': 'yx', 'yx': 'xy',
                 'rr': 'rr', 'll': 'll', 'rl': 'lr', 'lr': 'rl',
                 'I': 'I', 'Q': 'Q', 'U': 'U', 'V': 'V',
                 'pI': 'pI', 'pQ': 'pQ', 'pU': 'pU', 'pV': 'pV'}

# maps jones matrix element strings to jones integers
JONES_STR2NUM_DICT = {'Jxx': -5, 'Jyy': -6, 'Jxy': -7, 'Jyx': -8,
                      'xx': -5, 'x': -5, 'yy': -6, 'y': -6, 'xy': -7, 'yx': -8,  # Allow shorthand
                      'Jrr': -1, 'Jll': -2, 'Jrl': -3, 'Jlr': -4,
                      'rr': -1, 'r': -1, 'll': -2, 'l': -2, 'rl': -3, 'lr': -4}
# maps jones integers to jones matrix element strings
JONES_NUM2STR_DICT = {-1: 'Jrr', -2: 'Jll', -3: 'Jrl', -4: 'Jlr',
                      -5: 'Jxx', -6: 'Jyy', -7: 'Jxy', -8: 'Jyx'}


def LatLonAlt_from_XYZ(xyz):
    """
    Calculate lat/lon/alt from ECEF x,y,z.

    Args:
        xyz: numpy array, shape (Npts, 3), with ECEF x,y,z coordinates

    Returns:
        tuple of latitude, longitude, altitude numpy arrays (if Npts > 1) or
            values (if Npts = 1) in radians & meters
    """
    # convert to a numpy array
    xyz = np.array(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        if xyz.shape[0] == 3:
            warnings.warn('The expected shape of ECEF xyz array is (Npts, 3). '
                          'Support for arrays shaped (3, Npts) will go away in '
                          'version 1.5', DeprecationWarning)
            xyz_use = xyz.T
        else:
            raise ValueError('The expected shape of ECEF xyz array is (Npts, 3).')

    else:
        xyz_use = xyz

    if xyz.shape == (3, 3):
        warnings.warn('The xyz array in LatLonAlt_from_XYZ is being '
                      'interpreted as (Npts, 3). Historically this function '
                      'has supported (3, Npts) arrays, please verify that '
                      'array ordering is as expected. This warning will be '
                      'removed in version 1.5', DeprecationWarning)

    if xyz_use.ndim == 1:
        xyz_use = xyz_use[np.newaxis, :]

    # checking for acceptable values
    if (np.any(np.linalg.norm(xyz_use, axis=1) < 6.35e6)
            or np.any(np.linalg.norm(xyz_use, axis=1) > 6.39e6)):
        raise ValueError(
            'xyz values should be ECEF x, y, z coordinates in meters')

    # see wikipedia geodetic_datum and Datum transformations of
    # GPS positions PDF in docs/references folder
    gps_p = np.sqrt(xyz_use[:, 0]**2 + xyz_use[:, 1]**2)
    gps_theta = np.arctan2(xyz_use[:, 2] * gps_a, gps_p * gps_b)
    latitude = np.arctan2(xyz_use[:, 2] + e_prime_squared * gps_b
                          * np.sin(gps_theta)**3, gps_p - e_squared * gps_a
                          * np.cos(gps_theta)**3)

    longitude = np.arctan2(xyz_use[:, 1], xyz_use[:, 0])
    gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
    altitude = ((gps_p / np.cos(latitude)) - gps_N)

    if xyz.ndim == 1:
        longitude = longitude[0]
        latitude = latitude[0]
        altitude = altitude[0]
    return latitude, longitude, altitude


def XYZ_from_LatLonAlt(latitude, longitude, altitude):
    """
    Calculate ECEF x,y,z from lat/lon/alt values.

    Args:
        latitude: latitude in radians, can be a single value or a vector of length Npts
        longitude: longitude in radians, can be a single value or a vector of length Npts
        altitude: altitude in meters, can be a single value or a vector of length Npts

    Returns:
        numpy array, shape (Npts, 3) (if Npts > 1) or (3,) (if Npts = 1), with ECEF x,y,z coordinates
    """
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    altitude = np.array(altitude)
    Npts = latitude.size
    if longitude.size != Npts:
        raise ValueError(
            'latitude, longitude and altitude must all have the same length')
    if altitude.size != Npts:
        raise ValueError(
            'latitude, longitude and altitude must all have the same length')

    # see wikipedia geodetic_datum and Datum transformations of
    # GPS positions PDF in docs/references folder
    gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
    xyz = np.zeros((Npts, 3))
    xyz[:, 0] = ((gps_N + altitude) * np.cos(latitude) * np.cos(longitude))
    xyz[:, 1] = ((gps_N + altitude) * np.cos(latitude) * np.sin(longitude))
    xyz[:, 2] = ((gps_b**2 / gps_a**2 * gps_N + altitude) * np.sin(latitude))

    xyz = np.squeeze(xyz)
    return xyz


def rotECEF_from_ECEF(xyz, longitude):
    """
    Calculate a rotated ECEF from ECEF such that the x-axis goes through the
    specified longitude.

    Miriad (and maybe uvfits) expect antenna positions in this frame
    (with longitude of the array center/telescope location)

    Args:
        xyz: numpy array, shape (Npts, 3), with ECEF x,y,z coordinates
        longitude: longitude in radians to rotate coordinates to (usually the array center/telescope location)
    Returns:
        numpy array, shape (Npts, 3), with rotated ECEF coordinates
    """
    angle = -1 * longitude
    rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    return rot_matrix.dot(xyz.T).T


def ECEF_from_rotECEF(xyz, longitude):
    """
    Calculate ECEF from a rotated ECEF such that the x-axis goes through the
    specified longitude. (Inverse of rotECEF_from_ECEF)

    Args:
        xyz: numpy array, shape (Npts, 3), with rotated ECEF x,y,z coordinates
        longitude: longitude in radians to rotate coordinates to (usually the array center/telescope location)
    Returns:
        numpy array, shape (Npts, 3), with ECEF coordinates
    """
    angle = longitude
    rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    return rot_matrix.dot(xyz.T).T


def ENU_from_ECEF(xyz, latitude, longitude, altitude):
    """
    Calculate local ENU (east, north, up) coordinates from ECEF coordinates.

    Args:
        xyz: numpy array, shape (Npts, 3), with ECEF x,y,z coordinates
        latitude: latitude of center of ENU coordinates in radians
        longitude: longitude of center of ENU coordinates in radians
        altitude: altitude of center of ENU coordinates in radians

    Returns:
        numpy array, shape (Npts, 3), with local ENU coordinates
    """
    xyz = np.array(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        if xyz.shape[0] == 3:
            warnings.warn('The expected shape of ECEF xyz array is (Npts, 3). '
                          'Support for arrays shaped (3, Npts) will go away in '
                          'version 1.5', DeprecationWarning)
            xyz_in = xyz.T
            transpose = True
        else:
            raise ValueError('The expected shape of ECEF xyz array is (Npts, 3).')
    else:
        xyz_in = xyz
        transpose = False

    if xyz.shape == (3, 3):
        warnings.warn('The xyz array in ENU_from_ECEF is being '
                      'interpreted as (Npts, 3). Historically this function '
                      'has supported (3, Npts) arrays, please verify that '
                      'array ordering is as expected. This warning will be '
                      'removed in version 1.5', DeprecationWarning)

    if xyz_in.ndim == 1:
        xyz_in = xyz_in[np.newaxis, :]

    # check that these are sensible ECEF values -- their magnitudes need to be
    # on the order of Earth's radius
    ecef_magnitudes = np.linalg.norm(xyz_in, axis=1)
    sensible_radius_range = (6.35e6, 6.39e6)
    if (np.any(ecef_magnitudes <= sensible_radius_range[0])
            or np.any(ecef_magnitudes >= sensible_radius_range[1])):
        raise ValueError(
            'ECEF vector magnitudes must be on the order of the radius of the earth')

    xyz_center = XYZ_from_LatLonAlt(latitude, longitude, altitude)

    xyz_use = np.zeros_like(xyz_in)
    xyz_use[:, 0] = xyz_in[:, 0] - xyz_center[0]
    xyz_use[:, 1] = xyz_in[:, 1] - xyz_center[1]
    xyz_use[:, 2] = xyz_in[:, 2] - xyz_center[2]

    enu = np.zeros_like(xyz_use)
    enu[:, 0] = (-np.sin(longitude) * xyz_use[:, 0]
                 + np.cos(longitude) * xyz_use[:, 1])
    enu[:, 1] = (-np.sin(latitude) * np.cos(longitude) * xyz_use[:, 0]
                 - np.sin(latitude) * np.sin(longitude) * xyz_use[:, 1]
                 + np.cos(latitude) * xyz_use[:, 2])
    enu[:, 2] = (np.cos(latitude) * np.cos(longitude) * xyz_use[:, 0]
                 + np.cos(latitude) * np.sin(longitude) * xyz_use[:, 1]
                 + np.sin(latitude) * xyz_use[:, 2])
    if len(xyz.shape) == 1:
        enu = np.squeeze(enu)
    elif transpose:
        return enu.T

    return enu


def ECEF_from_ENU(enu, latitude, longitude, altitude):
    """
    Calculate ECEF coordinates from local ENU (east, north, up) coordinates.

    Args:
        enu: numpy array, shape (Npts, 3), with local ENU coordinates
        latitude: latitude of center of ENU coordinates in radians
        longitude: longitude of center of ENU coordinates in radians

    Returns:
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates
    """
    enu = np.array(enu)
    if enu.ndim > 1 and enu.shape[1] != 3:
        if enu.shape[0] == 3:
            warnings.warn('The expected shape of the ENU array is (Npts, 3). '
                          'Support for arrays shaped (3, Npts) will go away in '
                          'version 1.5', DeprecationWarning)
            enu_use = enu.T
            transpose = True
        else:
            raise ValueError('The expected shape of the ENU array array is (Npts, 3).')
    else:
        enu_use = enu
        transpose = False

    if enu.shape == (3, 3):
        warnings.warn('The enu array in ECEF_from_ENU is being '
                      'interpreted as (Npts, 3). Historically this function '
                      'has supported (3, Npts) arrays, please verify that '
                      'array ordering is as expected. This warning will be '
                      'removed in version 1.5', DeprecationWarning)

    if enu_use.ndim == 1:
        enu_use = enu_use[np.newaxis, :]

    xyz = np.zeros_like(enu_use)
    xyz[:, 0] = (-np.sin(latitude) * np.cos(longitude) * enu_use[:, 1]
                 - np.sin(longitude) * enu_use[:, 0]
                 + np.cos(latitude) * np.cos(longitude) * enu_use[:, 2])
    xyz[:, 1] = (-np.sin(latitude) * np.sin(longitude) * enu_use[:, 1]
                 + np.cos(longitude) * enu_use[:, 0]
                 + np.cos(latitude) * np.sin(longitude) * enu_use[:, 2])
    xyz[:, 2] = (np.cos(latitude) * enu_use[:, 1]
                 + np.sin(latitude) * enu_use[:, 2])

    xyz_center = XYZ_from_LatLonAlt(latitude, longitude, altitude)
    xyz[:, 0] = xyz[:, 0] + xyz_center[0]
    xyz[:, 1] = xyz[:, 1] + xyz_center[1]
    xyz[:, 2] = xyz[:, 2] + xyz_center[2]
    if len(enu.shape) == 1:
        xyz = np.squeeze(xyz)
    elif transpose:
        return xyz.T

    return xyz


def phase_uvw(ra, dec, xyz):
    """
    This code expects xyz locations relative to the telescope location in the
    same frame that ra/dec are in (e.g. icrs or gcrs) and returns uvws in the
    same frame.

    Note that this code is nearly identical to ENU_from_ECEF, except that it uses
    an arbitrary phasing center rather than a coordinate center.

    Args:
        ra: right ascension to phase to in desired frame
        dec: declination to phase to in desired frame
        xyz: locations relative to the array center in desired frame, shape (Nlocs, 3)

    Returns:
        uvw array in the same frame as xyz, ra and dec
    """
    if xyz.ndim == 1:
        xyz = xyz[np.newaxis, :]

    uvw = np.zeros_like(xyz)
    uvw[:, 0] = (-np.sin(ra) * xyz[:, 0]
                 + np.cos(ra) * xyz[:, 1])
    uvw[:, 1] = (-np.sin(dec) * np.cos(ra) * xyz[:, 0]
                 - np.sin(dec) * np.sin(ra) * xyz[:, 1]
                 + np.cos(dec) * xyz[:, 2])
    uvw[:, 2] = (np.cos(dec) * np.cos(ra) * xyz[:, 0]
                 + np.cos(dec) * np.sin(ra) * xyz[:, 1]
                 + np.sin(dec) * xyz[:, 2])
    return(uvw)


def unphase_uvw(ra, dec, uvw):
    """
    This code expects uvw locations in the same frame that ra/dec are in
    (e.g. icrs or gcrs) and returns relative xyz values in the same frame.

    Args:
        ra: right ascension data are phased to
        dec: declination data are phased to
        uvw: phased uvw values

    Returns:
        xyz locations relative to the array center in the phased frame
    """
    if uvw.ndim == 1:
        uvw = uvw[np.newaxis, :]

    xyz = np.zeros_like(uvw)
    xyz[:, 0] = (-np.sin(ra) * uvw[:, 0]
                 - np.sin(dec) * np.cos(ra) * uvw[:, 1]
                 + np.cos(dec) * np.cos(ra) * uvw[:, 2])

    xyz[:, 1] = (np.cos(ra) * uvw[:, 0]
                 - np.sin(dec) * np.sin(ra) * uvw[:, 1]
                 + np.cos(dec) * np.sin(ra) * uvw[:, 2])

    xyz[:, 2] = (np.cos(dec) * uvw[:, 1]
                 + np.sin(dec) * uvw[:, 2])

    return(xyz)


def get_iterable(x):
    warnings.warn('The get_iterable function is deprecated in favor of '
                  '_get_iterable because it is not API level code. This '
                  'function will be removed in version 1.5', DeprecationWarning)
    return _get_iterable(x)


def _get_iterable(x):
    """Helper function to ensure iterability."""
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)


def fits_gethduaxis(HDU, axis, strict_fits=True):
    warnings.warn('The fits_gethduaxis function is deprecated in favor of '
                  '_fits_gethduaxis because it is not API level code. This '
                  'function will be removed in version 1.5', DeprecationWarning)
    return _fits_gethduaxis(HDU, axis, strict_fits=strict_fits)


def _fits_gethduaxis(HDU, axis, strict_fits=True):
    """
    Helper function for making axis arrays for fits files.

    Args:
        HDU: a fits HDU
        axis: the axis number of interest
        strict_fits: boolean
            If True, require that the axis has cooresponding NAXIS, CRVAL,
            CDELT and CRPIX keywords. If False, allow CRPIX to be missing and
            set it equal to zero (as a way of supporting old calfits files).
            Default is False.
    Returns:
        numpy array of values for that axis
    """

    ax = str(axis)
    N = HDU.header['NAXIS' + ax]
    X0 = HDU.header['CRVAL' + ax]
    dX = HDU.header['CDELT' + ax]
    # add this for calfits backwards compatibility when the CRPIX values were often assumed to be 0
    try:
        Xi0 = HDU.header['CRPIX' + ax] - 1
    except(KeyError):
        if not strict_fits:
            from . import calfits
            calfits._warn_oldcalfits('This file')
            Xi0 = 0
        else:
            raise
    return dX * (np.arange(N) - Xi0) + X0


def get_lst_for_time(jd_array, latitude, longitude, altitude):
    """
    Get the lsts for a set of jd times at an earth location.

    Args:
        jd_array: an array of JD times to get lst for
        latitude: latitude of location to get lst for in degrees
        longitude: longitude of location to get lst for in degrees
        altitude: altitude of location to get lst for in meters

    Returns:
        an array of lst times corresponding to the jd_array
    """
    lsts = []
    lst_array = np.zeros_like(jd_array)
    for ind, jd in enumerate(np.unique(jd_array)):
        t = Time(jd, format='jd', location=(Angle(longitude, unit='deg'),
                                            Angle(latitude, unit='deg')))

        # avoid errors if iers.conf.auto_max_age is set to None, as we do in testing if the iers url is down
        if iers.conf.auto_max_age is None:  # pragma: no cover
            delta, status = t.get_delta_ut1_utc(return_status=True)
            if ((status == iers.TIME_BEFORE_IERS_RANGE) or (status == iers.TIME_BEYOND_IERS_RANGE)):
                warnings.warn('time is out of IERS range, setting delta ut1 utc to extrapolated value')
                t.delta_ut1_utc = delta

        lst_array[np.where(np.isclose(
            jd, jd_array, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').radian

    return lst_array


def fits_indexhdus(hdulist):
    warnings.warn('The fits_indexhdus function is deprecated in favor of '
                  '_fits_indexhdus because it is not API level code. This '
                  'function will be removed in version 1.5', DeprecationWarning)
    return _fits_indexhdus(hdulist)


def _fits_indexhdus(hdulist):
    """
    Helper function for fits I/O.

    Args:
        hdulist: a list of hdus

    Returns:
        dictionary of table names
    """
    tablenames = {}
    for i in range(len(hdulist)):
        try:
            tablenames[hdulist[i].header['EXTNAME']] = i
        except(KeyError):
            continue
    return tablenames


def _x_orientation_rep_dict(x_orientation):
    """"Helper function to create replacement dict based on x_orientation"""
    if x_orientation.lower() == 'east' or x_orientation.lower() == 'e':
        return {'x': 'e', 'y': 'n'}
    elif x_orientation.lower() == 'north' or x_orientation.lower() == 'n':
        return {'x': 'n', 'y': 'e'}
    else:
        raise ValueError('x_orientation not recognized.')


def polstr2num(pol, x_orientation=None):
    """
    Convert polarization str to number according to AIPS Memo 117.

    Prefer 'pI', 'pQ', 'pU' and 'pV' to make it clear that these are pseudo-Stokes,
    not true Stokes, but also supports 'I', 'Q', 'U', 'V'.

    Parameters
    ----------
    pol : str
        polarization string
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to allow for
        converting from E/N strings. See corresonding parameter on UVData
        for more details.

    Returns
    ----------
    int
        Number corresponding to string

    Raises
    ------
    ValueError
        If the pol string cannot be converted to a polarization number.

    Warns
    ------
    UserWarning
        If the x_orientation not recognized.
    """
    dict_use = copy.deepcopy(POL_STR2NUM_DICT)
    if x_orientation is not None:
        try:
            rep_dict = _x_orientation_rep_dict(x_orientation)
            for key, value in six.iteritems(POL_STR2NUM_DICT):
                new_key = key.replace('x', rep_dict['x']).replace('y', rep_dict['y'])
                dict_use[new_key] = value
        except ValueError:
            warnings.warn('x_orientation not recognized.')

    poldict = {k.lower(): v for k, v in six.iteritems(dict_use)}
    if isinstance(pol, str):
        out = poldict[pol.lower()]
    elif isinstance(pol, collections.Iterable):
        out = [poldict[key.lower()] for key in pol]
    else:
        raise ValueError('Polarization {p} cannot be converted to a polarization number.'.format(p=pol))
    return out


def polnum2str(num, x_orientation=None):
    """
    Convert polarization number to str according to AIPS Memo 117.

    Uses 'pI', 'pQ', 'pU' and 'pV' to make it clear that these are pseudo-Stokes,
    not true Stokes

    Parameters
    ----------
    num : int
        polarization number
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to convert to
        E/N strings. See corresonding parameter on UVData for more details.

    Returns
    ----------
    str
        String corresponding to polarization number

    Raises
    ------
    ValueError
        If the polarization number cannot be converted to a polarization string.

    Warns
    ------
    UserWarning
        If the x_orientation not recognized.
    """
    dict_use = copy.deepcopy(POL_NUM2STR_DICT)
    if x_orientation is not None:
        try:
            rep_dict = _x_orientation_rep_dict(x_orientation)
            for key, value in six.iteritems(POL_NUM2STR_DICT):
                new_val = value.replace('x', rep_dict['x']).replace('y', rep_dict['y'])
                dict_use[key] = new_val
        except ValueError:
            warnings.warn('x_orientation not recognized.')

    if isinstance(num, six.integer_types + (np.int32, np.int64)):
        out = dict_use[num]
    elif isinstance(num, collections.Iterable):
        out = [dict_use[i] for i in num]
    else:
        raise ValueError('Polarization {p} cannot be converted to string.'.format(p=num))
    return out


def jstr2num(jstr, x_orientation=None):
    """
    Convert jones polarization str to number according to calfits memo.

    Parameters
    ----------
    jstr : str
        antenna (jones) polarization string
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to allow for
        converting from E/N strings. See corresonding parameter on UVData
        for more details.

    Returns
    ----------
    int
        antenna (jones) polarization number corresponding to string

    Raises
    ------
    ValueError
        If the jones string cannot be converted to a polarization number.

    Warns
    ------
    UserWarning
        If the x_orientation not recognized.
    """
    dict_use = copy.deepcopy(JONES_STR2NUM_DICT)
    if x_orientation is not None:
        try:
            rep_dict = _x_orientation_rep_dict(x_orientation)
            for key, value in six.iteritems(JONES_STR2NUM_DICT):
                new_key = key.replace('x', rep_dict['x']).replace('y', rep_dict['y'])
                dict_use[new_key] = value
        except ValueError:
            warnings.warn('x_orientation not recognized.')

    jdict = {k.lower(): v for k, v in six.iteritems(dict_use)}
    if isinstance(jstr, str):
        out = jdict[jstr.lower()]
    elif isinstance(jstr, collections.Iterable):
        out = [jdict[key.lower()] for key in jstr]
    else:
        raise ValueError('Jones polarization {j} cannot be converted to index.'.format(j=jstr))
    return out


def jnum2str(jnum, x_orientation=None):
    """
    Convert jones polarization number to str according to calfits memo.

    Parameters
    ----------
    num : int
        antenna (jones) polarization number
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to convert to
        E/N strings. See corresonding parameter on UVData for more details.

    Returns
    ----------
    str
        antenna (jones) polarization string corresponding to number

    Raises
    ------
    ValueError
        If the jones polarization number cannot be converted to a jones polarization string.

    Warns
    ------
    UserWarning
        If the x_orientation not recognized.
    """
    dict_use = copy.deepcopy(JONES_NUM2STR_DICT)
    if x_orientation is not None:
        try:
            rep_dict = _x_orientation_rep_dict(x_orientation)
            for key, value in six.iteritems(JONES_NUM2STR_DICT):
                new_val = value.replace('x', rep_dict['x']).replace('y', rep_dict['y'])
                dict_use[key] = new_val
        except ValueError:
            warnings.warn('x_orientation not recognized.')

    if isinstance(jnum, six.integer_types + (np.int32, np.int64)):
        out = dict_use[jnum]
    elif isinstance(jnum, collections.Iterable):
        out = [dict_use[i] for i in jnum]
    else:
        raise ValueError('Jones polarization {j} cannot be converted to string.'.format(j=jnum))
    return out


def parse_polstr(polstr, x_orientation=None):
    """
    Parse a polarization string and return pyuvdata standard polarization string.

    See utils.POL_STR2NUM_DICT for options.

    Parameters
    ----------
    polstr : str
        polarization string
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to allow for
        converting from E/N strings. See corresonding parameter on UVData
        for more details.

    Returns
    ----------
    str
        AIPS Memo 117 standard string

    Raises
    ------
    ValueError
        If the pol string cannot be converted to a polarization number.

    Warns
    ------
    UserWarning
        If the x_orientation not recognized.
    """
    return polnum2str(polstr2num(polstr, x_orientation=x_orientation),
                      x_orientation=x_orientation)


def parse_jpolstr(jpolstr, x_orientation=None):
    """
    Parse a Jones polarization string and return pyuvdata standard jones string.

    See utils.JONES_STR2NUM_DICT for options.

    Parameters
    ----------
    jpolstr : str
        Jones polarization string

    Returns
    ----------
    str
        calfits memo standard string

    Raises
    ------
    ValueError
        If the jones string cannot be converted to a polarization number.

    Warns
    ------
    UserWarning
        If the x_orientation not recognized.
    """
    return jnum2str(jstr2num(jpolstr, x_orientation=x_orientation),
                    x_orientation=x_orientation)


def conj_pol(pol):
    """
    Returns the polarization for the conjugate baseline.
    For example, (1, 2, 'xy') = conj(2, 1, 'yx').
    The returned polarization is determined by assuming the antenna pair is reversed
    in the data, and finding the correct polarization correlation which will yield
    the requested baseline when conjugated. Note this means changing the polarization
    for linear cross-pols, but keeping auto-pol (e.g. xx) and Stokes the same.

    Args:
        pol: Polarization (str or int)

    Returns:
        cpol: Polarization as if antennas are swapped (type matches input)
    """

    deprecated_jones_dict = {'jxx': 'Jxx', 'jyy': 'Jyy', 'jxy': 'Jyx', 'jyx': 'Jxy',
                             'jrr': 'Jrr', 'jll': 'Jll', 'jrl': 'Jlr', 'jlr': 'Jrl'}

    cpol_dict = {k.lower(): v for k, v in six.iteritems(CONJ_POL_DICT)}

    if isinstance(pol, str):
        if pol.lower().startswith('j'):
            warnings.warn('conj_pol should not be called with jones matrix elements. '
                          'Support for the jones matrix elements will go away '
                          'in version 1.5', DeprecationWarning)
            cpol = deprecated_jones_dict[pol.lower()]
        else:
            cpol = cpol_dict[pol.lower()]
    elif isinstance(pol, collections.Iterable):
        cpol = [conj_pol(p) for p in pol]
    elif isinstance(pol, six.integer_types + (np.int32, np.int64)):
        cpol = polstr2num(cpol_dict[polnum2str(pol).lower()])
    else:
        raise ValueError('Polarization cannot be conjugated.')
    return cpol


def reorder_conj_pols(pols):
    """
    Reorders a list of pols, swapping pols that are conjugates of one another.
    For example ('xx', 'xy', 'yx', 'yy') -> ('xx', 'yx', 'xy', 'yy')
    This is useful for the _key2inds function in the case where an antenna
    pair is specified but the conjugate pair exists in the data. The conjugated
    data should be returned in the order of the polarization axis, so after conjugating
    the data, the pols need to be reordered.
    For example, if a file contains antpair (0, 1) and pols 'xy' and 'yx', but
    the user requests antpair (1, 0), they should get:
    [(1x, 0y), (1y, 0x)] = [conj(0y, 1x), conj(0x, 1y)]

    Args:
        pols: Polarization array (strings or ints)

    Returns:
        conj_order: Indices to reorder polarization axis
    """
    if not isinstance(pols, collections.Iterable):
        raise ValueError('reorder_conj_pols must be given an array of polarizations.')
    cpols = np.array([conj_pol(p) for p in pols])  # Array needed for np.where
    conj_order = [np.where(cpols == p)[0][0] if p in cpols else -1 for p in pols]
    if -1 in conj_order:
        raise ValueError('Not all conjugate pols exist in the polarization array provided.')
    return conj_order


def check_history_version(history, version_string):
    warnings.warn('The check_history_version function is deprecated in favor of '
                  '_check_history_version because it is not API level code. This '
                  'function will be removed in version 1.5', DeprecationWarning)
    return _check_history_version(history, version_string)


def _check_history_version(history, version_string):
    if (version_string.replace(' ', '') in history.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False


def check_histories(history1, history2):
    warnings.warn('The check_histories function is deprecated in favor of '
                  '_check_histories because it is not API level code. This '
                  'function will be removed in version 1.5', DeprecationWarning)
    return _check_histories(history1, history2)


def _check_histories(history1, history2):
    if (history1.replace('\n', '').replace(' ', '') == history2.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False


def combine_histories(history1, history2):
    warnings.warn('The combine_histories function is deprecated in favor of '
                  '_combine_histories because it is not API level code. This '
                  'function will be removed in version 1.5', DeprecationWarning)
    return _combine_histories(history1, history2)


def _combine_histories(history1, history2):
    hist2_words = history2.split(' ')
    add_hist = ''
    test_hist1 = ' ' + history1 + ' '
    for i, word in enumerate(hist2_words):
        if ' ' + word + ' ' not in test_hist1:
            add_hist += ' ' + word
            keep_going = (i + 1 < len(hist2_words))
            while keep_going:
                if ((hist2_words[i + 1] is ' ')
                        or (' ' + hist2_words[i + 1] + ' ' not in test_hist1)):
                    add_hist += ' ' + hist2_words[i + 1]
                    del(hist2_words[i + 1])
                    keep_going = (i + 1 < len(hist2_words))
                else:
                    keep_going = False

    return history1 + add_hist


def baseline_to_antnums(baseline, Nants_telescope):
    """
    Get the antenna numbers corresponding to a given baseline number.

    Args:
        baseline: integer baseline number
        Nant_telescope: integer number of antennas

    Returns:
        tuple with the two antenna numbers corresponding to the baseline.
    """
    if Nants_telescope > 2048:
        raise Exception('error Nants={Nants}>2048 not '
                        'supported'.format(Nants=Nants_telescope))
    if np.min(baseline) > 2**16:
        ant2 = (baseline - 2**16) % 2048 - 1
        ant1 = (baseline - 2**16 - (ant2 + 1)) / 2048 - 1
    else:
        ant2 = (baseline) % 256 - 1
        ant1 = (baseline - (ant2 + 1)) / 256 - 1
    return np.int32(ant1), np.int32(ant2)


def antnums_to_baseline(ant1, ant2, Nants_telescope, attempt256=False):
    """
    Get the baseline number corresponding to two given antenna numbers.

    Args:
        ant1: first antenna number (integer)
        ant2: second antenna number (integer)
        Nant_telescope: integer number of antennas
        attempt256: Option to try to use the older 256 standard used in
            many uvfits files (will use 2048 standard if there are more
            than 256 antennas). Default is False.

    Returns:
        integer baseline number corresponding to the two antenna numbers.
    """
    ant1, ant2 = np.int64((ant1, ant2))
    if Nants_telescope is not None and Nants_telescope > 2048:
        raise Exception('cannot convert ant1, ant2 to a baseline index '
                        'with Nants={Nants}>2048.'
                        .format(Nants=Nants_telescope))
    if attempt256:
        if (np.max(ant1) < 255 and np.max(ant2) < 255):
            return 256 * (ant1 + 1) + (ant2 + 1)
        else:
            print('Max antnums are {} and {}'.format(
                np.max(ant1), np.max(ant2)))
            message = 'antnums_to_baseline: found > 256 antennas, using ' \
                      '2048 baseline indexing. Beware compatibility ' \
                      'with CASA etc'
            warnings.warn(message)

    baseline = 2048 * (ant1 + 1) + (ant2 + 1) + 2**16

    if isinstance(baseline, np.ndarray):
        return np.asarray(baseline, dtype=np.int64)
    else:
        return np.int64(baseline)


def get_baseline_redundancies(baselines, baseline_vecs, tol=1.0, with_conjugates=False):
    """
    Find redundant baseline groups

    Parameters
    ----------
    baselines : array_like of int
        Baseline numbers, shape (Nbls,)
    baseline_vecs : array_like of float
        Baseline vectors in meters, shape shape (Nbls, 3)
    tol : float
        Absolute tolerance of redundancy, in meters.
    with_conjugates : bool
        Option to include baselines that are redundant when flipped.

    Returns
    -------
    baseline_groups : list of lists of int
        list of lists of redundant baseline numbers
    vec_bin_centers : list of array_like of float
        List of vectors describing redundant group centers
    lengths : list of float
        List of redundant group baseline lengths in meters
    baseline_ind_conj : list of int
        List of baselines that are redundant when reversed. Only returned if
        with_conjugates is True
    """
    Nbls = baselines.shape[0]

    if not baseline_vecs.shape == (Nbls, 3):
        raise ValueError("Baseline vectors must be shape (Nbls, 3)")

    baseline_vecs = copy.copy(baseline_vecs)              # Protect the vectors passed in.

    if with_conjugates:
        conjugates = []
        for bl in baseline_vecs:
            if bl[0] == 0:
                if bl[1] == 0:
                    conjugates.append(bl[2] < 0)
                else:
                    conjugates.append(bl[1] < 0)
            else:
                conjugates.append(bl[0] < 0)
        conjugates = np.array(conjugates, dtype=bool)
        baseline_vecs[conjugates] *= (-1)
        baseline_ind_conj = baselines[conjugates]
        bl_gps, vec_bin_centers, lens = get_baseline_redundancies(baselines, baseline_vecs, tol=tol, with_conjugates=False)
        return bl_gps, vec_bin_centers, lens, baseline_ind_conj

    # For each baseline, list all others that are within the tolerance distance.

    adj = {}   # Adjacency dictionary

    for bi, bv0 in enumerate(baseline_vecs):
        key0 = baselines[bi]
        adj[key0] = []
        for bj, bv1 in enumerate(baseline_vecs):
            dist = np.linalg.norm(bv1 - bv0)
            if dist < tol:
                key1 = baselines[bj]
                adj[key0].append(key1)

    # The adjacency list defines a set of graph edges.
    # For each baseline b0, loop over its adjacency list ai \in adj[b0]
    #   If adj[b0] is a subset of adj[ai], then ai is in a redundant group with b0

    bl_gps = []
    for k in adj.keys():
        a0 = adj[k] + [k, ]
        group = [k]
        for a in a0:
            if set(a0).issubset(adj[a]) and a not in group:
                group.append(a)
        group.sort()
        bl_gps.append(group)

    # Groups can be different lengths, but we need to take a unique over an axis
    # to properly identify unique groups
    # Pad out all the sub-lists to be the same length
    pad = len(max(bl_gps, key=len))
    bl_gps = np.array([i + [-1] * (pad - len(i)) for i in bl_gps])
    # We end up with multiple copies of each redundant group, so remove duplicates
    bl_gps = np.unique(bl_gps, axis=0).tolist()
    # remove the dummy pad baselines from each list
    bl_gps = [[bl for bl in gp if bl != -1] for gp in bl_gps]

    N_unique = len(bl_gps)
    vec_bin_centers = np.zeros((N_unique, 3))
    for gi, gp in enumerate(bl_gps):
        inds = [np.where(i == baselines)[0] for i in gp]
        vec_bin_centers[gi] = np.mean(baseline_vecs[inds, :], axis=0)

    lens = np.sqrt(np.sum(vec_bin_centers**2, axis=1))

    return bl_gps, vec_bin_centers, lens


def get_antenna_redundancies(antenna_numbers, antenna_positions, tol=1.0, include_autos=False):
    """
    Find redundant baseline groups based on antenna positions.

    Include all possible redundant baselines based on antenna positions.

    Parameters
    ----------
    antenna_numbers : array_like of int
        Antenna numbers, shape (Nants,).
    antenna_positions : array_like of float
        Antenna position vectors in the ENU (topocentric) frame in meters, shape (Nants, 3).
    tol : float
        Redundancy tolerance in meters.
    include_autos : bool
        Option to include autocorrelations.

    Returns
    -------
    baseline_groups : list of lists of int
        list of lists of redundant baseline numbers
    vec_bin_centers : list of array_like of float
        List of vectors describing redundant group centers
    lengths : list of float
        List of redundant group baseline lengths in meters
    """
    Nants = antenna_numbers.shape[0]

    bls = []
    bl_vecs = []

    for aj in range(Nants):
        mini = aj + 1
        if include_autos:
            mini = aj
        for ai in range(mini, Nants):
            anti, antj = antenna_numbers[ai], antenna_numbers[aj]
            bidx = antnums_to_baseline(anti, antj, Nants)
            bv = antenna_positions[aj] - antenna_positions[ai]
            # Enforce u-positive orientation
            if (bv[0] < 0 or ((bv[0] == 0) and bv[1] < 0)
               or ((bv[0] == 0) and (bv[1] == 0) and bv[2] < 0)):
                bv *= (-1)
                bidx = antnums_to_baseline(antj, anti, Nants)
            bl_vecs.append(bv)
            bls.append(bidx)
    bls = np.array(bls)
    bl_vecs = np.array(bl_vecs)
    return get_baseline_redundancies(bls, bl_vecs, tol=tol, with_conjugates=False)


def _reraise_context(fmt, *args):
    """Reraise an exception with its message modified to specify additional context.

    This function tries to help provide context when a piece of code
    encounters an exception while trying to get something done, and it wishes
    to propagate contextual information farther up the call stack. It is a
    consistent way to do it for both Python 2 and 3, since Python 2 does not
    provide Python 3’s `exception chaining <https://www.python.org/dev/peps/pep-3134/>`_ functionality.
    Instead of that more sophisticated infrastructure, this function just
    modifies the textual message associated with the exception being raised.
    If only a single argument is supplied, the exception text is prepended with
    the stringification of that argument. If multiple arguments are supplied,
    the first argument is treated as an old-fashioned ``printf``-type
    (``%``-based) format string, and the remaining arguments are the formatted
    values.
    Borrowed from pwkit (https://github.com/pkgw/pwkit/blob/master/pwkit/__init__.py)
    Example usage::
      from pyuvdata.utils import reraise_context
      filename = 'my-filename.txt'
      try:
        f = filename.open('rt')
        for line in f.readlines():
          # do stuff ...
      except Exception as e:
        reraise_context('while reading "%r"', filename)
        # The exception is reraised and so control leaves this function.
    If an exception with text ``"bad value"`` were to be raised inside the
    ``try`` block in the above example, its text would be modified to read
    ``"while reading \"my-filename.txt\": bad value"``.
    """
    import sys

    if len(args):
        cstr = fmt % args
    else:
        cstr = six.text_type(fmt)

    ex = sys.exc_info()[1]

    if isinstance(ex, EnvironmentError):
        ex.strerror = '%s: %s' % (cstr, ex.strerror)
        ex.args = (ex.errno, ex.strerror)
    else:
        if len(ex.args):
            cstr = '%s: %s' % (cstr, ex.args[0])
        ex.args = (cstr, ) + ex.args[1:]

    raise


def collapse(arr, alg, weights=None, axis=None, return_weights=False):
    ''' Parent function to collapse an array with a given algorithm.
    Args:
        arr (array): Input array to process.
        alg (str): Algorithm to use. Must be defined in this function with
            corresponding subfunction below.
        weights (array, optional): weights for collapse operation (e.g. weighted mean).
            NOTE: Some subfunctions do not use the weights. See corresponding doc strings.
        axis (int, tuple, optional): Axis or axes to collapse. Default is all.
        return_weights (Bool): Whether to return sum of weights. Default is False.
    '''
    collapse_dict = {'mean': mean_collapse, 'absmean': absmean_collapse,
                     'quadmean': quadmean_collapse, 'or': or_collapse,
                     'and': and_collapse}
    try:
        out = collapse_dict[alg](arr, weights=weights, axis=axis, return_weights=return_weights)
    except KeyError:
        raise ValueError('Collapse algorithm must be one of: '
                         + ', '.join(collapse_dict.keys()) + '.')
    return out


def mean_collapse(arr, weights=None, axis=None, return_weights=False):
    ''' Function to average data. This is similar to np.average, except it
    handles infs (by giving them zero weight) and zero weight axes (by forcing
    result to be inf with zero output weight).
    Args:
        arr - array to process
        weights - weights for average. If none, will default to equal weight for
                  all non-infinite data.
        axis - axis keyword to pass to np.sum
        return_weights - whether to return sum of weights. Default is False.
    '''
    arr = copy.deepcopy(arr)  # avoid changing outside
    if weights is None:
        weights = np.ones_like(arr)
    else:
        weights = copy.deepcopy(weights)
    weights = weights * np.logical_not(np.isinf(arr))
    arr[np.isinf(arr)] = 0
    weight_out = np.sum(weights, axis=axis)
    out = np.sum(weights * arr, axis=axis)
    where = (weight_out > 1e-10)
    out = np.true_divide(out, weight_out, where=where)
    out = np.where(where, out, np.inf)
    if return_weights:
        return out, weight_out
    else:
        return out


def absmean_collapse(arr, weights=None, axis=None, return_weights=False):
    ''' Function to average absolute value
    Args:
        arr - array to process
        weights - weights for average
        axis - axis keyword to pass to np.mean
        return_weights - whether to return sum of weights. Default is False.
    '''
    return mean_collapse(np.abs(arr), weights=weights, axis=axis, return_weights=return_weights)


def quadmean_collapse(arr, weights=None, axis=None, return_weights=False):
    ''' Function to average in quadrature
    Args:
        arr - array to process
        weights - weights for average
        axis - axis keyword to pass to np.mean
        return_weights - whether to return sum of weights. Default is False.
    '''
    out = mean_collapse(np.abs(arr)**2, weights=weights, axis=axis, return_weights=return_weights)
    if return_weights:
        return np.sqrt(out[0]), out[1]
    else:
        return np.sqrt(out)


def or_collapse(arr, weights=None, axis=None, return_weights=False):
    ''' Function to collapse axes using OR operation
    Args:
        arr - boolean array to process
        weights - NOT USED, but kept for symmetry with other averaging functions
        axis - axis or axes over which to OR
        return_weights - whether to return dummy weights array. NOTE: the dummy weights
            will simply be an array of ones. Default is False.
    '''
    if arr.dtype != np.bool:
        raise ValueError('Input to or_collapse function must be boolean array')
    out = np.any(arr, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn('Currently weights are not handled when OR-ing boolean arrays.')
    if return_weights:
        return out, np.ones_like(out, dtype=np.float)
    else:
        return out


def and_collapse(arr, weights=None, axis=None, return_weights=False):
    ''' Function to collapse axes using AND operation
    Args:
        arr - boolean array to process
        weights - NOT USED, but kept for symmetry with other averaging functions
        axis - axis or axes over which to AND
        return_weights - whether to return dummy weights array. NOTE: the dummy weights
            will simply be an array of ones. Default is False.
    '''
    if arr.dtype != np.bool:
        raise ValueError('Input to and_collapse function must be boolean array')
    out = np.all(arr, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn('Currently weights are not handled when AND-ing boolean arrays.')
    if return_weights:
        return out, np.ones_like(out, dtype=np.float)
    else:
        return out
