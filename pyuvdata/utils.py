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
from astropy.time import Time
from astropy.coordinates import Angle

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
                          'Support for arrays shaped (3, Npts) will go away in a '
                          'future version.', PendingDeprecationWarning)
            xyz_use = xyz.T
        else:
            raise ValueError('The expected shape of ECEF xyz array is (Npts, 3).')

    else:
        xyz_use = xyz

    if xyz.shape == (3, 3):
        warnings.warn('The xyz array in LatLonAlt_from_XYZ is being '
                      'interpreted as (Npts, 3). Historically this function '
                      'has supported (3, Npts) arrays, please verify that '
                      'array ordering is as expected.', PendingDeprecationWarning)

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
                          'Support for arrays shaped (3, Npts) will go away in a '
                          'future version.', PendingDeprecationWarning)
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
                      'array ordering is as expected.', PendingDeprecationWarning)

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
                          'Support for arrays shaped (3, Npts) will go away in a '
                          'future version.', PendingDeprecationWarning)
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
                      'array ordering is as expected.', PendingDeprecationWarning)

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
                  '_get_iterable because it is not API level code', DeprecationWarning)
    return _get_iterable(x)


def _get_iterable(x):
    """Helper function to ensure iterability."""
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)


def fits_gethduaxis(HDU, axis, strict_fits=True):
    warnings.warn('The fits_gethduaxis function is deprecated in favor of '
                  '_fits_gethduaxis because it is not API level code', DeprecationWarning)
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
        lst_array[np.where(np.isclose(
            jd, jd_array, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').radian

    return lst_array


def fits_indexhdus(hdulist):
    warnings.warn('The fits_indexhdus function is deprecated in favor of '
                  '_fits_indexhdus because it is not API level code', DeprecationWarning)
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


def polstr2num(pol):
    """
    Convert polarization str to number according to AIPS Memo 117.
    Prefer 'pI', 'pQ', 'pU' and 'pV' to make it clear that these are pseudo-Stokes,
    not true Stokes, but also support 'I', 'Q', 'U', 'V'.

    Args:
        pol: polarization string

    Returns:
        Number corresponding to string
    """
    # Use all upper case keys to support case in-sensitive handling
    # (cast input string to upper case for key comparison)
    poldict = {'PI': 1, 'PQ': 2, 'PU': 3, 'PV': 4,
               'I': 1, 'Q': 2, 'U': 3, 'V': 4,
               'RR': -1, 'LL': -2, 'RL': -3, 'LR': -4,
               'XX': -5, 'YY': -6, 'XY': -7, 'YX': -8}
    if isinstance(pol, str):
        out = poldict[pol.upper()]
    elif isinstance(pol, collections.Iterable):
            out = [poldict[key.upper()] for key in pol]
    else:
        raise ValueError('Polarization cannot be converted to index.')
    return out


def polnum2str(num):
    """
    Convert polarization number to str according to AIPS Memo 117.
    Use 'pI', 'pQ', 'pU' and 'pV' to make it clear that these are pseudo-Stokes, not true Stokes

    Args:
        num: polarization number

    Returns:
        String corresponding to string
    """
    str_list = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', '', 'pI', 'pQ', 'pU', 'pV']
    if isinstance(num, six.integer_types + (np.int32, np.int64)):
        out = str_list[num + 8]
    elif isinstance(num, collections.Iterable):
            out = [str_list[i + 8] for i in num]
    else:
        raise ValueError('Polarization cannot be converted to string.')
    return out


def jstr2num(jstr):
    """
    Convert jones polarization str to number according to calfits memo.

    Args:
        jones: antenna polarization string

    Returns:
        Number corresponding to string
    """
    jdict = {'jxx': -5, 'jyy': -6, 'jxy': -7, 'jyx': -8,
             'xx': -5, 'x': -5, 'yy': -6, 'y': -6, 'xy': -7, 'yx': -8,  # Allow shorthand
             'jrr': -1, 'jll': -2, 'jrl': -3, 'jlr': -4,
             'rr': -1, 'r': -1, 'll': -2, 'l': -2, 'rl': -3, 'lr': -4}
    if isinstance(jstr, str):
        out = jdict[jstr.lower()]
    elif isinstance(jstr, collections.Iterable):
            out = [jdict[key.lower()] for key in jstr]
    else:
        raise ValueError('Jones polarization cannot be converted to index.')
    return out


def jnum2str(jnum):
    """
    Convert jones polarization number to str according to calfits memo.

    Args:
        num: polarization number

    Returns:
        String corresponding to string
    """
    str_list = ['jyx', 'jxy', 'jyy', 'jxx', 'jlr', 'jrl', 'jll', 'jrr']
    if isinstance(jnum, six.integer_types + (np.int32, np.int64)):
        out = str_list[jnum + 8]
    elif isinstance(jnum, collections.Iterable):
            out = [str_list[i + 8] for i in jnum]
    else:
        raise ValueError('Polarization cannot be converted to string.')
    return out


def conj_pol(pol):
    """
    Returns the polarization for the conjugate baseline.
    For example, (1, 2, 'XY') = conj(2, 1, 'YX').
    The returned polarization is determined by assuming the antenna pair is reversed
    in the data, and finding the correct polarization correlation which will yield
    the requested baseline when conjugated. Note this means changing the polarization
    for linear cross-pols, but keeping auto-pol (e.g. XX) and Stokes the same.

    Args:
        pol: Polarization (str or int)

    Returns:
        cpol: Polarization as if antennas are swapped (type matches input)
    """
    cpol_dict = {'XX': 'XX', 'YY': 'YY', 'XY': 'YX', 'YX': 'XY',
                 'JXX': 'jxx', 'JYY': 'jyy', 'JXY': 'jyx', 'JYX': 'jxy',
                 'RR': 'RR', 'LL': 'LL', 'RL': 'LR', 'LR': 'RL',
                 'JRR': 'jrr', 'JLL': 'jll', 'JRL': 'jlr', 'JLR': 'jrl',
                 'I': 'I', 'Q': 'Q', 'U': 'U', 'V': 'V',
                 'PI': 'pI', 'PQ': 'pQ', 'PU': 'pU', 'PV': 'pV'}

    if isinstance(pol, str):
        cpol = cpol_dict[pol.upper()]
    elif isinstance(pol, collections.Iterable):
        cpol = [conj_pol(p) for p in pol]
    elif isinstance(pol, six.integer_types + (np.int32, np.int64)):
        cpol = polstr2num(cpol_dict[polnum2str(pol).upper()])
    else:
        raise ValueError('Polarization cannot be conjugated.')
    return cpol


def check_history_version(history, version_string):
    warnings.warn('The check_history_version function is deprecated in favor of '
                  '_check_history_version because it is not API level code', DeprecationWarning)
    return _check_history_version(history, version_string)


def _check_history_version(history, version_string):
    if (version_string.replace(' ', '') in history.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False


def check_histories(history1, history2):
    warnings.warn('The check_histories function is deprecated in favor of '
                  '_check_histories because it is not API level code', DeprecationWarning)
    return _check_histories(history1, history2)


def _check_histories(history1, history2):
    if (history1.replace('\n', '').replace(' ', '') == history2.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False


def combine_histories(history1, history2):
    warnings.warn('The combine_histories function is deprecated in favor of '
                  '_combine_histories because it is not API level code', DeprecationWarning)
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
