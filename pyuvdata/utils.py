"""Commonly used utility functions."""
import numpy as np
import collections
import warnings

# parameters for transforming between xyz & lat/lon/alt
gps_b = 6356752.31424518
gps_a = 6378137
e_squared = 6.69437999014e-3
e_prime_squared = 6.73949674228e-3


def LatLonAlt_from_XYZ(xyz):
    """
    Calculate lat/lon/alt from ECEF x,y,z.

    Args:
        xyz: numpy array, shape (3, Npts), with ECEF x,y,z coordinates

    Returns:
        tuple of latitude, longitude, altitude numpy arrays (if Npts > 1) or values (if Npts = 1) in radians & meters
    """
    # convert to a numpy array
    xyz = np.array(xyz)
    if xyz.shape[0] != 3:
        raise ValueError(
            'The first dimension of the ECEF xyz array must be length 3')
    if len(xyz.shape) == 1:
        Npts = 1
        xyz = xyz[:, np.newaxis]
    else:
        Npts = xyz.shape[1]

    # checking for acceptable values
    if np.any(np.linalg.norm(xyz, axis=0) < 6.35e6) or np.any(np.linalg.norm(xyz, axis=0) > 6.39e6):
        raise ValueError(
            'xyz values should be ECEF x, y, z coordinates in meters')

    # see wikipedia geodetic_datum and Datum transformations of
    # GPS positions PDF in docs/references folder
    gps_p = np.sqrt(xyz[0, :]**2 + xyz[1, :]**2)
    gps_theta = np.arctan2(xyz[2, :] * gps_a, gps_p * gps_b)
    latitude = np.arctan2(xyz[2, :] + e_prime_squared * gps_b *
                          np.sin(gps_theta)**3, gps_p - e_squared * gps_a *
                          np.cos(gps_theta)**3)

    longitude = np.arctan2(xyz[1, :], xyz[0, :])
    gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
    altitude = ((gps_p / np.cos(latitude)) - gps_N)

    if Npts == 1:
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
        numpy array, shape (3, Npts) (if Npts > 1) or (3,) (if Npts = 1), with ECEF x,y,z coordinates
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
    xyz = np.zeros((3, Npts))
    xyz[0, :] = ((gps_N + altitude) * np.cos(latitude) * np.cos(longitude))
    xyz[1, :] = ((gps_N + altitude) * np.cos(latitude) * np.sin(longitude))
    xyz[2, :] = ((gps_b**2 / gps_a**2 * gps_N + altitude) * np.sin(latitude))

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
        xyz: numpy array, shape (3, Npts), with ECEF x,y,z coordinates
        latitude: latitude of center of ENU coordinates in radians
        longitude: longitude of center of ENU coordinates in radians
        altitude: altitude of center of ENU coordinates in radians

    Returns:
        numpy array, shape (3, Npts), with local ENU coordinates
    """
    if xyz.shape[0] != 3:
        raise ValueError(
            'The first dimension of the ECEF xyz array must be length 3')
    if len(xyz.shape) == 1:
        Npts = 1
    else:
        Npts = xyz.shape[1]

    # check that these are sensible ECEF values -- their magnitudes need to be
    # on the order of Earth's radius
    ecef_magnitudes = np.linalg.norm(xyz, axis=0)
    sensible_radius_range = (6.35e6, 6.39e6)
    if np.any(ecef_magnitudes <= sensible_radius_range[0]) or np.any(ecef_magnitudes >= sensible_radius_range[1]):
        raise ValueError(
            'ECEF vector magnitudes must be on the order of the radius of the earth')

    xyz_center = XYZ_from_LatLonAlt(latitude, longitude, altitude)

    if Npts == 1:
        xyz = xyz[:, np.newaxis]
    xyz_use = np.zeros_like(xyz)
    xyz_use[0, :] = xyz[0, :] - xyz_center[0]
    xyz_use[1, :] = xyz[1, :] - xyz_center[1]
    xyz_use[2, :] = xyz[2, :] - xyz_center[2]
    xyz = np.squeeze(xyz)

    enu = np.zeros((3, Npts))
    enu[0, :] = (-np.sin(longitude) * xyz_use[0, :] +
                 np.cos(longitude) * xyz_use[1, :])
    enu[1, :] = (-np.sin(latitude) * np.cos(longitude) * xyz_use[0, :] -
                 np.sin(latitude) * np.sin(longitude) * xyz_use[1, :] +
                 np.cos(latitude) * xyz_use[2, :])
    enu[2, :] = (np.cos(latitude) * np.cos(longitude) * xyz_use[0, :] +
                 np.cos(latitude) * np.sin(longitude) * xyz_use[1, :] +
                 np.sin(latitude) * xyz_use[2, :])
    enu = np.squeeze(enu)

    return enu


def ECEF_from_ENU(enu, latitude, longitude, altitude):
    """
    Calculate ECEF coordinates from local ENU (east, north, up) coordinates.

    Args:
        enu: numpy array, shape (3, Npts), with local ENU coordinates
        latitude: latitude of center of ENU coordinates in radians
        longitude: longitude of center of ENU coordinates in radians

    Returns:
        numpy array, shape (3, Npts), with ECEF x,y,z coordinates
    """
    if enu.shape[0] != 3:
        raise ValueError(
            'The first dimension of the local ENU array must be length 3')
    if len(enu.shape) == 1:
        Npts = 1
    else:
        Npts = enu.shape[1]

    xyz = np.zeros((3, Npts))
    if Npts == 1:
        enu = enu[:, np.newaxis]
    xyz[0, :] = (-np.sin(latitude) * np.cos(longitude) * enu[1, :] -
                 np.sin(longitude) * enu[0, :] +
                 np.cos(latitude) * np.cos(longitude) * enu[2, :])
    xyz[1, :] = (-np.sin(latitude) * np.sin(longitude) * enu[1, :] +
                 np.cos(longitude) * enu[0, :] +
                 np.cos(latitude) * np.sin(longitude) * enu[2, :])
    xyz[2, :] = (np.cos(latitude) * enu[1, :] +
                 np.sin(latitude) * enu[2, :])
    enu = np.squeeze(enu)

    xyz_center = XYZ_from_LatLonAlt(latitude, longitude, altitude)
    xyz[0, :] = xyz[0, :] + xyz_center[0]
    xyz[1, :] = xyz[1, :] + xyz_center[1]
    xyz[2, :] = xyz[2, :] + xyz_center[2]
    xyz = np.squeeze(xyz)

    return xyz


def eq2top_m(ha, dec):
    """Return the 3x3 matrix converting equatorial coordinates to topocentric
    at the given hour angle (ha) and declination (dec).
    Borrowed from aipy."""
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    mat = np.array([[sin_H, cos_H, np.zeros_like(ha)],
                    [-sin_d * cos_H, sin_d * sin_H, cos_d],
                    [cos_d * cos_H, -cos_d * sin_H, sin_d]])
    if len(mat.shape) == 3:
        mat = mat.transpose([2, 0, 1])
    return mat


def top2eq_m(ha, dec):
    """Return the 3x3 matrix converting topocentric coordinates to equatorial
    at the given hour angle (ha) and declination (dec).
    Slightly changed from aipy to simply write the matrix instead of inverting."""
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    mat = np.array([[sin_H, -cos_H * sin_d, cos_d * cos_H],
                    [cos_H, sin_d * sin_H, -cos_d * sin_H],
                    [np.zeros_like(ha), cos_d, sin_d]])
    if len(mat.shape) == 3:
        mat = mat.transpose([2, 0, 1])
    return mat


def get_iterable(x):
    """Helper function to ensure iterability."""
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)


def fits_gethduaxis(HDU, axis, strict_fits=True):
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
            import calfits
            calfits._warn_oldcalfits('This file')
            Xi0 = 0
        else:
            raise
    return dX * (np.arange(N) - Xi0) + X0


def fits_indexhdus(hdulist):
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

    Args:
        pol: polarization string

    Returns:
        Number corresponding to string
    """
    poldict = {'I': 1, 'Q': 2, 'U': 3, 'V': 4,
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

    Args:
        num: polarization number

    Returns:
        String corresponding to string
    """
    str_list = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', '', 'I', 'Q', 'U', 'V']
    if isinstance(num, (int, long, np.int32, np.int64)):
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
    if isinstance(jnum, (int, long, np.int32, np.int64)):
        out = str_list[jnum + 8]
    elif isinstance(jnum, collections.Iterable):
            out = [str_list[i + 8] for i in jnum]
    else:
        raise ValueError('Polarization cannot be converted to string.')
    return out


def check_history_version(history, version_string):
    if (version_string.replace(' ', '') in history.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False


def check_histories(history1, history2):
    if (history1.replace('\n', '').replace(' ', '') == history2.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False
