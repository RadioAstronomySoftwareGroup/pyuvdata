# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Commonly used utility functions."""
from __future__ import absolute_import, division, print_function

import numpy as np
import six
import warnings
import copy
from scipy.spatial.distance import pdist, squareform
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.utils import iers

if six.PY2:
    from collections import Iterable
else:
    from collections.abc import Iterable

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
                 'ee': 'ee', 'nn': 'nn', 'en': 'ne', 'ne': 'en',
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


def LatLonAlt_from_XYZ(xyz, check_acceptability=True):
    """
    Calculate lat/lon/alt from ECEF x,y,z.

    Parameters
    ----------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.
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
    # convert to a numpy array
    xyz = np.array(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        raise ValueError('The expected shape of ECEF xyz array is (Npts, 3).')

    else:
        xyz_use = xyz

    if xyz_use.ndim == 1:
        xyz_use = xyz_use[np.newaxis, :]

    # checking for acceptable values
    if check_acceptability:
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

    Parameters
    ----------
    latitude :  ndarray or float
        latitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    longitude :  ndarray or float
        longitude, numpy array (if Npts > 1) or value (if Npts = 1) in radians
    altitude :  ndarray or float
        altitude, numpy array (if Npts > 1) or value (if Npts = 1) in meters

    Returns
    -------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.

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
    rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
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
    rot_matrix = np.array([[np.cos(angle), -1 * np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    return rot_matrix.dot(xyz.T).T


def ENU_from_ECEF(xyz, latitude, longitude, altitude):
    """
    Calculate local ENU (east, north, up) coordinates from ECEF coordinates.

    Parameters
    ----------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.
    latitude : float
        Latitude of center of ENU coordinates in radians.
    longitude : float
        Longitude of center of ENU coordinates in radians.
    altitude : float
        Altitude of center of ENU coordinates in radians.

    Returns
    -------
    ndarray of float
        numpy array, shape (Npts, 3), with local ENU coordinates

    """
    xyz = np.array(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        raise ValueError('The expected shape of ECEF xyz array is (Npts, 3).')

    xyz_in = xyz

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

    return enu


def ECEF_from_ENU(enu, latitude, longitude, altitude):
    """
    Calculate ECEF coordinates from local ENU (east, north, up) coordinates.

    Parameters
    ----------
    enu : ndarray of float
        numpy array, shape (Npts, 3), with local ENU coordinates.
    latitude : float
        Latitude of center of ENU coordinates in radians.
    longitude : float
        Longitude of center of ENU coordinates in radians.
    altitude : float
        Altitude of center of ENU coordinates in radians.


    Returns
    -------
    xyz : ndarray of float
        numpy array, shape (Npts, 3), with ECEF x,y,z coordinates.

    """
    enu = np.array(enu)
    if enu.ndim > 1 and enu.shape[1] != 3:
        raise ValueError('The expected shape of the ENU array is (Npts, 3).')

    enu_use = enu

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

    return xyz


def phase_uvw(ra, dec, initial_uvw):
    """
    Calculate phased uvws/positions from unphased ones in an icrs or gcrs frame.

    This code expects input uvws or positions relative to the telescope
    location in the same frame that ra/dec are in (e.g. icrs or gcrs) and
    returns phased ones in the same frame.

    Note that this code is nearly identical to ENU_from_ECEF, except that it
    uses an arbitrary phasing center rather than a coordinate center.

    Parameters
    ----------
    ra : float
        Right ascension of phase center.
    dec : float
        Declination of phase center.
    initial_uvw : ndarray of float
        Unphased uvws or positions relative to the array center,
        shape (Nlocs, 3).

    Returns
    -------
    uvw : ndarray of float
        uvw array in the same frame as initial_uvws, ra and dec.

    """
    if initial_uvw.ndim == 1:
        initial_uvw = initial_uvw[np.newaxis, :]

    uvw = np.zeros_like(initial_uvw)
    uvw[:, 0] = (-np.sin(ra) * initial_uvw[:, 0]
                 + np.cos(ra) * initial_uvw[:, 1])
    uvw[:, 1] = (-np.sin(dec) * np.cos(ra) * initial_uvw[:, 0]
                 - np.sin(dec) * np.sin(ra) * initial_uvw[:, 1]
                 + np.cos(dec) * initial_uvw[:, 2])
    uvw[:, 2] = (np.cos(dec) * np.cos(ra) * initial_uvw[:, 0]
                 + np.cos(dec) * np.sin(ra) * initial_uvw[:, 1]
                 + np.sin(dec) * initial_uvw[:, 2])
    return(uvw)


def unphase_uvw(ra, dec, uvw):
    """
    Calculate unphased uvws/positions from phased ones in an icrs or gcrs frame.

    This code expects phased uvws or positions in the same frame that ra/dec
    are in (e.g. icrs or gcrs) and returns unphased ones in the same frame.

    Parameters
    ----------
    ra : float
        Right ascension of phase center.
    dec : float
        Declination of phase center.
    uvw : ndarray of float
        Phased uvws or positions relative to the array center,
        shape (Nlocs, 3).

    Returns
    -------
    unphased_uvws : ndarray of float
        Unphased uvws or positions relative to the array center,
        shape (Nlocs, 3).

    """
    if uvw.ndim == 1:
        uvw = uvw[np.newaxis, :]

    unphased_uvws = np.zeros_like(uvw)
    unphased_uvws[:, 0] = (-np.sin(ra) * uvw[:, 0]
                           - np.sin(dec) * np.cos(ra) * uvw[:, 1]
                           + np.cos(dec) * np.cos(ra) * uvw[:, 2])

    unphased_uvws[:, 1] = (np.cos(ra) * uvw[:, 0]
                           - np.sin(dec) * np.sin(ra) * uvw[:, 1]
                           + np.cos(dec) * np.sin(ra) * uvw[:, 2])

    unphased_uvws[:, 2] = (np.cos(dec) * uvw[:, 1]
                           + np.sin(dec) * uvw[:, 2])

    return(unphased_uvws)


def uvcalibrate(uvdata, uvcal, inplace=True, prop_flags=True, flag_missing=True,
                Dterm_cal=False, delay_convention='minus', undo=False):
    """
    Calibrate a UVData object with a UVCal object.

    Parameters
    ----------
    uvdata: UVData object
    uvcal: UVCal object
    inplace: bool, optional
        if True edit uvdata in place, else deepcopy
    prop_flags : bool, optional
        if True, propagate calibration flags to data flags
        and doesn't use flagged gains. Otherwise, uses flagged gains and
        does not propagate calibration flags to data flags.
    flag_missing : bool, optional
        if True, flag baselines in uvdata
        if a participating antenna or polarization is missing in uvcal.
    Dterm_cal : bool, optional
        Calibrate the off-diagonal terms in the Jones matrix if present
        in uvcal. Default is False. Currently not implemented.
    delay_convention : str, optional
        Exponent sign to use in conversion of 'delay' to 'gain' cal_type
        if the input uvcal is not inherently 'gain' cal_type. Default to 'minus'.
    undo : bool, optional
        If True, undo the provided calibration. i.e. apply the calibration with
        flipped gain_convention. Flag propagation rules apply the same.

    Returns
    -------
    UVData, optional
        Returns if not inplace
    """
    # deepcopy for not inplace
    if not inplace:
        uvdata = copy.deepcopy(uvdata)

    # input checks
    if uvcal.cal_type == 'delay':
        # make a copy that is converted to gain
        uvcal = copy.deepcopy(uvcal)
        uvcal.convert_to_gain(delay_convention=delay_convention)

    # D-term calibration
    if Dterm_cal:
        # check for D-terms
        if -7 not in uvcal.jones_array and -8 not in uvcal.jones_array:
            raise ValueError("Cannot apply D-term calibration without -7 or -8"
                             "Jones polarization in uvcal object.")
        raise NotImplementedError("D-term calibration is not yet implemented.")

    # No D-term calibration
    else:
        # iterate over keys
        for key in uvdata.get_antpairpols():
            # get indices for this key
            blt_inds = uvdata.antpair2ind(key)
            pol_ind = np.argmin(np.abs(uvdata.polarization_array - polstr2num(key[2], uvdata.x_orientation)))

            # try to get gains for each antenna
            ant1 = (key[0], key[2][0])
            ant2 = (key[1], key[2][1])
            if not uvcal._has_key(*ant1) or not uvcal._has_key(*ant2):
                if flag_missing:
                    uvdata.flag_array[blt_inds, 0, :, pol_ind] = True
                continue
            gain = (uvcal.get_gains(ant1) * np.conj(uvcal.get_gains(ant2))).T  # tranpose to match uvdata shape
            flag = (uvcal.get_flags(ant1) | uvcal.get_flags(ant2)).T

            # propagate flags
            if prop_flags:
                mask = np.isclose(gain, 0.0) | flag
                gain[mask] = 1.0
                uvdata.flag_array[blt_inds, 0, :, pol_ind] += mask

            # apply to data
            mult_gains = uvcal.gain_convention == 'multiply'
            if undo:
                mult_gains = not mult_gains
            if mult_gains:
                uvdata.data_array[blt_inds, 0, :, pol_ind] *= gain
            else:
                uvdata.data_array[blt_inds, 0, :, pol_ind] /= gain

    # update attributes
    uvdata.history += "\nCalibrated with pyuvdata.utils.uvcalibrate."
    if undo:
        uvdata.vis_units = 'UNCALIB'
    else:
        if uvcal.gain_scale is not None:
            uvdata.vis_units = uvcal.gain_scale

    if not inplace:
        return uvdata


def apply_uvflag(uvd, uvf, inplace=True, unflag_first=False,
                 flag_missing=True, force_pol=True):
    """
    Apply flags from a UVFlag to a UVData instantiation.

    Note that if uvf.Nfreqs or uvf.Ntimes is 1, it will broadcast flags across
    that axis.

    Parameters
    ----------
    uvd : UVData object
        UVData object to add flags to.
    uvf : UVFlag object
        A UVFlag object in flag mode.
    inplace : bool
        If True overwrite flags in uvd, otherwise return new object
    unflag_first : bool
        If True, completely unflag the UVData before applying flags.
        Else, OR the inherent uvd flags with uvf flags.
    flag_missing : bool
        If input uvf is a baseline type and antpairs in uvd do not exist in uvf,
        flag them in uvd. Otherwise leave them untouched.
    force_pol : bool
        If True, broadcast flags to all polarizations if they do not match.
        Only works if uvf.Npols == 1.

    Returns
    -------
    UVData
        If not inplace, returns new UVData object with flags applied

    """
    # assertions
    if uvf.mode != 'flag':
        raise ValueError("UVFlag must be flag mode")

    if not inplace:
        uvd = copy.deepcopy(uvd)

    # make a deepcopy by default b/c it is generally edited inplace downstream
    uvf = copy.deepcopy(uvf)

    # convert to baseline type
    if uvf.type != 'baseline':
        # edits inplace
        uvf.to_baseline(uvd, force_pol=force_pol)

    else:
        # make sure polarizations match or force_pol
        uvd_pols, uvf_pols = uvd.polarization_array.tolist(), uvf.polarization_array.tolist()
        if set(uvd_pols) != set(uvf_pols):
            if uvf.Npols == 1 and force_pol:
                # if uvf is 1pol we can make them match: also edits inplace
                uvf.polarization_array = uvd.polarization_array
                uvf.Npols = len(uvf.polarization_array)
                uvf_pols = uvf.polarization_array.tolist()

            else:
                raise ValueError("Input uvf and uvd polarizations do not match")

        # make sure polarization ordering is correct: also edits inplace
        uvf.polarization_array = uvf.polarization_array[[uvd_pols.index(pol) for pol in uvf_pols]]

    # check time and freq shapes match: if Ntimes or Nfreqs is 1, allow implicit broadcasting
    if uvf.Ntimes == 1:
        mismatch_times = False
    elif uvf.Ntimes == uvd.Ntimes:
        tdiff = np.unique(uvf.time_array) - np.unique(uvd.time_array)
        mismatch_times = np.any(tdiff > np.max(np.abs(uvf._time_array.tols)))
    else:
        mismatch_times = True
    if mismatch_times:
        raise ValueError("UVFlag and UVData have mismatched time arrays.")

    if uvf.Nfreqs == 1:
        mismatch_freqs = False
    elif uvf.Nfreqs == uvd.Nfreqs:
        fdiff = np.unique(uvf.freq_array) - np.unique(uvd.freq_array)
        mismatch_freqs = np.any(fdiff > np.max(np.abs(uvf._freq_array.tols)))
    else:
        mismatch_freqs = True
    if mismatch_freqs:
        raise ValueError("UVFlag and UVData have mismatched frequency arrays.")

    # unflag if desired
    if unflag_first:
        uvd.flag_array[:] = False

    # iterate over antpairs and apply flags: TODO need to be able to handle conjugated antpairs
    uvf_antpairs = uvf.get_antpairs()
    for ap in uvd.get_antpairs():
        uvd_ap_inds = uvd.antpair2ind(ap)
        if ap not in uvf_antpairs:
            if flag_missing:
                uvd.flag_array[uvd_ap_inds] = True
            continue
        uvf_ap_inds = uvf.antpair2ind(*ap)
        # addition of boolean is OR
        uvd.flag_array[uvd_ap_inds] += uvf.flag_array[uvf_ap_inds]

    uvd.history += "\nFlagged with pyuvdata.utils.apply_uvflags."

    if not inplace:
        return uvd


def _get_iterable(x):
    """Return iterable version of input."""
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


def _fits_gethduaxis(HDU, axis):
    """
    Make axis arrays for fits files.

    Parameters
    ----------
    HDU : astropy.io.fits HDU object
        The HDU to make an axis array for.
    axis : int
        The axis number of interest (1-based).

    Returns
    -------
    ndarray of float
        Array of values for the specified axis.

    """
    ax = str(axis)
    N = HDU.header['NAXIS' + ax]
    X0 = HDU.header['CRVAL' + ax]
    dX = HDU.header['CDELT' + ax]
    Xi0 = HDU.header['CRPIX' + ax] - 1

    return dX * (np.arange(N) - Xi0) + X0


def get_lst_for_time(jd_array, latitude, longitude, altitude):
    """
    Get the lsts for a set of jd times at an earth location.

    Parameters
    ----------
    jd_array : ndarray of float
        JD times to get lsts for.
    latitude : float
        Latitude of location to get lst for in degrees.
    longitude : float
        Longitude of location to get lst for in degrees.
    altitude : float
        Altitude of location to get lst for in meters.

    Returns
    -------
    ndarray of float
        LSTs in radians corresponding to the jd_array.

    """
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


def _fits_indexhdus(hdulist):
    """
    Get a dict of table names and HDU numbers from a FITS HDU list.

    Parameters
    ----------
    hdulist : list of astropy.io.fits HDU objects
        List of HDUs to get names for

    Returns
    -------
    dict
        dictionary with table names as keys and HDU number as values.

    """
    tablenames = {}
    for i in range(len(hdulist)):
        try:
            tablenames[hdulist[i].header['EXTNAME']] = i
        except(KeyError):
            continue
    return tablenames


def _x_orientation_rep_dict(x_orientation):
    """Create replacement dict based on x_orientation."""
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
    -------
    int
        Number corresponding to string

    Raises
    ------
    ValueError
        If the pol string cannot be converted to a polarization number.

    Warns
    -----
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
    elif isinstance(pol, Iterable):
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
    -------
    str
        String corresponding to polarization number

    Raises
    ------
    ValueError
        If the polarization number cannot be converted to a polarization string.

    Warns
    -----
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
    elif isinstance(num, Iterable):
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
    -------
    int
        antenna (jones) polarization number corresponding to string

    Raises
    ------
    ValueError
        If the jones string cannot be converted to a polarization number.

    Warns
    -----
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
    elif isinstance(jstr, Iterable):
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
    -------
    str
        antenna (jones) polarization string corresponding to number

    Raises
    ------
    ValueError
        If the jones polarization number cannot be converted to a jones polarization string.

    Warns
    -----
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
    elif isinstance(jnum, Iterable):
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
    -------
    str
        AIPS Memo 117 standard string

    Raises
    ------
    ValueError
        If the pol string cannot be converted to a polarization number.

    Warns
    -----
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
    -------
    str
        calfits memo standard string

    Raises
    ------
    ValueError
        If the jones string cannot be converted to a polarization number.

    Warns
    -----
    UserWarning
        If the x_orientation not recognized.

    """
    return jnum2str(jstr2num(jpolstr, x_orientation=x_orientation),
                    x_orientation=x_orientation)


def conj_pol(pol):
    """
    Return the polarization for the conjugate baseline.

    For example, (1, 2, 'xy') = conj(2, 1, 'yx').
    The returned polarization is determined by assuming the antenna pair is
    reversed in the data, and finding the correct polarization correlation
    which will yield the requested baseline when conjugated. Note this means
    changing the polarization for linear cross-pols, but keeping auto-pol
    (e.g. xx) and Stokes the same.

    Parameters
    ----------
    pol : str or int
        Polarization string or integer.

    Returns
    -------
    cpol : str or int
        Polarization as if antennas are swapped (type matches input)

    """
    cpol_dict = {k.lower(): v for k, v in six.iteritems(CONJ_POL_DICT)}

    if isinstance(pol, str):
        cpol = cpol_dict[pol.lower()]
    elif isinstance(pol, Iterable):
        cpol = [conj_pol(p) for p in pol]
    elif isinstance(pol, six.integer_types + (np.int32, np.int64)):
        cpol = polstr2num(cpol_dict[polnum2str(pol).lower()])
    else:
        raise ValueError('Polarization not recognized, cannot be conjugated.')
    return cpol


def reorder_conj_pols(pols):
    """
    Reorder multiple pols, swapping pols that are conjugates of one another.

    For example ('xx', 'xy', 'yx', 'yy') -> ('xx', 'yx', 'xy', 'yy')
    This is useful for the _key2inds function in the case where an antenna
    pair is specified but the conjugate pair exists in the data. The conjugated
    data should be returned in the order of the polarization axis, so after
    conjugating the data, the pols need to be reordered.
    For example, if a file contains antpair (0, 1) and pols 'xy' and 'yx', but
    the user requests antpair (1, 0), they should get:
    [(1x, 0y), (1y, 0x)] = [conj(0y, 1x), conj(0x, 1y)]

    Parameters
    ----------
    pols : array_like of str or int
        Polarization array (strings or ints).

    Returns
    -------
    conj_order : ndarray of int
        Indices to reorder polarization array.
    """
    if not isinstance(pols, Iterable):
        raise ValueError('reorder_conj_pols must be given an array of polarizations.')
    cpols = np.array([conj_pol(p) for p in pols])  # Array needed for np.where
    conj_order = [np.where(cpols == p)[0][0] if p in cpols else -1 for p in pols]
    if -1 in conj_order:
        raise ValueError('Not all conjugate pols exist in the polarization array provided.')
    return conj_order


def _check_history_version(history, version_string):
    """Check if version_string is present in history string."""
    if (version_string.replace(' ', '') in history.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False


def _check_histories(history1, history2):
    """Check if two histories are the same."""
    if (history1.replace('\n', '').replace(' ', '') == history2.replace('\n', '').replace(' ', '')):
        return True
    else:
        return False


def _combine_histories(history1, history2):
    """Combine histories with minimal repeats."""
    hist2_words = history2.split(' ')
    add_hist = ''
    test_hist1 = ' ' + history1 + ' '
    for i, word in enumerate(hist2_words):
        if ' ' + word + ' ' not in test_hist1:
            add_hist += ' ' + word
            keep_going = (i + 1 < len(hist2_words))
            while keep_going:
                if ((hist2_words[i + 1] == ' ')
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

    Parameters
    ----------
    baseline : int or array_like of ints
        baseline number
    Nant_telescope : int
        number of antennas

    Returns
    -------
    int or array_like of int
        first antenna number(s)
    int or array_like of int
        second antenna number(s)

    """
    if Nants_telescope > 2048:
        raise Exception('error Nants={Nants}>2048 not '
                        'supported'.format(Nants=Nants_telescope))

    baseline = np.asarray(baseline, dtype=np.int64)
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

    Parameters
    ----------
    ant1 : int or array_like of int
        first antenna number
    ant2 : int or array_like of int
        second antenna number
    Nant_telescope : int
        number of antennas
    attempt256 : bool
        Option to try to use the older 256 standard used in
        many uvfits files (will use 2048 standard if there are more
        than 256 antennas). Default is False.

    Returns
    -------
    int or array of int
        baseline number corresponding to the two antenna numbers.

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


def baseline_index_flip(baseline, Nants_telescope):
    """Change baseline number to reverse antenna order."""
    ant1, ant2 = baseline_to_antnums(baseline, Nants_telescope)
    return antnums_to_baseline(ant2, ant1, Nants_telescope)


def get_baseline_redundancies(baselines, baseline_vecs, tol=1.0, with_conjugates=False):
    """
    Find redundant baseline groups.

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
        for bv in baseline_vecs:
            uneg = bv[0] < -tol
            uzer = np.isclose(bv[0], 0.0, atol=tol)
            vneg = bv[1] < -tol
            vzer = np.isclose(bv[1], 0.0, atol=tol)
            wneg = bv[2] < -tol
            conjugates.append(uneg or (uzer and vneg) or (uzer and vzer and wneg))

        conjugates = np.array(conjugates, dtype=bool)
        baseline_vecs[conjugates] *= (-1)
        baseline_ind_conj = baselines[conjugates]
        bl_gps, vec_bin_centers, lens = get_baseline_redundancies(baselines, baseline_vecs, tol=tol, with_conjugates=False)
        return bl_gps, vec_bin_centers, lens, baseline_ind_conj

    # For each baseline, list all others that are within the tolerance distance.
    adj_triu_mat = pdist(baseline_vecs) < tol
    adj = {}    # Adjacency dictionary

    for bi, col in enumerate(squareform(adj_triu_mat)):
        col[bi] = True
        adj[baselines[bi]] = baselines[col]

    # The adjacency list defines a set of graph edges.
    # For each baseline b0, loop over its adjacency list ai \in adj[b0]
    #   If adj[b0] is a subset of adj[ai], then ai is in a redundant group with b0
    bl_gps = []
    for k in adj.keys():
        a0 = adj[k]
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
    if np.sum([len(bg) for bg in bl_gps]) > Nbls:
        raise ValueError("Some baselines are falling into multiple"
                         " redundant groups. Lower the tolerance to resolve ambiguity.")

    return bl_gps, vec_bin_centers, lens


def get_antenna_redundancies(antenna_numbers, antenna_positions, tol=1.0, include_autos=False):
    """
    Find redundant baseline groups based on antenna positions.

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

    Notes
    -----
    The baseline numbers refer to antenna pairs (a1, a2) such that
    the baseline vector formed from ENU antenna positions,
        blvec = enu[a1] - enu[a2]
    is close to the other baselines in the group.

    This is achieved by putting baselines in a form of the u>0
    convention, but with a tolerance in defining the signs of
    vector components.

    To guarantee that the same baseline numbers are present in a UVData
    object, ``UVData.conjugate_bls('u>0', uvw_tol=tol)``, where `tol` is
    the tolerance used here.

    """
    Nants = antenna_numbers.size

    bls = []
    bl_vecs = []

    for aj in range(Nants):
        mini = aj + 1
        if include_autos:
            mini = aj
        for ai in range(mini, Nants):
            anti, antj = antenna_numbers[ai], antenna_numbers[aj]
            bidx = antnums_to_baseline(antj, anti, Nants)
            bv = antenna_positions[ai] - antenna_positions[aj]
            bl_vecs.append(bv)
            bls.append(bidx)
    bls = np.array(bls)
    bl_vecs = np.array(bl_vecs)
    gps, vecs, lens, conjs = get_baseline_redundancies(bls, bl_vecs, tol=tol, with_conjugates=True)
    # Flip the baselines in the groups.
    for gi, gp in enumerate(gps):
        for bi, bl in enumerate(gp):
            if bl in conjs:
                gps[gi][bi] = baseline_index_flip(bl, Nants)

    return gps, vecs, lens


def _reraise_context(fmt, *args):
    r"""
    Reraise an exception with its message modified to specify additional context.

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
    Borrowed from `pwkit <https://github.com/pkgw/pwkit/blob/master/pwkit/__init__.py>`_

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
    """
    Parent function to collapse an array with a given algorithm.

    Parameters
    ----------
    arr : array
        Input array to process.
    alg : str
        Algorithm to use. Must be defined in this function with
        corresponding subfunction below.
    weights: ndarray, optional
        weights for collapse operation (e.g. weighted mean).
        NOTE: Some subfunctions do not use the weights. See corresponding
        doc strings.
    axis : int or tuple, optional
        Axis or axes to collapse. Default is all.
    return_weights : bool
        Whether to return sum of weights.

    """
    collapse_dict = {'mean': mean_collapse, 'absmean': absmean_collapse,
                     'quadmean': quadmean_collapse, 'or': or_collapse,
                     'and': and_collapse}
    try:
        out = collapse_dict[alg](arr, weights=weights, axis=axis,
                                 return_weights=return_weights)
    except KeyError:
        raise ValueError('Collapse algorithm must be one of: '
                         + ', '.join(collapse_dict.keys()) + '.')
    return out


def mean_collapse(arr, weights=None, axis=None, return_weights=False):
    """
    Collapse by averaging data.

    This is similar to np.average, except it handles infs (by giving them
    zero weight) and zero weight axes (by forcing result to be inf with zero
    output weight).

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        weights for average. If none, will default to equal weight for all
        non-infinite data.
    axis : int or tuple, optional
        Axis or axes to collapse (passed to np.sum). Default is all.
    return_weights : bool
        Whether to return sum of weights.

    """
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
    """
    Collapse by averaging absolute value of data.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        weights for average. If none, will default to equal weight for all
        non-infinite data.
    axis : int or tuple, optional
        Axis or axes to collapse (passed to np.sum). Default is all.
    return_weights : bool
        Whether to return sum of weights.

    """
    return mean_collapse(np.abs(arr), weights=weights, axis=axis,
                         return_weights=return_weights)


def quadmean_collapse(arr, weights=None, axis=None, return_weights=False):
    """
    Collapse by averaging in quadrature.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        weights for average. If none, will default to equal weight for all
        non-infinite data.
    axis : int or tuple, optional
        Axis or axes to collapse (passed to np.sum). Default is all.
    return_weights : bool
        Whether to return sum of weights.

    """
    out = mean_collapse(np.abs(arr)**2, weights=weights, axis=axis, return_weights=return_weights)
    if return_weights:
        return np.sqrt(out[0]), out[1]
    else:
        return np.sqrt(out)


def or_collapse(arr, weights=None, axis=None, return_weights=False):
    """
    Collapse using OR operation.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        NOT USED, but kept for symmetry with other collapsing functions.
    axis : int or tuple, optional
        Axis or axes to collapse (take OR over). Default is all.
    return_weights : bool
        Whether to return dummy weights array.
        NOTE: the dummy weights will simply be an array of ones

    """
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
    """
    Collapse using AND operation.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        NOT USED, but kept for symmetry with other collapsing functions.
    axis : int or tuple, optional
        Axis or axes to collapse (take AND over). Default is all.
    return_weights : bool
        Whether to return dummy weights array.
        NOTE: the dummy weights will simply be an array of ones

    """
    if arr.dtype != np.bool:
        raise ValueError('Input to and_collapse function must be boolean array')
    out = np.all(arr, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn('Currently weights are not handled when AND-ing boolean arrays.')
    if return_weights:
        return out, np.ones_like(out, dtype=np.float)
    else:
        return out
