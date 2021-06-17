# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Commonly used utility functions."""
import re
import copy
import warnings
from collections.abc import Iterable

import numpy as np
from scipy.spatial.distance import cdist
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.utils import iers
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as units
from novas import compat as novas
from novas.compat import eph_manager
from . import _utils


def _str_to_bytes(s):
    warnings.warn(
        "_str_to_bytes is deprecated and will be removed in pyuvdata version 2.2. "
        "For an input string s, this function is a thin wrapper on s.encode('utf8'). "
        "The use of encode is preferred over calling this function.",
        DeprecationWarning,
    )
    return s.encode("utf8")


def _bytes_to_str(b):
    warnings.warn(
        "_bytes_to_str is deprecated and will be removed in pyuvdata version 2.2. "
        "For an input string s, this function is a thin wrapper on s.decode('utf8'). "
        "The use of decode is preferred over calling this function.",
        DeprecationWarning,
    )
    return b.decode("utf8")


__all__ = [
    "POL_STR2NUM_DICT",
    "POL_NUM2STR_DICT",
    "CONJ_POL_DICT",
    "JONES_STR2NUM_DICT",
    "JONES_NUM2STR_DICT",
    "LatLonAlt_from_XYZ",
    "XYZ_from_LatLonAlt",
    "rotECEF_from_ECEF",
    "ECEF_from_rotECEF",
    "ENU_from_ECEF",
    "ECEF_from_ENU",
    "phase_uvw",
    "unphase_uvw",
    "uvcalibrate",
    "apply_uvflag",
    "get_lst_for_time",
    "polstr2num",
    "polnum2str",
    "jstr2num",
    "jnum2str",
    "parse_polstr",
    "parse_jpolstr",
    "conj_pol",
    "reorder_conj_pols",
    "baseline_to_antnums",
    "antnums_to_baseline",
    "baseline_index_flip",
    "get_baseline_redundancies",
    "get_antenna_redundancies",
    "collapse",
    "mean_collapse",
    "absmean_collapse",
    "quadmean_collapse",
    "or_collapse",
    "and_collapse",
]

# fmt: off
# polarization constants
# maps polarization strings to polarization integers
POL_STR2NUM_DICT = {"pI": 1, "pQ": 2, "pU": 3, "pV": 4,
                    "I": 1, "Q": 2, "U": 3, "V": 4,  # support straight stokes names
                    "rr": -1, "ll": -2, "rl": -3, "lr": -4,
                    "xx": -5, "yy": -6, "xy": -7, "yx": -8}
# maps polarization integers to polarization strings
POL_NUM2STR_DICT = {1: "pI", 2: "pQ", 3: "pU", 4: "pV",
                    -1: "rr", -2: "ll", -3: "rl", -4: "lr",
                    -5: "xx", -6: "yy", -7: "xy", -8: "yx"}

# maps how polarizations change when antennas are swapped
CONJ_POL_DICT = {"xx": "xx", "yy": "yy", "xy": "yx", "yx": "xy",
                 "ee": "ee", "nn": "nn", "en": "ne", "ne": "en",
                 "rr": "rr", "ll": "ll", "rl": "lr", "lr": "rl",
                 "I": "I", "Q": "Q", "U": "U", "V": "V",
                 "pI": "pI", "pQ": "pQ", "pU": "pU", "pV": "pV"}

# maps jones matrix element strings to jones integers
# Add entries that don't start with "J" to allow shorthand versions
JONES_STR2NUM_DICT = {"Jxx": -5, "Jyy": -6, "Jxy": -7, "Jyx": -8,
                      "xx": -5, "x": -5, "yy": -6, "y": -6, "xy": -7, "yx": -8,
                      "Jrr": -1, "Jll": -2, "Jrl": -3, "Jlr": -4,
                      "rr": -1, "r": -1, "ll": -2, "l": -2, "rl": -3, "lr": -4}
# maps jones integers to jones matrix element strings
JONES_NUM2STR_DICT = {-1: "Jrr", -2: "Jll", -3: "Jrl", -4: "Jlr",
                      -5: "Jxx", -6: "Jyy", -7: "Jxy", -8: "Jyx"}

# maps uvdata pols to input feed polarizations
POL_TO_FEED_DICT = {"xx": ["x", "x"], "yy": ["y", "y"],
                    "xy": ["x", "y"], "yx": ["y", "x"],
                    "ee": ["e", "e"], "nn": ["n", "n"],
                    "en": ["e", "n"], "ne": ["n", "e"],
                    "rr": ["r", "r"], "ll": ["l", "l"],
                    "rl": ["r", "l"], "lr": ["l", "r"]}

# fmt: on


def _get_iterable(x):
    """Return iterable version of input."""
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


def _fits_gethduaxis(hdu, axis):
    """
    Make axis arrays for fits files.

    Parameters
    ----------
    hdu : astropy.io.fits HDU object
        The HDU to make an axis array for.
    axis : int
        The axis number of interest (1-based).

    Returns
    -------
    ndarray of float
        Array of values for the specified axis.

    """
    ax = str(axis)
    axis_num = hdu.header["NAXIS" + ax]
    val = hdu.header["CRVAL" + ax]
    delta = hdu.header["CDELT" + ax]
    index = hdu.header["CRPIX" + ax] - 1

    return delta * (np.arange(axis_num) - index) + val


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
            tablenames[hdulist[i].header["EXTNAME"]] = i
        except (KeyError):
            continue
    return tablenames


def _get_fits_extra_keywords(header, keywords_to_skip=None):
    """
    Get any extra keywords and return as dict.

    Parameters
    ----------
    header : FITS header object
        header object to get extra_keywords from.
    keywords_to_skip : list of str
        list of keywords to not include in extra keywords in addition to standard
        FITS keywords.

    Returns
    -------
    dict
        dict of extra keywords.
    """
    # List standard FITS header items that are still should not be included in
    # extra_keywords
    # These are the beginnings of FITS keywords to ignore, the actual keywords
    # often include integers following these names (e.g. NAXIS1, CTYPE3)
    std_fits_substrings = [
        "HISTORY",
        "SIMPLE",
        "BITPIX",
        "EXTEND",
        "BLOCKED",
        "GROUPS",
        "PCOUNT",
        "BSCALE",
        "BZERO",
        "NAXIS",
        "PTYPE",
        "PSCAL",
        "PZERO",
        "CTYPE",
        "CRVAL",
        "CRPIX",
        "CDELT",
        "CROTA",
        "CUNIT",
    ]

    if keywords_to_skip is not None:
        std_fits_substrings.extend(keywords_to_skip)

    extra_keywords = {}
    # find all the other header items and keep them as extra_keywords
    for key in header:
        # check if key contains any of the standard FITS substrings
        if np.any([sub in key for sub in std_fits_substrings]):
            continue
        if key == "COMMENT":
            extra_keywords[key] = str(header.get(key))
        elif key != "":
            extra_keywords[key] = header.get(key)

    return extra_keywords


def _check_history_version(history, version_string):
    """Check if version_string is present in history string."""
    if version_string.replace(" ", "") in history.replace("\n", "").replace(" ", ""):
        return True
    else:
        return False


def _check_histories(history1, history2):
    """Check if two histories are the same."""
    if history1.replace("\n", "").replace(" ", "") == history2.replace(
        "\n", ""
    ).replace(" ", ""):
        return True
    else:
        return False


def _combine_history_addition(history1, history2):
    """
    Find extra history to add to have minimal repeats.

    Parameters
    ----------
    history1 : str
        First history.
    history2 : str
        Second history

    Returns
    -------
    str
        Extra history to add to first history.

    """
    # first check if they're the same to avoid more complicated processing.
    if _check_histories(history1, history2):
        return None

    hist2_words = history2.split(" ")
    add_hist = ""
    test_hist1 = " " + history1 + " "
    for i, word in enumerate(hist2_words):
        if " " + word + " " not in test_hist1:
            add_hist += " " + word
            keep_going = i + 1 < len(hist2_words)
            while keep_going:
                if (hist2_words[i + 1] == " ") or (
                    " " + hist2_words[i + 1] + " " not in test_hist1
                ):
                    add_hist += " " + hist2_words[i + 1]
                    del hist2_words[i + 1]
                    keep_going = i + 1 < len(hist2_words)
                else:
                    keep_going = False

    if add_hist == "":
        add_hist = None
    return add_hist


def baseline_to_antnums(baseline, Nants_telescope):
    """
    Get the antenna numbers corresponding to a given baseline number.

    Parameters
    ----------
    baseline : int or array_like of ints
        baseline number
    Nants_telescope : int
        number of antennas

    Returns
    -------
    int or array_like of int
        first antenna number(s)
    int or array_like of int
        second antenna number(s)

    """
    if Nants_telescope > 2048:
        raise Exception(
            "error Nants={Nants}>2048 not supported".format(Nants=Nants_telescope)
        )

    return_array = isinstance(baseline, (np.ndarray, list, tuple))
    ant1, ant2 = _utils.baseline_to_antnums(
        np.ascontiguousarray(baseline, dtype=np.int64)
    )
    if return_array:
        return ant1, ant2
    else:
        return ant1.item(0), ant2.item(0)


def antnums_to_baseline(ant1, ant2, Nants_telescope, attempt256=False):
    """
    Get the baseline number corresponding to two given antenna numbers.

    Parameters
    ----------
    ant1 : int or array_like of int
        first antenna number
    ant2 : int or array_like of int
        second antenna number
    Nants_telescope : int
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
    if Nants_telescope is not None and Nants_telescope > 2048:
        raise Exception(
            "cannot convert ant1, ant2 to a baseline index "
            "with Nants={Nants}>2048.".format(Nants=Nants_telescope)
        )

    return_array = isinstance(ant1, (np.ndarray, list, tuple))
    baseline = _utils.antnums_to_baseline(
        np.ascontiguousarray(ant1, dtype=np.int64),
        np.ascontiguousarray(ant2, dtype=np.int64),
        attempt256=attempt256,
    )
    if return_array:
        return baseline
    else:
        return baseline.item(0)


def baseline_index_flip(baseline, Nants_telescope):
    """Change baseline number to reverse antenna order."""
    ant1, ant2 = baseline_to_antnums(baseline, Nants_telescope)
    return antnums_to_baseline(ant2, ant1, Nants_telescope)


def _x_orientation_rep_dict(x_orientation):
    """Create replacement dict based on x_orientation."""
    if x_orientation.lower() == "east" or x_orientation.lower() == "e":
        return {"x": "e", "y": "n"}
    elif x_orientation.lower() == "north" or x_orientation.lower() == "n":
        return {"x": "n", "y": "e"}
    else:
        raise ValueError("x_orientation not recognized.")


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
            for key, value in POL_STR2NUM_DICT.items():
                new_key = key.replace("x", rep_dict["x"]).replace("y", rep_dict["y"])
                dict_use[new_key] = value
        except ValueError:
            warnings.warn("x_orientation not recognized.")

    poldict = {k.lower(): v for k, v in dict_use.items()}
    if isinstance(pol, str):
        out = poldict[pol.lower()]
    elif isinstance(pol, Iterable):
        out = [poldict[key.lower()] for key in pol]
    else:
        raise ValueError(
            "Polarization {p} cannot be converted to a polarization number.".format(
                p=pol
            )
        )
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
            for key, value in POL_NUM2STR_DICT.items():
                new_val = value.replace("x", rep_dict["x"]).replace("y", rep_dict["y"])
                dict_use[key] = new_val
        except ValueError:
            warnings.warn("x_orientation not recognized.")

    if isinstance(num, (int, np.int32, np.int64)):
        out = dict_use[num]
    elif isinstance(num, Iterable):
        out = [dict_use[i] for i in num]
    else:
        raise ValueError(
            "Polarization {p} cannot be converted to string.".format(p=num)
        )
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
            for key, value in JONES_STR2NUM_DICT.items():
                new_key = key.replace("x", rep_dict["x"]).replace("y", rep_dict["y"])
                dict_use[new_key] = value
        except ValueError:
            warnings.warn("x_orientation not recognized.")

    jdict = {k.lower(): v for k, v in dict_use.items()}
    if isinstance(jstr, str):
        out = jdict[jstr.lower()]
    elif isinstance(jstr, Iterable):
        out = [jdict[key.lower()] for key in jstr]
    else:
        raise ValueError(
            "Jones polarization {j} cannot be converted to index.".format(j=jstr)
        )
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
        If the jones polarization number cannot be converted to a jones
        polarization string.

    Warns
    -----
    UserWarning
        If the x_orientation not recognized.

    """
    dict_use = copy.deepcopy(JONES_NUM2STR_DICT)
    if x_orientation is not None:
        try:
            rep_dict = _x_orientation_rep_dict(x_orientation)
            for key, value in JONES_NUM2STR_DICT.items():
                new_val = value.replace("x", rep_dict["x"]).replace("y", rep_dict["y"])
                dict_use[key] = new_val
        except ValueError:
            warnings.warn("x_orientation not recognized.")

    if isinstance(jnum, (int, np.int32, np.int64)):
        out = dict_use[jnum]
    elif isinstance(jnum, Iterable):
        out = [dict_use[i] for i in jnum]
    else:
        raise ValueError(
            "Jones polarization {j} cannot be converted to string.".format(j=jnum)
        )
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
    return polnum2str(
        polstr2num(polstr, x_orientation=x_orientation), x_orientation=x_orientation
    )


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
    return jnum2str(
        jstr2num(jpolstr, x_orientation=x_orientation), x_orientation=x_orientation
    )


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
    cpol_dict = {k.lower(): v for k, v in CONJ_POL_DICT.items()}

    if isinstance(pol, str):
        cpol = cpol_dict[pol.lower()]
    elif isinstance(pol, Iterable):
        cpol = [conj_pol(p) for p in pol]
    elif isinstance(pol, (int, np.int32, np.int64)):
        cpol = polstr2num(cpol_dict[polnum2str(pol).lower()])
    else:
        raise ValueError("Polarization not recognized, cannot be conjugated.")
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
        raise ValueError("reorder_conj_pols must be given an array of polarizations.")
    cpols = np.array([conj_pol(p) for p in pols])  # Array needed for np.where
    conj_order = [np.where(cpols == p)[0][0] if p in cpols else -1 for p in pols]
    if -1 in conj_order:
        raise ValueError(
            "Not all conjugate pols exist in the polarization array provided."
        )
    return conj_order


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
    xyz = np.asarray(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        raise ValueError("The expected shape of ECEF xyz array is (Npts, 3).")

    squeeze = xyz.ndim == 1

    if squeeze:
        xyz = xyz[np.newaxis, :]

    xyz = np.ascontiguousarray(xyz.T, dtype=np.float64)

    # checking for acceptable values
    if check_acceptability:
        norms = np.linalg.norm(xyz, axis=0)
        if not all(np.logical_and(norms >= 6.35e6, norms <= 6.39e6)):
            raise ValueError("xyz values should be ECEF x, y, z coordinates in meters")
    # this helper function returns one 2D array because it is less overhead for cython
    lla = _utils._lla_from_xyz(xyz)

    if squeeze:
        return lla[0, 0], lla[1, 0], lla[2, 0]
    return lla[0], lla[1], lla[2]


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
    latitude = np.ascontiguousarray(latitude, dtype=np.float64)
    longitude = np.ascontiguousarray(longitude, dtype=np.float64)
    altitude = np.ascontiguousarray(altitude, dtype=np.float64)

    n_pts = latitude.size

    if longitude.size != n_pts:
        raise ValueError(
            "latitude, longitude and altitude must all have the same length"
        )
    if altitude.size != n_pts:
        raise ValueError(
            "latitude, longitude and altitude must all have the same length"
        )

    xyz = _utils._xyz_from_latlonalt(latitude, longitude, altitude)
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
    sensible_radius_range = (6.35e6, 6.39e6)
    if np.any(ecef_magnitudes <= sensible_radius_range[0]) or np.any(
        ecef_magnitudes >= sensible_radius_range[1]
    ):
        raise ValueError(
            "ECEF vector magnitudes must be on the order of the radius of the earth"
        )

    # the cython utility expects (3, Npts) for faster manipulation
    # transpose after we get the array back to match the expected shape
    enu = _utils._ENU_from_ECEF(
        xyz,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
    )

    enu = enu.T
    if squeeze:
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
    xyz = _utils._ECEF_from_ENU(
        enu,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
    )
    xyz = xyz.T
    if squeeze:
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

    return _utils._phase_uvw(
        np.float64(ra),
        np.float64(dec),
        np.ascontiguousarray(initial_uvw.T, dtype=np.float64),
    ).T


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

    return _utils._unphase_uvw(
        np.float64(ra), np.float64(dec), np.ascontiguousarray(uvw.T, dtype=np.float64),
    ).T


def rotate_one_axis(xyz_array, rot_amount, rot_axis):
    """
    Rotate an array of 3D positions around the a single axis (x, y, or z).

    This function performs a basic rotation of 3D vectors about one of the priciple
    axes -- the x-axis, the y-axis, or the z-axis.

    Note that the rotations here obey the right-hand rule -- that is to say, from the
    perspective of the positive side of the axis of rotation, a positive rotation will
    cause points on the plane intersecting this axis to move in a counter-clockwise
    fashion.

    Parameters
    ----------
    xyz_array : ndarray of float
        Set of 3-dimensional vectors be rotated, in typical right-handed cartesian
        order, e.g. (x, y, z). Shape is (Nvectors, 3), or (3,) if Nvectors == 1.
    rot_amount : float or ndarray of float
        Amount (in radians) to rotate the given set of coordinates. Can either be a
        single float (or ndarray of shape (1,)) if rotating all vectors by the same
        amount, otherwise expected to be shape (Nvectors,).
    rot_axis : int
        Axis around which the rotation is applied. 0 is the x-axis, 1 is the y-axis,
        and 2 is the z-axis.

    Returns
    -------
    rotated_xyz : ndarray of float
        Set of rotated 3-dimensional vectors, shape (Nvector, 3).
    """
    # Make the rotation vectors consistent w/ what we expect (ndarray of float64).
    # The promotion of values to float64 is to suppress numerical precision issues,
    # since the matrix math can - in limited circumstances - introduce precision errors
    # of order 10x the limiting numerical precision of the float. For a float32/single,
    # thats a part in 1e6 (~arcsec-level errors), but for a float64 it translates to
    # a part in 1e15.
    if not isinstance(rot_amount, np.ndarray):
        rot_amount = np.array([rot_amount], dtype=np.float64)
    elif not isinstance(rot_amount[0].dtype, np.float64):
        rot_amount = rot_amount.astype(np.float64)

    if xyz_array.ndim == 1:
        xyz_array.shape = (1,) + xyz_array.shape
    if not isinstance(xyz_array[0, 0], np.float64):
        xyz_array = xyz_array.astype(np.float64)

    # Check and see how big of a rotation matrix we need
    n_rot = rot_amount.shape[0]
    rot_matrix = np.zeros((3, 3, n_rot), dtype=np.float64)

    # Figure out which pieces of the matrix we need to update
    temp_jdx = (rot_axis + 1) % 3
    temp_idx = (rot_axis + 2) % 3

    # Fill in the rotation matricies accordingly
    rot_matrix[rot_axis, rot_axis] = 1
    rot_matrix[temp_idx, temp_idx] = np.cos(rot_amount)
    rot_matrix[temp_jdx, temp_jdx] = rot_matrix[temp_idx, temp_idx]
    rot_matrix[temp_idx, temp_jdx] = np.sin(rot_amount)
    rot_matrix[temp_jdx, temp_idx] = -rot_matrix[temp_idx, temp_jdx]

    # This piece below could be potentially cythonized, although the numpy
    # implementation of matrix multiplication is already _pretty_ well-optimized.
    # On a quad-core 2.5 GHz Macbook, does about 10^7 vector rotations/sec.
    rotated_xyz = np.matmul(
        np.transpose(rot_matrix, [2, 0, 1]), xyz_array[:, :, np.newaxis]
    )[:, :, 0]

    return rotated_xyz


def rotate_two_axis(xyz_array, rot_amount1, rot_amount2, rot_axis1, rot_axis2):
    """
    Rotate an array of 3D positions sequentially around a pair of axes (x, y, or z).

    This function performs a sequential pair of basic rotations of 3D vectors about
    the priciple axes -- the x-axis, the y-axis, or the z-axis.

    Note that the rotations here obey the right-hand rule -- that is to say, from the
    perspective of the positive side of the axis of rotation, a positive rotation will
    cause points on the plane intersecting this axis to move in a counter-clockwise
    fashion.

    Parameters
    ----------
    xyz_array : ndarray of float
        Set of 3-dimensional vectors be rotated, in typical right-handed cartesian
        order, e.g. (x, y, z). Shape is (Nvectors, 3), or (3,) if Nvectors == 1.
    rot_amount1 : float or ndarray of float
        Amount (in radians) of rotatation to apply during the first rotation of the
        sequence, to the given set of coordinates. Can either be a single float (or
        ndarray of shape (1,)) if rotating all vectors by the same amount, otherwise
        expected to be shape (Nvectors,).
    rot_amount2 : float or ndarray of float
        Amount (in radians) of rotatation to apply during the second rotation of the
        sequence, to the given set of coordinates. Can either be a single float (or
        ndarray of shape (1,)) if rotating all vectors by the same amount, otherwise
        expected to be shape (Nvectors,).
    rot_axis1 : int
        Axis around which the first rotation is applied. 0 is the x-axis, 1 is the
        y-axis, and 2 is the z-axis.
    rot_axis2 : int
        Axis around which the second rotation is applied. 0 is the x-axis, 1 is the
        y-axis, and 2 is the z-axis.

    Returns
    -------
    rotated_xyz : ndarray of float
        Set of rotated 3-dimensional vectors, shape (Nvector, 3).

    """
    # Make the rotation vectors consistent w/ what we expect (ndarray of float64).
    # The promotion of values to float64 is to suppress numerical precision issues,
    # since the matrix math can - in limited circumstances - introduce precision errors
    # of order 10x the limiting numerical precision of the float. For a float32/single,
    # thats a part in 1e6 (~arcsec-level errors), but for a float64 it translates to
    # a part in 1e15.
    if not isinstance(rot_amount1, np.ndarray):
        rot_amount1 = np.array([rot_amount1], dtype=np.float64)
    elif not isinstance(rot_amount1[0].dtype, np.float64):
        rot_amount1 = rot_amount1.astype(np.float64)
    if not isinstance(rot_amount2, np.ndarray):
        rot_amount2 = np.array([rot_amount2], dtype=np.float64)
    elif not isinstance(rot_amount2[0].dtype, np.float64):
        rot_amount2 = rot_amount2.astype(np.float64)

    # Make sure that the xyz_array is shaped in the way what we want
    if xyz_array.ndim == 1:
        xyz_array = xyz_array[np.newaxis, :]
    if not isinstance(xyz_array[0, 0], np.float64):
        xyz_array = xyz_array.astype(np.float64)

    # Capture the case where someone wants to do a sequence of rotations on the same
    # axis. Also known as just rotating a single axis.
    if rot_axis1 == rot_axis2:
        return rotate_one_axis(xyz_array, rot_amount1 + rot_amount2, rot_axis1)

    # Figure out how many individual rotation matricies we need
    n_rot = max(rot_amount1.shape[0], rot_amount2.shape[0])

    rot_matrix = np.empty((3, 3, n_rot), dtype=np.float64)
    # There are two permulations per pair of axes -- when the pair is right-hand
    # oriented vs left-hand oriented. Check here which one it is. For example,
    # rotating first on the x-axis, second on the y-axis is considered a
    # "right-handed" pair, whereas z-axis first, then y-axis would be considered
    # a "left-handed" pair.
    lhd_order = np.mod(rot_axis2 - rot_axis1, 3) != 1

    temp_idx = [
        np.mod(rot_axis1 - lhd_order, 3),
        np.mod(rot_axis1 + 1 - lhd_order, 3),
        np.mod(rot_axis1 + 2 - lhd_order, 3),
    ]

    # We're using lots of sin and cos calculations -- doing them once upfront saves
    # quite a bit of time by eliminating redundant calculations
    sin_lo = np.sin(rot_amount2 if lhd_order else rot_amount1)
    cos_lo = np.cos(rot_amount2 if lhd_order else rot_amount1)
    sin_hi = np.sin(rot_amount1 if lhd_order else rot_amount2)
    cos_hi = np.cos(rot_amount1 if lhd_order else rot_amount2)

    # Take care of the diagonal terms first, since they aren't actually affected by the
    # order of rotational opertations
    rot_matrix[temp_idx[0], temp_idx[0]] = cos_hi
    rot_matrix[temp_idx[1], temp_idx[1]] = cos_lo
    rot_matrix[temp_idx[2], temp_idx[2]] = cos_lo * cos_hi

    # Now time for the off-diagonal terms, as a set of 3 pairs. The rotation matrix
    # for a left-hand oriented pair of rotation axes (e.g., x-rot, then y-rot) is just
    # a transpose of the right-hand orientation of the same pair (e.g., y-rot, then
    # x-rot).
    rot_matrix[temp_idx[0 + lhd_order], temp_idx[1 - lhd_order]] = sin_lo * sin_hi
    rot_matrix[temp_idx[0 - lhd_order], temp_idx[lhd_order - 1]] = (
        cos_lo * sin_hi * ((-1.0) ** lhd_order)
    )

    rot_matrix[temp_idx[1 - lhd_order], temp_idx[0 + lhd_order]] = 0.0
    rot_matrix[temp_idx[1 + lhd_order], temp_idx[2 - lhd_order]] = sin_lo * (
        (-1.0) ** (1 + lhd_order)
    )

    rot_matrix[temp_idx[lhd_order - 1], temp_idx[0 - lhd_order]] = sin_hi * (
        (-1.0) ** (1 + lhd_order)
    )
    rot_matrix[temp_idx[2 - lhd_order], temp_idx[1 + lhd_order]] = (
        sin_lo * cos_hi * ((-1.0) ** (lhd_order))
    )

    # This piece below could be potentially cythonized, although the numpy
    # implementation of matrix multiplication is already _pretty_ well-optimized.
    # On a quad-core 2.5 GHz Macbook, does about 10^7 vector rotations/sec.
    rotated_xyz = np.matmul(
        np.transpose(rot_matrix, [2, 0, 1]), xyz_array[:, :, np.newaxis]
    )[:, :, 0]

    return rotated_xyz


def calc_uvw(
    app_ra=None,
    app_dec=None,
    lst_array=None,
    uvw_array=None,
    antenna_positions=None,
    ant_1_array=None,
    ant_2_array=None,
    old_app_ra=None,
    old_app_dec=None,
    telescope_lon=None,
    telescope_lat=None,
    from_enu=False,
    to_enu=False,
):
    """
    Calculate an array of baseline coordinates, in either uvw or ENU.

    This routine is meant as a convenience function for producing baseline coordinates
    based under a few different circumstances:
    1) Calculating ENU coordinates using antenna positions
    2) Calculating uwv coordinates at a given sky position using antenna positions
    3) Converting from ENU coordinates to uvw coordinates
    4) Converting from uvw coordinate to ENU coordinates
    5) Converting from uvw coordinates at one sky position to another sky position

    Different conversion pathways have different parameters that are required.

    Parameters
    ----------
    app_ra : ndarray of float
        Apparent RA of the target phase center, required if calculating baseline
        coordinates in uvw-space (vs ENU-space). Shape is (Nblts,).
    app_ra : ndarray of float
        Apparent declination of the target phase center, required if calculating
        baseline coordinates in uvw-space (vs ENU-space). Shape is (Nblts,).
    old_app_ra : ndarray of float
        Apparent RA of the previous phase center, required if not deriving baseline
        coordinates from antenna positions and from_enu=False. Shape is (Nblts,).
    old_app_dec : ndarray of float
        Apparent declination of the previous phase center, required if not deriving
        baseline coordinates from antenna positions and from_enu=False. Shape is
        (Nblts,).
    lst_array : ndarray of float
        Local apparent sidereal time, required if deriving baseline coordinates from
        antenna positions, or converting to/from ENU coordinates. Shape is (Nblts,).
    uvw_array : ndarray of float
        Array of previous baseline coordinates (in either uvw or ENU), required if
        not deriving new coordinates from antenna positions.  Shape is (Nblts, 3).
    antenna_positions : ndarray of float
        List of antenna positions relative to array center, required if not providing
        uvw_array. Shape is (Nants, 3).
    ant_1_array : ndarray of int
        Indexing map for matching which antennas are associated with the first antenna
        in the pair for all baselines, used for selecting the relevant (zero-index)
        entries from antenna_positions, required if not providing uvw_array. Shape is
        (Nblts,).
    ant_2_array : ndarray of int
        Indexing map for matching which antennas are associated with the second antenna
        in the pair for all baselines, used for selecting the relevant (zero-index)
        entries from antenna_positions, required if not providing uvw_array. Shape is
        (Nblts,).
    telescope_lon : float
        Longitude of the phase center, units radians, required if deriving baseline
        coordinates from antenna positions, or converting to/from ENU coordinates.
    telescope_lat : float
        Latitude of the phase center, units radians, required if deriving baseline
        coordinates from antenna positions, or converting to/from ENU coordinates.
    from_enu : boolean
        Set to True if uvw_array is expressed in ENU coordinates. Default is False.
    to_enu : boolean
        Set to True if you would like the output expressed in EN coordinates. Default
        is False.

    Returns
    -------
    new_coords : ndarray of float64
        Set of baseline coordinates, shape (Nblts, 3).
    """
    if to_enu:
        if lst_array is None:
            raise ValueError(
                "Must include lst_array to calculate baselines" "in ENU coordinates!"
            )
        if telescope_lat is None:
            raise ValueError(
                "Must include telescope_lon to calculate baselines"
                "in ENU coordinates!"
            )
        # Unphased coordinates appear to be stored in ENU coordinates -- that's
        # equivalent to calculating uvw's based on zenith. We can use that to our
        # advantage and spoof the app_ra and app_dec based on LST and telescope_lat
        app_ra = lst_array
        app_dec = telescope_lon + np.zeros_like(lst_array)
    else:
        if (app_ra is None) or (app_dec is None):
            raise ValueError(
                "Must include app_ra and app_dec to calculate baselines in uvw "
                "coordinates!"
            )

    if uvw_array is None:
        # Assume at this point we are dealing w/ antenna positions
        if antenna_positions is None:
            raise ValueError(
                "Must include antenna_positions if not providing uvw_array!"
            )
        if (ant_1_array is None) or (ant_2_array is None):
            raise ValueError(
                "Must include ant_1_array and ant_2_array if not providing uvw_array!"
            )
        if lst_array is None:
            raise ValueError("Must include lst_array if not providing uvw_array!")
        if telescope_lon is None:
            raise ValueError("Must include telescope_lon if not providing uvw_array!")

        N_ants = antenna_positions.shape[0]
        # Use the app_ra, app_dec, and lst_array arrays to figure out how many unique
        # rotations are actually needed. If the ratio of Nblts to number of unique
        # entries is favorable, we can just rotate the antenna positions and save
        # outselves a bit of work.
        unique_mask = np.append(
            True,
            (
                (app_ra[:-1] != app_ra[1:])
                | (app_dec[:-1] != app_dec[1:])
                | (lst_array[:-1] != lst_array[1:])
            ),
        )

        # GHA -> Hour Angle as measured at Greenwich (because antenna coords are
        # centered such that x-plane intersects the meridian at longitude 0).
        unique_gha = np.repeat(
            (lst_array[unique_mask] - app_ra[unique_mask]) + telescope_lon, N_ants
        )
        unique_dec = np.repeat(app_dec[unique_mask], N_ants)
        N_unique = len(unique_gha) // N_ants

        # This is the last chance to bail on the "optimized" method -- if the number
        # of calculations required to rotate all antennas is more than that required
        # for rotating each of the individual baselines, then go the "direct" route.
        # N_ants here is the number of rotations needed per unique triplet of RA, Dec,
        # and GHA.
        if N_unique > (len(unique_mask) / N_ants):
            uvw_array = antenna_positions[ant_2_array] - antenna_positions[ant_1_array]
            gha_array = (lst_array - app_ra) + telescope_lon
            new_coords = rotate_two_axis(uvw_array, gha_array, app_dec, 2, 1)
        else:
            ant_vectors = np.tile(antenna_positions, (N_unique, 1))
            ant_rot_vectors = rotate_two_axis(ant_vectors, unique_gha, unique_dec, 2, 1)
            unique_mask[0] = False
            unique_map = np.cumsum(unique_mask) * N_ants

            new_coords = (
                ant_rot_vectors[unique_map + ant_2_array]
                - ant_rot_vectors[unique_map + ant_1_array]
            )
    else:
        if from_enu:
            if (telescope_lat is None) or (telescope_lon is None):
                raise ValueError(
                    "Must include telescope_lat and telescope_lat if moving between "
                    'ENU (i.e., "unphased") and uvw coordinates!'
                )
            if lst_array is None:
                raise ValueError(
                    'Must include lst_array if moving between ENU (i.e., "unphased") '
                    "and uvw coordinates!"
                )
            # Unphased coordinates appear to be stored in ENU coordinates -- that's
            # equivalent to calculating uvw's based on zenith. We can use that to our
            # advantage and spoof old_app_ra and old_app_dec based on lst_array and
            # telescope_lat
            old_app_ra = lst_array
            old_app_dec = np.zeros_like(lst_array) + telescope_lat

        if (old_app_ra is None) or (old_app_dec is None):
            raise ValueError(
                "Must include old_ra and old_dec values if data are phased!"
            )
        # For this operation, all we need is the delta-ha coverage, which _should_ be
        # entirely encapsulated by the change in RA.
        gha_delta_array = old_app_ra - app_ra

        # Notice that there's an axis re-orientation here, to go from uvw -> XYZ,
        # where X is pointing in the direction of the source. This is mostly here
        # for convenience and code legibility -- a slightly different pair of
        # rotations would give you the same result w/o needing to cycle the axes.
        new_coords = rotate_two_axis(
            rotate_one_axis(uvw_array[:, [2, 0, 1]], -old_app_dec, 1),
            gha_delta_array,
            app_dec,
            2,
            0,
        )
    # There's one last task to do, which is to re-align the axes from projected
    # XYZ -> uvw, where X (which points towards the source) falls on the w axis,
    # and Y and Z fall on the u and v axes, respectively.
    return new_coords[:, [1, 2, 0]]


def translate_sidereal_to_icrs(
    lon_coord, lat_coord, coord_frame, coord_epoch=None, time_array=None,
):
    """
    Translate a given set of coordinates from a specified frame into ICRS.

    Uses astropy to convert from a coordinates of a sidereal source into ICRS, which is
    what is used for calculating apparent right ascension and decliation coordinates.
    This function will support transforms from several frames, including GCRS,
    FK5 (i.e., J2000), FK4 (i.e., B1950), Galactic, Supergalactic, CIRS, HCRS, and
    a few others (basically anything that doesn't require knowing the observers
    location on Earth/other celestial body).

    Parameters
    ----------
    lon_coord : float or ndarray of floats
        Logitudinal coordinate to be transformed, typically expressed as the right
        ascension, in units of radians. Can either be a float, or an ndarray of
        floats with shape (Ncoords,).
    lat_coord : float or ndarray of floats
        Latitudinal coordinate to be transformed, typically expressed as the
        declination, in units of radians. Can either be a float, or an ndarray of
        floats with shape (Ncoords,).
    coord_frame : string
        Reference frame for the provided coordinates.  Expected to match a list of
        those supported within the astropy SkyCoord object. An incomplete list includes
        'gcrs', 'fk4', 'fk5', 'galactic', 'supergalactic', 'cirs', and 'hcrs'.
    coord_epoch : float
        Epoch for the provided coordinate frame. Optional parameter, only required
        when using either the FK4 (B1950) or FK5 (J2000) coordinate systems. Units are
        in fractional years.
    time_array : float or ndarray of floats
        Julian date(s) to which the coordinates correspond to, only used in frames
        with annular motion terms (e.g., abberation in GCRS). Can either be a float
        or an ndarray of floats with shape (Ntimes,).

    Returns
    -------
    icrs_ra : ndarray of floats
        ICRS right ascension coordinates, in units of radians, of either shape
        (Ncoords,) or (Ntimes,), depending on input parameters.
    icrs_dec : ndarray of floats
        ICRS declination coordinates, in units of radians, of either shape
        (Ncoords,) or (Ntimes,), depending on input parameters.
    """
    # Make life easier and turn these items into arrays if they're just bare floats
    if not isinstance(lon_coord, np.ndarray):
        lon_coord = np.array([lon_coord])
    if not isinstance(lat_coord, np.ndarray):
        lat_coord = np.array([lat_coord])

    # Make sure that the inputs have sensible lengths
    if len(lon_coord) != len(lat_coord):
        raise ValueError("Length of lon_coord and lat_coord must be the same!")

    # Make sure that time array matched up with what we expect. Thanks to astropy
    # weirdness, time_array has to be the same length as lat/lon coords
    if time_array is not None:
        if not isinstance(time_array, np.ndarray):
            time_array = np.ndarray([time_array])
        if len(time_array) not in (1, len(lon_coord)):
            raise ValueError(
                "Length of time_array must be either that of lat_coord/lon_coord or 1!"
            )
        elif len(time_array) != len(lon_coord):
            if len(time_array) == 1:
                time_array = np.repeat(time_array, len(lon_coord))
            else:
                lon_coord = np.repeat(lon_coord, len(time_array))
                lat_coord = np.repeat(lat_coord, len(time_array))

    coord_object = SkyCoord(
        lon_coord * units.rad,
        lat_coord * units.rad,
        frame=coord_frame,
        equinox=coord_epoch,
        obstime=time_array,
    )

    new_coord = coord_object.transform_to("icrs")
    return new_coord.ra.rad, new_coord.dec.rad


def translate_to_app_from_icrs(
    time_array,
    ra_coord,
    dec_coord,
    telescope_lon,
    telescope_lat,
    telescope_alt,
    pm_ra=None,
    pm_dec=None,
    rad_vel=None,
    parallax=None,
    use_astropy=False,
):
    """
    Translate a set of coordinates in ICRS to topocentric/apparent coordinates.

    This utility uses either astropy or NOVAS to calculate the apparent (i.e.,
    topocentric) coordinates of a source at a given time and location, given
    a set of coordinates expressed in the ICRS frame. These coordinates are most
    typically used for defining the phase center of the array (i.e, calculating
    baseline vectors).

    Parameters
    ----------
    time_array : float or ndarray of float
        Julian dates to calculate coordinate positions for. Can either be a single
        float, or an ndarray of shape is (Ntimes,).
    ra_coord : float or ndarray of float
        ICRS RA of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (with the exception of telescope location parameters).
    dec_coord : float or ndarray of float
        ICRS Dec of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (with the exception of telescope location parameters).
    telescope_lon : float
        ITRF longitude of the phase center of the array, expressed in units of radians.
    telescope_lat : float
        ITRF latitude of the phase center of the array, expressed in units of radians.
    telescope_alt : float
        Altitude (rel to sea-level) of the phase center of the array, expressed in
        units of meters.
    pm_ra : float or ndarray of float
        Proper motion in RA of the source, expressed in units of milliarcsec / year.
        Proper motion values are applied relative to the J2000 (i.e., RA/Dec ICRS
        values should be set to their expected values when the epoch is 2000.0).
        Only used if use_astropy=False, and is optional. Can either be a single float
        or array of shape (Ntimes,), although this must be consistent with other
        parameters (with the exception of telescope location parameters).
    pm_dec : float or ndarray of float
        Proper motion in Dec of the source, expressed in units of milliarcsec / year.
        Proper motion values are applied relative to the J2000 (i.e., RA/Dec ICRS
        values should be set to their expected values when the epoch is 2000.0).
        Only used if use_astropy=False, and is optional. Can either be a single float
        or array of shape (Ntimes,), although this must be consistent with other
        parameters (with the exception of telescope location parameters).
    rad_vel : float or ndarray of float
        Radial velocity of the source, expressed in units of km / sec. Only used if
        use_astropy=False, and is optional. Can either be a single float or array of
        shape (Ntimes,), although this must be consistent with other parameters
        (with the exception of telescope location parameters).
    parallax : float or ndarray of float
        Parallax of the source, expressed in milliarcseconds. Only used if
        use_astropy=False, and is optional. Can either be a single float or array of
        shape (Ntimes,), although this must be consistent with other parameters
        (with the exception of telescope location parameters).
    use_astropy : boolean
        Use astropy in order to calculate the apparent coordinates. Default is false,
        which instead uses NOVAS for deriving coordinate positions.

    Returns
    -------
    app_ra : ndarray of floats
        Apparent right ascension coordinates, in units of radians, of shape (Ntimes,).
    app_dec : ndarray of floats
        Apparent declination coordinates, in units of radians, of shape (Ntimes,).
    """
    if not isinstance(time_array, np.ndarray):
        time_array = np.array([time_array])
    if not isinstance(time_array, np.ndarray):
        ra_coord = np.array([ra_coord])
    if not isinstance(time_array, np.ndarray):
        dec_coord = np.array([ra_coord])

    # Check here to make sure that ra_coord and dec_coord are the same length,
    # either 1 or len(time_array)
    if (len(ra_coord) == 1) and (len(dec_coord) == 1):
        multi_coord = False
    elif (len(ra_coord) == len(time_array)) and (len(dec_coord) == len(time_array)):
        multi_coord = True
    else:
        raise ValueError(
            "ra_coord and dec_coord must be the same length, either 1 (single float) "
            "or of the same length as time_array!"
        )

    # Useful for both astropy and novas methods, the latter of which gives easy
    # access to the IERS data that we want.
    time_obj_array = Time(time_array, format="jd", scale="utc")

    telescope_lon = np.float64(telescope_lon)
    telescope_lat = np.float64(telescope_lat)
    telescope_alt = np.float64(telescope_alt)

    # Check the optional inputs, make sure that they're sensible
    if pm_ra is None:
        pm_ra = np.zeros_like(ra_coord)
    else:
        if not isinstance(pm_ra, np.ndarray):
            pm_ra = np.array([pm_ra])
        if len(pm_ra) != len(ra_coord):
            raise ValueError("pm_ra must be the same length as ra_coord and dec_coord!")

    if pm_dec is None:
        pm_dec = np.zeros_like(ra_coord)
    else:
        if not isinstance(pm_ra, np.ndarray):
            pm_dec = np.array([pm_dec])
        if len(pm_dec) != len(ra_coord):
            raise ValueError(
                "pm_dec must be the same length as ra_coord and dec_coord!"
            )

    if rad_vel is None:
        rad_vel = np.zeros_like(ra_coord)
    else:
        if not isinstance(rad_vel, np.ndarray):
            rad_vel = np.array([rad_vel])
        if len(rad_vel) != len(ra_coord):
            raise ValueError(
                "rad_vel must be the same length as ra_coord and dec_coord!"
            )

    if parallax is None:
        parallax = np.zeros_like(ra_coord)
    else:
        if not isinstance(parallax, np.ndarray):
            parallax = np.array([parallax])
        if len(parallax) != len(ra_coord):
            raise ValueError(
                "parallax must be the same length as ra_coord and dec_coord!"
            )

    if use_astropy:
        # Note that this implementation isn't perfect, but it does appear to agree
        # to better than 1 arcsec w/ what NOVAS produces. Errors are of similar size
        # to polar wobble and diurnal abberation terms.
        site_loc = EarthLocation.from_geodetic(
            telescope_lon * (180.0 / np.pi),
            telescope_lat * (180.0 / np.pi),
            height=telescope_alt,
        )

        # Note that the below is _very_ unhappy being fed pm, parallax,
        # or rad vel info into the AltAz frame. Something to investigate later.
        sou_info = SkyCoord(
            ra=ra_coord * units.rad, dec=dec_coord * units.rad, frame="icrs",
        )

        alt_az_data = sou_info.transform_to(
            AltAz(location=site_loc, obstime=time_obj_array)
        )
        az_vals = alt_az_data.az.value * (np.pi / 180.0)
        el_vals = alt_az_data.alt.value * (np.pi / 180.0)

        # Get coordinates to XYZ rel to ENU
        xyz_vector = np.transpose(
            np.array(
                [
                    np.sin(az_vals) * np.cos(el_vals),
                    np.cos(az_vals) * np.cos(el_vals),
                    np.sin(el_vals),
                ]
            )
        )

        # Re-project to HA/Dec
        xyz_prime = rotate_one_axis(xyz_vector, (np.pi / 2.0) - telescope_lat, 0)

        # Get the L(A)ST so that you can translate HA -> RA
        lst_array = time_obj_array.sidereal_time(
            "apparent", longitude=telescope_lon * units.rad
        ).rad

        # Do some fancy math, note the sign flip due to RA (increasing CCW) being
        # reverse-oriented relative to azimuth (increasing CW).
        app_ra = np.mod(
            (lst_array + np.arctan2(xyz_prime[:, 0], -xyz_prime[:, 1])), 2.0 * np.pi
        )
        app_dec = np.arcsin(xyz_prime[:, 2])
    else:
        # Call is needed to load high-precision ephem data in NOVAS
        jd_start, jd_end, number = eph_manager.ephem_open()

        # Get IERS data
        polar_motion_data = iers.earth_orientation_table.get()

        # Use interpolation to get the right values, set to zero (no corretion)
        # when outside of the nominal range
        pm_x_array = 1000.0 * np.interp(
            time_array - 2400000.5,
            polar_motion_data["MJD"].value,
            polar_motion_data["PM_x"].value,
            left=0.0,
            right=0.0,
        )
        pm_y_array = 1000.0 * np.interp(
            time_array - 2400000.5,
            polar_motion_data["MJD"].value,
            polar_motion_data["PM_y"].value,
            left=0.0,
            right=0.0,
        )

        # Define the obs location, which is needed to calculate diurnal abb term
        # and polar wobble corrections
        site_loc = novas.make_on_surface(
            telescope_lat * (180.0 / np.pi),  # latitude in deg
            telescope_lon * (180.0 / np.pi),  # Longitude in deg
            telescope_alt,  # Height in meters
            0.0,  # Temperature, set to 0 for now (no atm refrac)
            0.0,  # Pressure, set to 0 for now (no atm refrac)
        )

        # NOVAS wants things in terrestial time and UT1
        tt_time_array = time_obj_array.tt.value
        ut1_time_array = time_obj_array.ut1.value

        if np.any(tt_time_array < jd_start) or np.any(tt_time_array > jd_end):
            raise ValueError(
                "No current support for JPL ephems outside of 1700 - 2300 AD. "
                "Check back later (or possibly earlier)..."
            )

        app_ra = np.zeros_like(time_array)
        app_dec = np.zeros_like(time_array)

        for idx in range(len(time_array)):
            if (idx == 0) or multi_coord:
                # Create a catalog entry for the source in question
                sou_info = novas.make_cat_entry(
                    "kartos_fav_source",  # Dummy source name
                    "GKK",  # Catalog ID, fixed for now
                    0,  # Star ID number, fixed for now
                    ra_coord[idx] * (12.0 / np.pi),  # RA in hours
                    dec_coord[idx] * (180.0 / np.pi),  # Dec in deg
                    pm_ra[idx],  # Proper motion RA, disabled for now
                    pm_dec[idx],  # Proper motion Dec, disabled for now
                    parallax[idx],  # Parallax, in units of 1/parsec
                    rad_vel[idx],  # Radial velocity, disabled for now
                )

            # Update polar wobble parameters for a given timestamp
            novas.cel_pole(tt_time_array[idx], 2, pm_x_array[idx], pm_y_array[idx])

            # Calculate topocentric RA/Dec values
            [app_ra[idx], app_dec[idx]] = novas.topo_star(
                tt_time_array[idx],
                (tt_time_array[idx] - ut1_time_array[idx]) * 86400.0,
                sou_info,
                site_loc,
                0,
            )

        app_ra *= np.pi / 12.0
        app_dec *= np.pi / 180.0

    return app_ra, app_dec


def get_lst_for_time(jd_array, latitude, longitude, altitude, use_astropy=False):
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
    use_astropy: boolean
        Use astropy for the calculation of L(A)ST. Default is false, which instead
        triggers the use of NOVAS for the calculations.

    Returns
    -------
    ndarray of float
        LSTs in radians corresponding to the jd_array.

    """
    lst_array = np.zeros_like(jd_array)
    jd, reverse_inds = np.unique(jd_array, return_inverse=True)
    times = Time(
        jd,
        format="jd",
        scale="utc",
        location=(Angle(longitude, unit="deg"), Angle(latitude, unit="deg")),
    )
    if iers.conf.auto_max_age is None:  # pragma: no cover
        delta, status = times.get_delta_ut1_utc(return_status=True)
        if np.any(
            np.isin(status, (iers.TIME_BEFORE_IERS_RANGE, iers.TIME_BEYOND_IERS_RANGE))
        ):
            warnings.warn(
                "time is out of IERS range, setting delta ut1 utc to "
                "extrapolated value"
            )
            times.delta_ut1_utc = delta
    if use_astropy:
        lst_array = times.sidereal_time("apparent").radian[reverse_inds]
    else:
        tt_time_array = times.tt.value
        ut1_time_array = times.ut1.value
        polar_motion_data = iers.earth_orientation_table.get()

        pm_x_array = 1000.0 * np.interp(
            times.mjd,
            polar_motion_data["MJD"].value,
            polar_motion_data["PM_x"].value,
            left=0.0,
            right=0.0,
        )

        pm_y_array = 1000.0 * np.interp(
            times.mjd,
            polar_motion_data["MJD"].value,
            polar_motion_data["PM_y"].value,
            left=0.0,
            right=0.0,
        )

        for idx in range(len(times)):
            novas.cel_pole(tt_time_array[idx], 2, pm_x_array[idx], pm_y_array[idx])
            lst_array[reverse_inds == idx] = novas.sidereal_time(
                ut1_time_array[idx],
                0.0,
                (tt_time_array[idx] - ut1_time_array[idx]) * 86400.0,
            )
        # Convert from hours back to rad
        lst_array *= np.pi / 12.0

    return lst_array


def _adj_list(vecs, tol, n_blocks=None):
    """Identify neighbors of each vec in vecs, to distance tol."""
    n_items = len(vecs)
    max_items = 2 ** 10  # Max array size used is max_items**2. Avoid using > 1 GiB

    if n_blocks is None:
        n_blocks = max(n_items // max_items, 1)

    # We may sort blocks so that some pairs of blocks may be skipped.
    # Reorder vectors by x.

    order = np.argsort(vecs[:, 0])
    blocks = np.array_split(order, n_blocks)
    adj = [{k} for k in range(n_items)]  # Adjacency lists
    for b1 in blocks:
        for b2 in blocks:
            v1, v2 = vecs[b1], vecs[b2]
            # Check for no overlap, with tolerance.
            xmin1 = v1[0, 0] - tol
            xmax1 = v1[-1, 0] + tol
            xmin2 = v2[0, 0] - tol
            xmax2 = v2[-1, 0] + tol
            if max(xmin1, xmin2) > min(xmax1, xmax2):
                continue

            adj_mat = cdist(vecs[b1], vecs[b2]) < tol
            for bi, col in enumerate(adj_mat):
                adj[b1[bi]] = adj[b1[bi]].union(b2[col])
    return [frozenset(g) for g in adj]


def _find_cliques(adj, strict=False):
    n_items = len(adj)

    loc_gps = []
    visited = np.zeros(n_items, dtype=bool)
    for k in range(n_items):
        if visited[k]:
            continue
        a0 = adj[k]
        visited[k] = True
        if all(adj[it].__hash__() == a0.__hash__() for it in a0):
            group = list(a0)
            group.sort()
            visited[list(a0)] = True
            loc_gps.append(group)

    # Require all adjacency lists to be isolated maximal cliques:
    if strict:
        if not all(sorted(st) in loc_gps for st in adj):
            raise ValueError("Non-isolated cliques found in graph.")

    return loc_gps


def find_clusters(location_ids, location_vectors, tol, strict=False):
    """
    Find clusters of vectors (e.g. redundant baselines, times).

    Parameters
    ----------
    location_ids : array_like of int
        ID labels for locations.
    location_vectors : array_like of float
        location vectors, can be multidimensional
    tol : float
        tolerance for clusters
    strict : bool
        Require that all adjacency lists be isolated maximal cliques.
        This ensures that vectors do not fall into multiple clusters.
        Default: False

    Returns
    -------
    list of list of location_ids

    """
    location_vectors = np.asarray(location_vectors)
    location_ids = np.asarray(location_ids)
    if location_vectors.ndim == 1:
        location_vectors = location_vectors[:, np.newaxis]

    adj = _adj_list(location_vectors, tol)  # adj = list of sets

    loc_gps = _find_cliques(adj, strict=strict)
    loc_gps = [np.sort(location_ids[gp]).tolist() for gp in loc_gps]
    return loc_gps


def get_baseline_redundancies(baselines, baseline_vecs, tol=1.0, with_conjugates=False):
    """
    Find redundant baseline groups.

    Parameters
    ----------
    baselines : array_like of int
        Baseline numbers, shape (Nbls,)
    baseline_vecs : array_like of float
        Baseline vectors in meters, shape (Nbls, 3)
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

    baseline_vecs = copy.copy(baseline_vecs)  # Protect the vectors passed in.

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
        baseline_vecs[conjugates] *= -1
        baseline_ind_conj = baselines[conjugates]
        bl_gps, vec_bin_centers, lens = get_baseline_redundancies(
            baselines, baseline_vecs, tol=tol, with_conjugates=False
        )
        return bl_gps, vec_bin_centers, lens, baseline_ind_conj

    try:
        bl_gps = find_clusters(baselines, baseline_vecs, tol, strict=True)
    except ValueError as exc:
        raise ValueError(
            "Some baselines are falling into multiple"
            " redundant groups. Lower the tolerance to resolve ambiguity."
        ) from exc

    n_unique = len(bl_gps)
    vec_bin_centers = np.zeros((n_unique, 3))
    for gi, gp in enumerate(bl_gps):
        inds = [np.where(i == baselines)[0] for i in gp]
        vec_bin_centers[gi] = np.mean(baseline_vecs[inds, :], axis=0)

    lens = np.sqrt(np.sum(vec_bin_centers ** 2, axis=1))
    return bl_gps, vec_bin_centers, lens


def get_antenna_redundancies(
    antenna_numbers, antenna_positions, tol=1.0, include_autos=False
):
    """
    Find redundant baseline groups based on antenna positions.

    Parameters
    ----------
    antenna_numbers : array_like of int
        Antenna numbers, shape (Nants,).
    antenna_positions : array_like of float
        Antenna position vectors in the ENU (topocentric) frame in meters,
        shape (Nants, 3).
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
    gps, vecs, lens, conjs = get_baseline_redundancies(
        bls, bl_vecs, tol=tol, with_conjugates=True
    )
    # Flip the baselines in the groups.
    for gi, gp in enumerate(gps):
        for bi, bl in enumerate(gp):
            if bl in conjs:
                gps[gi][bi] = baseline_index_flip(bl, Nants)

    return gps, vecs, lens


def mean_collapse(
    arr, weights=None, axis=None, return_weights=False, return_weights_square=False
):
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
    return_weights_square: bool
        Whether to return the sum of the square of the weights. Default is False.

    """
    arr = copy.deepcopy(arr)  # avoid changing outside
    if weights is None:
        weights = np.ones_like(arr)
    else:
        weights = copy.deepcopy(weights)
    weights = weights * np.logical_not(np.isinf(arr))
    arr[np.isinf(arr)] = 0
    weight_out = np.sum(weights, axis=axis)
    if return_weights_square:
        weights_square = weights ** 2
        weights_square_out = np.sum(weights_square, axis=axis)
    out = np.sum(weights * arr, axis=axis)
    where = weight_out > 1e-10
    out = np.true_divide(out, weight_out, where=where)
    out = np.where(where, out, np.inf)
    if return_weights and return_weights_square:
        return out, weight_out, weights_square_out
    elif return_weights:
        return out, weight_out
    elif return_weights_square:
        return out, weights_square_out
    else:
        return out


def absmean_collapse(
    arr, weights=None, axis=None, return_weights=False, return_weights_square=False
):
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
    return_weights_square: bool
        whether to return the sum of the squares of the weights. Default is False.

    """
    return mean_collapse(
        np.abs(arr),
        weights=weights,
        axis=axis,
        return_weights=return_weights,
        return_weights_square=return_weights_square,
    )


def quadmean_collapse(
    arr, weights=None, axis=None, return_weights=False, return_weights_square=False
):
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
    return_weights_square: bool
        whether to return the sum of the squares of the weights. Default is False.

    """
    out = mean_collapse(
        np.abs(arr) ** 2,
        weights=weights,
        axis=axis,
        return_weights=return_weights,
        return_weights_square=return_weights_square,
    )
    if return_weights and return_weights_square:
        return np.sqrt(out[0]), out[1], out[2]
    elif return_weights or return_weights_square:
        return np.sqrt(out[0]), out[1]
    else:
        return np.sqrt(out)


def or_collapse(
    arr, weights=None, axis=None, return_weights=False, return_weights_square=False
):
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
    return_weights_square: bool
        NOT USED, but kept for symmetry with other collapsing functions.

    """
    if arr.dtype != np.bool_:
        raise ValueError("Input to or_collapse function must be boolean array")
    out = np.any(arr, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn("Currently weights are not handled when OR-ing boolean arrays.")
    if return_weights:
        return out, np.ones_like(out, dtype=np.float64)
    else:
        return out


def and_collapse(
    arr, weights=None, axis=None, return_weights=False, return_weights_square=False
):
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
    return_weights_square: bool
        NOT USED, but kept for symmetry with other collapsing functions.

    """
    if arr.dtype != np.bool_:
        raise ValueError("Input to and_collapse function must be boolean array")
    out = np.all(arr, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn("Currently weights are not handled when AND-ing boolean arrays.")
    if return_weights:
        return out, np.ones_like(out, dtype=np.float64)
    else:
        return out


def collapse(
    arr, alg, weights=None, axis=None, return_weights=False, return_weights_square=False
):
    """
    Parent function to collapse an array with a given algorithm.

    Parameters
    ----------
    arr : array
        Input array to process.
    alg : str
        Algorithm to use. Must be defined in this function with
        corresponding subfunction above.
    weights: ndarray, optional
        weights for collapse operation (e.g. weighted mean).
        NOTE: Some subfunctions do not use the weights. See corresponding
        doc strings.
    axis : int or tuple, optional
        Axis or axes to collapse. Default is all.
    return_weights : bool
        Whether to return sum of weights.
    return_weights_square: bool
        Whether to return the sum of the squares of the weights. Default is False.

    """
    collapse_dict = {
        "mean": mean_collapse,
        "absmean": absmean_collapse,
        "quadmean": quadmean_collapse,
        "or": or_collapse,
        "and": and_collapse,
    }
    try:
        out = collapse_dict[alg](
            arr,
            weights=weights,
            axis=axis,
            return_weights=return_weights,
            return_weights_square=return_weights_square,
        )
    except KeyError:
        raise ValueError(
            "Collapse algorithm must be one of: "
            + ", ".join(collapse_dict.keys())
            + "."
        )
    return out


def uvcalibrate(
    uvdata,
    uvcal,
    inplace=True,
    prop_flags=True,
    flag_missing=True,
    Dterm_cal=False,
    delay_convention="minus",
    undo=False,
    time_check=True,
    ant_check=True,
):
    """
    Calibrate a UVData object with a UVCal object.

    Parameters
    ----------
    uvdata : UVData object
        UVData object to calibrate.
    uvcal : UVCal object
        UVCal object containing the calibration.
    inplace : bool, optional
        if True edit uvdata in place, else return a calibrated copy
    prop_flags : bool, optional
        if True, propagate calibration flags to data flags
        and doesn't use flagged gains. Otherwise, uses flagged gains and
        does not propagate calibration flags to data flags.
    flag_missing : bool, optional
        Deprecated in favor of ant_check.
        If True, flag baselines in uvdata otherwise don't flag and
        don't calibrate the baseline if a participating antenna or polarization
        is missing in uvcal.
    Dterm_cal : bool, optional
        Calibrate the off-diagonal terms in the Jones matrix if present
        in uvcal. Default is False. Currently not implemented.
    delay_convention : str, optional
        Exponent sign to use in conversion of 'delay' to 'gain' cal_type
        if the input uvcal is not inherently 'gain' cal_type. Default to 'minus'.
    undo : bool, optional
        If True, undo the provided calibration. i.e. apply the calibration with
        flipped gain_convention. Flag propagation rules apply the same.
    time_check : bool
        Option to check that times match between the UVCal and UVData
        objects if UVCal has a single time or time range. Times are always
        checked if UVCal has multiple times.
    ant_check : bool
        Option to check that all antennas with data on the UVData
        object have calibration solutions in the UVCal object. If this option is
        set to False, uvcalibrate will proceed without erroring and data for
        antennas without calibrations will be flagged.

    Returns
    -------
    UVData, optional
        Returns if not inplace

    """
    if not inplace:
        uvdata = uvdata.copy()

    # Check whether the UVData antennas *that have data associated with them*
    # have associated data in the UVCal object
    uvdata_unique_nums = np.unique(np.append(uvdata.ant_1_array, uvdata.ant_2_array))
    uvdata.antenna_names = np.asarray(uvdata.antenna_names)
    uvdata_used_antnames = np.array(
        [
            uvdata.antenna_names[np.where(uvdata.antenna_numbers == antnum)][0]
            for antnum in uvdata_unique_nums
        ]
    )
    uvcal_unique_nums = np.unique(uvcal.ant_array)
    uvcal.antenna_names = np.asarray(uvcal.antenna_names)
    uvcal_used_antnames = np.array(
        [
            uvcal.antenna_names[np.where(uvcal.antenna_numbers == antnum)][0]
            for antnum in uvcal_unique_nums
        ]
    )

    ant_arr_match = uvcal_used_antnames.tolist() == uvdata_used_antnames.tolist()

    if not ant_arr_match:
        # check more carefully
        name_missing = []
        for this_ant_name in uvdata_used_antnames:
            wh_ant_match = np.nonzero(uvcal_used_antnames == this_ant_name)
            if wh_ant_match[0].size == 0:
                name_missing.append(this_ant_name)

        use_ant_nums = False
        if len(name_missing) > 0:
            if len(name_missing) == uvdata_used_antnames.size:
                # all antenna_names with data on UVData are missing on UVCal.
                if not ant_check:
                    warnings.warn(
                        "All antenna names with data on UVData are missing "
                        "on UVCal. Since ant_check is False, calibration will "
                        "proceed but all data will be flagged."
                    )
                else:
                    # this entire clause will be replaced with just raising a
                    # ValueError in version 2.2

                    # old behavior only required that antenna numbers were present,
                    # not names. Check numbers
                    number_missing = []
                    for this_ant_name in uvdata_used_antnames:
                        uvdata_ant_num = uvdata.antenna_numbers[
                            np.where(uvdata.antenna_names == this_ant_name)[0][0]
                        ]
                        if uvdata_ant_num not in uvcal_unique_nums:
                            number_missing.append(this_ant_name)

                    if len(number_missing) == 0:
                        # all have matching numbers on UVCal
                        use_ant_nums = True
                        warnings.warn(
                            "All antenna names with data on UVData are missing "
                            "on UVCal. They do all have matching antenna numbers on "
                            "UVCal. Currently the data will be calibrated using the "
                            "matching antenna number, but that will be deprecated in "
                            "version 2.2 and this will become an error.",
                            DeprecationWarning,
                        )
                    elif len(number_missing) < len(name_missing):
                        # Some have matching numbers on UVCal
                        use_ant_nums = True
                        both_missing = sorted(set(number_missing) & set(name_missing))
                        only_name_missing = sorted(
                            set(name_missing) - set(number_missing)
                        )
                        warnings.warn(
                            f"Antennas {only_name_missing} have data on UVData but "
                            "are missing on UVCal. They do have matching antenna "
                            "numbers on UVCal. Currently the data for these antennas "
                            "will be calibrated using the matching antenna number, "
                            "but that will be deprecated in "
                            "version 2.2 and this will become an error.",
                            DeprecationWarning,
                        )
                        if flag_missing is True:
                            warnings.warn(
                                f"Antennas {both_missing} have data on UVData but "
                                "are missing on UVCal. Currently calibration will "
                                "proceed and since flag_missing is True, the data "
                                "for these antennas will be flagged. This will "
                                "become an error in version 2.2, to continue "
                                "calibration and flag missing antennas in the "
                                "future, set ant_check=False.",
                                DeprecationWarning,
                            )
                        else:
                            warnings.warn(
                                f"Antennas {both_missing} have data on UVData but "
                                "are missing on UVCal. Currently calibration will "
                                "proceed and since flag_missing is False, the data "
                                "for these antennas will not be calibrated or "
                                "flagged. This will become an error in version 2.2, "
                                "to continue calibration and flag missing "
                                "antennas in the future, set ant_check=False.",
                                DeprecationWarning,
                            )
            else:
                # Only some antenna_names with data on UVData are missing on UVCal
                if not ant_check:
                    warnings.warn(
                        f"Antennas {name_missing} have data on UVData but are missing "
                        "on UVCal. Since ant_check is False, calibration will "
                        "proceed and the data for these antennas will be flagged."
                    )
                else:
                    # this entire clause will be replaced with just raising a
                    # ValueError in version 2.2
                    if flag_missing is True:
                        warnings.warn(
                            f"Antennas {name_missing} have data on UVData but "
                            "are missing on UVCal. Currently calibration will "
                            "proceed and since flag_missing is True, the data "
                            "for these antennas will be flagged. This will "
                            "become an error in version 2.2, to continue "
                            "calibration and flag missing antennas in the "
                            "future, set ant_check=False.",
                            DeprecationWarning,
                        )
                    else:
                        warnings.warn(
                            f"Antennas {name_missing} have data on UVData but "
                            "are missing on UVCal. Currently calibration will "
                            "proceed and since flag_missing is False, the data "
                            "for these antennas will not be calibrated or "
                            "flagged. This will become an error in version 2.2, "
                            "to continue calibration and flag missing "
                            "antennas in the future, set ant_check=False.",
                            DeprecationWarning,
                        )

    uvdata_times = np.unique(uvdata.time_array)
    downselect_cal_times = False
    if uvcal.Ntimes > 1:
        if uvcal.Ntimes < uvdata.Ntimes:
            raise ValueError(
                "The uvcal object has more than one time but fewer than the "
                "number of unique times on the uvdata object."
            )
        uvcal_times = np.unique(uvcal.time_array)
        try:
            time_arr_match = np.allclose(
                uvcal_times,
                uvdata_times,
                atol=uvdata._time_array.tols[1],
                rtol=uvdata._time_array.tols[0],
            )
        except ValueError:
            time_arr_match = False

        if not time_arr_match:
            # check more carefully
            uvcal_times_to_keep = []
            for this_time in uvdata_times:
                wh_time_match = np.nonzero(
                    np.isclose(
                        uvcal.time_array - this_time,
                        0,
                        atol=uvdata._time_array.tols[1],
                        rtol=uvdata._time_array.tols[0],
                    )
                )
                if wh_time_match[0].size > 0:
                    uvcal_times_to_keep.append(uvcal.time_array[wh_time_match][0])
                else:
                    warnings.warn(
                        f"Time {this_time} exists on UVData but not on UVCal. "
                        "This will become an error in version 2.2",
                        DeprecationWarning,
                    )
            if len(uvcal_times_to_keep) < uvcal.Ntimes:
                downselect_cal_times = True

    elif uvcal.time_range is None:
        # only one UVCal time, no time_range.
        # This cannot match if UVData.Ntimes > 1.
        # If they are both NTimes = 1, then check if they're close.
        if uvdata.Ntimes > 1 or not np.isclose(
            uvdata_times,
            uvcal.time_array,
            atol=uvdata._time_array.tols[1],
            rtol=uvdata._time_array.tols[0],
        ):
            if not time_check:
                warnings.warn(
                    "Times do not match between UVData and UVCal "
                    "but time_check is False, so calibration "
                    "will be applied anyway."
                )
            else:
                warnings.warn(
                    "Times do not match between UVData and UVCal. "
                    "Set time_check=False to apply calibration anyway. "
                    "This will become an error in version 2.2",
                    DeprecationWarning,
                )
    else:
        # time_array is length 1 and time_range exists: check uvdata_times in time_range
        if (
            np.min(uvdata_times) < uvcal.time_range[0]
            or np.max(uvdata_times) > uvcal.time_range[1]
        ):
            if not time_check:
                warnings.warn(
                    "Times do not match between UVData and UVCal "
                    "but time_check is False, so calibration "
                    "will be applied anyway."
                )
            else:
                warnings.warn(
                    "Times do not match between UVData and UVCal. "
                    "Set time_check=False to apply calibration anyway. "
                    "This will become an error in version 2.2",
                    DeprecationWarning,
                )

    downselect_cal_freq = False
    if uvdata.future_array_shapes:
        uvdata_freq_arr_use = uvdata.freq_array
    else:
        uvdata_freq_arr_use = uvdata.freq_array[0, :]
    try:
        freq_arr_match = np.allclose(
            np.sort(uvcal.freq_array[0, :]),
            np.sort(uvdata_freq_arr_use),
            atol=uvdata._freq_array.tols[1],
            rtol=uvdata._freq_array.tols[0],
        )
    except ValueError:
        freq_arr_match = False

    if freq_arr_match is False:
        # check more carefully
        uvcal_freqs_to_keep = []
        for this_freq in uvdata_freq_arr_use:
            wh_freq_match = np.nonzero(
                np.isclose(
                    uvcal.freq_array - this_freq,
                    0,
                    atol=uvdata._freq_array.tols[1],
                    rtol=uvdata._freq_array.tols[0],
                )
            )
            if wh_freq_match[0].size > 0:
                uvcal_freqs_to_keep.append(uvcal.freq_array[wh_freq_match][0])
            else:
                warnings.warn(
                    f"Frequency {this_freq} exists on UVData but not on UVCal. "
                    "This will become an error in version 2.2",
                    DeprecationWarning,
                )
        if len(uvcal_freqs_to_keep) < uvcal.Nfreqs:
            downselect_cal_freq = True

    uvdata_pol_strs = polnum2str(
        uvdata.polarization_array, x_orientation=uvdata.x_orientation
    )
    uvcal_pol_strs = jnum2str(uvcal.jones_array, x_orientation=uvcal.x_orientation)
    uvdata_feed_pols = {
        feed for pol in uvdata_pol_strs for feed in POL_TO_FEED_DICT[pol]
    }
    for feed in uvdata_feed_pols:
        # get diagonal jones str
        jones_str = parse_jpolstr(feed, x_orientation=uvcal.x_orientation)
        if jones_str not in uvcal_pol_strs:
            warnings.warn(
                f"Feed polarization {feed} exists on UVData but not on UVCal. "
                "This will become an error in version 2.2",
                DeprecationWarning,
            )

    # downselect UVCal times, frequencies
    if downselect_cal_freq or downselect_cal_times:
        if not downselect_cal_times:
            uvcal_times_to_keep = None
        elif not downselect_cal_freq:
            uvcal_freqs_to_keep = None

        # handle backwards compatibility: prevent downselecting to nothing
        # or to shapes that don't match
        if downselect_cal_times and len(uvcal_times_to_keep) < uvdata.Ntimes:
            downselect_cal_times = False
            uvcal_times_to_keep = None
        if downselect_cal_freq and len(uvcal_freqs_to_keep) < uvdata.Nfreqs:
            downselect_cal_freq = False
            uvcal_freqs_to_keep = None

    if downselect_cal_freq or downselect_cal_times:
        uvcal_use = uvcal.select(
            times=uvcal_times_to_keep, frequencies=uvcal_freqs_to_keep, inplace=False
        )

        new_uvcal = True
    else:
        uvcal_use = uvcal
        new_uvcal = False

    # input checks
    if uvcal_use.cal_type == "delay":
        if not new_uvcal:
            # make a copy to convert to gain
            uvcal_use = uvcal_use.copy()
            new_uvcal = True
        uvcal_use.convert_to_gain(delay_convention=delay_convention)

    # D-term calibration
    if Dterm_cal:
        # check for D-terms
        if -7 not in uvcal_use.jones_array and -8 not in uvcal_use.jones_array:
            raise ValueError(
                "Cannot apply D-term calibration without -7 or -8"
                "Jones polarization in uvcal object."
            )
        raise NotImplementedError("D-term calibration is not yet implemented.")

    # No D-term calibration
    else:
        # key is number, value is name
        uvdata_ant_dict = dict(zip(uvdata.antenna_numbers, uvdata.antenna_names))
        # opposite: key is name, value is number
        uvcal_ant_dict = dict(zip(uvcal.antenna_names, uvcal.antenna_numbers))

        # iterate over keys
        for key in uvdata.get_antpairpols():
            # get indices for this key
            blt_inds = uvdata.antpair2ind(key)
            pol_ind = np.argmin(
                np.abs(
                    uvdata.polarization_array - polstr2num(key[2], uvdata.x_orientation)
                )
            )

            # try to get gains for each antenna
            ant1_num = key[0]
            ant2_num = key[1]

            feed1, feed2 = POL_TO_FEED_DICT[key[2]]
            try:
                uvcal_ant1_num = uvcal_ant_dict[uvdata_ant_dict[ant1_num]]
            except KeyError:
                if use_ant_nums:
                    # backwards compatibility: use antenna numbers instead
                    # this will be removed in version 2.2
                    uvcal_ant1_num = ant1_num
                else:
                    uvcal_ant1_num = None
            try:
                uvcal_ant2_num = uvcal_ant_dict[uvdata_ant_dict[ant2_num]]
            except KeyError:
                if use_ant_nums:
                    # backwards compatibility: use antenna numbers instead
                    # this will be removed in version 2.2
                    uvcal_ant2_num = ant2_num
                else:
                    uvcal_ant2_num = None

            uvcal_key1 = (uvcal_ant1_num, feed1)
            uvcal_key2 = (uvcal_ant2_num, feed2)

            if uvcal_ant1_num is None or uvcal_ant2_num is None:
                if uvdata.future_array_shapes:
                    uvdata.flag_array[blt_inds, :, pol_ind] = True
                else:
                    uvdata.flag_array[blt_inds, 0, :, pol_ind] = True
                continue
            elif not uvcal_use._has_key(*uvcal_key1) or not uvcal_use._has_key(
                *uvcal_key2
            ):
                if flag_missing:
                    if uvdata.future_array_shapes:
                        uvdata.flag_array[blt_inds, :, pol_ind] = True
                    else:
                        uvdata.flag_array[blt_inds, 0, :, pol_ind] = True
                continue
            gain = (
                uvcal_use.get_gains(uvcal_key1)
                * np.conj(uvcal_use.get_gains(uvcal_key2))
            ).T  # tranpose to match uvdata shape
            flag = (uvcal_use.get_flags(uvcal_key1) | uvcal_use.get_flags(uvcal_key2)).T

            # propagate flags
            if prop_flags:
                mask = np.isclose(gain, 0.0) | flag
                gain[mask] = 1.0
                if uvdata.future_array_shapes:
                    uvdata.flag_array[blt_inds, :, pol_ind] += mask
                else:
                    uvdata.flag_array[blt_inds, 0, :, pol_ind] += mask

            # apply to data
            mult_gains = uvcal_use.gain_convention == "multiply"
            if undo:
                mult_gains = not mult_gains
            if uvdata.future_array_shapes:
                if mult_gains:
                    uvdata.data_array[blt_inds, :, pol_ind] *= gain
                else:
                    uvdata.data_array[blt_inds, :, pol_ind] /= gain
            else:
                if mult_gains:
                    uvdata.data_array[blt_inds, 0, :, pol_ind] *= gain
                else:
                    uvdata.data_array[blt_inds, 0, :, pol_ind] /= gain

    # update attributes
    uvdata.history += "\nCalibrated with pyuvdata.utils.uvcalibrate."
    if undo:
        uvdata.vis_units = "UNCALIB"
    else:
        if uvcal_use.gain_scale is not None:
            uvdata.vis_units = uvcal_use.gain_scale

    if not inplace:
        return uvdata


def apply_uvflag(
    uvd, uvf, inplace=True, unflag_first=False, flag_missing=True, force_pol=True
):
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
    if uvf.mode != "flag":
        raise ValueError("UVFlag must be flag mode")

    if not inplace:
        uvd = uvd.copy()

    # make a deepcopy by default b/c it is generally edited inplace downstream
    uvf = uvf.copy()

    # convert to baseline type
    if uvf.type != "baseline":
        # edits inplace
        uvf.to_baseline(uvd, force_pol=force_pol)

    else:
        # make sure polarizations match or force_pol
        uvd_pols, uvf_pols = (
            uvd.polarization_array.tolist(),
            uvf.polarization_array.tolist(),
        )
        if set(uvd_pols) != set(uvf_pols):
            if uvf.Npols == 1 and force_pol:
                # if uvf is 1pol we can make them match: also edits inplace
                uvf.polarization_array = uvd.polarization_array
                uvf.Npols = len(uvf.polarization_array)
                uvf_pols = uvf.polarization_array.tolist()

            else:
                raise ValueError("Input uvf and uvd polarizations do not match")

        # make sure polarization ordering is correct: also edits inplace
        uvf.polarization_array = uvf.polarization_array[
            [uvd_pols.index(pol) for pol in uvf_pols]
        ]

    # check time and freq shapes match: if Ntimes or Nfreqs is 1, allow
    # implicit broadcasting
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

    # iterate over antpairs and apply flags: TODO need to be able to handle
    # conjugated antpairs
    uvf_antpairs = uvf.get_antpairs()
    for ap in uvd.get_antpairs():
        uvd_ap_inds = uvd.antpair2ind(ap)
        if ap not in uvf_antpairs:
            if flag_missing:
                uvd.flag_array[uvd_ap_inds] = True
            continue
        uvf_ap_inds = uvf.antpair2ind(*ap)
        # addition of boolean is OR
        if uvd.future_array_shapes:
            uvd.flag_array[uvd_ap_inds] += uvf.flag_array[uvf_ap_inds, 0, :, :]
        else:
            uvd.flag_array[uvd_ap_inds] += uvf.flag_array[uvf_ap_inds]

    uvd.history += "\nFlagged with pyuvdata.utils.apply_uvflags."

    if not inplace:
        return uvd


def parse_ants(uv, ant_str, print_toggle=False, x_orientation=None):
    """
    Get antpair and polarization from parsing an aipy-style ant string.

    Used to support the the select function.
    Generates two lists of antenna pair tuples and polarization indices based
    on parsing of the string ant_str.  If no valid polarizations (pseudo-Stokes
    params, or combinations of [lr] or [xy]) or antenna numbers are found in
    ant_str, ant_pairs_nums and polarizations are returned as None.

    Parameters
    ----------
    uv : UVBase Object
        A UVBased object that supports the following functions and parameters:
        - get_ants
        - get_antpairs
        - get_pols
        These are used to construct the baseline ant_pair_nums
        and polarizations returned.
    ant_str : str
        String containing antenna information to parse. Can be 'all',
        'auto', 'cross', or combinations of antenna numbers and polarization
        indicators 'l' and 'r' or 'x' and 'y'.  Minus signs can also be used
        in front of an antenna number or baseline to exclude it from being
        output in ant_pairs_nums. If ant_str has a minus sign as the first
        character, 'all,' will be appended to the beginning of the string.
        See the tutorial for examples of valid strings and their behavior.
    print_toggle : bool
        Boolean for printing parsed baselines for a visual user check.
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to allow for
        converting from E/N strings. If input uv object has an `x_orientation`
        parameter and the input to this function is `None`, the value from the
        object will be used. Any input given to this function will override the
        value on the uv object. See corresonding parameter on UVData
        for more details.

    Returns
    -------
    ant_pairs_nums : list of tuples of int or None
        List of tuples containing the parsed pairs of antenna numbers, or
        None if ant_str is 'all' or a pseudo-Stokes polarizations.
    polarizations : list of int or None
        List of desired polarizations or None if ant_str does not contain a
        polarization specification.

    """
    required_attrs = ["get_ants", "get_antpairs", "get_pols"]
    if not all(hasattr(uv, attr) for attr in required_attrs):
        raise ValueError(
            "UVBased objects must have all the following attributes in order "
            f"to call 'parse_ants': {required_attrs}."
        )

    if x_orientation is None and (
        hasattr(uv, "x_orientation") and uv.x_orientation is not None
    ):
        x_orientation = uv.x_orientation

    ant_re = r"(\(((-?\d+[lrxy]?,?)+)\)|-?\d+[lrxy]?)"
    bl_re = "(^(%s_%s|%s),?)" % (ant_re, ant_re, ant_re)
    str_pos = 0
    ant_pairs_nums = []
    polarizations = []
    ants_data = uv.get_ants()
    ant_pairs_data = uv.get_antpairs()
    pols_data = uv.get_pols()
    warned_ants = []
    warned_pols = []

    if ant_str.startswith("-"):
        ant_str = "all," + ant_str

    while str_pos < len(ant_str):
        m = re.search(bl_re, ant_str[str_pos:])
        if m is None:
            if ant_str[str_pos:].upper().startswith("ALL"):
                if len(ant_str[str_pos:].split(",")) > 1:
                    ant_pairs_nums = uv.get_antpairs()
            elif ant_str[str_pos:].upper().startswith("AUTO"):
                for pair in ant_pairs_data:
                    if pair[0] == pair[1] and pair not in ant_pairs_nums:
                        ant_pairs_nums.append(pair)
            elif ant_str[str_pos:].upper().startswith("CROSS"):
                for pair in ant_pairs_data:
                    if not (pair[0] == pair[1] or pair in ant_pairs_nums):
                        ant_pairs_nums.append(pair)
            elif ant_str[str_pos:].upper().startswith("PI"):
                polarizations.append(polstr2num("pI"))
            elif ant_str[str_pos:].upper().startswith("PQ"):
                polarizations.append(polstr2num("pQ"))
            elif ant_str[str_pos:].upper().startswith("PU"):
                polarizations.append(polstr2num("pU"))
            elif ant_str[str_pos:].upper().startswith("PV"):
                polarizations.append(polstr2num("pV"))
            else:
                raise ValueError("Unparsible argument {s}".format(s=ant_str))

            comma_cnt = ant_str[str_pos:].find(",")
            if comma_cnt >= 0:
                str_pos += comma_cnt + 1
            else:
                str_pos = len(ant_str)
        else:
            m = m.groups()
            str_pos += len(m[0])
            if m[2] is None:
                ant_i_list = [m[8]]
                ant_j_list = list(uv.get_ants())
            else:
                if m[3] is None:
                    ant_i_list = [m[2]]
                else:
                    ant_i_list = m[3].split(",")

                if m[6] is None:
                    ant_j_list = [m[5]]
                else:
                    ant_j_list = m[6].split(",")

            for ant_i in ant_i_list:
                include_i = True
                if type(ant_i) == str and ant_i.startswith("-"):
                    ant_i = ant_i[1:]  # nibble the - off the string
                    include_i = False

                for ant_j in ant_j_list:
                    include_j = True
                    if type(ant_j) == str and ant_j.startswith("-"):
                        ant_j = ant_j[1:]
                        include_j = False

                    pols = None
                    ant_i, ant_j = str(ant_i), str(ant_j)
                    if not ant_i.isdigit():
                        ai = re.search(r"(\d+)([x,y,l,r])", ant_i).groups()

                    if not ant_j.isdigit():
                        aj = re.search(r"(\d+)([x,y,l,r])", ant_j).groups()

                    if ant_i.isdigit() and ant_j.isdigit():
                        ai = [ant_i, ""]
                        aj = [ant_j, ""]
                    elif ant_i.isdigit() and not ant_j.isdigit():
                        if "x" in ant_j or "y" in ant_j:
                            pols = ["x" + aj[1], "y" + aj[1]]
                        else:
                            pols = ["l" + aj[1], "r" + aj[1]]
                        ai = [ant_i, ""]
                    elif not ant_i.isdigit() and ant_j.isdigit():
                        if "x" in ant_i or "y" in ant_i:
                            pols = [ai[1] + "x", ai[1] + "y"]
                        else:
                            pols = [ai[1] + "l", ai[1] + "r"]
                        aj = [ant_j, ""]
                    elif not ant_i.isdigit() and not ant_j.isdigit():
                        pols = [ai[1] + aj[1]]

                    ant_tuple = (abs(int(ai[0])), abs(int(aj[0])))

                    # Order tuple according to order in object
                    if ant_tuple in ant_pairs_data:
                        pass
                    elif ant_tuple[::-1] in ant_pairs_data:
                        ant_tuple = ant_tuple[::-1]
                    else:
                        if not (
                            ant_tuple[0] in ants_data or ant_tuple[0] in warned_ants
                        ):
                            warned_ants.append(ant_tuple[0])
                        if not (
                            ant_tuple[1] in ants_data or ant_tuple[1] in warned_ants
                        ):
                            warned_ants.append(ant_tuple[1])
                        if pols is not None:
                            for pol in pols:
                                if not (pol.lower() in pols_data or pol in warned_pols):
                                    warned_pols.append(pol)
                        continue

                    if include_i and include_j:
                        if ant_tuple not in ant_pairs_nums:
                            ant_pairs_nums.append(ant_tuple)
                        if pols is not None:
                            for pol in pols:
                                if (
                                    pol.lower() in pols_data
                                    and polstr2num(pol, x_orientation=x_orientation)
                                    not in polarizations
                                ):
                                    polarizations.append(
                                        polstr2num(pol, x_orientation=x_orientation)
                                    )
                                elif not (
                                    pol.lower() in pols_data or pol in warned_pols
                                ):
                                    warned_pols.append(pol)
                    else:
                        if pols is not None:
                            for pol in pols:
                                if pol.lower() in pols_data:
                                    if uv.Npols == 1 and [pol.lower()] == pols_data:
                                        ant_pairs_nums.remove(ant_tuple)
                                    if (
                                        polstr2num(pol, x_orientation=x_orientation)
                                        in polarizations
                                    ):
                                        polarizations.remove(
                                            polstr2num(
                                                pol, x_orientation=x_orientation,
                                            )
                                        )
                                elif not (
                                    pol.lower() in pols_data or pol in warned_pols
                                ):
                                    warned_pols.append(pol)
                        elif ant_tuple in ant_pairs_nums:
                            ant_pairs_nums.remove(ant_tuple)

    if ant_str.upper() == "ALL":
        ant_pairs_nums = None
    elif len(ant_pairs_nums) == 0:
        if not ant_str.upper() in ["AUTO", "CROSS"]:
            ant_pairs_nums = None

    if len(polarizations) == 0:
        polarizations = None
    else:
        polarizations.sort(reverse=True)

    if print_toggle:
        print("\nParsed antenna pairs:")
        if ant_pairs_nums is not None:
            for pair in ant_pairs_nums:
                print(pair)

        print("\nParsed polarizations:")
        if polarizations is not None:
            for pol in polarizations:
                print(polnum2str(pol, x_orientation=x_orientation))

    if len(warned_ants) > 0:
        warnings.warn(
            "Warning: Antenna number {a} passed, but not present "
            "in the ant_1_array or ant_2_array".format(
                a=(",").join(map(str, warned_ants))
            )
        )

    if len(warned_pols) > 0:
        warnings.warn(
            "Warning: Polarization {p} is not present in "
            "the polarization_array".format(p=(",").join(warned_pols).upper())
        )

    return ant_pairs_nums, polarizations
