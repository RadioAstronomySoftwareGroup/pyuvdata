# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with polarizations."""
import warnings
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache, wraps
from typing import Iterable as IterableType

import numpy as np

__all__ = [
    "POL_STR2NUM_DICT",
    "POL_NUM2STR_DICT",
    "CONJ_POL_DICT",
    "JONES_STR2NUM_DICT",
    "JONES_NUM2STR_DICT",
    "XORIENTMAP",
    "polstr2num",
    "polnum2str",
    "jstr2num",
    "jnum2str",
    "conj_pol",
    "_x_orientation_rep_dict",
    "x_orientation_pol_map",
    "parse_polstr",
    "parse_jpolstr",
]

# fmt: off
# polarization constants
# maps polarization strings to polarization integers
POL_STR2NUM_DICT = {"pI": 1, "pQ": 2, "pU": 3, "pV": 4,
                    "I": 1, "Q": 2, "U": 3, "V": 4,  # support straight stokes names
                    "rr": -1, "ll": -2, "rl": -3, "lr": -4,
                    "xx": -5, "yy": -6, "xy": -7, "yx": -8,
                    "hh": -5, "vv": -6, "hv": -7, "vh": -8}

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

# maps uvdata pols to input feed polarizations. Note that this dict is also used for
# CASA MS writing, so the pseudo-stokes parameters are included here to provide mapping
# to a consistent (if non-physical) set of feeds for the pseudo-stokes visibilities,
# which are nominally supported by the CASA MS format.
POL_TO_FEED_DICT = {"xx": ["x", "x"], "yy": ["y", "y"],
                    "xy": ["x", "y"], "yx": ["y", "x"],
                    "ee": ["e", "e"], "nn": ["n", "n"],
                    "en": ["e", "n"], "ne": ["n", "e"],
                    "rr": ["r", "r"], "ll": ["l", "l"],
                    "rl": ["r", "l"], "lr": ["l", "r"],
                    "pI": ["I", "I"], "pQ": ["Q", "Q"],
                    "pU": ["U", "U"], "pV": ["V", "V"]}

# fmt: on

XORIENTMAP = {
    "east": "east",
    "north": "north",
    "e": "east",
    "n": "north",
    "ew": "east",
    "ns": "north",
}


def _x_orientation_rep_dict(x_orientation):
    """
    Create replacement dict based on x_orientation.

    Deprecated. Use x_orientation_pol_map instead.

    """
    warnings.warn(
        "This function (_x_orientation_rep_dict) is deprecated, use "
        "pyuvdata.utils.pol.x_orientation_pol_map instead.",
        DeprecationWarning,
    )

    return x_orientation_pol_map(x_orientation)


def x_orientation_pol_map(x_orientation: str) -> dict:
    """
    Return map from "x" and "y" pols to "e" and "n" based on x_orientation.

    Parameters
    ----------
    x_orientation : str
        String giving the x_orientation, one of "east" or "north".

    Returns
    -------
    dict
        Dictionary mapping "x" and "y" pols to "e" and "n" based on x_orientation.

    """
    try:
        if XORIENTMAP[x_orientation.lower()] == "east":
            return {"x": "e", "y": "n"}
        elif XORIENTMAP[x_orientation.lower()] == "north":
            return {"x": "n", "y": "e"}
    except KeyError as e:
        raise ValueError("x_orientation not recognized.") from e


def np_cache(function):
    function = lru_cache(function)

    @wraps(function)
    def wrapper(pol, x_orientation=None):
        try:
            return function(pol, x_orientation=x_orientation)
        except TypeError:
            if isinstance(pol, Iterable):
                # Assume the reason that we got a type error is that pol was an array.
                pol = tuple(pol)
            return function(pol, x_orientation=x_orientation)

    # copy lru_cache attributes over too
    wrapper.cache_info = function.cache_info
    wrapper.cache_clear = function.cache_clear

    return wrapper


@np_cache
def polstr2num(pol: str | IterableType[str], *, x_orientation: str | None = None):
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
    dict_use = deepcopy(POL_STR2NUM_DICT)
    if x_orientation is not None:
        try:
            rep_dict = x_orientation_pol_map(x_orientation)
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
            f"Polarization {pol} cannot be converted to a polarization number."
        )
    return out


@np_cache
def polnum2str(num, *, x_orientation=None):
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
    dict_use = deepcopy(POL_NUM2STR_DICT)
    if x_orientation is not None:
        try:
            rep_dict = x_orientation_pol_map(x_orientation)
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
        raise ValueError(f"Polarization {num} cannot be converted to string.")
    return out


@np_cache
def jstr2num(jstr, *, x_orientation=None):
    """
    Convert jones polarization str to number according to calfits memo.

    Parameters
    ----------
    jstr : str or array_like of str
        antenna (jones) polarization string(s) to convert.
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to allow for
        converting from E/N strings. See corresonding parameter on UVData
        for more details.

    Returns
    -------
    int or list of int
        antenna (jones) polarization number(s) corresponding to the input string(s)

    Raises
    ------
    ValueError
        If the jones string cannot be converted to a polarization number.

    Warns
    -----
    UserWarning
        If the x_orientation not recognized.

    """
    dict_use = deepcopy(JONES_STR2NUM_DICT)
    if x_orientation is not None:
        try:
            rep_dict = x_orientation_pol_map(x_orientation)
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
        raise ValueError(f"Jones polarization {jstr} cannot be converted to index.")
    return out


@np_cache
def jnum2str(jnum, *, x_orientation=None):
    """
    Convert jones polarization number to str according to calfits memo.

    Parameters
    ----------
    num : int or array_like of int
        antenna (jones) polarization number(s) to convert to strings
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to convert to
        E/N strings. See corresonding parameter on UVData for more details.

    Returns
    -------
    str or list of str
        antenna (jones) polarization string(s) corresponding to number

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
    dict_use = deepcopy(JONES_NUM2STR_DICT)
    if x_orientation is not None:
        try:
            rep_dict = x_orientation_pol_map(x_orientation)
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
        raise ValueError(f"Jones polarization {jnum} cannot be converted to string.")
    return out


@np_cache
def parse_polstr(polstr, *, x_orientation=None):
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


@np_cache
def parse_jpolstr(jpolstr, *, x_orientation=None):
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


def determine_pol_order(pols, *, order="AIPS"):
    """
    Determine order of input polarization numbers.

    Determines the order by which to sort a given list of polarizations, according to
    the ordering scheme. Two orders are currently supported: "AIPS" and "CASA". The
    main difference between the two is the grouping of same-handed polarizations for
    AIPS (whereas CASA orders the polarizations such that same-handed pols are on the
    ends of the array).

    Parameters
    ----------
    pols : array_like of str or int
        Polarization array (strings or ints).
    order : str
        Polarization ordering scheme, either "CASA" or "AIPS".

    Returns
    -------
    index_array : ndarray of int
        Indices to reorder polarization array.
    """
    if order == "AIPS":
        index_array = np.argsort(np.abs(pols))
    elif order == "CASA":
        casa_order = np.array([1, 2, 3, 4, -1, -3, -4, -2, -5, -7, -8, -6, 0])
        pol_inds = []
        for pol in pols:
            pol_inds.append(np.where(casa_order == pol)[0][0])
        index_array = np.argsort(pol_inds)
    else:
        raise ValueError('order must be either "AIPS" or "CASA".')

    return index_array
