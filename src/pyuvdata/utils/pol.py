# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with polarizations."""

import warnings
from collections.abc import Iterable, Iterable as IterableType
from copy import deepcopy
from functools import lru_cache, wraps

import numpy as np
import numpy.typing as npt

from . import tools

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

    if isinstance(num, int | np.int32 | np.int64):
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

    if isinstance(jnum, int | np.int32 | np.int64):
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
    elif isinstance(pol, int | np.int32 | np.int64):
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


def convert_feeds_to_pols(
    feed_array: npt.NDArray[str],
    include_cross_pols: bool = True,
    x_orientation: str | None = None,
    return_feed_pol_order: bool = False,
):
    """
    Get the polarizations given a feed array.

    Parameters
    ----------
    feed_array : ndarray of str
        Array of feed orientations. Options are: n/e or x/y or r/l.
    include_cross_pols : bool
        Option to include the cross polarizations (e.g. xy and yx or en and ne).
        Defaults to True if more than one feed, set to False for only one feed.
    x_orientation : str, optional
        Orientation of the x-axis. Options are 'east', 'north', 'e', 'n', 'ew', 'ns'.
    return_feed_pol_order : bool
        Option to return a list of tuples giving the ordering of the feeds for
        each pol. Default False.

    Returns
    -------
    polarization_array : ndarray of int
        Polarization integer array.
    feed_pol_order : list of tuples of int, optional
        List of feed index tuples for each pol.
    """
    n_feeds = np.asarray(feed_array).size

    if n_feeds < 1 or n_feeds > 2:
        raise ValueError(
            f"feed_array contains {n_feeds} feeds. Only 1 or 2 feeds is supported."
        )

    feed_pol_order = [(0, 0)]
    if n_feeds > 1:
        feed_pol_order.append((1, 1))
    else:
        include_cross_pols = False

    if include_cross_pols:
        feed_pol_order.extend([(0, 1), (1, 0)])

    pol_strings = []
    for pair in feed_pol_order:
        pol_strings.append(feed_array[pair[0]] + feed_array[pair[1]])
    polarization_array = np.array(
        [polstr2num(ps.upper(), x_orientation=x_orientation) for ps in pol_strings]
    )
    if return_feed_pol_order:
        return polarization_array, feed_pol_order
    else:
        return polarization_array


def _select_pol_helper(
    polarizations,
    obj_pol_array,
    obj_x_orientation=None,
    flex_pol=False,
    invert=False,
    strict=False,
    is_jones=False,
    warn_spacing=False,
):
    """
    Get polarization indices in a select.

    Parameters
    ----------
    polarizations : array_like of str or int
        The polarizations to keep in the object, each value passed here should exist in
        obj_pol_array (unless `invert=True`).
    obj_pol_array : array_like of int
        The polarizations present in the object.
    obj_x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is labelled as the x
        polarization ("east" or "north") to allow for converting from E/N strings.
    flex_pol : bool
        Demark whether or not the object is a flex-pol object, which by construct has
        no polarization axis to select on (it's folded onto the spw-axis instead).
        Default is False.
    invert : bool
        Normally indices matching given criteria are what are included in the
        subsequent list. However, if set to True, these indices are excluded
        instead. Default is False.
    strict : bool or None
        Normally, select will warn when an element of the selection criteria does not
        match any element for the parameter, as long as the selection criteria results
        in *at least one* element being selected. However, if set to True, an error is
        thrown if any selection criteria does not match what is given for the object
        parameters element. If set to None, then neither errors nor warnings are raised,
        unless no records are selected. Default is False.
    is_jones : bool
        Normally this function handles polarizations, but if set to True, Jones terms
        can be input instead. Default is False, it is recommended rather than using this
        setting directly one should use the function _select_jones_helper instead.
    warn_spacing : bool
        Whether or not to warn about polarization spacing. Default is False.

    Returns
    -------
    pol_inds : list of int
        Indices of polarization to keep on the object.
    selections : list of str
        list of selections done.

    """
    pol_inds = None
    selections = []

    str_eval = jstr2num if is_jones else polstr2num
    select_name = "jones polarization terms" if is_jones else "polarizations"
    term_name = "Jones term" if is_jones else "Polarization"
    arr_name = "jones_array" if is_jones else "polarization_array"
    plr_name = "jones polarization terms" if is_jones else "polarization values"

    if flex_pol or (polarizations is None):
        return pol_inds, selections

    polarizations = np.asarray(polarizations).flatten()
    selections.append(select_name)

    pol_nums = []
    for p_name in polarizations:
        if isinstance(p_name, str):
            p_num = str_eval(p_name, x_orientation=obj_x_orientation)
        else:
            p_num = p_name
        if p_num not in obj_pol_array:
            err_msg = f"{term_name} {p_name} is not present in the {arr_name}"
            tools._strict_raise(err_msg, strict=strict)
        else:
            pol_nums.append(p_num)

    pol_inds = np.nonzero(np.isin(obj_pol_array, pol_nums, invert=invert))[0]
    if len(pol_inds) == 0:
        raise ValueError(f"No data matching this {term_name.lower()} selection exists.")

    if (
        warn_spacing
        and (len(pol_inds) > 2)
        and not (tools._test_array_constant_spacing(pol_inds))
    ):
        warnings.warn(
            f"Selected {plr_name} are not evenly spaced. This will make it "
            "impossible to write this data out to some file types."
        )

    return pol_inds.tolist(), selections


def _select_jones_helper(
    jones,
    obj_jones_array,
    obj_x_orientation=None,
    flex_jones=False,
    invert=False,
    strict=False,
    warn_spacing=False,
):
    """
    Get Jones indices in a select.

    Parameters
    ----------
    jones : array_like of str or int
        The Jones terms to keep in the object, each value passed here should exist in
        obj_jones_array (unless `invert=True`).
    obj_jones_array : array_like of int
        The Jones terms present in the object.
    obj_x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is labelled as the x
        polarization ("east" or "north") to allow for converting from E/N strings.
    flex_jones : bool
        Demark whether or not the object is a flex-Jones object, which by construct has
        no Jones axis to select on (it's folded onto the spw-axis instead). Default is
        False.
    invert : bool
        Normally indices matching given criteria are what are included in the
        subsequent list. However, if set to True, these indices are excluded
        instead. Default is False.
    strict : bool or None
        Normally, select will warn when an element of the selection criteria does not
        match any element for the parameter, as long as the selection criteria results
        in *at least one* element being selected. However, if set to True, an error is
        thrown if any selection criteria does not match what is given for the object
        parameters element. If set to None, then neither errors nor warnings are raised,
        unless no records are selected. Default is False.
    warn_spacing : bool
        Whether or not to warn about Jones spacing. Default is False.

    Returns
    -------
    jones_inds : list of int
        Indices of Jones terms to keep on the object.
    selections : list of str
        list of selections done.

    """
    return _select_pol_helper(
        polarizations=jones,
        obj_pol_array=obj_jones_array,
        obj_x_orientation=obj_x_orientation,
        flex_pol=flex_jones,
        invert=invert,
        strict=strict,
        is_jones=True,
        warn_spacing=warn_spacing,
    )


def _select_feed_helper(
    feeds,
    obj_feed_array,
    obj_x_orientation=None,
    feed_inds=None,
    invert=False,
    strict=False,
):
    """
    Get Jones indices in a select.

    Parameters
    ----------
    feeds : array_like of str
        The feeds to keep in the object, each value passed here should exist in
        obj_feed_array (unless `invert=True`).
    obj_feed_array : array_like of str
        The Jones terms present in the object.
    obj_x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is labelled as the x
        polarization ("east" or "north") to allow for converting from E/N strings.
    invert : bool
        Normally indices matching given criteria are what are included in the
        subsequent list. However, if set to True, these indices are excluded
        instead. Default is False.
    strict : bool or None
        Normally, select will warn when an element of the selection criteria does not
        match any element for the parameter, as long as the selection criteria results
        in *at least one* element being selected. However, if set to True, an error is
        thrown if any selection criteria does not match what is given for the object
        parameters element. If set to None, then neither errors nor warnings are raised,
        unless no records are selected. Default is False.

    Returns
    -------
    feed_inds : list of int
        Indices of feeds to keep on the object.
    selections : list of str
        list of selections done.

    """
    feed_inds = None
    selections = []
    if feeds is None:
        return None, []

    selections.append("feeds")
    feeds = [f.lower() for f in tools._get_iterable(feeds)]

    x_orient_dict = {}
    if obj_x_orientation is not None:
        for key, value in x_orientation_pol_map(obj_x_orientation).items():
            if key in obj_feed_array and value in feeds:
                x_orient_dict[value] = key

    mask = np.zeros(len(obj_feed_array), dtype=bool)
    for item in feeds:
        if not ((item in obj_feed_array) or (item in x_orient_dict)):
            msg = f"Feed {item} is not present in the feed_array"
            tools._strict_raise(msg, strict=strict)
        mask |= np.isin(obj_feed_array, x_orient_dict.get(item, item))

    feed_inds = tools._where_combine(mask, inds=feed_inds, invert=invert)

    if len(feed_inds) == 0:
        raise ValueError("No data matching this feed selection exists.")

    return feed_inds.tolist(), selections


def _check_pol_spacing(*, polarization_array, strict=True, allow_resort=False):
    """
    Check if polarizations are evenly spaced.

    This is a requirement for writing uvfits and beamfits files.

    Parameters
    ----------
    polarization_array : array-like of int or UVParameter
        Array of polarization codes, shape (Npols,).
    strict : bool
        If set to True, then the function will raise an error if checks are failed.
        If set to False, then a warning is raised instead. If set to None, then
        no errors or warnings are raised.
    allow_resort : bool
        If set to False, polarizations are checked in their present order. If set to
        True, values are sorted prior to evaluating (useful for cases where the
        polarization index can be reindexed). Default is False.
    """
    if not tools._test_array_constant_spacing(
        polarization_array, allow_resort=allow_resort
    ):
        err_msg = (
            "The polarization values are not evenly spaced. This will "
            "make it impossible to write this data out to some file types."
        )
        tools._strict_raise(err_msg=err_msg, strict=strict)


def _check_jones_spacing(*, jones_array, strict=True, allow_resort=False):
    """
    Check if Jones polarization terms are equally spaced.

    This is a requirement for writing calfits files.

    Parameters
    ----------
    jones_array : array-like of int or UVParameter
        Array of Jones codes, shape (Njones,).
    strict : bool
        If set to True, then the function will raise an error if checks are failed.
        If set to False, then a warning is raised instead. If set to None, then
        no errors or warnings are raised.
    allow_resort : bool
        If set to False, jones terms are checked in their present order. If set to
        True, values are sorted prior to evaluating (useful for cases where the
        jones index can be reindexed). Default is False.
    """
    if not tools._test_array_constant_spacing(jones_array, allow_resort=allow_resort):
        err_msg = (
            "The jones values are not evenly spaced. This will "
            "make it impossible to write this data out to calfits."
        )
        tools._strict_raise(err_msg=err_msg, strict=strict)


def get_feeds_from_pols(polarization_array):
    """
    Return a list of expected feeds based on polarization values.

    Translates values in polarization_array or jones_array into a list of feeds
    expected to be present in the underlying telescope.

    Parameters
    ----------
    polarization_array : array_like of int
        Array listing the polarization codes present, based on the UVFITS numbering
        schedule. See utils.POL_NUM2STR_DICT for a mapping between codes and
        polarization types.

    Returns
    -------
    feed_array : list of str
        List of expected feed types given the polarizations present in the data. Will
        be one of "x", "y", "l", "r", and generally of length <= 2.
    """
    # Preserve order of feeds based on pols using dict.fromkeys
    feed_pols = list(
        dict.fromkeys(
            feed
            for pol in polarization_array
            for feed in POL_TO_FEED_DICT[POL_NUM2STR_DICT[pol]]
        )
    )
    return sorted(feed_pols)


def get_x_orientation_from_feeds(feed_array, feed_angle, tols=None):
    """
    Determine x-orientation equivalent value based on feed information.

    This is a helper function meant to provide a way of translating newer parameters
    (feed_array and feed_angle) describing feed orientation with the older
    "x-orientation" parameter.

    Parameters
    ----------
    feed_array : array-like of str or None
        List of feeds for a given telescope, should be one of "x", "y", "l", "r".
        Shape (Nants, Nfeeds) or (Nfeeds,), must match that of feed_angle, dtype str.
    feed_angle : array-like of float
        Orientation of the feed with respect to zenith (or with respect to north if
        pointed at zenith). Units is in rads, x-polarization is nominally pi / 2,
        and y-polarization (and l- and r-polarizations) is nominally 0. Shape (Nants,
        Nfeeds) or (Nfeeds,), must match that of feed_array, dtype float.
    tols : tuple of float
        Tolerances for feed_angle, used with `isclose`.

    Returns
    -------
    x_orientation : str
        One of "east", "north", or None, based on values present in feed_array and
        feed_angle. None denotes that either one (or both) of feed_array and feed_angle
        were None, or that the values were inconsistent with either "north" or "east"
        orientation.
    """
    if feed_array is None or feed_angle is None:
        # If feed info is unset, then return None
        return None

    rtol, atol = (0, 0) if tols is None else tols

    x_mask = np.isin(feed_array, ["x", "X"])

    # Wrap the feed angle so that everything lands between -45 deg and +135 deg. This
    # is done to prevent either 0 or 90 deg to be right on the "boundary" of the wrap.
    feed_angle = np.mod(np.asarray(feed_angle) + (np.pi / 4), np.pi) - (np.pi / 4)

    # Anything that's not 'x' should be oriented straight up (0 deg) for "east"
    # orientation, otherwise at -90 deg for "north".
    if np.allclose(feed_angle, np.where(x_mask, np.pi / 2, 0), rtol=rtol, atol=atol):
        # x is aligned toward the east, y toward the north
        return "east"
    if np.allclose(feed_angle, np.where(x_mask, 0, np.pi / 2), rtol=rtol, atol=atol):
        # x is aligned toward the north, y toward the east
        return "north"

    # No match? Then time to declare defeat.
    return None


def get_feeds_from_x_orientation(
    *,
    x_orientation,
    nants,
    feeds=None,
    feed_array=None,
    polarization_array=None,
    flex_polarization_array=None,
):
    """
    Determine feed angles based on equivalent x-orientation.

    This is a helper function meant to provide a way of translating the older
    "x-orientation" parameter into the newer parameters describing feed orientation
    (feed_array and feed_angle).

    Parameters
    ----------
    x_orientation : str
        String describing the orientation of the x-polarization. Must be one of "east"
        or "north" (or the associated aliases "n", "e", "ns", "ew")
    nants : int
        Number of antennas, used to determine the shape of the output.
    feeds : str or array-like of str
        List of feeds expected for the telescope. Must be one of "x", "y", "l", or "r".
        A single feed type can be provided as a string, otherwise the list should
        contain no more than two elements. If not provided (and polarization_array is
        also not provided), default is ["x", "y"]. Ignored if feed_array is set.
    feed_array : array-like of str or None
        List of feeds, given on a per-antenna basis. Each feed will be listed as one of
        "x", "y", "l", "r". Shape (Nants, Nfeeds), dtype str (or None of x_orientation
        is None).
    polarization_array : array-like of int or None
        Array listing the polarization codes present, based on the UVFITS numbering
        scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
        polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
        feeds present if not supplied, ignored if flex_polarization_array is set
        to anything but None.
    flex_polarization_array : array-like of int or None
        Array listing the polarization codes present per spectral window (used with
        certain "flexible-polarization" objects), based on the UVFITS numbering
        scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
        polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
        feeds present if not supplied.

    Returns
    -------
    Nfeeds : int
        Length of feeds (or None of x_orientation is None).
    feed_array : array-like of str or None
        List of feeds, given on a per-antenna basis. Each feed will be listed as one of
        "x", "y", "l", "r". Shape (Nants, Nfeeds), dtype str (or None of x_orientation
        is None).
    feed_angle : array-like of float
        Orientation of the feed with respect to zenith (or with respect to north if
        pointed at zenith). Units is in rads, x-polarization is nominally pi / 2,
        and y-polarization (and l- and r-polarizations) is nominally 0.
        Shape (Nants, Nfeeds) (or None of x_orientation is None).
    """
    if x_orientation is None:
        # If x_orientation is None, then there isn't anything to determine
        return None, None, None

    if x_orientation.lower() not in XORIENTMAP:
        raise ValueError(
            f"x_orientation not recognized, must be one of {list(XORIENTMAP)}."
        )

    x_orientation = XORIENTMAP[x_orientation.lower()]
    if (
        feeds is None
        and feed_array is None
        and not (flex_polarization_array is None and polarization_array is None)
    ):
        feed_map = {v: k for k, v in x_orientation_pol_map(x_orientation).items()}
        if flex_polarization_array is not None:
            feeds = get_feeds_from_pols(polarization_array=flex_polarization_array)
        elif polarization_array is not None:
            feeds = get_feeds_from_pols(polarization_array=polarization_array)

        # Handle pseudo-stokes feeds and directional baselines
        feeds = [
            feed_map.get(f, f) for f in feeds if f in ["l", "r", "x", "y", "e", "n"]
        ]
        if len(feeds) == 0:
            feeds = None
    elif isinstance(feeds, str):
        feeds = [feeds]

    if feeds is None and feed_array is None:
        warnings.warn(
            "Unknown polarization basis -- assuming linearly polarized (x/y) "
            "feeds for Telescope.feed_array."
        )
        feeds = ["x", "y"]

    if feed_array is None:
        # Check to make sure inputs here are valid
        if not isinstance(feeds, list | tuple | np.ndarray) or len(feeds) not in [1, 2]:
            raise ValueError("feeds must be a list or tuple of length 1 or 2.")
        if not all(item in ["l", "r", "x", "y"] for item in feeds):
            raise ValueError('feeds must contain only "x", "y", "l", and/or "r".')

        Nfeeds = len(feeds)
        feed_array = np.asarray(feeds) if (nants == 0) else np.tile(feeds, (nants, 1))
    else:
        feed_array = np.asarray(feed_array)
        Nfeeds = feed_array.shape[1]

    feed_angle = np.zeros(feed_array.shape, dtype=float)

    x_mask = feed_array == "x"
    if x_orientation == "east":
        feed_angle[x_mask] = np.pi / 2
    if x_orientation == "north":
        feed_angle[~x_mask] = np.pi / 2

    return Nfeeds, feed_array, feed_angle
