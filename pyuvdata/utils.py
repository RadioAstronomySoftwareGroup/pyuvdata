# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Commonly used utility functions."""
from __future__ import annotations

import copy
import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache, wraps
from typing import Iterable as IterableType

import erfa
import numpy as np
from astropy import units
from astropy.coordinates import Angle, Distance, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers
from scipy.spatial.distance import cdist

from . import _utils

try:
    from lunarsky import MoonLocation
    from lunarsky import SkyCoord as LunarSkyCoord
    from lunarsky import Time as LTime

    hasmoon = True
except ImportError:
    hasmoon = False


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
    "undo_old_uvw_calc",
    "old_uvw_calc",
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

# standard angle tolerance: 1 mas in radians.
RADIAN_TOL = 1 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)
# standard lst time tolerance: 5 ms (75 mas in radians), based on an expected RMS
# accuracy of 1 ms at 7 days out from issuance of Bulletin A (which are issued once a
# week with rapidly determined parameters and forecasted values of DUT1), the exact
# formula for which is t_err = 0.00025 (MJD-<Bulletin A Release Data>)**0.75 (in secs).
LST_RAD_TOL = 2 * np.pi * 5e-3 / (86400.0)

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

_range_dict = {
    "itrs": (6.35e6, 6.39e6, "Earth"),
    "mcmf": (1717100.0, 1757100.0, "Moon"),
}

if hasmoon:
    selenoids = {
        "SPHERE": _utils.Body.Moon_sphere,
        "GSFC": _utils.Body.Moon_gsfc,
        "GRAIL23": _utils.Body.Moon_grail23,
        "CE-1-LAM-GEO": _utils.Body.Moon_ce1lamgeo,
    }


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
        except KeyError:
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
        "GCOUNT",
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


def _test_array_constant(array, tols=None):
    """
    Check if an array contains constant values to some tolerance.

    Uses np.isclose on the min & max of the arrays with the given tolerances.

    Parameters
    ----------
    array : np.ndarray or UVParameter
        UVParameter or array to check for constant values.
    tols : tuple of float, optional
        length 2 tuple giving (rtol, atol) to pass to np.isclose, defaults to (0, 0) if
        passing an array, otherwise defaults to using the tolerance on the UVParameter.

    Returns
    -------
    bool
        True if the array is constant to the given tolerances, False otherwise.
    """
    # Import UVParameter here rather than at the top to avoid circular imports
    from pyuvdata.parameter import UVParameter

    if isinstance(array, UVParameter):
        array_to_test = array.value
        if tols is None:
            tols = array.tols
    else:
        array_to_test = array
        if tols is None:
            tols = (0, 0)
    assert isinstance(tols, tuple), "tols must be a length-2 tuple"
    assert len(tols) == 2, "tols must be a length-2 tuple"

    if array_to_test.size == 1:
        # arrays with 1 element are constant by definition
        return True

    # if min and max are equal don't bother with tolerance checking
    if np.min(array_to_test) == np.max(array_to_test):
        return True

    return np.isclose(
        np.min(array_to_test), np.max(array_to_test), rtol=tols[0], atol=tols[1]
    )


def _test_array_constant_spacing(array, tols=None):
    """
    Check if an array is constantly spaced to some tolerance.

    Calls _test_array_constant on the np.diff of the array.

    Parameters
    ----------
    array : np.ndarray or UVParameter
        UVParameter or array to check for constant spacing.
    tols : tuple of float, optional
        length 2 tuple giving (rtol, atol) to pass to np.isclose, defaults to (0, 0) if
        passing an array, otherwise defaults to using the tolerance on the UVParameter.

    Returns
    -------
    bool
        True if the array spacing is constant to the given tolerances, False otherwise.
    """
    # Import UVParameter here rather than at the top to avoid circular imports
    from pyuvdata.parameter import UVParameter

    if isinstance(array, UVParameter):
        array_to_test = array.value
        if tols is None:
            tols = array.tols
    else:
        array_to_test = array
        if tols is None:
            tols = (0, 0)
    assert isinstance(tols, tuple), "tols must be a length-2 tuple"
    assert len(tols) == 2, "tols must be a length-2 tuple"

    if array_to_test.size <= 2:
        # arrays with 1 or 2 elements are constantly spaced by definition
        return True

    array_diff = np.diff(array_to_test)
    return _test_array_constant(array_diff, tols=tols)


def _check_flex_spw_contiguous(spw_array, flex_spw_id_array):
    """
    Check if the spectral windows are contiguous for flex_spw datasets.

    This checks the flex_spw_id_array to make sure that all channels for each
    spectral window are together in one block, versus being interspersed (e.g.,
    channel #1 and #3 is in spw #1, channels #2 and #4 are in spw #2). In theory,
    UVH5 and UVData objects can handle this, but MIRIAD, MIR, UVFITS, and MS file
    formats cannot, so we just consider it forbidden.

    Parameters
    ----------
    spw_array : array of integers
        Array of spectral window numbers, shape (Nspws,).
    flex_spw_id_array : array of integers
        Array of spectral window numbers per frequency channel, shape (Nfreqs,).

    """
    exp_spw_ids = np.unique(spw_array)
    # This is an internal consistency check to make sure that the indexes match
    # up as expected -- this shouldn't error unless someone is mucking with
    # settings they shouldn't be.
    assert np.all(np.unique(flex_spw_id_array) == exp_spw_ids), (
        "There are some entries in flex_spw_id_array that are not in spw_array. "
        "This is a bug, please report it in an issue."
    )

    n_breaks = np.sum(flex_spw_id_array[1:] != flex_spw_id_array[:-1])
    if (n_breaks + 1) != spw_array.size:
        raise ValueError(
            "Channels from different spectral windows are interspersed with "
            "one another, rather than being grouped together along the "
            "frequency axis. Most file formats do not support such "
            "non-grouping of data."
        )


def _check_freq_spacing(
    freq_array,
    freq_tols,
    channel_width,
    channel_width_tols,
    flex_spw,
    future_array_shapes,
    spw_array,
    flex_spw_id_array,
    raise_errors=True,
):
    """
    Check if frequencies are evenly spaced and separated by their channel width.

    This is a requirement for writing uvfits & miriad files.

    Parameters
    ----------
    freq_array : array of float
        Array of frequencies, shape (1, Nfreqs) or (Nfreqs,) if future_array_shapes=True
    freq_tols : tuple of float
        freq_array tolerances (from uvobj._freq_array.tols).
    channel_width : float or array of float
        Channel widths, either a scalar or an array of shape (Nfreqs,) if flex_spw=True
        and/or future_array_shapes=True.
    channel_width_tols : tuple of float
        channel_width tolerances (from uvobj._channel_width.tols).
    future_array_shapes : bool
        Indicates that parameters have future shapes.
    flex_spw :  bool
        Indicates there are flexible spectral windows.
    spw_array : array of integers or None
        Array of spectral window numbers, shape (Nspws,). Required if flex_spw is True.
    flex_spw_id_array : array of integers or None
        Array of spectral window numbers per frequency channel, shape (Nfreqs,).
        Required if flex_spw is True.
    raise_errors : bool
        Option to raise errors if the various checks do not pass.

    Returns
    -------
    spacing_error : bool
        Flag that channel spacings or channel widths are not equal.
    chanwidth_error : bool
        Flag that channel spacing does not match channel width.

    """
    spacing_error = False
    chanwidth_error = False
    Nfreqs = freq_array.size
    if future_array_shapes:
        freq_spacing = np.diff(freq_array)
        freq_array_use = freq_array
    else:
        freq_spacing = np.diff(freq_array[0])
        freq_array_use = freq_array[0]

    if Nfreqs == 1:
        # Skip all of this if there is only 1 channel
        pass
    elif flex_spw:
        # Check to make sure that the flexible spectral window has indicies set up
        # correctly (grouped together) for this check
        _check_flex_spw_contiguous(spw_array, flex_spw_id_array)
        diff_chanwidth = np.diff(channel_width)
        freq_dir = []
        # We want to grab unique spw IDs, in the order that they appear in the data
        select_mask = np.append((np.diff(flex_spw_id_array) != 0), True)
        for idx in flex_spw_id_array[select_mask]:
            chan_mask = flex_spw_id_array == idx
            diffs = np.diff(freq_array_use[chan_mask])
            if diffs.size > 0:
                freq_dir += [np.sign(np.mean(diffs))] * np.sum(chan_mask)
            else:
                freq_dir += [1.0]

        # Pop off the first entry, since the above arrays are diff'd
        # (and thus one element shorter)
        freq_dir = np.array(freq_dir[1:])
        # Ignore cases where looking at the boundaries of spectral windows
        bypass_check = flex_spw_id_array[1:] != flex_spw_id_array[:-1]
        if not np.all(
            np.logical_or(
                bypass_check,
                np.isclose(diff_chanwidth, 0.0, rtol=freq_tols[0], atol=freq_tols[1]),
            )
        ):
            spacing_error = True
        if not np.all(
            np.logical_or(
                bypass_check,
                np.isclose(
                    freq_spacing,
                    channel_width[1:] * freq_dir,
                    rtol=freq_tols[0],
                    atol=freq_tols[1],
                ),
            )
        ):
            chanwidth_error = True
    else:
        freq_dir = np.sign(np.mean(freq_spacing))
        if not _test_array_constant(freq_spacing, freq_tols):
            spacing_error = True
        if future_array_shapes:
            if not _test_array_constant(channel_width, freq_tols):
                spacing_error = True
            else:
                if not np.isclose(
                    np.mean(freq_spacing),
                    np.mean(channel_width) * freq_dir,
                    rtol=channel_width_tols[0],
                    atol=channel_width_tols[1],
                ):
                    chanwidth_error = True
        else:
            if not np.isclose(
                np.mean(freq_spacing),
                channel_width * freq_dir,
                rtol=channel_width_tols[0],
                atol=channel_width_tols[1],
            ):
                chanwidth_error = True
    if raise_errors and spacing_error:
        raise ValueError(
            "The frequencies are not evenly spaced (probably because of a select "
            "operation) or has differing values of channel widths. Some file formats "
            "(e.g. uvfits, miriad) do not support unevenly spaced frequencies."
        )
    if raise_errors and chanwidth_error:
        raise ValueError(
            "The frequencies are separated by more than their channel width (probably "
            "because of a select operation). Some file formats (e.g. uvfits, miriad) "
            "do not support frequencies that are spaced by more than their channel "
            "width."
        )

    return spacing_error, chanwidth_error


def _sort_freq_helper(
    Nfreqs,
    freq_array,
    Nspws,
    spw_array,
    flex_spw,
    flex_spw_id_array,
    future_array_shapes,
    spw_order,
    channel_order,
    select_spw,
):
    """
    Figure out the frequency sorting order for object based frequency sorting.

    Parameters
    ----------
    Nfreqs :  int
        Number of frequencies, taken directly from the object parameter.
    freq_array :  array_like of float
        Frequency array, taken directly from the object parameter.
    Nfreqs :  int
        Number of spectral windows, taken directly from the object parameter.
    spw_array :  array_like of int
        Spectral window array, taken directly from the object parameter.
    flex_spw :  bool
        Flag indicating whether the object has flexible spectral windows, taken
        directly from the object parameter.
    flex_spw_id_array : array_like of int
        Array of SPW IDs for each channel, taken directly from the object parameter.
    future_array_shapes : bool
        Flag indicating whether the object uses the future array shapes, taken
        directly from the object parameter.
    spw_order : str or array_like of int
        A string describing the desired order of spectral windows along the
        frequecy axis. Allowed strings include `number` (sort on spectral window
        number) and `freq` (sort on median frequency). A '-' can be prepended
        to signify descending order instead of the default ascending order,
        e.g., if you have SPW #1 and 2, and wanted them ordered as [2, 1],
        you would specify `-number`. Alternatively, one can supply an index array
        of length Nspws that specifies how to shuffle the spws (this is not the desired
        final spw order).  Default is to apply no sorting of spectral windows.
    channel_order : str or array_like of int
        A string describing the desired order of frequency channels within a
        spectral window. Allowed strings include `freq`, which will sort channels
        within a spectral window by frequency. A '-' can be optionally prepended
        to signify descending order instead of the default ascending order.
        Alternatively, one can supply an index array of length Nfreqs that
        specifies the new order. Default is to apply no sorting of channels
        within a single spectral window. Note that proving an array_like of ints
        will cause the values given to `spw_order` and `select_spw` to be ignored.
    select_spw : int or array_like of int
        An int or array_like of ints which specifies which spectral windows to
        apply sorting. Note that setting this argument will cause the value
        given to `spw_order` to be ignored.

    Returns
    -------
    index_array : ndarray of int
        Array giving the desired order of the channels to be used for sorting along the
        frequency axis

    Raises
    ------
    UserWarning
        Raised if providing arguments to select_spw and channel_order (the latter
        overrides the former).
    ValueError
        Raised if select_spw contains values not in spw_array, or if channel_order
        is not the same length as freq_array.

    """
    if (spw_order is None) and (channel_order is None):
        warnings.warn(
            "Not specifying either spw_order or channel_order causes "
            "no sorting actions to be applied. Returning object unchanged."
        )
        return

    # Check to see if there are arguments we should be ignoring
    if isinstance(channel_order, (np.ndarray, list, tuple)):
        if select_spw is not None:
            warnings.warn(
                "The select_spw argument is ignored when providing an "
                "array_like of int for channel_order"
            )
        if spw_order is not None:
            warnings.warn(
                "The spw_order argument is ignored when providing an "
                "array_like of int for channel_order"
            )
        channel_order = np.asarray(channel_order)
        if not channel_order.size == Nfreqs or not np.all(
            np.sort(channel_order) == np.arange(Nfreqs)
        ):
            raise ValueError(
                "Index array for channel_order must contain all indicies for "
                "the frequency axis, without duplicates."
            )
        index_array = channel_order
    else:
        index_array = np.arange(Nfreqs)
        # Multipy by 1.0 here to make a cheap copy of the array to manipulate
        temp_freqs = 1.0 * (freq_array if future_array_shapes else freq_array[0, :])
        # Same trick for ints -- add 0 to make a cheap copy
        temp_spws = 0 + (
            flex_spw_id_array if flex_spw else (np.zeros(Nfreqs) + spw_array)
        )

        # Check whether or not we need to sort the channels in individual windows
        sort_spw = {idx: channel_order is not None for idx in spw_array}
        if select_spw is not None:
            if spw_order is not None:
                warnings.warn(
                    "The spw_order argument is ignored when providing an "
                    "argument for select_spw"
                )
            if channel_order is None:
                warnings.warn(
                    "Specifying select_spw without providing channel_order causes "
                    "no sorting actions to be applied. Returning object unchanged."
                )
                return
            if isinstance(select_spw, (np.ndarray, list, tuple)):
                sort_spw = {idx: idx in select_spw for idx in spw_array}
            else:
                sort_spw = {idx: idx == select_spw for idx in spw_array}
        elif spw_order is not None:
            if isinstance(spw_order, (np.ndarray, list, tuple)):
                spw_order = np.asarray(spw_order)
                if not spw_order.size == Nspws or not np.all(
                    np.sort(spw_order) == np.arange(Nspws)
                ):
                    raise ValueError(
                        "Index array for spw_order must contain all indicies for "
                        "the spw_array, without duplicates."
                    )
            elif spw_order not in ["number", "freq", "-number", "-freq", None]:
                raise ValueError(
                    "spw_order can only be one of 'number', '-number', "
                    "'freq', '-freq', None or an index array of length Nspws"
                )
            elif Nspws > 1:
                # Only need to do this step if we actually have multiple spws.

                # If the string starts with a '-', then we will flip the order at
                # the end of the operation
                flip_spws = spw_order[0] == "-"

                if "number" in spw_order:
                    spw_order = np.argsort(spw_array)
                elif "freq" in spw_order:
                    spw_order = np.argsort(
                        [np.median(temp_freqs[temp_spws == idx]) for idx in spw_array]
                    )
                if flip_spws:
                    spw_order = np.flip(spw_order)
            else:
                spw_order = np.arange(Nspws)
            # Now that we know the spw order, we can apply the first sort
            index_array = np.concatenate(
                [index_array[temp_spws == spw] for spw in spw_array[spw_order]]
            )
            temp_freqs = temp_freqs[index_array]
            temp_spws = temp_spws[index_array]
        # Spectral windows are assumed sorted at this point
        if channel_order is not None:
            if channel_order not in ["freq", "-freq"]:
                raise ValueError(
                    "channel_order can only be one of 'freq' or '-freq' or an index "
                    "array of length Nfreqs"
                )
            for idx in spw_array:
                if sort_spw[idx]:
                    select_mask = temp_spws == idx
                    subsort_order = index_array[select_mask]
                    subsort_order = subsort_order[np.argsort(temp_freqs[select_mask])]
                    index_array[select_mask] = (
                        np.flip(subsort_order)
                        if channel_order[0] == "-"
                        else subsort_order
                    )
    if np.all(index_array[1:] > index_array[:-1]):
        # Nothing to do - the data are already sorted!
        return

    return index_array


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
    if Nants_telescope > 2147483648:
        raise ValueError(
            "error Nants={Nants}>2147483648 not supported".format(Nants=Nants_telescope)
        )
    if np.any(np.asarray(baseline) < 0):
        raise ValueError("negative baseline numbers are not supported")
    if np.any(np.asarray(baseline) > 4611686018498691072):
        raise ValueError("baseline numbers > 4611686018498691072 are not supported")

    return_array = isinstance(baseline, (np.ndarray, list, tuple))
    ant1, ant2 = _utils.baseline_to_antnums(
        np.ascontiguousarray(baseline, dtype=np.uint64)
    )
    if return_array:
        return ant1, ant2
    else:
        return ant1.item(0), ant2.item(0)


def antnums_to_baseline(
    ant1, ant2, Nants_telescope, attempt256=False, use_miriad_convention=False
):
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
        many uvfits files. If there are antenna numbers >= 256, the 2048
        standard will be used unless there are antenna numbers >= 2048
        or Nants_telescope > 2048. In that case, the 2147483648 standard
        will be used. Default is False.
    use_miriad_convention : bool
        Option to use the MIRIAD convention where BASELINE id is
        `bl = 256 * ant1 + ant2` if `ant2 < 256`, otherwise
        `bl = 2048 * ant1 + ant2 + 2**16`.
        Note antennas should be 1-indexed (start at 1, not 0)

    Returns
    -------
    int or array of int
        baseline number corresponding to the two antenna numbers.

    """
    if Nants_telescope is not None and Nants_telescope > 2147483648:
        raise ValueError(
            "cannot convert ant1, ant2 to a baseline index "
            "with Nants={Nants}>2147483648.".format(Nants=Nants_telescope)
        )
    if np.any(np.concatenate((np.unique(ant1), np.unique(ant2))) >= 2147483648):
        raise ValueError(
            "cannot convert ant1, ant2 to a baseline index "
            "with antenna numbers greater than 2147483647."
        )
    if np.any(np.concatenate((np.unique(ant1), np.unique(ant2))) < 0):
        raise ValueError(
            "cannot convert ant1, ant2 to a baseline index "
            "with antenna numbers less than zero."
        )

    nants_less2048 = True
    if Nants_telescope is not None and Nants_telescope > 2048:
        nants_less2048 = False

    return_array = isinstance(ant1, (np.ndarray, list, tuple))
    baseline = _utils.antnums_to_baseline(
        np.ascontiguousarray(ant1, dtype=np.uint64),
        np.ascontiguousarray(ant2, dtype=np.uint64),
        attempt256=attempt256,
        nants_less2048=nants_less2048,
        use_miriad_convention=use_miriad_convention,
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
def polstr2num(pol: str | IterableType[str], x_orientation: str | None = None):
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


@np_cache
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


@np_cache
def jstr2num(jstr, x_orientation=None):
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


@np_cache
def jnum2str(jnum, x_orientation=None):
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


@np_cache
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


@np_cache
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


def determine_pol_order(pols, order="AIPS"):
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


def LatLonAlt_from_XYZ(xyz, frame="ITRS", ellipsoid=None, check_acceptability=True):
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
        lla = _utils._lla_from_xyz(xyz, _utils.Body.Earth.value)
    elif frame == "MCMF":
        lla = _utils._lla_from_xyz(xyz, selenoids[ellipsoid].value)
    else:
        raise ValueError(
            f'No spherical to cartesian transform defined for frame "{frame}".'
        )

    if squeeze:
        return lla[0, 0], lla[1, 0], lla[2, 0]
    return lla[0], lla[1], lla[2]


def XYZ_from_LatLonAlt(latitude, longitude, altitude, frame="ITRS", ellipsoid=None):
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
        xyz = _utils._xyz_from_latlonalt(
            latitude, longitude, altitude, _utils.Body.Earth.value
        )
    elif frame == "MCMF":
        if ellipsoid is None:
            ellipsoid = "SPHERE"

        xyz = _utils._xyz_from_latlonalt(
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


def ENU_from_ECEF(xyz, latitude, longitude, altitude, frame="ITRS", ellipsoid=None):
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
    frame : str
        Coordinate frame of xyz.
        Valid options are ITRS (default) or MCMF.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is MCMF.

    Returns
    -------
    ndarray of float
        numpy array, shape (Npts, 3), with local ENU coordinates

    """
    frame = frame.upper()
    if not hasmoon and frame == "MCMF":
        raise ValueError("Need to install `lunarsky` package to work with MCMF frame.")

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
    enu = _utils._ENU_from_ECEF(
        xyz,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
        # we have already forced the frame to conform to our options
        # and if we  don't have moon we have already errored.
        (_utils.Body.Earth.value if frame == "ITRS" else selenoids[ellipsoid].value),
    )
    enu = enu.T

    if squeeze:
        enu = np.squeeze(enu)

    return enu


def ECEF_from_ENU(enu, latitude, longitude, altitude, frame="ITRS", ellipsoid=None):
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
        Altitude of center of ENU coordinates in meters.
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
    frame = frame.upper()

    if frame not in ["ITRS", "MCMF"]:
        raise ValueError(f'No ECEF_from_ENU transform defined for frame "{frame}".')

    if not hasmoon and frame == "MCMF":
        raise ValueError("Need to install `lunarsky` package to work with MCMF frame.")

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
    xyz = _utils._ECEF_from_ENU(
        enu,
        np.ascontiguousarray(latitude, dtype=np.float64),
        np.ascontiguousarray(longitude, dtype=np.float64),
        np.ascontiguousarray(altitude, dtype=np.float64),
        # we have already forced the frame to conform to our options
        # and if we  don't have moon we have already errored.
        (_utils.Body.Earth.value if frame == "ITRS" else selenoids[ellipsoid].value),
    )
    xyz = xyz.T

    if squeeze:
        xyz = np.squeeze(xyz)

    return xyz


def old_uvw_calc(ra, dec, initial_uvw):
    """
    Calculate old uvws from unphased ones in an icrs or gcrs frame.

    This method should not be used and is only retained for testing the
    undo_old_uvw_calc method, which is needed for fixing phases.

    This code expects input uvws or positions relative to the telescope
    location in the same frame that ra/dec are in (e.g. icrs or gcrs) and
    returns phased ones in the same frame.

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

    return _utils._old_uvw_calc(
        np.float64(ra),
        np.float64(dec),
        np.ascontiguousarray(initial_uvw.T, dtype=np.float64),
    ).T


def undo_old_uvw_calc(ra, dec, uvw):
    """
    Undo the old phasing calculation on uvws in an icrs or gcrs frame.

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

    return _utils._undo_old_uvw_calc(
        np.float64(ra), np.float64(dec), np.ascontiguousarray(uvw.T, dtype=np.float64)
    ).T


def polar2_to_cart3(lon_array, lat_array):
    """
    Convert 2D polar coordinates into 3D cartesian coordinates.

    This is a simple routine for converting a set of spherical angular coordinates
    into a 3D cartesian vectors, where the x-direction is set by the position (0, 0).

    Parameters
    ----------
    lon_array : float or ndarray
        Longitude coordinates, which increases in the counter-clockwise direction.
        Units of radians. Can either be a float or ndarray -- if the latter, must have
        the same shape as lat_array.
    lat_array : float or ndarray
        Latitude coordinates, where 0 falls on the equator of the sphere.  Units of
        radians. Can either be a float or ndarray -- if the latter, must have the same
        shape as lat_array.

    Returns
    -------
    xyz_array : ndarray of float
        Cartesian coordinates of the given longitude and latitude on a unit sphere.
        Shape is (3, coord_shape), where coord_shape is the shape of lon_array and
        lat_array if they were provided as type ndarray, otherwise (3,).
    """
    # Check to make sure that we are not playing with mixed types
    if type(lon_array) is not type(lat_array):
        raise ValueError(
            "lon_array and lat_array must either both be floats or ndarrays."
        )
    if isinstance(lon_array, np.ndarray):
        if lon_array.shape != lat_array.shape:
            raise ValueError("lon_array and lat_array must have the same shape.")

    # Once we know that lon_array and lat_array are of the same shape,
    # time to create our 3D set of vectors!
    xyz_array = np.array(
        [
            np.cos(lon_array) * np.cos(lat_array),
            np.sin(lon_array) * np.cos(lat_array),
            np.sin(lat_array),
        ],
        dtype=float,
    )

    return xyz_array


def cart3_to_polar2(xyz_array):
    """
    Convert 3D cartesian coordinates into 2D polar coordinates.

    This is a simple routine for converting a set of 3D cartesian vectors into
    spherical coordinates, where the position (0, 0) lies along the x-direction.

    Parameters
    ----------
    xyz_array : ndarray of float
        Cartesian coordinates, need not be of unit vector length. Shape is
        (3, coord_shape).

    Returns
    -------
    lon_array : ndarray of float
        Longitude coordinates, which increases in the counter-clockwise direction.
        Units of radians, shape is (coord_shape,).
    lat_array : ndarray of float
        Latitude coordinates, where 0 falls on the equator of the sphere.  Units of
        radians, shape is (coord_shape,).
    """
    if not isinstance(xyz_array, np.ndarray):
        raise ValueError("xyz_array must be an ndarray.")
    if xyz_array.ndim == 0:
        raise ValueError("xyz_array must have ndim > 0")
    if xyz_array.shape[0] != 3:
        raise ValueError("xyz_array must be length 3 across the zeroth axis.")

    # The longitude coord is relatively easy to calculate, just take the X and Y
    # components and find the arctac of the pair.
    lon_array = np.mod(np.arctan2(xyz_array[1], xyz_array[0]), 2.0 * np.pi, dtype=float)

    # If we _knew_ that xyz_array was always of length 1, then this call could be a much
    # simpler one to arcsin. But to make this generic, we'll use the length of the XY
    # component along with arctan2.
    lat_array = np.arctan2(
        xyz_array[2], np.sqrt((xyz_array[0:2] ** 2.0).sum(axis=0)), dtype=float
    )

    # Return the two arrays
    return lon_array, lat_array


def _rotate_matmul_wrapper(xyz_array, rot_matrix, n_rot):
    """
    Apply a rotation matrix to a series of vectors.

    This is a simple convenience function which wraps numpy's matmul function for use
    with various vector rotation functions in this module. This code could, in
    principle, be replaced by a cythonized piece of code, although the matmul function
    is _pretty_ well optimized already. This function is not meant to be called by
    users, but is instead used by multiple higher-level utility functions (namely those
    that perform rotations).

    Parameters
    ----------
    xyz_array : ndarray of floats
        Array of vectors to be rotated. When nrot > 1, shape may be (n_rot, 3, n_vec)
        or (1, 3, n_vec), the latter is useful for when performing multiple rotations
        on a fixed set of vectors. If nrot = 1, shape may be (1, 3, n_vec), (3, n_vec),
        or (3,).
    rot_matrix : ndarray of floats
        Series of rotation matricies to be applied to the stack of vectors. Must be
        of shape (n_rot, 3, 3)
    n_rot : int
        Number of individual rotation matricies to be applied.

    Returns
    -------
    rotated_xyz : ndarray of floats
        Array of vectors that have been rotated, of shape (n_rot, 3, n_vectors,).
    """
    # Do a quick check to make sure that things look sensible
    if rot_matrix.shape != (n_rot, 3, 3):
        raise ValueError(
            "rot_matrix must be of shape (n_rot, 3, 3), where n_rot=%i." % n_rot
        )
    if (xyz_array.ndim == 3) and (
        (xyz_array.shape[0] not in [1, n_rot]) or (xyz_array.shape[-2] != 3)
    ):
        raise ValueError("Misshaped xyz_array - expected shape (n_rot, 3, n_vectors).")
    if (xyz_array.ndim < 3) and (xyz_array.shape[0] != 3):
        raise ValueError("Misshaped xyz_array - expected shape (3, n_vectors) or (3,).")
    rotated_xyz = np.matmul(rot_matrix, xyz_array)

    return rotated_xyz


def _rotate_one_axis(xyz_array, rot_amount, rot_axis):
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
        order, e.g. (x, y, z). Shape is (Nrot, 3, Nvectors).
    rot_amount : float or ndarray of float
        Amount (in radians) to rotate the given set of coordinates. Can either be a
        single float (or ndarray of shape (1,)) if rotating all vectors by the same
        amount, otherwise expected to be shape (Nrot,).
    rot_axis : int
        Axis around which the rotation is applied. 0 is the x-axis, 1 is the y-axis,
        and 2 is the z-axis.

    Returns
    -------
    rotated_xyz : ndarray of float
        Set of rotated 3-dimensional vectors, shape (Nrot, 3, Nvector).
    """
    # If rot_amount is None or all zeros, then this is just one big old no-op.
    if (rot_amount is None) or np.all(rot_amount == 0.0):
        if np.ndim(xyz_array) == 1:
            return deepcopy(xyz_array[np.newaxis, :, np.newaxis])
        elif np.ndim(xyz_array) == 2:
            return deepcopy(xyz_array[np.newaxis, :, :])
        else:
            return deepcopy(xyz_array)

    # Check and see how big of a rotation matrix we need
    n_rot = 1 if (not isinstance(rot_amount, np.ndarray)) else (rot_amount.shape[0])
    n_vec = xyz_array.shape[-1]

    # The promotion of values to float64 is to suppress numerical precision issues,
    # since the matrix math can - in limited circumstances - introduce precision errors
    # of order 10x the limiting numerical precision of the float. For a float32/single,
    # thats a part in 1e6 (~arcsec-level errors), but for a float64 it translates to
    # a part in 1e15.
    rot_matrix = np.zeros((3, 3, n_rot), dtype=np.float64)

    # Figure out which pieces of the matrix we need to update
    temp_jdx = (rot_axis + 1) % 3
    temp_idx = (rot_axis + 2) % 3

    # Fill in the rotation matricies accordingly
    rot_matrix[rot_axis, rot_axis] = 1
    rot_matrix[temp_idx, temp_idx] = np.cos(rot_amount, dtype=np.float64)
    rot_matrix[temp_jdx, temp_jdx] = rot_matrix[temp_idx, temp_idx]
    rot_matrix[temp_idx, temp_jdx] = np.sin(rot_amount, dtype=np.float64)
    rot_matrix[temp_jdx, temp_idx] = -rot_matrix[temp_idx, temp_jdx]

    # The rot matrix was shape (3, 3, n_rot) to help speed up filling in the elements
    # of each matrix, but now we want to flip it into its proper shape of (n_rot, 3, 3)
    rot_matrix = np.transpose(rot_matrix, axes=[2, 0, 1])

    if (n_rot == 1) and (n_vec == 1) and (xyz_array.ndim == 3):
        # This is a special case where we allow the rotation axis to "expand" along
        # the 0th axis of the rot_amount arrays. For xyz_array, if n_vectors = 1
        # but n_rot !=1, then it's a lot faster (by about 10x) to "switch it up" and
        # swap the n_vector and  n_rot axes, and then swap them back once everything
        # else is done.
        return np.transpose(
            _rotate_matmul_wrapper(
                np.transpose(xyz_array, axes=[2, 1, 0]), rot_matrix, n_rot
            ),
            axes=[2, 1, 0],
        )
    else:
        return _rotate_matmul_wrapper(xyz_array, rot_matrix, n_rot)


def _rotate_two_axis(xyz_array, rot_amount1, rot_amount2, rot_axis1, rot_axis2):
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
        order, e.g. (x, y, z). Shape is (Nrot, 3, Nvectors).
    rot_amount1 : float or ndarray of float
        Amount (in radians) of rotatation to apply during the first rotation of the
        sequence, to the given set of coordinates. Can either be a single float (or
        ndarray of shape (1,)) if rotating all vectors by the same amount, otherwise
        expected to be shape (Nrot,).
    rot_amount2 : float or ndarray of float
        Amount (in radians) of rotatation to apply during the second rotation of the
        sequence, to the given set of coordinates. Can either be a single float (or
        ndarray of shape (1,)) if rotating all vectors by the same amount, otherwise
        expected to be shape (Nrot,).
    rot_axis1 : int
        Axis around which the first rotation is applied. 0 is the x-axis, 1 is the
        y-axis, and 2 is the z-axis.
    rot_axis2 : int
        Axis around which the second rotation is applied. 0 is the x-axis, 1 is the
        y-axis, and 2 is the z-axis.

    Returns
    -------
    rotated_xyz : ndarray of float
        Set of rotated 3-dimensional vectors, shape (Nrot, 3, Nvector).

    """
    # Capture some special cases upfront, where we can save ourselves a bit of work
    no_rot1 = (rot_amount1 is None) or np.all(rot_amount1 == 0.0)
    no_rot2 = (rot_amount2 is None) or np.all(rot_amount2 == 0.0)
    if no_rot1 and no_rot2:
        # If rot_amount is None, then this is just one big old no-op.
        return deepcopy(xyz_array)
    elif no_rot1:
        # If rot_amount1 is None, then ignore it and just work w/ the 2nd rotation
        return _rotate_one_axis(xyz_array, rot_amount2, rot_axis2)
    elif no_rot2:
        # If rot_amount2 is None, then ignore it and just work w/ the 1st rotation
        return _rotate_one_axis(xyz_array, rot_amount1, rot_axis1)
    elif rot_axis1 == rot_axis2:
        # Capture the case where someone wants to do a sequence of rotations on the same
        # axis. Also known as just rotating a single axis.
        return _rotate_one_axis(xyz_array, rot_amount1 + rot_amount2, rot_axis1)

    # Figure out how many individual rotation matricies we need, accounting for the
    # fact that these can either be floats or ndarrays.
    n_rot = max(
        rot_amount1.shape[0] if isinstance(rot_amount1, np.ndarray) else 1,
        rot_amount2.shape[0] if isinstance(rot_amount2, np.ndarray) else 1,
    )
    n_vec = xyz_array.shape[-1]

    # The promotion of values to float64 is to suppress numerical precision issues,
    # since the matrix math can - in limited circumstances - introduce precision errors
    # of order 10x the limiting numerical precision of the float. For a float32/single,
    # thats a part in 1e6 (~arcsec-level errors), but for a float64 it translates to
    # a part in 1e15.
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
    sin_lo = np.sin(rot_amount2 if lhd_order else rot_amount1, dtype=np.float64)
    cos_lo = np.cos(rot_amount2 if lhd_order else rot_amount1, dtype=np.float64)
    sin_hi = np.sin(rot_amount1 if lhd_order else rot_amount2, dtype=np.float64)
    cos_hi = np.cos(rot_amount1 if lhd_order else rot_amount2, dtype=np.float64)

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

    # The rot matrix was shape (3, 3, n_rot) to help speed up filling in the elements
    # of each matrix, but now we want to flip it into its proper shape of (n_rot, 3, 3)
    rot_matrix = np.transpose(rot_matrix, axes=[2, 0, 1])

    if (n_rot == 1) and (n_vec == 1) and (xyz_array.ndim == 3):
        # This is a special case where we allow the rotation axis to "expand" along
        # the 0th axis of the rot_amount arrays. For xyz_array, if n_vectors = 1
        # but n_rot !=1, then it's a lot faster (by about 10x) to "switch it up" and
        # swap the n_vector and  n_rot axes, and then swap them back once everything
        # else is done.
        return np.transpose(
            _rotate_matmul_wrapper(
                np.transpose(xyz_array, axes=[2, 1, 0]), rot_matrix, n_rot
            ),
            axes=[2, 1, 0],
        )
    else:
        return _rotate_matmul_wrapper(xyz_array, rot_matrix, n_rot)


def calc_uvw(
    app_ra=None,
    app_dec=None,
    frame_pa=None,
    lst_array=None,
    use_ant_pos=True,
    uvw_array=None,
    antenna_positions=None,
    antenna_numbers=None,
    ant_1_array=None,
    ant_2_array=None,
    old_app_ra=None,
    old_app_dec=None,
    old_frame_pa=None,
    telescope_lat=None,
    telescope_lon=None,
    from_enu=False,
    to_enu=False,
):
    """
    Calculate an array of baseline coordinates, in either uvw or ENU.

    This routine is meant as a convenience function for producing baseline coordinates
    based under a few different circumstances:

    1) Calculating ENU coordinates using antenna positions
    2) Calculating uvw coordinates at a given sky position using antenna positions
    3) Converting from ENU coordinates to uvw coordinates
    4) Converting from uvw coordinate to ENU coordinates
    5) Converting from uvw coordinates at one sky position to another sky position

    Different conversion pathways have different parameters that are required.

    Parameters
    ----------
    app_ra : ndarray of float
        Apparent RA of the target phase center, required if calculating baseline
        coordinates in uvw-space (vs ENU-space). Shape is (Nblts,), units are
        radians.
    app_dec : ndarray of float
        Apparent declination of the target phase center, required if calculating
        baseline coordinates in uvw-space (vs ENU-space). Shape is (Nblts,),
        units are radians.
    frame_pa : ndarray of float
        Position angle between the great circle of declination in the apparent frame
        versus that of the reference frame, used for making sure that "North" on
        the derived maps points towards a particular celestial pole (not just the
        topocentric one). Required if not deriving baseline coordinates from antenna
        positions, from_enu=False, and a value for old_frame_pa is given. Shape is
        (Nblts,), units are radians.
    old_app_ra : ndarray of float
        Apparent RA of the previous phase center, required if not deriving baseline
        coordinates from antenna positions and from_enu=False. Shape is (Nblts,),
        units are radians.
    old_app_dec : ndarray of float
        Apparent declination of the previous phase center, required if not deriving
        baseline coordinates from antenna positions and from_enu=False. Shape is
        (Nblts,), units are radians.
    old_frame_pa : ndarray of float
        Frame position angle of the previous phase center, required if not deriving
        baseline coordinates from antenna positions, from_enu=False, and a value
        for frame_pa is supplied. Shape is (Nblts,), units are radians.
    lst_array : ndarray of float
        Local apparent sidereal time, required if deriving baseline coordinates from
        antenna positions, or converting to/from ENU coordinates. Shape is (Nblts,).
    use_ant_pos : bool
        Switch to determine whether to derive uvw values from the antenna positions
        (if set to True), or to use the previously calculated uvw coordinates to derive
        new the new baseline vectors (if set to False). Default is True.
    uvw_array : ndarray of float
        Array of previous baseline coordinates (in either uvw or ENU), required if
        not deriving new coordinates from antenna positions.  Shape is (Nblts, 3).
    antenna_positions : ndarray of float
        List of antenna positions relative to array center in ECEF coordinates,
        required if not providing `uvw_array`. Shape is (Nants, 3).
    antenna_numbers: ndarray of int
        List of antenna numbers, ordered in the same way as `antenna_positions` (e.g.,
        `antenna_numbers[0]` should given the number of antenna that resides at ECEF
        position given by `antenna_positions[0]`). Shape is (Nants,), requred if not
        providing `uvw_array`. Contains all unique entires of the joint set of
        `ant_1_array` and `ant_2_array`.
    ant_1_array : ndarray of int
        Antenna number of the first antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,).
    ant_2_array : ndarray of int
        Antenna number of the second antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,).
    telescope_lat : float
        Latitude of the phase center, units radians, required if deriving baseline
        coordinates from antenna positions, or converting to/from ENU coordinates.
    telescope_lon : float
        Longitude of the phase center, units radians, required if deriving baseline
        coordinates from antenna positions, or converting to/from ENU coordinates.
    from_enu : boolean
        Set to True if uvw_array is expressed in ENU coordinates. Default is False.
    to_enu : boolean
        Set to True if you would like the output expressed in ENU coordinates. Default
        is False.

    Returns
    -------
    new_coords : ndarray of float64
        Set of baseline coordinates, shape (Nblts, 3).
    """
    if to_enu:
        if lst_array is None and not use_ant_pos:
            raise ValueError(
                "Must include lst_array to calculate baselines in ENU coordinates!"
            )
        if telescope_lat is None:
            raise ValueError(
                "Must include telescope_lat to calculate baselines in ENU coordinates!"
            )
    else:
        if ((app_ra is None) or (app_dec is None)) and frame_pa is None:
            raise ValueError(
                "Must include both app_ra and app_dec, or frame_pa to calculate "
                "baselines in uvw coordinates!"
            )

    if use_ant_pos:
        # Assume at this point we are dealing w/ antenna positions
        if antenna_positions is None:
            raise ValueError("Must include antenna_positions if use_ant_pos=True.")
        if (ant_1_array is None) or (ant_2_array is None) or (antenna_numbers is None):
            raise ValueError(
                "Must include ant_1_array, ant_2_array, and antenna_numbers "
                "setting use_ant_pos=True."
            )
        if lst_array is None and not to_enu:
            raise ValueError(
                "Must include lst_array if use_ant_pos=True and not calculating "
                "baselines in ENU coordinates."
            )
        if telescope_lon is None:
            raise ValueError("Must include telescope_lon if use_ant_pos=True.")

        ant_dict = {ant_num: idx for idx, ant_num in enumerate(antenna_numbers)}
        ant_1_index = np.array(
            [ant_dict[ant_num] for ant_num in ant_1_array], dtype=int
        )
        ant_2_index = np.array(
            [ant_dict[ant_num] for ant_num in ant_2_array], dtype=int
        )

        N_ants = antenna_positions.shape[0]
        # Use the app_ra, app_dec, and lst_array arrays to figure out how many unique
        # rotations are actually needed. If the ratio of Nblts to number of unique
        # entries is favorable, we can just rotate the antenna positions and save
        # outselves a bit of work.
        if to_enu:
            # If to_enu, skip all this -- there's only one unique ha + dec combo
            unique_mask = np.zeros(len(ant_1_index), dtype=np.bool_)
            unique_mask[0] = True
        else:
            unique_mask = np.append(
                True,
                (
                    ((lst_array[:-1] - app_ra[:-1]) != (lst_array[1:] - app_ra[1:]))
                    | (app_dec[:-1] != app_dec[1:])
                ),
            )

        # GHA -> Hour Angle as measured at Greenwich (because antenna coords are
        # centered such that x-plane intersects the meridian at longitude 0).
        if to_enu:
            # Unprojected coordinates are given in the ENU convention -- that's
            # equivalent to calculating uvw's based on zenith. We can use that to our
            # advantage and spoof the gha and dec based on telescope lon and lat
            unique_gha = np.zeros(1) - telescope_lon
            unique_dec = np.zeros(1) + telescope_lat
            unique_pa = None
        else:
            unique_gha = (lst_array[unique_mask] - app_ra[unique_mask]) - telescope_lon
            unique_dec = app_dec[unique_mask]
            unique_pa = 0.0 if frame_pa is None else frame_pa[unique_mask]

        # Tranpose the ant vectors so that they are in the proper shape
        ant_vectors = np.transpose(antenna_positions)[np.newaxis, :, :]
        # Apply rotations, and then reorganize the ndarray so that you can access
        # individual antenna vectors quickly.
        ant_rot_vectors = np.reshape(
            np.transpose(
                _rotate_one_axis(
                    _rotate_two_axis(ant_vectors, unique_gha, unique_dec, 2, 1),
                    unique_pa,
                    0,
                ),
                axes=[0, 2, 1],
            ),
            (-1, 3),
        )

        unique_mask[0] = False
        unique_map = np.cumsum(unique_mask) * N_ants
        new_coords = (
            ant_rot_vectors[unique_map + ant_2_index]
            - ant_rot_vectors[unique_map + ant_1_index]
        )
    else:
        if uvw_array is None:
            raise ValueError("Must include uvw_array if use_ant_pos=False.")
        if from_enu:
            if to_enu:
                # Well this was pointless... returning your uvws unharmed
                return uvw_array
            # Unprojected coordinates appear to be stored in ENU coordinates -- that's
            # equivalent to calculating uvw's based on zenith. We can use that to our
            # advantage and spoof old_app_ra and old_app_dec based on lst_array and
            # telescope_lat
            if telescope_lat is None:
                raise ValueError(
                    "Must include telescope_lat if moving between "
                    "ENU (i.e., 'unprojected') and uvw coordinates!"
                )
            if lst_array is None:
                raise ValueError(
                    "Must include lst_array if moving between ENU "
                    "(i.e., 'unprojected') and uvw coordinates!"
                )
        else:
            if (old_frame_pa is None) and not (frame_pa is None or to_enu):
                raise ValueError(
                    "Must include old_frame_pa values if data are phased and "
                    "applying new position angle values (frame_pa)."
                )
            if ((old_app_ra is None) and not (app_ra is None or to_enu)) or (
                (old_app_dec is None) and not (app_dec is None or to_enu)
            ):
                raise ValueError(
                    "Must include old_app_ra and old_app_dec values when data are "
                    "already phased and phasing to a new position."
                )
        # For this operation, all we need is the delta-ha coverage, which _should_ be
        # entirely encapsulated by the change in RA.
        if (app_ra is None) and (old_app_ra is None):
            gha_delta_array = 0.0
        else:
            gha_delta_array = (lst_array if from_enu else old_app_ra) - (
                lst_array if to_enu else app_ra
            )

        # Notice below there's an axis re-orientation here, to go from uvw -> XYZ,
        # where X is pointing in the direction of the source. This is mostly here
        # for convenience and code legibility -- a slightly different pair of
        # rotations would give you the same result w/o needing to cycle the axes.

        # Up front, we want to trap the corner-case where the sky position you are
        # phasing up to hasn't changed, just the position angle (i.e., which way is
        # up on the map). This is a much easier transform to handle.
        if np.all(gha_delta_array == 0.0) and np.all(old_app_dec == app_dec):
            new_coords = _rotate_one_axis(
                uvw_array[:, [2, 0, 1], np.newaxis],
                frame_pa - (0.0 if old_frame_pa is None else old_frame_pa),
                0,
            )[:, :, 0]
        else:
            new_coords = _rotate_two_axis(
                _rotate_two_axis(  # Yo dawg, I heard you like rotation matricies...
                    uvw_array[:, [2, 0, 1], np.newaxis],
                    0.0 if (from_enu or old_frame_pa is None) else (-old_frame_pa),
                    (-telescope_lat) if from_enu else (-old_app_dec),
                    0,
                    1,
                ),
                gha_delta_array,
                telescope_lat if to_enu else app_dec,
                2,
                1,
            )

            # One final rotation applied here, to compensate for the fact that we want
            # the Dec-axis of our image (Fourier dual to the v-axis) to be aligned with
            # the chosen frame, if we not in ENU coordinates
            if not to_enu:
                new_coords = _rotate_one_axis(new_coords, frame_pa, 0)

            # Finally drop the now-vestigal last axis of the array
            new_coords = new_coords[:, :, 0]

    # There's one last task to do, which is to re-align the axes from projected
    # XYZ -> uvw, where X (which points towards the source) falls on the w axis,
    # and Y and Z fall on the u and v axes, respectively.
    return new_coords[:, [1, 2, 0]]


def transform_sidereal_coords(
    lon,
    lat,
    in_coord_frame,
    out_coord_frame,
    in_coord_epoch=None,
    out_coord_epoch=None,
    time_array=None,
):
    """
    Transform a given set of coordinates from one sidereal coordinate frame to another.

    Uses astropy to convert from a coordinates from sidereal frame into another.
    This function will support transforms from several frames, including GCRS,
    FK5 (i.e., J2000), FK4 (i.e., B1950), Galactic, Supergalactic, CIRS, HCRS, and
    a few others (basically anything that doesn't require knowing the observers
    location on Earth/other celestial body).

    Parameters
    ----------
    lon_coord : float or ndarray of floats
        Logitudinal coordinate to be transformed, typically expressed as the right
        ascension, in units of radians. Can either be a float, or an ndarray of
        floats with shape (Ncoords,). Must agree with lat_coord.
    lat_coord : float or ndarray of floats
        Latitudinal coordinate to be transformed, typically expressed as the
        declination, in units of radians. Can either be a float, or an ndarray of
        floats with shape (Ncoords,). Must agree with lon_coord.
    in_coord_frame : string
        Reference frame for the provided coordinates.  Expected to match a list of
        those supported within the astropy SkyCoord object. An incomplete list includes
        'gcrs', 'fk4', 'fk5', 'galactic', 'supergalactic', 'cirs', and 'hcrs'.
    out_coord_frame : string
        Reference frame to output coordinates in. Expected to match a list of
        those supported within the astropy SkyCoord object. An incomplete list includes
        'gcrs', 'fk4', 'fk5', 'galactic', 'supergalactic', 'cirs', and 'hcrs'.
    in_coord_epoch : float
        Epoch for the input coordinate frame. Optional parameter, only required
        when using either the FK4 (B1950) or FK5 (J2000) coordinate systems. Units are
        in fractional years.
    out_coord_epoch : float
        Epoch for the output coordinate frame. Optional parameter, only required
        when using either the FK4 (B1950) or FK5 (J2000) coordinate systems. Units are
        in fractional years.
    time_array : float or ndarray of floats
        Julian date(s) to which the coordinates correspond to, only used in frames
        with annular motion terms (e.g., abberation in GCRS). Can either be a float,
        or an ndarray of floats with shape (Ntimes,), assuming that either lat_coord
        and lon_coord are floats, or that Ntimes == Ncoords.

    Returns
    -------
    new_lat : float or ndarray of floats
        Longitudinal coordinates, in units of radians. Output will be an ndarray
        if any inputs were, with shape (Ncoords,) or (Ntimes,), depending on inputs.
    new_lon : float or ndarray of floats
        Latidudinal coordinates, in units of radians. Output will be an ndarray
        if any inputs were, with shape (Ncoords,) or (Ntimes,), depending on inputs.
    """
    lon_coord = lon * units.rad
    lat_coord = lat * units.rad

    # Check here to make sure that lat_coord and lon_coord are the same length,
    # either 1 or len(time_array)
    if lat_coord.shape != lon_coord.shape:
        raise ValueError("lon and lat must be the same shape.")

    if lon_coord.ndim == 0:
        lon_coord.shape += (1,)
        lat_coord.shape += (1,)

    # Check to make sure that we have a properly formatted epoch for our in-bound
    # coordinate frame
    in_epoch = None
    if isinstance(in_coord_epoch, str) or isinstance(in_coord_epoch, Time):
        # If its a string or a Time object, we don't need to do anything more
        in_epoch = Time(in_coord_epoch)
    elif in_coord_epoch is not None:
        if in_coord_frame.lower() in ["fk4", "fk4noeterms"]:
            in_epoch = Time(in_coord_epoch, format="byear")
        else:
            in_epoch = Time(in_coord_epoch, format="jyear")

    # Now do the same for the outbound frame
    out_epoch = None
    if isinstance(out_coord_epoch, str) or isinstance(out_coord_epoch, Time):
        # If its a string or a Time object, we don't need to do anything more
        out_epoch = Time(out_coord_epoch)
    elif out_coord_epoch is not None:
        if out_coord_frame.lower() in ["fk4", "fk4noeterms"]:
            out_epoch = Time(out_coord_epoch, format="byear")
        else:
            out_epoch = Time(out_coord_epoch, format="jyear")

    # Make sure that time array matched up with what we expect. Thanks to astropy
    # weirdness, time_array has to be the same length as lat/lon coords
    rep_time = False
    rep_crds = False
    if time_array is None:
        time_obj_array = None
    else:
        if isinstance(time_array, Time):
            time_obj_array = time_array
        else:
            time_obj_array = Time(time_array, format="jd", scale="utc")
        if (time_obj_array.size != 1) and (lon_coord.size != 1):
            if time_obj_array.shape != lon_coord.shape:
                raise ValueError(
                    "Shape of time_array must be either that of "
                    " lat_coord/lon_coord if len(time_array) > 1."
                )
        else:
            rep_crds = (time_obj_array.size != 1) and (lon_coord.size == 1)
            rep_time = (time_obj_array.size == 1) and (lon_coord.size != 1)
    if rep_crds:
        lon_coord = np.repeat(lon_coord, len(time_array))
        lat_coord = np.repeat(lat_coord, len(time_array))
    if rep_time:
        time_obj_array = Time(
            np.repeat(time_obj_array.jd, len(lon_coord)), format="jd", scale="utc"
        )
    coord_object = SkyCoord(
        lon_coord,
        lat_coord,
        frame=in_coord_frame,
        equinox=in_epoch,
        obstime=time_obj_array,
    )

    # Easiest, most general way to transform to the new frame is to create a dummy
    # SkyCoord with all the attributes needed -- note that we particularly need this
    # in order to use a non-standard equinox/epoch
    new_coord = coord_object.transform_to(
        SkyCoord(0, 0, unit="rad", frame=out_coord_frame, equinox=out_epoch)
    )

    return new_coord.spherical.lon.rad, new_coord.spherical.lat.rad


def transform_icrs_to_app(
    time_array,
    ra,
    dec,
    telescope_loc,
    telescope_frame="itrs",
    ellipsoid=None,
    epoch=2000.0,
    pm_ra=None,
    pm_dec=None,
    vrad=None,
    dist=None,
    astrometry_library=None,
):
    """
    Transform a set of coordinates in ICRS to topocentric/apparent coordinates.

    This utility uses one of three libraries (astropy, NOVAS, or ERFA) to calculate
    the apparent (i.e., topocentric) coordinates of a source at a given time and
    location, given a set of coordinates expressed in the ICRS frame. These coordinates
    are most typically used for defining the phase center of the array (i.e, calculating
    baseline vectors).

    As of astropy v4.2, the agreement between the three libraries is consistent down to
    the level of better than 1 mas, with the values produced by astropy and pyERFA
    consistent to bettter than 10 µas (this is not surprising, given that astropy uses
    pyERFA under the hood for astrometry). ERFA is the default as it outputs
    coordinates natively in the apparent frame (whereas NOVAS and astropy do not), as
    well as the fact that of the three libraries, it produces results the fastest.

    Parameters
    ----------
    time_array : float or array-like of float
        Julian dates to calculate coordinate positions for. Can either be a single
        float, or an array-like of shape (Ntimes,).
    ra : float or array-like of float
        ICRS RA of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (with the exception of telescope location parameters).
    dec : float or array-like of float
        ICRS Dec of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (with the exception of telescope location parameters).
    telescope_loc : array-like of floats or EarthLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, or a tuple
        of shape (3,) containing (in order) the latitude, longitude, and altitude,
        in units of radians, radians, and meters, respectively.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    epoch : int or float or str or Time object
        Epoch of the coordinate data supplied, only used when supplying proper motion
        values. If supplying a number, it will assumed to be in Julian years. Default
        is J2000.0.
    pm_ra : float or array-like of float
        Proper motion in RA of the source, expressed in units of milliarcsec / year.
        Proper motion values are applied relative to the J2000 (i.e., RA/Dec ICRS
        values should be set to their expected values when the epoch is 2000.0).
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Note that
        units are in dRA/dt, not cos(Dec)*dRA/dt. Not required.
    pm_dec : float or array-like of float
        Proper motion in Dec of the source, expressed in units of milliarcsec / year.
        Proper motion values are applied relative to the J2000 (i.e., RA/Dec ICRS
        values should be set to their expected values when the epoch is 2000.0).
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Not
        required.
    vrad : float or array-like of float
        Radial velocity of the source, expressed in units of km / sec. Can either be
        a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (namely ra_coord and dec_coord). Not required.
    dist : float or array-like of float
        Distance of the source, expressed in milliarcseconds. Can either be a single
        float or array of shape (Ntimes,), although this must be consistent with other
        parameters (namely ra_coord and dec_coord). Not required.
    astrometry_library : str
        Library used for running the coordinate conversions. Allowed options are
        'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas library),
        and 'astropy' (which uses the astropy utilities). Default is erfa unless
        the telescope_location is a MoonLocation object, in which case the default is
        astropy.

    Returns
    -------
    app_ra : ndarray of floats
        Apparent right ascension coordinates, in units of radians, of shape (Ntimes,).
    app_dec : ndarray of floats
        Apparent declination coordinates, in units of radians, of shape (Ntimes,).
    """
    if telescope_frame.upper() == "MCMF":
        if not hasmoon:
            raise ValueError(
                "Need to install `lunarsky` package to work with MCMF frame."
            )
        if ellipsoid is None:
            ellipsoid = "SPHERE"

    # Make sure that the library requested is actually permitted
    if astrometry_library is None:
        if hasmoon and isinstance(telescope_loc, MoonLocation):
            astrometry_library = "astropy"
        elif telescope_frame.upper() == "MCMF":
            astrometry_library = "astropy"
        else:
            astrometry_library = "erfa"

    if astrometry_library not in ["erfa", "novas", "astropy"]:
        raise ValueError(
            "Requested coordinate transformation library is not supported, please "
            "select either 'erfa', 'novas', or 'astropy' for astrometry_library."
        )

    ra_coord = ra * units.rad
    dec_coord = dec * units.rad

    # Check here to make sure that ra_coord and dec_coord are the same length,
    # either 1 or len(time_array)
    multi_coord = ra_coord.size != 1
    if ra_coord.shape != dec_coord.shape:
        raise ValueError("ra and dec must be the same shape.")

    pm_ra_coord = None if pm_ra is None else pm_ra * (units.mas / units.yr)
    pm_dec_coord = None if pm_dec is None else pm_dec * (units.mas / units.yr)
    d_coord = (
        None if (dist is None or np.all(dist == 0.0)) else Distance(dist * units.pc)
    )
    v_coord = None if vrad is None else vrad * (units.km / units.s)

    opt_list = [pm_ra_coord, pm_dec_coord, d_coord, v_coord]
    opt_names = ["pm_ra", "pm_dec", "dist", "vrad"]
    # Check the optional inputs, make sure that they're sensible
    for item, name in zip(opt_list, opt_names):
        if item is not None:
            if ra_coord.shape != item.shape:
                raise ValueError("%s must be the same shape as ra and dec." % name)

    if isinstance(telescope_loc, EarthLocation) or (
        hasmoon and isinstance(telescope_loc, MoonLocation)
    ):
        site_loc = telescope_loc
    elif telescope_frame.upper() == "MCMF":
        site_loc = MoonLocation.from_selenodetic(
            telescope_loc[1] * (180.0 / np.pi),
            telescope_loc[0] * (180.0 / np.pi),
            height=telescope_loc[2],
            ellipsoid=ellipsoid,
        )
    else:
        site_loc = EarthLocation.from_geodetic(
            telescope_loc[1] * (180.0 / np.pi),
            telescope_loc[0] * (180.0 / np.pi),
            height=telescope_loc[2],
        )

    if (
        hasmoon
        and isinstance(site_loc, MoonLocation)
        and astrometry_library != "astropy"
    ):
        raise NotImplementedError(
            "MoonLocation telescopes are only supported with the 'astropy' astrometry "
            "library"
        )

    # Useful for both astropy and novas methods, the latter of which gives easy
    # access to the IERS data that we want.
    if isinstance(time_array, Time):
        time_obj_array = time_array
    else:
        time_obj_array = Time(time_array, format="jd", scale="utc")

    if time_obj_array.size != 1:
        if (time_obj_array.shape != ra_coord.shape) and multi_coord:
            raise ValueError(
                "time_array must be of either of length 1 (single "
                "float) or same length as ra and dec."
            )
    elif time_obj_array.ndim == 0:
        # Make the array at least 1-dimensional so we don't run into indexing
        # issues later.
        time_obj_array = Time([time_obj_array])

    # Check to make sure that we have a properly formatted epoch for our in-bound
    # coordinate frame
    coord_epoch = None
    if isinstance(epoch, str) or isinstance(epoch, Time):
        # If its a string or a Time object, we don't need to do anything more
        coord_epoch = Time(epoch)
    elif epoch is not None:
        coord_epoch = Time(epoch, format="jyear")

    # Note if time_array is a single element
    multi_time = time_obj_array.size != 1

    # Get IERS data, which is needed for NOVAS and ERFA
    polar_motion_data = iers.earth_orientation_table.get()

    pm_x_array, pm_y_array = polar_motion_data.pm_xy(time_obj_array)
    delta_x_array, delta_y_array = polar_motion_data.dcip_xy(time_obj_array)

    pm_x_array = pm_x_array.to_value("arcsec")
    pm_y_array = pm_y_array.to_value("arcsec")
    delta_x_array = delta_x_array.to_value("marcsec")
    delta_y_array = delta_y_array.to_value("marcsec")
    # Catch the case where we don't have CIP delta values yet (they don't typically have
    # predictive values like the polar motion does)
    delta_x_array[np.isnan(delta_x_array)] = 0.0
    delta_y_array[np.isnan(delta_y_array)] = 0.0

    # If the source was instantiated w/ floats, it'll be a 0-dim object, which will
    # throw errors if we try to treat it as an array. Reshape to a 1D array of len 1
    # so that all the calls can be uniform
    if ra_coord.ndim == 0:
        ra_coord.shape += (1,)
        dec_coord.shape += (1,)
        if pm_ra_coord is not None:
            pm_ra
        if d_coord is not None:
            d_coord.shape += (1,)
        if v_coord is not None:
            v_coord.shape += (1,)

    # If there is an epoch and a proper motion, apply that motion now

    if astrometry_library == "astropy":
        # Astropy doesn't have (oddly enough) a way of getting at the apparent RA/Dec
        # directly, but we can cheat this by going to AltAz, and then coverting back
        # to apparent RA/Dec using the telescope lat and LAST.
        if (epoch is not None) and (pm_ra is not None) and (pm_dec is not None):
            # astropy is a bit weird in how it handles proper motion, so rather than
            # fight with it to do it all in one step, we separate it into two: first
            # apply proper motion to ICRS, then transform to topocentric.
            sky_coord = SkyCoord(
                ra=ra_coord,
                dec=dec_coord,
                pm_ra_cosdec=pm_ra_coord * np.cos(dec_coord),
                pm_dec=pm_dec_coord,
                frame="icrs",
            )

            sky_coord = sky_coord.apply_space_motion(dt=(time_obj_array - coord_epoch))
            ra_coord = sky_coord.ra
            dec_coord = sky_coord.dec
            if d_coord is not None:
                d_coord = d_coord.repeat(ra_coord.size)
            if v_coord is not None:
                v_coord = v_coord.repeat(ra_coord.size)

        if isinstance(site_loc, EarthLocation):
            sky_coord = SkyCoord(
                ra=ra_coord,
                dec=dec_coord,
                distance=d_coord,
                radial_velocity=v_coord,
                frame="icrs",
            )

            azel_data = sky_coord.transform_to(
                SkyCoord(
                    np.zeros_like(time_obj_array) * units.rad,
                    np.zeros_like(time_obj_array) * units.rad,
                    location=site_loc,
                    obstime=time_obj_array,
                    frame="altaz",
                )
            )
        else:
            sky_coord = LunarSkyCoord(
                ra=ra_coord,
                dec=dec_coord,
                distance=d_coord,
                radial_velocity=v_coord,
                frame="icrs",
            )

            azel_data = sky_coord.transform_to(
                LunarSkyCoord(
                    np.zeros_like(time_obj_array) * units.rad,
                    np.zeros_like(time_obj_array) * units.rad,
                    location=site_loc,
                    obstime=time_obj_array,
                    frame="lunartopo",
                )
            )
            time_obj_array = LTime(time_obj_array)

        time_obj_array.location = site_loc
        app_ha, app_dec = erfa.ae2hd(
            azel_data.az.rad, azel_data.alt.rad, site_loc.lat.rad
        )
        app_ra = np.mod(
            time_obj_array.sidereal_time("apparent").rad - app_ha, 2 * np.pi
        )

    elif astrometry_library == "novas":
        # Import the NOVAS library only if it's needed/available.
        try:
            import novas_de405  # noqa
            from novas import compat as novas
            from novas.compat import eph_manager
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "novas and/or novas_de405 are not installed but is required for "
                "NOVAS functionality"
            ) from e

        # Call is needed to load high-precision ephem data in NOVAS
        jd_start, jd_end, number = eph_manager.ephem_open()

        # Define the obs location, which is needed to calculate diurnal abb term
        # and polar wobble corrections
        site_loc = novas.make_on_surface(
            site_loc.lat.deg,  # latitude in deg
            site_loc.lon.deg,  # Longitude in deg
            site_loc.height.to_value("m"),  # Height in meters
            0.0,  # Temperature, set to 0 for now (no atm refrac)
            0.0,  # Pressure, set to 0 for now (no atm refrac)
        )

        # NOVAS wants things in terrestial time and UT1
        tt_time_array = time_obj_array.tt.jd
        ut1_time_array = time_obj_array.ut1.jd
        gast_array = time_obj_array.sidereal_time("apparent", "greenwich").rad

        if np.any(tt_time_array < jd_start) or np.any(tt_time_array > jd_end):
            raise ValueError(
                "No current support for JPL ephems outside of 1700 - 2300 AD. "
                "Check back later (or possibly earlier)..."
            )

        app_ra = np.zeros(tt_time_array.shape) + np.zeros(ra_coord.shape)
        app_dec = np.zeros(tt_time_array.shape) + np.zeros(ra_coord.shape)

        for idx in range(len(app_ra)):
            if multi_coord or (idx == 0):
                # Create a catalog entry for the source in question
                if pm_ra is None:
                    pm_ra_use = 0.0
                else:
                    pm_ra_use = pm_ra_coord.to_value("mas/yr") * np.cos(
                        dec_coord[idx].to_value("rad")
                    )

                if pm_dec is None:
                    pm_dec_use = 0.0
                else:
                    pm_dec_use = pm_dec_coord.to_value("mas/yr")

                if dist is None or np.any(dist == 0.0):
                    parallax = 0.0
                else:
                    parallax = d_coord[idx].kiloparsec ** -1.0

                if vrad is None:
                    vrad_use = 0.0
                else:
                    vrad_use = v_coord[idx].to_value("km/s")

                cat_entry = novas.make_cat_entry(
                    "dummy_name",  # Dummy source name
                    "GKK",  # Catalog ID, fixed for now
                    156,  # Star ID number, fixed for now
                    ra_coord[idx].to_value("hourangle"),
                    dec_coord[idx].to_value("deg"),
                    pm_ra_use,
                    pm_dec_use,
                    parallax,
                    vrad_use,
                )

            # Update polar wobble parameters for a given timestamp
            if multi_time or (idx == 0):
                gast = gast_array[idx]
                pm_x = pm_x_array[idx] * np.cos(gast) + pm_y_array[idx] * np.sin(gast)
                pm_y = pm_y_array[idx] * np.cos(gast) - pm_x_array[idx] * np.sin(gast)
                tt_time = tt_time_array[idx]
                ut1_time = ut1_time_array[idx]
                novas.cel_pole(tt_time, 2, delta_x_array[idx], delta_y_array[idx])

            # Calculate topocentric RA/Dec values
            [temp_ra, temp_dec] = novas.topo_star(
                tt_time, (tt_time - ut1_time) * 86400.0, cat_entry, site_loc, accuracy=0
            )
            xyz_array = polar2_to_cart3(
                temp_ra * (np.pi / 12.0), temp_dec * (np.pi / 180.0)
            )
            xyz_array = novas.wobble(tt_time, pm_x, pm_y, xyz_array, 1)

            app_ra[idx], app_dec[idx] = cart3_to_polar2(np.array(xyz_array))
    elif astrometry_library == "erfa":
        # liberfa wants things in radians
        pm_x_array *= np.pi / (3600.0 * 180.0)
        pm_y_array *= np.pi / (3600.0 * 180.0)

        if pm_ra is None:
            pm_ra_use = 0.0
        else:
            pm_ra_use = pm_ra_coord.to_value("rad/yr")

        if pm_dec is None:
            pm_dec_use = 0.0
        else:
            pm_dec_use = pm_dec_coord.to_value("rad/yr")

        if dist is None or np.any(dist == 0.0):
            parallax = 0.0
        else:
            parallax = d_coord.pc**-1.0

        if vrad is None:
            vrad_use = 0
        else:
            vrad_use = v_coord.to_value("km/s")

        [_, _, _, app_dec, app_ra, eqn_org] = erfa.atco13(
            ra_coord.to_value("rad"),
            dec_coord.to_value("rad"),
            pm_ra_use,
            pm_dec_use,
            parallax,
            vrad_use,
            time_obj_array.utc.jd1,
            time_obj_array.utc.jd2,
            time_obj_array.delta_ut1_utc,
            site_loc.lon.rad,
            site_loc.lat.rad,
            site_loc.height.to_value("m"),
            pm_x_array,
            pm_y_array,
            0,  # ait pressure, used for refraction (ignored)
            0,  # amb temperature, used for refraction (ignored)
            0,  # rel humidity, used for refraction (ignored)
            0,  # wavelength, used for refraction (ignored)
        )

        app_ra = np.mod(app_ra - eqn_org, 2 * np.pi)

    return app_ra, app_dec


def transform_app_to_icrs(
    time_array,
    app_ra,
    app_dec,
    telescope_loc,
    telescope_frame="itrs",
    ellipsoid=None,
    astrometry_library=None,
):
    """
    Transform a set of coordinates in topocentric/apparent to ICRS coordinates.

    This utility uses either astropy or erfa to calculate the ICRS  coordinates of
    a given set of apparent source coordinates. These coordinates are most typically
    used for defining the celestial/catalog position of a source. Note that at present,
    this is only implemented in astropy and pyERFA, although it could hypothetically
    be extended to NOVAS at some point.

    Parameters
    ----------
    time_array : float or ndarray of float
        Julian dates to calculate coordinate positions for. Can either be a single
        float, or an ndarray of shape (Ntimes,).
    app_ra : float or ndarray of float
        ICRS RA of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ncoord,). Note that if time_array is
        not a singleton value, then Ncoord must be equal to Ntimes.
    app_dec : float or ndarray of float
        ICRS Dec of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ncoord,). Note that if time_array is
        not a singleton value, then Ncoord must be equal to Ntimes.
    telescope_loc : tuple of floats or EarthLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, or a tuple
        of shape (3,) containing (in order) the latitude, longitude, and altitude,
        in units of radians, radians, and meters, respectively.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    astrometry_library : str
        Library used for running the coordinate conversions. Allowed options are
        'erfa' (which uses the pyERFA), and 'astropy' (which uses the astropy
        utilities). Default is erfa unless the telescope_location is a MoonLocation
        object, in which case the default is astropy.

    Returns
    -------
    icrs_ra : ndarray of floats
        ICRS right ascension coordinates, in units of radians, of either shape
        (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    icrs_dec : ndarray of floats
        ICRS declination coordinates, in units of radians, of either shape
        (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    """
    if telescope_frame.upper() == "MCMF":
        if not hasmoon:
            raise ValueError(
                "Need to install `lunarsky` package to work with MCMF frame."
            )
        if ellipsoid is None:
            ellipsoid = "SPHERE"

    # Make sure that the library requested is actually permitted
    if astrometry_library is None:
        if hasmoon and isinstance(telescope_loc, MoonLocation):
            astrometry_library = "astropy"
        elif telescope_frame.upper() == "MCMF":
            astrometry_library = "astropy"
        else:
            astrometry_library = "erfa"

    if astrometry_library not in ["erfa", "astropy"]:
        raise ValueError(
            "Requested coordinate transformation library is not supported, please "
            "select either 'erfa' or 'astropy' for astrometry_library."
        )

    ra_coord = app_ra * units.rad
    dec_coord = app_dec * units.rad

    # Check here to make sure that ra_coord and dec_coord are the same length,
    # either 1 or len(time_array)
    multi_coord = ra_coord.size != 1
    if ra_coord.shape != dec_coord.shape:
        raise ValueError("app_ra and app_dec must be the same shape.")

    if isinstance(telescope_loc, EarthLocation) or (
        hasmoon and isinstance(telescope_loc, MoonLocation)
    ):
        site_loc = telescope_loc
    elif telescope_frame.upper() == "MCMF":
        site_loc = MoonLocation.from_selenodetic(
            telescope_loc[1] * (180.0 / np.pi),
            telescope_loc[0] * (180.0 / np.pi),
            height=telescope_loc[2],
            ellipsoid=ellipsoid,
        )
    else:
        site_loc = EarthLocation.from_geodetic(
            telescope_loc[1] * (180.0 / np.pi),
            telescope_loc[0] * (180.0 / np.pi),
            height=telescope_loc[2],
        )

    if (
        hasmoon
        and isinstance(site_loc, MoonLocation)
        and astrometry_library != "astropy"
    ):
        raise NotImplementedError(
            "MoonLocation telescopes are only supported with the 'astropy' astrometry "
            "library"
        )

    assert time_array.size > 0
    if isinstance(time_array, Time):
        time_obj_array = time_array
    else:
        time_obj_array = Time(time_array, format="jd", scale="utc")

    if time_obj_array.size != 1:
        if (time_obj_array.shape != ra_coord.shape) and multi_coord:
            raise ValueError(
                "time_array must be of either of length 1 (single "
                "float) or same length as ra and dec."
            )
    elif time_obj_array.ndim == 0:
        # Make the array at least 1-dimensional so we don't run into indexing
        # issues later.
        time_obj_array = Time([time_obj_array])

    if astrometry_library == "astropy":
        if hasmoon and isinstance(site_loc, MoonLocation):
            time_obj_array = LTime(time_obj_array)

        time_obj_array.location = site_loc

        az_coord, el_coord = erfa.hd2ae(
            np.mod(
                time_obj_array.sidereal_time("apparent").rad - ra_coord.to_value("rad"),
                2 * np.pi,
            ),
            dec_coord.to_value("rad"),
            site_loc.lat.rad,
        )

        if isinstance(site_loc, EarthLocation):
            sky_coord = SkyCoord(
                az_coord * units.rad,
                el_coord * units.rad,
                frame="altaz",
                location=site_loc,
                obstime=time_obj_array,
            )
        else:
            sky_coord = LunarSkyCoord(
                az_coord * units.rad,
                el_coord * units.rad,
                frame="lunartopo",
                location=site_loc,
                obstime=time_obj_array,
            )

        coord_data = sky_coord.transform_to("icrs")
        icrs_ra = coord_data.ra.rad
        icrs_dec = coord_data.dec.rad
    elif astrometry_library == "erfa":
        # Get IERS data, which is needed for highest precision
        polar_motion_data = iers.earth_orientation_table.get()

        pm_x_array, pm_y_array = polar_motion_data.pm_xy(time_obj_array)
        pm_x_array = pm_x_array.to_value("rad")
        pm_y_array = pm_y_array.to_value("rad")

        bpn_matrix = erfa.pnm06a(time_obj_array.tt.jd1, time_obj_array.tt.jd2)
        cip_x, cip_y = erfa.bpn2xy(bpn_matrix)
        cio_s = erfa.s06(time_obj_array.tt.jd1, time_obj_array.tt.jd2, cip_x, cip_y)
        eqn_org = erfa.eors(bpn_matrix, cio_s)

        # Observed to ICRS via ERFA
        icrs_ra, icrs_dec = erfa.atoc13(
            "r",
            ra_coord.to_value("rad") + eqn_org,
            dec_coord.to_value("rad"),
            time_obj_array.utc.jd1,
            time_obj_array.utc.jd2,
            time_obj_array.delta_ut1_utc,
            site_loc.lon.rad,
            site_loc.lat.rad,
            site_loc.height.value,
            pm_x_array,
            pm_y_array,
            0,  # atm pressure, used for refraction (ignored)
            0,  # amb temperature, used for refraction (ignored)
            0,  # rel humidity, used for refraction (ignored)
            0,  # wavelength, used for refraction (ignored)
        )

    # Return back the two RA/Dec arrays
    return icrs_ra, icrs_dec


def calc_parallactic_angle(app_ra, app_dec, lst_array, telescope_lat):
    """
    Calculate the parallactic angle between RA/Dec and the AltAz frame.

    Parameters
    ----------
    app_ra : ndarray of floats
        Array of apparent RA values in units of radians, shape (Ntimes,).
    app_dec : ndarray of floats
        Array of apparent dec values in units of radians, shape (Ntimes,).
    telescope_lat : float
        Latitude of the observatory, in units of radians.
    lst_array : float or ndarray of float
        Array of local apparent sidereal timesto calculate position angle values
        for, in units of radians. Can either be a single float or an array of shape
        (Ntimes,).
    """
    # This is just a simple wrapped around the pas function in ERFA
    return erfa.pas(app_ra, app_dec, lst_array, telescope_lat)


def calc_frame_pos_angle(
    time_array,
    app_ra,
    app_dec,
    telescope_loc,
    ref_frame,
    ref_epoch=None,
    telescope_frame="itrs",
    ellipsoid=None,
    offset_pos=(np.pi / 360.0),
):
    """
    Calculate an position angle given apparent position and reference frame.

    This function is used to determine the position angle between the great
    circle of declination in apparent coordinates, versus that in a given
    reference frame. Note that this is slightly different than parallactic
    angle, which is the difference between apparent declination and elevation.

    Paramters
    ---------
    time_array : ndarray of floats
        Array of julian dates to calculate position angle values for, of shape
        (Ntimes,).
    app_ra : ndarray of floats
        Array of apparent RA values in units of radians, shape (Ntimes,).
    app_dec : ndarray of floats
        Array of apparent dec values in units of radians, shape (Ntimes,).
    telescope_loc : tuple of floats or EarthLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the observer.
        Can either be provided as an astropy EarthLocation, or an array-like of shape
        (3,) containing the latitude, longitude, and altitude, in that order, with units
        of radians, radians, and meters, respectively.
    ref_frame : str
        Coordinate frame to calculate position angles for. Can be any of the
        several supported frames in astropy (a limited list: fk4, fk5, icrs,
        gcrs, cirs, galactic).
    ref_epoch : str or flt
        Epoch of the coordinates, only used when ref_frame = fk4 or fk5. Given
        in unites of fractional years, either as a float or as a string with
        the epoch abbreviation (e.g, Julian epoch 2000.0 would be J2000.0).
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    offset_pos : float
        Distance of the offset position used to calculate the frame PA. Default
        is 0.5 degrees, which should be sufficent for most applications.


    Returns
    -------
    frame_pa : ndarray of floats
        Array of position angles, in units of radians.
    """
    # Check to see if the position angles should default to zero
    if (ref_frame is None) or (ref_frame == "topo"):
        # No-op detected, ENGAGE MAXIMUM SNARK!
        return np.zeros_like(time_array)

    assert offset_pos > 0, "offset_pos must be greater than 0."

    if telescope_frame == "mcmf" and ellipsoid is None:
        ellipsoid = "SPHERE"

    # This creates an array of unique entries of ra + dec + time, since the processing
    # time for each element can be non-negligible, and entries along the Nblt axis can
    # be highly redundant.
    unique_mask = np.union1d(
        np.union1d(
            np.unique(app_ra, return_index=True)[1],
            np.unique(app_dec, return_index=True)[1],
        ),
        np.unique(time_array, return_index=True)[1],
    )

    # Pluck out the unique entries for each
    unique_ra = app_ra[unique_mask]
    unique_dec = app_dec[unique_mask]
    unique_time = time_array[unique_mask]

    # Figure out how many elements we need to transform
    n_coord = len(unique_mask)

    # Offset north/south positions by 0.5 deg, such that the PA is determined over a
    # 1 deg arc.
    up_dec = unique_dec + offset_pos
    dn_dec = unique_dec - offset_pos
    up_ra = dn_ra = unique_ra

    # Wrap the positions if they happen to go over the poles
    up_ra[up_dec > (np.pi / 2.0)] = np.mod(
        up_ra[up_dec > (np.pi / 2.0)] + np.pi, 2.0 * np.pi
    )
    up_dec[up_dec > (np.pi / 2.0)] = np.pi - up_dec[up_dec > (np.pi / 2.0)]

    dn_ra[-dn_dec > (np.pi / 2.0)] = np.mod(
        dn_ra[dn_dec > (np.pi / 2.0)] + np.pi, 2.0 * np.pi
    )
    dn_dec[-dn_dec > (np.pi / 2.0)] = np.pi - dn_dec[-dn_dec > (np.pi / 2.0)]

    # Run the set of offset coordinates through the "reverse" transform. The two offset
    # positions are concat'd together to help reduce overheads
    ref_ra, ref_dec = calc_sidereal_coords(
        np.tile(unique_time, 2),
        np.concatenate((dn_ra, up_ra)),
        np.concatenate((dn_dec, up_dec)),
        telescope_loc,
        ref_frame,
        telescope_frame=telescope_frame,
        ellipsoid=ellipsoid,
        coord_epoch=ref_epoch,
    )

    # Use the pas function from ERFA to calculate the position angle. The negative sign
    # is here because we're measuring PA of app -> frame, but we want frame -> app.
    unique_pa = -erfa.pas(
        ref_ra[:n_coord], ref_dec[:n_coord], ref_ra[n_coord:], ref_dec[n_coord:]
    )

    # Finally, we have to go back through and "fill in" the redundant entries
    frame_pa = np.zeros_like(app_ra)
    for idx in range(n_coord):
        select_mask = np.logical_and(
            np.logical_and(unique_ra[idx] == app_ra, unique_dec[idx] == app_dec),
            unique_time[idx] == time_array,
        )
        frame_pa[select_mask] = unique_pa[idx]

    return frame_pa


def lookup_jplhorizons(
    target_name,
    time_array,
    telescope_loc=None,
    high_cadence=False,
    force_indv_lookup=None,
):
    """
    Lookup solar system body coordinates via the JPL-Horizons service.

    This utility is useful for generating ephemerides, which can then be interpolated in
    order to provide positional data for a target which is moving, such as planetary
    bodies and other solar system objects. Use of this function requires the
    installation of the `astroquery` module.


    Parameters
    ----------
    target_name : str
        Name of the target to gather an ephemeris for. Must match the name
        in the JPL-Horizons database.
    time_array : array-like of float
        Times in UTC Julian days to gather an ephemeris for.
    telescope_loc : array-like of float
        ITRF latitude, longitude, and altitude (rel to sea-level) of the observer. Must
        be an array-like of shape (3,) containing the latitude, longitude, and
        altitude, in that order, with units of radians, radians, and meters,
        respectively.
    high_cadence : bool
        If set to True, will calculate ephemeris points every 3 minutes in time, as
        opposed to the default of every 3 hours.
    force_indv_lookup : bool
        If set to True, will calculate coordinate values for each value found within
        `time_array`. If False, a regularized time grid is sampled that encloses the
        values contained within `time_array`. Default is False, unless `time_array` is
        of length 1, in which the default is set to True.


    Returns
    -------
    ephem_times : ndarray of float
        Times for which the ephemeris values were calculated, in UTC Julian days.
    ephem_ra : ndarray of float
        ICRS Right ascension of the target at the values within `ephem_times`, in
        units of radians.
    ephem_dec : ndarray of float
        ICRS Declination of the target at the values within `ephem_times`, in units
        of radians.
    ephem_dist : ndarray of float
        Distance of the target relative to the observer, at the values within
        `ephem_times`, in units of parsecs.
    ephem_vel : ndarray of float
        Velocity of the targets relative to the observer, at the values within
        `ephem_times`, in units of km/sec.
    """
    try:
        from astroquery.jplhorizons import Horizons
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "astroquery is not installed but is required for "
            "planet ephemeris functionality"
        ) from err
    from json import load as json_load
    from os.path import join as path_join

    from pyuvdata.data import DATA_PATH

    # Get the telescope location into a format that JPL-Horizons can understand,
    # which is nominally a dict w/ entries for lon (units of deg), lat (units of
    # deg), and elevation (units of km).
    if isinstance(telescope_loc, EarthLocation):
        site_loc = {
            "lon": telescope_loc.lon.deg,
            "lat": telescope_loc.lat.deg,
            "elevation": telescope_loc.height.to_value(unit=units.km),
        }
    elif hasmoon and isinstance(telescope_loc, MoonLocation):
        raise NotImplementedError(
            "Cannot lookup JPL positions for telescopes with a MoonLocation"
        )
    elif telescope_loc is None:
        # Setting to None will report the geocentric position
        site_loc = None
    else:
        site_loc = {
            "lon": telescope_loc[1] * (180.0 / np.pi),
            "lat": telescope_loc[0] * (180.0 / np.pi),
            "elevation": telescope_loc[2] * (0.001),  # m -> km
        }

    # If force_indv_lookup is True, or unset but only providing a single value, then
    # just calculate the RA/Dec for the times requested rather than creating a table
    # to interpolate from.
    if force_indv_lookup or (
        (np.array(time_array).size == 1) and (force_indv_lookup is None)
    ):
        epoch_list = np.unique(time_array)
        if len(epoch_list) > 50:
            raise ValueError(
                "Requesting too many individual ephem points from JPL-Horizons. This "
                "can be remedied by setting force_indv_lookup=False or limiting the "
                "number of values in time_array."
            )
    else:
        # When querying for multiple times, its faster (and kinder to the
        # good folks at JPL) to create a range to query, and then interpolate
        # between values. The extra buffer of 0.001 or 0.25 days for high and
        # low cadence is to give enough data points to allow for spline
        # interpolation of the data.
        if high_cadence:
            start_time = np.min(time_array) - 0.001
            stop_time = np.max(time_array) + 0.001
            step_time = "3m"
            n_entries = (stop_time - start_time) * (1440.0 / 3.0)
        else:
            # The start/stop time here are setup to maximize reusability of the
            # data, since astroquery appears to cache the results from previous
            # queries.
            start_time = (0.25 * np.floor(4.0 * np.min(time_array))) - 0.25
            stop_time = (0.25 * np.ceil(4.0 * np.max(time_array))) + 0.25
            step_time = "3h"
            n_entries = (stop_time - start_time) * (24.0 / 3.0)
        # We don't want to overtax the JPL service, so limit ourselves to 1000
        # individual queries at a time. Note that this is likely a conservative
        # cap for JPL-Horizons, but there should be exceptionally few applications
        # that actually require more than this.
        if n_entries > 1000:
            if (len(np.unique(time_array)) <= 50) and (force_indv_lookup is None):
                # If we have a _very_ sparse set of epochs, pass that along instead
                epoch_list = np.unique(time_array)
            else:
                # Otherwise, time to raise an error
                raise ValueError(
                    "Too many ephem points requested from JPL-Horizons. This "
                    "can be remedied by setting high_cadance=False or limiting "
                    "the number of values in time_array."
                )
        else:
            epoch_list = {
                "start": Time(start_time, format="jd").isot,
                "stop": Time(stop_time, format="jd").isot,
                "step": step_time,
            }
    # Check to make sure dates are within the 1700-2200 time range,
    # since not all targets are supported outside of this range
    if (np.min(time_array) < 2341973.0) or (np.max(time_array) > 2524593.0):
        raise ValueError(
            "No current support for JPL ephems outside of 1700 - 2300 AD. "
            "Check back later (or possibly earlier)..."
        )

    # JPL-Horizons has a separate catalog with what it calls 'major bodies',
    # and will throw an error if you use the wrong catalog when calling for
    # astrometry. We'll use the dict below to capture this behavior.
    with open(path_join(DATA_PATH, "jpl_major_bodies.json"), "r") as fhandle:
        major_body_dict = json_load(fhandle)

    target_id = target_name
    id_type = "smallbody"
    # If we find the target in the major body database, then we can extract the
    # target ID to make the query a bit more robust (otherwise JPL-Horizons will fail
    # on account that id will find multiple partial matches: e.g., "Mars" will be
    # matched with "Mars", "Mars Explorer", "Mars Barycenter"..., and JPL-Horizons will
    # not know which to choose).
    if target_name in major_body_dict.keys():
        target_id = major_body_dict[target_name]
        id_type = None

    query_obj = Horizons(
        id=target_id, location=site_loc, epochs=epoch_list, id_type=id_type
    )
    # If not in the major bodies catalog, try the minor bodies list, and if
    # still not found, throw an error.
    try:
        ephem_data = query_obj.ephemerides(extra_precision=True)
    except KeyError:
        # This is a fix for an astroquery + JPL-Horizons bug, that's related to
        # API change on JPL's side. In this case, the source is identified, but
        # astroquery can't correctly parse the return message from JPL-Horizons.
        # See astroquery issue #2169.
        ephem_data = query_obj.ephemerides(extra_precision=False)  # pragma: no cover
    except ValueError as err:
        query_obj._session.close()
        if "Unknown target" in str(err):
            raise ValueError(
                "Target ID is not recognized in either the small or major bodies "
                "catalogs, please consult the JPL-Horizons database for supported "
                "targets (https://ssd.jpl.nasa.gov/?horizons)."
            ) from err
        else:
            raise  # pragma: no cover
    # This is explicitly closed here to trap a bug that occassionally throws an
    # unexpected warning, see astroquery issue #1807
    query_obj._session.close()

    # Now that we have the ephem data, extract out the relevant data
    ephem_times = np.array(ephem_data["datetime_jd"])
    ephem_ra = np.array(ephem_data["RA"]) * (np.pi / 180.0)
    ephem_dec = np.array(ephem_data["DEC"]) * (np.pi / 180.0)
    ephem_dist = np.array(ephem_data["delta"])  # AU
    ephem_vel = np.array(ephem_data["delta_rate"])  # km/s

    return ephem_times, ephem_ra, ephem_dec, ephem_dist, ephem_vel


def interpolate_ephem(
    time_array, ephem_times, ephem_ra, ephem_dec, ephem_dist=None, ephem_vel=None
):
    """
    Interpolates ephemerides to give positions for requested times.

    This is a simple tool for calculated interpolated RA and Dec positions, as well
    as distances and velocities, for a given ephemeris. Under the hood, the method
    uses as cubic spline interpolation to calculate values at the requested times,
    provided that there are enough values to interpolate over to do so (requires
    >= 4 points), otherwise a linear interpolation is used.

    Parameters
    ----------
    time_array : array-like of floats
        Times to interpolate positions for, in UTC Julian days.
    ephem_times : array-like of floats
        Times in UTC Julian days which describe that match to the recorded postions
        of the target. Must be array-like, of shape (Npts,), where Npts is the number
        of ephemeris points.
    ephem_ra : array-like of floats
        Right ascencion of the target, at the times given in `ephem_times`. Units are
        in radians, must have the same shape as `ephem_times`.
    ephem_dec : array-like of floats
        Declination of the target, at the times given in `ephem_times`. Units are
        in radians, must have the same shape as `ephem_times`.
    ephem_dist : array-like of floats
        Distance of the target from the observer, at the times given in `ephem_times`.
        Optional argument, in units of parsecs. Must have the same shape as
        `ephem_times`.
    ephem_vel : array-like of floats
        Velocities of the target, at the times given in `ephem_times`. Optional
        argument, in units of km/sec. Must have the same shape as `ephem_times`.

    Returns
    -------
    ra_vals : ndarray of float
        Interpolated RA values, returned as an ndarray of floats with
        units of radians, and the same shape as `time_array`.
    dec_vals : ndarray of float
        Interpolated declination values, returned as an ndarray of floats with
        units of radians, and the same shape as `time_array`.
    dist_vals : None or ndarray of float
        If `ephem_dist` was provided, an ndarray of floats (with same shape as
        `time_array`) with the interpolated target distances, in units of parsecs.
        If `ephem_dist` was not provided, this returns as None.
    vel_vals : None or ndarray of float
        If `ephem_vals` was provided, an ndarray of floats (with same shape as
        `time_array`) with the interpolated target velocities, in units of km/sec.
        If `ephem_vals` was not provided, this returns as None.

    """
    # We're importing this here since it's only used for this one function
    from scipy.interpolate import interp1d

    ephem_shape = np.array(ephem_times).shape

    # Make sure that things look reasonable
    if np.array(ephem_ra).shape != ephem_shape:
        raise ValueError("ephem_ra must have the same shape as ephem_times.")

    if np.array(ephem_dec).shape != ephem_shape:
        raise ValueError("ephem_dec must have the same shape as ephem_times.")

    if (np.array(ephem_dist).shape != ephem_shape) and (ephem_dist is not None):
        raise ValueError("ephem_dist must have the same shape as ephem_times.")

    if (np.array(ephem_vel).shape != ephem_shape) and (ephem_vel is not None):
        raise ValueError("ephem_vel must have the same shape as ephem_times.")

    ra_vals = np.zeros_like(time_array, dtype=float)
    dec_vals = np.zeros_like(time_array, dtype=float)
    dist_vals = None if ephem_dist is None else np.zeros_like(time_array, dtype=float)
    vel_vals = None if ephem_vel is None else np.zeros_like(time_array, dtype=float)

    if len(ephem_times) == 1:
        ra_vals += ephem_ra
        dec_vals += ephem_dec
        if ephem_dist is not None:
            dist_vals += ephem_dist
        if ephem_vel is not None:
            vel_vals += ephem_vel
    else:
        if len(ephem_times) > 3:
            interp_kind = "cubic"
        else:
            interp_kind = "linear"

        # If we have values that line up perfectly, just use those directly
        select_mask = np.isin(time_array, ephem_times)
        if np.any(select_mask):
            time_select = time_array[select_mask]
            ra_vals[select_mask] = interp1d(ephem_times, ephem_ra, kind="nearest")(
                time_select
            )
            dec_vals[select_mask] = interp1d(ephem_times, ephem_dec, kind="nearest")(
                time_select
            )
            if ephem_dist is not None:
                dist_vals[select_mask] = interp1d(
                    ephem_times, ephem_dist, kind="nearest"
                )(time_select)
            if ephem_vel is not None:
                vel_vals[select_mask] = interp1d(
                    ephem_times, ephem_vel, kind="nearest"
                )(time_select)

        # If we have values lining up between grid points, use spline interpolation
        # to calculate their values
        select_mask = ~select_mask
        if np.any(select_mask):
            time_select = time_array[select_mask]
            ra_vals[select_mask] = interp1d(ephem_times, ephem_ra, kind=interp_kind)(
                time_select
            )
            dec_vals[select_mask] = interp1d(ephem_times, ephem_dec, kind=interp_kind)(
                time_select
            )
            if ephem_dist is not None:
                dist_vals[select_mask] = interp1d(
                    ephem_times, ephem_dist, kind=interp_kind
                )(time_select)
            if ephem_vel is not None:
                vel_vals[select_mask] = interp1d(
                    ephem_times, ephem_vel, kind=interp_kind
                )(time_select)

    return (ra_vals, dec_vals, dist_vals, vel_vals)


def calc_app_coords(
    lon_coord,
    lat_coord,
    coord_frame="icrs",
    coord_epoch=None,
    coord_times=None,
    coord_type="sidereal",
    time_array=None,
    lst_array=None,
    telescope_loc=None,
    telescope_frame="itrs",
    ellipsoid=None,
    pm_ra=None,
    pm_dec=None,
    vrad=None,
    dist=None,
):
    """
    Calculate apparent coordinates for several different coordinate types.

    This function calculates apparent positions at the current epoch.

    Parameters
    ----------
    lon_coord : float or ndarray of float
        Longitudinal (e.g., RA) coordinates, units of radians. Must match the same
        shape as lat_coord.
    lat_coord : float or ndarray of float
        Latitudinal (e.g., Dec) coordinates, units of radians. Must match the same
        shape as lon_coord.
    coord_frame : string
        The requested reference frame for the output coordinates, can be any frame
        that is presently supported by astropy.
    coord_epoch : float or str or Time object
        Epoch for ref_frame, nominally only used if converting to either the FK4 or
        FK5 frames, in units of fractional years. If provided as a float and the
        coord_frame is an FK4-variant, value will assumed to be given in Besselian
        years (i.e., 1950 would be 'B1950'), otherwise the year is assumed to be
        in Julian years.
    coord_times : float or ndarray of float
        Only used when `coord_type="ephem"`, the JD UTC time for each value of
        `lon_coord` and `lat_coord`. These values are used to interpolate `lon_coord`
        and `lat_coord` values to those times listed in `time_array`.
    coord_type : str
        Type of source to calculate coordinates for. Must be one of:
            "sidereal" (fixed RA/Dec),
            "ephem" (RA/Dec that moves with time),
            "driftscan" (fixed az/el position),
            "unprojected" (alias for "driftscan" with (Az, Alt) = (0 deg, 90 deg)).
    time_array : float or ndarray of float or Time object
        Times for which the apparent coordinates were calculated, in UTC JD. If more
        than a single element, must be the same shape as lon_coord and lat_coord if
        both of those are arrays (instead of single floats).
    telescope_loc : array-like of floats or EarthLocation or MoonLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, a lunarsky
        Moonlocation, or a tuple of shape (3,) containing (in order) the latitude,
        longitude, and altitude for a position on Earth in units of radians, radians,
        and meters, respectively.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    pm_ra : float or ndarray of float
        Proper motion in RA of the source, expressed in units of milliarcsec / year.
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Not
        required, motion is calculated relative to the value of `coord_epoch`.
    pm_dec : float or ndarray of float
        Proper motion in Dec of the source, expressed in units of milliarcsec / year.
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Not
        required, motion is calculated relative to the value of `coord_epoch`.
    vrad : float or ndarray of float
        Radial velocity of the source, expressed in units of km / sec. Can either be
        a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (namely ra_coord and dec_coord). Not required.
    dist : float or ndarray of float
        Distance of the source, expressed in milliarcseconds. Can either be a single
        float or array of shape (Ntimes,), although this must be consistent with other
        parameters (namely ra_coord and dec_coord). Not required.

    Returns
    -------
    app_ra : ndarray of floats
        Apparent right ascension coordinates, in units of radians.
    app_dec : ndarray of floats
        Apparent declination coordinates, in units of radians.
    """
    if isinstance(telescope_loc, EarthLocation) or (
        hasmoon and isinstance(telescope_loc, MoonLocation)
    ):
        site_loc = telescope_loc
        if hasmoon and isinstance(telescope_loc, MoonLocation):
            ellipsoid = MoonLocation.ellipsoid
    elif telescope_frame.upper() == "MCMF":
        if not hasmoon:
            raise ValueError(
                "Need to install `lunarsky` package to work with MCMF frame."
            )
        if ellipsoid is None:
            ellipsoid = "SPHERE"
        site_loc = MoonLocation.from_selenodetic(
            telescope_loc[1] * (180.0 / np.pi),
            telescope_loc[0] * (180.0 / np.pi),
            height=telescope_loc[2],
            ellipsoid=ellipsoid,
        )
    else:
        site_loc = EarthLocation.from_geodetic(
            telescope_loc[1] * (180.0 / np.pi),
            telescope_loc[0] * (180.0 / np.pi),
            height=telescope_loc[2],
        )

    if isinstance(site_loc, EarthLocation):
        frame = "itrs"
    else:
        frame = "mcmf"

    # Time objects and unique don't seem to play well together, so we break apart
    # their handling here
    if isinstance(time_array, Time):
        time_array = time_array.utc.jd

    unique_time_array, unique_mask = np.unique(time_array, return_index=True)

    if coord_type in ["driftscan", "unprojected"]:
        if lst_array is None:
            unique_lst = get_lst_for_time(
                unique_time_array,
                site_loc.lat.deg,
                site_loc.lon.deg,
                site_loc.height.to_value("m"),
                frame=frame,
                ellipsoid=ellipsoid,
            )
        else:
            unique_lst = lst_array[unique_mask]

    if coord_type == "sidereal":
        # If the coordinates are not in the ICRS frame, go ahead and transform them now
        if coord_frame != "icrs":
            icrs_ra, icrs_dec = transform_sidereal_coords(
                lon_coord,
                lat_coord,
                coord_frame,
                "icrs",
                in_coord_epoch=coord_epoch,
                time_array=unique_time_array,
            )
        else:
            icrs_ra = lon_coord
            icrs_dec = lat_coord
        unique_app_ra, unique_app_dec = transform_icrs_to_app(
            unique_time_array,
            icrs_ra,
            icrs_dec,
            site_loc,
            ellipsoid=ellipsoid,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            vrad=vrad,
            dist=dist,
        )

    elif coord_type == "driftscan":
        # Use the ERFA function ae2hd, which will do all the heavy
        # lifting for us
        unique_app_ha, unique_app_dec = erfa.ae2hd(
            lon_coord, lat_coord, site_loc.lat.rad
        )
        # The above returns HA/Dec, so we just need to rotate by
        # the LST to get back app RA and Dec
        unique_app_ra = np.mod(unique_app_ha + unique_lst, 2 * np.pi)
        unique_app_dec = unique_app_dec + np.zeros_like(unique_app_ra)
    elif coord_type == "ephem":
        interp_ra, interp_dec, _, _ = interpolate_ephem(
            unique_time_array, coord_times, lon_coord, lat_coord
        )
        if coord_frame != "icrs":
            icrs_ra, icrs_dec = transform_sidereal_coords(
                interp_ra,
                interp_dec,
                coord_frame,
                "icrs",
                in_coord_epoch=coord_epoch,
                time_array=unique_time_array,
            )
        else:
            icrs_ra = interp_ra
            icrs_dec = interp_dec
        # TODO: Vel and distance handling to be integrated here, once they are are
        # needed for velocity frame tracking
        unique_app_ra, unique_app_dec = transform_icrs_to_app(
            unique_time_array,
            icrs_ra,
            icrs_dec,
            site_loc,
            ellipsoid=ellipsoid,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
        )
    elif coord_type == "unprojected":
        # This is the easiest one - this is just supposed to be ENU, so set the
        # apparent coords to the current lst and telescope_lat.
        unique_app_ra = unique_lst.copy()
        unique_app_dec = np.zeros_like(unique_app_ra) + site_loc.lat.rad
    else:
        raise ValueError("Object type %s is not recognized." % coord_type)

    # Now that we've calculated all the unique values, time to backfill through the
    # "redundant" entries in the Nblt axis.
    app_ra = np.zeros(np.array(time_array).shape)
    app_dec = np.zeros(np.array(time_array).shape)

    for idx, unique_time in enumerate(unique_time_array):
        select_mask = time_array == unique_time
        app_ra[select_mask] = unique_app_ra[idx]
        app_dec[select_mask] = unique_app_dec[idx]

    return app_ra, app_dec


def calc_sidereal_coords(
    time_array,
    app_ra,
    app_dec,
    telescope_loc,
    coord_frame,
    telescope_frame="itrs",
    ellipsoid=None,
    coord_epoch=None,
):
    """
    Calculate sidereal coordinates given apparent coordinates.

    This function calculates coordinates in the requested frame (at a given epoch)
    from a set of apparent coordinates.

    Parameters
    ----------
    time_array : float or ndarray of float or Time object
        Times for which the apparent coordinates were calculated, in UTC JD. Must
        match the shape of app_ra and app_dec.
    app_ra : float or ndarray of float
        Array of apparent right ascension coordinates, units of radians. Must match
        the shape of time_array and app_dec.
    app_ra : float or ndarray of float
        Array of apparent right declination coordinates, units of radians. Must match
        the shape of time_array and app_dec.
    telescope_loc : tuple of floats or EarthLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, or a tuple
        of shape (3,) containing (in order) the latitude, longitude, and altitude,
        in units of radians, radians, and meters, respectively.
    coord_frame : string
        The requested reference frame for the output coordinates, can be any frame
        that is presently supported by astropy. Default is ICRS.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    coord_epoch : float or str or Time object
        Epoch for ref_frame, nominally only used if converting to either the FK4 or
        FK5 frames, in units of fractional years. If provided as a float and the
        ref_frame is an FK4-variant, value will assumed to be given in Besselian
        years (i.e., 1950 would be 'B1950'), otherwise the year is assumed to be
        in Julian years.

    Returns
    -------
    ref_ra : ndarray of floats
        Right ascension coordinates in the requested frame, in units of radians.
        Either shape (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    ref_dec : ndarray of floats
        Declination coordinates in the requested frame, in units of radians.
        Either shape (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    """
    # Check to make sure that we have a properly formatted epoch for our in-bound
    # coordinate frame
    epoch = None
    if isinstance(coord_epoch, str) or isinstance(coord_epoch, Time):
        # If its a string or a Time object, we don't need to do anything more
        epoch = Time(coord_epoch)
    elif coord_epoch is not None:
        if coord_frame.lower() in ["fk4", "fk4noeterms"]:
            epoch = Time(coord_epoch, format="byear")
        else:
            epoch = Time(coord_epoch, format="jyear")

    if telescope_frame == "mcmf" and ellipsoid is None:
        ellipsoid = "SPHERE"

    icrs_ra, icrs_dec = transform_app_to_icrs(
        time_array, app_ra, app_dec, telescope_loc, telescope_frame, ellipsoid=ellipsoid
    )

    if coord_frame == "icrs":
        ref_ra, ref_dec = (icrs_ra, icrs_dec)
    else:
        ref_ra, ref_dec = transform_sidereal_coords(
            icrs_ra,
            icrs_dec,
            "icrs",
            coord_frame,
            out_coord_epoch=epoch,
            time_array=time_array,
        )

    return ref_ra, ref_dec


def get_lst_for_time(
    jd_array=None,
    latitude=None,
    longitude=None,
    altitude=None,
    astrometry_library=None,
    frame="itrs",
    ellipsoid=None,
    telescope_loc=None,
):
    """
    Get the local apparent sidereal time for a set of jd times at an earth location.

    This function calculates the local apparent sidereal time (LAST), given a UTC time
    and a position on the Earth, using either the astropy or NOVAS libraries. It
    is important to note that there is an apporoximate 20 microsecond difference
    between the two methods, presumably due to small differences in the apparent
    reference frame. These differences will cancel out when calculating coordinates
    in the TOPO frame, so long as apparent coordinates are calculated using the
    same library (i.e., astropy or NOVAS). Failing to do so can introduce errors
    up to ~1 mas in the horizontal coordinate system (i.e., AltAz).

    Parameters
    ----------
    jd_array : ndarray of float
        JD times to get lsts for.
    latitude : float
        Latitude of location to get lst for in degrees. Cannot specify both `latitute`
        and `telescope_loc`.
    longitude : float
        Longitude of location to get lst for in degrees. Cannot specify both `longitude`
        and `telescope_loc`.
    altitude : float
        Altitude of location to get lst for in meters. Cannot specify both `altitude`
        and `telescope_loc`.
    astrometry_library : str
        Library used for running the LST calculations. Allowed options are 'erfa'
        (which uses the pyERFA), 'novas' (which uses the python-novas library),
        and 'astropy' (which uses the astropy utilities). Default is erfa unless
        the telescope_location is a MoonLocation object, in which case the default is
        astropy.
    frame : str
        Reference frame for latitude/longitude/altitude.
        Options are itrs (default) or mcmf.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    telescope_loc : tuple or EarthLocation or MoonLocation
        Alternative way of specifying telescope lat/lon/alt, either as a 3-element tuple
        or as an astropy EarthLocation (or lunarsky MoonLocation). Cannot supply both
        `telescope_loc` and `latitute`, `longitude`, or `altitude`.

    Returns
    -------
    ndarray of float
        LASTs in radians corresponding to the jd_array.

    """
    if astrometry_library is None:
        if frame == "itrs":
            astrometry_library = "erfa"
        else:
            astrometry_library = "astropy"

    if astrometry_library not in ["erfa", "astropy", "novas"]:
        raise ValueError(
            "Requested coordinate transformation library is not supported, please "
            "select either 'erfa' or 'astropy' for astrometry_library."
        )

    if isinstance(jd_array, np.ndarray):
        lst_array = np.zeros_like(jd_array)
        if lst_array.ndim == 0:
            lst_array = lst_array.reshape(1)
    else:
        lst_array = np.zeros(1)

    jd, reverse_inds = np.unique(jd_array, return_inverse=True)

    site_loc = None
    if telescope_loc is not None:
        if not all(item is None for item in [latitude, longitude, altitude]):
            raise ValueError(
                "Cannot set both telescope_loc and latitude/longitude/altitude"
            )
        if isinstance(telescope_loc, EarthLocation) or (
            hasmoon and isinstance(telescope_loc, MoonLocation)
        ):
            site_loc = telescope_loc
        else:
            latitude, longitude, altitude = telescope_loc

    if site_loc is None:
        if frame.upper() == "MCMF":
            if not hasmoon:
                raise ValueError(
                    "Need to install `lunarsky` package to work with MCMF frame."
                )
            if ellipsoid is None:
                ellipsoid = "SPHERE"

            site_loc = MoonLocation.from_selenodetic(
                Angle(longitude, unit="deg"),
                Angle(latitude, unit="deg"),
                altitude,
                ellipsoid=ellipsoid,
            )
        else:
            site_loc = EarthLocation.from_geodetic(
                Angle(longitude, unit="deg"),
                Angle(latitude, unit="deg"),
                height=altitude,
            )

    if isinstance(site_loc, EarthLocation):
        TimeClass = Time
    else:
        if not astrometry_library == "astropy":
            raise NotImplementedError(
                "The MCMF frame is only supported with the 'astropy' astrometry library"
            )
        TimeClass = LTime

    times = TimeClass(jd, format="jd", scale="utc", location=site_loc)

    if iers.conf.auto_max_age is None:  # pragma: no cover
        delta, status = times.get_delta_ut1_utc(return_status=True)
        if np.any(
            np.isin(status, (iers.TIME_BEFORE_IERS_RANGE, iers.TIME_BEYOND_IERS_RANGE))
        ):
            warnings.warn(
                "time is out of IERS range, setting delta ut1 utc to extrapolated value"
            )
            times.delta_ut1_utc = delta
    if astrometry_library == "erfa":
        # This appears to be what astropy is using under the hood,
        # so it _should_ be totally consistent.
        gast_array = erfa.gst06a(
            times.ut1.jd1, times.ut1.jd2, times.tt.jd1, times.tt.jd2
        )

        # Technically one should correct for the polar wobble here, but the differences
        # along the equitorial are miniscule -- of order 10s of nanoradians, well below
        # the promised accuracy of IERS -- and rotation matricies can be expensive.
        # We do want to correct though for for secular polar drift (s'/TIO locator),
        # which nudges the Earth rotation angle of order 47 uas per century.
        sp = erfa.sp00(times.tt.jd1, times.tt.jd2)

        lst_array = np.mod(gast_array + sp + site_loc.lon.rad, 2.0 * np.pi)[
            reverse_inds
        ]
    elif astrometry_library == "astropy":
        lst_array = times.sidereal_time("apparent").radian
        if lst_array.ndim == 0:
            lst_array = lst_array.reshape(1)
        lst_array = lst_array[reverse_inds]
    elif astrometry_library == "novas":
        # Import the NOVAS library only if it's needed/available.
        try:
            import novas_de405  # noqa
            from novas import compat as novas
            from novas.compat import eph_manager
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "novas and/or novas_de405 are not installed but is required for "
                "NOVAS functionality"
            ) from e

        jd_start, jd_end, number = eph_manager.ephem_open()

        tt_time_array = times.tt.value
        ut1_high_time_array = times.ut1.jd1
        ut1_low_time_array = times.ut1.jd2
        full_ut1_time_array = ut1_high_time_array + ut1_low_time_array
        polar_motion_data = iers.earth_orientation_table.get()

        delta_x_array = np.interp(
            times.mjd,
            polar_motion_data["MJD"].value,
            polar_motion_data["dX_2000A_B"].value,
            left=0.0,
            right=0.0,
        )

        delta_y_array = np.interp(
            times.mjd,
            polar_motion_data["MJD"].value,
            polar_motion_data["dY_2000A_B"].value,
            left=0.0,
            right=0.0,
        )

        # Catch the case where we don't have CIP delta values yet (they don't typically
        # have predictive values like the polar motion does)
        delta_x_array[np.isnan(delta_x_array)] = 0.0
        delta_y_array[np.isnan(delta_y_array)] = 0.0

        for idx in range(len(times)):
            novas.cel_pole(
                tt_time_array[idx], 2, delta_x_array[idx], delta_y_array[idx]
            )
            # The NOVAS routine will return Greenwich Apparent Sidereal Time (GAST),
            # in units of hours
            lst_array[reverse_inds == idx] = novas.sidereal_time(
                ut1_high_time_array[idx],
                ut1_low_time_array[idx],
                (tt_time_array[idx] - full_ut1_time_array[idx]) * 86400.0,
            )

        # Add the telescope lon to convert from GAST to LAST (local)
        lst_array = np.mod(lst_array + (longitude / 15.0), 24.0)

        # Convert from hours back to rad
        lst_array *= np.pi / 12.0

    return lst_array


def check_lsts_against_times(
    *,
    jd_array,
    lst_array,
    latitude,
    longitude,
    altitude,
    lst_tols,
    frame="itrs",
    ellipsoid=None,
):
    """
    Check that LSTs are consistent with the time_array and telescope location.

    This just calls `get_lst_for_time`, compares that result to the `lst_array`
    and warns if they are not within the tolerances specified by `lst_tols`.

    Parameters
    ----------
    jd_array : ndarray of float
        JD times to get lsts for.
    lst_array : ndarray of float
        LSTs to check to see if they match the jd_array at the location.
    latitude : float
        Latitude of location to check the lst for in degrees.
    longitude : float
        Longitude of location to check the lst for in degrees.
    altitude : float
        Altitude of location to check the lst for in meters.
    lst_tops : tuple of float
        A length 2 tuple giving the (relative, absolute) tolerances to check the
        LST agreement to. These are passed directly to numpy.allclose.
    frame : str
        Reference frame for latitude/longitude/altitude.
        Options are itrs (default) or mcmf.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE", "GSFC",
        "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.

    Returns
    -------
    None

    Warns
    -----
    If the `lst_array` does not match the calculated LSTs to the lst_tols.

    """
    if frame == "mcmf" and ellipsoid is None:
        ellipsoid = "SPHERE"

    # Don't worry about passing the astrometry library because we test that they agree
    # to better than our standard lst tolerances.
    lsts = get_lst_for_time(
        jd_array=jd_array,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        frame=frame,
        ellipsoid=ellipsoid,
    )

    if not np.allclose(lst_array, lsts, rtol=lst_tols[0], atol=lst_tols[1]):
        warnings.warn(
            "The lst_array is not self-consistent with the time_array and "
            "telescope location. Consider recomputing with the "
            "`set_lsts_from_time_array` method."
        )


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
            telescope_loc.x.to("m").value,
            telescope_loc.y.to("m").value,
            telescope_loc.z.to("m").value,
        )
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


def uvw_track_generator(
    *,
    lon_coord=None,
    lat_coord=None,
    coord_frame="icrs",
    coord_epoch=None,
    coord_type="sidereal",
    time_array=None,
    telescope_loc=None,
    telescope_frame="itrs",
    ellipsoid=None,
    antenna_positions=None,
    antenna_numbers=None,
    ant_1_array=None,
    ant_2_array=None,
    uvw_array=None,
    force_postive_u=False,
):
    """
    Calculate uvw coordinates (among other values) for a given position on the sky.

    This function is meant to be a user-friendly wrapper around several pieces of code
    for effectively simulating a track.

    Parameters
    ----------
    lon_coord : float or ndarray of float
        Longitudinal (e.g., RA) coordinates, units of radians. Must match the same
        shape as lat_coord.
    lat_coord : float or ndarray of float
        Latitudinal (e.g., Dec) coordinates, units of radians. Must match the same
        shape as lon_coord.
    coord_frame : string
        The requested reference frame for the output coordinates, can be any frame
        that is presently supported by astropy.
    coord_epoch : float or str or Time object, optional
        Epoch for ref_frame, nominally only used if converting to either the FK4 or
        FK5 frames, in units of fractional years. If provided as a float and the
        ref_frame is an FK4-variant, value will assumed to be given in Besselian
        years (i.e., 1950 would be 'B1950'), otherwise the year is assumed to be
        in Julian years.
    coord_type : str
        Type of source to calculate coordinates for. Must be one of:
            "sidereal" (fixed RA/Dec),
            "ephem" (RA/Dec that moves with time),
            "driftscan" (fixed az/el position),
            "unprojected" (alias for "driftscan" with (Az, Alt) = (0 deg, 90 deg)).
    time_array : ndarray of float or Time object
        Times for which the apparent coordinates were calculated, in UTC JD. Must
        match the shape of lon_coord and lat_coord.
    telescope_loc : array-like of floats or EarthLocation or MoonLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, a lunarsky
        Moonlocation, or a tuple of shape (3,) containing (in order) the latitude,
        longitude, and altitude for a position on Earth in units of degrees, degrees,
        and meters, respectively.
    telescope_frame : str, optional
        Reference frame for latitude/longitude/altitude. Options are itrs (default) or
        mcmf. Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    antenna_positions : ndarray of float
        List of antenna positions relative to array center in ECEF coordinates,
        required if not providing `uvw_array`. Shape is (Nants, 3).
    antenna_numbers: ndarray of int, optional
        List of antenna numbers, ordered in the same way as `antenna_positions` (e.g.,
        `antenna_numbers[0]` should given the number of antenna that resides at ECEF
        position given by `antenna_positions[0]`). Shape is (Nants,), requred if
        supplying ant_1_array and ant_2_array.
    ant_1_array : ndarray of int, optional
        Antenna number of the first antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,). If not supplied, then
        the method will automatically fill in ant_1_array with all unique antenna
        pairings for each time/position.
    ant_2_array : ndarray of int, optional
        Antenna number of the second antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,). If not supplied, then
        the method will automatically fill in ant_2_array with all unique antenna
        pairings for each time/position.
    uvw_array : ndarray of float, optional
        Array of baseline coordinates (in ENU), required if not deriving new coordinates
        from antenna positions. Setting this value will will cause antenna positions to
        be ignored. Shape is (Nblts, 3).
    force_positive_u : bool, optional
        If set to true, then forces the conjugation of each individual baseline to be
        set such that the uvw coordinates land on the positive-u side of the uv-plane.
        Default is False.

    Returns
    -------
    obs_dict : dict
        Dictionary containing the results of the simulation, which includes:
            "uvw" the uvw-coordinates (meters),
            "app_ra" apparent RA of the sources (radians),
            "app_dec"  apparent Dec of the sources (radians),
            "frame_pa"  ngle between apparent north and `coord_frame` north (radians),
            "lst" local apparent sidereal time (radians),
            "site_loc" EarthLocation or MoonLocation for the telescope site.
    """
    if isinstance(telescope_loc, EarthLocation) or (
        hasmoon and isinstance(telescope_loc, MoonLocation)
    ):
        site_loc = telescope_loc
    elif telescope_frame.upper() == "MCMF":
        if not hasmoon:
            raise ValueError(
                "Need to install `lunarsky` package to work with MCMF frame."
            )
        if ellipsoid is None:
            ellipsoid = "SPHERE"

        site_loc = MoonLocation.from_selenodetic(
            Angle(telescope_loc[1], unit="deg"),
            Angle(telescope_loc[0], unit="deg"),
            telescope_loc[2],
            ellipsoid=ellipsoid,
        )
    else:
        site_loc = EarthLocation.from_geodetic(
            Angle(telescope_loc[1], unit="deg"),
            Angle(telescope_loc[0], unit="deg"),
            height=telescope_loc[2],
        )

    if not isinstance(lon_coord, np.ndarray):
        lon_coord = np.array(lon_coord)
    if not isinstance(lat_coord, np.ndarray):
        lat_coord = np.array(lat_coord)
    if not isinstance(time_array, np.ndarray):
        time_array = np.array(time_array)

    if lon_coord.ndim == 0:
        lon_coord = lon_coord.reshape(1)
    if lat_coord.ndim == 0:
        lat_coord = lat_coord.reshape(1)
    if time_array.ndim == 0:
        time_array = time_array.reshape(1)

    Ntimes = len(time_array)
    if uvw_array is None:
        if all(item is None for item in [antenna_numbers, ant_1_array, ant_2_array]):
            antenna_numbers = np.arange(1, 1 + len(antenna_positions))
            ant_1_array = []
            ant_2_array = []
            for idx in range(len(antenna_positions)):
                for jdx in range(idx + 1, len(antenna_positions)):
                    ant_1_array.append(idx + 1)
                    ant_2_array.append(jdx + 1)

            Nbase = len(ant_1_array)

            ant_1_array = np.tile(ant_1_array, Ntimes)
            ant_2_array = np.tile(ant_2_array, Ntimes)
            if len(lon_coord) == len(time_array):
                lon_coord = np.repeat(lon_coord, Nbase)
                lat_coord = np.repeat(lat_coord, Nbase)

            time_array = np.repeat(time_array, Nbase)

    lst_array = get_lst_for_time(
        jd_array=time_array, telescope_loc=site_loc, frame=telescope_frame
    )
    app_ra, app_dec = calc_app_coords(
        lon_coord=lon_coord,
        lat_coord=lat_coord,
        coord_frame=coord_frame,
        coord_type=coord_type,
        time_array=time_array,
        lst_array=lst_array,
        telescope_loc=site_loc,
    )

    frame_pa = calc_frame_pos_angle(
        time_array, app_ra, app_dec, site_loc, coord_frame, ref_epoch=coord_epoch
    )

    uvws = calc_uvw(
        app_ra=app_ra,
        app_dec=app_dec,
        frame_pa=frame_pa,
        lst_array=lst_array,
        antenna_positions=antenna_positions,
        antenna_numbers=antenna_numbers,
        ant_1_array=ant_1_array,
        ant_2_array=ant_2_array,
        telescope_lon=site_loc.lon.rad,
        telescope_lat=site_loc.lat.rad,
        uvw_array=uvw_array,
        use_ant_pos=(uvw_array is None),
        from_enu=(uvw_array is not None),
    )

    if force_postive_u:
        mask = (uvws[:, 0] < 0.0) | ((uvws[:, 0] == 0.0) & (uvws[:, 1] < 0.0))
        uvws[mask, :] *= -1.0

    return {
        "uvw": uvws,
        "app_ra": app_ra,
        "app_dec": app_dec,
        "frame_pa": frame_pa,
        "lst": lst_array,
        "site_loc": site_loc,
    }


def _adj_list(vecs, tol, n_blocks=None):
    """Identify neighbors of each vec in vecs, to distance tol."""
    n_items = len(vecs)
    max_items = 2**10  # Max array size used is max_items**2. Avoid using > 1 GiB

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


def find_clusters_grid(baselines, baseline_vecs, tol=1.0):
    """
    Find redundant groups using a gridding algorithm developed by the HERA team.

    This is essentially a gridding approach, but it only keeps track of the grid
    points that have baselines assigned to them. It iterates through the
    baselines and assigns each baseline to a an existing group if it is within
    a grid spacing or makes a new group if there is no group. The location of
    the group is the baseline vector of the first baseline assigned to it, rounded
    to the grid spacing, so the resulting assigned grid point can depend on the
    order in which baseline vectors are passed to it. It is possible for a baseline
    to be assigned to a group that is up to but strictly less than 4 times the
    grid spacing from its true location, so we use a grid a factor of 4 smaller
    than the passed tolerance (`tol`). This method is quite robust for regular
    arrays if the tolerance is properly specified, but may not behave predictably
    for highly non-redundant arrays.

    Parameters
    ----------
    baselines : array_like of int
        Baseline numbers, shape (Nbls,)
    baseline_vecs : array_like of float
        Baseline vectors in meters, shape (Nbls, 3).
    tol : float
        Absolute tolerance of redundancy, in meters.

    Returns
    -------
    baseline_groups : list of lists of int
        list of lists of redundant baseline numbers
    baseline_ind_conj : list of int
        List of baselines that are redundant when reversed. Only returned if
        include_conjugates is True

    """
    bl_gps = {}
    # reduce the grid size to ensure baselines won't be assigned to a group
    # more than the tol away from their location. The factor of 4 is a personal
    # communication from Josh Dillon who developed this algorithm.
    grid_size = tol / 4.0

    p_or_m = (0, -1, 1)
    epsilons = [[dx, dy, dz] for dx in p_or_m for dy in p_or_m for dz in p_or_m]

    def check_neighbors(delta):
        # Check to make sure bl_gps doesn't have the key plus or minus rounding error
        for epsilon in epsilons:
            newKey = (
                delta[0] + epsilon[0],
                delta[1] + epsilon[1],
                delta[2] + epsilon[2],
            )
            if newKey in bl_gps:
                return newKey
        return

    baseline_ind_conj = []
    for bl_i, bl in enumerate(baselines):
        delta = tuple(np.round(baseline_vecs[bl_i] / grid_size).astype(int))
        new_key = check_neighbors(delta)
        if new_key is not None:
            # this has a match
            bl_gps[new_key].append(bl)
        else:
            # this is a new group
            bl_gps[delta] = [bl]

    bl_list = [sorted(gv) for gv in bl_gps.values()]

    return bl_list, baseline_ind_conj


def get_baseline_redundancies(
    baselines, baseline_vecs, tol=1.0, include_conjugates=False, use_grid_alg=None
):
    """
    Find redundant baseline groups.

    Parameters
    ----------
    baselines : array_like of int
        Baseline numbers, shape (Nbls,)
    baseline_vecs : array_like of float
        Baseline vectors in meters, shape (Nbls, 3).
    tol : float
        Absolute tolerance of redundancy, in meters.
    include_conjugates : bool
        Option to include baselines that are redundant when flipped.
    use_grid_alg : bool
        Option to use the gridding based algorithm (developed by the HERA team)
        to find redundancies rather than the older clustering algorithm.

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
        include_conjugates is True

    """
    if use_grid_alg is None:
        # This was added in v2.4.2 (Feb 2024). It should go away at some point.
        # Normally it would be in v2.6 or later, but if v3.0 comes out
        # very soon we could consider delaying the removal of this until v3.1
        warnings.warn(
            "The use_grid_alg parameter is not set. Defaulting to True to "
            "use the new gridding based algorithm (developed by the HERA team) "
            "rather than the older clustering based algorithm. This is change "
            "to the default, to use the clustering algorithm set use_grid_alg=False."
        )
        use_grid_alg = True

    Nbls = baselines.shape[0]

    if not baseline_vecs.shape == (Nbls, 3):
        raise ValueError("Baseline vectors must be shape (Nbls, 3)")

    baseline_vecs = copy.copy(baseline_vecs)  # Protect the vectors passed in.

    if include_conjugates:
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
            baselines,
            baseline_vecs,
            tol=tol,
            include_conjugates=False,
            use_grid_alg=use_grid_alg,
        )
        return bl_gps, vec_bin_centers, lens, baseline_ind_conj

    if use_grid_alg:
        output = find_clusters_grid(baselines, baseline_vecs, tol=1.0)
        bl_gps, baseline_ind_conj = output
    else:
        try:
            bl_gps = find_clusters(baselines, baseline_vecs, tol, strict=True)
        except ValueError as exc:
            raise ValueError(
                "Some baselines are falling into multiple redundant groups. "
                "Lower the tolerance to resolve ambiguity or use the gridding "
                "based algorithm (developed by the HERA team) to find redundancies "
                "by setting use_grid_alg=True."
            ) from exc

    n_unique = len(bl_gps)
    vec_bin_centers = np.zeros((n_unique, 3))
    for gi, gp in enumerate(bl_gps):
        inds = [np.where(i == baselines)[0] for i in gp]
        vec_bin_centers[gi] = np.mean(baseline_vecs[inds, :], axis=0)

    lens = np.sqrt(np.sum(vec_bin_centers**2, axis=1))
    return bl_gps, vec_bin_centers, lens


def get_antenna_redundancies(
    antenna_numbers, antenna_positions, tol=1.0, include_autos=False, use_grid_alg=None
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
    use_grid_alg : bool
        Option to use the gridding based algorithm (developed by the HERA team)
        to find redundancies rather than the older clustering algorithm.

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
    if use_grid_alg is None:
        # This was added in v2.4.2 (Feb 2024). It should go away at some point.
        # Normally it would be in v2.6 or later, but if v3.0 comes out
        # very soon we could consider delaying the removal of this until v3.1
        warnings.warn(
            "The use_grid_alg parameter is not set. Defaulting to True to "
            "use the new gridding based algorithm (developed by the HERA team) "
            "rather than the older clustering based algorithm. This is change "
            "to the default, to use the clustering algorithm set use_grid_alg=False."
        )
        use_grid_alg = True

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
        bls, bl_vecs, tol=tol, include_conjugates=True, use_grid_alg=use_grid_alg
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
        weights_square = weights**2
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
    except KeyError as err:
        raise ValueError(
            "Collapse algorithm must be one of: "
            + ", ".join(collapse_dict.keys())
            + "."
        ) from err
    return out


def uvcalibrate(
    uvdata,
    uvcal,
    inplace=True,
    prop_flags=True,
    Dterm_cal=False,
    flip_gain_conj=False,
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
    Dterm_cal : bool, optional
        Calibrate the off-diagonal terms in the Jones matrix if present
        in uvcal. Default is False. Currently not implemented.
    flip_gain_conj : bool, optional
        This function uses the UVData ant_1_array and ant_2_array to specify the
        antennas in the UVCal object. By default, the conjugation convention, which
        follows the UVData convention (i.e. ant2 - ant1), is that the applied
        gain = ant1_gain * conjugate(ant2_gain). If the other convention is required,
        set flip_gain_conj=True.
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
    if uvcal.cal_type == "gain" and uvcal.wide_band:
        raise ValueError(
            "uvcalibrate currently does not support wide-band calibrations"
        )
    if uvcal.cal_type == "delay" and uvcal.Nspws > 1:
        # To fix this, need to make UVCal.convert_to_gain support multiple spws
        raise ValueError(
            "uvcalibrate currently does not support multi spectral window delay "
            "calibrations"
        )

    if not inplace:
        uvdata = uvdata.copy()

    # check both objects
    uvdata.check()
    uvcal.check()

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
                    raise ValueError(
                        "All antenna names with data on UVData are missing "
                        "on UVCal. To continue with calibration "
                        "(and flag all the data), set ant_check=False."
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
                    raise ValueError(
                        f"Antennas {name_missing} have data on UVData but "
                        "are missing on UVCal. To continue calibration and "
                        "flag the data from missing antennas, set ant_check=False."
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
                    raise ValueError(
                        f"Time {this_time} exists on UVData but not on UVCal."
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
                raise ValueError(
                    "Times do not match between UVData and UVCal. "
                    "Set time_check=False to apply calibration anyway."
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
                raise ValueError(
                    "Times do not match between UVData and UVCal. "
                    "Set time_check=False to apply calibration anyway. "
                )

    downselect_cal_freq = False
    if uvcal.freq_array is not None:
        if uvdata.future_array_shapes:
            uvdata_freq_arr_use = uvdata.freq_array
        else:
            uvdata_freq_arr_use = uvdata.freq_array[0, :]
        if uvcal.future_array_shapes:
            uvcal_freq_arr_use = uvcal.freq_array
        else:
            uvcal_freq_arr_use = uvcal.freq_array[0, :]
        try:
            freq_arr_match = np.allclose(
                np.sort(uvcal_freq_arr_use),
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
                    raise ValueError(
                        f"Frequency {this_freq} exists on UVData but not on UVCal."
                    )
            if len(uvcal_freqs_to_keep) < uvcal.Nfreqs:
                downselect_cal_freq = True

    # check if uvdata.x_orientation isn't set (it's required for uvcal)
    uvd_x = uvdata.x_orientation
    if uvd_x is None:
        # use the uvcal x_orientation throughout
        uvd_x = uvcal.x_orientation
        warnings.warn(
            "UVData object does not have `x_orientation` specified but UVCal does. "
            "Matching based on `x` and `y` only "
        )

    uvdata_pol_strs = polnum2str(uvdata.polarization_array, x_orientation=uvd_x)
    uvcal_pol_strs = jnum2str(uvcal.jones_array, x_orientation=uvcal.x_orientation)
    uvdata_feed_pols = {
        feed for pol in uvdata_pol_strs for feed in POL_TO_FEED_DICT[pol]
    }
    for feed in uvdata_feed_pols:
        # get diagonal jones str
        jones_str = parse_jpolstr(feed, x_orientation=uvcal.x_orientation)
        if jones_str not in uvcal_pol_strs:
            raise ValueError(
                f"Feed polarization {feed} exists on UVData but not on UVCal. "
            )

    # downselect UVCal times, frequencies
    if downselect_cal_freq or downselect_cal_times:
        if not downselect_cal_times:
            uvcal_times_to_keep = None
        elif not downselect_cal_freq:
            uvcal_freqs_to_keep = None

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
        if uvdata.future_array_shapes:
            freq_array_use = uvdata.freq_array
        else:
            freq_array_use = uvdata.freq_array[0, :]
        if uvcal.future_array_shapes == uvdata.future_array_shapes:
            channel_width = uvdata.channel_width
        elif uvcal.future_array_shapes:
            channel_width = np.zeros(uvdata.Nfreqs, dtype=float) + uvdata.channel_width
        else:
            channel_width = uvdata.channel_width[0]
        uvcal_use.convert_to_gain(
            delay_convention=delay_convention,
            freq_array=freq_array_use,
            channel_width=channel_width,
        )

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
                np.abs(uvdata.polarization_array - polstr2num(key[2], uvd_x))
            )

            # try to get gains for each antenna
            ant1_num = key[0]
            ant2_num = key[1]

            feed1, feed2 = POL_TO_FEED_DICT[key[2]]
            try:
                uvcal_ant1_num = uvcal_ant_dict[uvdata_ant_dict[ant1_num]]
            except KeyError:
                uvcal_ant1_num = None
            try:
                uvcal_ant2_num = uvcal_ant_dict[uvdata_ant_dict[ant2_num]]
            except KeyError:
                uvcal_ant2_num = None

            uvcal_key1 = (uvcal_ant1_num, feed1)
            uvcal_key2 = (uvcal_ant2_num, feed2)
            if (uvcal_ant1_num is None or uvcal_ant2_num is None) or not (
                uvcal_use._has_key(*uvcal_key1) and uvcal_use._has_key(*uvcal_key2)
            ):
                if uvdata.future_array_shapes:
                    uvdata.flag_array[blt_inds, :, pol_ind] = True
                else:
                    uvdata.flag_array[blt_inds, 0, :, pol_ind] = True
                continue
            if flip_gain_conj:
                gain = (
                    np.conj(uvcal_use.get_gains(uvcal_key1))
                    * uvcal_use.get_gains(uvcal_key2)
                ).T  # tranpose to match uvdata shape
            else:
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
        uvdata.vis_units = "uncalib"
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
        if uvd.future_array_shapes == uvf.future_array_shapes:
            uvd.flag_array[uvd_ap_inds] += uvf.flag_array[uvf_ap_inds]
        elif uvd.future_array_shapes:
            uvd.flag_array[uvd_ap_inds] += uvf.flag_array[uvf_ap_inds, 0, :, :]
        else:
            uvd.flag_array[uvd_ap_inds, 0, :, :] += uvf.flag_array[uvf_ap_inds]

    uvd.history += "\nFlagged with pyuvdata.utils.apply_uvflags."

    if not inplace:
        return uvd


def parse_ants(uv, ant_str, print_toggle=False, x_orientation=None):
    """
    Get antpair and polarization from parsing an aipy-style ant string.

    Used to support the select function. Generates two lists of antenna pair
    tuples and polarization indices based on parsing of the string ant_str.
    If no valid polarizations (pseudo-Stokes params, or combinations of [lr]
    or [xy]) or antenna numbers are found in ant_str, ant_pairs_nums and
    polarizations are returned as None.

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
                if isinstance(ant_i, str) and ant_i.startswith("-"):
                    ant_i = ant_i[1:]  # nibble the - off the string
                    include_i = False

                for ant_j in ant_j_list:
                    include_j = True
                    if isinstance(ant_j, str) and ant_j.startswith("-"):
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
                                            polstr2num(pol, x_orientation=x_orientation)
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
            "Warning: Polarization {p} is not present in the polarization_array".format(
                p=(",").join(warned_pols).upper()
            )
        )

    return ant_pairs_nums, polarizations


def _combine_filenames(filename1, filename2):
    """Combine the filename attribute from multiple UVBase objects.

    The 4 cases are:
    1. `filename1` has been set, `filename2` has not
    2. `filename1` has not been set, `filename2` has
    3. `filename1` and `filename2` both have been set
    4. `filename1` and `filename2` both have not been set
    In case (1), we do not want to update the attribute, because it is
    already set correctly. In case (2), we want to replace `filename1`
    with the value from `filename2. In case (3), we want to take the union of
    the sets of the filenames. In case (4), we want the filename attribute
    to still be `None`.

    Parameters
    ----------
    filename1 : list of str or None
        The list of filenames for the first UVBase object. If it is not set, it
        should be `None`.
    filename2 : list of str or None
        The list of filenames for the second UVData object. If it is not set, it
        should be `None`.

    Returns
    -------
    combined_filenames : list of str or None
        The combined list, with potentially duplicate entries removed.
    """
    combined_filenames = filename1
    if filename1 is not None:
        if filename2 is not None:
            combined_filenames = sorted(set(filename1).union(set(filename2)))
    elif filename2 is not None:
        combined_filenames = filename2

    return combined_filenames


def _get_dset_shape(dset, indices):
    """
    Given a 3-tuple of indices, determine the indexed array shape.

    Parameters
    ----------
    dset : numpy array or h5py dataset
        A numpy array or a reference to an HDF5 dataset on disk. Requires the
        `dset.shape` attribute exists and returns a tuple.
    indices : tuple
        A 3-tuple with the indices to extract along each dimension of dset.
        Each element should contain a list of indices, a slice element,
        or a list of slice elements that will be concatenated after slicing.
        For data arrays with 4 dimensions, the second dimension (the old spw axis)
        should not be included because it can only be length one.

    Returns
    -------
    tuple
        a 3- or 4-tuple with the shape of the indexed array
    tuple
        a 3- or 4-tuple with indices used (will be different than input if dset has
        4 dimensions)
    """
    dset_shape = list(dset.shape)
    if len(dset_shape) == 4 and len(indices) == 3:
        indices = (indices[0], np.s_[:], indices[1], indices[2])

    for i, inds in enumerate(indices):
        # check for integer
        if isinstance(inds, (int, np.integer)):
            dset_shape[i] = 1
        # check for slice object
        if isinstance(inds, slice):
            dset_shape[i] = _get_slice_len(inds, dset_shape[i])
        # check for list
        if isinstance(inds, list):
            # check for list of integers
            if isinstance(inds[0], (int, np.integer)):
                dset_shape[i] = len(inds)
            elif isinstance(inds[0], slice):
                dset_shape[i] = sum((_get_slice_len(s, dset_shape[i]) for s in inds))

    return dset_shape, indices


def _convert_to_slices(
    indices, max_nslice_frac=0.1, max_nslice=None, return_index_on_fail=False
):
    """
    Convert list of indices to a list of slices.

    Parameters
    ----------
    indices : list
        A 1D list of integers for array indexing (boolean ndarrays are also supported).
    max_nslice_frac : float
        A float from 0 -- 1. If the number of slices
        needed to represent input 'indices' divided by len(indices)
        exceeds this fraction, then we determine that we cannot
        easily represent 'indices' with a list of slices.
    max_nslice : int
        Optional argument, defines the maximum number of slices for determining if
        `indices` can be easily represented with a list of slices. If set, then
        the argument supplied to `max_nslice_frac` is ignored.
    return_index_on_fail : bool
        If set to True and the list of input indexes cannot easily be respresented by
        a list of slices (as defined by `max_nslice` or `max_nslice_frac`), then return
        the input list of index values instead of a list of suboptimal slices.

    Returns
    -------
    slice_list : list
        Nominally the list of slice objects used to represent indices. However, if
        `return_index_on_fail=True` and input indexes cannot easily be respresented,
        return a 1-element list containing the input for `indices`.
    check : bool
        If True, indices is easily represented by slices
        (`max_nslice_frac` or `max_nslice` conditions met), otherwise False.

    Notes
    -----
    Example:
        if: indices = [1, 2, 3, 4, 10, 11, 12, 13, 14]
        then: slices = [slice(1, 5, 1), slice(11, 15, 1)]
    """
    # check for already a slice or a single index position
    if isinstance(indices, slice):
        return [indices], True
    if isinstance(indices, (int, np.integer)):
        return [slice(indices, indices + 1, 1)], True

    # check for boolean index
    if isinstance(indices, np.ndarray) and (indices.dtype == bool):
        eval_ind = np.where(indices)[0]
    else:
        eval_ind = indices
    # assert indices is longer than 2, or return trivial solutions
    if len(eval_ind) == 0:
        return [slice(0, 0, 0)], False
    if len(eval_ind) <= 2:
        return [
            slice(eval_ind[0], eval_ind[-1] + 1, max(eval_ind[-1] - eval_ind[0], 1))
        ], True

    # Catch the simplest case of "give me a single slice or exit"
    if (max_nslice == 1) and return_index_on_fail:
        step = eval_ind[1] - eval_ind[0]
        if all(np.diff(eval_ind) == step):
            return [slice(eval_ind[0], eval_ind[-1] + 1, step)], True
        return [indices], False

    # setup empty slices list
    Ninds = len(eval_ind)
    slices = []

    # iterate over indices
    start = last_step = None
    for ind in eval_ind:
        if last_step is None:
            # Check if this is the first slice, in which case start is None
            if start is None:
                start = ind
                continue
            last_step = ind - start
            last_ind = ind
            continue

        # calculate step from previous index
        step = ind - last_ind

        # if step != last_step, this ends the slice
        if step != last_step:
            # append to list
            slices.append(slice(start, last_ind + 1, last_step))

            # setup next step
            start = ind
            last_step = None

        last_ind = ind

    # Append the last slice
    slices.append(slice(start, ind + 1, last_step))

    # determine whether slices are a reasonable representation, and determine max_nslice
    # if only max_nslice_frac was supplied.
    if max_nslice is None:
        max_nslice = max_nslice_frac * Ninds
    check = len(slices) <= max_nslice

    if return_index_on_fail and not check:
        return [indices], check
    else:
        return slices, check


def _get_slice_len(s, axlen):
    """
    Get length of a slice s into array of len axlen.

    Parameters
    ----------
    s : slice object
        Slice object to index with
    axlen : int
        Length of axis s slices into

    Returns
    -------
    int
        Length of slice object
    """
    if s.start is None:
        start = 0
    else:
        start = s.start
    if s.stop is None:
        stop = axlen
    else:
        stop = np.min([s.stop, axlen])
    if s.step is None:
        step = 1
    else:
        step = s.step

    return ((stop - 1 - start) // step) + 1


def _index_dset(dset, indices, input_array=None):
    """
    Index a UVH5 data, flags or nsamples h5py dataset.

    Parameters
    ----------
    dset : h5py dataset
        A reference to an HDF5 dataset on disk.
    indices : tuple
        A 3-tuple with the indices to extract along each dimension of dset.
        Each element should contain a list of indices, a slice element,
        or a list of slice elements that will be concatenated after slicing.
        Indices must be provided such that all dimensions can be indexed
        simultaneously. For data arrays with 4 dimensions, the second dimension
        (the old spw axis) should not be included because it can only be length one.

    Returns
    -------
    ndarray
        The indexed dset

    Notes
    -----
    This makes and fills an empty array with dset indices.
    For trivial indexing, (e.g. a trivial slice), constructing
    a new array and filling it is suboptimal over direct
    indexing, e.g. dset[indices].
    This function specializes in repeated slices over the same axis,
    e.g. if indices is [[slice(0, 5), slice(10, 15), ...], ..., ]
    """
    # get dset and arr shape
    dset_shape = dset.shape
    arr_shape, indices = _get_dset_shape(dset, indices)

    if input_array is None:
        # create empty array of dset dtype
        arr = np.empty(arr_shape, dtype=dset.dtype)
    else:
        arr = input_array

    # get arr and dset indices for each dimension in indices
    dset_indices = []
    arr_indices = []
    for i, dset_inds in enumerate(indices):
        if isinstance(dset_inds, (int, np.integer)):
            # this dimension is len 1, so slice is fine
            arr_indices.append([slice(None)])
            dset_indices.append([[dset_inds]])

        elif isinstance(dset_inds, slice):
            # this dimension is just a slice, so slice is fine
            arr_indices.append([slice(None)])
            dset_indices.append([dset_inds])

        elif isinstance(dset_inds, (list, np.ndarray)):
            if isinstance(dset_inds[0], (int, np.integer)):
                # this is a list of integers, append slice
                arr_indices.append([slice(None)])
                dset_indices.append([dset_inds])
            elif isinstance(dset_inds[0], slice):
                # this is a list of slices, need list of slice lens
                slens = [_get_slice_len(s, dset_shape[i]) for s in dset_inds]
                ssums = [sum(slens[:j]) for j in range(len(slens))]
                arr_inds = [slice(s, s + l) for s, l in zip(ssums, slens)]
                arr_indices.append(arr_inds)
                dset_indices.append(dset_inds)

    if len(dset_shape) == 3:
        freq_dim = 1
        pol_dim = 2
    else:
        freq_dim = 2
        pol_dim = 3

    # iterate over each of the 3 axes and fill the array
    for blt_arr, blt_dset in zip(arr_indices[0], dset_indices[0]):
        for freq_arr, freq_dset in zip(arr_indices[freq_dim], dset_indices[freq_dim]):
            for pol_arr, pol_dset in zip(arr_indices[pol_dim], dset_indices[pol_dim]):
                if input_array is None:
                    # index dset and assign to arr
                    if len(dset_shape) == 3:
                        arr[blt_arr, freq_arr, pol_arr] = dset[
                            blt_dset, freq_dset, pol_dset
                        ]
                    else:
                        arr[blt_arr, :, freq_arr, pol_arr] = dset[
                            blt_dset, :, freq_dset, pol_dset
                        ]
                else:
                    # index arr and assign to dset
                    if len(dset_shape) == 3:
                        dset[blt_dset, freq_dset, pol_dset] = arr[
                            blt_arr, freq_arr, pol_arr
                        ]
                    else:
                        dset[blt_dset, :, freq_dset, pol_dset] = arr[
                            blt_arr, :, freq_arr, pol_arr
                        ]

    if input_array is None:
        return arr
    else:
        return


def determine_blt_order(
    time_array, ant_1_array, ant_2_array, baseline_array, Nbls, Ntimes
) -> tuple[str] | None:
    """Get the blt order from analysing metadata."""
    times = time_array
    ant1 = ant_1_array
    ant2 = ant_2_array
    bls = baseline_array

    time_bl = True
    time_a = True
    time_b = True
    bl_time = True
    a_time = True
    b_time = True
    bl_order = True
    a_order = True
    b_order = True
    time_order = True

    if Nbls == 1 and Ntimes == 1:
        return ("baseline", "time")  # w.l.o.g.

    for i, (t, a, b, bl) in enumerate(
        zip(times[1:], ant1[1:], ant2[1:], bls[1:]), start=1
    ):
        on_bl_boundary = i % Nbls == 0
        on_time_boundary = i % Ntimes == 0

        if t < times[i - 1]:
            time_bl = False
            time_a = False
            time_b = False
            time_order = False

            if not on_time_boundary:
                bl_time = False
                a_time = False
                b_time = False

            if bl == bls[i - 1]:
                bl_time = False
            if a == ant1[i - 1]:
                a_time = False
            if b == ant2[i - 1]:
                b_time = False

        elif t == times[i - 1]:
            if bl < bls[i - 1]:
                time_bl = False
            if a < ant1[i - 1]:
                time_a = False
            if b < ant2[i - 1]:
                time_b = False

        if bl < bls[i - 1]:
            bl_time = False
            bl_order = False
            if not on_bl_boundary:
                time_bl = False
        if a < ant1[i - 1]:
            a_time = False
            a_order = False
            if not on_bl_boundary:
                time_a = False
        if b < ant2[i - 1]:
            b_time = False
            b_order = False
            if not on_bl_boundary:
                time_b = False

        if not any(
            (
                time_bl,
                time_a,
                time_b,
                time_bl,
                bl_time,
                a_time,
                b_time,
                bl_order,
                a_order,
                b_order,
                time_order,
            )
        ):
            break

    if Nbls > 1 and Ntimes > 1:
        assert not (
            (time_bl and bl_time)
            or (time_a and a_time)
            or (time_b and b_time)
            or (time_order and a_order)
            or (time_order and b_order)
            or (a_order and b_order)
            or (time_order and bl_order)
        ), (
            "Something went wrong when trying to determine the order of the blts axis. "
            "Please raise an issue on github, as this is not meant to happen."
            "None of the following should ever be True: \n"
            f"\ttime_bl and bl_time: {time_bl and bl_time}\n"
            f"\ttime_a and a_time: {time_a and a_time}\n"
            f"\ttime_b and b_time: {time_b and b_time}\n"
            f"\ttime_order and a_order: {time_order and a_order}\n"
            f"\ttime_order and b_order: {time_order and b_order}\n"
            f"\ta_order and b_order: {a_order and b_order}\n"
            f"\ttime_order and bl_order: {time_order and bl_order}\n\n"
            "Please include the following information in your issue:\n"
            f"Nbls: {Nbls}\n"
            f"Ntimes: {Ntimes}\n"
            f"TIMES: {times}\n"
            f"ANT1: {ant1}\n"
            f"ANT2: {ant2}\n"
            f"BASELINES: {bls}\n"
        )

    if time_bl:
        return ("time", "baseline")
    if bl_time:
        return ("baseline", "time")
    if time_a:
        return ("time", "ant1")
    if a_time:
        return ("ant1", "time")
    if time_b:
        return ("time", "ant2")
    if b_time:
        return ("ant2", "time")
    if bl_order:
        return ("baseline",)
    if a_order:
        return ("ant1",)
    if b_order:
        return ("ant2",)
    if time_order:
        return ("time",)

    return None


def determine_rectangularity(
    time_array: np.ndarray,
    baseline_array: np.ndarray,
    nbls: int,
    ntimes: int,
    blt_order: str | tuple[str] | None = None,
):
    """Determine if the data is rectangular or not.

    Parameters
    ----------
    time_array : array_like
        Array of times in JD.
    baseline_array : array_like
        Array of baseline integers.
    nbls : int
        Number of baselines.
    ntimes : int
        Number of times.
    blt_order : str or tuple of str, optional
        If known, pass the blt_order, which can short-circuit the determination
        of rectangularity.

    Returns
    -------
    is_rect : bool
        True if the data is rectangular, False otherwise.
    time_axis_faster_than_bls : bool
        True if the data is rectangular and the time axis is the last axis (i.e. times
        change first, then bls). False either if baselines change first, OR if it is
        not rectangular.

    Notes
    -----
    Rectangular data is defined as data for which using regular slicing of size Ntimes
    or Nbls will give you either all the same time and all different baselines, or
    vice versa. This does NOT require that the baselines and times are sorted within
    that structure.
    """
    # check if the data is rectangular
    time_first = True
    bl_first = True

    if time_array.size != nbls * ntimes:
        return False, False
    elif nbls * ntimes == 1:
        return True, True
    elif nbls == 1:
        return True, True
    elif ntimes == 1:
        return True, False
    elif blt_order == ("baseline", "time"):
        return True, True
    elif blt_order == ("time", "baseline"):
        return True, False

    # That's all the easiest checks.
    if time_array[1] == time_array[0]:
        time_first = False
    if baseline_array[1] == baseline_array[0]:
        bl_first = False
    if not time_first and not bl_first:
        return False, False

    if time_first:
        time_array = time_array.reshape((nbls, ntimes))
        baseline_array = baseline_array.reshape((nbls, ntimes))
        if np.sum(np.abs(np.diff(time_array, axis=0))) != 0:
            return False, False
        if (np.diff(baseline_array, axis=1) != 0).any():
            return False, False
        return True, True
    elif bl_first:
        time_array = time_array.reshape((ntimes, nbls))
        baseline_array = baseline_array.reshape((ntimes, nbls))
        if np.sum(np.abs(np.diff(time_array, axis=1))) != 0:
            return False, False
        if (np.diff(baseline_array, axis=0) != 0).any():
            return False, False
        return True, False
