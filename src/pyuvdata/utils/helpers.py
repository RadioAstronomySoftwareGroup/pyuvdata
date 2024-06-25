# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Helper utilities."""

import warnings
from collections.abc import Iterable
from typing import Iterable as IterableType

import numpy as np
from astropy.coordinates import EarthLocation

from .coordinates import hasmoon
from .lst import get_lst_for_time

if hasmoon:
    from lunarsky import MoonLocation

_range_dict = {
    "itrs": (6.35e6, 6.39e6, "Earth"),
    "mcmf": (1717100.0, 1757100.0, "Moon"),
}


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


def _test_array_constant(array, *, tols=None):
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


def _test_array_constant_spacing(array, *, tols=None):
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


def _check_flex_spw_contiguous(*, spw_array, flex_spw_id_array):
    """
    Check if the spectral windows are contiguous for multi-spw datasets.

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
    *,
    freq_array,
    freq_tols,
    channel_width,
    channel_width_tols,
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
        Array of frequencies, shape (Nfreqs,).
    freq_tols : tuple of float
        freq_array tolerances (from uvobj._freq_array.tols).
    channel_width : array of float
        Channel widths, either a scalar or an array of shape (Nfreqs,).
    channel_width_tols : tuple of float
        channel_width tolerances (from uvobj._channel_width.tols).
    spw_array : array of integers or None
        Array of spectral window numbers, shape (Nspws,).
    flex_spw_id_array : array of integers or None
        Array of spectral window numbers per frequency channel, shape (Nfreqs,).
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

    # Check to make sure that the flexible spectral window has indicies set up
    # correctly (grouped together) for this check
    _check_flex_spw_contiguous(spw_array=spw_array, flex_spw_id_array=flex_spw_id_array)

    for spw_id in spw_array:
        mask = flex_spw_id_array == spw_id
        if sum(mask) > 1:
            freq_spacing = np.diff(freq_array[mask])
            freq_dir = -1.0 if all(freq_spacing < 0) else 1.0
            if not _test_array_constant(freq_spacing, tols=freq_tols):
                spacing_error = True
            if not _test_array_constant(channel_width[mask], tols=channel_width_tols):
                spacing_error = True
            elif not np.allclose(
                freq_spacing,
                np.mean(channel_width[mask]) * freq_dir,
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


def _get_iterable(x):
    """Return iterable version of input."""
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


def _sort_freq_helper(
    *,
    Nfreqs,  # noqa: N803
    freq_array,
    Nspws,
    spw_array,
    flex_spw_id_array,
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
    flex_spw_id_array : array_like of int
        Array of SPW IDs for each channel, taken directly from the object parameter.
    spw_order : str or array_like of int
        A string describing the desired order of spectral windows along the
        frequency axis. Allowed strings include `number` (sort on spectral window
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
        temp_freqs = 1.0 * freq_array
        # Same trick for ints -- add 0 to make a cheap copy
        temp_spws = 0 + flex_spw_id_array

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


def _sorted_unique_union(obj1, obj2=None):
    """
    Determine the union of unique elements from two lists.

    Convenience function for handling various actions with indices.

    Parameters
    ----------
    obj1 : list or tuple or set or 1D ndarray
        First list from which to determine unique entries.
    obj2 : list or tuple or set or 1D ndarray
        Second list from which to determine unique entries, which is joined with the
        first list. If None, the method will simply return the sorted list of unique
        elements in obj1.

    Returns
    -------
    sorted_unique : list
        List containing the union of unique entries between obj1 and obj2.
    """
    return sorted(set(obj1)) if obj2 is None else sorted(set(obj1).union(obj2))


def _sorted_unique_intersection(obj1, obj2=None):
    """
    Determine the intersection of unique elements from two lists.

    Convenience function for handling various actions with indices.

    Parameters
    ----------
    obj1 : list or tuple or set or 1D ndarray
        First list from which to determine unique entries.
    obj2 : list or tuple or set or 1D ndarray
        Second list from which to determine unique entries, which is intersected with
        the first list. If None, the method will simply return the sorted list of unique
        elements in obj1.

    Returns
    -------
    sorted_unique : list
        List containing the intersection of unique entries between obj1 and obj2.
    """
    return sorted(set(obj1)) if obj2 is None else sorted(set(obj1).intersection(obj2))


def _sorted_unique_difference(obj1, obj2=None):
    """
    Determine the difference of unique elements from two lists.

    Convenience function for handling various actions with indices.

    Parameters
    ----------
    obj1 : list or tuple or set or 1D ndarray
        First list from which to determine unique entries.
    obj2 : list or tuple or set or 1D ndarray
        Second list from which to determine unique entries, which is differenced with
        the first list. If None, the method will simply return the sorted list of unique
        elements in obj1.

    Returns
    -------
    sorted_unique : list
        List containing the difference in unique entries between obj1 and obj2.
    """
    return sorted(set(obj1)) if obj2 is None else sorted(set(obj1).difference(obj2))


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


def _convert_to_slices(
    indices, *, max_nslice_frac=0.1, max_nslice=None, return_index_on_fail=False
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


def slicify(ind: slice | None | IterableType[int]) -> slice | None | IterableType[int]:
    """Convert an iterable of integers into a slice object if possible."""
    if ind is None or isinstance(ind, slice):
        return ind
    if len(ind) == 0:
        return None

    if len(set(np.ediff1d(ind))) <= 1:
        return slice(ind[0], ind[-1] + 1, ind[1] - ind[0] if len(ind) > 1 else 1)
    else:
        # can't slicify
        return ind


def _check_range_overlap(val_range, range_type="time"):
    """
    Detect if any val_range in an array overlap.

    Parameters
    ----------
    val_range : np.array of float
        Array of ranges, shape (Nranges, 2).
    range_type : str
        Type of range (for good error messages)

    Returns
    -------
    bool
        True if any range overlaps.
    """
    # first check that time ranges are well formed (stop is >= than start)
    if np.any((val_range[:, 1] - val_range[:, 0]) < 0):
        raise ValueError(
            f"The {range_type} ranges are not well-formed, some stop {range_type}s "
            f"are after start {range_type}s."
        )

    # Sort by start time
    sorted_ranges = val_range[np.argsort(val_range[:, 0]), :]

    # then check if adjacent pairs overlap
    for ind in range(sorted_ranges.shape[0] - 1):
        range1 = sorted_ranges[ind]
        range2 = sorted_ranges[ind + 1]
        if range2[0] < range1[1]:
            return True


def _select_times_helper(
    *,
    times,
    time_range,
    lsts,
    lst_range,
    obj_time_array,
    obj_time_range,
    obj_lst_array,
    obj_lst_range,
    time_tols,
    lst_tols,
):
    """
    Get time indices in a select.

    Parameters
    ----------
    times : array_like of float
        The times to keep in the object, each value passed here should exist in the
        time_array. Can be None, cannot be set with `time_range`, `lsts` or `lst_array`.
    time_range : array_like of float
        The time range in Julian Date to keep in the object, must be length 2. Some of
        the times in the object should fall between the first and last elements. Can be
        None, cannot be set with `times`, `lsts` or `lst_array`.
    lsts : array_like of float
        The local sidereal times (LSTs) to keep in the object, each value passed here
        should exist in the lst_array. Can be None, cannot be set with `times`,
        `time_range`, or `lst_range`.
    lst_range : array_like of float
        The local sidereal time (LST) range in radians to keep in the
        object, must be of length 2. Some of the LSTs in the object should
        fall between the first and last elements. If the second value is
        smaller than the first, the LSTs are treated as having phase-wrapped
        around LST = 2*pi = 0, and the LSTs kept on the object will run from
        the larger value, through 0, and end at the smaller value. Can be None, cannot
        be set with `times`, `time_range`, or `lsts`.
    obj_time_array : array_like of float
        Time array on object. Can be None if `object_time_range` is set.
    obj_time_range : array_like of float
        Time range on object. Can be None if `object_time_array` is set.
    obj_lst_array : array_like of float
        LST array on object. Can be None if `object_lst_range` is set.
    obj_lst_range : array_like of float
        LST range on object. Can be None if `object_lst_array` is set.
    time_tols : tuple of float
        Length 2 tuple giving (rtol, atol) to use for time matching.
    lst_tols : tuple of float
        Length 2 tuple giving (rtol, atol) to use for lst matching.

    """
    have_times = times is not None
    have_time_range = time_range is not None
    have_lsts = lsts is not None
    have_lst_range = lst_range is not None
    n_time_params = np.count_nonzero(
        [have_times, have_time_range, have_lsts, have_lst_range]
    )
    if n_time_params > 1:
        raise ValueError(
            "Only one of [times, time_range, lsts, lst_range] may be "
            "specified per selection operation."
        )
    if n_time_params == 0:
        return None

    time_inds = np.zeros(0, dtype=np.int64)
    if times is not None:
        times = _get_iterable(times)
        if np.array(times).ndim > 1:
            times = np.array(times).flatten()

        if obj_time_range is not None:
            for jd in times:
                this_ind = np.nonzero(
                    np.logical_and(
                        (obj_time_range[:, 0] <= jd), (obj_time_range[:, 1] >= jd)
                    )
                )[0]
                if this_ind.size > 0:
                    time_inds = np.append(time_inds, this_ind)
                else:
                    raise ValueError(f"Time {jd} does not fall in any time_range.")
        else:
            for jd in times:
                if np.any(
                    np.isclose(obj_time_array, jd, rtol=time_tols[0], atol=time_tols[1])
                ):
                    time_inds = np.append(
                        time_inds,
                        np.where(
                            np.isclose(
                                obj_time_array, jd, rtol=time_tols[0], atol=time_tols[1]
                            )
                        )[0],
                    )
                else:
                    raise ValueError(f"Time {jd} is not present in the time_array.")

    if time_range is not None:
        if np.size(time_range) != 2:
            raise ValueError("time_range must be length 2.")

        if obj_time_range is not None:
            for tind, trange in enumerate(obj_time_range):
                if _check_range_overlap(np.stack((trange, time_range), axis=0)):
                    time_inds = np.append(time_inds, tind)
            attr_str = "time_range"
        else:
            time_inds = np.nonzero(
                (obj_time_array <= time_range[1]) & (obj_time_array >= time_range[0])
            )[0]
            attr_str = "time_array"
        if time_inds.size == 0:
            raise ValueError(
                f"No elements in {attr_str} between {time_range[0]} and "
                f"{time_range[1]}."
            )

    if (lsts is not None or lst_range is not None) and obj_lst_range is not None:
        # check for lsts wrapping around zero
        lst_range_wrap = obj_lst_range[:, 0] > obj_lst_range[:, 1]

    if lsts is not None:
        if np.any(np.asarray(lsts) > 2 * np.pi):
            warnings.warn(
                "The lsts parameter contained a value greater than 2*pi. "
                "LST values are assumed to be in radians, not hours."
            )
        lsts = _get_iterable(lsts)
        if np.array(lsts).ndim > 1:
            lsts = np.array(lsts).flatten()

        if obj_lst_range is not None:
            for lst in lsts:
                lst_ind = np.nonzero(
                    np.logical_and(
                        (obj_lst_range[:, 0] <= lst), (obj_lst_range[:, 1] >= lst)
                    )
                )[0]
                if lst_ind.size == 0 and np.any(lst_range_wrap):
                    for lr_ind in np.nonzero(lst_range_wrap)[0]:
                        if (obj_lst_range[lr_ind, 0] <= lst and lst <= 2 * np.pi) or (
                            lst >= 0 and lst <= obj_lst_range[lr_ind, 1]
                        ):
                            lst_ind = np.array([lr_ind])
                if lst_ind.size > 0:
                    time_inds = np.append(time_inds, lst_ind)
                else:
                    raise ValueError(f"LST {lst} does not fall in any lst_range")
        else:
            for lst in lsts:
                if np.any(
                    np.isclose(obj_lst_array, lst, rtol=lst_tols[0], atol=lst_tols[1])
                ):
                    time_inds = np.append(
                        time_inds,
                        np.where(
                            np.isclose(
                                obj_lst_array, lst, rtol=lst_tols[0], atol=lst_tols[1]
                            )
                        )[0],
                    )
                else:
                    raise ValueError(f"LST {lst} is not present in the lst_array")

    if lst_range is not None:
        if np.size(lst_range) != 2:
            raise ValueError("lst_range must be length 2.")
        if np.any(np.asarray(lst_range) > 2 * np.pi):
            warnings.warn(
                "The lst_range contained a value greater than 2*pi. "
                "LST values are assumed to be in radians, not hours."
            )
        if obj_lst_range is not None:
            for lind, lrange in enumerate(obj_lst_range):
                if not lst_range_wrap[lind] and lst_range[0] < lst_range[1]:
                    if _check_range_overlap(np.stack((lrange, lst_range), axis=0)):
                        time_inds = np.append(time_inds, lind)
                else:
                    if (lst_range[0] >= lrange[0] and lst_range[0] <= 2 * np.pi) or (
                        lst_range[1] <= lrange[1] and lst_range[1] >= 0
                    ):
                        time_inds = np.append(time_inds, lind)
            attr_str = "lst_range"
        else:
            if lst_range[1] < lst_range[0]:
                # we're wrapping around LST = 2*pi = 0
                lst_range_1 = [lst_range[0], 2 * np.pi]
                lst_range_2 = [0, lst_range[1]]
                time_inds1 = np.nonzero(
                    (obj_lst_array <= lst_range_1[1])
                    & (obj_lst_array >= lst_range_1[0])
                )[0]
                time_inds2 = np.nonzero(
                    (obj_lst_array <= lst_range_2[1])
                    & (obj_lst_array >= lst_range_2[0])
                )[0]
                time_inds = np.union1d(time_inds1, time_inds2)
            else:
                time_inds = np.nonzero(
                    (obj_lst_array <= lst_range[1]) & (obj_lst_array >= lst_range[0])
                )[0]
            attr_str = "lst_array"

        if time_inds.size == 0:
            raise ValueError(
                f"No elements in {attr_str} between {lst_range[0]} and "
                f"{lst_range[1]}."
            )
    return time_inds


def check_lsts_against_times(
    *,
    jd_array,
    lst_array,
    lst_tols,
    latitude=None,
    longitude=None,
    altitude=None,
    frame="itrs",
    ellipsoid=None,
    telescope_loc=None,
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
    telescope_loc : tuple or EarthLocation or MoonLocation
        Alternative way of specifying telescope lat/lon/alt, either as a 3-element tuple
        or as an astropy EarthLocation (or lunarsky MoonLocation). Cannot supply both
        `telescope_loc` and `latitute`, `longitude`, or `altitude`.

    Returns
    -------
    None

    Warns
    -----
    If the `lst_array` does not match the calculated LSTs to the lst_tols.

    """
    # Don't worry about passing the astrometry library because we test that they agree
    # to better than our standard lst tolerances.
    lsts = get_lst_for_time(
        jd_array=jd_array,
        telescope_loc=telescope_loc,
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


def determine_blt_order(
    *, time_array, ant_1_array, ant_2_array, baseline_array, Nbls, Ntimes  # noqa: N803
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
    *,
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
