# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Basic utility functions."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Iterable as IterableType

import numpy as np


def _get_iterable(x):
    """Return iterable version of input."""
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


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
