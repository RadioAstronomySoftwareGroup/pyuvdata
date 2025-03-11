# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Basic utility functions."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterable as IterableType

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
        If set to True and the list of input indexes cannot easily be represented by
        a list of slices (as defined by `max_nslice` or `max_nslice_frac`), then return
        the input list of index values instead of a list of suboptimal slices.

    Returns
    -------
    slice_list : list
        Nominally the list of slice objects used to represent indices. However, if
        `return_index_on_fail=True` and input indexes cannot easily be represented,
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
    if isinstance(indices, int | np.integer):
        return [slice(indices, indices + 1, 1)], True

    # check for boolean index
    if isinstance(indices, np.ndarray) and (indices.dtype == bool):
        eval_ind = np.where(indices)[0]
    else:
        eval_ind = indices
    # assert indices is longer than 2, or return trivial solutions
    if len(eval_ind) == 0:
        return [slice(0, 0)], False
    if len(eval_ind) <= 2:
        step = 1 if (len(eval_ind) < 2) else eval_ind[-1] - eval_ind[0]
        start = eval_ind[0]
        stop = eval_ind[-1] + step
        return [slice(start, None if (stop < 0) else stop, step)], True

    # Catch the simplest case of "give me a single slice or exit"
    if (max_nslice == 1) and return_index_on_fail:
        step = eval_ind[1] - eval_ind[0]
        start = eval_ind[0]
        stop = eval_ind[-1] + step
        if all(np.diff(eval_ind) == step):
            return [slice(start, None if (stop < 0) else stop, step)], True
        return [indices], False

    # setup empty slices list
    slices = []

    # iterate over indices
    start = eval_ind[0]
    step = None
    for ind in eval_ind[1:]:
        if step is None:
            step = ind - start
            stop = ind + step
            continue

        # if the next index doesn't line up w/ the stop, this ends the slice
        if ind != stop:
            # append to list
            slices.append(slice(start, None if (stop < 0) else stop, step))

            # setup next step
            start = ind
            stop = ind + 1  # Set this in case loop ends here
            step = None
        else:
            stop += step

    # Append the last slice
    slices.append(slice(start, None if (stop < 0) else stop, step))

    # determine whether slices are a reasonable representation, and determine max_nslice
    # if only max_nslice_frac was supplied.
    if max_nslice is None and max_nslice_frac is not None:
        max_nslice = max_nslice_frac * len(eval_ind)
    check = len(slices) <= max_nslice

    if return_index_on_fail and not check:
        return [indices], check
    else:
        return slices, check


def slicify(
    ind: slice | None | IterableType[int], allow_empty: bool = False
) -> slice | None | IterableType[int]:
    """
    Convert an iterable of integers into a slice object if possible.

    Parameters
    ----------
    ind : list
        A 1D list of integers for array indexing.
    allow_empty : bool
        If set to False (default) and ind is a zero-length list, None is returned. If
        set to True, then a "zero-length slice" (e.g., `slice(0,0)`) is returned
        instead.

    Returns
    -------
    index_obj : slice or list
        If the list of indices can be represented by a slice, a slice is returned,
        otherwise the list of indices is returned.
    """
    if ind is None or isinstance(ind, slice):
        return ind
    if len(ind) == 0:
        return slice(0, 0, 1) if allow_empty else None
    if len(ind) == 1:
        return slice(ind[0], ind[0] + 1, 1)

    step = ind[1] - ind[0]
    if all(np.ediff1d(ind) == step):
        start = ind[0]
        stop = ind[-1] + step
        return slice(start, None if (stop < 0) else stop, step)
    else:
        # can't slicify
        return ind


def _multidim_ind2sub(dims_dict, dims):
    """
    Build a flag index array based on a multi-dimensional index array.

    Parameters
    ----------
    dims_dict : dict
        Dict whose keys are the axes being selected on, and the values are list of
        index positions along that axis.
    dims : tuple
        Shape of the array being accessed.
    """
    Ndims = len(dims)
    indices = [None] * Ndims
    for axis in range(Ndims):
        arr = np.asarray(dims_dict.get(axis, np.arange(dims[axis])))
        indices[axis] = arr.reshape([-1 if axis == idx else 1 for idx in range(Ndims)])

    ravel_arr = np.ravel_multi_index(indices, dims=dims).flatten()
    new_dims = tuple(indices[idx].shape[idx] for idx in range(Ndims))

    return ravel_arr, new_dims


def _test_array_constant(array, *, tols=None, mask=...):
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
    mask : array-like (of ints or booleans) or Ellipses
        Mask which indicates which indices to evaluate. Default is all elements.

    Returns
    -------
    bool
        True if the array is constant to the given tolerances, False otherwise.
    """
    # Import UVParameter here rather than at the top to avoid circular imports
    from pyuvdata.parameter import UVParameter

    if isinstance(array, UVParameter):
        array_to_test = np.asarray(array.value)[mask]
        if tols is None:
            tols = array.tols
    else:
        array_to_test = np.asarray(array)[mask]
        if tols is None:
            tols = (0, 0)
    assert isinstance(tols, tuple), "tols must be a length-2 tuple"
    assert len(tols) == 2, "tols must be a length-2 tuple"

    if array_to_test.size < 2:
        # arrays with 0 or 1 elements are constant by definition
        return True

    min_val = np.min(array_to_test)
    max_val = np.max(array_to_test)

    # if min and max are equal don't bother with tolerance checking
    if min_val == max_val:
        return True

    return np.isclose(min_val, max_val, rtol=tols[0], atol=tols[1])


def _test_array_consistent(array, deltas, *, tols=None, mask=...):
    """
    Check if an the spacing of an array is consistent with expect intervals.

    Parameters
    ----------
    array : np.ndarray or UVParameter
        UVParameter or array to check for constant values.
    deltas : np.ndarray or UVParameter
        Expected widths of each entry in array, should be >= 0.
    tols : tuple of float, optional
        length 2 tuple giving (rtol, atol) to pass to np.isclose, defaults to (0, 0) if
        passing an array, otherwise defaults to using the tolerance on the UVParameter.
    mask : array-like (of ints or booleans) or Ellipses
        Mask which indicates which indices to evaluate. Default is all elements.

    Returns
    -------
    bool
        True if the array is constant to the given tolerances, False otherwise.
    """
    # Import UVParameter here rather than at the top to avoid circular imports
    from pyuvdata.parameter import UVParameter

    if isinstance(array, UVParameter):
        array_to_test = np.asarray(array.value)[mask]
        if tols is None:
            tols = array.tols
    else:
        array_to_test = np.asarray(array)[mask]
        if tols is None:
            tols = (0, 0)
    if isinstance(deltas, UVParameter):
        deltas_to_test = np.asarray(deltas.value)[mask]
    else:
        deltas_to_test = np.asarray(deltas)[mask]

    if deltas_to_test.size == 1:
        exp_deltas = deltas_to_test
    else:
        assert array_to_test.shape == deltas_to_test.shape, (
            "array and deltas must have same shape"
        )
        exp_deltas = (deltas_to_test[:-1] + deltas_to_test[1:]) * 0.5

    assert isinstance(tols, tuple), "tols must be a length-2 tuple"
    assert len(tols) == 2, "tols must be a length-2 tuple"

    if array is None or deltas is None or array_to_test.size < 2:
        # arrays with 0 or 1 elements are constant by definition
        return True

    # Call the mask after isclose to handle non-ndarrays like lists
    return np.allclose(
        np.abs(np.diff(array_to_test)), exp_deltas, rtol=tols[0], atol=tols[1]
    )


def _test_array_constant_spacing(array, *, tols=None, mask=..., allow_resort=False):
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
    mask : array-like (of ints or booleans) or Ellipses
        Mask which indicates which indices to evaluate. Default is all elements.
    allow_resort : bool
        If set to False, values in array are checked in their present order. If set to
        True, values are sorted prior to evaluating (useful for arrays that _can_ be
        reindexed). Default is False.

    Returns
    -------
    bool
        True if the array spacing is constant to the given tolerances, False otherwise.
    """
    # Import UVParameter here rather than at the top to avoid circular imports
    from pyuvdata.parameter import UVParameter

    if isinstance(array, UVParameter):
        array_to_test = np.asarray(array.value)[mask]
        if tols is None:
            tols = array.tols
    else:
        array_to_test = np.asarray(array)[mask]
        if tols is None:
            tols = (0, 0)

    if array is None or array_to_test.size <= 2:
        # arrays with 1 or 2 elements are constantly spaced by definition
        return True

    assert isinstance(tols, tuple), "tols must be a length-2 tuple"
    assert len(tols) == 2, "tols must be a length-2 tuple"

    if allow_resort:
        array_to_test = np.sort(array_to_test)

    array_diff = np.diff(array_to_test)
    return _test_array_constant(array_diff, tols=tols)


def _is_between(val, val_range, wrap=False, wrap_amount=(2 * np.pi)):
    """
    Detect if a value is between a specified range(s).

    Parameters
    ----------
    val : float or ndarray
        Value to evaluate, either float/singleton, otherwise of shape (Nranges,).
    val_range : np.array
        Array of ranges, shape (Nranges, 2).
    wrap : bool
        Apply wrapping. Default is False.
    wrap_amount : float
        Top end of the range for the wrap (bottom is 0). Default is 2 * pi.

    Returns
    -------
    bool
        True if any range overlaps
    """
    lo_lim = val_range[..., 0]
    hi_lim = val_range[..., 1]
    if val_range.ndim == 1:
        if wrap and (hi_lim < lo_lim):
            lo_lim = lo_lim - wrap_amount
    elif wrap:
        hi_lim[hi_lim < lo_lim] += wrap_amount

    mask = (val >= lo_lim) & (val <= hi_lim)
    if wrap:
        if val_range.ndim == 1:
            lo_lim = wrap_amount + lo_lim
            hi_lim = wrap_amount + hi_lim
        else:
            val += wrap_amount
        mask |= (val >= lo_lim) & (val <= hi_lim)

    return mask


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


def _strict_raise(
    err_msg: str, strict: (bool | None), err_type=ValueError, warn_type=UserWarning
):
    """
    Determine whether to raise a warning or an error.

    Parameters
    ----------
    err_msg : str
        Message to pass along with the warning/error.
    strict : bool | None
        If True, raise an error. If False, raise a warning. If None, no message is
        raised at all (warning is silenced).
    err_type : Exception
        Type of error to raise if `strict=True`. Default is ValueError.
    warn_type : Warning
        Type of warning to raise if `strict=False`. Default is UserWarning.
    """
    if strict:
        raise err_type(err_msg)
    elif strict is not None:
        warnings.warn(err_msg, warn_type)


def _eval_inds(inds, nrecs, name="inds", invert=False, strict=True):
    """
    Determine if indices are outside of the expected range.

    Parameters
    ----------
    inds : array-like of int
        Indices to check.
    nrecs : int
        Number of records in the underlying array.
    name : str
        Name of underlying array, default is "inds".
    invert : bool
        If False, inds are treated as the positions in the array that should be
        preserved, but if True, those positions are discarded instead. Default is False.
    strict : bool
        If True, raise an error. If False, raise a warning.

    Returns
    -------
    inds : ndarray of int
        Array of well-conditioned, sorted index values (whose value will be within the
        range of [0, nrecs - 1]).
    """
    if inds is None:
        return None

    inds = np.asarray(inds).flatten()
    mask = np.full(nrecs, invert, dtype=bool)

    if len(inds) > 0:
        fix_inds = False
        if max(inds) >= nrecs:
            _strict_raise(f"{name} contains indices that are too large", strict=strict)
            fix_inds = True
        if min(inds) < 0:
            _strict_raise(f"{name} contains indices that are negative", strict=strict)
            fix_inds = True

        if fix_inds:
            inds = [i for i in inds if ((i >= 0) and (i < nrecs))]

        mask[inds] = not invert

    return np.nonzero(mask)[0]


def _where_combine(mask, inds=None, invert=False, use_and=True):
    """
    Combine masked array with an existing index list.

    Parameters
    ----------
    mask : array-like of bool
        Array that marks whether or not entries meet matching criteria.
    inds : array-like of int or None
        Existing list of index positions that meet matching criteria. Can be None,
        in which case only mask is evaluated.
    invert : bool
        If False, then indices where mask == True are returned. But if set to True,
        indices where mask == False are returned instead. Default is False.
    use_and : bool
        If True, then what is returned is the intersection of value derived from both
        mask and inds. If False, then the union of mask and inds is returned instead.
        Default is True.

    Returns
    -------
    new_inds : ndarray of int
        Index positions which meet the selection criterion recorded in mask and inds.
    """
    eval_func = np.logical_and if use_and else np.logical_or
    if inds is not None:
        postmask = np.full(len(mask), invert, dtype=bool)
        postmask[inds] = not invert
        mask = eval_func(mask, postmask)

    return np.nonzero(np.logical_not(mask) if invert else mask)[0]


def _nants_to_nblts(uvd):
    """
    Obtain indices to convert (Nants,) to (Nblts,).

    Parameters
    ----------
    uvd : UVData object

    Returns
    -------
    ind1, ind2 : ndarray, ndarray
        index pairs to compose (Nblts,) shaped arrays for each
        baseline from an (Nants,) shaped array
    """
    ant_map = {ant: idx for idx, ant in enumerate(uvd.telescope.antenna_numbers)}

    ind1 = [ant_map[ant] for ant in uvd.ant_1_array]
    ind2 = [ant_map[ant] for ant in uvd.ant_2_array]

    return np.asarray(ind1), np.asarray(ind2)


def _ntimes_to_nblts(uvd):
    """
    Obtain indices to convert (Ntimes,) to (Nblts,).

    Parameters
    ----------
    uvd : UVData object
        UVData object

    Returns
    -------
    inds : ndarray
        Indices that, when applied to an array of shape (Ntimes,),
        correctly convert it to shape (Nblts,)
    """
    unique_t = np.unique(uvd.time_array)
    t = uvd.time_array

    inds = []
    for i in t:
        inds.append(np.where(unique_t == i)[0][0])

    return np.asarray(inds)
