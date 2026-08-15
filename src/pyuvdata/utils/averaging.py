# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for averaging data along a single axis."""

import numba
import numpy as np

from .averaging_numba import _mapped_average
from .types import BoolArray, FloatArray, IntArray


def mapped_average(
    data,
    *,
    index_map: IntArray,
    flags: BoolArray,
    weights: FloatArray,
    axis_weights: FloatArray | None = None,
    axis: int = 0,
    n_bins: int | None = None,
    weighted: bool = True,
    sum_mode: bool = False,
    propagate_flags: bool = False,
    allow_promotion: bool = True,
    weights_norm: FloatArray | None = None,
):
    """
    Average an array along one axis onto an arbitrary set of bins.

    This is a weighted average that leaves flagged entries out of the result, except
    where every entry contributing to a bin is flagged, in which case all of them are
    used (which is the usual radio astronomy convention, and keeps the result from
    collapsing to zero). The bins are specified per-element along the averaged axis, so
    they need be neither contiguous nor evenly sized, and elements can be dropped from
    the result entirely.

    Parameters
    ----------
    data : ndarray
        Array of values to average, shape (..., Naxis, ....).
    index_map : ndarray of int
        Array of shape (Naxis,), giving the index of the bin that each element along
        the middle axis that each value contributes to.
    flags : ndarray of bool
        Flags of data, shape (..., Naxis, ...).
    weights : ndarray of float
        Array of weights (or weights-equivalent, e.g., number of samples), shape
        (Nrow, Naxis, Ncol).
    axis_weights : ndarray of float, optional
        Array of shape (Naxis), giving an additional weight to apply to all elements
        belonging to a given position along the averaging axis. Defaults is None, which
        performs no additional weighting.
    axis : int
        Axis of `data` to average along. Default is 0.
    n_bins : int, optional
        Number of bins in the result. Default is None, which evaluates this based on
         the values provided in `index_map`.
    weighted : bool
        If True, weight each entry by the corresponding value in `weights`. If False,
        effectively perform a "flat" average (e.g., equal weighting across unflagged
        entries). Default is True.
    sum_mode : bool
        If True, sum the values that go into the data output rather than averaging.
        Note that the values are still scaled by the weights first if `weighted=True`.
        Default is False.
    propagate_flags : bool
        If True, mark a bin flagged if any of the entries contributing to it are
        flagged. If False, a bin is only marked flagged if all of the contributing
        entries are flagged. Default is False.
    allow_promotion : bool
        If True, will promote data and weights to the highest-precision dtype available
        during the averaging process (after which the data are re-cast back to their
        original dtypes), which reduces the impact of precision issues when averaging
        over large blocks of data (at the cost of some memory usage). If False,
        averaging is performed at the original dtype. Default is True.
    weights_norm : ndarray of float or None
        Values to normalize the value of out_weights by. Must have a shape that is
        broadcastable to `out_weights`. Default is None, which causes `out_weights`
        to be handed back as the summed total.

    Returns
    -------
    out_data : ndarray
        Averaged/summed data values, shape (..., Nbins, ...), same dtype as `data`.
    out_flags : ndarray of bool
        Flags for the averaged values, shape (..., Nbins, ...).
    out_weights : ndarray of float
        Sum of the weights that went into each bin, shape (..., Nbins, ...).

    Raises
    ------
    ValueError
        If the shapes of `data`, `flags` and `weights` do not match, or if `index_map`
        does not match the length of `data` along `axis`.

    """
    if flags.shape != data.shape or weights.shape != data.shape:
        raise ValueError("data, flags and weights must all have the same shape.")

    axis = range(data.ndim)[axis]  # normalize negative values
    if len(index_map) != data.shape[axis]:
        raise ValueError(
            "index_map must have the same length as data along the averaged axis."
        )

    if axis_weights is None:
        axis_weights = np.ones(data.shape[axis], dtype=np.float64)
    elif len(axis_weights) != data.shape[axis]:
        raise ValueError(
            "axis_weights must have the same length as data along the averaged axis."
        )
    axis_weights = np.ascontiguousarray(axis_weights, dtype=np.float64)

    index_map = np.ascontiguousarray(index_map, dtype=np.int64)
    if n_bins is None:
        n_bins = int(index_map.max()) + 1

    # Capture this before mucking with the precision.
    orig_data_dtype = data.dtype
    orig_weights_dtype = weights.dtype

    if weights.dtype == np.float16:
        # numba has no half-precision support, so promote here (which is what the
        # averaging would have had to do internally in any case).
        weights = weights.astype(np.float32)

    # Collapse to the 3D shape that the kernel wants. Note that these are all free
    # reshapes for C-ordered arrays, which is _also_ what the kernel wants.
    n_rows = int(np.prod(data.shape[:axis], dtype=int))
    n_axis = data.shape[axis]
    n_cols = int(np.prod(data.shape[axis + 1 :], dtype=int))
    flat_shape = (n_rows, n_axis, n_cols)
    out_shape = (n_rows, n_bins, n_cols)

    if allow_promotion:
        # Congrats on the promotion!
        data_dtype = np.promote_types(
            data.dtype, np.complex128 if np.iscomplexobj(data) else np.float64
        )
        weight_dtype = np.promote_types(weights.dtype, np.float64)
    else:
        # Accumulate at whatever precision the inputs are stored at.
        data_dtype = data.dtype
        weight_dtype = weights.dtype

    out_data = np.empty(out_shape, dtype=data_dtype)
    out_flags = np.empty(out_shape, dtype=bool)
    out_weights = np.empty(out_shape, dtype=weight_dtype)

    final_shape = data.shape[:axis] + (n_bins,) + data.shape[axis + 1 :]
    data = np.ascontiguousarray(data).reshape(flat_shape)
    flags = np.ascontiguousarray(flags).reshape(flat_shape)
    weights = np.ascontiguousarray(weights).reshape(flat_shape)

    # In testing, 4x the number of threads was about the best performance seen in terms
    # of the number of blocks to break things into. First try to break things up along
    # the outer-most axis, then resort to breaking up the inner-most axis.
    target_blocks = 4 * numba.get_num_threads()
    n_row_blocks = max(min(n_rows, target_blocks), 1)
    n_col_blocks = max(min(n_cols, -(-target_blocks // n_row_blocks)), 1)

    _mapped_average(
        data=data,
        flags=flags,
        weights=weights,
        axis_weights=axis_weights,
        index_map=index_map,
        out_data=out_data,
        out_flags=out_flags,
        out_weights=out_weights,
        n_row_blocks=n_row_blocks,
        n_col_blocks=n_col_blocks,
        weighted=weighted,
        sum_mode=sum_mode,
        propagate_flags=propagate_flags,
    )

    # Reshape just out_weights here so that we can do the normalization step if needed
    out_weights = out_weights.reshape(final_shape)
    if weights_norm is not None:
        out_weights /= weights_norm

    return (
        out_data.astype(orig_data_dtype, copy=False).reshape(final_shape),
        out_flags.reshape(final_shape),
        out_weights.astype(orig_weights_dtype, copy=False),
    )
