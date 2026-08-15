# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Numba-enhanced kernels for averaging data along a single axis."""

import numba
import numpy as np

from .types import BoolArray, FloatArray, Int64Array


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _mapped_average(
    data: np.ndarray,
    flags: BoolArray,
    weights: FloatArray,
    axis_weights: FloatArray,
    index_map: Int64Array,
    out_data: np.ndarray,
    out_flags: BoolArray,
    out_weights: FloatArray,
    n_row_blocks: int,
    n_col_blocks: int,
    weighted: bool,
    sum_mode: bool,
    propagate_flags: bool,
) -> None:
    """
    Average `data` along its middle axis onto the bins given by `index_map`.

    All arrays are expected to have a 3D shape (Nrows, Naxis, Ncols), with averaging
    performed along the middle axis. In practice, this allows many array types to be
    passed through (even if Nrows or Ncols is equal to 1). Work is split into blocks, to
    allow for multithreaded processing -- nominally work is split along the outer axis
    first (although in some cases, such as time-averaging, this isn't actually
    possible).

    data : ndarray
        Array of values to average, shape (Nrow, Naxis, Ncol).
    flags : ndarray of bool
        Flags of data, shape (Nrow, Naxis, Ncol).
    weights : ndarray of float
        Array of weights (or weights-equivalent, e.g., number of samples), shape
        (Nrow, Naxis, Ncol).
    axis_weights : ndarray of float
        Array of shape (Naxis), giving an additional weight to apply to all elements
        belonging to a given position along the averaging axis.
    index_map : ndarray of int
        Array of shape (Naxis,), giving the index of the bin that each element along
        the middle axis that each value contributes to.
    out_data : ndarray
        Array of values to assign averaged/summed values to, shape (Nrows, Nbin, Ncol).
    out_flags : ndarray of bool
        Flags for the averaged values, shape (Nrow, Nbins, Ncol).
    out_weights : ndarray of float
        Sum of the weights that went into each bin, shape (Nrow, Nbins, Ncol),
        same dtype as `weights`. Note that unlike `out_data`, these values are always
        summed.
    n_row_blocks : int
        How many blocks to split the rows (first axis) into for processing.
    n_col_blocks : int
        How many blocks to split the columns (last axis) into for processing.
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
    """
    #  I think this qualifies as a "middle out" algorithm.
    n_rows, n_axis, n_cols = data.shape
    n_bins = out_data.shape[1]
    row_step = (n_rows + n_row_blocks - 1) // n_row_blocks
    col_step = (n_cols + n_col_blocks - 1) // n_col_blocks

    for block in numba.prange(n_row_blocks * n_col_blocks):
        # Each thread takes a certain range of entries on the leftmost and rightmost
        # axes -- figure out this thread's range now.
        row_start = (block // n_col_blocks) * row_step
        row_stop = min(row_start + row_step, n_rows)
        col_start = (block % n_col_blocks) * col_step
        col_stop = min(col_start + col_step, n_cols)
        n_chunks = col_stop - col_start

        # Set up working arrays per-thread
        good_vals = np.zeros((n_bins, n_chunks), dtype=out_data.dtype)
        good_wgts = np.zeros((n_bins, n_chunks), dtype=out_weights.dtype)
        good_count = np.zeros((n_bins, n_chunks), dtype=np.int64)
        flag_vals = np.zeros((n_bins, n_chunks), dtype=out_data.dtype)
        flag_wgts = np.zeros((n_bins, n_chunks), dtype=out_weights.dtype)
        flag_count = np.zeros((n_bins, n_chunks), dtype=np.int64)

        for idx in range(row_start, row_stop):
            for jdx in range(n_axis):
                bin_jdx = index_map[jdx]
                if bin_jdx < 0:
                    # Negative entries demark values that are dropped entirely.
                    continue

                # Capture the per-index axis-based weight here
                axis_wgt = axis_weights[jdx]
                for kdx in range(n_chunks):
                    col = col_start + kdx
                    wgt = weights[idx, jdx, col] * axis_wgt

                    # For a weighted average the values are scaled by the weights here
                    if weighted:
                        val = data[idx, jdx, col] * wgt
                    else:
                        val = data[idx, jdx, col]

                    # Tally two sums -- all "goods" and all "flags". We'll use the
                    # latter if there are no good values that we can preserve, which
                    # matches prior behavior of various methods this function replaces.
                    if flags[idx, jdx, col]:
                        flag_vals[bin_jdx, kdx] += val
                        flag_wgts[bin_jdx, kdx] += wgt
                        flag_count[bin_jdx, kdx] += 1
                    else:
                        good_vals[bin_jdx, kdx] += val
                        good_wgts[bin_jdx, kdx] += wgt
                        good_count[bin_jdx, kdx] += 1

            for jdx in range(n_bins):
                for kdx in range(n_chunks):
                    # kdx indexes this thread's chunk, so shift back to the position
                    # on the full axis before recording anything.
                    col = col_start + kdx
                    n_good = good_count[jdx, kdx]
                    n_flag = flag_count[jdx, kdx]

                    if n_good == 0:
                        if n_flag == 0:
                            # Nothing at all contributed to this bin, so there is
                            # nothing to report beyond marking it flagged.
                            out_data[idx, jdx, col] = 0
                            out_weights[idx, jdx, col] = 0
                            out_flags[idx, jdx, col] = True
                            continue

                        # Nothing unflagged went into this bin, so fall back to the
                        # flagged values to populate the bin (matching prior behavior).
                        val = flag_vals[jdx, kdx]
                        wgt = flag_wgts[jdx, kdx]
                        count = n_flag
                    else:
                        val = good_vals[jdx, kdx]
                        wgt = good_wgts[jdx, kdx]
                        count = n_good

                    if sum_mode:
                        out_data[idx, jdx, col] = val
                    else:
                        out_data[idx, jdx, col] = val / (wgt if weighted else count)

                    out_weights[idx, jdx, col] = wgt
                    if propagate_flags:
                        out_flags[idx, jdx, col] = n_flag > 0
                    else:
                        out_flags[idx, jdx, col] = n_good == 0

                    # Zero out the temp arrays so that we can use in on the next
                    # binning cycle.
                    good_vals[jdx, kdx] = flag_vals[jdx, kdx] = 0
                    good_wgts[jdx, kdx] = flag_wgts[jdx, kdx] = 0
                    good_count[jdx, kdx] = flag_count[jdx, kdx] = 0
