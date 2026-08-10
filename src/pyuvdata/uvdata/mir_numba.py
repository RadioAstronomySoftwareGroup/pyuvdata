# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""
Numba kernels for unpacking MIR "packed data" records.

MIR stores the visibilities and autos as one contiguous "packdata" block per
integration, inside which each spectral record occupies a known subset of array values.
Unpacking that block is an expensive step in reading a MIR data set, so the kernels
here exist to do it in one pass over memory, in parallel across spectral records.

Note also that numba has no array type for non-native byte order, so callers must
byte-swap old-format (big-endian) MIR data before it reaches these kernels.
"""

import numba as nb
import numpy as np

# Value used by MIR to mark a flagged channel. The compression scheme cannot produce
# the most negative int16, which is what makes it useful for marking flags.
BAD_VAL = -32768

# Record count below which callers should copy with numpy rather than calling
# `copy_records`. Note that this may require some additional testing/tuning.
COPY_KERNEL_MIN_RECS = 24


@nb.njit(cache=True, parallel=True)
def unpack_scaled(
    packdata, sidx, nchan, out_off, norm, wt_val, all_flag, data_out, flag_out, wt_out
):
    """
    Unpack commonly-scaled records into visibilities.

    Reads in a "packdata" block of spectra, where each spectral record consists of a
    common exponent followed by Nchan pairs of real and imaginary values for each
    spectra, for a total of (2 * Nchan) + 1 int16 values. The common exponent gives a
    power-of-2 scale factor, which is multiplied by the normalization factor supplied by
    the user (typically used for scaling by system temperature/SEFD).

    Parameters
    ----------
    packdata : ndarray of int16
        The packed data block for a single integration.
    sidx : ndarray of int
        Index of where the record starts for each spectrum.
    nchan : ndarray of int
        Number of spectral channels in each record.
    out_off : ndarray of int
        Index at which each record's channels begin in the output buffers.
    norm : ndarray of float32
        Per-record multiplicative normalization applied to the data (1 for none). Note
        that float64 works too, but is cast down to single precision.
    wt_val : ndarray of float32
        Per-record value written to the weights of every unflagged channel. Note that
        float64 works too, but is cast down to single precision.
    all_flag : ndarray of bool
        Records for which every channel should be marked flagged. `norm` and `wt_val`
        are ignored for these records.
    data_out : ndarray of complex64
        Output buffer for the visibilities. Modified in place.
    flag_out : ndarray of bool
        Output buffer for the per-channel flags. Modified in place.
    wt_out : ndarray of float32
        Output buffer for the per-channel weights. Modified in place.
    """
    for irec in nb.prange(len(sidx)):
        start = sidx[irec]
        off = out_off[irec]

        # The common exponent sits at the head of the record; the channel pairs follow.
        flag_state = all_flag[irec]
        sfac = np.exp2(np.float32(packdata[start]))
        if flag_state:
            wgt = np.float32(1.0)
        else:
            # Cast explicitly here to preverve backwards compatibility/behavior with
            # testing against prior numpy-based behavior.
            sfac *= np.float32(norm[irec])
            wgt = np.float32(wt_val[irec])

        start += 1
        for jdx in range(nchan[irec]):
            re_val = packdata[start + (2 * jdx)]
            sub_off = off + jdx
            if re_val == BAD_VAL:
                # Only the real component is checked, but the writer marks both.
                data_out[sub_off] = np.float32(0)
                flag_out[sub_off] = True
                wt_out[sub_off] = np.float32(0.0)
            else:
                im_val = packdata[start + (2 * jdx) + 1]
                # Complex turns out to be the most efficient way to cast this.
                data_out[sub_off] = complex(re_val * sfac, im_val * sfac)
                flag_out[sub_off] = flag_state
                wt_out[sub_off] = wgt


@nb.njit(cache=True, parallel=True)
def copy_records(packdata, sidx, nvals, out_off, data_out):
    """
    Copy a set of records out of a packdata block into one contiguous buffer.

    Supports different dtypes (since the autos and crosses are stored as different
    types, floats vs ints respectively). Note the copy is written via slice assignment
    rather than an element loop so that it lowers to memcpy. Note that this method is
    parallelized to allow for multiple read/write operations simultaneously between
    spectral records.

    Parameters
    ----------
    packdata : ndarray
        The packed data block for a single integration.
    sidx : ndarray of int
        Index at which each record's values begin within `packdata`.
    nvals : ndarray of int
        Number of values to copy for each record.
    out_off : ndarray of int
        Index at which each record's values begin in the output buffer.
    data_out : ndarray
        Output buffer, of the same dtype as `packdata`. Modified in place.
    """
    for irec in nb.prange(len(sidx)):
        start = sidx[irec]
        off = out_off[irec]
        nval = nvals[irec]
        data_out[off : off + nval] = packdata[start : start + nval]
