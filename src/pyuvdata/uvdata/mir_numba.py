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
    packdata,
    sidx,
    nchan,
    out_off,
    norm,
    wt_val,
    all_flag,
    data_out,
    flag_out,
    wt_out,
    bp_idx=None,
    bp_gain=None,
    bp_weight=None,
    bp_flags=None,
    flag_bytes=None,
    chan_shift=None,
    chan_frac=None,
    chan_avg=None,
):
    """
    Unpack commonly-scaled records into visibilities.

    Reads in a "packdata" block of spectra, where each spectral record consists of a
    common exponent followed by Nchan pairs of real and imaginary values for each
    spectra, for a total of (2 * Nchan) + 1 int16 values. The common exponent gives a
    power-of-2 scale factor, which is multiplied by the normalization factor supplied by
    the user (typically used for scaling by system temperature/SEFD). The normalization
    is complex, and can carry a linear phase ramp across the channels of each record
    (e.g., the change in w following an antenna position update).

    Note that the order of operations here is to unpack the data first, then apply
    normalization/calibration factors and flags, then do any channel shifting, and
    finally perform any channel averaging. Channel shifting uses cubic convolution and
    will broadcast any flags from the input channels to the output channels it affects.

    Parameters
    ----------
    packdata : ndarray of int16
        The packed data block for a single integration.
    sidx : ndarray of int
        Index of where the record starts for each spectrum.
    nchan : ndarray of int
        Number of spectral channels in each record.
    out_off : ndarray of int
        Index at which each record's channels begin in the output buffers (after any
        averaging, if `chan_avg` is supplied).
    norm : ndarray of complex128
        Per-record normalization, of shape (Nrec, 2). The first column multiplies the
        first channel of the record, and each subsequent channel is multiplied by the
        value for the prior channel times the second column (i.e., a phase step). With
        no imaginary part in the first column and a step of exactly 1, the record is
        scaled by the real part alone (cast down to single precision).
    wt_val : ndarray of float32
        Per-record value written to the weights of every unflagged channel. Note that
        float64 works too, but is cast down to single precision.
    all_flag : ndarray of bool
        Records for which every channel should be marked flagged. `wt_val` and the
        amplitude of `norm` are ignored for these records (its phase is still applied).
    data_out : ndarray of complex64
        Output buffer for the visibilities. Modified in place.
    flag_out : ndarray of bool
        Output buffer for the per-channel flags. Modified in place.
    wt_out : ndarray of float32
        Output buffer for the per-channel weights. Modified in place.
    bp_idx : ndarray of int
        Row of the bandpass solutions to apply to each record, with -1 marking that
        there is no solution for the record (which is then flagged outright).
    bp_gain : ndarray of complex64
        Bandpass solutions, of shape (Nsoln, Nchan), multiplying the visibilities.
        Default is None, which applies no bandpass solutions.
    bp_weight : ndarray of float32
        Weight solutions, of shape (Nsoln, Nchan), multiplying the weights.
    bp_flags : ndarray of bool
        Flags for the bandpass solutions, of shape (Nsoln, Nchan).
    flag_bytes : ndarray of uint8
        Further flags for each record, one row per record (of at least Nchan // 8
        values), packed 8 channels to a byte with the first channel of each block of 8
        in the least significant bit (as COMPASS records them). Default is None, which
        applies no further flags.
    chan_shift : ndarray of int
        Whole number of channels to shift each record by (towards higher channel index),
        once unpacked and calibrated. Channels shifted in from beyond the edge of the
        record are flagged. Supplied along with `chan_frac` and `chan_avg`.
    chan_frac : ndarray of float
        Further fraction of a channel to shift each record by, in the range [0, 1).
    chan_avg : ndarray of int
        Number of channels to average together for each record (after shifting and
        calibration), weighted by each channel weight.
    """
    for irec in nb.prange(len(sidx)):
        start = sidx[irec]
        if chan_avg is None:
            # Unpack straight into the output buffers.
            rec_data, rec_flags, rec_wts = data_out, flag_out, wt_out
            off = out_off[irec]
        else:
            # Unpack into a buffer for just this record, to shift and average below.
            rec_data = np.empty(nchan[irec], dtype=np.complex64)
            rec_flags = np.empty(nchan[irec], dtype=np.bool_)
            rec_wts = np.empty(nchan[irec], dtype=np.float32)
            off = 0

        # The common exponent sits at the head of the record; the channel pairs follow.
        flag_state = all_flag[irec]
        sfac = np.exp2(np.float32(packdata[start]))
        rot = norm[irec, 0]
        step = norm[irec, 1]
        if flag_state:
            wgt = np.float32(1.0)
            rot /= abs(rot)
        else:
            wgt = np.float32(wt_val[irec])

        start += 1
        if (rot.imag == 0.0) and (step == 1.0):
            # Cast explicitly here to preverve backwards compatibility/behavior with
            # testing against prior numpy-based behavior.
            sfac *= np.float32(rot.real)
            for jdx in range(nchan[irec]):
                re_val = packdata[start + (2 * jdx)]
                sub_off = off + jdx
                if re_val == BAD_VAL:
                    # Only the real component is checked, but the writer marks both.
                    rec_data[sub_off] = np.float32(0)
                    rec_flags[sub_off] = True
                    rec_wts[sub_off] = np.float32(0.0)
                else:
                    im_val = packdata[start + (2 * jdx) + 1]
                    # Complex turns out to be the most efficient way to cast this.
                    rec_data[sub_off] = complex(re_val * sfac, im_val * sfac)
                    rec_flags[sub_off] = flag_state
                    rec_wts[sub_off] = wgt
        else:
            # Step the phasor along in double precision, which keeps the accumulated
            # error at ~Nchan * 1e-16 rather than the ~Nchan * 1e-7 of single precision.
            rot *= sfac
            for jdx in range(nchan[irec]):
                re_val = packdata[start + (2 * jdx)]
                sub_off = off + jdx
                if re_val == BAD_VAL:
                    rec_data[sub_off] = np.float32(0)
                    rec_flags[sub_off] = True
                    rec_wts[sub_off] = np.float32(0.0)
                else:
                    im_val = packdata[start + (2 * jdx) + 1]
                    rec_data[sub_off] = complex(re_val, im_val) * rot
                    rec_flags[sub_off] = flag_state
                    rec_wts[sub_off] = wgt
                rot *= step

        # Then any bandpass solutions and channel flags, applied channel by channel as
        # the record is on disk (i.e., ahead of any shift in frequency or averaging).
        if bp_gain is not None:
            row = bp_idx[irec]
            if row < 0:
                # No solution for this record, so it is flagged outright.
                for jdx in range(off, off + nchan[irec]):
                    rec_wts[jdx] *= np.float32(0)
                    rec_flags[jdx] = True
            else:
                for jdx in range(nchan[irec]):
                    rec_data[off + jdx] *= bp_gain[row, jdx]
                    rec_wts[off + jdx] *= bp_weight[row, jdx]
                    rec_flags[off + jdx] |= bp_flags[row, jdx]
        if flag_bytes is not None:
            # Worked through a byte at a time, since most hold no flags at all.
            for bdx in range(nchan[irec] // 8):
                flag_byte = flag_bytes[irec, bdx]
                if flag_byte != 0:
                    for kdx in range(8):
                        if (flag_byte >> kdx) & 1:
                            rec_flags[off + (8 * bdx) + kdx] = True

        if chan_avg is None:
            continue

        nch = nchan[irec]
        shift = chan_shift[irec]
        frac = chan_frac[irec]
        interp = frac != 0
        if interp:
            # Cubic convolution kernel, with alpha = -0.5 -- see the "cubic" branch of
            # `MirParser._generate_chanshift_kernel`.
            kern = np.empty(4, dtype=np.float32)
            kern[0] = (
                (-0.5 * ((1 + frac) ** 3))
                + (2.5 * ((1 + frac) ** 2))
                - (4 * (1 + frac))
                + 2
            )
            kern[1] = (1.5 * (frac**3)) - (2.5 * (frac**2)) + 1
            kern[2] = (1.5 * ((1 - frac) ** 3)) - (2.5 * ((1 - frac) ** 2)) + 1
            kern[3] = (
                (-0.5 * ((2 - frac) ** 3))
                + (2.5 * ((2 - frac) ** 2))
                - (4 * (2 - frac))
                + 2
            )

        # Shift first, at the resolution of the input. Without averaging, the shifted
        # record goes straight to the output. Otherwise, each block of channels is
        # summed as the loop goes, weighting every unflagged channel.
        avg = chan_avg[irec]
        ooff = out_off[irec]
        for odx in range(nch // avg):
            # Zero out the buffers for the current averaging block.
            vis_sum = np.complex128(0)
            wt_sum = np.float64(0)
            # Loop over the channels inside the current averaging block.
            for jdx in range(odx * avg, (odx + 1) * avg):
                src = jdx - shift
                ch_vis = np.complex64(0)
                ch_wt = np.float32(0)
                flagged = True
                if interp:
                    # The kernel runs from one channel above the source to two below.
                    if (src >= 2) and (src <= (nch - 2)):
                        flagged = False
                        # Loop over the 4 points in the cubic convolution kernel.
                        for kdx in range(4):
                            cdx = src + 1 - kdx
                            # Bail on a flag
                            if rec_flags[cdx]:
                                flagged = True
                                break
                            ch_vis += kern[kdx] * rec_data[cdx]
                            ch_wt += kern[kdx] * rec_wts[cdx]
                        # If anything is flagged, skip interpolating this channel
                        if flagged:
                            ch_vis = np.complex64(0)
                            ch_wt = np.float32(0)
                elif (src >= 0) and (src < nch):
                    # If not interpolating, just pull directly from the records.
                    ch_vis = rec_data[src]
                    ch_wt = rec_wts[src]
                    flagged = rec_flags[src]

                if avg == 1:
                    data_out[ooff + jdx] = ch_vis
                    wt_out[ooff + jdx] = ch_wt
                    flag_out[ooff + jdx] = flagged
                elif not flagged:
                    vis_sum += ch_vis * np.float64(ch_wt)
                    wt_sum += ch_wt

            if avg > 1:
                data_out[ooff + odx] = (vis_sum / wt_sum) if (wt_sum != 0) else 0
                wt_out[ooff + odx] = wt_sum / avg
                flag_out[ooff + odx] = wt_out[ooff + odx] == 0


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
