# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for the MIR packed-data unpacking kernels."""

import numpy as np
import pytest

from pyuvdata.uvdata import mir_numba

NCHAN = 64
NREC = 20


def make_packdata(nchan=NCHAN, nrec=NREC, seed=0, flag_frac=0.1):
    """Build a commonly-scaled packdata block plus its indexing arrays."""
    rng = np.random.default_rng(seed)
    nvals = 1 + (2 * nchan)
    packdata = rng.integers(-2000, 2000, nrec * nvals, dtype=np.int16)
    sidx = (np.arange(nrec) * nvals).astype(np.int64)
    # Common exponents, one per record.
    packdata[sidx] = rng.integers(-30, -10, nrec).astype(np.int16)

    # Flag whole channels by marking both components, as the MIR writer does.
    view = packdata.reshape(nrec, nvals)
    flagged = rng.random((nrec, nchan)) < flag_frac
    view[:, 1::2][flagged] = mir_numba.BAD_VAL
    view[:, 2::2][flagged] = mir_numba.BAD_VAL

    nchan_arr = np.full(nrec, nchan, dtype=np.int64)
    out_off = (np.arange(nrec) * nchan).astype(np.int64)
    return packdata, sidx, nchan_arr, out_off


def reference_scaled(packdata, sidx, nchan_arr, norm, wt_val, all_flag):
    # This is the old numpy-based way these records were written in, included now for
    # the sake of verifying backwards compatibility.
    data, flags, weights = [], [], []
    for irec, (start, nchan) in enumerate(zip(sidx, nchan_arr, strict=True)):
        raw = packdata[start + 1 : start + 1 + (2 * nchan)]
        rec_data = (np.exp2(np.float32(packdata[start])) * raw).view(np.complex64)
        rec_flags = raw[::2] == mir_numba.BAD_VAL
        rec_data[rec_flags] = 0.0
        rec_weights = (~rec_flags).astype(np.float32)
        if all_flag[irec]:
            # No usable normalization: flag the record, leave the values unscaled.
            rec_flags = np.ones_like(rec_flags)
        else:
            # Cast to match the kernel.
            rec_data *= np.float32(norm[irec])
            rec_weights *= np.float32(wt_val[irec])
        data.append(rec_data)
        flags.append(rec_flags)
        weights.append(rec_weights)
    return np.concatenate(data), np.concatenate(flags), np.concatenate(weights)


def run_scaled(kernel, packdata, sidx, nchan_arr, out_off, norm, wt_val, all_flag):
    ntot = int(nchan_arr.sum())
    data = np.empty(ntot, np.complex64)
    flags = np.empty(ntot, bool)
    weights = np.empty(ntot, np.float32)
    kernel(
        packdata, sidx, nchan_arr, out_off, norm, wt_val, all_flag, data, flags, weights
    )
    return data, flags, weights


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("add_flags", [True, False])
def test_unpack_scaled(dtype, add_flags):
    # Verify that everything works just like how it used to.
    packdata, sidx, nchan_arr, out_off = make_packdata()
    norm = np.random.uniform(0.5, 2.0, NREC).astype(dtype)
    wt_val = np.random.uniform(0.5, 2.0, NREC).astype(dtype)
    all_flag = np.zeros(NREC, bool)
    if add_flags:
        all_flag[::3] = True

    numba_data, numba_flags, numba_weights = run_scaled(
        mir_numba.unpack_scaled,
        packdata,
        sidx,
        nchan_arr,
        out_off,
        norm,
        wt_val,
        all_flag,
    )
    ref_data, ref_flags, ref_weights = reference_scaled(
        packdata, sidx, nchan_arr, norm, wt_val, all_flag
    )

    assert np.array_equal(numba_data, ref_data)
    assert np.array_equal(numba_flags, ref_flags)
    assert np.array_equal(numba_weights, ref_weights)

    if add_flags:
        assert np.all(numba_flags.reshape(NREC, NCHAN)[::3])


def test_unpack_scaled_ragged():
    # Make sure things work w/ the pseudo-cont/spectral ragged structure
    nchan_arr = np.array([4, 64, 64, 4, 64], dtype=np.int64)
    nvals = 1 + (2 * nchan_arr)
    sidx = np.concatenate(([0], np.cumsum(nvals)[:-1])).astype(np.int64)
    out_off = np.concatenate(([0], np.cumsum(nchan_arr)[:-1])).astype(np.int64)
    packdata = np.random.randint(-500, 500, int(nvals.sum())).astype(np.int16)
    # Make the common exponent something "normal"
    packdata[sidx] = -20

    # Populate some values here for the call
    norm = np.ones(len(nchan_arr), np.float32)
    wt_val = np.ones(len(nchan_arr), np.float32)
    all_flag = np.zeros(len(nchan_arr), bool)

    numba_data, numba_flags, numba_weights = run_scaled(
        mir_numba.unpack_scaled,
        packdata,
        sidx,
        nchan_arr,
        out_off,
        norm,
        wt_val,
        all_flag,
    )
    ref_data, ref_flags, ref_weights = reference_scaled(
        packdata, sidx, nchan_arr, norm, wt_val, all_flag
    )

    assert np.array_equal(numba_data, ref_data)
    assert np.array_equal(numba_flags, ref_flags)
    assert np.array_equal(numba_weights, ref_weights)


def test_copy_records_autos():
    nvals_arr = np.full(NREC, NCHAN, dtype=np.int64)
    sidx = (np.arange(NREC) * NCHAN).astype(np.int64)
    packdata = np.random.standard_normal(NREC * NCHAN).astype(np.float32)
    packdata[::17] = np.nan

    data = np.empty(NREC * NCHAN, np.float32)
    mir_numba.copy_records(packdata, sidx, nvals_arr, sidx, data)

    assert np.array_equal(data, packdata, equal_nan=True)


def test_copy_records_raw():
    nvals_arr = np.array([13, 129, 129, 13], dtype=np.int64)
    sidx = np.concatenate(([0], np.cumsum(nvals_arr)[:-1])).astype(np.int64)
    out_off = sidx.copy()
    packdata = np.random.randint(-100, 100, int(nvals_arr.sum())).astype(np.int16)
    out = np.empty(int(nvals_arr.sum()), np.int16)

    mir_numba.copy_records(packdata, sidx, nvals_arr, out_off, out)
    assert np.array_equal(out, packdata)
    # A genuine copy, not a view onto the source.
    out[0] += 1
    assert out[0] != packdata[0]


def test_copy_records_subset():
    packdata = np.arange(100, dtype=np.int16)
    sidx = np.array([10, 50], dtype=np.int64)
    nvals_arr = np.array([5, 5], dtype=np.int64)
    out_off = np.array([0, 5], dtype=np.int64)
    out = np.empty(10, np.int16)

    mir_numba.copy_records(packdata, sidx, nvals_arr, out_off, out)
    assert np.array_equal(out, np.concatenate([np.arange(10, 15), np.arange(50, 55)]))
