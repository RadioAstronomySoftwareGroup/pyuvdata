# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for the MIR packed-data unpacking kernels."""

import numpy as np
import pytest

from pyuvdata.uvdata import mir_numba
from pyuvdata.uvdata.mir_parser import MirParser

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
@pytest.mark.parametrize("ramp", [False, True])
def test_unpack_scaled(dtype, add_flags, ramp):
    # Verify that everything works just like how it used to, and that a phase ramp
    # folded into the normalization matches one applied after the fact.
    packdata, sidx, nchan_arr, out_off = make_packdata()
    norm_val = np.random.uniform(0.5, 2.0, NREC).astype(dtype)
    wt_val = np.random.uniform(0.5, 2.0, NREC).astype(dtype)
    all_flag = np.zeros(NREC, bool)
    if add_flags:
        all_flag[::3] = True
    # Phase of the first channel, and the step from one channel to the next
    pha_arr = np.zeros((NREC, 2))
    if ramp:
        pha_arr = np.random.uniform(-np.pi, np.pi, (NREC, 2)) * [1.0, 1e-2]
    norm = np.stack([norm_val, np.ones(NREC)], axis=1) * np.exp(1j * pha_arr)

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
        packdata, sidx, nchan_arr, norm_val, wt_val, all_flag
    )

    if ramp:
        # Flagged records still take the phase, just not the normalization
        ref_data = ref_data * np.exp(
            1j * (pha_arr[:, :1] + pha_arr[:, 1:] * np.arange(NCHAN))
        ).ravel().astype(np.complex64)
        np.testing.assert_allclose(numba_data, ref_data, rtol=1e-6)
    else:
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
    norm = np.ones((len(nchan_arr), 2), np.complex128)
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
        packdata, sidx, nchan_arr, norm[:, 0].real, wt_val, all_flag
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


@pytest.mark.parametrize("chan_avg", [1, 2, 30, 240])
@pytest.mark.parametrize("chan_shift,chan_frac", [[0, 0], [3, 0], [-5, 0], [2, 0.3]])
def test_unpack_scaled_regrid(chan_avg, chan_shift, chan_frac):
    """Shifting and averaging while unpacking matches _chanshift_vis + _rechunk_data."""
    nchan = 480
    packdata, sidx, nchan_arr, out_off = make_packdata(nchan=nchan)
    # Common exponents that leave values of order unity, to compare against.
    packdata[sidx] = -10
    norm = np.ones((NREC, 2), np.complex128)
    wt_val = np.random.default_rng(0).uniform(0.5, 2.0, NREC).astype(np.float32)
    wt_val[1] = 0.0
    all_flag = np.zeros(NREC, bool)
    all_flag[0] = True
    data, flags, weights = run_scaled(
        mir_numba.unpack_scaled,
        packdata,
        sidx,
        nchan_arr,
        out_off,
        norm,
        wt_val,
        all_flag,
    )

    shift_tuple = MirParser._generate_chanshift_kernel(
        chan_shift + chan_frac, "cubic", tol=0
    )
    ref_dict = MirParser._chanshift_vis(
        {
            irec: {
                "data": data.reshape(NREC, nchan)[irec],
                "flags": flags.reshape(NREC, nchan)[irec],
                "weights": weights.reshape(NREC, nchan)[irec],
            }
            for irec in range(NREC)
        },
        [shift_tuple] * NREC,
    )
    ref_dict = MirParser._rechunk_data(ref_dict, [chan_avg] * NREC)

    nout = nchan // chan_avg
    out_data = np.empty(NREC * nout, np.complex64)
    out_flags = np.empty(NREC * nout, bool)
    out_weights = np.empty(NREC * nout, np.float32)
    mir_numba.unpack_scaled(
        packdata,
        sidx,
        nchan_arr,
        np.arange(NREC) * nout,
        norm,
        wt_val,
        all_flag,
        out_data,
        out_flags,
        out_weights,
        chan_shift=np.full(NREC, chan_shift),
        chan_frac=np.full(NREC, chan_frac, dtype=float),
        chan_avg=np.full(NREC, chan_avg),
    )

    ref_data = np.concatenate([rec["data"] for rec in ref_dict.values()])
    ref_flags = np.concatenate([rec["flags"] for rec in ref_dict.values()])
    ref_weights = np.concatenate([rec["weights"] for rec in ref_dict.values()])
    assert np.array_equal(out_flags, ref_flags)
    # Zero weights only lead to flags once channels are averaged together.
    assert np.all(out_flags.reshape(NREC, nout)[: (1 if chan_avg == 1 else 2)])
    # Summed in double precision (and any interpolation done in a different order), so
    # these agree to within rounding.
    np.testing.assert_allclose(out_data, ref_data, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_weights, ref_weights, rtol=1e-5)


@pytest.mark.parametrize("apply_bp", [True, False])
@pytest.mark.parametrize("add_flags", [True, False])
def test_unpack_scaled_cal(apply_bp, add_flags):
    """Bandpass solutions and packed flags apply as numpy would, record by record."""
    rng = np.random.default_rng(1)
    packdata, sidx, nchan_arr, out_off = make_packdata()
    norm = np.ones((NREC, 2), np.complex128)
    wt_val = rng.uniform(0.5, 2.0, NREC).astype(np.float32)
    all_flag = np.zeros(NREC, bool)
    data, flags, weights = run_scaled(
        mir_numba.unpack_scaled,
        packdata,
        sidx,
        nchan_arr,
        out_off,
        norm,
        wt_val,
        all_flag,
    )

    # Three solutions, with every fourth record having none (and so flagged).
    bp_gain = (rng.standard_normal((3, NCHAN, 2))).astype(np.float32)
    bp_gain = bp_gain.view(np.complex64).reshape(3, NCHAN)
    bp_weight = rng.uniform(0.5, 2.0, (3, NCHAN)).astype(np.float32)
    bp_flags = rng.random((3, NCHAN)) < 0.1
    bp_idx = np.arange(NREC) % 4 - 1

    # Packed flags (least significant bit first), w/ every other record left empty.
    pack_flags = rng.random((NREC, NCHAN)) < 0.2
    pack_flags[1::2] = False
    flag_bytes = np.packbits(pack_flags, axis=1, bitorder="little")

    ref_data = data.reshape(NREC, NCHAN).copy()
    ref_flags = flags.reshape(NREC, NCHAN).copy()
    ref_weights = weights.reshape(NREC, NCHAN).copy()
    for irec in range(NREC):
        if apply_bp and bp_idx[irec] < 0:
            ref_data[irec] *= 1.0
            ref_weights[irec] *= 0.0
            ref_flags[irec] |= True
        elif apply_bp:
            ref_data[irec] *= bp_gain[bp_idx[irec]]
            ref_weights[irec] *= bp_weight[bp_idx[irec]]
            ref_flags[irec] |= bp_flags[bp_idx[irec]]
        if add_flags:
            ref_flags[irec] |= pack_flags[irec]

    compass_cal = {
        "bp_idx": bp_idx,
        "bp_gain": bp_gain if apply_bp else None,
        "bp_weight": bp_weight if apply_bp else None,
        "bp_flags": bp_flags if apply_bp else None,
        "flag_bytes": flag_bytes if add_flags else None,
    }
    mir_numba.unpack_scaled(
        packdata,
        sidx,
        nchan_arr,
        out_off,
        norm,
        wt_val,
        all_flag,
        data,
        flags,
        weights,
        **compass_cal,
    )

    # Where numpy runs complex multiplication with fused multiply-adds (e.g., on AVX2),
    # the bandpass products can round differently, by ~1 ulp of their magnitude.
    np.testing.assert_allclose(data, ref_data.ravel(), rtol=2.5e-7)
    assert np.array_equal(flags, ref_flags.ravel())
    assert np.array_equal(weights.view(np.uint32), ref_weights.ravel().view(np.uint32))
