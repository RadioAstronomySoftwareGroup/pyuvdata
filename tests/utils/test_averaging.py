# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for the masked averaging utilities."""

import numpy as np
import pytest

from pyuvdata.utils.averaging import mapped_average


@pytest.fixture
def rng():
    return np.random.default_rng(314)


def _reference(
    data, index_map, flags, weights, n_bins, sum_mode, propagate_flags, weighted=True
):
    """Slow, obvious implementation of mapped_average for a 3D array on axis 1."""
    n_rows, _, n_cols = data.shape
    avg = np.zeros((n_rows, n_bins, n_cols), dtype=data.dtype)
    flg = np.zeros((n_rows, n_bins, n_cols), dtype=bool)
    wgt = np.zeros((n_rows, n_bins, n_cols), dtype=np.float64)
    for row in range(n_rows):
        for bin_idx in range(n_bins):
            contrib = np.nonzero(index_map == bin_idx)[0]
            for col in range(n_cols):
                good = contrib[~flags[row, contrib, col]]
                use = good if good.size else contrib
                vals = data[row, use, col]
                wts = weights[row, use, col].astype(np.float64)
                num = np.sum(vals * wts) if weighted else np.sum(vals)
                den = np.sum(wts) if weighted else use.size
                avg[row, bin_idx, col] = num if sum_mode else num / den
                wgt[row, bin_idx, col] = np.sum(wts)
                flg[row, bin_idx, col] = (
                    (good.size != contrib.size) if propagate_flags else (good.size == 0)
                )
    return avg, flg, wgt


def _make(rng, n_rows=4, n_axis=12, n_cols=2, flag_frac=0.3):
    data = rng.normal(size=(n_rows, n_axis, n_cols)) + 1j * rng.normal(
        size=(n_rows, n_axis, n_cols)
    )
    flags = rng.uniform(size=data.shape) < flag_frac
    weights = rng.uniform(0.5, 2.0, size=data.shape)
    return data, flags, weights


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("sum_mode", [False, True])
@pytest.mark.parametrize("propagate_flags", [False, True])
def test_mapped_average_matches_reference(rng, weighted, sum_mode, propagate_flags):
    data, flags, weights = _make(rng)
    index_map = np.arange(12) // 4

    avg, flg, wgt = mapped_average(
        data,
        index_map=index_map,
        flags=flags,
        weights=weights,
        axis=1,
        weighted=weighted,
        sum_mode=sum_mode,
        propagate_flags=propagate_flags,
    )
    exp_avg, exp_flg, exp_wgt = _reference(
        data, index_map, flags, weights, 3, sum_mode, propagate_flags, weighted
    )

    np.testing.assert_allclose(avg, exp_avg)
    np.testing.assert_array_equal(flg, exp_flg)
    np.testing.assert_allclose(wgt, exp_wgt)


def test_mapped_average_unweighted(rng):
    """An unweighted average ignores the weights but still reports their sum."""
    data, flags, weights = _make(rng, flag_frac=0.0)
    index_map = np.arange(12) // 4
    # Make the weights lopsided enough that the two answers cannot coincide.
    weights[:] = 1.0
    weights[:, 0::4] = 100.0

    kwargs = {"index_map": index_map, "flags": flags, "weights": weights, "axis": 1}
    flat, _, flat_wgt = mapped_average(data, weighted=False, **kwargs)
    wtd, _, wtd_wgt = mapped_average(data, weighted=True, **kwargs)

    np.testing.assert_allclose(flat, data.reshape(4, 3, 4, 2).mean(axis=2))
    assert not np.allclose(flat, wtd)
    # The summed weights are reported the same way either way, since they are what the
    # caller needs to track how much data went into each bin.
    np.testing.assert_allclose(flat_wgt, wtd_wgt)


def test_mapped_average_unweighted_respects_flags(rng):
    """A flat average still only counts the unflagged contributors."""
    data, flags, weights = _make(rng, flag_frac=0.0)
    index_map = np.arange(12) // 4
    flags[0, 0:2, 0] = True

    flat, _, _ = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=1, weighted=False
    )
    # Only the two unflagged entries of that bin should have been counted.
    np.testing.assert_allclose(flat[0, 0, 0], np.mean(data[0, 2:4, 0]))


def test_mapped_average_axis_weights(rng):
    """axis_weights must fold in exactly as if broadcast into the weights array."""
    data, flags, weights = _make(rng)
    index_map = np.arange(12) // 4
    axis_weights = rng.uniform(0.5, 4.0, size=12)

    avg, flg, wgt = mapped_average(
        data,
        index_map=index_map,
        flags=flags,
        weights=weights,
        axis_weights=axis_weights,
        axis=1,
    )
    exp_avg, exp_flg, exp_wgt = _reference(
        data,
        index_map,
        flags,
        weights * axis_weights[np.newaxis, :, np.newaxis],
        3,
        False,
        False,
    )
    np.testing.assert_allclose(avg, exp_avg)
    np.testing.assert_array_equal(flg, exp_flg)
    np.testing.assert_allclose(wgt, exp_wgt)

    # All-ones must be identical to not passing it at all.
    plain = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=1
    )
    ones = mapped_average(
        data,
        index_map=index_map,
        flags=flags,
        weights=weights,
        axis_weights=np.ones(12),
        axis=1,
    )
    for got, want in zip(ones, plain, strict=True):
        np.testing.assert_allclose(got, want)


def test_mapped_average_axis_weights_unweighted(rng):
    """axis_weights still feed the reported weight sum when weighted is False."""
    data, flags, weights = _make(rng, flag_frac=0.0)
    index_map = np.arange(12) // 4
    axis_weights = rng.uniform(0.5, 4.0, size=12)

    avg, _, wgt = mapped_average(
        data,
        index_map=index_map,
        flags=flags,
        weights=weights,
        axis_weights=axis_weights,
        axis=1,
        weighted=False,
    )
    # The average itself ignores both sets of weights ...
    np.testing.assert_allclose(avg, data.reshape(4, 3, 4, 2).mean(axis=2))
    # ... but the summed weights still include the axis scaling.
    scaled = weights * axis_weights[np.newaxis, :, np.newaxis]
    np.testing.assert_allclose(wgt, scaled.reshape(4, 3, 4, 2).sum(axis=2))


def test_mapped_average_fully_flagged_bin(rng):
    """A bin with nothing unflagged should still use its (flagged) contributors."""
    data, flags, weights = _make(rng, flag_frac=0.0)
    index_map = np.arange(12) // 4
    flags[1, 4:8, 0] = True

    avg, flg, _ = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=1
    )

    expected = np.sum(data[1, 4:8, 0] * weights[1, 4:8, 0]) / np.sum(weights[1, 4:8, 0])
    np.testing.assert_allclose(avg[1, 1, 0], expected)
    # The result is flagged, but it is not zero.
    assert flg[1, 1, 0]
    assert avg[1, 1, 0] != 0
    # Nothing else got flagged along the way.
    assert np.count_nonzero(flg) == 1


def test_mapped_average_interleaved_bins(rng):
    """Bins need not be contiguous along the averaged axis."""
    data, flags, weights = _make(rng, n_axis=16)
    index_map = np.tile([0, 1, 2, 3], 4)

    avg, flg, wgt = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=1
    )

    # Sorting the axis so that each bin is contiguous must give the same answer.
    order = np.argsort(index_map, kind="stable")
    avg2, flg2, wgt2 = mapped_average(
        np.ascontiguousarray(data[:, order]),
        index_map=index_map[order],
        flags=np.ascontiguousarray(flags[:, order]),
        weights=np.ascontiguousarray(weights[:, order]),
        axis=1,
    )
    np.testing.assert_allclose(avg, avg2)
    np.testing.assert_array_equal(flg, flg2)
    np.testing.assert_allclose(wgt, wgt2)


def test_mapped_average_dropped_and_ragged(rng):
    """Negative entries drop values, and bins can be unevenly sized."""
    data, flags, weights = _make(rng, n_axis=10, flag_frac=0.0)
    # First bin gets 4 entries, second gets 3, and 3 entries are dropped entirely.
    index_map = np.array([0, 0, 0, 0, 1, 1, 1, -1, -1, -1])

    avg, _, wgt = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=1
    )

    assert avg.shape == (4, 2, 2)
    for bin_idx, sel in enumerate([slice(0, 4), slice(4, 7)]):
        expected = np.sum(data[:, sel] * weights[:, sel], axis=1) / np.sum(
            weights[:, sel], axis=1
        )
        np.testing.assert_allclose(avg[:, bin_idx], expected)
        np.testing.assert_allclose(wgt[:, bin_idx], np.sum(weights[:, sel], axis=1))


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_mapped_average_axis(rng, axis):
    """Averaging along any axis should match moving that axis to the middle."""
    data, flags, weights = _make(rng, n_rows=6, n_axis=6, n_cols=6)
    index_map = np.arange(6) // 2

    avg, flg, wgt = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=axis
    )

    norm_axis = range(data.ndim)[axis]
    exp_avg, exp_flg, exp_wgt = _reference(
        np.ascontiguousarray(np.moveaxis(data, norm_axis, 1)),
        index_map,
        np.ascontiguousarray(np.moveaxis(flags, norm_axis, 1)),
        np.ascontiguousarray(np.moveaxis(weights, norm_axis, 1)),
        3,
        False,
        False,
    )
    np.testing.assert_allclose(np.moveaxis(avg, norm_axis, 1), exp_avg)
    np.testing.assert_array_equal(np.moveaxis(flg, norm_axis, 1), exp_flg)
    np.testing.assert_allclose(np.moveaxis(wgt, norm_axis, 1), exp_wgt)


@pytest.mark.parametrize("n_rows", [1, 2, 200])
def test_mapped_average_block_decomposition(rng, n_rows):
    """Results must not depend on how the work gets split across threads.

    Averaging along the leading axis leaves nothing to split up by row, so the work has
    to be blocked by column instead -- which is the case that matters for averaging
    along a time axis.
    """
    data, flags, weights = _make(rng, n_rows=n_rows, n_axis=8, n_cols=64)
    index_map = np.arange(8) // 2

    # axis=1 with one row exercises pure column blocking, and with many rows exercises
    # pure row blocking. Both must agree with the serial reference.
    avg, flg, wgt = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=1
    )
    exp_avg, exp_flg, exp_wgt = _reference(
        data, index_map, flags, weights, 4, False, False
    )
    np.testing.assert_allclose(avg, exp_avg)
    np.testing.assert_array_equal(flg, exp_flg)
    np.testing.assert_allclose(wgt, exp_wgt)

    # Averaging along axis 0 is the shape a time average takes, and always collapses to
    # a single row internally no matter how big the array is.
    flat = data.reshape(-1, data.shape[-1])
    flat_flags = flags.reshape(-1, data.shape[-1])
    flat_weights = weights.reshape(-1, data.shape[-1])
    time_map = np.arange(flat.shape[0]) // 4
    avg0, flg0, wgt0 = mapped_average(
        flat, index_map=time_map, flags=flat_flags, weights=flat_weights, axis=0
    )
    exp0, expf0, expw0 = _reference(
        flat[np.newaxis],
        time_map,
        flat_flags[np.newaxis],
        flat_weights[np.newaxis],
        int(time_map.max()) + 1,
        False,
        False,
    )
    np.testing.assert_allclose(avg0, exp0[0])
    np.testing.assert_array_equal(flg0, expf0[0])
    np.testing.assert_allclose(wgt0, expw0[0])


def test_mapped_average_n_bins(rng):
    """An explicit n_bins can leave trailing bins with no contributors."""
    data, flags, weights = _make(rng, n_axis=8)
    index_map = np.arange(8) // 4

    avg, flg, wgt = mapped_average(
        data, index_map=index_map, flags=flags, weights=weights, axis=1, n_bins=3
    )
    assert avg.shape[1] == 3
    # The empty bin has nothing in it, and is reported as zeroed and flagged.
    np.testing.assert_allclose(avg[:, 2], 0)
    np.testing.assert_allclose(wgt[:, 2], 0)
    assert np.all(flg[:, 2])


def test_mapped_average_float16_weights(rng):
    """Half-precision weights get promoted rather than handed to numba."""
    data, flags, weights = _make(rng, flag_frac=0.0)
    index_map = np.arange(12) // 4

    avg, _, wgt = mapped_average(
        data,
        index_map=index_map,
        flags=flags,
        weights=weights.astype(np.float16),
        axis=1,
    )
    exp_avg, _, exp_wgt = _reference(
        data, index_map, flags, weights.astype(np.float16), 3, False, False
    )
    np.testing.assert_allclose(avg, exp_avg)
    # The weights are summed at full precision and only rounded back down to
    # half-precision on the way out, so that is what they have to be compared against.
    assert wgt.dtype == np.float16
    np.testing.assert_allclose(wgt, exp_wgt.astype(np.float16))


def test_mapped_average_preserves_dtype(rng):
    """Results come back at the input precision, having accumulated at full."""
    data, flags, weights = _make(rng)
    index_map = np.arange(12) // 4

    avg, _, wgt = mapped_average(
        data.astype(np.complex64),
        index_map=index_map,
        flags=flags,
        weights=weights.astype(np.float32),
        axis=1,
    )
    assert avg.dtype == np.complex64
    # Weights come back at the precision they were handed in at, whatever they were
    # accumulated (and optionally normalized) at along the way.
    assert wgt.dtype == np.float32


def test_mapped_average_no_promotion(rng):
    """allow_promotion=False accumulates at the stored precision instead."""
    data, flags, weights = _make(rng, n_axis=64, flag_frac=0.0)
    index_map = np.zeros(64, dtype=int)  # one big bin, so precision actually shows

    kwargs = {
        "index_map": index_map,
        "flags": flags,
        "weights": weights.astype(np.float32),
        "axis": 1,
    }
    low, _, low_wgt = mapped_average(
        data.astype(np.complex64), allow_promotion=False, **kwargs
    )
    high, _, high_wgt = mapped_average(
        data.astype(np.complex64), allow_promotion=True, **kwargs
    )

    # Both come back at the input precision either way -- allow_promotion controls
    # what happens during the accumulation, not what is handed back.
    assert low.dtype == high.dtype == np.complex64
    assert low_wgt.dtype == high_wgt.dtype == np.float32

    # The promoted result is the more accurate of the two.
    exact, _, _ = _reference(data, index_map, flags, weights, 1, False, False)
    assert np.abs(high - exact).max() <= np.abs(low - exact).max()


def test_mapped_average_real_data(rng):
    """The kernel handles real-valued data as well as complex."""
    data, flags, weights = _make(rng)
    index_map = np.arange(12) // 4

    avg, _, _ = mapped_average(
        data.real.copy(), index_map=index_map, flags=flags, weights=weights, axis=1
    )
    exp_avg, _, _ = _reference(
        data.real.copy(), index_map, flags, weights, 3, False, False
    )
    assert avg.dtype == np.float64
    np.testing.assert_allclose(avg, exp_avg)


def test_mapped_average_errors(rng):
    data, flags, weights = _make(rng)
    index_map = np.arange(12) // 4

    with pytest.raises(ValueError, match="must all have the same shape"):
        mapped_average(
            data, index_map=index_map, flags=flags[:, :-1], weights=weights, axis=1
        )

    with pytest.raises(ValueError, match="must all have the same shape"):
        mapped_average(
            data, index_map=index_map, flags=flags, weights=weights[:, :-1], axis=1
        )

    with pytest.raises(ValueError, match="same length as data along the averaged axis"):
        mapped_average(
            data, index_map=index_map[:-1], flags=flags, weights=weights, axis=1
        )

    with pytest.raises(
        ValueError, match="axis_weights must have the same length as data"
    ):
        mapped_average(
            data,
            index_map=index_map,
            flags=flags,
            weights=weights,
            axis_weights=np.ones(11),
            axis=1,
        )
