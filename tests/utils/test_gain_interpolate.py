# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for gain interpolation utility functions.

Performs a series of tests on the routines used to interpolate calibration solutions
onto a new set of times, covering the individual interpolation kernels, the flagging
that accompanies them, and the user-facing entry point that ties the two together.
"""

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import pyuvdata.utils.gain_interpolate as gi_utils

# Array dimensions used throughout, kept small so that the jit-disabled CI job, which
# runs everything through the python interpreter, stays quick.
NANTS = 3
NFREQS = 2
NTIMES = 12
NJONES = 2
NNEW = 7


@pytest.fixture(scope="module")
def old_times():
    return np.linspace(0.0, 1.0, NTIMES)


@pytest.fixture(scope="module")
def new_times():
    # Interior only, so that the extrapolation flags stay clear unless a test asks
    # for them.
    return np.linspace(0.05, 0.95, NNEW)


@pytest.fixture(scope="module")
def gain_array(old_times):
    """Smooth complex gains, varying in both amplitude and phase."""
    amp = 1.0 + 0.3 * np.sin(2 * np.pi * old_times)
    pha = 0.5 * np.cos(2 * np.pi * old_times)
    gains = amp * np.exp(1j * pha)
    return np.broadcast_to(
        gains[None, None, :, None], (NANTS, NFREQS, NTIMES, NJONES)
    ).copy()


@pytest.fixture(scope="module")
def delay_array(old_times):
    """Smooth real-valued solutions, standing in for delays."""
    vals = 2.0 - (3.0 * old_times) + (5.0 * old_times**2)
    return np.broadcast_to(
        vals[None, None, :, None], (NANTS, NFREQS, NTIMES, NJONES)
    ).copy()


@pytest.fixture(scope="module")
def blank_flags():
    return np.zeros((NANTS, NFREQS, NTIMES, NJONES), dtype=bool)


def make_flags(partial=(), dead=()):
    """Build a flag array, partly flagging some solutions and fully flagging others."""
    flags = np.zeros((NANTS, NFREQS, NTIMES, NJONES), dtype=bool)
    for ant, freq, jones in partial:
        flags[ant, freq, 2:5, jones] = True
    for ant, freq, jones in dead:
        flags[ant, freq, :, jones] = True
    return flags


def test_flag_outlier_cal_spike(delay_array):
    """An isolated outlier is caught, and nothing else is."""
    cal_array = delay_array.copy()
    cal_array[0, 0, 5, 0] += 10.0

    new_flags = gi_utils._flag_outlier_cal(cal_array, 2.0)

    assert new_flags[0, 0, 5, 0]
    new_flags[0, 0, 5, 0] = False
    assert not new_flags.any()


def test_flag_outlier_cal_smooth(delay_array):
    assert not gi_utils._flag_outlier_cal(delay_array, 2.0).any()


def test_flag_outlier_cal_nan(delay_array):
    cal_array = delay_array.copy()
    cal_array[1, 1, 7, 1] = np.nan

    assert gi_utils._flag_outlier_cal(cal_array, 2.0)[1, 1, 7, 1]


def test_flag_outlier_cal_edges(delay_array):
    """Outliers at the first and last sample are caught too."""
    cal_array = delay_array.copy()
    cal_array[0, 0, 0, 0] += 10.0
    cal_array[0, 0, -1, 0] += 10.0

    new_flags = gi_utils._flag_outlier_cal(cal_array, 2.0)

    assert new_flags[0, 0, 0, 0]
    assert new_flags[0, 0, -1, 0]


@pytest.mark.parametrize("is_gain_amp,is_gain_pha", [[True, False], [False, True]])
def test_flag_outlier_cal_gain(gain_array, is_gain_amp, is_gain_pha):
    """Both gain modes see a large excursion in amp and phase."""
    cal_array = gain_array.copy()
    cal_array[0, 1, 6, 0] *= 5.0 * np.exp(1j * 2.0)

    new_flags = gi_utils._flag_outlier_cal(
        cal_array, 0.5, is_gain_amp=is_gain_amp, is_gain_pha=is_gain_pha
    )

    assert new_flags[0, 1, 6, 0]


@pytest.mark.parametrize("gain_kwarg", ["is_gain_amp", "is_gain_pha"])
def test_flag_outlier_cal_zeros(gain_array, gain_kwarg):
    """Zeros are flagged, and the caller's array left alone."""
    cal_array = gain_array.copy()
    cal_array[2, 0, 4, 1] = 0.0
    exp_array = cal_array.copy()

    new_flags = gi_utils._flag_outlier_cal(cal_array, 0.5, **{gain_kwarg: True})

    assert new_flags[2, 0, 4, 1]
    # The zero is swapped for a NaN internally, which must not reach the caller.
    np.testing.assert_array_equal(exp_array, cal_array)


@pytest.mark.parametrize("gain_kwarg", ["is_gain_amp", "is_gain_pha"])
def test_flag_outlier_cal_errs(delay_array, gain_kwarg):
    with pytest.raises(ValueError, match="cal_array must be complex"):
        gi_utils._flag_outlier_cal(delay_array, 0.5, **{gain_kwarg: True})


def test_build_interp_flags_clean(delay_array, blank_flags, old_times, new_times):
    new_flags = gi_utils.build_interp_flags(
        delay_array, blank_flags, old_times, new_times
    )

    assert new_flags.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert not new_flags.any()


def test_build_interp_flags_adjacent(delay_array, old_times, new_times):
    """flag_adjacent widens the flags around a flagged sample."""
    flags = make_flags(partial=[[0, 0, 0]])

    with_adjacent = gi_utils.build_interp_flags(
        delay_array, flags, old_times, new_times, flag_adjacent=True
    )
    without = gi_utils.build_interp_flags(
        delay_array, flags, old_times, new_times, flag_adjacent=False
    )

    assert with_adjacent[0, 0, :, 0].sum() > without[0, 0, :, 0].sum()
    assert not with_adjacent[1, 1, :, 1].any()


def test_build_interp_flags_single_time(delay_array):
    flags = np.zeros((NANTS, NFREQS, 1, NJONES), dtype=bool)
    flags[0, 0, 0, 0] = True

    new_flags = gi_utils.build_interp_flags(
        delay_array[:, :, :1, :], flags, np.array([0.5]), np.linspace(0, 1, NNEW)
    )

    assert new_flags.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert new_flags[0, 0, :, 0].all()
    assert not new_flags[1, 1, :, 1].any()


def test_build_interp_flags_no_times(delay_array):
    flags = np.zeros((NANTS, NFREQS, 0, NJONES), dtype=bool)

    new_flags = gi_utils.build_interp_flags(
        delay_array[:, :, :0, :], flags, np.array([]), np.linspace(0, 1, NNEW)
    )

    assert new_flags.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert new_flags.all()


def test_build_interp_flags_unsorted(delay_array, blank_flags, old_times, new_times):
    """Time-reversed input gets sorted internally, and agrees."""
    sort_idx = np.argsort(old_times)[::-1]

    exp_flags = gi_utils.build_interp_flags(
        delay_array, blank_flags, old_times, new_times
    )
    new_flags = gi_utils.build_interp_flags(
        delay_array[:, :, sort_idx, :],
        blank_flags[:, :, sort_idx, :],
        old_times[sort_idx],
        new_times,
    )

    np.testing.assert_array_equal(exp_flags, new_flags)


def test_build_interp_flags_tol_lim(delay_array, blank_flags, old_times, new_times):
    cal_array = delay_array.copy()
    cal_array[0, 0, 6:, 0] += 50.0

    new_flags = gi_utils.build_interp_flags(
        cal_array, blank_flags, old_times, new_times, tol_lim=1.0
    )

    assert new_flags[0, 0, :, 0].any()
    assert not new_flags[1, 1, :, 1].any()


@pytest.mark.parametrize("is_gain_amp,is_gain_pha", [[True, False], [False, True]])
def test_build_interp_flags_gain(
    gain_array, blank_flags, old_times, new_times, is_gain_amp, is_gain_pha
):
    """Test the amplitude and phase forms of the tol_lim check."""
    cal_array = gain_array.copy()
    cal_array[0, 0, 6:, 0] *= 5.0 * np.exp(1j * 2.0)

    new_flags = gi_utils.build_interp_flags(
        cal_array,
        blank_flags,
        old_times,
        new_times,
        tol_lim=0.5,
        is_gain_amp=is_gain_amp,
        is_gain_pha=is_gain_pha,
    )

    assert new_flags[0, 0, :, 0].any()
    assert not new_flags[1, 1, :, 1].any()


def test_build_interp_flags_max_time_delta(
    delay_array, blank_flags, old_times, new_times
):
    loose = gi_utils.build_interp_flags(
        delay_array, blank_flags, old_times, new_times, max_time_delta=1.0
    )
    tight = gi_utils.build_interp_flags(
        delay_array, blank_flags, old_times, new_times, max_time_delta=1e-6
    )

    assert not loose.any()
    assert tight.all()


@pytest.mark.parametrize("allow_extrapolation", [True, False])
def test_build_interp_flags_extrapolation(
    delay_array, blank_flags, old_times, allow_extrapolation
):
    """A new time landing on an old one is not extrapolated."""
    probe_times = np.array([old_times[0] - 0.5, old_times[0], old_times[-1], 1.5])

    new_flags = gi_utils.build_interp_flags(
        delay_array,
        blank_flags,
        old_times,
        probe_times,
        allow_extrapolation=allow_extrapolation,
    )

    exp_flags = [not allow_extrapolation, False, False, not allow_extrapolation]
    np.testing.assert_array_equal(new_flags[0, 0, :, 0], exp_flags)


@pytest.mark.parametrize("gain_kwarg", ["is_gain_amp", "is_gain_pha"])
def test_build_interp_flags_errs(
    delay_array, blank_flags, old_times, new_times, gain_kwarg
):
    with pytest.raises(ValueError, match="cal_array must be complex"):
        gi_utils.build_interp_flags(
            delay_array, blank_flags, old_times, new_times, **{gain_kwarg: True}
        )


@pytest.mark.parametrize(
    "interp_func", [gi_utils._interp_linear_cal, gi_utils._interp_nearest_cal]
)
def test_interp_simple_roundtrip(interp_func, delay_array, blank_flags, old_times):
    new_cal = interp_func(old_times, delay_array, blank_flags, old_times)

    np.testing.assert_allclose(new_cal, delay_array)


@pytest.mark.parametrize(
    "interp_func", [gi_utils._interp_linear_cal, gi_utils._interp_nearest_cal]
)
@pytest.mark.parametrize("ignore_flags", [True, False])
def test_interp_simple_flags(
    interp_func, delay_array, old_times, new_times, ignore_flags
):
    """Flagged samples are dropped unless ignore_flags is set."""
    flags = make_flags(partial=[[0, 0, 0]], dead=[[1, 1, 1]])

    new_cal = interp_func(old_times, delay_array, flags, new_times, ignore_flags)

    assert new_cal.shape == (NANTS, NFREQS, NNEW, NJONES)
    if ignore_flags:
        assert np.isfinite(new_cal).all()
    else:
        # Nothing left to work from on the fully flagged solution.
        assert np.isnan(new_cal[1, 1, :, 1]).all()
        assert np.isfinite(new_cal[2, 0, :, 0]).all()


def test_interp_linear_cal(delay_array, blank_flags, old_times, new_times):
    """Test the linear interpolation against np.interp."""
    new_cal = gi_utils._interp_linear_cal(
        old_times, delay_array, blank_flags, new_times
    )

    exp_cal = np.interp(new_times, old_times, delay_array[0, 0, :, 0])
    np.testing.assert_allclose(new_cal[0, 0, :, 0], exp_cal)


def test_interp_nearest_cal(delay_array, blank_flags, old_times):
    """The nearest sample is picked, nudging just off each old time."""
    probe_times = old_times[:-1] + (0.01 * np.diff(old_times))

    new_cal = gi_utils._interp_nearest_cal(
        old_times, delay_array, blank_flags, probe_times
    )

    np.testing.assert_allclose(new_cal[0, 0, :, 0], delay_array[0, 0, :-1, 0])


def test_pchip_slopes_two_points():
    slopes = gi_utils._pchip_slopes(np.array([2.0]), np.array([3.0]))

    np.testing.assert_allclose(slopes, [3.0, 3.0])


@pytest.mark.parametrize(
    "delta,exp_slope",
    [
        [[1.0, -1.0], 0.0],  # secants change sign, so the knot is an extremum
        [[0.0, 1.0], 0.0],  # a zero secant marks an extremum as well
        [[1.0, 1.0], 1.0],  # matched secants give the harmonic mean back
    ],
)
def test_pchip_slopes_extrema(delta, exp_slope):
    """Test how an interior knot responds to the adjacent secants."""
    slopes = gi_utils._pchip_slopes(np.ones(2), np.array(delta))

    assert slopes[1] == exp_slope


def test_pchip_slopes_signed_zeros():
    """Verify that -0.0 and 0.0 secants both count as flat.

    The sign product is used rather than a direct comparison precisely because
    np.sign(-0.0) == np.sign(0.0), which would otherwise divide by zero here.
    """
    slopes = gi_utils._pchip_slopes(np.ones(3), np.array([-0.0, 0.0, -0.0]))

    assert np.isfinite(slopes).all()


def test_pchip_slopes_endpoint_limit():
    """An overshooting endpoint estimate is capped at three secants."""
    # The raw estimate here is 6.5 against a 3.0 cap.
    delta = np.array([1.0, -10.0])

    slopes = gi_utils._pchip_slopes(np.ones(2), delta)

    assert slopes[0] == 3 * delta[0]


@pytest.mark.parametrize("style", ["noise", "increasing", "flat", "plateau"])
def test_interp_pchip(style):
    """Test the cubic kernel against scipy, over a range of data shapes."""
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 10, NTIMES))
    if style == "noise":
        y = rng.normal(size=NTIMES)
    elif style == "increasing":
        y = np.sort(rng.uniform(size=NTIMES))
    elif style == "flat":
        y = np.zeros(NTIMES)
    else:
        y = np.repeat(rng.normal(size=(NTIMES + 1) // 2), 2)[:NTIMES]

    # Knots, interior points, and a point off each end.
    xq = np.concatenate([x, x[:-1] + (np.diff(x) / 3), [x[0] - 1.0, x[-1] + 1.0]])

    np.testing.assert_allclose(
        gi_utils._interp_pchip(x, y, xq),
        PchipInterpolator(x, y, extrapolate=True)(xq),
        atol=1e-10,
    )


def test_interp_pchip_4d(gain_array, old_times, new_times):
    """The batched cubic agrees with going solution by solution."""
    cal_array = np.abs(gain_array)

    new_cal = gi_utils._interp_pchip_4d(old_times, cal_array, new_times)

    for ant in range(NANTS):
        for freq in range(NFREQS):
            for jones in range(NJONES):
                np.testing.assert_allclose(
                    new_cal[ant, freq, :, jones],
                    gi_utils._interp_pchip(
                        old_times,
                        np.ascontiguousarray(cal_array[ant, freq, :, jones]),
                        new_times,
                    ),
                )


def test_interp_cubic_cal_roundtrip(delay_array, blank_flags, old_times):
    new_cal = gi_utils._interp_cubic_cal(old_times, delay_array, blank_flags, old_times)

    np.testing.assert_allclose(new_cal, delay_array)


def test_interp_cubic_cal_short_track(delay_array, old_times):
    n_keep = gi_utils._MIN_CUBIC_SAMPLES - 1
    flags = np.zeros((NANTS, NFREQS, n_keep, NJONES), dtype=bool)

    new_cal = gi_utils._interp_cubic_cal(
        old_times[:n_keep],
        delay_array[:, :, :n_keep, :],
        flags,
        np.linspace(0, 1, NNEW),
    )

    assert np.isnan(new_cal).all()


def test_interp_cubic_cal_flags(delay_array, old_times, new_times):
    """Partly flagged solutions get refit, dead ones get blanked."""
    flags = make_flags(partial=[[0, 0, 0]], dead=[[1, 1, 1]])

    new_cal = gi_utils._interp_cubic_cal(old_times, delay_array, flags, new_times)

    assert np.isnan(new_cal[1, 1, :, 1]).all()
    assert np.isfinite(new_cal[0, 0, :, 0]).all()
    assert np.isfinite(new_cal[2, 0, :, 0]).all()


def test_interp_cubic_cal_no_clean_solns(delay_array, old_times, new_times):
    """With no clean solution, the batched pass is skipped entirely."""
    flags = np.zeros((NANTS, NFREQS, NTIMES, NJONES), dtype=bool)
    flags[:, :, 0, :] = True

    new_cal = gi_utils._interp_cubic_cal(old_times, delay_array, flags, new_times)

    assert np.isfinite(new_cal).all()


def test_interp_cubic_cal_starved(delay_array, old_times, new_times):
    flags = np.zeros((NANTS, NFREQS, NTIMES, NJONES), dtype=bool)
    flags[0, 0, gi_utils._MIN_CUBIC_SAMPLES - 1 :, 0] = True

    new_cal = gi_utils._interp_cubic_cal(old_times, delay_array, flags, new_times)

    assert np.isnan(new_cal[0, 0, :, 0]).all()
    assert np.isfinite(new_cal[1, 1, :, 1]).all()


def test_prep_matrix_poly_lsqfit_cal(old_times, new_times):
    """The fit matrix holds Chebyshev polynomials of the first kind."""
    order = 5
    xval_matrix, eval_matrix = gi_utils._prep_matrix_poly_lsqfit_cal(
        old_times.reshape(1, -1), new_times.reshape(1, -1), np.array([order])
    )

    lo_val, hi_val = old_times.min(), old_times.max()
    old_norm = (2 * (old_times - lo_val) / (hi_val - lo_val)) - 1
    exp_matrix = np.column_stack(
        [np.polynomial.chebyshev.Chebyshev.basis(k)(old_norm) for k in range(order + 1)]
    )

    np.testing.assert_allclose(xval_matrix, exp_matrix, atol=1e-12)
    assert eval_matrix.shape == (NNEW, order + 1)


def test_prep_matrix_poly_lsqfit_cal_zero_order(old_times, new_times):
    xval_matrix, _ = gi_utils._prep_matrix_poly_lsqfit_cal(
        np.vstack([old_times, np.ones(NTIMES)]),
        np.vstack([new_times, np.ones(NNEW)]),
        np.array([2, 0]),
    )

    # Just the constant plus the two time terms.
    assert xval_matrix.shape == (NTIMES, 3)


def test_interp_lsqfit_cal_partial_flags(delay_array, old_times):
    """Test the masked branch, where the fit matrix is cut down."""
    flags = make_flags(partial=[[0, 0, 0]])
    xval_matrix = np.column_stack([np.ones(NTIMES), old_times, old_times**2])
    new_norm = np.linspace(0, 1, NNEW)
    eval_matrix = np.column_stack([np.ones(NNEW), new_norm, new_norm**2])

    new_cal = gi_utils._interp_lsqfit_cal(delay_array, flags, xval_matrix, eval_matrix)

    assert np.isfinite(new_cal).all()


def test_interp_lsqfit_cal_singular(delay_array, blank_flags, old_times):
    xval_matrix = np.ones((NTIMES, 3))
    xval_matrix[:, 1] = old_times
    # Duplicate a column, so that the normal-equation matrix is singular.
    xval_matrix[:, 2] = xval_matrix[:, 1]

    new_cal = gi_utils._interp_lsqfit_cal(
        delay_array, blank_flags, xval_matrix, np.ones((NNEW, 3))
    )

    assert np.isnan(new_cal).all()


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_interp_poly_cal(old_times, new_times, order):
    """A known polynomial is recovered, flags and all."""
    lo_val, hi_val = old_times.min(), old_times.max()
    old_norm = (2 * (old_times - lo_val) / (hi_val - lo_val)) - 1
    new_norm = (2 * (new_times - lo_val) / (hi_val - lo_val)) - 1

    rng = np.random.default_rng(7)
    coeffs = rng.normal(size=order + 1)
    old_vals = sum(coeffs[p] * old_norm**p for p in range(order + 1))
    exp_vals = sum(coeffs[p] * new_norm**p for p in range(order + 1))

    cal_array = np.broadcast_to(
        old_vals[None, None, :, None], (NANTS, NFREQS, NTIMES, NJONES)
    ).copy()

    new_cal = gi_utils._interp_poly_cal(
        cal_array,
        make_flags(partial=[[0, 0, 0]]),
        old_times.reshape(1, -1),
        new_times.reshape(1, -1),
        np.array([order]),
    )

    np.testing.assert_allclose(new_cal[0, 0, :, 0], exp_vals, atol=1e-10)
    np.testing.assert_allclose(new_cal[1, 1, :, 1], exp_vals, atol=1e-10)


def test_interp_poly_cal_multivar(delay_array, blank_flags, old_times, new_times):
    new_cal = gi_utils._interp_poly_cal(
        delay_array,
        blank_flags,
        np.vstack([old_times, np.sin(3 * old_times)]),
        np.vstack([new_times, np.sin(3 * new_times)]),
        np.array([2, 1]),
    )

    assert new_cal.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert np.isfinite(new_cal).all()


def test_interp_poly_cal_starved(delay_array, old_times, new_times):
    flags = np.zeros((NANTS, NFREQS, NTIMES, NJONES), dtype=bool)
    flags[0, 0, 2:, 0] = True

    new_cal = gi_utils._interp_poly_cal(
        delay_array,
        flags,
        old_times.reshape(1, -1),
        new_times.reshape(1, -1),
        np.array([3]),
    )

    assert np.isnan(new_cal[0, 0, :, 0]).all()
    assert np.isfinite(new_cal[1, 1, :, 1]).all()


def test_interp_poly_cal_errs(delay_array, blank_flags, old_times, new_times):
    with pytest.raises(ValueError, match="order cannot be negative."):
        gi_utils._interp_poly_cal(
            delay_array,
            blank_flags,
            old_times.reshape(1, -1),
            new_times.reshape(1, -1),
            np.array([-1]),
        )

    with pytest.raises(
        ValueError, match="Length of order must match the first dimension"
    ):
        gi_utils._interp_poly_cal(
            delay_array,
            blank_flags,
            old_times.reshape(1, -1),
            new_times.reshape(1, -1),
            np.array([2, 2]),
        )

    with pytest.raises(
        ValueError, match="old_var and new_var must contain the same number"
    ):
        gi_utils._interp_poly_cal(
            delay_array,
            blank_flags,
            np.vstack([old_times, old_times]),
            new_times.reshape(1, -1),
            np.array([2, 2]),
        )

    with pytest.raises(ValueError, match="Variable 1 is constant"):
        gi_utils._interp_poly_cal(
            delay_array,
            blank_flags,
            np.vstack([old_times, np.ones(NTIMES)]),
            np.vstack([new_times, np.ones(NNEW)]),
            np.array([2, 1]),
        )


@pytest.mark.parametrize("mode", ["real", "amp"])
def test_interp_dispatcher_update(gain_array, blank_flags, old_times, new_times, mode):
    """Verify the second-pass branches, which fold into an existing array.

    time_interp_cal always runs real before imag and amp before phase, so these two
    branches are only reachable by driving the dispatcher directly.
    """
    new_cal = np.ones((NANTS, NFREQS, NNEW, NJONES), dtype=complex)

    out_cal, out_flags = gi_utils._interp_dispatcher(
        mode=mode,
        old_times=old_times,
        old_cal=gain_array,
        old_flags=blank_flags,
        new_times=new_times,
        new_cal=new_cal,
        new_flags=np.zeros((NANTS, NFREQS, NNEW, NJONES), dtype=bool),
        kind="linear",
    )

    assert out_flags.shape == (NANTS, NFREQS, NNEW, NJONES)
    # The array handed in is updated in place rather than replaced.
    assert out_cal is new_cal


def test_interp_dispatcher_errs(gain_array, blank_flags, old_times, new_times):
    disp_kwargs = {
        "old_times": old_times,
        "old_cal": gain_array,
        "old_flags": blank_flags,
        "new_times": new_times,
    }

    with pytest.raises(ValueError, match="Unrecognised mode 'foo'"):
        gi_utils._interp_dispatcher(mode="foo", kind="linear", **disp_kwargs)

    with pytest.raises(ValueError, match="Unrecognised interpolation kind 'foo'"):
        gi_utils._interp_dispatcher(mode="amp", kind="foo", **disp_kwargs)


@pytest.mark.parametrize("kind", ["nearest", "linear", "cubic", "poly"])
@pytest.mark.parametrize("interp_mode", ["ampphase", "complex", "amp", "phase"])
def test_time_interp_cal(
    gain_array, blank_flags, old_times, new_times, kind, interp_mode
):
    """Test every kind and mode combination for the expected arrays."""
    new_cal, new_flags = gi_utils.time_interp_cal(
        old_times,
        gain_array,
        blank_flags,
        new_times,
        kind=kind,
        interp_mode=interp_mode,
    )

    assert new_cal.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert new_flags.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert new_flags.dtype == np.bool_
    assert np.iscomplexobj(new_cal)


@pytest.mark.parametrize("kind", ["nearest", "linear", "cubic"])
def test_time_interp_cal_roundtrip(gain_array, blank_flags, old_times, kind):
    new_cal, new_flags = gi_utils.time_interp_cal(
        old_times, gain_array, blank_flags, old_times, kind=kind
    )

    np.testing.assert_allclose(new_cal, gain_array, atol=1e-12)
    assert not new_flags.any()


def test_time_interp_cal_real(delay_array, blank_flags, old_times, new_times):
    new_cal, _ = gi_utils.time_interp_cal(
        old_times, delay_array, blank_flags, new_times, interp_mode="real"
    )

    assert new_cal.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert np.isfinite(new_cal).all()


def test_time_interp_cal_imag(gain_array, blank_flags, old_times, new_times):
    """The imaginary-only mode scales the result by 1j."""
    new_cal, _ = gi_utils.time_interp_cal(
        old_times, gain_array, blank_flags, new_times, interp_mode="imag"
    )

    assert np.iscomplexobj(new_cal)
    np.testing.assert_allclose(new_cal.real, 0.0, atol=1e-12)


@pytest.mark.parametrize("interp_mode", ["complex", "ampphase"])
def test_time_interp_cal_two_pass(gain_array, blank_flags, old_times, interp_mode):
    """Both two-pass modes fill the array in on the second call."""
    new_cal, _ = gi_utils.time_interp_cal(
        old_times, gain_array, blank_flags, old_times, interp_mode=interp_mode
    )

    np.testing.assert_allclose(new_cal, gain_array, atol=1e-12)


@pytest.mark.parametrize(
    "amp_kind,pha_kind", [["cubic", "linear"], ["poly", "nearest"], [None, "poly"]]
)
def test_time_interp_cal_split_kind(
    gain_array, blank_flags, old_times, new_times, amp_kind, pha_kind
):
    """Amplitude and phase can be interpolated in different ways."""
    new_cal, _ = gi_utils.time_interp_cal(
        old_times,
        gain_array,
        blank_flags,
        new_times,
        amp_kind=amp_kind,
        pha_kind=pha_kind,
    )

    assert np.isfinite(new_cal).all()


@pytest.mark.parametrize(
    "split_kwargs",
    [
        {"amp_poly_order": 2, "pha_poly_order": 4},
        {"pha_flag_delta": 0.25, "pha_max_time_delta": 0.5},
        {"amp_flag_delta": 0.25, "amp_max_time_delta": 0.5},
    ],
)
def test_time_interp_cal_split_kwargs(
    gain_array, blank_flags, old_times, new_times, split_kwargs
):
    """Test the amp and phase overrides of the shared keywords."""
    new_cal, new_flags = gi_utils.time_interp_cal(
        old_times, gain_array, blank_flags, new_times, kind="poly", **split_kwargs
    )

    assert new_cal.shape == (NANTS, NFREQS, NNEW, NJONES)
    assert new_flags.shape == (NANTS, NFREQS, NNEW, NJONES)


def test_time_interp_cal_var(gain_array, blank_flags, old_times, new_times):
    """A polynomial fit can take extra variables alongside time."""
    new_cal, _ = gi_utils.time_interp_cal(
        old_times,
        gain_array,
        blank_flags,
        new_times,
        kind="poly",
        old_var=np.vstack([np.sin(3 * old_times)]),
        new_var=np.vstack([np.sin(3 * new_times)]),
        poly_order=[3, 1],
    )

    assert np.isfinite(new_cal).all()


def test_time_interp_cal_fully_flagged(gain_array, old_times, new_times):
    flags = np.ones((NANTS, NFREQS, NTIMES, NJONES), dtype=bool)

    new_cal, new_flags = gi_utils.time_interp_cal(
        old_times, gain_array, flags, new_times, kind="cubic"
    )

    assert new_flags.all()
    assert np.isnan(new_cal).all()


@pytest.mark.parametrize("allow_extrapolation", [True, False])
def test_time_interp_cal_extrapolation(
    gain_array, blank_flags, old_times, allow_extrapolation
):
    """Only genuinely out-of-range times count as extrapolated."""
    probe_times = np.array([old_times[0] - 0.5, old_times[0], 0.5, old_times[-1], 1.5])

    _, new_flags = gi_utils.time_interp_cal(
        old_times,
        gain_array,
        blank_flags,
        probe_times,
        allow_extrapolation=allow_extrapolation,
    )

    exp_flags = [not allow_extrapolation, False, False, False, not allow_extrapolation]
    np.testing.assert_array_equal(new_flags[0, 0, :, 0], exp_flags)


def test_time_interp_cal_flag_isolation(gain_array, old_times):
    """Flagging one solution leaves the others alone."""
    flags = make_flags(partial=[[0, 0, 0]])

    _, new_flags = gi_utils.time_interp_cal(
        old_times, gain_array, flags, old_times, kind="linear"
    )

    assert new_flags[0, 0, :, 0].any()
    assert not new_flags[1, 1, :, 1].any()


@pytest.mark.parametrize(
    "err_kwargs,err_msg",
    [
        [{"interp_mode": "foo"}, "Unrecognised interp_mode 'foo'"],
        [{"kind": "foo"}, "Unrecognised interpolation kind 'foo'"],
        [{"amp_kind": "foo"}, "Unrecognised amplitude interpolation kind 'foo'"],
        [{"pha_kind": "foo"}, "Unrecognised phase interpolation kind 'foo'"],
    ],
)
def test_time_interp_cal_errs(
    gain_array, blank_flags, old_times, new_times, err_kwargs, err_msg
):
    with pytest.raises(ValueError, match=err_msg):
        gi_utils.time_interp_cal(
            old_times, gain_array, blank_flags, new_times, **err_kwargs
        )


def test_time_interp_cal_real_input_errs(
    delay_array, blank_flags, old_times, new_times
):
    with pytest.raises(ValueError, match="is not compatible with real-valued old_cal"):
        gi_utils.time_interp_cal(
            old_times, delay_array, blank_flags, new_times, interp_mode="ampphase"
        )


@pytest.mark.parametrize("drop_var", ["old_var", "new_var"])
def test_time_interp_cal_lone_var_errs(
    gain_array, blank_flags, old_times, new_times, drop_var
):
    var_kwargs = {
        "old_var": np.vstack([np.sin(3 * old_times)]),
        "new_var": np.vstack([np.sin(3 * new_times)]),
    }
    var_kwargs[drop_var] = None

    with pytest.raises(ValueError, match="must either both be set or both be left"):
        gi_utils.time_interp_cal(
            old_times, gain_array, blank_flags, new_times, kind="poly", **var_kwargs
        )


@pytest.mark.parametrize(
    "var_kwargs,err_msg",
    [
        [
            {"old_var": "double_time", "new_var": "time"},
            "old_var and new_var must contain the same number",
        ],
        [
            {"old_var": "short_time", "new_var": "time"},
            "old_var must have the same number of entries as old_times",
        ],
        [
            {"old_var": "time", "new_var": "short_time"},
            "new_var must have the same number of entries as new_times",
        ],
        [
            {"old_var": "sine", "new_var": "sine", "poly_order": [1, 2, 3]},
            "poly_order must either have 1 value",
        ],
        [
            {"old_var": "sine", "new_var": "sine", "poly_order": "not-a-number"},
            "invalid literal|could not convert",
        ],
    ],
)
def test_time_interp_cal_var_errs(
    gain_array, blank_flags, old_times, new_times, var_kwargs, err_msg
):
    var_lookup = {
        "time": [np.vstack([old_times]), np.vstack([new_times])],
        "double_time": [np.vstack([old_times] * 2), np.vstack([new_times] * 2)],
        "short_time": [np.vstack([old_times[:-1]]), np.vstack([new_times[:-1]])],
        "sine": [
            np.vstack([np.sin(3 * old_times)]),
            np.vstack([np.sin(3 * new_times)]),
        ],
    }
    var_kwargs = dict(var_kwargs)
    var_kwargs["old_var"] = var_lookup[var_kwargs["old_var"]][0]
    var_kwargs["new_var"] = var_lookup[var_kwargs["new_var"]][1]

    with pytest.raises(ValueError, match=err_msg):
        gi_utils.time_interp_cal(
            old_times, gain_array, blank_flags, new_times, kind="poly", **var_kwargs
        )


@pytest.mark.parametrize("kind,kwargs", [["cubic", {}], ["poly", {"poly_order": 3}]])
def test_time_interp_cal_undersupplied_flags_nans(
    gain_array, blank_flags, old_times, new_times, kind, kwargs
):
    # Verify that cubin/poly interpolation with too few solns get flagged
    few = slice(0, 3)
    new_cal, new_flags = gi_utils.time_interp_cal(
        old_times[few],
        gain_array[:, :, few],
        blank_flags[:, :, few],
        new_times,
        kind=kind,
        **kwargs,
    )

    assert not np.all(np.isfinite(new_cal)), "expected this case to produce NaN"
    assert np.all(new_flags[~np.isfinite(new_cal)])
