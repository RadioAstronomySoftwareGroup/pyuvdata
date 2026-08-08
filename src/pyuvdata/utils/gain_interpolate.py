# Copyright (c) 2026 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Gain interpolation related utilities."""

from typing import Literal

import numba as nb
import numpy as np

from .types import BoolArray, ComplexArray, FloatArray, IntArray

_MIN_CUBIC_SAMPLES = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flag_outlier_cal(cal_array, tol_lim, is_gain_amp=False, is_gain_pha=False):
    """
    Flag individual outliers in a set of calibration solutions.

    Marks a sample as bad where the deltas to both of its neighbors exceed the
    tolerance but the delta that skips it does not, which is the signature of an
    isolated outlier rather than a genuine excursion.

    Parameters
    ----------
    cal_array : ndarray of float or complexfloat
        Calibration solutions to check, shape (Nants, Nfreqs, Ntimes, Njones), dtype
        is either float or complex (assumed complex if `is_gain_phase` or `is_gain_amp`
        is set to True).
    tol_lim : float
        Largest delta between samples that is considered acceptable. Units match
        those of the deltas being tested, which depend on `is_gain_amp` and
        `is_gain_pha`.
    is_gain_amp : bool
        If True, treat `cal_array` as complex gains and test fractional changes in
        amplitude. Default is False.
    is_gain_pha : bool
        If True, treat `cal_array` as complex gains and test changes in phase, in
        radians. Default is False.

    Returns
    -------
    new_flags : ndarray of bool
        Flags for `cal_array`, same shape as `cal_array`. True where the sample is
        NaN (or zero and `is_gain_amp` or `is_gain_pha` is set), or looks like an
        isolated outlier.
    """
    # Set the initial flags based on nans (and zeros if the solns are complex).
    new_flags = np.isnan(cal_array)
    if is_gain_amp or is_gain_pha:
        if not np.issubdtype(cal_array.dtype, np.complexfloating):
            raise ValueError(
                "cal_array must be complex if is_gain_amp or is_gain_pha is True."
            )
        mask = cal_array == 0.0
        if np.any(mask):
            cal_array = np.where(mask, np.nan, cal_array)
            new_flags |= mask

    # Dividing through the NaNs when flagged is deliberate, so suppress the warnings.
    with np.errstate(invalid="ignore"):
        if is_gain_amp:
            delta_12 = np.abs(cal_array[:, :, 1:, :] / cal_array[:, :, :-1, :]) - 1
            delta_13 = np.abs(cal_array[:, :, 2:, :] / cal_array[:, :, :-2, :]) - 1
        elif is_gain_pha:
            delta_12 = np.angle(cal_array[:, :, 1:, :] / cal_array[:, :, :-1, :])
            delta_13 = np.angle(cal_array[:, :, 2:, :] / cal_array[:, :, :-2, :])
        else:
            delta_12 = np.real(cal_array[:, :, 1:, :] - cal_array[:, :, :-1, :])
            delta_13 = np.real(cal_array[:, :, 2:, :] - cal_array[:, :, :-2, :])

    # Where a delta between adjacent points exceeds the threshold, mark it as "bad"
    bad_12 = (delta_12 > tol_lim) | (delta_12 < -tol_lim)
    # Where a delta that skips a point is within the threshold, mark it as "good"
    good_13 = (delta_13 <= tol_lim) & (delta_13 >= -tol_lim)
    # Separately capture where the delta is NaN
    nan_12 = np.isnan(delta_12)

    # Where two bad deltas are adjacent, and the larger delta that skips the midpoint
    # is good, assume the midpoint is bad and flag it.
    new_flags[:, :, 1:-1, :] |= (bad_12[:, :, 1:, :] & bad_12[:, :, :-1, :]) & good_13

    # If an outlier delta is next to a NaN, assume its bad and flag it too.
    new_flags[:, :, 1:-1, :] |= bad_12[:, :, 1:, :] & nan_12[:, :, :-1, :]
    new_flags[:, :, 1:-1, :] |= nan_12[:, :, 1:, :] & bad_12[:, :, :-1, :]

    # Use the "goodness" of the second and second-to-last points to determine the
    # flags for the first and last points (if the deltas exceed the threshold).
    new_flags[:, :, 0, :] |= bad_12[:, :, 0, :] & (
        np.abs(delta_12[:, :, 1, :]) <= tol_lim
    )
    new_flags[:, :, -1, :] |= bad_12[:, :, -1, :] & (
        np.abs(delta_12[:, :, -2, :]) <= tol_lim
    )

    return new_flags


def build_interp_flags(
    cal_array,
    flag_array,
    old_times,
    new_times,
    tol_lim=0.0,
    is_gain_amp=False,
    is_gain_pha=False,
    flag_adjacent=True,
    max_time_delta=0.0,
    allow_extrapolation=True,
):
    """
    Build the flags that apply to a set of interpolated calibration solutions.

    Parameters
    ----------
    cal_array : ndarray
        Calibration solutions being interpolated, shape (Nants, Nfreqs, Ntimes, Njones),
        dtype is either float or complex (assumed complex if `is_gain_phase` or
        `is_gain_amp` is set to True).
    flag_array : ndarray of bool
        Flags for `cal_array`, shape (Nants, Nfreqs, Ntimes, Njones), dtype bool.
    old_times : ndarray of float
        Times corresponding to the time-axis of `cal_array`, shape (Ntimes,), dtype is
        float, units are Julian days.
    new_times : ndarray of float
        Times at which to evaluate the interpolation, shape (Nnew,), dtype is float,
        units are Julian days.
    tol_lim : float
        Largest delta between adjacent samples that is considered acceptable. If
        greater than zero, intervals whose delta exceeds this are flagged. Default
        is 0.0, which disables the check.
    is_gain_amp : bool
        If True, treat `cal_array` as complex gains and test fractional changes in
        amplitude. Default is False.
    is_gain_pha : bool
        If True, treat `cal_array` as complex gains and test changes in phase, in
        radians. Default is False.
    flag_adjacent : bool
        If True, propagate flags from `flag_array` to the adjacent intervals in
        `cal_array`. Default is True.
    max_time_delta : float
        Largest gap allowed between a new time and each of the old times that
        bracket it, in units of Julian days. A new time is flagged unless it lies
        within this distance of both of its neighbors, so that a solution sitting
        on top of one old time is still flagged if the other side of the interval
        is too far away. New times beyond the range of `old_times` are instead
        tested against the single nearest old time. Default is 0.0, which disables
        the check.
    allow_extrapolation : bool
        If True, allow new times that fall outside the range of `old_times`. If
        False, those times are flagged. Default is True.

    Returns
    -------
    new_flags : ndarray of bool
        Flags for the interpolated solutions, shape (Nants, Nfreqs, Nnew, Njones), dtype
        bool.
    """
    # Capture any NaN values in cal_array and flag them.
    flag_array = flag_array | np.isnan(cal_array)

    # If this is a gain amplitude array, also flag any zero values (which are invalid).
    if is_gain_amp or is_gain_pha:
        if not np.issubdtype(cal_array.dtype, np.complexfloating):
            raise ValueError(
                "cal_array must be complex if is_gain_amp or is_gain_pha is True."
            )
        flag_array |= cal_array == 0.0

    if len(old_times) < 2:
        # Cannot compute a delta with fewer than 2 points, so mark everything as
        # flagged and then, if there is a single time to work from, let its flags
        # stand in for every new time.
        new_flags = np.ones(
            (
                flag_array.shape[0],
                flag_array.shape[1],
                len(new_times),
                flag_array.shape[3],
            ),
            dtype=np.bool_,
        )
        if len(old_times) == 1:
            new_flags[:, :, :, :] = flag_array[:, :, 0:1, :]

        return new_flags

    if not np.all(np.diff(old_times) >= 0):
        sort_idx = np.argsort(old_times)
        old_times = old_times[sort_idx]
        cal_array = cal_array[:, :, sort_idx, :]
        flag_array = flag_array[:, :, sort_idx, :]

    # Create a blank set of flags for each time range
    window_flags = np.zeros(
        (
            flag_array.shape[0],
            flag_array.shape[1],
            flag_array.shape[2] + 1,
            flag_array.shape[3],
        ),
        dtype=np.bool_,
    )

    if tol_lim > 0.0:
        # Dividing through the NaNs when flagged is deliberate, so suppress warnings.
        with np.errstate(invalid="ignore"):
            if is_gain_amp:
                delta_12 = np.abs(cal_array[:, :, 1:, :] / cal_array[:, :, :-1, :]) - 1
            elif is_gain_pha:
                delta_12 = np.angle(cal_array[:, :, 1:, :] / cal_array[:, :, :-1, :])
            else:
                delta_12 = np.real(cal_array[:, :, 1:, :] - cal_array[:, :, :-1, :])

        window_flags[:, :, 1:-1, :] = (delta_12 > tol_lim) | (delta_12 < -tol_lim)
        window_flags[:, :, 0, :] = window_flags[:, :, 1, :]
        window_flags[:, :, -1, :] = window_flags[:, :, -2, :]

    if flag_adjacent:
        window_flags[:, :, 1:, :] |= flag_array
        window_flags[:, :, :-1, :] |= flag_array

    insert_idx = np.searchsorted(old_times, new_times, side="left")
    new_flags = window_flags[:, :, insert_idx, :]

    if max_time_delta > 0.0:
        time_deltas = np.maximum(
            new_times - old_times[np.clip(insert_idx - 1, 0, len(old_times) - 1)],
            old_times[np.clip(insert_idx, 0, len(old_times) - 1)] - new_times,
        )
        new_flags[:, :, (time_deltas > max_time_delta), :] = True

    if not allow_extrapolation:
        # Test the times themselves rather than the insertion indices, since a new
        # time that lands exactly on the first old time gets an index of zero without
        # actually being extrapolated.
        new_flags[:, :, (new_times < old_times[0]), :] = True
        new_flags[:, :, (new_times > old_times[-1]), :] = True

    return new_flags


@nb.njit(cache=True)
def _interp_linear_cal(old_times, old_cal, old_flags, new_times, ignore_flags=False):
    """
    Interpolate calibration solutions in time using linear interpolation.

    Parameters
    ----------
    old_times : ndarray of float
        Times corresponding to the third axis of `old_cal`, shape (Ntimes,), dtype is
        float. Units must be the same as `new_times`. Must be in ascending order.
    old_cal : ndarray
        Calibration solutions to interpolate, shape (Nants, Nfreqs, Ntimes, Njones),
        dtype is float.
    old_flags : ndarray of bool
        Flags for `old_cal`, where flagged entries are excluded from the
        interpolation if `ignore_flags=False`. Shape is (Nants, Nfreqs, Ntimes, Njones),
        dtype is bool.
    new_times : ndarray of float
        Times at which to evaluate the interpolation, shape (Nnew,), dtype is float,
        Units must be the same as `old_times`.
    ignore_flags : bool
        If True, interpolate against every sample and disregard `old_flags`. Default
        is False.

    Returns
    -------
    new_cal : ndarray
        Interpolated solutions, shape (Nants, Nfreqs, Nnew, Njones), dtype is float.
        Entries where too few unflagged samples were available are set to NaN.
    """
    new_cal = np.full(
        (old_cal.shape[0], old_cal.shape[1], len(new_times), old_cal.shape[3]),
        np.nan,
        dtype=old_cal.dtype,
    )

    for ant_idx in range(old_cal.shape[0]):
        for freq_idx in range(old_cal.shape[1]):
            for jones_idx in range(old_cal.shape[3]):
                flag_slice = old_flags[ant_idx, freq_idx, :, jones_idx]
                if ignore_flags or not np.any(flag_slice):
                    time_slice = old_times
                    cal_slice = old_cal[ant_idx, freq_idx, :, jones_idx]
                else:
                    if np.all(flag_slice):
                        continue
                    good_mask = ~flag_slice
                    time_slice = old_times[good_mask]
                    cal_slice = old_cal[ant_idx, freq_idx, good_mask, jones_idx]

                new_cal[ant_idx, freq_idx, :, jones_idx] = np.interp(
                    new_times, time_slice, cal_slice
                )

    return new_cal


@nb.njit(cache=True)
def _interp_nearest_cal(old_times, old_cal, old_flags, new_times, ignore_flags=False):
    """
    Interpolate calibration solutions in time by taking the nearest sample.

    Parameters
    ----------
    old_times : ndarray of float
        Times corresponding to the third axis of `old_cal`, shape (Ntimes,), dtype is
        float. Units must be the same as `new_times`. Must be in ascending order.
    old_cal : ndarray
        Calibration solutions to interpolate, shape (Nants, Nfreqs, Ntimes, Njones),
        dtype is float.
    old_flags : ndarray of bool
        Flags for `old_cal`, where flagged entries are excluded from the
        interpolation if `ignore_flags=False`. Shape is (Nants, Nfreqs, Ntimes, Njones),
        dtype is bool.
    new_times : ndarray of float
        Times at which to evaluate the interpolation, shape (Nnew,), dtype is float.
        Units must be the same as `old_times`.
    ignore_flags : bool
        If True, interpolate against every sample and disregard `old_flags`. Default
        is False.

    Returns
    -------
    new_cal : ndarray
        Interpolated solutions, shape (Nants, Nfreqs, Nnew, Njones), dtype is float.
        Entries where too few unflagged samples were available are set to NaN.
    """
    mid_times = 0.5 * (old_times[1:] + old_times[:-1])
    new_cal = old_cal[:, :, np.searchsorted(mid_times, new_times), :]

    if ignore_flags or not np.any(old_flags):
        return new_cal

    for ant_idx in range(old_cal.shape[0]):
        for freq_idx in range(old_cal.shape[1]):
            for jones_idx in range(old_cal.shape[3]):
                flag_slice = old_flags[ant_idx, freq_idx, :, jones_idx]
                if not np.any(flag_slice):
                    continue

                good_mask = ~flag_slice
                if not np.any(good_mask):
                    # Nothing left to select from, so mark the whole slice as bad.
                    new_cal[ant_idx, freq_idx, :, jones_idx] = np.nan
                    continue

                time_slice = old_times[good_mask]
                cal_slice = old_cal[ant_idx, freq_idx, good_mask, jones_idx]
                mid_times_slice = 0.5 * (time_slice[1:] + time_slice[:-1])
                new_cal[ant_idx, freq_idx, :, jones_idx] = cal_slice[
                    np.searchsorted(mid_times_slice, new_times)
                ]

    return new_cal


@nb.njit(cache=True)
def _pchip_slopes(h, delta):
    """
    Calculate the knot derivatives for a shape-preserving cubic interpolation.

    Derivatives are chosen such that the interpolant is monotonic on any interval
    where the data are (Fritsch & Carlson 1980). See Moler, "Numerical Computing with
    MATLAB", Chap 3.6 (pchiptx.m) for further details.

    Parameters
    ----------
    h : ndarray of float
        Spacing between adjacent knots, shape (Nvals).
    delta : ndarray of float
        Secant slope across each interval, shape (Nvals,).

    Returns
    -------
    d : ndarray of float
        Derivative at each knot, shape (Nvals + 1,).
    """
    # Note that this code has been largely adapted from the MATLAB version of the PCHIP
    # algorithm. I (Karto) have kept the variable names mostly the same to allow for
    # easier point-to-point comparison against that code.
    n = len(h) + 1
    d = np.zeros(n, dtype=np.float64)

    if n == 2:
        # With only two points the interpolant is just a straight line.
        d[0] = delta[0]
        d[1] = delta[0]
        return d

    # Check if the derivative goes through a zero-crossing between or at adjacent knots.
    is_nonflat = (np.sign(delta[:-1]) * np.sign(delta[1:])) > 0

    # Use a weighted mean based on distance (in time) from each point.
    w1 = (2 * h[1:]) + h[:-1]
    w2 = h[1:] + (2 * h[:-1])
    whmean = (w1 / np.where(is_nonflat, delta[:-1], 1.0)) + (
        w2 / np.where(is_nonflat, delta[1:], 1.0)
    )
    d[1:-1] = np.where(is_nonflat, (w1 + w2) / whmean, 0.0)

    # The two end knots have only one neighbor apiece, so they instead use the formula
    # from pchipend.
    for knot_idx, next_idx in ((0, 1), (-1, -2)):
        h1 = h[knot_idx]
        h2 = h[next_idx]
        del1 = delta[knot_idx]
        del2 = delta[next_idx]

        d_end = (((2 * h1) + h2) * del1 - (h1 * del2)) / (h1 + h2)

        if np.sign(d_end) != np.sign(del1):
            # The estimate points the "wrong way", so flatten it out.
            d_end = 0.0
        elif (np.sign(del1) != np.sign(del2)) and (np.abs(d_end) > (3 * np.abs(del1))):
            # Near a local extremum, limit the slope to preserve monotonic nature.
            d_end = 3 * del1

        d[knot_idx] = d_end

    return d


@nb.njit(cache=True)
def _interp_pchip(x, y, xq):
    """
    Interpolate a calibration solution in time using cubic interpolation.

    See Moler, "Numerical Computing with MATLAB", Chap 3.6 (pchiptx.m) for further
    details.

    Parameters
    ----------
    x : ndarray of float
        Sample points at which `y` has been evaluated.  Must be in ascending order.
        Shape (Nold), dtype is float.
    y : ndarray of float
        Values at the sample points. Shape (Nold), dtype is float.
    xq : ndarray of float
        Points at which to evaluate the interpolation, shape (Nnew,), dtype is float.

    Returns
    -------
    new_cal : ndarray of float
        Interpolated solutions, shape (Nnew,).
    """
    # Note that this code has been largely adapted from the MATLAB version of the PCHIP
    # algorithm. I (Karto) have kept the variable names mostly the same to allow for
    # easier point-to-point comparison against that code.
    n = len(x)
    h = x[1:] - x[:-1]
    delta = (y[1:] - y[:-1]) / h
    d = _pchip_slopes(h, delta)

    c = ((3 * delta) - (2 * d[:-1]) - d[1:]) / h
    b = (d[:-1] - (2 * delta) + d[1:]) / (h * h)

    k = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    s = xq - x[k]

    return y[k] + s * (d[k] + s * (c[k] + (s * b[k])))


@nb.njit(cache=True)
def _interp_cubic_slices(
    old_times: FloatArray,
    old_cal: FloatArray,
    old_flags: BoolArray,
    new_times: FloatArray,
    new_cal: FloatArray,
    n_good_times: IntArray,
):
    """
    Refit the individual solutions that a batched cubic interpolation could not.

    Operates in place on `new_cal`, overwriting the solutions that had samples
    dropped and blanking those that cannot be fit at all. Solutions with nothing
    flagged are left as the batched pass produced them.

    Parameters
    ----------
    old_times : ndarray of float
        Times corresponding to the third axis of `old_cal`, shape (Ntimes,), dtype is
        float. Units must be the same as `new_times`. Must be in ascending order.
    old_cal : ndarray
        Calibration solutions to interpolate, shape (Nants, Nfreqs, Ntimes, Njones),
        dtype is float.
    old_flags : ndarray of bool
        Flags for `old_cal`, where flagged entries are excluded from the
        interpolation. Shape is (Nants, Nfreqs, Ntimes, Njones), dtype is bool.
    new_times : ndarray of float
        Times at which to evaluate the interpolation, shape (Nnew,), dtype is float.
        Units must be the same as `old_times`.
    new_cal : ndarray of float
        Interpolated solutions to update in place, shape
        (Nants, Nfreqs, Nnew, Njones).
    n_good_times : ndarray of int
        Count of unflagged samples for each solution, shape
        (Nants, Nfreqs, Njones).

    Returns
    -------
    None
        `new_cal` is modified in place.
    """
    n_old_times = len(old_times)

    for ant_idx in range(old_cal.shape[0]):
        for freq_idx in range(old_cal.shape[1]):
            for jones_idx in range(old_cal.shape[3]):
                n_good = n_good_times[ant_idx, freq_idx, jones_idx]

                if n_good == n_old_times:
                    # Assume the batched pass already filled this slice in, so skip it.
                    continue
                elif n_good < _MIN_CUBIC_SAMPLES:
                    # Too few samples left to fit, which covers solutions flagged for
                    # the whole track as well as ones only sparsely sampled. Note that
                    # these have to be blanked explicitly -- the batched pass fills
                    # every solution, flags and all, so whatever it produced here is
                    # a fit to data that should not have been used.
                    new_cal[ant_idx, freq_idx, :, jones_idx] = np.nan
                    continue

                good_mask = ~old_flags[ant_idx, freq_idx, :, jones_idx]
                new_cal[ant_idx, freq_idx, :, jones_idx] = _interp_pchip(
                    old_times[good_mask],
                    old_cal[ant_idx, freq_idx, good_mask, jones_idx],
                    new_times,
                )


def _interp_pchip_4d(old_times, old_cal, new_times):
    """
    Interpolate a UVCal-like array of solutions in time using cubic interpolation.

    Similar to `_interp_pchip`, but fitting every antenna/freq/jones in a single pass,
    which is typically one to two orders of magnitude faster than going solution by
    solution.

    Parameters
    ----------
    old_times : ndarray of float
        Times corresponding to the third axis of `old_cal`, shape (Ntimes,), dtype is
        float. Units must be the same as `new_times`. Must be in ascending order.
    old_cal : ndarray
        Calibration solutions to interpolate, shape (Nants, Nfreqs, Ntimes, Njones),
        dtype is float.
    new_times : ndarray of float
        Times at which to evaluate the interpolation, shape (Nnew,), dtype is float.
        Units must be the same as `old_times`.

    Returns
    -------
    new_cal : ndarray of float
        Interpolated solutions, shape (Nants, Nfreqs, Nnew, Njones). Flags are not
        consulted, so solutions with any flagged samples must be refit afterwards.
    """
    # Note that this code has been largely adapted from the MATLAB version of the PCHIP
    # algorithm. I (Karto) have kept the variable names mostly the same to allow for
    # easier point-to-point comparison against that code.
    n = len(old_times)
    h = np.diff(old_times)[None, None, :, None]
    delta = (old_cal[:, :, 1:, :] - old_cal[:, :, :-1, :]) / h

    d = np.zeros(old_cal.shape, dtype=np.float64)
    delta_lo = delta[:, :, :-1, :]
    delta_hi = delta[:, :, 1:, :]

    # Check if the derivative goes through a zero-crossing between or at adjacent knots.
    is_nonflat = (np.sign(delta_lo) * np.sign(delta_hi)) > 0

    w1 = (2 * h[:, :, 1:, :]) + h[:, :, :-1, :]
    w2 = h[:, :, 1:, :] + (2 * h[:, :, :-1, :])
    whmean = (w1 / np.where(is_nonflat, delta_lo, 1.0)) + (
        w2 / np.where(is_nonflat, delta_hi, 1.0)
    )
    d[:, :, 1:-1, :] = np.where(is_nonflat, (w1 + w2) / whmean, 0.0)

    # The two end knots use the noncentered three-point formula from pchipend.
    for knot_idx, next_idx in ((0, 1), (-1, -2)):
        h1 = h[:, :, knot_idx, :]
        h2 = h[:, :, next_idx, :]
        del1 = delta[:, :, knot_idx, :]
        del2 = delta[:, :, next_idx, :]

        d_end = (((2 * h1) + h2) * del1 - (h1 * del2)) / (h1 + h2)

        # Where the estimate points the "wrong way", make it flat.
        d_end = np.where(np.sign(d_end) != np.sign(del1), 0.0, d_end)

        # Near a local extremum, limit the slope to preserve monotonic nature.
        limit_slope = (
            (np.sign(del1) != np.sign(del2)) & (np.abs(d_end) > (3 * np.abs(del1)))
        ) & (np.sign(d_end) == np.sign(del1))

        d[:, :, knot_idx, :] = np.where(limit_slope, 3 * del1, d_end)

    # As in _interp_pchip, recast each interval's cubic into plain polynomial
    # coefficients and evaluate in Horner form.
    d_lo = d[:, :, :-1, :]
    d_hi = d[:, :, 1:, :]
    c = ((3 * delta) - (2 * d_lo) - d_hi) / h
    b = (d_lo - (2 * delta) + d_hi) / (h * h)

    k = np.clip(np.searchsorted(old_times, new_times) - 1, 0, n - 2)
    s = (new_times - old_times[k])[None, None, :, None]

    return old_cal[:, :, k, :] + s * (
        d[:, :, k, :] + s * (c[:, :, k, :] + (s * b[:, :, k, :]))
    )


def _interp_cubic_cal(
    old_times: FloatArray,
    old_cal: FloatArray,
    old_flags: BoolArray,
    new_times: FloatArray,
):
    """
    Interpolate calibration solutions in time using cubic interpolation.

    Parameters
    ----------
    old_times : ndarray of float
        Times corresponding to the third axis of `old_cal`, shape (Ntimes,), dtype is
        float. Units must be the same as `new_times`. Must be in ascending order.
    old_cal : ndarray
        Calibration solutions to interpolate, shape (Nants, Nfreqs, Ntimes, Njones),
        dtype is float.
    old_flags : ndarray of bool
        Flags for `old_cal`, where flagged entries are excluded from the interpolation.
        Shape is (Nants, Nfreqs, Ntimes, Njones), dtype is bool.
    new_times : ndarray of float
        Times at which to evaluate the interpolation, shape (Nnew,), dtype is float.
        Units must be the same as `old_times`.

    Returns
    -------
    new_cal : ndarray
        Interpolated solutions, shape (Nants, Nfreqs, Nnew, Njones), dtype is float.
        Entries where too few unflagged samples were available are set to NaN.
    """
    n_old_times = len(old_times)

    if n_old_times < _MIN_CUBIC_SAMPLES:
        # Too few times for anything here to be fit.
        return np.full(
            (old_cal.shape[0], old_cal.shape[1], len(new_times), old_cal.shape[3]),
            np.nan,
            dtype=old_cal.dtype,
        )

    # Number of usable samples for each antenna/freq/jones solution.
    n_good_times = n_old_times - np.sum(old_flags, axis=2)

    # Every unflagged solution shares the same time grid, so the whole array can be
    # fit in one pass, which is typically 1-2 orders of magnitude faster than going
    # slice by slice. Worst case, this adds a ~10% overhead of all solutions are flagged
    # (though the logic check below traps this worst case scenario).
    if np.any(n_good_times == n_old_times):
        new_cal = np.asarray(
            _interp_pchip_4d(old_times, old_cal, new_times), dtype=old_cal.dtype
        )
    else:
        # Nothing is fully unflagged, so the batched pass would be thrown away in
        # its entirety -- skip it and fit every solution individually instead.
        new_cal = np.full(
            (old_cal.shape[0], old_cal.shape[1], len(new_times), old_cal.shape[3]),
            np.nan,
            dtype=old_cal.dtype,
        )

    if np.any(n_good_times != n_old_times):
        # If anything is flagged, we need to step through the solutions individually to
        # refit them.
        _interp_cubic_slices(
            old_times, old_cal, old_flags, new_times, new_cal, n_good_times
        )

    return new_cal


@nb.njit(cache=True)
def _prep_matrix_poly_lsqfit_cal(
    old_var: FloatArray, new_var: FloatArray, var_order: IntArray
):
    """
    Build the fit and evaluation matrices for a polynomial least-squares fit.

    Each variable is normalized onto [-1, 1] and expanded into Chebyshev
    polynomials, which span the same space as the raw powers but keep the normal
    equations far better conditioned at higher order.

    Parameters
    ----------
    old_var : ndarray of float
        Variables to fit against, shape (Nvar, Ntimes).
    new_var : ndarray of float
        Values of those variables at the new times, shape (Nvar, Nnew).
    var_order : ndarray of int
        Polynomial order for each variable, shape (Nvar,). An order of zero causes the
        variable to be dropped from the fit.

    Returns
    -------
    xval_matrix : ndarray of float
        Fit matrix evaluated at the old times, shape (Ntimes, Nterms), where Nterms
        is `sum(var_order) + 1`.
    eval_matrix : ndarray of float
        Fit matrix evaluated at the new times, shape (Nnew, Nterms).
    """
    n_var = old_var.shape[0]
    n_terms = sum(var_order) + 1
    n_old = old_var.shape[1]
    n_new = new_var.shape[1]

    xval_matrix = np.zeros((n_old, n_terms))
    eval_matrix = np.zeros((n_new, n_terms))
    xval_matrix[:, 0] = 1
    eval_matrix[:, 0] = 1

    marker = 1
    for idx in range(n_var):
        if var_order[idx] == 0:
            # This variable contributes nothing beyond the constant term, so skip
            # it (which also avoids normalizing a potentially constant variable).
            continue
        min_var = np.min(old_var[idx])
        max_var = np.max(old_var[idx])
        old_norm_var = (2 * (old_var[idx] - min_var) / (max_var - min_var)) - 1
        new_norm_var = (2 * (new_var[idx] - min_var) / (max_var - min_var)) - 1

        # Chebyshev polynomials of the first kind, built up via the following eqn:
        # T_0 = 1, T_1 = x, T_(k+1) = (2 * x * T_k) - T_(k-1).
        old_prev = np.ones(n_old)
        new_prev = np.ones(n_new)
        old_cheb = old_norm_var
        new_cheb = new_norm_var

        for _ in range(var_order[idx]):
            xval_matrix[:, marker] = old_cheb
            eval_matrix[:, marker] = new_cheb
            marker += 1

            old_next = (2 * old_norm_var * old_cheb) - old_prev
            old_prev = old_cheb
            old_cheb = old_next

            new_next = (2 * new_norm_var * new_cheb) - new_prev
            new_prev = new_cheb
            new_cheb = new_next

    return xval_matrix, eval_matrix


@nb.njit(cache=True)
def _interp_lsqfit_cal(
    old_cal: FloatArray,
    old_flags: BoolArray,
    xval_matrix: FloatArray,
    eval_matrix: FloatArray,
):
    """
    Fit and evaluate calibration solutions using a least-squares fit.

    Solves the system of equations for each antenna/freq/jones solution independently,
    using only the unflagged samples, and then interpolate/extrapolate the fits.

    Parameters
    ----------
    old_cal : ndarray
        Calibration solutions to interpolate, shape (Nants, Nfreqs, Ntimes, Njones),
        dtype is float.
    old_flags : ndarray of bool
        Flags for `old_cal`, where flagged entries are excluded from the
        interpolation. Shape is (Nants, Nfreqs, Ntimes, Njones), dtype is bool.
    xval_matrix : ndarray of float
        Fit matrix evaluated at the "old" (solved for) times, shape (Ntimes, Nterms).
    eval_matrix : ndarray of float
        Fit matrix evaluated at the "new" (interpolated) times, shape (Nnew, Nterms).

    Returns
    -------
    new_cal : ndarray
        Interpolated solutions, shape (Nants, Nfreqs, Nnew, Njones), dtype is float.
        Entries where too few unflagged samples were available are set to NaN.
    """
    new_cal = np.full(
        (old_cal.shape[0], old_cal.shape[1], eval_matrix.shape[0], old_cal.shape[3]),
        np.nan,
        dtype=old_cal.dtype,
    )
    n_terms = xval_matrix.shape[1]
    n_flags = np.sum(old_flags, axis=2)
    n_times = old_cal.shape[2]

    # The fit matrix is the same for every fully unflagged solution, so form its
    # normal-equation matrix once up front rather than per solution.
    full_xx_matrix = xval_matrix.T @ xval_matrix

    for ant_idx in range(old_cal.shape[0]):
        for freq_idx in range(old_cal.shape[1]):
            for jones_idx in range(old_cal.shape[3]):
                if n_times - n_flags[ant_idx, freq_idx, jones_idx] < n_terms:
                    # Too few unflagged samples to constrain the fit.
                    continue

                mask = ~old_flags[ant_idx, freq_idx, :, jones_idx]
                if n_flags[ant_idx, freq_idx, jones_idx]:
                    good_matrix = xval_matrix[mask, :]
                    xx_matrix = good_matrix.T @ good_matrix
                else:
                    good_matrix, xx_matrix = xval_matrix, full_xx_matrix

                # The astype is needed because numba requires both sides of a matmul
                # to share a dtype, and single-precision solutions are common enough
                # (calfits gains are typically complex64) to hit that otherwise.
                cal_vals = old_cal[ant_idx, freq_idx, mask, jones_idx].astype(
                    xval_matrix.dtype
                )
                xy_matrix = good_matrix.T @ cal_vals

                try:
                    fit_vals = np.linalg.solve(xx_matrix, xy_matrix)
                except Exception:
                    # Thrown w/ a singular matrix, leave this solution flagged.
                    # Note that the nosec here allows getting around an issue w/ bandit,
                    # since numba only handles the exception in this fashion.
                    continue  # nosec B112

                new_cal[ant_idx, freq_idx, :, jones_idx] = eval_matrix @ fit_vals

    return new_cal


@nb.njit(cache=True)
def _interp_poly_cal(
    old_cal: FloatArray,
    old_flags: BoolArray,
    old_var: FloatArray,
    new_var: FloatArray,
    order: IntArray,
):
    """
    Interpolate calibration solutions using a least-squares polynomial fit.

    Fits a Chebyshev polynomial sequence to calibration solutions against a set of
    independent variables such as time (provided in `old_var`) to each antenna,
    frequency, and jones index/solution independently, with the interpolation then
    performed using the fitted polynomial (provided in `new_var`).

    Parameters
    ----------
    old_cal : ndarray
        Calibration solutions to fit, shape (Nants, Nfreqs, Ntimes, Njones), dtype is
        float.
    old_flags : ndarray of bool
        Flags for `old_cal`, where flagged entries are excluded from the fit. Shape is
        (Nants, Nfreqs, Ntimes, Njones), dtype is bool.
    old_var : ndarray of float
        Variables to fit against, shape (Nvar, Ntimes), dtype float. Typically the first
        set of entries correspond to time (though not strictly required).
    new_var : ndarray of float
        Values of the variables to evaluate the fit at, shape (Nvar, Nnew), dtype float.
    order : ndarray of int
        Polynomial order to use, shape (Nvar,), dtype int.

    Returns
    -------
    new_cal : ndarray
        Fit solutions evaluated at `new_times`, shape (Nants, Nfreqs, Nnew, Njones).
        Entries where too few unflagged samples were available to constrain the fit
        are set to NaN.
    """
    n_var = old_var.shape[0]

    if np.any(order < 0):
        raise ValueError("order cannot be negative.")

    if n_var != order.size:
        raise ValueError(
            "Length of order must match the first dimension of old_var and new_var."
        )

    if old_var.shape[0] != new_var.shape[0]:
        raise ValueError(
            "old_var and new_var must contain the same number of variables."
        )

    for idx in range(n_var):
        if order[idx] > 0 and np.min(old_var[idx]) == np.max(old_var[idx]):
            raise ValueError(
                f"Variable {idx} is constant, and so cannot be fit with an order "
                f"{order[idx]} polynomial."
            )

    xval_matrix, eval_matrix = _prep_matrix_poly_lsqfit_cal(old_var, new_var, order)

    return _interp_lsqfit_cal(old_cal, old_flags, xval_matrix, eval_matrix)


def _interp_dispatcher(
    *,
    mode: Literal["real", "imag", "amp", "phase"],
    old_times: FloatArray,
    old_cal: FloatArray,
    old_flags: BoolArray,
    new_times: FloatArray,
    new_cal: FloatArray | ComplexArray | None = None,
    new_flags: BoolArray | None = None,
    kind: Literal["nearest", "linear", "cubic", "poly"],
    amp_kind: Literal["nearest", "linear", "cubic", "poly"] | None = None,
    pha_kind: Literal["nearest", "linear", "cubic", "poly"] | None = None,
    poly_order: int | None = None,
    amp_poly_order: int | None = None,
    pha_poly_order: int | None = None,
    old_var: FloatArray | None = None,
    new_var: FloatArray | None = None,
    flag_nearest: bool = True,
    allow_extrapolation: bool = False,
    max_time_delta: float = 1.0,
    amp_max_time_delta: float | None = None,
    pha_max_time_delta: float | None = None,
    flag_delta: float = 0.5,
    amp_flag_delta: float | None = None,
    pha_flag_delta: float | None = None,
):
    """
    Call various interpolation functions based on the specified kind.

    Note that this is a helper function, not intended to be called directly by users.
    Use `time_interp_cal` instead. See the docstring there for details on the parameters
    and return values.
    """
    # Making this explicit here that the two items are linked
    ignore_flags = flag_nearest

    build_flags = build_interp_flags(
        old_cal,
        old_flags,
        old_times,
        new_times,
        tol_lim=flag_delta,
        is_gain_amp=(mode == "amp"),
        is_gain_pha=(mode == "phase"),
        flag_adjacent=flag_nearest,
        max_time_delta=max_time_delta,
        allow_extrapolation=allow_extrapolation,
    )

    orig_dtype = old_cal.dtype
    if mode == "real":
        old_cal = old_cal.real
    elif mode == "imag":
        old_cal = old_cal.imag
    elif mode == "phase":
        old_cal = np.unwrap(np.angle(old_cal))
        if pha_kind is not None:
            kind = pha_kind
        if pha_poly_order is not None:
            poly_order = pha_poly_order
        if pha_flag_delta is not None:
            flag_delta = pha_flag_delta
        if pha_max_time_delta is not None:
            max_time_delta = pha_max_time_delta
    elif mode == "amp":
        old_cal = np.abs(old_cal)
        if amp_kind is not None:
            kind = amp_kind
        if amp_poly_order is not None:
            poly_order = amp_poly_order
        if amp_flag_delta is not None:
            flag_delta = amp_flag_delta
        if amp_max_time_delta is not None:
            max_time_delta = amp_max_time_delta
    else:
        raise ValueError(f"Unrecognised mode '{mode}'.")

    new_flags = build_flags if new_flags is None else (new_flags | build_flags)

    if kind == "nearest":
        interp_cal = _interp_nearest_cal(
            old_times, old_cal, old_flags, new_times, ignore_flags
        )
    elif kind == "linear":
        interp_cal = _interp_linear_cal(old_times, old_cal, old_flags, new_times)
    elif kind == "cubic":
        interp_cal = _interp_cubic_cal(old_times, old_cal, old_flags, new_times)
    elif kind == "poly":
        interp_cal = _interp_poly_cal(old_cal, old_flags, old_var, new_var, poly_order)
    else:
        raise ValueError(f"Unrecognised interpolation kind '{kind}'.")

    if new_cal is None:
        new_cal = interp_cal
        if np.issubdtype(orig_dtype, np.complexfloating):
            if (mode == "real") or (mode == "amp"):
                new_cal = new_cal.astype(orig_dtype)
            elif mode == "imag":
                new_cal = np.multiply(new_cal, 1j, dtype=orig_dtype)
            elif mode == "phase":
                new_cal = np.exp(1j * new_cal, dtype=orig_dtype)
    elif mode == "real":
        new_cal.real = interp_cal
    elif mode == "imag":
        new_cal.imag = interp_cal
    elif mode == "amp":
        new_cal *= interp_cal
    elif mode == "phase":
        new_cal *= np.exp(1j * interp_cal)

    return new_cal, new_flags


# ---------------------------------------------------------------------------
# Main method
# ---------------------------------------------------------------------------


def time_interp_cal(
    old_times: FloatArray,
    old_cal: FloatArray,
    old_flags: BoolArray,
    new_times: FloatArray,
    *,
    interp_mode: Literal[
        "real", "imag", "amp", "phase", "ampphase", "complex"
    ] = "ampphase",
    kind: Literal["nearest", "linear", "cubic", "poly"] = "linear",
    amp_kind: Literal["nearest", "linear", "cubic", "poly"] | None = None,
    pha_kind: Literal["nearest", "linear", "cubic", "poly"] | None = None,
    poly_order: int = 3,
    amp_poly_order: int | None = None,
    pha_poly_order: int | None = None,
    flag_nearest: bool = True,
    allow_extrapolation: bool = False,
    flag_delta: float = 0.0,
    amp_flag_delta: float | None = None,
    pha_flag_delta: float | None = None,
    max_time_delta: float = 1.0,
    amp_max_time_delta: float | None = None,
    pha_max_time_delta: float | None = None,
    old_var: FloatArray | None = None,
    new_var: FloatArray | None = None,
):
    """
    Interpolate calibration solutions to a new set of times.

    Interpolate a set of calibration solutions (gains or delays) based on the provided
    inputs. Note that complex gains are split into pairs of real-valued quantities
    before being interpolated, since multiple interpolation methods will not work on
    complex quantities. Complex gains can either be split into their real and imaginary
    components, or into phase and amplitude, which can be more useful when phase and
    amplitude vary on different timescales. To facilitate interpolation in this context,
    several keywords have "amp" and "phase" counterparts, which are used when
    interpolating in amplitude and phase, respectively.

    Parameters
    ----------
    old_times : ndarray of float
        Times corresponding to the third axis of `old_cal`, shape (Ntimes,), dtype is
        float, units are Julian days. Must be in ascending order.
    old_cal : ndarray
        Calibration solutions to interpolate, shape (Nants, Nfreqs, Ntimes, Njones).
        Must be complex unless `interp_mode` is "real".
    old_flags : ndarray of bool
        Flags for `old_cal`, where flagged entries are excluded from the
        interpolation. Shape is (Nants, Nfreqs, Ntimes, Njones), dtype is bool.
    new_times : ndarray of float
        Times at which to evaluate the interpolation, shape (Nnew,), dtype is float,
        units are Julian days.
    interp_mode : str
        Quantity to interpolate. Options are "ampphase" (amplitude and phase,
        interpolated separately), "complex" (real and imaginary parts, interpolated
        separately), "real" (real-values only), "imag" (imaginary values only), "amp"
        (amplitude only), and "phase" (phase only). Default is "ampphase".
    kind : str
        The type of interpolation to use. Options are "nearest", "linear", "cubic",
        and "poly" (polynomial fit using Chebyshev polynomials). Default is "linear".
    amp_kind : str or None
        If set, allows the interpolation kind to be set for amplitude separately.
        Default is None, which uses the value set for `kind` for interpolating
        amplitude.
    pha_kind : str or None
        If set, allows the interpolation kind to be set for phase separately. Default is
        None, which uses the value set for `kind` for interpolating phase.
    poly_order : int or sequence of int
        Polynomial order to use when `kind` is "poly". May be provided as a single int,
        or sequence of length Nvar + 1, which allows a different order polynomial to be
        fit against variables provided via `old_var` and `new_var` (with the first entry
        corresponding to the polynomial used for time). Note that basis set used for
        fitting are Chebyshev polynomials. Default is 3 for all variables (including
        time).
    amp_poly_order : int or None
        If set, allows the polynomial order to be set for amplitude separately. Default
        is None, which uses the value set for `poly_order` for interpolating amplitude.
        Only used when `interp_mode` is either "amp" or "ampphase".
    pha_poly_order : int or None
        If set, allows the polynomial order to be set for phase separately. Default is
        None, which uses the value set for `poly_order` for interpolating phase.
        Only used when `interp_mode` is either "phase" or "ampphase".
    flag_nearest : bool
        If True, interpolated points that lie adjacent to a flagged data point in
        `old_cal` (as recorded in `old_flags`) are also flagged. If False, then flagged
        entries are simply excluded during the interpolation processes. Default is True.
    allow_extrapolation : bool
        If True, allow new times that fall outside the range of `old_times`. Default is
        False, which flags times outside of this range.
    flag_delta : float
        Largest change between adjacent samples that is considered acceptable, used
        to flag intervals that appear to have discontinuities. Units are fractional for
        amplitude, radians for phase, otherwise they are whatever units are native to
        `old_cal`. Default is 0.0, which disables this flagging.
    amp_flag_delta : float or None
        If set, allows the flagging threshold to be set for amplitude separately. Units
        are in fractional amplitude. Default is None, which uses `flag_delta` for
        flagging amplitude interpolation.  Only used when `interp_mode` is either "amp"
        or "ampphase".
    pha_flag_delta : float or None
        If set, allows the flagging threshold to be set for phase separately. Units
        are in radians. Default is None, which uses `flag_delta` for flagging phase
        interpolation. Only used when `interp_mode` is either "phase" or "ampphase".
    max_time_delta : float
        Largest gap allowed between a new time and each of the old times that bracket
        it, units are days. A new time is flagged unless it lies within this distance
        of both of its neighbors, so that a solution sitting on top of one old time is
        still flagged if the other side of the interval is too far away. New times
        beyond the range of `old_times` are instead tested against the single nearest
        old time. Default is 1.0.
    amp_max_time_delta : float or None
        As `max_time_delta`, but applied when interpolating amplitude. Units are days.
        Only used when `interp_mode` is either "amp" or "ampphase". Default is None,
        which uses the value set for `max_time_delta`.
    pha_max_time_delta : float or None
        As `max_time_delta`, but applied when interpolating phase. Units are days.
        Only used when `interp_mode` is either "phase" or "ampphase". Default is None,
        which uses the value set for `max_time_delta`.
    old_var : ndarray of float or None
        Additional variables to fit against when `kind="poly"`, shape (Nvar, Ntimes),
        dtype is float. Note that time is always included as a fit variable, and do not
        need to also be supplied here (and will in fact result in a degenerate fit).
        Default is None, which causes to fit to be against time alone.
    new_var : ndarray of float
        Values of those variables at `new_times`, shape (Nvar, Nnew), dtype is
        float. Required if `old_var` is set.

    Returns
    -------
    new_cal : ndarray of float or complex
        Interpolated solutions, shape (Nants, Nfreqs, Nnew, Njones), matching the
        dtype of `old_cal` (nominally floating or complexfloating dtypes).
    new_flags : ndarray of bool
        Flags for `new_cal`, shape (Nants, Nfreqs, Nnew, Njones).

    Raises
    ------
    ValueError
        If `interp_mode` or any of the interpolation kinds is unrecognised, if
        `interp_mode` requires complex input but `old_cal` is real-valued, if only
        one of `old_var`/`new_var` is supplied, or if their shapes are inconsistent
        with each other or with the time arrays.

    """
    valid_modes = {"real", "imag", "amp", "phase", "ampphase", "complex"}
    if interp_mode not in valid_modes:
        raise ValueError(
            f"Unrecognised interp_mode '{interp_mode}'. Choose from {valid_modes}."
        )

    if not (np.issubdtype(old_cal.dtype, np.complexfloating) or interp_mode == "real"):
        raise ValueError(
            f"interp_mode '{interp_mode}' is not compatible with real-valued old_cal."
        )

    valid_kinds = {"nearest", "linear", "cubic", "poly"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Unrecognised interpolation kind '{kind}'. Choose from {valid_kinds}."
        )

    if amp_kind is not None and amp_kind not in valid_kinds:
        raise ValueError(
            f"Unrecognised amplitude interpolation kind '{amp_kind}'. Choose from "
            f"{valid_kinds}."
        )

    if pha_kind is not None and pha_kind not in valid_kinds:
        raise ValueError(
            f"Unrecognised phase interpolation kind '{pha_kind}'. Choose from "
            f"{valid_kinds}."
        )

    # Get the time arrays properly set up
    old_times = np.asarray(old_times, dtype=float)
    new_times = np.asarray(new_times, dtype=float)

    # Numba has no support for arrays that are not in the native byte order, which
    # data read out of a FITS file frequently are not, so byte-swap those up front.
    if not old_cal.dtype.isnative:
        old_cal = old_cal.astype(old_cal.dtype.newbyteorder("="))
    if not old_flags.dtype.isnative:
        old_flags = old_flags.astype(old_flags.dtype.newbyteorder("="))

    # Note that amp_kind and pha_kind can each select a polynomial fit on their own,
    # so the fit variables have to be built whenever any of the three asks for one.
    if "poly" in (kind, amp_kind, pha_kind):
        if (old_var is None) != (new_var is None):
            raise ValueError(
                "old_var and new_var must either both be set or both be left as None."
            )
        elif old_var is None:
            # Fit against time alone, shaped as a single row so that it matches the
            # (Nvar, Ntimes) layout that the fitting routines expect.
            old_var = old_times.reshape(1, -1)
            new_var = new_times.reshape(1, -1)
        else:
            if old_var.shape[0] != new_var.shape[0]:
                raise ValueError(
                    "old_var and new_var must contain the same number of variables."
                )
            if old_var.shape[-1] != len(old_times):
                raise ValueError(
                    "old_var must have the same number of entries as old_times."
                )
            if new_var.shape[1] != len(new_times):
                raise ValueError(
                    "new_var must have the same number of entries as new_times."
                )

            old_var = np.vstack((old_times, old_var))
            new_var = np.vstack((new_times, new_var))

        # Note that this has to come after the above, since the variable count only
        # includes time once the arrays have been assembled.
        n_var = old_var.shape[0]

        try:
            poly_order = np.full(n_var, poly_order, dtype=int)
            if amp_poly_order is not None:
                amp_poly_order = np.full(n_var, amp_poly_order, dtype=int)
            if pha_poly_order is not None:
                pha_poly_order = np.full(n_var, pha_poly_order, dtype=int)
        except ValueError as err:
            if "broadcast" in str(err):
                raise ValueError(
                    "poly_order must either have 1 value or must have length equal "
                    "to the total number of variables being fit against (1 for "
                    "time plus that contained in old_var/new_var)."
                ) from err
            raise err

    interp_kwargs = {
        "old_times": old_times,
        "old_cal": old_cal,
        "old_flags": old_flags,
        "new_times": new_times,
        "kind": kind,
        "amp_kind": amp_kind,
        "pha_kind": pha_kind,
        "poly_order": poly_order,
        "amp_poly_order": amp_poly_order,
        "pha_poly_order": pha_poly_order,
        "old_var": old_var,
        "new_var": new_var,
        "flag_delta": flag_delta,
        "amp_flag_delta": amp_flag_delta,
        "pha_flag_delta": pha_flag_delta,
        "flag_nearest": flag_nearest,
        "allow_extrapolation": allow_extrapolation,
        "max_time_delta": max_time_delta,
        "amp_max_time_delta": amp_max_time_delta,
        "pha_max_time_delta": pha_max_time_delta,
    }

    if interp_mode == "complex":
        new_cal, new_flags = _interp_dispatcher(mode="real", **interp_kwargs)
        new_cal, new_flags = _interp_dispatcher(
            mode="imag", new_cal=new_cal, new_flags=new_flags, **interp_kwargs
        )
    elif interp_mode == "ampphase":
        new_cal, new_flags = _interp_dispatcher(mode="amp", **interp_kwargs)
        new_cal, new_flags = _interp_dispatcher(
            mode="phase", new_cal=new_cal, new_flags=new_flags, **interp_kwargs
        )
    else:
        new_cal, new_flags = _interp_dispatcher(mode=interp_mode, **interp_kwargs)

    return new_cal, new_flags
