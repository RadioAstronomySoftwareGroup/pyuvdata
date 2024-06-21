# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for the baseline-time axis."""

import numpy as np


def determine_blt_order(
    *, time_array, ant_1_array, ant_2_array, baseline_array, Nbls, Ntimes  # noqa: N803
) -> tuple[str] | None:
    """Get the blt order from analysing metadata."""
    times = time_array
    ant1 = ant_1_array
    ant2 = ant_2_array
    bls = baseline_array

    time_bl = True
    time_a = True
    time_b = True
    bl_time = True
    a_time = True
    b_time = True
    bl_order = True
    a_order = True
    b_order = True
    time_order = True

    if Nbls == 1 and Ntimes == 1:
        return ("baseline", "time")  # w.l.o.g.

    for i, (t, a, b, bl) in enumerate(
        zip(times[1:], ant1[1:], ant2[1:], bls[1:]), start=1
    ):
        on_bl_boundary = i % Nbls == 0
        on_time_boundary = i % Ntimes == 0

        if t < times[i - 1]:
            time_bl = False
            time_a = False
            time_b = False
            time_order = False

            if not on_time_boundary:
                bl_time = False
                a_time = False
                b_time = False

            if bl == bls[i - 1]:
                bl_time = False
            if a == ant1[i - 1]:
                a_time = False
            if b == ant2[i - 1]:
                b_time = False

        elif t == times[i - 1]:
            if bl < bls[i - 1]:
                time_bl = False
            if a < ant1[i - 1]:
                time_a = False
            if b < ant2[i - 1]:
                time_b = False

        if bl < bls[i - 1]:
            bl_time = False
            bl_order = False
            if not on_bl_boundary:
                time_bl = False
        if a < ant1[i - 1]:
            a_time = False
            a_order = False
            if not on_bl_boundary:
                time_a = False
        if b < ant2[i - 1]:
            b_time = False
            b_order = False
            if not on_bl_boundary:
                time_b = False

        if not any(
            (
                time_bl,
                time_a,
                time_b,
                time_bl,
                bl_time,
                a_time,
                b_time,
                bl_order,
                a_order,
                b_order,
                time_order,
            )
        ):
            break

    if Nbls > 1 and Ntimes > 1:
        assert not (
            (time_bl and bl_time)
            or (time_a and a_time)
            or (time_b and b_time)
            or (time_order and a_order)
            or (time_order and b_order)
            or (a_order and b_order)
            or (time_order and bl_order)
        ), (
            "Something went wrong when trying to determine the order of the blts axis. "
            "Please raise an issue on github, as this is not meant to happen."
            "None of the following should ever be True: \n"
            f"\ttime_bl and bl_time: {time_bl and bl_time}\n"
            f"\ttime_a and a_time: {time_a and a_time}\n"
            f"\ttime_b and b_time: {time_b and b_time}\n"
            f"\ttime_order and a_order: {time_order and a_order}\n"
            f"\ttime_order and b_order: {time_order and b_order}\n"
            f"\ta_order and b_order: {a_order and b_order}\n"
            f"\ttime_order and bl_order: {time_order and bl_order}\n\n"
            "Please include the following information in your issue:\n"
            f"Nbls: {Nbls}\n"
            f"Ntimes: {Ntimes}\n"
            f"TIMES: {times}\n"
            f"ANT1: {ant1}\n"
            f"ANT2: {ant2}\n"
            f"BASELINES: {bls}\n"
        )

    if time_bl:
        return ("time", "baseline")
    if bl_time:
        return ("baseline", "time")
    if time_a:
        return ("time", "ant1")
    if a_time:
        return ("ant1", "time")
    if time_b:
        return ("time", "ant2")
    if b_time:
        return ("ant2", "time")
    if bl_order:
        return ("baseline",)
    if a_order:
        return ("ant1",)
    if b_order:
        return ("ant2",)
    if time_order:
        return ("time",)

    return None


def determine_rectangularity(
    *,
    time_array: np.ndarray,
    baseline_array: np.ndarray,
    nbls: int,
    ntimes: int,
    blt_order: str | tuple[str] | None = None,
):
    """Determine if the data is rectangular or not.

    Parameters
    ----------
    time_array : array_like
        Array of times in JD.
    baseline_array : array_like
        Array of baseline integers.
    nbls : int
        Number of baselines.
    ntimes : int
        Number of times.
    blt_order : str or tuple of str, optional
        If known, pass the blt_order, which can short-circuit the determination
        of rectangularity.

    Returns
    -------
    is_rect : bool
        True if the data is rectangular, False otherwise.
    time_axis_faster_than_bls : bool
        True if the data is rectangular and the time axis is the last axis (i.e. times
        change first, then bls). False either if baselines change first, OR if it is
        not rectangular.

    Notes
    -----
    Rectangular data is defined as data for which using regular slicing of size Ntimes
    or Nbls will give you either all the same time and all different baselines, or
    vice versa. This does NOT require that the baselines and times are sorted within
    that structure.
    """
    # check if the data is rectangular
    time_first = True
    bl_first = True

    if time_array.size != nbls * ntimes:
        return False, False
    elif nbls * ntimes == 1:
        return True, True
    elif nbls == 1:
        return True, True
    elif ntimes == 1:
        return True, False
    elif blt_order == ("baseline", "time"):
        return True, True
    elif blt_order == ("time", "baseline"):
        return True, False

    # That's all the easiest checks.
    if time_array[1] == time_array[0]:
        time_first = False
    if baseline_array[1] == baseline_array[0]:
        bl_first = False
    if not time_first and not bl_first:
        return False, False

    if time_first:
        time_array = time_array.reshape((nbls, ntimes))
        baseline_array = baseline_array.reshape((nbls, ntimes))
        if np.sum(np.abs(np.diff(time_array, axis=0))) != 0:
            return False, False
        if (np.diff(baseline_array, axis=1) != 0).any():
            return False, False
        return True, True
    elif bl_first:
        time_array = time_array.reshape((ntimes, nbls))
        baseline_array = baseline_array.reshape((ntimes, nbls))
        if np.sum(np.abs(np.diff(time_array, axis=1))) != 0:
            return False, False
        if (np.diff(baseline_array, axis=0) != 0).any():
            return False, False
        return True, False
