# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for the baseline-time axis."""

import numpy as np

from . import times as time_utils, tools


def determine_blt_order(
    *,
    time_array,
    ant_1_array,
    ant_2_array,
    baseline_array,
    Nbls,  # noqa: N803
    Ntimes,  # noqa: N803
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
        zip(times[1:], ant1[1:], ant2[1:], bls[1:], strict=True), start=1
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

    Note that blt_order being (time, baseline) does *not* guarantee
    rectangularity, even when Nblts == Nbls * Ntimes, since if `autos_first = True` is
    set on `reorder_blts`, then it will still set the blt_order attribute to
    (time, baseline), but they will not strictly be in that order (since it will
    actually be in autos-first order).
    """
    # check if the data is rectangular
    time_first = True
    bl_first = True

    if time_array.size != nbls * ntimes:
        return False, False
    elif nbls * ntimes == 1 or nbls == 1:
        return True, True
    elif ntimes == 1:
        return True, False
    elif blt_order == ("baseline", "time"):
        # Note that the opposite isn't true: time/baseline ordering does not always mean
        # that we have rectangularity, because of the autos_first keyword to
        # reorder_blts.
        return True, True

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


def _select_blt_preprocess(
    *,
    select_antenna_nums,
    select_antenna_names,
    bls,
    times,
    time_range,
    lsts,
    lst_range,
    blt_inds,
    phase_center_ids,
    antenna_names,
    antenna_numbers,
    ant_1_array,
    ant_2_array,
    baseline_array,
    time_array,
    time_tols,
    lst_array,
    lst_tols,
    phase_center_id_array,
):
    """Build up blt_inds and selections list for _select_preprocess.

    Parameters
    ----------
    select_antenna_nums : array_like of int, optional
        The antennas numbers to keep in the object (antenna positions and
        names for the removed antennas will be retained unless
        `keep_all_metadata` is False). This cannot be provided if
        `select_antenna_names` is also provided.
    select_antenna_names : array_like of str, optional
        The antennas names to keep in the object (antenna positions and
        names for the removed antennas will be retained unless
        `keep_all_metadata` is False). This cannot be provided if
        `select_antenna_nums` is also provided.
    bls : list of 2-tuples, optional
        A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) specifying
        baselines to keep in the object. The ordering of the numbers within the
        tuple does not matter. Note that this is different than what can be
        passed to the parameter of the same name on `select` -- this parameter
        does not accept 3-tuples or baseline numbers.
    times : array_like of float, optional
        The times to keep in the object, each value passed here should exist
        in the time_array. Cannot be used with `time_range`, `lsts`, or
        `lst_array`.
    time_range : array_like of float, optional
        The time range in Julian Date to keep in the object, must be length
        2. Some of the times in the object should fall between the first and
        last elements. Cannot be used with `times`, `lsts`, or `lst_array`.
    lsts : array_like of float, optional
        The local sidereal times (LSTs) to keep in the object, each value
        passed here should exist in the lst_array. Cannot be used with
        `times`, `time_range`, or `lst_range`.
    lst_range : array_like of float, optional
        The local sidereal time (LST) range in radians to keep in the
        object, must be of length 2. Some of the LSTs in the object should
        fall between the first and last elements. If the second value is
        smaller than the first, the LSTs are treated as having phase-wrapped
        around LST = 2*pi = 0, and the LSTs kept on the object will run from
        the larger value, through 0, and end at the smaller value.
    phase_center_ids : array_like of int, optional
        Phase center IDs to keep on the object (effectively a selection on
        baseline-times). Cannot be used with `catalog_names`.
    blt_inds : array_like of int, optional
        The baseline-time indices to keep in the object. This is
        not commonly used.
    ant_1_array : array_like of int
        Array of first antenna numbers to select on.
    ant_2_array : array_like of int
        Array of second antenna numbers to select on.
    baseline_array : array_like of int
        Array of baseline numbers to select on.
    time_array : array_like of float
        Array of times in JD to select on.
    lst_array : array_like of float
        Array of lsts in radians to select on.
    phase_center_id_array : array_like of int
        Array of phase center IDs to select on.

    Returns
    -------
    blt_inds : list of int
        list of baseline-time indices to keep. Can be None (to keep everything).
    selections : list of str
        list of selections done.
    """
    # Antennas, times and blt_inds all need to be combined into a set of
    # blts indices to keep.
    selections = []
    # test for blt_inds presence before adding inds from antennas & times
    if blt_inds is not None:
        blt_inds = tools._get_iterable(blt_inds)
        if np.array(blt_inds).ndim > 1:
            blt_inds = np.array(blt_inds).flatten()
        selections.append("baseline-times")

    if phase_center_ids is not None:
        phase_center_ids = np.array(tools._get_iterable(phase_center_ids))
        pc_blt_inds = np.nonzero(np.isin(phase_center_id_array, phase_center_ids))[0]
        if blt_inds is not None:
            # Use intersection (and) to join phase_center_ids
            # with blt_inds
            blt_inds = np.array(
                list(set(blt_inds).intersection(pc_blt_inds)), dtype=np.int64
            )
        else:
            blt_inds = pc_blt_inds

    if select_antenna_names is not None:
        if select_antenna_nums is not None:
            raise ValueError(
                "Only one of antenna_nums and antenna_names can be provided."
            )

        if not isinstance(select_antenna_names, list | tuple | np.ndarray):
            select_antenna_names = (select_antenna_names,)
        if np.array(select_antenna_names).ndim > 1:
            select_antenna_names = np.array(select_antenna_names).flatten()
        select_antenna_nums = []
        for s in select_antenna_names:
            if s not in antenna_names:
                raise ValueError(
                    f"Antenna name {s} is not present in the antenna_names" " array"
                )
            select_antenna_nums.append(
                antenna_numbers[np.where(np.array(antenna_names) == s)][0]
            )

    if select_antenna_nums is not None:
        select_antenna_nums = tools._get_iterable(select_antenna_nums)
        select_antenna_nums = np.asarray(select_antenna_nums)
        if select_antenna_nums.ndim > 1:
            select_antenna_nums = select_antenna_nums.flatten()
        selections.append("antennas")
        # Check to make sure that we actually have these antenna nums in the data
        ant_check = np.logical_or(
            np.isin(select_antenna_nums, ant_1_array),
            np.isin(select_antenna_nums, ant_2_array),
        )
        if not np.all(ant_check):
            raise ValueError(
                f"Antenna number {select_antenna_nums[~ant_check]} is not present "
                "in the ant_1_array or ant_2_array"
            )
        ant_blt_inds = np.where(
            np.logical_and(
                np.isin(ant_1_array, select_antenna_nums),
                np.isin(ant_2_array, select_antenna_nums),
            )
        )[0]
        ant_blt_inds = np.asarray(ant_blt_inds, dtype=np.int64)
    else:
        ant_blt_inds = None

    if bls is not None:
        bls_blt_inds = np.zeros(0, dtype=np.int64)
        for bl in bls:
            wh1 = np.where(np.logical_and(ant_1_array == bl[0], ant_2_array == bl[1]))[
                0
            ]
            if len(wh1) > 0:
                bls_blt_inds = np.append(bls_blt_inds, list(wh1))
            else:
                wh2 = np.where(
                    np.logical_and(ant_1_array == bl[1], ant_2_array == bl[0])
                )[0]

                if len(wh2) > 0:
                    bls_blt_inds = np.append(bls_blt_inds, list(wh2))
                else:
                    raise ValueError(
                        f"Antenna pair {bl} does not have any data "
                        "associated with it."
                    )
        selections.append("antenna pairs")

        if ant_blt_inds is not None:
            # Use intersection (and) to join antenna_names/nums & ant_pairs_nums
            ant_blt_inds = np.intersect1d(ant_blt_inds, bls_blt_inds)
        else:
            ant_blt_inds = bls_blt_inds

    if ant_blt_inds is not None:
        if blt_inds is not None:
            # Use intersection (and) to join antenna_names/nums/ant_pairs_nums
            # with blt_inds
            blt_inds = np.intersect1d(blt_inds, ant_blt_inds)
        else:
            blt_inds = ant_blt_inds

    time_blt_inds, time_selections = time_utils._select_times_helper(
        times=times,
        time_range=time_range,
        lsts=lsts,
        lst_range=lst_range,
        obj_time_array=time_array,
        obj_time_range=None,
        obj_lst_array=lst_array,
        obj_lst_range=None,
        time_tols=time_tols,
        lst_tols=lst_tols,
    )

    if time_blt_inds is not None:
        selections.extend(time_selections)
        if blt_inds is not None:
            # Use intesection (and) to join
            # antenna_names/nums/ant_pairs_nums/blt_inds with times
            blt_inds = np.intersect1d(blt_inds, time_blt_inds)
        else:
            blt_inds = time_blt_inds

    if blt_inds is not None:
        if len(blt_inds) == 0:
            raise ValueError("No baseline-times were found that match criteria")
        if max(blt_inds) >= baseline_array.size:
            raise ValueError("blt_inds contains indices that are too large")
        if min(blt_inds) < 0:
            raise ValueError("blt_inds contains indices that are negative")

        blt_inds = sorted(set(blt_inds))

    return blt_inds, selections
