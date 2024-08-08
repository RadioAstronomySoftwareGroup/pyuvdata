# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for baseline-time axis utility functions."""

import numpy as np
import pytest

from pyuvdata import utils


@pytest.mark.parametrize(
    "blt_order",
    [
        ("time", "baseline"),
        ("baseline", "time"),
        ("ant1", "time"),
        ("ant2", "time"),
        ("time", "ant1"),
        ("time", "ant2"),
        ("baseline",),
        ("time",),
        ("ant1",),
        ("ant2",),
        (),
        ([0, 2, 6, 4, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11, 13, 15, 17]),
    ],
)
def test_determine_blt_order(blt_order):
    nant = 3
    ntime = 2

    def getbl(ant1, ant2):
        return utils.antnums_to_baseline(ant1, ant2, Nants_telescope=nant)

    def getantbls():
        # Arrange them backwards so by default they are NOT sorted
        ant1 = np.arange(nant, dtype=int)[::-1]
        ant2 = ant1.copy()
        ANT1, ANT2 = np.meshgrid(ant1, ant2)

        return ANT1.flatten(), ANT2.flatten()

    def gettimebls(blt_order):
        ant1, ant2 = getantbls()
        time_array = np.linspace(
            2000, 1000, ntime
        )  # backwards so not sorted by default

        TIME = np.tile(time_array, len(ant1))
        ANT1 = np.repeat(ant1, len(time_array))
        ANT2 = np.repeat(ant2, len(time_array))
        BASELINE = getbl(ANT1, ANT2)

        lc = locals()
        if isinstance(blt_order, list):
            inds = np.array(blt_order)
        elif blt_order:
            inds = np.lexsort(tuple(lc[k.upper()] for k in blt_order[::-1]))
        else:
            inds = np.arange(len(TIME))

        return TIME[inds], ANT1[inds], ANT2[inds], BASELINE[inds]

    # time, bl
    TIME, ANT1, ANT2, BL = gettimebls(blt_order)
    order = utils.bltaxis.determine_blt_order(
        time_array=TIME,
        ant_1_array=ANT1,
        ant_2_array=ANT2,
        baseline_array=BL,
        Nbls=nant**2,
        Ntimes=ntime,
    )
    if isinstance(blt_order, list):
        assert order is None
    elif blt_order:
        assert order == blt_order
    else:
        assert order is None

    is_rect, time_first = utils.bltaxis.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=nant**2, ntimes=ntime
    )
    if blt_order in [("ant1", "time"), ("ant2", "time")]:
        # sorting by ant1/ant2 then time means we split the other ant into a
        # separate group
        assert not is_rect
        assert not time_first
    elif isinstance(blt_order, list):
        assert not is_rect
        assert not time_first
    else:
        assert is_rect
        assert time_first == (
            (len(blt_order) == 2 and blt_order[-1] == "time")
            or (len(blt_order) == 1 and blt_order[0] != "time")
            or not blt_order  # we by default move time first (backwards, but still)
        )


def test_determine_blt_order_size_1():
    times = np.array([2458119.5])
    ant1 = np.array([0])
    ant2 = np.array([1])
    bl = utils.antnums_to_baseline(ant1, ant2, Nants_telescope=2)

    order = utils.bltaxis.determine_blt_order(
        time_array=times,
        ant_1_array=ant1,
        ant_2_array=ant2,
        baseline_array=bl,
        Nbls=1,
        Ntimes=1,
    )
    assert order == ("baseline", "time")
    is_rect, time_first = utils.bltaxis.determine_rectangularity(
        time_array=times, baseline_array=bl, nbls=1, ntimes=1
    )
    assert is_rect
    assert time_first


def test_determine_rect_time_first():
    times = np.linspace(2458119.5, 2458120.5, 10)
    ant1 = np.arange(3)
    ant2 = np.arange(3)
    ANT1, ANT2 = np.meshgrid(ant1, ant2)
    bls = utils.antnums_to_baseline(ANT1.flatten(), ANT2.flatten(), Nants_telescope=3)

    rng = np.random.default_rng(12345)

    TIME = np.tile(times, len(bls))
    BL = np.concatenate([rng.permuted(bls) for i in range(len(times))])

    is_rect, time_first = utils.bltaxis.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=9, ntimes=10
    )
    assert not is_rect

    # now, permute time instead of bls
    TIME = np.concatenate([rng.permuted(times) for i in range(len(bls))])
    BL = np.tile(bls, len(times))
    is_rect, time_first = utils.bltaxis.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=9, ntimes=10
    )
    assert not is_rect

    TIME = np.array([1000.0, 1000.0, 2000.0, 1000.0])
    BLS = np.array([0, 0, 1, 0])

    is_rect, time_first = utils.bltaxis.determine_rectangularity(
        time_array=TIME, baseline_array=BLS, nbls=2, ntimes=2
    )
    assert not is_rect
