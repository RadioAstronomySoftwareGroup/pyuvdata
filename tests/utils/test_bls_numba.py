# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for numba-enhanced baseline number utility functions."""

import numpy as np
import pytest

import pyuvdata.utils.bls_numba as bl_utils


@pytest.mark.parametrize("ant1mod", [0, 6, 10])
@pytest.mark.parametrize("ant2mod", [0, 3, 20])
def test_minmax_ants(ant1mod, ant2mod):
    ant1 = np.array([1, 2, 3], dtype="uint64")
    ant2 = np.array([4, 5, 6], dtype="uint64")
    ant1 += ant1mod
    ant2 += ant2mod
    assert max(3 + ant1mod, 6 + ant2mod) == bl_utils._max_ant(ant1, ant2)
    assert min(1 + ant1mod, 4 + ant2mod) == bl_utils._min_ant(ant1, ant2)


@pytest.mark.parametrize("use_miriad_convention", [True, False])
@pytest.mark.parametrize("use2048", [True, False])
@pytest.mark.parametrize("use256", [True, False])
def test_antnums_to_baseline_roundtip_numba(use_miriad_convention, use2048, use256):
    if not use2048:
        ant1_gold = np.array([1, 2, 3, 1, 1, 1, 255, 2049], dtype="uint64")
        ant2_gold = np.array([1, 2, 3, 254, 255, 2049, 1, 2], dtype="uint64")
    else:
        ant1_gold = np.array([1, 2, 3, 1, 1, 1, 255, 256], dtype="uint64")
        ant2_gold = np.array([1, 2, 3, 254, 255, 256, 1, 2], dtype="uint64")

    bl_gold = np.zeros_like(ant1_gold)
    if use_miriad_convention or use256:
        bl_gold[:] = [257, 514, 771, 510, 511, 67840, 65281, 65538]
        if not use2048:
            bl_gold[-3:] = [2151745537, 65281, 524546]
    else:
        if use2048:
            bl_gold[:] = [67585, 69634, 71683, 67838, 67839, 67840, 587777, 589826]
        else:
            bl_gold[:] = [
                2151743489,
                4299227138,
                6446710787,
                2151743742,
                2151743743,
                2151745537,
                547612590081,
                4400198254594,
            ]

    if use256:
        ant1_gold = ant1_gold[:-3]
        ant2_gold = ant2_gold[:-3]
        bl_gold = bl_gold[:-3]

    bl = bl_utils._antnums_to_baseline(
        ant1_gold,
        ant2_gold,
        use_miriad_convention=use_miriad_convention,
        use2048=use2048,
        use256=use256,
    )

    np.testing.assert_array_equal(bl, bl_gold)

    ant1, ant2 = bl_utils._baseline_to_antnums(
        bl, np.max(bl), use_miriad_convention=use_miriad_convention
    )

    np.testing.assert_array_equal(ant1_gold, ant1)
    np.testing.assert_array_equal(ant2_gold, ant2)


def test_antnums_to_baseline_vec():
    ant1 = np.array([1, 2, 3, 1, 1, 1, 255, 256], dtype="uint64")
    ant2 = np.array([1, 2, 3, 254, 255, 256, 1, 2], dtype="uint64")
    bl_gold = np.array(
        [67585, 69634, 71683, 67838, 67839, 67840, 587777, 589826], dtype="uint64"
    )

    offset = np.uint64(65536)
    modulus = np.uint64(2048)
    bl = bl_utils._antnums_to_baseline_vec(ant1, ant2, offset, modulus)
    np.testing.assert_array_equal(bl, bl_gold)
