# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Numba-enhanced utilities for baseline numbers."""

import numba
import numpy as np
import numpy.typing as npt


@numba.njit()
def _max_ant(ant1: npt.NDArray[np.int64], ant2: npt.NDArray[np.int64]) -> np.int64:
    return np.maximum(np.max(ant1), np.max(ant2))


@numba.njit()
def _min_ant(ant1: npt.NDArray[np.int64], ant2: npt.NDArray[np.int64]) -> np.int64:
    return np.minimum(np.min(ant1), np.min(ant2))


@numba.njit()
def _baseline_to_antnums(
    baselines: npt.NDArray[np.uint64],
    max_baseline: np.uint64,
    use_miriad_convention: bool = False,
) -> npt.NDArray[np.uint64]:
    if max_baseline < np.uint64(65536):
        offset = np.uint64(0)
        bitmask = np.uint64(255)  # 2**8 - 1 (all bits)
        bitshift = np.uint64(8)
    elif max_baseline < np.uint64(4_259_840):
        offset = np.uint64(65536)  # (2 ** 16)
        bitmask = np.uint64(2047)  # 2**11 - 1 (all bits below the 11th)
        bitshift = np.uint64(11)
    else:
        offset = np.uint64(4_259_840)  # (2 ** 16) + (2 ** 22)
        bitmask = np.uint64(2_147_483_647)  # 2**32 - 1 (all bits below the 32nd)
        bitshift = np.uint64(31)

    nbls = baselines.shape[0]
    ant_arr = np.empty((2, nbls), dtype=np.uint64)
    a1 = ant_arr[0]
    a2 = ant_arr[1]

    # Go through entry by entry. Note that we use bitshift and bitmasking here b/c
    # after removing the offset, the ant1 and ant2 values are just encoded within a
    # set of bits in the baseline value
    for idx, bl in enumerate(baselines):
        bl = bl - offset
        a1[idx] = bl >> bitshift
        a2[idx] = bl & bitmask

    if use_miriad_convention:
        for idx, bl in enumerate(baselines):
            # Since for MIRIAD the ant numbers are >= 1, the sum of the offset plus the
            # bitmask plus 1 corresponds to baseline 1-1, the first bl number possible
            # with the 'new' numbering scheme.
            if bl <= (offset + bitmask):  # (Baseline 1-1) - 1
                a1[idx] = bl >> np.uint64(8)
                a2[idx] = bl & np.uint64(255)

    return ant_arr


@numba.vectorize("uint64(uint64,uint64,uint64,uint64)")
def _antnums_to_baseline_vec(ant1, ant2, offset, modulus):
    return (modulus * ant1) + ant2 + offset


@numba.njit()
def _antnums_to_baseline(
    ant1: npt.NDArray[np.uint64],
    ant2: npt.NDArray[np.uint64],
    use256: bool = False,
    use2048: bool = True,
    use_miriad_convention: bool = False,
) -> npt.NDArray[np.uint64]:
    if use256:
        offset = np.uint64(0)
        modulus = np.uint64(256)
    elif use2048:
        offset = np.uint64(65536)
        modulus = np.uint64(2048)
    else:
        offset = np.uint64(4_259_840)
        modulus = np.uint64(2_147_483_648)

    # miriad convention is special so get that out of the way quickly.
    if use_miriad_convention and not use256:
        bl_out = _antnums_to_baseline_vec(ant1, ant2, np.uint64(0), np.uint64(256))
        for index, ant2_val in enumerate(ant2):
            if ant2_val > np.uint64(255):  # MIRIAD uses 1-index antenna IDs
                bl_out[index] = modulus * ant1[index] + ant2_val + offset

        return bl_out
    else:
        return _antnums_to_baseline_vec(ant1, ant2, offset, modulus)
