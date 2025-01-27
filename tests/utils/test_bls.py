# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for baseline number utility functions."""

import importlib

import numpy as np
import pytest

import pyuvdata.utils.bls as bl_utils

hasbench = importlib.util.find_spec("pytest_benchmark") is not None


class FakeClass:
    def __init__(self):
        pass


def test_parse_ants_error():
    test_obj = FakeClass()
    with pytest.raises(
        ValueError,
        match=(
            "UVBased objects must have all the following attributes in order "
            "to call 'parse_ants': "
        ),
    ):
        bl_utils.parse_ants(test_obj, ant_str="")


def test_antnums_to_baseline_miriad_convention():
    ant1 = np.array([1, 2, 3, 1, 1, 1, 255, 256])  # Ant1 array should be 1-based
    ant2 = np.array([1, 2, 3, 254, 255, 256, 1, 2])  # Ant2 array should be 1-based
    bl_gold = np.array([257, 514, 771, 510, 511, 67840, 65281, 65538], dtype="uint64")

    n_ant = 256
    bl = bl_utils.antnums_to_baseline(
        ant1, ant2, Nants_telescope=n_ant, use_miriad_convention=True
    )
    np.testing.assert_allclose(bl, bl_gold)


@pytest.mark.filterwarnings("ignore:antnums_to_baseline")
@pytest.mark.skipif(not hasbench, reason="benchmark utility not installed")
@pytest.mark.parametrize(
    "nbls", [1, 10, 100, 1000, 10000, 100000, 1000000], ids=lambda x: f"len={x:}"
)
@pytest.mark.parametrize(
    "bl_start", [0, 2**16, 2**16 + 2**22], ids=lambda x: f"min={x:}"
)
def test_bls_to_ant(benchmark, bl_start, nbls):
    bls = np.arange(bl_start, bl_start + nbls)
    if nbls > 65535:
        bls += 65536
    nants_telescope = 2048 if bl_start < 2**16 + 2**22 else 2**16 + 2**22

    bls = np.ascontiguousarray(bls, dtype=np.uint64)

    antnums = benchmark(bl_utils._bls.baseline_to_antnums, bls)

    bls_out = bl_utils.antnums_to_baseline(
        antnums[0],
        antnums[1],
        Nants_telescope=nants_telescope,
        attempt256=bl_start < 2**16,
        use_miriad_convention=False,
    )

    assert np.array_equal(bls, bls_out)


@pytest.mark.filterwarnings("ignore:antnums_to_baseline")
@pytest.mark.skipif(not hasbench, reason="benchmark utility not installed")
@pytest.mark.parametrize(
    "nbls", [1, 10, 100, 1000, 10000, 100000, 1000000], ids=lambda x: f"len={x:}"
)
@pytest.mark.parametrize(
    "bl_start", [0, 2**16, 2**16 + 2**22], ids=lambda x: f"min={x:}"
)
def test_ants_to_bls(benchmark, bl_start, nbls):
    bls = np.arange(bl_start, bl_start + nbls)
    nants_telescope = 2048 if bl_start < 2**16 + 2**22 else 2**16 + 2**22
    if nbls > 65535:
        bls += 65536
    a1, a2 = bl_utils.baseline_to_antnums(bls, Nants_telescope=nants_telescope)

    a1 = np.ascontiguousarray(a1, dtype=np.uint64)
    a2 = np.ascontiguousarray(a2, dtype=np.uint64)

    bls_out = benchmark(
        bl_utils._bls.antnums_to_baseline,
        a1,
        a2,
        attempt256=bl_start < 2**16,
        nants_less2048=nants_telescope <= 2048,
        use_miriad_convention=False,
    )
    a1_out, a2_out = bl_utils.baseline_to_antnums(
        bls_out, Nants_telescope=nants_telescope
    )
    assert np.array_equal(a1, a1_out)
    assert np.array_equal(a2, a2_out)
