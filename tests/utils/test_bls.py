# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for baseline number utility functions."""

import numpy as np
import pytest

import pyuvdata.utils.bls as bl_utils


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
