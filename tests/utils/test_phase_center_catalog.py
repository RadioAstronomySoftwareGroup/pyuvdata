# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for phase center catalog utility functions."""

import pytest

import pyuvdata.utils.phase_center_catalog as ps_cat_utils


def test_generate_new_phase_center_id_errs():
    with pytest.raises(ValueError, match="Cannot specify old_id if no catalog"):
        ps_cat_utils.generate_new_phase_center_id(old_id=1)

    with pytest.raises(ValueError, match="Provided cat_id was found in reserved_ids"):
        ps_cat_utils.generate_new_phase_center_id(cat_id=1, reserved_ids=[1, 2, 3])
