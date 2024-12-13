# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for phase center catalog utility functions."""

import os

import pytest

import pyuvdata.utils.phase_center_catalog as ps_cat_utils
from pyuvdata import UVData
from pyuvdata.data import DATA_PATH

casa_tutorial_uvfits = os.path.join(
    DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits"
)


def test_generate_new_phase_center_id_errs():
    with pytest.raises(ValueError, match="Cannot specify old_id if no catalog"):
        ps_cat_utils.generate_new_phase_center_id(old_id=1)

    with pytest.raises(ValueError, match="Provided cat_id was found in reserved_ids"):
        ps_cat_utils.generate_new_phase_center_id(cat_id=1, reserved_ids=[1, 2, 3])


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_look_in_catalog_missing_entries():
    casa_uvfits = UVData()
    casa_uvfits.read(casa_tutorial_uvfits)
    phase_cat = casa_uvfits.phase_center_catalog

    # Try that this works normally if we do nothing
    assert ps_cat_utils.look_in_catalog(
        phase_cat, cat_name=phase_cat[0]["cat_name"]
    ) == (0, 5)

    # Now delete some keys
    for value in phase_cat.values():
        if "cat_times" in value:
            del value["cat_times"]
    # Now re-run the above and verify things work as expected
    assert ps_cat_utils.look_in_catalog(
        phase_cat, cat_name=phase_cat[0]["cat_name"]
    ) == (0, 5)
