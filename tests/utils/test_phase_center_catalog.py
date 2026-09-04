# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for phase center catalog utility functions."""

import numpy as np
import pytest

import pyuvdata.utils.phase_center_catalog as ps_cat_utils


def test_near_field_cat_times():
    """A timed near-field entry keeps its track as (Npts,) arrays."""
    entry = ps_cat_utils.generate_phase_center_cat_entry(
        cat_name="some_sat",
        cat_type="near_field",
        cat_lon=[0.4, 0.5, 0.6],
        cat_lat=[-0.3, -0.2, -0.1],
        cat_dist=[1e-13, 2e-13, 3e-13],
        cat_times=[0.0, 1.0, 2.0],
    )

    for key in ["cat_times", "cat_lon", "cat_lat", "cat_dist"]:
        assert entry[key].shape == (3,)

    np.testing.assert_allclose(entry["cat_lon"], [0.4, 0.5, 0.6], rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs,msg",
    (
        [
            {"cat_lat": -0.3, "cat_dist": 1e-13},
            "cat_lon, cat_lat and cat_dist must all be set",
        ],
        [
            {"cat_lon": 0.4, "cat_dist": 1e-13},
            "cat_lon, cat_lat and cat_dist must all be set",
        ],
        [
            {"cat_lon": 0.4, "cat_lat": -0.3},
            "cat_lon, cat_lat and cat_dist must all be set",
        ],
        [
            {
                "cat_lon": [0.4, 0.5],
                "cat_lat": [-0.3, -0.2],
                "cat_dist": [1e-13, 2e-13],
                "cat_times": [0.0, 0.0],
            },
            "cat_times cannot contain duplicate values",
        ],
        [
            {"cat_lon": [0.4], "cat_lat": [-0.3, -0.2], "cat_dist": [1e-13, 2e-13]},
            "Object properties -- lon, lat, pm_ra, pm_dec, dist, vrad",
        ],
        # A non-finite range would otherwise put NaNs in the w-coordinate and the
        # visibilities without complaint.
        [
            {
                "cat_lon": [0.4, 0.5],
                "cat_lat": [-0.3, -0.2],
                "cat_dist": [1e-13, np.nan],
            },
            "cat_dist must be finite",
        ],
        [
            {"cat_lon": [0.4, 0.5], "cat_lat": [-0.3, -0.2], "cat_dist": [1e-13, 0.0]},
            "cat_dist must be positive",
        ],
        # The range is checked for a fixed focus too, which astropy only catches for
        # negative values and only via a confusing message about parallax.
        [
            {"cat_lon": 0.4, "cat_lat": -0.3, "cat_dist": -1e-13, "cat_times": None},
            "cat_dist must be positive",
        ],
    ),
)
def test_near_field_cat_times_errs(kwargs, msg):
    """A moving near-field focus needs a complete, unambiguous track."""
    kwargs.setdefault("cat_times", [0.0, 1.0])

    with pytest.raises(ValueError, match=msg):
        ps_cat_utils.generate_phase_center_cat_entry(
            cat_name="some_sat", cat_type="near_field", **kwargs
        )


def test_generate_new_phase_center_id_errs():
    with pytest.raises(ValueError, match="Cannot specify old_id if no catalog"):
        ps_cat_utils.generate_new_phase_center_id(old_id=1)

    with pytest.raises(ValueError, match="Provided cat_id was found in reserved_ids"):
        ps_cat_utils.generate_new_phase_center_id(cat_id=1, reserved_ids=[1, 2, 3])


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_look_in_catalog_missing_entries(casa_uvfits):
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
