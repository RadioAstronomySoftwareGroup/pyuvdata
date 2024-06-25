# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for helper utility functions."""

import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from pyuvdata import utils
from pyuvdata.testing import check_warnings
from pyuvdata.utils import helpers

from .test_coordinates import hasmoon

if hasmoon:
    from lunarsky import MoonLocation


@pytest.mark.parametrize(
    "filename1,filename2,answer",
    [
        (["foo.uvh5"], ["bar.uvh5"], ["foo.uvh5", "bar.uvh5"]),
        (["foo.uvh5", "bar.uvh5"], ["foo.uvh5"], ["foo.uvh5", "bar.uvh5"]),
        (["foo.uvh5"], None, ["foo.uvh5"]),
        (None, ["bar.uvh5"], ["bar.uvh5"]),
        (None, None, None),
    ],
)
def test_combine_filenames(filename1, filename2, answer):
    combined_filenames = helpers._combine_filenames(filename1, filename2)
    if answer is None:
        assert combined_filenames is answer
    else:
        # use sets to test equality so that order doesn't matter
        assert set(combined_filenames) == set(answer)

    return


def test_deprecated_utils_import():
    with check_warnings(
        DeprecationWarning,
        match="The _check_histories function has moved, please import it from "
        "pyuvdata.utils.helpers. This warnings will become an error in version 3.2",
    ):
        utils._check_histories("foo", "foo")


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
    order = helpers.determine_blt_order(
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

    is_rect, time_first = helpers.determine_rectangularity(
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

    order = helpers.determine_blt_order(
        time_array=times,
        ant_1_array=ant1,
        ant_2_array=ant2,
        baseline_array=bl,
        Nbls=1,
        Ntimes=1,
    )
    assert order == ("baseline", "time")
    is_rect, time_first = helpers.determine_rectangularity(
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

    is_rect, time_first = helpers.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=9, ntimes=10
    )
    assert not is_rect

    # now, permute time instead of bls
    TIME = np.concatenate([rng.permuted(times) for i in range(len(bls))])
    BL = np.tile(bls, len(times))
    is_rect, time_first = helpers.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=9, ntimes=10
    )
    assert not is_rect

    TIME = np.array([1000.0, 1000.0, 2000.0, 1000.0])
    BLS = np.array([0, 0, 1, 0])

    is_rect, time_first = helpers.determine_rectangularity(
        time_array=TIME, baseline_array=BLS, nbls=2, ntimes=2
    )
    assert not is_rect


@pytest.mark.parametrize("err_state", ["err", "warn", "none"])
@pytest.mark.parametrize("tel_loc", ["Center", "Moon", "Earth", "Space"])
@pytest.mark.parametrize("check_frame", ["Moon", "Earth"])
@pytest.mark.parametrize("del_tel_loc", [False, None, True])
def test_check_surface_based_positions(err_state, tel_loc, check_frame, del_tel_loc):
    tel_loc_dict = {
        "Center": np.array([0, 0, 0]),
        "Moon": np.array([0, 0, 1.737e6]),
        "Earth": np.array([0, 6.37e6, 0]),
        "Space": np.array([4.22e7, 0, 0]),
    }
    tel_frame_dict = {"Moon": "mcmf", "Earth": "itrs"}

    ant_pos = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    if del_tel_loc:
        ant_pos += tel_loc_dict[tel_loc]

    fail_type = err_msg = err_type = None
    err_check = check_warnings
    if (tel_loc != check_frame) and (err_state != "none"):
        if tel_loc == "Center":
            fail_type = "below"
        elif tel_loc == "Space":
            fail_type = "above"
        else:
            fail_type = "above" if tel_loc == "Earth" else "below"

    if fail_type is not None:
        err_msg = (
            f"{tel_frame_dict[check_frame]} position vector magnitudes must be "
            f"on the order of the radius of {check_frame} -- they appear to lie well "
            f"{fail_type} this."
        )
        if err_state == "err":
            err_type = ValueError
            err_check = pytest.raises
        else:
            err_type = UserWarning

        with err_check(err_type, match=err_msg):
            status = helpers.check_surface_based_positions(
                telescope_loc=None if (del_tel_loc) else tel_loc_dict[tel_loc],
                antenna_positions=None if (del_tel_loc is None) else ant_pos,
                telescope_frame=tel_frame_dict[check_frame],
                raise_error=err_state == "err",
                raise_warning=err_state == "warn",
            )

        assert (err_state == "err") or (status == (tel_loc == check_frame))


@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
@pytest.mark.parametrize("tel_loc", ["Earth", "Moon"])
@pytest.mark.parametrize("check_frame", ["Earth", "Moon"])
def test_check_surface_based_positions_earthmoonloc(tel_loc, check_frame):
    frame = "mcmf" if (check_frame == "Moon") else "itrs"

    if tel_loc == "Earth":
        loc = EarthLocation.from_geodetic(0, 0, 0)
    else:
        loc = MoonLocation.from_selenodetic(0, 0, 0)

    if tel_loc == check_frame:
        assert helpers.check_surface_based_positions(
            telescope_loc=loc, telescope_frame=frame
        )
    else:
        with pytest.raises(ValueError, match=(f"{frame} position vector")):
            helpers.check_surface_based_positions(
                telescope_loc=[loc.x.value, loc.y.value, loc.z.value],
                telescope_frame=frame,
            )


def test_slicify():
    assert helpers.slicify(None) is None
    assert helpers.slicify(slice(None)) == slice(None)
    assert helpers.slicify([]) is None
    assert helpers.slicify([1, 2, 3]) == slice(1, 4, 1)
    assert helpers.slicify([1]) == slice(1, 2, 1)
    assert helpers.slicify([0, 2, 4]) == slice(0, 5, 2)
    assert helpers.slicify([0, 1, 2, 7]) == [0, 1, 2, 7]


@pytest.mark.parametrize(
    "obj1,obj2,union_result,interset_result,diff_result",
    [
        [[1, 2, 3], [3, 4, 5], [1, 2, 3, 4, 5], [3], [1, 2]],  # Partial overlap
        [[1, 2], [1, 2], [1, 2], [1, 2], []],  # Full overlap
        [[1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6], [], [1, 3, 5]],  # No overlap
        [[1, 2], None, [1, 2], [1, 2], [1, 2]],  # Nones
    ],
)
def test_sorted_unique_ops(obj1, obj2, union_result, interset_result, diff_result):
    assert helpers._sorted_unique_union(obj1, obj2) == union_result
    assert helpers._sorted_unique_intersection(obj1, obj2) == interset_result
    assert helpers._sorted_unique_difference(obj1, obj2) == diff_result
