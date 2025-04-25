# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import re

import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from pyuvdata import Telescope, UVCal
from pyuvdata.uvcal.initializers import new_uvcal, new_uvcal_from_uvdata
from pyuvdata.uvdata.initializers import new_uvdata

from ..utils.test_coordinates import selenoids


@pytest.fixture(scope="function")
def uvd_kw():
    return {
        "freq_array": np.linspace(100e6, 200e6, 10),
        "times": np.linspace(2459850, 2459851, 12),
        "telescope": Telescope.new(
            location=EarthLocation.from_geodetic(0, 0, 0),
            name="mock",
            instrument="mock",
            mount_type="fixed",
            antenna_positions={
                0: [0.0, 0.0, 0.0],
                1: [0.0, 0.0, 1.0],
                2: [0.0, 0.0, 2.0],
            },
        ),
        "polarization_array": np.array([-5, -6, -7, -8]),
    }


@pytest.fixture(scope="function")
def uvc_only_kw():
    return {
        "cal_style": "redundant",
        "gain_convention": "multiply",
        "x_orientation": "n",
        "jones_array": "linear",
        "cal_type": "gain",
    }


@pytest.fixture(scope="function")
def uvc_simplest():
    return {
        "freq_array": np.linspace(100e6, 200e6, 10),
        "time_array": np.linspace(2459850, 2459851, 12),
        "telescope": Telescope.new(
            location=EarthLocation.from_geodetic(0, 0, 0),
            name="mock",
            x_orientation="n",
            feeds=["x", "y"],
            mount_type="fixed",
            antenna_positions={
                0: [0.0, 0.0, 0.0],
                1: [0.0, 0.0, 1.0],
                2: [0.0, 0.0, 2.0],
            },
        ),
        "cal_style": "redundant",
        "gain_convention": "multiply",
        "jones_array": "linear",
        "cal_type": "gain",
    }


@pytest.fixture(scope="function")
def uvc_simplest_moon():
    pytest.importorskip("lunarsky")
    from lunarsky import MoonLocation

    return {
        "freq_array": np.linspace(100e6, 200e6, 10),
        "time_array": np.linspace(2459850, 2459851, 12),
        "telescope": Telescope.new(
            location=MoonLocation.from_selenodetic(0, 0, 0),
            name="mock",
            x_orientation="n",
            feeds=["x", "y"],
            mount_type="fixed",
            antenna_positions={
                0: [0.0, 0.0, 0.0],
                1: [0.0, 0.0, 1.0],
                2: [0.0, 0.0, 2.0],
            },
        ),
        "cal_style": "redundant",
        "gain_convention": "multiply",
        "jones_array": "linear",
        "cal_type": "gain",
    }


@pytest.mark.parametrize("selenoid", selenoids)
def test_new_uvcal_simple_moon(uvc_simplest_moon, selenoid):
    uvc_simplest_moon["telescope"].location.ellipsoid = selenoid
    uvc = UVCal.new(**uvc_simplest_moon)
    assert uvc.telescope._location.frame == "mcmf"
    assert uvc.telescope._location.ellipsoid == selenoid
    assert uvc.telescope.location == uvc_simplest_moon["telescope"].location
    assert uvc.telescope.location.ellipsoid == selenoid


def test_new_uvcal_time_range(uvc_simplest):
    tdiff = np.mean(np.diff(uvc_simplest["time_array"]))
    tstarts = uvc_simplest["time_array"] - tdiff / 2
    tends = uvc_simplest["time_array"] + tdiff / 2
    uvc_simplest["time_range"] = np.stack((tstarts, tends), axis=1)
    del uvc_simplest["time_array"]

    UVCal.new(**uvc_simplest)

    uvc_simplest["integration_time"] = tdiff * 86400
    uvc = UVCal.new(**uvc_simplest)
    assert uvc.Ntimes == 12

    uvc_simplest["integration_time"] = np.full((5,), tdiff * 86400)
    with pytest.raises(
        ValueError,
        match="integration_time must be the same length as the first axis of "
        "time_range.",
    ):
        uvc = UVCal.new(**uvc_simplest)


@pytest.mark.parametrize(
    ["update_dict", "err_msg"],
    [
        [{"ant_array": [0, 1, 2, 3]}, "The following ants are not in antenna_numbers"],
        [
            {"cal_style": "sky"},
            "If cal_style is 'sky', ref_antenna_name and sky_catalog must be provided.",
        ],
        [
            {"cal_style": "wrong", "ref_antenna_name": "mock", "sky_catalog": "mock"},
            "cal_style must be 'redundant' or 'sky', got",
        ],
        [{"bad_kwarg": True}, "Unrecognized keyword argument"],
        [
            {"freq_range": [100e6, 200e6]},
            re.escape("Provide *either* freq_range *or* freq_array"),
        ],
        [{"freq_array": None}, "You must provide either freq_array"],
        [
            {"cal_type": "wrong", "freq_range": [150e6, 180e6], "freq_array": None},
            "cal_type must be either 'gain' or 'delay'",
        ],
        [{"telescope": None}, "telescope must be a pyuvdata.Telescope object."],
        [
            {
                "telescope": Telescope.new(
                    location=EarthLocation.from_geodetic(0, 0, 0),
                    name="mock",
                    antenna_positions={
                        0: [0.0, 0.0, 0.0],
                        1: [0.0, 0.0, 1.0],
                        2: [0.0, 0.0, 2.0],
                    },
                )
            },
            "feed_array must be set on the Telescope object passed to `telescope`.",
        ],
    ],
)
def test_new_uvcal_bad_inputs(uvc_simplest, update_dict, err_msg):
    uvc_simplest.update(update_dict)
    with pytest.raises(ValueError, match=err_msg):
        new_uvcal(**uvc_simplest)


def test_new_uvcal_jones_array(uvc_simplest):
    uvc = {k: v for k, v in uvc_simplest.items() if k != "jones_array"}

    lin = new_uvcal(jones_array="linear", **uvc)
    assert lin.Njones == 4

    circ = new_uvcal(jones_array="circular", **uvc)
    assert circ.Njones == 4
    np.testing.assert_allclose(circ.jones_array, np.array([-1, -2, -3, -4]))

    custom = new_uvcal(jones_array=np.array([-1, -3]), **uvc)
    assert custom.Njones == 2

    linear_alt = new_uvcal(jones_array=["xx", "yy"], **uvc)
    assert linear_alt.Njones == 2
    np.testing.assert_allclose(linear_alt.jones_array, np.array([-5, -6]))

    linear_physical = new_uvcal(jones_array=["nn", "ee", "ne", "en"], **uvc)
    assert linear_physical.Njones == 4
    np.testing.assert_allclose(linear_physical.jones_array, np.array([-5, -6, -7, -8]))


def test_new_uvcal_set_sky(uvc_simplest):
    uvc = {k: v for k, v in uvc_simplest.items() if k != "cal_style"}

    sk = new_uvcal(cal_style="sky", ref_antenna_name="mock", sky_catalog="mock", **uvc)
    assert sk.cal_style == "sky"
    assert sk.ref_antenna_name == "mock"
    assert sk.sky_catalog == "mock"


def test_new_uvcal_set_extra_keywords(uvc_simplest):
    uvc = new_uvcal(extra_keywords={"test": "test", "test2": "test2"}, **uvc_simplest)
    assert uvc.extra_keywords["test"] == "test"
    assert uvc.extra_keywords["test2"] == "test2"


def test_new_uvcal_set_empty(uvc_simplest):
    uvc = new_uvcal(empty=True, **uvc_simplest)
    assert uvc.flag_array.dtype == bool


def test_new_uvcal_set_delay(uvc_simplest):
    uvc = {k: v for k, v in uvc_simplest.items() if k not in ("freq_array", "cal_type")}
    new = new_uvcal(
        delay_array=np.linspace(1, 10, 10), freq_range=[150e6, 180e6], empty=True, **uvc
    )
    assert new.cal_type == "delay"
    assert new.delay_array.shape[1] == new.Nspws
    assert new.quality_array is None


def test_new_uvcal_from_uvdata(uvd_kw, uvc_only_kw):
    uvd = new_uvdata(**uvd_kw)

    uvc = new_uvcal_from_uvdata(uvd, **uvc_only_kw)

    assert np.all(uvc.time_array == uvd_kw["times"])
    assert np.all(uvc.freq_array == uvd_kw["freq_array"])
    assert uvc.telescope.name == uvd_kw["telescope"].name

    uvc = new_uvcal_from_uvdata(uvd, time_array=uvd_kw["times"][:-1], **uvc_only_kw)
    assert np.all(uvc.time_array == uvd_kw["times"][:-1])

    uvc = new_uvcal_from_uvdata(
        uvd, freq_array=uvd_kw["freq_array"][:-1], **uvc_only_kw
    )
    assert np.all(uvc.freq_array == uvd_kw["freq_array"][:-1])

    uvc = new_uvcal_from_uvdata(
        uvd,
        antenna_positions={
            0: uvd_kw["telescope"].antenna_positions[0],
            1: uvd_kw["telescope"].antenna_positions[1],
        },
        antenna_diameters=[10.0, 10.0],
        mount_type=["fixed", "fixed"],
        **uvc_only_kw,
    )

    assert np.all(
        uvc.telescope.antenna_positions[0] == uvd_kw["telescope"].antenna_positions[0]
    )
    assert len(uvc.telescope.antenna_positions) == 2

    uvd.telescope.antenna_diameters = np.zeros(uvd.telescope.Nants, dtype=float) + 5.0
    uvc = new_uvcal_from_uvdata(uvd, **uvc_only_kw)
    assert np.all(uvc.telescope.antenna_diameters == uvd.telescope.antenna_diameters)


def test_new_uvcal_from_uvdata_errors(uvd_kw, uvc_only_kw):
    uvd = new_uvdata(**uvd_kw)

    uvc_only_kw.pop("x_orientation")
    with pytest.raises(
        ValueError,
        match=("Telescope feed info must be provided if not set on the UVData object."),
    ):
        new_uvcal_from_uvdata(uvd, **uvc_only_kw)


def test_new_uvcal_set_freq_range_for_gain_type(uvd_kw, uvc_only_kw):
    uvd = new_uvdata(**uvd_kw)
    uvc = new_uvcal_from_uvdata(uvd, freq_range=(150e6, 170e6), **uvc_only_kw)
    assert uvc.freq_range is None


def test_new_uvcal_set_spwarray_and_flexspwid(uvd_kw, uvc_only_kw):
    uvd = new_uvdata(**uvd_kw)
    uvc = new_uvcal_from_uvdata(uvd, spw_array=np.array([0]), **uvc_only_kw)
    assert np.all(uvc.spw_array == np.array([0]))

    with pytest.raises(
        ValueError,
        match=(
            "spw_array must be the same length as the number of unique spws "
            "in the UVData object"
        ),
    ):
        new_uvcal_from_uvdata(uvd, spw_array=np.array([0, 1]), **uvc_only_kw)


def test_new_uvcal_get_freq_range_without_spwids(uvd_kw, uvc_only_kw):
    uvd = new_uvdata(**uvd_kw)

    uvc_only_kw["cal_type"] = "delay"
    uvd.flex_spw_id_array = None

    uvc = new_uvcal_from_uvdata(uvd, **uvc_only_kw)
    assert uvc.freq_range.min() == uvd.freq_array.min()
    assert uvc.freq_range.max() == uvd.freq_array.max()


@pytest.mark.parametrize("diameters", ["uvdata", "kwargs", None, "both"])
def test_new_uvcal_from_uvdata_specify_numbers_names(uvd_kw, uvc_only_kw, diameters):
    uvd = new_uvdata(**uvd_kw)

    if diameters in ["uvdata", "both"]:
        uvd.telescope.antenna_diameters = (
            np.zeros(uvd.telescope.Nants, dtype=float) + 5.0
        )
    elif diameters in ["kwargs", "both"]:
        uvc_only_kw["antenna_diameters"] = np.zeros(1, dtype=float) + 5.0

    with pytest.raises(
        ValueError, match="Cannot specify both antenna_numbers and antenna_names"
    ):
        new_uvcal_from_uvdata(
            uvd,
            antenna_numbers=uvd.telescope.antenna_numbers,
            antenna_names=uvd.telescope.antenna_names,
            **uvc_only_kw,
        )

    uvc = new_uvcal_from_uvdata(
        uvd, antenna_numbers=uvd.telescope.antenna_numbers[:1], **uvc_only_kw
    )
    uvc2 = new_uvcal_from_uvdata(
        uvd, antenna_names=uvd.telescope.antenna_names[:1], **uvc_only_kw
    )
    if diameters is not None:
        assert np.all(uvc.telescope.antenna_diameters == 5.0)

    uvc.history = uvc2.history
    assert uvc == uvc2


def test_new_uvcal_with_history(uvd_kw, uvc_only_kw):
    uvd = new_uvdata(**uvd_kw)
    uvc = new_uvcal_from_uvdata(uvd, history="my substring", **uvc_only_kw)
    assert "my substring" in uvc.history


def test_new_uvcal_ant_array_list(uvd_kw, uvc_only_kw):
    uvd = new_uvdata(**uvd_kw)
    uvc = new_uvcal_from_uvdata(uvd, ant_array=[1, 2, 3], **uvc_only_kw)
    assert np.array_equal(np.array([1, 2], dtype=np.uint64), uvc.ant_array)
