# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests of in-memory initialization of UVData objects."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from pyuvdata import Telescope, UVData
from pyuvdata.utils import polnum2str
from pyuvdata.uvdata.initializers import (
    configure_blt_rectangularity,
    get_freq_params,
    get_spw_params,
    get_time_params,
)

from ..utils.test_coordinates import selenoids


@pytest.fixture(scope="function")
def simplest_working_params() -> dict[str, Any]:
    return {
        "freq_array": np.linspace(1e8, 2e8, 100),
        "polarization_array": ["xx", "yy"],
        "telescope": Telescope.new(
            location=EarthLocation.from_geodetic(0, 0, 0),
            name="test",
            instrument="test",
            antenna_positions={
                0: [0.0, 0.0, 0.0],
                1: [0.0, 0.0, 1.0],
                2: [0.0, 0.0, 2.0],
            },
        ),
        "times": np.linspace(2459855, 2459856, 20),
    }


@pytest.fixture
def lunar_simple_params() -> dict[str, Any]:
    pytest.importorskip("lunarsky")
    from lunarsky import MoonLocation

    return {
        "freq_array": np.linspace(1e8, 2e8, 100),
        "polarization_array": ["xx", "yy"],
        "telescope": Telescope.new(
            location=MoonLocation.from_selenodetic(0, 0, 0),
            name="test",
            instrument="test",
            antenna_positions={
                0: [0.0, 0.0, 0.0],
                1: [0.0, 0.0, 1.0],
                2: [0.0, 0.0, 2.0],
            },
        ),
        "times": np.linspace(2459855, 2459856, 20),
    }


@pytest.mark.parametrize("selenoid", selenoids)
def test_lunar_simple_new_uvdata(lunar_simple_params: dict[str, Any], selenoid: str):
    lunar_simple_params["telescope"].location.ellipsoid = selenoid
    uvd = UVData.new(**lunar_simple_params)

    assert uvd.telescope._location.frame == "mcmf"
    assert uvd.telescope._location.ellipsoid == selenoid


@pytest.mark.parametrize(
    ["update_dict", "err_msg"],
    [
        [{"vis_units": "foo"}, "vis_units must be one of"],
        [{"derp": "foo"}, "Keyword argument derp is not a valid UVData attribute"],
        [{"telescope": None}, "telescope must be a pyuvdata.Telescope object."],
        [
            {
                "telescope": Telescope.new(
                    location=EarthLocation.from_geodetic(0, 0, 0),
                    name="test",
                    antenna_positions={
                        0: [0.0, 0.0, 0.0],
                        1: [0.0, 0.0, 1.0],
                        2: [0.0, 0.0, 2.0],
                    },
                )
            },
            "instrument must be set on the Telescope object passed to `telescope`.",
        ],
        [
            {
                "antpairs": np.array([(0, 1), (0, 2), (1, 2), (0, 3)]),
                "do_blt_outer": False,
            },
            "Length of time array must match the number of antpairs.",
        ],
    ],
)
def test_bad_inputs(simplest_working_params: dict[str, Any], update_dict, err_msg):
    simplest_working_params.update(update_dict)
    with pytest.raises(ValueError, match=err_msg):
        UVData.new(**simplest_working_params)


@pytest.mark.parametrize(
    ["update_dict", "blt_order"],
    [
        [{}, ("time", "ant1")],
        [{"time_axis_faster_than_bls": True}, ("ant1", "ant2")],
        [
            {
                "antpairs": np.array([(0, 1), (0, 2), (1, 2), (0, 1)]),
                "times": np.array([2459855, 2459855, 2459855.5, 2459855.5]),
                "integration_time": np.full((4,), 12.0, dtype=float),
                "do_blt_outer": False,
            },
            None,
        ],
    ],
)
def test_blt_order(simplest_working_params, update_dict, blt_order):
    simplest_working_params.update(update_dict)
    uvd = UVData.new(**simplest_working_params)
    assert uvd.blt_order == blt_order


def test_bad_time_inputs(simplest_working_params: dict[str, Any]):
    with pytest.raises(ValueError, match="time_array must be a numpy array"):
        get_time_params(
            telescope_location=simplest_working_params["telescope"].location,
            time_array="hello this is a string",
        )

    with pytest.raises(
        TypeError, match="integration_time must be array_like of floats"
    ):
        get_time_params(
            telescope_location=simplest_working_params["telescope"].location,
            integration_time={"a": "dict"},
            time_array=simplest_working_params["times"],
        )

    with pytest.raises(
        ValueError, match="integration_time must be the same shape as time_array"
    ):
        get_time_params(
            integration_time=np.ones(len(simplest_working_params["times"]) + 1),
            telescope_location=simplest_working_params["telescope"].location,
            time_array=simplest_working_params["times"],
        )


def test_bad_freq_inputs(simplest_working_params: dict[str, Any]):
    badp = {k: v for k, v in simplest_working_params.items() if k != "freq_array"}
    with pytest.raises(ValueError, match="freq_array must be a numpy array"):
        UVData.new(freq_array="hello this is a string", **badp)

    badp = {k: v for k, v in simplest_working_params.items() if k != "channel_width"}
    with pytest.raises(TypeError, match="channel_width must be array_like of floats"):
        UVData.new(channel_width={"a": "dict"}, **badp)

    badp = {k: v for k, v in simplest_working_params.items() if k != "channel_width"}
    with pytest.raises(
        ValueError, match="channel_width must be the same shape as freq_array"
    ):
        UVData.new(
            channel_width=np.ones(len(simplest_working_params["freq_array"]) + 1),
            **badp,
        )


def test_bad_rectangularity_inputs():
    with pytest.raises(
        ValueError,
        match="If times and antpairs differ in length, times must all be unique",
    ):
        configure_blt_rectangularity(
            times=np.array([0, 1, 2, 3, 3]),
            antpairs=np.array([(0, 1), (0, 2), (1, 2), (0, 1)]),
        )

    with pytest.raises(
        ValueError,
        match="If times and antpairs differ in length, antpairs must all be unique",
    ):
        configure_blt_rectangularity(
            times=np.array([0, 1, 2, 3, 4]),
            antpairs=np.array([(0, 1), (0, 2), (1, 2), (1, 2)]),
        )

    with pytest.raises(ValueError, match="It is impossible to determine"):
        configure_blt_rectangularity(
            times=np.array([0, 1, 2, 3]),
            antpairs=np.array([(0, 1), (0, 2), (1, 2), (0, 3)]),
        )

    with pytest.raises(
        ValueError, match="times must be unique if do_blt_outer is True"
    ):
        configure_blt_rectangularity(
            times=np.array([0, 1, 2, 3, 3]),
            antpairs=np.array([(0, 1), (0, 2), (1, 2), (0, 1), (0, 2)]),
            do_blt_outer=True,
        )

    with pytest.raises(
        ValueError, match="antpairs must be unique if do_blt_outer is True"
    ):
        configure_blt_rectangularity(
            times=np.array([0, 1, 2, 3, 4]),
            antpairs=np.array([(0, 1), (0, 2), (1, 2), (0, 1), (0, 1)]),
            do_blt_outer=True,
        )

    with pytest.raises(
        ValueError, match="blts_are_rectangular is True, but times and antpairs"
    ):
        configure_blt_rectangularity(
            times=np.array([0, 1, 2]),
            antpairs=np.array([(0, 1), (0, 2), (1, 2), (0, 2)]),
            blts_are_rectangular=True,
            do_blt_outer=False,
        )


def test_alternate_time_inputs():
    loc = EarthLocation.from_geodetic(0, 0, 0)

    time_array = np.linspace(2459855, 2459856, 20)
    integration_time = (time_array[1] - time_array[0]) * 24 * 60 * 60

    times, ints = get_time_params(
        time_array=time_array, integration_time=integration_time, telescope_location=loc
    )
    times2, ints2 = get_time_params(
        time_array=time_array,
        integration_time=integration_time * np.ones_like(time_array),
        telescope_location=loc,
    )

    uvd = UVData()
    np.testing.assert_allclose(
        times, times2, rtol=uvd._time_array.tols[0], atol=uvd._time_array.tols[1]
    )
    np.testing.assert_allclose(
        ints,
        ints2,
        rtol=uvd._integration_time.tols[0],
        atol=uvd._integration_time.tols[1],
    )

    times3, ints3 = get_time_params(time_array=time_array, telescope_location=loc)
    np.testing.assert_allclose(
        times, times3, rtol=uvd._time_array.tols[0], atol=uvd._time_array.tols[1]
    )
    np.testing.assert_allclose(
        ints,
        ints3,
        rtol=uvd._integration_time.tols[0],
        atol=uvd._integration_time.tols[1],
    )

    # Single time
    with pytest.warns(
        UserWarning, match="integration_time not provided, and cannot be inferred"
    ):
        _, ints4 = get_time_params(time_array=time_array[:1], telescope_location=loc)
    np.testing.assert_allclose(
        ints4,
        1.0,
        rtol=uvd._integration_time.tols[0],
        atol=uvd._integration_time.tols[1],
    )


def test_alternate_freq_inputs():
    freq_array = np.linspace(1e8, 2e8, 15)
    channel_width = freq_array[1] - freq_array[0]

    freqs, widths = get_freq_params(freq_array, channel_width=channel_width)

    freqs2, widths2 = get_freq_params(
        freq_array=freq_array, channel_width=channel_width * np.ones_like(freq_array)
    )
    uvd = UVData()
    np.testing.assert_allclose(
        freqs, freqs2, rtol=uvd._freq_array.tols[0], atol=uvd._freq_array.tols[1]
    )
    np.testing.assert_allclose(
        widths,
        widths2,
        rtol=uvd._channel_width.tols[0],
        atol=uvd._channel_width.tols[1],
    )

    freqs3, widths3 = get_freq_params(freq_array=freq_array)
    np.testing.assert_allclose(
        freqs, freqs3, rtol=uvd._freq_array.tols[0], atol=uvd._freq_array.tols[1]
    )
    np.testing.assert_allclose(
        widths,
        widths3,
        rtol=uvd._channel_width.tols[0],
        atol=uvd._channel_width.tols[1],
    )

    # Single frequency
    with pytest.warns(
        UserWarning, match="channel_width not provided, and cannot be inferred"
    ):
        _, widths4 = get_freq_params(freq_array=freq_array[:1])
    np.testing.assert_allclose(
        widths4, 1.0, rtol=uvd._channel_width.tols[0], atol=uvd._channel_width.tols[1]
    )


def test_empty(simplest_working_params: dict[str, Any]):
    uvd = UVData.new(empty=True, **simplest_working_params)

    assert uvd.data_array.shape == (uvd.Nblts, uvd.Nfreqs, uvd.Npols)
    assert uvd.flag_array.shape == uvd.data_array.shape == uvd.nsample_array.shape
    assert not np.any(uvd.flag_array)
    assert np.all(uvd.nsample_array == 1)
    assert np.all(uvd.data_array == 0)


def test_passing_data(simplest_working_params: dict[str, Any]):
    uvd = UVData.new(empty=True, **simplest_working_params)
    shape = uvd.data_array.shape

    uvd = UVData.new(
        data_array=np.zeros(shape, dtype=complex), **simplest_working_params
    )

    assert np.all(uvd.data_array == 0)
    assert np.all(uvd.flag_array == 0)
    assert np.all(uvd.nsample_array == 1)

    uvd = UVData.new(
        data_array=np.zeros(shape, dtype=complex),
        flag_array=np.ones(shape, dtype=bool),
        **simplest_working_params,
    )

    assert np.all(uvd.data_array == 0)
    assert np.all(uvd.flag_array)
    assert np.all(uvd.nsample_array == 1)

    uvd = UVData.new(
        data_array=np.zeros(shape, dtype=complex),
        flag_array=np.ones(shape, dtype=bool),
        nsample_array=np.ones(shape, dtype=float),
        **simplest_working_params,
    )

    assert np.all(uvd.data_array == 0)
    assert np.all(uvd.flag_array)
    assert np.all(uvd.nsample_array == 1)


def test_passing_bad_data(simplest_working_params: dict[str, Any]):
    uvd = UVData.new(empty=True, **simplest_working_params)
    shape = uvd.data_array.shape

    with pytest.raises(ValueError, match="Data array shape"):
        uvd = UVData.new(
            data_array=np.zeros((1, 2, 3), dtype=float), **simplest_working_params
        )

    with pytest.raises(ValueError, match="Flag array shape"):
        uvd = UVData.new(
            data_array=np.zeros(shape, dtype=complex),
            flag_array=np.ones((1, 2, 3), dtype=float),
            **simplest_working_params,
        )

    with pytest.raises(ValueError, match="nsample array shape"):
        uvd = UVData.new(
            data_array=np.zeros(shape, dtype=complex),
            flag_array=np.ones(shape, dtype=bool),
            nsample_array=np.ones((1, 2, 3), dtype=float),
            **simplest_working_params,
        )


def test_passing_kwargs(simplest_working_params: dict[str, Any]):
    uvd = UVData.new(blt_order=("time", "baseline"), **simplest_working_params)

    assert uvd.blt_order == ("time", "baseline")


def test_blt_rect():
    utimes = np.linspace(2459855, 2459856, 20)
    uaps = np.array([[1, 1], [1, 2], [2, 3]])

    nbls, ntimes, rect, axis, times, bls, _ = configure_blt_rectangularity(
        times=utimes, antpairs=uaps, time_axis_faster_than_bls=False
    )

    assert nbls == 3
    assert ntimes == 20
    assert rect
    assert not axis
    assert len(times) == len(bls)
    assert times[1] == times[0]

    nbls, ntimes, rect, axis, times, bls, _ = configure_blt_rectangularity(
        times=utimes, antpairs=uaps, time_axis_faster_than_bls=True
    )

    assert nbls == 3
    assert ntimes == 20
    assert rect
    assert axis
    assert len(times) == len(bls)
    assert times[1] != times[0]

    TIMES = np.repeat(utimes, len(uaps))
    BLS = np.tile(uaps, (len(utimes), 1))

    nbls, ntimes, rect, axis, times, bls, _ = configure_blt_rectangularity(
        times=TIMES, antpairs=BLS, blts_are_rectangular=True
    )

    assert nbls == 3
    assert ntimes == 20
    assert rect
    assert not axis
    assert len(times) == len(bls)
    assert times[1] == times[0]

    TIMES = np.tile(utimes, len(uaps))
    BLS = np.repeat(uaps, len(utimes), axis=0)

    nbls, ntimes, rect, axis, times, bls, _ = configure_blt_rectangularity(
        times=TIMES, antpairs=BLS, blts_are_rectangular=True
    )

    assert nbls == 3
    assert ntimes == 20
    assert rect
    assert axis
    assert len(times) == len(bls)
    assert times[1] != times[0]

    nbls, ntimes, rect, axis, times, bls, _ = configure_blt_rectangularity(
        times=TIMES, antpairs=BLS, blts_are_rectangular=False
    )

    assert nbls == 3
    assert ntimes == 20
    assert not rect
    assert not axis
    assert len(times) == len(bls)
    assert times[1] != times[0]

    nbls, ntimes, rect, axis, times, bls, _ = configure_blt_rectangularity(
        times=TIMES, antpairs=BLS
    )

    assert nbls == 3
    assert ntimes == 20
    assert rect
    assert axis
    assert len(times) == len(bls)
    assert times[1] != times[0]


def test_set_phase_params(simplest_working_params):
    obj = UVData.new(**simplest_working_params)

    pcc = obj.phase_center_catalog

    new = UVData.new(phase_center_catalog=pcc, **simplest_working_params)
    assert new.phase_center_catalog == pcc

    pccnew = copy.deepcopy(pcc)
    pccnew[1] = copy.deepcopy(pccnew[0])
    pccnew[1]["cat_type"] = "driftscan"
    pccnew[1]["cat_name"] = "another_unprojected"

    with pytest.raises(
        ValueError,
        match=(
            "If phase_center_catalog has more than one key, "
            "phase_center_id_array must be provided"
        ),
    ):
        UVData.new(phase_center_catalog=pccnew, **simplest_working_params)


def test_get_spw_params():
    idarray = np.array([0, 0, 0, 0, 0])
    freq = np.linspace(0, 1, 5)

    _id, spw = get_spw_params(flex_spw_id_array=idarray, freq_array=freq)
    assert np.all(spw == 0)

    idarray = np.array([0, 0, 0, 0, 1])
    _id, spw = get_spw_params(flex_spw_id_array=idarray, freq_array=freq)
    assert np.all(spw == [0, 1])

    with pytest.raises(
        ValueError,
        match="If spw_array has more than one entry, flex_spw_id_array must be",
    ):
        get_spw_params(spw_array=np.array([0, 1]))

    _id, spw = get_spw_params(spw_array=np.array([1]), freq_array=np.zeros(10))
    assert len(_id) == 10
    assert len(spw) == 1
    assert np.all(_id) == 1

    # Passing both spw_array and flex_spws, but getting them right
    _id, spw = get_spw_params(
        spw_array=np.array([0, 1]),
        flex_spw_id_array=np.concatenate(
            (np.zeros(10, dtype=int), np.ones(10, dtype=int))
        ),
    )
    assert len(spw) == 2

    # Passing both spw_array and flex_spws, but getting them wrong
    with pytest.raises(
        ValueError,
        match="spw_array and flex_spw_id_array must have the same number of unique",
    ):
        _id, spw = get_spw_params(
            spw_array=np.array([0, 1]), flex_spw_id_array=np.zeros(10, dtype=int)
        )


def test_passing_directional_pols(simplest_working_params):
    kw = {**simplest_working_params, **{"polarization_array": ["ee"]}}

    with pytest.raises(KeyError, match="'ee'"):
        UVData.new(**kw)

    uvd = UVData.new(x_orientation="east", **kw)
    assert polnum2str(uvd.polarization_array[0], x_orientation="east") == "ee"
