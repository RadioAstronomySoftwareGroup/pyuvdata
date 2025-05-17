# Copyright (c) 2023 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for calh5 object"""

import os

import h5py
import numpy as np
import pytest
from astropy.units import Quantity

from pyuvdata import UVCal, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.uvcal import FastCalH5Meta
from pyuvdata.uvdata import FastUVH5Meta

from ..utils.test_coordinates import selenoids
from . import extend_jones_axis, time_array_to_time_range


@pytest.mark.parametrize("time_range", [True, False])
def test_calh5_write_read_loop_gain(gain_data, tmp_path, time_range):
    calobj = gain_data
    if time_range:
        calobj = time_array_to_time_range(calobj)

    # add total_quality_array so that can be tested as well
    calobj.total_quality_array = np.ones(
        calobj._total_quality_array.expected_shape(calobj)
    )
    # add instrument
    calobj.telescope.instrument = calobj.telescope.name

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)
    calobj2 = UVCal.from_file(write_file)

    assert calobj == calobj2

    cal_meta = FastCalH5Meta(write_file)
    calobj3 = UVCal()
    calobj3.read_calh5(cal_meta)

    assert calobj == calobj3


def test_calh5_write_read_loop_multi_spw_gain(multi_spw_gain, tmp_path):
    calobj = multi_spw_gain

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)
    calobj2 = UVCal.from_file(write_file)

    assert calobj == calobj2


def test_calh5_write_read_loop_wideband_gain(wideband_gain, tmp_path):
    calobj = wideband_gain

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)
    calobj2 = UVCal.from_file(write_file)

    assert calobj == calobj2


@pytest.mark.parametrize("time_range", [True, False])
def test_calh5_write_read_loop_delay(delay_data, tmp_path, time_range):
    calobj = delay_data
    if time_range:
        calobj = time_array_to_time_range(calobj)

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)
    calobj2 = UVCal.from_file(write_file)

    assert calobj == calobj2


def test_calh5_write_read_loop_multi_spw_delay(multi_spw_delay, tmp_path):
    calobj = multi_spw_delay

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)
    calobj2 = UVCal.from_file(write_file)

    assert calobj == calobj2


def test_calh5_loop_bitshuffle(gain_data, tmp_path):
    pytest.importorskip("hdf5plugin")

    calobj = gain_data
    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True, data_compression="bitshuffle")
    calobj2 = UVCal.from_file(write_file)

    assert calobj == calobj2


@pytest.mark.parametrize("selenoid", selenoids)
def test_calh5_loop_moon(tmp_path, gain_data, selenoid):
    pytest.importorskip("lunarsky")
    from lunarsky import MoonLocation

    cal_in = gain_data

    enu_antpos = utils.ENU_from_ECEF(
        (cal_in.telescope.antenna_positions + cal_in.telescope._location.xyz()),
        center_loc=cal_in.telescope.location,
    )
    cal_in.telescope.location = MoonLocation.from_selenodetic(
        lat=cal_in.telescope.location.lat,
        lon=cal_in.telescope.location.lon,
        height=cal_in.telescope.location.height,
        ellipsoid=selenoid,
    )

    new_full_antpos = utils.ECEF_from_ENU(
        enu=enu_antpos, center_loc=cal_in.telescope.location
    )
    cal_in.telescope.antenna_positions = (
        new_full_antpos - cal_in.telescope._location.xyz()
    )
    cal_in.set_lsts_from_time_array()
    cal_in.check()

    write_file = str(tmp_path / "outtest.calh5")
    cal_in.write_calh5(write_file, clobber=True)

    cal_out = UVCal.from_file(write_file)

    assert cal_out.telescope._location.frame == "mcmf"
    assert cal_out.telescope._location.ellipsoid == selenoid

    assert cal_in == cal_out


def test_calh5_meta(gain_data, tmp_path):
    calobj = gain_data

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)
    cal_meta = FastCalH5Meta(write_file)

    cal_meta2 = FastCalH5Meta(write_file)
    assert cal_meta == cal_meta2

    write_file2 = str(tmp_path / "outtest2.calh5")
    calobj.write_calh5(write_file, clobber=True)

    cal_meta2 = FastCalH5Meta(write_file2)
    assert cal_meta != cal_meta2

    uvh5_meta = FastUVH5Meta(os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5"))
    assert cal_meta != uvh5_meta

    ant_nums = cal_meta.antenna_numbers
    jpol_nums = cal_meta.jones_array
    jpol_names = cal_meta.pols

    assert cal_meta.has_key(ant_nums[5])
    assert cal_meta.has_key(ant_nums[5], jpol_nums[0])
    assert cal_meta.has_key(ant_nums[5], jpol_names[0])

    assert not cal_meta.has_key(600)
    assert not cal_meta.has_key(ant_nums[5], "ll")

    np.testing.assert_allclose(
        Quantity(list(cal_meta.telescope_location_obj.geocentric)).to("m").value,
        Quantity(list(calobj.telescope.location.geocentric)).to("m").value,
        rtol=0,
        atol=1e-3,
    )

    # remove history to test adding pyuvdata version
    cal_meta.history = ""
    calobj2 = cal_meta.to_uvcal()
    calobj3 = calobj.copy(metadata_only=True)
    calobj3.history = calobj2.history

    assert calobj2 == calobj3


@pytest.mark.parametrize("time_range", [True, False])
def test_calh5_no_lsts(gain_data, tmp_path, time_range):
    calobj = gain_data
    if time_range:
        calobj = time_array_to_time_range(calobj)

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)

    # remove lst_array from file; check that it's correctly computed on read
    with h5py.File(write_file, "r+") as h5f:
        if time_range:
            del h5f["/Header/lst_range"]
        else:
            del h5f["/Header/lst_array"]

    calobj2 = UVCal.from_file(write_file)

    assert calobj == calobj2


def test_none_extra_keywords(gain_data, tmp_path):
    """Test that we can round-trip None values in extra_keywords"""
    cal_obj = gain_data
    test_calh5 = UVCal()
    testfile = str(tmp_path / "none_extra_keywords.calh5")

    cal_obj.extra_keywords["foo"] = None

    cal_obj.write_calh5(testfile)
    test_calh5.read(testfile)

    assert test_calh5 == cal_obj

    # also confirm dataset is empty/null
    with h5py.File(testfile, "r") as h5f:
        assert h5f["Header/extra_keywords/foo"].shape is None

    return


def test_read_write_calh5_errors(gain_data, tmp_path):
    """
    Test raising errors in write_calh5 function.
    """
    cal_obj = gain_data

    cal_out = UVCal()
    testfile = str(tmp_path / "outtest.calh5")
    with open(testfile, "a"):
        os.utime(testfile, None)

    # assert IOError if file exists
    with pytest.raises(IOError, match="File exists; skipping"):
        cal_obj.write_calh5(testfile, clobber=False)

    # use clobber=True to write out anyway
    cal_obj.write_calh5(testfile, clobber=True)
    cal_out.read(testfile)

    # make sure filenames are what we expect
    assert cal_obj.filename == ["zen.2457698.40355.xx.gain.calfits"]
    assert cal_out.filename == ["outtest.calh5"]

    assert cal_obj == cal_out

    # check error if missing required params
    with h5py.File(testfile, "r+") as h5f:
        del h5f["/Header/cal_type"]

    with pytest.raises(KeyError, match="cal_type not found in"):
        cal_out.read(testfile)

    # check error if missing required telescope params
    with h5py.File(testfile, "r+") as h5f:
        del h5f["/Header/telescope_name"]

    with pytest.raises(KeyError, match="telescope_name not found in"):
        cal_out.read(testfile)

    # error if trying to write a metadata only object
    cal_obj2 = cal_obj.copy(metadata_only=True)
    with pytest.raises(
        ValueError, match="Cannot write out metadata only objects to a calh5 file."
    ):
        cal_obj2.write_calh5(testfile, clobber=True)


@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.filterwarnings("ignore:Selected times are not evenly spaced. This is not")
@pytest.mark.parametrize(
    ["caltype", "time_range", "param_dict"],
    [
        ["gain", False, {"antenna_nums": np.array([65, 96, 9, 97, 89, 22, 20, 72])}],
        ["delay", False, {"antenna_names": np.array(["ant9"])}],
        ["delay", False, {"times": np.arange(2, 5)}],
        ["gain", True, {"times": 0}],
        ["gain", False, {"lsts": np.arange(2, 5)}],
        ["gain", True, {"lsts": np.arange(0, 2)}],
        ["delay", True, {"lsts": 0}],
        ["gain", False, {"time_range": [2457698, 2457720]}],
        ["gain", True, {"lst_range": [0, 3]}],
        ["gain", False, {"lst_range": [0, 3]}],
        ["gain", False, {"lst_range": [4, 2]}],
        ["delay", True, {"lst_range": [4, 2]}],
        ["gain", False, {"freq_chans": np.arange(4, 8)}],
        ["delay", False, {"spws": np.array([1, 3])}],
        ["gain", False, {"freq_chans": 1}],
        ["delay", False, {"spws": np.array([1, 2])}],
        ["gain", False, {"jones": ["xx", "yy"]}],
        ["delay", False, {"jones": -5}],
        [
            "gain",
            False,
            {
                "antenna_nums": np.array([65, 96, 9, 97, 89, 22, 20, 72]),
                "freq_chans": np.arange(2, 9),
                "times": np.arange(2, 5),
                "jones": ["xx", "yy"],
            },
        ],
        [
            "delay",
            False,
            {
                "antenna_nums": np.array([65, 96, 9, 97, 89, 22, 20, 72]),
                "spws": [1, 2],
                "times": 0,
                "jones": -5,
            },
        ],
        [
            "delay",
            False,
            {
                "antenna_nums": np.array([65, 96, 9, 97, 89, 22, 20, 72]),
                "spws": [1, 2],
                "lsts": 0,
                "jones": -5,
            },
        ],
        [
            "gain",
            False,
            {
                "freq_chans": np.arange(2, 9),
                "times": np.arange(2, 5),
                "jones": ["xx", "yy"],
            },
        ],
        ["delay", False, {"spws": [1, 2], "times": 0, "jones": [-5, -6]}],
    ],
)
def test_calh5_partial_read(
    gain_data, multi_spw_delay, tmp_path, caltype, time_range, param_dict
):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = multi_spw_delay

    orig_time_array = calobj.time_array
    orig_lst_array = calobj.lst_array
    total_quality = True

    if time_range:
        calobj = time_array_to_time_range(calobj)
        calobj.lst_range[1, 1] = 0.1

    for par, val in param_dict.items():
        if par == "times":
            param_dict[par] = orig_time_array[val]
        elif par == "lsts":
            param_dict[par] = orig_lst_array[val]

        if par.startswith("antenna"):
            total_quality = False

    extend_jones_axis(calobj, total_quality=total_quality)

    write_file = str(tmp_path / "outtest.calh5")
    calobj.write_calh5(write_file, clobber=True)

    calobj2 = calobj.copy()

    calobj2.select(**param_dict)

    calobj3 = UVCal.from_file(write_file, **param_dict)

    assert calobj2 == calobj3

    if time_range and "lsts" in param_dict:
        calobj2 = calobj.copy()
        param_dict["lsts"] = 2
        with pytest.raises(ValueError, match="LST 2 does not fall in any lst_range"):
            calobj2.select(**param_dict, strict=True)
