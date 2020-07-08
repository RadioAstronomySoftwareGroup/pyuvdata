# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD_cal object."""
import pytest
import os

import numpy as np

from pyuvdata import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

# set up FHD file list
testdir = os.path.join(DATA_PATH, "fhd_cal_data/")
testfile_prefix = "1061316296_"
obs_testfile = os.path.join(testdir, testfile_prefix + "obs.sav")
cal_testfile = os.path.join(testdir, testfile_prefix + "cal.sav")
settings_testfile = os.path.join(testdir, testfile_prefix + "settings.txt")


def test_read_fhdcal_raw_write_read_calfits(tmp_path):
    """
    FHD cal to calfits loopback test.

    Read in FHD cal files, write out as calfits, read back in and check for
    object equality.
    """
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(cal_testfile, obs_testfile, settings_file=settings_testfile)

    assert np.max(fhd_cal.gain_array) < 2.0

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal

    return


def test_read_fhdcal_fit_write_read_calfits(tmp_path):
    # do it again with fit gains (rather than raw)
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(
        cal_testfile, obs_testfile, settings_file=settings_testfile, raw=False
    )
    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal

    return


def test_extra_history(tmp_path):
    """Test that setting the extra_history keyword works."""
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    extra_history = "Some extra history for testing\n"
    fhd_cal.read_fhd_cal(
        cal_testfile,
        obs_testfile,
        settings_file=settings_testfile,
        extra_history=extra_history,
    )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal
    assert extra_history in fhd_cal.history

    return


def test_extra_history_strings(tmp_path):
    # try again with a list of history strings
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    extra_history = ["Some extra history for testing", "And some more history as well"]
    fhd_cal.read_fhd_cal(
        cal_testfile,
        obs_testfile,
        settings_file=settings_testfile,
        extra_history=extra_history,
    )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal
    for line in extra_history:
        assert line in fhd_cal.history

    return


def test_flags_galaxy(tmp_path):
    """Test files with time, freq and tile flags and galaxy models behave."""
    testdir = os.path.join(DATA_PATH, "fhd_cal_data/flag_set")
    obs_testfile_flag = os.path.join(testdir, testfile_prefix + "obs.sav")
    cal_testfile_flag = os.path.join(testdir, testfile_prefix + "cal.sav")
    settings_testfile_flag = os.path.join(testdir, testfile_prefix + "settings.txt")

    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(
        cal_testfile_flag, obs_testfile_flag, settings_file=settings_testfile_flag
    )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal


def test_break_read_fhdcal():
    """Try various cases of missing files."""
    fhd_cal = UVCal()
    pytest.raises(TypeError, fhd_cal.read_fhd_cal, cal_testfile)  # Missing obs

    with uvtest.check_warnings(UserWarning, "No settings file"):
        fhd_cal.read_fhd_cal(cal_testfile, obs_testfile)

    # Check only pyuvdata version history with no settings file
    assert fhd_cal.history == "\n" + fhd_cal.pyuvdata_version_str


def test_read_multi(tmp_path):
    """Test reading in multiple files."""
    testdir2 = os.path.join(DATA_PATH, "fhd_cal_data/set2")
    obs_testfile_list = [
        obs_testfile,
        os.path.join(testdir2, testfile_prefix + "obs.sav"),
    ]
    cal_testfile_list = [
        cal_testfile,
        os.path.join(testdir2, testfile_prefix + "cal.sav"),
    ]
    settings_testfile_list = [
        settings_testfile,
        os.path.join(testdir2, testfile_prefix + "settings.txt"),
    ]

    fhd_cal = UVCal()
    calfits_cal = UVCal()

    with uvtest.check_warnings(UserWarning, "UVParameter diffuse_model does not match"):
        fhd_cal.read_fhd_cal(
            cal_testfile_list, obs_testfile_list, settings_file=settings_testfile_list
        )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal


def test_break_read_multi():
    """Test errors for different numbers of files."""
    testdir2 = os.path.join(DATA_PATH, "fhd_cal_data/set2")
    obs_testfile_list = [
        obs_testfile,
        os.path.join(testdir2, testfile_prefix + "obs.sav"),
    ]
    cal_testfile_list = [
        cal_testfile,
        os.path.join(testdir2, testfile_prefix + "cal.sav"),
    ]
    settings_testfile_list = [
        settings_testfile,
        os.path.join(testdir2, testfile_prefix + "settings.txt"),
    ]

    fhd_cal = UVCal()
    pytest.raises(
        ValueError,
        fhd_cal.read_fhd_cal,
        cal_testfile_list,
        obs_testfile_list[0],
        settings_file=settings_testfile_list,
    )
    pytest.raises(
        ValueError,
        fhd_cal.read_fhd_cal,
        cal_testfile_list,
        obs_testfile_list,
        settings_file=settings_testfile_list[0],
    )
    pytest.raises(
        ValueError,
        fhd_cal.read_fhd_cal,
        cal_testfile_list,
        obs_testfile_list + obs_testfile_list,
        settings_file=settings_testfile_list,
    )
    pytest.raises(
        ValueError,
        fhd_cal.read_fhd_cal,
        cal_testfile_list,
        obs_testfile_list,
        settings_file=settings_testfile_list + settings_testfile_list,
    )
    pytest.raises(
        ValueError,
        fhd_cal.read_fhd_cal,
        cal_testfile_list[0],
        obs_testfile_list,
        settings_file=settings_testfile_list[0],
    )
    pytest.raises(
        ValueError,
        fhd_cal.read_fhd_cal,
        cal_testfile_list[0],
        obs_testfile_list[0],
        settings_file=settings_testfile_list,
    )
