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

# set up FHD files
testdir = os.path.join(DATA_PATH, "fhd_cal_data/")
testfile_prefix = "1061316296_"
obs_testfile = os.path.join(testdir, testfile_prefix + "obs.sav")
cal_testfile = os.path.join(testdir, testfile_prefix + "cal.sav")
settings_testfile = os.path.join(testdir, testfile_prefix + "settings.txt")
layout_testfile = os.path.join(testdir, testfile_prefix + "layout.sav")

testdir2 = os.path.join(DATA_PATH, "fhd_cal_data/set2")
obs_file_multi = [
    obs_testfile,
    os.path.join(testdir2, testfile_prefix + "obs.sav"),
]
cal_file_multi = [
    cal_testfile,
    os.path.join(testdir2, testfile_prefix + "cal.sav"),
]
layout_file_multi = [layout_testfile, layout_testfile]
settings_file_multi = [
    settings_testfile,
    os.path.join(testdir2, testfile_prefix + "settings.txt"),
]


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
@pytest.mark.parametrize("raw", [True, False])
def test_read_fhdcal_raw_write_read_calfits(raw, tmp_path):
    """
    FHD cal to calfits loopback test.

    Read in FHD cal files, write out as calfits, read back in and check for
    object equality.
    """
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(
        cal_testfile,
        obs_testfile,
        layout_file=layout_testfile,
        settings_file=settings_testfile,
        raw=raw,
    )

    assert np.max(fhd_cal.gain_array) < 2.0

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal

    # check metadata only read
    fhd_cal.read_fhd_cal(
        cal_testfile,
        obs_testfile,
        layout_file=layout_testfile,
        settings_file=settings_testfile,
        raw=raw,
        read_data=False,
    )
    calfits_cal2 = calfits_cal.copy(metadata_only=True)

    # this file set has a mismatch in Nsources between the cal file & settings
    # file for some reason. I think it's just an issue with the files chosen
    assert fhd_cal.Nsources != calfits_cal2.Nsources
    fhd_cal.Nsources = calfits_cal2.Nsources

    # there is a loss in precision for float auto scale values in the
    # settings file vs the cal file
    assert (
        fhd_cal.extra_keywords["autoscal".upper()]
        != calfits_cal2.extra_keywords["autoscal".upper()]
    )
    fhd_cal.extra_keywords["autoscal".upper()] = calfits_cal2.extra_keywords[
        "autoscal".upper()
    ]

    assert fhd_cal == calfits_cal2

    return


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
def test_read_fhdcal_multimode():
    """
    Read cal with multiple mode_fit values.
    """
    fhd_cal = UVCal()
    fhd_cal.read_fhd_cal(
        os.path.join(testdir, testfile_prefix + "multimode_cal.sav"),
        obs_testfile,
        layout_file=layout_testfile,
        settings_file=os.path.join(testdir, testfile_prefix + "multimode_settings.txt"),
        raw=False,
    )
    assert fhd_cal.extra_keywords["MODE_FIT"] == "[90, 150, 230, 320, 400, 524]"

    fhd_cal2 = fhd_cal.copy(metadata_only=True)

    # check metadata only read
    fhd_cal.read_fhd_cal(
        os.path.join(testdir, testfile_prefix + "multimode_cal.sav"),
        obs_testfile,
        layout_file=layout_testfile,
        settings_file=os.path.join(testdir, testfile_prefix + "multimode_settings.txt"),
        raw=False,
        read_data=False,
    )

    # this file set has a mismatch in Nsources between the cal file & settings
    # file for some reason. I think it's just an issue with the files chosen
    assert fhd_cal.Nsources != fhd_cal2.Nsources
    fhd_cal.Nsources = fhd_cal2.Nsources

    # there is a loss in precision for float auto scale values in the
    # settings file vs the cal file
    assert (
        fhd_cal.extra_keywords["autoscal".upper()]
        != fhd_cal2.extra_keywords["autoscal".upper()]
    )
    fhd_cal.extra_keywords["autoscal".upper()] = fhd_cal2.extra_keywords[
        "autoscal".upper()
    ]
    assert fhd_cal == fhd_cal2

    return


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
@pytest.mark.parametrize(
    "extra_history",
    [
        "Some extra history for testing\n",
        ["Some extra history for testing", "And some more history as well"],
    ],
)
def test_extra_history(extra_history, tmp_path):
    """Test that setting the extra_history keyword works."""
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(
        cal_testfile,
        obs_testfile,
        layout_file=layout_testfile,
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


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
def test_flags_galaxy(tmp_path):
    """Test files with time, freq and tile flags and galaxy models behave."""
    testdir = os.path.join(DATA_PATH, "fhd_cal_data/flag_set")
    obs_testfile_flag = os.path.join(testdir, testfile_prefix + "obs.sav")
    cal_testfile_flag = os.path.join(testdir, testfile_prefix + "cal.sav")
    settings_testfile_flag = os.path.join(testdir, testfile_prefix + "settings.txt")

    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(
        cal_testfile_flag,
        obs_testfile_flag,
        layout_file=layout_testfile,
        settings_file=settings_testfile_flag,
    )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal


def test_unknown_telescope():
    fhd_cal = UVCal()

    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Telescope location derived from obs lat/lon/alt values does not match "
            "the location in the layout file. ",
            "Telescope foo is not in known_telescopes.",
        ],
    ):
        fhd_cal.read_fhd_cal(
            cal_testfile,
            os.path.join(testdir, testfile_prefix + "telescopefoo_obs.sav"),
            layout_file=layout_testfile,
            settings_file=settings_testfile,
        )
    assert fhd_cal.telescope_name == "foo"


@pytest.mark.parametrize(
    "cal_file,obs_file,layout_file,settings_file,nfiles",
    [
        [cal_testfile, obs_testfile, layout_testfile, settings_testfile, 1],
        [cal_file_multi, obs_file_multi, layout_file_multi, settings_file_multi, 2],
    ],
)
def test_break_read_fhdcal(cal_file, obs_file, layout_file, settings_file, nfiles):
    """Try various cases of missing files."""
    fhd_cal = UVCal()
    pytest.raises(TypeError, fhd_cal.read_fhd_cal, cal_file)  # Missing obs

    message_list = [
        "No settings file",
        "Telescope location derived from obs lat/lon/alt values does not match the "
        "location in the layout file.",
    ]
    if nfiles > 1:
        message_list *= 2
        message_list.append("UVParameter diffuse_model does not match")

    with uvtest.check_warnings(UserWarning, message_list):
        fhd_cal.read_fhd_cal(cal_file, obs_file, layout_file=layout_file)

    # Check only pyuvdata version history with no settings file
    expected_history = "\n" + fhd_cal.pyuvdata_version_str
    if nfiles > 1:
        expected_history += " Combined data along time axis using pyuvdata."
    assert fhd_cal.history == expected_history

    message_list = (
        ["No layout file, antenna_postions will not be defined."] * nfiles
        + ["UVParameter diffuse_model does not match"] * (nfiles - 1)
        + [
            "The antenna_positions parameter is not set. It will be a required "
            "parameter starting in pyuvdata version 2.3"
        ]
        * (4 * nfiles - 3)
    )

    warning_list = [UserWarning] * (2 * nfiles - 1) + [DeprecationWarning] * (
        4 * nfiles - 3
    )
    with uvtest.check_warnings(warning_list, message_list):
        fhd_cal.read_fhd_cal(cal_file, obs_file, settings_file=settings_file)

    # Check no antenna_positions
    assert fhd_cal.antenna_positions is None


def test_read_multi(tmp_path):
    """Test reading in multiple files."""
    fhd_cal = UVCal()
    calfits_cal = UVCal()

    with uvtest.check_warnings(
        UserWarning,
        [
            "UVParameter diffuse_model does not match",
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file.",
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file.",
        ],
    ):
        fhd_cal.read_fhd_cal(
            cal_file_multi,
            obs_file_multi,
            settings_file=settings_file_multi,
            layout_file=layout_file_multi,
        )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal


@pytest.mark.parametrize(
    "cal_file,obs_file,layout_file,settings_file,message",
    [
        [
            cal_file_multi[0],
            obs_file_multi,
            layout_file_multi,
            settings_file_multi,
            "Number of obs_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi[0],
            layout_file_multi,
            settings_file_multi,
            "Number of obs_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi[0],
            settings_file_multi,
            "Number of layout_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi,
            settings_file_multi[0],
            "Number of settings_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi + obs_file_multi,
            layout_file_multi,
            settings_file_multi,
            "Number of obs_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi + layout_file_multi,
            settings_file_multi,
            "Number of layout_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi,
            settings_file_multi + settings_file_multi,
            "Number of settings_files must match number of cal_files",
        ],
        [
            cal_file_multi[0],
            obs_file_multi[0],
            layout_file_multi,
            settings_file_multi,
            "Number of layout_files must match number of cal_files",
        ],
        [
            cal_file_multi[0],
            obs_file_multi[0],
            layout_file_multi[0],
            settings_file_multi,
            "Number of settings_files must match number of cal_files",
        ],
    ],
)
def test_break_read_multi(cal_file, obs_file, layout_file, settings_file, message):
    """Test errors for different numbers of files."""
    fhd_cal = UVCal()

    with pytest.raises(ValueError, match=message):
        fhd_cal.read_fhd_cal(
            cal_file, obs_file, layout_file=layout_file, settings_file=settings_file
        )
