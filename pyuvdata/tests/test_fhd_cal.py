# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD_cal object.

"""
from __future__ import absolute_import, division, print_function

import pytest
import os

from pyuvdata import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

# set up FHD file list
testdir = os.path.join(DATA_PATH, 'fhd_cal_data/')
testfile_prefix = '1061316296_'
obs_testfile = os.path.join(testdir, testfile_prefix + 'obs.sav')
cal_testfile = os.path.join(testdir, testfile_prefix + 'cal.sav')
settings_testfile = os.path.join(testdir, testfile_prefix + 'settings.txt')


def test_ReadFHDcalWriteReadcalfits():
    """
    FHD cal to calfits loopback test.

    Read in FHD cal files, write out as calfits, read back in and check for object
    equality.
    """
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(cal_testfile, obs_testfile, settings_file=settings_testfile)

    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal

    # do it again with fit gains (rather than raw)
    fhd_cal.read_fhd_cal(cal_testfile, obs_testfile,
                         settings_file=settings_testfile, raw=False)
    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal


def test_extra_history():
    """
    test that setting the extra_history keyword works
    """
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    extra_history = 'Some extra history for testing\n'
    fhd_cal.read_fhd_cal(cal_testfile, obs_testfile,
                         settings_file=settings_testfile,
                         extra_history=extra_history)

    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal
    assert extra_history in fhd_cal.history

    # try again with a list of history strings
    extra_history = ['Some extra history for testing',
                     'And some more history as well']
    fhd_cal.read_fhd_cal(cal_testfile, obs_testfile,
                         settings_file=settings_testfile,
                         extra_history=extra_history)

    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal
    for line in extra_history:
        assert line in fhd_cal.history


def test_flags_galaxy():
    """
    test that files with time, freq and tile flags and galaxy models behave as expected
    """
    testdir = os.path.join(DATA_PATH, 'fhd_cal_data/flag_set')
    obs_testfile_flag = os.path.join(testdir, testfile_prefix + 'obs.sav')
    cal_testfile_flag = os.path.join(testdir, testfile_prefix + 'cal.sav')
    settings_testfile_flag = os.path.join(testdir, testfile_prefix + 'settings.txt')

    fhd_cal = UVCal()
    calfits_cal = UVCal()
    fhd_cal.read_fhd_cal(cal_testfile_flag, obs_testfile_flag,
                         settings_file=settings_testfile_flag)

    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal


def test_breakReadFHDcal():
    """Try various cases of missing files."""
    fhd_cal = UVCal()
    pytest.raises(TypeError, fhd_cal.read_fhd_cal, cal_testfile)  # Missing obs

    uvtest.checkWarnings(fhd_cal.read_fhd_cal, [cal_testfile, obs_testfile],
                         message=['No settings file'])

    # Check only pyuvdata version history with no settings file
    assert fhd_cal.history == '\n' + fhd_cal.pyuvdata_version_str


def test_read_multi():
    """Test reading in multiple files."""
    testdir2 = os.path.join(DATA_PATH, 'fhd_cal_data/set2')
    obs_testfile_list = [obs_testfile, os.path.join(testdir2, testfile_prefix + 'obs.sav')]
    cal_testfile_list = [cal_testfile, os.path.join(testdir2, testfile_prefix + 'cal.sav')]
    settings_testfile_list = [settings_testfile, os.path.join(testdir2, testfile_prefix + 'settings.txt')]

    fhd_cal = UVCal()
    calfits_cal = UVCal()
    uvtest.checkWarnings(fhd_cal.read_fhd_cal, [cal_testfile_list, obs_testfile_list],
                         {'settings_file': settings_testfile_list},
                         message='UVParameter diffuse_model does not match')

    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    assert fhd_cal == calfits_cal


def test_break_read_multi():
    """Test errors for different numbers of files."""

    testdir2 = os.path.join(DATA_PATH, 'fhd_cal_data/set2')
    obs_testfile_list = [obs_testfile, os.path.join(testdir2, testfile_prefix + 'obs.sav')]
    cal_testfile_list = [cal_testfile, os.path.join(testdir2, testfile_prefix + 'cal.sav')]
    settings_testfile_list = [settings_testfile, os.path.join(testdir2, testfile_prefix + 'settings.txt')]

    fhd_cal = UVCal()
    pytest.raises(ValueError, fhd_cal.read_fhd_cal, cal_testfile_list,
                  obs_testfile_list[0], settings_file=settings_testfile_list)
    pytest.raises(ValueError, fhd_cal.read_fhd_cal, cal_testfile_list,
                  obs_testfile_list, settings_file=settings_testfile_list[0])
    pytest.raises(ValueError, fhd_cal.read_fhd_cal, cal_testfile_list,
                  obs_testfile_list + obs_testfile_list, settings_file=settings_testfile_list)
    pytest.raises(ValueError, fhd_cal.read_fhd_cal, cal_testfile_list,
                  obs_testfile_list, settings_file=settings_testfile_list + settings_testfile_list)
    pytest.raises(ValueError, fhd_cal.read_fhd_cal, cal_testfile_list[0],
                  obs_testfile_list, settings_file=settings_testfile_list[0])
    pytest.raises(ValueError, fhd_cal.read_fhd_cal, cal_testfile_list[0],
                  obs_testfile_list[0], settings_file=settings_testfile_list)
