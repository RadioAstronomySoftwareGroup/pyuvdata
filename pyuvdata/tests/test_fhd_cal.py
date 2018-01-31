"""Tests for FHD_cal object."""
import nose.tools as nt
import os
from pyuvdata import UVCal
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import numpy as np

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
    if uvtest.pre_1_14_numpy:
        fhd_cal.read_fhd_cal(cal_testfile, obs_testfile, settings_file=settings_testfile)
    else:
        # numpy 1.14 introduced a new deprecation warning
        n_scipy_warnings, scipy_warn_list, scipy_category_list = \
            uvtest.get_scipy_warnings(n_scipy_warnings=605)
        uvtest.checkWarnings(fhd_cal.read_fhd_cal, [cal_testfile, obs_testfile],
                             {'settings_file': settings_testfile},
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)
    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    nt.assert_equal(fhd_cal, calfits_cal)

    # do it again with fit gains (rather than raw)
    if uvtest.pre_1_14_numpy:
        fhd_cal.read_fhd_cal(cal_testfile, obs_testfile,
                             settings_file=settings_testfile, raw=False)
    else:
        # numpy 1.14 introduced a new deprecation warning
        n_scipy_warnings, scipy_warn_list, scipy_category_list = \
            uvtest.get_scipy_warnings(n_scipy_warnings=605)
        uvtest.checkWarnings(fhd_cal.read_fhd_cal, [cal_testfile, obs_testfile],
                             {'settings_file': settings_testfile, 'raw': False},
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)
    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    nt.assert_equal(fhd_cal, calfits_cal)


def test_extra_history():
    """
    test that setting the extra_history keyword works
    """
    fhd_cal = UVCal()
    calfits_cal = UVCal()
    extra_history = 'Some extra history for testing\n'
    if uvtest.pre_1_14_numpy:
        fhd_cal.read_fhd_cal(cal_testfile, obs_testfile,
                             settings_file=settings_testfile,
                             extra_history=extra_history)
    else:
        # numpy 1.14 introduced a new deprecation warning
        n_scipy_warnings, scipy_warn_list, scipy_category_list = \
            uvtest.get_scipy_warnings(n_scipy_warnings=605)
        uvtest.checkWarnings(fhd_cal.read_fhd_cal, [cal_testfile, obs_testfile],
                             {'settings_file': settings_testfile,
                              'extra_history': extra_history},
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)
    outfile = os.path.join(DATA_PATH, 'test/outtest_FHDcal_1061311664.calfits')
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal.read_calfits(outfile)
    nt.assert_equal(fhd_cal, calfits_cal)
    nt.assert_true(extra_history in fhd_cal.history)


def test_breakReadFHDcal():
    """Try various cases of missing files."""
    fhd_cal = UVCal()
    nt.assert_raises(StandardError, fhd_cal.read_fhd_cal, cal_testfile)  # Missing obs

    if uvtest.pre_1_14_numpy:
        uvtest.checkWarnings(fhd_cal.read_fhd_cal, [cal_testfile, obs_testfile],
                             message=['No settings file'])
    else:
        # numpy 1.14 introduced a new deprecation warning
        n_scipy_warnings, scipy_warn_list, scipy_category_list = \
            uvtest.get_scipy_warnings(n_scipy_warnings=605)
        warn_list = scipy_warn_list + ['No settings file']
        category_list = scipy_category_list + [UserWarning]
        uvtest.checkWarnings(fhd_cal.read_fhd_cal, [cal_testfile, obs_testfile],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)
    # Check only pyuvdata version history with no settings file
    nt.assert_equal(fhd_cal.history, '\n' + fhd_cal.pyuvdata_version_str)
