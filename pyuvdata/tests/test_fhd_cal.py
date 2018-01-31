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
