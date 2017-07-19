"""Tests for FHD object."""
import nose.tools as nt
import os
from pyuvdata import UVData
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import numpy as np

# set up FHD file list
testdir = os.path.join(DATA_PATH, 'fhd_vis_data/')
testfile_prefix = '1061316296_'
testfile_suffix = ['flags.sav', 'vis_XX.sav', 'params.sav', 'vis_YY.sav',
                   'vis_model_XX.sav', 'vis_model_YY.sav', 'settings.txt']
testfiles = []
for s in testfile_suffix:
    testfiles.append(testdir + testfile_prefix + s)


def test_ReadFHDWriteReadUVFits():
    """
    FHD to uvfits loopback test.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    fhd_uv.read_fhd(testfiles)
    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    nt.assert_equal(fhd_uv, uvfits_uv)
    del(fhd_uv)
    del(uvfits_uv)


def test_breakReadFHD():
    """Try various cases of incomplete file lists."""
    fhd_uv = UVData()
    nt.assert_raises(StandardError, fhd_uv.read_fhd, testfiles[1:])  # Missing flags
    del(fhd_uv)
    fhd_uv = UVData()
    subfiles = [item for sublist in [testfiles[0:2], testfiles[3:]] for item in sublist]
    nt.assert_raises(StandardError, fhd_uv.read_fhd, subfiles)  # Missing params
    del(fhd_uv)
    fhd_uv = UVData()
    nt.assert_raises(StandardError, fhd_uv.read_fhd, ['foo'])  # No data files
    del(fhd_uv)
    fhd_uv = UVData()
    uvtest.checkWarnings(fhd_uv.read_fhd, [testfiles[:-1]], message=['No settings'])
    # Check only pyuvdata history with no settings file
    nt.assert_equal(fhd_uv.history, fhd_uv.pyuvdata_version_str)  # Check empty history with no settings
    del(fhd_uv)


def test_ReadFHD_model():
    """FHD to uvfits loopback test with model visibilities."""
    fhd_uv = UVData()
    uvfits_uv = UVData()
    fhd_uv.read_fhd(testfiles, use_model=True)
    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296_model.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296_model.uvfits'))
    nt.assert_equal(fhd_uv, uvfits_uv)
    del(fhd_uv)
    del(uvfits_uv)


def test_multi_files():
    """
    Reading multiple files at once.
    """
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    test1 = list(np.array(testfiles)[[0, 1, 2, 4, 6]])
    test2 = list(np.array(testfiles)[[0, 2, 3, 5, 6]])
    fhd_uv1.read_fhd([test1, test2])
    fhd_uv2.read_fhd(testfiles)

    nt.assert_true(uvutils.check_histories(fhd_uv2.history + ' Combined data '
                                           'along polarization axis using pyuvdata.',
                                           fhd_uv1.history))

    fhd_uv1.history = fhd_uv2.history
    nt.assert_equal(fhd_uv1, fhd_uv2)
