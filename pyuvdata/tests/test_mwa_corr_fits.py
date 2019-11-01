# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MWACorrFITS object.

"""

import pytest
import os

from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest

# set up MWA correlator file list
testdir = os.path.join(DATA_PATH, 'mwa_corr_fits_testfiles/')

testfiles = ['1131733552.metafits', '1131733552_20151116182537_mini_gpubox01_00.fits',
             '1131733552_20151116182637_mini_gpubox06_01.fits', '1131733552_mini_01.mwaf',
             '1131733552_2.metafits']
filelist = [testdir + i for i in testfiles]


def test_ReadMWAWriteUVFits():
    """
    MWA correlator fits to uvfits loopback test.

    Read in MWA correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    mwa_uv = UVData()
    uvfits_uv = UVData()
    messages = ['telescope_location is not set',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, [filelist[0:2]], nwarnings=2,
                         message=messages)
    testfile = os.path.join(DATA_PATH, 'test/outtest_MWAcorr.uvfits')
    mwa_uv.write_uvfits(testfile, spoof_nonessential=True, force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    assert mwa_uv == uvfits_uv


def test_ReadMWAWriteUVFits_flags():
    """
    Test handling of flag files
    """
    mwa_uv = UVData()
    subfiles = [filelist[0], filelist[1], filelist[3]]
    messages = ['mwaf files submitted with use_cotter_flags=False',
                'telescope_location is not set',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, [subfiles], nwarnings=3,
                         message=messages)


def test_noncontiguous_coarse():
    """
    Read in MWA correlator files and check that non-contiguous coarse channel
    warning is given.
    """
    mwa_uv = UVData()
    messages = ['telescope_location is not set',
                'coarse channels are not contiguous for this observation',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, [filelist[0:3]], nwarnings=3,
                         message=messages)


def break_ReadMWAcorrFITS():
    """
    Break read_mwa_corr_fits by submitting files incorrectly.
    """
    # no data files
    mwa_uv = UVData()
    pytest.raises(ValueError, mwa_uv.read_mwa_corr_fits, filelist[0])
    del(mwa_uv)
    # no metafits file
    mwa_uv = UVData()
    pytest.raises(ValueError, mwa_uv.read_mwa_corr_fits, filelist[1])
    del(mwa_uv)
    # more than one metafits file
    mwa_uv = UVData()
    subfiles = [filelist[0], filelist[1], filelist[4]]
    pytest.raises(ValueError, mwa_uv.read_mwa_corr_fits, subfiles)
    del(mwa_uv)
    # no flag file with use_cotter_flags=True
    mwa_uv = UVData()
    pytest.raises(ValueError, mwa_uv.read_mwa_corr_fits, filelist[0:2],
                  use_cotter_flags=True)
    del(mwa_uv)
