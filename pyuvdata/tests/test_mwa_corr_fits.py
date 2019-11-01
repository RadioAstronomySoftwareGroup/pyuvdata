# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MWACorrFITS object.

"""

import pytest
import os
import shutil
import copy
import numpy as np

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
    messages = ['no flag files submitted', 'telescope_location is not set',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, [filelist[0:2]], nwarnings=3,
                         message=messages)
    testfile = os.path.join(DATA_PATH, 'test/outtest_MWAcorr.uvfits')
    mwa_uv.write_uvfits(testfile, spoof_nonessential=True, force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    assert mwa_uv == uvfits_uv


def test_noncontiguous_coarse():
    """
    Read in MWA correlator files and check that non-contiguous coarse channel
    warning is given.
    """
    mwa_uv = UVData()
    messages = ['no flag files submitted', 'telescope_location is not set',
                'coarse channels are not contiguous for this observation',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, [filelist[0:3]], nwarnings=4,
                         message=messages)

