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

filelist = [testdir + '1131733552.metafits', testdir + '1131733552_20151116182537_mini_gpubox01_00.fits']


def test_ReadMWAWriteUVFits():
    """
    MWA correlator fits to uvfits loopback test.

    Read in MWA correlator files, write out as uvfits, read back in and check for object
    equality.
    """
    mwa_uv = UVData()
    uvfits_uv = UVData()
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, [filelist], nwarnings=3,
                         message=['no flag files submitted', 'telescope_location is not set',
                         'some coarse channel files were not submitted'])
    testfile = os.path.join(DATA_PATH, 'test/outtest_MWAcorr.uvfits')
    mwa_uv.reorder_pols()
    mwa_uv.write_uvfits(testfile, spoof_nonessential=True, force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    assert mwa_uv == uvfits_uv
