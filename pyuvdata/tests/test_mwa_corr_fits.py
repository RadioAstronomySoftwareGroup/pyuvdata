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
             '1131733552_mod.metafits']
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
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[filelist[0:2]],
                         nwarnings=2, message=messages)
    testfile = os.path.join(DATA_PATH, 'test/outtest_MWAcorr.uvfits')
    mwa_uv.write_uvfits(testfile, spoof_nonessential=True, force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    assert mwa_uv == uvfits_uv


def test_ReadMWA_multi():
    """
    Test reading in two sets of files.
    """
    set1 = filelist[0:2]
    set2 = [filelist[0], filelist[2]]
    mwa_uv = UVData()
    messages = ['telescope_location is not set',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[[[set1], [set2]]],
                         nwarnings=2, message=messages)


def test_ReadMWAWriteUVFits_flags():
    """
    Test handling of flag files in loopback test.

    Read in MWA correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    mwa_uv = UVData()
    subfiles = [filelist[0], filelist[1], filelist[3]]
    messages = ['mwaf files submitted with use_cotter_flags=False',
                'telescope_location is not set',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[subfiles],
                         nwarnings=3, message=messages)


def test_multiple_coarse():
    """
    Read in MWA correlator files with two different orderings of the files
    and check for object equality.
    """
    mwa_uv = UVData()
    messages = ['telescope_location is not set',
                'coarse channels are not contiguous for this observation',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[filelist[0:3]],
                         nwarnings=3, message=messages)


@pytest.mark.parametrize("files,err_msg",
                         [([filelist[0]], "no data files submitted"),
                          ([filelist[1]], "no metafits file submitted"),
                          ([filelist[0], filelist[1], filelist[4]],
                           "multiple metafits files in filelist")])
def test_break_ReadMWAcorrFITS(files, err_msg):
    """
    Break read_mwa_corr_fits by submitting files incorrectly.
    """
    mwa_uv = UVData()
    with pytest.raises(ValueError) as cm:
        mwa_uv.read_mwa_corr_fits(files)
    assert str(cm.value).startswith(err_msg)
    del(mwa_uv)
