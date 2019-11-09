# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MWACorrFITS object.

"""

import pytest
import os
import numpy as np

from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
from astropy.io import fits

# set up MWA correlator file list
testdir = os.path.join(DATA_PATH, 'mwa_corr_fits_testfiles/')

testfiles = ['1131733552.metafits', '1131733552_20151116182537_mini_gpubox01_00.fits',
             '1131733552_20151116182637_mini_gpubox06_01.fits', '1131733552_mini_01.mwaf',
             '1131733552_mini_06.mwaf']
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
    mwa_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvfits_uv.read_uvfits(testfile)
    assert mwa_uv == uvfits_uv


def test_ReadMWAWriteUVFits_meta_mod():
    """
    MWA correlator fits to uvfits loopback test with a modified metafits file.

    Read in MWA correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    mwa_uv = UVData()
    uvfits_uv = UVData()
    mod_metafile = os.path.join(DATA_PATH, 'test/1131733552_mod.metafits')
    with fits.open(filelist[0]) as meta:
        meta[0].header['channels'] = '127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150'
        v_factor = 1.204
        length = float(meta[1].data['length'][10][3:])
        length /= v_factor
        meta[1].data['length'][11] = str(length)
        meta.writeto(mod_metafile)
    messages = ['telescope_location is not set',
                'some coarse channel files were not submitted']
    files = [filelist[1], mod_metafile]
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[files],
                         nwarnings=2, message=messages)
    testfile = os.path.join(DATA_PATH, 'test/outtest_MWAcorr.uvfits')
    mwa_uv.write_uvfits(testfile, spoof_nonessential=True)
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
                'some coarse channel files were not submitted',
                'telescope_location is not set',
                'some coarse channel files were not submitted',
                'Combined frequencies are not contiguous. This will make it impossible to write this data out to some file types.']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[[[set1], [set2]]],
                         nwarnings=5, message=messages)


def test_ReadMWA_multi_concat():
    """
    Test reading in two sets of files with fast concatenation.
    """
    # modify file so that time arrays are matching
    mod_mini_6 = os.path.join(DATA_PATH, 'test/mini_gpubox06_01.fits')
    with fits.open(filelist[2]) as mini6:
        mini6[1].header['time'] = 1447698337
        mini6.writeto(mod_mini_6)
    set1 = filelist[0:2]
    set2 = [filelist[0], mod_mini_6]
    mwa_uv = UVData()
    messages = ['telescope_location is not set',
                'some coarse channel files were not submitted',
                'telescope_location is not set',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[[[set1], [set2]]],
                         func_kwargs={"axis": "freq"}, nwarnings=4, message=messages)


def test_ReadMWA_flags():
    """
    Test handling of flag files
    """
    mwa_uv = UVData()
    subfiles = [filelist[0], filelist[1], filelist[3], filelist[4]]
    messages = ['mwaf files submitted with use_cotter_flags=False',
                'telescope_location is not set',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[subfiles],
                         nwarnings=3, message=messages)
    del(mwa_uv)
    mwa_uv = UVData()
    messages = ['telescope_location is not set',
                'some coarse channel files were not submitted',
                'reading in cotter flag files is not yet available']
    uvtest.checkWarnings(mwa_uv.read_mwa_corr_fits, func_args=[subfiles],
                         func_kwargs={'use_cotter_flags': True},
                         nwarnings=3, message=messages)
    del(mwa_uv)
    mwa_uv = UVData()
    with pytest.raises(ValueError) as cm:
        mwa_uv.read_mwa_corr_fits(subfiles[0:2], use_cotter_flags=True)
    assert str(cm.value).startswith('no flag files submitted')
    del(mwa_uv)


def test_multiple_coarse():
    """
    Read in MWA correlator files with two different orderings of the files
    and check for object equality.
    """
    order1 = [filelist[0:3]]
    order2 = [filelist[0], filelist[2], filelist[1]]
    mwa_uv1 = UVData()
    mwa_uv2 = UVData()
    messages = ['telescope_location is not set',
                'coarse channels are not contiguous for this observation',
                'some coarse channel files were not submitted']
    uvtest.checkWarnings(mwa_uv1.read_mwa_corr_fits, func_args=[order1],
                         nwarnings=3, message=messages)
    uvtest.checkWarnings(mwa_uv2.read_mwa_corr_fits, func_args=[order2],
                         nwarnings=3, message=messages)
    assert mwa_uv1 == mwa_uv2


def test_fine_channels():
    """
    Test that error is raised if files with different numbers of fine channels
    are submitted.
    """
    mwa_uv = UVData()
    bad_fine = os.path.join(DATA_PATH, 'test/bad_gpubox06_01.fits')
    with fits.open(filelist[2]) as mini6:
        mini6[1].data = np.concatenate((mini6[1].data, mini6[1].data))
        mini6.writeto(bad_fine)
    with pytest.raises(ValueError) as cm:
        mwa_uv.read_mwa_corr_fits([bad_fine, filelist[1]])
    assert str(cm.value).startswith('files submitted have different fine')
    del(mwa_uv)


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


def test_file_extension():
    """
    Test that error is raised if a file with an extension that is not fits,
    metafits, or mwaf is submitted.
    """
    mwa_uv = UVData()
    bad_ext = os.path.join(DATA_PATH, 'test/1131733552.meta')
    with fits.open(filelist[0]) as meta:
        meta.writeto(bad_ext)
    with pytest.raises(ValueError) as cm:
        mwa_uv.read_mwa_corr_fits(bad_ext)
    assert str(cm.value).startswith('only fits, metafits, and mwaf files supported')
    del(mwa_uv)
