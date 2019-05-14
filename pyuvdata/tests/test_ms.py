# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MS object.

"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import shutil
import copy
import numpy as np

from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
from pyuvdata.uvfits import UVFITS


@uvtest.skipIf_no_casa
def test_cotter_ms():
    """Test reading in an ms made from MWA data with cotter (no dysco compression)"""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, '1102865728_small.ms')
    UV.read(testfile)

    # check that a select on read works
    UV2 = UVData()
    uvtest.checkWarnings(UV2.read, [testfile], {'freq_chans': np.arange(2)},
                         message='Warning: select on read keyword set')
    UV.select(freq_chans=np.arange(2))
    assert UV == UV2
    del(UV)


@uvtest.skipIf_no_casa
def test_readNRAO():
    """Test reading in a CASA tutorial ms file."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    expected_extra_keywords = ['DATA_COL']

    UV.read(testfile)
    assert sorted(expected_extra_keywords) == sorted(list(UV.extra_keywords.keys()))


@uvtest.skipIf_no_casa
def test_read_LWA():
    """Test reading in an LWA ms file."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'lwasv_cor_58342_05_00_14.ms.tar.gz')
    expected_extra_keywords = ['DATA_COL']

    import tarfile
    with tarfile.open(testfile) as tf:
        new_filename = os.path.join(DATA_PATH, tf.getnames()[0])
        tf.extractall(path=DATA_PATH)

    UV.read(new_filename, file_type='ms')
    assert sorted(expected_extra_keywords) == sorted(list(UV.extra_keywords.keys()))

    assert UV.history == UV.pyuvdata_version_str

    # delete the untarred folder
    shutil.rmtree(new_filename)


@uvtest.skipIf_no_casa
def test_noSPW():
    """Test reading in a PAPER ms converted by CASA from a uvfits with no spw axis."""
    UV = UVData()
    testfile_no_spw = os.path.join(
        DATA_PATH, 'zen.2456865.60537.xy.uvcRREAAM.ms')
    UV.read(testfile_no_spw)
    del(UV)


@uvtest.skipIf_no_casa
def test_spwnotsupported():
    """Test errors on reading in an ms file with multiple spws."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1scan.ms')
    pytest.raises(ValueError, UV.read, testfile)
    del(UV)


@uvtest.skipIf_no_casa
def test_multi_len_spw():
    """Test errors on reading in an ms file with multiple spws with variable lenghth."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'multi_len_spw.ms')
    with pytest.raises(ValueError) as cm:
        UV.read(testfile)
    assert str(cm.value).startswith('Sorry.  Files with more than one spectral')


@uvtest.skipIf_no_casa
def test_extra_pol_setup():
    """Test reading in an ms file with extra polarization setups (not used in data)."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'X5707_1spw_1scan_10chan_1time_1bl_noatm.ms.tar.gz')

    import tarfile
    with tarfile.open(testfile) as tf:
        new_filename = os.path.join(DATA_PATH, tf.getnames()[0])
        tf.extractall(path=DATA_PATH)

    UV.read(new_filename, file_type='ms')

    # delete the untarred folder
    shutil.rmtree(new_filename)


@uvtest.skipIf_no_casa
def test_readMSreadUVFITS():
    """
    Test that a uvdata object instantiated from an ms file created with CASA's
    importuvfits is equal to a uvdata object instantiated from the original
    uvfits file (tests equivalence with importuvfits in uvdata).
    Since the histories are different, this test sets both uvdata
    histories to identical empty strings before comparing them.
    """
    ms_uv = UVData()
    uvfits_uv = UVData()
    ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    uvfits_file = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [uvfits_file],
                         message='Telescope EVLA is not')
    ms_uv.read(ms_file)
    # set histories to identical blank strings since we do not expect
    # them to be the same anyways.
    ms_uv.history = ""
    uvfits_uv.history = ""

    # the objects won't be equal because uvfits adds some optional parameters
    # and the ms sets default antenna diameters even thoug the uvfits file doesn't have them
    assert uvfits_uv != ms_uv
    # they are equal if only required parameters are checked:
    assert uvfits_uv.__eq__(ms_uv, check_extra=False)

    # set those parameters to none to check that the rest of the objects match
    ms_uv.antenna_diameters = None

    for p in uvfits_uv.extra():
        fits_param = getattr(uvfits_uv, p)
        ms_param = getattr(ms_uv, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uvfits_uv, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uvfits_uv.extra_keywords = {}
    ms_uv.extra_keywords = {}

    assert uvfits_uv == ms_uv
    del(ms_uv)
    del(uvfits_uv)


@uvtest.skipIf_no_casa
def test_readMSWriteUVFITS():
    """
    read ms, write uvfits test.
    Read in ms file, write out as uvfits, read back in and check for
    object equality.
    """
    ms_uv = UVData()
    uvfits_uv = UVData()
    ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    testfile = os.path.join(DATA_PATH, 'test/outtest_uvfits')
    ms_uv.read(ms_file)
    ms_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    assert uvfits_uv == ms_uv
    del(ms_uv)
    del(uvfits_uv)


@uvtest.skipIf_no_casa
def test_readMSWriteMiriad():
    """
    read ms, write miriad test.
    Read in ms file, write out as miriad, read back in and check for
    object equality.
    """
    ms_uv = UVData()
    miriad_uv = UVData()
    ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad')
    ms_uv.read(ms_file)
    ms_uv.write_miriad(testfile, clobber=True)
    uvtest.checkWarnings(miriad_uv.read_miriad, [testfile],
                         message='Telescope EVLA is not')

    assert miriad_uv == ms_uv


@uvtest.skipIf_no_casa
def test_multi_files():
    """
    Reading multiple files at once.
    """
    uv_full = UVData()
    uv_multi = UVData()
    uvfits_file = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile1 = os.path.join(DATA_PATH, 'multi_1.ms')
    testfile2 = os.path.join(DATA_PATH, 'multi_2.ms')
    uv_multi.read([testfile1, testfile2])
    # Casa scrambles the history parameter. Replace for now.
    uv_multi.history = uv_full.history

    # the objects won't be equal because uvfits adds some optional parameters
    # and the ms sets default antenna diameters even thoug the uvfits file doesn't have them
    assert uv_multi != uv_full
    # they are equal if only required parameters are checked:
    assert uv_multi.__eq__(uv_full, check_extra=False)

    # set those parameters to none to check that the rest of the objects match
    uv_multi.antenna_diameters = None

    for p in uv_full.extra():
        fits_param = getattr(uv_full, p)
        ms_param = getattr(uv_multi, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uv_full, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uv_full.extra_keywords = {}
    uv_multi.extra_keywords = {}

    assert uv_multi == uv_full
    del(uv_full)
    del(uv_multi)


@uvtest.skipIf_no_casa
def test_multi_files_axis():
    """
    Reading multiple files at once, setting axis keyword
    """
    uv_full = UVData()
    uv_multi = UVData()
    uvfits_file = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile1 = os.path.join(DATA_PATH, 'multi_1.ms')
    testfile2 = os.path.join(DATA_PATH, 'multi_2.ms')
    uv_multi.read([testfile1, testfile2], axis='freq')
    # Casa scrambles the history parameter. Replace for now.
    uv_multi.history = uv_full.history

    # the objects won't be equal because uvfits adds some optional parameters
    # and the ms sets default antenna diameters even thoug the uvfits file doesn't have them
    assert uv_multi != uv_full
    # they are equal if only required parameters are checked:
    assert uv_multi.__eq__(uv_full, check_extra=False)

    # set those parameters to none to check that the rest of the objects match
    uv_multi.antenna_diameters = None

    for p in uv_full.extra():
        fits_param = getattr(uv_full, p)
        ms_param = getattr(uv_multi, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uv_full, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uv_full.extra_keywords = {}
    uv_multi.extra_keywords = {}

    assert uv_multi == uv_full


@uvtest.skipIf_no_casa
def test_bad_col_name():
    """
    Test error with invalid column name.
    """
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')

    with pytest.raises(ValueError) as cm:
        UV.read_ms(testfile, data_column='FOO')
    assert str(cm.value).startswith('Invalid data_column value supplied')
