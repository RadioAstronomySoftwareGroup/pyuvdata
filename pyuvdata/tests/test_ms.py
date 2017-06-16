"""Tests for MS object."""
import nose.tools as nt
import os
import copy
from pyuvdata import UVData
import glob as glob
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
import numpy as np


def test_readNRAO():
    """Test reading in a CASA tutorial ms file."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    expected_extra_keywords = ['data_column', 'antenna_positions']

    uvtest.checkWarnings(UV.read_ms,
                         [testfile],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    nt.assert_equal(expected_extra_keywords.sort(),
                    UV.extra_keywords.keys().sort())
    del(UV)


def test_noSPW():
    """Test reading in a PAPER ms convertes by CASA from a uvfits with no spw axis."""
    UV = UVData()
    testfile_no_spw = os.path.join(
        DATA_PATH, 'zen.2456865.60537.xy.uvcRREAAM.ms')
    uvtest.checkWarnings(UV.read_ms, [testfile_no_spw],
                         nwarnings=1)
    del(UV)

#!!!This test does not seem to work because importuvfits did not appear to preserve
#!!!multiple subarray values going from .uvfits -> .ms
#
# def test_breakReadMS():
#    UV=UVData()
#    multi_subarray_file=os.path.join(DATA_PATH,'multi_subarray.ms')
#    nt.assert_raises(ValueError,UV.read_ms,multi_subarray_file)
#    del(UV)

# Need a method to test casacore import error!


def test_spwnotsupported():
    """Test errors on reading in an ms file with multiple spws."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1scan.ms')
    nt.assert_raises(ValueError, UV.read_ms, testfile)
    del(UV)


def test_readMSreadUVFITS():
    """
    this test tests that a uvdata object instantiated 
    from an ms file
    created with CASA's importuvfits 
    is equal to a uvdata object instantiated from the original
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
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    # set histories to identical blank strings since we do not expect
    # them to be the same anyways.
    ms_uv.history = ""
    uvfits_uv.history = ""
    nt.assert_equal(uvfits_uv, ms_uv)
    del(ms_uv)
    del(uvfits_uv)


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
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    ms_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    nt.assert_equal(uvfits_uv, ms_uv)
    del(ms_uv)
    del(uvfits_uv)


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
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    ms_uv.write_miriad(testfile, clobber=True)
    uvtest.checkWarnings(miriad_uv.read_miriad, [testfile],
                         message='Telescope EVLA is not')
    nt.assert_equal(miriad_uv, ms_uv)
    del(ms_uv)
    del(miriad_uv)


def test_readUVFITS_readMS():
    """
    Test reading in an ms produced by casa importuvfits and compare to original uvfits
    """
    ms_uv = UVData()
    uvfits_uv = UVData()
    ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    uvfits_file = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [uvfits_file],
                         message='Telescope EVLA is not')
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file, True, True, 'DATA', 'AIPS', True],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    # Casa scrambles the history parameter. Replace for now.
    ms_uv.history = uvfits_uv.history
    nt.assert_equal(ms_uv, uvfits_uv)
    del(ms_uv)
    del(uvfits_uv)


def test_multi_files():
    """
    Reading multiple files at once.
    """
    uv_full = UVData()
    uv_multi = UVData()
    uvfits_file = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [
                         uvfits_file], message='Telescope EVLA is not')
    testfile1 = os.path.join(DATA_PATH, 'multi_1.ms')
    testfile2 = os.path.join(DATA_PATH, 'multi_2.ms')
    uv_multi.read_ms([testfile1, testfile2])
    # Casa scrambles the history parameter. Replace for now.
    uv_multi.history = uv_full.history
    nt.assert_equal(uv_multi, uv_full)
    del(uv_full)
    del(uv_multi)
