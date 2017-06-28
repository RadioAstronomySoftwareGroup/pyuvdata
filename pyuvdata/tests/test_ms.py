"""Tests for MS object."""
import nose.tools as nt
import os
import copy
import numpy as np
from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
from pyuvdata import UVFITS


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


def test_spwnotsupported():
    """Test errors on reading in an ms file with multiple spws."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1scan.ms')
    nt.assert_raises(ValueError, UV.read_ms, testfile)
    del(UV)


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
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    # set histories to identical blank strings since we do not expect
    # them to be the same anyways.
    ms_uv.history = ""
    uvfits_uv.history = ""

    # the objects won't be equal because uvfits adds some optional parameters
    nt.assert_false(uvfits_uv == ms_uv)
    # they are equal if only required parameters are checked:
    nt.assert_true(uvfits_uv.__eq__(ms_uv, check_extra=False))

    # set those parameters to none to check that the rest of the objects match
    for p in uvfits_uv.extra():
        fits_param = getattr(uvfits_uv, p)
        ms_param = getattr(ms_uv, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uvfits_uv, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uvfits_uv.extra_keywords = {}
    ms_uv.extra_keywords = {}

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
    # the objects will not be equal because extra_keywords are not written to
    # or read from miriad files
    nt.assert_false(miriad_uv == ms_uv)
    # they are equal if only required parameters are checked:
    nt.assert_true(miriad_uv.__eq__(ms_uv, check_extra=False))

    # remove the extra_keywords to check that the rest of the objects are equal
    ms_uv.extra_keywords = {}
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
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    # Casa scrambles the history parameter. Replace for now.
    ms_uv.history = uvfits_uv.history

    # the objects won't be equal because uvfits adds some optional parameters
    nt.assert_false(uvfits_uv == ms_uv)
    # they are equal if only required parameters are checked:
    nt.assert_true(uvfits_uv.__eq__(ms_uv, check_extra=False))

    # set those parameters to none to check that the rest of the objects match
    for p in uvfits_uv.extra():
        fits_param = getattr(uvfits_uv, p)
        ms_param = getattr(ms_uv, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uvfits_uv, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uvfits_uv.extra_keywords = {}
    ms_uv.extra_keywords = {}

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

    # the objects won't be equal because uvfits adds some optional parameters
    nt.assert_false(uv_multi == uv_full)
    # they are equal if only required parameters are checked:
    nt.assert_true(uv_multi.__eq__(uv_full, check_extra=False))

    # set those parameters to none to check that the rest of the objects match
    for p in uv_full.extra():
        fits_param = getattr(uv_full, p)
        ms_param = getattr(uv_multi, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uv_full, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uv_full.extra_keywords = {}
    uv_multi.extra_keywords = {}

    nt.assert_equal(uv_multi, uv_full)
    del(uv_full)
    del(uv_multi)
