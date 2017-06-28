"""Tests for UVFITS object."""
import nose.tools as nt
import os
from pyuvdata import UVData
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import copy
import numpy as np


def test_ReadNRAO():
    """Test reading in a CASA tutorial uvfits file."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    expected_extra_keywords = ['OBSERVER', 'SORTORD', 'SPECSYS',
                               'RESTFREQ', 'ORIGIN']
    uvtest.checkWarnings(UV.read_uvfits, [testfile], message='Telescope EVLA is not')
    nt.assert_equal(expected_extra_keywords.sort(),
                    UV.extra_keywords.keys().sort())
    del(UV)


def test_noSPW():
    """Test reading in a PAPER uvfits file with no spw axis."""
    UV = UVData()
    testfile_no_spw = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAAM.uvfits')
    uvtest.checkWarnings(UV.read_uvfits, [testfile_no_spw], known_warning='paper_uvfits')
    del(UV)


# this test commented out because the file is too large to include in the repo
# def test_readRTS():
#    """Test reading in an RTS UVFITS file."""
#     UV = UVData()
#     testfile = os.path.join(DATA_PATH, 'pumav2_SelfCal300_Peel300_01.uvfits')
#     test = UV.read_uvfits(testfile)
#     nt.assert_true(test)

def test_breakReadUVFits():
    """Test errors on reading in a uvfits file with subarrays and other problems."""
    UV = UVData()
    multi_subarray_file = os.path.join(DATA_PATH, 'multi_subarray.uvfits')
    nt.assert_raises(ValueError, UV.read_uvfits, multi_subarray_file)

    del(UV)


def test_spwnotsupported():
    """Test errors on reading in a uvfits file with multiple spws."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1scan.uvfits')
    nt.assert_raises(ValueError, UV.read_uvfits, testfile)
    del(UV)


def test_readwriteread():
    """
    CASA tutorial uvfits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    uv_in = UVData()
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [testfile], message='Telescope EVLA is not')
    uv_in.write_uvfits(write_file)
    uvtest.checkWarnings(uv_out.read_uvfits, [write_file], message='Telescope EVLA is not')
    nt.assert_equal(uv_in, uv_out)

    # check error if timesys is 'IAT'
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [testfile], message='Telescope EVLA is not')
    uv_in.timesys = 'IAT'
    nt.assert_raises(ValueError, uv_in.write_uvfits, write_file)

    del(uv_in)
    del(uv_out)


def test_ReadUVFitsWriteMiriad():
    """
    read uvfits, write miriad test.
    Read in uvfits file, write out as miriad, read back in and check for
    object equality.
    """
    uvfits_uv = UVData()
    miriad_uv = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad')
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uvfits_uv.write_miriad(testfile, clobber=True)
    uvtest.checkWarnings(miriad_uv.read_miriad, [testfile], message='Telescope EVLA is not')

    # the objects will not be equal because extra_keywords are not written to
    # or read from miriad files
    nt.assert_false(miriad_uv == uvfits_uv)
    # they are equal if only required parameters are checked:
    nt.assert_true(miriad_uv.__eq__(uvfits_uv, check_extra=False))

    # remove the extra_keywords to check that the rest of the objects are equal
    uvfits_uv.extra_keywords = {}
    nt.assert_equal(miriad_uv, uvfits_uv)

    # check that setting the phase_type keyword also works
    uvtest.checkWarnings(miriad_uv.read_miriad, [testfile], {'phase_type': 'phased'},
                         message='Telescope EVLA is not')

    # check that setting the phase_type to drift raises an error
    nt.assert_raises(ValueError, miriad_uv.read_miriad, testfile, phase_type='drift'),

    # check that setting it works after selecting a single time
    uvfits_uv.select(times=uvfits_uv.time_array[0])
    uvfits_uv.write_miriad(testfile, clobber=True)
    uvtest.checkWarnings(miriad_uv.read_miriad, [testfile],
                         message='Telescope EVLA is not')

    nt.assert_equal(miriad_uv, uvfits_uv)

    del(uvfits_uv)
    del(miriad_uv)



def test_multi_files():
    """
    Reading multiple files at once.
    """
    uv_full = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile1 = os.path.join(DATA_PATH, 'test/uv1.uvfits')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvfits(testfile1)
    uv2.write_uvfits(testfile2)
    uvtest.checkWarnings(uv1.read_uvfits, [[testfile1, testfile2]], nwarnings=2,
                         category=[UserWarning, UserWarning],
                         message=['Telescope EVLA is not', 'Telescope EVLA is not'])
    # Check history is correct, before replacing and doing a full object check
    nt.assert_equal(uv_full.history + '  Downselected to specific frequencies'
                    ' using pyuvdata. Combined data along frequency axis using'
                    ' pyuvdata.', uv1.history.replace('\n', ''))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)


def test_readMSWriteUVFits_CASAHistory():
    """
    read in .ms file.
    Write to a uvfits file, read back in and check for casa_history parameter
    """
    ms_uv = UVData()
    uvfits_uv = UVData()
    ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    testfile = os.path.join(DATA_PATH, 'test/outtest_uvfits')
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file], message='Telescope EVLA is not',
                         nwarnings=0)
    ms_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    nt.assert_equal(ms_uv, uvfits_uv)
    del(uvfits_uv)
    del(ms_uv)
