"""Tests for UVFITS object."""
import numpy as np
import copy
import os
import nose.tools as nt
from pyuvdata import UVData
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


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

    # check that if x_orientation is set, it's read back out properly
    uv_in.x_orientation = 'east'
    uv_in.write_uvfits(write_file)
    uvtest.checkWarnings(uv_out.read_uvfits, [write_file], message='Telescope EVLA is not')
    nt.assert_equal(uv_in, uv_out)

    # check that if antenna_diameters is set, it's read back out properly
    uvtest.checkWarnings(uv_in.read_uvfits, [testfile], message='Telescope EVLA is not')
    uv_in.antenna_diameters = np.zeros((uv_in.Nants_telescope,), dtype=np.float) + 14.0
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


def test_extra_keywords():
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    uv_in.extra_keywords['testdict'] = {'testkey': 23}
    uvtest.checkWarnings(uv_in.check, message=['testdict in extra_keywords is a '
                                               'list, array or dict'])
    nt.assert_raises(TypeError, uv_in.write_uvfits, testfile, run_check=False)
    uv_in.extra_keywords.pop('testdict')

    uv_in.extra_keywords['testlist'] = [12, 14, 90]
    uvtest.checkWarnings(uv_in.check, message=['testlist in extra_keywords is a '
                                               'list, array or dict'])
    nt.assert_raises(TypeError, uv_in.write_uvfits, testfile, run_check=False)
    uv_in.extra_keywords.pop('testlist')

    uv_in.extra_keywords['testarr'] = np.array([12, 14, 90])
    uvtest.checkWarnings(uv_in.check, message=['testarr in extra_keywords is a '
                                               'list, array or dict'])
    nt.assert_raises(TypeError, uv_in.write_uvfits, testfile, run_check=False)
    uv_in.extra_keywords.pop('testarr')

    # check for warnings with extra_keywords keys that are too long
    uv_in.extra_keywords['test_long_key'] = True
    uvtest.checkWarnings(uv_in.check, message=['key test_long_key in extra_keywords '
                                               'is longer than 8 characters'])
    uvtest.checkWarnings(uv_in.write_uvfits, [testfile], {'run_check': False},
                         message=['key test_long_key in extra_keywords is longer than 8 characters'])
    uv_in.extra_keywords.pop('test_long_key')

    # check handling of boolean keywords
    uv_in.extra_keywords['bool'] = True
    uv_in.extra_keywords['bool2'] = False
    uv_in.write_uvfits(testfile)
    uvtest.checkWarnings(uv_out.read_uvfits, [testfile], message='Telescope EVLA is not')

    nt.assert_equal(uv_in, uv_out)
    uv_in.extra_keywords.pop('bool')
    uv_in.extra_keywords.pop('bool2')

    # check handling of int-like keywords
    uv_in.extra_keywords['int1'] = np.int(5)
    uv_in.extra_keywords['int2'] = 7
    uv_in.write_uvfits(testfile)
    uvtest.checkWarnings(uv_out.read_uvfits, [testfile], message='Telescope EVLA is not')

    nt.assert_equal(uv_in, uv_out)
    uv_in.extra_keywords.pop('int1')
    uv_in.extra_keywords.pop('int2')

    # check handling of float-like keywords
    uv_in.extra_keywords['float1'] = np.int64(5.3)
    uv_in.extra_keywords['float2'] = 6.9
    uv_in.write_uvfits(testfile)
    uvtest.checkWarnings(uv_out.read_uvfits, [testfile], message='Telescope EVLA is not')

    nt.assert_equal(uv_in, uv_out)
    uv_in.extra_keywords.pop('float1')
    uv_in.extra_keywords.pop('float2')

    # check handling of complex-like keywords
    uv_in.extra_keywords['complex1'] = np.complex64(5.3 + 1.2j)
    uv_in.extra_keywords['complex2'] = 6.9 + 4.6j
    uv_in.write_uvfits(testfile)
    uvtest.checkWarnings(uv_out.read_uvfits, [testfile], message='Telescope EVLA is not')

    nt.assert_equal(uv_in, uv_out)


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
                         message=['Telescope EVLA is not'])
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific frequencies using pyuvdata. '
                                           'Combined data along frequency axis '
                                           'using pyuvdata.', uv1.history))

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
    ms_uv.read_ms(ms_file)
    ms_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    nt.assert_equal(ms_uv, uvfits_uv)
    del(uvfits_uv)
    del(ms_uv)
