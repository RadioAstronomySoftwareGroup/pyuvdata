"""Tests for UVFITS object."""
import nose.tools as nt
from uvdata import UVData
import uvdata.tests as uvtest


def test_ReadNRAO():
    """Test reading in a CASA tutorial uvfits file."""
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    expected_extra_keywords = ['OBSERVER', 'SORTORD', 'SPECSYS',
                               'RESTFREQ', 'ORIGIN']
    status = uvtest.checkWarnings(UV.read_uvfits, [testfile],
                                  message='Telescope EVLA is not')
    nt.assert_true(status)
    nt.assert_equal(expected_extra_keywords.sort(),
                    UV.extra_keywords.keys().sort())
    del(UV)


def test_noSPW():
    """Test reading in a PAPER uvfits file with no spw axis."""
    UV = UVData()
    testfile_no_spw = '../data/zen.2456865.60537.xy.uvcRREAAM.uvfits'
    status = uvtest.checkWarnings(UV.read_uvfits, [testfile_no_spw],
                                  known_warning='paper_uvfits')
    nt.assert_true(status)
    del(UV)


# this test commented out because the file is too large to include in the repo
# def test_readRTS():
#    """Test reading in an RTS UVFITS file."""
#     UV = UVData()
#     testfile = '../data/pumav2_SelfCal300_Peel300_01.uvfits'
#     test = UV.read_uvfits(testfile)
#     nt.assert_true(test)

def test_breakReadUVFits():
    """Test errors on reading in a uvfits file with subarrays."""
    UV = UVData()
    multi_subarray_file = '../data/multi_subarray.uvfits'
    nt.assert_raises(ValueError, UV.read_uvfits, multi_subarray_file)
    del(UV)


def test_spwnotsupported():
    """Test errors on reading in a uvfits file with multiple spws."""
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1scan.uvfits'
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
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    write_file = '../data/test/outtest_casa.uvfits'
    read_status = uvtest.checkWarnings(uv_in.read_uvfits, [testfile],
                                       message='Telescope EVLA is not')
    uv_in.write_uvfits(write_file)
    write_status = uvtest.checkWarnings(uv_out.read_uvfits, [write_file],
                                        message='Telescope EVLA is not')
    nt.assert_true(read_status)
    nt.assert_true(write_status)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


def test_ReadUVFitsWriteMiriad():
    """
    uvfits to miriad loopback test. ***THIS TEST IS KNOWN TO CURRENTLY FAIL.***

    Read in uvfits file, write out as miriad, read back in and check for
    object equality.
    """
    uvfits_uv = UVData()
    miriad_uv = UVData()
    uvfits_file = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    testfile = '../data/test/outtest_miriad'
    read_status = uvtest.checkWarnings(uvfits_uv.read_uvfits, [uvfits_file],
                                       message='Telescope EVLA is not')
    uvfits_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read_miriad(testfile)
    nt.assert_true(read_status)
    nt.assert_equal(miriad_uv, uvfits_uv)
    del(uvfits_uv)
    del(miriad_uv)
