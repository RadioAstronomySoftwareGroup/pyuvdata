import nose.tools as nt
from uvdata import UVData
import uvdata.tests as uvtest


def test_ReadNRAO():
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
    UV = UVData()
    testfile_no_spw = '../data/zen.2456865.60537.xy.uvcRREAAM.uvfits'
    status = uvtest.checkWarnings(UV.read_uvfits, [testfile_no_spw],
                                  known_warning='paper_uvfits')
    nt.assert_true(status)
    del(UV)


# this test commented out because the file is too large to include in the repo
# def test_readRTS():
#     UV = UVData()
#     testfile = '../data/pumav2_SelfCal300_Peel300_01.uvfits'
#     test = UV.read_uvfits(testfile)
#     nt.assert_true(test)

def test_breakReadUVFits():
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    multi_subarray_file = '../data/multi_subarray.uvfits'
    nt.assert_raises(ValueError, UV.read_uvfits, multi_subarray_file)
    del(UV)


def test_writeNRAO():
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    write_file = '../data/test/outtest_casa_1src_1spw.uvfits'
    status = uvtest.checkWarnings(UV.read_uvfits, [testfile],
                                  message='Telescope EVLA is not')
    UV.write_uvfits(write_file)
    nt.assert_true(status)
    del(UV)


def test_spwnotsupported():
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1scan.uvfits'
    nt.assert_raises(ValueError, UV.read_uvfits, testfile)
    del(UV)


def test_readwriteread():
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
