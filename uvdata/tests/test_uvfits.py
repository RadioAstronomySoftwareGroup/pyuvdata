import nose.tools as nt
import os
import os.path as op
import shutil
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
from astropy.io import fits
from test_functions import *


def test_ReadNRAO():
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    expected_extra_keywords = ['OBSERVER', 'SORTORD', 'SPECSYS',
                               'RESTFREQ', 'ORIGIN']
    read_out, status = checkWarnings(UV.read, [testfile, 'uvfits'],
                                     message='Telescope EVLA is not')
    nt.assert_true(read_out)
    nt.assert_true(status)
    nt.assert_equal(expected_extra_keywords.sort(),
                    UV.extra_keywords.keys().sort())
    del(UV)


def test_noSPW():
    UV = UVData()
    testfile_no_spw = '../data/zen.2456865.60537.xy.uvcRREAAM.uvfits'
    read_output, status = checkWarnings(UV.read, [testfile_no_spw, 'uvfits'],
                                        known_warning='paper_uvfits')
    nt.assert_true(read_output)
    nt.assert_true(status)
    del(UV)


# def test_readRTS():
#     UV = UVData()
#     testfile = '../data/pumav2_SelfCal300_Peel300_01.uvfits'
#     test = UV.read_uvfits(testfile)
#     nt.assert_true(test)

def test_BreakReadUVFits():
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    multi_subarray_file = '../data/multi_subarray.uvfits'
    nt.assert_raises(ValueError, UV.read, testfile, 'vufits')  # Wrong filetype
    del(UV)
    UV = UVData()
    nt.assert_raises(ValueError, UV.read, multi_subarray_file, 'uvfits')
    del(UV)


def test_writeNRAO():
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    write_file = '../data/test/outtest_casa_1src_1spw.uvfits'
    read_out, status = checkWarnings(UV.read, [testfile, 'uvfits'],
                                     message='Telescope EVLA is not')
    test = UV.write(write_file, file_type='uvfits')
    nt.assert_true(status)
    nt.assert_true(test)
    del(UV)


def test_spwnotsupported():
    UV = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1scan.uvfits'
    nt.assert_raises(ValueError, UV.read, testfile, 'uvfits')
    del(UV)


def test_readwriteread():
    uv_in = UVData()
    uv_out = UVData()
    testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
    write_file = '../data/test/outtest_casa.uvfits'
    read_out, read_status = checkWarnings(uv_in.read, [testfile, 'uvfits'],
                                          message='Telescope EVLA is not')
    uv_in.write(write_file, file_type='uvfits')
    write_out, write_status = checkWarnings(uv_out.read, [write_file, 'uvfits'],
                                            message='Telescope EVLA is not')
    nt.assert_true(read_status)
    nt.assert_true(write_status)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)
