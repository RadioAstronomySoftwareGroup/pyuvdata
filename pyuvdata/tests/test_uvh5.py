# -*- coding: utf-8 -*-

"""Tests for HDF5 object

"""
from __future__ import absolute_import, division, print_function

import os
import copy
import numpy as np
import nose.tools as nt
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
import warnings


def test_ReadMiriadWriteUVH5ReadUVH5():
    """
    Miriad round trip test
    """
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_miriad.h5')
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         nwarnings=1, category=[UserWarning],
                         message=['Altitude is not present'])
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read_uvh5(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


def test_ReadUVFITSWriteUVH5ReadUVH5():
    """
    UVFITS round trip test
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.h5')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read_uvh5(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


def test_ReadUVH5Errors():
    """
    Test raising errors in read_uvh5 function
    """
    uv_in = UVData()
    fake_file = os.path.join(DATA_PATH, 'fake_file.hdf5')
    nt.assert_raises(IOError, uv_in.read_uvh5, fake_file)

    return


def test_WriteUVH5Errors():
    """
    Test raising errors in write_uvh5 function
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.h5')
    with open(testfile, 'a'):
        os.utime(testfile, None)
    nt.assert_raises(ValueError, uv_in.write_uvh5, testfile)

    # use clobber=True to write out anyway
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read_uvh5(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


def test_UVH5OptionalParameters():
    """
    Test reading and writing optional parameters not in sample files
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.h5')

    # set optional parameters
    uv_in.x_orientation = 'east'
    uv_in.antenna_diameters = np.ones_like(uv_in.antenna_numbers) * 1.
    uv_in.uvplane_reference_time = 0

    # write out and read back in
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read_uvh5(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return
