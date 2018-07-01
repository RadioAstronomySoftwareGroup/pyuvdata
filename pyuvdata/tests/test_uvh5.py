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


def test_UVH5CompressionOptions():
    """
    Test writing data with compression filters
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits_compression.h5')

    # write out and read back in
    uv_in.write_uvh5(testfile, clobber=True, data_compression="lzf",
                     flags_compression=None, nsample_compression=None)
    uv_out.read_uvh5(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


def test_UVH5ReadMultiple_files():
    """
    Test reading multiple uvh5 files
    """
    uv_full = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile1 = os.path.join(DATA_PATH, 'test/uv1.h5')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2.h5')
    uvtest.checkWarnings(uv_full.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvh5(testfile1, clobber=True)
    uv2.write_uvh5(testfile2, clobber=True)
    uv1.read_uvh5([testfile1, testfile2])
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific frequencies using pyuvdata. '
                                           'Combined data along frequency axis using'
                                           ' pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # clean up
    os.remove(testfile1)
    os.remove(testfile2)

    return


def test_UVH5PartialRead():
    """
    Test reading in only part of a dataset from disk
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uvh5_uv.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.h5')
    uvh5_uv.write_uvh5(testfile, clobber=True)

    # select on antennas
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])
    uvh5_uv.read_uvh5(testfile, antenna_nums=ants_to_keep)
    uvh5_uv2.read_uvh5(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # select on frequency channels
    chans_to_keep = np.arange(12, 22)
    uvh5_uv.read_uvh5(testfile, freq_chans=chans_to_keep)
    uvh5_uv2.read_uvh5(testfile)
    uvh5_uv2.select(freq_chans=chans_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # select on pols
    pols_to_keep = [-1, -2]
    uvh5_uv.read_uvh5(testfile, polarizations=pols_to_keep)
    uvh5_uv2.read_uvh5(testfile)
    uvh5_uv2.select(polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # now test selecting on multiple axes
    # frequencies first
    uvh5_uv.read_uvh5(testfile, antenna_nums=ants_to_keep,
                      freq_chans=chans_to_keep,
                      polarizations=pols_to_keep)
    uvh5_uv2.read_uvh5(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # baselines first
    ants_to_keep = np.array([0, 1])
    uvh5_uv.read_uvh5(testfile, antenna_nums=ants_to_keep,
                      freq_chans=chans_to_keep,
                      polarizations=pols_to_keep)
    uvh5_uv2.read_uvh5(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # polarizations first
    ants_to_keep = np.array([0, 1, 2, 3, 6, 7, 8, 11, 14, 18, 19, 20, 21, 22])
    chans_to_keep = np.arange(12, 64)
    uvh5_uv.read_uvh5(testfile, antenna_nums=ants_to_keep,
                      freq_chans=chans_to_keep,
                      polarizations=pols_to_keep)
    uvh5_uv2.read_uvh5(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # clean up
    os.remove(testfile)

    return


def test_UVH5PartialWrite():
    """
    Test writing an entire UVH5 file in pieces
    """
    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(full_uvh5.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.h5')
    full_uvh5.write_uvh5(testfile, clobber=True)
    full_uvh5.read_uvh5(testfile)

    # delete data arrays in partial file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.h5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # write to file by iterating over antpairpol
    antpairpols = full_uvh5.get_antpairpols()
    for key in antpairpols:
        data = full_uvh5.get_data(key, squeeze='none')
        flags = full_uvh5.get_flags(key, squeeze='none')
        nsamples = full_uvh5.get_nsamples(key, squeeze='none')
        partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                     bls=key)

    # now read in the full file and make sure that it matches the original
    partial_uvh5.read_uvh5(partial_testfile)
    nt.assert_equal(full_uvh5, partial_uvh5)

    # start over, and write frequencies
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)
    Nfreqs = full_uvh5.Nfreqs
    Hfreqs = Nfreqs // 2
    freqs1 = np.arange(Hfreqs)
    freqs2 = np.arange(Hfreqs, Nfreqs)
    data = full_uvh5.data_array[:, :, freqs1, :]
    flags = full_uvh5.flag_array[:, :, freqs1, :]
    nsamples = full_uvh5.nsample_array[:, :, freqs1, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 freq_chans=freqs1)
    data = full_uvh5.data_array[:, :, freqs2, :]
    flags = full_uvh5.flag_array[:, :, freqs2, :]
    nsamples = full_uvh5.nsample_array[:, :, freqs2, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 freq_chans=freqs2)

    # read in the full file and make sure it matches
    partial_uvh5.read_uvh5(partial_testfile)
    nt.assert_equal(full_uvh5, partial_uvh5)

    # start over, write chunks of blts
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)
    Nblts = full_uvh5.Nblts
    Hblts = Nblts // 2
    blts1 = np.arange(Hblts)
    blts2 = np.arange(Hblts, Nblts)
    data = full_uvh5.data_array[blts1, :, :, :]
    flags = full_uvh5.flag_array[blts1, :, :, :]
    nsamples = full_uvh5.nsample_array[blts1, :, :, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 blt_inds=blts1)
    data = full_uvh5.data_array[blts2, :, :, :]
    flags = full_uvh5.flag_array[blts2, :, :, :]
    nsamples = full_uvh5.nsample_array[blts2, :, :, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 blt_inds=blts2)

    # read in the full file and make sure it matches
    partial_uvh5.read_uvh5(partial_testfile)
    nt.assert_equal(full_uvh5, partial_uvh5)

    # start over, write groups of pols
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)
    Npols = full_uvh5.Npols
    Hpols = Npols // 2
    pols1 = np.arange(Hpols)
    pols2 = np.arange(Hpols, Npols)
    data = full_uvh5.data_array[:, :, :, pols1]
    flags = full_uvh5.flag_array[:, :, :, pols1]
    nsamples = full_uvh5.nsample_array[:, :, :, pols1]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 polarizations=full_uvh5.polarization_array[:Hpols])
    data = full_uvh5.data_array[:, :, :, pols2]
    flags = full_uvh5.flag_array[:, :, :, pols2]
    nsamples = full_uvh5.nsample_array[:, :, :, pols2]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 polarizations=full_uvh5.polarization_array[Hpols:])

    # read in the full file and make sure it matches
    partial_uvh5.read_uvh5(partial_testfile)
    nt.assert_equal(full_uvh5, partial_uvh5)

    # clean up
    os.remove(testfile)
    os.remove(partial_testfile)

    return


def test_UVH5PartialWriteErrors():
    """
    Test errors in uvh5_write_part method
    """
    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(full_uvh5.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.h5')
    full_uvh5.write_uvh5(testfile, clobber=True)
    full_uvh5.read_uvh5(testfile)

    # get a waterfall
    antpairpols = full_uvh5.get_antpairpols()
    key = antpairpols[0]
    data = full_uvh5.get_data(key, squeeze='none')
    flags = full_uvh5.get_data(key, squeeze='none')
    nsamples = full_uvh5.get_data(key, squeeze='none')

    # delete data arrays in partial file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # try to write to a file that doesn't exists
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.h5')
    if os.path.exists(partial_testfile):
        os.remove(partial_testfile)
    nt.assert_raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data,
                     flags, nsamples, bls=key)

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # pass in arrays that are different sizes
    nt.assert_raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data,
                     flags[:, :, :, 0], nsamples, bls=key)
    nt.assert_raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data,
                     flags, nsamples[:, :, :, 0], bls=key)

    # pass in arrays that are the same size, but don't match expected shape
    nt.assert_raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data[:, :, :, 0],
                     flags[:, :, :, 0], nsamples[:, :, :, 0])

    # clean up
    os.remove(testfile)
    os.remove(partial_testfile)

    return


def test_UVH5InitializeFile():
    """
    Test initializing a UVH5 file on disk
    """
    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(full_uvh5.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.h5')
    full_uvh5.write_uvh5(testfile, clobber=True)
    full_uvh5.read_uvh5(testfile)
    full_uvh5.data_array = None
    full_uvh5.flag_array = None
    full_uvh5.nsample_array = None

    # initialize file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.h5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # read it in and make sure that the metadata matches the original
    partial_uvh5.read_uvh5(partial_testfile, read_data=False)
    nt.assert_equal(partial_uvh5, full_uvh5)

    # add options for compression
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True, data_compression="lzf",
                                      flags_compression=None, nsample_compression=None)
    partial_uvh5.read_uvh5(partial_testfile, read_data=False)
    nt.assert_equal(partial_uvh5, full_uvh5)

    # check that an error is raised then file exists and clobber is False
    nt.assert_raises(ValueError, partial_uvh5.initialize_uvh5_file, partial_testfile, clobber=False)

    # clean up
    os.remove(testfile)
    os.remove(partial_testfile)

    return
