# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for HDF5 object

"""
from __future__ import absolute_import, division, print_function

import os
import copy
import numpy as np
import nose.tools as nt
from astropy.time import Time

from pyuvdata import UVData
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest

try:
    import h5py
except(ImportError):
    pass


@uvtest.skipIf_no_h5py
def test_ReadMiriadWriteUVH5ReadUVH5():
    """
    Miriad round trip test
    """
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_miriad.uvh5')
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         nwarnings=1, category=[UserWarning],
                         message=['Altitude is not present'])
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    nt.assert_equal(uv_in, uv_out)

    # also test round-tripping phased data
    uv_in.phase_to_time(Time(np.mean(uv_in.time_array), format='jd'))
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)

    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_ReadUVFITSWriteUVH5ReadUVH5():
    """
    UVFITS round trip test
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_ReadUVH5Errors():
    """
    Test raising errors in read function
    """
    uv_in = UVData()
    fake_file = os.path.join(DATA_PATH, 'fake_file.uvh5')
    nt.assert_raises(IOError, uv_in.read_uvh5, fake_file)
    nt.assert_raises(ValueError, uv_in.read_uvh5, ['list of', 'fake files'], read_data=False)

    return


@uvtest.skipIf_no_h5py
def test_WriteUVH5Errors():
    """
    Test raising errors in write_uvh5 function
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    with open(testfile, 'a'):
        os.utime(testfile, None)
    nt.assert_raises(ValueError, uv_in.write_uvh5, testfile)

    # use clobber=True to write out anyway
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5OptionalParameters():
    """
    Test reading and writing optional parameters not in sample files
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')

    # set optional parameters
    uv_in.x_orientation = 'east'
    uv_in.antenna_diameters = np.ones_like(uv_in.antenna_numbers) * 1.
    uv_in.uvplane_reference_time = 0

    # write out and read back in
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5CompressionOptions():
    """
    Test writing data with compression filters
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits_compression.uvh5')

    # write out and read back in
    uv_in.write_uvh5(testfile, clobber=True, data_compression="lzf",
                     flags_compression=None, nsample_compression=None)
    uv_out.read(testfile)
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5ReadMultiple_files():
    """
    Test reading multiple uvh5 files
    """
    uv_full = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile1 = os.path.join(DATA_PATH, 'test/uv1.uvh5')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2.uvh5')
    uvtest.checkWarnings(uv_full.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvh5(testfile1, clobber=True)
    uv2.write_uvh5(testfile2, clobber=True)
    uv1.read([testfile1, testfile2])
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific frequencies using pyuvdata. '
                                            'Combined data along frequency axis using'
                                            ' pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # clean up
    os.remove(testfile1)
    os.remove(testfile2)

    return


@uvtest.skipIf_no_h5py
def test_UVH5PartialRead():
    """
    Test reading in only part of a dataset from disk
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uvh5_uv.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    uvh5_uv.write_uvh5(testfile, clobber=True)

    # select on antennas
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # select on frequency channels
    chans_to_keep = np.arange(12, 22)
    uvh5_uv.read(testfile, freq_chans=chans_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(freq_chans=chans_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # select on pols
    pols_to_keep = [-1, -2]
    uvh5_uv.read(testfile, polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # select on read using time_range
    unique_times = np.unique(uvh5_uv.time_array)
    uvtest.checkWarnings(uvh5_uv.read, [testfile],
                         {'time_range': [unique_times[0], unique_times[1]]},
                         message=['Warning: "time_range" keyword is set'])
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(times=unique_times[0:2])
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # now test selecting on multiple axes
    # frequencies first
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # baselines first
    ants_to_keep = np.array([0, 1])
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # polarizations first
    ants_to_keep = np.array([0, 1, 2, 3, 6, 7, 8, 11, 14, 18, 19, 20, 21, 22])
    chans_to_keep = np.arange(12, 64)
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    nt.assert_equal(uvh5_uv, uvh5_uv2)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5PartialWrite():
    """
    Test writing an entire UVH5 file in pieces
    """
    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(full_uvh5.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    full_uvh5.write_uvh5(testfile, clobber=True)
    full_uvh5.read(testfile)

    # delete data arrays in partial file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
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
    partial_uvh5.read(partial_testfile)
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
    partial_uvh5.read(partial_testfile)
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
    partial_uvh5.read(partial_testfile)
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
    partial_uvh5.read(partial_testfile)
    nt.assert_equal(full_uvh5, partial_uvh5)

    # clean up
    os.remove(testfile)
    os.remove(partial_testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5PartialWriteIrregular():
    """
    Test writing a uvh5 file using irregular intervals
    """
    def initialize_with_zeros(uvd, filename):
        """
        Initialize a file with all zeros for data arrays
        """
        uvd.initialize_uvh5_file(filename, clobber=True)
        data_shape = (uvd.Nblts, 1, uvd.Nfreqs, uvd.Npols)
        data = np.zeros(data_shape, dtype=np.complex64)
        flags = np.zeros(data_shape, dtype=np.bool)
        nsamples = np.zeros(data_shape, dtype=np.float32)
        with h5py.File(filename, 'r+') as f:
            dgrp = f['/Data']
            data_dset = dgrp['visdata']
            flags_dset = dgrp['flags']
            nsample_dset = dgrp['nsamples']
            data_dset = data
            flags_dset = flags
            nsample_dset = nsamples
        return

    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(full_uvh5.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    full_uvh5.write_uvh5(testfile, clobber=True)
    full_uvh5.read(testfile)

    # delete data arrays in partial file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single blt to file
    blt_inds = np.arange(1)
    data = full_uvh5.data_array[blt_inds, :, :, :]
    flags = full_uvh5.flag_array[blt_inds, :, :, :]
    nsamples = full_uvh5.nsample_array[blt_inds, :, :, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples, blt_inds=blt_inds)

    # also write the arrays to the partial object
    partial_uvh5.data_array[blt_inds, :, :, :] = data
    partial_uvh5.flag_array[blt_inds, :, :, :] = flags
    partial_uvh5.nsample_array[blt_inds, :, :, :] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # do it again, with a single frequency
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single freq to file
    freq_inds = np.arange(1)
    data = full_uvh5.data_array[:, :, freq_inds, :]
    flags = full_uvh5.flag_array[:, :, freq_inds, :]
    nsamples = full_uvh5.nsample_array[:, :, freq_inds, :]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 freq_chans=freq_inds)

    # also write the arrays to the partial object
    partial_uvh5.data_array[:, :, freq_inds, :] = data
    partial_uvh5.flag_array[:, :, freq_inds, :] = flags
    partial_uvh5.nsample_array[:, :, freq_inds, :] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # do it again, with a single polarization
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # write a single pol to file
    pol_inds = np.arange(1)
    data = full_uvh5.data_array[:, :, :, pol_inds]
    flags = full_uvh5.flag_array[:, :, :, pol_inds]
    nsamples = full_uvh5.nsample_array[:, :, :, pol_inds]
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 polarizations=partial_uvh5.polarization_array[pol_inds])

    # also write the arrays to the partial object
    partial_uvh5.data_array[:, :, :, pol_inds] = data
    partial_uvh5.flag_array[:, :, :, pol_inds] = flags
    partial_uvh5.nsample_array[:, :, :, pol_inds] = nsamples

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # test irregularly spaced blts and freqs
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    freq_inds = [0, 2, 3, 4]
    data_shape = (len(blt_inds), 1, len(freq_inds), full_uvh5.Npols)
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            data[iblt, :, ifreq, :] = full_uvh5.data_array[blt_idx, :, freq_idx, :]
            flags[iblt, :, ifreq, :] = full_uvh5.flag_array[blt_idx, :, freq_idx, :]
            nsamples[iblt, :, ifreq, :] = full_uvh5.nsample_array[blt_idx, :, freq_idx, :]
    uvtest.checkWarnings(partial_uvh5.write_uvh5_part, [partial_testfile, data, flags, nsamples],
                         {'blt_inds': blt_inds, 'freq_chans': freq_inds},
                         message='Selected frequencies are not evenly spaced')

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            partial_uvh5.data_array[blt_idx, :, freq_idx, :] = data[iblt, :, ifreq, :]
            partial_uvh5.flag_array[blt_idx, :, freq_idx, :] = flags[iblt, :, ifreq, :]
            partial_uvh5.nsample_array[blt_idx, :, freq_idx, :] = nsamples[iblt, :, ifreq, :]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # test irregularly spaced freqs and pols
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    freq_inds = [0, 1, 2, 7]
    pol_inds = [0, 1, 3]
    data_shape = (full_uvh5.Nblts, 1, len(freq_inds), len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            data[:, :, ifreq, ipol] = full_uvh5.data_array[:, :, freq_idx, pol_idx]
            flags[:, :, ifreq, ipol] = full_uvh5.flag_array[:, :, freq_idx, pol_idx]
            nsamples[:, :, ifreq, ipol] = full_uvh5.nsample_array[:, :, freq_idx, pol_idx]
    uvtest.checkWarnings(partial_uvh5.write_uvh5_part, [partial_testfile, data, flags, nsamples],
                         {'freq_chans': freq_inds, 'polarizations': full_uvh5.polarization_array[pol_inds]},
                         nwarnings=2, message=['Selected frequencies are not evenly spaced',
                                               'Selected polarization values are not evenly spaced'])

    # also write the arrays to the partial object
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            partial_uvh5.data_array[:, :, freq_idx, pol_idx] = data[:, :, ifreq, ipol]
            partial_uvh5.flag_array[:, :, freq_idx, pol_idx] = flags[:, :, ifreq, ipol]
            partial_uvh5.nsample_array[:, :, freq_idx, pol_idx] = nsamples[:, :, ifreq, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # test irregularly spaced blts and pols
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    pol_inds = [0, 1, 3]
    data_shape = (len(blt_inds), 1, full_uvh5.Nfreqs, len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            data[iblt, :, :, ipol] = full_uvh5.data_array[blt_idx, :, :, pol_idx]
            flags[iblt, :, :, ipol] = full_uvh5.flag_array[blt_idx, :, :, pol_idx]
            nsamples[iblt, :, :, ipol] = full_uvh5.nsample_array[blt_idx, :, :, pol_idx]
    uvtest.checkWarnings(partial_uvh5.write_uvh5_part, [partial_testfile, data, flags, nsamples],
                         {'blt_inds': blt_inds, 'polarizations': full_uvh5.polarization_array[pol_inds]},
                         message='Selected polarization values are not evenly spaced')

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            partial_uvh5.data_array[blt_idx, :, :, pol_idx] = data[iblt, :, :, ipol]
            partial_uvh5.flag_array[blt_idx, :, :, pol_idx] = flags[iblt, :, :, ipol]
            partial_uvh5.nsample_array[blt_idx, :, :, pol_idx] = nsamples[iblt, :, :, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # test irregularly spaced freqs and pols
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    freq_inds = [0, 1, 2, 7]
    pol_inds = [0, 1, 3]
    data_shape = (full_uvh5.Nblts, 1, len(freq_inds), len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            data[:, :, ifreq, ipol] = full_uvh5.data_array[:, :, freq_idx, pol_idx]
            flags[:, :, ifreq, ipol] = full_uvh5.flag_array[:, :, freq_idx, pol_idx]
            nsamples[:, :, ifreq, ipol] = full_uvh5.nsample_array[:, :, freq_idx, pol_idx]
    uvtest.checkWarnings(partial_uvh5.write_uvh5_part, [partial_testfile, data, flags, nsamples],
                         {'freq_chans': freq_inds, 'polarizations': full_uvh5.polarization_array[pol_inds]},
                         nwarnings=2, message=['Selected frequencies are not evenly spaced',
                                               'Selected polarization values are not evenly spaced'])

    # also write the arrays to the partial object
    for ifreq, freq_idx in enumerate(freq_inds):
        for ipol, pol_idx in enumerate(pol_inds):
            partial_uvh5.data_array[:, :, freq_idx, pol_idx] = data[:, :, ifreq, ipol]
            partial_uvh5.flag_array[:, :, freq_idx, pol_idx] = flags[:, :, ifreq, ipol]
            partial_uvh5.nsample_array[:, :, freq_idx, pol_idx] = nsamples[:, :, ifreq, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # test irregularly spaced everything
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros(partial_uvh5, partial_testfile)

    # make a mostly empty object in memory to match what we'll write to disk
    partial_uvh5.data_array = np.zeros_like(full_uvh5.data_array, dtype=np.complex64)
    partial_uvh5.flag_array = np.zeros_like(full_uvh5.flag_array, dtype=np.bool)
    partial_uvh5.nsample_array = np.zeros_like(full_uvh5.nsample_array, dtype=np.float32)

    # define blts and freqs
    blt_inds = [0, 1, 2, 7]
    freq_inds = [0, 2, 3, 4]
    pol_inds = [0, 1, 3]
    data_shape = (len(blt_inds), 1, len(freq_inds), len(pol_inds))
    data = np.zeros(data_shape, dtype=np.complex64)
    flags = np.zeros(data_shape, dtype=np.bool)
    nsamples = np.zeros(data_shape, dtype=np.float32)
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            for ipol, pol_idx in enumerate(pol_inds):
                data[iblt, :, ifreq, ipol] = full_uvh5.data_array[blt_idx, :, freq_idx, pol_idx]
                flags[iblt, :, ifreq, ipol] = full_uvh5.flag_array[blt_idx, :, freq_idx, pol_idx]
                nsamples[iblt, :, ifreq, ipol] = full_uvh5.nsample_array[blt_idx, :, freq_idx, pol_idx]
    uvtest.checkWarnings(partial_uvh5.write_uvh5_part, [partial_testfile, data, flags, nsamples],
                         {'blt_inds': blt_inds, 'freq_chans': freq_inds,
                          'polarizations': full_uvh5.polarization_array[pol_inds]},
                         nwarnings=2, message=['Selected frequencies are not evenly spaced',
                                               'Selected polarization values are not evenly spaced'])

    # also write the arrays to the partial object
    for iblt, blt_idx in enumerate(blt_inds):
        for ifreq, freq_idx in enumerate(freq_inds):
            for ipol, pol_idx in enumerate(pol_inds):
                partial_uvh5.data_array[blt_idx, :, freq_idx, pol_idx] = data[iblt, :, ifreq, ipol]
                partial_uvh5.flag_array[blt_idx, :, freq_idx, pol_idx] = flags[iblt, :, ifreq, ipol]
                partial_uvh5.nsample_array[blt_idx, :, freq_idx, pol_idx] = nsamples[iblt, :, ifreq, ipol]

    # read in the file and make sure it matches
    partial_uvh5_file = UVData()
    partial_uvh5_file.read(partial_testfile)
    nt.assert_equal(partial_uvh5_file, partial_uvh5)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5PartialWriteErrors():
    """
    Test errors in uvh5_write_part method
    """
    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(full_uvh5.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    full_uvh5.write_uvh5(testfile, clobber=True)
    full_uvh5.read(testfile)

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
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
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

    # initialize a file on disk, and pass in a different object so check_header fails
    empty_uvd = UVData()
    nt.assert_raises(AssertionError, empty_uvd.write_uvh5_part, partial_testfile, data,
                     flags, nsamples, bls=key)

    # clean up
    os.remove(testfile)
    os.remove(partial_testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5InitializeFile():
    """
    Test initializing a UVH5 file on disk
    """
    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(full_uvh5.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    full_uvh5.write_uvh5(testfile, clobber=True)
    full_uvh5.read(testfile)
    full_uvh5.data_array = None
    full_uvh5.flag_array = None
    full_uvh5.nsample_array = None

    # initialize file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # read it in and make sure that the metadata matches the original
    partial_uvh5.read(partial_testfile, read_data=False)
    nt.assert_equal(partial_uvh5, full_uvh5)

    # add options for compression
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True, data_compression="lzf",
                                      flags_compression=None, nsample_compression=None)
    partial_uvh5.read(partial_testfile, read_data=False)
    nt.assert_equal(partial_uvh5, full_uvh5)

    # check that an error is raised then file exists and clobber is False
    nt.assert_raises(ValueError, partial_uvh5.initialize_uvh5_file, partial_testfile, clobber=False)

    # clean up
    os.remove(testfile)
    os.remove(partial_testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5SingleIntegrationTime():
    """
    Check backwards compatibility warning for files with a single integration time
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv_in.write_uvh5(testfile, clobber=True)

    # change integration_time in file to be a single number
    with h5py.File(testfile, 'r+') as f:
        int_time = f['/Header/integration_time'].value[0]
        del(f['/Header/integration_time'])
        f['/Header/integration_time'] = int_time
    uvtest.checkWarnings(uv_out.read_uvh5, [testfile], message='outtest_uvfits.uvh5 appears to be an old uvh5 format')
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5LstArray():
    """
    Test different cases of the lst_array
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv_in.write_uvh5(testfile, clobber=True)

    # remove lst_array from file; check that it's correctly computed on read
    with h5py.File(testfile, 'r+') as f:
        del(f['/Header/lst_array'])
    uv_out.read_uvh5(testfile)
    nt.assert_equal(uv_in, uv_out)

    # now change what's in the file and make sure a warning is raised
    uv_in.write_uvh5(testfile, clobber=True)
    with h5py.File(testfile, 'r+') as f:
        lst_array = f['/Header/lst_array'].value
        del(f['/Header/lst_array'])
        f['/Header/lst_array'] = 2 * lst_array
    uvtest.checkWarnings(uv_out.read_uvh5, [testfile],
                         message='LST values stored in outtest_uvfits.uvh5 are not self-consistent')
    uv_out.lst_array = lst_array
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5StringBackCompat():
    """
    Test backwards compatibility handling of strings
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv_in.write_uvh5(testfile, clobber=True)

    # write a string-type data as-is, without casting to np.string_
    with h5py.File(testfile, 'r+') as f:
        del(f['Header/instrument'])
        f['Header/instrument'] = uv_in.instrument
    uvtest.checkWarnings(uv_out.read_uvh5, [testfile],
                         message='Strings in metadata of outtest_uvfits.uvh5 are not the correct type')
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5ReadHeaderSpecialCases():
    """
    Test special cases values when reading files
    """
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits.uvh5')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv_in.write_uvh5(testfile, clobber=True)

    # change some of the metadata to trip certain if/else clauses
    with h5py.File(testfile, 'r+') as f:
        del(f['Header/history'])
        del(f['Header/vis_units'])
        del(f['Header/phase_type'])
        f['Header/history'] = np.string_('blank history')
        f['Header/phase_type'] = np.string_('blah')
    uv_out.read_uvh5(testfile)

    # make input and output values match now
    uv_in.history = uv_out.history
    uv_in.set_unknown_phase_type()
    uv_in.phase_center_ra = None
    uv_in.phase_center_dec = None
    uv_in.phase_center_epoch = None
    uv_in.vis_units = 'UNCALIB'
    nt.assert_equal(uv_in, uv_out)

    # clean up
    os.remove(testfile)

    return
