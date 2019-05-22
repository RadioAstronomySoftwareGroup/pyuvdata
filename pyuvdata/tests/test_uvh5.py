# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for HDF5 object

"""
from __future__ import absolute_import, division, print_function

import os
import six
import copy
import numpy as np
import pytest
from astropy.time import Time

from pyuvdata import UVData
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest

try:
    import h5py
    from pyuvdata import uvh5
    from pyuvdata.uvh5 import _hera_corr_dtype
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
    assert uv_in == uv_out

    # also test round-tripping phased data
    uv_in.phase_to_time(Time(np.mean(uv_in.time_array), format='jd'))
    uv_in.write_uvh5(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out

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
    uvtest.checkWarnings(uv_out.read, [testfile], message='Telescope EVLA is not')
    assert uv_in == uv_out

    # also test writing double-precision data_array
    uv_in.data_array = uv_in.data_array.astype(np.complex128)
    uv_in.write_uvh5(testfile, clobber=True)
    uvtest.checkWarnings(uv_out.read, [testfile], message='Telescope EVLA is not')
    assert uv_in == uv_out

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
    pytest.raises(IOError, uv_in.read_uvh5, fake_file)
    pytest.raises(ValueError, uv_in.read_uvh5, ['list of', 'fake files'], read_data=False)

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

    # assert IOError if file exists
    pytest.raises(IOError, uv_in.write_uvh5, testfile, clobber=False)

    # use clobber=True to write out anyway
    uv_in.write_uvh5(testfile, clobber=True)
    uvtest.checkWarnings(uv_out.read, [testfile], message='Telescope EVLA is not')
    assert uv_in == uv_out

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

    # reorder_blts
    uv_in.reorder_blts()

    # write out and read back in
    uv_in.write_uvh5(testfile, clobber=True)
    uvtest.checkWarnings(uv_out.read, [testfile], message='Telescope EVLA is not')
    assert uv_in == uv_out

    # test with blt_order = bda as well (single entry in tuple)
    uv_in.reorder_blts(order='bda')

    uv_in.write_uvh5(testfile, clobber=True)
    uvtest.checkWarnings(uv_out.read, [testfile], message='Telescope EVLA is not')
    assert uv_in == uv_out

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
    uvtest.checkWarnings(uv_out.read, [testfile], message='Telescope EVLA is not')
    assert uv_in == uv_out

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
    uvtest.checkWarnings(uv1.read, [[testfile1, testfile2]], nwarnings=2,
                         message='Telescope EVLA is not')
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis using'
                                    ' pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # clean up
    os.remove(testfile1)
    os.remove(testfile2)

    return


@uvtest.skipIf_no_h5py
def test_UVH5ReadMultiple_files_axis():
    """
    Test reading multiple uvh5 files with setting axis
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
    uvtest.checkWarnings(uv1.read, [[testfile1, testfile2]], {'axis': 'freq'},
                         nwarnings=2, message='Telescope EVLA is not')
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis using'
                                    ' pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

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
    uvh5_uv.telescope_name = 'PAPER'
    uvh5_uv.write_uvh5(testfile, clobber=True)

    # select on antennas
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep)
    assert uvh5_uv == uvh5_uv2

    # select on frequency channels
    chans_to_keep = np.arange(12, 22)
    uvh5_uv.read(testfile, freq_chans=chans_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(freq_chans=chans_to_keep)
    assert uvh5_uv == uvh5_uv2

    # select on pols
    pols_to_keep = [-1, -2]
    uvh5_uv.read(testfile, polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # select on read using time_range
    unique_times = np.unique(uvh5_uv.time_array)
    uvtest.checkWarnings(uvh5_uv.read, [testfile],
                         {'time_range': [unique_times[0], unique_times[1]]},
                         message=['Warning: "time_range" keyword is set'])
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(times=unique_times[0:2])
    assert uvh5_uv == uvh5_uv2

    # now test selecting on multiple axes
    # frequencies first
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # baselines first
    ants_to_keep = np.array([0, 1])
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # polarizations first
    ants_to_keep = np.array([0, 1, 2, 3, 6, 7, 8, 11, 14, 18, 19, 20, 21, 22])
    chans_to_keep = np.arange(12, 64)
    uvh5_uv.read(testfile, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(testfile)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

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
    full_uvh5.telescope_name = "PAPER"
    # cut down the file size to decrease testing time
    full_uvh5.select(antenna_nums=[3, 7, 24])
    full_uvh5.lst_array = uvutils.get_lst_for_time(full_uvh5.time_array,
                                                   *full_uvh5.telescope_location_lat_lon_alt_degrees)
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
    assert full_uvh5 == partial_uvh5

    # test add_to_history
    key = antpairpols[0]
    data = full_uvh5.get_data(key, squeeze='none')
    flags = full_uvh5.get_flags(key, squeeze='none')
    nsamples = full_uvh5.get_nsamples(key, squeeze='none')
    partial_uvh5.write_uvh5_part(partial_testfile, data, flags, nsamples,
                                 bls=key, add_to_history="foo")
    partial_uvh5.read(partial_testfile, read_data=False)
    assert 'foo' in partial_uvh5.history

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
    assert full_uvh5 == partial_uvh5

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
    assert full_uvh5, partial_uvh5

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
    assert full_uvh5 == partial_uvh5

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
    full_uvh5.telescope_name = "PAPER"
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
    assert partial_uvh5_file == partial_uvh5

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
    assert partial_uvh5_file == partial_uvh5

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
    assert partial_uvh5_file == partial_uvh5

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
    assert partial_uvh5_file == partial_uvh5

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
    assert partial_uvh5_file == partial_uvh5

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
    assert partial_uvh5_file == partial_uvh5

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
    assert partial_uvh5_file == partial_uvh5

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
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(testfile)
    os.remove(partial_testfile)

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
    full_uvh5.telescope_name = "PAPER"
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
    pytest.raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data,
                  flags, nsamples, bls=key)

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True)

    # pass in arrays that are different sizes
    pytest.raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data,
                  flags[:, :, :, 0], nsamples, bls=key)
    pytest.raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data,
                  flags, nsamples[:, :, :, 0], bls=key)

    # pass in arrays that are the same size, but don't match expected shape
    pytest.raises(AssertionError, partial_uvh5.write_uvh5_part, partial_testfile, data[:, :, :, 0],
                  flags[:, :, :, 0], nsamples[:, :, :, 0])

    # initialize a file on disk, and pass in a different object so check_header fails
    empty_uvd = UVData()
    pytest.raises(AssertionError, empty_uvd.write_uvh5_part, partial_testfile, data,
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
    full_uvh5.telescope_name = "PAPER"
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
    assert partial_uvh5 == full_uvh5

    # check that IOError is raised then when clobber == False
    pytest.raises(IOError, partial_uvh5.initialize_uvh5_file, partial_testfile, clobber=False)

    # add options for compression
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True, data_compression="lzf",
                                      flags_compression=None, nsample_compression=None)
    partial_uvh5.read(partial_testfile, read_data=False)
    assert partial_uvh5 == full_uvh5

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
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # change integration_time in file to be a single number
    with h5py.File(testfile, 'r+') as f:
        int_time = f['/Header/integration_time'][0]
        del(f['/Header/integration_time'])
        f['/Header/integration_time'] = int_time
    uvtest.checkWarnings(uv_out.read_uvh5, [testfile],
                         message='outtest_uvfits.uvh5 appears to be an old uvh5 format',
                         category=DeprecationWarning)
    assert uv_in == uv_out

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
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # remove lst_array from file; check that it's correctly computed on read
    with h5py.File(testfile, 'r+') as f:
        del(f['/Header/lst_array'])
    uv_out.read_uvh5(testfile)
    assert uv_in == uv_out

    # now change what's in the file and make sure a warning is raised
    uv_in.write_uvh5(testfile, clobber=True)
    with h5py.File(testfile, 'r+') as f:
        lst_array = f['/Header/lst_array'][:]
        del(f['/Header/lst_array'])
        f['/Header/lst_array'] = 2 * lst_array
    uvtest.checkWarnings(uv_out.read_uvh5, [testfile],
                         message='LST values stored in outtest_uvfits.uvh5 are not self-consistent')
    uv_out.lst_array = lst_array
    assert uv_in == uv_out

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
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # write a string-type data as-is, without casting to np.string_
    with h5py.File(testfile, 'r+') as f:
        del(f['Header/instrument'])
        f['Header/instrument'] = uv_in.instrument
    uvtest.checkWarnings(uv_out.read_uvh5, [testfile],
                         message='Strings in metadata of outtest_uvfits.uvh5 are not the correct type',
                         category=DeprecationWarning)
    assert uv_in == uv_out

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
    uv_in.telescope_name = "PAPER"
    uv_in.write_uvh5(testfile, clobber=True)

    # change some of the metadata to trip certain if/else clauses
    with h5py.File(testfile, 'r+') as f:
        del(f['Header/history'])
        del(f['Header/vis_units'])
        del(f['Header/phase_type'])
        del(f['Header/latitude'])
        del(f['Header/longitude'])
        f['Header/history'] = np.string_('blank history')
        f['Header/phase_type'] = np.string_('blah')
        f['Header/latitude'] = uv_in.telescope_location_lat_lon_alt[0]
        f['Header/longitude'] = uv_in.telescope_location_lat_lon_alt[1]
    uvtest.checkWarnings(uv_out.read_uvh5, [testfile], category=DeprecationWarning,
                         message='It seems that the latitude and longitude are in radians')

    # make input and output values match now
    uv_in.history = uv_out.history
    uv_in.set_unknown_phase_type()
    uv_in.phase_center_ra = None
    uv_in.phase_center_dec = None
    uv_in.phase_center_epoch = None
    uv_in.vis_units = 'UNCALIB'
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5ReadInts():
    """
    Test reading visibility data saved as integers
    """
    uv_in = UVData()
    uv_out = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    uv_in.read_uvh5(uvh5_file)
    uv_in.write_uvh5(testfile, clobber=True)

    # read it back in to make sure data is the same
    uv_out.read_uvh5(testfile)
    assert uv_in == uv_out

    # now read in as np.complex128
    uv_in.read_uvh5(uvh5_file, data_array_dtype=np.complex128)
    assert uv_in == uv_out
    assert uv_in.data_array.dtype == np.dtype(np.complex128)

    # clean up
    os.remove(testfile)

    # raise error
    pytest.raises(ValueError, uv_in.read_uvh5, uvh5_file, data_array_dtype=np.int32)

    return


@uvtest.skipIf_no_h5py
def test_UVH5WriteInts():
    """
    Test writing visibility data as integers
    """
    uv_in = UVData()
    uv_out = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest.uvh5')
    uv_in.read_uvh5(uvh5_file)
    uv_in.write_uvh5(testfile, clobber=True, data_write_dtype=_hera_corr_dtype)

    # read it back in to make sure data is the same
    uv_out.read_uvh5(testfile)
    assert uv_in == uv_out

    # also check that the datatype on disk is the right type
    with h5py.File(testfile, 'r') as f:
        visdata_dtype = f['Data/visdata'].dtype
        assert 'r' in visdata_dtype.names
        assert 'i' in visdata_dtype.names
        assert visdata_dtype['r'].kind == 'i'
        assert visdata_dtype['i'].kind == 'i'

    # clean up
    os.remove(testfile)

    return


@uvtest.skipIf_no_h5py
def test_UVH5PartialReadInts():
    """
    Test reading in only part of a dataset from disk
    """
    uvh5_uv = UVData()
    uvh5_uv2 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')

    # select on antennas
    ants_to_keep = np.array([0, 1])
    uvh5_uv.read(uvh5_file, antenna_nums=ants_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(antenna_nums=ants_to_keep)
    assert uvh5_uv == uvh5_uv2

    # select on frequency channels
    chans_to_keep = np.arange(12, 22)
    uvh5_uv.read(uvh5_file, freq_chans=chans_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(freq_chans=chans_to_keep)
    assert uvh5_uv == uvh5_uv2

    # select on pols
    pols_to_keep = [-5, -6]
    uvh5_uv.read(uvh5_file, polarizations=pols_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # select on read using time_range
    unique_times = np.unique(uvh5_uv.time_array)
    uvtest.checkWarnings(uvh5_uv.read, [uvh5_file],
                         {'time_range': [unique_times[0], unique_times[1]]},
                         message=['Warning: "time_range" keyword is set'])
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(times=unique_times[0:2])
    assert uvh5_uv == uvh5_uv2

    # now test selecting on multiple axes
    # frequencies first
    uvh5_uv.read(uvh5_file, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # baselines first
    ants_to_keep = np.array([0, 1])
    uvh5_uv.read(uvh5_file, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    # polarizations first
    ants_to_keep = np.array([0, 1])
    chans_to_keep = np.arange(12, 64)
    uvh5_uv.read(uvh5_file, antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                 polarizations=pols_to_keep)
    uvh5_uv2.read(uvh5_file)
    uvh5_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                    polarizations=pols_to_keep)
    assert uvh5_uv == uvh5_uv2

    return


@uvtest.skipIf_no_h5py
def test_UVH5PartialWriteInts():
    """
    Test writing an entire UVH5 file in pieces with integer outputs
    """
    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')
    full_uvh5.read_uvh5(uvh5_file)

    # delete data arrays in partial file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True,
                                      data_write_dtype=_hera_corr_dtype)

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
    assert full_uvh5 == partial_uvh5

    # start over, and write frequencies
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True,
                                      data_write_dtype=_hera_corr_dtype)
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
    assert full_uvh5 == partial_uvh5

    # start over, write chunks of blts
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True,
                                      data_write_dtype=_hera_corr_dtype)
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
    assert full_uvh5 == partial_uvh5

    # start over, write groups of pols
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_uvh5.initialize_uvh5_file(partial_testfile, clobber=True,
                                      data_write_dtype=_hera_corr_dtype)
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
    assert full_uvh5 == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


@uvtest.skipIf_no_h5py
def test_read_complex_astype():
    # make a testfile with a test dataset
    test_file = os.path.join(DATA_PATH, 'test', 'test_file.h5')
    test_data_shape = (2, 3, 4, 5)
    test_data = np.zeros(test_data_shape, dtype=np.complex64)
    test_data.real = 1.
    test_data.imag = 2.
    with h5py.File(test_file, 'w') as f:
        dgrp = f.create_group('Data')
        dset = dgrp.create_dataset('testdata', test_data_shape,
                                   dtype=_hera_corr_dtype)
        with dset.astype(_hera_corr_dtype):
            dset[:, :, :, :, 'r'] = test_data.real
            dset[:, :, :, :, 'i'] = test_data.imag

    # test that reading the data back in works as expected
    indices = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
    with h5py.File(test_file, 'r') as f:
        dset = f['Data/testdata']
        file_data = uvh5._read_complex_astype(dset, indices, np.complex64)

    assert np.allclose(file_data, test_data)

    # test errors
    # test passing in a forbidden output datatype
    with h5py.File(test_file, 'r') as f:
        dset = f['Data/testdata']
        pytest.raises(ValueError, uvh5._read_complex_astype, dset, indices, np.int32)

    # clean up
    os.remove(test_file)

    return


@uvtest.skipIf_no_h5py
def test_write_complex_astype():
    # make sure we can write data out
    test_file = os.path.join(DATA_PATH, 'test', 'test_file.h5')
    test_data_shape = (2, 3, 4, 5)
    test_data = np.zeros(test_data_shape, dtype=np.complex64)
    test_data.real = 1.
    test_data.imag = 2.
    with h5py.File(test_file, 'w') as f:
        dgrp = f.create_group('Data')
        dset = dgrp.create_dataset('testdata', test_data_shape,
                                   dtype=_hera_corr_dtype)
        inds = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
        uvh5._write_complex_astype(test_data, dset, inds)

    # read the data back in to confirm it's right
    with h5py.File(test_file, 'r') as f:
        dset = f['Data/testdata']
        file_data = np.zeros(test_data_shape, dtype=np.complex64)
        with dset.astype(_hera_corr_dtype):
            file_data.real = dset['r'][:, :, :, :]
            file_data.imag = dset['i'][:, :, :, :]

    assert np.allclose(file_data, test_data)

    return


@uvtest.skipIf_no_h5py
def test_check_uvh5_dtype_errors():
    # test passing in something that's not a dtype
    pytest.raises(ValueError, uvh5._check_uvh5_dtype, 'hi')

    # test using a dtype with bad field names
    dtype = np.dtype([('a', '<i4'), ('b', '<i4')])
    pytest.raises(ValueError, uvh5._check_uvh5_dtype, dtype)

    # test having different types for 'r' and 'i' fields
    dtype = np.dtype([('r', '<i4'), ('i', '<f4')])
    pytest.raises(ValueError, uvh5._check_uvh5_dtype, dtype)

    return


@uvtest.skipIf_no_h5py
def test_UVH5PartialWriteIntsIrregular():
    """
    Test writing a uvh5 file using irregular intervals
    """
    def initialize_with_zeros_ints(uvd, filename):
        """
        Initialize a file with all zeros for data arrays
        """
        uvd.initialize_uvh5_file(filename, clobber=True, data_write_dtype=_hera_corr_dtype)
        data_shape = (uvd.Nblts, 1, uvd.Nfreqs, uvd.Npols)
        data = np.zeros(data_shape, dtype=np.complex64)
        flags = np.zeros(data_shape, dtype=np.bool)
        nsamples = np.zeros(data_shape, dtype=np.float32)
        with h5py.File(filename, 'r+') as f:
            dgrp = f['/Data']
            data_dset = dgrp['visdata']
            flags_dset = dgrp['flags']
            nsample_dset = dgrp['nsamples']
            with data_dset.astype(_hera_corr_dtype):
                data_dset[:, :, :, :, 'r'] = data.real
                data_dset[:, :, :, :, 'i'] = data.imag
            flags_dset = flags
            nsample_dset = nsamples
        return

    full_uvh5 = UVData()
    partial_uvh5 = UVData()
    uvh5_file = os.path.join(DATA_PATH, 'zen.2458432.34569.uvh5')
    full_uvh5.read(uvh5_file)

    # delete data arrays in partial file
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # do it again, with a single frequency
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # do it again, with a single polarization
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # test irregularly spaced blts and freqs
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # test irregularly spaced freqs and pols
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # test irregularly spaced blts and pols
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # test irregularly spaced freqs and pols
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # test irregularly spaced everything
    # reinitialize
    partial_uvh5 = copy.deepcopy(full_uvh5)
    partial_uvh5.data_array = None
    partial_uvh5.flag_array = None
    partial_uvh5.nsample_array = None

    # initialize file on disk
    partial_testfile = os.path.join(DATA_PATH, 'test', 'outtest_partial.uvh5')
    initialize_with_zeros_ints(partial_uvh5, partial_testfile)

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
    assert partial_uvh5_file == partial_uvh5

    # clean up
    os.remove(partial_testfile)

    return


@uvtest.skipIf_no_h5py
@pytest.mark.skipif(not six.PY3, reason="Skipping. This test is only relevant in python3.")
def test_antenna_names_not_list():
    """Test if antenna_names is cast to an array, dimensions are preserved in np.string_ call during uvh5 write."""
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_uvfits_ant_names.uvh5')
    uvtest.checkWarnings(uv_in.read_uvfits, [uvfits_file], message='Telescope EVLA is not')

    # simulate a user defining antenna names as an array of unicode
    uv_in.antenna_names = np.array(uv_in.antenna_names, dtype='U')

    uv_in.write_uvh5(testfile, clobber=True)
    uvtest.checkWarnings(uv_out.read, [testfile], message='Telescope EVLA is not')

    # recast as list since antenna names should be a list and will be cast as list on read
    uv_in.antenna_names = uv_in.antenna_names.tolist()
    assert uv_in == uv_out

    # clean up
    os.remove(testfile)

    return
