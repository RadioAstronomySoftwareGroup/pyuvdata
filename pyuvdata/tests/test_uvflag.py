# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import division
import unittest
import nose.tools as nt
import os
import numpy as np
import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata import UVCal
from pyuvdata.data import DATA_PATH
from pyuvdata import UVFlag
from pyuvdata.uvflag import lst_from_uv
from pyuvdata.uvflag import flags2waterfall
from pyuvdata.uvflag import and_rows_cols
from pyuvdata import version as uvversion
import shutil
import copy
import six


test_d_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
test_c_file = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
test_f_file = test_d_file + '.testuvflag.h5'
test_outfile = os.path.join(DATA_PATH, 'test', 'outtest_uvflag.h5')

pyuvdata_version_str = uvversion.version + '.'
if uvversion.git_hash is not '':
    pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin
                             + '.  Git hash: ' + uvversion.git_hash
                             + '.  Git branch: ' + uvversion.git_branch
                             + '.  Git description: ' + uvversion.git_description + '.')


def test_init_UVData():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, history='I made a UVFlag object', label='test')
    nt.assert_true(uvf.metric_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'baseline')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    nt.assert_true(np.all(uvf.lst_array == uv.lst_array))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.ant_1_array == uv.ant_1_array))
    nt.assert_true(np.all(uvf.ant_2_array == uv.ant_2_array))
    nt.assert_true('I made a UVFlag object' in uvf.history)
    nt.assert_true('Flag object with type "baseline"' in uvf.history)
    nt.assert_true(pyuvdata_version_str in uvf.history)
    nt.assert_true(uvf.label == 'test')


def test_init_UVData_copy_flags():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = uvtest.checkWarnings(UVFlag, [uv], {'copy_flags': True, 'mode': 'metric'},
                               nwarnings=1, message='Copying flags to type=="baseline"')
    nt.assert_false(hasattr(uvf, 'metric_array'))  # Should be flag due to copy flags
    nt.assert_true(np.array_equal(uvf.flag_array, uv.flag_array))
    nt.assert_true(uvf.weights_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'baseline')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    nt.assert_true(np.all(uvf.lst_array == uv.lst_array))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.ant_1_array == uv.ant_1_array))
    nt.assert_true(np.all(uvf.ant_2_array == uv.ant_2_array))
    nt.assert_true('Flag object with type "baseline"' in uvf.history)
    nt.assert_true(pyuvdata_version_str in uvf.history)


def test_init_UVCal():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    nt.assert_true(uvf.metric_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'antenna')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    lst = lst_from_uv(uvc)
    nt.assert_true(np.all(uvf.lst_array == lst))
    nt.assert_true(np.all(uvf.freq_array == uvc.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uvc.jones_array))
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true('Flag object with type "antenna"' in uvf.history)
    nt.assert_true(pyuvdata_version_str in uvf.history)


def test_init_cal_copy_flags():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = uvtest.checkWarnings(UVFlag, [uv], {'copy_flags': True, 'mode': 'metric'},
                               nwarnings=1, message='Copying flags to type=="antenna"')
    nt.assert_false(hasattr(uvf, 'metric_array'))  # Should be flag due to copy flags
    nt.assert_true(np.array_equal(uvf.flag_array, uv.flag_array))
    nt.assert_true(uvf.weights_array.shape == uv.flag_array.shape)
    nt.assert_true(uvf.type == 'antenna')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true(pyuvdata_version_str in uvf.history)


def test_init_waterfall_uvd():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, waterfall=True)
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols))
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols))
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.lst_array == np.unique(uv.lst_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(pyuvdata_version_str in uvf.history)


def test_init_waterfall_uvc():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, waterfall=True)
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(pyuvdata_version_str in uvf.history)


def test_init_waterfall_flag():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, waterfall=True, mode='flag')
    nt.assert_true(uvf.flag_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(not np.any(uvf.flag_array))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(pyuvdata_version_str in uvf.history)


def test_init_waterfall_copy_flags():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = uvtest.checkWarnings(UVFlag, [uv], {'copy_flags': True, 'mode': 'flag', 'waterfall': True},
                               nwarnings=1, message='Copying flags into waterfall')
    nt.assert_false(hasattr(uvf, 'flag_array'))  # Should be metric due to copy flags
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(pyuvdata_version_str in uvf.history)


@uvtest.skipIf_no_h5py
def test_read_write_loop():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, label='test')
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + pyuvdata_version_str
    uvf.history += ' Read by ' + pyuvdata_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


@uvtest.skipIf_no_h5py
def test_read_write_ant():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, mode='flag', label='test')
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + pyuvdata_version_str
    uvf.history += ' Read by ' + pyuvdata_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


@uvtest.skipIf_no_h5py
def test_read_write_nocompress():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, label='test')
    uvf.write(test_outfile, clobber=True, data_compression=None)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + pyuvdata_version_str
    uvf.history += ' Read by ' + pyuvdata_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


@uvtest.skipIf_no_h5py
def test_read_write_nocompress_flag():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, mode='flag', label='test')
    uvf.write(test_outfile, clobber=True, data_compression=None)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + pyuvdata_version_str
    uvf.history += ' Read by ' + pyuvdata_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


@uvtest.skipIf_no_h5py
def test_init_list():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uv.time_array -= 1
    uvf = UVFlag([uv, test_f_file])
    uvf1 = UVFlag(uv)
    uvf2 = UVFlag(test_f_file)
    nt.assert_true(np.array_equal(np.concatenate((uvf1.metric_array, uvf2.metric_array), axis=0),
                                  uvf.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.weights_array, uvf2.weights_array), axis=0),
                                  uvf.weights_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.time_array, uvf2.time_array)),
                                  uvf.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.baseline_array, uvf2.baseline_array)),
                                  uvf.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_1_array, uvf2.ant_1_array)),
                                  uvf.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_2_array, uvf2.ant_2_array)),
                                  uvf.ant_2_array))
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))


@uvtest.skipIf_no_h5py
def test_read_list():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uv.time_array -= 1
    uvf = UVFlag(uv)
    uvf.write(test_outfile, clobber=True)
    uvf.read([test_outfile, test_f_file])
    uvf1 = UVFlag(uv)
    uvf2 = UVFlag(test_f_file)
    nt.assert_true(np.array_equal(np.concatenate((uvf1.metric_array, uvf2.metric_array), axis=0),
                                  uvf.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.weights_array, uvf2.weights_array), axis=0),
                                  uvf.weights_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.time_array, uvf2.time_array)),
                                  uvf.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.baseline_array, uvf2.baseline_array)),
                                  uvf.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_1_array, uvf2.ant_1_array)),
                                  uvf.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_2_array, uvf2.ant_2_array)),
                                  uvf.ant_2_array))
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))


@uvtest.skipIf_no_h5py
def test_read_error():
    nt.assert_raises(IOError, UVFlag, 'foo')


@uvtest.skipIf_no_h5py
def test_read_change_type():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.write(test_outfile, clobber=True)
    nt.assert_true(hasattr(uvf, 'ant_array'))
    uvf.read(test_f_file)
    nt.assert_false(hasattr(uvf, 'ant_array'))
    nt.assert_true(hasattr(uvf, 'baseline_array'))
    nt.assert_true(hasattr(uvf, 'ant_1_array'))
    nt.assert_true(hasattr(uvf, 'ant_2_array'))
    uvf.read(test_outfile)
    nt.assert_true(hasattr(uvf, 'ant_array'))
    nt.assert_false(hasattr(uvf, 'baseline_array'))
    nt.assert_false(hasattr(uvf, 'ant_1_array'))
    nt.assert_false(hasattr(uvf, 'ant_2_array'))


@uvtest.skipIf_no_h5py
def test_read_change_mode():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, mode='flag')
    nt.assert_true(hasattr(uvf, 'flag_array'))
    nt.assert_false(hasattr(uvf, 'metric_array'))
    uvf.write(test_outfile, clobber=True)
    uvf.read(test_f_file)
    nt.assert_true(hasattr(uvf, 'metric_array'))
    nt.assert_false(hasattr(uvf, 'flag_array'))
    uvf.read(test_outfile)
    nt.assert_true(hasattr(uvf, 'flag_array'))
    nt.assert_false(hasattr(uvf, 'metric_array'))


@uvtest.skipIf_no_h5py
def test_write_no_clobber():
    uvf = UVFlag(test_f_file)
    nt.assert_raises(ValueError, uvf.write, test_f_file)


def test_lst_from_uv():
    uv = UVData()
    uv.read_miriad(test_d_file)
    lst_array = lst_from_uv(uv)
    nt.assert_true(np.allclose(uv.lst_array, lst_array))


def test_lst_from_uv_error():
    nt.assert_raises(ValueError, lst_from_uv, 4)


@uvtest.skipIf_no_h5py
def test_add():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_1_array, uv2.ant_1_array)),
                                  uv3.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_2_array, uv2.ant_2_array)),
                                  uv3.ant_2_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)),
                                  uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=0),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along time axis with ' + pyuvdata_version_str in uv3.history)


@uvtest.skipIf_no_h5py
def test_add_baseline():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.baseline_array += 100  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='baseline')
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_1_array, uv2.ant_1_array)),
                                  uv3.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_2_array, uv2.ant_2_array)),
                                  uv3.ant_2_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)),
                                  uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=0),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along baseline axis with ' + pyuvdata_version_str in uv3.history)


def test_add_antenna():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv1 = UVFlag(uvc)
    uv2 = copy.deepcopy(uv1)
    uv2.ant_array += 100  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='antenna')
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_array, uv2.ant_array)),
                                  uv3.ant_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=0),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.lst_array, uv3.lst_array))
    nt.assert_true(uv3.type == 'antenna')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along antenna axis with ' + pyuvdata_version_str in uv3.history)


@uvtest.skipIf_no_h5py
def test_add_frequency():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.freq_array += 1e4  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='frequency')
    nt.assert_true(np.array_equal(np.concatenate((uv1.freq_array, uv2.freq_array)),
                                  uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.baseline_array, uv3.baseline_array))
    nt.assert_true(np.array_equal(uv1.ant_1_array, uv3.ant_1_array))
    nt.assert_true(np.array_equal(uv1.ant_2_array, uv3.ant_2_array))
    nt.assert_true(np.array_equal(uv1.lst_array, uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=2),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=2),
                                  uv3.weights_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along frequency axis with ' + pyuvdata_version_str in uv3.history)


@uvtest.skipIf_no_h5py
def test_add_pol():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.polarization_array += 1  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='polarization')
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.baseline_array, uv3.baseline_array))
    nt.assert_true(np.array_equal(uv1.ant_1_array, uv3.ant_1_array))
    nt.assert_true(np.array_equal(uv1.ant_2_array, uv3.ant_2_array))
    nt.assert_true(np.array_equal(uv1.lst_array, uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=3),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=3),
                                  uv3.weights_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(np.concatenate((uv1.polarization_array, uv2.polarization_array)),
                                  uv3.polarization_array))
    nt.assert_true('Data combined along polarization axis with ' + pyuvdata_version_str in uv3.history)


def test_add_flag():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uv1 = UVFlag(uv, mode='flag')
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_1_array, uv2.ant_1_array)),
                                  uv3.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_2_array, uv2.ant_2_array)),
                                  uv3.ant_2_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)),
                                  uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.flag_array, uv2.flag_array), axis=0),
                                  uv3.flag_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'flag')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along time axis with ' + pyuvdata_version_str in uv3.history)


def test_add_errors():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv1 = UVFlag(uv)
    # Mismatched classes
    nt.assert_raises(ValueError, uv1.__add__, 3)
    # Mismatched types
    uv2 = UVFlag(uvc)
    nt.assert_raises(ValueError, uv1.__add__, uv2)
    # Mismatched modes
    uv3 = UVFlag(uv, mode='flag')
    nt.assert_raises(ValueError, uv1.__add__, uv3)
    # Invalid axes
    nt.assert_raises(ValueError, uv1.__add__, uv1, axis='antenna')
    nt.assert_raises(ValueError, uv2.__add__, uv2, axis='baseline')


@uvtest.skipIf_no_h5py
def test_inplace_add():
    uv1a = UVFlag(test_f_file)
    uv1b = copy.deepcopy(uv1a)
    uv2 = copy.deepcopy(uv1a)
    uv2.time_array += 1
    uv1a += uv2
    nt.assert_true(uv1a.__eq__(uv1b + uv2))


@uvtest.skipIf_no_h5py
def test_clear_unused_attributes():
    uv = UVFlag(test_f_file)
    nt.assert_true(hasattr(uv, 'baseline_array') & hasattr(uv, 'ant_1_array')
                   & hasattr(uv, 'ant_2_array') & hasattr(uv, 'Nants_telescope'))
    uv.type = 'antenna'
    uv.clear_unused_attributes()
    nt.assert_false(hasattr(uv, 'baseline_array') | hasattr(uv, 'ant_1_array')
                    | hasattr(uv, 'ant_2_array') | hasattr(uv, 'Nants_telescope'))
    uv.mode = 'flag'
    nt.assert_true(hasattr(uv, 'metric_array'))
    uv.clear_unused_attributes()
    nt.assert_false(hasattr(uv, 'metric_array'))

    # Start over
    uv = UVFlag(test_f_file)
    uv.ant_array = np.array([4])
    uv.flag_array = np.array([5])
    uv.clear_unused_attributes()
    nt.assert_false(hasattr(uv, 'ant_array'))
    nt.assert_false(hasattr(uv, 'flag_array'))


@uvtest.skipIf_no_h5py
def test_not_equal():
    uvf1 = UVFlag(test_f_file)
    # different class
    nt.assert_false(uvf1.__eq__(5))
    # different mode
    uvf2 = uvf1.copy()
    uvf2.mode = 'flag'
    nt.assert_false(uvf1.__eq__(uvf2))
    # different type
    uvf2 = uvf1.copy()
    uvf2.type = 'antenna'
    nt.assert_false(uvf1.__eq__(uvf2))
    # array different
    uvf2 = uvf1.copy()
    uvf2.freq_array += 1
    nt.assert_false(uvf1.__eq__(uvf2))
    # history different
    uvf2 = uvf1.copy()
    uvf2.history += 'hello'
    nt.assert_false(uvf1.__eq__(uvf2, check_history=True))


@uvtest.skipIf_no_h5py
def test_to_waterfall_bl():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)


@uvtest.skipIf_no_h5py
def test_to_waterfall_bl_multi_pol():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis='pol')  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()  # Keep a copy to run with keep_pol=False
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)
    nt.assert_true(len(uvf.polarization_array) == 2)
    # Repeat with keep_pol=False
    uvf2.to_waterfall(keep_pol=False)
    nt.assert_true(uvf2.type == 'waterfall')
    nt.assert_true(uvf2.metric_array.shape == (len(uvf2.time_array), len(uvf.freq_array), 1))
    nt.assert_true(uvf2.weights_array.shape == uvf2.metric_array.shape)
    nt.assert_true(len(uvf2.polarization_array) == 1)
    nt.assert_true(uvf2.polarization_array[0] == np.string_(','.join(map(str, uvf.polarization_array))))


@uvtest.skipIf_no_h5py
def test_collapse_pol():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis='pol')  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()
    uvf2.collapse_pol()
    nt.assert_true(len(uvf2.polarization_array) == 1)
    nt.assert_true(uvf2.polarization_array[0] == np.string_(','.join(map(str, uvf.polarization_array))))
    nt.assert_equal(uvf2.mode, 'metric')
    nt.assert_true(hasattr(uvf2, 'metric_array'))
    nt.assert_false(hasattr(uvf2, 'flag_array'))

    # test writing it out and reading in to make sure polarization_array has correct type
    uvf2.write(test_outfile, clobber=True)
    uvf = UVFlag(test_outfile)
    nt.assert_equal(uvf, uvf2)
    os.remove(test_outfile)


@uvtest.skipIf_no_h5py
def test_collapse_pol_or():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis='pol')  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()
    uvf2.collapse_pol(method='or')
    nt.assert_true(len(uvf2.polarization_array) == 1)
    nt.assert_true(uvf2.polarization_array[0] == np.string_(','.join(map(str, uvf.polarization_array))))
    nt.assert_equal(uvf2.mode, 'flag')
    nt.assert_true(hasattr(uvf2, 'flag_array'))
    nt.assert_false(hasattr(uvf2, 'metric_array'))


@uvtest.skipIf_no_h5py
def test_collapse_single_pol():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvtest.checkWarnings(uvf.collapse_pol, [], {}, nwarnings=1,
                         message='Cannot collapse polarization')
    nt.assert_equal(uvf, uvf2)


@uvtest.skipIf_no_h5py
def test_collapse_pol_flag():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis='pol')  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()
    uvf2.collapse_pol()
    nt.assert_true(len(uvf2.polarization_array) == 1)
    nt.assert_true(uvf2.polarization_array[0] == np.string_(','.join(map(str, uvf.polarization_array))))
    nt.assert_equal(uvf2.mode, 'metric')
    nt.assert_true(hasattr(uvf2, 'metric_array'))
    nt.assert_false(hasattr(uvf2, 'flag_array'))


@uvtest.skipIf_no_h5py
def test_to_waterfall_bl_flags():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)
    nt.assert_true(len(uvf.lst_array) == len(uvf.time_array))


@uvtest.skipIf_no_h5py
def test_to_waterfall_bl_flags_or():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall(method='or')
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(uvf.flag_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                            len(uvf.polarization_array)))
    nt.assert_true(np.array_equal(uvf.weights_array, np.ones_like(uvf.flag_array, np.float)))
    nt.assert_true(len(uvf.lst_array) == len(uvf.time_array))
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.weights_array[0, 0, 0, 0] = 0.2
    uvtest.checkWarnings(uvf.to_waterfall, [], {'method': 'or'}, nwarnings=1,
                         message='Currently weights are')
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(uvf.flag_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                            len(uvf.polarization_array)))
    nt.assert_true(np.array_equal(uvf.weights_array, np.ones_like(uvf.flag_array, np.float)))
    nt.assert_true(len(uvf.lst_array) == len(uvf.time_array))


def test_to_waterfall_ant():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)
    nt.assert_true(len(uvf.lst_array) == len(uvf.time_array))


@uvtest.skipIf_no_h5py
def test_to_waterfall_waterfall():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    uvtest.checkWarnings(uvf.to_waterfall, [], {}, nwarnings=1,
                         message='This object is already a waterfall')


def test_to_baseline_flags():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.to_baseline(uv)
    nt.assert_equal(uvf.type, 'baseline')
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    times = np.unique(uvf.time_array)
    ntrue = 0.0
    ind = np.where(uvf.time_array == times[0])[0]
    ntrue += len(ind)
    nt.assert_true(np.all(uvf.flag_array[ind, 0, 10, 0]))
    ind = np.where(uvf.time_array == times[1])[0]
    ntrue += len(ind)
    nt.assert_true(np.all(uvf.flag_array[ind, 0, 15, 0]))
    nt.assert_true(uvf.flag_array.mean() == ntrue / uvf.flag_array.size)


def test_to_baseline_metric():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.to_baseline(uv)
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    times = np.unique(uvf.time_array)
    ind = np.where(uvf.time_array == times[0])[0]
    nt0 = len(ind)
    nt.assert_true(np.all(uvf.metric_array[ind, 0, 10, 0] == 3.2))
    ind = np.where(uvf.time_array == times[1])[0]
    nt1 = len(ind)
    nt.assert_true(np.all(uvf.metric_array[ind, 0, 15, 0] == 2.1))
    nt.assert_true(np.isclose(uvf.metric_array.mean(),
                              (3.2 * nt0 + 2.1 * nt1) / uvf.metric_array.size))


def test_baseline_to_baseline():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    uvf2 = uvf.copy()
    uvf.to_baseline(uv)
    nt.assert_equal(uvf, uvf2)


@uvtest.skipIf_no_h5py
def test_to_baseline_errors():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    nt.assert_raises(ValueError, uvf.to_baseline, 7.3)  # invalid matching object
    uvf = UVFlag(uvc)
    nt.assert_raises(ValueError, uvf.to_baseline, uv)  # Cannot pass in antenna type
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    uvf2 = uvf.copy()
    uvf.polarization_array[0] = -4
    nt.assert_raises(ValueError, uvf.to_baseline, uv)  # Mismatched pols
    uvf.__iadd__(uvf2, axis='polarization')
    nt.assert_raises(ValueError, uvf.to_baseline, uv)  # Mismatched pols, can't be forced


def test_to_baseline_force_pol():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.polarization_array[0] = -4  # Change pol, but force pol anyway
    uvf.to_baseline(uv, force_pol=True)
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    nt.assert_true(np.array_equal(uvf.polarization_array, uv.polarization_array))
    times = np.unique(uvf.time_array)
    ntrue = 0.0
    ind = np.where(uvf.time_array == times[0])[0]
    ntrue += len(ind)
    nt.assert_true(np.all(uvf.flag_array[ind, 0, 10, 0]))
    ind = np.where(uvf.time_array == times[1])[0]
    ntrue += len(ind)
    nt.assert_true(np.all(uvf.flag_array[ind, 0, 15, 0]))
    nt.assert_true(uvf.flag_array.mean() == ntrue / uvf.flag_array.size)


def test_to_baseline_metric_force_pol():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.polarization_array[0] = -4
    uvf.to_baseline(uv, force_pol=True)
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    nt.assert_true(np.array_equal(uvf.polarization_array, uv.polarization_array))
    times = np.unique(uvf.time_array)
    ind = np.where(uvf.time_array == times[0])[0]
    nt0 = len(ind)
    nt.assert_true(np.all(uvf.metric_array[ind, 0, 10, 0] == 3.2))
    ind = np.where(uvf.time_array == times[1])[0]
    nt1 = len(ind)
    nt.assert_true(np.all(uvf.metric_array[ind, 0, 15, 0] == 2.1))
    nt.assert_true(np.isclose(uvf.metric_array.mean(),
                              (3.2 * nt0 + 2.1 * nt1) / uvf.metric_array.size))


def test_to_antenna_flags():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.to_antenna(uvc)
    nt.assert_true(uvf.type == 'antenna')
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    nt.assert_true(np.all(uvf.flag_array[:, 0, 10, 0, 0]))
    nt.assert_true(np.all(uvf.flag_array[:, 0, 15, 1, 0]))
    nt.assert_true(uvf.flag_array.mean() == 2. * uvc.Nants_data / uvf.flag_array.size)


def test_to_antenna_metric():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.to_antenna(uvc)
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    nt.assert_true(np.all(uvf.metric_array[:, 0, 10, 0, 0] == 3.2))
    nt.assert_true(np.all(uvf.metric_array[:, 0, 15, 1, 0] == 2.1))
    nt.assert_true(np.isclose(uvf.metric_array.mean(),
                              (3.2 + 2.1) * uvc.Nants_data / uvf.metric_array.size))


def test_to_antenna_flags_match_uvflag():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf2 = uvf.copy()
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.to_antenna(uvf2)
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    nt.assert_true(np.all(uvf.flag_array[:, 0, 10, 0, 0]))
    nt.assert_true(np.all(uvf.flag_array[:, 0, 15, 1, 0]))
    nt.assert_true(uvf.flag_array.mean() == 2. * uvc.Nants_data / uvf.flag_array.size)


def test_antenna_to_antenna():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf2 = uvf.copy()
    uvf.to_antenna(uvc)
    nt.assert_equal(uvf, uvf2)


@uvtest.skipIf_no_h5py
def test_to_antenna_errors():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    nt.assert_raises(ValueError, uvf.to_antenna, 7.3)  # invalid matching object
    uvf = UVFlag(uv)
    nt.assert_raises(ValueError, uvf.to_antenna, uvc)  # Cannot pass in baseline type
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    uvf2 = uvf.copy()
    uvf.polarization_array[0] = -4
    nt.assert_raises(ValueError, uvf.to_antenna, uvc)  # Mismatched pols
    uvf.__iadd__(uvf2, axis='polarization')
    nt.assert_raises(ValueError, uvf.to_antenna, uvc)  # Mismatched pols, can't be forced


def test_to_antenna_force_pol():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvc.select(jones=-5)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[0, 10, 0] = True  # Flag time0, chan10
    uvf.flag_array[1, 15, 0] = True  # Flag time1, chan15
    uvf.polarization_array[0] = -4  # Change pol, but force pol anyway
    uvf.to_antenna(uvc, force_pol=True)
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    nt.assert_true(np.array_equal(uvf.polarization_array, uvc.jones_array))
    nt.assert_true(np.all(uvf.flag_array[:, 0, 10, 0, 0]))
    nt.assert_true(np.all(uvf.flag_array[:, 0, 15, 1, 0]))
    nt.assert_true(uvf.flag_array.mean() == 2 * uvc.Nants_data / uvf.flag_array.size)


def test_to_antenna_metric_force_pol():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvc.select(jones=-5)
    uvf = UVFlag(uvc)
    uvf.to_waterfall()
    uvf.metric_array[0, 10, 0] = 3.2  # Fill in time0, chan10
    uvf.metric_array[1, 15, 0] = 2.1  # Fill in time1, chan15
    uvf.polarization_array[0] = -4
    uvf.to_antenna(uvc, force_pol=True)
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    nt.assert_true(np.array_equal(uvf.polarization_array, uvc.jones_array))
    nt.assert_true(np.all(uvf.metric_array[:, 0, 10, 0, 0] == 3.2))
    nt.assert_true(np.all(uvf.metric_array[:, 0, 15, 1, 0] == 2.1))
    nt.assert_true(np.isclose(uvf.metric_array.mean(),
                              (3.2 + 2.1) * uvc.Nants_data / uvf.metric_array.size))


@uvtest.skipIf_no_h5py
def test_copy():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    nt.assert_true(uvf == uvf2)
    # Make sure it's a copy and not just pointing to same object
    uvf.to_waterfall()
    nt.assert_false(uvf == uvf2)


@uvtest.skipIf_no_h5py
def test_or():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.flag_array = np.ones_like(uvf2.flag_array)
    uvf.flag_array[0] = True
    uvf2.flag_array[0] = False
    uvf2.flag_array[1] = False
    uvf3 = uvf | uvf2
    nt.assert_true(np.all(uvf3.flag_array[0]))
    nt.assert_false(np.any(uvf3.flag_array[1]))
    nt.assert_true(np.all(uvf3.flag_array[2:]))


@uvtest.skipIf_no_h5py
def test_or_error():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    uvf.to_flag()
    nt.assert_raises(ValueError, uvf.__or__, uvf2)


@uvtest.skipIf_no_h5py
def test_or_add_history():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.history = 'Different history'
    uvf3 = uvf | uvf2
    nt.assert_true(uvf.history in uvf3.history)
    nt.assert_true(uvf2.history in uvf3.history)
    nt.assert_true("Flags OR'd with:" in uvf3.history)


@uvtest.skipIf_no_h5py
def test_ior():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.flag_array = np.ones_like(uvf2.flag_array)
    uvf.flag_array[0] = True
    uvf2.flag_array[0] = False
    uvf2.flag_array[1] = False
    uvf |= uvf2
    nt.assert_true(np.all(uvf.flag_array[0]))
    nt.assert_false(np.any(uvf.flag_array[1]))
    nt.assert_true(np.all(uvf.flag_array[2:]))


@uvtest.skipIf_no_h5py
def test_to_flag():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    nt.assert_true(hasattr(uvf, 'flag_array'))
    nt.assert_false(hasattr(uvf, 'metric_array'))
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true('Converted to mode "flag"' in uvf.history)


@uvtest.skipIf_no_h5py
def test_to_flag_threshold():
    uvf = UVFlag(test_f_file)
    uvf.metric_array = np.zeros_like(uvf.metric_array)
    uvf.metric_array[0, 0, 4, 0] = 2.
    uvf.to_flag(threshold=1.)
    nt.assert_true(hasattr(uvf, 'flag_array'))
    nt.assert_false(hasattr(uvf, 'metric_array'))
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_equal(uvf.flag_array[0, 0, 4, 0], True)
    nt.assert_equal(np.sum(uvf.flag_array), 1.)
    nt.assert_true('Converted to mode "flag"' in uvf.history)


@uvtest.skipIf_no_h5py
def test_flag_to_flag():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.to_flag()
    nt.assert_equal(uvf, uvf2)


@uvtest.skipIf_no_h5py
def test_to_flag_unknown_mode():
    uvf = UVFlag(test_f_file)
    uvf.mode = 'foo'
    nt.assert_raises(ValueError, uvf.to_flag)


@uvtest.skipIf_no_h5py
def test_to_metric_baseline():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.flag_array[:, :, 10] = True
    uvf.flag_array[1, :, :] = True
    nt.assert_true(hasattr(uvf, 'flag_array'))
    nt.assert_false(hasattr(uvf, 'metric_array'))
    nt.assert_true(uvf.mode == 'flag')
    uvf.to_metric(convert_wgts=True)
    nt.assert_true(hasattr(uvf, 'metric_array'))
    nt.assert_false(hasattr(uvf, 'flag_array'))
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true('Converted to mode "metric"' in uvf.history)
    nt.assert_true(np.isclose(uvf.weights_array[1], 0.0).all())
    nt.assert_true(np.isclose(uvf.weights_array[:, :, 10], 0.0).all())


@uvtest.skipIf_no_h5py
def test_to_metric_waterfall():
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    uvf.to_flag()
    uvf.flag_array[:, 10] = True
    uvf.flag_array[1, :, :] = True
    uvf.to_metric(convert_wgts=True)
    nt.assert_true(np.isclose(uvf.weights_array[1], 0.0).all())
    nt.assert_true(np.isclose(uvf.weights_array[:, 10], 0.0).all())


def test_to_metric_antenna():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc, mode='flag')
    uvf.flag_array[10, :, :, 1, :] = True
    uvf.flag_array[15, :, 3, :, :] = True
    uvf.to_metric(convert_wgts=True)
    nt.assert_true(np.isclose(uvf.weights_array[10, :, :, 1, :], 0.0).all())
    nt.assert_true(np.isclose(uvf.weights_array[15, :, 3, :, :], 0.0).all())


@uvtest.skipIf_no_h5py
def test_metric_to_metric():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    uvf.to_metric()
    nt.assert_equal(uvf, uvf2)


@uvtest.skipIf_no_h5py
def test_to_metric_unknown_mode():
    uvf = UVFlag(test_f_file)
    uvf.mode = 'foo'
    nt.assert_raises(ValueError, uvf.to_metric)


@uvtest.skipIf_no_h5py
def test_antpair2ind():
    uvf = UVFlag(test_f_file)
    ind = uvf.antpair2ind(uvf.ant_1_array[0], uvf.ant_2_array[0])
    nt.assert_true(np.all(uvf.ant_1_array[ind] == uvf.ant_1_array[0]))
    nt.assert_true(np.all(uvf.ant_2_array[ind] == uvf.ant_2_array[0]))


@uvtest.skipIf_no_h5py
def test_antpair2ind_nonbaseline():
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    nt.assert_raises(ValueError, uvf.antpair2ind, 0, 3)


@uvtest.skipIf_no_h5py
def test_baseline_to_antnums():
    uvf = UVFlag(test_f_file)
    a1, a2 = uvf.baseline_to_antnums(uvf.baseline_array[0])
    nt.assert_equal(a1, uvf.ant_1_array[0])
    nt.assert_equal(a2, uvf.ant_2_array[0])


@uvtest.skipIf_no_h5py
def test_get_baseline_nums():
    uvf = UVFlag(test_f_file)
    bls = uvf.get_baseline_nums()
    nt.assert_true(np.array_equal(bls, np.unique(uvf.baseline_array)))


@uvtest.skipIf_no_h5py
def test_get_antpairs():
    uvf = UVFlag(test_f_file)
    antpairs = uvf.get_antpairs()
    for a1, a2 in antpairs:
        ind = np.where((uvf.ant_1_array == a1) & (uvf.ant_2_array == a2))[0]
        nt.assert_true(len(ind) > 0)
    for a1, a2 in zip(uvf.ant_1_array, uvf.ant_2_array):
        nt.assert_true((a1, a2) in antpairs)


@uvtest.skipIf_no_h5py
def test_missing_Nants_telescope():
    import h5py

    testfile = os.path.join(DATA_PATH, 'test_missing_Nants.h5')
    shutil.copyfile(test_f_file, testfile)

    with h5py.File(testfile, 'r+') as f:
        del(f['/Header/Nants_telescope'])
    uvf = uvtest.checkWarnings(UVFlag, [testfile], {}, nwarnings=1,
                               message='Nants_telescope not availabe')
    uvf2 = UVFlag(test_f_file)
    uvf2.Nants_telescope = None
    nt.assert_true(uvf == uvf2)
    os.remove(testfile)


def test_combine_metrics_inplace():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.metric_array *= 2
    uvf3 = uvf.copy()
    uvf3.metric_array *= 3
    uvf.combine_metrics([uvf2, uvf3])
    factor = np.sqrt((1 + 4 + 9) / 3.) / 2.
    nt.assert_true(np.allclose(uvf.metric_array,
                   np.abs(uvf2.metric_array) * factor))


def test_combine_metrics_not_inplace():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.metric_array *= 2
    uvf3 = uvf.copy()
    uvf3.metric_array *= 3
    uvf4 = uvf.combine_metrics([uvf2, uvf3], inplace=False)
    factor = np.sqrt((1 + 4 + 9) / 3.)
    nt.assert_true(np.allclose(uvf4.metric_array,
                   np.abs(uvf.metric_array) * factor))


def test_combine_metrics_not_uvflag():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    nt.assert_raises(ValueError, uvf.combine_metrics, 'bubblegum')


def test_combine_metrics_not_metric():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.to_flag()
    nt.assert_raises(ValueError, uvf.combine_metrics, uvf2)


def test_combine_metrics_wrong_shape():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    np.random.seed(44)
    uvf.metric_array = np.random.normal(size=uvf.metric_array.shape)
    uvf2 = uvf.copy()
    uvf2.to_waterfall()
    nt.assert_raises(ValueError, uvf.combine_metrics, uvf2)


def test_super():

    class test_class(UVFlag):

        def __init__(self, input, mode='metric', copy_flags=False,
                     waterfall=False, history='', label='', property='prop'):

            super(test_class, self).__init__(input, mode=mode, copy_flags=copy_flags,
                                             waterfall=waterfall, history=history,
                                             label=label)

            self.property = property

    uv = UVData()
    uv.read_miriad(test_d_file)

    tc = test_class(uv, property='property')

    # UVFlag.__init__ is tested, so just see if it has a metric array
    nt.assert_true(hasattr(tc, 'metric_array'))
    # Check that it has the property
    nt.assert_true(tc.property == 'property')


def test_flags2waterfall():
    uv = UVData()
    uv.read_miriad(test_d_file)

    np.random.seed(0)
    uv.flag_array = np.random.randint(0, 2, size=uv.flag_array.shape, dtype=bool)
    wf = flags2waterfall(uv)
    nt.assert_almost_equal(np.mean(wf), np.mean(uv.flag_array))
    nt.assert_equal(wf.shape, (uv.Ntimes, uv.Nfreqs))

    wf = flags2waterfall(uv, keep_pol=True)
    nt.assert_equal(wf.shape, (uv.Ntimes, uv.Nfreqs, uv.Npols))

    # Test external flag_array
    uv.flag_array = np.zeros_like(uv.flag_array)
    f = np.random.randint(0, 2, size=uv.flag_array.shape, dtype=bool)
    wf = flags2waterfall(uv, flag_array=f)
    nt.assert_almost_equal(np.mean(wf), np.mean(f))
    nt.assert_equal(wf.shape, (uv.Ntimes, uv.Nfreqs))

    # UVCal version
    uvc = UVCal()
    uvc.read_calfits(test_c_file)

    uvc.flag_array = np.random.randint(0, 2, size=uvc.flag_array.shape, dtype=bool)
    wf = flags2waterfall(uvc)
    nt.assert_almost_equal(np.mean(wf), np.mean(uvc.flag_array))
    nt.assert_equal(wf.shape, (uvc.Ntimes, uvc.Nfreqs))

    wf = flags2waterfall(uvc, keep_pol=True)
    nt.assert_equal(wf.shape, (uvc.Ntimes, uvc.Nfreqs, uvc.Njones))


def test_flags2waterfall_errors():

    # First argument must be UVData or UVCal object
    nt.assert_raises(ValueError, flags2waterfall, 5)

    uv = UVData()
    uv.read_miriad(test_d_file)
    # Flag array must have same shape as uv.flag_array
    nt.assert_raises(ValueError, flags2waterfall, uv, np.array([4, 5]))


def test_and_rows_cols():
    d = np.zeros((10, 20), np.bool)
    d[1, :] = True
    d[:, 2] = True
    d[5, 10:20] = True
    d[5:8, 5] = True

    o = and_rows_cols(d)
    nt.assert_true(o[1, :].all())
    nt.assert_true(o[:, 2].all())
    nt.assert_false(o[5, :].all())
    nt.assert_false(o[:, 5].all())
