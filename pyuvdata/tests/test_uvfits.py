# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for UVFITS object.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import copy
import os

import six
import pytest
import astropy
from astropy.io import fits

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
    uvtest.checkWarnings(UV.read, [testfile], message='Telescope EVLA is not')
    assert expected_extra_keywords.sort() == list(UV.extra_keywords.keys()).sort()

    # test reading in header data first, then metadata and then data
    UV2 = UVData()
    uvtest.checkWarnings(UV2.read, [testfile], {'read_data': False, 'read_metadata': False},
                         message='Telescope EVLA is not')
    assert expected_extra_keywords.sort() == list(UV2.extra_keywords.keys()).sort()
    with pytest.raises(ValueError) as cm:
        UV2.check()
    assert str(cm.value).startswith('Required UVParameter')

    UV2.read(testfile, read_data=False)
    assert UV2.check()
    UV2.read(testfile)
    assert UV == UV2
    # test reading in header & metadata first, then data
    UV2 = UVData()
    uvtest.checkWarnings(UV2.read, [testfile], {'read_data': False},
                         message='Telescope EVLA is not')
    assert expected_extra_keywords.sort() == list(UV2.extra_keywords.keys()).sort()
    assert UV2.check()
    UV2.read(testfile)
    assert UV == UV2

    # check error trying to read metadata after data is already present
    with pytest.raises(ValueError) as cm:
        UV2.read(testfile, read_data=False)
    assert str(cm.value).startswith('data_array is already defined, cannot read metadata')


@pytest.mark.filterwarnings("ignore:Required Antenna frame keyword")
@pytest.mark.filterwarnings("ignore:telescope_location is not set")
def test_noSPW():
    """Test reading in a PAPER uvfits file with no spw axis."""
    UV = UVData()
    testfile_no_spw = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAAM.uvfits')
    UV.read(testfile_no_spw)
    del(UV)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_breakReadUVFits():
    """Test errors on reading in a uvfits file with subarrays and other problems."""
    UV = UVData()
    multi_subarray_file = os.path.join(DATA_PATH, 'multi_subarray.uvfits')
    with pytest.raises(ValueError) as cm:
        UV.read(multi_subarray_file)
    assert str(cm.value).startswith('This file appears to have multiple subarray')


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_source_group_params():
    # make a file with a single source to test that it works
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
    uv_in.read(testfile)
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = vis_hdu.data.parnames
        group_parameter_list = []

        lst_ind = 0
        for index, name in enumerate(par_names):
            par_value = vis_hdu.data.par(name)
            # lst_array needs to be split in 2 parts to get high enough accuracy
            if name.lower() == 'lst':
                if lst_ind == 0:
                    # first lst entry, par_value has full lst value (astropy adds the 2 values)
                    lst_array_1 = np.float32(par_value)
                    lst_array_2 = np.float32(par_value - np.float64(lst_array_1))
                    par_value = lst_array_1
                    lst_ind = 1
                else:
                    par_value = lst_array_2

            # need to account for PZERO values
            group_parameter_list.append(par_value - vis_hdr['PZERO' + str(index + 1)])

        par_names.append('SOURCE')
        source_array = np.ones_like(vis_hdu.data.par('BASELINE'))
        group_parameter_list.append(source_array)

        vis_hdu = fits.GroupData(raw_data_array, parnames=par_names,
                                 pardata=group_parameter_list, bitpix=-32)
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames['AIPS AN']]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)

    uv_out = UVData()
    uv_out.read(write_file)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_multisource_error():
    # make a file with multiple sources to test error condition
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
    uv_in.read(testfile)
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = vis_hdu.data.parnames
        group_parameter_list = []

        lst_ind = 0
        for index, name in enumerate(par_names):
            par_value = vis_hdu.data.par(name)
            # lst_array needs to be split in 2 parts to get high enough accuracy
            if name.lower() == 'lst':
                if lst_ind == 0:
                    # first lst entry, par_value has full lst value (astropy adds the 2 values)
                    lst_array_1 = np.float32(par_value)
                    lst_array_2 = np.float32(par_value - np.float64(lst_array_1))
                    par_value = lst_array_1
                    lst_ind = 1
                else:
                    par_value = lst_array_2

            # need to account for PZERO values
            group_parameter_list.append(par_value - vis_hdr['PZERO' + str(index + 1)])

        par_names.append('SOURCE')
        source_array = np.ones_like(vis_hdu.data.par('BASELINE'))
        mid_index = source_array.shape[0] // 2
        source_array[mid_index:] = source_array[mid_index:] * 2
        group_parameter_list.append(source_array)

        vis_hdu = fits.GroupData(raw_data_array, parnames=par_names,
                                 pardata=group_parameter_list, bitpix=-32)
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames['AIPS AN']]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)

    with pytest.raises(ValueError) as cm:
        uv_in.read(write_file)
    assert str(cm.value).startswith('This file has multiple sources')


def test_spwnotsupported():
    """Test errors on reading in a uvfits file with multiple spws."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1scan.uvfits')
    with pytest.raises(ValueError) as cm:
        UV.read(testfile)
    assert str(cm.value).startswith('Sorry.  Files with more than one spectral'
                                    'window (spw) are not yet supported')


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
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
    uv_in.read(testfile)
    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # test that it works with write_lst = False
    uv_in.write_uvfits(write_file, write_lst=False)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that if x_orientation is set, it's read back out properly
    uv_in.x_orientation = 'east'
    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that if antenna_diameters is set, it's read back out properly
    uv_in.read(testfile)
    uv_in.antenna_diameters = np.zeros((uv_in.Nants_telescope,), dtype=np.float) + 14.0
    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that if antenna_numbers are > 256 everything works
    uv_in.read(testfile)
    uv_in.antenna_numbers = uv_in.antenna_numbers + 256
    uv_in.ant_1_array = uv_in.ant_1_array + 256
    uv_in.ant_2_array = uv_in.ant_2_array + 256
    uv_in.baseline_array = uv_in.antnums_to_baseline(uv_in.ant_1_array, uv_in.ant_2_array)
    uvtest.checkWarnings(uv_in.write_uvfits, [write_file],
                         message='antnums_to_baseline: found > 256 antennas, using 2048 baseline')
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check missing telescope_name, timesys vs timsys spelling, xyz_telescope_frame=????
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()

        vis_hdr.pop('TELESCOP')

        vis_hdu.header = vis_hdr

        ant_hdu = hdu_list[hdunames['AIPS AN']]
        ant_hdr = ant_hdu.header.copy()

        time_sys = ant_hdr.pop('TIMSYS')
        ant_hdr['TIMESYS'] = time_sys
        ant_hdr['FRAME'] = '????'

        ant_hdu.header = ant_hdr

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)

    uv_out.read(write_file)
    assert uv_out.telescope_name == 'EVLA'
    assert uv_out.timesys == time_sys

    # check error if timesys is 'IAT'
    uv_in.read(testfile)
    uv_in.timesys = 'IAT'
    with pytest.raises(ValueError) as cm:
        uv_in.write_uvfits(write_file)
    assert str(cm.value).startswith('This file has a time system IAT. '
                                    'Only "UTC" time system files are supported')
    uv_in.timesys = 'UTC'

    # check error if one time & no inttime specified
    uv_singlet = uv_in.select(times=uv_in.time_array[0], inplace=False)
    uv_singlet.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = np.array(vis_hdu.data.parnames)
        pars_use = np.where(par_names != 'INTTIM')[0]
        par_names = par_names[pars_use].tolist()

        group_parameter_list = [vis_hdu.data.par(name) for name in par_names]

        vis_hdu = fits.GroupData(raw_data_array, parnames=par_names,
                                 pardata=group_parameter_list, bitpix=-32)
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr

        ant_hdu = hdu_list[hdunames['AIPS AN']]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)

    with pytest.raises(ValueError) as cm:
        uvtest.checkWarnings(uv_out.read, func_args=[write_file],
                             message=['Telescope EVLA is not',
                                      'ERFA function "utcut1" yielded 1 of "dubious year (Note 3)"',
                                      'ERFA function "utctai" yielded 1 of "dubious year (Note 3)"',
                                      'LST values stored in this file are not self-consistent'],
                             nwarnings=4,
                             category=[UserWarning, astropy._erfa.core.ErfaWarning,
                             astropy._erfa.core.ErfaWarning, UserWarning])
    assert str(cm.value).startswith('integration time not specified and only one time present')

    # check that unflagged data with nsample = 0 will cause warnings
    uv_in.nsample_array[list(range(11, 22))] = 0
    uv_in.flag_array[list(range(11, 22))] = False
    uvtest.checkWarnings(uv_in.write_uvfits, [write_file], message='Some unflagged data has nsample = 0')


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_extra_keywords():
    uv_in = UVData()
    uv_out = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
    uv_in.read(uvfits_file)

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    uv_in.extra_keywords['testdict'] = {'testkey': 23}
    uvtest.checkWarnings(uv_in.check, message=['testdict in extra_keywords is a '
                                               'list, array or dict'])
    if six.PY2:
        with pytest.raises(TypeError) as cm:
            uv_in.write_uvfits(testfile, run_check=False)
        assert str(cm.value).startswith("Extra keyword testdict is of <type 'dict'>")
    else:
        with pytest.raises(TypeError) as cm:
            uv_in.write_uvfits(testfile, run_check=False)
        assert str(cm.value).startswith("Extra keyword testdict is of <class 'dict'>")
    uv_in.extra_keywords.pop('testdict')

    uv_in.extra_keywords['testlist'] = [12, 14, 90]
    uvtest.checkWarnings(uv_in.check, message=['testlist in extra_keywords is a '
                                               'list, array or dict'])
    if six.PY2:
        with pytest.raises(TypeError) as cm:
            uv_in.write_uvfits(testfile, run_check=False)
        assert str(cm.value).startswith("Extra keyword testlist is of <type 'list'>")
    else:
        with pytest.raises(TypeError) as cm:
            uv_in.write_uvfits(testfile, run_check=False)
        assert str(cm.value).startswith("Extra keyword testlist is of <class 'list'>")
    uv_in.extra_keywords.pop('testlist')

    uv_in.extra_keywords['testarr'] = np.array([12, 14, 90])
    uvtest.checkWarnings(uv_in.check, message=['testarr in extra_keywords is a '
                                               'list, array or dict'])
    if six.PY2:
        with pytest.raises(TypeError) as cm:
            uv_in.write_uvfits(testfile, run_check=False)
        assert str(cm.value).startswith("Extra keyword testarr is of <type 'numpy.ndarray'>")
    else:
        with pytest.raises(TypeError) as cm:
            uv_in.write_uvfits(testfile, run_check=False)
        assert str(cm.value).startswith("Extra keyword testarr is of <class 'numpy.ndarray'>")
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
    uv_out.read(testfile)

    assert uv_in == uv_out
    uv_in.extra_keywords.pop('bool')
    uv_in.extra_keywords.pop('bool2')

    # check handling of int-like keywords
    uv_in.extra_keywords['int1'] = np.int(5)
    uv_in.extra_keywords['int2'] = 7
    uv_in.write_uvfits(testfile)
    uv_out.read(testfile)

    assert uv_in == uv_out
    uv_in.extra_keywords.pop('int1')
    uv_in.extra_keywords.pop('int2')

    # check handling of float-like keywords
    uv_in.extra_keywords['float1'] = np.int64(5.3)
    uv_in.extra_keywords['float2'] = 6.9
    uv_in.write_uvfits(testfile)
    uv_out.read(testfile)

    assert uv_in == uv_out
    uv_in.extra_keywords.pop('float1')
    uv_in.extra_keywords.pop('float2')

    # check handling of complex-like keywords
    uv_in.extra_keywords['complex1'] = np.complex64(5.3 + 1.2j)
    uv_in.extra_keywords['complex2'] = 6.9 + 4.6j
    uv_in.write_uvfits(testfile)
    uv_out.read(testfile)

    assert uv_in == uv_out
    uv_in.extra_keywords.pop('complex1')
    uv_in.extra_keywords.pop('complex2')

    # check handling of comment keywords
    uv_in.extra_keywords['comment'] = ('this is a very long comment that will '
                                       'be broken into several lines\nif '
                                       'everything works properly.')
    uv_in.write_uvfits(testfile)
    uv_out.read(testfile)

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_roundtrip_blt_order():
    uv_in = UVData()
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
    uv_in.read(testfile)

    uv_in.reorder_blts()

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # test with bda as well (single entry in tuple)
    uv_in.reorder_blts(order='bda')

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:Required Antenna frame keyword")
@pytest.mark.filterwarnings("ignore:telescope_location is not set")
def test_select_read():
    uvfits_uv = UVData()
    uvfits_uv2 = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')

    # select on antennas
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])
    uvfits_uv.read(uvfits_file, antenna_nums=ants_to_keep)
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(antenna_nums=ants_to_keep)
    assert uvfits_uv == uvfits_uv2

    # select on frequency channels
    chans_to_keep = np.arange(12, 22)
    uvfits_uv.read(uvfits_file, freq_chans=chans_to_keep)
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(freq_chans=chans_to_keep)
    assert uvfits_uv == uvfits_uv2

    # check writing & reading single frequency files
    uvfits_uv.select(freq_chans=[0])
    testfile = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
    uvfits_uv.write_uvfits(testfile)
    uvfits_uv2.read(testfile)
    assert uvfits_uv == uvfits_uv2

    # select on pols
    pols_to_keep = [-1, -2]
    uvfits_uv.read(uvfits_file, polarizations=pols_to_keep)
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(polarizations=pols_to_keep)
    assert uvfits_uv == uvfits_uv2

    # select on read using time_range
    unique_times = np.unique(uvfits_uv.time_array)
    uvtest.checkWarnings(uvfits_uv.read, [uvfits_file],
                         {'time_range': [unique_times[0], unique_times[1]]},
                         nwarnings=2,
                         message=['Warning: "time_range" keyword is set',
                                  'Telescope EVLA is not'])
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(times=unique_times[0:2])
    assert uvfits_uv == uvfits_uv2

    # now test selecting on multiple axes
    # frequencies first
    uvfits_uv.read(uvfits_file, antenna_nums=ants_to_keep,
                   freq_chans=chans_to_keep, polarizations=pols_to_keep)
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                      polarizations=pols_to_keep)
    assert uvfits_uv == uvfits_uv2

    # baselines first
    ants_to_keep = np.array([0, 1])
    uvfits_uv.read(uvfits_file, antenna_nums=ants_to_keep,
                   freq_chans=chans_to_keep, polarizations=pols_to_keep)
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                      polarizations=pols_to_keep)
    assert uvfits_uv == uvfits_uv2

    # polarizations first
    ants_to_keep = np.array([0, 1, 2, 3, 6, 7, 8, 11, 14, 18, 19, 20, 21, 22])
    chans_to_keep = np.arange(12, 64)
    uvfits_uv.read(uvfits_file, antenna_nums=ants_to_keep,
                   freq_chans=chans_to_keep, polarizations=pols_to_keep)
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(antenna_nums=ants_to_keep, freq_chans=chans_to_keep,
                      polarizations=pols_to_keep)
    assert uvfits_uv == uvfits_uv2

    # repeat with no spw file
    uvfitsfile_no_spw = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAAM.uvfits')

    # select on antennas
    ants_to_keep = np.array([2, 4, 5])
    uvfits_uv.read(uvfitsfile_no_spw, antenna_nums=ants_to_keep)
    uvfits_uv2.read(uvfitsfile_no_spw)
    uvfits_uv2.select(antenna_nums=ants_to_keep)
    assert uvfits_uv == uvfits_uv2

    # select on frequency channels
    chans_to_keep = np.arange(4, 8)
    uvfits_uv.read(uvfitsfile_no_spw, freq_chans=chans_to_keep)
    uvfits_uv2.read(uvfitsfile_no_spw)
    uvfits_uv2.select(freq_chans=chans_to_keep)
    assert uvfits_uv == uvfits_uv2

    # select on pols
    # this requires writing a new file because the no spw file we have has only 1 pol
    with fits.open(uvfits_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data
        raw_data_array = raw_data_array[:, :, :, 0, :, :, :]

        vis_hdr['NAXIS'] = 6

        vis_hdr['NAXIS5'] = vis_hdr['NAXIS6']
        vis_hdr['CTYPE5'] = vis_hdr['CTYPE6']
        vis_hdr['CRVAL5'] = vis_hdr['CRVAL6']
        vis_hdr['CDELT5'] = vis_hdr['CDELT6']
        vis_hdr['CRPIX5'] = vis_hdr['CRPIX6']
        vis_hdr['CROTA5'] = vis_hdr['CROTA6']

        vis_hdr['NAXIS6'] = vis_hdr['NAXIS7']
        vis_hdr['CTYPE6'] = vis_hdr['CTYPE7']
        vis_hdr['CRVAL6'] = vis_hdr['CRVAL7']
        vis_hdr['CDELT6'] = vis_hdr['CDELT7']
        vis_hdr['CRPIX6'] = vis_hdr['CRPIX7']
        vis_hdr['CROTA6'] = vis_hdr['CROTA7']

        vis_hdr.pop('NAXIS7')
        vis_hdr.pop('CTYPE7')
        vis_hdr.pop('CRVAL7')
        vis_hdr.pop('CDELT7')
        vis_hdr.pop('CRPIX7')
        vis_hdr.pop('CROTA7')

        par_names = vis_hdu.data.parnames

        group_parameter_list = [vis_hdu.data.par(ind) for
                                ind in range(len(par_names))]

        vis_hdu = fits.GroupData(raw_data_array, parnames=par_names,
                                 pardata=group_parameter_list, bitpix=-32)
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr

        ant_hdu = hdu_list[hdunames['AIPS AN']]

        write_file = os.path.join(DATA_PATH, 'test/outtest_casa.uvfits')
        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)

    pols_to_keep = [-1, -2]
    uvfits_uv.read(write_file, polarizations=pols_to_keep)
    uvfits_uv2.read(uvfits_file)
    uvfits_uv2.select(polarizations=pols_to_keep)
    assert uvfits_uv == uvfits_uv2


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
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
    uvfits_uv.read(uvfits_file)
    uvfits_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read_miriad(testfile)

    assert miriad_uv == uvfits_uv

    # check that setting the phase_type keyword also works
    miriad_uv.read_miriad(testfile, phase_type='phased')

    # check that setting the phase_type to drift raises an error
    with pytest.raises(ValueError) as cm:
        miriad_uv.read_miriad(testfile, phase_type='drift')
    assert str(cm.value).startswith('phase_type is "drift" but the RA values are constant.')

    # check that setting it works after selecting a single time
    uvfits_uv.select(times=uvfits_uv.time_array[0])
    uvfits_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read_miriad(testfile)

    assert miriad_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_multi_files():
    """
    Reading multiple files at once.
    """
    uv_full = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile1 = os.path.join(DATA_PATH, 'test/uv1.uvfits')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2.uvfits')
    uv_full.read(uvfits_file)
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvfits(testfile1)
    uv2.write_uvfits(testfile2)
    uv1.read([testfile1, testfile2])
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis '
                                    'using pyuvdata.', uv1.history)

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # again, setting axis
    uv1.read([testfile1, testfile2], axis='freq')
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis '
                                    'using pyuvdata.', uv1.history)

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # check with metadata_only
    uv_full = UVData()
    uv_full.read(uvfits_file, read_data=False)
    uv1 = UVData()
    uv1.read([testfile1, testfile2], read_data=False)

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis '
                                    'using pyuvdata.', uv1.history)

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # check raises error if read_data and read_metadata are False
    with pytest.raises(ValueError) as cm:
        uv1.read([testfile1, testfile2], read_data=False, read_metadata=False)
    assert str(cm.value).startswith('A list of files cannot be used when just '
                                    'reading the header')

    # check raises error if only reading data on a list of files (metadata already read)
    uv1 = UVData()
    uv1.read(uvfits_file, read_data=False)
    with pytest.raises(ValueError) as cm:
        uv1.read([testfile1, testfile2])
    assert str(cm.value).startswith('A list of files cannot be used when just '
                                    'reading data (metadata already exists)')


@uvtest.skipIf_no_casa
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readMSWriteUVFits_CASAHistory():
    """
    read in .ms file.
    Write to a uvfits file, read back in and check for casa_history parameter
    """
    ms_uv = UVData()
    uvfits_uv = UVData()
    ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    testfile = os.path.join(DATA_PATH, 'test/outtest.uvfits')
    ms_uv.read_ms(ms_file)
    ms_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvfits_uv.read(testfile)
    assert ms_uv == uvfits_uv
