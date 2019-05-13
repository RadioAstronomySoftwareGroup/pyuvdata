# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for calfits object

"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
from astropy.io import fits

from pyuvdata import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils


def test_readwriteread():
    """
    Omnical fits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(testfile)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out

    # test without freq_range parameter
    cal_in.freq_range = None
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_readwriteread_delays():
    """
    Read-Write-Read test with a fits calibration files containing delays.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    cal_in.read_calfits(testfile)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del(cal_in)
    del(cal_out)


def test_errors():
    """
    Test for various errors.

    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    cal_in.read_calfits(testfile)

    cal_in.set_unknown_cal_type()
    pytest.raises(ValueError, cal_in.write_calfits, write_file, run_check=False, clobber=True)

    # change values for various axes in flag and total quality hdus to not match primary hdu
    cal_in.read_calfits(testfile)
    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.delay_array = np.ones(cal_in._delay_array.expected_shape(cal_in), dtype=np.float64)
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    header_vals_to_double = [{'flag': 'CDELT2'}, {'flag': 'CDELT3'},
                             {'flag': 'CRVAL5'}, {'totqual': 'CDELT1'},
                             {'totqual': 'CDELT2'}, {'totqual': 'CRVAL4'}]
    for i, hdr_dict in enumerate(header_vals_to_double):
        cal_in.write_calfits(write_file, clobber=True)

        unit = list(hdr_dict.keys())[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils._fits_indexhdus(F)
        ant_hdu = F[hdunames['ANTENNAS']]
        flag_hdu = F[hdunames['FLAGS']]
        flag_hdr = flag_hdu.header
        totqualhdu = F[hdunames['TOTQLTY']]
        totqualhdr = totqualhdu.header

        if unit == 'flag':
            flag_hdr[keyword] *= 2
        elif unit == 'totqual':
            totqualhdr[keyword] *= 2

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])
        flag_hdu = fits.ImageHDU(data=flag_hdu.data, header=flag_hdr)
        hdulist.append(flag_hdu)
        totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
        hdulist.append(totqualhdu)

        hdulist.writeto(write_file, overwrite=True)

        pytest.raises(ValueError, cal_out.read_calfits, write_file, strict_fits=True)

    # repeat for gain type file
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(testfile)

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.gain_array = np.ones(cal_in._gain_array.expected_shape(cal_in), dtype=np.complex64)
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    header_vals_to_double = [{'totqual': 'CDELT1'}, {'totqual': 'CDELT2'},
                             {'totqual': 'CDELT3'}, {'totqual': 'CRVAL4'}]

    for i, hdr_dict in enumerate(header_vals_to_double):
        cal_in.write_calfits(write_file, clobber=True)

        unit = list(hdr_dict.keys())[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils._fits_indexhdus(F)
        ant_hdu = F[hdunames['ANTENNAS']]
        totqualhdu = F[hdunames['TOTQLTY']]
        totqualhdr = totqualhdu.header

        if unit == 'totqual':
            totqualhdr[keyword] *= 2

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])
        totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
        hdulist.append(totqualhdu)

        hdulist.writeto(write_file, overwrite=True)

        pytest.raises(ValueError, cal_out.read_calfits, write_file, strict_fits=True)


def test_extra_keywords():
    cal_in = UVCal()
    cal_out = UVCal()
    calfits_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    testfile = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(calfits_file)

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    cal_in.extra_keywords['testdict'] = {'testkey': 23}
    uvtest.checkWarnings(cal_in.check, message=['testdict in extra_keywords is a '
                                                'list, array or dict'])
    pytest.raises(TypeError, cal_in.write_calfits, testfile, run_check=False)
    cal_in.extra_keywords.pop('testdict')

    cal_in.extra_keywords['testlist'] = [12, 14, 90]
    uvtest.checkWarnings(cal_in.check, message=['testlist in extra_keywords is a '
                                                'list, array or dict'])
    pytest.raises(TypeError, cal_in.write_calfits, testfile, run_check=False)
    cal_in.extra_keywords.pop('testlist')

    cal_in.extra_keywords['testarr'] = np.array([12, 14, 90])
    uvtest.checkWarnings(cal_in.check, message=['testarr in extra_keywords is a '
                                                'list, array or dict'])
    pytest.raises(TypeError, cal_in.write_calfits, testfile, run_check=False)
    cal_in.extra_keywords.pop('testarr')

    # check for warnings with extra_keywords keys that are too long
    cal_in.extra_keywords['test_long_key'] = True
    uvtest.checkWarnings(cal_in.check, message=['key test_long_key in extra_keywords '
                                                'is longer than 8 characters'])
    uvtest.checkWarnings(cal_in.write_calfits, [testfile], {'run_check': False,
                                                            'clobber': True},
                         message=['key test_long_key in extra_keywords is longer than 8 characters'])
    cal_in.extra_keywords.pop('test_long_key')

    # check handling of boolean keywords
    cal_in.extra_keywords['bool'] = True
    cal_in.extra_keywords['bool2'] = False
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out
    cal_in.extra_keywords.pop('bool')
    cal_in.extra_keywords.pop('bool2')

    # check handling of int-like keywords
    cal_in.extra_keywords['int1'] = np.int(5)
    cal_in.extra_keywords['int2'] = 7
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out
    cal_in.extra_keywords.pop('int1')
    cal_in.extra_keywords.pop('int2')

    # check handling of float-like keywords
    cal_in.extra_keywords['float1'] = np.int64(5.3)
    cal_in.extra_keywords['float2'] = 6.9
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out
    cal_in.extra_keywords.pop('float1')
    cal_in.extra_keywords.pop('float2')

    # check handling of complex-like keywords
    cal_in.extra_keywords['complex1'] = np.complex64(5.3 + 1.2j)
    cal_in.extra_keywords['complex2'] = 6.9 + 4.6j
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out
    cal_in.extra_keywords.pop('complex1')
    cal_in.extra_keywords.pop('complex2')

    # check handling of comment keywords
    cal_in.extra_keywords['comment'] = ('this is a very long comment that will '
                                        'be broken into several lines\nif '
                                        'everything works properly.')
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out


def test_read_oldcalfits_gain():
    """
    Test for proper behavior with old calfits gain-style files.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(testfile)

    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    # now read in the file and remove various CRPIX and CRVAL keywords to
    # emulate old calfits files
    header_vals_to_remove = [{'primary': 'CRVAL5'}, {'primary': 'CRPIX4'},
                             {'totqual': 'CRVAL4'}, {'primary': 'CALSTYLE'}]
    messages = [write_file, 'This file', write_file, write_file]
    messages = [m + ' appears to be an old calfits format' for m in messages]
    for i, hdr_dict in enumerate(header_vals_to_remove):
        cal_in.write_calfits(write_file, clobber=True)

        unit = list(hdr_dict.keys())[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils._fits_indexhdus(F)
        ant_hdu = F[hdunames['ANTENNAS']]
        totqualhdu = F[hdunames['TOTQLTY']]
        totqualhdr = totqualhdu.header

        if unit == 'primary':
            primary_hdr.pop(keyword)
        elif unit == 'totqual':
            totqualhdr.pop(keyword)

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])
        totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
        hdulist.append(totqualhdu)

        hdulist.writeto(write_file, overwrite=True)

        uvtest.checkWarnings(cal_out.read_calfits, [write_file], message=messages[i],
                             category=DeprecationWarning)
        assert cal_in == cal_out
        if keyword.startswith('CR'):
            pytest.raises(KeyError, cal_out.read_calfits, write_file, strict_fits=True)


def test_read_oldcalfits_delay():
    """
    Test for proper behavior with old calfits delay-style files.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    cal_in.read_calfits(testfile)

    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    # now read in the file and remove various CRPIX and CRVAL keywords to
    # emulate old calfits files
    header_vals_to_remove = [{'primary': 'CRVAL5'}, {'flag': 'CRVAL5'},
                             {'flag': 'CRPIX4'}, {'totqual': 'CRVAL4'},
                             {'primary': 'CALSTYLE'}]
    messages = [write_file, 'This file', 'This file', write_file, write_file]
    messages = [m + ' appears to be an old calfits format' for m in messages]
    for i, hdr_dict in enumerate(header_vals_to_remove):
        cal_in.write_calfits(write_file, clobber=True)

        unit = list(hdr_dict.keys())[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils._fits_indexhdus(F)
        ant_hdu = F[hdunames['ANTENNAS']]
        flag_hdu = F[hdunames['FLAGS']]
        flag_hdr = flag_hdu.header
        totqualhdu = F[hdunames['TOTQLTY']]
        totqualhdr = totqualhdu.header

        if unit == 'primary':
            primary_hdr.pop(keyword)
        elif unit == 'flag':
            flag_hdr.pop(keyword)
        elif unit == 'totqual':
            totqualhdr.pop(keyword)

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])
        flag_hdu = fits.ImageHDU(data=flag_hdu.data, header=flag_hdr)
        hdulist.append(flag_hdu)
        totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
        hdulist.append(totqualhdu)

        hdulist.writeto(write_file, overwrite=True)

        uvtest.checkWarnings(cal_out.read_calfits, [write_file], message=messages[i],
                             category=DeprecationWarning)
        assert cal_in == cal_out
        if keyword.startswith('CR'):
            pytest.raises(KeyError, cal_out.read_calfits, write_file, strict_fits=True)


def test_read_oldcalfits_delay_nofreqaxis():
    """
    Test for proper behavior with old calfits delay-style files that have no freq axis.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    cal_in.read_calfits(testfile)

    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    # now read in the file and remove the freq axis to emulate old calfits files
    cal_in.write_calfits(write_file, clobber=True)

    F = fits.open(write_file)
    data = F[0].data
    primary_hdr = F[0].header
    hdunames = uvutils._fits_indexhdus(F)
    ant_hdu = F[hdunames['ANTENNAS']]
    flag_hdu = F[hdunames['FLAGS']]
    flag_hdr = flag_hdu.header
    totqualhdu = F[hdunames['TOTQLTY']]
    totqualhdr = totqualhdu.header

    axis_keyword_base = ['CTYPE', 'CUNIT', 'CRPIX', 'CRVAL', 'CDELT']
    for keyword in axis_keyword_base:
        primary_hdr.pop(keyword + '4')

    # need to renumber spw & antenna indices
    for keyword in axis_keyword_base:
        primary_hdr[keyword + '4'] = primary_hdr.pop(keyword + '5')
        primary_hdr[keyword + '5'] = primary_hdr.pop(keyword + '6')

    data = data[:, :, 0, :, :, :]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, ant_hdu])
    flag_hdu = fits.ImageHDU(data=flag_hdu.data, header=flag_hdr)
    hdulist.append(flag_hdu)
    totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
    hdulist.append(totqualhdu)

    hdulist.writeto(write_file, overwrite=True)

    message = write_file + ' appears to be an old calfits format'
    uvtest.checkWarnings(cal_out.read_calfits, [write_file], message=message,
                         category=DeprecationWarning)
    assert cal_in == cal_out


def test_input_flag_array():
    """
    Test when data file has input flag array.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_input_flags.fits')
    cal_in.read_calfits(testfile)
    cal_in.input_flag_array = np.zeros(cal_in._input_flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out

    # Repeat for delay version
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    cal_in.read_calfits(testfile)
    cal_in.input_flag_array = np.zeros(cal_in._input_flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del(cal_in)
    del(cal_out)


def test_jones():
    """
    Test when data file has more than one element in Jones matrix.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_jones.fits')
    cal_in.read_calfits(testfile)

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.gain_array = np.ones(cal_in._gain_array.expected_shape(cal_in), dtype=np.complex64)
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out

    # Repeat for delay version
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    cal_in.read_calfits(testfile)

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.delay_array = np.ones(cal_in._delay_array.expected_shape(cal_in), dtype=np.float64)
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del(cal_in)
    del(cal_out)


def test_readwriteread_total_quality_array():
    """
    Test when data file has a total quality array.

    Currently we have no such file, so we will artificially create one and
    check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_total_quality_array.fits')
    cal_in.read_calfits(testfile)

    # Create filler total quality array
    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del(cal_in)
    del(cal_out)

    # also test delay-type calibrations
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_total_quality_array_delays.fits')
    cal_in.read_calfits(testfile)

    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del(cal_in)
    del(cal_out)


def test_total_quality_array_size():
    """
    Test that total quality array defaults to the proper size
    """

    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    cal_in.read_calfits(testfile)

    # Create filler total quality array
    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    proper_shape = (cal_in.Nspws, cal_in.Nfreqs, cal_in.Ntimes, cal_in.Njones)
    assert cal_in.total_quality_array.shape == proper_shape
    del(cal_in)

    # also test delay-type calibrations
    cal_in = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    cal_in.read_calfits(testfile)

    cal_in.total_quality_array = np.zeros(cal_in._total_quality_array.expected_shape(cal_in))

    proper_shape = (cal_in.Nspws, 1, cal_in.Ntimes, cal_in.Njones)
    assert cal_in.total_quality_array.shape == proper_shape
    del(cal_in)


def test_write_time_precision():
    """
    Test that times are being written to appropriate precision (see issue 311).
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(testfile)
    # overwrite time array to break old code
    dt = cal_in.integration_time / (24. * 60. * 60.)
    cal_in.time_array = dt * np.arange(cal_in.Ntimes)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_read_noversion_history():
    """
    Test that version info gets added to the history if it's missing
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(testfile)

    cal_in.write_calfits(write_file, clobber=True)

    F = fits.open(write_file)
    data = F[0].data
    primary_hdr = F[0].header
    hdunames = uvutils._fits_indexhdus(F)
    ant_hdu = F[hdunames['ANTENNAS']]

    primary_hdr['HISTORY'] = ''

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, ant_hdu])

    hdulist.writeto(write_file, overwrite=True)

    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_spw_zero_indexed_gain():
    """
    Test that old files with zero-indexed spw array are read correctly for gain-type
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(testfile)

    cal_in.write_calfits(write_file, clobber=True)

    F = fits.open(write_file)
    data = F[0].data
    primary_hdr = F[0].header
    hdunames = uvutils._fits_indexhdus(F)
    ant_hdu = F[hdunames['ANTENNAS']]

    primary_hdr['CRVAL5'] = 0

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, ant_hdu])

    hdulist.writeto(write_file, overwrite=True)

    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_spw_zero_indexed_delay():
    """
    Test that old files with zero-indexed spw array are read correctly for delay-type
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    cal_in.read_calfits(testfile)

    cal_in.write_calfits(write_file, clobber=True)
    F = fits.open(write_file)
    data = F[0].data
    primary_hdr = F[0].header
    hdunames = uvutils._fits_indexhdus(F)
    ant_hdu = F[hdunames['ANTENNAS']]
    flag_hdu = F[hdunames['FLAGS']]
    flag_hdr = flag_hdu.header

    primary_hdr['CRVAL5'] = 0

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, ant_hdu])
    flag_hdu = fits.ImageHDU(data=flag_hdu.data, header=flag_hdr)
    hdulist.append(flag_hdu)

    hdulist.writeto(write_file, overwrite=True)

    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_write_freq_spacing_not_channel_width():
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    cal_in.read_calfits(testfile)

    # select every other frequency -- then evenly spaced but doesn't match channel width
    cal_in.select(freq_chans=np.arange(0, 10, 2))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
