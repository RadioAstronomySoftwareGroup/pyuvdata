"""Tests for calfits object"""
import nose.tools as nt
import os
import astropy
from astropy.io import fits
from pyuvdata.uvcal import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils
import numpy as np


def test_readwriteread():
    """
    Omnical fits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    uv_in.read_calfits(testfile)
    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)

    # test without freq_range parameter
    uv_in.freq_range = None
    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)


def test_readwriteread_delays():
    """
    Read-Write-Read test with a fits calibration files containing delays.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality
    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    message = testfile + ' appears to be an old calfits format'
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])
    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


def test_errors():
    """
    Test for various errors.

    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    message = testfile + ' appears to be an old calfits format'
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])

    uv_in.set_unknown_cal_type()
    nt.assert_raises(ValueError, uv_in.write_calfits, write_file, run_check=False, clobber=True)

    # change values for various axes in flag and total quality hdus to not match primary hdu
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])

    # Create filler jones info
    uv_in.jones_array = np.array([-5, -6, -7, -8])
    uv_in.Njones = 4
    uv_in.flag_array = np.zeros(uv_in._flag_array.expected_shape(uv_in), dtype=bool)
    uv_in.delay_array = np.ones(uv_in._delay_array.expected_shape(uv_in), dtype=np.float64)
    uv_in.quality_array = np.zeros(uv_in._quality_array.expected_shape(uv_in))

    # add total_quality_array so that can be tested as well
    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    header_vals_to_double = [{'flag': 'CDELT2'}, {'flag': 'CDELT3'},
                             {'flag': 'CRVAL5'}, {'totqual': 'CDELT1'},
                             {'totqual': 'CDELT2'}, {'totqual': 'CRVAL4'}]
    for i, hdr_dict in enumerate(header_vals_to_double):
        uv_in.write_calfits(write_file, clobber=True)

        unit = hdr_dict.keys()[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils.fits_indexhdus(F)
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

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        print(unit, keyword)
        nt.assert_raises(ValueError, uv_out.read_calfits, write_file, strict_fits=True)

    # repeat for gain type file
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    uv_in.read_calfits(testfile)

    # Create filler jones info
    uv_in.jones_array = np.array([-5, -6, -7, -8])
    uv_in.Njones = 4
    uv_in.flag_array = np.zeros(uv_in._flag_array.expected_shape(uv_in), dtype=bool)
    uv_in.gain_array = np.ones(uv_in._gain_array.expected_shape(uv_in), dtype=np.complex64)
    uv_in.quality_array = np.zeros(uv_in._quality_array.expected_shape(uv_in))

    # add total_quality_array so that can be tested as well
    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    header_vals_to_double = [{'totqual': 'CDELT1'}, {'totqual': 'CDELT2'},
                             {'totqual': 'CDELT3'}, {'totqual': 'CRVAL4'}]

    for i, hdr_dict in enumerate(header_vals_to_double):
        uv_in.write_calfits(write_file, clobber=True)

        unit = hdr_dict.keys()[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils.fits_indexhdus(F)
        ant_hdu = F[hdunames['ANTENNAS']]
        totqualhdu = F[hdunames['TOTQLTY']]
        totqualhdr = totqualhdu.header

        if unit == 'totqual':
            totqualhdr[keyword] *= 2

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])
        totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
        hdulist.append(totqualhdu)

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        nt.assert_raises(ValueError, uv_out.read_calfits, write_file, strict_fits=True)


def test_read_oldcalfits():
    """
    Test for proper behavior with old calfits files.
    """
    # start with gain type files
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    uv_in.read_calfits(testfile)

    # add total_quality_array so that can be tested as well
    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    # now read in the file and remove various CRPIX and CRVAL keywords to
    # emulate old calfits files
    header_vals_to_remove = [{'primary': 'CRVAL5'}, {'primary': 'CRPIX4'},
                             {'totqual': 'CRVAL4'}]
    messages = [write_file, 'This file', write_file]
    messages = [m + ' appears to be an old calfits format' for m in messages]
    for i, hdr_dict in enumerate(header_vals_to_remove):
        uv_in.write_calfits(write_file, clobber=True)

        unit = hdr_dict.keys()[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils.fits_indexhdus(F)
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

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        uvtest.checkWarnings(uv_out.read_calfits, [write_file], nwarnings=1,
                             message=messages[i], category=[UserWarning])
        nt.assert_equal(uv_in, uv_out)
        nt.assert_raises(KeyError, uv_out.read_calfits, write_file, strict_fits=True)

    # now with delay type files
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_firstcal.fits')
    message = testfile + ' appears to be an old calfits format'
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])

    # add total_quality_array so that can be tested as well
    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    # now read in the file and remove various CRPIX and CRVAL keywords to
    # emulate old calfits files
    header_vals_to_remove = [{'primary': 'CRVAL5'}, {'flag': 'CRVAL5'},
                             {'flag': 'CRPIX4'}, {'totqual': 'CRVAL4'}]
    messages = [write_file, 'This file', 'This file', write_file]
    messages = [m + ' appears to be an old calfits format' for m in messages]
    for i, hdr_dict in enumerate(header_vals_to_remove):
        uv_in.write_calfits(write_file, clobber=True)

        unit = hdr_dict.keys()[0]
        keyword = hdr_dict[unit]

        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils.fits_indexhdus(F)
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

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        uvtest.checkWarnings(uv_out.read_calfits, [write_file], nwarnings=1,
                             message=messages[i], category=[UserWarning])
        nt.assert_equal(uv_in, uv_out)
        nt.assert_raises(KeyError, uv_out.read_calfits, write_file, strict_fits=True)


def test_input_flag_array():
    """
    Test when data file has input flag array.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_input_flags.fits')
    uv_in.read_calfits(testfile)
    uv_in.input_flag_array = np.zeros(uv_in._input_flag_array.expected_shape(uv_in), dtype=bool)
    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)

    # Repeat for delay version
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    message = testfile + ' appears to be an old calfits format'
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])
    uv_in.input_flag_array = np.zeros(uv_in._input_flag_array.expected_shape(uv_in), dtype=bool)
    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


def test_jones():
    """
    Test when data file has more than one element in Jones matrix.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_jones.fits')
    uv_in.read_calfits(testfile)

    # Create filler jones info
    uv_in.jones_array = np.array([-5, -6, -7, -8])
    uv_in.Njones = 4
    uv_in.flag_array = np.zeros(uv_in._flag_array.expected_shape(uv_in), dtype=bool)
    uv_in.gain_array = np.ones(uv_in._gain_array.expected_shape(uv_in), dtype=np.complex64)
    uv_in.quality_array = np.zeros(uv_in._quality_array.expected_shape(uv_in))

    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)

    # Repeat for delay version
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    message = testfile + ' appears to be an old calfits format'
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])

    # Create filler jones info
    uv_in.jones_array = np.array([-5, -6, -7, -8])
    uv_in.Njones = 4
    uv_in.flag_array = np.zeros(uv_in._flag_array.expected_shape(uv_in), dtype=bool)
    uv_in.delay_array = np.ones(uv_in._delay_array.expected_shape(uv_in), dtype=np.float64)
    uv_in.quality_array = np.zeros(uv_in._quality_array.expected_shape(uv_in))

    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


def test_readwriteread_total_quality_array():
    """
    Test when data file has a total quality array.

    Currently we have no such file, so we will artificially create one and
    check for internal consistency.
    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_total_quality_array.fits')
    uv_in.read_calfits(testfile)

    # Create filler total quality array
    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)

    # also test delay-type calibrations
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_total_quality_array_delays.fits')
    message = testfile + ' appears to be an old calfits format'
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])

    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


def test_total_quality_array_size():
    """
    Test that total quality array defaults to the proper size
    """

    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
    uv_in.read_calfits(testfile)

    # Create filler total quality array
    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    proper_shape = (uv_in.Nspws, uv_in.Nfreqs, uv_in.Ntimes, uv_in.Njones)
    nt.assert_equal(uv_in.total_quality_array.shape, proper_shape)
    del(uv_in)

    # also test delay-type calibrations
    uv_in = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    message = testfile + ' appears to be an old calfits format'
    uvtest.checkWarnings(uv_in.read_calfits, [testfile], nwarnings=1,
                         message=message, category=[UserWarning])

    uv_in.total_quality_array = np.zeros(uv_in._total_quality_array.expected_shape(uv_in))

    proper_shape = (uv_in.Nspws, 1, uv_in.Ntimes, uv_in.Njones)
    nt.assert_equal(uv_in.total_quality_array.shape, proper_shape)
    del(uv_in)
