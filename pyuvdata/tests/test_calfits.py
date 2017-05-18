"""Tests for calfits object"""
import nose.tools as nt
import os
from pyuvdata.uvcal import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
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
    uv_in.read_calfits(testfile)
    uv_in.write_calfits(write_file, clobber=True)
    uv_out.read_calfits(write_file)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


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
    uv_in.read_calfits(testfile)
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
    uv_in.read_calfits(testfile)

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
