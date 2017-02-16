"""Tests for calfits object"""
import nose.tools as nt
import os
from pyuvdata.cal import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


def test_readwriteread():
    """
    Omnical fits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, 'test123.fits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')
    read_status = uvtest.checkWarnings(uv_in.read_calfits, [testfile],
                                       nwarnings=0,
                                       message='Telescope EVLA is not')
    uv_in.write_calfits(write_file, clobber=True)
    write_status = uvtest.checkWarnings(uv_out.read_calfits, [write_file],
                                        nwarnings=0,
                                        message='Telescope EVLA is not')
    nt.assert_true(read_status)
    nt.assert_true(write_status)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


def test_read_delay_file():
    """
    Read in a Firstcal calibration fits file that only saves delays.
    """
    uv_in = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
    read_status = uvtest.checkWarnings(uv_in.read_calfits, [testfile],
                                       nwarnings=0,
                                       message='Telescope EVLA is not')


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
    read_status = uvtest.checkWarnings(uv_in.read_calfits, [testfile],
                                       nwarnings=0,
                                       message='Telescope EVLA is not')
    uv_in.write_calfits(write_file, clobber=True)
    write_status = uvtest.checkWarnings(uv_out.read_calfits, [write_file],
                                        nwarnings=0,
                                        message='Telescope EVLA is not')
    nt.assert_true(read_status)
    nt.assert_true(write_status)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)
