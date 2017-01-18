"""Tests for calfits object"""
import nose.tools as nt
import os
from uvdata.cal import UVCal
import uvdata.tests as uvtest
from uvdata.data import DATA_PATH

def test_readwriteread():
    """
    Omnical fits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    uv_in = UVCal()
    uv_out = UVCal()
    testfile = os.path.join(DATA_PATH, '/Users/sherlock/src/pyuvdata/uvdata/data/zen.2457698.50098.xx.fits')
    write_file = os.path.join(DATA_PATH, 'test/outtest_omnical.fits')

    read_status = uvtest.checkWarnings(uv_in.read_calfits, [testfile],
                                       message='Telescope EVLA is not')
    import IPython; IPython.embed()
    uv_in.write_calfits(write_file, run_check=False)
    write_status = uvtest.checkWarnings(uv_out.read_calfits, [write_file],
                                        message='Telescope EVLA is not')
    nt.assert_true(read_status)
    nt.assert_true(write_status)
    nt.assert_equal(uv_in, uv_out)

test_readwriteread()
