"""Tests for Miriad object."""
import nose.tools as nt
import os
import numpy as np
import ephem
from pyuvdata import UVData
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


def test_ReadMiriadWriteUVFits():
    """
    Miriad to uvfits loopback test.

    Read in Miriad files, write out as uvfits, read back in and check for
    object equality.
    """
    miriad_uv = UVData()
    uvfits_uv = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uvfits')
    miriad_status = uvtest.checkWarnings(miriad_uv.read_miriad, [miriad_file],
                                         known_warning='miriad')
    miriad_uv.write_uvfits(testfile, spoof_nonessential=True,
                           force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    nt.assert_true(miriad_status)
    nt.assert_equal(miriad_uv, uvfits_uv)
    del(miriad_uv)
    del(uvfits_uv)


def test_breakReadMiriad():
    """Test Miriad file checking."""
    UV = UVData()
    nt.assert_raises(IOError, UV.read_miriad, 'foo')
    del(UV)


def test_writePAPER():
    """Test reading & writing PAPER Miriad file."""
    UV = UVData()
    testfile = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    status = uvtest.checkWarnings(UV.read_miriad, [testfile],
                                  known_warning='miriad')
    UV.write_miriad(write_file, clobber=True)
    nt.assert_true(status)
    del(UV)


def test_readWriteReadMiriad():
    """
    PAPER file Miriad loopback test.

    Read in Miriad PAPER file, write out as new Miriad file, read back in and
    check for object equality.
    """
    uv_in = UVData()
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    status = uvtest.checkWarnings(uv_in.read_miriad, [testfile],
                                  known_warning='miriad')
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read_miriad(write_file)

    nt.assert_true(status)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


def test_rwrMiriad_antpos_issues():
    """
    test warnings and errors associated with antenna position issues in Miriad files

    Read in Miriad PAPER file, mess with various antpos issues and write out as
    a new Miriad file, read back in and check for appropriate behavior.
    """
    uv_in = UVData()
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    status = uvtest.checkWarnings(uv_in.read_miriad, [testfile],
                                  known_warning='miriad')
    uv_in.antenna_positions = None
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read_miriad(write_file)

    nt.assert_true(status)
    nt.assert_equal(uv_in, uv_out)

    status = uvtest.checkWarnings(uv_in.read_miriad, [testfile],
                                  known_warning='miriad')
    ants_with_data = list(set(uv_in.ant_1_array).union(uv_in.ant_2_array))
    ant_ind = np.where(uv_in.antenna_numbers == ants_with_data[0])[0]
    uv_in.antenna_positions[ant_ind, :] = [0, 0, 0]
    uv_in.write_miriad(write_file, clobber=True, no_antnums=True)
    status = uvtest.checkWarnings(uv_out.read_miriad, [write_file],
                                  message=['antenna number'])

    nt.assert_true(status)
    nt.assert_equal(uv_in, uv_out)

    status = uvtest.checkWarnings(uv_in.read_miriad, [testfile],
                                  known_warning='miriad')
    uv_in.antenna_positions = None
    ants_with_data = sorted(list(set(uv_in.ant_1_array).union(uv_in.ant_2_array)))
    new_nums = []
    new_names = []
    for a in ants_with_data:
        new_nums.append(a)
        ind = np.where(uv_in.antenna_numbers == a)[0][0]
        new_names.append(uv_in.antenna_names[ind])
    uv_in.antenna_numbers = np.array(new_nums)
    uv_in.antenna_names = new_names
    uv_in.Nants_telescope = len(uv_in.antenna_numbers)
    uv_in.write_miriad(write_file, clobber=True, no_antnums=True)
    uv_out.read_miriad(write_file)

    nt.assert_true(status)
    nt.assert_equal(uv_in, uv_out)

'''
This test is commented out since we no longer believe AIPY phases correctly
to the astrometric ra/dec.  Hopefully we can reinstitute it one day.
def test_ReadMiriadPhase():
    unphasedfile = os.path.join(DATA_PATH, 'new.uvA')
    phasedfile = os.path.join(DATA_PATH, 'new.uvA.phased')
    unphased_uv = UVData()
    phased_uv = UVData()
    # test that phasing makes files equal
    unphased_out, unphased_status = uvtest.checkWarnings(unphased.read, [unphasedfile, 'miriad'],
                           known_warning='miriad')
    unphased.phase(ra=0.0, dec=0.0, epoch=ephem.J2000)
    phased_out, phased_status = uvtest.checkWarnings(phased.read, [phasedfile, 'miriad'],
                           known_warning='miriad')
    nt.assert_true(unphased_status)
    nt.assert_true(phased_status)
    nt.assert_equal(unphased, phased)
'''
