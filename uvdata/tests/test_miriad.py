import nose.tools as nt
import os
import os.path as op
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
import ephem
import uvdata.utils as ut


def test_ReadMiriadWriteUVFits():
    miriad_uv = UVData()
    uvfits_uv = UVData()
    miriad_file = '../data/zen.2456865.60537.xy.uvcRREAA'
    testfile = '../data/test/outtest_miriad.uvfits'
    miriad_out, miriad_status = ut.checkWarnings(miriad_uv.read, [miriad_file, 'miriad'],
                                                 known_warning='miriad')
    miriad_uv.write(testfile, file_type='uvfits', spoof_nonessential=True,
                    force_phase=True)
    uvfits_uv.read(testfile, 'uvfits')
    nt.assert_true(miriad_status)
    nt.assert_equal(miriad_uv, uvfits_uv)
    del(miriad_uv)
    del(uvfits_uv)


def test_breakReadMiriad():
    UV = UVData()
    nt.assert_raises(IOError, UV.read, 'foo', 'miriad')
    del(UV)


def test_writePAPER():
    testfile = '../data/zen.2456865.60537.xy.uvcRREAA'
    write_file = '../data/test/outtest_miriad.uv'
    UV = UVData()
    read_out, status = ut.checkWarnings(UV.read, [testfile, 'miriad'],
                                        known_warning='miriad')
    test = UV.write(write_file, file_type='miriad', clobber=True)
    nt.assert_true(status)
    nt.assert_true(test)
    del(UV)


def test_readWriteReadMiriad():
    testfile = '../data/zen.2456865.60537.xy.uvcRREAA'
    write_file = '../data/test/outtest_miriad.uv'
    uv_in = UVData()
    uv_out = UVData()
    read_out, status = ut.checkWarnings(uv_in.read, [testfile, 'miriad'],
                                        known_warning='miriad')
    uv_in.write(write_file, file_type='miriad', clobber=True)
    uv_out.read(write_file, 'miriad')

    nt.assert_true(status)
    nt.assert_equal(uv_in, uv_out)
    del(uv_in)
    del(uv_out)


'''
This test is commented out since we no longer believe AIPY phases correctly
to the astrometric ra/dec.  Hopefully we can reinstitute it one day.
def test_ReadMiriadPhase():
    unphasedfile = '../data/new.uvA'
    phasedfile = '../data/new.uvA.phased'
    unphased_uv = UVData()
    phased_uv = UVData()
    # test that phasing makes files equal
    unphased_out, unphased_status = ut.checkWarnings(unphased.read, [unphasedfile, 'miriad'],
                           known_warning='miriad')
    unphased.phase(ra=0.0, dec=0.0, epoch=ephem.J2000)
    phased_out, phased_status = ut.checkWarnings(phased.read, [phasedfile, 'miriad'],
                           known_warning='miriad')
    nt.assert_true(unphased_status)
    nt.assert_true(phased_status)
    nt.assert_equal(unphased, phased)
'''
