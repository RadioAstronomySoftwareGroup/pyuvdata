import unittest
import os
import os.path as op
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
import ephem
from test_functions import *


class TestReadMiriad(unittest.TestCase):
    def setUp(self):
        self.datafile = '../data/zen.2456865.60537.xy.uvcRREAA'
        if not op.exists(self.datafile):
            raise(IOError, 'miriad file not found')
        self.miriad_uv = UVData()
        self.uvfits_uv = UVData()

        self.unphasedfile = '../data/new.uvA'
        if not op.exists(self.unphasedfile):
            raise(IOError, 'miriad file not found')
        self.unphased = UVData()

        self.phasedfile = '../data/new.uvA.phased'
        if not os.path.exists(self.phasedfile):
            raise(IOError, 'miriad file not found')
        self.phased = UVData()

        self.test_file_directory = '../data/test/'

        status = checkWarnings(self.miriad_uv.read, [self.datafile, 'miriad'],
                               known_warning='miriad')

        self.assertTrue(status[1])

    def test_ReadMiriad(self):
        # Test loop with writing/reading uvfits
        uvfits_testfile = op.join(self.test_file_directory,
                                  'outtest_miriad.uvfits')
        # Simultaneously test the general write function for case of uvfits
        self.miriad_uv.write(uvfits_testfile, file_type='uvfits',
                             spoof_nonessential=True,
                             force_phase=True)
        self.uvfits_uv.read(uvfits_testfile, 'uvfits')

        self.assertEqual(self.miriad_uv, self.uvfits_uv)

        # Test exception
        self.assertRaises(IOError, self.miriad_uv.read, 'foo', 'miriad')
    '''
    This test is commented out since we no longer believe AIPY phases correctly
    to the astrometric ra/dec.  Hopefully we can reinstitute it one day.
    def test_ReadMiriadPhase(self):
        # test that phasing makes files equal
        status = checkWarnings(self.unphased.read, [self.unphasedfile, 'miriad'],
                               known_warning='miriad')
        self.unphased.phase(ra=0.0, dec=0.0, epoch=ephem.J2000)
        status = checkWarnings(self.phased.read, [self.phasedfile, 'miriad'],
                               known_warning='miriad')

        self.assertEqual(self.unphased, self.phased)
    '''


class TestWriteMiriad(unittest.TestCase):
    def setUp(self):
        self.test_file_directory = '../data/test/'
        if not os.path.exists(self.test_file_directory):
            print('making test directory')
            os.mkdir(self.test_file_directory)

    def test_writePAPER(self):
        testfile = '../data/zen.2456865.60537.xy.uvcRREAA'
        UV = UVData()
        status = checkWarnings(UV.read, [testfile, 'miriad'], known_warning='miriad')

        write_file = op.join(self.test_file_directory,
                             'outtest_miriad.uv')

        test = UV.write(write_file, file_type='miriad', clobber=True)
        self.assertTrue(test)
        del(UV)

    def test_readwriteread(self):
        testfile = '../data/zen.2456865.60537.xy.uvcRREAA'
        uv_in = UVData()
        uv_out = UVData()

        status = checkWarnings(uv_in.read, [testfile, 'miriad'],
                               known_warning='miriad')

        write_file = op.join(self.test_file_directory,
                             'outtest_miriad.uv')

        uv_in.write(write_file, file_type='miriad', clobber=True)

        uv_out.read(write_file, 'miriad')

        self.assertEqual(uv_in, uv_out)
        del(uv_in)
        del(uv_out)
