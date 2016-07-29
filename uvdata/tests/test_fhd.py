import unittest
import os
import os.path as op
import shutil
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
from test_functions import *


class TestReadFHD(unittest.TestCase):
    def setUp(self):
        self.test_file_directory = '../data/test/'
        if not op.exists(self.test_file_directory):
            print('making test directory')
            os.mkdir(self.test_file_directory)

        testdir = '../data/fhd_vis_data/'
        testfile_prefix = '1061316296_'
        testfile_suffix = ['flags.sav', 'vis_XX.sav', 'params.sav',
                           'vis_YY.sav', 'vis_model_XX.sav',
                           'vis_model_YY.sav', 'settings.txt']
        self.testfiles = []
        for s in testfile_suffix:
            self.testfiles.append(testdir + testfile_prefix + s)

    # def tearDown(self):
    #      shutil.rmtree(self.test_file_directory)

    def test_ReadFHD(self):

        fhd_uv = UVData()
        uvfits_uv = UVData()
        fhd_uv.read(self.testfiles, 'fhd')
        fhd_uv.write(op.join(self.test_file_directory,
                             'outtest_FHD_1061316296.uvfits'),
                     file_type='uvfits',
                     spoof_nonessential=True)

        uvfits_uv.read(op.join(self.test_file_directory,
                               'outtest_FHD_1061316296.uvfits'), 'uvfits')

        self.assertEqual(fhd_uv, uvfits_uv)
        del(fhd_uv)
        del(uvfits_uv)

        # Try various cases of incomplete file lists
        fhd_uv = UVData()
        self.assertRaises(StandardError, fhd_uv.read, self.testfiles[1:], 'fhd')  # Missing flags
        del(fhd_uv)
        fhd_uv = UVData()
        subfiles = [item for sublist in [self.testfiles[0:2], self.testfiles[3:]] for item in sublist]
        self.assertRaises(StandardError, fhd_uv.read, subfiles, 'fhd')  # Missing params
        del(fhd_uv)
        fhd_uv = UVData()
        self.assertRaises(StandardError, fhd_uv.read, ['foo'], 'fhd')  # No data files
        del(fhd_uv)
        fhd_uv = UVData()
        self.assertTrue(checkWarnings(self, fhd_uv.read, [self.testfiles[:-1],
                                      'fhd'], message=['No settings']))
        self.assertEqual(fhd_uv.history, '')  # Check empty history with no settings
        del(fhd_uv)

    def test_ReadFHD_model(self):

        fhd_uv = UVData()
        uvfits_uv = UVData()
        fhd_uv.read(self.testfiles, 'fhd')

        fhd_uv.write(op.join(self.test_file_directory,
                             'outtest_FHD_1061316296_model.uvfits'),
                     file_type='uvfits',
                     spoof_nonessential=True)

        uvfits_uv.read(op.join(self.test_file_directory,
                       'outtest_FHD_1061316296_model.uvfits'), 'uvfits')

        self.assertEqual(fhd_uv, uvfits_uv)

        del(fhd_uv)
        del(uvfits_uv)
