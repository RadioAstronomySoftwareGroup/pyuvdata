import unittest
import os
import os.path as op
import shutil
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
from astropy.io import fits
from test_functions import *


class TestReadUVFits(unittest.TestCase):
    def test_ReadNRAO(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        testfile_no_spw = '../data/zen.2456865.60537.xy.uvcRREAAM.uvfits'
        UV = UVData()
        self.assertRaises(ValueError, UV.read, testfile, 'vufits')  # Wrong filetype
        test = checkWarnings(self, UV.read, [testfile, 'uvfits'],
                             warning_message='Telescope EVLA is not')
        self.assertTrue(test)
        expected_extra_keywords = ['OBSERVER', 'SORTORD', 'SPECSYS',
                                   'RESTFREQ', 'ORIGIN']
        self.assertEqual(expected_extra_keywords.sort(),
                         UV.extra_keywords.keys().sort())

        del(UV)
        UV = UVData()
        test = checkWarnings(self, UV.read, [testfile_no_spw, 'uvfits'],
                             known_warning='paper_uvfits')
        self.assertTrue(test)

        del(UV)

    def test_breaks(self):
        # Here I'll group together tests that should raise exceptions
        multi_subarray_file = '../data/test/multi_subarray.uvfits'
        if not op.exists(multi_subarray_file):
            # Make uvfits file with multiple subarrays
            # First read/write using pyuvdata to force use of antenna arrays.
            UV = UVData()
            UV.read('../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits', 'uvfits')
            UV.write('../data/test/temp.uvfits', 'uvfits')
            # Then use astropy.io.fits to change the subarray array
            F = fits.open('../data/test/temp.uvfits')
            F[0].data['SUBARRAY'][2] = 2.0
            F.writeto(multi_subarray_file)
            os.remove('../data/test/temp.uvfits')
            del(UV)
        UV = UVData()
        self.assertRaises(ValueError, UV.read, multi_subarray_file, 'uvfits')

    # def test_readRTS(self):
    #     testfile = '../data/pumav2_SelfCal300_Peel300_01.uvfits'
    #     UV = UVData()
    #     test = UV.read_uvfits(testfile)
    #     self.assertTrue(test)


class TestWriteUVFits(unittest.TestCase):
    def setUp(self):
        self.test_file_directory = '../data/test/'

    # def tearDown(self):
    #      shutil.rmtree(self.test_file_directory)

    def test_writeNRAO(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        # testfile = '../data/PRISim_output_manual_conversion.uvfits'
        UV = UVData()
        checkWarnings(self, UV.read, [testfile, 'uvfits'],
                      warning_message='Telescope EVLA is not')

        write_file = op.join(self.test_file_directory,
                             'outtest_casa_1src_1spw.uvfits')

        test = UV.write(write_file, file_type='uvfits')
        self.assertTrue(test)
        del(UV)

    def test_spwnotsupported(self):
        # testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        # testfile = '../data/PRISim_output_manual_conversion.uvfits'
        testfile = '../data/day2_TDEM0003_10s_norx_1scan.uvfits'
        self.assertTrue(op.exists(testfile))
        UV = UVData()
        self.assertRaises(ValueError, UV.read, testfile, 'uvfits')
        del(UV)

    def test_readwriteread(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        uv_in = UVData()
        uv_out = UVData()
        checkWarnings(self, uv_in.read, [testfile, 'uvfits'],
                      warning_message='Telescope EVLA is not')
        write_file = op.join(self.test_file_directory,
                             'outtest_casa.uvfits')
        uv_in.write(write_file, file_type='uvfits')
        checkWarnings(self, uv_out.read, [write_file, 'uvfits'],
                      warning_message='Telescope EVLA is not')
        self.assertEqual(uv_in, uv_out)
        del(uv_in)
        del(uv_out)
