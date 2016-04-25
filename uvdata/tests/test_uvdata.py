import unittest
import inspect
import os
import os.path as op
import shutil
from uvdata.uv import UVData


class TestUVDataInit(unittest.TestCase):
    def setUp(self):
        self.default_attributes = ['data_array', 'nsample_array',
                                   'flag_array', 'Ntimes', 'Nbls', 'Nblts',
                                   'Nfreqs', 'Npols', 'Nspws', 'uvw_array',
                                   'time_array', 'ant_1_array', 'ant_2_array',
                                   'baseline_array', 'freq_array',
                                   'polarization_array', 'spw_array',
                                   'phase_center_ra', 'phase_center_dec',
                                   'integration_time', 'channel_width',
                                   'object_name', 'telescope_name',
                                   'instrument', 'latitude', 'longitude',
                                   'altitude', 'dateobs', 'history',
                                   'vis_units', 'phase_center_epoch', 'Nants',
                                   'antenna_names', 'antenna_indices',
                                   'extra_keywords']
        self.uv_object = UVData()

    def tearDown(self):
        del(self.uv_object)

    def test_default_attributes_exist(self):
        for attribute in self.default_attributes:
            self.assertTrue(hasattr(self.uv_object, attribute),
                            msg='expected attribute ' + attribute +
                            ' does not exist')

#    def test_default_attribute_values(self):
#        for attribute in self.default_attributes:
#            self.assertIsNone(getattr(self.uv_object, attribute),
#                              msg='attribute ' + attribute +
#                              ' is not None as expected')

    def test_unexpected_attributes(self):
        attributes = [i for i in self.uv_object.__dict__.keys() if i[0] != '_']
        for attribute in attributes:
            self.assertTrue(attribute in self.default_attributes,
                            msg='unexpected attribute ' + attribute +
                            ' found in UVData')


class TestUVmethods(unittest.TestCase):
    def setUp(self):
        self.uv_object = UVData()
        self.uv_object.Nants = 128

    def test_ij2bl(self):
        self.assertEqual(self.uv_object.baseline_to_antnums(67585),
                         (0, 0))
        # self.Nants = 2049
        # self.assertRaises(StandardError,self.uv_object.bl_to_ij,67585)
        # self.Nants = 128

    def test_bl2ij(self):
        self.assertEqual(self.uv_object.antnums_to_baseline(0, 0),
                         67585)
        self.assertEqual(self.uv_object.antnums_to_baseline(257, 256),
                         592130)


class TestReadUVFits(unittest.TestCase):
    def test_ReadNRAO(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        UV = UVData()
        test = UV.read_uvfits(testfile)
        self.assertTrue(test)
    #
    # def test_readRTS(self):
    #     testfile = '../data/pumav2_SelfCal300_Peel300_01.uvfits'
    #     UV = UVData()
    #     test = UV.read_uvfits(testfile)
    #     self.assertTrue(test)


class TestWriteUVFits(unittest.TestCase):
    def setUp(self):
        self.test_file_directory = '../data/test/'
        if not os.path.exists(self.test_file_directory):
            print('making test directory')
            os.mkdir(self.test_file_directory)

    # def tearDown(self):
    #      shutil.rmtree(self.test_file_directory)

    def test_writeNRAO(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        # testfile = '../data/PRISim_output_manual_conversion.uvfits'
        UV = UVData()
        UV.read_uvfits(testfile)
        # test = UV.write_uvfits('outtest.uvfits')
        test = UV.write_uvfits(op.join(self.test_file_directory,
                                       'outtest_casa_1src_1spw.uvfits'))
        self.assertTrue(test)

    def test_spwnotsupported(self):
        # testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        # testfile = '../data/PRISim_output_manual_conversion.uvfits'
        testfile = '../data/day2_TDEM0003_10s_norx_1scan.uvfits'
        self.assertTrue(os.path.exists(testfile))
        UV = UVData()
        self.assertRaises(IOError, UV.read_uvfits, testfile)

    def test_readwriteread(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        UV = UVData()
        UV.read_uvfits(testfile)
        UV.write_uvfits(op.join(self.test_file_directory,
                                'outtest_casa.uvfits'))
        test = UV.read_uvfits(op.join(self.test_file_directory,
                                      'outtest_casa.uvfits'))
        self.assertTrue(test)


class TestReadFHD(unittest.TestCase):
    def setUp(self):
        self.test_file_directory = '../data/test/'
        if not os.path.exists(self.test_file_directory):
            print('making test directory')
            os.mkdir(self.test_file_directory)

        testdir = '../data/fhd_vis_data/'
        testfile_prefix = '1061321792_'
        testfile_suffix = ['flags.sav', 'vis_XX.sav', 'params.sav',
                           'vis_YY.sav', 'settings.txt']
        self.testfiles = []
        for s in testfile_suffix:
            self.testfiles.append(testdir + testfile_prefix + s)

    # def tearDown(self):
    #      shutil.rmtree(self.test_file_directory)

    def test_ReadFHD(self):
        UV = UVData()
        test = UV.read_fhd(self.testfiles)
        self.assertTrue(test)

        UV.write_uvfits(op.join(self.test_file_directory,
                                'outtest_FHD.uvfits'),
                        spoof_nonessential=True)
        test = UV.read_uvfits('outtest_FHD.uvfits')
        self.assertTrue(test)


if __name__ == '__main__':
    unittest.main()
