import unittest
import inspect
import os
import os.path as op
import shutil
from uvdata.uv import UVData
import numpy as np


class TestUVDataInit(unittest.TestCase):
    def setUp(self):
        self.required_properties = ['data_array', 'nsample_array',
                                    'flag_array', 'Ntimes', 'Nbls', 'Nblts',
                                    'Nfreqs', 'Npols', 'Nspws', 'uvw_array',
                                    'time_array', 'ant_1_array', 'ant_2_array',
                                    'baseline_array', 'freq_array',
                                    'polarization_array', 'spw_array',
                                    'phase_center_ra', 'phase_center_dec',
                                    'integration_time', 'channel_width',
                                    'object_name', 'telescope_name',
                                    'instrument', 'latitude', 'longitude',
                                    'altitude', 'history',
                                    'vis_units', 'phase_center_epoch', 'Nants',
                                    'antenna_names', 'antenna_indices']

        self.extra_properties = ['extra_keywords', 'dateobs',
                                 'xyz_telescope_frame',
                                 'x_telescope', 'y_telescope', 'z_telescope',
                                 'antenna_positions', 'GST0', 'RDate',
                                 'earth_omega', 'DUT1', 'TIMESYS',
                                 'uvplane_reference_time']
        self.uv_object = UVData()

    def tearDown(self):
        del(self.uv_object)

    def test_property_iter(self):
        all = []
        for prop in self.uv_object.property_iter():
            all.append(prop)
        for a in self.required_properties + self.extra_properties:
            self.assertTrue(a in all,
                            msg='expected attribute ' + a +
                            ' not returned in property_iter')

    def test_required_property_iter(self):
        required = []
        for prop in self.uv_object.required_property_iter():
            required.append(prop)
        for a in self.required_properties:
            self.assertTrue(a in required,
                            msg='expected attribute ' + a +
                            ' not returned in required_property_iter')

    def test_extra_property_iter(self):
        extra = []
        for prop in self.uv_object.extra_property_iter():
            extra.append(prop)
        for a in self.extra_properties:
            self.assertTrue(a in extra,
                            msg='expected attribute ' + a +
                            ' not returned in extra_property_iter')

    def test_attributes_exist(self):
        expected_attributes = self.required_properties + self.extra_properties
        for a in expected_attributes:
            self.assertTrue(hasattr(self.uv_object, a),
                            msg='expected attribute ' + a + ' does not exist')

    def test_unexpected_attributes(self):
        expected_attributes = self.required_properties + self.extra_properties
        attributes = [i for i in self.uv_object.__dict__.keys() if i[0] != '_']
        for a in attributes:
            self.assertTrue(a in expected_attributes,
                            msg='unexpected attribute ' + a +
                            ' found in UVData')


class TestUVmethods(unittest.TestCase):
    def setUp(self):
        self.uv_object = UVData()
        self.uv_object.Nants = 128

    def tearDown(self):
        del(self.uv_object)

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
        del(UV)
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

        write_file = op.join(self.test_file_directory,
                             'outtest_casa_1src_1spw.uvfits')

        test = UV.write_uvfits(write_file)
        self.assertTrue(test)
        del(UV)

    def test_spwnotsupported(self):
        # testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        # testfile = '../data/PRISim_output_manual_conversion.uvfits'
        testfile = '../data/day2_TDEM0003_10s_norx_1scan.uvfits'
        self.assertTrue(os.path.exists(testfile))
        UV = UVData()
        self.assertRaises(ValueError, UV.read_uvfits, testfile)
        del(UV)

    def test_readwriteread(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        uv_in = UVData()
        uv_out = UVData()

        uv_in.read_uvfits(testfile)

        write_file = op.join(self.test_file_directory,
                             'outtest_casa.uvfits')

        uv_in.write_uvfits(write_file)

        uv_out.read_uvfits(write_file)

        self.assertEqual(uv_in, uv_out)
        del(uv_in)
        del(uv_out)


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
        supress_test = False

        files_present = True
        for f in self.testfiles:
            if not os.path.isfile(f):
                files_present = False

        if supress_test:
            print('FHD test supressed')
        elif not files_present:
            print('FHD files not present, skipping test')
        else:
            fhd_uv = UVData()
            uvfits_uv = UVData()
            fhd_uv.read_fhd(self.testfiles)

            fhd_uv.write_uvfits(op.join(self.test_file_directory,
                                        'outtest_FHD_1061321792.uvfits'),
                                spoof_nonessential=True)

            uvfits_uv.read_uvfits(op.join(self.test_file_directory,
                                  'outtest_FHD_1061321792.uvfits'))

            self.assertEqual(fhd_uv, uvfits_uv)

            del(fhd_uv)
            del(uvfits_uv)
class TestReadMiriad(unittest.TestCase):
    def setUp(self):
        self.supress_miriad_test = True
        if self.supress_miriad_test: return True
        self.datafile = '../../data/zen.2456865.60537.uvcRRE'
        if not os.path.exists(self.datafile):
            raise(IOError,'miriad file not found')
    def test_ReadMiriad(self):
        if self.supress_miriad_test: return True
        miriad_uv = UVData()
        status = miriad_uv.read_miriad(self.datafile)
        self.assertTrue(status)
if __name__ == '__main__':
    unittest.main()
