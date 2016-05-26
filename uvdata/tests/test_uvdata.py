import unittest
import inspect
import os
import os.path as op
import shutil
from uvdata.uv import UVData
import numpy as np
import copy


class TestUVDataInit(unittest.TestCase):
    def setUp(self):
        self.required_properties = ['data_array', 'nsample_array',
                                    'flag_array', 'Ntimes', 'Nbls', 'Nblts',
                                    'Nfreqs', 'Npols', 'Nspws', 'uvw_array',
                                    'time_array', 'ant_1_array', 'ant_2_array',
                                    'lst_array',
                                    'baseline_array', 'freq_array',
                                    'polarization_array', 'spw_array',
                                    'integration_time', 'channel_width',
                                    'object_name', 'telescope_name',
                                    'instrument', 'latitude', 'longitude',
                                    'altitude', 'history',
                                    'vis_units', 'phase_center_epoch',
                                    'Nants_data', 'Nants_telescope',
                                    'antenna_names', 'antenna_indices']

        self.extra_properties = ['extra_keywords', 'dateobs',
                                 'xyz_telescope_frame',
                                 'x_telescope', 'y_telescope', 'z_telescope',
                                 'antenna_positions', 'GST0', 'RDate',
                                 'earth_omega', 'DUT1', 'TIMESYS',
                                 'uvplane_reference_time',
                                 'phase_center_ra', 'phase_center_dec',
                                 'zenith_ra', 'zenith_dec']
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
        self.uv_object.Nants_telescope.value = 128
        self.testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'

    def tearDown(self):
        del(self.uv_object)

    def test_bl2ij(self):
        self.assertEqual(self.uv_object.baseline_to_antnums(67585),
                         (0, 0))
        Nants = self.uv_object.Nants_telescope.value
        self.uv_object.Nants_telescope.value = 2049
        self.assertRaises(StandardError, self.uv_object.baseline_to_antnums, 67585)
        self.uv_object.Nants_telescope.value = Nants  # reset

    def test_ij2bl(self):
        self.assertEqual(self.uv_object.antnums_to_baseline(0, 0),
                         67585)
        self.assertEqual(self.uv_object.antnums_to_baseline(257, 256),
                         592130)
        # Check attempt256
        self.assertEqual(self.uv_object.antnums_to_baseline(0, 0, attempt256=True),
                         257)
        self.assertEqual(self.uv_object.antnums_to_baseline(257, 256,
                         attempt256=True), 592130)
        Nants = self.uv_object.Nants_telescope.value
        self.uv_object.Nants_telescope.value = 2049
        self.assertRaises(StandardError, self.uv_object.antnums_to_baseline, 0, 0)
        self.uv_object.Nants_telescope.value = Nants  # reset

    def test_data_equality(self):
        try:
            self.uv_object.check()
        except ValueError:
            self.uv_object.read(self.testfile, 'uvfits')
        self.assertEqual(self.uv_object, self.uv_object)
        self.uv_object2 = copy.deepcopy(self.uv_object)
        self.uv_object2.data_array.value[0, 0, 0, 0] += 1  # Force data to be not equal
        self.assertNotEqual(self.uv_object, self.uv_object2)
        # check class equality test
        self.assertNotEqual(self.uv_object, self.uv_object.data_array)

    def test_setXYZ_from_LatLon(self):
        self.uv_object.latitude.set_degrees(-26.7)
        self.uv_object.longitude.set_degrees(116.7)
        self.uv_object.altitude.value = None
        # Test that exception is raised.
        self.assertRaises(ValueError, self.uv_object.setXYZ_from_LatLon)
        self.uv_object.altitude.value = 377.8
        status = self.uv_object.setXYZ_from_LatLon()
        # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        # to give additional precision.
        ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)
        out_xyz = (self.uv_object.x_telescope.value, self.uv_object.y_telescope.value,
                   self.uv_object.z_telescope.value)
        self.assertTrue(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))

    def test_check(self):
        try:
            self.uv_object.check()
        except ValueError:
            self.uv_object.read(self.testfile, 'uvfits')
        self.assertTrue(self.uv_object.check())
        # Now break it in every way I can.
        # String cases
        units = self.uv_object.vis_units.value
        self.uv_object.vis_units.value = 1
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.vis_units.value = units  # reset it
        # Single value cases
        Nblts = self.uv_object.Nblts.value
        self.uv_object.Nblts.value = 4
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.Nblts.value = np.float(Nblts)
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.Nblts.value = Nblts  # reset
        # Array cases
        data = self.uv_object.data_array.value
        self.uv_object.data_array.value = np.array([4, 5, 6], dtype=np.complex64)
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.data_array.value = np.real(data)
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.data_array.value = data  # reset
        # List cases
        antenna_names = self.uv_object.antenna_names.value
        self.uv_object.antenna_names.value = [1] * self.uv_object.antenna_names.expected_size(self.uv_object)[0]
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.antenna_names.value = antenna_names  # reset
        self.assertTrue(self.uv_object.check())


class TestReadUVFits(unittest.TestCase):
    def test_ReadNRAO(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        UV = UVData()
        self.assertRaises(ValueError, UV.read, testfile, 'vufits')  # Wrong filetype
        test = UV.read(testfile, 'uvfits')
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
        UV.read(testfile, 'uvfits')

        write_file = op.join(self.test_file_directory,
                             'outtest_casa_1src_1spw.uvfits')

        test = UV.write(write_file)
        self.assertTrue(test)
        del(UV)

    def test_spwnotsupported(self):
        # testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        # testfile = '../data/PRISim_output_manual_conversion.uvfits'
        testfile = '../data/day2_TDEM0003_10s_norx_1scan.uvfits'
        self.assertTrue(os.path.exists(testfile))
        UV = UVData()
        self.assertRaises(ValueError, UV.read, testfile, 'uvfits')
        del(UV)

    def test_readwriteread(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        uv_in = UVData()
        uv_out = UVData()

        uv_in.read(testfile, 'uvfits')

        write_file = op.join(self.test_file_directory,
                             'outtest_casa.uvfits')

        uv_in.write(write_file)

        uv_out.read(write_file, 'uvfits')

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
                             spoof_nonessential=True)

        uvfits_uv.read(op.join(self.test_file_directory,
                               'outtest_FHD_1061316296.uvfits'),'uvfits')

        self.assertEqual(fhd_uv, uvfits_uv)

        del(fhd_uv)
        del(uvfits_uv)

    def test_ReadFHD_model(self):

        fhd_uv = UVData()
        uvfits_uv = UVData()
        fhd_uv.read(self.testfiles, 'fhd', use_model=True)

        fhd_uv.write(op.join(self.test_file_directory,
                             'outtest_FHD_1061316296_model.uvfits'),
                             spoof_nonessential=True)

        uvfits_uv.read(op.join(self.test_file_directory,
                       'outtest_FHD_1061316296_model.uvfits'), 'uvfits')

        self.assertEqual(fhd_uv, uvfits_uv)

        del(fhd_uv)
        del(uvfits_uv)


class TestReadMiriad(unittest.TestCase):
    def setUp(self):
        self.datafile = '../data/zen.2456865.60537.xy.uvcRREAA'
        if not os.path.exists(self.datafile):
            raise(IOError, 'miriad file not found')
        self.miriad_uv = UVData()
        self.uvfits_uv = UVData()
        self.test_file_directory = '../data/test/'

    def test_ReadMiriad(self):
        status = self.miriad_uv.read(self.datafile, 'miriad')
        self.assertTrue(status)

        # Test loop with writing/reading uvfits
        uvfits_testfile = op.join(self.test_file_directory,
                                  'outtest_miriad.uvfits')
        # Simultaneously test the general write function for case of uvfits
        self.miriad_uv.write(uvfits_testfile, spoof_nonessential=True,
                             force_phase=True)
        self.uvfits_uv.read(uvfits_testfile, 'uvfits')

        self.assertEqual(self.miriad_uv, self.uvfits_uv)

if __name__ == '__main__':
    unittest.main()
