import unittest
import os
import os.path as op
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
import numpy as np
import copy
import ephem
from test_functions import *


class TestUVDataInit(unittest.TestCase):
    def setUp(self):
        self.required_parameters = ['_data_array', '_nsample_array',
                                    '_flag_array', '_Ntimes', '_Nbls',
                                    '_Nblts', '_Nfreqs', '_Npols', '_Nspws',
                                    '_uvw_array', '_time_array', '_ant_1_array',
                                    '_ant_2_array', '_lst_array',
                                    '_baseline_array', '_freq_array',
                                    '_polarization_array', '_spw_array',
                                    '_integration_time', '_channel_width',
                                    '_object_name', '_telescope_name',
                                    '_instrument', '_telescope_location',
                                    '_history', '_vis_units',
                                    '_phase_center_epoch', '_Nants_data',
                                    '_Nants_telescope', '_antenna_names',
                                    '_antenna_numbers']

        self.required_properties = ['data_array', 'nsample_array',
                                    'flag_array', 'Ntimes', 'Nbls',
                                    'Nblts', 'Nfreqs', 'Npols', 'Nspws',
                                    'uvw_array', 'time_array', 'ant_1_array',
                                    'ant_2_array', 'lst_array',
                                    'baseline_array', 'freq_array',
                                    'polarization_array', 'spw_array',
                                    'integration_time', 'channel_width',
                                    'object_name', 'telescope_name',
                                    'instrument', 'telescope_location',
                                    # 'telescope_location_lat_lon_alt',
                                    # 'telescope_location_lat_lon_alt_degrees'
                                    'history', 'vis_units',
                                    'phase_center_epoch', 'Nants_data',
                                    'Nants_telescope', 'antenna_names',
                                    'antenna_numbers']

        self.extra_parameters = ['_extra_keywords', '_dateobs',
                                 '_antenna_positions', '_GST0', '_RDate',
                                 '_earth_omega', '_DUT1', '_TIMESYS',
                                 '_uvplane_reference_time',
                                 '_phase_center_ra', '_phase_center_dec',
                                 '_zenith_ra', '_zenith_dec']

        self.extra_properties = ['extra_keywords', 'dateobs',
                                 'antenna_positions', 'GST0', 'RDate',
                                 'earth_omega', 'DUT1', 'TIMESYS',
                                 'uvplane_reference_time',
                                 #  'phase_center_ra_degrees', 'phase_center_dec_degrees',
                                 #  'zenith_ra_degrees', 'zenith_dec_degrees',
                                 'phase_center_ra', 'phase_center_dec',
                                 'zenith_ra', 'zenith_dec']

        self.known_telescopes = ['PAPER', 'HERA', 'MWA']

        self.uv_object = UVData()

    def tearDown(self):
        del(self.uv_object)

    def test_parameter_iter(self):
        all = []
        for prop in self.uv_object.parameter_iter():
            all.append(prop)
        for a in self.required_parameters + self.extra_parameters:
            self.assertTrue(a in all,
                            msg='expected attribute ' + a +
                            ' not returned in parameter_iter')

    def test_required_parameter_iter(self):
        required = []
        for prop in self.uv_object.required_parameter_iter():
            required.append(prop)
        for a in self.required_parameters:
            self.assertTrue(a in required,
                            msg='expected attribute ' + a +
                            ' not returned in required_parameter_iter')

    def test_extra_parameter_iter(self):
        extra = []
        for prop in self.uv_object.extra_parameter_iter():
            extra.append(prop)
        for a in self.extra_parameters:
            self.assertTrue(a in extra,
                            msg='expected attribute ' + a +
                            ' not returned in extra_parameter_iter')

    def test_parameters_exist(self):
        expected_parameters = self.required_parameters + self.extra_parameters
        for a in expected_parameters:
            self.assertTrue(hasattr(self.uv_object, a),
                            msg='expected parameter ' + a + ' does not exist')

    def test_unexpected_attributes(self):
        expected_attributes = self.required_parameters + \
            self.extra_parameters + self.required_properties + \
            self.extra_properties
        attributes = [i for i in self.uv_object.__dict__.keys() if i[0] != '_']
        for a in attributes:
            self.assertTrue(a in expected_attributes,
                            msg='unexpected attribute ' + a +
                            ' found in UVData')

    def test_properties(self):
        prop_dict = dict(zip(self.required_properties + self.extra_properties,
                             self.required_parameters + self.extra_parameters))
        for k, v in prop_dict.iteritems():
            rand_num = np.random.rand()
            setattr(self.uv_object, k, rand_num)
            this_param = getattr(self.uv_object, v)
            try:
                self.assertEqual(rand_num, this_param.value)
            except:
                print('setting {prop_name} to a random number failed'.format(prop_name=k))
                raise(AssertionError)

    def test_known_telescopes(self):
        self.assertEqual(self.known_telescopes.sort(),
                         self.uv_object.known_telescopes().sort())


class TestUVmethods(unittest.TestCase):
    def setUp(self):
        self.uv_object = UVData()
        self.uv_object.Nants_telescope = 128
        self.testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'

    def tearDown(self):
        del(self.uv_object)

    def test_bl2ij(self):
        self.assertEqual(self.uv_object.baseline_to_antnums(67585),
                         (0, 0))
        Nants = self.uv_object.Nants_telescope
        self.uv_object.Nants_telescope = 2049
        self.assertRaises(StandardError, self.uv_object.baseline_to_antnums,
                          67585)
        self.uv_object.Nants_telescope = Nants  # reset

    def test_ij2bl(self):
        self.assertEqual(self.uv_object.antnums_to_baseline(0, 0),
                         67585)
        self.assertEqual(self.uv_object.antnums_to_baseline(257, 256),
                         592130)
        # Check attempt256
        self.assertEqual(self.uv_object.antnums_to_baseline(0, 0,
                         attempt256=True), 257)
        self.assertEqual(checkWarnings(self, self.uv_object.antnums_to_baseline,
                                       [257, 256], {'attempt256': True},
                                       message='found > 256 antennas'),
                         592130)
        Nants = self.uv_object.Nants_telescope
        self.uv_object.Nants_telescope = 2049
        self.assertRaises(StandardError, self.uv_object.antnums_to_baseline,
                          0, 0)
        self.uv_object.Nants_telescope = Nants  # reset

    def test_data_equality(self):
        try:
            self.uv_object.check()
        except ValueError:
            checkWarnings(self, self.uv_object.read, [self.testfile, 'uvfits'],
                          message='Telescope EVLA is not')
        self.assertEqual(self.uv_object, self.uv_object)
        self.uv_object2 = copy.deepcopy(self.uv_object)
        self.uv_object2.data_array[0, 0, 0, 0] += 1  # Force data to be not equal
        self.assertNotEqual(self.uv_object, self.uv_object2)
        # check class equality test
        self.assertNotEqual(self.uv_object, self.uv_object.data_array)

        # Check some UVParameter specific inequalities.
        self.uv_object2.data_array = 1.0  # Test values not same class
        # Note that due to peculiarity of order of operations, need to reverse arrays.
        self.assertNotEqual(self.uv_object2._data_array,
                            self.uv_object._data_array)
        self.uv_object2.data_array = np.array([1, 2, 3])  # Test different shapes
        self.assertNotEqual(self.uv_object._data_array,
                            self.uv_object2._data_array)
        self.uv_object2.Ntimes = 1000.0  # Test values that are not close
        self.assertNotEqual(self.uv_object._Ntimes, self.uv_object2._Ntimes)
        self.uv_object2.vis_units = 'foo'  # Test unequal strings
        self.assertNotEqual(self.uv_object._vis_units,
                            self.uv_object2._vis_units)
        self.uv_object2.antenna_names[0] = 'Bob'  # Test unequal string in list
        self.assertNotEqual(self.uv_object._antenna_names,
                            self.uv_object2._antenna_names)

    def test_set_XYZ_from_LatLonAlt(self):
        self.uv_object.telescope_location_lat_lon_alt_degrees = (-26.7, 116.7, 377.8)
        # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        # to give additional precision.
        ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)
        out_xyz = self.uv_object.telescope_location
        self.assertTrue(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))

    def test_set_LatLonAlt_from_XYZ(self):
        self.uv_object.telescope_location = np.array([-2562123.42683, 5094215.40141, -2848728.58869])
        # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        # to give additional precision.
        ref_latlonalt = (-26.7, 116.7, 377.8)
        out_latlonalt = self.uv_object.telescope_location_lat_lon_alt_degrees
        self.assertTrue(np.allclose(ref_latlonalt, out_latlonalt, rtol=0,
                                    atol=1e-3))

    def test_check(self):
        try:
            self.uv_object.check()
        except ValueError:
            checkWarnings(self, self.uv_object.read, [self.testfile, 'uvfits'],
                          message='Telescope EVLA is not')
        self.assertTrue(self.uv_object.check())
        # Now break it in every way I can.
        # String cases
        units = self.uv_object.vis_units
        self.uv_object.vis_units = 1
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.vis_units = units  # reset it
        # Single value cases
        Nblts = self.uv_object.Nblts
        self.uv_object.Nblts = 4
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.Nblts = np.float(Nblts)
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.Nblts = Nblts  # reset
        # Array cases
        data = self.uv_object.data_array
        self.uv_object.data_array = np.array([4, 5, 6], dtype=np.complex64)
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.data_array = np.real(data)
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.data_array = data  # reset
        # List cases
        antenna_names = self.uv_object.antenna_names
        self.uv_object.antenna_names = [1] * self.uv_object._antenna_names.expected_size(self.uv_object)[0]
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.antenna_names = antenna_names  # reset
        # Sanity check
        uvws = self.uv_object.uvw_array
        self.uv_object.uvw_array = 1e-4 * np.ones_like(self.uv_object.uvw_array)
        self.assertRaises(ValueError, self.uv_object.check)
        self.uv_object.uvw_array = uvws
        self.assertTrue(self.uv_object.check())


class TestPhase(unittest.TestCase):
    def setUp(self):
        self.test_file_directory = '../data/test/'
        if not op.exists(self.test_file_directory):
            print('making test directory')
            os.mkdir(self.test_file_directory)

    def test_phase_unphasePAPER(self):
        testfile = '../data/zen.2456865.60537.xy.uvcRREAA'
        UV_raw = UVData()
        status = checkWarnings(self, UV_raw.read, [testfile, 'miriad'],
                               known_warning='miriad')

        UV_phase = UVData()
        status = checkWarnings(self, UV_phase.read, [testfile, 'miriad'],
                               known_warning='miriad')
        UV_phase.phase(ra=0., dec=0., epoch=ephem.J2000)
        UV_phase.unphase_to_drift()

        self.assertEqual(UV_raw, UV_phase)
        del(UV_phase)
        del(UV_raw)
