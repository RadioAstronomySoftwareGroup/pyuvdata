import unittest
import inspect
import os
import os.path as op
import shutil
import astropy.time #necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
import numpy as np
import copy
import ephem
import warnings
import collections
from astropy.io import fits
import sys


def get_iterable(x):
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)


def clearWarnings():
    # Quick code to make warnings reproducible
    for name, mod in list(sys.modules.items()):
        try:
            reg = getattr(mod, "__warningregistry__", None)
        except ImportError:
            continue
        if reg:
            reg.clear()


def checkWarnings(obj, func, func_args=[], func_kwargs={},
                  warning_cat=UserWarning,
                  nwarnings=1, warning_message=None, known_tel_miriad=False):

    if known_tel_miriad:
        # The default warnings for known telescopes when reading miriad files
        warning_cat = [UserWarning] * 5
        warning_message = ['altitude is not set.', 'x_telescope is not set.',
                           'xyz_telescope_frame is not set.', 'y_telescope is not set.',
                           'z_telescope is not set.']
        nwarnings = 5
    warning_cat = get_iterable(warning_cat)
    warning_message = get_iterable(warning_message)

    clearWarnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # All warnings triggered
        status = func(*func_args, **func_kwargs)  # Run function
        # Verify
        if len(w) != nwarnings:
            obj.assertTrue(False)  # Fail the test, move on
        else:
            for i, w_i in enumerate(w):
                obj.assertIs(w_i.category, warning_cat[i])
                if warning_message[i] is not None:
                    obj.assertIn(warning_message[i], str(w_i.message))
    return status


test_file_directory = '../data/test/'
if not os.path.exists(test_file_directory):
    print('making test directory')
    os.mkdir(test_file_directory)


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
                                    '_instrument', '_latitude', '_longitude',
                                    '_altitude', '_history', '_vis_units',
                                    '_phase_center_epoch', '_Nants_data',
                                    '_Nants_telescope', '_antenna_names',
                                    '_antenna_indices']

        self.required_properties = ['data_array', 'nsample_array',
                                    'flag_array', 'Ntimes', 'Nbls',
                                    'Nblts', 'Nfreqs', 'Npols', 'Nspws',
                                    'uvw_array', 'time_array', 'ant_1_array',
                                    'ant_2_array', 'lst_array',
                                    'baseline_array', 'freq_array',
                                    'polarization_array', 'spw_array',
                                    'integration_time', 'channel_width',
                                    'object_name', 'telescope_name',
                                    'instrument', 'latitude', 'longitude',
                                    'altitude', 'history', 'vis_units',
                                    'phase_center_epoch', 'Nants_data',
                                    'Nants_telescope', 'antenna_names',
                                    'antenna_indices']

        self.extra_parameters = ['_extra_keywords', '_dateobs',
                                 '_xyz_telescope_frame',
                                 '_x_telescope', '_y_telescope', '_z_telescope',
                                 '_antenna_positions', '_GST0', '_RDate',
                                 '_earth_omega', '_DUT1', '_TIMESYS',
                                 '_uvplane_reference_time',
                                 '_phase_center_ra', '_phase_center_dec',
                                 '_zenith_ra', '_zenith_dec']

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
            self.assertEqual(rand_num, this_param.value)


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
                                       warning_message='found > 256 antennas'),
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
            self.uv_object.read(self.testfile, 'uvfits')
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
        self.uv_object._latitude.set_degrees(-26.7)
        self.uv_object._longitude.set_degrees(116.7)
        self.uv_object.altitude = None
        # Test that exception is raised.
        self.assertRaises(ValueError, self.uv_object.set_XYZ_from_LatLonAlt)
        self.uv_object.altitude = 377.8
        status = self.uv_object.set_XYZ_from_LatLonAlt()
        # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        # to give additional precision.
        ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)
        out_xyz = (self.uv_object.x_telescope,
                   self.uv_object.y_telescope,
                   self.uv_object.z_telescope)
        self.assertTrue(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))

    def test_set_LatLonAlt_from_XYZ(self):
        self.uv_object.xyz_telescope_frame = 'ITRF'
        self.uv_object.x_telescope = -2562123.42683
        self.uv_object.y_telescope = 5094215.40141
        self.uv_object.z_telescope = None
        # Test that exception is raised.
        self.assertRaises(ValueError, self.uv_object.set_LatLonAlt_from_XYZ)
        self.uv_object.z_telescope = -2848728.58869
        status = self.uv_object.set_LatLonAlt_from_XYZ()
        # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
        # to give additional precision.
        ref_latlonalt = (-26.7, 116.7, 377.8)
        out_latlonalt = (self.uv_object._latitude.degrees(),
                         self.uv_object._longitude.degrees(),
                         self.uv_object.altitude)
        self.assertTrue(np.allclose(ref_latlonalt, out_latlonalt, rtol=0,
                                    atol=1e-3))

    def test_check(self):
        try:
            self.uv_object.check()
        except ValueError:
            self.uv_object.read(self.testfile, 'uvfits')
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


class TestReadUVFits(unittest.TestCase):
    def test_ReadNRAO(self):
        testfile = '../data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
        testfile_no_spw = '../data/zen.2456865.60537.xy.uvcRREAAM.uvfits'
        UV = UVData()
        self.assertRaises(ValueError, UV.read, testfile, 'vufits')  # Wrong filetype
        test = UV.read(testfile, 'uvfits')
        self.assertTrue(test)
        expected_extra_keywords = ['OBSERVER', 'SORTORD', 'SPECSYS',
                                   'RESTFREQ', 'ORIGIN']
        self.assertEqual(expected_extra_keywords.sort(),
                         UV.extra_keywords.keys().sort())

        del(UV)
        UV = UVData()
        test = checkWarnings(self, UV.read, [testfile_no_spw, 'uvfits'],
                             warning_message=['Required Antenna frame keyword'
                                              ' not set, since this is a PAPER'
                                              ' file, setting to ITRF',
                                              'altitude is not set.',
                                              'latitude is not set.',
                                              'longitude is not set.'],
                             warning_cat=[UserWarning] * 4,
                             nwarnings=4)
        self.assertTrue(test)

        del(UV)

    def test_breaks(self):
        # Here I'll group together tests that should raise exceptions
        multi_subarray_file = '../data/test/multi_subarray.uvfits'
        if not os.path.exists(multi_subarray_file):
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
        UV.read(testfile, 'uvfits')

        write_file = op.join(self.test_file_directory,
                             'outtest_casa_1src_1spw.uvfits')

        test = UV.write(write_file, file_type='uvfits')
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

        uv_in.write(write_file, file_type='uvfits')

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
                                      'fhd'], warning_message='No settings'))
        self.assertEqual(fhd_uv.history, '')  # Check empty history with no settings
        del(fhd_uv)

    def test_ReadFHD_model(self):

        fhd_uv = UVData()
        uvfits_uv = UVData()
        fhd_uv.read(self.testfiles, 'fhd', use_model=True)

        fhd_uv.write(op.join(self.test_file_directory,
                             'outtest_FHD_1061316296_model.uvfits'),
                     file_type='uvfits',
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

        self.unphasedfile = '../data/new.uvA'
        if not os.path.exists(self.unphasedfile):
            raise(IOError, 'miriad file not found')
        self.unphased = UVData()

        self.phasedfile = '../data/new.uvA.phased'
        if not os.path.exists(self.phasedfile):
            raise(IOError, 'miriad file not found')
        self.phased = UVData()

        self.test_file_directory = '../data/test/'

        status = checkWarnings(self, self.miriad_uv.read, [self.datafile, 'miriad'],
                               known_tel_miriad=True)

        self.assertTrue(status)

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
        status = checkWarnings(self, self.unphased.read, [self.unphasedfile, 'miriad'],
                               known_tel_miriad=True)
        self.unphased.phase(ra=0.0, dec=0.0, epoch=ephem.J2000)
        status = checkWarnings(self, self.phased.read, [self.phasedfile, 'miriad'],
                               known_tel_miriad=True)

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
        status = checkWarnings(self, UV.read, [testfile, 'miriad'],
                               known_tel_miriad=True)

        write_file = op.join(self.test_file_directory,
                             'outtest_miriad.uv')

        test = UV.write(write_file, file_type='miriad', clobber=True)
        self.assertTrue(test)
        del(UV)

    def test_readwriteread(self):
        testfile = '../data/zen.2456865.60537.xy.uvcRREAA'
        uv_in = UVData()
        uv_out = UVData()

        status = checkWarnings(self, uv_in.read, [testfile, 'miriad'],
                               known_tel_miriad=True)

        write_file = op.join(self.test_file_directory,
                             'outtest_miriad.uv')

        uv_in.write(write_file, file_type='miriad', clobber=True)

        status = checkWarnings(self, uv_out.read, [write_file, 'miriad'],
                               known_tel_miriad=True)

        self.assertEqual(uv_in, uv_out)
        del(uv_in)
        del(uv_out)

class TestPhase(unittest.TestCase):
    def setUp(self):
        self.test_file_directory = '../data/test/'
        if not os.path.exists(self.test_file_directory):
            print('making test directory')
            os.mkdir(self.test_file_directory)

    def test_phase_unphasePAPER(self):
        testfile = '../data/zen.2456865.60537.xy.uvcRREAA'
        UV_raw = UVData()
        UV_raw.read(testfile, 'miriad')

        UV_phase = UVData()
        UV_phase.read(testfile, 'miriad')
        UV_phase.phase(ra=0.,dec=0.,epoch=ephem.J2000)
        UV_phase.unphase_to_drift()       
 
        self.assertEqual(UV_raw,UV_phase)
        del(UV_phase)
        del(UV_raw)

if __name__ == '__main__':
    unittest.main()
