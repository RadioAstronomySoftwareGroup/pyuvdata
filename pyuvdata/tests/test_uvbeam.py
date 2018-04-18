"""Tests for uvbeam object."""
import nose.tools as nt
import os
import numpy as np
import copy
import healpy as hp
from pyuvdata import UVBeam
import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
import pyuvdata.version as uvversion
from pyuvdata.data import DATA_PATH

filenames = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
cst_files = [os.path.join(DATA_PATH, f) for f in filenames]


def fill_dummy_beam(beam_obj, beam_type, pixel_coordinate_system):
    # this should only be used for HEALPix Efield beams because those cannot
    # yet be constructed from the test data.
    if beam_type != 'efield' or pixel_coordinate_system != 'healpix':
        raise ValueError('Use actual test data files for beams that are not Healpix efield beams.')

    beam_obj.set_simple()
    beam_obj.telescope_name = 'testscope'
    beam_obj.feed_name = 'testfeed'
    beam_obj.feed_version = '0.1'
    beam_obj.model_name = 'testmodel'
    beam_obj.model_version = '1.0'
    beam_obj.history = 'random data for test'

    pyuvdata_version_str = ('  Read/written with pyuvdata version: '
                            + uvversion.version + '.')
    if uvversion.git_hash is not '':
        pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin
                                 + '.  Git hash: ' + uvversion.git_hash
                                 + '.  Git branch: ' + uvversion.git_branch
                                 + '.  Git description: ' + uvversion.git_description + '.')
    beam_obj.history += pyuvdata_version_str

    beam_obj.pixel_coordinate_system = pixel_coordinate_system
    if pixel_coordinate_system == 'healpix':
        beam_obj.nside = 512
        beam_obj.ordering = 'ring'
        beam_obj.pixel_array = np.arange(0, 6 * (beam_obj.nside) ^ 2)
        beam_obj.Npixels = len(beam_obj.pixel_array)
    elif pixel_coordinate_system == 'az_za':
        beam_obj.axis1_array = np.radians(np.arange(-180.0, 180.0, 5.0))
        beam_obj.Naxes1 = len(beam_obj.axis1_array)
        beam_obj.axis2_array = np.radians(np.arange(0, 90.0, 5.0))
        beam_obj.Naxes2 = len(beam_obj.axis2_array)

    beam_obj.freq_array = np.arange(150e6, 160e6, 1e5)
    beam_obj.freq_array = beam_obj.freq_array[np.newaxis, :]
    beam_obj.Nfreqs = beam_obj.freq_array.shape[1]
    beam_obj.spw_array = np.array([0])
    beam_obj.Nspws = len(beam_obj.spw_array)
    beam_obj.data_normalization = 'peak'
    beam_obj.bandpass_array = np.zeros((beam_obj.Nspws, beam_obj.Nfreqs)) + 1.

    if beam_type == 'power':
        beam_obj.set_power()
        beam_obj.polarization_array = np.array([-5, -6, -7, -8])
        beam_obj.Npols = len(beam_obj.polarization_array)
        beam_obj.Naxes_vec = 1

        if pixel_coordinate_system == 'healpix':
            data_size_tuple = (beam_obj.Naxes_vec, beam_obj.Nspws, beam_obj.Npols,
                               beam_obj.Nfreqs, beam_obj.Npixels)
        else:
            data_size_tuple = (beam_obj.Naxes_vec, beam_obj.Nspws, beam_obj.Npols,
                               beam_obj.Nfreqs, beam_obj.Naxes2, beam_obj.Naxes1)

        beam_obj.data_array = np.square(np.random.normal(0.0, 0.2, size=data_size_tuple))
    else:
        beam_obj.set_efield()
        beam_obj.feed_array = np.array(['x', 'y'])
        beam_obj.Nfeeds = len(beam_obj.feed_array)
        beam_obj.Naxes_vec = 2

        if pixel_coordinate_system == 'healpix':
            beam_obj.basis_vector_array = np.zeros((beam_obj.Naxes_vec, 2,
                                                    beam_obj.Npixels))
            beam_obj.basis_vector_array[0, 0, :] = 1.0
            beam_obj.basis_vector_array[1, 1, :] = 1.0

            data_size_tuple = (beam_obj.Naxes_vec, beam_obj.Nspws, beam_obj.Nfeeds,
                               beam_obj.Nfreqs, beam_obj.Npixels)
        else:
            beam_obj.basis_vector_array = np.zeros((beam_obj.Naxes_vec, 2,
                                                    beam_obj.Naxes2, beam_obj.Naxes1))
            beam_obj.basis_vector_array[0, 0, :, :] = 1.0
            beam_obj.basis_vector_array[1, 1, :, :] = 1.0

            data_size_tuple = (beam_obj.Naxes_vec, beam_obj.Nspws, beam_obj.Nfeeds,
                               beam_obj.Nfreqs, beam_obj.Naxes2, beam_obj.Naxes1)

        beam_obj.data_array = (np.random.normal(0.0, 0.2, size=data_size_tuple)
                               + 1j * np.random.normal(0.0, 0.2, size=data_size_tuple))

    # add optional parameters for testing purposes
    beam_obj.extra_keywords = {'KEY1': 'test_keyword'}
    beam_obj.reference_input_impedance = 340.
    beam_obj.reference_output_impedance = 50.
    beam_obj.receiver_temperature_array = np.random.normal(50.0, 5, size=(beam_obj.Nspws, beam_obj.Nfreqs))
    beam_obj.loss_array = np.random.normal(50.0, 5, size=(beam_obj.Nspws, beam_obj.Nfreqs))
    beam_obj.mismatch_array = np.random.normal(0.0, 1.0, size=(beam_obj.Nspws, beam_obj.Nfreqs))
    beam_obj.s_parameters = np.random.normal(0.0, 0.3, size=(4, beam_obj.Nspws, beam_obj.Nfreqs))

    return beam_obj


class TestUVBeamInit(object):
    def setUp(self):
        """Setup for basic parameter, property and iterator tests."""
        self.required_parameters = ['_beam_type', '_Nfreqs', '_Naxes_vec', '_Nspws',
                                    '_pixel_coordinate_system',
                                    '_freq_array', '_spw_array',
                                    '_data_normalization',
                                    '_data_array', '_bandpass_array',
                                    '_telescope_name', '_feed_name',
                                    '_feed_version', '_model_name',
                                    '_model_version', '_history',
                                    '_antenna_type']

        self.required_properties = ['beam_type', 'Nfreqs', 'Naxes_vec', 'Nspws',
                                    'pixel_coordinate_system',
                                    'freq_array', 'spw_array',
                                    'data_normalization',
                                    'data_array', 'bandpass_array',
                                    'telescope_name', 'feed_name',
                                    'feed_version', 'model_name',
                                    'model_version', 'history',
                                    'antenna_type']

        self.extra_parameters = ['_Naxes1', '_Naxes2', '_Npixels', '_Nfeeds', '_Npols',
                                 '_axis1_array', '_axis2_array', '_nside', '_ordering',
                                 '_pixel_array', '_feed_array', '_polarization_array',
                                 '_basis_vector_array',
                                 '_extra_keywords', '_Nelements',
                                 '_element_coordinate_system',
                                 '_element_location_array', '_delay_array',
                                 '_gain_array', '_coupling_matrix',
                                 '_reference_input_impedance', '_reference_output_impedance',
                                 '_receiver_temperature_array',
                                 '_loss_array', '_mismatch_array',
                                 '_s_parameters']

        self.extra_properties = ['Naxes1', 'Naxes2', 'Npixels', 'Nfeeds', 'Npols',
                                 'axis1_array', 'axis2_array', 'nside', 'ordering',
                                 'pixel_array', 'feed_array', 'polarization_array',
                                 'basis_vector_array', 'extra_keywords', 'Nelements',
                                 'element_coordinate_system',
                                 'element_location_array', 'delay_array',
                                 'gain_array', 'coupling_matrix',
                                 'reference_input_impedance', 'reference_output_impedance',
                                 'receiver_temperature_array',
                                 'loss_array', 'mismatch_array',
                                 's_parameters']

        self.other_properties = ['pyuvdata_version_str']

        self.beam_obj = UVBeam()

    def teardown(self):
        """Test teardown: delete object."""
        del(self.beam_obj)

    def test_parameter_iter(self):
        "Test expected parameters."
        all = []
        for prop in self.beam_obj:
            all.append(prop)
        for a in self.required_parameters + self.extra_parameters:
            nt.assert_true(a in all, msg='expected attribute ' + a
                           + ' not returned in object iterator')

    def test_required_parameter_iter(self):
        "Test expected required parameters."
        required = []
        for prop in self.beam_obj.required():
            required.append(prop)
        for a in self.required_parameters:
            nt.assert_true(a in required, msg='expected attribute ' + a
                           + ' not returned in required iterator')

    def test_extra_parameter_iter(self):
        "Test expected optional parameters."
        extra = []
        for prop in self.beam_obj.extra():
            extra.append(prop)
        for a in self.extra_parameters:
            nt.assert_true(a in extra, msg='expected attribute ' + a
                           + ' not returned in extra iterator')

    def test_unexpected_parameters(self):
        "Test for extra parameters."
        expected_parameters = self.required_parameters + self.extra_parameters
        attributes = [i for i in self.beam_obj.__dict__.keys() if i[0] == '_']
        for a in attributes:
            nt.assert_true(a in expected_parameters,
                           msg='unexpected parameter ' + a + ' found in UVData')

    def test_unexpected_attributes(self):
        "Test for extra attributes."
        expected_attributes = self.required_properties + \
            self.extra_properties + self.other_properties
        attributes = [i for i in self.beam_obj.__dict__.keys() if i[0] != '_']
        for a in attributes:
            nt.assert_true(a in expected_attributes,
                           msg='unexpected attribute ' + a + ' found in UVData')

    def test_properties(self):
        "Test that properties can be get and set properly."
        prop_dict = dict(zip(self.required_properties + self.extra_properties,
                             self.required_parameters + self.extra_parameters))
        for k, v in prop_dict.iteritems():
            rand_num = np.random.rand()
            setattr(self.beam_obj, k, rand_num)
            this_param = getattr(self.beam_obj, v)
            try:
                nt.assert_equal(rand_num, this_param.value)
            except(AssertionError):
                print('setting {prop_name} to a random number failed'.format(prop_name=k))
                raise(AssertionError)


def test_errors():
    beam_obj = UVBeam()
    nt.assert_raises(ValueError, beam_obj._convert_to_filetype, 'foo')


def test_peak_normalize():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    orig_bandpass_array = copy.deepcopy(efield_beam.bandpass_array)
    maxima = np.zeros(efield_beam.Nfreqs)
    for freq_i in range(efield_beam.Nfreqs):
        maxima[freq_i] = np.amax(abs(efield_beam.data_array[:, :, :, freq_i]))
    efield_beam.peak_normalize()
    nt.assert_equal(np.amax(abs(efield_beam.data_array)), 1)
    nt.assert_equal(np.sum(abs(efield_beam.bandpass_array - orig_bandpass_array * maxima)), 0)
    nt.assert_equal(efield_beam.data_normalization, 'peak')

    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    orig_bandpass_array = copy.deepcopy(power_beam.bandpass_array)
    maxima = np.zeros(efield_beam.Nfreqs)
    for freq_i in range(efield_beam.Nfreqs):
        maxima[freq_i] = np.amax(power_beam.data_array[:, :, :, freq_i])
    power_beam.peak_normalize()
    nt.assert_equal(np.amax(abs(power_beam.data_array)), 1)
    nt.assert_equal(np.sum(abs(power_beam.bandpass_array - orig_bandpass_array * maxima)), 0)
    nt.assert_equal(power_beam.data_normalization, 'peak')

    power_beam.data_normalization = 'solid_angle'
    nt.assert_raises(NotImplementedError, power_beam.peak_normalize)


def test_efield_to_power():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')

    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    new_power_beam = copy.deepcopy(efield_beam)
    new_power_beam.efield_to_power(calc_cross_pols=False)

    # The values in the beam file only have 4 sig figs, so they don't match precisely
    diff = np.abs(new_power_beam.data_array - power_beam.data_array)
    reldiff = diff / power_beam.data_array
    nt.assert_true(np.max(reldiff) < 0.002)

    # set data_arrays equal to test the rest of the object
    power_beam.data_array = new_power_beam.data_array
    nt.assert_equal(new_power_beam, power_beam)

    # test with non-orthogonal basis vectors
    # first construct a beam with non-orthogonal basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[0, 1, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, :, :, :] = efield_beam.basis_vector_array[1, :, :, :]
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, :, :, :, :, :] = np.sqrt(2) * efield_beam.data_array[0, :, :, :, :, :]
    new_data[1, :, :, :, :, :] = (efield_beam.data_array[1, :, :, :, :, :]
                                  - efield_beam.data_array[0, :, :, :, :, :])
    efield_beam2 = copy.deepcopy(efield_beam)
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = copy.deepcopy(efield_beam2)
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    nt.assert_equal(new_power_beam, new_power_beam2)

    # now try a different rotation to non-orthogonal basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, :, :, :] = efield_beam.basis_vector_array[0, :, :, :]
    new_basis_vecs[1, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, 1, :, :] = np.sqrt(0.5)
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, :, :, :, :, :] = (efield_beam.data_array[0, :, :, :, :, :]
                                  - efield_beam.data_array[1, :, :, :, :, :])
    new_data[1, :, :, :, :, :] = np.sqrt(2) * efield_beam.data_array[1, :, :, :, :, :]
    efield_beam2 = copy.deepcopy(efield_beam)
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = copy.deepcopy(efield_beam2)
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    nt.assert_equal(new_power_beam, new_power_beam2)

    # now construct a beam with  orthogonal but rotated basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[0, 1, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, 0, :, :] = -1 * np.sqrt(0.5)
    new_basis_vecs[1, 1, :, :] = np.sqrt(0.5)
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, :, :, :, :, :] = np.sqrt(0.5) * (efield_beam.data_array[0, :, :, :, :, :]
                                                 + efield_beam.data_array[1, :, :, :, :, :])
    new_data[1, :, :, :, :, :] = np.sqrt(0.5) * (-1 * efield_beam.data_array[0, :, :, :, :, :]
                                                 + efield_beam.data_array[1, :, :, :, :, :])
    efield_beam2 = copy.deepcopy(efield_beam)
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = copy.deepcopy(efield_beam2)
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    nt.assert_equal(new_power_beam, new_power_beam2)

    # test calculating cross pols
    new_power_beam = copy.deepcopy(efield_beam)
    new_power_beam.efield_to_power(calc_cross_pols=True)
    nt.assert_true(np.all(np.abs(new_power_beam.data_array[:, :, 0, :, :,
                                                           np.where(new_power_beam.axis1_array == 0)[0]])
                          > np.abs(new_power_beam.data_array[:, :, 2, :, :,
                                                             np.where(new_power_beam.axis1_array == 0)[0]])))
    nt.assert_true(np.all(np.abs(new_power_beam.data_array[:, :, 0, :, :,
                                                           np.where(new_power_beam.axis1_array == np.pi / 2.)[0]])
                          > np.abs(new_power_beam.data_array[:, :, 2, :, :,
                                                             np.where(new_power_beam.axis1_array == np.pi / 2.)[0]])))
    # test writing out & reading back in power files (with cross pols which are complex)
    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')
    new_power_beam.write_beamfits(write_file, clobber=True)
    new_power_beam2 = UVBeam()
    new_power_beam2.read_beamfits(write_file)
    nt.assert_equal(new_power_beam, new_power_beam2)

    # test keeping basis vectors
    new_power_beam = copy.deepcopy(efield_beam)
    new_power_beam.efield_to_power(calc_cross_pols=False, keep_basis_vector=True)
    nt.assert_true(np.allclose(new_power_beam.data_array, np.abs(efield_beam.data_array)**2))

    # test raises error if beam is already a power beam
    nt.assert_raises(ValueError, power_beam.efield_to_power)

    # test raises error if input efield beam has Naxes_vec=3
    efield_beam.Naxes_vec = 3
    nt.assert_raises(ValueError, efield_beam.efield_to_power)

    # TODO: add testing in healpix once we can convert efield beams to healpix


def test_az_za_to_healpix():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    power_beam.select(axis2_inds=np.where(power_beam.axis2_array <= np.pi / 2.)[0])

    power_beam_healpix = copy.deepcopy(power_beam)
    power_beam_healpix.az_za_to_healpix()

    npix = hp.nside2npix(power_beam_healpix.nside)
    nt.assert_true(power_beam_healpix.Npixels <= npix * 0.55)

    # Test error if not az_za
    power_beam.pixel_coordinate_system = 'sin_zenith'
    nt.assert_raises(ValueError, power_beam.az_za_to_healpix)


def test_select_axis():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_input_impedance = 340.
    power_beam.reference_output_impedance = 50.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    old_history = power_beam.history
    # Test selecting on axis1
    inds1_to_keep = np.arange(14, 63)

    power_beam2 = power_beam.select(axis1_inds=inds1_to_keep, inplace=False)

    nt.assert_equal(len(inds1_to_keep), power_beam2.Naxes1)
    for i in inds1_to_keep:
        nt.assert_true(power_beam.axis1_array[i] in power_beam2.axis1_array)
    for i in np.unique(power_beam2.axis1_array):
        nt.assert_true(i in power_beam.axis1_array)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific parts of first image axis '
                                           'using pyuvdata.', power_beam2.history))

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

    # test writing beamfits with only one element in axis1
    inds_to_keep = [len(inds1_to_keep) + 1]
    power_beam2 = power_beam.select(axis1_inds=inds_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with indices not included in data
    nt.assert_raises(ValueError, power_beam2.select, axis1_inds=[power_beam.Naxes1 - 1])

    # check for warnings and errors associated with unevenly spaced image pixels
    power_beam2 = copy.deepcopy(power_beam)
    uvtest.checkWarnings(power_beam2.select, [], {'axis1_inds': [0, 5, 6]},
                         message='Selected values along first image axis are not evenly spaced')
    nt.assert_raises(ValueError, power_beam2.write_beamfits, write_file_beamfits)

    # Test selecting on axis2
    inds2_to_keep = np.arange(5, 14)

    power_beam2 = power_beam.select(axis2_inds=inds2_to_keep, inplace=False)

    nt.assert_equal(len(inds2_to_keep), power_beam2.Naxes2)
    for i in inds2_to_keep:
        nt.assert_true(power_beam.axis2_array[i] in power_beam2.axis2_array)
    for i in np.unique(power_beam2.axis2_array):
        nt.assert_true(i in power_beam.axis2_array)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific parts of second image axis '
                                           'using pyuvdata.', power_beam2.history))

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

    # test writing beamfits with only one element in axis2
    inds_to_keep = [len(inds2_to_keep) + 1]
    power_beam2 = power_beam.select(axis2_inds=inds_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with indices not included in data
    nt.assert_raises(ValueError, power_beam2.select, axis2_inds=[power_beam.Naxes2 - 1])

    # check for warnings and errors associated with unevenly spaced image pixels
    power_beam2 = copy.deepcopy(power_beam)
    uvtest.checkWarnings(power_beam2.select, [], {'axis2_inds': [0, 5, 6]},
                         message='Selected values along second image axis are not evenly spaced')
    nt.assert_raises(ValueError, power_beam2.write_beamfits, write_file_beamfits)


def test_select_frequencies():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding several times
    while power_beam.Nfreqs < 8:
        new_beam = copy.deepcopy(power_beam)
        new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
        power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_input_impedance = 340.
    power_beam.reference_output_impedance = 50.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    old_history = power_beam.history
    freqs_to_keep = power_beam.freq_array[0, np.arange(2, 7)]

    power_beam2 = power_beam.select(frequencies=freqs_to_keep, inplace=False)

    nt.assert_equal(len(freqs_to_keep), power_beam2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in power_beam2.freq_array)
    for f in np.unique(power_beam2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific frequencies using pyuvdata.',
                                           power_beam2.history))

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')
    # test writing beamfits with only one frequency

    freqs_to_keep = power_beam.freq_array[0, 5]
    power_beam2 = power_beam.select(frequencies=freqs_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with frequencies not included in data
    nt.assert_raises(ValueError, power_beam.select, frequencies=[np.max(power_beam.freq_array) + 10])

    # check for warnings and errors associated with unevenly spaced frequencies
    power_beam2 = copy.deepcopy(power_beam)
    uvtest.checkWarnings(power_beam2.select, [],
                         {'frequencies': power_beam2.freq_array[0, [0, 5, 6]]},
                         message='Selected frequencies are not evenly spaced')
    nt.assert_raises(ValueError, power_beam2.write_beamfits, write_file_beamfits)

    # Test selecting on freq_chans
    chans_to_keep = np.arange(2, 7)

    power_beam2 = power_beam.select(freq_chans=chans_to_keep, inplace=False)

    nt.assert_equal(len(chans_to_keep), power_beam2.Nfreqs)
    for chan in chans_to_keep:
        nt.assert_true(power_beam.freq_array[0, chan] in power_beam2.freq_array)
    for f in np.unique(power_beam2.freq_array):
        nt.assert_true(f in power_beam.freq_array[0, chans_to_keep])

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific frequencies using pyuvdata.',
                                           power_beam2.history))

    # Test selecting both channels and frequencies
    freqs_to_keep = power_beam.freq_array[0, np.arange(6, 8)]  # Overlaps with chans
    all_chans_to_keep = np.arange(2, 8)

    power_beam2 = power_beam.select(frequencies=freqs_to_keep,
                                    freq_chans=chans_to_keep,
                                    inplace=False)

    nt.assert_equal(len(all_chans_to_keep), power_beam2.Nfreqs)
    for chan in all_chans_to_keep:
        nt.assert_true(power_beam.freq_array[0, chan] in power_beam2.freq_array)
    for f in np.unique(power_beam2.freq_array):
        nt.assert_true(f in power_beam.freq_array[0, all_chans_to_keep])


def test_select_feeds():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {'KEY1': 'test_keyword'}
    efield_beam.reference_input_impedance = 340.
    efield_beam.reference_output_impedance = 50.
    efield_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.loss_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs))

    old_history = efield_beam.history
    feeds_to_keep = ['x']

    efield_beam2 = efield_beam.select(feeds=feeds_to_keep, inplace=False)

    nt.assert_equal(len(feeds_to_keep), efield_beam2.Nfeeds)
    for f in feeds_to_keep:
        nt.assert_true(f in efield_beam2.feed_array)
    for f in np.unique(efield_beam2.feed_array):
        nt.assert_true(f in feeds_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific feeds using pyuvdata.',
                                           efield_beam2.history))

    # check for errors associated with feeds not included in data
    nt.assert_raises(ValueError, efield_beam.select, feeds=['N'])

    # check for error with selecting polarizations on efield beams
    nt.assert_raises(ValueError, efield_beam.select, polarizations=[-5, -6])

    # Test check basis vectors
    efield_beam.basis_vector_array[0, 1, :, :] = 1.0
    nt.assert_raises(ValueError, efield_beam.check)

    efield_beam.basis_vector_array[0, 0, :, :] = np.sqrt(0.5)
    efield_beam.basis_vector_array[0, 1, :, :] = np.sqrt(0.5)
    nt.assert_true(efield_beam.check())

    efield_beam.basis_vector_array = None
    nt.assert_raises(ValueError, efield_beam.check)


def test_select_polarizations():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol='xx',
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more polarizations for testing by copying and adding several times
    while power_beam.Npols < 4:
        new_beam = copy.deepcopy(power_beam)
        new_beam.polarization_array = power_beam.polarization_array - power_beam.Npols
        power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_input_impedance = 340.
    power_beam.reference_output_impedance = 50.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    old_history = power_beam.history
    pols_to_keep = [-5, -6]

    power_beam2 = power_beam.select(polarizations=pols_to_keep,
                                    inplace=False)

    nt.assert_equal(len(pols_to_keep), power_beam2.Npols)
    for p in pols_to_keep:
        nt.assert_true(p in power_beam2.polarization_array)
    for p in np.unique(power_beam2.polarization_array):
        nt.assert_true(p in pols_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific polarizations using pyuvdata.',
                                           power_beam2.history))

    # check for errors associated with polarizations not included in data
    nt.assert_raises(ValueError, power_beam.select, polarizations=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    uvtest.checkWarnings(power_beam.select, [], {'polarizations': power_beam.polarization_array[[0, 1, 3]]},
                         message='Selected polarizations are not evenly spaced')
    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')
    nt.assert_raises(ValueError, power_beam.write_beamfits, write_file_beamfits)

    # check for error with selecting on feeds on power beams
    nt.assert_raises(ValueError, power_beam.select, feeds=['x'])


def test_select():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(power_beam)
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_input_impedance = 340.
    power_beam.reference_output_impedance = 50.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    # now test selecting along all axes at once
    old_history = power_beam.history

    inds1_to_keep = np.arange(14, 63)
    inds2_to_keep = np.arange(5, 14)
    freqs_to_keep = [power_beam.freq_array[0, 0]]
    pols_to_keep = [-5]

    power_beam2 = power_beam.select(axis1_inds=inds1_to_keep,
                                    axis2_inds=inds2_to_keep,
                                    frequencies=freqs_to_keep,
                                    polarizations=pols_to_keep,
                                    inplace=False)

    nt.assert_equal(len(inds1_to_keep), power_beam2.Naxes1)
    for i in inds1_to_keep:
        nt.assert_true(power_beam.axis1_array[i] in power_beam2.axis1_array)
    for i in np.unique(power_beam2.axis1_array):
        nt.assert_true(i in power_beam.axis1_array)

    nt.assert_equal(len(inds2_to_keep), power_beam2.Naxes2)
    for i in inds2_to_keep:
        nt.assert_true(power_beam.axis2_array[i] in power_beam2.axis2_array)
    for i in np.unique(power_beam2.axis2_array):
        nt.assert_true(i in power_beam.axis2_array)

    nt.assert_equal(len(freqs_to_keep), power_beam2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in power_beam2.freq_array)
    for f in np.unique(power_beam2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    nt.assert_equal(len(pols_to_keep), power_beam2.Npols)
    for p in pols_to_keep:
        nt.assert_true(p in power_beam2.polarization_array)
    for p in np.unique(power_beam2.polarization_array):
        nt.assert_true(p in pols_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific parts of first image axis, '
                                           'parts of second image axis, '
                                           'frequencies, polarizations using pyuvdata.',
                                           power_beam2.history))

    # repeat for efield beam
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(efield_beam)
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {'KEY1': 'test_keyword'}
    efield_beam.reference_input_impedance = 340.
    efield_beam.reference_output_impedance = 50.
    efield_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.loss_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs))

    feeds_to_keep = ['x']

    efield_beam2 = efield_beam.select(axis1_inds=inds1_to_keep,
                                      axis2_inds=inds2_to_keep,
                                      frequencies=freqs_to_keep,
                                      feeds=feeds_to_keep,
                                      inplace=False)

    nt.assert_equal(len(inds1_to_keep), efield_beam2.Naxes1)
    for i in inds1_to_keep:
        nt.assert_true(efield_beam.axis1_array[i] in efield_beam2.axis1_array)
    for i in np.unique(efield_beam2.axis1_array):
        nt.assert_true(i in efield_beam.axis1_array)

    nt.assert_equal(len(inds2_to_keep), efield_beam2.Naxes2)
    for i in inds2_to_keep:
        nt.assert_true(efield_beam.axis2_array[i] in efield_beam2.axis2_array)
    for i in np.unique(efield_beam2.axis2_array):
        nt.assert_true(i in efield_beam.axis2_array)

    nt.assert_equal(len(freqs_to_keep), efield_beam2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in efield_beam2.freq_array)
    for f in np.unique(efield_beam2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    nt.assert_equal(len(feeds_to_keep), efield_beam2.Nfeeds)
    for f in feeds_to_keep:
        nt.assert_true(f in efield_beam2.feed_array)
    for f in np.unique(efield_beam2.feed_array):
        nt.assert_true(f in feeds_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific parts of first image axis, '
                                           'parts of second image axis, '
                                           'frequencies, feeds using pyuvdata.',
                                           efield_beam2.history))


def test_add():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(power_beam)
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_input_impedance = 340.
    power_beam.reference_output_impedance = 50.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    # Add along first image axis
    beam1 = power_beam.select(axis1_inds=np.arange(0, 180), inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(180, 360), inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(power_beam.history
                                           + '  Downselected to specific parts of '
                                           'first image axis using pyuvdata. '
                                           'Combined data along first image axis '
                                           'using pyuvdata.', beam1.history))
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Out of order - axis1
    beam1 = power_beam.select(axis1_inds=np.arange(180, 360), inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(0, 180), inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Add along second image axis
    beam1 = power_beam.select(axis2_inds=np.arange(0, 90), inplace=False)
    beam2 = power_beam.select(axis2_inds=np.arange(90, 181), inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(power_beam.history
                                           + '  Downselected to specific parts of '
                                           'second image axis using pyuvdata. '
                                           'Combined data along second image axis '
                                           'using pyuvdata.', beam1.history))
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Out of order - axis2
    beam1 = power_beam.select(axis2_inds=np.arange(90, 181), inplace=False)
    beam2 = power_beam.select(axis2_inds=np.arange(0, 90), inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Add frequencies
    beam1 = power_beam.select(freq_chans=0, inplace=False)
    beam2 = power_beam.select(freq_chans=1, inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(power_beam.history
                                           + '  Downselected to specific frequencies '
                                           'using pyuvdata. Combined data along '
                                           'frequency axis using pyuvdata.',
                                           beam1.history))
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Out of order - freqs
    beam1 = power_beam.select(freq_chans=1, inplace=False)
    beam2 = power_beam.select(freq_chans=0, inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Add polarizations
    beam1 = power_beam.select(polarizations=-5, inplace=False)
    beam2 = power_beam.select(polarizations=-6, inplace=False)
    beam1 += beam2
    nt.assert_true(uvutils.check_histories(power_beam.history
                                           + '  Downselected to specific polarizations '
                                           'using pyuvdata. Combined data along '
                                           'polarization axis using pyuvdata.',
                                           beam1.history))
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Out of order - pols
    beam1 = power_beam.select(polarizations=-6, inplace=False)
    beam2 = power_beam.select(polarizations=-5, inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # Add feeds
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(efield_beam)
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {'KEY1': 'test_keyword'}
    efield_beam.reference_input_impedance = 340.
    efield_beam.reference_output_impedance = 50.
    efield_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.loss_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs))

    beam1 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    beam2 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    beam1 += beam2
    nt.assert_true(uvutils.check_histories(efield_beam.history
                                           + '  Downselected to specific feeds '
                                           'using pyuvdata. Combined data along '
                                           'feed axis using pyuvdata.',
                                           beam1.history))
    beam1.history = efield_beam.history
    nt.assert_equal(beam1, efield_beam)

    # Out of order - feeds
    beam1 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    beam2 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    beam1 += beam2
    beam1.history = efield_beam.history
    nt.assert_equal(beam1, efield_beam)

    # Add multiple axes
    beam_ref = copy.deepcopy(power_beam)
    beam1 = power_beam.select(axis1_inds=np.arange(0, power_beam.Naxes1 / 2),
                              polarizations=power_beam.polarization_array[0],
                              inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(power_beam.Naxes1 / 2,
                                                   power_beam.Naxes1),
                              polarizations=power_beam.polarization_array[1],
                              inplace=False)
    beam1 += beam2
    nt.assert_true(uvutils.check_histories(power_beam.history
                                           + '  Downselected to specific parts of '
                                           'first image axis, polarizations using '
                                           'pyuvdata. Combined data along first '
                                           'image, polarization axis using pyuvdata.',
                                           beam1.history))
    # Zero out missing data in reference object
    beam_ref.data_array[:, :, 0, :, :, power_beam.Naxes1 / 2:] = 0.0
    beam_ref.data_array[:, :, 1, :, :, :power_beam.Naxes1 / 2] = 0.0
    beam1.history = power_beam.history
    nt.assert_equal(beam1, beam_ref)

    # Another combo with efield
    beam_ref = copy.deepcopy(efield_beam)
    beam1 = efield_beam.select(axis1_inds=np.arange(0, efield_beam.Naxes1 / 2),
                               axis2_inds=np.arange(0, efield_beam.Naxes2 / 2),
                               inplace=False)
    beam2 = efield_beam.select(axis1_inds=np.arange(efield_beam.Naxes1 / 2,
                                                    efield_beam.Naxes1),
                               axis2_inds=np.arange(efield_beam.Naxes2 / 2,
                                                    efield_beam.Naxes2),
                               inplace=False)
    beam1 += beam2
    nt.assert_true(uvutils.check_histories(efield_beam.history
                                           + '  Downselected to specific parts of '
                                           'first image axis, parts of second '
                                           'image axis using pyuvdata. Combined '
                                           'data along first image, second image '
                                           'axis using pyuvdata.',
                                           beam1.history))

    # Zero out missing data in reference object
    beam_ref.data_array[:, :, :, :, :efield_beam.Naxes2 / 2,
                        efield_beam.Naxes1 / 2:] = 0.0
    beam_ref.data_array[:, :, :, :, efield_beam.Naxes2 / 2:,
                        :efield_beam.Naxes1 / 2] = 0.0

    beam_ref.basis_vector_array[:, :, :efield_beam.Naxes2 / 2,
                                efield_beam.Naxes1 / 2:] = 0.0
    beam_ref.basis_vector_array[:, :, efield_beam.Naxes2 / 2:,
                                :efield_beam.Naxes1 / 2] = 0.0
    beam1.history = efield_beam.history
    nt.assert_equal(beam1, beam_ref)

    # Test error on converting to healpix
    nt.assert_raises(ValueError, efield_beam.az_za_to_healpix)

    # Another combo with healpix efield
    efield_beam = fill_dummy_beam(efield_beam, 'efield', 'healpix')

    beam_ref = copy.deepcopy(efield_beam)
    beam1 = efield_beam.select(freq_chans=np.arange(0, 50),
                               feeds=efield_beam.feed_array[0], inplace=False)
    beam2 = efield_beam.select(freq_chans=np.arange(50, 100),
                               feeds=efield_beam.feed_array[1], inplace=False)
    beam1 += beam2
    nt.assert_true(uvutils.check_histories(efield_beam.history
                                           + '  Downselected to specific frequencies, '
                                           'feeds using pyuvdata. Combined data '
                                           'along frequency, feed axis using pyuvdata.',
                                           beam1.history))
    # Zero out missing data in reference object
    beam_ref.data_array[:, :, 1, :50, :] = 0.0
    beam_ref.data_array[:, :, 0, 50:, :] = 0.0
    beam1.history = efield_beam.history
    nt.assert_equal(beam1, beam_ref)

    # Add without inplace
    efield_beam = fill_dummy_beam(efield_beam, 'efield', 'healpix')
    beam1 = efield_beam.select(pixels=efield_beam.pixel_array[0:efield_beam.Npixels / 2],
                               inplace=False)
    beam2 = efield_beam.select(pixels=efield_beam.pixel_array[efield_beam.Npixels / 2:],
                               inplace=False)
    beam1 = beam1 + beam2
    nt.assert_true(uvutils.check_histories(efield_beam.history
                                           + '  Downselected to specific healpix pixels '
                                           'using pyuvdata. Combined data '
                                           'along healpix pixel axis using pyuvdata.',
                                           beam1.history))
    beam1.history = efield_beam.history
    nt.assert_equal(beam1, efield_beam)

    # Check warnings
    # generate more frequencies for testing by copying and adding several times
    while power_beam.Nfreqs < 8:
        new_beam = copy.deepcopy(power_beam)
        new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
        power_beam += new_beam

    beam1 = power_beam.select(freq_chans=np.arange(0, 4), inplace=False)
    beam2 = power_beam.select(freq_chans=np.arange(5, 8), inplace=False)
    uvtest.checkWarnings(beam1.__add__, [beam2],
                         message='Combined frequencies are not evenly spaced')

    # generate more polarizations for testing by copying and adding several times
    while power_beam.Npols < 4:
        new_beam = copy.deepcopy(power_beam)
        new_beam.polarization_array = power_beam.polarization_array - power_beam.Npols
        power_beam += new_beam

    power_beam.receiver_temperature_array = np.ones((1, 8))
    beam1 = power_beam.select(polarizations=power_beam.polarization_array[0:2],
                              inplace=False)
    beam2 = power_beam.select(polarizations=power_beam.polarization_array[3],
                              inplace=False)
    uvtest.checkWarnings(beam1.__iadd__, [beam2],
                         message='Combined polarizations are not evenly spaced')

    beam1 = power_beam.select(polarizations=power_beam.polarization_array[0:2],
                              inplace=False)
    beam2 = power_beam.select(polarizations=power_beam.polarization_array[2:3],
                              inplace=False)
    beam2.receiver_temperature_array = None
    nt.assert_false(beam1.receiver_temperature_array is None)
    uvtest.checkWarnings(beam1.__iadd__, [beam2],
                         message=['Only one of the UVBeam objects being combined '
                                  'has optional parameter'])
    nt.assert_true(beam1.receiver_temperature_array is None)

    # Combining histories
    beam1 = power_beam.select(polarizations=power_beam.polarization_array[0:2], inplace=False)
    beam2 = power_beam.select(polarizations=power_beam.polarization_array[2:4], inplace=False)
    beam2.history += ' testing the history. Read/written with pyuvdata'
    beam1 += beam2
    nt.assert_true(uvutils.check_histories(power_beam.history
                                           + '  Downselected to specific polarizations '
                                           'using pyuvdata. Combined data along '
                                           'polarization axis using pyuvdata. '
                                           'testing the history.',
                                           beam1.history))
    beam1.history = power_beam.history
    nt.assert_equal(beam1, power_beam)

    # ------------------------
    # Test failure modes of add function

    # Wrong class
    beam1 = copy.deepcopy(power_beam)
    nt.assert_raises(ValueError, beam1.__iadd__, np.zeros(5))

    params_to_change = {'beam_type': 'efield', 'data_normalization': 'solid_angle',
                        'telescope_name': 'foo', 'feed_name': 'foo',
                        'feed_version': 'v12', 'model_name': 'foo',
                        'model_version': 'v12', 'pixel_coordinate_system': 'sin_zenith',
                        'Naxes_vec': 3, 'nside': 16, 'ordering': 'nested'}

    beam1 = power_beam.select(freq_chans=0, inplace=False)
    for param, value in params_to_change.iteritems():
        beam2 = power_beam.select(freq_chans=1, inplace=False)
        setattr(beam2, param, value)
        nt.assert_raises(ValueError, beam1.__iadd__, beam2)

    # Overlapping data
    beam2 = copy.deepcopy(power_beam)
    nt.assert_raises(ValueError, beam1.__iadd__, beam2)


def test_healpix():
    # put all the testing on healpix in this one function to minimize slow calls
    # to uvbeam.az_za_to_healpix()
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(power_beam)
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_input_impedance = 340.
    power_beam.reference_output_impedance = 50.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    power_beam_healpix = copy.deepcopy(power_beam)
    power_beam_healpix.az_za_to_healpix()

    # test that Npixels make sense
    n_max_pix = power_beam.Naxes1 * power_beam.Naxes2
    nt.assert_true(power_beam_healpix.Npixels <= n_max_pix)

    # -----------------------
    # test selecting on pixels
    old_history = power_beam_healpix.history
    pixels_to_keep = np.arange(31, 184)

    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep, inplace=False)

    nt.assert_equal(len(pixels_to_keep), power_beam_healpix2.Npixels)
    for pi in pixels_to_keep:
        nt.assert_true(pi in power_beam_healpix2.pixel_array)
    for pi in np.unique(power_beam_healpix2.pixel_array):
        nt.assert_true(pi in pixels_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific healpix pixels using pyuvdata.',
                                           power_beam_healpix2.history))

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

    # test writing beamfits with only one pixel
    pixels_to_keep = [43]
    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep, inplace=False)
    power_beam_healpix2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with pixels not included in data
    nt.assert_raises(ValueError, power_beam_healpix.select,
                     pixels=[12 * power_beam_healpix.nside**2 + 10])

    # test writing beamfits with non-contiguous pixels
    pixels_to_keep = np.arange(2, 150, 4)

    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep, inplace=False)
    power_beam_healpix2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors selecting pixels on non-healpix beams
    nt.assert_raises(ValueError, power_beam.select, pixels=pixels_to_keep)

    # -----------------
    # check for errors selecting axis1_inds on healpix beams
    inds1_to_keep = np.arange(14, 63)
    nt.assert_raises(ValueError, power_beam_healpix.select, axis1_inds=inds1_to_keep)

    # check for errors selecting axis2_inds on healpix beams
    inds2_to_keep = np.arange(5, 14)
    nt.assert_raises(ValueError, power_beam_healpix.select, axis2_inds=inds2_to_keep)

    # ------------------------
    # test selecting along all axes at once for healpix beams
    freqs_to_keep = [power_beam_healpix.freq_array[0, 0]]
    pols_to_keep = [-5]

    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep,
                                                    frequencies=freqs_to_keep,
                                                    polarizations=pols_to_keep,
                                                    inplace=False)

    nt.assert_equal(len(pixels_to_keep), power_beam_healpix2.Npixels)
    for pi in pixels_to_keep:
        nt.assert_true(pi in power_beam_healpix2.pixel_array)
    for pi in np.unique(power_beam_healpix2.pixel_array):
        nt.assert_true(pi in pixels_to_keep)

    nt.assert_equal(len(freqs_to_keep), power_beam_healpix2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in power_beam_healpix2.freq_array)
    for f in np.unique(power_beam_healpix2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    nt.assert_equal(len(pols_to_keep), power_beam_healpix2.Npols)
    for p in pols_to_keep:
        nt.assert_true(p in power_beam_healpix2.polarization_array)
    for p in np.unique(power_beam_healpix2.polarization_array):
        nt.assert_true(p in pols_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific healpix pixels, frequencies, '
                                           'polarizations using pyuvdata.',
                                           power_beam_healpix2.history))

    # repeat for efield beam
    efield_beam = UVBeam()
    efield_beam = fill_dummy_beam(efield_beam, 'efield', 'healpix')
    old_history = efield_beam.history

    freqs_to_keep = efield_beam.freq_array[0, np.arange(31, 56)]
    feeds_to_keep = ['x']

    efield_beam2 = efield_beam.select(pixels=pixels_to_keep,
                                      frequencies=freqs_to_keep,
                                      feeds=feeds_to_keep,
                                      inplace=False)

    nt.assert_equal(len(pixels_to_keep), efield_beam2.Npixels)
    for pi in pixels_to_keep:
        nt.assert_true(pi in efield_beam2.pixel_array)
    for pi in np.unique(efield_beam2.pixel_array):
        nt.assert_true(pi in pixels_to_keep)

    nt.assert_equal(len(freqs_to_keep), efield_beam2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in efield_beam2.freq_array)
    for f in np.unique(efield_beam2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    nt.assert_equal(len(feeds_to_keep), efield_beam2.Nfeeds)
    for f in feeds_to_keep:
        nt.assert_true(f in efield_beam2.feed_array)
    for f in np.unique(efield_beam2.feed_array):
        nt.assert_true(f in feeds_to_keep)

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific healpix pixels, frequencies, '
                                           'feeds using pyuvdata.',
                                           efield_beam2.history))

    # -------------------
    # Test adding a different combo with healpix
    beam_ref = copy.deepcopy(power_beam_healpix)
    beam1 = power_beam_healpix.select(
        pixels=power_beam_healpix.pixel_array[0:power_beam_healpix.Npixels / 2],
        freq_chans=0, inplace=False)
    beam2 = power_beam_healpix.select(
        pixels=power_beam_healpix.pixel_array[power_beam_healpix.Npixels / 2:],
        freq_chans=1, inplace=False)
    beam1 += beam2
    nt.assert_true(uvutils.check_histories(power_beam_healpix.history
                                           + '  Downselected to specific healpix '
                                           'pixels, frequencies using pyuvdata. '
                                           'Combined data along healpix pixel, '
                                           'frequency axis using pyuvdata.',
                                           beam1.history))
    # Zero out missing data in reference object
    beam_ref.data_array[:, :, :, 0, power_beam_healpix.Npixels / 2:] = 0.0
    beam_ref.data_array[:, :, :, 1, :power_beam_healpix.Npixels / 2] = 0.0
    beam1.history = power_beam_healpix.history
    nt.assert_equal(beam1, beam_ref)

    # ---------------
    # Test error: adding overlapping data with healpix
    beam1 = copy.deepcopy(power_beam_healpix)
    beam2 = copy.deepcopy(power_beam_healpix)
    nt.assert_raises(ValueError, beam1.__iadd__, beam2)


def test_get_beam_functions():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1',
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # assert get_beam fails
    nt.assert_raises(AssertionError, power_beam._get_beam, 'I')

    # Convert to healpix
    power_beam.az_za_to_healpix()

    # Check that non-peak normalizations error
    nt.assert_raises(ValueError, power_beam.get_beam_area)
    nt.assert_raises(ValueError, power_beam.get_beam_sq_area)
    power_beam.data_normalization = 'solid_angle'
    nt.assert_raises(ValueError, power_beam.get_beam_area)
    nt.assert_raises(ValueError, power_beam.get_beam_sq_area)
    # change it to peak for rest of checks
    power_beam.data_normalization = 'peak'

    # Check sizes of output
    numfreqs = power_beam.freq_array.shape[-1]
    beam_int = power_beam.get_beam_area()
    beam_sq_int = power_beam.get_beam_sq_area()
    nt.assert_equal(beam_int.shape[0], numfreqs)
    nt.assert_equal(beam_sq_int.shape[0], numfreqs)

    # Check for the case of a uniform beam over the whole sky
    dOmega = hp.nside2pixarea(power_beam.nside)
    npix = power_beam.Npixels
    power_beam.data_array = np.ones_like(power_beam.data_array)
    nt.assert_almost_equal(np.sum(power_beam.get_beam_area()), numfreqs * npix * dOmega)
    power_beam.data_array = 2. * np.ones_like(power_beam.data_array)
    nt.assert_almost_equal(np.sum(power_beam.get_beam_sq_area()), numfreqs * 4. * npix * dOmega)

    # check XX and YY beam areas work and match to within 5 sigfigs
    XX_area = power_beam.get_beam_area('XX')
    xx_area = power_beam.get_beam_area('xx')
    nt.assert_almost_equal(xx_area, XX_area)
    YY_area = power_beam.get_beam_area('YY')
    nt.assert_almost_equal(YY_area / XX_area, 1.0, places=5)
    XX_area = power_beam.get_beam_sq_area("XX")
    YY_area = power_beam.get_beam_sq_area("YY")
    nt.assert_almost_equal(YY_area / XX_area, 1.0, places=5)

    # Check that if stokes I is in the beam polarization_array, it just uses it
    power_beam.polarization_array = [1, 2]
    nt.assert_almost_equal(np.sum(power_beam.get_beam_area()), 2. * numfreqs * npix * dOmega)
    nt.assert_almost_equal(np.sum(power_beam.get_beam_sq_area()), 4. * numfreqs * npix * dOmega)

    # Check to make sure only pseudo Stokes I is accepted
    nt.assert_raises(NotImplementedError, power_beam.get_beam_area, pol='Q')
    nt.assert_raises(NotImplementedError, power_beam.get_beam_sq_area, pol='Q')

    # Check polarization error
    power_beam.polarization_array = [9, 18, 27, -5]
    nt.assert_raises(ValueError, power_beam.get_beam_area)
    nt.assert_raises(ValueError, power_beam.get_beam_sq_area)

    # Check if power is scalar
    power_beam.data_array = np.vstack([power_beam.data_array, power_beam.data_array])
    nt.assert_raises(ValueError, power_beam.get_beam_area)
    nt.assert_raises(ValueError, power_beam.get_beam_sq_area)

    # Check only power beams accepted
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1',
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    nt.assert_raises(ValueError, efield_beam.get_beam_area)
    nt.assert_raises(ValueError, efield_beam.get_beam_sq_area)

    # Check only healpix accepted
    az_za_beam = UVBeam()
    az_za_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1',
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # change data_normalization to peak for rest of checks
    az_za_beam.data_normalization = 'peak'
    nt.assert_raises(ValueError, az_za_beam.get_beam_area)
    nt.assert_raises(ValueError, az_za_beam.get_beam_sq_area)
