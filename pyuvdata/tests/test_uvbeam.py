"""Tests for uvbeam object."""
import nose.tools as nt
import os
import numpy as np
import copy
from pyuvdata import UVBeam
import pyuvdata.tests as uvtest
import pyuvdata.version as uvversion
from pyuvdata.data import DATA_PATH


def fill_dummy_beam(beam_obj, beam_type, pixel_coordinate_system):
    beam_obj.set_simple()
    beam_obj.telescope_name = 'testscope'
    beam_obj.feed_name = 'testfeed'
    beam_obj.feed_version = '0.1'
    beam_obj.model_name = 'testmodel'
    beam_obj.model_version = '1.0'
    beam_obj.history = 'random data for test'

    pyuvdata_version_str = ('  Read/written with pyuvdata version: ' +
                            uvversion.version + '.')
    if uvversion.git_hash is not '':
        pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin +
                                 '.  Git hash: ' + uvversion.git_hash +
                                 '.  Git branch: ' + uvversion.git_branch +
                                 '.  Git description: ' + uvversion.git_description + '.')
    beam_obj.history += pyuvdata_version_str

    beam_obj.pixel_coordinate_system = pixel_coordinate_system
    if pixel_coordinate_system == 'healpix':
        beam_obj.nside = 512
        beam_obj.ordering = 'ring'
        beam_obj.pixel_array = np.arange(0, 256)
        beam_obj.Npixels = len(beam_obj.pixel_array)
    elif pixel_coordinate_system == 'az_za':
        beam_obj.axis1_array = np.arange(-180.0, 180.0, 5.0)
        beam_obj.Naxes1 = len(beam_obj.axis1_array)
        beam_obj.axis2_array = np.arange(-90.0, 90.0, 5.0)
        beam_obj.Naxes2 = len(beam_obj.axis2_array)

    beam_obj.freq_array = np.arange(150e6, 160e6, 1e5)
    beam_obj.freq_array = beam_obj.freq_array[np.newaxis, :]
    beam_obj.Nfreqs = beam_obj.freq_array.shape[1]
    beam_obj.spw_array = np.array([0])
    beam_obj.Nspws = len(beam_obj.spw_array)
    beam_obj.data_normalization = 'peak'

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

        beam_obj.data_array = (np.random.normal(0.0, 0.2, size=data_size_tuple) +
                               1j * np.random.normal(0.0, 0.2, size=data_size_tuple))

    # add optional parameters for testing purposes
    beam_obj.extra_keywords = {'KEY1': 'test_keyword'}
    beam_obj.system_temperature_array = np.random.normal(50.0, 5, size=(beam_obj.Nspws, beam_obj.Nfreqs))
    beam_obj.loss_array = np.random.normal(50.0, 5, size=(beam_obj.Nspws, beam_obj.Nfreqs))
    beam_obj.mismatch_array = np.random.normal(0.0, 1.0, size=(beam_obj.Nspws, beam_obj.Nfreqs))
    beam_obj.s_parameters = np.random.normal(0.0, 0.3, size=(beam_obj.Nspws, beam_obj.Nfreqs))

    return beam_obj


class TestUVBeamInit(object):
    def setUp(self):
        """Setup for basic parameter, property and iterator tests."""
        self.required_parameters = ['_beam_type', '_Nfreqs', '_Naxes_vec', '_Nspws',
                                    '_pixel_coordinate_system',
                                    '_freq_array', '_spw_array',
                                    '_data_normalization',
                                    '_data_array',
                                    '_telescope_name', '_feed_name',
                                    '_feed_version', '_model_name',
                                    '_model_version', '_history',
                                    '_antenna_type']

        self.required_properties = ['beam_type', 'Nfreqs', 'Naxes_vec', 'Nspws',
                                    'pixel_coordinate_system',
                                    'freq_array', 'spw_array',
                                    'data_normalization',
                                    'data_array',
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
                                 '_system_temperature_array',
                                 '_loss_array', '_mismatch_array',
                                 '_s_parameters']

        self.extra_properties = ['Naxes1', 'Naxes2', 'Npixels', 'Nfeeds', 'Npols',
                                 'axis1_array', 'axis2_array', 'nside', 'ordering',
                                 'pixel_array', 'feed_array', 'polarization_array',
                                 'basis_vector_array', 'extra_keywords', 'Nelements',
                                 'element_coordinate_system',
                                 'element_location_array', 'delay_array',
                                 'gain_array', 'coupling_matrix',
                                 'system_temperature_array',
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
            nt.assert_true(a in all, msg='expected attribute ' + a +
                           ' not returned in object iterator')

    def test_required_parameter_iter(self):
        "Test expected required parameters."
        required = []
        for prop in self.beam_obj.required():
            required.append(prop)
        for a in self.required_parameters:
            nt.assert_true(a in required, msg='expected attribute ' + a +
                           ' not returned in required iterator')

    def test_extra_parameter_iter(self):
        "Test expected optional parameters."
        extra = []
        for prop in self.beam_obj.extra():
            extra.append(prop)
        for a in self.extra_parameters:
            nt.assert_true(a in extra, msg='expected attribute ' + a +
                           ' not returned in extra iterator')

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


class TestUVBeamSelect(object):
    def setUp(self):
        """Set up test"""
        self.power_beam = UVBeam()
        self.power_beam = fill_dummy_beam(self.power_beam, 'power', 'az_za')

        self.efield_beam = UVBeam()
        self.efield_beam = fill_dummy_beam(self.efield_beam, 'efield', 'az_za')

    def teardown(self):
        """Tear down test"""
        del(self.power_beam)
        del(self.efield_beam)

    def test_select_axis1(self):
        old_history = self.power_beam.history
        inds1_to_keep = np.arange(14, 63)

        self.power_beam2 = self.power_beam.select(axis1_inds=inds1_to_keep, inplace=False)

        nt.assert_equal(len(inds1_to_keep), self.power_beam2.Naxes1)
        for i in inds1_to_keep:
            nt.assert_true(self.power_beam.axis1_array[i] in self.power_beam2.axis1_array)
        for i in np.unique(self.power_beam2.axis1_array):
            nt.assert_true(i in self.power_beam.axis1_array)

        nt.assert_equal(old_history + '  Downselected to specific parts of '
                        'first image axis using pyuvdata.', self.power_beam2.history)

        write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

        # test writing beamfits with only one element in axis1
        inds_to_keep = [len(inds1_to_keep) + 1]
        self.power_beam2 = self.power_beam.select(axis1_inds=inds_to_keep, inplace=False)
        self.power_beam2.write_beamfits(write_file_beamfits, clobber=True)

        # check for errors associated with indices not included in data
        nt.assert_raises(ValueError, self.power_beam2.select, axis1_inds=[self.power_beam.Naxes1 - 1])

        # check for warnings and errors associated with unevenly spaced image pixels
        self.power_beam2 = copy.deepcopy(self.power_beam)
        uvtest.checkWarnings(self.power_beam2.select, [], {'axis1_inds': [0, 5, 6]},
                             message='Selected values along first image axis are not evenly spaced')
        nt.assert_raises(ValueError, self.power_beam2.write_beamfits, write_file_beamfits)

    def test_select_axis2(self):
        old_history = self.power_beam.history
        inds2_to_keep = np.arange(5, 22)

        self.power_beam2 = self.power_beam.select(axis2_inds=inds2_to_keep, inplace=False)

        nt.assert_equal(len(inds2_to_keep), self.power_beam2.Naxes2)
        for i in inds2_to_keep:
            nt.assert_true(self.power_beam.axis2_array[i] in self.power_beam2.axis2_array)
        for i in np.unique(self.power_beam2.axis2_array):
            nt.assert_true(i in self.power_beam.axis2_array)

        nt.assert_equal(old_history + '  Downselected to specific parts of '
                        'second image axis using pyuvdata.', self.power_beam2.history)

        write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

        # test writing beamfits with only one element in axis2
        inds_to_keep = [len(inds2_to_keep) + 1]
        self.power_beam2 = self.power_beam.select(axis2_inds=inds_to_keep, inplace=False)
        self.power_beam2.write_beamfits(write_file_beamfits, clobber=True)

        # check for errors associated with indices not included in data
        nt.assert_raises(ValueError, self.power_beam2.select, axis2_inds=[self.power_beam.Naxes2 - 1])

        # check for warnings and errors associated with unevenly spaced image pixels
        self.power_beam2 = copy.deepcopy(self.power_beam)
        uvtest.checkWarnings(self.power_beam2.select, [], {'axis2_inds': [0, 5, 6]},
                             message='Selected values along second image axis are not evenly spaced')
        nt.assert_raises(ValueError, self.power_beam2.write_beamfits, write_file_beamfits)

    def test_select_frequencies(self):
        old_history = self.power_beam.history
        freqs_to_keep = self.power_beam.freq_array[0, np.arange(7, 94)]

        self.power_beam2 = self.power_beam.select(frequencies=freqs_to_keep, inplace=False)

        nt.assert_equal(len(freqs_to_keep), self.power_beam2.Nfreqs)
        for f in freqs_to_keep:
            nt.assert_true(f in self.power_beam2.freq_array)
        for f in np.unique(self.power_beam2.freq_array):
            nt.assert_true(f in freqs_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific frequencies '
                        'using pyuvdata.', self.power_beam2.history)

        write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')
        # test writing beamfits with only one frequency

        freqs_to_keep = self.power_beam.freq_array[0, 51]
        self.power_beam2 = self.power_beam.select(frequencies=freqs_to_keep, inplace=False)
        self.power_beam2.write_beamfits(write_file_beamfits, clobber=True)

        # check for errors associated with frequencies not included in data
        nt.assert_raises(ValueError, self.power_beam.select, frequencies=[np.max(self.power_beam.freq_array) + 10])

        # check for warnings and errors associated with unevenly spaced frequencies
        self.power_beam2 = copy.deepcopy(self.power_beam)
        uvtest.checkWarnings(self.power_beam2.select, [], {'frequencies': self.power_beam2.freq_array[0, [0, 5, 6]]},
                             message='Selected frequencies are not evenly spaced')
        nt.assert_raises(ValueError, self.power_beam2.write_beamfits, write_file_beamfits)

    def test_select_freq_chans(self):
        old_history = self.power_beam.history
        chans_to_keep = np.arange(7, 94)

        self.power_beam2 = self.power_beam.select(freq_chans=chans_to_keep, inplace=False)

        nt.assert_equal(len(chans_to_keep), self.power_beam2.Nfreqs)
        for chan in chans_to_keep:
            nt.assert_true(self.power_beam.freq_array[0, chan] in self.power_beam2.freq_array)
        for f in np.unique(self.power_beam2.freq_array):
            nt.assert_true(f in self.power_beam.freq_array[0, chans_to_keep])

        nt.assert_equal(old_history + '  Downselected to specific frequencies '
                        'using pyuvdata.', self.power_beam2.history)

        # Test selecting both channels and frequencies
        freqs_to_keep = self.power_beam.freq_array[0, np.arange(93, 100)]  # Overlaps with chans
        all_chans_to_keep = np.arange(7, 100)

        self.power_beam2 = self.power_beam.select(frequencies=freqs_to_keep,
                                                  freq_chans=chans_to_keep,
                                                  inplace=False)

        nt.assert_equal(len(all_chans_to_keep), self.power_beam2.Nfreqs)
        for chan in all_chans_to_keep:
            nt.assert_true(self.power_beam.freq_array[0, chan] in self.power_beam2.freq_array)
        for f in np.unique(self.power_beam2.freq_array):
            nt.assert_true(f in self.power_beam.freq_array[0, all_chans_to_keep])

    def test_select_feeds(self):

        old_history = self.efield_beam.history
        feeds_to_keep = ['x']

        self.efield_beam2 = self.efield_beam.select(feeds=feeds_to_keep,
                                                    inplace=False)

        nt.assert_equal(len(feeds_to_keep), self.efield_beam2.Nfeeds)
        for f in feeds_to_keep:
            nt.assert_true(f in self.efield_beam2.feed_array)
        for f in np.unique(self.efield_beam2.feed_array):
            nt.assert_true(f in feeds_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific feeds '
                        'using pyuvdata.', self.efield_beam2.history)

        # check for errors associated with feeds not included in data
        nt.assert_raises(ValueError, self.efield_beam2.select, feeds=['N'])

        # check for error with feeds on power beams
        nt.assert_raises(ValueError, self.power_beam.select, feeds=['x'])

    def test_select_polarizations(self):

        old_history = self.power_beam.history
        pols_to_keep = [-5, -6]

        self.power_beam2 = self.power_beam.select(polarizations=pols_to_keep,
                                                  inplace=False)

        nt.assert_equal(len(pols_to_keep), self.power_beam2.Npols)
        for p in pols_to_keep:
            nt.assert_true(p in self.power_beam2.polarization_array)
        for p in np.unique(self.power_beam2.polarization_array):
            nt.assert_true(p in pols_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific polarizations '
                        'using pyuvdata.', self.power_beam2.history)

        # check for errors associated with polarizations not included in data
        nt.assert_raises(ValueError, self.power_beam2.select, polarizations=[-3, -4])

        # check for warnings and errors associated with unevenly spaced polarizations
        uvtest.checkWarnings(self.power_beam.select, [], {'polarizations': self.power_beam.polarization_array[[0, 1, 3]]},
                             message='Selected polarizations are not evenly spaced')
        write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')
        nt.assert_raises(ValueError, self.power_beam.write_beamfits, write_file_beamfits)

        # check for error with polarizations on efield beams
        nt.assert_raises(ValueError, self.efield_beam.select, polarizations=[-5, -6])

    def test_select(self):
        # now test selecting along all axes at once
        old_history = self.power_beam.history

        inds1_to_keep = np.arange(14, 63)
        inds2_to_keep = np.arange(5, 22)
        freqs_to_keep = self.power_beam.freq_array[0, np.arange(31, 56)]
        pols_to_keep = [-5]

        self.power_beam2 = self.power_beam.select(axis1_inds=inds1_to_keep,
                                                  axis2_inds=inds2_to_keep,
                                                  frequencies=freqs_to_keep,
                                                  polarizations=pols_to_keep,
                                                  inplace=False)

        nt.assert_equal(len(inds1_to_keep), self.power_beam2.Naxes1)
        for i in inds1_to_keep:
            nt.assert_true(self.power_beam.axis1_array[i] in self.power_beam2.axis1_array)
        for i in np.unique(self.power_beam2.axis1_array):
            nt.assert_true(i in self.power_beam.axis1_array)

        nt.assert_equal(len(inds2_to_keep), self.power_beam2.Naxes2)
        for i in inds2_to_keep:
            nt.assert_true(self.power_beam.axis2_array[i] in self.power_beam2.axis2_array)
        for i in np.unique(self.power_beam2.axis2_array):
            nt.assert_true(i in self.power_beam.axis2_array)

        nt.assert_equal(len(freqs_to_keep), self.power_beam2.Nfreqs)
        for f in freqs_to_keep:
            nt.assert_true(f in self.power_beam2.freq_array)
        for f in np.unique(self.power_beam2.freq_array):
            nt.assert_true(f in freqs_to_keep)

        nt.assert_equal(len(pols_to_keep), self.power_beam2.Npols)
        for p in pols_to_keep:
            nt.assert_true(p in self.power_beam2.polarization_array)
        for p in np.unique(self.power_beam2.polarization_array):
            nt.assert_true(p in pols_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific parts of '
                        'first image axis, parts of second image axis, '
                        'frequencies, polarizations using pyuvdata.',
                        self.power_beam2.history)

        # repeat for efield beam
        feeds_to_keep = ['x']

        self.efield_beam2 = self.efield_beam.select(axis1_inds=inds1_to_keep,
                                                    axis2_inds=inds2_to_keep,
                                                    frequencies=freqs_to_keep,
                                                    feeds=feeds_to_keep,
                                                    inplace=False)

        nt.assert_equal(len(inds1_to_keep), self.efield_beam2.Naxes1)
        for i in inds1_to_keep:
            nt.assert_true(self.efield_beam.axis1_array[i] in self.efield_beam2.axis1_array)
        for i in np.unique(self.efield_beam2.axis1_array):
            nt.assert_true(i in self.efield_beam.axis1_array)

        nt.assert_equal(len(inds2_to_keep), self.efield_beam2.Naxes2)
        for i in inds2_to_keep:
            nt.assert_true(self.efield_beam.axis2_array[i] in self.efield_beam2.axis2_array)
        for i in np.unique(self.efield_beam2.axis2_array):
            nt.assert_true(i in self.efield_beam.axis2_array)

        nt.assert_equal(len(freqs_to_keep), self.efield_beam2.Nfreqs)
        for f in freqs_to_keep:
            nt.assert_true(f in self.efield_beam2.freq_array)
        for f in np.unique(self.efield_beam2.freq_array):
            nt.assert_true(f in freqs_to_keep)

        nt.assert_equal(len(feeds_to_keep), self.efield_beam2.Nfeeds)
        for f in feeds_to_keep:
            nt.assert_true(f in self.efield_beam2.feed_array)
        for f in np.unique(self.efield_beam2.feed_array):
            nt.assert_true(f in feeds_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific parts of '
                        'first image axis, parts of second image axis, '
                        'frequencies, feeds using pyuvdata.',
                        self.efield_beam2.history)
