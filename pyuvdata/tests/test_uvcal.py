"""Tests for uvcal object."""
import nose.tools as nt
import os
import numpy as np
import copy
import ephem
from pyuvdata.uvcal import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


class TestUVCalInit(object):
    def setUp(self):
        """Setup for basic parameter, property and iterator tests."""
        self.required_parameters = ['_Nfreqs', '_Njones', '_Ntimes', '_Nspws',
                                    '_Nants_data', '_Nants_telescope',
                                    '_antenna_names', '_antenna_numbers',
                                    '_ant_array',
                                    '_telescope_name', '_freq_array',
                                    '_channel_width', '_spw_array',
                                    '_jones_array', '_time_array', '_time_range',
                                    '_integration_time',
                                    '_gain_convention', '_flag_array',
                                    '_quality_array', '_cal_type',
                                    '_x_orientation', '_history']

        self.required_properties = ['Nfreqs', 'Njones', 'Ntimes', 'Nspws',
                                    'Nants_data', 'Nants_telescope',
                                    'antenna_names', 'antenna_numbers',
                                    'ant_array',
                                    'telescope_name', 'freq_array',
                                    'channel_width', 'spw_array',
                                    'jones_array', 'time_array', 'time_range',
                                    'integration_time',
                                    'gain_convention', 'flag_array',
                                    'quality_array', 'cal_type',
                                    'x_orientation', 'history']

        self.extra_parameters = ['_gain_array', '_delay_array',
                                 '_input_flag_array', '_freq_range',
                                 '_observer', '_git_origin_cal',
                                 '_git_hash_cal', '_total_quality_array']

        self.extra_properties = ['gain_array', 'delay_array',
                                 'input_flag_array', 'freq_range',
                                 'observer', 'git_origin_cal',
                                 'git_hash_cal', 'total_quality_array']

        self.other_properties = ['pyuvdata_version_str']

        self.uv_cal_object = UVCal()

    def teardown(self):
        """Test teardown: delete object."""
        del(self.uv_cal_object)

    def test_parameter_iter(self):
        "Test expected parameters."
        all = []
        for prop in self.uv_cal_object:
            all.append(prop)
        for a in self.required_parameters + self.extra_parameters:
            nt.assert_true(a in all, msg='expected attribute ' + a +
                           ' not returned in object iterator')

    def test_required_parameter_iter(self):
        "Test expected required parameters."
        required = []
        for prop in self.uv_cal_object.required():
            required.append(prop)
        for a in self.required_parameters:
            nt.assert_true(a in required, msg='expected attribute ' + a +
                           ' not returned in required iterator')

    def test_unexpected_parameters(self):
        "Test for extra parameters."
        expected_parameters = self.required_parameters + self.extra_parameters
        attributes = [i for i in self.uv_cal_object.__dict__.keys() if i[0] == '_']
        for a in attributes:
            nt.assert_true(a in expected_parameters,
                           msg='unexpected parameter ' + a + ' found in UVCal')

    def test_unexpected_attributes(self):
        "Test for extra attributes."
        expected_attributes = self.required_properties + \
            self.extra_properties + self.other_properties
        attributes = [i for i in self.uv_cal_object.__dict__.keys() if i[0] != '_']
        for a in attributes:
            nt.assert_true(a in expected_attributes,
                           msg='unexpected attribute ' + a + ' found in UVCal')

    def test_properties(self):
        "Test that properties can be get and set properly."
        prop_dict = dict(zip(self.required_properties + self.extra_properties,
                             self.required_parameters + self.extra_parameters))
        for k, v in prop_dict.iteritems():
            rand_num = np.random.rand()
            setattr(self.uv_cal_object, k, rand_num)
            this_param = getattr(self.uv_cal_object, v)
            try:
                nt.assert_equal(rand_num, this_param.value)
            except:
                print('setting {prop_name} to a random number failed'.format(prop_name=k))
                raise(AssertionError)


class TestUVCalBasicMethods(object):
    def setUp(self):
        """Set up test"""
        self.gain_object = UVCal()
        gainfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
        self.gain_object.read_calfits(gainfile)
        self.gain_object2 = copy.deepcopy(self.gain_object)
        self.delay_object = UVCal()
        delayfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')
        message = delayfile + ' appears to be an old calfits format'
        uvtest.checkWarnings(self.delay_object.read_calfits, [delayfile], nwarnings=1,
                             message=message, category=[UserWarning])

    def teardown(self):
        """Tear down test"""
        del(self.gain_object)
        del(self.gain_object2)

    def test_equality(self):
        """Basic equality test"""
        nt.assert_equal(self.gain_object, self.gain_object)

    def test_check(self):
        """Test that parameter checks run properly"""
        nt.assert_true(self.gain_object.check())

    def test_nants_data_telescope(self):
        self.gain_object.Nants_data = self.gain_object.Nants_telescope - 1
        nt.assert_true(self.gain_object.check)
        self.gain_object.Nants_data = self.gain_object.Nants_telescope + 1
        nt.assert_raises(ValueError, self.gain_object.check)

    def test_set_gain(self):
        self.delay_object.set_gain()
        nt.assert_true(self.delay_object._gain_array.required)
        nt.assert_false(self.delay_object._delay_array.required)
        nt.assert_equal(self.delay_object._gain_array.form, self.delay_object._flag_array.form)
        nt.assert_equal(self.delay_object._gain_array.form, self.delay_object._quality_array.form)

    def test_set_delay(self):
        self.gain_object.set_delay()
        nt.assert_true(self.gain_object._delay_array.required)
        nt.assert_false(self.gain_object._gain_array.required)
        nt.assert_equal(self.gain_object._gain_array.form, self.gain_object._flag_array.form)
        nt.assert_equal(self.gain_object._delay_array.form, self.gain_object._quality_array.form)

    def test_set_unknown(self):
        self.gain_object.set_unknown_cal_type()
        nt.assert_false(self.gain_object._delay_array.required)
        nt.assert_false(self.gain_object._gain_array.required)
        nt.assert_equal(self.gain_object._gain_array.form, self.gain_object._flag_array.form)
        nt.assert_equal(self.gain_object._gain_array.form, self.gain_object._quality_array.form)

    def test_convert_filetype(self):
        # error testing
        nt.assert_raises(ValueError, self.gain_object._convert_to_filetype, 'uvfits')

    def test_convert_to_gain(self):
        conventions = ['minus', 'plus']
        for c in conventions:
            self.new_object = copy.deepcopy(self.delay_object)

            self.new_object.convert_to_gain(delay_convention=c)
            nt.assert_true(np.isclose(np.max(np.absolute(self.new_object.gain_array)), 1.,
                                      rtol=self.new_object._gain_array.tols[0],
                                      atol=self.new_object._gain_array.tols[1]))
            nt.assert_true(np.isclose(np.min(np.absolute(self.new_object.gain_array)), 1.,
                                      rtol=self.new_object._gain_array.tols[0],
                                      atol=self.new_object._gain_array.tols[1]))

            if c == 'minus':
                conv = -1
            else:
                conv = 1
            nt.assert_true(np.allclose(np.angle(self.new_object.gain_array[:, :, 10, :, :]) % (2 * np.pi),
                                       (conv * 2 * np.pi * self.delay_object.delay_array[:, :, 0, :, :] *
                                       self.delay_object.freq_array[0, 10]) % (2 * np.pi),
                                       rtol=self.new_object._gain_array.tols[0],
                                       atol=self.new_object._gain_array.tols[1]))
            nt.assert_true(np.allclose(self.delay_object.quality_array,
                                       self.new_object.quality_array[:, :, 10, :, :],
                                       rtol=self.new_object._quality_array.tols[0],
                                       atol=self.new_object._quality_array.tols[1]))

        # error testing
        nt.assert_raises(ValueError, self.delay_object.convert_to_gain, delay_convention='bogus')
        nt.assert_raises(ValueError, self.gain_object.convert_to_gain)
        self.gain_object.set_unknown_cal_type()
        nt.assert_raises(ValueError, self.gain_object.convert_to_gain)


class TestUVCalSelectGain(object):
    def setUp(self):
        """Set up test"""
        self.gain_object = UVCal()
        gainfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
        self.gain_object.read_calfits(gainfile)
        self.gain_object2 = copy.deepcopy(self.gain_object)

    def teardown(self):
        """Tear down test"""
        del(self.gain_object)
        del(self.gain_object2)

    def test_select_antennas(self):
        old_history = self.gain_object.history
        ants_to_keep = np.array([65, 96, 9, 97, 89, 22, 20, 72])
        self.gain_object2.select(antenna_nums=ants_to_keep)

        nt.assert_equal(len(ants_to_keep), self.gain_object2.Nants_data)
        for ant in ants_to_keep:
            nt.assert_true(ant in self.gain_object2.ant_array)
        for ant in self.gain_object2.ant_array:
            nt.assert_true(ant in ants_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific antennas '
                        'using pyuvdata.', self.gain_object2.history)

        # now test using antenna_names to specify antennas to keep
        ants_to_keep = np.array(sorted(list(ants_to_keep)))
        ant_names = []
        for a in ants_to_keep:
            ind = np.where(self.gain_object.antenna_numbers == a)[0][0]
            ant_names.append(self.gain_object.antenna_names[ind])

        self.gain_object3 = self.gain_object.select(antenna_names=ant_names, inplace=False)

        nt.assert_equal(self.gain_object2, self.gain_object3)

        # check for errors associated with antennas not included in data, bad names or providing numbers and names
        nt.assert_raises(ValueError, self.gain_object.select,
                         antenna_nums=np.max(self.gain_object.ant_array) + np.arange(1, 3))
        nt.assert_raises(ValueError, self.gain_object.select, antenna_names='test1')
        nt.assert_raises(ValueError, self.gain_object.select,
                         antenna_nums=ants_to_keep, antenna_names=ant_names)

        # check that write_calfits works with Nants_data < Nants_telescope
        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        status = self.gain_object2.write_calfits(write_file_calfits, clobber=True)

    def test_select_times(self):
        # add another time to allow for better testing of selections
        new_time = np.max(self.gain_object.time_array) + self.gain_object.integration_time
        self.gain_object.time_array = np.append(self.gain_object.time_array, new_time)
        self.gain_object.Ntimes += 1
        self.gain_object.flag_array = np.concatenate((self.gain_object.flag_array,
                                                      self.gain_object.flag_array[:, :, :, [-1], :]),
                                                     axis=3)
        self.gain_object.gain_array = np.concatenate((self.gain_object.gain_array,
                                                      self.gain_object.gain_array[:, :, :, [-1], :]),
                                                     axis=3)
        self.gain_object.quality_array = np.concatenate((self.gain_object.quality_array,
                                                         self.gain_object.quality_array[:, :, :, [-1], :]),
                                                        axis=3)
        nt.assert_true(self.gain_object.check())
        self.gain_object2 = copy.deepcopy(self.gain_object)

        old_history = self.gain_object.history
        times_to_keep = self.gain_object.time_array[[2, 0]]

        self.gain_object2.select(times=times_to_keep)

        nt.assert_equal(len(times_to_keep), self.gain_object2.Ntimes)
        for t in times_to_keep:
            nt.assert_true(t in self.gain_object2.time_array)
        for t in np.unique(self.gain_object2.time_array):
            nt.assert_true(t in times_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific times '
                        'using pyuvdata.', self.gain_object2.history)

        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        # test writing calfits with only one time
        self.gain_object2 = copy.deepcopy(self.gain_object)
        times_to_keep = self.gain_object.time_array[[1]]
        self.gain_object2.select(times=times_to_keep)
        self.gain_object2.write_calfits(write_file_calfits, clobber=True)

        # check for errors associated with times not included in data
        nt.assert_raises(ValueError, self.gain_object.select,
                         times=[np.min(self.gain_object.time_array) - self.gain_object.integration_time])

        # check for warnings and errors associated with unevenly spaced times
        self.gain_object2 = copy.deepcopy(self.gain_object)
        uvtest.checkWarnings(self.gain_object2.select, [], {'times': self.gain_object2.time_array[[0, 2, 3]]},
                             message='Selected times are not evenly spaced')
        nt.assert_raises(ValueError, self.gain_object2.write_calfits, write_file_calfits)

    def test_select_frequencies(self):
        old_history = self.gain_object.history
        freqs_to_keep = self.gain_object.freq_array[0, np.arange(73, 944)]

        self.gain_object2.select(frequencies=freqs_to_keep)

        nt.assert_equal(len(freqs_to_keep), self.gain_object2.Nfreqs)
        for f in freqs_to_keep:
            nt.assert_true(f in self.gain_object2.freq_array)
        for f in np.unique(self.gain_object2.freq_array):
            nt.assert_true(f in freqs_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific frequencies '
                        'using pyuvdata.', self.gain_object2.history)

        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        # test writing calfits with only one frequency
        self.gain_object2 = copy.deepcopy(self.gain_object)
        freqs_to_keep = self.gain_object.freq_array[0, 51]
        self.gain_object2.select(frequencies=freqs_to_keep)
        self.gain_object2.write_calfits(write_file_calfits, clobber=True)

        # check for errors associated with frequencies not included in data
        nt.assert_raises(ValueError, self.gain_object.select, frequencies=[np.max(self.gain_object.freq_array) + self.gain_object.channel_width])

        # check for warnings and errors associated with unevenly spaced frequencies
        self.gain_object2 = copy.deepcopy(self.gain_object)
        uvtest.checkWarnings(self.gain_object2.select, [], {'frequencies': self.gain_object2.freq_array[0, [0, 5, 6]]},
                             message='Selected frequencies are not evenly spaced')
        nt.assert_raises(ValueError, self.gain_object2.write_calfits, write_file_calfits)

    def test_select_freq_chans(self):
        old_history = self.gain_object.history
        chans_to_keep = np.arange(73, 944)

        self.gain_object2.select(freq_chans=chans_to_keep)

        nt.assert_equal(len(chans_to_keep), self.gain_object2.Nfreqs)
        for chan in chans_to_keep:
            nt.assert_true(self.gain_object.freq_array[0, chan] in self.gain_object2.freq_array)
        for f in np.unique(self.gain_object2.freq_array):
            nt.assert_true(f in self.gain_object.freq_array[0, chans_to_keep])

        nt.assert_equal(old_history + '  Downselected to specific frequencies '
                        'using pyuvdata.', self.gain_object2.history)

        # Test selecting both channels and frequencies
        freqs_to_keep = self.gain_object.freq_array[0, np.arange(930, 1000)]  # Overlaps with chans
        all_chans_to_keep = np.arange(73, 1000)

        self.gain_object2 = copy.deepcopy(self.gain_object)
        self.gain_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

        nt.assert_equal(len(all_chans_to_keep), self.gain_object2.Nfreqs)
        for chan in all_chans_to_keep:
            nt.assert_true(self.gain_object.freq_array[0, chan] in self.gain_object2.freq_array)
        for f in np.unique(self.gain_object2.freq_array):
            nt.assert_true(f in self.gain_object.freq_array[0, all_chans_to_keep])

    def test_select_polarizations(self):
        # add more jones terms to allow for better testing of selections
        while self.gain_object.Njones < 4:
            new_jones = np.min(self.gain_object.jones_array) - 1
            self.gain_object.jones_array = np.append(self.gain_object.jones_array, new_jones)
            self.gain_object.Njones += 1
            self.gain_object.flag_array = np.concatenate((self.gain_object.flag_array,
                                                          self.gain_object.flag_array[:, :, :, :, [-1]]),
                                                         axis=4)
            self.gain_object.gain_array = np.concatenate((self.gain_object.gain_array,
                                                          self.gain_object.gain_array[:, :, :, :, [-1]]),
                                                         axis=4)
            self.gain_object.quality_array = np.concatenate((self.gain_object.quality_array,
                                                             self.gain_object.quality_array[:, :, :, :, [-1]]),
                                                            axis=4)
        nt.assert_true(self.gain_object.check())
        self.gain_object2 = copy.deepcopy(self.gain_object)

        old_history = self.gain_object.history
        jones_to_keep = [-5, -6]

        self.gain_object2.select(jones=jones_to_keep)

        nt.assert_equal(len(jones_to_keep), self.gain_object2.Njones)
        for j in jones_to_keep:
            nt.assert_true(j in self.gain_object2.jones_array)
        for j in np.unique(self.gain_object2.jones_array):
            nt.assert_true(j in jones_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific jones polarization terms '
                        'using pyuvdata.', self.gain_object2.history)

        # check for errors associated with polarizations not included in data
        nt.assert_raises(ValueError, self.gain_object2.select, jones=[-3, -4])

        # check for warnings and errors associated with unevenly spaced polarizations
        uvtest.checkWarnings(self.gain_object.select, [], {'jones': self.gain_object.jones_array[[0, 1, 3]]},
                             message='Selected jones polarization terms are not evenly spaced')
        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        nt.assert_raises(ValueError, self.gain_object.write_calfits, write_file_calfits)

    def test_select(self):
        # now test selecting along all axes at once
        old_history = self.gain_object.history

        ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
        freqs_to_keep = self.gain_object.freq_array[0, np.arange(31, 56)]
        times_to_keep = self.gain_object.time_array[[1, 2]]
        jones_to_keep = [-5]

        self.gain_object2.select(antenna_nums=ants_to_keep, frequencies=freqs_to_keep,
                                 times=times_to_keep, jones=jones_to_keep)

        nt.assert_equal(len(ants_to_keep), self.gain_object2.Nants_data)
        for ant in ants_to_keep:
            nt.assert_true(ant in self.gain_object2.ant_array)
        for ant in self.gain_object2.ant_array:
            nt.assert_true(ant in ants_to_keep)

        nt.assert_equal(len(times_to_keep), self.gain_object2.Ntimes)
        for t in times_to_keep:
            nt.assert_true(t in self.gain_object2.time_array)
        for t in np.unique(self.gain_object2.time_array):
            nt.assert_true(t in times_to_keep)

        nt.assert_equal(len(freqs_to_keep), self.gain_object2.Nfreqs)
        for f in freqs_to_keep:
            nt.assert_true(f in self.gain_object2.freq_array)
        for f in np.unique(self.gain_object2.freq_array):
            nt.assert_true(f in freqs_to_keep)

        nt.assert_equal(len(jones_to_keep), self.gain_object2.Njones)
        for j in jones_to_keep:
            nt.assert_true(j in self.gain_object2.jones_array)
        for j in np.unique(self.gain_object2.jones_array):
            nt.assert_true(j in jones_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific antennas, '
                        'times, frequencies, jones polarization terms using pyuvdata.',
                        self.gain_object2.history)


class TestUVCalSelectDelay(object):
    def setUp(self):
        """Set up test"""
        self.delay_object = UVCal()
        delayfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')

        # add an input flag array to the file to test for that.
        write_file = os.path.join(DATA_PATH, 'test/outtest_input_flags.fits')
        uv_in = UVCal()
        message = delayfile + ' appears to be an old calfits format'
        uvtest.checkWarnings(uv_in.read_calfits, [delayfile], nwarnings=1,
                             message=message, category=[UserWarning])
        uv_in.input_flag_array = np.zeros(uv_in._input_flag_array.expected_shape(uv_in), dtype=bool)
        uv_in.write_calfits(write_file, clobber=True)

        self.delay_object.read_calfits(write_file)
        self.delay_object2 = copy.deepcopy(self.delay_object)

    def teardown(self):
        """Tear down test"""
        del(self.delay_object)
        del(self.delay_object2)

    def test_select_antennas(self):
        old_history = self.delay_object.history
        ants_to_keep = np.array([65, 96, 9, 97, 89, 22, 20, 72])
        self.delay_object2.select(antenna_nums=ants_to_keep)

        nt.assert_equal(len(ants_to_keep), self.delay_object2.Nants_data)
        for ant in ants_to_keep:
            nt.assert_true(ant in self.delay_object2.ant_array)
        for ant in self.delay_object2.ant_array:
            nt.assert_true(ant in ants_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific antennas '
                        'using pyuvdata.', self.delay_object2.history)

        # now test using antenna_names to specify antennas to keep
        self.delay_object3 = copy.deepcopy(self.delay_object)
        ants_to_keep = np.array(sorted(list(ants_to_keep)))
        ant_names = []
        for a in ants_to_keep:
            ind = np.where(self.delay_object3.antenna_numbers == a)[0][0]
            ant_names.append(self.delay_object3.antenna_names[ind])

        self.delay_object3.select(antenna_names=ant_names)

        nt.assert_equal(self.delay_object2, self.delay_object3)

        # check for errors associated with antennas not included in data, bad names or providing numbers and names
        nt.assert_raises(ValueError, self.delay_object.select,
                         antenna_nums=np.max(self.delay_object.ant_array) + np.arange(1, 3))
        nt.assert_raises(ValueError, self.delay_object.select, antenna_names='test1')
        nt.assert_raises(ValueError, self.delay_object.select,
                         antenna_nums=ants_to_keep, antenna_names=ant_names)

    def test_select_times(self):
        # add another time to allow for better testing of selections
        new_time = np.max(self.delay_object.time_array) + self.delay_object.integration_time
        self.delay_object.time_array = np.append(self.delay_object.time_array, new_time)
        self.delay_object.Ntimes += 1
        self.delay_object.flag_array = np.concatenate((self.delay_object.flag_array,
                                                       self.delay_object.flag_array[:, :, :, [-1], :]),
                                                      axis=3)
        self.delay_object.input_flag_array = np.concatenate((self.delay_object.input_flag_array,
                                                             self.delay_object.input_flag_array[:, :, :, [-1], :]),
                                                            axis=3)
        self.delay_object.delay_array = np.concatenate((self.delay_object.delay_array,
                                                        self.delay_object.delay_array[:, :, :, [-1], :]),
                                                       axis=3)
        self.delay_object.quality_array = np.concatenate((self.delay_object.quality_array,
                                                          self.delay_object.quality_array[:, :, :, [-1], :]),
                                                         axis=3)
        nt.assert_true(self.delay_object.check())
        self.delay_object2 = copy.deepcopy(self.delay_object)

        old_history = self.delay_object.history
        times_to_keep = self.delay_object.time_array[[2, 0]]

        self.delay_object2.select(times=times_to_keep)

        nt.assert_equal(len(times_to_keep), self.delay_object2.Ntimes)
        for t in times_to_keep:
            nt.assert_true(t in self.delay_object2.time_array)
        for t in np.unique(self.delay_object2.time_array):
            nt.assert_true(t in times_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific times '
                        'using pyuvdata.', self.delay_object2.history)

        # check for errors associated with times not included in data
        nt.assert_raises(ValueError, self.delay_object.select,
                         times=[np.min(self.delay_object.time_array) - self.delay_object.integration_time])

        # check for warnings and errors associated with unevenly spaced times
        self.delay_object2 = copy.deepcopy(self.delay_object)
        uvtest.checkWarnings(self.delay_object2.select, [], {'times': self.delay_object2.time_array[[0, 2, 3]]},
                             message='Selected times are not evenly spaced')
        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        nt.assert_raises(ValueError, self.delay_object2.write_calfits, write_file_calfits)

    def test_select_frequencies(self):
        old_history = self.delay_object.history
        freqs_to_keep = self.delay_object.freq_array[0, np.arange(73, 944)]

        self.delay_object2.select(frequencies=freqs_to_keep)

        nt.assert_equal(len(freqs_to_keep), self.delay_object2.Nfreqs)
        for f in freqs_to_keep:
            nt.assert_true(f in self.delay_object2.freq_array)
        for f in np.unique(self.delay_object2.freq_array):
            nt.assert_true(f in freqs_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific frequencies '
                        'using pyuvdata.', self.delay_object2.history)

        # check for errors associated with frequencies not included in data
        nt.assert_raises(ValueError, self.delay_object.select, frequencies=[np.max(self.delay_object.freq_array) + self.delay_object.channel_width])

        # check for warnings and errors associated with unevenly spaced frequencies
        self.delay_object2 = copy.deepcopy(self.delay_object)
        uvtest.checkWarnings(self.delay_object2.select, [], {'frequencies': self.delay_object2.freq_array[0, [0, 5, 6]]},
                             message='Selected frequencies are not evenly spaced')
        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        nt.assert_raises(ValueError, self.delay_object2.write_calfits, write_file_calfits)

    def test_select_freq_chans(self):
        old_history = self.delay_object.history
        chans_to_keep = np.arange(73, 944)

        self.delay_object2.select(freq_chans=chans_to_keep)

        nt.assert_equal(len(chans_to_keep), self.delay_object2.Nfreqs)
        for chan in chans_to_keep:
            nt.assert_true(self.delay_object.freq_array[0, chan] in self.delay_object2.freq_array)
        for f in np.unique(self.delay_object2.freq_array):
            nt.assert_true(f in self.delay_object.freq_array[0, chans_to_keep])

        nt.assert_equal(old_history + '  Downselected to specific frequencies '
                        'using pyuvdata.', self.delay_object2.history)

        # Test selecting both channels and frequencies
        freqs_to_keep = self.delay_object.freq_array[0, np.arange(930, 1000)]  # Overlaps with chans
        all_chans_to_keep = np.arange(73, 1000)

        self.delay_object2 = copy.deepcopy(self.delay_object)
        self.delay_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

        nt.assert_equal(len(all_chans_to_keep), self.delay_object2.Nfreqs)
        for chan in all_chans_to_keep:
            nt.assert_true(self.delay_object.freq_array[0, chan] in self.delay_object2.freq_array)
        for f in np.unique(self.delay_object2.freq_array):
            nt.assert_true(f in self.delay_object.freq_array[0, all_chans_to_keep])

    def test_select_polarizations(self):
        # add more jones terms to allow for better testing of selections
        while self.delay_object.Njones < 4:
            new_jones = np.min(self.delay_object.jones_array) - 1
            self.delay_object.jones_array = np.append(self.delay_object.jones_array, new_jones)
            self.delay_object.Njones += 1
            self.delay_object.flag_array = np.concatenate((self.delay_object.flag_array,
                                                           self.delay_object.flag_array[:, :, :, :, [-1]]),
                                                          axis=4)
            self.delay_object.input_flag_array = np.concatenate((self.delay_object.input_flag_array,
                                                                 self.delay_object.input_flag_array[:, :, :, :, [-1]]),
                                                                axis=4)
            self.delay_object.delay_array = np.concatenate((self.delay_object.delay_array,
                                                            self.delay_object.delay_array[:, :, :, :, [-1]]),
                                                           axis=4)
            self.delay_object.quality_array = np.concatenate((self.delay_object.quality_array,
                                                              self.delay_object.quality_array[:, :, :, :, [-1]]),
                                                             axis=4)
        nt.assert_true(self.delay_object.check())
        self.delay_object2 = copy.deepcopy(self.delay_object)

        old_history = self.delay_object.history
        jones_to_keep = [-5, -6]

        self.delay_object2.select(jones=jones_to_keep)

        nt.assert_equal(len(jones_to_keep), self.delay_object2.Njones)
        for j in jones_to_keep:
            nt.assert_true(j in self.delay_object2.jones_array)
        for j in np.unique(self.delay_object2.jones_array):
            nt.assert_true(j in jones_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific jones polarization terms '
                        'using pyuvdata.', self.delay_object2.history)

        # check for errors associated with polarizations not included in data
        nt.assert_raises(ValueError, self.delay_object2.select, jones=[-3, -4])

        # check for warnings and errors associated with unevenly spaced polarizations
        uvtest.checkWarnings(self.delay_object.select, [], {'jones': self.delay_object.jones_array[[0, 1, 3]]},
                             message='Selected jones polarization terms are not evenly spaced')
        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        nt.assert_raises(ValueError, self.delay_object.write_calfits, write_file_calfits)

    def test_select(self):
        # now test selecting along all axes at once
        old_history = self.delay_object.history

        ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
        freqs_to_keep = self.delay_object.freq_array[0, np.arange(31, 56)]
        times_to_keep = self.delay_object.time_array[[1, 2]]
        jones_to_keep = [-5]

        self.delay_object2.select(antenna_nums=ants_to_keep, frequencies=freqs_to_keep,
                                  times=times_to_keep, jones=jones_to_keep)

        nt.assert_equal(len(ants_to_keep), self.delay_object2.Nants_data)
        for ant in ants_to_keep:
            nt.assert_true(ant in self.delay_object2.ant_array)
        for ant in self.delay_object2.ant_array:
            nt.assert_true(ant in ants_to_keep)

        nt.assert_equal(len(times_to_keep), self.delay_object2.Ntimes)
        for t in times_to_keep:
            nt.assert_true(t in self.delay_object2.time_array)
        for t in np.unique(self.delay_object2.time_array):
            nt.assert_true(t in times_to_keep)

        nt.assert_equal(len(freqs_to_keep), self.delay_object2.Nfreqs)
        for f in freqs_to_keep:
            nt.assert_true(f in self.delay_object2.freq_array)
        for f in np.unique(self.delay_object2.freq_array):
            nt.assert_true(f in freqs_to_keep)

        nt.assert_equal(len(jones_to_keep), self.delay_object2.Njones)
        for j in jones_to_keep:
            nt.assert_true(j in self.delay_object2.jones_array)
        for j in np.unique(self.delay_object2.jones_array):
            nt.assert_true(j in jones_to_keep)

        nt.assert_equal(old_history + '  Downselected to specific antennas, '
                        'times, frequencies, jones polarization terms using pyuvdata.',
                        self.delay_object2.history)
