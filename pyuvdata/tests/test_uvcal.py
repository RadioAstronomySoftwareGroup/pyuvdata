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
        self.required_parameters = ['_Nfreqs', '_Njones', '_Ntimes', '_history',
                                    '_Nants_data', '_antenna_names', '_antenna_numbers',
                                    '_Nants_telescope', '_freq_array',
                                    '_jones_array', '_time_array',
                                    '_gain_convention', '_flag_array',
                                    '_quality_array', '_cal_type',
                                    '_x_orientation']

        self.required_properties = ['Nfreqs', 'Njones', 'Ntimes', 'history',
                                    'Nants_data', 'antenna_names', 'antenna_numbers',
                                    'Nants_telescope', 'freq_array',
                                    'jones_array', 'time_array',
                                    'gain_convention', 'flag_array',
                                    'quality_array', 'cal_type',
                                    'x_orientation']

        self.extra_parameters = ['_gain_array', '_delay_array',
                                 '_input_flag_array']

        self.extra_properties = ['gain_array', 'delay_array',
                                 'input_flag_array']

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

    def test_unexpected_attributes(self):
        "Test for extra attributes."
        expected_attributes = self.required_properties + \
            self.extra_properties + self.other_properties
        attributes = [i for i in self.uv_cal_object.__dict__.keys() if i[0] != '_']
        for a in attributes:
            nt.assert_true(a in expected_attributes,
                           msg='unexpected attribute ' + a + ' found in UVData')

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
        self.uv_cal_object = UVCal()
        self.testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
        self.uv_cal_object.read_calfits(self.testfile)
        self.uv_cal_object2 = copy.deepcopy(self.uv_cal_object)

    def teardown(self):
        """Tear down test"""
        del(self.uv_cal_object)
        del(self.uv_cal_object2)

    def test_equality(self):
        """Basic equality test"""
        nt.assert_equal(self.uv_cal_object, self.uv_cal_object)

    def test_check(self):
        """Test that parameter checks run properly"""
        nt.assert_true(self.uv_cal_object.check())

    def test_nants_data_telescope(self):
        self.uv_cal_object.Nants_data = self.uv_cal_object.Nants_telescope - 1
        nt.assert_true(self.uv_cal_object.check)
        self.uv_cal_object.Nants_data = self.uv_cal_object.Nants_telescope + 1
        nt.assert_raises(ValueError, self.uv_cal_object.check)

    def test_set_gain(self):
        self.uv_cal_object.set_gain()
        nt.assert_true(self.uv_cal_object._gain_array.required)
        nt.assert_false(self.uv_cal_object._delay_array.required)
        nt.assert_equal(self.uv_cal_object._gain_array.form, self.uv_cal_object._flag_array.form)
        nt.assert_equal(self.uv_cal_object._gain_array.form, self.uv_cal_object._quality_array.form)

    def test_set_delay(self):
        self.uv_cal_object.set_delay()
        nt.assert_true(self.uv_cal_object._delay_array.required)
        nt.assert_false(self.uv_cal_object._gain_array.required)
        nt.assert_equal(self.uv_cal_object._gain_array.form, self.uv_cal_object._flag_array.form)
        nt.assert_equal(self.uv_cal_object._delay_array.form, self.uv_cal_object._quality_array.form)

    def test_set_unknown(self):
        self.uv_cal_object.set_unknown_cal_type()
        nt.assert_false(self.uv_cal_object._delay_array.required)
        nt.assert_false(self.uv_cal_object._gain_array.required)
        nt.assert_equal(self.uv_cal_object._gain_array.form, self.uv_cal_object._flag_array.form)
        nt.assert_equal(self.uv_cal_object._gain_array.form, self.uv_cal_object._quality_array.form)


class TestUVCalSelectGain(object):
    def setUp(self):
        """Set up test"""
        self.gain_object = UVCal()
        self.gainfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.fitsA')
        self.gain_object.read_calfits(self.gainfile)
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
        self.gain_object3 = copy.deepcopy(self.gain_object)
        ants_to_keep = np.array(sorted(list(ants_to_keep)))
        ant_names = []
        for a in ants_to_keep:
            ind = np.where(self.gain_object3.antenna_numbers == a)[0][0]
            ant_names.append(self.gain_object3.antenna_names[ind])

        self.gain_object3.select(antenna_names=ant_names)

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

        # check for errors associated with times not included in data
        nt.assert_raises(ValueError, self.gain_object.select,
                         times=[np.min(self.gain_object.time_array) - self.gain_object.integration_time])

        # check for warnings and errors associated with unevenly spaced times
        # NOTE this test requires a file with more than 3 times.
        # self.gain_object2 = copy.deepcopy(self.gain_object)
        # status = uvtest.checkWarnings(self.gain_object2.select, [], {'times': self.gain_object2.time_array[0, [0, 2, 3]]},
        #                               message='Selected times are not evenly spaced')
        # nt.assert_true(status)
        # write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        # nt.assert_raises(ValueError, self.gain_object2.write_calfits, write_file_calfits)

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

        # check for errors associated with frequencies not included in data
        nt.assert_raises(ValueError, self.gain_object.select, frequencies=[np.max(self.gain_object.freq_array) + self.gain_object.channel_width])

        # check for warnings and errors associated with unevenly spaced frequencies
        self.gain_object2 = copy.deepcopy(self.gain_object)
        status = uvtest.checkWarnings(self.gain_object2.select, [], {'frequencies': self.gain_object2.freq_array[0, [0, 5, 6]]},
                                      message='Selected frequencies are not evenly spaced')
        nt.assert_true(status)
        write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
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

    # NOTE this test requires a file with more than 1 pol. more than 3 pols are required to fully test.
    # def test_select_polarizations(self):
    #     old_history = self.gain_object.history
    #     print(self.gain_object.Njones)
    #     print(self.gain_object.jones_array)
    #     # print(np.random.choice(np.arange(self.gain_object.Njones), 2, replace=False))
    #     jones_to_keep = [-5, -6]
    #
    #     self.gain_object2.select(jones=jones_to_keep)
    #
    #     nt.assert_equal(len(jones_to_keep), self.gain_object2.Njones)
    #     for j in jones_to_keep:
    #         nt.assert_true(j in self.gain_object2.jones_array)
    #     for j in np.unique(self.gain_object2.jones_array):
    #         nt.assert_true(j in jones_to_keep)
    #
    #     nt.assert_equal(old_history + '  Downselected to specific jones polarization terms '
    #                     'using pyuvdata.', self.gain_object2.history)
    #
    #     # check for errors associated with polarizations not included in data
    #     nt.assert_raises(ValueError, self.gain_object2.select, jones=[-3, -4])
    #
    #     # check for warnings and errors associated with unevenly spaced polarizations
    #     status = uvtest.checkWarnings(self.gain_object.select, [], {'jones': self.gain_object.polarization_array[[0, 1, 3]]},
    #                                   message='Selected jones polarization terms are not evenly spaced')
    #     nt.assert_true(status)
    #     write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    #     nt.assert_raises(ValueError, self.gain_object.write_calfits, write_file_calfits)

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
        self.delayfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')

        # add an input flag array to the file to test for that.
        write_file = os.path.join(DATA_PATH, 'test/outtest_input_flags.fits')
        uv_in = UVCal()
        uv_in.read_calfits(self.delayfile)
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
        # NOTE this test requires a file with more than 3 times.
        # self.delay_object2 = copy.deepcopy(self.delay_object)
        # status = uvtest.checkWarnings(self.delay_object2.select, [], {'times': self.delay_object2.time_array[0, [0, 2, 3]]},
        #                               message='Selected times are not evenly spaced')
        # nt.assert_true(status)
        # write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
        # nt.assert_raises(ValueError, self.delay_object2.write_calfits, write_file_calfits)

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
        status = uvtest.checkWarnings(self.delay_object2.select, [], {'frequencies': self.delay_object2.freq_array[0, [0, 5, 6]]},
                                      message='Selected frequencies are not evenly spaced')
        nt.assert_true(status)
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

    # NOTE this test requires a file with more than 1 pol. more than 3 pols are required to fully test.
    # def test_select_polarizations(self):
    #     old_history = self.delay_object.history
    #     print(self.delay_object.Njones)
    #     print(self.delay_object.jones_array)
    #     # print(np.random.choice(np.arange(self.delay_object.Njones), 2, replace=False))
    #     jones_to_keep = [-5, -6]
    #
    #     self.delay_object2.select(jones=jones_to_keep)
    #
    #     nt.assert_equal(len(jones_to_keep), self.delay_object2.Njones)
    #     for j in jones_to_keep:
    #         nt.assert_true(j in self.delay_object2.jones_array)
    #     for j in np.unique(self.delay_object2.jones_array):
    #         nt.assert_true(j in jones_to_keep)
    #
    #     nt.assert_equal(old_history + '  Downselected to specific jones polarization terms '
    #                     'using pyuvdata.', self.delay_object2.history)
    #
    #     # check for errors associated with polarizations not included in data
    #     nt.assert_raises(ValueError, self.delay_object2.select, jones=[-3, -4])
    #
    #     # check for warnings and errors associated with unevenly spaced polarizations
    #     status = uvtest.checkWarnings(self.delay_object.select, [], {'jones': self.delay_object.polarization_array[[0, 1, 3]]},
    #                                   message='Selected jones polarization terms are not evenly spaced')
    #     nt.assert_true(status)
    #     write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    #     nt.assert_raises(ValueError, self.delay_object.write_calfits, write_file_calfits)

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
