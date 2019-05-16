# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvcal object.

"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
import copy

from pyuvdata import UVCal
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


@pytest.fixture(scope='function')
def uvcal_data():
    """Set up some uvcal iter tests."""
    required_parameters = ['_Nfreqs', '_Njones', '_Ntimes', '_Nspws',
                           '_Nants_data', '_Nants_telescope',
                           '_antenna_names', '_antenna_numbers',
                           '_ant_array',
                           '_telescope_name', '_freq_array',
                           '_channel_width', '_spw_array',
                           '_jones_array', '_time_array', '_time_range',
                           '_integration_time',
                           '_gain_convention', '_flag_array',
                           '_quality_array', '_cal_type', '_cal_style',
                           '_x_orientation', '_history']

    required_properties = ['Nfreqs', 'Njones', 'Ntimes', 'Nspws',
                           'Nants_data', 'Nants_telescope',
                           'antenna_names', 'antenna_numbers',
                           'ant_array',
                           'telescope_name', 'freq_array',
                           'channel_width', 'spw_array',
                           'jones_array', 'time_array', 'time_range',
                           'integration_time',
                           'gain_convention', 'flag_array',
                           'quality_array', 'cal_type', 'cal_style',
                           'x_orientation', 'history']

    extra_parameters = ['_gain_array', '_delay_array', '_sky_field',
                        '_sky_catalog', '_ref_antenna_name', '_Nsources',
                        '_baseline_range', '_diffuse_model',
                        '_input_flag_array', '_freq_range',
                        '_observer', '_git_origin_cal',
                        '_git_hash_cal', '_total_quality_array',
                        '_extra_keywords']

    extra_properties = ['gain_array', 'delay_array', 'sky_field',
                        'sky_catalog', 'ref_antenna_name', 'Nsources',
                        'baseline_range', 'diffuse_model',
                        'input_flag_array', 'freq_range',
                        'observer', 'git_origin_cal',
                        'git_hash_cal', 'total_quality_array',
                        'extra_keywords']

    other_properties = ['pyuvdata_version_str']

    uv_cal_object = UVCal()

    # yields the data we need but will continue to the del call after tests
    yield uv_cal_object, required_parameters, required_properties, extra_parameters, extra_properties, other_properties

    # some post-test object cleanup
    del(uv_cal_object)
    return


def test_parameter_iter(uvcal_data):
    """Test expected parameters."""
    (uv_cal_object, required_parameters, required_properties,
     extra_parameters, extra_properties, other_properties) = uvcal_data
    all = []
    for prop in uv_cal_object:
        all.append(prop)
    for a in required_parameters + extra_parameters:
        assert a in all, 'expected attribute ' + a + ' not returned in object iterator'


def test_required_parameter_iter(uvcal_data):
    """Test expected required parameters."""
    (uv_cal_object, required_parameters, required_properties,
     extra_parameters, extra_properties, other_properties) = uvcal_data
    required = []
    for prop in uv_cal_object.required():
        required.append(prop)
    for a in required_parameters:
        assert a in required, 'expected attribute ' + a + ' not returned in required iterator'


def test_unexpected_parameters(uvcal_data):
    """Test for extra parameters."""
    (uv_cal_object, required_parameters, required_properties,
     extra_parameters, extra_properties, other_properties) = uvcal_data
    expected_parameters = required_parameters + extra_parameters
    attributes = [i for i in uv_cal_object.__dict__.keys() if i[0] == '_']
    for a in attributes:
        assert a in expected_parameters, 'unexpected parameter ' + a + ' found in UVCal'


def test_unexpected_attributes(uvcal_data):
    """Test for extra attributes."""
    (uv_cal_object, required_parameters, required_properties,
     extra_parameters, extra_properties, other_properties) = uvcal_data
    expected_attributes = required_properties + \
        extra_properties + other_properties
    attributes = [i for i in uv_cal_object.__dict__.keys() if i[0] != '_']
    for a in attributes:
        assert a in expected_attributes, 'unexpected attribute ' + a + ' found in UVCal'


def test_properties(uvcal_data):
    """Test that properties can be get and set properly."""
    (uv_cal_object, required_parameters, required_properties,
     extra_parameters, extra_properties, other_properties) = uvcal_data
    prop_dict = dict(list(zip(required_properties + extra_properties,
                              required_parameters + extra_parameters)))
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(uv_cal_object, k, rand_num)
        this_param = getattr(uv_cal_object, v)
        try:
            assert rand_num == this_param.value
        except(AssertionError):
            print('setting {prop_name} to a random number failed'.format(prop_name=k))
            raise(AssertionError)


@pytest.fixture(scope='function')
def gain_data():
    """Initialize for some basic uvcal tests."""
    gain_object = UVCal()
    gainfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    gain_object.read_calfits(gainfile)

    gain_object2 = copy.deepcopy(gain_object)
    delay_object = UVCal()
    delayfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
    delay_object.read_calfits(delayfile)

    class DataHolder(object):
        def __init__(self, gain_object, gain_object2, delay_object):
            self.gain_object = gain_object
            self.gain_object2 = gain_object2
            self.delay_object = delay_object

    gain_data = DataHolder(gain_object, gain_object2, delay_object)
    yield gain_data

    del(gain_data)


def test_equality(gain_data):
    """Basic equality test"""
    assert gain_data.gain_object == gain_data.gain_object


def test_check(gain_data):
    """Test that parameter checks run properly"""
    assert gain_data.gain_object.check()


def test_nants_data_telescope_larger(gain_data):
    # make sure it's okay for Nants_telescope to be strictly greater than Nants_data
    gain_data.gain_object.Nants_telescope += 1
    # add dummy information for "new antenna" to pass object check
    gain_data.gain_object.antenna_names = np.concatenate(
        (gain_data.gain_object.antenna_names, ["dummy_ant"]))
    gain_data.gain_object.antenna_numbers = np.concatenate(
        (gain_data.gain_object.antenna_numbers, [20]))
    assert gain_data.gain_object.check()


def test_ant_array_not_in_antnums(gain_data):
    # make sure an error is raised if antennas with data not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    gain_data.gain_object.antenna_names = gain_data.gain_object.antenna_names[1:]
    gain_data.gain_object.antenna_numbers = gain_data.gain_object.antenna_numbers[1:]
    gain_data.gain_object.Nants_telescope = gain_data.gain_object.antenna_numbers.size
    with pytest.raises(ValueError) as cm:
        gain_data.gain_object.check()
    assert str(cm.value).startswith('All antennas in ant_array must be in antenna_numbers')


def test_set_gain(gain_data):
    gain_data.delay_object.set_gain()
    assert gain_data.delay_object._gain_array.required
    assert not gain_data.delay_object._delay_array.required
    assert gain_data.delay_object._gain_array.form == gain_data.delay_object._flag_array.form
    assert gain_data.delay_object._gain_array.form == gain_data.delay_object._quality_array.form


def test_set_delay(gain_data):
    gain_data.gain_object.set_delay()
    assert gain_data.gain_object._delay_array.required
    assert not gain_data.gain_object._gain_array.required
    assert gain_data.gain_object._gain_array.form == gain_data.gain_object._flag_array.form
    assert gain_data.gain_object._delay_array.form == gain_data.gain_object._quality_array.form


def test_set_unknown(gain_data):
    gain_data.gain_object.set_unknown_cal_type()
    assert not gain_data.gain_object._delay_array.required
    assert not gain_data.gain_object._gain_array.required
    assert gain_data.gain_object._gain_array.form == gain_data.gain_object._flag_array.form
    assert gain_data.gain_object._gain_array.form == gain_data.gain_object._quality_array.form


def test_convert_filetype(gain_data):
    # error testing
    pytest.raises(ValueError, gain_data.gain_object._convert_to_filetype, 'uvfits')


def test_convert_to_gain(gain_data):
    conventions = ['minus', 'plus']
    for c in conventions:
        gain_data.new_object = copy.deepcopy(gain_data.delay_object)

        gain_data.new_object.convert_to_gain(delay_convention=c)
        assert np.isclose(np.max(np.absolute(gain_data.new_object.gain_array)), 1.,
                          rtol=gain_data.new_object._gain_array.tols[0],
                          atol=gain_data.new_object._gain_array.tols[1])
        assert np.isclose(np.min(np.absolute(gain_data.new_object.gain_array)), 1.,
                          rtol=gain_data.new_object._gain_array.tols[0],
                          atol=gain_data.new_object._gain_array.tols[1])

        if c == 'minus':
            conv = -1
        else:
            conv = 1
        assert np.allclose(np.angle(gain_data.new_object.gain_array[:, :, 10, :, :]) % (2 * np.pi),
                           (conv * 2 * np.pi * gain_data.delay_object.delay_array[:, :, 0, :, :]
                           * gain_data.delay_object.freq_array[0, 10]) % (2 * np.pi),
                           rtol=gain_data.new_object._gain_array.tols[0],
                           atol=gain_data.new_object._gain_array.tols[1])
        assert np.allclose(gain_data.delay_object.quality_array,
                           gain_data.new_object.quality_array[:, :, 10, :, :],
                           rtol=gain_data.new_object._quality_array.tols[0],
                           atol=gain_data.new_object._quality_array.tols[1])

        assert gain_data.new_object.history == (gain_data.delay_object.history
                                                + '  Converted from delays to gains using pyuvdata.')

    # test a file with a total_quality_array
    gain_data.new_object = copy.deepcopy(gain_data.delay_object)
    tqa_size = gain_data.new_object.delay_array.shape[1:]
    gain_data.new_object.total_quality_array = np.ones(tqa_size)
    gain_data.new_object.convert_to_gain(delay_convention='minus')
    assert np.isclose(np.max(np.absolute(gain_data.new_object.gain_array)), 1.,
                      rtol=gain_data.new_object._gain_array.tols[0],
                      atol=gain_data.new_object._gain_array.tols[1])
    assert np.isclose(np.min(np.absolute(gain_data.new_object.gain_array)), 1.,
                      rtol=gain_data.new_object._gain_array.tols[0],
                      atol=gain_data.new_object._gain_array.tols[1])
    assert np.allclose(np.angle(gain_data.new_object.gain_array[:, :, 10, :, :]) % (2 * np.pi),
                       (-1 * 2 * np.pi * gain_data.delay_object.delay_array[:, :, 0, :, :]
                        * gain_data.delay_object.freq_array[0, 10]) % (2 * np.pi),
                       rtol=gain_data.new_object._gain_array.tols[0],
                       atol=gain_data.new_object._gain_array.tols[1])
    assert np.allclose(gain_data.delay_object.quality_array,
                       gain_data.new_object.quality_array[:, :, 10, :, :],
                       rtol=gain_data.new_object._quality_array.tols[0],
                       atol=gain_data.new_object._quality_array.tols[1])

    assert gain_data.new_object.history == (gain_data.delay_object.history
                                            + '  Converted from delays to gains using pyuvdata.')

    # error testing
    pytest.raises(ValueError, gain_data.delay_object.convert_to_gain, delay_convention='bogus')
    pytest.raises(ValueError, gain_data.gain_object.convert_to_gain)
    gain_data.gain_object.set_unknown_cal_type()
    pytest.raises(ValueError, gain_data.gain_object.convert_to_gain)


def test_select_antennas(gain_data):
    old_history = gain_data.gain_object.history
    ants_to_keep = np.array([65, 96, 9, 97, 89, 22, 20, 72])
    gain_data.gain_object2.select(antenna_nums=ants_to_keep)

    assert len(ants_to_keep) == gain_data.gain_object2.Nants_data
    for ant in ants_to_keep:
        assert ant in gain_data.gain_object2.ant_array
    for ant in gain_data.gain_object2.ant_array:
        assert ant in ants_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific antennas using pyuvdata.',
                                    gain_data.gain_object2.history)

    # now test using antenna_names to specify antennas to keep
    ants_to_keep = np.array(sorted(list(ants_to_keep)))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(gain_data.gain_object.antenna_numbers == a)[0][0]
        ant_names.append(gain_data.gain_object.antenna_names[ind])

    gain_data.gain_object3 = gain_data.gain_object.select(antenna_names=ant_names, inplace=False)

    assert gain_data.gain_object2 == gain_data.gain_object3

    # check for errors associated with antennas not included in data, bad names or providing numbers and names
    pytest.raises(ValueError, gain_data.gain_object.select,
                  antenna_nums=np.max(gain_data.gain_object.ant_array) + np.arange(1, 3))
    pytest.raises(ValueError, gain_data.gain_object.select, antenna_names='test1')
    pytest.raises(ValueError, gain_data.gain_object.select,
                  antenna_nums=ants_to_keep, antenna_names=ant_names)

    # check that write_calfits works with Nants_data < Nants_telescope
    write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    gain_data.gain_object2.write_calfits(write_file_calfits, clobber=True)

    # check that reading it back in works too
    new_gain_object = UVCal()
    new_gain_object.read_calfits(write_file_calfits)
    assert gain_data.gain_object2 == new_gain_object

    # check that total_quality_array is handled properly when present
    gain_data.gain_object.total_quality_array = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    uvtest.checkWarnings(gain_data.gain_object.select, [], {'antenna_names': ant_names,
                                                            'inplace': True},
                         message='Cannot preserve total_quality_array')
    assert gain_data.gain_object.total_quality_array is None


def test_select_times(gain_data):
    # self.gain_object2 = copy.deepcopy(self.gain_object)

    old_history = gain_data.gain_object.history
    times_to_keep = gain_data.gain_object.time_array[2:5]

    gain_data.gain_object2.select(times=times_to_keep)

    assert len(times_to_keep) == gain_data.gain_object2.Ntimes
    for t in times_to_keep:
        assert t in gain_data.gain_object2.time_array
    for t in np.unique(gain_data.gain_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific times using pyuvdata.',
                                    gain_data.gain_object2.history)

    write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    # test writing calfits with only one time
    gain_data.gain_object2 = copy.deepcopy(gain_data.gain_object)
    times_to_keep = gain_data.gain_object.time_array[[1]]
    gain_data.gain_object2.select(times=times_to_keep)
    gain_data.gain_object2.write_calfits(write_file_calfits, clobber=True)

    # check for errors associated with times not included in data
    pytest.raises(ValueError, gain_data.gain_object.select,
                  times=[np.min(gain_data.gain_object.time_array) - gain_data.gain_object.integration_time])

    # check for warnings and errors associated with unevenly spaced times
    gain_data.gain_object2 = copy.deepcopy(gain_data.gain_object)
    uvtest.checkWarnings(gain_data.gain_object2.select, [], {'times': gain_data.gain_object2.time_array[[0, 2, 3]]},
                         message='Selected times are not evenly spaced')
    pytest.raises(ValueError, gain_data.gain_object2.write_calfits, write_file_calfits)


def test_select_frequencies(gain_data):
    old_history = gain_data.gain_object.history
    freqs_to_keep = gain_data.gain_object.freq_array[0, np.arange(4, 8)]

    # add dummy total_quality_array
    gain_data.gain_object.total_quality_array = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    gain_data.gain_object2.total_quality_array = np.zeros(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))

    gain_data.gain_object2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == gain_data.gain_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in gain_data.gain_object2.freq_array
    for f in np.unique(gain_data.gain_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    gain_data.gain_object2.history)

    write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    # test writing calfits with only one frequency
    gain_data.gain_object2 = copy.deepcopy(gain_data.gain_object)
    freqs_to_keep = gain_data.gain_object.freq_array[0, 5]
    gain_data.gain_object2.select(frequencies=freqs_to_keep)
    gain_data.gain_object2.write_calfits(write_file_calfits, clobber=True)

    # check for errors associated with frequencies not included in data
    pytest.raises(ValueError, gain_data.gain_object.select,
                  frequencies=[np.max(gain_data.gain_object.freq_array) + gain_data.gain_object.channel_width])

    # check for warnings and errors associated with unevenly spaced frequencies
    gain_data.gain_object2 = copy.deepcopy(gain_data.gain_object)
    uvtest.checkWarnings(gain_data.gain_object2.select, [], {'frequencies': gain_data.gain_object2.freq_array[0, [0, 5, 6]]},
                         message='Selected frequencies are not evenly spaced')
    pytest.raises(ValueError, gain_data.gain_object2.write_calfits, write_file_calfits)


def test_select_freq_chans(gain_data):
    old_history = gain_data.gain_object.history
    chans_to_keep = np.arange(4, 8)

    # add dummy total_quality_array
    gain_data.gain_object.total_quality_array = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    gain_data.gain_object2.total_quality_array = np.zeros(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))

    gain_data.gain_object2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == gain_data.gain_object2.Nfreqs
    for chan in chans_to_keep:
        assert gain_data.gain_object.freq_array[0, chan] in gain_data.gain_object2.freq_array
    for f in np.unique(gain_data.gain_object2.freq_array):
        assert f in gain_data.gain_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    gain_data.gain_object2.history)

    # Test selecting both channels and frequencies
    freqs_to_keep = gain_data.gain_object.freq_array[0, np.arange(7, 10)]  # Overlaps with chans
    all_chans_to_keep = np.arange(4, 10)

    gain_data.gain_object2 = copy.deepcopy(gain_data.gain_object)
    gain_data.gain_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == gain_data.gain_object2.Nfreqs
    for chan in all_chans_to_keep:
        assert gain_data.gain_object.freq_array[0, chan] in gain_data.gain_object2.freq_array
    for f in np.unique(gain_data.gain_object2.freq_array):
        assert f in gain_data.gain_object.freq_array[0, all_chans_to_keep]


def test_select_polarizations(gain_data):
    # add more jones terms to allow for better testing of selections
    while gain_data.gain_object.Njones < 4:
        new_jones = np.min(gain_data.gain_object.jones_array) - 1
        gain_data.gain_object.jones_array = np.append(gain_data.gain_object.jones_array, new_jones)
        gain_data.gain_object.Njones += 1
        gain_data.gain_object.flag_array = np.concatenate((gain_data.gain_object.flag_array,
                                                           gain_data.gain_object.flag_array[:, :, :, :, [-1]]),
                                                          axis=4)
        gain_data.gain_object.gain_array = np.concatenate((gain_data.gain_object.gain_array,
                                                           gain_data.gain_object.gain_array[:, :, :, :, [-1]]),
                                                          axis=4)
        gain_data.gain_object.quality_array = np.concatenate((gain_data.gain_object.quality_array,
                                                              gain_data.gain_object.quality_array[:, :, :, :, [-1]]),
                                                             axis=4)
    # add dummy total_quality_array
    gain_data.gain_object.total_quality_array = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))

    assert gain_data.gain_object.check()
    gain_data.gain_object2 = copy.deepcopy(gain_data.gain_object)

    old_history = gain_data.gain_object.history
    jones_to_keep = [-5, -6]

    gain_data.gain_object2.select(jones=jones_to_keep)

    assert len(jones_to_keep) == gain_data.gain_object2.Njones
    for j in jones_to_keep:
        assert j in gain_data.gain_object2.jones_array
    for j in np.unique(gain_data.gain_object2.jones_array):
        assert j in jones_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific jones polarization terms '
                                    'using pyuvdata.',
                                    gain_data.gain_object2.history)

    # check for errors associated with polarizations not included in data
    pytest.raises(ValueError, gain_data.gain_object2.select, jones=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    uvtest.checkWarnings(gain_data.gain_object.select, [], {'jones': gain_data.gain_object.jones_array[[0, 1, 3]]},
                         message='Selected jones polarization terms are not evenly spaced')
    write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    pytest.raises(ValueError, gain_data.gain_object.write_calfits, write_file_calfits)


def test_select(gain_data):
    # now test selecting along all axes at once
    old_history = gain_data.gain_object.history

    ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
    freqs_to_keep = gain_data.gain_object.freq_array[0, np.arange(2, 5)]
    times_to_keep = gain_data.gain_object.time_array[[1, 2]]
    jones_to_keep = [-5]

    gain_data.gain_object2.select(antenna_nums=ants_to_keep, frequencies=freqs_to_keep,
                                  times=times_to_keep, jones=jones_to_keep)

    assert len(ants_to_keep) == gain_data.gain_object2.Nants_data
    for ant in ants_to_keep:
        assert ant in gain_data.gain_object2.ant_array
    for ant in gain_data.gain_object2.ant_array:
        assert ant in ants_to_keep

    assert len(times_to_keep) == gain_data.gain_object2.Ntimes
    for t in times_to_keep:
        assert t in gain_data.gain_object2.time_array
    for t in np.unique(gain_data.gain_object2.time_array):
        assert t in times_to_keep

    assert len(freqs_to_keep) == gain_data.gain_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in gain_data.gain_object2.freq_array
    for f in np.unique(gain_data.gain_object2.freq_array):
        assert f in freqs_to_keep

    assert len(jones_to_keep) == gain_data.gain_object2.Njones
    for j in jones_to_keep:
        assert j in gain_data.gain_object2.jones_array
    for j in np.unique(gain_data.gain_object2.jones_array):
        assert j in jones_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific antennas, times, '
                                    'frequencies, jones polarization terms '
                                    'using pyuvdata.',
                                    gain_data.gain_object2.history)


@pytest.fixture(scope='function')
def delay_data():
    """Initialization for some basic uvcal tests."""

    delay_object = UVCal()
    delayfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')

    # add an input flag array to the file to test for that.
    write_file = os.path.join(DATA_PATH, 'test/outtest_input_flags.fits')
    uv_in = UVCal()
    uv_in.read_calfits(delayfile)
    uv_in.input_flag_array = np.zeros(uv_in._input_flag_array.expected_shape(uv_in), dtype=bool)
    uv_in.write_calfits(write_file, clobber=True)
    delay_object.read_calfits(write_file)

    class DataHolder(object):
        def __init__(self, delay_object):
            self.delay_object = delay_object
            self.delay_object2 = copy.deepcopy(delay_object)

    delay_data = DataHolder(delay_object)

    # yield the data for testing, then del after tests finish
    yield delay_data

    del(delay_data)


def test_select_antennas_delay(delay_data):
    old_history = delay_data.delay_object.history
    ants_to_keep = np.array([65, 96, 9, 97, 89, 22, 20, 72])
    delay_data.delay_object2.select(antenna_nums=ants_to_keep)

    assert len(ants_to_keep) == delay_data.delay_object2.Nants_data
    for ant in ants_to_keep:
        assert ant in delay_data.delay_object2.ant_array
    for ant in delay_data.delay_object2.ant_array:
        assert ant in ants_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific antennas using pyuvdata.',
                                    delay_data.delay_object2.history)

    # now test using antenna_names to specify antennas to keep
    delay_data.delay_object3 = copy.deepcopy(delay_data.delay_object)
    ants_to_keep = np.array(sorted(list(ants_to_keep)))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(delay_data.delay_object3.antenna_numbers == a)[0][0]
        ant_names.append(delay_data.delay_object3.antenna_names[ind])

    delay_data.delay_object3.select(antenna_names=ant_names)

    assert delay_data.delay_object2 == delay_data.delay_object3

    # check for errors associated with antennas not included in data, bad names or providing numbers and names
    pytest.raises(ValueError, delay_data.delay_object.select,
                  antenna_nums=np.max(delay_data.delay_object.ant_array) + np.arange(1, 3))
    pytest.raises(ValueError, delay_data.delay_object.select, antenna_names='test1')
    pytest.raises(ValueError, delay_data.delay_object.select,
                  antenna_nums=ants_to_keep, antenna_names=ant_names)

    # check that total_quality_array is handled properly when present
    delay_data.delay_object.total_quality_array = np.zeros(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    uvtest.checkWarnings(delay_data.delay_object.select, [],
                         {'antenna_names': ant_names, 'inplace': True},
                         message='Cannot preserve total_quality_array')
    assert delay_data.delay_object.total_quality_array is None


def test_select_times_delay(delay_data):
    old_history = delay_data.delay_object.history
    times_to_keep = delay_data.delay_object.time_array[2:5]

    delay_data.delay_object2.select(times=times_to_keep)

    assert len(times_to_keep) == delay_data.delay_object2.Ntimes
    for t in times_to_keep:
        assert t in delay_data.delay_object2.time_array
    for t in np.unique(delay_data.delay_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific times using pyuvdata.',
                                    delay_data.delay_object2.history)

    # check for errors associated with times not included in data
    pytest.raises(ValueError, delay_data.delay_object.select,
                  times=[np.min(delay_data.delay_object.time_array) - delay_data.delay_object.integration_time])

    # check for warnings and errors associated with unevenly spaced times
    delay_data.delay_object2 = copy.deepcopy(delay_data.delay_object)
    uvtest.checkWarnings(delay_data.delay_object2.select, [], {'times': delay_data.delay_object2.time_array[[0, 2, 3]]},
                         message='Selected times are not evenly spaced')
    write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    pytest.raises(ValueError, delay_data.delay_object2.write_calfits, write_file_calfits)


def test_select_frequencies_delay(delay_data):
    old_history = delay_data.delay_object.history
    freqs_to_keep = delay_data.delay_object.freq_array[0, np.arange(73, 944)]

    # add dummy total_quality_array
    delay_data.delay_object.total_quality_array = np.zeros(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    delay_data.delay_object2.total_quality_array = np.zeros(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))

    delay_data.delay_object2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == delay_data.delay_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in delay_data.delay_object2.freq_array
    for f in np.unique(delay_data.delay_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    delay_data.delay_object2.history)

    # check for errors associated with frequencies not included in data
    pytest.raises(ValueError, delay_data.delay_object.select,
                  frequencies=[np.max(delay_data.delay_object.freq_array) + delay_data.delay_object.channel_width])

    # check for warnings and errors associated with unevenly spaced frequencies
    delay_data.delay_object2 = copy.deepcopy(delay_data.delay_object)
    uvtest.checkWarnings(delay_data.delay_object2.select, [], {'frequencies': delay_data.delay_object2.freq_array[0, [0, 5, 6]]},
                         message='Selected frequencies are not evenly spaced')
    write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    pytest.raises(ValueError, delay_data.delay_object2.write_calfits, write_file_calfits)


def test_select_freq_chans_delay(delay_data):
    old_history = delay_data.delay_object.history
    chans_to_keep = np.arange(73, 944)

    # add dummy total_quality_array
    delay_data.delay_object.total_quality_array = np.zeros(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    delay_data.delay_object2.total_quality_array = np.zeros(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))

    delay_data.delay_object2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == delay_data.delay_object2.Nfreqs
    for chan in chans_to_keep:
        assert delay_data.delay_object.freq_array[0, chan] in delay_data.delay_object2.freq_array
    for f in np.unique(delay_data.delay_object2.freq_array):
        assert f in delay_data.delay_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    delay_data.delay_object2.history)

    # Test selecting both channels and frequencies
    freqs_to_keep = delay_data.delay_object.freq_array[0, np.arange(930, 1000)]  # Overlaps with chans
    all_chans_to_keep = np.arange(73, 1000)

    delay_data.delay_object2 = copy.deepcopy(delay_data.delay_object)
    delay_data.delay_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == delay_data.delay_object2.Nfreqs
    for chan in all_chans_to_keep:
        assert delay_data.delay_object.freq_array[0, chan] in delay_data.delay_object2.freq_array
    for f in np.unique(delay_data.delay_object2.freq_array):
        assert f in delay_data.delay_object.freq_array[0, all_chans_to_keep]


def test_select_polarizations_delay(delay_data):
    # add more jones terms to allow for better testing of selections
    while delay_data.delay_object.Njones < 4:
        new_jones = np.min(delay_data.delay_object.jones_array) - 1
        delay_data.delay_object.jones_array = np.append(delay_data.delay_object.jones_array, new_jones)
        delay_data.delay_object.Njones += 1
        delay_data.delay_object.flag_array = np.concatenate((delay_data.delay_object.flag_array,
                                                             delay_data.delay_object.flag_array[:, :, :, :, [-1]]),
                                                            axis=4)
        delay_data.delay_object.input_flag_array = np.concatenate((delay_data.delay_object.input_flag_array,
                                                                   delay_data.delay_object.input_flag_array[:, :, :, :, [-1]]),
                                                                  axis=4)
        delay_data.delay_object.delay_array = np.concatenate((delay_data.delay_object.delay_array,
                                                              delay_data.delay_object.delay_array[:, :, :, :, [-1]]),
                                                             axis=4)
        delay_data.delay_object.quality_array = np.concatenate((delay_data.delay_object.quality_array,
                                                                delay_data.delay_object.quality_array[:, :, :, :, [-1]]),
                                                               axis=4)
    # add dummy total_quality_array
    delay_data.delay_object.total_quality_array = np.zeros(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    assert delay_data.delay_object.check()
    delay_data.delay_object2 = copy.deepcopy(delay_data.delay_object)

    old_history = delay_data.delay_object.history
    jones_to_keep = [-5, -6]

    delay_data.delay_object2.select(jones=jones_to_keep)

    assert len(jones_to_keep) == delay_data.delay_object2.Njones
    for j in jones_to_keep:
        assert j in delay_data.delay_object2.jones_array
    for j in np.unique(delay_data.delay_object2.jones_array):
        assert j in jones_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific jones polarization terms '
                                    'using pyuvdata.',
                                    delay_data.delay_object2.history)

    # check for errors associated with polarizations not included in data
    pytest.raises(ValueError, delay_data.delay_object2.select, jones=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    uvtest.checkWarnings(delay_data.delay_object.select, [], {'jones': delay_data.delay_object.jones_array[[0, 1, 3]]},
                         message='Selected jones polarization terms are not evenly spaced')
    write_file_calfits = os.path.join(DATA_PATH, 'test/select_test.calfits')
    pytest.raises(ValueError, delay_data.delay_object.write_calfits, write_file_calfits)


def test_select_delay(delay_data):
    # now test selecting along all axes at once
    old_history = delay_data.delay_object.history

    ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
    freqs_to_keep = delay_data.delay_object.freq_array[0, np.arange(31, 56)]
    times_to_keep = delay_data.delay_object.time_array[[1, 2]]
    jones_to_keep = [-5]

    delay_data.delay_object2.select(antenna_nums=ants_to_keep, frequencies=freqs_to_keep,
                                    times=times_to_keep, jones=jones_to_keep)

    assert len(ants_to_keep) == delay_data.delay_object2.Nants_data
    for ant in ants_to_keep:
        assert ant in delay_data.delay_object2.ant_array
    for ant in delay_data.delay_object2.ant_array:
        assert ant in ants_to_keep

    assert len(times_to_keep) == delay_data.delay_object2.Ntimes
    for t in times_to_keep:
        assert t in delay_data.delay_object2.time_array
    for t in np.unique(delay_data.delay_object2.time_array):
        assert t in times_to_keep

    assert len(freqs_to_keep) == delay_data.delay_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in delay_data.delay_object2.freq_array
    for f in np.unique(delay_data.delay_object2.freq_array):
        assert f in freqs_to_keep

    assert len(jones_to_keep) == delay_data.delay_object2.Njones
    for j in jones_to_keep:
        assert j in delay_data.delay_object2.jones_array
    for j in np.unique(delay_data.delay_object2.jones_array):
        assert j in jones_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific antennas, times, '
                                    'frequencies, jones polarization terms '
                                    'using pyuvdata.',
                                    delay_data.delay_object2.history)


def test_add_antennas(gain_data):
    """Test adding antennas between two UVCal objects"""
    gain_object_full = copy.deepcopy(gain_data.gain_object)
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    gain_data.gain_object.select(antenna_nums=ants1)
    gain_data.gain_object2.select(antenna_nums=ants2)
    gain_data.gain_object += gain_data.gain_object2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(gain_object_full.history
                                    + '  Downselected to specific '
                                    'antennas using pyuvdata. Combined '
                                    'data along antenna axis using pyuvdata.',
                                    gain_data.gain_object.history)
    gain_data.gain_object.history = gain_object_full.history
    assert gain_data.gain_object == gain_object_full

    # test for when total_quality_array is present
    gain_data.gain_object.select(antenna_nums=ants1)
    gain_data.gain_object.total_quality_array = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    uvtest.checkWarnings(gain_data.gain_object.__iadd__, [gain_data.gain_object2],
                         message='Total quality array detected')
    assert gain_data.gain_object.total_quality_array is None


def test_add_frequencies(gain_data):
    """Test adding frequencies between two UVCal objects"""
    gain_object_full = copy.deepcopy(gain_data.gain_object)
    freqs1 = gain_data.gain_object.freq_array[0, np.arange(0, 5)]
    freqs2 = gain_data.gain_object2.freq_array[0, np.arange(5, 10)]
    gain_data.gain_object.select(frequencies=freqs1)
    gain_data.gain_object2.select(frequencies=freqs2)
    gain_data.gain_object += gain_data.gain_object2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(gain_object_full.history
                                    + '  Downselected to specific '
                                    'frequencies using pyuvdata. Combined '
                                    'data along frequency axis using pyuvdata.',
                                    gain_data.gain_object.history)
    gain_data.gain_object.history = gain_object_full.history
    assert gain_data.gain_object == gain_object_full

    # test for when total_quality_array is present in first file but not second
    gain_data.gain_object.select(frequencies=freqs1)
    tqa = np.ones(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.zeros(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    gain_data.gain_object.total_quality_array = tqa
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in second file but not first
    gain_data.gain_object.select(frequencies=freqs1)
    tqa = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.ones(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    gain_data.gain_object.total_quality_array = None
    gain_data.gain_object2.total_quality_array = tqa2
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in both
    gain_data.gain_object.select(frequencies=freqs1)
    tqa = np.ones(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.ones(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    gain_data.gain_object.total_quality_array = tqa
    gain_data.gain_object2.total_quality_array = tqa2
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])

    # Out of order - freqs
    gain_data.gain_object = copy.deepcopy(gain_object_full)
    gain_data.gain_object2 = copy.deepcopy(gain_object_full)
    gain_data.gain_object.select(frequencies=freqs2)
    gain_data.gain_object2.select(frequencies=freqs1)
    gain_data.gain_object += gain_data.gain_object2
    gain_data.gain_object.history = gain_object_full.history
    assert gain_data.gain_object == gain_object_full


def test_add_times(gain_data):
    """Test adding times between two UVCal objects"""
    gain_object_full = copy.deepcopy(gain_data.gain_object)
    Nt2 = gain_data.gain_object.Ntimes // 2
    times1 = gain_data.gain_object.time_array[:Nt2]
    times2 = gain_data.gain_object.time_array[Nt2:]
    gain_data.gain_object.select(times=times1)
    gain_data.gain_object2.select(times=times2)
    gain_data.gain_object += gain_data.gain_object2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(gain_object_full.history
                                    + '  Downselected to specific '
                                    'times using pyuvdata. Combined '
                                    'data along time axis using pyuvdata.',
                                    gain_data.gain_object.history)
    gain_data.gain_object.history = gain_object_full.history
    assert gain_data.gain_object == gain_object_full

    # test for when total_quality_array is present in first file but not second
    gain_data.gain_object.select(times=times1)
    tqa = np.ones(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.zeros(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    gain_data.gain_object.total_quality_array = tqa
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in second file but not first
    gain_data.gain_object.select(times=times1)
    tqa = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.ones(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    gain_data.gain_object.total_quality_array = None
    gain_data.gain_object2.total_quality_array = tqa2
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in both
    gain_data.gain_object.select(times=times1)
    tqa = np.ones(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.ones(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    gain_data.gain_object.total_quality_array = tqa
    gain_data.gain_object2.total_quality_array = tqa2
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])


def test_add_jones(gain_data):
    """Test adding Jones axes between two UVCal objects"""
    gain_object_original = copy.deepcopy(gain_data.gain_object)
    # artificially change the Jones value to permit addition
    gain_data.gain_object2.jones_array[0] = -6
    gain_data.gain_object += gain_data.gain_object2

    # check dimensionality of resulting object
    assert gain_data.gain_object.gain_array.shape[-1] == 2
    assert sorted(gain_data.gain_object.jones_array) == [-6, -5]

    # test for when total_quality_array is present in first file but not second
    gain_data.gain_object = copy.deepcopy(gain_object_original)
    tqa = np.ones(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.zeros(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    gain_data.gain_object.total_quality_array = tqa
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in second file but not first
    gain_data.gain_object = copy.deepcopy(gain_object_original)
    tqa = np.zeros(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.ones(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    gain_data.gain_object2.total_quality_array = tqa2
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in both
    gain_data.gain_object = copy.deepcopy(gain_object_original)
    tqa = np.ones(
        gain_data.gain_object._total_quality_array.expected_shape(gain_data.gain_object))
    tqa2 = np.ones(
        gain_data.gain_object2._total_quality_array.expected_shape(gain_data.gain_object2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    gain_data.gain_object.total_quality_array = tqa
    gain_data.gain_object2.total_quality_array = tqa2
    gain_data.gain_object += gain_data.gain_object2
    assert np.allclose(gain_data.gain_object.total_quality_array, tot_tqa,
                       rtol=gain_data.gain_object._total_quality_array.tols[0],
                       atol=gain_data.gain_object._total_quality_array.tols[1])


def test_add(gain_data):
    """Test miscellaneous aspects of add method"""
    # test not-in-place addition
    gain_object = copy.deepcopy(gain_data.gain_object)
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    gain_data.gain_object.select(antenna_nums=ants1)
    gain_data.gain_object2.select(antenna_nums=ants2)
    gain_object_add = gain_data.gain_object + gain_data.gain_object2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(gain_object.history
                                    + '  Downselected to specific '
                                    'antennas using pyuvdata. Combined '
                                    'data along antenna axis using pyuvdata.',
                                    gain_object_add.history)
    gain_object_add.history = gain_object.history
    assert gain_object_add == gain_object

    # test history concatenation
    gain_data.gain_object.history = gain_object.history
    gain_data.gain_object2.history = 'Some random history string OMNI_RUN:'
    gain_data.gain_object += gain_data.gain_object2
    assert uvutils._check_histories(gain_object.history
                                    + ' Combined data along antenna axis '
                                    'using pyuvdata. Some random '
                                    'history string',
                                    gain_data.gain_object.history)


def test_add_multiple_axes(gain_data):
    """Test addition along multiple axes"""
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    freqs1 = gain_data.gain_object.freq_array[0, np.arange(0, 5)]
    freqs2 = gain_data.gain_object2.freq_array[0, np.arange(5, 10)]
    Nt2 = gain_data.gain_object.Ntimes // 2
    times1 = gain_data.gain_object.time_array[:Nt2]
    times2 = gain_data.gain_object.time_array[Nt2:]
    # artificially change the Jones value to permit addition
    gain_data.gain_object2.jones_array[0] = -6

    # perform select
    gain_data.gain_object.select(antenna_nums=ants1, frequencies=freqs1,
                                 times=times1)
    gain_data.gain_object2.select(antenna_nums=ants2, frequencies=freqs2,
                                  times=times2)

    gain_data.gain_object += gain_data.gain_object2

    # check resulting dimensionality
    assert len(gain_data.gain_object.ant_array) == 19
    assert len(gain_data.gain_object.freq_array[0, :]) == 10
    assert len(gain_data.gain_object.time_array) == gain_data.gain_object.Ntimes
    assert len(gain_data.gain_object.jones_array) == 2


def test_add_errors(gain_data):
    """Test behavior that will raise errors"""
    # test addition of two identical objects
    pytest.raises(ValueError, gain_data.gain_object.__add__, gain_data.gain_object2)

    # test addition of UVCal and non-UVCal object (empty list)
    pytest.raises(ValueError, gain_data.gain_object.__add__, [])

    # test compatibility param mismatch
    gain_data.gain_object2.telescope_name = "PAPER"
    pytest.raises(ValueError, gain_data.gain_object.__add__, gain_data.gain_object2)


def test_jones_warning(gain_data):
    """Test having non-contiguous Jones elements"""
    gain_data.gain_object2.jones_array[0] = -6
    gain_data.gain_object += gain_data.gain_object2
    gain_data.gain_object2.jones_array[0] = -8
    uvtest.checkWarnings(gain_data.gain_object.__iadd__, [gain_data.gain_object2],
                         message='Combined Jones elements')
    assert sorted(gain_data.gain_object.jones_array) == [-8, -6, -5]


def test_frequency_warnings(gain_data):
    """Test having uneven or non-contiguous frequencies"""
    # test having unevenly spaced frequency separations
    go1 = copy.deepcopy(gain_data.gain_object)
    go2 = copy.deepcopy(gain_data.gain_object2)
    freqs1 = gain_data.gain_object.freq_array[0, np.arange(0, 5)]
    freqs2 = gain_data.gain_object2.freq_array[0, np.arange(5, 10)]
    gain_data.gain_object.select(frequencies=freqs1)
    gain_data.gain_object2.select(frequencies=freqs2)

    # change the last frequency bin to be smaller than the others
    df = gain_data.gain_object2.freq_array[0, -1] - gain_data.gain_object2.freq_array[0, -2]
    gain_data.gain_object2.freq_array[0, -1] = gain_data.gain_object2.freq_array[0, -2] + df / 2
    uvtest.checkWarnings(gain_data.gain_object.__iadd__, [gain_data.gain_object2],
                         message='Combined frequencies are not evenly spaced')
    assert len(gain_data.gain_object.freq_array[0, :]) == gain_data.gain_object.Nfreqs

    # now check having "non-contiguous" frequencies
    gain_data.gain_object = copy.deepcopy(go1)
    gain_data.gain_object2 = copy.deepcopy(go2)
    freqs1 = gain_data.gain_object.freq_array[0, np.arange(0, 5)]
    freqs2 = gain_data.gain_object2.freq_array[0, np.arange(5, 10)]
    gain_data.gain_object.select(frequencies=freqs1)
    gain_data.gain_object2.select(frequencies=freqs2)

    # artificially space out frequencies
    gain_data.gain_object.freq_array[0, :] *= 10
    gain_data.gain_object2.freq_array[0, :] *= 10
    uvtest.checkWarnings(gain_data.gain_object.__iadd__, [gain_data.gain_object2],
                         message='Combined frequencies are not contiguous')
    freqs1 *= 10
    freqs2 *= 10
    freqs = np.concatenate([freqs1, freqs2])
    assert np.allclose(gain_data.gain_object.freq_array[0, :], freqs,
                       rtol=gain_data.gain_object._freq_array.tols[0],
                       atol=gain_data.gain_object._freq_array.tols[1])


def test_parameter_warnings(gain_data):
    """Test changing a parameter that will raise a warning"""
    # change observer and select frequencies
    gain_data.gain_object2.observer = 'mystery_person'
    freqs1 = gain_data.gain_object.freq_array[0, np.arange(0, 5)]
    freqs2 = gain_data.gain_object2.freq_array[0, np.arange(5, 10)]
    gain_data.gain_object.select(frequencies=freqs1)
    gain_data.gain_object2.select(frequencies=freqs2)
    uvtest.checkWarnings(gain_data.gain_object.__iadd__, [gain_data.gain_object2],
                         message='UVParameter observer does not match')
    freqs = np.concatenate([freqs1, freqs2])
    assert np.allclose(gain_data.gain_object.freq_array, freqs,
                       rtol=gain_data.gain_object._freq_array.tols[0],
                       atol=gain_data.gain_object._freq_array.tols[1])


def test_multi_files(gain_data):
    """Test read function when multiple files are included"""
    gain_object_full = copy.deepcopy(gain_data.gain_object)
    Nt2 = gain_data.gain_object.Ntimes // 2
    # Break up delay object into two objects, divided in time
    times1 = gain_data.gain_object.time_array[:Nt2]
    times2 = gain_data.gain_object.time_array[Nt2:]
    gain_data.gain_object.select(times=times1)
    gain_data.gain_object2.select(times=times2)
    # Write those objects to files
    f1 = os.path.join(DATA_PATH, 'test/read_multi1.calfits')
    f2 = os.path.join(DATA_PATH, 'test/read_multi2.calfits')
    gain_data.gain_object.write_calfits(f1, clobber=True)
    gain_data.gain_object2.write_calfits(f2, clobber=True)
    # Read both files together
    gain_data.gain_object.read_calfits([f1, f2])
    assert uvutils._check_histories(gain_object_full.history
                                    + '  Downselected to specific times'
                                    ' using pyuvdata. Combined data '
                                    'along time axis using pyuvdata.',
                                    gain_data.gain_object.history)
    gain_data.gain_object.history = gain_object_full.history
    assert gain_data.gain_object == gain_object_full


def test_add_antennas_delay(delay_data):
    """Test adding antennas between two UVCal objects"""
    delay_object_full = copy.deepcopy(delay_data.delay_object)
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    delay_data.delay_object.select(antenna_nums=ants1)
    delay_data.delay_object2.select(antenna_nums=ants2)
    delay_data.delay_object += delay_data.delay_object2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(delay_object_full.history
                                    + '  Downselected to specific '
                                    'antennas using pyuvdata. Combined '
                                    'data along antenna axis using pyuvdata.',
                                    delay_data.delay_object.history)
    delay_data.delay_object.history = delay_object_full.history
    assert delay_data.delay_object == delay_object_full

    # test for when total_quality_array is present
    delay_data.delay_object.select(antenna_nums=ants1)
    delay_data.delay_object.total_quality_array = np.zeros(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    uvtest.checkWarnings(delay_data.delay_object.__iadd__, [delay_data.delay_object2],
                         message='Total quality array detected')
    assert delay_data.delay_object.total_quality_array is None

    # test for when input_flag_array is present in first file but not second
    delay_data.delay_object.select(antenna_nums=ants1)
    ifa = np.zeros(
        delay_data.delay_object._input_flag_array.expected_shape(delay_data.delay_object)).astype(np.bool)
    ifa2 = np.ones(
        delay_data.delay_object2._input_flag_array.expected_shape(delay_data.delay_object2)).astype(np.bool)
    tot_ifa = np.concatenate([ifa, ifa2], axis=0)
    delay_data.delay_object.input_flag_array = ifa
    delay_data.delay_object2.input_flag_array = None
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.input_flag_array, tot_ifa)

    # test for when input_flag_array is present in second file but not first
    delay_data.delay_object.select(antenna_nums=ants1)
    ifa = np.ones(
        delay_data.delay_object._input_flag_array.expected_shape(delay_data.delay_object)).astype(np.bool)
    ifa2 = np.zeros(
        delay_data.delay_object2._input_flag_array.expected_shape(delay_data.delay_object2)).astype(np.bool)
    tot_ifa = np.concatenate([ifa, ifa2], axis=0)
    delay_data.delay_object.input_flag_array = None
    delay_data.delay_object2.input_flag_array = ifa2
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.input_flag_array, tot_ifa)

    # Out of order - antennas
    delay_data.delay_object = copy.deepcopy(delay_object_full)
    delay_data.delay_object2 = copy.deepcopy(delay_data.delay_object)
    delay_data.delay_object.select(antenna_nums=ants2)
    delay_data.delay_object2.select(antenna_nums=ants1)
    delay_data.delay_object += delay_data.delay_object2
    delay_data.delay_object.history = delay_object_full.history
    assert delay_data.delay_object == delay_object_full


def test_add_times_delay(delay_data):
    """Test adding times between two UVCal objects"""
    delay_object_full = copy.deepcopy(delay_data.delay_object)
    Nt2 = delay_data.delay_object.Ntimes // 2
    times1 = delay_data.delay_object.time_array[:Nt2]
    times2 = delay_data.delay_object.time_array[Nt2:]
    delay_data.delay_object.select(times=times1)
    delay_data.delay_object2.select(times=times2)
    delay_data.delay_object += delay_data.delay_object2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(delay_object_full.history
                                    + '  Downselected to specific '
                                    'times using pyuvdata. Combined '
                                    'data along time axis using pyuvdata.',
                                    delay_data.delay_object.history)
    delay_data.delay_object.history = delay_object_full.history
    assert delay_data.delay_object == delay_object_full

    # test for when total_quality_array is present in first file but not second
    delay_data.delay_object.select(times=times1)
    tqa = np.ones(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    tqa2 = np.zeros(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    delay_data.delay_object.total_quality_array = tqa
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.total_quality_array, tot_tqa,
                       rtol=delay_data.delay_object._total_quality_array.tols[0],
                       atol=delay_data.delay_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in second file but not first
    delay_data.delay_object.select(times=times1)
    tqa = np.zeros(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    tqa2 = np.ones(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    delay_data.delay_object.total_quality_array = None
    delay_data.delay_object2.total_quality_array = tqa2
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.total_quality_array, tot_tqa,
                       rtol=delay_data.delay_object._total_quality_array.tols[0],
                       atol=delay_data.delay_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in both
    delay_data.delay_object.select(times=times1)
    tqa = np.ones(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    tqa2 = np.ones(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    delay_data.delay_object.total_quality_array = tqa
    delay_data.delay_object2.total_quality_array = tqa2
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.total_quality_array, tot_tqa,
                       rtol=delay_data.delay_object._total_quality_array.tols[0],
                       atol=delay_data.delay_object._total_quality_array.tols[1])

    # test for when input_flag_array is present in first file but not second
    delay_data.delay_object.select(times=times1)
    ifa = np.zeros(
        delay_data.delay_object._input_flag_array.expected_shape(delay_data.delay_object)).astype(np.bool)
    ifa2 = np.ones(
        delay_data.delay_object2._input_flag_array.expected_shape(delay_data.delay_object2)).astype(np.bool)
    tot_ifa = np.concatenate([ifa, ifa2], axis=3)
    delay_data.delay_object.input_flag_array = ifa
    delay_data.delay_object2.input_flag_array = None
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.input_flag_array, tot_ifa)

    # test for when input_flag_array is present in second file but not first
    delay_data.delay_object.select(times=times1)
    ifa = np.ones(
        delay_data.delay_object._input_flag_array.expected_shape(delay_data.delay_object)).astype(np.bool)
    ifa2 = np.zeros(
        delay_data.delay_object2._input_flag_array.expected_shape(delay_data.delay_object2)).astype(np.bool)
    tot_ifa = np.concatenate([ifa, ifa2], axis=3)
    delay_data.delay_object.input_flag_array = None
    delay_data.delay_object2.input_flag_array = ifa2
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.input_flag_array, tot_ifa)

    # Out of order - times
    delay_data.delay_object = copy.deepcopy(delay_object_full)
    delay_data.delay_object2 = copy.deepcopy(delay_data.delay_object)
    delay_data.delay_object.select(times=times2)
    delay_data.delay_object2.select(times=times1)
    delay_data.delay_object += delay_data.delay_object2
    delay_data.delay_object.history = delay_object_full.history
    assert delay_data.delay_object == delay_object_full


def test_add_jones_delay(delay_data):
    """Test adding Jones axes between two UVCal objects"""
    delay_object_original = copy.deepcopy(delay_data.delay_object)
    # artificially change the Jones value to permit addition
    delay_data.delay_object2.jones_array[0] = -6
    delay_data.delay_object += delay_data.delay_object2

    # check dimensionality of resulting object
    assert delay_data.delay_object.delay_array.shape[-1] == 2
    assert sorted(delay_data.delay_object.jones_array) == [-6, -5]

    # test for when total_quality_array is present in first file but not second
    delay_data.delay_object = copy.deepcopy(delay_object_original)
    tqa = np.ones(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    tqa2 = np.zeros(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    delay_data.delay_object.total_quality_array = tqa
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.total_quality_array, tot_tqa,
                       rtol=delay_data.delay_object._total_quality_array.tols[0],
                       atol=delay_data.delay_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in second file but not first
    delay_data.delay_object = copy.deepcopy(delay_object_original)
    tqa = np.zeros(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    tqa2 = np.ones(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    delay_data.delay_object2.total_quality_array = tqa2
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.total_quality_array, tot_tqa,
                       rtol=delay_data.delay_object._total_quality_array.tols[0],
                       atol=delay_data.delay_object._total_quality_array.tols[1])

    # test for when total_quality_array is present in both
    delay_data.delay_object = copy.deepcopy(delay_object_original)
    tqa = np.ones(
        delay_data.delay_object._total_quality_array.expected_shape(delay_data.delay_object))
    tqa2 = np.ones(
        delay_data.delay_object2._total_quality_array.expected_shape(delay_data.delay_object2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    delay_data.delay_object.total_quality_array = tqa
    delay_data.delay_object2.total_quality_array = tqa2
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.total_quality_array, tot_tqa,
                       rtol=delay_data.delay_object._total_quality_array.tols[0],
                       atol=delay_data.delay_object._total_quality_array.tols[1])

    # test for when input_flag_array is present in first file but not second
    delay_data.delay_object = copy.deepcopy(delay_object_original)
    ifa = np.zeros(
        delay_data.delay_object._input_flag_array.expected_shape(delay_data.delay_object)).astype(np.bool)
    ifa2 = np.ones(
        delay_data.delay_object2._input_flag_array.expected_shape(delay_data.delay_object2)).astype(np.bool)
    tot_ifa = np.concatenate([ifa, ifa2], axis=4)
    delay_data.delay_object.input_flag_array = ifa
    delay_data.delay_object2.input_flag_array = None
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.input_flag_array, tot_ifa)

    # test for when input_flag_array is present in second file but not first
    delay_data.delay_object = copy.deepcopy(delay_object_original)
    ifa = np.ones(
        delay_data.delay_object._input_flag_array.expected_shape(delay_data.delay_object)).astype(np.bool)
    ifa2 = np.zeros(
        delay_data.delay_object2._input_flag_array.expected_shape(delay_data.delay_object2)).astype(np.bool)
    tot_ifa = np.concatenate([ifa, ifa2], axis=4)
    delay_data.delay_object.input_flag_array = None
    delay_data.delay_object2.input_flag_array = ifa2
    delay_data.delay_object += delay_data.delay_object2
    assert np.allclose(delay_data.delay_object.input_flag_array, tot_ifa)

    # Out of order - jones
    delay_data.delay_object = copy.deepcopy(delay_object_original)
    delay_data.delay_object2 = copy.deepcopy(delay_object_original)
    delay_data.delay_object.jones_array[0] = -6
    delay_data.delay_object += delay_data.delay_object2
    delay_data.delay_object2 = copy.deepcopy(delay_data.delay_object)
    delay_data.delay_object.select(jones=-5)
    delay_data.delay_object.history = delay_object_original.history
    assert delay_data.delay_object == delay_object_original
    delay_data.delay_object2.select(jones=-6)
    delay_data.delay_object2.jones_array[:] = -5
    delay_data.delay_object2.history = delay_object_original.history
    assert delay_data.delay_object2 == delay_object_original


def test_add_errors_delay(delay_data):
    """Test behavior that will raise errors"""
    # test addition of two identical objects
    pytest.raises(ValueError, delay_data.delay_object.__add__, delay_data.delay_object2)


def test_multi_files_delay(delay_data):
    """Test read function when multiple files are included"""
    delay_object_full = copy.deepcopy(delay_data.delay_object)
    Nt2 = delay_data.delay_object.Ntimes // 2
    # Break up delay object into two objects, divided in time
    times1 = delay_data.delay_object.time_array[:Nt2]
    times2 = delay_data.delay_object.time_array[Nt2:]
    delay_data.delay_object.select(times=times1)
    delay_data.delay_object2.select(times=times2)
    # Write those objects to files
    f1 = os.path.join(DATA_PATH, 'test/read_multi1.calfits')
    f2 = os.path.join(DATA_PATH, 'test/read_multi2.calfits')
    delay_data.delay_object.write_calfits(f1, clobber=True)
    delay_data.delay_object2.write_calfits(f2, clobber=True)
    # Read both files together
    delay_data.delay_object.read_calfits([f1, f2])
    assert uvutils._check_histories(delay_object_full.history
                                    + '  Downselected to specific times'
                                    ' using pyuvdata. Combined data '
                                    'along time axis using pyuvdata.',
                                    delay_data.delay_object.history)
    delay_data.delay_object.history = delay_object_full.history
    assert delay_data.delay_object == delay_object_full


def test_deprecated_x_orientation():
    cal_in = UVCal()
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
    cal_in.read_calfits(testfile)

    cal_in.x_orientation = 'e'

    uvtest.checkWarnings(cal_in.check, category=DeprecationWarning,
                         message=['x_orientation e is not one of [east, north], '
                                  'converting to "east".'])

    cal_in.x_orientation = 'N'
    uvtest.checkWarnings(cal_in.check, category=DeprecationWarning,
                         message=['x_orientation N is not one of [east, north], '
                                  'converting to "north".'])

    cal_in.x_orientation = 'foo'
    pytest.raises(ValueError, uvtest.checkWarnings, cal_in.check,
                  category=DeprecationWarning,
                  message=['x_orientation n is not one of [east, north], '
                           'cannot be converted.'])
