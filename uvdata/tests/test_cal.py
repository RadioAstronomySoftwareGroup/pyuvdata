"""Tests for uvcal object."""
import nose.tools as nt
import os
import numpy as np
import copy
import ephem
from uvdata.cal import UVCal
import uvdata.tests as uvtest
from uvdata.data import DATA_PATH


class TestUVCalInit(object):
    def setUp(self):
        """Setup for basic parameter, property and iterator tests."""
        self.required_parameters = ['_Nfreqs', '_Npols', '_Ntimes', '_history',
                                    '_Nants_data', '_antenna_names', '_antenna_numbers',
                                    '_Nants_telescope', '_freq_array',
                                    '_polarization_array', '_time_array',
                                    '_gain_convention', '_flag_array',
                                    '_quality_array', '_cal_type',
                                    '_x_orientation']

        self.required_properties = ['Nfreqs', 'Npols', 'Ntimes', 'history',
                                    'Nants_data', 'antenna_names', 'antenna_numbers',
                                    'Nants_telescope', 'freq_array',
                                    'polarization_array', 'time_array',
                                    'gain_convention', 'flag_array',
                                    'quality_array', 'cal_type',
                                    'x_orientation']

        self.extra_parameters = ['_gain_array', '_delay_array',
                                 '_input_flag_array']

        self.extra_properties = ['gain_array', 'delay_array',
                                 'input_flag_array']

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
            self.extra_properties
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
        self.testfile = os.path.join(DATA_PATH, 'test123.fits')
        self.uv_cal_object.read_calfits(self.testfile)
        uvtest.checkWarnings(self.uv_cal_object.read_calfits, [self.testfile],
                             message='Telescope EVLA is not')
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

    def test_set_delay(self):
        self.uv_cal_object.set_delay()
        nt.assert_true(self.uv_cal_object._delay_array.required)
        nt.assert_false(self.uv_cal_object._gain_array.required)
        nt.assert_equal(self.uv_cal_object._delay_array.form, self.uv_cal_object._flag_array.form)
        nt.assert_equal(self.uv_cal_object._delay_array.form, self.uv_cal_object._quality_array.form)
        nt.assert_equal(self.uv_cal_object._quality_array.form, self.uv_cal_object._flag_array.form)
