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
                                    '_quality_array', '_delay_gain_switch',
                                    '_x_orientation']

        self.required_properties = ['Nfreqs', 'Npols', 'Ntimes', 'history',
                                    'Nants_data', 'antenna_names', 'antenna_numbers',
                                    'Nants_telescope', 'freq_array',
                                    'polarization_array', 'time_array',
                                    'gain_convention', 'flag_array',
                                    'quality_array', 'delay_gain_switch',
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
