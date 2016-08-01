import unittest
import numpy as np
import nose.tools as nt
from uvdata.telescopes import Telescope


required_parameters = ['_telescope_name', '_telescope_location']
required_properties = ['telescope_name', 'telescope_location']
other_attributes = ['citation', 'telescope_location_lat_lon_alt',
                    'telescope_location_lat_lon_alt_degrees']
known_telescopes = ['PAPER', 'HERA', 'MWA']


def test_parameter_iter():
    telescope_obj = Telescope()
    all = []
    for prop in telescope_obj.parameter_iter():
        all.append(prop)
    for a in required_parameters:
        nt.assert_true(a in all, msg='expected attribute ' + a +
                       ' not returned in parameter_iter')


def test_required_parameter_iter():
    telescope_obj = Telescope()
    required = []
    for prop in telescope_obj.required_parameter_iter():
        required.append(prop)
    for a in required_parameters:
        nt.assert_true(a in required, msg='expected attribute ' + a +
                       ' not returned in required_parameter_iter')


def test_parameters_exist():
    telescope_obj = Telescope()
    expected_parameters = required_parameters
    for a in expected_parameters:
        nt.assert_true(hasattr(telescope_obj, a),
                       msg='expected parameter ' + a + ' does not exist')


def test_unexpected_attributes():
    telescope_obj = Telescope()
    expected_attributes = required_properties + other_attributes
    attributes = [i for i in telescope_obj.__dict__.keys() if i[0] != '_']
    for a in attributes:
        nt.assert_true(a in expected_attributes,
                       msg='unexpected attribute ' + a + ' found in Telescope')


def test_properties():
    telescope_obj = Telescope()
    prop_dict = dict(zip(required_properties, required_parameters))
    for k, v in prop_dict.iteritems():
        rand_num = np.random.rand()
        setattr(telescope_obj, k, rand_num)
        this_param = getattr(telescope_obj, v)
        try:
            nt.assert_equal(rand_num, this_param.value)
        except:
            print('setting {prop_name} to a random number failed'.format(prop_name=k))
            raise(AssertionError)


def test_known_telescopes():
    nt.assert_equal(uvdata.telescopes.known_telescopes.sort(),
                    known_telescopes.sort())
