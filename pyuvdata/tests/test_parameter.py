# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import nose.tools as nt
import numpy as np

from pyuvdata import parameter as uvp
from pyuvdata.uvbase import UVBase


def test_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name='p1', value=1)
    param2 = uvp.AngleParameter(name='p2', value=1)
    nt.assert_not_equal(param1, param2)


def test_value_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name='p1', value=3)
    param2 = uvp.UVParameter(name='p2', value=np.array([3, 4, 5]))
    nt.assert_not_equal(param1, param2)
    nt.assert_not_equal(param2, param1)
    param3 = uvp.UVParameter(name='p2', value='Alice')
    nt.assert_not_equal(param1, param3)


def test_array_inequality():
    """Test equality error for different array values."""
    param1 = uvp.UVParameter(name='p1', value=np.array([0, 1, 3]))
    param2 = uvp.UVParameter(name='p2', value=np.array([0, 2, 4]))
    nt.assert_not_equal(param1, param2)
    param3 = uvp.UVParameter(name='p3', value=np.array([0, 1]))
    nt.assert_not_equal(param1, param3)


def test_string_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name='p1', value='Alice')
    param2 = uvp.UVParameter(name='p2', value='Bob')
    nt.assert_not_equal(param1, param2)


def test_string_list_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name='p1', value=['Alice', 'Eve'])
    param2 = uvp.UVParameter(name='p2', value=['Bob', 'Eve'])
    nt.assert_not_equal(param1, param2)


def test_string_equality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name='p1', value='Alice')
    param2 = uvp.UVParameter(name='p2', value='Alice')
    nt.assert_equal(param1, param2)


def test_integer_inequality():
    """Test equality error for different non-array, non-string values."""
    param1 = uvp.UVParameter(name='p1', value=1)
    param2 = uvp.UVParameter(name='p2', value=2)
    nt.assert_not_equal(param1, param2)


def test_dict_equality():
    """Test equality for dict values."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1})
    param2 = uvp.UVParameter(name='p2', value={'v1': 1})
    nt.assert_equal(param1, param2)


def test_dict_inequality_int():
    """Test equality error for integer dict values."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1, 's1': 'test'})
    param2 = uvp.UVParameter(name='p2', value={'v1': 2, 's1': 'test'})
    nt.assert_not_equal(param1, param2)


def test_dict_inequality_str():
    """Test equality error for string dict values."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1, 's1': 'test'})
    param4 = uvp.UVParameter(name='p3', value={'v1': 1, 's1': 'foo'})
    nt.assert_not_equal(param1, param4)


def test_dict_inequality_keys():
    """Test equality error for different keys."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1, 's1': 'test'})
    param3 = uvp.UVParameter(name='p3', value={'v3': 1, 's1': 'test'})
    nt.assert_not_equal(param1, param3)


def test_equality_check_fail():
    """Test equality error for non string, dict or array values."""
    param1 = uvp.UVParameter(name='p1', value=uvp.UVParameter(name='p1', value='Alice'))
    param2 = uvp.UVParameter(name='p2', value=uvp.UVParameter(name='p1', value='Bob'))
    nt.assert_not_equal(param1, param2)


def test_notclose():
    """Test equality error for values not with tols."""
    param1 = uvp.UVParameter(name='p1', value=1.0)
    param2 = uvp.UVParameter(name='p2', value=1.001)
    nt.assert_not_equal(param1, param2)


def test_close():
    """Test equality error for values within tols."""
    param1 = uvp.UVParameter(name='p1', value=1.0)
    param2 = uvp.UVParameter(name='p2', value=1.000001)
    nt.assert_equal(param1, param2)


def test_acceptability():
    """Test check_acceptability function."""
    param1 = uvp.UVParameter(name='p1', value=1000, acceptable_range=(1, 10))
    nt.assert_false(param1.check_acceptability()[0])

    param1 = uvp.UVParameter(name='p1', value=np.random.rand(100), acceptable_range=(.1, .9))
    nt.assert_true(param1.check_acceptability()[0])
    param1 = uvp.UVParameter(name='p1', value=np.random.rand(100) * 1e-4, acceptable_range=(.1, .9))
    nt.assert_false(param1.check_acceptability()[0])

    param2 = uvp.UVParameter(name='p2', value=5, acceptable_range=(1, 10))
    nt.assert_true(param2.check_acceptability()[0])
    param2 = uvp.UVParameter(name='p2', value=5, acceptable_vals=[1, 10])
    nt.assert_false(param2.check_acceptability()[0])


def test_string_acceptability():
    """Test check_acceptability function with strings."""
    param1 = uvp.UVParameter(name='p1', value='Bob', form='str',
                             acceptable_vals=['Alice', 'Eve'])
    nt.assert_false(param1.check_acceptability()[0])
    param2 = uvp.UVParameter(name='p2', value='Eve', form='str',
                             acceptable_vals=['Alice', 'Eve'])
    nt.assert_true(param2.check_acceptability()[0])


def test_expected_shape():
    """Test missing shape param."""
    class TestUV(UVBase):
        def __init__(self):
            self._p1 = uvp.UVParameter(name='p1', required=False)
            self._p2 = uvp.UVParameter(name='p2', form=('p1',))
            self._p3 = uvp.UVParameter(name='p3', form=(2,))
            super(TestUV, self).__init__()
    obj = TestUV()
    obj.p2 = np.array([0, 5, 8])
    obj.p3 = np.array([4, 9])
    nt.assert_raises(ValueError, obj.check)
    nt.assert_equal(obj._p3.expected_shape(obj), (2,))


def test_angle_set_degree_none():
    param1 = uvp.AngleParameter(name='p2', value=1)
    param1.set_degrees(None)

    nt.assert_equal(None, param1.value)


def test_location_set_lat_lon_alt_none():
    param1 = uvp.LocationParameter(name='p2', value=1)
    param1.set_lat_lon_alt(None)

    nt.assert_equal(None, param1.value)


def test_location_set_lat_lon_alt_degrees_none():
    param1 = uvp.LocationParameter(name='p2', value=1)
    param1.set_lat_lon_alt_degrees(None)

    nt.assert_equal(None, param1.value)


def test_location_acceptable_none():
    param1 = uvp.LocationParameter(name='p2', value=1, acceptable_range=None)

    nt.assert_true(param1.check_acceptability())
