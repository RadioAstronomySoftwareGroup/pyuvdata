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


def test_dict_inequality():
    """Test equality error for dict values."""
    param1 = uvp.UVParameter(name='p1', value={'v1': 1})
    param2 = uvp.UVParameter(name='p2', value={'v1': 2})
    nt.assert_not_equal(param1, param2)
    param3 = uvp.UVParameter(name='p3', value={'v3': 1})
    nt.assert_not_equal(param1, param3)


def test_equality_check_fail():
    """Test equality error for dict values."""
    param1 = uvp.UVParameter(name='p1', value=np.array([0, 1, 2]))
    param2 = uvp.UVParameter(name='p2', value={'v1': 2})
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
