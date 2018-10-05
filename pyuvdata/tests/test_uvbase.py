# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvbase object.

"""
from __future__ import absolute_import, division, print_function

import nose.tools as nt
import numpy as np
import copy

from pyuvdata.uvbase import UVBase
from pyuvdata import parameter as uvp


class UVTest(UVBase):
    def __init__(self):
        """ UVBase test object. """
        # add some test UVParameters to the class

        self._int1 = uvp.UVParameter('int1', description='integer value',
                                     expected_type=int, value=3)

        self._int2 = uvp.UVParameter('int2', description='integer value',
                                     expected_type=int, value=5)

        self._float1 = uvp.UVParameter('float1', description='float value',
                                       expected_type=np.float, value=18.2)

        self._string = uvp.UVParameter('string', description='string value',
                                       expected_type=str, value='test')

        self._floatarr = uvp.UVParameter('floatarr', description='float array',
                                         form=('int1', 'int2'),
                                         expected_type=np.float,
                                         value=np.random.rand(self._int1.value,
                                                              self._int2.value))

        self._strlist = uvp.UVParameter('strlist', description='string list',
                                        form=('int1',), expected_type=str,
                                        value=['s' + str(i) for i in np.arange(self._int1.value)])

        self._intlist = uvp.UVParameter('intlist', description='integer list',
                                        form=('int1',), expected_type=int,
                                        value=list(range(self._int1.value)))
        super(UVTest, self).__init__()


def test_equality():
    """Basic equality test."""
    test_obj = UVTest()
    nt.assert_equal(test_obj, test_obj)


def test_inequality():
    """Check that inequality is handled correctly."""
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.float1 = 13
    nt.assert_not_equal(test_obj, test_obj2)


def test_class_inequality():
    """Test equality error for different classes."""
    test_obj = UVTest()
    nt.assert_not_equal(test_obj, test_obj._floatarr)


def test_check():
    """Test simple check function."""
    test_obj = UVTest()
    nt.assert_true(test_obj.check())


def test_string_check():
    """Test check function with wrong type (string)."""
    test_obj = UVTest()
    test_obj.string = 1
    nt.assert_raises(ValueError, test_obj.check)


def test_single_value_check():
    """Test check function with wrong dimensions or type."""
    test_obj = UVTest()
    test_obj.int1 += 4
    nt.assert_raises(ValueError, test_obj.check)
    test_obj.int1 = np.float(4)
    nt.assert_raises(ValueError, test_obj.check)


def test_array():
    """Test check function with wrong array dimensions or type."""
    test_obj = UVTest()

    test_obj.floatarr = test_obj.floatarr + 1j * test_obj.floatarr
    nt.assert_raises(ValueError, test_obj.check)

    test_obj.floatarr = np.array([4, 5, 6], dtype=np.float)
    nt.assert_raises(ValueError, test_obj.check)


def test_list_dims():
    """Test check function with wrong list dimensions."""
    test_obj = UVTest()
    test_obj.strlist = ['s' + str(i) for i in np.arange(test_obj.int2)]
    nt.assert_raises(ValueError, test_obj.check)


def test_list_dims():
    """Test check function with wrong list type."""
    test_obj = UVTest()
    test_obj.intlist = ['s' + str(i) for i in np.arange(test_obj.int1)]
    nt.assert_raises(ValueError, test_obj.check)

    test_obj.intlist = [i for i in np.arange(test_obj.int1)]
    test_obj.intlist[1] = 'test'
    nt.assert_raises(ValueError, test_obj.check)
