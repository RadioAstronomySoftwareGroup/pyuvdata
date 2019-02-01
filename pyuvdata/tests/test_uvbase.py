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
from pyuvdata.uvbase import _warning
from pyuvdata import parameter as uvp

ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)


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

        self._string1 = uvp.UVParameter('string1', description='string value',
                                        form='str', value='test')

        self._string2 = uvp.UVParameter('string2', description='string value',
                                        expected_type='str', value='test')

        self._floatarr = uvp.UVParameter('floatarr', description='float array',
                                         form=('int1', 'int2'),
                                         expected_type=np.float,
                                         value=np.random.rand(self._int1.value,
                                                              self._int2.value))

        self._floatarr2 = uvp.UVParameter('floatarr2', description='float array',
                                          form=4, expected_type=np.float,
                                          value=np.random.rand(4))

        self._strlist = uvp.UVParameter('strlist', description='string list',
                                        form=('int1',), expected_type=str,
                                        value=['s' + str(i) for i in np.arange(self._int1.value)])

        self._intlist = uvp.UVParameter('intlist', description='integer list',
                                        form=('int1',), expected_type=int,
                                        value=list(range(self._int1.value)))

        self._angle = uvp.AngleParameter('angle', description='angle',
                                         expected_type=np.float, value=np.pi / 4.)

        self._location = uvp.LocationParameter('location', description='location',
                                               value=np.array(ref_xyz))

        self._optional_int1 = uvp.UVParameter('optional_int1', description='optional integer value',
                                              expected_type=int, value=3, required=False)

        self._optional_int2 = uvp.UVParameter('optional_int2', description='optional integer value',
                                              expected_type=int, value=5, required=False)

        super(UVTest, self).__init__()


def test_equality():
    """Basic equality test."""
    test_obj = UVTest()
    nt.assert_equal(test_obj, test_obj)


def test_equality_nocheckextra():
    """Test that objects are equal if optional params are different and check_extra is false"""
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.optional_int1 = 4
    nt.assert_true(test_obj.__eq__(test_obj2, check_extra=False))


def test_inequality_extra():
    """Basic equality test."""
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.optional_int1 = 4
    nt.assert_not_equal(test_obj, test_obj2)


def test_inequality_different_extras():
    """Basic equality test."""
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2._optional_int3 = uvp.UVParameter('optional_int3', description='optional integer value',
                                               expected_type=int, value=7, required=False)
    nt.assert_not_equal(test_obj, test_obj2)


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


def test_check_required():
    """Test simple check function."""
    test_obj = UVTest()
    nt.assert_true(test_obj.check(check_extra=False))


def test_string_type_check():
    """Test check function with wrong type (string)."""
    test_obj = UVTest()
    test_obj.string1 = 1
    nt.assert_raises(ValueError, test_obj.check)


def test_string_form_check():
    """Test check function with wrong type (string)."""
    test_obj = UVTest()
    test_obj.string2 = 1
    nt.assert_raises(ValueError, test_obj.check)


def test_single_value_check():
    """Test check function with wrong type."""
    test_obj = UVTest()
    test_obj.int1 = np.float(test_obj.int1)
    nt.assert_raises(ValueError, test_obj.check)


def test_check_array_type():
    """Test check function with wrong array type."""
    test_obj = UVTest()
    test_obj.floatarr = test_obj.floatarr + 1j * test_obj.floatarr
    nt.assert_raises(ValueError, test_obj.check)


def test_check_array_shape():
    """Test check function with wrong array dimensions."""
    test_obj = UVTest()
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


def test_angle():
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.angle_degrees = 45.
    nt.assert_equal(test_obj, test_obj2)


def test_angle_none():
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj.angle = None
    test_obj2.angle_degrees = None
    nt.assert_equal(test_obj, test_obj2)


def test_location():
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.location_lat_lon_alt = ref_latlonalt
    nt.assert_equal(test_obj, test_obj2)


def test_location_degree():
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.location_lat_lon_alt_degrees = (np.rad2deg(ref_latlonalt[0]), np.rad2deg(ref_latlonalt[1]), ref_latlonalt[2])
    nt.assert_equal(test_obj, test_obj2)


def test_location_none():
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj.location = None
    test_obj2.location_lat_lon_alt = None
    nt.assert_equal(test_obj, test_obj2)


def test_location_degree_none():
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj.location = None
    test_obj2.location_lat_lon_alt_degrees = None
    nt.assert_equal(test_obj, test_obj2)


def test_warning():
    output = _warning("hello world")
    nt.assert_equal(output, "hello world\n")
