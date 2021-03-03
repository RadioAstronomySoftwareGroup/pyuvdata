# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvbase object.

"""
import pytest
import numpy as np
from astropy import units
from astropy.time import Time
from astropy.coordinates import Distance

from pyuvdata.uvbase import UVBase
from pyuvdata.uvbase import _warning
from pyuvdata import parameter as uvp
from pyuvdata import tests as uvtest

ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)


class UVTest(UVBase):
    def __init__(self):
        """ UVBase test object. """
        # add some test UVParameters to the class

        self._int1 = uvp.UVParameter(
            "int1", description="integer value", expected_type=int, value=3
        )

        self._int2 = uvp.UVParameter(
            "int2", description="integer value", expected_type=int, value=5
        )

        self._float1 = uvp.UVParameter(
            "float1", description="float value", expected_type=float, value=18.2
        )

        self._string1 = uvp.UVParameter(
            "string1", description="string value", form="str", value="test"
        )

        self._string2 = uvp.UVParameter(
            "string2", description="string value", expected_type="str", value="test"
        )

        self._floatarr = uvp.UVParameter(
            "floatarr",
            description="float array",
            form=("int1", "int2"),
            expected_type=float,
            value=np.random.rand(self._int1.value, self._int2.value),
        )

        self._floatarr2 = uvp.UVParameter(
            "floatarr2",
            description="float array",
            form=4,
            expected_type=float,
            value=np.random.rand(4),
        )

        self._strlist = uvp.UVParameter(
            "strlist",
            description="string list",
            form=("int1",),
            expected_type=str,
            value=["s" + str(i) for i in np.arange(self._int1.value)],
        )

        self._intlist = uvp.UVParameter(
            "intlist",
            description="integer list",
            form=("int1",),
            expected_type=int,
            value=list(range(self._int1.value)),
        )

        self._angle = uvp.AngleParameter(
            "angle", description="angle", expected_type=float, value=np.pi / 4.0
        )

        self._location = uvp.LocationParameter(
            "location", description="location", value=np.array(ref_xyz)
        )

        self._time = uvp.UVParameter(
            "time",
            description="astropy Time object",
            value=Time("2015-03-01 00:00:00", scale="utc"),
            expected_type=Time,
        )

        self._optional_int1 = uvp.UVParameter(
            "optional_int1",
            description="optional integer value",
            expected_type=int,
            value=3,
            required=False,
        )

        self._optional_int2 = uvp.UVParameter(
            "optional_int2",
            description="optional integer value",
            expected_type=int,
            value=5,
            required=False,
        )

        self._unset_int1 = uvp.UVParameter(
            "unset_int1",
            description="An unset parameter.",
            expected_type=int,
            value=None,
            required=False,
        )

        self._quantity_array = uvp.UVParameter(
            "quantity_array",
            description="A quantity object.",
            expected_type=units.Quantity,
            value=self._floatarr2.value * units.m,
            form=self._floatarr2.value.size,
        )

        self._quantity_scalar = uvp.UVParameter(
            "quantity_scalar",
            description="A quantity but also a single element.",
            expected_type=units.Quantity,
            value=2 * units.m,
            form=(),
        )

        super(UVTest, self).__init__()


def test_equality():
    """Basic equality test."""
    test_obj = UVTest()
    assert test_obj == test_obj


def test_equality_nocheckextra():
    """Test equality if optional params are different and check_extra is false."""
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.optional_int1 = 4
    assert test_obj.__eq__(test_obj2, check_extra=False)


def test_inequality_extra():
    """Basic equality test."""
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.optional_int1 = 4
    assert test_obj != test_obj2


def test_inequality_different_extras():
    """Basic equality test."""
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2._optional_int3 = uvp.UVParameter(
        "optional_int3",
        description="optional integer value",
        expected_type=int,
        value=7,
        required=False,
    )
    assert test_obj != test_obj2


def test_inequality():
    """Check that inequality is handled correctly."""
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.float1 = 13
    assert test_obj != test_obj2


def test_class_inequality():
    """Test equality error for different classes."""
    test_obj = UVTest()
    assert test_obj != test_obj._floatarr


def test_check():
    """Test simple check function."""
    test_obj = UVTest()
    assert test_obj.check()


def test_check_required():
    """Test simple check function."""
    test_obj = UVTest()
    assert test_obj.check(check_extra=False)


def test_string_type_check():
    """Test check function with wrong type (string)."""
    test_obj = UVTest()
    test_obj.string1 = 1
    pytest.raises(ValueError, test_obj.check)


def test_string_form_check():
    """Test check function with wrong type (string)."""
    test_obj = UVTest()
    test_obj.string2 = 1
    pytest.raises(ValueError, test_obj.check)


def test_single_value_check():
    """Test check function with wrong type."""
    test_obj = UVTest()
    test_obj.int1 = np.float64(test_obj.int1)
    pytest.raises(ValueError, test_obj.check)


def test_check_array_type():
    """Test check function with wrong array type."""
    test_obj = UVTest()
    test_obj.floatarr = test_obj.floatarr + 1j * test_obj.floatarr
    pytest.raises(ValueError, test_obj.check)


def test_check_quantity_type():
    """Test check function with wrong array type."""
    test_obj = UVTest()
    test_obj.floatarr = (test_obj.floatarr + 1j * test_obj.floatarr) * units.m
    with uvtest.check_warnings(
        UserWarning,
        "Parameter _floatarr is a Quantity object, but the expected type "
        "is a precision identifier: (<class 'float'>, <class 'numpy.floating'>). "
        "Testing the precision of the value, "
        "but this check will fail in a future version.",
    ):
        with pytest.raises(
            ValueError, match="UVParameter _floatarr is not the appropriate type. "
        ):
            test_obj.check()


def test_wrong_quantity_type():
    """Test check when given the wrong kind of Quantity."""
    test_obj = UVTest()
    test_obj._quantity_array.expected_type = Distance
    with pytest.raises(
        ValueError,
        match="UVParameter _quantity_array is a Quantity "
        "object but not the appropriate type.",
    ):
        test_obj.check()


def test_quantity_scalar_type():
    """Test check when a scalar quantity has odd expected_type."""
    test_obj = UVTest()
    test_obj._quantity_scalar.expected_type = (float, np.floating)
    with uvtest.check_warnings(
        UserWarning,
        "Parameter _quantity_scalar is a Quantity object, but the expected type "
        "is a precision identifier: (<class 'float'>, <class 'numpy.floating'>). "
        "Testing the precision of the value, but this "
        "check will fail in a future version.",
    ):
        test_obj.check()


def test_check_array_shape():
    """Test check function with wrong array dimensions."""
    test_obj = UVTest()
    test_obj.floatarr = np.array([4, 5, 6], dtype=np.float64)
    pytest.raises(ValueError, test_obj.check)


def test_list_dims():
    """Test check function with wrong list dimensions."""
    test_obj = UVTest()
    test_obj.strlist = ["s" + str(i) for i in np.arange(test_obj.int2)]
    pytest.raises(ValueError, test_obj.check)


def test_list_type():
    """Test check function with wrong list type."""
    test_obj = UVTest()
    test_obj.intlist = ["s" + str(i) for i in np.arange(test_obj.int1)]
    pytest.raises(ValueError, test_obj.check)

    test_obj.intlist = list(np.arange(test_obj.int1))
    test_obj.intlist[1] = "test"
    pytest.raises(ValueError, test_obj.check)


def test_angle():
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.angle_degrees = 45.0
    assert test_obj == test_obj2


def test_angle_none():
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj.angle = None
    test_obj2.angle_degrees = None
    assert test_obj == test_obj2


def test_location():
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.location_lat_lon_alt = ref_latlonalt
    assert test_obj == test_obj2


def test_location_degree():
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.location_lat_lon_alt_degrees = (
        np.rad2deg(ref_latlonalt[0]),
        np.rad2deg(ref_latlonalt[1]),
        ref_latlonalt[2],
    )
    assert test_obj == test_obj2


def test_location_none():
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj.location = None
    test_obj2.location_lat_lon_alt = None
    assert test_obj == test_obj2


def test_location_degree_none():
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj.location = None
    test_obj2.location_lat_lon_alt_degrees = None
    assert test_obj == test_obj2


def test_warning():
    output = _warning("hello world")
    assert output, "hello world\n"


def test_check_ignore_unset():
    test_obj = UVTest()
    test_obj._unset_int1.required = True
    pytest.raises(ValueError, test_obj.check)
    assert test_obj.check(ignore_requirements=True)
