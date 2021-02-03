# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import pytest
import numpy as np
import astropy.units as units

from pyuvdata import parameter as uvp
from pyuvdata.uvbase import UVBase


def test_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name="p1", value=1)
    param2 = uvp.AngleParameter(name="p2", value=1)
    assert param1 != param2


def test_value_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name="p1", value=3)
    param2 = uvp.UVParameter(name="p2", value=np.array([3, 4, 5]))
    assert param1 != param2
    assert param2 != param1
    param3 = uvp.UVParameter(name="p2", value="Alice")
    assert param1 != param3


def test_array_inequality():
    """Test equality error for different array values."""
    param1 = uvp.UVParameter(name="p1", value=np.array([0, 1, 3]))
    param2 = uvp.UVParameter(name="p2", value=np.array([0, 2, 4]))
    assert param1 != param2
    param3 = uvp.UVParameter(name="p3", value=np.array([0, 1]))
    assert param1 != param3


def test_array_equality_nans():
    """Test array equality with nans present."""
    param1 = uvp.UVParameter(name="p1", value=np.array([0, 1, np.nan]))
    param2 = uvp.UVParameter(name="p2", value=np.array([0, 1, np.nan]))
    assert param1 == param2


@pytest.mark.parametrize("atol", [0.001, 1 * units.mm])
@pytest.mark.parametrize(
    "vals",
    (
        units.Quantity([0 * units.cm, 100 * units.cm, 3000 * units.mm]),
        units.Quantity([0.09 * units.cm, 100.09 * units.cm, 2999.1 * units.mm]),
        np.array([0, 1000, 3000]) * units.mm,
    ),
)
def test_quantity_equality(atol, vals):
    """Test equality for different quantity values."""
    param1 = uvp.UVParameter(name="p1", value=np.array([0, 1, 3]) * units.m, tols=atol)
    param2 = uvp.UVParameter(name="p2", value=vals, tols=atol)
    assert param1 == param2


def test_quantity_equality_error():
    """Test equality for different quantity values."""
    param1 = uvp.UVParameter(
        name="p1", value=np.array([0, 1, 3]) * units.m, tols=1 * units.mJy
    )
    param2 = uvp.UVParameter(
        name="p2",
        value=units.Quantity([0 * units.cm, 100 * units.cm, 3000 * units.mm]),
        tols=1 * units.mm,
    )
    with pytest.raises(units.UnitsError):
        assert param1 == param2


@pytest.mark.parametrize(
    "vals,p2_atol",
    (
        (np.array([0, 2, 4]) * units.m, 1 * units.mm),
        (np.array([0, 1, 3]) * units.mm, 1 * units.mm),
        (np.array([0, 1, 3]) * units.Jy, 1 * units.mJy),
        (
            units.Quantity([0.101 * units.cm, 100.09 * units.cm, 2999.1 * units.mm]),
            1 * units.mm,
        ),
        (
            units.Quantity([0.09 * units.cm, 100.11 * units.cm, 2999.1 * units.mm]),
            1 * units.mm,
        ),
        (np.array([0, 1000, 2998.9]) * units.mm, 1 * units.mm),
    ),
)
def test_quantity_inequality(vals, p2_atol):
    param1 = uvp.UVParameter(
        name="p1", value=np.array([0, 1, 3]) * units.m, tols=1 * units.mm
    )
    param2 = uvp.UVParameter(name="p2", value=vals, tols=p2_atol)
    assert param1 != param2


def test_quantity_equality_nans():
    """Test array equality with nans present."""
    param1 = uvp.UVParameter(name="p1", value=np.array([0, 1, np.nan] * units.m))
    param2 = uvp.UVParameter(name="p2", value=np.array([0, 1, np.nan] * units.m))
    assert param1 == param2


def test_string_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name="p1", value="Alice")
    param2 = uvp.UVParameter(name="p2", value="Bob")
    assert param1 != param2


def test_string_list_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name="p1", value=["Alice", "Eve"])
    param2 = uvp.UVParameter(name="p2", value=["Bob", "Eve"])
    assert param1 != param2


def test_string_equality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name="p1", value="Alice")
    param2 = uvp.UVParameter(name="p2", value="Alice")
    assert param1 == param2


def test_integer_inequality():
    """Test equality error for different non-array, non-string values."""
    param1 = uvp.UVParameter(name="p1", value=1)
    param2 = uvp.UVParameter(name="p2", value=2)
    assert param1 != param2


def test_dict_equality():
    """Test equality for dict values."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1})
    param2 = uvp.UVParameter(name="p2", value={"v1": 1})
    assert param1 == param2


def test_dict_inequality_int():
    """Test equality error for integer dict values."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "s1": "test"})
    param2 = uvp.UVParameter(name="p2", value={"v1": 2, "s1": "test"})
    assert param1 != param2


def test_dict_inequality_str():
    """Test equality error for string dict values."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "s1": "test"})
    param4 = uvp.UVParameter(name="p3", value={"v1": 1, "s1": "foo"})
    assert param1 != param4


def test_dict_inequality_keys():
    """Test equality error for different keys."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "s1": "test"})
    param3 = uvp.UVParameter(name="p3", value={"v3": 1, "s1": "test"})
    assert param1 != param3


def test_equality_check_fail():
    """Test equality error for non string, dict or array values."""
    param1 = uvp.UVParameter(name="p1", value=uvp.UVParameter(name="p1", value="Alice"))
    param2 = uvp.UVParameter(name="p2", value=uvp.UVParameter(name="p1", value="Bob"))
    assert param1 != param2


def test_notclose():
    """Test equality error for values not with tols."""
    param1 = uvp.UVParameter(name="p1", value=1.0)
    param2 = uvp.UVParameter(name="p2", value=1.001)
    assert param1 != param2


def test_close():
    """Test equality error for values within tols."""
    param1 = uvp.UVParameter(name="p1", value=1.0)
    param2 = uvp.UVParameter(name="p2", value=1.000001)
    assert param1 == param2


def test_acceptability():
    """Test check_acceptability function."""
    param1 = uvp.UVParameter(name="p1", value=1000, acceptable_range=(1, 10))
    assert not param1.check_acceptability()[0]

    param1 = uvp.UVParameter(
        name="p1", value=np.random.rand(100), acceptable_range=(0.1, 0.9)
    )
    assert param1.check_acceptability()[0]
    param1 = uvp.UVParameter(
        name="p1", value=np.random.rand(100) * 1e-4, acceptable_range=(0.1, 0.9)
    )
    assert not param1.check_acceptability()[0]

    param2 = uvp.UVParameter(name="p2", value=5, acceptable_range=(1, 10))
    assert param2.check_acceptability()[0]
    param2 = uvp.UVParameter(name="p2", value=5, acceptable_vals=[1, 10])
    assert not param2.check_acceptability()[0]


def test_string_acceptability():
    """Test check_acceptability function with strings."""
    param1 = uvp.UVParameter(
        name="p1", value="Bob", form="str", acceptable_vals=["Alice", "Eve"]
    )
    assert not param1.check_acceptability()[0]
    param2 = uvp.UVParameter(
        name="p2", value="Eve", form="str", acceptable_vals=["Alice", "Eve"]
    )
    assert param2.check_acceptability()[0]


def test_expected_shape():
    """Test missing shape param."""

    class TestUV(UVBase):
        def __init__(self):
            self._p1 = uvp.UVParameter(name="p1", required=False)
            self._p2 = uvp.UVParameter(name="p2", form=("p1",))
            self._p3 = uvp.UVParameter(name="p3", form=(2,))
            super(TestUV, self).__init__()

    obj = TestUV()
    obj.p2 = np.array([0, 5, 8])
    obj.p3 = np.array([4, 9])
    pytest.raises(ValueError, obj.check)
    assert obj._p3.expected_shape(obj) == (2,)


def test_angle_set_degree_none():
    param1 = uvp.AngleParameter(name="p2", value=1)
    param1.set_degrees(None)

    assert param1.value is None


def test_location_set_lat_lon_alt_none():
    param1 = uvp.LocationParameter(name="p2", value=1)
    param1.set_lat_lon_alt(None)

    assert param1.value is None


def test_location_set_lat_lon_alt_degrees_none():
    param1 = uvp.LocationParameter(name="p2", value=1)
    param1.set_lat_lon_alt_degrees(None)

    assert param1.value is None


def test_location_acceptable_none():
    param1 = uvp.LocationParameter(name="p2", value=1, acceptable_range=None)

    assert param1.check_acceptability()


def test_non_builtin_expected_type():
    with pytest.raises(ValueError) as cm:
        uvp.UVParameter(
            "_test", expected_type="integer",
        )
    assert str(cm.value).startswith("Input expected_type is a string with value")


def test_strict_expected_type():
    param1 = uvp.UVParameter("_test", expected_type=np.float64, strict_type_check=True)
    assert param1.expected_type == np.float64


@pytest.mark.parametrize(
    "in_type,out_type",
    [
        (np.float64, (float, np.floating)),
        (int, (int, np.integer)),
        (np.complex64, (complex, np.complexfloating)),
        (np.uint, (np.unsignedinteger)),
        # str type tests the pass through fallback
        (str, str),
        # check builtin attributes too
        ("str", str),
        ("int", (int, np.integer)),
        ("float", (float, np.floating)),
        ("complex", (complex, np.complexfloating)),
    ],
)
def test_generic_type_conversion(in_type, out_type):
    param1 = uvp.UVParameter("_test", expected_type=in_type)
    assert param1.expected_type == out_type


def test_strict_expected_type_equality():
    # make sure equality passes if one is strict and one is generic
    param1 = uvp.UVParameter(
        "_test1",
        value=np.float64(3.0),
        expected_type=np.float64,
        strict_type_check=True,
    )
    param2 = uvp.UVParameter(
        "_test2", value=3.0, expected_type=float, strict_type_check=False
    )
    assert param1 == param2
    assert param2 == param1

    # make sure it fails when both are strict and different
    param3 = uvp.UVParameter(
        "_test3", value=3.0, expected_type=float, strict_type_check=True
    )
    assert param1 != param3
    assert param3 != param1
    assert param2 == param3

    # also try different precision values
    param4 = uvp.UVParameter(
        "_test4",
        value=np.float32(3.0),
        expected_type=np.float32,
        strict_type_check=True,
    )
    assert param1 != param4

    # make sure it passes when both are strict and equivalent
    param5 = uvp.UVParameter(
        "_test5",
        value=np.float64(3.0),
        expected_type=np.float64,
        strict_type_check=True,
    )
    assert param1 == param5

    return


def test_scalar_array_parameter_mismatch():
    param1 = uvp.UVParameter("_test1", value=3.0, expected_type=float)
    param2 = uvp.UVParameter("_test2", value=np.asarray([3.0]), expected_type=float)
    assert param1 != param2
    assert param2 != param1

    return


def test_value_none_parameter_mismatch():
    param1 = uvp.UVParameter("_test1", value=3.0, expected_type=float)
    param2 = uvp.UVParameter("_test2", value=None)
    assert param1 != param2
    assert param2 != param1

    return
