# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
import copy

import astropy.units as units
import numpy as np
import pytest
from astropy.coordinates import CartesianRepresentation, Latitude, Longitude, SkyCoord

from pyuvdata import parameter as uvp
from pyuvdata import utils
from pyuvdata.tests.test_utils import (
    frame_selenoid,
    ref_latlonalt,
    ref_latlonalt_moon,
    ref_xyz,
    ref_xyz_moon,
)
from pyuvdata.uvbase import UVBase


@pytest.fixture
def sky_in():
    yield SkyCoord(
        ra=Longitude(5.0, unit="hourangle"),
        dec=Latitude(-30, unit="deg"),
        frame="fk5",
        equinox="J2000",
    )


def test_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name="p1", value=1)
    param2 = uvp.AngleParameter(name="p2", value=1)
    # use `__ne__` rather than `!=` throughout so we can cover print lines
    assert param1.__ne__(param2, silent=False)


def test_value_class_inequality():
    """Test equality error for different uvparameter classes."""
    param1 = uvp.UVParameter(name="p1", value=3)
    param2 = uvp.UVParameter(name="p2", value=np.array([3, 4, 5]))
    assert param1.__ne__(param2, silent=False)
    assert param2.__ne__(param1, silent=False)
    param3 = uvp.UVParameter(name="p2", value="Alice")
    assert param1.__ne__(param3, silent=False)


def test_array_inequality():
    """Test equality error for different array values."""
    param1 = uvp.UVParameter(name="p1", value=np.array([0, 1, 3]))
    param2 = uvp.UVParameter(name="p2", value=np.array([0, 2, 4]))
    assert param1.__ne__(param2, silent=False)
    param3 = uvp.UVParameter(name="p3", value=np.array([0, 1]))
    assert param1.__ne__(param3, silent=False)


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
    assert param1.__ne__(param2, silent=False)


def test_quantity_equality_nans():
    """Test array equality with nans present."""
    param1 = uvp.UVParameter(name="p1", value=np.array([0, 1, np.nan] * units.m))
    param2 = uvp.UVParameter(name="p2", value=np.array([0, 1, np.nan] * units.m))
    assert param1 == param2


def test_string_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name="p1", value="Alice")
    param2 = uvp.UVParameter(name="p2", value="Bob")
    assert param1.__ne__(param2, silent=False)


def test_string_list_inequality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name="p1", value=["Alice", "Eve"])
    param2 = uvp.UVParameter(name="p2", value=["Bob", "Eve"])
    assert param1.__ne__(param2, silent=False)


def test_string_equality():
    """Test equality error for different string values."""
    param1 = uvp.UVParameter(name="p1", value="Alice")
    param2 = uvp.UVParameter(name="p2", value="Alice")
    assert param1 == param2


def test_integer_inequality():
    """Test equality error for different non-array, non-string values."""
    param1 = uvp.UVParameter(name="p1", value=1)
    param2 = uvp.UVParameter(name="p2", value=2)
    assert param1.__ne__(param2, silent=False)


def test_dict_equality():
    """Test equality for dict values."""
    param1 = uvp.UVParameter(
        name="p1", value={"v1": 1, "n1": None, "s1": "foo", "arr1": [3, 4, 5]}
    )
    param2 = uvp.UVParameter(
        name="p2", value={"v1": 1, "n1": None, "s1": "foo", "arr1": [3, 4, 5]}
    )
    assert param1 == param2


def test_dict_inequality_int():
    """Test equality error for integer dict values."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "s1": "test", "n1": None})
    param2 = uvp.UVParameter(name="p2", value={"v1": 2, "s1": "test", "n1": None})
    assert param1.__ne__(param2, silent=False)


def test_dict_inequality_str():
    """Test equality error for string dict values."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "s1": "test", "n1": None})
    param4 = uvp.UVParameter(name="p3", value={"v1": 1, "s1": "foo", "n1": None})
    assert param1.__ne__(param4, silent=False)


def test_dict_inequality_none():
    """Test equality error for string dict values."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "s1": "test", "n1": None})
    param4 = uvp.UVParameter(name="p3", value={"v1": 1, "s1": "test", "n1": 2})
    assert param1.__ne__(param4, silent=False)


def test_dict_inequality_arr():
    """Test equality error for string dict values."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "arr1": [3, 4, 5]})
    param4 = uvp.UVParameter(name="p3", value={"v1": 1, "arr1": [3, 4]})
    assert param1.__ne__(param4, silent=False)

    param4 = uvp.UVParameter(name="p3", value={"v1": 1, "arr1": [3, 4, 6]})
    assert param1.__ne__(param4, silent=False)


def test_dict_inequality_keys():
    """Test equality error for different keys."""
    param1 = uvp.UVParameter(name="p1", value={"v1": 1, "s1": "test", "n1": None})
    param3 = uvp.UVParameter(name="p3", value={"v3": 1, "s1": "test", "n1": None})
    assert param1.__ne__(param3, silent=False)


def test_nested_dict_equality():
    """Test equality for nested dicts."""
    param1 = uvp.UVParameter(
        name="p1", value={"d1": {"v1": 1, "s1": "test"}, "d2": {"v1": 1, "s1": "test"}}
    )
    param3 = uvp.UVParameter(
        name="p3", value={"d1": {"v1": 1, "s1": "test"}, "d2": {"v1": 1, "s1": "test"}}
    )
    assert param1 == param3


def test_nested_dict_inequality():
    """Test equality error for nested dicts."""
    param1 = uvp.UVParameter(
        name="p1", value={"d1": {"v1": 1, "s1": "test"}, "d2": {"v1": 1, "s1": "test"}}
    )
    param3 = uvp.UVParameter(
        name="p3", value={"d1": {"v1": 2, "s1": "test"}, "d2": {"v1": 1, "s1": "test"}}
    )
    assert param1.__ne__(param3, silent=False)


def test_recarray_equality():
    """Test equality for recarray."""
    names = ["foo", "bar", "gah"]
    values = [
        np.arange(35, dtype=float),
        np.arange(35, dtype=int),
        np.array(["gah " + str(ind) for ind in range(35)]),
    ]
    dtype = []
    for val in values:
        dtype.append(val.dtype)
    dtype_obj = np.dtype(list(zip(names, dtype)))
    recarr1 = np.rec.fromarrays(values, dtype=dtype_obj)
    recarr2 = copy.deepcopy(recarr1)
    param1 = uvp.UVParameter(name="p1", value=recarr1)
    param3 = uvp.UVParameter(name="p3", value=recarr2)
    assert param1 == param3


@pytest.mark.parametrize(
    ["names2", "values2", "msg"],
    [
        [
            ["foo", "bar", "gah"],
            [
                np.arange(35, dtype=float),
                np.arange(35, dtype=int) + 1,
                np.array(["gah " + str(ind) for ind in range(35)]),
            ],
            "p1 parameter value is a recarray, values in field bar are not close.",
        ],
        [
            ["foo", "bar", "gah"],
            [
                np.arange(35, dtype=float),
                np.arange(35, dtype=int),
                np.array(["bah " + str(ind) for ind in range(35)]),
            ],
            "p1 parameter value is a recarray, values in field gah are not close.",
        ],
        [
            ["fob", "bar", "gah"],
            [
                np.arange(35, dtype=float),
                np.arange(35, dtype=int),
                np.array(["gah " + str(ind) for ind in range(35)]),
            ],
            "p1 parameter value is a recarray, field names "
            "are different. Left has names ('foo', 'bar', 'gah'), right has "
            "names ('fob', 'bar', 'gah').",
        ],
        [
            None,
            np.arange(35, dtype=float),
            "p1 parameter value is a recarray, but other is not.",
        ],
    ],
)
def test_recarray_inequality(capsys, names2, values2, msg):
    """Test inequality for recarray."""
    names1 = ["foo", "bar", "gah"]
    values1 = [
        np.arange(35, dtype=float),
        np.arange(35, dtype=int),
        np.array(["gah " + str(ind) for ind in range(35)]),
    ]
    dtype = []
    for val in values1:
        dtype.append(val.dtype)
    dtype_obj1 = np.dtype(list(zip(names1, dtype)))
    recarr1 = np.rec.fromarrays(values1, dtype=dtype_obj1)
    param1 = uvp.UVParameter(name="p1", value=recarr1)

    if names2 is None:
        param2 = uvp.UVParameter(name="p2", value=values2)
    else:
        dtype = []
        for val in values2:
            dtype.append(val.dtype)
        dtype_obj2 = np.dtype(list(zip(names2, dtype)))
        recarr2 = np.rec.fromarrays(values2, dtype=dtype_obj2)

        param2 = uvp.UVParameter(name="p2", value=recarr2)

    assert param1.__ne__(param2, silent=False)
    captured = capsys.readouterr()
    assert captured.out.startswith(msg)


def test_equality_check_fail():
    """Test equality error for non string, dict or array values."""
    param1 = uvp.UVParameter(name="p1", value=uvp.UVParameter(name="p1", value="Alice"))
    param2 = uvp.UVParameter(name="p2", value=uvp.UVParameter(name="p1", value="Bob"))
    assert param1.__ne__(param2, silent=False)


def test_notclose():
    """Test equality error for values not with tols."""
    param1 = uvp.UVParameter(name="p1", value=1.0)
    param2 = uvp.UVParameter(name="p2", value=1.001)
    assert param1.__ne__(param2, silent=False)


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


@pytest.mark.parametrize(["frame", "selenoid"], frame_selenoid)
def test_location_xyz_latlonalt_match(frame, selenoid):
    if frame == "itrs":
        xyz_val = ref_xyz
        latlonalt_val = ref_latlonalt
    else:
        xyz_val = ref_xyz_moon[selenoid]
        latlonalt_val = ref_latlonalt_moon

    param1 = uvp.LocationParameter(
        name="p1", value=xyz_val, frame=frame, ellipsoid=selenoid
    )
    np.testing.assert_allclose(latlonalt_val, param1.lat_lon_alt())

    if selenoid == "SPHERE":
        param1 = uvp.LocationParameter(name="p1", value=xyz_val, frame=frame)
        np.testing.assert_allclose(latlonalt_val, param1.lat_lon_alt())

    param2 = uvp.LocationParameter(name="p2", frame=frame, ellipsoid=selenoid)
    param2.set_lat_lon_alt(latlonalt_val)

    np.testing.assert_allclose(xyz_val, param2.value)

    param3 = uvp.LocationParameter(name="p2", frame=frame, ellipsoid=selenoid)
    latlonalt_deg_val = np.array(
        [
            latlonalt_val[0] * 180 / np.pi,
            latlonalt_val[1] * 180 / np.pi,
            latlonalt_val[2],
        ]
    )
    param3.set_lat_lon_alt_degrees(latlonalt_deg_val)

    np.testing.assert_allclose(xyz_val, param3.value)


def test_location_acceptability():
    """Test check_acceptability with LocationParameters"""
    val = np.array([0.5, 0.5, 0.5])
    param1 = uvp.LocationParameter("p1", value=val, acceptable_range=[0, 1])
    assert param1.check_acceptability()[0]

    val += 0.5
    param1 = uvp.LocationParameter("p1", value=val, acceptable_range=[0, 1])
    assert not param1.check_acceptability()[0]

    param1 = uvp.LocationParameter("p1", value=val, frame="foo")
    acceptable, reason = param1.check_acceptability()
    assert not acceptable
    assert reason == f"Frame must be one of {utils._range_dict.keys()}"


@pytest.mark.parametrize(
    "sky2",
    [
        SkyCoord(
            ra=Longitude(5.0, unit="hourangle"),
            dec=Latitude(-30, unit="deg"),
            frame="fk5",
            equinox="J2000",
        ),
        SkyCoord(
            ra=Longitude(5.0, unit="hourangle"),
            dec=Latitude(-30, unit="deg") + Latitude(0.0005, unit="arcsec"),
            frame="fk5",
            equinox="J2000",
        ),
    ],
)
def test_skycoord_param_equality(sky_in, sky2):
    param1 = uvp.SkyCoordParameter(name="sky1", value=sky_in)
    param2 = uvp.SkyCoordParameter(name="sky2", value=sky2)

    assert param1 == param2


@pytest.mark.parametrize(
    "change", ["frame", "representation", "separation", "shape", "type"]
)
def test_skycoord_param_inequality(sky_in, change):
    param1 = uvp.SkyCoordParameter(name="sky1", value=sky_in)

    if change == "frame":
        param2 = uvp.SkyCoordParameter(name="sky2", value=sky_in.transform_to("icrs"))
    elif change == "representation":
        sky2 = sky_in.copy()
        sky2.representation_type = CartesianRepresentation
        param2 = uvp.SkyCoordParameter(name="sky2", value=sky2)
    elif change == "separation":
        sky2 = SkyCoord(
            ra=Longitude(5.0, unit="hourangle"),
            dec=Latitude(-30, unit="deg") + Latitude(0.002, unit="arcsec"),
            frame="fk5",
            equinox="J2000",
        )
        param2 = uvp.SkyCoordParameter(name="sky2", value=sky2)
    elif change == "shape":
        sky2 = SkyCoord(
            ra=Longitude([5.0, 5.0], unit="hourangle"),
            dec=Latitude([-30, -30], unit="deg"),
            frame="fk5",
            equinox="J2000",
        )
        param2 = uvp.SkyCoordParameter(name="sky2", value=sky2)
    elif change == "type":
        sky2 = Longitude(5.0, unit="hourangle")
        param2 = uvp.SkyCoordParameter(name="sky2", value=sky2)

    assert param1.__ne__(param2, silent=False)


def test_non_builtin_expected_type():
    with pytest.raises(ValueError) as cm:
        uvp.UVParameter("_test", expected_type="integer")
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
        (bool, (bool, np.bool_)),
        # str type tests the pass through fallback
        (str, str),
        # check builtin attributes too
        ("str", str),
        ("int", (int, np.integer)),
        ("float", (float, np.floating)),
        ("complex", (complex, np.complexfloating)),
        ("bool", (bool, np.bool_)),
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
    assert param1.__ne__(param3, silent=False)
    assert param3 != param1
    assert param2 == param3

    # also try different precision values
    param4 = uvp.UVParameter(
        "_test4",
        value=np.float32(3.0),
        expected_type=np.float32,
        strict_type_check=True,
    )
    assert param1.__ne__(param4, silent=False)

    # make sure it passes when both are strict and equivalent
    param5 = uvp.UVParameter(
        "_test5",
        value=np.float64(3.0),
        expected_type=np.float64,
        strict_type_check=True,
    )
    assert param1 == param5

    # check that it fails for an incompatible generic type
    param6 = uvp.UVParameter(
        "_test6", value=3, expected_type=int, strict_type_check=False
    )
    assert param1 != param6
    assert param6 != param1

    return


def test_strict_expected_type_equality_arrays():
    # make sure it also works with numpy arrays when the dtype matches the strict type
    param1 = uvp.UVParameter(
        "_test1",
        value=np.full((2, 3), 3.0, dtype=np.float64),
        expected_type=np.float64,
        strict_type_check=True,
    )
    param2 = uvp.UVParameter(
        "_test2",
        value=np.full((2, 3), 3.0, dtype=float),
        expected_type=float,
        strict_type_check=False,
    )
    assert param1 == param2
    assert param2 == param1

    param3 = uvp.UVParameter(
        "_test3",
        value=np.full((2, 3), 3.0, dtype=float),
        expected_type=float,
        strict_type_check=True,
    )
    assert param1.__ne__(param3, silent=False)
    assert param3 != param1
    assert param2 == param3

    # also try different precision values
    param4 = uvp.UVParameter(
        "_test4",
        value=np.full((2, 3), 3.0, dtype=np.float32),
        expected_type=np.float32,
        strict_type_check=True,
    )
    assert param1.__ne__(param4, silent=False)

    # make sure it passes when both are strict and equivalent
    param5 = uvp.UVParameter(
        "_test5",
        value=np.full((2, 3), 3.0, dtype=np.float64),
        expected_type=np.float64,
        strict_type_check=True,
    )
    assert param1 == param5

    # check that it fails for an incompatible generic type
    param6 = uvp.UVParameter(
        "_test6",
        value=np.full((2, 3), 3, dtype=int),
        expected_type=int,
        strict_type_check=False,
    )
    assert param1.__ne__(param6, silent=False)
    assert param6.__ne__(param1, silent=False)


def test_scalar_array_parameter_mismatch():
    param1 = uvp.UVParameter("_test1", value=3.0, expected_type=float)
    param2 = uvp.UVParameter("_test2", value=np.asarray([3.0]), expected_type=float)
    assert param1.__ne__(param2, silent=False)
    assert param2.__ne__(param1, silent=False)

    return


def test_value_none_parameter_mismatch():
    param1 = uvp.UVParameter("_test1", value=3.0, expected_type=float)
    param2 = uvp.UVParameter("_test2", value=None)
    assert param1.__ne__(param2, silent=False)
    assert param2.__ne__(param1, silent=False)

    return


def test_spoof():
    param = uvp.UVParameter("test", expected_type=float, required=False, spoof_val=1.0)
    assert param.value is None
    param.apply_spoof()
    assert param.value == 1.0


def test_compare_value_err():
    param = uvp.UVParameter("_test1", value=3.0, tols=[0, 1], expected_type=float)
    with pytest.raises(
        ValueError,
        match="UVParameter value and supplied values are of different types.",
    ):
        param.compare_value("test")


@pytest.mark.parametrize(
    "value,param_value,value_type,status",
    [
        (np.array([1, 2]), np.array([1, 2, 3]), float, False),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), float, True),
        (np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3]), float, True),
        (np.array([2, 3, 4]), np.array([1, 2, 3]), float, True),
        (np.array([4, 5, 6]), np.array([1, 2, 3]), float, False),
        (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3]), float, False),
        ("test_me", "dont_test_me", str, False),
        ("test_me", "test_me", str, True),
    ],
)
def test_compare_value(value, param_value, value_type, status):
    param = uvp.UVParameter(
        "_test1",
        value=param_value,
        tols=None if isinstance(value_type, str) else [0, 1],
        expected_type=value_type,
    )
    assert param.compare_value(value) == status
