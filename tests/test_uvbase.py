# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvbase object."""

import re

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Distance, EarthLocation, Latitude, Longitude, SkyCoord
from astropy.time import Time

from pyuvdata import Telescope, parameter as uvp
from pyuvdata.testing import check_warnings
from pyuvdata.uvbase import UVBase, _warning

ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)


class UVTest(UVBase):
    def __init__(self):
        """UVBase test object."""
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
            "location",
            description="location",
            value=EarthLocation.from_geocentric(*ref_xyz, unit="m"),
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

        values = [
            np.arange(35, dtype=float),
            np.arange(35, dtype=int),
            np.array(["gah " + str(ind) for ind in range(35)]),
        ]
        dtype_list = [val.dtype for val in values]

        self._recarray = uvp.UVParameter(
            "recarray",
            description="A recarray object.",
            expected_type=[dtype.type for dtype in dtype_list],
            value=np.rec.fromarrays(
                values,
                dtype=np.dtype(
                    list(zip(["foo", "bar", "gah"], dtype_list, strict=True))
                ),
            ),
            form=(35,),
        )

        self._skycoord_array = uvp.SkyCoordParameter(
            "skycoord_array",
            description="A skycoord array.",
            value=SkyCoord(
                ra=Longitude([5.0, 5.1], unit="hourangle"),
                dec=Latitude([-30, -30], unit="deg"),
                frame="fk5",
                equinox="J2000",
            ),
            form=(2),
        )

        self._skycoord_scalar = uvp.SkyCoordParameter(
            "skycoord_scalar",
            description="A skycoord scalar.",
            value=SkyCoord(
                ra=Longitude(5.0, unit="hourangle"),
                dec=Latitude(-30, unit="deg"),
                frame="fk5",
                equinox="J2000",
            ),
            form=(),
        )

        self._telescope = uvp.UVParameter(
            "telescope",
            description="A telescope.",
            value=Telescope.new(
                location=EarthLocation.from_geodetic(0, 0, 0),
                name="mock",
                antenna_positions={
                    0: [0.0, 0.0, 0.0],
                    1: [0.0, 0.0, 1.0],
                    2: [0.0, 0.0, 2.0],
                },
            ),
            expected_type=Telescope,
        )

        super().__init__()

        self.telescope = Telescope.from_known_telescopes("mwa")


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


def test_inequality_extra(capsys):
    """Basic equality test."""
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.optional_int1 = 4
    # use `__ne__` rather than `!=` throughout so we can cover print lines
    assert test_obj.__ne__(test_obj2, silent=False)

    captured = capsys.readouterr()
    assert captured.out.startswith(
        "optional_int1 parameter value can be cast to an array and tested with "
        "np.allclose. The values are not close\nparameter _optional_int1 does "
        "not match. Left is 3, right is 4."
    )


def test_inequality_different_extras(capsys):
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
    assert test_obj.__ne__(test_obj2, silent=False)

    captured = capsys.readouterr()
    assert captured.out.startswith(
        "Sets of extra parameters do not match. Left is ['_optional_int1', "
        "'_optional_int2', '_unset_int1'], right is ['_optional_int1', "
        "'_optional_int2', '_optional_int3', '_unset_int1']."
    )

    assert test_obj != test_obj2


def test_inequality(capsys):
    """Check that inequality is handled correctly."""
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.float1 = 13
    assert test_obj.__ne__(test_obj2, silent=False)

    captured = capsys.readouterr()
    assert captured.out.startswith(
        "float1 parameter value can be cast to an array and tested with "
        "np.allclose. The values are not close\nparameter _float1 does not "
        "match. Left is 18.2, right is 13."
    )


def test_class_inequality(capsys):
    """Test equality error for different classes."""
    test_obj = UVTest()
    assert test_obj.__ne__(test_obj._floatarr, silent=False)

    captured = capsys.readouterr()
    assert captured.out.startswith("Classes do not match")


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
    with (
        check_warnings(
            UserWarning,
            "Parameter _floatarr is a Quantity object, but the expected type "
            "is a precision identifier: (<class 'float'>, <class 'numpy.floating'>). "
            "Testing the precision of the value, "
            "but this check will fail in a future version.",
        ),
        pytest.raises(
            ValueError, match="UVParameter _floatarr is not the appropriate type. "
        ),
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
    with check_warnings(
        UserWarning,
        "Parameter _quantity_scalar is a Quantity object, but the expected type "
        "is a precision identifier: (<class 'float'>, <class 'numpy.floating'>). "
        "Testing the precision of the value, but this "
        "check will fail in a future version.",
    ):
        test_obj.check()


def test_recarray_dtype():
    """Test check when a recarray has wrong expected type."""
    test_obj = UVTest()
    test_obj.check()

    test_obj._recarray.expected_type = float
    with pytest.raises(
        ValueError,
        match="Parameter _recarray is a recarray, but the expected type is not a list "
        "with a length equal to the number of columns in the recarray.",
    ):
        test_obj.check()

    test_obj._recarray.expected_type = [int, float]
    with pytest.raises(
        ValueError,
        match="Parameter _recarray is a recarray, but the expected type is not a list "
        "with a length equal to the number of columns in the recarray.",
    ):
        test_obj.check()

    test_obj._recarray.expected_type = [float, float, str]
    with pytest.raises(
        ValueError,
        match="Parameter _recarray is a recarray, the columns do not all have the "
        "expected types.",
    ):
        test_obj.check()


def test_skycoord_shape():
    test_obj = UVTest()
    test_obj._skycoord_array.form = (3,)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "UVParameter _skycoord_array is not expected shape. "
            "Parameter shape is (2,), expected shape is (3,)."
        ),
    ):
        test_obj.check()


def test_skycoord_type():
    test_obj = UVTest()
    test_obj.skycoord_scalar = Longitude(5.0, unit="hourangle")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "UVParameter _skycoord_scalar should be a subclass of a SkyCoord object"
        ),
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


def test_name_error():
    test_obj = UVTest()
    test_obj._location.name = "place"
    with pytest.raises(ValueError, match="UVParameter _location does not follow the"):
        test_obj.check()


def test_telescope_inequality(capsys):
    test_obj = UVTest()
    test_obj2 = test_obj.copy()
    test_obj2.telescope = Telescope.from_known_telescopes("hera")

    assert test_obj.__ne__(test_obj2, silent=False)

    captured = capsys.readouterr()
    assert captured.out.startswith("parameter _telescope does not match.")
