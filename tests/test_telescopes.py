# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for telescope objects and functions."""

import os
import warnings

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.units import Quantity

import pyuvdata
from pyuvdata import Telescope, UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.telescopes import (
    get_antenna_params,
    ignore_telescope_param_update_warnings_for,
    unignore_telescope_param_update_warnings_for,
)
from pyuvdata.testing import check_warnings

required_parameters = [
    "_name",
    "_location",
    "_Nants",
    "_antenna_names",
    "_antenna_numbers",
    "_antenna_positions",
]
required_properties = [
    "name",
    "location",
    "Nants",
    "antenna_names",
    "antenna_numbers",
    "antenna_positions",
]
extra_parameters = [
    "_antenna_diameters",
    "_instrument",
    "_mount_type",
    "_Nfeeds",
    "_feed_array",
    "_feed_angle",
]
extra_properties = [
    "antenna_diameters",
    "instrument",
    "mount_type",
    "Nfeeds",
    "feed_array",
    "feed_angle",
]
other_attributes = [
    "citation",
    "telescope_location_lat_lon_alt",
    "telescope_location_lat_lon_alt_degrees",
    "pyuvdata_version_str",
]
astropy_sites = EarthLocation.get_site_names()
while "" in astropy_sites:
    astropy_sites.remove("")

# Using set here is a quick way to drop duplicate entries
expected_known_telescopes = list(
    set(astropy_sites + ["PAPER", "HERA", "SMA", "SZA", "OVRO-LWA", "ATA"])
)


@pytest.fixture(scope="function")
def simplest_working_params():
    return {
        "antenna_positions": {
            0: [0.0, 0.0, 0.0],
            1: [0.0, 0.0, 1.0],
            2: [0.0, 0.0, 2.0],
        },
        "location": EarthLocation.from_geodetic(0, 0, 0),
        "name": "test",
    }


# Tests for Telescope object
def test_parameter_iter():
    "Test expected parameters."
    telescope_obj = pyuvdata.Telescope()
    all_params = []
    for prop in telescope_obj:
        all_params.append(prop)
    for a in required_parameters:
        assert a in all_params, (
            "expected attribute " + a + " not returned in object iterator"
        )


def test_required_parameter_iter():
    "Test expected required parameters."
    telescope_obj = pyuvdata.Telescope()
    required = []
    for prop in telescope_obj.required():
        required.append(prop)
    for a in required_parameters:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )


def test_extra_parameter_iter():
    "Test expected optional parameters."
    telescope_obj = pyuvdata.Telescope()
    extra = []
    for prop in telescope_obj.extra():
        extra.append(prop)
    for a in extra_parameters:
        assert a in extra, "expected attribute " + a + " not returned in extra iterator"


def test_unexpected_parameters():
    "Test for extra parameters."
    telescope_obj = pyuvdata.Telescope()
    expected_parameters = required_parameters + extra_parameters
    attributes = [i for i in list(telescope_obj.__dict__.keys()) if i[0] == "_"]
    for a in attributes:
        assert a in expected_parameters, (
            "unexpected parameter " + a + " found in Telescope"
        )


def test_unexpected_attributes():
    "Test for extra attributes."
    telescope_obj = pyuvdata.Telescope()
    expected_attributes = required_properties + other_attributes
    attributes = [i for i in list(telescope_obj.__dict__.keys()) if i[0] != "_"]
    for a in attributes:
        assert a in expected_attributes, (
            "unexpected attribute " + a + " found in Telescope"
        )


def test_properties():
    "Test that properties can be get and set properly."
    telescope_obj = pyuvdata.Telescope()
    prop_dict = dict(list(zip(required_properties, required_parameters, strict=True)))
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(telescope_obj, k, rand_num)
        this_param = getattr(telescope_obj, v)
        try:
            assert rand_num == this_param.value
        except AssertionError:
            print(f"setting {k} to a random number failed")
            raise


def test_known_telescopes():
    """Test known_telescopes function returns expected results."""
    assert sorted(pyuvdata.telescopes.known_telescopes()) == sorted(
        expected_known_telescopes
    )


def test_update_params_from_known():
    """Cover some edge cases not covered by UVData/UVCal/UVFlag tests."""
    tel = Telescope()
    with pytest.raises(
        ValueError,
        match="The telescope name attribute must be set to update from "
        "known_telescopes.",
    ):
        tel.update_params_from_known_telescopes()

    hera_tel = Telescope.from_known_telescopes(
        "hera", x_orientation="east", feeds=["x", "y"]
    )
    hera_tel_test = Telescope.from_known_telescopes(
        "hera",
        antenna_diameters=hera_tel.antenna_diameters,
        x_orientation="east",
        feeds=["x", "y"],
    )
    assert hera_tel == hera_tel_test

    hera_tel_test.antenna_diameters = None
    with check_warnings(
        UserWarning,
        match=[
            "antenna_diameters is not set because the number of antenna_diameters "
            "on known_telescopes_dict is more than one and does not match Nants "
            "for telescope hera."
        ],
    ):
        hera_tel_test.update_params_from_known_telescopes(
            antenna_diameters=hera_tel.antenna_diameters[0:10]
        )
    assert hera_tel_test.antenna_diameters is None

    hera_tel_test._select_along_param_axis({"Nants": [0]})
    assert hera_tel != hera_tel_test
    with check_warnings(
        UserWarning,
        [
            "telescope_location, Nants, antenna_names, antenna_numbers, "
            "antenna_positions, antenna_diameters are not set"
        ],
    ):
        hera_tel_test.update_params_from_known_telescopes(
            overwrite=True, mount_type=None
        )

    with check_warnings(
        DeprecationWarning, "The Telescope.x_orientation attribute is deprecated"
    ):
        assert hera_tel_test.x_orientation == "east"
        assert hera_tel_test.get_x_orientation_from_feeds() == "east"

    hera_tel_test.antenna_diameters = hera_tel.antenna_diameters
    hera_tel_test.feed_array = None
    hera_tel_test.feed_angle = None
    hera_tel_test.mount_type = None
    with check_warnings(
        UserWarning,
        match=[
            "mount_type is not set because the number of mount_type "
            "on known_telescopes_dict is more than one and does not match Nants "
            "for telescope hera.",
            "feed_array is not set because the number of feed_array "
            "on known_telescopes_dict is more than one and does not match Nants "
            "for telescope hera.",
            "feed_angle is not set because the number of feed_angle "
            "on known_telescopes_dict is more than one and does not match Nants "
            "for telescope hera.",
        ],
    ):
        hera_tel_test.update_params_from_known_telescopes(
            mount_type=hera_tel.mount_type[0:10],
            feed_array=hera_tel.feed_array[0:10],
            feed_angle=hera_tel.feed_angle[0:10],
        )

    hera_tel_test.mount_type = ["fixed"] * hera_tel_test.Nants
    hera_tel_test.mount_type[0] = "phased"
    with check_warnings(
        UserWarning,
        [
            "telescope_location, Nants, antenna_names, antenna_numbers, "
            "antenna_positions, antenna_diameters are not set",
            "Nants has changed, but no information present in the "
            "known telescopes to set mount_type",
        ],
    ):
        hera_tel_test.update_params_from_known_telescopes(
            overwrite=True, mount_type=None
        )

    assert hera_tel_test.mount_type is None

    mwa_tel = Telescope.from_known_telescopes("mwa")
    mwa_tel2 = Telescope()
    mwa_tel2.name = "mwa"
    mwa_tel2.update_params_from_known_telescopes(warn=False)

    assert mwa_tel == mwa_tel2

    vla_tel = Telescope.from_known_telescopes("vla", run_check=False)
    vla_tel2 = Telescope()
    vla_tel2.name = "vla"

    with check_warnings(
        UserWarning,
        match="telescope_location are not set or are being overwritten. "
        "telescope_location are set using values from astropy sites for vla.",
    ):
        vla_tel2.update_params_from_known_telescopes(warn=True, run_check=False)

    assert vla_tel == vla_tel2


def test_from_known():
    for inst in pyuvdata.telescopes.known_telescopes():
        # don't run check b/c some telescopes won't have antenna info defined
        telescope_obj = Telescope.from_known_telescopes(inst, run_check=False)
        assert telescope_obj.name == inst


def test_get_telescope_no_loc():
    with pytest.raises(
        KeyError,
        match="Missing location information in known_telescopes_dict "
        "for telescope test.",
    ):
        Telescope.from_known_telescopes("test", citation="")


def test_hera_loc():
    hera_file = os.path.join(DATA_PATH, "zen.2458098.45361.HH_downselected.uvh5")
    hera_data = UVData()

    hera_data.read(hera_file, read_data=False, file_type="uvh5")

    telescope_obj = Telescope.from_known_telescopes("HERA")

    np.testing.assert_allclose(
        telescope_obj._location.xyz(),
        hera_data.telescope._location.xyz(),
        rtol=hera_data.telescope._location.tols[0],
        atol=hera_data.telescope._location.tols[1],
    )


def test_alternate_antenna_inputs():
    antpos_dict = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0]),
        2: np.array([0.0, 0.0, 2.0]),
    }

    antpos_array = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float)
    antnum = np.array([0, 1, 2])
    antname = np.array(["000", "001", "002"])

    pos, names, nums = get_antenna_params(antenna_positions=antpos_dict)
    pos2, names2, nums2 = get_antenna_params(
        antenna_positions=antpos_array, antenna_numbers=antnum, antenna_names=antname
    )

    np.testing.assert_allclose(pos, pos2, rtol=0, atol=1e-3)
    assert np.all(names == names2)
    assert np.all(nums == nums2)

    antpos_dict = {
        "000": np.array([0, 0, 0]),
        "001": np.array([0, 0, 1]),
        "002": np.array([0, 0, 2]),
    }
    pos, names, nums = get_antenna_params(antenna_positions=antpos_dict)
    np.testing.assert_allclose(pos, pos2, rtol=0, atol=1e-3)
    assert np.all(names == names2)
    assert np.all(nums == nums2)


def test_new_errors(simplest_working_params):
    simplest_working_params["location"] = Quantity([0, 0, 0], unit="m")
    with pytest.raises(
        ValueError,
        match="telescope_location is not a supported type. It must be one of ",
    ):
        Telescope.new(**simplest_working_params)


@pytest.mark.parametrize(
    ["kwargs", "err_msg"],
    [
        [
            {
                "antenna_positions": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "Either antenna_numbers or antenna_names must be provided",
        ],
        [
            {
                "antenna_positions": {1: [0, 1, 2], "2": [3, 4, 5]},
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "antenna_positions must be a dictionary with keys that are all type int "
            "or all type str",
        ],
        [
            {
                "antenna_positions": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
                "antenna_names": ["foo", "bar", "baz"],
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "Antenna names must be integers",
        ],
        [
            {
                "antenna_positions": "foo",
                "antenna_numbers": [0, 1, 2],
                "antenna_names": ["foo", "bar", "baz"],
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "antenna_positions must be a numpy array",
        ],
        [
            {
                "antenna_positions": np.array([0, 0, 0]),
                "antenna_numbers": np.array([0]),
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "antenna_positions must be a 2D array",
        ],
        [
            {
                "antenna_positions": {
                    0: [0.0, 0.0, 0.0],
                    1: [0.0, 0.0, 1.0],
                    2: [0.0, 0.0, 2.0],
                },
                "antenna_names": ["foo", "bar", "foo"],
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "Duplicate antenna names found",
        ],
        [
            {
                "antenna_positions": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
                "antenna_numbers": [0, 1, 0],
                "antenna_names": ["foo", "bar", "baz"],
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "Duplicate antenna numbers found",
        ],
        [
            {
                "antenna_positions": {
                    0: [0.0, 0.0, 0.0],
                    1: [0.0, 0.0, 1.0],
                    2: [0.0, 0.0, 2.0],
                },
                "antenna_names": ["foo", "bar"],
                "location": EarthLocation.from_geodetic(0, 0, 0),
                "name": "test",
            },
            "antenna_numbers and antenna_names must have the same length",
        ],
    ],
)
def test_bad_antenna_inputs(kwargs, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        Telescope.new(**kwargs)


@pytest.mark.parametrize("xorient", ["e", "n", "east", "NORTH"])
def test_passing_xorient(simplest_working_params, xorient):
    with check_warnings(UserWarning, "Unknown polarization basis"):
        tel = Telescope.new(
            x_orientation=xorient, mount_type=["fixed"] * 3, **simplest_working_params
        )

    with check_warnings(
        DeprecationWarning, "The Telescope.x_orientation attribute is deprecated"
    ):
        name = "east" if xorient.lower().startswith("e") else "north"
        assert tel.x_orientation == name
        assert tel.get_x_orientation_from_feeds() == name


@pytest.mark.parametrize("xorient", ["e", "n", "east", "NORTH"])
@pytest.mark.parametrize("feed_angle_offset", [-np.pi, 0, np.pi])
def test_x_orient_wrap(simplest_working_params, xorient, feed_angle_offset):
    with check_warnings(UserWarning, "Unknown polarization basis"):
        tel = Telescope.new(
            x_orientation=xorient, mount_type=["fixed"] * 3, **simplest_working_params
        )

    tel.feed_angle += feed_angle_offset

    name = "east" if xorient.lower().startswith("e") else "north"
    assert tel.get_x_orientation_from_feeds() == name


def test_xorient_dep_warning(simplest_working_params):
    tel = Telescope.new(
        feeds=["x", "y"],
        x_orientation="east",
        mount_type="fixed",
        **simplest_working_params,
    )
    with check_warnings(
        DeprecationWarning, ["The Telescope.x_orientation attribute is deprecated"] * 3
    ):
        assert tel.x_orientation == "east"
        tel.x_orientation = "north"
        assert tel.x_orientation == "north"


def test_passing_diameters(simplest_working_params):
    tel = Telescope.new(
        antenna_diameters=np.array([14.0, 15.0, 16.0]), **simplest_working_params
    )
    np.testing.assert_allclose(tel.antenna_diameters, np.array([14.0, 15.0, 16.0]))


def test_get_enu_antpos():
    filename = os.path.join(DATA_PATH, "zen.2458432.34569.uvh5")

    tel = Telescope.from_hdf5(filename)
    # no center, no pick data ants
    antpos = tel.get_enu_antpos()
    assert antpos.shape == (tel.Nants, 3)
    assert np.isclose(antpos[0, 0], -105.0353015431546, rtol=0, atol=1e-3)


def test_ignore_param_updates_error():
    with pytest.raises(ValueError, match="'deathstar' is not a known telescope"):
        ignore_telescope_param_update_warnings_for("deathstar")

    with pytest.raises(ValueError, match="'deathstar' is not a known telescope"):
        unignore_telescope_param_update_warnings_for("deathstar")


def test_update_without_warning():
    ignore_telescope_param_update_warnings_for("hera")
    t = Telescope()
    t.name = "HERA"
    with warnings.catch_warnings():
        # error if there is a warning
        warnings.simplefilter("error")
        t.update_params_from_known_telescopes()
    unignore_telescope_param_update_warnings_for("hera")


@pytest.mark.parametrize("warn", [True, False])
@pytest.mark.parametrize("drop_param", ["feed_angle", "feed_array", "mount_type"])
def test_feed_errs(simplest_working_params, drop_param, warn):
    tel = Telescope.new(**simplest_working_params)
    tel.Nfeeds = 2
    if not warn:
        tel._feed_array.required = drop_param != "feed_array"
        tel._feed_angle.required = drop_param != "feed_angle"
        tel._mount_type.required = drop_param != "mount_type"
    if drop_param != "feed_array":
        tel.feed_array = np.full((tel.Nants, tel.Nfeeds), "x")
    if drop_param != "feed_angle":
        tel.feed_angle = np.zeros((tel.Nants, tel.Nfeeds))
    if drop_param != "mount_type":
        tel.mount_type = ["fixed"] * tel.Nants
    msg = "Parameter feed_array, feed_angle, and mount_type must be set together."
    if warn:
        with check_warnings(UserWarning, match=msg):
            tel.check()
        assert tel.feed_array is None
        assert tel.feed_angle is None
        assert tel.mount_type is None
    else:
        with pytest.raises(ValueError, match=msg):
            tel.check()


def test_feed_order_error(simplest_working_params):
    feed_array = np.tile(["l", "r"], (3, 1))
    feed_angle = np.arange(6, dtype=float).reshape((3, 2))
    mount_type = ["fixed"] * 3
    tel = Telescope.new(
        feed_array=feed_array,
        feed_angle=feed_angle,
        mount_type=mount_type,
        **simplest_working_params,
    )

    with pytest.raises(ValueError, match="order must be one of: 'AIPS', 'CASA', or"):
        tel.reorder_feeds("XYZ")


def test_feed_order_noop(simplest_working_params):
    # Make sure that no feeds just returns
    tel = Telescope.new(**simplest_working_params)
    tel2 = tel.copy()

    with check_warnings(None, match=""):
        tel.reorder_feeds("AIPS")

    assert tel == tel2


@pytest.mark.parametrize(
    "order,flipped", [["AIPS", True], ["CASA", True], [["l", "r", "x"], False]]
)
def test_feed_order(simplest_working_params, order, flipped):
    feed_array = np.tile(["l", "r"], (3, 1))
    feed_angle = np.arange(6, dtype=float).reshape((3, 2))
    mount_type = ["fixed"] * 3
    tel = Telescope.new(
        feed_array=feed_array,
        feed_angle=feed_angle,
        mount_type=mount_type,
        **simplest_working_params,
    )
    feed_array = feed_array.copy()
    feed_angle = feed_angle.copy()

    tel.reorder_feeds(order=order)
    assert np.array_equal(tel.feed_array, feed_array[:, :: (-1 if flipped else 1)])
    assert np.array_equal(tel.feed_angle, feed_angle[:, :: (-1 if flipped else 1)])
    assert tel.Nfeeds == 2


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        [{"order": [0, 1, 2, 3, 4]}, "If order is an index array, it must contain all"],
        [{"order": [1]}, "If order is an index array, it must contain all"],
        [{"order": "foo"}, "order must be one of 'number', 'name', '-number', '-name'"],
    ],
)
def test_antenna_order_errors(simplest_working_params, kwargs, msg):
    tel = Telescope.new(**simplest_working_params)
    with pytest.raises(ValueError, match=msg):
        tel.reorder_antennas(**kwargs)


@pytest.mark.parametrize("order", ["name", "number"])
def test_antenna_order(simplest_working_params, order):
    tel = Telescope.new(**simplest_working_params)
    tel2 = tel.copy()
    tel3 = tel.copy()

    # Test no-op
    tel.reorder_antennas(order=order)
    assert tel == tel2

    # Test reverse ordering
    tel.reorder_antennas(order=("-" + order))
    tel2.reorder_antennas(order=[2, 1, 0])
    assert tel == tel2
    assert tel != tel3

    # Reverse the reverse!
    tel.reorder_antennas(order=order)
    tel2.reorder_antennas(order=[2, 1, 0])
    assert tel == tel2
    assert tel == tel3


@pytest.mark.parametrize("add_method", ["__add__", "__iadd__"])
@pytest.mark.parametrize("axis", ["Nfeeds", "Nants"])
@pytest.mark.parametrize("scenario", ["overlap", "inv_overlap", "interleave", "concat"])
def test_telescope_add(add_method, axis, scenario):
    tel1 = Telescope.from_known_telescopes("hera")
    # Set the feed information so that we can check it
    tel1.feed_array = np.array([["x", "y"]] * tel1.Nants)
    tel1.feed_angle = np.array([[np.pi / 2, 0.0]] * tel1.Nants)
    tel1.Nfeeds = 2

    tel2 = tel1.copy()
    check_tel = tel1.copy()

    nitem = getattr(tel1, axis)
    if scenario == "overlap":
        tel1_dict = {}
        tel2_dict = {axis: np.arange(nitem // 2, nitem)}
    elif scenario == "inv_overlap":
        tel1_dict = {axis: np.arange(nitem // 2, nitem)}
        tel2_dict = {}
    elif scenario == "interleave":
        tel1_dict = {axis: np.arange(0, nitem, 2)}
        tel2_dict = {axis: np.arange(1, nitem, 2)}
    elif scenario == "concat":
        tel1_dict = {axis: np.arange(0, nitem - 1)}
        tel2_dict = {axis: np.arange(nitem - 1, nitem)}

    tel1._select_along_param_axis(tel1_dict)
    tel2._select_along_param_axis(tel2_dict)
    tel3 = getattr(tel1, add_method)(tel2)

    if axis == "Nants":
        tel3.reorder_antennas(order="number")
    elif axis == "Nfeeds":
        tel3.reorder_feeds(order="AIPS")

    assert tel3 == check_tel


@pytest.mark.parametrize("add_method", ["__add__", "__iadd__"])
def test_telescope_add_noop(simplest_working_params, add_method):
    tel1 = Telescope.new(**simplest_working_params)
    tel2 = tel1.copy()
    tel3 = getattr(tel1, add_method)(tel2)
    assert tel2 == tel3


def test_telescope_add_diff_feeds(simplest_working_params):
    feed_array = np.array([["l", "r"], ["r", "l"], ["x", "y"]])
    feed_angle = np.arange(6, dtype=float).reshape((3, 2))
    tel1 = Telescope.new(
        feed_array=feed_array,
        feed_angle=feed_angle,
        mount_type="fixed",
        **simplest_working_params,
    )
    tel2 = tel1.copy()
    check_tel = tel1.copy()

    tel1._select_along_param_axis({"Nants": [0, 1]})
    tel2._select_along_param_axis({"Nants": [2]})
    tel3 = tel1 + tel2
    assert check_tel == tel3


@pytest.mark.parametrize("add_method", ["__add__", "__iadd__"])
def test_telescope_add_errs(simplest_working_params, add_method):
    tel = Telescope.new(**simplest_working_params)
    with pytest.raises(
        ValueError, match=r"Only Telescope \(or subclass\) objects can be added"
    ):
        getattr(tel, add_method)(None)

    tel2 = tel.copy()
    tel2.name = "other test"
    with pytest.raises(ValueError, match="Parameter Telescope.name does not match."):
        getattr(tel, add_method)(tel2)


def test_telescope_mount_feed_multicast(simplest_working_params):
    tel = Telescope.new(
        **simplest_working_params,
        mount_type=["fixed"] * 3,
        feed_array=np.array([["x", "y"]] * 3),
        feed_angle=[[0, np.pi / 2]] * 3,
    )
    tel2 = Telescope.new(
        **simplest_working_params,
        mount_type="fixed",
        feed_array=["x", "y"],
        feed_angle=[0, np.pi / 2],
    )

    assert tel == tel2


def test_known_telescopes_keywords(simplest_working_params):
    tel1 = Telescope.new(**simplest_working_params)
    tel2 = Telescope.from_known_telescopes(
        name=tel1.name,
        location=tel1.location,
        antenna_numbers=tel1.antenna_numbers,
        antenna_names=tel1.antenna_names,
        antenna_positions=tel1.antenna_positions,
    )

    assert tel1 == tel2
