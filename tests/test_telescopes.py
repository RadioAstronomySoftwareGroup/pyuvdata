# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for telescope objects and functions.

"""
import copy
import os

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.units import Quantity

import pyuvdata
from pyuvdata import Telescope, UVData, get_telescope
from pyuvdata.data import DATA_PATH
from pyuvdata.telescopes import _KNOWN_TELESCOPES, get_antenna_params
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
extra_parameters = ["_antenna_diameters", "_x_orientation", "_instrument"]
extra_properties = ["antenna_diameters", "x_orientation", "instrument"]
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
    set(astropy_sites + ["PAPER", "HERA", "SMA", "SZA", "OVRO-LWA"])
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
    prop_dict = dict(list(zip(required_properties, required_parameters)))
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(telescope_obj, k, rand_num)
        this_param = getattr(telescope_obj, v)
        try:
            assert rand_num == this_param.value
        except AssertionError:
            print("setting {prop_name} to a random number failed".format(prop_name=k))
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

    hera_tel = Telescope.from_known_telescopes("hera")
    known_dict = copy.deepcopy(_KNOWN_TELESCOPES)
    known_dict["HERA"]["antenna_diameters"] = hera_tel.antenna_diameters

    hera_tel_test = Telescope.from_known_telescopes(
        "hera", known_telescope_dict=known_dict
    )
    assert hera_tel == hera_tel_test

    known_dict["HERA"]["antenna_diameters"] = hera_tel.antenna_diameters[0:10]
    known_dict["HERA"]["x_orientation"] = "east"
    hera_tel_test.antenna_diameters = None
    with check_warnings(
        UserWarning,
        match=[
            "antenna_diameters are not set because the number of antenna_diameters "
            "on known_telescopes_dict is more than one and does not match Nants "
            "for telescope hera.",
            "x_orientation are not set or are being overwritten. x_orientation "
            "are set using values from known telescopes for hera.",
        ],
    ):
        hera_tel_test.update_params_from_known_telescopes(
            known_telescope_dict=known_dict
        )
    assert hera_tel_test.antenna_diameters is None
    assert hera_tel_test.x_orientation == "east"

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

        with check_warnings(
            DeprecationWarning,
            match="This method is deprecated and will be removed in version 3.2. "
            "If you just need a telescope location, use the "
            "known_telescope_location function. For a full Telescope object use "
            "the classmethod Telescope.from_known_telescopes.",
        ):
            tel_obj2 = get_telescope(inst)

        assert tel_obj2 == telescope_obj


def test_old_attr_names():
    mwa_tel = Telescope.from_known_telescopes("mwa")
    with check_warnings(
        DeprecationWarning,
        match="The Telescope.telescope_name attribute is deprecated, use "
        "Telescope.name instead. This will become an error in version 3.2.",
    ):
        assert mwa_tel.telescope_name == mwa_tel.name

    with check_warnings(
        DeprecationWarning,
        match="The Telescope.telescope_location attribute is deprecated, use "
        "Telescope.location instead (which contains an astropy "
        "EarthLocation object). This will become an error in version 3.2.",
    ):
        np.testing.assert_allclose(mwa_tel.telescope_location, mwa_tel._location.xyz())

    with check_warnings(
        DeprecationWarning,
        match="The Telescope.telescope_name attribute is deprecated, use "
        "Telescope.name instead. This will become an error in version 3.2.",
    ):
        mwa_tel.telescope_name = "foo"
    assert mwa_tel.name == "foo"

    hera_tel = Telescope.from_known_telescopes("hera")
    with check_warnings(
        DeprecationWarning,
        match="The Telescope.telescope_location attribute is deprecated, use "
        "Telescope.location instead (which should be set to an astropy "
        "EarthLocation object). This will become an error in version 3.2.",
    ):
        mwa_tel.telescope_location = hera_tel._location.xyz()
    assert mwa_tel._location == hera_tel._location


@pytest.mark.filterwarnings("ignore:Directly accessing the KNOWN_TELESCOPES")
def test_old_known_tel_dict_keys():
    from pyuvdata.telescopes import KNOWN_TELESCOPES

    hera_tel = Telescope.from_known_telescopes("hera")

    warn_msg = [
        "Directly accessing the KNOWN_TELESCOPES dict is deprecated. If you "
        "need a telescope location, use the known_telescope_location function. "
        "For a full Telescope object use the classmethod "
        "Telescope.from_known_telescopes."
    ] * 2

    with check_warnings(DeprecationWarning, match=warn_msg):
        assert KNOWN_TELESCOPES["HERA"]["latitude"] == hera_tel.location.lat.rad

    with check_warnings(DeprecationWarning, match=warn_msg):
        assert KNOWN_TELESCOPES["HERA"]["longitude"] == hera_tel.location.lon.rad

    with check_warnings(DeprecationWarning, match=warn_msg):
        assert KNOWN_TELESCOPES["HERA"][
            "altitude"
        ] == hera_tel.location.height.to_value("m")

    with check_warnings(DeprecationWarning, match=warn_msg):
        np.testing.assert_allclose(
            KNOWN_TELESCOPES["HERA"]["center_xyz"],
            Quantity(hera_tel.location.geocentric).to_value("m"),
        )
    with check_warnings(DeprecationWarning, match=warn_msg):
        assert KNOWN_TELESCOPES["HERA"]["citation"] == hera_tel.citation

    assert len(KNOWN_TELESCOPES["MWA"]) == 1
    for key, val in KNOWN_TELESCOPES.items():
        assert val == _KNOWN_TELESCOPES[key]


def test_get_telescope_no_loc():
    test_telescope_dict = {"test": {"citation": ""}}
    with pytest.raises(
        KeyError,
        match="Missing location information in known_telescopes_dict "
        "for telescope test.",
    ):
        Telescope.from_known_telescopes(
            "test", known_telescope_dict=test_telescope_dict
        )


def test_hera_loc():
    hera_file = os.path.join(DATA_PATH, "zen.2458098.45361.HH.uvh5_downselected")
    hera_data = UVData()
    hera_data.read(hera_file, read_data=False, file_type="uvh5")

    telescope_obj = Telescope.from_known_telescopes("HERA")

    assert np.allclose(
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

    assert np.allclose(pos, pos2)
    assert np.all(names == names2)
    assert np.all(nums == nums2)

    antpos_dict = {
        "000": np.array([0, 0, 0]),
        "001": np.array([0, 0, 1]),
        "002": np.array([0, 0, 2]),
    }
    pos, names, nums = get_antenna_params(antenna_positions=antpos_dict)
    assert np.allclose(pos, pos2)
    assert np.all(names == names2)
    assert np.all(nums == nums2)


def test_new_errors(simplest_working_params):
    simplest_working_params["location"] = Quantity([0, 0, 0], unit="m")
    with pytest.raises(
        ValueError,
        match="telescope_location has an unsupported type, it must be one of ",
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
    tel = Telescope.new(x_orientation=xorient, **simplest_working_params)
    if xorient.lower().startswith("e"):
        assert tel.x_orientation == "east"
    else:
        assert tel.x_orientation == "north"


def test_passing_diameters(simplest_working_params):
    tel = Telescope.new(
        antenna_diameters=np.array([14.0, 15.0, 16.0]), **simplest_working_params
    )
    np.testing.assert_allclose(tel.antenna_diameters, np.array([14.0, 15.0, 16.0]))


def test_get_enu_antpos():
    filename = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5")

    tel = Telescope.from_hdf5(filename)
    # no center, no pick data ants
    antpos = tel.get_enu_antpos()
    assert antpos.shape == (tel.Nants, 3)
    assert np.isclose(antpos[0, 0], 19.340211050751535)
