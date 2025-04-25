# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for polarization utility functions."""

import copy

import numpy as np
import pytest

from pyuvdata import utils
from pyuvdata.testing import check_warnings
from pyuvdata.uvbeam.uvbeam import _convert_feeds_to_pols


def test_pol_funcs():
    """Test utility functions to convert between polarization strings and numbers"""

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    pol_str = ["yx", "xy", "yy", "xx", "lr", "rl", "ll", "rr", "pI", "pQ", "pU", "pV"]
    assert pol_nums == utils.polstr2num(pol_str)
    assert pol_str == utils.polnum2str(pol_nums)
    # Check individuals
    assert utils.polstr2num("YY") == -6
    assert utils.polnum2str(4) == "pV"
    # Check errors
    pytest.raises(KeyError, utils.polstr2num, "foo")
    pytest.raises(ValueError, utils.polstr2num, 1)
    pytest.raises(ValueError, utils.polnum2str, 7.3)
    # Check parse
    assert utils.parse_polstr("xX") == "xx"
    assert utils.parse_polstr("XX") == "xx"
    assert utils.parse_polstr("i") == "pI"


def test_pol_funcs_x_orientation():
    """Test functions to convert between pol strings and numbers with x_orientation."""

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]

    x_orient1 = "e"
    pol_str = ["ne", "en", "nn", "ee", "lr", "rl", "ll", "rr", "pI", "pQ", "pU", "pV"]
    assert pol_nums == utils.polstr2num(pol_str, x_orientation=x_orient1)
    assert pol_str == utils.polnum2str(pol_nums, x_orientation=x_orient1)
    # Check individuals
    assert utils.polstr2num("NN", x_orientation=x_orient1) == -6
    assert utils.polnum2str(4) == "pV"
    # Check errors
    pytest.raises(KeyError, utils.polstr2num, "foo", x_orientation=x_orient1)
    pytest.raises(ValueError, utils.polstr2num, 1, x_orientation=x_orient1)
    pytest.raises(ValueError, utils.polnum2str, 7.3, x_orientation=x_orient1)
    # Check parse
    assert utils.parse_polstr("eE", x_orientation=x_orient1) == "ee"
    assert utils.parse_polstr("xx", x_orientation=x_orient1) == "ee"
    assert utils.parse_polstr("NN", x_orientation=x_orient1) == "nn"
    assert utils.parse_polstr("yy", x_orientation=x_orient1) == "nn"
    assert utils.parse_polstr("i", x_orientation=x_orient1) == "pI"

    x_orient2 = "n"
    pol_str = ["en", "ne", "ee", "nn", "lr", "rl", "ll", "rr", "pI", "pQ", "pU", "pV"]
    assert pol_nums == utils.polstr2num(pol_str, x_orientation=x_orient2)
    assert pol_str == utils.polnum2str(pol_nums, x_orientation=x_orient2)
    # Check individuals
    assert utils.polstr2num("EE", x_orientation=x_orient2) == -6
    assert utils.polnum2str(4) == "pV"
    # Check errors
    pytest.raises(KeyError, utils.polstr2num, "foo", x_orientation=x_orient2)
    pytest.raises(ValueError, utils.polstr2num, 1, x_orientation=x_orient2)
    pytest.raises(ValueError, utils.polnum2str, 7.3, x_orientation=x_orient2)
    # Check parse
    assert utils.parse_polstr("nN", x_orientation=x_orient2) == "nn"
    assert utils.parse_polstr("xx", x_orientation=x_orient2) == "nn"
    assert utils.parse_polstr("EE", x_orientation=x_orient2) == "ee"
    assert utils.parse_polstr("yy", x_orientation=x_orient2) == "ee"
    assert utils.parse_polstr("i", x_orientation=x_orient2) == "pI"

    # check warnings for non-recognized x_orientation
    with check_warnings(UserWarning, "x_orientation not recognized"):
        assert utils.polstr2num("xx", x_orientation="foo") == -5

    with check_warnings(UserWarning, "x_orientation not recognized"):
        assert utils.polnum2str(-6, x_orientation="foo") == "yy"


def test_jones_num_funcs():
    """Test functions to convert between jones polarization strings and numbers."""

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    jstr = ["Jyx", "Jxy", "Jyy", "Jxx", "Jlr", "Jrl", "Jll", "Jrr"]
    assert jnums == utils.jstr2num(jstr)
    assert jstr, utils.jnum2str(jnums)
    # Check shorthands
    jstr = ["yx", "xy", "yy", "y", "xx", "x", "lr", "rl", "ll", "l", "rr", "r"]
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == utils.jstr2num(jstr)
    # Check individuals
    assert utils.jstr2num("jyy") == -6
    assert utils.jnum2str(-7) == "Jxy"
    # Check errors
    pytest.raises(KeyError, utils.jstr2num, "foo")
    pytest.raises(ValueError, utils.jstr2num, 1)
    pytest.raises(ValueError, utils.jnum2str, 7.3)

    # check parse method
    assert utils.pol.parse_jpolstr("x") == "Jxx"
    assert utils.pol.parse_jpolstr("xy") == "Jxy"
    assert utils.pol.parse_jpolstr("XY") == "Jxy"


def test_jones_num_funcs_x_orientation():
    """Test functions to convert jones pol strings and numbers with x_orientation."""

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    x_orient1 = "east"
    jstr = ["Jne", "Jen", "Jnn", "Jee", "Jlr", "Jrl", "Jll", "Jrr"]
    assert jnums == utils.jstr2num(jstr, x_orientation=x_orient1)
    assert jstr == utils.jnum2str(jnums, x_orientation=x_orient1)
    # Check shorthands
    jstr = ["ne", "en", "nn", "n", "ee", "e", "lr", "rl", "ll", "l", "rr", "r"]
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == utils.jstr2num(jstr, x_orientation=x_orient1)
    # Check individuals
    assert utils.jstr2num("jnn", x_orientation=x_orient1) == -6
    assert utils.jnum2str(-7, x_orientation=x_orient1) == "Jen"
    # Check errors
    pytest.raises(KeyError, utils.jstr2num, "foo", x_orientation=x_orient1)
    pytest.raises(ValueError, utils.jstr2num, 1, x_orientation=x_orient1)
    pytest.raises(ValueError, utils.jnum2str, 7.3, x_orientation=x_orient1)

    # check parse method
    assert utils.pol.parse_jpolstr("e", x_orientation=x_orient1) == "Jee"
    assert utils.pol.parse_jpolstr("x", x_orientation=x_orient1) == "Jee"
    assert utils.pol.parse_jpolstr("y", x_orientation=x_orient1) == "Jnn"
    assert utils.pol.parse_jpolstr("en", x_orientation=x_orient1) == "Jen"
    assert utils.pol.parse_jpolstr("NE", x_orientation=x_orient1) == "Jne"

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    x_orient2 = "north"
    jstr = ["Jen", "Jne", "Jee", "Jnn", "Jlr", "Jrl", "Jll", "Jrr"]
    assert jnums == utils.jstr2num(jstr, x_orientation=x_orient2)
    assert jstr == utils.jnum2str(jnums, x_orientation=x_orient2)
    # Check shorthands
    jstr = ["en", "ne", "ee", "e", "nn", "n", "lr", "rl", "ll", "l", "rr", "r"]
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == utils.jstr2num(jstr, x_orientation=x_orient2)
    # Check individuals
    assert utils.jstr2num("jee", x_orientation=x_orient2) == -6
    assert utils.jnum2str(-7, x_orientation=x_orient2) == "Jne"
    # Check errors
    pytest.raises(KeyError, utils.jstr2num, "foo", x_orientation=x_orient2)
    pytest.raises(ValueError, utils.jstr2num, 1, x_orientation=x_orient2)
    pytest.raises(ValueError, utils.jnum2str, 7.3, x_orientation=x_orient2)

    # check parse method
    assert utils.pol.parse_jpolstr("e", x_orientation=x_orient2) == "Jee"
    assert utils.pol.parse_jpolstr("x", x_orientation=x_orient2) == "Jnn"
    assert utils.pol.parse_jpolstr("y", x_orientation=x_orient2) == "Jee"
    assert utils.pol.parse_jpolstr("en", x_orientation=x_orient2) == "Jen"
    assert utils.pol.parse_jpolstr("NE", x_orientation=x_orient2) == "Jne"

    # check warnings for non-recognized x_orientation
    with check_warnings(UserWarning, "x_orientation not recognized"):
        assert utils.jstr2num("x", x_orientation="foo") == -5

    with check_warnings(UserWarning, "x_orientation not recognized"):
        assert utils.jnum2str(-6, x_orientation="foo") == "Jyy"


def test_conj_pol():
    """Test function to conjugate pols"""

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    cpol_nums = [-7, -8, -6, -5, -3, -4, -2, -1, 1, 2, 3, 4]
    assert pol_nums == utils.conj_pol(cpol_nums)
    assert utils.conj_pol(pol_nums) == cpol_nums
    # fmt: off
    pol_str = ['yx', 'xy', 'yy', 'xx', 'ee', 'nn', 'en', 'ne', 'lr', 'rl', 'll',
               'rr', 'pI', 'pQ', 'pU', 'pV']
    cpol_str = ['xy', 'yx', 'yy', 'xx', 'ee', 'nn', 'ne', 'en', 'rl', 'lr', 'll',
                'rr', 'pI', 'pQ', 'pU', 'pV']
    # fmt: on
    assert pol_str == utils.conj_pol(cpol_str)
    assert utils.conj_pol(pol_str) == cpol_str
    assert [pol_str, pol_nums] == utils.conj_pol([cpol_str, cpol_nums])

    # Test error with jones
    cjstr = ["Jxy", "Jyx", "Jyy", "Jxx", "Jrl", "Jlr", "Jll", "Jrr"]
    assert pytest.raises(KeyError, utils.conj_pol, cjstr)

    # Test invalid pol
    with pytest.raises(
        ValueError, match="Polarization not recognized, cannot be conjugated."
    ):
        utils.conj_pol(2.3)


def test_reorder_conj_pols_non_list():
    pytest.raises(ValueError, utils.pol.reorder_conj_pols, 4)


def test_reorder_conj_pols_strings():
    pols = ["xx", "xy", "yx"]
    corder = utils.pol.reorder_conj_pols(pols)
    assert np.array_equal(corder, [0, 2, 1])


def test_reorder_conj_pols_ints():
    pols = [-5, -7, -8]  # 'xx', 'xy', 'yx'
    corder = utils.pol.reorder_conj_pols(pols)
    assert np.array_equal(corder, [0, 2, 1])


def test_reorder_conj_pols_missing_conj():
    pols = ["xx", "xy"]  # Missing 'yx'
    pytest.raises(ValueError, utils.pol.reorder_conj_pols, pols)


def test_determine_pol_order_err():
    with pytest.raises(ValueError, match='order must be either "AIPS" or "CASA".'):
        utils.pol.determine_pol_order([], order="ABC")


@pytest.mark.parametrize(
    "pols,aips_order,casa_order",
    [
        [[-8, -7, -6, -5], [3, 2, 1, 0], [3, 1, 0, 2]],
        [[-5, -6, -7, -8], [0, 1, 2, 3], [0, 2, 3, 1]],
        [[1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 3]],
    ],
)
@pytest.mark.parametrize("order", ["CASA", "AIPS"])
def test_pol_order(pols, aips_order, casa_order, order):
    check = utils.pol.determine_pol_order(pols, order=order)

    if order == "CASA":
        assert all(check == casa_order)
    if order == "AIPS":
        assert all(check == aips_order)


def test_x_orientation_pol_map():
    assert utils.x_orientation_pol_map("north") == {"x": "n", "y": "e"}


@pytest.mark.parametrize(
    ("input_kwargs", "output"),
    [
        ({"feed_array": ["x"]}, [-5]),
        ({"feed_array": ["x", "y"]}, [-5, -6, -7, -8]),
        ({"feed_array": ["r", "l"]}, [-1, -2, -3, -4]),
        ({"feed_array": ["x", "y"], "include_cross_pols": False}, [-5, -6]),
        (
            {
                "feed_array": ["e", "n"],
                "x_orientation": "east",
                "return_feed_pol_order": True,
            },
            ([-5, -6, -7, -8], [(0, 0), (1, 1), (0, 1), (1, 0)]),
        ),
        (
            {
                "feed_array": ["n", "e"],
                "x_orientation": "north",
                "return_feed_pol_order": True,
            },
            ([-5, -6, -7, -8], [(0, 0), (1, 1), (0, 1), (1, 0)]),
        ),
        (
            {
                "feed_array": ["e", "n"],
                "x_orientation": "north",
                "return_feed_pol_order": True,
            },
            ([-6, -5, -8, -7], [(0, 0), (1, 1), (0, 1), (1, 0)]),
        ),
    ],
)
def test_convert_feeds_to_pols(input_kwargs, output):
    ret_val = utils.pol.convert_feeds_to_pols(**input_kwargs)
    if not isinstance(output, tuple):
        assert ret_val.tolist() == output
    else:
        assert ret_val[0].tolist() == output[0]
        assert ret_val[1] == output[1]

        dep_kwargs = copy.deepcopy(input_kwargs)
        dep_kwargs["calc_cross_pols"] = dep_kwargs.pop("include_cross_pols", True)
        del dep_kwargs["return_feed_pol_order"]
        with check_warnings(
            DeprecationWarning,
            match="This method (uvbeam._convert_feeds_to_pols) is deprecated in "
            "favor of utils.pol.convert_feeds_to_pols. This will become an error "
            "in version 3.3",
        ):
            dep_ret_val = _convert_feeds_to_pols(**dep_kwargs)
        assert dep_ret_val[0].tolist() == output[0]
        assert dep_ret_val[1] == output[1]


def test_convert_feeds_to_pols_errors():
    with pytest.raises(
        ValueError, match="feed_array contains 3 feeds. Only 1 or 2 feeds is supported."
    ):
        utils.pol.convert_feeds_to_pols(["x", "y", "z"])

    with pytest.raises(
        ValueError, match="feed_array contains 0 feeds. Only 1 or 2 feeds is supported."
    ):
        utils.pol.convert_feeds_to_pols([])


@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [{"feeds": ["l", "r"], "x_orientation": "xyz"}, "x_orientation not recognized"],
        [
            {"feeds": ["l", "r", "x"], "x_orientation": "n"},
            "feeds must be a list or tuple of length 1 or 2.",
        ],
        [
            {"feeds": ["u", "v"], "x_orientation": "n"},
            'feeds must contain only "x", "y", "l", and/or "r".',
        ],
    ],
)
def test_get_feeds_from_x_orientation_errors(kwargs, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        utils.pol.get_feeds_from_x_orientation(nants=1, **kwargs)


@pytest.mark.parametrize(
    "kwargs,exp_feed_array,exp_feed_angle,has_warn",
    [
        [{"x_orientation": "e", "nants": 1}, [["x", "y"]], [[np.pi / 2, 0]], True],
        [{"x_orientation": None, "nants": 2}, None, None, False],
        [{"x_orientation": "n", "nants": 1, "feeds": "x"}, [["x"]], [[0]], False],
        [
            {
                "x_orientation": "east",
                "nants": 2,
                "polarization_array": [-5, -6, -7, -8],
            },
            [["x", "y"], ["x", "y"]],
            [[np.pi / 2, 0], [np.pi / 2, 0]],
            False,
        ],
        [
            {
                "x_orientation": "east",
                "nants": 2,
                "flex_polarization_array": [-1, -2, -3, -4],
                "polarization_array": [0],
            },
            [["l", "r"], ["l", "r"]],
            [[0, 0], [0, 0]],
            False,
        ],
    ],
)
def test_get_feeds_from_x_orientation(kwargs, exp_feed_array, exp_feed_angle, has_warn):
    with check_warnings(
        UserWarning if has_warn else None,
        match=("Unknown polarization basis" if has_warn else None),
    ):
        Nfeeds, feed_array, feed_angle = utils.pol.get_feeds_from_x_orientation(
            **kwargs
        )

    if exp_feed_angle is None:
        assert Nfeeds is None
        assert feed_array is None
        assert feed_angle is None
    else:
        assert np.array_equal(feed_array, exp_feed_array)
        assert np.array_equal(feed_angle, exp_feed_angle)
        assert Nfeeds == feed_array.shape[1]
        assert Nfeeds == feed_angle.shape[1]
