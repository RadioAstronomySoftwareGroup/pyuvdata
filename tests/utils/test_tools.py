# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for helper utility functions."""

import pytest

from pyuvdata import utils
from pyuvdata.testing import check_warnings


@pytest.mark.parametrize(
    "filename1,filename2,answer",
    [
        (["foo.uvh5"], ["bar.uvh5"], ["foo.uvh5", "bar.uvh5"]),
        (["foo.uvh5", "bar.uvh5"], ["foo.uvh5"], ["foo.uvh5", "bar.uvh5"]),
        (["foo.uvh5"], None, ["foo.uvh5"]),
        (None, ["bar.uvh5"], ["bar.uvh5"]),
        (None, None, None),
    ],
)
def test_combine_filenames(filename1, filename2, answer):
    combined_filenames = utils.tools._combine_filenames(filename1, filename2)
    if answer is None:
        assert combined_filenames is answer
    else:
        # use sets to test equality so that order doesn't matter
        assert set(combined_filenames) == set(answer)

    return


def test_slicify():
    assert utils.tools.slicify(None) is None
    assert utils.tools.slicify(slice(None)) == slice(None)
    assert utils.tools.slicify([]) is None
    assert utils.tools.slicify([1, 2, 3]) == slice(1, 4, 1)
    assert utils.tools.slicify([1]) == slice(1, 2, 1)
    assert utils.tools.slicify([0, 2, 4]) == slice(0, 5, 2)
    assert utils.tools.slicify([0, 1, 2, 7]) == [0, 1, 2, 7]


@pytest.mark.parametrize(
    "obj1,obj2,union_result,interset_result,diff_result",
    [
        [[1, 2, 3], [3, 4, 5], [1, 2, 3, 4, 5], [3], [1, 2]],  # Partial overlap
        [[1, 2], [1, 2], [1, 2], [1, 2], []],  # Full overlap
        [[1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6], [], [1, 3, 5]],  # No overlap
        [[1, 2], None, [1, 2], [1, 2], [1, 2]],  # Nones
    ],
)
def test_sorted_unique_ops(obj1, obj2, union_result, interset_result, diff_result):
    assert utils.tools._sorted_unique_union(obj1, obj2) == union_result
    assert utils.tools._sorted_unique_intersection(obj1, obj2) == interset_result
    assert utils.tools._sorted_unique_difference(obj1, obj2) == diff_result


def test_deprecated_utils_import():
    with check_warnings(
        DeprecationWarning,
        match="The _check_histories function has moved, please import it from "
        "pyuvdata.utils.history. This warnings will become an error in version 3.2",
    ):
        utils._check_histories("foo", "foo")
