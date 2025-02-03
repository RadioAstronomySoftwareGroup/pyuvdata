# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for helper utility functions."""

import numpy as np
import pytest

from pyuvdata import utils
from pyuvdata.parameter import UVParameter
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


@pytest.mark.parametrize("strict", [True, False, None])
def test_strict_raise(strict):
    if strict:
        with pytest.raises(ValueError, match="This is a test"):
            utils.tools._strict_raise("This is a test", strict=strict)
    else:
        with check_warnings(None if strict is None else UserWarning, "This is a test"):
            utils.tools._strict_raise("This is a test", strict=strict)


@pytest.mark.parametrize(
    "inds,nrecs,exp_output,nwarn",
    [
        [[0, 1, 2], 3, [0, 1, 2], 0],
        [[0, 1, 2], 2, [0, 1], 1],
        [[-1, 0, 1, 2], 3, [0, 1, 2], 1],
        [[-1, 0, 1, 2, 3], 3, [0, 1, 2], 2],
        [[1], 3, [1], 0],
    ],
)
def test_eval_inds(inds, nrecs, exp_output, nwarn):
    with check_warnings(
        UserWarning if nwarn > 0 else None, ["inds contains indices that are"] * nwarn
    ):
        output = utils.tools._eval_inds(inds=inds, nrecs=nrecs, strict=False)
    assert all(exp_output == output)


@pytest.mark.parametrize("is_param", [True, False])
@pytest.mark.parametrize(
    "inp_arr,tols,exp_outcome",
    [
        [np.array([0, 0, 0, 0]), (0, 0), True],
        [[0, 0, 0, 0], None, True],
        [[0, 0, 0, 1], (0, 0), False],
        [[0, 0, 0, 1], None, False],
        [[0, 0, 0, 1], (1, 0), True],
    ],
)
def test_array_constant(inp_arr, is_param, tols, exp_outcome):
    if is_param:
        kwargs = {"value": inp_arr}
        if tols is not None:
            kwargs["tols"] = tols
        inp_arr = UVParameter("test", **kwargs)
    assert exp_outcome == utils.tools._test_array_constant(inp_arr, tols=tols)


@pytest.mark.parametrize("is_param", [True, False])
@pytest.mark.parametrize(
    "inp_arr,inp2_arr,tols,exp_outcome",
    [
        [np.array([0, 0, 0, 0]), [0, 0, 0, 0], (0, 0), True],
        [[1, 2, 3, 4], np.array([1, 1, 1, 1]), None, True],
        [[0, 0, 0, 1], [0, 0, 0, 0], (0, 0), False],
        [[0, 0, 0, 1], [0, 0, 0, 0], None, False],
        [[1, 2, 3, 4], [0, 0, 0, 0], (0, 1), True],
    ],
)
def test_array_consistent(inp_arr, inp2_arr, is_param, tols, exp_outcome):
    if is_param:
        kwargs = {"value": inp_arr}
        if tols is not None:
            kwargs["tols"] = tols
        inp_arr = UVParameter("test", **kwargs)
        inp2_arr = UVParameter("test2", value=inp2_arr)
    assert exp_outcome == utils.tools._test_array_consistent(
        inp_arr, inp2_arr, tols=tols
    )
