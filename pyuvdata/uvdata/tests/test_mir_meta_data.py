# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MirMetaData class and associated subclasses.

Performs a series of tests on the MirMetaData, and the associated subclasses, which are
the python-based handlers for Mir metadata products. Tests in this module are designed
to probe the functions of the individual class methods and attributes, and not
necessarily how they interact with each other (inside the `MirParser` class) or with
pyuvdata at large (via the `UVData` class).
"""
import os

import numpy as np
import pytest

from ... import tests as uvtest
from ..mir_meta_data import MirMetaData, MirMetaError


@pytest.fixture(scope="function")
def mir_in_data(mir_data_main):
    yield mir_data_main.in_data.copy()


@pytest.fixture(scope="function")
def mir_bl_data(mir_data_main):
    yield mir_data_main.bl_data.copy()


@pytest.fixture(scope="function")
def mir_sp_data(mir_data_main):
    yield mir_data_main.sp_data.copy()


@pytest.fixture(scope="function")
def mir_eng_data(mir_data_main):
    yield mir_data_main.eng_data.copy()


@pytest.fixture(scope="function")
def mir_codes_data(mir_data_main):
    yield mir_data_main.codes_data.copy()


@pytest.fixture(scope="function")
def mir_ac_data(mir_data_main):
    yield mir_data_main.ac_data.copy()


def test_mir_meta_init(mir_data):
    """
    Test that the initialization of MirMetaData objects behave as expected. This
    includes subtypes that are part of the MirParser class.
    """
    attr_list = list(mir_data._metadata_attrs.keys())
    # Drop "ac_data", since it's a synthetic table
    attr_list.remove("ac_data")

    for item in attr_list:
        # antpos_data is a little special, since it's a text file, so we can't use the
        # generic read function for MirMetaData here.
        attr = getattr(mir_data, item)
        if item != "antpos_data":
            # Read in the metadata w/ the generic read
            meta_attr = MirMetaData(
                attr._filetype,
                attr.dtype,
                attr._header_key,
                attr._binary_dtype,
                attr._pseudo_header_key,
                mir_data.filepath,
            )
            # Now make a placeholder subtype
            new_attr = type(attr)()

            # Plug in the __dict__, which _should_ transfer over all the writable attrs
            new_attr.__dict__ = meta_attr.__dict__
            assert attr == new_attr

        new_attr = type(attr)(mir_data.filepath)
        assert attr == new_attr


def test_mir_meta_iter(mir_data):
    """
    Test that MirMetaData objects iterate as expected, which is that they should yield
    the full group for each index position (similar to ndarray).
    """
    # Test the unflagged case
    in_data = mir_data.in_data
    for idx, item in enumerate(in_data):
        assert in_data._data[idx] == item

    # Now the flagged case
    in_data._mask[1::2] = False
    for idx, item in enumerate(in_data):
        assert in_data._data[2 * idx] == item


def test_mir_meta_copy(mir_in_data):
    """
    Verify that the copy operation of MirMetaData produces a duplicate dataset that is
    NOT a reference to the same data in memory.
    """
    other = mir_in_data.copy()
    for item in vars(other):
        this_attr = getattr(mir_in_data, item)
        other_attr = getattr(other, item)

        # Nones str, and tuple can be duplicates, since they're both immutable.
        if not (isinstance(this_attr, (str, tuple)) or this_attr is None):
            assert this_attr is not other_attr

        assert np.all(this_attr == other_attr)


@pytest.mark.parametrize(
    "comp_obj,err_msg", [[0, "Cannot compare MirInData with int."]]
)
def test_mir_meta_eq_errs(mir_in_data, comp_obj, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        mir_in_data.__eq__(comp_obj)


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("result", [True, False])
def test_mir_meta_eq(mir_sp_data, result, verbose):
    copy_data = mir_sp_data.copy()
    comp_func = copy_data.__eq__ if result else copy_data.__ne__

    assert comp_func(mir_sp_data, verbose=verbose) == result

    # Now muck a single field
    copy_data._data["corrchunk"][::2] = -1
    assert comp_func(mir_sp_data, verbose=verbose) != result
    assert (
        comp_func(mir_sp_data, verbose=verbose, ignore_params=["corrchunk"]) == result
    )

    # Muck the param list to make sure things work okay
    assert (
        comp_func(mir_sp_data, verbose=verbose, ignore_params=["mjd", "corrchunk"])
        == result
    )

    # Make sure that mask diffs are also handled correctly.
    copy_data._mask[::2] = False
    assert comp_func(mir_sp_data, verbose=verbose, use_mask=True) != result

    # Now flag both datasets at the offending position
    mir_sp_data._mask[::2] = False
    assert comp_func(mir_sp_data, verbose=verbose, use_mask=True) == result

    # Check that diff data sizes are handled correctly
    copy_data._data = np.concatenate((copy_data._data, copy_data._data))
    assert comp_func(mir_sp_data, verbose=verbose) != result

    # Test the masks w/ a None
    copy_data._mask = None
    assert comp_func(mir_sp_data, verbose=verbose, use_mask=True) != result


@pytest.mark.parametrize(
    "field,op,err_type,err_msg",
    [
        ["blah", "eq", MirMetaError, "select_field blah not found"],
        ["inhid", "blah", ValueError, "select_comp must be one of:"],
    ],
)
def test_mir_meta_where_errs(mir_in_data, field, op, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        mir_in_data.where(field, op, [1])


@pytest.mark.parametrize("return_keys", [True, False])
@pytest.mark.parametrize(
    "tup,goodidx,headkeys",
    [
        [("corrchunk", "eq", 0), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("corrchunk", "==", 0), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("blhid", "ne", 4), np.arange(0, 15), np.arange(0, 15) + 1],
        [("blhid", "!=", 4), np.arange(0, 15), np.arange(0, 15) + 1],
        [("inhid", "between", [0, 2]), np.arange(20), np.arange(20) + 1],
        [("fres", "outside", [-1, 1]), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("ipq", "lt", [10]), np.arange(20), np.arange(20) + 1],
        [("ipq", "<", [10]), np.arange(20), np.arange(20) + 1],
        [("nch", "le", 4), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("nch", "<=", 4), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("dataoff", "gt", 0.0), np.arange(1, 20), np.arange(1, 20) + 1],
        [("dataoff", ">", 0.0), np.arange(1, 20), np.arange(1, 20) + 1],
        [("wt", "ge", 0.0), np.arange(20), np.arange(20) + 1],
        [("wt", ">=", 0.0), np.arange(20), np.arange(20) + 1],
    ],
)
def test_mir_meta_where(mir_sp_data, tup, goodidx, headkeys, return_keys):
    where_arr = mir_sp_data.where(*tup, return_header_keys=return_keys)
    assert np.all(where_arr == headkeys) if return_keys else np.all(where_arr[goodidx])


def test_mir_meta_where_mask(mir_sp_data):
    where_arr = mir_sp_data.where("inhid", "eq", 1)
    assert np.all(where_arr)

    mask = np.ones_like(mir_sp_data._mask)
    mask[::2] = False
    where_arr = mir_sp_data.where("inhid", "eq", 1, mask=mask)
    assert not np.any(where_arr[::2])
    assert np.all(where_arr[1::2])


def test_mir_meta_where_multidim(mir_data):
    we_mask = mir_data.we_data.where("flags", "eq", 0)
    assert len(we_mask) == len(mir_data.we_data)
    assert we_mask.ndim == 1
    assert np.all(we_mask)


def test_mir_meta_where_pseudo_key(mir_data):
    eng_keys = mir_data.eng_data.where("tsys", "gt", 0, return_header_keys=True)
    assert isinstance(eng_keys, list)
    for key in eng_keys:
        assert isinstance(key, tuple)
        assert key in mir_data.eng_data._header_key_index_dict


@pytest.mark.parametrize(
    "kwargs,err_type,err_msg",
    [
        [{"index": 0, "where": 0}, ValueError, "Only one of index, header_key"],
        [{"index": 0, "use_mask": True}, ValueError, "Cannot set use_mask=True"],
        [{"where": 0}, ValueError, "Argument for where must be either"],
        [{"where": [(0,)]}, ValueError, "Argument for where must be either"],
        [{"where": ("a", "eq", -1)}, MirMetaError, "Argument for where has no match"],
    ],
)
def test_mir_meta_index_query_errs(mir_in_data, kwargs, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        mir_in_data._index_query(**kwargs)


@pytest.mark.parametrize(
    "arg,output",
    [
        [{"index": [2, 4, 6]}, [2, 4, 6]],
        [{"header_key": [2, 4, 6]}, [1, 3, 5]],
        [{"where": ("inhid", "eq", -1)}, [False] * 20],
        [{"where": ("inhid", "eq", -1), "and_where_args": False}, [True] * 20],
        [{"where": [("inhid", "eq", 1), ("a", "eq", 1)]}, [True] * 20],
    ],
)
def test_mir_meta_index_query(mir_sp_data, arg, output):
    assert np.all(output == mir_sp_data._index_query(**arg))


@pytest.mark.parametrize(
    "field_name,err_type,err_msg",
    [[[0, 1], ValueError, "field_name must either be a str or list of str."]],
)
def test_mir_meta_get_value_errs(mir_in_data, field_name, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        mir_in_data.get_value(field_name)


@pytest.mark.parametrize(
    "field_name,arg,output",
    [
        ["inhid", {}, [1] * 20],
        ["corrchunk", {"index": [0, 5, 10, 15]}, [0] * 4],
        ["blhid", {"where": ("blhid", "eq", 2)}, [2] * 5],
        [
            ["blhid", "sphid"],
            {"index": [1, 4], "return_tuples": False},
            [[1, 1], [2, 5]],
        ],
        [["igq", "sphid"], {"index": [0, 3], "return_tuples": True}, [(0, 1), (0, 4)]],
    ],
)
def test_mir_meta_get_value(mir_sp_data, field_name, arg, output):
    assert np.all(
        np.array(output) == np.array(mir_sp_data.get_value(field_name, **arg))
    )


@pytest.mark.parametrize(
    "field_name,msg",
    [
        ["dataoff", 'Values in "dataoff" are typically only used'],
        ["inhid", "Changing fields that tie to header keys can result in"],
        ["blhid", "Changing fields that tie to header keys can result in"],
        ["sphid", "Changing fields that tie to header keys can result in"],
    ],
)
def test_mir_meta_set_value_warns(mir_sp_data, field_name, msg):
    with uvtest.check_warnings(UserWarning, msg):
        mir_sp_data.set_value(field_name, 1)


@pytest.mark.parametrize(
    "field_name,arg,set_value,output",
    [
        ["iant1", {}, 9, [9] * 4],
        ["iant1", {}, [1, 2, 3, 4], [1, 2, 3, 4]],
        ["iant2", {"index": 2}, 9, [4, 4, 9, 4]],
        ["iant2", {"header_key": 4}, 9, [4, 4, 4, 9]],
        ["iant2", {"header_key": 4}, 9, [4, 4, 4, 9]],
        ["u", {"where": ("u", "ne", 0)}, 0, [0, 0, 0, 0]],
    ],
)
def test_mir_meta_set_value(mir_bl_data, field_name, arg, set_value, output):
    mir_bl_copy = mir_bl_data.copy()
    mir_bl_data.set_value(field_name, set_value, **arg)
    assert np.all(mir_bl_data._data[field_name] == output)
    assert np.all(
        mir_bl_data._stored_values[field_name] == mir_bl_copy._data[field_name]
    )


@pytest.mark.parametrize(
    "arg,output",
    [
        [{}, True],
        [{"where": ("u", "ne", 0)}, True],
        [{"index": [1, 3]}, [False, True, False, True]],
        [{"header_key": [3, 4, 3]}, [False, False, True, True]],  # Guilty spark
    ],
)
def test_mir_meta_generate_mask(mir_bl_data, arg, output):
    assert np.all(mir_bl_data._generate_mask(**arg) == output)


@pytest.mark.parametrize(
    "arg,output",
    [
        [{"reset": True}, True],
        [{"where": (("u", "eq", 0), ("u", "ne", 0))}, False],
        [
            {"where": (("u", "eq", 0), ("u", "ne", 0)), "and_where_args": False},
            [True, False, False, True],
        ],
        [{"where": ("u", "ne", 0), "and_mask": False}, True],
        [{"index": [1, 3], "reset": True}, [False, True, False, True]],
        [{"header_key": [3, 4, 3]}, [False, False, False, True]],
        [
            {"mask": [True, False, True, False], "use_mask": False},
            [True, False, False, False],
        ],
    ],
)
def test_mir_meta_set_mask(mir_bl_data, arg, output):
    # Set the mask a priori to give us something to compare with.
    mir_bl_data._mask[:] = [True, False, False, True]

    check = mir_bl_data.set_mask(**arg)
    assert np.all(mir_bl_data._mask == output)
    assert ("reset" in arg) or (
        check == np.any(mir_bl_data._mask != [True, False, False, True])
    )


@pytest.mark.parametrize(
    "arg,output",
    [
        [{}, [True, False, False, True]],
        [{"where": ("u", "ne", 0)}, [True, False, False, True]],
        [{"index": [1, 3]}, [False, True]],
        [{"header_key": [3, 4, 3]}, [False, True, False]],
    ],
)
def test_mir_meta_get_mask(mir_bl_data, arg, output):
    mir_bl_data._mask[:] = [True, False, False, True]
    assert np.all(mir_bl_data.get_mask(**arg) == output)


@pytest.mark.parametrize(
    "arg,output",
    [
        [{}, [1, 2, 3, 4]],
        [{"where": ("u", "ne", 0)}, [1, 2, 3, 4]],
        [{"index": [1, 3]}, [2, 4]],
        [{"index": [1, 3], "force_list": True}, [[2, 4]]],
    ],
)
def test_mir_meta_get_header_keys(mir_bl_data, arg, output):
    assert np.all(np.array(mir_bl_data.get_header_keys(**arg)) == np.array(output))


def test_mir_meta_get_header_pseudo_keys(mir_eng_data):
    assert mir_eng_data.get_header_keys() == [(1, 1), (4, 1)]


def test_mir_meta_set_header_key_index_dict(mir_sp_data):
    mir_sp_data._set_header_key_index_dict()
    for key, value in mir_sp_data._header_key_index_dict.items():
        assert key == (value + 1)

    mir_sp_data._data["sphid"] = np.flip(mir_sp_data._data["sphid"])
    mir_sp_data._set_header_key_index_dict()
    for key, value in mir_sp_data._header_key_index_dict.items():
        assert (20 - key) == value


def test_mir_meta_generate_new_header_keys_err(mir_in_data):
    with pytest.raises(ValueError, match="Both objects must be of the same type."):
        mir_in_data._generate_new_header_keys(0)


def test_mir_meta_generate_new_header_keys_noop(mir_eng_data):
    assert mir_eng_data._generate_new_header_keys(mir_eng_data) == {}


def test_mir_meta_generate_new_header_keys(mir_bl_data):
    update_dict = mir_bl_data._generate_new_header_keys(mir_bl_data)
    assert update_dict == {"blhid": {1: 5, 2: 6, 3: 7, 4: 8}}


def test_mir_meta_sort_by_header_key(mir_bl_data):
    mir_bl_copy = mir_bl_data.copy()
    mir_bl_data._sort_by_header_key()
    assert mir_bl_copy == mir_bl_data

    mir_bl_data._data["blhid"] = np.flip(mir_bl_data._data["blhid"])
    mir_bl_data._sort_by_header_key()
    assert mir_bl_copy != mir_bl_data
    assert mir_bl_copy._header_key_index_dict == mir_bl_data._header_key_index_dict
    assert np.all(mir_bl_copy._data["blhid"] == mir_bl_copy._data["blhid"])


@pytest.mark.parametrize(
    "fields,args,mask_data,comp_dict",
    [
        ["inhid", {}, True, {}],
        ["inhid", {}, None, {1: list(range(1, 11))}],
        ["inhid", {"use_mask": False}, True, {1: list(range(1, 21))}],
        ["inhid", {}, False, {1: list(range(1, 21))}],
        ["inhid", {"return_index": True}, False, {1: list(range(20))}],
        [
            ["inhid", "corrchunk"],
            {"return_index": True},
            False,
            {
                (1, 0): [0, 5, 10, 15],
                (1, 1): [1, 6, 11, 16],
                (1, 2): [2, 7, 12, 17],
                (1, 3): [3, 8, 13, 18],
                (1, 4): [4, 9, 14, 19],
            },
        ],
    ],
)
def test_mir_meta_group_by(mir_sp_data, fields, args, mask_data, comp_dict):
    if mask_data:
        mir_sp_data._mask[:] = False
    elif mask_data is None:
        mir_sp_data._mask[10:] = False

    group_dict = mir_sp_data.group_by(fields, **args)

    assert group_dict.keys() == comp_dict.keys()

    for key in group_dict:
        assert np.all(group_dict[key] == comp_dict[key])


def test_mir_meta_reset_values_errs(mir_in_data):
    with pytest.raises(ValueError, match="No stored values for field foo."):
        mir_in_data.reset_values("foo")


@pytest.mark.parametrize("list_args", [True, False])
def test_mir_meta_reset_values(mir_sp_data, list_args):
    mir_sp_copy = mir_sp_data.copy()
    mir_sp_data["fsky"] = 0
    mir_sp_data["vres"] = 0

    assert mir_sp_data != mir_sp_copy
    assert np.all(mir_sp_data["fsky"] != mir_sp_copy["fsky"])
    assert np.all(mir_sp_data["vres"] != mir_sp_copy["vres"])

    if list_args:
        for item in ["fsky", "vres"]:
            mir_sp_data.reset_values(item)
    else:
        mir_sp_data.reset_values()

    assert mir_sp_data == mir_sp_copy
    assert mir_sp_data._stored_values == {}


def test_mir_meta_reset(mir_sp_data):
    mir_sp_copy = mir_sp_data.copy()
    mir_sp_data["fsky"] = 0
    mir_sp_data["wt"] = 0
    mir_sp_data._mask[:] = False

    assert mir_sp_data != mir_sp_copy

    mir_sp_data.reset()
    assert mir_sp_data == mir_sp_copy


@pytest.mark.parametrize(
    "update_dict,err_type,err_msg",
    [
        [{1: {1: 1}}, ValueError, "update_dict must have keys that are type str"],
        [{"foo": {1: 2}}, ValueError, "Field group foo not found in this object."],
    ],
)
def test_mir_meta_update_fields_errs(mir_in_data, update_dict, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        mir_in_data._update_fields(update_dict, raise_err=True)


def test_mir_meta_update_fields(mir_bl_data):
    mir_bl_copy = mir_bl_data.copy()
    update_dict = {
        "foo": {1: 1},
        "inhid": {1: 10},
        "blhid": {1: 0},
        ("iant1", "iant2"): {(1, 4): (4, 1)},
    }

    mir_bl_data._update_fields(update_dict)
    assert mir_bl_data != mir_bl_copy
    assert np.all(mir_bl_data["inhid"] == [10, 10, 10, 10])
    assert np.all(mir_bl_data["blhid"] == [0, 2, 3, 4])
    assert np.all(mir_bl_data["iant1"] == [4, 4, 4, 4])
    assert np.all(mir_bl_data["iant2"] == [1, 1, 1, 1])


def test_mir_meta_write_errs(mir_sp_data, tmp_path):
    filepath = os.path.join(tmp_path, "meta_write_errs")
    mir_sp_data.write(filepath)
    mir_sp_data["gunnLO"] = 0.0

    with pytest.raises(FileExistsError, match="File already exists, must set over"):
        mir_sp_data.write(filepath)

    with pytest.raises(ValueError, match="Conflicting header keys detected"):
        mir_sp_data.write(filepath, append_data=True, check_index=True)


def test_mir_meta_write_append(mir_sp_data, tmp_path):
    filepath = os.path.join(tmp_path, "meta_write_append")
    new_obj = type(mir_sp_data)()

    mir_sp_data.write(filepath)
    # Test the no-op
    mir_sp_data.write(filepath, append_data=True, check_index=True)
    new_obj.read(filepath)
    assert mir_sp_data == new_obj

    # Now try writing two separate halves of the data one at a time.
    mir_sp_data._mask[::2] = False
    mir_sp_data.write(filepath, overwrite=True)
    mir_sp_data._mask[::2] = True
    mir_sp_data._mask[1::2] = False
    mir_sp_data.write(filepath, append_data=True, check_index=True)
    new_obj.read(filepath)
    assert mir_sp_data == new_obj


@pytest.mark.parametrize(
    "cmd,args,err_type,err_msg",
    [
        ["int_other", {}, ValueError, "Both objects must be of the same type."],
        ["", {"merge": True, "discard_flagged": True}, ValueError, "Cannot both merge"],
        ["muck_key", {"merge": True}, ValueError, "Cannot merge if header keys"],
        ["muck_data", {}, MirMetaError, "Cannot combine objects, as both contain"],
        ["", {"merge": False}, MirMetaError, "Cannot add objects together if merge="],
    ],
)
def test_mir_meta_add_check_errs(mir_sp_data, cmd, args, err_type, err_msg):
    if cmd == "int_other":
        other = 0
    else:
        other = mir_sp_data.copy()
        if cmd == "muck_key":
            other._header_key_index_dict[-1] = -1
        elif cmd == "muck_data":
            other["gunnLO"] = -1.0

    with pytest.raises(err_type, match=err_msg):
        mir_sp_data._add_check(other, **args)


@pytest.mark.parametrize(
    "cmd,args,comp_results",
    [
        [["noop"], {"merge": True}, [[], [0, 1, 2, 3], [], [True] * 4]],
        [["comb_mod"], {}, [[0], [1, 2, 3], [True], [True] * 3]],
        [["mask_mod"], {}, [[0, 1], [2, 3], [True] * 2, [True] * 2]],
        [["mask_mod"], {"overwrite": True}, [[], [0, 1, 2, 3], [], [True] * 4]],
        [["flip_mod"], {"overwrite": True}, [[], [0, 1, 2, 3], [], [True] * 4]],
    ],
)
def test_mir_meta_add_check_merge(mir_bl_data, args, cmd, comp_results):
    mir_bl_copy = mir_bl_data.copy()
    # Use this as a way to mark the copy as altered
    if "comb_mod" in cmd:
        mir_bl_copy._data["u"][0] = 0.0
        mir_bl_copy._mask[0] = False
        mir_bl_data._data["u"][2] = 0.0
        mir_bl_data._mask[2] = False
    elif "flip_mod" in cmd:
        mir_bl_data["u"] = 0.0
    elif "noop" not in cmd:
        mir_bl_copy["u"] = 0.0

    if "mask_mod" in cmd:
        mir_bl_copy._mask[0:2] = False
        mir_bl_data._mask[2:4] = False

    result_tuple = mir_bl_data._add_check(mir_bl_copy, **args)
    for item, jtem in zip(result_tuple, comp_results):
        assert np.array_equal(item, jtem)


@pytest.mark.parametrize(
    "cmd,comp_results",
    [
        [["partial"], [[0, 1, 2, 3], [2, 3], [True] * 4, [False] * 2]],
        [["partial", "flip"], [[2, 3], [0, 1, 2, 3], [False] * 2, [True] * 4]],
        [["full"], [[0, 1, 2, 3], [0, 1, 2, 3], [True] * 4, [False] * 4]],
        [["full", "flip"], [[0, 1, 2, 3], [0, 1, 2, 3], [False] * 4, [True] * 4]],
    ],
)
def test_mir_meta_add_check_concat(mir_bl_data, cmd, comp_results):
    mir_bl_copy = mir_bl_data.copy()
    # Use this as a way to mark the copy as altered
    if "partial" in cmd:
        mir_bl_copy._data["blhid"][2:4] = [7, 8]
    elif "full" in cmd:
        mir_bl_copy._data["blhid"][:] = [5, 6, 7, 8]

    mir_bl_copy._set_header_key_index_dict()
    mir_bl_copy._mask[:] = False

    if "flip" in cmd:
        result_tuple = mir_bl_copy._add_check(mir_bl_data)
    else:
        result_tuple = mir_bl_data._add_check(mir_bl_copy)

    for item, jtem in zip(result_tuple, comp_results):
        assert np.array_equal(item, jtem)


def test_mir_meta_add_errs(mir_in_data):
    with pytest.raises(ValueError, match="Both objects must be of the same type."):
        mir_in_data += 0


@pytest.mark.parametrize("this_none", [True, False])
@pytest.mark.parametrize("other_none", [True, False])
@pytest.mark.parametrize("method", ["__add__", "__iadd__"])
def test_mir_meta_add_none(mir_in_data, this_none, other_none, method):
    this = mir_in_data.copy(skip_data=this_none)
    other = mir_in_data.copy(skip_data=other_none)

    result = getattr(this, method)(other)

    if this_none and other_none:
        assert (this == result) and (other == result)
    else:
        assert mir_in_data == result


@pytest.mark.parametrize("method", ["__add__", "__iadd__"])
def test_mir_meta_add_concat(mir_bl_data, method):
    mir_bl_copy = mir_bl_data.copy()
    mir_bl_copy._data["blhid"] = [1, 3, 5, 7]
    mir_bl_copy._data["u"] = [15, 16, 17, 18]
    mir_bl_copy._mask[:] = [True] * 4
    mir_bl_copy._set_header_key_index_dict()

    mir_bl_data._data["blhid"] = [2, 4, 6, 8]
    mir_bl_data._data["u"] = [11, 12, 13, 14]
    mir_bl_data._mask[:] = [False] * 4
    mir_bl_data._set_header_key_index_dict()

    result = getattr(mir_bl_data, method)(mir_bl_copy)
    if method == "__iadd__":
        assert mir_bl_data is result

    assert np.all(result._data["u"] == np.arange(11, 19))
    assert np.all(~result._mask[:4])
    assert np.all(result._mask[4:])


def test_mir_sp_recalc_dataoff(mir_sp_data):
    dataoff_arr = mir_sp_data["dataoff"].copy()

    # Now update one mask position, a pseudo-cont record
    pseudo_rec_size = 18
    mir_sp_data._mask[0] = False
    mir_sp_data._recalc_dataoff()
    assert np.all(mir_sp_data["dataoff"] == (dataoff_arr[1:] - pseudo_rec_size))

    # Now flag all the pseudo-cont records
    mir_sp_data._mask[[0, 5, 10, 15]] = False
    mir_sp_data._recalc_dataoff()
    assert np.all(
        mir_sp_data["dataoff"]
        == np.concatenate(
            (
                dataoff_arr[1:5] - (1 * pseudo_rec_size),
                dataoff_arr[6:10] - (2 * pseudo_rec_size),
                dataoff_arr[11:15] - (3 * pseudo_rec_size),
                dataoff_arr[16:20] - (4 * pseudo_rec_size),
            )
        )
    )

    # Finally, ignore the mask when doing dataoff and make sure it returns correctly
    mir_sp_data._recalc_dataoff(use_mask=False)
    assert np.all(mir_sp_data._data["dataoff"] == dataoff_arr)


def test_mir_meta_get_record_size_info_errs():
    with pytest.raises(TypeError, match="Cannot use this method on objects other than"):
        MirMetaData(None, None, None)._get_record_size_info()


@pytest.mark.parametrize(
    "attr,val_size,rec_size_arr",
    [["sp_data", 2, ([18] + ([65538] * 4)) * 4], ["ac_data", 4, [65536] * 16]],
)
def test_mir_meta_get_record_size_info(mir_data, attr, val_size, rec_size_arr):
    comp_rec_size, comp_val = getattr(mir_data, attr)._get_record_size_info()
    assert val_size == comp_val
    assert np.all(comp_rec_size == rec_size_arr)


@pytest.mark.parametrize("reindex", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("mod_mask", [True, False])
def test_mir_meta_generate_recpos_dict(mir_sp_data, reindex, use_mask, mod_mask):
    if mod_mask:
        mir_sp_data._mask[::2] = False

    dataoff_arr = mir_sp_data.get_value("dataoff", use_mask=use_mask).astype(int) // 2
    rec_size_arr = (mir_sp_data.get_value("nch", use_mask=use_mask).astype(int) * 2) + 1

    if use_mask and mod_mask and reindex:
        dataoff_arr = np.cumsum(rec_size_arr) - rec_size_arr

    int_dict, sp_dict = mir_sp_data._generate_recpos_dict(
        use_mask=use_mask, reindex=reindex
    )
    assert int_dict == {
        1: {
            "inhid": 1,
            "record_size": 1048680 // (1 + (use_mask and mod_mask and reindex)),
            "record_start": 0,
        }
    }
    for key in int_dict:
        sp_dict = sp_dict[key]
        for value, dataoff, recsize in zip(sp_dict.values(), dataoff_arr, rec_size_arr):
            assert value["start_idx"] == dataoff
            assert value["end_idx"] == dataoff + recsize
            assert value["chan_avg"] == 1

    assert list(mir_sp_data.get_value("sphid", use_mask=use_mask)) == list(sp_dict)


def test_mir_codes_get_code_names(mir_codes_data):
    assert sorted(mir_codes_data.get_code_names()) == sorted(
        [
            "aq",
            "band",
            "blcd",
            "bq",
            "cocd",
            "cq",
            "dec",
            "filever",
            "gq",
            "ifc",
            "offtype",
            "oq",
            "pol",
            "pos",
            "pq",
            "project",
            "pstate",
            "ra",
            "rec",
            "ref_time",
            "sb",
            "source",
            "stype",
            "svtype",
            "taper",
            "tel1",
            "tel2",
            "tq",
            "trans",
            "ut",
            "vctype",
            "vrad",
        ]
    )


@pytest.mark.parametrize(
    "args,kwargs,err_type,err_msg",
    [
        [(0, 0, 0), {}, MirMetaError, "select_field must either be one of the native"],
        [("source", "gt", 0), {}, ValueError, 'select_comp must be "eq", "==", "ne"'],
    ],
)
def test_mir_codes_where_errs(mir_codes_data, args, kwargs, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        mir_codes_data.where(*args, **kwargs)


@pytest.mark.parametrize(
    "args,kwargs,output",
    [
        [("source", "eq", "3c84"), {}, [1]],
        [("pol", "ne", ["hh", "vv"]), {}, [2, 3]],
        [("tel1", "eq", [1, 5, 8]), {}, [1, 5, 8]],
        [
            ("vrad", "eq", "         0.0"),
            {"return_header_keys": False},
            (92 * [False]) + [True] + (6 * [False]),
        ],
    ],
)
def test_mir_codes_where(mir_codes_data, args, kwargs, output):
    assert np.all(output == mir_codes_data.where(*args, **kwargs))


def test_mir_codes_get_item_err(mir_codes_data):
    with pytest.raises(MirMetaError, match="foo does not match any code or field"):
        mir_codes_data["foo"]


@pytest.mark.parametrize(
    "vname,output",
    [
        ["filever", ["3"]],
        ["source", {1: "3c84", "3c84": 1}],
        ["sb", {"l": 0, "u": 1, 0: "l", 1: "u"}],
        ["ut", {"Jul 24 2020  4:34:39.00PM": 1, 1: "Jul 24 2020  4:34:39.00PM"}],
    ],
)
def test_mir_codes_get_item(mir_codes_data, vname, output):
    assert output == mir_codes_data[vname]


@pytest.mark.parametrize("name", ["v_name", "icode", "code", "ncode"])
def test_mir_codes_get_item_dtype(mir_codes_data, name):
    assert np.all(mir_codes_data[name] == mir_codes_data._data[name])


def test_mir_codes_generate_new_header_keys_errs_and_warns(mir_codes_data):
    # This _could_ be parameterized, although each test is so customized that
    # it's easier to code this as a single passthrough.
    with pytest.raises(ValueError, match="Both objects must be of the same type."):
        mir_codes_data._generate_new_header_keys(0)

    mir_codes_copy = mir_codes_data.copy()
    mir_codes_copy.set_value("code", "1", where=("v_name", "eq", "filever"))
    with pytest.raises(ValueError, match="The codes for filever in codes_read"):
        mir_codes_data._generate_new_header_keys(mir_codes_copy)

    mir_codes_copy.set_value("code", "3", where=("v_name", "eq", "filever"))

    mir_codes_copy.set_value("code", ["3", "4", "5"], where=("v_name", "eq", "aq"))
    with uvtest.check_warnings(UserWarning, "Codes for aq not in the recognized list"):
        check_dict = mir_codes_data._generate_new_header_keys(mir_codes_copy)

    assert list(check_dict) == [("v_name", "icode")]
    check_dict = check_dict[("v_name", "icode")]
    assert check_dict == {
        ("aq", 0): ("aq", 3),
        ("aq", 1): ("aq", 4),
        ("aq", 2): ("aq", 5),
    }


@pytest.mark.parametrize(
    "code_row,update_dict",
    [
        [("source", 1, "3c84", 0), {}],
        [
            ("source", 1, "3c279", 0),
            {
                "isource": {1: 2},
                ("v_name", "icode"): {
                    ("source", 1): ("source", 2),
                    ("stype", 1): ("stype", 2),
                    ("svtype", 1): ("svtype", 2),
                },
            },
        ],
        [
            ("stype", 1, "ephem", 0),
            {
                "isource": {1: 2},
                ("v_name", "icode"): {
                    ("source", 1): ("source", 2),
                    ("stype", 1): ("stype", 2),
                    ("svtype", 1): ("svtype", 2),
                },
            },
        ],
        [
            ("project", 2, "retune", 0),
            {"iproject": {2: 1}, ("v_name", "icode"): {("project", 2): ("project", 1)}},
        ],
        [("project", 2, "do_a_thing", 0), {}],
    ],
)
def test_mir_codes_generate_new_header_keys(mir_codes_data, code_row, update_dict):
    mir_codes_copy = mir_codes_data.copy()
    mir_codes_copy._data[mir_codes_data.where("v_name", "eq", code_row[0])] = code_row

    assert update_dict == mir_codes_copy._generate_new_header_keys(mir_codes_data)


def test_mir_acdata_read_errs(mir_data):
    with pytest.raises(AssertionError) as err:
        mir_data.ac_data.read(mir_data.filepath, nchunks=1)
    str(err.value).startswith("Could not determine auto-correlation record size.")


def test_mir_make_key_mask_cipher(mir_ac_data, mir_eng_data):
    assert np.array_equal(
        mir_eng_data._make_key_mask(mir_ac_data),
        mir_eng_data._make_key_mask(mir_ac_data, use_cipher=False),
    )


def test_mir_make_key_mask(mir_in_data, mir_bl_data, mir_sp_data):
    assert not mir_in_data._make_key_mask(mir_bl_data)
    assert not mir_in_data._make_key_mask(mir_sp_data)
    assert not mir_bl_data._make_key_mask(mir_sp_data)
    assert not mir_in_data._make_key_mask(mir_in_data)
    assert not mir_bl_data._make_key_mask(mir_bl_data)
    assert not mir_sp_data._make_key_mask(mir_sp_data)

    # Modify the mask for baselines
    mir_bl_data._mask[::2] = False

    # in_data mask should not change
    assert not mir_in_data._make_key_mask(mir_bl_data)

    # But sp_data mask should!
    assert mir_sp_data._make_key_mask(mir_bl_data, reverse=True)

    assert np.array_equal(mir_sp_data["sphid"], [6, 7, 8, 9, 10, 16, 17, 18, 19, 20])

    # Finally, set the single ingtegration as bad, and make sure it cascades down
    mir_in_data._mask[:] = False
    mir_bl_data._make_key_mask(mir_in_data, reverse=True)
    mir_sp_data._make_key_mask(mir_bl_data, reverse=True)

    assert not any(mir_in_data._mask)
    assert not any(mir_bl_data._mask)
    assert not any(mir_sp_data._mask)
