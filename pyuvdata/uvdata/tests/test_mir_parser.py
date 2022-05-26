# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MirParser class.

Performs a series of tests on the MirParser, which is the python-based reader for MIR
data in pyuvdata. Tests in this module are specific to the way that MIR is read into
python, not neccessarily how pyuvdata (by way of the UVData class) interacts with that
data.
"""
import numpy as np
import h5py
import pytest
import os

from pytest_cases import parametrize
from ..mir_parser import MirParser, MirMetaData, MirMetaError
from ... import tests as uvtest


@pytest.fixture(scope="module")
def compass_soln_file(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mir_parser", numbered=True)
    filename = os.path.join(tmp_path, "compass_soln.mat")
    with h5py.File(filename, "w") as file:
        # Set up some basic indexing for our one-baseline test file
        file["antArr"] = np.array([[1, 4]])
        file["ant1Arr"] = np.array([[1]])
        file["ant2Arr"] = np.array([[4]])
        file["rx1Arr"] = np.repeat([0, 0, 1, 1], 4).reshape(1, -1)
        file["rx2Arr"] = np.repeat([0, 0, 1, 1], 4).reshape(1, -1)
        file["sbArr"] = np.repeat([0, 1, 0, 1], 4).reshape(1, -1)
        file["winArr"] = np.tile([[1, 2, 3, 4]], 4)

        # Make a set of bp solns that are easy to recreate in the test (to verify
        # that we actually have the solutions that we expect).
        bp_soln = np.arange(16 * 16384) + (np.flip(np.arange(16 * 16384)) * 1j)

        file["bandpassArr"] = np.reshape(
            np.concatenate((bp_soln, np.conj(np.reciprocal(bp_soln)))),
            (2, 16, 16384),
        ).astype(np.complex64)

        # This number is pulled from the test mir_data object, in in_data["mjd"].
        file["mjdArr"] = np.array([[59054.69153811]])

        # Set up a picket fence of flags for the "normal" flagging. Note we use
        # uint8 here because of the compression scheme COMPASS uses.
        file["flagArr"] = np.full((1, 1, 16, 2048), 170, dtype=np.uint8)

        # Set up the wide flags so that the first half of the spectrum is flagged.
        file["wideFlagArr"] = np.tile(
            ((np.arange(2048) < 1024) * 255).reshape(1, 1, -1).astype(np.uint8),
            (1, 16, 1),
        )

    yield filename


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
            new_attr = MirMetaData(
                attr._filetype,
                attr.dtype,
                attr._header_key,
                attr._binary_dtype,
                attr._pseudo_header_key,
                mir_data.filepath,
            )
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
    "comp_obj,err_msg",
    [
        [0, "Both objects must be MirMetaData (sub-) types."],
        [MirMetaData(0, 0, 0), "Cannot compare MirInData with different dtypes."],
    ],
)
def test_mir_meta_eq_errs(mir_in_data, comp_obj, err_msg):
    with pytest.raises(ValueError) as err:
        mir_in_data.__eq__(comp_obj)
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("result", [True, False])
def test_mir_meta_eq(mir_sp_data, result, verbose):
    copy_data = mir_sp_data.copy()
    print(type(copy_data))
    comp_func = getattr(copy_data, "__eq__") if result else getattr(copy_data, "__ne__")

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
    assert comp_func(mir_sp_data, verbose=verbose, ignore_mask=False) != result

    # Now flag both datasets at the offending position
    mir_sp_data._mask[::2] = False
    assert comp_func(mir_sp_data, verbose=verbose, ignore_mask=False) == result

    # Check that diff data sizes are handled correctly
    copy_data._data = np.concatenate((copy_data._data, copy_data._data))
    assert comp_func(mir_sp_data, verbose=verbose) != result

    # Test the masks w/ a None
    copy_data._mask = None
    assert comp_func(mir_sp_data, verbose=verbose, ignore_mask=False) != result


@pytest.mark.parametrize(
    "field,op,err_type,err_msg",
    [
        ["blah", "eq", MirMetaError, "select_field blah not found"],
        ["inhid", "blah", ValueError, "select_comp must be one of"],
    ],
)
def test_mir_meta_where_errs(mir_in_data, field, op, err_type, err_msg):
    with pytest.raises(err_type) as err:
        mir_in_data.where(field, op, [1])
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize("return_keys", [True, False])
@pytest.mark.parametrize(
    "tup,goodidx,headkeys",
    [
        [("corrchunk", "eq", 0), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("blhid", "ne", 4), np.arange(0, 15), np.arange(0, 15) + 1],
        [("inhid", "btw", [0, 2]), np.arange(20), np.arange(20) + 1],
        [("fres", "out", [-1, 1]), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("ipq", "lt", [10]), np.arange(20), np.arange(20) + 1],
        [("nch", "le", 4), np.arange(0, 20, 5), np.arange(0, 20, 5) + 1],
        [("dataoff", "gt", 0.0), np.arange(1, 20), np.arange(1, 20) + 1],
        [("wt", "ge", 0.0), np.arange(20), np.arange(20) + 1],
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
    with pytest.raises(err_type) as err:
        mir_in_data._index_query(**kwargs)
    assert str(err.value).startswith(err_msg)


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
    [
        [[0, 1], ValueError, "field_name must either be a str or list of str."],
    ],
)
def test_mir_meta_get_value_errs(mir_in_data, field_name, err_type, err_msg):
    with pytest.raises(err_type) as err:
        mir_in_data.get_value(field_name)
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "field_name,arg,output",
    [
        ["inhid", {}, [1] * 20],
        ["corrchunk", {"index": [0, 5, 10, 15]}, [0] * 4],
        ["blhid", {"where": ("blhid", "eq", 2)}, [2] * 5],
        [["blhid", "sphid"], {"index": [1, 4]}, [[1, 1], [2, 5]]],
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
        [{"mask": [True, False, True, False]}, [True, False, False, False]],
    ],
)
def test_mir_meta_set_mask(mir_bl_data, arg, output):
    # Set the mask apriori to give us something to compare with.
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


def test_mir_meta_generate_header_key_index_dict(mir_sp_data):
    key_dict = mir_sp_data._generate_header_key_index_dict()
    for key, value in key_dict.items():
        assert key == (value + 1)

    mir_sp_data._data["sphid"] = np.flip(mir_sp_data._data["sphid"])
    key_dict = mir_sp_data._generate_header_key_index_dict()
    for key, value in key_dict.items():
        assert (20 - key) == value


def test_mir_meta_generate_new_header_keys_err(mir_in_data):
    with pytest.raises(ValueError) as err:
        mir_in_data._generate_new_header_keys(0)
    assert str(err.value).startswith("Both objects must be of the same type.")


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
    with pytest.raises(ValueError) as err:
        mir_in_data.reset_values("foo")
    assert str(err.value).startswith("No stored values for field foo.")


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
    with pytest.raises(err_type) as err:
        mir_in_data._update_fields(update_dict, raise_err=True)
    assert str(err.value).startswith(err_msg)


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


def test_mir_meta_tofile_errs(mir_sp_data, tmp_path):
    filepath = os.path.join(tmp_path, "meta_tofile_errs")
    mir_sp_data.tofile(filepath)
    mir_sp_data["gunnLO"] = 0.0

    with pytest.raises(FileExistsError) as err:
        mir_sp_data.tofile(filepath)
    assert str(err.value).startswith("File already exists, must set overwrite")

    with pytest.raises(ValueError) as err:
        mir_sp_data.tofile(filepath, append_data=True, check_index=True)
    assert str(err.value).startswith("Conflicting header keys detected")


def test_mir_meta_tofile_append(mir_sp_data, tmp_path):
    filepath = os.path.join(tmp_path, "meta_tofile_append")
    new_obj = type(mir_sp_data)()

    mir_sp_data.tofile(filepath)
    # Test the no-op
    mir_sp_data.tofile(filepath, append_data=True, check_index=True)
    new_obj.fromfile(filepath)
    assert mir_sp_data == new_obj

    # Now try writing two separate halves of the data one at a time.
    mir_sp_data._mask[::2] = False
    mir_sp_data.tofile(filepath, overwrite=True)
    mir_sp_data._mask[::2] = True
    mir_sp_data._mask[1::2] = False
    mir_sp_data.tofile(filepath, append_data=True, check_index=True)
    new_obj.fromfile(filepath)
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

    with pytest.raises(err_type) as err:
        mir_sp_data._add_check(other, **args)
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "cmd,args,comp_results",
    [
        [["noop"], {"merge": True}, [[], [0, 1, 2, 3], [], [True] * 4]],
        [["combmod"], {}, [[0], [1, 2, 3], [True], [True] * 3]],
        [["maskmod"], {}, [[0, 1], [2, 3], [True] * 2, [True] * 2]],
        [["maskmod"], {"overwrite": True}, [[], [0, 1, 2, 3], [], [True] * 4]],
        [["flipmod"], {"overwrite": True}, [[], [0, 1, 2, 3], [], [True] * 4]],
    ],
)
def test_mir_meta_add_check_merge(mir_bl_data, args, cmd, comp_results):
    mir_bl_copy = mir_bl_data.copy()
    # Use this as a way to mark the copy as altered
    if "combmod" in cmd:
        print("hi")
        mir_bl_copy._data["u"][0] = 0.0
        mir_bl_copy._mask[0] = False
        mir_bl_data._data["u"][2] = 0.0
        mir_bl_data._mask[2] = False
    elif "flipmod" in cmd:
        mir_bl_data["u"] = 0.0
    elif "noop" not in cmd:
        mir_bl_copy["u"] = 0.0

    if "maskmod" in cmd:
        mir_bl_copy._mask[0:2] = False
        mir_bl_data._mask[2:4] = False

    result_tuple = mir_bl_data._add_check(mir_bl_copy, **args)
    print(result_tuple)
    print(comp_results)
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

    mir_bl_copy._header_key_index_dict = mir_bl_copy._generate_header_key_index_dict()
    mir_bl_copy._mask[:] = False

    if "flip" in cmd:
        result_tuple = mir_bl_copy._add_check(mir_bl_data)
    else:
        result_tuple = mir_bl_data._add_check(mir_bl_copy)

    print(result_tuple)
    print(comp_results)

    for item, jtem in zip(result_tuple, comp_results):
        assert np.array_equal(item, jtem)


def test_mir_meta_add_errs(mir_in_data):
    with pytest.raises(ValueError) as err:
        mir_in_data += 0
    assert str(err.value).startswith("Both objects must be of the same type.")


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
    mir_bl_copy._data["u"] = [11, 13, 15, 17]
    mir_bl_copy._mask[:] = [True, True, False, True]
    mir_bl_copy._header_key_index_dict = mir_bl_copy._generate_header_key_index_dict()

    mir_bl_data._data["blhid"] = [2, 4, 6, 8]
    mir_bl_data._data["u"] = [12, 14, 16, 18]
    mir_bl_data._mask[:] = [True, False, True, True]
    mir_bl_data._header_key_index_dict = mir_bl_data._generate_header_key_index_dict()

    result = getattr(mir_bl_data, method)(mir_bl_copy)
    if method == "__iadd__":
        assert mir_bl_data is result

    assert np.all(result._data["u"] == np.arange(11, 19))
    assert np.all(result._mask[:3])
    assert np.all(~result._mask[3:-3])
    assert np.all(result._mask[-3:])


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

    # Finally, ignore the mask when doing dataof and make sure it returns whats expected
    mir_sp_data._recalc_dataoff(use_mask=False)
    assert np.all(mir_sp_data._data["dataoff"] == dataoff_arr)


@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("mod_mask", [True, False])
def test_mir_sp_generate_dataoff_dict(mir_sp_data, use_mask, mod_mask):
    if mod_mask:
        mir_sp_data._mask[::2] = False

    dataoff_arr = mir_sp_data.get_value("dataoff", use_mask=use_mask).astype(int) // 2
    rec_size_arr = (mir_sp_data.get_value("nch", use_mask=use_mask).astype(int) * 2) + 1
    int_dict, sp_dict = mir_sp_data._generate_dataoff_dict(use_mask=use_mask)
    assert int_dict == {
        1: {
            "inhid": 1,
            "record_size": 1048680 // (1 + (use_mask and mod_mask)),
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
        [("source", "gt", 0), {}, ValueError, 'select_comp must be "eq" or "ne" when'],
    ],
)
def test_mir_codes_where_errs(mir_codes_data, args, kwargs, err_type, err_msg):
    with pytest.raises(err_type) as err:
        mir_codes_data.where(*args, **kwargs)
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "args,kwargs,output",
    [
        [("source", "eq", "3c84"), {}, [1]],
        [("pol", "ne", ["hh", "vv"]), {}, [2, 3]],
        [("tel1", "eq", [1, 5, 8]), {}, [1, 5, 8]],
        [
            ("vrad", "eq", "         0.0"),
            {"return_header_keys": False},
            (98 * [False]) + [True],
        ],
    ],
)
def test_mir_codes_where(mir_codes_data, args, kwargs, output):
    assert np.all(output == mir_codes_data.where(*args, **kwargs))


def test_mir_codes_get_item_err(mir_codes_data):
    with pytest.raises(MirMetaError) as err:
        mir_codes_data["foo"]
    assert str(err.value).startswith("foo does not match any code or field")


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
    with pytest.raises(ValueError) as err:
        mir_codes_data._generate_new_header_keys(0)
    assert str(err.value).startswith("Both objects must be of the same type.")

    mir_codes_copy = mir_codes_data.copy()
    mir_codes_copy.set_value("code", "1", where=("v_name", "eq", "filever"))
    with pytest.raises(ValueError) as err:
        mir_codes_data._generate_new_header_keys(mir_codes_copy)
    assert str(err.value).startswith("The codes for filever in codes_read")
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
        [("project", 2, "doathing", 0), {}],
    ],
)
def test_mir_codes_generate_new_header_keys(mir_codes_data, code_row, update_dict):
    mir_codes_copy = mir_codes_data.copy()
    mir_codes_copy._data[mir_codes_data.where("v_name", "eq", code_row[0])] = code_row

    assert update_dict == mir_codes_copy._generate_new_header_keys(mir_codes_data)


def test_mir_ac_fromfile_errs(mir_data):
    with pytest.raises(IndexError) as err:
        mir_data.ac_data.fromfile(mir_data.filepath, nchunks=-1)
    assert str(err.value).startswith("Could not determine auto-correlation record")


def test_mir_parser_index_uniqueness(mir_data):
    """
    Mir index uniqueness check

    Make sure that there are no duplicate indicies for things that are primary keys
    for the various table-like structures that are used in MIR
    """
    inhid_list = mir_data.in_data["inhid"]
    assert np.all(np.unique(inhid_list) == sorted(inhid_list))

    blhid_list = mir_data.bl_data["blhid"]
    assert np.all(np.unique(blhid_list) == sorted(blhid_list))

    sphid_list = mir_data.sp_data["sphid"]
    assert np.all(np.unique(sphid_list) == sorted(sphid_list))


def test_mir_parser_index_valid(mir_data):
    """
    Mir index validity check

    Make sure that all indexes are non-negative
    """
    assert np.all(mir_data.in_data["inhid"] >= 0)

    assert np.all(mir_data.bl_data["blhid"] >= 0)

    assert np.all(mir_data.sp_data["sphid"] >= 0)


def test_mir_parser_index_linked(mir_data):
    """
    Mir index link check

    Make sure that all referenced indicies have matching pairs in their parent tables
    """
    inhid_set = set(np.unique(mir_data.in_data["inhid"]))

    # Should not exist is has_auto=False
    # See `mir_data_object` above.
    if mir_data.ac_data is not None:
        assert set(np.unique(mir_data.ac_data["inhid"])).issubset(inhid_set)
    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto

    assert set(np.unique(mir_data.bl_data["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_data["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_data["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.sp_data["inhid"])).issubset(inhid_set)

    blhid_set = set(np.unique(mir_data.bl_data["blhid"]))

    assert set(np.unique(mir_data.sp_data["blhid"])).issubset(blhid_set)


def test_mir_parser_unload_data(mir_data):
    """
    Check that the unload_data function works as expected
    """
    # Spoof for just this test
    mir_data.raw_data = mir_data.vis_data

    attr_list = ["vis_data", "raw_data", "auto_data"]

    for attr in attr_list:
        assert getattr(mir_data, attr) is not None

    mir_data.unload_data()

    for attr in attr_list:
        assert getattr(mir_data, attr) is None


@pytest.mark.parametrize(
    "attr",
    [
        "antpos_data",
        "bl_data",
        "codes_data",
        "eng_data",
        "in_data",
        "sp_data",
        "we_data",
        "ac_data",
    ],
)
def test_mir_write_item(mir_data, attr, tmp_path):
    """
    Mir write tester.

    Test writing out individual components of the metadata of a MIR dataset.
    """
    filepath = os.path.join(tmp_path, "test_write%s" % attr)
    orig_attr = getattr(mir_data, attr)
    orig_attr.tofile(filepath)
    check_attr = orig_attr.copy(skip_data=True)
    check_attr.fromfile(filepath)
    assert orig_attr == check_attr


def test_mir_write_vis_data_err(mir_data, tmp_path):
    mir_data.unload_data()
    with pytest.raises(ValueError) as err:
        mir_data.write_vis_data(tmp_path)
    assert str(err.value).startswith("Cannot write data if not already loaded.")


@pytest.mark.parametrize(
    "winsel,is_eq",
    [
        [None, True],
        [list(range(8)), True],
        [list(range(4)), False],
        [[1], False],
    ],
)
def test_mir_read_auto_data(mir_data, winsel, is_eq):
    auto_data1 = mir_data.read_auto_data()
    auto_data2 = mir_data.read_auto_data(winsel=winsel)

    assert auto_data1.keys() == auto_data2.keys()

    for key in auto_data1:
        assert np.array_equal(auto_data1[key], auto_data2[key], equal_nan=True) == is_eq


def test_mir_raw_data(mir_data, tmp_path):
    """
    Test reading and writing of raw data.
    """
    filepath = os.path.join(tmp_path, "test_write_raw")
    mir_data.load_data(load_raw=True)

    mir_data.write_vis_data(filepath)
    # Sub out the file we need to read from
    mir_data._file_dict = {filepath: item for item in mir_data._file_dict.values()}
    raw_data = mir_data.read_vis_data(return_vis=False)

    assert raw_data.keys() == mir_data.raw_data.keys()

    for key in raw_data.keys():
        for subkey in ["raw_data", "scale_fac"]:
            assert np.array_equal(raw_data[key][subkey], mir_data.raw_data[key][subkey])


@pytest.mark.parametrize("data_type", ["none", "raw", "vis", "load"])
def test_mir_write_full(mir_data, tmp_path, data_type):
    """
    Mir write dataset tester.

    Make sure we can round-trip a MIR dataset correctly.
    """
    # We want to clear our the auto data here, since we can't _yet_ write that out
    mir_data.unload_data()
    if data_type in ["vis", "raw"]:
        mir_data.load_data(load_vis=(data_type == "vis"), apply_tsys=False)

    mir_data._clear_auto()

    # Write out our test dataset
    filepath = os.path.join(tmp_path, "test_write_full_%s.mir" % data_type)

    with uvtest.check_warnings(
        None if (data_type != "none") else UserWarning,
        None if (data_type != "none") else "No data loaded, writing metadata only",
    ):
        mir_data.tofile(filepath, load_data=(data_type == "load"))

    # Read in test dataset.
    mir_copy = MirParser(filepath)
    if data_type != "none":
        mir_copy.load_data(load_raw=(data_type in ["raw", "load"]), apply_tsys=False)

    # The objects won't be equal off the bat - a couple of things to handle first.
    assert mir_data != mir_copy

    # _file_dict has the filepath as a key, so we handle this in a special way.
    assert list(mir_data._file_dict.values()) == list(mir_copy._file_dict.values())
    mir_data._file_dict = mir_copy._file_dict = None

    # Filename obviously _should_ be different...
    assert mir_data.filepath != mir_copy.filepath
    mir_data.filepath = mir_copy.filepath = None

    # Check for final equality with the above exceptions handled.
    assert mir_data == mir_copy


def test_compass_flag_sphid_apply(mir_data, compass_soln_file):
    """
    Test COMPASS per-sphid flagging.

    Test that applying COMPASS flags on a per-sphid basis works as expected.
    """
    # Unflag previously flagged data
    for entry in mir_data.vis_data.values():
        entry["vis_flags"][:] = False

    compass_solns = mir_data._read_compass_solns(compass_soln_file)
    mir_data._apply_compass_solns(compass_solns, apply_bp=False, apply_flags=True)
    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data.get_value("corrchunk", header_key=key) != 0:
            assert np.all(entry["vis_flags"][1::2])
            assert not np.any(entry["vis_flags"][::2])


def test_compass_flag_wide_apply(mir_data, compass_soln_file):
    """
    Test COMPASS wide flagging.

    Test that applying COMPASS flags on a per-baseline (all time) basis works correctly.
    """
    # Make sure that apriori flags are preserved
    for entry in mir_data.vis_data.values():
        entry["vis_flags"][:] = False
        entry["vis_flags"][-1] = True

    mir_data.in_data["mjd"] += 1
    with uvtest.check_warnings(
        UserWarning, "No metadata from COMPASS matches that in this data set."
    ):
        compass_solns = mir_data._read_compass_solns(compass_soln_file)

    mir_data._apply_compass_solns(compass_solns, apply_bp=False, apply_flags=True)

    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data.get_value("corrchunk", header_key=key) != 0:
            assert np.all(entry["vis_flags"][:8192])
            assert not np.any(entry["vis_flags"][8192:-1])
            assert np.all(entry["vis_flags"][-1])


@pytest.mark.parametrize("muck_solns", ["none", "some", "all"])
def test_compass_bp_apply(mir_data, compass_soln_file, muck_solns):
    """
    Test COMPASS bandpass calibraiton.

    Test that applying COMPASS bandpass solutions works correctly.
    """
    tempval = np.complex64(1 + 1j)
    for entry in mir_data.vis_data.values():
        entry["vis_data"][:] = tempval
        entry["vis_flags"][:] = False

    if muck_solns != "none":
        mir_data.bl_data["iant1"] += 1
        if muck_solns == "all":
            mir_data.bl_data["iant2"] += 1

    with uvtest.check_warnings(
        None if (muck_solns == "none") else UserWarning,
        None if (muck_solns == "none") else "No metadata from COMPASS matches",
    ):
        compass_solns = mir_data._read_compass_solns(compass_soln_file)

    mir_data._apply_compass_solns(compass_solns, apply_bp=True, apply_flags=False)

    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data.get_value("corrchunk", header_key=key) != 0:
            # If muck_solns is not some, then all the values should agree with our
            # temp value above, otherwise none should
            assert (muck_solns != "some") == np.allclose(entry["vis_data"], tempval)
            assert (muck_solns != "none") == np.all(entry["vis_flags"])


def test_compass_error(mir_data, compass_soln_file):
    """
    Test COMPASS-related errors.

    Verify that known error conditions trigger expected errors.
    """
    mir_data.unload_data()

    compass_solns = mir_data._read_compass_solns(compass_soln_file)

    with pytest.raises(ValueError) as err:
        mir_data._apply_compass_solns(compass_solns)

    assert str(err.value).startswith("Visibility data must be loaded")


@pytest.mark.parametrize(
    "field,comp,value,vis_keys",
    [
        ["mjd", "btw", [60000.0, 50000.0], np.arange(1, 21)],
        ["source", "ne", "nosourcehere", np.arange(1, 21)],
        ["ant", "eq", 4, np.arange(1, 21)],
        ["ant1", "ne", 8, np.arange(1, 21)],
        ["ant1rx", "eq", 0, [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]],
        ["corrchunk", "ne", [1, 2, 3, 4], np.arange(1, 21, 5)],
        ["N", "lt", 0.0, []],
    ],
)
def test_select(mir_data, field, comp, value, vis_keys):
    """Verify that select throws warnings as expected."""
    mir_data.select(where=(field, comp, value))

    # Confirm that we have all the indexes we should internally
    assert mir_data._check_data_index()

    # Cross-reference with the list we provide to be sure we have everything.
    assert np.all(np.isin(list(mir_data.vis_data), vis_keys))


def test_select_reset(mir_data):
    """Verify that running reset with select returns all entries as expected."""
    mir_copy = mir_data.copy()

    # Select based on something that should not exist.
    mir_data.select(where=("mjd", "eq", 0.0))
    assert len(mir_data.vis_data) == 0

    # Now run reset
    mir_data.select(reset=True, update_data=True)
    assert mir_data == mir_copy


def test_eq_errs(mir_data):
    """Verify that the __eq__ method throws appropriate errors."""
    with pytest.raises(ValueError) as err:
        mir_data.__eq__(0)
    assert str(err.value).startswith("Cannot compare MirParser with int.")


@pytest.mark.parametrize(
    "metadata_only,mod_attr,mod_val,exp_state",
    [
        [False, "auto_data", {}, False],
        [False, "auto_data", {1: np.zeros(4), 2: np.zeros(4)}, False],
        [True, "auto_data", {1: np.zeros(4), 2: np.zeros(4)}, True],
        [
            False,
            "vis_data",
            {
                idx: {
                    "vis_data": np.ones(2, dtype=np.complex64),
                    "vis_flags": np.ones(4, dtype=bool),
                }
                for idx in range(1, 21)
            },
            False,
        ],
        [False, "in_data", np.array([1, 2, 3, 4]), False],
        [True, "in_data", np.array([1, 2, 3, 4]), False],
        [True, "abc", "def", False],
        [False, "abc", "def", False],
        [False, "_has_auto", True, True],
        [False, "_has_auto", False, False],
        [False, "_has_auto", None, False],
        [True, "zero_data", None, True],
        [False, "zero_data", None, False],
        [False, "unload_data", None, True],
    ],
)
@pytest.mark.parametrize("flip", [False, True])
def test_eq(mir_data, metadata_only, mod_attr, mod_val, exp_state, flip):
    """Verify that __eq__ works as expected"""
    mir_copy = mir_data.copy()

    target_obj = mir_copy if flip else mir_data
    if "zero_data" == mod_attr:
        for attr in ["vis_data", "auto_data"]:
            for key in getattr(target_obj, attr).keys():
                if isinstance(getattr(target_obj, attr)[key], dict):
                    for subkey in getattr(target_obj, attr)[key].keys():
                        if subkey == "scale_fac":
                            getattr(target_obj, attr)[key][subkey] = 0
                        else:
                            getattr(target_obj, attr)[key][subkey][:] = 0
                else:
                    getattr(target_obj, attr)[key][:] = 0
    elif "unload_data" == mod_attr:
        mir_data.unload_data()
        mir_copy.unload_data()
    else:
        setattr(target_obj, mod_attr, mod_val)

    assert mir_data.__eq__(mir_copy, metadata_only=metadata_only) == exp_state

    assert mir_data.__ne__(mir_copy, metadata_only=metadata_only) != exp_state


def test_scan_int_start_errs(mir_data):
    """Verify scan_int_start throws errors when expected."""
    with pytest.raises(ValueError) as err:
        mir_data.scan_int_start(mir_data.filepath, allowed_inhid=[-1])
    assert str(err.value).startswith("Index value inhid in sch_read does not match")


def test_scan_int_start(mir_data):
    """Verify that we can correctly scan integration starting periods."""
    true_dict = {1: {"inhid": 1, "record_size": 1048680, "record_start": 0}}
    assert true_dict == mir_data.scan_int_start(mir_data.filepath, allowed_inhid=[1])


@pytest.mark.parametrize(
    "filepath,int_start_dict,err_type,err_msg",
    [
        [["a"], None, ValueError, "Must either set both or neither of filepath and"],
        [None, ["a"], ValueError, "Must either set both or neither of filepath and"],
        [["a", "b"], ["c"], ValueError, "Both filepath and int_start_dict must"],
    ],
)
def test_fix_int_start_errs(mir_data, filepath, int_start_dict, err_type, err_msg):
    """Conirm that fix_int_start throws errors as expected."""
    with pytest.raises(err_type) as err:
        mir_data.fix_int_start(filepath, int_start_dict)
    assert str(err.value).startswith(err_msg)


def test_fix_int_start(mir_data):
    """Verify that we can fix a "bad" integration start record."""
    bad_dict = {
        mir_data.filepath: {2: {"inhid": 1, "record_size": 120, "record_start": 120}}
    }
    good_dict = {
        mir_data.filepath: {2: {"inhid": 1, "record_size": 1048680, "record_start": 0}}
    }
    # Muck with the records so that the inhids don't match that on disk.
    mir_data.sp_data._data["inhid"][:] = 2
    mir_data.bl_data._data["inhid"][:] = 2
    mir_data.in_data._data["inhid"][:] = 2
    mir_data.sp_data._data["nch"][:] = 1
    mir_data._sp_dict[2] = mir_data._sp_dict.pop(1)

    # Plug in the bad dict
    mir_data._file_dict = bad_dict
    with pytest.raises(ValueError) as err:
        mir_data.read_vis_data(return_vis=False)
    assert str(err.value).startswith("Values in int_start_dict do not match")

    # This should _hopefully_ generate the good dict
    mir_data.fix_int_start()

    assert good_dict == mir_data._file_dict

    # Attempt to load the data
    _ = mir_data.read_vis_data(return_vis=False)

    # Make sure that things work if we don't inherit stuff from object
    check_dict = mir_data.fix_int_start([mir_data.filepath], list(bad_dict.values()))

    assert good_dict == check_dict


def test_read_packdata_err(mir_data):
    with pytest.raises(ValueError) as err:
        mir_data.read_packdata(mir_data._file_dict, [1, 2])
    assert str(err.value).startswith("inhid_arr contains keys not found in file_dict.")


def test_read_packdata_mmap(mir_data):
    """Test that reading in vis data with mmap works just as well as np.fromfile"""
    mmap_data = mir_data.read_packdata(
        mir_data._file_dict, mir_data.in_data["inhid"], use_mmap=True
    )

    reg_data = mir_data.read_packdata(
        mir_data._file_dict, mir_data.in_data["inhid"], use_mmap=False
    )

    assert mmap_data.keys() == reg_data.keys()
    for key in mmap_data.keys():
        assert np.array_equal(mmap_data[key], reg_data[key])


def test_read_packdata_make_packdata(mir_data):
    """Verify that making packdata produces the same result as reading packdata"""
    mir_data.load_data(load_raw=True)

    read_data = mir_data.read_packdata(mir_data._file_dict, mir_data.in_data["inhid"])

    make_data = mir_data.make_packdata(
        list(mir_data._file_dict.values())[0], mir_data._sp_dict, mir_data.raw_data
    )

    assert read_data.keys() == make_data.keys()
    for key in read_data.keys():
        assert np.array_equal(read_data[key], make_data[key])


def test_apply_tsys_errs(mir_data):
    """
    Test that apply_tsys throws errors as expected.

    Note that we test these errors in sequence since it's a lot more efficient to do
    these operations on the same object one after another.
    """
    with pytest.raises(ValueError) as err:
        mir_data.apply_tsys()
    assert str(err.value).startswith(
        "Cannot apply tsys again if it has been applied already."
    )

    mir_data.apply_tsys(invert=True)
    with pytest.raises(ValueError) as err:
        mir_data.apply_tsys(invert=True)
    assert str(err.value).startswith(
        "Cannot undo tsys application if it was never applied."
    )

    mir_data.unload_data()
    with pytest.raises(ValueError) as err:
        mir_data.apply_tsys(invert=True)
    assert str(err.value).startswith(
        "Must call load_data first before applying tsys normalization."
    )


def test_apply_tsys_warn(mir_data):
    """Verify that apply_tsys throws warnings when tsys values aren't found."""
    with uvtest.check_warnings(UserWarning, "Changing fields that tie to header keys"):
        mir_data.eng_data["antennaNumber"] = -1

    mir_data._tsys_applied = False

    with uvtest.check_warnings(
        UserWarning,
        [
            ("No tsys for blhid %i found (1-4 baseline, inhid 1)." % idx)
            for idx in range(1, 5)
        ],
    ):
        mir_data.apply_tsys()

    assert np.all(
        [np.all(data_dict["vis_flags"]) for data_dict in mir_data.vis_data.values()]
    )


def test_apply_tsys(mir_data):
    """Test that apply_tsys works on vis_data as expected."""
    mir_copy = mir_data.copy()
    # Calculate the scaling factors directly. The factor of 2 comes from DSB -> SSB
    rxa_norm = mir_data.jypk * 2 * (np.prod(mir_data.eng_data["tsys"]) ** 0.5)
    rxb_norm = mir_data.jypk * 2 * (np.prod(mir_data.eng_data["tsys_rx2"]) ** 0.5)
    # The first 5 records should be rxa, and 5 rxb, then 5 rxa, then 5 rxb
    norm_list = np.concatenate(
        (
            np.ones(5) * rxa_norm,
            np.ones(5) * rxb_norm,
            np.ones(5) * rxa_norm,
            np.ones(5) * rxb_norm,
        )
    )

    mir_data.unload_data()
    mir_data.load_data(load_vis=True, apply_tsys=False)
    mir_copy.unload_data()
    mir_copy.load_data(load_vis=True, apply_tsys=True)
    for key, norm_fac in zip(mir_data.vis_data.keys(), norm_list):
        assert np.allclose(
            norm_fac * mir_data.vis_data[key]["vis_data"],
            mir_copy.vis_data[key]["vis_data"],
        )
        assert np.allclose(
            mir_data.vis_data[key]["vis_flags"], mir_copy.vis_data[key]["vis_flags"]
        )

    mir_copy.apply_tsys(invert=True)
    for key, norm_fac in zip(mir_data.vis_data.keys(), norm_list):
        assert np.allclose(
            mir_data.vis_data[key]["vis_data"], mir_copy.vis_data[key]["vis_data"]
        )
        assert np.allclose(
            mir_data.vis_data[key]["vis_flags"], mir_copy.vis_data[key]["vis_flags"]
        )


def test_apply_flags_err(mir_data):
    mir_data.unload_data()
    with pytest.raises(ValueError) as err:
        mir_data.apply_flags()
    assert str(err.value).startswith("Cannot apply flags if vis_data are not loaded.")


@pytest.mark.parametrize("sphid_arr", [[1], list(range(1, 21)), [10, 15]])
def test_apply_flags(mir_data, sphid_arr):
    mir_data.sp_data.set_value("flags", 1, header_key=sphid_arr)
    mir_data.apply_flags()
    for key, value in mir_data.vis_data.items():
        assert np.all(value["vis_flags"]) == (key in sphid_arr)


def test_check_data_index(mir_data):
    """Verify that check_data_index returns True/False as expected."""
    # Spoof this for the sake of this test
    mir_data.raw_data = mir_data.vis_data

    assert mir_data._check_data_index()

    # Now muck with the records so that this becomes False
    for item in ["sp_data", "ac_data"]:
        getattr(mir_data, item)._data[0] = -1
        assert not mir_data._check_data_index()
        getattr(mir_data, item)._data[0] = 1
        assert mir_data._check_data_index()

    for item in ["vis_data", "raw_data", "auto_data"]:
        getattr(mir_data, item).update({-1: None})
        assert not mir_data._check_data_index()
        del getattr(mir_data, item)[-1]
        assert mir_data._check_data_index()


@pytest.mark.parametrize("select_auto", [True, False])
@pytest.mark.parametrize("select_vis", [True, False])
@pytest.mark.parametrize("select_raw", [True, False])
def test_downselect_data(mir_data, select_vis, select_raw, select_auto):
    if select_raw:
        # Create the raw data in case we need it.
        mir_data.raw_data = mir_data.convert_vis_to_raw(mir_data.vis_data)
        if not select_vis:
            # Unload this if we don't need it
            mir_data.vis_data = None

    mir_copy = mir_data.copy()

    # Manually downselect the data that we need.
    if select_vis or select_raw:
        mir_data.sp_data._mask[1::2] = False
    if select_auto:
        mir_data.ac_data._mask[1::2] = False

    mir_data._downselect_data(
        select_vis=select_vis, select_raw=select_raw, select_auto=select_auto
    )

    if select_vis or select_auto or select_raw:
        assert mir_data != mir_copy
    else:
        assert mir_data == mir_copy

    assert mir_data._check_data_index()

    # If we downselected, make sure we plug back in the original data.
    if select_vis or select_raw:
        mir_data.sp_data._mask[:] = True
    if select_auto:
        mir_data.ac_data._mask[:] = True

    # Make sure that the metadata all look good.
    assert mir_data.__eq__(mir_copy, metadata_only=True)

    if select_vis or select_auto or select_raw:
        with pytest.raises(MirMetaError) as err:
            mir_data._downselect_data(
                select_vis=select_vis, select_raw=select_raw, select_auto=select_auto
            )
        assert str(err.value).startswith("Missing spectral records in data attributes")

        # Any data attributes we wiped out, manually downselect the records in the
        # copy to make sure that everything agrees as we expect.
        if select_raw:
            mir_copy.raw_data = {
                key: value
                for idx, (key, value) in enumerate(mir_copy.raw_data.items())
                if (np.mod(idx, 2) == 0)
            }
        if select_auto:
            mir_copy.auto_data = {
                key: value
                for idx, (key, value) in enumerate(mir_copy.auto_data.items())
                if (np.mod(idx, 2) == 0)
            }
        if select_vis:
            mir_copy.vis_data = {
                key: value
                for idx, (key, value) in enumerate(mir_copy.vis_data.items())
                if (np.mod(idx, 2) == 0)
            }

    assert mir_data == mir_copy


@pytest.mark.parametrize("unload_auto", [True, False])
@pytest.mark.parametrize("unload_vis", [True, False])
@pytest.mark.parametrize("unload_raw", [True, False])
def test_unload_data(mir_data, unload_vis, unload_raw, unload_auto):
    """Verify that unload_data unloads data as expected."""
    # Spoof raw_data for just this test.
    mir_data.raw_data = mir_data.vis_data

    mir_data.unload_data(
        unload_vis=unload_vis, unload_raw=unload_raw, unload_auto=unload_auto
    )

    assert mir_data.vis_data is None if unload_vis else mir_data.vis_data is not None
    assert mir_data._tsys_applied != unload_vis

    assert mir_data.raw_data is None if unload_raw else mir_data.raw_data is not None

    assert mir_data.auto_data is None if unload_auto else mir_data.auto_data is not None


def test_load_data_err(mir_data):
    mir_data._clear_auto()

    with pytest.raises(ValueError) as err:
        mir_data.load_data(load_auto=True)
    assert str(err.value).startswith("This object has no auto-correlation data to")


@pytest.mark.parametrize(
    "optype,kwargs,warn_msg",
    [
        ["", {"load_vis": True, "load_raw": True}, "Cannot load raw and vis data"],
        ["load_raw", {"load_vis": True}, "Converting previously loaded data since"],
        ["muck_vis", {"allow_downselect": True}, "Cannot downselect cross-correlation"],
        ["muck_auto", {"allow_downselect": True}, "Cannot downselect auto-correlation"],
    ],
)
def test_load_data_warn(mir_data, optype, kwargs, warn_msg):
    if optype == "load_raw":
        mir_data.load_data(load_raw=True, load_vis=False, load_auto=False)
    elif optype == "muck_vis":
        mir_data.vis_data = {}
    elif optype == "muck_auto":
        mir_data.auto_data = {}

    with uvtest.check_warnings(UserWarning, warn_msg):
        mir_data.load_data(**kwargs)


@pytest.mark.parametrize("load_vis", [None, False])
def test_load_data_defaults(mir_data, load_vis):
    """Check that the default behavior of load_vis acts as expected."""
    # Blow away the old data first before we attempt.
    mir_data.unload_data()
    mir_data.load_data(load_vis=load_vis)

    if load_vis is None:
        assert mir_data.vis_data is not None
        assert mir_data.raw_data is None
    else:
        assert mir_data.vis_data is None
        assert mir_data.raw_data is not None

    assert (mir_data.vis_data is not None) == (load_vis is None)
    assert mir_data._tsys_applied == (load_vis is None)
    assert (mir_data.raw_data is None) == (load_vis is None)

    assert mir_data.auto_data is not None


def test_load_data_conv(mir_data):
    """Test that the conversion operation of load_data operates as expected."""
    mir_copy = mir_data.copy()

    mir_data.unload_data()
    assert mir_data.vis_data is None

    mir_data.load_data(load_vis=True, allow_conversion=False)
    mir_copy.load_data(load_vis=True, allow_conversion=True)

    assert mir_copy.vis_data is not None
    assert mir_copy == mir_data


def test_update_filter_update_data(mir_data):
    """
    Test that _update_filter behaves as expected with update_data.
    """
    mir_copy = mir_data.copy()

    # Manually unload the data, see if update_data will fix it.
    mir_data.vis_data = {}
    mir_data.auto_data = {}

    mir_data._update_filter(update_data=True)
    assert mir_data == mir_copy

    # Now see what happens if we don't explicitly allow for data to be updated.
    mir_data.vis_data = {}
    mir_data.auto_data = {}
    with uvtest.check_warnings(UserWarning, "Unable to update data attributes,"):
        mir_data._update_filter()

    assert mir_data.vis_data is None
    assert mir_data.auto_data is None


def test_reset(mir_data):
    mir_copy = mir_data.copy()
    mir_copy.rechunk(8)

    assert mir_data != mir_copy

    mir_copy.reset()
    mir_data.unload_data()
    assert mir_data == mir_copy

    for item in mir_data._metadata_attrs.values():
        assert item._stored_values == {}


@pytest.mark.parametrize(
    "unload_data,warn_msg",
    [
        [False, "Writing out raw data with tsys applied."],
        [True, "No data loaded, writing metadata only"],
    ],
)
def test_tofile_warn(mir_data, tmp_path, unload_data, warn_msg):
    """Test that tofile throws errors as expected."""
    testfile = os.path.join(
        tmp_path, "test_tofile_warn_%s.mir" % ("meta" if unload_data else "tsysapp")
    )
    if unload_data:
        mir_data.unload_data()

    with uvtest.check_warnings(UserWarning, warn_msg):
        mir_data.tofile(testfile)

    # Drop the data and autos here to make the comparison a bit easier.
    mir_data.unload_data()
    mir_data._clear_auto()

    mir_copy = MirParser(testfile)
    assert (
        mir_copy._file_dict[mir_copy.filepath] == mir_data._file_dict[mir_data.filepath]
    )
    assert mir_copy.filepath != mir_data.filepath
    assert mir_copy.filepath == testfile
    mir_copy.filepath = mir_data.filepath
    mir_copy._file_dict = mir_data._file_dict
    assert mir_copy == mir_data


@pytest.mark.parametrize("inplace", [True, False])
def test_rechunk_raw(inplace):
    """Test that rechunk_vis properly averages data"""
    raw_data = {
        5: {
            "raw_data": np.arange(-16384, 16384, dtype=np.int16),
            "scale_fac": np.int16(1),
        }
    }

    # First up, test what should be a no-op
    raw_copy = MirParser._rechunk_raw(raw_data, [1], inplace=inplace)

    assert (raw_copy is raw_data) == inplace

    assert raw_data.keys() == raw_copy.keys()
    assert raw_data[5]["scale_fac"] == 1
    assert np.all(raw_data[5]["raw_data"] == np.arange(-16384, 16384))

    # Now let's actually do some averaging and make sure it works as expected.
    raw_copy = MirParser._rechunk_raw(raw_data, [2], inplace=inplace)
    assert (raw_copy is raw_data) == inplace
    # Scale factor drops on account of having gotten rid of one sig binary digit
    # through the averaging process
    assert raw_copy[5]["scale_fac"] == 0
    # This is what raw_data _should_ look like after averaging. Note two aranges used
    # here because the spacing for real and imag is the same, but not real vs imag.
    assert np.all(
        raw_copy[5]["raw_data"]
        == np.vstack(
            (np.arange(-32766, 32768, 8), np.arange(-32764, 32768, 8))
        ).T.flatten()
    )
    raw_data = raw_copy

    # Finally, test that flagging works as expected
    raw_data[5]["raw_data"][2:] = -32768  # Marks channel as flagged
    raw_copy = MirParser._rechunk_raw(raw_data, [4096], inplace=inplace)
    assert (raw_copy is raw_data) == inplace
    # Scale factor should not change
    assert raw_copy[5]["scale_fac"] == 0
    # First channel should just contain channel 1 data, second channel should be flagged
    assert np.all(raw_copy[5]["raw_data"] == [-32766, -32764, -32768, -32768])


@pytest.mark.parametrize("inplace", [True, False])
def test_rechunk_vis(inplace):
    """Test that rechunk_raw properly averages data"""
    # Chicago FTW!
    vis_data = {
        25624: {
            "vis_data": (np.arange(1024) + np.flip(np.arange(1024) * 1j)),
            "vis_flags": np.zeros(1024, dtype=bool),
        }
    }
    check_vals = np.arange(1024) + np.flip(np.arange(1024) * 1j)

    # First up, test no averaging
    vis_copy = MirParser._rechunk_vis(vis_data, [1], inplace=inplace)

    assert (vis_copy is vis_data) == inplace

    assert vis_data.keys() == vis_copy.keys()
    assert np.all(vis_data[25624]["vis_flags"] == np.zeros(1024, dtype=bool))
    assert np.all(vis_data[25624]["vis_data"] == check_vals)

    # Next, test averaging w/o flags
    vis_copy = MirParser._rechunk_vis(vis_data, [4], inplace=inplace)
    check_vals = np.mean(check_vals.reshape(256, 4), axis=1)

    assert (vis_copy is vis_data) == inplace
    assert vis_data.keys() == vis_copy.keys()
    assert np.all(vis_copy[25624]["vis_flags"] == np.zeros(256, dtype=bool))
    assert np.all(vis_copy[25624]["vis_data"] == check_vals)
    vis_data = vis_copy

    # Finally, check what happens if we flag data
    vis_data[25624]["vis_flags"][1:] = True
    vis_copy = MirParser._rechunk_vis(vis_data, [128], inplace=inplace)
    assert (vis_copy is vis_data) == inplace
    assert vis_data.keys() == vis_copy.keys()
    assert np.all(vis_copy[25624]["vis_flags"] == [False, True])
    assert np.all(vis_copy[25624]["vis_data"] == [check_vals[0], 0.0])


@pytest.mark.parametrize("inplace", [True, False])
def test_rechunk_auto(inplace):
    auto_data = {5675309: np.arange(-1024, 1024, dtype=np.float32)}

    # First up, test no averaging
    auto_copy = MirParser._rechunk_auto(auto_data, [1], inplace=inplace)
    assert (auto_copy is auto_data) == inplace
    assert auto_data.keys() == auto_copy.keys()
    assert np.all(auto_copy[5675309] == np.arange(-1024, 1024, dtype=np.float32))

    # First up, test no averaging
    auto_copy = MirParser._rechunk_auto(auto_data, [512], inplace=inplace)
    assert (auto_copy is auto_data) == inplace
    assert auto_data.keys() == auto_copy.keys()
    assert np.all(auto_copy[5675309] == [-768.5, -256.5, 255.5, 767.5])


@pytest.mark.parametrize(
    "chan_avg,drop_data,err_type,err_msg",
    [
        [0.5, False, ValueError, "chan_avg must be of type int."],
        [-1, False, ValueError, "chan_avg cannot be a number less than"],
        [3, False, ValueError, "chan_avg does not go evenly into "],
        [2, True, ValueError, "Index values do not match data keys."],
    ],
)
def test_rechunk_errs(mir_data, chan_avg, drop_data, err_type, err_msg):
    """Verify that rechunk throws errors as expected."""
    if drop_data:
        mir_data.vis_data = {}

    # Rather than parameterizing this, because the underlying object isn't changed,
    # check for the different load states here, since the error should get thrown
    # no matter which thing you are loading.
    with pytest.raises(err_type) as err:
        mir_data.rechunk(chan_avg)
    assert str(err.value).startswith(err_msg)


def test_rechunk_nop(mir_data):
    """Test that setting chan_avg to 1 doesn't change the object."""
    mir_copy = mir_data.copy()

    mir_data.rechunk(1)
    assert mir_data == mir_copy


def test_rechunk_on_the_fly(mir_data):
    # Unload the autos for this test, since we do not _yet_ have support for on-the-fly
    # rechunking of that data.
    mir_data.unload_data(unload_vis=False, unload_auto=True)

    mir_data.rechunk(8)
    mir_copy = mir_data.copy()

    mir_copy.unload_data()
    mir_copy.load_data(load_vis=True, load_auto=False)

    assert mir_data == mir_copy


def test_rechunk_raw_vs_vis(mir_data):
    mir_copy = mir_data.copy()
    mir_copy.load_data(load_raw=True)

    # This will just rechunk the raw data
    mir_copy.rechunk(8)

    # This will rechunk the vis data
    mir_data.rechunk(8)

    # This will convert raw to vis in the copy
    with uvtest.check_warnings(UserWarning, "Converting previously loaded data"):
        mir_copy.load_data(allow_conversion=True, load_vis=True)

    assert mir_copy == mir_data


@pytest.mark.parametrize(
    "muck_data,kwargs,err_type,err_msg",
    [
        [["in_data"], {}, ValueError, "Cannot merge objects due to conflicts"],
        [["file"], {}, ValueError, "Duplicate metadata found for the following"],
        [["auto"], {}, ValueError, "Cannot combine two MirParser objects if one "],
        [["jypk"], {}, ValueError, "Cannot combine objects where the jypk value"],
        [["all"], {}, TypeError, "Cannot add a MirParser object an object of "],
        [[], {"merge": False}, ValueError, "Must set merge=True in order to"],
        [["int_start"], {"merge": True}, ValueError, "These two objects were"],
        [["file"], {"merge": True}, ValueError, "Cannot merge objects that"],
        [["file", "antpos"], {"merge": False}, ValueError, "Antenna positions differ"],
    ],
)
def test_add_errs(mir_data, muck_data, kwargs, err_type, err_msg):
    """Verify that __add__ throws errors as expected"""
    mir_copy = mir_data.copy()

    if "in_data" in muck_data:
        mir_data.in_data["mjd"] = 0.0
    if "auto" in muck_data:
        mir_data._clear_auto()
    if "jypk" in muck_data:
        mir_data.jypk = 1.0
    if "file" in muck_data:
        mir_data._file_dict = {}
    if "int_start" in muck_data:
        for key in mir_data._file_dict:
            mir_data._file_dict[key] = {}
    if "antpos" in muck_data:
        mir_data.antpos_data["xyz_pos"] = 0.0
    if "all" in muck_data:
        mir_data = np.arange(100)

    with pytest.raises(err_type) as err:
        mir_data.__add__(mir_copy, **kwargs)
    assert str(err.value).startswith(err_msg) or ("all" in muck_data)

    with pytest.raises(err_type) as err:
        mir_copy.__add__(mir_data, **kwargs)
    assert str(err.value).startswith(err_msg)


def test_add_merge(mir_data):
    """
    Verify that the __add__ method behaves as expected under 'simple' scenarios, i.e.,
    where overwrite or force are not neccessary.
    """
    mir_copy = mir_data.copy()
    mir_orig = mir_data.copy()

    # So this is a _very_ simple check, but make sure that combining two
    # objects that have all data loaded returns an equivalent object.
    assert mir_data == (mir_data + mir_data)

    # Now try in-place adding
    mir_data += mir_data
    assert mir_data == mir_copy

    # Alright, now try running a select and split the data into two.
    mir_data.select(where=("corrchunk", "eq", [0, 1, 2]))
    mir_copy.select(where=("corrchunk", "ne", [0, 1, 2]))

    # Verify that we have changed some things
    assert mir_data != mir_orig
    assert mir_data != mir_copy
    assert mir_orig != mir_copy

    # Now combine the two, and see what comes out.
    mir_data += mir_copy
    assert mir_data == mir_orig

    # Hey, that was fun, let's try selecting on bl next!
    mir_data.select(reset=True, update_data=True)
    mir_copy.select(reset=True, update_data=True)

    mir_data.select(where=("sb", "eq", "l"))
    mir_copy.select(where=("sb", "eq", "u"))

    # The reset unloads the data, so fix that now
    mir_data.load_data(load_vis=True, load_auto=True, apply_tsys=True)
    mir_copy.load_data(load_vis=True, load_auto=True, apply_tsys=True)

    # Verify that we have changed some things
    assert mir_data != mir_orig
    assert mir_data != mir_copy
    assert mir_orig != mir_copy

    # Now combine the two, and see what comes out.
    mir_data += mir_copy
    assert mir_data == mir_orig

    # Finally, let's try something a little different. Drop autos on one object, and
    # do a filter where the union of the two objects does NOT give you back the sum
    # total of the other object
    mir_data.select(reset=True, update_data=True)
    mir_copy.select(reset=True, update_data=True)

    mir_copy._clear_auto()

    mir_data.select(where=("corrchunk", "eq", [1, 2]))
    mir_copy.select(where=("corrchunk", "eq", [3, 4]))
    mir_data.load_data(load_vis=True, apply_tsys=True)
    mir_copy.load_data(load_vis=True, apply_tsys=True)

    with uvtest.check_warnings(
        UserWarning, "Both objects do not have auto-correlation data."
    ):
        mir_data.__iadd__(mir_copy, force=True)

    # Make sure we got all the data entries
    assert mir_data._check_data_index()

    # Make sure auto properties propagated correctly.
    assert (not mir_data._has_auto) and (mir_data.auto_data is None)
    mir_orig._clear_auto()

    # Finally, make sure the object isn't the same, but after a reset and reload,
    # we get the same object back (modulo the auto-correlation data).
    assert mir_data != mir_orig
    mir_data.select(reset=True, update_data=True)
    mir_data.load_data(load_vis=True, apply_tsys=True)
    assert mir_data == mir_orig


@parametrize("drop_auto", [True, False])
@parametrize("drop_raw", [True, False])
@parametrize("drop_vis", [True, False, "jypk", "tsys"])
def test_add_drop_data(mir_data, drop_auto, drop_raw, drop_vis):
    mir_data.raw_data = mir_data.convert_vis_to_raw(mir_data.vis_data)
    mir_copy = mir_data.copy()

    if drop_auto:
        mir_copy.auto_data = None
    if drop_raw:
        mir_copy.raw_data = None
    if drop_vis:
        if drop_vis == "jypk":
            mir_copy.jypk = 0.0
        elif drop_vis == "tsys":
            mir_copy._tsys_applied = False
        else:
            mir_copy.vis_data = None

    result = mir_data.__add__(mir_copy, overwrite=(drop_vis == "jypk"))

    assert (result.auto_data is None) == bool(drop_auto)
    assert (result.raw_data is None) == bool(drop_raw)
    assert (result.vis_data is None) == bool(drop_vis)


@pytest.mark.parametrize(
    "muck_attr",
    [
        "ac_data",
        "antpos_data",
        "bl_data",
        "eng_data",
        "in_data",
        "sp_data",
        "we_data",
        "all",
        "codes",
    ],
)
def test_add_overwrite(mir_data, muck_attr):
    """Verify that the overwrite option on __add__ works as expected."""
    mir_copy = mir_data.copy()

    prot_fields = [
        "inhid",
        "blhid",
        "sphid",
        "ints",
        "antenna",
        "antennaNumber",
        "achid",
        "v_name",
        "icode",
        "ncode",
        "dataoff",
    ]
    if muck_attr == "all":
        for item in [
            "ac_data",
            "antpos_data",
            "bl_data",
            "eng_data",
            "in_data",
            "sp_data",
            "we_data",
        ]:
            for field in getattr(mir_data, item).dtype.names:
                if field not in prot_fields:
                    getattr(mir_data, item)[field] = -1
    elif muck_attr == "codes":
        mir_data.codes_data.set_value("code", "1", where=("v_name", "eq", "filever"))
    else:
        for field in getattr(mir_data, muck_attr).dtype.names:
            if field not in prot_fields:
                getattr(mir_data, muck_attr)[field] = -1

    # After mucking, verfiy that at least something looks different
    assert mir_data != mir_copy

    # mir_copy contains the good data, so adding it second will overwrite the bad data.
    assert mir_data.__add__(mir_copy, overwrite=True) == mir_copy

    # On the other hand, if we add mir_data second, the bad values should get propagated
    assert mir_copy.__add__(mir_data, overwrite=True) == mir_data


def test_add_concat_warn(mir_data, tmp_path):
    filepath = os.path.join(tmp_path, "add_concat_warn")

    with uvtest.check_warnings(UserWarning, "Writing out raw data with tsys applied."):
        mir_data.tofile(filepath)

    mir_copy = MirParser(filepath)
    with uvtest.check_warnings(
        UserWarning,
        [
            "Duplicate metadata found for the following attributes",
            "These two objects contain data taken at the exact same time",
            "Both objects do not have auto-correlation data.",
        ],
    ):
        mir_copy.__iadd__(mir_data, force=True, merge=False)

    assert mir_copy != mir_data
    for item in mir_copy._metadata_attrs:
        if item == "codes_data":
            assert mir_data.codes_data == mir_copy.codes_data
        else:
            assert len(getattr(mir_copy, item)) == (2 * len(getattr(mir_data, item)))


def test_add_concat(mir_data, tmp_path):
    filepath = os.path.join(tmp_path, "add_concat")

    # Clear out the autos, since we can't write them as full records _yet_
    mir_data._clear_auto()
    mir_copy = mir_data.copy()

    # Preserve particular fields that we want to propagate into the next file.
    prot_fields = [
        "inhid",
        "blhid",
        "sphid",
        "ints",
        "antenna",
        "xyz_pos",
        "antennaNumber",
        "achid",
        "v_name",
        "icode",
        "ncode",
        "dataoff",
        "nch",
        "iant1",
        "iant2",
        "tsys",
        "tsys_rx2",
        "ant1rx",
        "ant2rx",
    ]

    for item in ["bl_data", "eng_data", "in_data", "sp_data", "we_data"]:
        for field in getattr(mir_copy, item).dtype.names:
            if field not in prot_fields:
                getattr(mir_copy, item)[field] = 156

    mir_copy.codes_data.set_value("code", "3c279", where=("v_name", "eq", "source"))

    with uvtest.check_warnings(UserWarning, "Writing out raw data with tsys applied."):
        mir_copy.tofile(filepath, overwrite=True)
    # Read the file in so that we have a dict here to work with.
    mir_copy.fromfile(filepath, load_vis=True)

    new_obj = mir_data + mir_copy

    # Make sure that the
    for item, this_attr in new_obj._metadata_attrs.items():
        other_attr = mir_data._metadata_attrs[item]
        if item == "antpos_data":
            assert this_attr == other_attr
        else:
            assert this_attr != other_attr
            if item == "codes_data":
                # We add 3 here since the 1 extra source creates 3 new codes entries
                assert len(this_attr) == (3 + len(other_attr))

                # Make sure that the source actually got updated as expected
                assert this_attr["source"]["3c279"] == 2
            else:
                # Otherwise, the number of entries should double for all attributes
                assert len(this_attr) == (2 * len(other_attr))
                for field in this_attr.dtype.names:
                    if field not in prot_fields:
                        assert np.all(this_attr[field][len(other_attr) :] == 156)


@pytest.mark.parametrize(
    "kern_type,tol,err_type,err_msg",
    [
        ["cubic", -1, ValueError, "tol must be in the range [0, 0.5]."],
        ["abc", 0.5, ValueError, 'Kernel type of "abc" not recognized,'],
    ],
)
def test_generate_chanshift_kernel_errs(kern_type, tol, err_type, err_msg):
    """ "Verify that _generate_chanshift_kernel throws errors as expected."""
    with pytest.raises(err_type) as err:
        MirParser._generate_chanshift_kernel(1.5, kern_type, tol=tol)
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "kern_type,chan_shift,alpha,tol,exp_coarse,exp_kern",
    [
        ["nearest", 0, -0.5, 1e-3, 0, []],
        ["linear", 0, -0.5, 1e-3, 0, []],
        ["cubic", 0, -0.5, 1e-3, 0, []],
        ["nearest", -1, -0.5, 1e-3, -1, []],
        ["linear", 2, -0.5, 1e-3, 2, []],
        ["cubic", -3, -0.5, 1e-3, -3, []],
        ["nearest", 1.5, -0.5, 1e-3, 2, []],
        ["linear", 2.0005, -0.5, 1e-3, 2, []],
        ["cubic", -3.1, -0.5, 0.2, -3, []],
        ["linear", 1.3, -0.5, 1e-3, 1, [0.7, 0.3]],
        ["linear", -1.3, -0.5, 1e-3, -2, [0.3, 0.7]],
        ["cubic", -3.5, 0, 1e-3, -4, [0, 0.5, 0.5, 0]],
        ["cubic", 1.4, 0.0, 1e-3, 1, [0.0, 0.352, 0.648, 0.0]],
        ["cubic", 1.4, -0.5, 1e-4, 1, [-0.048, 0.424, 0.696, -0.072]],
        ["cubic", 1.4, -1.0, 0, 1, [-0.096, 0.496, 0.744, -0.144]],
    ],
)
def test_generate_chanshift_kernel(
    mir_data, kern_type, chan_shift, alpha, tol, exp_coarse, exp_kern
):
    """Test that _generate_chanshift_kernel produces kernels as expected."""
    (coarse_shift, kern_size, kern) = MirParser._generate_chanshift_kernel(
        chan_shift, kern_type, alpha_fac=alpha, tol=tol
    )

    assert coarse_shift == exp_coarse
    assert len(exp_kern) == kern_size
    if kern is None:
        assert exp_kern == []
    else:
        assert np.allclose(exp_kern, kern)


@pytest.mark.parametrize("check_flags", [True, False])
@pytest.mark.parametrize("fwd_dir", [True, False])
@pytest.mark.parametrize(
    "inplace,return_vis", [[True, False], [False, True], [False, False]]
)
def test_chanshift_raw_vals(inplace, return_vis, fwd_dir, check_flags):
    """Test that _chanshift_raw modifies spectra as expected."""
    # Create a dataset to set against using a dummy impulse in the DC channel
    raw_vals = []
    raw_vals.extend([32767 if check_flags else 0] * 8)
    raw_vals.extend([-32768 if check_flags else 32767] * 2)
    raw_vals.extend([32767 if check_flags else 0] * 6)

    raw_dict = {
        123: {"raw_data": np.array(raw_vals, dtype=np.int16), "scale_fac": np.int16(0)}
    }

    # Test no-op
    new_dict = MirParser._chanshift_raw(
        raw_dict, [(0, 0, None)], inplace=inplace, return_vis=return_vis
    )
    if inplace:
        assert new_dict is raw_dict
    if return_vis:
        new_dict = MirParser.convert_vis_to_raw(new_dict)

    assert np.all(raw_vals == new_dict[123]["raw_data"])
    assert new_dict[123]["scale_fac"] == 0

    # Now try a simple one-channel shift
    new_dict = MirParser._chanshift_raw(
        raw_dict,
        [(1 if fwd_dir else -1, 0, None)],
        inplace=inplace,
        return_vis=return_vis,
    )
    if inplace:
        assert new_dict is raw_dict
    if return_vis:
        new_dict = MirParser.convert_vis_to_raw(new_dict)

    good_slice = slice(None if fwd_dir else 2, -2 if fwd_dir else None)
    flag_slice = slice(None if fwd_dir else -2, 2 if fwd_dir else None)
    # Note that the shift of 2 is required since each channel has a real and imag
    # component. The first two entries are dropped because they _should_ be flagged.
    assert np.all(
        raw_vals[good_slice]
        == np.roll(new_dict[123]["raw_data"], -2 if fwd_dir else 2)[good_slice]
    )
    assert np.all(new_dict[123]["raw_data"][flag_slice] == -32768)
    assert new_dict[123]["scale_fac"] == 0

    # Refresh the values, in case we are doing this in-place
    if inplace:
        raw_dict = {
            123: {
                "raw_data": np.array(raw_vals, dtype=np.int16),
                "scale_fac": np.int16(0),
            }
        }

    # Last check, try a linear interpolation step
    new_dict = MirParser._chanshift_raw(
        raw_dict,
        [(1 if fwd_dir else -2, 2, np.array([0.5, 0.5], dtype=np.float32))],
        inplace=inplace,
        return_vis=return_vis,
    )
    if inplace:
        assert new_dict is raw_dict
    if return_vis:
        new_dict = MirParser.convert_vis_to_raw(new_dict)

    if fwd_dir:
        assert np.all(new_dict[123]["raw_data"][14:16] == (32767 if check_flags else 0))
        assert np.all(
            new_dict[123]["raw_data"][10:14] == (-32768 if check_flags else 32767)
        )
        assert np.all(new_dict[123]["raw_data"][4:10] == (32767 if check_flags else 0))
        assert np.all(new_dict[123]["raw_data"][0:4] == -32768)
    else:
        assert np.all(new_dict[123]["raw_data"][0:4] == (32767 if check_flags else 0))
        assert np.all(
            new_dict[123]["raw_data"][4:8] == (-32768 if check_flags else 32767)
        )
        assert np.all(new_dict[123]["raw_data"][8:12] == (32767 if check_flags else 0))
        assert np.all(new_dict[123]["raw_data"][12:16] == -32768)
    assert new_dict[123]["scale_fac"] == (0 if check_flags else -1)


@pytest.mark.parametrize(
    "check_flags,flag_adj", [[False, True], [True, False], [True, True]]
)
@pytest.mark.parametrize("fwd_dir", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_chanshift_vis(check_flags, flag_adj, fwd_dir, inplace):
    """Test that _chanshift_vis modifies spectra as expected."""
    check_val = -(1 + 2j) if check_flags else (1 + 2j)
    vis_vals = [check_val if check_flags else 0] * 4
    vis_vals.append((3 + 4j) if check_flags else check_val)
    vis_vals.extend([check_val if check_flags else 0] * 3)
    flag_vals = [False] * 4
    flag_vals.append(check_flags)
    flag_vals.extend([False] * 3)

    vis_dict = {
        456: {
            "vis_data": np.array(vis_vals, dtype=np.complex64),
            "vis_flags": np.array(flag_vals, dtype=bool),
        }
    }

    # Test no-op
    new_dict = MirParser._chanshift_vis(
        vis_dict, [(0, 0, None)], flag_adj=flag_adj, inplace=inplace
    )

    if inplace:
        assert new_dict is vis_dict

    assert np.all(vis_vals == new_dict[456]["vis_data"])
    assert np.all(new_dict[456]["vis_flags"] == flag_vals)

    # Now try a simple one-channel shift
    new_dict = MirParser._chanshift_vis(
        vis_dict,
        [(1 if fwd_dir else -1, 0, None)],
        flag_adj=flag_adj,
        inplace=inplace,
    )

    if inplace:
        assert new_dict is vis_dict

    good_slice = slice(None if fwd_dir else 1, -1 if fwd_dir else None)
    flag_slice = slice(None if fwd_dir else -1, 1 if fwd_dir else None)

    assert np.all(
        vis_vals[good_slice]
        == np.roll(new_dict[456]["vis_data"], -1 if fwd_dir else 1)[good_slice]
    )
    assert np.all(
        flag_vals[good_slice]
        == np.roll(new_dict[456]["vis_flags"], -1 if fwd_dir else 1)[good_slice]
    )

    assert np.all(new_dict[456]["vis_data"][flag_slice] == 0.0)
    assert np.all(new_dict[456]["vis_flags"][flag_slice])

    # Refresh the values, in case we are doing this in-place
    if inplace:
        vis_dict = {
            456: {
                "vis_data": np.array(vis_vals, dtype=np.complex64),
                "vis_flags": np.array(flag_vals, dtype=bool),
            }
        }

    # Last check, try a linear interpolation step
    new_dict = MirParser._chanshift_vis(
        vis_dict,
        [(1 if fwd_dir else -2, 2, np.array([0.75, 0.25], dtype=np.float32))],
        flag_adj=flag_adj,
        inplace=inplace,
    )
    if inplace:
        assert new_dict is vis_dict

    exp_vals = np.roll(vis_vals, 2 if fwd_dir else -2)
    exp_flags = np.roll(flag_vals, 2 if fwd_dir else -2)
    exp_vals[None if fwd_dir else -2 : 2 if fwd_dir else None] = 0.0
    exp_flags[None if fwd_dir else -2 : 2 if fwd_dir else None] = True
    mod_slice = slice(4 - (-1 if fwd_dir else 2), 6 - (-1 if fwd_dir else 2))
    if flag_adj:
        exp_flags[mod_slice] = check_flags
        exp_vals[mod_slice] = 0 if check_flags else [check_val * 0.75, check_val * 0.25]
    else:
        exp_vals[mod_slice] = [check_val * 0.25, check_val * 0.75]
        exp_flags[mod_slice] = False

    assert np.all(new_dict[456]["vis_data"] == exp_vals)
    assert np.all(new_dict[456]["vis_flags"] == exp_flags)


@pytest.mark.parametrize(
    "filever,irec,err_type,err_msg",
    [
        ["2", 3, ValueError, "MIR file format < v4.0 detected,"],
        ["4", 3, ValueError, "Receiver code 3 not recognized."],
    ],
)
def test_redoppler_data_errs(mir_data, filever, irec, err_type, err_msg):
    """Verift that redoppler_data throws errors as expected."""
    mir_data.codes_data.set_value("code", filever, where=("v_name", "eq", "filever"))
    mir_data.bl_data["irec"] = irec

    with pytest.raises(err_type) as err:
        mir_data.redoppler_data()
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize("plug_vals", [True, False])
@pytest.mark.parametrize("diff_rx", [True, False])
@pytest.mark.parametrize("use_raw", [True, False])
def test_redoppler_data(mir_data, plug_vals, diff_rx, use_raw):
    """Verify that redoppler_data behaves as expected."""
    # We have to spoof the filever because the test file is technically v3
    mir_data.codes_data.set_value("code", "4", where=("v_name", "eq", "filever"))

    if use_raw:
        mir_data.raw_data = mir_data.convert_vis_to_raw(mir_data.vis_data)
        mir_data.vis_data = None

    mir_copy = mir_data.copy()
    # This first attempt should basically just be a no-op
    mir_copy.redoppler_data()

    assert mir_data == mir_copy

    # Alright, let's tweak the data now to give us something to compare
    for sphid, nch in zip(mir_data.sp_data["sphid"], mir_data.sp_data["nch"]):
        if use_raw:
            mir_data.raw_data[sphid]["raw_data"][:] = np.arange(nch * 2)
            mir_data.raw_data[sphid]["scale_fac"] = np.int16(0)
        else:
            mir_data.vis_data[sphid]["vis_data"][:] = np.arange(nch)
            mir_data.vis_data[sphid]["vis_flags"][:] = False

    rxb_blhids = mir_data.bl_data["blhid"][mir_data.bl_data["ant1rx"] == 1]

    freq_shift = (
        (139.6484375e-6)
        * (1 + (diff_rx & np.isin(mir_data.sp_data["blhid"], rxb_blhids)))
        * (mir_data.sp_data["corrchunk"] != 0)
    )

    if plug_vals:
        # Note we need the factor of two here now to simulate the an error
        # that is currently present
        # TODO: Remove this once the underlying issue is fixed.
        mir_data.sp_data["fDDS"] = -(freq_shift / 2)
        freq_shift = None

    mir_data.redoppler_data(freq_shift=freq_shift)

    # Alright, let's tweak the data now to give us something to compare
    for sp_rec in mir_data.sp_data:
        sphid = sp_rec["sphid"]
        nch = sp_rec["nch"]
        chan_shift = int(-np.sign(sp_rec["fres"]) * (sp_rec["corrchunk"] != 0))
        chan_shift *= 2 if ((sp_rec["blhid"] in rxb_blhids) and diff_rx) else 1
        if chan_shift == 0:
            if use_raw:
                assert np.all(
                    mir_data.raw_data[sphid]["raw_data"] == np.arange(nch * 2)
                )
            else:
                assert np.all(mir_data.vis_data[sphid]["vis_data"] == np.arange(nch))
        elif chan_shift < 0:
            if use_raw:
                assert np.all(
                    mir_data.raw_data[sphid]["raw_data"][: chan_shift * 2]
                    == np.arange(-(2 * chan_shift), nch * 2)
                )
            else:
                assert np.all(
                    mir_data.vis_data[sphid]["vis_data"][:chan_shift]
                    == np.arange(-chan_shift, nch)
                )
        else:
            if use_raw:
                assert np.all(
                    mir_data.raw_data[sphid]["raw_data"][chan_shift * 2 :]
                    == np.arange((nch - chan_shift) * 2)
                )
            else:
                assert np.all(
                    mir_data.vis_data[sphid]["vis_data"][chan_shift:]
                    == np.arange((nch - chan_shift))
                )


# Below are a series of checks that are designed to check to make sure that the
# MirParser class is able to produce consistent values from an engineering data
# set (originally stored in /data/engineering/mir_data/200724_16:35:14), to make
# sure that we haven't broken the ability of the reader to handle the data.


def test_mir_remember_me_record_lengths(mir_data):
    """
    Mir record length checker

    Make sure the test file contains the right number of records
    """
    # Check to make sure we've got the right number of records everywhere

    # ac_data only exists if has_auto=True
    if mir_data.ac_data._data is not None:
        assert len(mir_data.ac_data) == 2
    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto

    assert len(mir_data.bl_data) == 4

    assert len(mir_data.codes_data) == 99

    assert len(mir_data.eng_data) == 2

    assert len(mir_data.in_data) == 1

    assert len(mir_data.sp_data) == 20

    assert len(mir_data.vis_data) == 20

    assert len(mir_data.we_data) == 1


def test_mir_remember_me_codes_data(mir_data):
    """
    Mir codes_read checker.

    Make sure that certain values in the codes_read file of the test data set match
    whatwe know to be 'true' at the time of observations.
    """
    assert mir_data.codes_data["filever"][0] == "3"

    assert mir_data.codes_data["ref_time"][0] == "Jul 24, 2020"

    assert mir_data.codes_data["ut"][1] == "Jul 24 2020  4:34:39.00PM"

    assert mir_data.codes_data["source"][1] == "3c84"

    assert mir_data.codes_data["ra"][1] == "03:19:48.15"

    assert mir_data.codes_data["dec"][1] == "+41:30:42.1"


def test_mir_remember_me_in_data(mir_data):
    """
    Mir in_read checker.

    Make sure that certain values in the in_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Check to make sure that things seem right in in_read
    assert np.all(mir_data.in_data["traid"] == 484)

    assert np.all(mir_data.in_data["proid"] == 484)

    assert np.all(mir_data.in_data["inhid"] == 1)

    assert np.all(mir_data.in_data["ints"] == 1)

    assert np.all(mir_data.in_data["souid"] == 1)

    assert np.all(mir_data.in_data["isource"] == 1)

    assert np.all(mir_data.in_data["ivrad"] == 1)

    assert np.all(mir_data.in_data["ira"] == 1)

    assert np.all(mir_data.in_data["idec"] == 1)

    assert np.all(mir_data.in_data["epoch"] == 2000.0)

    assert np.all(mir_data.in_data["tile"] == 0)

    assert np.all(mir_data.in_data["obsflag"] == 0)

    assert np.all(mir_data.in_data["obsmode"] == 0)

    assert np.all(np.round(mir_data.in_data["mjd"]) == 59055)

    assert np.all(mir_data.in_data["spareshort"] == 0)

    assert np.all(mir_data.in_data["spareint6"] == 0)


def test_mir_remember_me_bl_data(mir_data):
    """
    Mir bl_read checker.

    Make sure that certain values in the bl_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Now check bl_read
    assert np.all(mir_data.bl_data["blhid"] == np.arange(1, 5))

    assert np.all(mir_data.bl_data["isb"] == [0, 0, 1, 1])

    assert np.all(mir_data.bl_data["ipol"] == [0, 0, 0, 0])

    assert np.all(mir_data.bl_data["ant1rx"] == [0, 1, 0, 1])

    assert np.all(mir_data.bl_data["ant2rx"] == [0, 1, 0, 1])

    assert np.all(mir_data.bl_data["pointing"] == 0)

    assert np.all(mir_data.bl_data["irec"] == [0, 3, 0, 3])

    assert np.all(mir_data.bl_data["iant1"] == 1)

    assert np.all(mir_data.bl_data["iant2"] == 4)

    assert np.all(mir_data.bl_data["iblcd"] == 2)

    assert np.all(mir_data.bl_data["spareint1"] == 0)

    assert np.all(mir_data.bl_data["spareint2"] == 0)

    assert np.all(mir_data.bl_data["spareint3"] == 0)

    assert np.all(mir_data.bl_data["spareint4"] == 0)

    assert np.all(mir_data.bl_data["spareint5"] == 0)

    assert np.all(mir_data.bl_data["spareint6"] == 0)

    assert np.all(mir_data.bl_data["wtave"] == 0.0)

    assert np.all(mir_data.bl_data["sparedbl4"] == 0.0)

    assert np.all(mir_data.bl_data["sparedbl5"] == 0.0)

    assert np.all(mir_data.bl_data["sparedbl6"] == 0.0)


def test_mir_remember_me_eng_data(mir_data):
    """
    Mir eng_read checker.

    Make sure that certain values in the eng_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    # Now check eng_read
    assert np.all(mir_data.eng_data["antennaNumber"] == [1, 4])

    assert np.all(mir_data.eng_data["padNumber"] == [5, 8])

    assert np.all(mir_data.eng_data["trackStatus"] == 1)

    assert np.all(mir_data.eng_data["commStatus"] == 1)

    assert np.all(mir_data.eng_data["inhid"] == 1)


def test_mir_remember_me_we_data(mir_data):
    """
    Mir we_read checker.

    Make sure that certain values in the we_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    assert np.all(mir_data.we_data["ints"] == 1)

    assert np.all(mir_data.we_data["flags"] == 0)


def test_mir_remember_me_ac_data(mir_data):
    """
    Mir ac_read checker.

    Make sure that certain values in the autoCorrelations file of the test data set
    match what we know to be 'true' at the time of observations.
    """
    # Now check ac_read

    # ac_read only exists if has_auto=True
    if mir_data.ac_data is not None:

        assert np.all(mir_data.ac_data["inhid"] == 1)

        assert np.all(mir_data.ac_data["achid"] == np.arange(1, 3))

        assert np.all(mir_data.ac_data["antenna"] == [1, 4])

        assert np.all(mir_data.ac_data["nchunks"] == 8)

        assert np.all(mir_data.ac_data["datasize"] == 1048576)

    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto


def test_mir_remember_me_sp_data(mir_data):
    """
    Mir sp_read checker.

    Make sure that certain values in the sp_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Now check sp_read
    assert np.all(mir_data.sp_data["sphid"] == np.arange(1, 21))

    assert np.all(mir_data.sp_data["sphid"] == np.arange(1, 21))

    assert np.all(mir_data.sp_data["igq"] == 0)

    assert np.all(mir_data.sp_data["ipq"] == 1)

    assert np.all(mir_data.sp_data["igq"] == 0)

    assert np.all(mir_data.sp_data["iband"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data.sp_data["ipstate"] == 0)

    assert np.all(mir_data.sp_data["tau0"] == 0.0)

    assert np.all(mir_data.sp_data["cabinLO"] == 0.0)

    assert np.all(mir_data.sp_data["corrLO1"] == 0.0)

    assert np.all(mir_data.sp_data["vradcat"] == 0.0)

    assert np.all(mir_data.sp_data["nch"] == [4, 16384, 16384, 16384, 16384] * 4)

    assert np.all(mir_data.sp_data["corrblock"] == [0, 1, 1, 1, 1] * 4)

    assert np.all(mir_data.sp_data["corrchunk"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data.sp_data["correlator"] == 1)

    assert np.all(mir_data.sp_data["iddsmode"] == 0)

    assert np.all(mir_data.sp_data["spareshort"] == 0)

    assert np.all(mir_data.sp_data["spareint3"] == 0)

    assert np.all(mir_data.sp_data["spareint4"] == 0)

    assert np.all(mir_data.sp_data["spareint5"] == 0)

    assert np.all(mir_data.sp_data["spareint6"] == 0)

    assert np.all(mir_data.sp_data["tssb"] == 0.0)

    assert np.all(mir_data.sp_data["fDDS"] == 0.0)

    assert np.all(mir_data.sp_data["sparedbl3"] == 0.0)

    assert np.all(mir_data.sp_data["sparedbl4"] == 0.0)

    assert np.all(mir_data.sp_data["sparedbl5"] == 0.0)

    assert np.all(mir_data.sp_data["sparedbl6"] == 0.0)


def test_mir_remember_me_vis_data(mir_data):
    """
    Mir sch_read checker.

    Make sure that certain values in the sch_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    mir_data.load_data(load_raw=True)
    # Now check sch_read related values. Thanks to a glitch in the data recorder,
    # all of the pseudo-cont values are the same for the test file.
    assert np.all(
        sp_raw["scale_fac"] == -26 if (np.mod(idx, 5) == 0) else True
        for idx, sp_raw in enumerate(mir_data.raw_data.values())
    )

    check_arr = np.array([-4302, -20291, -5261, -21128, -4192, -19634, -4999, -16346])

    assert np.all(
        np.all(sp_raw["raw_data"] == check_arr) if (np.mod(idx, 5) == 0) else True
        for idx, sp_raw in enumerate(mir_data.raw_data.values())
    )
