# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MirParser class.

Performs a series of tests on the MirParser, which is the python-based reader for MIR
data in pyuvdata. Tests in this module are specific to the way that MIR is read into
python, not necessarily how pyuvdata (by way of the UVData class) interacts with that
data.
"""
import copy
import os

import h5py
import numpy as np
import pytest

from ... import tests as uvtest
from ...data import DATA_PATH
from ..mir_parser import (
    NEW_AUTO_DTYPE,
    NEW_AUTO_HEADER,
    NEW_VIS_DTYPE,
    NEW_VIS_HEADER,
    OLD_AUTO_HEADER,
    MirMetaError,
    MirPackdataError,
    MirParser,
)


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

        bp_soln = np.reshape(
            np.concatenate((bp_soln / 2, np.conj(np.reciprocal(bp_soln)))),
            (2, 16, 16384),
        ).astype(np.complex64)

        file["reBandpassArr"] = bp_soln.real
        file["imBandpassArr"] = bp_soln.imag

        # Populate the SEFD values
        file["sefdArr"] = np.ones(bp_soln.shape)

        # This number is pulled from the test mir_data object, in in_data["mjd"].
        file["mjdArr"] = np.array([[59054.69153811]])

        # Set up a picket fence of flags for the "normal" flagging. Note we use
        # uint8 here because of the compression scheme COMPASS uses.
        file["flagArr"] = np.full((1, 1, 16, 2048), 170, dtype=np.uint8)

        # Set up the static flags so that the first half of the spectrum is flagged.
        file["staticFlagArr"] = np.tile(
            ((np.arange(2048) < 1024) * 255).reshape(1, 1, -1).astype(np.uint8),
            (8, 16, 1),
        )

    yield filename


def test_mir_parser_index_uniqueness(mir_data):
    """
    Mir index uniqueness check

    Make sure that there are no duplicate indices for things that are primary keys
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

    Make sure that all referenced indices have matching pairs in their parent tables
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
    mir_data.raw_data = {1: {"data": [1] * 16384, "scale_fac": [1]}}

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
    orig_attr.write(filepath)
    check_attr = orig_attr.copy(skip_data=True)
    check_attr.read(filepath)
    assert orig_attr == check_attr


def test_mir_write_vis_data_err(mir_data, tmp_path):
    mir_data.unload_data()
    with pytest.raises(ValueError, match="Cannot write data if not already loaded."):
        mir_data._write_cross_data(tmp_path)


def test_mir_raw_data(mir_data, tmp_path):
    """
    Test reading and writing of raw data.
    """
    filepath = os.path.join(tmp_path, "test_write_raw")
    mir_data.load_data(load_raw=True)

    mir_data._write_cross_data(filepath)
    # Sub out the file we need to read from
    mir_data._file_dict = {filepath: list(mir_data._file_dict.values())[0]}
    raw_data = mir_data._read_data("cross", scale_data=False)

    assert raw_data.keys() == mir_data.raw_data.keys()

    for key in raw_data.keys():
        for subkey in ["data", "scale_fac"]:
            assert np.array_equal(raw_data[key][subkey], mir_data.raw_data[key][subkey])


def test_mir_auto_data_errs(mir_data):
    mir_data.unload_data()
    with pytest.raises(ValueError, match="Cannot write data if not already loaded."):
        mir_data._write_auto_data(None)


def test_mir_auto_data(mir_data: MirParser, tmp_path):
    """
    Test reading and writing of auto data.
    """
    filepath = os.path.join(tmp_path, "test_write_auto")

    mir_data._write_auto_data(filepath)
    # Sub out the file we need to read from, and fix a couple of attributes that changed
    # since we are no longer spoofing values (after reading in data from old-style file)
    mir_data._file_dict = {filepath: list(mir_data._file_dict.values())[0]}
    mir_data._file_dict[filepath]["auto"]["filetype"] = "ach_read"
    mir_data._file_dict[filepath]["auto"]["read_hdr_fmt"] = NEW_AUTO_HEADER
    int_dict, mir_data._ac_dict = mir_data.ac_data._generate_recpos_dict(
        data_dtype=NEW_AUTO_DTYPE,
        data_nvals=1,
        pad_nvals=0,
        scale_data=False,
        hdr_fmt=NEW_AUTO_HEADER,
        reindex=True,
    )
    mir_data._file_dict[filepath]["auto"]["int_dict"] = int_dict
    auto_data = mir_data._read_data("auto")

    assert auto_data.keys() == mir_data.auto_data.keys()

    for key in auto_data.keys():
        for subkey in ["data", "flags"]:
            assert np.array_equal(
                auto_data[key][subkey], mir_data.auto_data[key][subkey]
            )


@pytest.mark.filterwarnings("ignore", message=["No cross data", "No auto data"])
@pytest.mark.parametrize("data_type", ["none", "raw", "vis", "load", "no_auto"])
def test_mir_write_full(mir_data, tmp_path, data_type):
    """
    Mir write dataset tester.

    Make sure we can round-trip a MIR dataset correctly.
    """
    # We want to clear our the auto data here, since we can't _yet_ write that out
    mir_data.unload_data()
    if data_type == "no_auto":
        mir_data._clear_auto()

    if data_type in ["vis", "raw", "no_auto"]:
        mir_data.load_data(load_raw=(data_type == "raw"), apply_tsys=False)

    # Write out our test dataset
    filepath = os.path.join(tmp_path, "test_write_full_%s.mir" % data_type)

    mir_data.write(filepath, load_data=(data_type == "load"))
    with uvtest.check_warnings(
        None if (data_type != "none") else UserWarning,
        None if (data_type != "none") else ["No cross data", "No auto data"],
    ):
        mir_data.write(filepath, load_data=(data_type == "load"))

    # Read in test dataset.
    mir_copy = MirParser(filepath, has_auto=mir_data._has_auto)
    if data_type != "none":
        mir_copy.load_data(load_raw=(data_type in ["raw", "load"]), apply_tsys=False)

    # The objects won't be equal off the bat - a couple of things to handle first.
    assert mir_data != mir_copy

    # _file_dict has the filepath as a key, so we handle this in a special way.
    assert mir_data._file_dict.values() != mir_copy._file_dict.values()
    mir_data._file_dict = mir_copy._file_dict = None

    # Filename obviously _should_ be different...
    assert mir_data.filepath != mir_copy.filepath
    mir_data.filepath = mir_copy.filepath = None

    # Take care of some auto-specific stuff, which because we spoofed the original
    # ac_data attribute, won't be _exactly_ the same.
    if mir_copy._has_auto:
        assert mir_data._ac_dict.values() != mir_copy._ac_dict.values()
        mir_data._ac_dict = mir_copy._ac_dict = None
        assert np.any(mir_data.ac_data["dataoff"] != mir_copy.ac_data._data["dataoff"])
        mir_data.ac_data._data["dataoff"] = mir_copy.ac_data._data["dataoff"] = 0

    # Check for final equality with the above exceptions handled.
    assert mir_data == mir_copy


def test_compass_read_err(mir_data: MirParser, compass_soln_file):
    with pytest.raises(ValueError, match="Cannot call read_compass_solns"):
        mir_data.read_compass_solns(compass_soln_file)

    # MirParser will complain if data are already loaded when attempting to read in
    # the COMPASS solutions, so unloading it should allow it resolve the error above.
    mir_data.unload_data()
    mir_data.read_compass_solns(compass_soln_file)


def test_compass_flag_sphid_apply(mir_data: MirParser, compass_soln_file):
    """
    Test COMPASS per-sphid flagging.

    Test that applying COMPASS flags on a per-sphid basis works as expected.
    """
    # Unflag previously flagged data
    for entry in mir_data.vis_data.values():
        entry["flags"][:] = False

    assert mir_data._compass_bp_soln is None
    vis_data = mir_data.vis_data
    mir_data.vis_data = None
    mir_data.read_compass_solns(compass_soln_file, load_flags=True, load_bandpass=False)
    mir_data.vis_data = vis_data
    mir_data._apply_compass_solns(mir_data.vis_data)
    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data.get_value("corrchunk", header_key=key) != 0:
            assert not np.all(entry["flags"][1::2])
            assert np.any(entry["flags"][::2])


def test_compass_flag_static_apply(mir_data, compass_soln_file):
    """
    Test COMPASS static flagging.

    Test that applying COMPASS flags on a per-baseline (all time) basis works correctly.
    """
    # Make sure that a priori flags are preserved
    for entry in mir_data.vis_data.values():
        entry["flags"][:] = False
        entry["flags"][-1] = True

    mir_data.in_data["mjd"] += 1

    vis_data = mir_data.vis_data
    mir_data.vis_data = None
    with uvtest.check_warnings(
        UserWarning, "No metadata from COMPASS matches that in this data set."
    ):
        mir_data.read_compass_solns(
            compass_soln_file, load_flags=True, load_bandpass=False
        )

    assert mir_data._compass_bp_soln is None
    mir_data.vis_data = vis_data
    mir_data._apply_compass_solns(mir_data.vis_data)

    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data.get_value("corrchunk", header_key=key) != 0:
            assert np.all(entry["flags"][:8192])
            assert not np.any(entry["flags"][8192:-1])
            assert np.all(entry["flags"][-1])


@pytest.mark.parametrize("muck_solns", ["none", "some", "all"])
def test_compass_bp_apply(mir_data: MirParser, compass_soln_file, muck_solns):
    """
    Test COMPASS bandpass calibration.

    Test that applying COMPASS bandpass solutions works correctly.
    """
    tempval = np.complex64(1 + 1j)
    for entry in mir_data.vis_data.values():
        entry["data"][:] = tempval
        entry["flags"][:] = False

    if muck_solns != "none":
        mir_data.bl_data["iant1"] += 1
        if muck_solns == "all":
            mir_data.bl_data["iant2"] += 1

    vis_data = mir_data.vis_data
    mir_data.vis_data = None

    with uvtest.check_warnings(
        None if (muck_solns == "none") else UserWarning,
        None if (muck_solns == "none") else "No metadata from COMPASS matches",
    ):
        mir_data.read_compass_solns(
            compass_soln_file, load_flags=False, load_bandpass=True
        )

    assert mir_data._compass_static_flags is None
    assert mir_data._compass_sphid_flags is None

    mir_data.vis_data = vis_data

    mir_data._apply_compass_solns(mir_data.vis_data)

    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data.get_value("corrchunk", header_key=key) != 0:
            # If muck_solns is not some, then all the values should agree with our
            # temp value above, otherwise none should
            assert np.allclose(entry["data"], tempval * (1 + (muck_solns == "none")))
            assert (muck_solns != "none") == np.all(entry["flags"])


def test_compass_no_op(mir_data: MirParser, compass_soln_file):
    mir_data.read_compass_solns(
        compass_soln_file, load_flags=False, load_bandpass=False
    )
    assert not mir_data._has_compass_soln
    assert mir_data._compass_bp_soln is None
    assert mir_data._compass_sphid_flags is None
    assert mir_data._compass_static_flags is None


def test_compass_rechunk_routing(mir_data: MirParser, compass_soln_file):
    mir_data.unload_data()
    mir_data.read_compass_solns(compass_soln_file)
    mir_data.rechunk(16)
    mir_data.load_data()

    mir_copy = MirParser(
        mir_data.filepath, compass_soln=compass_soln_file, has_auto=True
    )
    mir_copy.load_data()
    mir_copy.rechunk(16)

    assert mir_data == mir_copy


@pytest.mark.parametrize(
    "field,comp,value,vis_keys",
    [
        ["mjd", "between", [60000.0, 50000.0], np.arange(1, 21)],
        ["source", "ne", "nosourcehere", np.arange(1, 21)],
        ["ant", "eq", 4, np.arange(1, 21)],
        ["ant1", "!=", 8, np.arange(1, 21)],
        ["ant1rx", "==", 0, [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]],
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
    with pytest.raises(ValueError, match="Cannot compare MirParser with int."):
        mir_data.__eq__(0)


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
                    "data": np.ones(2, dtype=np.complex64),
                    "flags": np.ones(4, dtype=bool),
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
        [False, "meta_attr", None, False],
        [False, "_compass_solns", None, False],
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
    elif "meta_attr" == mod_attr:
        del mir_copy._metadata_attrs["ac_data"]
    elif "_compass_solns" == mod_attr:
        mir_data._compass_solns = {1: {1: {1: {1: 1}}}}
        mir_copy._compass_solns = {}
    else:
        setattr(target_obj, mod_attr, mod_val)

    assert mir_data.__eq__(mir_copy, metadata_only=metadata_only) == exp_state

    assert mir_data.__ne__(mir_copy, metadata_only=metadata_only) != exp_state


def test_scan_int_headers_errs():
    """Verify _scan_int_headers throws errors when expected."""
    with pytest.raises(MirPackdataError, match="Cannot read scan start if headers"):
        MirParser._scan_int_headers(
            os.path.join(DATA_PATH, "sma_test.mir/autoCorrelations"),
            hdr_fmt=OLD_AUTO_HEADER,
        )


@pytest.mark.parametrize("use_dict", [True, False])
def test_scan_int_headers(use_dict):
    """Verify that we can correctly scan integration starting periods."""
    true_dict = {1: {"inhid": 1, "record_size": 1048680, "record_start": 0}}
    assert true_dict == MirParser._scan_int_headers(
        os.path.join(DATA_PATH, "sma_test.mir/sch_read"),
        hdr_fmt=NEW_VIS_HEADER,
        old_int_dict=true_dict if use_dict else None,
    )


def test_scan_int_header_bad_record_size():
    with pytest.raises(MirPackdataError, match="record_size was negative/invalid."):
        MirParser._scan_int_headers(
            os.path.join(DATA_PATH, "sma_test.mir/autoCorrelations"),
            hdr_fmt=OLD_AUTO_HEADER,
            old_int_dict={1: {"inhid": 1, "record_size": -120, "record_start": 0}},
        )


def test_scan_int_header_record_conflict():
    old_int_dict = {
        1: {"inhid": 1, "record_size": 1048680, "record_start": 120},
        2: {"inhid": 2, "record_size": 1048680, "record_start": 1048680},
    }
    new_dict = MirParser._scan_int_headers(
        os.path.join(DATA_PATH, "sma_test.mir/sch_read"),
        hdr_fmt=NEW_VIS_HEADER,
        old_int_dict=old_int_dict,
    )
    assert new_dict == {}


def test_fix_int_dict_auto(mir_data: MirParser):
    """Verify that fix_init_dict behaves as expected for autos."""
    # All the auto check can do is verify that things behave as expected when looking
    # at inhid, since there's no record size information in the file.
    file_dict_copy = copy.deepcopy(mir_data._file_dict)
    mir_data._fix_int_dict("auto")

    assert file_dict_copy == mir_data._file_dict


def test_fix_int_dict_cross(mir_data):
    """Verify that we can fix a "bad" integration start record."""
    bad_entry = {
        2: {"inhid": 1, "record_size": 120, "record_start": 120},
        3: {"inhid": 2, "record_size": 1048680, "record_start": 1048680},
    }

    good_dict = {
        mir_data.filepath: {
            "cross": {
                "int_dict": {
                    2: {"inhid": 1, "record_size": 1048680, "record_start": 0}
                },
                "filetype": "sch_read",
                "read_hdr_fmt": NEW_VIS_HEADER,
                "read_data_fmt": NEW_VIS_DTYPE,
                "common_scale": True,
            }
        }
    }

    # Muck with the records so that the inhid does not match that on disk.
    mir_data.sp_data._data["inhid"][:] = 2
    mir_data.bl_data._data["inhid"][:] = 2
    mir_data.in_data._data["inhid"][:] = 2
    mir_data.sp_data._data["nch"][:] = 1
    mir_data._sp_dict[2] = mir_data._sp_dict.pop(1)

    # Plug in the bad entry
    mir_data._file_dict = good_dict
    mir_data._file_dict[mir_data.filepath]["cross"]["int_dict"] = bad_entry.copy()
    # This should _hopefully_ generate the good dict
    mir_data._fix_int_dict("cross")
    assert good_dict == mir_data._file_dict

    # Plug in the bad entry again
    mir_data._file_dict[mir_data.filepath]["cross"]["int_dict"] = bad_entry.copy()
    with uvtest.check_warnings(UserWarning, "Values in int_dict do not match"):
        mir_data._read_data("cross", scale_data=False)

    assert good_dict == mir_data._file_dict

    # Attempt to load the data
    _ = mir_data._read_data("cross", scale_data=False)


@pytest.mark.parametrize(
    "kwargs,muck_int_dict,errfunc,errtype,errmsg",
    [
        [{"raise_err": True}, True, pytest.raises, MirPackdataError, "File indexing "],
        [{}, True, uvtest.check_warnings, UserWarning, "File indexing information "],
        [{"raise_err": True}, False, pytest.raises, ValueError, "inhid_arr contains "],
        [{}, False, uvtest.check_warnings, UserWarning, "inhid_arr contains keys not"],
    ],
)
def test_read_packdata_inhid_err(
    kwargs, muck_int_dict, errfunc, errtype, errmsg, mir_data
):
    if muck_int_dict:
        mir_data._file_dict[mir_data.filepath]["cross"]["int_dict"][1] = {
            "inhid": 2,
            "record_size": 1048680,
            "record_start": 0,
        }
        inhid_arr = [1]
    else:
        inhid_arr = [1, 2]
    with errfunc(errtype, match=errmsg):
        mir_data._read_packdata(mir_data._file_dict, inhid_arr=inhid_arr, **kwargs)


def test_read_packdata_mmap(mir_data):
    """Test that reading in vis data with mmap works just as well as np.read"""
    mmap_data = mir_data._read_packdata(
        mir_data._file_dict, mir_data.in_data["inhid"], use_mmap=True
    )

    reg_data = mir_data._read_packdata(
        mir_data._file_dict, mir_data.in_data["inhid"], use_mmap=False
    )

    assert mmap_data.keys() == reg_data.keys()
    for key in mmap_data.keys():
        assert np.array_equal(mmap_data[key], reg_data[key])


@pytest.mark.parametrize("attr", ["_make_packdata", "_read_data"])
def test_data_errs(mir_data, attr):
    with pytest.raises(ValueError, match="Argument for data_type not recognized"):
        getattr(mir_data, attr)(None)


@pytest.mark.parametrize(
    "compass_soln,kwargs,err_msg",
    [
        [False, {}, "Cannot apply calibration if no tables loaded."],
        [True, {"scale_data": False}, "Cannot return raw data if setting apply_cal=Tr"],
    ],
)
def test_read_data_errs(mir_data, compass_soln, kwargs, err_msg):
    mir_data._has_compass_soln = compass_soln
    with pytest.raises(ValueError, match=err_msg):
        mir_data._read_data("cross", apply_cal=True, **kwargs)


def test_read_packdata__make_packdata(mir_data: MirParser):
    """Verify that making packdata produces the same result as reading packdata"""
    mir_data.load_data(load_raw=True)

    _read_data = mir_data._read_packdata(
        mir_data._file_dict, mir_data.in_data["inhid"], "cross"
    )

    make_data = mir_data._make_packdata(
        mir_data._file_dict[mir_data.filepath]["cross"]["int_dict"],
        mir_data._sp_dict,
        mir_data.raw_data,
        "cross",
    )

    assert _read_data.keys() == make_data.keys()
    for key in _read_data.keys():
        assert np.array_equal(_read_data[key][0], make_data[key])


def test_apply_tsys_errs(mir_data):
    """
    Test that apply_tsys throws errors as expected.

    Note that we test these errors in sequence since it's a lot more efficient to do
    these operations on the same object one after another.
    """
    with pytest.raises(ValueError, match="Cannot apply tsys again "):
        mir_data.apply_tsys()

    mir_data.apply_tsys(invert=True)
    with pytest.raises(ValueError, match="Cannot undo tsys application if it was nev"):
        mir_data.apply_tsys(invert=True)

    mir_data.unload_data()
    with pytest.raises(ValueError, match="Must call load_data first before applying"):
        mir_data.apply_tsys(invert=True)


def test_apply_tsys_warn(mir_data):
    """Verify that apply_tsys throws warnings when tsys values aren't found."""
    with uvtest.check_warnings(UserWarning, "Changing fields that tie to header keys"):
        mir_data.eng_data["antenna"] = -1

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
        [np.all(data_dict["flags"]) for data_dict in mir_data.vis_data.values()]
    )


def test_apply_tsys_missing_recs(mir_data):
    mir_copy = mir_data.copy()
    mir_data.unload_data()
    mir_data.load_data(load_cross=True, apply_tsys=False)

    with uvtest.check_warnings(UserWarning, "Changing fields that tie to header"):
        mir_copy.eng_data["antenna"] = [2, 3]

    mir_copy.unload_data()
    with uvtest.check_warnings(UserWarning, ["No tsys for blhid"] * 4):
        mir_copy.load_data(load_cross=True, apply_tsys=True)

    for key in mir_data.vis_data:
        assert np.allclose(
            mir_data.vis_data[key]["data"], mir_copy.vis_data[key]["data"]
        )
        assert all(mir_copy.vis_data[key]["flags"])


@pytest.mark.parametrize("bad_vals", [False, True])
@pytest.mark.parametrize("use_cont_det", [True, False])
def test_apply_tsys(mir_data, use_cont_det, bad_vals):
    """Test that apply_tsys works on vis_data as expected."""
    mir_copy = mir_data.copy()
    mir_copy._tsys_use_cont_det = use_cont_det

    # Unload and load regular data without tsys application
    mir_data.unload_data()
    mir_data.load_data(load_cross=True, apply_tsys=False)

    if bad_vals:
        mir_copy.eng_data["tsys"] = 0.0
        mir_copy.eng_data["tsys_rx2"] = 0.0
        mir_copy.sp_data["wt"] = 0.0

        for idict in mir_data.vis_data.values():
            idict["flags"][:] = True
        norm_list = np.ones(20)
    elif use_cont_det:
        # Calculate the scaling factors directly. The factor of 2 comes from DSB -> SSB
        rxa_norm = mir_data.jypk * 2 * (np.prod(mir_data.eng_data["tsys"]) ** 0.5)
        rxb_norm = mir_data.jypk * 2 * (np.prod(mir_data.eng_data["tsys_rx2"]) ** 0.5)
        # The first 5 records should be rxa, and 5 rxb, then 5 rxa, then 5 rxb
        norm_list = np.array(
            [rxa_norm] * 5 + [rxb_norm] * 5 + [rxa_norm] * 5 + [rxb_norm] * 5
        )
    else:
        norm_list = (
            2
            * mir_data.jypk
            * (mir_data.sp_data["wt"] / mir_data.in_data["rinteg"][0]) ** -0.5
        )

    mir_copy.unload_data()
    mir_copy.load_data(load_cross=True, apply_tsys=True)

    for key, norm_fac in zip(mir_data.vis_data.keys(), norm_list):
        assert np.allclose(
            norm_fac * mir_data.vis_data[key]["data"], mir_copy.vis_data[key]["data"]
        )
        assert np.array_equal(
            mir_data.vis_data[key]["flags"], mir_copy.vis_data[key]["flags"]
        )

    mir_copy.apply_tsys(invert=True)
    for key in mir_data.vis_data.keys():
        assert np.allclose(
            mir_data.vis_data[key]["data"], mir_copy.vis_data[key]["data"]
        )
        assert np.array_equal(
            mir_data.vis_data[key]["flags"], mir_copy.vis_data[key]["flags"]
        )


def test_apply_flags_err(mir_data):
    mir_data.unload_data()
    with pytest.raises(ValueError, match="Cannot apply flags if vis_data are not load"):
        mir_data.apply_flags()


@pytest.mark.parametrize("sphid_arr", [[1], list(range(1, 21)), [10, 15]])
def test_apply_flags(mir_data, sphid_arr):
    mir_data.sp_data.set_value("flags", 1, header_key=sphid_arr)
    mir_data.apply_flags()
    for key, value in mir_data.vis_data.items():
        assert np.all(value["flags"]) == (key in sphid_arr)


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
        mir_data.raw_data = mir_data._convert_vis_to_raw(mir_data.vis_data)
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

    # If we down-selected, make sure we plug back in the original data.
    if select_vis or select_raw:
        mir_data.sp_data._mask[:] = True
    if select_auto:
        mir_data.ac_data._mask[:] = True

    # Make sure that the metadata all look good.
    assert mir_data.__eq__(mir_copy, metadata_only=True)

    if select_vis or select_auto or select_raw:
        with pytest.raises(MirMetaError, match="Missing spectral records in data attr"):
            mir_data._downselect_data(
                select_vis=select_vis, select_raw=select_raw, select_auto=select_auto
            )

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
    mir_data.raw_data = {1: {"data": [1] * 16384, "scale_fac": [1]}}

    mir_data.unload_data(
        unload_vis=unload_vis, unload_raw=unload_raw, unload_auto=unload_auto
    )

    assert mir_data.vis_data is None if unload_vis else mir_data.vis_data is not None
    assert mir_data._tsys_applied != unload_vis

    assert mir_data.raw_data is None if unload_raw else mir_data.raw_data is not None

    assert mir_data.auto_data is None if unload_auto else mir_data.auto_data is not None


@pytest.mark.parametrize(
    "kwargs,err_type,err_msg",
    [[{"load_auto": True}, ValueError, "This object has no auto-correlation data"]],
)
def test_load_data_err(mir_data, kwargs, err_type, err_msg):
    mir_data._clear_auto()

    with pytest.raises(err_type, match=err_msg):
        mir_data.load_data(**kwargs)


@pytest.mark.parametrize(
    "optype,kwargs,warn_msg",
    [
        ["load_raw", {"load_cross": True}, "Converting previously loaded data since"],
        ["muck_vis", {"allow_downselect": True}, "Cannot downselect cross-correlation"],
        ["muck_auto", {"allow_downselect": True}, "Cannot downselect auto-correlation"],
    ],
)
def test_load_data_warn(mir_data, optype, kwargs, warn_msg):
    if optype == "load_raw":
        mir_data.load_data(load_raw=True, load_cross=True, load_auto=False)
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
    mir_data.load_data(load_cross=load_vis)

    assert (mir_data.vis_data is not None) == (load_vis is None)
    assert mir_data._tsys_applied == (load_vis is None)
    assert mir_data.raw_data is None

    assert mir_data.auto_data is not None


def test_load_data_conv(mir_data):
    """Test that the conversion operation of load_data operates as expected."""
    mir_copy = mir_data.copy()

    mir_data.unload_data()
    assert mir_data.vis_data is None

    mir_data.load_data(load_cross=True, allow_conversion=False)
    mir_copy.load_data(load_cross=True, allow_conversion=True)

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
        [True, ["No cross data loaded,", "No auto data loaded,"]],
    ],
)
def test_write_warn(mir_data, tmp_path, unload_data, warn_msg):
    """Test that write throws errors as expected."""
    testfile = os.path.join(
        tmp_path, "test_write_warn_%s.mir" % ("meta" if unload_data else "tsysapp")
    )
    if unload_data:
        mir_data.unload_data()

    with uvtest.check_warnings(UserWarning, warn_msg):
        mir_data.write(testfile)

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
        5: {"data": np.arange(-16384, 16384, dtype=np.int16), "scale_fac": np.int16(1)}
    }

    # First up, test what should be a no-op
    raw_copy = MirParser._rechunk_raw(raw_data, [1], inplace=inplace)

    assert (raw_copy is raw_data) == inplace

    assert raw_data.keys() == raw_copy.keys()
    assert raw_data[5]["scale_fac"] == 1
    assert np.all(raw_data[5]["data"] == np.arange(-16384, 16384))

    # Now let's actually do some averaging and make sure it works as expected.
    raw_copy = MirParser._rechunk_raw(raw_data, [2], inplace=inplace)
    assert (raw_copy is raw_data) == inplace
    # Scale factor drops on account of having gotten rid of one sig binary digit
    # through the averaging process
    assert raw_copy[5]["scale_fac"] == 0
    # This is what raw_data _should_ look like after averaging. Note two arange arrays
    # are used here because the spacing for real _or_ imag is the same, but not real
    # and imag together.
    assert np.all(
        raw_copy[5]["data"]
        == np.vstack(
            (np.arange(-32766, 32768, 8), np.arange(-32764, 32768, 8))
        ).T.flatten()
    )
    raw_data = raw_copy

    # Finally, test that flagging works as expected
    raw_data[5]["data"][2:] = -32768  # Marks channel as flagged
    raw_copy = MirParser._rechunk_raw(raw_data, [4096], inplace=inplace)
    assert (raw_copy is raw_data) == inplace
    # Scale factor should not change
    assert raw_copy[5]["scale_fac"] == 0
    # First channel should just contain channel 1 data, second channel should be flagged
    assert np.all(raw_copy[5]["data"] == [-32766, -32764, -32768, -32768])


@pytest.mark.parametrize("inplace", [True, False])
def test_rechunk_cross(inplace):
    """Test that rechunk_raw properly averages data"""
    # Chicago FTW!
    vis_data = {
        25624: {
            "data": (np.arange(1024) + np.flip(np.arange(1024) * 1j)),
            "flags": np.zeros(1024, dtype=bool),
            "weights": np.ones(1024, dtype=np.float32),
        }
    }
    check_vals = np.arange(1024) + np.flip(np.arange(1024) * 1j)
    vis_orig = copy.deepcopy(vis_data)

    # First up, test no averaging
    vis_copy = MirParser._rechunk_data(vis_data, [1], inplace=inplace)

    assert (vis_copy is vis_data) == inplace

    assert vis_data.keys() == vis_copy.keys()
    assert np.all(vis_data[25624]["flags"] == np.zeros(1024, dtype=bool))
    assert np.all(vis_data[25624]["data"] == check_vals)
    assert np.all(vis_data[25624]["weights"] == np.ones(1024))

    # Next, test averaging w/o flags and weighting
    vis_copy = MirParser._rechunk_data(vis_data, [4], inplace=inplace, weight_data=True)
    check_vals = np.mean(check_vals.reshape(256, 4), axis=1)

    assert (vis_copy is vis_data) == inplace
    assert vis_data.keys() == vis_copy.keys()
    assert np.all(vis_copy[25624]["flags"] == np.zeros(256, dtype=bool))
    assert np.all(vis_copy[25624]["data"] == check_vals)
    assert np.all(vis_copy[25624]["weights"] == np.ones(256))
    vis_data = copy.deepcopy(vis_orig)

    # Next, test averaging w/o weighting and w/o flags
    vis_copy = MirParser._rechunk_data(
        vis_data, [4], inplace=inplace, weight_data=False
    )
    assert np.all(vis_copy[25624]["flags"] == np.zeros(256, dtype=bool))
    assert np.all(vis_copy[25624]["data"] == check_vals)
    assert np.all(vis_copy[25624]["weights"] == np.ones(256))
    vis_data = copy.deepcopy(vis_orig)

    # Check what happens if we flag data
    vis_data[25624]["flags"][4:] = True
    vis_copy = MirParser._rechunk_data(
        vis_data, [512], inplace=inplace, norm_weights=False
    )
    assert (vis_copy is vis_data) == inplace
    assert vis_data.keys() == vis_copy.keys()
    assert np.all(vis_copy[25624]["flags"] == [False, True])
    assert np.all(vis_copy[25624]["weights"] == [4.0, 0.0])
    assert np.all(vis_copy[25624]["data"] == [check_vals[0], 0.0])
    vis_data = copy.deepcopy(vis_orig)

    # And lastly, check what happens if weights are not rechunked (nsamples weights)
    vis_data[25624]["flags"][4:] = True
    vis_copy = MirParser._rechunk_data(
        vis_data, [512], inplace=inplace, norm_weights=True
    )
    assert (vis_copy is vis_data) == inplace
    assert vis_data.keys() == vis_copy.keys()
    assert np.all(vis_copy[25624]["flags"] == [False, True])
    assert np.all(vis_copy[25624]["data"] == [check_vals[0], 0.0])
    assert np.all(vis_copy[25624]["weights"] == [1 / 128, 0.0])


@pytest.mark.parametrize("inplace", [True, False])
def test_rechunk_auto(inplace):
    auto_data = {
        8675309: {
            "data": np.arange(-1024, 1024, dtype=np.float32),
            "flags": np.zeros(2048, dtype=bool),
            "weights": np.ones(2048, dtype=np.float32),
        }
    }

    # First up, test no averaging
    auto_copy = MirParser._rechunk_data(auto_data, [1], inplace=inplace)
    assert (auto_copy is auto_data) == inplace
    assert auto_data.keys() == auto_copy.keys()
    assert np.all(auto_copy[8675309]["data"] == np.arange(-1024, 1024))
    assert np.all(auto_copy[8675309]["weights"] == np.ones(2048))

    # First up, test no averaging
    auto_copy = MirParser._rechunk_data(auto_data, [512], inplace=inplace)
    assert (auto_copy is auto_data) == inplace
    assert auto_data.keys() == auto_copy.keys()
    assert np.all(auto_copy[8675309]["data"] == [-768.5, -256.5, 255.5, 767.5])
    assert np.all(auto_copy[8675309]["weights"] == np.ones(4))


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

    # Rather than parametrize this, because the underlying object isn't changed,
    # check for the different load states here, since the error should get thrown
    # no matter which thing you are loading.
    with pytest.raises(err_type, match=err_msg):
        mir_data.rechunk(chan_avg)


def test_rechunk_nop(mir_data):
    """Test that setting chan_avg to 1 doesn't change the object."""
    mir_copy = mir_data.copy()

    mir_data.rechunk(1)
    assert mir_data == mir_copy


def test_rechunk_on_the_fly(mir_data):
    mir_data.rechunk(8)
    mir_copy = mir_data.copy()

    mir_copy.unload_data()
    mir_copy.load_data(load_cross=True, load_auto=True)

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
        mir_copy.load_data(allow_conversion=True, load_cross=True)

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
        [["add_file"], {"merge": True}, ValueError, "These two objects were"],
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
    if "add_file" in muck_data:
        mir_data._file_dict["foo.mir"] = {}
    if "antpos" in muck_data:
        mir_data.antpos_data["xyz_pos"] = 0.0
    if "all" in muck_data:
        mir_data = np.arange(100)

    with pytest.raises(err_type) as err:
        mir_data.__add__(mir_copy, **kwargs)
    if not ("all" in muck_data):
        assert str(err.value).startswith(err_msg)

    with pytest.raises(err_type, match=err_msg):
        mir_copy.__add__(mir_data, **kwargs)


def test_add_merge(mir_data):
    """
    Verify that the __add__ method behaves as expected under 'simple' scenarios, i.e.,
    where overwrite or force are not necessary.
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
    mir_data.load_data(load_cross=True, load_auto=True, apply_tsys=True)
    mir_copy.load_data(load_cross=True, load_auto=True, apply_tsys=True)

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
    mir_data.load_data(load_cross=True, apply_tsys=True)
    mir_copy.load_data(load_cross=True, apply_tsys=True)

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
    mir_data.load_data(load_cross=True, apply_tsys=True)
    assert mir_data == mir_orig


@pytest.mark.parametrize("drop_auto", [True, False])
@pytest.mark.parametrize("drop_raw", [True, False])
@pytest.mark.parametrize("drop_vis", [True, False, "jypk", "tsys"])
def test_add_drop_data(mir_data, drop_auto, drop_raw, drop_vis):
    mir_data.raw_data = mir_data._convert_vis_to_raw(mir_data.vis_data)
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
                    getattr(mir_data, item)[field] = 255
    elif muck_attr == "codes":
        mir_data.codes_data.set_value("code", "1", where=("v_name", "eq", "filever"))
    else:
        for field in getattr(mir_data, muck_attr).dtype.names:
            if field not in prot_fields:
                getattr(mir_data, muck_attr)[field] = 255

    # After mucking, verify that at least something looks different
    assert mir_data != mir_copy

    # mir_copy contains the good data, so adding it second will overwrite the bad data.
    assert mir_data.__add__(mir_copy, overwrite=True) == mir_copy

    # On the other hand, if we add mir_data second, the bad values should get propagated
    assert mir_copy.__add__(mir_data, overwrite=True) == mir_data


def test_add_concat_warn(mir_data, tmp_path):
    filepath = os.path.join(tmp_path, "add_concat_warn")

    with uvtest.check_warnings(UserWarning, "Writing out raw data with tsys applied."):
        mir_data.write(filepath)

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
        "antrx",
        "ant1rx",
        "ant2rx",
        "isb",
    ]

    for item in ["bl_data", "eng_data", "in_data", "sp_data", "we_data", "ac_data"]:
        for field in getattr(mir_copy, item).dtype.names:
            if field not in prot_fields:
                getattr(mir_copy, item)[field] = 156

    mir_copy.codes_data.set_value("code", "3c279", where=("v_name", "eq", "source"))

    with uvtest.check_warnings(UserWarning, "Writing out raw data with tsys applied."):
        mir_copy.write(filepath, overwrite=True)
    # Read the file in so that we have a dict here to work with.
    mir_copy.read(filepath, load_cross=True, has_auto=True)

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
        ["cubic", -1, ValueError, "tol must be between 0 and 0.5."],
        ["abc", 0.5, ValueError, 'Kernel type of "abc" not recognized,'],
    ],
)
def test_generate_chanshift_kernel_errs(kern_type, tol, err_type, err_msg):
    """ "Verify that _generate_chanshift_kernel throws errors as expected."""
    with pytest.raises(err_type, match=err_msg):
        MirParser._generate_chanshift_kernel(1.5, kern_type, tol=tol)


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
        123: {"data": np.array(raw_vals, dtype=np.int16), "scale_fac": np.int16(0)}
    }

    # Test no-op
    new_dict = MirParser._chanshift_raw(
        raw_dict, [(0, 0, None)], inplace=inplace, return_vis=return_vis
    )
    if inplace:
        assert new_dict is raw_dict
    if return_vis:
        new_dict = MirParser._convert_vis_to_raw(new_dict)

    assert np.all(raw_vals == new_dict[123]["data"])
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
        new_dict = MirParser._convert_vis_to_raw(new_dict)

    good_slice = slice(None if fwd_dir else 2, -2 if fwd_dir else None)
    flag_slice = slice(None if fwd_dir else -2, 2 if fwd_dir else None)
    # Note that the shift of 2 is required since each channel has a real and imag
    # component. The first two entries are dropped because they _should_ be flagged.
    assert np.all(
        raw_vals[good_slice]
        == np.roll(new_dict[123]["data"], -2 if fwd_dir else 2)[good_slice]
    )
    assert np.all(new_dict[123]["data"][flag_slice] == -32768)
    assert new_dict[123]["scale_fac"] == 0

    # Refresh the values, in case we are doing this in-place
    if inplace:
        raw_dict = {
            123: {"data": np.array(raw_vals, dtype=np.int16), "scale_fac": np.int16(0)}
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
        new_dict = MirParser._convert_vis_to_raw(new_dict)

    if fwd_dir:
        assert np.all(new_dict[123]["data"][14:16] == (32767 if check_flags else 0))
        assert np.all(
            new_dict[123]["data"][10:14] == (-32768 if check_flags else 32767)
        )
        assert np.all(new_dict[123]["data"][4:10] == (32767 if check_flags else 0))
        assert np.all(new_dict[123]["data"][0:4] == -32768)
    else:
        assert np.all(new_dict[123]["data"][0:4] == (32767 if check_flags else 0))
        assert np.all(new_dict[123]["data"][4:8] == (-32768 if check_flags else 32767))
        assert np.all(new_dict[123]["data"][8:12] == (32767 if check_flags else 0))
        assert np.all(new_dict[123]["data"][12:16] == -32768)
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
    weight_vals = np.ones(8)
    weight_vals[flag_vals] = 0.0
    vis_dict = {
        456: {
            "data": np.array(vis_vals, dtype=np.complex64),
            "flags": np.array(flag_vals, dtype=bool),
            "weights": np.array(weight_vals, dtype=np.float32),
        }
    }

    # Test no-op
    new_dict = MirParser._chanshift_vis(
        vis_dict, [(0, 0, None)], flag_adj=flag_adj, inplace=inplace
    )

    if inplace:
        assert new_dict is vis_dict

    assert np.all(vis_vals == new_dict[456]["data"])
    assert np.all(flag_vals == new_dict[456]["flags"])
    assert np.all(weight_vals == new_dict[456]["weights"])

    # Now try a simple one-channel shift
    new_dict = MirParser._chanshift_vis(
        vis_dict, [(1 if fwd_dir else -1, 0, None)], flag_adj=flag_adj, inplace=inplace
    )

    if inplace:
        assert new_dict is vis_dict

    good_slice = slice(None if fwd_dir else 1, -1 if fwd_dir else None)
    flag_slice = slice(None if fwd_dir else -1, 1 if fwd_dir else None)

    assert np.all(
        vis_vals[good_slice]
        == np.roll(new_dict[456]["data"], -1 if fwd_dir else 1)[good_slice]
    )
    assert np.all(
        flag_vals[good_slice]
        == np.roll(new_dict[456]["flags"], -1 if fwd_dir else 1)[good_slice]
    )
    assert np.all(
        weight_vals[good_slice]
        == np.roll(new_dict[456]["weights"], -1 if fwd_dir else 1)[good_slice]
    )

    assert np.all(new_dict[456]["data"][flag_slice] == 0.0)
    assert np.all(new_dict[456]["flags"][flag_slice])
    assert np.all(new_dict[456]["weights"][flag_slice] == 0.0)

    # Refresh the values, in case we are doing this in-place
    if inplace:
        vis_dict = {
            456: {
                "data": np.array(vis_vals, dtype=np.complex64),
                "flags": np.array(flag_vals, dtype=bool),
                "weights": np.array(weight_vals, dtype=np.float32),
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
    exp_weights = np.roll(weight_vals, 2 if fwd_dir else -2)
    exp_vals[None if fwd_dir else -2 : 2 if fwd_dir else None] = 0.0
    exp_flags[None if fwd_dir else -2 : 2 if fwd_dir else None] = True
    exp_weights[None if fwd_dir else -2 : 2 if fwd_dir else None] = 0.0
    mod_slice = slice(4 - (-1 if fwd_dir else 2), 6 - (-1 if fwd_dir else 2))
    if flag_adj:
        exp_flags[mod_slice] = check_flags
        exp_vals[mod_slice] = 0 if check_flags else [check_val * 0.75, check_val * 0.25]
        exp_weights[mod_slice] = 0 if check_flags else 1
    else:
        exp_vals[mod_slice] = [check_val * 0.25, check_val * 0.75]
        exp_flags[mod_slice] = False
        exp_weights[mod_slice] = [0.25, 0.75]

    assert np.all(new_dict[456]["data"] == exp_vals)
    assert np.all(new_dict[456]["flags"] == exp_flags)
    assert np.all(new_dict[456]["weights"] == exp_weights)


@pytest.mark.parametrize(
    "filever,irec,err_type,err_msg",
    [
        ["2", 3, ValueError, "MIR file format < v4.0 detected,"],
        ["4", 3, ValueError, "Receiver code 3 not recognized."],
    ],
)
def test_redoppler_data_errs(mir_data, filever, irec, err_type, err_msg):
    """Verify that redoppler_data throws errors as expected."""
    mir_data.codes_data.set_value("code", filever, where=("v_name", "eq", "filever"))
    mir_data.bl_data["irec"] = irec

    with pytest.raises(err_type, match=err_msg):
        mir_data.redoppler_data()


@pytest.mark.parametrize("plug_vals", [True, False])
@pytest.mark.parametrize("diff_rx", [True, False])
@pytest.mark.parametrize("use_raw", [True, False])
def test_redoppler_data(mir_data, plug_vals, diff_rx, use_raw):
    """Verify that redoppler_data behaves as expected."""
    # We have to spoof the filever because the test file is technically v3
    mir_data.codes_data.set_value("code", "4", where=("v_name", "eq", "filever"))

    if use_raw:
        mir_data.raw_data = mir_data._convert_vis_to_raw(mir_data.vis_data)
        mir_data.vis_data = None

    mir_copy = mir_data.copy()
    # This first attempt should basically just be a no-op
    mir_copy.redoppler_data()

    assert mir_data == mir_copy

    # Alright, let's tweak the data now to give us something to compare
    for sphid, nch in zip(mir_data.sp_data["sphid"], mir_data.sp_data["nch"]):
        if use_raw:
            mir_data.raw_data[sphid]["data"][:] = np.arange(nch * 2)
            mir_data.raw_data[sphid]["scale_fac"] = np.int16(0)
        else:
            mir_data.vis_data[sphid]["data"][:] = np.arange(nch)
            mir_data.vis_data[sphid]["flags"][:] = False

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
                assert np.all(mir_data.raw_data[sphid]["data"] == np.arange(nch * 2))
            else:
                assert np.all(mir_data.vis_data[sphid]["data"] == np.arange(nch))
        elif chan_shift < 0:
            if use_raw:
                assert np.all(
                    mir_data.raw_data[sphid]["data"][: chan_shift * 2]
                    == np.arange(-(2 * chan_shift), nch * 2)
                )
            else:
                assert np.all(
                    mir_data.vis_data[sphid]["data"][:chan_shift]
                    == np.arange(-chan_shift, nch)
                )
        else:
            if use_raw:
                assert np.all(
                    mir_data.raw_data[sphid]["data"][chan_shift * 2 :]
                    == np.arange((nch - chan_shift) * 2)
                )
            else:
                assert np.all(
                    mir_data.vis_data[sphid]["data"][chan_shift:]
                    == np.arange((nch - chan_shift))
                )


def test_fix_acdata(mir_data):
    # So we have to do a bit of metadata manipulation here in order to make this work
    # for total test coverage. Spoof a dataset where there's only 1 sideband but two
    # integrations. First up - just double the number of int headers.
    mir_data.in_data._data = np.tile(mir_data.in_data._data, 2)
    mir_data.in_data._data["inhid"] = [1, 2]
    mir_data.in_data._mask = np.tile(mir_data.in_data._mask, 2)
    mir_data.in_data._set_header_key_index_dict()

    # Now, anything that's LSB, make that part of integration #2 and USB
    sel_mask = mir_data.bl_data._data["isb"] == 0
    mir_data.bl_data._data["isb"][sel_mask] = 1
    mir_data.bl_data._data["inhid"][sel_mask] = 2

    # Finally, duplicate ac_data to be double te size, mapping the new entries to
    # integration #2.
    mir_data.ac_data._data = np.tile(mir_data.ac_data._data, 2)
    mir_data.ac_data._data["inhid"][16:] = 2
    mir_data.ac_data._mask = np.tile(mir_data.ac_data._mask, 2)
    mir_data.ac_data._set_header_key_index_dict()

    # Finally, call fix_acdata, which should (among other things), appropriately fill
    # in the frequency information.
    mir_data._fix_acdata()

    # Now check that fsky is correctly set
    assert np.array_equal(
        mir_data.ac_data["fsky"][:16],
        np.tile(np.repeat(mir_data.sp_data["fsky"][11:15], 2), 2),
    )
    assert np.array_equal(
        mir_data.ac_data["fsky"][16:],
        np.tile(np.repeat(mir_data.sp_data["fsky"][1:5], 2), 2),
    )

    # Make sure these values are actually different
    assert np.all(
        ~np.isin(mir_data.ac_data["fsky"][:16], mir_data.sp_data["fsky"][1:5])
    )


@pytest.mark.parametrize(
    "mask_name,errtype,errmsg",
    [
        [123, ValueError, "mask_name must be a string."],
        ["duplicate", ValueError, "There already exists a stored set of masks with"],
    ],
)
def test_mir_save_mask_err(mir_data: MirParser, mask_name, errtype, errmsg):
    if mask_name == "duplicate":
        mir_data.save_mask("duplicate")
    with pytest.raises(errtype, match=errmsg):
        mir_data.save_mask(mask_name)


@pytest.mark.parametrize(
    "mask_name,errtype,errmsg",
    [
        ["123", ValueError, "No stored masks for this object."],
        ["nomatch", ValueError, "No stored set of masks with the name"],
    ],
)
def test_mir_restore_mask_err(mir_data: MirParser, mask_name, errtype, errmsg):
    if mask_name == "nomatch":
        mir_data.save_mask("test")
    with pytest.raises(errtype, match=errmsg):
        mir_data.restore_mask(mask_name)


def test_mir_save_restore_mask_loop(mir_data: MirParser):
    """Simple check to make sure saving/restoring of masks work as expected."""
    mir_data.save_mask("start")
    mir_copy = mir_data.copy()
    # Deselect all records. Skip updating the data here to make the later comparison
    # a little bit easier.
    mir_copy.select(("inhid", "ne", 1), update_data=False)

    # Make sure that an update actually happened.
    assert mir_data != mir_copy

    # Restore the old masks, make sure that the objects are now the same.
    mir_copy.restore_mask("start")
    assert mir_data == mir_copy


@pytest.mark.parametrize("from_file", [False, True])
def test_mir_fix_v3_noop(mir_data, from_file):
    # The test file is already v3, so this should be identical
    if from_file:
        mir_copy = MirParser(
            mir_data.filepath,
            make_v3_compliant=True,
            nchunks=8,
            has_auto=True,
            load_auto=True,
            load_cross=True,
        )
    else:
        mir_copy = mir_data.copy()
        mir_copy._make_v3_compliant()

    assert mir_copy == mir_data


@pytest.mark.parametrize("muck_antrx", [False, True])
def test_mir_fix_v3(mir_data, muck_antrx):
    mir_data = MirParser()._load_test_data(has_auto=True)
    mir_copy = mir_data.copy()
    mir_copy.codes_data._data[0]["code"] = "2"
    if muck_antrx:
        mir_copy.bl_data["ant1rx"] = mir_copy.bl_data["ant2rx"] = 0

    with uvtest.check_warnings(UserWarning, "Pre v.3 MIR file format detected"):
        mir_copy._make_v3_compliant()

    assert mir_copy.codes_data["filever"] == ["2"]
    mir_copy.codes_data._data[0]["code"] = "3"

    for item in ["lst", "ara", "adec", "mjd"]:
        assert np.isclose(mir_copy.in_data[item], mir_data.in_data[item], atol=3e-4)
        mir_copy.in_data[item] = mir_data.in_data[item]

    assert mir_copy == mir_data


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
        assert len(mir_data.ac_data) == 16
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
    what we know to be 'true' at the time of observations.
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
    assert np.all(mir_data.eng_data["antenna"] == [1, 4])

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

        assert np.all(mir_data.ac_data["achid"] == np.arange(1, 17))

        assert np.all(mir_data.ac_data["antenna"] == ([1] * 8) + ([4] * 8))

    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto


def test_mir_remember_me_sp_data(mir_data):
    """
    Mir sp_read checker.

    Make sure that certain values in the sp_read file of the test data set match what
    we know to be 'true' at the time of observations. This includes values that were
    spare at time of observation (and thus stored as zero), but have since been assigned
    in subsequent versions. The primary goal in this case is to ensure that all keys
    in the dtype exist as currently expected.
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

    assert np.all(mir_data.sp_data["gunnMult"] == 0)

    assert np.all(mir_data.sp_data["amp"] == 0)

    assert np.all(mir_data.sp_data["phase"] == 0)

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
        np.all(sp_raw["data"] == check_arr) if (np.mod(idx, 5) == 0) else True
        for idx, sp_raw in enumerate(mir_data.raw_data.values())
    )


def test_mir_parser_read_path_vs_str():
    from pathlib import Path

    sma_data_path = str(os.path.join(DATA_PATH, "sma_test.mir"))
    sma_str_init = MirParser(
        sma_data_path, load_cross=True, load_auto=True, has_auto=True
    )
    sma_path_init = MirParser(
        Path(sma_data_path), load_cross=True, load_auto=True, has_auto=True
    )
    assert sma_str_init == sma_path_init
