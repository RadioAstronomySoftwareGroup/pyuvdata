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
from ..mir_parser import MirParser
from ... import tests as uvtest
import copy


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


def test_mir_parser_index_uniqueness(mir_data):
    """
    Mir index uniqueness check

    Make sure that there are no duplicate indicies for things that are primary keys
    for the various table-like structures that are used in MIR
    """
    inhid_list = mir_data._in_read["inhid"]
    assert np.all(np.unique(inhid_list) == sorted(inhid_list))

    blhid_list = mir_data._bl_read["blhid"]
    assert np.all(np.unique(blhid_list) == sorted(blhid_list))

    sphid_list = mir_data._sp_read["sphid"]
    assert np.all(np.unique(sphid_list) == sorted(sphid_list))


def test_mir_parser_index_valid(mir_data):
    """
    Mir index validity check

    Make sure that all indexes are non-negative
    """
    assert np.all(mir_data._in_read["inhid"] >= 0)

    assert np.all(mir_data._bl_read["blhid"] >= 0)

    assert np.all(mir_data._sp_read["sphid"] >= 0)


def test_mir_parser_index_linked(mir_data):
    """
    Mir index link check

    Make sure that all referenced indicies have matching pairs in their parent tables
    """
    inhid_set = set(np.unique(mir_data._in_read["inhid"]))

    # Should not exist is has_auto=False
    # See `mir_data_object` above.
    if mir_data._ac_read is not None:
        assert set(np.unique(mir_data._ac_read["inhid"])).issubset(inhid_set)
    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto

    assert set(np.unique(mir_data._bl_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data._eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data._eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data._sp_read["inhid"])).issubset(inhid_set)

    blhid_set = set(np.unique(mir_data._bl_read["blhid"]))

    assert set(np.unique(mir_data._sp_read["blhid"])).issubset(blhid_set)


def test_mir_parser_unload_data(mir_data):
    """
    Check that the unload_data function works as expected
    """
    attr_list = ["vis_data", "raw_data", "auto_data"]

    for attr in attr_list:
        assert getattr(mir_data, attr) is not None

    mir_data.unload_data()

    for attr in attr_list:
        assert getattr(mir_data, attr) is None


@pytest.mark.parametrize("filter_type", ["use_in", "use_bl", "use_sp"])
def test_mir_parser_update_filter(mir_data, filter_type):
    """
    Verify that filtering operations work as expected.
    """
    keywarg = {filter_type: []}
    mir_data._update_filter(**keywarg)

    attr_list = ["in_data", "bl_data", "eng_data", "sp_data", "ac_data"]
    for attr in attr_list:
        assert len(getattr(mir_data, attr)) == 0


def test_mir_auto_read(mir_data):
    """
    Mir read tester

    Make sure that Mir autocorrelations are read correctly
    """
    mir_data.fromfile(mir_data.filepath, has_auto=True)

    with pytest.raises(IndexError) as cm:
        ac_data = mir_data.scan_auto_data(mir_data.filepath, nchunks=999)
    str(cm.value).startswith("Could not determine auto-correlation record size!")

    ac_data = mir_data.scan_auto_data(mir_data.filepath)
    assert np.all(ac_data["nchunks"] == 8)
    int_start_dict = {inhid: None for inhid in np.unique(ac_data["inhid"])}

    mir_data.load_data(load_vis=False, load_auto=True)

    # Select the relevant auto records, which should be for spwin 0-3
    auto_data = mir_data.read_auto_data(mir_data.filepath, int_start_dict, ac_data)

    for ac1, ac2 in zip(auto_data, mir_data.auto_data):
        assert np.all(
            np.logical_or(ac1 == ac2, np.logical_and(np.isnan(ac1), np.isnan(ac2)))
        )
    mir_data.unload_data()


@pytest.mark.parametrize(
    "attr,read_func,write_func",
    [
        ["_antpos_read", "read_antennas", "write_antennas"],
        ["_bl_read", "read_bl_data", "write_bl_data"],
        ["_codes_read", "read_codes_data", "write_codes_data"],
        ["_eng_read", "read_eng_data", "write_eng_data"],
        ["_in_read", "read_in_data", "write_in_data"],
        ["_sp_read", "read_sp_data", "write_sp_data"],
        ["_we_read", "read_we_data", "write_we_data"],
    ],
)
def test_mir_write_item(mir_data, attr, read_func, write_func, tmp_path):
    """
    Mir write tester.

    Test writing out individual components of the metadata of a MIR dataset.
    """
    filepath = os.path.join(tmp_path, "test_write%s" % attr)
    orig_attr = getattr(mir_data, attr)
    getattr(mir_data, write_func)(filepath, orig_attr)
    check_attr = getattr(mir_data, read_func)(filepath)
    assert np.array_equal(orig_attr, check_attr)


def test_mir_raw_data(mir_data, tmp_path):
    """
    Test reading and writing of raw data.
    """
    filepath = os.path.join(tmp_path, "test_write_raw")
    mir_data.write_rawdata(filepath, mir_data.raw_data, mir_data.sp_data)
    int_start_dict = mir_data.scan_int_start(
        filepath, allowed_inhid=mir_data.in_data["inhid"]
    )

    (raw_data, _) = mir_data.read_vis_data(
        filepath, int_start_dict, mir_data.sp_data, return_raw=True
    )

    assert raw_data.keys() == mir_data.raw_data.keys()

    for key in raw_data.keys():
        for subkey in ["raw_data", "scale_fac"]:
            assert np.array_equal(raw_data[key][subkey], mir_data.raw_data[key][subkey])


@pytest.mark.parametrize("data_type", ["none", "raw", "vis"])
def test_mir_write_full(mir_data, tmp_path, data_type):
    """
    Mir write dataset tester.

    Make sure we can round-trip a MIR dataset correctly.
    """
    # We want to clear our the auto data here, since we can't _yet_ write that out
    mir_data.unload_data()
    if data_type == "vis":
        mir_data.load_data(load_vis=True, apply_tsys=False)

    mir_data._has_auto = False
    mir_data._ac_filter = mir_data._ac_read = mir_data.ac_data = None

    # We're doing a bit of slight-of-hand here to account for the fact that the ordering
    # of codes_read does not matter for data handling (and we can't easily recreate
    # that ordering). All this does is make the ordering consistent.
    mir_data._codes_read = mir_data.make_codes_read(mir_data.codes_dict)

    # Write out our test dataset
    filepath = os.path.join(tmp_path, "test_write_full_%s.mir" % data_type)

    with uvtest.check_warnings(
        None if (data_type != "none") else UserWarning,
        None if (data_type != "none") else "No data loaded, writing metadata only",
    ):
        mir_data.tofile(
            filepath, write_raw=(data_type != "vis"), load_data=(data_type == "raw")
        )

    # Read in test dataset.
    mir_copy = MirParser(filepath)
    if data_type != "none":
        mir_copy.load_data(load_raw=(data_type == "raw"), apply_tsys=False)

    # The objects won't be equal off the bad - a couple of things to handle first.
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
        if mir_data.sp_data["corrchunk"][mir_data._sphid_dict[key]] != 0:
            assert np.all(entry["vis_flags"][1::2])
            assert not np.any(entry["vis_flags"][::2])

    # Make sure that things work when the flags are all set to True
    for entry in mir_data.vis_data.values():
        entry["vis_flags"][:] = True
    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data["corrchunk"][mir_data._sphid_dict[key]] != 0:
            assert np.all(entry["vis_flags"])


def test_compass_flag_wide_apply(mir_data, compass_soln_file):
    """
    Test COMPASS wide flagging.

    Test that applying COMPASS flags on a per-baseline (all time) basis works correctly.
    """
    for entry in mir_data.vis_data.values():
        entry["vis_flags"][:] = False

    mir_data.in_data["mjd"] += 1
    with uvtest.check_warnings(
        UserWarning, "No metadata from COMPASS matches that in this data set."
    ):
        compass_solns = mir_data._read_compass_solns(compass_soln_file)

    mir_data._apply_compass_solns(compass_solns, apply_bp=False, apply_flags=True)

    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data["corrchunk"][mir_data._sphid_dict[key]] != 0:
            assert np.all(entry["vis_flags"][:8192])
            assert not np.any(entry["vis_flags"][8192:])

    # Make sure that things work when the flags are all set to True
    for entry in mir_data.vis_data.values():
        entry["vis_flags"][:] = True
    for key, entry in mir_data.vis_data.items():
        if mir_data.sp_data["corrchunk"][mir_data._sphid_dict[key]] != 0:
            assert np.all(entry["vis_flags"])

    mir_data._apply_compass_solns(compass_solns, apply_bp=False, apply_flags=True)


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
        if mir_data.sp_data["corrchunk"][mir_data._sphid_dict[key]] != 0:
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


@pytest.mark.parametrize("load_scheme", ["raw", "vis", "both", "drop"])
def test_compare_rechunk(mir_data, load_scheme):
    """Compare rechunking with different options."""
    mir_copy = mir_data.copy()

    # Rechunk by a random factor of two.
    mir_data.rechunk(8)

    # Verify that a chance has actually occured
    assert mir_data != mir_copy

    # Drop whatever data we aren't comparing.
    if load_scheme == "drop":
        # Drop the file dict, which means we can't load the data anymore.
        mir_copy._file_dict = {}
        mir_copy.rechunk(8)

        # Modify these two attributes so that they'll definitely match the above
        # (after the rechunking is done, of course)
        mir_data._file_dict = {}
        for item in ["nch", "fres", "vres"]:
            mir_data._sp_read[item] = mir_data.sp_data[item]
    else:
        mir_data.unload_data(
            unload_vis=(load_scheme == "raw"),
            unload_raw=(load_scheme == "vis"),
        )
        with uvtest.check_warnings(
            UserWarning, "Setting load_data or load_raw to True will unload"
        ):
            mir_copy.rechunk(
                8, load_vis=(load_scheme != "raw"), load_raw=(load_scheme != "vis")
            )

        if load_scheme != "raw":
            mir_copy.apply_tsys()

    assert mir_copy == mir_data


@pytest.mark.parametrize(
    "field,comp,err_msg",
    [
        ["sphid", "abc", "select_comp must be one of"],
        ["abc", "eq", "select_field abc not found in structured array."],
    ],
)
def test_parse_select_compare_errs(mir_data, field, comp, err_msg):
    """Verify that _parse_select_compare throws expected errors."""
    with pytest.raises(ValueError) as err:
        mir_data._parse_select_compare(field, comp, None, mir_data.sp_data)

    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "comp,value,index_arr",
    [
        ["eq", [1, 2, 3, 4], np.arange(1, 5)],
        ["ne", [1, 2, 3, 4], np.arange(5, 21)],
        ["le", 10, np.arange(1, 11)],
        ["gt", 10, np.arange(11, 21)],
        ["lt", 15, np.arange(1, 16)],
        ["ge", 15, np.arange(15, 21)],
        ["btw", [5, 15], np.arange(5, 16)],
        ["out", [5, 15], [1, 2, 3, 4, 16, 17, 18, 19, 20]],
    ],
)
def test_parse_select_compare(mir_data, comp, value, index_arr):
    """Test test_parse_select_compare"""
    mask_arr = mir_data._parse_select_compare("sphid", comp, value, mir_data.sp_data)

    assert np.all(np.isin(mir_data.sp_data["sphid"][mask_arr], index_arr))


@pytest.mark.parametrize(
    "field,comp,err_type,err_msg",
    [
        ["source", "btw", ValueError, 'select_comp must be either "eq" or "ne" '],
        ["source", "eq", ValueError, "If select_field matches a key in codes_dict"],
        ["windSpeed", "eq", NotImplementedError, "Selecting based on we_read"],
        ["awesomeness", "eq", ValueError, "Field name awesomeness not recognized"],
    ],
)
def test_parse_select_errs(mir_data, field, comp, err_type, err_msg):
    """Verify that _parse_select throws expected errors."""
    with pytest.raises(err_type) as err:
        mir_data._parse_select(field, comp, None, None, None, None)

    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "field,comp,value,inhid,blhid,sphid",
    [
        ["source", "eq", "3c84", [1], [1, 2, 3, 4], np.arange(1, 21)],
        ["mjd", "lt", 0.0, [], [1, 2, 3, 4], np.arange(1, 21)],
        ["sb", "ne", "u", [1], [1, 2], np.arange(1, 21)],
        ["fsky", "ge", 230.0, [1], [1, 2, 3, 4], np.arange(11, 21)],
        ["padNumber", "ne", [5, 6, 7, 8], [1], [], np.arange(1, 21)],
    ],
)
def test_parse_select(mir_data, field, comp, value, inhid, blhid, sphid):
    """Verify that _parse_select works as expected."""
    use_in = np.ones_like(mir_data._in_filter)
    use_bl = np.ones_like(mir_data._bl_filter)
    use_sp = np.ones_like(mir_data._sp_filter)
    mir_data._parse_select(field, comp, value, use_in, use_bl, use_sp)

    assert np.all(np.isin(mir_data.in_data["inhid"][use_in], inhid))
    assert np.all(np.isin(mir_data.bl_data["blhid"][use_bl], blhid))
    assert np.all(np.isin(mir_data.sp_data["sphid"][use_sp], sphid))

    # Test that flagged items remain flagged.
    use_in[:] = False
    use_bl[:] = False
    use_sp[:] = False
    mir_data._update_filter(use_in=use_in, use_bl=use_bl, use_sp=use_sp)
    mir_data._parse_select(field, comp, value, use_in, use_bl, use_sp)
    for arr in [use_in, use_bl, use_sp]:
        assert not np.any(arr)


@pytest.mark.parametrize(
    "field,comp,value,use_in,err_type,err_msg",
    [
        [None, 0, None, None, ValueError, "select_field, select_comp, and select_val"],
        [0, 0, "a", None, ValueError, "select_field must be a string."],
        ["mjd", 0, "a", None, ValueError, 'select_comp must be one of "eq", "ne",'],
        ["mjd", "lt", "a", None, ValueError, "select_val must be a single number"],
        ["mjd", "lt", [0, 1], None, ValueError, "select_val must be a single number"],
        ["mjd", "btw", "b", None, ValueError, 'If select_comp is "btw" or "out",'],
        ["mjd", "out", ["a", "b"], None, ValueError, 'If select_comp is "btw" or'],
        ["mjd", "out", 0.0, None, ValueError, 'If select_comp is "btw" or "out",'],
        ["mjd", "btw", [1, 2, 3], None, ValueError, 'If select_comp is "btw" or "out"'],
        [None, None, None, 0.5, IndexError, "use_in, use_bl, and use_sp must be set"],
    ],
)
def test_select_errs(mir_data, field, comp, value, use_in, err_type, err_msg):
    """Verify that select throws errors as expected."""
    with pytest.raises(err_type) as err:
        mir_data.select(
            select_field=field,
            select_comp=comp,
            select_val=value,
            use_in=use_in,
        )

    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "field,comp,value,use_in,reset,warn_msg",
    [
        ["a", "b", "c", None, True, "Resetting data selection, all other arguments"],
        ["a", "b", "c", [0], False, "Selecting data using use_in, use_bl and/or"],
        [None, None, None, None, False, "No arguments supplied to select_field"],
    ],
)
def test_select_warn(mir_data, field, comp, value, use_in, reset, warn_msg):
    """Verify that select throws warnings as expected."""
    with uvtest.check_warnings(UserWarning, warn_msg):
        mir_data.select(
            select_field=field,
            select_comp=comp,
            select_val=value,
            use_in=use_in,
            reset=reset,
        )

    # Verify that the indexing looks okay, even with the warning
    assert mir_data._check_data_index()


@pytest.mark.parametrize(
    "field,comp,value,vis_keys",
    [
        ["mjd", "btw", [60000.0, 50000.0], np.arange(1, 21)],
        ["source", "ne", "nosourcehere", np.arange(1, 21)],
        ["ant", "eq", 4, np.arange(1, 21)],
        ["ant1", "ne", 8, np.arange(1, 21)],
        ["ant1rx", "eq", 0, [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]],
        ["corrchunk", "ne", [1, 2, 3, 4], np.arange(1, 21, 5)],
    ],
)
def test_select(mir_data, field, comp, value, vis_keys):
    """Verify that select throws warnings as expected."""
    mir_data.select(select_field=field, select_comp=comp, select_val=value)

    # Confirm that we have all the indexes we should internally
    assert mir_data._check_data_index()

    # Cross-reference with the list we provide to be sure we have everything.
    assert np.all(np.isin(list(mir_data.vis_data.keys()), vis_keys))


@pytest.mark.parametrize(
    "use_in,use_bl,use_sp, vis_keys",
    [
        [[0], None, None, np.arange(1, 21)],
        [None, [0], None, np.arange(1, 6)],
        [None, None, [0], [1]],
        [None, [0], [1], [2]],
        [[0], [1], [2], []],
        [[0], [0], None, np.arange(1, 6)],
    ],
)
def test_select_use_mask(mir_data, use_in, use_bl, use_sp, vis_keys):
    """Check that use_* parameters for select work as expected"""
    mir_data.select(use_in=use_in, use_bl=use_bl, use_sp=use_sp)

    # Confirm that we have all the indexes we should internally
    assert mir_data._check_data_index()

    # Cross-reference with the list we provide to be sure we have everything.
    assert np.all(np.isin(list(mir_data.vis_data.keys()), vis_keys))


def test_select_reset(mir_data):
    """Verify that running reset with select returns all entries as expected."""
    mir_copy = mir_data.copy()
    # Unload the data on the copy, since that's done on reset
    mir_copy.unload_data()

    # Select based on something that should not exist.
    mir_data.select(select_field="mjd", select_comp="eq", select_val=0.0)
    assert len(mir_data.vis_data.keys()) == 0

    # Now run reset
    mir_data.select(reset=True)
    assert mir_data == mir_copy


def test_eq_errs(mir_data):
    """Verify that the __eq__ method throws appropriate errors."""
    with pytest.raises(ValueError) as err:
        mir_data.__eq__(0)
    assert str(err.value).startswith("Cannot compare MirParser with non-MirParser")


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
        [False, "codes_dict", {"fileer": 3}, False],
        [True, "codes_dict", {"filever": 3}, False],
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
        for attr in ["raw_data", "vis_data", "auto_data"]:
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


@pytest.mark.parametrize(
    "read_arr,index_field,err_type,err_msg",
    [
        [0, 0, TypeError, "read_arr must be of type ndarray."],
        [np.zeros(10), 0, TypeError, "index_field must be of string type."],
        [np.array([], dtype=np.dtype([("", "i8")])), "a", ValueError, "index_field a "],
    ],
)
def test_segment_by_index_errs(read_arr, index_field, err_type, err_msg):
    """Verify that segment_by_index produces errors as expected."""
    with pytest.raises(err_type) as err:
        MirParser.segment_by_index(read_arr, index_field)
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "attr,index_field,index_arr",
    [
        ["in_data", "inhid", np.arange(1).reshape(-1, 1)],
        ["bl_data", "inhid", np.arange(4).reshape(1, -1)],
        ["bl_data", "blhid", np.arange(4).reshape(-1, 1)],
        ["sp_data", "inhid", np.arange(20).reshape(1, -1)],
        ["sp_data", "blhid", np.arange(20).reshape(4, 5)],
        ["sp_data", "sphid", np.arange(20).reshape(-1, 1)],
    ],
)
def test_segment_by_index(mir_data, attr, index_field, index_arr):
    """
    Verify that segment_by_index breaks apart a given array correctly based
    on the values in the indexing field.
    """
    read_arr = getattr(mir_data, attr)
    subarr_dict, posarr_dict = MirParser.segment_by_index(read_arr, index_field)
    for key, index in zip(subarr_dict.keys(), index_arr):
        assert np.array_equal(read_arr[index], subarr_dict[key])
        assert np.array_equal(index, posarr_dict[key])
        assert np.all(read_arr[index][index_field] == key)


def test_scan_int_start_errs(mir_data):
    """Verify scan_int_start throws errors when expected."""
    with pytest.raises(ValueError) as err:
        mir_data.scan_int_start(mir_data.filepath, allowed_inhid=[-1])
    assert str(err.value).startswith("Index value inhid in sch_read does not match")


def test_calc_int_start(mir_data):
    """Verify that we can correctly calculate integration starting periods."""
    true_dict = {1: (1, 1048680, 0)}
    assert true_dict == mir_data.calc_int_start(mir_data._sp_read)

    # Now see what happens if we break the ordering
    mod_dict = {0: (0, 1048680 // 2, 0), 1: (1, 1048680 // 2, 8 + 1048680 // 2)}
    mir_data._sp_read["inhid"] = np.mod(np.arange(20), 2)
    assert mod_dict == mir_data.calc_int_start(mir_data._sp_read)


def test_scan_int_start(mir_data):
    """Verify that we can correctly scan integration starting periods."""
    true_dict = {1: (1, 1048680, 0)}
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
    bad_dict = {mir_data.filepath: {2: (1, 120, 120)}}
    good_dict = {mir_data.filepath: {2: (1, 1048680, 0)}}
    mir_data.sp_data["inhid"][:] = 2
    mir_data.sp_data["nch"][:] = 1

    mir_data._file_dict = bad_dict
    with pytest.raises(ValueError) as err:
        mir_data.read_vis_data(
            [mir_data.filepath],
            [mir_data._file_dict[mir_data.filepath]],
            mir_data.sp_data,
            return_raw=True,
            return_vis=False,
        )
    assert str(err.value).startswith("Values in int_start_dict do not match")
    mir_data.fix_int_start()

    assert good_dict == mir_data._file_dict

    _ = mir_data.read_vis_data(
        [mir_data.filepath],
        [mir_data._file_dict[mir_data.filepath]],
        mir_data.sp_data,
        return_raw=True,
        return_vis=False,
    )

    # Make sure that things work if we don't inherit stuff from object
    check_dict = mir_data.fix_int_start([mir_data.filepath], list(bad_dict.values()))

    assert good_dict == check_dict


def test_scan_auto_data_err(tmp_path):
    """Verify that scan_auto_data throws appropriate errors."""
    with pytest.raises(FileNotFoundError) as err:
        MirParser.scan_auto_data(tmp_path)
    assert str(err.value).startswith("Cannot find file")


def test_read_packdata_mmap(mir_data):
    """Test that reading in vis data with mmap works just as well as np.fromfile"""
    mmap_data = mir_data.read_packdata(
        mir_data.filepath, mir_data._file_dict[mir_data.filepath], use_mmap=True
    )

    reg_data = mir_data.read_packdata(
        mir_data.filepath, mir_data._file_dict[mir_data.filepath], use_mmap=False
    )

    assert mmap_data.keys() == reg_data.keys()
    for key in mmap_data.keys():
        assert np.array_equal(mmap_data[key], reg_data[key])


def test_read_packdata_make_packdata(mir_data):
    """Verify that making packdata produces the same result as reading packdata"""
    read_data = mir_data.read_packdata(
        mir_data.filepath,
        mir_data._file_dict[mir_data.filepath],
    )

    make_data = mir_data.make_packdata(mir_data.sp_data, mir_data.raw_data)

    assert read_data.keys() == make_data.keys()
    for key in read_data.keys():
        assert np.array_equal(read_data[key], make_data[key])


def test_read_vis_data_errs(mir_data):
    """Conirm that read_vis_data throws errors as expected."""
    with pytest.raises(ValueError) as err:
        mir_data.read_vis_data(["a"], ["b", "c"], mir_data.sp_data)
    assert str(err.value).startswith(
        "Must provide a sequence of the same length for filepath and int_start_dict."
    )


def test_read_auto_data_errs(mir_data):
    """Conirm that read_auto_data throws errors as expected."""
    with pytest.raises(ValueError) as err:
        mir_data.read_auto_data(["a"], ["b", "c"], mir_data.ac_data)
    assert str(err.value).startswith(
        "Must provide a sequence of the same length for filepath and int_start_dict."
    )


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
    mir_data.eng_data["antennaNumber"][:] = -1
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


def test_check_data_index(mir_data):
    """Verify that check_data_index returns True/False as expected."""
    assert mir_data._check_data_index()

    # Now muck with the records so that this becomes False
    for item in ["sp_data", "ac_data"]:
        getattr(mir_data, item)[0] = -1
        assert not mir_data._check_data_index()
        getattr(mir_data, item)[0] = 1
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
    mir_copy = mir_data.copy()

    # Manually downselect the data that we need.
    if select_vis or select_raw:
        mir_data.sp_data = mir_data.sp_data[::2]
    if select_auto:
        mir_data.ac_data = mir_data.ac_data[::2]

    mir_data._downselect_data(
        select_vis=select_vis, select_raw=select_raw, select_auto=select_auto
    )

    if select_vis or select_auto or select_raw:
        assert mir_data != mir_copy
    else:
        assert mir_data == mir_copy

    # Verify that data indexes match metadata ones.
    if select_vis != select_raw:
        # Temporarily mark these as unloaded so that the changes in sp_data don't
        # impact the unchanged data attribute.
        mir_data._vis_data_loaded = select_vis
        mir_data._raw_data_loaded = select_raw

    assert mir_data._check_data_index()

    # Put these back to True, since all data should be loaded.
    mir_data._vis_data_loaded = mir_data._raw_data_loaded = True

    # If we downselected, make sure we plug back in the original data.
    if select_vis or select_raw:
        mir_data.sp_data = mir_data._sp_read.copy()
    if select_auto:
        mir_data.ac_data = mir_data._ac_read.copy()

    # Make sure that the metadata all look good.
    assert mir_data.__eq__(mir_copy, metadata_only=True)

    if select_vis or select_auto or select_raw:
        with pytest.raises(KeyError) as err:
            mir_data._downselect_data(
                select_vis=select_vis, select_raw=select_raw, select_auto=select_auto
            )
        # No idea why the single quotes are required here. I'm just gonna go with the
        # flow, althougn maybe this will need to get fixed later.
        assert str(err.value).startswith(
            "'Missing spectral records in data attributes. Run load_data instead.'"
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


def test_downselect_data_no_op(mir_data):
    """
    Verify that running downselect with an object without a file dict does not
    change any of the attributes of the object.
    """
    # Drop the file dict
    mir_data._file_dict = {}

    # Change the atributes so that normally, downselect would drop records
    mir_data.sp_data = mir_data.sp_data[::2]
    mir_data.ac_data = mir_data.ac_data[::2]

    # Make a copy of the data
    mir_copy = mir_data.copy()

    mir_data._downselect_data(select_vis=True, select_raw=True, select_auto=True)

    # Finally, check that nothing has changed.
    assert mir_data == mir_copy


def test_unload_data_err(mir_data):
    """Verify that unload_data throws an error when no file_dict is found"""
    mir_data._file_dict = {}

    with pytest.raises(ValueError) as err:
        mir_data.unload_data()
    assert str(err.value).startswith(
        "Cannot unload data as there is no file to load data from."
    )


@pytest.mark.parametrize("unload_auto", [True, False])
@pytest.mark.parametrize("unload_vis", [True, False])
@pytest.mark.parametrize("unload_raw", [True, False])
def test_unload_data(mir_data, unload_vis, unload_raw, unload_auto):
    """Verify that unload_data unloads data as expected."""
    mir_data.unload_data(
        unload_vis=unload_vis, unload_raw=unload_raw, unload_auto=unload_auto
    )

    assert mir_data.vis_data is None if unload_vis else mir_data.vis_data is not None
    assert mir_data._vis_data_loaded != unload_vis
    assert mir_data._tsys_applied != unload_vis

    assert mir_data.raw_data is None if unload_raw else mir_data.raw_data is not None
    assert mir_data._raw_data_loaded != unload_raw

    assert mir_data.auto_data is None if unload_auto else mir_data.auto_data is not None
    assert mir_data._auto_data_loaded != unload_auto


@pytest.mark.parametrize(
    "load_raw,load_vis,unload_raw,err_type,err_msg",
    [
        [True, False, True, ValueError, "Cannot load raw data from disk "],
        [False, True, True, ValueError, "No file to load vis_data from"],
        [False, True, False, ValueError, "No file to load vis_data from, but raw_data"],
    ],
)
def test_load_data_errs(mir_data, load_raw, load_vis, unload_raw, err_type, err_msg):
    """
    Check that load_data throws errors as expected, specfically in the case where the
    file_dict has been removed from the object.
    """
    mir_data.unload_data(unload_raw=unload_raw)
    mir_data._file_dict = {}

    with pytest.raises(err_type) as err:
        mir_data.load_data(load_raw=load_raw, load_vis=load_vis)
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "load_vis,load_raw,downsel,conv,dropfl,warn_msg",
    [
        [True, False, False, False, True, "No file to load from, and vis data"],
        [False, True, False, False, True, "No file to load from, and raw data"],
        [False, False, True, False, True, "allow_downselect argument ignored because"],
        [True, True, False, True, False, "Cannot load raw data AND convert"],
        [True, False, False, True, False, "Raw data not loaded, cannot convert"],
        [True, False, True, True, False, "Loaded raw data does not contain all "],
        [True, False, True, False, False, "Cannot downselect cross data"],
        [False, False, True, False, False, "Cannot downselect auto data"],
    ],
)
def test_load_data_warn(mir_data, load_vis, load_raw, downsel, conv, dropfl, warn_msg):
    """Check that load_data throws appropriate warnings."""
    if dropfl:
        mir_data._file_dict = {}

    if conv:
        if downsel:
            mir_data.raw_data = {}
        else:
            mir_data.unload_data()
    else:
        if downsel:
            if load_vis or load_raw:
                mir_data.vis_data = mir_data.raw_data = {}
            else:
                mir_data.auto_data = {}

    with uvtest.check_warnings(UserWarning, warn_msg):
        mir_data.load_data(
            load_vis=load_vis,
            load_raw=load_raw,
            allow_downselect=downsel,
            allow_conversion=conv,
        )


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

    assert mir_data._vis_data_loaded == (load_vis is None)
    assert mir_data._tsys_applied == (load_vis is None)
    assert mir_data._raw_data_loaded != (load_vis is None)

    assert mir_data.auto_data is None
    assert not mir_data._auto_data_loaded


@pytest.mark.parametrize("drop_file_dict", [False, True])
def test_load_data_conv(mir_data, drop_file_dict):
    """Test that the conversion operation of load_data operates as expected."""
    mir_copy = mir_data.copy()
    mir_data.unload_data(unload_raw=False, unload_vis=True, unload_auto=False)

    if drop_file_dict:
        mir_data._file_dict = {}
        mir_copy._file_dict = {}

    assert mir_data.vis_data is None

    mir_data.load_data(load_vis=True, allow_conversion=True)

    assert mir_data.vis_data is not None
    assert mir_copy == mir_data


def test_update_filter_update_data(mir_data):
    """
    Test that _update_filter behaves as expected with update_data.
    """
    mir_copy = mir_data.copy()
    # Corrupt the data, see if update_data will fix it.
    mir_data.vis_data[1]["vis_data"][:] = 0.0
    mir_data.raw_data[1]["raw_data"][:] = 0.0
    mir_data.auto_data[1][:] = 0.0
    mir_data._update_filter(update_data=True)
    assert mir_data == mir_copy

    mir_copy.unload_data()
    mir_data._data_mucked = True
    with uvtest.check_warnings(
        UserWarning, "Unable to update data attributes, unloading them now."
    ):
        mir_data._update_filter(update_data=True)

    mir_data._data_mucked = False
    assert mir_data == mir_copy


def test_update_filter_allow_downsel(mir_data):
    """
    Test that _update_filter behaves as expected wwhen allowing downselections.
    """
    # Mark the data so that *_data and _*_read attributes are different
    mir_data.in_data["mjd"][:] = -1
    mir_data.eng_data["padNumber"][:] = -1
    mir_data.bl_data["u"][:] = -1
    mir_data.sp_data["fsky"][:] = -1
    mir_data.we_data["N"][:] = -1

    # Drop the autos
    mir_data._has_auto = False
    mir_data.unload_data(unload_vis=False, unload_raw=False)
    mir_data.ac_data = mir_data._ac_read = mir_data._ac_filter = None

    mir_copy = mir_data.copy()
    # If no filters applied, all data should be used, and allow_downselect should
    # produce the same thing.
    mir_data._update_filter(allow_downselect=True)

    assert mir_data == mir_copy

    mir_data._update_filter(
        use_bl=np.array([True, True, False, False]), allow_downselect=True
    )
    mir_copy._update_filter(use_bl=np.array([True, True, False, False]))

    assert mir_data != mir_copy
    mir_data._update_filter(allow_downselect=True)
    mir_copy._update_filter(update_data=False)
    # If can't downselect because you are making the filter bigger, then the old
    # metadata should get loaded, and now the two copies should be the same.
    assert mir_data == mir_copy


def test_tofile_append_errs(mir_data, tmp_path):
    """
    Test that tofile throws errors as expected. Note that we kind of have to test these
    in sequence since they require subsequent modifications to the MirParser object.
    """
    testfile = os.path.join(tmp_path, "test_tofile_errs.mir")

    # Write the mir dataset first to file to check for errors. This first call to
    # append should be okay because there's nothing yet to append to.
    mir_data.tofile(testfile, append_data=True)

    with pytest.raises(ValueError) as err:
        mir_data.tofile(testfile, append_data=True)
    assert str(err.value).startswith(
        "Cannot append data when integration header IDs overlap."
    )

    mir_data.in_data["inhid"] = 100

    with pytest.raises(ValueError) as err:
        mir_data.tofile(testfile, append_data=True)
    assert str(err.value).startswith(
        "Cannot append data when baseline header IDs overlap."
    )

    mir_data.bl_data["blhid"] = 100

    with pytest.raises(ValueError) as err:
        mir_data.tofile(testfile, append_data=True)
    assert str(err.value).startswith(
        "Cannot append data when spectral record header IDs overlap."
    )


@pytest.mark.parametrize(
    "write_raw,unload_data,warn_msg",
    [
        [False, False, "Writing out raw data with tsys applied."],
        [True, True, "No data loaded, writing metadata only"],
    ],
)
def test_tofile_warn(mir_data, tmp_path, write_raw, unload_data, warn_msg):
    """Test that tofile throws errors as expected."""
    testfile = os.path.join(
        tmp_path, "test_tofile_warn_%s.mir" % ("meta" if unload_data else "tsysapp")
    )
    if unload_data:
        mir_data.unload_data()

    with uvtest.check_warnings(UserWarning, warn_msg):
        mir_data.tofile(testfile, write_raw=write_raw)

    # Drop the data and autos here to make the comparison a bit easier.
    mir_data.unload_data()
    mir_data._has_auto = False
    mir_data.ac_data = mir_data._ac_read = mir_data._ac_filter = None

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
    "chan_avg,muck_data,drop_file,drop_data,err_type,err_msg",
    [
        [0.5, False, False, False, ValueError, "chan_avg must be of type int."],
        [-1, False, False, False, ValueError, "chan_avg cannot be a number less than"],
        [3, False, False, False, ValueError, "chan_avg does not go evenly into "],
        [2, True, False, False, ValueError, "Cannot load data due to modifications"],
        [2, False, True, False, ValueError, "Cannot unload data as there is no file"],
        [2, False, False, True, ValueError, "Index values do not match data keys."],
    ],
)
def test_rechunk_errs(
    mir_data, chan_avg, muck_data, drop_file, drop_data, err_type, err_msg
):
    """Verify that rechunk throws errors as expected."""
    mir_data._data_mucked = muck_data
    if drop_file:
        mir_data._file_dict = {}

    if drop_data:
        mir_data.vis_data = {}

    # Rather than parameterizing this, because the underlying object isn't changed,
    # check for the different load states here, since the error should get thrown
    # no matter which thing you are loading.
    check_list = []
    if not drop_data:
        check_list.extend([(True, True), (True, False), (False, True)])
    if not (drop_file or muck_data):
        # Some errors should report even if not loading data.
        check_list.append((False, False))
    for (load_vis, load_raw) in check_list:
        with pytest.raises(err_type) as err:
            mir_data.rechunk(chan_avg, load_vis=load_vis, load_raw=load_raw)
        assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize(
    "unload_data,load_raw,load_vis,warn_msg",
    [
        [False, True, False, "Setting load_data or load_raw to True will unload "],
        [False, False, True, "Setting load_data or load_raw to True will unload "],
        [False, True, True, "Setting load_data or load_raw to True will unload "],
        [True, False, False, "No data loaded to average, returning."],
    ],
)
def test_rechunk_warn(mir_data, unload_data, load_raw, load_vis, warn_msg):
    """Verify that rechunk throws warnings as expected."""
    if unload_data:
        mir_data.unload_data()

    with uvtest.check_warnings(UserWarning, warn_msg):
        mir_data.rechunk(2, load_vis=load_vis, load_raw=load_raw)


def test_rechunk_nop(mir_data):
    """Test that setting chan_avg to 1 doesn't change the object."""
    mir_copy = mir_data.copy()

    mir_data.rechunk(1)
    assert mir_data == mir_copy


@pytest.mark.parametrize(
    "arr1,arr2,index_field,err_type,err_msg",
    [
        [np.array(0), np.array(0.0), (None,), ValueError, "Both arrays must be of the"],
        [np.array(0), np.array(0), None, ValueError, "index_name must be a string or"],
        [np.array(0), np.array(0), (None,), ValueError, "index_name must be a string"],
        [
            np.array([], dtype=np.dtype([("", "i8")])),
            np.array([], dtype=np.dtype([("", "i8")])),
            "test",
            ValueError,
            "index_name test not a recognized field in either array.",
        ],
    ],
)
def test_arr_index_overlap_errs(mir_data, arr1, arr2, index_field, err_type, err_msg):
    """Verify that _arr_index_overlap throws errors as expected."""
    with pytest.raises(err_type) as err:
        mir_data._combine_read_arr_check(arr1, arr2, index_field)
    assert str(err.value).startswith(err_msg)


def test_arr_index_overlap(mir_data):
    """Test that test_arr_index gives results as expected"""
    data_arr = mir_data.sp_data.copy()
    copy_arr = mir_data.sp_data.copy()

    idx1, idx2 = mir_data._arr_index_overlap(data_arr, copy_arr, "sphid")
    assert np.all(idx1 == np.arange(20))
    assert np.all(idx2 == np.arange(20))

    # Check that adding extra keys doesn't change the full selction here
    idx1, idx2 = mir_data._arr_index_overlap(
        data_arr, copy_arr, ("sphid", "blhid", "corrchunk")
    )
    assert np.all(idx1 == np.arange(20))
    assert np.all(idx2 == np.arange(20))

    copy_arr = copy_arr[::2]
    idx1, idx2 = mir_data._arr_index_overlap(data_arr, copy_arr, "sphid")
    assert np.all(idx1 == np.arange(0, 20, 2))
    assert np.all(idx2 == np.arange(10))

    data_arr = data_arr[::4]
    idx1, idx2 = mir_data._arr_index_overlap(data_arr, copy_arr, "sphid")
    assert np.all(idx1 == np.arange(5))
    assert np.all(idx2 == np.arange(0, 10, 2))

    # Verify that if we muck all the other fields, we still get matches.
    for field in data_arr.dtype.names:
        if field != "sphid":
            # Plug in truly random data -- this should never, _ever_ match
            data_arr[field] = np.random.rand(5) + 1.0
            copy_arr[field] = np.random.rand(10)

    idx1, idx2 = mir_data._arr_index_overlap(data_arr, copy_arr, "sphid")
    assert np.all(idx1 == np.arange(5))
    assert np.all(idx2 == np.arange(0, 10, 2))


@pytest.mark.parametrize("any_match", [False, True])
@pytest.mark.parametrize(
    "attr,index_name",
    [
        ["_ac_read", "achid"],
        ["_antpos_read", "antenna"],
        ["_bl_read", "blhid"],
        ["_codes_read", ("v_name", "icode", "ncode")],
        ["_eng_read", ("inhid", "antennaNumber")],
        ["_in_read", "inhid"],
        ["_sp_read", "sphid"],
        ["_we_read", "ints"],
        ["ac_data", "achid"],
        ["antpos_data", "antenna"],
        ["bl_data", "blhid"],
        ["eng_data", ("inhid", "antennaNumber")],
        ["in_data", "inhid"],
        ["sp_data", "sphid"],
        ["we_data", "ints"],
    ],
)
def test_combine_read_arr_check(attr, any_match, index_name, mir_data):
    read_arr = getattr(mir_data, attr)

    if attr in ["in_data", "_in_read", "we_data", "_we_read"]:
        # These are length 1 arrays, which we want to tweak so that we can do the
        # any_match comparison below.
        read_arr = np.tile(read_arr, 2)
        read_arr[index_name][1] = 2

    copy_arr = read_arr.copy()

    # These are indexing fields that we should not change
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
    ]

    if any_match:
        for field in copy_arr.dtype.names:
            if field not in prot_fields:
                copy_arr[field][1:] = -1

        assert not MirParser._combine_read_arr_check(read_arr, copy_arr, index_name)

    # Make sure the arrays are compatible
    assert MirParser._combine_read_arr_check(
        read_arr, copy_arr, index_name=index_name, any_match=any_match
    )

    # If we cut down the array by two, we should still get a positive result.
    copy_arr = copy_arr[::2]
    assert MirParser._combine_read_arr_check(
        read_arr, copy_arr, index_name=index_name, any_match=any_match
    )

    # Now nuke the non-indexing fields any verify that the check returns False
    for field in copy_arr.dtype.names:
        if field not in prot_fields:
            copy_arr[field] = -1

    assert not MirParser._combine_read_arr_check(
        read_arr, copy_arr, index_name=index_name, any_match=any_match
    )


def test_combine_read_arr_errs(mir_data):
    """Verify that _combine_read_arr throws errors as expected."""
    with pytest.raises(ValueError) as err:
        mir_data._combine_read_arr(
            mir_data.sp_data, mir_data.bl_data, "sphid", overwrite=True
        )
    assert str(err.value).startswith("Both arrays must be of the same dtype.")

    mir_data.sp_data["fsky"] = 0.0
    with pytest.raises(ValueError) as err:
        mir_data._combine_read_arr(mir_data.sp_data, mir_data._sp_read, "sphid")
    assert str(err.value).startswith("Arrays have overlapping indicies with different")


@pytest.mark.parametrize("overwrite", [True, False])
def test_combine_read_arr(mir_data, overwrite):
    """Verify that _combine_read_arr combines arrays as expected."""
    data_arr = mir_data.bl_data.copy()
    copy_arr = mir_data.bl_data.copy()
    if overwrite:
        # Corrupt the metadata to test overwrite=True
        for field in data_arr.dtype.names:
            if field != "blhid":
                data_arr[field] = np.random.rand(4)

    # Verify that combining identical arrays pops out the same array.
    assert np.all(
        mir_data.bl_data
        == MirParser._combine_read_arr(data_arr, copy_arr, "blhid", overwrite=overwrite)
    )

    # Now drop elements from the first array, and verify that we still get
    # the full array back (because copy_arr is still there).
    assert np.all(
        mir_data.bl_data
        == MirParser._combine_read_arr(
            data_arr[:1], copy_arr, "blhid", overwrite=overwrite
        )
    )

    # Okay, last test -- what happens if we _don't_ have overlap
    new_arr, idx_arr = MirParser._combine_read_arr(
        data_arr[::2], copy_arr[1::2], "blhid", return_indices=True
    )

    # Check that all the index values that we expect are there
    assert np.all(idx_arr == [True, True])
    # If overwrite, the two arrays will be different, otherwise they'll be the same
    assert np.all(new_arr == copy_arr) != overwrite
    # Finally, check that the order of the new array is what we expect.
    assert np.all(new_arr["blhid"] == [1, 2, 3, 4])


@pytest.mark.parametrize(
    "unload_data,muck_data,force,err_type,err_msg",
    [
        ["auto", None, False, ValueError, "Cannot combine objects where one has auto"],
        ["vis", None, False, ValueError, "Cannot combine objects where one has vis"],
        ["raw", None, False, ValueError, "Cannot combine objects where one has raw"],
        [None, "tsys", False, ValueError, "Cannot combine objects where one has tsys"],
        [None, "in_data", False, ValueError, "Objects appear to contain overlapping"],
        [None, "_in_read", False, ValueError, "Objects appear to come from different"],
        [None, "all", False, TypeError, "Cannot add a MirParser object an object of "],
        [
            "all",
            None,
            True,
            ValueError,
            "Cannot combine objects with force=True when no vis or raw ",
        ],
        [
            None,
            "tsys",
            True,
            ValueError,
            "Cannot combine objects with force=True where one object has tsys",
        ],
    ],
)
def test_add_errs(mir_data, unload_data, muck_data, force, err_type, err_msg):
    """Verify that __add__ throws errors as expected"""
    mir_copy = mir_data.copy()

    mir_data.unload_data(
        unload_vis=(unload_data in ["vis", "all"]),
        unload_raw=(unload_data in ["raw", "all"]),
        unload_auto=(unload_data in ["auto", "all"]),
    )

    if muck_data is not None:
        if muck_data == "tsys":
            mir_data.apply_tsys(invert=True)
        elif muck_data == "in_data":
            mir_data.in_data["mjd"] = 0.0
        elif muck_data == "_in_read":
            mir_data._in_read["mjd"] = 0.0
        elif muck_data == "all":
            mir_data = np.arange(100)

    if force:
        mir_data._in_read["mjd"] = 0.0

    with pytest.raises(err_type) as err:
        mir_data.__add__(mir_copy, force=force)
    assert str(err.value).startswith(err_msg) or (muck_data == "all")

    with pytest.raises(err_type) as err:
        mir_copy.__add__(mir_data, force=force)
    assert str(err.value).startswith(err_msg)


def test_add_simple(mir_data):
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
    mir_data.select("corrchunk", "eq", [0, 1, 2])
    mir_copy.select("corrchunk", "ne", [0, 1, 2])

    # Verify that we have changed some things
    assert mir_data != mir_orig
    assert mir_data != mir_copy
    assert mir_orig != mir_copy

    # Now combine the two, and see what comes out.
    mir_data += mir_copy
    assert mir_data == mir_orig

    # Hey, that was fun, let's try selecting on bl next!
    mir_data.select(reset=True)
    mir_copy.select(reset=True)

    mir_data.select("sb", "eq", "l")
    mir_copy.select("sb", "eq", "u")

    # The reset unloads the data, so fix that now
    mir_data.load_data(load_vis=True, load_raw=True, load_auto=True, apply_tsys=True)
    mir_copy.load_data(load_vis=True, load_raw=True, load_auto=True, apply_tsys=True)

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
    mir_data.select(reset=True)
    mir_copy.select(reset=True)

    mir_copy._has_auto = False
    mir_copy._ac_filter = mir_copy.ac_data = mir_copy._ac_read = None
    mir_copy.auto_data = None

    mir_data.select("corrchunk", "eq", [1, 2])
    mir_copy.select("corrchunk", "eq", [3, 4])
    mir_data.load_data(load_vis=True, load_raw=True, load_auto=True, apply_tsys=True)
    mir_copy.load_data(load_vis=True, load_raw=True, load_auto=True, apply_tsys=True)

    with uvtest.check_warnings(
        UserWarning, "Both objects do not have auto-correlation data."
    ):
        mir_data += mir_copy

    # Make sure we got all the data entries
    assert mir_data._check_data_index()

    # Make sure auto properties propagated correctly.
    assert not (mir_data._has_auto or mir_data._auto_data_loaded)
    mir_orig._has_auto = mir_orig._auto_data_loaded = False
    for item in ["_ac_filter", "ac_data", "_ac_read", "auto_data"]:
        assert getattr(mir_data, item) is None
        setattr(mir_orig, item, None)

    # Finally, make sure the object isn't the same, but after a reset and reload,
    # we get the same object back (modulo the auto-correlation data).
    assert mir_data != mir_orig
    mir_data.select(reset=True)
    mir_data.load_data(load_vis=True, load_raw=True, load_auto=True, apply_tsys=True)
    assert mir_data == mir_orig


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
        mir_data.codes_dict["filever"] = -1
    else:
        for field in getattr(mir_data, muck_attr).dtype.names:
            if field not in prot_fields:
                getattr(mir_data, muck_attr)[field] = -1

    # After mucking, verfiy that at least something looks different
    assert mir_data != mir_copy

    # mir_copy contains the good data, so adding it second will overwrite the bad data.
    with uvtest.check_warnings(
        UserWarning, "Data in objects appears to overlap, but with differing metadata."
    ):
        assert mir_data.__add__(mir_copy, overwrite=True) == mir_copy

    # On the other hand, if we add mir_data second, the bad values should get propagated
    with uvtest.check_warnings(
        UserWarning, "Data in objects appears to overlap, but with differing metadata."
    ):
        assert mir_copy.__add__(mir_data, overwrite=True) == mir_data


@pytest.mark.parametrize(
    "muck_list",
    [
        ["_ac_read"],
        ["_antpos_read"],
        ["_bl_read"],
        ["_codes_read"],
        ["_eng_read"],
        ["_in_read"],
        ["_sp_read"],
        ["_we_read"],
        [
            "_ac_read",
            "_antpos_read",
            "_bl_read",
            "_codes_read",
            "_eng_read",
            "_in_read",
            "_sp_read",
            "_we_read",
        ],
    ],
)
def test_add_force(mir_data, muck_list):
    """Verify that the 'force' option on __add__ works as expected."""
    mir_copy = mir_data.copy()
    mir_orig = mir_data.copy()

    # Confirm that the below is basically a no-op
    with uvtest.check_warnings(None):
        mir_data.__iadd__(mir_copy, force=True)
    assert mir_data == mir_orig

    # Now mess w/ the metadata, that will flag the data as being from a different file
    for item in muck_list:
        for field in getattr(mir_data, item).dtype.names:
            getattr(mir_data, item)[field] = -1

    with uvtest.check_warnings(
        UserWarning, "Objects here do not appear to be from the same file,"
    ):
        mir_data = mir_data.__add__(mir_copy, force=True)

    # Since we have all records in hand, what we end up with should be the same file.
    assert mir_data._file_dict == {}
    mir_data._file_dict = copy.deepcopy(mir_orig._file_dict)
    assert mir_data == mir_orig

    # Now actually execute an select, muck the data, and then add.
    mir_data.select("sb", "eq", "l")
    mir_copy.select("sb", "eq", "u")

    for item in muck_list:
        for field in getattr(mir_data, item).dtype.names:
            getattr(mir_data, item)[field] = -1

    with uvtest.check_warnings(
        UserWarning, "Objects here do not appear to be from the same file,"
    ):
        mir_data = mir_data.__add__(mir_copy, force=True)

    # Again, since we all records together, we should have an equivalent object
    assert mir_data._file_dict == {}
    mir_data._file_dict = copy.deepcopy(mir_orig._file_dict)
    assert mir_data == mir_orig

    # Finally, grab something that doesn't combine to be the same object
    mir_data.select("sb", "eq", "u")
    for item in muck_list:
        for field in getattr(mir_data, item).dtype.names:
            getattr(mir_data, item)[field] = -1

    with uvtest.check_warnings(
        UserWarning, "Objects here do not appear to be from the same file,"
    ):
        mir_data = mir_data.__add__(mir_copy, force=True)
    # Again, since we all records together, we should have an equivalent object
    assert mir_data._file_dict == {}
    mir_data._file_dict = copy.deepcopy(mir_orig._file_dict)
    assert mir_data != mir_orig


@pytest.mark.parametrize(
    "skip_muck,err_type,err_msg",
    [
        ["no_obj", TypeError, "Can only concat MirParser objects."],
        ["auto", ValueError, "Cannot combine objects both with and without auto"],
        ["nofile", ValueError, "Cannot concat objects without an associated file"],
        ["file", ValueError, "At least one object to be concatenated has been loaded"],
        ["ants", ValueError, "Two of the objects provided do not have the same ant"],
        ["_in_read", ValueError, "Two of the objects provided appear to hold ident"],
        ["_eng_read", ValueError, "Two of the objects provided appear to hold ident"],
        ["_bl_read", ValueError, "Two of the objects provided appear to hold ident"],
        ["_sp_read", ValueError, "Two of the objects provided appear to hold ident"],
        ["_we_read", ValueError, "Two of the objects provided appear to hold ident"],
        ["codes", ValueError, "codes_dict contains different keys between objects,"],
        ["dup_in_read", ValueError, "Two of the objects appear to have overlapping"],
        ["muck_pol", ValueError, "Cannot concat objects, differing polarization "],
        ["muck_band", ValueError, "Cannot concat objects, differing correlator"],
        ["muck_filever", ValueError, "Cannot concat objects, differing file"],
        ["muck_sb", ValueError, "Cannot concat objects, sb key"],
    ],
)
def test_concat_err(mir_data, skip_muck, err_type, err_msg):
    """
    Verify that concat throws errors as expected. So this is admittedly a long test,
    because the error checking is sequential, and without two actual files to concat,
    we have to make several modifications to the object arising from one.
    """
    mir_copy = mir_data.copy()

    diff_list = {
        "_in_read": ["inhid"],
        "_eng_read": ["inhid"],
        "_bl_read": ["inhid", "blhid"],
        "_sp_read": ["inhid", "blhid", "sphid"],
        "_we_read": ["ints"],
    }

    # We want to go through the above arrays and modify them so that concat
    # doesn't error out (unless we want it to).
    for key, muck_list in diff_list.items():
        if skip_muck != key:
            for field in muck_list:
                getattr(mir_copy, key)[field] = getattr(mir_copy, key)[field] + (
                    np.max(getattr(mir_copy, key)[field])
                )

    if skip_muck != "file":
        # If we don't want to have the filenames agree, change that now.
        for idx, key in enumerate(list(mir_copy._file_dict)):
            mir_copy._file_dict[str(idx)] = mir_copy._file_dict.pop(key)
    if skip_muck == "no_obj":
        # What if you try to concat w/ a non-MirParser object?
        mir_copy = None
    elif skip_muck == "codes":
        # What if you make it so that the keys in codes_dict don't agree?
        mir_copy._codes_read["v_name"][0] = b"abcdefg"
    elif skip_muck == "ants":
        # What if you muck the antenna positions?
        mir_copy._antpos_read[:] = 0.0
    elif skip_muck == "nofile":
        # What if one object doesn't have a file_dict?
        mir_copy._file_dict = {}
    elif skip_muck == "auto":
        # What if one file doesn't have autos?
        mir_copy._has_auto = False
    elif skip_muck == "dup_in_read":
        # What if one object has partially overlapping data?
        mir_copy._in_read = np.tile(mir_copy._in_read, 2)
        mir_copy._in_read["inhid"][0] = 1
        # Note we need this to prevent an earlier error on codes_read being identical
        mir_copy._codes_read["code"][-1] = b"-1"
    elif skip_muck == "muck_filever":
        mir_copy._codes_read["code"][0] = b"-1"
    elif skip_muck == "muck_pol":
        # What if codes_dict has different polarization states?
        mir_copy._codes_read["code"][mir_copy._codes_read["v_name"] == b"pol"] = [
            b"a",
            b"b",
            b"c",
            b"d",
        ]
    elif skip_muck == "muck_band":
        # What if codes_dict indicates a different correlator config?
        mir_copy._codes_read["code"][mir_copy._codes_read["v_name"] == b"band"] = [
            b"a",
            b"b",
            b"c",
            b"d",
            b"e",
        ]
    elif skip_muck == "muck_sb":
        # What if codes_dict indicates a different correlator config?
        mir_copy._codes_read["code"][mir_copy._codes_read["v_name"] == b"sb"] = [
            b"lsb",
            b"usb",
        ]

    with pytest.raises(err_type) as err:
        MirParser.concat((mir_data, mir_copy), force=False)
    assert str(err.value).startswith(err_msg)


def test_concat_warn(mir_data):
    """Verify that concat throws warnings as expected."""
    mir_copy = mir_data.copy()
    mir_copy.unload_data()

    # Providing a single object to concat should basically spit out an equivlaent
    # object, with data unloaded.
    with uvtest.check_warnings(None):
        assert mir_copy == MirParser.concat((mir_data,))

    # Modify _file_dict so that concat doesn't immediately balk
    for idx, key in enumerate(list(mir_copy._file_dict)):
        mir_copy._file_dict[str(idx)] = mir_copy._file_dict.pop(key)

    with uvtest.check_warnings(
        UserWarning,
        "Objects may contain the same data, pushing forward",
    ):
        _ = MirParser.concat((mir_data, mir_copy), force=True)

    # If we want to get at some deeper warnings, we need to make our objects a little
    # less obviously identical. Modify the index codes to allow us to move forward.
    diff_list = {
        "_in_read": ["inhid"],
        "_eng_read": ["inhid"],
        "_bl_read": ["inhid", "blhid"],
        "_sp_read": ["inhid", "blhid", "sphid"],
        "_we_read": ["ints"],
        "_ac_read": ["achid"],
    }

    # These two items let us check to additional errors.
    mir_copy._has_auto = False
    mir_copy._antpos_read["xyz_pos"] = 0.0

    # We want to go through the above arrays and modify them so that concat
    # doesn't error out.
    for key, muck_list in diff_list.items():
        for field in muck_list:
            getattr(mir_copy, key)[field] = getattr(mir_copy, key)[field] + (
                np.max(getattr(mir_copy, key)[field])
            )

    # Make it look as though our copy has two integrations, with one overlapping
    # the other dataset.
    mir_copy._in_read = np.tile(mir_copy._in_read, 2)
    mir_copy._in_read["inhid"][0] = 1
    codes_dict = copy.deepcopy(mir_copy.codes_dict)
    for item in ["ut", "ra", "dec", "vrad"]:
        codes_dict[item][2] = codes_dict[item][1]
    mir_copy._codes_read = MirParser.make_codes_read(codes_dict)

    # Alright, time to check our warnings!
    with uvtest.check_warnings(
        UserWarning,
        [
            "Some (but not all) objects have auto-correlation data",
            "Some objects have different antenna positions than others.",
            "Objects may contain overlapping data, pushing ",
        ],
    ):
        _ = MirParser.concat((mir_data, mir_copy), force=True)


@pytest.mark.parametrize(
    "kern_type,tol,err_type,err_msg",
    [
        ["cubic", -1, ValueError, "tol must be in the range [0, 0.5]."],
        ["abc", 0.5, ValueError, 'Kernel type of "abc" not recognized,'],
    ],
)
def test_generate_chanshift_kernel_errs(mir_data, kern_type, tol, err_type, err_msg):
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
    mir_data.codes_dict["filever"] = filever
    mir_data.bl_data["irec"] = irec

    with pytest.raises(err_type) as err:
        mir_data.redoppler_data()
    assert str(err.value).startswith(err_msg)


@pytest.mark.parametrize("plug_vals", [True, False])
@pytest.mark.parametrize("diff_rx", [True, False])
def test_redoppler_data(mir_data, plug_vals, diff_rx):
    """Verify that redoppler_data behaves as expected."""
    # We have to spoof the filever because the test file is technically v3
    mir_data.codes_dict["filever"] = "4"

    mir_copy = mir_data.copy()
    # This first attempt should basically just be a no-op
    mir_copy.redoppler_data()

    assert mir_data == mir_copy

    # Alright, let's tweak the data now to give us something to compare
    for sphid, nch in zip(mir_data.sp_data["sphid"], mir_data.sp_data["nch"]):
        mir_data.vis_data[sphid]["vis_data"][:] = np.arange(nch)
        mir_data.vis_data[sphid]["vis_flags"][:] = False
        mir_data.raw_data[sphid]["raw_data"][:] = np.arange(nch * 2)
        mir_data.vis_data[sphid]["scale_fac"] = np.int16(0)

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
            assert np.all(mir_data.vis_data[sphid]["vis_data"] == np.arange(nch))
            assert np.all(mir_data.raw_data[sphid]["raw_data"] == np.arange(nch * 2))
        elif chan_shift < 0:
            assert np.all(
                mir_data.vis_data[sphid]["vis_data"][:chan_shift]
                == np.arange(-chan_shift, nch)
            )
            assert np.all(
                mir_data.raw_data[sphid]["raw_data"][: chan_shift * 2]
                == np.arange(-(2 * chan_shift), nch * 2)
            )
        else:
            assert np.all(
                mir_data.vis_data[sphid]["vis_data"][chan_shift:]
                == np.arange((nch - chan_shift))
            )
            assert np.all(
                mir_data.raw_data[sphid]["raw_data"][chan_shift * 2 :]
                == np.arange((nch - chan_shift) * 2)
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

    # ac_read only exists if has_auto=True
    if mir_data._ac_read is not None:
        assert len(mir_data._ac_read) == 2
    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto

    assert len(mir_data._bl_read) == 4

    assert len(mir_data._codes_read) == 99

    assert len(mir_data._eng_read) == 2

    assert len(mir_data._in_read) == 1

    assert len(mir_data.raw_data) == 20

    assert len(mir_data._sp_read) == 20

    assert len(mir_data.vis_data) == 20

    assert len(mir_data._we_read) == 1


def test_mir_remember_me_codes_read(mir_data):
    """
    Mir codes_read checker.

    Make sure that certain values in the codes_read file of the test data set match
    whatwe know to be 'true' at the time of observations.
    """
    assert mir_data._codes_read[0][0] == b"filever"

    assert mir_data._codes_read[0][2] == b"3"

    assert mir_data._codes_read[90][0] == b"ref_time"

    assert mir_data._codes_read[90][1] == 0

    assert mir_data._codes_read[90][2] == b"Jul 24, 2020"

    assert mir_data._codes_read[90][3] == 0

    assert mir_data._codes_read[91][0] == b"ut"

    assert mir_data._codes_read[91][1] == 1

    assert mir_data._codes_read[91][2] == b"Jul 24 2020  4:34:39.00PM"

    assert mir_data._codes_read[91][3] == 0

    assert mir_data._codes_read[93][0] == b"source"

    assert mir_data._codes_read[93][2] == b"3c84"

    assert mir_data._codes_read[97][0] == b"ra"

    assert mir_data._codes_read[97][2] == b"03:19:48.15"

    assert mir_data._codes_read[98][0] == b"dec"

    assert mir_data._codes_read[98][2] == b"+41:30:42.1"


def test_mir_remember_me_in_read(mir_data):
    """
    Mir in_read checker.

    Make sure that certain values in the in_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Check to make sure that things seem right in in_read
    assert np.all(mir_data._in_read["traid"] == 484)

    assert np.all(mir_data._in_read["proid"] == 484)

    assert np.all(mir_data._in_read["inhid"] == 1)

    assert np.all(mir_data._in_read["ints"] == 1)

    assert np.all(mir_data._in_read["souid"] == 1)

    assert np.all(mir_data._in_read["isource"] == 1)

    assert np.all(mir_data._in_read["ivrad"] == 1)

    assert np.all(mir_data._in_read["ira"] == 1)

    assert np.all(mir_data._in_read["idec"] == 1)

    assert np.all(mir_data._in_read["epoch"] == 2000.0)

    assert np.all(mir_data._in_read["tile"] == 0)

    assert np.all(mir_data._in_read["obsflag"] == 0)

    assert np.all(mir_data._in_read["obsmode"] == 0)

    assert np.all(np.round(mir_data._in_read["mjd"]) == 59055)

    assert np.all(mir_data._in_read["spareshort"] == 0)

    assert np.all(mir_data._in_read["spareint6"] == 0)


def test_mir_remember_me_bl_read(mir_data):
    """
    Mir bl_read checker.

    Make sure that certain values in the bl_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Now check bl_read
    assert np.all(mir_data._bl_read["blhid"] == np.arange(1, 5))

    assert np.all(mir_data._bl_read["isb"] == [0, 0, 1, 1])

    assert np.all(mir_data._bl_read["ipol"] == [0, 0, 0, 0])

    assert np.all(mir_data._bl_read["ant1rx"] == [0, 1, 0, 1])

    assert np.all(mir_data._bl_read["ant2rx"] == [0, 1, 0, 1])

    assert np.all(mir_data._bl_read["pointing"] == 0)

    assert np.all(mir_data._bl_read["irec"] == [0, 3, 0, 3])

    assert np.all(mir_data._bl_read["iant1"] == 1)

    assert np.all(mir_data._bl_read["iant2"] == 4)

    assert np.all(mir_data._bl_read["iblcd"] == 2)

    assert np.all(mir_data._bl_read["spareint1"] == 0)

    assert np.all(mir_data._bl_read["spareint2"] == 0)

    assert np.all(mir_data._bl_read["spareint3"] == 0)

    assert np.all(mir_data._bl_read["spareint4"] == 0)

    assert np.all(mir_data._bl_read["spareint5"] == 0)

    assert np.all(mir_data._bl_read["spareint6"] == 0)

    assert np.all(mir_data._bl_read["wtave"] == 0.0)

    assert np.all(mir_data._bl_read["sparedbl4"] == 0.0)

    assert np.all(mir_data._bl_read["sparedbl5"] == 0.0)

    assert np.all(mir_data._bl_read["sparedbl6"] == 0.0)


def test_mir_remember_me_eng_read(mir_data):
    """
    Mir eng_read checker.

    Make sure that certain values in the eng_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    # Now check eng_read
    assert np.all(mir_data._eng_read["antennaNumber"] == [1, 4])

    assert np.all(mir_data._eng_read["padNumber"] == [5, 8])

    assert np.all(mir_data._eng_read["trackStatus"] == 1)

    assert np.all(mir_data._eng_read["commStatus"] == 1)

    assert np.all(mir_data._eng_read["inhid"] == 1)


def test_mir_remember_me_we_read(mir_data):
    """
    Mir we_read checker.

    Make sure that certain values in the we_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    assert np.all(mir_data._we_read["scanNumber"] == 1)

    assert np.all(mir_data._we_read["flags"] == 0)


def test_mir_remember_me_ac_read(mir_data):
    """
    Mir ac_read checker.

    Make sure that certain values in the autoCorrelations file of the test data set
    match what we know to be 'true' at the time of observations.
    """
    # Now check ac_read

    # ac_read only exists if has_auto=True
    if mir_data._ac_read is not None:

        assert np.all(mir_data._ac_read["inhid"] == 1)

        assert np.all(mir_data._ac_read["achid"] == np.arange(1, 3))

        assert np.all(mir_data._ac_read["antenna"] == [1, 4])

        assert np.all(mir_data._ac_read["nchunks"] == 8)

        assert np.all(mir_data._ac_read["datasize"] == 1048576)

    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto


def test_mir_remember_me_sp_read(mir_data):
    """
    Mir sp_read checker.

    Make sure that certain values in the sp_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Now check sp_read
    assert np.all(mir_data._sp_read["sphid"] == np.arange(1, 21))

    assert np.all(mir_data._sp_read["sphid"] == np.arange(1, 21))

    assert np.all(mir_data._sp_read["igq"] == 0)

    assert np.all(mir_data._sp_read["ipq"] == 1)

    assert np.all(mir_data._sp_read["igq"] == 0)

    assert np.all(mir_data._sp_read["iband"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data._sp_read["ipstate"] == 0)

    assert np.all(mir_data._sp_read["tau0"] == 0.0)

    assert np.all(mir_data._sp_read["cabinLO"] == 0.0)

    assert np.all(mir_data._sp_read["corrLO1"] == 0.0)

    assert np.all(mir_data._sp_read["vradcat"] == 0.0)

    assert np.all(mir_data._sp_read["nch"] == [4, 16384, 16384, 16384, 16384] * 4)

    assert np.all(mir_data._sp_read["corrblock"] == [0, 1, 1, 1, 1] * 4)

    assert np.all(mir_data._sp_read["corrchunk"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data._sp_read["correlator"] == 1)

    assert np.all(mir_data._sp_read["iddsmode"] == 0)

    assert np.all(mir_data._sp_read["spareshort"] == 0)

    assert np.all(mir_data._sp_read["spareint3"] == 0)

    assert np.all(mir_data._sp_read["spareint4"] == 0)

    assert np.all(mir_data._sp_read["spareint5"] == 0)

    assert np.all(mir_data._sp_read["spareint6"] == 0)

    assert np.all(mir_data._sp_read["tssb"] == 0.0)

    assert np.all(mir_data._sp_read["fDDS"] == 0.0)

    assert np.all(mir_data._sp_read["sparedbl3"] == 0.0)

    assert np.all(mir_data._sp_read["sparedbl4"] == 0.0)

    assert np.all(mir_data._sp_read["sparedbl5"] == 0.0)

    assert np.all(mir_data._sp_read["sparedbl6"] == 0.0)


def test_mir_remember_me_sch_read(mir_data):
    """
    Mir sch_read checker.

    Make sure that certain values in the sch_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
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
