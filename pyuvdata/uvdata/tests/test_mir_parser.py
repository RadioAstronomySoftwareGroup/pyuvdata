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
from ..mir_parser import MirParser, MirMetaError
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


@pytest.mark.parametrize("data_type", ["none", "raw", "vis"])
def test_mir_write_full(mir_data, tmp_path, data_type):
    """
    Mir write dataset tester.

    Make sure we can round-trip a MIR dataset correctly.
    """
    # We want to clear our the auto data here, since we can't _yet_ write that out
    mir_data.unload_data()
    if data_type != "none":
        mir_data.load_data(load_vis=(data_type == "vis"), apply_tsys=False)

    mir_data._clear_auto()

    # Write out our test dataset
    filepath = os.path.join(tmp_path, "test_write_full_%s.mir" % data_type)

    with uvtest.check_warnings(
        None if (data_type != "none") else UserWarning,
        None if (data_type != "none") else "No data loaded, writing metadata only",
    ):
        mir_data.tofile(filepath)

    # Read in test dataset.
    mir_copy = MirParser(filepath)
    if data_type != "none":
        mir_copy.load_data(load_raw=(data_type == "raw"), apply_tsys=False)

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
    mir_data.raw_data = {}
    mir_data.auto_data = {}
    mir_data._update_filter(update_data=True)
    assert mir_data == mir_copy


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


@pytest.mark.parametrize(
    "unload_data,muck_data,force,err_type,err_msg",
    [
        [None, "in_data", False, ValueError, "Cannot merge objects due to conflicts"],
        [None, "all", False, TypeError, "Cannot add a MirParser object an object of "],
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
        elif muck_data == "all":
            mir_data = np.arange(100)

    if force:
        mir_data.in_data["mjd"] = 0.0

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
        mir_data.codes_data.set_value("code", "1", where=("v_name", "eq", b"filever"))
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
    mir_data.codes_data.set_value("code", filever, where=("v_name", "eq", b"filever"))
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
    mir_data.codes_data.set_value("code", "4", where=("v_name", "eq", b"filever"))

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
