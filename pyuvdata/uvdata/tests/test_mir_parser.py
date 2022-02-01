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
            np.concatenate((bp_soln, np.conj(np.reciprocal(bp_soln)))), (2, 16, 16384),
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


# __add__
# __eq__
# __iadd__
# __init__
# __iter__
# __ne__
# _check_data_index
# _combine_read_arr
# _combine_read_arr_check
# _downselect_data
# _parse_select
# _parse_select_compare
# _read_compass_solns
# _rechunk_auto
# _rechunk_raw
# _rechunk_vis
# _update_filter
# apply_tsys
# calc_int_start
# concat
# convert_raw_to_vis
# convert_vis_to_raw
# copy
# fix_int_start
# fromfile
# load_data
# make_codes_dict
# make_codes_read
# make_packdata
# read_auto_data
# read_packdata
# read_vis_data
# rechunk
# redoppler_data
# scan_auto_data
# scan_int_start
# segment_by_index
# select
# tofile

# Below are a series of checks that are designed to check to make sure that the
# MirParser class is able to produce consistent values from an engineering data
# set (originally stored in /data/engineering/mir_data/200724_16:35:14), to make
# sure that we haven't broken the ability of the reader to handle the data. Since
# this file is the basis for the above checks, we've put this here rather than in
# test_mir_parser.py


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

    assert np.all(mir_data._sp_read["spareint2"] == 0)

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
