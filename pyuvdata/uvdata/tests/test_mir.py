# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Mir object.

"""
import os

import pytest
import numpy as np

from ... import UVData
from ...data import DATA_PATH
from ...uvdata.mir import mir_parser


@pytest.fixture
def mir_data_object():
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = mir_parser.MirParser(
        testfile, load_vis=True, load_raw=True, load_auto=True,
    )

    yield mir_data

    # cleanup
    del mir_data


@pytest.fixture
def uv_in_uvfits(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = str(tmp_path / "outtest_mir.uvfits")

    # Currently only one source and one spectral window are supported.
    uv_in.read(testfile)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.fixture
def uv_in_uvh5(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = str(tmp_path / "outtest_mir.uvh5")

    # Currently only one source and one spectral window are supported.
    uv_in.read(testfile)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


def test_read_mir_write_uvfits(uv_in_uvfits):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    mir_uv, uvfits_uv, testfile = uv_in_uvfits

    mir_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvfits_uv.read_uvfits(testfile)

    # Check the history first via find
    assert 0 == uvfits_uv.history.find(
        mir_uv.history + "  Read/written with pyuvdata version:"
    )

    mir_uv.history = uvfits_uv.history
    assert mir_uv == uvfits_uv


def test_read_mir_write_uvh5(uv_in_uvh5):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    mir_uv, uvh5_uv, testfile = uv_in_uvh5

    mir_uv.write_uvh5(testfile)
    uvh5_uv.read_uvh5(testfile)

    # Check the history first via find
    assert 0 == uvh5_uv.history.find(
        mir_uv.history + "  Read/written with pyuvdata version:"
    )

    # test fails because of updated history, so this is our workaround for now.
    mir_uv.history = uvh5_uv.history

    assert mir_uv == uvh5_uv


def test_write_mir(uv_in_uvfits, err_type=NotImplementedError):
    """
    Mir writer test

    Check and make sure that attempts to use the writer return a
    'not implemented; error.
    """
    mir_uv, uvfits_uv, testfile = uv_in_uvfits

    # Check and see if the correct error is raised
    with pytest.raises(err_type):
        mir_uv.write_mir("dummy.mir")


def test_read_mir_no_records(
    err_type=IndexError, err_msg="No valid records matching those selections!"
):
    """
    Mir no-records check

    Make sure that mir correctly handles the case where no matching records are found
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_in = UVData()
    with pytest.raises(err_type, match=err_msg):
        uv_in.read_mir(testfile, isource=-1)


def test_mir_auto_read(
    err_type=IndexError, err_msg="Could not determine auto-correlation record size!"
):
    """
    Mir read tester

    Make sure that Mir autocorrelations are read correctly
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = mir_parser.MirParser(testfile)
    with pytest.raises(err_type, match=err_msg):
        ac_data = mir_data.scan_auto_data(testfile, nchunks=999)

    ac_data = mir_data.scan_auto_data(testfile)
    assert np.all(ac_data["nchunks"] == 8)

    mir_data.load_data(load_vis=False, load_auto=True)

    # Select the relevant auto records, which should be for spwin 0-3
    auto_data = mir_data.read_auto_data(testfile, ac_data)[:, 0:4, :, :]
    assert np.all(
        np.logical_or(
            auto_data == mir_data.auto_data,
            np.logical_and(np.isnan(auto_data), np.isnan(mir_data.auto_data)),
        )
    )
    mir_data.unload_data()


def test_mir_index_uniqueness(mir_data_object):
    """
    Mir index uniqueness check

    Make sure that there are no duplicate indicies for things that are primary keys
    for the various table-like structures that are used in MIR
    """
    mir_data = mir_data_object

    inhid_list = mir_data.in_read["inhid"]
    assert np.all(np.unique(inhid_list) == sorted(inhid_list))

    blhid_list = mir_data.bl_read["blhid"]
    assert np.all(np.unique(blhid_list) == sorted(blhid_list))

    sphid_list = mir_data.sp_read["sphid"]
    assert np.all(np.unique(sphid_list) == sorted(sphid_list))


def test_mir_index_valid(mir_data_object):
    """
    Mir index validity check

    Make sure that all indexes are non-negative
    """
    mir_data = mir_data_object

    assert np.all(mir_data.in_read["inhid"] >= 0)

    assert np.all(mir_data.bl_read["blhid"] >= 0)

    assert np.all(mir_data.sp_read["sphid"] >= 0)


def test_mir_index_linked(mir_data_object):
    """
    Mir index link check

    Make sure that all referenced indicies have matching pairs in their parent tables
    """
    mir_data = mir_data_object
    inhid_set = set(np.unique(mir_data.in_read["inhid"]))
    blhid_set = set(np.unique(mir_data.bl_read["blhid"]))

    assert set(np.unique(mir_data.ac_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.bl_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.sp_read["inhid"])).issubset(inhid_set)

    blhid_set = set(np.unique(mir_data.bl_read["blhid"]))

    assert set(np.unique(mir_data.sp_read["blhid"])).issubset(blhid_set)
