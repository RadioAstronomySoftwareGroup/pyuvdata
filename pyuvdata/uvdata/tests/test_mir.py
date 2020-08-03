# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Mir object.

"""
import os

import pytest
import numpy as np

from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.uvdata.mir import mir_parser


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


def test_read_mir_write_uvfits(uv_in_uvfits):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    mir_uv, uvfits_uv, testfile = uv_in_uvfits

    mir_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvfits_uv.read_uvfits(testfile)

    # test fails because of updated history, so this is our workaround for now.
    mir_uv.history = ""
    uvfits_uv.history = ""

    assert mir_uv == uvfits_uv


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


def test_mir_auto_read():
    """
    Mir read tester

    Make sure that Mir autocorrelations are read correctly
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = mir_parser.MirParser(testfile)
    mir_data.ac_data["nchunks"][:] = 4
    mir_data.load_data(load_vis=False, load_auto=True)

    auto_data = mir_data.read_auto_data(testfile, mir_data.ac_data)
    assert np.all(
        np.logical_or(
            auto_data == mir_data.auto_data,
            np.logical_and(np.isnan(auto_data), np.isnan(mir_data.auto_data)),
        )
    )
