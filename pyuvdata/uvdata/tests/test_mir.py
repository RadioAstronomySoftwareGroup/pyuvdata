# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Mir object.

"""
import os

import pytest

from pyuvdata import UVData
from pyuvdata.data import DATA_PATH


@pytest.fixture
def uv_in_uvfits(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = str(tmp_path / "outtest_mir.uvfits")

    # Currently only one source and one spectral window are supported.
    uv_in.read_mir(testfile, isource=1, irec=0, isb=0, corrchunk=1)
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
