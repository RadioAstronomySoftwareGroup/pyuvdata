# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Mir object.

"""
import os
import shutil
import copy
import numpy as np
import pytest
from astropy.time import Time, TimeDelta
from astropy import constants as const

from pyuvdata import UVData
from ..miriad import Miriad
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


@pytest.fixture
def uv_in_uvfits(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = str(tmp_path / "outtest_mir.uvfits")

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
    assert mir_uv == uvfits_uv
