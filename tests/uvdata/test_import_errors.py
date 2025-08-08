# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for import error handling."""

import pytest

import pyuvdata.utils.io.ms as ms_utils
from pyuvdata import UVData
from pyuvdata.datasets import fetch_data


def test_ms_file_checks_no_casacore():
    try:
        import casacore  # noqa
    except ImportError:
        with pytest.raises(ImportError, match=ms_utils.no_casa_message):
            ms_utils._ms_utils_call_checks("foo")


def test_ms_read_no_casacore(casa_uvfits):
    uvd = UVData()

    try:
        import casacore  # noqa
    except ImportError:
        testfile = fetch_data("vla_casa_tutorial_ms")
        with pytest.raises(ImportError, match=ms_utils.no_casa_message):
            uvd.read_ms(testfile)

        uvd = casa_uvfits
        with pytest.raises(ImportError, match=ms_utils.no_casa_message):
            uvd.write_ms("foo")


def test_miriad_windows():
    uvd = UVData()

    try:
        import pyuvdata.uvdata.aipy_extracts  # noqa
    except ImportError:
        testfile = fetch_data("paper_2014_miriad")
        with pytest.raises(
            ImportError,
            match="The miriad extension is not built but is required for reading "
            "miriad files. Note that miriad is currently not supported on Windows.",
        ):
            uvd.read_miriad(testfile)
