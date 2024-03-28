# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for ms_cal object."""
import os

import pytest

from ...data import DATA_PATH
from ..uvcal import UVCal

pytest.importorskip("casacore")

allowed_failures = ["_filename", "_history"]


@pytest.fixture(scope="session")
def sma_pcal_main():
    uvobj = UVCal()
    testfile = os.path.join(DATA_PATH, "sma.ms.pha.gcal")
    uvobj.read(testfile)

    yield uvobj


@pytest.fixture(scope="function")
def sma_pcal(sma_pcal_main):
    """Make function level SMA MS phase cal object."""
    uvobj = sma_pcal_main.copy()
    yield uvobj


@pytest.fixture(scope="session")
def sma_dcal_main():
    uvobj = UVCal()
    testfile = os.path.join(DATA_PATH, "sma.ms.dcal")
    uvobj.read(testfile)

    yield uvobj


@pytest.fixture(scope="function")
def sma_dcal(sma_dcal_main):
    """Make function level SMA MS delay cal object."""
    uvobj = sma_dcal_main.copy()
    yield uvobj


@pytest.fixture(scope="session")
def sma_bcal_main():
    uvobj = UVCal()
    testfile = os.path.join(DATA_PATH, "sma.ms.bcal")
    uvobj.read(testfile)

    yield uvobj


@pytest.fixture(scope="function")
def sma_bcal(sma_bcal_main):
    """Make function level SMA MS bandpass cal object."""
    uvobj = sma_bcal_main.copy()
    yield uvobj


@pytest.mark.parametrize(
    "write_func,filename",
    [["write_ms_cal", "ms_cal_loopback.ms"], ["write_calh5", "ms_cal_loopback.calh5"]],
)
def test_ms_cal_wideband_loopback(sma_pcal, tmp_path, write_func, filename):
    uvcal = UVCal()
    testfile = os.path.join(tmp_path, filename)
    getattr(sma_pcal, write_func)(testfile, clobber=True)

    uvcal.read(testfile, use_future_array_shapes=True)
    # Check that the histories line up
    assert sma_pcal.history in uvcal.history
    assert sma_pcal.__eq__(uvcal, allowed_failures=allowed_failures)


@pytest.mark.parametrize(
    "write_func,filename",
    [["write_ms_cal", "ms_cal_delay.ms"], ["write_calh5", "ms_cal_delay.calh5"]],
)
def test_ms_cal_delay_loopback(sma_dcal, tmp_path, write_func, filename):
    uvcal = UVCal()
    testfile = os.path.join(tmp_path, filename)
    getattr(sma_dcal, write_func)(testfile, clobber=True)

    uvcal.read(testfile, use_future_array_shapes=True)
    # Check that the histories line up
    assert sma_dcal.history in uvcal.history
    assert sma_dcal.__eq__(uvcal, allowed_failures=allowed_failures)


@pytest.mark.parametrize(
    "write_func,filename",
    [["write_ms_cal", "ms_cal_bandpass.ms"], ["write_calh5", "ms_cal_bandpass.calh5"]],
)
def test_ms_cal_bandpass_loopback(sma_bcal, tmp_path, write_func, filename):
    uvcal = UVCal()
    testfile = os.path.join(tmp_path, filename)
    getattr(sma_bcal, write_func)(testfile, clobber=True)

    uvcal.read(testfile, use_future_array_shapes=True)
    # Check that the histories line up
    assert sma_bcal.history in uvcal.history
    assert sma_bcal.__eq__(uvcal, allowed_failures=allowed_failures)
