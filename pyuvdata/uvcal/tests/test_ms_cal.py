# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for ms_cal object."""
import os

import pytest

from ... import ms_utils
from ... import tests as uvtest
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
    """Make function level NRAO ms object."""
    uvobj = sma_pcal_main.copy()
    yield uvobj


@pytest.mark.parametrize("check_warning", [True, False])
@pytest.mark.parametrize(
    "frame,epoch,msg",
    (
        ["fk5", 1991.1, "Frame fk5 (epoch 1991.1) does not have a corresponding match"],
        ["fk4", 1991.1, "Frame fk4 (epoch 1991.1) does not have a corresponding match"],
        ["icrs", 2021.0, "Frame icrs (epoch 2021) does not have a corresponding"],
    ),
)
def test_parse_pyuvdata_frame_ref_errors(check_warning, frame, epoch, msg):
    """
    Test errors with matching CASA frames to astropy frame/epochs
    """
    if check_warning:
        with uvtest.check_warnings(UserWarning, match=msg):
            ms_utils._parse_pyuvdata_frame_ref(frame, epoch, raise_error=False)
    else:
        with pytest.raises(ValueError) as cm:
            ms_utils._parse_pyuvdata_frame_ref(frame, epoch)
        assert str(cm.value).startswith(msg)


@pytest.mark.parametrize("check_warning", [True, False])
@pytest.mark.parametrize(
    "frame,errtype,msg",
    (
        ["JNAT", NotImplementedError, "Support for the JNAT frame is not yet"],
        ["AZEL", NotImplementedError, "Support for the AZEL frame is not yet"],
        ["GALACTIC", NotImplementedError, "Support for the GALACTIC frame is not yet"],
        ["ABC", ValueError, "The coordinate frame ABC is not one of the supported"],
        ["123", ValueError, "The coordinate frame 123 is not one of the supported"],
    ),
)
def test_parse_casa_frame_ref_errors(check_warning, frame, errtype, msg):
    """
    Test errors with matching CASA frames to astropy frame/epochs
    """
    if check_warning:
        with uvtest.check_warnings(UserWarning, match=msg):
            ms_utils._parse_casa_frame_ref(frame, raise_error=False)
    else:
        with pytest.raises(errtype) as cm:
            ms_utils._parse_casa_frame_ref(frame)
        assert str(cm.value).startswith(msg)


def test_ms_cal_wideband_loopback(sma_pcal, tmp_path):
    uvcal = UVCal()
    testfile = os.path.join(tmp_path, "ms_cal_loopback.ms")
    sma_pcal.write_ms_cal(testfile, clobber=True)

    uvcal.read(testfile)
    # Check that the histories line up
    assert sma_pcal.history in uvcal.history
    assert sma_pcal.__eq__(uvcal, allowed_failures=allowed_failures)
