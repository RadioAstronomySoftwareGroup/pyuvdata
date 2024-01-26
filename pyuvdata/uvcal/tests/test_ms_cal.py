# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for ms_cal object."""
import pytest

import pyuvdata.tests as uvtest

from ... import ms_utils

pytest.importorskip("casacore")


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
