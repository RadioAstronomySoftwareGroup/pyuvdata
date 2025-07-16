# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for ms utils."""

import os

import numpy as np
import pytest

import pyuvdata.utils.io.ms as ms_utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

pytest.importorskip("casacore")


@pytest.mark.parametrize("check_warning", [True, False])
@pytest.mark.parametrize(
    "frame,epoch,msg",
    (
        ["fk5", 1991.1, r"Frame fk5 \(epoch 1991.1\) does not have a corresponding"],
        ["fk4", 1991.1, r"Frame fk4 \(epoch 1991.1\) does not have a corresponding"],
        ["icrs", 2021.0, r"Frame icrs \(epoch 2021\) does not have a corresponding"],
    ),
)
def test_parse_pyuvdata_frame_ref_errors(check_warning, frame, epoch, msg):
    """
    Test errors with matching CASA frames to astropy frame/epochs
    """
    if check_warning:
        with check_warnings(UserWarning, match=msg):
            ms_utils._parse_pyuvdata_frame_ref(frame, epoch, raise_error=False)
    else:
        with pytest.raises(ValueError, match=msg):
            ms_utils._parse_pyuvdata_frame_ref(frame, epoch)


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
        with check_warnings(UserWarning, match=msg):
            ms_utils._parse_casa_frame_ref(frame, raise_error=False)
    else:
        with pytest.raises(errtype, match=msg):
            ms_utils._parse_casa_frame_ref(frame)


@pytest.mark.parametrize(
    "invert_check,errtype,errmsg",
    [
        [False, FileNotFoundError, "not found or not recognized as an MS table."],
        [True, FileExistsError, " already exists."],
    ],
)
def test_ms_file_checks(invert_check, errtype, errmsg):
    with pytest.raises(errtype, match=errmsg):
        ms_utils._ms_utils_call_checks(
            os.path.join(DATA_PATH, "sma_test.mir"), invert_check=invert_check
        )


@pytest.mark.parametrize(
    "make_change,warntype,warnmsg",
    [
        [False, None, None],
        [None, None, None],
        [True, UserWarning, "Different windows in this MS file contain different"],
    ],
)
def test_ms_source_multispw(tmp_path, make_change, warntype, warnmsg):
    from casacore import tables

    filename = os.path.join(tmp_path, "source_multispw.ms")

    with tables.default_ms(filename):
        pass

    with tables.table(
        filename + "::SOURCE",
        readonly=False,
        tabledesc=tables.required_ms_desc("SOURCE"),
    ) as tb_source:
        tb_source.addrows(2)
        tb_source.putcol("SPECTRAL_WINDOW_ID", [0, 1])
        if make_change is None:
            tb_source.putcol("TIME", np.array([0.0, 1.0]))
        elif make_change:
            tb_source.putcol("DIRECTION", np.array([[1.0, 2.0], [3.0, 4.0]]))
        else:
            tb_source.putcol("DIRECTION", np.array([[1.0, 2.0], [1.0, 2.0]]))

    with check_warnings(warntype, warnmsg):
        sou_dict = ms_utils.read_ms_source(filename)
    if make_change is None:
        assert np.array_equal(sou_dict[0]["cat_lon"], [0.0, 0.0])
        assert np.array_equal(sou_dict[0]["cat_lat"], [0.0, 0.0])
    else:
        assert sou_dict[0]["cat_lon"] == (3.0 if make_change else 1.0)
        assert sou_dict[0]["cat_lat"] == (4.0 if make_change else 2.0)


def test_field_no_ref(tmp_path):
    from casacore import tables

    filename = os.path.join(tmp_path, "field_noref.ms")

    with tables.default_ms(filename):
        pass

    with tables.table(
        filename + "::FIELD", readonly=False, tabledesc=tables.required_ms_desc("FIELD")
    ) as tb_field:
        tb_field.putcolkeyword("PHASE_DIR", "MEASINFO", {"type": "direction"})

    with check_warnings(UserWarning, "Coordinate reference frame not detected"):
        field_dict = ms_utils.read_ms_field(filename)

    assert field_dict["frame"] == "icrs"
    assert field_dict["epoch"] == 2000.0


def test_read_ms_pointing_err():
    filename = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")

    with pytest.raises(NotImplementedError):
        ms_utils.read_ms_pointing(filename)


def test_read_ms_history_err(tmp_path):
    with pytest.raises(FileNotFoundError):
        ms_utils.read_ms_history(os.path.join(tmp_path, "foo"), "abc", raise_err=True)


def test_history_roundtrip(tmp_path):
    from casacore import tables

    filename = os.path.join(tmp_path, "history_test.ms")

    with tables.default_ms(filename):
        pass

    pyuvdata_hist = "pyuvdata_hist"

    exp_history = (
        "Begin measurement set history\n"
        "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE;"
        "OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
        ";;APP1;MSG1;0;-1;pyuvdata;INFO;5000000000.0\n"
        ";;APP2;MSG2;1;-1;pyuvdata;INFO;5100000000.0\n"
        "End measurement set history.\n"
    ) + pyuvdata_hist

    with tables.table(
        filename + "::HISTORY",
        readonly=False,
        tabledesc=tables.required_ms_desc("HISTORY"),
    ) as history_table:
        history_table.addrows(2)

        # Skip APP_PARAMS - we've seen MS files where this is empty so test this
        # Plug in an empty array into CLI_COMMAND - some LOFAR-based files have this
        # which caused a bug in the past (see Issue #1587)
        history_table.putcol("CLI_COMMAND", np.asarray([[], []], dtype="str"))
        history_table.putcol("APPLICATION", ["APP1", "APP2"])
        history_table.putcol("MESSAGE", ["MSG1", "MSG2"])
        history_table.putcol("OBJECT_ID", [0, 1])
        history_table.putcol("OBSERVATION_ID", [-1, -1])
        history_table.putcol("ORIGIN", ["pyuvdata", "pyuvdata"])
        history_table.putcol("PRIORITY", ["INFO", "INFO"])
        history_table.putcol("TIME", [5e9, 5.1e9])

    read_str = ms_utils.read_ms_history(filename, pyuvdata_hist)

    assert exp_history == read_str

    writefile = os.path.join(tmp_path, "history_write.ms")
    with tables.default_ms(writefile):
        pass

    ms_utils.write_ms_history(writefile, history=read_str)
    write_str = ms_utils.read_ms_history(writefile, pyuvdata_hist)
    assert write_str == (read_str + "\n")
