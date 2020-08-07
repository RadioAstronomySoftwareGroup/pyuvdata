# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MirParser class.

Performs a series of tests on the MirParser, which is the python-based reader for MIR
data in pyuvdata. Tests in this module are specific to the way that MIR is read into
python, not neccessarily how pyuvdata (by way of the UVData class) interacts with that
data.
"""
import os

import pytest
import numpy as np

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


def test_mir_parser_index_uniqueness(mir_data_object):
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


def test_mir_parser_index_valid(mir_data_object):
    """
    Mir index validity check

    Make sure that all indexes are non-negative
    """
    mir_data = mir_data_object

    assert np.all(mir_data.in_read["inhid"] >= 0)

    assert np.all(mir_data.bl_read["blhid"] >= 0)

    assert np.all(mir_data.sp_read["sphid"] >= 0)


def test_mir_parser_index_linked(mir_data_object):
    """
    Mir index link check

    Make sure that all referenced indicies have matching pairs in their parent tables
    """
    mir_data = mir_data_object
    inhid_set = set(np.unique(mir_data.in_read["inhid"]))

    assert set(np.unique(mir_data.ac_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.bl_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.sp_read["inhid"])).issubset(inhid_set)

    blhid_set = set(np.unique(mir_data.bl_read["blhid"]))

    assert set(np.unique(mir_data.sp_read["blhid"])).issubset(blhid_set)
