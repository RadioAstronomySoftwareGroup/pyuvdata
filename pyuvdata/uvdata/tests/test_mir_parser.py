# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MirParser class.

Performs a series of tests on the MirParser, which is the python-based reader for MIR
data in pyuvdata. Tests in this module are specific to the way that MIR is read into
python, not neccessarily how pyuvdata (by way of the UVData class) interacts with that
data.
"""
import numpy as np
import pytest


def test_mir_parser_index_uniqueness(mir_data):
    """
    Mir index uniqueness check

    Make sure that there are no duplicate indicies for things that are primary keys
    for the various table-like structures that are used in MIR
    """
    inhid_list = mir_data.in_read["inhid"]
    assert np.all(np.unique(inhid_list) == sorted(inhid_list))

    blhid_list = mir_data.bl_read["blhid"]
    assert np.all(np.unique(blhid_list) == sorted(blhid_list))

    sphid_list = mir_data.sp_read["sphid"]
    assert np.all(np.unique(sphid_list) == sorted(sphid_list))


def test_mir_parser_index_valid(mir_data):
    """
    Mir index validity check

    Make sure that all indexes are non-negative
    """
    assert np.all(mir_data.in_read["inhid"] >= 0)

    assert np.all(mir_data.bl_read["blhid"] >= 0)

    assert np.all(mir_data.sp_read["sphid"] >= 0)


def test_mir_parser_index_linked(mir_data):
    """
    Mir index link check

    Make sure that all referenced indicies have matching pairs in their parent tables
    """
    inhid_set = set(np.unique(mir_data.in_read["inhid"]))

    # Should not exist is has_auto=False
    if mir_data.ac_read is not None:
        assert set(np.unique(mir_data.ac_read["inhid"])).issubset(inhid_set)
    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto

    assert set(np.unique(mir_data.bl_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.eng_read["inhid"])).issubset(inhid_set)

    assert set(np.unique(mir_data.sp_read["inhid"])).issubset(inhid_set)

    blhid_set = set(np.unique(mir_data.bl_read["blhid"]))

    assert set(np.unique(mir_data.sp_read["blhid"])).issubset(blhid_set)


def test_mir_parser_unload_data(mir_data):
    """
    Check that the unload_data function works as expected
    """
    attr_list = ["vis_data", "raw_data", "auto_data", "raw_scale_fac"]

    for attr in attr_list:
        assert getattr(mir_data, attr) is not None

    mir_data.unload_data()

    for attr in attr_list:
        assert getattr(mir_data, attr) is None


@pytest.mark.parametrize("filter_type", ["use_in", "use_bl", "use_sp"])
def test_mir_parser_update_filter(mir_data, filter_type):
    """
    Verify that filtering operations work as expected.
    """
    getattr(mir_data, filter_type)[:] = False
    mir_data._update_filter()

    attr_list = ["in_data", "bl_data", "eng_data", "sp_data", "ac_data"]
    for attr in attr_list:
        assert len(getattr(mir_data, attr)) == 0


# Below are a series of checks that are designed to check to make sure that the
# MirParser class is able to produce consistent values from an engineering data
# set (originally stored in /data/engineering/mir_data/200724_16:35:14), to make
# sure that we haven't broken the ability of the reader to handle the data. Since
# this file is the basis for the above checks, we've put this here rather than in
# test_mir_parser.py


def test_mir_remember_me_record_lengths(mir_data):
    """
    Mir record length checker

    Make sure the test file contains the right number of records
    """
    # Check to make sure we've got the right number of records everywhere

    # ac_read only exists if has_auto=True
    if mir_data.ac_read is not None:
        assert len(mir_data.ac_read) == 2
    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto

    assert len(mir_data.bl_read) == 4

    assert len(mir_data.codes_read) == 99

    assert len(mir_data.eng_read) == 2

    assert len(mir_data.in_read) == 1

    assert len(mir_data.raw_data) == 20

    assert len(mir_data.raw_scale_fac) == 20

    assert len(mir_data.sp_read) == 20

    assert len(mir_data.vis_data) == 20

    assert len(mir_data.we_read) == 1


def test_mir_remember_me_codes_read(mir_data):
    """
    Mir codes_read checker.

    Make sure that certain values in the codes_read file of the test data set match
    whatwe know to be 'true' at the time of observations.
    """
    assert mir_data.codes_read[0][0] == b"filever"

    assert mir_data.codes_read[0][2] == b"3"

    assert mir_data.codes_read[90][0] == b"ref_time"

    assert mir_data.codes_read[90][1] == 0

    assert mir_data.codes_read[90][2] == b"Jul 24, 2020"

    assert mir_data.codes_read[90][3] == 0

    assert mir_data.codes_read[91][0] == b"ut"

    assert mir_data.codes_read[91][1] == 1

    assert mir_data.codes_read[91][2] == b"Jul 24 2020  4:34:39.00PM"

    assert mir_data.codes_read[91][3] == 0

    assert mir_data.codes_read[93][0] == b"source"

    assert mir_data.codes_read[93][2] == b"3c84"

    assert mir_data.codes_read[97][0] == b"ra"

    assert mir_data.codes_read[97][2] == b"03:19:48.15"

    assert mir_data.codes_read[98][0] == b"dec"

    assert mir_data.codes_read[98][2] == b"+41:30:42.1"


def test_mir_remember_me_in_read(mir_data):
    """
    Mir in_read checker.

    Make sure that certain values in the in_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Check to make sure that things seem right in in_read
    assert np.all(mir_data.in_read["traid"] == 484)

    assert np.all(mir_data.in_read["proid"] == 484)

    assert np.all(mir_data.in_read["inhid"] == 1)

    assert np.all(mir_data.in_read["ints"] == 1)

    assert np.all(mir_data.in_read["souid"] == 1)

    assert np.all(mir_data.in_read["isource"] == 1)

    assert np.all(mir_data.in_read["ivrad"] == 1)

    assert np.all(mir_data.in_read["ira"] == 1)

    assert np.all(mir_data.in_read["idec"] == 1)

    assert np.all(mir_data.in_read["epoch"] == 2000.0)

    assert np.all(mir_data.in_read["tile"] == 0)

    assert np.all(mir_data.in_read["obsflag"] == 0)

    assert np.all(mir_data.in_read["obsmode"] == 0)

    assert np.all(np.round(mir_data.in_read["mjd"]) == 59055)

    assert np.all(mir_data.in_read["spareshort"] == 0)

    assert np.all(mir_data.in_read["spareint6"] == 0)


def test_mir_remember_me_bl_read(mir_data):
    """
    Mir bl_read checker.

    Make sure that certain values in the bl_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Now check bl_read
    assert np.all(mir_data.bl_read["blhid"] == np.arange(1, 5))

    assert np.all(mir_data.bl_read["isb"] == [0, 0, 1, 1])

    assert np.all(mir_data.bl_read["ipol"] == [0, 0, 0, 0])

    assert np.all(mir_data.bl_read["ant1rx"] == [0, 1, 0, 1])

    assert np.all(mir_data.bl_read["ant2rx"] == [0, 1, 0, 1])

    assert np.all(mir_data.bl_read["pointing"] == 0)

    assert np.all(mir_data.bl_read["irec"] == [0, 3, 0, 3])

    assert np.all(mir_data.bl_read["iant1"] == 1)

    assert np.all(mir_data.bl_read["iant2"] == 4)

    assert np.all(mir_data.bl_read["iblcd"] == 2)

    assert np.all(mir_data.bl_read["spareint1"] == 0)

    assert np.all(mir_data.bl_read["spareint2"] == 0)

    assert np.all(mir_data.bl_read["spareint3"] == 0)

    assert np.all(mir_data.bl_read["spareint4"] == 0)

    assert np.all(mir_data.bl_read["spareint5"] == 0)

    assert np.all(mir_data.bl_read["spareint6"] == 0)

    assert np.all(mir_data.bl_read["sparedbl3"] == 0.0)

    assert np.all(mir_data.bl_read["sparedbl4"] == 0.0)

    assert np.all(mir_data.bl_read["sparedbl5"] == 0.0)

    assert np.all(mir_data.bl_read["sparedbl6"] == 0.0)


def test_mir_remember_me_eng_read(mir_data):
    """
    Mir bl_read checker.

    Make sure that certain values in the eng_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    # Now check eng_read
    assert np.all(mir_data.eng_read["antennaNumber"] == [1, 4])

    assert np.all(mir_data.eng_read["padNumber"] == [5, 8])

    assert np.all(mir_data.eng_read["trackStatus"] == 1)

    assert np.all(mir_data.eng_read["commStatus"] == 1)

    assert np.all(mir_data.eng_read["inhid"] == 1)


def test_mir_remember_me_ac_read(mir_data):
    """
    Mir bl_read checker.

    Make sure that certain values in the autoCorrelations file of the test data set
    match what we know to be 'true' at the time of observations.
    """
    # Now check ac_read

    # ac_read only exists if has_auto=True
    if mir_data.ac_read is not None:

        assert np.all(mir_data.ac_read["inhid"] == 1)

        assert np.all(mir_data.ac_read["achid"] == np.arange(1, 3))

        assert np.all(mir_data.ac_read["antenna"] == [1, 4])

        assert np.all(mir_data.ac_read["nchunks"] == 8)

        assert np.all(mir_data.ac_read["datasize"] == 1048596)

        assert np.all(mir_data.we_read["scanNumber"] == 1)

        assert np.all(mir_data.we_read["flags"] == 0)

    else:
        # This should only occur when has_auto=False
        assert not mir_data._has_auto


def test_mir_remember_me_sp_read(mir_data):
    """
    Mir sp_read checker.

    Make sure that certain values in the sp_read file of the test data set match what
    we know to be 'true' at the time of observations, including that spare values are
    stored as zero.
    """
    # Now check sp_read
    assert np.all(mir_data.sp_read["sphid"] == np.arange(1, 21))

    assert np.all(mir_data.sp_read["sphid"] == np.arange(1, 21))

    assert np.all(mir_data.sp_read["igq"] == 0)

    assert np.all(mir_data.sp_read["ipq"] == 1)

    assert np.all(mir_data.sp_read["igq"] == 0)

    assert np.all(mir_data.sp_read["iband"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data.sp_read["ipstate"] == 0)

    assert np.all(mir_data.sp_read["tau0"] == 0.0)

    assert np.all(mir_data.sp_read["cabinLO"] == 0.0)

    assert np.all(mir_data.sp_read["corrLO1"] == 0.0)

    assert np.all(mir_data.sp_read["vradcat"] == 0.0)

    assert np.all(mir_data.sp_read["nch"] == [4, 16384, 16384, 16384, 16384] * 4)

    assert np.all(mir_data.sp_read["corrblock"] == [0, 1, 1, 1, 1] * 4)

    assert np.all(mir_data.sp_read["corrchunk"] == [0, 1, 2, 3, 4] * 4)

    assert np.all(mir_data.sp_read["correlator"] == 1)

    assert np.all(mir_data.sp_read["spareint2"] == 0)

    assert np.all(mir_data.sp_read["spareint3"] == 0)

    assert np.all(mir_data.sp_read["spareint4"] == 0)

    assert np.all(mir_data.sp_read["spareint5"] == 0)

    assert np.all(mir_data.sp_read["spareint6"] == 0)

    assert np.all(mir_data.sp_read["sparedbl1"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl2"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl3"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl4"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl5"] == 0.0)

    assert np.all(mir_data.sp_read["sparedbl6"] == 0.0)


def test_mir_remember_me_sch_read(mir_data):
    """
    Mir sch_read checker.

    Make sure that certain values in the sch_read file of the test data set match what
    we know to be 'true' at the time of observations.
    """
    # Now check sch_read related values. Thanks to a glitch in the data recorder,
    # all of the pseudo-cont values are the same
    assert np.all(mir_data.raw_scale_fac[0::5] == [-26] * 4)

    assert (
        np.array(mir_data.raw_data[0::5]).flatten().tolist()
        == [-4302, -20291, -5261, -21128, -4192, -19634, -4999, -16346] * 4
    )
