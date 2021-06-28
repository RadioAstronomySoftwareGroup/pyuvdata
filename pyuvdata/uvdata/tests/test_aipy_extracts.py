# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for aipy_extracts

"""
import os
import shutil
import pytest
import numpy as np

from pyuvdata.data import DATA_PATH


aipy_extracts = pytest.importorskip("pyuvdata.uvdata.aipy_extracts")


def test_bl2ij():
    """Test bl2ij function"""
    # small baseline number
    bl = 258
    assert aipy_extracts.bl2ij(bl)[0] == 0
    assert aipy_extracts.bl2ij(bl)[1] == 1

    # large baseline number
    bl = 67587
    assert aipy_extracts.bl2ij(bl)[0] == 0
    assert aipy_extracts.bl2ij(bl)[1] == 2
    return


def test_ij2bl():
    """Test ij2bl function"""
    # test < 256 antennas
    i = 1
    j = 2
    assert aipy_extracts.ij2bl(i, j) == 515

    # test > 256 antennas
    i = 2
    j = 257
    assert aipy_extracts.ij2bl(i, j) == 71938

    # test case where i > j
    i = 257
    j = 2
    assert aipy_extracts.ij2bl(i, j) == 71938
    return


def test_parse_ants():
    """Test parsing ant strings to tuples"""
    nants = 4
    cases = {
        "all": [],
        "auto": [("auto", 1)],
        "cross": [("auto", 0)],
        "-auto": [("auto", 0)],
        "-cross": [("auto", 1)],
        "all,cross": [("auto", 0)],  # Masks are and'd, should be equal to just cross
        "0_1": [(aipy_extracts.ij2bl(0, 1), 1)],
        "0_1,1_2": [(aipy_extracts.ij2bl(0, 1), 1), (aipy_extracts.ij2bl(1, 2), 1)],
        "0x_1x": [(aipy_extracts.ij2bl(0, 1), 1, "xx")],
        "0_1x": [
            (aipy_extracts.ij2bl(0, 1), 1, "xx"),
            (aipy_extracts.ij2bl(0, 1), 1, "yx"),
        ],
        "0y_1": [
            (aipy_extracts.ij2bl(0, 1), 1, "yx"),
            (aipy_extracts.ij2bl(0, 1), 1, "yy"),
        ],
        "(0x,0y)_1x": [
            (aipy_extracts.ij2bl(0, 1), 1, "xx"),
            (aipy_extracts.ij2bl(0, 1), 1, "yx"),
        ],
        "(0,1)_2": [(aipy_extracts.ij2bl(0, 2), 1), (aipy_extracts.ij2bl(1, 2), 1)],
        "0_(1,2)": [(aipy_extracts.ij2bl(0, 1), 1), (aipy_extracts.ij2bl(0, 2), 1)],
        "(0,1)_(2,3)": [
            (aipy_extracts.ij2bl(0, 2), 1),
            (aipy_extracts.ij2bl(0, 3), 1),
            (aipy_extracts.ij2bl(1, 2), 1),
            (aipy_extracts.ij2bl(1, 3), 1),
        ],
        "0_(1,-2)": [(aipy_extracts.ij2bl(0, 1), 1), (aipy_extracts.ij2bl(0, 2), 0)],
        "(-0,1)_(2,-3)": [
            (aipy_extracts.ij2bl(0, 2), 0),
            (aipy_extracts.ij2bl(0, 3), 0),
            (aipy_extracts.ij2bl(1, 2), 1),
            (aipy_extracts.ij2bl(1, 3), 0),
        ],
        "0,1,all": [],
    }
    for i in range(nants):
        cases[str(i)] = [(aipy_extracts.ij2bl(x, i), 1) for x in range(nants)]
        cases["-" + str(i)] = [(aipy_extracts.ij2bl(x, i), 0) for x in range(nants)]
    # inelegantly paste on the new pol parsing flag on the above tests
    # XXX really should add some new tests for the new pol parsing
    for k in cases:
        cases[k] = [(v + (-1,))[:3] for v in cases[k]]
    for ant_str in cases:
        assert aipy_extracts.parse_ants(ant_str, nants) == cases[ant_str]

    # check that malformed antstr raises and error
    pytest.raises(ValueError, aipy_extracts.parse_ants, "(0_1)_2", nants)
    return


def test_uv_wrhd(tmp_path):
    """Test _wrdh method on UV object"""
    test_file = str(tmp_path / "miriad_test.uv")
    uv = aipy_extracts.UV(test_file, status="new", corrmode="r")

    # test writing freqs
    freqs = [3, 1, 0.1, 0.2, 2, 0.2, 0.3, 3, 0.3, 0.4]
    uv._wrhd("freqs", freqs)

    # test writing other values
    uv._wrhd("nchan0", 1024)

    # test that we wrote something
    del uv
    assert os.path.isdir(test_file)

    # clean up
    shutil.rmtree(test_file)
    return


def test_uv_wrhd_special(tmp_path):
    """Test _wrhd_special method on UV object"""
    test_file = str(tmp_path / "miriad_test.uv")
    uv = aipy_extracts.UV(test_file, status="new", corrmode="r")
    freqs = [3, 1, 0.1, 0.2, 2, 0.2, 0.3, 3, 0.3, 0.4]
    uv._wrhd_special("freqs", freqs)

    # check that we wrote something to disk
    assert os.path.isdir(test_file)

    # check that anything besides 'freqs' raises an error
    pytest.raises(ValueError, uv._wrhd_special, "foo", 12)

    # clean up after ourselves
    del uv
    shutil.rmtree(test_file)
    return


def test_uv_rdhd_special(tmp_path):
    """Test _rdhd_special method on UV object"""
    infile = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    test_file = str(tmp_path / "miriad_test.uv")
    if os.path.exists(test_file):
        shutil.rmtree(test_file)
    # make a new file using an old one as a template
    uv1 = aipy_extracts.UV(infile)
    uv2 = aipy_extracts.UV(test_file, status="new", corrmode="r")
    uv2.init_from_uv(uv1)

    # define freqs to write
    freqs = [3, 1, 0.1, 0.2, 2, 0.2, 0.3, 3, 0.3, 0.4]
    uv2._wrhd_special("freqs", freqs)

    # add a single record; otherwise, opening the file fails
    preamble, data = uv1.read()
    uv2.write(preamble, data)
    del uv1
    del uv2

    # open a new file and check that freqs match the written ones
    uv3 = aipy_extracts.UV(test_file)
    freqs2 = uv3._rdhd_special("freqs")
    assert freqs == freqs2

    # rdhr should correctly handle "special" cases as well -- verify that it does so
    freqs2 = uv3._rdhd("freqs")
    assert freqs == freqs2

    # check that anything besides 'freqs' raises an error
    pytest.raises(ValueError, uv3._rdhd_special, "foo")

    # cleean up after ourselves
    shutil.rmtree(test_file)
    return


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("insert_blank", [False, True])
def test_pipe(tmp_path, raw, insert_blank):
    """Test pipe method on UV object"""
    infile = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    test_file = os.path.join(tmp_path, "miriad_test.uv")

    aipy_uv = aipy_extracts.UV(infile)
    aipy_uv2 = aipy_extracts.UV(test_file, status="new")
    aipy_uv2.init_from_uv(aipy_uv)
    if insert_blank:
        # Insert a blank record, which _should_ get skipped on write
        # such that pipe will still produce an identical file
        aipy_uv2.write((np.zeros(3), 0.0, (0, 0)), None)
    aipy_uv2.pipe(aipy_uv, raw=raw)
    aipy_uv2.close()
    aipy_uv2 = aipy_extracts.UV(test_file)
    aipy_uv.rewind()
    with pytest.raises(OSError, match="No data read"):
        while True:
            rec_orig = aipy_uv.read()
            rec_new = aipy_uv2.read()
            assert np.all(rec_new[0][0] == rec_orig[0][0])
            assert rec_new[0][1] == rec_orig[0][1]
            assert rec_new[0][2] == rec_orig[0][2]
            assert np.ma.allequal(rec_orig[1], rec_new[1])
    aipy_uv.close()
    aipy_uv2.close()


@pytest.mark.parametrize("exclude", [[], "telescop", "history", "dummy"])
def test_init_from_uv_exclude(tmp_path, exclude):
    infile = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    test_file = os.path.join(tmp_path, "miriad_test.uv")

    aipy_uv = aipy_extracts.UV(infile)
    ((uvw, time, (idx, jdx)), data) = aipy_uv.read()

    aipy_uv2 = aipy_extracts.UV(test_file, status="new")
    # Manually add the variable to the object to make sure the exclude works
    # even if the value has already been initiated. Check to make sure we aren't
    # adding an empty list up front.
    if exclude:
        aipy_uv2.add_var(exclude, "d")
    aipy_uv2.init_from_uv(aipy_uv, exclude=exclude)
    aipy_uv2.write((uvw, time, (idx, jdx)), data)
    aipy_uv2.close()

    aipy_uv2 = aipy_extracts.UV(test_file)

    # Again check that exclude isn't just an empty list, and then make sure that
    # the excluded variable isn't actually in the data set.
    if exclude:
        assert exclude not in aipy_uv2.vartable

    for item in aipy_uv.variables():
        if item != exclude:
            assert np.all(aipy_uv[item] == aipy_uv2[item])

    aipy_uv.close()
    aipy_uv2.close()


@pytest.mark.parametrize(
    "var_dict", [{}, {"latitud": 0.0}, {"longitu": 0.0}, {"history": "abc"}],
)
def test_init_from_uv_override(tmp_path, var_dict):
    infile = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    test_file = os.path.join(tmp_path, "miriad_test.uv")

    aipy_uv = aipy_extracts.UV(infile)
    ((uvw, time, (idx, jdx)), data) = aipy_uv.read()

    aipy_uv2 = aipy_extracts.UV(test_file, status="new")
    aipy_uv2.init_from_uv(aipy_uv, override=var_dict)
    aipy_uv2.write((uvw, time, (idx, jdx)), data)
    aipy_uv2.close()

    aipy_uv2 = aipy_extracts.UV(test_file)

    for item in aipy_uv.variables():
        if item in var_dict.keys():
            assert aipy_uv2[item] == var_dict[item]
        else:
            assert np.all(aipy_uv[item] == aipy_uv2[item])

    aipy_uv.close()
    aipy_uv2.close()


def test_write_no_flags(tmp_path):
    """
    Verify that if flags are deleted, write will recreate them appropriately
    """
    infile = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    test_file = os.path.join(tmp_path, "miriad_test.uv")

    aipy_uv = aipy_extracts.UV(infile)
    ((uvw, time, (idx, jdx)), data) = aipy_uv.read()

    aipy_uv2 = aipy_extracts.UV(test_file, status="new")
    aipy_uv2.init_from_uv(aipy_uv)

    # If the mask is remove (all set to False), then write write will create
    # a mask based on where the data have been zeroed out (i.e., default MIRIAD
    # behavior).
    data.mask = 0.0
    aipy_uv2.write((uvw, time, (idx, jdx)), data)
    # Clear the mask entirely, which will also set all the flags to zero.
    data = np.ma.array(data.data, fill_value=data.fill_value)
    aipy_uv2.write((uvw, time, (idx, jdx)), data)
    aipy_uv2.close()

    aipy_uv2 = aipy_extracts.UV(test_file)
    (_, data2) = aipy_uv2.read()
    # Check for equality w/ both masks and data
    assert np.all(data.data == data2.data)
    # Next, check that the masks all look correct
    assert np.all(data.mask == data2.mask)
    (_, data2) = aipy_uv2.read()
    assert np.all(data.mask == data2.mask)
    aipy_uv.close()
    aipy_uv2.close()


def test_add_to_header(tmp_path):
    """
    Verify that if flags are deleted, write will create a mask based on where
    the data have been zeroed out (i.e., default MIRIAD behavior).
    """
    infile = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    test_file = os.path.join(tmp_path, "miriad_test.uv")

    # Create a fake entry in the item table so that we can run this test -- the double
    # values means an array of length 2 for aipy_extracts.
    aipy_extracts.itemtable["xyz"] = "dd"

    aipy_uv = aipy_extracts.UV(infile)
    aipy_uv2 = aipy_extracts.UV(test_file, status="new")

    aipy_uv2["xyz"] = (2.0, 2.0)
    aipy_uv2.init_from_uv(aipy_uv)
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    aipy_uv2 = aipy_extracts.UV(test_file)
    assert np.all(aipy_uv2["xyz"] == np.array([2.0, 2.0]))
    aipy_uv.close()
    aipy_uv2.close()


@pytest.mark.parametrize(
    "polstr,antstr,ants_exp,nrec_exp",
    [
        [-1, "all", (0, 1, 2, 3, 4, 5), 399],
        [-1, "cross", (0, 1, 2, 3, 4, 5), 285],
        [-1, "auto", (0, 1, 2, 3, 4, 5), 114],
        [-1, "(0,1)_(2,3)", (0, 1, 2, 3), 76],
        [-1, "(0,-1)_(-2,3)", (0, 1, 2, 3, 4, 5), 342],
        [-1, "(0x,1x)_(2y,3y)", (0, 1, 2, 3,), 76],
        [-1, "4,5,", (0, 1, 2, 3, 4, 5), 209],
        ["xx", -1, (), 0],
        ["xy", -1, (0, 1, 2, 3, 4, 5), 399],
        ["yx", -1, (), 0],
        ["yy", -1, (), 0],
        ["xy", "(0,1)_(2,3)", (0, 1, 2, 3), 76],
        ["xy", "4,5,", (0, 1, 2, 3, 4, 5), 209],
        ["xy", "0,1,(2)_(3)", (0, 1, 2, 3, 4, 5), 228],
    ],
)
def test_uv_selector(polstr, antstr, ants_exp, nrec_exp):
    infile = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")
    aipy_uv = aipy_extracts.UV(infile)

    aipy_extracts.uv_selector(aipy_uv, ants=antstr, pol_str=polstr)
    nrec = 0
    with pytest.raises(OSError, match="No data read"):
        while True:
            (_, _, bl_ants), _ = aipy_uv.read()
            nrec += 1
            # Make sure we only have ants we expect
            assert np.all(np.isin(bl_ants, ants_exp))
            if isinstance(polstr, str):
                assert aipy_uv["pol"] == aipy_extracts.str2pol[polstr]

    assert nrec == nrec_exp
    aipy_uv.close()
