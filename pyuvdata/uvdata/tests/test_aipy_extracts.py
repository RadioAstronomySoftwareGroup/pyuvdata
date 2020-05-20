# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for aipy_extracts

"""
import os
import shutil
import pytest

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
        "0_1": [(aipy_extracts.ij2bl(0, 1), 1)],
        "0_1,1_2": [(aipy_extracts.ij2bl(0, 1), 1), (aipy_extracts.ij2bl(1, 2), 1)],
        "0x_1x": [(aipy_extracts.ij2bl(0, 1), 1, "xx")],
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

    # check that anything besides 'freqs' raises an error
    pytest.raises(ValueError, uv3._rdhd_special, "foo")

    # cleean up after ourselves
    shutil.rmtree(test_file)
    return
