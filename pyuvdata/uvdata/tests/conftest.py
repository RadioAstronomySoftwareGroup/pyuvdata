# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""pytest fixtures for UVData tests."""
import os

import pytest

import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.uvdata.mir_parser import MirParser

casa_tutorial_uvfits = os.path.join(
    DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits"
)
paper_miriad_file = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")


@pytest.fixture(scope="session")
def casa_uvfits_main():
    """Read in CASA tutorial uvfits file."""
    uv_in = UVData()
    with uvtest.check_warnings(
        UserWarning,
        [
            "Telescope EVLA is not in known_telescopes",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_in.read(casa_tutorial_uvfits, use_future_array_shapes=True)

    yield uv_in

    # cleanup
    del uv_in


@pytest.fixture(scope="function")
def casa_uvfits(casa_uvfits_main):
    """Make function level CASA tutorial uvfits object."""
    casa_uvfits = casa_uvfits_main.copy()
    yield casa_uvfits

    # clean up when done
    del casa_uvfits

    return


@pytest.fixture(scope="session")
def hera_uvh5_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
    uv_object.read(testfile, use_future_array_shapes=True)

    yield uv_object

    # cleanup
    del uv_object


@pytest.fixture(scope="function")
def hera_uvh5(hera_uvh5_main):
    # read in test file for the resampling in time functions
    uv_object = hera_uvh5_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="session")
def paper_miriad_main():
    """Read in PAPER miriad file."""
    pytest.importorskip("pyuvdata.uvdata.aipy_extracts")
    uv_in = UVData()
    uv_in.read(paper_miriad_file, use_future_array_shapes=True)

    yield uv_in

    # cleanup
    del uv_in


@pytest.fixture(scope="function")
def paper_miriad(paper_miriad_main):
    """Make function level PAPER miriad object."""
    uv_in = paper_miriad_main.copy()

    yield uv_in

    # cleanup
    del uv_in


@pytest.fixture(scope="session")
def sma_mir_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    with uvtest.check_warnings(
        UserWarning,
        match=[
            "> 25 ms errors detected reading in LST values from MIR data. ",
            "The lst_array is not self-consistent with the time_array and telescope ",
        ],
    ):
        uv_object.read(testfile, use_future_array_shapes=True)
    uv_object.set_lsts_from_time_array()

    yield uv_object


@pytest.fixture(scope="function")
def sma_mir(sma_mir_main):
    # read in test file for the resampling in time functions
    uv_object = sma_mir_main.copy()

    yield uv_object


@pytest.fixture(scope="session")
def mir_data_main():
    mir_data = MirParser()

    yield mir_data._load_test_data(load_cross=True, load_auto=True, has_auto=True)


@pytest.fixture(scope="function")
def mir_data(mir_data_main):
    mir_data = mir_data_main.copy()

    yield mir_data


@pytest.fixture(scope="session")
def uv_phase_comp_main():
    file1 = os.path.join(DATA_PATH, "1133866760.uvfits")
    file2 = os.path.join(DATA_PATH, "1133866760_rephase.uvfits")
    # These files came from an external source, don't want to rewrite them, so use
    # checkwarnings to capture the warning about non-real autos
    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Fixing auto-correlations to be be real-only, after some imaginary "
            "values were detected in data_array."
        ]
        * 2,
    ):
        uvd1 = UVData.from_file(file1, use_future_array_shapes=True)
        uvd2 = UVData.from_file(file2, use_future_array_shapes=True)

    yield uvd1, uvd2


@pytest.fixture(scope="function")
def uv_phase_comp(uv_phase_comp_main):
    uvd1, uvd2 = uv_phase_comp_main
    uvd1_copy = uvd1.copy()
    uvd2_copy = uvd2.copy()

    yield uvd1_copy, uvd2_copy
