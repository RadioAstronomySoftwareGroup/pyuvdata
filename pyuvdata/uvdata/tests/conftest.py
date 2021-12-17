# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""pytest fixtures for UVData tests."""
import os

import pytest

from pyuvdata.data import DATA_PATH
from pyuvdata import UVData
from pyuvdata.uvdata.mir_parser import MirParser
import pyuvdata.tests as uvtest

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
        uv_in.read(casa_tutorial_uvfits)

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
def mir_data_main():
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = MirParser(
        testfile, load_vis=True, load_raw=True, load_auto=True, has_auto=True,
    )

    yield mir_data


@pytest.fixture(scope="function")
def mir_data(mir_data_main):
    mir_data = mir_data_main.copy()

    yield mir_data


@pytest.fixture(scope="session")
def hera_uvh5_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
    uv_object.read(testfile)

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
    uv_in.read(paper_miriad_file)

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


@pytest.fixture(params=[True, False])
def mir_data_object(request):
    """Make MIR data object for tests. Param to read autocorr data."""
    has_auto = request.param
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = MirParser(
        testfile, load_vis=True, load_raw=True, load_auto=True, has_auto=has_auto
    )

    yield mir_data

    # cleanup
    del mir_data
