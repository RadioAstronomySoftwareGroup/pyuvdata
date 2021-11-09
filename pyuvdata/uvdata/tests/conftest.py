# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing fixtures for UVData."""
import os

import pytest

from pyuvdata.data import DATA_PATH
from pyuvdata import UVData


@pytest.fixture(scope="session")
def hera_uvh5_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
    uv_object.read(testfile)

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def hera_uvh5(hera_uvh5_main):
    # read in test file for the resampling in time functions
    uv_object = hera_uvh5_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return
