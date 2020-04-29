# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""
import os
import shutil

import pytest
from pyuvdata.data import DATA_PATH


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Make data/test directory to put test output files in."""
    testdir = os.path.join(DATA_PATH, "tutorial_output/")
    if not os.path.exists(testdir):
        print("making test directory")
        os.mkdir(testdir)

    yield

    shutil.rmtree(testdir)
