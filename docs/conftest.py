# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package(tmp_path_factory):
    """Make data/test directory to put test output files in."""
    cwd = Path.cwd()
    tmp_path = tmp_path_factory.mktemp("uvdata_tests")
    try:
        os.chdir(tmp_path)
        yield
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp_path)
