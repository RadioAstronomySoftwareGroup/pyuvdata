# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""
from __future__ import absolute_import, division, print_function

import os
import pytest
import six.moves.urllib as urllib
from astropy.utils import iers

from pyuvdata.data import DATA_PATH


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Make data/test directory to put test output files in."""
    testdir = os.path.join(DATA_PATH, 'test/')
    if not os.path.exists(testdir):
        print('making test directory')
        os.mkdir(testdir)

    # try to download the iers table. If it fails, turn off auto downloading for the tests
    # and turn it back on in teardown_package (done by extending auto_max_age)
    try:
        iers_a = iers.IERS_A.open(iers.IERS_A_URL)
    except(urllib.error.URLError):
        iers.conf.auto_max_age = None

    yield

    iers.conf.auto_max_age = 30
