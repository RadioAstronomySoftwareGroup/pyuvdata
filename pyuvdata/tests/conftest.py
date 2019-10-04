# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""
from __future__ import absolute_import, division, print_function

import os
import pytest
import six
import six.moves.urllib as urllib
from astropy.utils import iers
from astropy.time import Time

from pyuvdata.data import DATA_PATH


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Make data/test directory to put test output files in."""
    testdir = os.path.join(DATA_PATH, 'test/')
    if not os.path.exists(testdir):
        print('making test directory')
        os.mkdir(testdir)

    # Try to download the latest IERS table. If the download succeeds, run a
    # computation that requires the values, so they are cached for all future
    # tests. If it fails, turn off auto downloading for the tests and turn it
    # back on once all tests are completed (done by extending auto_max_age).
    # Also, the checkWarnings function will ignore IERS-related warnings.
    try:
        iers.IERS.iers_table = iers.IERS_A.open(iers.IERS_A_URL)
        t1 = Time.now()
        t1.ut1
    except(urllib.error.URLError):
        if six.PY3:
            # python 3 offers a mirror for the download url.
            try:
                iers.IERS.iers_table = iers.IERS_A.open(iers.IERS_A_URL_MIRROR)
                t1 = Time.now()
                t1.ut1
            except(urllib.error.URLError):
                iers.conf.auto_max_age = None
        else:
            iers.conf.auto_max_age = None

    yield

    iers.conf.auto_max_age = 30
