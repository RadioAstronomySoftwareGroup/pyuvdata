# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""
import pytest
from astropy.utils import iers
from astropy.time import Time
from astropy.coordinates import EarthLocation


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    """Handle possible IERS issues."""
    # Do a calculation that requires a current IERS table. This will trigger
    # automatic downloading of the IERS table if needed, including trying the
    # mirror site in python 3 (but won't redownload if a current one exists).
    # If there's not a current IERS table and it can't be downloaded, turn off
    # auto downloading for the tests and turn it back on once all tests are
    # completed (done by extending auto_max_age).
    # Also, the check_warnings function will ignore IERS-related warnings.
    try:
        t1 = Time.now()
        t1.ut1
    except (Exception):
        iers.conf.auto_max_age = None

    # Also ensure that we're downloading the site data from astropy
    EarthLocation._get_site_registry(force_download=True)

    yield

    iers.conf.auto_max_age = 30
