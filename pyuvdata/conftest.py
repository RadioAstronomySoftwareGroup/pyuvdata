# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""
import os

from astropy.utils import iers
from astropy.time import Time
from astropy.coordinates import EarthLocation
import numpy as np
import pytest

from pyuvdata import UVData, UVCal
from pyuvdata.data import DATA_PATH


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


@pytest.fixture(scope="session")
def uvcalibrate_init_data_main():
    """Make initial uvcalibrate inputs."""
    uvdata = UVData()
    uvdata.read(
        os.path.join(DATA_PATH, "zen.2458098.45361.HH.uvh5_downselected"),
        file_type="uvh5",
    )
    uvcal = UVCal()
    uvcal.read_calfits(
        os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni.calfits_downselected")
    )

    yield uvdata, uvcal


@pytest.fixture(scope="function")
def uvcalibrate_init_data(uvcalibrate_init_data_main):
    """Make function level initial uvcalibrate inputs."""
    uvdata_in, uvcal_in = uvcalibrate_init_data_main

    uvdata = uvdata_in.copy()
    uvcal = uvcal_in.copy()

    yield uvdata, uvcal


@pytest.fixture(scope="session")
def uvcalibrate_data_main(uvcalibrate_init_data_main):
    """Make uvcalibrate inputs."""
    uvdata_in, uvcal_in = uvcalibrate_init_data_main

    uvdata = uvdata_in.copy()
    uvcal = uvcal_in.copy()

    # fix the antenna names in the uvcal object to match the uvdata object
    uvcal.antenna_names = np.array(
        [name.replace("ant", "HH") for name in uvcal.antenna_names]
    )

    yield uvdata, uvcal


@pytest.fixture(scope="function")
def uvcalibrate_data(uvcalibrate_data_main):
    """Make function level uvcalibrate inputs."""
    uvdata_in, uvcal_in = uvcalibrate_data_main

    uvdata = uvdata_in.copy()
    uvcal = uvcal_in.copy()

    yield uvdata, uvcal
