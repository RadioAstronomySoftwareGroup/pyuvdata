# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""
import os

import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers

from pyuvdata import UVCal, UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings


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
    except Exception:
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
    with check_warnings(
        UserWarning,
        match="telescope_location, antenna_positions, antenna_diameters are "
        "not set or are being overwritten. telescope_location, antenna_positions, "
        "antenna_diameters are set using values from known telescopes for HERA.",
    ):
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

    warn_str = (
        "telescope_location, Nants, antenna_names, antenna_numbers, "
        "antenna_positions, antenna_diameters are not set or are being "
        "overwritten. telescope_location, Nants, antenna_names, "
        "antenna_numbers, antenna_positions, antenna_diameters are set "
        "using values from known telescopes for HERA."
    )
    with check_warnings(UserWarning, warn_str):
        uvcal.set_telescope_params(overwrite=True)

    with check_warnings(UserWarning, warn_str):
        uvdata.set_telescope_params(overwrite=True)

    yield uvdata, uvcal


@pytest.fixture(scope="function")
def uvcalibrate_data(uvcalibrate_data_main):
    """Make function level uvcalibrate inputs."""
    uvdata_in, uvcal_in = uvcalibrate_data_main

    uvdata = uvdata_in.copy()
    uvcal = uvcal_in.copy()

    yield uvdata, uvcal


@pytest.fixture(scope="session")
def uvcalibrate_uvdata_oldfiles_main():
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.uvh5"))

    yield uvd


@pytest.fixture(scope="function")
def uvcalibrate_uvdata_oldfiles(uvcalibrate_uvdata_oldfiles_main):
    uvd = uvcalibrate_uvdata_oldfiles_main.copy()

    yield uvd
