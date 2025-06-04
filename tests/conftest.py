# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Testing environment setup and teardown for pytest."""

import os

import numpy as np
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
        t1.ut1  # noqa B018
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
    uvdata.read(os.path.join(DATA_PATH, "zen.2458098.45361.HH_downselected.uvh5"))

    uvcal = UVCal()
    uvcal.read_calfits(
        os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni_downselected.calfits")
    )

    uvcal.pol_convention = "avg"
    uvcal.gain_scale = "Jy"

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

    warn_str = [
        "telescope_location, Nants, antenna_names, antenna_numbers, antenna_positions, "
        "mount_type, antenna_diameters are not set or are being overwritten. "
        "telescope_location, Nants, antenna_names, antenna_numbers, antenna_positions, "
        "mount_type, antenna_diameters are set using values from known telescopes for "
        "HERA."
    ]
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
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    with check_warnings(
        UserWarning,
        match=[
            "The uvw_array does not match the expected values",
            "Fixing auto-correlations to be be real-only",
        ],
    ):
        uvd = UVData.from_file(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA"))

    yield uvd


@pytest.fixture(scope="function")
def uvcalibrate_uvdata_oldfiles(uvcalibrate_uvdata_oldfiles_main):
    uvd = uvcalibrate_uvdata_oldfiles_main.copy()

    yield uvd


@pytest.fixture()
def az_za_coords():
    az_array = np.deg2rad(np.linspace(0, 350, 36))
    za_array = np.deg2rad(np.linspace(0, 90, 10))

    return az_array, za_array


@pytest.fixture()
def az_za_deg_grid(az_za_coords):
    az_array, za_array = az_za_coords
    freqs = np.linspace(100, 200, 11) * 1e8

    az_vals, za_vals = np.meshgrid(az_array, za_array)

    return az_vals.flatten(), za_vals.flatten(), freqs


@pytest.fixture()
def xy_grid():
    nfreqs = 20
    freqs = np.linspace(100e6, 130e6, nfreqs)

    xy_half_n = 250
    zmax = np.radians(90)  # Degrees
    arr = np.arange(-xy_half_n, xy_half_n)
    x_arr, y_arr = np.meshgrid(arr, arr)
    x_arr = x_arr.flatten()
    y_arr = y_arr.flatten()
    radius = np.sqrt(x_arr**2 + y_arr**2) / float(xy_half_n)
    za_array = radius * zmax
    az_array = np.arctan2(y_arr, x_arr)

    return az_array, za_array, freqs
