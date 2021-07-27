# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""pytest fixtures for UVCal tests."""
import os

import pytest
import numpy as np

from pyuvdata.data import DATA_PATH
from pyuvdata import UVCal
import pyuvdata.tests as uvtest
import pyuvdata.uvcal.tests.test_fhd_cal as test_fhd_cal


@pytest.fixture(scope="session")
def gain_data_main():
    """Read in gain calfits file."""
    gain_object = UVCal()
    gainfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    gain_object.read_calfits(gainfile)

    yield gain_object

    del gain_object


@pytest.fixture(scope="function")
def gain_data(gain_data_main):
    """Make function level gain uvcal object."""
    gain_object = gain_data_main.copy()

    yield gain_object

    del gain_object


@pytest.fixture(scope="session")
def delay_data_main():
    """Read in delay calfits file, add input flag array."""
    delay_object = UVCal()
    delayfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    delay_object.read_calfits(delayfile)

    # yield the data for testing, then del after tests finish
    yield delay_object

    del delay_object


@pytest.fixture(scope="function")
def delay_data(delay_data_main):
    """Make function level delay uvcal object."""
    delay_object = delay_data_main.copy()

    yield delay_object

    del delay_object


@pytest.fixture(scope="session")
def delay_data_inputflag_main(delay_data_main):
    """Add an input flag array to delay object."""
    delay_object = delay_data_main.copy()

    # add an input flag array for testing
    delay_object.input_flag_array = np.zeros(
        delay_object._input_flag_array.expected_shape(delay_object), dtype=bool
    )

    # yield the data for testing, then del after tests finish
    yield delay_object

    del delay_object


@pytest.fixture(scope="function")
def delay_data_inputflag(delay_data_inputflag_main):
    """Make function level delay uvcal object."""
    delay_object = delay_data_inputflag_main.copy()

    yield delay_object

    del delay_object


@pytest.fixture(scope="session")
def delay_data_inputflag_future_main(delay_data_inputflag_main):
    delay_object = delay_data_inputflag_main.copy()

    # convert to future array shapes, drop freq_array, set Nfreqs=1
    with uvtest.check_warnings(
        UserWarning, match="When converting a delay-style cal to future array shapes",
    ):
        delay_object.use_future_array_shapes()

    delay_object.freq_array = None
    delay_object.channel_width = None
    delay_object.Nfreqs = 1

    delay_object.check()

    yield delay_object

    del delay_object


@pytest.fixture(scope="function")
def delay_data_inputflag_future(delay_data_inputflag_future_main):
    """Make function level future shape delay uvcal object."""
    delay_object = delay_data_inputflag_future_main.copy()

    yield delay_object

    del delay_object


@pytest.fixture(scope="session")
def fhd_cal_raw_main():
    """Read in raw FHD cal."""
    fhd_cal = UVCal()
    with uvtest.check_warnings(
        UserWarning, match="Telescope location derived from obs lat/lon/alt values"
    ):
        fhd_cal.read_fhd_cal(
            test_fhd_cal.cal_testfile,
            test_fhd_cal.obs_testfile,
            layout_file=test_fhd_cal.layout_testfile,
            settings_file=test_fhd_cal.settings_testfile,
            raw=True,
        )

    yield fhd_cal

    del fhd_cal


@pytest.fixture(scope="function")
def fhd_cal_raw(fhd_cal_raw_main):
    """Make function raw FHD cal object."""
    fhd_cal = fhd_cal_raw_main.copy()

    yield fhd_cal

    del fhd_cal


@pytest.fixture(scope="session")
def fhd_cal_fit_main():
    """Read in fit FHD cal."""
    fhd_cal = UVCal()
    with uvtest.check_warnings(
        UserWarning, match="Telescope location derived from obs lat/lon/alt values"
    ):
        fhd_cal.read_fhd_cal(
            test_fhd_cal.cal_testfile,
            test_fhd_cal.obs_testfile,
            layout_file=test_fhd_cal.layout_testfile,
            settings_file=test_fhd_cal.settings_testfile,
            raw=False,
        )
    yield fhd_cal

    del fhd_cal


@pytest.fixture(scope="function")
def fhd_cal_fit(fhd_cal_fit_main):
    """Make function fit FHD cal object."""
    fhd_cal = fhd_cal_fit_main.copy()

    yield fhd_cal

    del fhd_cal
