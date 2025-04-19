# Copyright (c) 2021 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""pytest fixtures for UVCal tests."""

import os

import numpy as np
import pytest

from pyuvdata import UVCal
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

from . import test_fhd_cal


@pytest.fixture(scope="session")
def gain_data_main():
    """Read in gain calfits file."""
    gainfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    gain_object = UVCal.from_file(gainfile)
    gain_object.freq_range = None

    yield gain_object

    del gain_object


@pytest.fixture(scope="function")
def gain_data(gain_data_main):
    """Make function level gain uvcal object."""
    gain_object = gain_data_main.copy()

    yield gain_object


@pytest.fixture(scope="session")
def delay_data_main():
    """Read in delay calfits file."""
    delayfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    delay_object = UVCal.from_file(delayfile)

    # yield the data for testing, then del after tests finish
    yield delay_object


@pytest.fixture(scope="function")
def delay_data(delay_data_main):
    """Make function level delay uvcal object."""
    delay_object = delay_data_main.copy()

    yield delay_object


@pytest.fixture(scope="session")
def fhd_cal_raw_main():
    """Read in raw FHD cal."""
    fhd_cal = UVCal.from_file(
        test_fhd_cal.cal_testfile,
        obs_file=test_fhd_cal.obs_testfile,
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
    fhd_cal = UVCal.from_file(
        test_fhd_cal.cal_testfile,
        obs_file=test_fhd_cal.obs_testfile,
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


@pytest.fixture
def multi_spw_gain(gain_data):
    gain_obj = gain_data.copy()
    gain_obj.channel_width = (
        np.zeros(gain_obj.Nfreqs, dtype=np.float64) + gain_obj.channel_width
    )
    gain_obj.Nspws = 2
    gain_obj.flex_spw_id_array = np.concatenate(
        (
            np.ones(gain_obj.Nfreqs // 2, dtype=int),
            np.full(gain_obj.Nfreqs // 2, 2, dtype=int),
        )
    )
    gain_obj.spw_array = np.array([1, 2])
    spw2_inds = np.nonzero(gain_obj.flex_spw_id_array == 2)[0]
    spw2_chan_width = gain_obj.channel_width[0] * 2
    gain_obj.freq_array[spw2_inds] = gain_obj.freq_array[
        spw2_inds[0]
    ] + spw2_chan_width * np.arange(spw2_inds.size)
    gain_obj.channel_width[spw2_inds] = spw2_chan_width
    gain_obj.check(check_freq_spacing=True)

    yield gain_obj

    del gain_obj


@pytest.fixture
def wideband_gain(gain_data):
    gain_obj = gain_data.copy()
    gain_obj.flex_spw_id_array = None
    gain_obj._set_wide_band()

    gain_obj.spw_array = np.array([1, 2, 3])
    gain_obj.Nspws = 3
    gain_obj.gain_array = gain_obj.gain_array[:, 0:3, :, :]
    gain_obj.flag_array = gain_obj.flag_array[:, 0:3, :, :]
    gain_obj.quality_array = gain_obj.quality_array[:, 0:3, :, :]

    gain_obj.freq_range = np.zeros((gain_obj.Nspws, 2), dtype=gain_obj.freq_array.dtype)
    gain_obj.freq_range[0, :] = gain_obj.freq_array[[0, 2]]
    gain_obj.freq_range[1, :] = gain_obj.freq_array[[2, 4]]
    gain_obj.freq_range[2, :] = gain_obj.freq_array[[4, 6]]

    gain_obj.channel_width = None
    gain_obj.freq_array = None
    gain_obj.flex_spw_id_array = None
    gain_obj.Nfreqs = 1

    with check_warnings(None):
        gain_obj.check(check_freq_spacing=True)

    yield gain_obj


@pytest.fixture
def multi_spw_delay(delay_data):
    delay_obj = delay_data.copy()
    delay_obj.Nspws = 3
    delay_obj.spw_array = np.array([1, 2, 3])

    # copy the delay array to the second SPW
    delay_obj.delay_array = np.repeat(delay_obj.delay_array, delay_obj.Nspws, axis=1)
    delay_obj.flag_array = np.repeat(delay_obj.flag_array, delay_obj.Nspws, axis=1)
    delay_obj.quality_array = np.repeat(
        delay_obj.quality_array, delay_obj.Nspws, axis=1
    )

    delay_obj.freq_range = np.repeat(delay_obj.freq_range, delay_obj.Nspws, axis=0)
    # Make the second & third SPWs be contiguous with a 10 MHz range
    delay_obj.freq_range[1, 0] = delay_obj.freq_range[0, 1]
    delay_obj.freq_range[1, 1] = delay_obj.freq_range[1, 0] + 10e6
    delay_obj.freq_range[2, 0] = delay_obj.freq_range[1, 1]
    delay_obj.freq_range[2, 1] = delay_obj.freq_range[1, 1] + 10e6

    with check_warnings(None):
        delay_obj.check()

    yield delay_obj
