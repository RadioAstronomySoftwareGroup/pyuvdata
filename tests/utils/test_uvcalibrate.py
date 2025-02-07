# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for uvcalibrate function."""

import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from pyuvdata import UVCal, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings
from pyuvdata.utils import uvcalibrate
from pyuvdata.utils.uvcalibrate import _get_pol_conventions


class TestGetPolConventions:
    def tets_nothing_specified(self):
        with check_warnings(
            UserWarning,
            match=[
                "pol_convention is not specified on the UVCal object",
                "Neither uvd_pol_convention not uvc_pol_convention are specified",
            ],
        ):
            uvc, uvd = _get_pol_conventions(
                uvdata=SimpleNamespace(pol_convention=None),
                uvcal=SimpleNamespace(pol_convention=None),
                undo=False,
                uvc_pol_convention=None,
                uvd_pol_convention=None,
            )
        assert uvc is None
        assert uvd is None

    def test_uvc_pol_convention_set(self):
        uvc, uvd = _get_pol_conventions(
            uvdata=SimpleNamespace(pol_convention=None),
            uvcal=SimpleNamespace(pol_convention="avg"),
            undo=False,
            uvc_pol_convention=None,
            uvd_pol_convention=None,
        )
        assert uvc == "avg"
        assert uvd == "avg"

    def test_uvc_uvcal_different(self):
        with pytest.raises(
            ValueError, match="uvc_pol_convention is set, and different"
        ):
            _get_pol_conventions(
                uvdata=SimpleNamespace(pol_convention=None),
                uvcal=SimpleNamespace(pol_convention="sum"),
                undo=False,
                uvc_pol_convention="avg",
                uvd_pol_convention=None,
            )

    def test_uvd_nor_uvdata_set(self):
        with pytest.warns(
            UserWarning, match="pol_convention is not specified on the UVData object"
        ):
            uvc, uvd = _get_pol_conventions(
                uvdata=SimpleNamespace(pol_convention=None),
                uvcal=SimpleNamespace(pol_convention="avg"),
                undo=True,
                uvc_pol_convention=None,
                uvd_pol_convention=None,
            )
        assert uvc == "avg"
        assert uvd == "avg"

    @pytest.mark.parametrize("undo", [True, False])
    def test_only_objects_set(self, undo):
        uvc, uvd = _get_pol_conventions(
            uvdata=SimpleNamespace(pol_convention="sum" if undo else None),
            uvcal=SimpleNamespace(pol_convention="avg"),
            undo=undo,
            uvc_pol_convention=None,
            uvd_pol_convention=None if undo else "sum",
        )
        assert uvc == "avg"
        assert uvd == "sum"

    def test_uvd_uvdata_different(self):
        with pytest.raises(
            ValueError, match="Both uvd_pol_convention and uvdata.pol_convention"
        ):
            _get_pol_conventions(
                uvdata=SimpleNamespace(pol_convention="sum"),
                uvcal=SimpleNamespace(pol_convention="avg"),
                undo=True,
                uvc_pol_convention="avg",
                uvd_pol_convention="avg",
            )

    def test_calibrate_already_calibrated(self):
        with pytest.raises(
            ValueError, match="You are trying to calibrate already-calibrated data"
        ):
            _get_pol_conventions(
                uvdata=SimpleNamespace(pol_convention="avg"),
                uvcal=SimpleNamespace(pol_convention="avg"),
                undo=False,
                uvc_pol_convention="avg",
                uvd_pol_convention="avg",
            )

    @pytest.mark.parametrize("which", ["uvc", "uvd"])
    def test_bad_convention(self, which):
        good = SimpleNamespace(pol_convention="avg")
        bad = SimpleNamespace(pol_convention="what a silly convention")

        with pytest.raises(
            ValueError, match=f"{which}_pol_convention must be 'sum' or 'avg'"
        ):
            _get_pol_conventions(
                uvdata=good if which == "uvc" else bad,
                uvcal=good if which == "uvd" else bad,
                undo=True,
                uvc_pol_convention=None,
                uvd_pol_convention=None,
            )


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
def test_uvcalibrate_apply_gains_oldfiles(uvcalibrate_uvdata_oldfiles):
    # read data
    uvd = uvcalibrate_uvdata_oldfiles

    # give it an x_orientation
    uvd.telescope.set_feeds_from_x_orientation(
        x_orientation="east",
        polarization_array=uvd.polarization_array,
        flex_polarization_array=uvd.flex_spw_polarization_array,
    )

    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits"))
    # downselect to match each other in shape (but not in actual values!)
    uvd.select(frequencies=uvd.freq_array[:10])
    uvc.select(times=uvc.time_array[:3])
    uvc.gain_scale = "Jy"
    uvc.pol_convention = "avg"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All antenna names with data on UVData are missing "
            "on UVCal. To continue with calibration "
            "(and flag all the data), set ant_check=False."
        ),
    ):
        uvcalibrate(uvd, uvc, prop_flags=True, ant_check=True, inplace=False)

    ants_expected = [
        "The uvw_array does not match the expected values",
        "All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
    ]
    missing_times = [2457698.4036761867, 2457698.4038004624]

    time_expected = f"Time {missing_times[0]} exists on UVData but not on UVCal."

    freq_expected = f"Frequency {uvd.freq_array[0]} exists on UVData but not on UVCal."

    with (
        check_warnings(UserWarning, match=ants_expected),
        pytest.raises(ValueError, match=time_expected),
    ):
        uvcalibrate(uvd, uvc, prop_flags=True, ant_check=False, inplace=False)

    uvc.select(times=uvc.time_array[0])

    time_expected = [
        "Times do not match between UVData and UVCal but time_check is False, so "
        "calibration will be applied anyway."
    ]

    with (
        check_warnings(UserWarning, match=ants_expected + time_expected),
        pytest.raises(ValueError, match=freq_expected),
    ):
        uvcalibrate(uvd, uvc, prop_flags=True, ant_check=False, time_check=False)


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
def test_uvcalibrate_delay_oldfiles(uvcalibrate_uvdata_oldfiles):
    uvd = uvcalibrate_uvdata_oldfiles

    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits"))
    # downselect to match
    uvc.select(times=uvc.time_array[3])
    uvc.gain_convention = "multiply"
    uvc.pol_convention = "avg"
    uvc.gain_scale = "Jy"

    freq_array_use = np.squeeze(uvd.freq_array)
    chan_with_use = uvd.channel_width

    ant_expected = [
        "The uvw_array does not match the expected values",
        "All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
        "Times do not match between UVData and UVCal but time_check is False, so "
        "calibration will be applied anyway.",
        r"UVData object does not have `x_orientation` specified but UVCal does",
    ]
    with check_warnings(UserWarning, match=ant_expected):
        uvdcal = uvcalibrate(
            uvd, uvc, prop_flags=False, ant_check=False, time_check=False, inplace=False
        )

    uvc.convert_to_gain(freq_array=freq_array_use, channel_width=chan_with_use)
    with check_warnings(UserWarning, match=ant_expected):
        uvdcal2 = uvcalibrate(
            uvd, uvc, prop_flags=False, ant_check=False, time_check=False, inplace=False
        )

    assert uvdcal == uvdcal2


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.parametrize("flip_gain_conj", [False, True])
@pytest.mark.parametrize("gain_convention", ["divide", "multiply"])
@pytest.mark.parametrize("time_range", [None, "Ntimes", 3])
def test_uvcalibrate(uvcalibrate_data, flip_gain_conj, gain_convention, time_range):
    uvd, uvc = uvcalibrate_data

    if time_range is not None:
        tstarts = uvc.time_array - uvc.integration_time / (86400 * 2)
        tends = uvc.time_array + uvc.integration_time / (86400 * 2)
        if time_range == "Ntimes":
            uvc.time_range = np.stack((tstarts, tends), axis=1)
        else:
            nt_per_range = int(np.ceil(uvc.Ntimes / time_range))
            tstart_inds = np.array(np.arange(time_range) * nt_per_range)
            tstarts_use = tstarts[tstart_inds]
            tend_inds = np.array((np.arange(time_range) + 1) * nt_per_range - 1)
            tend_inds[-1] = -1
            tends_use = tends[tend_inds]
            uvc.select(times=uvc.time_array[0:time_range])
            uvc.time_range = np.stack((tstarts_use, tends_use), axis=1)
        uvc.time_array = None
        uvc.lst_array = None
        uvc.set_lsts_from_time_array()

    uvc.gain_convention = gain_convention

    if gain_convention == "divide":
        # set the gain_scale to None to test handling
        uvc.gain_scale = None
        cal_warn_msg = [
            "gain_scale is not set, so there is no way to know",
            "gain_scale should be set if pol_convention is set",
        ]
        cal_warn_type = UserWarning
        undo_warn_msg = [
            "pol_convention is not specified on the UVData object",
            "gain_scale is not set, so there is no way to know",
            "gain_scale should be set if pol_convention is set",
        ]
        undo_warn_type = [UserWarning, UserWarning, UserWarning]
    else:
        cal_warn_msg = ""
        cal_warn_type = None
        undo_warn_msg = ""
        undo_warn_type = None

    with check_warnings(cal_warn_type, match=cal_warn_msg):
        uvdcal = uvcalibrate(uvd, uvc, inplace=False, flip_gain_conj=flip_gain_conj)
    if gain_convention == "divide":
        assert uvdcal.vis_units == "uncalib"
    else:
        assert uvdcal.vis_units == "Jy"

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    if flip_gain_conj:
        gain_product = (uvc.get_gains(ant1).conj() * uvc.get_gains(ant2)).T
    else:
        gain_product = (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T

    if time_range is not None and time_range != "Ntimes":
        gain_product = gain_product[:, np.newaxis]
        gain_product = np.repeat(gain_product, nt_per_range, axis=1)
        current_shape = gain_product.shape
        new_shape = (current_shape[0] * current_shape[1], current_shape[-1])
        gain_product = gain_product.reshape(new_shape)
        gain_product = gain_product[: uvd.Ntimes]

    if gain_convention == "divide":
        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key), uvd.get_data(key) / gain_product
        )
    else:
        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key), uvd.get_data(key) * gain_product
        )

    # test undo
    with check_warnings(undo_warn_type, match=undo_warn_msg):
        uvdcal = uvcalibrate(
            uvdcal,
            uvc,
            prop_flags=True,
            ant_check=False,
            inplace=False,
            undo=True,
            flip_gain_conj=flip_gain_conj,
        )

    np.testing.assert_array_almost_equal(uvd.get_data(key), uvdcal.get_data(key))
    assert uvdcal.vis_units == "uncalib"


@pytest.mark.filterwarnings("ignore:Combined frequencies are separated by more than")
def test_uvcalibrate_dterm_handling(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # test d-term exception
    with pytest.raises(
        ValueError, match="Cannot apply D-term calibration without -7 or -8"
    ):
        uvcalibrate(uvd, uvc, d_term_cal=True)

    # d-term not implemented error
    uvcDterm = uvc.copy()
    uvcDterm.jones_array = np.array([-7, -8])
    uvcDterm = uvc + uvcDterm
    with pytest.raises(
        NotImplementedError, match="D-term calibration is not yet implemented."
    ):
        uvcalibrate(uvd, uvcDterm, d_term_cal=True)


@pytest.mark.filterwarnings("ignore:Changing number of antennas, but preserving")
def test_uvcalibrate_flag_propagation(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # test flag propagation
    uvc.flag_array[0] = True
    uvc.gain_array[1] = 0.0
    uvdcal = uvcalibrate(uvd, uvc, prop_flags=True, ant_check=False, inplace=False)

    assert np.all(uvdcal.get_flags(1, 13, "xx"))  # assert completely flagged
    assert np.all(uvdcal.get_flags(0, 12, "xx"))  # assert completely flagged
    np.testing.assert_array_almost_equal(
        uvd.get_data(1, 13, "xx"), uvdcal.get_data(1, 13, "xx")
    )
    np.testing.assert_array_almost_equal(
        uvd.get_data(0, 12, "xx"), uvdcal.get_data(0, 12, "xx")
    )

    uvc_sub = uvc.select(antenna_nums=[1, 12], inplace=False)

    uvdata_unique_nums = np.unique(np.append(uvd.ant_1_array, uvd.ant_2_array))
    uvd.telescope.antenna_names = np.array(uvd.telescope.antenna_names)
    missing_ants = uvdata_unique_nums.tolist()
    missing_ants.remove(1)
    missing_ants.remove(12)
    missing_ant_names = [
        uvd.telescope.antenna_names[
            np.where(uvd.telescope.antenna_numbers == antnum)[0][0]
        ]
        for antnum in missing_ants
    ]

    exp_err = (
        f"Antennas {missing_ant_names} have data on UVData but "
        "are missing on UVCal. To continue calibration and "
        "flag the data from missing antennas, set ant_check=False."
    )

    with pytest.raises(ValueError) as errinfo:
        uvdcal = uvcalibrate(
            uvd, uvc_sub, prop_flags=True, ant_check=True, inplace=False
        )

    assert exp_err == str(errinfo.value)

    uvc_sub.gain_scale = "Jy"
    with pytest.warns(UserWarning) as warninfo:
        uvdcal = uvcalibrate(
            uvd, uvc_sub, prop_flags=True, ant_check=False, inplace=False
        )
    warns = {warn.message.args[0] for warn in warninfo}
    ant_expected = {
        f"Antennas {missing_ant_names} have data on UVData but are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed and the data for these antennas will be flagged."
    }

    assert warns == ant_expected
    assert np.all(uvdcal.get_flags(13, 24, "xx"))  # assert completely flagged


@pytest.mark.filterwarnings("ignore:Cannot preserve total_quality_array")
def test_uvcalibrate_flag_propagation_name_mismatch(uvcalibrate_init_data):
    uvd, uvc = uvcalibrate_init_data
    uvc.gain_scale = "Jy"

    # test flag propagation
    uvc.flag_array[0] = True
    uvc.gain_array[1] = 0.0
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All antenna names with data on UVData are missing "
            "on UVCal. To continue with calibration "
            "(and flag all the data), set ant_check=False."
        ),
    ):
        uvdcal = uvcalibrate(uvd, uvc, prop_flags=True, ant_check=True, inplace=False)

    with check_warnings(
        UserWarning,
        match="All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
    ):
        uvdcal = uvcalibrate(uvd, uvc, prop_flags=True, ant_check=False, inplace=False)

    assert np.all(uvdcal.get_flags(1, 13, "xx"))  # assert completely flagged
    assert np.all(uvdcal.get_flags(0, 12, "xx"))  # assert completely flagged
    np.testing.assert_array_almost_equal(
        uvd.get_data(1, 13, "xx"), uvdcal.get_data(1, 13, "xx")
    )
    np.testing.assert_array_almost_equal(
        uvd.get_data(0, 12, "xx"), uvdcal.get_data(0, 12, "xx")
    )


def test_uvcalibrate_extra_cal_antennas(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # remove some antennas from the data
    uvd.select(antenna_nums=[0, 1, 12, 13])

    uvdcal = uvcalibrate(uvd, uvc, inplace=False)

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    np.testing.assert_array_almost_equal(
        uvdcal.get_data(key),
        uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
    )


def test_uvcalibrate_antenna_names_mismatch(uvcalibrate_init_data):
    uvd, uvc = uvcalibrate_init_data
    uvc.gain_scale = "Jy"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All antenna names with data on UVData are missing "
            "on UVCal. To continue with calibration "
            "(and flag all the data), set ant_check=False."
        ),
    ):
        uvcalibrate(uvd, uvc, inplace=False)

    # now test that they're all flagged if ant_check is False
    with check_warnings(
        UserWarning,
        match="All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
    ):
        uvdcal = uvcalibrate(uvd, uvc, ant_check=False, inplace=False)

    assert np.all(uvdcal.flag_array)  # assert completely flagged


@pytest.mark.parametrize("time_range", [True, False])
def test_uvcalibrate_time_mismatch(uvcalibrate_data, time_range):
    uvd, uvc = uvcalibrate_data
    uvc.gain_scale = "Jy"
    if time_range:
        tstarts = uvc.time_array - uvc.integration_time / (86400 * 2)
        tends = uvc.time_array + uvc.integration_time / (86400 * 2)
        original_time_range = np.stack((tstarts, tends), axis=1)
        uvc.time_range = original_time_range
        uvc.time_array = None
        uvc.lst_array = None
        uvc.set_lsts_from_time_array()

    # change times to get warnings
    if time_range:
        uvc.time_range = uvc.time_range + 1
        uvc.set_lsts_from_time_array()
        expected_err = "Time_ranges on UVCal do not cover all UVData times."
        with pytest.raises(ValueError, match=expected_err):
            uvcalibrate(uvd, uvc, inplace=False)
    else:
        uvc.time_array = uvc.time_array + 1
        uvc.set_lsts_from_time_array()
        expected_err = {
            f"Time {this_time} exists on UVData but not on UVCal."
            for this_time in np.unique(uvd.time_array)
        }

        with pytest.raises(ValueError) as errinfo:
            uvcalibrate(uvd, uvc, inplace=False)
        assert str(errinfo.value) in expected_err

    # for time_range, make the time ranges not cover some UVData times
    if time_range:
        uvc.time_range = original_time_range
        uvc.time_range[0, 1] = uvc.time_range[0, 0] + uvc.integration_time[0] / (
            86400 * 4
        )
        uvc.set_lsts_from_time_array()
        with pytest.raises(ValueError, match=expected_err):
            uvcalibrate(uvd, uvc, inplace=False)

        uvc.phase_center_id_array = np.arange(uvc.Ntimes)
        uvc.phase_center_catalog = {0: None}
        uvc.select(phase_center_ids=0)
        with check_warnings(
            UserWarning, match="Time_range on UVCal does not cover all UVData times"
        ):
            _ = uvcalibrate(uvd, uvc, inplace=False, time_check=False)


def test_uvcalibrate_time_wrong_size(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # downselect by one time to get error
    uvc.select(times=uvc.time_array[1:])
    with pytest.raises(
        ValueError,
        match="The uvcal object has more than one time but fewer than the "
        "number of unique times on the uvdata object.",
    ):
        uvcalibrate(uvd, uvc, inplace=False)


@pytest.mark.filterwarnings("ignore:The time_array and time_range attributes")
@pytest.mark.filterwarnings("ignore:The lst_array and lst_range attributes")
@pytest.mark.parametrize("time_range", [True, False])
def test_uvcalibrate_single_time_types(uvcalibrate_data, time_range):
    uvd, uvc = uvcalibrate_data

    # only one time
    uvc.select(times=uvc.time_array[0])
    if time_range:
        # check cal runs fine with a good time range
        uvc.time_range = np.reshape(
            np.array([np.min(uvd.time_array), np.max(uvd.time_array)]), (1, 2)
        )
        uvc.set_lsts_from_time_array()
        with pytest.raises(
            ValueError, match="The time_array and time_range attributes are both set"
        ):
            uvdcal = uvcalibrate(uvd, uvc, inplace=False, time_check=False)
        uvc.time_array = uvc.lst_array = None
        uvdcal = uvcalibrate(uvd, uvc, inplace=False)

        key = (1, 13, "xx")
        ant1 = (1, "Jxx")
        ant2 = (13, "Jxx")

        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key),
            uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
        )

        # then change time_range to get warnings
        uvc.time_range = np.array(uvc.time_range) + 1
        uvc.set_lsts_from_time_array()

    if time_range:
        msg_start = "Time_range on UVCal does not cover all UVData times"
    else:
        msg_start = "Times do not match between UVData and UVCal"
    err_msg = msg_start + ". Set time_check=False to apply calibration anyway."
    warn_msg = [
        msg_start + " but time_check is False, so calibration will be applied anyway."
    ]

    with pytest.raises(ValueError, match=err_msg):
        uvcalibrate(uvd, uvc, inplace=False)

    if not time_range:
        with check_warnings(UserWarning, match=warn_msg):
            uvdcal = uvcalibrate(uvd, uvc, inplace=False, time_check=False)

        key = (1, 13, "xx")
        ant1 = (1, "Jxx")
        ant2 = (13, "Jxx")

        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key),
            uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
        )


@pytest.mark.filterwarnings("ignore:Combined frequencies are separated by more than")
def test_uvcalibrate_extra_cal_times(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc2 = uvc.copy()
    uvc2.time_array = uvc.time_array + 1
    uvc2.set_lsts_from_time_array()
    uvc_use = uvc + uvc2

    uvdcal = uvcalibrate(uvd, uvc_use, inplace=False)

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    np.testing.assert_array_almost_equal(
        uvdcal.get_data(key),
        uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
    )


def test_uvcalibrate_freq_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # change some frequencies to get warnings
    maxf = np.max(uvc.freq_array)
    uvc.freq_array[uvc.Nfreqs // 2 :] = uvc.freq_array[uvc.Nfreqs // 2 :] + maxf
    expected_err = {
        f"Frequency {this_freq} exists on UVData but not on UVCal."
        for this_freq in uvd.freq_array[uvd.Nfreqs // 2 :]
    }
    # structured this way rather than using the match parameter because expected_err
    # is a set.
    with pytest.raises(ValueError) as errinfo:
        uvcalibrate(uvd, uvc, inplace=False)
    assert str(errinfo.value) in expected_err


@pytest.mark.filterwarnings("ignore:Combined frequencies are not evenly spaced.")
@pytest.mark.filterwarnings("ignore:Selected frequencies are not contiguous.")
def test_uvcalibrate_extra_cal_freqs(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc2 = uvc.copy()
    uvc2.freq_array = uvc.freq_array + np.max(uvc.freq_array)
    uvc_use = uvc + uvc2

    uvdcal = uvcalibrate(uvd, uvc_use, inplace=False)

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    np.testing.assert_array_almost_equal(
        uvdcal.get_data(key),
        uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
    )


def test_uvcalibrate_feedpol_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # downselect the feed polarization to get warnings
    uvc.select(
        jones=utils.jstr2num(
            "Jnn", x_orientation=uvc.telescope.get_x_orientation_from_feeds()
        )
    )
    with pytest.raises(
        ValueError, match=("Feed polarization e exists on UVData but not on UVCal.")
    ):
        uvcalibrate(uvd, uvc, inplace=False)


def test_uvcalibrate_x_orientation_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # next check None uvd_x
    uvd.telescope.set_feeds_from_x_orientation(None)
    uvc.telescope.set_feeds_from_x_orientation(
        "east", polarization_array=uvc.jones_array
    )
    with pytest.warns(
        UserWarning,
        match=r"UVData object does not have `x_orientation` specified but UVCal does",
    ):
        uvcalibrate(uvd, uvc, inplace=False)


def test_uvcalibrate_wideband_gain(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc.flex_spw_id_array = None
    uvc._set_wide_band()
    uvc.spw_array = np.array([1, 2, 3])
    uvc.Nspws = 3
    uvc.gain_array = uvc.gain_array[:, 0:3, :, :]
    uvc.flag_array = uvc.flag_array[:, 0:3, :, :]
    uvc.quality_array = uvc.quality_array[:, 0:3, :, :]
    uvc.total_quality_array = uvc.total_quality_array[0:3, :, :]

    uvc.freq_range = np.zeros((uvc.Nspws, 2), dtype=uvc.freq_array.dtype)
    uvc.freq_range[0, :] = uvc.freq_array[[0, 2]]
    uvc.freq_range[1, :] = uvc.freq_array[[2, 4]]
    uvc.freq_range[2, :] = uvc.freq_array[[4, 6]]

    uvc.channel_width = None
    uvc.freq_array = None
    uvc.Nfreqs = 1

    uvc.check()
    with pytest.raises(
        ValueError,
        match="uvcalibrate currently does not support wide-band calibrations",
    ):
        uvcalibrate(uvd, uvc, inplace=False)


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Nfreqs will be required to be 1 for wide_band cals")
@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
def test_uvcalibrate_delay_multispw(uvcalibrate_uvdata_oldfiles):
    uvd = uvcalibrate_uvdata_oldfiles

    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits"))
    # downselect to match
    uvc.select(times=uvc.time_array[3])
    uvc.gain_convention = "multiply"

    uvc.Nspws = 3
    uvc.spw_array = np.array([1, 2, 3])

    # copy the delay array to the second SPW
    uvc.delay_array = np.repeat(uvc.delay_array, uvc.Nspws, axis=1)
    uvc.flag_array = np.repeat(uvc.flag_array, uvc.Nspws, axis=1)
    uvc.quality_array = np.repeat(uvc.quality_array, uvc.Nspws, axis=1)

    uvc.freq_range = np.repeat(uvc.freq_range, uvc.Nspws, axis=0)
    # Make the second & third SPWs be contiguous with a 10 MHz range
    uvc.freq_range[1, 0] = uvc.freq_range[0, 1]
    uvc.freq_range[1, 1] = uvc.freq_range[1, 0] + 10e6
    uvc.freq_range[2, 0] = uvc.freq_range[1, 1]
    uvc.freq_range[2, 1] = uvc.freq_range[1, 1] + 10e6

    uvc.check()
    with pytest.raises(
        ValueError,
        match="uvcalibrate currently does not support multi spectral window delay "
        "calibrations",
    ):
        uvcalibrate(uvd, uvc, inplace=False)


@pytest.mark.filterwarnings(
    "ignore:pol_convention is not specified on the UVCal object"
)
@pytest.mark.filterwarnings(
    "ignore:pol_convention is not specified on the UVData object"
)
@pytest.mark.filterwarnings(
    "ignore:Neither uvd_pol_convention nor uvc_pol_convention are specified"
)
@pytest.mark.parametrize("convention_on_object", [True, False])
@pytest.mark.parametrize("uvc_pol_convention", ["sum", "avg", None])
@pytest.mark.parametrize("uvd_pol_convention", ["sum", "avg", None])
@pytest.mark.parametrize("polkind", ["linear", "circular"])  # stokes not possible yet
def test_uvcalibrate_pol_conventions(
    uvcalibrate_data,
    convention_on_object,
    uvc_pol_convention,
    uvd_pol_convention,
    polkind,
):
    uvd, uvc = uvcalibrate_data

    # Set defaults
    uvd.pol_convention = None
    uvc.pol_convention = None
    uvc.gain_array[:] = 1.0
    uvd.data_array[:] = 1.0

    # if polkind=='stokes':
    #     uvd.polarization_array = np.array([1,2])
    #     uvc.jones_array = -np.array([5,6])
    if polkind == "circular":
        uvd.polarization_array = -np.array([1, 2])
        uvc.jones_array = -np.array([1, 2])
    else:
        uvd.polarization_array = -np.array([5, 6])
        uvc.jones_array = -np.array([5, 6])

    uvdpol = uvd_pol_convention

    if convention_on_object:
        uvc.pol_convention = uvc_pol_convention
        uvcpol = None
    else:
        uvcpol = uvc_pol_convention

    # go forwards and back
    calib = uvcalibrate(
        uvd,
        uvc,
        uvd_pol_convention=uvdpol,
        uvc_pol_convention=uvcpol,
        undo=False,
        inplace=False,
    )
    roundtrip = uvcalibrate(
        calib,
        uvc,
        uvd_pol_convention=uvdpol,
        uvc_pol_convention=uvcpol,
        undo=True,
        inplace=False,
    )

    assert calib.pol_convention == uvd_pol_convention or uvc_pol_convention
    assert roundtrip.pol_convention is None

    # Check we went around the loop properly.
    np.testing.assert_allclose(
        roundtrip.data_array,
        uvd.data_array,
        rtol=uvd._data_array.tols[0],
        atol=uvd._data_array.tols[1],
    )

    if (
        uvc_pol_convention == uvd_pol_convention
        or uvc_pol_convention is None
        or uvd_pol_convention is None
    ):
        np.testing.assert_almost_equal(calib.data_array, 1.0)
    else:
        if uvc_pol_convention == "sum":
            # Then uvd pol convention is 'avg', so it has I = (XX+YY)/2, i.e. XX ~ I,
            # but the cal intrinsically assumed that I = (XX+YY), i.e. XX ~ I/2.
            # Therefore, the result should be 2.0
            np.testing.assert_allclose(
                calib.data_array,
                2.0,
                rtol=uvd._data_array.tols[0],
                atol=uvd._data_array.tols[1],
            )
        else:
            # the opposite
            np.testing.assert_allclose(
                calib.data_array,
                0.5,
                rtol=uvd._data_array.tols[0],
                atol=uvd._data_array.tols[1],
            )


def test_gain_scale_wrong(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc.gain_scale = "mK"
    uvd.vis_units = "Jy"

    with pytest.raises(
        ValueError, match="Cannot undo calibration if gain_scale is not the same"
    ):
        uvcalibrate(uvd, uvc, undo=True)


def test_uvdata_pol_array_in_stokes(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # Set polarization_array to be in Stokes I, Q, U, V
    uvd.polarization_array = np.array([1, 2, 3, 4])

    with pytest.raises(
        NotImplementedError,
        match=(
            "It is currently not possible to calibrate or de-calibrate data with "
            "stokes polarizations"
        ),
    ):
        uvcalibrate(uvd, uvc)
