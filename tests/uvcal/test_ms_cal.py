# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for ms_cal object."""

import os

import numpy as np
import pytest
from astropy.units import Quantity

from pyuvdata import UVCal
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

pytest.importorskip("casacore")

allowed_failures = ["_filename", "_history"]

pytestmark = pytest.mark.filterwarnings(
    "ignore:key CASA_Version in extra_keywords is longer than 8 characters",
    "ignore:telescope_location is not set",
    "ignore:Unknown polarization basis for solutions",
    "ignore:Unknown x_orientation basis for solutions",
)


sma_warnings = [
    "Unknown polarization basis for solutions, jones_array values may be spurious.",
    "Unknown x_orientation basis for solutions, assuming",
    "Setting telescope_location to value in known_telescopes for SMA.",
]


@pytest.fixture(scope="session")
def sma_pcal_main():
    uvobj = UVCal()
    testfile = os.path.join(DATA_PATH, "sma.ms.pha.gcal")
    with check_warnings(UserWarning, match=sma_warnings):
        uvobj.read(testfile)

    uvobj.gain_scale = "Jy"

    yield uvobj


@pytest.fixture(scope="function")
def sma_pcal(sma_pcal_main):
    """Make function level SMA MS phase cal object."""
    uvobj = sma_pcal_main.copy()
    yield uvobj


@pytest.fixture(scope="session")
def sma_dcal_main():
    uvobj = UVCal()
    testfile = os.path.join(DATA_PATH, "sma.ms.dcal")
    with check_warnings(UserWarning, match=sma_warnings):
        uvobj.read(testfile)

    yield uvobj


@pytest.fixture(scope="function")
def sma_dcal(sma_dcal_main):
    """Make function level SMA MS delay cal object."""
    uvobj = sma_dcal_main.copy()
    yield uvobj


@pytest.fixture(scope="session")
def sma_bcal_main():
    uvobj = UVCal()
    testfile = os.path.join(DATA_PATH, "sma.ms.bcal")
    with check_warnings(UserWarning, match=sma_warnings):
        uvobj.read(testfile)

    uvobj.gain_scale = "Jy"

    yield uvobj


@pytest.fixture(scope="function")
def sma_bcal(sma_bcal_main):
    """Make function level SMA MS bandpass cal object."""
    uvobj = sma_bcal_main.copy()
    yield uvobj


@pytest.mark.parametrize(
    "write_func,filename",
    [["write_ms_cal", "ms_cal_loopback.ms"], ["write_calh5", "ms_cal_loopback.calh5"]],
)
def test_ms_cal_wideband_loopback(sma_pcal, tmp_path, write_func, filename):
    uvcal = UVCal()
    testfile = os.path.join(tmp_path, filename)
    getattr(sma_pcal, write_func)(testfile, clobber=True)

    uvcal.read(testfile)
    # Check that the histories line up
    assert sma_pcal.history in uvcal.history
    assert sma_pcal.__eq__(uvcal, allowed_failures=allowed_failures)


@pytest.mark.parametrize(
    "write_func,filename",
    [["write_ms_cal", "ms_cal_delay.ms"], ["write_calh5", "ms_cal_delay.calh5"]],
)
def test_ms_cal_delay_loopback(sma_dcal, tmp_path, write_func, filename):
    uvcal = UVCal()
    testfile = os.path.join(tmp_path, filename)
    getattr(sma_dcal, write_func)(testfile, clobber=True)

    uvcal.read(testfile)
    # Check that the histories line up
    assert sma_dcal.history in uvcal.history
    assert sma_dcal.__eq__(uvcal, allowed_failures=allowed_failures)


@pytest.mark.parametrize(
    "write_func,filename",
    [["write_ms_cal", "ms_cal_bandpass.ms"], ["write_calh5", "ms_cal_bandpass.calh5"]],
)
def test_ms_cal_bandpass_loopback(sma_bcal, tmp_path, write_func, filename):
    uvcal = UVCal()
    testfile = os.path.join(tmp_path, filename)
    getattr(sma_bcal, write_func)(testfile, clobber=True)

    uvcal.read(testfile)
    # Check that the histories line up
    assert sma_bcal.history in uvcal.history
    assert sma_bcal.__eq__(uvcal, allowed_failures=allowed_failures)


def test_ms_cal_wrong_ms_type():
    filepath = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    uvc = UVCal()

    with pytest.raises(
        ValueError, match="This seems to be a Measurement Set containing visibilities"
    ):
        uvc.read_ms_cal(filepath)


def test_ms_cal_time_ranges(gain_data, tmp_path):
    filepath = os.path.join(tmp_path, "mscal_time_range.ms")
    gain_data.flex_spw_id_array = np.full(gain_data.Nfreqs, gain_data.spw_array[0])
    gain_data.set_lsts_from_time_array()

    # Spoof the time range for this test
    gain_data.time_range = np.concatenate(
        (
            gain_data.time_array[:, np.newaxis] - 0.5,
            gain_data.time_array[:, np.newaxis] + 0.5,
        ),
        axis=1,
    )
    gain_data._set_lsts_helper()
    gain_data.time_array = gain_data.lst_array = None
    gain_data.check()

    gain_data.write_ms_cal(filepath)

    uvc = UVCal()
    uvc.read(filepath)

    # Spoof history and extra_keywords
    uvc.history = gain_data.history
    uvc.extra_keywords = gain_data.extra_keywords
    uvc.scan_number_array = gain_data.scan_number_array

    assert uvc == gain_data


def test_ms_cal_write_err(tmp_path):
    uvc = UVCal()
    uvc.cal_type = "unknown"
    uvc.jones_array = [1, 2, 3]
    filepath = os.path.join(tmp_path, "blank.ms")
    os.mkdir(filepath)

    with pytest.raises(FileExistsError, match="File already exists, must set clobber"):
        uvc.write_ms_cal(filepath)

    with pytest.raises(ValueError, match="tables cannot support Njones > 2."):
        uvc.write_ms_cal(filepath, clobber=True)

    uvc.jones_array = [1, 2]

    with pytest.raises(ValueError, match="only supports UVCal objects with gain"):
        uvc.write_ms_cal(filepath, clobber=True)

    uvc.gain_convention = "divide"
    with pytest.raises(ValueError, match="cal_type must either"):
        uvc.write_ms_cal(filepath, clobber=True)


def test_ms_default_setting():
    uvc1 = UVCal()
    uvc2 = UVCal()
    testfile = os.path.join(DATA_PATH, "sma.ms.pha.gcal")
    with check_warnings(UserWarning, match=sma_warnings[2:]):
        uvc1.read_ms_cal(
            testfile,
            default_x_orientation="north",
            default_jones_array=np.array([-1, -2]),
        )

    with check_warnings(UserWarning, match=sma_warnings):
        uvc2.read(testfile)

    assert uvc1.telescope.get_x_orientation_from_feeds() == "north"
    assert uvc2.telescope.get_x_orientation_from_feeds() == "east"
    assert np.array_equal(uvc1.jones_array, [-1, -2])
    assert np.array_equal(uvc2.jones_array, [-5, -6])


def test_ms_muck_ants(sma_pcal, tmp_path):
    from casacore import tables

    uvc = UVCal()
    testfile = os.path.join(tmp_path, "muck.ms")

    sma_pcal.write_ms_cal(testfile)

    with tables.table(
        os.path.join(testfile, "ANTENNA"), readonly=False, ack=False
    ) as tb_ant:
        for idx in range(tb_ant.nrows()):
            tb_ant.putcell("NAME", idx, "")

    with tables.table(
        os.path.join(testfile, "OBSERVATION"), readonly=False, ack=False
    ) as tb_obs:
        tb_obs.removecols("TELESCOPE_LOCATION")
        tb_obs.putcell("TELESCOPE_NAME", 0, "FOO")

    uvc.read(testfile)

    assert uvc.telescope.name == "FOO"
    assert uvc.telescope.antenna_names == sma_pcal.telescope.antenna_names

    new_pos = Quantity(list(uvc.telescope.location.geocentric)).to("m").value
    old_pos = Quantity(list(sma_pcal.telescope.location.geocentric)).to("m").value

    # Verify that the telescope positions are coarsely aligned (doesn't need to be
    # perfect here, just "good enough").
    np.testing.assert_allclose(old_pos, new_pos, rtol=3e-5)

    # The important thing is to verify that the antenna positions _actually_ line up
    # modulo the relative difference in the array center position.
    np.testing.assert_allclose(
        uvc.telescope.antenna_positions + (new_pos - old_pos),
        sma_pcal.telescope.antenna_positions,
        rtol=1e-8,
        atol=1e-3,
    )


def test_ms_total_quality(sma_pcal, tmp_path):
    uvc = UVCal()
    testfile = os.path.join(tmp_path, "total_qual.ms")

    sma_pcal.total_quality_array = np.full(
        (1, sma_pcal.Nspws, sma_pcal.Ntimes, sma_pcal.Njones), 2.0
    )
    sma_pcal.write_ms_cal(testfile)

    uvc.read(testfile)

    assert not np.allclose(sma_pcal.quality_array, uvc.quality_array)
    np.testing.assert_allclose(
        sma_pcal.quality_array * sma_pcal.total_quality_array,
        uvc.quality_array,
        rtol=uvc._quality_array.tols[0],
        atol=uvc._quality_array.tols[1],
    )


def test_ms_to_calfits(sma_bcal, tmp_path):
    uvc = UVCal()

    # Select one spw to allow for
    sma_bcal.select(spws=0)

    # Set mount information for testing
    sma_bcal.telescope.mount_type = ["alt-az+nasmyth-l"] * 8
    testfile = os.path.join(tmp_path, "ms_calfits_roundtrip.calfits")

    with check_warnings(
        UserWarning,
        match=[
            "The calfits format does not support",
            "key CASA_Version in extra_keyword",
        ],
    ):
        sma_bcal.write_calfits(testfile)

    uvc.read_calfits(testfile)

    assert uvc.__eq__(
        sma_bcal,
        allowed_failures=(
            "filename",
            "extra_keywords",
            "phase_center_catalog",
            "phase_center_id_array",
            "Nphase",
            "scan_number_array",
        ),
    )


def test_ms_feed_order(sma_pcal, tmp_path):
    uvc = UVCal()
    testfile = os.path.join(tmp_path, "reorder_feeds.ms")
    sma_pcal.telescope.reorder_feeds(order=["y", "x", "l", "r"])
    sma_pcal.write_ms_cal(testfile, clobber=True)
    uvc.read(testfile)

    # Just set this up front
    uvc.history = sma_pcal.history

    assert uvc == sma_pcal


def test_ms_tcal_read_write(tmp_path):
    datafile = os.path.join(DATA_PATH, "sma.ms.tcal")
    testfile = os.path.join(tmp_path, "tcal_read_write.ms")

    with check_warnings(UserWarning, match=sma_warnings):
        uvc = UVCal.from_file(datafile)

    uvc.write_ms_cal(testfile)
    uvc2 = UVCal.from_file(testfile)
    uvc2.history = uvc.history

    assert uvc == uvc2


def test_ms_polcal_read_write(tmp_path):
    datafile = os.path.join(DATA_PATH, "sma.ms.dterms.pcal")
    testfile = os.path.join(tmp_path, "polcal_read_write.ms")

    with check_warnings(UserWarning, match=sma_warnings[1:]):
        uvc = UVCal.from_file(datafile, default_jones_array=np.array([-3, -4]))

    uvc.write_ms_cal(testfile)
    uvc2 = UVCal.from_file(testfile)
    uvc2.history = uvc.history

    assert uvc == uvc2


def test_ms_polcal_jones_warning():
    datafile = os.path.join(DATA_PATH, "sma.ms.dterms.pcal")
    with check_warnings(
        UserWarning, match=["Cross-handed Jones terms expected"] + sma_warnings[1:]
    ):
        UVCal.from_file(datafile, default_jones_array=np.array([-1, -2]))


def test_ms_nonpolcal_jones_warning():
    datafile = os.path.join(DATA_PATH, "sma.ms.tcal")
    with check_warnings(
        UserWarning, match=["Same-handed Jones terms expected"] + sma_warnings[1:]
    ):
        UVCal.from_file(datafile, default_jones_array=np.array([-3, -4]))


@pytest.mark.parametrize("jones", ["xx", "yy", ["xx", "yy"]])
def test_ms_select_loopback(sma_pcal, jones, tmp_path):
    sma_pcal.select(jones=jones)
    testfile = os.path.join(tmp_path, "pol_select_ms_loopback.ms")
    sma_pcal.write_ms_cal(testfile, clobber=True)

    uvc = UVCal.from_file(testfile)
    uvc.history = sma_pcal.history
    assert uvc == sma_pcal


def test_ms_mixed_jones_error(sma_pcal, tmp_path):
    testfile = os.path.join(tmp_path, "mixed_jones_error.ms")
    sma_pcal.jones_array = [-1, -3]
    with pytest.raises(
        ValueError,
        match="CASA MSCal tables cannot store cross-hand and same-hand Jones",
    ):
        sma_pcal.write_ms_cal(testfile)


def test_ms_dterm_warning(sma_pcal, tmp_path):
    testfile = os.path.join(tmp_path, "dterm_warning.ms")
    sma_pcal.jones_array = [-7, -8]
    sma_pcal.gain_array *= 2
    with check_warnings(
        UserWarning,
        match="CASA MSCal tables store cross-handed Jones terms as leakages ",
    ):
        sma_pcal.write_ms_cal(testfile, clobber=True)
