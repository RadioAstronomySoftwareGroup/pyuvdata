# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for reading Miriad calibration tables."""

import os

import numpy as np
import pytest

from pyuvdata import UVCal
from pyuvdata.datasets import fetch_data
from pyuvdata.testing import check_warnings

aipy_extracts = pytest.importorskip(
    "pyuvdata.uvdata.aipy_extracts", exc_type=ImportError
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Altitude is not present in file",
    "ignore:Unknown x_orientation basis for solutions",
)


@pytest.fixture(scope="function")
def atca_path():
    return str(fetch_data("atca_miriad"))


def _copy_with(
    tmp_path, source, name, *, override=None, gains=None, items=None, exclude=None
):
    """Write a copy of a Miriad data set, optionally altering it on the way out."""
    outfile = os.path.join(tmp_path, name)
    uv_in = aipy_extracts.UV(source)
    uv_out = aipy_extracts.UV(outfile, status="new")
    uv_out.init_from_uv(
        uv_in,
        override={} if override is None else override,
        exclude=[] if exclude is None else exclude,
    )
    # Variables have to be registered after init_from_uv, which rebuilds the vartable,
    # and written before the records so that they get flushed along with them.
    for key, value in (items or {}).items():
        if key not in aipy_extracts.itemtable:
            uv_out.add_var(key, "a")
            uv_out[key] = value
    uv_out.pipe(uv_in)
    if gains is not None:
        uv_out["gains"] = gains
    for key, value in (items or {}).items():
        if key in aipy_extracts.itemtable:
            uv_out[key] = value
    uv_in.close()
    uv_out.close()

    return outfile


# Items that have to be dropped to leave a data set carrying a single table.
_DROP_GAINS = ["gains", "nsols", "ngains", "ntau"]
_DROP_BANDPASS = ["bandpass", "nchan0", "nspect0", "nbpsols", "freqs"]
_DROP_LEAKAGE = ["leakage"]


@pytest.fixture(scope="function")
def delay_path(tmp_path, atca_path):
    """A copy of the ATCA set with tau terms added to the gains table."""
    uv = aipy_extracts.UV(atca_path)
    times, gains, _ = uv["gains"]
    uv.close()
    nsols, nants, _ = gains.shape
    # tau is stored in nanoseconds, as -2 * pi * tau
    delays = np.linspace(-0.5, 0.5, nsols * nants, dtype=np.float32)
    delays = (-2 * np.pi * delays).reshape(nsols, nants, 1)

    return _copy_with(
        tmp_path,
        atca_path,
        "atca_delays.uv",
        gains=(times, gains, delays),
        items={"freq0": 2.1},
    )


@pytest.mark.parametrize(
    "soln_type,cal_type,wide_band,jones",
    [
        ("gains", "gain", True, [-5, -6]),
        ("bandpass", "gain", False, [-5, -6]),
        # leakages are carried as the cross-handed Jones terms
        ("leakage", "gain", True, [-7, -8]),
    ],
)
def test_read_atca(atca_path, soln_type, cal_type, wide_band, jones):
    """Check each table this data set carries comes back with the right shape."""
    uvc = UVCal()
    uvc.read_miriad_cal(atca_path, soln_type=soln_type)

    assert uvc.cal_type == cal_type
    assert uvc.wide_band == wide_band
    assert np.array_equal(uvc.jones_array, jones)
    # Miriad multiplies the solutions into the data rather than dividing
    assert uvc.gain_convention == "multiply"
    assert uvc.gain_array.shape == (
        uvc.Nants_data,
        uvc.Nspws if wide_band else uvc.Nfreqs,
        uvc.Ntimes,
        uvc.Njones,
    )
    # a flagged solution is stored as an exact zero
    assert np.array_equal(uvc.flag_array, uvc.gain_array == 0)


def test_values_match_low_level(atca_path):
    """The gains must survive the trip onto the UVCal object untouched."""
    uv = aipy_extracts.UV(atca_path)
    _, gains, _ = uv["gains"]
    uv.close()

    uvc = UVCal()
    uvc.read_miriad_cal(atca_path, soln_type="gains")
    # (Ntimes, Nants, Njones) -> (Nants, Nspws, Ntimes, Njones), replicated over spws
    assert np.array_equal(uvc.gain_array[:, 0], np.transpose(gains, (1, 0, 2)))


def test_freq_range_covers_data(atca_path):
    """Each wide band window must span the channels it is meant to apply to."""
    uvc = UVCal()
    uvc.read_miriad_cal(atca_path, soln_type="gains")

    uv = aipy_extracts.UV(atca_path)
    freq_array, _, flex_spw_id_array, spw_array = uv.get_freq_axis()
    uv.close()

    for idx, spw in enumerate(spw_array):
        spw_freqs = freq_array[flex_spw_id_array == spw]
        assert uvc.freq_range[idx, 0] <= spw_freqs.min()
        assert uvc.freq_range[idx, 1] >= spw_freqs.max()


def test_bandpass_uses_own_freq_axis(atca_path):
    """The bandpass axis comes from the freqs item, not the visibilities."""
    uv = aipy_extracts.UV(atca_path)
    nchan0 = uv["nchan0"]
    uv.close()

    uvc = UVCal()
    uvc.read_miriad_cal(atca_path, soln_type="bandpass")
    assert uvc.Nfreqs == nchan0
    assert uvc.freq_array.size == nchan0


def test_reference_antenna(atca_path):
    """Miriad pins the reference antenna to zero phase, which is recoverable."""
    uvc = UVCal()
    uvc.read_miriad_cal(atca_path, soln_type="gains")
    ant_num = uvc.telescope.antenna_numbers[
        list(uvc.telescope.antenna_names).index(uvc.ref_antenna_name)
    ]
    ant_idx = np.nonzero(uvc.ant_array == ant_num)[0]
    assert ant_idx.size == 1
    assert np.allclose(np.angle(uvc.gain_array[ant_idx]), 0)


@pytest.mark.filterwarnings("ignore:antenna number 0 has visibilities")
def test_antennas_from_solutions(tmp_path, atca_path):
    """Antennas are screened on the solutions, not on their positions.

    An antenna sitting on the array reference point has a position of exactly
    (0,0,0), which is also how Miriad marks an antenna that is not present, so the
    positions cannot be used to tell the two apart.
    """
    uv = aipy_extracts.UV(atca_path)
    nants = uv["nants"]
    antpos = uv["antpos"].copy()
    antpos.reshape(3, nants)[:, 0] = 0.0
    uv.close()

    testfile = _copy_with(tmp_path, atca_path, "refpad.uv", override={"antpos": antpos})
    uvc = UVCal()
    uvc.read_miriad_cal(testfile, soln_type="gains")
    assert uvc.Nants_data == nants
    assert 0 in uvc.ant_array


def test_unpopulated_antenna_dropped(tmp_path, atca_path):
    """An antenna with no solutions at all is left out."""
    uv = aipy_extracts.UV(atca_path)
    nants = uv["nants"]
    times, gains, _ = uv["gains"]
    gains = gains.copy()
    gains[:, 3, :] = 0
    uv.close()

    testfile = _copy_with(tmp_path, atca_path, "deadant.uv", gains=(times, gains, None))
    uvc = UVCal()
    uvc.read_miriad_cal(testfile, soln_type="gains")
    assert uvc.Nants_data == nants - 1
    assert 3 not in uvc.ant_array


@pytest.mark.parametrize(
    "kwargs,err,msg",
    [
        ({}, ValueError, "Cannot determine which calibration table to read"),
        ({"soln_type": "delays"}, ValueError, "has no delays table"),
        ({"soln_type": "nonsense"}, ValueError, "soln_type must be one of"),
    ],
)
def test_read_errors(atca_path, kwargs, err, msg):
    with pytest.raises(err, match=msg):
        UVCal().read_miriad_cal(atca_path, **kwargs)


def test_read_missing_file():
    with pytest.raises(OSError, match="not found"):
        UVCal().read_miriad_cal("/no/such/miriad/set")


def test_read_list_error(atca_path):
    with pytest.raises(ValueError, match="Use the generic `UVCal.read` method"):
        UVCal().read_miriad_cal([atca_path, atca_path])


def test_read_delays(delay_path):
    """The tau terms come back as a delay object, in seconds."""
    uv = aipy_extracts.UV(delay_path)
    _, _, theta = uv["gains"]
    freq0 = uv["freq0"]
    uv.close()

    uvc = UVCal()
    uvc.read_miriad_cal(delay_path, soln_type="delays")

    assert uvc.cal_type == "delay"
    assert uvc.wide_band
    assert uvc.delay_array.shape == (uvc.Nants_data, uvc.Nspws, uvc.Ntimes, uvc.Njones)
    # Miriad writes -2 * pi * tau with tau in nanoseconds
    want = -1e-9 * np.transpose(theta, (1, 0, 2)) / (2 * np.pi)
    assert np.allclose(uvc.delay_array[:, 0, :, :1], want)
    # one delay per antenna, replicated over the Jones axis
    assert np.allclose(uvc.delay_array[..., 0], uvc.delay_array[..., -1])
    assert uvc.extra_keywords["FREQ0"] == freq0
    assert "freq0" in uvc.history


def test_delays_and_gains_agree(delay_path):
    """The gains and delays tables describe the same antennas and times."""
    gain_obj = UVCal()
    gain_obj.read_miriad_cal(delay_path, soln_type="gains")
    delay_obj = UVCal()
    delay_obj.read_miriad_cal(delay_path, soln_type="delays")

    assert np.array_equal(gain_obj.ant_array, delay_obj.ant_array)
    assert np.array_equal(gain_obj.time_array, delay_obj.time_array)
    assert np.array_equal(gain_obj.jones_array, delay_obj.jones_array)
    assert gain_obj.cal_type == "gain"
    assert delay_obj.cal_type == "delay"


def test_single_feed_circular(tmp_path, atca_path):
    """A single circular feed takes its Jones component from the data pol."""
    uv = aipy_extracts.UV(atca_path)
    times, gains, _ = uv["gains"]
    uv.close()
    # keep one feed, and relabel the data as circular
    testfile = _copy_with(
        tmp_path,
        atca_path,
        "atca_1feed.uv",
        override={"pol": -1},
        gains=(times, gains[:, :, :1], None),
    )
    uvc = UVCal()
    uvc.read_miriad_cal(testfile, soln_type="gains")
    assert np.array_equal(uvc.jones_array, [-1])
    assert uvc.Njones == 1


@pytest.mark.parametrize("leakage", [False, True])
def test_two_feed_circular(tmp_path, atca_path, leakage):
    """Circular data gets circular Jones codes, cross-handed for leakages."""
    testfile = _copy_with(tmp_path, atca_path, "atca_circ.uv", override={"pol": -2})
    uvc = UVCal()
    uvc.read_miriad_cal(testfile, soln_type="leakage" if leakage else "gains")
    assert np.array_equal(uvc.jones_array, [-3, -4] if leakage else [-1, -2])


def test_x_orientation_from_default(atca_path):
    """A supplied default is used in place of the warning."""
    uvc = UVCal()
    # only the altitude warning -- the x_orientation one must not be raised
    with check_warnings(UserWarning, match="Altitude is not present in file"):
        uvc.read_miriad_cal(atca_path, soln_type="gains", default_x_orientation="north")
    assert uvc.telescope.get_x_orientation_from_feeds() == "north"


@pytest.mark.parametrize(
    "override,msg",
    [({"pol": 1}, "unrecognized pol code"), ({"npol": 3}, "expected only 1 or 2")],
)
def test_feed_errors(tmp_path, atca_path, override, msg):
    """Pol codes and feed counts that cannot describe a Jones vector."""
    uv = aipy_extracts.UV(atca_path)
    times, gains, _ = uv["gains"]
    uv.close()
    gains_arg = None
    if "npol" in override:
        # nfeeds is written from the gains table, not from npol
        gains_arg = (times, np.repeat(gains, 2, axis=2)[:, :, :3], None)
        override = {}
    testfile = _copy_with(
        tmp_path, atca_path, "atca_badpol.uv", override=override, gains=gains_arg
    )
    with pytest.raises(ValueError, match=msg):
        UVCal().read_miriad_cal(testfile, soln_type="gains")


def test_soln_type_autodetect(tmp_path, atca_path):
    """With only one table present, soln_type does not have to be given."""
    testfile = _copy_with(
        tmp_path, atca_path, "gains_only.uv", exclude=_DROP_BANDPASS + _DROP_LEAKAGE
    )
    uvc = UVCal()
    uvc.read_miriad_cal(testfile)
    assert uvc.cal_type == "gain"
    assert uvc.wide_band


def test_leakage_time_fallback(tmp_path, atca_path):
    """With no gains or bandpass to date it against, leakage uses the data time."""
    testfile = _copy_with(
        tmp_path, atca_path, "leakage_only.uv", exclude=_DROP_GAINS + _DROP_BANDPASS
    )
    uvc = UVCal()
    uvc.read_miriad_cal(testfile, soln_type="leakage")

    uv = aipy_extracts.UV(testfile)
    uv.read(raw=True)
    first_time = uv["time"]
    uv.close()
    assert uvc.Ntimes == 1
    assert np.allclose(uvc.time_range, first_time)


def test_x_orientation_from_file(tmp_path, atca_path):
    """A recorded xorient is used in preference to the default."""
    testfile = _copy_with(tmp_path, atca_path, "xorient.uv", items={"xorient": "north"})
    uvc = UVCal()
    with check_warnings(UserWarning, match="Altitude is not present in file"):
        uvc.read_miriad_cal(testfile, soln_type="gains")
    assert uvc.telescope.get_x_orientation_from_feeds() == "north"


def test_suspect_interval_warning(tmp_path, atca_path):
    """mfcal hardcodes the interval, so flag it when it dwarfs the cadence."""
    uv = aipy_extracts.UV(atca_path)
    _, gains, _ = uv["gains"]
    uv.close()
    # four solutions a minute apart, against a declared interval of half a day
    times = 2457080.5 + np.arange(4) / 1440.0
    gains = np.repeat(gains, 4, axis=0)

    testfile = _copy_with(
        tmp_path,
        atca_path,
        "long_interval.uv",
        gains=(times, gains, None),
        items={"interval": 0.5},
    )
    with check_warnings(
        UserWarning,
        match=[
            "solution interval",
            "Altitude is not present in file",
            "Unknown x_orientation basis",
        ],
    ):
        UVCal().read_miriad_cal(testfile, soln_type="gains")
