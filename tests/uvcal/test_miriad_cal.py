# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for reading Miriad calibration tables."""

import os

import numpy as np
import pytest

from pyuvdata import UVCal
from pyuvdata.datasets import fetch_data

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


def _copy_with(tmp_path, source, name, *, override=None, gains=None):
    """Write a copy of a Miriad data set, optionally altering it on the way out."""
    outfile = os.path.join(tmp_path, name)
    uv_in = aipy_extracts.UV(source)
    uv_out = aipy_extracts.UV(outfile, status="new")
    uv_out.init_from_uv(uv_in, override={} if override is None else override)
    uv_out.pipe(uv_in)
    if gains is not None:
        uv_out["gains"] = gains
    uv_in.close()
    uv_out.close()

    return outfile


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
