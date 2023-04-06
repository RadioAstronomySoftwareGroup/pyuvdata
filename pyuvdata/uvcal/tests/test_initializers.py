import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from pyuvdata.uvcal.initializers import new_uvcal


@pytest.fixture(scope="function")
def uvc_kw():
    return {
        "freq_array": np.linspace(100e6, 200e6, 10),
        "time_array": np.linspace(2459850, 2459851, 12),
        "antenna_positions": {
            0: [0.0, 0.0, 0.0],
            1: [0.0, 0.0, 1.0],
            2: [0.0, 0.0, 2.0],
        },
        "telescope_location": EarthLocation.from_geodetic(0, 0, 0),
        "telescope_name": "mock",
        "cal_style": "redundant",
        "gain_convention": "multiply",
        "x_orientation": "n",
        "jones_array": "linear",
        "cal_type": "gain",
    }


def test_new_uvcal_simplest(uvc_kw):
    uvc = new_uvcal(**uvc_kw)
    assert uvc.Nants_data == 3
    assert uvc.Nants_telescope == 3
    assert uvc.Nfreqs == 10
    assert uvc.Ntimes == 12


def test_new_uvcal_bad_inputs(uvc_kw):
    with pytest.raises(NotImplementedError):
        new_uvcal(wide_band=True, **uvc_kw)

    with pytest.raises(ValueError, match="flex_spw must be True for this constructor"):
        new_uvcal(flex_spw=False, **uvc_kw)

    with pytest.raises(
        ValueError, match="The following ants are not in antenna_numbers"
    ):
        new_uvcal(ant_array=[0, 1, 2, 3], **uvc_kw)

    with pytest.raises(
        ValueError,
        match="If spw_array is not length 1, flex_spw_id_array must be provided",
    ):
        new_uvcal(spw_array=[0, 1], **uvc_kw)

    with pytest.raises(
        ValueError, match="If cal_type is delay, delay_array must be provided"
    ):
        new_uvcal(
            cal_type="delay", **{k: v for k, v in uvc_kw.items() if k != "cal_type"}
        )

    with pytest.raises(
        ValueError, match="If cal_style is sky, ref_antenna_name must be provided"
    ):
        new_uvcal(
            cal_style="sky", **{k: v for k, v in uvc_kw.items() if k != "cal_style"}
        )

    with pytest.raises(
        ValueError, match="If cal_style is sky, sky_catalog must be provided"
    ):
        new_uvcal(
            cal_style="sky",
            ref_antenna_name="mock",
            **{k: v for k, v in uvc_kw.items() if k != "cal_style"}
        )

    with pytest.raises(
        ValueError, match="If cal_style is sky, sky_field must be provided"
    ):
        new_uvcal(
            cal_style="sky",
            ref_antenna_name="mock",
            sky_catalog="mock",
            **{k: v for k, v in uvc_kw.items() if k != "cal_style"}
        )

    with pytest.raises(ValueError, match="Unrecognized keyword argument"):
        new_uvcal(bad_kwarg=True, **uvc_kw)


def test_new_uvcal_jones_array(uvc_kw):
    uvc = {k: v for k, v in uvc_kw.items() if k != "jones_array"}

    lin = new_uvcal(jones_array="linear", **uvc)
    assert lin.Njones == 4

    circ = new_uvcal(jones_array="circular", **uvc)
    assert circ.Njones == 4
    assert np.allclose(circ.jones_array, np.array([-1, -2, -3, -4]))

    custom = new_uvcal(jones_array=np.array([-1, -3]), **uvc)
    assert custom.Njones == 2


def test_new_uvcal_set_delay(uvc_kw):
    # TODO: actually implement this
    uvc = {k: v for k, v in uvc_kw.items() if k != "cal_type"}
    dl = new_uvcal(delay_array=np.linspace(0, 1, 10), **uvc)
    assert dl.cal_type == "delay"
