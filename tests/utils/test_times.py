# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for time related utility functions."""

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import EarthLocation

from pyuvdata import utils

from .test_coordinates import selenoids


def test_astrometry_lst(astrometry_args):
    """
    Check for consistency between astrometry libraries when calculating LAST

    This test evaluates consistency in calculating local apparent sidereal time when
    using the different astrometry libraries available in pyuvdata, namely: astropy,
    pyERFA, and python-novas. Between these three, we expect agreement within 6 µs in
    most instances, although for pyuvdata we tolerate differences of up to ~60 µs
    (which translates to 1 mas in sky position error) since we don't expect to need
    astrometry better than this.
    """
    pytest.importorskip("novas")
    pytest.importorskip("novas_de405")
    astrometry_list = ["erfa", "astropy", "novas"]
    lst_results = [None, None, None, None]
    # These values were indepedently calculated using erfa v1.7.2, which at the
    # time of coding agreed to < 50 µs with astropy v4.2.1 and novas 3.1.1.5. We
    # use those values here as a sort of history check to make sure that something
    # hasn't changed in the underlying astrometry libraries without being caught
    lst_results[3] = np.array(
        [0.8506741803481069, 2.442973468758589, 4.1728965710160555, 1.0130589895999587]
    )

    for idx, name in enumerate(astrometry_list):
        # Note that the units aren't right here (missing a rad-> deg conversion), but
        # the above values were calculated using the arguments below.
        lst_results[idx] = utils.get_lst_for_time(
            jd_array=astrometry_args["time_array"],
            latitude=astrometry_args["telescope_loc"][0],
            longitude=astrometry_args["telescope_loc"][1],
            altitude=astrometry_args["telescope_loc"][2],
            astrometry_library=name,
        )

    for idx in range(len(lst_results) - 1):
        for jdx in range(idx + 1, len(lst_results)):
            alpha_time = lst_results[idx] * units.rad
            beta_time = lst_results[jdx] * units.rad
            assert np.all(np.abs(alpha_time - beta_time).to_value("mas") < 1.0)


@pytest.mark.parametrize("astrometry_lib", ["astropy", "novas", "erfa"])
def test_lst_for_time_smooth(astrometry_lib):
    """
    Test that LSTs are smooth and do not have large discontinuities.

    Inspired by a bug found by the HERA validation team in our original implemenatation
    using the erfa library.
    """
    if astrometry_lib == "novas":
        pytest.importorskip("novas")
        pytest.importorskip("novas_de405")

    hera_loc = EarthLocation.from_geodetic(
        lat=-30.72152612068957, lon=21.428303826863015, height=1051.6900000218302
    )

    start_time = 2458101.5435486115
    n_times = 28728
    integration_time = 1.0

    daysperhour = 1 / 24.0
    hourspersec = 1 / 60.0**2
    dayspersec = daysperhour * hourspersec
    inttime_days = integration_time * dayspersec
    duration = inttime_days * n_times
    end_time = start_time + duration - inttime_days
    times = np.linspace(start_time, end_time + inttime_days, n_times, endpoint=False)

    uv_lsts = utils.get_lst_for_time(
        times,
        latitude=hera_loc.lat.deg,
        longitude=hera_loc.lon.deg,
        altitude=hera_loc.height.value,
        astrometry_library=astrometry_lib,
        frame="itrs",
    )

    dtimes = times - int(times[0])
    poly_fit = np.poly1d(np.polyfit(dtimes, uv_lsts, 2))
    diff_poly = uv_lsts - poly_fit(dtimes)
    assert np.max(np.abs(diff_poly)) < 1e-10


@pytest.mark.parametrize("astrolib", ["novas", "astropy", "erfa"])
def test_lst_for_time_float_vs_array(astrometry_args, astrolib):
    """
    Test for equality when passing a single float vs an ndarray (of length 1) when
    calling get_lst_for_time.
    """
    if astrolib == "novas":
        pytest.importorskip("novas")
        pytest.importorskip("novas_de405")

    r2d = 180.0 / np.pi

    lst_array = utils.get_lst_for_time(
        jd_array=np.array(astrometry_args["time_array"][0]),
        latitude=astrometry_args["telescope_loc"][0] * r2d,
        longitude=astrometry_args["telescope_loc"][1] * r2d,
        altitude=astrometry_args["telescope_loc"][2],
        astrometry_library=astrolib,
    )

    check_lst = utils.get_lst_for_time(
        jd_array=astrometry_args["time_array"][0],
        telescope_loc=np.multiply(astrometry_args["telescope_loc"], [r2d, r2d, 1]),
        astrometry_library=astrolib,
    )

    assert np.all(lst_array == check_lst)


def test_get_lst_for_time_errors(astrometry_args):
    with pytest.raises(
        ValueError,
        match="Requested coordinate transformation library is not supported, please "
        "select either 'erfa' or 'astropy' for astrometry_library.",
    ):
        utils.get_lst_for_time(
            jd_array=np.array(astrometry_args["time_array"][0]),
            latitude=astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
            longitude=astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
            altitude=astrometry_args["telescope_loc"][2],
            astrometry_library="foo",
        )

    with pytest.raises(
        ValueError,
        match="Cannot set both telescope_loc and latitude/longitude/altitude",
    ):
        utils.get_lst_for_time(
            np.array(astrometry_args["time_array"][0]),
            latitude=astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
            telescope_loc=astrometry_args["telescope_loc"][2],
        )

    with pytest.raises(
        ValueError,
        match="Must supply all of latitude, longitude and altitude if "
        "telescope_loc is not supplied",
    ):
        utils.get_lst_for_time(
            np.array(astrometry_args["time_array"][0]),
            latitude=astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
        )


@pytest.mark.filterwarnings("ignore:The get_frame_attr_names")
@pytest.mark.parametrize("selenoid", selenoids)
def test_lst_for_time_moon(astrometry_args, selenoid):
    """Test the get_lst_for_time function with MCMF frame"""
    pytest.importorskip("lunarsky")
    from lunarsky import MoonLocation, SkyCoord as LSkyCoord, Time as LTime

    lat, lon, alt = (0.6875, 24.433, 0)  # Degrees

    # check error if try to use the wrong astrometry library
    with pytest.raises(
        NotImplementedError,
        match="The MCMF frame is only supported with the 'astropy' astrometry library",
    ):
        lst_array = utils.get_lst_for_time(
            jd_array=astrometry_args["time_array"],
            latitude=lat,
            longitude=lon,
            altitude=alt,
            frame="mcmf",
            ellipsoid=selenoid,
            astrometry_library="novas",
        )

    lst_array = utils.get_lst_for_time(
        jd_array=astrometry_args["time_array"],
        latitude=lat,
        longitude=lon,
        altitude=alt,
        frame="mcmf",
        ellipsoid=selenoid,
    )

    # Verify that lsts are close to local zenith RA
    loc = MoonLocation.from_selenodetic(lon, lat, alt, ellipsoid=selenoid)
    for ii, tt in enumerate(
        LTime(astrometry_args["time_array"], format="jd", scale="utc", location=loc)
    ):
        src = LSkyCoord(alt="90d", az="0d", frame="lunartopo", obstime=tt, location=loc)
        # TODO: would be nice to get this down to utils.RADIAN_TOL
        # seems like maybe the ellipsoid isn't being used properly?
        assert np.isclose(lst_array[ii], src.transform_to("icrs").ra.rad, atol=1e-5)

    # test default ellipsoid
    if selenoid == "SPHERE":
        lst_array_default = utils.get_lst_for_time(
            jd_array=astrometry_args["time_array"],
            latitude=lat,
            longitude=lon,
            altitude=alt,
            frame="mcmf",
        )
        np.testing.assert_allclose(
            lst_array, lst_array_default, rtol=0, atol=utils.RADIAN_TOL
        )


def test_get_lst_for_time_no_novas_errors(astrometry_args):
    try:
        import novas_de405  # noqa
        from novas import compat as novas  # noqa
        from novas.compat import eph_manager  # noqa
    except ImportError:
        with pytest.raises(
            ImportError,
            match="novas and/or novas_de405 are not installed but is required for "
            "NOVAS functionality",
        ):
            utils.get_lst_for_time(
                jd_array=astrometry_args["time_array"],
                latitude=astrometry_args["telescope_loc"][0],
                longitude=astrometry_args["telescope_loc"][1],
                altitude=astrometry_args["telescope_loc"][2],
                astrometry_library="novas",
            )
