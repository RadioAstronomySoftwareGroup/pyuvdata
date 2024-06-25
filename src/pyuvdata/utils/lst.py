# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working LSTs."""
import warnings

import erfa
import numpy as np
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time
from astropy.utils import iers

try:
    from lunarsky import MoonLocation
    from lunarsky import Time as LTime

    hasmoon = True
except ImportError:
    hasmoon = False


def get_lst_for_time(
    jd_array=None,
    *,
    telescope_loc=None,
    latitude=None,
    longitude=None,
    altitude=None,
    astrometry_library=None,
    frame="itrs",
    ellipsoid=None,
):
    """
    Get the local apparent sidereal time for a set of jd times at an earth location.

    This function calculates the local apparent sidereal time (LAST), given a UTC time
    and a position on the Earth, using either the astropy or NOVAS libraries. It
    is important to note that there is an apporoximate 20 microsecond difference
    between the two methods, presumably due to small differences in the apparent
    reference frame. These differences will cancel out when calculating coordinates
    in the TOPO frame, so long as apparent coordinates are calculated using the
    same library (i.e., astropy or NOVAS). Failing to do so can introduce errors
    up to ~1 mas in the horizontal coordinate system (i.e., AltAz).

    Parameters
    ----------
    jd_array : ndarray of float
        JD times to get lsts for.
    telescope_loc : tuple or EarthLocation or MoonLocation
        Alternative way of specifying telescope lat/lon/alt, either as a 3-element tuple
        or as an astropy EarthLocation (or lunarsky MoonLocation). Cannot supply both
        `telescope_loc` and `latitute`, `longitude`, or `altitude`.
    latitude : float
        Latitude of location to get lst for in degrees. Cannot specify both `latitute`
        and `telescope_loc`.
    longitude : float
        Longitude of location to get lst for in degrees. Cannot specify both `longitude`
        and `telescope_loc`.
    altitude : float
        Altitude of location to get lst for in meters. Cannot specify both `altitude`
        and `telescope_loc`.
    astrometry_library : str
        Library used for running the LST calculations. Allowed options are 'erfa'
        (which uses the pyERFA), 'novas' (which uses the python-novas library),
        and 'astropy' (which uses the astropy utilities). Default is erfa unless
        the telescope_location is a MoonLocation object, in which case the default is
        astropy.
    frame : str
        Reference frame for latitude/longitude/altitude. Options are itrs (default)
        or mcmf. Not used if telescope_loc is an EarthLocation or MoonLocation object.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.  Not used if telescope_loc is
        an EarthLocation or MoonLocation object.

    Returns
    -------
    ndarray of float
        LASTs in radians corresponding to the jd_array.

    """
    site_loc = None
    if telescope_loc is not None:
        if not all(item is None for item in [latitude, longitude, altitude]):
            raise ValueError(
                "Cannot set both telescope_loc and latitude/longitude/altitude"
            )
        if isinstance(telescope_loc, EarthLocation) or (
            hasmoon and isinstance(telescope_loc, MoonLocation)
        ):
            site_loc = telescope_loc
            if isinstance(telescope_loc, EarthLocation):
                frame = "ITRS"
            else:
                frame = "MCMF"
        else:
            latitude, longitude, altitude = telescope_loc

    if site_loc is None:
        if frame.upper() == "MCMF":
            if not hasmoon:
                raise ValueError(
                    "Need to install `lunarsky` package to work with MCMF frame."
                )
            if ellipsoid is None:
                ellipsoid = "SPHERE"

            site_loc = MoonLocation.from_selenodetic(
                Angle(longitude, unit="deg"),
                Angle(latitude, unit="deg"),
                altitude,
                ellipsoid=ellipsoid,
            )
        else:
            site_loc = EarthLocation.from_geodetic(
                Angle(longitude, unit="deg"),
                Angle(latitude, unit="deg"),
                height=altitude,
            )
    if astrometry_library is None:
        if frame == "itrs":
            astrometry_library = "erfa"
        else:
            astrometry_library = "astropy"

    if astrometry_library not in ["erfa", "astropy", "novas"]:
        raise ValueError(
            "Requested coordinate transformation library is not supported, please "
            "select either 'erfa' or 'astropy' for astrometry_library."
        )

    if isinstance(jd_array, np.ndarray):
        lst_array = np.zeros_like(jd_array)
        if lst_array.ndim == 0:
            lst_array = lst_array.reshape(1)
    else:
        lst_array = np.zeros(1)

    jd, reverse_inds = np.unique(jd_array, return_inverse=True)

    if isinstance(site_loc, EarthLocation):
        TimeClass = Time
    else:
        if not astrometry_library == "astropy":
            raise NotImplementedError(
                "The MCMF frame is only supported with the 'astropy' astrometry library"
            )
        TimeClass = LTime

    times = TimeClass(jd, format="jd", scale="utc", location=site_loc)

    if iers.conf.auto_max_age is None:  # pragma: no cover
        delta, status = times.get_delta_ut1_utc(return_status=True)
        if np.any(
            np.isin(status, (iers.TIME_BEFORE_IERS_RANGE, iers.TIME_BEYOND_IERS_RANGE))
        ):
            warnings.warn(
                "time is out of IERS range, setting delta ut1 utc to extrapolated value"
            )
            times.delta_ut1_utc = delta
    if astrometry_library == "erfa":
        # This appears to be what astropy is using under the hood,
        # so it _should_ be totally consistent.
        gast_array = erfa.gst06a(
            times.ut1.jd1, times.ut1.jd2, times.tt.jd1, times.tt.jd2
        )

        # Technically one should correct for the polar wobble here, but the differences
        # along the equitorial are miniscule -- of order 10s of nanoradians, well below
        # the promised accuracy of IERS -- and rotation matricies can be expensive.
        # We do want to correct though for for secular polar drift (s'/TIO locator),
        # which nudges the Earth rotation angle of order 47 uas per century.
        sp = erfa.sp00(times.tt.jd1, times.tt.jd2)

        lst_array = np.mod(gast_array + sp + site_loc.lon.rad, 2.0 * np.pi)[
            reverse_inds
        ]
    elif astrometry_library == "astropy":
        lst_array = times.sidereal_time("apparent").radian
        if lst_array.ndim == 0:
            lst_array = lst_array.reshape(1)
        lst_array = lst_array[reverse_inds]
    elif astrometry_library == "novas":
        # Import the NOVAS library only if it's needed/available.
        try:
            import novas_de405  # noqa
            from novas import compat as novas
            from novas.compat import eph_manager
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "novas and/or novas_de405 are not installed but is required for "
                "NOVAS functionality"
            ) from e

        jd_start, jd_end, number = eph_manager.ephem_open()

        tt_time_array = times.tt.value
        ut1_high_time_array = times.ut1.jd1
        ut1_low_time_array = times.ut1.jd2
        full_ut1_time_array = ut1_high_time_array + ut1_low_time_array
        polar_motion_data = iers.earth_orientation_table.get()

        delta_x_array = np.interp(
            times.mjd,
            polar_motion_data["MJD"].value,
            polar_motion_data["dX_2000A_B"].value,
            left=0.0,
            right=0.0,
        )

        delta_y_array = np.interp(
            times.mjd,
            polar_motion_data["MJD"].value,
            polar_motion_data["dY_2000A_B"].value,
            left=0.0,
            right=0.0,
        )

        # Catch the case where we don't have CIP delta values yet (they don't typically
        # have predictive values like the polar motion does)
        delta_x_array[np.isnan(delta_x_array)] = 0.0
        delta_y_array[np.isnan(delta_y_array)] = 0.0

        for idx in range(len(times)):
            novas.cel_pole(
                tt_time_array[idx], 2, delta_x_array[idx], delta_y_array[idx]
            )
            # The NOVAS routine will return Greenwich Apparent Sidereal Time (GAST),
            # in units of hours
            lst_array[reverse_inds == idx] = novas.sidereal_time(
                ut1_high_time_array[idx],
                ut1_low_time_array[idx],
                (tt_time_array[idx] - full_ut1_time_array[idx]) * 86400.0,
            )

        # Add the telescope lon to convert from GAST to LAST (local)
        lst_array = np.mod(lst_array + (longitude / 15.0), 24.0)

        # Convert from hours back to rad
        lst_array *= np.pi / 12.0

    lst_array = np.reshape(lst_array, jd_array.shape)

    return lst_array
