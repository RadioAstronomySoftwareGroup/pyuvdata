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

from .tools import _check_range_overlap, _get_iterable


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


def check_lsts_against_times(
    *,
    jd_array,
    lst_array,
    lst_tols,
    latitude=None,
    longitude=None,
    altitude=None,
    frame="itrs",
    ellipsoid=None,
    telescope_loc=None,
):
    """
    Check that LSTs are consistent with the time_array and telescope location.

    This just calls `get_lst_for_time`, compares that result to the `lst_array`
    and warns if they are not within the tolerances specified by `lst_tols`.

    Parameters
    ----------
    jd_array : ndarray of float
        JD times to get lsts for.
    lst_array : ndarray of float
        LSTs to check to see if they match the jd_array at the location.
    latitude : float
        Latitude of location to check the lst for in degrees.
    longitude : float
        Longitude of location to check the lst for in degrees.
    altitude : float
        Altitude of location to check the lst for in meters.
    lst_tops : tuple of float
        A length 2 tuple giving the (relative, absolute) tolerances to check the
        LST agreement to. These are passed directly to numpy.allclose.
    frame : str
        Reference frame for latitude/longitude/altitude.
        Options are itrs (default) or mcmf.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE", "GSFC",
        "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    telescope_loc : tuple or EarthLocation or MoonLocation
        Alternative way of specifying telescope lat/lon/alt, either as a 3-element tuple
        or as an astropy EarthLocation (or lunarsky MoonLocation). Cannot supply both
        `telescope_loc` and `latitute`, `longitude`, or `altitude`.

    Returns
    -------
    None

    Warns
    -----
    If the `lst_array` does not match the calculated LSTs to the lst_tols.

    """
    # Don't worry about passing the astrometry library because we test that they agree
    # to better than our standard lst tolerances.
    lsts = get_lst_for_time(
        jd_array=jd_array,
        telescope_loc=telescope_loc,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        frame=frame,
        ellipsoid=ellipsoid,
    )

    if not np.allclose(lst_array, lsts, rtol=lst_tols[0], atol=lst_tols[1]):
        warnings.warn(
            "The lst_array is not self-consistent with the time_array and "
            "telescope location. Consider recomputing with the "
            "`set_lsts_from_time_array` method."
        )


def _select_times_helper(
    *,
    times,
    time_range,
    lsts,
    lst_range,
    obj_time_array,
    obj_time_range,
    obj_lst_array,
    obj_lst_range,
    time_tols,
    lst_tols,
):
    """
    Get time indices in a select.

    Parameters
    ----------
    times : array_like of float
        The times to keep in the object, each value passed here should exist in the
        time_array. Can be None, cannot be set with `time_range`, `lsts` or `lst_array`.
    time_range : array_like of float
        The time range in Julian Date to keep in the object, must be length 2. Some of
        the times in the object should fall between the first and last elements. Can be
        None, cannot be set with `times`, `lsts` or `lst_array`.
    lsts : array_like of float
        The local sidereal times (LSTs) to keep in the object, each value passed here
        should exist in the lst_array. Can be None, cannot be set with `times`,
        `time_range`, or `lst_range`.
    lst_range : array_like of float
        The local sidereal time (LST) range in radians to keep in the
        object, must be of length 2. Some of the LSTs in the object should
        fall between the first and last elements. If the second value is
        smaller than the first, the LSTs are treated as having phase-wrapped
        around LST = 2*pi = 0, and the LSTs kept on the object will run from
        the larger value, through 0, and end at the smaller value. Can be None, cannot
        be set with `times`, `time_range`, or `lsts`.
    obj_time_array : array_like of float
        Time array on object. Can be None if `object_time_range` is set.
    obj_time_range : array_like of float
        Time range on object. Can be None if `object_time_array` is set.
    obj_lst_array : array_like of float
        LST array on object. Can be None if `object_lst_range` is set.
    obj_lst_range : array_like of float
        LST range on object. Can be None if `object_lst_array` is set.
    time_tols : tuple of float
        Length 2 tuple giving (rtol, atol) to use for time matching.
    lst_tols : tuple of float
        Length 2 tuple giving (rtol, atol) to use for lst matching.

    """
    have_times = times is not None
    have_time_range = time_range is not None
    have_lsts = lsts is not None
    have_lst_range = lst_range is not None
    n_time_params = np.count_nonzero(
        [have_times, have_time_range, have_lsts, have_lst_range]
    )
    if n_time_params > 1:
        raise ValueError(
            "Only one of [times, time_range, lsts, lst_range] may be "
            "specified per selection operation."
        )
    if n_time_params == 0:
        return None

    time_inds = np.zeros(0, dtype=np.int64)
    if times is not None:
        times = _get_iterable(times)
        if np.array(times).ndim > 1:
            times = np.array(times).flatten()

        if obj_time_range is not None:
            for jd in times:
                this_ind = np.nonzero(
                    np.logical_and(
                        (obj_time_range[:, 0] <= jd), (obj_time_range[:, 1] >= jd)
                    )
                )[0]
                if this_ind.size > 0:
                    time_inds = np.append(time_inds, this_ind)
                else:
                    raise ValueError(f"Time {jd} does not fall in any time_range.")
        else:
            for jd in times:
                if np.any(
                    np.isclose(obj_time_array, jd, rtol=time_tols[0], atol=time_tols[1])
                ):
                    time_inds = np.append(
                        time_inds,
                        np.where(
                            np.isclose(
                                obj_time_array, jd, rtol=time_tols[0], atol=time_tols[1]
                            )
                        )[0],
                    )
                else:
                    raise ValueError(f"Time {jd} is not present in the time_array.")

    if time_range is not None:
        if np.size(time_range) != 2:
            raise ValueError("time_range must be length 2.")

        if obj_time_range is not None:
            for tind, trange in enumerate(obj_time_range):
                if _check_range_overlap(np.stack((trange, time_range), axis=0)):
                    time_inds = np.append(time_inds, tind)
            attr_str = "time_range"
        else:
            time_inds = np.nonzero(
                (obj_time_array <= time_range[1]) & (obj_time_array >= time_range[0])
            )[0]
            attr_str = "time_array"
        if time_inds.size == 0:
            raise ValueError(
                f"No elements in {attr_str} between {time_range[0]} and "
                f"{time_range[1]}."
            )

    if (lsts is not None or lst_range is not None) and obj_lst_range is not None:
        # check for lsts wrapping around zero
        lst_range_wrap = obj_lst_range[:, 0] > obj_lst_range[:, 1]

    if lsts is not None:
        if np.any(np.asarray(lsts) > 2 * np.pi):
            warnings.warn(
                "The lsts parameter contained a value greater than 2*pi. "
                "LST values are assumed to be in radians, not hours."
            )
        lsts = _get_iterable(lsts)
        if np.array(lsts).ndim > 1:
            lsts = np.array(lsts).flatten()

        if obj_lst_range is not None:
            for lst in lsts:
                lst_ind = np.nonzero(
                    np.logical_and(
                        (obj_lst_range[:, 0] <= lst), (obj_lst_range[:, 1] >= lst)
                    )
                )[0]
                if lst_ind.size == 0 and np.any(lst_range_wrap):
                    for lr_ind in np.nonzero(lst_range_wrap)[0]:
                        if (obj_lst_range[lr_ind, 0] <= lst and lst <= 2 * np.pi) or (
                            lst >= 0 and lst <= obj_lst_range[lr_ind, 1]
                        ):
                            lst_ind = np.array([lr_ind])
                if lst_ind.size > 0:
                    time_inds = np.append(time_inds, lst_ind)
                else:
                    raise ValueError(f"LST {lst} does not fall in any lst_range")
        else:
            for lst in lsts:
                if np.any(
                    np.isclose(obj_lst_array, lst, rtol=lst_tols[0], atol=lst_tols[1])
                ):
                    time_inds = np.append(
                        time_inds,
                        np.where(
                            np.isclose(
                                obj_lst_array, lst, rtol=lst_tols[0], atol=lst_tols[1]
                            )
                        )[0],
                    )
                else:
                    raise ValueError(f"LST {lst} is not present in the lst_array")

    if lst_range is not None:
        if np.size(lst_range) != 2:
            raise ValueError("lst_range must be length 2.")
        if np.any(np.asarray(lst_range) > 2 * np.pi):
            warnings.warn(
                "The lst_range contained a value greater than 2*pi. "
                "LST values are assumed to be in radians, not hours."
            )
        if obj_lst_range is not None:
            for lind, lrange in enumerate(obj_lst_range):
                if not lst_range_wrap[lind] and lst_range[0] < lst_range[1]:
                    if _check_range_overlap(np.stack((lrange, lst_range), axis=0)):
                        time_inds = np.append(time_inds, lind)
                else:
                    if (lst_range[0] >= lrange[0] and lst_range[0] <= 2 * np.pi) or (
                        lst_range[1] <= lrange[1] and lst_range[1] >= 0
                    ):
                        time_inds = np.append(time_inds, lind)
            attr_str = "lst_range"
        else:
            if lst_range[1] < lst_range[0]:
                # we're wrapping around LST = 2*pi = 0
                lst_range_1 = [lst_range[0], 2 * np.pi]
                lst_range_2 = [0, lst_range[1]]
                time_inds1 = np.nonzero(
                    (obj_lst_array <= lst_range_1[1])
                    & (obj_lst_array >= lst_range_1[0])
                )[0]
                time_inds2 = np.nonzero(
                    (obj_lst_array <= lst_range_2[1])
                    & (obj_lst_array >= lst_range_2[0])
                )[0]
                time_inds = np.union1d(time_inds1, time_inds2)
            else:
                time_inds = np.nonzero(
                    (obj_lst_array <= lst_range[1]) & (obj_lst_array >= lst_range[0])
                )[0]
            attr_str = "lst_array"

        if time_inds.size == 0:
            raise ValueError(
                f"No elements in {attr_str} between {lst_range[0]} and "
                f"{lst_range[1]}."
            )
    return time_inds
