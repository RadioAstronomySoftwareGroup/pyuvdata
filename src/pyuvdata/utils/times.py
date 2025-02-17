# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working LSTs."""

import warnings

import erfa
import numpy as np
from astropy import units
from astropy.time import Time
from astropy.utils import iers

from .coordinates import get_loc_obj
from .tools import (
    _is_between,
    _strict_raise,
    _test_array_constant,
    _test_array_constant_spacing,
    _where_combine,
)


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
        Alternative way of specifying telescope lat/lon/alt, either as a 3-element
        tuple with latitude and longitude in degrees, or as an astropy
        EarthLocation (or lunarsky MoonLocation). Cannot supply both
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
    if telescope_loc is None:
        if all(item is not None for item in [latitude, longitude, altitude]):
            telescope_loc = (latitude, longitude, altitude)
        else:
            raise ValueError(
                "Must supply all of latitude, longitude and altitude if "
                "telescope_loc is not supplied"
            )
    else:
        if not all(item is None for item in [latitude, longitude, altitude]):
            raise ValueError(
                "Cannot set both telescope_loc and latitude/longitude/altitude"
            )
    site_loc, on_moon = get_loc_obj(
        telescope_loc,
        telescope_frame=frame,
        ellipsoid=ellipsoid,
        angle_units=units.deg,
        return_moon=True,
    )

    if astrometry_library is None:
        if not on_moon:
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

    if not on_moon:
        TimeClass = Time
    else:
        if not astrometry_library == "astropy":
            raise NotImplementedError(
                "The MCMF frame is only supported with the 'astropy' astrometry library"
            )
        from lunarsky import Time as LTime

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
        except ImportError as e:
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
        lst_array = np.mod(lst_array + (site_loc.lon.deg / 15.0), 24.0)

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
    time_inds=None,
    invert=False,
    strict=False,
    warn_spacing=False,
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
    time_inds : array_like of int, optional
        The time indices to keep in the object. This is not commonly used.
    invert : bool
        Normally indices matching given criteria are what are included in the
        subsequent list. However, if set to True, these indices are excluded
        instead. Default is False.
    strict : bool or None
        Normally, select will warn when an element of the selection criteria does not
        match any element for the parameter, as long as the selection criteria results
        in *at least one* element being selected. However, if set to True, an error is
        thrown if any selection criteria does not match what is given for the object
        parameters element. If set to None, then neither errors nor warnings are raised,
        unless no records are selected. Default is False.
    warn_spacing
        Whether or not to warn about time spacing. Only used if no ranges from the
        object are supplied. Default is False.

    Returns
    -------
    time_inds : list of int
        Indices of times to keep on the object.
    selections : list of str
        list of selections done.

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
        return None, []

    if time_range is not None:
        time_range = np.asarray(time_range)
        if time_range.size != 2:
            raise ValueError("time_range must be length 2.")

    if have_lsts and np.any(np.asarray(lsts) > 2 * np.pi):
        warnings.warn(
            "The lsts parameter contained a value greater than 2*pi. "
            "LST values are assumed to be in radians, not hours."
        )

    if have_lst_range:
        lst_range = np.asarray(lst_range)
        if lst_range.size != 2:
            raise ValueError("lst_range must be length 2.")
        if np.any(lst_range > 2 * np.pi):
            warnings.warn(
                "The lst_range contained a value greater than 2*pi. "
                "LST values are assumed to be in radians, not hours."
            )

    selections = ["lsts"] if (have_lsts or have_lst_range) else ["times"]
    sel_name = "LST" if (have_lsts or have_lst_range) else "Time"
    obj_range = obj_lst_range if (have_lsts or have_lst_range) else obj_time_range
    obj_array = obj_lst_array if (have_lsts or have_lst_range) else obj_time_array
    sel_range = lst_range if (have_lsts or have_lst_range) else time_range
    sel_array = lsts if (have_lsts or have_lst_range) else times
    rtol, atol = lst_tols if (have_lsts or have_lst_range) else time_tols

    if obj_range is not None:
        err_txt = "does not fall in any"
        attr_name = "lst_range" if (have_lsts or have_lst_range) else "time_range"
    else:
        err_txt = "is not present in the"
        attr_name = "lst_array" if (have_lsts or have_lst_range) else "time_array"

    if sel_range is not None:
        # This is a range-based selection, so act accordingly
        if obj_range is not None:
            wrap_range = have_lst_range and (
                (sel_range[0] > sel_range[1]) or any(obj_range[:, 0] < obj_range[:, 1])
            )
            if wrap_range:
                # If we _do_ need to wrap ranges,
                obj_range = np.mod(obj_range, 2 * np.pi)
                lo_lim, hi_lim = np.mod(sel_range, 2 * np.pi)
                if lo_lim > hi_lim:
                    lo_lim -= 2 * np.pi

                # If LST is wrapped in obj_range, then do the unwrap
                obj_range[obj_range[:, 0] > obj_range[:, 1], 1] += 2 * np.pi

                # Check that the select end-time is after the range start-time and the
                # select start-time is before the range end-time.
                mask = (obj_range[:, 0] <= hi_lim) & (obj_range[:, 1] >= lo_lim)

                # Now check the wrap -- push the start/stop up one wrap, and then
                # re-evaluate the mask one one time.
                lo_lim += 2 * np.pi
                hi_lim += 2 * np.pi
                mask |= (obj_range[:, 0] <= hi_lim) & (obj_range[:, 1] >= lo_lim)
            else:
                # If we don't need to wrap ranges, then this is a lot easier to handle.
                # Check that the select end-time is after the range start-time and the
                # select start-time is before the range end-time.
                mask = obj_range[:, 0] <= sel_range[1]
                mask &= obj_range[:, 1] >= sel_range[0]
        else:
            wrap_range = have_lst_range and (sel_range[0] > sel_range[1])
            # Get sel_range into the correct shape
            mask = _is_between(obj_array, sel_range, wrap=wrap_range)
        if not any(mask):
            msg = (
                f"No elements in {attr_name} between {sel_range[0]} and {sel_range[1]}."
            )
            _strict_raise(msg, strict=strict)
    else:
        # This is a match-based selection, so move forward accordingly
        if obj_range is None:
            # Because of tols, everything is in effect a range -- construct the
            # effect object ranges that we need.
            del_range = np.maximum(abs(obj_array * rtol), abs(atol))
            obj_range = np.vstack([obj_array - del_range, obj_array + del_range]).T

        mask = np.zeros(len(obj_range), dtype=bool)
        for item in np.asarray(sel_array).flat:
            submask = _is_between(item, obj_range, wrap=have_lsts)
            if not any(submask):
                msg = f"{sel_name} {item} {err_txt} {attr_name}."
                _strict_raise(msg, strict=strict)
            mask |= submask

    time_inds = _where_combine(mask, inds=time_inds, invert=invert)

    if len(time_inds) == 0:
        raise ValueError(
            f"No data matching this {sel_name.lower()} selection present in object."
        )

    time_inds = time_inds.tolist()
    if warn_spacing:
        if (len(time_inds) > 1) and (obj_time_range is not None):
            warnings.warn(
                "Selected times include multiple time ranges. This "
                "is not supported by some file formats."
            )
        elif (
            (obj_time_range is None)
            and (obj_time_array is not None)
            and (not _test_array_constant(np.diff(obj_time_array[time_inds])))
        ):
            warnings.warn(
                "Selected times are not evenly spaced. This "
                "is not supported by some file formats."
            )

    return time_inds, selections


def _check_time_spacing(
    *,
    times_array,
    time_range=None,
    time_tols=None,
    integration_time=None,
    int_tols=None,
    strict=True,
):
    """
    Check if times are evenly spaced.

    This is a requirement for calfits files.

    Parameters
    ----------
    times_array : array-like of float or UVParameter
        Array of times, shape (Ntimes,).
    time_range : array-like of float or UVParameter
        Array of time ranges, shape (Ntime_ranges). Optional.
    time_tols : tuple of float
        time_array tolerances (from uvobj._time_array.tols).  Optional.
    integration_time : array-like of float or UVParameter
        Array of integration times, shape (Ntimes,).  Optional.
    channel_width_tols : tuple of float
        integration_time tolerances (from uvobj._integration_time.tols). Optional.
    strict : bool
        If set to True, then the function will raise an error if checks are failed.
        If set to False, then a warning is raised instead. If set to None, then
        no errors or warnings are raised. Default is True.
    """
    # Import UVParameter here rather than at the top to avoid circular imports
    from pyuvdata.parameter import UVParameter

    if isinstance(time_range, UVParameter):
        time_range = time_range.value

    if not _test_array_constant_spacing(times_array, tols=time_tols):
        err_msg = (
            "The times are not evenly spaced. This will make it impossible to write "
            "this data out to calfits."
        )
        _strict_raise(err_msg=err_msg, strict=strict)
    if time_range is not None and len(time_range) != 1:
        err_msg = (
            "Object contains multiple time ranges. This will make it impossible to "
            "write this data out to calfits."
        )
        _strict_raise(err_msg=err_msg, strict=strict)
    if not _test_array_constant(integration_time):
        err_msg = (
            "The integration times are variable. The calfits format "
            "does not support variable integration times."
        )
        _strict_raise(err_msg=err_msg, strict=strict)
