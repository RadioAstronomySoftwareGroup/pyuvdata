# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""A module defining functions for initializing UVData objects from scratch."""
from __future__ import annotations

import warnings
from itertools import combinations_with_replacement
from typing import Any, Literal, Sequence

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from .. import Telescope, __version__, utils
from ..telescopes import Locations


def get_time_params(
    *,
    telescope_location: Locations,
    time_array: np.ndarray,
    integration_time: float | np.ndarray | None = None,
    astrometry_library: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Configure time parameters for new UVData object."""
    if not isinstance(time_array, np.ndarray):
        raise ValueError(f"time_array must be a numpy array, got {type(time_array)}")

    lst_array = utils.get_lst_for_time(
        time_array,
        latitude=telescope_location.lat.deg,
        longitude=telescope_location.lon.deg,
        altitude=telescope_location.height.to_value("m"),
        frame="itrs" if isinstance(telescope_location, EarthLocation) else "mcmf",
        ellipsoid=(
            None
            if isinstance(telescope_location, EarthLocation)
            else telescope_location.ellipsoid
        ),
        astrometry_library=astrometry_library,
    )

    if integration_time is None:
        if time_array.ndim == 2:
            integration_time = np.squeeze(np.diff(time_array, axis=1) * 86400)
        else:
            utimes = np.sort(list(set(time_array)))
            if len(utimes) > 1:
                integration_time = np.diff(utimes) * 86400
                integration_time = np.concatenate(
                    [integration_time, integration_time[-1:]]
                )
            else:
                warnings.warn(
                    "integration_time not provided, and cannot be inferred from "
                    "time_array, setting to 1 second"
                )
                integration_time = np.array([1.0])

    if np.isscalar(integration_time):
        if time_array.ndim == 2:
            integration_time = np.full_like(time_array[:, 0], integration_time)
        else:
            integration_time = np.full_like(time_array, integration_time)

    try:
        integration_time = np.asarray(integration_time, dtype=float)
    except TypeError as e:
        raise TypeError("integration_time must be array_like of floats") from e

    if time_array.ndim == 2:
        if integration_time.size != time_array.shape[0]:
            raise ValueError(
                "integration_time must be the same length as the first axis of "
                "time_range."
            )
    else:
        if integration_time.shape != time_array.shape:
            raise ValueError("integration_time must be the same shape as time_array.")

    return lst_array, integration_time


def get_freq_params(
    freq_array: np.ndarray, *, channel_width: float | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Configure frequency parameters for new UVData object."""
    if not isinstance(freq_array, np.ndarray):
        raise ValueError("freq_array must be a numpy array.")

    if channel_width is None:
        if freq_array.size > 1:
            channel_width = freq_array[1:] - freq_array[:-1]
            channel_width = np.concatenate([channel_width, channel_width[-1:]])
        else:
            warnings.warn(
                "channel_width not provided, and cannot be inferred from freq_array, "
                "setting to 1 Hz"
            )
            channel_width = np.array([1.0])
    elif np.isscalar(channel_width):
        channel_width = np.full_like(freq_array, channel_width)

    try:
        channel_width = np.asarray(channel_width, dtype=float)
    except TypeError as e:
        raise TypeError("channel_width must be array_like of floats") from e

    if channel_width.shape != freq_array.shape:
        raise ValueError("channel_width must be the same shape as freq_array.")

    return freq_array, channel_width


def get_baseline_params(
    *, antenna_numbers: np.ndarray, antpairs: np.ndarray
) -> np.ndarray:
    """Configure baseline parameters for new UVData object."""
    return utils.antnums_to_baseline(
        antpairs[:, 0], antpairs[:, 1], Nants_telescope=len(antenna_numbers)
    )


def configure_blt_rectangularity(
    *,
    times: np.ndarray,
    antpairs: np.ndarray,
    do_blt_outer: bool | None = None,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
    time_sized_arrays: tuple[np.ndarray] = (),
):
    """Configure blt rectangularity parameters for new UVData object."""
    # Whatever we do, we have to find the unique set of times and baselines, either
    # to check the user input, or to use it.
    unique_times = np.unique(times)
    unique_antpairs = np.unique(antpairs, axis=0)
    nt, nbl = len(unique_times), len(unique_antpairs)

    if do_blt_outer is None:
        if len(times) != len(antpairs):
            if nt != len(times):
                raise ValueError(
                    "If times and antpairs differ in length, times must all be unique"
                )
            if nbl != len(antpairs):
                raise ValueError(
                    "If times and antpairs differ in length, "
                    "antpairs must all be unique"
                )

            # We must have to do
            do_blt_outer = True
        elif nbl != len(antpairs) or nt != len(times):
            do_blt_outer = False
        else:
            raise ValueError(
                "It is impossible to determine whether you intend to do an outer "
                "product of the times and antpairs, since both only contain unique "
                "values (suggesting you want the outer product) but they are both "
                "the same length (suggesting you don't want the outer product). "
                "Set do_blt_outer to True or False to resolve this ambiguity."
            )

    if do_blt_outer:
        # Make sure times and antpairs are all unique.
        if nt != len(times):
            raise ValueError("times must be unique if do_blt_outer is True.")
        if nbl != len(antpairs):
            raise ValueError("antpairs must be unique if do_blt_outer is True.")

        if time_axis_faster_than_bls is None:
            time_axis_faster_than_bls = False

        blts_are_rectangular = True

        if time_axis_faster_than_bls:
            times = np.tile(unique_times, nbl)
            time_sized_arrays = tuple(np.tile(arr, nbl) for arr in time_sized_arrays)
            antpairs = np.repeat(unique_antpairs, nt, axis=0)
        else:
            antpairs = np.tile(antpairs, (nt, 1))
            times = np.repeat(times, nbl)
            time_sized_arrays = tuple(np.repeat(arr, nbl) for arr in time_sized_arrays)

    elif blts_are_rectangular:
        # Do a basic check that this is true.
        if len(times) != len(antpairs) or len(times) != nt * nbl:
            raise ValueError(
                "blts_are_rectangular is True, but times and antpairs are "
                "not rectangular."
            )

        # It's still possible that they're not really rectangular, but we trust
        # the user to know what they're doing.

        # Since this is fast, we just do it, rather than trusting the user.
        time_axis_faster_than_bls = np.abs(times[1] - times[0]) > 0

    elif blts_are_rectangular is False:
        time_axis_faster_than_bls = None
    else:
        # We don't know if it's rectangular or not.
        # Let's try to figure it out.
        baselines = utils.antnums_to_baseline(
            antpairs[:, 0],
            antpairs[:, 1],
            Nants_telescope=len(np.unique(unique_antpairs)),
        )

        (blts_are_rectangular, time_axis_faster_than_bls) = (
            utils.bltaxis.determine_rectangularity(
                time_array=times, baseline_array=baselines, nbls=nbl, ntimes=nt
            )
        )

    return (
        nbl,
        nt,
        blts_are_rectangular,
        time_axis_faster_than_bls,
        times,
        antpairs,
        time_sized_arrays,
    )


def set_phase_params(obj, *, phase_center_catalog, phase_center_id_array, time_array):
    """Configure phase center parameters for new UVData object."""
    if phase_center_catalog is None:
        obj._add_phase_center(cat_name="unprojected", cat_type="unprojected")
    else:
        for key, cat in phase_center_catalog.items():
            obj._add_phase_center(cat_id=key, **cat)

    if phase_center_id_array is None:
        if len(obj.phase_center_catalog) > 1:
            raise ValueError(
                "If phase_center_catalog has more than one key, phase_center_id_array "
                "must be provided."
            )

        phase_center_id_array = np.full(
            len(time_array), next(iter(obj.phase_center_catalog.keys())), dtype=int
        )
    obj.phase_center_id_array = phase_center_id_array
    obj._set_app_coords_helper()


def get_spw_params(
    *,
    flex_spw_id_array: np.ndarray | None = None,
    freq_array: np.ndarray | None = None,
    spw_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Configure spectral window parameters for new UVData object."""
    if flex_spw_id_array is None:
        if spw_array is None:
            flex_spw_id_array = np.zeros(freq_array.shape[0], dtype=int)
            spw_array = np.array([0])
        elif spw_array is not None:
            if len(spw_array) > 1:
                raise ValueError(
                    "If spw_array has more than one entry, flex_spw_id_array must be "
                    "provided."
                )
            flex_spw_id_array = np.full(freq_array.shape[0], spw_array[0], dtype=int)
    elif spw_array is None:
        spw_array = np.sort(np.unique(flex_spw_id_array))
    elif len(np.unique(spw_array)) != len(np.unique(flex_spw_id_array)):
        raise ValueError(
            "spw_array and flex_spw_id_array must have the same number of unique "
            "values."
        )

    return flex_spw_id_array, spw_array


def new_uvdata(
    *,
    freq_array: np.ndarray,
    polarization_array: np.ndarray | list[str | int] | tuple[str | int],
    times: np.ndarray,
    telescope: Telescope | None = None,
    antpairs: Sequence[tuple[int, int]] | np.ndarray | None = None,
    do_blt_outer: bool | None = None,
    integration_time: float | np.ndarray | None = None,
    channel_width: float | np.ndarray | None = None,
    telescope_location: Locations | None = None,
    telescope_name: str | None = None,
    antenna_positions: np.ndarray | dict[str | int, np.ndarray] | None = None,
    antenna_names: list[str] | None = None,
    antenna_numbers: list[int] | None = None,
    blts_are_rectangular: bool | None = None,
    data_array: np.ndarray | None = None,
    flag_array: np.ndarray | None = None,
    nsample_array: np.ndarray | None = None,
    flex_spw_id_array: np.ndarray | None = None,
    history: str = "",
    instrument: str | None = None,
    vis_units: Literal["Jy", "K str", "uncalib"] = "uncalib",
    antname_format: str = "{0:03d}",
    empty: bool = False,
    time_axis_faster_than_bls: bool | None = None,
    phase_center_catalog: dict[str, Any] | None = None,
    phase_center_id_array: np.ndarray | None = None,
    x_orientation: Literal["east", "north", "e", "n", "ew", "ns"] | None = None,
    astrometry_library: str | None = None,
    **kwargs,
):
    """Initialize a new UVData object from keyword arguments.

    Parameters
    ----------
    freq_array : ndarray of float
        Array of frequencies in Hz.
    polarization_array : sequence of int or str
        Array of polarization integers or strings (eg. 'xx' or 'ee')
    times : ndarray of float, optional
        Array of times in Julian Date. These may be the *unique* times of the data if
        each baseline observes the same set of times, otherwise they should be an
        Nblts-length array of each time observed by each baseline. It is recommended
        to set the ``do_blt_outer`` parameter to specify whether to apply the times
        to each baseline.
    telescope : pyuvdata.Telescope
        Telescope object containing the telescope-related metadata including
        telescope name and location, x_orientation and antenna names, numbers
        and positions.
    antpairs : sequence of 2-tuples of int or 2D array of int, optional
        Antenna pairs in the data. If an ndarray, must have shape (N antpairs, 2).
        These may be the *unique* antpairs of the data if
        each antpair observes the same set of times, otherwise they should be an
        Nblts-length array of each antpair at each time. It is recommended
        to set the ``do_blt_outer`` parameter to specify whether to apply the times
        to each baseline. If not provided, use all unique antpairs determined from
        the ``antenna_numbers``.
    do_blt_outer : bool, optional
        If True, the final ``time_array`` and ``baseline_array`` will contain
        ``len(times)*len(antpairs)`` entries (one for each pair cartesian product of
        times and antpairs). If False, the final ``time_array`` and ``baseline_array``
        will be the same as the input ``times`` and ``antpairs``. If not provided,
        it will be set to True if ``times`` and ``antpairs`` are different lengths
        and all unique, otherwise it will be set to False.
    integration_time : float or ndarray of float, optional
        Integration time in seconds. If not provided, it will be derived from the
        time_array, as the difference between successive times (with the last time-diff
        appended). If not provided and the number of unique times is one, then
        a warning will be raised and the integration time set to 1 second.
        If a float is provided, it will be used for all integrations.
        If an ndarray is provided, it must have the same shape as time_array (or
        unique_times, if that is what is provided).
    channel_width : float or ndarray of float, optional
        Channel width in Hz. If not provided, it will be derived from the freq_array,
        as the difference between successive frequencies (with the last frequency-diff
        appended). If a float is provided, it will be used for all channels.
        If not provided and freq_array is length-one, the channel_width will be set to
        1 Hz (and a warning issued). If an ndarray is provided, it must have the same
        shape as freq_array.
    telescope_location : EarthLocation or MoonLocation object
        Deprecated. Telescope location as an astropy EarthLocation object or
        MoonLocation object. Not required or used if a Telescope object is
        passed to `telescope`.
    telescope_name : str
        Deprecated. Telescope name. Not required or used if a Telescope object
        is passed to `telescope`.
    antenna_positions : ndarray of float or dict of ndarray of float
        Deprecated. Array of antenna positions in ECEF coordinates in meters.
        If a dict, keys can either be antenna numbers or antenna names, and values are
        position arrays. Keys are interpreted as antenna numbers if they are integers,
        otherwise they are interpreted as antenna names if strings. You cannot
        provide a mix of different types of keys.
        Not required or used if a Telescope object is passed to `telescope`.
    antenna_names : list of str, optional
        Deprecated. List of antenna names. Not required or used if a Telescope
        object is passed to `telescope` or if antenna_positions is a dict with
        string keys. Otherwise, if not provided, antenna numbers will be used
        to form the antenna_names, according to the antname_format
    antenna_numbers : list of int, optional
        Deprecated. List of antenna numbers. Not required or used if a Telescope
        object is passed to `telescope` or if antenna_positions is a dict with
        integer keys. Otherwise, if not provided, antenna names will be used to
        form the antenna_numbers, but in this case the antenna_names must be
        strings that can be converted to integers.
    antname_format : str, optional
        Deprecated. Format string for antenna names. Default is '{0:03d}'.
    x_orientation : str, optional
        Deprecated. Orientation of the x-axis. Options are 'east', 'north',
        'e', 'n', 'ew', 'ns'. Not used if a Telescope object is passed to
        `telescope`.
    instrument : str, optional
        Deprecated. Instrument name. Default is the ``telescope_name``. Not used
        if a Telescope object is passed to `telescope`.
    blts_are_rectangular : bool, optional
        Set to True if the time_array and antpair_array are rectangular, i.e. if
        they are formed from the outer product of a unique set of times/antenna pairs.
        If not provided, it will be inferred from the provided arrays.
    data_array : ndarray of complex, optional
        Array of data. If not provided, and ``empty=True`` it will be initialized to all
        zeros. Otherwise a purely metadata-only object will be created. It must have
        shape ``(Nblts, Nfreqs, Npols)``
    flag_array : ndarray of bool, optional
        Array of flags. If not provided, and ``empty=True`` it will be initialized to
        all False. Otherwise a purely metadata-only object will be created. It must have
        shape ``(Nblts, Nfreqs, Npols)``
    nsample_array : ndarray of float, optional
        Array of nsamples. If not provided, and ``empty=True`` it will be initialized to
        all zeros. Otherwise a purely metadata-only object will be created. It must have
        shape ``(Nblts, Nfreqs, Npols)``
    flex_spw_id_array : ndarray of int, optional
        Array of spectral window IDs, one for each frequency. If not provided, it will
        be initialized to all zeros. It must have shape ``(Nfreqs,)``.
    history : str, optional
        History string to be added to the object. Default is a simple string
        containing the date and time and pyuvdata version.
    vis_units : str, optional
        Visibility units. Default is 'uncalib'. Must be one of 'Jy', 'K str', or
        'uncalib'.
    empty : bool, optional
        Set to True to create an empty (but not metadata-only) UVData object.
        Default is False.
    time_axis_faster_than_bls : bool, optional
        Set to True if the time axis is faster than the baseline axis, under the
        assumption that the blt axis is rectangular. This *sets* the blt order in the
        case that unique times and unique baselines are provided, otherwise it is
        inferred from the provided arrays.
    phase_center_catalog : dict, optional
        Dictionary of phase center information. Each key is a source id, and each
        value is a dictionary. By default, a single phase center is assumed, at
        zenith, assumed to be unprojected.
    phase_center_id_array : ndarray of int, optional
        Array of phase center ids. If not provided, it will be initialized to the first
        id found in ``phase_center_catalog``. It must have shape ``(Nblts,)``.
    astrometry_library : str
        Library used for calculating LSTs. Allowed options are 'erfa' (which uses
        the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
        (which uses the astropy utilities). Default is erfa unless the
        telescope_location frame is MCMF (on the moon), in which case the default
        is astropy.

    Other Parameters
    ----------------
    All other keyword parameters are set on the UVData object as attributes.

    Returns
    -------
    obj : UVData object
        A UVData object with the specified metadata.
    """
    # import here to avoid circular import
    from pyuvdata import UVData

    obj = UVData()

    if telescope is None:
        required_without_tel = {
            "antenna_positions": antenna_positions,
            "telescope_location": telescope_location,
            "telescope_name": telescope_name,
        }
        for key, value in required_without_tel.items():
            if value is None:
                raise ValueError(f"{key} is required if telescope is not provided.")
        antenna_diameters = kwargs.pop("antenna_diameters", None)
        old_params = {
            "telescope_name": telescope_name,
            "telescope_location": telescope_location,
            "antenna_positions": antenna_positions,
            "antenna_names": antenna_names,
            "antenna_numbers": antenna_numbers,
            "instrument": instrument,
            "x_orientation": x_orientation,
            "antenna_diameters": antenna_diameters,
        }
        warn_params = []
        for param, val in old_params.items():
            if val is not None:
                warn_params.append(param)
        warnings.warn(
            f"Passing {', '.join(warn_params)} is deprecated in favor of passing "
            "a Telescope object via the `telescope` parameter. This will become "
            "an error in version 3.2",
            DeprecationWarning,
        )
    else:
        required_on_tel = [
            "name",
            "location",
            "antenna_positions",
            "antenna_names",
            "antenna_numbers",
            "Nants",
            "instrument",
        ]
        for key in required_on_tel:
            if getattr(telescope, key) is None:
                raise ValueError(
                    f"{key} must be set on the Telescope object passed to `telescope`."
                )

    if telescope is None:
        if instrument is None:
            instrument = telescope_name
        telescope = Telescope.new(
            name=telescope_name,
            location=telescope_location,
            antenna_positions=antenna_positions,
            antenna_names=antenna_names,
            antenna_numbers=antenna_numbers,
            antname_format=antname_format,
            instrument=instrument,
            x_orientation=x_orientation,
            antenna_diameters=antenna_diameters,
        )

    lst_array, integration_time = get_time_params(
        telescope_location=telescope.location,
        time_array=times,
        integration_time=integration_time,
        astrometry_library=astrometry_library,
    )

    if antpairs is None:
        bl_order = ("ant1", "ant2")
        antpairs = list(combinations_with_replacement(telescope.antenna_numbers, 2))
        do_blt_outer = True
    else:
        bl_order = None

    (
        nbls,
        ntimes,
        blts_are_rectangular,
        time_axis_faster_than_bls,
        time_array,
        antpairs,
        (lst_array, integration_time),
    ) = configure_blt_rectangularity(
        times=times,
        antpairs=np.asarray(antpairs).astype(int),
        do_blt_outer=do_blt_outer,
        blts_are_rectangular=blts_are_rectangular,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
        time_sized_arrays=(lst_array, integration_time),
    )

    if not do_blt_outer:
        if time_array.size != antpairs.shape[0]:
            raise ValueError("Length of time array must match the number of antpairs.")

    if bl_order is not None and blts_are_rectangular:
        if time_axis_faster_than_bls:
            blt_order = ("ant1", "ant2")
        else:
            blt_order = ("time", "ant1")
    else:
        blt_order = None

    baseline_array = get_baseline_params(
        antenna_numbers=telescope.antenna_numbers, antpairs=antpairs
    )

    # Re-get the ant arrays because the baseline array may have changed
    ant_1_array, ant_2_array = antpairs.T

    freq_array, channel_width = get_freq_params(
        freq_array=freq_array, channel_width=channel_width
    )

    flex_spw_id_array, spw_array = get_spw_params(
        flex_spw_id_array=flex_spw_id_array, freq_array=freq_array
    )

    polarization_array = np.array(polarization_array)
    if polarization_array.dtype.kind != "i":
        polarization_array = utils.polstr2num(
            polarization_array, x_orientation=x_orientation
        )

    if vis_units not in ["Jy", "K str", "uncalib"]:
        raise ValueError("vis_units must be one of 'Jy', 'K str', or 'uncalib'.")

    history += (
        f"Object created by new_uvdata() at {Time.now().iso} using "
        f"pyuvdata version {__version__}."
    )

    # Now set all the metadata
    obj.telescope = telescope
    # set the appropriate telescope attributes as required
    obj._set_telescope_requirements()

    obj.freq_array = freq_array
    obj.polarization_array = polarization_array
    obj.baseline_array = baseline_array
    obj.ant_1_array = ant_1_array
    obj.ant_2_array = ant_2_array
    obj.time_array = time_array
    obj.lst_array = lst_array
    obj.channel_width = channel_width
    obj.history = history
    obj.vis_units = vis_units
    obj.Nants_data = len(set(np.concatenate([ant_1_array, ant_2_array])))
    obj.Nbls = nbls
    obj.Nblts = len(baseline_array)
    obj.Nfreqs = len(freq_array)
    obj.Npols = len(polarization_array)
    obj.Ntimes = ntimes
    obj.Nspws = len(spw_array)
    obj.spw_array = spw_array
    obj.flex_spw_id_array = flex_spw_id_array
    obj.integration_time = integration_time
    obj.blt_order = blt_order

    set_phase_params(
        obj,
        phase_center_catalog=phase_center_catalog,
        phase_center_id_array=phase_center_id_array,
        time_array=time_array,
    )
    obj.set_uvws_from_antenna_positions(update_vis=False)

    # Set optional parameters that the user passes in
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise ValueError(f"Keyword argument {key} is not a valid UVData attribute.")

    # Set data parameters
    shape = (obj.Nblts, obj.Nfreqs, obj.Npols)

    if empty:
        obj.data_array = np.zeros(shape, dtype=complex)
        obj.flag_array = np.zeros(shape, dtype=bool)
        obj.nsample_array = np.ones(shape, dtype=float)

    elif data_array is not None:
        if data_array.shape != shape:
            raise ValueError(
                f"Data array shape {data_array.shape} does not match "
                f"expected shape {shape}."
            )

        obj.data_array = data_array

        if flag_array is not None:
            if flag_array.shape != shape:
                raise ValueError(
                    f"Flag array shape {flag_array.shape} does not match expected "
                    f"shape {shape}."
                )
            obj.flag_array = flag_array
        else:
            obj.flag_array = np.zeros(shape, dtype=bool)

        if nsample_array is not None:
            if nsample_array.shape != shape:
                raise ValueError(
                    f"nsample array shape {nsample_array.shape} does not match expected"
                    f" shape {shape}."
                )
            obj.nsample_array = nsample_array
        else:
            obj.nsample_array = np.ones(shape, dtype=float)

    obj.check()
    return obj
