# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""From-memory initializers for UVCal objects."""
from __future__ import annotations

from typing import Literal

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from .. import __version__, utils
from ..docstrings import combine_docstrings
from ..uvdata.initializers import (
    XORIENTMAP,
    Locations,
    get_antenna_params,
    get_freq_params,
    get_spw_params,
    get_time_params,
)


def new_uvcal(
    *,
    time_array: np.ndarray,
    antenna_positions: np.ndarray | dict[str | int, np.ndarray],
    telescope_location: Locations,
    telescope_name: str,
    cal_style: Literal["sky", "redundant"],
    gain_convention: Literal["divide", "multiply"],
    jones_array: np.ndarray | str,
    x_orientation: Literal["east", "north", "e", "n", "ew", "ns"],
    freq_array: np.ndarray | None = None,
    freq_range: np.ndarray | None = None,
    cal_type: Literal["delay", "gain"] | None = None,
    integration_time: float | np.ndarray | None = None,
    channel_width: float | np.ndarray | None = None,
    antenna_names: list[str] | None = None,
    antenna_numbers: list[int] | None = None,
    antname_format: str = "{0:03d}",
    ant_array: np.ndarray | None = None,
    flex_spw_id_array: np.ndarray | None = None,
    ref_antenna_name: str | None = None,
    sky_catalog: str | None = None,
    empty: bool = False,
    data: dict[str, np.ndarray] | None = None,
    astrometry_library: str | None = None,
    history: str = "",
    **kwargs,
):
    r"""Create a new UVCal object with default parameters.

    Parameters
    ----------
    time_array : ndarray of float
        Array of times in Julian Date.
    antenna_positions : ndarray of float or dict of ndarray of float
        Array of antenna positions in ECEF coordinates in meters.
        If a dict, keys can either be antenna numbers or antenna names, and values are
        position arrays. Keys are interpreted as antenna numbers if they are integers,
        otherwise they are interpreted as antenna names if strings. You cannot
        provide a mix of different types of keys.
    telescope_location : ndarray of float or str
        Telescope location as an astropy EarthLocation object or MoonLocation object.
    telescope_name : str
        Telescope name.
    cal_style : str
        Calibration style. Options are 'sky' or 'redundant'.
    freq_array : ndarray of float, optional
        Array of frequencies in Hz. If given, the freq_range parameter is ignored, and
        the cal type is assumed to be "gain".
    freq_range : ndarray of float, optional
        Frequency ranges in Hz. Must be given if ``freq_array`` is not given. If given,
        the calibration is assumed to be wide band. The array shape should be (Nspws, 2)
    gain_convention : str
        Gain convention. Options are 'divide' or 'multiply'.
    x_orientation : str
        Orientation of the x-axis. Options are 'east', 'north', 'e', 'n', 'ew', 'ns'.
    jones_array : ndarray of int or str
        Array of Jones polarization integers. If a string, options are 'linear' or
        'circular' (which will be converted to the appropriate Jones integers as
        length-4 arrays).
    cal_type : str, optional
        Calibration type. Options are 'delay', 'gain'. Forced to be 'gain' if
        ``freq_array`` is given, and by *default* set to 'delay' if not.
    integration_time : float or ndarray of float, optional
        Integration time in seconds. If not provided, it will be derived from the
        time_array, as the difference between successive times (with the last time-diff
        appended). If not provided and the number of unique times is one, then
        a warning will be raised and the integration time set to 1 second.
        If a float is provided, it will be used for all integrations.
        If an ndarray is provided, it must have the same shape as time_array.
    channel_width : float or ndarray of float, optional
        Channel width in Hz. If not provided, it will be derived from the freq_array,
        as the difference between successive frequencies (with the last frequency-diff
        appended). If a float is provided, it will be used for all channels.
        If not provided and freq_array is length-one, the channel_width will be set to
        1 Hz (and a warning issued). If an ndarray is provided, it must have the same
        shape as freq_array.
    antenna_names : list of str, optional
        List of antenna names. If not provided, antenna numbers will be used to form
        the antenna_names, according to the antname_format. antenna_names need not be
        provided if antenna_positions is a dict with string keys.
    antenna_numbers : list of int, optional
        List of antenna numbers. If not provided, antenna names will be used to form
        the antenna_numbers, but in this case the antenna_names must be strings that
        can be converted to integers. antenna_numbers need not be provided if
        antenna_positions is a dict with integer keys.
    antname_format : str, optional
        Format string for antenna names. Default is '{0:03d}'.
    ant_array : ndarray of int, optional
        Array of antenna numbers actually found in data (in the order of the data
        in gain_array etc.)
    flex_spw_id_array : ndarray of int, optional
        Array of spectral window IDs. If not provided, it will be set to an array of
        zeros and only one spw will be used.
    ref_antenna_name : str, optional
        Name of reference antenna. Only required for sky calibrations.
    sky_catalog : str, optional
        Name of sky catalog. Only required for sky calibrations.
    empty : bool, optional
        If True, create an empty UVCal object, i.e. add initialized data arrays (eg.
        gain_array). By default, this function creates a metadata-only object. You can
        pass in data arrays directly to the constructor if you want to create a
        fully-populated object.
    data : dict of array_like, optional
        Dictionary containing optional data arrays. Possible keys are:
        'gain_array', 'delay_array', 'quality_array', 'flag_array',
        and 'total_quality_array'. If any entry is provided, the output will contain
        all necessary data-like arrays. Any key *not* provided in this case will be
        set to default "empty" values (e.g. all ones for gains, all False for flags) if
        they are required or `None` if they are not required (i.e. input_flag_array,
        quality_array, total_quality_array).
    history : str, optional
        History string to be added to the object. Default is a simple string
        containing the date and time and pyuvdata version.
    astrometry_library : str
        Library used for calculating LSTs. Allowed options are 'erfa' (which uses
        the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
        (which uses the astropy utilities). Default is erfa unless the
        telescope_location frame is MCMF (on the moon), in which case the default
        is astropy.
    \*\*kwargs
        All other keyword arguments are added to the object as attributes.

    Returns
    -------
    UVCal
        A new UVCal object with default parameters.

    """
    from .uvcal import UVCal

    uvc = UVCal()

    antenna_positions, antenna_names, antenna_numbers = get_antenna_params(
        antenna_positions, antenna_names, antenna_numbers, antname_format
    )
    if ant_array is None:
        ant_array = antenna_numbers
    else:
        # Ensure they all exist
        missing = [ant for ant in ant_array if ant not in antenna_numbers]
        if missing:
            raise ValueError(
                f"The following ants are not in antenna_numbers: {missing}"
            )

    lst_array, integration_time = get_time_params(
        telescope_location,
        time_array,
        integration_time,
        astrometry_library=astrometry_library,
    )

    if (freq_range is not None) and (
        freq_array is not None
        or channel_width is not None
        or flex_spw_id_array is not None
    ):
        raise ValueError(
            "Provide *either* freq_range *or* freq_array (and "
            "optionally, channel_width, and flex_spw_id_array)."
        )

    if freq_array is not None:
        freq_array, channel_width = get_freq_params(freq_array, channel_width)
        flex_spw_id_array, spw_array = get_spw_params(
            flex_spw_id_array, freq_array=freq_array
        )
        wide_band = False
        freq_range = None
        cal_type = "gain"
    elif freq_range is not None:
        # We're in wide-band mode.
        flex_spw_id_array = None
        wide_band = True
        cal_type = cal_type or "delay"
        freq_range = np.atleast_2d(np.array(freq_range))
        spw_array = np.arange(len(freq_range))
    else:
        raise ValueError(
            "You must provide either freq_array (optionally along with channel_width, "
            " and flex_spw_id_array) or freq_range."
        )

    if cal_type not in ("gain", "delay"):
        raise ValueError(f"cal_type must be either 'gain' or 'delay', got {cal_type}")

    x_orientation = XORIENTMAP[x_orientation.lower()]

    if isinstance(jones_array, str):
        if jones_array == "linear":
            jones_array = np.array([-5, -6, -7, -8])
        elif jones_array == "circular":
            jones_array = np.array([-1, -2, -3, -4])
    else:
        jones_array = np.array(jones_array)
        if jones_array.dtype.kind != "i":
            jones_array = utils.jstr2num(jones_array, x_orientation=x_orientation)

    history += (
        f"Object created by new_uvcal() at {Time.now().iso} using "
        f"pyuvdata version {__version__}."
    )

    if cal_style not in ("redundant", "sky"):
        raise ValueError(f"cal_style must be 'redundant' or 'sky', got {cal_style}")

    if cal_style == "sky" and (ref_antenna_name is None or sky_catalog is None):
        raise ValueError(
            "If cal_style is 'sky', ref_antenna_name and sky_catalog must be provided."
        )

    # Now set all the metadata
    uvc.freq_array = freq_array
    uvc.antenna_positions = antenna_positions
    uvc.telescope_location = np.array(
        [
            telescope_location.x.to_value("m"),
            telescope_location.y.to_value("m"),
            telescope_location.z.to_value("m"),
        ]
    )
    uvc.telescope_name = telescope_name
    uvc.time_array = time_array
    uvc.lst_array = lst_array
    uvc.integration_time = integration_time
    uvc.channel_width = channel_width
    uvc.antenna_names = antenna_names
    uvc.antenna_numbers = antenna_numbers
    uvc.history = history
    uvc.ant_array = ant_array
    uvc.telescope_name = telescope_name
    uvc.cal_style = cal_style
    uvc.cal_type = cal_type
    uvc.gain_convention = gain_convention
    uvc.x_orientation = x_orientation
    uvc.ref_antenna_name = ref_antenna_name
    uvc.sky_catalog = sky_catalog
    uvc.jones_array = jones_array
    uvc.freq_range = freq_range

    if cal_type == "delay":
        uvc._set_delay()
    elif cal_type == "gain":
        uvc._set_gain()

    if cal_style == "redundant":
        uvc._set_redundant()
    elif cal_style == "sky":
        uvc._set_sky()

    uvc._set_future_array_shapes()
    uvc._set_wide_band(wide_band)

    uvc.Nants_data = len(ant_array)
    uvc.Nants_telescope = len(antenna_numbers)
    uvc.Nfreqs = len(freq_array) if freq_array is not None else 1

    uvc.Ntimes = len(time_array)
    uvc.Nspws = len(spw_array)
    uvc.Njones = len(jones_array)

    uvc.flex_spw_id_array = flex_spw_id_array
    uvc.spw_array = spw_array

    if not wide_band:
        uvc._set_flex_spw()

    for k, v in kwargs.items():
        if hasattr(uvc, k):
            setattr(uvc, k, v)
        else:
            raise ValueError(f"Unrecognized keyword argument: {k}")

    if empty or data:
        data = data or {}
        shape = (
            uvc.Nants_data,
            uvc.Nspws if wide_band else uvc.Nfreqs,
            uvc.Ntimes,
            uvc.Njones,
        )

        # Flag array
        uvc.flag_array = data.get("flag_array", np.zeros(shape, dtype=bool))
        uvc.input_flag_array = data.get("input_flag_array", None)

        if cal_type == "delay":
            uvc.delay_array = data.get("delay_array", np.zeros(shape, dtype=float))
        else:
            uvc.gain_array = data.get("gain_array", np.ones(shape, dtype=complex))

        uvc.quality_array = data.get("quality_array", None)
        uvc.total_quality_array = data.get("total_quality_array", None)

    uvc.check()
    return uvc


@combine_docstrings(new_uvcal)
def new_uvcal_from_uvdata(
    uvdata,
    *,
    cal_style: Literal["sky", "redundant"],
    gain_convention: Literal["divide", "multiply"],
    cal_type: Literal["gain", "delay"] = "gain",
    jones_array: np.ndarray | str | None = None,
    wide_band: bool = False,
    include_uvdata_history: bool = True,
    spw_array: np.ndarray | None = None,
    **kwargs,
):
    """Construct a UVCal object with default attributes from a UVData object.

    Internally, this function takes whatever default parameters it can from the
    UVData object, and then overwrites them with any parameters passed in as
    kwargs. It then uses :func:`new_uvcal` to construct the UVCal object.

    Parameters
    ----------
    uvdata : UVData
        UVData object to use as a template.
    cal_type : str
        Calibration type. Must be 'gain' or 'delay'. Default 'gain'.
        Note that in contrast to :func:`new_uvcal`, this function sets the default
        to 'gain', instead of trying to set it based on the input freq_range.
    jones_array : array_like of int or str, optional
        Array of Jones polarizations. Taken from UVData object is possible (i.e. it is
        single-polarization data), otherwise must be specified. May be specified
        as 'circular' or 'linear' to indicate all circular or all linear polarizations.
    wide_band : bool
        Whether the calibration is wide-band (i.e. one calibration parameter per
        spectral window instead of per channel). Automatically set to True if cal_type
        is 'delay', otherwise the default is False. Note that if wide_band is True,
        the `freq_range` parameter is required.
    include_uvdata_history : bool
        Whether to include the UVData object's history in the UVCal object's history.
        Default is True.
    spw_array : array_like of int, optional
        Array of spectral window numbers. This will be taken from the UVData object
        if at all possible. The only time it is useful to pass this in explicitly is
        if ``wide_band=True`` and the spectral windows desired are not just
        ``(0, ..., Nspw-1)``.
    """  # noqa: RST203
    from ..uvdata import UVData

    if not isinstance(uvdata, UVData):
        raise ValueError("uvdata must be a UVData object.")

    uses_future_shapes = uvdata.future_array_shapes
    if not uses_future_shapes:
        uvdata.use_future_array_shapes()

    if cal_type == "delay":
        wide_band = True

    # If any time-length params are in kwargs, we can't use ANY of them from the UVData
    if "integration_time" in kwargs or "time_array" in kwargs:
        integration_time = kwargs.pop("integration_time", None)
        time_array = kwargs.pop("time_array", None)
    else:
        indx = np.unique(uvdata.time_array, return_index=True)[1]
        integration_time = uvdata.integration_time[indx]
        time_array = uvdata.time_array[indx]

    # Get frequency-type info from kwargs and uvdata
    if cal_type == "gain" and not wide_band:
        if "channel_width" in kwargs or "freq_array" in kwargs:
            channel_width = kwargs.pop("channel_width", None)
            freq_array = kwargs.pop("freq_array", None)
            flex_spw_id_array = kwargs.pop("flex_spw_id_array", None)
        else:
            freq_array = uvdata.freq_array
            channel_width = uvdata.channel_width
            flex_spw_id_array = getattr(
                uvdata, "flex_spw_id_array", kwargs.get("flex_spw_id_array", None)
            )
        freq_range = None
        if "freq_range" in kwargs:
            del kwargs["freq_range"]
        if spw_array is not None and len(spw_array) != len(
            np.unique(flex_spw_id_array)
        ):
            raise ValueError(
                "spw_array must be the same length as the number of unique spws in the "
                f"UVData object. Got {spw_array} and {np.unique(flex_spw_id_array)}."
            )
    else:
        channel_width = None
        freq_array = None
        flex_spw_id_array = None
        _spwids = getattr(uvdata, "flex_spw_id_array", None)

        if spw_array is None:
            if _spwids is None:
                spw_array = np.array([0])
            else:
                spw_array = np.sort(np.unique(_spwids))

        freq_range = kwargs.pop("freq_range", None)
        if freq_range is None:
            freq_range = []
            for spw in spw_array:
                if _spwids is None:
                    freqs = uvdata.freq_array
                else:
                    freqs = uvdata.freq_array[_spwids == spw]
                freq_range.append([freqs.min(), freqs.max()])
            freq_range = np.array(freq_range)

        # Add spw_array to kwargs to get passed through explicitly, since they are
        # not computable directly from flex_spw_id_array in this case
        kwargs["spw_array"] = spw_array

    # Figure out how to mesh the antenna parameters given with those in the uvd
    if "antenna_positions" not in kwargs:
        if "antenna_numbers" in kwargs or "antenna_names" in kwargs:
            # User can provide a subset of antenna numbers or names, but not positions
            antenna_numbers = kwargs.pop("antenna_numbers", None)
            antenna_names = kwargs.pop("antenna_names", None)

            if antenna_numbers is not None and antenna_names is not None:
                raise ValueError(
                    "Cannot specify both antenna_numbers and antenna_names but not "
                    "antenna_positions"
                )

            if antenna_numbers is not None:
                idx = [
                    i
                    for i, v in enumerate(uvdata.antenna_numbers)
                    if v in antenna_numbers
                ]
            elif antenna_names is not None:
                idx = [
                    i for i, v in enumerate(uvdata.antenna_names) if v in antenna_names
                ]

            antenna_numbers = np.asarray(uvdata.antenna_numbers)[idx]
            antenna_names = np.asarray(uvdata.antenna_names)[idx]
            antenna_positions = uvdata.antenna_positions[idx]
        else:
            antenna_numbers = uvdata.antenna_numbers
            antenna_names = uvdata.antenna_names
            antenna_positions = uvdata.antenna_positions

        # Sort them, because that's what the old function did
        sort_idx = np.argsort(antenna_numbers)
        antenna_numbers = np.asarray(antenna_numbers)[sort_idx]
        antenna_names = ((np.asarray(antenna_names))[sort_idx]).tolist()
        antenna_positions = antenna_positions[sort_idx]

    else:
        antenna_positions = kwargs.pop("antenna_positions")
        antenna_numbers = kwargs.pop("antenna_numbers", None)
        antenna_names = kwargs.pop("antenna_names", None)

    ant_array = kwargs.pop(
        "ant_array", np.union1d(uvdata.ant_1_array, uvdata.ant_2_array)
    )

    # Just in case a user inputs their own ant_array kwarg
    # make sure this is a numpy array for the following interactions
    if not isinstance(ant_array, np.ndarray):
        ant_array = np.asarray(ant_array)

    if antenna_numbers is not None:
        ant_array = np.intersect1d(
            ant_array, np.asarray(antenna_numbers, dtype=ant_array.dtype)
        )
    elif isinstance(antenna_positions, dict):
        ant_array = np.intersect1d(
            ant_array, np.asarray(list(antenna_positions.keys()), dtype=ant_array.dtype)
        )

    if jones_array is None:
        if np.all(uvdata.polarization_array < -4):
            if uvdata.Npols == 1 and uvdata.polarization_array[0] > -7:
                # single pol data, make a single pol cal object
                jones_array = uvdata.polarization_array
            else:
                jones_array = np.array([-5, -6])
        elif np.all(uvdata.polarization_array < 0):
            if uvdata.Npols == 1 and uvdata.polarization_array[0] > -3:
                # single pol data, make a single pol cal object
                jones_array = uvdata.polarization_array
            else:
                jones_array = np.array([-1, -2])
        else:
            raise ValueError(
                "Since uvdata object is in psuedo-stokes polarization, you must "
                "set jones_array."
            )

    if "history" not in kwargs:
        history = "Initialized from a UVData object with pyuvdata."
        if include_uvdata_history:
            history += " UVData history is: " + uvdata.history
    else:
        history = kwargs.pop("history")

    new = new_uvcal(
        cal_style=cal_style,
        gain_convention=gain_convention,
        jones_array=jones_array,
        x_orientation=kwargs.pop("x_orientation", uvdata.x_orientation),
        freq_array=freq_array,
        freq_range=freq_range,
        cal_type=cal_type,
        time_array=time_array,
        antenna_positions=antenna_positions,
        antenna_names=antenna_names,
        antenna_numbers=antenna_numbers,
        ant_array=ant_array,
        telescope_location=kwargs.pop(
            "telescope_location",
            EarthLocation(
                lat=uvdata.telescope_location_lat_lon_alt_degrees[0],
                lon=uvdata.telescope_location_lat_lon_alt_degrees[1],
                height=uvdata.telescope_location_lat_lon_alt_degrees[2],
            ),
        ),
        telescope_name=kwargs.pop("telescope_name", uvdata.telescope_name),
        integration_time=integration_time,
        channel_width=channel_width,
        flex_spw_id_array=flex_spw_id_array,
        history=history,
        **kwargs,
    )

    # Now convert the UVData object and UVCal object to proper array shapes
    if not uses_future_shapes:
        uvdata.use_current_array_shapes()
        if not wide_band:
            new.use_current_array_shapes()

    return new
