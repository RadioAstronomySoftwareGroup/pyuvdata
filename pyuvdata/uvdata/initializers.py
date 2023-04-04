"""A module defining functions for initializing UVData objects from scratch."""
from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from .. import __version__, utils

try:
    from lunarsky import MoonLocation

    hasmoon = True
    Locations = MoonLocation | EarthLocation
except ImportError:
    hasmoon = False
    Locations = EarthLocation


def new_uvdata(
    freq_array: np.ndarray,
    polarization_array: np.ndarray | list[str | int] | tuple[str | int],
    antenna_positions: np.ndarray | dict[str | int, np.ndarray],
    telescope_location: Locations,
    telescope_name: str,
    unique_times: np.ndarray | list[float] | tuple[float] | None = None,
    time_array: np.ndarray | None = None,
    unique_antpairs: Sequence[tuple[int, int]] | None = None,
    antpair_array: np.ndarray
    | list[tuple[int, int]]
    | tuple[tuple[int, int]]
    | None = None,
    integration_time: float | np.ndarray | None = None,
    channel_width: float | np.ndarray | None = None,
    antenna_names: list[str] | None = None,
    antenna_numbers: list[int] | None = None,
    blts_are_rectangular: bool | None = None,
    data_array: np.ndarray | None = None,
    flag_array: np.ndarray | None = None,
    nsample_array: np.ndarray | None = None,
    flex_spw: bool = False,
    history: str = "",
    instrument: str = "",
    vis_units: Literal["Jy", "K str", "uncalib"] = "uncalib",
    antname_format: str = "{0:03d}",
    empty: bool = False,
    time_axis_faster_than_bls: bool | None = None,
    phase_center_catalog: dict[str, Any] | None = None,
    phase_center_id_array: np.ndarray | None = None,
    **kwargs,
):
    """Initialize a new UVData object from keyword arguments.

    Parameters
    ----------
    freq_array : ndarray of float
        Array of frequencies in Hz.
    polarization_array : sequence of int or str
        Array of polarization integers or strings (eg. 'xx' or 'ee')
    antenna_positions : ndarray of float or dict of ndarray of float
        Array of antenna positions in ECEF coordinates in meters.
        If a dict, keys are antenna names or numbers and values are
        antenna positions in ECEF coordinates in meters.
    telescope_location : astropy EarthLocation or MoonLocation
        Location of the telescope.
    telescope_name : str
        Name of the telescope.
    unique_times : ndarray of float, optional
        Array of unique times in Julian Date. If provided, unique_antpairs must also
        be provided, and the blt axis of the UVData object will be formed from the
        outer product of unique_times and unique_antpairs. If not provided, time_array
        must be provided.
    time_array : ndarray of float, optional
        Array of times in Julian Date. If provided, antpair_array must also be
        provided. Required if unique_times is not provided. If time_array is provided,
        it will be passed on as-is to the UVData object.
    unique_antpairs : sequence of tuple of int, optional
        Sequence of unique antenna pairs in the data. If provided, unique_times must
        also be provided. If not, antpair_array must be provided. If provided, the blt
        axis of the UVData object will be formed from the outer product of unique_times
        and unique_antpairs.
    antpair_array : ndarray of int, optional
        Array of antenna pairs in the data. If provided, time_array must also be
        provided. Required if unique_antpairs is not provided. If antpair_array is
        provided, it will be converted to baseline_array in the same order as provided,
        and this is provided to the UVData object.
    integration_time : float or ndarray of float, optional
        Integration time in seconds. If not provided, it will be derived from the
        time_array, as the difference between successive times (with the last time
        appended). If a float is provided, it will be used for all integrations.
        If an ndarray is provided, it must have the same shape as time_array.
    channel_width : float or ndarray of float, optional
        Channel width in Hz. If not provided, it will be derived from the freq_array,
        as the difference between successive frequencies (with the last frequency
        appended). If a float is provided, it will be used for all channels.
        If an ndarray is provided, it must have the same shape as freq_array.
    antenna_names : list of str, optional
        List of antenna names. If not provided, antenna numbers will be used to form
        the antenna_names, according to the antname_format. antenna_names need not be
        provided if antenna_positions is a dict with string keys.
    antenna_numbers : list of int, optional
        List of antenna numbers. If not provided, antenna names will be used to form
        the antenna_numbers, but in this case the antenna_names must be strings that
        can be converted to integers. antenna_numbers need not be provided if
        antenna_positions is a dict with integer keys.
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
    flex_spw : bool, optional
        Set to True if the frequency axis is not contiguous. Default is False.
    history : str, optional
        History string to be added to the object. Default is a simple string
        containing the date and time and pyuvdata version.
    instrument : str, optional
        Instrument name. Default is the ``telescope_name``.
    vis_units : str, optional
        Visibility units. Default is 'uncalib'. Must be one of 'Jy', 'K str', or
        'uncalib'.
    antname_format : str, optional
        Format string for antenna names. Default is '{0:03d}'.
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
        zenith, assumed to be unprojected (i.e. drift scan).
    phase_center_id_array : ndarray of int, optional
        Array of phase center ids. If not provided, it will be initialized to the first
        id found in ``phase_center_catalog``. It must have shape ``(Nblts,)``.

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

    antenna_positions, antenna_names, antenna_numbers = get_antenna_params(
        antenna_positions, antenna_names, antenna_numbers, antname_format
    )

    baseline_array, unique_baselines = get_baseline_params(
        antenna_numbers, antpair_array, unique_antpairs
    )

    (
        nbls,
        ntimes,
        blts_are_rectangular,
        time_axis_faster_than_bls,
        time_array,
        baseline_array,
    ) = configure_blt_rectangularity(
        time_array=time_array,
        unique_times=unique_times,
        baseline_array=baseline_array,
        unique_baselines=unique_baselines,
        blts_are_rectangular=blts_are_rectangular,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
    )

    # Re-get the ant arrays because the baseline array may have changed
    ant_1_array, ant_2_array = utils.baseline_to_antnums(
        baseline_array, Nants_telescope=len(antenna_numbers)
    )

    lst_array, integration_time = get_time_params(
        telescope_location, time_array, integration_time
    )

    freq_array, channel_width = get_freq_params(freq_array, channel_width)

    polarization_array = np.array(polarization_array)
    if polarization_array.dtype.kind != "i":
        polarization_array = utils.polstr2num(polarization_array)

    if not instrument:
        instrument = telescope_name

    if vis_units not in ["Jy", "K str", "uncalib"]:
        raise ValueError("vis_units must be one of 'Jy', 'K str', or 'uncalib'.")

    phase_center_catalog, phase_center_id_array, app_ra, app_dec = get_phase_params(
        phase_center_catalog, phase_center_id_array, telescope_location, time_array
    )

    history += (
        f"Object created by new_uvdata() at {Time.now().iso} using "
        f"pyuvdata version {__version__}."
    )

    # Now set all the metadata
    obj.freq_array = freq_array
    obj.polarization_array = polarization_array
    obj.antenna_positions = antenna_positions
    obj.telescope_location = (
        [
            telescope_location.x.to_value("m"),
            telescope_location.y.to_value("m"),
            telescope_location.z.to_value("m"),
        ],
    )
    obj.telescope_name = telescope_name
    obj.baseline_array = baseline_array
    obj.ant_1_array = ant_1_array
    obj.ant_2_array = ant_2_array
    obj.time_array = time_array
    obj.lst_array = lst_array
    obj.integration_time = integration_time
    obj.channel_width = channel_width
    obj.antenna_names = antenna_names
    obj.antenna_numbers = antenna_numbers
    obj.flex_spw = flex_spw
    obj.history = history
    obj.instrument = instrument
    obj.vis_units = vis_units
    obj.Nants_data = len(set(np.concatenate([ant_1_array, ant_2_array])))
    obj.Nants_telescope = len(antenna_numbers)
    obj.Nbls = nbls
    obj.Nblts = len(baseline_array)
    obj.Nfreqs = len(freq_array)
    obj.Npols = len(polarization_array)
    obj.Ntimes = ntimes
    obj.Nspws = 1
    obj.spw_array = np.array([0])
    obj.future_array_shapes = True
    obj.phase_center_catalog = phase_center_catalog
    obj.phase_center_id_array = phase_center_id_array
    obj.phase_center_app_ra = app_ra
    obj.phase_center_app_dec = app_dec

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

    return obj


def get_antenna_params(
    antenna_positions: np.ndarray | dict[str | int, np.ndarray],
    antenna_names: list[str] | None = None,
    antenna_numbers: list[int] | None = None,
    antname_format: str = "{0:03d}",
) -> tuple[np.ndarray, list[str], list[int]]:
    """Configure antenna parameters for new UVData object."""
    # Get Antenna Parameters
    if isinstance(antenna_positions, dict):
        if isinstance(next(iter(antenna_positions.keys())), int):
            antenna_numbers = list(antenna_positions.keys())
        else:
            antenna_names = list(antenna_positions.keys())
        antenna_positions = np.array(list(antenna_positions.values()))

    if antenna_numbers is None and antenna_names is None:
        raise ValueError(
            "Either antenna_numbers or antenna_names must be provided unless "
            "antenna_positions is a dict."
        )

    if antenna_names is None:
        antenna_names = [antname_format.format(i) for i in antenna_numbers]
    elif antenna_numbers is None:
        try:
            antenna_numbers = [int(name) for name in antenna_names]
        except ValueError as e:
            raise ValueError(
                "Antenna names must be integers if antenna_numbers is not provided."
            ) from e

    if not isinstance(antenna_positions, np.ndarray):
        raise ValueError("antenna_positions must be a numpy array or a dictionary.")

    if antenna_positions.shape != (len(antenna_numbers), 3):
        raise ValueError(
            "antenna_positions must be a 2D array with shape (N_antennas, 3), "
            f"got {antenna_positions.shape}"
        )

    if len(antenna_names) != len(set(antenna_names)):
        raise ValueError("Duplicate antenna names found.")

    if len(antenna_numbers) != len(set(antenna_numbers)):
        raise ValueError("Duplicate antenna numbers found.")

    if len(antenna_numbers) != len(antenna_names):
        raise ValueError("antenna_numbers and antenna_names must have the same length.")

    return antenna_positions, antenna_names, antenna_numbers


def get_time_params(
    telescope_location: Locations,
    time_array: np.ndarray,
    integration_time: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Configure time parameters for new UVData object."""
    if not isinstance(time_array, np.ndarray):
        raise ValueError("time_array must be a numpy array.")

    t = Time(time_array, format="jd", scale="utc", location=telescope_location)
    lst_array = t.sidereal_time("apparent").radian

    if integration_time is None:
        integration_time = (t[1:] - t[:-1]).to_value("s")
        integration_time = np.concatenate([integration_time, integration_time[-1:]])
    elif np.isscalar(integration_time):
        integration_time = np.full_like(time_array, integration_time)

    if not isinstance(integration_time, np.ndarray):
        raise ValueError("integration_time must be a numpy array or a scalar.")

    if integration_time.shape != time_array.shape:
        raise ValueError("integration_time must be the same shape as time_array.")

    return lst_array, integration_time


def get_freq_params(
    freq_array: np.ndarray, channel_width: float | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Configure frequency parameters for new UVData object."""
    if not isinstance(freq_array, np.ndarray):
        raise ValueError("freq_array must be a numpy array.")

    if channel_width is None:
        channel_width = freq_array[1:] - freq_array[:-1]
        channel_width = np.concatenate([channel_width, channel_width[-1:]])
    elif np.isscalar(channel_width):
        channel_width = np.full_like(freq_array, channel_width)

    if not isinstance(channel_width, np.ndarray):
        raise ValueError("channel_width must be a numpy array or a scalar.")

    if channel_width.shape != freq_array.shape:
        raise ValueError("channel_width must be the same shape as freq_array.")

    return freq_array, channel_width


def get_baseline_params(
    antenna_positions: np.ndarray,
    antpairs: Sequence[tuple[int, int]] | None = None,
    unique_antpairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Configure baseline parameters for new UVData object."""
    if unique_antpairs is None and antpairs is None:
        raise ValueError("Either antpairs or unique_antpairs must be provided.")

    if unique_antpairs is None:
        ant_1_array, ant_2_array = zip(*antpairs)
        baseline_array = utils.antnums_to_baseline(
            ant_1_array, ant_2_array, len(antenna_positions)
        )
        unique_baselines = None
    else:
        ant_1_array, ant_2_array = zip(*unique_antpairs)
        unique_baselines = utils.antnums_to_baseline(
            ant_1_array, ant_2_array, len(antenna_positions)
        )
        baseline_array = None

    return baseline_array, unique_baselines


def configure_blt_rectangularity(
    time_array: np.ndarray | None = None,
    unique_times: np.ndarray | None = None,
    baseline_array: np.ndarray | None = None,
    unique_baselines: np.ndarray | None = None,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
):
    """Configure blt rectangularity parameters for new UVData object."""
    if unique_baselines is None and baseline_array is None:
        raise ValueError("Either baseline_array or unique_baselines must be provided.")

    if unique_times is None and time_array is None:
        raise ValueError("Either time_array or unique_times must be provided.")

    if unique_baselines is not None and baseline_array is not None:
        raise ValueError(
            "Only one of baseline_array or unique_baselines can be provided."
        )

    if unique_times is not None and time_array is not None:
        raise ValueError("Only one of time_array or unique_times can be provided.")

    if unique_times is not None and baseline_array is not None:
        raise ValueError(
            "If unique_times are provided, unique_baselines must be provided "
            "(not baseline_array)."
        )

    if unique_baselines is not None and time_array is not None:
        raise ValueError(
            "If unique_baselines are provided, unique_times must be provided as well "
            "(not time_array)."
        )

    if unique_times is not None and blts_are_rectangular is False:
        raise ValueError(
            "If unique_times are provided, blts_are_rectangular must be True."
        )

    if unique_times is not None:
        # We are rectangular
        blts_are_rectangular = True

        if time_axis_faster_than_bls is None:
            time_axis_faster_than_bls = True

        if time_axis_faster_than_bls:
            time_array = np.tile(unique_times, len(unique_baselines))
            baseline_array = np.repeat(unique_baselines, len(unique_times))
        else:
            baseline_array = np.tile(unique_baselines, len(unique_times))
            time_array = np.repeat(unique_times, len(unique_baselines))
    elif blts_are_rectangular:
        time_axis_faster_than_bls = np.abs(time_array[1] - time_array[0]) > 0

        unique_times = np.unique(time_array)

        if time_axis_faster_than_bls:
            unique_baselines = baseline_array[:: len(unique_times)]
        else:
            nbls = len(baseline_array) // len(unique_times)
            unique_baselines = baseline_array[:nbls]

    elif blts_are_rectangular is False:
        time_axis_faster_than_bls = None
        unique_baselines = np.unique(baseline_array)
        unique_times = np.unique(time_array)
    else:
        # We don't know if it's rectangular or not.
        # Let's try to figure it out.
        unique_baselines = np.unique(baseline_array)
        unique_times = np.unique(time_array)

        (
            blts_are_rectangular,
            time_axis_faster_than_bls,
        ) = utils.determine_rectangularity(
            time_array=time_array,
            baseline_array=baseline_array,
            nbls=len(unique_baselines),
            ntimes=len(unique_times),
        )

    return (
        len(unique_baselines),
        len(unique_times),
        blts_are_rectangular,
        time_axis_faster_than_bls,
        time_array,
        baseline_array,
    )


def get_phase_params(
    phase_center_catalog, phase_center_id_array, telescope_location, time_array
):
    """Configure phase center parameters for new UVData object."""
    if phase_center_catalog is None:
        phase_center_catalog = {
            0: {
                "cat_type": "unprojected",
                "cat_lon": 0.0,
                "cat_lat": np.pi / 2,
                "cat_name": "zenith",
                "cat_frame": "altaz",
            }
        }

    if phase_center_id_array is None:
        phase_center_id_array = np.full(
            len(time_array), next(iter(phase_center_catalog.keys())), dtype=int
        )

    app_ra = np.zeros_like(time_array)
    app_dec = np.zeros_like(time_array)
    for key, cat in phase_center_catalog.items():
        if cat["cat_frame"] == "altaz":
            these_times = time_array[phase_center_id_array == key]
            unique_times, indx = np.unique(these_times, return_inverse=True)

            t = Time(
                unique_times, format="jd", scale="utc", location=telescope_location
            )

            coords = SkyCoord(
                alt=np.ones(len(t)) * cat["cat_lat"] * un.rad,
                az=np.ones(len(t)) * cat["cat_lon"] * un.rad,
                frame="altaz",
                obstime=t,
                location=telescope_location,
            )
            ra = coords.icrs.ra.rad[indx]
            dec = coords.icrs.dec.rad[indx]

            app_ra[phase_center_id_array == key] = ra
            app_dec[phase_center_id_array == key] = dec
        elif cat["cat_frame"] == "icrs":
            app_ra[phase_center_id_array == key] = cat["cat_lon"]
            app_dec[phase_center_id_array == key] = cat["cat_lat"]
        else:
            raise NotImplementedError(f"Unrecognized cat_frame: {cat['cat_frame']}")

    return phase_center_catalog, phase_center_id_array, app_ra, app_dec
