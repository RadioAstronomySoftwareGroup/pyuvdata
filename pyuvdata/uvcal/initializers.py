"""From-memory initializers for UVCal objects."""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from astropy.time import Time

from .. import __version__
from ..uvdata.initializers import (
    Locations,
    get_antenna_params,
    get_freq_params,
    get_time_params,
)


def new_uvcal(
    freq_array: np.ndarray,
    time_array: np.ndarray,
    antenna_positions: np.ndarray | dict[str | int, np.ndarray],
    telescope_location: Locations,
    telescope_name: str,
    cal_style: Literal["sky", "redundant"],
    gain_convention: Literal["divide", "multiply"],
    x_orientation: Literal["east", "north", "e", "n", "ew", "ns"],
    jones_array: np.ndarray | str,
    delay_array: np.ndarray | None = None,
    spw_array: Sequence[int] | np.ndarray = (0,),
    cal_type: Literal["delay", "gain", "unknown"] | None = None,
    integration_time: float | np.ndarray | None = None,
    channel_width: float | np.ndarray | None = None,
    antenna_names: list[str] | None = None,
    antenna_numbers: list[int] | None = None,
    antname_format: str = "{0:03d}",
    ant_array: np.ndarray | None = None,
    wide_band: bool = False,
    flex_spw_id_array: np.ndarray | None = None,
    ref_antenna_name: str | None = None,
    sky_catalog: str | None = None,
    sky_field: str | None = None,
    empty: bool = False,
    history: str = "",
    **kwargs,
):
    """Create a new UVCal object with default parameters.

    Returns
    -------
    UVCal
        A new UVCal object with default parameters.

    """
    from .uvcal import UVCal

    uvc = UVCal()

    if wide_band:
        raise NotImplementedError(
            "Wideband calibrations are not yet supported for this constructor."
        )
    if not kwargs.get("flex_spw", True):
        raise ValueError("flex_spw must be True for this constructor.")

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
        telescope_location, time_array, integration_time
    )

    freq_array, channel_width = get_freq_params(freq_array, channel_width)

    spw_array = np.array(spw_array)
    if len(spw_array) != 1 and flex_spw_id_array is None:
        raise ValueError(
            "If spw_array is not length 1, flex_spw_id_array must be provided."
        )
    elif len(spw_array) == 1:
        flex_spw_id_array = np.zeros_like(freq_array, dtype=int)

    if jones_array == "linear":
        jones_array = np.array([-5, -6, -7, -8])
    elif jones_array == "circular":
        jones_array = np.array([-1, -2, -3, -4])
    else:
        jones_array = np.array(jones_array)

    history += (
        f"Object created by new_uvcal() at {Time.now().iso} using "
        f"pyuvdata version {__version__}."
    )

    XORIENTMAP = {
        "east": "east",
        "north": "north",
        "e": "east",
        "n": "north",
        "ew": "east",
        "ns": "north",
    }
    x_orientation = XORIENTMAP[x_orientation.lower()]

    if delay_array is not None and cal_type is None:
        cal_type = "delay"
    elif cal_type == "delay" and delay_array is None:
        raise ValueError("If cal_type is delay, delay_array must be provided.")

    if cal_style == "sky":
        if ref_antenna_name is None:
            raise ValueError("If cal_style is sky, ref_antenna_name must be provided.")
        if sky_catalog is None:
            raise ValueError("If cal_style is sky, sky_catalog must be provided.")
        if sky_field is None:
            raise ValueError("If cal_style is sky, sky_field must be provided.")

    # Now set all the metadata
    uvc.freq_array = freq_array
    uvc.antenna_positions = antenna_positions
    uvc.telescope_location = [
        telescope_location.x.to_value("m"),
        telescope_location.y.to_value("m"),
        telescope_location.z.to_value("m"),
    ]
    uvc.telescope_name = telescope_name
    uvc.time_array = time_array
    uvc.lst_array = lst_array
    uvc.integration_time = integration_time
    uvc.channel_width = channel_width
    uvc.antenna_names = antenna_names
    uvc.antenna_numbers = antenna_numbers
    uvc.flex_spw = True
    uvc.history = history
    uvc.ant_array = ant_array
    uvc.telescope_name = telescope_name
    uvc.cal_style = cal_style
    uvc.cal_type = cal_type
    uvc.gain_convention = gain_convention
    uvc.x_orientation = x_orientation
    uvc.ref_antenna_name = ref_antenna_name
    uvc.sky_catalog = sky_catalog
    uvc.sky_field = sky_field
    uvc.flex_spw_id_array = flex_spw_id_array
    uvc.jones_array = jones_array

    if cal_type == "delay":
        uvc._set_delay()
    elif cal_type == "gain":
        uvc._set_gain()
    elif cal_type == "unknown":
        uvc._set_unknown_cal_type()

    if cal_style == "redundant":
        uvc._set_redundant()
    elif cal_style == "sky":
        uvc._set_sky()

    print(uvc.cal_type)

    uvc.Nants_data = len(ant_array)
    uvc.Nants_telescope = len(antenna_numbers)
    uvc.Nfreqs = len(freq_array)
    uvc.Ntimes = len(time_array)
    uvc.Nspws = 1
    uvc.Njones = len(jones_array)

    uvc.spw_array = np.array([0])

    # We always make the following true.
    uvc._set_future_array_shapes()
    uvc._set_flex_spw()

    for k, v in kwargs.items():
        if hasattr(uvc, k):
            setattr(uvc, k, v)
        else:
            raise ValueError(f"Unrecognized keyword argument: {k}")

    if empty:
        fshape = (uvc.Nants_data, uvc.Nfreqs, uvc.Ntimes, uvc.Njones)
        sshape = (uvc.Nants_data, uvc.Nspws, uvc.Ntimes, uvc.Njones)

        # Flag array
        uvc.flag_array = np.zeros(fshape, dtype=bool)
        uvc.gain_array = np.ones(fshape, dtype=complex)
        uvc.input_flag_array = np.zeros(fshape, dtype=bool)
        if cal_type == "delay":
            uvc.quality_array = np.zeros(sshape, dtype=float)
            uvc.total_quality_array = np.zeros(sshape[1:], dtype=float)
        else:
            uvc.quality_array = np.zeros(fshape, dtype=float)
            uvc.total_quality_array = np.zeros(fshape[1:], dtype=float)

    uvc.check()
    return uvc
