# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for plotting."""

import copy

import numpy as np

from .. import BeamInterface, UVBeam
from ..analytic_beam import AnalyticBeam, UnpolarizedAnalyticBeam
from .coordinates import _get_hpix_obj, hpx_latlon_to_zenithangle_azimuth
from .pol import polnum2str


def beam_plot(
    *,
    beam_obj: UVBeam | AnalyticBeam,
    freq: int | float,
    beam_type: str | None = None,
    complex_type: str = "real",
    logcolor: bool | None = None,
    plt_kwargs: dict | None = None,
    norm_kwargs: dict | None = None,
    max_zenith_deg: float = 90.0,
    savefile: str | None = None,
):
    """
    Make a pretty plot of a beam.

    Parameters
    ----------
    beam_obj : UVBeam or AnalyticBeam
        The beam to plot.
    freq : int or float
        Either the index into the freq_array for UVBeam objects (int) or the
        frequency to evaluate the beam at in Hz (float) for AnalyticBeam objects.
    beam_type : str
        Required for analytic beams to specify the beam type to plot. Ignored for
        UVBeams.
    complex_type : str
        What to plot for complex beams, options are: [real, imag, abs, phase].
        Defaults to "real" for complex beams. Ignored for real beams
        (i.e. power beams, same feed).
    logcolor : bool, optional
        Option to use log scaling for the color. Defaults to True for power
        beams and False for E-field beams. Results in using
        matplotlib.colors.LogNorm or matplotlib.colors.SymLogNorm if the data
        have negative values.
    plt_kwargs : dict, optional
        Keywords to be passed into the matplotlib.pyplot.imshow call.
    norm_kwargs : dict, optional
        Keywords to be passed into the norm object, typically vmin/vmax, plus
        linthresh for SymLogNorm.
    max_zenith_deg : float
        Maximum zenith angle to include in the plot in degrees. Default is
        90 to go down to the horizon.
    savefile : str
        File to save the plot to.

    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, SymLogNorm
    except ImportError as e:
        raise ImportError(
            "matplotlib is not installed but is required for plotting "
            "functionality. Install 'matplotlib' using conda or pip."
        ) from e

    allowed_complex_types = ["real", "imag", "abs", "phase"]
    if complex_type not in allowed_complex_types:
        raise ValueError(
            f"complex_type must be one of {allowed_complex_types}, but it "
            f"is {complex_type}."
        )

    complex_func = {"real": np.real, "imag": np.imag, "abs": np.abs, "phase": np.angle}

    if isinstance(beam_obj, UVBeam):
        beam_type = beam_obj.beam_type

    feed_labels = np.degrees(beam_obj.feed_angle).astype(str)
    feed_labels[np.isclose(beam_obj.feed_angle, 0)] = "N/S"
    feed_labels[np.isclose(beam_obj.feed_angle, np.pi / 2)] = "E/W"

    if beam_type == "efield":
        nfeedpol = beam_obj.Nfeeds
        feedpol_label = feed_labels
        if issubclass(type(beam_obj), UnpolarizedAnalyticBeam):
            feedpol_label = beam_obj.feed_array
        if logcolor is None:
            logcolor = False
    else:
        nfeedpol = beam_obj.Npols
        pol_strs = polnum2str(beam_obj.polarization_array)
        if np.max(beam_obj.polarization_array) <= -5 and not issubclass(
            type(beam_obj), UnpolarizedAnalyticBeam
        ):
            # linear pols, use feed angles.
            feedpol_label = [""] * nfeedpol
            for col_i, polstr in enumerate(pol_strs):
                feed0_ind = np.nonzero(beam_obj.feed_array == polstr[0])[0][0]
                feed1_ind = np.nonzero(beam_obj.feed_array == polstr[1])[0][0]
                if feed0_ind == feed1_ind:
                    feedpol_label[col_i] = feed_labels[feed0_ind]
                else:
                    feedpol_label[col_i] = "-".join(
                        [feed_labels[feed0_ind], feed_labels[feed1_ind]]
                    )
        else:
            feedpol_label = pol_strs
        if logcolor is None:
            logcolor = True

    if isinstance(beam_obj, UVBeam):
        naxes_vec = beam_obj.Naxes_vec
        name = beam_obj.telescope_name
        freq_title = beam_obj.freq_array[freq]

        reg_grid = True
        if beam_obj.pixel_coordinate_system == "healpix":
            HEALPix = _get_hpix_obj()

            hpx_obj = HEALPix(nside=beam_obj.nside, order=beam_obj.ordering)
            hpx_lon, hpx_lat = hpx_obj.healpix_to_lonlat(beam_obj.pixel_array)
            za_array, az_array = hpx_latlon_to_zenithangle_azimuth(
                hpx_lat.rad, hpx_lon.rad
            )
            pts_use = np.nonzero(za_array <= np.radians(max_zenith_deg))[0]
            za_array = za_array[pts_use]
            az_array = az_array[pts_use]
            reg_grid = False
        else:
            za_use = np.nonzero(beam_obj.axis2_array <= np.radians(max_zenith_deg))[0]
            az_array, za_array = np.meshgrid(
                beam_obj.axis1_array, beam_obj.axis2_array[za_use]
            )

        beam_vals = copy.deepcopy(beam_obj.data_array)[:, :, freq]
        if reg_grid:
            beam_vals = beam_vals[:, :, za_use, :]
        else:
            beam_vals = beam_vals[:, :, pts_use]
    elif issubclass(type(beam_obj), AnalyticBeam):
        name = beam_obj.__class__.__name__
        freq_title = freq

        naxes_vec = beam_obj.Naxes_vec
        if beam_type == "power":
            naxes_vec = 1
        reg_grid = True

        az_grid = np.deg2rad(np.arange(0, 360))
        za_grid = np.deg2rad(np.arange(0, 91)) * (max_zenith_deg / 90.0)
        az_array, za_array = np.meshgrid(az_grid, za_grid)
        bi = BeamInterface(beam_obj, beam_type=beam_type)
        beam_vals = bi.compute_response(
            az_array=az_array.flatten(),
            za_array=za_array.flatten(),
            freq_array=np.asarray([freq]),
        )
        if issubclass(type(beam_obj), UnpolarizedAnalyticBeam):
            beam_vals = beam_vals[0, :]
            naxes_vec = 1
            if nfeedpol == 1:
                feedpol_label = [""]
        beam_vals = beam_vals.reshape(naxes_vec, nfeedpol, za_grid.size, az_grid.size)
    si_prefix = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3}
    freq_str = f"{freq_title:.0f} Hz"
    for prefix, multiplier in si_prefix.items():
        if freq_title > multiplier:
            freq_str = f"{freq_title / multiplier:.0f} {prefix}Hz"
            break

    az_za_radial_val = np.sin(za_array)
    # get 4 radial ticks with values spaced sinusoidally (so ~linear in the plot),
    # rounded to the nearest 5 degrees
    radial_ticks_deg = (
        np.round(
            np.degrees(np.arcsin(np.linspace(0, np.sin(np.radians(max_zenith_deg)), 5)))
            / 5
        ).astype(int)
        * 5
    )[1:]

    if np.any(np.iscomplexobj(beam_vals)):
        beam_vals = complex_func[complex_type](beam_vals)
        type_label = ", " + complex_type
    else:
        type_label = ""

    norm_use = None
    colormap = "viridis"
    if norm_kwargs is None:
        norm_kwargs = {}
    if plt_kwargs is None:
        plt_kwargs = {}
    if logcolor:
        if np.min(beam_vals) < 0:
            min_pos_abs = np.min(np.abs(beam_vals)[np.abs(beam_vals) > 0])
            default_norm_kwargs = {
                "linthresh": min_pos_abs,
                "vmax": np.max(np.abs(beam_vals)),
                "vmin": -1 * np.max(np.abs(beam_vals)),
            }

            for key, value in default_norm_kwargs.items():
                if key not in norm_kwargs:
                    norm_kwargs[key] = value
            norm_use = SymLogNorm(**norm_kwargs)
            colormap = "PRGn"
        else:
            norm_use = LogNorm(**norm_kwargs)
    else:
        if len(norm_kwargs) > 0:
            for key in ["vmax", "vmin"]:
                if key in norm_kwargs:
                    plt_kwargs[key] = norm_kwargs[key]
        if np.min(beam_vals) < 0:
            colormap = "PRGn"
            default_norm_kwargs = {
                "vmax": np.max(np.abs(beam_vals)),
                "vmin": -1 * np.max(np.abs(beam_vals)),
            }
            for key, value in default_norm_kwargs.items():
                if key not in plt_kwargs:
                    plt_kwargs[key] = value

    if naxes_vec == 2:
        vec_label = ["azimuth", "zenith angle"]
    else:
        if beam_type == "power":
            vec_label = ["power"]
        else:
            vec_label = ["E-field"]

    nrow = naxes_vec
    ncol = nfeedpol
    if naxes_vec == 1 and nfeedpol == 4:
        nrow = 2
        ncol = 2
    fig_size = (5 * ncol, 5 * nrow)
    fig, ax = plt.subplots(
        nrow, ncol, subplot_kw={"projection": "polar"}, figsize=fig_size, squeeze=False
    )

    for row_i in range(nrow):
        for col_i in range(ncol):
            ax_use = ax[row_i, col_i]
            if nrow == naxes_vec and ncol == nfeedpol:
                vec_i = row_i
                fp_i = col_i
            else:
                vec_i = 0
                fp_i = row_i * 2 + col_i

            ax_use.grid(True)
            if reg_grid:
                pl = ax_use.pcolormesh(
                    az_array,
                    az_za_radial_val,
                    beam_vals[vec_i, fp_i],
                    cmap=colormap,
                    norm=norm_use,
                    **plt_kwargs,
                )
            else:
                pl = ax_use.scatter(
                    az_array,
                    az_za_radial_val,
                    c=beam_vals[vec_i, fp_i],
                    cmap=colormap,
                    norm=norm_use,
                    **plt_kwargs,
                )
            ax_use.set_rmax(np.max(az_za_radial_val))

            _ = ax_use.set_title(
                f"{feedpol_label[fp_i]} {name} {vec_label[vec_i]} "
                f"response ({freq_str}){type_label}",
                fontsize="medium",
            )
            _ = fig.colorbar(pl, ax=ax_use, shrink=0.5, pad=0.1)
            _ = ax_use.set_yticks(np.sin(np.deg2rad(radial_ticks_deg)))
            _ = ax_use.set_yticklabels(
                [f"{rt}" + r"$\degree$" for rt in radial_ticks_deg]
            )

    fig.tight_layout()

    if savefile is None:  # pragma: nocover
        plt.show()
    else:
        plt.savefig(savefile, bbox_inches="tight")
    plt.close()
