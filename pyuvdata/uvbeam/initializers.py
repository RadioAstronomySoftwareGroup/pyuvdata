# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""From-memory initializers for UVBeam objects."""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from astropy.time import Time

from .. import __version__, utils
from ..uvdata.initializers import XORIENTMAP


def new_uvbeam(
    *,
    telescope_name: str,
    data_normalization: Literal["physical", "peak", "solid_angle"],
    freq_array: npt.NDArray[np.float],
    feed_name: str = "default",
    feed_version: str = "0.0",
    model_name: str = "default",
    model_version: str = "0.0",
    feed_array: npt.NDArray[np.str] | None = None,
    polarization_array: (
        npt.NDArray[np.str | np.int] | list[str | int] | tuple[str | int] | None
    ) = None,
    x_orientation: Literal["east", "north", "e", "n", "ew", "ns"] | None = None,
    pixel_coordinate_system: (
        Literal["az_za", "orthoslant_zenith", "healpix"] | None
    ) = None,
    axis1_array: npt.NDArray[np.float] | None = None,
    axis2_array: npt.NDArray[np.float] | None = None,
    nside: int | None = None,
    ordering: Literal["ring", "nested"] | None = None,
    healpix_pixel_array: npt.NDArray[np.int] | None = None,
    basis_vector_array: npt.NDArray[np.float] | None = None,
    bandpass_array: npt.NDArray[np.float] | None = None,
    element_location_array: npt.NDArray[np.float] | None = None,
    element_coordinate_system: Literal["n-e", "x-y"] | None = None,
    delay_array: npt.NDArray[np.float] | None = None,
    gain_array: npt.NDArray[np.float] | None = None,
    coupling_matrix: npt.NDArray[np.float] | None = None,
    data_array: npt.NDArray[np.float] | None = None,
    history: str = "",
):
    r"""Create a new UVBeam object with default parameters.

    Parameters
    ----------
    telescope_name : str
        Telescope name.
    data_normalization : str
        Normalization standard of data_array, options are: "physical", "peak" or
        "solid_angle". Physical normalization means that the frequency dependence
        of the antenna sensitivity is included in the data_array while the
        frequency dependence of the receiving chain is included in the
        bandpass_array. Peak normalized means that for each frequency the
        data_array is separately normalized such that the peak is 1 (so the beam
        is dimensionless) and all direction-independent frequency dependence is
        moved to the bandpass_array (if the beam_type is "efield", then peak
        normalized means that the absolute value of the peak is 1). Solid angle
        normalized means the peak normalize beam is divided by the integral of
        the beam over the sphere, so the beam has dimensions of 1/steradian.
    freq_array : ndarray of float
        Array of frequencies in Hz.
    feed_name : str
        Name of the physical feed.
    feed_version : str
        Version of the physical feed.
    model_name : str
        Name of the beam model.
    model_version: str
        Version of the beam model.
    feed_array : ndarray of str
        Array of feed orientations. Options are: n/e or x/y or r/l. Must be
        provided for an E-field beam.
    polarization_array : ndarray of str or int
        Array of polarization integers or strings (eg. 'xx' or 'ee'). Must be
        provided for a power beam.
    x_orientation : str, optional
        Orientation of the x-axis. Options are 'east', 'north', 'e', 'n', 'ew', 'ns'.
    pixel_coordinate_system : str
        Pixel coordinate system, options are "az_za", "orthoslant_zenith" and "healpix".
        Forced to be "healpix" if ``nside`` is given and by *default* set to
        "az_za" if not.
    axis1_array : ndarray of float
        Coordinates along first pixel axis (e.g. azimuth for an azimuth/zenith
        angle coordinate system). Should not provided for healpix coordinates.
    axis2_array : ndarray of float
        Coordinates along second pixel axis (e.g. zenith angle for an azimuth/zenith
        angle coordinate system). Should not provided for healpix coordinates.
    nside : int
        Healpix nside parameter, should only be provided for healpix coordinates.
    healpix_pixel_array : ndarray of int
        Healpix pixels to include. If nside is provided, defaults to all the pixels
        in the Healpix map.
    ordering : str
        Healpix ordering parameter, defaults to "ring" if nside is provided.
    basis_vector_array : ndarray of float
        Beam basis vector components, essentially the mapping between the
        directions that the electrical field values are recorded in to the
        directions aligned with the pixel coordinate system (or azimuth/zenith
        angle for HEALPix beams). Defaults to unit vectors aligned with the
        pixel coordinate systems for E-field beams.
    bandpass_array : ndarray of float
        Frequency dependence of the beam. Depending on the ``data_normalization``
        this may contain only the frequency dependence of the receiving chain
        ("physical" normalization) or all the frequency dependence ("peak"
        normalization). Must be the same length as the ``freq_array``. Defaults
        to an array of all ones the length of the ``freq_array``.
    element_location_array : ndarray of float
        Array of phased array element locations in the element_coordinate_system.
        Must be a 2 dimensional array where the first dimension indexes the
        coordinate system (so should be length 2) and the second dimension indexes
        the phased array elements. Only used for phase array antennas.
    element_coordinate_system : str
        Coordinate system for describing the layout of a phased array feed.
        Options are: n-e or x-y. Defaults to "x-y" if ``element_location_array``
        is provided. Only used for phase array antennas.
    delay_array : ndarray of float
        Array of element delays in seconds. Defaults to an array of all zeros if
        ``element_location_array`` is provided. Only used for phase array antennas.
    gain_array : ndarray of float
        Array of element gains in dB. Defaults to an array of all ones if
        ``element_location_array`` is provided. Only used for phase array antennas.
    coupling_matrix : ndarray of float
        Matrix of complex element couplings in dB. Must be an ndarray of shape
        (Nelements, Nelements, Nfeeds, Nfeeds, Nfreqs). Defaults to an array
        with couplings of one for the same element and zero for other elements if
        ``element_location_array`` is provided. Only used for phase array antennas.
    data_array : ndarray of float, optional
        Either complex E-field values (if `feed_array` is given) or power values
        (if `polarization_array` is given) for the beam model. Units are
        normalized to either peak or solid angle as given by data_normalization.
        If None (the default), the data_array is initialized to an array of the
        appropriate shape and type containing all zeros.
    history : str, optional
        History string to be added to the object. Default is a simple string
        containing the date and time and pyuvdata version.
    \*\*kwargs
        All other keyword arguments are added to the object as attributes.

    Returns
    -------
    UVCal
        A new UVCal object with default parameters.

    """
    from .uvbeam import UVBeam

    uvb = UVBeam()

    if (feed_array is not None and polarization_array is not None) or (
        feed_array is None and polarization_array is None
    ):
        raise ValueError("Provide *either* feed_array *or* polarization_array")

    if feed_array is not None:
        uvb.beam_type = "efield"
        uvb.feed_array = np.asarray(feed_array)

        uvb.Nfeeds = uvb.feed_array.size
        uvb._set_efield()
    else:
        uvb.beam_type = "power"
        polarization_array = np.asarray(polarization_array)
        if polarization_array.dtype.kind != "i":
            polarization_array = np.asarray(utils.polstr2num(polarization_array))
        uvb.polarization_array = polarization_array

        uvb.Npols = uvb.polarization_array.size
        uvb._set_power()

    uvb._set_future_array_shapes()

    if (nside is not None) and (axis1_array is not None or axis2_array is not None):
        raise ValueError(
            "Provide *either* nside (and optionally healpix_pixel_array and "
            "ordering) *or* axis1_array and axis2_array."
        )

    if nside is not None or healpix_pixel_array is not None:
        if nside is None:
            raise ValueError("nside must be provided if healpix_pixel_array is given.")
        if healpix_pixel_array is None:
            healpix_pixel_array = np.arange(12 * nside**2, dtype=int)
        if ordering is None:
            ordering = "ring"

        uvb.nside = nside
        uvb.pixel_array = healpix_pixel_array
        uvb.ordering = ordering

        uvb.Npixels = healpix_pixel_array.size

        uvb.pixel_coordinate_system = "healpix"
        uvb.Naxes_vec = 2
        uvb.Ncomponents_vec = 2
    elif axis1_array is not None and axis2_array is not None:
        uvb.axis1_array = axis1_array
        uvb.axis2_array = axis2_array

        uvb.Naxes1 = axis1_array.size
        uvb.Naxes2 = axis2_array.size

        if pixel_coordinate_system is not None:
            allowed_pcs = list(uvb.coordinate_system_dict.keys())
            if uvb.pixel_coordinate_system not in allowed_pcs:
                raise ValueError(
                    f"pixel_coordinate_system must be one of {allowed_pcs}"
                )

        uvb.pixel_coordinate_system = pixel_coordinate_system or "az_za"
        uvb.Naxes_vec = 2
        uvb.Ncomponents_vec = 2
    else:
        raise ValueError(
            "Either nside or both axis1_array and axis2_array must be provided."
        )

    if uvb.beam_type == "power":
        uvb.Naxes_vec = 1

    uvb._set_cs_params()

    uvb.telescope_name = telescope_name
    uvb.feed_name = feed_name
    uvb.feed_version = feed_version
    uvb.model_name = model_name
    uvb.model_version = model_version

    uvb.data_normalization = data_normalization

    if not isinstance(freq_array, np.ndarray):
        raise ValueError("freq_array must be a numpy ndarray.")
    if freq_array.ndim != 1:
        raise ValueError("freq_array must be one dimensional.")
    uvb.freq_array = freq_array
    uvb.Nfreqs = freq_array.size

    if x_orientation is not None:
        uvb.x_orientation = XORIENTMAP[x_orientation.lower()]

    if basis_vector_array is not None:
        if uvb.pixel_coordinate_system == "healpix":
            bv_shape = (uvb.Naxes_vec, uvb.Ncomponents_vec, uvb.Npixels)
        else:
            bv_shape = (uvb.Naxes_vec, uvb.Ncomponents_vec, uvb.Naxes2, uvb.Naxes1)
        if basis_vector_array.shape != bv_shape:
            raise ValueError(
                f"basis_vector_array shape {basis_vector_array.shape} does not match "
                f"expected shape {bv_shape}."
            )
        uvb.basis_vector_array = basis_vector_array
    elif uvb.beam_type == "efield":
        if uvb.pixel_coordinate_system == "healpix":
            basis_vector_array = np.zeros(
                (uvb.Naxes_vec, uvb.Ncomponents_vec, uvb.Npixels), dtype=float
            )
            basis_vector_array[0, 0] = np.ones(uvb.Npixels, dtype=float)
            basis_vector_array[1, 1] = np.ones(uvb.Npixels, dtype=float)
        else:
            basis_vector_array = np.zeros(
                (uvb.Naxes_vec, uvb.Ncomponents_vec, uvb.Naxes2, uvb.Naxes1),
                dtype=float,
            )
            basis_vector_array[0, 0] = np.ones((uvb.Naxes2, uvb.Naxes1), dtype=float)
            basis_vector_array[1, 1] = np.ones((uvb.Naxes2, uvb.Naxes1), dtype=float)
        uvb.basis_vector_array = basis_vector_array

    if bandpass_array is not None:
        if bandpass_array.shape != freq_array.shape:
            raise ValueError(
                "The bandpass array must have the same shape as the freq_array."
            )
        uvb.bandpass_array = bandpass_array
    else:
        uvb.bandpass_array = np.ones_like(uvb.freq_array)

    if element_location_array is not None:
        if feed_array is None:
            raise ValueError(
                "feed_array must be provided if element_location_array is given."
            )

        if element_location_array.ndim != 2:
            raise ValueError("element_location_array must be 2 dimensional")
        shape = element_location_array.shape
        if shape[0] != 2:
            raise ValueError(
                "The first dimension of element_location_array must be length 2"
            )
        if shape[1] <= 1:
            raise ValueError(
                "The second dimension of element_location_array must be >= 2."
            )

        if element_coordinate_system is None:
            element_coordinate_system = "x-y"
        if delay_array is None:
            delay_array = np.zeros(shape[1])
        else:
            if delay_array.shape != (shape[1],):
                raise ValueError(
                    "delay_array must be one dimensional with length "
                    "equal to the second dimension of element_location_array"
                )
        if gain_array is None:
            gain_array = np.ones(shape[1])
        else:
            if gain_array.shape != (shape[1],):
                raise ValueError(
                    "gain_array must be one dimensional with length "
                    "equal to the second dimension of element_location_array"
                )
        coupling_shape = (shape[1], shape[1], uvb.Nfeeds, uvb.Nfeeds, uvb.Nfreqs)
        if coupling_matrix is None:
            coupling_matrix = np.zeros(coupling_shape, dtype=complex)
            for element in range(shape[1]):
                coupling_matrix[element, element] = np.ones(
                    (uvb.Nfeeds, uvb.Nfeeds, uvb.Nfreqs), dtype=complex
                )
        else:
            if coupling_matrix.shape != coupling_shape:
                raise ValueError(
                    f"coupling_matrix shape {coupling_matrix.shape} does not "
                    f"match expected shape {coupling_shape}."
                )

        uvb.antenna_type = "phased_array"
        uvb._set_phased_array()
        uvb.element_coordinate_system = element_coordinate_system
        uvb.element_location_array = element_location_array
        uvb.Nelements = shape[1]
        uvb.delay_array = delay_array
        uvb.gain_array = gain_array
        uvb.coupling_matrix = coupling_matrix
    else:
        uvb.antenna_type = "simple"
        uvb._set_simple()

    # Set data parameters
    if uvb.beam_type == "efield":
        data_type = complex
        polax = uvb.Nfeeds
    else:
        data_type = float
        polax = uvb.Npols

    if uvb.pixel_coordinate_system == "healpix":
        pixax = (uvb.Npixels,)
    else:
        pixax = (uvb.Naxes2, uvb.Naxes1)

    data_shape = (uvb.Naxes_vec, polax, uvb.Nfreqs) + pixax

    if data_array is not None:
        if not isinstance(data_array, np.ndarray):
            raise ValueError("data_array must be a numpy ndarray")
        if data_array.shape != data_shape:
            raise ValueError(
                f"Data array shape {data_array.shape} does not match "
                f"expected shape {data_shape}."
            )
        uvb.data_array = data_array
    else:
        uvb.data_array = np.zeros(data_shape, dtype=data_type)

    history += (
        f"Object created by new_uvbeam() at {Time.now().iso} using "
        f"pyuvdata version {__version__}."
    )
    uvb.history = history

    uvb.check()
    return uvb
