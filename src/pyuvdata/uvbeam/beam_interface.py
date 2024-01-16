# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Definition for BeamInterface object."""
import copy
import warnings

import numpy as np

from .analytic_beam import AnalyticBeam
from .uvbeam import UVBeam

# Other methods we may want to include:
#  - beam area
#  - beam squared area
#  - efield to power
#  - efield_to_pstokes


class BeamInterface:
    """
    Definition for a unified beam interface.

    This object provides a unified interface for UVBeam and AnalyticBeam objects
    to compute beam response values in any direction.

    Attributes
    ----------
    beam : pyuvdata.UVBeam or pyuvdata.AnalyticBeam
        Beam object to use for computations
    beam_type : str
        The beam type, either "efield" or "power".
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne).
        Used if beam is a UVBeam and and the input UVBeam is an Efield beam but
        beam_type is "power".
        Ignored otherwise (the cross pol inclusion is set by the beam object.)

    """

    def __init__(self, beam, beam_type=None, include_cross_pols=None):
        if not isinstance(beam, UVBeam) or isinstance(beam, AnalyticBeam):
            raise ValueError("beam must be a UVBeam or an AnalyticBeam instance.")
        self.beam = beam
        if isinstance(beam, UVBeam):
            self._isuvbeam = True
            if beam_type is None or beam_type == beam.beam_type:
                self.beam_type = beam.beam_type
            elif beam_type == "power":
                warnings.Warn(
                    "`beam` is an efield UVBeam but `beam_type` is specified as "
                    "'power'. Converting efield beam to power."
                )
                self.beam.efield_to_power(calc_cross_pols=include_cross_pols)
            else:
                raise ValueError(
                    "`beam` is a power UVBeam but `beam_type` is specified as 'efield'."
                    "It's not possible to convert a power beam to an efield beam, "
                    "either provide an efield UVBeam or do not specify `beam_type`."
                )
        else:
            # AnalyticBeam
            self._isuvbeam = False
            self.beam_type = beam_type

    def compute_response(
        self,
        az_array,
        za_array,
        freq_array,
        az_za_grid=False,
        freq_interp_tol=None,
        reuse_spline=False,
        spline_opts=None,
    ):
        """
        Calculate beam responses, by interpolating UVBeams or evaluating AnalyticBeams.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to interpolate to in radians, either specifying the
            azimuth positions for every interpolation point or specifying the
            azimuth vector for a meshgrid if az_za_grid is True.
        za_array : array_like of floats, optional
            Zenith values to interpolate to in radians, either specifying the
            zenith positions for every interpolation point or specifying the
            zenith vector for a meshgrid if az_za_grid is True.
        az_za_grid : bool
            Option to treat the `az_array` and `za_array` as the input vectors
            for points on a mesh grid.
        freq_array : array_like of floats, optional
            Frequency values to interpolate to.
        freq_interp_tol : float
            Frequency distance tolerance [Hz] of nearest neighbors.
            If *all* elements in freq_array have nearest neighbor distances within
            the specified tolerance then return the beam at each nearest neighbor,
            otherwise interpolate the beam.
        reuse_spline : bool
            Save the interpolation functions for reuse. Only applies for
            `az_za_simple` interpolation.
        spline_opts : dict
            Provide options to numpy.RectBivariateSpline. This includes spline
            order parameters `kx` and `ky`, and smoothing parameter `s`.
            Only applies for `az_za_simple` interpolation.

        Returns
        -------
        array_like of float or complex
            An array of computed values, shape (Naxes_vec, Nfeeds or Npols,
            freq_array.size, az_array.size)
        """
        if self._isuvbeam:
            interp_data, _ = self.beam.interp(
                az_array=az_array,
                za_array=za_array,
                az_za_grid=az_za_grid,
                freq_array=freq_array,
                freq_interp_tol=freq_interp_tol,
                reuse_spline=reuse_spline,
                spline_opts=spline_opts,
            )
        else:
            if az_za_grid:
                if az_array is None or za_array is None:
                    raise ValueError(
                        "If az_za_grid is set to True, az_array and za_array must be "
                        "provided."
                    )
                az_array_use, za_array_use = np.meshgrid(az_array, za_array)
                az_array_use = az_array_use.flatten()
                za_array_use = za_array_use.flatten()
            else:
                az_array_use = copy.copy(az_array)
                za_array_use = copy.copy(za_array)

            if self.beam_type == "efield":
                interp_data = self.efield_eval(az_array_use, za_array_use, freq_array)
            else:
                interp_data = self.power_eval(az_array_use, za_array_use, freq_array)

        return interp_data
