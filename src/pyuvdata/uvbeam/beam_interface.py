# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Definition for BeamInterface object."""

from __future__ import annotations

import copy
import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .analytic_beam import AnalyticBeam
from .uvbeam import UVBeam

# Other methods we may want to include:
#  - beam area
#  - beam squared area
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

    def __init__(
        self,
        beam: AnalyticBeam | UVBeam,
        beam_type: Literal["efield", "power"] | None = None,
        include_cross_pols: bool = True,
    ):
        if not isinstance(beam, UVBeam) and not issubclass(type(beam), AnalyticBeam):
            raise ValueError(
                "beam must be a UVBeam or an AnalyticBeam instance, not a "
                f"{type(beam)}."
            )
        self.beam = beam
        if isinstance(beam, UVBeam):
            self._isuvbeam = True
            if beam_type is None or beam_type == beam.beam_type:
                self.beam_type = beam.beam_type
            elif beam_type == "power":
                warnings.warn(
                    "Input beam is an efield UVBeam but beam_type is specified as "
                    "'power'. Converting efield beam to power."
                )
                self.beam.efield_to_power(calc_cross_pols=include_cross_pols)
            else:
                raise ValueError(
                    "Input beam is a power UVBeam but beam_type is specified as "
                    "'efield'. It's not possible to convert a power beam to an "
                    "efield beam, either provide an efield UVBeam or do not "
                    "specify `beam_type`."
                )
        else:
            # AnalyticBeam
            self._isuvbeam = False
            self.beam_type = beam_type

    def __eq__(self, other: Any, silent: bool = False) -> bool:
        """
        Test if classes match and parameters are equal.

        Parameters
        ----------
        other : class
            Other class instance to check
        silent : bool
            Option to turn off printing explanations of why equality fails. Useful to
            prevent __ne__ from printing lots of messages.

        Returns
        -------
        bool
            True if the two instances are equivalent.

        """
        if isinstance(other, self.__class__):
            # First check that the beam is the same
            # If analytic, also check that the beam_type is the same
            if self.beam.__ne__(other.beam, silent=silent):
                if not silent:
                    print("Beams do not match.")
                return False
            if not self._isuvbeam and self.beam_type != other.beam_type:
                if not silent:
                    print(
                        "Beam types do not match. "
                        f"Left is {self.beam_type},"
                        f" right is {other.beam_type}."
                    )
                return False
        else:
            if not silent:
                print("Classes do not match")
            return False

        return True

    def __ne__(self, other, *, check_extra=True, silent=True):
        """
        Test if classes match and parameters are not equal.

        Parameters
        ----------
        other : class
            Other class instance to check
        silent : bool
            Option to turn off printing explanations of why equality fails. Useful to
            prevent __ne__ from printing lots of messages.

        Returns
        -------
        bool
            True if the two instances are equivalent.

        """
        return not self.__eq__(other, silent=silent)

    def compute_response(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float] | None,
        az_za_grid: bool = False,
        interpolation_function=None,
        freq_interp_kind=None,
        freq_interp_tol: float = 1.0,
        reuse_spline: bool = False,
        spline_opts: dict | None = None,
        check_azza_domain: bool = True,
    ):
        """
        Calculate beam responses, by interpolating UVBeams or evaluating AnalyticBeams.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to compute the response for in radians, either
            specifying the azimuth positions for every interpolation point or
            specifying the azimuth vector for a meshgrid if az_za_grid is True.
        za_array : array_like of floats, optional
            Zenith values to compute the response for in radians, either
            specifying the zenith positions for every interpolation point or
            specifying the zenith vector for a meshgrid if az_za_grid is True.
        freq_array : array_like of floats or None
            Frequency values to compute the response for in Hz. If beam is a UVBeam
            this can be set to None to get the responses at the UVBeam frequencies.
            It must be a numpy array if beam is an analytic beam.
        az_za_grid : bool
            Option to treat the `az_array` and `za_array` as the input vectors
            for points on a mesh grid.
        interpolation_function : str, optional
            Specify the interpolation function to use. Defaults to: "az_za_simple" for
            objects with the "az_za" pixel_coordinate_system and "healpix_simple" for
            objects with the "healpix" pixel_coordinate_system. Only applies if
            beam is a UVBeam.
        freq_interp_kind : str
            Interpolation method to use frequency. See scipy.interpolate.interp1d
            for details. Defaults to "cubic".
        freq_interp_tol : float
            Frequency distance tolerance [Hz] of nearest neighbors.
            If *all* elements in freq_array have nearest neighbor distances within
            the specified tolerance then return the beam at each nearest neighbor,
            otherwise interpolate the beam. Only applies if beam is a UVBeam.
        reuse_spline : bool
            Save the interpolation functions for reuse.  Only applies if beam is
            a UVBeam and interpolation_function is "az_za_simple".
        spline_opts : dict
            Provide options to numpy.RectBivariateSpline. This includes spline
            order parameters `kx` and `ky`, and smoothing parameter `s`. Only
            applies if beam is a UVBeam and interpolation_function is "az_za_simple".
        check_azza_domain : bool
            Whether to check the domain of az/za to ensure that they are covered by the
            intrinsic data array. Checking them can be quite computationally expensive.
            Conversely, if the passed az/za are outside of the domain, they will be
            silently extrapolated and the behavior is not well-defined. Only
            applies if beam is a UVBeam and interpolation_function is "az_za_simple".

        Returns
        -------
        array_like of float or complex
            An array of computed values, shape (Naxes_vec, Nfeeds or Npols,
            freq_array.size, az_array.size)
        """
        if not isinstance(az_array, np.ndarray) or az_array.ndim != 1:
            raise ValueError("az_array must be a one-dimensional numpy array")
        if not isinstance(za_array, np.ndarray) or za_array.ndim != 1:
            raise ValueError("za_array must be a one-dimensional numpy array")

        if self._isuvbeam:
            interp_data, _ = self.beam.interp(
                az_array=az_array,
                za_array=za_array,
                az_za_grid=az_za_grid,
                freq_array=freq_array,
                interpolation_function=interpolation_function,
                freq_interp_kind=freq_interp_kind,
                freq_interp_tol=freq_interp_tol,
                reuse_spline=reuse_spline,
                spline_opts=spline_opts,
                check_azza_domain=check_azza_domain,
            )
        else:
            if not isinstance(freq_array, np.ndarray) or freq_array.ndim != 1:
                raise ValueError("freq_array must be a one-dimensional numpy array")
            if az_za_grid:
                az_array_use, za_array_use = np.meshgrid(az_array, za_array)
                az_array_use = az_array_use.flatten()
                za_array_use = za_array_use.flatten()
            else:
                az_array_use = copy.copy(az_array)
                za_array_use = copy.copy(za_array)

            if self.beam_type == "efield":
                interp_data = self.beam.efield_eval(
                    az_array=az_array_use, za_array=za_array_use, freq_array=freq_array
                )
            else:
                interp_data = self.beam.power_eval(
                    az_array=az_array_use, za_array=za_array_use, freq_array=freq_array
                )

        return interp_data
