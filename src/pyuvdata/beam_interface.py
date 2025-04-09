# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Definition for BeamInterface object."""

from __future__ import annotations

import copy
import warnings
from dataclasses import InitVar, asdict, dataclass, replace
from itertools import product
from typing import Literal

import numpy as np
import numpy.typing as npt

from .analytic_beam import AnalyticBeam
from .utils import pol as upol
from .uvbeam import UVBeam

# Other methods we may want to include:
#  - beam area
#  - beam squared area
#  - efield_to_pstokes


@dataclass
class BeamInterface:
    """
    Definition for a unified beam interface.

    This object provides a unified interface for UVBeam and AnalyticBeam objects
    to compute beam response values in any direction.

    Attributes
    ----------
    beam : pyuvdata.UVBeam or pyuvdata.AnalyticBeam or BeamInterface
        Beam object to use for computations. If a BeamInterface is passed, a new
        view of the same object is created.
    beam_type : str
        The beam type, either "efield" or "power".
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne).
        Used if beam is a UVBeam and and the input UVBeam is an Efield beam but
        beam_type is "power".
        Ignored otherwise (the cross pol inclusion is set by the beam object.)

    """

    beam: AnalyticBeam | UVBeam
    beam_type: Literal["efield", "power"] | None = None
    include_cross_pols: InitVar[bool] = True

    def __post_init__(self, include_cross_pols: bool):
        """
        Post-initialization validation and conversions.

        Parameters
        ----------
        include_cross_pols : bool
            Option to include the cross polarized beams (e.g. xy and yx or en and ne)
            for the power beam.

        """
        if isinstance(self.beam, BeamInterface):
            self.beam = self.beam.beam
            self.__post_init__(include_cross_pols=include_cross_pols)
            return

        if not isinstance(self.beam, UVBeam) and not issubclass(
            type(self.beam), AnalyticBeam
        ):
            raise ValueError(
                "beam must be a UVBeam or an AnalyticBeam instance, not a "
                f"{type(self.beam)}."
            )
        if isinstance(self.beam, UVBeam):
            if self.beam_type is None or self.beam_type == self.beam.beam_type:
                self.beam_type = self.beam.beam_type
            elif self.beam_type == "power":
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
        elif self.beam_type is None:
            self.beam_type = "efield"

    @property
    def Npols(self):  # noqa N802
        """The number of polarizations in the beam."""
        return self.beam.Npols or len(self.polarization_array)

    @property
    def polarization_array(self):
        """The polarizations defined on the beam."""
        return self.beam.polarization_array

    @property
    def feed_array(self):
        """The feeds for which the beam is defined."""
        return self.beam.feed_array

    @property
    def feed_angle(self):
        """The feeds for which the beam is defined."""
        return self.beam.feed_angle

    @property
    def Nfeeds(self):  # noqa N802
        """The number of feeds defined on the beam."""
        return self.beam.Nfeeds or len(self.feed_array)

    def clone(self, **kw):
        """Return a new instance with updated parameters."""
        return replace(self, **kw)

    def as_power_beam(
        self, include_cross_pols: bool | None = None, allow_beam_mutation: bool = False
    ):
        """Return a new interface instance that is in the power-beam mode.

        If already in the power-beam mode, this is a no-op. Note that this might be
        slighty unexpected, because the effect of `include_cross_pols` is not accounted
        for in this case.

        Parameters
        ----------
        include_cross_pols : bool, optional
            Whether to include cross-pols in the power beam. By default, this is True
            for E-field beams, and takes the same value as the existing beam if the
            existing beam is already a power beam.
        allow_beam_mutation : bool, optional
            Whether to allow the underlying beam to be updated in-place.
        """
        if self.beam_type == "power":
            if include_cross_pols is None:
                # By default, keep the value of include_cross_pols the same.
                include_cross_pols = self.Npols > 2

            if self.Npols > 1 and (
                (include_cross_pols and self.Npols != 4)
                or (not include_cross_pols and self.Npols != 2)
            ):
                warnings.warn(
                    "as_power_beam does not modify cross pols when the beam is"
                    f"already in power mode! You have polarizations: "
                    f"{self.polarization_array} but asked to "
                    f"*{'include' if include_cross_pols else 'not include'}* "
                    "cross-pols."
                )
            return self

        if include_cross_pols is None:
            include_cross_pols = True

        beam = self.beam if allow_beam_mutation else copy.deepcopy(self.beam)

        # We cannot simply use .clone() here, because we need to be able to pass
        # include_cross_pols, which can only be passed to the constructor proper.
        this = asdict(self)
        this["beam"] = beam
        this["beam_type"] = "power"
        this["include_cross_pols"] = include_cross_pols
        with warnings.catch_warnings():
            # Don't emit the warning that we're converting to power, because that is
            # explicitly desired.
            warnings.simplefilter("ignore", UserWarning)
            return BeamInterface(**this)

    def with_feeds(self, feeds, *, maintain_ordering: bool = True):
        """Return a new interface instance with updated feed_array.

        Parameters
        ----------
        feeds : array_like of str
            The feeds to keep in the beam. Each value should be a string, e.g. 'n', 'x'.
        maintain_ordering : bool, optional
            If True, maintain the same polarization ordering as in the beam currently.
            If False, change ordering to match the input feeds, which are turned into
            pols (if a power beam) by using product(feeds, feeds).
        """
        if not self._isuvbeam:
            if maintain_ordering:
                feeds = [fd for fd in self.feed_array if fd in feeds]
            return self.clone(beam=self.beam.clone(feed_array=feeds))
        if self.beam_type == "power":
            # Down-select polarizations based on the feeds input.
            possible_pols = [f1 + f2 for f1, f2 in product(feeds, feeds)]
            possible_pol_ints = upol.polstr2num(
                possible_pols, x_orientation=self.beam.get_x_orientation_from_feeds()
            )
            if maintain_ordering:
                use_pols = [
                    p for p in self.beam.polarization_array if p in possible_pol_ints
                ]
            else:
                use_pols = [
                    p for p in possible_pol_ints if p in self.beam.polarization_array
                ]

            new_beam = self.beam.select(polarizations=use_pols, inplace=False)
        else:
            if maintain_ordering:
                feeds = [fd for fd in self.feed_array if fd in feeds]

            new_beam = self.beam.select(feeds=feeds, inplace=False)
        return self.clone(beam=new_beam)

    @property
    def _isuvbeam(self):
        return isinstance(self.beam, UVBeam)

    def compute_response(
        self,
        *,
        az_array: npt.NDArray[float],
        za_array: npt.NDArray[float],
        freq_array: npt.NDArray[float] | None,
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
            applies if beam is a UVBeam and interpolation_function is "az_za_simple"
            or "az_za_map_coordinates".
        check_azza_domain : bool
            Whether to check the domain of az/za to ensure that they are covered by the
            intrinsic data array. Checking them can be quite computationally expensive.
            Conversely, if the passed az/za are outside of the domain, they will be
            silently extrapolated and the behavior is not well-defined. Only
            applies if beam is a UVBeam and interpolation_function is "az_za_simple"
            or "az_za_map_coordinates".

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
                return_basis_vector=False,
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
