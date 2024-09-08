# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Analytic beam class definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from astropy.constants import c as speed_of_light
from scipy.special import j1

from .. import utils
from ..docstrings import combine_docstrings
from .uvbeam import UVBeam, _convert_feeds_to_pols

__all__ = ["AnalyticBeam", "AiryBeam", "GaussianBeam", "ShortDipoleBeam", "UniformBeam"]


@dataclass
class AnalyticBeam(ABC):
    """
    Define an analytic beam base class.

    Attributes
    ----------
    basis_vec_dict : dict
        Allowed basis vector types (keys) with their number of axes (which will be set
        on the object as Naxes_vec).
    basis_vector_type : str
        The type of basis vectors this beam is defined with. Must be a key in
        `basis_vec_dict`. Currently, only "az_za" is supported, which means basis
        vectors aligned with the zenith angle and azimuth.
    Naxes_vec : int
        Number of basis vectors. This is set based on the `basis_vector_type`.
    Nfeeds : int
        Number of feeds. This is set based on the `feed_array`.

    Parameters
    ----------
    feed_array : np.ndarray of str
        Feeds to define this beam for, e.g. x & y or n & e (for "north" and "east")
        or r & l.
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.
    x_orientation : str
        Physical orientation of the feed for the x feed. Not meaningful for
        unpolarized analytic beams, but clarifies the orientation of the dipole
        for the ShortDipoleBeam and matches with the meaning on UVBeam objects.

    """

    # In the future, this might allow for cartesian basis vectors in some orientation.
    # In that case, the Naxes_vec would be 3 rather than 2
    _basis_vec_dict = {"az_za": 2}

    @property
    @abstractmethod
    def basis_vector_type(self):
        """Require that a basis_vector_type is defined in concrete classes."""

    def __init__(
        self,
        *,
        feed_array: npt.NDArray[np.str] | None = None,
        include_cross_pols: bool = True,
        x_orientation: Literal["east", "north"] = "east",
    ):
        if self.basis_vector_type not in self._basis_vec_dict:
            raise ValueError(
                f"basis_vector_type is {self.basis_vector_type}, must be one of "
                f"{list(self._basis_vec_dict.keys())}"
            )
        self.Naxes_vec = self._basis_vec_dict[self.basis_vector_type]

        if feed_array is not None:
            for feed in feed_array:
                allowed_feeds = ["n", "e", "x", "y", "r", "l"]
                if feed not in allowed_feeds:
                    raise ValueError(f"Feeds must be one of: {allowed_feeds}")
        else:
            feed_array = ["x", "y"]
        self.feed_array = np.asarray(feed_array)

        self.Nfeeds = self.feed_array.size
        self.x_orientation = x_orientation

        self.polarization_array, _ = _convert_feeds_to_pols(
            self.feed_array, include_cross_pols, x_orientation=self.x_orientation
        )
        self.Npols = self.polarization_array.size

    def _check_eval_inputs(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ):
        """Check the inputs for the eval methods."""
        if az_array.ndim > 1 or za_array.ndim > 1 or freq_array.ndim > 1:
            raise ValueError(
                "az_array, za_array and freq_array must all be one dimensional."
            )

        if az_array.shape != za_array.shape:
            raise ValueError("az_array and za_array must have the same shape.")

    def _get_empty_data_array(
        self, npts: int, nfreqs: int, beam_type: str = "efield"
    ) -> npt.NDArray[np.float]:
        """Get the empty data to fill in the eval methods."""
        if beam_type == "efield":
            return np.zeros((self.Naxes_vec, self.Nfeeds, nfreqs, npts), dtype=complex)
        else:
            if self.Npols > self.Nfeeds:
                # crosspols are included
                dtype_use = complex
            else:
                dtype_use = float
            return np.zeros((1, self.Npols, nfreqs, npts), dtype=dtype_use)

    @abstractmethod
    def _efield_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """
        Evaluate the efield at the given coordinates.

        This is the method where the subclasses should implement the actual specific
        beam evaluation.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to evaluate the beam at in radians. Must be the same shape
            as za_array.
        za_array : array_like of floats, optional
            Zenith values to evaluate the beam at in radians. Must be the same shape
            as az_array.
        freq_array : array_like of floats, optional
            Frequency values to evaluate the beam at in Hertz.

        Returns
        -------
        array_like of complex
            An array of beam values. The shape of the evaluated data will be:
            (Naxes_vec, Nfeeds, freq_array.size, az_array.size)

        """

    def efield_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """
        Evaluate the efield at the given coordinates.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to evaluate the beam at in radians. Must be the same shape
            as za_array.
        za_array : array_like of floats, optional
            Zenith values to evaluate the beam at in radians. Must be the same shape
            as az_array.
        freq_array : array_like of floats, optional
            Frequency values to evaluate the beam at in Hertz.

        Returns
        -------
        array_like of complex
            An array of beam values. The shape of the evaluated data will be:
            (Naxes_vec, Nfeeds, freq_array.size, az_array.size)

        """
        self._check_eval_inputs(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        )

        return self._efield_eval(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        ).astype(complex)

    @abstractmethod
    def _power_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """
        Evaluate the power at the given coordinates.

        This is the method where the subclasses should implement the actual specific
        beam evaluation.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to evaluate the beam at in radians. Must be the same shape
            as za_array.
        za_array : array_like of floats, optional
            Zenith values to evaluate the beam at in radians. Must be the same shape
            as az_array.
        freq_array : array_like of floats, optional
            Frequency values to evaluate the beam at in Hertz.

        Returns
        -------
        array_like of float
            An array of beam values. The shape of the evaluated data will be:
            (1, Npols, freq_array.size, az_array.size)

        """

    def power_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """
        Evaluate the power at the given coordinates.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to evaluate the beam at in radians. Must be the same shape
            as za_array.
        za_array : array_like of floats, optional
            Zenith values to evaluate the beam at in radians. Must be the same shape
            as az_array.
        freq_array : array_like of floats, optional
            Frequency values to evaluate the beam at in Hertz.

        Returns
        -------
        array_like of float or complex
            An array of beam values. The shape of the evaluated data will be:
            (1, Npols, freq_array.size, az_array.size). The dtype will be
            a complex type if cross-pols are included, otherwise it will be a
            float type.

        """
        self._check_eval_inputs(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        )

        if self.Npols > self.Nfeeds:
            # cross pols are included
            expected_type = complex
        else:
            expected_type = float

        return self._power_eval(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        ).astype(expected_type)

    @combine_docstrings(UVBeam.new)
    def to_uvbeam(
        self,
        freq_array: npt.NDArray[np.float],
        beam_type: Literal["efield", "power"] = "efield",
        pixel_coordinate_system: (
            Literal["az_za", "orthoslant_zenith", "healpix"] | None
        ) = None,
        **kwargs,
    ):
        """Generate a UVBeam object from an AnalyticBeam object.

        This method evaluates the analytic beam at a set of locations to
        create a UVBeam object. This can be useful for testing and some other
        operations, but it is of course an approximation.

        Parameters
        ----------
        freq_array : ndarray of float
            Array of frequencies in Hz to evaluate the beam at.
        beam_type : str
            Beam type, either "efield" or "power".
        pixel_coordinate_system : str
            Pixel coordinate system, options are "az_za", "orthoslant_zenith" and
            "healpix". Forced to be "healpix" if ``nside`` is given and by
            *default* set to "az_za" if not. Currently, only "az_za" and "healpix"
            are implemented.

        """
        if beam_type not in ["efield", "power"]:
            raise ValueError("Beam type must be 'efield' or 'power'")

        if beam_type == "efield":
            feed_array = self.feed_array
            polarization_array = None
        else:
            feed_array = None
            polarization_array = self.polarization_array

        if pixel_coordinate_system is not None:
            allowed_coord_sys = list(UVBeam().coordinate_system_dict.keys())
            if pixel_coordinate_system not in allowed_coord_sys:
                raise ValueError(
                    f"Unknown coordinate system {pixel_coordinate_system}. UVBeam "
                    f"supported coordinate systems are: {allowed_coord_sys}."
                )

            if pixel_coordinate_system not in ["az_za", "healpix"]:
                raise NotImplementedError(
                    "Currently this method only supports 'az_za' and 'healpix' "
                    "pixel_coordinate_systems."
                )

        uvb = UVBeam.new(
            telescope_name="Analytic Beam",
            data_normalization="physical",
            feed_name=self.__repr__(),
            feed_version="1.0",
            model_name=self.__repr__(),
            model_version="1.0",
            freq_array=freq_array,
            feed_array=feed_array,
            polarization_array=polarization_array,
            x_orientation=self.x_orientation,
            **kwargs,
        )

        if uvb.pixel_coordinate_system == "healpix":
            try:
                from astropy_healpix import HEALPix
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "astropy_healpix is not installed but is "
                    "required for healpix functionality. "
                    "Install 'astropy-healpix' using conda or pip."
                ) from e
            hp_obj = HEALPix(nside=uvb.nside, order=uvb.ordering)
            hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(uvb.pixel_array)
            za_array, az_array = utils.coordinates.hpx_latlon_to_zenithangle_azimuth(
                hpx_lat.radian, hpx_lon.radian
            )

        else:
            az_array, za_array = np.meshgrid(uvb.axis1_array, uvb.axis2_array)
            az_array = az_array.flatten()
            za_array = za_array.flatten()

        if beam_type == "efield":
            eval_function = "efield_eval"
        else:
            eval_function = "power_eval"

        data_array = getattr(self, eval_function)(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        )

        if uvb.pixel_coordinate_system == "az_za":
            data_array = data_array.reshape(uvb.data_array.shape)

        uvb.data_array = data_array

        uvb.check()
        return uvb


def diameter_to_sigma(diameter: float, freq_array: npt.NDArray[np.float]) -> float:
    """
    Find the sigma that gives a beam width similar to an Airy disk.

    Find the stddev of a gaussian with fwhm equal to that of
    an Airy disk's main lobe for a given diameter.

    Parameters
    ----------
    diameter : float
        Antenna diameter in meters
    freq_array : array of float
        Frequencies in Hz

    Returns
    -------
    sigma : float
        The standard deviation in zenith angle radians for a Gaussian beam
        with FWHM equal to that of an Airy disk's main lobe for an aperture
        with the given diameter.

    """
    wavelengths = speed_of_light.to("m/s").value / freq_array

    scalar = 2.2150894  # Found by fitting a Gaussian to an Airy disk function

    sigma = np.arcsin(scalar * wavelengths / (np.pi * diameter)) * 2 / 2.355

    return sigma


@dataclass
class GaussianBeam(AnalyticBeam):
    """
    Define a Gaussian beam, optionally with frequency dependent size.

    Parameters
    ----------
    sigma : float
        Standard deviation in radians for the gaussian beam. Only one of sigma
        and diameter should be set.
    sigma_type : str
        Either "efield" or "power" to indicate whether the sigma specifies the size of
        the efield or power beam. Ignored if `sigma` is None.
    diameter : float
        Dish diameter in meters to use to define the size of the gaussian beam, by
        matching the FWHM of the gaussian to the FWHM of an Airy disk. This will result
        in a frequency dependent beam.  Only one of sigma and diameter should be set.
    spectral_index : float
        Option to scale the gaussian beam width as a power law with frequency. If set
        to anything other than zero, the beam will be frequency dependent and the
        `reference_frequency` must be set. Ignored if `sigma` is None.
    reference_frequency : float
        The reference frequency for the beam width power law, required if `sigma` is not
        None and `spectral_index` is not zero. Ignored if `sigma` is None.
    feed_array : np.ndarray of str
        Feeds to define this beam for, e.g. n & e or x & y or r & l.
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.

    """

    sigma: float | None = None
    sigma_type: Literal["efield", "power"] = "efield"
    diameter: float | None = None
    spectral_index: float = 0.0
    reference_frequency: float = None

    basis_vector_type = "az_za"

    def __init__(
        self,
        *,
        sigma: float | None = None,
        sigma_type: Literal["efield", "power"] = "efield",
        diameter: float | None = None,
        spectral_index: float = 0.0,
        reference_frequency: float = None,
        feed_array: npt.NDArray[np.str] | None = None,
        include_cross_pols: bool = True,
    ):
        if (diameter is None and sigma is None) or (
            diameter is not None and sigma is not None
        ):
            if diameter is None:
                raise ValueError("Either diameter or sigma must be set.")
            else:
                raise ValueError("Only one of diameter or sigma can be set.")

        self.diameter = diameter

        if sigma is not None:
            if sigma_type == "efield":
                self.sigma = sigma
            else:
                self.sigma = np.sqrt(2) * sigma

            if spectral_index != 0.0 and reference_frequency is None:
                raise ValueError(
                    "reference_frequency must be set if `spectral_index` is not zero."
                )
            if reference_frequency is None:
                reference_frequency = 1.0
            self.spectral_index = spectral_index
            self.reference_frequency = reference_frequency

        super().__init__(feed_array=feed_array, include_cross_pols=include_cross_pols)

    def get_sigmas(self, freq_array: npt.NDArray[np.float]) -> npt.NDArray[np.float]:
        """
        Get the sigmas for the gaussian beam using the diameter (if defined).

        Parameters
        ----------
        freq_array : array of floats
            Frequency values to get the sigmas for in Hertz.

        Returns
        -------
        sigmas : array_like of float
            Beam sigma values as a function of frequency. Size will match the
            freq_array size.

        """
        if self.diameter is not None:
            sigmas = diameter_to_sigma(self.diameter, freq_array)
        elif self.sigma is not None:
            sigmas = (
                self.sigma
                * (freq_array / self.reference_frequency) ** self.spectral_index
            )
        return sigmas

    def _efield_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the efield at the given coordinates."""
        sigmas = self.get_sigmas(freq_array)

        values = np.exp(
            -(za_array[np.newaxis, ...] ** 2) / (2 * sigmas[:, np.newaxis] ** 2)
        )
        data_array = self._get_empty_data_array(az_array.size, freq_array.size)
        # This is different than what is in pyuvsim because it means that we have the
        # same response to any polarized source in both feeds. We think this is correct
        # for an azimuthally symmetric analytic beam but it is a change.
        # On pyuvsim we essentially have each basis vector go into only one feed.
        for fn in np.arange(self.Nfeeds):
            data_array[0, fn, :, :] = values / np.sqrt(2.0)
            data_array[1, fn, :, :] = values / np.sqrt(2.0)

        return data_array

    def _power_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the power at the given coordinates."""
        sigmas = self.get_sigmas(freq_array)

        values = np.exp(
            -(za_array[np.newaxis, ...] ** 2) / (sigmas[:, np.newaxis] ** 2)
        )
        data_array = self._get_empty_data_array(
            az_array.size, freq_array.size, beam_type="power"
        )
        for fn in np.arange(self.Npols):
            # For power beams the first axis is shallow because we don't have to worry
            # about polarization.
            data_array[0, fn, :, :] = values

        return data_array


@dataclass
class AiryBeam(AnalyticBeam):
    """
    Define an Airy beam.

    Parameters
    ----------
    diameter : float
        Dish diameter in meters.
    feed_array : np.ndarray of str
        Feeds to define this beam for, e.g. n & e or x & y or r & l.
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.

    """

    diameter: float

    basis_vector_type = "az_za"

    def __init__(
        self,
        diameter: float,
        *,
        feed_array: npt.NDArray[np.str] | None = None,
        include_cross_pols: bool = True,
    ):
        super().__init__(feed_array=feed_array, include_cross_pols=include_cross_pols)

        self.diameter = diameter

    def _efield_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the efield at the given coordinates."""
        data_array = self._get_empty_data_array(az_array.size, freq_array.size)

        za_grid, f_grid = np.meshgrid(za_array, freq_array)
        kvals = (2.0 * np.pi) * f_grid / speed_of_light.to("m/s").value
        xvals = (self.diameter / 2.0) * np.sin(za_grid) * kvals
        values = np.zeros_like(xvals)
        nz = xvals != 0.0
        ze = xvals == 0.0
        values[nz] = 2.0 * j1(xvals[nz]) / xvals[nz]
        values[ze] = 1.0

        # This is different than what is in pyuvsim because it means that we have the
        # same response to any polarized source in both feeds. We think this is correct
        # for an azimuthally symmetric analytic beam but it is a change.
        # On pyuvsim we essentially have each basis vector go into only one feed.
        for fn in np.arange(self.Nfeeds):
            data_array[0, fn, :, :] = values / np.sqrt(2.0)
            data_array[1, fn, :, :] = values / np.sqrt(2.0)

        return data_array

    def _power_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the power at the given coordinates."""
        data_array = self._get_empty_data_array(
            az_array.size, freq_array.size, beam_type="power"
        )

        za_grid, f_grid = np.meshgrid(za_array, freq_array)
        kvals = (2.0 * np.pi) * f_grid / speed_of_light.to("m/s").value
        xvals = (self.diameter / 2.0) * np.sin(za_grid) * kvals
        values = np.zeros_like(xvals)
        nz = xvals != 0.0
        ze = xvals == 0.0
        values[nz] = (2.0 * j1(xvals[nz]) / xvals[nz]) ** 2
        values[ze] = 1.0

        for fn in np.arange(self.Npols):
            # For power beams the first axis is shallow because we don't have to worry
            # about polarization.
            data_array[0, fn, :, :] = values

        return data_array


@dataclass
class ShortDipoleBeam(AnalyticBeam):
    """
    Define an analytic short (Hertzian) dipole beam for two crossed dipoles.

    Parameters
    ----------
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne)
        for the power beam.
    x_orientation : str
        The orientation of the dipole labeled 'x'. The default ("east") means
        that the x dipole is aligned east-west and that the y dipole is aligned
        north-south.

    """

    x_orientation: Literal["east", "north"] = "east"

    basis_vector_type = "az_za"

    def __init__(
        self,
        *,
        x_orientation: Literal["east", "north"] = "east",
        include_cross_pols: bool = True,
    ):
        feed_array = ["e", "n"]

        super().__init__(
            feed_array=feed_array,
            include_cross_pols=include_cross_pols,
            x_orientation=x_orientation,
        )

    def _efield_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the efield at the given coordinates."""
        data_array = self._get_empty_data_array(az_array.size, freq_array.size)

        az_fgrid = np.repeat(az_array[np.newaxis], freq_array.size, axis=0)
        za_fgrid = np.repeat(za_array[np.newaxis], freq_array.size, axis=0)

        # The first dimension is for [azimuth, zenith angle] in that order
        # the second dimension is for feed [e, n] in that order
        data_array[0, 0] = -np.sin(az_fgrid)
        data_array[0, 1] = np.cos(az_fgrid)
        data_array[1, 0] = np.cos(za_fgrid) * np.cos(az_fgrid)
        data_array[1, 1] = np.cos(za_fgrid) * np.sin(az_fgrid)

        return data_array

    def _power_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the power at the given coordinates."""
        data_array = self._get_empty_data_array(
            az_array.size, freq_array.size, beam_type="power"
        )

        az_fgrid = np.repeat(az_array[np.newaxis], freq_array.size, axis=0)
        za_fgrid = np.repeat(za_array[np.newaxis], freq_array.size, axis=0)

        # these are just the sum in quadrature of the efield components.
        # some trig work is done to reduce the number of cos/sin evaluations
        data_array[0, 0] = 1 - (np.sin(za_fgrid) * np.cos(az_fgrid)) ** 2
        data_array[0, 1] = 1 - (np.sin(za_fgrid) * np.sin(az_fgrid)) ** 2

        if self.Npols > self.Nfeeds:
            # cross pols are included
            data_array[0, 2] = -(np.sin(za_fgrid) ** 2) * np.sin(2.0 * az_fgrid) / 2.0
            data_array[0, 3] = data_array[0, 2]

        return data_array


@dataclass
class UniformBeam(AnalyticBeam):
    """
    Define a uniform beam.

    Parameters
    ----------
    feed_array : np.ndarray of str
        Feeds to define this beam for, e.g. n & e or x & y or r & l.
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.

    """

    basis_vector_type = "az_za"

    def __init__(
        self,
        *,
        feed_array: npt.NDArray[np.str] | None = None,
        include_cross_pols: bool = True,
    ):
        super().__init__(feed_array=feed_array, include_cross_pols=include_cross_pols)

    def _efield_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the efield at the given coordinates."""
        data_array = self._get_empty_data_array(az_array.size, freq_array.size)

        # This is different than what is in pyuvsim because it means that we have the
        # same response to any polarized source in both feeds. We think this is correct
        # for an azimuthally symmetric analytic beam but it is a change.
        # On pyuvsim we essentially have each basis vector go into only one feed.
        data_array = data_array + 1.0 / np.sqrt(2.0)

        return data_array

    def _power_eval(
        self,
        *,
        az_array: npt.NDArray[np.float],
        za_array: npt.NDArray[np.float],
        freq_array: npt.NDArray[np.float],
    ) -> npt.NDArray[np.float]:
        """Evaluate the power at the given coordinates."""
        data_array = self._get_empty_data_array(
            az_array.size, freq_array.size, beam_type="power"
        )

        data_array = data_array + 1.0

        return data_array
