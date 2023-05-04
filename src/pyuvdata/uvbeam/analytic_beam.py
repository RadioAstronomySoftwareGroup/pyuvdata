# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Analytic beam class definitions."""
from abc import ABC, abstractmethod

import numpy as np
from astropy.constants import c as speed_of_light

from .uvbeam import _convert_feeds_to_pols

__all__ = ["AnalyticBeam", "GaussianBeam"]


# TODO: write the `update_cross_pols` method.
# TODO: write a `to_uvbeam` method to allow for interpolation testing/exploration


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
        Feeds to define this beam for, e.g. N & E or x & y or R & L.
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or EN and NE) for
        the power beam.

    """

    # In the future, this might allow for cartesian basis vectors in some orientation.
    # In that case, the Naxes_vec would be 3 rather than 2
    basis_vec_dict = {"az_za": 2}

    basis_vector_type = None

    def __init__(self, feed_array=None, include_cross_pols=True):
        if self.basis_vector_type is not None:
            if self.basis_vector_type not in self.basis_vec_dict:
                raise ValueError(
                    f"basis_vector_type is {self.basis_vector_type}, must be one of "
                    f"{self.basis_vec_dict.keys()}"
                )
            self.Naxes_vec = self.basis_vec_dict[self.basis_vector_type]

        if feed_array is not None:
            # TODO: think about rotated dipoles
            for feed in feed_array:
                if feed not in ["N", "E", "x", "y", "R", "L"]:
                    raise ValueError
        else:
            feed_array = ["x", "y"]
        self.feed_array = feed_array

        if self.feed_array is not None:
            self.Nfeeds = self.feed_array.size
        else:
            self.Nfeeds = 1

        # TODO: Add x_orientation here if needed.
        self.polarization_array, _ = _convert_feeds_to_pols(
            self.feed_array, include_cross_pols
        )
        self.Npols = self.polarization_array.size

        # TODO: think about where x_orientation should live, here or on interface class

    def _check_eval_inputs(self, az_array, za_array, freq_array):
        """Check the inputs for the eval methods."""
        if az_array.ndim > 1 or za_array.ndim > 1 or freq_array.ndim > 1:
            raise ValueError(
                "az_array, za_array and freq_array must all be one dimensional."
            )

        if az_array.shape != za_array.shape:
            raise ValueError("az_array and za_array must have the same shape.")

    def _get_empty_data_array(self, az_array, za_array, freq_array, beam_type="efield"):
        """Get the empty data to fill in the eval methods."""
        if beam_type == "efield":
            feed_pol_size = self.Nfeeds
        else:
            feed_pol_size = self.Npols
        return np.zeros(
            (self.Naxes_vec, feed_pol_size, freq_array.size, az_array.size), dtype=float
        )

    @abstractmethod
    def _efield_eval(self, az_array, za_array, freq_array):
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

    def efield_eval(self, az_array, za_array, freq_array):
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
        self._check_eval_inputs(az_array, za_array, freq_array)

        return self._efield_eval(az_array, za_array, freq_array)

    @abstractmethod
    def _power_eval(self, az_array, za_array, freq_array):
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

    @abstractmethod
    def power_eval(self, az_array, za_array, freq_array):
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
        array_like of float
            An array of beam values. The shape of the evaluated data will be:
            (1, Npols, freq_array.size, az_array.size)

        """
        self._check_eval_inputs(az_array, za_array, freq_array)

        return self._power_eval(az_array, za_array, freq_array)


def diameter_to_sigma(diam, freqs):
    """
    Find the sigma that gives a beam width similar to an Airy disk.

    Find the stddev of a gaussian with fwhm equal to that of
    an Airy disk's main lobe for a given diameter.

    Parameters
    ----------
    diam : float
        Antenna diameter in meters
    freqs : array
        Frequencies in Hz

    Returns
    -------
    sigma : float
        The standard deviation in zenith angle radians for a Gaussian beam
        with FWHM equal to that of an Airy disk's main lobe for an aperture
        with the given diameter.

    """
    wavelengths = speed_of_light.to("m/s").value / freqs

    scalar = 2.2150894  # Found by fitting a Gaussian to an Airy disk function

    sigma = np.arcsin(scalar * wavelengths / (np.pi * diam)) * 2 / 2.355

    return sigma


class GaussianBeam(AnalyticBeam):
    """
    Define a Gaussian beam, optionally with frequency dependent size.

    Parameters
    ----------
    sigma : float
        standard deviation in radians for the gaussian beam.
    sigma_type : str
        Either "efield" or "power" to indicate whether the sigma specifies the size of
        the efield or power beam. Ignored if `sigma` is None.
    diameter : float
        Dish diameter in meters to use to define the size of the gaussian beam, by
        matching the FWHM of the gaussian to the FWHM of an Airy disk. This will result
        in a frequency dependent beam.
    spectral_index : float
        Option to scale the gaussian beam width as a power law with frequency. If set
        to anything other than zero, the beam will be frequency dependent and the
        `reference_freq` must be set. Ignored if `sigma` is None.
    reference_freq : float
        The reference frequency for the beam width power law, required if `sigma` is not
        None and `spectral_index` is not zero. Ignored if `sigma` is None.

    """

    basis_vector_type = "az_za"

    def __init__(
        self,
        feed_array=None,
        sigma=None,
        sigma_type="efield",
        diameter=None,
        spectral_index=0.0,
        reference_freq=None,
    ):
        super().__init__(feed_array)

        if (diameter is None and sigma is None) or (
            diameter is not None and sigma is not None
        ):
            raise ValueError("One of diameter or sigma must be set but not both.")

        if sigma is not None:
            if sigma_type == "efield":
                self.sigma = sigma
            elif sigma_type == "power":
                self.sigma = np.sqrt(2) * sigma
            else:
                raise ValueError(
                    f"sigma_type is {sigma_type}, must be either 'efield' or 'power'"
                )

            if (spectral_index != 0.0) and (reference_freq is None):
                raise ValueError(
                    "reference_freq must be set if `spectral_index` is not zero."
                )
            elif reference_freq is None:
                reference_freq = 1.0
            self.spectral_index = spectral_index
            self.reference_freq = reference_freq

        if diameter is not None:
            self.diameter = diameter

    def get_sigmas(self, freq_array):
        """
        Get the sigmas for the gaussian beam using the diameter (if defined).

        Parameters
        ----------
        freq_array : array_like of floats, optional
            Frequency values to evaluate the beam at in Hertz.

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
                self.sigma * (freq_array / self.reference_freq) ** self.spectral_index
            )
        return sigmas

    def _efield_eval(self, az_array, za_array, freq_array):
        """Evaluate the efield at the given coordinates."""
        sigmas = self.get_sigmas(freq_array)

        values = np.exp(
            -(za_array[np.newaxis, ...] ** 2) / (2 * sigmas[:, np.newaxis] ** 2)
        )
        data_array = self._get_empty_data_array(az_array, za_array, freq_array)
        # This is different than what is in pyuvsim because it means that we have the
        # same response to any polarized source in both feeds. We think this is correct
        # for an azimuthally symmetric analytic beam but it is a change.
        # On pyuvsim we essentially have each basis vector go into only one feed.
        for fn in np.arange(self.Nfeeds):
            data_array[0, fn, :, :] = values / 2.0
            data_array[1, fn, :, :] = values / 2.0

    def _power_eval(self, az_array, za_array, freq_array):
        """Evaluate the power at the given coordinates."""
        sigmas = self.get_sigmas(freq_array)

        values = np.exp(
            -(za_array[np.newaxis, ...] ** 2) / (sigmas[:, np.newaxis] ** 2)
        )
        data_array = self._get_empty_data_array(
            az_array, za_array, freq_array, beam_type="power"
        )
        for fn in np.arange(self.Npols):
            # For power beams the first axis is shallow because we don't have to worry
            # about polarization.
            data_array[0, fn, :, :] = values
