# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Primary container for radio telescope antenna beams."""
import os
import warnings
import copy

import numpy as np
from scipy import interpolate
from astropy import units
from astropy.coordinates import Angle

from ..uvbase import UVBase
from .. import parameter as uvp
from .. import utils as uvutils

__all__ = ["UVBeam"]


class UVBeam(UVBase):
    """
    A class for defining a radio telescope antenna beam.

    Attributes
    ----------
    UVParameter objects: For full list see UVBeam Parameters
        (http://pyuvdata.readthedocs.io/en/latest/uvbeam_parameters.html).
        Some are always required, some are required for certain beam_types,
        antenna_types and pixel_coordinate_systems and others are always optional.

    """

    coordinate_system_dict = {
        "az_za": {
            "axes": ["azimuth", "zen_angle"],
            "description": "uniformly gridded azimuth, zenith angle coordinate system, "
            "where az runs from East to North in radians",
        },
        "orthoslant_zenith": {
            "axes": ["zenorth_x", "zenorth_y"],
            "description": "orthoslant projection at zenith where y points North, "
            "x point East",
        },
        "healpix": {
            "axes": ["hpx_inds"],
            "description": "HEALPix map with zenith at the north pole and "
            "az, za coordinate axes (for the basis_vector_array) "
            "where az runs from East to North",
        },
    }

    interpolation_function_dict = {
        "az_za_simple": {
            "description": "scipy RectBivariate spline interpolation",
            "func": "_interp_az_za_rect_spline",
        },
        "healpix_simple": {
            "description": "healpix nearest-neighbor bilinear interpolation",
            "func": "_interp_healpix_bilinear",
        },
    }

    def __init__(self):
        """Create a new UVBeam object."""
        # add the UVParameters to the class
        self._Nfreqs = uvp.UVParameter(
            "Nfreqs", description="Number of frequency channels", expected_type=int
        )

        self._Nspws = uvp.UVParameter(
            "Nspws",
            description="Number of spectral windows "
            "(ie non-contiguous spectral chunks). "
            "More than one spectral window is not "
            "currently supported.",
            expected_type=int,
        )

        desc = (
            "Number of basis vectors specified at each pixel, options "
            'are 2 or 3 (or 1 if beam_type is "power")'
        )
        self._Naxes_vec = uvp.UVParameter(
            "Naxes_vec", description=desc, expected_type=int, acceptable_vals=[2, 3]
        )

        desc = (
            "Number of basis vectors components specified at each pixel, options "
            "are 2 or 3.  Only required for E-field beams."
        )
        self._Ncomponents_vec = uvp.UVParameter(
            "Ncomponents_vec",
            description=desc,
            expected_type=int,
            acceptable_vals=[2, 3],
            required=False,
        )

        desc = (
            'Pixel coordinate system, options are: "'
            + '", "'.join(list(self.coordinate_system_dict.keys()))
            + '".'
        )
        for key in self.coordinate_system_dict:
            desc = desc + (
                ' "'
                + key
                + '" is a '
                + self.coordinate_system_dict[key]["description"]
                + ". It has axes ["
                + ", ".join(self.coordinate_system_dict[key]["axes"])
                + "]."
            )
        self._pixel_coordinate_system = uvp.UVParameter(
            "pixel_coordinate_system",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=list(self.coordinate_system_dict.keys()),
        )

        desc = (
            "Number of elements along the first pixel axis. "
            'Not required if pixel_coordinate_system is "healpix".'
        )
        self._Naxes1 = uvp.UVParameter(
            "Naxes1", description=desc, expected_type=int, required=False
        )

        desc = (
            "Coordinates along first pixel axis. "
            'Not required if pixel_coordinate_system is "healpix".'
        )
        self._axis1_array = uvp.UVParameter(
            "axis1_array",
            description=desc,
            expected_type=float,
            required=False,
            form=("Naxes1",),
        )

        desc = (
            "Number of elements along the second pixel axis. "
            'Not required if pixel_coordinate_system is "healpix".'
        )
        self._Naxes2 = uvp.UVParameter(
            "Naxes2", description=desc, expected_type=int, required=False
        )

        desc = (
            "Coordinates along second pixel axis. "
            'Not required if pixel_coordinate_system is "healpix".'
        )
        self._axis2_array = uvp.UVParameter(
            "axis2_array",
            description=desc,
            expected_type=float,
            required=False,
            form=("Naxes2",),
        )

        desc = (
            "Healpix nside parameter. Only required if "
            "pixel_coordinate_system is 'healpix'."
        )
        self._nside = uvp.UVParameter(
            "nside", description=desc, expected_type=int, required=False
        )

        desc = (
            'Healpix ordering parameter, allowed values are "ring" and "nested". '
            'Only required if pixel_coordinate_system is "healpix".'
        )
        self._ordering = uvp.UVParameter(
            "ordering",
            description=desc,
            expected_type=str,
            required=False,
            acceptable_vals=["ring", "nested"],
        )

        desc = (
            "Number of healpix pixels. Only required if "
            "pixel_coordinate_system is 'healpix'."
        )
        self._Npixels = uvp.UVParameter(
            "Npixels", description=desc, expected_type=int, required=False
        )

        desc = (
            "Healpix pixel numbers. Only required if "
            "pixel_coordinate_system is 'healpix'."
        )
        self._pixel_array = uvp.UVParameter(
            "pixel_array",
            description=desc,
            expected_type=int,
            required=False,
            form=("Npixels",),
        )

        desc = (
            "String indicating beam type. Allowed values are 'efield', " "and 'power'."
        )
        self._beam_type = uvp.UVParameter(
            "beam_type",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["efield", "power"],
        )

        desc = (
            "Beam basis vector components -- directions for which the "
            "electric field values are recorded in the pixel coordinate system. "
            'Not required if beam_type is "power". The shape depends on the '
            'pixel_coordinate_system, if it is "healpix", the shape is: '
            "(Naxes_vec, Ncomponents_vec, Npixels), otherwise it is "
            "(Naxes_vec, Ncomponents_vec, Naxes2, Naxes1)"
        )
        self._basis_vector_array = uvp.UVParameter(
            "basis_vector_array",
            description=desc,
            required=False,
            expected_type=float,
            form=("Naxes_vec", "Ncomponents_vec", "Naxes2", "Naxes1"),
            acceptable_range=(0, 1),
            tols=1e-3,
        )

        self._Nfeeds = uvp.UVParameter(
            "Nfeeds",
            description="Number of feeds. " 'Not required if beam_type is "power".',
            expected_type=int,
            acceptable_vals=[1, 2],
            required=False,
        )

        desc = (
            "Array of feed orientations. shape (Nfeeds). "
            'options are: N/E or x/y or R/L. Not required if beam_type is "power".'
        )
        self._feed_array = uvp.UVParameter(
            "feed_array",
            description=desc,
            required=False,
            expected_type=str,
            form=("Nfeeds",),
            acceptable_vals=["N", "E", "x", "y", "R", "L"],
        )

        self._Npols = uvp.UVParameter(
            "Npols",
            description="Number of polarizations. "
            'Only required if beam_type is "power".',
            expected_type=int,
            required=False,
        )

        desc = (
            "Array of polarization integers, shape (Npols). "
            "Uses the same convention as UVData: pseudo-stokes 1:4 (pI, pQ, pU, pV);  "
            "circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX). "
            'Only required if beam_type is "power".'
        )
        self._polarization_array = uvp.UVParameter(
            "polarization_array",
            description=desc,
            required=False,
            expected_type=int,
            form=("Npols",),
            acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
        )

        desc = (
            "Array of frequencies, center of the channel, "
            "shape (Nspws, Nfreqs), units Hz"
        )
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=("Nspws", "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )  # mHz

        self._spw_array = uvp.UVParameter(
            "spw_array",
            description="Array of spectral window " "Numbers, shape (Nspws)",
            form=("Nspws",),
            expected_type=int,
        )

        desc = (
            "Normalization standard of data_array, options are: "
            '"physical", "peak" or "solid_angle". Physical normalization '
            "means that the frequency dependence of the antenna sensitivity "
            "is included in the data_array while the frequency dependence "
            "of the receiving chain is included in the bandpass_array. "
            "Peak normalized means that for each frequency the data_array"
            "is separately normalized such that the peak is 1 (so the beam "
            "is dimensionless) and all direction-independent frequency "
            'dependence is moved to the bandpass_array (if the beam_type is "efield", '
            "then peak normalized means that the absolute value of the peak is 1). "
            "Solid angle normalized means the peak normalized "
            "beam is divided by the integral of the beam over the sphere, "
            "so the beam has dimensions of 1/stradian."
        )
        self._data_normalization = uvp.UVParameter(
            "data_normalization",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["physical", "peak", "solid_angle"],
        )

        desc = (
            "Depending on beam type, either complex E-field values "
            "('efield' beam type) or power values ('power' beam type) for "
            "beam model. Units are normalized to either peak or solid angle as "
            "given by data_normalization. The shape depends on the beam_type "
            "and pixel_coordinate_system, if it is 'healpix', the shape "
            "is: (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, Npixels), "
            "otherwise it is "
            "(Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)."
        )
        self._data_array = uvp.UVParameter(
            "data_array",
            description=desc,
            expected_type=complex,
            form=("Naxes_vec", "Nspws", "Nfeeds", "Nfreqs", "Naxes2", "Naxes1"),
            tols=1e-3,
        )

        desc = (
            "Frequency dependence of the beam. Depending on the data_normalization, "
            "this may contain only the frequency dependence of the receiving "
            'chain ("physical" normalization) or all the frequency dependence '
            '("peak" normalization).'
        )
        self._bandpass_array = uvp.UVParameter(
            "bandpass_array",
            description=desc,
            expected_type=float,
            form=("Nspws", "Nfreqs"),
            tols=1e-3,
        )

        # --------- metadata -------------
        self._telescope_name = uvp.UVParameter(
            "telescope_name",
            description="Name of telescope " "(string)",
            form="str",
            expected_type=str,
        )

        self._feed_name = uvp.UVParameter(
            "feed_name",
            description="Name of physical feed " "(string)",
            form="str",
            expected_type=str,
        )

        self._feed_version = uvp.UVParameter(
            "feed_version",
            description="Version of physical feed " "(string)",
            form="str",
            expected_type=str,
        )

        self._model_name = uvp.UVParameter(
            "model_name",
            description="Name of beam model " "(string)",
            form="str",
            expected_type=str,
        )

        self._model_version = uvp.UVParameter(
            "model_version",
            description="Version of beam model " "(string)",
            form="str",
            expected_type=str,
        )

        self._history = uvp.UVParameter(
            "history",
            description="String of history, units English",
            form="str",
            expected_type=str,
        )

        # ---------- phased_array stuff -------------
        desc = (
            'String indicating antenna type. Allowed values are "simple", and '
            '"phased_array"'
        )
        self._antenna_type = uvp.UVParameter(
            "antenna_type",
            form="str",
            expected_type=str,
            description=desc,
            acceptable_vals=["simple", "phased_array"],
        )

        desc = (
            'Required if antenna_type = "phased_array". Number of elements '
            "in phased array"
        )
        self._Nelements = uvp.UVParameter(
            "Nelements", required=False, description=desc, expected_type=int
        )

        desc = (
            'Required if antenna_type = "phased_array". Element coordinate '
            "system, options are: N-E or x-y"
        )
        self._element_coordinate_system = uvp.UVParameter(
            "element_coordinate_system",
            required=False,
            description=desc,
            expected_type=str,
            acceptable_vals=["N-E", "x-y"],
        )

        desc = (
            'Required if antenna_type = "phased_array". Array of element '
            "locations in element coordinate system,  shape: (2, Nelements)"
        )
        self._element_location_array = uvp.UVParameter(
            "element_location_array",
            required=False,
            description=desc,
            form=(2, "Nelements"),
            expected_type=float,
        )

        desc = (
            'Required if antenna_type = "phased_array". Array of element '
            "delays, units: seconds, shape: (Nelements)"
        )
        self._delay_array = uvp.UVParameter(
            "delay_array",
            required=False,
            description=desc,
            form=("Nelements",),
            expected_type=float,
        )

        desc = (
            'Required if antenna_type = "phased_array". Array of element '
            "gains, units: dB, shape: (Nelements)"
        )
        self._gain_array = uvp.UVParameter(
            "gain_array",
            required=False,
            description=desc,
            form=("Nelements",),
            expected_type=float,
        )

        desc = (
            'Required if antenna_type = "phased_array". Matrix of complex '
            "element couplings, units: dB, "
            "shape: (Nelements, Nelements, Nfeed, Nfeed, Nspws, Nfreqs)"
        )
        self._coupling_matrix = uvp.UVParameter(
            "coupling_matrix",
            required=False,
            description=desc,
            form=("Nelements", "Nelements", "Nfeed", "Nfeed", "Nspws", "Nfreqs"),
            expected_type=complex,
        )

        # -------- extra, non-required parameters ----------
        desc = (
            "Orientation of the physical dipole corresponding to what is "
            'labelled as the x polarization. Options are "east" '
            '(indicating east/west orientation) and "north" (indicating '
            "north/south orientation)"
        )
        self._x_orientation = uvp.UVParameter(
            "x_orientation",
            description=desc,
            required=False,
            expected_type=str,
            acceptable_vals=["east", "north"],
        )

        desc = (
            "String indicating interpolation function. Must be set to use "
            'the interp_* methods. Allowed values are : "'
            + '", "'.join(list(self.interpolation_function_dict.keys()))
            + '".'
        )
        self._interpolation_function = uvp.UVParameter(
            "interpolation_function",
            required=False,
            form="str",
            expected_type=str,
            description=desc,
            acceptable_vals=list(self.interpolation_function_dict.keys()),
        )
        desc = (
            "String indicating frequency interpolation kind. "
            "See scipy.interpolate.interp1d for details. Default is linear."
        )
        self._freq_interp_kind = uvp.UVParameter(
            "freq_interp_kind",
            required=False,
            form="str",
            expected_type=str,
            description=desc,
        )
        self.freq_interp_kind = "linear"

        desc = (
            "Any user supplied extra keywords, type=dict. Keys should be "
            "8 character or less strings if writing to beam fits files. "
            'Use the special key "comment" for long multi-line string comments.'
        )
        self._extra_keywords = uvp.UVParameter(
            "extra_keywords",
            required=False,
            description=desc,
            value={},
            spoof_val={},
            expected_type=dict,
        )

        desc = (
            "Reference impedance of the beam model. The radiated E-farfield "
            "or the realised gain depend on the impedance of the port used to "
            "excite the simulation. This is the reference impedance (Z0) of "
            "the simulation. units: Ohms"
        )
        self._reference_impedance = uvp.UVParameter(
            "reference_impedance",
            required=False,
            description=desc,
            expected_type=float,
            tols=1e-3,
        )

        desc = "Array of receiver temperatures, shape (Nspws, Nfreqs), units K"
        self._receiver_temperature_array = uvp.UVParameter(
            "receiver_temperature_array",
            required=False,
            description=desc,
            form=("Nspws", "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )

        desc = "Array of antenna losses, shape (Nspws, Nfreqs), units dB?"
        self._loss_array = uvp.UVParameter(
            "loss_array",
            required=False,
            description=desc,
            form=("Nspws", "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )

        desc = "Array of antenna-amplifier mismatches, shape (Nspws, Nfreqs), units ?"
        self._mismatch_array = uvp.UVParameter(
            "mismatch_array",
            required=False,
            description=desc,
            form=("Nspws", "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )

        desc = (
            "S parameters of receiving chain, shape (4, Nspws, Nfreqs), "
            "ordering: s11, s12, s21, s22. see "
            "https://en.wikipedia.org/wiki/Scattering_parameters#Two-Port_S-Parameters"
        )
        self._s_parameters = uvp.UVParameter(
            "s_parameters",
            required=False,
            description=desc,
            form=(4, "Nspws", "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )

        super(UVBeam, self).__init__()

    def _set_cs_params(self):
        """Set parameters depending on pixel_coordinate_system."""
        if self.pixel_coordinate_system == "healpix":
            self._Naxes1.required = False
            self._axis1_array.required = False
            self._Naxes2.required = False
            self._axis2_array.required = False
            self._nside.required = True
            self._ordering.required = True
            self._Npixels.required = True
            self._pixel_array.required = True
            self._basis_vector_array.form = ("Naxes_vec", "Ncomponents_vec", "Npixels")
            if self.beam_type == "power":
                self._data_array.form = (
                    "Naxes_vec",
                    "Nspws",
                    "Npols",
                    "Nfreqs",
                    "Npixels",
                )
            else:
                self._data_array.form = (
                    "Naxes_vec",
                    "Nspws",
                    "Nfeeds",
                    "Nfreqs",
                    "Npixels",
                )
        else:
            self._Naxes1.required = True
            self._axis1_array.required = True
            self._Naxes2.required = True
            self._axis2_array.required = True
            if self.pixel_coordinate_system == "az_za":
                self._axis1_array.acceptable_range = [0, 2.0 * np.pi]
                self._axis2_array.acceptable_range = [0, np.pi]
            self._nside.required = False
            self._ordering.required = False
            self._Npixels.required = False
            self._pixel_array.required = False
            self._basis_vector_array.form = (
                "Naxes_vec",
                "Ncomponents_vec",
                "Naxes2",
                "Naxes1",
            )
            if self.beam_type == "power":
                self._data_array.form = (
                    "Naxes_vec",
                    "Nspws",
                    "Npols",
                    "Nfreqs",
                    "Naxes2",
                    "Naxes1",
                )
            else:
                self._data_array.form = (
                    "Naxes_vec",
                    "Nspws",
                    "Nfeeds",
                    "Nfreqs",
                    "Naxes2",
                    "Naxes1",
                )

    def set_cs_params(self):
        """
        Set parameters depending on pixel_coordinate_system.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_cs_params` instead.
        """
        warnings.warn(
            "`set_cs_params` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_cs_params` instead.",
            DeprecationWarning,
        )
        self._set_cs_params()

    def _set_efield(self):
        """Set beam_type to 'efield' and adjust required parameters."""
        self.beam_type = "efield"
        self._Naxes_vec.acceptable_vals = [2, 3]
        self._Ncomponents_vec.required = True
        self._basis_vector_array.required = True
        self._Nfeeds.required = True
        self._feed_array.required = True
        self._Npols.required = False
        self._polarization_array.required = False
        self._data_array.expected_type = complex
        # call set_cs_params to fix data_array form
        self._set_cs_params()

    def set_efield(self):
        """
        Set beam_type to 'efield' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_efield` instead.
        """
        warnings.warn(
            "`set_efield` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_efield` instead.",
            DeprecationWarning,
        )
        self._set_efield()

    def _set_power(self):
        """Set beam_type to 'power' and adjust required parameters."""
        self.beam_type = "power"
        self._Naxes_vec.acceptable_vals = [1, 2, 3]
        self._basis_vector_array.required = False
        self._Ncomponents_vec.required = False
        self._Nfeeds.required = False
        self._feed_array.required = False
        self._Npols.required = True
        self._polarization_array.required = True

        # If cross pols are included, the power beam is complex. Otherwise it's real
        self._data_array.expected_type = float
        for pol in self.polarization_array:
            if pol in [3, 4, -3, -4, -7, -8]:
                self._data_array.expected_type = complex

        # call set_cs_params to fix data_array form
        self._set_cs_params()

    def set_power(self):
        """
        Set beam_type to 'power' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_power` instead.
        """
        warnings.warn(
            "`set_power` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_power` instead.",
            DeprecationWarning,
        )
        self._set_power()

    def _set_simple(self):
        """Set antenna_type to 'simple' and adjust required parameters."""
        self.antenna_type = "simple"
        self._Nelements.required = False
        self._element_coordinate_system.required = False
        self._element_location_array.required = False
        self._delay_array.required = False
        self._gain_array.required = False
        self._coupling_matrix.required = False

    def set_simple(self):
        """
        Set antenna_type to 'simple' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_simple` instead.
        """
        warnings.warn(
            "`set_simple` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_simple` instead.",
            DeprecationWarning,
        )
        self._set_simple()

    def _set_phased_array(self):
        """Set antenna_type to 'phased_array' and adjust required parameters."""
        self.antenna_type = "phased_array"
        self._Nelements.required = True
        self._element_coordinate_system.required = True
        self._element_location_array.required = True
        self._delay_array.required = True
        self._gain_array.required = True
        self._coupling_matrix.required = True

    def set_phased_array(self):
        """
        Set antenna_type to 'simple' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_phased_array` instead.
        """
        warnings.warn(
            "`set_phased_array` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_phased_array` instead.",
            DeprecationWarning,
        )
        self._set_phased_array()

    def check(self, check_extra=True, run_check_acceptability=True):
        """
        Check that all required parameters are set reasonably.

        Check that required parameters exist and have appropriate shapes.
        Optionally check if the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check if values in required parameters are acceptable.

        """
        # first make sure the required parameters and forms are set properly
        # for the pixel_coordinate_system
        self._set_cs_params()

        # first run the basic check from UVBase
        super(UVBeam, self).check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # check that basis_vector_array are basis vectors
        if self.basis_vector_array is not None:
            if np.max(np.linalg.norm(self.basis_vector_array, axis=1)) > (1 + 1e-15):
                raise ValueError("basis vectors must have lengths of 1 or less.")

        # issue warning if extra_keywords keys are longer than 8 characters
        for key in list(self.extra_keywords.keys()):
            if len(key) > 8:
                warnings.warn(
                    "key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 if written "
                    "to a fits file format.".format(key=key)
                )

        # issue warning if extra_keywords values are lists, arrays or dicts
        for key, value in self.extra_keywords.items():
            if isinstance(value, (list, dict, np.ndarray)):
                warnings.warn(
                    "{key} in extra_keywords is a list, array or dict, "
                    "which will raise an error when writing fits "
                    "files".format(key=key)
                )

        return True

    def peak_normalize(self):
        """Convert to peak normalization."""
        if self.data_normalization == "solid_angle":
            raise NotImplementedError(
                "Conversion from solid_angle to peak "
                "normalization is not yet implemented"
            )
        for i in range(self.Nfreqs):
            max_val = abs(self.data_array[:, :, :, i, :]).max()
            self.data_array[:, :, :, i, :] /= max_val
            self.bandpass_array[:, i] *= max_val
        self.data_normalization = "peak"

    def efield_to_power(
        self,
        calc_cross_pols=True,
        keep_basis_vector=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=True,
    ):
        """
        Convert E-field beam to power beam.

        Parameters
        ----------
        calc_cross_pols : bool
            If True, calculate the crossed polarization beams
            (e.g. 'xy' and 'yx'), otherwise only calculate the same
            polarization beams (e.g. 'xx' and 'yy').
        keep_basis_vector : bool
            If True, keep the directionality information and
            just multiply the efields for each basis vector separately
            (caution: this is not what is standardly meant by the power beam).
        inplace : bool
            Option to apply conversion directly on self or to return a new
            UVBeam object.
        run_check : bool
            Option to check for the existence and proper shapes of the required
            parameters after converting to power.
        run_check_acceptability : bool
            Option to check acceptable range of the values of required parameters
            after converting to power.
        check_extra : bool
            Option to check optional parameters as well as required ones.

        """
        if inplace:
            beam_object = self
        else:
            beam_object = self.copy()

        if beam_object.beam_type != "efield":
            raise ValueError("beam_type must be efield")

        efield_data = beam_object.data_array
        efield_naxes_vec = beam_object.Naxes_vec

        feed_pol_order = [(0, 0)]
        if beam_object.Nfeeds > 1:
            feed_pol_order.append((1, 1))

        if calc_cross_pols:
            beam_object.Npols = beam_object.Nfeeds ** 2
            if beam_object.Nfeeds > 1:
                feed_pol_order.extend([(0, 1), (1, 0)])
        else:
            beam_object.Npols = beam_object.Nfeeds

        pol_strings = []
        for pair in feed_pol_order:
            pol_strings.append(
                beam_object.feed_array[pair[0]] + beam_object.feed_array[pair[1]]
            )
        beam_object.polarization_array = np.array(
            [
                uvutils.polstr2num(ps.upper(), x_orientation=self.x_orientation)
                for ps in pol_strings
            ]
        )

        if not keep_basis_vector:
            beam_object.Naxes_vec = 1

        # adjust requirements, fix data_array form
        beam_object._set_power()
        power_data = np.zeros(
            beam_object._data_array.expected_shape(beam_object), dtype=np.complex128
        )

        if keep_basis_vector:
            for pol_i, pair in enumerate(feed_pol_order):
                power_data[:, :, pol_i] = efield_data[:, :, pair[0]] * np.conj(
                    efield_data[:, :, pair[1]]
                )

        else:
            for pol_i, pair in enumerate(feed_pol_order):
                if efield_naxes_vec == 2:
                    for comp_i in range(2):
                        power_data[0, :, pol_i] += (
                            (
                                efield_data[0, :, pair[0]]
                                * np.conj(efield_data[0, :, pair[1]])
                            )
                            * beam_object.basis_vector_array[0, comp_i] ** 2
                            + (
                                efield_data[1, :, pair[0]]
                                * np.conj(efield_data[1, :, pair[1]])
                            )
                            * beam_object.basis_vector_array[1, comp_i] ** 2
                            + (
                                efield_data[0, :, pair[0]]
                                * np.conj(efield_data[1, :, pair[1]])
                                + efield_data[1, :, pair[0]]
                                * np.conj(efield_data[0, :, pair[1]])
                            )
                            * (
                                beam_object.basis_vector_array[0, comp_i]
                                * beam_object.basis_vector_array[1, comp_i]
                            )
                        )
                else:
                    raise ValueError(
                        "Conversion to power with 3-vector efields "
                        "is not currently supported because we have "
                        "no examples to work with."
                    )

        power_data = np.real_if_close(power_data, tol=10)

        beam_object.data_array = power_data
        beam_object.Nfeeds = None
        beam_object.feed_array = None
        if not keep_basis_vector:
            beam_object.basis_vector_array = None
            beam_object.Ncomponents_vec = None

        history_update_string = " Converted from efield to power using pyuvdata."

        beam_object.history = beam_object.history + history_update_string

        if run_check:
            beam_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return beam_object

    def _stokes_matrix(self, pol_index):
        """
        Calculate Pauli matrices for pseudo-Stokes conversion.

        Derived from https://arxiv.org/pdf/1401.2095.pdf, the Pauli
        indices are reordered from the quantum mechanical
        convention to an order which gives the ordering of the pseudo-Stokes vector
        ['pI', 'pQ', 'pU, 'pV'].

        Parameters
        ----------
        pol_index : int
            Polarization index for which the Pauli matrix is generated, the index
            must lie between 0 and 3 ('pI': 0, 'pQ': 1, 'pU': 2, 'pV':3).

        Returns
        -------
        array of float
            Pauli matrix for pol_index. Shape: (2, 2)
        """
        if pol_index < 0:
            raise ValueError("n must be positive integer.")
        if pol_index > 4:
            raise ValueError("n should lie between 0 and 3.")
        if pol_index == 0:
            pauli_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        if pol_index == 1:
            pauli_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
        if pol_index == 2:
            pauli_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
        if pol_index == 3:
            pauli_mat = np.array([[0.0, -1.0j], [1.0j, 0.0]])

        return pauli_mat

    def _construct_mueller(self, jones, pol_index1, pol_index2):
        """
        Generate Mueller components.

        Following https://arxiv.org/pdf/1802.04151.pdf. Using equation:

                Mij = Tr(J sigma_i J^* sigma_j)

        where sigma_i and sigma_j are Pauli matrices.

        Parameters
        ----------
        jones : array of float
            Jones matrices containing the electric field for the dipole arms
            or linear polarizations. Shape: (Npixels, 2, 2) for Healpix beams or
            (Naxes1 * Naxes2, 2, 2) otherwise.
        pol_index1 : int
            Polarization index referring to the first index of Mij (i).
        pol_index2 : int
            Polarization index referring to the second index of Mij (j).

        Returns
        -------
        array of float
            Mueller array containing the Mij values, shape: (Npixels,) for Healpix beams
            or (Naxes1 * Naxes2,) otherwise.
        """
        pauli_mat1 = self._stokes_matrix(pol_index1)
        pauli_mat2 = self._stokes_matrix(pol_index2)

        mueller = 0.5 * np.einsum(
            "...ab,...bc,...cd,...ad", pauli_mat1, jones, pauli_mat2, np.conj(jones)
        )
        mueller = np.abs(mueller)

        return mueller

    def efield_to_pstokes(
        self,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Convert E-field to pseudo-stokes power.

        Following https://arxiv.org/pdf/1802.04151.pdf, using the equation:

                M_ij = Tr(sigma_i J sigma_j J^*)

        where sigma_i and sigma_j are Pauli matrices.

        Parameters
        ----------
        inplace : bool
            Option to apply conversion directly on self or to return a new
            UVBeam object.
        run_check : bool
            Option to check for the existence and proper shapes of the required
            parameters after converting to power.
        run_check_acceptability : bool
            Option to check acceptable range of the values of required parameters
            after converting to power.
        check_extra : bool
            Option to check optional parameters as well as required ones.

        """
        if inplace:
            beam_object = self
        else:
            beam_object = self.copy()

        if beam_object.beam_type != "efield":
            raise ValueError("beam_type must be efield.")

        efield_data = beam_object.data_array
        _sh = beam_object.data_array.shape
        Nfreqs = beam_object.Nfreqs

        if self.pixel_coordinate_system != "healpix":
            Naxes2, Naxes1 = beam_object.Naxes2, beam_object.Naxes1
            npix = Naxes1 * Naxes2
            efield_data = efield_data.reshape(efield_data.shape[:-2] + (npix,))
            _sh = efield_data.shape

        # construct jones matrix containing the electric field

        pol_strings = ["pI", "pQ", "pU", "pV"]
        power_data = np.zeros(
            (1, 1, len(pol_strings), _sh[-2], _sh[-1]), dtype=np.complex128
        )
        beam_object.polarization_array = np.array(
            [
                uvutils.polstr2num(ps.upper(), x_orientation=self.x_orientation)
                for ps in pol_strings
            ]
        )

        for fq_i in range(Nfreqs):
            jones = np.zeros((_sh[-1], 2, 2), dtype=np.complex128)
            pol_strings = ["pI", "pQ", "pU", "pV"]
            jones[:, 0, 0] = efield_data[0, 0, 0, fq_i, :]
            jones[:, 0, 1] = efield_data[0, 0, 1, fq_i, :]
            jones[:, 1, 0] = efield_data[1, 0, 0, fq_i, :]
            jones[:, 1, 1] = efield_data[1, 0, 1, fq_i, :]

            for pol_i in range(len(pol_strings)):
                power_data[:, :, pol_i, fq_i, :] = self._construct_mueller(
                    jones, pol_i, pol_i
                )

        if self.pixel_coordinate_system != "healpix":
            power_data = power_data.reshape(power_data.shape[:-1] + (Naxes2, Naxes1))
        beam_object.data_array = power_data
        beam_object.polarization_array = np.array(
            [
                uvutils.polstr2num(ps.upper(), x_orientation=self.x_orientation)
                for ps in pol_strings
            ]
        )
        beam_object.Naxes_vec = 1
        beam_object._set_power()

        history_update_string = (
            " Converted from efield to pseudo-stokes power using pyuvdata."
        )
        beam_object.Npols = beam_object.Nfeeds ** 2
        beam_object.history = beam_object.history + history_update_string
        beam_object.Nfeeds = None
        beam_object.feed_array = None
        beam_object.basis_vector_array = None
        beam_object.Ncomponents_vec = None

        if run_check:
            beam_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return beam_object

    def _interp_freq(self, freq_array, kind="linear", tol=1.0):
        """
        Interpolate function along frequency axis.

        Parameters
        ----------
        freq_array : array_like of floats
            Frequency values to interpolate to.
        kind : str
            Interpolation method to use frequency.
            See scipy.interpolate.interp1d for details.

        Returns
        -------
        interp_data : array_like of float or complex
            The array of interpolated data values,
            shape: (Naxes_vec, Nspws, Nfeeds or Npols, freq_array.size,
            Npixels or (Naxis2, Naxis1))
        interp_bandpass : array_like of float
            The interpolated bandpass. shape: (Nspws, freq_array.size)

        """
        assert isinstance(freq_array, np.ndarray)
        assert freq_array.ndim == 1

        # use the beam at nearest neighbors if not kind is 'nearest'
        if kind == "nearest":
            freq_dists = np.abs(self.freq_array - freq_array.reshape(-1, 1))
            nearest_inds = np.argmin(freq_dists, axis=1)
            interp_arrays = [
                self.data_array[:, :, :, nearest_inds, :],
                self.bandpass_array[:, nearest_inds],
            ]

        # otherwise interpolate the beam
        else:
            if self.Nfreqs == 1:
                raise ValueError("Only one frequency in UVBeam so cannot interpolate.")

            if np.min(freq_array) < np.min(self.freq_array) or np.max(
                freq_array
            ) > np.max(self.freq_array):
                raise ValueError(
                    "at least one interpolation frequency is outside of "
                    "the UVBeam freq_array range."
                )

            def get_lambda(real_lut, imag_lut=None):
                # Returns function objects for interpolation reuse
                if imag_lut is None:
                    return lambda freqs: real_lut(freqs)
                else:
                    return lambda freqs: (real_lut(freqs) + 1j * imag_lut(freqs))

            interp_arrays = []
            for data, ax in zip([self.data_array, self.bandpass_array], [3, 1]):
                if np.iscomplexobj(data):
                    # interpolate real and imaginary parts separately
                    real_lut = interpolate.interp1d(
                        self.freq_array[0, :], data.real, kind=kind, axis=ax
                    )
                    imag_lut = interpolate.interp1d(
                        self.freq_array[0, :], data.imag, kind=kind, axis=ax
                    )
                    lut = get_lambda(real_lut, imag_lut)
                else:
                    lut = interpolate.interp1d(
                        self.freq_array[0, :], data, kind=kind, axis=ax
                    )
                    lut = get_lambda(lut)

                interp_arrays.append(lut(freq_array))

        return tuple(interp_arrays)

    def _interp_az_za_rect_spline(
        self,
        az_array,
        za_array,
        freq_array,
        freq_interp_kind="linear",
        freq_interp_tol=1.0,
        polarizations=None,
        reuse_spline=False,
        spline_opts=None,
    ):
        """
        Interpolate in az_za coordinate system with a simple spline.

        Parameters
        ----------
        az_array : array_like of floats
            Azimuth values to interpolate to in radians, specifying the azimuth
            positions for every interpolation point (same length as `za_array`).
        za_array : array_like of floats
            Zenith values to interpolate to in radians, specifying the zenith
            positions for every interpolation point (same length as `az_array`).
        freq_array : array_like of floats
            Frequency values to interpolate to.
        freq_interp_kind : str
            Interpolation method to use frequency.
            See scipy.interpolate.interp1d for details.
        polarizations : list of str
            polarizations to interpolate if beam_type is 'power'.
            Default is all polarizations in self.polarization_array.
        reuse_spline : bool
            Save the interpolation functions for reuse.
        spline_opts : dict
            Options (kx, ky, s) for numpy.RectBivariateSpline.

        Returns
        -------
        interp_data : array_like of float or complex
            The array of interpolated data values,
            shape: (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, az_array.size)
        interp_basis_vector : array_like of float
            The array of interpolated basis vectors,
            shape: (Naxes_vec, Ncomponents_vec, az_array.size)
        interp_bandpass : array_like of float
            The interpolated bandpass. shape: (Nspws, freq_array.size)

        """
        if self.pixel_coordinate_system != "az_za":
            raise ValueError('pixel_coordinate_system must be "az_za"')

        if freq_array is not None:
            assert isinstance(freq_array, np.ndarray)
            input_data_array, interp_bandpass = self._interp_freq(
                freq_array, kind=freq_interp_kind, tol=freq_interp_tol
            )
            input_nfreqs = freq_array.size
        else:
            input_data_array = self.data_array
            input_nfreqs = self.Nfreqs
            freq_array = self.freq_array[0]
            interp_bandpass = self.bandpass_array[0]

        if az_array is None or za_array is None:
            return input_data_array, self.basis_vector_array, interp_bandpass

        assert isinstance(az_array, np.ndarray)
        assert isinstance(za_array, np.ndarray)
        assert az_array.ndim == 1
        assert az_array.shape == za_array.shape

        npoints = az_array.size

        axis1_diff = np.diff(self.axis1_array)[0]
        axis2_diff = np.diff(self.axis2_array)[0]
        max_axis_diff = np.max([axis1_diff, axis2_diff])

        phi_length = np.abs(self.axis1_array[0] - self.axis1_array[-1]) + axis1_diff

        phi_vals, theta_vals = np.meshgrid(self.axis1_array, self.axis2_array)

        assert input_data_array.shape[3] == input_nfreqs

        if np.iscomplexobj(input_data_array):
            data_type = np.complex128
        else:
            data_type = np.float64

        if np.isclose(phi_length, 2 * np.pi, atol=axis1_diff):
            # phi wraps around, extend array in each direction to improve interpolation
            extend_length = 3
            phi_use = np.concatenate(
                (
                    np.flip(phi_vals[:, :extend_length] * (-1) - axis1_diff, axis=1),
                    phi_vals,
                    phi_vals[:, -1 * extend_length :] + extend_length * axis1_diff,
                ),
                axis=1,
            )
            theta_use = np.concatenate(
                (
                    theta_vals[:, :extend_length],
                    theta_vals,
                    theta_vals[:, -1 * extend_length :],
                ),
                axis=1,
            )

            low_slice = input_data_array[:, :, :, :, :, :extend_length]
            high_slice = input_data_array[:, :, :, :, :, -1 * extend_length :]

            data_use = np.concatenate((high_slice, input_data_array, low_slice), axis=5)
        else:
            phi_use = phi_vals
            theta_use = theta_vals
            data_use = input_data_array

        if self.basis_vector_array is not None:
            if np.any(self.basis_vector_array[0, 1, :] > 0) or np.any(
                self.basis_vector_array[1, 0, :] > 0
            ):
                # Input basis vectors are not aligned to the native theta/phi
                # coordinate system
                raise NotImplementedError(
                    "interpolation for input basis "
                    "vectors that are not aligned to the "
                    "native theta/phi coordinate system "
                    "is not yet supported"
                )
            else:
                # The basis vector array comes in defined at the rectangular grid.
                # Redefine it for the interpolation points
                interp_basis_vector = np.zeros(
                    [self.Naxes_vec, self.Ncomponents_vec, npoints]
                )
                interp_basis_vector[0, 0, :] = np.ones(npoints)  # theta hat
                interp_basis_vector[1, 1, :] = np.ones(npoints)  # phi hat
        else:
            interp_basis_vector = None

        def get_lambda(real_lut, imag_lut=None):
            # Returns function objects for interpolation reuse
            if imag_lut is None:
                return lambda za, az: real_lut(za, az, grid=False)
            else:
                return lambda za, az: (
                    real_lut(za, az, grid=False) + 1j * imag_lut(za, az, grid=False)
                )

        # Npols is only defined for power beams.  For E-field beams need Nfeeds.
        if self.beam_type == "power":
            # get requested polarization indices
            if polarizations is None:
                Npol_feeds = self.Npols
                pol_inds = np.arange(Npol_feeds)
            else:
                pols = [
                    uvutils.polstr2num(p, x_orientation=self.x_orientation)
                    for p in polarizations
                ]
                pol_inds = []
                for pol in pols:
                    if pol not in self.polarization_array:
                        raise ValueError(
                            "Requested polarization {} not found "
                            "in self.polarization_array".format(pol)
                        )
                    pol_inds.append(np.where(self.polarization_array == pol)[0][0])
                pol_inds = np.asarray(pol_inds)
                Npol_feeds = len(pol_inds)

        else:
            Npol_feeds = self.Nfeeds
            pol_inds = np.arange(Npol_feeds)

        data_shape = (self.Naxes_vec, self.Nspws, Npol_feeds, input_nfreqs, npoints)
        interp_data = np.zeros(data_shape, dtype=data_type)

        if spline_opts is None or not isinstance(spline_opts, dict):
            spline_opts = {}
        if reuse_spline and not hasattr(self, "saved_interp_functions"):
            int_dict = {}
            self.saved_interp_functions = int_dict
        for index1 in range(self.Nspws):
            for index3 in range(input_nfreqs):
                freq = freq_array[index3]
                for index0 in range(self.Naxes_vec):
                    for pol_return_ind, index2 in enumerate(pol_inds):
                        do_interp = True
                        key = (index1, freq, index2, index0)
                        if reuse_spline:
                            if key in self.saved_interp_functions.keys():
                                do_interp = False
                                lut = self.saved_interp_functions[key]

                        if do_interp:
                            if np.iscomplexobj(data_use):
                                # interpolate real and imaginary parts separately
                                real_lut = interpolate.RectBivariateSpline(
                                    theta_use[:, 0],
                                    phi_use[0, :],
                                    data_use[index0, index1, index2, index3, :].real,
                                    **spline_opts,
                                )
                                imag_lut = interpolate.RectBivariateSpline(
                                    theta_use[:, 0],
                                    phi_use[0, :],
                                    data_use[index0, index1, index2, index3, :].imag,
                                    **spline_opts,
                                )
                                lut = get_lambda(real_lut, imag_lut)
                            else:
                                lut = interpolate.RectBivariateSpline(
                                    theta_use[:, 0],
                                    phi_use[0, :],
                                    data_use[index0, index1, index2, index3, :],
                                    **spline_opts,
                                )
                                lut = get_lambda(lut)
                            if reuse_spline:
                                self.saved_interp_functions[key] = lut
                        if index0 == 0 and index1 == 0 and index2 == 0 and index3 == 0:
                            for point_i in range(npoints):
                                pix_dists = np.sqrt(
                                    (theta_use - za_array[point_i]) ** 2.0
                                    + (phi_use - az_array[point_i]) ** 2.0
                                )
                                if np.min(pix_dists) > (max_axis_diff * 2.0):
                                    raise ValueError(
                                        "at least one interpolation location "
                                        "is outside of the UVBeam pixel coverage."
                                    )
                        interp_data[index0, index1, pol_return_ind, index3, :] = lut(
                            za_array, az_array
                        )

        return interp_data, interp_basis_vector, interp_bandpass

    def _interp_healpix_bilinear(
        self,
        az_array,
        za_array,
        freq_array,
        freq_interp_kind="linear",
        freq_interp_tol=1.0,
        polarizations=None,
        reuse_spline=False,
    ):
        """
        Interpolate in Healpix coordinate system with a simple bilinear function.

        Parameters
        ----------
        az_array : array_like of floats
            Azimuth values to interpolate to in radians, specifying the azimuth
            positions for every interpolation point (same length as `za_array`).
        za_array : array_like of floats
            Zenith values to interpolate to in radians, specifying the zenith
            positions for every interpolation point (same length as `az_array`).
        freq_array : array_like of floats
            Frequency values to interpolate to.
        freq_interp_kind : str
            Interpolation method to use frequency.
            See scipy.interpolate.interp1d for details.
        polarizations : list of str
            polarizations to interpolate if beam_type is 'power'.
            Default is all polarizations in self.polarization_array.

        Returns
        -------
        interp_data : array_like of float or complex
            The array of interpolated data values,
            shape: (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, az_array.size)
        interp_basis_vector : array_like of float
            The array of interpolated basis vectors,
            shape: (Naxes_vec, Ncomponents_vec, az_array.size)
        interp_bandpass : array_like of float
            The interpolated bandpass. shape: (Nspws, freq_array.size)

        """
        try:
            from astropy_healpix import HEALPix
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "astropy_healpix is not installed but is "
                "required for healpix functionality. "
                'Install "astropy-healpix" using conda or pip.'
            ) from e

        if self.pixel_coordinate_system != "healpix":
            raise ValueError('pixel_coordinate_system must be "healpix"')

        if not self.Npixels == 12 * self.nside ** 2:
            raise ValueError(
                "simple healpix interpolation requires full sky healpix maps."
            )
        if not np.max(np.abs(np.diff(self.pixel_array))) == 1:
            raise ValueError(
                "simple healpix interpolation requires healpix pixels to be in order."
            )

        if freq_array is not None:
            assert isinstance(freq_array, np.ndarray)
            input_data_array, interp_bandpass = self._interp_freq(
                freq_array, kind=freq_interp_kind, tol=freq_interp_tol
            )
            input_nfreqs = freq_array.size
        else:
            input_data_array = self.data_array
            input_nfreqs = self.Nfreqs
            freq_array = self.freq_array[0]
            interp_bandpass = self.bandpass_array[0]

        if az_array is None or za_array is None:
            return input_data_array, self.basis_vector_array, interp_bandpass

        assert isinstance(az_array, np.ndarray)
        assert isinstance(za_array, np.ndarray)
        assert az_array.ndim == 1
        assert az_array.shape == za_array.shape

        npoints = az_array.size

        # Npols is only defined for power beams.  For E-field beams need Nfeeds.
        if self.beam_type == "power":
            # get requested polarization indices
            if polarizations is None:
                Npol_feeds = self.Npols
                pol_inds = np.arange(Npol_feeds)
            else:
                pols = [
                    uvutils.polstr2num(p, x_orientation=self.x_orientation)
                    for p in polarizations
                ]
                pol_inds = []
                for pol in pols:
                    if pol not in self.polarization_array:
                        raise ValueError(
                            f"Requested polarization {pol} not found "
                            "in self.polarization_array"
                        )
                    pol_inds.append(np.where(self.polarization_array == pol)[0][0])
                pol_inds = np.asarray(pol_inds)
                Npol_feeds = len(pol_inds)
        else:
            Npol_feeds = self.Nfeeds
            pol_inds = np.arange(Npol_feeds)

        if np.iscomplexobj(input_data_array):
            data_type = np.complex128
        else:
            data_type = np.float64
        interp_data = np.zeros(
            (self.Naxes_vec, self.Nspws, Npol_feeds, input_nfreqs, len(az_array)),
            dtype=data_type,
        )

        if self.basis_vector_array is not None:
            if np.any(self.basis_vector_array[0, 1, :] > 0) or np.any(
                self.basis_vector_array[1, 0, :] > 0
            ):
                """ Input basis vectors are not aligned to the native theta/phi
                coordinate system """
                raise NotImplementedError(
                    "interpolation for input basis "
                    "vectors that are not aligned to the "
                    "native theta/phi coordinate system "
                    "is not yet supported"
                )
            else:
                """ The basis vector array comes in defined at the rectangular grid.
                Redefine it for the interpolation points """
                interp_basis_vector = np.zeros(
                    [self.Naxes_vec, self.Ncomponents_vec, npoints]
                )
                interp_basis_vector[0, 0, :] = np.ones(npoints)  # theta hat
                interp_basis_vector[1, 1, :] = np.ones(npoints)  # phi hat
        else:
            interp_basis_vector = None

        hp_obj = HEALPix(nside=self.nside, order=self.ordering)
        lat_array = Angle(np.pi / 2, units.radian) - Angle(za_array, units.radian)
        lon_array = Angle(az_array, units.radian)
        for index1 in range(self.Nspws):
            for index3 in range(input_nfreqs):
                for index0 in range(self.Naxes_vec):
                    for index2 in range(Npol_feeds):
                        if np.iscomplexobj(input_data_array):
                            # interpolate real and imaginary parts separately
                            real_hmap = hp_obj.interpolate_bilinear_lonlat(
                                lon_array,
                                lat_array,
                                input_data_array[
                                    index0, index1, pol_inds[index2], index3, :
                                ].real,
                            )
                            imag_hmap = hp_obj.interpolate_bilinear_lonlat(
                                lon_array,
                                lat_array,
                                input_data_array[
                                    index0, index1, pol_inds[index2], index3, :
                                ].imag,
                            )

                            hmap = real_hmap + 1j * imag_hmap
                        else:
                            # interpolate once
                            hmap = hp_obj.interpolate_bilinear_lonlat(
                                lon_array,
                                lat_array,
                                input_data_array[
                                    index0, index1, pol_inds[index2], index3, :
                                ],
                            )

                        interp_data[index0, index1, index2, index3, :] = hmap

        return interp_data, interp_basis_vector, interp_bandpass

    def interp(
        self,
        az_array=None,
        za_array=None,
        az_za_grid=False,
        healpix_nside=None,
        healpix_inds=None,
        freq_array=None,
        freq_interp_tol=1.0,
        polarizations=None,
        return_bandpass=False,
        reuse_spline=False,
        spline_opts=None,
        new_object=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Interpolate beam to given frequency, az & za locations or Healpix pixel centers.

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
        healpix_nside : int, optional
            HEALPix nside parameter if interpolating to HEALPix pixels.
        healpix_inds : array_like of int, optional
            HEALPix indices to interpolate to. Defaults to all indices in the
            map if `healpix_nside` is set and `az_array` and `za_array` are None.
        freq_array : array_like of floats, optional
            Frequency values to interpolate to.
        freq_interp_tol : float
            Frequency distance tolerance [Hz] of nearest neighbors.
            If *all* elements in freq_array have nearest neighbor distances within
            the specified tolerance then return the beam at each nearest neighbor,
            otherwise interpolate the beam.
        polarizations : list of str
            polarizations to interpolate if beam_type is 'power'.
            Default is all polarizations in self.polarization_array.
        new_object : bool
            Option to return a new UVBeam object with the interpolated data,
            if possible. Note that this is only possible for Healpix pixels or
            if az_za_grid is True and `az_array` and `za_array` are evenly spaced
            or for frequency only interpolation.
        reuse_spline : bool
            Save the interpolation functions for reuse. Only applies for
            `az_za_simple` interpolation.
        spline_opts : dict
            Provide options to numpy.RectBivariateSpline. This includes spline
            order parameters `kx` and `ky`, and smoothing parameter `s`.
            Only applies for `az_za_simple` interpolation.
        run_check : bool
            Only used if new_object is True. Option to check for the existence
            and proper shapes of required parameters on the new object.
        check_extra : bool
            Only used if new_object is True. Option to check optional parameters
            as well as required ones on the new object.
        run_check_acceptability : bool
            Only used if new_object is True. Option to check acceptable range
            of the values of required parameters on the new object.

        Returns
        -------
        array_like of float or complex or a UVBeam object
            Either an array of interpolated values or a UVBeam object if
            `new_object` is True. The shape of the interpolated data will be:
            (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs or freq_array.size if
            freq_array is passed, Npixels/(Naxis1, Naxis2) or az_array.size if
            az/za_arrays are passed)
        interp_basis_vector : array_like of float, optional
            an array of interpolated basis vectors (or self.basis_vector_array
            if az/za_arrays are not passed), shape: (Naxes_vec, Ncomponents_vec,
            Npixels/(Naxis1, Naxis2) or az_array.size if az/za_arrays are passed)
        interp_bandpass : array_like of float, optional
            The interpolated bandpass, only returned if `return_bandpass` is True.
            shape: (Nspws, freq_array.size)

        """
        if self.interpolation_function is None:
            raise ValueError("interpolation_function must be set on object first")
        if self.freq_interp_kind is None:
            raise ValueError("freq_interp_kind must be set on object first")

        if new_object:
            if not az_za_grid and az_array is not None:
                raise ValueError(
                    "A new object can only be returned if "
                    "az_za_grid is True or for Healpix pixels or "
                    "for frequency only interpolation."
                )

        kind_use = self.freq_interp_kind
        if freq_array is not None:
            # get frequency distances
            freq_dists = np.abs(self.freq_array - freq_array.reshape(-1, 1))
            nearest_dist = np.min(freq_dists, axis=1)
            interp_bool = np.any(nearest_dist >= freq_interp_tol)

            # use the beam at nearest neighbors if not interp_bool
            if not interp_bool:
                kind_use = "nearest"

        if az_za_grid:
            az_array_use, za_array_use = np.meshgrid(az_array, za_array)
            az_array_use = az_array_use.flatten()
            za_array_use = za_array_use.flatten()
        else:
            az_array_use = copy.copy(az_array)
            za_array_use = copy.copy(za_array)

        if healpix_nside is not None or healpix_inds is not None:
            if healpix_nside is None:
                raise ValueError("healpix_nside must be set if healpix_inds is set.")
            if az_array is not None or za_array is not None:
                raise ValueError(
                    "healpix_nside and healpix_inds can not be "
                    "set if az_array or za_array is set."
                )
            try:
                from astropy_healpix import HEALPix
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "astropy_healpix is not installed but is "
                    "required for healpix functionality. "
                    'Install "astropy-healpix" using conda or pip.'
                ) from e

            hp_obj = HEALPix(nside=healpix_nside)
            if healpix_inds is None:
                healpix_inds = np.arange(hp_obj.npix)

            hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(healpix_inds)

            za_array_use = (Angle(np.pi / 2, units.radian) - hpx_lat).radian
            az_array_use = hpx_lon.radian

        interp_func = self.interpolation_function_dict[self.interpolation_function][
            "func"
        ]

        extra_keyword_dict = {}
        if interp_func == "_interp_az_za_rect_spline":
            extra_keyword_dict["reuse_spline"] = reuse_spline
            extra_keyword_dict["spline_opts"] = spline_opts

        interp_data, interp_basis_vector, interp_bandpass = getattr(self, interp_func)(
            az_array_use,
            za_array_use,
            freq_array,
            freq_interp_kind=kind_use,
            polarizations=polarizations,
            **extra_keyword_dict,
        )

        # return just the interpolated arrays
        if not new_object:
            if return_bandpass:
                return interp_data, interp_basis_vector, interp_bandpass
            else:
                return interp_data, interp_basis_vector

        # return a new UVBeam object with interpolated data
        else:
            # make a new object
            new_uvb = self.copy()

            history_update_string = " Interpolated"
            if freq_array is not None:
                history_update_string += " in frequency"
                new_uvb.Nfreqs = freq_array.size
                new_uvb.freq_array = freq_array.reshape(1, -1)
                new_uvb.bandpass_array = interp_bandpass
                new_uvb.freq_interp_kind = kind_use

            if az_array is not None:
                if freq_array is not None:
                    history_update_string += " and"
                if new_uvb.pixel_coordinate_system != "az_za":
                    input_desc = self.coordinate_system_dict[
                        new_uvb.pixel_coordinate_system
                    ]["description"]
                    output_desc = self.coordinate_system_dict["az_za"]["description"]
                    history_update_string += (
                        " from " + input_desc + " to " + output_desc
                    )
                    new_uvb.pixel_coordinate_system = "az_za"
                    new_uvb.Npixels = None
                    new_uvb.pixel_array = None
                    new_uvb.nside = None
                    new_uvb.ordering = None
                else:
                    history_update_string += " to a new azimuth/zenith angle grid"

                interp_data = interp_data.reshape(
                    interp_data.shape[:-1] + (za_array.size, az_array.size)
                )
                if interp_basis_vector is not None:
                    interp_basis_vector = interp_basis_vector.reshape(
                        interp_basis_vector.shape[:-1] + (za_array.size, az_array.size)
                    )

                new_uvb.axis1_array = az_array
                new_uvb.axis2_array = za_array
                new_uvb.Naxes1 = new_uvb.axis1_array.size
                new_uvb.Naxes2 = new_uvb.axis2_array.size

            elif healpix_nside is not None:
                if freq_array is not None:
                    history_update_string += " and"
                if new_uvb.pixel_coordinate_system != "healpix":
                    input_desc = self.coordinate_system_dict[
                        new_uvb.pixel_coordinate_system
                    ]["description"]
                    output_desc = self.coordinate_system_dict["healpix"]["description"]
                    history_update_string += (
                        " from " + input_desc + " to " + output_desc
                    )
                    new_uvb.pixel_coordinate_system = "healpix"
                    new_uvb.Naxes1 = None
                    new_uvb.axis1_array = None
                    new_uvb.Naxes2 = None
                    new_uvb.axis2_array = None
                else:
                    history_update_string += " to a new healpix grid"

                new_uvb.pixel_array = healpix_inds
                new_uvb.Npixels = new_uvb.pixel_array.size
                new_uvb.nside = healpix_nside
                new_uvb.ordering = "ring"

            history_update_string += (
                " using pyuvdata with interpolation_function = "
                + new_uvb.interpolation_function
            )
            if freq_array is not None:
                history_update_string += (
                    " and freq_interp_kind = " + new_uvb.freq_interp_kind
                )
            history_update_string += "."
            new_uvb.history = new_uvb.history + history_update_string
            new_uvb.data_array = interp_data
            if new_uvb.basis_vector_array is not None:
                new_uvb.basis_vector_array = interp_basis_vector

            if hasattr(new_uvb, "saved_interp_functions"):
                delattr(new_uvb, "saved_interp_functions")

            new_uvb._set_cs_params()
            if run_check:
                new_uvb.check(
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                )
            return new_uvb

    def to_healpix(
        self,
        nside=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=True,
    ):
        """
        Convert beam to the healpix coordinate system.

        The interpolation is done using the interpolation method specified in
        self.interpolation_function.

        Note that this interpolation isn't perfect. Interpolating an Efield beam
        and then converting to power gives a different result than converting
        to power and then interpolating at about a 5% level.

        Parameters
        ----------
        nside : int
            The nside to use for the Healpix map. If not specified, use
            the nside that gives the closest resolution that is higher than the
            input resolution.
        run_check : bool
            Option to check for the existence and proper shapes of required
            parameters after converting to healpix.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of required parameters
            after combining objects
        inplace : bool
            Option to perform the select directly on self or return a new UVBeam
            object.

        """
        try:
            from astropy_healpix import HEALPix
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "astropy_healpix is not installed but is "
                "required for healpix functionality. "
                'Install "astropy-healpix" using conda or pip.'
            ) from e

        if nside is None:
            min_res = np.min(
                np.array([np.diff(self.axis1_array)[0], np.diff(self.axis2_array)[0]])
            )
            nside_min_res = np.sqrt(3 / np.pi) * np.radians(60.0) / min_res
            nside = int(2 ** np.ceil(np.log2(nside_min_res)))
            hp_obj = HEALPix(nside=nside)
            assert hp_obj.pixel_resolution.to_value(units.radian) < min_res
        else:
            hp_obj = HEALPix(nside=nside)

        pixels = np.arange(hp_obj.npix)
        hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(pixels)

        hpx_theta = (Angle(np.pi / 2, units.radian) - hpx_lat).radian
        hpx_phi = hpx_lon.radian

        phi_vals, theta_vals = np.meshgrid(self.axis1_array, self.axis2_array)

        # Don't ask for interpolation to pixels that aren't inside the beam area
        pix_dists = np.sqrt(
            (theta_vals.ravel() - hpx_theta.reshape(-1, 1)) ** 2
            + (phi_vals.ravel() - hpx_phi.reshape(-1, 1)) ** 2
        )

        inds_to_use = np.argwhere(
            np.min(pix_dists, axis=1)
            < hp_obj.pixel_resolution.to_value(units.radian) * 2
        ).squeeze(1)

        if inds_to_use.size < hp_obj.npix:
            pixels = pixels[inds_to_use]

        beam_object = self.interp(
            healpix_nside=nside,
            healpix_inds=pixels,
            new_object=True,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        if not inplace:
            return beam_object
        else:
            for p in beam_object:
                param = getattr(beam_object, p)
                setattr(self, p, param)

    def _get_beam(self, pol):
        """
        Get the healpix power beam map corresponding to the specififed polarization.

        pseudo-stokes I: 'pI', Q: 'pQ', U: 'pU' and V: 'pV' or linear dipole
        polarization: 'XX', 'YY', etc.

        Parameters
        ----------
        pol : str or int
            polarization string or integer, Ex. a pseudo-stokes pol 'pI', or
            a linear pol 'XX'.

        Returns
        -------
        UVBeam
            healpix beam
        """
        # assert map is in healpix coords
        assert (
            self.pixel_coordinate_system == "healpix"
        ), "pixel_coordinate_system must be healpix"
        # assert beam_type is power
        assert self.beam_type == "power", "beam_type must be power"
        if isinstance(pol, (str, np.str_)):
            pol = uvutils.polstr2num(pol, x_orientation=self.x_orientation)
        pol_array = self.polarization_array
        if pol in pol_array:
            stokes_p_ind = np.where(np.isin(pol_array, pol))[0][0]
            beam = self.data_array[0, 0, stokes_p_ind]
        else:
            raise ValueError("Do not have the right polarization information")

        return beam

    def get_beam_area(self, pol="pI"):
        """
        Compute the integral of the beam in units of steradians.

        Pseudo-Stokes 'pI' (I), 'pQ'(Q), 'pU'(U), 'pV'(V) beam and linear
        dipole 'XX', 'XY', 'YX' and 'YY' are supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 and Kohn et al. (2018) or
        https://arxiv.org/pdf/1802.04151.pdf for details.

        Parameters
        ----------
        pol : str or int
            polarization string or integer, Ex. a pseudo-stokes pol 'pI', or a
            linear pol 'XX'.

        Returns
        -------
        omega : float
            Integral of the beam across the sky, units: steradians.

        """
        if isinstance(pol, (str, np.str_)):
            pol = uvutils.polstr2num(pol, x_orientation=self.x_orientation)
        if self.beam_type != "power":
            raise ValueError("beam_type must be power")
        if self.Naxes_vec > 1:
            raise ValueError("Expect scalar for power beam, found vector")
        if self._data_normalization.value != "peak":
            raise ValueError("beam must be peak normalized")
        if self.pixel_coordinate_system != "healpix":
            raise ValueError("Currently only healpix format supported")

        nside = self.nside

        # get beam
        beam = self._get_beam(pol)

        # get integral
        omega = np.sum(beam, axis=-1) * np.pi / (3.0 * nside ** 2)

        return omega

    def get_beam_sq_area(self, pol="pI"):
        """
        Compute the integral of the beam^2 in units of steradians.

        Pseudo-Stokes 'pI' (I), 'pQ'(Q), 'pU'(U), 'pV'(V) beam and
        linear dipole 'XX', 'XY', 'YX' and 'YY' are supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        pol : str or int
            polarization string or integer, Ex. a pseudo-stokes pol 'pI', or a
            linear pol 'XX'.

        Returns
        -------
        omega : float
            Integral of the beam^2 across the sky, units: steradians.

        """
        if isinstance(pol, (str, np.str_)):
            pol = uvutils.polstr2num(pol, x_orientation=self.x_orientation)
        if self.beam_type != "power":
            raise ValueError("beam_type must be power")
        if self.Naxes_vec > 1:
            raise ValueError("Expect scalar for power beam, found vector")
        if self._data_normalization.value != "peak":
            raise ValueError("beam must be peak normalized")
        if self.pixel_coordinate_system != "healpix":
            raise ValueError("Currently only healpix format supported")

        nside = self.nside

        # get beam
        beam = self._get_beam(pol)

        # get integral
        omega = np.sum(beam ** 2, axis=-1) * np.pi / (3.0 * nside ** 2)

        return omega

    def __add__(
        self,
        other,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Combine two UVBeam objects.

        Objects can be added along frequency, feed or polarization
        (for efield or power beams), and/or pixel axes.

        Parameters
        ----------
        other : UVBeam object
            UVBeam object to add to self.
        inplace : bool
            Option to overwrite self as we go, otherwise create a third object
            as the sum of the two.
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters after combining objects.

        """
        if inplace:
            this = self
        else:
            this = self.copy()
        # Check that both objects are UVBeam and valid
        this.check(check_extra=check_extra, run_check_acceptability=False)
        if not issubclass(other.__class__, this.__class__):
            if not issubclass(this.__class__, other.__class__):
                raise ValueError(
                    "Only UVBeam (or subclass) objects can be added "
                    "to a UVBeam (or subclass) object"
                )
        other.check(check_extra=check_extra, run_check_acceptability=False)

        # Check objects are compatible
        compatibility_params = [
            "_beam_type",
            "_data_normalization",
            "_telescope_name",
            "_feed_name",
            "_feed_version",
            "_model_name",
            "_model_version",
            "_pixel_coordinate_system",
            "_Naxes_vec",
            "_nside",
            "_ordering",
        ]
        for a in compatibility_params:
            if getattr(this, a) != getattr(other, a):
                msg = (
                    "UVParameter " + a[1:] + " does not match. Cannot combine objects."
                )
                raise ValueError(msg)

        # check for presence of optional parameters with a frequency axis in
        # both objects
        optional_freq_params = [
            "_receiver_temperature_array",
            "_loss_array",
            "_mismatch_array",
            "_s_parameters",
        ]
        for attr in optional_freq_params:
            this_attr = getattr(this, attr)
            other_attr = getattr(other, attr)
            if (
                this_attr.value is None or other_attr.value is None
            ) and this_attr != other_attr:
                warnings.warn(
                    "Only one of the UVBeam objects being combined "
                    "has optional parameter {attr}. After the sum the "
                    "final object will not have {attr}".format(attr=attr)
                )
                if this_attr.value is not None:
                    this_attr.value = None
                    setattr(this, attr, this_attr)

        # Build up history string
        history_update_string = " Combined data along "
        n_axes = 0

        # Check we don't have overlapping data
        if this.beam_type == "power":
            both_pol = np.intersect1d(this.polarization_array, other.polarization_array)
        else:
            both_pol = np.intersect1d(this.feed_array, other.feed_array)

        both_freq = np.intersect1d(this.freq_array[0, :], other.freq_array[0, :])

        if this.pixel_coordinate_system == "healpix":
            both_pixels = np.intersect1d(this.pixel_array, other.pixel_array)
        else:
            both_axis1 = np.intersect1d(this.axis1_array, other.axis1_array)
            both_axis2 = np.intersect1d(this.axis2_array, other.axis2_array)

        if len(both_pol) > 0:
            if len(both_freq) > 0:
                if self.pixel_coordinate_system == "healpix":
                    if len(both_pixels) > 0:
                        raise ValueError(
                            "These objects have overlapping data and"
                            " cannot be combined."
                        )
                else:
                    if len(both_axis1) > 0:
                        if len(both_axis2) > 0:
                            raise ValueError(
                                "These objects have overlapping data and"
                                " cannot be combined."
                            )

        if this.pixel_coordinate_system == "healpix":
            temp = np.nonzero(~np.in1d(other.pixel_array, this.pixel_array))[0]
            if len(temp) > 0:
                pix_new_inds = temp
                history_update_string += "healpix pixel"
                n_axes += 1
            else:
                pix_new_inds = []
        else:
            temp = np.nonzero(~np.in1d(other.axis1_array, this.axis1_array))[0]
            if len(temp) > 0:
                ax1_new_inds = temp
                history_update_string += "first image"
                n_axes += 1
            else:
                ax1_new_inds = []

            temp = np.nonzero(~np.in1d(other.axis2_array, this.axis2_array))[0]
            if len(temp) > 0:
                ax2_new_inds = temp
                if n_axes > 0:
                    history_update_string += ", second image"
                else:
                    history_update_string += "second image"
                n_axes += 1
            else:
                ax2_new_inds = []

        temp = np.nonzero(~np.in1d(other.freq_array[0, :], this.freq_array[0, :]))[0]
        if len(temp) > 0:
            fnew_inds = temp
            if n_axes > 0:
                history_update_string += ", frequency"
            else:
                history_update_string += "frequency"
            n_axes += 1
        else:
            fnew_inds = []

        if this.beam_type == "power":
            temp = np.nonzero(
                ~np.in1d(other.polarization_array, this.polarization_array)
            )[0]
            if len(temp) > 0:
                pnew_inds = temp
                if n_axes > 0:
                    history_update_string += ", polarization"
                else:
                    history_update_string += "polarization"
                n_axes += 1
            else:
                pnew_inds = []
        else:
            temp = np.nonzero(~np.in1d(other.feed_array, this.feed_array))[0]
            if len(temp) > 0:
                pnew_inds = temp
                if n_axes > 0:
                    history_update_string += ", feed"
                else:
                    history_update_string += "feed"
                n_axes += 1
            else:
                pnew_inds = []

        # Pad out self to accommodate new data
        if this.pixel_coordinate_system == "healpix":
            if len(pix_new_inds) > 0:
                data_pix_axis = 4
                data_pad_dims = tuple(
                    list(this.data_array.shape[0:data_pix_axis])
                    + [len(pix_new_inds)]
                    + list(this.data_array.shape[data_pix_axis + 1 :])
                )
                data_zero_pad = np.zeros(data_pad_dims)

                this.pixel_array = np.concatenate(
                    [this.pixel_array, other.pixel_array[pix_new_inds]]
                )
                order = np.argsort(this.pixel_array)
                this.pixel_array = this.pixel_array[order]

                this.data_array = np.concatenate(
                    [this.data_array, data_zero_pad], axis=data_pix_axis
                )[:, :, :, :, order]

                if this.beam_type == "efield":
                    basisvec_pix_axis = 2
                    basisvec_pad_dims = tuple(
                        list(this.basis_vector_array.shape[0:basisvec_pix_axis])
                        + [len(pix_new_inds)]
                        + list(this.basis_vector_array.shape[basisvec_pix_axis + 1 :])
                    )
                    basisvec_zero_pad = np.zeros(basisvec_pad_dims)

                    this.basis_vector_array = np.concatenate(
                        [this.basis_vector_array, basisvec_zero_pad],
                        axis=basisvec_pix_axis,
                    )[:, :, order]
        else:
            if len(ax1_new_inds) > 0:
                data_ax1_axis = 5
                data_pad_dims = tuple(
                    list(this.data_array.shape[0:data_ax1_axis])
                    + [len(ax1_new_inds)]
                    + list(this.data_array.shape[data_ax1_axis + 1 :])
                )
                data_zero_pad = np.zeros(data_pad_dims)

                this.axis1_array = np.concatenate(
                    [this.axis1_array, other.axis1_array[ax1_new_inds]]
                )
                order = np.argsort(this.axis1_array)
                this.axis1_array = this.axis1_array[order]
                this.data_array = np.concatenate(
                    [this.data_array, data_zero_pad], axis=data_ax1_axis
                )[:, :, :, :, :, order]

                if this.beam_type == "efield":
                    basisvec_ax1_axis = 3
                    basisvec_pad_dims = tuple(
                        list(this.basis_vector_array.shape[0:basisvec_ax1_axis])
                        + [len(ax1_new_inds)]
                        + list(this.basis_vector_array.shape[basisvec_ax1_axis + 1 :])
                    )
                    basisvec_zero_pad = np.zeros(basisvec_pad_dims)

                    this.basis_vector_array = np.concatenate(
                        [this.basis_vector_array, basisvec_zero_pad],
                        axis=basisvec_ax1_axis,
                    )[:, :, :, order]

            if len(ax2_new_inds) > 0:
                data_ax2_axis = 4
                data_pad_dims = tuple(
                    list(this.data_array.shape[0:data_ax2_axis])
                    + [len(ax2_new_inds)]
                    + list(this.data_array.shape[data_ax2_axis + 1 :])
                )
                data_zero_pad = np.zeros(data_pad_dims)

                this.axis2_array = np.concatenate(
                    [this.axis2_array, other.axis2_array[ax2_new_inds]]
                )
                order = np.argsort(this.axis2_array)
                this.axis2_array = this.axis2_array[order]

                this.data_array = np.concatenate(
                    [this.data_array, data_zero_pad], axis=data_ax2_axis
                )[:, :, :, :, order, ...]

                if this.beam_type == "efield":
                    basisvec_ax2_axis = 2
                    basisvec_pad_dims = tuple(
                        list(this.basis_vector_array.shape[0:basisvec_ax2_axis])
                        + [len(ax2_new_inds)]
                        + list(this.basis_vector_array.shape[basisvec_ax2_axis + 1 :])
                    )
                    basisvec_zero_pad = np.zeros(basisvec_pad_dims)

                    this.basis_vector_array = np.concatenate(
                        [this.basis_vector_array, basisvec_zero_pad],
                        axis=basisvec_ax2_axis,
                    )[:, :, order, ...]

        if len(fnew_inds) > 0:
            faxis = 3
            data_pad_dims = tuple(
                list(this.data_array.shape[0:faxis])
                + [len(fnew_inds)]
                + list(this.data_array.shape[faxis + 1 :])
            )
            data_zero_pad = np.zeros(data_pad_dims)

            this.freq_array = np.concatenate(
                [this.freq_array, other.freq_array[:, fnew_inds]], axis=1
            )
            order = np.argsort(this.freq_array[0, :])
            this.freq_array = this.freq_array[:, order]

            this.bandpass_array = np.concatenate(
                [this.bandpass_array, np.zeros((1, len(fnew_inds)))], axis=1
            )[:, order]

            this.data_array = np.concatenate(
                [this.data_array, data_zero_pad], axis=faxis
            )[:, :, :, order, ...]
            if this.receiver_temperature_array is not None:
                this.receiver_temperature_array = np.concatenate(
                    [this.receiver_temperature_array, np.zeros((1, len(fnew_inds)))],
                    axis=1,
                )[:, order]
            if this.loss_array is not None:
                this.loss_array = np.concatenate(
                    [this.loss_array, np.zeros((1, len(fnew_inds)))], axis=1
                )[:, order]
            if this.mismatch_array is not None:
                this.mismatch_array = np.concatenate(
                    [this.mismatch_array, np.zeros((1, len(fnew_inds)))], axis=1
                )[:, order]
            if this.s_parameters is not None:
                this.s_parameters = np.concatenate(
                    [this.s_parameters, np.zeros((4, 1, len(fnew_inds)))], axis=2
                )[:, :, order]

        if len(pnew_inds) > 0:
            paxis = 2
            data_pad_dims = tuple(
                list(this.data_array.shape[0:paxis])
                + [len(pnew_inds)]
                + list(this.data_array.shape[paxis + 1 :])
            )
            data_zero_pad = np.zeros(data_pad_dims)

            if this.beam_type == "power":
                this.polarization_array = np.concatenate(
                    [this.polarization_array, other.polarization_array[pnew_inds]]
                )
                order = np.argsort(np.abs(this.polarization_array))
                this.polarization_array = this.polarization_array[order]
            else:
                this.feed_array = np.concatenate(
                    [this.feed_array, other.feed_array[pnew_inds]]
                )
                order = np.argsort(this.feed_array)
                this.feed_array = this.feed_array[order]

            this.data_array = np.concatenate(
                [this.data_array, data_zero_pad], axis=paxis
            )[:, :, order, ...]

        # Now populate the data
        if this.beam_type == "power":
            this.Npols = this.polarization_array.shape[0]
            pol_t2o = np.nonzero(
                np.in1d(this.polarization_array, other.polarization_array)
            )[0]
        else:
            this.Nfeeds = this.feed_array.shape[0]
            pol_t2o = np.nonzero(np.in1d(this.feed_array, other.feed_array))[0]

        freq_t2o = np.nonzero(np.in1d(this.freq_array[0, :], other.freq_array[0, :]))[0]

        if this.pixel_coordinate_system == "healpix":
            this.Npixels = this.pixel_array.shape[0]
            pix_t2o = np.nonzero(np.in1d(this.pixel_array, other.pixel_array))[0]
            this.data_array[
                np.ix_(np.arange(this.Naxes_vec), [0], pol_t2o, freq_t2o, pix_t2o)
            ] = other.data_array
            if this.beam_type == "efield":
                this.basis_vector_array[
                    np.ix_(np.arange(this.Naxes_vec), np.arange(2), pix_t2o)
                ] = other.basis_vector_array
        else:
            this.Naxes1 = this.axis1_array.shape[0]
            this.Naxes2 = this.axis2_array.shape[0]
            ax1_t2o = np.nonzero(np.in1d(this.axis1_array, other.axis1_array))[0]
            ax2_t2o = np.nonzero(np.in1d(this.axis2_array, other.axis2_array))[0]
            this.data_array[
                np.ix_(
                    np.arange(this.Naxes_vec), [0], pol_t2o, freq_t2o, ax2_t2o, ax1_t2o
                )
            ] = other.data_array
            if this.beam_type == "efield":
                this.basis_vector_array[
                    np.ix_(np.arange(this.Naxes_vec), np.arange(2), ax2_t2o, ax1_t2o)
                ] = other.basis_vector_array

        this.bandpass_array[np.ix_([0], freq_t2o)] = other.bandpass_array

        if this.receiver_temperature_array is not None:
            this.receiver_temperature_array[
                np.ix_([0], freq_t2o)
            ] = other.receiver_temperature_array
        if this.loss_array is not None:
            this.loss_array[np.ix_([0], freq_t2o)] = other.loss_array
        if this.mismatch_array is not None:
            this.mismatch_array[np.ix_([0], freq_t2o)] = other.mismatch_array
        if this.s_parameters is not None:
            this.s_parameters[np.ix_(np.arange(4), [0], freq_t2o)] = other.s_parameters

        this.Nfreqs = this.freq_array.shape[1]

        # Check specific requirements
        if this.Nfreqs > 1:
            freq_separation = np.diff(this.freq_array[0, :])
            if not np.isclose(
                np.min(freq_separation),
                np.max(freq_separation),
                rtol=this._freq_array.tols[0],
                atol=this._freq_array.tols[1],
            ):
                warnings.warn(
                    "Combined frequencies are not evenly spaced. This will "
                    "make it impossible to write this data out to some file types."
                )

        if self.beam_type == "power" and this.Npols > 2:
            pol_separation = np.diff(this.polarization_array)
            if np.min(pol_separation) < np.max(pol_separation):
                warnings.warn(
                    "Combined polarizations are not evenly spaced. This will "
                    "make it impossible to write this data out to some file types."
                )

        if n_axes > 0:
            history_update_string += " axis using pyuvdata."
            this.history += history_update_string

        this.history = uvutils._combine_histories(this.history, other.history)

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return this

    def __iadd__(self, other):
        """
        Add in place.

        Parameters
        ----------
        other : UVBeam object
            Another UVBeam object to adding to self.
        """
        self.__add__(other, inplace=True)
        return self

    def select(
        self,
        axis1_inds=None,
        axis2_inds=None,
        pixels=None,
        frequencies=None,
        freq_chans=None,
        feeds=None,
        polarizations=None,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Downselect data to keep on the object along various axes.

        Axes that can be selected along include image axis indices or pixels
        (if healpix), frequencies and feeds or polarizations (if power).

        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        axis1_indss : array_like of int, optional
            The indices along the first image axis to keep in the object.
            Cannot be set if pixel_coordinate_system is "healpix".
        axis2_inds : array_like of int, optional
            The indices along the second image axis to keep in the object.
            Cannot be set if pixel_coordinate_system is "healpix".
        pixels : array_like of int, optional
            The healpix pixels to keep in the object.
            Cannot be set if pixel_coordinate_system is not "healpix".
        frequencies : array_like of float, optional
            The frequencies to keep in the object.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        feeds : array_like of int, optional
            The feeds to keep in the object. Cannot be set if the beam_type is "power".
        polarizations : array_like of int, optional
            The polarizations to keep in the object.
            Cannot be set if the beam_type is "efield".
        inplace : bool
            Option to perform the select directly on self or return
            a new UVBeam object, which is a subselection of self.
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after downselecting data on this object.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters after  downselecting data on this object.

        """
        if inplace:
            beam_object = self
        else:
            beam_object = self.copy()

        # build up history string as we go
        history_update_string = "  Downselected to specific "
        n_selects = 0

        if axis1_inds is not None:
            if beam_object.pixel_coordinate_system == "healpix":
                raise ValueError(
                    "axis1_inds cannot be used with healpix coordinate system"
                )

            history_update_string += "parts of first image axis"
            n_selects += 1

            axis1_inds = sorted(set(axis1_inds))
            if min(axis1_inds) < 0 or max(axis1_inds) > beam_object.Naxes1 - 1:
                raise ValueError("axis1_inds must be > 0 and < Naxes1")
            beam_object.Naxes1 = len(axis1_inds)
            beam_object.axis1_array = beam_object.axis1_array[axis1_inds]

            if beam_object.Naxes1 > 1:
                axis1_spacing = np.diff(beam_object.axis1_array)
                if not np.isclose(
                    np.min(axis1_spacing),
                    np.max(axis1_spacing),
                    rtol=beam_object._axis1_array.tols[0],
                    atol=beam_object._axis1_array.tols[1],
                ):
                    warnings.warn(
                        "Selected values along first image axis are "
                        "not evenly spaced. This is not supported by "
                        "the regularly gridded beam fits format"
                    )

            beam_object.data_array = beam_object.data_array[:, :, :, :, :, axis1_inds]
            if beam_object.beam_type == "efield":
                beam_object.basis_vector_array = beam_object.basis_vector_array[
                    :, :, :, axis1_inds
                ]

        if axis2_inds is not None:
            if beam_object.pixel_coordinate_system == "healpix":
                raise ValueError(
                    "axis2_inds cannot be used with healpix coordinate system"
                )

            if n_selects > 0:
                history_update_string += ", parts of second image axis"
            else:
                history_update_string += "parts of second image axis"
            n_selects += 1

            axis2_inds = sorted(set(axis2_inds))
            if min(axis2_inds) < 0 or max(axis2_inds) > beam_object.Naxes2 - 1:
                raise ValueError("axis2_inds must be > 0 and < Naxes2")
            beam_object.Naxes2 = len(axis2_inds)
            beam_object.axis2_array = beam_object.axis2_array[axis2_inds]

            if beam_object.Naxes2 > 1:
                axis2_spacing = np.diff(beam_object.axis2_array)
                if not np.isclose(
                    np.min(axis2_spacing),
                    np.max(axis2_spacing),
                    rtol=beam_object._axis2_array.tols[0],
                    atol=beam_object._axis2_array.tols[1],
                ):
                    warnings.warn(
                        "Selected values along second image axis are "
                        "not evenly spaced. This is not supported by "
                        "the regularly gridded beam fits format"
                    )

            beam_object.data_array = beam_object.data_array[:, :, :, :, axis2_inds, :]
            if beam_object.beam_type == "efield":
                beam_object.basis_vector_array = beam_object.basis_vector_array[
                    :, :, axis2_inds, :
                ]

        if pixels is not None:
            if beam_object.pixel_coordinate_system != "healpix":
                raise ValueError(
                    "pixels can only be used with healpix coordinate system"
                )

            history_update_string += "healpix pixels"
            n_selects += 1

            pix_inds = np.zeros(0, dtype=np.int64)
            for p in pixels:
                if p in beam_object.pixel_array:
                    pix_inds = np.append(
                        pix_inds, np.where(beam_object.pixel_array == p)[0]
                    )
                else:
                    raise ValueError(
                        "Pixel {p} is not present in the pixel_array".format(p=p)
                    )

            pix_inds = sorted(set(pix_inds))
            beam_object.Npixels = len(pix_inds)
            beam_object.pixel_array = beam_object.pixel_array[pix_inds]

            beam_object.data_array = beam_object.data_array[:, :, :, :, pix_inds]
            if beam_object.beam_type == "efield":
                beam_object.basis_vector_array = beam_object.basis_vector_array[
                    :, :, pix_inds
                ]

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if frequencies is None:
                frequencies = beam_object.freq_array[0, freq_chans]
            else:
                frequencies = uvutils._get_iterable(frequencies)
                frequencies = np.sort(
                    list(set(frequencies) | set(beam_object.freq_array[0, freq_chans]))
                )

        if frequencies is not None:
            frequencies = uvutils._get_iterable(frequencies)
            if n_selects > 0:
                history_update_string += ", frequencies"
            else:
                history_update_string += "frequencies"
            n_selects += 1

            freq_inds = np.zeros(0, dtype=np.int64)
            # this works because we only allow one SPW. This will have to be
            # reworked when we support more.
            freq_arr_use = beam_object.freq_array[0, :]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError(
                        "Frequency {f} is not present in the freq_array".format(f=f)
                    )

            freq_inds = sorted(set(freq_inds))
            beam_object.Nfreqs = len(freq_inds)
            beam_object.freq_array = beam_object.freq_array[:, freq_inds]
            beam_object.bandpass_array = beam_object.bandpass_array[:, freq_inds]

            if beam_object.Nfreqs > 1:
                freq_separation = (
                    beam_object.freq_array[0, 1:] - beam_object.freq_array[0, :-1]
                )
                if not np.isclose(
                    np.min(freq_separation),
                    np.max(freq_separation),
                    rtol=beam_object._freq_array.tols[0],
                    atol=beam_object._freq_array.tols[1],
                ):
                    warnings.warn(
                        "Selected frequencies are not evenly spaced. This "
                        "is not supported by the regularly gridded beam fits format"
                    )

            if beam_object.pixel_coordinate_system == "healpix":
                beam_object.data_array = beam_object.data_array[:, :, :, freq_inds, :]
            else:
                beam_object.data_array = beam_object.data_array[
                    :, :, :, freq_inds, :, :
                ]

            if beam_object.antenna_type == "phased_array":
                beam_object.coupling_matrix = beam_object.coupling_matrix[
                    :, :, :, :, :, freq_inds
                ]

            if beam_object.receiver_temperature_array is not None:
                rx_temp_array = beam_object.receiver_temperature_array
                beam_object.receiver_temperature_array = rx_temp_array[:, freq_inds]

            if beam_object.loss_array is not None:
                beam_object.loss_array = beam_object.loss_array[:, freq_inds]

            if beam_object.mismatch_array is not None:
                beam_object.mismatch_array = beam_object.mismatch_array[:, freq_inds]

            if beam_object.s_parameters is not None:
                beam_object.s_parameters = beam_object.s_parameters[:, :, freq_inds]

        if feeds is not None:
            if beam_object.beam_type == "power":
                raise ValueError("feeds cannot be used with power beams")

            feeds = uvutils._get_iterable(feeds)
            if n_selects > 0:
                history_update_string += ", feeds"
            else:
                history_update_string += "feeds"
            n_selects += 1

            feed_inds = np.zeros(0, dtype=np.int64)
            for f in feeds:
                if f in beam_object.feed_array:
                    feed_inds = np.append(
                        feed_inds, np.where(beam_object.feed_array == f)[0]
                    )
                else:
                    raise ValueError(
                        "Feed {f} is not present in the feed_array".format(f=f)
                    )

            feed_inds = sorted(set(feed_inds))
            beam_object.Nfeeds = len(feed_inds)
            beam_object.feed_array = beam_object.feed_array[feed_inds]

            if beam_object.pixel_coordinate_system == "healpix":
                beam_object.data_array = beam_object.data_array[:, :, feed_inds, :, :]
            else:
                beam_object.data_array = beam_object.data_array[
                    :, :, feed_inds, :, :, :
                ]

        if polarizations is not None:
            if beam_object.beam_type == "efield":
                raise ValueError("polarizations cannot be used with efield beams")

            polarizations = uvutils._get_iterable(polarizations)
            if n_selects > 0:
                history_update_string += ", polarizations"
            else:
                history_update_string += "polarizations"
            n_selects += 1

            pol_inds = np.zeros(0, dtype=np.int64)
            for p in polarizations:
                if p in beam_object.polarization_array:
                    pol_inds = np.append(
                        pol_inds, np.where(beam_object.polarization_array == p)[0]
                    )
                else:
                    raise ValueError(
                        "polarization {p} is not present in the"
                        " polarization_array".format(p=p)
                    )

            pol_inds = sorted(set(pol_inds))
            beam_object.Npols = len(pol_inds)
            beam_object.polarization_array = beam_object.polarization_array[pol_inds]

            if len(pol_inds) > 2:
                pol_separation = (
                    beam_object.polarization_array[1:]
                    - beam_object.polarization_array[:-1]
                )
                if np.min(pol_separation) < np.max(pol_separation):
                    warnings.warn(
                        "Selected polarizations are not evenly spaced. This "
                        "is not supported by the regularly gridded beam fits format"
                    )

            if beam_object.pixel_coordinate_system == "healpix":
                beam_object.data_array = beam_object.data_array[:, :, pol_inds, :, :]
            else:
                beam_object.data_array = beam_object.data_array[:, :, pol_inds, :, :, :]

        history_update_string += " using pyuvdata."
        beam_object.history = beam_object.history + history_update_string

        # check if object is self-consistent
        if run_check:
            beam_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return beam_object

    def _convert_from_filetype(self, other):
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        if filetype == "beamfits":
            from . import beamfits

            other_obj = beamfits.BeamFITS()
        else:
            raise ValueError("filetype must be beamfits")
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_beamfits(
        self, filename, run_check=True, check_extra=True, run_check_acceptability=True
    ):
        """
        Read in data from a beamfits file.

        Parameters
        ----------
        filename : str or list of str
            The beamfits file or list of files to read from.
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptabilit : bool
            Option to check acceptable range of the values of
            required parameters after reading in the file.

        """
        from . import beamfits

        if isinstance(filename, (list, tuple)):
            self.read_beamfits(
                filename[0],
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            if len(filename) > 1:
                for f in filename[1:]:
                    beam2 = UVBeam()
                    beam2.read_beamfits(
                        f,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                    )
                    self += beam2
                del beam2
        else:
            beamfits_obj = beamfits.BeamFITS()
            beamfits_obj.read_beamfits(
                filename,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            self._convert_from_filetype(beamfits_obj)
            del beamfits_obj

    def _read_cst_beam_yaml(self, filename):
        """
        Parse a CST beam yaml file.

        Paramters
        ---------
        filename : str
            Filename to parse.

        Returns
        -------
        dict
            Containing all the info from the yaml file.

        """
        import yaml

        with open(filename, "r") as file:
            settings_dict = yaml.safe_load(file)

        required_keys = [
            "telescope_name",
            "feed_name",
            "feed_version",
            "model_name",
            "model_version",
            "history",
            "frequencies",
            "filenames",
            "feed_pol",
        ]

        for key in required_keys:
            if key not in settings_dict:
                raise ValueError(
                    "{key} is a required key in CST settings files "
                    "but is not present.".format(key=key)
                )

        return settings_dict

    def read_cst_beam(
        self,
        filename,
        beam_type="power",
        feed_pol=None,
        rotate_pol=None,
        frequency=None,
        telescope_name=None,
        feed_name=None,
        feed_version=None,
        model_name=None,
        model_version=None,
        history=None,
        x_orientation=None,
        reference_impedance=None,
        extra_keywords=None,
        frequency_select=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in data from a cst file.

        Parameters
        ----------
        filename : str
            Either a settings yaml file or a cst text file or
            list of cst text files to read from. If a list is passed,
            the files are combined along the appropriate axes.

            Settings yaml files must include the following keywords:

                |  - telescope_name (str)
                |  - feed_name (str)
                |  - feed_version (str)
                |  - model_name (str)
                |  - model_version (str)
                |  - history (str)
                |  - frequencies (list(float))
                |  - cst text filenames (list(str)) -- path relative to yaml file
                |  - feed_pol (str) or (list(str))

            and they may include the following optional keywords:

                |  - x_orientation (str): Optional but strongly encouraged!
                |  - ref_imp (float): beam model reference impedance
                |  - sim_beam_type (str): e.g. 'E-farfield'
                |  - all other fields will go into the extra_keywords attribute

            More details and an example are available in the docs
            (cst_settings_yaml.rst).
            Specifying any of the associated keywords to this function will
            override the values in the settings file.
        beam_type : str
            What beam_type to read in ('power' or 'efield').
        feed_pol : str
            The feed or polarization or list of feeds or polarizations the
            files correspond to.
            Defaults to 'x' (meaning x for efield or xx for power beams).
        rotate_pol : bool
            If True, assume the structure in the simulation is symmetric under
            90 degree rotations about the z-axis (so that the y polarization can be
            constructed by rotating the x polarization or vice versa).
            Default: True if feed_pol is a single value or a list with all
            the same values in it, False if it is a list with varying values.
        frequency : float or list of float, optional
            The frequency or list of frequencies corresponding to the filename(s).
            This is assumed to be in the same order as the files.
            If not passed, the code attempts to parse it from the filenames.
        telescope_name : str, optional
            The name of the telescope corresponding to the filename(s).
        feed_name : str, optional
            The name of the feed corresponding to the filename(s).
        feed_version : str, optional
            The version of the feed corresponding to the filename(s).
        model_name : str, optional
            The name of the model corresponding to the filename(s).
        model_version : str, optional
            The version of the model corresponding to the filename(s).
        history : str, optional
            A string detailing the history of the filename(s).
        x_orientation : str, optional
            Orientation of the physical dipole corresponding to what is
            labelled as the x polarization. Options are "east" (indicating
            east/west orientation) and "north" (indicating north/south orientation)
        reference_impedance : float, optional
            The reference impedance of the model(s).
        extra_keywords : dict, optional
            A dictionary containing any extra_keywords.
        frequency_select : list of float, optional
            Only used if the file is a yaml file. Indicates which frequencies
            to include (only read in files for those frequencies)
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as
            required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters after reading in the file.

        """
        from . import cst_beam

        if isinstance(filename, np.ndarray):
            if len(filename.shape) > 1:
                raise ValueError("filename can not be a multi-dimensional array")
            filename = filename.tolist()
        if isinstance(filename, (list, tuple)):
            if len(filename) == 1:
                filename = filename[0]

        if not isinstance(filename, (list, tuple)) and filename.endswith("yaml"):
            settings_dict = self._read_cst_beam_yaml(filename)
            yaml_dir = os.path.dirname(filename)
            cst_filename = [
                os.path.join(yaml_dir, f) for f in settings_dict["filenames"]
            ]

            overriding_keywords = {
                "feed_pol": feed_pol,
                "frequency": frequency,
                "telescope_name": telescope_name,
                "feed_name": feed_name,
                "feed_version": feed_version,
                "model_name": model_name,
                "model_version": model_version,
                "history": history,
            }
            if "ref_imp" in settings_dict:
                overriding_keywords["reference_impedance"] = reference_impedance
            if "x_orientation" in settings_dict:
                overriding_keywords["x_orientation"] = reference_impedance
            for key, val in overriding_keywords.items():
                if val is not None:
                    warnings.warn(
                        "The {key} keyword is set, overriding the "
                        "value in the settings yaml file.".format(key=key)
                    )

            if feed_pol is None:
                feed_pol = settings_dict["feed_pol"]
            if frequency is None:
                frequency = settings_dict["frequencies"]
            if telescope_name is None:
                telescope_name = settings_dict["telescope_name"]
            if feed_name is None:
                feed_name = settings_dict["feed_name"]
            if feed_version is None:
                feed_version = str(settings_dict["feed_version"])
            if model_name is None:
                model_name = settings_dict["model_name"]
            if model_version is None:
                model_version = str(settings_dict["model_version"])
            if history is None:
                history = settings_dict["history"]
            if reference_impedance is None and "ref_imp" in settings_dict:
                reference_impedance = float(settings_dict["ref_imp"])
            if x_orientation is None and "x_orientation" in settings_dict:
                x_orientation = settings_dict["x_orientation"]

            if extra_keywords is None:
                extra_keywords = {}

            known_keys = [
                "telescope_name",
                "feed_name",
                "feed_version",
                "model_name",
                "model_version",
                "history",
                "frequencies",
                "filenames",
                "feed_pol",
                "ref_imp",
                "x_orientation",
            ]
            # One of the standard paramters in the settings yaml file is
            # longer than 8 characters.
            # This causes warnings and straight truncation when writing to
            # beamfits files
            # To avoid these, this defines a standard renaming of that paramter
            rename_extra_keys_map = {"sim_beam_type": "sim_type"}
            for key, value in settings_dict.items():
                if key not in known_keys:
                    if key in rename_extra_keys_map.keys():
                        extra_keywords[rename_extra_keys_map[key]] = value
                    else:
                        extra_keywords[key] = value

            if frequency_select is not None:
                freq_inds = []
                for freq in frequency_select:
                    freq_array = np.array(frequency, dtype=np.float64)
                    close_inds = np.where(
                        np.isclose(
                            freq_array,
                            freq,
                            rtol=self._freq_array.tols[0],
                            atol=self._freq_array.tols[1],
                        )
                    )[0]
                    if close_inds.size > 0:
                        for ind in close_inds:
                            freq_inds.append(ind)
                    else:
                        raise ValueError(f"frequency {freq} not in frequency list")
                freq_inds = np.array(freq_inds)
                frequency = freq_array[freq_inds].tolist()
                cst_filename = np.array(cst_filename)[freq_inds].tolist()
                if len(cst_filename) == 1:
                    cst_filename = cst_filename[0]
                if isinstance(feed_pol, list):
                    if rotate_pol is None:
                        # if a mix of feed pols, don't rotate by default
                        # do this here in case selections confuse this test
                        if np.any(np.array(feed_pol) != feed_pol[0]):
                            rotate_pol = False
                        else:
                            rotate_pol = True
                    feed_pol = np.array(feed_pol)[freq_inds].tolist()

        else:
            cst_filename = filename

        if feed_pol is None:
            # default to x (done here in case it's specified in a yaml file)
            feed_pol = "x"
        if history is None:
            # default to empty (done here in case it's specified in a yaml file)
            history = ""

        if isinstance(frequency, np.ndarray):
            if len(frequency.shape) > 1:
                raise ValueError("frequency can not be a multi-dimensional array")
            frequency = frequency.tolist()
        if isinstance(frequency, (list, tuple)):
            if len(frequency) == 1:
                frequency = frequency[0]

        if isinstance(feed_pol, np.ndarray):
            if len(feed_pol.shape) > 1:
                raise ValueError("frequency can not be a multi-dimensional array")
            feed_pol = feed_pol.tolist()
        if isinstance(feed_pol, (list, tuple)):
            if len(feed_pol) == 1:
                feed_pol = feed_pol[0]

        if isinstance(cst_filename, (list, tuple)):
            if frequency is not None:
                if isinstance(frequency, (list, tuple)):
                    if not len(frequency) == len(cst_filename):
                        raise ValueError(
                            "If frequency and filename are both "
                            "lists they need to be the same length"
                        )
                    freq = frequency[0]
                else:
                    freq = frequency
            else:
                freq = None

            if isinstance(feed_pol, (list, tuple)):
                if not len(feed_pol) == len(cst_filename):
                    raise ValueError(
                        "If feed_pol and filename are both "
                        "lists they need to be the same length"
                    )
                pol = feed_pol[0]
                if rotate_pol is None:
                    # if a mix of feed pols, don't rotate by default
                    if np.any(np.array(feed_pol) != feed_pol[0]):
                        rotate_pol = False
                    else:
                        rotate_pol = True
            else:
                pol = feed_pol
                if rotate_pol is None:
                    rotate_pol = True
            if isinstance(freq, (list, tuple)):
                raise ValueError("frequency can not be a nested list")
            if isinstance(pol, (list, tuple)):
                raise ValueError("feed_pol can not be a nested list")
            self.read_cst_beam(
                cst_filename[0],
                beam_type=beam_type,
                feed_pol=pol,
                rotate_pol=rotate_pol,
                frequency=freq,
                telescope_name=telescope_name,
                feed_name=feed_name,
                feed_version=feed_version,
                model_name=model_name,
                model_version=model_version,
                history=history,
                x_orientation=x_orientation,
                reference_impedance=reference_impedance,
                extra_keywords=extra_keywords,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            for file_i, f in enumerate(cst_filename[1:]):
                if isinstance(f, (list, tuple)):
                    raise ValueError("filename can not be a nested list")

                if isinstance(frequency, (list, tuple)):
                    freq = frequency[file_i + 1]
                elif frequency is not None:
                    freq = frequency
                else:
                    freq = None
                if isinstance(feed_pol, (list, tuple)):
                    pol = feed_pol[file_i + 1]
                else:
                    pol = feed_pol
                beam2 = UVBeam()
                beam2.read_cst_beam(
                    f,
                    beam_type=beam_type,
                    feed_pol=pol,
                    rotate_pol=rotate_pol,
                    frequency=freq,
                    telescope_name=telescope_name,
                    feed_name=feed_name,
                    feed_version=feed_version,
                    model_name=model_name,
                    model_version=model_version,
                    history=history,
                    x_orientation=x_orientation,
                    reference_impedance=reference_impedance,
                    extra_keywords=extra_keywords,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                )
                self += beam2
            del beam2
        else:
            if isinstance(frequency, (list, tuple)):
                raise ValueError("Too many frequencies specified")
            if isinstance(feed_pol, (list, tuple)):
                raise ValueError("Too many feed_pols specified")
            if rotate_pol is None:
                rotate_pol = True
            cst_beam_obj = cst_beam.CSTBeam()
            cst_beam_obj.read_cst_beam(
                cst_filename,
                beam_type=beam_type,
                feed_pol=feed_pol,
                rotate_pol=rotate_pol,
                frequency=frequency,
                telescope_name=telescope_name,
                feed_name=feed_name,
                feed_version=feed_version,
                model_name=model_name,
                model_version=model_version,
                history=history,
                x_orientation=x_orientation,
                reference_impedance=reference_impedance,
                extra_keywords=extra_keywords,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            self._convert_from_filetype(cst_beam_obj)
            del cst_beam_obj

    def read_mwa_beam(
        self,
        h5filepath,
        delays=None,
        amplitudes=None,
        pixels_per_deg=5,
        freq_range=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in the full embedded element MWA beam.

        Note that the azimuth convention in for the UVBeam object is different than the
        azimuth convention in the mwa_pb repo. In that repo, the azimuth convention is
        changed from the native FEKO convention (the FEKO convention is the same as the
        UVBeam convention). The convention in the mwa_pb repo has a different zero point
        and a different direction (so it is in a left handed coordinate system).

        Parameters
        ----------
        h5filepath : str
            path to input h5 file containing the MWA full embedded element spherical
            harmonic modes. Download via
            `wget http://cerberus.mwa128t.org/mwa_full_embedded_element_pattern.h5`
        delays : array of ints
            Array of MWA beamformer delay steps. Should be shape (n_pols, n_dipoles).
        amplitudes : array of floats
            Array of dipole amplitudes, these are absolute values
            (i.e. relatable to physical units).
            Should be shape (n_pols, n_dipoles).
        pixels_per_deg : float
            Number of theta/phi pixels per degree. Sets the resolution of the beam.
        freq_range : array_like of float
            Range of frequencies to include in Hz, defaults to all available
            frequencies. Must be length 2.
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters after reading in the file.

        """
        from . import mwa_beam

        mwabeam_obj = mwa_beam.MWABeam()
        mwabeam_obj.read_mwa_beam(
            h5filepath,
            delays=delays,
            amplitudes=amplitudes,
            pixels_per_deg=pixels_per_deg,
            freq_range=freq_range,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )
        self._convert_from_filetype(mwabeam_obj)
        del mwabeam_obj

    def write_beamfits(
        self,
        filename,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        clobber=False,
    ):
        """
        Write the data to a beamfits file.

        Parameters
        ----------
        filename : str
            The beamfits file to write to.
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters before writing the file.
        check_extra : bool
            Option to check optional parameters as well as
            required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters before writing the file.
        clobber : bool
            Option to overwrite the filename if the file already exists.

        """
        beamfits_obj = self._convert_to_filetype("beamfits")
        beamfits_obj.write_beamfits(
            filename,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            clobber=clobber,
        )
        del beamfits_obj
