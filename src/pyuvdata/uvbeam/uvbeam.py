# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Primary container for radio telescope antenna beams."""

from __future__ import annotations

import copy
import importlib
import os
import warnings
from typing import Literal

import numpy as np
import yaml
from astropy import units
from astropy.utils.data import cache_contents, is_url_in_cache
from docstring_parser import DocstringStyle
from scipy import interpolate, ndimage

from .. import parameter as uvp, utils
from ..docstrings import combine_docstrings, copy_replace_short_description
from ..uvbase import UVBase
from . import _uvbeam, initializers

__all__ = ["UVBeam"]


def _convert_feeds_to_pols(feed_array, calc_cross_pols, x_orientation=None):
    warnings.warn(
        "This method (uvbeam._convert_feeds_to_pols) is deprecated in favor of "
        "utils.pol.convert_feeds_to_pols. This will become an error in version 3.3",
        DeprecationWarning,
    )
    return utils.pol.convert_feeds_to_pols(
        feed_array,
        include_cross_pols=calc_cross_pols,
        x_orientation=x_orientation,
        return_feed_pol_order=True,
    )


class UVBeam(UVBase):
    """
    A class for defining a radio telescope antenna beam.

    Attributes
    ----------
    UVParameter objects :
        For full list see the documentation on ReadTheDocs:
        http://pyuvdata.readthedocs.io/en/latest/.
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
        "az_za_map_coordinates": {
            "description": "scipy map_coordinates interpolation",
            "func": "_interp_az_za_map_coordinates",
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

        desc = (
            "Number of basis vectors used to represent the antenna response in each "
            "pixel. These need not align with the pixel coordinate system or even be "
            "orthogonal. The mapping of these basis vectors to directions aligned with"
            "the pixel coordinate system is contained in the `basis_vector_array`. "
            "The allowed values for this parameter are 2 or 3 (or 1 if beam_type is "
            "'power')."
        )
        self._Naxes_vec = uvp.UVParameter(
            "Naxes_vec", description=desc, expected_type=int, acceptable_vals=[2, 3]
        )

        desc = (
            "Number of orthogonal components required to map each basis vector to "
            "vectors aligned with the pixel coordinate system. This can be equal to or "
            "smaller than `Naxes_vec`. The allowed values for this parameter are 2 or "
            "3. Only required for E-field beams."
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

        desc = "String indicating beam type. Allowed values are 'efield', and 'power'."
        self._beam_type = uvp.UVParameter(
            "beam_type",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["efield", "power"],
        )

        desc = (
            "Beam basis vector components, essentially the mapping between the "
            "directions that the electrical field values are recorded in to the "
            "directions aligned with the pixel coordinate system (or azimuth/zenith "
            "angle for HEALPix beams)."
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
            acceptable_range=(-1, 1),
            tols=1e-3,
        )

        self._Nfeeds = uvp.UVParameter(
            "Nfeeds",
            description="Number of feeds.",
            expected_type=int,
            acceptable_vals=[1, 2],
        )

        desc = "Array of feed labels. shape (Nfeeds). options are: x/y or r/l."
        self._feed_array = uvp.UVParameter(
            "feed_array",
            description=desc,
            expected_type=str,
            form=("Nfeeds",),
            acceptable_vals=["x", "y", "r", "l"],
        )

        desc = (
            "Position angle of a given feed, shape (Nfeeds,), units of radians. "
            "A feed angle of 0 is typically oriented toward zenith for steerable "
            "antennas, otherwise toward north for fixed antennas (e.g., HERA, LWA)."
            'More details on this can be found on the "Conventions" page of the docs.'
        )
        self._feed_angle = uvp.UVParameter(
            "feed_angle",
            description=desc,
            form=("Nfeeds",),
            expected_type=float,
            tols=1e-6,  # 10x (~2pix) single precision limit
        )

        desc = (
            "Antenna mount type, which describes the optics of the antenna in question."
            'Supported options include: "alt-az" (primary rotates in azimuth and'
            'elevation), "equatorial" (primary rotates in hour angle and declination), '
            '"orbiting" (antenna is in motion, and its orientation depends on orbital'
            'parameters), "x-y" (primary rotates first in the plane connecting east, '
            "west, and zenith, and then perpendicular to that plane), "
            '"alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary '
            'mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree '
            'tertiary mirror), "phased" (antenna is "electronically steered" by '
            'summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna '
            'beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other" '
            '(also referred to in some formats as "bizarre"). See the "Conventions" '
            "page of the documentation for further details."
        )
        self._mount_type = uvp.UVParameter(
            "mount_type",
            description=desc,
            expected_type=str,
            acceptable_vals=[
                "alt-az",
                "equatorial",
                "orbiting",
                "x-y",
                "alt-az+nasmyth-r",
                "alt-az+nasmyth-l",
                "phased",
                "fixed",
                "other",
            ],
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

        desc = "Array of frequencies, center of the channel, shape (Nfreqs,), units Hz."
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,
        )  # mHz

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
            "given by data_normalization. The shape depends on the beam_type and "
            "pixel_coordinate_system. If it is a 'healpix' beam, the shape is: "
            "(Naxes_vec, Nfeeds or Npols, Nfreqs, Npixels), if it is not a healpix "
            "beam it is (Naxes_vec, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)."
        )
        self._data_array = uvp.UVParameter(
            "data_array",
            description=desc,
            expected_type=complex,
            form=("Naxes_vec", "Nfeeds", "Nfreqs", "Naxes2", "Naxes1"),
            tols=1e-3,
        )

        desc = (
            "Frequency dependence of the beam. Depending on the data_normalization, "
            "this may contain only the frequency dependence of the receiving "
            "chain ('physical' normalization) or all the frequency dependence "
            "('peak' normalization). Shape is (Nfreqs,)."
        )
        self._bandpass_array = uvp.UVParameter(
            "bandpass_array",
            description=desc,
            expected_type=float,
            form=("Nfreqs",),
            tols=1e-3,
        )

        # --------- metadata -------------
        self._telescope_name = uvp.UVParameter(
            "telescope_name",
            description="Name of telescope (string)",
            form="str",
            expected_type=str,
        )

        self._feed_name = uvp.UVParameter(
            "feed_name",
            description="Name of physical feed (string)",
            form="str",
            expected_type=str,
        )

        self._feed_version = uvp.UVParameter(
            "feed_version",
            description="Version of physical feed (string)",
            form="str",
            expected_type=str,
        )

        self._model_name = uvp.UVParameter(
            "model_name",
            description="Name of beam model (string)",
            form="str",
            expected_type=str,
        )

        self._model_version = uvp.UVParameter(
            "model_version",
            description="Version of beam model (string)",
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
            "system, options are: n-e or x-y"
        )
        self._element_coordinate_system = uvp.UVParameter(
            "element_coordinate_system",
            required=False,
            description=desc,
            expected_type=str,
            acceptable_vals=["n-e", "x-y"],
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
            "shape: (Nelements, Nelements, Nfeeds, Nfeeds, Nfreqs)."
        )
        self._coupling_matrix = uvp.UVParameter(
            "coupling_matrix",
            required=False,
            description=desc,
            form=("Nelements", "Nelements", "Nfeeds", "Nfeeds", "Nfreqs"),
            expected_type=complex,
        )

        # -------- extra, non-required parameters ----------
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

        desc = "Array of receiver temperatures, units K. Shape (Nfreqs,)."
        self._receiver_temperature_array = uvp.UVParameter(
            "receiver_temperature_array",
            required=False,
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,
        )

        desc = "Array of antenna losses, units dB? Shape (Nfreqs,)."
        self._loss_array = uvp.UVParameter(
            "loss_array",
            required=False,
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,
        )

        desc = "Array of antenna-amplifier mismatches, units ? Shape (Nfreqs,)."
        self._mismatch_array = uvp.UVParameter(
            "mismatch_array",
            required=False,
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,
        )

        desc = (
            "S parameters of receiving chain, ordering: s11, s12, s21, s22. see "
            "https://en.wikipedia.org/wiki/Scattering_parameters#Two-Port_S-Parameters"
            "Shape (4, Nfreqs)."
        )
        self._s_parameters = uvp.UVParameter(
            "s_parameters",
            required=False,
            description=desc,
            form=(4, "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )

        desc = (
            "List of strings containing the unique basenames (not the full path) of "
            "input files."
        )
        self._filename = uvp.UVParameter(
            "filename", required=False, description=desc, expected_type=str
        )

        super().__init__()

    def __getattr__(self, __name):
        """Handle getting old attributes."""
        if __name == "x_orientation":
            warnings.warn(
                "The UVBeam.x_orientation attribute is deprecated, and has "
                "been superseded by Telescope.feed_angle and Telescope.feed_array. "
                "This will become an error in version 3.4. To set the equivalent "
                "value in the future, you can substitute accessing this parameter "
                "with a call to UVBeam.get_x_orientation_from_feeds().",
                DeprecationWarning,
            )
            return self.get_x_orientation_from_feeds()

        return super().__getattribute__(__name)

    def __setattr__(self, __name, __value):
        """Handle setting old attributes."""
        if __name == "x_orientation":
            warnings.warn(
                "The UVBeam.x_orientation attribute is deprecated, and has "
                "been superseded by UVBeam.feed_angle and UVBeam.feed_array. "
                "This will become an error in version 3.4. To get the equivalent "
                "value in the future, you can substitute accessing this parameter "
                "with a call to UVBeam.set_feeds_from_x_orientation().",
                DeprecationWarning,
            )
            if __value is not None:
                return self.set_feeds_from_x_orientation(__value)

        return super().__setattr__(__name, __value)

    def _fix_feeds(self):
        if self.beam_type == "power" and self.Nfeeds is None:
            warnings.warn(
                "Feed information now required for power beams, will default to "
                "linear feeds with x-orientation aligned to the east (populated based "
                "on what is preesnt in UVBeam.polarization_array). This will become an "
                "error in version 3.4.",
                DeprecationWarning,
            )
            self.set_feeds_from_x_orientation(x_orientation="east")
        if self.feed_array is not None and (
            ("e" in self.feed_array) or ("n" in self.feed_array)
        ):
            warnings.warn(
                'Support for physically oriented feeds (e.g., "n", and "e") has been '
                'deprecated, with the only allowed options now available being "x", '
                '"y", "l", or "r". The physical orientation of the feed is now '
                "recorded in the UVBeam.feed_angle parameter (which must now be set on "
                "all UVBeam objects). This will become an error in version 3.4.",
                DeprecationWarning,
            )
            feed_map = {"e": "x", "n": "y"}
            feed_ang_map = {
                "e": np.pi / 2,
                "n": 0.0,
                "x": np.pi / 2,
                "y": 0.0,
                "r": 0.0,
                "l": 0.0,
            }
            if self.feed_angle is None:
                self.feed_angle = [feed_ang_map.get(f, 0.0) for f in self.feed_array]

            for idx, feed in enumerate(self.feed_array):
                if feed in feed_map:
                    self.feed_array[idx] = feed_map[feed]
                    self.feed_angle[idx] = feed_ang_map[feed]
        if self.feed_angle is None:
            warnings.warn(
                "The feed_angle parameter must be set for UVBeam objects, setting "
                "based on x-polarization feeds being aligned to east (and all others "
                "to north). This will become an error in version 3.4.",
                DeprecationWarning,
            )
            self.set_feeds_from_x_orientation("east")
        if self.mount_type is None:
            warnings.warn(
                "The mount_type parameter must be set for UVBeam objects, setting to "
                '"fixed" by default for now. This will become an error in version 3.4.',
                DeprecationWarning,
            )
            self.mount_type = "fixed"

    def get_x_orientation_from_feeds(self) -> Literal["east", "north", None]:
        """
        Get x-orientation equivalent value based on feed information.

        Returns
        -------
        x_orientation : str
            One of "east", "north", or None, based on values present in
            UVBeam.feed_array and UVBeam.feed_angle.
        """
        self._fix_feeds()
        return utils.pol.get_x_orientation_from_feeds(
            feed_array=self.feed_array,
            feed_angle=self.feed_angle,
            tols=self._feed_angle.tols,
        )

    def set_feeds_from_x_orientation(self, x_orientation):
        """
        Set feed information based on x-orientation value.

        Populates newer parameters describing feed-orientation (`UVBeam.feed_array`
        and `UVBeam.feed_angle`) based on the "older" x-orientation string. Note that
        this method will overwrite any previously populated values.

        Parameters
        ----------
        x_orientation : str
            String describing how the x-orientation is oriented. Must be either "north"/
            "n"/"ns" (x-polarization of antenna has a position angle of 0 degrees with
            respect to zenith/north) or "east"/"e"/"ew" (x-polarization of antenna has a
            position angle of 90 degrees with respect to zenith/north).
        """
        self.Nfeeds, self.feed_array, self.feed_angle = (
            utils.pol.get_feeds_from_x_orientation(
                x_orientation=x_orientation,
                feeds=self.feed_array,
                polarization_array=self.polarization_array,
                nants=0,
            )
        )

        # Do a quick compatibility check w/ the old feed types.
        self._fix_feeds()

    @staticmethod
    @combine_docstrings(initializers.new_uvbeam, style=DocstringStyle.NUMPYDOC)
    def new(**kwargs):
        """
        Create a new UVBeam object.

        All parameters are passed through to
        the :func:`~pyuvdata.uvbeam.initializers.new_uvbeam` function.

        Returns
        -------
        UVBeam
            A new UVBeam object.
        """
        return initializers.new_uvbeam(**kwargs)

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
                self._data_array.form = ("Naxes_vec", "Npols", "Nfreqs", "Npixels")
            else:
                self._data_array.form = ("Naxes_vec", "Nfeeds", "Nfreqs", "Npixels")
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
                    "Npols",
                    "Nfreqs",
                    "Naxes2",
                    "Naxes1",
                )
            else:
                self._data_array.form = (
                    "Naxes_vec",
                    "Nfeeds",
                    "Nfreqs",
                    "Naxes2",
                    "Naxes1",
                )

    def _set_efield(self):
        """Set beam_type to 'efield' and adjust required parameters."""
        self.beam_type = "efield"
        self._Naxes_vec.acceptable_vals = [2, 3]
        self._Ncomponents_vec.required = True
        self._basis_vector_array.required = True
        self._Npols.required = False
        self._polarization_array.required = False
        self._data_array.expected_type = uvp._get_generic_type(complex)
        # call set_cs_params to fix data_array form
        self._set_cs_params()

    def _set_power(self):
        """Set beam_type to 'power' and adjust required parameters."""
        self.beam_type = "power"
        self._Naxes_vec.acceptable_vals = [1, 2, 3]
        self._basis_vector_array.required = False
        self._Ncomponents_vec.required = False
        self._Npols.required = True
        self._polarization_array.required = True

        # If cross pols are included, the power beam is complex. Otherwise it's real
        self._data_array.expected_type = uvp._get_generic_type(float)
        for pol in self.polarization_array:
            if pol in [-3, -4, -7, -8]:
                self._data_array.expected_type = uvp._get_generic_type(complex)

        # call set_cs_params to fix data_array form
        self._set_cs_params()

    def _set_simple(self):
        """Set antenna_type to 'simple' and adjust required parameters."""
        self.antenna_type = "simple"
        self._Nelements.required = False
        self._element_coordinate_system.required = False
        self._element_location_array.required = False
        self._delay_array.required = False
        self._gain_array.required = False
        self._coupling_matrix.required = False

    def _set_phased_array(self):
        """Set antenna_type to 'phased_array' and adjust required parameters."""
        self.antenna_type = "phased_array"
        self._Nelements.required = True
        self._element_coordinate_system.required = True
        self._element_location_array.required = True
        self._delay_array.required = True
        self._gain_array.required = True
        self._coupling_matrix.required = True

    def _fix_auto_power(self):
        """Remove imaginary component of auto polarization power beams."""
        if self.beam_type != "power" or self.polarization_array is None:
            warnings.warn(
                "Cannot use _fix_autos if beam_type is not 'power', or "
                "polarization_array is None. Leaving data_array untouched."
            )
            return

        auto_pol_list = ["xx", "yy", "rr", "ll", "pI", "pQ", "pU", "pV"]
        pol_screen = np.array(
            [
                utils.POL_NUM2STR_DICT[pol] in auto_pol_list
                for pol in self.polarization_array
            ]
        )

        # Set any auto pol beams to be real-only by taking the absolute value
        if np.all(pol_screen):
            # If we only have auto pol beams the data_array should be float not complex
            self.data_array = np.abs(self.data_array)
        elif np.any(pol_screen):
            self.data_array[:, pol_screen] = np.abs(self.data_array[:, pol_screen])

    def _check_auto_power(self, *, fix_auto_power=False, warn_tols=(0, 0)):
        """
        Check for complex auto polarization power beams.

        Parameters
        ----------
        fix_auto_power : bool
            If auto polarization power beams with imaginary values are found,
            fix those values so that they are real-only in data_array.
        warn_tols : tuple of float
            Tolerances (relative, absolute) to use in comparing max imaginary part of
            auto polarization power beams to zero (passed to numpy.isclose). If the max
            imaginary part is close to zero within the tolerances and fix_auto_power is
            True, silently fix them to be zero and do not warn.

        """
        if self.beam_type != "power" or self.polarization_array is None:
            warnings.warn(
                "Cannot use _check_auto_power if beam_type is not 'power', or "
                "polarization_array is None."
            )
            return

        # Verify here that the auto polarization power beams do not have any
        # imaginary components
        auto_pol_list = ["xx", "yy", "rr", "ll", "pI", "pQ", "pU", "pV"]
        pol_screen = np.array(
            [
                utils.POL_NUM2STR_DICT[pol] in auto_pol_list
                for pol in self.polarization_array
            ]
        )
        pol_axis = 1
        if np.any(pol_screen) and np.any(
            np.iscomplex(np.rollaxis(self.data_array, pol_axis)[pol_screen])
        ):
            max_imag = np.max(
                np.abs(np.imag(np.rollaxis(self.data_array, pol_axis)[pol_screen]))
            )
            if fix_auto_power:
                if not np.isclose(max_imag, 0, rtol=warn_tols[0], atol=warn_tols[1]):
                    warnings.warn(
                        "Fixing auto polarization power beams to be be real-only, "
                        "after some imaginary values were detected in data_array. "
                        f"Largest imaginary component was {max_imag}."
                    )
                self._fix_auto_power()
            else:
                raise ValueError(
                    "Some auto polarization power beams have non-real values in "
                    f"data_array. Largest imaginary component was {max_imag}. "
                    "You can attempt to fix this by setting fix_auto_power=True."
                )

    def check(
        self,
        *,
        check_extra=True,
        run_check_acceptability=True,
        check_auto_power=False,
        fix_auto_power=False,
    ):
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
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array. Ignored if
            check_auto_power is False.

        """
        # Do a quick compatibility check w/ the old feed types.
        self._fix_feeds()

        # first make sure the required parameters and forms are set properly
        # _set_cs_params is called by _set_efield/_set_power
        if self.beam_type == "efield":
            self._set_efield()
        elif self.beam_type == "power":
            self._set_power()

        if self.antenna_type == "simple":
            self._set_simple()
        elif self.antenna_type == "phased_array":
            self._set_phased_array()

        if (
            self.beam_type == "power"
            and run_check_acceptability
            and check_auto_power
            and self.polarization_array is not None
        ):
            self._check_auto_power(fix_auto_power=fix_auto_power)

        # first run the basic check from UVBase
        super().check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # check that basis_vector_array are not longer than 1
        if self.basis_vector_array is not None and np.max(
            np.linalg.norm(self.basis_vector_array, axis=1)
        ) > (1 + 1e-15):
            raise ValueError("basis vectors must have lengths of 1 or less.")

        # Check if the interpolation points are evenly-spaced
        if self.pixel_coordinate_system == "az_za":
            for i, ax_param in enumerate((self._axis1_array, self._axis2_array)):
                ax = ax_param.value
                if len(ax) < 3:
                    continue

                if not utils.tools._test_array_constant_spacing(ax, tols=ax_param.tols):
                    raise ValueError(
                        f"axis{i + 1}_array must be evenly spaced in az_za coordinates."
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
            max_val = abs(self.data_array[:, :, i]).max()
            self.data_array[:, :, i, :] /= max_val
            self.bandpass_array[i] *= max_val
        self.data_normalization = "peak"

    def efield_to_power(
        self,
        *,
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
        # Do a quick compatibility check w/ the old feed types.
        self._fix_feeds()

        if self.beam_type != "efield":
            raise ValueError("beam_type must be efield")

        if self.antenna_type == "phased_array":
            raise NotImplementedError(
                "Conversion to power is not yet implemented for phased_array antennas"
            )

        if inplace:
            beam_object = self
        else:
            beam_object = self.copy()

        if beam_object.Nfeeds == 1:
            # There are no cross pols with one feed. Set this so the power beam is real
            calc_cross_pols = False

        beam_object.polarization_array, feed_pol_order = (
            utils.pol.convert_feeds_to_pols(
                beam_object.feed_array,
                include_cross_pols=calc_cross_pols,
                x_orientation=beam_object.get_x_orientation_from_feeds(),
                return_feed_pol_order=True,
            )
        )
        beam_object.Npols = beam_object.polarization_array.size

        efield_data = beam_object.data_array
        efield_naxes_vec = beam_object.Naxes_vec
        if not keep_basis_vector:
            beam_object.Naxes_vec = 1

        # adjust requirements, fix data_array form
        beam_object._set_power()
        power_data = np.zeros(
            beam_object._data_array.expected_shape(beam_object), dtype=np.complex128
        )

        if keep_basis_vector:
            for pol_i, pair in enumerate(feed_pol_order):
                power_data[:, pol_i] = efield_data[:, pair[0]] * np.conj(
                    efield_data[:, pair[1]]
                )
        else:
            for pol_i, pair in enumerate(feed_pol_order):
                if efield_naxes_vec == 2:
                    for comp_i in range(2):
                        power_data[0, pol_i] += (
                            (efield_data[0, pair[0]] * np.conj(efield_data[0, pair[1]]))
                            * beam_object.basis_vector_array[0, comp_i] ** 2
                            + (
                                efield_data[1, pair[0]]
                                * np.conj(efield_data[1, pair[1]])
                            )
                            * beam_object.basis_vector_array[1, comp_i] ** 2
                            + (
                                efield_data[0, pair[0]]
                                * np.conj(efield_data[1, pair[1]])
                                + efield_data[1, pair[0]]
                                * np.conj(efield_data[0, pair[1]])
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

        if not calc_cross_pols:
            max_abs_imag = np.max(np.abs(power_data.imag))
            if not np.isclose(
                max_abs_imag,
                0,
                rtol=beam_object._data_array.tols[0],
                atol=beam_object._data_array.tols[1],
            ):  # pragma: no cover
                warnings.warn(
                    "The calculated power beam has a non-zero imaginary component "
                    f"(the maximum absolute imaginary component is {max_abs_imag}). "
                    "The power beam should be real because the crosspols are not "
                    "calculated. Setting the power beam equal to the real part of the "
                    "calculated power beam."
                )
            power_data = power_data.real

        beam_object.data_array = power_data
        if not keep_basis_vector:
            beam_object.basis_vector_array = None
            beam_object.Ncomponents_vec = None

        if calc_cross_pols:
            # Sometimes the auto pol beams can have a small complex part due to
            # numerical precision errors. Fix that (with warnings if the complex part
            # is larger than the tolerances).
            beam_object._check_auto_power(
                fix_auto_power=True, warn_tols=beam_object._data_array.tols
            )

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

    def _construct_mueller(self, *, jones, pol_index1, pol_index2):
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
        *,
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
        # Do a quick compatibility check w/ the old feed types.
        self._fix_feeds()

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
            (1, len(pol_strings), _sh[-2], _sh[-1]), dtype=np.complex128
        )
        beam_object.polarization_array = np.array(
            [
                utils.polstr2num(
                    ps.upper(), x_orientation=self.get_x_orientation_from_feeds()
                )
                for ps in pol_strings
            ]
        )

        for fq_i in range(Nfreqs):
            jones = np.zeros((_sh[-1], 2, 2), dtype=np.complex128)
            pol_strings = ["pI", "pQ", "pU", "pV"]
            jones[:, 0, 0] = efield_data[0, 0, fq_i, :]
            jones[:, 0, 1] = efield_data[0, 1, fq_i, :]
            jones[:, 1, 0] = efield_data[1, 0, fq_i, :]
            jones[:, 1, 1] = efield_data[1, 1, fq_i, :]

            for pol_i in range(len(pol_strings)):
                power_data[:, pol_i, fq_i, :] = self._construct_mueller(
                    jones=jones, pol_index1=pol_i, pol_index2=pol_i
                )
        assert not np.any(np.iscomplex(power_data)), (
            "The calculated pstokes beams are complex but should be real. This is a "
            "bug, please report it in our issue log"
        )
        power_data = np.abs(power_data)

        if self.pixel_coordinate_system != "healpix":
            power_data = power_data.reshape(power_data.shape[:-1] + (Naxes2, Naxes1))
        beam_object.data_array = power_data
        beam_object.polarization_array = np.array(
            [
                utils.polstr2num(
                    ps.upper(), x_orientation=self.get_x_orientation_from_feeds()
                )
                for ps in pol_strings
            ]
        )
        beam_object.Naxes_vec = 1
        beam_object._set_power()

        history_update_string = (
            " Converted from efield to pseudo-stokes power using pyuvdata."
        )
        beam_object.Npols = beam_object.Nfeeds**2
        beam_object.history = beam_object.history + history_update_string
        beam_object.basis_vector_array = None
        beam_object.Ncomponents_vec = None

        if run_check:
            beam_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return beam_object

    def _interp_freq(self, freq_array, *, kind="linear", tol=1.0):
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
            The array of interpolated data values, shape: (Naxes_vec, Nfeeds or Npols,
            freq_array.size, Npixels or (Naxis2, Naxis1))
        interp_bandpass : array_like of float
            The interpolated bandpass. shape: (freq_array.size)
        interp_coupling_matrix : array_like of complex, optional
            The interpolated coupling matrix. Only returned if antenna_type is
            "phased_array".
            shape: (Nelements, Nelements, Nfeeds, Nfeeds, freq_array.size)

        """
        assert isinstance(freq_array, np.ndarray)
        assert freq_array.ndim == 1

        # use the beam at nearest neighbors if kind is 'nearest'
        if kind == "nearest":
            freq_dists = np.abs(self.freq_array[np.newaxis] - freq_array.reshape(-1, 1))
            nearest_inds = np.argmin(freq_dists, axis=1)
            interp_arrays = [
                self.data_array[:, :, nearest_inds, :],
                self.bandpass_array[nearest_inds],
            ]
            if self.antenna_type == "phased_array":
                interp_arrays.append(self.coupling_matrix[..., nearest_inds])

        # otherwise interpolate the beam
        else:
            beam_freqs = copy.copy(self.freq_array)
            data_axis = 2
            bandpass_axis = 0

            if self.Nfreqs == 1:
                raise ValueError("Only one frequency in UVBeam so cannot interpolate.")

            if np.min(freq_array) < np.min(self.freq_array) or np.max(
                freq_array
            ) > np.max(self.freq_array):
                raise ValueError(
                    "at least one interpolation frequency is outside of "
                    "the UVBeam freq_array range. Beam frequency range is: "
                    f"{[np.min(self.freq_array), np.max(self.freq_array)]}, "
                    "interpolation frequency range is: "
                    f"{[np.min(freq_array), np.max(freq_array)]}"
                )

            def get_lambda(real_lut, imag_lut=None):
                # Returns function objects for interpolation reuse
                if imag_lut is None:
                    return lambda freqs: real_lut(freqs)
                else:
                    return lambda freqs: (real_lut(freqs) + 1j * imag_lut(freqs))

            interp_arrays = []
            for data, ax in zip(
                [self.data_array, self.bandpass_array],
                [data_axis, bandpass_axis],
                strict=True,
            ):
                if np.iscomplexobj(data):
                    # interpolate real and imaginary parts separately
                    real_lut = interpolate.interp1d(
                        beam_freqs, data.real, kind=kind, axis=ax
                    )
                    imag_lut = interpolate.interp1d(
                        beam_freqs, data.imag, kind=kind, axis=ax
                    )
                    lut = get_lambda(real_lut, imag_lut)
                else:
                    lut = interpolate.interp1d(beam_freqs, data, kind=kind, axis=ax)
                    lut = get_lambda(lut)

                interp_arrays.append(lut(freq_array))

            if self.antenna_type == "phased_array":
                # interpolate real and imaginary parts separately
                real_lut = interpolate.interp1d(
                    beam_freqs, self.coupling_matrix.real, kind=kind, axis=-1
                )
                imag_lut = interpolate.interp1d(
                    beam_freqs, self.coupling_matrix.imag, kind=kind, axis=-1
                )
                lut = get_lambda(real_lut, imag_lut)

                interp_arrays.append(lut(freq_array))

        exp_ndim = 1

        assert interp_arrays[1].ndim == exp_ndim

        return tuple(interp_arrays)

    def _handle_input_for_freq_interpolation(
        self, freq_array, *, freq_interp_kind="cubic", freq_interp_tol=1.0
    ):
        """
        Handle "interp" inputs prior to calling "_interp_freq".

        This helper function that is used by the az_za and healpix beam
        interpolation methods. Prior to performing interpolation along the
        azimuth/zenith angle axes, this function checks if the provided frequency
        array is not None. If it is not None, it performs frequency interpolation
        using "_interp_freq". If the frequency array is None, it returns the
        intrinsic data array and bandpass array.
        """
        if freq_array is not None:
            assert isinstance(freq_array, np.ndarray)
            interp_arrays = self._interp_freq(
                freq_array, kind=freq_interp_kind, tol=freq_interp_tol
            )
            if self.antenna_type == "phased_array":
                (input_data_array, interp_bandpass, interp_coupling_matrix) = (
                    interp_arrays
                )
            else:
                input_data_array, interp_bandpass = interp_arrays
                interp_coupling_matrix = None
            input_nfreqs = freq_array.size
        else:
            input_data_array = self.data_array
            input_nfreqs = self.Nfreqs
            freq_array = self.freq_array
            interp_bandpass = self.bandpass_array[0]
            if self.antenna_type == "phased_array":
                interp_coupling_matrix = self.coupling_matrix
            else:
                interp_coupling_matrix = None

        return (
            input_data_array,
            interp_bandpass,
            interp_coupling_matrix,
            input_nfreqs,
            freq_array,
        )

    def _check_interpolation_domain(self, az_array, za_array, phi_use, theta_use):
        """Check if the interpolation domain is covered by the intrinsic data array."""
        max_axis_diff = max(np.diff(self.axis1_array)[0], np.diff(self.axis2_array)[0])
        za_sq_dist = np.full(len(za_array), np.inf)
        az_sq_dist = np.full(len(az_array), np.inf)
        if (len(theta_use) + len(phi_use)) > len(za_array):
            # If there are fewer interpolation points than grid points, go
            # through the grid points one-by-one to spot any outliers
            for idx in range(az_array.size):
                za_sq_dist[idx] = np.min((theta_use - za_array[idx]) ** 2.0)
                az_sq_dist[idx] = np.min((phi_use - az_array[idx]) ** 2.0)
        else:
            # Otherwise, if we have lots of interpolation points, then it's faster
            # to evaluate the grid steps one-by-one.
            for theta_val in theta_use:
                temp_arr = np.square(za_array - theta_val)
                za_sq_dist = np.where(za_sq_dist > temp_arr, temp_arr, za_sq_dist)

            for phi_val in phi_use:
                temp_arr = np.square(az_array - phi_val)
                az_sq_dist = np.where(az_sq_dist > temp_arr, temp_arr, az_sq_dist)

        if np.any(np.sqrt(az_sq_dist + za_sq_dist) > (max_axis_diff * 2.0)):
            if np.any(np.sqrt(za_sq_dist) > (max_axis_diff * 2.0)):
                msg = " The zenith angles values are outside UVBeam coverage."
            elif np.any(np.sqrt(az_sq_dist) > (max_axis_diff * 2.0)):
                msg = " The azimuth values are outside UVBeam coverage."

            raise ValueError(
                "at least one interpolation location "
                "is outside of the UVBeam pixel coverage." + msg
            )

    def _prepare_coordinate_data(self, input_data_array):
        """Prepare coordinate data for interpolation functions."""
        freq_axis = 2
        axis1_diff = np.diff(self.axis1_array)[0]
        phi_length = np.abs(self.axis1_array[0] - self.axis1_array[-1]) + axis1_diff
        phi_vals, theta_vals = self.axis1_array, self.axis2_array

        if np.isclose(phi_length, 2 * np.pi, atol=axis1_diff):
            # phi wraps around, extend array in each direction to improve interpolation
            extend_length = 3
            phi_use = np.concatenate(
                (
                    phi_vals[0] + (np.arange(-extend_length, 0) * axis1_diff),
                    phi_vals,
                    phi_vals[-1] + (np.arange(1, extend_length + 1) * axis1_diff),
                )
            )

            low_slice = input_data_array[..., :extend_length]
            high_slice = input_data_array[..., -1 * extend_length :]

            data_use = np.concatenate(
                (high_slice, input_data_array, low_slice), axis=freq_axis + 2
            )
        else:
            phi_use = phi_vals
            data_use = input_data_array

        return data_use, phi_use, theta_vals

    def _prepare_polarized_inputs(self, polarizations):
        """Prepare inputs for polarized interpolation functions."""
        # Npols is only defined for power beams.  For E-field beams need Nfeeds.
        if self.beam_type == "power":
            # get requested polarization indices
            if polarizations is None:
                Npol_feeds = self.Npols
                pol_inds = np.arange(Npol_feeds)
            else:
                pols = [
                    utils.polstr2num(
                        p, x_orientation=self.get_x_orientation_from_feeds()
                    )
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

        return Npol_feeds, pol_inds

    def _prepare_basis_vector_array(self, npoints):
        """Prepare basis vector array for interpolation functions."""
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

        return interp_basis_vector

    def _interp_az_za_rect_spline(
        self,
        *,
        az_array,
        za_array,
        freq_array,
        freq_interp_kind="cubic",
        freq_interp_tol=1.0,
        polarizations=None,
        reuse_spline=False,
        spline_opts=None,
        check_azza_domain: bool = True,
        return_basis_vector: bool = False,
    ):
        """
        Interpolate in az_za coordinate system using RectBivariateSpline.

        Uses the :func:`scipy.interpolate.RectBivariateSpline` function to perform
        interpolation in the azimuth-zenith angle coordinate system.

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
            Option to save the interpolation functions for reuse, default is False.
        spline_opts : dict, optional
            Option to specify (kx, ky, s) for numpy.RectBivariateSpline. Note that
            this parameter is ignored if this function has been called previously
            on this object instance and reuse_spline is True.
        check_azza_domain : bool
            Whether to check the domain of az/za to ensure that they are covered by the
            intrinsic data array. Checking them can be quite computationally expensive.
        return_basis_vector : bool
            Whether to return the interpolated basis vectors. Prior to v3.1.1 these
            were always returned. Now they are not by default.

        Returns
        -------
        interp_data : array_like of float or complex
            The array of interpolated data values,
            shape: (Naxes_vec, Nfeeds or Npols, freq_array.size, az_array.size)
        interp_basis_vector : array_like of float
            The array of interpolated basis vectors,
            shape: (Naxes_vec, Ncomponents_vec, az_array.size)
        interp_bandpass : array_like of float
            The interpolated bandpass. shape: (freq_array.size,)
        interp_coupling_matrix : array_like of complex, optional
            The interpolated coupling matrix. Only returned if antenna_type is
            "phased_array".
            shape: (Nelements, Nelements, Nfeeds, Nfeeds, freq_array.size)

        """
        if self.pixel_coordinate_system != "az_za":
            raise ValueError(
                "pixel_coordinate_system must be 'az_za' to use this interpolation "
                "function"
            )

        # Perform initial frequency interpolation to get the data array
        (
            input_data_array,
            interp_bandpass,
            interp_coupling_matrix,
            input_nfreqs,
            freq_array,
        ) = self._handle_input_for_freq_interpolation(
            freq_array=freq_array,
            freq_interp_kind=freq_interp_kind,
            freq_interp_tol=freq_interp_tol,
        )

        # If az_array and za_array are not provided, return the interpolated data
        if az_array is None or za_array is None:
            interp_arrays = [input_data_array, self.basis_vector_array, interp_bandpass]
            if self.antenna_type == "phased_array":
                interp_arrays.append(interp_coupling_matrix)
            return tuple(interp_arrays)

        freq_axis = 2

        # Check input arrays
        assert isinstance(az_array, np.ndarray)
        assert isinstance(za_array, np.ndarray)
        assert az_array.ndim == 1
        assert az_array.shape == za_array.shape
        assert input_data_array.shape[freq_axis] == input_nfreqs

        # Get the data type
        if np.iscomplexobj(input_data_array):
            data_type = np.complex128
        else:
            data_type = np.float64

        # Prepare the data for interpolation
        data_use, phi_use, theta_use = self._prepare_coordinate_data(input_data_array)

        # Prepare basis functions
        if return_basis_vector:
            interp_basis_vector = self._prepare_basis_vector_array(az_array.size)
        else:
            interp_basis_vector = None

        # Get number of polarizations and indices
        Npol_feeds, pol_inds = self._prepare_polarized_inputs(polarizations)

        # Check if the interpolation points are within the data array
        if check_azza_domain:
            self._check_interpolation_domain(az_array, za_array, phi_use, theta_use)

        interp_data = np.zeros(
            (self.Naxes_vec, Npol_feeds, input_nfreqs, az_array.size), dtype=data_type
        )

        def get_lambda(real_lut, imag_lut=None, **kwargs):
            # Returns function objects for interpolation reuse
            if imag_lut is None:
                return lambda za, az: real_lut(za, az, **kwargs)
            else:
                return lambda za, az: (
                    real_lut(za, az, **kwargs) + 1j * imag_lut(za, az, **kwargs)
                )

        if spline_opts is None or not isinstance(spline_opts, dict):
            spline_opts = {}
        if reuse_spline and not hasattr(self, "saved_interp_functions"):
            int_dict = {}
            self.saved_interp_functions = int_dict

        for index3 in range(input_nfreqs):
            freq = freq_array[index3]
            for index0 in range(self.Naxes_vec):
                for pol_return_ind, index2 in enumerate(pol_inds):
                    do_interp = True
                    key = (freq, index2, index0)

                    if reuse_spline and key in self.saved_interp_functions:
                        do_interp = False
                        lut = self.saved_interp_functions[key]

                    if do_interp:
                        data_inds = (index0, index2, index3)

                        if np.iscomplexobj(data_use):
                            # interpolate real and imaginary parts separately
                            real_lut = interpolate.RectBivariateSpline(
                                theta_use,
                                phi_use,
                                data_use[data_inds].real,
                                **spline_opts,
                            )
                            imag_lut = interpolate.RectBivariateSpline(
                                theta_use,
                                phi_use,
                                data_use[data_inds].imag,
                                **spline_opts,
                            )
                            lut = get_lambda(real_lut, imag_lut, grid=False)
                        else:
                            lut = interpolate.RectBivariateSpline(
                                theta_use, phi_use, data_use[data_inds], **spline_opts
                            )
                            lut = get_lambda(lut, grid=False)

                        if reuse_spline:
                            self.saved_interp_functions[key] = lut

                    interp_data[index0, pol_return_ind, index3, :] = lut(
                        za_array, az_array
                    )

        interp_arrays = [interp_data, interp_basis_vector, interp_bandpass]
        if self.antenna_type == "phased_array":
            interp_arrays.append(interp_coupling_matrix)
        return tuple(interp_arrays)

    def _interp_az_za_map_coordinates(
        self,
        *,
        az_array,
        za_array,
        freq_array,
        freq_interp_kind="cubic",
        freq_interp_tol=1.0,
        polarizations=None,
        spline_opts=None,
        check_azza_domain: bool = True,
        reuse_spline: bool = False,
        return_basis_vector: bool = False,
    ):
        """
        Interpolate in az_za coordinate system using map_coordinates.

        Uses the :func:`scipy.ndimage.map_coordinates` function to perform
        interpolation in the azimuth-zenith angle coordinate system.

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
        spline_opts : dict, optional
            Option to specify (kx, ky, s) for numpy.RectBivariateSpline. Note that
            this parameter is ignored if this function has been called previously
            on this object instance and reuse_spline is True.
        check_azza_domain : bool
            Whether to check the domain of az/za to ensure that they are covered by the
            intrinsic data array. Checking them can be quite computationally expensive.
        return_basis_vector : bool
            Whether to return the interpolated basis vector. Default is False as of
            v3.1.1, but was previously True.

        Returns
        -------
        interp_data : array_like of float or complex
            The array of interpolated data values,
            shape: (Naxes_vec, Nfeeds or Npols, freq_array.size, az_array.size)
        interp_basis_vector : array_like of float
            The array of interpolated basis vectors,
            shape: (Naxes_vec, Ncomponents_vec, az_array.size)
        interp_bandpass : array_like of float
            The interpolated bandpass. shape: (freq_array.size,)
        interp_coupling_matrix : array_like of complex, optional
            The interpolated coupling matrix. Only returned if antenna_type is
            "phased_array".
            shape: (Nelements, Nelements, Nfeeds, Nfeeds, freq_array.size)

        """
        if self.pixel_coordinate_system != "az_za":
            raise ValueError(
                "pixel_coordinate_system must be 'az_za' to use this interpolation "
                "function"
            )

        # Perform initial frequency interpolation to get the data array
        (
            input_data_array,
            interp_bandpass,
            interp_coupling_matrix,
            input_nfreqs,
            freq_array,
        ) = self._handle_input_for_freq_interpolation(
            freq_array=freq_array,
            freq_interp_kind=freq_interp_kind,
            freq_interp_tol=freq_interp_tol,
        )

        if az_array is None or za_array is None:
            interp_arrays = [input_data_array, self.basis_vector_array, interp_bandpass]
            if self.antenna_type == "phased_array":
                interp_arrays.append(interp_coupling_matrix)
            return tuple(interp_arrays)

        freq_axis = 2

        # Check input arrays
        assert isinstance(az_array, np.ndarray)
        assert isinstance(za_array, np.ndarray)
        assert az_array.ndim == 1
        assert az_array.shape == za_array.shape
        assert input_data_array.shape[freq_axis] == input_nfreqs

        # Get the data type
        if np.iscomplexobj(input_data_array):
            data_type = np.complex128
        else:
            data_type = np.float64

        # Prepare the data for interpolation
        data_use, phi_use, theta_use = self._prepare_coordinate_data(input_data_array)

        # Prepare basis functions
        if return_basis_vector:
            interp_basis_vector = self._prepare_basis_vector_array(az_array.size)
        else:
            interp_basis_vector = None

        # Get number of polarizations and indices
        Npol_feeds, pol_inds = self._prepare_polarized_inputs(polarizations)

        # Check if the interpolation points are within the data array
        if check_azza_domain:
            self._check_interpolation_domain(az_array, za_array, phi_use, theta_use)

        interp_data = np.zeros(
            (self.Naxes_vec, Npol_feeds, input_nfreqs, az_array.size), dtype=data_type
        )

        if spline_opts is None or not isinstance(spline_opts, dict):
            spline_opts = {}

        az_array -= phi_use.min()
        az_array *= (phi_use.size - 1) / (phi_use.max() - phi_use.min())

        za_array -= theta_use.min()
        za_array *= (theta_use.size - 1) / (theta_use.max() - theta_use.min())

        for index3 in range(input_nfreqs):
            for index0 in range(self.Naxes_vec):
                for pol_return_ind, index2 in enumerate(pol_inds):
                    ndimage.map_coordinates(
                        data_use[index0, index2, index3],
                        [za_array, az_array],
                        output=interp_data[index0, pol_return_ind, index3],
                        **spline_opts,
                    )

        interp_arrays = [interp_data, interp_basis_vector, interp_bandpass]
        if self.antenna_type == "phased_array":
            interp_arrays.append(interp_coupling_matrix)
        return tuple(interp_arrays)

    def _interp_healpix_bilinear(
        self,
        *,
        az_array,
        za_array,
        freq_array,
        freq_interp_kind="cubic",
        freq_interp_tol=1.0,
        polarizations=None,
        reuse_spline=False,
        return_basis_vector: bool = False,
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
        return_basis_vector : bool
            Whether to return the interpolated basis vectors. Prior to v3.1.1 these
            were always returned. Now they are not by default.

        Returns
        -------
        interp_data : array_like of float or complex
            The array of interpolated data values,
            shape: (Naxes_vec, Nfeeds or Npols, Nfreqs, az_array.size)
        interp_basis_vector : array_like of float
            The array of interpolated basis vectors,
            shape: (Naxes_vec, Ncomponents_vec, az_array.size)
        interp_bandpass : array_like of float
            The interpolated bandpass. shape: (freq_array.size,)
        interp_coupling_matrix : array_like of complex, optional
            The interpolated coupling matrix. Only returned if antenna_type is
            "phased_array".
            shape: (Nelements, Nelements, Nfeeds, Nfeeds, freq_array.size)

        """
        if self.pixel_coordinate_system != "healpix":
            raise ValueError(
                "pixel_coordinate_system must be 'healpix' to use this interpolation "
                "function"
            )

        try:
            from astropy_healpix import HEALPix
        except ImportError as e:
            raise ImportError(
                "astropy_healpix is not installed but is "
                "required for healpix functionality. "
                "Install 'astropy-healpix' using conda or pip."
            ) from e

        if not self.Npixels == 12 * self.nside**2:
            raise ValueError(
                "simple healpix interpolation requires full sky healpix maps."
            )
        if not np.max(np.abs(np.diff(self.pixel_array))) == 1:
            raise ValueError(
                "simple healpix interpolation requires healpix pixels to be in order."
            )

        # Perform initial frequency interpolation to get the data array
        (
            input_data_array,
            interp_bandpass,
            interp_coupling_matrix,
            input_nfreqs,
            freq_array,
        ) = self._handle_input_for_freq_interpolation(
            freq_array=freq_array,
            freq_interp_kind=freq_interp_kind,
            freq_interp_tol=freq_interp_tol,
        )

        if az_array is None or za_array is None:
            interp_arrays = [input_data_array, self.basis_vector_array, interp_bandpass]
            if self.antenna_type == "phased_array":
                interp_arrays.append(interp_coupling_matrix)
            return tuple(interp_arrays)

        # Check input arrays
        assert isinstance(az_array, np.ndarray)
        assert isinstance(za_array, np.ndarray)
        assert az_array.ndim == 1
        assert az_array.shape == za_array.shape

        # Get number of polarizations and indices
        Npol_feeds, pol_inds = self._prepare_polarized_inputs(polarizations)

        if np.iscomplexobj(input_data_array):
            data_type = np.complex128
        else:
            data_type = np.float64

        interp_data = np.zeros(
            (self.Naxes_vec, Npol_feeds, input_nfreqs, len(az_array)), dtype=data_type
        )

        # Prepare basis functions
        if return_basis_vector:
            interp_basis_vector = self._prepare_basis_vector_array(az_array.size)
        else:
            interp_basis_vector = None

        hp_obj = HEALPix(nside=self.nside, order=self.ordering)
        lat_array, lon_array = utils.coordinates.zenithangle_azimuth_to_hpx_latlon(
            za_array, az_array
        )
        lon_array = lon_array * units.rad
        lat_array = lat_array * units.rad

        for index3 in range(input_nfreqs):
            for index0 in range(self.Naxes_vec):
                for index2 in range(Npol_feeds):
                    data_inds = (index0, pol_inds[index2], index3)
                    if np.iscomplexobj(input_data_array):
                        # interpolate real and imaginary parts separately
                        real_hmap = hp_obj.interpolate_bilinear_lonlat(
                            lon_array, lat_array, input_data_array[data_inds].real
                        )
                        imag_hmap = hp_obj.interpolate_bilinear_lonlat(
                            lon_array, lat_array, input_data_array[data_inds].imag
                        )

                        hmap = real_hmap + 1j * imag_hmap
                    else:
                        # interpolate once
                        hmap = hp_obj.interpolate_bilinear_lonlat(
                            lon_array, lat_array, input_data_array[data_inds]
                        )

                    interp_data[index0, index2, index3, :] = hmap

        interp_arrays = [interp_data, interp_basis_vector, interp_bandpass]
        if self.antenna_type == "phased_array":
            interp_arrays.append(interp_coupling_matrix)
        return tuple(interp_arrays)

    def interp(
        self,
        *,
        az_array=None,
        za_array=None,
        interpolation_function=None,
        freq_interp_kind=None,
        az_za_grid=False,
        healpix_nside=None,
        healpix_inds=None,
        freq_array=None,
        freq_interp_tol=1.0,
        polarizations=None,
        return_bandpass=False,
        return_coupling=False,
        return_basis_vector: bool | None = None,
        reuse_spline=False,
        spline_opts=None,
        new_object=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        check_azza_domain: bool = True,
    ):
        """
        Interpolate beam to given frequency, az & za locations or Healpix pixel centers.

        Uses the function specified in `interpolation_function`, which defaults to
        "az_za_simple" for objects with the "az_za" pixel_coordinate_system and
        "healpix_simple" for objects with the "healpix" pixel_coordinate_system.
        Currently supported interpolation functions include:

        - "az_za_simple": Uses scipy RectBivariate spline interpolation, can only be
          used on objects with an "az_za" pixel_coordinate_system.
        - "az_za_map_coordinates": Uses scipy map_coordinates interpolation, can only
          be used on objects with an "az_za" pixel_coordinate_system.
        - "healpix_simple": Uses HEALPix nearest-neighbor bilinear interpolation, can
          only be used on objects with a "healpix" pixel_coordinate_system.

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
        interpolation_function : str, optional
            Specify the interpolation function to use. Defaults to: "az_za_simple" for
            objects with the "az_za" pixel_coordinate_system and "healpix_simple" for
            objects with the "healpix" pixel_coordinate_system. "az_za_map_coordinates"
            is also available for objects with the "az_za" pixel_coordinate_system.
        freq_interp_kind : str
            Interpolation method to use frequency. See scipy.interpolate.interp1d
            for details. Defaults to "cubic" (Note that this is a change. It used to
            default to "linear" when it was assigned to the object. However, multiple
            groups have found that a linear interpolation leads to nasty artifacts in
            visibility simulations for EoR applications.)
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
        return_bandpass : bool
            Option to return the bandpass. Only applies if `new_object` is False.
        return_coupling : bool
            Option to return the interpolated coupling matrix, only applies if
            `antenna_type` is "phased_array" and `new_object` is False.
        new_object : bool
            Option to return a new UVBeam object with the interpolated data,
            if possible. Note that this is only possible for Healpix pixels or
            if az_za_grid is True and `az_array` and `za_array` are evenly spaced
            or for frequency only interpolation.
        reuse_spline : bool
            Save the interpolation functions for reuse. Only applies for
            `az_za_simple` and `az_za_map_coordinates` interpolation.
        spline_opts : dict, optional
            Provide options to numpy.RectBivariateSpline. This includes spline
            order parameters `kx` and `ky`, and smoothing parameter `s`.
            Applies for `az_za_simple` and `az_za_map_coordinates` interpolation.
            Note that this parameter is ignored if this function has been called
            previously on this object instance and reuse_spline is True.
        run_check : bool
            Only used if new_object is True. Option to check for the existence
            and proper shapes of required parameters on the new object.
        check_extra : bool
            Only used if new_object is True. Option to check optional parameters
            as well as required ones on the new object.
        run_check_acceptability : bool
            Only used if new_object is True. Option to check acceptable range
            of the values of required parameters on the new object.
        check_azza_domain : bool
            Whether to check the domain of az/za to ensure that they are covered by the
            intrinsic data array. Checking them can be quite computationally expensive.
            Conversely, if the passed az/za are outside of the domain, they will be
            silently extrapolated and the behavior is not well-defined. Only
            applies for `az_za_simple` and `az_za_map_coordinates` interpolation.
        return_basis_vector : bool
            Whether to return the interpolated basis vectors. Prior to v3.1.1 these
            were always returned. In v3.3+ they will _not_ be returned by default
            (and not computed by default, unless new_object=True).

        Returns
        -------
        array_like of float or complex or a UVBeam object
            Either an array of interpolated values or a UVBeam object if
            `new_object` is True. The shape of the interpolated data will be:
            (Naxes_vec, Nfeeds or Npols, Nfreqs or freq_array.size if
            freq_array is passed, Npixels/(Naxis1, Naxis2) or az_array.size if
            az/za_arrays are passed) or (Naxes_vec, Nfeeds or Npols,
            Nfreqs or freq_array.size if freq_array is passed, Npixels/(Naxis1, Naxis2)
        interp_basis_vector : array_like of float, optional
            The array of interpolated basis vectors (or self.basis_vector_array
            if az/za_arrays are not passed). Only returned if `new_object` is False.
            shape: (Naxes_vec, Ncomponents_vec, Npixels/(Naxis1, Naxis2) or
            az_array.size if az/za_arrays are passed)
        interp_bandpass : array_like of float, optional
            The interpolated bandpass, only returned if `return_bandpass` is True and
            `new_object` is False. Shape: (freq_array.size,)
        interp_coupling_matrix : array_like of complex, optional
            The interpolated coupling matrix. Only returned if return_coupling is True
            and `new_object` is False.
            Shape: (Nelements, Nelements, Nfeeds, Nfeeds, freq_array.size)

        """
        if new_object:
            # To create a new object, we always need the interpolated basis vectors.
            return_basis_vector = True

        if return_basis_vector is None:
            return_basis_vector = True
            warnings.warn(
                "The default value for `return_basis_vector` is True, but in v3.3 it "
                "will be set to False. Silence this warning by explicitly setting it "
                " to either True or False.",
                category=DeprecationWarning,
            )

        if interpolation_function is None:
            if self.pixel_coordinate_system == "az_za":
                interpolation_function = "az_za_simple"
            elif self.pixel_coordinate_system == "healpix":
                interpolation_function = "healpix_simple"
            else:
                raise ValueError(
                    "There is no default interpolation function for objects with "
                    f"pixel_coordinate_system: {self.pixel_coordinate_system}"
                )

        if freq_interp_kind is None:
            freq_interp_kind = "cubic"

        if return_coupling is True and self.antenna_type != "phased_array":
            raise ValueError(
                "return_coupling can only be set if antenna_type is phased_array"
            )

        if new_object and not az_za_grid and az_array is not None:
            raise ValueError(
                "A new object can only be returned if "
                "az_za_grid is True or for Healpix pixels or "
                "for frequency only interpolation."
            )

        allowed_interp_funcs = list(self.interpolation_function_dict.keys())
        if interpolation_function not in allowed_interp_funcs:
            raise ValueError(
                "interpolation_function not recognized, must be one of "
                f"{allowed_interp_funcs}"
            )
        interp_func_name = interpolation_function
        interp_func = self.interpolation_function_dict[interpolation_function]["func"]

        if freq_array is not None:
            if freq_array.ndim != 1:
                raise ValueError("freq_array must be one-dimensional")

            # get frequency distances
            freq_dists = np.abs(self.freq_array - freq_array.reshape(-1, 1))
            nearest_dist = np.min(freq_dists, axis=1)
            interp_bool = np.any(nearest_dist >= freq_interp_tol)

            # use the beam at nearest neighbors if not interp_bool
            if not interp_bool:
                freq_interp_kind = "nearest"

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
            except ImportError as e:
                raise ImportError(
                    "astropy_healpix is not installed but is "
                    "required for healpix functionality. "
                    "Install 'astropy-healpix' using conda or pip."
                ) from e

            hp_obj = HEALPix(nside=healpix_nside)
            if healpix_inds is None:
                healpix_inds = np.arange(hp_obj.npix)

            hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(healpix_inds)
            za_array_use, az_array_use = (
                utils.coordinates.hpx_latlon_to_zenithangle_azimuth(
                    hpx_lat.radian, hpx_lon.radian
                )
            )

        extra_keyword_dict = {}
        if interp_func in [
            "_interp_az_za_rect_spline",
            "_interp_az_za_map_coordinates",
        ]:
            extra_keyword_dict["reuse_spline"] = reuse_spline
            extra_keyword_dict["spline_opts"] = spline_opts
            extra_keyword_dict["check_azza_domain"] = check_azza_domain

        interp_arrays = getattr(self, interp_func)(
            az_array=az_array_use,
            za_array=za_array_use,
            freq_array=freq_array,
            freq_interp_kind=freq_interp_kind,
            polarizations=polarizations,
            return_basis_vector=return_basis_vector,
            **extra_keyword_dict,
        )

        if self.antenna_type == "simple":
            interp_data, interp_basis_vector, interp_bandpass = interp_arrays
        else:
            (
                interp_data,
                interp_basis_vector,
                interp_bandpass,
                interp_coupling_matrix,
            ) = interp_arrays
        # return just the interpolated arrays
        if not new_object:
            interp_arrays = [interp_data, interp_basis_vector]
            if return_bandpass:
                interp_arrays.append(interp_bandpass)
            if return_coupling:
                interp_arrays.append(interp_coupling_matrix)
            return tuple(interp_arrays)

        # return a new UVBeam object with interpolated data
        else:
            # make a new object
            new_uvb = self.copy()

            history_update_string = " Interpolated"
            if freq_array is not None:
                history_update_string += " in frequency"
                new_uvb.Nfreqs = freq_array.size
                new_uvb.freq_array = freq_array
                new_uvb.bandpass_array = interp_bandpass

                if self.antenna_type == "phased_array":
                    new_uvb.coupling_matrix = interp_coupling_matrix

                optional_freq_params = [
                    "receiver_temperature_array",
                    "loss_array",
                    "mismatch_array",
                    "s_parameters",
                ]
                for param_name in optional_freq_params:
                    if getattr(self, param_name) is not None:
                        warnings.warn(
                            f"Input object has {param_name} defined but we do not "
                            "currently support interpolating it in frequency. Returned "
                            "object will have it set to None."
                        )
                        setattr(new_uvb, param_name, None)

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
                f" using pyuvdata with interpolation_function = {interp_func_name}"
            )
            if freq_array is not None:
                history_update_string += f" and freq_interp_kind = {freq_interp_kind}"
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
        *,
        nside=None,
        interpolation_function=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=True,
    ):
        """
        Convert beam to the healpix coordinate system using interpolation.

        Note that this interpolation isn't perfect. Interpolating an Efield beam
        and then converting to power gives a different result than converting
        to power and then interpolating at about a 5% level.

        Uses the function specified in `interpolation_function`, defaults to the
        "az_za_simple" for objects with the "az_za" pixel_coordinate_system.
        Currently supported interpolation functions for beams that are not already in
        a healpix coordinate system include:

        - "az_za_simple": Uses scipy RectBivariate spline interpolation, can only be
          used on objects with an "az_za" pixel_coordinate_system.
        - "az_za_map_coordinates": Uses scipy map_coordinates interpolation, can only
          be used on objects with an "az_za" pixel_coordinate

        Parameters
        ----------
        nside : int
            The nside to use for the Healpix map. If not specified, use
            the nside that gives the closest resolution that is higher than the
            input resolution.
        interpolation_function : str, optional
            Specify the interpolation function to use. Defaults to to:
            "az_za_simple" for objects with the "az_za" pixel_coordinate_system.
            "az_za_map_coordinates" is also available for objects with the "az_za"
            pixel_coordinate_system.
        run_check : bool
            Option to check for the existence and proper shapes of required
            parameters after converting to healpix.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of required parameters
            after combining objects
        inplace : bool
            Option to perform the interpolation directly on self or return a new
            UVBeam object.

        """
        if self.pixel_coordinate_system == "healpix":
            if inplace:
                return
            else:
                return self.copy()

        try:
            from astropy_healpix import HEALPix
        except ImportError as e:
            raise ImportError(
                "astropy_healpix is not installed but is "
                "required for healpix functionality. "
                "Install 'astropy-healpix' using conda or pip."
            ) from e

        if nside is None:
            min_res = np.min(
                np.abs(
                    np.array(
                        [np.diff(self.axis1_array)[0], np.diff(self.axis2_array)[0]]
                    )
                )
            )
            nside_min_res = np.sqrt(3 / np.pi) * np.radians(60.0) / min_res
            nside = int(2 ** np.ceil(np.log2(nside_min_res)))
            hp_obj = HEALPix(nside=nside)
            assert hp_obj.pixel_resolution.to_value(units.radian) < min_res
        else:
            hp_obj = HEALPix(nside=nside)

        pixels = np.arange(hp_obj.npix)
        hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(pixels)
        hpx_zen_ang, hpx_az = utils.coordinates.hpx_latlon_to_zenithangle_azimuth(
            hpx_lat.radian, hpx_lon.radian
        )

        inds_to_use = _uvbeam.find_healpix_indices(
            np.ascontiguousarray(self.axis2_array, dtype=np.float64),
            np.ascontiguousarray(self.axis1_array, dtype=np.float64),
            np.ascontiguousarray(hpx_zen_ang, dtype=np.float64),
            np.ascontiguousarray(hpx_az, dtype=np.float64),
            np.float64(hp_obj.pixel_resolution.to_value(units.radian)),
        )

        pixels = pixels[inds_to_use]

        beam_object = self.interp(
            healpix_nside=nside,
            healpix_inds=pixels,
            new_object=True,
            interpolation_function=interpolation_function,
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
        Get the healpix power beam map corresponding to the specified polarization.

        pseudo-stokes I: 'pI', Q: 'pQ', U: 'pU' and V: 'pV' or linear dipole
        polarization: 'XX', 'YY', etc.

        Parameters
        ----------
        pol : str or int
            polarization string or integer, Ex. a pseudo-stokes pol 'pI', or
            a linear pol 'XX'.

        Returns
        -------
        np.ndarray of float
            Healpix map of beam powers for a single pol, shape: (Nfreqs, Npixels)
        """
        # assert map is in healpix coords
        assert self.pixel_coordinate_system == "healpix", (
            "pixel_coordinate_system must be healpix"
        )
        # assert beam_type is power
        assert self.beam_type == "power", "beam_type must be power"
        if isinstance(pol, str | np.str_):
            pol = utils.polstr2num(
                pol, x_orientation=self.get_x_orientation_from_feeds()
            )
        pol_array = self.polarization_array
        if pol in pol_array:
            stokes_p_ind = np.where(np.isin(pol_array, pol))[0][0]
            beam = self.data_array[0, stokes_p_ind]
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
        if isinstance(pol, str | np.str_):
            pol = utils.polstr2num(
                pol, x_orientation=self.get_x_orientation_from_feeds()
            )
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
        omega = np.sum(beam, axis=-1) * np.pi / (3.0 * nside**2)

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
        if isinstance(pol, str | np.str_):
            pol = utils.polstr2num(
                pol, x_orientation=self.get_x_orientation_from_feeds()
            )
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
        omega = np.sum(beam**2, axis=-1) * np.pi / (3.0 * nside**2)

        return omega

    def __add__(
        self,
        other,
        *,
        verbose_history=False,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        warn_spacing=True,
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
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            two objects are different, the combined object will keep all the history of
            both input objects (if many objects are combined in succession this can
            lead to very long histories). If False and if the histories for the two
            objects are different, the combined object will have the history of the
            first object and only the parts of the second object history that are unique
            (this is done word by word and can result in hard to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters after combining objects.
        warn_spacing : bool
            Option to raise warnings about spacing that would prevent writing to
            beamfits file-format. Default is True.

        """
        if inplace:
            this = self
        else:
            this = self.copy()
        # Check that both objects are UVBeam and valid
        # Note this will fix the old "e" and "n" feed types
        this.check(check_extra=check_extra, run_check_acceptability=False)
        if not issubclass(other.__class__, this.__class__) and not issubclass(
            this.__class__, other.__class__
        ):
            raise ValueError(
                "Only UVBeam (or subclass) objects can be added "
                "to a UVBeam (or subclass) object"
            )
        other.check(check_extra=check_extra, run_check_acceptability=False)

        # Check objects are compatible
        compatibility_params = [
            "_beam_type",
            "_antenna_type",
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
        if this.antenna_type == "phased_array":
            compatibility_params.extend(
                [
                    "_Nelements",
                    "_delay_array",
                    "_element_coordinate_system",
                    "_element_location_array",
                ]
            )
        for a in compatibility_params:
            if getattr(this, a) != getattr(other, a):
                msg = (
                    "UVParameter " + a[1:] + " does not match. Cannot combine objects."
                )
                raise ValueError(msg)

        # check for presence of optional parameters
        optional_params = [
            "_receiver_temperature_array",
            "_loss_array",
            "_mismatch_array",
            "_s_parameters",
        ]

        for attr in optional_params:
            this_attr = getattr(this, attr)
            other_attr = getattr(other, attr)
            if (this_attr.value is None) != (other_attr.value is None):
                warnings.warn(
                    "Only one of the UVBeam objects being combined "
                    f"has optional parameter {attr}. After the sum the "
                    f"final object will not have {attr}"
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

        both_freq = np.intersect1d(this.freq_array, other.freq_array)

        if this.pixel_coordinate_system == "healpix":
            both_pixels = np.intersect1d(this.pixel_array, other.pixel_array)
        else:
            both_axis1 = np.intersect1d(this.axis1_array, other.axis1_array)
            both_axis2 = np.intersect1d(this.axis2_array, other.axis2_array)

        if len(both_pol) > 0 and len(both_freq) > 0:
            if self.pixel_coordinate_system == "healpix":
                if len(both_pixels) > 0:
                    raise ValueError(
                        "These objects have overlapping data and cannot be combined."
                    )
            else:
                if len(both_axis1) > 0 and len(both_axis2) > 0:
                    raise ValueError(
                        "These objects have overlapping data and cannot be combined."
                    )

        # Update filename parameter
        this.filename = utils.tools._combine_filenames(this.filename, other.filename)
        if this.filename is not None:
            this._filename.form = (len(this.filename),)

        if this.pixel_coordinate_system == "healpix":
            temp = np.nonzero(~np.isin(other.pixel_array, this.pixel_array))[0]
            if len(temp) > 0:
                pix_new_inds = temp
                history_update_string += "healpix pixel"
                n_axes += 1
            else:
                pix_new_inds = []
        else:
            temp = np.nonzero(~np.isin(other.axis1_array, this.axis1_array))[0]
            if len(temp) > 0:
                ax1_new_inds = temp
                history_update_string += "first image"
                n_axes += 1
            else:
                ax1_new_inds = []

            temp = np.nonzero(~np.isin(other.axis2_array, this.axis2_array))[0]
            if len(temp) > 0:
                ax2_new_inds = temp
                if n_axes > 0:
                    history_update_string += ", second image"
                else:
                    history_update_string += "second image"
                n_axes += 1
            else:
                ax2_new_inds = []

        temp = np.nonzero(~np.isin(other.freq_array, this.freq_array))[0]
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
                ~np.isin(other.polarization_array, this.polarization_array)
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
            temp = np.nonzero(~np.isin(other.feed_array, this.feed_array))[0]
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
                data_pix_axis = 3
                data_pad_dims = tuple(
                    list(this.data_array.shape[0:data_pix_axis])
                    + [len(pix_new_inds)]
                    + list(this.data_array.shape[data_pix_axis + 1 :])
                )
                data_zero_pad = np.zeros(data_pad_dims, dtype=this.data_array.dtype)

                this.pixel_array = np.concatenate(
                    [this.pixel_array, other.pixel_array[pix_new_inds]]
                )
                order = np.argsort(this.pixel_array)
                this.pixel_array = this.pixel_array[order]

                this.data_array = np.concatenate(
                    [this.data_array, data_zero_pad], axis=data_pix_axis
                )[..., order]

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
                data_ax1_axis = 4
                data_pad_dims = tuple(
                    list(this.data_array.shape[0:data_ax1_axis])
                    + [len(ax1_new_inds)]
                    + list(this.data_array.shape[data_ax1_axis + 1 :])
                )
                data_zero_pad = np.zeros(data_pad_dims, dtype=this.data_array.dtype)

                this.axis1_array = np.concatenate(
                    [this.axis1_array, other.axis1_array[ax1_new_inds]]
                )
                order = np.argsort(this.axis1_array)
                this.axis1_array = this.axis1_array[order]
                this.data_array = np.concatenate(
                    [this.data_array, data_zero_pad], axis=data_ax1_axis
                )[..., order]

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
                data_ax2_axis = 3
                data_pad_dims = tuple(
                    list(this.data_array.shape[0:data_ax2_axis])
                    + [len(ax2_new_inds)]
                    + list(this.data_array.shape[data_ax2_axis + 1 :])
                )
                data_zero_pad = np.zeros(data_pad_dims, dtype=this.data_array.dtype)

                this.axis2_array = np.concatenate(
                    [this.axis2_array, other.axis2_array[ax2_new_inds]]
                )
                order = np.argsort(this.axis2_array)
                this.axis2_array = this.axis2_array[order]

                this.data_array = np.concatenate(
                    [this.data_array, data_zero_pad], axis=data_ax2_axis
                )[..., order, :]

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
            faxis = 2
            data_pad_dims = tuple(
                list(this.data_array.shape[0:faxis])
                + [len(fnew_inds)]
                + list(this.data_array.shape[faxis + 1 :])
            )
            data_zero_pad = np.zeros(data_pad_dims, dtype=this.data_array.dtype)

            this.freq_array = np.concatenate(
                [this.freq_array, other.freq_array[fnew_inds]]
            )
            order = np.argsort(this.freq_array)
            this.freq_array = this.freq_array[order]

            this.bandpass_array = np.concatenate(
                [this.bandpass_array, np.zeros(len(fnew_inds))]
            )[order]
            this.data_array = np.concatenate(
                [this.data_array, data_zero_pad], axis=faxis
            )[:, :, order, ...]
            if this.receiver_temperature_array is not None:
                this.receiver_temperature_array = np.concatenate(
                    [this.receiver_temperature_array, np.zeros(len(fnew_inds))]
                )[order]
            if this.loss_array is not None:
                this.loss_array = np.concatenate(
                    [this.loss_array, np.zeros(len(fnew_inds))]
                )[order]
            if this.mismatch_array is not None:
                this.mismatch_array = np.concatenate(
                    [this.mismatch_array, np.zeros(len(fnew_inds))]
                )[order]
            if this.s_parameters is not None:
                this.s_parameters = np.concatenate(
                    [this.s_parameters, np.zeros((4, len(fnew_inds)))], axis=1
                )[:, order]
            if this.antenna_type == "phased_array":
                coupling_pad_dims = tuple(
                    list(this.coupling_matrix.shape[0:-1]) + [len(fnew_inds)]
                )
                coupling_zero_pad = np.zeros(
                    coupling_pad_dims, dtype=this.coupling_matrix.dtype
                )
                this.coupling_matrix = np.concatenate(
                    [this.coupling_matrix, coupling_zero_pad], axis=-1
                )[..., order]

        if len(pnew_inds) > 0:
            paxis = 1
            data_pad_dims = tuple(
                list(this.data_array.shape[0:paxis])
                + [len(pnew_inds)]
                + list(this.data_array.shape[paxis + 1 :])
            )
            data_zero_pad = np.zeros(data_pad_dims, dtype=this.data_array.dtype)

            if this.beam_type == "power":
                initial_pol_array = this.polarization_array.copy()
                this.polarization_array = np.concatenate(
                    [this.polarization_array, other.polarization_array[pnew_inds]]
                )
                order = np.argsort(np.abs(this.polarization_array))
                this.polarization_array = this.polarization_array[order]
            else:
                this.feed_array = np.concatenate(
                    [this.feed_array, other.feed_array[pnew_inds]]
                )
                this.feed_angle = np.concatenate(
                    [this.feed_angle, other.feed_angle[pnew_inds]]
                )
                order = np.argsort(this.feed_array)
                this.feed_array = this.feed_array[order]
                this.feed_angle = this.feed_angle[order]

                if this.antenna_type == "phased_array":
                    # have to concat twice because two axes are feed axes
                    coupling_pad_dims = tuple(
                        list(this.coupling_matrix.shape[0:2])
                        + [len(pnew_inds)]
                        + list(this.coupling_matrix.shape[3:])
                    )
                    coupling_zero_pad = np.zeros(
                        coupling_pad_dims, dtype=this.coupling_matrix.dtype
                    )
                    this.coupling_matrix = np.concatenate(
                        [this.coupling_matrix, coupling_zero_pad], axis=2
                    )[:, :, order]

                    coupling_pad_dims = tuple(
                        list(this.coupling_matrix.shape[0:3])
                        + [len(pnew_inds)]
                        + list(this.coupling_matrix.shape[4:])
                    )
                    coupling_zero_pad = np.zeros(
                        coupling_pad_dims, dtype=this.coupling_matrix.dtype
                    )
                    this.coupling_matrix = np.concatenate(
                        [this.coupling_matrix, coupling_zero_pad], axis=3
                    )[:, :, :, order]

            this.data_array = np.concatenate(
                [this.data_array, data_zero_pad], axis=paxis
            )[:, order, ...]

        # Now populate the data
        if this.beam_type == "power":
            this.Npols = this.polarization_array.shape[0]
            pol_t2o = np.nonzero(
                np.isin(this.polarization_array, other.polarization_array)
            )[0]

            if len(pnew_inds) > 0:
                # if this does not have cross pols but other does promote to complex
                cross_pols = [-3, -4, -7, -8]
                if (
                    np.intersect1d(other.polarization_array, cross_pols).size > 0
                    and np.intersect1d(initial_pol_array, cross_pols).size == 0
                ):
                    if this.data_array.dtype == np.float32:
                        dtype_use = np.complex64
                    else:
                        dtype_use = complex
                    this.data_array = np.asarray(this.data_array, dtype=dtype_use)
        else:
            this.Nfeeds = this.feed_array.shape[0]
            pol_t2o = np.nonzero(np.isin(this.feed_array, other.feed_array))[0]

        freq_t2o = np.nonzero(np.isin(this.freq_array, other.freq_array))[0]

        if this.pixel_coordinate_system == "healpix":
            this.Npixels = this.pixel_array.shape[0]
            pix_t2o = np.nonzero(np.isin(this.pixel_array, other.pixel_array))[0]
            this.data_array[
                np.ix_(np.arange(this.Naxes_vec), pol_t2o, freq_t2o, pix_t2o)
            ] = other.data_array
            if this.beam_type == "efield":
                this.basis_vector_array[
                    np.ix_(np.arange(this.Naxes_vec), np.arange(2), pix_t2o)
                ] = other.basis_vector_array
        else:
            this.Naxes1 = this.axis1_array.shape[0]
            this.Naxes2 = this.axis2_array.shape[0]
            ax1_t2o = np.nonzero(np.isin(this.axis1_array, other.axis1_array))[0]
            ax2_t2o = np.nonzero(np.isin(this.axis2_array, other.axis2_array))[0]
            this.data_array[
                np.ix_(np.arange(this.Naxes_vec), pol_t2o, freq_t2o, ax2_t2o, ax1_t2o)
            ] = other.data_array
            if this.beam_type == "efield":
                this.basis_vector_array[
                    np.ix_(np.arange(this.Naxes_vec), np.arange(2), ax2_t2o, ax1_t2o)
                ] = other.basis_vector_array

        this.bandpass_array[np.ix_(freq_t2o)] = other.bandpass_array
        if this.receiver_temperature_array is not None:
            this.receiver_temperature_array[np.ix_(freq_t2o)] = (
                other.receiver_temperature_array
            )
        if this.loss_array is not None:
            this.loss_array[np.ix_(freq_t2o)] = other.loss_array
        if this.mismatch_array is not None:
            this.mismatch_array[np.ix_(freq_t2o)] = other.mismatch_array
        if this.s_parameters is not None:
            this.s_parameters[np.ix_(np.arange(4), freq_t2o)] = other.s_parameters

        if this.antenna_type == "phased_array":
            this.coupling_matrix[
                np.ix_(
                    np.arange(this.Nelements),
                    np.arange(this.Nelements),
                    pol_t2o,
                    pol_t2o,
                    freq_t2o,
                )
            ] = other.coupling_matrix

        this.Nfreqs = this.freq_array.size

        # Check specific requirements
        if warn_spacing and not utils.tools._test_array_constant_spacing(
            this._freq_array
        ):
            warnings.warn(
                "Combined frequencies are not evenly spaced. This will "
                "make it impossible to write this data out to some file types."
            )

        if (
            self.beam_type == "power"
            and warn_spacing
            and not utils.tools._test_array_constant_spacing(this._polarization_array)
        ):
            warnings.warn(
                "Combined polarizations are not evenly spaced. This will "
                "make it impossible to write this data out to some file types."
            )

        if n_axes > 0:
            history_update_string += " axis using pyuvdata."
            histories_match = utils.history._check_histories(
                this.history, other.history
            )

            this.history += history_update_string
            if not histories_match:
                if verbose_history:
                    this.history += " Next object history follows. " + other.history
                else:
                    extra_history = utils.history._combine_history_addition(
                        this.history, other.history
                    )
                    if extra_history is not None:
                        this.history += (
                            " Unique part of next object history follows. "
                            + extra_history
                        )

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

    def _select_by_index(
        self,
        *,
        axis1_inds,
        axis2_inds,
        pix_inds,
        freq_inds,
        feed_inds,
        pol_inds,
        history_update_string,
    ):
        """
        Perform select based on indexing arrays.

        Parameters
        ----------
        ant_inds : list of int
            list of antenna indices to keep. Can be None (to keep everything).
        time_inds : list of int
            list of time indices to keep. Can be None (to keep everything).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        jones_inds : list of int
            list of jones indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.
        """
        # Create a dictionary to pass to _select_along_param_axis
        ind_dict = {
            "Naxes1": axis1_inds,
            "Naxes2": axis2_inds,
            "Npixels": pix_inds,
            "Nfreqs": freq_inds,
            "Nfeeds": feed_inds,
            "Npols": pol_inds,
        }

        cross_pol = [-3, -4, -7, -8]
        had_cross = any(np.isin(cross_pol, self.polarization_array))

        # During each loop interval, we pop off an element of this dict, so continue
        # until the dict is empty.
        self._select_along_param_axis(ind_dict)

        if pol_inds is not None and (
            had_cross and not any(np.isin(cross_pol, self.polarization_array))
        ):
            # selecting from object with cross-pols down to non-cross pols so
            # data_array should become real
            if np.any(np.iscomplex(self.data_array)):
                warnings.warn(
                    "Polarization select should result in a real array but the "
                    "imaginary part is not zero."
                )
            else:
                self.data_array = np.abs(self.data_array)

        # Update the history string
        self.history += history_update_string

    def select(
        self,
        *,
        axis1_inds=None,
        axis2_inds=None,
        pixels=None,
        frequencies=None,
        freq_chans=None,
        feeds=None,
        polarizations=None,
        invert=False,
        strict=False,
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        warn_spacing=True,
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
        feeds : array_like of str, optional
            The feeds to keep in the object. If the `x_orientation` attribute is set,
            the physical dipole strings (e.g. "n", "e") are also supported.
            Cannot be set if the beam_type is "power".
        polarizations : array_like of int or str, optional
            The polarizations to keep in the object.
            Cannot be set if the beam_type is "efield". If passing strings, the
            canonical polarization strings (e.g. "xx", "rr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "nn", "ee") are also supported.
        invert : bool
            Normally records matching given criteria are what are included in the
            subsequent object. However, if set to True, these records are excluded
            instead. Default is False.
        strict : bool or None
            Normally, select will warn when no records match a one element of a
            parameter, as long as *at least one* element matches with what is in the
            object. However, if set to True, an error is thrown if any element
            does not match. If set to None, then neither errors nor warnings are raised.
            Default is False.
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
        warn_spacing : bool
            Option to raise warnings about spacing that would prevent writing to
            beamfits file-format. Default is True.

        """
        # Do a quick compatibility check w/ the old feed types.
        self._fix_feeds()

        if inplace:
            beam_object = self
        else:
            beam_object = self.copy()

        selections = []

        if feeds is not None:
            if beam_object.beam_type == "power":
                raise ValueError("feeds cannot be used with power beams")

            if beam_object.antenna_type == "phased_array":
                warnings.warn(
                    "Downselecting feeds on phased array beams will lead to loss of "
                    "information that cannot be recovered by selecting the other feed "
                    "because the cross-feed coupling matrix elements can only be "
                    "represented when all feeds are present."
                )

        if polarizations is not None and beam_object.beam_type == "efield":
            raise ValueError("polarizations cannot be used with efield beams")

        if axis1_inds is not None:
            if beam_object.pixel_coordinate_system == "healpix":
                raise ValueError(
                    "axis1_inds cannot be used with healpix coordinate system"
                )

            selections.append("parts of first image axis")
            axis1_inds = utils.tools._eval_inds(
                axis1_inds, self.Naxes1, name="axis1_inds", strict=strict, invert=invert
            )
            if len(axis1_inds) == 0:
                raise ValueError(
                    "No data matching this first image axis selection exists."
                )
            if len(axis1_inds) > 1 and not utils.tools._test_array_constant_spacing(
                beam_object.axis1_array[axis1_inds], tols=beam_object._axis1_array.tols
            ):
                raise ValueError(
                    "Selected values along first image axis must be evenly spaced."
                )

        if axis2_inds is not None:
            if beam_object.pixel_coordinate_system == "healpix":
                raise ValueError(
                    "axis2_inds cannot be used with healpix coordinate system"
                )
            selections.append("parts of second image axis")
            axis2_inds = utils.tools._eval_inds(
                axis2_inds, self.Naxes2, name="axis2_inds", strict=strict, invert=invert
            )

            if len(axis2_inds) == 0:
                raise ValueError(
                    "No data matching this second image axis selection exists."
                )
            if len(axis2_inds) > 1 and not utils.tools._test_array_constant_spacing(
                beam_object.axis2_array[axis2_inds], tols=beam_object._axis2_array.tols
            ):
                raise ValueError(
                    "Selected values along second image axis must be evenly spaced."
                )

        pix_inds = None
        if pixels is not None:
            if beam_object.pixel_coordinate_system != "healpix":
                raise ValueError(
                    "pixels can only be used with healpix coordinate system"
                )
            selections.append("healpix pixels")
            mask = np.zeros(self.Npixels, dtype=bool)
            for p in pixels:
                submask = np.isin(self.pixel_array, p)
                if (not invert and not any(submask)) or (invert and all(submask)):
                    err_msg = f"Pixel {p} is not present in the pixel_array"
                    utils.tools._strict_raise(err_msg, strict=strict)
                mask |= submask
            pix_inds = utils.tools._where_combine(mask, pix_inds, invert=invert)

            if len(pix_inds) == 0:
                raise ValueError("No data matching this pixel selection exists.")

        freq_inds, _, freq_selections = utils.frequency._select_freq_helper(
            frequencies=frequencies,
            freq_chans=freq_chans,
            obj_freq_array=self.freq_array,
            freq_tols=self._freq_array.tols,
            invert=invert,
            strict=strict,
            warn_spacing=warn_spacing,
        )
        selections.extend(freq_selections)

        feed_inds, feed_selections = utils.pol._select_feed_helper(
            feeds=feeds,
            obj_feed_array=self.feed_array,
            obj_x_orientation=self.get_x_orientation_from_feeds(),
            invert=invert,
            strict=strict,
        )
        selections.extend(feed_selections)

        pol_inds, pol_selections = utils.pol._select_pol_helper(
            polarizations=polarizations,
            obj_pol_array=self.polarization_array,
            obj_x_orientation=self.get_x_orientation_from_feeds(),
            invert=invert,
            strict=strict,
            warn_spacing=warn_spacing,
        )
        selections.extend(pol_selections)

        # build up history string from selections
        history_update_string = ""
        if len(selections) > 0:
            history_update_string = (
                "  Downselected to specific "
                + ", ".join(selections)
                + " using pyuvdata."
            )

        beam_object._select_by_index(
            axis1_inds=axis1_inds,
            axis2_inds=axis2_inds,
            pix_inds=pix_inds,
            freq_inds=freq_inds,
            feed_inds=feed_inds,
            pol_inds=pol_inds,
            history_update_string=history_update_string,
        )

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

    def read_beamfits(self, filename, **kwargs):
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
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.
        freq_range : tuple of float in Hz
            If given, the lower and upper limit of the frequencies to read in. Default
            is to read in all frequencies. Restricting the frequencies reduces peak
            memory usage.
        mount_type : str
            Antenna mount type to use, which describes the optics of the antenna in
            question, if the keyword is not found in the BeamFITS file being loaded.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.
        az_range : tuple of float in deg
            The azimuth range to read in, if the beam is specified in az/za coordinates.
            Default is to read in all azimuths. Restricting the azimuth reduces peak
            memory usage.
        za_range : tuple of float in deg
            The zenith angle range to read in, if the beam is specified in za/za
            coordinates. Default is to read in all za. Restricting the za reduces peak
            memory.
        """
        from . import beamfits

        if isinstance(filename, list | tuple):
            self.read_beamfits(filename[0], **kwargs)
            if len(filename) > 1:
                for f in filename[1:]:
                    beam2 = UVBeam()
                    beam2.read_beamfits(f, *kwargs)
                    self += beam2
                del beam2
        else:
            beamfits_obj = beamfits.BeamFITS()
            beamfits_obj.read_beamfits(filename, **kwargs)
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
        with open(filename) as file:
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
                    f"{key} is a required key in CST settings files but is not present."
                )

        return settings_dict

    def read_cst_beam(
        self,
        filename,
        *,
        beam_type="power",
        feed_pol=None,
        feed_array=None,
        feed_angle=None,
        rotate_pol=None,
        mount_type=None,
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
        check_auto_power=True,
        fix_auto_power=True,
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
        feed_array : str or array-like of str
            Feeds to define this beam for, e.g. x & y or r & l. Only used for power
            beams (feeds are set by feed_pol for efield beams).
        feed_angle : str or array-like of float
            Position angle of a given feed, units of radians. A feed angle of 0 is
            typically oriented toward zenith for steerable antennas, otherwise toward
            north for fixed antennas (e.g., HERA, LWA). More details on this can be
            found on the "Conventions" page of the docs. Must match shape of feed_pol
            for efield beams, or feed_angle for power beams.
        rotate_pol : bool
            If True, assume the structure in the simulation is symmetric under
            90 degree rotations about the z-axis (so that the y polarization can be
            constructed by rotating the x polarization or vice versa).
            Default: True if feed_pol is a single value or a list with all
            the same values in it, False if it is a list with varying values.
        mount_type : str
            Antenna mount type, which describes the optics of the antenna in question.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.
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
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.

        """
        from . import cst_beam

        if isinstance(filename, np.ndarray):
            if len(filename.shape) > 1:
                raise ValueError("filename can not be a multi-dimensional array")
            filename = filename.tolist()
        if isinstance(filename, list | tuple) and len(filename) == 1:
            filename = filename[0]

        if not isinstance(filename, list | tuple) and filename.endswith("yaml"):
            settings_dict = self._read_cst_beam_yaml(filename)
            if not isinstance(settings_dict["filenames"], list):
                raise ValueError("filenames in yaml file must be a list.")
            if not isinstance(settings_dict["frequencies"], list):
                raise ValueError("frequencies in yaml file must be a list.")
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
                "x_orientation": x_orientation,
                "feed_array": feed_array,
                "feed_angle": feed_angle,
                "mount_type": mount_type,
            }
            if "ref_imp" in settings_dict:
                overriding_keywords["reference_impedance"] = reference_impedance

            for key, val in overriding_keywords.items():
                if key in settings_dict and val is not None:
                    warnings.warn(
                        f"The {key} keyword is set, overriding the "
                        "value in the settings yaml file."
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
            if x_orientation is None:
                x_orientation = settings_dict.get("x_orientation", None)
            if feed_array is None:
                feed_array = settings_dict.get("feed_array", None)
            if feed_angle is None:
                feed_angle = settings_dict.get("feed_angle", None)
            if mount_type is None:
                mount_type = settings_dict.get("mount_type", None)

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
                "feed_array",
                "feed_angle",
                "mount_type",
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
                    if key in rename_extra_keys_map:
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
        if isinstance(frequency, list | tuple) and len(frequency) == 1:
            frequency = frequency[0]

        if isinstance(feed_pol, np.ndarray):
            if len(feed_pol.shape) > 1:
                raise ValueError("feed_pol can not be a multi-dimensional array")
            feed_pol = feed_pol.tolist()
        if isinstance(feed_pol, list | tuple) and len(feed_pol) == 1:
            feed_pol = feed_pol[0]

        if feed_array is not None:
            if isinstance(feed_array, np.ndarray) and (feed_array.ndim > 1):
                raise ValueError("feed_array cannot be a multi-dimensional array.")
            if beam_type != "power" and np.any(feed_array != feed_pol):
                raise ValueError(
                    "Cannot set feed_array for efield beams in read_cst_beam, use "
                    "feed_pol instead."
                )
            feed_array = np.atleast_1d(feed_array)

        if feed_angle is not None:
            if isinstance(feed_angle, np.ndarray) and (feed_angle.ndim > 1):
                raise ValueError("feed_angle cannot be a multi-dimensional array.")
            feed_angle = np.atleast_1d(feed_angle)

            if beam_type == "efield":
                exp_len = len(feed_pol) if isinstance(feed_pol, list | tuple) else 1
                if exp_len != len(feed_angle):
                    raise ValueError(
                        "feed_pol and feed_angle must contain the same number of "
                        "elements for efield beams."
                    )
                if len(feed_angle) == 1:
                    feed_angle = feed_angle[0]
                else:
                    feed_angle = feed_angle.tolist()
            else:
                if feed_array is None:
                    raise ValueError(
                        "Must set either both or neither of feed_array and feed_angle "
                        "for power beams."
                    )
                if len(feed_angle) != len(feed_array):
                    raise ValueError(
                        "feed_array and feed_angle must contain the same number of "
                        "elements for power beams."
                    )

        if isinstance(cst_filename, list | tuple):
            if frequency is not None:
                if isinstance(frequency, list | tuple):
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

            if isinstance(feed_pol, list | tuple):
                if not len(feed_pol) == len(cst_filename):
                    raise ValueError(
                        "If feed_pol and filename are both "
                        "lists they need to be the same length"
                    )
                pol = feed_pol[0]
                if rotate_pol is None:
                    # if a mix of feed pols, don't rotate by default
                    rotate_pol = all(feed == feed_pol[0] for feed in feed_pol)
            else:
                pol = feed_pol
                if rotate_pol is None:
                    rotate_pol = True
            if isinstance(feed_angle, list):
                ang = feed_angle[0]
            else:
                ang = feed_angle
            if isinstance(freq, list | tuple):
                raise ValueError("frequency can not be a nested list")
            if isinstance(pol, list | tuple):
                raise ValueError("feed_pol can not be a nested list")
            self.read_cst_beam(
                cst_filename[0],
                beam_type=beam_type,
                feed_pol=pol,
                feed_array=feed_array,
                feed_angle=ang,
                rotate_pol=rotate_pol,
                mount_type=mount_type,
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
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )
            for file_i, f in enumerate(cst_filename[1:]):
                if isinstance(f, list | tuple):
                    raise ValueError("filename can not be a nested list")

                if isinstance(frequency, list | tuple):
                    freq = frequency[file_i + 1]
                elif frequency is not None:
                    freq = frequency
                else:
                    freq = None
                if isinstance(feed_pol, list | tuple):
                    pol = feed_pol[file_i + 1]
                else:
                    pol = feed_pol
                if isinstance(feed_angle, list):
                    ang = feed_angle[file_i + 1]
                else:
                    ang = feed_angle
                beam2 = UVBeam()
                beam2.read_cst_beam(
                    f,
                    beam_type=beam_type,
                    feed_pol=pol,
                    feed_array=feed_array,
                    feed_angle=ang,
                    rotate_pol=rotate_pol,
                    mount_type=mount_type,
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
                    check_auto_power=check_auto_power,
                    fix_auto_power=fix_auto_power,
                )
                self += beam2
            if len(cst_filename) > 1:
                del beam2
        else:
            if isinstance(frequency, list | tuple):
                raise ValueError("Too many frequencies specified")
            if isinstance(feed_pol, list | tuple):
                raise ValueError("Too many feed_pols specified")
            if rotate_pol is None:
                rotate_pol = True
            cst_beam_obj = cst_beam.CSTBeam()
            cst_beam_obj.read_cst_beam(
                cst_filename,
                beam_type=beam_type,
                feed_pol=feed_pol,
                feed_array=feed_array,
                feed_angle=feed_angle,
                rotate_pol=rotate_pol,
                mount_type=mount_type,
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
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )
            self._convert_from_filetype(cst_beam_obj)
            del cst_beam_obj

        if not isinstance(filename, list | tuple) and filename.endswith("yaml"):
            # update filelist
            basename = os.path.basename(filename)
            self.filename = utils.tools._combine_filenames(self.filename, [basename])
            self._filename.form = (len(self.filename),)

    def read_feko_beam(self, filename, **kwargs):
        """
        Read in data from a FEKO file.

        Parameters
        ----------
        filename : str
            A FEKO text file.
        beam_type : str
            What beam_type to read in ('power' or 'efield').
        feed_pol : str
            The feed polarization that the files corresponds to, e.g. x, y, r or l.
            Defaults to 'x'.
        feed_angle : float
            Position angle of the feed, units of radians. A feed angle of 0 is
            typically oriented toward zenith for steerable antennas, otherwise toward
            north for fixed antennas (e.g., HERA, LWA). More details on this can be
            found on the "Conventions" page of the docs.
        mount_type : str
            Antenna mount type, which describes the optics of the antenna in question.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.
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
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.

        """
        from . import feko_beam

        feko_beam_obj = feko_beam.FEKOBeam()
        feko_beam_obj.read_feko_beam(filename, **kwargs)
        self._convert_from_filetype(feko_beam_obj)
        del feko_beam_obj

    def read_mwa_beam(self, h5filepath, **kwargs):
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
            `wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5`
            (This reader is based on https://github.com/MWATelescope/mwa_pb).
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
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.

        """
        from . import mwa_beam

        mwabeam_obj = mwa_beam.MWABeam()
        mwabeam_obj.read_mwa_beam(h5filepath, **kwargs)
        self._convert_from_filetype(mwabeam_obj)
        del mwabeam_obj

    def read(
        self,
        filename,
        *,
        file_type=None,
        skip_bad_files=False,
        # checking parameters
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        check_auto_power=True,
        fix_auto_power=True,
        # beamfits parameters
        az_range=None,
        za_range=None,
        # beamfits & mwa parameters
        freq_range=None,
        # beamfits & cst parameters
        mount_type=None,
        # cst parameters
        beam_type="power",
        feed_pol=None,
        feed_array=None,
        feed_angle=None,
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
        # mwa parameters
        delays=None,
        amplitudes=None,
        pixels_per_deg=5,
    ):
        """
        Read a generic file into a UVBeam object.

        Some parameters only apply to certain file types.

        Parameters
        ----------
        filename : str or array_like of str
            The file(s) or list(s) (or array(s)) of files to read from.

            For cst yaml files only:

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

        file_type : str
            One of ['mwa_beam', 'beamfits', 'cst', 'feko'] or None.
            If None, the code attempts to guess what the file type is.
            based on file extensions
            (mwa_beam: .hdf5, .h5; cst: .yaml, .txt; beamfits: .fits, .beamfits).
            Note that if a list of datasets is passed, the file type is
            determined from the first dataset.
        skip_bad_files : bool
            Option when reading multiple files to catch read errors such that
            the read continues even if one or more files are corrupted. Files
            that produce errors will be printed. Default is False (files will
            not be skipped).

        Checking
        --------
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.

        BeamFITS
        --------
        az_range : tuple of float in deg
            The azimuth range to read in, if the beam is specified in az/za coordinates.
            Default is to read in all azimuths. Restricting the azimuth reduces peak
            memory usage. Only used for beamfits files that have their coordinates
            in az/za grid.
        za_range : tuple of float in deg
            The zenith angle range to read in, if the beam is specified in za/za
            coordinates. Default is to read in all za. Restricting the za reduces peak
            memory. Only used for beamfits files that have their coordinates
            in az/za grid.
        mount_type : str
            Antenna mount type, which describes the optics of the antenna in question.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.
        freq_range : array_like of float
            Range of frequencies to include in Hz, defaults to all available
            frequencies. Must be length 2. This will cause a *partial read* (i.e.
            reduce peak memory usage).

        CST
        ---
        beam_type : str
            What beam_type to read in ('power' or 'efield').
        feed_pol : str
            The feed or polarization or list of feeds or polarizations the
            files correspond to.
            Defaults to 'x' (meaning x for efield or xx for power beams).
        feed_array : str or array-like of str
            Feeds to define this beam for, e.g. x & y or r & l. Only used for power
            beams (feeds are set by feed_pol for efield beams).
        feed_angle : str or array-like of float
            Position angle of a given feed, units of radians. A feed angle of 0 is
            typically oriented toward zenith for steerable antennas, otherwise toward
            north for fixed antennas (e.g., HERA, LWA). More details on this can be
            found on the "Conventions" page of the docs. Must match shape of feed_pol
            for efield beams, or feed_angle for power beams.
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
        mount_type : str
            Antenna mount type, which describes the optics of the antenna in question.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.

        FEKO
        ----
        beam_type : str
            What beam_type to read in ('power' or 'efield').
        feed_pol : str or list of str
            The feed polarization that the files corresponds to, e.g. x, y, r or l.
            Defaults to 'x'.
        feed_angle : str or array-like of float
            Position angle of a given feed, units of radians. A feed angle of 0 is
            typically oriented toward zenith for steerable antennas, otherwise toward
            north for fixed antennas (e.g., HERA, LWA). More details on this can be
            found on the "Conventions" page of the docs. Must match shape of feed_pol
            for efield beams, or feed_angle for power beams.
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
        mount_type : str
            Antenna mount type, which describes the optics of the antenna in question.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.

        MWA
        ---
        delays : array of ints
            Array of MWA beamformer delay steps. Should be shape (n_pols, n_dipoles).
            Only applies to mwa_beam type files.
        amplitudes : array of floats
            Array of dipole amplitudes, these are absolute values
            (i.e. relatable to physical units).
            Should be shape (n_pols, n_dipoles).
            Only applies to mwa_beam type files.
        pixels_per_deg : float
            Number of theta/phi pixels per degree. Sets the resolution of the beam.
            Only applies to mwa_beam type files.
        freq_range : array_like of float
            Range of frequencies to include in Hz, defaults to all available
            frequencies. Must be length 2.

        Raises
        ------
        ValueError
            If the file_type is not set and cannot be determined from the file name.
        """
        if isinstance(filename, list | tuple | np.ndarray):
            multi = True
        else:
            multi = False

        if file_type is None:
            if multi:
                test_file = filename[0]
            else:
                test_file = filename

            basename, extension = os.path.splitext(test_file)
            extension = extension.lower()
            if extension == ".fits" or extension == ".beamfits":
                file_type = "beamfits"
            elif extension == ".hdf5" or extension == ".h5":
                file_type = "mwa_beam"
            elif extension == ".txt" or extension == ".yaml":
                file_type = "cst"
            elif extension == ".ffe":
                file_type = "feko"
                if multi:
                    feed_pol = utils.tools._get_iterable(feed_pol)
                    if len(feed_pol) != len(filename):
                        raise ValueError(
                            "If multiple FEKO files are passed, the feed_pol must "
                            "be a list or array of the same length giving the "
                            "feed_pol for each file."
                        )
                    feed_angle = utils.tools._get_iterable(feed_angle)
                    if len(feed_angle) != len(filename):
                        raise ValueError(
                            "If multiple FEKO files are passed, the feed_angle must "
                            "be a list or array of the same length giving the "
                            "feed_angle for each file."
                        )

        if file_type is None:
            raise ValueError(
                "File type could not be determined, use the "
                "file_type keyword to specify the type."
            )

        if file_type == "cst":
            # cst beams are already set up for multi
            # beam reading. Let it handle the hard work.
            self.read_cst_beam(
                filename,
                beam_type=beam_type,
                feed_pol=feed_pol,
                feed_array=feed_array,
                feed_angle=feed_angle,
                rotate_pol=rotate_pol,
                mount_type=mount_type,
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
                frequency_select=frequency_select,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )
        else:
            if multi:
                file_num = 0
                file_warnings = ""
                unread = True
                if file_type == "feko":
                    feed_pol_use = feed_pol[file_num]
                    feed_angle_use = feed_angle[file_num]
                else:
                    feed_pol_use = feed_pol
                    feed_angle_use = feed_angle

                while unread and file_num < len(filename):
                    try:
                        self.read(
                            filename[file_num],
                            file_type=file_type,
                            skip_bad_files=skip_bad_files,
                            # checking parameters
                            run_check=run_check,
                            check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability,
                            check_auto_power=check_auto_power,
                            fix_auto_power=fix_auto_power,
                            # beamfits parameters
                            az_range=az_range,
                            za_range=za_range,
                            # beamfits & mwa parameters
                            freq_range=freq_range,
                            # cst parameters
                            # leave these in case we restructure the multi
                            # reading later
                            beam_type=beam_type,
                            feed_pol=feed_pol_use,
                            feed_array=feed_array,
                            feed_angle=feed_angle_use,
                            rotate_pol=rotate_pol,
                            mount_type=mount_type,
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
                            frequency_select=frequency_select,
                            # mwa parameters
                            delays=delays,
                            amplitudes=amplitudes,
                            pixels_per_deg=pixels_per_deg,
                        )
                        unread = False
                    except ValueError as err:
                        file_warnings += (
                            f"Failed to read {filename[file_num]} "
                            f"due to ValueError {err}\n"
                        )
                        file_num += 1
                        if skip_bad_files is False:
                            raise
                beam_list = []
                if len(filename) >= file_num + 1:
                    for fname in filename[file_num + 1 :]:
                        beam2 = UVBeam()
                        if file_type == "feko":
                            feed_pol_use = feed_pol[file_num + 1]
                            feed_angle_use = feed_angle[file_num + 1]
                        else:
                            feed_pol_use = feed_pol
                            feed_angle_use = feed_angle
                        try:
                            beam2.read(
                                fname,
                                file_type=file_type,
                                skip_bad_files=skip_bad_files,
                                # checking parameters
                                run_check=run_check,
                                check_extra=check_extra,
                                run_check_acceptability=run_check_acceptability,
                                check_auto_power=check_auto_power,
                                fix_auto_power=fix_auto_power,
                                # beamfits parameters
                                az_range=az_range,
                                za_range=za_range,
                                # beamfits & mwa parameters
                                freq_range=freq_range,
                                # cst parameters
                                # leave these in case we restructure the multi
                                # reading later
                                beam_type=beam_type,
                                feed_pol=feed_pol_use,
                                feed_array=feed_array,
                                feed_angle=feed_angle_use,
                                rotate_pol=rotate_pol,
                                mount_type=mount_type,
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
                                frequency_select=frequency_select,
                                # mwa parameters
                                delays=delays,
                                amplitudes=amplitudes,
                                pixels_per_deg=pixels_per_deg,
                            )
                            beam_list.append(beam2)
                        except ValueError as err:
                            file_warnings += (
                                f"Failed to read {filename[file_num]} "
                                f"due to ValueError {err}\n"
                            )
                            if skip_bad_files:
                                continue
                            else:
                                raise
                        file_num += 1
                if unread is True:
                    warnings.warn(
                        "########################################################\n"
                        "ALL FILES FAILED ON READ - NO READABLE FILES IN FILENAME\n"
                        "########################################################"
                    )
                elif len(file_warnings) > 0:
                    warnings.warn(file_warnings)

                # Too much work to rewrite __add__ to operate on lists
                # of files, so instead doing a binary tree merge
                beam_list = [self] + beam_list
                while len(beam_list) > 1:
                    # for an odd number of files, the second argument will be shorter
                    # so the last element in the first list won't be combined, but it
                    # will not be lost, so it's ok.
                    for beam1, beam2 in zip(
                        beam_list[0::2], beam_list[1::2], strict=False
                    ):
                        beam1.__iadd__(beam2)
                    beam_list = beam_list[0::2]
                # Because self was at the beginning of the list,
                # everything is merged into it at the end of this loop
            else:
                if file_type == "mwa_beam":
                    self.read_mwa_beam(
                        filename,
                        delays=delays,
                        amplitudes=amplitudes,
                        pixels_per_deg=pixels_per_deg,
                        freq_range=freq_range,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                        check_auto_power=check_auto_power,
                        fix_auto_power=fix_auto_power,
                    )
                elif file_type == "beamfits":
                    self.read_beamfits(
                        filename,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                        check_auto_power=check_auto_power,
                        fix_auto_power=fix_auto_power,
                        az_range=az_range,
                        za_range=za_range,
                        freq_range=freq_range,
                        mount_type=mount_type,
                    )
                elif file_type == "feko":
                    self.read_feko_beam(
                        filename,
                        beam_type=beam_type,
                        feed_pol=feed_pol,
                        feed_angle=feed_angle,
                        mount_type=mount_type,
                        telescope_name=telescope_name,
                        feed_name=feed_name,
                        feed_version=feed_version,
                        model_name=model_name,
                        model_version=model_version,
                        history=history,
                        reference_impedance=reference_impedance,
                        extra_keywords=extra_keywords,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                        check_auto_power=check_auto_power,
                        fix_auto_power=fix_auto_power,
                    )

    @classmethod
    @copy_replace_short_description(read, style=DocstringStyle.NUMPYDOC)
    def from_file(cls, filename, **kwargs):
        """Initialize a new UVBeam object by reading the input file(s)."""
        uvbeam = cls()
        uvbeam.read(filename, **kwargs)
        return uvbeam

    def write_beamfits(self, filename, **kwargs):
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
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.
        clobber : bool
            Option to overwrite the filename if the file already exists.

        """
        beamfits_obj = self._convert_to_filetype("beamfits")
        beamfits_obj.write_beamfits(filename, **kwargs)
        del beamfits_obj


def _uvbeam_constructor(loader, node):
    """
    Define a yaml constructor for UVBeam objects.

    The yaml must specify a "filename" field pointing to the UVBeam readable file
    and any desired arguments to the UVBeam.from_file method. If the file does not
    exist, checks pyuvsim cache for file treating the "filename" as the cache url.

    Parameters
    ----------
    loader: yaml.Loader
        An instance of a yaml Loader object.
    node: yaml.Node
        A yaml node object.

    Returns
    -------
    UVBeam
        An instance of a UVBeam.

    """
    values = loader.construct_mapping(node, deep=True)
    if "freq_range" in values and len(values["freq_range"]) != 2:
        raise ValueError(
            "freq_range in yaml constructor must have 2 elements: "
            f"{values['freq_range']}"
        )
    if "filename" not in values:
        raise ValueError("yaml entries for UVBeam must specify a filename.")

    files_use = values["filename"]
    if isinstance(values["filename"], str):
        files_use = [values["filename"]]

    if "path_variable" in values:
        path_parts = (values.pop("path_variable")).split(".")
        var_name = path_parts[-1]
        if len(path_parts) == 1:
            raise ValueError(
                "If 'path_variable' is specified, it should take the form of a "
                "module.variable_name where the variable name can be imported "
                "from the module."
            )
        else:
            module = (".").join(path_parts[:-1])
            module = importlib.import_module(module)
        path_var = getattr(module, var_name)
        for f_i in range(len(files_use)):
            files_use[f_i] = os.path.join(path_var, files_use[f_i])

    for i, file in enumerate(files_use):
        # if file does not exist, check pyuvsim cache defined from astropy prescription
        # treat file as download url to check astropy cache for file
        if not os.path.exists(file) and is_url_in_cache(file, pkgname="pyuvsim"):
            files_use[i] = cache_contents("pyuvsim")[file]

    if len(files_use) == 1:
        files_use = files_use[0]
    values["filename"] = files_use

    beam = UVBeam.from_file(**values)

    return beam


yaml.add_constructor("!UVBeam", _uvbeam_constructor, Loader=yaml.SafeLoader)
yaml.add_constructor("!UVBeam", _uvbeam_constructor, Loader=yaml.FullLoader)


def _uvbeam_representer(dumper, beam):
    """
    Define a yaml representer for UVbeams.

    Note: since all the possible selects cannot be extracted from the object,
    the object generated from this yaml may not be an exact match for the object
    in memory. Also note that the filename parameter must not be None and must
    point to an existing file. It's likely that the user will need to update
    the filename parameter to include the full path.

    Parameters
    ----------
    dumper: yaml.Dumper
        An instance of a yaml Loader object.
    beam: UVBeam
        A UVbeam object, which must have a filename defined on it.

    Returns
    -------
    str
        The yaml representation of the UVbeam.

    """
    if beam.filename is None:
        raise ValueError(
            "beam must have a filename defined to be able to represent it in a yaml."
        )
    files_use = beam.filename
    if isinstance(files_use, str):
        files_use = [files_use]
    for file in files_use:
        if not os.path.exists(file):
            raise ValueError(
                "all entries in the filename parameter must be existing files "
                f"to be able to represent it in a yaml. {file} does not exist"
            )
    if len(files_use) == 1:
        files_use = files_use[0]

    # Add mount_type, since it's a new required keyword that some file formats did
    # not originally support.
    mapping = {"filename": files_use, "mount_type": beam.mount_type}

    return dumper.represent_mapping("!UVBeam", mapping)


yaml.add_representer(UVBeam, _uvbeam_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(UVBeam, _uvbeam_representer, Dumper=yaml.Dumper)
