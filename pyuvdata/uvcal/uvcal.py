# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Primary container for radio interferometer calibration solutions."""

import copy
import os
import threading
import warnings

import numpy as np
from docstring_parser import DocstringStyle

from .. import parameter as uvp
from .. import telescopes as uvtel
from .. import utils as uvutils
from ..docstrings import combine_docstrings
from ..uvbase import UVBase
from . import initializers

__all__ = ["UVCal"]

_future_array_shapes_warning = (
    "The shapes of several attributes will be changing in the future to remove the "
    "deprecated spectral window axis. You can call the `use_future_array_shapes` "
    "method to convert to the future array shapes now or set the parameter of the same "
    "name on this method to both convert to the future array shapes and silence this "
    "warning. See the UVCal tutorial on ReadTheDocs for more details about these "
    "shape changes."
)


class UVCal(UVBase):
    """
    A class defining calibration solutions for interferometric data.

    Attributes
    ----------
    UVParameter objects :
        For full list see the documentation on ReadTheDocs:
        http://pyuvdata.readthedocs.io/en/latest/.
        Some are always required, some are required for certain cal_types and cal_styles
        and others are always optional.

    """

    def __init__(self):
        self._Nfreqs = uvp.UVParameter(
            "Nfreqs", description="Number of frequency channels", expected_type=int
        )
        self._Njones = uvp.UVParameter(
            "Njones",
            description="Number of Jones calibration "
            "parameters (Number of Jones matrix elements "
            "calculated in calibration).",
            expected_type=int,
        )
        desc = (
            "Number of times with different calibrations calculated "
            "(if a calibration is calculated over a range of integrations, "
            "this gives the number of separate calibrations along the time axis)."
        )
        self._Ntimes = uvp.UVParameter("Ntimes", description=desc, expected_type=int)
        self._history = uvp.UVParameter(
            "history",
            description="String of history, units English",
            form="str",
            expected_type=str,
        )
        self._Nspws = uvp.UVParameter(
            "Nspws",
            description="Number of spectral windows "
            "(ie non-contiguous spectral chunks). ",
            expected_type=int,
        )

        desc = "Name of telescope. e.g. HERA. String."
        self._telescope_name = uvp.UVParameter(
            "telescope_name", description=desc, form="str", expected_type=str
        )

        desc = (
            "Number of antennas that have data associated with them "
            "(i.e. length of ant_array), which may be smaller than the number"
            "of antennas in the telescope (i.e. length of antenna_numbers)."
        )
        self._Nants_data = uvp.UVParameter(
            "Nants_data", description=desc, expected_type=int
        )

        desc = (
            "Number of antennas in the antenna_numbers array. May be larger "
            "than the number of antennas with gains associated with them."
        )
        self._Nants_telescope = uvp.UVParameter(
            "Nants_telescope", description=desc, expected_type=int
        )

        desc = (
            "Telescope location: xyz in ITRF (earth-centered frame). "
            "Can also be accessed using telescope_location_lat_lon_alt or "
            "telescope_location_lat_lon_alt_degrees properties"
        )
        self._telescope_location = uvp.LocationParameter(
            "telescope_location", description=desc, tols=1e-3, required=True
        )

        desc = (
            "Array of integer antenna numbers that appear in self.gain_array,"
            " with shape (Nants_data,). "
            "This array is ordered to match the inherent ordering of the zeroth"
            " axis of self.gain_array."
        )
        self._ant_array = uvp.UVParameter(
            "ant_array", description=desc, expected_type=int, form=("Nants_data",)
        )

        desc = (
            "Array of antenna names with shape (Nants_telescope,). "
            "Ordering of elements matches ordering of antenna_numbers."
        )
        self._antenna_names = uvp.UVParameter(
            "antenna_names",
            description=desc,
            form=("Nants_telescope",),
            expected_type=str,
        )

        desc = (
            "Array of all integer-valued antenna numbers in the telescope with "
            "shape (Nants_telescope,). Ordering of elements matches that of "
            "antenna_names. This array is not necessarily identical to "
            "ant_array, in that this array holds all antenna numbers "
            "associated with the telescope, not just antennas with data, and "
            "has an in principle non-specific ordering."
        )
        self._antenna_numbers = uvp.UVParameter(
            "antenna_numbers",
            description=desc,
            form=("Nants_telescope",),
            expected_type=int,
        )

        desc = (
            "Array giving coordinates of antennas relative to "
            "telescope_location (ITRF frame), shape (Nants_telescope, 3), "
            "units meters. See the tutorial page in the documentation "
            "for an example of how to convert this to topocentric frame."
        )
        self._antenna_positions = uvp.UVParameter(
            "antenna_positions",
            description=desc,
            form=("Nants_telescope", 3),
            expected_type=float,
            tols=1e-3,  # 1 mm
            required=True,
        )

        desc = (
            "Option to support 'wide-band' calibration solutions with gains or delays "
            "that apply over a range of frequencies rather than having distinct values "
            "at each frequency. Delay type cal solutions are always 'wide-band' if "
            "future_array_shapes is True. If it is True several other parameters are "
            "affected: future_array_shapes is also True; the data-like arrays have a "
            "spw axis that is Nspws long rather than a frequency axis that is Nfreqs "
            "long; the `freq_range` parameter is required and the `freq_array` and "
            "`channel_width` parameters are not required."
        )
        self._wide_band = uvp.UVParameter(
            "wide_band", description=desc, expected_type=bool, value=False
        )

        self._spw_array = uvp.UVParameter(
            "spw_array",
            description="Array of spectral window numbers, shape (Nspws).",
            form=("Nspws",),
            expected_type=int,
        )

        # this dimensionality of freq_array does not allow for different spws
        # to have different numbers of channels
        desc = (
            "Array of frequencies, center of the channel, "
            "shape (1, Nfreqs) or (Nfreqs,) if future_array_shapes=True, units Hz."
            "Not required if future_array_shapes=True and wide_band=True."
            "Should not be set if future_array_shapes=True and wide_band=True."
        )
        # TODO: Spw axis to be collapsed in future release
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=(1, "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )  # mHz

        desc = (
            "Width of frequency channels (Hz). If flex_spw = False and "
            "future_array_shapes=False, then it is a "
            "single value of type = float, otherwise it is an array of shape "
            "(Nfreqs,), type = float."
            "Should not be set if future_array_shapes=True and wide_band=True."
        )
        self._channel_width = uvp.UVParameter(
            "channel_width", description=desc, expected_type=float, tols=1e-3
        )  # 1 mHz

        desc = (
            "Required if cal_type='delay' or wide_band=True. Frequency range that "
            "solutions are valid for. If future_array_shapes is False it is a "
            "length 2 array with [start_frequency, end_frequency], otherwise it is an "
            "array of shape (Nspws, 2). Units are Hz."
            "Should not be set if cal_type='gain' and wide_band=False."
        )
        self._freq_range = uvp.UVParameter(
            "freq_range",
            required=False,
            description=desc,
            form=2,
            expected_type=float,
            tols=1e-3,
        )

        desc = (
            "Array of antenna polarization integers, shape (Njones). "
            "linear pols -5:-8 (jxx, jyy, jxy, jyx)."
            "circular pols -1:-4 (jrr, jll. jrl, jlr)."
        )

        self._jones_array = uvp.UVParameter(
            "jones_array",
            description=desc,
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 0)),
            form=("Njones",),
        )

        desc = (
            "Time range (in JD) that cal solutions are valid for."
            "list: [start_time, end_time] in JD. Should only be set if Ntimes is 1."
        )
        self._time_range = uvp.UVParameter(
            "time_range", description=desc, form=2, expected_type=float, required=False
        )

        desc = (
            "Array of calibration solution times, center of integration, "
            "shape (Ntimes), units Julian Date"
        )
        self._time_array = uvp.UVParameter(
            "time_array",
            description=desc,
            form=("Ntimes",),
            expected_type=float,
            tols=1e-3 / (60.0 * 60.0 * 24.0),
        )

        # standard angle tolerance: 1 mas in radians.
        desc = "Array of lsts, center of integration, shape (Ntimes), units radians"
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Ntimes",),
            expected_type=float,
            tols=uvutils.RADIAN_TOL,
            required=True,
        )

        desc = (
            "Integration time of a time bin, units seconds. "
            "If future_array_shapes=False, then it is a single value of type = float, "
            "otherwise it is an array of shape (Ntimes), type = float."
        )
        self._integration_time = uvp.UVParameter(
            "integration_time", description=desc, expected_type=float, tols=1e-3
        )  # 1ms

        desc = (
            "The convention for applying the calibration solutions to data."
            'Values are "divide" or "multiply", indicating that to calibrate '
            "one should divide or multiply uncalibrated data by gains. "
            "Mathematically this indicates the alpha exponent in the equation: "
            "calibrated data = gain^alpha * uncalibrated data. A value of "
            '"divide" represents alpha=-1 and "multiply" represents alpha=1.'
        )
        self._gain_convention = uvp.UVParameter(
            "gain_convention",
            form="str",
            expected_type=str,
            description=desc,
            acceptable_vals=["divide", "multiply"],
        )

        desc = (
            "Array of flags to be applied to calibrated data (logical OR "
            "of input and flag generated by calibration). True is flagged. "
            "Shape: (Nants_data, 1, Nfreqs, Ntimes, Njones) or "
            "(Nants_data, Nfreqs, Ntimes, Njones) if future_array_shapes=True and "
            "wide_band=False or (Nants_data, Nspws, Ntimes, Njones) if wide_band=True, "
            "type = bool."
        )
        self._flag_array = uvp.UVParameter(
            "flag_array",
            description=desc,
            form=("Nants_data", 1, "Nfreqs", "Ntimes", "Njones"),
            expected_type=bool,
        )

        desc = (
            "Array of qualities of calibration solutions. "
            "The shape depends on cal_type, if the cal_type is 'gain', the shape is: "
            "(Nants_data, 1, Nfreqs, Ntimes, Njones) or "
            "(Nants_data, Nfreqs, Ntimes, Njones) if future_array_shapes=True and "
            "wide_band=False or (Nants_data, Nspws, Ntimes, Njones) if wide_band=True, "
            "if the cal_type is 'delay', the shape is "
            "(Nants_data, 1, 1, Ntimes, Njones) or (Nants_data, Nspws, Ntimes, Njones) "
            "if future_array_shapes=True. The type is float."
        )
        self._quality_array = uvp.UVParameter(
            "quality_array",
            description=desc,
            form=("Nants_data", 1, "Nfreqs", "Ntimes", "Njones"),
            expected_type=float,
            required=False,
        )

        desc = (
            "Orientation of the physical dipole corresponding to what is "
            'labelled as the x polarization. Options are "east" '
            '(indicating east/west orientation) and "north" (indicating '
            "north/south orientation)"
        )
        self._x_orientation = uvp.UVParameter(
            "x_orientation",
            description=desc,
            expected_type=str,
            acceptable_vals=["east", "north"],
        )

        # --- cal_type parameters ---
        desc = "cal type parameter. Values are delay or gain."
        self._cal_type = uvp.UVParameter(
            "cal_type",
            form="str",
            expected_type=str,
            value="gain",
            description=desc,
            acceptable_vals=["delay", "gain", "unknown"],
        )

        desc = (
            'Required if cal_type = "gain". Array of gains, '
            "shape: (Nants_data, 1, Nfreqs, Ntimes, Njones) or "
            "(Nants_data, Nfreqs, Ntimes, Njones) if future_array_shapes=True, or "
            "(Nants_data, Nspws, Ntimes, Njones) if wide_band=True, "
            "type = complex float."
        )
        self._gain_array = uvp.UVParameter(
            "gain_array",
            description=desc,
            required=False,
            form=("Nants_data", 1, "Nfreqs", "Ntimes", "Njones"),
            expected_type=complex,
        )

        desc = (
            'Required if cal_type = "delay". Array of delays with units of seconds. '
            "Shape: (Nants_data, 1, 1, Ntimes, Njones) or "
            "(Nants_data, Nspws, Ntimes, Njones) if future_array_shapes=True, "
            "type=float."
        )
        self._delay_array = uvp.UVParameter(
            "delay_array",
            description=desc,
            required=False,
            form=("Nants_data", "Nspws", 1, "Ntimes", "Njones"),
            expected_type=float,
        )

        # --- flexible spectral window information ---

        desc = (
            "Option to construct a 'flexible spectral window', which stores"
            "all spectral channels across the frequency axis of data_array. "
            "Allows for spectral windows of variable sizes, and channels of "
            "varying widths."
        )
        self._flex_spw = uvp.UVParameter(
            "flex_spw", description=desc, expected_type=bool, value=False
        )

        desc = (
            "Required if flex_spw = True and will always be required for non-wide-band "
            "objects starting in version 3.0. Maps individual channels along the "
            "frequency axis to individual spectral windows, as listed in the "
            "spw_array. Shape (Nfreqs), type = int."
        )
        self._flex_spw_id_array = uvp.UVParameter(
            "flex_spw_id_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=int,
            required=False,
        )

        desc = "Flag indicating that this object is using the future array shapes."
        self._future_array_shapes = uvp.UVParameter(
            "future_array_shapes", description=desc, expected_type=bool, value=False
        )

        # --- cal_style parameters ---
        desc = "Style of calibration. Values are sky or redundant."
        self._cal_style = uvp.UVParameter(
            "cal_style",
            form="str",
            expected_type=str,
            description=desc,
            acceptable_vals=["sky", "redundant"],
        )

        desc = (
            "Deprecated, will be removed in version 2.5. Only used if cal_style is "
            "'sky'. Short string describing field center or dominant source."
        )
        self._sky_field = uvp.UVParameter(
            "sky_field", form="str", required=False, expected_type=str, description=desc
        )

        desc = 'Required if cal_style = "sky". Name of calibration catalog.'
        self._sky_catalog = uvp.UVParameter(
            "sky_catalog",
            form="str",
            required=False,
            expected_type=str,
            description=desc,
        )

        desc = 'Required if cal_style = "sky". Phase reference antenna.'
        self._ref_antenna_name = uvp.UVParameter(
            "ref_antenna_name",
            form="str",
            required=False,
            expected_type=str,
            description=desc,
        )

        desc = "Number of sources used."
        self._Nsources = uvp.UVParameter(
            "Nsources", required=False, expected_type=int, description=desc
        )

        desc = "Range of baselines used for calibration."
        self._baseline_range = uvp.UVParameter(
            "baseline_range",
            form=2,
            required=False,
            expected_type=float,
            description=desc,
        )

        desc = "Name of diffuse model."
        self._diffuse_model = uvp.UVParameter(
            "diffuse_model",
            form="str",
            required=False,
            expected_type=str,
            description=desc,
        )

        # --- truly optional parameters ---
        desc = (
            "The gain scale of the calibration, which indicates the units of the "
            "calibrated visibilities. For example, Jy or K str."
        )
        self._gain_scale = uvp.UVParameter(
            "gain_scale",
            form="str",
            expected_type=str,
            description=desc,
            required=False,
        )

        desc = (
            "Deprecated, support will be removed in version 2.5. Array of input flags, "
            "True is flagged. shape: (Nants_data, 1, Nfreqs, Ntimes, Njones) or "
            "(Nants_data, Nfreqs, Ntimes, Njones) if future_array_shapes=True, "
            "type = bool."
        )
        self._input_flag_array = uvp.UVParameter(
            "input_flag_array",
            description=desc,
            required=False,
            form=("Nants_data", 1, "Nfreqs", "Ntimes", "Njones"),
            expected_type=bool,
        )

        desc = "Origin (on github for e.g) of calibration software. Url and branch."
        self._git_origin_cal = uvp.UVParameter(
            "git_origin_cal",
            form="str",
            expected_type=str,
            description=desc,
            required=False,
        )

        desc = (
            "Commit hash of calibration software (from git_origin_cal) used "
            "to generate solutions."
        )
        self._git_hash_cal = uvp.UVParameter(
            "git_hash_cal",
            form="str",
            expected_type=str,
            description=desc,
            required=False,
        )

        desc = "Name of observer who calculated solutions in this file."
        self._observer = uvp.UVParameter(
            "observer", form="str", description=desc, expected_type=str, required=False
        )

        desc = (
            "Array of qualities of the calibration for entire arrays. "
            "The shape depends on cal_type, if the cal_type is 'gain', "
            "the shape is: (1, Nfreqs, Ntimes, Njones) or "
            "(Nfreqs, Ntimes, Njones) if future_array_shapes=True and wide_band=False, "
            "or (Nspws, Ntimes, Njones) if wide_band=True. "
            "If the cal_type is 'delay', the shape is (1, 1, Ntimes, Njones) or "
            "(Nspws, Ntimes, Njones) if future_array_shapes=True, type = float."
        )
        self._total_quality_array = uvp.UVParameter(
            "total_quality_array",
            description=desc,
            form=(1, "Nfreqs", "Ntimes", "Njones"),
            expected_type=float,
            required=False,
        )

        desc = (
            "Any user supplied extra keywords, type=dict. Keys should be "
            "8 character or less strings if writing to calfits files. "
            "Use the special key 'comment' for long multi-line string comments."
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
            "List of strings containing the unique basenames (not the full path) of "
            "input files."
        )
        self._filename = uvp.UVParameter(
            "filename", required=False, description=desc, expected_type=str
        )

        super(UVCal, self).__init__()

    @staticmethod
    @combine_docstrings(initializers.new_uvcal, style=DocstringStyle.NUMPYDOC)
    def new(**kwargs):
        """
        Create a new UVCal object.

        All parameters are passed through to
        the :func:`~pyuvdata.uvcal.initializers.new_uvcal` function.

        Returns
        -------
        UVCal
            A new UVCal object.
        """
        return initializers.new_uvcal(**kwargs)

    def _set_flex_spw(self):
        """
        Set flex_spw to True, and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by the file-reading methods to indicate that an object has multiple spectral
        windows concatenated together across the frequency axis.
        """
        assert (
            not self.wide_band
        ), "Cannot set objects wide_band objects to have flexible spectral windows."
        # Mark once-optional arrays as now required
        self.flex_spw = True
        self._flex_spw_id_array.required = True
        # Now make sure that chan_width is set to be an array
        self._channel_width.form = ("Nfreqs",)

    def _set_wide_band(self, wide_band=True):
        """
        Set the wide_band parameter and adjust required parameters.

        The wide_band can only be set to True if future_array_shapes is True.

        This method should not be called directly by users; instead it is called
        by the file-reading methods to indicate that an object is a wide-band
        calibration solution which supports gain or delay values per spectral window.

        """
        if wide_band:
            assert (
                self.future_array_shapes
            ), "future_array_shapes must be True to set wide_band to True."
            # TODO think about what to test for in flex_spw_id_array
            assert (
                not self.flex_spw
            ), "Cannot set objects with flexible spectral windows to wide_band."
        elif self.future_array_shapes:
            assert self.cal_type != "delay", (
                "delay objects cannot have wide_band=False if future_array_shapes is "
                "True"
            )
        self.wide_band = wide_band

        if wide_band:
            self._freq_array.required = False
            self._channel_width.required = False
            # in version 3.0, also set _flex_spw_id_array to not be required
            self._freq_range.required = True

            data_shape_params = [
                "gain_array",
                "delay_array",
                "flag_array",
                "input_flag_array",
                "quality_array",
            ]
            data_form = ("Nants_data", "Nspws", "Ntimes", "Njones")
            tot_qual_form = ("Nspws", "Ntimes", "Njones")

            for param_name in self._data_params:
                if param_name in data_shape_params:
                    getattr(self, "_" + param_name).form = data_form
                elif param_name == "total_quality_array":
                    getattr(self, "_" + param_name).form = tot_qual_form

        else:
            self._freq_array.required = True
            self._channel_width.required = True
            # in version 3.0, also set _flex_spw_id_array to be required
            self._freq_range.required = False

            if self.future_array_shapes:
                # can only get here if not a delay solution
                data_shape_params = [
                    "gain_array",
                    "flag_array",
                    "input_flag_array",
                    "quality_array",
                ]

                data_form = ("Nants_data", "Nfreqs", "Ntimes", "Njones")
                tot_qual_form = ("Nfreqs", "Ntimes", "Njones")

                for param_name in self._data_params:
                    if param_name in data_shape_params:
                        getattr(self, "_" + param_name).form = data_form
                    elif param_name == "total_quality_array":
                        getattr(self, "_" + param_name).form = tot_qual_form

    def _set_gain(self):
        """Set cal_type to 'gain' and adjust required parameters."""
        self.cal_type = "gain"
        self._gain_array.required = True
        self._delay_array.required = False
        self._freq_range.required = False
        self._freq_array.required = True
        self._channel_width.required = True
        self._quality_array.form = self._gain_array.form
        self._total_quality_array.form = self._gain_array.form[1:]

    def _set_delay(self):
        """Set cal_type to 'delay' and adjust required parameters."""
        self.cal_type = "delay"
        self._gain_array.required = False
        self._delay_array.required = True
        self._freq_range.required = True
        self._freq_array.required = False
        self._channel_width.required = False
        self._quality_array.form = self._delay_array.form
        self._total_quality_array.form = self._delay_array.form[1:]

        if self.future_array_shapes:
            self._set_wide_band()

    def _set_unknown_cal_type(self):
        """Set cal_type to 'unknown' and adjust required parameters.

        Deprecated.
        """
        warnings.warn(
            "Setting the cal_type to 'unknown' is deprecated. This will become an "
            "error in version 2.5",
            DeprecationWarning,
        )
        self.cal_type = "unknown"
        self._gain_array.required = False
        self._delay_array.required = False
        self._freq_range.required = False
        self._freq_array.required = True
        self._quality_array.form = self._gain_array.form
        self._total_quality_array.form = self._gain_array.form[1:]

    def _set_sky(self):
        """Set cal_style to 'sky' and adjust required parameters."""
        self.cal_style = "sky"
        self._sky_catalog.required = True
        self._ref_antenna_name.required = True

    def _set_redundant(self):
        """Set cal_style to 'redundant' and adjust required parameters."""
        self.cal_style = "redundant"
        self._sky_catalog.required = False
        self._ref_antenna_name.required = False

    @property
    def _data_params(self):
        """List of strings giving the data-like parameters."""
        return [
            "gain_array",
            "delay_array",
            "flag_array",
            "quality_array",
            "total_quality_array",
            "input_flag_array",
        ]

    @property
    def _required_data_params(self):
        """List of strings giving the required data-like parameters."""
        cal_type = self._cal_type.value
        if cal_type is None:
            raise ValueError(
                "Cannot get required data params because cal_type is not set."
            )

        if cal_type == "gain":
            return ["gain_array", "flag_array"]
        elif cal_type == "delay":
            return ["delay_array", "flag_array"]
        else:
            return ["flag_array"]

    @property
    def data_like_parameters(self):
        """Iterate defined parameters which are data-like (not metadata-like)."""
        for key in self._data_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def metadata_only(self):
        """
        Property that determines whether this is a metadata only object.

        An object is metadata only if data_array, nsample_array and flag_array
        are all None.
        """
        metadata_only = all(d is None for d in self.data_like_parameters)

        for param_name in self._required_data_params:
            getattr(self, "_" + param_name).required = not metadata_only

        return metadata_only

    def _set_future_array_shapes(self):
        """
        Set future_array_shapes to True and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by file-reading methods and `use_future_array_shapes` to indicate the
        `future_array_shapes` is True and define expected parameter shapes.

        """
        self.future_array_shapes = True
        self._freq_array.form = ("Nfreqs",)
        self._channel_width.form = ("Nfreqs",)
        self._integration_time.form = ("Ntimes",)
        self._freq_range.form = ("Nspws", 2)

        data_shape_params = [
            "gain_array",
            "flag_array",
            "input_flag_array",
            "quality_array",
        ]
        if self.cal_type == "delay":
            self._set_wide_band()
            data_shape_params.append("delay_array")

        if self.wide_band:
            data_form = ("Nants_data", "Nspws", "Ntimes", "Njones")
            tot_qual_form = ("Nspws", "Ntimes", "Njones")
        else:
            data_form = ("Nants_data", "Nfreqs", "Ntimes", "Njones")
            tot_qual_form = ("Nfreqs", "Ntimes", "Njones")

        for param_name in self._data_params:
            if param_name in data_shape_params:
                getattr(self, "_" + param_name).form = data_form
            if param_name == "delay_array":
                # only get here if cal_type is not "delay"
                self._delay_array.form = ("Nants_data", "Nspws", "Ntimes", "Njones")
            elif param_name == "total_quality_array":
                getattr(self, "_" + param_name).form = tot_qual_form

    def use_future_array_shapes(self):
        """
        Change the array shapes of this object to match the planned future shapes.

        This method sets allows users to convert to the planned array shapes changes
        before the changes go into effect. This method sets the `future_array_shapes`
        parameter on this object to True.

        """
        if self.future_array_shapes:
            return

        self._set_future_array_shapes()
        if not self.metadata_only:
            # remove the length-1 spw axis for all data-like parameters
            # except the delay array, which should have the length-1 freq axis removed
            for param_name in self._data_params:
                param_value = getattr(self, param_name)
                if param_value is None:
                    continue
                if param_name == "delay_array":
                    setattr(self, param_name, (param_value)[:, :, 0, :, :])
                elif param_name == "total_quality_array":
                    setattr(self, param_name, (param_value)[0, :, :, :])
                else:
                    setattr(self, param_name, (param_value)[:, 0, :, :, :])

            if self.cal_type == "delay":
                warnings.warn(
                    "When converting a delay-style cal to future array shapes the "
                    "flag_array (and input_flag_array if it exists) must drop the "
                    "frequency axis so that it will be the same shape as the "
                    "delay_array. This will be done using the "
                    "`pyuvdata.utils.and_collapse` function which will only flag an "
                    "antpol-time if all of the frequecies are flagged for that "
                    "antpol-time. To preserve the full flag information, create a "
                    "UVFlag object from this cal object before this operation. "
                    "In the future, these flag arrays will be removed from UVCal "
                    "objects in favor of using UVFlag objects."
                )
                self.flag_array = uvutils.and_collapse(self.flag_array, axis=1)[
                    :, np.newaxis, :, :
                ]
                if self.input_flag_array is not None:
                    self.input_flag_array = uvutils.and_collapse(
                        self.input_flag_array, axis=1
                    )[:, np.newaxis, :, :]

        if self.cal_type == "delay":
            self.freq_array = None
            self.channel_width = None
            self.Nfreqs = 1
        else:
            self.freq_range = None

        # remove the length-1 spw axis for the freq_array
        if self.freq_array is not None:
            self.freq_array = self.freq_array[0, :]

        if self.freq_range is not None:
            # force freq_range to have an spw axis
            self.freq_range = np.repeat(
                (np.asarray(self.freq_range))[np.newaxis, :], self.Nspws, axis=0
            )

        # force integration_time to be an array of length Ntimes
        self.integration_time = (
            np.zeros(self.Ntimes, dtype=np.float64) + self.integration_time
        )

        if not self.flex_spw and self.channel_width is not None:
            # make channel_width be an array of length Nfreqs rather than a single value
            # (not needed with flexible spws because this is already done in that case)
            self.channel_width = (
                np.zeros(self.Nfreqs, dtype=np.float64) + self.channel_width
            )

    def use_current_array_shapes(self):
        """
        Change the array shapes of this object to match the current shapes.

        This method sets allows users to convert back to the current array shapes.
        This method sets the `future_array_shapes` parameter on this object to False.
        """
        warnings.warn(
            "This method will be removed in version 3.0 when the current array shapes "
            "are no longer supported.",
            DeprecationWarning,
        )
        if not self.future_array_shapes:
            return

        if self.Nspws > 1 and self.wide_band:
            raise ValueError(
                "Cannot use current array shapes if Nspws > 1 and wide_band is True"
            )

        if self.cal_type != "delay" and self.wide_band:
            raise ValueError(
                "Cannot use current array shapes if cal_style is not 'delay' and "
                "wide_band is True."
            )

        if not self.flex_spw:
            if self.channel_width is not None:
                unique_channel_widths = np.unique(self.channel_width)
                if unique_channel_widths.size > 1:
                    raise ValueError(
                        "channel_width parameter contains multiple unique values, but "
                        "only one spectral window is present. Cannot collapse "
                        "channel_width to a single value."
                    )
                self._channel_width.form = ()
                self.channel_width = unique_channel_widths[0]

        unique_integration_times = np.unique(self.integration_time)
        if unique_integration_times.size > 1:
            raise ValueError(
                "integration_time parameter contains multiple unique values. "
                "Cannot collapse integration_time to a single value."
            )
        self._integration_time.form = ()
        self.integration_time = unique_integration_times[0]

        self.future_array_shapes = False
        self.wide_band = False

        gain_shape_params = ["gain_array", "flag_array", "input_flag_array"]
        delay_shape_params = ["delay_array"]
        if self.cal_type == "delay":
            delay_shape_params.append("quality_array")
        else:
            gain_shape_params.append("quality_array")

        for param_name in self._data_params:
            if param_name in gain_shape_params:
                getattr(self, "_" + param_name).form = (
                    "Nants_data",
                    1,
                    "Nfreqs",
                    "Ntimes",
                    "Njones",
                )
            elif param_name in delay_shape_params:
                getattr(self, "_" + param_name).form = (
                    "Nants_data",
                    "Nspws",
                    1,
                    "Ntimes",
                    "Njones",
                )
            elif param_name == "total_quality_array":
                if self.cal_type == "delay":
                    getattr(self, "_" + param_name).form = (1, 1, "Ntimes", "Njones")
                else:
                    getattr(self, "_" + param_name).form = (
                        1,
                        "Nfreqs",
                        "Ntimes",
                        "Njones",
                    )

        if not self.metadata_only:
            for param_name in self._data_params:
                param_value = getattr(self, param_name)
                if param_value is None:
                    continue
                if param_name == "delay_array":
                    setattr(
                        self,
                        param_name,
                        (getattr(self, param_name))[:, :, np.newaxis, :, :],
                    )
                elif param_name == "total_quality_array":
                    setattr(
                        self,
                        param_name,
                        (getattr(self, param_name))[np.newaxis, :, :, :],
                    )
                else:
                    setattr(
                        self,
                        param_name,
                        (getattr(self, param_name))[:, np.newaxis, :, :, :],
                    )
            if self.cal_type == "delay":
                # make the flag array have a frequency axis again
                self.flag_array = np.repeat(self.flag_array, self.Nfreqs, axis=2)
                if self.input_flag_array is not None:
                    self.input_flag_array = np.repeat(
                        self.input_flag_array, self.Nfreqs, axis=2
                    )

        self._freq_array.form = (1, "Nfreqs")
        if self.freq_array is not None:
            self.freq_array = self.freq_array[np.newaxis, :]
        elif self.Nfreqs == 1:
            self.freq_array = np.full((1, 1), self.freq_range[0, 0])

        self._freq_range.form = (2,)
        if self.freq_range is not None:
            self.freq_range = self.freq_range[0, :].tolist()

    def set_telescope_params(self, overwrite=False):
        """
        Set telescope related parameters.

        If the telescope_name is in the known_telescopes, set the telescope
        location to the value for the known telescope. Also set the antenna positions
        if they are not set on the object and are available for the telescope.

        Parameters
        ----------
        overwrite : bool
            Option to overwrite existing telescope-associated parameters with
            the values from the known telescope.

        Raises
        ------
        ValueError
            if the telescope_name is not in known telescopes
        """
        telescope_obj = uvtel.get_telescope(self.telescope_name)
        if telescope_obj is not False:
            if self.telescope_location is None or overwrite is True:
                warnings.warn(
                    "telescope_location is not set. Using known values "
                    f"for {telescope_obj.telescope_name}."
                )
                self.telescope_location = telescope_obj.telescope_location

            if telescope_obj.antenna_positions is not None and (
                self.antenna_positions is None or overwrite is True
            ):
                ant_inds = []
                telescope_ant_inds = []
                # first try to match using names only
                for index, antname in enumerate(self.antenna_names):
                    if antname in telescope_obj.antenna_names:
                        ant_inds.append(index)
                        telescope_ant_inds.append(
                            np.where(telescope_obj.antenna_names == antname)[0][0]
                        )
                # next try using numbers
                if len(ant_inds) != self.Nants_telescope:
                    for index, antnum in enumerate(self.antenna_numbers):
                        # only update if not already found
                        if (
                            index not in ant_inds
                            and antnum in telescope_obj.antenna_numbers
                        ):
                            this_ant_ind = np.where(
                                telescope_obj.antenna_numbers == antnum
                            )[0][0]
                            # make sure we don't already have this antenna associated
                            # with another antenna
                            if this_ant_ind not in telescope_ant_inds:
                                ant_inds.append(index)
                                telescope_ant_inds.append(this_ant_ind)
                if len(ant_inds) != self.Nants_telescope:
                    warnings.warn(
                        "Not all antennas have positions in the telescope object. "
                        "Not setting antenna_positions."
                    )
                else:
                    params_set = ["antenna_positions"]
                    if overwrite:
                        self.antenna_names = telescope_obj.antenna_names
                        self.antenna_numbers = telescope_obj.antenna_numbers
                        self.antenna_positions = telescope_obj.antenna_positions
                        self.Nants_telescope = telescope_obj.Nants_telescope
                        params_set += [
                            "antenna_names",
                            "antenna_numbers",
                            "Nants_telescope",
                        ]
                    else:
                        telescope_ant_inds = np.array(telescope_ant_inds)
                        self.antenna_positions = telescope_obj.antenna_positions[
                            telescope_ant_inds, :
                        ]
                    params_set_str = ", ".join(params_set)
                    warnings.warn(
                        f"{params_set_str} are not set or are being "
                        "overwritten. Using known values for "
                        f"{telescope_obj.telescope_name}."
                    )

        else:
            raise ValueError(
                f"Telescope {self.telescope_name} is not in known_telescopes."
            )

    def _set_lsts_helper(self, astrometry_library=None):
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
        self.lst_array = uvutils.get_lst_for_time(
            jd_array=self.time_array,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            astrometry_library=astrometry_library,
            frame=self._telescope_location.frame,
        )
        return

    def set_lsts_from_time_array(self, background=False, astrometry_library=None):
        """Set the lst_array based from the time_array.

        Parameters
        ----------
        background : bool, False
            When set to True, start the calculation on a threading.Thread in the
            background and return the thread to the user.

        Returns
        -------
        proc : None or threading.Thread instance
            When background is set to True, a thread is returned which must be
            joined before the lst_array exists on the UVCal object.

        """
        if not background:
            self._set_lsts_helper(astrometry_library=astrometry_library)
            return
        else:
            proc = threading.Thread(
                target=self._set_lsts_helper,
                kwargs={"astrometry_library": astrometry_library},
            )
            proc.start()
            return proc

    def _check_flex_spw_contiguous(self):
        """
        Check if the spectral windows are contiguous for flex_spw datasets.

        This checks the flex_spw_id_array to make sure that all channels for each
        spectral window are together in one block, versus being interspersed (e.g.,
        channel #1 and #3 is in spw #1, channels #2 and #4 are in spw #2).

        """
        if self.flex_spw:
            uvutils._check_flex_spw_contiguous(self.spw_array, self.flex_spw_id_array)

    def _check_freq_spacing(self, raise_errors=True):
        """
        Check if frequencies are evenly spaced and separated by their channel width.

        This is a requirement for writing calfits files.

        Parameters
        ----------
        raise_errors : bool
            Option to raise errors if the various checks do not pass.

        Returns
        -------
        spacing_error : bool
            Flag that channel spacings or channel widths are not equal.
        chanwidth_error : bool
            Flag that channel spacing does not match channel width.

        """
        if self.freq_array is None and self.Nfreqs == 1:
            return False, False
        return uvutils._check_freq_spacing(
            self.freq_array,
            self._freq_array.tols,
            self.channel_width,
            self._channel_width.tols,
            self.flex_spw,
            self.future_array_shapes,
            self.spw_array,
            self.flex_spw_id_array,
            raise_errors=raise_errors,
        )

    def check(
        self,
        check_extra=True,
        run_check_acceptability=True,
        check_freq_spacing=False,
        lst_tol=uvutils.LST_RAD_TOL,
    ):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check all parameters, otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.
        check_freq_spacing :  bool
            Option to check if frequencies are evenly spaced and the spacing is
            equal to their channel_width. This is not required for UVCal
            objects in general but is required to write to calfits files.
        lst_tol : float or None
            Tolerance level at which to test LSTs against their expected values. If
            provided as a float, must be in units of radians. If set to None, the
            default precision tolerance from the `lst_array` parameter is used (1 mas).
            Default value is 75 mas,  which is set by the predictive uncertainty in IERS
            calculations of DUT1 (RMS is of order 1 ms, with with a 5-sigma threshold
            for detection is used to prevent false issues from being reported), which
            for some observatories sets the precision with which these values are
            written.

        Returns
        -------
        bool
            True if check passes

        Raises
        ------
        ValueError
            if parameter shapes or types are wrong or do not have acceptable
            values (if run_check_acceptability is True)

        """
        # Make sure requirements are set properly for cal_style
        if self.cal_style == "sky":
            self._set_sky()
        elif self.cal_style == "redundant":
            self._set_redundant()

        # if wide_band is True, Nfreqs must be 1.
        if self.wide_band:
            if self.Nfreqs != 1:
                warnings.warn(
                    "Nfreqs will be required to be 1 for wide_band cals (including "
                    "all delay cals) starting in version 3.0",
                    category=DeprecationWarning,
                )

        # deprecate 'unknown' cal_type
        if self.cal_type == "unknown":
            warnings.warn(
                "The 'unknown' cal_type is deprecated and will be removed in version "
                "2.5",
                DeprecationWarning,
            )

        # deprecate sky_field
        if self.sky_field is not None:
            warnings.warn(
                "The sky_field parameter is deprecated and will be removed in version "
                "2.5",
                DeprecationWarning,
            )

        # first run the basic check from UVBase
        super(UVCal, self).check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # require that all entries in ant_array exist in antenna_numbers
        if not all(ant in self.antenna_numbers for ant in self.ant_array):
            raise ValueError("All antennas in ant_array must be in antenna_numbers.")

        # issue warning if extra_keywords keys are longer than 8 characters
        for key in self.extra_keywords.keys():
            if len(key) > 8:
                warnings.warn(
                    "key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 if written "
                    "to a calfits file format.".format(key=key)
                )

        # issue warning if extra_keywords values are lists, arrays or dicts
        for key, value in self.extra_keywords.items():
            if isinstance(value, (list, dict, np.ndarray)):
                warnings.warn(
                    "{key} in extra_keywords is a list, array or dict, "
                    "which will raise an error when writing calfits "
                    "files".format(key=key)
                )

        if not self.wide_band and self.cal_type == "gain":
            if self.flex_spw_id_array is None:
                warnings.warn(
                    "flex_spw_id_array is not set. It will be required starting in "
                    "version 3.0 for non-wide-band objects",
                    DeprecationWarning,
                )
            else:
                # Check that all values in flex_spw_id_array are entries in the
                # spw_array
                if not np.all(np.isin(self.flex_spw_id_array, self.spw_array)):
                    raise ValueError(
                        "All values in the flex_spw_id_array must exist in the "
                        "spw_array."
                    )
        # warn if freq_range or freq_array set when it shouldn't be
        if (
            self.cal_type == "gain"
            and not self.wide_band
            and self.freq_range is not None
        ):
            warnings.warn(
                "The freq_range attribute should not be set if cal_type='gain' "
                "and wide_band=False. This will become an error in version 3.0.",
                DeprecationWarning,
            )
        if self.wide_band:
            if self.freq_array is not None:
                warnings.warn(
                    "The freq_array attribute should not be set if wide_band=True. "
                    "This will become an error in version 3.0.",
                    DeprecationWarning,
                )

            if self.channel_width is not None:
                warnings.warn(
                    "The channel_width attribute should not be set if wide_band=True. "
                    "This will become an error in version 3.0.",
                    DeprecationWarning,
                )

        if self.input_flag_array is not None:
            warnings.warn(
                "The input_flag_array is deprecated and will be removed in version 2.5",
                DeprecationWarning,
            )

        if check_freq_spacing:
            self._check_freq_spacing()

        if run_check_acceptability:
            # Check antenna positions
            uvutils.check_surface_based_positions(
                antenna_positions=self.antenna_positions,
                telescope_loc=self.telescope_location,
                telescope_frame=self._telescope_location.frame,
                raise_error=False,
            )

            lat, lon, alt = self.telescope_location_lat_lon_alt_degrees
            uvutils.check_lsts_against_times(
                jd_array=self.time_array,
                lst_array=self.lst_array,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                lst_tols=self._lst_array.tols if lst_tol is None else [0, lst_tol],
                frame=self._telescope_location.frame,
            )

        return True

    def copy(self, metadata_only=False):
        """
        Make and return a copy of the UVCal object.

        Parameters
        ----------
        metadata_only : bool
            If True, only copy the metadata of the object.

        Returns
        -------
        UVCal
            Copy of self.
        """
        if not metadata_only:
            return super(UVCal, self).copy()
        else:
            uv = UVCal()
            # include all attributes, not just UVParameter ones.
            for attr in self.__iter__(uvparams_only=False):
                # skip properties
                if isinstance(getattr(type(self), attr, None), property):
                    continue

                # skip data like parameters
                # parameter names have a leading underscore we want to ignore
                if attr.lstrip("_") in self._data_params:
                    continue
                setattr(uv, attr, copy.deepcopy(getattr(self, attr)))

            return uv

    def _has_key(self, antnum=None, jpol=None):
        """
        Check if this UVCal has the requested antenna or polarization.

        Parameters
        ----------
        antnum : int
            Antenna number to check.
        jpol : str or int
            Antenna polarization string or integer to check.

        Returns
        -------
        bool
            Boolean indicator of whether the antenna and/or antenna
            polarization is present on this object.
        """
        if antnum is not None:
            if antnum not in self.ant_array:
                return False
        if jpol is not None:
            if isinstance(jpol, (str, np.str_)):
                jpol = uvutils.jstr2num(jpol, x_orientation=self.x_orientation)
            if jpol not in self.jones_array:
                return False

        return True

    def ant2ind(self, antnum):
        """
        Get the index in data arrays for an antenna number.

        Parameters
        ----------
        antnum : int
            Antenna number to get index for.

        Returns
        -------
        int
            Antenna index in data arrays.
        """
        if not self._has_key(antnum=antnum):
            raise ValueError("{} not found in ant_array".format(antnum))

        return np.argmin(np.abs(self.ant_array - antnum))

    def jpol2ind(self, jpol):
        """
        Get the index in data arrays for an antenna polarization.

        Parameters
        ----------
        jpol : int or str
            Antenna polarization to get index for.

        Returns
        -------
        int
            Antenna polarization index in data arrays
        """
        if isinstance(jpol, (str, np.str_)):
            jpol = uvutils.jstr2num(jpol, x_orientation=self.x_orientation)

        if not self._has_key(jpol=jpol):
            raise ValueError("{} not found in jones_array".format(jpol))

        return np.argmin(np.abs(self.jones_array - jpol))

    def _slice_array(self, key, data_array, squeeze_pol=True):
        """
        Slice a data array given a data key.

        Parameters
        ----------
        key : int or length 2 tuple of ints or int and str
            Antenna or antenna and polarization to get slice for. If it's a length
            2 tuple, the second value must be an antenna polarization int or string
            parsable by jpol2ind.
        data_array : :class: numpy ndarray
            Array to get slice of. Must have the shape of the gain_array or delay_array.
        squeeze_pol : bool
            Option to squeeze pol dimension if possible.

        Returns
        -------
        :class: numpy ndarray
            Slice of the data_array for the key.
        """
        key = uvutils._get_iterable(key)
        if len(key) == 1:
            # interpret as a single antenna
            if self.future_array_shapes:
                output = data_array[self.ant2ind(key[0]), :, :, :]
            else:
                output = data_array[self.ant2ind(key[0]), 0, :, :, :]
            if squeeze_pol and output.shape[-1] == 1:
                output = output[:, :, 0]
            return output
        elif len(key) == 2:
            # interpret as an antenna-pol pair
            if self.future_array_shapes:
                output = data_array[self.ant2ind(key[0]), :, :, self.jpol2ind(key[1])]
            else:
                output = data_array[
                    self.ant2ind(key[0]), 0, :, :, self.jpol2ind(key[1])
                ]
            return output

    def _parse_key(self, ant, jpol=None):
        """
        Parse key inputs and return a standard antenna-polarization key.

        Parameters
        ----------
        ant : int or length 2 tuple of ints or int and str
            Antenna or antenna and polarization to get key for. If it's a length
            2 tuple, the second value must be an antenna polarization int or string
            parsable by jpol2ind.
        jpol : int or str
            Antenna polarization int or string parsable by jpol2ind. Only used
            if `ant` is an integer.

        Returns
        -------
        tuple
            Standard key tuple.

        """
        if isinstance(ant, (list, tuple)):
            # interpret ant as (ant,) or (ant, jpol)
            key = tuple(ant)
        elif isinstance(ant, (int, np.integer)):
            # interpret ant as antenna number
            key = (ant,)
            # add jpol if fed
            if jpol is not None:
                key += (jpol,)

        return key

    def get_gains(self, ant, jpol=None, squeeze_pol=True):
        """
        Get the gain associated with an antenna and/or polarization.

        Parameters
        ----------
        ant : int or length 2 tuple of ints or int and str
            Antenna or antenna and polarization to get gains for. If it's a length
            2 tuple, the second value must be an antenna polarization int or string
            parsable by jpol2ind.
        jpol : int or str, optional
            Instrumental polarization to request. Ex. 'Jxx'
        squeeze_pol : bool
            Option to squeeze pol dimension if possible.

        Returns
        -------
        complex ndarray
            Gain solution of shape (Nfreqs, Ntimes, Njones) or (Nfreqs, Ntimes)
            if jpol is set or if squeeze_pol is True and Njones = 1.
        """
        if self.cal_type != "gain":
            raise ValueError("cal_type must be 'gain' for get_gains() method")

        return self._slice_array(
            self._parse_key(ant, jpol=jpol), self.gain_array, squeeze_pol=squeeze_pol
        )

    def get_flags(self, ant, jpol=None, squeeze_pol=True):
        """
        Get the flags associated with an antenna and/or polarization.

        Parameters
        ----------
        ant : int or length 2 tuple of ints or int and str
            Antenna or antenna and polarization to get gains for. If it's a length
            2 tuple, the second value must be an antenna polarization int or string
            parsable by jpol2ind.
        jpol : int or str, optional
            Instrumental polarization to request. Ex. 'Jxx'
        squeeze_pol : bool
            Option to squeeze pol dimension if possible.

        Returns
        -------
        boolean ndarray
            Flags of shape (Nfreqs, Ntimes, Njones) or (Nfreqs, Ntimes)
            if jpol is set or if squeeze_pol is True and Njones = 1.
        """
        return self._slice_array(
            self._parse_key(ant, jpol=jpol), self.flag_array, squeeze_pol=squeeze_pol
        )

    def get_quality(self, ant, jpol=None, squeeze_pol=True):
        """
        Get the qualities associated with an antenna and/or polarization.

        Parameters
        ----------
        ant : int or length 2 tuple of ints or int and str
            Antenna or antenna and polarization to get gains for. If it's a length
            2 tuple, the second value must be an antenna polarization int or string
            parsable by jpol2ind.
        jpol : int or str, optional
            Instrumental polarization to request. Ex. 'Jxx'
        squeeze_pol : bool
            Option to squeeze pol dimension if possible.

        Returns
        -------
        float ndarray
            Qualities of shape (Nfreqs, Ntimes, Njones) or (Nfreqs, Ntimes)
            if jpol is not None or if squeeze_pol is True and Njones = 1.
        """
        return self._slice_array(
            self._parse_key(ant, jpol=jpol), self.quality_array, squeeze_pol=squeeze_pol
        )

    def reorder_antennas(
        self,
        order="number",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Arrange the antenna axis according to desired order.

        Parameters
        ----------
        order: str or array like of int
            If a string, allowed values are "name" and "number" to sort on the antenna
            name or number respectively. A '-' can be prepended to signify descending
            order instead of the default ascending order (e.g. "-number"). An array of
            integers of length Nants representing indexes along the existing `ant_array`
            can also be supplied to sort in any desired order (note these are indices
            into the `ant_array` not antenna numbers).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if order is not an allowed string or is an array that does not
            contain all the required numbers.

        """
        if isinstance(order, (np.ndarray, list, tuple)):
            order = np.array(order)
            if not order.size == self.Nants_data or not np.all(
                np.sort(order) == np.arange(self.Nants_data)
            ):
                raise ValueError(
                    "If order is an index array, it must contain all indicies for the"
                    "ant_array, without duplicates."
                )
            index_array = order
        else:
            if order not in ["number", "name", "-number", "-name"]:
                raise ValueError(
                    "order must be one of 'number', 'name', '-number', '-name' or an "
                    "index array of length Nants_data"
                )

            if "number" in order:
                index_array = np.argsort(self.ant_array)
            elif "name" in order:
                temp = np.asarray(self.antenna_names)
                dtype_use = temp.dtype
                name_array = np.zeros_like(self.ant_array, dtype=dtype_use)
                # there has to be a better way to do this without a loop...
                for ind, ant in enumerate(self.ant_array):
                    name_array[ind] = self.antenna_names[
                        np.nonzero(self.antenna_numbers == ant)[0][0]
                    ]
                index_array = np.argsort(name_array)

            if order[0] == "-":
                index_array = np.flip(index_array)

        if np.all(index_array[1:] > index_array[:-1]):
            # Nothing to do - the data are already sorted!
            return

        # update all the relevant arrays
        self.ant_array = self.ant_array[index_array]
        for param_name in self._data_params:
            if param_name == "total_quality_array":
                continue
            param = getattr(self, param_name)
            if param is not None:
                setattr(self, param_name, param[index_array])

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def reorder_freqs(
        self,
        spw_order=None,
        channel_order=None,
        select_spw=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Arrange the frequency axis according to desired order.

        Parameters
        ----------
        spw_order : str or array_like of int
            A string describing the desired order of spectral windows along the
            frequecy axis. Allowed strings include `number` (sort on spectral window
            number) and `freq` (sort on median frequency). A '-' can be prepended
            to signify descending order instead of the default ascending order,
            e.g., if you have SPW #1 and 2, and wanted them ordered as [2, 1],
            you would specify `-number`. Alternatively, one can supply an index array
            of length Nspws that specifies how to shuffle the spws (this is not the
            desired final spw order). Default is to apply no sorting of spectral
            windows.
        channel_order : str or array_like of int
            A string describing the desired order of frequency channels within a
            spectral window. Allowed strings are "freq" and "-freq", which will sort
            channels within a spectral window by ascending or descending frequency
            respectively.  Alternatively, one can supply an index array of length
            Nfreqs that specifies the new order. Default is to apply no sorting of
            channels within a single spectral window. Note that proving an array_like
            of ints will cause the values given to `spw_order` and `select_spw` to be
            ignored.
        select_spw : int or array_like of int
            An int or array_like of ints which specifies which spectral windows to
            apply sorting. Note that setting this argument will cause the value
            given to `spw_order` to be ignored.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if select_spw contains values not in spw_array, or if channel_order
            is not the same length as freq_array.

        """
        if self.Nspws == 1 and self.Nfreqs == 1:
            warnings.warn(
                "Cannot reorder the frequency/spw axis with only one frequency and "
                "spw. Returning the object unchanged."
            )
            return

        if self.wide_band:
            # we have an spw axis, not a frequency axis
            if channel_order is not None or select_spw is not None:
                warnings.warn(
                    "channel_order and select_spws are ignored for wide-band "
                    "calibration solutions"
                )
            if spw_order is None:
                # default sensibly
                spw_order = "number"

            flip_spws = spw_order[0] == "-"

            if isinstance(spw_order, (np.ndarray, list, tuple)):
                spw_order = np.asarray(spw_order)
                if not spw_order.size == self.Nspws or not np.all(
                    np.sort(spw_order) == np.arange(self.Nspws)
                ):
                    raise ValueError(
                        "If spw_order is an array, it must contain all indicies for "
                        "the spw_array, without duplicates."
                    )
                index_array = np.asarray(
                    [
                        np.nonzero(self.spw_array == spw)[0][0]
                        for spw in self.spw_array[spw_order]
                    ]
                )
            else:
                if spw_order not in ["number", "freq", "-number", "-freq", None]:
                    raise ValueError(
                        "spw_order can only be one of 'number', '-number', "
                        "'freq', '-freq', None or an index array of length Nspws"
                    )
                if "number" in spw_order:
                    index_array = np.argsort(self.spw_array)
                elif "freq" in spw_order:
                    mean_freq = np.mean(self.freq_range, axis=1)
                    index_array = np.argsort(mean_freq)

            if flip_spws:
                index_array = np.flip(index_array)

        else:
            index_array = uvutils._sort_freq_helper(
                self.Nfreqs,
                self.freq_array,
                self.Nspws,
                self.spw_array,
                self.flex_spw,
                self.flex_spw_id_array,
                self.future_array_shapes,
                spw_order,
                channel_order,
                select_spw,
            )

            if index_array is None:
                # This only happens if no sorting is needed
                return

        # update all the relevant arrays
        if self.future_array_shapes:
            if self.wide_band:
                self.spw_array = self.spw_array[index_array]
                self.freq_range = self.freq_range[index_array]
            else:
                self.freq_array = self.freq_array[index_array]
            for param_name in self._data_params:
                param = getattr(self, param_name)
                if param is not None:
                    if param_name == "total_quality_array":
                        self.total_quality_array = self.total_quality_array[index_array]
                    else:
                        setattr(self, param_name, param[:, index_array])
        else:
            self.freq_array = self.freq_array[:, index_array]
            if self.cal_type != "delay":
                for param_name in self._data_params:
                    param = getattr(self, param_name)
                    if param is not None:
                        if param_name == "total_quality_array":
                            self.total_quality_array = self.total_quality_array[
                                :, index_array
                            ]
                        else:
                            setattr(self, param_name, param[:, :, index_array])
        if self.flex_spw_id_array is not None:
            self.flex_spw_id_array = self.flex_spw_id_array[index_array]

            if self.Nspws > 1:
                # Reorder the spw-axis items based on their first appearance in the data
                orig_spw_array = self.spw_array
                unique_index = np.sort(
                    np.unique(self.flex_spw_id_array, return_index=True)[1]
                )
                self.spw_array = self.flex_spw_id_array[unique_index]
                spw_index = np.asarray(
                    [np.nonzero(orig_spw_array == spw)[0][0] for spw in self.spw_array]
                )
                if self.freq_range is not None and self.future_array_shapes:
                    # this can go away in v3 becuse freq_range will always be None
                    self.freq_range = self.freq_range[spw_index, :]

        if (self.future_array_shapes or self.flex_spw) and not self.wide_band:
            self.channel_width = self.channel_width[index_array]

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def reorder_times(
        self,
        order="time",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Arrange the time axis according to desired order.

        Parameters
        ----------
        order: str or array like of int
            If a string, allowed value is "time" or "-time" to sort on the time in
            ascending or descending order respectively. An array of integers of length
            Ntimes representing indexes along the existing `time_array` can also be
            supplied to sort in any desired order.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if order is not an allowed string or is an array that does not
            contain all the required indices.

        """
        if isinstance(order, (np.ndarray, list, tuple)):
            order = np.array(order)
            if not order.size == self.Ntimes or not np.all(
                np.sort(order) == np.arange(self.Ntimes)
            ):
                raise ValueError(
                    "If order is an array, it must contain all indicies for "
                    "the time axis, without duplicates."
                )
            index_array = order
        else:
            if order not in ["time", "-time"]:
                raise ValueError(
                    "order must be one of 'time', '-time' or an "
                    "index array of length Ntimes"
                )

            index_array = np.argsort(self.time_array)

            if order[0] == "-":
                index_array = np.flip(index_array)

        if np.all(index_array[1:] > index_array[:-1]):
            # Nothing to do - the data are already sorted!
            return

        # update all the relevant arrays
        self.time_array = self.time_array[index_array]
        self.lst_array = self.lst_array[index_array]
        if self.future_array_shapes:
            self.integration_time = self.integration_time[index_array]
            for param_name in self._data_params:
                param = getattr(self, param_name)
                if param is not None:
                    if param_name == "total_quality_array":
                        self.total_quality_array = self.total_quality_array[
                            :, index_array
                        ]
                    else:
                        setattr(self, param_name, param[:, :, index_array])
        else:
            for param_name in self._data_params:
                param = getattr(self, param_name)
                if param is not None:
                    if param_name == "total_quality_array":
                        self.total_quality_array = self.total_quality_array[
                            :, :, index_array
                        ]
                    else:
                        setattr(self, param_name, param[:, :, :, index_array])

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def reorder_jones(
        self,
        order="name",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Arrange the jones element axis according to desired order.

        Parameters
        ----------
        order: str or array like of int
            If a string, allowed values are "name" and "number" to sort on the jones
            element name or number respectively. A '-' can be prepended to signify
            descending order instead of the default ascending order (e.g. "-number").
            An array of integers of length Njones representing indexes along the
            existing `jones_array` can also be supplied to sort in any desired order.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if order is not an allowed string or is an array that does not
            contain all the required indices.

        """
        if isinstance(order, (np.ndarray, list, tuple)):
            order = np.array(order)
            if not order.size == self.Njones or not np.all(
                np.sort(order) == np.arange(self.Njones)
            ):
                raise ValueError(
                    "If order is an array, it must contain all indicies for "
                    "the jones axis, without duplicates."
                )
            index_array = order
        else:
            if order not in ["number", "name", "-number", "-name"]:
                raise ValueError(
                    "order must be one of 'number', 'name', '-number', '-name' or an "
                    "index array of length Njones"
                )

            if "number" in order:
                index_array = np.argsort(self.jones_array)
            elif "name" in order:
                name_array = np.asarray(
                    uvutils.jnum2str(self.jones_array, x_orientation=self.x_orientation)
                )
                index_array = np.argsort(name_array)

            if order[0] == "-":
                index_array = np.flip(index_array)

        if np.all(index_array[1:] > index_array[:-1]):
            # Nothing to do - the data are already sorted!
            return

        # update all the relevant arrays
        self.jones_array = self.jones_array[index_array]
        if self.future_array_shapes:
            for param_name in self._data_params:
                param = getattr(self, param_name)
                if param is not None:
                    if param_name == "total_quality_array":
                        self.total_quality_array = self.total_quality_array[
                            :, :, index_array
                        ]
                    else:
                        setattr(self, param_name, param[:, :, :, index_array])
        else:
            for param_name in self._data_params:
                param = getattr(self, param_name)
                if param is not None:
                    if param_name == "total_quality_array":
                        self.total_quality_array = self.total_quality_array[
                            :, :, :, index_array
                        ]
                    else:
                        setattr(self, param_name, param[:, :, :, :, index_array])

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def convert_to_gain(
        self,
        freq_array=None,
        channel_width=None,
        delay_convention="minus",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Convert non-gain cal_types to gains.

        For the delay cal_type the gain is calculated as:
            gain = 1 * exp((+/-) * 2 * pi * j * delay * frequency)
            where the (+/-) is dictated by the delay_convention

        Parameters
        ----------
        delay_convention : str
            Exponent sign to use in the conversion, can be "plus" or "minus".
        freq_array : array of float
            Frequencies to convert to gain at, units Hz. Not providing a freq_array is
            deprecated, but until version 3.0, if it is not provided and `freq_array`
            exists on the object, `freq_array` will be used.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting.

        """
        if self.cal_type == "gain":
            raise ValueError("The data is already a gain cal_type.")
        elif self.cal_type != "delay":
            # TODO remove this when the unknown cal_type is removed.
            raise ValueError("cal_type is unknown, cannot convert to gain")

        if self.Nspws > 1:
            raise ValueError(
                "convert_to_gain currently does not support multiple spectral windows"
            )

        if delay_convention == "minus":
            conv = -1
        elif delay_convention == "plus":
            conv = 1
        else:
            raise ValueError("delay_convention can only be 'minus' or 'plus'")

        if freq_array is None or channel_width is None:
            if self.freq_array is None or self.channel_width is None:
                raise ValueError(
                    "freq_array and channel_width must be provided if there is no "
                    "freq_array or no channel_width on the object."
                )

            warnings.warn(
                "In version 3.0 and later freq_array and channel_width will be "
                "required parameters. Using the freq_array and channel_width on the "
                "object.",
                category=DeprecationWarning,
            )
            if self.future_array_shapes:
                freq_array_use = self.freq_array
                channel_width = self.channel_width
            else:
                freq_array_use = self.freq_array[0, :]
                channel_width = self.channel_width
            Nfreqs_use = self.Nfreqs
        else:
            if freq_array.ndim > 1:
                raise ValueError("freq_array parameter must be a one dimensional array")
            if self.future_array_shapes:
                if (
                    not isinstance(channel_width, np.ndarray)
                    or channel_width.shape != freq_array.shape
                ):
                    raise ValueError(
                        "This object is using the future array shapes, so the "
                        "channel_width parameter be an array shaped like the freq_array"
                    )
            else:
                if isinstance(channel_width, np.ndarray):
                    if channel_width.size > 1:
                        raise ValueError(
                            "This object is using the current array shapes, so the "
                            "channel_width parameter must be a scalar value"
                        )
                    channel_width = channel_width[0]
            if self.freq_range is not None:
                # Already errored if more than one spw, so just use the first one here
                if not self.future_array_shapes:
                    freq_range_use = np.asarray(self.freq_range)
                else:
                    freq_range_use = self.freq_range[0, :]
                if np.any(freq_array < freq_range_use[0]) or np.any(
                    freq_array > freq_range_use[1]
                ):
                    raise ValueError(
                        "freq_array contains values outside the freq_range."
                    )
            freq_array_use = freq_array
            Nfreqs_use = freq_array.size

        self.history += "  Converted from delays to gains using pyuvdata."

        if self.future_array_shapes:
            phase_array = np.zeros(
                (self.Nants_data, Nfreqs_use, self.Ntimes, self.Njones)
            )
        else:
            phase_array = np.zeros(
                (self.Nants_data, 1, Nfreqs_use, self.Ntimes, self.Njones)
            )

        if self.future_array_shapes:
            temp = (
                conv
                * 2
                * np.pi
                * np.dot(
                    self.delay_array[:, 0, :, :, np.newaxis],
                    freq_array_use[np.newaxis, :],
                )
            )
            temp = np.transpose(temp, (0, 3, 1, 2))
            phase_array = temp
        else:
            temp = (
                conv
                * 2
                * np.pi
                * np.dot(
                    self.delay_array[:, 0, 0, :, :, np.newaxis],
                    freq_array_use[np.newaxis, :],
                )
            )
            temp = np.transpose(temp, (0, 3, 1, 2))
            phase_array[:, 0, :, :, :] = temp

        gain_array = np.exp(1j * phase_array)
        if self.future_array_shapes:
            freq_axis = 1
        else:
            freq_axis = 2
        self._set_gain()
        self._set_wide_band(wide_band=False)
        self.channel_width = channel_width
        self.freq_range = None
        self.gain_array = gain_array
        self.delay_array = None
        if self.quality_array is not None:
            new_quality = np.repeat(self.quality_array, Nfreqs_use, axis=freq_axis)
            self.quality_array = new_quality
        if self.Nfreqs > 1 and not self.future_array_shapes:
            if (
                self.freq_array is None
                or self.Nfreqs != Nfreqs_use
                or not np.allclose(
                    self.freq_array,
                    freq_array_use,
                    rtol=self._freq_array.tols[0],
                    atol=self._freq_array.tols[1],
                )
            ):
                warnings.warn(
                    "Existing flag array has a frequency axis of length > 1 but "
                    "frequencies do not match freq_array. The existing flag array "
                    "(and input_flag_array if it exists) will be collapsed using "
                    "the `pyuvdata.utils.and_collapse` function which will only "
                    "flag an antpol-time if all of the frequecies are flagged for "
                    "that antpol-time. Then it will be broadcast to all the new "
                    "frequencies. To preserve the original flag information, "
                    "create a UVFlag object from this cal object before this "
                    "operation. In the future, these flag arrays will be removed from "
                    "UVCal objects in favor of using UVFlag objects."
                )
                new_flag_array = np.expand_dims(
                    uvutils.and_collapse(self.flag_array, axis=freq_axis),
                    axis=freq_axis,
                )
                self.flag_array = np.repeat(new_flag_array, Nfreqs_use, axis=freq_axis)
                if self.input_flag_array is not None:
                    new_input_flag_array = np.expand_dims(
                        uvutils.and_collapse(self.input_flag_array, axis=freq_axis),
                        axis=freq_axis,
                    )
                    self.input_flag_array = np.repeat(
                        new_input_flag_array, Nfreqs_use, axis=freq_axis
                    )
        else:
            new_flag_array = np.repeat(self.flag_array, Nfreqs_use, axis=freq_axis)
            self.flag_array = new_flag_array
            if self.input_flag_array is not None:
                new_input_flag_array = np.repeat(
                    self.input_flag_array, Nfreqs_use, axis=freq_axis
                )
                self.input_flag_array = new_input_flag_array

        if self.total_quality_array is not None:
            if self.future_array_shapes:
                freq_axis = 0
            else:
                freq_axis = 1
            new_total_quality_array = np.repeat(
                self.total_quality_array, Nfreqs_use, axis=freq_axis
            )
            self.total_quality_array = new_total_quality_array
        if self.future_array_shapes:
            self.freq_array = freq_array_use
        else:
            self.freq_array = freq_array_use[np.newaxis, :]
        self.Nfreqs = Nfreqs_use

        self.flex_spw_id_array = np.full(self.Nfreqs, self.spw_array[0], dtype=int)

        # check if object is self-consistent
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def __add__(
        self,
        other,
        verbose_history=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=False,
    ):
        """
        Combine two UVCal objects along antenna, frequency, time, and/or Jones axis.

        Parameters
        ----------
        other : :class: UVCal
            Another UVCal object which will be added to self.
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            two objects are different, the combined object will keep all the history of
            both input objects (if many objects are combined in succession this can
            lead to very long histories). If False and if the histories for the two
            objects are different, the combined object will have the history of the
            first object and only the parts of the second object history that are unique
            (this is done word by word and can result in hard to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        inplace : bool
            Option to overwrite self as we go, otherwise create a third object
            as the sum of the two.
        """
        if inplace:
            this = self
        else:
            this = self.copy()
        # Check that both objects are UVCal and valid
        this.check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )
        if not issubclass(other.__class__, this.__class__):
            if not issubclass(this.__class__, other.__class__):
                raise ValueError(
                    "Only UVCal (or subclass) objects can be added to "
                    "a UVCal (or subclass) object"
                )
        other.check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # check that both objects have the same array shapes
        if this.future_array_shapes != other.future_array_shapes:
            raise ValueError(
                "Both objects must have the same `future_array_shapes` parameter. "
                "Use the `use_future_array_shapes` or `use_current_array_shapes` "
                "methods to convert them."
            )

        # Check to make sure that both objects are consistent w/ use of flex_spw
        if this.flex_spw != other.flex_spw:
            raise ValueError(
                "To combine these data, flex_spw must be set to the same "
                "value (True or False) for both objects."
            )

        # Check that both objects are either wide_band or not
        if this.wide_band != other.wide_band:
            raise ValueError(
                "To combine these data, wide_band must be set to the same "
                "value (True or False) for both objects."
            )

        this_has_spw_id = this.flex_spw_id_array is not None
        other_has_spw_id = other.flex_spw_id_array is not None
        if this_has_spw_id != other_has_spw_id:
            warnings.warn(
                "One object has the flex_spw_id_array set and one does not. Combined "
                "object will have it set."
            )

        # Check objects are compatible
        compatibility_params = [
            "_cal_type",
            "_telescope_name",
            "_gain_convention",
            "_x_orientation",
            "_cal_style",
            "_ref_antenna_name",
        ]
        if not this.future_array_shapes:
            compatibility_params.append("_integration_time")
            if not this.flex_spw:
                compatibility_params.append("_channel_width")

        warning_params = [
            "_observer",
            "_git_hash_cal",
            "_sky_field",
            "_sky_catalog",
            "_Nsources",
            "_baseline_range",
            "_diffuse_model",
        ]

        for a in compatibility_params:
            if getattr(this, a) != getattr(other, a):
                msg = (
                    "UVParameter " + a[1:] + " does not match. Cannot combine objects."
                )
                raise ValueError(msg)
        for a in warning_params:
            if getattr(this, a) != getattr(other, a):
                msg = "UVParameter " + a[1:] + " does not match. Combining anyway."
                warnings.warn(msg)

        # Build up history string
        history_update_string = " Combined data along "
        n_axes = 0

        # Check we don't have overlapping data
        both_jones, this_jones_ind, other_jones_ind = np.intersect1d(
            this.jones_array, other.jones_array, return_indices=True
        )
        both_times, this_times_ind, other_times_ind = np.intersect1d(
            this.time_array, other.time_array, return_indices=True
        )
        if this.cal_type != "delay" and not this.wide_band:
            # With flexible spectral window, the handling here becomes a bit funky,
            # because we are allowed to have channels with the same frequency *if* they
            # belong to different spectral windows (one real-life example: might want
            # to preserve guard bands in the correlator, which can have overlaping RF
            # frequency channels)
            if this.flex_spw:
                this_freq_ind = np.array([], dtype=np.int64)
                other_freq_ind = np.array([], dtype=np.int64)
                both_freq = np.array([], dtype=float)
                both_spw = np.intersect1d(this.spw_array, other.spw_array)
                for idx in both_spw:
                    this_mask = np.where(this.flex_spw_id_array == idx)[0]
                    other_mask = np.where(other.flex_spw_id_array == idx)[0]
                    if this.future_array_shapes:
                        both_spw_freq, this_spw_ind, other_spw_ind = np.intersect1d(
                            this.freq_array[this_mask],
                            other.freq_array[other_mask],
                            return_indices=True,
                        )
                    else:
                        both_spw_freq, this_spw_ind, other_spw_ind = np.intersect1d(
                            this.freq_array[0, this_mask],
                            other.freq_array[0, other_mask],
                            return_indices=True,
                        )
                    this_freq_ind = np.append(this_freq_ind, this_mask[this_spw_ind])
                    other_freq_ind = np.append(
                        other_freq_ind, other_mask[other_spw_ind]
                    )
                    both_freq = np.append(both_freq, both_spw_freq)
            else:
                if this.future_array_shapes:
                    both_freq, this_freq_ind, other_freq_ind = np.intersect1d(
                        this.freq_array, other.freq_array, return_indices=True
                    )
                else:
                    both_freq, this_freq_ind, other_freq_ind = np.intersect1d(
                        this.freq_array[0, :],
                        other.freq_array[0, :],
                        return_indices=True,
                    )

        elif this.wide_band:
            # this is really about spws, but that replaces the freq axis for wide_band
            both_freq, this_freq_ind, other_freq_ind = np.intersect1d(
                this.spw_array, other.spw_array, return_indices=True
            )
        else:
            # delay type cal
            # Make a non-empty array so we raise an error if other data is duplicated
            both_freq = [0]
            this_freq_ind = []

        both_ants, this_ants_ind, other_ants_ind = np.intersect1d(
            this.ant_array, other.ant_array, return_indices=True
        )
        if len(both_jones) > 0:
            if len(both_times) > 0:
                if len(both_freq) > 0:
                    if len(both_ants) > 0:
                        raise ValueError(
                            "These objects have overlapping data and"
                            " cannot be combined."
                        )

        # Next, we want to make sure that the ordering of the _overlapping_ data is
        # the same, so that things can get plugged together in a sensible way.
        if len(this_ants_ind) != 0:
            this_argsort = np.argsort(this_ants_ind)
            other_argsort = np.argsort(other_ants_ind)

            if np.any(this_argsort != other_argsort):
                temp_ind = np.arange(this.Nants_data)
                temp_ind[this_ants_ind[this_argsort]] = temp_ind[
                    this_ants_ind[other_argsort]
                ]

                this.reorder_antennas(temp_ind)

        if len(this_times_ind) != 0:
            this_argsort = np.argsort(this_times_ind)
            other_argsort = np.argsort(other_times_ind)

            if np.any(this_argsort != other_argsort):
                temp_ind = np.arange(this.Ntimes)
                temp_ind[this_times_ind[this_argsort]] = temp_ind[
                    this_times_ind[other_argsort]
                ]

                this.reorder_times(temp_ind)

        if len(this_freq_ind) != 0:
            this_argsort = np.argsort(this_freq_ind)
            other_argsort = np.argsort(other_freq_ind)
            if np.any(this_argsort != other_argsort):
                if this.wide_band:
                    temp_ind = np.arange(this.Nspws)
                    temp_ind[this_freq_ind[this_argsort]] = temp_ind[
                        this_freq_ind[other_argsort]
                    ]

                    this.reorder_freqs(spw_order=temp_ind)
                else:
                    temp_ind = np.arange(this.Nfreqs)
                    temp_ind[this_freq_ind[this_argsort]] = temp_ind[
                        this_freq_ind[other_argsort]
                    ]

                    this.reorder_freqs(channel_order=temp_ind)

        if len(this_jones_ind) != 0:
            this_argsort = np.argsort(this_jones_ind)
            other_argsort = np.argsort(other_jones_ind)
            if np.any(this_argsort != other_argsort):
                temp_ind = np.arange(this.Njones)
                temp_ind[this_jones_ind[this_argsort]] = temp_ind[
                    this_jones_ind[other_argsort]
                ]

                this.reorder_jones(temp_ind)

        # Update filename parameter
        this.filename = uvutils._combine_filenames(this.filename, other.filename)
        if this.filename is not None:
            this._filename.form = (len(this.filename),)

        temp = np.nonzero(~np.in1d(other.ant_array, this.ant_array))[0]
        if len(temp) > 0:
            anew_inds = temp
            history_update_string += "antenna"
            n_axes += 1
        else:
            anew_inds = []

        temp = np.nonzero(~np.in1d(other.time_array, this.time_array))[0]
        if len(temp) > 0:
            tnew_inds = temp
            if n_axes > 0:
                history_update_string += ", time"
            else:
                history_update_string += "time"
            n_axes += 1
        else:
            tnew_inds = []

        if this.cal_type == "gain" and not this.wide_band:
            # find the freq indices in "other" but not in "this"
            if self.flex_spw:
                other_mask = np.ones_like(other.flex_spw_id_array, dtype=bool)
                for idx in np.intersect1d(this.spw_array, other.spw_array):
                    if this.future_array_shapes:
                        other_mask[other.flex_spw_id_array == idx] = np.isin(
                            other.freq_array[other.flex_spw_id_array == idx],
                            this.freq_array[this.flex_spw_id_array == idx],
                            invert=True,
                        )
                    else:
                        other_mask[other.flex_spw_id_array == idx] = np.isin(
                            other.freq_array[0, other.flex_spw_id_array == idx],
                            this.freq_array[0, this.flex_spw_id_array == idx],
                            invert=True,
                        )
                temp = np.where(other_mask)[0]
            else:
                if this.future_array_shapes:
                    temp = np.nonzero(~np.in1d(other.freq_array, this.freq_array))[0]
                else:
                    temp = np.nonzero(
                        ~np.in1d(other.freq_array[0, :], this.freq_array[0, :])
                    )[0]
            if len(temp) > 0:
                fnew_inds = temp
                if n_axes > 0:
                    history_update_string += ", frequency"
                else:
                    history_update_string += "frequency"
                n_axes += 1
            else:
                fnew_inds = []
        elif this.wide_band:
            temp = np.nonzero(~np.in1d(other.spw_array, this.spw_array))[0]
            if len(temp) > 0:
                fnew_inds = temp
                if n_axes > 0:
                    history_update_string += ", spectral window"
                else:
                    history_update_string += "spectral window"
                n_axes += 1
            else:
                fnew_inds = []
        else:
            # adding along frequency axis is not supported for old delay-type
            # delay type, set fnew_inds to an empty list
            fnew_inds = []

        temp = np.nonzero(~np.in1d(other.jones_array, this.jones_array))[0]
        if len(temp) > 0:
            jnew_inds = temp
            if n_axes > 0:
                history_update_string += ", jones"
            else:
                history_update_string += "jones"
            n_axes += 1
        else:
            jnew_inds = []

        # Initialize tqa variables
        can_combine_tqa = True
        if this.cal_type == "delay":
            Nf_tqa = 1
        else:
            Nf_tqa = this.Nfreqs

        if (
            not self.metadata_only
            and this.input_flag_array is None
            and other.input_flag_array is not None
        ):
            this.input_flag_array = np.full(
                this._input_flag_array.expected_shape(this), True, dtype=bool
            )

        if (
            not self.metadata_only
            and this.quality_array is None
            and other.quality_array is not None
        ):
            this.quality_array = np.zeros(
                this._quality_array.expected_shape(this), dtype=float
            )

        # Pad out self to accommodate new data
        ant_order = None
        if len(anew_inds) > 0:
            this.ant_array = np.concatenate(
                [this.ant_array, other.ant_array[anew_inds]]
            )
            ant_order = np.argsort(this.ant_array)
            if not this.metadata_only:
                data_array_shape = getattr(this, this._required_data_params[0]).shape
                if this.future_array_shapes:
                    zero_pad_data = np.zeros(
                        (len(anew_inds), data_array_shape[1], this.Ntimes, this.Njones)
                    )
                    zero_pad_flags = np.zeros(
                        (len(anew_inds), data_array_shape[1], this.Ntimes, this.Njones)
                    )
                else:
                    zero_pad_data = np.zeros(
                        (
                            len(anew_inds),
                            1,
                            data_array_shape[2],
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    zero_pad_flags = np.zeros(
                        (len(anew_inds), 1, this.Nfreqs, this.Ntimes, this.Njones)
                    )
                if this.cal_type == "delay":
                    this.delay_array = np.concatenate(
                        [this.delay_array, zero_pad_data], axis=0
                    )
                else:
                    this.gain_array = np.concatenate(
                        [this.gain_array, zero_pad_data], axis=0
                    )
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad_flags], axis=0
                ).astype(np.bool_)
                if this.quality_array is not None:
                    this.quality_array = np.concatenate(
                        [this.quality_array, zero_pad_data], axis=0
                    )

                # If total_quality_array exists, we set it to None and warn the user
                if (
                    this.total_quality_array is not None
                    or other.total_quality_array is not None
                ):
                    warnings.warn(
                        "Total quality array detected in at least one file; the "
                        "array in the new object will be set to 'None' because "
                        "whole-array values cannot be combined when adding antennas"
                    )
                    this.total_quality_array = None
                    can_combine_tqa = False

                if this.input_flag_array is not None:
                    if self.future_array_shapes:
                        zero_pad = np.zeros(
                            (
                                len(anew_inds),
                                this.input_flag_array.shape[1],
                                this.Ntimes,
                                this.Njones,
                            )
                        )
                    else:
                        zero_pad = np.zeros(
                            (len(anew_inds), 1, this.Nfreqs, this.Ntimes, this.Njones)
                        )
                    this.input_flag_array = np.concatenate(
                        [this.input_flag_array, 1 - zero_pad], axis=0
                    ).astype(np.bool_)

        f_order = None
        if len(fnew_inds) > 0:
            if this.wide_band:
                this.spw_array = np.concatenate(
                    [this.spw_array, other.spw_array[fnew_inds]]
                )
                this.freq_range = np.concatenate(
                    [this.freq_range, other.freq_range[fnew_inds]], axis=0
                )
            elif self.future_array_shapes:
                this.freq_array = np.concatenate(
                    [this.freq_array, other.freq_array[fnew_inds]]
                )
            else:
                this.freq_array = np.concatenate(
                    [this.freq_array, other.freq_array[:, fnew_inds]], axis=1
                )

            if this.flex_spw and not this.wide_band:
                this.flex_spw_id_array = np.concatenate(
                    [this.flex_spw_id_array, other.flex_spw_id_array[fnew_inds]]
                )
                concat_spw_array = np.concatenate([this.spw_array, other.spw_array])
                # We want to preserve per-spw information based on first appearance
                # in the concatenated array.
                unique_index = np.sort(
                    np.unique(this.flex_spw_id_array, return_index=True)[1]
                )
                this.spw_array = this.flex_spw_id_array[unique_index]
                spw_index = np.asarray(
                    [
                        np.nonzero(concat_spw_array == spw)[0][0]
                        for spw in this.spw_array
                    ]
                )
                if this.freq_range is not None and other.freq_range is not None:
                    # this can be removed in v3.0
                    if this.future_array_shapes:
                        this.freq_range = np.concatenate(
                            [this.freq_range, other.freq_range], axis=0
                        )
                        this.freq_range = this.freq_range[spw_index, :]
                    else:
                        temp = np.concatenate([this.freq_range, other.freq_range])
                        this.freq_range = np.asarray([np.min(temp), np.max(temp)])
                elif this.freq_range is not None or other.freq_range is not None:
                    warnings.warn(
                        "One object has the freq_range set and one does not. Combined "
                        "object will not have it set."
                    )
                    this.freq_range = None

                # If we have a multi-spw data set, need to sort out the order of
                # the individual windows first.
                f_order = np.concatenate(
                    [
                        np.where(this.flex_spw_id_array == idx)[0]
                        for idx in sorted(this.spw_array)
                    ]
                )

                # With spectral windows sorted, check and see if channels within
                # windows need sorting. If they are ordered in ascending or descending
                # fashion, leave them be. If not, sort in ascending order
                for idx in this.spw_array:
                    select_mask = this.flex_spw_id_array[f_order] == idx
                    check_freqs = (
                        this.freq_array[f_order[select_mask]]
                        if this.future_array_shapes
                        else this.freq_array[0, f_order[select_mask]]
                    )
                    if (not np.all(check_freqs[1:] > check_freqs[:-1])) and (
                        not np.all(check_freqs[1:] < check_freqs[:-1])
                    ):
                        subsort_order = f_order[select_mask]
                        f_order[select_mask] = subsort_order[np.argsort(check_freqs)]

                spw_sort_index = np.asarray(
                    [
                        np.nonzero(this.spw_array == spw)[0][0]
                        for spw in sorted(this.spw_array)
                    ]
                )
                this.spw_array = np.array(sorted(this.spw_array))
                if this.freq_range is not None and this.future_array_shapes:
                    # this can be removed in v3.0
                    this.freq_range = this.freq_range[spw_sort_index, :]
            else:
                if this_has_spw_id or other_has_spw_id:
                    this.flex_spw_id_array = np.full(
                        this.freq_array.size, this.spw_array[0], dtype=int
                    )
                if this.wide_band:
                    f_order = np.argsort(this.spw_array)
                else:
                    if this.future_array_shapes:
                        f_order = np.argsort(this.freq_array)
                    else:
                        f_order = np.argsort(this.freq_array[0, :])

                    if this.freq_range is not None and other.freq_range is not None:
                        temp = np.concatenate((this.freq_range, other.freq_range))
                        this.freq_range = [np.min(temp), np.max(temp)]
                        if this.future_array_shapes:
                            this.freq_range = np.array(this.freq_range)[np.newaxis, :]
                    elif this.freq_range is not None or other.freq_range is not None:
                        warnings.warn(
                            "One object has the freq_range set and one does not. "
                            "Combined object will not have it set."
                        )
                        this.freq_range = None

            if not this.wide_band and (this.flex_spw or this.future_array_shapes):
                this.channel_width = np.concatenate(
                    [this.channel_width, other.channel_width[fnew_inds]]
                )

            if not self.metadata_only:
                data_array_shape = getattr(this, this._required_data_params[0]).shape
                if self.future_array_shapes:
                    zero_pad = np.zeros(
                        (data_array_shape[0], len(fnew_inds), this.Ntimes, this.Njones)
                    )
                    if this.cal_type == "gain":
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad], axis=1
                        )
                    else:
                        this.delay_array = np.concatenate(
                            [this.delay_array, zero_pad], axis=1
                        )

                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad], axis=1
                    ).astype(np.bool_)
                    if this.quality_array is not None:
                        this.quality_array = np.concatenate(
                            [this.quality_array, zero_pad], axis=1
                        )

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros((len(fnew_inds), this.Ntimes, this.Njones))
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=0
                        )
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros((len(fnew_inds), this.Ntimes, this.Njones))
                        this.total_quality_array = np.zeros(
                            (Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=0
                        )
                else:
                    zero_pad = np.zeros(
                        (
                            data_array_shape[0],
                            1,
                            len(fnew_inds),
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    this.gain_array = np.concatenate(
                        [this.gain_array, zero_pad], axis=2
                    )
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad], axis=2
                    ).astype(np.bool_)
                    if this.quality_array is not None:
                        this.quality_array = np.concatenate(
                            [this.quality_array, zero_pad], axis=2
                        )

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (1, len(fnew_inds), this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (1, len(fnew_inds), this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.zeros(
                            (1, Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )

                if this.input_flag_array is not None:
                    if self.future_array_shapes:
                        zero_pad = np.zeros(
                            (
                                this.flag_array.shape[0],
                                len(fnew_inds),
                                this.Ntimes,
                                this.Njones,
                            )
                        )
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=1
                        ).astype(np.bool_)
                    else:
                        zero_pad = np.zeros(
                            (
                                this.flag_array.shape[0],
                                1,
                                len(fnew_inds),
                                this.Ntimes,
                                this.Njones,
                            )
                        )
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=2
                        ).astype(np.bool_)

        t_order = None
        if len(tnew_inds) > 0:
            this.time_array = np.concatenate(
                [this.time_array, other.time_array[tnew_inds]]
            )
            this.lst_array = np.concatenate(
                [this.lst_array, other.lst_array[tnew_inds]]
            )
            t_order = np.argsort(this.time_array)
            if self.future_array_shapes:
                this.integration_time = np.concatenate(
                    [this.integration_time, other.integration_time[tnew_inds]]
                )

            if not self.metadata_only:
                data_array_shape = getattr(this, this._required_data_params[0]).shape
                if self.future_array_shapes:
                    zero_pad_data = np.zeros(
                        (
                            data_array_shape[0],
                            data_array_shape[1],
                            len(tnew_inds),
                            this.Njones,
                        )
                    )
                    zero_pad_flags = np.zeros(
                        (
                            this.flag_array.shape[0],
                            this.flag_array.shape[1],
                            len(tnew_inds),
                            this.Njones,
                        )
                    )
                    if this.cal_type == "delay":
                        this.delay_array = np.concatenate(
                            [this.delay_array, zero_pad_data], axis=2
                        )
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=2
                        )
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=2
                    ).astype(np.bool_)
                    if this.quality_array is not None:
                        this.quality_array = np.concatenate(
                            [this.quality_array, zero_pad_data], axis=2
                        )

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (this.quality_array.shape[1], len(tnew_inds), this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (this.quality_array.shape[1], len(tnew_inds), this.Njones)
                        )
                        this.total_quality_array = np.zeros(
                            (Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )

                    if this.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.input_flag_array.shape[0],
                                this.input_flag_array.shape[1],
                                len(tnew_inds),
                                this.Njones,
                            )
                        )
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=2
                        ).astype(np.bool_)

                else:
                    zero_pad_data = np.zeros(
                        (
                            data_array_shape[0],
                            1,
                            data_array_shape[2],
                            len(tnew_inds),
                            this.Njones,
                        )
                    )
                    zero_pad_flags = np.zeros(
                        (
                            this.flag_array.shape[0],
                            1,
                            this.flag_array.shape[2],
                            len(tnew_inds),
                            this.Njones,
                        )
                    )
                    if this.cal_type == "delay":
                        this.delay_array = np.concatenate(
                            [this.delay_array, zero_pad_data], axis=3
                        )
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=3
                        )
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=3
                    ).astype(np.bool_)
                    if this.quality_array is not None:
                        this.quality_array = np.concatenate(
                            [this.quality_array, zero_pad_data], axis=3
                        )

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (1, data_array_shape[2], len(tnew_inds), this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (1, data_array_shape[2], len(tnew_inds), this.Njones)
                        )
                        this.total_quality_array = np.zeros(
                            (1, Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )

                    if this.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.input_flag_array.shape[0],
                                1,
                                this.input_flag_array.shape[2],
                                len(tnew_inds),
                                this.Njones,
                            )
                        )
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=3
                        ).astype(np.bool_)

        j_order = None
        if len(jnew_inds) > 0:
            this.jones_array = np.concatenate(
                [this.jones_array, other.jones_array[jnew_inds]]
            )
            j_order = np.argsort(np.abs(this.jones_array))
            if not self.metadata_only:
                data_array_shape = getattr(this, this._required_data_params[0]).shape
                if self.future_array_shapes:
                    zero_pad_data = np.zeros(
                        (
                            data_array_shape[0],
                            data_array_shape[1],
                            data_array_shape[2],
                            len(jnew_inds),
                        )
                    )
                    zero_pad_flags = np.zeros(
                        (
                            this.flag_array.shape[0],
                            this.flag_array.shape[1],
                            this.flag_array.shape[2],
                            len(jnew_inds),
                        )
                    )
                    if this.cal_type == "delay":
                        this.delay_array = np.concatenate(
                            [this.delay_array, zero_pad_data], axis=3
                        )
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=3
                        )
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=3
                    ).astype(np.bool_)
                    if this.quality_array is not None:
                        this.quality_array = np.concatenate(
                            [this.quality_array, zero_pad_data], axis=3
                        )

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (data_array_shape[1], data_array_shape[2], len(jnew_inds))
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (data_array_shape[1], data_array_shape[2], len(jnew_inds))
                        )
                        this.total_quality_array = np.zeros(
                            (Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )

                    if this.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.input_flag_array.shape[0],
                                this.input_flag_array.shape[1],
                                this.input_flag_array.shape[2],
                                len(jnew_inds),
                            )
                        )
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=3
                        ).astype(np.bool_)
                else:
                    zero_pad_data = np.zeros(
                        (
                            data_array_shape[0],
                            1,
                            data_array_shape[2],
                            data_array_shape[3],
                            len(jnew_inds),
                        )
                    )
                    zero_pad_flags = np.zeros(
                        (
                            this.flag_array.shape[0],
                            1,
                            this.flag_array.shape[2],
                            this.flag_array.shape[3],
                            len(jnew_inds),
                        )
                    )
                    if this.cal_type == "delay":
                        this.delay_array = np.concatenate(
                            [this.delay_array, zero_pad_data], axis=4
                        )
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=4
                        )
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=4
                    ).astype(np.bool_)
                    if this.quality_array is not None:
                        this.quality_array = np.concatenate(
                            [this.quality_array, zero_pad_data], axis=4
                        )

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                1,
                                data_array_shape[2],
                                data_array_shape[3],
                                len(jnew_inds),
                            )
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=3
                        )
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                1,
                                data_array_shape[2],
                                data_array_shape[3],
                                len(jnew_inds),
                            )
                        )
                        this.total_quality_array = np.zeros(
                            (1, Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=3
                        )

                    if this.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.input_flag_array.shape[0],
                                1,
                                this.input_flag_array.shape[2],
                                this.input_flag_array.shape[3],
                                len(jnew_inds),
                            )
                        )
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=4
                        ).astype(np.bool_)

        # Now populate the data
        if not self.metadata_only:
            jones_t2o = np.nonzero(np.in1d(this.jones_array, other.jones_array))[0]
            times_t2o = np.nonzero(np.in1d(this.time_array, other.time_array))[0]
            if self.wide_band:
                freqs_t2o = np.nonzero(np.in1d(this.spw_array, other.spw_array))[0]
            elif self.future_array_shapes:
                freqs_t2o = np.nonzero(np.in1d(this.freq_array, other.freq_array))[0]
            else:
                if self.cal_type == "gain":
                    freqs_t2o = np.nonzero(
                        np.in1d(this.freq_array[0, :], other.freq_array[0, :])
                    )[0]
                else:
                    freqs_t2o = [0]
            ants_t2o = np.nonzero(np.in1d(this.ant_array, other.ant_array))[0]
            if self.future_array_shapes:
                if this.cal_type == "delay":
                    this.delay_array[
                        np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                    ] = other.delay_array
                else:
                    this.gain_array[
                        np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                    ] = other.gain_array
                if other.quality_array is not None:
                    this.quality_array[
                        np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                    ] = other.quality_array
                this.flag_array[np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)] = (
                    other.flag_array
                )
                if this.total_quality_array is not None:
                    if other.total_quality_array is not None:
                        this.total_quality_array[
                            np.ix_(freqs_t2o, times_t2o, jones_t2o)
                        ] = other.total_quality_array
                if this.input_flag_array is not None:
                    if other.input_flag_array is not None:
                        this.input_flag_array[
                            np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                        ] = other.input_flag_array
            else:
                if this.cal_type == "delay":
                    this.delay_array[
                        np.ix_(ants_t2o, [0], [0], times_t2o, jones_t2o)
                    ] = other.delay_array
                    if other.quality_array is not None:
                        this.quality_array[
                            np.ix_(ants_t2o, [0], [0], times_t2o, jones_t2o)
                        ] = other.quality_array
                else:
                    this.gain_array[
                        np.ix_(ants_t2o, [0], freqs_t2o, times_t2o, jones_t2o)
                    ] = other.gain_array
                    if other.quality_array is not None:
                        this.quality_array[
                            np.ix_(ants_t2o, [0], freqs_t2o, times_t2o, jones_t2o)
                        ] = other.quality_array
                this.flag_array[
                    np.ix_(ants_t2o, [0], freqs_t2o, times_t2o, jones_t2o)
                ] = other.flag_array
                if this.total_quality_array is not None:
                    if other.total_quality_array is not None:
                        if this.cal_type == "delay":
                            this.total_quality_array[
                                np.ix_([0], [0], times_t2o, jones_t2o)
                            ] = other.total_quality_array
                        else:
                            this.total_quality_array[
                                np.ix_([0], freqs_t2o, times_t2o, jones_t2o)
                            ] = other.total_quality_array
                if this.input_flag_array is not None:
                    if other.input_flag_array is not None:
                        this.input_flag_array[
                            np.ix_(ants_t2o, [0], freqs_t2o, times_t2o, jones_t2o)
                        ] = other.input_flag_array

            # Fix ordering
            ant_axis_num = 0
            if this.future_array_shapes:
                faxis_num = 1
                taxis_num = 2
                jaxis_num = 3
            else:
                faxis_num = 2
                taxis_num = 3
                jaxis_num = 4

            axis_dict = {
                ant_axis_num: {"inds": anew_inds, "order": ant_order},
                faxis_num: {"inds": fnew_inds, "order": f_order},
                taxis_num: {"inds": tnew_inds, "order": t_order},
                jaxis_num: {"inds": jnew_inds, "order": j_order},
            }
            for axis, subdict in axis_dict.items():
                for name, param in zip(this._data_params, this.data_like_parameters):
                    if param is None:
                        continue
                    axis_delta = 0
                    if name == "total_quality_array":
                        axis_delta = 1

                    if len(subdict["inds"]) > 0:
                        unique_order_diffs = np.unique(np.diff(subdict["order"]))
                        if np.array_equal(unique_order_diffs, np.array([1])):
                            # everything is already in order
                            continue
                        setattr(
                            this,
                            name,
                            np.take(param, subdict["order"], axis=axis - axis_delta),
                        )

        if len(anew_inds) > 0:
            this.ant_array = this.ant_array[ant_order]
        if len(fnew_inds) > 0:
            if this.wide_band:
                this.spw_array = this.spw_array[f_order]
                this.freq_range = this.freq_range[f_order]
            else:
                if this.future_array_shapes:
                    this.freq_array = this.freq_array[f_order]
                else:
                    this.freq_array = this.freq_array[:, f_order]
                if this.flex_spw or this.future_array_shapes:
                    this.channel_width = this.channel_width[f_order]
                if this.flex_spw_id_array is not None:
                    this.flex_spw_id_array = this.flex_spw_id_array[f_order]
        if len(tnew_inds) > 0:
            this.time_array = this.time_array[t_order]
            this.lst_array = this.lst_array[t_order]
            if self.future_array_shapes:
                this.integration_time = this.integration_time[t_order]
        if len(jnew_inds) > 0:
            this.jones_array = this.jones_array[j_order]

        # Update N parameters (e.g. Njones)
        this.Njones = this.jones_array.shape[0]
        this.Ntimes = this.time_array.shape[0]
        if this.cal_type == "gain" and not this.wide_band:
            this.Nfreqs = this.freq_array.size
        this.Nspws = this.spw_array.size
        this.Nants_data = len(
            np.unique(this.ant_array.tolist() + other.ant_array.tolist())
        )

        # Check specific requirements
        if this.cal_type == "gain" and this.Nfreqs > 1:
            spacing_error, chanwidth_error = this._check_freq_spacing(
                raise_errors=False
            )

            if spacing_error:
                warnings.warn(
                    "Combined frequencies are not evenly spaced or have differing "
                    "values of channel widths. This will make it impossible to write "
                    "this data out to some file types."
                )
            elif chanwidth_error:
                warnings.warn(
                    "Combined frequencies are separated by more than their "
                    "channel width. This will make it impossible to write this data "
                    "out to some file types."
                )

        if this.Njones > 2:
            if not uvutils._test_array_constant_spacing(this._jones_array):
                warnings.warn(
                    "Combined Jones elements are not evenly spaced. This will "
                    "make it impossible to write this data out to some file types."
                )

        if n_axes > 0:
            history_update_string += " axis using pyuvdata."

            histories_match = uvutils._check_histories(this.history, other.history)

            this.history += history_update_string
            if not histories_match:
                if verbose_history:
                    this.history += " Next object history follows. " + other.history
                else:
                    extra_history = uvutils._combine_history_addition(
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

    def __iadd__(
        self, other, run_check=True, check_extra=True, run_check_acceptability=True
    ):
        """
        Combine two UVCal objects in place.

        Along antenna, frequency, time, and/or Jones axis.

        Parameters
        ----------
        other : :class: UVCal
            Another UVCal object which will be added to self.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        """
        self.__add__(
            other,
            inplace=True,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )
        return self

    def fast_concat(
        self,
        other,
        axis,
        inplace=False,
        verbose_history=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Concatenate two UVCal objects along specified axis with almost no checking.

        Warning! This method assumes all the metadata along other axes is sorted
        the same way. The __add__ method is much safer, it checks all the metadata,
        but it is slower. Some quick checks are run, but this method doesn't
        make any guarantees that the resulting object is correct.

        Parameters
        ----------
        other : UVCal object or list of UVCal objects
            UVCal object or list of UVCal objects which will be added to self.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. Allowed values are: 'antenna', 'time', 'freq', 'spw',
            'jones' ('freq' is not allowed for delay or wideband objects and 'spw' is
            only allowed for wideband objects).
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
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
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.

        """
        if inplace:
            this = self
        else:
            this = self.copy()
        if not isinstance(other, (list, tuple, np.ndarray)):
            # if this is a UVCal object already, stick it in a list
            other = [other]
        # Check that both objects are UVCal and valid
        this.check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )
        for obj in other:
            if not issubclass(obj.__class__, this.__class__):
                if not issubclass(this.__class__, obj.__class__):
                    raise ValueError(
                        "Only UVCal (or subclass) objects can be "
                        "added to a UVCal (or subclass) object"
                    )
            obj.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        # check that all objects have the same array shapes
        for obj in other:
            if this.future_array_shapes != obj.future_array_shapes:
                raise ValueError(
                    "All objects must have the same `future_array_shapes` parameter. "
                    "Use the `use_future_array_shapes` or `use_current_array_shapes` "
                    "methods to convert them."
                )

        # Check that all objects are consistent w/ use of flex_spw
        for obj in other:
            if this.flex_spw != obj.flex_spw:
                raise ValueError(
                    "To combine these data, flex_spw must be set to the same "
                    "value (True or False) for all objects."
                )

        # Check that all objects are consistent w/ use of wide_band or not
        for obj in other:
            if this.wide_band != obj.wide_band:
                raise ValueError(
                    "To combine these data, wide_band must be set to the same "
                    "value (True or False) for all objects."
                )

        this_has_spw_id = this.flex_spw_id_array is not None
        other_has_spw_id = np.array(
            [obj.flex_spw_id_array is not None for obj in other]
        )
        if not np.all(other_has_spw_id == this_has_spw_id):
            warnings.warn(
                "Some objects have the flex_spw_id_array set and some do not. Combined "
                "object will have it set."
            )

        allowed_axes = ["antenna", "time", "jones"]
        if this.wide_band is True:
            allowed_axes.append("spw")
        elif self.cal_type == "gain":
            allowed_axes.append("freq")
        if axis not in allowed_axes:
            raise ValueError("Axis must be one of: " + ", ".join(allowed_axes))

        # Check objects are compatible
        compatibility_params = [
            "_cal_type",
            "_telescope_name",
            "_gain_convention",
            "_x_orientation",
            "_cal_style",
            "_ref_antenna_name",
        ]
        if not this.future_array_shapes:
            compatibility_params.append("_integration_time")
            if not this.flex_spw:
                compatibility_params.append("_channel_width")

        warning_params = [
            "_observer",
            "_git_hash_cal",
            "_sky_field",
            "_sky_catalog",
            "_Nsources",
            "_baseline_range",
            "_diffuse_model",
        ]

        history_update_string = " Combined data along "

        if axis == "freq" or axis == "spw":
            if axis == "freq":
                history_update_string += "frequency"
            else:
                history_update_string += "spectral window"
            compatibility_params += [
                "_jones_array",
                "_ant_array",
                "_time_array",
                "_integration_time",
                "_lst_array",
                "_time_range",
            ]
        elif axis == "jones":
            history_update_string += "jones"
            compatibility_params += [
                "_freq_array",
                "_channel_width",
                "_ant_array",
                "_time_array",
                "_integration_time",
                "_lst_array",
                "_time_range",
            ]
        elif axis == "antenna":
            history_update_string += "antenna"
            compatibility_params += [
                "_freq_array",
                "_channel_width",
                "_jones_array",
                "_time_array",
                "_integration_time",
                "_lst_array",
                "_time_range",
            ]
        elif axis == "time":
            history_update_string += "time"
            compatibility_params += [
                "_freq_array",
                "_channel_width",
                "_jones_array",
                "_ant_array",
            ]
            if not this.future_array_shapes:
                compatibility_params += ["_integration_time"]

        history_update_string += " axis using pyuvdata."
        histories_match = []
        for obj in other:
            histories_match.append(uvutils._check_histories(this.history, obj.history))

        this.history += history_update_string
        for obj_num, obj in enumerate(other):
            if not histories_match[obj_num]:
                if verbose_history:
                    this.history += " Next object history follows. " + obj.history
                else:
                    extra_history = uvutils._combine_history_addition(
                        this.history, obj.history
                    )
                    if extra_history is not None:
                        this.history += (
                            " Unique part of next object history follows. "
                            + extra_history
                        )
        # Actually check compatibility parameters
        for obj in other:
            for a in compatibility_params:
                params_match = getattr(this, a) == getattr(obj, a)
                if not params_match:
                    msg = (
                        "UVParameter "
                        + a[1:]
                        + " does not match. Cannot combine objects."
                    )
                    raise ValueError(msg)

            for a in warning_params:
                params_match = getattr(this, a) == getattr(obj, a)
                if not params_match:
                    msg = "UVParameter " + a[1:] + " does not match. Combining anyway."
                    warnings.warn(msg)

        total_quality_exists = [this.total_quality_array is not None] + [
            obj.total_quality_array is not None for obj in other
        ]
        this_tqa_exp_shape = this._total_quality_array.expected_shape(this)

        input_flag_exists = [this.input_flag_array is not None] + [
            obj.input_flag_array is not None for obj in other
        ]
        this_ifa_exp_shape = this._input_flag_array.expected_shape(this)

        quality_exists = [this.quality_array is not None] + [
            obj.quality_array is not None for obj in other
        ]
        this_qa_exp_shape = this._quality_array.expected_shape(this)

        freq_range_exists = [this.freq_range is not None] + [
            obj.freq_range is not None for obj in other
        ]

        if axis == "antenna":
            this.Nants_data = sum([this.Nants_data] + [obj.Nants_data for obj in other])
            this.ant_array = np.concatenate(
                [this.ant_array] + [obj.ant_array for obj in other]
            )
            axis_num = 0
        elif axis == "freq":
            this.Nfreqs = sum([this.Nfreqs] + [obj.Nfreqs for obj in other])
            if this.future_array_shapes:
                this.freq_array = np.concatenate(
                    [this.freq_array] + [obj.freq_array for obj in other]
                )
            else:
                this.freq_array = np.concatenate(
                    [this.freq_array] + [obj.freq_array for obj in other], axis=1
                )
            if this.flex_spw or this.future_array_shapes:
                this.channel_width = np.concatenate(
                    [this.channel_width] + [obj.channel_width for obj in other]
                )
            if this.flex_spw:
                this.flex_spw_id_array = np.concatenate(
                    [this.flex_spw_id_array] + [obj.flex_spw_id_array for obj in other]
                )
                concat_spw_array = np.concatenate(
                    [this.spw_array] + [obj.spw_array for obj in other]
                )
                # We want to preserve per-spw information based on first appearance
                # in the concatenated array.
                unique_index = np.sort(
                    np.unique(this.flex_spw_id_array, return_index=True)[1]
                )
                this.spw_array = this.flex_spw_id_array[unique_index]
                this.Nspws = this.spw_array.size
                spw_index = np.asarray(
                    [
                        np.nonzero(concat_spw_array == spw)[0][0]
                        for spw in this.spw_array
                    ]
                )
                if np.all(freq_range_exists):
                    if this.future_array_shapes:
                        this.freq_range = np.concatenate(
                            [this.freq_range] + [obj.freq_range for obj in other],
                            axis=0,
                        )
                        this.freq_range = this.freq_range[spw_index, :]
                    else:
                        temp = np.concatenate(
                            ([this.freq_range] + [obj.freq_range for obj in other])
                        )
                        this.freq_range = np.array([np.min(temp), np.max(temp)])
                elif np.any(freq_range_exists):
                    warnings.warn(
                        "Some objects have the freq_range set and some do not. "
                        "Combined object will not have it set."
                    )
                    this.freq_range = None
            else:
                if this_has_spw_id or other_has_spw_id:
                    this.flex_spw_id_array = np.full(
                        this.Nfreqs, this.spw_array[0], dtype=int
                    )
                if np.all(freq_range_exists):
                    temp = np.concatenate(
                        ([this.freq_range] + [obj.freq_range for obj in other])
                    )
                    this.freq_range = [np.min(temp), np.max(temp)]
                    if this.future_array_shapes:
                        this.freq_range = np.array(this.freq_range)
                        this.freq_range = this.freq_range[np.newaxis, :]
                elif np.any(freq_range_exists):
                    warnings.warn(
                        "Some objects have the freq_range set and and some do not. "
                        "Combined object will not have it set."
                    )
                    this.freq_range = None

            spacing_error, chanwidth_error = this._check_freq_spacing(
                raise_errors=False
            )
            if spacing_error:
                warnings.warn(
                    "Combined frequencies are not evenly spaced or have differing "
                    "values of channel widths. This will make it impossible to write "
                    "this data out to some file types."
                )
            elif chanwidth_error:
                warnings.warn(
                    "Combined frequencies are separated by more than their "
                    "channel width. This will make it impossible to write this data "
                    "out to some file types."
                )

            if this.future_array_shapes:
                axis_num = 1
            else:
                axis_num = 2
        elif axis == "spw":
            # only get here for a wide-band cal (so only if future_array_shapes is True)
            this.Nspws = sum([this.Nspws] + [obj.Nspws for obj in other])
            this.spw_array = np.concatenate(
                [this.spw_array] + [obj.spw_array for obj in other]
            )
            this.freq_range = np.concatenate(
                [this.freq_range] + [obj.freq_range for obj in other], axis=0
            )
            axis_num = 1
        elif axis == "time":
            this.Ntimes = sum([this.Ntimes] + [obj.Ntimes for obj in other])
            this.time_array = np.concatenate(
                [this.time_array] + [obj.time_array for obj in other]
            )
            if this.future_array_shapes:
                this.integration_time = np.concatenate(
                    [this.integration_time] + [obj.integration_time for obj in other]
                )
            this.lst_array = np.concatenate(
                [this.lst_array] + [obj.lst_array for obj in other]
            )
            if this.future_array_shapes:
                axis_num = 2
            else:
                axis_num = 3
        elif axis == "jones":
            this.Njones = sum([this.Njones] + [obj.Njones for obj in other])
            this.jones_array = np.concatenate(
                [this.jones_array] + [obj.jones_array for obj in other]
            )
            if this.future_array_shapes:
                axis_num = 3
            else:
                axis_num = 4

        if not self.metadata_only:
            this.flag_array = np.concatenate(
                [this.flag_array] + [obj.flag_array for obj in other], axis=axis_num
            )

            if this.cal_type == "gain":
                this.gain_array = np.concatenate(
                    [this.gain_array] + [obj.gain_array for obj in other], axis=axis_num
                )
            else:
                this.delay_array = np.concatenate(
                    [this.delay_array] + [obj.delay_array for obj in other],
                    axis=axis_num,
                )

            if np.any(input_flag_exists):
                if np.all(input_flag_exists):
                    this.input_flag_array = np.concatenate(
                        [this.input_flag_array]
                        + [obj.input_flag_array for obj in other],
                        axis=axis_num,
                    )
                else:
                    ifa_list = []
                    if this.input_flag_array is None:
                        ifa_list.append(np.full(this_ifa_exp_shape, True, dtype=bool))
                    else:
                        ifa_list.append(this.input_flag_array)
                    for obj in other:
                        if obj.input_flag_array is None:
                            ifa_list.append(
                                np.full(
                                    (obj._input_flag_array.expected_shape(obj)),
                                    True,
                                    dtype=bool,
                                )
                            )
                        else:
                            ifa_list.append(obj.input_flag_array)
                    this.input_flag_array = np.concatenate(ifa_list, axis=axis_num)

            if np.any(quality_exists):
                if np.all(quality_exists):
                    this.quality_array = np.concatenate(
                        [this.quality_array] + [obj.quality_array for obj in other],
                        axis=axis_num,
                    )
                else:
                    qa_list = []
                    if this.quality_array is None:
                        qa_list.append(np.zeros(this_qa_exp_shape, dtype=float))
                    else:
                        qa_list.append(this.quality_array)
                    for obj in other:
                        if obj.quality_array is None:
                            qa_list.append(
                                np.zeros(
                                    (obj._quality_array.expected_shape(obj)),
                                    dtype=float,
                                )
                            )
                        else:
                            qa_list.append(obj.quality_array)
                    this.quality_array = np.concatenate(qa_list, axis=axis_num)

            if np.any(total_quality_exists):
                if axis == "antenna":
                    warnings.warn(
                        "Total quality array detected in at least one file; the "
                        "array in the new object will be set to 'None' because "
                        "whole-array values cannot be combined when adding antennas"
                    )
                    this.total_quality_array = None
                else:
                    if np.all(total_quality_exists):
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array]
                            + [obj.total_quality_array for obj in other],
                            axis=axis_num - 1,
                        )
                    else:
                        tqa_list = []
                        if this.total_quality_array is None:
                            tqa_list.append(np.zeros(this_tqa_exp_shape, dtype=float))
                        else:
                            tqa_list.append(this.total_quality_array)
                        for obj in other:
                            if obj.total_quality_array is None:
                                tqa_list.append(
                                    np.zeros(
                                        (obj._total_quality_array.expected_shape(obj)),
                                        dtype=float,
                                    )
                                )
                            else:
                                tqa_list.append(obj.total_quality_array)
                        this.total_quality_array = np.concatenate(
                            tqa_list, axis=axis_num - 1
                        )

        # update filename attribute
        for obj in other:
            this.filename = uvutils._combine_filenames(this.filename, obj.filename)
        if this.filename is not None:
            this._filename.form = len(this.filename)

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return this

    def select(
        self,
        antenna_nums=None,
        antenna_names=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        jones=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=True,
    ):
        """
        Downselect data to keep on the object along various axes.

        Axes that can be selected along include antennas, frequencies, times and
        antenna polarization (jones).

        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained).
            This cannot be provided if `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to keep in the object (antenna positions and
            names for the removed antennas will be retained).
            This cannot be provided if `antenna_nums` is also provided.
        frequencies : array_like of float, optional
            The frequencies to keep in the object, each value passed here should
            exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        spws : array_like of in, optional
            The spectral window numbers to keep in the object. If this is not a
            wide-band object and `frequencies` or `freq_chans` is not None, frequencies
            that match any of the specifications will be kept (i.e. the selections will
            be OR'ed together).
        times : array_like of float, optional
            The times to keep in the object, each value passed here should
            exist in the time_array.
        jones : array_like of int or str, optional
            The antenna polarizations numbers to keep in the object, each value
            passed here should exist in the jones_array. If passing strings, the
            canonical polarization strings (e.g. "Jxx", "Jrr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "Jnn", "Jee") are also supported.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        inplace : bool
            Option to perform the select directly on self or return a new UVCal
            object with just the selected data (the default is True, meaning the
            select will be done on self).
        """
        if inplace:
            cal_object = self
        else:
            cal_object = self.copy()

        # build up history string as we go
        history_update_string = "  Downselected to specific "
        n_selects = 0

        if antenna_names is not None:
            if antenna_nums is not None:
                raise ValueError(
                    "Only one of antenna_nums and antenna_names can be provided."
                )

            antenna_names = uvutils._get_iterable(antenna_names)
            antenna_nums = []
            for s in antenna_names:
                if s not in cal_object.antenna_names:
                    raise ValueError(
                        f"Antenna name {s} is not present in the antenna_names array"
                    )
                ind = np.where(np.array(cal_object.antenna_names) == s)[0][0]
                antenna_nums.append(cal_object.antenna_numbers[ind])

        if antenna_nums is not None:
            antenna_nums = uvutils._get_iterable(antenna_nums)
            history_update_string += "antennas"
            n_selects += 1

            ant_inds = np.zeros(0, dtype=np.int64)
            for ant in antenna_nums:
                if ant in cal_object.ant_array:
                    ant_inds = np.append(
                        ant_inds, np.where(cal_object.ant_array == ant)[0]
                    )
                else:
                    raise ValueError(
                        f"Antenna number {ant} is not present in the array"
                    )

            ant_inds = sorted(set(ant_inds))
            cal_object.Nants_data = len(ant_inds)
            cal_object.ant_array = cal_object.ant_array[ant_inds]
            if not self.metadata_only:
                cal_object.flag_array = cal_object.flag_array[ant_inds]
                if cal_object.quality_array is not None:
                    cal_object.quality_array = cal_object.quality_array[ant_inds]
                if cal_object.cal_type == "delay":
                    cal_object.delay_array = cal_object.delay_array[ant_inds]
                else:
                    cal_object.gain_array = cal_object.gain_array[ant_inds]

                if cal_object.input_flag_array is not None:
                    cal_object.input_flag_array = cal_object.input_flag_array[ant_inds]

                if cal_object.total_quality_array is not None:
                    warnings.warn(
                        "Cannot preserve total_quality_array when changing "
                        "number of antennas; discarding"
                    )
                    cal_object.total_quality_array = None

        if times is not None:
            times = uvutils._get_iterable(times)
            if n_selects > 0:
                history_update_string += ", times"
            else:
                history_update_string += "times"
            n_selects += 1

            time_inds = np.zeros(0, dtype=np.int64)
            for jd in times:
                if jd in cal_object.time_array:
                    time_inds = np.append(
                        time_inds, np.where(cal_object.time_array == jd)[0]
                    )
                else:
                    raise ValueError(
                        "Time {t} is not present in the time_array".format(t=jd)
                    )

            time_inds = sorted(set(time_inds))
            cal_object.Ntimes = len(time_inds)
            cal_object.time_array = cal_object.time_array[time_inds]
            if cal_object.lst_array is not None:
                cal_object.lst_array = cal_object.lst_array[time_inds]
            if self.future_array_shapes:
                cal_object.integration_time = cal_object.integration_time[time_inds]

            if cal_object.Ntimes > 1:
                if not uvutils._test_array_constant_spacing(cal_object._time_array):
                    warnings.warn(
                        "Selected times are not evenly spaced. This "
                        "is not supported by the calfits format."
                    )

            if not self.metadata_only:
                if self.future_array_shapes:
                    cal_object.flag_array = cal_object.flag_array[:, :, time_inds, :]
                    if cal_object.quality_array is not None:
                        cal_object.quality_array = cal_object.quality_array[
                            :, :, time_inds, :
                        ]
                    if cal_object.cal_type == "delay":
                        cal_object.delay_array = cal_object.delay_array[
                            :, :, time_inds, :
                        ]
                    else:
                        cal_object.gain_array = cal_object.gain_array[
                            :, :, time_inds, :
                        ]

                    if cal_object.input_flag_array is not None:
                        cal_object.input_flag_array = cal_object.input_flag_array[
                            :, :, time_inds, :
                        ]

                    if cal_object.total_quality_array is not None:
                        cal_object.total_quality_array = cal_object.total_quality_array[
                            :, time_inds, :
                        ]
                else:
                    cal_object.flag_array = cal_object.flag_array[:, :, :, time_inds, :]
                    if cal_object.quality_array is not None:
                        cal_object.quality_array = cal_object.quality_array[
                            :, :, :, time_inds, :
                        ]
                    if cal_object.cal_type == "delay":
                        cal_object.delay_array = cal_object.delay_array[
                            :, :, :, time_inds, :
                        ]
                    else:
                        cal_object.gain_array = cal_object.gain_array[
                            :, :, :, time_inds, :
                        ]

                    if cal_object.input_flag_array is not None:
                        cal_object.input_flag_array = cal_object.input_flag_array[
                            :, :, :, time_inds, :
                        ]

                    if cal_object.total_quality_array is not None:
                        cal_object.total_quality_array = cal_object.total_quality_array[
                            :, :, time_inds, :
                        ]

        if spws is not None:
            if cal_object.Nspws == 1:
                warnings.warn(
                    "Cannot select on spws if Nspws=1. Ignoring the spw parameter."
                )
            else:
                if not cal_object.wide_band:
                    assert cal_object.flex_spw is True, (
                        "The `flex_spw` parameter must be True if there are multiple "
                        "spectral windows and the `wide_band` parameter is not True."
                    )
                    # Translate the spws into frequencies
                    if frequencies is None:
                        if self.future_array_shapes:
                            frequencies = self.freq_array[
                                np.isin(cal_object.flex_spw_id_array, spws)
                            ]
                        else:
                            frequencies = self.freq_array[
                                0, np.isin(cal_object.flex_spw_id_array, spws)
                            ]
                else:
                    assert self.future_array_shapes, (
                        "The `future_array_shapes` parameter must be True if the "
                        "`wide_band` parameter is True"
                    )
                    if n_selects > 0:
                        history_update_string += ", spectral windows"
                    else:
                        history_update_string += "spectral windows"
                    n_selects += 1

                    # Check and see that all requested spws are available
                    spw_check = np.isin(spws, cal_object.spw_array)
                    if not np.all(spw_check):
                        raise ValueError(
                            f"SPW number {spws[np.where(~spw_check)[0][0]]} is not "
                            "present in the spw_array"
                        )

                    spw_inds = np.where(np.isin(cal_object.spw_array, spws))[0]

                    spw_inds = sorted(set(spw_inds))
                    cal_object.Nspws = len(spw_inds)
                    cal_object.freq_range = cal_object.freq_range[spw_inds, :]
                    cal_object.spw_array = cal_object.spw_array[spw_inds]

                    if not cal_object.metadata_only:
                        if cal_object.cal_type == "delay":
                            cal_object.delay_array = cal_object.delay_array[
                                :, spw_inds, :, :
                            ]
                        else:
                            cal_object.gain_array = cal_object.gain_array[
                                :, spw_inds, :, :
                            ]

                        cal_object.flag_array = cal_object.flag_array[:, spw_inds, :, :]
                        if cal_object.input_flag_array is not None:
                            cal_object.input_flag_array = cal_object.input_flag_array[
                                :, spw_inds, :, :
                            ]
                        if cal_object.quality_array is not None:
                            cal_object.quality_array = cal_object.quality_array[
                                :, spw_inds, :, :
                            ]
                        if cal_object.total_quality_array is not None:
                            tqa = cal_object.total_quality_array[spw_inds, :, :]
                            cal_object.total_quality_array = tqa

        if self.freq_array is None and (
            freq_chans is not None or frequencies is not None
        ):
            raise ValueError(
                "Cannot select on frequencies because this is a wide_band object with "
                "no freq_array."
            )
        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if frequencies is None:
                if self.future_array_shapes:
                    frequencies = cal_object.freq_array[freq_chans]
                else:
                    frequencies = cal_object.freq_array[0, freq_chans]
            else:
                frequencies = uvutils._get_iterable(frequencies)
                if self.future_array_shapes:
                    frequencies = np.sort(
                        list(set(frequencies) | set(cal_object.freq_array[freq_chans]))
                    )
                else:
                    frequencies = np.sort(
                        list(
                            set(frequencies) | set(cal_object.freq_array[0, freq_chans])
                        )
                    )

        if frequencies is not None:
            frequencies = uvutils._get_iterable(frequencies)
            if n_selects > 0:
                history_update_string += ", frequencies"
            else:
                history_update_string += "frequencies"
            n_selects += 1

            if cal_object.future_array_shapes:
                freq_arr_use = self.freq_array
            else:
                freq_arr_use = self.freq_array[0, :]

            # Check and see that all requested freqs are available
            freq_check = np.isin(frequencies, freq_arr_use)
            if not np.all(freq_check):
                raise ValueError(
                    f"Frequency {frequencies[np.where(~freq_check)[0][0]]} is not "
                    "present in the freq_array"
                )
            freq_inds = np.where(np.isin(freq_arr_use, frequencies))[0]

            freq_inds = sorted(set(freq_inds))
            cal_object.Nfreqs = len(freq_inds)
            if cal_object.future_array_shapes:
                cal_object.freq_array = cal_object.freq_array[freq_inds]
            else:
                cal_object.freq_array = cal_object.freq_array[:, freq_inds]

            if cal_object.future_array_shapes or cal_object.flex_spw:
                cal_object.channel_width = cal_object.channel_width[freq_inds]

            if cal_object.flex_spw_id_array is not None:
                cal_object.flex_spw_id_array = cal_object.flex_spw_id_array[freq_inds]

            if cal_object.flex_spw:
                spw_mask = np.isin(cal_object.spw_array, cal_object.flex_spw_id_array)
                cal_object.spw_array = cal_object.spw_array[spw_mask]
                cal_object.Nspws = cal_object.spw_array.size
                if cal_object.freq_range is not None and cal_object.future_array_shapes:
                    cal_object.freq_range = np.zeros(
                        (cal_object.Nspws, 2), dtype=cal_object.freq_array.dtype
                    )
                    for index, spw in enumerate(cal_object.spw_array):
                        spw_inds = np.nonzero(cal_object.flex_spw_id_array == spw)[0]
                        cal_object.freq_range[index, 0] = np.min(
                            cal_object.freq_array[spw_inds]
                        )
                        cal_object.freq_range[index, 1] = np.max(
                            cal_object.freq_array[spw_inds]
                        )
            else:
                if cal_object.freq_range is not None:
                    cal_object.freq_range = [
                        np.min(cal_object.freq_array),
                        np.max(cal_object.freq_array),
                    ]
                    if cal_object.future_array_shapes:
                        cal_object.freq_range = np.asarray(cal_object.freq_range)[
                            np.newaxis, :
                        ]

            if cal_object.Nfreqs > 1:
                spacing_error, chanwidth_error = cal_object._check_freq_spacing(
                    raise_errors=False
                )
                if spacing_error:
                    warnings.warn(
                        "Selected frequencies are not evenly spaced. This "
                        "will make it impossible to write this data out to "
                        "some file types"
                    )
                elif chanwidth_error:
                    warnings.warn(
                        "Selected frequencies are not contiguous. This "
                        "will make it impossible to write this data out to "
                        "some file types."
                    )

            if not cal_object.metadata_only:
                if not cal_object.future_array_shapes:
                    cal_object.flag_array = cal_object.flag_array[:, :, freq_inds, :, :]
                    if cal_object.input_flag_array is not None:
                        cal_object.input_flag_array = cal_object.input_flag_array[
                            :, :, freq_inds, :, :
                        ]

                if cal_object.cal_type == "delay":
                    pass
                else:
                    if cal_object.future_array_shapes:
                        cal_object.flag_array = cal_object.flag_array[
                            :, freq_inds, :, :
                        ]
                        if cal_object.input_flag_array is not None:
                            cal_object.input_flag_array = cal_object.input_flag_array[
                                :, freq_inds, :, :
                            ]
                        if cal_object.quality_array is not None:
                            cal_object.quality_array = cal_object.quality_array[
                                :, freq_inds, :, :
                            ]
                        cal_object.gain_array = cal_object.gain_array[
                            :, freq_inds, :, :
                        ]

                        if cal_object.total_quality_array is not None:
                            tqa = cal_object.total_quality_array[freq_inds, :, :]
                            cal_object.total_quality_array = tqa
                    else:
                        if cal_object.quality_array is not None:
                            cal_object.quality_array = cal_object.quality_array[
                                :, :, freq_inds, :, :
                            ]
                        cal_object.gain_array = cal_object.gain_array[
                            :, :, freq_inds, :, :
                        ]

                        if cal_object.total_quality_array is not None:
                            tqa = cal_object.total_quality_array[:, freq_inds, :, :]
                            cal_object.total_quality_array = tqa

        if jones is not None:
            jones = uvutils._get_iterable(jones)
            if np.array(jones).ndim > 1:
                jones = np.array(jones).flatten()
            if n_selects > 0:
                history_update_string += ", jones polarization terms"
            else:
                history_update_string += "jones polarization terms"
            n_selects += 1

            jones_inds = np.zeros(0, dtype=np.int64)
            for j in jones:
                if isinstance(j, str):
                    j_num = uvutils.jstr2num(j, x_orientation=self.x_orientation)
                else:
                    j_num = j
                if j_num in cal_object.jones_array:
                    jones_inds = np.append(
                        jones_inds, np.where(cal_object.jones_array == j_num)[0]
                    )
                else:
                    raise ValueError(
                        "Jones term {j} is not present in the jones_array".format(j=j)
                    )

            jones_inds = sorted(set(jones_inds))
            cal_object.Njones = len(jones_inds)
            cal_object.jones_array = cal_object.jones_array[jones_inds]
            if len(jones_inds) > 2:
                jones_separation = (
                    cal_object.jones_array[1:] - cal_object.jones_array[:-1]
                )
                if not uvutils._test_array_constant(jones_separation):
                    warnings.warn(
                        "Selected jones polarization terms are not evenly spaced. This "
                        "is not supported by the calfits format"
                    )

            if not cal_object.metadata_only:
                if cal_object.future_array_shapes:
                    cal_object.flag_array = cal_object.flag_array[:, :, :, jones_inds]
                    if cal_object.quality_array is not None:
                        cal_object.quality_array = cal_object.quality_array[
                            :, :, :, jones_inds
                        ]
                    if cal_object.cal_type == "delay":
                        cal_object.delay_array = cal_object.delay_array[
                            :, :, :, jones_inds
                        ]
                    else:
                        cal_object.gain_array = cal_object.gain_array[
                            :, :, :, jones_inds
                        ]

                    if cal_object.input_flag_array is not None:
                        cal_object.input_flag_array = cal_object.input_flag_array[
                            :, :, :, jones_inds
                        ]

                    if cal_object.total_quality_array is not None:
                        cal_object.total_quality_array = cal_object.total_quality_array[
                            :, :, jones_inds
                        ]
                else:
                    cal_object.flag_array = cal_object.flag_array[
                        :, :, :, :, jones_inds
                    ]
                    if cal_object.quality_array is not None:
                        cal_object.quality_array = cal_object.quality_array[
                            :, :, :, :, jones_inds
                        ]
                    if cal_object.cal_type == "delay":
                        cal_object.delay_array = cal_object.delay_array[
                            :, :, :, :, jones_inds
                        ]
                    else:
                        cal_object.gain_array = cal_object.gain_array[
                            :, :, :, :, jones_inds
                        ]

                    if cal_object.input_flag_array is not None:
                        cal_object.input_flag_array = cal_object.input_flag_array[
                            :, :, :, :, jones_inds
                        ]

                    if cal_object.total_quality_array is not None:
                        cal_object.total_quality_array = cal_object.total_quality_array[
                            :, :, :, jones_inds
                        ]

        if n_selects > 0:
            history_update_string += " using pyuvdata."
            cal_object.history = cal_object.history + history_update_string

        # check if object is self-consistent
        if run_check:
            cal_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return cal_object

    def _convert_from_filetype(self, other):
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        if filetype == "calfits":
            from . import calfits

            other_obj = calfits.CALFITS()
        else:
            raise ValueError("filetype must be calfits.")
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    @classmethod
    @combine_docstrings(initializers.new_uvcal_from_uvdata)
    def initialize_from_uvdata(
        cls,
        uvdata,
        gain_convention,
        cal_style,
        future_array_shapes=True,
        metadata_only=True,
        times=None,
        frequencies=None,
        jones=None,
        **kwargs,
    ):
        """
        Initialize this object based on a UVData object.

        Parameters
        ----------
        uvdata : UVData object
            The UVData object to initialize from.
        future_array_shapes : bool
            Option to use the future array shapes (see `use_future_array_shapes`
            for details). Note that this option is deprecated and will be removed
            in v3.
        metadata_only : bool
            Option to only initialize the metadata. If False, this method also
            initializes the data-like arrays to zeros/ones as appropriate
            (or False for the flag_array) with the appropriate sizes.
        times : array_like of float, optional
            Deprecated alias for ``time_array``. Will be removed in v2.5.
        frequencies : array_like of float, optional
            Deprecated alias for ``freq_array``. Will be removed in v2.5.
        jones : array_like of int, optional
            Deprecated alias for ``jones_array``. Will be removed in v2.5.
        """  # noqa: D207,RST203
        if times is not None:
            warnings.warn(
                "The times keyword is deprecated in favor of time_array and will be "
                "removed in v2.5.",
                DeprecationWarning,
            )
            kwargs["time_array"] = np.array(times)
        if frequencies is not None:
            warnings.warn(
                "The frequencies keyword is deprecated in favor of freq_array and will "
                "be removed in v2.5.",
                DeprecationWarning,
            )
            kwargs["freq_array"] = np.array(frequencies)
        if jones is not None:
            warnings.warn(
                "The jones keyword is deprecated in favor of jones_array and will be "
                "removed in v2.5.",
                DeprecationWarning,
            )
            kwargs["jones_array"] = jones

        new = initializers.new_uvcal_from_uvdata(
            uvdata=uvdata,
            gain_convention=gain_convention,
            cal_style=cal_style,
            empty=not metadata_only,
            **kwargs,
        )

        if future_array_shapes != new.future_array_shapes:
            if future_array_shapes:
                new.use_future_array_shapes()
            else:
                new.use_current_array_shapes()

        return new

    def read_calfits(self, filename, **kwargs):
        """
        Read in data from calfits file(s).

        Parameters
        ----------
        filename : str
            The calfits file to read from.
        read_data : bool
            Read in the gains or delays, quality arrays and flag arrays.
            If set to False, only the metadata will be read in. Setting read_data to
            False results in a metadata only object.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters after reading in the file.
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        """
        from . import calfits

        if isinstance(filename, (list, tuple)):
            warnings.warn(
                "Reading multiple files from file specific read methods is deprecated. "
                "Use the generic `UVCal.read` method instead. This will become an "
                "error in version 2.5",
                DeprecationWarning,
            )

            # cannot just call `read` here and let it handle the recursion because we
            # can get a max recursion depth error. So leave the old handling for
            # recursion until v2.5
            self.read_calfits(filename[0], **kwargs)
            if len(filename) > 1:
                for f in filename[1:]:
                    uvcal2 = UVCal()
                    uvcal2.read_calfits(f, **kwargs)
                    self += uvcal2
                del uvcal2
        else:
            calfits_obj = calfits.CALFITS()
            calfits_obj.read_calfits(filename, **kwargs)
            self._convert_from_filetype(calfits_obj)
            del calfits_obj

    def read_fhd_cal(
        self, cal_file, obs_file, layout_file=None, settings_file=None, **kwargs
    ):
        """
        Read data from an FHD cal.sav file.

        Parameters
        ----------
        cal_file : str or list of str
            The cal.sav file or list of files to read from.
        obs_file : str or list of str
            The obs.sav file or list of files to read from.
        layout_file : str
            The FHD layout file. Required for antenna_positions to be set.
        settings_file : str or list of str, optional
            The settings_file or list of files to read from. Optional,
            but very useful for provenance.
        raw : bool
            Option to use the raw (per antenna, per frequency) solution or
            to use the fitted (polynomial over phase/amplitude) solution.
            Default is True (meaning use the raw solutions).
        read_data : bool
            Read in the gains, quality array and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object. If read_data is False, a settings file must be
            provided.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        extra_history : str or list of str, optional
            String(s) to add to the object's history parameter.
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters after reading in the file.
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        """
        from . import fhd_cal

        if (
            isinstance(cal_file, (list, tuple))
            or isinstance(obs_file, (list, tuple))
            or isinstance(layout_file, (list, tuple))
            or isinstance(settings_file, (list, tuple))
        ):
            warnings.warn(
                "Reading multiple files from file specific read methods is deprecated. "
                "Use the generic `UVCal.read` method instead. This will become an "
                "error in version 2.5",
                DeprecationWarning,
            )
            # cannot just call `read` here and let it handle the recursion because we
            # can get a max recursion depth error. So leave the old handling for
            # recursion until v2.5

        if isinstance(cal_file, (list, tuple)):
            if isinstance(obs_file, (list, tuple)):
                if len(obs_file) != len(cal_file):
                    raise ValueError(
                        "Number of obs_files must match number of cal_files"
                    )
            else:
                raise ValueError("Number of obs_files must match number of cal_files")

            if layout_file is not None:
                if isinstance(layout_file, (list, tuple)):
                    if len(layout_file) != len(cal_file):
                        raise ValueError(
                            "Number of layout_files must match number of cal_files"
                        )
                else:
                    raise ValueError(
                        "Number of layout_files must match number of cal_files"
                    )
                layout_file_use = layout_file[0]
            else:
                layout_file_use = None

            if settings_file is not None:
                if isinstance(settings_file, (list, tuple)):
                    if len(settings_file) != len(cal_file):
                        raise ValueError(
                            "Number of settings_files must match number of cal_files"
                        )
                else:
                    raise ValueError(
                        "Number of settings_files must match number of cal_files"
                    )
                settings_file_use = settings_file[0]
            else:
                settings_file_use = None

            self.read_fhd_cal(
                cal_file[0],
                obs_file[0],
                layout_file=layout_file_use,
                settings_file=settings_file_use,
                **kwargs,
            )
            if len(cal_file) > 1:
                for ind, f in enumerate(cal_file[1:]):
                    uvcal2 = UVCal()
                    if settings_file is not None:
                        settings_file_use = settings_file[ind + 1]
                    if layout_file is not None:
                        layout_file_use = layout_file[ind + 1]
                    uvcal2.read_fhd_cal(
                        f,
                        obs_file[ind + 1],
                        layout_file=layout_file_use,
                        settings_file=settings_file_use,
                        **kwargs,
                    )

                    self += uvcal2
                del uvcal2
        else:
            if isinstance(obs_file, (list, tuple)):
                raise ValueError("Number of obs_files must match number of cal_files")
            if layout_file is not None:
                if isinstance(layout_file, (list, tuple)) and len(layout_file) > 1:
                    raise ValueError(
                        "Number of layout_files must match number of cal_files"
                    )
            if settings_file is not None:
                if isinstance(settings_file, (list, tuple)) and len(settings_file) > 1:
                    raise ValueError(
                        "Number of settings_files must match number of cal_files"
                    )

            fhd_cal_obj = fhd_cal.FHDCal()
            fhd_cal_obj.read_fhd_cal(
                cal_file,
                obs_file,
                layout_file=layout_file,
                settings_file=settings_file,
                **kwargs,
            )
            self._convert_from_filetype(fhd_cal_obj)
            del fhd_cal_obj

    def read(
        self,
        filename,
        *,
        axis=None,
        file_type=None,
        read_data=True,
        use_future_array_shapes=False,
        # checking parameters
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        # file-type specific parameters
        # FHD
        obs_file=None,
        layout_file=None,
        settings_file=None,
        raw=True,
        extra_history=None,
    ):
        """
        Read a generic file into a UVCal object.

        This method supports a number of different types of files.
        Universal parameters (required and optional) are listed directly below,
        followed by parameters used by all file types related to checking. Each file
        type also has its own set of optional parameters that are listed at the end of
        this docstring.

        Parameters
        ----------
        filename : str or array_like of str
            The file(s) or list(s) (or array(s)) of files to read from.
        file_type : str
            One of ['calfits', 'fhd'] or None. If None, the code attempts to guess what
            the file type is based on file extensions (FHD: .sav, .txt;
            uvfits: .calfits). Note that if a list of datasets is passed, the file type
            is determined from the first dataset.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'antenna', 'time', 'freq', 'spw', 'jones' ('freq' is
            not allowed for delay or wideband objects and 'spw' is only allowed for
            wideband objects). Only used if multiple files are passed.
        read_data : bool
            Read in the gains or delays, quality arrays and flag arrays.
            If set to False, only the metadata will be read in. Setting read_data to
            False results in a metadata only object.
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.

        Checking
        --------
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters after reading in the file.

        FHD
        ---
        obs_file : str or list of str
            The obs.sav file or list of files to read from. This is required for FHD
            files.
        layout_file : str
            The FHD layout file. Required for antenna_positions to be set.
        settings_file : str or list of str, optional
            The settings_file or list of files to read from. Optional,
            but very useful for provenance.
        raw : bool
            Option to use the raw (per antenna, per frequency) solution or
            to use the fitted (polynomial over phase/amplitude) solution.
            Default is True (meaning use the raw solutions).

        """
        if isinstance(filename, (list, tuple, np.ndarray)):
            for ind in range(len(filename)):
                if isinstance(filename[ind], (list, tuple, np.ndarray)):
                    raise ValueError(
                        "If filename is a list, tuple or array it cannot be nested or "
                        "multi dimensional."
                    )
            _, extension = os.path.splitext(filename[0])
            n_files = len(filename)
        else:
            _, extension = os.path.splitext(filename)
            n_files = 1

        multi = False
        if n_files > 1:
            multi = True
        elif isinstance(filename, (list, tuple, np.ndarray)):
            filename = filename[0]

        if file_type is None:
            if extension == ".sav" or extension == ".txt":
                file_type = "fhd"
            elif "fits" in extension:
                file_type = "calfits"
            else:
                raise ValueError(
                    "File type could not be determined, use the "
                    "file_type keyword to specify the type."
                )

        if file_type not in ["calfits", "fhd"]:
            raise ValueError("The only supported file_types are 'calfits' and 'fhd'.")

        obs_file_use = None
        layout_file_use = None
        settings_file_use = None
        if file_type == "fhd":
            if obs_file is None:
                raise ValueError("obs_file parameter must be set for FHD files.")
            else:
                if isinstance(obs_file, (list, tuple)):
                    n_obs = len(obs_file)
                    obs_file_use = obs_file[0]
                else:
                    n_obs = 1
                    obs_file_use = obs_file
            if n_obs != n_files:
                raise ValueError("Number of obs_files must match number of cal_files")

            if layout_file is not None:
                if isinstance(layout_file, (list, tuple)):
                    n_layout = len(layout_file)
                    layout_file_use = layout_file[0]
                else:
                    n_layout = 1
                    layout_file_use = layout_file
                if n_layout != n_files:
                    raise ValueError(
                        "Number of layout_files must match number of cal_files"
                    )

            if settings_file is not None:
                if isinstance(settings_file, (list, tuple)):
                    n_settings = len(settings_file)
                    settings_file_use = settings_file[0]
                else:
                    n_settings = 1
                    settings_file_use = settings_file
                if n_settings != n_files:
                    raise ValueError(
                        "Number of settings_files must match number of cal_files"
                    )

        if multi:
            self.read(
                filename[0],
                file_type=file_type,
                read_data=read_data,
                use_future_array_shapes=use_future_array_shapes,
                # checking parameters
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                # file-type specific parameters
                # FHD
                obs_file=obs_file_use,
                layout_file=layout_file_use,
                settings_file=settings_file_use,
                raw=raw,
                extra_history=extra_history,
            )
            uv_list = []
            for ind, file in enumerate(filename[1:]):
                if file_type == "fhd":
                    file_index = ind + 1
                    obs_file_use = obs_file[file_index]
                    if layout_file is not None:
                        layout_file_use = layout_file[file_index]
                    if settings_file is not None:
                        settings_file_use = settings_file[file_index]

                uvcal2 = UVCal()
                uvcal2.read(
                    file,
                    read_data=read_data,
                    file_type=file_type,
                    use_future_array_shapes=use_future_array_shapes,
                    # checking parameters
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    # file-type specific parameters
                    # FHD
                    obs_file=obs_file_use,
                    layout_file=layout_file_use,
                    settings_file=settings_file_use,
                    raw=raw,
                    extra_history=extra_history,
                )
                uv_list.append(uvcal2)
            # Concatenate once at end
            if axis is not None:
                # fast_concat to operates on lists
                self.fast_concat(
                    uv_list,
                    axis,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    inplace=True,
                )
            else:
                # Too much work to rewrite __add__ to operate on lists
                # of files, so instead doing a binary tree merge
                uv_list = [self] + uv_list
                while len(uv_list) > 1:
                    for uv1, uv2 in zip(uv_list[0::2], uv_list[1::2]):
                        uv1.__iadd__(
                            uv2,
                            run_check=run_check,
                            check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability,
                        )
                    uv_list = uv_list[0::2]
                # Because self was at the beginning of the list,
                # everything is merged into it at the end of this loop
        else:
            if file_type == "calfits":
                self.read_calfits(
                    filename,
                    read_data=read_data,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    use_future_array_shapes=use_future_array_shapes,
                )

            elif file_type == "fhd":
                self.read_fhd_cal(
                    filename,
                    obs_file,
                    layout_file=layout_file,
                    settings_file=settings_file,
                    raw=raw,
                    read_data=read_data,
                    extra_history=extra_history,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    use_future_array_shapes=use_future_array_shapes,
                )

    @classmethod
    def from_file(
        cls,
        filename,
        *,
        axis=None,
        file_type=None,
        read_data=True,
        use_future_array_shapes=False,
        # checking parameters
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        # file-type specific parameters
        # FHD
        obs_file=None,
        layout_file=None,
        settings_file=None,
        raw=True,
        extra_history=None,
    ):
        """
        Initialize a new UVCal object by reading the input file.

        This method supports a number of different types of files.
        Universal parameters (required and optional) are listed directly below,
        followed by parameters used by all file types related to checking. Each file
        type also has its own set of optional parameters that are listed at the end of
        this docstring.

        Parameters
        ----------
        filename : str or array_like of str
            The file(s) or list(s) (or array(s)) of files to read from.
        file_type : str
            One of ['calfits', 'fhd'] or None. If None, the code attempts to guess what
            the file type is based on file extensions (FHD: .sav, .txt;
            uvfits: .calfits). Note that if a list of datasets is passed, the file type
            is determined from the first dataset.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'antenna', 'time', 'freq', 'spw', 'jones' ('freq' is
            not allowed for delay or wideband objects and 'spw' is only allowed for
            wideband objects). Only used if multiple files are passed.
        read_data : bool
            Read in the gains or delays, quality arrays and flag arrays.
            If set to False, only the metadata will be read in. Setting read_data to
            False results in a metadata only object.
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.

        Checking
        --------
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters after reading in the file.

        FHD
        ---
        obs_file : str or list of str
            The obs.sav file or list of files to read from. This is required for FHD
            files.
        layout_file : str
            The FHD layout file. Required for antenna_positions to be set.
        settings_file : str or list of str, optional
            The settings_file or list of files to read from. Optional,
            but very useful for provenance.
        raw : bool
            Option to use the raw (per antenna, per frequency) solution or
            to use the fitted (polynomial over phase/amplitude) solution.
            Default is True (meaning use the raw solutions).

        """
        uvc = cls()
        uvc.read(
            filename,
            axis=axis,
            file_type=file_type,
            read_data=read_data,
            use_future_array_shapes=use_future_array_shapes,
            # checking parameters
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            # file-type specific parameters
            # FHD
            obs_file=obs_file,
            layout_file=layout_file,
            settings_file=settings_file,
            raw=raw,
            extra_history=extra_history,
        )
        return uvc

    def write_calfits(
        self,
        filename,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        clobber=False,
    ):
        """
        Write the data to a calfits file.

        Parameters
        ----------
        filename : str
            The calfits file to write to.
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters before writing the file.
        clobber : bool
            Option to overwrite the filename if the file already exists.

        Raises
        ------
        ValueError
            If the UVCal object is a metadata only object.

        """
        if self.metadata_only:
            raise ValueError(
                "Cannot write out metadata only objects to a calfits file."
            )

        calfits_obj = self._convert_to_filetype("calfits")
        calfits_obj.write_calfits(
            filename,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            clobber=clobber,
        )
        del calfits_obj
