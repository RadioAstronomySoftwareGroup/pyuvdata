# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Primary container for radio interferometer calibration solutions."""

import copy
import numpy as np
import threading
import warnings

from ..uvbase import UVBase
from .. import parameter as uvp
from .. import telescopes as uvtel
from .. import utils as uvutils
from ..uvdata import UVData

__all__ = ["UVCal"]


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
            "telescope_location",
            description=desc,
            acceptable_range=(6.35e6, 6.39e6),
            tols=1e-3,
            required=False,
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
            required=False,
        )

        desc = (
            "Option to support 'wide-band' calibration solutions with gains or delays "
            "that apply over a range of frequencies rather than having distinct values "
            "at each frequency. Delay type cal solutions are always 'wide-band' if "
            "future_array_shapes is True. If it is True several other parameters are "
            "affected: future_array_shapes is also True; the data-like arrays have a "
            "spw axis that is Nspws long rather than a frequency axis that is Nfreqs "
            "long; the `freq_range` parameter is required and the `freq_array` "
            "parameter is not required."
        )
        self._wide_band = uvp.UVParameter(
            "wide_band", description=desc, expected_type=bool, value=False,
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
        )
        self._channel_width = uvp.UVParameter(
            "channel_width", description=desc, expected_type=float, tols=1e-3,
        )  # 1 mHz

        desc = (
            "Required if cal_type='delay' or wide_band=True. Frequency range that "
            "solutions are valid for. If future_array_shapes is False it is a "
            "list: [start_frequency, end_frequency], otherwise it is an array of shape "
            "(Nspws, 2). Units are Hz."
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
            "list: [start_time, end_time] in JD. Should only be set in Ntimes is 1."
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
        radian_tol = 1 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)
        desc = "Array of lsts, center of integration, shape (Ntimes), units radians"
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Ntimes",),
            expected_type=float,
            tols=radian_tol,
            required=False,
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
            "The shape depends on cal_type, if the cal_type is 'gain' or "
            "'unknown', the shape is: (Nants_data, 1, Nfreqs, Ntimes, Njones) or "
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
        desc = "cal type parameter. Values are delay, gain or unknown."
        self._cal_type = uvp.UVParameter(
            "cal_type",
            form="str",
            expected_type=str,
            value="unknown",
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
            "flex_spw", description=desc, expected_type=bool, value=False,
        )

        desc = (
            "Required if flex_spw = True. Maps individual channels along the "
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
            "future_array_shapes", description=desc, expected_type=bool, value=False,
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
            "Required if cal_style = 'sky'. Short string describing field "
            "center or dominant source."
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
            "Array of input flags, True is flagged. shape: "
            "(Nants_data, 1, Nfreqs, Ntimes, Njones) or "
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
            "The shape depends on cal_type, if the cal_type is 'gain' or "
            "'unknown', the shape is: (1, Nfreqs, Ntimes, Njones) or "
            "(Nfreqs, Ntimes, Njones) if future_array_shapes=True, "
            "if the cal_type is 'delay', the shape is (1, 1, Ntimes, Njones) or "
            "(1, Ntimes, Njones) if future_array_shapes=True, type = float."
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
            "filename", required=False, description=desc, expected_type=str,
        )

        super(UVCal, self).__init__()

    def _set_flex_spw(self):
        """
        Set flex_spw to True, and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by the file-reading methods to indicate that an object has multiple spectral
        windows concatenated together across the frequency axis.
        """
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
        elif self.future_array_shapes:
            assert self.cal_type != "delay", (
                "delay objects cannot have wide_band=False if future_array_shapes is "
                "True"
            )
        self.wide_band = wide_band

        if wide_band:
            self._freq_array.required = False
            self._channel_width.required = False
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

    def _set_unknown_cal_type(self):
        """Set cal_type to 'unknown' and adjust required parameters."""
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
        self._sky_field.required = True
        self._sky_catalog.required = True
        self._ref_antenna_name.required = True

    def _set_redundant(self):
        """Set cal_style to 'redundant' and adjust required parameters."""
        self.cal_style = "redundant"
        self._sky_field.required = False
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
            cal_type = "unknown"

        if cal_type == "gain":
            return ["gain_array", "flag_array", "quality_array"]
        elif cal_type == "delay":
            return ["delay_array", "flag_array", "quality_array"]
        else:
            return ["flag_array", "quality_array"]

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
        if self.Nspws > 1:
            raise ValueError("Cannot use current array shapes if Nspws > 1.")

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
        self.freq_array = self.freq_array[np.newaxis, :]

        self.freq_range = self.freq_range[0, :].tolist()
        self._freq_range.form = (2,)

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
                    warnings.warn(
                        "antenna_positions is not set. Using known values "
                        f"for {telescope_obj.telescope_name}."
                    )
                    telescope_ant_inds = np.array(telescope_ant_inds)
                    self.antenna_positions = telescope_obj.antenna_positions[
                        telescope_ant_inds, :
                    ]
        else:
            raise ValueError(
                f"Telescope {self.telescope_name} is not in known_telescopes."
            )

    def _set_lsts_helper(self):
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
        unique_times, inverse_inds = np.unique(self.time_array, return_inverse=True)
        unique_lst_array = uvutils.get_lst_for_time(
            unique_times, latitude, longitude, altitude
        )
        self.lst_array = unique_lst_array[inverse_inds]
        return

    def set_lsts_from_time_array(self, background=False):
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
            self._set_lsts_helper()
            return
        else:
            proc = threading.Thread(target=self._set_lsts_helper)
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
        else:
            # If this isn't a flex_spw data set, then there is only 1 spectral window,
            # which means that the check always passes
            pass
        return True

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
        self, check_extra=True, run_check_acceptability=True, check_freq_spacing=False
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

        # If the telescope location is not set issue a deprecation warning
        if self.telescope_location is None:
            warnings.warn(
                "The telescope_location is not set. It will be a required "
                "parameter starting in pyuvdata version 2.3",
                category=DeprecationWarning,
            )

        # If the antenna positions parameter is not set issue a deprecation warning
        if self.antenna_positions is None:
            warnings.warn(
                "The antenna_positions parameter is not set. It will be a required "
                "parameter starting in pyuvdata version 2.3",
                category=DeprecationWarning,
            )

        # If the antenna positions parameter is not set issue a deprecation warning
        if self.lst_array is None:
            warnings.warn(
                "The lst_array is not set. It will be a required "
                "parameter starting in pyuvdata version 2.3",
                category=DeprecationWarning,
            )

        # if wide_band is True, Nfreqs must be 1.
        if self.wide_band:
            if self.Nfreqs != 1:
                warnings.warn(
                    "Nfreqs will be required to be 1 for wide_band cals (including "
                    "all delay cals) starting in version 3.0",
                    category=DeprecationWarning,
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

        if check_freq_spacing:
            self._check_freq_spacing()

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
                if isinstance(self.freq_range, list):
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
        new_quality = np.repeat(self.quality_array, Nfreqs_use, axis=freq_axis)
        self._set_gain()
        self._set_wide_band(wide_band=False)
        self.channel_width = channel_width
        self.gain_array = gain_array
        self.quality_array = new_quality
        self.delay_array = None
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

        # check if object is self-consistent
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
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

        # Check to make sure that both objects are consistent w/ use of flex_spw
        if this.flex_spw != other.flex_spw:
            raise ValueError(
                "To combine these data, flex_spw must be set to the same "
                "value (True or False) for both objects."
            )

        # check that both objects have the same array shapes
        if this.future_array_shapes != other.future_array_shapes:
            raise ValueError(
                "Both objects must have the same `future_array_shapes` parameter. "
                "Use the `use_future_array_shapes` or `use_current_array_shapes` "
                "methods to convert them."
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
        if this.cal_type == "delay":
            compatibility_params.append("_freq_range")

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
        both_jones = np.intersect1d(this.jones_array, other.jones_array)
        both_times = np.intersect1d(this.time_array, other.time_array)
        if this.cal_type != "delay":

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

        else:
            # delay type cal
            # Make a non-empty array so we raise an error if other data is duplicated
            both_freq = [0]

        both_ants = np.intersect1d(this.ant_array, other.ant_array)
        if len(both_jones) > 0:
            if len(both_times) > 0:
                if len(both_freq) > 0:
                    if len(both_ants) > 0:
                        raise ValueError(
                            "These objects have overlapping data and"
                            " cannot be combined."
                        )

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

        # adding along frequency axis is not supported for delay-type cal files
        if this.cal_type == "gain":
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
        else:
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

        # Pad out self to accommodate new data
        if len(anew_inds) > 0:
            this.ant_array = np.concatenate(
                [this.ant_array, other.ant_array[anew_inds]]
            )
            order = np.argsort(this.ant_array)
            this.ant_array = this.ant_array[order]
            if not self.metadata_only:
                if self.future_array_shapes:
                    zero_pad_data = np.zeros(
                        (
                            len(anew_inds),
                            this.quality_array.shape[1],
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    zero_pad_flags = np.zeros(
                        (
                            len(anew_inds),
                            this.quality_array.shape[1],
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                else:
                    zero_pad_data = np.zeros(
                        (
                            len(anew_inds),
                            1,
                            this.quality_array.shape[2],
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    zero_pad_flags = np.zeros(
                        (len(anew_inds), 1, this.Nfreqs, this.Ntimes, this.Njones,)
                    )
                if this.cal_type == "delay":
                    this.delay_array = np.concatenate(
                        [this.delay_array, zero_pad_data], axis=0
                    )[order]
                else:
                    this.gain_array = np.concatenate(
                        [this.gain_array, zero_pad_data], axis=0
                    )[order]
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad_flags], axis=0
                ).astype(np.bool_)[order]
                this.quality_array = np.concatenate(
                    [this.quality_array, zero_pad_data], axis=0
                )[order]

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

                if (
                    this.input_flag_array is not None
                    or other.input_flag_array is not None
                ):
                    if self.future_array_shapes:
                        zero_pad = np.zeros(
                            (
                                len(anew_inds),
                                this.quality_array.shape[1],
                                this.Ntimes,
                                this.Njones,
                            )
                        )
                    else:
                        zero_pad = np.zeros(
                            (len(anew_inds), 1, this.Nfreqs, this.Ntimes, this.Njones,)
                        )
                    if this.input_flag_array is not None:
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=0
                        ).astype(np.bool_)[order]
                    elif other.input_flag_array is not None:
                        if self.future_array_shapes:
                            this.input_flag_array = np.array(
                                1
                                - np.zeros(
                                    (
                                        this.Nants_data,
                                        this.quality_array.shape[1],
                                        this.Ntimes,
                                        this.Njones,
                                    )
                                )
                            ).astype(np.bool_)
                        else:
                            this.input_flag_array = np.array(
                                1
                                - np.zeros(
                                    (
                                        this.Nants_data,
                                        1,
                                        this.Nfreqs,
                                        this.Ntimes,
                                        this.Njones,
                                    )
                                )
                            ).astype(np.bool_)
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=0
                        ).astype(np.bool_)[order]

        if len(fnew_inds) > 0:
            # Exploit the fact that quality array has the same dimensions as the
            # main data.
            # Also do not need to worry about different cases for gain v. delay type
            if self.future_array_shapes:
                zero_pad = np.zeros(
                    (
                        this.quality_array.shape[0],
                        len(fnew_inds),
                        this.Ntimes,
                        this.Njones,
                    )
                )
                this.freq_array = np.concatenate(
                    [this.freq_array, other.freq_array[fnew_inds]]
                )
            else:
                zero_pad = np.zeros(
                    (
                        this.quality_array.shape[0],
                        1,
                        len(fnew_inds),
                        this.Ntimes,
                        this.Njones,
                    )
                )
                this.freq_array = np.concatenate(
                    [this.freq_array, other.freq_array[:, fnew_inds]], axis=1
                )

            if this.flex_spw:
                this.flex_spw_id_array = np.concatenate(
                    [this.flex_spw_id_array, other.flex_spw_id_array[fnew_inds]]
                )
                this.spw_array = np.concatenate([this.spw_array, other.spw_array])
                # We want to preserve per-spw information based on first appearance
                # in the concatenated array.
                unique_index = np.sort(np.unique(this.spw_array, return_index=True)[1])
                this.spw_array = this.spw_array[unique_index]
                if this.future_array_shapes:
                    this.freq_range = np.concatenate(
                        [this.freq_range, other.freq_range], axis=0
                    )
                    this.freq_range = this.freq_range[unique_index, :]
                this.Nspws = len(this.spw_array)

                # If we have a flex/multi-spw data set, need to sort out the order of
                # the individual windows first.
                order = np.concatenate(
                    [
                        np.where(this.flex_spw_id_array == idx)[0]
                        for idx in sorted(this.spw_array)
                    ]
                )

                # With spectral windows sorted, check and see if channels within
                # windows need sorting. If they are ordered in ascending or descending
                # fashion, leave them be. If not, sort in ascending order
                for idx in this.spw_array:
                    select_mask = this.flex_spw_id_array[order] == idx
                    check_freqs = (
                        this.freq_array[order[select_mask]]
                        if this.future_array_shapes
                        else this.freq_array[0, order[select_mask]]
                    )
                    if (not np.all(check_freqs[1:] > check_freqs[:-1])) and (
                        not np.all(check_freqs[1:] < check_freqs[:-1])
                    ):
                        subsort_order = order[select_mask]
                        order[select_mask] = subsort_order[np.argsort(check_freqs)]
                this.flex_spw_id_array = this.flex_spw_id_array[order]
                this.spw_array = np.array(sorted(this.spw_array))
            else:
                if this.future_array_shapes:
                    order = np.argsort(this.freq_array)
                else:
                    order = np.argsort(this.freq_array[0, :])

            if this.future_array_shapes:
                this.freq_array = this.freq_array[order]
            else:
                this.freq_array = this.freq_array[:, order]

            if this.flex_spw or this.future_array_shapes:
                this.channel_width = np.concatenate(
                    [this.channel_width, other.channel_width[fnew_inds]]
                )
                this.channel_width = this.channel_width[order]

            if not self.metadata_only:
                if self.future_array_shapes:
                    this.gain_array = np.concatenate(
                        [this.gain_array, zero_pad], axis=1
                    )[:, order, :, :]
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad], axis=1
                    ).astype(np.bool_)[:, order, :, :]
                    this.quality_array = np.concatenate(
                        [this.quality_array, zero_pad], axis=1
                    )[:, order, :, :]

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros((len(fnew_inds), this.Ntimes, this.Njones))
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=0
                        )[order, :, :]
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros((len(fnew_inds), this.Ntimes, this.Njones))
                        this.total_quality_array = np.zeros(
                            (Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=0
                        )[order, :, :]
                else:
                    this.gain_array = np.concatenate(
                        [this.gain_array, zero_pad], axis=2
                    )[:, :, order, :, :]
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad], axis=2
                    ).astype(np.bool_)[:, :, order, :, :]
                    this.quality_array = np.concatenate(
                        [this.quality_array, zero_pad], axis=2
                    )[:, :, order, :, :]

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (1, len(fnew_inds), this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )[:, order, :, :]
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (1, len(fnew_inds), this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.zeros(
                            (1, Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )[:, order, :, :]

                if (
                    this.input_flag_array is not None
                    or other.input_flag_array is not None
                ):
                    if self.future_array_shapes:
                        zero_pad = np.zeros(
                            (
                                this.flag_array.shape[0],
                                len(fnew_inds),
                                this.Ntimes,
                                this.Njones,
                            )
                        )
                        if this.input_flag_array is not None:
                            this.input_flag_array = np.concatenate(
                                [this.input_flag_array, 1 - zero_pad], axis=1
                            ).astype(np.bool_)[:, order, :, :]
                        elif other.input_flag_array is not None:
                            this.input_flag_array = np.array(
                                1
                                - np.zeros(
                                    (
                                        this.flag_array.shape[0],
                                        this.flag_array.shape[1],
                                        this.flag_array.shape[2],
                                        this.Njones,
                                    )
                                )
                            ).astype(np.bool_)
                            this.input_flag_array = np.concatenate(
                                [this.input_flag_array, 1 - zero_pad], axis=1
                            ).astype(np.bool_)[:, order, :, :]
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
                        if this.input_flag_array is not None:
                            this.input_flag_array = np.concatenate(
                                [this.input_flag_array, 1 - zero_pad], axis=2
                            ).astype(np.bool_)[:, :, order, :, :]
                        elif other.input_flag_array is not None:
                            this.input_flag_array = np.array(
                                1
                                - np.zeros(
                                    (
                                        this.flag_array.shape[0],
                                        1,
                                        this.flag_array.shape[2],
                                        this.flag_array.shape[3],
                                        this.Njones,
                                    )
                                )
                            ).astype(np.bool_)
                            this.input_flag_array = np.concatenate(
                                [this.input_flag_array, 1 - zero_pad], axis=2
                            ).astype(np.bool_)[:, :, order, :, :]

        if len(tnew_inds) > 0:
            # Exploit the fact that quality array has the same dimensions as
            # the main data
            this.time_array = np.concatenate(
                [this.time_array, other.time_array[tnew_inds]]
            )
            this.lst_array = np.concatenate(
                [this.lst_array, other.lst_array[tnew_inds]]
            )
            order = np.argsort(this.time_array)
            this.time_array = this.time_array[order]
            this.lst_array = this.lst_array[order]
            if self.future_array_shapes:
                this.integration_time = np.concatenate(
                    [this.integration_time, other.integration_time[tnew_inds]]
                )
                this.integration_time = this.integration_time[order]

            if not self.metadata_only:
                if self.future_array_shapes:
                    zero_pad_data = np.zeros(
                        (
                            this.quality_array.shape[0],
                            this.quality_array.shape[1],
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
                        )[:, :, order, :]
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=2
                        )[:, :, order, :]
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=2
                    ).astype(np.bool_)[:, :, order, :]
                    this.quality_array = np.concatenate(
                        [this.quality_array, zero_pad_data], axis=2
                    )[:, :, order, :]

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (this.quality_array.shape[1], len(tnew_inds), this.Njones,)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )[:, order, :]
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (this.quality_array.shape[1], len(tnew_inds), this.Njones,)
                        )
                        this.total_quality_array = np.zeros(
                            (Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=1
                        )[:, order, :]

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
                        ).astype(np.bool_)[:, :, order, :]
                    elif other.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.flag_array.shape[0],
                                this.flag_array.shape[1],
                                len(tnew_inds),
                                this.Njones,
                            )
                        )
                        this.input_flag_array = np.array(
                            1
                            - np.zeros(
                                (
                                    this.flag_array.shape[0],
                                    this.flag_array.shape[1],
                                    this.flag_array.shape[2],
                                    this.Njones,
                                )
                            )
                        ).astype(np.bool_)
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=2
                        ).astype(np.bool_)[:, :, order, :]
                else:
                    zero_pad_data = np.zeros(
                        (
                            this.quality_array.shape[0],
                            1,
                            this.quality_array.shape[2],
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
                        )[:, :, :, order, :]
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=3
                        )[:, :, :, order, :]
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=3
                    ).astype(np.bool_)[:, :, :, order, :]
                    this.quality_array = np.concatenate(
                        [this.quality_array, zero_pad_data], axis=3
                    )[:, :, :, order, :]

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                1,
                                this.quality_array.shape[2],
                                len(tnew_inds),
                                this.Njones,
                            )
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )[:, :, order, :]
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                1,
                                this.quality_array.shape[2],
                                len(tnew_inds),
                                this.Njones,
                            )
                        )
                        this.total_quality_array = np.zeros(
                            (1, Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )[:, :, order, :]

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
                        ).astype(np.bool_)[:, :, :, order, :]
                    elif other.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.flag_array.shape[0],
                                1,
                                this.flag_array.shape[2],
                                len(tnew_inds),
                                this.Njones,
                            )
                        )
                        this.input_flag_array = np.array(
                            1
                            - np.zeros(
                                (
                                    this.flag_array.shape[0],
                                    1,
                                    this.flag_array.shape[2],
                                    this.flag_array.shape[3],
                                    this.Njones,
                                )
                            )
                        ).astype(np.bool_)
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=3
                        ).astype(np.bool_)[:, :, :, order, :]

        if len(jnew_inds) > 0:
            # Exploit the fact that quality array has the same dimensions as
            # the main data
            this.jones_array = np.concatenate(
                [this.jones_array, other.jones_array[jnew_inds]]
            )
            order = np.argsort(np.abs(this.jones_array))
            this.jones_array = this.jones_array[order]
            if not self.metadata_only:
                if self.future_array_shapes:
                    zero_pad_data = np.zeros(
                        (
                            this.quality_array.shape[0],
                            this.quality_array.shape[1],
                            this.quality_array.shape[2],
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
                        )[:, :, :, order]
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=3
                        )[:, :, :, order]
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=3
                    ).astype(np.bool_)[:, :, :, order]
                    this.quality_array = np.concatenate(
                        [this.quality_array, zero_pad_data], axis=3
                    )[:, :, :, order]

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                this.quality_array.shape[1],
                                this.quality_array.shape[2],
                                len(jnew_inds),
                            )
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )[:, :, order]
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                this.quality_array.shape[1],
                                this.quality_array.shape[2],
                                len(jnew_inds),
                            )
                        )
                        this.total_quality_array = np.zeros(
                            (Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=2
                        )[:, :, order]

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
                        ).astype(np.bool_)[:, :, :, order]
                    elif other.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.flag_array.shape[0],
                                this.flag_array.shape[1],
                                this.flag_array.shape[2],
                                len(jnew_inds),
                            )
                        )
                        this.input_flag_array = np.array(
                            1
                            - np.zeros(
                                (
                                    this.flag_array.shape[0],
                                    this.flag_array.shape[1],
                                    this.flag_array.shape[2],
                                    this.Njones,
                                )
                            )
                        ).astype(np.bool_)
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=3
                        ).astype(np.bool_)[:, :, :, order]
                else:
                    zero_pad_data = np.zeros(
                        (
                            this.quality_array.shape[0],
                            1,
                            this.quality_array.shape[2],
                            this.quality_array.shape[3],
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
                        )[:, :, :, :, order]
                    else:
                        this.gain_array = np.concatenate(
                            [this.gain_array, zero_pad_data], axis=4
                        )[:, :, :, :, order]
                    this.flag_array = np.concatenate(
                        [this.flag_array, 1 - zero_pad_flags], axis=4
                    ).astype(np.bool_)[:, :, :, :, order]
                    this.quality_array = np.concatenate(
                        [this.quality_array, zero_pad_data], axis=4
                    )[:, :, :, :, order]

                    if this.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                1,
                                this.quality_array.shape[2],
                                this.quality_array.shape[3],
                                len(jnew_inds),
                            )
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=3
                        )[:, :, :, order]
                    elif other.total_quality_array is not None and can_combine_tqa:
                        zero_pad = np.zeros(
                            (
                                1,
                                this.quality_array.shape[2],
                                this.quality_array.shape[3],
                                len(jnew_inds),
                            )
                        )
                        this.total_quality_array = np.zeros(
                            (1, Nf_tqa, this.Ntimes, this.Njones)
                        )
                        this.total_quality_array = np.concatenate(
                            [this.total_quality_array, zero_pad], axis=3
                        )[:, :, :, order]

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
                        ).astype(np.bool_)[:, :, :, :, order]
                    elif other.input_flag_array is not None:
                        zero_pad = np.zeros(
                            (
                                this.flag_array.shape[0],
                                1,
                                this.flag_array.shape[2],
                                this.flag_array.shape[3],
                                len(jnew_inds),
                            )
                        )
                        this.input_flag_array = np.array(
                            1
                            - np.zeros(
                                (
                                    this.flag_array.shape[0],
                                    1,
                                    this.flag_array.shape[2],
                                    this.flag_array.shape[3],
                                    this.Njones,
                                )
                            )
                        ).astype(np.bool_)
                        this.input_flag_array = np.concatenate(
                            [this.input_flag_array, 1 - zero_pad], axis=4
                        ).astype(np.bool_)[:, :, :, :, order]

        # Now populate the data
        if not self.metadata_only:
            jones_t2o = np.nonzero(np.in1d(this.jones_array, other.jones_array))[0]
            times_t2o = np.nonzero(np.in1d(this.time_array, other.time_array))[0]
            if self.future_array_shapes:
                freqs_t2o = np.nonzero(np.in1d(this.freq_array, other.freq_array))[0]
            else:
                freqs_t2o = np.nonzero(
                    np.in1d(this.freq_array[0, :], other.freq_array[0, :])
                )[0]
            ants_t2o = np.nonzero(np.in1d(this.ant_array, other.ant_array))[0]
            if self.future_array_shapes:
                if this.cal_type == "delay":
                    this.delay_array[
                        np.ix_(ants_t2o, [0], times_t2o, jones_t2o)
                    ] = other.delay_array
                    this.quality_array[
                        np.ix_(ants_t2o, [0], times_t2o, jones_t2o)
                    ] = other.quality_array
                    this.flag_array[
                        np.ix_(ants_t2o, [0], times_t2o, jones_t2o)
                    ] = other.flag_array
                else:
                    this.gain_array[
                        np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                    ] = other.gain_array
                    this.quality_array[
                        np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                    ] = other.quality_array
                    this.flag_array[
                        np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                    ] = other.flag_array
                if this.total_quality_array is not None:
                    if other.total_quality_array is not None:
                        if this.cal_type == "delay":
                            this.total_quality_array[
                                np.ix_([0], times_t2o, jones_t2o)
                            ] = other.total_quality_array
                        else:
                            this.total_quality_array[
                                np.ix_(freqs_t2o, times_t2o, jones_t2o)
                            ] = other.total_quality_array
                if this.input_flag_array is not None:
                    if other.input_flag_array is not None:
                        if this.cal_type == "delay":
                            this.input_flag_array[
                                np.ix_(ants_t2o, [0], times_t2o, jones_t2o)
                            ] = other.input_flag_array
                        else:
                            this.input_flag_array[
                                np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)
                            ] = other.input_flag_array
            else:
                if this.cal_type == "delay":
                    this.delay_array[
                        np.ix_(ants_t2o, [0], [0], times_t2o, jones_t2o)
                    ] = other.delay_array
                    this.quality_array[
                        np.ix_(ants_t2o, [0], [0], times_t2o, jones_t2o)
                    ] = other.quality_array
                else:
                    this.gain_array[
                        np.ix_(ants_t2o, [0], freqs_t2o, times_t2o, jones_t2o)
                    ] = other.gain_array
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

        # Update N parameters (e.g. Njones)
        this.Njones = this.jones_array.shape[0]
        this.Ntimes = this.time_array.shape[0]
        if this.cal_type == "gain":
            this.Nfreqs = this.freq_array.size
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
        self, other, run_check=True, check_extra=True, run_check_acceptability=True,
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
                        cal_object.quality_array = cal_object.quality_array[
                            :, spw_inds, :, :
                        ]
                        if cal_object.total_quality_array is not None:
                            tqa = cal_object.total_quality_array[spw_inds, :, :]
                            cal_object.total_quality_array = tqa

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

            if cal_object.flex_spw:
                cal_object.flex_spw_id_array = cal_object.flex_spw_id_array[freq_inds]
                spw_mask = np.isin(cal_object.spw_array, cal_object.flex_spw_id_array)
                cal_object.spw_array = cal_object.spw_array[spw_mask]
                if cal_object.freq_range is not None and cal_object.future_array_shapes:
                    cal_object.freq_range = cal_object.freq_range[spw_mask, :]
                cal_object.Nspws = len(cal_object.spw_array)

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
    def initialize_from_uvdata(
        cls,
        uvdata,
        gain_convention,
        cal_style,
        future_array_shapes=True,
        metadata_only=True,
        include_uvdata_history=True,
        cal_type="gain",
        times=None,
        integration_time=None,
        time_range=None,
        frequencies=None,
        channel_width=None,
        flex_spw=None,
        flex_spw_id_array=None,
        wide_band=None,
        freq_range=None,
        spw_array=None,
        jones=None,
        ref_antenna_name=None,
        sky_catalog=None,
        sky_field=None,
        diffuse_model=None,
        baseline_range=None,
        Nsources=None,  # noqa
        observer=None,
        gain_scale=None,
        git_hash_cal=None,
        git_origin_cal=None,
        extra_keywords=None,
    ):
        """
        Initialize this object based on a UVData object.

        Parameters
        ----------
        uvdata : UVData object
            The UVData object to initialize from.
        gain_convention : str
            What gain convention the UVCal object should be initialized to
            ("multiply" or "divide").
        cal_style : str
            What calibration style the UVCal object should be initialized to
            ("sky" or "redundant").
        future_array_shapes : bool
            Option to use the future array shapes (see `use_future_array_shapes`
            for details).
        metadata_only : bool
            Option to only initialize the metadata. If False, this method also
            initializes the data-like arrays to zeros (or False for the flag_array)
            with the appropriate sizes.
        include_uvdata_history : bool
            Option to include the history from the uvdata object in the uvcal history.
        cal_type : str
            What cal_type the UVCal object should be initialized to
            ("gain", or "delay").
        times : array_like of float, optional
            Calibration times in decimal Julian date. If None, use all unique times from
            uvdata.
        integration_time : float or array_like of float, optional
            Calibration integration time in seconds, an array of shape (Ntimes,)
            or a scalar if `future_array_shapes` is False. Required if `time_array` is
            not None, ignored otherwise.
        time_range : array_like of float, optional
            Range of times that calibration is valid for in decimal Julian dates,
            shape (2,). Should only be set if `time_array` is size (1,)
        frequencies : array_like of float, optional
            Calibration frequencies (units Hz), shape (Nfreqs,). Defaulted to the
            freq_array from uvdata if `cal_type="gain"` and `wide_band` is not set to
            `True`. Ignored if `cal_type="delay"` or `wide_band=True`.
        channel_width : float or array_like of float, optional
            Calibration channel width in Hz, an array of shape (Nfreqs,)
            or a scalar if `future_array_shapes` is False. Required if freq_array is
            not None and `cal_type="gain"` and `wide_band` is not set to
            `True`, ignored otherwise.
        flex_spw : bool, optional
            Option to use flexible spectral windows. Ignored if freq_array is None or
            `cal_type="delay"` or `wide_band=True`.
        flex_spw_id_array : array_like of int, optional
            Array giving the spectral window value for each frequency channel,
            shape (Nfreqs,). Ignored if freq_array is None or `cal_type="delay"` or
            `wide_band=True`. Required if freq_array is not None and flex_spw is True
            and `cal_type="gain"` and `wide_band` is not set to `True`.
        wide_band : bool, optional
            Option to use wide-band calibration. Requires `future_array_shapes` to be
            `True`. Defaulted to `True` if `future_array_shapes` is True and
            `cal_type="delay"`, defaulted to `False` otherwise.
        freq_range : array_like of float, optional
            Frequency range that solutions are valid for in Hz, shape (Nspws, 2) if
            `future_array_shapes` is True, shape (2,) otherwise.
            Defaulted to the min, max of freq_array if `wide_band` is True or
            `cal_type="delay"`. Defaulting is done per spectral window if uvdata has
            multiple spectral windows and `future_array_shapes` is True.
        spw_array : array_like of int, optional
            Array giving the spectral window numbers. Required if either `wide_band` is
            True or `cal_type="delay"` and if freq_range is not None and has multiple
            spectral windows, ignored otherwise. Defaulted to uvdata.spw_array if
            either `wide_band` is True or `cal_type="delay"` and if freq_range is None.
        jones : array_like of int, optional
            Calibration Jones elements. If None, defaults to [-5, -6] (jxx, jyy) if
            uvdata is in linear pol. [-1, -2] (jrr, jll) if uvdata is in circular pol.
            A ValueError is raised if jones_array is None and uvdata is in
            psuedo-stokes.
        ref_antenna_name : str, optional
            Phase reference antenna, required if cal_style = "sky".
        sky_catalog : str, optional
            Name of calibration catalog, required if cal_sky = "sky".
        sky_field : str, optional
            Short string describing field center or dominant source, required if
            cal_sky = "sky".
        diffuse_model : str, optional
            Name of diffuse model.
        baseline_range : array_like of float, optional
            Range of baselines used for calibration.
        Nsources : int, optional
            Number of sources used.
        observer : str, optional
            Name of observer who calculated calibration solutions.
        gain_scale : str, optional
            The gain scale of the calibration, which indicates the units of the
            calibrated visibilities. For example, Jy or K str.
        git_hash_cal : str, optional
            Commit hash of calibration software (from git_origin_cal) used to generate
            solutions.
        git_origin_cal : str, optional
            Origin (on github for e.g) of calibration software. Url and branch.
        extra_keywords : dict, optional
            Any user supplied extra keywords, type=dict.

        Raises
        ------
        ValueError
            If cal_style is 'sky' and ref_antenna_name, sky_catalog or sky_field are not
            provided;
            if freq_array is not None, flex_spw is True and flex_spw_id_array is None;
            if freq_array and channel_width are None and the uvdata object does not use
            flexible spectral windows and the uvdata channel width varies;
            if time_array and integration_time are None and the uvdata integration
            time varies;
            if time_array is not None and integration_time is not specified or is the
            wrong type;
            if jones_array is None and uvdata is in psuedo-stokes.

        """
        if not issubclass(type(uvdata), UVData):
            raise ValueError("uvdata must be a UVData (or subclassed) object.")

        uvc = cls()

        if cal_type not in ["delay", "gain"]:
            raise ValueError("cal_type must be either 'gain' or 'delay'.")

        if cal_type == "gain":
            uvc._set_gain()
        elif cal_type == "delay":
            uvc._set_delay()

        if future_array_shapes:
            uvc._set_future_array_shapes()

        if wide_band is not None:
            uvc._set_wide_band(wide_band=wide_band)

        uvc.cal_style = cal_style
        uvc.gain_convention = gain_convention

        if cal_style == "sky" and (
            ref_antenna_name is None or sky_catalog is None or sky_field is None
        ):
            raise ValueError(
                "If cal_style is 'sky', ref_antenna_name, sky_catalog and sky_field "
                "must all be provided."
            )
        if ref_antenna_name is not None:
            uvc.ref_antenna_name = ref_antenna_name
        if sky_catalog is not None:
            uvc.sky_catalog = sky_catalog
        if sky_field is not None:
            uvc.sky_field = sky_field
        if diffuse_model is not None:
            uvc.diffuse_model = diffuse_model
        if baseline_range is not None:
            uvc.baseline_range = baseline_range
        if Nsources is not None:
            uvc.Nsources = Nsources
        if observer is not None:
            uvc.observer = observer
        if gain_scale is not None:
            uvc.gain_scale = gain_scale
        if git_hash_cal is not None:
            uvc.git_hash_cal = git_hash_cal
        if git_origin_cal is not None:
            uvc.git_origin_cal = git_origin_cal
        if extra_keywords is not None:
            uvc.extra_keywords = extra_keywords

        params_to_copy = [
            "telescope_name",
            "telescope_location",
            "antenna_numbers",
            "antenna_names",
            "antenna_positions",
            "Nants_telescope",
            "Nants_data",
            "x_orientation",
        ]
        if uvc.cal_type != "delay" and not uvc.wide_band:
            if frequencies is None:
                params_to_copy.extend(["Nfreqs", "flex_spw", "spw_array", "Nspws"])
                if uvdata.flex_spw:
                    uvc._set_flex_spw()
                if uvdata.future_array_shapes == uvc.future_array_shapes:
                    params_to_copy.extend(["freq_array"])
                else:
                    if uvc.future_array_shapes:
                        uvc.freq_array = uvdata.freq_array[0, :]
                    else:
                        uvc.freq_array = uvdata.freq_array[np.newaxis, :]

                if (
                    uvdata.flex_spw
                    or uvdata.future_array_shapes == uvc.future_array_shapes
                ):
                    params_to_copy.extend(["channel_width"])
                else:
                    if uvc.future_array_shapes:
                        uvc.channel_width = np.full(
                            uvc.freq_array.size, uvdata.channel_width, dtype=np.float64
                        )
                    else:
                        uvdata_channel_widths = np.unique(uvdata.channel_width)
                        if uvdata_channel_widths.size == 1:
                            uvc.channel_width = uvdata_channel_widths[0]
                        else:
                            raise ValueError(
                                "uvdata has varying channel widths but does not have "
                                "flexible spectral windows and future_array_shapes is "
                                "False. Please specify frequencies and channel_width."
                            )
                if uvdata.flex_spw:
                    params_to_copy.extend(["flex_spw_id_array"])
            else:
                if frequencies.ndim != 1:
                    raise ValueError("Frequencies must be a 1 dimensional array")

                if future_array_shapes:
                    uvc.freq_array = frequencies
                else:
                    uvc.freq_array = frequencies[np.newaxis, :]
                uvc.Nfreqs = frequencies.size

                if flex_spw:
                    uvc._set_flex_spw()
                    if flex_spw_id_array is None:
                        raise ValueError(
                            "If frequencies is provided and flex_spw is True, a "
                            "flex_spw_id_array must be provided."
                        )
                    uvc.flex_spw_id_array = flex_spw_id_array
                    uvc.spw_array = np.unique(uvc.flex_spw_id_array)
                    uvc.Nspws = uvc.spw_array.size
                else:
                    uvc.spw_array = np.array([0])
                    uvc.Nspws = 1
                if channel_width is None:
                    raise ValueError(
                        "channel_width must be provided if frequencies is provided"
                    )
                if future_array_shapes or flex_spw:
                    if isinstance(channel_width, (np.ndarray, list)):
                        uvc.channel_width = np.asarray(channel_width)
                    else:
                        uvc.channel_width = np.full(
                            uvc.Nfreqs, channel_width, dtype=np.float64
                        )
                else:
                    if isinstance(channel_width, (np.ndarray, list)):
                        raise ValueError(
                            "channel_width must be scalar if both future_array_shapes "
                            "and flex_spw are False."
                        )
                    uvc.channel_width = channel_width
        else:
            uvc.Nfreqs = 1
            if freq_range is None:
                if uvc.future_array_shapes:
                    params_to_copy.extend(["spw_array", "Nspws"])
                    if uvdata.flex_spw:
                        uvc.freq_range = np.zeros((uvdata.Nspws, 2), dtype=float)
                        for spw_ind, spw in enumerate(uvdata.spw_array):
                            if uvdata.future_array_shapes:
                                freqs_in_spw = uvdata.freq_array[
                                    np.nonzero(uvdata.flex_spw_id_array == spw)
                                ]
                            else:
                                freqs_in_spw = uvdata.freq_array[
                                    0, np.nonzero(uvdata.flex_spw_id_array == spw)
                                ]
                            uvc.freq_range[spw_ind, :] = np.asarray(
                                [np.min(freqs_in_spw), np.max(freqs_in_spw)]
                            )
                    else:
                        uvc.freq_range = np.asarray(
                            [[np.min(uvdata.freq_array), np.max(uvdata.freq_array)]]
                        )
                else:
                    uvc.Nspws = 1
                    uvc.spw_array = np.asarray([0])
                    uvc.freq_range = [
                        np.min(uvdata.freq_array),
                        np.max(uvdata.freq_array),
                    ]
            else:
                freq_range_use = np.asarray(freq_range)
                if future_array_shapes:
                    if freq_range_use.shape == (2,):
                        freq_range_use = freq_range_use[np.newaxis, :]
                    if freq_range_use.ndim != 2 or freq_range_use.shape[1] != 2:
                        raise ValueError(
                            "if future_array_shapes is True, freq_range must be an "
                            "array shaped like (Nspws, 2)."
                        )
                    uvc.freq_range = freq_range_use
                    uvc.Nspws = uvc.freq_range.shape[0]
                    if uvc.Nspws > 1:
                        if spw_array is None:
                            raise ValueError(
                                "An spw_array must be provided for delay or wide-band "
                                "cals if freq_range has multiple spectral windows"
                            )
                        uvc.spw_array = spw_array
                    else:
                        uvc.spw_array = np.asarray([0])
                else:
                    uvc.Nspws = 1
                    uvc.spw_array = np.asarray([0])
                    if freq_range_use.size == 2:
                        uvc.freq_range = np.squeeze(freq_range_use).tolist()
                    else:
                        raise ValueError(
                            "if future_array_shapes is False, freq_range must have "
                            "2 elements."
                        )

        for param_name in params_to_copy:
            setattr(uvc, param_name, getattr(uvdata, param_name))

        # sort the antenna information (the order in the UVData object may be strange)
        ant_order = np.argsort(uvc.antenna_numbers)
        uvc.antenna_numbers = uvc.antenna_numbers[ant_order]
        uvc.antenna_names = ((np.asarray(uvc.antenna_names))[ant_order]).tolist()
        uvc.antenna_positions = uvc.antenna_positions[ant_order, :]

        if times is None:
            # get all unique times
            uvc.time_array = np.unique(uvdata.time_array)
            uvdata_int_times = np.unique(uvdata.integration_time)
            if uvdata_int_times.size == 1:
                uvdata_int_times = uvdata_int_times[0]
                if uvc.future_array_shapes:
                    uvc.integration_time = np.full(
                        uvc.time_array.size, uvdata_int_times, dtype=np.float64
                    )
                else:
                    uvc.integration_time = uvdata_int_times
            else:
                raise ValueError(
                    "uvdata integration times vary. Please specify times and "
                    "integration_time"
                )
        else:
            uvc.time_array = times
            if integration_time is None:
                raise ValueError(
                    "integation_time must be provided if times is provided"
                )
            if future_array_shapes:
                if isinstance(integration_time, (np.ndarray, list)):
                    uvc.integration_time = np.asarray(integration_time)
                else:
                    uvc.integration_time = np.full(
                        uvc.time_array.size, integration_time, dtype=np.float64
                    )
            else:
                if isinstance(integration_time, (np.ndarray, list)):
                    raise ValueError(
                        "integration_time must be scalar if future_array_shapes is "
                        "False."
                    )
                uvc.integration_time = integration_time
        uvc.Ntimes = uvc.time_array.size
        uvc.set_lsts_from_time_array()

        if time_range is not None:
            uvc.time_range = time_range

        if jones is None:
            if np.all(uvdata.polarization_array < -4):
                if uvdata.Npols == 1 and uvdata.polarization_array[0] > -7:
                    # single pol data, make a single pol cal object
                    uvc.jones_array = uvdata.polarization_array
                else:
                    uvc.jones_array = np.array([-5, -6])
            elif np.all(uvdata.polarization_array < 0):
                if uvdata.Npols == 1 and uvdata.polarization_array[0] > -3:
                    # single pol data, make a single pol cal object
                    uvc.jones_array = uvdata.polarization_array
                else:
                    uvc.jones_array = np.array([-1, -2])
            else:
                raise ValueError(
                    "jones parameter is None and uvdata object is in "
                    "psuedo-stokes polarization. Please set jones."
                )
        else:
            uvc.jones_array = np.asarray(jones)
        uvc.Njones = uvc.jones_array.size

        uvc.ant_array = np.union1d(uvdata.ant_1_array, uvdata.ant_2_array)

        uvc.history = "Initialized from a UVData object with pyuvdata."
        if include_uvdata_history:
            uvc.history += " UVData history is: " + uvdata.history

        if not metadata_only:
            for param in uvc._required_data_params:
                uvparam = getattr(uvc, "_" + param)
                expected_type = uvparam.expected_type
                # all data like params on UVCal have expected types that are tuples.
                # since uvc is re-initialized at the start of this method, the user
                # can't affect this, so don't need handling for non-tuples
                dtype_use = expected_type[0]
                setattr(
                    uvc, param, np.zeros(uvparam.expected_shape(uvc), dtype=dtype_use),
                )

        uvc.check()

        return uvc

    def read_calfits(
        self,
        filename,
        read_data=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in data from calfits file(s).

        Parameters
        ----------
        filename : str or list of str
            The calfits file(s) to read from.
        read_data : bool
            Read in the gains or delays, quality arrays and flag arrays.
            If set to False, only the metadata will be read in. Setting read_data to
            False results in a metadata only object.
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters after reading in the file.

        """
        from . import calfits

        if isinstance(filename, (list, tuple)):
            self.read_calfits(
                filename[0],
                read_data=read_data,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            if len(filename) > 1:
                for f in filename[1:]:
                    uvcal2 = UVCal()
                    uvcal2.read_calfits(
                        f,
                        read_data=read_data,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                    )
                    self += uvcal2
                del uvcal2
        else:
            calfits_obj = calfits.CALFITS()
            calfits_obj.read_calfits(
                filename,
                read_data=read_data,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            self._convert_from_filetype(calfits_obj)
            del calfits_obj

    def read_fhd_cal(
        self,
        cal_file,
        obs_file,
        layout_file=None,
        settings_file=None,
        raw=True,
        read_data=True,
        extra_history=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
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
            a metadata only object. Note that if read_data is False, metadata is
            derived entirely from the obs_file, which may result in slightly different
            values than if it is derived from the cal file.
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

        """
        from . import fhd_cal

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
                raw=raw,
                read_data=read_data,
                extra_history=extra_history,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
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
                        raw=raw,
                        read_data=read_data,
                        extra_history=extra_history,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
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
                raw=raw,
                read_data=read_data,
                extra_history=extra_history,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            self._convert_from_filetype(fhd_cal_obj)
            del fhd_cal_obj

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
