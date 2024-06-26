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

from .. import Telescope
from .. import parameter as uvp
from .. import utils
from ..docstrings import combine_docstrings, copy_replace_short_description
from ..uvbase import UVBase
from . import initializers

__all__ = ["UVCal"]


def _time_param_check(this, other):
    """Check if any time parameter is defined in this but not in other or vice-versa."""
    if not isinstance(other, list):
        other = [other]
    time_params = ["time_array", "time_range"]
    for param in time_params:
        if getattr(this, param) is not None:
            for obj in other:
                if getattr(obj, param) is None:
                    raise ValueError(
                        f"Some objects have a {param} while others do not. All "
                        f"objects must either have or not have a {param}."
                    )
        else:
            for obj in other:
                if getattr(obj, param) is not None:
                    raise ValueError(
                        f"Some objects have a {param} while others do not. All "
                        f"objects must either have or not have a {param}."
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

        self._telescope = uvp.UVParameter(
            "telescope",
            description=(
                ":class:`pyuvdata.Telescope` object containing the telescope "
                "metadata."
            ),
            expected_type=Telescope,
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
            "Array of integer antenna numbers that appear in self.gain_array,"
            " with shape (Nants_data,). "
            "This array is ordered to match the inherent ordering of the zeroth"
            " axis of self.gain_array."
        )
        self._ant_array = uvp.UVParameter(
            "ant_array", description=desc, expected_type=int, form=("Nants_data",)
        )

        desc = (
            "Option to support 'wide-band' calibration solutions with gains or delays "
            "that apply over a range of frequencies rather than having distinct values "
            "at each frequency. Delay type cal solutions are always 'wide-band'. If it "
            "is True several other parameters are affected: the data-like arrays have "
            "a spw axis that is Nspws long rather than a frequency axis that is Nfreqs "
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
            "Array of frequencies, center of the channel, shape (Nfreqs,) , units Hz."
            "Should not be set if wide_band=True."
        )
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,  # mHz
        )

        desc = (
            "Width of frequency channels (Hz). Array of shape (Nfreqs,), type = float."
            "Should not be set if wide_band=True."
        )
        self._channel_width = uvp.UVParameter(
            "channel_width",
            description=desc,
            expected_type=float,
            form=("Nfreqs",),
            tols=1e-3,  # 1 mHz
        )

        desc = (
            "Required if cal_type='delay' or wide_band=True. Frequency range that "
            "solutions are valid for, with [start_frequency, end_frequency] provided "
            "for each spectral window. Array of shape (Nspws, 2). Units are Hz."
            "Should not be set if cal_type='gain' and wide_band=False."
        )
        self._freq_range = uvp.UVParameter(
            "freq_range",
            required=False,
            description=desc,
            form=("Nspws", 2),
            expected_type=float,
            tols=1e-3,
        )

        desc = (
            "Array of antenna polarization integers, shape (Njones). "
            "linear pols -5:-8 (jxx, jyy, jxy, jyx)."
            "circular pols -1:-4 (jrr, jll. jrl, jlr)."
            "unknown 0."
        )

        self._jones_array = uvp.UVParameter(
            "jones_array",
            description=desc,
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 1)),
            form=("Njones",),
        )

        desc = (
            "Array of calibration solution times, center of integration, shape "
            "(Ntimes), units Julian Date. Should only be set cal solutions were "
            "calculated per integration (rather than over a range of integrations). "
            "Only one of time_range and time_array should be set."
        )
        self._time_array = uvp.UVParameter(
            "time_array",
            description=desc,
            form=("Ntimes",),
            expected_type=float,
            tols=1e-3 / (60.0 * 60.0 * 24.0),
            required=False,
        )

        desc = (
            "Time range (in JD) that cal solutions are valid for. This should be an "
            "array with shape (Ntimes, 2) where the second axis gives the start_time "
            "and end_time (in that order) in JD. Should only be set if the cal "
            "solutions apply over a range of times. Only one of time_range and "
            "time_array should be set."
        )
        self._time_range = uvp.UVParameter(
            "time_range",
            description=desc,
            form=("Ntimes", 2),
            expected_type=float,
            tols=1e-3 / (60.0 * 60.0 * 24.0),
            required=False,
        )

        desc = (
            "Array of lsts, center of integration, shape (Ntimes), units radians. "
            "Should only be set cal solutions were calculated per integration (rather "
            "than over a range of integrations). Only one of lst_range and lst_array "
            "should be set."
        )
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Ntimes",),
            expected_type=float,
            tols=utils.RADIAN_TOL,
            required=False,
        )

        desc = (
            "LST range (in JD) that cal solutions are valid for. This should be an "
            "array with shape (Ntimes, 2) where the second axis gives the start_lst "
            "and end_lst (in that order) in radians. Should only be set if the cal "
            "solutions apply over a range of times. Only one of lst_range and "
            "lst_array should be set."
        )
        self._lst_range = uvp.UVParameter(
            "lst_range",
            description=desc,
            form=("Ntimes", 2),
            expected_type=float,
            tols=utils.RADIAN_TOL,
            required=False,
        )

        desc = (
            "Integration time of a time bin, units seconds. Shape (Ntimes), type float."
        )
        self._integration_time = uvp.UVParameter(
            "integration_time",
            description=desc,
            expected_type=float,
            form=("Ntimes",),
            tols=1e-3,  # 1ms
        )

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
            "Shape: (Nants_data, Nfreqs, Ntimes, Njones) if wide_band=False or "
            "(Nants_data, Nspws, Ntimes, Njones) if wide_band=True, type = bool."
        )
        self._flag_array = uvp.UVParameter(
            "flag_array",
            description=desc,
            form=("Nants_data", "Nfreqs", "Ntimes", "Njones"),
            expected_type=bool,
        )

        desc = (
            "Array of qualities of calibration solutions. "
            "The shape depends on cal_type, if the cal_type is 'gain', the shape is: "
            "(Nants_data, Nfreqs, Ntimes, Njones) if ide_band=False or "
            "(Nants_data, Nspws, Ntimes, Njones) if wide_band=True, "
            "if the cal_type is 'delay', the shape is "
            "(Nants_data, Nspws, Ntimes, Njones). The type is float."
        )
        self._quality_array = uvp.UVParameter(
            "quality_array",
            description=desc,
            form=("Nants_data", "Nfreqs", "Ntimes", "Njones"),
            expected_type=float,
            required=False,
        )

        desc = (
            "Orientation of the physical dipole corresponding to what is "
            'labelled as the x polarization. Options are "east" '
            '(indicating east/west orientation) and "north" (indicating '
            "north/south orientation)"
        )

        # --- cal_type parameters ---
        desc = "cal type parameter. Values are delay or gain."
        self._cal_type = uvp.UVParameter(
            "cal_type",
            form="str",
            expected_type=str,
            value="gain",
            description=desc,
            acceptable_vals=["delay", "gain"],
        )

        desc = (
            'Required if cal_type = "gain". Array of gains, '
            "shape:  (Nants_data, Nfreqs, Ntimes, Njones) if wide_band=False, or "
            "(Nants_data, Nspws, Ntimes, Njones) if wide_band=True, "
            "type = complex float."
        )
        self._gain_array = uvp.UVParameter(
            "gain_array",
            description=desc,
            required=False,
            form=("Nants_data", "Nfreqs", "Ntimes", "Njones"),
            expected_type=complex,
        )

        desc = (
            'Required if cal_type = "delay". Array of delays with units of seconds. '
            "Shape: (Nants_data, Nspws, Ntimes, Njones), type=float."
        )
        self._delay_array = uvp.UVParameter(
            "delay_array",
            description=desc,
            required=False,
            form=("Nants_data", "Nspws", "Ntimes", "Njones"),
            expected_type=float,
        )

        # --- flexible spectral window information ---

        desc = (
            "Required for non-wide-band objects. Maps individual channels along the "
            "frequency axis to individual spectral windows, as listed in the "
            "spw_array. Shape (Nfreqs), type = int."
        )
        self._flex_spw_id_array = uvp.UVParameter(
            "flex_spw_id_array", description=desc, form=("Nfreqs",), expected_type=int
        )

        desc = (
            "Optional parameter that allows for labeling individual spectral windows "
            "with different polarizations. If set, Njones must be set to 1 (i.e., only "
            "one Jones vector per spectral window allowed). Shape (Nspws), type = int."
        )
        self._flex_jones_array = uvp.UVParameter(
            "flex_jones_array",
            description=desc,
            form=("Nspws",),
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 1)),
            required=False,
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

        desc = (
            'Optional argument, only used if cal_style = "sky". Allows one to specify'
            "a different phase reference antenna per time interval. Shape (Ntimes,), "
            "type=int."
        )
        self._ref_antenna_array = uvp.UVParameter(
            "ref_antenna_array",
            description=desc,
            form=("Ntimes",),
            expected_type=int,
            required=False,
        )

        desc = "Number of sources used."
        self._Nsources = uvp.UVParameter(
            "Nsources", required=False, expected_type=int, description=desc
        )

        desc = (
            "Specifies the number of phase centers contained within the calibration "
            "solution catalog."
        )
        self._Nphase = uvp.UVParameter(
            "Nphase", required=False, expected_type=int, description=desc
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
            "the shape is: (Nfreqs, Ntimes, Njones) if wide_band=False, "
            "or (Nspws, Ntimes, Njones) if wide_band=True. If the cal_type is 'delay', "
            "the shape is (Nspws, Ntimes, Njones), type = float."
        )
        self._total_quality_array = uvp.UVParameter(
            "total_quality_array",
            description=desc,
            form=("Nfreqs", "Ntimes", "Njones"),
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

        desc = (
            "Optional when reading an MS cal table. Retains the scan number when "
            "reading a measurement set. Shape (Nblts), type = int."
        )
        self._scan_number_array = uvp.UVParameter(
            "scan_number_array",
            description=desc,
            form=("Ntimes",),
            expected_type=int,
            required=False,
        )

        desc = (
            "Optional parameter, similar to the UVData parameter of the same name. "
            "Dictionary that acts as a catalog, containing information on individual "
            "phase centers. Keys are the catalog IDs of the different phase centers in "
            "the object (matched to the parameter ``phase_center_id_array``). At a "
            "minimum, each dictionary must contain the keys: "
            "'cat_name' giving the phase center name (this does not have to be unique, "
            "non-unique values can be used to indicate sets of phase centers that make "
            "up a mosaic observation), "
            "'cat_type', which can be 'sidereal' (fixed position in RA/Dec), 'ephem' "
            "(position in RA/Dec which moves with time), 'driftscan' (fixed postion in "
            "Az/El, NOT the same as the old ``phase_type`` = 'drift') or 'unprojected' "
            "(baseline coordinates in ENU, but data are not phased, similar to "
            "the old ``phase_type`` = 'drift') "
            "'cat_lon' (longitude coord, e.g. RA, either a single value or a one "
            "dimensional array of length Npts --the number of ephemeris data points-- "
            "for ephem type phase centers), "
            "'cat_lat' (latitude coord, e.g. Dec., either a single value or a one "
            "dimensional array of length Npts --the number of ephemeris data points-- "
            "for ephem type phase centers), "
            "'cat_frame' (coordinate frame, e.g. icrs, must be a frame supported by "
            "astropy). "
            "Other optional keys include "
            "'cat_epoch' (epoch and equinox of the coordinate frame, not needed for "
            "frames without an epoch (e.g. ICRS) unless the there is proper motion), "
            "'cat_times' (times for the coordinates, only used for 'ephem' types), "
            "'cat_pm_ra' (proper motion in RA), "
            "'cat_pm_dec' (proper motion in Dec), "
            "'cat_dist' (physical distance to the source in parsec, useful if parallax "
            "is important, either a single value or a one dimensional array of length "
            "Npts --the number of ephemeris data points-- for ephem type phase "
            "centers.), "
            "'cat_vrad' (rest frame velocity in km/s, either a single value or a one "
            "dimensional array of length Npts --the number of ephemeris data points-- "
            "for ephem type phase centers.), and "
            "'info_source' (describes where catalog info came from). "
            "Most typically used with MS calibration tables."
        )

        self._phase_center_catalog = uvp.UVParameter(
            "phase_center_catalog", description=desc, expected_type=dict, required=False
        )

        desc = (
            "Optional parameter, similar to the UVData parameter of the same name. "
            "Maps individual indices along the Ntimes axis to a key in "
            "`phase_center_catalog`, which maps to a dict containing the other "
            "metadata for each phase center. Used to specify where the data were "
            "phased to when calibration tables were derived. Most typically used when "
            "reading MS calibration tables. Shape (Nblts), type = int."
        )
        self._phase_center_id_array = uvp.UVParameter(
            "phase_center_id_array",
            description=desc,
            form=("Ntimes",),
            expected_type=int,
            required=False,
        )

        super(UVCal, self).__init__()

        # Assign attributes to UVParameters after initialization, since UVBase.__init__
        # will link the properties to the underlying UVParameter.value attributes
        # initialize the telescope object
        self.telescope = Telescope()

        # set the appropriate telescope attributes as required
        self._set_telescope_requirements()

    def _set_telescope_requirements(self):
        """Set the UVParameter required fields appropriately for UVCal."""
        self.telescope._instrument.required = False
        self.telescope._x_orientation.required = True

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

    def _set_wide_band(self, wide_band=True):
        """
        Set the wide_band parameter and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by the file-reading methods to indicate that an object is a wide-band
        calibration solution which supports gain or delay values per spectral window.

        """
        if wide_band:
            assert self.flex_spw_id_array is None or np.array_equal(
                self.flex_spw_id_array, self.spw_array
            ), "flex_spw_id_array must be unset or equal to spw_array to set wide_band"
        else:
            assert self.cal_type != "delay", "delay objects cannot have wide_band=False"
        self.wide_band = wide_band

        if wide_band:
            self._freq_array.required = False
            self._channel_width.required = False
            self._flex_spw_id_array.required = False
            self._freq_range.required = True
            self.flex_spw_id_array = None

            data_shape_params = [
                "gain_array",
                "delay_array",
                "flag_array",
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
            self._flex_spw_id_array.required = True
            self._freq_range.required = False
            self._flex_spw_id_array.required = True

            # can only get here if not a delay solution
            data_shape_params = ["gain_array", "flag_array", "quality_array"]

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
        self._freq_range.required = self.wide_band
        self._freq_array.required = not self.wide_band
        self._channel_width.required = not self.wide_band
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
        self._set_wide_band()

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

    def remove_flex_jones(self, *, combine_spws=True):
        """
        Convert a flex-Jones UVCal object into one with a standard Jones axis.

        This will convert a flexible-Jones dataset into one with standard
        polarization handling, which is required for some operations or writing in
        certain filetypes. Note that depending on how it is used, this can inflate
        the size of data-like parameters by up to a factor of Nspws (the true value
        depends on the number of unique entries in `flex_jones_array`).

        Parameters
        ----------
        combine_spws : bool
            If set to True, the method will attempt to recombine multiple windows
            carrying different Jones vectors/information into a single (multi-Jones)
            spectral window. Functionally, this is the inverse of what is done in the
            `convert_to_flex_jones` method. If set to False, the method will effectively
            "inflate" the jones-axis of UVCal parameters such that all windows have the
            same Jones codes (though the added entries will be flagged and will carry no
            calibration information). Default is True.
        """
        if self.flex_jones_array is None:
            # There isn't anything to do, so just move along
            return

        jones_array = list(reversed(np.unique(self.flex_jones_array)))
        n_jones = len(jones_array)

        if n_jones == 1 and (self.Nspws == 1 or not combine_spws):
            # Just remove the flex_jones_array and fix the polarization array
            self.jones_array = np.array(jones_array)
            self.flex_jones_array = None
            return

        if combine_spws:
            # check to see if there are spectral windows that have matching freq_array
            # and channel_width (up to sorting). If so, they need to be combined.
            if self.wide_band:
                freq_array_check = self.freq_range
                chan_width_check = np.zeros_like(freq_array_check)
                index_check = self.spw_array
                ftol = self._freq_range.tols
                ctol = [0, 0]
            else:
                freq_array_check = self.freq_array
                chan_width_check = self.channel_width

                index_check = self.flex_spw_id_array
                ftol = self._freq_array.tols
                ctol = self._channel_width.tols

            # Build the main spw map, along with some dicts to save intermediate results
            # from frequencies and Jones vectors
            spw_map = {}
            spw_freqs = {}
            spw_jones_tally = {}
            for jones1, spw1 in zip(self.flex_jones_array, self.spw_array):
                freqs1 = freq_array_check[index_check == spw1]
                chwidth1 = chan_width_check[index_check == spw1]
                # Default to spw1 at the start, so if no match it will correctly link
                # back to itself.
                spw_match = spw1
                for spw2, (freqs2, chwidth2) in spw_freqs.items():
                    if np.allclose(freqs1, freqs2, rtol=ftol[0], atol=ftol[1]):
                        if np.allclose(chwidth1, chwidth2, rtol=ctol[0], atol=ctol[1]):
                            spw_match = spw2
                            break
                # If spw is its own match, save freqs and make an entry in jones dict
                if spw1 == spw_match:
                    spw_freqs[spw1] = (freqs1, chwidth1)
                    spw_jones_tally[spw1] = []

                # Save matching spw, record which Jones vector this window has
                spw_map[spw1] = spw_match
                spw_jones_tally[spw_match].append(jones1)

            # Once all windows are processed, look for failures
            for jones_subarr in spw_jones_tally.values():
                if len(np.unique(jones_subarr)) != len(jones_subarr):
                    raise ValueError(
                        "Some spectral windows have identical frequencies, "
                        "channel widths and polarizations, so spws cannot be "
                        "combined. Set combine_spws=False to avoid this error."
                    )
                elif len(np.unique(jones_subarr)) != len(jones_array):
                    warnings.warn(
                        "combine_spws is True but there are not matched spws for all "
                        "polarizations, so spws will not be combined."
                    )
                    combine_spws = False
                    break

        if not combine_spws:
            # If not combining spws, then map everything back to itself
            index_check = self.spw_array if self.wide_band else self.flex_spw_id_array
            spw_map = dict(zip(self.spw_array, self.spw_array))

        # Reverse mapping map for various metadata values
        rev_map = {value: key for key, value in spw_map.items()}
        spw_array = sorted(rev_map)
        if self.flex_spw_id_array is not None:
            freq_array = np.concatenate(
                [self.freq_array[self.flex_spw_id_array == spw] for spw in spw_array]
            )
            chan_width = np.concatenate(
                [self.channel_width[self.flex_spw_id_array == spw] for spw in spw_array]
            )
            flex_spw_id_array = np.concatenate(
                [
                    self.flex_spw_id_array[self.flex_spw_id_array == spw]
                    for spw in spw_array
                ]
            )
            freq_range = None
            n_freqs = len(flex_spw_id_array)
        else:
            old_spw_arr = list(self.spw_array)
            freq_array = chan_width = None
            flex_spw_id_array = None
            freq_range = np.concatenate(
                [self.freq_range[old_spw_arr.index(spw)] for spw in spw_array]
            ).reshape((-1, 2))
            n_freqs = 1

        # Update metadata attributes
        self.spw_array = np.array(spw_array)
        self.flex_spw_id_array = flex_spw_id_array
        self.freq_array = freq_array
        self.channel_width = chan_width
        self.freq_range = freq_range
        self.jones_array = np.array(jones_array)

        # Adjust the length-related attributes
        self.Nspws = len(spw_array)
        self.Nfreqs = n_freqs
        self.Njones = n_jones

        # Finally, update data attrs if they exist
        if not self.metadata_only:
            freq_index = self.spw_array if self.wide_band else self.flex_spw_id_array
            for name, param in zip(self._data_params, self.data_like_parameters):
                if param is None:
                    continue
                # Update the parameter shape with
                new_shape = list(param.shape)
                new_shape[1] = self.Nspws if self.wide_band else self.Nfreqs
                new_shape[3] = self.Njones
                new_param = np.full(new_shape, name == "flag_array", dtype=param.dtype)
                for idx, old_spw in enumerate(spw_map):
                    new_fidx = freq_index == spw_map[old_spw]
                    # index_check is recycled from the original combine_windows check
                    # above (and marks the original indexing positions of the object)
                    old_fidx = index_check == old_spw
                    jones_idx = jones_array.index(self.flex_jones_array[idx])
                    new_param[:, new_fidx, :, jones_idx] = param[:, old_fidx, :, 0]
                setattr(self, name, new_param)

        # Finally, drop flex_jones now that we no longer need it's info
        self.flex_jones_array = None

    def _make_flex_jones(self):
        """
        Convert a regular UVCal object into one with flex-Jones enabled.

        This is an internal helper function, which is not designed to be called by
        users, but rather individual read/write functions for the UVCal object.
        This will convert a regular UVCal object into one that uses flexible
        Jones arrays, which allows for each spectral window to have its own unique
        Jones vector/code, useful for storing data more compactly when different
        windows have different Jones values recorded. Note that at this time,
        only one Jones code per-spw is allowed -- if more than one Jones code
        is found to have unflagged data in a given spectral window, then an error is
        returned.
        """
        if self.metadata_only:
            raise ValueError(
                "Cannot make a metadata_only UVCal object flex-Jones because flagging "
                "info is required. Consider using `convert_to_flex_jones` instead, but "
                "be aware that the behavior is somewhat different"
            )

        if self.Njones == 1:
            # This is basically a no-op, fix the relevant attributes and exit
            if self.flex_jones_array is None:
                self.flex_jones_array = np.full(self.Nspws, self.jones_array[0])
                self.jones_array = np.array([0])
            return

        jones_idx_arr = np.full(self.Nspws, -1)
        spw_id_arr = self.spw_array if self.wide_band else self.flex_spw_id_array
        for idx, spw in enumerate(self.spw_array):
            spw_screen = spw_id_arr == spw

            # For each window, we want to check that there is only one polarization with
            # any unflagged data, which we can do by seeing if not all of the flags
            # are set across the non-polarization axes (hence the ~np.all()).
            jones_check = ~np.all(self.flag_array[:, spw_screen], axis=(0, 1, 2))

            if sum(jones_check) > 1:
                raise ValueError(
                    "Cannot make a flex-pol UVCal object, as some windows have "
                    "unflagged data in multiple Jones codes."
                )
            elif np.any(jones_check):
                jones_idx_arr[idx] = np.where(jones_check)[0][0]

        # If one window was all flagged out, but the others all belong to the same pol,
        # assume we just want that Jones code.
        if len(np.unique(jones_idx_arr[jones_idx_arr >= 0])) == 1:
            jones_idx_arr[:] = np.unique(jones_idx_arr[jones_idx_arr >= 0])

        # Now that we have Jones values sorted out, update metadata attributes
        self.flex_jones_array = self.jones_array[jones_idx_arr]
        self.jones_array = np.array([0])
        self.Njones = 1

        # Now go through one-by-one with data-like parameters and update
        for name, param in zip(self._data_params, self.data_like_parameters):
            # Grab the shape and update the Jones (last) axis to be length 1. Note we
            # do it this way since total_quality_array has a different shape
            if param is None:
                continue
            new_shape = list(param.shape)
            new_shape[-1] = 1

            # We can use empty here, since we know that we will be filling all values
            new_param = np.empty(new_shape, dtype=param.dtype)

            for spw, jones_idx in zip(self.spw_array, jones_idx_arr):
                # Process each window individually, since jones code can vary
                spw_screen = spw_id_arr == spw

                # Grab from the right Jones-position and plug values in
                new_param[:, spw_screen, :, 0] = param[:, spw_screen, :, jones_idx]

            # Update the attribute with the new values
            setattr(self, name, new_param)

    def convert_to_flex_jones(self):
        """
        Convert a regular UVCal object into a flex-Jones object.

        This effectively combines the frequency and polarization axis with polarization
        changing slowest. Saving data to uvh5 files this way can speed up some kinds
        of data access.

        """
        if self.flex_jones_array is not None:
            raise ValueError("This is already a flex-pol object")

        if self.Njones == 1:
            # This is basically a no-op, fix the relevant attributes and exit
            self.flex_jones_array = np.full(self.Nspws, self.jones_array[0])
            self.jones_array = np.array([0])
            return

        old_nspws = self.Nspws * self.Njones
        njones = self.Njones
        new_nspws = self.Nspws * (self.Njones - 1)
        new_spw_ids = list(set(range(1, old_nspws + 1)).difference(self.spw_array))
        spw_array = np.concatenate((self.spw_array, new_spw_ids[:new_nspws]))
        flex_spw_id_arr = None

        if not self.wide_band:
            spw_reshape = spw_array.reshape((self.Njones, -1))
            flex_spw_id_arr = np.repeat(
                self.flex_spw_id_array[np.newaxis], self.Njones, axis=0
            )
            for idx in range(1, self.Njones):
                spw_map = dict(zip(spw_reshape[0], spw_reshape[idx]))
                flex_spw_id_arr[idx] = [spw_map[jdx] for jdx in flex_spw_id_arr[idx]]
            flex_spw_id_arr = flex_spw_id_arr.flatten()

        self.flex_jones_array = np.repeat(self.jones_array, self.Nspws, axis=0)
        self.jones_array = np.array([0])

        # Update metadata attributes
        self.spw_array = spw_array
        self.flex_spw_id_array = flex_spw_id_arr

        # If defined, set the frequency-related attributes
        if self.freq_array is not None:
            self.freq_array = np.tile(self.freq_array, njones)
        if self.freq_range is not None:
            self.freq_range = np.tile(self.freq_range, (njones, 1))
        if self.channel_width is not None:
            self.channel_width = np.tile(self.channel_width, njones)

        # Adjust the length-related attributes
        self.Nspws *= self.Njones
        self.Nfreqs *= 1 if self.wide_band else self.Njones
        self.Njones = 1

        # Finally, if we have it, update the metadata.
        if not self.metadata_only:
            for name, param in zip(self._data_params, self.data_like_parameters):
                if param is None:
                    continue
                # Make into a list so that its mutable
                old_shape = param.shape
                # ("Nants_data", "Nfreqs", "Ntimes", "Njones"),
                # Extend the second axis by njones, shrink the last axis to 1
                new_shape = [old_shape[0], old_shape[1] * njones, old_shape[2], 1]
                param = np.transpose(param, (0, 3, 1, 2)).reshape(new_shape)
                setattr(self, name, param)

    def set_telescope_params(
        self,
        *,
        warn=True,
        overwrite=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
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
        self.telescope.update_params_from_known_telescopes(
            overwrite=overwrite,
            warn=warn,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

    def _set_lsts_helper(self, *, astrometry_library=None):
        if self.time_array is not None:
            self.lst_array = utils.get_lst_for_time(
                jd_array=self.time_array, telescope_loc=self.telescope.location
            )

        if self.time_range is not None:
            self.lst_range = utils.get_lst_for_time(
                jd_array=self.time_range, telescope_loc=self.telescope.location
            )
        return

    def set_lsts_from_time_array(self, *, background=False, astrometry_library=None):
        """Set the lst_array or lst_range from the time_array or time_range.

        Parameters
        ----------
        background : bool, False
            When set to True, start the calculation on a threading.Thread in the
            background and return the thread to the user.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

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
        Check if the spectral windows are contiguous.

        This checks the flex_spw_id_array to make sure that all channels for each
        spectral window are together in one block, versus being interspersed (e.g.,
        channel #1 and #3 is in spw #1, channels #2 and #4 are in spw #2).

        """
        if not self.wide_band:
            utils.frequency._check_flex_spw_contiguous(
                spw_array=self.spw_array, flex_spw_id_array=self.flex_spw_id_array
            )

    def _check_freq_spacing(self, *, raise_errors=True):
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
        if (self.freq_array is None) or (self.Nfreqs == 1):
            return False, False
        return utils.frequency._check_freq_spacing(
            freq_array=self.freq_array,
            freq_tols=self._freq_array.tols,
            channel_width=self.channel_width,
            channel_width_tols=self._channel_width.tols,
            spw_array=self.spw_array,
            flex_spw_id_array=self.flex_spw_id_array,
            raise_errors=raise_errors,
        )

    def _add_phase_center(
        self,
        cat_name,
        *,
        cat_type=None,
        cat_lon=None,
        cat_lat=None,
        cat_frame=None,
        cat_epoch=None,
        cat_times=None,
        cat_pm_ra=None,
        cat_pm_dec=None,
        cat_dist=None,
        cat_vrad=None,
        info_source="user",
        force_update=False,
        cat_id=None,
    ):
        """
        Add an entry to the internal object/source catalog or find a matching one.

        This is a helper function for identifying a adding a phase center to the
        internal catalog, contained within the attribute `phase_center_catalog`, unless
        a phase center already exists that matches the passed parameters. If a matching
        phase center is found, the catalog ID associated with that phase center is
        returned.

        Parameters
        ----------
        cat_name : str
            Name of the phase center to be added.
        cat_type : str
            Type of phase center to be added. Must be one of:
                "sidereal" (fixed RA/Dec),
                "ephem" (RA/Dec that moves with time),
                "driftscan" (fixed az/el position),
                "unprojected" (no w-projection, equivalent to the old
                `phase_type` == "drift").
        cat_lon : float or ndarray
            Value of the longitudinal coordinate (e.g., RA, Az, l) in radians of the
            phase center. No default unless `cat_type="unprojected"`, in which case the
            default is zero. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        cat_lat : float or ndarray
            Value of the latitudinal coordinate (e.g., Dec, El, b) in radians of the
            phase center. No default unless `cat_type="unprojected"`, in which case the
            default is pi/2. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        cat_frame : str
            Coordinate frame that cat_lon and cat_lat are given in. Only used
            for sidereal and ephem targets. Can be any of the several supported frames
            in astropy (a limited list: fk4, fk5, icrs, gcrs, cirs, galactic).
        cat_epoch : str or float
            Epoch of the coordinates, only used when cat_frame = fk4 or fk5. Given
            in units of fractional years, either as a float or as a string with the
            epoch abbreviation (e.g, Julian epoch 2000.0 would be J2000.0).
        cat_times : ndarray of floats
            Only used when `cat_type="ephem"`. Describes the time for which the values
            of `cat_lon` and `cat_lat` are calculated, in units of JD. Shape is (Npts,).
        cat_pm_ra : float
            Proper motion in RA, in units of mas/year. Only used for sidereal phase
            centers.
        cat_pm_dec : float
            Proper motion in Dec, in units of mas/year. Only used for sidereal phase
            centers.
        cat_dist : float or ndarray of float
            Distance of the source, in units of pc. Only used for sidereal and ephem
            phase centers. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        cat_vrad : float or ndarray of float
            Radial velocity of the source, in units of km/s. Only used for sidereal and
            ephem phase centers. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        info_source : str
            Optional string describing the source of the information provided. Used
            primarily in UVCal to denote when an ephemeris has been supplied by the
            JPL-Horizons system, user-supplied, or read in by one of the various file
            interpreters. Default is 'user'.
        force_update : bool
            Normally, `_add_phase_center` will throw an error if there already exists a
            phase_center with the given cat_id. However, if one sets
            `force_update=True`, the method will overwrite the existing entry in
            `phase_center_catalog` with the parameters supplied. Note that doing this
            will _not_ update other attributes of the `UVCal` object. Default is False.
        cat_id : int
            An integer signifying the ID number for the phase center, used in the
            `phase_center_id_array` attribute. If a matching phase center entry exists
            already, that phase center ID will be returned, which may be different than
            the value specified to this parameter. The default is for the method to
            assign this value automatically.

        Returns
        -------
        cat_id : int
            The unique ID number for the phase center that either matches the specified
            parameters or was added to the internal catalog. If a matching entry was
            found, this may not be the value passed to the `cat_id` parameter. This
            value is used in the `phase_center_id_array` attribute to denote which
            source a given baseline-time corresponds to.

        Raises
        ------
        ValueError
            If attempting to add a non-unique source name or if adding a sidereal
            source without coordinates.

        """
        cat_entry = utils.phase_center_catalog.generate_phase_center_cat_entry(
            cat_name=cat_name,
            cat_type=cat_type,
            cat_lon=cat_lon,
            cat_lat=cat_lat,
            cat_frame=cat_frame,
            cat_epoch=cat_epoch,
            cat_times=cat_times,
            cat_pm_ra=cat_pm_ra,
            cat_pm_dec=cat_pm_dec,
            cat_dist=cat_dist,
            cat_vrad=cat_vrad,
            info_source=info_source,
        )

        # We want to create a unique ID for each source, for use in indexing arrays.
        # The logic below ensures that we pick the lowest positive integer that is
        # not currently being used by another source
        if cat_id is None or not force_update:
            cat_id = utils.phase_center_catalog.generate_new_phase_center_id(
                phase_center_catalog=self.phase_center_catalog, cat_id=cat_id
            )

        if self.phase_center_catalog is None:
            # Initialize an empty dict to plug entries into
            self.phase_center_catalog = {}
        else:
            # Let's warn if this entry has the same name as an existing one
            temp_id, cat_diffs = utils.phase_center_catalog.look_in_catalog(
                self.phase_center_catalog, phase_dict=cat_entry
            )

            # If the source does have the same name, check to see if all the
            # attributes match. If so, no problem, go about your business
            if temp_id is not None:
                if cat_diffs == 0:
                    # Everything matches, return the catalog ID of the matching entry
                    return temp_id
                warnings.warn(
                    f"The provided name {cat_name} is already used but has different "
                    "parameters. Adding another entry with the same name but a "
                    "different ID and parameters."
                )

        # If source is unique, begin creating a dictionary for it
        self.phase_center_catalog[cat_id] = cat_entry
        self.Nphase = len(self.phase_center_catalog)

        return cat_id

    def _remove_phase_center(self, defunct_id):
        """
        Remove an entry from the internal object/source catalog.

        Removes an entry from the attribute `phase_center_catalog`.

        Parameters
        ----------
        defunct_id : int
            Catalog ID of the source to be removed

        Raises
        ------
        IndexError
            If the name provided is not found as a key in `phase_center_catalog`
        """
        if defunct_id not in self.phase_center_catalog:
            raise IndexError("No source by that ID contained in the catalog.")

        del self.phase_center_catalog[defunct_id]
        self.Nphase = len(self.phase_center_catalog)

    def _clear_unused_phase_centers(self):
        """
        Remove objects dictionaries and names that are no longer in use.

        Goes through the `phase_center_catalog` attribute in of a UVCal object and
        clears out entries that are no longer being used, and appropriately updates
        `phase_center_id_array` accordingly. This function is not typically called
        by users, but instead is used by other methods.

        """
        unique_cat_ids = np.unique(self.phase_center_id_array)
        defunct_list = []
        Nphase = 0
        for cat_id in self.phase_center_catalog:
            if cat_id in unique_cat_ids:
                Nphase += 1
            else:
                defunct_list.append(cat_id)

        # Check the number of "good" sources we have -- if we haven't dropped any,
        # then we are free to bail, otherwise update the Nphase attribute
        if Nphase == self.Nphase:
            return

        # Time to kill the entries that are no longer in the source stack
        for defunct_id in defunct_list:
            self._remove_phase_center(defunct_id)

    def print_phase_center_info(
        self,
        catalog_identifier=None,
        *,
        hms_format=None,
        return_str=False,
        print_table=True,
    ):
        """
        Print out the details of the phase centers.

        Prints out an ASCII table that contains the details of the
        `phase_center_catalog` attribute, which acts as the internal source catalog
        for UVCal objects.

        Parameters
        ----------
        catalog_identifier : str or int or list of str or int
            Optional parameter which, if provided, will cause the method to only return
            information on the phase center(s) with the matching name(s) or catalog ID
            number(s). Default is to print out information on all catalog entries.
        hms_format : bool
            Optional parameter, which if selected, can be used to force coordinates to
            be printed out in Hours-Min-Sec (if set to True) or Deg-Min-Sec (if set to
            False) format. Default is to print out in HMS if all the objects have
            coordinate frames of icrs, gcrs, fk5, fk4, and top; otherwise, DMS format
            is used.
        return_str: bool
            If set to True, the method returns an ASCII string which contains all the
            table infrmation. Default is False.
        print_table : bool
            If set to True, prints the table to the terminal window. Default is True.

        Returns
        -------
        table_str : bool
            If return_str=True, an ASCII string containing the entire table text

        Raises
        ------
        ValueError
            If `cat_name` matches no keys in `phase_center_catalog`.
        """
        return utils.phase_center_catalog.print_phase_center_info(
            self.phase_center_catalog,
            catalog_identifier=catalog_identifier,
            hms_format=hms_format,
            return_str=return_str,
            print_table=print_table,
        )

    def _update_phase_center_id(self, cat_id, *, new_id=None, reserved_ids=None):
        """
        Update a phase center with a new catalog ID number.

        Parameters
        ----------
        cat_id : int
            Current catalog ID of the phase center, which corresponds to a key in the
            attribute `phase_center_catalog`.
        new_id : int
            Optional argument. If supplied, then the method will attempt to use the
            provided value as the new catalog ID, provided that an existing catalog
            entry is not already using the same value. If not supplied, then the
            method will automatically assign a value.
        reserved_ids : array-like in int
            Optional argument. An array-like of ints that denotes which ID numbers
            are already reserved. Useful for when combining two separate catalogs.

        Raises
        ------
        ValueError
            If not using the method on a multi-phase-ctr data set, if there's no entry
            that matches `cat_name`, or of the value `new_id` is already taken.
        """
        new_id = utils.phase_center_catalog.generate_new_phase_center_id(
            phase_center_catalog=self.phase_center_catalog,
            cat_id=new_id,
            old_id=cat_id,
            reserved_ids=reserved_ids,
        )

        # If new_id is None, it means that the existing ID was fine and needs no update
        if new_id is not None:
            self.phase_center_id_array[self.phase_center_id_array == cat_id] = new_id
            self.phase_center_catalog[new_id] = self.phase_center_catalog.pop(cat_id)

    def _consolidate_phase_center_catalogs(
        self, *, reference_catalog=None, other=None, ignore_name=False
    ):
        """
        Consolidate phase center catalogs with a reference or another object.

        This is a helper method which updates the phase_center_catalog and related
        parameters to make this object consistent with a reference catalog or with
        another object so the second object can be added or concatenated to this object.
        If both `reference_catalog` and `other` are provided, both this object and the
        one passed to `other` will have their catalogs updated.

        Parameters
        ----------
        reference_catalog : dict
            A reference catalog to make this object consistent with.
        other : UVCal or UVData object
            Nominally a UVCal object which self needs to be consistent with because it
            will be added to self. The phase_center_catalog from other is used as the
            reference catalog if the reference_catalog is None. If `reference_catalog`
            is also set, the phase_center_catalog on other will also be modified to be
            consistent with the `reference_catalog`. Note that a UVData object can also
            be supplied (in order to ensure consistency between two related objects).
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when identifying matching phase centers. If set to
            True, phase centers that are the same up to their name will be combined with
            the name set to the reference catalog name or the name found in the first
            UVCal object. If set to False, phase centers that are the same up to the
            name will be kept as separate phase centers. Default is False.

        """
        if reference_catalog is None and other is None:
            raise ValueError(
                "Either the reference_catalog or the other parameter must be set."
            )

        if other is not None:
            # If exists, first update other to be consistent with the reference
            if reference_catalog is not None:
                other._consolidate_phase_center_catalogs(
                    reference_catalog=reference_catalog
                )
            # then use the updated other as the reference
            reference_catalog = other.phase_center_catalog

        reserved_ids = list(reference_catalog)
        # First loop, we want to update all the catalog IDs so that we know there
        # are no conflicts with the reference
        for cat_id in list(self.phase_center_catalog):
            self._update_phase_center_id(cat_id, reserved_ids=reserved_ids)

        # Next loop, we want to update the IDs of sources that are in the reference.
        for cat_id in list(reference_catalog):
            # Normally one would wrap this in an items() call above, except that for
            # testing it's sometimes convenient to use self.phase_center_catalog as
            # the ref catalog, which causes a RunTime error due to updates to the dict.
            cat_entry = reference_catalog[cat_id]
            match_id, match_diffs = utils.phase_center_catalog.look_in_catalog(
                self.phase_center_catalog, phase_dict=cat_entry, ignore_name=ignore_name
            )
            if match_id is None or match_diffs != 0:
                # If no match, just add the entry
                self._add_phase_center(cat_id=cat_id, **cat_entry)
                continue

            # If match_diffs is 0 then all the keys in the phase center catalog
            # match, so this is functionally the same source
            self._update_phase_center_id(match_id, new_id=cat_id)
            # look_in_catalog ignores the "info_source" field, so update it to match
            self.phase_center_catalog[cat_id]["info_source"] = cat_entry["info_source"]
            if ignore_name:
                # Make the names match if names were ignored in matching
                self.phase_center_catalog[cat_id]["cat_name"] = cat_entry["cat_name"]

    def check(
        self,
        *,
        check_extra=True,
        run_check_acceptability=True,
        check_freq_spacing=False,
        lst_tol=utils.LST_RAD_TOL,
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

        self._set_telescope_requirements()

        # if wide_band is True, Nfreqs must be 1.
        if self.wide_band:
            if self.Nfreqs != 1:
                raise ValueError("Nfreqs is required to be 1 for wide_band cals.")

        # call metadata_only to make sure that parameter requirements are set properly
        self.metadata_only

        # first run the basic check from UVBase
        super(UVCal, self).check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # then run telescope object check
        self.telescope.check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # deprecate having both time_array and time_range set
        time_like_pairs = [("time_array", "time_range"), ("lst_array", "lst_range")]
        for pair in time_like_pairs:
            if (
                getattr(self, pair[0]) is not None
                and getattr(self, pair[1]) is not None
            ):
                raise ValueError(
                    f"The {pair[0]} and {pair[1]} attributes are both set, but only "
                    "one should be set."
                )
            elif getattr(self, pair[0]) is None and getattr(self, pair[1]) is None:
                raise ValueError(f"Either {pair[0]} or {pair[1]} must be set.")

        # check that corresponding lst/time parameters are set
        for tp_ind, param in enumerate(time_like_pairs[0]):
            lst_param = time_like_pairs[1][tp_ind]
            if getattr(self, param) is not None and getattr(self, lst_param) is None:
                raise ValueError(
                    f"If {param} is present, {lst_param} must also be present."
                )

        # check that time ranges are well formed and do not overlap
        if self.time_range is not None:
            if utils.tools._check_range_overlap(self.time_range):
                raise ValueError("Some time_ranges overlap.")
            # note: do not check lst range overlap because of branch cut.
            # Assume they are ok if time_ranges are ok.

        # require that all entries in ant_array exist in antenna_numbers
        if not all(ant in self.telescope.antenna_numbers for ant in self.ant_array):
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

        if not self.wide_band:
            if not np.all(np.isin(self.flex_spw_id_array, self.spw_array)):
                raise ValueError(
                    "All values in the flex_spw_id_array must exist in the spw_array."
                )
            # warn if freq_range or freq_array set when it shouldn't be
            if self.freq_range is not None:
                raise ValueError(
                    "The freq_range attribute should not be set if wide_band=False."
                )
        if self.wide_band:
            if self.freq_array is not None:
                raise ValueError(
                    "The freq_array attribute should not be set if wide_band=True."
                )
            if self.channel_width is not None:
                raise ValueError(
                    "The channel_width attribute should not be set if wide_band=True."
                )
            if self.flex_spw_id_array is not None:
                raise ValueError(
                    "The flex_spw_id_array attribute should not be set if "
                    "wide_band=True."
                )

        if check_freq_spacing:
            self._check_freq_spacing()

        if run_check_acceptability:
            # Check antenna positions
            utils.coordinates.check_surface_based_positions(
                antenna_positions=self.telescope.antenna_positions,
                telescope_loc=self.telescope.location,
                raise_error=False,
            )

            if self.time_array is not None:
                utils.times.check_lsts_against_times(
                    jd_array=self.time_array,
                    lst_array=self.lst_array,
                    telescope_loc=self.telescope.location,
                    lst_tols=self._lst_array.tols if lst_tol is None else [0, lst_tol],
                )
            if self.time_range is not None:
                utils.times.check_lsts_against_times(
                    jd_array=self.time_range,
                    lst_array=self.lst_range,
                    telescope_loc=self.telescope.location,
                    lst_tols=self._lst_array.tols if lst_tol is None else [0, lst_tol],
                )
        return True

    def copy(self, *, metadata_only=False):
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

    def _key_exists(self, *, antnum=None, jpol=None):
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
                jpol = utils.jstr2num(jpol, x_orientation=self.telescope.x_orientation)
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
        if not self._key_exists(antnum=antnum):
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
            jpol = utils.jstr2num(jpol, x_orientation=self.telescope.x_orientation)

        if not self._key_exists(jpol=jpol):
            raise ValueError("{} not found in jones_array".format(jpol))

        return np.argmin(np.abs(self.jones_array - jpol))

    def _slice_array(self, key, data_array, *, squeeze_pol=True):
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
        key = utils.tools._get_iterable(key)
        if len(key) == 1:
            # interpret as a single antenna
            output = data_array[self.ant2ind(key[0]), :, :, :]
            if squeeze_pol and output.shape[-1] == 1:
                output = output[:, :, 0]
        elif len(key) == 2:
            # interpret as an antenna-pol pair
            output = data_array[self.ant2ind(key[0]), :, :, self.jpol2ind(key[1])]

        return output

    def _parse_key(self, ant, *, jpol=None):
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

    def get_gains(self, ant, jpol=None, *, squeeze_pol=True):
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

    def get_flags(self, ant, jpol=None, *, squeeze_pol=True):
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

    def get_quality(self, ant, jpol=None, *, squeeze_pol=True):
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

    def get_time_array(self):
        """
        Get a time array of calibration solution times.

        Times are for the center of the integration, shape (Ntimes), units in Julian
        Date. If a time_range is defined on the object, the times are the mean of the
        start and stop times for each range. Otherwise return the time_array (which can
        be None).
        """
        if self.time_range is not None:
            return np.mean(self.time_range, axis=1)
        else:
            return self.time_array

    def get_lst_array(self, *, astrometry_library=None):
        """
        Get an lst array of calibration solution lsts.

        LSTs are for the center of the integration, shape (Ntimes), units in radians.
        If an lst_range is defined on the object, the lsts are the LST of the mean
        of the start and stop times for each range. Otherwise return the lst_array
        (which can be None).

        Parameters
        ----------
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        """
        if self.lst_range is not None:
            return utils.get_lst_for_time(
                jd_array=self.get_time_array(), telescope_loc=self.telescope.location
            )
        else:
            return self.lst_array

    def reorder_antennas(
        self,
        order="number",
        *,
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
                temp = np.asarray(self.telescope.antenna_names)
                dtype_use = temp.dtype
                name_array = np.zeros_like(self.ant_array, dtype=dtype_use)
                # there has to be a better way to do this without a loop...
                for ind, ant in enumerate(self.ant_array):
                    name_array[ind] = self.telescope.antenna_names[
                        np.nonzero(self.telescope.antenna_numbers == ant)[0][0]
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
        *,
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
            index_array = utils.frequency._sort_freq_helper(
                Nfreqs=self.Nfreqs,
                freq_array=self.freq_array,
                Nspws=self.Nspws,
                spw_array=self.spw_array,
                flex_spw_id_array=self.flex_spw_id_array,
                spw_order=spw_order,
                channel_order=channel_order,
                select_spw=select_spw,
            )

            if index_array is None:
                # This only happens if no sorting is needed
                return

        # update all the relevant arrays
        if self.wide_band:
            self.spw_array = self.spw_array[index_array]
            self.freq_range = self.freq_range[index_array]
            if self.flex_jones_array is not None:
                self.flex_jones_array = self.flex_jones_array[index_array]
        else:
            self.freq_array = self.freq_array[index_array]
        for param_name in self._data_params:
            param = getattr(self, param_name)
            if param is not None:
                if param_name == "total_quality_array":
                    self.total_quality_array = self.total_quality_array[index_array]
                else:
                    setattr(self, param_name, param[:, index_array])
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
                if self.flex_jones_array is not None:
                    self.flex_jones_array = self.flex_jones_array[spw_index]

        if not self.wide_band:
            self.channel_width = self.channel_width[index_array]

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def reorder_times(
        self,
        order="time",
        *,
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
            Ntimes representing indexes along the existing `time_array` or `time_range`
            can also be supplied to sort in any desired order.

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

            if self.time_array is not None and self.time_range is not None:
                raise ValueError(
                    "The time_array and time_range attributes are both set."
                )

            if self.time_range is not None:
                index_array = np.argsort(self.time_range[:, 0])
            else:
                index_array = np.argsort(self.time_array)

            if order[0] == "-":
                index_array = np.flip(index_array)

        if np.all(index_array[1:] > index_array[:-1]):
            # Nothing to do - the data are already sorted!
            return

        # update all the relevant arrays
        if self.time_array is not None:
            self.time_array = self.time_array[index_array]
        if self.time_range is not None:
            self.time_range = self.time_range[index_array]
        if self.lst_array is not None:
            self.lst_array = self.lst_array[index_array]
        if self.lst_range is not None:
            self.lst_range = self.lst_range[index_array]
        if self.phase_center_id_array is not None:
            self.phase_center_id_array = self.phase_center_id_array[index_array]
        if self.ref_antenna_array is not None:
            self.ref_antenna_array = self.ref_antenna_array[index_array]
        self.integration_time = self.integration_time[index_array]
        for param_name in self._data_params:
            param = getattr(self, param_name)
            if param is not None:
                if param_name == "total_quality_array":
                    self.total_quality_array = self.total_quality_array[:, index_array]
                else:
                    setattr(self, param_name, param[:, :, index_array])

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def reorder_jones(
        self,
        order="name",
        *,
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
                    utils.jnum2str(
                        self.jones_array, x_orientation=self.telescope.x_orientation
                    )
                )
                index_array = np.argsort(name_array)

            if order[0] == "-":
                index_array = np.flip(index_array)

        if np.all(index_array[1:] > index_array[:-1]):
            # Nothing to do - the data are already sorted!
            return

        # update all the relevant arrays
        self.jones_array = self.jones_array[index_array]
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

    def convert_to_gain(
        self,
        *,
        freq_array,
        channel_width,
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
            Frequencies to convert to gain at, units Hz. Required, shape (Nfreqs,).
        channel_width : array of float
            Channel widths for each channel, units Hz. Required, shape (Nfreqs,).
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

        if freq_array.ndim > 1:
            raise ValueError("freq_array parameter must be a one dimensional array")

        if (
            not isinstance(channel_width, np.ndarray)
            or channel_width.shape != freq_array.shape
        ):
            raise ValueError(
                "The channel_width parameter be an array shaped like the freq_array"
            )

        if self.freq_range is not None:
            # Already errored if more than one spw, so just use the first one here
            freq_range_use = self.freq_range[0, :]
            if np.any(freq_array < freq_range_use[0]) or np.any(
                freq_array > freq_range_use[1]
            ):
                raise ValueError("freq_array contains values outside the freq_range.")
        freq_array_use = freq_array
        Nfreqs_use = freq_array.size

        self.history += "  Converted from delays to gains using pyuvdata."

        phase_array = np.zeros((self.Nants_data, Nfreqs_use, self.Ntimes, self.Njones))

        temp = (
            conv
            * 2
            * np.pi
            * np.dot(
                self.delay_array[:, 0, :, :, np.newaxis], freq_array_use[np.newaxis, :]
            )
        )
        temp = np.transpose(temp, (0, 3, 1, 2))
        phase_array = temp

        gain_array = np.exp(1j * phase_array)
        freq_axis = 1
        self._set_gain()
        self._set_wide_band(wide_band=False)
        self.channel_width = channel_width
        self.freq_range = None
        self.gain_array = gain_array
        self.delay_array = None
        if self.quality_array is not None:
            new_quality = np.repeat(self.quality_array, Nfreqs_use, axis=freq_axis)
            self.quality_array = new_quality
        new_flag_array = np.repeat(self.flag_array, Nfreqs_use, axis=freq_axis)
        self.flag_array = new_flag_array

        if self.total_quality_array is not None:
            freq_axis = 0
            new_total_quality_array = np.repeat(
                self.total_quality_array, Nfreqs_use, axis=freq_axis
            )
            self.total_quality_array = new_total_quality_array
        self.freq_array = freq_array_use
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
        *,
        verbose_history=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=False,
        ignore_name=False,
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
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when combining two UVCal objects. If set to True,
            phase centers that are the same up to their name will be combined with the
            name set to the name found in the first UVCal object in the sum. If set to
            False, phase centers that are the same up to the name will be kept as
            separate phase centers. Default is False.
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

        if (this.flex_jones_array is None) != (other.flex_jones_array is None):
            raise ValueError(
                "To combine these data, both objects must be either set to regular "
                "or flex-jones."
            )

        if (this.ref_antenna_array is None) != (other.ref_antenna_array is None):
            raise ValueError(
                "To combine these data, both or neither object must have the "
                "ref_antenna_array parameter set."
            )

        # Check that the objects both (or neither) have catalogs and phase ID arrays
        this_pc_cat = this.phase_center_catalog is None
        other_pc_cat = other.phase_center_catalog is None
        this_pc_ids = this.phase_center_id_array is None
        other_pc_ids = other.phase_center_id_array is None

        if (this_pc_cat != other_pc_cat) or (this_pc_ids != other_pc_ids):
            raise ValueError(
                "To combine these data, phase_center_id_array and "
                "_phase_center_catalog must be set for all objects."
            )

        # Check that both objects are either wide_band or not
        if this.wide_band != other.wide_band:
            raise ValueError(
                "To combine these data, wide_band must be set to the same "
                "value (True or False) for both objects."
            )

        # check that both either have or don't have time_array and time_range
        _time_param_check(this, other)

        # Check objects are compatible
        compatibility_params = [
            "_cal_type",
            "_telescope",
            "_gain_convention",
            "_cal_style",
            "_ref_antenna_name",
        ]

        warning_params = [
            "_observer",
            "_git_hash_cal",
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

        if this.time_range is not None:
            if utils.tools._check_range_overlap(
                np.concatenate((this.time_range, other.time_range), axis=0)
            ):
                raise ValueError("A time_range overlaps in the two objects.")
            both_times, this_times_ind, other_times_ind = np.intersect1d(
                this.time_range[:, 0], other.time_range[:, 0], return_indices=True
            )
        else:
            both_times, this_times_ind, other_times_ind = np.intersect1d(
                this.time_array, other.time_array, return_indices=True
            )

        if this.wide_band:
            # this is really about spws, but that replaces the freq axis for wide_band
            both_freq, this_freq_ind, other_freq_ind = np.intersect1d(
                this.spw_array, other.spw_array, return_indices=True
            )
        else:
            # With non-wideband objects, the handling here becomes a bit funky,
            # because we are allowed to have channels with the same frequency *if* they
            # belong to different spectral windows (one real-life example: might want
            # to preserve guard bands in the correlator, which can have overlaping RF
            # frequency channels)
            this_freq_ind = np.array([], dtype=np.int64)
            other_freq_ind = np.array([], dtype=np.int64)
            both_freq = np.array([], dtype=float)
            both_spw = np.intersect1d(this.spw_array, other.spw_array)
            for idx in both_spw:
                this_mask = np.where(this.flex_spw_id_array == idx)[0]
                other_mask = np.where(other.flex_spw_id_array == idx)[0]
                both_spw_freq, this_spw_ind, other_spw_ind = np.intersect1d(
                    this.freq_array[this_mask],
                    other.freq_array[other_mask],
                    return_indices=True,
                )
                this_freq_ind = np.append(this_freq_ind, this_mask[this_spw_ind])
                other_freq_ind = np.append(other_freq_ind, other_mask[other_spw_ind])
                both_freq = np.append(both_freq, both_spw_freq)

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

        # First, handle the internal source catalogs, since merging them is kind of a
        # weird, one-off process (i.e., nothing is cat'd across a particular axis)
        if this.phase_center_catalog is not None:
            this._consolidate_phase_center_catalogs(
                other=other, ignore_name=ignore_name
            )
            if (len(both_times) > 0) and (this.phase_center_id_array is not None):
                if not np.array_equal(
                    this.phase_center_id_array[this_times_ind],
                    other.phase_center_id_array[other_times_ind],
                ):
                    # TODO: I think this check actually needs to be expanded to other
                    # attributes, and along other axes
                    raise ValueError(
                        "Cannot combine objects due to overlapping times with "
                        "different phase centers."
                    )

        if (len(both_times) > 0) and (this.ref_antenna_array is not None):
            if not np.array_equal(
                this.ref_antenna_array[this_times_ind],
                other.ref_antenna_array[other_times_ind],
            ):
                raise ValueError(
                    "Cannot combine objects due to overlapping times with "
                    "different reference antennas."
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
        this.filename = utils.tools._combine_filenames(this.filename, other.filename)
        if this.filename is not None:
            this._filename.form = (len(this.filename),)

        temp = np.nonzero(~np.isin(other.ant_array, this.ant_array))[0]
        if len(temp) > 0:
            anew_inds = temp
            history_update_string += "antenna"
            n_axes += 1
        else:
            anew_inds = []

        if this.time_range is not None:
            temp = np.nonzero(~np.isin(other.time_range[:, 0], this.time_range[:, 0]))[
                0
            ]
        else:
            temp = np.nonzero(~np.isin(other.time_array, this.time_array))[0]

        if len(temp) > 0:
            tnew_inds = temp
            if n_axes > 0:
                history_update_string += ", time"
            else:
                history_update_string += "time"
            n_axes += 1
        else:
            tnew_inds = []

        if this.wide_band:
            temp = np.nonzero(~np.isin(other.spw_array, this.spw_array))[0]
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
            # find the freq indices in "other" but not in "this"
            other_mask = np.ones_like(other.flex_spw_id_array, dtype=bool)
            for idx in np.intersect1d(this.spw_array, other.spw_array):
                other_mask[other.flex_spw_id_array == idx] = np.isin(
                    other.freq_array[other.flex_spw_id_array == idx],
                    this.freq_array[this.flex_spw_id_array == idx],
                    invert=True,
                )
            temp = np.where(other_mask)[0]
            if len(temp) > 0:
                fnew_inds = temp
                if n_axes > 0:
                    history_update_string += ", frequency"
                else:
                    history_update_string += "frequency"
                n_axes += 1
            else:
                fnew_inds = []

        temp = np.nonzero(~np.isin(other.jones_array, this.jones_array))[0]
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
                zero_pad_data = np.zeros(
                    (len(anew_inds), data_array_shape[1], this.Ntimes, this.Njones)
                )
                zero_pad_flags = np.zeros(
                    (len(anew_inds), data_array_shape[1], this.Ntimes, this.Njones)
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

        f_order = None
        if len(fnew_inds) > 0:
            if this.wide_band:
                this.spw_array = np.concatenate(
                    [this.spw_array, other.spw_array[fnew_inds]]
                )
                this.freq_range = np.concatenate(
                    [this.freq_range, other.freq_range[fnew_inds]], axis=0
                )
                if this.flex_jones_array is not None:
                    this.flex_jones_array = np.concatenate(
                        [this.flex_jones_array, other.flex_jones_array[fnew_inds]],
                        axis=0,
                    )
                f_order = np.argsort(this.spw_array)
            else:
                this.flex_spw_id_array = np.concatenate(
                    [this.flex_spw_id_array, other.flex_spw_id_array[fnew_inds]]
                )
                this.freq_array = np.concatenate(
                    [this.freq_array, other.freq_array[fnew_inds]]
                )
                this.channel_width = np.concatenate(
                    [this.channel_width, other.channel_width[fnew_inds]]
                )

                # We want to preserve per-spw information based on first appearance
                # in the concatenated array.
                unique_index = np.sort(
                    np.unique(this.flex_spw_id_array, return_index=True)[1]
                )
                this.spw_array = np.sort(this.flex_spw_id_array[unique_index])
                if this.flex_jones_array is not None:
                    concat_spw_array = np.concatenate([this.spw_array, other.spw_array])
                    spw_index = np.asarray(
                        [
                            np.nonzero(concat_spw_array == spw)[0][0]
                            for spw in this.spw_array
                        ]
                    )
                    this.flex_jones_array = np.concatenate(
                        [this.flex_jones_array, other.flex_jones_array]
                    )
                    this.flex_jones_array = this.flex_jones_array[spw_index]
                # If we have a multi-spw data set, need to sort out the order of
                # the individual windows first.
                f_order = np.concatenate(
                    [
                        np.where(this.flex_spw_id_array == idx)[0]
                        for idx in this.spw_array
                    ]
                )

                # With spectral windows sorted, check and see if channels within
                # windows need sorting. If they are ordered in ascending or descending
                # fashion, leave them be. If not, sort in ascending order
                for idx in this.spw_array:
                    select_mask = this.flex_spw_id_array[f_order] == idx
                    check_freqs = this.freq_array[f_order[select_mask]]
                    if (not np.all(check_freqs[1:] > check_freqs[:-1])) and (
                        not np.all(check_freqs[1:] < check_freqs[:-1])
                    ):
                        subsort_order = f_order[select_mask]
                        f_order[select_mask] = subsort_order[np.argsort(check_freqs)]

            if not self.metadata_only:
                data_array_shape = getattr(this, this._required_data_params[0]).shape
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

        t_order = None
        if len(tnew_inds) > 0:
            if this.time_array is not None:
                this.time_array = np.concatenate(
                    [this.time_array, other.time_array[tnew_inds]]
                )
            if this.time_range is not None:
                this.time_range = np.concatenate(
                    [this.time_range, other.time_range[tnew_inds]]
                )
            if this.lst_array is not None:
                this.lst_array = np.concatenate(
                    [this.lst_array, other.lst_array[tnew_inds]]
                )
            if this.lst_range is not None:
                this.lst_range = np.concatenate(
                    [this.lst_range, other.lst_range[tnew_inds]]
                )
            if this.phase_center_id_array is not None:
                this.phase_center_id_array = np.concatenate(
                    [this.phase_center_id_array, other.phase_center_id_array[tnew_inds]]
                )
            if this.ref_antenna_array is not None:
                this.ref_antenna_array = np.concatenate(
                    [this.ref_antenna_array, other.ref_antenna_array[tnew_inds]]
                )
            if this.time_range is not None:
                t_order = np.argsort(this.time_range[:, 0])
            else:
                t_order = np.argsort(this.time_array)
            this.integration_time = np.concatenate(
                [this.integration_time, other.integration_time[tnew_inds]]
            )

            if not self.metadata_only:
                data_array_shape = getattr(this, this._required_data_params[0]).shape
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

        j_order = None
        if len(jnew_inds) > 0:
            this.jones_array = np.concatenate(
                [this.jones_array, other.jones_array[jnew_inds]]
            )
            j_order = np.argsort(np.abs(this.jones_array))
            if not self.metadata_only:
                data_array_shape = getattr(this, this._required_data_params[0]).shape
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

        # Now populate the data
        if not self.metadata_only:
            jones_t2o = np.nonzero(np.isin(this.jones_array, other.jones_array))[0]
            if this.time_range is not None:
                times_t2o = np.nonzero(
                    (np.isin(this.time_range[:, 0], other.time_range[:, 0]))
                    & (np.isin(this.time_range[:, 1], other.time_range[:, 1]))
                )[0]
            else:
                times_t2o = np.nonzero(np.isin(this.time_array, other.time_array))[0]

            if self.wide_band:
                freqs_t2o = np.nonzero(np.isin(this.spw_array, other.spw_array))[0]
            else:
                freqs_t2o = np.zeros(this.freq_array.shape, dtype=bool)
                for spw_id in set(this.spw_array).intersection(other.spw_array):
                    mask = this.flex_spw_id_array == spw_id
                    freqs_t2o[mask] |= np.isin(
                        this.freq_array[mask],
                        other.freq_array[other.flex_spw_id_array == spw_id],
                    )
                freqs_t2o = np.nonzero(freqs_t2o)[0]
            ants_t2o = np.nonzero(np.isin(this.ant_array, other.ant_array))[0]
            if this.cal_type == "delay":
                this.delay_array[np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)] = (
                    other.delay_array
                )
            else:
                this.gain_array[np.ix_(ants_t2o, freqs_t2o, times_t2o, jones_t2o)] = (
                    other.gain_array
                )
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

            # Fix ordering
            ant_axis_num = 0
            faxis_num = 1
            taxis_num = 2
            jaxis_num = 3

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
                this.freq_array = this.freq_array[f_order]
                this.channel_width = this.channel_width[f_order]
                this.flex_spw_id_array = this.flex_spw_id_array[f_order]
        if len(tnew_inds) > 0:
            if this.time_array is not None:
                this.time_array = this.time_array[t_order]
            if this.time_range is not None:
                this.time_range = this.time_range[t_order]
            if this.lst_array is not None:
                this.lst_array = this.lst_array[t_order]
            if this.lst_range is not None:
                this.lst_range = this.lst_range[t_order]
            this.integration_time = this.integration_time[t_order]
            if self.phase_center_id_array is not None:
                this.phase_center_id_array = this.phase_center_id_array[t_order]
            if self.ref_antenna_array is not None:
                this.ref_antenna_array = this.ref_antenna_array[t_order]
        if len(jnew_inds) > 0:
            this.jones_array = this.jones_array[j_order]

        # Update N parameters (e.g. Njones)
        this.Njones = this.jones_array.shape[0]
        if this.time_range is not None:
            this.Ntimes = this.time_range.shape[0]
        else:
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
                    "this data out to calfits files."
                )
            elif chanwidth_error:
                warnings.warn(
                    "Combined frequencies are separated by more than their "
                    "channel width. This will make it impossible to write this data "
                    "out to calfits files."
                )

        if this.Njones > 2:
            if not utils.tools._test_array_constant_spacing(this._jones_array):
                warnings.warn(
                    "Combined Jones elements are not evenly spaced. This will "
                    "make it impossible to write this data out to calfits files."
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

    def __iadd__(
        self,
        other,
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        ignore_name=False,
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
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when combining two UVCal objects. If set to True,
            phase centers that are the same up to their name will be combined with the
            name set to the name found in the first UVCal object in the sum. If set to
            False, phase centers that are the same up to the name will be kept as
            separate phase centers. Default is False.
        """
        self.__add__(
            other,
            inplace=True,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            ignore_name=ignore_name,
        )
        return self

    def fast_concat(
        self,
        other,
        axis,
        *,
        ignore_name=None,
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
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when combining two UVCal objects. If set to True,
            phase centers that are the same up to their name will be combined with the
            name set to the name found in the first UVCal object in the sum. If set to
            False, phase centers that are the same up to the name will be kept as
            separate phase centers. Default is False.
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

        # Check that all objects are consistent w/ use of flex-Jones
        for obj in other:
            if (this.flex_jones_array is None) != (obj.flex_jones_array is None):
                raise ValueError(
                    "To combine these data, all objects must be either set to regular "
                    "or flex-jones."
                )

        for obj in other:
            this_pc_cat = this.phase_center_catalog is None
            other_pc_cat = obj.phase_center_catalog is None
            this_pc_ids = this.phase_center_id_array is None
            other_pc_ids = obj.phase_center_id_array is None

            if (this_pc_cat != other_pc_cat) or (this_pc_ids != other_pc_ids):
                raise ValueError(
                    "To combine these data, phase_center_id_array and "
                    "_phase_center_catalog must be set for all objects."
                )

        # Check that all objects are consistent w/ use of wide_band or not
        for obj in other:
            if this.wide_band != obj.wide_band:
                raise ValueError(
                    "To combine these data, wide_band must be set to the same "
                    "value (True or False) for all objects."
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
            "_telescope",
            "_gain_convention",
            "_cal_style",
            "_ref_antenna_name",
        ]

        warning_params = [
            "_observer",
            "_git_hash_cal",
            "_sky_catalog",
            "_Nsources",
            "_baseline_range",
            "_diffuse_model",
        ]

        history_update_string = " Combined data along "

        # Check separately for phase_center_id_array, since it's optional
        if axis != "time" and this.phase_center_id_array is not None:
            compatibility_params += ["_phase_center_id_array"]

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
                "_lst_range",
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
                "_lst_range",
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
                "_lst_range",
            ]
        elif axis == "time":
            history_update_string += "time"
            compatibility_params += [
                "_freq_array",
                "_channel_width",
                "_jones_array",
                "_ant_array",
            ]
            _time_param_check(this, other)

        history_update_string += " axis using pyuvdata."
        histories_match = []
        for obj in other:
            histories_match.append(
                utils.history._check_histories(this.history, obj.history)
            )

        this.history += history_update_string
        for obj_num, obj in enumerate(other):
            if not histories_match[obj_num]:
                if verbose_history:
                    this.history += " Next object history follows. " + obj.history
                else:
                    extra_history = utils.history._combine_history_addition(
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

        # update the phase_center_catalog to make them consistent across objects
        # Doing this as a binary tree merge
        # The left object in each loop will have its phase center IDs updated.
        if this.phase_center_catalog is not None:
            uv_list = [this] + other
            while len(uv_list) > 1:
                for uv1, uv2 in zip(uv_list[0::2], uv_list[1::2]):
                    uv1._consolidate_phase_center_catalogs(
                        other=uv2, ignore_name=ignore_name
                    )
                uv_list = uv_list[0::2]

        total_quality_exists = [this.total_quality_array is not None] + [
            obj.total_quality_array is not None for obj in other
        ]
        this_tqa_exp_shape = this._total_quality_array.expected_shape(this)

        quality_exists = [this.quality_array is not None] + [
            obj.quality_array is not None for obj in other
        ]
        this_qa_exp_shape = this._quality_array.expected_shape(this)

        if axis == "antenna":
            this.Nants_data = sum([this.Nants_data] + [obj.Nants_data for obj in other])
            this.ant_array = np.concatenate(
                [this.ant_array] + [obj.ant_array for obj in other]
            )
            axis_num = 0
        elif axis == "freq":
            this.Nfreqs = sum([this.Nfreqs] + [obj.Nfreqs for obj in other])
            this.freq_array = np.concatenate(
                [this.freq_array] + [obj.freq_array for obj in other]
            )
            this.channel_width = np.concatenate(
                [this.channel_width] + [obj.channel_width for obj in other]
            )
            this.flex_spw_id_array = np.concatenate(
                [this.flex_spw_id_array] + [obj.flex_spw_id_array for obj in other]
            )
            # We want to preserve per-spw information based on first appearance
            # in the concatenated array.
            unique_index = np.sort(
                np.unique(this.flex_spw_id_array, return_index=True)[1]
            )
            this.spw_array = this.flex_spw_id_array[unique_index]
            this.Nspws = this.spw_array.size
            if this.flex_jones_array is not None:
                concat_spw_array = np.concatenate(
                    [this.spw_array] + [obj.spw_array for obj in other]
                )
                spw_index = np.asarray(
                    [
                        np.nonzero(concat_spw_array == spw)[0][0]
                        for spw in this.spw_array
                    ]
                )
                this.flex_jones_array = np.concatenate(
                    [this.flex_jones_array] + [obj.flex_jones_array for obj in other]
                )
                this.flex_jones_array = this.flex_jones_array[spw_index]

            spacing_error, chanwidth_error = this._check_freq_spacing(
                raise_errors=False
            )
            if spacing_error:
                warnings.warn(
                    "Combined frequencies are not evenly spaced or have differing "
                    "values of channel widths. This will make it impossible to write "
                    "this data out to calfits files."
                )
            elif chanwidth_error:
                warnings.warn(
                    "Combined frequencies are separated by more than their "
                    "channel width. This will make it impossible to write this data "
                    "out to calfits files."
                )

            axis_num = 1
        elif axis == "spw":
            # only get here for a wide-band cal
            this.Nspws = sum([this.Nspws] + [obj.Nspws for obj in other])
            this.spw_array = np.concatenate(
                [this.spw_array] + [obj.spw_array for obj in other]
            )
            this.freq_range = np.concatenate(
                [this.freq_range] + [obj.freq_range for obj in other], axis=0
            )
            if this.flex_jones_array is not None:
                this.flex_jones_array = np.concatenate(
                    [this.flex_jones_array] + [obj.flex_jones_array for obj in other],
                    axis=0,
                )
            axis_num = 1
        elif axis == "time":
            this.Ntimes = sum([this.Ntimes] + [obj.Ntimes for obj in other])
            if this.time_array is not None:
                this.time_array = np.concatenate(
                    [this.time_array] + [obj.time_array for obj in other]
                )
            if this.time_range is not None:
                this.time_range = np.concatenate(
                    [this.time_range] + [obj.time_range for obj in other], axis=0
                )
            this.integration_time = np.concatenate(
                [this.integration_time] + [obj.integration_time for obj in other]
            )
            if this.lst_array is not None:
                this.lst_array = np.concatenate(
                    [this.lst_array] + [obj.lst_array for obj in other]
                )
            if this.lst_range is not None:
                this.lst_range = np.concatenate(
                    [this.lst_range] + [obj.lst_range for obj in other], axis=0
                )
            if this.phase_center_id_array is not None:
                this.phase_center_id_array = np.concatenate(
                    (
                        [this.phase_center_id_array]
                        + [obj.phase_center_id_array for obj in other]
                    ),
                    axis=0,
                )
            if this.ref_antenna_array is not None:
                this.ref_antenna_array = np.concatenate(
                    [this.ref_antenna_array] + [obj.ref_antenna_array for obj in other],
                    axis=0,
                )
            axis_num = 2
        elif axis == "jones":
            this.Njones = sum([this.Njones] + [obj.Njones for obj in other])
            this.jones_array = np.concatenate(
                [this.jones_array] + [obj.jones_array for obj in other]
            )
            axis_num = 3

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
            this.filename = utils.tools._combine_filenames(this.filename, obj.filename)
        if this.filename is not None:
            this._filename.form = len(this.filename)

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return this

    def _select_preprocess(
        self,
        *,
        antenna_nums,
        antenna_names,
        frequencies,
        freq_chans,
        spws,
        times,
        time_range,
        lsts,
        lst_range,
        jones,
        phase_center_ids,
        catalog_names,
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
            The times to keep in the object, each value passed here should exist in
            the time_array or be contained in a time_range on the object.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        jones : array_like of int or str, optional
            The antenna polarizations numbers to keep in the object, each value
            passed here should exist in the jones_array. If passing strings, the
            canonical polarization strings (e.g. "Jxx", "Jrr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "Jnn", "Jee") are also supported.
        phase_center_ids : array_like of int, optional
            Phase center IDs to keep on the object (effectively a selection on
            baseline-times). Cannot be used with `catalog_names`.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to keep in the object, which should
            match exactly in spelling and capitalization. Cannot be used with
            `phase_center_ids`.

        Returns
        -------
        ant_inds : list of int
            list of antenna indices to keep. Can be None (to keep everything).
        time_inds : list of int
            list of time indices to keep. Can be None (to keep everything).
        spw_inds : list of int
            list of spw indices to keep. Can be None (to keep everything).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        pol_inds : list of int
            list of polarization indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.

        """
        # build up history string as we go
        history_update_string = "  Downselected to specific "
        n_selects = 0

        if antenna_names is not None:
            if antenna_nums is not None:
                raise ValueError(
                    "Only one of antenna_nums and antenna_names can be provided."
                )

            antenna_names = utils.tools._get_iterable(antenna_names)
            antenna_nums = []
            for s in antenna_names:
                if s not in self.telescope.antenna_names:
                    raise ValueError(
                        f"Antenna name {s} is not present in the antenna_names array"
                    )
                ind = np.where(np.array(self.telescope.antenna_names) == s)[0][0]
                antenna_nums.append(self.telescope.antenna_numbers[ind])

        if antenna_nums is not None:
            antenna_nums = utils.tools._get_iterable(antenna_nums)
            history_update_string += "antennas"
            n_selects += 1

            ant_inds = np.zeros(0, dtype=np.int64)
            for ant in antenna_nums:
                if ant in self.ant_array:
                    ant_inds = np.append(ant_inds, np.where(self.ant_array == ant)[0])
                else:
                    raise ValueError(
                        f"Antenna number {ant} is not present in the array"
                    )

            ant_inds = sorted(set(ant_inds))
        else:
            ant_inds = None

        if (phase_center_ids is not None) and (catalog_names is not None):
            raise ValueError("Cannot set both phase_center_ids and catalog_names.")

        if self.phase_center_id_array is None or self.phase_center_catalog is None:
            if (phase_center_ids is not None) or (catalog_names is not None):
                raise ValueError(
                    "Both phase_center_id_array and phase_center_catalog attributes of "
                    "the UVCal object must be set in order to select on phase center "
                    "IDs or catalog names."
                )

        if catalog_names is not None:
            phase_center_ids = utils.phase_center_catalog.look_for_name(
                self.phase_center_catalog, catalog_names
            )

        time_inds = utils.times._select_times_helper(
            times=times,
            time_range=time_range,
            lsts=lsts,
            lst_range=lst_range,
            obj_time_array=self.time_array,
            obj_time_range=self.time_range,
            obj_lst_array=self.lst_array,
            obj_lst_range=self.lst_range,
            time_tols=self._time_array.tols,
            lst_tols=self._lst_array.tols,
        )
        if time_inds is not None:
            time_inds = sorted(set(time_inds.tolist()))

        if times is not None or time_range is not None:
            if n_selects > 0:
                history_update_string += ", times"
            else:
                history_update_string += "times"
            n_selects += 1

        if lsts is not None or lst_range is not None:
            if n_selects > 0:
                history_update_string += ", lsts"
            else:
                history_update_string += "lsts"
            n_selects += 1

        if phase_center_ids is not None:
            pc_check = np.isin(self.phase_center_id_array, phase_center_ids)
            time_inds = utils.tools._sorted_unique_intersection(
                np.where(pc_check)[0], time_inds
            )

            update_substring = (
                "phase center IDs" if (catalog_names is None) else "catalog names"
            )
            if n_selects > 0:
                history_update_string += ", " + update_substring
            else:
                history_update_string += update_substring
            n_selects += 1

        if time_inds is not None and self.time_range is None:
            # don't warn if time_range is not None because calfits does not support
            # multiple time_range values
            time_inds_arr = np.array(time_inds)
            if time_inds_arr.size > 1:
                time_ind_separation = time_inds_arr[1:] - time_inds_arr[:-1]
                if not utils.tools._test_array_constant(time_ind_separation):
                    warnings.warn(
                        "Selected times are not evenly spaced. This "
                        "is not supported by the calfits format."
                    )

        if spws is not None:
            if self.Nspws == 1:
                warnings.warn(
                    "Cannot select on spws if Nspws=1. Ignoring the spw parameter."
                )
                spw_inds = None
            else:
                if not self.wide_band:
                    # Translate the spws into frequencies
                    freq_chans = utils.tools._sorted_unique_union(
                        np.where(np.isin(self.flex_spw_id_array, spws))[0], freq_chans
                    )
                    spw_inds = None
                else:
                    if n_selects > 0:
                        history_update_string += ", spectral windows"
                    else:
                        history_update_string += "spectral windows"
                    n_selects += 1

                    # Check and see that all requested spws are available
                    spw_check = np.isin(spws, self.spw_array)
                    if not np.all(spw_check):
                        raise ValueError(
                            f"SPW number {spws[np.where(~spw_check)[0][0]]} is not "
                            "present in the spw_array"
                        )

                    spw_inds = np.where(np.isin(self.spw_array, spws))[0]
        else:
            spw_inds = None

        if self.freq_array is None and (
            freq_chans is not None or frequencies is not None
        ):
            raise ValueError(
                "Cannot select on frequencies because this is a wide_band object with "
                "no freq_array."
            )

        if frequencies is not None:
            frequencies = utils.tools._get_iterable(frequencies)
            freq_arr_use = self.freq_array

            freq_check = np.isin(frequencies, freq_arr_use)
            if not np.all(freq_check):
                raise ValueError(
                    f"Frequency {frequencies[np.where(~freq_check)[0][0]]} is not "
                    "present in the freq_array"
                )

            freq_chans = utils.tools._sorted_unique_union(
                np.where(np.isin(freq_arr_use, frequencies))[0], freq_chans
            )

        if freq_chans is not None:
            if n_selects > 0:
                history_update_string += ", frequencies"
            else:
                history_update_string += "frequencies"
            n_selects += 1

            # Check and see that all requested freqs are available
            if frequencies is not None:
                pass

            freq_inds = np.array(sorted(utils.tools._get_iterable(freq_chans)))

            if len(freq_inds) > 1:
                freq_ind_separation = freq_inds[1:] - freq_inds[:-1]
                if self.flex_spw_id_array is not None:
                    freq_ind_separation = freq_ind_separation[
                        np.diff(self.flex_spw_id_array[freq_inds]) == 0
                    ]
                if not utils.tools._test_array_constant(freq_ind_separation):
                    warnings.warn(
                        "Selected frequencies are not evenly spaced. This "
                        "will make it impossible to write this data out to "
                        "calfits files"
                    )
                elif np.max(freq_ind_separation) > 1:
                    warnings.warn(
                        "Selected frequencies are not contiguous. This "
                        "will make it impossible to write this data out to "
                        "calfits files."
                    )

            freq_inds = sorted(set(freq_inds))
        else:
            freq_inds = None

        if jones is not None:
            jones = utils.tools._get_iterable(jones)
            if np.array(jones).ndim > 1:
                jones = np.array(jones).flatten()
            if n_selects > 0:
                history_update_string += ", jones polarization terms"
            else:
                history_update_string += "jones polarization terms"
            n_selects += 1

            jones_inds = np.zeros(0, dtype=np.int64)
            jones_spws = np.zeros(0, dtype=np.int64)
            for j in jones:
                if isinstance(j, str):
                    j_num = utils.jstr2num(
                        j, x_orientation=self.telescope.x_orientation
                    )
                else:
                    j_num = j
                if j_num in self.jones_array:
                    jones_inds = np.append(
                        jones_inds, np.where(self.jones_array == j_num)[0]
                    )
                elif (
                    self.flex_jones_array is not None and j_num in self.flex_jones_array
                ):
                    jones_spws = np.append(
                        jones_spws, np.where(self.flex_jones_array == j_num)[0]
                    )
                else:
                    raise ValueError(
                        "Jones term {j} is not present in the jones_array".format(j=j)
                    )
            if len(jones_spws) > 0:
                # Since this is a flex-pol data set, we need to filter on the freq
                # axis instead of the pol axis
                jones_inds = None

                if not self.wide_band:
                    jones_chans = np.where(
                        np.isin(self.flex_spw_id_array, self.spw_array[jones_spws])
                    )[0]
                    freq_inds = utils.tools._sorted_unique_intersection(
                        jones_chans, freq_inds
                    )
                spw_inds = utils.tools._sorted_unique_intersection(jones_spws, spw_inds)

                # Trap a corner case here where the frequency and polarization selects
                # on a flex-pol data set end up with no actual data being selected.
                if (len(spw_inds) == 0) or (not self.wide_band and len(freq_inds) == 0):
                    raise ValueError(
                        "No data matching this Jones selection in this flex-Jones "
                        " UVCal object."
                    )
                spacing_check = utils.tools._test_array_constant_spacing(
                    np.unique(self.flex_jones_array[spw_inds])
                )
            else:
                jones_inds = sorted(set(jones_inds))
                spacing_check = utils.tools._test_array_constant_spacing(
                    self.jones_array[jones_inds]
                )
            if not spacing_check:
                warnings.warn(
                    "Selected jones polarization terms are not evenly spaced. This "
                    "will make it impossible to write this data out to calfits files."
                )
        else:
            jones_inds = None

        history_update_string += " using pyuvdata."

        if n_selects == 0:
            history_update_string = ""

        return (
            ant_inds,
            time_inds,
            spw_inds,
            freq_inds,
            jones_inds,
            history_update_string,
        )

    def _select_by_index(
        self,
        *,
        ant_inds,
        time_inds,
        spw_inds,
        freq_inds,
        jones_inds,
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
        # Create a dictionary that we can loop over an update if need be
        ind_dict = {
            "Nants_data": ant_inds,
            "Ntimes": time_inds,
            "Nspws": spw_inds,
            "Nfreqs": freq_inds,
            "Njones": jones_inds,
        }

        # During each loop interval, we pop off an element of this dict, so continue
        # until the dict is empty.
        while len(ind_dict):
            # This is an easy way to grab the first key in the dict
            key = next(iter(ind_dict))
            # Grab the corresponding index array
            ind_arr = ind_dict.pop(key)

            # If nothing to select on, bail!
            if ind_arr is None:
                continue

            for param in self:
                # For each attribute, if the value is None, then bail, otherwise
                # attempt to figure out along which axis ind_arr will apply.
                attr = getattr(self, param)
                if attr.value is not None:
                    try:
                        sel_axis = attr.form.index(key)
                    except (AttributeError, ValueError):
                        # If form is not a tuple/list (and therefore not
                        # array-like), it'll throw an AttributeError, and if key is
                        # not found in the tuple/list, it'll throw a ValueError.
                        # In both cases, skip!
                        continue

                    if isinstance(attr.value, np.ndarray):
                        # If we're working with an ndarray, use take to slice along
                        # the axis that we want to grab from.
                        attr.value = attr.value.take(ind_arr, axis=sel_axis)
                    elif isinstance(attr.value, list):
                        # If this is a list, it _should_ always have 1-dimension.
                        assert sel_axis == 0, (
                            "Something is wrong, sel_axis != 0 when selecting on a "
                            "list, which should not be possible. Please file an "
                            "issue in our GitHub issue log so that we can fix it."
                        )
                        attr.value = [attr.value[idx] for idx in ind_arr]

            if key == "Nants_data":
                # Count the number of unique antennas after antenna-based selection
                self.Nants_data = len(ind_arr)
                if self.total_quality_array is not None:
                    warnings.warn(
                        "Changing number of antennas, but preserving the "
                        "total_quality_array, which may have been defined based "
                        "in part on antennas which will be removed."
                    )
            elif key == "Ntimes":
                # Process post time-specific selection actions
                self.Ntimes = len(ind_arr)
            elif key == "Nfreqs":
                # Process post freq-specific selection actions
                self.Nfreqs = len(ind_arr)
                if self.flex_spw_id_array is not None:
                    # If we are dropping channels, then evaluate the spw axis
                    ind_dict["Nspws"] = np.where(
                        np.isin(self.spw_array, self.flex_spw_id_array)
                    )[0]
            elif key == "Njones":
                # Count the number of unique pols after pol-based selection
                self.Njones = len(ind_arr)
            elif key == "Nspws":
                # Count the number of unique pols after spw-based selection
                self.Nspws = len(ind_arr)

        # Update the history string
        self.history += history_update_string

    def select(
        self,
        *,
        antenna_nums=None,
        antenna_names=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        jones=None,
        phase_center_ids=None,
        catalog_names=None,
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
            The times to keep in the object, each value passed here should exist in
            the time_array or be contained in a time_range on the object.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        jones : array_like of int or str, optional
            The antenna polarizations numbers to keep in the object, each value
            passed here should exist in the jones_array. If passing strings, the
            canonical polarization strings (e.g. "Jxx", "Jrr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "Jnn", "Jee") are also supported.
        phase_center_ids : array_like of int, optional
            Phase center IDs to keep on the object (effectively a selection on
            baseline-times). Cannot be used with `catalog_names`.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to keep in the object, which should
            match exactly in spelling and capitalization. Cannot be used with
            `phase_center_ids`.
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

        # Figure out which index positions we want to hold on to.
        (
            ant_inds,
            time_inds,
            spw_inds,
            freq_inds,
            jones_inds,
            history_update_string,
        ) = cal_object._select_preprocess(
            antenna_nums=antenna_nums,
            antenna_names=antenna_names,
            frequencies=frequencies,
            freq_chans=freq_chans,
            spws=spws,
            times=times,
            time_range=time_range,
            lsts=lsts,
            lst_range=lst_range,
            jones=jones,
            phase_center_ids=phase_center_ids,
            catalog_names=catalog_names,
        )

        # Call the low-level selection method.
        cal_object._select_by_index(
            ant_inds=ant_inds,
            time_inds=time_inds,
            freq_inds=freq_inds,
            spw_inds=spw_inds,
            jones_inds=jones_inds,
            history_update_string=history_update_string,
        )

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
        elif filetype == "calh5":
            from . import calh5

            other_obj = calh5.CalH5()
        elif filetype == "ms":
            from . import ms_cal

            other_obj = ms_cal.MSCal()
        else:
            raise ValueError("filetype must be calh5, calfits, or ms.")
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    @classmethod
    @combine_docstrings(initializers.new_uvcal_from_uvdata)
    def initialize_from_uvdata(
        cls, uvdata, *, gain_convention, cal_style, metadata_only=True, **kwargs
    ):
        """
        Initialize this object based on a UVData object.

        Parameters
        ----------
        uvdata : UVData object
            The UVData object to initialize from.
        metadata_only : bool
            Option to only initialize the metadata. If False, this method also
            initializes the data-like arrays to zeros/ones as appropriate
            (or False for the flag_array) with the appropriate sizes.
        """  # noqa: D207,RST203
        new = initializers.new_uvcal_from_uvdata(
            uvdata=uvdata,
            gain_convention=gain_convention,
            cal_style=cal_style,
            empty=not metadata_only,
            **kwargs,
        )

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
            Defunct option, will result in an error in version 3.2.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        """
        from . import calfits

        if isinstance(filename, (list, tuple)):
            raise ValueError(
                "Use the generic `UVCal.read` method to read multiple files."
            )

        calfits_obj = calfits.CALFITS()
        calfits_obj.read_calfits(filename, **kwargs)
        self._convert_from_filetype(calfits_obj)
        del calfits_obj

    def read_calh5(self, filename, **kwargs):
        """
        Read in data from calh5 file(s).

        Parameters
        ----------
        filename : string, path, FastCalH5Meta, h5py.File
            The CalH5 file to read from. A file name or path or a FastCalH5Meta or
            h5py File object. Must contains a "/Header" group for CalH5 files
            conforming to spec.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_names` is also provided. Ignored if
            read_data is False.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_nums` is also provided. Ignored if
            read_data is False.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array. Ignored if
            read_data is False.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the
            object. Ignored if read_data is False.
        spws : array_like of in, optional
            The spectral window numbers to keep in the object. If this is not a
            wide-band object and `frequencies` or `freq_chans` is not None, frequencies
            that match any of the specifications will be kept (i.e. the selections will
            be OR'ed together).
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        jones : array_like of int, optional
            The jones polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
            Ignored if read_data is False.
        phase_center_ids : array_like of int, optional
            Phase center IDs to keep on the object (effectively a selection on
            baseline-times). Cannot be used with `catalog_names`.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to keep in the object, which should
            match exactly in spelling and capitalization. Cannot be used with
            `phase_center_ids`.
        read_data : bool
            Read in the data-like arrays (gains/delays, flags, qualities). If set to
            False, only the metadata will be read in. Setting read_data to False
            results in a metadata only object.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
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
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If filename doesn't exist.
        ValueError
            If incompatible select keywords are set (e.g. `times` and `time_range`) or
            select keywords exclude all data or if keywords are set to the wrong type.

        """
        from . import calh5

        if isinstance(filename, (list, tuple)):
            raise ValueError(
                "Use the generic `UVCal.read` method to read multiple files."
            )

        calh5_obj = calh5.CalH5()
        calh5_obj.read_calh5(filename, **kwargs)
        self._convert_from_filetype(calh5_obj)
        del calh5_obj

    def read_fhd_cal(
        self, cal_file, *, obs_file, layout_file=None, settings_file=None, **kwargs
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
            Defunct option, will result in an error in version 3.2.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        """
        from . import fhd_cal

        if isinstance(cal_file, (list, tuple)):
            raise ValueError(
                "Use the generic `UVCal.read` method to read multiple files."
            )

        fhd_cal_obj = fhd_cal.FHDCal()
        fhd_cal_obj.read_fhd_cal(
            cal_file=cal_file,
            obs_file=obs_file,
            layout_file=layout_file,
            settings_file=settings_file,
            **kwargs,
        )
        self._convert_from_filetype(fhd_cal_obj)
        del fhd_cal_obj

    def read_ms_cal(self, filename, **kwargs):
        """
        Read in data from an MS calibration table.

        Parameters
        ----------
            filename : str
                The measurement set to read from.
            run_check : bool
                Option to check for the existence and proper shapes of
                parameters after reading in the file.
            default_x_orientation : str
                By default, if not found on read, the x_orientation parameter will be
                set to "east" and a warning will be raised. However, if a value for
                default_x_orientation is provided, it will be used instead and the
                warning will be suppressed.
            default_jones_array : ndarray of int
                By default, if not found on read, the jones_array parameter will be
                set to [0, 0] (unknown pol type) and a warning will be raised. However,
                if a value for default_jones_array is provided, it will be used instead
                and the warning will be suppressed.
            check_extra : bool
                Option to check optional parameters as well as required ones.
            run_check_acceptability : bool
                Option to check acceptable range of the values of
                parameters after reading in the file.
            astrometry_library : str
                Library used for calculating LSTs. Allowed options are 'erfa' (which
                uses the pyERFA), 'novas' (which uses the python-novas library), and
                'astropy' (which uses the astropy utilities). Default is erfa unless
                the telescope_location frame is MCMF (on the moon), in which case the
                default is astropy.
        """
        from . import ms_cal

        if isinstance(filename, (list, tuple)):
            raise ValueError(
                "Use the generic `UVCal.read` method to read multiple files."
            )

        ms_cal_obj = ms_cal.MSCal()
        ms_cal_obj.read_ms_cal(filename, **kwargs)
        self._convert_from_filetype(ms_cal_obj)
        del ms_cal_obj

    def read(
        self,
        filename,
        *,
        axis=None,
        file_type=None,
        read_data=True,
        background_lsts=True,
        use_future_array_shapes=None,
        astrometry_library=None,
        # selecting parameters
        antenna_nums=None,
        antenna_names=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        jones=None,
        phase_center_ids=None,
        catalog_names=None,
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
        followed by parameters used by all file types related to selecting and
        checking. Each file type also has its own set of optional parameters
        that are listed at the end of this docstring.

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
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        use_future_array_shapes : bool
            Defunct option, will result in an error in version 3.2.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Selecting
        ---------
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_names` is also provided. Ignored if
            read_data is False.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_nums` is also provided. Ignored if
            read_data is False.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array. Ignored if
            read_data is False.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the
            object. Ignored if read_data is False.
        spws : array_like of in, optional
            The spectral window numbers to keep in the object. If this is not a
            wide-band object and `frequencies` or `freq_chans` is not None, frequencies
            that match any of the specifications will be kept (i.e. the selections will
            be OR'ed together).
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        jones : array_like of int, optional
            The jones polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
            Ignored if read_data is False.
        phase_center_ids : array_like of int, optional
            Phase center IDs to keep on the object (effectively a selection on
            baseline-times). Cannot be used with `catalog_names`.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to keep in the object, which should
            match exactly in spelling and capitalization. Cannot be used with
            `phase_center_ids`.

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
        # Check for the defunct keyword up front
        self._set_future_array_shapes(use_future_array_shapes=use_future_array_shapes)

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
            elif "h5" in extension:
                file_type = "calh5"

        if file_type is None:
            # Nothing could be auto-determined, so let's jump to the next level and
            # try to look at the file contents and see what's up. In paticular, if this
            # is a directory, there's some additional clues for us to look for.
            file_test = filename[0] if multi else filename

            if os.path.isdir(file_test):
                if os.path.exists(os.path.join(file_test, "OBSERVATION")):
                    # It's a measurement set.
                    file_type = "ms"

        if file_type is None:
            raise ValueError(
                "File type could not be determined, use the "
                "file_type keyword to specify the type."
            )

        if file_type not in ["calfits", "fhd", "calh5", "ms"]:
            raise ValueError(
                "The only supported file_types are 'calfits', 'calh5', 'fhd', and 'ms'."
            )

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
                background_lsts=background_lsts,
                astrometry_library=astrometry_library,
                # selecting parameters
                antenna_nums=antenna_nums,
                antenna_names=antenna_names,
                frequencies=frequencies,
                freq_chans=freq_chans,
                spws=spws,
                times=times,
                time_range=time_range,
                lsts=lsts,
                lst_range=lst_range,
                jones=jones,
                phase_center_ids=phase_center_ids,
                catalog_names=catalog_names,
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
                    background_lsts=background_lsts,
                    astrometry_library=astrometry_library,
                    # selecting parameters
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    frequencies=frequencies,
                    freq_chans=freq_chans,
                    spws=spws,
                    times=times,
                    time_range=time_range,
                    lsts=lsts,
                    lst_range=lst_range,
                    jones=jones,
                    phase_center_ids=phase_center_ids,
                    catalog_names=catalog_names,
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
            if file_type in ["fhd", "calfits", "ms"]:
                if (
                    antenna_nums is not None
                    or antenna_names is not None
                    or frequencies is not None
                    or freq_chans is not None
                    or spws is not None
                    or times is not None
                    or lsts is not None
                    or time_range is not None
                    or lst_range is not None
                    or jones is not None
                    or phase_center_ids is not None
                    or catalog_names is not None
                ):
                    select = True
                    warnings.warn(
                        "Warning: select on read keyword set, but "
                        f'file_type is "{file_type}" which does not support select '
                        "on read. Entire file will be read and then select "
                        "will be performed"
                    )
                    # these file types do not have select on read, so set all
                    # select parameters
                    select_antenna_nums = antenna_nums
                    select_antenna_names = antenna_names
                    select_frequencies = frequencies
                    select_freq_chans = freq_chans
                    select_spws = spws
                    select_times = times
                    select_lsts = lsts
                    select_time_range = time_range
                    select_lst_range = lst_range
                    select_jones = jones
                    select_phase_center_ids = phase_center_ids
                    select_catalog_names = catalog_names
                else:
                    select = False
            elif file_type in ["calh5"]:
                select = False

            if file_type == "calfits":
                self.read_calfits(
                    filename,
                    read_data=read_data,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    astrometry_library=astrometry_library,
                )

            elif file_type == "fhd":
                self.read_fhd_cal(
                    cal_file=filename,
                    obs_file=obs_file,
                    layout_file=layout_file,
                    settings_file=settings_file,
                    raw=raw,
                    read_data=read_data,
                    extra_history=extra_history,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    astrometry_library=astrometry_library,
                )
            elif file_type == "calh5":
                self.read_calh5(
                    filename,
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    frequencies=frequencies,
                    freq_chans=freq_chans,
                    spws=spws,
                    times=times,
                    time_range=time_range,
                    lsts=lsts,
                    lst_range=lst_range,
                    jones=jones,
                    phase_center_ids=phase_center_ids,
                    catalog_names=catalog_names,
                    read_data=read_data,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    astrometry_library=astrometry_library,
                )
            elif file_type == "ms":
                self.read_ms_cal(
                    filename,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    astrometry_library=astrometry_library,
                )

            if select:
                self.select(
                    antenna_nums=select_antenna_nums,
                    antenna_names=select_antenna_names,
                    frequencies=select_frequencies,
                    freq_chans=select_freq_chans,
                    spws=select_spws,
                    times=select_times,
                    lsts=select_lsts,
                    time_range=select_time_range,
                    lst_range=select_lst_range,
                    jones=select_jones,
                    phase_center_ids=select_phase_center_ids,
                    catalog_names=select_catalog_names,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                )

    @classmethod
    @copy_replace_short_description(read, style=DocstringStyle.NUMPYDOC)
    def from_file(cls, filename, **kwargs):
        """Initialize a new UVCal object by reading the input file."""
        uvc = cls()
        uvc.read(filename, **kwargs)
        return uvc

    def write_calfits(
        self,
        filename,
        *,
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

    def write_calh5(self, filename, **kwargs):
        """
        Write the data to a calh5 file.

        Write an in-memory UVCal object to a CalH5 file.

        Parameters
        ----------
        filename : str
            The CalH5 file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the gain_array or delay. Default is None
            (no filter/compression). In addition to the normal HDF5 filter values, the
            user may specify "bitshuffle" which will set the compression to `32008` for
            bitshuffle and will set the `compression_opts` to `(0, 2)` to allow
            bitshuffle to automatically determine the block size and to use the LZF
            filter after bitshuffle. Using `bitshuffle` requires having the
            `hdf5plugin` package installed.  Dataset must be chunked to use compression.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is the
            LZF filter. Dataset must be chunked.
        quality_compression : str
            HDF5 filter to apply when writing the quality_array and/or
            total_quality_array if they are defined. Default is the LZF filter. Dataset
            must be chunked.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the file located at `filename` already exists and clobber=False,
            an IOError is raised.

        Notes
        -----
        The HDF5 library allows for the application of "filters" when writing
        data, which can provide moderate to significant levels of compression
        for the datasets in question.  Testing has shown that for some typical
        cases of UVData objects (empty/sparse flag_array objects, and/or uniform
        nsample_arrays), the built-in LZF filter provides significant
        compression for minimal computational overhead.

        """
        if self.metadata_only:
            raise ValueError("Cannot write out metadata only objects to a calh5 file.")

        calh5_obj = self._convert_to_filetype("calh5")
        calh5_obj.write_calh5(filename, **kwargs)
        del calh5_obj

    def write_ms_cal(self, filename, **kwargs):
        """
        Write the data to a MS calibration table.

        Parameters
        ----------
        filename : str
            The measurement set to write to.
        clobber : bool
            Option to overwrite the file if it already exists.

        """
        ms_cal_obj = self._convert_to_filetype("ms")
        ms_cal_obj.write_ms_cal(filename, **kwargs)
        del ms_cal_obj
