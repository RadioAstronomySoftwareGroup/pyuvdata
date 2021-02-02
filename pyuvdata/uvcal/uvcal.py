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

__all__ = ["UVCal"]


class UVCal(UVBase):
    """
    A class defining calibration solutions.

    Currently supported file types: calfits

    Attributes
    ----------
    UVParameter objects: For full list see UVCal Parameters
        (http://pyuvdata.readthedocs.io/en/latest/uvcal_parameters.html).
        Some are always required, some are required for certain cal_types
        and others are always optional.

    """

    def __init__(self):
        self._Nfreqs = uvp.UVParameter(
            "Nfreqs", description="Number of frequency channels", expected_type=int
        )
        self._Njones = uvp.UVParameter(
            "Njones",
            description="Number of Jones calibration"
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
            "(ie non-contiguous spectral chunks). "
            "More than one spectral window is not "
            "currently supported.",
            expected_type=int,
        )

        desc = (
            "Time range (in JD) that cal solutions are valid for."
            "list: [start_time, end_time] in JD. Should only be set in Ntimes is 1."
        )
        self._time_range = uvp.UVParameter(
            "time_range", description=desc, form=2, expected_type=float, required=False
        )

        desc = "Name of telescope. e.g. HERA. String."
        self._telescope_name = uvp.UVParameter(
            "telescope_name", description=desc, form="str", expected_type=str
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

        self._spw_array = uvp.UVParameter(
            "spw_array",
            description="Array of spectral window numbers, shape (Nspws).",
            form=("Nspws",),
            expected_type=int,
        )

        desc = (
            "Array of frequencies, center of the channel, "
            "shape (Nspws, Nfreqs), units Hz."
        )
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=("Nspws", "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )  # mHz

        desc = "Channel width of of a frequency bin. Units Hz."
        self._channel_width = uvp.UVParameter(
            "channel_width", description=desc, expected_type=float, tols=1e-3
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

        # standard angle tolerance: 10 mas in radians.
        # Should perhaps be decreased to 1 mas in the future
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)
        desc = "Array of lsts, center of integration, shape (Ntimes), units radians"
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Ntimes",),
            expected_type=float,
            tols=radian_tol,
            required=False,
        )

        desc = "Integration time of a time bin, units seconds."
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
            "Shape: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), type = bool."
        )
        self._flag_array = uvp.UVParameter(
            "flag_array",
            description=desc,
            form=("Nants_data", "Nspws", "Nfreqs", "Ntimes", "Njones"),
            expected_type=bool,
        )

        desc = (
            "Array of qualities of calibration solutions. "
            "The shape depends on cal_type, if the cal_type is 'gain' or "
            "'unknown', the shape is: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), "
            "if the cal_type is 'delay', the shape is "
            "(Nants_data, Nspws, 1, Ntimes, Njones). The type is float."
        )
        self._quality_array = uvp.UVParameter(
            "quality_array",
            description=desc,
            form=("Nants_data", "Nspws", "Nfreqs", "Ntimes", "Njones"),
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
            "shape: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), type = complex float."
        )
        self._gain_array = uvp.UVParameter(
            "gain_array",
            description=desc,
            required=False,
            form=("Nants_data", "Nspws", "Nfreqs", "Ntimes", "Njones"),
            expected_type=complex,
        )

        desc = (
            'Required if cal_type = "delay". Array of delays with units of seconds. '
            "Shape: (Nants_data, Nspws, 1, Ntimes, Njones), type = float."
        )
        self._delay_array = uvp.UVParameter(
            "delay_array",
            description=desc,
            required=False,
            form=("Nants_data", "Nspws", 1, "Ntimes", "Njones"),
            expected_type=float,
        )

        desc = (
            "Required if cal_type = 'delay'. Frequency range that solutions "
            "are valid for. list: [start_frequency, end_frequency] in Hz."
        )
        self._freq_range = uvp.UVParameter(
            "freq_range",
            required=False,
            description=desc,
            form=2,
            expected_type=float,
            tols=1e-3,
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
            'Required if cal_style = "sky". Short string describing field '
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
            "Array of input flags, True is flagged. shape: (Nants_data, Nspws, "
            "Nfreqs, Ntimes, Njones), type = bool."
        )
        self._input_flag_array = uvp.UVParameter(
            "input_flag_array",
            description=desc,
            required=False,
            form=("Nants_data", "Nspws", "Nfreqs", "Ntimes", "Njones"),
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
            'The shape depends on cal_type, if the cal_type is "gain" or '
            '"unknown", the shape is: (Nspws, Nfreqs, Ntimes, Njones), '
            'if the cal_type is "delay", the shape is (Nspws, 1, Ntimes, Njones), '
            "type = float."
        )
        self._total_quality_array = uvp.UVParameter(
            "total_quality_array",
            description=desc,
            form=("Nspws", "Nfreqs", "Ntimes", "Njones"),
            expected_type=float,
            required=False,
        )

        desc = (
            "Any user supplied extra keywords, type=dict. Keys should be "
            "8 character or less strings if writing to calfits files. "
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

        super(UVCal, self).__init__()

    def _set_gain(self):
        """Set cal_type to 'gain' and adjust required parameters."""
        self.cal_type = "gain"
        self._gain_array.required = True
        self._delay_array.required = False
        self._freq_range.required = False
        self._quality_array.form = self._gain_array.form
        self._total_quality_array.form = self._gain_array.form[1:]

    def set_gain(self):
        """
        Set cal_type to 'gain' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_gain` instead.
        """
        warnings.warn(
            "`set_gain` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_gain` instead.",
            DeprecationWarning,
        )
        self._set_gain()

    def _set_delay(self):
        """Set cal_type to 'delay' and adjust required parameters."""
        self.cal_type = "delay"
        self._gain_array.required = False
        self._delay_array.required = True
        self._freq_range.required = True
        self._quality_array.form = self._delay_array.form
        self._total_quality_array.form = self._delay_array.form[1:]

    def set_delay(self):
        """
        Set cal_type to 'delay' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_delay` instead.
        """
        warnings.warn(
            "`set_delay` is deprecated, and will be removed in pyuvdata version "
            "2.2. Use `_set_delay` instead.",
            DeprecationWarning,
        )
        self._set_delay()

    def _set_unknown_cal_type(self):
        """Set cal_type to 'unknown' and adjust required parameters."""
        self.cal_type = "unknown"
        self._gain_array.required = False
        self._delay_array.required = False
        self._freq_range.required = False
        self._quality_array.form = self._gain_array.form
        self._total_quality_array.form = self._gain_array.form[1:]

    def set_unknown_cal_type(self):
        """
        Set cal_type to 'unknown' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_unknown_cal_type` instead.
        """
        warnings.warn(
            "`set_unknown_cal_type` is deprecated, and will be removed in "
            "pyuvdata version 2.2. Use `_set_unknown_cal_type` instead.",
            DeprecationWarning,
        )
        self._set_unknown_cal_type()

    def _set_sky(self):
        """Set cal_style to 'sky' and adjust required parameters."""
        self.cal_style = "sky"
        self._sky_field.required = True
        self._sky_catalog.required = True
        self._ref_antenna_name.required = True

    def set_sky(self):
        """
        Set cal_style to 'sky' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_sky` instead.
        """
        warnings.warn(
            "`set_sky` is deprecated, and will be removed in "
            "pyuvdata version 2.2. Use `_set_sky` instead.",
            DeprecationWarning,
        )
        self._set_sky()

    def _set_redundant(self):
        """Set cal_style to 'redundant' and adjust required parameters."""
        self.cal_style = "redundant"
        self._sky_field.required = False
        self._sky_catalog.required = False
        self._ref_antenna_name.required = False

    def set_redundant(self):
        """
        Set cal_style to 'redundant' and adjust required parameters.

        This method is deprecated, and will be removed in pyuvdata v2.2. Use
        `_set_redundant` instead.
        """
        warnings.warn(
            "`set_redundant` is deprecated, and will be removed in "
            "pyuvdata version 2.2. Use `_set_redundant` instead.",
            DeprecationWarning,
        )
        self._set_redundant()

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

        cal_type = self._cal_type.value
        if cal_type is None:
            cal_type = "unknown"

        required_params = {
            "gain": ["gain_array", "flag_array", "quality_array"],
            "delay": ["delay_array", "flag_array", "quality_array"],
            "unknown": ["flag_array", "quality_array"],
        }

        for param_name in self._data_params:
            if param_name in required_params[cal_type]:
                getattr(self, "_" + param_name).required = not metadata_only

        return metadata_only

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
            joined before the lst_array exists on the UVData object.

        """
        if not background:
            self._set_lsts_helper()
            return
        else:
            proc = threading.Thread(target=self._set_lsts_helper)
            proc.start()
            return proc

    def check(self, check_extra=True, run_check_acceptability=True):
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
            output = data_array[self.ant2ind(key[0]), 0, :, :, :]
            if squeeze_pol and output.shape[-1] == 1:
                output = output[:, :, 0]
            return output
        elif len(key) == 2:
            # interpret as an antenna-pol pair
            return data_array[self.ant2ind(key[0]), 0, :, :, self.jpol2ind(key[1])]

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
            Gain solution of shape (Nfreqs, Ntimes, Npol) or (Nfreqs, Ntimes)
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
            Flags of shape (Nfreqs, Ntimes, Npol) or (Nfreqs, Ntimes)
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
            Qualities of shape (Nfreqs, Ntimes, Npol) or (Nfreqs, Ntimes)
            if jpol is not None or if squeeze_pol is True and Njones = 1.
        """
        return self._slice_array(
            self._parse_key(ant, jpol=jpol), self.quality_array, squeeze_pol=squeeze_pol
        )

    def convert_to_gain(
        self,
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
        elif self.cal_type == "delay":
            if delay_convention == "minus":
                conv = -1
            elif delay_convention == "plus":
                conv = 1
            else:
                raise ValueError('delay_convention can only be "minus" or "plus"')

            self.history += "  Converted from delays to gains using pyuvdata."

            phase_array = np.zeros(
                (self.Nants_data, self.Nspws, self.Nfreqs, self.Ntimes, self.Njones)
            )
            for si in range(self.Nspws):
                temp = (
                    conv
                    * 2
                    * np.pi
                    * np.dot(
                        self.delay_array[:, si, 0, :, :, np.newaxis],
                        self.freq_array[si, np.newaxis, :],
                    )
                )
                temp = np.transpose(temp, (0, 3, 1, 2))
                phase_array[:, si, :, :, :] = temp

            gain_array = np.exp(1j * phase_array)
            new_quality = np.repeat(
                self.quality_array[:, :, :, :, :], self.Nfreqs, axis=2
            )
            self._set_gain()
            self.gain_array = gain_array
            self.quality_array = new_quality
            self.delay_array = None
            if self.total_quality_array is not None:
                new_total_quality_array = np.repeat(
                    self.total_quality_array[:, :, :, :], self.Nfreqs, axis=1
                )
                self.total_quality_array = new_total_quality_array

            # check if object is self-consistent
            if run_check:
                self.check(
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                )
        else:
            raise ValueError("cal_type is unknown, cannot convert to gain")

    def __add__(
        self,
        other,
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

        # Check objects are compatible
        compatibility_params = [
            "_cal_type",
            "_integration_time",
            "_channel_width",
            "_telescope_name",
            "_gain_convention",
            "_x_orientation",
            "_cal_style",
            "_ref_antenna_name",
        ]
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
            both_freq = np.intersect1d(this.freq_array[0, :], other.freq_array[0, :])
        else:
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
            temp = np.nonzero(~np.in1d(other.freq_array[0, :], this.freq_array[0, :]))[
                0
            ]
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
                zero_pad_data = np.zeros(
                    (
                        len(anew_inds),
                        this.Nspws,
                        this.quality_array.shape[2],
                        this.Ntimes,
                        this.Njones,
                    )
                )
                zero_pad_flags = np.zeros(
                    (len(anew_inds), this.Nspws, this.Nfreqs, this.Ntimes, this.Njones)
                )
                if this.cal_type == "delay":
                    this.delay_array = np.concatenate(
                        [this.delay_array, zero_pad_data], axis=0
                    )[order, :, :, :, :]
                else:
                    this.gain_array = np.concatenate(
                        [this.gain_array, zero_pad_data], axis=0
                    )[order, :, :, :, :]
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad_flags], axis=0
                ).astype(np.bool_)[order, :, :, :, :]
                this.quality_array = np.concatenate(
                    [this.quality_array, zero_pad_data], axis=0
                )[order, :, :, :, :]

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
                    zero_pad = np.zeros(
                        (
                            len(anew_inds),
                            this.Nspws,
                            this.Nfreqs,
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    this.input_flag_array = np.concatenate(
                        [this.input_flag_array, 1 - zero_pad], axis=0
                    ).astype(np.bool_)[order, :, :, :, :]
                elif other.input_flag_array is not None:
                    zero_pad = np.zeros(
                        (
                            len(anew_inds),
                            this.Nspws,
                            this.Nfreqs,
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    this.input_flag_array = np.array(
                        1
                        - np.zeros(
                            (
                                this.Nants_data,
                                this.Nspws,
                                this.Nfreqs,
                                this.Ntimes,
                                this.Njones,
                            )
                        )
                    ).astype(np.bool_)
                    this.input_flag_array = np.concatenate(
                        [this.input_flag_array, 1 - zero_pad], axis=0
                    ).astype(np.bool_)[order, :, :, :, :]

        if len(fnew_inds) > 0:
            # Exploit the fact that quality array has the same dimensions as the
            # main data.
            # Also do not need to worry about different cases for gain v. delay type
            zero_pad = np.zeros(
                (
                    this.quality_array.shape[0],
                    this.Nspws,
                    len(fnew_inds),
                    this.Ntimes,
                    this.Njones,
                )
            )
            this.freq_array = np.concatenate(
                [this.freq_array, other.freq_array[:, fnew_inds]], axis=1
            )
            order = np.argsort(this.freq_array[0, :])
            this.freq_array = this.freq_array[:, order]
            if not self.metadata_only:
                this.gain_array = np.concatenate([this.gain_array, zero_pad], axis=2)[
                    :, :, order, :, :
                ]
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad], axis=2
                ).astype(np.bool_)[:, :, order, :, :]
                this.quality_array = np.concatenate(
                    [this.quality_array, zero_pad], axis=2
                )[:, :, order, :, :]

                if this.total_quality_array is not None and can_combine_tqa:
                    zero_pad = np.zeros(
                        (this.Nspws, len(fnew_inds), this.Ntimes, this.Njones)
                    )
                    this.total_quality_array = np.concatenate(
                        [this.total_quality_array, zero_pad], axis=1
                    )[:, order, :, :]
                elif other.total_quality_array is not None and can_combine_tqa:
                    zero_pad = np.zeros(
                        (this.Nspws, len(fnew_inds), this.Ntimes, this.Njones)
                    )
                    this.total_quality_array = np.zeros(
                        (this.Nspws, Nf_tqa, this.Ntimes, this.Njones)
                    )
                    this.total_quality_array = np.concatenate(
                        [this.total_quality_array, zero_pad], axis=1
                    )[:, order, :, :]

                if this.input_flag_array is not None:
                    zero_pad = np.zeros(
                        (
                            this.input_flag_array.shape[0],
                            this.Nspws,
                            len(fnew_inds),
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    this.input_flag_array = np.concatenate(
                        [this.input_flag_array, 1 - zero_pad], axis=2
                    ).astype(np.bool_)[:, :, order, :, :]
                elif other.input_flag_array is not None:
                    zero_pad = np.zeros(
                        (
                            this.flag_array.shape[0],
                            this.Nspws,
                            len(fnew_inds),
                            this.Ntimes,
                            this.Njones,
                        )
                    )
                    this.input_flag_array = np.array(
                        1
                        - np.zeros(
                            (
                                this.flag_array.shape[0],
                                this.Nspws,
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
            if not self.metadata_only:
                zero_pad_data = np.zeros(
                    (
                        this.quality_array.shape[0],
                        this.Nspws,
                        this.quality_array.shape[2],
                        len(tnew_inds),
                        this.Njones,
                    )
                )
                zero_pad_flags = np.zeros(
                    (
                        this.flag_array.shape[0],
                        this.Nspws,
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
                            this.Nspws,
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
                            this.Nspws,
                            this.quality_array.shape[2],
                            len(tnew_inds),
                            this.Njones,
                        )
                    )
                    this.total_quality_array = np.zeros(
                        (this.Nspws, Nf_tqa, this.Ntimes, this.Njones)
                    )
                    this.total_quality_array = np.concatenate(
                        [this.total_quality_array, zero_pad], axis=2
                    )[:, :, order, :]

                if this.input_flag_array is not None:
                    zero_pad = np.zeros(
                        (
                            this.input_flag_array.shape[0],
                            this.Nspws,
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
                            this.Nspws,
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
                                this.Nspws,
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
                zero_pad_data = np.zeros(
                    (
                        this.quality_array.shape[0],
                        this.Nspws,
                        this.quality_array.shape[2],
                        this.quality_array.shape[3],
                        len(jnew_inds),
                    )
                )
                zero_pad_flags = np.zeros(
                    (
                        this.flag_array.shape[0],
                        this.Nspws,
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
                            this.Nspws,
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
                            this.Nspws,
                            this.quality_array.shape[2],
                            this.quality_array.shape[3],
                            len(jnew_inds),
                        )
                    )
                    this.total_quality_array = np.zeros(
                        (this.Nspws, Nf_tqa, this.Ntimes, this.Njones)
                    )
                    this.total_quality_array = np.concatenate(
                        [this.total_quality_array, zero_pad], axis=3
                    )[:, :, :, order]

                if this.input_flag_array is not None:
                    zero_pad = np.zeros(
                        (
                            this.input_flag_array.shape[0],
                            this.Nspws,
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
                            this.Nspws,
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
                                this.Nspws,
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
            freqs_t2o = np.nonzero(
                np.in1d(this.freq_array[0, :], other.freq_array[0, :])
            )[0]
            ants_t2o = np.nonzero(np.in1d(this.ant_array, other.ant_array))[0]
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

        # Update N parameters (e.g. Npols)
        this.Njones = this.jones_array.shape[0]
        this.Ntimes = this.time_array.shape[0]
        if this.cal_type == "gain":
            this.Nfreqs = this.freq_array.shape[1]
        this.Nants_data = len(
            np.unique(this.ant_array.tolist() + other.ant_array.tolist())
        )

        # Check specific requirements
        if this.cal_type == "gain" and this.Nfreqs > 1:
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
            elif np.max(freq_separation) > this.channel_width:
                warnings.warn(
                    "Combined frequencies are not contiguous. This will make "
                    "it impossible to write this data out to some file types."
                )

        if this.Njones > 2:
            jones_separation = np.diff(this.jones_array)
            if np.min(jones_separation) < np.max(jones_separation):
                warnings.warn(
                    "Combined Jones elements are not evenly spaced. This will "
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
        times : array_like of float, optional
            The times to keep in the object, each value passed here should
            exist in the time_array.
        jones : array_like of int, optional
            The antenna polarizations numbers to keep in the object, each value
            passed here should exist in the jones_array.
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
                cal_object.flag_array = cal_object.flag_array[ant_inds, :, :, :, :]
                cal_object.quality_array = cal_object.quality_array[
                    ant_inds, :, :, :, :
                ]
                if cal_object.cal_type == "delay":
                    cal_object.delay_array = cal_object.delay_array[
                        ant_inds, :, :, :, :
                    ]
                else:
                    cal_object.gain_array = cal_object.gain_array[ant_inds, :, :, :, :]

                if cal_object.input_flag_array is not None:
                    cal_object.input_flag_array = cal_object.input_flag_array[
                        ant_inds, :, :, :, :
                    ]

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

            if cal_object.Ntimes > 1:
                time_separation = np.diff(cal_object.time_array)
                if not np.isclose(
                    np.min(time_separation),
                    np.max(time_separation),
                    rtol=cal_object._time_array.tols[0],
                    atol=cal_object._time_array.tols[1],
                ):
                    warnings.warn(
                        "Selected times are not evenly spaced. This "
                        "is not supported by the calfits format."
                    )

            if not self.metadata_only:
                cal_object.flag_array = cal_object.flag_array[:, :, :, time_inds, :]
                cal_object.quality_array = cal_object.quality_array[
                    :, :, :, time_inds, :
                ]
                if cal_object.cal_type == "delay":
                    cal_object.delay_array = cal_object.delay_array[
                        :, :, :, time_inds, :
                    ]
                else:
                    cal_object.gain_array = cal_object.gain_array[:, :, :, time_inds, :]

                if cal_object.input_flag_array is not None:
                    cal_object.input_flag_array = cal_object.input_flag_array[
                        :, :, :, time_inds, :
                    ]

                if cal_object.total_quality_array is not None:
                    cal_object.total_quality_array = cal_object.total_quality_array[
                        :, :, time_inds, :
                    ]

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if frequencies is None:
                frequencies = cal_object.freq_array[0, freq_chans]
            else:
                frequencies = uvutils._get_iterable(frequencies)
                frequencies = np.sort(
                    list(set(frequencies) | set(cal_object.freq_array[0, freq_chans]))
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
            freq_arr_use = cal_object.freq_array[0, :]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError(
                        "Frequency {f} is not present in the freq_array".format(f=f)
                    )

            freq_inds = sorted(set(freq_inds))
            cal_object.Nfreqs = len(freq_inds)
            cal_object.freq_array = cal_object.freq_array[:, freq_inds]

            if cal_object.Nfreqs > 1:
                freq_separation = (
                    cal_object.freq_array[0, 1:] - cal_object.freq_array[0, :-1]
                )
                if not np.isclose(
                    np.min(freq_separation),
                    np.max(freq_separation),
                    rtol=cal_object._freq_array.tols[0],
                    atol=cal_object._freq_array.tols[1],
                ):
                    warnings.warn(
                        "Selected frequencies are not evenly spaced. This "
                        "is not supported by the calfits format"
                    )

            if not self.metadata_only:
                cal_object.flag_array = cal_object.flag_array[:, :, freq_inds, :, :]
                if cal_object.cal_type == "delay":
                    pass
                else:
                    cal_object.quality_array = cal_object.quality_array[
                        :, :, freq_inds, :, :
                    ]
                    cal_object.gain_array = cal_object.gain_array[:, :, freq_inds, :, :]

                if cal_object.input_flag_array is not None:
                    cal_object.input_flag_array = cal_object.input_flag_array[
                        :, :, freq_inds, :, :
                    ]

                if cal_object.cal_type == "delay":
                    pass
                else:
                    if cal_object.total_quality_array is not None:
                        cal_object.total_quality_array = cal_object.total_quality_array[
                            :, freq_inds, :, :
                        ]

        if jones is not None:
            jones = uvutils._get_iterable(jones)
            if n_selects > 0:
                history_update_string += ", jones polarization terms"
            else:
                history_update_string += "jones polarization terms"
            n_selects += 1

            jones_inds = np.zeros(0, dtype=np.int64)
            for j in jones:
                if j in cal_object.jones_array:
                    jones_inds = np.append(
                        jones_inds, np.where(cal_object.jones_array == j)[0]
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
                if np.min(jones_separation) < np.max(jones_separation):
                    warnings.warn(
                        "Selected jones polarization terms are not evenly spaced. This "
                        "is not supported by the calfits format"
                    )

            if not self.metadata_only:
                cal_object.flag_array = cal_object.flag_array[:, :, :, :, jones_inds]
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

        """
        calfits_obj = self._convert_to_filetype("calfits")
        calfits_obj.write_calfits(
            filename,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            clobber=clobber,
        )
        del calfits_obj
