# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading MS calibration tables."""

import os
import warnings

import numpy as np
from astropy.time import Time
from docstring_parser import DocstringStyle

from .. import ms_utils
from ..docstrings import copy_replace_short_description
from .uvcal import UVCal

__all__ = ["MSCal"]

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
try:
    import casacore.tables as tables
except ImportError as error:  # pragma: no cover
    casa_present = False
    casa_error = error


class MSCal(UVCal):
    """
    Defines an MS-specific subclass of UVCal for reading MS calibration tables.

    This class should not be interacted with directly, instead use the read_ms_cal
    method on the UVCal class.
    """

    @copy_replace_short_description(UVCal.read_ms_cal, style=DocstringStyle.NUMPYDOC)
    def read_ms_cal(
        self,
        filepath,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        astrometry_library=None,
    ):
        """Read gains from an MS calibration table."""
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        if not os.path.exists(filepath):
            raise ValueError("No file found with the path %s." % filepath)

        # Set some initial things from the get go -- no legacy support
        self._set_flex_spw()
        self._set_future_array_shapes()

        # Use the utility function to verify this actually is an MS file
        ms_utils._ms_utils_call_checks(filepath)
        # True by fiat!
        self.flex_spw = self.future_array_shapes = True
        self.filename = filepath

        # get the history info from the ms_utils
        try:
            self.history = ms_utils.read_ms_hist(filepath, self.pyuvdata_version_str)
        except FileNotFoundError:
            self.history = self.pyuvdata_version_str

        tb_main = tables.table(filepath, ack=False)
        main_info_dict = tb_main.info()

        # 'Measurement Set' of type for main MS
        if main_info_dict["type"] == "Measurement Set":
            raise ValueError(
                "This seems to be a Measurement Set containing visibilities, "
                "not a calibration table."
            )

        if main_info_dict["subType"] == "G Jones":
            self._set_gain
            # This is a so-called "wideband" gains calibration table, i.e. not bandpass
            self.wide_band = True
            self.cal_type = "gain"
        elif main_info_dict["subType"] == "B Jones":
            self._set_gain
            # This is a bandpass solution
            self.wide_band = False
            self.cal_type = "gain"
        elif main_info_dict["subType"] == "K Jones":
            # This is a delay solution? Need to understand the units...
            self.wide_band = True
            self.cal_type = "delay"
            raise NotImplementedError("No support yet for delay solutions.")
        else:
            warnings.warn(
                "Cannot recognize solution type, treating as wideband gain solutions."
            )
            self.wide_band = True

        main_keywords = tb_main.getkeywords()
        for keyword in ["CASA_Version", "MSName"]:
            if keyword in main_keywords:
                self.extra_keywords[keyword] = main_keywords[keyword]

        # tb_field = tables.table(filepath + "/FIELD", ack=False)

        # open table with antenna location information
        ant_info = ms_utils.read_ms_ant(filepath)
        obs_info = ms_utils.read_ms_obs(filepath)

        self.observer = obs_info["observer"]
        self.telescope_name = obs_info["telescope_name"]
        self._telescope_location.frame = ant_info["telescope_frame"]

        # check to see if a TELESCOPE_LOCATION column is present in the observation
        # table. This is non-standard, but inserted by pyuvdata
        if "telescope_location" in obs_info:
            self.telescope_location = obs_info["telescope_location"]
        else:
            # get it from known telescopes
            try:
                self.set_telescope_params()
            except ValueError:
                # If no telescope is found, the we will set the telescope position to be
                # the mean of the antenna positions (this is not ideal!)
                self.telescope_location = np.mean(ant_info["antenna_positions"], axis=0)

        self.antenna_names = ant_info["antenna_names"]
        self.Nants_telescope = len(self.antenna_names)
        self.antenna_numbers = ant_info["antenna_numbers"]
        # MS-format seems to want to preserve the blank entries in the gains tables
        # This looks to be the same for MS files.
        self.ant_array = self.antenna_numbers
        self.Nants_data = self.Nants_telescope

        self.antenna_positions = ant_info["antenna_positions"]
        # Subtract off telescope location to get relative ECEF
        self.antenna_positions -= self.telescope_location.reshape(1, 3)

        # importuvfits measurement sets store antenna names in the STATION column.
        # cotter measurement sets store antenna names in the NAME column, which is
        # inline with the MS definition doc. In that case all the station names are
        # the same. Default to using what the MS definition doc specifies, unless
        # we read importuvfits in the history, or if the antenna column is not filled.
        if self.Nants_telescope != len(np.unique(self.antenna_names)) or (
            "" in self.antenna_names
        ):
            self.antenna_names = ant_info["station_names"]

        # Extract out the reference antenna information.
        ref_ant_array = tb_main.getcol("ANTENNA2")
        if not all(ref_ant_array[0] == ref_ant_array):
            warnings.warn(
                "Multiple ref ants detected, which UVCal cannot handle. Using the "
                "first entry as the default"
            )

        self.ref_antenna_name = self.antenna_names[
            np.where(self.antenna_numbers == ref_ant_array[0])[0][0]
        ]

        spw_info = ms_utils.read_ms_spw(filepath)

        self.spw_array = spw_info["assoc_spw_id"]
        self.Nspws = len(self.spw_array)
        self.Nfreqs = sum(spw_info["num_chan"])

        self.freq_array = np.zeros(self.Nfreqs, dtype=float)
        self.channel_width = np.zeros(self.Nfreqs, dtype=float)
        self.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)
        spw_slice_dict = {}
        for spw_id, spw_idx, spw_nchan, spw_end_chan, spw_freqs, spw_chan_width in zip(
            self.spw_array,
            spw_info["row_idx"],
            spw_info["num_chan"],
            np.cumsum(spw_info["num_chan"]),
            spw_info["chan_freq"],
            spw_info["chan_width"],
        ):
            spw_slice = slice(spw_end_chan - spw_nchan, spw_end_chan)
            spw_slice_dict[spw_idx] = spw_slice
            self.freq_array[spw_slice] = spw_freqs
            self.channel_width[spw_slice] = spw_chan_width
            self.flex_spw_id_array[spw_slice] = spw_id

        # Don't think CASA is going to support a non-sky based model, but hey, who knows
        self.cal_style = "sky"
        self.gain_convention = "multiply"  # Need to verify this or "divide"
        self.time_array = tb_main.getcol("TIME")
        self.Nsources = len(np.unique(tb_main.getcol("FIELD_ID")))

        # Just assume that the gain scale is always in Jy
        self.gain_scale = "Jy"
        if self.wide_band:
            self.freq_range = np.vstack(
                (
                    self.freq_array - (self.channel_width / 2),
                    self.freq_array + (self.channel_width / 2),
                )
            ).T
        self.sky_catalog = "CASA (import)"

        # MAIN LOOP
        self.Njones = tb_main.getcell("CPARAM", 0).shape[1]
        if main_keywords["PolBasis"].lower() == "unknown":
            warnings.warn(
                "Unknown polarization basis for solutions, jones_array values "
                "may be spurious."
            )
            self.jones_array = np.zeros(self.Njones, dtype=int)
        else:
            raise NotImplementedError("Not sure how to read this file yet...")

        time_dict = {}
        for time in tb_main.getcol("TIME"):
            try:
                _ = time_dict[time]
            except KeyError:
                # Check to see if there are any nearby entries, accounting for the fact
                # that MS stores times in seconds and time_array tolerances are
                # specified in days.
                close_check = np.isclose(
                    list(time_dict),
                    time,
                    rtol=self._time_array.tols[0] * 86400,
                    atol=self._time_array.tols[1] * 86400,
                )
                if any(close_check):
                    # Fill in the first closest entry matched
                    time_dict[time] = np.where(close_check)[0][0]
                else:
                    # Otherwise, plug in a new entry
                    time_dict[time] = len(time_dict)

        time_arr = np.array(list(time_dict))
        self.time_array = Time(
            time_arr / 86400.0, format="mjd", scale=ms_utils.read_time_scale(tb_main)
        ).utc.jd
        self.integration_time = np.zeros_like(time_arr)

        self.Ntimes = len(self.time_array)

        # Make a map to things.
        ant_dict = {ant: idx for idx, ant in enumerate(self.antenna_numbers)}
        cal_arr_shape = (self.Nants_data, self.Nfreqs, self.Ntimes, self.Njones)
        ms_cal_soln = np.zeros(cal_arr_shape, dtype=complex)
        self.quality_array = np.zeros(cal_arr_shape, dtype=float)
        self.flag_array = np.ones(cal_arr_shape, dtype=bool)
        self.total_quality_array = None  # Always None for now, no similar array in MS

        for idx in range(tb_main.nrows()):
            try:
                time_idx = time_dict[tb_main.getcell("TIME", idx)]
                ant_idx = ant_dict[tb_main.getcell("ANTENNA1", idx)]
                cal_soln = tb_main.getcell("CPARAM", idx)
                cal_qual = tb_main.getcell("PARAMERR", idx)
                cal_flag = tb_main.getcell("FLAG", idx)

                int_time = tb_main.getcell("INTERVAL", idx)
                self.integration_time[time_idx] = int_time

                spw_slice = spw_slice_dict[tb_main.getcell("SPECTRAL_WINDOW_ID", idx)]

                ms_cal_soln[ant_idx, spw_slice, time_idx, :] = cal_soln
                self.quality_array[ant_idx, spw_slice, time_idx, :] = cal_qual
                self.flag_array[ant_idx, spw_slice, time_idx, :] = cal_flag
            except KeyError:
                # If there's no entry that matches, it's because we've effectively
                # flagged some index value such that it has no entries in the table.
                # skip recording this row.
                continue

        # I think this is always east.
        self.x_orientation = "east"
        # Use if this is a delay soln
        if self.cal_type == "gain":
            self.gain_array = ms_cal_soln
        elif self.cal_type == "delay":
            self.delay_array = ms_cal_soln

        self.set_lsts_from_time_array(astrometry_library=astrometry_library)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def write_ms_cal(self, filename):
        """Write out a MS calibration table."""
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        raise NotImplementedError("Not implemented yet!")
