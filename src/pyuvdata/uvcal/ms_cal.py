# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading MS calibration tables."""

import os
import warnings

import numpy as np
from astropy.time import Time
from docstring_parser import DocstringStyle

from .. import utils
from ..docstrings import copy_replace_short_description
from ..utils.io import ms as ms_utils
from . import UVCal

__all__ = ["MSCal"]

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
try:
    from casacore import tables
except ImportError as error:
    casa_present = False
    casa_error = error


DEFAULT_CAT_DICT = {
    0: {
        "cat_name": "unknown",
        "cat_type": "sidereal",
        "cat_lon": -1.0,
        "cat_lat": -1.0,
        "cat_frame": "icrs",
        "cat_epoch": 2000.0,
        "cat_times": None,
        "cat_pm_ra": None,
        "cat_pm_dec": None,
        "cat_vrad": None,
        "cat_dist": None,
        "info_source": "file",
    }
}


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
        *,
        default_x_orientation=None,
        default_jones_array=None,
        default_mount_type="other",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        astrometry_library=None,
    ):
        """Read in an MS-formatted gains table."""
        # Use the utility function to verify this actually is an MS file
        ms_utils._ms_utils_call_checks(filepath)

        self.filename = [os.path.basename(filepath)]
        self._filename.form = (1,)

        # get the history info from the ms_utils
        self.history = ms_utils.read_ms_history(
            filepath, self.pyuvdata_version_str, raise_err=False
        )

        tb_main = tables.table(filepath, ack=False)
        main_info_dict = tb_main.info()

        # 'Measurement Set' of type for main MS
        if main_info_dict["type"] == "Measurement Set":
            raise ValueError(
                "This seems to be a Measurement Set containing visibilities, "
                "not a calibration table."
            )

        casa_subtype = main_info_dict["subType"]

        if casa_subtype in ["G Jones", "D Jones", "T Jones"]:
            # This is a so-called "wideband" gains calibration table, i.e. not bandpass
            self._set_wide_band(True)
            self._set_gain()
        elif casa_subtype == "B Jones":
            # This is a bandpass solution
            self._set_wide_band(False)
            self._set_gain()
        elif casa_subtype == "K Jones":
            # This is a delay solution
            self._set_wide_band(True)
            self._set_delay()
        else:
            # I don't know what this is, so don't proceed any further.
            raise NotImplementedError(  # pragma: no cover
                f"Calibration type {main_info_dict['subType']} is not "
                "recognized/supported by UVCal. Please file an issue in our "
                "GitHub issue log so that we can add support for it."
            )

        par_type = tb_main.getkeyword("ParType")
        if par_type == "Complex":
            cal_column = "CPARAM"
        elif par_type == "Float":
            cal_column = "FPARAM"
        else:
            raise NotImplementedError(  # pragma: no cover
                f"Parameter type {par_type} is not recognized/supported by "
                "UVCal. Please file an issue in our GitHub issue log so that we "
                "can add support for it."
            )

        main_keywords = tb_main.getkeywords()
        for keyword in ["CASA_Version", "MSName"]:
            if keyword in main_keywords:
                self.extra_keywords[keyword] = main_keywords[keyword]

        # open table with antenna location information
        ant_info = ms_utils.read_ms_antenna(filepath)
        obs_info = ms_utils.read_ms_observation(filepath)

        self.observer = obs_info["observer"]
        self.telescope.name = obs_info["telescope_name"]
        self.telescope.location = ms_utils.get_ms_telescope_location(
            tb_ant_dict=ant_info, obs_dict=obs_info
        )

        self.telescope.antenna_names = ant_info["antenna_names"]
        self.telescope.Nants = len(self.telescope.antenna_names)
        self.telescope.antenna_numbers = ant_info["antenna_numbers"]
        self.telescope.mount_type = ant_info["antenna_mount"]
        self.telescope.antenna_diameters = ant_info["antenna_diameters"]
        # MS-format seems to want to preserve the blank entries in the gains tables
        # This looks to be the same for MS files.
        self.ant_array = self.telescope.antenna_numbers
        self.Nants_data = self.telescope.Nants

        self.telescope.antenna_positions = ant_info["antenna_positions"]
        # Subtract off telescope location to get relative ECEF
        self.telescope.antenna_positions -= self.telescope._location.xyz().reshape(1, 3)
        self.phase_center_catalog, field_id_map = ms_utils.read_ms_field(
            filepath, return_phase_center_catalog=True
        )
        self.Nphase = len(np.unique(tb_main.getcol("FIELD_ID")))
        if (self.phase_center_catalog == DEFAULT_CAT_DICT) and (self.Nphase == 1):
            # If this is the default, we know that this was spoofed by pyuvdata, in
            # which case we'll set the phase setuff to None
            self.phase_center_catalog = None
            self.Nphase = None

        # importuvfits measurement sets store antenna names in the STATION column.
        # cotter measurement sets store antenna names in the NAME column, which is
        # inline with the MS definition doc. In that case all the station names are
        # the same. Default to using what the MS definition doc specifies, unless
        # we read importuvfits in the history, or if the antenna column is not filled.
        if self.telescope.Nants != len(np.unique(self.telescope.antenna_names)) or (
            "" in self.telescope.antenna_names
        ):
            self.telescope.antenna_names = ant_info["station_names"]

        spw_info = ms_utils.read_ms_spectral_window(filepath)

        self.spw_array = np.array(spw_info["assoc_spw_id"])
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
            strict=True,
        ):
            spw_slice = slice(spw_end_chan - spw_nchan, spw_end_chan)
            spw_slice_dict[spw_idx] = spw_slice
            self.freq_array[spw_slice] = spw_freqs
            self.channel_width[spw_slice] = spw_chan_width
            self.flex_spw_id_array[spw_slice] = spw_id

        self.gain_convention = "divide"  # N.b., manually verified by Karto in CASA v6.4

        # Just assume that the gain scale is always in Jy
        nchan = self.Nfreqs

        if self.wide_band:
            self.freq_range = np.vstack(
                (
                    self.freq_array - (self.channel_width / 2),
                    self.freq_array + (self.channel_width / 2),
                )
            ).T
            self.freq_array = None
            self.channel_width = None
            self.flex_spw_id_array = None
            self.Nfreqs = 1

        # MAIN LOOP
        self.Njones = tb_main.getcell(cal_column, 0).shape[1]
        if main_keywords["PolBasis"].lower() == "unknown":
            self.jones_array = main_keywords.get("pyuvdata_jones", default_jones_array)
            self.flex_jones_array = main_keywords.get("pyuvdata_flex_jones", None)
            if self.jones_array is None:
                warnings.warn(
                    "Unknown polarization basis for solutions, jones_array values "
                    "may be spurious."
                )
                self.jones_array = np.array(
                    [-7, -8] if (casa_subtype == "D Jones") else [-5, -6]
                )
        else:
            raise NotImplementedError(  # pragma: no cover
                "Polarization basis {} is not recognized/supported by UVCal. Please "
                "file an issue in our GitHub issue log so that we can add support for "
                "it.".format(main_keywords["PolBasis"])
            )

        if casa_subtype == "D Jones":
            if any(item not in [-3, -4, -7, -8, 0] for item in self.jones_array):
                warnings.warn(
                    "Cross-handed Jones terms expected but jones_array contains "
                    f"same-handed terms ({casa_subtype} subtype), use caution."
                )
        elif any(item in [-3, -4, -7, -8] for item in self.jones_array):
            warnings.warn(
                "Same-handed Jones terms expected but jones_array contains "
                f"cross-handed terms ({casa_subtype} subtype), use caution."
            )

        self.sky_catalog = main_keywords.get("pyuvdata_sky_catalog", None)
        self.gain_scale = main_keywords.get("pyuvdata_gain_scale", None)
        self.pol_convention = main_keywords.get("pyuvdata_polconv", None)
        self.observer = main_keywords.get("pyuvdata_observer", None)
        if "pyuvdata_cal_style" in main_keywords:
            self.cal_style = main_keywords["pyuvdata_cal_style"]
            if self.cal_style == "sky":
                self._set_sky()
            elif self.cal_style == "redundant":
                self._set_redundant()
        else:
            self.sky_catalog = "CASA (import)"
            self._set_sky()

        time_dict = {}
        row_timeidx_map = []
        time_count = 0
        for time in tb_main.getcol("TIME"):
            try:
                row_timeidx_map.append(time_dict[time])
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
                    time_dict[time] = time_count
                    time_count += 1
                # Finally, update the row_map with the correct value
                row_timeidx_map.append(time_dict[time])

        self.time_array = np.zeros(time_count, dtype=float)
        self.integration_time = np.zeros(time_count, dtype=float)
        self.Ntimes = time_count

        # Make a map to things.
        ant_dict = {ant: idx for idx, ant in enumerate(self.telescope.antenna_numbers)}
        cal_arr_shape = (self.Nants_data, nchan, self.Ntimes, self.Njones)

        ms_cal_soln = np.zeros(
            cal_arr_shape, dtype=complex if (self.cal_type == "gain") else float
        )
        self.quality_array = np.zeros(cal_arr_shape, dtype=float)
        self.flag_array = np.ones(cal_arr_shape, dtype=bool)
        self.total_quality_array = None  # Always None for now, no similar array in MS
        self.scan_number_array = np.zeros_like(self.time_array, dtype=int)
        self.phase_center_id_array = np.zeros_like(self.time_array, dtype=int)
        self.ref_antenna_array = np.zeros_like(self.time_array, dtype=int)
        has_exp = "EXPOSURE" in tb_main.colnames()
        exp_time = 0.0  # Default value if no exposure stored
        int_arr = np.zeros_like(self.time_array, dtype=float)

        for row_idx, time_idx in enumerate(row_timeidx_map):
            try:
                ant_idx = ant_dict[tb_main.getcell("ANTENNA1", row_idx)]
                ref_ant = tb_main.getcell("ANTENNA2", row_idx)
                time_val = tb_main.getcell("TIME", row_idx)
                cal_soln = tb_main.getcell(cal_column, row_idx)
                cal_qual = tb_main.getcell("PARAMERR", row_idx)
                cal_flag = tb_main.getcell("FLAG", row_idx)
                field_id = tb_main.getcell("FIELD_ID", row_idx)
                scan_num = tb_main.getcell("SCAN_NUMBER", row_idx)
                int_time = tb_main.getcell("INTERVAL", row_idx)
                spw_id = tb_main.getcell("SPECTRAL_WINDOW_ID", row_idx)
                if has_exp:
                    exp_time = tb_main.getcell("EXPOSURE", row_idx)

                # Figure out which spectral slice this corresponds to
                spw_slice = spw_slice_dict[spw_id]

                # Finally, start plugging in solns to various parameters. Note that
                # because of the conjugation scheme, normally we'd have to flip this for
                # CASA, except that the Antenna1 entries appear to be "pre-conjugated",
                # and thus no flip is necessary for gains solns.
                # TODO: Verify this is the case for delay solns as well.
                ms_cal_soln[ant_idx, spw_slice, time_idx, :] = cal_soln
                self.time_array[time_idx] = time_val
                self.integration_time[time_idx] = exp_time
                int_arr[time_idx] = int_time
                self.quality_array[ant_idx, spw_slice, time_idx, :] = cal_qual
                self.flag_array[ant_idx, spw_slice, time_idx, :] = cal_flag
                self.phase_center_id_array[time_idx] = field_id
                self.scan_number_array[time_idx] = scan_num
                self.ref_antenna_array[time_idx] = ref_ant
            except KeyError:
                # If there's no entry that matches, it's because we've effectively
                # flagged some index value such that it has no entries in the table.
                # skip recording this row.
                continue

        if len(np.unique(self.ref_antenna_array)) > 1:
            self.ref_antenna_name = "various"
        else:
            # if there aren't multiple ref ants, default to storing just the one name.
            refant = self.ref_antenna_array[0]
            self.ref_antenna_array = None
            try:
                self.ref_antenna_name = self.telescope.antenna_names[
                    np.where(self.telescope.antenna_numbers == refant)[0][0]
                ]
            except IndexError:
                if self.cal_style == "sky":
                    self.ref_antenna_name = "unknown reference antenna"

        # Convert the time from MJD secs (CASA standard) to JD date (pyuvdata std)
        self.time_array = Time(
            self.time_array / 86400.0,
            format="mjd",
            scale=ms_utils._get_time_scale(tb_main),
        ).utc.jd

        if not all(int_arr == 0.0):
            # If intervals have been identified, that means that we want to make our
            # solutions have time ranges rather than fixed times. solve this now.
            self.time_range = np.zeros((len(self.time_array), 2), dtype=float)
            self.time_range[:, 0] = self.time_array - (int_arr / (2 * 86400))
            self.time_range[:, 1] = self.time_array + (int_arr / (2 * 86400))
            self.time_array = None

        if self.Nphase is None:
            # If we have no catalog, then blank out the ID array here, since it
            # contains no actually useful info.
            self.phase_center_id_array = None
        elif len(field_id_map) != 0:
            # There's a little bit of cleanup to do w/ field_id, since the values here
            # correspond to the row in FIELD rather than the source ID. Map that now.
            self.phase_center_id_array = np.array(
                [field_id_map[idx] for idx in self.phase_center_id_array]
            )

        x_orientation = None
        if "pyuvdata_nfeeds" in main_keywords:
            self.telescope.Nfeeds = main_keywords["pyuvdata_nfeeds"]
            self.telescope.feed_array = (
                np.array([main_keywords["pyuvdata_feed_array"]])
                .view("U1")
                .reshape(-1, self.telescope.Nfeeds)
            )
            self.telescope.feed_angle = main_keywords["pyuvdata_feed_angle"]
        else:
            x_orientation = main_keywords.get("pyuvdata_xorient", default_x_orientation)
            if x_orientation is None:
                x_orientation = "east"
                warnings.warn(
                    'Unknown x_orientation basis for solutions, assuming "east".'
                )

        # Use if this is a delay soln
        if self.cal_type == "gain":
            self.gain_array = ms_cal_soln
        elif self.cal_type == "delay":
            # Delays are stored in nanoseconds -- convert to seconds (std for UVCal)
            self.delay_array = ms_cal_soln * 1e-9

        if casa_subtype == "T Jones":
            self.Njones = 2
            for name, param in zip(
                self._data_params, self.data_like_parameters, strict=True
            ):
                if param is not None:
                    setattr(self, name, np.repeat(param, 2, axis=-1))

        elif (
            (self.Njones == 2)
            and (len(self.jones_array) == 1)
            and (np.all(self.flag_array[..., 0]) or np.all(self.flag_array[..., 1]))
        ):
            # Capture a "corner" case where, because CASA always wants 2-elements
            # across the Jones-axis for G Jones subtype, there's extra padding along
            # that axis when Njones == 1.
            self.Njones = 1
            good_idx = int(np.all(self.flag_array[..., 0]))
            for name, param in zip(
                self._data_params, self.data_like_parameters, strict=True
            ):
                if param is not None:
                    setattr(self, name, param[..., good_idx : good_idx + 1])

        self.set_lsts_from_time_array(astrometry_library=astrometry_library)

        # Skip check since we're going to run it below
        self.set_telescope_params(
            x_orientation=x_orientation, mount_type=default_mount_type, run_check=False
        )

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    @copy_replace_short_description(UVCal.write_ms_cal, style=DocstringStyle.NUMPYDOC)
    def write_ms_cal(self, filename, clobber=False):
        """Write out a MS calibration table."""
        try:
            ms_utils._ms_utils_call_checks(filename, invert_check=True)
        except FileExistsError as err:
            if not clobber:
                raise FileExistsError(
                    "File already exists, must set clobber=True to proceed."
                ) from err

        if len(self.jones_array) > 2:
            # I don't know if this is ever possible, but at least thusfar in testing,
            # CASA always wants to see Njones == 2. Easy enough to pad when Njones == 1,
            # but we risk losing information when Njones > 2.
            raise ValueError("CASA MS calibration tables cannot support Njones > 2.")

        if self.gain_convention != "divide":
            raise ValueError(
                'MS writer only supports UVCal objects with gain_convention="divide".'
            )

        if self.cal_type not in ["gain", "delay"]:
            raise ValueError('cal_type must either be "gain" or "delay".')

        has_cross_jones = any(item in [-3, -4, -7, -8] for item in self.jones_array)
        has_flex_jones = self.flex_jones_array is not None
        if has_cross_jones:
            if any(item in [-1, -2, -5, -6] for item in self.jones_array):
                raise ValueError(
                    "CASA MSCal tables cannot store cross-hand and same-hand Jones "
                    "terms together, use select to separate them."
                )
            if self.cal_type == "gain" and np.any(abs(self.gain_array) > 1):
                warnings.warn(
                    "CASA MSCal tables store cross-handed Jones terms as leakages "
                    "(i.e., d-terms), which are recorded as a fractional complex "
                    "quantities that are separated from the typical (non-polarimetric) "
                    "gain phases and amplitudes."
                )

        # Initialize our calibration file, and get things set up with the appropriate
        # columns for us to write to in the main table. This is a little different
        # Depending on whether the table is a gains or delays file (complex vs floats).
        ms_utils.init_ms_cal_file(filename, delay_table=(self.cal_type == "delay"))

        # There's a little bit of extra handling required here for the different gains
        # types, which are encoded in the first letter of the subtype name. Best docs
        # for this actually exist from AIPS++ days, which can be found at:
        # https://casa.nrao.edu/aips2_docs/user/SynthesisRef/node27.html#calibrater
        #
        # Short summary here:
        # G == Ant-based, multiplicative complex gains
        #       --> In UVCal terms, wide_band gains
        # B == Ant-based complex bandpass; a frequency-dependent version of G
        # T == Ant-based pol-independent (tropospheric) gain; a pol-indep version of G
        #       --> This basically means Njones is forced to 1
        # D == Ant-based instrumental polarization leakage
        # M == Baseline-based complex gains; baseline-based version of G (blech)
        # MF == Baseline-based complex bandpass: baseline-based version of B (2x blech)
        # K == Ant-based delays (above suggests bsl-based, but CASA 6.x says ant-based)

        if self.cal_type == "gain":
            if has_cross_jones:
                casa_subtype = "D Jones"
            elif self.wide_band:
                casa_subtype = "T Jones" if has_flex_jones else "G Jones"
            else:
                casa_subtype = "B Jones"
            cal_column = "CPARAM"
            cal_array = self.gain_array
        elif self.cal_type == "delay":
            casa_subtype = "K Jones"
            cal_column = "FPARAM"
            # Convert from pyuvdata pref'd seconds to CASA-pref'd nanoseconds
            cal_array = self.delay_array * 1e9

        with tables.table(filename, ack=False, readonly=False) as ms:
            # Update the top-level info with the correct gains subtype.
            info_dict = ms.info()
            info_dict["subType"] = casa_subtype
            ms.putinfo(info_dict)

            # Update the keywords that we know
            ms.putkeyword("VisCal", casa_subtype)
            if len(self.extra_keywords) != 0:
                extra_copy = self.extra_keywords.copy()
                for key in ("CASA_Version", "MSName"):
                    if key in self.extra_keywords:
                        ms.putkeyword(key, extra_copy.pop(key))
                if len(extra_copy) != 0:
                    ms.putkeyword("pyuvdata_extra", extra_copy)

            ms.putkeyword("pyuvdata_jones", self.jones_array)
            if has_flex_jones:
                ms.putkeyword("pyuvdata_flex_jones", self.flex_jones_array)

            if self.sky_catalog is not None:
                ms.putkeyword("pyuvdata_sky_catalog", self.sky_catalog)

            if self.gain_scale is not None:
                ms.putkeyword("pyuvdata_gain_scale", self.gain_scale)

            if self.pol_convention is not None:
                ms.putkeyword("pyuvdata_polconv", self.pol_convention)

            if self.observer is not None:
                ms.putkeyword("pyuvdata_observer", self.observer)

            if self.cal_style is not None:
                ms.putkeyword("pyuvdata_cal_style", self.cal_style)

            if self.telescope.Nfeeds is not None:
                ms.putkeyword("pyuvdata_nfeeds", self.telescope.Nfeeds)
                # Compress the array of strings down to a single string, just to avoid
                # any weirdness with casacore handling
                ms.putkeyword(
                    "pyuvdata_feed_array", "".join(self.telescope.feed_array.flat)
                )
                ms.putkeyword("pyuvdata_feed_angle", self.telescope.feed_angle)

            # Now start the heavy lifting of putting in the data.
            ############################################################################
            # astropy's Time has some overheads associated with it, so use unique to run
            # this date conversion as few times as possible. Note that the default for
            # MS is MJD UTC seconds, versus JD UTC days for UVData.
            if self.time_array is None:
                time_array = np.mean(self.time_range, axis=1)
                interval_array = np.diff(self.time_range, axis=1) * 86400
            else:
                time_array = self.time_array
                interval_array = np.zeros_like(time_array)

            time_array = Time(time_array, format="jd", scale="utc").mjd * 86400.0
            exposure_array = self.integration_time

            # For some reason, CASA seems to want to pad the main table with zero
            # entries for the "blank" antennas, similar to what's seen in the ANTENNA
            # table. So we'll calculate this up front for convenience.
            Nants_casa = np.max(self.telescope.antenna_numbers) + 1

            # Add all the rows we need up front, which will allow us to fill the
            # columns all in one shot.
            ms.addrows(self.Ntimes * self.Nspws * Nants_casa)

            # Assume we have _more_ than one spectral window, where each needs to be
            # handled  separately, since they can have differing numbers of channels.
            # (n.b., tables.putvarcol can write complex tables like these, but its
            # slower and more memory-intensive than putcol).
            spw_sel_dict = {}
            for spw_id in self.spw_array:
                if self.wide_band:
                    spw_selection = np.equal(self.spw_array, spw_id)
                else:
                    spw_selection = np.equal(self.flex_spw_id_array, spw_id)
                spw_nchan = sum(spw_selection)
                spw_selection = utils.tools.slicify(
                    np.nonzero(spw_selection)[0], allow_empty=True
                )
                spw_sel_dict[spw_id] = (spw_selection, spw_nchan)

            # Based on some analysis of ALMA/ACA data, various routines in CASA appear
            # to prefer data be grouped together on a "per-scan" basis, then per-spw,
            # and then the more usual selections of per-time, per-ant1, etc.

            ant_array = np.tile(np.arange(Nants_casa), self.Ntimes * self.Nspws)
            if self.ref_antenna_array is None:
                try:
                    # Cast list here to deal w/ ndarrays
                    refant = str(
                        self.telescope.antenna_numbers[
                            list(self.telescope.antenna_names).index(
                                self.ref_antenna_name
                            )
                        ]
                    )
                except ValueError:
                    # We don't know what the refant was, so mark this accordingly
                    refant = -1

                refant_array = np.full_like(ant_array, refant)
            else:
                refant_array = np.tile(
                    np.repeat(self.ref_antenna_array, Nants_casa), self.Nspws
                )

            # Time-based properties need to be repeated on Nants_casa, since the antenna
            # axis is the "fastest" moving on the collapsed data. So repeat each entry
            # by the number of antennas, before tiling the whole thing by the number of
            # spectral windows (the preferred outer-most axis).
            interval_array = np.tile(np.repeat(interval_array, Nants_casa), self.Nspws)
            time_array = np.tile(np.repeat(time_array, Nants_casa), self.Nspws)
            exposure_array = np.tile(np.repeat(exposure_array, Nants_casa), self.Nspws)

            # Move on to the time-based optional parameters
            if self.scan_number_array is None:
                scan_number_array = np.zeros_like(time_array, dtype=int)
            else:
                scan_number_array = np.tile(
                    np.repeat(self.scan_number_array, Nants_casa), self.Nspws
                )

            if self.phase_center_id_array is None:
                field_ids = np.zeros_like(time_array, dtype=int)
            else:
                # We have to do an extra bit of work here, as he ID number needs to
                # match to the row number in the FIELD table). We do that by looking at
                # the ordering of phase_center_catalog, which is what's used to write
                # out the FIELD table.
                field_ids = np.zeros_like(self.phase_center_id_array)
                for idx, cat_id in enumerate(self.phase_center_catalog):
                    field_ids[self.phase_center_id_array == cat_id] = idx
                field_ids = np.tile(np.repeat(field_ids, Nants_casa), self.Nspws)

            # spw ids are on the outer most axis, so they just need to be repeated.
            spw_id_array = np.repeat(np.arange(self.Nspws), Nants_casa * self.Ntimes)

            # Set this always to zero for now -- not sure when its needed yet.
            obs_id_array = np.zeros_like(spw_id_array)

            # Calculate the number of times we need to duplicate records b/c of the way
            # that CASA stores information.
            ms.putcol("TIME", time_array)
            ms.putcol("FIELD_ID", field_ids)
            ms.putcol("SPECTRAL_WINDOW_ID", spw_id_array)
            ms.putcol("ANTENNA1", ant_array)
            ms.putcol("ANTENNA2", refant_array)
            ms.putcol("INTERVAL", interval_array)
            ms.putcol("EXPOSURE", exposure_array)
            ms.putcol("SCAN_NUMBER", scan_number_array)
            ms.putcol("OBSERVATION_ID", obs_id_array)

            # Determine polarization order for writing out in CASA standard order, check
            # if this order can be represented by a single slice.
            pol_order = utils.pol.determine_pol_order(self.jones_array, order="CASA")
            pol_order = utils.tools.slicify(pol_order, allow_empty=True)

            # Alright, all the easy stuff is over, time to move on to the the heavy
            # lifting, which we'll do spectral window by spectral window.
            for count, spw_id in enumerate(self.spw_array):
                spw_sel, spw_nchan = spw_sel_dict[spw_id]
                subarr_shape = (self.Ntimes, Nants_casa, spw_nchan, self.Njones)

                qual_arr = np.ones(
                    (self.Nants_data, spw_nchan, self.Ntimes, self.Njones), dtype=float
                )
                if self.quality_array is not None:
                    qual_arr *= self.quality_array[:, spw_sel, :, :]
                if self.total_quality_array is not None:
                    qual_arr *= self.total_quality_array[0][spw_sel, :, :]

                # We're going to leave a placeholder for SNR for now, since it's
                # somewhat redundant and not totally clear how it's calculated.
                data_dict = {
                    cal_column: cal_array[:, spw_sel, :, :],
                    "FLAG": self.flag_array[:, spw_sel, :, :],
                    "PARAMERR": qual_arr,
                    "SNR": np.zeros_like(qual_arr),
                }
                # N.b. (Karto): WEIGHT was totally unfilled in every example file I
                # poked at that was generated via CASA. It's possible it's not actually
                # used for anything or is otherwise vestigial, so leave it be for now.

                for item in data_dict:
                    subarr = data_dict[item]
                    # The data out of pyuvdata is ordered as (Ant, Freq, Time, Jones),
                    # but CASA expects (Time, Ants, Freq, Jones), so reorder accordingly
                    subarr = np.transpose(subarr, [2, 0, 1, 3])

                    # Set zeros by default ()
                    new_subarr = np.zeros(subarr_shape, dtype=subarr.dtype)
                    if item == "FLAG":
                        # Mark all entries flagged by default
                        new_subarr[...] = True

                    # Do a little bit of casting magic to plug in the antenna numbers
                    # that we actualy have (versus the spoofed/padded ones).
                    new_subarr[:, self.ant_array, :, :] = subarr[:, :, :, pol_order]

                    # Finally, get this into a 3D array that we can use in
                    new_subarr = np.reshape(
                        new_subarr, (self.Ntimes * Nants_casa, spw_nchan, self.Njones)
                    )
                    if self.Njones == 1 and not has_flex_jones:
                        # Handle a "corner"-case where CASA _really_ wants there to be 2
                        # Jones elements. It can totally ignore one, but it will crash
                        # if the data layout doesn't have 2-elements in the jones axis
                        new_subarr = np.repeat(new_subarr, 2, axis=2)
                        bad_idx = int(self.jones_array[0] in [-1, -3, -5, -7])
                        new_subarr[..., bad_idx] = item == "FLAG"

                    # Plug this pack in to our data dict
                    data_dict[item] = new_subarr

                # Finally, time to plug the values into the MS table
                for key in data_dict:
                    ms.putcol(
                        key,
                        data_dict[key],
                        (count * (Nants_casa * self.Ntimes)),
                        (Nants_casa * self.Ntimes),
                    )

        # Alright, main table is done (and because w/ used `with`, already closed).
        # Finally, write all of the supporting tables that we need.
        ms_utils.write_ms_antenna(filename, uvobj=self)
        if self.phase_center_catalog is None:
            cat_dict = DEFAULT_CAT_DICT
            ms_utils.write_ms_field(
                filename, phase_center_catalog=cat_dict, time_val=0.0
            )
        else:
            ms_utils.write_ms_field(filename, uvobj=self)

        ms_utils.write_ms_history(filename, uvobj=self)
        ms_utils.write_ms_observation(filename, uvobj=self)
        ms_utils.write_ms_spectral_window(filename, uvobj=self)
