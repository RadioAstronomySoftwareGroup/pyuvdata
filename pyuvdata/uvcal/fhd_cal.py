# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading FHD calibration save files."""

import os
import numpy as np
import warnings
from scipy.io.idl import readsav

from .uvcal import UVCal
from .. import utils as uvutils
from ..uvdata.fhd import get_fhd_history, get_fhd_layout_info

__all__ = ["FHDCal"]


class FHDCal(UVCal):
    """
    Defines a FHD-specific subclass of UVCal for reading FHD calibration save files.

    This class should not be interacted with directly, instead use the read_fhd_cal
    method on the UVCal class.

    """

    def read_fhd_cal(
        self,
        cal_file,
        obs_file,
        layout_file=None,
        settings_file=None,
        raw=True,
        read_data=True,
        background_lsts=True,
        extra_history=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read data from an FHD cal.sav file.

        Parameters
        ----------
        cal_file : str
            The cal.sav file to read from.
        obs_file : str
            The obs.sav file to read from.
        layout_file : str
            The FHD layout file. Required for antenna_positions to be set.
        settings_file : str, optional
            The settings_file to read from. Optional, but very useful for provenance.
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

        """
        this_dict = readsav(obs_file, python_dict=True)
        obs_data = this_dict["obs"]
        bl_info = obs_data["BASELINE_INFO"][0]

        self.Nspws = 1
        self.spw_array = np.array([0])

        self.Nfreqs = int(obs_data["N_FREQ"][0])
        self.freq_array = np.zeros((1, len(bl_info["FREQ"][0])), dtype=np.float64)
        self.freq_array[0, :] = bl_info["FREQ"][0]
        self.channel_width = float(obs_data["FREQ_RES"][0])

        # FHD only calculates one calibration over all the times.
        # obs_data.n_time /cal_data.n_times gives the number of times that goes into
        # that one calibration, UVCal.Ntimes gives the number of separate calibrations
        # along the time axis.
        self.Ntimes = 1
        time_array = bl_info["jdate"][0]

        self.time_array = np.array([np.mean(time_array)])

        # this is generated in FHD by subtracting the JD of neighboring
        # integrations. This can have limited accuracy, so it can be slightly
        # off the actual value.
        # (e.g. 1.999426... rather than 2)
        time_res = obs_data["TIME_RES"][0]
        # time_res is constrained to be a scalar currently
        self.integration_time = np.float64(time_res)

        # array of used frequencies (1: used, 0: flagged)
        freq_use = bl_info["freq_use"][0]
        # array of used antennas (1: used, 0: flagged)
        ant_use = bl_info["tile_use"][0]
        # array of used times (1: used, 0: flagged)
        time_use = bl_info["time_use"][0]

        time_array_use = time_array[np.where(time_use > 0)]
        self.time_range = [np.min(time_array_use), np.max(time_array_use)]

        self.telescope_name = obs_data["instrument"][0].decode("utf8")
        latitude = np.deg2rad(float(obs_data["LAT"][0]))
        longitude = np.deg2rad(float(obs_data["LON"][0]))
        altitude = float(obs_data["ALT"][0])

        # get the stuff FHD read from the antenna table (in layout file)
        if layout_file is not None:
            obs_tile_names = [
                ant.decode("utf8").strip() for ant in bl_info["TILE_NAMES"][0].tolist()
            ]
            obs_tile_names = [
                "Tile" + "0" * (3 - len(ant)) + ant for ant in obs_tile_names
            ]

            layout_param_dict = get_fhd_layout_info(
                layout_file,
                self.telescope_name,
                latitude,
                longitude,
                altitude,
                self._lst_array.tols,
                self._telescope_location.tols,
                obs_tile_names,
                run_check_acceptability=True,
            )

            layout_params_to_ignore = [
                "gst0",
                "rdate",
                "earth_omega",
                "dut1",
                "timesys",
                "diameters",
            ]
            for key, value in layout_param_dict.items():
                if key not in layout_params_to_ignore:
                    setattr(self, key, value)

        else:
            warnings.warn("No layout file, antenna_postions will not be defined.")

            self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
            self.antenna_names = [
                ant.decode("utf8").strip() for ant in bl_info["TILE_NAMES"][0].tolist()
            ]
            if self.telescope_name.lower() == "mwa":
                self.antenna_names = [
                    "Tile" + "0" * (3 - len(ant)) + ant for ant in self.antenna_names
                ]
            self.Nants_telescope = len(self.antenna_names)
            self.antenna_numbers = np.arange(self.Nants_telescope)

        self.antenna_names = np.asarray(self.antenna_names)

        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        # need to make sure telescope location is defined properly before this call
        if self.telescope_location is not None:
            proc = self.set_lsts_from_time_array(background=background_lsts)

        self._set_sky()
        self.sky_field = "phase center (RA, Dec): ({ra}, {dec})".format(
            ra=obs_data["orig_phasera"][0], dec=obs_data["orig_phasedec"][0]
        )
        self.gain_convention = "divide"
        self.x_orientation = "east"

        self._set_gain()

        # currently don't have branch info. may change in future.
        self.git_origin_cal = "https://github.com/EoRImaging/FHD"
        self.git_hash_cal = obs_data["code_version"][0].decode("utf8")

        if "DELAYS" in obs_data.dtype.names:
            if obs_data["delays"][0] is not None:
                self.extra_keywords["delays"] = (
                    "[" + ", ".join(str(int(d)) for d in obs_data["delays"][0]) + "]"
                )
        if settings_file is not None:
            self.history, self.observer = get_fhd_history(
                settings_file, return_user=True
            )
        else:
            warnings.warn("No settings file, history will be incomplete")
            self.history = ""

        if extra_history is not None:
            if isinstance(extra_history, (list, tuple)):
                self.history += "\n" + "\n".join(extra_history)
            else:
                self.history += "\n" + extra_history

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            if self.history.endswith("\n"):
                self.history += self.pyuvdata_version_str
            else:
                self.history += "\n" + self.pyuvdata_version_str

        if not read_data:
            n_pols = int(obs_data["N_POL"])
            # FHD only has the diagonal elements (jxx, jyy), so limit to 2
            self.Njones = int(np.min([n_pols, 2]))

            # Note that FHD antenna arrays are 1-indexed so we subtract 1
            # to get 0-indexed arrays
            ant_1_array = bl_info["TILE_A"][0] - 1
            ant_2_array = bl_info["TILE_B"][0] - 1

            self.Nants_data = int(np.union1d(ant_1_array, ant_2_array).size)

            # get details from settings file if it's available
            if settings_file is not None:
                keywords = [
                    "ref_antenna_name",
                    "catalog_name",
                    "n_sources",
                    "min_cal_baseline",
                    "max_cal_baseline",
                    "galaxy_model",
                    "diffuse_model",
                    "auto_scale",
                    "n_vis_cal",
                    "time_avg",
                    "conv_thresh",
                ]
                if not raw:
                    keywords += [
                        "polyfit",
                        "bandpass",
                        "mode_fit",
                        "amp_degree",
                        "phase_degree",
                    ]

                settings_lines = {}
                with open(settings_file, "r") as read_obj:
                    cal_start = False
                    for line in read_obj:
                        if not cal_start:
                            if line.startswith("##CAL"):
                                cal_start = True
                        else:
                            if line.startswith("##"):
                                break
                            # in cal structure section
                            for kw in keywords:
                                if line.strip().startswith(kw.upper()):
                                    settings_lines[kw] = line.split()[1:]
                self.ref_antenna_name = settings_lines["ref_antenna_name"][0]
                self.Nsources = int(settings_lines["n_sources"][0])
                self.sky_catalog = settings_lines["catalog_name"][0]
                self.baseline_range = [
                    float(settings_lines["min_cal_baseline"][0]),
                    float(settings_lines["max_cal_baseline"][0]),
                ]
                galaxy_model = int(settings_lines["galaxy_model"][0])
                diffuse_model = settings_lines["diffuse_model"][0]
                auto_scale = settings_lines["auto_scale"]
                n_vis_cal = np.int64(settings_lines["n_vis_cal"][0])
                time_avg = int(settings_lines["time_avg"][0])
                conv_thresh = float(settings_lines["conv_thresh"][0])

                if not raw:
                    polyfit = int(settings_lines["polyfit"][0])
                    bandpass = int(settings_lines["bandpass"][0])
                    mode_fit = settings_lines["mode_fit"]
                    # for some reason, it's a float if it's one value,
                    # and integers otherwise
                    if len(mode_fit) == 1:
                        mode_fit = float(mode_fit[0])
                    else:
                        mode_fit = np.array(mode_fit, dtype=np.int64)

                    amp_degree = int(settings_lines["amp_degree"][0])
                    phase_degree = int(settings_lines["phase_degree"][0])

        else:
            this_dict = readsav(cal_file, python_dict=True)
            cal_data = this_dict["cal"]
            self.Njones = int(cal_data["n_pol"][0])
            self.Nants_data = int(cal_data["n_tile"][0])

            self.sky_catalog = cal_data["skymodel"][0]["catalog_name"][0].decode("utf8")
            self.ref_antenna_name = (
                cal_data["ref_antenna_name"][0].decode("utf8").strip()
            )
            self.Nsources = int(cal_data["skymodel"][0]["n_sources"][0])
            self.baseline_range = [
                float(cal_data["min_cal_baseline"][0]),
                float(cal_data["max_cal_baseline"][0]),
            ]

            galaxy_model = cal_data["skymodel"][0]["galaxy_model"][0]
            diffuse_model = cal_data["skymodel"][0]["diffuse_model"][0]

            auto_scale = cal_data["auto_scale"][0]
            n_vis_cal = cal_data["n_vis_cal"][0]
            time_avg = cal_data["time_avg"][0]
            conv_thresh = cal_data["conv_thresh"][0]

            if not raw:
                polyfit = cal_data["polyfit"][0]
                bandpass = cal_data["bandpass"][0]
                mode_fit = cal_data["mode_fit"][0]
                amp_degree = cal_data["amp_degree"][0]
                phase_degree = cal_data["phase_degree"][0]

            # Now read data like arrays
            fit_gain_array_in = cal_data["gain"][0]
            fit_gain_array = np.zeros(
                self._gain_array.expected_shape(self), dtype=np.complex128
            )
            for jones_i, arr in enumerate(fit_gain_array_in):
                fit_gain_array[:, 0, :, 0, jones_i] = arr
            if raw:
                res_gain_array_in = cal_data["gain_residual"][0]
                res_gain_array = np.zeros(
                    self._gain_array.expected_shape(self), dtype=np.complex128
                )
                for jones_i, arr in enumerate(res_gain_array_in):
                    res_gain_array[:, 0, :, 0, jones_i] = arr
                self.gain_array = fit_gain_array + res_gain_array
            else:
                self.gain_array = fit_gain_array

            # FHD doesn't really have a chi^2 measure. What is has is a convergence
            # measure. The solution converged well if this is less than the convergence
            # threshold ('conv_thresh' in extra_keywords).
            self.quality_array = np.zeros_like(self.gain_array, dtype=np.float64)
            convergence = cal_data["convergence"][0]
            for jones_i, arr in enumerate(convergence):
                self.quality_array[:, 0, :, 0, jones_i] = arr

            # Currently this can't include the times because the flag array
            # dimensions has to match the gain array dimensions.
            # This is somewhat artificial...
            self.flag_array = np.zeros_like(self.gain_array, dtype=np.bool_)
            flagged_ants = np.where(ant_use == 0)[0]
            for ant in flagged_ants:
                self.flag_array[ant, :] = 1
            flagged_freqs = np.where(freq_use == 0)[0]
            for freq in flagged_freqs:
                self.flag_array[:, :, freq] = 1

        if self.telescope_name.lower() == "mwa":
            self.ref_antenna_name = (
                "Tile" + "0" * (3 - len(self.ref_antenna_name)) + self.ref_antenna_name
            )
        # In Python 3, we sometimes get Unicode, sometimes bytes
        # doesn't reliably show up in tests though, so excluding it from coverage
        if isinstance(galaxy_model, bytes):  # pragma: nocover
            galaxy_model = galaxy_model.decode("utf8")
        if galaxy_model == 0:
            galaxy_model = None
        else:
            galaxy_model = "gsm"

        if isinstance(diffuse_model, bytes):
            diffuse_model = diffuse_model.decode("utf8")
        if diffuse_model == "":
            diffuse_model = None
        else:
            diffuse_model = os.path.basename(diffuse_model)

        if galaxy_model is not None:
            if diffuse_model is not None:
                self.diffuse_model = galaxy_model + " + " + diffuse_model
            else:
                self.diffuse_model = galaxy_model
        elif diffuse_model is not None:
            self.diffuse_model = diffuse_model

        # FHD only has the diagonal elements (jxx, jyy) and if there's only one
        # present it must be jxx
        if self.Njones == 1:
            self.jones_array = np.array([-5])
        else:
            self.jones_array = np.array([-5, -6])

        self.ant_array = np.arange(self.Nants_data)

        self.extra_keywords["autoscal".upper()] = (
            "[" + ", ".join(str(d) for d in auto_scale) + "]"
        )
        self.extra_keywords["nvis_cal".upper()] = n_vis_cal
        self.extra_keywords["time_avg".upper()] = time_avg
        self.extra_keywords["cvgthres".upper()] = conv_thresh

        if not raw:
            self.extra_keywords["polyfit".upper()] = polyfit
            self.extra_keywords["bandpass".upper()] = bandpass
            if isinstance(mode_fit, (list, tuple, np.ndarray)):
                self.extra_keywords["mode_fit".upper()] = (
                    "[" + ", ".join(str(m) for m in mode_fit) + "]"
                )
            else:
                self.extra_keywords["mode_fit".upper()] = mode_fit
            self.extra_keywords["amp_deg".upper()] = amp_degree
            self.extra_keywords["phse_deg".upper()] = phase_degree

        # wait for LSTs if set in background
        if proc is not None:
            proc.join()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
