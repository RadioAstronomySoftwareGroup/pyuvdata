# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading FHD calibration save files."""

import os
import warnings

import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation
from docstring_parser import DocstringStyle
from scipy.io import readsav

from .. import utils
from ..docstrings import copy_replace_short_description
from ..utils.io import fhd as fhd_utils
from . import UVCal

__all__ = ["FHDCal"]


class FHDCal(UVCal):
    """
    Defines a FHD-specific subclass of UVCal for reading FHD calibration save files.

    This class should not be interacted with directly, instead use the read_fhd_cal
    method on the UVCal class.

    """

    @copy_replace_short_description(UVCal.read_fhd_cal, style=DocstringStyle.NUMPYDOC)
    def read_fhd_cal(
        self,
        *,
        cal_file,
        obs_file,
        layout_file=None,
        settings_file=None,
        raw=True,
        read_data=True,
        default_mount_type="other",
        background_lsts=True,
        extra_history=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        astrometry_library=None,
    ):
        """Read data from an FHD cal.sav file."""
        if not read_data and settings_file is None:
            raise ValueError("A settings_file must be provided if read_data is False.")

        filenames = fhd_utils.fhd_filenames(
            obs_file=obs_file,
            layout_file=layout_file,
            settings_file=settings_file,
            cal_file=cal_file,
        )
        self.filename = filenames
        self._filename.form = (len(self.filename),)

        this_dict = readsav(obs_file, python_dict=True)
        obs_data = this_dict["obs"]
        bl_info = obs_data["BASELINE_INFO"][0]
        astrometry = obs_data["ASTR"][0]

        self.Nspws = 1
        self.spw_array = np.array([0])

        self.Nfreqs = int(obs_data["N_FREQ"][0])
        self.freq_array = np.zeros(len(bl_info["FREQ"][0]), dtype=np.float64)
        self.freq_array[:] = bl_info["FREQ"][0]
        self.channel_width = np.full(self.Nfreqs, float(obs_data["FREQ_RES"][0]))

        self.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)

        # FHD only calculates one calibration over all the times.
        # obs_data.n_time /cal_data.n_times gives the number of times that goes into
        # that one calibration, UVCal.Ntimes gives the number of separate calibrations
        # along the time axis.
        self.Ntimes = 1
        time_array = bl_info["jdate"][0]

        # this is generated in FHD by subtracting the JD of neighboring
        # integrations. This can have limited accuracy, so it can be slightly
        # off the actual value.
        # (e.g. 1.999426... rather than 2)
        # time_res is constrained to be a scalar currently
        self.integration_time = np.array([np.float64(obs_data["TIME_RES"][0])])

        # array of used frequencies (1: used, 0: flagged)
        freq_use = bl_info["freq_use"][0]
        # array of used antennas (1: used, 0: flagged)
        ant_use = bl_info["tile_use"][0]
        # array of used times (1: used, 0: flagged)
        time_use = bl_info["time_use"][0]

        time_array_use = time_array[np.where(time_use > 0)]
        # extend the range by 1/4 the integration time in each direction
        # to make sure that the original data times are covered by the range.
        # Note that this leaves gaps between adjacent files in principal, but
        # using 1/2 the integration time occasionally led to time_ranges overlapping
        # slightly because of precision issues.
        intime_jd = self.integration_time / (24.0 * 3600.0)
        self.time_range = np.reshape(
            np.asarray(
                [
                    np.min(time_array_use) - intime_jd / 4.0,
                    np.max(time_array_use) + intime_jd / 4.0,
                ]
            ),
            (1, 2),
        )

        self.telescope.name = obs_data["instrument"][0].decode("utf8")
        latitude = np.deg2rad(float(obs_data["LAT"][0]))
        longitude = np.deg2rad(float(obs_data["LON"][0]))
        altitude = float(obs_data["ALT"][0])

        # This is a bit of a kludge because nothing like a phase center name
        # exists in FHD files.
        # At least for the MWA, obs.ORIG_PHASERA and obs.ORIG_PHASEDEC specify
        # the field the telescope was nominally pointing at
        # (May need to be revisited, but probably isn't too important)
        cat_name = (
            "Field RA(deg): "
            + str(obs_data["ORIG_PHASERA"][0])
            + ", Dec:"
            + str(obs_data["ORIG_PHASEDEC"][0])
        )
        # For the MWA, this can sometimes be converted to EoR fields
        if (
            self.telescope.name.lower() == "mwa"
            and np.isclose(obs_data["ORIG_PHASERA"][0], 0)
            and np.isclose(obs_data["ORIG_PHASEDEC"][0], -27)
        ):
            cat_name = "EoR 0 Field"

        cat_id = self._add_phase_center(
            cat_name=cat_name,
            cat_type="sidereal",
            cat_lon=np.deg2rad(float(obs_data["OBSRA"][0])),
            cat_lat=np.deg2rad(float(obs_data["OBSDEC"][0])),
            cat_frame=astrometry["RADECSYS"][0].decode().lower(),
            cat_epoch=astrometry["EQUINOX"][0],
            info_source="file",
        )
        self.phase_center_id_array = np.zeros(self.Ntimes, dtype=int) + cat_id

        # get the stuff FHD read from the antenna table (in layout file)
        if layout_file is not None:
            obs_tile_names = [
                ant.decode("utf8") for ant in bl_info["TILE_NAMES"][0].tolist()
            ]
            if self.telescope.name.lower() == "mwa":
                obs_tile_names = [
                    "Tile" + "0" * (3 - len(ant.strip())) + ant.strip()
                    for ant in obs_tile_names
                ]

            layout_param_dict = fhd_utils.get_fhd_layout_info(
                layout_file=layout_file,
                telescope_name=self.telescope.name,
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                obs_tile_names=obs_tile_names,
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

            telescope_attrs = {
                "telescope_location": "location",
                "Nants_telescope": "Nants",
                "antenna_names": "antenna_names",
                "antenna_numbers": "antenna_numbers",
                "antenna_positions": "antenna_positions",
                "diameters": "antenna_diameters",
            }

            for key, value in layout_param_dict.items():
                if key in layout_params_to_ignore:
                    continue
                if key in telescope_attrs:
                    setattr(self.telescope, telescope_attrs[key], value)
                else:
                    setattr(self, key, value)

        else:
            warnings.warn(
                "No layout file, antenna_postions will not be defined "
                "and antenna_names might be incorrect."
            )

            self.telescope.location = EarthLocation.from_geodetic(
                lat=latitude * units.rad,
                lon=longitude * units.rad,
                height=altitude * units.m,
            )
            # FHD stores antenna numbers, not names, in the "TILE_NAMES" field
            self.telescope.antenna_names = [
                ant.decode("utf8") for ant in bl_info["TILE_NAMES"][0].tolist()
            ]
            self.telescope.antenna_numbers = np.array(
                [int(ant) for ant in self.telescope.antenna_names]
            )
            if self.telescope.name.lower() == "mwa":
                self.telescope.antenna_names = [
                    "Tile" + "0" * (3 - len(ant.strip())) + ant.strip()
                    for ant in self.telescope.antenna_names
                ]
            self.telescope.Nants = len(self.telescope.antenna_names)

        self.telescope.antenna_names = np.asarray(self.telescope.antenna_names)

        # need to make sure telescope location is defined properly before this call
        proc = self.set_lsts_from_time_array(
            background=background_lsts, astrometry_library=astrometry_library
        )

        self._set_sky()
        self.gain_convention = "divide"
        self.gain_scale = "Jy"
        self.pol_convetions = "sum"
        self._set_gain()

        # currently don't have branch info. may change in future.
        self.git_origin_cal = "https://github.com/EoRImaging/FHD"
        self.git_hash_cal = obs_data["code_version"][0].decode("utf8")

        if "DELAYS" in obs_data.dtype.names and obs_data["delays"][0] is not None:
            self.extra_keywords["delays"] = (
                "[" + ", ".join(str(int(d)) for d in obs_data["delays"][0]) + "]"
            )
        if settings_file is not None:
            self.history, self.observer = fhd_utils.get_fhd_history(
                settings_file, return_user=True
            )
        else:
            warnings.warn("No settings file, history will be incomplete")
            self.history = ""

        if extra_history is not None:
            if isinstance(extra_history, list | tuple):
                self.history += "\n" + "\n".join(extra_history)
            else:
                self.history += "\n" + extra_history

        if not utils.history._check_history_version(
            self.history, self.pyuvdata_version_str
        ):
            if self.history.endswith("\n"):
                self.history += self.pyuvdata_version_str
            else:
                self.history += "\n" + self.pyuvdata_version_str

        if not read_data:
            n_pols = int(obs_data["N_POL"][0])
            # FHD only has the diagonal elements (jxx, jyy), so limit to 2
            self.Njones = int(np.min([n_pols, 2]))

            # for calibration FHD includes all antennas in the antenna table,
            # regardless of whether or not they have data
            self.Nants_data = len(self.telescope.antenna_names)

            # get details from settings file
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
            with open(settings_file) as read_obj:
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
            self.baseline_range = np.asarray(
                [
                    float(settings_lines["min_cal_baseline"][0]),
                    float(settings_lines["max_cal_baseline"][0]),
                ]
            )
            galaxy_model = int(settings_lines["galaxy_model"][0])
            if len(settings_lines["diffuse_model"]) > 0:
                diffuse_model = settings_lines["diffuse_model"][0]
            else:
                diffuse_model = ""
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
            self.baseline_range = np.asarray(
                [
                    float(cal_data["min_cal_baseline"][0]),
                    float(cal_data["max_cal_baseline"][0]),
                ]
            )

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
                fit_gain_array[:, :, 0, jones_i] = arr
            if raw:
                res_gain_array_in = cal_data["gain_residual"][0]
                res_gain_array = np.zeros(
                    self._gain_array.expected_shape(self), dtype=np.complex128
                )
                for jones_i, arr in enumerate(res_gain_array_in):
                    res_gain_array[:, :, 0, jones_i] = arr
                self.gain_array = fit_gain_array + res_gain_array
            else:
                self.gain_array = fit_gain_array

            # FHD doesn't really have a chi^2 measure. What is has is a convergence
            # measure. The solution converged well if this is less than the convergence
            # threshold ('conv_thresh' in extra_keywords).
            self.quality_array = np.zeros_like(self.gain_array, dtype=np.float64)
            convergence = cal_data["convergence"][0]
            for jones_i, arr in enumerate(convergence):
                self.quality_array[:, :, 0, jones_i] = arr

            # Currently this can't include the times because the flag array
            # dimensions has to match the gain array dimensions.
            # This is somewhat artificial...
            self.flag_array = np.zeros_like(self.gain_array, dtype=np.bool_)
            flagged_ants = np.where(ant_use == 0)[0]
            for ant in flagged_ants:
                self.flag_array[ant] = 1
            flagged_freqs = np.where(freq_use == 0)[0]
            for freq in flagged_freqs:
                self.flag_array[:, freq] = 1

        if self.telescope.name.lower() == "mwa":
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

        if self.telescope.name == "mwa":
            # Setting this explicitly to avoid superfluous warnings, given that MWA
            # seems to be the most common telescope to pass through FHD
            self.telescope.mount_type = ["phased"] * self.telescope.Nants

        self.set_telescope_params(x_orientation="east", mount_type=default_mount_type)

        # for calibration FHD creates gain array of shape (Nfreqs, Nants_telescope)
        # rather than (Nfreqs, Nants_data). This means the antenna array will
        # contain all antennas in the antenna table instead of only those
        # which had data in the original uvfits file
        self.ant_array = self.telescope.antenna_numbers

        self.extra_keywords["autoscal".upper()] = (
            "[" + ", ".join(str(d) for d in auto_scale) + "]"
        )
        self.extra_keywords["nvis_cal".upper()] = n_vis_cal
        self.extra_keywords["time_avg".upper()] = time_avg
        self.extra_keywords["cvgthres".upper()] = conv_thresh

        if not raw:
            self.extra_keywords["polyfit".upper()] = polyfit
            self.extra_keywords["bandpass".upper()] = bandpass
            if isinstance(mode_fit, list | tuple | np.ndarray):
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
