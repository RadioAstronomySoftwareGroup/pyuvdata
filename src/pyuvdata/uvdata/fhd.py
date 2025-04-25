# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading FHD save files."""

from __future__ import annotations

import warnings

import numpy as np
from astropy import constants as const, units
from astropy.coordinates import EarthLocation
from docstring_parser import DocstringStyle
from scipy.io import readsav

from .. import utils
from ..docstrings import copy_replace_short_description
from ..utils.io import fhd as fhd_utils
from . import UVData

__all__ = ["FHD"]


class FHD(UVData):
    """
    Defines a FHD-specific subclass of UVData for reading FHD save files.

    This class should not be interacted with directly, instead use the read_fhd
    method on the UVData class.
    """

    @copy_replace_short_description(UVData.read_fhd, style=DocstringStyle.NUMPYDOC)
    def read_fhd(
        self,
        vis_files: list[str] | np.ndarray | str,
        *,
        params_file: str,
        obs_file: str | None = None,
        flags_file: str | None = None,
        layout_file: str | None = None,
        settings_file: str | None = None,
        background_lsts=True,
        read_data=True,
        default_mount_type="other",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=True,
        astrometry_library=None,
    ):
        """Read in data from a list of FHD files."""
        datafiles_dict = {}
        use_model = None
        if isinstance(vis_files, str):
            vis_files = [vis_files]
        for filename in vis_files:
            if filename is None:
                continue
            if filename.lower().endswith("xx.sav"):
                if "xx" in list(datafiles_dict.keys()):
                    raise ValueError("multiple xx datafiles in vis_files")
                datafiles_dict["xx"] = filename
            elif filename.lower().endswith("yy.sav"):
                if "yy" in list(datafiles_dict.keys()):
                    raise ValueError("multiple yy datafiles in vis_files")
                datafiles_dict["yy"] = filename
            elif filename.lower().endswith("xy.sav"):
                if "xy" in list(datafiles_dict.keys()):
                    raise ValueError("multiple xy datafiles in vis_files")
                datafiles_dict["xy"] = filename
            elif filename.lower().endswith("yx.sav"):
                if "yx" in list(datafiles_dict.keys()):
                    raise ValueError("multiple yx datafiles in vis_files")
                datafiles_dict["yx"] = filename
            else:
                raise ValueError("unrecognized file in vis_files")

            if "_vis_model_" in filename:
                this_model = True
            else:
                this_model = False

            if use_model is None:
                use_model = this_model
            elif this_model != use_model:
                raise ValueError(
                    "The vis_files parameter has a mix of model and data files."
                )

        if len(datafiles_dict) < 1 and read_data is True:
            raise ValueError(
                "The vis_files parameter must be passed if read_data is True"
            )
        if flags_file is None and read_data is True:
            raise ValueError(
                "The flags_file parameter must be passed if read_data is True"
            )

        if obs_file is None and read_data is False:
            raise ValueError(
                "The obs_file parameter must be passed if read_data is False."
            )

        if layout_file is None:
            warnings.warn(
                "The layout_file parameter was not passed, so antenna_postions will "
                "not be defined and antenna names and numbers might be incorrect."
            )

        if settings_file is None:
            warnings.warn(
                "The settings_file parameter was not passed, so some history "
                "information will be missing."
            )

        filenames = fhd_utils.fhd_filenames(
            vis_files=vis_files,
            params_file=params_file,
            obs_file=obs_file,
            flags_file=flags_file,
            layout_file=layout_file,
            settings_file=settings_file,
        )

        self.filename = filenames
        self._filename.form = (len(self.filename),)

        if not read_data:
            obs_dict = readsav(obs_file, python_dict=True)
            this_obs = obs_dict["obs"]
            self.Npols = int(this_obs[0]["N_POL"])
        else:
            vis_data = {}
            for pol, file in datafiles_dict.items():
                this_dict = readsav(file, python_dict=True)
                if use_model:
                    vis_data[pol] = this_dict["vis_model_ptr"]
                else:
                    vis_data[pol] = this_dict["vis_ptr"]
                this_obs = this_dict["obs"]
            self.Npols = len(list(vis_data.keys()))

        obs = this_obs
        bl_info = obs["BASELINE_INFO"][0]
        astrometry = obs["ASTR"][0]
        fhd_pol_list = []
        for pol in obs["POL_NAMES"][0]:
            fhd_pol_list.append(pol.decode("utf8").lower())

        params_dict = readsav(params_file, python_dict=True)
        params = params_dict["params"]

        if read_data:
            flag_file_dict = readsav(flags_file, python_dict=True)
            # The name for this variable changed recently (July 2016). Test for both.
            vis_weights_data = {}
            if "flag_arr" in flag_file_dict:
                weights_key = "flag_arr"
            elif "vis_weights" in flag_file_dict:
                weights_key = "vis_weights"
            else:
                raise ValueError(
                    "No recognized key for visibility weights in flags_file."
                )
            for index, w in enumerate(flag_file_dict[weights_key]):
                vis_weights_data[fhd_pol_list[index]] = w

        self.Ntimes = int(obs["N_TIME"][0])
        self.Nbls = int(obs["NBASELINES"][0])
        self.Nblts = params["UU"][0].size
        self.Nfreqs = int(obs["N_FREQ"][0])
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)

        self.vis_units = "Jy"
        self.pol_convention = "sum"

        # bl_info.JDATE (a vector of length Ntimes) is the only safe date/time
        # to use in FHD files.
        # (obs.JD0 (float) and params.TIME (vector of length Nblts) are
        #   context dependent and are not safe
        #   because they depend on the phasing of the visibilities)
        # the values in bl_info.JDATE are the JD for each integration.
        # We need to expand up to Nblts.
        int_times = list(utils.tools._get_iterable(bl_info["JDATE"][0]))
        bin_offset = bl_info["BIN_OFFSET"][0]
        if self.Ntimes != len(int_times):
            warnings.warn(
                "Ntimes does not match the number of unique times in the data"
            )
        self.time_array = np.zeros(self.Nblts)
        if self.Ntimes == 1:
            self.time_array.fill(int_times[0])
        else:
            for ii in range(0, len(int_times)):
                if ii < (len(int_times) - 1):
                    self.time_array[bin_offset[ii] : bin_offset[ii + 1]] = int_times[ii]
                else:
                    self.time_array[bin_offset[ii] :] = int_times[ii]

        # this is generated in FHD by subtracting the JD of neighboring
        # integrations. This can have limited accuracy, so it can be slightly
        # off the actual value.
        # (e.g. 1.999426... rather than 2)
        time_res = obs["TIME_RES"]
        # time_res is constrained to be a scalar currently
        self.integration_time = (
            np.ones_like(self.time_array, dtype=np.float64) * time_res[0]
        )
        # # --- observation information ---
        self.telescope.name = obs["INSTRUMENT"][0].decode("utf8")

        # This is a bit of a kludge because nothing like a phase center name exists
        # in FHD files.
        # At least for the MWA, obs.ORIG_PHASERA and obs.ORIG_PHASEDEC specify
        # the field the telescope was nominally pointing at
        # (May need to be revisited, but probably isn't too important)
        cat_name = (
            "Field RA(deg): "
            + str(obs["ORIG_PHASERA"][0])
            + ", Dec:"
            + str(obs["ORIG_PHASEDEC"][0])
        )
        # For the MWA, this can sometimes be converted to EoR fields
        if (
            self.telescope.name.lower() == "mwa"
            and np.isclose(obs["ORIG_PHASERA"][0], 0)
            and np.isclose(obs["ORIG_PHASEDEC"][0], -27)
        ):
            cat_name = "EoR 0 Field"

        self.telescope.instrument = self.telescope.name
        latitude = np.deg2rad(float(obs["LAT"][0]))
        longitude = np.deg2rad(float(obs["LON"][0]))
        altitude = float(obs["ALT"][0])

        # get the stuff FHD read from the antenna table (in layout file)
        if layout_file is not None:
            # in older FHD versions, incorrect tile names for the mwa
            # might have been stored in bl_info, so pull the obs_tile_names
            # and check them against the layout file
            obs_tile_names = [
                ant.decode("utf8") for ant in bl_info["TILE_NAMES"][0].tolist()
            ]
            if self.telescope.name.lower() == "mwa" and obs_tile_names[0][0] != "T":
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

            telescope_attrs = {
                "telescope_location": "location",
                "Nants_telescope": "Nants",
                "antenna_names": "antenna_names",
                "antenna_numbers": "antenna_numbers",
                "antenna_positions": "antenna_positions",
                "diameters": "antenna_diameters",
            }

            for key, value in layout_param_dict.items():
                if key in telescope_attrs:
                    setattr(self.telescope, telescope_attrs[key], value)
                else:
                    setattr(self, key, value)

        else:
            self.telescope.location = EarthLocation.from_geodetic(
                lat=latitude * units.rad,
                lon=longitude * units.rad,
                height=altitude * units.m,
            )

            # we don't have layout info, so go ahead and set the antenna_names,
            # antenna_numbers and Nants_telescope from the baseline info struct.
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

        # Polarization information
        lin_pol_order = ["xx", "yy", "xy", "yx"]
        linear_pol_dict = dict(zip(lin_pol_order, np.arange(5, 9) * -1, strict=True))
        pol_list = []
        if read_data:
            for pol in lin_pol_order:
                if pol in vis_data:
                    pol_list.append(linear_pol_dict[pol])
            self.polarization_array = np.asarray(pol_list)
        else:
            # Use Npols because for FHD, npol fully specifies which pols to use
            pol_strings = lin_pol_order[: self.Npols]
            self.polarization_array = np.asarray(
                [linear_pol_dict[pol] for pol in pol_strings]
            )

        if self.telescope.name.lower() == "mwa":
            # Setting this explicitly to avoid superfluous warnings, given that MWA
            # seems to be the most common telescope to pass through FHD
            self.telescope.mount_type = ["phased"] * self.telescope.Nants
            self.telescope.set_feeds_from_x_orientation(
                x_orientation="east", polarization_array=self.polarization_array
            )

        self.set_telescope_params(mount_type=default_mount_type)

        # need to make sure telescope location is defined properly before this call
        proc = self.set_lsts_from_time_array(
            background=background_lsts, astrometry_library=astrometry_library
        )

        if not np.isclose(obs["OBSRA"][0], obs["PHASERA"][0]) or not np.isclose(
            obs["OBSDEC"][0], obs["PHASEDEC"][0]
        ):
            warnings.warn(
                "These visibilities may have been phased "
                "improperly -- without changing the uvw locations"
            )

        cat_id = self._add_phase_center(
            cat_name=cat_name,
            cat_type="sidereal",
            cat_lon=np.deg2rad(float(obs["OBSRA"][0])),
            cat_lat=np.deg2rad(float(obs["OBSDEC"][0])),
            cat_frame=astrometry["RADECSYS"][0].decode().lower(),
            cat_epoch=astrometry["EQUINOX"][0],
        )
        self.phase_center_id_array = np.zeros(self.Nblts, dtype=int) + cat_id

        # Note that FHD antenna arrays are 1-indexed so we subtract 1
        # to get 0-indexed arrays
        ind_1_array = bl_info["TILE_A"][0] - 1
        ind_2_array = bl_info["TILE_B"][0] - 1
        self.ant_1_array = self.telescope.antenna_numbers[ind_1_array]
        self.ant_2_array = self.telescope.antenna_numbers[ind_2_array]
        self.Nants_data = int(np.union1d(self.ant_1_array, self.ant_2_array).size)

        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        if self.Nbls != len(np.unique(self.baseline_array)):
            warnings.warn(
                "Nbls does not match the number of unique baselines in the data"
            )

        self.freq_array = np.zeros(len(bl_info["FREQ"][0]), dtype=np.float64)
        self.freq_array[:] = bl_info["FREQ"][0]

        self.channel_width = np.full(self.Nfreqs, float(obs["FREQ_RES"][0]))

        # In FHD, uvws are in seconds not meters.
        # FHD follows the FITS uvw direction convention, which is opposite
        # ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        self.uvw_array = np.zeros((self.Nblts, 3))
        self.uvw_array[:, 0] = (-1) * params["UU"][0] * const.c.to_value("m/s")
        self.uvw_array[:, 1] = (-1) * params["VV"][0] * const.c.to_value("m/s")
        self.uvw_array[:, 2] = (-1) * params["WW"][0] * const.c.to_value("m/s")

        # history: add the first few lines from the settings file
        if settings_file is not None:
            self.history = fhd_utils.get_fhd_history(settings_file)
        else:
            self.history = ""

        if not utils.history._check_history_version(
            self.history, self.pyuvdata_version_str
        ):
            self.history += self.pyuvdata_version_str

        if read_data:
            self.data_array = np.zeros(
                (self.Nblts, self.Nfreqs, self.Npols), dtype=np.complex128
            )
            self.nsample_array = np.zeros(
                (self.Nblts, self.Nfreqs, self.Npols), dtype=np.float64
            )
            self.flag_array = np.zeros(
                (self.Nblts, self.Nfreqs, self.Npols), dtype=np.bool_
            )
            for pol, vis in vis_data.items():
                pol_i = pol_list.index(linear_pol_dict[pol])
                # FHD follows the FITS uvw direction convention, which is opposite
                # ours and Miriad's.
                # So conjugate the visibilities and flip the uvws:
                self.data_array[:, :, pol_i] = np.conj(vis)
                self.flag_array[:, :, pol_i] = vis_weights_data[pol] <= 0
                self.nsample_array[:, :, pol_i] = np.abs(vis_weights_data[pol])

        # wait for LSTs if set in background
        if proc is not None:
            proc.join()

        self._set_app_coords_helper()

        # check if object has all required uv_properties set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )
