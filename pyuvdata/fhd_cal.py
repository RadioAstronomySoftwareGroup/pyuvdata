from scipy.io.idl import readsav
import numpy as np
import warnings
from uvcal import UVCal


class FHD_cal(UVCal):
    """
    Defines a FHD-specific subclass of UVCal for reading FHD calibration save files.
    This class should not be interacted with directly, instead use the read_fhd_cal
    method on the UVCal class.
    """

    def read_fhd_cal(self, cal_file, obs_file, settings_file=None, raw=True,
                     extra_history=None, run_check=True, check_extra=True,
                     run_check_acceptability=True):
        """
        Read data from an FHD cal.sav file.

        Args:
            cal_file: The cal.sav file to read from.
            obs_file: The obs.sav file to read from.
            settings_file: The settings_file to read from. Optional, but very
                useful for provenance.
            raw: Option to use the raw (per antenna, per frequency) solution or
                to use the fitted (polynomial over phase/amplitude) solution.
                Default is True (meaning use the raw solutions).
            extra_history: Optional string or list of strings to add to the
                object's history parameter. Default is None.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """

        this_dict = readsav(cal_file, python_dict=True)
        cal_data = this_dict['cal']

        this_dict = readsav(obs_file, python_dict=True)
        obs_data = this_dict['obs']

        self.Nfreqs = cal_data['n_freq'][0]
        self.freq_array = cal_data['freq'][0]
        self.channel_width = self.freq_array[1] - self.freq_array[0]

        # FHD only calculates one calibration over all the times.
        # cal_data.n_times gives the number of times that goes into that one
        # calibration, UVCal.Ntimes gives the number of separate calibrations
        # along the time axis.
        self.Ntimes = 1
        time_array = obs_data['baseline_info'][0]['jdate'][0]
        self.integration_time = np.round(np.mean(np.diff(time_array)) * 24 * 3600, 2)

        self.Nspws = 1
        self.spw_array = np.array([0])
        self.Njones = cal_data['n_pol'][0]
        # FHD only has the diagonal elements (jxx, jyy) and if there's only one
        # present it must be jxx
        if self.Njones == 1:
            self.jones_array = [-5]
        else:
            self.jones_array = [-5, -6]

        self.telescope_name = obs_data['instrument'][0]

        self.Nants_data = cal_data['n_tile'][0]
        self.Nants_telescope = cal_data['n_tile'][0]
        self.antenna_names = cal_data.['tile_names'][0]
        self.antenna_numbers = np.arange(self.Nants_telescope)
        self.ant_array = np.arange(self.Nants_data)

        self.gain_convention = 'divide'
        self.x_orientation = 'east'

        self.cal_type = 'gain'
        fit_gain_array_in = cal_data['gain'][0]
        fit_gain_array = np.zeros(self._gain_array.expected_shape, dtype=np.complex_)
        for jones_i, arr in enumerate(fit_gain_array_in):
            fit_gain_array[:, 0, :, 0, jones_i] = arr
        if raw:
            res_gain_array_in = cal_data['gain_residual'][0]
            res_gain_array = np.zeros(self._gain_array.expected_shape, dtype=np.complex_)
            for jones_i, arr in enumerate(fit_gain_array_in):
                res_gain_array[:, 0, :, 0, jones_i] = arr
            self.gain_array = fit_gain_array + res_gain_array
        else:
            self.gain_array = fit_gain_array

        # FHD doesn't really have a chi^2 measure. What is has is a convergence measure.
        # The solution converged well if this is less than the convergence
        # threshold ('conv_thresh' in extra_keywords).
        self.quality_array = np.zeros_like(self.gain_array, dtype=np.float)
        convergence = cal_data['convergence'][0]
        for jones_i, arr in enumerate(convergence):
            self.quality_array[:, 0, :, 0, jones_i] = arr

        # array of used frequencies (1: used, 0: flagged)
        freq_use = obs_data['baseline_info'][0]['freq_use'][0]
        # array of used antennas (1: used, 0: flagged)
        ant_use = obs_data['baseline_info'][0]['tile_use'][0]
        # array of used times (1: used, 0: flagged)
        time_use = obs_data['baseline_info'][0]['time_use'][0]

        time_array_use = time_array[np.where(time_use > 0)]
        self.time_range = [np.min(time_array_use), np.max(time_array_use)]

        # Currently this can't include the times because the flag array
        # dimensions has to match the gain array dimensions. This is somewhat artificial...
        self.flag_array = np.zeros_like(self.gain_array, dtype=np.bool)
        flagged_ants = np.where(ant_use == 0)[0]
        for ant in flagged_ants:
            flag_array[ant, :] = 1
        flagged_freqs = np.where(freq_use == 0)[0]
        for freq in flagged_freqs:
            flag_array[:, :, freq] = 1

        # currently don't have branch info. may change in future.
        self.git_origin_cal = 'https://github.com/EoRImaging/FHD'
        self.git_hash_cal = obs_data['code_version'][0]

        self.extra_keywords['auto_scale'] = \
            '[' + ', '.join(str(d) for d in cal_data['auto_scale'][0]) + ']'
        self.extra_keywords['n_vis_cal'] = cal_data['n_vis_cal'][0]
        self.extra_keywords['time_avg'] = cal_data['time_avg'][0]
        self.extra_keywords['conv_thresh'] = cal_data['conv_thresh'][0]
        if 'DELAYS' in obs_data.dtype.names:
            self.extra_keywords['mwa_delays'] = \
                '[' + ', '.join(str(int(d)) for d in obs_data['delays'][0]) + ']'

        if not raw:
            self.extra_keywords['polyfit'] = cal_data['polyfit'][0]
            self.extra_keywords['bandpass'] = cal_data['bandpass'][0]
            self.extra_keywords['mode_fit'] = cal_data['mode_fit'][0]
            self.extra_keywords['amp_degee'] = cal_data['amp_degree'][0]
            self.extra_keywords['phase_degree'] = cal_data['phase_degree'][0]

        if settings_file is not None:
            settings_lines = open(settings_file, 'r').readlines()
            main_loc = None
            command_loc = None
            obs_loc = None
            user_line = None
            for ind, line in enumerate(settings_lines):
                if line.startswith('##MAIN'):
                    main_loc = ind
                if line.startswith('##COMMAND_LINE'):
                    command_loc = ind
                if line.startswith('##OBS'):
                    obs_loc = ind
                if line.startswith('User'):
                    user_line = ind
                if (main_loc is not None and command_loc is not None and
                        obs_loc is not None and user_line is not None):
                    break

            main_lines = settings_lines[main_loc + 1:command_loc]
            command_lines = settings_lines[command_loc + 1:obs_loc]
            history_lines = ['FHD history\n'] + main_lines + command_lines
            for ind, line in enumerate(history_lines):
                history_lines[ind] = line.rstrip().replace('\t', ' ')
            self.history = '\n'.join(history_lines)
            self.observer = settings_lines[user_line].split()[1]

        if extra_history is not None:
            if isinstance(extra_history, (list, tuple)):
                self.history += '\n' + '\n'.join(extra_history)
            else:
                self.history += '\n' + extra_history

        # new proposed keywords:
        # cal_style: sky
        # sky_field: 'phase center: RA, Dec' -- obs.orig_phasera, obs.orig_phasedec
        # sky_catalog: cal.skymodel.catalog_name
        # ref_antenna_name: ref_antenna_name
        # n_sources: cal.skymodel.n_sources
        # cal_baseline_range: min_cal_baseline to max_cal_baseline
        # diffuse_model: cal.skymodel.galaxy_model or cal.skymodel.diffuse_model or both
