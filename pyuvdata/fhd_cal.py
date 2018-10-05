# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import six
import warnings
from scipy.io.idl import readsav

from . import UVCal
from . import utils as uvutils
from .fhd import get_fhd_history


class FHDCal(UVCal):
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

        self.Nspws = 1
        self.spw_array = np.array([0])

        self.Nfreqs = int(cal_data['n_freq'][0])
        self.freq_array = np.zeros((self.Nspws, len(cal_data['freq'][0])), dtype=np.float_)
        self.freq_array[0, :] = cal_data['freq'][0]
        self.channel_width = float(np.mean(np.diff(self.freq_array)))

        # FHD only calculates one calibration over all the times.
        # cal_data.n_times gives the number of times that goes into that one
        # calibration, UVCal.Ntimes gives the number of separate calibrations
        # along the time axis.
        self.Ntimes = 1
        time_array = obs_data['baseline_info'][0]['jdate'][0]
        self.integration_time = np.round(np.mean(np.diff(time_array)) * 24 * 3600, 2)
        self.time_array = np.array([np.mean(time_array)])

        self.Njones = int(cal_data['n_pol'][0])
        # FHD only has the diagonal elements (jxx, jyy) and if there's only one
        # present it must be jxx
        if self.Njones == 1:
            self.jones_array = np.array([-5])
        else:
            self.jones_array = np.array([-5, -6])

        self.telescope_name = uvutils._bytes_to_str(obs_data['instrument'][0])

        self.Nants_data = int(cal_data['n_tile'][0])
        self.Nants_telescope = int(cal_data['n_tile'][0])
        self.antenna_names = np.array([uvutils._bytes_to_str(n) for n in cal_data['tile_names'][0].tolist()])
        self.antenna_numbers = np.arange(self.Nants_telescope)
        self.ant_array = np.arange(self.Nants_data)

        self.set_sky()
        self.sky_field = 'phase center (RA, Dec): ({ra}, {dec})'.format(
            ra=obs_data['orig_phasera'][0], dec=obs_data['orig_phasedec'][0])
        self.sky_catalog = uvutils._bytes_to_str(cal_data['skymodel'][0]['catalog_name'][0])
        self.ref_antenna_name = uvutils._bytes_to_str(cal_data['ref_antenna_name'][0])
        self.Nsources = int(cal_data['skymodel'][0]['n_sources'][0])
        self.baseline_range = [float(cal_data['min_cal_baseline'][0]),
                               float(cal_data['max_cal_baseline'][0])]

        galaxy_model = cal_data['skymodel'][0]['galaxy_model'][0]
        if isinstance(galaxy_model, six.binary_type):  # In Python 3, we sometimes get Unicode, sometimes bytes
            galaxy_model = uvutils._bytes_to_str(galaxy_model)
        if galaxy_model == 0:
            galaxy_model = None
        else:
            galaxy_model = 'gsm'

        diffuse_model = cal_data['skymodel'][0]['diffuse_model'][0]
        if isinstance(diffuse_model, six.binary_type):
            diffuse_model = uvutils._bytes_to_str(diffuse_model)
        if diffuse_model == '':
            diffuse_model = None
        else:
            diffuse_model = os.path.basename(diffuse_model)

        if galaxy_model is not None:
            if diffuse_model is not None:
                self.diffuse_model = galaxy_model + ' + ' + diffuse_model
            else:
                self.diffuse_model = galaxy_model
        elif diffuse_model is not None:
            self.diffuse_model = diffuse_model

        self.gain_convention = 'divide'
        self.x_orientation = 'east'

        self.set_gain()
        fit_gain_array_in = cal_data['gain'][0]
        fit_gain_array = np.zeros(self._gain_array.expected_shape(self), dtype=np.complex_)
        for jones_i, arr in enumerate(fit_gain_array_in):
            fit_gain_array[:, 0, :, 0, jones_i] = arr
        if raw:
            res_gain_array_in = cal_data['gain_residual'][0]
            res_gain_array = np.zeros(self._gain_array.expected_shape(self), dtype=np.complex_)
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
            self.flag_array[ant, :] = 1
        flagged_freqs = np.where(freq_use == 0)[0]
        for freq in flagged_freqs:
            self.flag_array[:, :, freq] = 1

        # currently don't have branch info. may change in future.
        self.git_origin_cal = 'https://github.com/EoRImaging/FHD'
        self.git_hash_cal = uvutils._bytes_to_str(obs_data['code_version'][0])

        self.extra_keywords['autoscal'] = \
            '[' + ', '.join(str(d) for d in cal_data['auto_scale'][0]) + ']'
        self.extra_keywords['nvis_cal'] = cal_data['n_vis_cal'][0]
        self.extra_keywords['time_avg'] = cal_data['time_avg'][0]
        self.extra_keywords['cvgthres'] = cal_data['conv_thresh'][0]
        if 'DELAYS' in obs_data.dtype.names:
            if obs_data['delays'][0] is not None:
                self.extra_keywords['delays'] = \
                    '[' + ', '.join(str(int(d)) for d in obs_data['delays'][0]) + ']'

        if not raw:
            self.extra_keywords['polyfit'] = cal_data['polyfit'][0]
            self.extra_keywords['bandpass'] = cal_data['bandpass'][0]
            self.extra_keywords['mode_fit'] = cal_data['mode_fit'][0]
            self.extra_keywords['amp_deg'] = cal_data['amp_degree'][0]
            self.extra_keywords['phse_deg'] = cal_data['phase_degree'][0]

        if settings_file is not None:
            self.history, self.observer = get_fhd_history(settings_file, return_user=True)
        else:
            warnings.warn('No settings file, history will be incomplete')
            self.history = ''

        if extra_history is not None:
            if isinstance(extra_history, (list, tuple)):
                self.history += '\n' + '\n'.join(extra_history)
            else:
                self.history += '\n' + extra_history

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            if self.history.endswith('\n'):
                self.history += self.pyuvdata_version_str
            else:
                self.history += '\n' + self.pyuvdata_version_str

        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)
