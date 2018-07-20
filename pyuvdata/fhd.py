# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading FHD save files.

"""
from __future__ import absolute_import, division, print_function

from astropy import constants as const
from scipy.io.idl import readsav
from itertools import islice
import numpy as np
import warnings
from .uvdata import UVData
from . import utils as uvutils


def get_fhd_history(settings_file, return_user=False):
    """
    Small function to get the important history from an FHD settings text file.

    Includes information about the command line call, the user, machine name and date
    """
    with open(settings_file, 'r') as f:
        settings_lines = f.readlines()
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
        if (main_loc is not None and command_loc is not None
                and obs_loc is not None and user_line is not None):
            break

    main_lines = settings_lines[main_loc + 1:command_loc]
    command_lines = settings_lines[command_loc + 1:obs_loc]
    history_lines = ['FHD history\n'] + main_lines + command_lines
    for ind, line in enumerate(history_lines):
        history_lines[ind] = line.rstrip().replace('\t', ' ')
    history = '\n'.join(history_lines)
    user = settings_lines[user_line].split()[1]

    if return_user:
        return history, user
    else:
        return history


class FHD(UVData):
    """
    Defines a FHD-specific subclass of UVData for reading FHD save files.
    This class should not be interacted with directly, instead use the read_fhd
    method on the UVData class.
    """

    def read_fhd(self, filelist, use_model=False, run_check=True, check_extra=True,
                 run_check_acceptability=True):
        """
        Read in data from a list of FHD files.

        Args:
            filelist: The list of FHD save files to read from. Must include at
                least one polarization file, a params file and a flag file.
            use_model: Option to read in the model visibilities rather than the
                dirty visibilities. Default is False.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        datafiles = {}
        params_file = None
        flags_file = None
        settings_file = None
        if use_model:
            data_name = '_vis_model_'
        else:
            data_name = '_vis_'
        for file in filelist:
            if file.lower().endswith(data_name + 'xx.sav'):
                datafiles['xx'] = xx_datafile = file
            elif file.lower().endswith(data_name + 'yy.sav'):
                datafiles['yy'] = yy_datafile = file
            elif file.lower().endswith(data_name + 'xy.sav'):
                datafiles['xy'] = xy_datafile = file
            elif file.lower().endswith(data_name + 'yx.sav'):
                datafiles['yx'] = yx_datafile = file
            elif file.lower().endswith('_params.sav'):
                params_file = file
            elif file.lower().endswith('_flags.sav'):
                flags_file = file
            elif file.lower().endswith('_settings.txt'):
                settings_file = file
            else:
                continue

        if len(datafiles) < 1:
            raise Exception('No data files included in file list')
        if params_file is None:
            raise Exception('No params file included in file list')
        if flags_file is None:
            raise Exception('No flags file included in file list')
        if settings_file is None:
            warnings.warn('No settings file included in file list')

        # TODO: add checking to make sure params, flags and datafiles are
        # consistent with each other

        vis_data = {}
        for pol, file in datafiles.items():
            this_dict = readsav(file, python_dict=True)
            if use_model:
                vis_data[pol] = this_dict['vis_model_ptr']
            else:
                vis_data[pol] = this_dict['vis_ptr']
            this_obs = this_dict['obs']
            data_shape = vis_data[pol].shape

        obs = this_obs
        bl_info = obs['BASELINE_INFO'][0]
        meta_data = obs['META_DATA'][0]
        astrometry = obs['ASTR'][0]
        fhd_pol_list = []
        for pol in obs['POL_NAMES'][0]:
            fhd_pol_list.append(pol.decode("utf-8").lower())

        params_dict = readsav(params_file, python_dict=True)
        params = params_dict['params']

        flag_file_dict = readsav(flags_file, python_dict=True)
        # The name for this variable changed recently (July 2016). Test for both.
        vis_weights_data = {}
        if 'flag_arr' in flag_file_dict:
            weights_key = 'flag_arr'
        elif 'vis_weights' in flag_file_dict:
            weights_key = 'vis_weights'
        else:
            raise ValueError('No recognized key for visibility weights in flags_file.')
        for index, w in enumerate(flag_file_dict[weights_key]):
            vis_weights_data[fhd_pol_list[index]] = w

        self.Ntimes = int(obs['N_TIME'][0])
        self.Nbls = int(obs['NBASELINES'][0])
        self.Nblts = data_shape[0]
        self.Nfreqs = int(obs['N_FREQ'][0])
        self.Npols = len(list(vis_data.keys()))
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.vis_units = 'JY'

        lin_pol_order = ['xx', 'yy', 'xy', 'yx']
        linear_pol_dict = dict(zip(lin_pol_order, np.arange(5, 9) * -1))
        pol_list = []
        for pol in lin_pol_order:
            if pol in vis_data:
                pol_list.append(linear_pol_dict[pol])
        self.polarization_array = np.asarray(pol_list)

        self.data_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs,
                                    self.Npols), dtype=np.complex_)
        self.nsample_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs,
                                       self.Npols), dtype=np.float_)
        self.flag_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs,
                                    self.Npols), dtype=np.bool_)
        for pol, vis in vis_data.items():
            pol_i = pol_list.index(linear_pol_dict[pol])
            self.data_array[:, 0, :, pol_i] = vis
            self.flag_array[:, 0, :, pol_i] = vis_weights_data[pol] <= 0
            self.nsample_array[:, 0, :, pol_i] = np.abs(vis_weights_data[pol])

        # In FHD, uvws are in seconds not meters!
        self.uvw_array = np.zeros((self.Nblts, 3))
        self.uvw_array[:, 0] = params['UU'][0] * const.c.to('m/s').value
        self.uvw_array[:, 1] = params['VV'][0] * const.c.to('m/s').value
        self.uvw_array[:, 2] = params['WW'][0] * const.c.to('m/s').value

        # bl_info.JDATE (a vector of length Ntimes) is the only safe date/time
        # to use in FHD files.
        # (obs.JD0 (float) and params.TIME (vector of length Nblts) are
        #   context dependent and are not safe
        #   because they depend on the phasing of the visibilities)
        # the values in bl_info.JDATE are the JD for each integration.
        # We need to expand up to Nblts.
        int_times = bl_info['JDATE'][0]
        bin_offset = bl_info['BIN_OFFSET'][0]
        if self.Ntimes != len(int_times):
            warnings.warn('Ntimes does not match the number of unique times in the data')
        self.time_array = np.zeros(self.Nblts)
        if self.Ntimes == 1:
            self.time_array.fill(int_times)
        else:
            for ii in range(0, len(int_times)):
                if ii < (len(int_times) - 1):
                    self.time_array[bin_offset[ii]:bin_offset[ii + 1]] = int_times[ii]
                else:
                    self.time_array[bin_offset[ii]:] = int_times[ii]

        # Note that FHD antenna arrays are 1-indexed so we subtract 1
        # to get 0-indexed arrays
        self.ant_1_array = bl_info['TILE_A'][0] - 1
        self.ant_2_array = bl_info['TILE_B'][0] - 1

        self.Nants_data = int(len(np.unique(self.ant_1_array.tolist() + self.ant_2_array.tolist())))

        self.antenna_names = [uvutils._bytes_to_str(b) for b in bl_info['TILE_NAMES'][0].tolist()]
        self.Nants_telescope = len(self.antenna_names)
        self.antenna_numbers = np.arange(self.Nants_telescope)

        self.baseline_array = \
            self.antnums_to_baseline(self.ant_1_array,
                                     self.ant_2_array)
        if self.Nbls != len(np.unique(self.baseline_array)):
            warnings.warn('Nbls does not match the number of unique baselines in the data')

        if len(bl_info['FREQ'][0]) != self.Nfreqs:
            warnings.warn('Nfreqs does not match the number of frequencies in the data')
        self.freq_array = np.zeros((self.Nspws, len(bl_info['FREQ'][0])), dtype=np.float_)
        self.freq_array[0, :] = bl_info['FREQ'][0]

        if not np.isclose(obs['OBSRA'][0], obs['PHASERA'][0]) or \
                not np.isclose(obs['OBSDEC'][0], obs['PHASEDEC'][0]):
            warnings.warn('These visibilities may have been phased '
                          'improperly -- without changing the uvw locations')

        self.set_phased()
        self.phase_center_ra_degrees = np.float(obs['OBSRA'][0])
        self.phase_center_dec_degrees = np.float(obs['OBSDEC'][0])

        # this is generated in FHD by subtracting the JD of neighboring
        # integrations. This can have limited accuracy, so it can be slightly
        # off the actual value.
        # (e.g. 1.999426... rather than 2)
        self.integration_time = float(obs['TIME_RES'][0])
        self.channel_width = float(obs['FREQ_RES'][0])

        # # --- observation information ---
        self.telescope_name = str(obs['INSTRUMENT'][0].decode())

        # This is a bit of a kludge because nothing like object_name exists
        # in FHD files.
        # At least for the MWA, obs.ORIG_PHASERA and obs.ORIG_PHASEDEC specify
        # the field the telescope was nominally pointing at
        # (May need to be revisited, but probably isn't too important)
        self.object_name = 'Field RA(deg): ' + str(obs['ORIG_PHASERA'][0]) + \
                           ', Dec:' + str(obs['ORIG_PHASEDEC'][0])
        # For the MWA, this can sometimes be converted to EoR fields
        if self.telescope_name.lower() == 'mwa':
            if np.isclose(obs['ORIG_PHASERA'][0], 0) and \
                    np.isclose(obs['ORIG_PHASEDEC'][0], -27):
                object_name = 'EoR 0 Field'

        self.instrument = self.telescope_name
        self.telescope_location_lat_lon_alt_degrees = (float(obs['LAT'][0]),
                                                       float(obs['LON'][0]),
                                                       float(obs['ALT'][0]))

        self.set_lsts_from_time_array()

        # history: add the first few lines from the settings file
        if settings_file is not None:
            self.history = get_fhd_history(settings_file)
        else:
            self.history = ''

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        self.phase_center_epoch = astrometry['EQUINOX'][0]

        # TODO Once FHD starts reading and saving the antenna table info from
        #    uvfits, that information should be read into the following optional
        #    parameters:
        # 'xyz_telescope_frame'
        # 'x_telescope'
        # 'y_telescope'
        # 'z_telescope'
        # 'antenna_positions'
        # 'GST0'
        # 'RDate'
        # 'earth_omega'
        # 'DUT1'
        # 'TIMESYS'

        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        # check if object has all required uv_properties set
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)
