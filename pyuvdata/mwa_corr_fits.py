# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading MWA correlator FITS files."""
import numpy as np
import warnings

# import tracemalloc

from astropy.io import fits
from astropy.time import Time
from astropy import constants as const
from scipy.special import erf
from math import sqrt

from . import UVData
from . import utils as uvutils


def input_output_mapping():
    """Build a mapping dictionary from pfb input to output numbers."""
    # the polyphase filter bank maps inputs to outputs, which the MWA
    # correlator then records as the antenna indices.
    # the following is taken from mwa_build_lfiles/mwac_utils.c
    # inputs are mapped to outputs via pfb_mapper as follows
    # (from mwa_build_lfiles/antenna_mapping.h):
    # floor(index/4) + index%4 * 16 = input
    # for the first 64 outputs, pfb_mapper[output] = input
    pfb_mapper = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
                  4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
                  8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                  12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47,
                  63]
    # build a mapper for all 256 inputs
    pfb_inputs_to_outputs = {}
    for p in range(4):
        for i in range(64):
            pfb_inputs_to_outputs[pfb_mapper[i] + p * 64] = p * 64 + i
    return pfb_inputs_to_outputs


def bisection_search(sortlist, value):
    """
    Implement a simple bisection search function.

    Parameters
    ----------
    sortlist: sorted list
        List to be searched for value.
    value: float
        Value to be found in list.
    """
    # find nearest value in list of keys
    bottom = 0
    top = len(sortlist) - 1
    while True:
        mid = int((top + bottom)/2)
        if value < sortlist[bottom]:
            print(str(value) + 'rounded up to' + str(sortlist[bottom]))
            return sortlist[bottom]
        elif value > sortlist[top]:
            print(str(value) + 'clipped to' + str(sortlist[top]))
            return sortlist[top]
        elif value == sortlist[mid]:
            return value
        elif value < sortlist[mid] and value > sortlist[mid - 1]:
            return (sortlist[mid - 1], sortlist[mid])
        elif value > sortlist[mid] and value < sortlist[mid + 1]:
            return (sortlist[mid], sortlist[mid + 1])
        elif value < sortlist[mid]:
            top = mid
        else:
            bottom = mid


def linear_interp(x, x1, x2, y1, y2):
    """
    Interpolate a function between two points.

    Parameters
    ----------
    x: float
        The function input value.
    x1: float
        Function input value at first point.
    x2: float
        Function input value at second point.
    y1: float
        Function output value at first point.
    y2: float
        Function output value at second point.
    """
    y = (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1)
    return y


def sig_lookup_table(x, bits):
    """
    Build a lookup table to correct the xx and yy autos.

    Parameters
    ----------
    x : numpy array
        Array to act as inputs into the inverse correction function.
    bits : int
        Number of quantization bits.
    """
    # note: this table maps sighat to sig; correlator outputs are sighat^2
    # assign the upper level of the quantization
    m = 2 ** (bits - 1) - 1
    # create an array
    y = np.array([range(m)])
    # create a sparse array to perform the function accross
    xx, yy = np.meshgrid(x, y, sparse=True)
    # compute terms of summation
    z = (2 * yy + 1) * erf((yy + .5) / (xx * sqrt(2)))
    # sum terms
    zsum = z.sum(axis=0)
    # create a new array that is the standard deviation of the quantized signal
    sighat = np.sqrt(m ** 2 - zsum)
    # put arrays into a dictionary for lookup
    sig_table = {i: j for i, j in zip(sighat, x)}
    return sig_table


def cov_lookup_table(rho, bits, xsig, ysig):
    # TODO: fix this docstring
    '''corr is an analog correlator output, bits is the number of quantization bits,
    xsig and ysig are the corresponding standard deviations of the inputs to x.
    This function returns the expected digitized correlator output from analog signal inputs'''
    # initialize data structure: dict of dicts
    # TODO: change this for single xsig, ysig
    # note: correlator output is sig^2, so formula is modified to take sig^2
    cov_lookup = {}
    for i in range(len(xsig)):
        for j in range(i, len(ysig)):
            cov_lookup[(xsig[i], ysig[j])] = {}
    # assign the upper level of the quantization
    lev = 2 ** (bits - 1) - 1
    # create variables for summation
    level_sum = np.arange(-lev, lev, 1)
    # create an integration grid for midpoint summation
    for k in rho:
        x = np.arange(.000005, k, .00001)
        ii, jj, kk, xxsig, yysig = np.meshgrid(level_sum, level_sum, x, xsig, ysig, sparse=True)
        # set up summation in integrand
        z = np.exp(-(1 / (2 * (1 - kk ** 2))) * (((ii + .5) ** 2 / xxsig) + ((jj + .5) ** 2 / yysig) - 2 * kk * (ii + .5) * (jj + .5) / (np.sqrt(xxsig) * np.sqrt(yysig))))
        # sum over i
        zs1 = z.sum(0)
        # sum over j
        zs2 = zs1.sum(0)
        # multiply by a term with x to complete the integrand
        integrand = np.multiply(1 / np.sqrt(1 - kk ** 2), zs2)
        integrand = integrand[0, 0, :, :, :]
        # compute the midpoint Riemann sum
        result = (1 / (2 * np.pi)) * .00001 * integrand.sum(0)
        for p in range(len(xsig)):
            for q in range(p, len(ysig)):
                cov_lookup[(xsig[p], ysig[q])][result[p, q]] = k
    return cov_lookup


class MWACorrFITS(UVData):
    """
    UVData subclass for reading MWA correlator fits files.

    This class should not be interacted with directly; instead use the
    read_mwa_corr_fits method on the UVData class.
    """

    def correct_cable_length(self, cable_lens):
        """
        Apply a cable length correction to the data array.

        Parameters
        ----------
        cable_lens : list of strings
        A list of strings containing the cable lengths for each antenna.
        """
        # "the velocity factor of electic fields in RG-6 like coax"
        # from MWA_Tools/CONV2UVFITS/convutils.h
        v_factor = 1.204
        # check if the cable length already has the velocity factor applied
        cable_array = []
        for i in cable_lens:
            if i[0:3] == 'EL_':
                cable_array.append(float(i[3:]))
            else:
                cable_array.append(float(i) * v_factor)
        # build array of differences
        cable_len_diffs = np.zeros((self.Nblts, 1))
        for j in range(self.Nblts):
            cable_len_diffs[j] = cable_array[self.ant_2_array[j]] - cable_array[self.ant_1_array[j]]
        self.data_array *= np.exp(-1j * 2 * np.pi * cable_len_diffs / const.c.to('m/s').value
                                  * self.freq_array.reshape(1, self.Nfreqs))[:, :, None]


    def flag_init(self, num_fine_chan, edge_width=80e3, start_flag=2.0,
                  end_flag=2.0, flag_dc_offset=True):
        """
        Do routine flagging of the edges, beginning and end of obs, as well as
        the center fine channel of each coarse channel.

        Parameters
        ----------
        edge_width: float
            The width to flag on the edge of each coarse channel, in hz. Set to
            0 for no edge flagging.
        start_flag: float
            The number of seconds to flag at the beginning of the observation.
            Set to 0 for no flagging.
        end_flag: floats
            The number of seconds to flag at the end of the observation. Set to
            0 for no flagging.
        flag_dc_offset: bool
            Set to True to flag the center fine channel of each coarse channel.

        Raises
        ------
        ValueError
            If edge_width is not an integer multiple of the channel_width of the data (0 also acceptable).
            If start_flag is not an integer multiple of the integration time (0 also acceptable).
            If end_flag is not an integer multiple of the integration time (0 also acceptable).
        """
        if (edge_width % self.channel_width) > 0:
            raise ValueError("The edge_width must be an integer multiple of the"
                             "channel_width of the data or zero.")
        if (start_flag % self.integration_time[0]) > 0:
            raise ValueError("The start_flag must be an integer multiple of the"
                             "integration_time of the data or zero.")
        if (end_flag % self.integration_time[0]) > 0:
            raise ValueError("The end_flag must be an integer multiple of the"
                             "integration_time of the data or zero.")

        num_ch_flag = int(edge_width / self.channel_width)
        num_start_flag = int(start_flag / self.integration_time[0])
        num_end_flag = int(end_flag / self.integration_time[0])

        if num_ch_flag > 0:
            edge_inds = []
            for ch_count in range(num_ch_flag):
                # count up from the left
                left_chans = list(range(ch_count, self.Nfreqs, num_fine_chan))
                # count down from the right
                right_chans = list(range(self.Nfreqs - 1 - ch_count, 0, -num_fine_chan))
                edge_inds = edge_inds + left_chans + right_chans

            self.flag_array[:, :, edge_inds, :] = True

        if flag_dc_offset:
            center_inds = list(range(num_fine_chan // 2, self.Nfreqs, num_fine_chan))

            self.flag_array[:, :, center_inds, :] = True

        if (num_start_flag > 0) or (num_end_flag > 0):
            shape = self.flag_array.shape
            reshape = [self.Ntimes, self.Nbls, self.Nspws, self.Nfreqs, self.Npols]
            self.flag_array = np.reshape(self.flag_array, reshape)
            if num_start_flag > 0:
                self.flag_array[:num_start_flag, :, :, :, :] = True
            if num_end_flag > 0:
                self.flag_array[-num_end_flag:, :, :, :, :] = True
            self.flag_array = np.reshape(self.flag_array, shape)


    def van_vleck_correction(self):
        """Apply a van vleck correction to the data array."""
        # get indices for autos
        # print(self.data_array.shape)
        autos = np.where(self.ant_1_array[0: self.Nbls] == self.ant_2_array[0: self.Nbls])[0]
        # get indices for crosses
        crosses = np.where(self.ant_1_array[0: self.Nbls] != self.ant_2_array[0: self.Nbls])[0]
        # generate dict for getting auto pols
        # polarizations are ordered yy, yx, xy, xx
        # TODO: generalize this for any polarization ordering
        pol_dict = {0: (0, 0), 1: (0, 3), 2: (3, 0), 3: (3, 3)}
        # so one weird thing is at low sigma things get rounded up to 0.06
        # create correction matrices
        # print(self.data_array.real[:, autos, :, :].shape)
        # TODO: think about how to make this
        min_auto = np.min(self.data_array.real[:, autos, :, [[0], [3]]][self.data_array.real[:, autos, :, [[0], [3]]] > 0.0])
        print(min_auto)
        range_min = np.max([1.06, min_auto])
        max_auto = np.max(self.data_array.real[:, autos, :, [[0], [3]]])
        print(max_auto)
        # TODO: think about how fine to make this mesh        
        sigs = np.arange(range_min - 1, max_auto + 1, 0.000001)
        sig_lookup = sig_lookup_table(sigs, 4)
        sig_keys = sorted(sig_lookup.keys())

        # print('after building sigma lookup table')
        # print(tracemalloc.get_traced_memory())

        # at this point, data_array.shape = (Ntimes, Nbls, Nfreqs, Npols)
        # TODO: generalize this for any data_array shape
        # correct xx and yy autos
        pols = [0, 3]
        for i in pols:
            for j in range(self.Nfreqs):
                print('processessing polarization ' + str(i) + 'and frequency ' + str(j))
                for k in autos:
                    for l in range(self.Ntimes):
                        # print('data for auto' + str(k) + 'and time' + str(self.time_array[l]))
                        # do a bisection search through sorted(sig_lookup.keys)
                        # don't correct zeros
                        if self.data_array.real[l, k, j, i] != 0.0:
                            # need to take the square root before correcting
                            sig_hat = bisection_search(sig_keys, np.sqrt(self.data_array.real[l, k, j, i]))
                            if isinstance(sig_hat, tuple):
                                # do a linear interpolation
                                sig_corr = linear_interp(self.data_array.real[l, k, j, i],
                                                         sig_hat[0], sig_hat[1],
                                                         sig_lookup[sig_hat[0]], sig_lookup[sig_hat[1]])
                                # print(str(self.data_array.real[l, k, j, i]) + 'converted to' + str(sig_corr))
                                self.data_array.real[l, k, j, i] = sig_corr**2
                            else:
                                # correct self.data_array.real[l, k, j, i]
                                # print(str(self.data_array.real[l, k, j, i]) + 'converted to' + str(sig_lookup[sigkey]))
                                self.data_array.real[l, k, j, i] = sig_lookup[sig_hat]
                        else:
                            continue
        # del(sig_lookup)
        # del(sig_keys)
        # print('after correcting autos')
        # print(tracemalloc.get_traced_memory())

#==============================================================================
#         pols = [0, 1, 2, 3]
#         for i in pols:
#             # look up auto pol inds
#             pol_inds = pol_dict[i]
#             if i == 1 or i == 2:
#                 bls = np.arange(self.Nbls)
#             else:
#                 bls = crosses
#             for j in range(self.Nfreqs):
#                 # adjust freq ind if necessary
#                 for k in bls:
#                     auto1 = autos[self.ant_1_array[k]]
#                     auto2 = autos[self.ant_2_array[k]]
#                     # get indices/values for xx/yy autos: sig1, sig2
#                     for l in range(self.Ntimes):
#                         negative = False
#                         sig1 = self.data_array[l, auto1, j, pol_inds[0]].real
#                         sig2 = self.data_array[l, auto2, j, pol_inds[1]].real
#==============================================================================
                        # generate the lookups for these sigs
                        # get a reasonable range for rho
                        # cov_lookup = cov_lookup_table(rho, sig1, sig2)
                        # the keys of that dict are kappahat
                        # I search these keys to get the right rho value
                        # covkey_real = bisection_search(sorted(vv_table.keys()), self.data_array.real[l, k, j, i])
                        # covkey_imag = bisection_search(sorted(vv_table.keys()), self.data_array.imag[l, k, j, i])
                        # self.data_array[l, k, j, i] = vv_table[covkey_real] + 1j * vv_table[covkey_imag]
                        # print('data for bls' + str(k) + 'and time' + str(self.time_array[l]))
                        # need to correct absolute value, so check if less than zero
                        # correct self.data_array.real[l, k, j, i]
                        # correct self.data_array.imag[l, k, j, i]


    def read_mwa_corr_fits(self, filelist, use_cotter_flags=False, correct_cable_len=False,
                           phase_to_pointing_center=False, correct_van_vleck=False,
                           run_check=True, check_extra=True, run_check_acceptability=True,
			   flag_init=True, edge_width=80e3, start_flag=2.0, end_flag=2.0,
			   flag_dc_offset=True):
        """
        Read in MWA correlator gpu box files.

        Parameters
        ----------
        filelist : list of str
            The list of MWA correlator files to read from. Must include at
            least one fits file and only one metafits file per data set.
            Can also be a list of lists to read multiple data sets.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
        use_cotter_flags : bool
            Option to use cotter output mwaf flag files. Otherwise flagging
            will only be applied to missing data and bad antennas.
        correct_cable_len : bool
            Option to apply a cable delay correction.
        phase_to_pointing_center : bool
            Option to phase to the observation pointing center.
        correct_van_vleck : bool
            Option to apply a van vleck correction.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done).
        flag_init: bool
            Set to True in order to do routine flagging of coarse channel edges,
            start or end integrations, or the center fine channel of each coarse
            channel. See associated keywords.
        edge_width: float
            Only used if flag_init is True. The width to flag on the edge of
            each coarse channel, in hz. Errors if not equal to integer multiple
            of channel_width. Set to 0 for no edge flagging.
        start_flag: float
            Only used if flag_init is True. The number of seconds to flag at the
            beginning of the observation. Set to 0 for no flagging. Errors if
            not equal to an integer multiple of the integration time.
        end_flag: floats
            Only used if flag_init is True. The number of seconds to flag at the
            end of the observation. Set to 0 for no flagging. Errors if not
            equal to an integer multiple of the integration time.
        flag_dc_offset: bool
            Only used if flag_init is True. Set to True to flag the center fine
            channel of each coarse channel.

        Raises
        ------
        ValueError
            If required files are missing or multiple files metafits files are included in filelist.
            If files from different observations are included in filelist.
            If files in fileslist have different fine channel widths
            If file types other than fits, metafits, and mwaf files are included in filelist.

        """
        # tracemalloc.start()
        metafits_file = None
        obs_id = None
        bscale = None
        file_dict = {}
        start_time = 0.0
        end_time = 0.0
        included_file_nums = []
        cotter_warning = False
        num_fine_chans = 0

        # iterate through files and organize
        # create a list of included coarse channels
        # find the first and last times that have data
        for file in filelist:
            if file.lower().endswith('.metafits'):
                # force only one metafits file
                if metafits_file is not None:
                    raise ValueError('multiple metafits files in filelist')
                metafits_file = file
            # organize data files
            elif file.lower().endswith('.fits'):
                # get the file number from the file name;
                # this will later be mapped to a coarse channel
                file_num = int(file.split('_')[-2][-2:])
                if file_num not in included_file_nums:
                    included_file_nums.append(file_num)
                with fits.open(file) as data:
                    # check obs id
                    if obs_id is None:
                        obs_id = data[0].header['OBSID']
                    else:
                        if data[0].header['OBSID'] != obs_id:
                            raise ValueError('files from different observations submitted in same list')
                    # check headers for first and last times containing data
                    first_time = data[1].header['TIME'] + data[1].header['MILLITIM'] / 1000.0
                    last_time = data[-1].header['TIME'] + data[-1].header['MILLITIM'] / 1000.0
                    if start_time == 0.0:
                        start_time = first_time
                    elif start_time > first_time:
                        start_time = first_time
                    if end_time < last_time:
                        end_time = last_time
                    # get number of fine channels
                    if num_fine_chans == 0:
                        num_fine_chans = data[1].header['NAXIS2']
                    elif num_fine_chans != data[1].header['NAXIS2']:
                        raise ValueError('files submitted have different fine channel widths')
                    # get scaling info
                    if bscale is None:
                        bscale = data[0].header['BSCALE']
                # organize files
                if 'data' not in file_dict.keys():
                    file_dict['data'] = [file]
                else:
                    file_dict['data'].append(file)
            # look for flag files
            elif file.lower().endswith('.mwaf'):
                if use_cotter_flags is False and cotter_warning is False:
                    warnings.warn('mwaf files submitted with use_cotter_flags=False')
                    cotter_warning = True
                elif 'flags' not in file_dict.keys():
                    file_dict['flags'] = [file]
                else:
                    file_dict['flags'].append(file)
            else:
                raise ValueError('only fits, metafits, and mwaf files supported')

        # checks:
        if metafits_file is None:
            raise ValueError('no metafits file submitted')
        if 'data' not in file_dict.keys():
            raise ValueError('no data files submitted')
        if 'flags' not in file_dict.keys() and use_cotter_flags:
            raise ValueError('no flag files submitted. Rerun with flag files \
                             or use_cotter_flags=False')

        # first set parameters that are always true
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.phase_type = 'drift'
        self.vis_units = 'uncalib'
        self.Npols = 4
        self.xorientation = 'east'

        # get information from metafits file
        with fits.open(metafits_file, memmap=True) as meta:
            meta_hdr = meta[0].header

            # get a list of coarse channels
            coarse_chans = meta_hdr['CHANNELS'].split(',')
            coarse_chans = np.array(sorted([int(i) for i in coarse_chans]))

            # integration time in seconds
            int_time = meta_hdr['INTTIME']

            # pointing center in degrees
            ra_deg = meta_hdr['RA']
            dec_deg = meta_hdr['DEC']
            ra_rad = np.pi * ra_deg / 180
            dec_rad = np.pi * dec_deg / 180

            # get parameters from header
            # this assumes no averaging by this code so will need to be updated
            self.channel_width = float(meta_hdr.pop('FINECHAN') * 1000)
            self.history = str(meta_hdr['HISTORY'])
            if not uvutils._check_history_version(self.history,
                                                  self.pyuvdata_version_str):
                self.history += self.pyuvdata_version_str
            meta_hdr.remove('HISTORY', remove_all=True)
            self.instrument = meta_hdr['TELESCOP']
            self.telescope_name = meta_hdr.pop('TELESCOP')
            self.object_name = meta_hdr.pop('FILENAME')

            # get rid of the instrument keyword so it doesn't get put back in
            meta_hdr.remove('INSTRUME')
            # get rid of keywords that uvfits.py gets rid of
            bad_keys = ['SIMPLE', 'EXTEND', 'BITPIX', 'NAXIS', 'DATE-OBS']
            for key in bad_keys:
                meta_hdr.remove(key, remove_all=True)
            # store remaining keys in extra keywords
            for key in meta_hdr:
                if key == 'COMMENT':
                    self.extra_keywords[key] = str(meta_hdr.get(key))
                elif key != '':
                    self.extra_keywords[key] = meta_hdr.get(key)

            # get antenna data from metafits file table
            meta_tbl = meta[1].data

            # because of polarization, each antenna # is listed twice
            antenna_numbers = meta_tbl['Antenna'][1::2]
            antenna_names = meta_tbl['TileName'][1::2]
            antenna_flags = meta_tbl['Flag'][1::2]
            cable_lens = meta_tbl['Length'][1::2]

            # get antenna postions in enu coordinates
            antenna_positions = np.zeros((len(antenna_numbers), 3))
            antenna_positions[:, 0] = meta_tbl['East'][1::2]
            antenna_positions[:, 1] = meta_tbl['North'][1::2]
            antenna_positions[:, 2] = meta_tbl['Height'][1::2]

        # reorder antenna parameters from metafits ordering
        reordered_inds = antenna_numbers.argsort()
        self.antenna_numbers = antenna_numbers[reordered_inds]
        self.antenna_names = list(antenna_names[reordered_inds])
        antenna_positions = antenna_positions[reordered_inds, :]
        antenna_flags = antenna_flags[reordered_inds]
        cable_lens = cable_lens[reordered_inds]

        # find flagged antenna
        flagged_ants = self.antenna_numbers[np.where(antenna_flags == 1)]

        # set parameters from other parameters
        self.Nants_data = len(self.antenna_numbers)
        self.Nants_telescope = len(self.antenna_numbers)
        self.Nbls = int(len(self.antenna_numbers) * (len(self.antenna_numbers) + 1) / 2.0)

        # get telescope parameters
        self.set_telescope_params()

        # build time array of centers
        time_array = np.arange(start_time + int_time / 2.0, end_time
                               + int_time / 2.0 + int_time, int_time)

        # convert from unix to julian times
        julian_time_array = [Time(i, format='unix', scale='utc').jd
                             for i in time_array]

        # convert to integers
        float_time_array = np.array([float(i) for i in julian_time_array])
        # build into time array
        self.time_array = np.repeat(float_time_array, self.Nbls)

        self.Ntimes = len(time_array)

        self.Nblts = int(self.Nbls * self.Ntimes)

        # convert times to lst
        self.lst_array = uvutils.get_lst_for_time(self.time_array,
                                                  *self.telescope_location_lat_lon_alt_degrees)

        self.integration_time = np.array([int_time for i in range(self.Nblts)])

        # convert antenna positions from enu to ecef
        # antenna positions are "relative to
        # the centre of the array in local topocentric \"east\", \"north\",
        # \"height\". Units are meters."
        antenna_positions_ecef = uvutils.ECEF_from_ENU(antenna_positions,
                                                       *self.telescope_location_lat_lon_alt)
        # make antenna positions relative to telescope location
        self.antenna_positions = antenna_positions_ecef - self.telescope_location

        # make initial antenna arrays, where ant_1 <= ant_2
        ant_1_array = []
        ant_2_array = []
        for i in range(self.Nants_telescope):
            for j in range(i, self.Nants_telescope):
                ant_1_array.append(i)
                ant_2_array.append(j)

        self.ant_1_array = np.tile(np.array(ant_1_array), self.Ntimes)
        self.ant_2_array = np.tile(np.array(ant_2_array), self.Ntimes)

        self.baseline_array = \
            self.antnums_to_baseline(self.ant_1_array, self.ant_2_array)

        # create self.uvw_array
        self.set_uvws_from_antenna_positions(allow_phasing=False)

        # coarse channel mapping:
        # channels in group 0-128 go in order; channels in group 129-155 go in
        # reverse order
        # that is, if the lowest channel is 127, it will be assigned to the
        # first file
        # channel 128 will be assigned to the second file
        # then the highest channel will be assigned to the third file
        # and the next hightest channel assigned to the fourth file, and so on
        count = 0
        # count the number of channels that are in group 0-128
        for i in coarse_chans:
            if i <= 128:
                count += 1
        # map all file numbers to coarse channel numbers
        file_nums_to_coarse = {i + 1: coarse_chans[i] if i < count else
                               coarse_chans[(len(coarse_chans) + count - i - 1)]
                               for i in range(len(coarse_chans))}
        # map included coarse channels to file numbers
        coarse_to_incl_files = {}
        for i in included_file_nums:
            coarse_to_incl_files[file_nums_to_coarse[i]] = i
        # sort included coarse channels
        included_coarse_chans = sorted(coarse_to_incl_files.keys())
        # map included file numbers to an index that orders them
        file_nums_to_index = {}
        for i in included_coarse_chans:
            file_nums_to_index[coarse_to_incl_files[i]] = included_coarse_chans.index(i)
        # check that coarse channels are contiguous.
        chans = np.array(included_coarse_chans)
        for i in np.diff(chans):
            if i != 1:
                warnings.warn('coarse channels are not contiguous for this observation')
                break

        # warn user if not all coarse channels are included
        if len(included_coarse_chans) != len(coarse_chans):
            warnings.warn('some coarse channel files were not submitted')

        # build frequency array
        self.Nfreqs = len(included_coarse_chans) * num_fine_chans
        self.freq_array = np.zeros((self.Nspws, self.Nfreqs))

        # each coarse channel is split into 128 fine channels of width 10 kHz.
        # The first fine channel for each coarse channel is centered on the
        # lower bound frequency of that channel and its center frequency is
        # computed as fine_center = coarse_channel_number * 1280-640 (kHz).
        # If the fine channels have been averaged (added) by some factor, the
        # center of the resulting channel is found by averaging the centers of
        # the first and last fine channels it is made up of.
        # That is, avg_fine_center=(lowest_fine_center+highest_fine_center)/2
        # where highest_fine_center=lowest_fine_center+(avg_factor-1)*10 kHz
        # so avg_fine_center=(lowest_fine_center+lowest_fine_center+(avg_factor-1)*10)/2
        #                   =lowest_fine_center+((avg_factor-1)*10)/2
        #                   =lowest_fine_center+offset
        # Calculate offset=((avg_factor-1)*10)/2 to build the frequency array
        avg_factor = self.channel_width / 10000
        width = self.channel_width / 1000
        offset = (avg_factor - 1) * 10 / 2.0

        for i in range(len(included_coarse_chans)):
            # get the lowest fine freq of the coarse channel (kHz)
            lower_fine_freq = included_coarse_chans[i] * 1280 - 640
            # find the center of the lowest averaged channel
            first_center = lower_fine_freq + offset
            # add the channel centers for this coarse channel into
            # the frequency array (converting from kHz to Hz)
            self.freq_array[0, int(i * num_fine_chans):int((i + 1) * num_fine_chans)] = \
                np.arange(first_center, first_center + num_fine_chans * width, width) * 1000

        # print('just before data dump')        
        # print(tracemalloc.get_traced_memory())

        # read data into an array with dimensions (time, freq, baselines*pols)
        data_dump = np.zeros((self.Ntimes, self.Nfreqs, self.Nbls * self.Npols), dtype=np.complex64)
        # read data files
        for file in file_dict['data']:
            # get the file number from the file name
            file_num = int(file.split('_')[-2][-2:])
            # map file number to frequency index
            freq_ind = file_nums_to_index[file_num] * num_fine_chans
            with fits.open(file, memmap=False, do_not_scale_image_data=False) as hdu_list:
                # count number of times
                end_list = len(hdu_list)
                for i in range(1, end_list):
                    time = hdu_list[i].header['TIME'] + hdu_list[i].header['MILLITIM'] / 1000.0 + int_time / 2.0
                    time_ind = np.where(time_array == time)[0][0]
                    # dump data into matrix
                    # and take data from real to complex numbers
                    data_dump[time_ind, freq_ind:freq_ind + num_fine_chans, :] = \
                        hdu_list[i].data[:, 0::2] + 1j * hdu_list[i].data[:, 1::2]

        # print('just after data dump, before reordering')
        # print(tracemalloc.get_traced_memory())

        # polarizations are ordered yy, yx, xy, xx
        self.polarization_array = np.array([-6, -8, -7, -5])

        # initialize matrices for data reordering
        self.nsample_array = np.zeros((self.Ntimes, self.Nbls, self.Nfreqs, self.Npols), dtype=np.float32)
        self.data_array = np.zeros((self.Ntimes, self.Nbls, self.Nfreqs, self.Npols), dtype=np.complex64)
        self.flag_array = np.full((self.Ntimes, self.Nbls, self.Nfreqs, self.Npols), True)
        # build mapper from antenna numbers and polarizations to pfb inputs
        corr_ants_to_pfb_inputs = {}
        for i in range(len(antenna_numbers)):
            for p in range(2):
                corr_ants_to_pfb_inputs[(antenna_numbers[i], p)] = 2 * i + p

        # for mapping, start with a pair of antennas/polarizations
        # this is the pair we want to find the data for
        # map the pair to the corresponding pfb input indices
        # map the pfb input indices to the pfb output indices
        # these are the indices for the data corresponding to the initial antenna/pol pair
        pfb_inputs_to_outputs = input_output_mapping()
        for ant1 in range(128):
            for ant2 in range(ant1, 128):
                for p1 in range(2):
                    for p2 in range(2):
                        # generate the indices in self.data_array for this combination
                        # baselines are ordered (0,0),(0,1),...,(0,127),(1,1),.....
                        # polarizion of 0 (1) corresponds to y (x)
                        pol_ind = int(2 * p1 + p2)
                        bls_ind = int(128 * ant1 - ant1 * (ant1 + 1) / 2 + ant2)
                        # find the pfb input indices for this combination
                        (ind1_1, ind1_2) = (corr_ants_to_pfb_inputs[(ant1, p1)],
                                            corr_ants_to_pfb_inputs[(ant2, p2)])
                        # find the pfb output indices
                        (ind2_1, ind2_2) = (pfb_inputs_to_outputs[(ind1_1)],
                                            pfb_inputs_to_outputs[(ind1_2)])
                        out_ant1 = int(ind2_1 / 2)
                        out_ant2 = int(ind2_2 / 2)
                        out_p1 = ind2_1 % 2
                        out_p2 = ind2_2 % 2
                        # the correlator has ind2_2 <= ind2_1 except for
                        # redundant data. The redundant data is not perfectly
                        # redundant; sometimes the values of redundant data
                        # are off by one in the imaginary part.
                        # For consistency, we are ignoring the redundant values
                        # that have ind2_2 > ind2_1
                        if ind2_2 > ind2_1:
                            # get the index for the data
                            data_index = int(2 * out_ant2 * (out_ant2 + 1) + 4 * out_ant1 + 2 * out_p2 + out_p1)
                            # need to take the complex conjugate of the data
                            self.data_array[:, bls_ind, :, pol_ind] = np.conj(data_dump[:, :, data_index])
                        else:
                            data_index = int(2 * out_ant1 * (out_ant1 + 1) + 4 * out_ant2 + 2 * out_p1 + out_p2)
                            self.data_array[:, bls_ind, :, pol_ind] = data_dump[:, :, data_index]
                        # unflag where the data is
                        self.flag_array[:, bls_ind, :, pol_ind] = False
                        # nsamples = 1 where the data is
                        self.nsample_array[:, bls_ind, :, pol_ind] = 1.0

        # generage baseline flags for flagged ants
        bad_ant_inds = []
        for ant1 in range(128):
            for ant2 in range(ant1, 128):
                if ant1 in flagged_ants or ant2 in flagged_ants:
                    bad_ant_inds.append(int(128 * ant1 - ant1 * (ant1 + 1) / 2 + ant2))

        self.flag_array[:, bad_ant_inds, :, :] = True

        # print('after reordering, before van vleck')
        # print(tracemalloc.get_traced_memory())

        # TODO: think about placing this later in code        
        # van vleck correction
        if correct_van_vleck:
            # scale the data
            # number of samples per 10 kHz fine channel is 20000/s
            # TODO: think about nsamples
            nsamples = 20000 * self.integration_time[0] * self.channel_width / 10000
            # print('nsamples: ' + str(nsamples))
            # TODO: think about rounding
            # round the data after scaling to get rid of hash?
            # round_factor = int(nsamples/10000)+5
            self.data_array = self.data_array / (nsamples * bscale)
            # take advantage of cicular polarization! divide by two
            self.data_array = self.data_array / 2.0
            # self.data_array = np.around(self.data_array, round_factor)
            self.van_vleck_correction()
            # rescale the data
            self.data_array = self.data_array * (nsamples * bscale*2)

        # print('after van vleck correction')
        # print(tracemalloc.get_traced_memory())

        # combine baseline and time axes
        self.data_array = self.data_array.reshape((self.Nblts, self.Nfreqs, self.Npols))
        self.flag_array = self.flag_array.reshape((self.Nblts, self.Nfreqs, self.Npols))
        self.nsample_array = self.nsample_array.reshape((self.Nblts, self.Nfreqs, self.Npols))

        # cable delay corrections
        if correct_cable_len:
            self.correct_cable_length(cable_lens)

        # add spectral window index
        self.data_array = self.data_array[:, np.newaxis, :, :]
        self.flag_array = self.flag_array[:, np.newaxis, :, :]
        self.nsample_array = self.nsample_array[:, np.newaxis, :, :]

        # because of an annoying discrepancy between file conventions, in order
        # to be consistent with the uvw vector direction, all the data must
        # be conjugated
        self.data_array = np.conj(self.data_array)

        # reorder polarizations
        self.reorder_pols()

        # phasing
        if phase_to_pointing_center:
            self.phase(ra_rad, dec_rad)

        if flag_init:
            self.flag_init(num_fine_chans, edge_width=edge_width,
                           start_flag=start_flag, end_flag=end_flag,
                           flag_dc_offset=flag_dc_offset)

        if use_cotter_flags:
            raise NotImplementedError('reading in cotter flag files is not yet available')

        # tracemalloc.stop()
