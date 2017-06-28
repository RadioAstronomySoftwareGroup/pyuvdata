"""Class for reading and writing Miriad files."""
from astropy import constants as const
import os
import shutil
import numpy as np
import warnings
import aipy as a
from uvdata import UVData
import telescopes as uvtel


class Miriad(UVData):
    """
    Defines a Miriad-specific subclass of UVData for reading and writing Miriad files.
    This class should not be interacted with directly, instead use the read_miriad
    and write_miriad methods on the UVData class.
    """

    def _pol_to_ind(self, pol):
        if self.polarization_array is None:
            raise(ValueError, "Can't index polarization {p} because "
                  "polarization_array is not set".format(p=pol))
        pol_ind = np.argwhere(self.polarization_array == pol)
        if len(pol_ind) != 1:
            raise(ValueError, "multiple matches for pol={pol} in "
                  "polarization_array".format(pol=pol))
        return pol_ind

    def read_miriad(self, filepath, correct_lat_lon=True, run_check=True,
                    check_extra=True, run_check_acceptability=True, phase_type=None):
        """
        Read in data from a miriad file.

        Args:
            filepath: The miriad file directory to read from.
            correct_lat_lon: flag -- that only matters if altitude is missing --
                to update the latitude and longitude from the known_telescopes list
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        if not os.path.exists(filepath):
            raise(IOError, filepath + ' not found')
        uv = a.miriad.UV(filepath)

        miriad_header_data = {'Nfreqs': 'nchan',
                              'Npols': 'npol',
                              'integration_time': 'inttime',
                              'channel_width': 'sdf',  # in Ghz!
                              'object_name': 'source',
                              'telescope_name': 'telescop'
                              }
        for item in miriad_header_data:
            if isinstance(uv[miriad_header_data[item]], str):
                header_value = uv[miriad_header_data[item]].replace('\x00', '')
            else:
                header_value = uv[miriad_header_data[item]]
            setattr(self, item, header_value)

        latitude = uv['latitud']  # in units of radians
        longitude = uv['longitu']
        try:
            altitude = uv['altitude']
            self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
        except(KeyError):
            # get info from known telescopes. Check to make sure the lat/lon values match reasonably well
            telescope_obj = uvtel.get_telescope(self.telescope_name)
            if telescope_obj is not False:

                tol = 2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0)  # 1mas in radians
                lat_close = np.isclose(telescope_obj.telescope_location_lat_lon_alt[0],
                                       latitude, rtol=0, atol=tol)
                lon_close = np.isclose(telescope_obj.telescope_location_lat_lon_alt[1],
                                       longitude, rtol=0, atol=tol)
                if correct_lat_lon:
                    self.telescope_location_lat_lon_alt = telescope_obj.telescope_location_lat_lon_alt
                else:
                    self.telescope_location_lat_lon_alt = (latitude, longitude, telescope_obj.telescope_location_lat_lon_alt[2])
                if lat_close and lon_close:
                    if correct_lat_lon:
                        warnings.warn('Altitude is not present in Miriad file, '
                                      'using known location values for '
                                      '{telescope_name}.'.format(telescope_name=telescope_obj.telescope_name))
                    else:
                        warnings.warn('Altitude is not present in Miriad file, '
                                      'using known location altitude value '
                                      'for {telescope_name} and lat/lon from '
                                      'file.'.format(telescope_name=telescope_obj.telescope_name))
                else:
                    warn_string = ('Altitude is not present in file ')
                    if not lat_close and not lon_close:
                        warn_string = warn_string + 'and latitude and longitude values do not match values '
                    else:
                        if not lat_close:
                            warn_string = warn_string + 'and latitude value does not match value '
                        else:
                            warn_string = warn_string + 'and longitude value does not match value '
                    if correct_lat_lon:
                        warn_string = (warn_string + 'for {telescope_name} in known '
                                       'telescopes. Using values from known '
                                       'telescopes.'.format(telescope_name=telescope_obj.telescope_name))
                        warnings.warn(warn_string)
                    else:
                        warn_string = (warn_string + 'for {telescope_name} in known '
                                       'telescopes. Using altitude value from known '
                                       'telescopes and lat/lon from '
                                       'file.'.format(telescope_name=telescope_obj.telescope_name))
                        warnings.warn(warn_string)
            else:
                warnings.warn('Altitude is not present in Miriad file, and ' +
                              'telescope {telescope_name} is not in ' +
                              'known_telescopes. Telescope location not '
                              'set.'.format(telescope_name=self.telescope_name))

        self.history = uv['history']
        if (self.pyuvdata_version_str.replace(' ', '') not in
                self.history.replace('\n', '').replace(' ', '')):
            self.history += self.pyuvdata_version_str
        self.channel_width *= 1e9  # change from GHz to Hz

        # check for pyuvdata variables that are not recognized miriad variables
        if 'visunits' in uv.vartable.keys():
            self.vis_units = uv['visunits'].replace('\x00', '')
        else:
            self.vis_units = 'UNCALIB'  # assume no calibration
        if 'instrume' in uv.vartable.keys():
            self.instrument = uv['instrume'].replace('\x00', '')
        else:
            self.instrument = self.telescope_name  # set instrument = telescope

        if 'dut1' in uv.vartable.keys():
            self.dut1 = uv['dut1']
        if 'degpdy' in uv.vartable.keys():
            self.earth_omega = uv['degpdy']
        if 'gst0' in uv.vartable.keys():
            self.gst0 = uv['gst0']
        if 'rdate' in uv.vartable.keys():
            self.rdate = uv['rdate'].replace('\x00', '')
        if 'timesys' in uv.vartable.keys():
            self.timesys = uv['timesys'].replace('\x00', '')

        # read through the file and get the data
        _source = uv['source']  # check source of initial visibility
        data_accumulator = {}
        pol_list = []
        for (uvw, t, (i, j)), d, f in uv.all(raw=True):
            # control for the case of only a single spw not showing up in
            # the dimension
            # Note that the (i, j) tuple is calculated from a baseline number in
            # aipy (see miriad_wrap.h). The i, j values are also adjusted by aipy
            # to start at 0 rather than 1.
            if len(d.shape) == 1:
                d.shape = (1,) + d.shape
                self.Nspws = d.shape[0]
                self.spw_array = np.arange(self.Nspws)
            else:
                raise(ValueError, """Sorry.  Files with more than one spectral
                      window (spw) are not yet supported. A great
                      project for the interested student!""")
            try:
                cnt = uv['cnt']
            except(KeyError):
                cnt = np.ones(d.shape, dtype=np.float)
            ra = uv['ra']
            dec = uv['dec']
            lst = uv['lst']
            source = uv['source']
            if source != _source:
                raise(ValueError, 'This appears to be a multi source file, which is not supported.')
            else:
                _source = source

            try:
                data_accumulator[uv['pol']].append([uvw, t, i, j, d, f, cnt,
                                                    ra, dec])
            except(KeyError):
                data_accumulator[uv['pol']] = [[uvw, t, i, j, d, f, cnt,
                                                ra, dec]]
                pol_list.append(uv['pol'])
                # NB: flag types in miriad are usually ints

        for pol, data in data_accumulator.iteritems():
            data_accumulator[pol] = np.array(data)

        self.polarization_array = np.array(pol_list)
        if len(self.polarization_array) != self.Npols:
            warnings.warn('npols={npols} but found {n} pols in data file'.format(
                npols=self.Npols, n=len(self.polarization_array)))

        # makes a data_array (and flag_array) of zeroes to be filled in by
        #   data values
        # any missing data will have zeros

        # use set to get the unique list of all times ever listed in the file
        # iterate over polarizations and all spectra (bls and times) in two
        # nested loops, then flatten into a single vector, then set
        # then list again.

        times = list(set(
            np.concatenate([[k[1] for k in d] for d in data_accumulator.values()])))
        times = np.sort(times)

        ant_i_unique = list(set(
            np.concatenate([[k[2] for k in d] for d in data_accumulator.values()])))
        ant_j_unique = list(set(
            np.concatenate([[k[3] for k in d] for d in data_accumulator.values()])))

        sorted_unique_ants = sorted(list(set(ant_i_unique + ant_j_unique)))
        ant_i_unique = np.array(ant_i_unique)
        ant_j_unique = np.array(ant_j_unique)

        # Determine maximum digits needed to distinguish different values
        ndig_ant = np.ceil(np.log10(sorted_unique_ants[-1])).astype(int) + 1
        # Be excessive in precision because we use the floating point values as dictionary keys later
        prec_t = - 2 * np.floor(np.log10(self._time_array.tols[-1])).astype(int)
        ndig_t = (np.ceil(np.log10(times[-1])).astype(int) + prec_t + 2)
        blts = []
        for d in data_accumulator.values():
            for k in d:
                blt = ["{1:.{0}f}".format(prec_t, k[1]).zfill(ndig_t),
                       str(k[2]).zfill(ndig_ant), str(k[3]).zfill(ndig_ant)]
                blt = "_".join(blt)
                blts.append(blt)
        unique_blts = np.unique(np.array(blts))

        reverse_inds = dict(zip(unique_blts, range(len(unique_blts))))
        self.Nants_data = len(sorted_unique_ants)

        # Miriad has no way to keep track of antenna numbers, so the antenna
        # numbers are simply the index for each antenna in any array that
        # describes antenna attributes (e.g. antpos for the antenna_postions).
        # Therefore on write, nants (which gives the size of the antpos array)
        # needs to be increased to be the max value of antenna_numbers+1 and the
        # antpos array needs to be inflated with zeros at locations where we
        # don't have antenna information. These inflations need to be undone at
        # read. If the file was written by pyuvdata, then the variable antnums
        # will be present and we can use it, otherwise we need to test for zeros
        # in the antpos array and/or antennas with no visibilities.
        try:
            # The antnums variable will only exist if the file was written with pyuvdata.
            # For some reason Miriad doesn't handle an array of integers properly,
            # so we convert to floats on write and back here
            self.antenna_numbers = uv['antnums'].astype(int)
            self.Nants_telescope = len(self.antenna_numbers)
        except(KeyError):
            self.antenna_numbers = None
            self.Nants_telescope = None

        nants = uv['nants']
        try:
            antpos = uv['antpos'].reshape(3, nants).T
            if self.Nants_telescope is not None:
                # in this case there is an antnums variable
                # (meaning that the file was written with pyuvdata), so we'll use it
                if nants == self.Nants_telescope:
                    # no inflation, so just use the positions
                    self.antenna_positions = antpos
                else:
                    # there is some inflation, just use the ones that appear in antnums
                    self.antenna_positions = np.zeros((self.Nants_telescope, 3), dtype=antpos.dtype)
                    for ai, num in enumerate(self.antenna_numbers):
                        self.antenna_positions[ai, :] = antpos[num, :]
            else:
                # there is no antnums variable (meaning that this file was not
                # written by pyuvdata), so we test for antennas with non-zero
                # positions and/or that appear in the visibility data
                # (meaning that they have entries in ant_1_array or ant_2_array)
                antpos_length = np.sqrt(np.sum(np.abs(antpos)**2, axis=1))
                good_antpos = np.where(antpos_length > 0)[0]
                # take the union of the antennas with good positions (good_antpos)
                # and the antennas that have visisbilities (sorted_unique_ants)
                # if there are antennas with visibilities but zeroed positions we issue a warning below
                ants_use = set(good_antpos).union(sorted_unique_ants)
                # ants_use are the antennas we'll keep track of in the UVData
                # object, so they dictate Nants_telescope
                self.Nants_telescope = len(ants_use)
                self.antenna_numbers = np.array(list(ants_use))
                self.antenna_positions = np.zeros((self.Nants_telescope, 3), dtype=antpos.dtype)
                for ai, num in enumerate(self.antenna_numbers):
                    self.antenna_positions[ai, :] = antpos[num, :]
                    if antpos_length[num] == 0:
                        warnings.warn('antenna number {n} has visibilities '
                                      'associated with it, but it has a position'
                                      ' of (0,0,0)'.format(n=num))
        except(KeyError):
            # there is no antpos variable
            self.antenna_positions = None

        if self.antenna_numbers is None:
            # there are no antenna_numbers or antenna_positions, so just use
            # the antennas present in the visibilities
            # (Nants_data will therefore match Nants_telescope)
            self.antenna_numbers = np.array(sorted_unique_ants)
            self.Nants_telescope = len(self.antenna_numbers)

        # antenna names is a foreign concept in miriad but required in other formats.
        try:
            # Here we deal with the way pyuvdata tacks it on to keep the name information if we have it:
            # This is a horrible hack to save & recover antenna_names array. Miriad can't handle arrays
            # of strings or a long enough single string to put them all into one string
            # so we convert them into hex values and then into floats on write and convert
            # back to strings here
            ant_name_flt = uv['antnames']
            ant_name_list = [('%x' % elem.astype(np.int64)).decode('hex') for elem in ant_name_flt]
            self.antenna_names = ant_name_list
        except(KeyError):
            self.antenna_names = self.antenna_numbers.astype(str).tolist()

        # form up a grid which indexes time and baselines along the 'long'
        # axis of the visdata array

        tij_grid = np.array(map(lambda x: map(float, x.split("_")), unique_blts))
        t_grid, ant_i_grid, ant_j_grid = tij_grid.T
        # set the data sizes
        try:
            self.Nblts = uv['nblts']
            if self.Nblts != len(t_grid):
                warnings.warn('Nblts does not match the number of unique blts in the data')
                self.Nblts = len(t_grid)
        except(KeyError):
            self.Nblts = len(t_grid)
        try:
            self.Ntimes = uv['ntimes']
            if self.Ntimes != len(times):
                warnings.warn('Ntimes does not match the number of unique times in the data')
                self.Ntimes = len(times)
        except(KeyError):
            self.Ntimes = len(times)

        self.time_array = t_grid
        self.ant_1_array = ant_i_grid.astype(int)
        self.ant_2_array = ant_j_grid.astype(int)

        self.baseline_array = self.antnums_to_baseline(ant_i_grid.astype(int),
                                                       ant_j_grid.astype(int))
        try:
            self.Nbls = uv['nbls']
            if self.Nbls != len(np.unique(self.baseline_array)):
                warnings.warn('Nbls does not match the number of unique baselines in the data')
                self.Nbls = len(np.unique(self.baseline_array))
        except(KeyError):
            self.Nbls = len(np.unique(self.baseline_array))

        # slot the data into a grid
        self.data_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs,
                                    self.Npols), dtype=np.complex64)
        self.flag_array = np.ones(self.data_array.shape, dtype=np.bool)
        self.uvw_array = np.zeros((self.Nblts, 3))
        # NOTE: Using our lst calculator, which uses astropy,
        # instead of aipy values which come from pyephem.
        # The differences are of order 5 seconds.
        if self.telescope_location is not None:
            self.set_lsts_from_time_array()
        self.nsample_array = np.ones(self.data_array.shape, dtype=np.float)
        self.freq_array = (np.arange(self.Nfreqs) * self.channel_width +
                           uv['sfreq'] * 1e9)
        # Tile freq_array to shape (Nspws, Nfreqs).
        # Currently does not actually support Nspws>1!
        self.freq_array = np.tile(self.freq_array, (self.Nspws, 1))

        # Temporary arrays to hold polarization axis, which will be collapsed
        ra_pol_list = np.zeros((self.Nblts, self.Npols))
        dec_pol_list = np.zeros((self.Nblts, self.Npols))
        uvw_pol_list = np.zeros((self.Nblts, 3, self.Npols))
        c_ns = const.c.to('m/ns').value
        for pol, data in data_accumulator.iteritems():
            pol_ind = self._pol_to_ind(pol)
            for ind, d in enumerate(data):
                blt = ["{1:.{0}f}".format(prec_t, d[1]).zfill(ndig_t),
                       str(d[2]).zfill(ndig_ant), str(d[3]).zfill(ndig_ant)]
                blt = "_".join(blt)
                blt_index = reverse_inds[blt]

                self.data_array[blt_index, :, :, pol_ind] = d[4]
                self.flag_array[blt_index, :, :, pol_ind] = d[5]
                self.nsample_array[blt_index, :, :, pol_ind] = d[6]

                # because there are uvws/ra/dec for each pol, and one pol may not
                # have that visibility, we collapse along the polarization
                # axis but avoid any missing visbilities
                uvw = d[0] * c_ns
                uvw.shape = (1, 3)
                uvw_pol_list[blt_index, :, pol_ind] = uvw
                ra_pol_list[blt_index, pol_ind] = d[7]
                dec_pol_list[blt_index, pol_ind] = d[8]

        # Collapse pol axis for ra_list, dec_list, and uvw_list
        ra_list = np.zeros(self.Nblts)
        dec_list = np.zeros(self.Nblts)
        for blt_index in xrange(self.Nblts):
            test = ~np.all(self.flag_array[blt_index, :, :, :], axis=(0, 1))
            good_pol = np.where(test)[0]
            if len(good_pol) == 1:
                # Only one good pol, use it
                self.uvw_array[blt_index, :] = uvw_pol_list[blt_index, :, good_pol]
                ra_list[blt_index] = ra_pol_list[blt_index, good_pol]
                dec_list[blt_index] = dec_pol_list[blt_index, good_pol]
            elif len(good_pol) > 1:
                # Multiple good pols, check for consistency. pyuvdata does not
                # support pol-dependent uvw, ra, or dec.
                if np.any(np.diff(uvw_pol_list[blt_index, :, good_pol], axis=0)):
                    raise ValueError('uvw values are different by polarization.')
                else:
                    self.uvw_array[blt_index, :] = uvw_pol_list[blt_index, :, good_pol[0]]
                if np.any(np.diff(ra_pol_list[blt_index, good_pol])):
                    raise ValueError('ra values are different by polarization.')
                else:
                    ra_list[blt_index] = ra_pol_list[blt_index, good_pol[0]]
                if np.any(np.diff(dec_pol_list[blt_index, good_pol])):
                    raise ValueError('dec values are different by polarization.')
                else:
                    dec_list[blt_index] = dec_pol_list[blt_index, good_pol[0]]
            else:
                # No good pols for this blt. Fill with first one.
                self.uvw_array[blt_index, :] = uvw_pol_list[blt_index, :, 0]
                ra_list[blt_index] = ra_pol_list[blt_index, 0]
                dec_list[blt_index] = dec_pol_list[blt_index, 0]

        # first check to see if the phase_type was specified.
        if phase_type is not None:
            if phase_type is 'phased':
                self.set_phased()
            elif phase_type is 'drift':
                self.set_drift()
            else:
                raise ValueError('The phase_type was not recognized. '
                                 'Set the phase_type to "drift" or "phased" to '
                                 'reflect the phasing status of the data')
        else:
            # check if ra is constant throughout file; if it is,
            # file is tracking if not, file is drift scanning
            if self.Ntimes > 1:
                blt_good = np.where(~np.all(self.flag_array, axis=(1, 2, 3)))
                if np.isclose(np.mean(np.diff(ra_list[blt_good])), 0.):
                    self.set_phased()
                else:
                    self.set_drift()
            else:
                # if there's only one time, checking for consistent RAs doesn't work.
                # instead check for the presence of an epoch variable, which isn't
                # really a good option, but at least it prevents crashes.
                if 'epoch' in uv.vartable.keys():
                    self.set_phased()
                else:
                    self.set_drift()

        if self.phase_type == 'phased':
            # check that the RA values do not vary
            blt_good = np.where(~np.all(self.flag_array, axis=(1, 2, 3)))
            if not np.isclose(np.mean(np.diff(ra_list[blt_good])), 0.):
                raise(ValueError, 'phase_type is "phased" but the RA values are varying.')
            self.phase_center_ra = ra_list[0]
            self.phase_center_dec = dec_list[0]
            self.phase_center_epoch = uv['epoch']
        else:
            # check that the RA values are not constant (if more than one time present)
            blt_good = np.where(~np.all(self.flag_array, axis=(1, 2, 3)))
            if np.isclose(np.mean(np.diff(ra_list[blt_good])), 0.) and self.Ntimes > 1:
                raise(ValueError, 'phase_type is "drift" but the RA values are constant.')
            self.zenith_ra = ra_list
            self.zenith_dec = dec_list

        try:
            self.set_telescope_params()
        except ValueError, ve:
            warnings.warn(str(ve))

        # check if object has all required uv_properties set
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

    def write_miriad(self, filepath, run_check=True, check_extra=True,
                     run_check_acceptability=True,
                     clobber=False, no_antnums=False):
        """
        Write the data to a miriad file.

        Args:
            filename: The miriad file directory to write to.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
            no_antnums: Option to not write the antnums variable to the file.
                Should only be used for testing purposes.
        """
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        # check for multiple spws
        if self.data_array.shape[1] > 1:
            raise ValueError('write_miriad currently only handles single spw files.')

        if os.path.exists(filepath):
            if clobber:
                print 'File exists: clobbering'
                shutil.rmtree(filepath)
            else:
                raise ValueError('File exists: skipping')

        if self.Nfreqs > 1:
            freq_spacing = self.freq_array[0, 1:] - self.freq_array[0, :-1]
            if not np.isclose(np.min(freq_spacing), np.max(freq_spacing),
                              rtol=self._freq_array.tols[0], atol=self._freq_array.tols[1]):
                raise ValueError('The frequencies are not evenly spaced (probably '
                                 'because of a select operation). The miriad format '
                                 'does not support unevenly spaced frequencies.')
            if not np.isclose(np.max(freq_spacing), self.channel_width,
                              rtol=self._freq_array.tols[0], atol=self._freq_array.tols[1]):
                raise ValueError('The frequencies are separated by more than their '
                                 'channel width (probably because of a select operation). '
                                 'The miriad format does not support frequencies '
                                 'that are spaced by more than their channel width.')

        uv = a.miriad.UV(filepath, status='new')

        # initialize header variables
        uv._wrhd('obstype', 'mixed-auto-cross')
        # avoid inserting extra \n.
        if not self.history[-1] == '\n':
            self.history += '\n'
        uv._wrhd('history', self.history)

        # recognized miriad variables
        uv.add_var('nchan', 'i')
        uv['nchan'] = self.Nfreqs
        uv.add_var('npol', 'i')
        uv['npol'] = self.Npols
        uv.add_var('nspect', 'i')
        uv['nspect'] = self.Nspws
        uv.add_var('inttime', 'd')
        uv['inttime'] = self.integration_time
        uv.add_var('sdf', 'd')
        uv['sdf'] = self.channel_width / 1e9  # in GHz
        uv.add_var('source', 'a')
        uv['source'] = self.object_name
        uv.add_var('telescop', 'a')
        uv['telescop'] = self.telescope_name
        uv.add_var('latitud', 'd')
        uv['latitud'] = self.telescope_location_lat_lon_alt[0]
        uv.add_var('longitu', 'd')
        uv['longitu'] = self.telescope_location_lat_lon_alt[1]
        uv.add_var('nants', 'i')

        # Miriad has no way to keep track of antenna numbers, so the antenna
        # numbers are simply the index for each antenna in any array that
        # describes antenna attributes (e.g. antpos for the antenna_postions).
        # Therefore on write, nants (which gives the size of the antpos array)
        # needs to be increased to be the max value of antenna_numbers+1 and the
        # antpos array needs to be inflated with zeros at locations where we
        # don't have antenna information. These inflations need to be undone at
        # read. If the file was written by pyuvdata, then the variable antnums
        # will be present and we can use it, otherwise we need to test for zeros
        # in the antpos array and/or antennas with no visibilities.
        nants = np.max(self.antenna_numbers) + 1
        uv['nants'] = nants
        if self.antenna_positions is not None:
            antpos = np.zeros((nants, 3), dtype=self.antenna_positions.dtype)
            for ai, num in enumerate(self.antenna_numbers):
                antpos[num, :] = self.antenna_positions[ai, :]
            uv.add_var('antpos', 'd')
            uv['antpos'] = antpos.T.flatten()

        uv.add_var('sfreq', 'd')
        uv['sfreq'] = self.freq_array[0, 0] / 1e9  # first spw; in GHz
        if self.phase_type == 'phased':
            uv.add_var('epoch', 'r')
            uv['epoch'] = self.phase_center_epoch

        # required pyuvdata variables that are not recognized miriad variables
        uv.add_var('ntimes', 'i')
        uv['ntimes'] = self.Ntimes
        uv.add_var('nbls', 'i')
        uv['nbls'] = self.Nbls
        uv.add_var('nblts', 'i')
        uv['nblts'] = self.Nblts
        uv.add_var('visunits', 'a')
        uv['visunits'] = self.vis_units
        uv.add_var('instrume', 'a')
        uv['instrume'] = self.instrument
        uv.add_var('altitude', 'd')
        uv['altitude'] = self.telescope_location_lat_lon_alt[2]

        # optional pyuvdata variables that are not recognized miriad variables
        if self.dut1 is not None:
            uv.add_var('dut1', 'd')
            uv['dut1'] = self.dut1
        if self.earth_omega is not None:
            uv.add_var('degpdy', 'd')
            uv['degpdy'] = self.earth_omega
        if self.gst0 is not None:
            uv.add_var('gst0', 'd')
            uv['gst0'] = self.gst0
        if self.rdate is not None:
            uv.add_var('rdate', 'a')
            uv['rdate'] = self.rdate
        if self.timesys is not None:
            uv.add_var('timesys', 'a')
            uv['timesys'] = self.timesys

        if not no_antnums:
            # Add in the antenna_numbers so we have them if we read this file back in.
            # For some reason Miriad doesn't handle an array of integers properly,
            # so convert to floats here and integers on read.
            uv.add_var('antnums', 'd')
            uv['antnums'] = self.antenna_numbers.astype(np.float64)

        # antenna names is a foreign concept in miriad but required in other formats.
        # This is a horrible hack to save antenna_names array. Miriad can't handle arrays
        # of strings or a long enough single string to put them all into one string
        # so we convert them into hex values and then into floats and convert
        # back to strings on read
        ant_name_flt = np.array([int(elem.encode("hex"), 16) for elem in self.antenna_names]).astype(np.float64)
        uv.add_var('antnames', 'd')
        uv['antnames'] = ant_name_flt

        # variables that can get updated with every visibility
        uv.add_var('pol', 'i')
        uv.add_var('lst', 'd')
        uv.add_var('cnt', 'd')
        uv.add_var('ra', 'd')
        uv.add_var('dec', 'd')

        # write data
        c_ns = const.c.to('m/ns').value
        for viscnt, blt in enumerate(self.data_array):
            uvw = (self.uvw_array[viscnt] / c_ns).astype(np.double)  # NOTE issue 50 on conjugation
            t = self.time_array[viscnt]
            i = self.ant_1_array[viscnt]
            j = self.ant_2_array[viscnt]

            uv['lst'] = self.lst_array[viscnt]
            if self.phase_type == 'phased':
                uv['ra'] = self.phase_center_ra
                uv['dec'] = self.phase_center_dec
            elif self.phase_type == 'drift':
                uv['ra'] = self.zenith_ra[viscnt]
                uv['dec'] = self.zenith_dec[viscnt]
            else:
                raise ValueError('The phasing type of the data is unknown. '
                                 'Set the phase_type to "drift" or "phased" to '
                                 'reflect the phasing status of the data')

            # NOTE only writing spw 0, not supporting multiple spws for write
            for polcnt, pol in enumerate(self.polarization_array):
                uv['pol'] = pol.astype(np.int)
                uv['cnt'] = self.nsample_array[viscnt, 0, :, polcnt].astype(np.double)

                data = self.data_array[viscnt, 0, :, polcnt]
                flags = self.flag_array[viscnt, 0, :, polcnt]
                if i > j:
                    i, j, data = j, i, np.conjugate(data)
                preamble = (uvw, t, (i, j))

                uv.write(preamble, data, flags)
