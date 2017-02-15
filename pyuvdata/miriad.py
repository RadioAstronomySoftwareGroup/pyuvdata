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
    Defines a Miriad-speficic subclass of UVData for reading and writing Miriad files.
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

    def read_miriad(self, filepath, correct_lat_lon=True, run_check=True, run_check_acceptability=True):
        """
        Read in data from a miriad file.

        Args:
            filepath: The miriad file directory to read from.
            correct_lat_lon: flag -- that only matters if altitude is missing --
                to update the latitude and longitude from the known_telescopes list
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after reading in the file. Default is True.
        """
        if not os.path.exists(filepath):
            raise(IOError, filepath + ' not found')
        uv = a.miriad.UV(filepath)

        miriad_header_data = {'Nfreqs': 'nchan',
                              'Npols': 'npol',
                              'integration_time': 'inttime',
                              'channel_width': 'sdf',  # in Ghz!
                              'object_name': 'source',
                              # NB: telescope_name and instrument are treated
                              # as the same
                              'telescope_name': 'telescop',
                              'instrument': 'telescop',
                              'Nants_telescope': 'nants',
                              'antenna_positions': 'antpos',  # take deltas
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
                # attribute_list = [a for a in dir(telescope_obj) if not a.startswith('__') and
                #                   not callable(getattr(telescope_obj, a))]

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
        try:
            self.antenna_positions = \
                self.antenna_positions.reshape(3, self.Nants_telescope).T
        except(ValueError):  # workaround for known errors with bad antenna positions
            self.antenna_positions = None
        self.channel_width *= 1e9  # change from GHz to Hz

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
        self.polarization_array = np.array(pol_list)
        if len(self.polarization_array) > self.Npols:
            print "WARNING: npols={npols} but found {l} pols in data file".format(
                npols=self.Npols, l=len(self.polarization_array))

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

        unique_blts = []
        for d in data_accumulator.values():
            for k in d:
                blt = [k[1], k[2], k[3]]
                if blt not in unique_blts:
                    unique_blts.append(blt)
        self.Nants_data = len(sorted_unique_ants)
        try:
            # for some reason Miriad doesn't handle an array of integers properly,
            # so convert to floats on write and back here
            self.antenna_numbers = uv['antnums'].astype(int)
        except(KeyError):
            if np.max(sorted_unique_ants) < self.Nants_telescope:
                self.antenna_numbers = np.arange(self.Nants_telescope)
            else:
                raise ValueError('Max antenna number is larger than supported '
                                 'by the antenna position array and antnums is '
                                 'not present to convert numbers to indices')
        try:
            # horrible hack to save & recover antenna_names array. Miriad can't handle arrays
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
        t_grid = []
        ant_i_grid = []
        ant_j_grid = []
        for t in times:
            for ant_i in ant_i_unique:
                for ant_j in ant_j_unique:
                    if ant_i > ant_j:
                        continue
                    if [t, ant_i, ant_j] not in unique_blts:
                        continue
                    t_grid.append(t)
                    ant_i_grid.append(ant_i)
                    ant_j_grid.append(ant_j)
        ant_i_grid = np.array(ant_i_grid)
        ant_j_grid = np.array(ant_j_grid)
        t_grid = np.array(t_grid)

        # set the data sizes
        try:
            self.Nblts = uv['nblts']
            if self.Nblts != len(t_grid):
                raise ValueError('Nblts does not match the number of unique blts in the data')
        except(KeyError):
            self.Nblts = len(t_grid)
        try:
            self.Ntimes = uv['ntimes']
            if self.Ntimes != len(times):
                raise ValueError('Ntimes does not match the number of unique times in the data')
        except(KeyError):
            self.Ntimes = len(times)
        self.time_array = t_grid
        self.ant_1_array = ant_i_grid
        self.ant_2_array = ant_j_grid

        self.baseline_array = self.antnums_to_baseline(ant_i_grid,
                                                       ant_j_grid)
        try:
            self.Nbls = uv['nbls']
            if self.Nbls != len(np.unique(self.baseline_array)):
                raise ValueError('Nbls does not match the number of unique baselines in the data')
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

        for pol, data in data_accumulator.iteritems():
            pol_ind = self._pol_to_ind(pol)
            for ind, d in enumerate(data):
                t, ant_i, ant_j = d[1], d[2], d[3]
                blt_index = np.where(np.logical_and(np.logical_and(t == t_grid,
                                                                   ant_i == ant_i_grid),
                                                    ant_j == ant_j_grid))[0].squeeze()
                self.data_array[blt_index, :, :, pol_ind] = d[4]
                self.flag_array[blt_index, :, :, pol_ind] = d[5]
                self.nsample_array[blt_index, :, :, pol_ind] = d[6]

                # because there are uvws/ra/dec for each pol, and one pol may not
                # have that visibility, we collapse along the polarization
                # axis but avoid any missing visbilities
                uvw = d[0] * const.c.to('m/ns').value
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

        # check if ra is constant throughout file; if it is,
        # file is tracking if not, file is drift scanning
        blt_good = np.where(~np.all(self.flag_array, axis=(1, 2, 3)))
        if np.isclose(np.mean(np.diff(ra_list[blt_good])), 0.):
            self.set_phased()
            self.phase_center_ra = ra_list[0]
            self.phase_center_dec = dec_list[0]
            self.phase_center_epoch = uv['epoch']
        else:
            self.set_drift()
            self.zenith_ra = ra_list
            self.zenith_dec = dec_list

        self.vis_units = 'UNCALIB'  # assume no calibration

        try:
            self.set_telescope_params()
        except ValueError, ve:
            warnings.warn(str(ve))

        # check if object has all required uv_properties set
        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

    def write_miriad(self, filepath, run_check=True, run_check_acceptability=True, clobber=False):
        """
        Write the data to a miriad file.

        Args:
            filename: The miriad file directory to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
        """
        # check for multiple spws
        if self.data_array.shape[1] > 1:
            raise ValueError('write_miriad currently only handles single spw files.')

        if os.path.exists(filepath):
            if clobber:
                print 'File exists: clobbering'
                shutil.rmtree(filepath)
            else:
                raise ValueError('File exists: skipping')

        uv = a.miriad.UV(filepath, status='new')

        # initialize header variables
        uv._wrhd('obstype', 'mixed-auto-cross')
        uv._wrhd('history', self.history + '\n')

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
        uv['nants'] = self.Nants_telescope
        try:
            uv.add_var('antpos', 'd')
            uv['antpos'] = self.antenna_positions.T.flatten()
        except(AttributeError):
            pass  # don't write bad antenna positions
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

        # for some reason Miriad doesn't handle an array of integers properly,
        # so convert to floats here and integers on read
        uv.add_var('antnums', 'd')
        uv['antnums'] = self.antenna_numbers.astype(np.float64)

        # horrible hack to save antenna_names array. Miriad can't handle arrays
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
        for viscnt, blt in enumerate(self.data_array):
            uvw = (self.uvw_array[viscnt] /
                   const.c.to('m/ns').value).astype(np.double)  # NOTE issue 50 on conjugation
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
        if run_check:
            """Check for acceptable units."""
            self.check(run_check_acceptability=run_check_acceptability)
