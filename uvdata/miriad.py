from astropy import constants as const
import os
import numpy as np
import warnings
import aipy as a
import uvdata


class Miriad(uvdata.uv.UVData):

    def miriad_pol_to_ind(self, pol):
        if self.polarization_array.value is None:
            raise(ValueError, "Can't index polarization {p} because "
                  "polarization_array is not set".format(p=pol))
        pol_ind = np.argwhere(self.polarization_array.value == pol)
        if len(pol_ind) != 1:
            raise(ValueError, "multiple matches for pol={pol} in "
                  "polarization_array".format(pol=pol))
        return pol_ind

    def read_miriad(self, filepath, FLEXIBLE_OPTION=True):
        # map uvdata attributes to miriad data values
        # those which we can get directly from the miriad file
        # (some, like n_times, have to be calculated)
        if not os.path.exists(filepath):
            raise(IOError, filepath + ' not found')
        uv = a.miriad.UV(filepath)

        miriad_header_data = {'Nfreqs': 'nchan',
                              'Npols': 'npol',
                              # 'Nspws': 'nspec',  # not always available
                              'integration_time': 'inttime',
                              'channel_width': 'sdf',  # in Ghz!
                              'object_name': 'source',
                              'telescope_name': 'telescop',
                              # same as telescope_name for now
                              'instrument': 'telescop',
                              'latitude': 'latitud',
                              'longitude': 'longitu',  # in units of radians
                              # (get the first time in the ever changing header)
                              'dateobs': 'time',
                              # 'history': 'history',
                              'Nants_telescope': 'nants',
                              'phase_center_epoch': 'epoch',
                              'antenna_positions': 'antpos',  # take deltas
                              }
        for item in miriad_header_data:
            if isinstance(uv[miriad_header_data[item]], str):
                header_value = uv[miriad_header_data[item]].replace('\x00', '')
            else:
                header_value = uv[miriad_header_data[item]]
            getattr(self, item).value = header_value

        if self.telescope_name.value.startswith('PAPER') and \
                self.altitude.value is None:
            print "WARNING: Altitude not found for telescope PAPER. "
            print "setting to 1100m"
            self.altitude.value = 1100.

        self.history.value = uv['history']
        try:
            self.antenna_positions.value = \
                self.antenna_positions.value.reshape(3, self.Nants_telescope.value).T
        except(ValueError):
            self.antenna_positions.value = None
        self.channel_width.value *= 1e9  # change from GHz to Hz

        # read through the file and get the data
        data_accumulator = {}
        for (uvw, t, (i, j)), d, f in uv.all(raw=True):
            # control for the case of only a single spw not showing up in
            # the dimension
            if len(d.shape) == 1:
                d.shape = (1,) + d.shape
                self.Nspws.value = d.shape[0]
                self.spw_array.value = np.arange(self.Nspws.value)
            else:
                raise(ValueError, """Sorry.  Files with more than one spectral
                      window (spw) are not yet supported. A great
                      project for the interested student!""")
            try:
                cnt = uv['cnt']
            except(KeyError):
                cnt = np.ones(d.shape, dtype=np.int)
            zenith_ra = uv['ra']
            zenith_dec = uv['dec']
            lst = uv['lst']

            try:
                data_accumulator[uv['pol']].append([uvw, t, i, j, d, f, cnt,
                                                    zenith_ra, zenith_dec])
            except(KeyError):
                data_accumulator[uv['pol']] = [[uvw, t, i, j, d, f, cnt,
                                                zenith_ra, zenith_dec]]
                # NB: flag types in miriad are usually ints
        self.polarization_array.value = np.sort(data_accumulator.keys())
        if len(self.polarization_array.value) > self.Npols.value:
            print "WARNING: npols={npols} but found {l} pols in data file".format(
                npols=self.Npols.value, l=len(self.polarization_array.value))
        if FLEXIBLE_OPTION:
            # makes a data_array (and flag_array) of zeroes to be filled in by
            #   data values
            # any missing data will have zeros

            # use set to get the unique list of all times ever listed in the file
            # iterate over polarizations and all spectra (bls and times) in two
            # nested loops, then flatten into a single vector, then set
            # then list again.
            times = list(set(
                np.ravel([[k[1] for k in d] for d in data_accumulator.values()])))
            times = np.sort(times)

            ant_i_unique = list(set(
                np.ravel([[k[2] for k in d] for d in data_accumulator.values()])))
            ant_j_unique = list(set(
                np.ravel([[k[3] for k in d] for d in data_accumulator.values()])))

            self.Nants_data.value = max(len(ant_i_unique), len(ant_j_unique))
            self.antenna_indices.value = np.arange(self.Nants_telescope.value)
            self.antenna_names.value = self.antenna_indices.value.astype(str).tolist()
            # form up a grid which indexes time and baselines along the 'long'
            # axis of the visdata array
            t_grid = []
            ant_i_grid = []
            ant_j_grid = []
            for t in times:
                for ant_i in ant_i_unique:
                    for ant_j in ant_j_unique:
                        t_grid.append(t)
                        ant_i_grid.append(ant_i)
                        ant_j_grid.append(ant_j)
            ant_i_grid = np.array(ant_i_grid)
            ant_j_grid = np.array(ant_j_grid)
            t_grid = np.array(t_grid)

            # set the data sizes
            self.Nblts.value = len(t_grid)
            self.Ntimes.value = len(times)
            self.time_array.value = t_grid
            self.ant_1_array.value = ant_i_grid
            self.ant_2_array.value = ant_j_grid

            self.baseline_array.value = self.antnums_to_baseline(ant_i_grid,
                                                                 ant_j_grid)
            self.Nbls.value = len(np.unique(self.baseline_array.value))
            # slot the data into a grid
            self.data_array.value = np.zeros((self.Nblts.value,
                                              self.Nspws.value,
                                              self.Nfreqs.value,
                                              self.Npols.value),
                                             dtype=np.complex64)
            self.flag_array.value = np.ones(self.data_array.value.shape, dtype=np.bool)
            self.uvw_array.value = np.zeros((3, self.Nblts.value))
            # NOTE: Using our lst calculator, which uses astropy,
            # instead of aipy values which come from pyephem.
            # The differences are of order 5 seconds.
            self.set_lsts_from_time_array()
            self.nsample_array.value = np.ones(self.data_array.value.shape,
                                               dtype=np.int)
            self.freq_array.value = (np.arange(self.Nfreqs.value) *
                                     self.channel_width.value +
                                     uv['sfreq'] * 1e9)
            # Tile freq_array to dimensions (Nspws, Nfreqs).
            # Currently does not actually support Nspws>1!
            self.freq_array.value = np.tile(self.freq_array.value,
                                            (self.Nspws.value, 1))

            ra_list = np.zeros(self.Nblts.value)
            dec_list = np.zeros(self.Nblts.value)

            for pol, data in data_accumulator.iteritems():
                pol_ind = self.miriad_pol_to_ind(pol)
                for ind, d in enumerate(data):
                    t, ant_i, ant_j = d[1], d[2], d[3]
                    blt_index = np.where(np.logical_and(np.logical_and(t == t_grid,
                                                                       ant_i == ant_i_grid),
                                                        ant_j == ant_j_grid))[0].squeeze()
                    self.data_array.value[blt_index, :, :, pol_ind] = d[4]
                    self.flag_array.value[blt_index, :, :, pol_ind] = d[5]
                    self.nsample_array.value[blt_index, :, :, pol_ind] = d[6]

                    # because there are uvws/ra/dec for each pol, and one pol may not
                    # have that visibility, we collapse along the polarization
                    # axis but avoid any missing visbilities
                    uvw = d[0] * const.c.to('m/ns').value
                    uvw.shape = (1, 3)
                    self.uvw_array.value[:, blt_index] = uvw
                    ra_list[blt_index] = d[7]
                    dec_list[blt_index] = d[8]

            # check if ra is constant throughout file; if it is,
            # file is tracking if not, file is drift scanning
            if np.isclose(np.mean(np.diff(ra_list)), 0.):
                self.phase_center_ra.value = ra_list[0]
                self.phase_center_dec.value = dec_list[0]
            else:
                self.zenith_ra.value = ra_list
                self.zenith_dec.value = dec_list

            # enforce drift scan/ phased convention
            # convert lat/lon to x/y/z_telescope
            #    LLA to ECEF (see pdf in docs)

        if not FLEXIBLE_OPTION:
            pass
            # this option would accumulate things requiring
            # baselines and times are sorted in the same
            #          order for each polarization
            # and that there are the same number of baselines
            #          and pols per timestep
            # TBD impliment

        # NOTES:
        # pyuvdata is natively 0 indexed as is miriad
        # miriad uses the same pol2num standard as aips/casa

        self.vis_units.value = 'UNCALIB'  # assume no calibration

        # things that might not be required?
        # 'GST0'  : None,
        # 'RDate'  : None,  # date for which the GST0 or whatever... applies
        # 'earth_omega'  : 360.985,
        # 'DUT1'  : 0.0,        # DUT1 (google it) AIPS 117 calls it UT1UTC
        # 'TIMESYS'  : 'UTC',   # We only support UTC

        #

        # Phasing rule: if alt/az is set and ra/dec are None,
        #  then its a drift scan

        # check if object has all required uv_properties set
        self.check()
        return True
