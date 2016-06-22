from astropy import constants as const
import os
import numpy as np
import warnings
import aipy as a
import uvdata


class Miriad(uvdata.uv.UVData):

    def miriad_pol_to_ind(self, pol):
        if self.polarization_array is None:
            raise(ValueError, "Can't index polarization {p} because "
                  "polarization_array is not set".format(p=pol))
        pol_ind = np.argwhere(self.polarization_array == pol)
        if len(pol_ind) != 1:
            raise(ValueError, "multiple matches for pol={pol} in "
                  "polarization_array".format(pol=pol))
        return pol_ind

    def read_miriad(self, filepath, FLEXIBLE_OPTION=True, run_check=True, run_sanity_check=True):
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
            setattr(self, item, header_value)

        if self.telescope_name.startswith('PAPER') and \
                self.altitude is None:
            print "WARNING: Altitude not found for telescope PAPER. "
            print "setting to 1100m"
            self.altitude = 1100.

        self.history = uv['history']
        try:
            self.antenna_positions = \
                self.antenna_positions.reshape(3, self.Nants_telescope).T
        except(ValueError):
            self.antenna_positions = None
        self.channel_width *= 1e9  # change from GHz to Hz

        # read through the file and get the data
        _source = uv['source']  # check source of initial visibility
        data_accumulator = {}
        for (uvw, t, (i, j)), d, f in uv.all(raw=True):
            # control for the case of only a single spw not showing up in
            # the dimension
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
                cnt = np.ones(d.shape, dtype=np.int)
            zenith_ra = uv['ra']
            zenith_dec = uv['dec']
            lst = uv['lst']
            source = uv['source']
            if source != _source:
                raise(ValueError, 'This appears to be a multi source file, which is not supported.')
            else:
                _source = source

            try:
                data_accumulator[uv['pol']].append([uvw, t, i, j, d, f, cnt,
                                                    zenith_ra, zenith_dec])
            except(KeyError):
                data_accumulator[uv['pol']] = [[uvw, t, i, j, d, f, cnt,
                                                zenith_ra, zenith_dec]]
                # NB: flag types in miriad are usually ints
        self.polarization_array = np.sort(data_accumulator.keys())
        if len(self.polarization_array) > self.Npols:
            print "WARNING: npols={npols} but found {l} pols in data file".format(
                npols=self.Npols, l=len(self.polarization_array))
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

            self.Nants_data = max(len(ant_i_unique), len(ant_j_unique))
            self.antenna_indices = np.arange(self.Nants_telescope)
            self.antenna_names = self.antenna_indices.astype(str).tolist()
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
            self.Nblts = len(t_grid)
            self.Ntimes = len(times)
            self.time_array = t_grid
            self.ant_1_array = ant_i_grid
            self.ant_2_array = ant_j_grid

            self.baseline_array = self.antnums_to_baseline(ant_i_grid,
                                                           ant_j_grid)
            self.Nbls = len(np.unique(self.baseline_array))
            # slot the data into a grid
            self.data_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs,
                                        self.Npols), dtype=np.complex64)
            self.flag_array = np.ones(self.data_array.shape, dtype=np.bool)
            self.uvw_array = np.zeros((3, self.Nblts))
            # NOTE: Using our lst calculator, which uses astropy,
            # instead of aipy values which come from pyephem.
            # The differences are of order 5 seconds.
            self.set_lsts_from_time_array()
            self.nsample_array = np.ones(self.data_array.shape, dtype=np.int)
            self.freq_array = (np.arange(self.Nfreqs) * self.channel_width +
                               uv['sfreq'] * 1e9)
            # Tile freq_array to dimensions (Nspws, Nfreqs).
            # Currently does not actually support Nspws>1!
            self.freq_array = np.tile(self.freq_array, (self.Nspws, 1))

            ra_list = np.zeros(self.Nblts)
            dec_list = np.zeros(self.Nblts)

            for pol, data in data_accumulator.iteritems():
                pol_ind = self.miriad_pol_to_ind(pol)
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
                    self.uvw_array[:, blt_index] = uvw
                    ra_list[blt_index] = d[7]
                    dec_list[blt_index] = d[8]

            # check if ra is constant throughout file; if it is,
            # file is tracking if not, file is drift scanning
            if np.isclose(np.mean(np.diff(ra_list)), 0.):
                self.phase_center_ra = ra_list[0]
                self.phase_center_dec = dec_list[0]
            else:
                self.zenith_ra = ra_list
                self.zenith_dec = dec_list

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

        self.vis_units = 'UNCALIB'  # assume no calibration

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
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        return True

    def write_miriad(self, filepath, run_check=True, run_sanity_check=True):
        #check for multiple spws
        if self.data_array.shape[1] > 1: 
            raise ValueError('write_miriad currently only handles single spw files.')

        uv = a.miriad.UV(filepath,status='new')

        #initialize header variables
        uv._wrhd('obstype','mixed-auto-cross')
        uv._wrhd('history', self.history + '\n')

        #required miriad variables
        uv.add_var('nchan', 'i'); uv['nchan'] = self.Nfreqs
        uv.add_var('npol', 'i'); uv['npol'] = self.Npols
        uv.add_var('nspect', 'i'); uv['nspect'] = self.Nspws

        #required pyuvdata variables that are not recognized miriad variables
        uv.add_var('ntimes', 'i'); uv['ntimes'] = self.Ntimes
        uv.add_var('nbls', 'i'); uv['nbls'] = self.Nbls
        uv.add_var('nblts', 'i'); uv['nblts'] = self.Nblts
        uv.add_var('visunits', 'a'); uv['visunits'] = self.vis_units

        #variables that can get updated with every visibility
        uv.add_var('pol', 'i')
        uv.add_var('lst', 'd')
        uv.add_var('cnt', 'i')
 
        #write data
        for polcnt, pol in enumerate(self.polarization_array):
            uv['pol'] = pol
            for viscnt, blt in enumerate(self.data_array):
                uvw = self.uvw_array[:,viscnt] #Note issue 50 on conjugation
                t = self.time_array[viscnt]
                i = self.ant_1_array[viscnt]
                j = self.ant_2_array[viscnt]
                preamble = (uvw,t,(i,j))
   
                uv['lst'] = self.lst_array[viscnt]
 
                #NOTE only writing spw 0, not supporting multiple spws for write
                uv['cnt'] = self.nsample_array[viscnt,0,:,polcnt]                

                data = self.data_array[viscnt,0,:,polcnt]
                flags = self.flag_array[viscnt,0,:,polcnt]
                
                uv.write(preamble, data, flags)
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        return True
