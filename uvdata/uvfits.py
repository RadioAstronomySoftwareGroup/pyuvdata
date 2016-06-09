from astropy import constants as const
from astropy.time import Time
from astropy.io import fits
import numpy as np
import warnings
import uvdata


class UVFITS(uvdata.uv.UVData):

    uvfits_required_extra = ['xyz_telescope_frame', 'x_telescope',
                             'y_telescope', 'z_telescope',
                             'antenna_positions', 'GST0', 'RDate',
                             'earth_omega', 'DUT1', 'TIMESYS']

    def _gethduaxis(self, D, axis):
        ax = str(axis)
        N = D.header['NAXIS' + ax]
        X0 = D.header['CRVAL' + ax]
        dX = D.header['CDELT' + ax]
        Xi0 = D.header['CRPIX' + ax] - 1
        return np.arange(X0 - dX * Xi0, X0 - dX * Xi0 + N * dX, dX)

    def _indexhdus(self, hdulist):
        # input a list of hdus
        # return a dictionary of table names
        tablenames = {}
        for i in range(len(hdulist)):
            try:
                tablenames[hdulist[i].header['EXTNAME']] = i
            except(KeyError):
                continue
        return tablenames

    def read_uvfits(self, filename):

        F = fits.open(filename)
        D = F[0]  # assumes the visibilities are in the primary hdu
        hdunames = self._indexhdus(F)  # find the rest of the tables

        # astropy.io fits reader scales date according to relevant PZER0 (?)
        time0_array = D.data['DATE']
        try:
            # uvfits standard is to have 2 DATE parameters, both floats:
            # DATE (full day) and _DATE (fractional day)
            time1_array = D.data['_DATE']
            self.time_array.value = (time0_array.astype(np.double) +
                                     time1_array.astype(np.double))
        except(KeyError):
            # cotter uvfits files have one DATE that is a double
            self.time_array.value = time0_array
            if np.finfo(time0_array[0]).precision < 5:
                raise ValueError('JDs in this file are not precise to '
                                 'better than a second.')
            if (np.finfo(time0_array[0]).precision > 5 and
                    np.finfo(time0_array[0]).precision < 8):
                warnings.warn('The JDs in this file have sub-second '
                              'precision, but not sub-millisecond. '
                              'Use with caution.')

        self.Ntimes.value = len(np.unique(self.time_array.value))

        # if antenna arrays are present, use them. otherwise use baseline array
        try:
            # Note: uvfits antennas are 1 indexed,
            # need to subtract one to get to 0-indexed
            self.ant_1_array.value = np.int32(D.data.field('ANTENNA1')) - 1
            self.ant_2_array.value = np.int32(D.data.field('ANTENNA2')) - 1
        except:
            # cannot set this to be the baseline array because it uses the
            # 256 convention, not our 2048 convention
            bl_input_array = np.int64(D.data.field('BASELINE'))

            # get antenna arrays based on uvfits baseline array
            self.ant_1_array.value, self.ant_2_array.value = \
                self.baseline_to_antnums(bl_input_array)

        # get self.baseline_array using our convention
        self.baseline_array.value = \
            self.antnums_to_baseline(self.ant_1_array.value,
                                     self.ant_2_array.value)
        self.Nbls.value = len(np.unique(self.baseline_array.value))

        # initialize internal variables based on the antenna table
        self.Nants_data.value = int(np.max([len(np.unique(self.ant_1_array.value)),
                                            len(np.unique(self.ant_2_array.value))]))

        # check if we have an spw dimension
        if D.header['NAXIS'] == 7:
            if D.header['NAXIS5'] > 1:
                raise ValueError('Sorry.  Files with more than one spectral' +
                                 'window (spw) are not yet supported. A ' +
                                 'great project for the interested student!')
            self.Nspws.value = D.header['NAXIS5']
            self.data_array.value = (D.data.field('DATA')
                                     [:, 0, 0, :, :, :, 0] +
                                     1j * D.data.field('DATA')
                                     [:, 0, 0, :, :, :, 1])
            self.flag_array.value = (D.data.field('DATA')
                                     [:, 0, 0, :, :, :, 2] <= 0)
            self.nsample_array.value = np.abs(D.data.field('DATA')
                                              [:, 0, 0, :, :, :, 2])
            self.Nspws.value = D.header['NAXIS5']
            assert(self.Nspws.value == self.data_array.value.shape[1])

            # the axis number for phase center depends on if the spw exists
            # subtract 1 to be zero-indexed
            self.spw_array.value = np.int32(self._gethduaxis(D, 5)) - 1

            self.phase_center_ra.set_degrees(np.array(D.header['CRVAL6']).astype(np.float64))
            self.phase_center_dec.set_degrees(np.array(D.header['CRVAL7']).astype(np.float64))
        else:
            # in many uvfits files the spw axis is left out,
            # here we put it back in so the dimensionality stays the same
            self.data_array.value = (D.data.field('DATA')[:, 0, 0, :, :, 0] +
                                     1j * D.data.field('DATA')
                                     [:, 0, 0, :, :, 1])
            self.data_array.value = self.data_array.value[:, np.newaxis, :, :]
            self.flag_array.value = (D.data.field('DATA')
                                     [:, 0, 0, :, :, 2] <= 0)
            self.flag_array.value = self.flag_array.value[:, np.newaxis, :, :]
            self.nsample_array.value = np.abs(D.data.field('DATA')
                                              [:, 0, 0, :, :, 2])
            self.nsample_array.value = (self.nsample_array.value
                                        [:, np.newaxis, :, :])

            # the axis number for phase center depends on if the spw exists
            self.Nspws.value = 1
            self.spw_array.value = np.array([0])

            self.phase_center_ra.set_degrees(np.array(D.header['CRVAL5']).astype(np.float64))
            self.phase_center_dec.set_degrees(np.array(D.header['CRVAL6']).astype(np.float64))

        # get dimension sizes
        self.Nfreqs.value = D.header['NAXIS4']
        assert(self.Nfreqs.value == self.data_array.value.shape[2])
        self.Npols.value = D.header['NAXIS3']
        assert(self.Npols.value == self.data_array.value.shape[3])
        self.Nblts.value = D.header['GCOUNT']
        assert(self.Nblts.value == self.data_array.value.shape[0])

        # read baseline vectors in units of seconds, return in meters
        self.uvw_array.value = (np.array(np.stack((D.data.field('UU'),
                                                   D.data.field('VV'),
                                                   D.data.field('WW')))) *
                                const.c.to('m/s').value)

        self.freq_array.value = self._gethduaxis(D, 4)
        self.channel_width.value = D.header['CDELT4']

        try:
            self.integration_time.value = float(D.data.field('INTTIM')[0])
        except:
            if self.Ntimes.value > 1:
                self.integration_time.value = \
                    float(np.diff(np.sort(list(set(self.time_array.value))))[0])
            else:
                raise ValueError('integration time not specified and only '
                                 'one time present')

        self.freq_array.value.shape = (1,) + self.freq_array.value.shape

        self.polarization_array.value = np.int32(self._gethduaxis(D, 3))

        # other info -- not required but frequently used
        self.object_name.value = D.header.get('OBJECT', None)
        self.telescope_name.value = D.header.get('TELESCOP', None)
        self.instrument.value = D.header.get('INSTRUME', None)
        self.latitude.set_degrees(D.header.get('LAT', None))
        self.longitude.set_degrees(D.header.get('LON', None))
        self.altitude.value = D.header.get('ALT', None)
        self.dateobs.value = D.header.get('DATE-OBS', None)
        self.history.value = str(D.header.get('HISTORY', ''))
        self.vis_units.value = D.header.get('BUNIT', 'UNCALIB')
        self.phase_center_epoch.value = D.header.get('EPOCH', None)

        # find all the header items after the history and keep them as a dict
        etcpointer = 0
        for thing in D.header:
            etcpointer += 1
            if thing == 'HISTORY':
                etcpointer += 1
                break
        for key in D.header[etcpointer:]:
            if key == '':
                continue
            if key == 'HISTORY':
                continue
            self.extra_keywords.value[key] = D.header[key]

        # READ the antenna table
        ant_hdu = F[hdunames['AIPS AN']]

        # stuff in columns
        self.antenna_names.value = ant_hdu.data.field('ANNAME').tolist()

        self.antenna_indices.value = ant_hdu.data.field('NOSTA')

        self.Nants_telescope.value = len(self.antenna_indices.value)

        # stuff in the header
        if self.telescope_name.value is None:
            self.telescope_name.value = ant_hdu.header['ARRNAM']

        try:
            self.xyz_telescope_frame.value = ant_hdu.header['FRAME']
        except:
            cotter_version = D.header.get('COTVER', None)
            if cotter_version is None:
                if self.telescope_name.value == 'PAPER':
                    warnings.warn('Required Antenna frame keyword not set, '
                                  'since this is a PAPER file, setting to ITRF')
                    self.xyz_telescope_frame.value = 'ITRF'
                else:
                    warnings.warn('Required Antenna frame keyword not set, '
                                  'setting to ????')
                    self.xyz_telescope_frame.value = '????'
            else:
                warnings.warn('Required Antenna frame keyword not set, '
                              ' since this is a Cotter file, setting to ITRF')
                self.xyz_telescope_frame.value = 'ITRF'

        # VLA incorrectly sets ARRAYX/ARRAYY/ARRAYZ to 0, and puts array center
        # in the antenna positions themselves
        if (np.isclose(ant_hdu.header['ARRAYX'], 0) and
                np.isclose(ant_hdu.header['ARRAYY'], 0) and
                np.isclose(ant_hdu.header['ARRAYZ'], 0)):
            self.x_telescope.value = np.mean(ant_hdu.data['STABXYZ'][:, 0])
            self.y_telescope.value = np.mean(ant_hdu.data['STABXYZ'][:, 1])
            self.z_telescope.value = np.mean(ant_hdu.data['STABXYZ'][:, 2])
            self.antenna_positions.value = (ant_hdu.data.field('STABXYZ') -
                                            np.array([self.x_telescope.value,
                                                      self.y_telescope.value,
                                                      self.z_telescope.value]))

        else:
            self.x_telescope.value = ant_hdu.header['ARRAYX']
            self.y_telescope.value = ant_hdu.header['ARRAYY']
            self.z_telescope.value = ant_hdu.header['ARRAYZ']
            self.antenna_positions.value = ant_hdu.data.field('STABXYZ')
        self.GST0.value = ant_hdu.header['GSTIA0']
        self.RDate.value = ant_hdu.header['RDATE']
        self.earth_omega.value = ant_hdu.header['DEGPDY']
        self.DUT1.value = ant_hdu.header['UT1UTC']
        try:
            self.TIMESYS.value = ant_hdu.header['TIMESYS']
        except(KeyError):
            # CASA misspells this one
            self.TIMESYS.value = ant_hdu.header['TIMSYS']

        del(D)

        if (self.latitude.value is None or self.longitude.value is None or
                self.altitude.value is None):
            self.set_LatLonAlt_from_XYZ()

        self.set_lsts_from_time_array()

        # check if object has all required uv_properties set
        self.check()
        return True

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False):
        # first check if object has all required uv_properties set
        self.check()

        if self.phase_center_ra.value is None or self.phase_center_dec.value is None:
            if force_phase:
                print('The data does not have a defined phase center. ' +
                      'Phasing to zenith of the first timestamp.')
                self.phase(time=self.time_array.value[0])
            else:
                raise ValueError('The data does not have a defined phase ' +
                                 'center, which means it is a drift scan. ' +
                                 'Set force_phase to true to phase the data ' +
                                 'zenith of the first timestamp before ' +
                                 'writing a uvfits file.')

        for p in self.extra_property_iter():
            prop = getattr(self, p)
            if p in self.uvfits_required_extra:
                if prop.value is None:
                    if spoof_nonessential:
                        # spoof extra keywords required for uvfits
                        prop.apply_spoof(self)
                        setattr(self, p, prop)
                    else:
                        raise ValueError('Required attribute {attribute} '
                                         'for uvfits not defined. Define or '
                                         'set spoof_nonessential to True to '
                                         'spoof this attribute.'
                                         .format(attribute=p))

        weights_array = self.nsample_array.value * \
            np.where(self.flag_array.value, -1, 1)
        data_array = self.data_array.value[:, np.newaxis, np.newaxis, :, :, :,
                                           np.newaxis]
        weights_array = weights_array[:, np.newaxis, np.newaxis, :, :, :,
                                      np.newaxis]
        # uvfits_array_data shape will be  (Nblts,1,1,[Nspws],Nfreqs,Npols,3)
        uvfits_array_data = np.concatenate([data_array.real,
                                           data_array.imag,
                                           weights_array], axis=6)

        uvw_array_sec = self.uvw_array.value / const.c.to('m/s').value
        # jd_midnight = np.floor(self.time_array.value[0] - 0.5) + 0.5
        tzero = np.float32(self.time_array.value[0])

        # uvfits convention is that time_array + relevant PZERO = actual JD
        # We are setting PZERO4 = float32(first time of observation)
        time_array = np.float32(self.time_array.value - np.float64(tzero))

        int_time_array = (np.zeros_like((time_array), dtype=np.float) +
                          self.integration_time.value)

        baselines_use = self.antnums_to_baseline(self.ant_1_array.value,
                                                 self.ant_2_array.value,
                                                 attempt256=True)
        # Set up dictionaries for populating hdu
        # Note that uvfits antenna arrays are 1-indexed so we add 1
        # to our 0-indexed arrays
        group_parameter_dict = {'UU      ': uvw_array_sec[0],
                                'VV      ': uvw_array_sec[1],
                                'WW      ': uvw_array_sec[2],
                                'DATE    ': time_array,
                                'BASELINE': baselines_use,
                                'ANTENNA1': self.ant_1_array.value + 1,
                                'ANTENNA2': self.ant_2_array.value + 1,
                                'INTTIM': int_time_array}
        pscal_dict = {'UU      ': 1.0, 'VV      ': 1.0, 'WW      ': 1.0,
                      'DATE    ': 1.0, 'BASELINE': 1.0, 'ANTENNA1': 1.0,
                      'ANTENNA2': 1.0, 'INTTIM': 1.0}
        pzero_dict = {'UU      ': 0.0, 'VV      ': 0.0, 'WW      ': 0.0,
                      'DATE    ': tzero, 'BASELINE': 0.0, 'ANTENNA1': 0.0,
                      'ANTENNA2': 0.0, 'INTTIM': 0.0}

        # list contains arrays of [u,v,w,date,baseline];
        # each array has shape (Nblts)
        if (np.max(self.ant_1_array.value) < 255 and
                np.max(self.ant_2_array.value) < 255):
            # if the number of antennas is less than 256 then include both the
            # baseline array and the antenna arrays in the group parameters.
            # Otherwise just use the antenna arrays
            parnames_use = ['UU      ', 'VV      ', 'WW      ',
                            'DATE    ', 'BASELINE', 'ANTENNA1',
                            'ANTENNA2', 'INTTIM']
        else:
            parnames_use = ['UU      ', 'VV      ', 'WW      ',
                            'DATE    ', 'ANTENNA1', 'ANTENNA2', 'INTTIM']

        group_parameter_list = [group_parameter_dict[parname] for
                                parname in parnames_use]
        hdu = fits.GroupData(uvfits_array_data, parnames=parnames_use,
                             pardata=group_parameter_list, bitpix=-32)
        hdu = fits.GroupsHDU(hdu)

        for i, key in enumerate(parnames_use):
            hdu.header['PSCAL' + str(i + 1) + '  '] = pscal_dict[key]
            hdu.header['PZERO' + str(i + 1) + '  '] = pzero_dict[key]

        # ISO string of first time in self.time_array
        hdu.header['DATE-OBS'] = Time(self.time_array.value[0], scale='utc',
                                      format='jd').iso

        hdu.header['CTYPE2  '] = 'COMPLEX '
        hdu.header['CRVAL2  '] = 1.0
        hdu.header['CRPIX2  '] = 1.0
        hdu.header['CDELT2  '] = 1.0

        hdu.header['CTYPE3  '] = 'STOKES  '
        hdu.header['CRVAL3  '] = self.polarization_array.value[0]
        hdu.header['CRPIX3  '] = 1.0
        try:
            hdu.header['CDELT3  '] = np.diff(self.polarization_array.value)[0]
        except(IndexError):
            hdu.header['CDELT3  '] = 1.0

        hdu.header['CTYPE4  '] = 'FREQ    '
        hdu.header['CRVAL4  '] = self.freq_array.value[0, 0]
        hdu.header['CRPIX4  '] = 1.0
        hdu.header['CDELT4  '] = np.diff(self.freq_array.value[0])[0]

        hdu.header['CTYPE5  '] = 'IF      '
        hdu.header['CRVAL5  '] = 1.0
        hdu.header['CRPIX5  '] = 1.0
        hdu.header['CDELT5  '] = 1.0

        hdu.header['CTYPE6  '] = 'RA'
        hdu.header['CRVAL6  '] = self.phase_center_ra.degrees()

        hdu.header['CTYPE7  '] = 'DEC'
        hdu.header['CRVAL7  '] = self.phase_center_dec.degrees()

        hdu.header['BUNIT   '] = self.vis_units.value
        hdu.header['BSCALE  '] = 1.0
        hdu.header['BZERO   '] = 0.0

        hdu.header['OBJECT  '] = self.object_name.value
        hdu.header['TELESCOP'] = self.telescope_name.value
        hdu.header['LAT     '] = self.latitude.degrees()
        hdu.header['LON     '] = self.longitude.degrees()
        hdu.header['ALT     '] = self.altitude.value
        hdu.header['INSTRUME'] = self.instrument.value
        hdu.header['EPOCH   '] = float(self.phase_center_epoch.value)

        for line in self.history.value.splitlines():
            hdu.header.add_history(line)

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.value.iteritems():
            # header keywords have to be 8 characters or less
            keyword = key[:8].upper()
            # print "keyword=-value-", keyword+'='+'-'+str(value)+'-'
            hdu.header[keyword] = value

        # ADD the ANTENNA table
        staxof = np.zeros(self.Nants_telescope.value)

        # 0 specifies alt-az, 6 would specify a phased array
        mntsta = np.zeros(self.Nants_telescope.value)

        # beware, X can mean just about anything
        poltya = np.full((self.Nants_telescope.value), 'X', dtype=np.object_)
        polaa = [90.0] + np.zeros(self.Nants_telescope.value)
        poltyb = np.full((self.Nants_telescope.value), 'Y', dtype=np.object_)
        polab = [0.0] + np.zeros(self.Nants_telescope.value)

        col1 = fits.Column(name='ANNAME', format='8A',
                           array=self.antenna_names.value)
        col2 = fits.Column(name='STABXYZ', format='3D',
                           array=self.antenna_positions.value)
        col3 = fits.Column(name='NOSTA', format='1J',
                           array=self.antenna_indices.value)
        col4 = fits.Column(name='MNTSTA', format='1J', array=mntsta)
        col5 = fits.Column(name='STAXOF', format='1E', array=staxof)
        col6 = fits.Column(name='POLTYA', format='1A', array=poltya)
        col7 = fits.Column(name='POLAA', format='1E', array=polaa)
        # col8 = fits.Column(name='POLCALA', format='3E', array=polcala)
        col9 = fits.Column(name='POLTYB', format='1A', array=poltyb)
        col10 = fits.Column(name='POLAB', format='1E', array=polab)
        # col11 = fits.Column(name='POLCALB', format='3E', array=polcalb)
        # note ORBPARM is technically required, but we didn't put it in

        cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col9,
                             col10])

        ant_hdu = fits.BinTableHDU.from_columns(cols)

        ant_hdu.header['EXTNAME'] = 'AIPS AN'
        ant_hdu.header['EXTVER'] = 1

        ant_hdu.header['ARRAYX'] = self.x_telescope.value
        ant_hdu.header['ARRAYY'] = self.y_telescope.value
        ant_hdu.header['ARRAYZ'] = self.z_telescope.value
        ant_hdu.header['FRAME'] = self.xyz_telescope_frame.value
        ant_hdu.header['GSTIA0'] = self.GST0.value
        ant_hdu.header['FREQ'] = self.freq_array.value[0, 0]
        ant_hdu.header['RDATE'] = self.RDate.value
        ant_hdu.header['UT1UTC'] = self.DUT1.value

        ant_hdu.header['TIMSYS'] = self.TIMESYS.value
        if self.TIMESYS == 'IAT':
            warnings.warn('This file has an "IAT" time system. Files of '
                          'this type are not properly supported')
        ant_hdu.header['ARRNAM'] = self.telescope_name.value
        ant_hdu.header['NO_IF'] = self.Nspws.value
        ant_hdu.header['DEGPDY'] = self.earth_omega.value
        # ant_hdu.header['IATUTC'] = 35.

        # set mandatory parameters which are not supported by this object
        # (or that we just don't understand)
        ant_hdu.header['NUMORB'] = 0

        # note: Bart had this set to 3. We've set it 0 after aips 117. -jph
        ant_hdu.header['NOPCAL'] = 0

        ant_hdu.header['POLTYPE'] = 'X-Y LIN'

        # note: we do not support the concept of "frequency setups"
        # -- lists of spws given in a SU table.
        ant_hdu.header['FREQID'] = -1

        # if there are offsets in images, this could be the culprit
        ant_hdu.header['POLARX'] = 0.0
        ant_hdu.header['POLARY'] = 0.0

        ant_hdu.header['DATUTC'] = 0  # ONLY UTC SUPPORTED

        # we always output right handed coordinates
        ant_hdu.header['XYZHAND'] = 'RIGHT'

        # ADD the FQ table
        # skipping for now and limiting to a single spw

        # write the file
        hdulist = fits.HDUList(hdus=[hdu, ant_hdu])
        hdulist.writeto(filename, clobber=True)

        return True
