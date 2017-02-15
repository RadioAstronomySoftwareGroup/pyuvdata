"""Class for reading and writing uvfits files."""
from astropy import constants as const
from astropy.time import Time
from astropy.io import fits
import numpy as np
import warnings
from uvdata import UVData
import parameter as uvp


class UVFITS(UVData):
    """
    Defines a uvfits-specific subclass of UVData for reading and writing uvfits files.
    This class should not be interacted with directly, instead use the read_uvfits
    and write_uvfits methods on the UVData class.

    Attributes:
        uvfits_required_extra: Names of optional UVParameters that are required
            for uvfits.
    """

    uvfits_required_extra = ['antenna_positions', 'gst0', 'rdate',
                             'earth_omega', 'dut1', 'timesys']

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

    def read_uvfits(self, filename, run_check=True, run_check_acceptability=True):
        """
        Read in data from a uvfits file.

        Args:
            filename: The uvfits file to read from.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after reading in the file. Default is True.
        """
        F = fits.open(filename)
        D = F[0]  # assumes the visibilities are in the primary hdu
        hdr = D.header.copy()
        hdunames = self._indexhdus(F)  # find the rest of the tables

        # astropy.io fits reader scales date according to relevant PZER0 (?)
        time0_array = D.data['DATE']
        try:
            # uvfits standard is to have 2 DATE parameters, both floats:
            # DATE (full day) and _DATE (fractional day)
            time1_array = D.data['_DATE']
            self.time_array = (time0_array.astype(np.double) +
                               time1_array.astype(np.double))
        except(KeyError):
            # cotter uvfits files have one DATE that is a double
            self.time_array = time0_array
            if np.finfo(time0_array[0]).precision < 5:
                raise ValueError('JDs in this file are not precise to '
                                 'better than a second.')
            if (np.finfo(time0_array[0]).precision > 5 and
                    np.finfo(time0_array[0]).precision < 8):
                warnings.warn('The JDs in this file have sub-second '
                              'precision, but not sub-millisecond. '
                              'Use with caution.')

        self.Ntimes = len(np.unique(self.time_array))

        # if antenna arrays are present, use them. otherwise use baseline array
        try:
            # Note: uvfits antennas are 1 indexed,
            # need to subtract one to get to 0-indexed
            self.ant_1_array = np.int32(D.data.field('ANTENNA1')) - 1
            self.ant_2_array = np.int32(D.data.field('ANTENNA2')) - 1
            subarray = np.int32(D.data.field('SUBARRAY')) - 1
            # error on files with multiple subarrays
            if len(set(subarray)) > 1:
                raise ValueError('This file appears to have multiple subarray '
                                 'values; only files with one subarray are '
                                 'supported.')
        except(KeyError):
            # cannot set this to be the baseline array because it uses the
            # 256 convention, not our 2048 convention
            bl_input_array = np.int64(D.data.field('BASELINE'))

            # get antenna arrays based on uvfits baseline array
            self.ant_1_array, self.ant_2_array = \
                self.baseline_to_antnums(bl_input_array)
        # check for multi source files
        try:
            source = D.data.field('SOURCE')
            if len(set(source)) > 1:
                raise ValueError('This file has multiple sources. Only single '
                                 'source observations are supported.')
        except:
            pass

        # get self.baseline_array using our convention
        self.baseline_array = \
            self.antnums_to_baseline(self.ant_1_array,
                                     self.ant_2_array)
        self.Nbls = len(np.unique(self.baseline_array))

        # initialize internal variables based on the antenna table
        self.Nants_data = int(len(np.unique(self.ant_1_array.tolist() + self.ant_2_array.tolist())))

        self.set_phased()
        # check if we have an spw dimension
        if hdr.pop('NAXIS') == 7:
            if hdr['NAXIS5'] > 1:
                raise ValueError('Sorry.  Files with more than one spectral' +
                                 'window (spw) are not yet supported. A ' +
                                 'great project for the interested student!')
            self.data_array = (D.data.field('DATA')[:, 0, 0, :, :, :, 0] +
                               1j * D.data.field('DATA')[:, 0, 0, :, :, :, 1])
            self.flag_array = (D.data.field('DATA')[:, 0, 0, :, :, :, 2] <= 0)
            self.nsample_array = np.abs(D.data.field('DATA')[:, 0, 0, :, :, :, 2])
            self.Nspws = hdr.pop('NAXIS5')
            assert(self.Nspws == self.data_array.shape[1])

            # the axis number for phase center depends on if the spw exists
            # subtract 1 to be zero-indexed
            self.spw_array = np.int32(self._gethduaxis(D, 5)) - 1

            self.phase_center_ra_degrees = np.float(hdr.pop('CRVAL6'))
            self.phase_center_dec_degrees = np.float(hdr.pop('CRVAL7'))
        else:
            # in many uvfits files the spw axis is left out,
            # here we put it back in so the dimensionality stays the same
            self.data_array = (D.data.field('DATA')[:, 0, 0, :, :, 0] +
                               1j * D.data.field('DATA')[:, 0, 0, :, :, 1])
            self.data_array = self.data_array[:, np.newaxis, :, :]
            self.flag_array = (D.data.field('DATA')[:, 0, 0, :, :, 2] <= 0)
            self.flag_array = self.flag_array[:, np.newaxis, :, :]
            self.nsample_array = np.abs(D.data.field('DATA')[:, 0, 0, :, :, 2])
            self.nsample_array = (self.nsample_array[:, np.newaxis, :, :])

            # the axis number for phase center depends on if the spw exists
            self.Nspws = 1
            self.spw_array = np.array([0])

            self.phase_center_ra_degrees = np.float(hdr.pop('CRVAL5'))
            self.phase_center_dec_degrees = np.float(hdr.pop('CRVAL6'))

        # get shapes
        self.Nfreqs = hdr.pop('NAXIS4')
        assert(self.Nfreqs == self.data_array.shape[2])
        self.Npols = hdr.pop('NAXIS3')
        assert(self.Npols == self.data_array.shape[3])
        self.Nblts = hdr.pop('GCOUNT')
        assert(self.Nblts == self.data_array.shape[0])

        # read baseline vectors in units of seconds, return in meters
        self.uvw_array = (np.array(np.stack((D.data.field('UU'),
                                             D.data.field('VV'),
                                             D.data.field('WW')))) *
                          const.c.to('m/s').value).T

        self.freq_array = self._gethduaxis(D, 4)
        self.channel_width = hdr.pop('CDELT4')

        try:
            self.integration_time = float(D.data.field('INTTIM')[0])
        except:
            if self.Ntimes > 1:
                self.integration_time = \
                    float(np.diff(np.sort(list(set(self.time_array))))[0]) * 86400
            else:
                raise ValueError('integration time not specified and only '
                                 'one time present')

        self.freq_array.shape = (self.Nspws,) + self.freq_array.shape

        self.polarization_array = np.int32(self._gethduaxis(D, 3))

        # other info -- not required but frequently used
        self.object_name = hdr.pop('OBJECT', None)
        self.telescope_name = hdr.pop('TELESCOP', None)
        self.instrument = hdr.pop('INSTRUME', None)
        latitude_degrees = hdr.pop('LAT', None)
        longitude_degrees = hdr.pop('LON', None)
        altitude = hdr.pop('ALT', None)
        self.history = str(hdr.get('HISTORY', ''))
        while 'HISTORY' in hdr.keys():
            hdr.remove('HISTORY')
        self.vis_units = hdr.pop('BUNIT', 'UNCALIB')
        self.phase_center_epoch = hdr.pop('EPOCH', None)

        # remove standard FITS header items that are still around
        std_fits_substrings = ['SIMPLE', 'BITPIX', 'EXTEND', 'BLOCKED',
                               'GROUPS', 'PCOUNT', 'BSCALE', 'BZERO', 'NAXIS',
                               'PTYPE', 'PSCAL', 'PZERO', 'CTYPE', 'CRVAL',
                               'CRPIX', 'CDELT', 'CROTA']
        for key in hdr.keys():
            for sub in std_fits_substrings:
                if key.find(sub) > -1:
                    hdr.remove(key)

        # find all the remaining header items and keep them as extra_keywords
        for key in hdr:
            if key == '':
                continue
            if key == 'COMMENT':
                self.extra_keywords[key] = str(hdr.get(key))
            else:
                self.extra_keywords[key] = hdr.get(key)

        # READ the antenna table
        ant_hdu = F[hdunames['AIPS AN']]

        # stuff in columns
        ant_names = ant_hdu.data.field('ANNAME').tolist()
        self.antenna_names = []
        for name in ant_names:
            self.antenna_names.append(name.replace('\x00!', ''))

        # subtract one to get to 0-indexed values rather than 1-indexed values
        self.antenna_numbers = ant_hdu.data.field('NOSTA') - 1

        self.Nants_telescope = len(self.antenna_numbers)

        # stuff in the header
        if self.telescope_name is None:
            self.telescope_name = ant_hdu.header['ARRNAM']

        try:
            xyz_telescope_frame = ant_hdu.header['FRAME']
        except:
            warnings.warn('Required Antenna frame keyword not set, '
                          'setting to ????')
            xyz_telescope_frame = '????'

        # VLA incorrectly sets ARRAYX/ARRAYY/ARRAYZ to 0, and puts array center
        # in the antenna positions themselves

        if (np.isclose(ant_hdu.header['ARRAYX'], 0) and
                np.isclose(ant_hdu.header['ARRAYY'], 0) and
                np.isclose(ant_hdu.header['ARRAYZ'], 0)):
            x_telescope = np.mean(ant_hdu.data['STABXYZ'][:, 0])
            y_telescope = np.mean(ant_hdu.data['STABXYZ'][:, 1])
            z_telescope = np.mean(ant_hdu.data['STABXYZ'][:, 2])
            self.antenna_positions = (ant_hdu.data.field('STABXYZ') -
                                      np.array([x_telescope,
                                                y_telescope,
                                                z_telescope]))

        else:
            x_telescope = ant_hdu.header['ARRAYX']
            y_telescope = ant_hdu.header['ARRAYY']
            z_telescope = ant_hdu.header['ARRAYZ']
            self.antenna_positions = ant_hdu.data.field('STABXYZ')

        if xyz_telescope_frame == 'ITRF':
            self.telescope_location = np.array([x_telescope, y_telescope, z_telescope])
        else:
            if latitude_degrees is not None and longitude_degrees is not None and altitude is not None:
                self.telescope_location_lat_lon_alt_degrees = (latitude_degrees, longitude_degrees, altitude)

        self.gst0 = ant_hdu.header['GSTIA0']
        self.rdate = ant_hdu.header['RDATE']
        self.earth_omega = ant_hdu.header['DEGPDY']
        self.dut1 = ant_hdu.header['UT1UTC']
        try:
            self.timesys = ant_hdu.header['TIMESYS']
        except(KeyError):
            # CASA misspells this one
            self.timesys = ant_hdu.header['TIMSYS']

        del(D)

        try:
            self.set_telescope_params()
        except ValueError, ve:
            warnings.warn(str(ve))

        self.set_lsts_from_time_array()

        # check if object has all required UVParameters set
        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False, run_check=True, run_check_acceptability=True):
        """
        Write the data to a uvfits file.

        Args:
            filename: The uvfits file to write to.
            spoof_nonessential: Option to spoof the values of optional
                UVParameters that are not set but are required for uvfits files.
                Default is False.
            force_phase: Option to automatically phase drift scan data to
                zenith of the first timestamp. Default is False.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters before writing the file. Default is True.
        """
        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

        if self.phase_type == 'phased':
            pass
        elif self.phase_type == 'drift':
            if force_phase:
                print('The data are in drift mode and do not have a '
                      'defined phase center. Phasing to zenith of the first '
                      'timestamp.')
                self.phase_to_time(self.time_array[0])
            else:
                raise ValueError('The data are in drift mode. ' +
                                 'Set force_phase to true to phase the data ' +
                                 'to zenith of the first timestamp before ' +
                                 'writing a uvfits file.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to drift or phased to '
                             'reflect the phasing status of the data')

        for p in self.extra():
            param = getattr(self, p)
            if param.name in self.uvfits_required_extra:
                if param.value is None:
                    if spoof_nonessential:
                        # spoof extra keywords required for uvfits
                        if isinstance(param, uvp.AntPositionParameter):
                            param.apply_spoof(self, 'Nants_telescope')
                        else:
                            param.apply_spoof()
                        setattr(self, p, param)
                    else:
                        raise ValueError('Required attribute {attribute} '
                                         'for uvfits not defined. Define or '
                                         'set spoof_nonessential to True to '
                                         'spoof this attribute.'
                                         .format(attribute=p))

        weights_array = self.nsample_array * \
            np.where(self.flag_array, -1, 1)
        data_array = self.data_array[:, np.newaxis, np.newaxis, :, :, :, np.newaxis]
        weights_array = weights_array[:, np.newaxis, np.newaxis, :, :, :,
                                      np.newaxis]
        # uvfits_array_data shape will be  (Nblts,1,1,[Nspws],Nfreqs,Npols,3)
        uvfits_array_data = np.concatenate([data_array.real,
                                           data_array.imag,
                                           weights_array], axis=6)

        uvw_array_sec = self.uvw_array / const.c.to('m/s').value
        # jd_midnight = np.floor(self.time_array[0] - 0.5) + 0.5
        tzero = np.float32(self.time_array[0])

        # uvfits convention is that time_array + relevant PZERO = actual JD
        # We are setting PZERO4 = float32(first time of observation)
        time_array = np.float32(self.time_array - np.float64(tzero))

        int_time_array = (np.zeros_like((time_array), dtype=np.float) +
                          self.integration_time)

        baselines_use = self.antnums_to_baseline(self.ant_1_array,
                                                 self.ant_2_array,
                                                 attempt256=True)
        # Set up dictionaries for populating hdu
        # Note that uvfits antenna arrays are 1-indexed so we add 1
        # to our 0-indexed arrays
        group_parameter_dict = {'UU      ': uvw_array_sec[:, 0],
                                'VV      ': uvw_array_sec[:, 1],
                                'WW      ': uvw_array_sec[:, 2],
                                'DATE    ': time_array,
                                'BASELINE': baselines_use,
                                'ANTENNA1': self.ant_1_array + 1,
                                'ANTENNA2': self.ant_2_array + 1,
                                'SUBARRAY': np.ones_like(self.ant_1_array),
                                'INTTIM': int_time_array}
        pscal_dict = {'UU      ': 1.0, 'VV      ': 1.0, 'WW      ': 1.0,
                      'DATE    ': 1.0, 'BASELINE': 1.0, 'ANTENNA1': 1.0,
                      'ANTENNA2': 1.0, 'SUBARRAY': 1.0, 'INTTIM': 1.0}
        pzero_dict = {'UU      ': 0.0, 'VV      ': 0.0, 'WW      ': 0.0,
                      'DATE    ': tzero, 'BASELINE': 0.0, 'ANTENNA1': 0.0,
                      'ANTENNA2': 0.0, 'SUBARRAY': 0.0, 'INTTIM': 0.0}

        # list contains arrays of [u,v,w,date,baseline];
        # each array has shape (Nblts)
        if (np.max(self.ant_1_array) < 255 and
                np.max(self.ant_2_array) < 255):
            # if the number of antennas is less than 256 then include both the
            # baseline array and the antenna arrays in the group parameters.
            # Otherwise just use the antenna arrays
            parnames_use = ['UU      ', 'VV      ', 'WW      ',
                            'DATE    ', 'BASELINE', 'ANTENNA1',
                            'ANTENNA2', 'SUBARRAY', 'INTTIM']
        else:
            parnames_use = ['UU      ', 'VV      ', 'WW      ', 'DATE    ',
                            'ANTENNA1', 'ANTENNA2', 'SUBARRAY', 'INTTIM']

        group_parameter_list = [group_parameter_dict[parname] for
                                parname in parnames_use]
        hdu = fits.GroupData(uvfits_array_data, parnames=parnames_use,
                             pardata=group_parameter_list, bitpix=-32)
        hdu = fits.GroupsHDU(hdu)

        for i, key in enumerate(parnames_use):
            hdu.header['PSCAL' + str(i + 1) + '  '] = pscal_dict[key]
            hdu.header['PZERO' + str(i + 1) + '  '] = pzero_dict[key]

        # ISO string of first time in self.time_array
        hdu.header['DATE-OBS'] = Time(self.time_array[0], scale='utc',
                                      format='jd').isot

        hdu.header['CTYPE2  '] = 'COMPLEX '
        hdu.header['CRVAL2  '] = 1.0
        hdu.header['CRPIX2  '] = 1.0
        hdu.header['CDELT2  '] = 1.0

        hdu.header['CTYPE3  '] = 'STOKES  '
        hdu.header['CRVAL3  '] = self.polarization_array[0]
        hdu.header['CRPIX3  '] = 1.0
        try:
            hdu.header['CDELT3  '] = np.diff(self.polarization_array)[0]
        except(IndexError):
            hdu.header['CDELT3  '] = 1.0

        hdu.header['CTYPE4  '] = 'FREQ    '
        hdu.header['CRVAL4  '] = self.freq_array[0, 0]
        hdu.header['CRPIX4  '] = 1.0
        hdu.header['CDELT4  '] = np.diff(self.freq_array[0])[0]

        hdu.header['CTYPE5  '] = 'IF      '
        hdu.header['CRVAL5  '] = 1.0
        hdu.header['CRPIX5  '] = 1.0
        hdu.header['CDELT5  '] = 1.0

        hdu.header['CTYPE6  '] = 'RA'
        hdu.header['CRVAL6  '] = self.phase_center_ra_degrees

        hdu.header['CTYPE7  '] = 'DEC'
        hdu.header['CRVAL7  '] = self.phase_center_dec_degrees

        hdu.header['BUNIT   '] = self.vis_units
        hdu.header['BSCALE  '] = 1.0
        hdu.header['BZERO   '] = 0.0

        hdu.header['OBJECT  '] = self.object_name
        hdu.header['TELESCOP'] = self.telescope_name
        hdu.header['LAT     '] = self.telescope_location_lat_lon_alt_degrees[0]
        hdu.header['LON     '] = self.telescope_location_lat_lon_alt_degrees[1]
        hdu.header['ALT     '] = self.telescope_location_lat_lon_alt[2]
        hdu.header['INSTRUME'] = self.instrument
        hdu.header['EPOCH   '] = float(self.phase_center_epoch)

        for line in self.history.splitlines():
            hdu.header.add_history(line)

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.iteritems():
            # header keywords have to be 8 characters or less
            keyword = key[:8].upper()
            # print "keyword=-value-", keyword+'='+'-'+str(value)+'-'
            if keyword == 'COMMENT':
                for line in value.splitlines():
                    hdu.header.add_comment(line)
            else:
                hdu.header[keyword] = value

        # ADD the ANTENNA table
        staxof = np.zeros(self.Nants_telescope)

        # 0 specifies alt-az, 6 would specify a phased array
        mntsta = np.zeros(self.Nants_telescope)

        # beware, X can mean just about anything
        poltya = np.full((self.Nants_telescope), 'X', dtype=np.object_)
        polaa = [90.0] + np.zeros(self.Nants_telescope)
        poltyb = np.full((self.Nants_telescope), 'Y', dtype=np.object_)
        polab = [0.0] + np.zeros(self.Nants_telescope)

        col1 = fits.Column(name='ANNAME', format='8A',
                           array=self.antenna_names)
        col2 = fits.Column(name='STABXYZ', format='3D',
                           array=self.antenna_positions)
        # convert to 1-indexed from 0-indexed indicies
        col3 = fits.Column(name='NOSTA', format='1J',
                           array=self.antenna_numbers + 1)
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

        # write XYZ coordinates if not already defined
        ant_hdu.header['ARRAYX'] = self.telescope_location[0]
        ant_hdu.header['ARRAYY'] = self.telescope_location[1]
        ant_hdu.header['ARRAYZ'] = self.telescope_location[2]
        ant_hdu.header['FRAME'] = 'ITRF'
        ant_hdu.header['GSTIA0'] = self.gst0
        ant_hdu.header['FREQ'] = self.freq_array[0, 0]
        ant_hdu.header['RDATE'] = self.rdate
        ant_hdu.header['UT1UTC'] = self.dut1

        ant_hdu.header['TIMSYS'] = self.timesys
        if self.timesys == 'IAT':
            warnings.warn('This file has an "IAT" time system. Files of '
                          'this type are not properly supported')
        ant_hdu.header['ARRNAM'] = self.telescope_name
        ant_hdu.header['NO_IF'] = self.Nspws
        ant_hdu.header['DEGPDY'] = self.earth_omega
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
