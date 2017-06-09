import astropy
from astropy.io import fits
import numpy as np
import warnings
from uvcal import UVCal
import utils as uvutils


class CALFITS(UVCal):
    """
    Defines a calfits-specific class for reading and writing calfits files.
    """

    def write_calfits(self, filename, run_check=True,
                      run_check_acceptability=True, clobber=False):
        """
        Write the data to a calfits file.

        Args:
            filename: The calfits file to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptability of the values of
                required parameters before writing the file. Default is True.

        """
        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

        if self.Nfreqs > 1:
            freq_spacing = self.freq_array[0, 1:] - self.freq_array[0, :-1]
            if not np.isclose(np.min(freq_spacing), np.max(freq_spacing),
                              rtol=self._freq_array.tols[0], atol=self._freq_array.tols[1]):
                raise ValueError('The frequencies are not evenly spaced (probably '
                                 'because of a select operation). The calfits format '
                                 'does not support unevenly spaced frequencies.')
            if np.isclose(freq_spacing[0], self.channel_width):
                freq_spacing = self.channel_width
            else:
                rounded_spacing = np.around(freq_spacing, int(np.ceil(np.log10(self._freq_array.tols[1]) * -1)))
                freq_spacing = rounded_spacing[0]
        else:
            freq_spacing = self.channel_width

        if self.Ntimes > 1:
            time_spacing = np.diff(self.time_array)
            if not np.isclose(np.min(time_spacing), np.max(time_spacing),
                              rtol=self._time_array.tols[0], atol=self._time_array.tols[1]):
                raise ValueError('The times are not evenly spaced (probably '
                                 'because of a select operation). The calfits format '
                                 'does not support unevenly spaced times.')
            if np.isclose(time_spacing[0], self.integration_time):
                time_spacing = self.integration_time
            else:
                rounded_spacing = np.around(time_spacing, int(np.ceil(np.log10(self._time_array.tols[1]) * -1)))
                time_spacing = rounded_spacing[0]
        else:
            time_spacing = self.integration_time

        if self.Njones > 1:
            jones_spacing = np.diff(self.jones_array)
            if np.min(jones_spacing) < np.max(jones_spacing):
                raise ValueError('The jones values are not evenly spaced.'
                                 'The calibration fits file format does not'
                                 ' support unevenly spaced polarizations.')

        prihdr = fits.Header()
        if self.total_quality_array is not None:
            totqualhdr = fits.Header()
            totqualhdr['EXTNAME'] = 'TOTQLTY'
        if self.cal_type != 'gain':
            sechdr = fits.Header()
            sechdr['EXTNAME'] = 'FLAGS'
        # Conforming to fits format
        prihdr['SIMPLE'] = True
        prihdr['BITPIX'] = 32
        prihdr['TELESCOP'] = self.telescope_name
        prihdr['GNCONVEN'] = self.gain_convention
        prihdr['CALTYPE'] = self.cal_type
        prihdr['INTTIME'] = self.integration_time
        prihdr['CHWIDTH'] = self.channel_width
        prihdr['XORIENT'] = self.x_orientation
        if self.cal_type == 'delay':
            prihdr['FRQRANGE'] = ','.join(map(str, self.freq_range))
        elif self.freq_range is not None:
            prihdr['FRQRANGE'] = ','.join(map(str, self.freq_range))
        prihdr['TMERANGE'] = ','.join(map(str, self.time_range))
        for line in self.history.splitlines():
            prihdr.add_history(line)

        for p in self.extra():
            ep = getattr(self, p)
            if ep.form is 'str':
                prihdr['{0}'.format(p.upper().replace('_', '')[:8])] = ep.value
            else:
                continue

        if self.observer:
            prihdr['OBSERVER'] = self.observer
        if self.git_origin_cal:
            prihdr['ORIGCAL'] = self.git_origin_cal
        if self.git_hash_cal:
            prihdr['HASHCAL'] = self.git_hash_cal

        if self.cal_type == 'unknown':
            raise ValueError("unknown calibration type. Do not know how to"
                             "store parameters")

        if self.cal_type == 'gain':
            # Set header variable for gain.
            prihdr['CTYPE4'] = ('FREQS', 'Frequency.')
            prihdr['CUNIT4'] = 'Hz'
            prihdr['CRPIX4'] = 1
            prihdr['CRVAL4'] = self.freq_array[0][0]
            prihdr['CDELT4'] = freq_spacing

            # Nspws axis: number of spectral windows
            prihdr['CTYPE5'] = ('NSPWS', 'Number of spectral windows.')
            prihdr['CUNIT5'] = 'Integer'
            prihdr['CRPIX5'] = 1
            prihdr['CRVAL5'] = 1
            prihdr['CDELT5'] = 1

            # ANTAXIS axis number differs between delay and gain because there's no frequency axis in delay
            prihdr['CTYPE6'] = ('ANTAXIS', 'See ANTARR in ANTENNA extension for values.')
            prihdr['CUNIT6'] = 'Integer'
            prihdr['CRPIX6'] = 1
            prihdr['CRVAL6'] = 1
            prihdr['CDELT6'] = -1

            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = 'Integer'
            prihdr['CDELT1'] = 1
            prihdr['CRPIX1'] = 1
            prihdr['CRVAL1'] = 1
            if self.input_flag_array is not None:
                pridata = np.concatenate([self.gain_array.real[:, :, :, :, :, np.newaxis],
                                          self.gain_array.imag[:, :, :, :, :, np.newaxis],
                                          self.flag_array[:, :, :, :, :, np.newaxis],
                                          self.input_flag_array[:, :, :, :, :, np.newaxis],
                                          self.quality_array[:, :, :, :, :, np.newaxis]],
                                         axis=-1)
            else:
                pridata = np.concatenate([self.gain_array.real[:, :, :, :, :, np.newaxis],
                                          self.gain_array.imag[:, :, :, :, :, np.newaxis],
                                          self.flag_array[:, :, :, :, :, np.newaxis],
                                          self.quality_array[:, :, :, :, :, np.newaxis]],
                                         axis=-1)

        if self.total_quality_array is not None:
            # Set headers for the hdu containing the total_quality_array
            totqualhdr['CTYPE1'] = ('JONES', 'Jones matrix array')
            totqualhdr['CUNIT1'] = ('Integer', 'representative integer for polarization.')
            totqualhdr['CRPIX1'] = 1
            totqualhdr['CRVAL1'] = self.jones_array[0]  # always start with first jones.
            if self.Njones > 1:
                totqualhdr['CDELT1'] = jones_spacing[0]
            else:
                totqualhdr['CDELT1'] = -1

            totqualhdr['CTYPE2'] = ('TIME', 'Time axis.')
            totqualhdr['CUNIT2'] = ('JD', 'Time in julian date format')
            totqualhdr['CRPIX2'] = 1
            totqualhdr['CRVAL2'] = self.time_array[0]
            totqualhdr['CDELT2'] = time_spacing

            totqualhdr['CTYPE3'] = ('FREQS', 'Valid frequencies to apply delay.')
            totqualhdr['CUNIT3'] = 'Hz'
            totqualhdr['CRPIX3'] = 1
            totqualhdr['CRVAL3'] = self.freq_array[0][0]
            totqualhdr['CDELT3'] = freq_spacing

            # Nspws axis: number of spectral windows
            totqualhdr['CTYPE4'] = ('NSPWS', 'Number of spectral windows.')
            totqualhdr['CUNIT4'] = 'Integer'
            totqualhdr['CRPIX4'] = 1
            totqualhdr['CRVAL4'] = 1
            totqualhdr['CDELT4'] = 1
            totqualdata = self.total_quality_array

        if self.cal_type == 'delay':
            # Set header variable for delay.

            # ANTAXIS axis number differs between delay and gain because there's no frequency axis in delay
            prihdr['CTYPE5'] = ('ANTAXIS', 'See ANTARR in ANTENNA extension for values.')
            prihdr['CUNIT5'] = 'Integer'
            prihdr['CRVAL5'] = 1
            prihdr['CRPIX5'] = 1
            prihdr['CDELT5'] = -1

            # Nspws axis: number of spectral windows
            prihdr['CTYPE4'] = ('NSPWS', 'Number of spectral windows.')
            prihdr['CUNIT4'] = 'Integer'
            prihdr['CRPIX4'] = 1
            prihdr['CRVAL4'] = 1
            prihdr['CDELT4'] = 1

            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = 'Integer'
            prihdr['CRPIX1'] = 1
            prihdr['CRVAL1'] = 1
            prihdr['CDELT1'] = 1

            pridata = np.concatenate([self.delay_array[:, :, :, :, np.newaxis],
                                      self.quality_array[:, :, :, :, np.newaxis]],
                                     axis=-1)

            # Set headers for the second hdu containing the flags. Only in cal_type=delay.
            sechdr['CTYPE2'] = ('JONES', 'Jones matrix array')
            sechdr['CUNIT2'] = ('Integer', 'representative integer for polarization.')
            sechdr['CRPIX2'] = 1
            sechdr['CRVAL2'] = self.jones_array[0]  # always start with first jones.
            if self.Njones > 1:
                sechdr['CDELT2'] = jones_spacing[0]
            else:
                sechdr['CDELT2'] = -1

            sechdr['CTYPE3'] = ('TIME', 'Time axis.')
            sechdr['CUNIT3'] = ('JD', 'Time in julian date format')
            sechdr['CRPIX3'] = 1
            sechdr['CRVAL3'] = self.time_array[0]
            sechdr['CDELT3'] = time_spacing

            sechdr['CTYPE4'] = ('FREQS', 'Valid frequencies to apply delay.')
            sechdr['CUNIT4'] = 'Hz'
            sechdr['CRPIX4'] = 1
            sechdr['CRVAL4'] = self.freq_array[0][0]
            sechdr['CDELT4'] = freq_spacing

            # Nspws axis: number of spectral windows
            sechdr['CTYPE5'] = ('NSPWS', 'Number of spectral windows.')
            sechdr['CUNIT5'] = 'Integer'
            sechdr['CRPIX5'] = 1
            sechdr['CRVAL5'] = 1
            sechdr['CDELT5'] = 1

            sechdr['CTYPE6'] = ('ANTAXIS', 'See ANTARR in ANTENNA extension for values.')

            if self.input_flag_array is not None:
                secdata = np.concatenate([self.flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis],
                                          self.input_flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis]],
                                         axis=-1)
                sechdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
                sechdr['CUNIT1'] = 'Integer'
                sechdr['CRPIX1'] = 1
                sechdr['CRVAL1'] = 1
                sechdr['CDELT1'] = 1

            else:
                secdata = self.flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis]  # Can't be bool
                sechdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
                sechdr['CUNIT1'] = 'Integer'
                sechdr['CRPIX1'] = 1
                sechdr['CRVAL1'] = 1
                sechdr['CDELT1'] = 1

        # primary header ctypes for NAXIS [ for both gain and delay cal_type.]
        # Check polarizations.
        prihdr['CTYPE2'] = ('JONES', 'Jones matrix array')
        prihdr['CUNIT2'] = ('Integer', 'representative integer for polarization.')
        prihdr['CRPIX2'] = 1
        prihdr['CRVAL2'] = self.jones_array[0]  # always start with first jones.
        if self.Njones > 1:
            prihdr['CDELT2'] = jones_spacing[0]
        else:
            prihdr['CDELT2'] = -1

        prihdr['CTYPE3'] = ('TIME', 'Time axis.')
        prihdr['CUNIT3'] = ('JD', 'Time in julian date format')
        prihdr['CRPIX3'] = 1
        prihdr['CRVAL3'] = self.time_array[0]
        prihdr['CDELT3'] = time_spacing

        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)

        col1 = fits.Column(name='ANTNAME', format='8A',
                           array=self.antenna_names)
        col2 = fits.Column(name='ANTINDEX', format='D',
                           array=self.antenna_numbers)
        if self.Nants_data == self.Nants_telescope:
            col3 = fits.Column(name='ANTARR', format='D',
                               array=self.ant_array)
        else:
            # ant_array is shorter than the other columns.
            # Pad the extra rows with -1s. Need to undo on read.
            nants_add = self.Nants_telescope - self.Nants_data
            ant_array_use = np.append(self.ant_array,
                                      np.zeros(nants_add, dtype=np.int) - 1)
            col3 = fits.Column(name='ANTARR', format='D',
                               array=ant_array_use)
        cols = fits.ColDefs([col1, col2, col3])
        ant_hdu = fits.BinTableHDU.from_columns(cols)
        ant_hdu.header['EXTNAME'] = 'ANTENNAS'

        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])

        if self.cal_type != 'gain':
            sechdu = fits.ImageHDU(data=secdata, header=sechdr)
            hdulist.append(sechdu)

        if self.total_quality_array is not None:
            totqualhdu = fits.ImageHDU(data=totqualdata, header=totqualhdr)
            hdulist.append(totqualhdu)

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(filename, clobber=clobber)
        else:
            hdulist.writeto(filename, overwrite=clobber)

    def read_calfits(self, filename, run_check=True, run_check_acceptability=True,
                     strict_fits=False):
        F = fits.open(filename)
        data = F[0].data
        hdr = F[0].header.copy()
        hdunames = uvutils.fits_indexhdus(F)

        anthdu = F[hdunames['ANTENNAS']]
        self.Nants_telescope = anthdu.header['NAXIS2']
        antdata = anthdu.data
        self.antenna_names = map(str, antdata['ANTNAME'])
        self.antenna_numbers = map(int, antdata['ANTINDEX'])
        self.ant_array = np.array(map(int, antdata['ANTARR']))
        if np.min(self.ant_array) < 0:
            # ant_array was shorter than the other columns, so it was padded with -1s.
            # Remove the padded entries.
            self.ant_array = self.ant_array[np.where(self.ant_array >= 0)[0]]

        self.channel_width = hdr['CHWIDTH']
        self.integration_time = hdr['INTTIME']
        self.telescope_name = hdr['TELESCOP']
        self.history = str(hdr.get('HISTORY', ''))
        if self.pyuvdata_version_str not in self.history.replace('\n', ''):
            if self.history.endswith('\n'):
                self.history += self.pyuvdata_version_str
            else:
                self.history += '\n' + self.pyuvdata_version_str
        while 'HISTORY' in hdr.keys():
            hdr.remove('HISTORY')
        self.time_range = map(float, hdr['TMERANGE'].split(','))
        self.gain_convention = hdr['GNCONVEN']
        self.x_orientation = hdr['XORIENT']
        self.cal_type = hdr['CALTYPE']
        if self.cal_type == 'delay':
            self.freq_range = map(float, hdr['FRQRANGE'].split(','))
        else:
            try:
                self.freq_range = map(float, hdr['FRQRANGE'].split(','))
            except(KeyError):
                pass
        try:
            self.observer = hdr['OBSERVER']
        except(KeyError):
            pass
        try:
            self.git_origin_cal = hdr['ORIGCAL']
        except(KeyError):
            pass
        try:
            self.git_hash_cal = hdr['HASHCAL']
        except(KeyError):
            pass

        # generate polarization and time array for either cal_type.
        self.Njones = hdr['NAXIS2']
        self.jones_array = uvutils.fits_gethduaxis(F[0], 2, strict_fits=strict_fits)
        self.Ntimes = hdr['NAXIS3']
        self.time_array = uvutils.fits_gethduaxis(F[0], 3, strict_fits=strict_fits)

        # get data.
        if self.cal_type == 'gain':
            self.set_gain()
            self.gain_array = data[:, :, :, :, :, 0] + 1j * data[:, :, :, :, :, 1]
            self.flag_array = data[:, :, :, :, :, 2].astype('bool')
            if hdr['NAXIS1'] == 5:
                self.input_flag_array = data[:, :, :, :, :, 3].astype('bool')
                self.quality_array = data[:, :, :, :, :, 4]
            else:
                self.quality_array = data[:, :, :, :, :, 3]

            self.Nants_data = hdr['NAXIS6']

            self.Nspws = hdr['NAXIS5']
            # add this for backwards compatibility when the spw CRVAL wasn't recorded
            try:
                self.spw_array = uvutils.fits_gethduaxis(F[0], 5, strict_fits=strict_fits)
            except(KeyError):
                if not strict_fits:
                    warnings.warn('{file} appears to be an old calfits format '
                                  'which does not fully conform to the FITS standard. '
                                  'Setting default values now, set strict_fits=True '
                                  'to error rather than warn on this problem, '
                                  'rewrite this file with write_calfits to ensure '
                                  'FITS compliance.'.format(file=filename))
                    self.spw_array = np.array([0])
                else:
                    raise

            # generate frequency array from primary data unit.
            self.Nfreqs = hdr['NAXIS4']
            self.freq_array = uvutils.fits_gethduaxis(F[0], 4, strict_fits=strict_fits)
            self.freq_array.shape = (self.Nspws,) + self.freq_array.shape

        if self.cal_type == 'delay':
            self.set_delay()
            self.delay_array = data[:, :, :, :, 0]
            self.quality_array = data[:, :, :, :, 1]
            sechdu = F[hdunames['FLAGS']]
            flag_data = sechdu.data
            flag_hdr = sechdu.header
            if sechdu.header['NAXIS1'] == 2:
                self.flag_array = flag_data[:, :, :, :, :, 0].astype('bool')
                self.input_flag_array = flag_data[:, :, :, :, :, 1]
            else:
                self.flag_array = flag_data[:, :, :, :, :, 0].astype('bool')

            self.Nants_data = hdr['NAXIS5']

            self.Nspws = hdr['NAXIS4']
            # add this for backwards compatibility when the spw CRVAL wasn't recorded
            try:
                self.spw_array = uvutils.fits_gethduaxis(F[0], 4, strict_fits=strict_fits)
            except(KeyError):
                if not strict_fits:
                    warnings.warn('{file} appears to be an old calfits format '
                                  'which does not fully conform to the FITS standard. '
                                  'Setting default values now, set strict_fits=True '
                                  'to error rather than warn on this problem, '
                                  'rewrite this file with write_calfits to ensure '
                                  'FITS compliance.'.format(file=filename))
                    self.spw_array = np.array([0])
                else:
                    raise

            # generate frequency array from flag data unit (no freq axis in primary).
            self.Nfreqs = sechdu.header['NAXIS4']
            self.freq_array = uvutils.fits_gethduaxis(sechdu, 4, strict_fits=strict_fits)
            self.freq_array.shape = (self.Nspws,) + self.freq_array.shape

            # add this for backwards compatibility when the spw CRVAL wasn't recorded
            try:
                spw_array = uvutils.fits_gethduaxis(sechdu, 5, strict_fits=strict_fits)
            except(KeyError):
                if not strict_fits:
                    warnings.warn('{file} appears to be an old calfits format '
                                  'which does not fully conform to the FITS standard. '
                                  'Setting default values now, set strict_fits=True '
                                  'to error rather than warn on this problem, '
                                  'rewrite this file with write_calfits to ensure '
                                  'FITS compliance.'.format(file=filename))
                    spw_array = np.array([0])
                else:
                    raise
            if not np.allclose(spw_array, self.spw_array):
                raise ValueError('Spectral window values are different in FLAGS HDU than in primary HDU')

            time_array = uvutils.fits_gethduaxis(sechdu, 3, strict_fits=strict_fits)
            if not np.allclose(time_array, self.time_array,
                               rtol=self._time_array.tols[0],
                               atol=self._time_array.tols[0]):
                raise ValueError('Time values are different in FLAGS HDU than in primary HDU')

            jones_array = uvutils.fits_gethduaxis(sechdu, 2, strict_fits=strict_fits)
            if not np.allclose(jones_array, self.jones_array,
                               rtol=self._jones_array.tols[0],
                               atol=self._jones_array.tols[0]):
                raise ValueError('Jones values are different in FLAGS HDU than in primary HDU')

        # get total quality array if present
        try:
            totqualhdu = F[hdunames['TOTQLTY']]
            self.total_quality_array = totqualhdu.data

            try:
                spw_array = uvutils.fits_gethduaxis(totqualhdu, 4, strict_fits=strict_fits)
            except(KeyError):
                if not strict_fits:
                    warnings.warn('{file} appears to be an old calfits format '
                                  'which does not fully conform to the FITS standard. '
                                  'Setting default values now, set strict_fits=True '
                                  'to error rather than warn on this problem, '
                                  'rewrite this file with write_calfits to ensure '
                                  'FITS compliance.'.format(file=filename))
                    spw_array = np.array([0])
                else:
                    raise
            if not np.allclose(spw_array, self.spw_array):
                raise ValueError('Spectral window values are different in TOTQLTY HDU than in primary HDU')

            freq_array = uvutils.fits_gethduaxis(totqualhdu, 3, strict_fits=strict_fits)
            freq_array.shape = (self.Nspws,) + freq_array.shape
            if not np.allclose(freq_array, self.freq_array,
                               rtol=self._freq_array.tols[0],
                               atol=self._freq_array.tols[0]):
                raise ValueError('Frequency values are different in TOTQLTY HDU than in primary HDU')

            time_array = uvutils.fits_gethduaxis(totqualhdu, 2, strict_fits=strict_fits)
            if not np.allclose(time_array, self.time_array,
                               rtol=self._time_array.tols[0],
                               atol=self._time_array.tols[0]):
                raise ValueError('Time values are different in TOTQLTY HDU than in primary HDU')

            jones_array = uvutils.fits_gethduaxis(totqualhdu, 1, strict_fits=strict_fits)
            if not np.allclose(jones_array, self.jones_array,
                               rtol=self._jones_array.tols[0],
                               atol=self._jones_array.tols[0]):
                raise ValueError('Jones values are different in TOTQLTY HDU than in primary HDU')

        except KeyError:
            self.total_quality_array = None

        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)
