import astropy
from astropy.io import fits
import numpy as np
from uvcal import UVCal
import datetime


class CALFITS(UVCal):
    """
    Defines a calfits-specific class for reading and writing uvfits files.
    """

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

    def write_calfits(self, filename, spoof_nonessential=False,
                      run_check=True, run_check_acceptability=True, clobber=False):
        """
        Write the data to a uvfits file.

        Args:
            filename: The uvfits file to write to.
            spoof_nonessential: Option to spoof the values of optional
                UVParameters that are not set but are required for uvfits files.
                Default is False.
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
            freq_spacing = freq_spacing[0]
        else:
            freq_spacing = self.channel_width  # or should this be the length of the frequency range?

        if self.Ntimes > 1:
            time_spacing = np.diff(self.time_array)
            if not np.isclose(np.min(time_spacing), np.max(time_spacing),
                              rtol=self._time_array.tols[0], atol=self._time_array.tols[1]):
                raise ValueError('The times are not evenly spaced (probably '
                                 'because of a select operation). The calfits format '
                                 'does not support unevenly spaced times.')
            time_spacing = time_spacing[0]
        else:
            time_spacing = self.integration_time  # or should this be the length of the time range?

        if self.Njones > 1:
            jones_spacing = np.diff(self.jones_array)
            if np.min(jones_spacing) < np.max(jones_spacing):
                raise ValueError('The jones values are not evenly spaced.'
                                 'The calibration fits file format does not'
                                 ' support unevenly spaced polarizations.')

        today = datetime.date.today().strftime("Date: %d, %b %Y")
        prihdr = fits.Header()
        if self.cal_type != 'gain':
            sechdr = fits.Header()
            sechdr['EXTNAME'] = 'FLAGS'
        # Conforming to fits format
        prihdr['SIMPLE'] = True
        prihdr['BITPIX'] = 32
        prihdr['NAXIS'] = 5
        prihdr['TELESCOP'] = self.telescope_name
        prihdr['GNCONVEN'] = self.gain_convention
        prihdr['NTIMES'] = self.Ntimes
        prihdr['NFREQS'] = self.Nfreqs
        prihdr['NANTSDAT'] = self.Nants_data
        prihdr['NJONES'] = self.Njones
        prihdr['CALTYPE'] = self.cal_type
        prihdr['INTTIME'] = self.integration_time
        prihdr['CHWIDTH'] = self.channel_width
        prihdr['NANTSTEL'] = self.Nants_telescope
        prihdr['NSPWS'] = self.Nspws
        prihdr['XORIENT'] = self.x_orientation
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
            prihdr['CUNIT4'] = ('Hz', 'Units of frequecy.')
            prihdr['CRVAL4'] = self.freq_array[0][0]
            prihdr['CDELT4'] = freq_spacing

            # Nspws axis: number of spectral windows
            prihdr['CTYPE5'] = ('NSPWS', 'Number of spectral windows.')
            prihdr['CUNIT5'] = ('Integer', 'Number of windows. Increment.')
            prihdr['CRVAL5'] = 0
            prihdr['CDELT5'] = 1

            # ANTAXIS axis number differs between delay and gain because there's no frequency axis in delay
            prihdr['CTYPE6'] = ('ANTAXIS', 'See ANTARR in ANTENNA extension for values.')
            prihdr['CUNIT6'] = 'Integer'
            prihdr['CRVAL6'] = 0
            prihdr['CDELT6'] = -1

            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = ('Integer', 'Number of image arrays. Increment.')
            prihdr['CDELT1'] = 1
            if self.input_flag_array is not None:
                prihdr['CRVAL1'] = (5, 'Number of image arrays.')
                pridata = np.concatenate([self.gain_array.real[:, :, :, :, :, np.newaxis],
                                          self.gain_array.imag[:, :, :, :, :, np.newaxis],
                                          self.flag_array[:, :, :, :, :, np.newaxis],
                                          self.input_flag_array[:, :, :, :, :, np.newaxis],
                                          self.quality_array[:, :, :, :, :, np.newaxis]],
                                         axis=-1)
            else:
                prihdr['CRVAL1'] = (4, 'Number of image arrays.')
                pridata = np.concatenate([self.gain_array.real[:, :, :, :, :, np.newaxis],
                                          self.gain_array.imag[:, :, :, :, :, np.newaxis],
                                          self.flag_array[:, :, :, :, :, np.newaxis],
                                          self.quality_array[:, :, :, :, :, np.newaxis]],
                                         axis=-1)

        if self.cal_type == 'delay':
            # Set header variable for delay.

            # ANTAXIS axis number differs between delay and gain because there's no frequency axis in delay
            prihdr['CTYPE5'] = ('ANTAXIS', 'See ANTARR in ANTENNA extension for values.')
            prihdr['CUNIT5'] = 'Integer'
            prihdr['CRVAL5'] = 0
            prihdr['CDELT5'] = -1

            # Nspws axis: number of spectral windows
            prihdr['CTYPE4'] = ('NSPWS', 'Number of spectral windows.')
            prihdr['CUNIT4'] = ('Integer', 'Number of windows. Increment.')
            prihdr['CRVAL4'] = 0
            prihdr['CDELT4'] = 1

            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = ('Integer', 'Number of image arrays. Value.')
            prihdr['CRVAL1'] = (2, 'Number of image arrays.')
            prihdr['CDELT1'] = 1

            pridata = np.concatenate([self.delay_array[:, :, :, :, np.newaxis],
                                      self.quality_array[:, :, :, :, np.newaxis]],
                                     axis=-1)

            # Set headers for the second hdu containing the flags. Only in cal_type=delay.
            sechdr['CTYPE2'] = ('JONES', 'Jones matrix array')
            sechdr['CUNIT2'] = ('Integer', 'representative integer for polarization.')
            sechdr['CRVAL2'] = self.jones_array[0]  # always start with first jones.
            if self.Njones > 1:
                sechdr['CDELT2'] = jones_spacing[0]
            else:
                sechdr['CDELT2'] = -1

            sechdr['CTYPE3'] = ('TIME', 'Time axis.')
            sechdr['CUNIT3'] = ('JD', 'Time in julian date format')
            sechdr['CRVAL3'] = self.time_array[0]
            sechdr['CDELT3'] = time_spacing

            sechdr['CTYPE4'] = ('FREQS', 'Valid frequencies to apply delay.')
            sechdr['CUNIT4'] = ('Hz', 'Units of frequecy.')
            sechdr['CRVAL4'] = self.freq_array[0][0]
            sechdr['CDELT4'] = freq_spacing

            # Nspws axis: number of spectral windows
            prihdr['CTYPE5'] = ('NSPWS', 'Number of spectral windows.')
            prihdr['CUNIT5'] = ('Integer', 'Number of windows. Increment.')
            prihdr['CRVAL5'] = 0
            prihdr['CDELT5'] = 1

            sechdr['CTYPE6'] = ('ANTAXIS', 'See ANTARR in ANTENNA extension for values.')

            if self.input_flag_array is not None:
                secdata = np.concatenate([self.flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis],
                                          self.input_flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis]],
                                         axis=-1)
                sechdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
                sechdr['CUNIT1'] = ('Integer', 'Number of image arrays. Value.')
                sechdr['CRVAL1'] = (2, 'Number of image arrays.')
                sechdr['CDELT1'] = 1

            else:
                secdata = self.flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis]  # Can't be bool
                sechdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
                sechdr['CUNIT1'] = ('Integer', 'Number of image arrays. Value.')
                sechdr['CRVAL1'] = (1, 'Number of image arrays.')
                sechdr['CDELT1'] = 1

        # primary header ctypes for NAXIS [ for both gain and delay cal_type.]
        # Check polarizations.
        prihdr['CTYPE2'] = ('JONES', 'Jones matrix array')
        prihdr['CUNIT2'] = ('Integer', 'representative integer for polarization.')
        prihdr['CRVAL2'] = self.jones_array[0]  # always start with first jones.
        if self.Njones > 1:
            prihdr['CDELT2'] = jones_spacing[0]
        else:
            prihdr['CDELT2'] = -1

        prihdr['CTYPE3'] = ('TIME', 'Time axis.')
        prihdr['CUNIT3'] = ('JD', 'Time in julian date format')
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

        if self.cal_type != 'gain':
            prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)
            sechdu = fits.ImageHDU(data=secdata, header=sechdr)
            hdulist = fits.HDUList([prihdu, ant_hdu, sechdu])

        else:
            prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)
            hdulist = fits.HDUList([prihdu, ant_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(filename, clobber=clobber)
        else:
            hdulist.writeto(filename, overwrite=clobber)

    def read_calfits(self, filename, run_check=True, run_check_acceptability=True):
        F = fits.open(filename)
        data = F[0].data
        hdr = F[0].header.copy()
        hdunames = self._indexhdus(F)

        anthdu = F[hdunames['ANTENNAS']]
        antdata = anthdu.data
        self.antenna_names = map(str, antdata['ANTNAME'])
        self.antenna_numbers = map(int, antdata['ANTINDEX'])
        self.ant_array = np.array(map(int, antdata['ANTARR']))
        if np.min(self.ant_array) < 0:
            # ant_array was shorter than the other columns, so it was padded with -1s.
            # Remove the padded entries.
            self.ant_array = self.ant_array[np.where(self.ant_array >= 0)[0]]

        self.Nfreqs = hdr['NFREQS']
        self.Njones = hdr['NJONES']
        self.Ntimes = hdr['NTIMES']
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
        self.freq_range = map(float, hdr['FRQRANGE'].split(','))
        self.time_range = map(float, hdr['TMERANGE'].split(','))
        self.Nspws = hdr['NSPWS']
        self.Nants_data = hdr['NANTSDAT']
        self.Nants_telescope = hdr['NANTSTEL']
        self.gain_convention = hdr['GNCONVEN']
        self.x_orientation = hdr['XORIENT']
        self.cal_type = hdr['CALTYPE']
        try:
            self.observer = hdr['OBSERVER']
        except:
            pass
        try:
            self.git_origin_cal = hdr['ORIGCAL']
        except:
            pass
        try:
            self.git_hash_cal = hdr['HASHCAL']
        except:
            pass

        # get data.
        if self.cal_type == 'gain':
            self.set_gain()
            self.gain_array = data[:, :, :, :, :, 0] + 1j * data[:, :, :, :, :, 1]
            self.flag_array = data[:, :, :, :, :, 2].astype('bool')
            if hdr['CRVAL1'] == 5:
                self.input_flag_array = data[:, :, :, :, :, 3].astype('bool')
                self.quality_array = data[:, :, :, :, :, 4]
            else:
                self.quality_array = data[:, :, :, :, :, 3]

            # generate frequency array from primary data unit.
            self.freq_array = np.arange(self.Nfreqs).reshape(1, -1) * hdr['CDELT4'] + hdr['CRVAL4']

        if self.cal_type == 'delay':
            self.set_delay()
            self.delay_array = data[:, :, :, :, 0]
            self.quality_array = data[:, :, :, :, 1]
            sechdu = F[hdunames['FLAGS']]
            flag_data = sechdu.data
            flag_hdr = sechdu.header
            if sechdu.header['CRVAL1'] == 2:
                self.flag_array = flag_data[:, :, :, :, :, 0].astype('bool')
                self.input_flag_array = flag_data[:, :, :, :, :, 1]
            else:
                self.flag_array = flag_data[:, :, :, :, :, 0].astype('bool')

            # generate frequency array from flag data unit (no freq axis in primary).
            self.freq_array = np.arange(self.Nfreqs).reshape(1, -1) * flag_hdr['CDELT4'] + flag_hdr['CRVAL4']

        # generate polarization and time array for either cal_type.
        self.jones_array = np.arange(self.Njones) * hdr['CDELT2'] + hdr['CRVAL2']
        self.time_array = np.arange(self.Ntimes) * hdr['CDELT3'] + hdr['CRVAL3']

        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)
