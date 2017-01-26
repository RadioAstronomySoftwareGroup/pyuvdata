from astropy.io import fits
import numpy as np
from .cal import UVCal
import datetime


class CALFITS(UVCal):
    """
    Defines a calfits-specific class for reading and writing uvfits files.
    """

    uvfits_required_extra = []

    def write_calfits(self, filename, spoof_nonessential=False,
                     run_check=True, run_sanity_check=True):
        """
        Write the data to a uvfits file.

        Args:
            filename: The uvfits file to write to.
            spoof_nonessential: Option to spoof the values of optional
                UVParameters that are not set but are required for uvfits files.
                Default is False.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_sanity_check: Option to sanity check the values of
                required parameters before writing the file. Default is True.

        """
        if run_check:
            self.check(run_sanity_check=run_sanity_check)

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

        # Need to add in switch for gain/delay.
        # This is first draft of writing to FITS.
        today = datetime.date.today().strftime("Date: %d, %b %Y")
        prihdr = fits.Header()
        #Conforming to fits format
        prihdr['SIMPLE'] = True
        prihdr['BITPIX'] = 32
        prihdr['NAXIS'] = 5
        #prihdr['NAXIS1'] = (self.Nants_data, 'Number and antennas')
        prihdr['TELESCOP'] = ('HERA', 'Telescope of calibration')
        prihdr['OBSERVER'] = 'Observer'
        prihdr['DATE'] = today
        prihdr['CALPIPE'] = ('Omnical', 'Calibration pipeline')
        prihdr['ORIGIN'] = ('origin', 'git origin of pipeline')
        prihdr['HASH'] = ('git hash', 'git hash number')
        prihdr['GNCONVEN'] = self.gain_convention
        prihdr['NTIMES'] = self.Ntimes
        prihdr['NFREQS'] = self.Nfreqs
        prihdr['NANTSDAT'] = self.Nants_data
        prihdr['NPOLS'] = self.Npols

        prihdr['NANTSTEL'] = self.Nants_telescope
        prihdr['HISTORY'] = self.history
        prihdr['NSPWS'] = self.Nspws
        prihdr['XORIENT'] = self.x_orientation

        if np.all(self.gain_array):
            # Set header variable for gain.
            prihdr['CTYPE4'] = ('FREQS', 'Frequency.')
            prihdr['CUNIT4'] = ('GHz', 'Units of frequecy.')
            prihdr['CRVAL4'] = self.freq_array[0][0]
            prihdr['CDELT4'] = self.freq_array[0][1] - self.freq_array[0][0]

            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = ('Integer', 'Number of image arrays. Increment.')
            prihdr['CRVAL1'] = (4, 'Number of image arryays.')
            prihdr['CDELT1'] = 1

            pridata = np.concatenate([self.gain_array.real[:, :, :, :, np.newaxis],
                                      self.gain_array.imag[:, :, :, :, np.newaxis],
                                      self.flag_array[:, :, :, :, np.newaxis],
                                      self.quality_array[:, :, :, :, np.newaxis]],
                                     axis=-1)

        elif np.all(self.delay_array):
            # Set header variable for gain.
            prihdr['CTYPE4'] = ('FREQS', 'Valid frequencies to apply delay.')
            prihdr['CUNIT4'] = ('GHz', 'Units of frequecy.')
            prihdr['CRVAL4'] = self.freq_array[0][0]
            prihdr['CDELT4'] = self.freq_array[0][1] - self.freq_array[0][0]
            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = ('Integer', 'Number of image arrays. Value.')
            prihdr['CRVAL1'] = (3, 'Number of image arrays.')
            prihdr['CDELT1'] = 1

            pridata = np.concatenate([self.delay_array[:, :, :, :, np.newaxis],
                                      self.flag_array[:, :, :, :, np.newaxis],
                                      self.quality_array[:, :, :, :, np.newaxis]],
                                     axis=-1)

        # header ctypes for NAXIS
        prihdr['CTYPE5'] = ('ANTS', 'Antenna numbering.')
        prihdr['CUNIT5'] = 'Integer'
        prihdr['CRVAL5'] = 0
        prihdr['CDELT5'] = -1

        prihdr['CTYPE3'] = ('TIME', 'Time axis.')
        prihdr['CUNIT3'] = ('JD', 'Time in julian date format')
        prihdr['CRVAL3'] = self.time_array[0]
        prihdr['CDELT3'] = self.time_array[1] - self.time_array[0]

        prihdr['CTYPE2'] = ('POLS', 'Polarization array')
        prihdr['CUNIT2'] = ('Integer', 'representative integer for polarization.')
        prihdr['CRVAL2'] = self.polarization_array[0] #always start with xx data etc.
        prihdr['CDELT2'] = 1*np.sign(self.polarization_array[0])

        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)

        col1 = fits.Column(name='ANTNAME', format='8A',
                           array=self.antenna_names)
        col2 = fits.Column(name='ANTINDEX', format='D',
                           array=self.antenna_numbers)
        cols = fits.ColDefs([col1, col2])
        ant_hdu = fits.BinTableHDU.from_columns(cols)

        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])
        hdulist.writeto(filename)

    def read_calfits(self, filename, run_check=True, run_sanity_check=True):
        F = fits.open(filename)
        data = F[0].data
        hdr = F[0].header.copy()

        antdata = F[1].data
        self.antenna_names = map(str, antdata['ANTNAME'])
        self.antenna_numbers = map(int, antdata['ANTINDEX'])
        self.Nfreqs = hdr['NFREQS']
        self.Npols = hdr['NPOLS']
        self.Ntimes = hdr['NTIMES']
        self.history = str(hdr.get('HISTORY', ''))
        self.Nspws = hdr['NSPWS']
        self.Nants_data = hdr['NANTSDAT']
        self.Nants_telescope = hdr['NANTSTEL']
        self.gain_convention = hdr['GNCONVEN']
        self.x_orientation = hdr['XORIENT']

        # get data
        if data.shape[-1] == 4:
            self.gain_array = data[:, :, :, :, 0] + 1j*data[:, :, :, :, 1]
            self.flag_array = np.array(data[:, :, :, :, 2], dtype=np.bool)
            self.quality_array = data[:, :, :, :, 3]
            self.set_gain()
        if data.shape[-1] == 3:
            self.delay_array = data[:, :, :, : 0]
            self.flag_array = np.array(data[:, :, :, :, 1], dtype=np.bool)
            self.quality_array = data[:, :, :, :, 2]
            self.set_delay()

        # generate frequency, polarization, and time array.
        self.freq_array = np.arange(self.Nfreqs).reshape(1,-1)*hdr['CDELT4'] + hdr['CRVAL4']
        self.polarization_array = np.arange(self.Npols)*hdr['CDELT2'] + hdr['CRVAL2']
        self.time_array = np.arange(self.Ntimes)*hdr['CDELT3'] + hdr['CRVAL3']

        if run_check:
            self.check(run_sanity_check=run_sanity_check)
