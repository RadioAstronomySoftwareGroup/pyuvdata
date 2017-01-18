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
        prihdr['NAXIS1'] = (self.Nants_data, 'Number and antennas')

        prihdr['NAXIS3'] = (self.Ntimes, 'Number of time samples')
        prihdr['NAXIS4'] = (self.Npols, 'Number of polarizations')

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
        prihdr['END']

        if np.all(self.gain_array):
            coldat = fits.Column(name='GAIN', format='M',
                                 array=self.gain_array)
            colflg = fits.Column(name='FLAG', format='L',
                                 array=self.flag_array)
            colqual = fits.Column(name='QUALITY', format='D',
                                  array=self.quality_array)
            # Set header variable for gain.
            prihdr['NAXIS2'] = (self.Nfreqs, 'Number of frequency channels')
            prihdr['CTYPE2'] = ('FREQS', 'Frequency.')
            prihdr['CUNIT2'] = ('GHz', 'Unit of frequecy.')
            prihdr['CRVAL2'] = self.freq_array[0]
            prihdr['CDELT2'] = self.freq_array[1] - self.freq_array[0]
            # set the last axis for number of arrays.
            prihdr['NAXIS5'] = (4, 'Number of data arrays:gain.real, \
                                    gain.imag, flag, quality')
            prihdr['CTYPE5'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT5'] = ('Integer', 'Number of image arrays. Increment.')
            prihdr['CRVAL5'] = 0
            prihdr['CDELT5'] = 1
        elif np.all(self.delay_array):
            coldat = fits.Column(name='DELAY', format='D',
                                 array=self.delay_array)
            colflg = fits.Column(name='FLAG', format='L',
                                 array=self.flag_array)
            colqual = fits.Column(name='QUALITY', format='D',
                                  array=self.quality_array)
            # Set header variable for gain.
            prihdr['NAXIS2'] = (1, 'Number of delay solutions.')
            prihdr['CTYPE2'] = ('DELAYS', 'Delay number.')
            prihdr['CUNIT2'] = ('ns', 'Nanosecond units.')
            prihdr['CRVAL2'] = 0
            prihdr['CDELT2'] = 0
            # set the last axis for number of arrays.
            prihdr['NAXIS5'] = (3, 'Number of data arrays:delay,flag, quality')
            prihdr['CTYPE5'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT5'] = ('Integer', 'Number of image arrays. Increment.')
            prihdr['CRVAL5'] = 0
            prihdr['CDELT5'] = 1

        #header ctypes for NAXIS
        prihdr['CTYPE1'] = ('ANTS', 'Antenna numbering.')
        prihdr['CUNIT1'] = 'Integer'
        prihdr['CRVAL1'] = 0
        prihdr['CDELT1'] = -1

        prihdr['CTYPE3'] = ('TIME', 'Time axis.')
        prihdr['CUNIT3'] = ('JD', 'Time in julian date format')
        prihdr['CRVAL3'] = time_array[0]
        prihdr['CDELT3'] = time_array[1] - time_array[0]

        prihdr['CTYPE4'] = ('POLS', 'Polarization array')
        prihdr['CUNIT4'] = ('Integer', 'representative integer for polarization.')
        prihdr['CRVAL4'] = -5
        prihdr['CDELT4'] = 1
        colnam = fits.Column(name='ANTNAME', format='A10',
                             array=self.antenna_names)
        colnum = fits.Column(name='ANTINDEX', format='I',
                             array=self.antenna_numbers)
        colf = fits.Column(name='FREQ', format='D',
                           array=self.freq_array)
        colp = fits.Column(name='POL', format='I',
                           array=self.polarization_array)
        colt = fits.Column(name='TIME', format='D',
                           array=self.time_array)

        pridata = np.concatenate([self.gain_array.real,
                                  self.gain_array.imag,
                                  self.flag_array,
                                  self.quality_array])
        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)

        cols = fits.ColDefs([colnam, colnum, colf, colp,
                             colt, coldat, colflg, colqual])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        hdulist = fits.HDUList([prihdu, tbhdu])
        hdulist.writeto(filename)

    def read_calfits(self, filename, run_check=True, run_sanity_check=True):
        F = fits.open(filename)
        D = F[1]
        hdr = F[0].header.copy()

        self.Nfreqs = hdr['NFREQS']
        self.Npols = hdr['NPOLS']
        self.Ntimes = hdr['NTIMES']
        self.history = str(hdr.get('HISTORY', ''))
        self.Nspws = hdr['NSPWS']
        self.Nants_data = hdr['NANTSDAT']
        self.antenna_names = np.sort(np.unique(D.data['ANTNAME']))
        self.antenna_numbers = np.sort(np.unique(D.data['ANTINDEX']))
        self.Nants_telescope = hdr['NANTSTEL']
        self.gain_convention = hdr['GNCONVEN']
        self.x_orientation = hdr['XORIENT']

        ptypes = {'Nfreqs': self.Nfreqs,
                  'Npols': self.Npols,
                  'Ntimes': self.Ntimes,
                  'Nants_data': self.Nants_data}

        self.freq_array = np.sort(np.unique(D.data['FREQ'])).reshape(1,-1)
        self.polarization_array = np.sort(np.unique(D.data['POL']))
        self.time_array = np.sort(np.unique(D.data['TIME']))
        try:
            rs = [ptypes[i] for i in self._gain_array.form]
            self.gain_array = D.data['GAIN'].reshape(rs)
            self.set_gain()
        except(KeyError):
            rs = [ptypes[i] for i in self._delay_array.form]
            self.delay_array = D.data['DELAY'].reshape(rs)
            self.set_delay()
        rs = [ptypes[i] for i in self._flag_array.form]
        self.flag_array = D.data['FLAG'].reshape(rs)
        rs = [ptypes[i] for i in self._quality_array.form]
        self.quality_array = D.data['QUALITY'].reshape(rs)

        if run_check:
            self.check(run_sanity_check=run_sanity_check)
