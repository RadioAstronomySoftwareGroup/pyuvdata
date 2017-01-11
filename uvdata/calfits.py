from astropy.io import fits
import numpy as np
from .cal import UVCal
import datetime


class CALFITS(UVCal):
    """
    Defines a calfits-specific class for reading and writing uvfits files.
    """

    uvfits_required_extra = []

    def write_uvfits(self, filename, spoof_nonessential=False,
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
            self.check(run_sanity_check=True)

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
        prihdr['DATE'] = today
        prihdr['ORIGIN'] = 'blank'
        prihdr['HASH'] = 'blank'
        prihdr['GNCONVEN'] = self.gain_convention
        prihdr['NTIMES'] = self.Ntimes
        prihdr['NFREQS'] = self.Nfreqs
        prihdr['NANTSDAT'] = self.Nants_data
        prihdr['NPOLS'] = self.Npols

        prihdr['NANTSTEL'] = self.Nants_telescope
        prihdr['HISTORY'] = self.history
        prihdr['NSPWS'] = self.Nspws
        prihdr['XORIENT'] = self.x_orientation

        prihdu = fits.PrimaryHDU(header=prihdr)
        colnam = fits.Column(name='ANTNAME', format='A10',
                             array=self.antenna_names)
        colnum = fits.Column(name='ANTINDEX', format='I',
                             array=self.antenna_numbers)
        colf = fits.Column(name='FREQ', format='D',
                           array=self.freq_array)
        colp = fits.Column(name='POL', format='A4',
                           array=self.polarization_array)
        colt = fits.Column(name='TIME', format='D',
                           array=self.time_array)
        if self.gain_array:
            coldat = fits.Column(name='GAIN', format='M',
                                 array=self.gain_array)
            colflg = fits.Column(name='FLAG', format='L',
                                 array=self.flag_array)
            colqual = fits.Column(name='QUALITY', format='D',
                                  array=self.quality_array)
        elif self.delay_array:
            coldat = fits.Column(name='DELAY', format='D',
                                 array=self.delay_array)
            colflg = fits.Column(name='FLAG', format='L',
                                 array=self.flag_array)
            colqual = fits.Column(name='QUALITY', format='D'
                                  array=self.quality_array)


        cols = fits.ColDefs([colnam, colnum, colf, colp,
                             colt, coldat, colflg, colqual])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        hdulist = fits.HDUList([prihdu, tbhdu])
        hdulist.writeto(filename)

    def read_uvfits(self, filename):
        F = fits.open(filename)
        D = F[1]
        hdr = F[0].header.copy()

        self.Nfreqs = hdr['NFREQS']
        self.Npols = hdr['NPOLS']
        self.Ntimes = hdr['NTIMES']
        self.history = hdr['HISTORY']
        self.Nspws = hdr['NSPWS']
        self.Nants_data = hdr['NANTSDAT']
        self.antenna_names = np.sort(np.unique(D.data['ANTNAME']))
        self.antenna_numbers = np.sort(np.unique(D.data['ANTINDEX']))
        self.Nants_telescope = hdr['NANTSTEL']

        ptypes = {'Nfreqs': self.Nfreqs,
                  'Npols': self.Npols,
                  'Ntimes': self.Ntimes,
                  'Nants_data': self.Nants_data}

        # import IPython; IPython.embed()
        self.freq_array = np.sort(np.unique(D.data['FREQ']))
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
        else:
            raise("No data to load. Aborting.")

        rs = [ptypes[i] for i in self._flag_array.form]
        self.flag_array = D.data['FLAG'].reshape(rs)
        rs = [ptypes[i] for i in self._quality_array.form]
        self.quality_array = D.data['QUALITY'].reshape(rs)
