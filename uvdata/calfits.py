from .cal import UVCal


class CALFITS(UVCal):
    """
    Defines a calfits-specific class for reading and writing uvfits files.
    """

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
        prihdu = fits.PrimaryHDU(header=prihdr)
        colnam = fits.Column(name='ANT NAME', format='A10',
                             array=self.antenna_names)
        colnum = fits.Column(name='ANT INDEX', format='I',
                             array=self.antenna_numbers)
        colf = fits.Column(name='FREQ (MHZ)', format='E',
                           array=self.freq_array)
        colp = fits.Column(name='POL', format='A4',
                           array=self.polarization_array)
        colt = fits.Column(name='TIME (JD)', format='D', array=self.time_array)
        coldat = fits.Column(name='GAIN', format='M', array=self.gain_array)
        colflg = fits.Column(name='FLAG', format='L', array=self.flag_array)
        cols = fits.ColDefs([colnam, colnum, colf, colp, colt, coldat, colflg])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        hdulist = fits.HDUList([prihdu, tbhdu])
        hdulist.writeto(outfn)
