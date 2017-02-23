from astropy.io import fits
import numpy as np
from uvcal import UVCal
import datetime


class CALFITS(UVCal):
    """
    Defines a calfits-specific class for reading and writing uvfits files.
    """

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

        # Need to add in switch for gain/delay.
        # This is first draft of writing to FITS.
        today = datetime.date.today().strftime("Date: %d, %b %Y")
        prihdr = fits.Header()
        #Conforming to fits format
        prihdr['SIMPLE'] = True
        prihdr['BITPIX'] = 32
        prihdr['NAXIS'] = 5
        prihdr['TELESCOP'] = self.telescope_name
        prihdr['GNCONVEN'] = self.gain_convention
        prihdr['NTIMES'] = self.Ntimes
        prihdr['NFREQS'] = self.Nfreqs
        prihdr['NANTSDAT'] = self.Nants_data
        prihdr['NPOLS'] = self.Npols
        prihdr['CALTYPE'] = self.cal_type
        prihdr['INTTIME'] = self.integration_time
        prihdr['CHWIDTH'] = self.channel_width
        prihdr['NANTSTEL'] = self.Nants_telescope
        prihdr['NSPWS'] = self.Nspws
        prihdr['XORIENT'] = self.x_orientation
        for line in self.history.splitlines():
            prihdr.add_history(line)

        for p in self.extra():
            ep = getattr(self, p)
            if ep.form is 'str':
                prihdr['{0}'.format(p.upper().replace('_', '')[:8])] = ep.value
            else:
                continue

        # if self.pipeline: prihdr['CALPIPE'] = self.pipeline
        # if self.observer: prihdr['OBSERVER'] = self.observer
        # if self.git_origin: prihdr['ORIGIN'] = self.git_origin
        # if self.git_hash: prihdr['HASH'] = self.git_hash

        if self.cal_type == 'gain':
            # Set header variable for gain.
            prihdr['CTYPE4'] = ('FREQS', 'Frequency.')
            prihdr['CUNIT4'] = ('Hz', 'Units of frequecy.')
            prihdr['CRVAL4'] = self.freq_array[0][0]
            prihdr['CDELT4'] = self.channel_width

            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = ('Integer', 'Number of image arrays. Increment.')
            prihdr['CRVAL1'] = (3, 'Number of image arryays.')
            prihdr['CDELT1'] = 1

            pridata = np.concatenate([self.gain_array.real[:, :, :, :, np.newaxis],
                                      self.gain_array.imag[:, :, :, :, np.newaxis],
                                      # self.flag_array[:, :, :, :, np.newaxis],
                                      self.quality_array[:, :, :, :, np.newaxis]],
                                     axis=-1)

        if self.cal_type == 'delay':
            # Set header variable for gain.
            prihdr['CTYPE4'] = ('FREQS', 'Valid frequencies to apply delay.')
            prihdr['CUNIT4'] = ('Hz', 'Units of frequecy.')
            prihdr['CRVAL4'] = self.freq_array[0][0]
            prihdr['CDELT4'] = self.channel_width
            # set the last axis for number of arrays.
            prihdr['CTYPE1'] = ('Narrays', 'Number of image arrays.')
            prihdr['CUNIT1'] = ('Integer', 'Number of image arrays. Value.')
            prihdr['CRVAL1'] = (2, 'Number of image arrays.')
            prihdr['CDELT1'] = 1

            pridata = np.concatenate([self.delay_array[:, :, :, :, np.newaxis],
                                      # self.flag_array[:, :, :, :, np.newaxis],
                                      self.quality_array[:, :, :, :, np.newaxis]],
                                     axis=-1)

        if self.cal_type == 'unknown':
            raise ValueError("unknown calibration type. Do not know how to"
                             "store parameters")

        # header ctypes for NAXIS
        prihdr['CTYPE5'] = ('ANTS', 'Antenna numbering.')
        prihdr['CUNIT5'] = 'Integer'
        prihdr['CRVAL5'] = 0
        prihdr['CDELT5'] = -1

        prihdr['CTYPE3'] = ('TIME', 'Time axis.')
        prihdr['CUNIT3'] = ('JD', 'Time in julian date format')
        prihdr['CRVAL3'] = self.time_array[0]
        prihdr['CDELT3'] = self.integration_time

        # more checks for polarization. check ordering.
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

        sechdr = fits.Header()
        secdata = self.flag_array.astype(np.int64) # Can't be bool
        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)
        sechdu = fits.ImageHDU(data=secdata, header=sechdr)
        hdulist = fits.HDUList([prihdu, ant_hdu, sechdu])
        hdulist.writeto(filename, clobber=clobber)

    def read_calfits(self, filename, run_check=True, run_check_acceptability=True):
        F = fits.open(filename)
        data = F[0].data
        hdr = F[0].header.copy()

        antdata = F[1].data
        self.antenna_names = map(str, antdata['ANTNAME'])
        self.antenna_numbers = map(int, antdata['ANTINDEX'])

        self.Nfreqs = hdr['NFREQS']
        self.Npols = hdr['NPOLS']
        self.Ntimes = hdr['NTIMES']
        self.channel_width = hdr['CHWIDTH']
        self.integration_time = hdr['INTTIME']
        self.telescope_name = hdr['TELESCOP']
        self.history = str(hdr.get('HISTORY', ''))
        while 'HISTORY' in hdr.keys():
            hdr.remove('HISTORY')
        self.Nspws = hdr['NSPWS']
        self.Nants_data = hdr['NANTSDAT']
        self.Nants_telescope = hdr['NANTSTEL']
        self.gain_convention = hdr['GNCONVEN']
        self.x_orientation = hdr['XORIENT']
        self.cal_type = hdr['CALTYPE']
        try:
            self.observer = hdr['OBSERVER']
        except: pass
        try:
            self.pipeline = hdr['CALPIPE']
        except: pass
        try:
            self.git_origin = hd['ORIGIN']
        except: pass
        try:
            self.git_hash = hd['HASH']
        except: pass

        # get data. XXX check file type for switch.
        if self.cal_type == 'gain':
            self.set_gain()
            self.gain_array = data[:, :, :, :, 0] + 1j*data[:, :, :, :, 1]
            self.quality_array = data[:, :, :, :, 2]
        if self.cal_type == 'delay':
            self.set_delay()
            self.delay_array = data[:, :, :, :, 0]
            self.quality_array = data[:, :, :, :, 1]

        flag_data = F[2].data
        self.flag_array = np.array(flag_data, dtype=np.bool)
        # generate frequency, polarization, and time array.
        self.freq_array = np.arange(self.Nfreqs).reshape(1,-1)*hdr['CDELT4'] + hdr['CRVAL4']
        self.polarization_array = np.arange(self.Npols)*hdr['CDELT2'] + hdr['CRVAL2']
        self.time_array = np.arange(self.Ntimes)*hdr['CDELT3'] + hdr['CRVAL3']

        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)
