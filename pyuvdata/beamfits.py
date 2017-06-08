"""Class for reading and writing beamfits files."""
import numpy as np
from astropy.io import fits
from uvbeam import UVBeam


class BeamFITS(UVBeam):
    """
    Defines a beamfits-specific subclass of UVBeam for reading and writing beamfits files.
    This class should not be interacted with directly, instead use the read_beamfits
    and write_beamfits methods on the UVBeam class.

    """

    def read_beamfits(self, filename, run_check=True,
                      run_check_acceptability=True):
        """
        Read the data from a beamfits file.

        Args:
            filename: The beamfits file to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptability of the values of
                required parameters after reading in the file. Default is True.
        """
        F = fits.open(filename)
        primary_hdu = F[0]
        primary_header = primary_hdu.header.copy()
        hdunames = uvutils.fits_indexhdus(F)  # find the rest of the tables

        self.telescope_name = primary_header.pop['TELESCOP']
        self.feed_name = primary_header.pop['FEED']
        self.feed_version = primary_header.pop['FEEDVER']
        self.model_name = primary_header.pop['MODEL']
        self.model_version = primary_header.pop['MODELVER']

        # shapes
        self.Nfreqs = primary_header.pop['NAXIS2']
        self.Nspws = primary_header.pop['NAXIS3']
        self.Npixels = primary_header.pop['NAXIS4']
        self.Naxes = primary_header.pop['NAXIS5']
        self.Nfeeds = primary_header.pop['NAXIS6']

        self.coordinate_system = primary_header.pop['COORDSYS']
        self.feed_array = primary_header.pop['FEEDLIST']

        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

    def write_beamfits(self, filename, run_check=True,
                       run_check_acceptability=True, clobber=False):
        """
        Write the data to a beamfits file.

        Args:
            filename: The beamfits file to write to.
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
                                 'because of a select operation). The beamfits format '
                                 'does not support unevenly spaced frequencies.')
        else:
            freq_spacing = 1

        primary_header = fits.Header()

        # Conforming to fits format
        primary_header['SIMPLE'] = True
        primary_header['BITPIX'] = 32

        # metadata
        primary_header['TELESCOP'] = self.telescope_name
        primary_header['FEED'] = self.feed_name
        primary_header['FEEDVER'] = self.feed_version
        primary_header['MODEL'] = self.model_name
        primary_header['MODELVER'] = self.model_version

        primary_header['COORDSYS'] = self.coordinate_system
        primary_header['FEEDLIST'] = '[' + ', '.join(self.feed_array) + ']'

        # set up complex axis
        primary_header['CTYPE1'] = ('COMPLEX', 'real, imaginary')
        primary_header['CUNIT1'] = 'Integer'
        primary_header['CRVAL1'] = 1
        primary_header['CRPIX1'] = 1
        primary_header['CDELT1'] = 1

        # set up frequency axis
        primary_header['CTYPE2'] = 'FREQ'
        primary_header['CUNIT2'] = ('Hz')
        primary_header['CRVAL2'] = self.freq_array[0, 0]
        primary_header['CRPIX2'] = 1.0
        primary_header['CDELT2'] = freq_spacing

        # set up spw axis
        primary_header['CTYPE3'] = ('IF', 'Spectral window axis')
        primary_header['CUNIT3'] = 'Integer'
        primary_header['CRVAL3'] = 1.0
        primary_header['CRPIX3'] = 1.0
        primary_header['CDELT3'] = 1.0

        # set up pixel axis
        primary_header['CTYPE4'] = ('PIXIND', 'pixel: index into pixel_location_array.')
        primary_header['CUNIT4'] = 'Integer'
        primary_header['CRVAL4'] = 1
        primary_header['CRPIX4'] = 1
        primary_header['CDELT4'] = 1

        # set up basis vector axis
        primary_header['CTYPE5'] = ('VECIND', 'Basis vector: index into basis_vector_array.')
        primary_header['CUNIT5'] = 'Integer'
        primary_header['CRVAL5'] = 1
        primary_header['CRPIX5'] = 1
        primary_header['CDELT5'] = 1

        # set up feed axis
        primary_header['CTYPE6'] = ('FEEDIND', 'Feed: index into "FEEDLIST".')
        primary_header['CUNIT6'] = 'Integer'
        primary_header['CRVAL6'] = 1
        primary_header['CRPIX6'] = 1
        primary_header['CDELT6'] = 1

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.iteritems():
            # header keywords have to be 8 characters or less
            keyword = key[:8].upper()
            # print "keyword=-value-", keyword+'='+'-'+str(value)+'-'
            if keyword == 'COMMENT':
                for line in value.splitlines():
                    primary_header.add_comment(line)
            else:
                primary_header[keyword] = value

        for line in self.history.splitlines():
            primary_header.add_history(line)

        primary_data = np.concatenate([self.efield_array.real[:, :, :, :, :, np.newaxis],
                                       self.efield_array.imag[:, :, :, :, :, np.newaxis]],
                                      axis=-1)

        primary_hdu = fits.PrimaryHDU(data=primary_data, header=primary_header)

        second_header = fits.Header()
        second_header['EXTNAME'] = 'COORDS'
        second_header['COORLIST'] = '[' + ', '.self.coordinate_system_dict[self.coordinate_system]['naxes'] + ']'

        # set up dummy array axis
        second_header['CTYPE1'] = ('ARRAYNUM', 'array order is pixel locations, basis vectors')
        second_header['CUNIT1'] = 'Integer'
        second_header['CRVAL1'] = 1
        second_header['CRPIX1'] = 1
        second_header['CDELT1'] = 1

        # set up pixel axis
        second_header['CTYPE2'] = ('PIXIND', 'pixel index')
        second_header['CUNIT2'] = 'Integer'
        second_header['CRVAL2'] = 1
        second_header['CRPIX2'] = 1
        second_header['CDELT2'] = 1

        # set up coordinate system axis
        second_header['CTYPE3'] = ('COORDIND', 'Coordinates: index into COORLIST.')
        second_header['CUNIT3'] = 'Integer'
        second_header['CRVAL3'] = 1
        second_header['CRPIX3'] = 1
        second_header['CDELT3'] = 1

        second_data = np.concatenate([self.pixel_location_array[:, :, np.newaxis],
                                      self.basis_vector_array[:, :, np.newaxis]],
                                     axis=-1)
        second_hdu = fits.ImageHDU(data=second_data, header=second_header)

        hdulist = fits.HDUList([primary_hdu, second_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(filename, clobber=clobber)
        else:
            hdulist.writeto(filename, overwrite=clobber)
