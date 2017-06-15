"""Class for reading and writing beamfits files."""
import numpy as np
import astropy
from astropy.io import fits
from uvbeam import UVBeam
import utils as uvutils


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

        data = primary_hdu.data
        self.efield_array = data[:, :, :, :, :, 0] + 1j * data[:, :, :, :, :, 1]

        self.efield_units = primary_header.pop('BUNIT', None)
        self.telescope_name = primary_header.pop('TELESCOP')
        self.feed_name = primary_header.pop('FEED')
        self.feed_version = primary_header.pop('FEEDVER')
        self.model_name = primary_header.pop('MODEL')
        self.model_version = primary_header.pop('MODELVER')

        # shapes
        self.Nfreqs = primary_header.pop('NAXIS2')
        self.Nspws = primary_header.pop('NAXIS3')
        self.Npixels = primary_header.pop('NAXIS4')
        self.Naxes_vec = primary_header.pop('NAXIS5')
        self.Nfeeds = primary_header.pop('NAXIS6')

        self.feed_array = primary_header.pop('FEEDLIST')[1:-1].split(', ')

        self.freq_array = uvutils.fits_gethduaxis(primary_hdu, 2)
        self.freq_array.shape = (self.Nspws,) + self.freq_array.shape
        self.spw_array = uvutils.fits_gethduaxis(primary_hdu, 3)

        self.history = str(primary_header.get('HISTORY', ''))
        if self.pyuvdata_version_str not in self.history.replace('\n', ''):
            self.history += self.pyuvdata_version_str
        while 'HISTORY' in primary_header.keys():
            primary_header.remove('HISTORY')

        # remove standard FITS header items that are still around
        std_fits_substrings = ['SIMPLE', 'BITPIX', 'EXTEND', 'BLOCKED',
                               'GROUPS', 'PCOUNT', 'BSCALE', 'BZERO', 'NAXIS',
                               'PTYPE', 'PSCAL', 'PZERO', 'CTYPE', 'CRVAL',
                               'CRPIX', 'CDELT', 'CROTA']
        for key in primary_header.keys():
            for sub in std_fits_substrings:
                if key.find(sub) > -1:
                    primary_header.remove(key)

        # find all the remaining header items and keep them as extra_keywords
        for key in primary_header:
            if key == '':
                continue
            if key == 'COMMENT':
                self.extra_keywords[key] = str(primary_header.get(key))
            else:
                self.extra_keywords[key] = primary_header.get(key)

        # read COORDS HDU
        coords_hdu = F[hdunames['COORDS']]
        coords_data = coords_hdu.data
        coords_header = coords_hdu.header

        # pixel_location_array had a shallow dimension added to make it fit
        # with the basis_vector_array
        self.pixel_location_array = coords_data[0, :, :, 0]
        self.basis_vector_array = coords_data[:, :, :, 1]

        coord_list = coords_header['COORLIST'][1:-1].split(', ')
        coords_Npixels = coords_header['NAXIS2']
        self.Naxes_pix = coords_header['NAXIS3']
        coords_Naxes_vec = coords_header['NAXIS4']

        self.pixel_coordinate_system = coords_header['COORDSYS']
        if self.Naxes_pix != self.coordinate_system_dict[self.pixel_coordinate_system]['naxes']:
            raise ValueError('Naxes_pix does not match pixel_coordinate_system')

        if coords_Naxes_vec != self.Naxes_vec:
            raise ValueError('Number of vector coordinate axes in COORDS HDU does not match primary HDU')

        if coords_Npixels != self.Npixels:
            raise ValueError('Number of pixels in COORDS HDU does not match primary HDU')

        if coord_list != self.coordinate_system_dict[self.pixel_coordinate_system]['axis_list']:
            raise ValueError('Coordinate axis list does not match coordinate system')

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
            freq_spacing = freq_spacing[0]
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

        primary_header['BUNIT'] = self.efield_units
        primary_header['BSCALE'] = 1.0
        primary_header['BZERO'] = 0.0

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
        primary_header['CRVAL3'] = 1
        primary_header['CRPIX3'] = 1
        primary_header['CDELT3'] = 1

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

        coords_header = fits.Header()
        coords_header['EXTNAME'] = 'COORDS'
        coords_header['COORDSYS'] = self.pixel_coordinate_system
        coords_header['COORLIST'] = '[' + ', '.join(self.coordinate_system_dict[self.pixel_coordinate_system]['axis_list']) + ']'

        # set up dummy array axis
        coords_header['CTYPE1'] = ('ARRAYNUM', 'array order is pixel locations, basis vectors')
        coords_header['CUNIT1'] = 'Integer'
        coords_header['CRVAL1'] = 1
        coords_header['CRPIX1'] = 1
        coords_header['CDELT1'] = 1

        # set up pixel axis
        coords_header['CTYPE2'] = ('PIXIND', 'pixel index')
        coords_header['CUNIT2'] = 'Integer'
        coords_header['CRVAL2'] = 1
        coords_header['CRPIX2'] = 1
        coords_header['CDELT2'] = 1

        # set up pixel coordinate system axis (length Naxis_pix)
        coords_header['CTYPE3'] = ('PIXCOORD', 'Coordinates: index into PIXCOORD.')
        coords_header['CUNIT3'] = 'Integer'
        coords_header['CRVAL3'] = 1
        coords_header['CRPIX3'] = 1
        coords_header['CDELT3'] = 1

        # set up vector coordinate system axis (length Naxis_vec)
        coords_header['CTYPE4'] = ('VECCOORD', 'Coordinates: index into VECCOORD.')
        coords_header['CUNIT4'] = 'Integer'
        coords_header['CRVAL4'] = 1
        coords_header['CRPIX4'] = 1
        coords_header['CDELT4'] = 1

        # concatenate the pixel_location_array and basis_vector_array,
        # expanding the pixel_location_array to fit
        expanded_locs = np.stack([self.pixel_location_array for _ in range(self.Naxes_vec)], axis=0)

        coords_data = np.concatenate([expanded_locs[:, :, :, np.newaxis],
                                      self.basis_vector_array[:, :, :, np.newaxis]],
                                     axis=-1)
        coords_hdu = fits.ImageHDU(data=coords_data, header=coords_header)

        hdulist = fits.HDUList([primary_hdu, coords_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(filename, clobber=clobber)
        else:
            hdulist.writeto(filename, overwrite=clobber)
