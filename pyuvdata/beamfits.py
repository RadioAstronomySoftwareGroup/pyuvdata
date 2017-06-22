"""Class for reading and writing beamfits files."""
import numpy as np
import astropy
from astropy.io import fits
from uvbeam import UVBeam
import utils as uvutils

# fits axes:
# power:
# 	1) axes1
# 	2) axes2
# 	3) freq
# 	4) stokes (polarization)
# 	5) spw
# 	6) basis vector (can be length 1)
#
# efield:
# 	1) axes1
# 	2) axes2
# 	3) freq
# 	4) feed
# 	5) spw
# 	6) basis vector
# 	7) complex
#
# basisvec hdu (not needed for power):
# 	1) axes1
# 	2) axes2
# 	3) coord components (length 2)
# 	4) basis vector index (length Naxes_vec)


class BeamFITS(UVBeam):
    """
    Defines a beamfits-specific subclass of UVBeam for reading and writing
    regularly gridded beam fits files. This class should not be interacted with
    directly, instead use the read_beamfits and write_beamfits methods on the
    UVBeam class.
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

        # only support simple antenna_types for now. Phased arrays should be supported as well
        self.set_simple()

        self.beam_type = primary_header.pop('BTYPE', None)
        if self.beam_type is not None:
            self.beam_type = self.beam_type.lower()
        else:
            bunit = primary_header.pop('BUNIT', None)
            if bunit is not None and bunit.lower().strip() == 'jy/beam':
                self.beam_type = 'power'

        if self.beam_type == 'intensity':
            self.beam_type = 'power'

        n_dimensions = primary_header.pop('NAXIS')
        ctypes = [primary_header[ctype] for ctype in (key for key in primary_header if 'ctype' in key.lower())]

        if self.beam_type == 'power':
            self.set_power()
            self.data_array = data
            if primary_header.pop('CTYPE4').lower().strip() == 'stokes':
                self.Npols = primary_header.pop('NAXIS4')
            self.polarization_array = np.int32(uvutils.fits_gethduaxis(primary_hdu, 4))
        elif self.beam_type == 'efield':
            self.set_efield()
            if n_dimensions < 7:
                raise (ValueError, 'beam_type is efield and data dimensionality is too low')
            self.data_array = data[0, :, :, :, :, :, :] + 1j * data[1, :, :, :, :, :, :]
            if primary_header.pop('CTYPE4').lower().strip() == 'feedind':
                self.Nfeeds = primary_header.pop('NAXIS4')
            feedlist = primary_header.pop('FEEDLIST', None)
            if feedlist is not None:
                self.feed_array = np.array(feedlist[1:-1].split(', '))
        else:
            raise ValueError('Unknown beam_type: {type}, beam_type should be '
                             '"efield" or "power".'.format(type=self.beam_type))

        self.data_normalization = primary_header.pop('NORMSTD', None)

        self.pixel_coordinate_system = primary_header.pop('COORDSYS', None)
        coord_list = ctypes[0:2]
        if self.pixel_coordinate_system is None:
            for cs, coords in self.coordinate_system_dict.iteritems():
                if coords == coord_list:
                    self.pixel_coordinate_system = cs
        else:
            if coord_list != self.coordinate_system_dict[self.pixel_coordinate_system]:
                raise ValueError('Coordinate axis list does not match coordinate system')

        self.telescope_name = primary_header.pop('TELESCOP')
        self.feed_name = primary_header.pop('FEED', None)
        self.feed_version = primary_header.pop('FEEDVER', None)
        self.model_name = primary_header.pop('MODEL', None)
        self.model_version = primary_header.pop('MODELVER', None)

        # shapes
        self.Naxes1 = primary_header.pop('NAXIS1')
        self.Naxes2 = primary_header.pop('NAXIS2')
        if primary_header.pop('CTYPE3').lower().strip() == 'freq':
            self.Nfreqs = primary_header.pop('NAXIS3')

        if n_dimensions > 4:
            if primary_header.pop('CTYPE5').lower().strip() == 'if':
                self.Nspws = primary_header.pop('NAXIS5', None)
                # subtract 1 to be zero-indexed
                self.spw_array = uvutils.fits_gethduaxis(primary_hdu, 5) - 1

        if n_dimensions > 5:
            if primary_header.pop('CTYPE6').lower().strip() == 'vecind':
                self.Naxes_vec = primary_header.pop('NAXIS6', None)

        if (self.Nspws is None or self.Naxes_vec is None) and self.beam_type == 'power':
            if self.Nspws is None:
                self.Nspws = 1
                self.spw_array = np.array([0])
            if self.Naxes_vec is None:
                self.Naxes_vec = 1

            # add extra empty dimensions to data_array as appropriate
            while len(self.data_array.shape) < 6:
                self.data_array = np.expand_dims(self.data_array, axis=0)

        self.axis1_array = uvutils.fits_gethduaxis(primary_hdu, 1)
        self.axis2_array = uvutils.fits_gethduaxis(primary_hdu, 2)
        self.freq_array = uvutils.fits_gethduaxis(primary_hdu, 3)
        self.freq_array.shape = (self.Nspws,) + self.freq_array.shape

        self.history = str(primary_header.get('HISTORY', ''))
        if not uvutils.check_history_version(self.history, self.pyuvdata_version_str):
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

        if self.beam_type == 'efield':
            # read BASISVEC HDU
            basisvec_hdu = F[hdunames['BASISVEC']]
            self.basis_vector_array = basisvec_hdu.data
            basisvec_header = basisvec_hdu.header

            basisvec_coord_list = [basisvec_header['CTYPE1'], basisvec_header['CTYPE2']]
            basisvec_axis1_array = uvutils.fits_gethduaxis(basisvec_hdu, 1)
            basisvec_axis2_array = uvutils.fits_gethduaxis(basisvec_hdu, 2)
            basisvec_Naxes_vec = basisvec_header['NAXIS4']

            basisvec_cs = basisvec_header['COORDSYS']
            if basisvec_cs != self.pixel_coordinate_system:
                raise ValueError('Pixel coordinate system in BASISVEC HDU does not match primary HDU')

            if basisvec_coord_list != coord_list:
                raise ValueError('Pixel coordinate list in BASISVEC HDU does not match primary HDU')

            if basisvec_Naxes_vec != self.Naxes_vec:
                raise ValueError('Number of vector coordinate axes in BASISVEC HDU does not match primary HDU')

            if not np.all(basisvec_axis1_array == self.axis1_array):
                raise ValueError('First image axis in BASISVEC HDU does not match primary HDU')

            if not np.all(basisvec_axis2_array == self.axis2_array):
                raise ValueError('Second image axis in BASISVEC HDU does not match primary HDU')

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

        if self.Naxes1 > 1:
            axis1_spacing = np.diff(self.axis1_array)
            if not np.isclose(np.min(axis1_spacing), np.max(axis1_spacing),
                              rtol=self._axis1_array.tols[0], atol=self._axis1_array.tols[1]):
                raise ValueError('The pixels are not evenly spaced along first axis. '
                                 'The beamfits format does not support unevenly spaced pixels.')
            axis1_spacing = axis1_spacing[0]
        else:
            axis1_spacing = 1

        if self.Naxes2 > 1:
            axis2_spacing = np.diff(self.axis2_array)
            if not np.isclose(np.min(axis2_spacing), np.max(axis2_spacing),
                              rtol=self._axis2_array.tols[0], atol=self._axis2_array.tols[1]):
                raise ValueError('The pixels are not evenly spaced along second axis. '
                                 'The beamfits format does not support unevenly spaced pixels.')
            axis2_spacing = axis2_spacing[0]
        else:
            axis2_spacing = 1

        primary_header = fits.Header()

        # Conforming to fits format
        primary_header['SIMPLE'] = True
        primary_header['BITPIX'] = 32

        primary_header['BTYPE'] = self.beam_type
        primary_header['NORMSTD'] = self.data_normalization
        primary_header['COORDSYS'] = self.pixel_coordinate_system

        # metadata
        primary_header['TELESCOP'] = self.telescope_name
        primary_header['FEED'] = self.feed_name
        primary_header['FEEDVER'] = self.feed_version
        primary_header['MODEL'] = self.model_name
        primary_header['MODELVER'] = self.model_version

        if self.beam_type == 'efield':
            primary_header['FEEDLIST'] = '[' + ', '.join(self.feed_array) + ']'

        # set up first image axis
        primary_header['CTYPE1'] = (self.coordinate_system_dict[self.pixel_coordinate_system][0])
        primary_header['CRVAL1'] = self.axis1_array[0]
        primary_header['CRPIX1'] = 1
        primary_header['CDELT1'] = axis1_spacing

        # set up second image axis
        primary_header['CTYPE2'] = (self.coordinate_system_dict[self.pixel_coordinate_system][1])
        primary_header['CRVAL2'] = self.axis2_array[0]
        primary_header['CRPIX2'] = 1
        primary_header['CDELT2'] = axis2_spacing

        # set up frequency axis
        primary_header['CTYPE3'] = 'FREQ'
        primary_header['CUNIT3'] = ('Hz')
        primary_header['CRVAL3'] = self.freq_array[0, 0]
        primary_header['CRPIX3'] = 1
        primary_header['CDELT3'] = freq_spacing

        # set up feed or pol axis
        if self.beam_type == "power":
            if self.Npols > 1:
                pol_spacing = np.diff(self.polarization_array)
                if np.min(pol_spacing) < np.max(pol_spacing):
                    raise ValueError('The polarization values are not evenly spaced (probably '
                                     'because of a select operation). The uvfits format '
                                     'does not support unevenly spaced polarizations.')
                pol_spacing = pol_spacing[0]
            else:
                pol_spacing = 1

            primary_header['CTYPE4'] = ('STOKES', 'Polarization integers, see AIPS memo 117')
            primary_header['CRVAL4'] = self.polarization_array[0]
            primary_header['CDELT4'] = pol_spacing

            primary_data = self.data_array
        elif self.beam_type == "efield":
            primary_header['CTYPE4'] = ('FEEDIND', 'Feed: index into "FEEDLIST".')
            primary_header['CRVAL4'] = 1
            primary_header['CDELT4'] = 1

            primary_data = np.concatenate([self.data_array.real[np.newaxis, :, :, :, :, :, :],
                                           self.data_array.imag[np.newaxis, :, :, :, :, :, :]],
                                          axis=0)
        else:
            raise ValueError('Unknown beam_type: {type}, beam_type should be '
                             '"efield" or "power".'.format(type=self.beam_type))
        primary_header['CRPIX4'] = 1

        # set up spw axis
        primary_header['CTYPE5'] = ('IF', 'Spectral window axis')
        primary_header['CUNIT5'] = 'Integer'
        primary_header['CRVAL5'] = 1
        primary_header['CRPIX5'] = 1
        primary_header['CDELT5'] = 1

        # set up basis vector axis
        primary_header['CTYPE6'] = ('VECIND', 'Basis vector index.')
        primary_header['CUNIT6'] = 'Integer'
        primary_header['CRVAL6'] = 1
        primary_header['CRPIX6'] = 1
        primary_header['CDELT6'] = 1

        # set up complex axis
        primary_header['CTYPE7'] = ('COMPLEX', 'real, imaginary')
        primary_header['CRVAL7'] = 1
        primary_header['CRPIX7'] = 1
        primary_header['CDELT7'] = 1

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.iteritems():
            # header keywords have to be 8 characters or less
            keyword = key[:8].upper()
            if keyword == 'COMMENT':
                for line in value.splitlines():
                    primary_header.add_comment(line)
            else:
                primary_header[keyword] = value

        for line in self.history.splitlines():
            primary_header.add_history(line)

        primary_hdu = fits.PrimaryHDU(data=primary_data, header=primary_header)

        basisvec_header = fits.Header()
        basisvec_header['EXTNAME'] = 'BASISVEC'
        basisvec_header['COORDSYS'] = self.pixel_coordinate_system

        # set up first image axis
        basisvec_header['CTYPE1'] = (self.coordinate_system_dict[self.pixel_coordinate_system][0])
        basisvec_header['CRVAL1'] = self.axis1_array[0]
        basisvec_header['CRPIX1'] = 1
        basisvec_header['CDELT1'] = axis1_spacing

        # set up second image axis
        basisvec_header['CTYPE2'] = (self.coordinate_system_dict[self.pixel_coordinate_system][1])
        basisvec_header['CRVAL2'] = self.axis2_array[0]
        basisvec_header['CRPIX2'] = 1
        basisvec_header['CDELT2'] = axis2_spacing

        # set up pixel coordinate system axis (length 2)
        basisvec_header['CTYPE3'] = ('AXISIND', 'Axis index')
        basisvec_header['CUNIT3'] = 'Integer'
        basisvec_header['CRVAL3'] = 1
        basisvec_header['CRPIX3'] = 1
        basisvec_header['CDELT3'] = 1

        # set up vector coordinate system axis (length Naxis_vec)
        basisvec_header['CTYPE4'] = ('VECCOORD', 'Basis vector index')
        basisvec_header['CUNIT4'] = 'Integer'
        basisvec_header['CRVAL4'] = 1
        basisvec_header['CRPIX4'] = 1
        basisvec_header['CDELT4'] = 1

        basisvec_data = self.basis_vector_array
        basisvec_hdu = fits.ImageHDU(data=basisvec_data, header=basisvec_header)

        hdulist = fits.HDUList([primary_hdu, basisvec_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(filename, clobber=clobber)
        else:
            hdulist.writeto(filename, overwrite=clobber)
