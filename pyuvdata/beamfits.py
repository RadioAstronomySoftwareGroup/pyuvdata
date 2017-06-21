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

        self.beam_type = primary_header.pop('BUNIT', None)
        if self.beam_type == 'power':
            self.set_power()
            self.data_array = data
            self.Npols = primary_header.pop('NAXIS4')
            self.polarization_array = uvutils.fits_gethduaxis(primary_hdu, 4)
        elif self.beam_type == 'efield':
            self.set_efield()
            self.data_array = data[0, :, :, :, :, :, :] + 1j * data[1, :, :, :, :, :, :]
            self.Nfeeds = primary_header.pop('NAXIS4')
            self.feed_array = primary_header.pop('FEEDLIST')[1:-1].split(', ')
        else:
            raise ValueError('Unknown beam_type: {type}, beam_type should be '
                             '"efield" or "power".'.format(type=self.beam_type))

        self.data_normalization = primary_header.pop('NORMSTD')
        self.telescope_name = primary_header.pop('TELESCOP')
        self.feed_name = primary_header.pop('FEED')
        self.feed_version = primary_header.pop('FEEDVER')
        self.model_name = primary_header.pop('MODEL')
        self.model_version = primary_header.pop('MODELVER')

        # shapes
        self.Naxes1 = primary_header.pop('NAXIS1')
        self.Naxes2 = primary_header.pop('NAXIS2')
        self.Nfreqs = primary_header.pop('NAXIS3')
        self.Nspws = primary_header.pop('NAXIS5')
        self.Naxes_vec = primary_header.pop('NAXIS6')

        self.axis1_array = uvutils.fits_gethduaxis(primary_hdu, 1)
        self.axis2_array = uvutils.fits_gethduaxis(primary_hdu, 2)
        self.freq_array = uvutils.fits_gethduaxis(primary_hdu, 3)
        self.freq_array.shape = (self.Nspws,) + self.freq_array.shape
        self.spw_array = uvutils.fits_gethduaxis(primary_hdu, 5)

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

        # read BASISVEC HDU
        basisvec_hdu = F[hdunames['BASISVEC']]
        self.basis_vector_array = basisvec_hdu.data
        basisvec_header = basisvec_hdu.header

        coord_list = [basisvec_header['CTYPE1'], basisvec_header['CTYPE2']]
        basisvec_Naxes1 = basisvec_header['NAXIS1']
        basisvec_Naxes2 = basisvec_header['NAXIS2']
        basisvec_Naxes_vec = basisvec_header['NAXIS4']

        self.pixel_coordinate_system = basisvec_header['COORDSYS']
        if basisvec_Naxes_vec != self.Naxes_vec:
            raise ValueError('Number of vector coordinate axes in BASISVEC HDU does not match primary HDU')

        if basisvec_Naxes1 != self.Naxes1:
            raise ValueError('Number of elements in first image axis in BASISVEC HDU does not match primary HDU')

        if basisvec_Naxes2 != self.Naxes2:
            raise ValueError('Number of elements in second image axis in BASISVEC HDU does not match primary HDU')

        if coord_list != self.coordinate_system_dict[self.pixel_coordinate_system]:
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

        primary_header['BUNIT'] = self.beam_type
        primary_header['NORMSTD'] = self.data_normalization

        # metadata
        primary_header['TELESCOP'] = self.telescope_name
        primary_header['FEED'] = self.feed_name
        primary_header['FEEDVER'] = self.feed_version
        primary_header['MODEL'] = self.model_name
        primary_header['MODELVER'] = self.model_version

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
            # print "keyword=-value-", keyword+'='+'-'+str(value)+'-'
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
