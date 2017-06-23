"""Class for reading and writing beamfits files."""
import numpy as np
import astropy
from astropy.io import fits
from uvbeam import UVBeam
import utils as uvutils

hpx_primary_ax_nums = {'pixel': 1, 'freq': 2, 'feed_pol': 3, 'spw': 4,
                       'basisvec': 5, 'complex': 6}
reg_primary_ax_nums = {'img_ax1': 1, 'img_ax2': 2, 'freq': 3, 'feed_pol': 4,
                       'spw': 5, 'basisvec': 6, 'complex': 7}

hxp_basisvec_ax_nums = {'pixel': 1, 'coord': 2, 'basisvec': 3}
reg_basisvec_ax_nums = {'img_ax1': 1, 'img_ax2': 2, 'coord': 3, 'basisvec': 4}


class BeamFITS(UVBeam):
    """
    Defines a fits-specific subclass of UVBeam for reading and writing
    regularly gridded or healpix beam fits files. This class should not be
    interacted with directly, instead use the read_beamfits and write_beamfits
    methods on the UVBeam class.

    The format defined here for healpix beams is not compatible with true healpix
    formats because it needs to support multiple dimensions (e.g. polarization,
    frequency, efield vectors).
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
        ctypes = [primary_header[ctype] for ctype in (key for key in primary_header
                                                      if 'ctype' in key.lower())]

        self.pixel_coordinate_system = primary_header.pop('COORDSYS', None)
        if self.pixel_coordinate_system is None:
            if ctypes[0] == 'Pix_Ind':
                self.pixel_coordinate_system = 'healpix'
            else:
                for cs, coords in self.coordinate_system_dict.iteritems():
                    if coords == ctypes[0:2]:
                        coord_list = ctypes[0:2]
                        self.pixel_coordinate_system = cs
        else:
            if self.pixel_coordinate_system == 'healpix':
                if ctypes[0] != 'Pix_Ind':
                    raise ValueError('First axis must be "Pix_Ind" for healpix beams')
            else:
                coord_list = ctypes[0:2]
                if coord_list != self.coordinate_system_dict[self.pixel_coordinate_system]:
                    raise ValueError('Coordinate axis list does not match coordinate system')

        if self.pixel_coordinate_system == 'healpix':
            # get pixel values out of HPX_IND extension
            hpx_hdu = F[hdunames['HPX_INDS']]
            self.Npixels = hpx_hdu.header['NAXIS2']
            hpx_data = hpx_hdu.data
            self.pixel_array = hpx_data['hpx_inds']

            ax_nums = hpx_primary_ax_nums
            self.nside = primary_header.pop('NSIDE', None)
            self.ordering = primary_header.pop('ORDERING', None)
            data_Npixels = primary_header.pop('NAXIS' + str(ax_nums['pixel']))
            if data_Npixels != self.Npixels:
                raise ValueError('Number of pixels in HPX_IND extension does '
                                 'not match number of pixels in data array')
        else:
            ax_nums = reg_primary_ax_nums
            self.Naxes1 = primary_header.pop('NAXIS' + str(ax_nums['img_ax1']))
            self.Naxes2 = primary_header.pop('NAXIS' + str(ax_nums['img_ax2']))

            self.axis1_array = uvutils.fits_gethduaxis(primary_hdu, ax_nums['img_ax1'])
            self.axis2_array = uvutils.fits_gethduaxis(primary_hdu, ax_nums['img_ax2'])

        n_efield_dims = max([ax_nums[key] for key in ax_nums])

        if self.beam_type == 'power':
            self.set_power()
            self.data_array = data
            if primary_header.pop('CTYPE' + str(ax_nums['feed_pol'])).lower().strip() == 'stokes':
                self.Npols = primary_header.pop('NAXIS' + str(ax_nums['feed_pol']))
            self.polarization_array = np.int32(uvutils.fits_gethduaxis(primary_hdu,
                                                                       ax_nums['feed_pol']))
        elif self.beam_type == 'efield':
            self.set_efield()
            if n_dimensions < n_efield_dims:
                raise (ValueError, 'beam_type is efield and data dimensionality is too low')
            complex_arrs = np.split(data, 2, axis=0)
            self.data_array = np.squeeze(complex_arrs[0] + 1j * complex_arrs[1], axis=0)
            if primary_header.pop('CTYPE' + str(ax_nums['feed_pol'])).lower().strip() == 'feedind':
                self.Nfeeds = primary_header.pop('NAXIS' + str(ax_nums['feed_pol']))
            feedlist = primary_header.pop('FEEDLIST', None)
            if feedlist is not None:
                self.feed_array = np.array(feedlist[1:-1].split(', '))
        else:
            raise ValueError('Unknown beam_type: {type}, beam_type should be '
                             '"efield" or "power".'.format(type=self.beam_type))

        self.data_normalization = primary_header.pop('NORMSTD', None)

        self.telescope_name = primary_header.pop('TELESCOP')
        self.feed_name = primary_header.pop('FEED', None)
        self.feed_version = primary_header.pop('FEEDVER', None)
        self.model_name = primary_header.pop('MODEL', None)
        self.model_version = primary_header.pop('MODELVER', None)

        # shapes
        if primary_header.pop('CTYPE' + str(ax_nums['freq'])).lower().strip() == 'freq':
            self.Nfreqs = primary_header.pop('NAXIS' + str(ax_nums['freq']))

        if n_dimensions > ax_nums['spw'] - 1:
            if primary_header.pop('CTYPE' + str(ax_nums['spw'])).lower().strip() == 'if':
                self.Nspws = primary_header.pop('NAXIS' + str(ax_nums['spw']), None)
                # subtract 1 to be zero-indexed
                self.spw_array = uvutils.fits_gethduaxis(primary_hdu, ax_nums['spw']) - 1

        if n_dimensions > ax_nums['basisvec'] - 1:
            if primary_header.pop('CTYPE' + str(ax_nums['basisvec'])).lower().strip() == 'vecind':
                self.Naxes_vec = primary_header.pop('NAXIS' + str(ax_nums['basisvec']), None)

        if (self.Nspws is None or self.Naxes_vec is None) and self.beam_type == 'power':
            if self.Nspws is None:
                self.Nspws = 1
                self.spw_array = np.array([0])
            if self.Naxes_vec is None:
                self.Naxes_vec = 1

            # add extra empty dimensions to data_array as appropriate
            while len(self.data_array.shape) < n_efield_dims - 1:
                self.data_array = np.expand_dims(self.data_array, axis=0)

        self.freq_array = uvutils.fits_gethduaxis(primary_hdu, ax_nums['freq'])
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

            if self.pixel_coordinate_system == 'healpix':
                basisvec_ax_nums = hxp_basisvec_ax_nums
                if basisvec_header['CTYPE' + str(basisvec_ax_nums['pixel'])] != 'Pix_Ind':
                    raise ValueError('First axis in BASISVEC HDU must be "Pix_Ind" for healpix beams')

                basisvec_Npixels = basisvec_header.pop('NAXIS' + str(basisvec_ax_nums['pixel']))

                if basisvec_Npixels != self.Npixels:
                    raise ValueError('Number of pixels in BASISVEC HDU does not match '
                                     'primary HDU')
            else:
                basisvec_ax_nums = reg_basisvec_ax_nums
                basisvec_coord_list = [basisvec_header['CTYPE' + str(basisvec_ax_nums['img_ax1'])],
                                       basisvec_header['CTYPE' + str(basisvec_ax_nums['img_ax2'])]]
                basisvec_axis1_array = uvutils.fits_gethduaxis(basisvec_hdu,
                                                               basisvec_ax_nums['img_ax1'])
                basisvec_axis2_array = uvutils.fits_gethduaxis(basisvec_hdu,
                                                               basisvec_ax_nums['img_ax2'])
                if not np.all(basisvec_axis1_array == self.axis1_array):
                    raise ValueError('First image axis in BASISVEC HDU does not match '
                                     'primary HDU')
                if not np.all(basisvec_axis2_array == self.axis2_array):
                    raise ValueError('Second image axis in BASISVEC HDU does not '
                                     'match primary HDU')
                if basisvec_coord_list != coord_list:
                    raise ValueError('Pixel coordinate list in BASISVEC HDU does not '
                                     'match primary HDU')

            basisvec_Naxes_vec = basisvec_header['NAXIS' + str(basisvec_ax_nums['basisvec'])]

            basisvec_cs = basisvec_header['COORDSYS']
            if basisvec_cs != self.pixel_coordinate_system:
                raise ValueError('Pixel coordinate system in BASISVEC HDU does '
                                 'not match primary HDU')

            if basisvec_Naxes_vec != self.Naxes_vec:
                raise ValueError('Number of vector coordinate axes in BASISVEC '
                                 'HDU does not match primary HDU')

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
                              rtol=self._freq_array.tols[0],
                              atol=self._freq_array.tols[1]):
                raise ValueError('The frequencies are not evenly spaced (probably '
                                 'because of a select operation). The beamfits format '
                                 'does not support unevenly spaced frequencies.')
            freq_spacing = freq_spacing[0]
        else:
            freq_spacing = 1

        if self.pixel_coordinate_system == 'healpix':
            ax_nums = hpx_primary_ax_nums
        else:
            ax_nums = reg_primary_ax_nums
            if self.Naxes1 > 1:
                axis1_spacing = np.diff(self.axis1_array)
                if not np.isclose(np.min(axis1_spacing), np.max(axis1_spacing),
                                  rtol=self._axis1_array.tols[0],
                                  atol=self._axis1_array.tols[1]):
                    raise ValueError('The pixels are not evenly spaced along first axis. '
                                     'The beam fits format does not support '
                                     'unevenly spaced pixels.')
                axis1_spacing = axis1_spacing[0]
            else:
                axis1_spacing = 1

            if self.Naxes2 > 1:
                axis2_spacing = np.diff(self.axis2_array)
                if not np.isclose(np.min(axis2_spacing), np.max(axis2_spacing),
                                  rtol=self._axis2_array.tols[0],
                                  atol=self._axis2_array.tols[1]):
                    raise ValueError('The pixels are not evenly spaced along second axis. '
                                     'The beam fits format does not support '
                                     'unevenly spaced pixels.')
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

        if self.pixel_coordinate_system == 'healpix':
            primary_header['NSIDE'] = self.nside
            primary_header['ORDERING'] = self.ordering

            # set up pixel axis
            primary_header['CTYPE' + str(ax_nums['pixel'])] = \
                ('Pix_Ind', 'Index into pixel array in HPX_INDS extension.')
            primary_header['CRVAL' + str(ax_nums['pixel'])] = 1
            primary_header['CRPIX' + str(ax_nums['pixel'])] = 1
            primary_header['CDELT' + str(ax_nums['pixel'])] = 1

        else:
            # set up first image axis
            primary_header['CTYPE' + str(ax_nums['img_ax1'])] = \
                (self.coordinate_system_dict[self.pixel_coordinate_system][0])
            primary_header['CRVAL' + str(ax_nums['img_ax1'])] = self.axis1_array[0]
            primary_header['CRPIX' + str(ax_nums['img_ax1'])] = 1
            primary_header['CDELT' + str(ax_nums['img_ax1'])] = axis1_spacing

            # set up second image axis
            primary_header['CTYPE' + str(ax_nums['img_ax2'])] = \
                (self.coordinate_system_dict[self.pixel_coordinate_system][1])
            primary_header['CRVAL' + str(ax_nums['img_ax2'])] = self.axis2_array[0]
            primary_header['CRPIX' + str(ax_nums['img_ax2'])] = 1
            primary_header['CDELT' + str(ax_nums['img_ax2'])] = axis2_spacing

        # set up frequency axis
        primary_header['CTYPE' + str(ax_nums['freq'])] = 'FREQ'
        primary_header['CUNIT' + str(ax_nums['freq'])] = ('Hz')
        primary_header['CRVAL' + str(ax_nums['freq'])] = self.freq_array[0, 0]
        primary_header['CRPIX' + str(ax_nums['freq'])] = 1
        primary_header['CDELT' + str(ax_nums['freq'])] = freq_spacing

        # set up feed or pol axis
        if self.beam_type == "power":
            if self.Npols > 1:
                pol_spacing = np.diff(self.polarization_array)
                if np.min(pol_spacing) < np.max(pol_spacing):
                    raise ValueError('The polarization values are not evenly '
                                     'spaced (probably because of a select operation). '
                                     'The uvfits format does not support unevenly '
                                     'spaced polarizations.')
                pol_spacing = pol_spacing[0]
            else:
                pol_spacing = 1

            primary_header['CTYPE' + str(ax_nums['feed_pol'])] = \
                ('STOKES', 'Polarization integers, see AIPS memo 117')
            primary_header['CRVAL' + str(ax_nums['feed_pol'])] = self.polarization_array[0]
            primary_header['CDELT' + str(ax_nums['feed_pol'])] = pol_spacing

            primary_data = self.data_array
        elif self.beam_type == "efield":
            primary_header['CTYPE' + str(ax_nums['feed_pol'])] = \
                ('FEEDIND', 'Feed: index into "FEEDLIST".')
            primary_header['CRVAL' + str(ax_nums['feed_pol'])] = 1
            primary_header['CDELT' + str(ax_nums['feed_pol'])] = 1

            np.expand_dims(self.data_array.real, axis=0)
            primary_data = np.concatenate([np.expand_dims(self.data_array.real, axis=0),
                                           np.expand_dims(self.data_array.imag, axis=0)],
                                          axis=0)
        else:
            raise ValueError('Unknown beam_type: {type}, beam_type should be '
                             '"efield" or "power".'.format(type=self.beam_type))
        primary_header['CRPIX' + str(ax_nums['feed_pol'])] = 1

        # set up spw axis
        primary_header['CTYPE' + str(ax_nums['spw'])] = ('IF', 'Spectral window axis')
        primary_header['CUNIT' + str(ax_nums['spw'])] = 'Integer'
        primary_header['CRVAL' + str(ax_nums['spw'])] = 1
        primary_header['CRPIX' + str(ax_nums['spw'])] = 1
        primary_header['CDELT' + str(ax_nums['spw'])] = 1

        # set up basis vector axis
        primary_header['CTYPE' + str(ax_nums['basisvec'])] = ('VECIND', 'Basis vector index.')
        primary_header['CUNIT' + str(ax_nums['basisvec'])] = 'Integer'
        primary_header['CRVAL' + str(ax_nums['basisvec'])] = 1
        primary_header['CRPIX' + str(ax_nums['basisvec'])] = 1
        primary_header['CDELT' + str(ax_nums['basisvec'])] = 1

        if self.beam_type == 'efield':
            # set up complex axis
            primary_header['CTYPE' + str(ax_nums['complex'])] = ('COMPLEX', 'real, imaginary')
            primary_header['CRVAL' + str(ax_nums['complex'])] = 1
            primary_header['CRPIX' + str(ax_nums['complex'])] = 1
            primary_header['CDELT' + str(ax_nums['complex'])] = 1

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

        if self.pixel_coordinate_system == 'healpix':
            basisvec_ax_nums = hxp_basisvec_ax_nums

            # set up pixel axis
            basisvec_header['CTYPE' + str(basisvec_ax_nums['pixel'])] = \
                ('Pix_Ind', 'Index into pixel array in HPX_INDS extension.')
            basisvec_header['CRVAL' + str(basisvec_ax_nums['pixel'])] = 1
            basisvec_header['CRPIX' + str(basisvec_ax_nums['pixel'])] = 1
            basisvec_header['CDELT' + str(basisvec_ax_nums['pixel'])] = 1

        else:
            basisvec_ax_nums = reg_basisvec_ax_nums

            # set up first image axis
            basisvec_header['CTYPE' + str(basisvec_ax_nums['img_ax1'])] = \
                (self.coordinate_system_dict[self.pixel_coordinate_system][0])
            basisvec_header['CRVAL' + str(basisvec_ax_nums['img_ax1'])] = self.axis1_array[0]
            basisvec_header['CRPIX' + str(basisvec_ax_nums['img_ax1'])] = 1
            basisvec_header['CDELT' + str(basisvec_ax_nums['img_ax1'])] = axis1_spacing

            # set up second image axis
            basisvec_header['CTYPE' + str(basisvec_ax_nums['img_ax2'])] = \
                (self.coordinate_system_dict[self.pixel_coordinate_system][1])
            basisvec_header['CRVAL' + str(basisvec_ax_nums['img_ax2'])] = self.axis2_array[0]
            basisvec_header['CRPIX' + str(basisvec_ax_nums['img_ax2'])] = 1
            basisvec_header['CDELT' + str(basisvec_ax_nums['img_ax2'])] = axis2_spacing

        # set up pixel coordinate system axis (length 2)
        basisvec_header['CTYPE' + str(basisvec_ax_nums['coord'])] = ('AXISIND', 'Axis index')
        basisvec_header['CUNIT' + str(basisvec_ax_nums['coord'])] = 'Integer'
        basisvec_header['CRVAL' + str(basisvec_ax_nums['coord'])] = 1
        basisvec_header['CRPIX' + str(basisvec_ax_nums['coord'])] = 1
        basisvec_header['CDELT' + str(basisvec_ax_nums['coord'])] = 1

        # set up vector coordinate system axis (length Naxis_vec)
        basisvec_header['CTYPE' + str(basisvec_ax_nums['basisvec'])] = \
            ('VECCOORD', 'Basis vector index')
        basisvec_header['CUNIT' + str(basisvec_ax_nums['basisvec'])] = 'Integer'
        basisvec_header['CRVAL' + str(basisvec_ax_nums['basisvec'])] = 1
        basisvec_header['CRPIX' + str(basisvec_ax_nums['basisvec'])] = 1
        basisvec_header['CDELT' + str(basisvec_ax_nums['basisvec'])] = 1

        basisvec_data = self.basis_vector_array
        basisvec_hdu = fits.ImageHDU(data=basisvec_data, header=basisvec_header)

        hdulist = fits.HDUList([primary_hdu, basisvec_hdu])

        if self.pixel_coordinate_system == 'healpix':
            # make healpix pixel number column. 'K' format indicates 64-bit integer
            c1 = fits.Column(name='hpx_inds', format='K', array=self.pixel_array)
            coldefs = fits.ColDefs([c1])
            hpx_hdu = fits.BinTableHDU.from_columns(coldefs)
            hpx_hdu.header['EXTNAME'] = 'HPX_INDS'
            hdulist.append(hpx_hdu)

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(filename, clobber=clobber)
        else:
            hdulist.writeto(filename, overwrite=clobber)
