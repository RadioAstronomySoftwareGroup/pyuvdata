"""Tests for BeamFits object."""
import nose.tools as nt
import os
import numpy as np
import astropy
from astropy.io import fits
from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH
import pyuvdata.version as uvversion
import pyuvdata.utils as uvutils


def fill_dummy_beam(beam_obj, beam_type):
    beam_obj.set_simple()
    beam_obj.telescope_name = 'testscope'
    beam_obj.feed_name = 'testfeed'
    beam_obj.feed_version = '0.1'
    beam_obj.model_name = 'testmodel'
    beam_obj.model_version = '1.0'
    beam_obj.history = 'random data for test'

    pyuvdata_version_str = ('  Read/written with pyuvdata version: ' +
                            uvversion.version + '.')
    if uvversion.git_hash is not '':
        pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin +
                                 '.  Git hash: ' + uvversion.git_hash +
                                 '.  Git branch: ' + uvversion.git_branch +
                                 '.  Git description: ' + uvversion.git_description + '.')
    beam_obj.history += pyuvdata_version_str

    beam_obj.pixel_coordinate_system = 'az_za'
    beam_obj.axis1_array = np.arange(-180.0, 180.0, 5.0)
    beam_obj.Naxes1 = len(beam_obj.axis1_array)
    beam_obj.axis2_array = np.arange(-90.0, 90.0, 5.0)
    beam_obj.Naxes2 = len(beam_obj.axis2_array)

    beam_obj.freq_array = np.arange(150e6, 160e6, 1e6)
    beam_obj.freq_array = beam_obj.freq_array[np.newaxis, :]
    beam_obj.Nfreqs = beam_obj.freq_array.shape[1]
    beam_obj.spw_array = np.array([0])
    beam_obj.Nspws = len(beam_obj.spw_array)
    beam_obj.data_normalization = 'peak'

    if beam_type == 'power':
        beam_obj.set_power()
        beam_obj.polarization_array = np.array([-5, -6, -7, -8])
        beam_obj.Npols = len(beam_obj.polarization_array)
        beam_obj.Naxes_vec = 1

        data_size_tuple = (beam_obj.Naxes_vec, beam_obj.Nspws, beam_obj.Npols,
                           beam_obj.Nfreqs, beam_obj.Naxes2, beam_obj.Naxes1)
        beam_obj.data_array = np.square(np.random.normal(0.0, 0.2, size=data_size_tuple))
    else:
        beam_obj.set_efield()
        beam_obj.feed_array = ['x', 'y']
        beam_obj.Nfeeds = len(beam_obj.feed_array)
        beam_obj.Naxes_vec = 2
        beam_obj.basis_vector_array = np.zeros((beam_obj.Naxes_vec, 2, beam_obj.Naxes2, beam_obj.Naxes1))
        beam_obj.basis_vector_array[0, 0, :, :] = 1.0
        beam_obj.basis_vector_array[1, 1, :, :] = 1.0

        data_size_tuple = (beam_obj.Naxes_vec, beam_obj.Nspws, beam_obj.Nfeeds,
                           beam_obj.Nfreqs, beam_obj.Naxes2, beam_obj.Naxes1)
        beam_obj.data_array = (np.random.normal(0.0, 0.2, size=data_size_tuple) +
                               1j * np.random.normal(0.0, 0.2, size=data_size_tuple))

    beam_obj.extra_keywords = {'KEY1': 'test_keyword'}

    return beam_obj


def test_writeread():
    beam_in = UVBeam()
    beam_out = UVBeam()
    # fill UVBeam object with dummy data for now for testing purposes
    beam_in = fill_dummy_beam(beam_in, 'efield')

    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    nt.assert_equal(beam_in, beam_out)

    # redo for power beam
    beam_in = fill_dummy_beam(beam_in, 'power')

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    nt.assert_equal(beam_in, beam_out)


def test_errors():
    beam_in = UVBeam()
    beam_out = UVBeam()
    beam_in = fill_dummy_beam(beam_in, 'efield')
    beam_in.beam_type = 'foo'

    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')
    nt.assert_raises(ValueError, beam_in.write_beamfits, write_file, clobber=True)

    # now change values for various items in basisvec hdu to not match primary hdu
    beam_in.beam_type = 'efield'

    header_vals_to_change = [{'COORDSYS': 'foo'}, {'CTYPE1': 'foo'},
                             {'CTYPE2': 'foo'},
                             {'CDELT1': np.diff(beam_in.axis1_array)[0] * 2},
                             {'CDELT2': np.diff(beam_in.axis2_array)[0] * 2},
                             {'NAXIS4': ''}]

    for i, hdr_dict in enumerate(header_vals_to_change):
        beam_in.write_beamfits(write_file, clobber=True)

        keyword = hdr_dict.keys()[0]
        new_val = hdr_dict[keyword]
        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils.fits_indexhdus(F)
        basisvec_hdu = F[hdunames['BASISVEC']]
        basisvec_hdr = basisvec_hdu.header
        basisvec_data = basisvec_hdu.data

        if 'NAXIS' in keyword:
            ax_num = keyword.split('NAXIS')[1]
            if ax_num != '':
                ax_num = int(ax_num)
                ax_use = len(basisvec_data.shape) - ax_num
                new_arrays = np.split(basisvec_data, basisvec_hdr[keyword], axis=ax_use)
                basisvec_data = new_arrays[0]
            else:
                basisvec_data = np.split(basisvec_data, basisvec_hdr['NAXIS1'],
                                         axis=len(basisvec_data.shape) - 1)[0]
        else:
            basisvec_hdr[keyword] = new_val

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        basisvec_hdu = fits.ImageHDU(data=basisvec_data, header=basisvec_hdr)
        hdulist = fits.HDUList([prihdu, basisvec_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        nt.assert_raises(ValueError, beam_out.read_beamfits, write_file)


def test_casa_beam():
    # test reading in CASA power beam. Some header items are missing...
    beam_in = UVBeam()
    beam_out = UVBeam()
    casa_file = os.path.join(DATA_PATH, 'HERABEAM.FITS')
    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')
    beam_in.read_beamfits(casa_file, run_check=False)

    # fill in missing parameters
    beam_in.data_normalization = 'peak'
    beam_in.feed_name = 'casa_ideal'
    beam_in.feed_version = 'v0'
    beam_in.model_name = 'casa_airy'
    beam_in.model_version = 'v0'

    # this file is actually in sine projection RA/DEC at zenith at a particular time.
    # For now pretend it's in sine projection of az/za
    beam_in.pixel_coordinate_system = 'sin_zenith'

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    nt.assert_equal(beam_in, beam_out)
