"""Tests for BeamFits object."""
import nose.tools as nt
import os
import numpy as np
import astropy
from astropy.io import fits
from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils
from .test_uvbeam import fill_dummy_beam


def test_writeread():
    beam_in = UVBeam()
    beam_out = UVBeam()
    # fill UVBeam object with dummy data for now for testing purposes
    beam_in = fill_dummy_beam(beam_in, 'efield', 'az_za')

    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    nt.assert_equal(beam_in, beam_out)

    # redo for power beam
    del(beam_in)
    beam_in = UVBeam()
    beam_in = fill_dummy_beam(beam_in, 'power', 'az_za')

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)
    nt.assert_equal(beam_in, beam_out)

    # now replace 'power' with 'intensity' for btype
    F = fits.open(write_file)
    data = F[0].data
    primary_hdr = F[0].header
    primary_hdr['BTYPE'] = 'Intensity'
    hdunames = uvutils.fits_indexhdus(F)
    basisvec_hdu = F[hdunames['BASISVEC']]
    bandpass_hdu = F[hdunames['BANDPARM']]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, basisvec_hdu, bandpass_hdu])

    if float(astropy.__version__[0:3]) < 1.3:
        hdulist.writeto(write_file, clobber=True)
    else:
        hdulist.writeto(write_file, overwrite=True)

    beam_out.read_beamfits(write_file)
    nt.assert_equal(beam_in, beam_out)

    # now remove coordsys but leave ctypes 1 & 2
    F = fits.open(write_file)
    data = F[0].data
    primary_hdr = F[0].header
    primary_hdr.pop('COORDSYS')
    hdunames = uvutils.fits_indexhdus(F)
    basisvec_hdu = F[hdunames['BASISVEC']]
    bandpass_hdu = F[hdunames['BANDPARM']]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, basisvec_hdu, bandpass_hdu])

    if float(astropy.__version__[0:3]) < 1.3:
        hdulist.writeto(write_file, clobber=True)
    else:
        hdulist.writeto(write_file, overwrite=True)

    beam_out.read_beamfits(write_file)
    nt.assert_equal(beam_in, beam_out)


def test_writeread_healpix():
    beam_in = UVBeam()
    beam_out = UVBeam()
    # fill UVBeam object with dummy data for now for testing purposes
    beam_in = fill_dummy_beam(beam_in, 'efield', 'healpix')

    write_file = os.path.join(DATA_PATH, 'test/outtest_beam_hpx.fits')

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    nt.assert_equal(beam_in, beam_out)

    # redo for power beam
    del(beam_in)
    beam_in = UVBeam()
    beam_in = fill_dummy_beam(beam_in, 'power', 'healpix')

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    nt.assert_equal(beam_in, beam_out)

    # now remove coordsys but leave ctype 1
    F = fits.open(write_file)
    data = F[0].data
    primary_hdr = F[0].header
    primary_hdr.pop('COORDSYS')
    hdunames = uvutils.fits_indexhdus(F)
    basisvec_hdu = F[hdunames['BASISVEC']]
    hpx_hdu = F[hdunames['HPX_INDS']]
    bandpass_hdu = F[hdunames['BANDPARM']]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, basisvec_hdu, hpx_hdu, bandpass_hdu])

    if float(astropy.__version__[0:3]) < 1.3:
        hdulist.writeto(write_file, clobber=True)
    else:
        hdulist.writeto(write_file, overwrite=True)

    beam_out.read_beamfits(write_file)
    nt.assert_equal(beam_in, beam_out)


def test_errors():
    beam_in = UVBeam()
    beam_out = UVBeam()
    beam_in = fill_dummy_beam(beam_in, 'efield', 'az_za')
    beam_in.beam_type = 'foo'

    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')
    nt.assert_raises(ValueError, beam_in.write_beamfits, write_file, clobber=True)
    nt.assert_raises(ValueError, beam_in.write_beamfits, write_file,
                     clobber=True, run_check=False)

    beam_in.beam_type = 'efield'
    beam_in.antenna_type = 'phased_array'
    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')
    nt.assert_raises(ValueError, beam_in.write_beamfits, write_file, clobber=True)

    # now change values for various items in primary hdu to test errors
    beam_in = fill_dummy_beam(beam_in, 'efield', 'az_za')

    header_vals_to_change = [{'BTYPE': 'foo'}, {'COORDSYS': 'sin_zenith'},
                             {'NAXIS': ''}]

    for i, hdr_dict in enumerate(header_vals_to_change):
        beam_in.write_beamfits(write_file, clobber=True)

        keyword = hdr_dict.keys()[0]
        new_val = hdr_dict[keyword]
        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils.fits_indexhdus(F)
        basisvec_hdu = F[hdunames['BASISVEC']]
        bandpass_hdu = F[hdunames['BANDPARM']]

        if 'NAXIS' in keyword:
            ax_num = keyword.split('NAXIS')[1]
            if ax_num != '':
                ax_num = int(ax_num)
                ax_use = len(data.shape) - ax_num
                new_arrays = np.split(data, primary_hdr[keyword], axis=ax_use)
                data = new_arrays[0]
            else:
                data = np.squeeze(np.split(data, primary_hdr['NAXIS1'],
                                  axis=len(data.shape) - 1)[0])
        else:
            primary_hdr[keyword] = new_val

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, basisvec_hdu, bandpass_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        nt.assert_raises(ValueError, beam_out.read_beamfits, write_file)

    # now change values for various items in basisvec hdu to not match primary hdu
    beam_in = fill_dummy_beam(beam_in, 'efield', 'az_za')

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
        bandpass_hdu = F[hdunames['BANDPARM']]

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
        hdulist = fits.HDUList([prihdu, basisvec_hdu, bandpass_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        nt.assert_raises(ValueError, beam_out.read_beamfits, write_file)


def test_healpix_errors():
    beam_in = UVBeam()
    beam_out = UVBeam()
    write_file = os.path.join(DATA_PATH, 'test/outtest_beam_hpx.fits')

    # now change values for various items in primary hdu to test errors
    beam_in = fill_dummy_beam(beam_in, 'efield', 'healpix')

    header_vals_to_change = [{'CTYPE1': 'foo'}, {'NAXIS1': ''}]

    for i, hdr_dict in enumerate(header_vals_to_change):
        beam_in.write_beamfits(write_file, clobber=True)

        keyword = hdr_dict.keys()[0]
        new_val = hdr_dict[keyword]
        F = fits.open(write_file)
        data = F[0].data
        primary_hdr = F[0].header
        hdunames = uvutils.fits_indexhdus(F)
        basisvec_hdu = F[hdunames['BASISVEC']]
        hpx_hdu = F[hdunames['HPX_INDS']]
        bandpass_hdu = F[hdunames['BANDPARM']]

        if 'NAXIS' in keyword:
            ax_num = keyword.split('NAXIS')[1]
            if ax_num != '':
                ax_num = int(ax_num)
                ax_use = len(data.shape) - ax_num
                new_arrays = np.split(data, primary_hdr[keyword], axis=ax_use)
                data = new_arrays[0]
            else:
                data = np.squeeze(np.split(data, primary_hdr['NAXIS1'],
                                  axis=len(data.shape) - 1)[0])
        else:
            primary_hdr[keyword] = new_val

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, basisvec_hdu, hpx_hdu, bandpass_hdu])

        if float(astropy.__version__[0:3]) < 1.3:
            hdulist.writeto(write_file, clobber=True)
        else:
            hdulist.writeto(write_file, overwrite=True)

        nt.assert_raises(ValueError, beam_out.read_beamfits, write_file)

    # now change values for various items in basisvec hdu to not match primary hdu
    beam_in = fill_dummy_beam(beam_in, 'efield', 'healpix')

    header_vals_to_change = [{'CTYPE1': 'foo'}, {'NAXIS1': ''}]

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
        hpx_hdu = F[hdunames['HPX_INDS']]
        bandpass_hdu = F[hdunames['BANDPARM']]

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
        hdulist = fits.HDUList([prihdu, basisvec_hdu, hpx_hdu, bandpass_hdu])

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


def test_multi_files():
    """
    Reading multiple files at once.
    """
    beam_full = UVBeam()
    # fill UVBeam object with dummy data for now for testing purposes
    beam_full = fill_dummy_beam(beam_full, 'efield', 'az_za')

    testfile1 = os.path.join(DATA_PATH, 'test/outtest_beam1.fits')
    testfile2 = os.path.join(DATA_PATH, 'test/outtest_beam2.fits')

    beam1 = beam_full.select(freq_chans=np.arange(0, 50), inplace=False)
    beam2 = beam_full.select(freq_chans=np.arange(50, 100), inplace=False)
    beam1.write_beamfits(testfile1, clobber=True)
    beam2.write_beamfits(testfile2, clobber=True)
    beam1.read_beamfits([testfile1, testfile2])
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(beam_full.history + '  Downselected '
                                           'to specific frequencies using pyuvdata. '
                                           'Combined data along frequency axis using'
                                           ' pyuvdata.', beam1.history))

    beam1.history = beam_full.history
    nt.assert_equal(beam1, beam_full)
