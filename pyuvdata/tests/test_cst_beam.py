import nose.tools as nt
import os
import numpy as np
from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam
import pyuvdata.tests as uvtest

filenames = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
cst_files = [os.path.join(DATA_PATH, f) for f in filenames]


def test_read_power():
    beam1 = UVBeam()
    beam2 = UVBeam()

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequencies provided. Detected frequencies are')

    beam2.read_cst_beam(cst_files, beam_type='power', frequencies=[150e6, 123e6], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'power')
    nt.assert_equal(beam1.data_array.shape, (1, 1, 2, 2, 181, 360))
    nt.assert_equal(beam1, beam2)

    # test the bit about checking if the input is a list/tuple or not
    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequencies provided. Detected frequencies are')


def test_read_efield():
    beam1 = UVBeam()
    beam2 = UVBeam()

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequencies provided. Detected frequencies are')

    beam2.read_cst_beam(cst_files, beam_type='efield', frequencies=[150e6, 123e6], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'efield')
    nt.assert_equal(beam1.data_array.shape, (2, 1, 2, 2, 181, 360))
    nt.assert_equal(beam1, beam2)


def test_readcst_writebeamfits():
    beam_in = UVBeam()
    beam_out = UVBeam()
    testfile = os.path.join(DATA_PATH, 'test/outtest_beam.fits')

    uvtest.checkWarnings(beam_in.read_cst_beam, [cst_files],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequencies provided. Detected frequencies are')
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile)

    nt.assert_equal(beam_in, beam_out)

    uvtest.checkWarnings(beam_in.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequencies provided. Detected frequencies are')
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile)

    nt.assert_equal(beam_in, beam_out)


def test_readpower_writehealpixfits():
    beam_in = UVBeam()
    beam_out = UVBeam()
    testfile = os.path.join(DATA_PATH, 'test/outtest_beam.fits')

    uvtest.checkWarnings(beam_in.read_cst_beam, [cst_files],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequencies provided. Detected frequencies are')
    beam_in.az_za_to_healpix()
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile)

    nt.assert_equal(beam_in.pixel_coordinate_system, 'healpix')
    nt.assert_equal(beam_in.beam_type, 'power')
    nt.assert_equal(beam_in.data_array.shape[0:4], (1, 1, 2, 2))
    nt.assert_true(beam_in.Npixels < 360 * 181)
    nt.assert_equal(beam_in, beam_out)

    uvtest.checkWarnings(beam_in.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequencies provided. Detected frequencies are')
    nt.assert_raises(ValueError, beam_in.az_za_to_healpix)
