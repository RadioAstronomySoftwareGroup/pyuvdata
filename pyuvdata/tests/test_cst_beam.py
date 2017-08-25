import nose.tools as nt
import os
import numpy as np
from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam

filenames = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
cst_files = [os.path.join(DATA_PATH, f) for f in filenames]


def test_read():
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam1.read_cst_power(cst_files, telescope_name='TEST', feed_name='bob',
                         feed_version='0.1', model_name='E-field pattern - Rigging height 4.9m',
                         model_version='1.0')

    beam2.read_cst_power(cst_files, frequencies=[150e6, 123e6], telescope_name='TEST',
                         feed_name='bob', feed_version='0.1',
                         model_name='E-field pattern - Rigging height 4.9m',
                         model_version='1.0')

    nt.assert_equal(beam1, beam2)

    # test the bit about checking if the input is a list/tuple or not
    beam1.read_cst_power(cst_files[0], telescope_name='TEST', feed_name='bob',
                         feed_version='0.1', model_name='E-field pattern - Rigging height 4.9m',
                         model_version='1.0')


def test_readcst_writebeamfits():
    beam_in = UVBeam()
    beam_out = UVBeam()
    testfile = os.path.join(DATA_PATH, 'test/outtest_beam.fits')

    beam_in.read_cst_power(cst_files, telescope_name='TEST', feed_name='bob',
                           feed_version='0.1', model_name='E-field pattern - Rigging height 4.9m',
                           model_version='1.0')
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile)

    print(beam_in.history)
    print(beam_out.history)
    nt.assert_equal(beam_in, beam_out)


def test_readcst_writehealpixfits():
    beam_in = UVBeam()
    beam_out = UVBeam()
    testfile = os.path.join(DATA_PATH, 'test/outtest_beam.fits')

    beam_in.read_cst_power(cst_files, telescope_name='TEST', feed_name='bob',
                           feed_version='0.1', model_name='E-field pattern - Rigging height 4.9m',
                           model_version='1.0')
    beam_in.az_za_to_healpix()
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile)
