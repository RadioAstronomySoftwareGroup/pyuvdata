import nose.tools as nt
import os
import numpy as np
import copy
from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam
from pyuvdata.cst_beam import CSTBeam
import pyuvdata.tests as uvtest

filenames = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
cst_files = [os.path.join(DATA_PATH, f) for f in filenames]


def test_frequencyparse():
    beam1 = CSTBeam()

    parsed_freqs = [beam1.name2freq(f) for f in cst_files]
    nt.assert_equal(parsed_freqs, [150e6, 123e6])

    test_path = '/pyuvdata_1510194907049/_t_env/lib/python2.7/site-packages/pyuvdata/data/'
    test_files = [os.path.join(test_path, f) for f in filenames]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    nt.assert_equal(parsed_freqs, [150e6, 123e6])

    test_names = ['HERA_Sim_120.87kHz.txt', 'HERA_Sim_120.87GHz.txt', 'HERA_Sim_120.87Hz.txt']
    test_files = [os.path.join(test_path, f) for f in test_names]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    nt.assert_equal(parsed_freqs, [120.87e3, 120.87e9, 120.87])


def test_read_power():
    beam1 = UVBeam()
    beam2 = UVBeam()

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=2,
                         message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1', feed_pol=['x'],
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'power')
    nt.assert_equal(beam1.data_array.shape, (1, 1, 2, 2, 181, 360))
    nt.assert_equal(beam1, beam2)

    # test passing in other polarization
    beam2.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                        feed_pol='y', telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_true(np.allclose(beam2.polarization_array, np.array([-6, -5])))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :]))

    # test single frequency
    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam([cst_files[0]], beam_type='power', frequency=[150e6],
                        telescope_name='TEST', feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'power')
    nt.assert_equal(beam1.data_array.shape, (1, 1, 2, 1, 181, 360))
    nt.assert_equal(beam1, beam2)

    # test single frequency and not rotating the polarization
    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0', 'rotate_pol': False},
                         message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam([cst_files[0]], beam_type='power', frequency=[150e6],
                        rotate_pol=False, telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'power')
    nt.assert_equal(beam1.polarization_array, np.array([-5]))
    nt.assert_equal(beam1.data_array.shape, (1, 1, 1, 1, 181, 360))
    nt.assert_equal(beam1, beam2)

    # test reading in multiple polarization files
    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='power', frequency=[150e6],
                        feed_pol=['x', 'y'], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_equal(beam1.data_array.shape, (1, 1, 2, 1, 181, 360))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam1.data_array[:, :, 1, :, :, :]))

    # test errors
    nt.assert_raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                     frequency=[150e6, 123e6, 100e6], telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, cst_files[0], beam_type='power',
                     frequency=[150e6, 123e6], telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, [cst_files[0], cst_files[0], cst_files[0]],
                     beam_type='power',
                     feed_pol=['x', 'y'], telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, cst_files[0], beam_type='power',
                     feed_pol=['x', 'y'], telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')


def test_read_efield():
    beam1 = UVBeam()
    beam2 = UVBeam()

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=2,
                         message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'efield')
    nt.assert_equal(beam1.data_array.shape, (2, 1, 2, 2, 181, 360))
    nt.assert_equal(beam1, beam2)

    # test passing in other polarization
    beam2.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                        feed_pol='y', telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_true(beam2.feed_array[0], 'x')
    nt.assert_true(beam2.feed_array[1], 'y')
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :]))

    # test single frequency and not rotating the polarization
    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0', 'rotate_pol': False},
                         message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam(cst_files[0], beam_type='efield', frequency=[150e6],
                        rotate_pol=False, telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'efield')
    nt.assert_equal(beam1.feed_array, np.array(['x']))
    nt.assert_equal(beam1.data_array.shape, (2, 1, 1, 1, 181, 360))
    nt.assert_equal(beam1, beam2)

    # test reading in multiple polarization files
    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='efield', frequency=[150e6],
                        feed_pol=['x', 'y'], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_equal(beam1.data_array.shape, (2, 1, 2, 1, 181, 360))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam1.data_array[:, :, 1, :, :, :]))


def test_readcst_writebeamfits():
    beam_in = UVBeam()
    beam_out = UVBeam()
    testfile = os.path.join(DATA_PATH, 'test/outtest_beam.fits')

    uvtest.checkWarnings(beam_in.read_cst_beam, [cst_files],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=2,
                         message='No frequency provided. Detected frequency is')
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile)

    nt.assert_equal(beam_in, beam_out)

    uvtest.checkWarnings(beam_in.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=2,
                         message='No frequency provided. Detected frequency is')
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
                          'model_version': '1.0'}, nwarnings=2,
                         message='No frequency provided. Detected frequency is')
    beam_in.az_za_to_healpix()
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile)

    nt.assert_equal(beam_in.pixel_coordinate_system, 'healpix')
    nt.assert_equal(beam_in.beam_type, 'power')
    nt.assert_equal(beam_in.data_array.shape[0:4], (1, 1, 2, 2))
    nt.assert_equal(beam_in, beam_out)

    uvtest.checkWarnings(beam_in.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=2,
                         message='No frequency provided. Detected frequency is')
    nt.assert_raises(ValueError, beam_in.az_za_to_healpix)
