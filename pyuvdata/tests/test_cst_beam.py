# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

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


def test_basic_frequencyparse():
    beam1 = CSTBeam()

    parsed_freqs = [beam1.name2freq(f) for f in cst_files]
    nt.assert_equal(parsed_freqs, [150e6, 123e6])


def test_frequencyparse_extra_numbers():
    beam1 = CSTBeam()
    test_path = os.path.join('pyuvdata_1510194907049', '_t_env', 'lib',
                             'python2.7', 'site-packages', 'pyuvdata', 'data')
    test_files = [os.path.join(test_path, f) for f in filenames]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    nt.assert_equal(parsed_freqs, [150e6, 123e6])


def test_frequencyparse_nicf_path():
    beam1 = CSTBeam()
    test_path = os.path.join('Simulations', 'Radiation_patterns',
                             'E-field pattern-Rigging height4.9m',
                             'HERA_4.9m_E-pattern_100-200MHz')
    test_files = [os.path.join(test_path, f) for f in filenames]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    nt.assert_equal(parsed_freqs, [150e6, 123e6])


def test_frequencyparse_decimal_nonMHz():
    beam1 = CSTBeam()
    test_path = os.path.join('Simulations', 'Radiation_patterns',
                             'E-field pattern-Rigging height4.9m',
                             'HERA_4.9m_E-pattern_100-200MHz')
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

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'power')
    nt.assert_equal(beam1.data_array.shape, (1, 1, 2, 2, 181, 360))
    nt.assert_equal(np.max(beam1.data_array), 8275.5409)

    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, np.where(beam1.axis1_array == 0)[0]],
                               beam1.data_array[:, :, 1, :, :, np.where(beam1.axis1_array == np.pi / 2.)[0]]))

    # test passing in other polarization
    beam2.read_cst_beam(np.array(cst_files), beam_type='power', frequency=np.array([150e6, 123e6]),
                        feed_pol='y', telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    nt.assert_true(np.allclose(beam1.freq_array, beam2.freq_array))

    nt.assert_true(np.allclose(beam2.polarization_array, np.array([-6, -5])))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :]))

    # test single frequency
    uvtest.checkWarnings(beam1.read_cst_beam, [[cst_files[0]]],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequency provided. Detected frequency is')

    nt.assert_equal(beam1.freq_array, [150e6])
    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'power')
    nt.assert_equal(beam1.data_array.shape, (1, 1, 2, 1, 181, 360))

    # test single frequency and not rotating the polarization
    uvtest.checkWarnings(beam2.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0', 'rotate_pol': False},
                         message='No frequency provided. Detected frequency is')

    nt.assert_equal(beam2.freq_array, [150e6])
    nt.assert_equal(beam2.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam2.beam_type, 'power')
    nt.assert_equal(beam2.polarization_array, np.array([-5]))
    nt.assert_equal(beam2.data_array.shape, (1, 1, 1, 1, 181, 360))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array))

    # test reading in multiple polarization files
    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='power', frequency=[150e6],
                        feed_pol=np.array(['xx', 'yy']), telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_equal(beam1.data_array.shape, (1, 1, 2, 1, 181, 360))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam1.data_array[:, :, 1, :, :, :]))

    # test reading in cross polarization files
    beam2.read_cst_beam([cst_files[0]], beam_type='power', frequency=[150e6],
                        feed_pol=np.array(['xy']), telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_true(np.allclose(beam2.polarization_array, np.array([-7, -8])))
    nt.assert_equal(beam2.data_array.shape, (1, 1, 2, 1, 181, 360))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :]))

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

    nt.assert_raises(ValueError, beam1.read_cst_beam, [[cst_files[0]], [cst_files[1]]],
                     beam_type='power', frequency=[150e6, 123e6],
                     feed_pol=['x'], telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, np.array([[cst_files[0]], [cst_files[1]]]),
                     beam_type='power', frequency=[150e6, 123e6],
                     feed_pol=['x'], telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                     frequency=[[150e6], [123e6]], telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                     frequency=np.array([[150e6], [123e6]]), telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                     feed_pol=[['x'], ['y']], frequency=150e6,
                     telescope_name='TEST',
                     feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    nt.assert_raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                     feed_pol=np.array([['x'], ['y']]), frequency=150e6,
                     telescope_name='TEST',
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

    nt.assert_equal(beam1.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam1.beam_type, 'efield')
    nt.assert_equal(beam1.data_array.shape, (2, 1, 2, 2, 181, 360))
    nt.assert_equal(np.max(np.abs(beam1.data_array)), 90.97)

    # test passing in other polarization
    beam2.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                        feed_pol='y', telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_equal(beam2.feed_array[0], 'y')
    nt.assert_equal(beam2.feed_array[1], 'x')
    nt.assert_equal(beam1.data_array.shape, (2, 1, 2, 2, 181, 360))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :]))

    # test single frequency and not rotating the polarization
    uvtest.checkWarnings(beam2.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0', 'rotate_pol': False},
                         message='No frequency provided. Detected frequency is')

    nt.assert_equal(beam2.pixel_coordinate_system, 'az_za')
    nt.assert_equal(beam2.beam_type, 'efield')
    nt.assert_equal(beam2.feed_array, np.array(['x']))
    nt.assert_equal(beam2.data_array.shape, (2, 1, 1, 1, 181, 360))

    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, 1, :, :], beam2.data_array[:, :, 0, 0, :, :]))

    # test reading in multiple polarization files
    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='efield', frequency=[150e6],
                        feed_pol=['x', 'y'], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    nt.assert_equal(beam1.data_array.shape, (2, 1, 2, 1, 181, 360))
    nt.assert_true(np.allclose(beam1.data_array[:, :, 0, :, :, :], beam1.data_array[:, :, 1, :, :, :]))


def test_no_deg_units():
    # need to write a modified file to test headers not in degrees
    testfile = os.path.join(DATA_PATH, 'test', 'HERA_NicCST_150MHz_modified.txt')
    with open(cst_files[0], 'r') as file:
        line1 = file.readline()
        line2 = file.readline()

    data = np.loadtxt(cst_files[0], skiprows=2)

    raw_names = line1.split(']')
    raw_names = [raw_name for raw_name in raw_names if '\n' not in raw_name]
    column_names = []
    column_names_simple = []
    units = []
    for raw_name in raw_names:
        column_name, unit = tuple(raw_name.split('['))
        column_names.append(column_name)
        column_names_simple.append(''.join(column_name.lower().split(' ')))
        if unit != 'deg.':
            units.append(unit)
        else:
            units.append('    ')

    new_column_headers = []
    for index, name in enumerate(column_names):
        new_column_headers.append(name + '[' + units[index] + ']')

    new_header = ''
    for col in new_column_headers:
        new_header += '{:12}'.format(col)

    beam1 = UVBeam()
    beam2 = UVBeam()

    # format to match existing file
    existing_format = ['%8.3f', '%15.3f', '%20.3e', '%19.3e', '%19.3f', '%19.3e', '%19.3f', '%19.3e']
    np.savetxt(testfile, data, fmt=existing_format, header=new_header + '\n' + line2, comments='')
    # this errors because the phi 2pi rotation doesn't work (because they are degrees but the code thinks they're radians)
    nt.assert_raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
                     frequency=np.array([150e6]), feed_pol='y',
                     telescope_name='TEST', feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    theta_col = np.where(np.array(column_names_simple) == 'theta')[0][0]
    phi_col = np.where(np.array(column_names_simple) == 'phi')[0][0]
    theta_phase_col = np.where(np.array(column_names_simple) == 'phase(theta)')[0][0]
    phi_phase_col = np.where(np.array(column_names_simple) == 'phase(phi)')[0][0]

    data[:, theta_col] = np.radians(data[:, theta_col])
    data[:, phi_col] = np.radians(data[:, phi_col])
    data[:, theta_phase_col] = np.radians(data[:, theta_phase_col])
    data[:, phi_phase_col] = np.radians(data[:, phi_phase_col])

    np.savetxt(testfile, data, fmt=existing_format, header=new_header + '\n' + line2, comments='')
    # this errors because theta isn't regularly gridded (too few sig figs)
    nt.assert_raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
                     frequency=np.array([150e6]), feed_pol='y',
                     telescope_name='TEST', feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    # use more decimal places for theta so that it is regularly gridded
    new_format = ['%15.12e', '%15.3e', '%20.3e', '%19.3e', '%19.3f', '%19.3e', '%19.3f', '%19.3e']
    np.savetxt(testfile, data, fmt=new_format, header=new_header + '\n' + line2, comments='')
    # this errors because phi isn't regularly gridded (too few sig figs)
    nt.assert_raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
                     frequency=np.array([150e6]), feed_pol='y',
                     telescope_name='TEST', feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')

    # use more decimal places so that it is regularly gridded and matches data
    new_format = ['%15.12e', '%15.12e', '%20.3e', '%19.3e', '%19.12f', '%19.3e', '%19.12f', '%19.3e']
    np.savetxt(testfile, data, fmt=new_format, header=new_header + '\n' + line2, comments='')

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=1,
                         message='No frequency provided. Detected frequency is')

    uvtest.checkWarnings(beam2.read_cst_beam, [testfile],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=1,
                         message='No frequency provided. Detected frequency is')

    nt.assert_equal(beam1, beam2)

    # remove a row to make data not on a grid to catch that error
    data = data[1:, :]

    new_format = ['%15.12e', '%15.12e', '%20.3e', '%19.3e', '%19.12f', '%19.3e', '%19.12f', '%19.3e']
    np.savetxt(testfile, data, fmt=new_format, header=new_header + '\n' + line2, comments='')
    # this errors because theta & phi aren't on a strict grid
    nt.assert_raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
                     frequency=np.array([150e6]), feed_pol='y',
                     telescope_name='TEST', feed_name='bob', feed_version='0.1',
                     model_name='E-field pattern - Rigging height 4.9m',
                     model_version='1.0')
