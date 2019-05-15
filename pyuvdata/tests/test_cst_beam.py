# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np

from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam
from pyuvdata.cst_beam import CSTBeam
import pyuvdata.tests as uvtest

filenames = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
cst_folder = 'NicCSTbeams'
cst_files = [os.path.join(DATA_PATH, cst_folder, f) for f in filenames]
cst_yaml_file = os.path.join(DATA_PATH, cst_folder, 'NicCSTbeams.yaml')
cst_yaml_vivaldi = os.path.join(DATA_PATH, cst_folder, 'HERA_Vivaldi_CST_beams.yaml')


def test_basic_frequencyparse():
    beam1 = CSTBeam()

    parsed_freqs = [beam1.name2freq(f) for f in cst_files]
    assert parsed_freqs == [150e6, 123e6]


def test_frequencyparse_extra_numbers():
    beam1 = CSTBeam()
    test_path = os.path.join('pyuvdata_1510194907049', '_t_env', 'lib',
                             'python2.7', 'site-packages', 'pyuvdata', 'data')
    test_files = [os.path.join(test_path, f) for f in filenames]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    assert parsed_freqs == [150e6, 123e6]


def test_frequencyparse_nicf_path():
    beam1 = CSTBeam()
    test_path = os.path.join('Simulations', 'Radiation_patterns',
                             'E-field pattern-Rigging height4.9m',
                             'HERA_4.9m_E-pattern_100-200MHz')
    test_files = [os.path.join(test_path, f) for f in filenames]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    assert parsed_freqs == [150e6, 123e6]


def test_frequencyparse_decimal_nonMHz():
    beam1 = CSTBeam()
    test_path = os.path.join('Simulations', 'Radiation_patterns',
                             'E-field pattern-Rigging height4.9m',
                             'HERA_4.9m_E-pattern_100-200MHz')
    test_names = ['HERA_Sim_120.87kHz.txt', 'HERA_Sim_120.87GHz.txt', 'HERA_Sim_120.87Hz.txt']
    test_files = [os.path.join(test_path, f) for f in test_names]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    assert parsed_freqs == [120.87e3, 120.87e9, 120.87]


@uvtest.skipIf_no_yaml
def test_read_yaml():
    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {'software': 'CST 2016', 'sim_type': 'E-farfield',
                      'layout': '1 antenna', 'port_num': 1}

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'HERA', 'feed_name': 'Dipole',
                          'feed_version': '1.0', 'model_name': 'Dipole - Rigging height 4.9 m',
                          'model_version': '1.0', 'x_orientation': 'east',
                          'reference_impedance': 100,
                          'history': 'Derived from https://github.com/Nicolas-Fagnoni/Simulations.'
                          '\nOnly 2 files included to keep test data volume low.',
                          'extra_keywords': extra_keywords},
                         nwarnings=2, message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam(cst_yaml_file, beam_type='efield')
    assert beam1 == beam2

    assert beam2.reference_impedance == 100
    assert beam2.extra_keywords == extra_keywords


@uvtest.skipIf_no_yaml
def test_read_yaml_override():
    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {'software': 'CST 2016', 'sim_type': 'E-farfield',
                      'layout': '1 antenna', 'port_num': 1}

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'test', 'feed_name': 'Dipole',
                          'feed_version': '1.0', 'model_name': 'Dipole - Rigging height 4.9 m',
                          'model_version': '1.0', 'x_orientation': 'east',
                          'reference_impedance': 100,
                          'history': 'Derived from https://github.com/Nicolas-Fagnoni/Simulations.'
                          '\nOnly 2 files included to keep test data volume low.',
                          'extra_keywords': extra_keywords},
                         nwarnings=2, message='No frequency provided. Detected frequency is')

    uvtest.checkWarnings(beam2.read_cst_beam, [cst_yaml_file],
                         {'beam_type': 'efield', 'telescope_name': 'test'},
                         message='The telescope_name keyword is set, overriding '
                         'the value in the settings yaml file.')
    assert beam1 == beam2

    assert beam2.reference_impedance == 100
    assert beam2.extra_keywords == extra_keywords


@uvtest.skipIf_no_yaml
def test_read_yaml_freq_select():
    # test frequency_select
    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {'software': 'CST 2016', 'sim_type': 'E-farfield',
                      'layout': '1 antenna', 'port_num': 1}
    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'efield', 'telescope_name': 'HERA', 'feed_name': 'Dipole',
                          'feed_version': '1.0', 'model_name': 'Dipole - Rigging height 4.9 m',
                          'model_version': '1.0', 'x_orientation': 'east',
                          'reference_impedance': 100,
                          'history': 'Derived from https://github.com/Nicolas-Fagnoni/Simulations.'
                          '\nOnly 2 files included to keep test data volume low.',
                          'extra_keywords': extra_keywords},
                         nwarnings=1, message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam(cst_yaml_file, beam_type='efield', frequency_select=[150e6])

    assert beam1 == beam2

    # test error with using frequency_select where no such frequency
    pytest.raises(ValueError, beam2.read_cst_beam, cst_yaml_file,
                  beam_type='power', frequency_select=[180e6])


@uvtest.skipIf_no_yaml
def test_read_yaml_feed_pol_list():
    # make yaml with a list of (the same) feed_pols
    import yaml

    test_yaml_file = os.path.join(DATA_PATH, cst_folder, 'test_cst_settings.yaml')
    with open(cst_yaml_file, 'r') as file:
        settings_dict = yaml.safe_load(file)

    settings_dict['feed_pol'] = ['x', 'x']

    with open(test_yaml_file, 'w') as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {'software': 'CST 2016', 'sim_type': 'E-farfield',
                      'layout': '1 antenna', 'port_num': 1}

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'efield', 'telescope_name': 'HERA', 'feed_name': 'Dipole',
                          'feed_version': '1.0', 'model_name': 'Dipole - Rigging height 4.9 m',
                          'model_version': '1.0', 'x_orientation': 'east',
                          'reference_impedance': 100,
                          'history': 'Derived from https://github.com/Nicolas-Fagnoni/Simulations.'
                          '\nOnly 2 files included to keep test data volume low.',
                          'extra_keywords': extra_keywords},
                         nwarnings=2, message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam(test_yaml_file, beam_type='efield')
    assert beam1 == beam2

    assert beam2.reference_impedance == 100
    assert beam2.extra_keywords == extra_keywords

    # also test with frequency_select
    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'efield', 'telescope_name': 'HERA', 'feed_name': 'Dipole',
                          'feed_version': '1.0', 'model_name': 'Dipole - Rigging height 4.9 m',
                          'model_version': '1.0', 'x_orientation': 'east',
                          'reference_impedance': 100,
                          'history': 'Derived from https://github.com/Nicolas-Fagnoni/Simulations.'
                          '\nOnly 2 files included to keep test data volume low.',
                          'extra_keywords': extra_keywords},
                         nwarnings=1, message='No frequency provided. Detected frequency is')

    beam2.read_cst_beam(test_yaml_file, beam_type='efield', frequency_select=[150e6])
    assert beam1 == beam2

    os.remove(test_yaml_file)


@uvtest.skipIf_no_yaml
def test_read_yaml_multi_pol():
    # make yaml for one freq, 2 pols
    import yaml

    test_yaml_file = os.path.join(DATA_PATH, cst_folder, 'test_cst_settings.yaml')
    with open(cst_yaml_file, 'r') as file:
        settings_dict = yaml.safe_load(file)

    settings_dict['feed_pol'] = ['x', 'y']
    first_file = settings_dict['filenames'][0]
    settings_dict['filenames'] = [first_file, first_file]
    first_freq = settings_dict['frequencies'][0]
    settings_dict['frequencies'] = [first_freq, first_freq]

    with open(test_yaml_file, 'w') as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {'software': 'CST 2016', 'sim_type': 'E-farfield',
                      'layout': '1 antenna', 'port_num': 1}

    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='efield', frequency=[150e6],
                        feed_pol=['x', 'y'], telescope_name='HERA',
                        feed_name='Dipole', feed_version='1.0',
                        model_name='Dipole - Rigging height 4.9 m',
                        model_version='1.0', x_orientation='east',
                        reference_impedance=100,
                        history='Derived from https://github.com/Nicolas-Fagnoni/Simulations.'
                        '\nOnly 2 files included to keep test data volume low.',
                        extra_keywords=extra_keywords)

    beam2.read_cst_beam(test_yaml_file, beam_type='efield')
    assert beam1 == beam2

    # also test with frequency_select
    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='efield', frequency=[150e6],
                        feed_pol=['x', 'y'], telescope_name='HERA',
                        feed_name='Dipole', feed_version='1.0',
                        model_name='Dipole - Rigging height 4.9 m',
                        model_version='1.0', x_orientation='east',
                        reference_impedance=100,
                        history='Derived from https://github.com/Nicolas-Fagnoni/Simulations.'
                        '\nOnly 2 files included to keep test data volume low.',
                        extra_keywords=extra_keywords)

    beam2.read_cst_beam(test_yaml_file, beam_type='efield', frequency_select=[150e6])
    assert beam2.feed_array.tolist() == ['x', 'y']
    assert beam1 == beam2

    os.remove(test_yaml_file)


@uvtest.skipIf_no_yaml
def test_read_yaml_errors():
    # test error if required key is not present in yaml file
    import yaml

    test_yaml_file = os.path.join(DATA_PATH, cst_folder, 'test_cst_settings.yaml')
    with open(cst_yaml_file, 'r') as file:
        settings_dict = yaml.safe_load(file)

    settings_dict.pop('telescope_name')

    with open(test_yaml_file, 'w') as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    pytest.raises(ValueError, beam1.read_cst_beam, test_yaml_file,
                  beam_type='power')

    os.remove(test_yaml_file)


def test_read_power():
    beam1 = UVBeam()
    beam2 = UVBeam()

    uvtest.checkWarnings(beam1.read_cst_beam, [cst_files],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'}, nwarnings=2,
                         message='No frequency provided. Detected frequency is')

    assert beam1.pixel_coordinate_system == 'az_za'
    assert beam1.beam_type == 'power'
    assert beam1.data_array.shape == (1, 1, 2, 2, 181, 360)
    assert np.max(beam1.data_array) == 8275.5409

    assert np.allclose(beam1.data_array[:, :, 0, :, :, np.where(beam1.axis1_array == 0)[0]],
                       beam1.data_array[:, :, 1, :, :, np.where(beam1.axis1_array == np.pi / 2.)[0]])

    # test passing in other polarization
    beam2.read_cst_beam(np.array(cst_files), beam_type='power', frequency=np.array([150e6, 123e6]),
                        feed_pol='y', telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')

    assert np.allclose(beam1.freq_array, beam2.freq_array)

    assert np.allclose(beam2.polarization_array, np.array([-6, -5]))
    assert np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :])


def test_read_power_single_freq():
    # test single frequency
    beam1 = UVBeam()
    beam2 = UVBeam()

    uvtest.checkWarnings(beam1.read_cst_beam, [[cst_files[0]]],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0'},
                         message='No frequency provided. Detected frequency is')

    assert beam1.freq_array == [150e6]
    assert beam1.pixel_coordinate_system == 'az_za'
    assert beam1.beam_type == 'power'
    assert beam1.data_array.shape == (1, 1, 2, 1, 181, 360)

    # test single frequency and not rotating the polarization
    uvtest.checkWarnings(beam2.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'power', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0', 'rotate_pol': False},
                         message='No frequency provided. Detected frequency is')

    assert beam2.freq_array == [150e6]
    assert beam2.pixel_coordinate_system == 'az_za'
    assert beam2.beam_type == 'power'
    assert beam2.polarization_array == np.array([-5])
    assert beam2.data_array.shape == (1, 1, 1, 1, 181, 360)
    assert np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array)


def test_read_power_multi_pol():
    # test reading in multiple polarization files
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='power', frequency=[150e6],
                        feed_pol=np.array(['xx', 'yy']), telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    assert beam1.data_array.shape == (1, 1, 2, 1, 181, 360)
    assert np.allclose(beam1.data_array[:, :, 0, :, :, :], beam1.data_array[:, :, 1, :, :, :])

    # test reading in cross polarization files
    beam2.read_cst_beam([cst_files[0]], beam_type='power', frequency=[150e6],
                        feed_pol=np.array(['xy']), telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    assert np.allclose(beam2.polarization_array, np.array([-7, -8]))
    assert beam2.data_array.shape == (1, 1, 2, 1, 181, 360)
    assert np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :])


def test_read_errors():
    # test errors
    beam1 = UVBeam()

    pytest.raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                  frequency=[150e6, 123e6, 100e6], telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, cst_files[0], beam_type='power',
                  frequency=[150e6, 123e6], telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, [cst_files[0], cst_files[0], cst_files[0]],
                  beam_type='power',
                  feed_pol=['x', 'y'], telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, cst_files[0], beam_type='power',
                  feed_pol=['x', 'y'], telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, [[cst_files[0]], [cst_files[1]]],
                  beam_type='power', frequency=[150e6, 123e6],
                  feed_pol=['x'], telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, np.array([[cst_files[0]], [cst_files[1]]]),
                  beam_type='power', frequency=[150e6, 123e6],
                  feed_pol=['x'], telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                  frequency=[[150e6], [123e6]], telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                  frequency=np.array([[150e6], [123e6]]), telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
                  feed_pol=[['x'], ['y']], frequency=150e6,
                  telescope_name='TEST',
                  feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    pytest.raises(ValueError, beam1.read_cst_beam, cst_files, beam_type='power',
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

    assert beam1.pixel_coordinate_system == 'az_za'
    assert beam1.beam_type == 'efield'
    assert beam1.data_array.shape == (2, 1, 2, 2, 181, 360)
    assert np.max(np.abs(beam1.data_array)) == 90.97

    # test passing in other polarization
    beam2.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                        feed_pol='y', telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    assert beam2.feed_array[0] == 'y'
    assert beam2.feed_array[1] == 'x'
    assert beam1.data_array.shape == (2, 1, 2, 2, 181, 360)
    assert np.allclose(beam1.data_array[:, :, 0, :, :, :], beam2.data_array[:, :, 0, :, :, :])

    # test single frequency and not rotating the polarization
    uvtest.checkWarnings(beam2.read_cst_beam, [cst_files[0]],
                         {'beam_type': 'efield', 'telescope_name': 'TEST', 'feed_name': 'bob',
                          'feed_version': '0.1', 'model_name': 'E-field pattern - Rigging height 4.9m',
                          'model_version': '1.0', 'rotate_pol': False},
                         message='No frequency provided. Detected frequency is')

    assert beam2.pixel_coordinate_system == 'az_za'
    assert beam2.beam_type == 'efield'
    assert beam2.feed_array == np.array(['x'])
    assert beam2.data_array.shape == (2, 1, 1, 1, 181, 360)

    assert np.allclose(beam1.data_array[:, :, 0, 1, :, :], beam2.data_array[:, :, 0, 0, :, :])

    # test reading in multiple polarization files
    beam1.read_cst_beam([cst_files[0], cst_files[0]], beam_type='efield', frequency=[150e6],
                        feed_pol=['x', 'y'], telescope_name='TEST',
                        feed_name='bob', feed_version='0.1',
                        model_name='E-field pattern - Rigging height 4.9m',
                        model_version='1.0')
    assert beam1.data_array.shape == (2, 1, 2, 1, 181, 360)
    assert np.allclose(beam1.data_array[:, :, 0, :, :, :], beam1.data_array[:, :, 1, :, :, :])


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
    pytest.raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
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
    pytest.raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
                  frequency=np.array([150e6]), feed_pol='y',
                  telescope_name='TEST', feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    # use more decimal places for theta so that it is regularly gridded
    new_format = ['%15.12e', '%15.3e', '%20.3e', '%19.3e', '%19.3f', '%19.3e', '%19.3f', '%19.3e']
    np.savetxt(testfile, data, fmt=new_format, header=new_header + '\n' + line2, comments='')
    # this errors because phi isn't regularly gridded (too few sig figs)
    pytest.raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
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

    assert beam1 == beam2

    # remove a row to make data not on a grid to catch that error
    data = data[1:, :]

    new_format = ['%15.12e', '%15.12e', '%20.3e', '%19.3e', '%19.12f', '%19.3e', '%19.12f', '%19.3e']
    np.savetxt(testfile, data, fmt=new_format, header=new_header + '\n' + line2, comments='')
    # this errors because theta & phi aren't on a strict grid
    pytest.raises(ValueError, beam1.read_cst_beam, testfile, beam_type='efield',
                  frequency=np.array([150e6]), feed_pol='y',
                  telescope_name='TEST', feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')


def test_wrong_column_names():
    # need to write modified files to test headers with wrong column names
    testfile = os.path.join(DATA_PATH, 'test', 'HERA_NicCST_150MHz_modified.txt')
    with open(cst_files[0], 'r') as file:
        line1 = file.readline()
        line2 = file.readline()

    data = np.loadtxt(cst_files[0], skiprows=2)

    raw_names = line1.split(']')
    raw_names = [raw_name for raw_name in raw_names if '\n' not in raw_name]
    column_names = []
    missing_power_column_names = []
    extra_power_column_names = []
    column_names_simple = []
    units = []
    for raw_name in raw_names:
        column_name, unit = tuple(raw_name.split('['))
        column_names.append(column_name)
        column_names_simple.append(''.join(column_name.lower().split(' ')))
        units.append(unit)
        if column_name.strip() == 'Abs(V   )':
            missing_power_column_names.append('Power')
        else:
            missing_power_column_names.append(column_name)
        if column_name.strip() == 'Abs(Theta)':
            extra_power_column_names.append('Abs(E   )')
        else:
            extra_power_column_names.append(column_name)

    missing_power_column_headers = []
    for index, name in enumerate(missing_power_column_names):
        missing_power_column_headers.append(name + '[' + units[index] + ']')

    extra_power_column_headers = []
    for index, name in enumerate(extra_power_column_names):
        extra_power_column_headers.append(name + '[' + units[index] + ']')

    missing_power_header = ''
    for col in missing_power_column_headers:
        missing_power_header += '{:12}'.format(col)

    beam1 = UVBeam()

    # format to match existing file
    existing_format = ['%8.3f', '%15.3f', '%20.3e', '%19.3e', '%19.3f', '%19.3e', '%19.3f', '%19.3e']
    np.savetxt(testfile, data, fmt=existing_format, header=missing_power_header + '\n' + line2, comments='')
    # this errors because there's no recognized power column
    pytest.raises(ValueError, beam1.read_cst_beam, testfile, beam_type='power',
                  frequency=np.array([150e6]),
                  telescope_name='TEST', feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')

    extra_power_header = ''
    for col in extra_power_column_headers:
        extra_power_header += '{:12}'.format(col)
    np.savetxt(testfile, data, fmt=existing_format, header=extra_power_header + '\n' + line2, comments='')
    # this errors because there's multiple recognized power columns
    pytest.raises(ValueError, beam1.read_cst_beam, testfile, beam_type='power',
                  frequency=np.array([150e6]),
                  telescope_name='TEST', feed_name='bob', feed_version='0.1',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')


@uvtest.skipIf_no_yaml
def test_HERA_yaml():
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam1.read_cst_beam(cst_yaml_vivaldi, beam_type='efield', frequency_select=[150e6])

    assert beam1.reference_impedance == 100
    extra_keywords = {'software': 'CST 2016', 'sim_type': 'E-farfield',
                      'layout': '1 antenna', 'port_num': 1}
    assert beam1.extra_keywords == extra_keywords

    beam2.read_cst_beam(cst_yaml_vivaldi, beam_type='power', frequency_select=[150e6])

    beam1.efield_to_power(calc_cross_pols=False)

    # The values in the beam file only have 4 sig figs, so they don't match precisely
    diff = np.abs(beam1.data_array - beam2.data_array)
    assert np.max(diff) < 2
    reldiff = diff / beam2.data_array
    assert np.max(reldiff) < 0.002

    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [0.002, 0]
    beam1._data_array.tols = tols

    assert beam1.history != beam2.history
    beam1.history = beam2.history

    assert beam1 == beam2
