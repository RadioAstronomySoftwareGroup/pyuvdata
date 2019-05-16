# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvbeam object.

"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
import copy

from pyuvdata import UVBeam
import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH

try:
    import healpy as hp
    healpy_installed = True
except(ImportError):
    healpy_installed = False

filenames = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
cst_folder = 'NicCSTbeams'
cst_files = [os.path.join(DATA_PATH, cst_folder, f) for f in filenames]


@pytest.fixture(scope='function')
def uvbeam_data():
    """Setup and teardown for basic parameter, property and iterator tests."""
    required_parameters = ['_beam_type', '_Nfreqs', '_Naxes_vec', '_Nspws',
                           '_pixel_coordinate_system',
                           '_freq_array', '_spw_array',
                           '_data_normalization',
                           '_data_array', '_bandpass_array',
                           '_telescope_name', '_feed_name',
                           '_feed_version', '_model_name',
                           '_model_version', '_history',
                           '_antenna_type']

    required_properties = ['beam_type', 'Nfreqs', 'Naxes_vec', 'Nspws',
                           'pixel_coordinate_system',
                           'freq_array', 'spw_array',
                           'data_normalization',
                           'data_array', 'bandpass_array',
                           'telescope_name', 'feed_name',
                           'feed_version', 'model_name',
                           'model_version', 'history',
                           'antenna_type']

    extra_parameters = ['_Naxes1', '_Naxes2', '_Npixels', '_Nfeeds', '_Npols',
                        '_Ncomponents_vec',
                        '_axis1_array', '_axis2_array', '_nside', '_ordering',
                        '_pixel_array', '_feed_array', '_polarization_array',
                        '_basis_vector_array',
                        '_extra_keywords', '_Nelements',
                        '_element_coordinate_system',
                        '_element_location_array', '_delay_array',
                        '_x_orientation',
                        '_interpolation_function', '_freq_interp_kind',
                        '_gain_array', '_coupling_matrix',
                        '_reference_impedance',
                        '_receiver_temperature_array',
                        '_loss_array', '_mismatch_array',
                        '_s_parameters']

    extra_properties = ['Naxes1', 'Naxes2', 'Npixels', 'Nfeeds', 'Npols',
                        'Ncomponents_vec',
                        'axis1_array', 'axis2_array', 'nside', 'ordering',
                        'pixel_array', 'feed_array', 'polarization_array',
                        'basis_vector_array', 'extra_keywords', 'Nelements',
                        'element_coordinate_system',
                        'element_location_array', 'delay_array',
                        'x_orientation',
                        'interpolation_function', 'freq_interp_kind',
                        'gain_array', 'coupling_matrix',
                        'reference_impedance',
                        'receiver_temperature_array',
                        'loss_array', 'mismatch_array',
                        's_parameters']

    other_properties = ['pyuvdata_version_str']

    beam_obj = UVBeam()

    class DataHolder():
        def __init__(self, beam_obj, required_parameters, required_properties,
                     extra_parameters, extra_properties, other_properties):
            self.beam_obj = beam_obj
            self.required_parameters = required_parameters
            self.required_properties = required_properties
            self.extra_parameters = extra_parameters
            self.extra_properties = extra_properties
            self.other_properties = other_properties

    uvbeam_data = DataHolder(beam_obj, required_parameters, required_properties,
                             extra_parameters, extra_properties, other_properties)
    # yields the data we need but will continue to the del call after tests
    yield uvbeam_data

    # some post-test object cleanup
    del(uvbeam_data)

    return


def test_parameter_iter(uvbeam_data):
    """Test expected parameters."""
    all = []
    for prop in uvbeam_data.beam_obj:
        all.append(prop)
    for a in uvbeam_data.required_parameters + uvbeam_data.extra_parameters:
        assert a in all, 'expected attribute ' + a + ' not returned in object iterator'


def test_required_parameter_iter(uvbeam_data):
    """Test expected required parameters."""
    required = []
    for prop in uvbeam_data.beam_obj.required():
        required.append(prop)
    for a in uvbeam_data.required_parameters:
        assert a in required, 'expected attribute ' + a + ' not returned in required iterator'


def test_extra_parameter_iter(uvbeam_data):
    """Test expected optional parameters."""
    extra = []
    for prop in uvbeam_data.beam_obj.extra():
        extra.append(prop)
    for a in uvbeam_data.extra_parameters:
        assert a in extra, 'expected attribute ' + a + ' not returned in extra iterator'


def test_unexpected_parameters(uvbeam_data):
    """Test for extra parameters."""
    expected_parameters = uvbeam_data.required_parameters + uvbeam_data.extra_parameters
    attributes = [i for i in uvbeam_data.beam_obj.__dict__.keys() if i[0] == '_']
    for a in attributes:
        assert a in expected_parameters, 'unexpected parameter ' + a + ' found in UVBeam'


def test_unexpected_attributes(uvbeam_data):
    """Test for extra attributes."""
    expected_attributes = uvbeam_data.required_properties + \
        uvbeam_data.extra_properties + uvbeam_data.other_properties
    attributes = [i for i in uvbeam_data.beam_obj.__dict__.keys() if i[0] != '_']
    for a in attributes:
        assert a in expected_attributes, 'unexpected attribute ' + a + ' found in UVBeam'


def test_properties(uvbeam_data):
    "Test that properties can be get and set properly."
    prop_dict = dict(list(zip(uvbeam_data.required_properties + uvbeam_data.extra_properties,
                              uvbeam_data.required_parameters + uvbeam_data.extra_parameters)))
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(uvbeam_data.beam_obj, k, rand_num)
        this_param = getattr(uvbeam_data.beam_obj, v)
        try:
            assert rand_num == this_param.value
        except(AssertionError):
            print('setting {prop_name} to a random number failed'.format(prop_name=k))
            raise(AssertionError)


def test_errors():
    beam_obj = UVBeam()
    pytest.raises(ValueError, beam_obj._convert_to_filetype, 'foo')


def test_peak_normalize():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    orig_bandpass_array = copy.deepcopy(efield_beam.bandpass_array)
    maxima = np.zeros(efield_beam.Nfreqs)
    for freq_i in range(efield_beam.Nfreqs):
        maxima[freq_i] = np.amax(abs(efield_beam.data_array[:, :, :, freq_i]))
    efield_beam.peak_normalize()
    assert np.amax(abs(efield_beam.data_array)) == 1
    assert np.sum(abs(efield_beam.bandpass_array - orig_bandpass_array * maxima)) == 0
    assert efield_beam.data_normalization == 'peak'

    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    orig_bandpass_array = copy.deepcopy(power_beam.bandpass_array)
    maxima = np.zeros(efield_beam.Nfreqs)
    for freq_i in range(efield_beam.Nfreqs):
        maxima[freq_i] = np.amax(power_beam.data_array[:, :, :, freq_i])
    power_beam.peak_normalize()
    assert np.amax(abs(power_beam.data_array)) == 1
    assert np.sum(abs(power_beam.bandpass_array - orig_bandpass_array * maxima)) == 0
    assert power_beam.data_normalization == 'peak'

    power_beam.data_normalization = 'solid_angle'
    pytest.raises(NotImplementedError, power_beam.peak_normalize)


def test_stokes_matrix():
    beam = UVBeam()
    pytest.raises(ValueError, beam._stokes_matrix, -2)
    pytest.raises(ValueError, beam._stokes_matrix, 5)


@uvtest.skipIf_no_healpy
def test_efield_to_pstokes():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')

    pstokes_beam = copy.deepcopy(efield_beam)
    pstokes_beam.interpolation_function = 'az_za_simple'
    pstokes_beam.to_healpix()
    pstokes_beam.efield_to_pstokes()

    pstokes_beam_2 = copy.deepcopy(efield_beam)
    pstokes_beam_2.interpolation_function = 'az_za_simple'
    pstokes_beam_2.to_healpix()
    # convert to pstokes after interpolating
    beam_return = pstokes_beam_2.efield_to_pstokes(inplace=False)

    pstokes_beam = copy.deepcopy(efield_beam)

    # interpolate after converting to pstokes
    pstokes_beam.interpolation_function = 'az_za_simple'
    pstokes_beam.efield_to_pstokes()
    pstokes_beam.to_healpix()

    pstokes_beam.peak_normalize()
    beam_return.peak_normalize()

    # NOTE:  So far, the following doesn't hold unless the beams are peak_normalized again.
    #        This seems to be the fault of interpolation
    assert np.allclose(pstokes_beam.data_array, beam_return.data_array, atol=1e-2)

    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')
    pytest.raises(ValueError, power_beam.efield_to_pstokes)


def test_efield_to_power():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')

    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=False, inplace=False)

    # The values in the beam file only have 4 sig figs, so they don't match precisely
    diff = np.abs(new_power_beam.data_array - power_beam.data_array)
    assert np.max(diff) < 2
    reldiff = diff / power_beam.data_array
    assert np.max(reldiff) < 0.002

    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [0.002, 0]
    power_beam._data_array.tols = tols
    # modify the history to match
    power_beam.history += ' Converted from efield to power using pyuvdata.'
    assert power_beam == new_power_beam

    # test with non-orthogonal basis vectors
    # first construct a beam with non-orthogonal basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[0, 1, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, :, :, :] = efield_beam.basis_vector_array[1, :, :, :]
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, :, :, :, :, :] = np.sqrt(2) * efield_beam.data_array[0, :, :, :, :, :]
    new_data[1, :, :, :, :, :] = (efield_beam.data_array[1, :, :, :, :, :]
                                  - efield_beam.data_array[0, :, :, :, :, :])
    efield_beam2 = copy.deepcopy(efield_beam)
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = copy.deepcopy(efield_beam2)
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2

    if healpy_installed:
        # check that this raises an error if trying to convert to HEALPix:
        efield_beam2.interpolation_function = 'az_za_simple'
        pytest.raises(NotImplementedError, efield_beam2.to_healpix,
                      inplace=False)

    # now try a different rotation to non-orthogonal basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, :, :, :] = efield_beam.basis_vector_array[0, :, :, :]
    new_basis_vecs[1, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, 1, :, :] = np.sqrt(0.5)
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, :, :, :, :, :] = (efield_beam.data_array[0, :, :, :, :, :]
                                  - efield_beam.data_array[1, :, :, :, :, :])
    new_data[1, :, :, :, :, :] = np.sqrt(2) * efield_beam.data_array[1, :, :, :, :, :]
    efield_beam2 = copy.deepcopy(efield_beam)
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = copy.deepcopy(efield_beam2)
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2

    # now construct a beam with  orthogonal but rotated basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[0, 1, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, 0, :, :] = -1 * np.sqrt(0.5)
    new_basis_vecs[1, 1, :, :] = np.sqrt(0.5)
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, :, :, :, :, :] = np.sqrt(0.5) * (efield_beam.data_array[0, :, :, :, :, :]
                                                 + efield_beam.data_array[1, :, :, :, :, :])
    new_data[1, :, :, :, :, :] = np.sqrt(0.5) * (-1 * efield_beam.data_array[0, :, :, :, :, :]
                                                 + efield_beam.data_array[1, :, :, :, :, :])
    efield_beam2 = copy.deepcopy(efield_beam)
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = copy.deepcopy(efield_beam2)
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2

    # test calculating cross pols
    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=True, inplace=False)
    assert np.all(np.abs(new_power_beam.data_array[:, :, 0, :, :,
                         np.where(new_power_beam.axis1_array == 0)[0]])
                  > np.abs(new_power_beam.data_array[:, :, 2, :, :,
                           np.where(new_power_beam.axis1_array == 0)[0]]))
    assert np.all(np.abs(new_power_beam.data_array[:, :, 0, :, :,
                         np.where(new_power_beam.axis1_array == np.pi / 2.)[0]])
                  > np.abs(new_power_beam.data_array[:, :, 2, :, :,
                           np.where(new_power_beam.axis1_array == np.pi / 2.)[0]]))
    # test writing out & reading back in power files (with cross pols which are complex)
    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')
    new_power_beam.write_beamfits(write_file, clobber=True)
    new_power_beam2 = UVBeam()
    new_power_beam2.read_beamfits(write_file)
    assert new_power_beam == new_power_beam2

    # test keeping basis vectors
    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=False,
                                                 keep_basis_vector=True,
                                                 inplace=False)
    assert np.allclose(new_power_beam.data_array, np.abs(efield_beam.data_array)**2)

    # test raises error if beam is already a power beam
    pytest.raises(ValueError, power_beam.efield_to_power)

    # test raises error if input efield beam has Naxes_vec=3
    efield_beam.Naxes_vec = 3
    pytest.raises(ValueError, efield_beam.efield_to_power)


def test_freq_interpolation():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # test frequency interpolation returns data arrays for small and large tolerances
    freq_orig_vals = np.array([123e6, 150e6])
    interp_data, interp_bandpass = power_beam._interp_freq(freq_orig_vals, tol=0.0, new_object=False)
    assert isinstance(interp_data, np.ndarray)
    assert isinstance(interp_bandpass, np.ndarray)

    interp_data, interp_bandpass = power_beam._interp_freq(freq_orig_vals, tol=1.0, new_object=False)
    assert isinstance(interp_data, np.ndarray)
    assert isinstance(interp_bandpass, np.ndarray)

    # test frequency interpolation returns new UVBeam for small and large tolerances
    power_beam.saved_interp_functions = {}
    interp_data, interp_bandpass = power_beam._interp_freq(freq_orig_vals, tol=0.0, new_object=True)
    assert isinstance(interp_data, UVBeam)
    assert interp_bandpass is None
    np.testing.assert_array_almost_equal(interp_data.freq_array[0], freq_orig_vals)
    assert interp_data.freq_interp_kind == 'linear'
    assert not hasattr(interp_data, 'saved_interp_functions')  # test that saved functions are erased in new obj

    interp_data, interp_bandpass = power_beam._interp_freq(freq_orig_vals, tol=1.0, new_object=True)
    assert isinstance(interp_data, UVBeam)
    assert interp_bandpass is None
    np.testing.assert_array_almost_equal(interp_data.freq_array[0], freq_orig_vals)
    assert interp_data.freq_interp_kind == 'nearest'  # assert interp kind is 'nearest' when within tol

    # using only one freq chan should trigger a ValueError if interp_bool is True
    # unless requesting the original frequency channel such that interp_bool is False.
    # Therefore, to test that interp_bool is False returns array slice as desired,
    # test that ValueError is not raised in this case.
    # Other ways of testing this (e.g. interp_data_array.flags['OWNDATA']) does not work
    _pb = power_beam.select(frequencies=power_beam.freq_array[0, :1], inplace=False)
    try:
        interp_data_array, interp_bandp = _pb._interp_freq(_pb.freq_array[0], kind='linear')
    except ValueError:
        raise AssertionError("UVBeam._interp_freq didn't return an array slice as expected")

    # test errors if one frequency
    power_beam_singlef = power_beam.select(freq_chans=[0], inplace=False)
    pytest.raises(ValueError, power_beam_singlef._interp_freq, np.array([150e6]))

    # assert freq_interp_kind ValueError
    power_beam.interpolation_function = 'az_za_simple'
    power_beam.freq_interp_kind = None
    pytest.raises(ValueError, power_beam.interp, az_array=power_beam.axis1_array,
                  za_array=power_beam.axis2_array,
                  freq_array=freq_orig_vals, polarizations=['xx'])


def test_power_spatial_interpolation():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x', 'y'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # check that interpolating to existing points gives the same answer
    za_orig_vals, az_orig_vals = np.meshgrid(power_beam.axis2_array,
                                             power_beam.axis1_array)
    az_orig_vals = az_orig_vals.ravel(order='C')
    za_orig_vals = za_orig_vals.ravel(order='C')
    freq_orig_vals = np.array([123e6, 150e6])

    # test error if no interpolation function is set
    pytest.raises(ValueError, power_beam.interp, az_array=az_orig_vals,
                  za_array=za_orig_vals, freq_array=freq_orig_vals)

    power_beam.interpolation_function = 'az_za_simple'
    interp_data_array, interp_basis_vector = power_beam.interp(az_array=az_orig_vals,
                                                               za_array=za_orig_vals,
                                                               freq_array=freq_orig_vals)

    data_array_compare = power_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order='F')

    assert np.allclose(data_array_compare, interp_data_array)

    # test only a single polarization
    interp_data_array, interp_basis_vector = power_beam.interp(az_array=az_orig_vals,
                                                               za_array=za_orig_vals,
                                                               freq_array=freq_orig_vals,
                                                               polarizations=['xx'])

    data_array_compare = power_beam.data_array[:, :, :1]
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order='F')
    assert np.allclose(data_array_compare, interp_data_array)

    # test no errors using different points
    az_interp_vals = np.array(np.arange(0, 2 * np.pi, np.pi / 9.0).tolist()
                              + np.arange(0, 2 * np.pi, np.pi / 9.0).tolist())
    za_interp_vals = np.array((np.zeros((18)) + np.pi / 4).tolist()
                              + (np.zeros((18)) + np.pi / 12).tolist())
    freq_interp_vals = np.arange(125e6, 145e6, 5e6)

    # Test requesting separate polarizations on different calls while reusing splines.
    interp_data_array, interp_basis_vector = power_beam.interp(az_array=az_interp_vals[:2],
                                                               za_array=za_interp_vals[:2],
                                                               freq_array=freq_interp_vals, polarizations=['xx'], reuse_spline=True)

    interp_data_array, interp_basis_vector = power_beam.interp(az_array=az_interp_vals[:2],
                                                               za_array=za_interp_vals[:2],
                                                               freq_array=freq_interp_vals, polarizations=['yy'], reuse_spline=True)

    # test reusing the spline fit.
    orig_data_array, interp_basis_vector = power_beam.interp(az_array=az_interp_vals,
                                                             za_array=za_interp_vals,
                                                             freq_array=freq_interp_vals, reuse_spline=True)

    reused_data_array, interp_basis_vector = power_beam.interp(az_array=az_interp_vals,
                                                               za_array=za_interp_vals,
                                                               freq_array=freq_interp_vals, reuse_spline=True)

    assert np.all(reused_data_array == orig_data_array)
    del power_beam.saved_interp_functions

    # test errors if frequency interp values outside range
    pytest.raises(ValueError, power_beam.interp, az_array=az_interp_vals,
                  za_array=za_interp_vals, freq_array=np.array([100]))

    # test errors if positions outside range
    power_beam.select(axis2_inds=np.where(power_beam.axis2_array <= np.pi / 2.)[0])
    pytest.raises(ValueError, power_beam.interp, az_array=az_interp_vals,
                  za_array=za_interp_vals + np.pi / 2)

    # test no errors only frequency interpolation
    interp_data_array, interp_basis_vector = power_beam.interp(freq_array=freq_interp_vals)

    # assert polarization value error
    pytest.raises(ValueError, power_beam.interp, az_array=az_interp_vals,
                  za_array=za_interp_vals,
                  polarizations=['pI'])


def test_efield_spatial_interpolation():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')

    za_orig_vals, az_orig_vals = np.meshgrid(efield_beam.axis2_array,
                                             efield_beam.axis1_array)
    az_orig_vals = az_orig_vals.ravel(order='C')
    za_orig_vals = za_orig_vals.ravel(order='C')
    freq_orig_vals = np.array([123e6, 150e6])
    efield_beam.interpolation_function = 'az_za_simple'
    interp_data_array, interp_basis_vector = efield_beam.interp(az_array=az_orig_vals,
                                                                za_array=za_orig_vals,
                                                                freq_array=freq_orig_vals)

    interp_data_array = interp_data_array.reshape(efield_beam.data_array.shape, order='F')
    interp_basis_vector = interp_basis_vector.reshape(efield_beam.basis_vector_array.shape, order='F')

    assert np.allclose(efield_beam.data_array, interp_data_array)
    assert np.allclose(efield_beam.basis_vector_array, interp_basis_vector)

    # test no errors using different points
    az_interp_vals = np.array(np.arange(0, 2 * np.pi, np.pi / 9.0).tolist()
                              + np.arange(0, 2 * np.pi, np.pi / 9.0).tolist())
    za_interp_vals = np.array((np.zeros((18)) + np.pi / 4).tolist()
                              + (np.zeros((18)) + np.pi / 12).tolist())
    freq_interp_vals = np.arange(125e6, 145e6, 10e6)

    interp_data_array, interp_basis_vector = efield_beam.interp(az_array=az_interp_vals,
                                                                za_array=za_interp_vals,
                                                                freq_array=freq_interp_vals)

    # test reusing the spline fit
    orig_data_array, interp_basis_vector = efield_beam.interp(az_array=az_interp_vals,
                                                              za_array=za_interp_vals,
                                                              freq_array=freq_interp_vals, reuse_spline=True)

    reused_data_array, interp_basis_vector = efield_beam.interp(az_array=az_interp_vals,
                                                                za_array=za_interp_vals,
                                                                freq_array=freq_interp_vals, reuse_spline=True)

    assert np.all(reused_data_array == orig_data_array)

    select_data_array_orig, interp_basis_vector = efield_beam.interp(az_array=az_interp_vals[0:1],
                                                                     za_array=za_interp_vals[0:1],
                                                                     freq_array=np.array([127e6]))

    select_data_array_reused, interp_basis_vector = efield_beam.interp(az_array=az_interp_vals[0:1],
                                                                       za_array=za_interp_vals[0:1],
                                                                       freq_array=np.array([127e6]), reuse_spline=True)
    assert np.allclose(select_data_array_orig, select_data_array_reused)
    del efield_beam.saved_interp_functions


def test_interp_longitude_branch_cut():
    beam = UVBeam()
    beam.read_cst_beam(cst_files, beam_type='power', frequency=[150e6, 123e6],
                       telescope_name='TEST', feed_name='bob',
                       feed_version='0.1', feed_pol=['x'],
                       model_name='E-field pattern - Rigging height 4.9m',
                       model_version='1.0')

    beam.interpolation_function = 'az_za_simple'
    interp_data_array, interp_basis_vector = beam.interp(
        az_array=np.deg2rad(np.repeat(np.array([[-1], [359], [0], [360]]), 181, axis=1).flatten()),
        za_array=np.repeat(beam.axis2_array[np.newaxis, :], 4, axis=0).flatten())

    interp_data_array = interp_data_array.reshape(beam.Naxes_vec, beam.Nspws,
                                                  beam.Npols, beam.Nfreqs,
                                                  4, beam.Naxes2)

    assert(np.allclose(interp_data_array[:, :, :, :, 0, :],
                       interp_data_array[:, :, :, :, 1, :],
                       rtol=beam._data_array.tols[0],
                       atol=beam._data_array.tols[1]))

    assert(np.allclose(interp_data_array[:, :, :, :, 2, :],
                       interp_data_array[:, :, :, :, 3, :],
                       rtol=beam._data_array.tols[0],
                       atol=beam._data_array.tols[1]))

    # repeat with efield
    beam = UVBeam()
    beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                       telescope_name='TEST', feed_name='bob',
                       feed_version='0.1', feed_pol=['x'],
                       model_name='E-field pattern - Rigging height 4.9m',
                       model_version='1.0')

    beam.interpolation_function = 'az_za_simple'
    interp_data_array, interp_basis_vector = beam.interp(
        az_array=np.deg2rad(np.repeat(np.array([[-1], [359], [0], [360]]), 181, axis=1).flatten()),
        za_array=np.repeat(beam.axis2_array[np.newaxis, :], 4, axis=0).flatten())

    interp_data_array = interp_data_array.reshape(beam.Naxes_vec, beam.Nspws,
                                                  beam.Nfeeds, beam.Nfreqs,
                                                  4, beam.Naxes2)

    assert(np.allclose(interp_data_array[:, :, :, :, 0, :],
                       interp_data_array[:, :, :, :, 1, :],
                       rtol=beam._data_array.tols[0],
                       atol=beam._data_array.tols[1]))

    assert(np.allclose(interp_data_array[:, :, :, :, 2, :],
                       interp_data_array[:, :, :, :, 3, :],
                       rtol=beam._data_array.tols[0],
                       atol=beam._data_array.tols[1]))


@uvtest.skipIf_no_healpy
def test_healpix_interpolation():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files, beam_type='efield', frequency=[150e6, 123e6],
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')
    power_beam.interpolation_function = 'az_za_simple'
    power_beam.to_healpix()

    # check that interpolating to existing points gives the same answer
    power_beam.interpolation_function = 'healpix_simple'
    za_orig_vals, az_orig_vals = hp.pix2ang(power_beam.nside, np.arange(hp.nside2npix(power_beam.nside)))
    az_orig_vals = az_orig_vals.ravel(order='C')
    za_orig_vals = za_orig_vals.ravel(order='C')
    freq_orig_vals = np.array([123e6, 150e6])

    interp_data_array, interp_basis_vector = power_beam.interp(az_array=az_orig_vals,
                                                               za_array=za_orig_vals,
                                                               freq_array=freq_orig_vals)
    data_array_compare = power_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order='F')
    assert np.allclose(data_array_compare, interp_data_array)

    # basis_vector exception
    power_beam.basis_vector_array[0, 1, :] = 10.0
    pytest.raises(NotImplementedError, power_beam.interp, az_array=az_orig_vals, za_array=za_orig_vals)

    # now convert to power beam
    power_beam.efield_to_power()
    interp_data_array, interp_basis_vector = power_beam.interp(az_array=az_orig_vals,
                                                               za_array=za_orig_vals,
                                                               freq_array=freq_orig_vals)
    data_array_compare = power_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order='F')
    assert np.allclose(data_array_compare, interp_data_array)

    # assert not feeding frequencies gives same answer
    interp_data_array2, interp_basis_vector2 = power_beam.interp(az_array=az_orig_vals, za_array=za_orig_vals)
    assert np.allclose(interp_data_array, interp_data_array2)

    # assert not feeding az_array gives same answer
    interp_data_array2, interp_basis_vector2 = power_beam.interp(az_array=az_orig_vals, za_array=za_orig_vals)
    assert np.allclose(interp_data_array, interp_data_array2)

    # test requesting polarization gives the same answer
    interp_data_array2, interp_basis_vector2 = power_beam.interp(az_array=az_orig_vals, za_array=za_orig_vals,
                                                                 polarizations=['yy'])
    assert np.allclose(interp_data_array[:, :, 1:2], interp_data_array2[:, :, :1])

    # change complex data_array to real data_array and test again
    power_beam.data_array = np.abs(power_beam.data_array)
    interp_data_array, interp_basis_vector = power_beam.interp(az_array=az_orig_vals,
                                                               za_array=za_orig_vals,
                                                               freq_array=freq_orig_vals)
    data_array_compare = power_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order='F')
    assert np.allclose(data_array_compare, interp_data_array)

    # test no inputs equals same answer
    interp_data_array2, interp_basis_vector2 = power_beam.interp()
    assert np.allclose(interp_data_array, interp_data_array2)

    # assert polarization value error
    pytest.raises(ValueError, power_beam.interp, az_array=az_orig_vals, za_array=za_orig_vals,
                  polarizations=['pI'])

    # healpix coord exception
    power_beam.pixel_coordinate_system = 'foo'
    pytest.raises(ValueError, power_beam.interp, az_array=az_orig_vals, za_array=za_orig_vals)


@uvtest.skipIf_no_healpy
def test_to_healpix():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    power_beam.select(axis2_inds=np.where(power_beam.axis2_array <= np.pi / 2.)[0])

    power_beam.interpolation_function = 'az_za_simple'
    power_beam_healpix = power_beam.to_healpix(inplace=False)

    # check that history is updated appropriately
    assert power_beam_healpix.history == (power_beam.history
                                          + ' Interpolated from regularly gridded '
                                          'azimuth/zenith_angle to HEALPix using pyuvdata '
                                          'with interpolation_function = az_za_simple.')

    npix = hp.nside2npix(power_beam_healpix.nside)
    assert power_beam_healpix.Npixels <= npix * 0.55

    # Test error if not az_za
    power_beam.pixel_coordinate_system = 'sin_zenith'
    pytest.raises(ValueError, power_beam.to_healpix)

    # Now check Efield interpolation
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    efield_beam.interpolation_function = 'az_za_simple'
    interp_then_sq = efield_beam.to_healpix(inplace=False)
    interp_then_sq.efield_to_power(calc_cross_pols=False)

    # convert to power and then interpolate to compare.
    # Don't use power read from file because it has rounding errors that will dominate this comparison
    sq_then_interp = efield_beam.efield_to_power(calc_cross_pols=False, inplace=False)
    sq_then_interp.to_healpix()

    # square then interpolate is different from interpolate then square at a
    # higher level than normally allowed in the equality.
    # We can live with it for now, may need to improve it later
    diff = np.abs(interp_then_sq.data_array - sq_then_interp.data_array)
    assert np.max(diff) < 0.5
    reldiff = diff / sq_then_interp.data_array
    assert np.max(reldiff) < 0.05

    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [0.05, 0]
    sq_then_interp._data_array.tols = tols

    # check history changes
    interp_history_add = (' Interpolated from regularly gridded '
                          'azimuth/zenith_angle to HEALPix using pyuvdata '
                          'with interpolation_function = az_za_simple.')
    sq_history_add = ' Converted from efield to power using pyuvdata.'
    assert sq_then_interp.history == efield_beam.history + sq_history_add + interp_history_add
    assert interp_then_sq.history == efield_beam.history + interp_history_add + sq_history_add

    # now change history on one so we can compare the rest of the object
    sq_then_interp.history = efield_beam.history + interp_history_add + sq_history_add

    assert sq_then_interp == interp_then_sq


def test_select_axis():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_impedance = 340.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    old_history = power_beam.history
    # Test selecting on axis1
    inds1_to_keep = np.arange(14, 63)

    power_beam2 = power_beam.select(axis1_inds=inds1_to_keep, inplace=False)

    assert len(inds1_to_keep) == power_beam2.Naxes1
    for i in inds1_to_keep:
        assert power_beam.axis1_array[i] in power_beam2.axis1_array
    for i in np.unique(power_beam2.axis1_array):
        assert i in power_beam.axis1_array

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific parts of first image axis '
                                    'using pyuvdata.', power_beam2.history)

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

    # test writing beamfits with only one element in axis1
    inds_to_keep = [len(inds1_to_keep) + 1]
    power_beam2 = power_beam.select(axis1_inds=inds_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with indices not included in data
    pytest.raises(ValueError, power_beam2.select, axis1_inds=[power_beam.Naxes1 - 1])

    # check for warnings and errors associated with unevenly spaced image pixels
    power_beam2 = copy.deepcopy(power_beam)
    uvtest.checkWarnings(power_beam2.select, [], {'axis1_inds': [0, 5, 6]},
                         message='Selected values along first image axis are not evenly spaced')
    pytest.raises(ValueError, power_beam2.write_beamfits, write_file_beamfits)

    # Test selecting on axis2
    inds2_to_keep = np.arange(5, 14)

    power_beam2 = power_beam.select(axis2_inds=inds2_to_keep, inplace=False)

    assert len(inds2_to_keep) == power_beam2.Naxes2
    for i in inds2_to_keep:
        assert power_beam.axis2_array[i] in power_beam2.axis2_array
    for i in np.unique(power_beam2.axis2_array):
        assert i in power_beam.axis2_array

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific parts of second image axis '
                                    'using pyuvdata.', power_beam2.history)

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

    # test writing beamfits with only one element in axis2
    inds_to_keep = [len(inds2_to_keep) + 1]
    power_beam2 = power_beam.select(axis2_inds=inds_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with indices not included in data
    pytest.raises(ValueError, power_beam2.select, axis2_inds=[power_beam.Naxes2 - 1])

    # check for warnings and errors associated with unevenly spaced image pixels
    power_beam2 = copy.deepcopy(power_beam)
    uvtest.checkWarnings(power_beam2.select, [], {'axis2_inds': [0, 5, 6]},
                         message='Selected values along second image axis are not evenly spaced')
    pytest.raises(ValueError, power_beam2.write_beamfits, write_file_beamfits)


def test_select_frequencies():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding several times
    while power_beam.Nfreqs < 8:
        new_beam = copy.deepcopy(power_beam)
        new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
        power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_impedance = 340.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    old_history = power_beam.history
    freqs_to_keep = power_beam.freq_array[0, np.arange(2, 7)]

    power_beam2 = power_beam.select(frequencies=freqs_to_keep, inplace=False)

    assert len(freqs_to_keep) == power_beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    power_beam2.history)

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')
    # test writing beamfits with only one frequency

    freqs_to_keep = power_beam.freq_array[0, 5]
    power_beam2 = power_beam.select(frequencies=freqs_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with frequencies not included in data
    pytest.raises(ValueError, power_beam.select, frequencies=[np.max(power_beam.freq_array) + 10])

    # check for warnings and errors associated with unevenly spaced frequencies
    power_beam2 = copy.deepcopy(power_beam)
    uvtest.checkWarnings(power_beam2.select, [],
                         {'frequencies': power_beam2.freq_array[0, [0, 5, 6]]},
                         message='Selected frequencies are not evenly spaced')
    pytest.raises(ValueError, power_beam2.write_beamfits, write_file_beamfits)

    # Test selecting on freq_chans
    chans_to_keep = np.arange(2, 7)

    power_beam2 = power_beam.select(freq_chans=chans_to_keep, inplace=False)

    assert len(chans_to_keep) == power_beam2.Nfreqs
    for chan in chans_to_keep:
        assert power_beam.freq_array[0, chan] in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in power_beam.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    power_beam2.history)

    # Test selecting both channels and frequencies
    freqs_to_keep = power_beam.freq_array[0, np.arange(6, 8)]  # Overlaps with chans
    all_chans_to_keep = np.arange(2, 8)

    power_beam2 = power_beam.select(frequencies=freqs_to_keep,
                                    freq_chans=chans_to_keep,
                                    inplace=False)

    assert len(all_chans_to_keep) == power_beam2.Nfreqs
    for chan in all_chans_to_keep:
        assert power_beam.freq_array[0, chan] in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in power_beam.freq_array[0, all_chans_to_keep]


def test_select_feeds():
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {'KEY1': 'test_keyword'}
    efield_beam.reference_impedance = 340.
    efield_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.loss_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs))

    old_history = efield_beam.history
    feeds_to_keep = ['x']

    efield_beam2 = efield_beam.select(feeds=feeds_to_keep, inplace=False)

    assert len(feeds_to_keep) == efield_beam2.Nfeeds
    for f in feeds_to_keep:
        assert f in efield_beam2.feed_array
    for f in np.unique(efield_beam2.feed_array):
        assert f in feeds_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific feeds using pyuvdata.',
                                    efield_beam2.history)

    # check for errors associated with feeds not included in data
    pytest.raises(ValueError, efield_beam.select, feeds=['N'])

    # check for error with selecting polarizations on efield beams
    pytest.raises(ValueError, efield_beam.select, polarizations=[-5, -6])

    # Test check basis vectors
    efield_beam.basis_vector_array[0, 1, :, :] = 1.0
    pytest.raises(ValueError, efield_beam.check)

    efield_beam.basis_vector_array[0, 0, :, :] = np.sqrt(0.5)
    efield_beam.basis_vector_array[0, 1, :, :] = np.sqrt(0.5)
    assert efield_beam.check()

    efield_beam.basis_vector_array = None
    pytest.raises(ValueError, efield_beam.check)


def test_select_polarizations():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol='xx',
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more polarizations for testing by copying and adding several times
    while power_beam.Npols < 4:
        new_beam = copy.deepcopy(power_beam)
        new_beam.polarization_array = power_beam.polarization_array - power_beam.Npols
        power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_impedance = 340.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    old_history = power_beam.history
    pols_to_keep = [-5, -6]

    power_beam2 = power_beam.select(polarizations=pols_to_keep,
                                    inplace=False)

    assert len(pols_to_keep) == power_beam2.Npols
    for p in pols_to_keep:
        assert p in power_beam2.polarization_array
    for p in np.unique(power_beam2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific polarizations using pyuvdata.',
                                    power_beam2.history)

    # check for errors associated with polarizations not included in data
    pytest.raises(ValueError, power_beam.select, polarizations=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    uvtest.checkWarnings(power_beam.select, [], {'polarizations': power_beam.polarization_array[[0, 1, 3]]},
                         message='Selected polarizations are not evenly spaced')
    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')
    pytest.raises(ValueError, power_beam.write_beamfits, write_file_beamfits)

    # check for error with selecting on feeds on power beams
    pytest.raises(ValueError, power_beam.select, feeds=['x'])


def test_select():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(power_beam)
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_impedance = 340.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    # now test selecting along all axes at once
    old_history = power_beam.history

    inds1_to_keep = np.arange(14, 63)
    inds2_to_keep = np.arange(5, 14)
    freqs_to_keep = [power_beam.freq_array[0, 0]]
    pols_to_keep = [-5]

    power_beam2 = power_beam.select(axis1_inds=inds1_to_keep,
                                    axis2_inds=inds2_to_keep,
                                    frequencies=freqs_to_keep,
                                    polarizations=pols_to_keep,
                                    inplace=False)

    assert len(inds1_to_keep) == power_beam2.Naxes1
    for i in inds1_to_keep:
        assert power_beam.axis1_array[i] in power_beam2.axis1_array
    for i in np.unique(power_beam2.axis1_array):
        assert i in power_beam.axis1_array

    assert len(inds2_to_keep) == power_beam2.Naxes2
    for i in inds2_to_keep:
        assert power_beam.axis2_array[i] in power_beam2.axis2_array
    for i in np.unique(power_beam2.axis2_array):
        assert i in power_beam.axis2_array

    assert len(freqs_to_keep) == power_beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in freqs_to_keep

    assert len(pols_to_keep) == power_beam2.Npols
    for p in pols_to_keep:
        assert p in power_beam2.polarization_array
    for p in np.unique(power_beam2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific parts of first image axis, '
                                    'parts of second image axis, '
                                    'frequencies, polarizations using pyuvdata.',
                                    power_beam2.history)

    # repeat for efield beam
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(efield_beam)
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {'KEY1': 'test_keyword'}
    efield_beam.reference_impedance = 340.
    efield_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.loss_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs))

    feeds_to_keep = ['x']

    efield_beam2 = efield_beam.select(axis1_inds=inds1_to_keep,
                                      axis2_inds=inds2_to_keep,
                                      frequencies=freqs_to_keep,
                                      feeds=feeds_to_keep,
                                      inplace=False)

    assert len(inds1_to_keep) == efield_beam2.Naxes1
    for i in inds1_to_keep:
        assert efield_beam.axis1_array[i] in efield_beam2.axis1_array
    for i in np.unique(efield_beam2.axis1_array):
        assert i in efield_beam.axis1_array

    assert len(inds2_to_keep) == efield_beam2.Naxes2
    for i in inds2_to_keep:
        assert efield_beam.axis2_array[i] in efield_beam2.axis2_array
    for i in np.unique(efield_beam2.axis2_array):
        assert i in efield_beam.axis2_array

    assert len(freqs_to_keep) == efield_beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in efield_beam2.freq_array
    for f in np.unique(efield_beam2.freq_array):
        assert f in freqs_to_keep

    assert len(feeds_to_keep) == efield_beam2.Nfeeds
    for f in feeds_to_keep:
        assert f in efield_beam2.feed_array
    for f in np.unique(efield_beam2.feed_array):
        assert f in feeds_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific parts of first image axis, '
                                    'parts of second image axis, '
                                    'frequencies, feeds using pyuvdata.',
                                    efield_beam2.history)


def test_add():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(power_beam)
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_impedance = 340.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    # Add along first image axis
    beam1 = power_beam.select(axis1_inds=np.arange(0, 180), inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(180, 360), inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(power_beam.history
                                    + '  Downselected to specific parts of '
                                    'first image axis using pyuvdata. '
                                    'Combined data along first image axis '
                                    'using pyuvdata.', beam1.history)
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - axis1
    beam1 = power_beam.select(axis1_inds=np.arange(180, 360), inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(0, 180), inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Add along second image axis
    beam1 = power_beam.select(axis2_inds=np.arange(0, 90), inplace=False)
    beam2 = power_beam.select(axis2_inds=np.arange(90, 181), inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(power_beam.history
                                    + '  Downselected to specific parts of '
                                    'second image axis using pyuvdata. '
                                    'Combined data along second image axis '
                                    'using pyuvdata.', beam1.history)
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - axis2
    beam1 = power_beam.select(axis2_inds=np.arange(90, 181), inplace=False)
    beam2 = power_beam.select(axis2_inds=np.arange(0, 90), inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Add frequencies
    beam1 = power_beam.select(freq_chans=0, inplace=False)
    beam2 = power_beam.select(freq_chans=1, inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(power_beam.history
                                    + '  Downselected to specific frequencies '
                                    'using pyuvdata. Combined data along '
                                    'frequency axis using pyuvdata.',
                                    beam1.history)
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - freqs
    beam1 = power_beam.select(freq_chans=1, inplace=False)
    beam2 = power_beam.select(freq_chans=0, inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Add polarizations
    beam1 = power_beam.select(polarizations=-5, inplace=False)
    beam2 = power_beam.select(polarizations=-6, inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(power_beam.history
                                    + '  Downselected to specific polarizations '
                                    'using pyuvdata. Combined data along '
                                    'polarization axis using pyuvdata.',
                                    beam1.history)
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - pols
    beam1 = power_beam.select(polarizations=-6, inplace=False)
    beam2 = power_beam.select(polarizations=-5, inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Add feeds
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(efield_beam)
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {'KEY1': 'test_keyword'}
    efield_beam.reference_impedance = 340.
    efield_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.loss_array = np.random.normal(50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs))
    efield_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs))

    beam1 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    beam2 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(efield_beam.history
                                    + '  Downselected to specific feeds '
                                    'using pyuvdata. Combined data along '
                                    'feed axis using pyuvdata.',
                                    beam1.history)
    beam1.history = efield_beam.history
    assert beam1 == efield_beam

    # Out of order - feeds
    beam1 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    beam2 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    beam1 += beam2
    beam1.history = efield_beam.history
    assert beam1, efield_beam

    # Add multiple axes
    beam_ref = copy.deepcopy(power_beam)
    beam1 = power_beam.select(axis1_inds=np.arange(0, power_beam.Naxes1 // 2),
                              polarizations=power_beam.polarization_array[0],
                              inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(power_beam.Naxes1 // 2,
                                                   power_beam.Naxes1),
                              polarizations=power_beam.polarization_array[1],
                              inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(power_beam.history
                                    + '  Downselected to specific parts of '
                                    'first image axis, polarizations using '
                                    'pyuvdata. Combined data along first '
                                    'image, polarization axis using pyuvdata.',
                                    beam1.history)
    # Zero out missing data in reference object
    beam_ref.data_array[:, :, 0, :, :, power_beam.Naxes1 // 2:] = 0.0
    beam_ref.data_array[:, :, 1, :, :, :power_beam.Naxes1 // 2] = 0.0
    beam1.history = power_beam.history
    assert beam1 == beam_ref

    # Another combo with efield
    beam_ref = copy.deepcopy(efield_beam)
    beam1 = efield_beam.select(axis1_inds=np.arange(0, efield_beam.Naxes1 // 2),
                               axis2_inds=np.arange(0, efield_beam.Naxes2 // 2),
                               inplace=False)
    beam2 = efield_beam.select(axis1_inds=np.arange(efield_beam.Naxes1 // 2,
                                                    efield_beam.Naxes1),
                               axis2_inds=np.arange(efield_beam.Naxes2 // 2,
                                                    efield_beam.Naxes2),
                               inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(efield_beam.history
                                    + '  Downselected to specific parts of '
                                    'first image axis, parts of second '
                                    'image axis using pyuvdata. Combined '
                                    'data along first image, second image '
                                    'axis using pyuvdata.',
                                    beam1.history)

    # Zero out missing data in reference object
    beam_ref.data_array[:, :, :, :, :efield_beam.Naxes2 // 2,
                        efield_beam.Naxes1 // 2:] = 0.0
    beam_ref.data_array[:, :, :, :, efield_beam.Naxes2 // 2:,
                        :efield_beam.Naxes1 // 2] = 0.0

    beam_ref.basis_vector_array[:, :, :efield_beam.Naxes2 // 2,
                                efield_beam.Naxes1 // 2:] = 0.0
    beam_ref.basis_vector_array[:, :, efield_beam.Naxes2 // 2:,
                                :efield_beam.Naxes1 // 2] = 0.0
    beam1.history = efield_beam.history
    assert beam1, beam_ref

    # Check warnings
    # generate more frequencies for testing by copying and adding several times
    while power_beam.Nfreqs < 8:
        new_beam = copy.deepcopy(power_beam)
        new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
        power_beam += new_beam

    beam1 = power_beam.select(freq_chans=np.arange(0, 4), inplace=False)
    beam2 = power_beam.select(freq_chans=np.arange(5, 8), inplace=False)
    uvtest.checkWarnings(beam1.__add__, [beam2],
                         message='Combined frequencies are not evenly spaced')

    # generate more polarizations for testing by copying and adding several times
    while power_beam.Npols < 4:
        new_beam = copy.deepcopy(power_beam)
        new_beam.polarization_array = power_beam.polarization_array - power_beam.Npols
        power_beam += new_beam

    power_beam.receiver_temperature_array = np.ones((1, 8))
    beam1 = power_beam.select(polarizations=power_beam.polarization_array[0:2],
                              inplace=False)
    beam2 = power_beam.select(polarizations=power_beam.polarization_array[3],
                              inplace=False)
    uvtest.checkWarnings(beam1.__iadd__, [beam2],
                         message='Combined polarizations are not evenly spaced')

    beam1 = power_beam.select(polarizations=power_beam.polarization_array[0:2],
                              inplace=False)
    beam2 = power_beam.select(polarizations=power_beam.polarization_array[2:3],
                              inplace=False)
    beam2.receiver_temperature_array = None
    assert beam1.receiver_temperature_array is not None
    uvtest.checkWarnings(beam1.__iadd__, [beam2],
                         message=['Only one of the UVBeam objects being combined '
                                  'has optional parameter'])
    assert beam1.receiver_temperature_array is None

    # Combining histories
    beam1 = power_beam.select(polarizations=power_beam.polarization_array[0:2], inplace=False)
    beam2 = power_beam.select(polarizations=power_beam.polarization_array[2:4], inplace=False)
    beam2.history += ' testing the history. Read/written with pyuvdata'
    beam1 += beam2
    assert uvutils._check_histories(power_beam.history
                                    + '  Downselected to specific polarizations '
                                    'using pyuvdata. Combined data along '
                                    'polarization axis using pyuvdata. '
                                    'testing the history.',
                                    beam1.history)
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # ------------------------
    # Test failure modes of add function

    # Wrong class
    beam1 = copy.deepcopy(power_beam)
    pytest.raises(ValueError, beam1.__iadd__, np.zeros(5))

    params_to_change = {'beam_type': 'efield', 'data_normalization': 'solid_angle',
                        'telescope_name': 'foo', 'feed_name': 'foo',
                        'feed_version': 'v12', 'model_name': 'foo',
                        'model_version': 'v12', 'pixel_coordinate_system': 'sin_zenith',
                        'Naxes_vec': 3, 'nside': 16, 'ordering': 'nested'}

    beam1 = power_beam.select(freq_chans=0, inplace=False)
    for param, value in params_to_change.items():
        beam2 = power_beam.select(freq_chans=1, inplace=False)
        setattr(beam2, param, value)
        pytest.raises(ValueError, beam1.__iadd__, beam2)

    # Overlapping data
    beam2 = copy.deepcopy(power_beam)
    pytest.raises(ValueError, beam1.__iadd__, beam2)


@uvtest.skipIf_no_healpy
def test_healpix():
    # put all the testing on healpix in this one function to minimize slow calls
    # to uvbeam.to_healpix()
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1', feed_pol=['x'],
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(power_beam)
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {'KEY1': 'test_keyword'}
    power_beam.reference_impedance = 340.
    power_beam.receiver_temperature_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.loss_array = np.random.normal(50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.mismatch_array = np.random.normal(0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs))
    power_beam.s_parameters = np.random.normal(0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs))

    power_beam.interpolation_function = 'az_za_simple'
    power_beam_healpix = power_beam.to_healpix(inplace=False)

    # test that Npixels make sense
    n_max_pix = power_beam.Naxes1 * power_beam.Naxes2
    assert power_beam_healpix.Npixels <= n_max_pix

    # -----------------------
    # test selecting on pixels
    old_history = power_beam_healpix.history
    pixels_to_keep = np.arange(31, 184)

    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep, inplace=False)

    assert len(pixels_to_keep) == power_beam_healpix2.Npixels
    for pi in pixels_to_keep:
        assert pi in power_beam_healpix2.pixel_array
    for pi in np.unique(power_beam_healpix2.pixel_array):
        assert pi in pixels_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific healpix pixels using pyuvdata.',
                                    power_beam_healpix2.history)

    write_file_beamfits = os.path.join(DATA_PATH, 'test/select_beam.fits')

    # test writing beamfits with only one pixel
    pixels_to_keep = [43]
    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep, inplace=False)
    power_beam_healpix2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with pixels not included in data
    pytest.raises(ValueError, power_beam_healpix.select,
                  pixels=[12 * power_beam_healpix.nside**2 + 10])

    # test writing beamfits with non-contiguous pixels
    pixels_to_keep = np.arange(2, 150, 4)

    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep, inplace=False)
    power_beam_healpix2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors selecting pixels on non-healpix beams
    pytest.raises(ValueError, power_beam.select, pixels=pixels_to_keep)

    # -----------------
    # check for errors selecting axis1_inds on healpix beams
    inds1_to_keep = np.arange(14, 63)
    pytest.raises(ValueError, power_beam_healpix.select, axis1_inds=inds1_to_keep)

    # check for errors selecting axis2_inds on healpix beams
    inds2_to_keep = np.arange(5, 14)
    pytest.raises(ValueError, power_beam_healpix.select, axis2_inds=inds2_to_keep)

    # ------------------------
    # test selecting along all axes at once for healpix beams
    freqs_to_keep = [power_beam_healpix.freq_array[0, 0]]
    pols_to_keep = [-5]

    power_beam_healpix2 = power_beam_healpix.select(pixels=pixels_to_keep,
                                                    frequencies=freqs_to_keep,
                                                    polarizations=pols_to_keep,
                                                    inplace=False)

    assert len(pixels_to_keep) == power_beam_healpix2.Npixels
    for pi in pixels_to_keep:
        assert pi in power_beam_healpix2.pixel_array
    for pi in np.unique(power_beam_healpix2.pixel_array):
        assert pi in pixels_to_keep

    assert len(freqs_to_keep) == power_beam_healpix2.Nfreqs
    for f in freqs_to_keep:
        assert f in power_beam_healpix2.freq_array
    for f in np.unique(power_beam_healpix2.freq_array):
        assert f in freqs_to_keep

    assert len(pols_to_keep) == power_beam_healpix2.Npols
    for p in pols_to_keep:
        assert p in power_beam_healpix2.polarization_array
    for p in np.unique(power_beam_healpix2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific healpix pixels, frequencies, '
                                    'polarizations using pyuvdata.',
                                    power_beam_healpix2.history)

    # repeat for efield beam
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1', feed_pol=['x'],
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')
    # generate more frequencies for testing by copying and adding
    new_beam = copy.deepcopy(efield_beam)
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    efield_beam.interpolation_function = 'az_za_simple'
    efield_beam.to_healpix()
    old_history = efield_beam.history

    freqs_to_keep = np.array([efield_beam.freq_array[0, 0]])
    feeds_to_keep = ['x']

    efield_beam2 = efield_beam.select(pixels=pixels_to_keep,
                                      frequencies=freqs_to_keep,
                                      feeds=feeds_to_keep,
                                      inplace=False)

    assert len(pixels_to_keep) == efield_beam2.Npixels
    for pi in pixels_to_keep:
        assert pi in efield_beam2.pixel_array
    for pi in np.unique(efield_beam2.pixel_array):
        assert pi in pixels_to_keep

    assert freqs_to_keep.size == efield_beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in efield_beam2.freq_array
    for f in np.unique(efield_beam2.freq_array):
        assert f in freqs_to_keep

    assert len(feeds_to_keep) == efield_beam2.Nfeeds
    for f in feeds_to_keep:
        assert f in efield_beam2.feed_array
    for f in np.unique(efield_beam2.feed_array):
        assert f in feeds_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific healpix pixels, frequencies, '
                                    'feeds using pyuvdata.',
                                    efield_beam2.history)

    # -------------------
    # Test adding a different combo with healpix
    beam_ref = copy.deepcopy(power_beam_healpix)
    beam1 = power_beam_healpix.select(
        pixels=power_beam_healpix.pixel_array[0:power_beam_healpix.Npixels // 2],
        freq_chans=0, inplace=False)
    beam2 = power_beam_healpix.select(
        pixels=power_beam_healpix.pixel_array[power_beam_healpix.Npixels // 2:],
        freq_chans=1, inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(power_beam_healpix.history
                                    + '  Downselected to specific healpix '
                                    'pixels, frequencies using pyuvdata. '
                                    'Combined data along healpix pixel, '
                                    'frequency axis using pyuvdata.',
                                    beam1.history)
    # Zero out missing data in reference object
    beam_ref.data_array[:, :, :, 0, power_beam_healpix.Npixels // 2:] = 0.0
    beam_ref.data_array[:, :, :, 1, :power_beam_healpix.Npixels // 2] = 0.0
    beam1.history = power_beam_healpix.history
    assert beam1 == beam_ref

    # Test adding another combo with efield
    beam_ref = copy.deepcopy(efield_beam)
    beam1 = efield_beam.select(freq_chans=0, feeds=efield_beam.feed_array[0],
                               inplace=False)
    beam2 = efield_beam.select(freq_chans=1, feeds=efield_beam.feed_array[1],
                               inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(efield_beam.history
                                    + '  Downselected to specific frequencies, '
                                    'feeds using pyuvdata. Combined data '
                                    'along frequency, feed axis using pyuvdata.',
                                    beam1.history)
    # Zero out missing data in reference object
    beam_ref.data_array[:, :, 1, 0, :] = 0.0
    beam_ref.data_array[:, :, 0, 1, :] = 0.0
    beam1.history = efield_beam.history
    assert beam1 == beam_ref

    # Add without inplace
    beam1 = efield_beam.select(pixels=efield_beam.pixel_array[0:efield_beam.Npixels // 2],
                               inplace=False)
    beam2 = efield_beam.select(pixels=efield_beam.pixel_array[efield_beam.Npixels // 2:],
                               inplace=False)
    beam1 = beam1 + beam2
    assert uvutils._check_histories(efield_beam.history
                                    + '  Downselected to specific healpix pixels '
                                    'using pyuvdata. Combined data '
                                    'along healpix pixel axis using pyuvdata.',
                                    beam1.history)
    beam1.history = efield_beam.history
    assert beam1 == efield_beam

    # ---------------
    # Test error: adding overlapping data with healpix
    beam1 = copy.deepcopy(power_beam_healpix)
    beam2 = copy.deepcopy(power_beam_healpix)
    pytest.raises(ValueError, beam1.__iadd__, beam2)

    # ---------------
    # Test beam area methods
    # Check that non-peak normalizations error
    pytest.raises(ValueError, power_beam_healpix.get_beam_area)
    pytest.raises(ValueError, power_beam_healpix.get_beam_sq_area)
    healpix_norm = copy.deepcopy(power_beam_healpix)
    healpix_norm.data_normalization = 'solid_angle'
    pytest.raises(ValueError, healpix_norm.get_beam_area)
    pytest.raises(ValueError, healpix_norm.get_beam_sq_area)

    # change it back to 'physical'
    healpix_norm.data_normalization = 'physical'
    # change it to peak for rest of checks
    healpix_norm.peak_normalize()

    # Check sizes of output
    numfreqs = healpix_norm.freq_array.shape[-1]
    beam_int = healpix_norm.get_beam_area(pol='xx')
    beam_sq_int = healpix_norm.get_beam_sq_area(pol='xx')
    assert beam_int.shape[0] == numfreqs
    assert beam_sq_int.shape[0] == numfreqs

    # Check for the case of a uniform beam over the whole sky
    dOmega = hp.nside2pixarea(healpix_norm.nside)
    npix = healpix_norm.Npixels
    healpix_norm.data_array = np.ones_like(healpix_norm.data_array)
    assert np.allclose(np.sum(healpix_norm.get_beam_area(pol='xx')), numfreqs * npix * dOmega)
    healpix_norm.data_array = 2. * np.ones_like(healpix_norm.data_array)
    assert np.allclose(np.sum(healpix_norm.get_beam_sq_area(pol='xx')), numfreqs * 4. * npix * dOmega)

    # check XX and YY beam areas work and match to within 5 sigfigs
    XX_area = healpix_norm.get_beam_area('XX')
    xx_area = healpix_norm.get_beam_area('xx')
    assert np.allclose(xx_area, XX_area)
    YY_area = healpix_norm.get_beam_area('YY')
    assert np.allclose(YY_area / XX_area, np.ones(numfreqs))
    # nt.assert_almost_equal(YY_area / XX_area, 1.0, places=5)
    XX_area = healpix_norm.get_beam_sq_area("XX")
    YY_area = healpix_norm.get_beam_sq_area("YY")
    assert np.allclose(YY_area / XX_area, np.ones(numfreqs))
    # nt.assert_almost_equal(YY_area / XX_area, 1.0, places=5)

    # Check that if pseudo-Stokes I (pI) is in the beam polarization_array, it just uses it
    healpix_norm.polarization_array = [1, 2]
    # nt.assert_almost_equal(np.sum(healpix_norm.get_beam_area()), 2. * numfreqs * npix * dOmega)
    # nt.assert_almost_equal(np.sum(healpix_norm.get_beam_sq_area()), 4. * numfreqs * npix * dOmega)

    # Check error if desired pol is allowed but isn't in the polarization_array
    pytest.raises(ValueError, healpix_norm.get_beam_area, pol='xx')
    pytest.raises(ValueError, healpix_norm.get_beam_sq_area, pol='xx')

    # Check polarization error
    healpix_norm.polarization_array = [9, 18, 27, -4]
    pytest.raises(ValueError, healpix_norm.get_beam_area, pol='xx')
    pytest.raises(ValueError, healpix_norm.get_beam_sq_area, pol='xx')

    healpix_norm_fullpol = efield_beam.efield_to_power(inplace=False)
    healpix_norm_fullpol.peak_normalize()
    XX_area = healpix_norm_fullpol.get_beam_sq_area("XX")
    YY_area = healpix_norm_fullpol.get_beam_sq_area("YY")
    XY_area = healpix_norm_fullpol.get_beam_sq_area("XY")
    YX_area = healpix_norm_fullpol.get_beam_sq_area("YX")
    # check if XY beam area is equal to beam YX beam area
    assert np.allclose(XY_area, YX_area)
    # check if XY/YX beam area is less than XX/YY beam area
    assert np.all(np.less(XY_area, XX_area))
    assert np.all(np.less(XY_area, YY_area))
    assert np.all(np.less(YX_area, XX_area))
    assert np.all(np.less(YX_area, YY_area))

    # Check if power is scalar
    healpix_vec_norm = efield_beam.efield_to_power(keep_basis_vector=True,
                                                   calc_cross_pols=False,
                                                   inplace=False)
    healpix_vec_norm.peak_normalize()
    pytest.raises(ValueError, healpix_vec_norm.get_beam_area)
    pytest.raises(ValueError, healpix_vec_norm.get_beam_sq_area)

    # Check only power beams accepted
    pytest.raises(ValueError, efield_beam.get_beam_area)
    pytest.raises(ValueError, efield_beam.get_beam_sq_area)

    # check pseudo-Stokes parameters
    efield_beam = UVBeam()
    efield_beam.read_cst_beam(cst_files[0], beam_type='efield', frequency=150e6,
                              telescope_name='TEST', feed_name='bob',
                              feed_version='0.1',
                              model_name='E-field pattern - Rigging height 4.9m',
                              model_version='1.0')

    efield_beam.interpolation_function = 'az_za_simple'
    efield_beam.to_healpix()
    efield_beam.efield_to_pstokes()
    efield_beam.peak_normalize()
    pI_area = efield_beam.get_beam_sq_area("pI")
    pQ_area = efield_beam.get_beam_sq_area("pQ")
    pU_area = efield_beam.get_beam_sq_area("pU")
    pV_area = efield_beam.get_beam_sq_area("pV")
    assert np.all(np.less(pQ_area, pI_area))
    assert np.all(np.less(pU_area, pI_area))
    assert np.all(np.less(pV_area, pI_area))

    # check backwards compatability with pstokes nomenclature and int polnum
    I_area = efield_beam.get_beam_area('I')
    pI_area = efield_beam.get_beam_area('pI')
    area1 = efield_beam.get_beam_area(1)
    assert np.allclose(I_area, pI_area)
    assert np.allclose(I_area, area1)

    # check efield beam type is accepted for pseudo-stokes and power for linear polarizations
    pytest.raises(ValueError, healpix_vec_norm.get_beam_sq_area, 'pI')
    pytest.raises(ValueError, efield_beam.get_beam_sq_area, 'xx')


def test_get_beam_functions():
    power_beam = UVBeam()
    power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                             telescope_name='TEST', feed_name='bob',
                             feed_version='0.1',
                             model_name='E-field pattern - Rigging height 4.9m',
                             model_version='1.0')

    pytest.raises(AssertionError, power_beam._get_beam, 'xx')

    # Check only healpix accepted (HEALPix checks are in test_healpix)
    # change data_normalization to peak for rest of checks
    power_beam.peak_normalize()
    pytest.raises(ValueError, power_beam.get_beam_area)
    pytest.raises(ValueError, power_beam.get_beam_sq_area)

    if healpy_installed:
        power_beam = UVBeam()
        power_beam.read_cst_beam(cst_files[0], beam_type='power', frequency=150e6,
                                 telescope_name='TEST', feed_name='bob',
                                 feed_version='0.1',
                                 model_name='E-field pattern - Rigging height 4.9m',
                                 model_version='1.0')
        power_beam.interpolation_function = 'az_za_simple'
        power_beam.to_healpix()
        power_beam.peak_normalize()
        power_beam._get_beam('xx')
        pytest.raises(ValueError, power_beam._get_beam, 4)
