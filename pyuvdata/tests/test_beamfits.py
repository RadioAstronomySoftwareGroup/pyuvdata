"""Tests for BeamFits object."""
import nose.tools as nt
import os
import numpy as np
from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH
import pyuvdata.version as uvversion


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
    beam_obj = UVBeam()
    beam_obj = fill_dummy_beam(beam_obj, 'efield')
    beam_obj.beam_type = 'foo'

    write_file = os.path.join(DATA_PATH, 'test/outtest_beam.fits')
    nt.assert_raises(ValueError, beam_obj.write_beamfits, write_file, clobber=True)
