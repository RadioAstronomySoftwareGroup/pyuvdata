# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvdata object.

"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
import copy
from astropy.time import Time
from astropy.coordinates import Angle

from pyuvdata import UVData, UVCal
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

from collections import Counter


@pytest.fixture(scope='function')
def uvdata_props():
    required_parameters = ['_data_array', '_nsample_array',
                           '_flag_array', '_Ntimes', '_Nbls',
                           '_Nblts', '_Nfreqs', '_Npols', '_Nspws',
                           '_uvw_array', '_time_array', '_ant_1_array',
                           '_ant_2_array', '_lst_array',
                           '_baseline_array', '_freq_array',
                           '_polarization_array', '_spw_array',
                           '_integration_time', '_channel_width',
                           '_object_name', '_telescope_name',
                           '_instrument', '_telescope_location',
                           '_history', '_vis_units', '_Nants_data',
                           '_Nants_telescope', '_antenna_names',
                           '_antenna_numbers', '_phase_type']

    required_properties = ['data_array', 'nsample_array',
                           'flag_array', 'Ntimes', 'Nbls',
                           'Nblts', 'Nfreqs', 'Npols', 'Nspws',
                           'uvw_array', 'time_array', 'ant_1_array',
                           'ant_2_array', 'lst_array',
                           'baseline_array', 'freq_array',
                           'polarization_array', 'spw_array',
                           'integration_time', 'channel_width',
                           'object_name', 'telescope_name',
                           'instrument', 'telescope_location',
                           'history', 'vis_units', 'Nants_data',
                           'Nants_telescope', 'antenna_names',
                           'antenna_numbers', 'phase_type']

    extra_parameters = ['_extra_keywords', '_antenna_positions',
                        '_x_orientation', '_antenna_diameters',
                        '_blt_order',
                        '_gst0', '_rdate', '_earth_omega', '_dut1',
                        '_timesys', '_uvplane_reference_time',
                        '_phase_center_ra', '_phase_center_dec',
                        '_phase_center_epoch', '_phase_center_frame']

    extra_properties = ['extra_keywords', 'antenna_positions',
                        'x_orientation', 'antenna_diameters', 'blt_order', 'gst0',
                        'rdate', 'earth_omega', 'dut1', 'timesys',
                        'uvplane_reference_time',
                        'phase_center_ra', 'phase_center_dec',
                        'phase_center_epoch', 'phase_center_frame']

    other_properties = ['telescope_location_lat_lon_alt',
                        'telescope_location_lat_lon_alt_degrees',
                        'phase_center_ra_degrees', 'phase_center_dec_degrees',
                        'pyuvdata_version_str']

    uv_object = UVData()

    class DataHolder():
        def __init__(self, uv_object, required_parameters, required_properties,
                     extra_parameters, extra_properties, other_properties):
            self.uv_object = uv_object
            self.required_parameters = required_parameters
            self.required_properties = required_properties
            self.extra_parameters = extra_parameters
            self.extra_properties = extra_properties
            self.other_properties = other_properties

    uvdata_props = DataHolder(uv_object, required_parameters, required_properties,
                              extra_parameters, extra_properties, other_properties)
    # yields the data we need but will continue to the del call after tests
    yield uvdata_props

    # some post-test object cleanup
    del(uvdata_props)

    return


def test_parameter_iter(uvdata_props):
    "Test expected parameters."
    all = []
    for prop in uvdata_props.uv_object:
        all.append(prop)
    for a in uvdata_props.required_parameters + uvdata_props.extra_parameters:
        assert a in all, 'expected attribute ' + a + ' not returned in object iterator'


def test_required_parameter_iter(uvdata_props):
    "Test expected required parameters."
    required = []
    for prop in uvdata_props.uv_object.required():
        required.append(prop)
    for a in uvdata_props.required_parameters:
        assert a in required, 'expected attribute ' + a + ' not returned in required iterator'


def test_extra_parameter_iter(uvdata_props):
    "Test expected optional parameters."
    extra = []
    for prop in uvdata_props.uv_object.extra():
        extra.append(prop)
    for a in uvdata_props.extra_parameters:
        assert a in extra, 'expected attribute ' + a + ' not returned in extra iterator'


def test_unexpected_parameters(uvdata_props):
    "Test for extra parameters."
    expected_parameters = uvdata_props.required_parameters + uvdata_props.extra_parameters
    attributes = [i for i in uvdata_props.uv_object.__dict__.keys() if i[0] == '_']
    for a in attributes:
        assert a in expected_parameters, 'unexpected parameter ' + a + ' found in UVData'


def test_unexpected_attributes(uvdata_props):
    "Test for extra attributes."
    expected_attributes = uvdata_props.required_properties + \
        uvdata_props.extra_properties + uvdata_props.other_properties
    attributes = [i for i in uvdata_props.uv_object.__dict__.keys() if i[0] != '_']
    for a in attributes:
        assert a in expected_attributes, 'unexpected attribute ' + a + ' found in UVData'


def test_properties(uvdata_props):
    "Test that properties can be get and set properly."
    prop_dict = dict(list(zip(uvdata_props.required_properties + uvdata_props.extra_properties,
                              uvdata_props.required_parameters + uvdata_props.extra_parameters)))
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(uvdata_props.uv_object, k, rand_num)
        this_param = getattr(uvdata_props.uv_object, v)
        try:
            assert rand_num == this_param.value
        except(AssertionError):
            print('setting {prop_name} to a random number failed'.format(prop_name=k))
            raise(AssertionError)


@pytest.fixture(scope='function')
def uvdata_data():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    class DataHolder():
        def __init__(self, uv_object):
            self.uv_object = uv_object
            self.uv_object2 = copy.deepcopy(uv_object)

    uvdata_data = DataHolder(uv_object)
    # yields the data we need but will continue to the del call after tests
    yield uvdata_data

    # some post-test object cleanup
    del(uvdata_data)

    return


def test_equality(uvdata_data):
    """Basic equality test."""
    assert uvdata_data.uv_object == uvdata_data.uv_object


def test_check(uvdata_data):
    """Test simple check function."""
    assert uvdata_data.uv_object.check()
    # Check variety of special cases
    uvdata_data.uv_object.Nants_data += 1
    pytest.raises(ValueError, uvdata_data.uv_object.check)
    uvdata_data.uv_object.Nants_data -= 1
    uvdata_data.uv_object.Nbls += 1
    pytest.raises(ValueError, uvdata_data.uv_object.check)
    uvdata_data.uv_object.Nbls -= 1
    uvdata_data.uv_object.Ntimes += 1
    pytest.raises(ValueError, uvdata_data.uv_object.check)
    uvdata_data.uv_object.Ntimes -= 1

    # Check case where all data is autocorrelations
    # Currently only test files that have autos are fhd files
    testdir = os.path.join(DATA_PATH, 'fhd_vis_data/')
    file_list = [testdir + '1061316296_flags.sav',
                 testdir + '1061316296_vis_XX.sav',
                 testdir + '1061316296_params.sav',
                 testdir + '1061316296_layout.sav',
                 testdir + '1061316296_settings.txt']

    uvtest.checkWarnings(uvdata_data.uv_object.read_fhd, [file_list], known_warning='fhd')

    uvdata_data.uv_object.select(blt_inds=np.where(uvdata_data.uv_object.ant_1_array
                                                   == uvdata_data.uv_object.ant_2_array)[0])
    assert uvdata_data.uv_object.check()

    # test auto and cross corr uvw_array
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA"))
    autos = np.isclose(uvd.ant_1_array - uvd.ant_2_array, 0.0)
    auto_inds = np.where(autos)[0]
    cross_inds = np.where(~autos)[0]

    # make auto have non-zero uvw coords, assert ValueError
    uvd.uvw_array[auto_inds[0], 0] = 0.1
    pytest.raises(ValueError, uvd.check)

    # make cross have |uvw| zero, assert ValueError
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA"))
    uvd.uvw_array[cross_inds[0]][:] = 0.0
    pytest.raises(ValueError, uvd.check)


def test_nants_data_telescope_larger(uvdata_data):
    # make sure it's okay for Nants_telescope to be strictly greater than Nants_data
    uvdata_data.uv_object.Nants_telescope += 1
    # add dummy information for "new antenna" to pass object check
    uvdata_data.uv_object.antenna_names = np.concatenate(
        (uvdata_data.uv_object.antenna_names, ["dummy_ant"]))
    uvdata_data.uv_object.antenna_numbers = np.concatenate(
        (uvdata_data.uv_object.antenna_numbers, [20]))
    uvdata_data.uv_object.antenna_positions = np.concatenate(
        (uvdata_data.uv_object.antenna_positions, np.zeros((1, 3))), axis=0)
    assert uvdata_data.uv_object.check()


def test_ant1_array_not_in_antnums(uvdata_data):
    # make sure an error is raised if antennas in ant_1_array not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    uvdata_data.uv_object.antenna_names = uvdata_data.uv_object.antenna_names[1:]
    uvdata_data.uv_object.antenna_numbers = uvdata_data.uv_object.antenna_numbers[1:]
    uvdata_data.uv_object.antenna_positions = uvdata_data.uv_object.antenna_positions[1:, :]
    uvdata_data.uv_object.Nants_telescope = uvdata_data.uv_object.antenna_numbers.size
    with pytest.raises(ValueError) as cm:
        uvdata_data.uv_object.check()
    assert str(cm.value).startswith('All antennas in ant_1_array must be in antenna_numbers')


def test_ant2_array_not_in_antnums(uvdata_data):
    # make sure an error is raised if antennas in ant_2_array not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    uvdata_data.uv_object.antenna_names = uvdata_data.uv_object.antenna_names[:-1]
    uvdata_data.uv_object.antenna_numbers = uvdata_data.uv_object.antenna_numbers[:-1]
    uvdata_data.uv_object.antenna_positions = uvdata_data.uv_object.antenna_positions[:-1, :]
    uvdata_data.uv_object.Nants_telescope = uvdata_data.uv_object.antenna_numbers.size
    with pytest.raises(ValueError) as cm:
        uvdata_data.uv_object.check()
    assert str(cm.value).startswith('All antennas in ant_2_array must be in antenna_numbers')


def test_converttofiletype(uvdata_data):
    fhd_obj = uvdata_data.uv_object._convert_to_filetype('fhd')
    uvdata_data.uv_object._convert_from_filetype(fhd_obj)
    assert uvdata_data.uv_object, uvdata_data.uv_object2

    pytest.raises(ValueError, uvdata_data.uv_object._convert_to_filetype, 'foo')


@pytest.fixture(scope='function')
def uvdata_baseline():
    uv_object = UVData()
    uv_object.Nants_telescope = 128
    uv_object2 = UVData()
    uv_object2.Nants_telescope = 2049

    class DataHolder():
        def __init__(self, uv_object, uv_object2):
            self.uv_object = uv_object
            self.uv_object2 = uv_object2

    uvdata_baseline = DataHolder(uv_object, uv_object2)

    # yields the data we need but will continue to the del call after tests
    yield uvdata_baseline

    # Post test clean-up
    del(uvdata_baseline)
    return


def test_baseline_to_antnums(uvdata_baseline):
    """Test baseline to antnum conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.baseline_to_antnums(67585) == (0, 0)
    pytest.raises(Exception, uvdata_baseline.uv_object2.baseline_to_antnums, 67585)

    ant_pairs = [(10, 20), (280, 310)]
    for pair in ant_pairs:
        if np.max(np.array(pair)) < 255:
            bl = uvdata_baseline.uv_object.antnums_to_baseline(
                pair[0], pair[1], attempt256=True)
            ant_pair_out = uvdata_baseline.uv_object.baseline_to_antnums(bl)
            assert pair == ant_pair_out

        bl = uvdata_baseline.uv_object.antnums_to_baseline(
            pair[0], pair[1], attempt256=False)
        ant_pair_out = uvdata_baseline.uv_object.baseline_to_antnums(bl)
        assert pair == ant_pair_out


def test_antnums_to_baselines(uvdata_baseline):
    """Test antums to baseline conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0) == 67585
    assert uvdata_baseline.uv_object.antnums_to_baseline(257, 256) == 594177
    assert uvdata_baseline.uv_object.baseline_to_antnums(594177) == (257, 256)
    # Check attempt256
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0, attempt256=True) == 257
    assert uvdata_baseline.uv_object.antnums_to_baseline(257, 256) == 594177
    uvtest.checkWarnings(uvdata_baseline.uv_object.antnums_to_baseline, [257, 256],
                         {'attempt256': True}, message='found > 256 antennas')
    pytest.raises(Exception, uvdata_baseline.uv_object2.antnums_to_baseline, 0, 0)
    # check a len-1 array returns as an array
    ant1 = np.array([1])
    ant2 = np.array([2])
    assert isinstance(uvdata_baseline.uv_object.antnums_to_baseline(ant1, ant2), np.ndarray)


def test_known_telescopes():
    """Test known_telescopes method returns expected results."""
    uv_object = UVData()
    known_telescopes = ['PAPER', 'HERA', 'MWA']
    # calling np.sort().tolist() because [].sort() acts inplace and returns None
    # Before test had None == None
    assert np.sort(known_telescopes).tolist() == np.sort(uv_object.known_telescopes()).tolist()


def test_HERA_diameters():
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')

    uv_in.telescope_name = 'HERA'
    uvtest.checkWarnings(uv_in.set_telescope_params, message='antenna_diameters '
                         'is not set. Using known values for HERA.')

    assert uv_in.telescope_name == 'HERA'
    assert uv_in.antenna_diameters is not None

    uv_in.check()


def test_generic_read():
    uv_in = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read, [uvfits_file], {'read_data': False},
                         message='Telescope EVLA is not')
    unique_times = np.unique(uv_in.time_array)

    pytest.raises(ValueError, uv_in.read, uvfits_file, times=unique_times[0:2],
                  time_range=[unique_times[0], unique_times[1]])

    pytest.raises(ValueError, uv_in.read, uvfits_file,
                  antenna_nums=uv_in.antenna_numbers[0],
                  antenna_names=uv_in.antenna_names[1])

    pytest.raises(ValueError, uv_in.read, 'foo')


def test_phase_unphaseHERA():
    """
    Read in drift data, phase to an RA/DEC, unphase and check for object equality.
    """
    testfile = os.path.join(DATA_PATH, 'hera_testfile')
    UV_raw = UVData()
    # Note the RA/DEC values in the raw file were calculated from the lat/long
    # in the file, which don't agree with our known_telescopes.
    # So for this test we use the lat/lon in the file.
    uvtest.checkWarnings(UV_raw.read_miriad, [testfile], {'correct_lat_lon': False},
                         message='Altitude is not present in file and latitude and '
                                 'longitude values do not match')
    UV_phase = UVData()
    uvtest.checkWarnings(UV_phase.read_miriad, [testfile], {'correct_lat_lon': False},
                         message='Altitude is not present in file and '
                                 'latitude and longitude values do not match')
    UV_phase.phase(0., 0., epoch="J2000")
    UV_phase.unphase_to_drift()
    # check that phase + unphase gets back to raw
    assert UV_raw == UV_phase

    # check that phase + unphase work using gcrs
    UV_phase.phase(Angle('5d').rad, Angle('30d').rad, phase_frame='gcrs')
    UV_phase.unphase_to_drift()
    assert UV_raw == UV_phase

    # check that phase + unphase work using a different epoch
    UV_phase.phase(Angle('180d').rad, Angle('90d'), epoch=Time('2010-01-01T00:00:00', format='isot', scale='utc'))
    UV_phase.unphase_to_drift()
    assert UV_raw == UV_phase

    # check that phase + unphase work with one baseline
    UV_raw_small = UV_raw.select(blt_inds=[0], inplace=False)
    UV_phase_small = copy.deepcopy(UV_raw_small)
    UV_phase_small.phase(Angle('23h').rad, Angle('15d').rad)
    UV_phase_small.unphase_to_drift()
    assert UV_raw_small == UV_phase_small

    # check that they match if you phase & unphase using antenna locations
    # first replace the uvws with the right values
    antenna_enu = uvutils.ENU_from_ECEF((UV_raw.antenna_positions + UV_raw.telescope_location),
                                        *UV_raw.telescope_location_lat_lon_alt)
    uvw_calc = np.zeros_like(UV_raw.uvw_array)
    unique_times, unique_inds = np.unique(UV_raw.time_array, return_index=True)
    for ind, jd in enumerate(unique_times):
        inds = np.where(UV_raw.time_array == jd)[0]
        for bl_ind in inds:
            ant1_index = np.where(UV_raw.antenna_numbers == UV_raw.ant_1_array[bl_ind])[0][0]
            ant2_index = np.where(UV_raw.antenna_numbers == UV_raw.ant_2_array[bl_ind])[0][0]
            uvw_calc[bl_ind, :] = antenna_enu[ant2_index, :] - antenna_enu[ant1_index, :]

    UV_raw_new = copy.deepcopy(UV_raw)
    UV_raw_new.uvw_array = uvw_calc
    UV_phase.phase(0., 0., epoch="J2000", use_ant_pos=True)
    UV_phase2 = copy.deepcopy(UV_raw_new)
    UV_phase2.phase(0., 0., epoch="J2000")

    # The uvw's only agree to ~1mm. should they be better?
    assert np.allclose(UV_phase2.uvw_array, UV_phase.uvw_array, atol=1e-3)
    # the data array are just multiplied by the w's for phasing, so a difference
    # at the 1e-3 level makes the data array different at that level too.
    # -> change the tolerance on data_array for this test
    UV_phase2._data_array.tols = (0, 1e-3)
    assert UV_phase2 == UV_phase

    # check that phase + unphase gets back to raw using antpos
    UV_phase.unphase_to_drift(use_ant_pos=True)
    assert UV_raw_new == UV_phase

    # check that phasing to zenith with one timestamp has small changes
    # (it won't be identical because of precession/nutation changing the coordinate axes)
    # use gcrs rather than icrs to reduce differences (don't include abberation)
    UV_raw_small = UV_raw.select(times=UV_raw.time_array[0], inplace=False)
    UV_phase_simple_small = copy.deepcopy(UV_raw_small)
    UV_phase_simple_small.phase_to_time(time=Time(UV_raw.time_array[0], format='jd'),
                                        phase_frame='gcrs')

    # it's unclear to me how close this should be...
    assert np.allclose(UV_phase_simple_small.uvw_array, UV_raw_small.uvw_array, atol=1e-2)

    # check error if not passing a Time object to phase_to_time
    pytest.raises(TypeError, UV_raw.phase_to_time, UV_raw.time_array[0])

    # check errors when trying to unphase drift or unknown data
    pytest.raises(ValueError, UV_raw.unphase_to_drift)
    UV_raw.set_unknown_phase_type()
    pytest.raises(ValueError, UV_raw.unphase_to_drift)
    UV_raw.set_drift()

    # check errors when trying to phase phased or unknown data
    UV_phase.phase(0., 0., epoch="J2000")
    pytest.raises(ValueError, UV_phase.phase, 0., 0., epoch="J2000")
    pytest.raises(ValueError, UV_phase.phase_to_time,
                  UV_phase.time_array[0])

    UV_phase.set_unknown_phase_type()
    pytest.raises(ValueError, UV_phase.phase, 0., 0., epoch="J2000")
    pytest.raises(ValueError, UV_phase.phase_to_time,
                  UV_phase.time_array[0])

    # check errors when trying to phase to an unsupported frame
    UV_phase = copy.deepcopy(UV_raw)
    pytest.raises(ValueError, UV_phase.phase, 0., 0., epoch="J2000", phase_frame='cirs')

    del(UV_phase)
    del(UV_raw)


def test_phasing():
    """ Use MWA files phased to 2 different places to test phasing. """
    file1 = os.path.join(DATA_PATH, '1133866760.uvfits')
    file2 = os.path.join(DATA_PATH, '1133866760_rephase.uvfits')
    uvd1 = UVData()
    uvd2 = UVData()
    uvd1.read_uvfits(file1)
    uvd2.read_uvfits(file2)

    uvd1_drift = copy.deepcopy(uvd1)
    uvd1_drift.unphase_to_drift(phase_frame='gcrs')
    uvd1_drift_antpos = copy.deepcopy(uvd1)
    uvd1_drift_antpos.unphase_to_drift(phase_frame='gcrs', use_ant_pos=True)

    uvd2_drift = copy.deepcopy(uvd2)
    uvd2_drift.unphase_to_drift(phase_frame='gcrs')
    uvd2_drift_antpos = copy.deepcopy(uvd2)
    uvd2_drift_antpos.unphase_to_drift(phase_frame='gcrs', use_ant_pos=True)

    # the tolerances here are empirical -- based on what was seen in the external
    # phasing test. See the phasing memo in docs/references for details
    assert np.allclose(uvd1_drift.uvw_array, uvd2_drift.uvw_array, atol=2e-2)
    assert np.allclose(uvd1_drift_antpos.uvw_array, uvd2_drift_antpos.uvw_array)

    uvd2_rephase = copy.deepcopy(uvd2_drift)
    uvd2_rephase.phase(uvd1.phase_center_ra,
                       uvd1.phase_center_dec,
                       uvd1.phase_center_epoch,
                       phase_frame='gcrs')
    uvd2_rephase_antpos = copy.deepcopy(uvd2_drift_antpos)
    uvd2_rephase_antpos.phase(uvd1.phase_center_ra,
                              uvd1.phase_center_dec,
                              uvd1.phase_center_epoch,
                              phase_frame='gcrs',
                              use_ant_pos=True)

    # the tolerances here are empirical -- based on what was seen in the external
    # phasing test. See the phasing memo in docs/references for details
    assert np.allclose(uvd1.uvw_array, uvd2_rephase.uvw_array, atol=2e-2)
    assert np.allclose(uvd1.uvw_array, uvd2_rephase_antpos.uvw_array, atol=5e-3)

    # rephase the drift objects to the original pointing and verify that they match
    uvd1_drift.phase(uvd1.phase_center_ra, uvd1.phase_center_dec,
                     uvd1.phase_center_epoch, phase_frame='gcrs')
    uvd1_drift_antpos.phase(uvd1.phase_center_ra, uvd1.phase_center_dec,
                            uvd1.phase_center_epoch, phase_frame='gcrs',
                            use_ant_pos=True)

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the external
    # phasing test. See the phasing memo in docs/references for details
    assert np.allclose(uvd1.uvw_array, uvd1_drift.uvw_array, atol=1e-4)
    assert np.allclose(uvd1.uvw_array, uvd1_drift_antpos.uvw_array, atol=5e-3)

    uvd2_drift.phase(uvd2.phase_center_ra, uvd2.phase_center_dec,
                     uvd2.phase_center_epoch, phase_frame='gcrs')
    uvd2_drift_antpos.phase(uvd2.phase_center_ra, uvd2.phase_center_dec,
                            uvd2.phase_center_epoch, phase_frame='gcrs',
                            use_ant_pos=True)

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the external
    # phasing test. See the phasing memo in docs/references for details
    assert np.allclose(uvd2.uvw_array, uvd2_drift.uvw_array, atol=1e-4)
    assert np.allclose(uvd2.uvw_array, uvd2_drift_antpos.uvw_array, atol=2e-2)


def test_set_phase_unknown():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [
                         testfile], message='Telescope EVLA is not')

    uv_object.set_unknown_phase_type()
    assert uv_object.phase_type == 'unknown'
    assert not uv_object._phase_center_epoch.required
    assert not uv_object._phase_center_ra.required
    assert not uv_object._phase_center_dec.required
    assert uv_object.check()


def test_select_blts():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uvtest.checkWarnings(uv_object.read_miriad, [testfile],
                         known_warning='miriad')
    old_history = uv_object.history
    blt_inds = np.array([172, 182, 132, 227, 144, 44, 16, 104, 385, 134, 326, 140, 116,
                         218, 178, 391, 111, 276, 274, 308, 38, 64, 317, 76, 239, 246,
                         34, 39, 83, 184, 208, 60, 374, 295, 118, 337, 261, 21, 375,
                         396, 355, 187, 95, 122, 186, 113, 260, 264, 156, 13, 228, 291,
                         302, 72, 137, 216, 299, 341, 207, 256, 223, 250, 268, 147, 73,
                         32, 142, 383, 221, 203, 258, 286, 324, 265, 170, 236, 8, 275,
                         304, 117, 29, 167, 15, 388, 171, 82, 322, 248, 160, 85, 66,
                         46, 272, 328, 323, 152, 200, 119, 359, 23, 363, 56, 219, 257,
                         11, 307, 336, 289, 136, 98, 37, 163, 158, 80, 125, 40, 298,
                         75, 320, 74, 57, 346, 121, 129, 332, 238, 93, 18, 330, 339,
                         381, 234, 176, 22, 379, 199, 266, 100, 90, 292, 205, 58, 222,
                         350, 109, 273, 191, 368, 88, 101, 65, 155, 2, 296, 306, 398,
                         369, 378, 254, 67, 249, 102, 348, 392, 20, 28, 169, 262, 269,
                         287, 86, 300, 143, 177, 42, 290, 284, 123, 189, 175, 97, 340,
                         242, 342, 331, 282, 235, 344, 63, 115, 78, 30, 226, 157, 133,
                         71, 35, 212, 333])

    selected_data = uv_object.data_array[np.sort(blt_inds), :, :, :]

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(blt_inds=blt_inds)
    assert len(blt_inds) == uv_object2.Nblts

    # verify that histories are different
    assert not uvutils._check_histories(old_history, uv_object2.history)

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific baseline-times using pyuvdata.',
                                    uv_object2.history)

    assert np.all(selected_data == uv_object2.data_array)

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(blt_inds=blt_inds[np.newaxis, :])
    assert len(blt_inds) == uv_object2.Nblts

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific baseline-times using pyuvdata.',
                                    uv_object2.history)
    assert np.all(selected_data == uv_object2.data_array)

    # check that just doing the metadata works properly
    uv_object3 = copy.deepcopy(uv_object)
    pytest.raises(ValueError, uv_object3.select, blt_inds=blt_inds, metadata_only=True)
    uv_object3.data_array = None
    pytest.raises(ValueError, uv_object3.select, blt_inds=blt_inds, metadata_only=True)
    uv_object3.flag_array = None
    pytest.raises(ValueError, uv_object3.select, blt_inds=blt_inds, metadata_only=True)
    uv_object3.nsample_array = None
    uv_object4 = uv_object3.select(blt_inds=blt_inds, metadata_only=True, inplace=False)
    for param in uv_object4:
        param_name = getattr(uv_object4, param).name
        if param_name not in ['data_array', 'flag_array', 'nsample_array']:
            assert getattr(uv_object4, param) == getattr(uv_object2, param)
        else:
            assert getattr(uv_object4, param_name) is None

    # also check with inplace=True
    uv_object3.select(blt_inds=blt_inds, metadata_only=True)
    assert uv_object3 == uv_object4

    # check for errors associated with out of bounds indices
    pytest.raises(ValueError, uv_object.select, blt_inds=np.arange(-10, -5))
    pytest.raises(ValueError, uv_object.select, blt_inds=np.arange(uv_object.Nblts + 1, uv_object.Nblts + 10))


def test_select_antennas():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    unique_ants = np.unique(
        uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist())
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])

    blts_select = [(a1 in ants_to_keep) & (a2 in ants_to_keep) for (a1, a2) in
                   zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(antenna_nums=ants_to_keep)

    assert len(ants_to_keep) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in ants_to_keep:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        assert ant in ants_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific antennas using pyuvdata.',
                                    uv_object2.history)

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(antenna_nums=ants_to_keep[np.newaxis, :])

    assert len(ants_to_keep) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in ants_to_keep:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        assert ant in ants_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific antennas using pyuvdata.',
                                    uv_object2.history)

    # now test using antenna_names to specify antennas to keep
    uv_object3 = copy.deepcopy(uv_object)
    ants_to_keep = np.array(sorted(list(ants_to_keep)))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=ant_names)

    assert uv_object2 == uv_object3

    # check that it also works with higher dimension array
    uv_object3 = copy.deepcopy(uv_object)
    ants_to_keep = np.array(sorted(list(ants_to_keep)))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=[ant_names])

    assert uv_object2 == uv_object3

    # test removing metadata associated with antennas that are no longer present
    # also add (different) antenna_diameters to test downselection
    uv_object.antenna_diameters = 1. * np.ones((uv_object.Nants_telescope,), dtype=np.float)
    for i in range(uv_object.Nants_telescope):
        uv_object.antenna_diameters += i
    uv_object4 = copy.deepcopy(uv_object)
    uv_object4.select(antenna_nums=ants_to_keep, keep_all_metadata=False)
    assert uv_object4.Nants_telescope == 9
    assert set(uv_object4.antenna_numbers) == set(ants_to_keep)
    for a in ants_to_keep:
        idx1 = uv_object.antenna_numbers.tolist().index(a)
        idx2 = uv_object4.antenna_numbers.tolist().index(a)
        assert uv_object.antenna_names[idx1] == uv_object4.antenna_names[idx2]
        assert np.allclose(uv_object.antenna_positions[idx1, :],
                           uv_object4.antenna_positions[idx2, :])
        assert uv_object.antenna_diameters[idx1], uv_object4.antenna_diameters[idx2]

    # remove antenna_diameters from object
    uv_object.antenna_diameters = None

    # check for errors associated with antennas not included in data, bad names or providing numbers and names
    pytest.raises(ValueError, uv_object.select,
                  antenna_nums=np.max(unique_ants) + np.arange(1, 3))
    pytest.raises(ValueError, uv_object.select, antenna_names='test1')
    pytest.raises(ValueError, uv_object.select,
                  antenna_nums=ants_to_keep, antenna_names=ant_names)


def sort_bl(p):
    """Sort a tuple that starts with a pair of antennas, and may have stuff after."""
    if p[1] >= p[0]:
        return p
    return (p[1], p[0]) + p[2:]


def test_select_bls():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    first_ants = [6, 2, 7, 2, 21, 27, 8]
    second_ants = [0, 20, 8, 1, 2, 3, 22]
    new_unique_ants = np.unique(first_ants + second_ants)
    ant_pairs_to_keep = list(zip(first_ants, second_ants))
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    sorted_pairs_object = [sort_bl(p) for p in zip(
        uv_object.ant_1_array, uv_object.ant_2_array)]

    blts_select = [sort_bl((a1, a2)) in sorted_pairs_to_keep for (a1, a2) in
                   zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(bls=ant_pairs_to_keep)
    sorted_pairs_object2 = [sort_bl(p) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object2
    for pair in sorted_pairs_object2:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific baselines using pyuvdata.',
                                    uv_object2.history)

    # check select with polarizations
    first_ants = [6, 2, 7, 2, 21, 27, 8]
    second_ants = [0, 20, 8, 1, 2, 3, 22]
    pols = ['RR', 'RR', 'RR', 'RR', 'RR', 'RR', 'RR']
    new_unique_ants = np.unique(first_ants + second_ants)
    bls_to_keep = list(zip(first_ants, second_ants, pols))
    sorted_bls_to_keep = [sort_bl(p) for p in bls_to_keep]

    sorted_pairs_object = [sort_bl(p) for p in zip(
        uv_object.ant_1_array, uv_object.ant_2_array)]

    blts_select = [sort_bl((a1, a2, 'RR')) in sorted_bls_to_keep for (a1, a2) in
                   zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(bls=bls_to_keep)
    sorted_pairs_object2 = [sort_bl(p) + ('RR',) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        assert ant in new_unique_ants
    for bl in sorted_bls_to_keep:
        assert bl in sorted_pairs_object2
    for bl in sorted_pairs_object2:
        assert bl in sorted_bls_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific baselines, polarizations using pyuvdata.',
                                    uv_object2.history)

    # check that you can use numpy integers with out errors:
    first_ants = list(map(np.int32, [6, 2, 7, 2, 21, 27, 8]))
    second_ants = list(map(np.int32, [0, 20, 8, 1, 2, 3, 22]))
    ant_pairs_to_keep = list(zip(first_ants, second_ants))

    uv_object2 = uv_object.select(bls=ant_pairs_to_keep, inplace=False)
    sorted_pairs_object2 = [sort_bl(p) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object2
    for pair in sorted_pairs_object2:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific baselines using pyuvdata.',
                                    uv_object2.history)

    # check that you can specify a single pair without errors
    uv_object2.select(bls=(0, 6))
    sorted_pairs_object2 = [sort_bl(p) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]
    assert list(set(sorted_pairs_object2)) == [(0, 6)]

    # check for errors associated with antenna pairs not included in data and bad inputs
    pytest.raises(ValueError, uv_object.select,
                  bls=list(zip(first_ants, second_ants)) + [0, 6])
    pytest.raises(ValueError, uv_object.select,
                  bls=[(uv_object.antenna_names[0], uv_object.antenna_names[1])])
    pytest.raises(ValueError, uv_object.select, bls=(5, 1))
    pytest.raises(ValueError, uv_object.select, bls=(0, 5))
    pytest.raises(ValueError, uv_object.select, bls=(27, 27))
    pytest.raises(ValueError, uv_object.select, bls=(6, 0, 'RR'), polarizations='RR')
    pytest.raises(ValueError, uv_object.select, bls=(6, 0, 8))


def test_select_times():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 3, 5, 6, 7, 10, 14]]

    Nblts_selected = np.sum([t in times_to_keep for t in uv_object.time_array])

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(times=times_to_keep)

    assert len(times_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific times using pyuvdata.',
                                    uv_object2.history)
    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(times=times_to_keep[np.newaxis, :])

    assert len(times_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific times using pyuvdata.',
                                    uv_object2.history)

    # check for errors associated with times not included in data
    pytest.raises(ValueError, uv_object.select, times=[np.min(unique_times) - uv_object.integration_time[0]])


def test_select_frequencies():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    freqs_to_keep = uv_object.freq_array[0, np.arange(12, 22)]

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    uv_object2.history)

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep[np.newaxis, :])

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    uv_object2.history)

    # check that selecting one frequency works
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep[0])
    assert 1 == uv_object2.Nfreqs
    assert freqs_to_keep[0] in uv_object2.freq_array
    for f in uv_object2.freq_array:
        assert f in [freqs_to_keep[0]]

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    uv_object2.history)

    # check for errors associated with frequencies not included in data
    pytest.raises(ValueError, uv_object.select, frequencies=[
                  np.max(uv_object.freq_array) + uv_object.channel_width])

    # check for warnings and errors associated with unevenly spaced or non-contiguous frequencies
    uv_object2 = copy.deepcopy(uv_object)
    uvtest.checkWarnings(uv_object2.select, [], {'frequencies': uv_object2.freq_array[0, [0, 5, 6]]},
                         message='Selected frequencies are not evenly spaced')
    write_file_uvfits = os.path.join(DATA_PATH, 'test/select_test.uvfits')
    write_file_miriad = os.path.join(DATA_PATH, 'test/select_test.uv')
    pytest.raises(ValueError, uv_object2.write_uvfits, write_file_uvfits)
    pytest.raises(ValueError, uv_object2.write_miriad, write_file_miriad)

    uv_object2 = copy.deepcopy(uv_object)
    uvtest.checkWarnings(uv_object2.select, [], {'frequencies': uv_object2.freq_array[0, [0, 2, 4]]},
                         message='Selected frequencies are not contiguous')
    pytest.raises(ValueError, uv_object2.write_uvfits, write_file_uvfits)
    pytest.raises(ValueError, uv_object2.write_miriad, write_file_miriad)


def test_select_freq_chans():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    chans_to_keep = np.arange(12, 22)

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == uv_object2.Nfreqs
    for chan in chans_to_keep:
        assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in uv_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    uv_object2.history)

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(freq_chans=chans_to_keep[np.newaxis, :])

    assert len(chans_to_keep) == uv_object2.Nfreqs
    for chan in chans_to_keep:
        assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in uv_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata.',
                                    uv_object2.history)

    # Test selecting both channels and frequencies
    freqs_to_keep = uv_object.freq_array[0, np.arange(20, 30)]  # Overlaps with chans
    all_chans_to_keep = np.arange(12, 30)

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == uv_object2.Nfreqs
    for chan in all_chans_to_keep:
        assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in uv_object.freq_array[0, all_chans_to_keep]


def test_select_polarizations():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    pols_to_keep = [-1, -2]

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(polarizations=pols_to_keep)

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        assert p in uv_object2.polarization_array
    for p in np.unique(uv_object2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific polarizations using pyuvdata.',
                                    uv_object2.history)

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(polarizations=[pols_to_keep])

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        assert p in uv_object2.polarization_array
    for p in np.unique(uv_object2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific polarizations using pyuvdata.',
                                    uv_object2.history)

    # check for errors associated with polarizations not included in data
    pytest.raises(ValueError, uv_object2.select, polarizations=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    uvtest.checkWarnings(uv_object.select, [], {'polarizations': uv_object.polarization_array[[0, 1, 3]]},
                         message='Selected polarization values are not evenly spaced')
    write_file_uvfits = os.path.join(DATA_PATH, 'test/select_test.uvfits')
    pytest.raises(ValueError, uv_object.write_uvfits, write_file_uvfits)


def test_select():
    # now test selecting along all axes at once
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history

    blt_inds = np.array([1057, 461, 1090, 354, 528, 654, 882, 775, 369, 906, 748,
                         875, 296, 773, 554, 395, 1003, 476, 762, 976, 1285, 874,
                         717, 383, 1281, 924, 264, 1163, 297, 857, 1258, 1000, 180,
                         1303, 1139, 393, 42, 135, 789, 713, 527, 1218, 576, 100,
                         1311, 4, 653, 724, 591, 889, 36, 1033, 113, 479, 322,
                         118, 898, 1263, 477, 96, 935, 238, 195, 531, 124, 198,
                         992, 1131, 305, 154, 961, 6, 1175, 76, 663, 82, 637,
                         288, 1152, 845, 1290, 379, 1225, 1240, 733, 1172, 937, 1325,
                         817, 416, 261, 1316, 957, 723, 215, 237, 270, 1309, 208,
                         17, 1028, 895, 574, 166, 784, 834, 732, 1022, 1068, 1207,
                         356, 474, 313, 137, 172, 181, 925, 201, 190, 1277, 1044,
                         1242, 702, 567, 557, 1032, 1352, 504, 545, 422, 179, 780,
                         280, 890, 774, 884])

    unique_ants = np.unique(
        uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist())
    ants_to_keep = np.array([11, 6, 20, 26, 2, 27, 3, 7, 14])

    ant_pairs_to_keep = [(2, 11), (20, 26), (6, 7), (3, 27), (14, 6)]
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    freqs_to_keep = uv_object.freq_array[0, np.arange(31, 39)]

    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 2, 6, 8, 10, 13, 14]]

    pols_to_keep = [-1, -3]

    # Independently count blts that should be selected
    blts_blt_select = [i in blt_inds for i in np.arange(uv_object.Nblts)]
    blts_ant_select = [(a1 in ants_to_keep) & (a2 in ants_to_keep) for (a1, a2) in
                       zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    blts_pair_select = [sort_bl((a1, a2)) in sorted_pairs_to_keep for (a1, a2) in
                        zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    blts_time_select = [t in times_to_keep for t in uv_object.time_array]
    Nblts_select = np.sum([bi & (ai | pi) & ti for (bi, ai, pi, ti) in
                           zip(blts_blt_select, blts_ant_select, blts_pair_select,
                               blts_time_select)])

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(blt_inds=blt_inds, antenna_nums=ants_to_keep,
                      bls=ant_pairs_to_keep, frequencies=freqs_to_keep,
                      times=times_to_keep, polarizations=pols_to_keep)

    assert Nblts_select == uv_object2.Nblts
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        assert ant in ants_to_keep

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        assert p in uv_object2.polarization_array
    for p in np.unique(uv_object2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific baseline-times, antennas, '
                                    'baselines, times, frequencies, '
                                    'polarizations using pyuvdata.',
                                    uv_object2.history)

    # test that a ValueError is raised if the selection eliminates all blts
    pytest.raises(ValueError, uv_object.select,
                  times=unique_times[0], antenna_nums=1)


def test_select_not_inplace():
    # Test non-inplace select
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    uv1 = uv_object.select(freq_chans=np.arange(32), inplace=False)
    uv1 += uv_object.select(freq_chans=np.arange(32, 64), inplace=False)
    assert uvutils._check_histories(old_history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis '
                                    'using pyuvdata.', uv1.history)

    uv1.history = old_history
    assert uv1 == uv_object


def test_conjugate_bls():
    uv1 = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv1.read_uvfits, [testfile], message='Telescope EVLA is not')

    # file comes in with ant1<ant2
    assert(np.min(uv1.ant_2_array - uv1.ant_1_array) >= 0)

    # check everything swapped & conjugated when go to ant2<ant1
    uv2 = copy.deepcopy(uv1)
    uv2.conjugate_bls(convention='ant2<ant1')
    assert(np.min(uv2.ant_1_array - uv2.ant_2_array) >= 0)

    assert(np.allclose(uv1.ant_1_array, uv2.ant_2_array))
    assert(np.allclose(uv1.ant_2_array, uv2.ant_1_array))
    assert(np.allclose(uv1.uvw_array, -1 * uv2.uvw_array,
                       rtol=uv1._uvw_array.tols[0], atol=uv1._uvw_array.tols[1]))

    # complicated because of the polarization swaps
    # polarization_array = [-1 -2 -3 -4]
    assert(np.allclose(uv1.data_array[:, :, :, :2],
                       np.conj(uv2.data_array[:, :, :, :2]),
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))

    assert(np.allclose(uv1.data_array[:, :, :, 2],
                       np.conj(uv2.data_array[:, :, :, 3]),
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))

    assert(np.allclose(uv1.data_array[:, :, :, 3],
                       np.conj(uv2.data_array[:, :, :, 2]),
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))

    # check everything returned to original values with original convention
    uv2.conjugate_bls(convention='ant1<ant2')
    assert(uv1 == uv2)

    # conjugate a particular set of blts
    blts_to_conjugate = np.arange(uv2.Nblts // 2)
    blts_not_conjugated = np.arange(uv2.Nblts // 2, uv2.Nblts)
    uv2.conjugate_bls(convention=blts_to_conjugate)

    assert(np.allclose(uv1.ant_1_array[blts_to_conjugate], uv2.ant_2_array[blts_to_conjugate]))
    assert(np.allclose(uv1.ant_2_array[blts_to_conjugate], uv2.ant_1_array[blts_to_conjugate]))
    assert(np.allclose(uv1.ant_1_array[blts_not_conjugated], uv2.ant_1_array[blts_not_conjugated]))
    assert(np.allclose(uv1.ant_2_array[blts_not_conjugated], uv2.ant_2_array[blts_not_conjugated]))

    assert(np.allclose(uv1.uvw_array[blts_to_conjugate],
                       -1 * uv2.uvw_array[blts_to_conjugate],
                       rtol=uv1._uvw_array.tols[0], atol=uv1._uvw_array.tols[1]))
    assert(np.allclose(uv1.uvw_array[blts_not_conjugated],
                       uv2.uvw_array[blts_not_conjugated],
                       rtol=uv1._uvw_array.tols[0], atol=uv1._uvw_array.tols[1]))

    # complicated because of the polarization swaps
    # polarization_array = [-1 -2 -3 -4]
    assert(np.allclose(uv1.data_array[blts_to_conjugate, :, :, :2],
                       np.conj(uv2.data_array[blts_to_conjugate, :, :, :2]),
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))
    assert(np.allclose(uv1.data_array[blts_not_conjugated, :, :, :2],
                       uv2.data_array[blts_not_conjugated, :, :, :2],
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))

    assert(np.allclose(uv1.data_array[blts_to_conjugate, :, :, 2],
                       np.conj(uv2.data_array[blts_to_conjugate, :, :, 3]),
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))
    assert(np.allclose(uv1.data_array[blts_not_conjugated, :, :, 2],
                       uv2.data_array[blts_not_conjugated, :, :, 2],
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))

    assert(np.allclose(uv1.data_array[blts_to_conjugate, :, :, 3],
                       np.conj(uv2.data_array[blts_to_conjugate, :, :, 2]),
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))
    assert(np.allclose(uv1.data_array[blts_not_conjugated, :, :, 3],
                       uv2.data_array[blts_not_conjugated, :, :, 3],
                       rtol=uv1._data_array.tols[0], atol=uv1._data_array.tols[1]))

    # check uv half plane conventions
    uv2.conjugate_bls(convention='u<0', use_enu=False)
    assert(np.max(uv2.uvw_array[:, 0]) <= 0)

    uv2.conjugate_bls(convention='u>0', use_enu=False)
    assert(np.min(uv2.uvw_array[:, 0]) >= 0)

    uv2.conjugate_bls(convention='v<0', use_enu=False)
    assert(np.max(uv2.uvw_array[:, 1]) <= 0)

    uv2.conjugate_bls(convention='v>0', use_enu=False)
    assert(np.min(uv2.uvw_array[:, 1]) >= 0)

    # unphase to drift to test using ENU positions
    uv2.unphase_to_drift(use_ant_pos=True)
    uv2.conjugate_bls(convention='u<0')
    assert(np.max(uv2.uvw_array[:, 0]) <= 0)

    uv2.conjugate_bls(convention='u>0')
    assert(np.min(uv2.uvw_array[:, 0]) >= 0)

    uv2.conjugate_bls(convention='v<0')
    assert(np.max(uv2.uvw_array[:, 1]) <= 0)

    uv2.conjugate_bls(convention='v>0')
    assert(np.min(uv2.uvw_array[:, 1]) >= 0)

    # test errors
    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention='foo')
    assert str(cm.value).startswith('convention must be one of')

    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention=np.arange(5) - 1)
    assert str(cm.value).startswith('If convention is an index array')

    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention=[uv2.Nblts])

    assert str(cm.value).startswith('If convention is an index array')


def test_reorder_pols():
    # Test function to fix polarization order
    uv1 = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv1.read_uvfits, [testfile], message='Telescope EVLA is not')
    uv2 = copy.deepcopy(uv1)
    # reorder uv2 manually
    order = [1, 3, 2, 0]
    uv2.polarization_array = uv2.polarization_array[order]
    uv2.data_array = uv2.data_array[:, :, :, order]
    uv2.nsample_array = uv2.nsample_array[:, :, :, order]
    uv2.flag_array = uv2.flag_array[:, :, :, order]
    uv1.reorder_pols(order=order)
    assert uv1 == uv2

    # Restore original order
    uvtest.checkWarnings(uv1.read_uvfits, [testfile], message='Telescope EVLA is not')
    uv2.reorder_pols()
    assert uv1 == uv2

    uv1.reorder_pols(order='AIPS')
    # check that we have aips ordering
    aips_pols = np.array([-1, -2, -3, -4]).astype(int)
    assert np.all(uv1.polarization_array == aips_pols)

    uv2 = copy.deepcopy(uv1)
    uv2.reorder_pols(order='CASA')
    # check that we have casa ordering
    casa_pols = np.array([-1, -3, -4, -2]).astype(int)
    assert np.all(uv2.polarization_array == casa_pols)
    order = np.array([0, 2, 3, 1])
    assert np.all(uv2.data_array == uv1.data_array[:, :, :, order])
    assert np.all(uv2.flag_array == uv1.flag_array[:, :, :, order])

    uv2.reorder_pols(order='AIPS')
    # check that we have aips ordering again
    assert uv1 == uv2

    # check error on unknown order
    pytest.raises(ValueError, uv2.reorder_pols, {'order': 'foo'})

    # check error if order is an array of the wrong length
    with pytest.raises(ValueError) as cm:
        uv2.reorder_pols(order=[3, 2, 1])
    assert str(cm.value).startswith('If order is an index array, it must')

    # check warning for order_pols:
    uvtest.checkWarnings(uv2.order_pols, [], {'order': 'AIPS'},
                         message=('order_pols method will be deprecated in '
                                  'favor of reorder_pols'),
                         category=DeprecationWarning)


def test_reorder_blts():
    uv1 = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv1.read_uvfits, [testfile], message='Telescope EVLA is not')

    # test default reordering in detail
    uv2 = copy.deepcopy(uv1)
    uv2.reorder_blts()
    assert(uv2.blt_order == ('time', 'baseline'))
    assert(np.min(np.diff(uv2.time_array)) >= 0)
    for this_time in np.unique(uv2.time_array):
        bls_2 = uv2.baseline_array[np.where(uv2.time_array == this_time)]
        bls_1 = uv1.baseline_array[np.where(uv2.time_array == this_time)]
        assert(bls_1.shape == bls_2.shape)
        assert(np.min(np.diff(bls_2)) >= 0)
        bl_inds = [np.where(bls_1 == bl)[0][0] for bl in bls_2]
        assert(np.allclose(bls_1[bl_inds], bls_2))

        uvw_1 = uv1.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        uvw_2 = uv2.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        assert(uvw_1.shape == uvw_2.shape)
        assert(np.allclose(uvw_1[bl_inds, :], uvw_2))

        data_1 = uv1.data_array[np.where(uv2.time_array == this_time)[0], :, :, :]
        data_2 = uv2.data_array[np.where(uv2.time_array == this_time)[0], :, :, :]
        assert(data_1.shape == data_2.shape)
        assert(np.allclose(data_1[bl_inds, :, :, :], data_2))

    # check that ordering by time, ant1 is identical to time, baseline
    uv3 = copy.deepcopy(uv1)
    uv3.reorder_blts(order='time', minor_order='ant1')
    assert(uv3.blt_order == ('time', 'ant1'))
    assert(np.min(np.diff(uv3.time_array)) >= 0)
    uv3.blt_order = uv2.blt_order
    assert(uv2 == uv3)

    uv3.reorder_blts(order='time', minor_order='ant2')
    assert(uv3.blt_order == ('time', 'ant2'))
    assert(np.min(np.diff(uv3.time_array)) >= 0)

    # check that loopback works
    uv3.reorder_blts()
    assert(uv2 == uv3)

    # sort with a specified index array
    new_order = np.lexsort((uv3.baseline_array, uv3.time_array))
    uv3.reorder_blts(order=new_order)
    assert(uv3.blt_order is None)
    assert(np.min(np.diff(uv3.time_array)) >= 0)
    uv3.blt_order = ('time', 'baseline')
    assert(uv2 == uv3)

    # test sensible defaulting if minor order = major order
    uv3.reorder_blts(order='time', minor_order='time')
    assert(uv2 == uv3)

    # test all combinations of major, minor order
    uv3.reorder_blts(order='baseline')
    assert(uv3.blt_order == ('baseline', 'time'))
    assert(np.min(np.diff(uv3.baseline_array)) >= 0)

    uv3.reorder_blts(order='ant1')
    assert(uv3.blt_order == ('ant1', 'ant2'))
    assert(np.min(np.diff(uv3.ant_1_array)) >= 0)

    uv3.reorder_blts(order='ant1', minor_order='time')
    assert(uv3.blt_order == ('ant1', 'time'))
    assert(np.min(np.diff(uv3.ant_1_array)) >= 0)

    uv3.reorder_blts(order='ant1', minor_order='baseline')
    assert(uv3.blt_order == ('ant1', 'baseline'))
    assert(np.min(np.diff(uv3.ant_1_array)) >= 0)

    uv3.reorder_blts(order='ant2')
    assert(uv3.blt_order == ('ant2', 'ant1'))
    assert(np.min(np.diff(uv3.ant_2_array)) >= 0)

    uv3.reorder_blts(order='ant2', minor_order='time')
    assert(uv3.blt_order == ('ant2', 'time'))
    assert(np.min(np.diff(uv3.ant_2_array)) >= 0)

    uv3.reorder_blts(order='ant2', minor_order='baseline')
    assert(uv3.blt_order == ('ant2', 'baseline'))
    assert(np.min(np.diff(uv3.ant_2_array)) >= 0)

    uv3.reorder_blts(order='bda')
    assert(uv3.blt_order == ('bda',))
    assert(np.min(np.diff(uv3.integration_time)) >= 0)
    assert(np.min(np.diff(uv3.baseline_array)) >= 0)

    # test doing conjugation along with a reorder
    # the file is already conjugated this way, so should be equal
    uv3.reorder_blts(order='time', conj_convention='ant1<ant2')
    assert(uv2 == uv3)

    # test errors
    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order='foo')
    assert str(cm.value).startswith('order must be one of')

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order=np.arange(5))
    assert str(cm.value).startswith('If order is an index array, it must')

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order=np.arange(5, dtype=np.float))
    assert str(cm.value).startswith('If order is an index array, it must')

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order=np.arange(uv3.Nblts), minor_order='time')
    assert str(cm.value).startswith('Minor order cannot be set if order is an index array')

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order='bda', minor_order='time')
    assert str(cm.value).startswith('minor_order cannot be specified if order is')

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order='baseline', minor_order='ant1')
    assert str(cm.value).startswith('minor_order conflicts with order')

    with pytest.raises(ValueError) as cm:
        uv3.reorder_blts(order='time', minor_order='foo')
    assert str(cm.value).startswith('minor_order can only be one of')


def test_add():
    uv_full = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Add frequencies
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis '
                                    'using pyuvdata.', uv1.history)

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add frequencies - out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add polarizations
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific polarizations using pyuvdata. '
                                    'Combined data along polarization axis '
                                    'using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations - out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add times
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times using pyuvdata. '
                                    'Combined data along baseline-time axis '
                                    'using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific baseline-times using pyuvdata. '
                                    'Combined data along baseline-time axis '
                                    'using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines - out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv3 = copy.deepcopy(uv_full)
    ants = uv_full.get_ants()
    ants1 = ants[0:6]
    ants2 = ants[6:12]
    ants3 = ants[12:]

    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ants1]
    ind2 = [i for i in range(uv2.Nblts) if uv2.ant_1_array[i] in ants2]
    ind3 = [i for i in range(uv3.Nblts) if uv3.ant_1_array[i] in ants3]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv3.select(blt_inds=ind3)
    uv3.data_array = uv3.data_array[-1::-1, :, :, :]
    uv3.nsample_array = uv3.nsample_array[-1::-1, :, :, :]
    uv3.flag_array = uv3.flag_array[-1::-1, :, :, :]
    uv3.uvw_array = uv3.uvw_array[-1::-1, :]
    uv3.time_array = uv3.time_array[-1::-1]
    uv3.lst_array = uv3.lst_array[-1::-1]
    uv3.ant_1_array = uv3.ant_1_array[-1::-1]
    uv3.ant_2_array = uv3.ant_2_array[-1::-1]
    uv3.baseline_array = uv3.baseline_array[-1::-1]
    uv1 += uv3
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific baseline-times using pyuvdata. '
                                    'Combined data along baseline-time axis '
                                    'using pyuvdata. Combined data along '
                                    'baseline-time axis using pyuvdata.',
                                    uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add multiple axes
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2],
               polarizations=uv1.polarization_array[0:2])
    uv2.select(times=times[len(times) // 2:],
               polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times, polarizations using '
                                    'pyuvdata. Combined data along '
                                    'baseline-time, polarization axis '
                                    'using pyuvdata.', uv1.history)
    blt_ind1 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) // 2]])
    blt_ind2 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) // 2:]])
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.nsample_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.flag_array[blt_ind1, :, :, 2:] = True
    uv_ref.data_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.nsample_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.flag_array[blt_ind2, :, :, 0:2] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Another combo
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2:], freq_chans=np.arange(32, 64))
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times, frequencies using '
                                    'pyuvdata. Combined data along '
                                    'baseline-time, frequency axis using '
                                    'pyuvdata.', uv1.history)
    blt_ind1 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) // 2]])
    blt_ind2 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) // 2:]])
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.nsample_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.flag_array[blt_ind1, :, 32:, :] = True
    uv_ref.data_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.nsample_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.flag_array[blt_ind2, :, 0:32, :] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Add without inplace
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 = uv1 + uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times using pyuvdata. '
                                    'Combined data along baseline-time '
                                    'axis using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    uvtest.checkWarnings(uv1.__add__, [uv2],
                         message='Combined frequencies are not evenly spaced')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    uvtest.checkWarnings(uv1.__iadd__, [uv2],
                         message='Combined frequencies are not contiguous')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    uvtest.checkWarnings(uv1.__iadd__, [uv2],
                         message='Combined polarizations are not evenly spaced')

    # Combining histories
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += ' testing the history. AIPS WTSCAL = 1.0'
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific polarizations using pyuvdata. '
                                    'Combined data along polarization '
                                    'axis using pyuvdata. testing the history.',
                                    uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # test add of autocorr-only and crosscorr-only objects
    uv_full = UVData()
    uv_full.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA'))
    bls = uv_full.get_antpairs()
    autos = [bl for bl in bls if bl[0] == bl[1]]
    cross = sorted(set(bls) - set(autos))
    uv_auto = uv_full.select(bls=autos, inplace=False)
    uv_cross = uv_full.select(bls=cross, inplace=False)
    uv1 = uv_auto + uv_cross
    assert uv1.Nbls == uv_auto.Nbls + uv_cross.Nbls
    uv2 = uv_cross + uv_auto
    assert uv2.Nbls == uv_auto.Nbls + uv_cross.Nbls


def test_add_drift():
    uv_full = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    uvtest.checkWarnings(uv_full.unphase_to_drift, category=DeprecationWarning,
                         message='The xyz array in ENU_from_ECEF is being '
                                 'interpreted as (Npts, 3)')
    # Add frequencies
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency '
                                    'axis using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific polarizations using pyuvdata. '
                                    'Combined data along polarization '
                                    'axis using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add times
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times using pyuvdata. '
                                    'Combined data along baseline-time '
                                    'axis using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific baseline-times using pyuvdata. '
                                    'Combined data along baseline-time '
                                    'axis using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add multiple axes
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2],
               polarizations=uv1.polarization_array[0:2])
    uv2.select(times=times[len(times) // 2:],
               polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times, polarizations using '
                                    'pyuvdata. Combined data along '
                                    'baseline-time, polarization '
                                    'axis using pyuvdata.', uv1.history)
    blt_ind1 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) // 2]])
    blt_ind2 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) // 2:]])
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.nsample_array[blt_ind1, :, :, 2:] = 0.0
    uv_ref.flag_array[blt_ind1, :, :, 2:] = True
    uv_ref.data_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.nsample_array[blt_ind2, :, :, 0:2] = 0.0
    uv_ref.flag_array[blt_ind2, :, :, 0:2] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Another combo
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2:], freq_chans=np.arange(32, 64))
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times, frequencies using '
                                    'pyuvdata. Combined data along '
                                    'baseline-time, frequency '
                                    'axis using pyuvdata.', uv1.history)
    blt_ind1 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) // 2]])
    blt_ind2 = np.array([ind for ind in range(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) // 2:]])
    # Zero out missing data in reference object
    uv_ref.data_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.nsample_array[blt_ind1, :, 32:, :] = 0.0
    uv_ref.flag_array[blt_ind1, :, 32:, :] = True
    uv_ref.data_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.nsample_array[blt_ind2, :, 0:32, :] = 0.0
    uv_ref.flag_array[blt_ind2, :, 0:32, :] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Add without inplace
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 = uv1 + uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times using pyuvdata. '
                                    'Combined data along baseline-time '
                                    'axis using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    uvtest.checkWarnings(uv1.__add__, [uv2],
                         message='Combined frequencies are not evenly spaced')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    uvtest.checkWarnings(uv1.__iadd__, [uv2],
                         message='Combined frequencies are not contiguous')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    uvtest.checkWarnings(uv1.__iadd__, [uv2],
                         message='Combined polarizations are not evenly spaced')

    # Combining histories
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += ' testing the history. AIPS WTSCAL = 1.0'
    uv1 += uv2
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific polarizations using pyuvdata. '
                                    'Combined data along polarization '
                                    'axis using pyuvdata. testing the history.',
                                    uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full


def test_break_add():
    # Test failure modes of add function
    uv_full = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Wrong class
    uv1 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    pytest.raises(ValueError, uv1.__iadd__, np.zeros(5))

    # One phased, one not
    uv2 = copy.deepcopy(uv_full)
    uvtest.checkWarnings(uv2.unphase_to_drift, category=DeprecationWarning,
                         message='The xyz array in ENU_from_ECEF is being '
                                 'interpreted as (Npts, 3)')
    pytest.raises(ValueError, uv1.__iadd__, uv2)

    # Different units
    uv2 = copy.deepcopy(uv_full)
    uv2.select(freq_chans=np.arange(32, 64))
    uv2.vis_units = "Jy"
    pytest.raises(ValueError, uv1.__iadd__, uv2)

    # Overlapping data
    uv2 = copy.deepcopy(uv_full)
    pytest.raises(ValueError, uv1.__iadd__, uv2)

    # Different integration_time
    uv2 = copy.deepcopy(uv_full)
    uv2.select(freq_chans=np.arange(32, 64))
    uv2.integration_time *= 2
    pytest.raises(ValueError, uv1.__iadd__, uv2)


def test_fast_concat():
    uv_full = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Add frequencies
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.fast_concat(uv2, 'freq', inplace=True)
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis '
                                    'using pyuvdata.', uv1.history)

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add frequencies - out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uvtest.checkWarnings(uv2.fast_concat, [uv1, 'freq'], {'inplace': True},
                         message='Combined frequencies are not evenly spaced')
    assert uv2.Nfreqs == uv_full.Nfreqs
    assert uv2._freq_array != uv_full._freq_array
    assert uv2._data_array != uv_full._data_array

    # reorder frequencies and test that they are equal
    index_array = np.argsort(uv2.freq_array[0, :])
    uv2.freq_array = uv2.freq_array[:, index_array]
    uv2.data_array = uv2.data_array[:, :, index_array, :]
    uv2.nsample_array = uv2.nsample_array[:, :, index_array, :]
    uv2.flag_array = uv2.flag_array[:, :, index_array, :]
    uv2.history = uv_full.history
    assert uv2._freq_array == uv_full._freq_array
    assert uv2 == uv_full

    # Add polarizations
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1.fast_concat(uv2, 'polarization', inplace=True)
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific polarizations using pyuvdata. '
                                    'Combined data along polarization axis '
                                    'using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations - out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uvtest.checkWarnings(uv2.fast_concat, [uv1, 'polarization'], {'inplace': True},
                         message='Combined polarizations are not evenly spaced')
    assert uv2._polarization_array != uv_full._polarization_array
    assert uv2._data_array != uv_full._data_array

    # reorder pols
    uv2.reorder_pols()
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add times
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1.fast_concat(uv2, 'blt', inplace=True)
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times using pyuvdata. '
                                    'Combined data along baseline-time axis '
                                    'using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    # divide in half to keep in order
    ind1 = np.arange(uv1.Nblts // 2)
    ind2 = np.arange(uv1.Nblts // 2, uv1.Nblts)
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1.fast_concat(uv2, 'blt', inplace=True)
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific baseline-times using pyuvdata. '
                                    'Combined data along baseline-time axis '
                                    'using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1, uv_full

    # Add baselines out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv2.fast_concat(uv1, 'blt', inplace=True)
    # test freq & pol arrays equal
    assert uv2._freq_array == uv_full._freq_array
    assert uv2._polarization_array == uv_full._polarization_array

    # test Nblt length arrays not equal but same shape
    assert uv2._ant_1_array != uv_full._ant_1_array
    assert uv2.ant_1_array.shape == uv_full.ant_1_array.shape
    assert uv2._ant_2_array != uv_full._ant_2_array
    assert uv2.ant_2_array.shape == uv_full.ant_2_array.shape
    assert uv2._uvw_array != uv_full._uvw_array
    assert uv2.uvw_array.shape == uv_full.uvw_array.shape
    assert uv2._time_array != uv_full._time_array
    assert uv2.time_array.shape == uv_full.time_array.shape
    assert uv2._baseline_array != uv_full._baseline_array
    assert uv2.baseline_array.shape == uv_full.baseline_array.shape
    assert uv2._data_array != uv_full._data_array
    assert uv2.data_array.shape == uv_full.data_array.shape

    # reorder blts to enable comparison
    uv2.reorder_blts()
    assert uv2.blt_order == ('time', 'baseline')
    uv2.blt_order = None
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # add baselines such that Nants_data needs to change
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv2.fast_concat(uv1, 'blt', inplace=True)

    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific baseline-times using pyuvdata. '
                                    'Combined data along baseline-time '
                                    'axis using pyuvdata.', uv2.history)

    # test freq & pol arrays equal
    assert uv2._freq_array == uv_full._freq_array
    assert uv2._polarization_array == uv_full._polarization_array

    # test Nblt length arrays not equal but same shape
    assert uv2._ant_1_array != uv_full._ant_1_array
    assert uv2.ant_1_array.shape == uv_full.ant_1_array.shape
    assert uv2._ant_2_array != uv_full._ant_2_array
    assert uv2.ant_2_array.shape == uv_full.ant_2_array.shape
    assert uv2._uvw_array != uv_full._uvw_array
    assert uv2.uvw_array.shape == uv_full.uvw_array.shape
    assert uv2._time_array != uv_full._time_array
    assert uv2.time_array.shape == uv_full.time_array.shape
    assert uv2._baseline_array != uv_full._baseline_array
    assert uv2.baseline_array.shape == uv_full.baseline_array.shape
    assert uv2._data_array != uv_full._data_array
    assert uv2.data_array.shape == uv_full.data_array.shape

    # reorder blts to enable comparison
    uv2.reorder_blts()
    assert uv2.blt_order == ('time', 'baseline')
    uv2.blt_order = None
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add multiple axes
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2],
               polarizations=uv1.polarization_array[0:2])
    uv2.select(times=times[len(times) // 2:],
               polarizations=uv2.polarization_array[2:4])
    pytest.raises(ValueError, uv1.fast_concat, uv2, 'blt', inplace=True)

    # Another combo
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2:], freq_chans=np.arange(32, 64))
    pytest.raises(ValueError, uv1.fast_concat, uv2, 'blt', inplace=True)

    # Add without inplace
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 = uv1.fast_concat(uv2, 'blt', inplace=False)
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific times using pyuvdata. '
                                    'Combined data along baseline-time '
                                    'axis using pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    uvtest.checkWarnings(uv1.fast_concat, [uv2, 'freq'],
                         message='Combined frequencies are not evenly spaced')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    uvtest.checkWarnings(uv1.fast_concat, [uv2, 'freq'],
                         message='Combined frequencies are not contiguous')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    uvtest.checkWarnings(uv1.fast_concat, [uv2, 'polarization'],
                         message='Combined polarizations are not evenly spaced')

    # Combining histories
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += ' testing the history. AIPS WTSCAL = 1.0'
    uv1.fast_concat(uv2, 'polarization', inplace=True)
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific polarizations using pyuvdata. '
                                    'Combined data along polarization '
                                    'axis using pyuvdata. testing the history.',
                                    uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # test add of autocorr-only and crosscorr-only objects
    uv_full = UVData()
    uv_full.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA'))
    bls = uv_full.get_antpairs()
    autos = [bl for bl in bls if bl[0] == bl[1]]
    cross = sorted(set(bls) - set(autos))
    uv_auto = uv_full.select(bls=autos, inplace=False)
    uv_cross = uv_full.select(bls=cross, inplace=False)
    uv1 = uv_auto.fast_concat(uv_cross, 'blt')
    assert uv1.Nbls == uv_auto.Nbls + uv_cross.Nbls
    uv2 = uv_cross.fast_concat(uv_auto, 'blt')
    assert uv2.Nbls == uv_auto.Nbls + uv_cross.Nbls


def test_fast_concat_errors():
    uv_full = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    pytest.raises(ValueError, uv1.fast_concat, uv2, 'foo', inplace=True)

    cal = UVCal()
    pytest.raises(ValueError, uv1.fast_concat, cal, 'freq', inplace=True)


def test_key2inds():
    # Test function to interpret key as antpair, pol
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant2, pol))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal([0], indp[0])
    # Any of these inputs can also be a tuple of a tuple, so need to be checked twice.
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, pol),))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal([0], indp[0])

    # Combo with pol as string
    ind1, ind2, indp = uv._key2inds((ant1, ant2, uvutils.polnum2str(pol)))
    assert np.array_equal([0], indp[0])
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, uvutils.polnum2str(pol)),))
    assert np.array_equal([0], indp[0])

    # Check conjugation
    ind1, ind2, indp = uv._key2inds((ant2, ant1, pol))
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal([0], indp[1])
    # Conjugation with pol as string
    ind1, ind2, indp = uv._key2inds((ant2, ant1, uvutils.polnum2str(pol)))
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal([0], indp[1])
    assert np.array_equal([], indp[0])

    # Antpair only
    ind1, ind2, indp = uv._key2inds((ant1, ant2))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])
    ind1, ind2, indp = uv._key2inds(((ant1, ant2)))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])

    # Baseline number only
    ind1, ind2, indp = uv._key2inds(uv.antnums_to_baseline(ant1, ant2))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])
    ind1, ind2, indp = uv._key2inds((uv.antnums_to_baseline(ant1, ant2),))
    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.arange(uv.Npols), indp[0])

    # Pol number only
    ind1, ind2, indp = uv._key2inds(pol)
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])
    ind1, ind2, indp = uv._key2inds((pol))
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])

    # Pol string only
    ind1, ind2, indp = uv._key2inds('LL')
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([1]), indp[0])
    ind1, ind2, indp = uv._key2inds(('LL'))
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([1]), indp[0])

    # Test invalid keys
    pytest.raises(KeyError, uv._key2inds, 'I')  # pol str not in data
    pytest.raises(KeyError, uv._key2inds, -8)  # pol num not in data
    pytest.raises(KeyError, uv._key2inds, 6)  # bl num not in data
    pytest.raises(KeyError, uv._key2inds, (1, 1))  # ant pair not in data
    pytest.raises(KeyError, uv._key2inds, (1, 1, 'rr'))  # ant pair not in data
    pytest.raises(KeyError, uv._key2inds, (0, 1, 'xx'))  # pol not in data

    # Test autos are handled correctly
    uv.ant_2_array[0] = uv.ant_1_array[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant1, pol))
    assert np.array_equal(ind1, [0])
    assert np.array_equal(ind2, [])


def test_key2inds_conj_all_pols():
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds((ant2, ant1))

    # Pols in data are 'rr', 'll', 'rl', 'lr'
    # So conjugated order should be [0, 1, 3, 2]
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal(np.array([]), indp[0])
    assert np.array_equal([0, 1, 3, 2], indp[1])


def test_key2inds_conj_all_pols_fringe():
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    uv.select(polarizations=['rl'])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    # Mix one instance of this baseline.
    uv.ant_1_array[0] = ant2
    uv.ant_2_array[0] = ant1
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant2))

    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])
    assert np.array_equal(np.array([]), indp[1])


def test_key2inds_conj_all_pols_bl_fringe():
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    uv.select(polarizations=['rl'])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    # Mix one instance of this baseline.
    uv.ant_1_array[0] = ant2
    uv.ant_2_array[0] = ant1
    uv.baseline_array[0] = uvutils.antnums_to_baseline(ant2, ant1, uv.Nants_telescope)
    bl = uvutils.antnums_to_baseline(ant1, ant2, uv.Nants_telescope)
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds(bl)

    assert np.array_equal(bltind, ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([0]), indp[0])
    assert np.array_equal(np.array([]), indp[1])


def test_key2inds_conj_all_pols_missing_data():
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    uv.select(polarizations=['rl'])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]

    pytest.raises(KeyError, uv._key2inds, (ant2, ant1))


def test_key2inds_conj_all_pols_bls():
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bl = uvutils.antnums_to_baseline(ant2, ant1, uv.Nants_telescope)
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds(bl)

    # Pols in data are 'rr', 'll', 'rl', 'lr'
    # So conjugated order should be [0, 1, 3, 2]
    assert np.array_equal(bltind, ind2)
    assert np.array_equal(np.array([]), ind1)
    assert np.array_equal(np.array([]), indp[0])
    assert np.array_equal([0, 1, 3, 2], indp[1])


def test_key2inds_conj_all_pols_missing_data_bls():
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    uv.select(polarizations=['rl'])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bl = uvutils.antnums_to_baseline(ant2, ant1, uv.Nants_telescope)

    pytest.raises(KeyError, uv._key2inds, bl)


def test_smart_slicing():
    # Test function to slice data
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # ind1 reg, ind2 empty, pol reg
    ind1 = 10 * np.arange(9)
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a view was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 5.43
    assert d[1, 0, 0] == uv.data_array[ind1[1], 0, 0, indp[0]]

    # force copy
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), force_copy=True)
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 4.3
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 reg, ind2 empty, pol not reg
    ind1 = 10 * np.arange(9)
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 1.2
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 not reg, ind2 empty, pol reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 8.2
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 not reg, ind2 empty, pol not reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 3.4
    assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 empty, ind2 reg, pol reg
    # Note conjugation test ensures the result is a copy, not a view.
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 reg, pol not reg
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 not reg, pol reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 not reg, pol not reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1, ind2 not empty, pol reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(uv.data_array[ind1, :, :, :],
                       np.conj(uv.data_array[ind2, :, :, :]), axis=0)
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # ind1, ind2 not empty, pol not reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(uv.data_array[ind1, :, :, :],
                       np.conj(uv.data_array[ind2, :, :, :]), axis=0)
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # test single element
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp], axis=1)
    assert np.all(d == dcheck)

    # test single element
    ind1 = []
    ind2 = [45]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    assert np.all(d == np.conj(dcheck))

    # Full squeeze
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), squeeze='full')
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # Test invalid squeeze
    pytest.raises(ValueError, uv._smart_slicing, uv.data_array, ind1, ind2,
                  (indp, []), squeeze='notasqueeze')


def test_get_data():
    # Test get_data function for easy access to data
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = np.squeeze(uv.data_array[bltind, :, :, 0])
    d = uv.get_data(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_data(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_data((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_data((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith('no more than 3 key values can be passed')

    # Check conjugation
    d = uv.get_data(ant2, ant1, pol)
    assert np.all(dcheck == np.conj(d))

    # Check cross pol conjugation
    d = uv.get_data(ant2, ant1, uv.polarization_array[2])
    d1 = uv.get_data(ant1, ant2, uv.polarization_array[3])
    assert np.all(d == np.conj(d1))

    # Antpair only
    dcheck = np.squeeze(uv.data_array[bltind, :, :, :])
    d = uv.get_data(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    dcheck = np.squeeze(uv.data_array[:, :, :, 0])
    d = uv.get_data(pol)
    assert np.all(dcheck == d)


def test_get_flags():
    # Test function for easy access to flags
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = np.squeeze(uv.flag_array[bltind, :, :, 0])
    d = uv.get_flags(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_flags(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_flags((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_flags((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith('no more than 3 key values can be passed')

    # Check conjugation
    d = uv.get_flags(ant2, ant1, pol)
    assert np.all(dcheck == d)
    assert d.dtype == np.bool

    # Antpair only
    dcheck = np.squeeze(uv.flag_array[bltind, :, :, :])
    d = uv.get_flags(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    dcheck = np.squeeze(uv.flag_array[:, :, :, 0])
    d = uv.get_flags(pol)
    assert np.all(dcheck == d)


def test_get_nsamples():
    # Test function for easy access to nsample array
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = np.squeeze(uv.nsample_array[bltind, :, :, 0])
    d = uv.get_nsamples(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_nsamples(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_nsamples((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_nsamples((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith('no more than 3 key values can be passed')

    # Check conjugation
    d = uv.get_nsamples(ant2, ant1, pol)
    assert np.all(dcheck == d)

    # Antpair only
    dcheck = np.squeeze(uv.nsample_array[bltind, :, :, :])
    d = uv.get_nsamples(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    dcheck = np.squeeze(uv.nsample_array[:, :, :, 0])
    d = uv.get_nsamples(pol)
    assert np.all(dcheck == d)


def test_antpair2ind():
    # Test for baseline-time axis indexer
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uvtest.checkWarnings(uv.read_miriad, [testfile],
                         message='Altitude is not present in Miriad file')

    # get indices
    inds = uv.antpair2ind(0, 1, ordered=False)
    np.testing.assert_array_equal(inds, np.array([1, 22, 43, 64, 85, 106, 127, 148, 169,
                                                  190, 211, 232, 253, 274, 295, 316,
                                                  337, 358, 379]))
    assert inds.dtype == np.int

    # conjugate (and use key rather than arg expansion)
    inds2 = uv.antpair2ind((1, 0), ordered=False)
    np.testing.assert_array_equal(inds, inds2)

    # test ordered
    inds3 = uv.antpair2ind(1, 0, ordered=True)
    np.testing.assert_array_equal(inds, inds2)

    # test autos w/ and w/o ordered
    inds4 = uv.antpair2ind(0, 0, ordered=True)
    inds5 = uv.antpair2ind(0, 0, ordered=False)
    np.testing.assert_array_equal(inds4, inds5)

    # test exceptions
    pytest.raises(ValueError, uv.antpair2ind, 1)
    pytest.raises(ValueError, uv.antpair2ind, 'bar', 'foo')
    pytest.raises(ValueError, uv.antpair2ind, 0, 1, 'foo')


def test_get_times():
    # Test function for easy access to times, to work in conjunction with get_data
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Get an antpair/pol combo (pol shouldn't actually effect result)
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = uv.time_array[bltind]
    d = uv.get_times(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_times(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_times((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_times((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith('no more than 3 key values can be passed')

    # Check conjugation
    d = uv.get_times(ant2, ant1, pol)
    assert np.all(dcheck == d)

    # Antpair only
    d = uv.get_times(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    d = uv.get_times(pol)
    assert np.all(d == uv.time_array)


def test_antpairpol_iter():
    # Test generator
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    pol_dict = {uvutils.polnum2str(uv.polarization_array[i]): i for i in range(uv.Npols)}
    keys = []
    pols = set()
    bls = set()
    for key, d in uv.antpairpol_iter():
        keys += key
        bl = uv.antnums_to_baseline(key[0], key[1])
        blind = np.where(uv.baseline_array == bl)[0]
        bls.add(bl)
        pols.add(key[2])
        dcheck = np.squeeze(uv.data_array[blind, :, :, pol_dict[key[2]]])
        assert np.all(dcheck == d)
    assert len(bls) == len(uv.get_baseline_nums())
    assert len(pols) == uv.Npols


def test_get_ants():
    # Test function to get unique antennas in data
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    ants = uv.get_ants()
    for ant in ants:
        assert (ant in uv.ant_1_array) or (ant in uv.ant_2_array)
    for ant in uv.ant_1_array:
        assert ant in ants
    for ant in uv.ant_2_array:
        assert ant in ants


def test_get_ENU_antpos():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA"))
    # no center, no pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=False, pick_data_ants=False)
    assert len(ants) == 113
    assert np.isclose(antpos[0, 0], 19.340211050751535)
    assert ants[0] == 0
    # test default behavior
    antpos2, ants = uvtest.checkWarnings(uvd.get_ENU_antpos, category=DeprecationWarning,
                                         message='The default for the `center` '
                                                 'keyword has changed')
    assert np.all(antpos == antpos2)
    # center
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=False)
    assert np.isclose(antpos[0, 0], 22.472442651767714)
    # pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    assert ants[0] == 9
    assert np.isclose(antpos[0, 0], -0.0026981323386223721)


def test_get_pols():
    # Test function to get unique polarizations in string format
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    pols = uv.get_pols()
    pols_data = ['rr', 'll', 'lr', 'rl']
    assert sorted(pols) == sorted(pols_data)


def test_get_pols_x_orientation():
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read, [miriad_file], known_warning='miriad')

    uv_in.x_orientation = 'east'

    pols = uv_in.get_pols()
    pols_data = ['en']
    assert pols == pols_data

    uv_in.x_orientation = 'north'

    pols = uv_in.get_pols()
    pols_data = ['ne']
    assert pols == pols_data


def test_deprecated_x_orientation():
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read, [miriad_file], known_warning='miriad')

    uv_in.x_orientation = 'e'

    uvtest.checkWarnings(uv_in.check, category=DeprecationWarning,
                         message=['x_orientation e is not one of [east, north], '
                                  'converting to "east".'])

    uv_in.x_orientation = 'N'
    uvtest.checkWarnings(uv_in.check, category=DeprecationWarning,
                         message=['x_orientation N is not one of [east, north], '
                                  'converting to "north".'])

    uv_in.x_orientation = 'foo'
    pytest.raises(ValueError, uvtest.checkWarnings, uv_in.check,
                  category=DeprecationWarning,
                  message=['x_orientation n is not one of [east, north], '
                           'cannot be converted.'])


def test_get_feedpols():
    # Test function to get unique antenna feed polarizations in data. String format.
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    pols = uv.get_feedpols()
    pols_data = ['r', 'l']
    assert sorted(pols) == sorted(pols_data)

    # Test break when pseudo-Stokes visibilities are present
    uv.polarization_array[0] = 1  # pseudo-Stokes I
    pytest.raises(ValueError, uv.get_feedpols)


def test_parse_ants():
    # Test function to get correct antenna pairs and polarizations
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile], message='Telescope EVLA is not')

    # All baselines
    ant_str = 'all'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Auto correlations
    ant_str = 'auto'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(ant_pairs_nums) == Counter([])
    assert isinstance(polarizations, type(None))

    # Cross correlations
    ant_str = 'cross'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(uv.get_antpairs()) == Counter(ant_pairs_nums)
    assert isinstance(polarizations, type(None))

    # pseudo-Stokes params
    ant_str = 'pI,pq,pU,pv'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    pols_expected = [4, 3, 2, 1]
    assert isinstance(ant_pairs_nums, type(None))
    assert Counter(polarizations) == Counter(pols_expected)

    # Unparsible string
    ant_str = 'none'
    pytest.raises(ValueError, uv.parse_ants, ant_str)

    # Single antenna number
    ant_str = '0'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8),
                          (0, 11), (0, 14), (0, 18), (0, 19), (0, 20),
                          (0, 21), (0, 22), (0, 23), (0, 24), (0, 26),
                          (0, 27)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single antenna number not in the data
    ant_str = '10'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Antenna')
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Single antenna number with polarization, both not in the data
    ant_str = '10x'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=2,
                                                         message=['Warning: Antenna', 'Warning: Polarization'])
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Multiple antenna numbers as list
    ant_str = '22,26'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(0, 22), (0, 26), (1, 22), (1, 26), (2, 22), (2, 26),
                          (3, 22), (3, 26), (6, 22), (6, 26), (7, 22),
                          (7, 26), (8, 22), (8, 26), (11, 22), (11, 26),
                          (14, 22), (14, 26), (18, 22), (18, 26),
                          (19, 22), (19, 26), (20, 22), (20, 26),
                          (21, 22), (21, 26), (22, 23), (22, 24),
                          (22, 26), (22, 27), (23, 26), (24, 26),
                          (26, 27)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single baseline
    ant_str = '1_3'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single baseline with polarization
    ant_str = '1l_3r'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    pols_expected = [-4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Single baseline with single polarization in first entry
    ant_str = '1l_3,2x_3'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Polarization')
    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Single baseline with single polarization in last entry
    ant_str = '1_3l,2_3x'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Polarization')
    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines as list
    ant_str = '1_2,1_3,1_11'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Multiples baselines with polarizations as list
    ant_str = '1r_2l,1l_3l,1r_11r'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    pols_expected = [-1, -2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Specific baselines with parenthesis
    ant_str = '(1,3)_11'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11), (3, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Specific baselines with parenthesis
    ant_str = '1_(3,11)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations
    ant_str = '(1l,2r)_(3l,6r)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [-1, -2, -3, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers with - for avoidance
    ant_str = '1_(-3,11)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific antenna number
    ant_str = '1,-3'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(0, 1), (1, 2), (1, 6), (1, 7), (1, 8), (1, 11),
                          (1, 14), (1, 18), (1, 19), (1, 20), (1, 21),
                          (1, 22), (1, 23), (1, 24), (1, 26), (1, 27)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific baseline (same expected antenna pairs as above example)
    ant_str = '1,-1_3'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations and - for avoidance
    ant_str = '1l_(-3r,11l)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers and pseudo-Stokes parameters
    ant_str = '(1l,2r)_(3l,6r),pI,pq'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [2, 1, -1, -2, -3, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines with multiple polarizations, one pol to be removed
    ant_str = '1l_2,1l_3,-1l_3r'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines with multiple polarizations, one pol (not in data) to be removed
    ant_str = '1l_2,1l_3,-1x_3y'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Polarization')
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Test print toggle on single baseline with polarization
    ant_str = '1l_2l'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str, print_toggle=True)
    ant_pairs_expected = [(1, 2)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Test ant_str='auto' on file with auto correlations
    uv = UVData()
    testfile = os.path.join(DATA_PATH, 'hera_testfile')
    uvtest.checkWarnings(uv.read_miriad, [testfile], nwarnings=1,
                         message='Altitude is not')

    ant_str = 'auto'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(9, 9), (10, 10), (20, 20)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Test cross correlation extraction on data with auto + cross
    ant_str = 'cross'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(9, 10), (9, 20), (10, 20)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove only polarization of single baseline
    ant_str = 'all,-9x_10x'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(9, 9), (9, 20), (10, 10), (10, 20), (20, 20)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Test appending all to beginning of strings that start with -
    ant_str = '-9'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(10, 10), (10, 20), (20, 20)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))


def test_select_with_ant_str():
    # Test select function with ant_str argument
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile], message='Telescope EVLA is not')
    inplace = False

    # Check error thrown if ant_str passed with antenna_nums,
    # antenna_names, ant_pairs_nums, or polarizations
    pytest.raises(ValueError, uv.select,
                  ant_str='',
                  antenna_nums=[],
                  inplace=inplace)
    pytest.raises(ValueError, uv.select,
                  ant_str='',
                  antenna_nums=[],
                  inplace=inplace)
    pytest.raises(ValueError, uv.select,
                  ant_str='',
                  antenna_nums=[],
                  inplace=inplace)
    pytest.raises(ValueError, uv.select,
                  ant_str='',
                  antenna_nums=[],
                  inplace=inplace)

    # All baselines
    ant_str = 'all'
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Auto correlations
    ant_str = 'auto'
    pytest.raises(ValueError, uv.select, ant_str=ant_str, inplace=inplace)
    # No auto correlations in this data

    # Cross correlations
    ant_str = 'cross'
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())
    # All baselines in data are cross correlations

    # pseudo-Stokes params
    ant_str = 'pI,pq,pU,pv'
    pytest.raises(ValueError, uv.select, ant_str=ant_str, inplace=inplace)

    # Unparsible string
    ant_str = 'none'
    pytest.raises(ValueError, uv.select, ant_str=ant_str, inplace=inplace)

    # Single antenna number
    ant_str = '0'
    ant_pairs = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (0, 11),
                 (0, 14), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22),
                 (0, 23), (0, 24), (0, 26), (0, 27)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single antenna number not present in data
    ant_str = '10'
    uv2 = uvtest.checkWarnings(uv.select, [], {'ant_str': ant_str, 'inplace': inplace},
                               nwarnings=1, message='Warning: Antenna')

    # Multiple antenna numbers as list
    ant_str = '22,26'
    ant_pairs = [(0, 22), (0, 26), (1, 22), (1, 26), (2, 22), (2, 26),
                 (3, 22), (3, 26), (6, 22), (6, 26), (7, 22),
                 (7, 26), (8, 22), (8, 26), (11, 22), (11, 26),
                 (14, 22), (14, 26), (18, 22), (18, 26), (19, 22),
                 (19, 26), (20, 22), (20, 26), (21, 22), (21, 26),
                 (22, 23), (22, 24), (22, 26), (22, 27), (23, 26),
                 (24, 26), (26, 27)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single baseline
    ant_str = '1_3'
    ant_pairs = [(1, 3)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single baseline with polarization
    ant_str = '1l_3r'
    ant_pairs = [(1, 3)]
    pols = ['lr']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Single baseline with single polarization in first entry
    ant_str = '1l_3,2x_3'
    # x,y pols not present in data
    uv2 = uvtest.checkWarnings(uv.select, [],
                               {'ant_str': ant_str, 'inplace': inplace},
                               nwarnings=1, message='Warning: Polarization')
    # with polarizations in data
    ant_str = '1l_3,2_3'
    ant_pairs = [(1, 3), (2, 3)]
    pols = ['ll', 'lr']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Single baseline with single polarization in last entry
    ant_str = '1_3l,2_3x'
    # x,y pols not present in data
    uv2 = uvtest.checkWarnings(uv.select, [],
                               {'ant_str': ant_str, 'inplace': inplace},
                               nwarnings=1, message='Warning: Polarization')
    # with polarizations in data
    ant_str = '1_3l,2_3'
    ant_pairs = [(1, 3), (2, 3)]
    pols = ['ll', 'rl']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Multiple baselines as list
    ant_str = '1_2,1_3,1_10'
    # Antenna number 10 not in data
    uv2 = uvtest.checkWarnings(uv.select, [],
                               {'ant_str': ant_str, 'inplace': inplace},
                               nwarnings=1, message='Warning: Antenna')
    ant_pairs = [(1, 2), (1, 3)]
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Multiples baselines with polarizations as list
    ant_str = '1r_2l,1l_3l,1r_11r'
    ant_pairs = [(1, 2), (1, 3), (1, 11)]
    pols = ['rr', 'll', 'rl']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Specific baselines with parenthesis
    ant_str = '(1,3)_11'
    ant_pairs = [(1, 11), (3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Specific baselines with parenthesis
    ant_str = '1_(3,11)'
    ant_pairs = [(1, 3), (1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations
    ant_str = '(1l,2r)_(3l,6r)'
    ant_pairs = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols = ['rr', 'll', 'rl', 'lr']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Antenna numbers with - for avoidance
    ant_str = '1_(-3,11)'
    ant_pairs = [(1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    ant_str = '(-1,3)_11'
    ant_pairs = [(3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific antenna number
    ant_str = '1,-3'
    ant_pairs = [(0, 1), (1, 2), (1, 6), (1, 7), (1, 8), (1, 11),
                 (1, 14), (1, 18), (1, 19), (1, 20), (1, 21),
                 (1, 22), (1, 23), (1, 24), (1, 26), (1, 27)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific baseline
    ant_str = '1,-1_3'
    ant_pairs = [(0, 1), (1, 2), (1, 6), (1, 7), (1, 8), (1, 11),
                 (1, 14), (1, 18), (1, 19), (1, 20), (1, 21),
                 (1, 22), (1, 23), (1, 24), (1, 26), (1, 27)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations and - for avoidance
    ant_str = '1l_(-3r,11l)'
    ant_pairs = [(1, 11)]
    pols = ['ll']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Test pseudo-Stokes params with select
    ant_str = 'pi,pQ'
    pols = ['pQ', 'pI']
    uv.polarization_array = np.array([4, 3, 2, 1])
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Test ant_str = 'auto' on file with auto correlations
    uv = UVData()
    testfile = os.path.join(DATA_PATH, 'hera_testfile')
    uvtest.checkWarnings(uv.read_miriad, [testfile], nwarnings=1,
                         message='Altitude is not')

    ant_str = 'auto'
    ant_pairs = [(9, 9), (10, 10), (20, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Test cross correlation extraction on data with auto + cross
    ant_str = 'cross'
    ant_pairs = [(9, 10), (9, 20), (10, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove only polarization of single baseline
    ant_str = 'all,-9x_10x'
    ant_pairs = [(9, 9), (9, 20), (10, 10), (10, 20), (20, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Test appending all to beginning of strings that start with -
    ant_str = '-9'
    ant_pairs = [(10, 10), (10, 20), (20, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())


def test_set_uvws_from_antenna_pos():
    # Test set_uvws_from_antenna_positions function with phased data
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, '1133866760.uvfits')
    uv_object.read_uvfits(testfile)
    orig_uvw_array = np.copy(uv_object.uvw_array)
    pytest.raises(ValueError, uv_object.set_uvws_from_antenna_positions)
    uvtest.checkWarnings(
        pytest.raises,
        [ValueError, uv_object.set_uvws_from_antenna_positions, True, 'xyz'],
        message='Warning: Data will be unphased'
    )
    uvtest.checkWarnings(
        pytest.raises,
        [ValueError, uv_object.set_uvws_from_antenna_positions, True, 'gcrs', 'xyz'],
        message='Warning: Data will be unphased'
    )
    uvtest.checkWarnings(
        uv_object.set_uvws_from_antenna_positions,
        [True, 'gcrs', 'gcrs'],
        message='Warning: Data will be unphased'
    )
    max_diff = np.amax(np.absolute(np.subtract(orig_uvw_array,
                                               uv_object.uvw_array)))
    assert np.isclose(max_diff, 0., atol=2)


def test_get_antenna_redundancies():
    uv0 = UVData()
    uv0.read_uvfits(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))

    red_gps, centers, lengths = uv0.get_antenna_redundancies(include_autos=False,
                                                             conjugate_bls=True)

    # assert all baselines are in the data (because it's conjugated to match)
    for i, gp in enumerate(red_gps):
        for bl in gp:
            assert bl in uv0.baseline_array
    old_bl_array = np.copy(uv0.baseline_array)

    # conjugate data differently
    uv0.conjugate_bls(convention='ant1<ant2')
    new_red_gps, new_centers, new_lengths = uv0.get_antenna_redundancies(
        include_autos=False)

    # new and old baseline Numbers are not the same (different conjugation)
    assert not np.allclose(uv0.baseline_array, old_bl_array)

    # all redundancy info is the same
    assert red_gps == new_red_gps
    assert np.allclose(centers, new_centers)
    assert np.allclose(lengths, new_lengths)


def test_redundancy_contract_expand():
    # Test that a UVData object can be reduced to one baseline from each redundant group
    # and restored to its original form.

    uv0 = UVData()
    uv0.read_uvfits(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))

    tol = 0.02   # Fails at lower precision because some baselines falling into multiple redundant groups

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_antenna_redundancies(tol=tol, conjugate_bls=True)
    for i, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            uv0.data_array[inds] += complex(i)

    uv2 = uv0.compress_by_redundancy(tol=tol, inplace=False)

    # Compare in-place to separated compression.
    uv3 = copy.deepcopy(uv0)
    uv3.compress_by_redundancy(tol=tol)
    assert uv2 == uv3

    # check inflating gets back to the original
    uvtest.checkWarnings(
        uv2.inflate_by_redundancy,
        [tol],
        nwarnings=2,
        category=[DeprecationWarning, UserWarning],
        message=['The default for the `center` keyword has changed.',
                 'Missing some redundant groups. Filling in available data.']
    )
    uv2.history = uv0.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv0.reorder_blts()
    uv2.reorder_blts()
    uv2._uvw_array.tols = [0, tol]

    assert uv2 == uv0

    uv3 = uv2.compress_by_redundancy(tol=tol, inplace=False)
    uvtest.checkWarnings(
        uv3.inflate_by_redundancy,
        [tol],
        nwarnings=2,
        category=[DeprecationWarning, UserWarning],
        message=['The default for the `center` keyword has changed.',
                 'Missing some redundant groups. Filling in available data.']
    )
    # Confirm that we get the same result looping inflate -> compress -> inflate.
    uv3.reorder_blts()

    uv2.history = uv3.history
    assert uv2 == uv3


def test_redundancy_contract_expand_nblts_not_nbls_times_ntimes():
    uv0 = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv0.read_uvfits, [testfile], message='Telescope EVLA is not')

    # check that Nblts != Nbls * Ntimes
    assert uv0.Nblts != uv0.Nbls * uv0.Ntimes

    tol = 1.0

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_antenna_redundancies(tol=tol,
                                                             conjugate_bls=True)
    for i, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds, ...] *= 0
            uv0.data_array[inds, ...] += complex(i)

    uv2 = uv0.compress_by_redundancy(tol=tol, inplace=False)

    # check inflating gets back to the original
    uvtest.checkWarnings(uv2.inflate_by_redundancy, {tol: tol},
                         nwarnings=2, category=[DeprecationWarning, UserWarning],
                         message=['The default for the `center` keyword has changed.',
                                  'Missing some redundant groups. Filling in available data.'])

    uv2.history = uv0.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv0.reorder_blts()
    uv2.reorder_blts()
    uv2._uvw_array.tols = [0, tol]

    blt_inds = []
    missing_inds = []
    for bl, t in zip(uv0.baseline_array, uv0.time_array):
        antpair = uv2.baseline_to_antnums(bl)
        if (bl, t) in zip(uv2.baseline_array, uv2.time_array):
            this_ind = np.where((uv2.baseline_array == bl) & (uv2.time_array == t))[0]
            blt_inds.append(this_ind[0])
        else:
            # this is missing because of the compress_by_redundancy step
            missing_inds.append(np.where((uv0.baseline_array == bl) & (uv0.time_array == t))[0])

    uv3 = uv2.select(blt_inds=blt_inds, inplace=False)

    orig_inds_keep = list(np.arange(uv0.Nblts))
    for ind in missing_inds:
        orig_inds_keep.remove(ind)
    uv1 = uv0.select(blt_inds=orig_inds_keep, inplace=False)

    assert uv3 == uv1


def test_compress_redundancy_metadata_only():
    uv0 = UVData()
    uv0.read_uvfits(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))
    tol = 0.01

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_antenna_redundancies(tol=tol,
                                                             conjugate_bls=True)
    for i, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            uv0.data_array[inds] += complex(i)

    uv2 = copy.deepcopy(uv0)
    uv2.data_array = None
    uv2.flag_array = None
    uv2.nsample_array = None
    uv2.compress_by_redundancy(tol=tol, inplace=True, metadata_only=True)

    uv0.compress_by_redundancy(tol=tol)
    uv0.data_array = None
    uv0.flag_array = None
    uv0.nsample_array = None
    assert uv0 == uv2


def test_redundancy_missing_groups():
    # Check that if I try to inflate a compressed UVData that is missing redundant groups, it will
    # raise the right warnings and fill only what data are available.

    uv0 = UVData()
    uv0.read_uvfits(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))
    tol = 0.02
    Nselect = 19

    uv0.compress_by_redundancy(tol=tol)
    fname = 'temp_hera19_missingreds.uvfits'

    bls = np.unique(uv0.baseline_array)[:Nselect]         # First twenty baseline groups
    uv0.select(bls=[uv0.baseline_to_antnums(bl) for bl in bls])
    uv0.write_uvfits(fname)
    uv1 = UVData()
    uv1.read_uvfits(fname)
    os.remove(fname)

    assert uv0 == uv1  # Check that writing compressed files causes no issues.

    uvtest.checkWarnings(
        uv1.inflate_by_redundancy,
        [tol],
        nwarnings=2,
        category=[DeprecationWarning, UserWarning],
        message=['The default for the `center` keyword has changed.',
                 'Missing some redundant groups. Filling in available data.']
    )

    uv2 = uv1.compress_by_redundancy(tol=tol, inplace=False)

    assert np.unique(uv2.baseline_array).size == Nselect


def test_quick_redundant_vs_redundant_test_array():
    """Verify the quick redundancy calc returns the same groups as a known array."""
    uv = UVData()
    uv.read_uvfits(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))
    uv.select(times=uv.time_array[0])
    uv.unphase_to_drift()
    uvtest.checkWarnings(uv.conjugate_bls, func_kwargs={'convention': 'u>0', 'use_enu': True},
                         message=['The default for the `center`'],
                         nwarnings=1, category=DeprecationWarning)
    tol = 0.05
    # a quick and dirty redundancy calculation
    unique_bls, baseline_inds = np.unique(uv.baseline_array, return_index=True)
    uvw_vectors = np.take(uv.uvw_array, baseline_inds, axis=0)
    uvw_diffs = np.expand_dims(uvw_vectors, axis=0) - np.expand_dims(uvw_vectors, axis=1)
    uvw_diffs = np.linalg.norm(uvw_diffs, axis=2)

    reds = np.where(uvw_diffs < tol, unique_bls, 0)
    reds = np.ma.masked_where(reds == 0, reds)
    groups = []
    for bl in reds:
        grp = []
        grp.extend(bl.compressed())
        for other_bls in reds:
            if set(reds.compressed()).issubset(other_bls.compressed()):
                grp.extend(other_bls.compressed())
        grp = np.unique(grp).tolist()
        groups.append(grp)

    pad = len(max(groups, key=len))
    groups = np.array([i + [-1] * (pad - len(i)) for i in groups])
    groups = np.unique(groups, axis=0)
    groups = [[bl for bl in grp if bl != -1] for grp in groups]
    groups.sort(key=len)

    redundant_groups, centers, lengths, conj_inds = uv.get_baseline_redundancies(tol=tol)
    redundant_groups.sort(key=len)
    assert groups == redundant_groups


def test_redundancy_finder_when_nblts_not_nbls_times_ntimes():
    """Test the redundancy finder functions when Nblts != Nbls * Ntimes."""
    tol = 1  # meter
    uv = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile], message='Telescope EVLA is not')
    uvtest.checkWarnings(uv.conjugate_bls, func_kwargs={'convention': 'u>0', 'use_enu': True},
                         message=['The default for the `center`'],
                         nwarnings=1, category=DeprecationWarning)
    # check that Nblts != Nbls * Ntimes
    assert uv.Nblts != uv.Nbls * uv.Ntimes

    # a quick and dirty redundancy calculation
    unique_bls, baseline_inds = np.unique(uv.baseline_array, return_index=True)
    uvw_vectors = np.take(uv.uvw_array, baseline_inds, axis=0)
    uvw_diffs = np.expand_dims(uvw_vectors, axis=0) - np.expand_dims(uvw_vectors, axis=1)
    uvw_diffs = np.linalg.norm(uvw_diffs, axis=2)

    reds = np.where(uvw_diffs < tol, unique_bls, 0)
    reds = np.ma.masked_where(reds == 0, reds)
    groups = []
    for bl in reds:
        grp = []
        grp.extend(bl.compressed())
        for other_bls in reds:
            if set(reds.compressed()).issubset(other_bls.compressed()):
                grp.extend(other_bls.compressed())
        grp = np.unique(grp).tolist()
        groups.append(grp)

    pad = len(max(groups, key=len))
    groups = np.array([i + [-1] * (pad - len(i)) for i in groups])
    groups = np.unique(groups, axis=0)
    groups = [[bl for bl in grp if bl != -1] for grp in groups]
    groups.sort(key=len)

    redundant_groups, centers, lengths, conj_inds = uv.get_baseline_redundancies(tol=tol)
    redundant_groups.sort(key=len)
    assert groups == redundant_groups


def test_overlapping_data_add():
    # read in test data
    uv = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile], message='Telescope EVLA is not')

    # slice into four objects
    blts1 = np.arange(500)
    blts2 = np.arange(500, 1360)
    uv1 = uv.select(polarizations=[-1, -2], blt_inds=blts1, inplace=False)
    uv2 = uv.select(polarizations=[-3, -4], blt_inds=blts1, inplace=False)
    uv3 = uv.select(polarizations=[-1, -2], blt_inds=blts2, inplace=False)
    uv4 = uv.select(polarizations=[-3, -4], blt_inds=blts2, inplace=False)

    # combine and check for equality
    uvfull = uv1 + uv2
    uvfull += uv3
    uvfull += uv4
    extra_history = ("Downselected to specific baseline-times, polarizations using pyuvdata. "
                     "Combined data along polarization axis using pyuvdata. Combined data along "
                     "baseline-time axis using pyuvdata. Overwrote invalid data using pyuvdata.")
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history)
    uvfull.history = uv.history  # make histories match
    assert uv == uvfull

    # check combination not-in-place
    uvfull = uv1 + uv2
    uvfull += uv3
    uvfull = uvfull + uv4
    uvfull.history = uv.history  # make histories match
    assert uv == uvfull

    # test raising error for adding objects incorrectly (i.e., having the object
    # with data to be overwritten come second)
    uvfull = uv1 + uv2
    uvfull += uv3
    pytest.raises(ValueError, uv4.__iadd__, uvfull)
    pytest.raises(ValueError, uv4.__add__, uv4, uvfull)

    # write individual objects out, and make sure that we can read in the list
    uv1_out = os.path.join(DATA_PATH, "uv1.uvfits")
    uv1.write_uvfits(uv1_out)
    uv2_out = os.path.join(DATA_PATH, "uv2.uvfits")
    uv2.write_uvfits(uv2_out)
    uv3_out = os.path.join(DATA_PATH, "uv3.uvfits")
    uv3.write_uvfits(uv3_out)
    uv4_out = os.path.join(DATA_PATH, "uv4.uvfits")
    uv4.write_uvfits(uv4_out)

    uvfull = UVData()
    uvtest.checkWarnings(uvfull.read, [[uv1_out, uv2_out, uv3_out, uv4_out]],
                         nwarnings=4, message='Telescope EVLA is not')
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history)
    uvfull.history = uv.history  # make histories match
    assert uvfull == uv

    # clean up after ourselves
    os.remove(uv1_out)
    os.remove(uv2_out)
    os.remove(uv3_out)
    os.remove(uv4_out)

    return


def test_lsts_from_time_with_only_unique():
    """Test `set_lsts_from_time_array` with only unique values is identical to full array."""
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uv = UVData()
    uvtest.checkWarnings(uv.read_miriad, [miriad_file],
                         known_warning='miriad')
    lat, lon, alt = uv.telescope_location_lat_lon_alt_degrees
    # calculate the lsts for all elements in time array
    full_lsts = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    # use `set_lst_from_time_array` to set the uv.lst_array using only unique values
    uv.set_lsts_from_time_array()
    assert np.array_equal(full_lsts, uv.lst_array)
