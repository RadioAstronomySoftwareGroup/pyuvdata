# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvdata object.

"""
from __future__ import absolute_import, division, print_function

import nose.tools as nt
import os
import numpy as np
import copy
import six
from astropy.time import Time
from astropy.coordinates import Angle

from pyuvdata import UVData
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

if six.PY2:
    nt.assert_count_equal = nt.assert_items_equal


class TestUVDataInit(object):
    def setUp(self):
        """Setup for basic parameter, property and iterator tests."""
        self.required_parameters = ['_data_array', '_nsample_array',
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

        self.required_properties = ['data_array', 'nsample_array',
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

        self.extra_parameters = ['_extra_keywords', '_antenna_positions',
                                 '_x_orientation', '_antenna_diameters',
                                 '_gst0', '_rdate', '_earth_omega', '_dut1',
                                 '_timesys', '_uvplane_reference_time',
                                 '_phase_center_ra', '_phase_center_dec',
                                 '_phase_center_epoch', '_phase_center_frame']

        self.extra_properties = ['extra_keywords', 'antenna_positions',
                                 'x_orientation', 'antenna_diameters', 'gst0',
                                 'rdate', 'earth_omega', 'dut1', 'timesys',
                                 'uvplane_reference_time',
                                 'phase_center_ra', 'phase_center_dec',
                                 'phase_center_epoch', 'phase_center_frame']

        self.other_properties = ['telescope_location_lat_lon_alt',
                                 'telescope_location_lat_lon_alt_degrees',
                                 'phase_center_ra_degrees', 'phase_center_dec_degrees',
                                 'pyuvdata_version_str']

        self.uv_object = UVData()

    def teardown(self):
        """Test teardown: delete object."""
        del(self.uv_object)

    def test_order_pols(self):
        test_uv1 = UVData()
        testfile = os.path.join(
            DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
        uvtest.checkWarnings(test_uv1.read_uvfits, [testfile],
                             message='Telescope EVLA is not')
        test_uv1.order_pols(order='AIPS')
        # check that we have aips ordering
        aips_pols = np.array([-1, -2, -3, -4]).astype(int)
        nt.assert_true(np.all(test_uv1.polarization_array == aips_pols))
        test_uv2 = copy.deepcopy(test_uv1)
        test_uv2.order_pols(order='CASA')
        casa_pols = np.array([-1, -3, -4, -2]).astype(int)
        nt.assert_true(np.all(test_uv2.polarization_array == casa_pols))
        order = np.array([0, 2, 3, 1])
        nt.assert_true(np.all(test_uv2.data_array == test_uv1.data_array[:, :, :, order]))
        nt.assert_true(np.all(test_uv2.flag_array == test_uv1.flag_array[:, :, :, order]))
        # check that we have casa ordering
        test_uv2.order_pols(order='AIPS')
        # check that we have aips ordering again
        nt.assert_equal(test_uv1, test_uv2)
        uvtest.checkWarnings(test_uv2.order_pols, ['unknown'], message='Invalid order supplied')
        del(test_uv1)
        del(test_uv2)

    def test_parameter_iter(self):
        "Test expected parameters."
        all = []
        for prop in self.uv_object:
            all.append(prop)
        for a in self.required_parameters + self.extra_parameters:
            nt.assert_true(a in all, msg='expected attribute ' + a
                           + ' not returned in object iterator')

    def test_required_parameter_iter(self):
        "Test expected required parameters."
        required = []
        for prop in self.uv_object.required():
            required.append(prop)
        for a in self.required_parameters:
            nt.assert_true(a in required, msg='expected attribute ' + a
                           + ' not returned in required iterator')

    def test_extra_parameter_iter(self):
        "Test expected optional parameters."
        extra = []
        for prop in self.uv_object.extra():
            extra.append(prop)
        for a in self.extra_parameters:
            nt.assert_true(a in extra, msg='expected attribute ' + a
                           + ' not returned in extra iterator')

    def test_unexpected_parameters(self):
        "Test for extra parameters."
        expected_parameters = self.required_parameters + self.extra_parameters
        attributes = [i for i in self.uv_object.__dict__.keys() if i[0] == '_']
        for a in attributes:
            nt.assert_true(a in expected_parameters,
                           msg='unexpected parameter ' + a + ' found in UVData')

    def test_unexpected_attributes(self):
        "Test for extra attributes."
        expected_attributes = self.required_properties + \
            self.extra_properties + self.other_properties
        attributes = [i for i in self.uv_object.__dict__.keys() if i[0] != '_']
        for a in attributes:
            nt.assert_true(a in expected_attributes,
                           msg='unexpected attribute ' + a + ' found in UVData')

    def test_properties(self):
        "Test that properties can be get and set properly."
        prop_dict = dict(list(zip(self.required_properties + self.extra_properties,
                                  self.required_parameters + self.extra_parameters)))
        for k, v in prop_dict.items():
            rand_num = np.random.rand()
            setattr(self.uv_object, k, rand_num)
            this_param = getattr(self.uv_object, v)
            try:
                nt.assert_equal(rand_num, this_param.value)
            except(AssertionError):
                print('setting {prop_name} to a random number failed'.format(prop_name=k))
                raise(AssertionError)


class TestUVDataBasicMethods(object):
    def setUp(self):
        """Setup for tests of basic methods."""
        self.uv_object = UVData()
        self.testfile = os.path.join(
            DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
        uvtest.checkWarnings(self.uv_object.read_uvfits, [self.testfile],
                             message='Telescope EVLA is not')
        self.uv_object2 = copy.deepcopy(self.uv_object)

    def teardown(self):
        """Test teardown: delete objects."""
        del(self.uv_object)
        del(self.uv_object2)

    def test_equality(self):
        """Basic equality test."""
        nt.assert_equal(self.uv_object, self.uv_object)

    def test_check(self):
        """Test simple check function."""
        nt.assert_true(self.uv_object.check())
        # Check variety of special cases
        self.uv_object.Nants_data += 1
        nt.assert_raises(ValueError, self.uv_object.check)
        self.uv_object.Nants_data -= 1
        self.uv_object.Nbls += 1
        nt.assert_raises(ValueError, self.uv_object.check)
        self.uv_object.Nbls -= 1
        self.uv_object.Ntimes += 1
        nt.assert_raises(ValueError, self.uv_object.check)
        self.uv_object.Ntimes -= 1

        # Check case where all data is autocorrelations
        # Currently only test files that have autos are fhd files
        testdir = os.path.join(DATA_PATH, 'fhd_vis_data/')
        file_list = [testdir + '1061316296_flags.sav',
                     testdir + '1061316296_vis_XX.sav',
                     testdir + '1061316296_params.sav',
                     testdir + '1061316296_layout.sav',
                     testdir + '1061316296_settings.txt']

        uvtest.checkWarnings(self.uv_object.read_fhd, [file_list], known_warning='fhd')

        self.uv_object.select(blt_inds=np.where(self.uv_object.ant_1_array
                                                == self.uv_object.ant_2_array)[0])
        nt.assert_true(self.uv_object.check())

        # test auto and cross corr uvw_array
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA"))
        autos = np.isclose(uvd.ant_1_array - uvd.ant_2_array, 0.0)
        auto_inds = np.where(autos)[0]
        cross_inds = np.where(~autos)[0]

        # make auto have non-zero uvw coords, assert ValueError
        uvd.uvw_array[auto_inds[0], 0] = 0.1
        nt.assert_raises(ValueError, uvd.check)

        # make cross have |uvw| zero, assert ValueError
        uvd.read_miriad(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA"))
        uvd.uvw_array[cross_inds[0]][:] = 0.0
        nt.assert_raises(ValueError, uvd.check)

    def test_nants_data_telescope(self):
        self.uv_object.Nants_data = self.uv_object.Nants_telescope - 1
        nt.assert_true(self.uv_object.check)
        self.uv_object.Nants_data = self.uv_object.Nants_telescope + 1
        nt.assert_raises(ValueError, self.uv_object.check)

    def test_converttofiletype(self):
        fhd_obj = self.uv_object._convert_to_filetype('fhd')
        self.uv_object._convert_from_filetype(fhd_obj)
        nt.assert_equal(self.uv_object, self.uv_object2)

        nt.assert_raises(
            ValueError, self.uv_object._convert_to_filetype, 'foo')


class TestBaselineAntnumMethods(object):
    """Setup for tests on antnum, baseline conversion."""

    def setup(self):
        self.uv_object = UVData()
        self.uv_object.Nants_telescope = 128
        self.uv_object2 = UVData()
        self.uv_object2.Nants_telescope = 2049

    def teardown(self):
        """Test teardown: delete objects."""
        del(self.uv_object)
        del(self.uv_object2)

    def test_baseline_to_antnums(self):
        """Test baseline to antnum conversion for 256 & larger conventions."""
        nt.assert_equal(self.uv_object.baseline_to_antnums(67585), (0, 0))
        nt.assert_raises(
            Exception, self.uv_object2.baseline_to_antnums, 67585)

        ant_pairs = [(10, 20), (280, 310)]
        for pair in ant_pairs:
            if np.max(np.array(pair)) < 255:
                bl = self.uv_object.antnums_to_baseline(
                    pair[0], pair[1], attempt256=True)
                ant_pair_out = self.uv_object.baseline_to_antnums(bl)
                nt.assert_equal(pair, ant_pair_out)

            bl = self.uv_object.antnums_to_baseline(
                pair[0], pair[1], attempt256=False)
            ant_pair_out = self.uv_object.baseline_to_antnums(bl)
            nt.assert_equal(pair, ant_pair_out)

    def test_antnums_to_baselines(self):
        """Test antums to baseline conversion for 256 & larger conventions."""
        nt.assert_equal(self.uv_object.antnums_to_baseline(0, 0), 67585)
        nt.assert_equal(self.uv_object.antnums_to_baseline(257, 256), 594177)
        nt.assert_equal(self.uv_object.baseline_to_antnums(594177), (257, 256))
        # Check attempt256
        nt.assert_equal(self.uv_object.antnums_to_baseline(
            0, 0, attempt256=True), 257)
        nt.assert_equal(self.uv_object.antnums_to_baseline(257, 256), 594177)
        uvtest.checkWarnings(self.uv_object.antnums_to_baseline, [257, 256],
                             {'attempt256': True}, message='found > 256 antennas')
        nt.assert_raises(
            Exception, self.uv_object2.antnums_to_baseline, 0, 0)
        # check a len-1 array returns as an array
        ant1 = np.array([1])
        ant2 = np.array([2])
        nt.assert_true(isinstance(self.uv_object.antnums_to_baseline(ant1, ant2), np.ndarray))


def test_known_telescopes():
    """Test known_telescopes method returns expected results."""
    uv_object = UVData()
    known_telescopes = ['PAPER', 'HERA', 'MWA']
    nt.assert_equal(known_telescopes.sort(),
                    uv_object.known_telescopes().sort())


def test_HERA_diameters():
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')

    uv_in.telescope_name = 'HERA'
    uvtest.checkWarnings(uv_in.set_telescope_params, message='antenna_diameters '
                         'is not set. Using known values for HERA.')

    nt.assert_equal(uv_in.telescope_name, 'HERA')
    nt.assert_true(uv_in.antenna_diameters is not None)

    uv_in.check()


def test_generic_read():
    uv_in = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_in.read, [uvfits_file], {'read_data': False},
                         message='Telescope EVLA is not')
    unique_times = np.unique(uv_in.time_array)

    nt.assert_raises(ValueError, uv_in.read, uvfits_file, times=unique_times[0:2],
                     time_range=[unique_times[0], unique_times[1]])

    nt.assert_raises(ValueError, uv_in.read, uvfits_file,
                     antenna_nums=uv_in.antenna_numbers[0],
                     antenna_names=uv_in.antenna_names[1])

    nt.assert_raises(ValueError, uv_in.read, 'foo')


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
    nt.assert_equal(UV_raw, UV_phase)

    # check that phase + unphase work using gcrs
    UV_phase.phase(Angle('5d').rad, Angle('30d').rad, phase_frame='gcrs')
    UV_phase.unphase_to_drift()
    nt.assert_equal(UV_raw, UV_phase)

    # check that phase + unphase work using a different epoch
    UV_phase.phase(Angle('180d').rad, Angle('90d'), epoch=Time('2010-01-01T00:00:00', format='isot', scale='utc'))
    UV_phase.unphase_to_drift()
    nt.assert_equal(UV_raw, UV_phase)

    # check that phase + unphase work with one baseline
    UV_raw_small = UV_raw.select(blt_inds=[0], inplace=False)
    UV_phase_small = copy.deepcopy(UV_raw_small)
    UV_phase_small.phase(Angle('23h').rad, Angle('15d').rad)
    UV_phase_small.unphase_to_drift()
    nt.assert_equal(UV_raw_small, UV_phase_small)

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
    nt.assert_true(np.allclose(UV_phase2.uvw_array, UV_phase.uvw_array, atol=1e-3))
    # the data array are just multiplied by the w's for phasing, so a difference
    # at the 1e-3 level makes the data array different at that level too.
    # -> change the tolerance on data_array for this test
    UV_phase2._data_array.tols = (0, 1e-3)
    nt.assert_equal(UV_phase2, UV_phase)

    # check that phase + unphase gets back to raw using antpos
    UV_phase.unphase_to_drift(use_ant_pos=True)
    nt.assert_equal(UV_raw_new, UV_phase)

    # check that phasing to zenith with one timestamp has small changes
    # (it won't be identical because of precession/nutation changing the coordinate axes)
    # use gcrs rather than icrs to reduce differences (don't include abberation)
    UV_raw_small = UV_raw.select(times=UV_raw.time_array[0], inplace=False)
    UV_phase_simple_small = copy.deepcopy(UV_raw_small)
    UV_phase_simple_small.phase_to_time(time=Time(UV_raw.time_array[0], format='jd'),
                                        phase_frame='gcrs')

    # it's unclear to me how close this should be...
    nt.assert_true(np.allclose(UV_phase_simple_small.uvw_array, UV_raw_small.uvw_array, atol=1e-2))

    # check error if not passing a Time object to phase_to_time
    nt.assert_raises(TypeError, UV_raw.phase_to_time, UV_raw.time_array[0])

    # check errors when trying to unphase drift or unknown data
    nt.assert_raises(ValueError, UV_raw.unphase_to_drift)
    UV_raw.set_unknown_phase_type()
    nt.assert_raises(ValueError, UV_raw.unphase_to_drift)
    UV_raw.set_drift()

    # check errors when trying to phase phased or unknown data
    UV_phase.phase(0., 0., epoch="J2000")
    nt.assert_raises(ValueError, UV_phase.phase, 0., 0., epoch="J2000")
    nt.assert_raises(ValueError, UV_phase.phase_to_time,
                     UV_phase.time_array[0])

    UV_phase.set_unknown_phase_type()
    nt.assert_raises(ValueError, UV_phase.phase, 0., 0., epoch="J2000")
    nt.assert_raises(ValueError, UV_phase.phase_to_time,
                     UV_phase.time_array[0])

    # check errors when trying to phase to an unsupported frame
    UV_phase = copy.deepcopy(UV_raw)
    nt.assert_raises(ValueError, UV_phase.phase, 0., 0., epoch="J2000", phase_frame='cirs')

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
    nt.assert_true(np.allclose(uvd1_drift.uvw_array, uvd2_drift.uvw_array, atol=2e-2))
    nt.assert_true(np.allclose(uvd1_drift_antpos.uvw_array, uvd2_drift_antpos.uvw_array))

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
    nt.assert_true(np.allclose(uvd1.uvw_array, uvd2_rephase.uvw_array, atol=2e-2))
    nt.assert_true(np.allclose(uvd1.uvw_array, uvd2_rephase_antpos.uvw_array, atol=5e-3))

    # rephase the drift objects to the original pointing and verify that they match
    uvd1_drift.phase(uvd1.phase_center_ra, uvd1.phase_center_dec,
                     uvd1.phase_center_epoch, phase_frame='gcrs')
    uvd1_drift_antpos.phase(uvd1.phase_center_ra, uvd1.phase_center_dec,
                            uvd1.phase_center_epoch, phase_frame='gcrs',
                            use_ant_pos=True)

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the external
    # phasing test. See the phasing memo in docs/references for details
    nt.assert_true(np.allclose(uvd1.uvw_array, uvd1_drift.uvw_array, atol=1e-4))
    nt.assert_true(np.allclose(uvd1.uvw_array, uvd1_drift_antpos.uvw_array, atol=5e-3))

    uvd2_drift.phase(uvd2.phase_center_ra, uvd2.phase_center_dec,
                     uvd2.phase_center_epoch, phase_frame='gcrs')
    uvd2_drift_antpos.phase(uvd2.phase_center_ra, uvd2.phase_center_dec,
                            uvd2.phase_center_epoch, phase_frame='gcrs',
                            use_ant_pos=True)

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the external
    # phasing test. See the phasing memo in docs/references for details
    nt.assert_true(np.allclose(uvd2.uvw_array, uvd2_drift.uvw_array, atol=1e-4))
    nt.assert_true(np.allclose(uvd2.uvw_array, uvd2_drift_antpos.uvw_array, atol=2e-2))


def test_set_phase_unknown():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [
                         testfile], message='Telescope EVLA is not')

    uv_object.set_unknown_phase_type()
    nt.assert_equal(uv_object.phase_type, 'unknown')
    nt.assert_false(uv_object._phase_center_epoch.required)
    nt.assert_false(uv_object._phase_center_ra.required)
    nt.assert_false(uv_object._phase_center_dec.required)
    nt.assert_true(uv_object.check())


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
    nt.assert_equal(len(blt_inds), uv_object2.Nblts)

    # verify that histories are different
    nt.assert_false(uvutils._check_histories(old_history, uv_object2.history))

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific baseline-times using pyuvdata.',
                                            uv_object2.history))

    nt.assert_true(np.all(selected_data == uv_object2.data_array))

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(blt_inds=blt_inds[np.newaxis, :])
    nt.assert_equal(len(blt_inds), uv_object2.Nblts)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific baseline-times using pyuvdata.',
                                            uv_object2.history))
    nt.assert_true(np.all(selected_data == uv_object2.data_array))

    # check for errors associated with out of bounds indices
    nt.assert_raises(ValueError, uv_object.select, blt_inds=np.arange(-10, -5))
    nt.assert_raises(ValueError, uv_object.select, blt_inds=np.arange(
        uv_object.Nblts + 1, uv_object.Nblts + 10))


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

    nt.assert_equal(len(ants_to_keep), uv_object2.Nants_data)
    nt.assert_equal(Nblts_selected, uv_object2.Nblts)
    for ant in ants_to_keep:
        nt.assert_true(
            ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array)
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        nt.assert_true(ant in ants_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific antennas using pyuvdata.',
                                            uv_object2.history))

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(antenna_nums=ants_to_keep[np.newaxis, :])

    nt.assert_equal(len(ants_to_keep), uv_object2.Nants_data)
    nt.assert_equal(Nblts_selected, uv_object2.Nblts)
    for ant in ants_to_keep:
        nt.assert_true(
            ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array)
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        nt.assert_true(ant in ants_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific antennas using pyuvdata.',
                                            uv_object2.history))

    # now test using antenna_names to specify antennas to keep
    uv_object3 = copy.deepcopy(uv_object)
    ants_to_keep = np.array(sorted(list(ants_to_keep)))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=ant_names)

    nt.assert_equal(uv_object2, uv_object3)

    # check that it also works with higher dimension array
    uv_object3 = copy.deepcopy(uv_object)
    ants_to_keep = np.array(sorted(list(ants_to_keep)))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=[ant_names])

    nt.assert_equal(uv_object2, uv_object3)

    # check for errors associated with antennas not included in data, bad names or providing numbers and names
    nt.assert_raises(ValueError, uv_object.select,
                     antenna_nums=np.max(unique_ants) + np.arange(1, 3))
    nt.assert_raises(ValueError, uv_object.select, antenna_names='test1')
    nt.assert_raises(ValueError, uv_object.select,
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

    nt.assert_equal(len(new_unique_ants), uv_object2.Nants_data)
    nt.assert_equal(Nblts_selected, uv_object2.Nblts)
    for ant in new_unique_ants:
        nt.assert_true(
            ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array)
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        nt.assert_true(ant in new_unique_ants)
    for pair in sorted_pairs_to_keep:
        nt.assert_true(pair in sorted_pairs_object2)
    for pair in sorted_pairs_object2:
        nt.assert_true(pair in sorted_pairs_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific baselines using pyuvdata.',
                                            uv_object2.history))

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

    nt.assert_equal(len(new_unique_ants), uv_object2.Nants_data)
    nt.assert_equal(Nblts_selected, uv_object2.Nblts)
    for ant in new_unique_ants:
        nt.assert_true(
            ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array)
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        nt.assert_true(ant in new_unique_ants)
    for bl in sorted_bls_to_keep:
        nt.assert_true(bl in sorted_pairs_object2)
    for bl in sorted_pairs_object2:
        nt.assert_true(bl in sorted_bls_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific baselines, polarizations using pyuvdata.',
                                            uv_object2.history))

    # check that you can use numpy integers with out errors:
    first_ants = list(map(np.int32, [6, 2, 7, 2, 21, 27, 8]))
    second_ants = list(map(np.int32, [0, 20, 8, 1, 2, 3, 22]))
    ant_pairs_to_keep = list(zip(first_ants, second_ants))

    uv_object2 = uv_object.select(bls=ant_pairs_to_keep, inplace=False)
    sorted_pairs_object2 = [sort_bl(p) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]

    nt.assert_equal(len(new_unique_ants), uv_object2.Nants_data)
    nt.assert_equal(Nblts_selected, uv_object2.Nblts)
    for ant in new_unique_ants:
        nt.assert_true(
            ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array)
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        nt.assert_true(ant in new_unique_ants)
    for pair in sorted_pairs_to_keep:
        nt.assert_true(pair in sorted_pairs_object2)
    for pair in sorted_pairs_object2:
        nt.assert_true(pair in sorted_pairs_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific baselines using pyuvdata.',
                                            uv_object2.history))

    # check that you can specify a single pair without errors
    uv_object2.select(bls=(0, 6))
    sorted_pairs_object2 = [sort_bl(p) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]
    nt.assert_equal(list(set(sorted_pairs_object2)), [(0, 6)])

    # check for errors associated with antenna pairs not included in data and bad inputs
    nt.assert_raises(ValueError, uv_object.select,
                     bls=list(zip(first_ants, second_ants)) + [0, 6])
    nt.assert_raises(ValueError, uv_object.select,
                     bls=[(uv_object.antenna_names[0], uv_object.antenna_names[1])])
    nt.assert_raises(ValueError, uv_object.select, bls=(5, 1))
    nt.assert_raises(ValueError, uv_object.select, bls=(0, 5))
    nt.assert_raises(ValueError, uv_object.select, bls=(27, 27))
    nt.assert_raises(ValueError, uv_object.select, bls=(6, 0, 'RR'), polarizations='RR')
    nt.assert_raises(ValueError, uv_object.select, bls=(6, 0, 8))


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

    nt.assert_equal(len(times_to_keep), uv_object2.Ntimes)
    nt.assert_equal(Nblts_selected, uv_object2.Nblts)
    for t in times_to_keep:
        nt.assert_true(t in uv_object2.time_array)
    for t in np.unique(uv_object2.time_array):
        nt.assert_true(t in times_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific times using pyuvdata.',
                                            uv_object2.history))
    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(times=times_to_keep[np.newaxis, :])

    nt.assert_equal(len(times_to_keep), uv_object2.Ntimes)
    nt.assert_equal(Nblts_selected, uv_object2.Nblts)
    for t in times_to_keep:
        nt.assert_true(t in uv_object2.time_array)
    for t in np.unique(uv_object2.time_array):
        nt.assert_true(t in times_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific times using pyuvdata.',
                                            uv_object2.history))

    # check for errors associated with times not included in data
    nt.assert_raises(ValueError, uv_object.select, times=[
                     np.min(unique_times) - uv_object.integration_time[0]])


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

    nt.assert_equal(len(freqs_to_keep), uv_object2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in uv_object2.freq_array)
    for f in np.unique(uv_object2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific frequencies using pyuvdata.',
                                            uv_object2.history))

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep[np.newaxis, :])

    nt.assert_equal(len(freqs_to_keep), uv_object2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in uv_object2.freq_array)
    for f in np.unique(uv_object2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific frequencies using pyuvdata.',
                                            uv_object2.history))

    # check that selecting one frequency works
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep[0])
    nt.assert_equal(1, uv_object2.Nfreqs)
    nt.assert_true(freqs_to_keep[0] in uv_object2.freq_array)
    for f in uv_object2.freq_array:
        nt.assert_true(f in [freqs_to_keep[0]])

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific frequencies using pyuvdata.',
                                            uv_object2.history))

    # check for errors associated with frequencies not included in data
    nt.assert_raises(ValueError, uv_object.select, frequencies=[
                     np.max(uv_object.freq_array) + uv_object.channel_width])

    # check for warnings and errors associated with unevenly spaced or non-contiguous frequencies
    uv_object2 = copy.deepcopy(uv_object)
    uvtest.checkWarnings(uv_object2.select, [], {'frequencies': uv_object2.freq_array[0, [0, 5, 6]]},
                         message='Selected frequencies are not evenly spaced')
    write_file_uvfits = os.path.join(DATA_PATH, 'test/select_test.uvfits')
    write_file_miriad = os.path.join(DATA_PATH, 'test/select_test.uv')
    nt.assert_raises(ValueError, uv_object2.write_uvfits, write_file_uvfits)
    nt.assert_raises(ValueError, uv_object2.write_miriad, write_file_miriad)

    uv_object2 = copy.deepcopy(uv_object)
    uvtest.checkWarnings(uv_object2.select, [], {'frequencies': uv_object2.freq_array[0, [0, 2, 4]]},
                         message='Selected frequencies are not contiguous')
    nt.assert_raises(ValueError, uv_object2.write_uvfits, write_file_uvfits)
    nt.assert_raises(ValueError, uv_object2.write_miriad, write_file_miriad)


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

    nt.assert_equal(len(chans_to_keep), uv_object2.Nfreqs)
    for chan in chans_to_keep:
        nt.assert_true(uv_object.freq_array[0, chan] in uv_object2.freq_array)
    for f in np.unique(uv_object2.freq_array):
        nt.assert_true(f in uv_object.freq_array[0, chans_to_keep])

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific frequencies using pyuvdata.',
                                            uv_object2.history))

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(freq_chans=chans_to_keep[np.newaxis, :])

    nt.assert_equal(len(chans_to_keep), uv_object2.Nfreqs)
    for chan in chans_to_keep:
        nt.assert_true(uv_object.freq_array[0, chan] in uv_object2.freq_array)
    for f in np.unique(uv_object2.freq_array):
        nt.assert_true(f in uv_object.freq_array[0, chans_to_keep])

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific frequencies using pyuvdata.',
                                            uv_object2.history))

    # Test selecting both channels and frequencies
    freqs_to_keep = uv_object.freq_array[0, np.arange(
        20, 30)]  # Overlaps with chans
    all_chans_to_keep = np.arange(12, 30)

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    nt.assert_equal(len(all_chans_to_keep), uv_object2.Nfreqs)
    for chan in all_chans_to_keep:
        nt.assert_true(uv_object.freq_array[0, chan] in uv_object2.freq_array)
    for f in np.unique(uv_object2.freq_array):
        nt.assert_true(f in uv_object.freq_array[0, all_chans_to_keep])


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

    nt.assert_equal(len(pols_to_keep), uv_object2.Npols)
    for p in pols_to_keep:
        nt.assert_true(p in uv_object2.polarization_array)
    for p in np.unique(uv_object2.polarization_array):
        nt.assert_true(p in pols_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific polarizations using pyuvdata.',
                                            uv_object2.history))

    # check that it also works with higher dimension array
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(polarizations=[pols_to_keep])

    nt.assert_equal(len(pols_to_keep), uv_object2.Npols)
    for p in pols_to_keep:
        nt.assert_true(p in uv_object2.polarization_array)
    for p in np.unique(uv_object2.polarization_array):
        nt.assert_true(p in pols_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific polarizations using pyuvdata.',
                                            uv_object2.history))

    # check for errors associated with polarizations not included in data
    nt.assert_raises(ValueError, uv_object2.select, polarizations=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    uvtest.checkWarnings(uv_object.select, [], {'polarizations': uv_object.polarization_array[[0, 1, 3]]},
                         message='Selected polarization values are not evenly spaced')
    write_file_uvfits = os.path.join(DATA_PATH, 'test/select_test.uvfits')
    nt.assert_raises(ValueError, uv_object.write_uvfits, write_file_uvfits)


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

    nt.assert_equal(Nblts_select, uv_object2.Nblts)
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        nt.assert_true(ant in ants_to_keep)

    nt.assert_equal(len(freqs_to_keep), uv_object2.Nfreqs)
    for f in freqs_to_keep:
        nt.assert_true(f in uv_object2.freq_array)
    for f in np.unique(uv_object2.freq_array):
        nt.assert_true(f in freqs_to_keep)

    for t in np.unique(uv_object2.time_array):
        nt.assert_true(t in times_to_keep)

    nt.assert_equal(len(pols_to_keep), uv_object2.Npols)
    for p in pols_to_keep:
        nt.assert_true(p in uv_object2.polarization_array)
    for p in np.unique(uv_object2.polarization_array):
        nt.assert_true(p in pols_to_keep)

    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific baseline-times, antennas, '
                                            'baselines, times, frequencies, '
                                            'polarizations using pyuvdata.',
                                            uv_object2.history))

    # test that a ValueError is raised if the selection eliminates all blts
    nt.assert_raises(ValueError, uv_object.select,
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
    nt.assert_true(uvutils._check_histories(old_history + '  Downselected to '
                                            'specific frequencies using pyuvdata. '
                                            'Combined data along frequency axis '
                                            'using pyuvdata.', uv1.history))

    uv1.history = old_history
    nt.assert_equal(uv1, uv_object)


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
    nt.assert_equal(uv1, uv2)

    # Restore original order
    uvtest.checkWarnings(uv1.read_uvfits, [testfile], message='Telescope EVLA is not')
    uv2.reorder_pols()
    nt.assert_equal(uv1, uv2)


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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific frequencies using pyuvdata. '
                                            'Combined data along frequency axis '
                                            'using pyuvdata.', uv1.history))

    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add frequencies - out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv2 += uv1
    uv2.history = uv_full.history
    nt.assert_equal(uv2, uv_full)

    # Add polarizations
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific polarizations using pyuvdata. '
                                            'Combined data along polarization axis '
                                            'using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add polarizations - out of order
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2 += uv1
    uv2.history = uv_full.history
    nt.assert_equal(uv2, uv_full)

    # Add times
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 += uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times using pyuvdata. '
                                            'Combined data along baseline-time axis '
                                            'using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific baseline-times using pyuvdata. '
                                            'Combined data along baseline-time axis '
                                            'using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific baseline-times using pyuvdata. '
                                            'Combined data along baseline-time axis '
                                            'using pyuvdata. Combined data along '
                                            'baseline-time axis using pyuvdata.',
                                            uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times, polarizations using '
                                            'pyuvdata. Combined data along '
                                            'baseline-time, polarization axis '
                                            'using pyuvdata.', uv1.history))
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
    nt.assert_equal(uv1, uv_ref)

    # Another combo
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2:], freq_chans=np.arange(32, 64))
    uv1 += uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times, frequencies using '
                                            'pyuvdata. Combined data along '
                                            'baseline-time, frequency axis using '
                                            'pyuvdata.', uv1.history))
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
    nt.assert_equal(uv1, uv_ref)

    # Add without inplace
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 = uv1 + uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times using pyuvdata. '
                                            'Combined data along baseline-time '
                                            'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific polarizations using pyuvdata. '
                                            'Combined data along polarization '
                                            'axis using pyuvdata. testing the history.',
                                            uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # test add of autocorr-only and crosscorr-only objects
    uv_full = UVData()
    uv_full.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA'))
    bls = uv_full.get_antpairs()
    autos = [bl for bl in bls if bl[0] == bl[1]]
    cross = sorted(set(bls) - set(autos))
    uv_auto = uv_full.select(bls=autos, inplace=False)
    uv_cross = uv_full.select(bls=cross, inplace=False)
    uv1 = uv_auto + uv_cross
    nt.assert_equal(uv1.Nbls, uv_auto.Nbls + uv_cross.Nbls)
    uv2 = uv_cross + uv_auto
    nt.assert_equal(uv2.Nbls, uv_auto.Nbls + uv_cross.Nbls)


def test_add_drift():
    uv_full = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    uvtest.checkWarnings(uv_full.unphase_to_drift, category=PendingDeprecationWarning,
                         message='The xyz array in ENU_from_ECEF is being '
                                 'interpreted as (Npts, 3)')
    # Add frequencies
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific frequencies using pyuvdata. '
                                            'Combined data along frequency '
                                            'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add polarizations
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific polarizations using pyuvdata. '
                                            'Combined data along polarization '
                                            'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add times
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 += uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times using pyuvdata. '
                                            'Combined data along baseline-time '
                                            'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific baseline-times using pyuvdata. '
                                            'Combined data along baseline-time '
                                            'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times, polarizations using '
                                            'pyuvdata. Combined data along '
                                            'baseline-time, polarization '
                                            'axis using pyuvdata.', uv1.history))
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
    nt.assert_equal(uv1, uv_ref)

    # Another combo
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv_ref = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2:], freq_chans=np.arange(32, 64))
    uv1 += uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times, frequencies using '
                                            'pyuvdata. Combined data along '
                                            'baseline-time, frequency '
                                            'axis using pyuvdata.', uv1.history))
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
    nt.assert_equal(uv1, uv_ref)

    # Add without inplace
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) // 2])
    uv2.select(times=times[len(times) // 2:])
    uv1 = uv1 + uv2
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific times using pyuvdata. '
                                            'Combined data along baseline-time '
                                            'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

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
    nt.assert_true(uvutils._check_histories(uv_full.history + '  Downselected to '
                                            'specific polarizations using pyuvdata. '
                                            'Combined data along polarization '
                                            'axis using pyuvdata. testing the history.',
                                            uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)


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
    nt.assert_raises(ValueError, uv1.__iadd__, np.zeros(5))

    # One phased, one not
    uv2 = copy.deepcopy(uv_full)
    uvtest.checkWarnings(uv2.unphase_to_drift, category=PendingDeprecationWarning,
                         message='The xyz array in ENU_from_ECEF is being '
                                 'interpreted as (Npts, 3)')
    nt.assert_raises(ValueError, uv1.__iadd__, uv2)

    # Different units
    uv2 = copy.deepcopy(uv_full)
    uv2.select(freq_chans=np.arange(32, 64))
    uv2.vis_units = "Jy"
    nt.assert_raises(ValueError, uv1.__iadd__, uv2)

    # Overlapping data
    uv2 = copy.deepcopy(uv_full)
    nt.assert_raises(ValueError, uv1.__iadd__, uv2)

    # Different integration_time
    uv2 = copy.deepcopy(uv_full)
    uv2.select(freq_chans=np.arange(32, 64))
    uv2.integration_time *= 2
    nt.assert_raises(ValueError, uv1.__iadd__, uv2)


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
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal([0], indp[0]))
    # Any of these inputs can also be a tuple of a tuple, so need to be checked twice.
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, pol)))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal([0], indp[0]))

    # Combo with pol as string
    ind1, ind2, indp = uv._key2inds((ant1, ant2, uvutils.polnum2str(pol)))
    nt.assert_true(np.array_equal([0], indp[0]))
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, uvutils.polnum2str(pol))))
    nt.assert_true(np.array_equal([0], indp[0]))

    # Check conjugation
    ind1, ind2, indp = uv._key2inds((ant2, ant1, pol))
    nt.assert_true(np.array_equal(bltind, ind2))
    nt.assert_true(np.array_equal(np.array([]), ind1))
    nt.assert_true(np.array_equal([0], indp[1]))
    # Conjugation with pol as string
    ind1, ind2, indp = uv._key2inds((ant2, ant1, uvutils.polnum2str(pol)))
    nt.assert_true(np.array_equal(bltind, ind2))
    nt.assert_true(np.array_equal(np.array([]), ind1))
    nt.assert_true(np.array_equal([0], indp[1]))
    nt.assert_true(np.array_equal([], indp[0]))

    # Antpair only
    ind1, ind2, indp = uv._key2inds((ant1, ant2))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp[0]))
    ind1, ind2, indp = uv._key2inds(((ant1, ant2)))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp[0]))

    # Baseline number only
    ind1, ind2, indp = uv._key2inds(uv.antnums_to_baseline(ant1, ant2))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp[0]))
    ind1, ind2, indp = uv._key2inds((uv.antnums_to_baseline(ant1, ant2)))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp[0]))

    # Pol number only
    ind1, ind2, indp = uv._key2inds(pol)
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([0]), indp[0]))
    ind1, ind2, indp = uv._key2inds((pol))
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([0]), indp[0]))

    # Pol string only
    ind1, ind2, indp = uv._key2inds('LL')
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([1]), indp[0]))
    ind1, ind2, indp = uv._key2inds(('LL'))
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([1]), indp[0]))

    # Test invalid keys
    nt.assert_raises(KeyError, uv._key2inds, 'I')  # pol str not in data
    nt.assert_raises(KeyError, uv._key2inds, -8)  # pol num not in data
    nt.assert_raises(KeyError, uv._key2inds, 6)  # bl num not in data
    nt.assert_raises(KeyError, uv._key2inds, (1, 1))  # ant pair not in data
    nt.assert_raises(KeyError, uv._key2inds, (1, 1, 'rr'))  # ant pair not in data
    nt.assert_raises(KeyError, uv._key2inds, (0, 1, 'xx'))  # pol not in data

    # Test autos are handled correctly
    uv.ant_2_array[0] = uv.ant_1_array[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant1, pol))
    nt.assert_true(np.array_equal(ind1, [0]))
    nt.assert_true(np.array_equal(ind2, []))


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
    nt.assert_true(np.all(d == dcheck))
    nt.assert_false(d.flags.writeable)
    # Ensure a view was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 5.43
    nt.assert_equal(d[1, 0, 0], uv.data_array[ind1[1], 0, 0, indp[0]])

    # force copy
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), force_copy=True)
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))
    nt.assert_true(d.flags.writeable)
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 4.3
    nt.assert_not_equal(d[1, 0, 0], uv.data_array[ind1[1], 0, 0, indp[0]])

    # ind1 reg, ind2 empty, pol not reg
    ind1 = 10 * np.arange(9)
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))
    nt.assert_false(d.flags.writeable)
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 1.2
    nt.assert_not_equal(d[1, 0, 0], uv.data_array[ind1[1], 0, 0, indp[0]])

    # ind1 not reg, ind2 empty, pol reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))
    nt.assert_false(d.flags.writeable)
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 8.2
    nt.assert_not_equal(d[1, 0, 0], uv.data_array[ind1[1], 0, 0, indp[0]])

    # ind1 not reg, ind2 empty, pol not reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))
    nt.assert_false(d.flags.writeable)
    # Ensure a copy was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 3.4
    nt.assert_not_equal(d[1, 0, 0], uv.data_array[ind1[1], 0, 0, indp[0]])

    # ind1 empty, ind2 reg, pol reg
    # Note conjugation test ensures the result is a copy, not a view.
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1 empty, ind2 reg, pol not reg
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1 empty, ind2 not reg, pol reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1 empty, ind2 not reg, pol not reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1, ind2 not empty, pol reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(uv.data_array[ind1, :, :, :],
                       np.conj(uv.data_array[ind2, :, :, :]), axis=0)
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))

    # ind1, ind2 not empty, pol not reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(uv.data_array[ind1, :, :, :],
                       np.conj(uv.data_array[ind2, :, :, :]), axis=0)
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))

    # test single element
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp], axis=1)
    nt.assert_true(np.all(d == dcheck))

    # test single element
    ind1 = []
    ind2 = [45]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    nt.assert_true(np.all(d == np.conj(dcheck)))

    # Full squeeze
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), squeeze='full')
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))

    # Test invalid squeeze
    nt.assert_raises(ValueError, uv._smart_slicing, uv.data_array, ind1, ind2,
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
    nt.assert_true(np.all(dcheck == d))

    # Check conjugation
    d = uv.get_data(ant2, ant1, pol)
    nt.assert_true(np.all(dcheck == np.conj(d)))

    # Check cross pol conjugation
    d = uv.get_data(ant2, ant1, uv.polarization_array[2])
    d1 = uv.get_data(ant1, ant2, uv.polarization_array[3])
    nt.assert_true(np.all(d == np.conj(d1)))

    # Antpair only
    dcheck = np.squeeze(uv.data_array[bltind, :, :, :])
    d = uv.get_data(ant1, ant2)
    nt.assert_true(np.all(dcheck == d))

    # Pol number only
    dcheck = np.squeeze(uv.data_array[:, :, :, 0])
    d = uv.get_data(pol)
    nt.assert_true(np.all(dcheck == d))


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
    nt.assert_true(np.all(dcheck == d))

    # Check conjugation
    d = uv.get_flags(ant2, ant1, pol)
    nt.assert_true(np.all(dcheck == d))
    nt.assert_equal(d.dtype, np.bool)

    # Antpair only
    dcheck = np.squeeze(uv.flag_array[bltind, :, :, :])
    d = uv.get_flags(ant1, ant2)
    nt.assert_true(np.all(dcheck == d))

    # Pol number only
    dcheck = np.squeeze(uv.flag_array[:, :, :, 0])
    d = uv.get_flags(pol)
    nt.assert_true(np.all(dcheck == d))


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
    nt.assert_true(np.all(dcheck == d))

    # Check conjugation
    d = uv.get_nsamples(ant2, ant1, pol)
    nt.assert_true(np.all(dcheck == d))

    # Antpair only
    dcheck = np.squeeze(uv.nsample_array[bltind, :, :, :])
    d = uv.get_nsamples(ant1, ant2)
    nt.assert_true(np.all(dcheck == d))

    # Pol number only
    dcheck = np.squeeze(uv.nsample_array[:, :, :, 0])
    d = uv.get_nsamples(pol)
    nt.assert_true(np.all(dcheck == d))


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
    nt.assert_true(inds.dtype == np.int)

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
    nt.assert_raises(ValueError, uv.antpair2ind, 1)
    nt.assert_raises(ValueError, uv.antpair2ind, 'bar', 'foo')
    nt.assert_raises(ValueError, uv.antpair2ind, 0, 1, 'foo')


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
    nt.assert_true(np.all(dcheck == d))

    # Check conjugation
    d = uv.get_times(ant2, ant1, pol)
    nt.assert_true(np.all(dcheck == d))

    # Antpair only
    d = uv.get_times(ant1, ant2)
    nt.assert_true(np.all(dcheck == d))

    # Pol number only
    d = uv.get_times(pol)
    nt.assert_true(np.all(d == uv.time_array))


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
        nt.assert_true(np.all(dcheck == d))
    nt.assert_equal(len(bls), len(uv.get_baseline_nums()))
    nt.assert_equal(len(pols), uv.Npols)


def test_get_ants():
    # Test function to get unique antennas in data
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    ants = uv.get_ants()
    for ant in ants:
        nt.assert_true((ant in uv.ant_1_array) or (ant in uv.ant_2_array))
    for ant in uv.ant_1_array:
        nt.assert_true(ant in ants)
    for ant in uv.ant_2_array:
        nt.assert_true(ant in ants)


def test_get_ENU_antpos():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA"))
    # no center, no pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=False, pick_data_ants=False)
    nt.assert_equal(len(ants), 113)
    nt.assert_almost_equal(antpos[0, 0], 19.340211050751535)
    nt.assert_equal(ants[0], 0)
    # center
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=False)
    nt.assert_almost_equal(antpos[0, 0], 22.472442651767714)
    # pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    nt.assert_equal(ants[0], 9)
    nt.assert_almost_equal(antpos[0, 0], -0.0026981323386223721)


def test_get_pols():
    # Test function to get unique polarizations in string format
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    pols = uv.get_pols()
    pols_data = ['rr', 'll', 'lr', 'rl']
    nt.assert_count_equal(pols, pols_data)


def test_get_feedpols():
    # Test function to get unique antenna feed polarizations in data. String format.
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    pols = uv.get_feedpols()
    pols_data = ['r', 'l']
    nt.assert_count_equal(pols, pols_data)

    # Test break when pseudo-Stokes visibilities are present
    uv.polarization_array[0] = 1  # pseudo-Stokes I
    nt.assert_raises(ValueError, uv.get_feedpols)


def test_parse_ants():
    # Test function to get correct antenna pairs and polarizations
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile], message='Telescope EVLA is not')

    # All baselines
    ant_str = 'all'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    nt.assert_is_instance(ant_pairs_nums, type(None))
    nt.assert_is_instance(polarizations, type(None))

    # Auto correlations
    ant_str = 'auto'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    nt.assert_count_equal(ant_pairs_nums, [])
    nt.assert_is_instance(polarizations, type(None))

    # Cross correlations
    ant_str = 'cross'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    nt.assert_count_equal(uv.get_antpairs(), ant_pairs_nums)
    nt.assert_is_instance(polarizations, type(None))

    # pseudo-Stokes params
    ant_str = 'pI,pq,pU,pv'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    pols_expected = [4, 3, 2, 1]
    nt.assert_is_instance(ant_pairs_nums, type(None))
    nt.assert_count_equal(polarizations, pols_expected)

    # Unparsible string
    ant_str = 'none'
    nt.assert_raises(ValueError, uv.parse_ants, ant_str)

    # Single antenna number
    ant_str = '0'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8),
                          (0, 11), (0, 14), (0, 18), (0, 19), (0, 20),
                          (0, 21), (0, 22), (0, 23), (0, 24), (0, 26),
                          (0, 27)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Single antenna number not in the data
    ant_str = '10'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Antenna')
    nt.assert_is_instance(ant_pairs_nums, type(None))
    nt.assert_is_instance(polarizations, type(None))

    # Single antenna number with polarization, both not in the data
    ant_str = '10x'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=2,
                                                         message=['Warning: Antenna', 'Warning: Polarization'])
    nt.assert_is_instance(ant_pairs_nums, type(None))
    nt.assert_is_instance(polarizations, type(None))

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
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Single baseline
    ant_str = '1_3'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Single baseline with polarization
    ant_str = '1l_3r'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    pols_expected = [-4]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Single baseline with single polarization in first entry
    ant_str = '1l_3,2x_3'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Polarization')
    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -4]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Single baseline with single polarization in last entry
    ant_str = '1_3l,2_3x'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Polarization')
    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -3]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Multiple baselines as list
    ant_str = '1_2,1_3,1_11'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Multiples baselines with polarizations as list
    ant_str = '1r_2l,1l_3l,1r_11r'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    pols_expected = [-1, -2, -3]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Specific baselines with parenthesis
    ant_str = '(1,3)_11'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11), (3, 11)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Specific baselines with parenthesis
    ant_str = '1_(3,11)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 11)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Antenna numbers with polarizations
    ant_str = '(1l,2r)_(3l,6r)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [-1, -2, -3, -4]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Antenna numbers with - for avoidance
    ant_str = '1_(-3,11)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Remove specific antenna number
    ant_str = '1,-3'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(0, 1), (1, 2), (1, 6), (1, 7), (1, 8), (1, 11),
                          (1, 14), (1, 18), (1, 19), (1, 20), (1, 21),
                          (1, 22), (1, 23), (1, 24), (1, 26), (1, 27)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Remove specific baseline (same expected antenna pairs as above example)
    ant_str = '1,-1_3'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Antenna numbers with polarizations and - for avoidance
    ant_str = '1l_(-3r,11l)'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    pols_expected = [-2]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Antenna numbers and pseudo-Stokes parameters
    ant_str = '(1l,2r)_(3l,6r),pI,pq'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [2, 1, -1, -2, -3, -4]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Multiple baselines with multiple polarizations, one pol to be removed
    ant_str = '1l_2,1l_3,-1l_3r'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Multiple baselines with multiple polarizations, one pol (not in data) to be removed
    ant_str = '1l_2,1l_3,-1x_3y'
    ant_pairs_nums, polarizations = uvtest.checkWarnings(uv.parse_ants,
                                                         [ant_str], {},
                                                         nwarnings=1,
                                                         message='Warning: Polarization')
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2, -4]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Test print toggle on single baseline with polarization
    ant_str = '1l_2l'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str, print_toggle=True)
    ant_pairs_expected = [(1, 2)]
    pols_expected = [-2]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_count_equal(polarizations, pols_expected)

    # Test ant_str='auto' on file with auto correlations
    uv = UVData()
    testfile = os.path.join(DATA_PATH, 'hera_testfile')
    uvtest.checkWarnings(uv.read_miriad, [testfile], nwarnings=1,
                         message='Altitude is not')

    ant_str = 'auto'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(9, 9), (10, 10), (20, 20)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Test cross correlation extraction on data with auto + cross
    ant_str = 'cross'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(9, 10), (9, 20), (10, 20)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Remove only polarization of single baseline
    ant_str = 'all,-9x_10x'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(9, 9), (9, 20), (10, 10), (10, 20), (20, 20)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))

    # Test appending all to beginning of strings that start with -
    ant_str = '-9'
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(10, 10), (10, 20), (20, 20)]
    nt.assert_count_equal(ant_pairs_nums, ant_pairs_expected)
    nt.assert_is_instance(polarizations, type(None))


def test_select_with_ant_str():
    # Test select function with ant_str argument
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile], message='Telescope EVLA is not')
    inplace = False

    # Check error thrown if ant_str passed with antenna_nums,
    # antenna_names, ant_pairs_nums, or polarizations
    nt.assert_raises(ValueError, uv.select,
                     ant_str='',
                     antenna_nums=[],
                     inplace=inplace)
    nt.assert_raises(ValueError, uv.select,
                     ant_str='',
                     antenna_nums=[],
                     inplace=inplace)
    nt.assert_raises(ValueError, uv.select,
                     ant_str='',
                     antenna_nums=[],
                     inplace=inplace)
    nt.assert_raises(ValueError, uv.select,
                     ant_str='',
                     antenna_nums=[],
                     inplace=inplace)

    # All baselines
    ant_str = 'all'
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), uv.get_antpairs())
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Auto correlations
    ant_str = 'auto'
    nt.assert_raises(ValueError, uv.select, ant_str=ant_str, inplace=inplace)
    # No auto correlations in this data

    # Cross correlations
    ant_str = 'cross'
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), uv.get_antpairs())
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())
    # All baselines in data are cross correlations

    # pseudo-Stokes params
    ant_str = 'pI,pq,pU,pv'
    nt.assert_raises(ValueError, uv.select, ant_str=ant_str, inplace=inplace)

    # Unparsible string
    ant_str = 'none'
    nt.assert_raises(ValueError, uv.select, ant_str=ant_str, inplace=inplace)

    # Single antenna number
    ant_str = '0'
    ant_pairs = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (0, 11),
                 (0, 14), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22),
                 (0, 23), (0, 24), (0, 26), (0, 27)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

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
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Single baseline
    ant_str = '1_3'
    ant_pairs = [(1, 3)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Single baseline with polarization
    ant_str = '1l_3r'
    ant_pairs = [(1, 3)]
    pols = ['lr']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), pols)

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
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), pols)

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
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), pols)

    # Multiple baselines as list
    ant_str = '1_2,1_3,1_10'
    # Antenna number 10 not in data
    uv2 = uvtest.checkWarnings(uv.select, [],
                               {'ant_str': ant_str, 'inplace': inplace},
                               nwarnings=1, message='Warning: Antenna')
    ant_pairs = [(1, 2), (1, 3)]
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Multiples baselines with polarizations as list
    ant_str = '1r_2l,1l_3l,1r_11r'
    ant_pairs = [(1, 2), (1, 3), (1, 11)]
    pols = ['rr', 'll', 'rl']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), pols)

    # Specific baselines with parenthesis
    ant_str = '(1,3)_11'
    ant_pairs = [(1, 11), (3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Specific baselines with parenthesis
    ant_str = '1_(3,11)'
    ant_pairs = [(1, 3), (1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Antenna numbers with polarizations
    ant_str = '(1l,2r)_(3l,6r)'
    ant_pairs = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols = ['rr', 'll', 'rl', 'lr']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), pols)

    # Antenna numbers with - for avoidance
    ant_str = '1_(-3,11)'
    ant_pairs = [(1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    ant_str = '(-1,3)_11'
    ant_pairs = [(3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Remove specific antenna number
    ant_str = '1,-3'
    ant_pairs = [(0, 1), (1, 2), (1, 6), (1, 7), (1, 8), (1, 11),
                 (1, 14), (1, 18), (1, 19), (1, 20), (1, 21),
                 (1, 22), (1, 23), (1, 24), (1, 26), (1, 27)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Remove specific baseline
    ant_str = '1,-1_3'
    ant_pairs = [(0, 1), (1, 2), (1, 6), (1, 7), (1, 8), (1, 11),
                 (1, 14), (1, 18), (1, 19), (1, 20), (1, 21),
                 (1, 22), (1, 23), (1, 24), (1, 26), (1, 27)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Antenna numbers with polarizations and - for avoidance
    ant_str = '1l_(-3r,11l)'
    ant_pairs = [(1, 11)]
    pols = ['ll']
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), pols)

    # Test pseudo-Stokes params with select
    ant_str = 'pi,pQ'
    pols = ['pQ', 'pI']
    uv.polarization_array = np.array([4, 3, 2, 1])
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), uv.get_antpairs())
    nt.assert_count_equal(uv2.get_pols(), pols)

    # Test ant_str = 'auto' on file with auto correlations
    uv = UVData()
    testfile = os.path.join(DATA_PATH, 'hera_testfile')
    uvtest.checkWarnings(uv.read_miriad, [testfile], nwarnings=1,
                         message='Altitude is not')

    ant_str = 'auto'
    ant_pairs = [(9, 9), (10, 10), (20, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Test cross correlation extraction on data with auto + cross
    ant_str = 'cross'
    ant_pairs = [(9, 10), (9, 20), (10, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Remove only polarization of single baseline
    ant_str = 'all,-9x_10x'
    ant_pairs = [(9, 9), (9, 20), (10, 10), (10, 20), (20, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())

    # Test appending all to beginning of strings that start with -
    ant_str = '-9'
    ant_pairs = [(10, 10), (10, 20), (20, 20)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    nt.assert_count_equal(uv2.get_antpairs(), ant_pairs)
    nt.assert_count_equal(uv2.get_pols(), uv.get_pols())


def test_set_uvws_from_antenna_pos():
    # Test set_uvws_from_antenna_positions function with phased data
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, '1133866760.uvfits')
    uv_object.read_uvfits(testfile)
    orig_uvw_array = np.copy(uv_object.uvw_array)
    nt.assert_raises(ValueError, uv_object.set_uvws_from_antenna_positions)
    uvtest.checkWarnings(
        nt.assert_raises,
        [ValueError, uv_object.set_uvws_from_antenna_positions, True, 'xyz'],
        message='Warning: Data will be unphased'
    )
    uvtest.checkWarnings(
        nt.assert_raises,
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
    nt.assert_almost_equal(max_diff, 0., 2)
