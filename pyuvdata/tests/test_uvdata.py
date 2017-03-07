"""Tests for uvdata object."""
import nose.tools as nt
import os
import numpy as np
import copy
import ephem
from pyuvdata import UVData
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


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
                                 '_gst0', '_rdate', '_earth_omega', '_dut1',
                                 '_timesys', '_uvplane_reference_time',
                                 '_phase_center_ra', '_phase_center_dec',
                                 '_phase_center_epoch',
                                 '_zenith_ra', '_zenith_dec']

        self.extra_properties = ['extra_keywords', 'antenna_positions', 'gst0',
                                 'rdate', 'earth_omega', 'dut1', 'timesys',
                                 'uvplane_reference_time',
                                 'phase_center_ra', 'phase_center_dec',
                                 'phase_center_epoch',
                                 'zenith_ra', 'zenith_dec']

        self.other_properties = ['telescope_location_lat_lon_alt',
                                 'telescope_location_lat_lon_alt_degrees',
                                 'phase_center_ra_degrees', 'phase_center_dec_degrees',
                                 'zenith_ra_degrees', 'zenith_dec_degrees']

        self.uv_object = UVData()

    def teardown(self):
        """Test teardown: delete object."""
        del(self.uv_object)

    def test_parameter_iter(self):
        "Test expected parameters."
        all = []
        for prop in self.uv_object:
            all.append(prop)
        for a in self.required_parameters + self.extra_parameters:
            nt.assert_true(a in all, msg='expected attribute ' + a +
                           ' not returned in object iterator')

    def test_required_parameter_iter(self):
        "Test expected required parameters."
        required = []
        for prop in self.uv_object.required():
            required.append(prop)
        for a in self.required_parameters:
            nt.assert_true(a in required, msg='expected attribute ' + a +
                           ' not returned in required iterator')

    def test_extra_parameter_iter(self):
        "Test expected optional parameters."
        extra = []
        for prop in self.uv_object.extra():
            extra.append(prop)
        for a in self.extra_parameters:
            nt.assert_true(a in extra, msg='expected attribute ' + a +
                           ' not returned in extra iterator')

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
        prop_dict = dict(zip(self.required_properties + self.extra_properties,
                             self.required_parameters + self.extra_parameters))
        for k, v in prop_dict.iteritems():
            rand_num = np.random.rand()
            setattr(self.uv_object, k, rand_num)
            this_param = getattr(self.uv_object, v)
            try:
                nt.assert_equal(rand_num, this_param.value)
            except:
                print('setting {prop_name} to a random number failed'.format(prop_name=k))
                raise(AssertionError)


class TestUVDataBasicMethods(object):
    def setUp(self):
        """Setup for tests of basic methods."""
        self.uv_object = UVData()
        self.testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
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

    def test_nants_data_telescope(self):
        self.uv_object.Nants_data = self.uv_object.Nants_telescope - 1
        nt.assert_true(self.uv_object.check)
        self.uv_object.Nants_data = self.uv_object.Nants_telescope + 1
        nt.assert_raises(ValueError, self.uv_object.check)


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
        nt.assert_raises(StandardError, self.uv_object2.baseline_to_antnums, 67585)

    def test_antnums_to_baselines(self):
        """Test antums to baseline conversion for 256 & larger conventions."""
        nt.assert_equal(self.uv_object.antnums_to_baseline(0, 0), 67585)
        nt.assert_equal(self.uv_object.antnums_to_baseline(257, 256), 592130)
        # Check attempt256
        nt.assert_equal(self.uv_object.antnums_to_baseline(0, 0, attempt256=True), 257)
        nt.assert_equal(self.uv_object.antnums_to_baseline(257, 256), 592130)
        nt.assert_true(uvtest.checkWarnings(self.uv_object.antnums_to_baseline,
                                            [257, 256], {'attempt256': True},
                                            message='found > 256 antennas'))
        nt.assert_raises(StandardError, self.uv_object2.antnums_to_baseline, 0, 0)


def test_known_telescopes():
    """Test known_telescopes method returns expected results."""
    uv_object = UVData()
    known_telescopes = ['PAPER', 'HERA', 'MWA']
    nt.assert_equal(known_telescopes.sort(),
                    uv_object.known_telescopes().sort())


def test_phase_unphaseHERA():
    """
    Read in drift data, phase to an RA/DEC, unphase and check for object equality.
    """
    testfile = os.path.join(DATA_PATH, 'hera_testfile')
    UV_raw = UVData()
    # Note the RA/DEC values in the raw file were calculated from the lat/long
    # in the file, which don't agree with our known_telescopes.
    # So for this test we use the lat/lon in the file.
    status = uvtest.checkWarnings(UV_raw.read_miriad, [testfile],
                                  {'correct_lat_lon': False}, known_warning='miriad')

    UV_phase = UVData()
    status = uvtest.checkWarnings(UV_phase.read_miriad, [testfile],
                                  {'correct_lat_lon': False}, known_warning='miriad')
    UV_phase.phase(0., 0., ephem.J2000)
    UV_phase.unphase_to_drift()

    nt.assert_equal(UV_raw, UV_phase)
    del(UV_phase)
    del(UV_raw)


def test_select_blts():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    blt_inds = np.random.choice(uv_object.Nblts, uv_object.Nblts / 10, replace=False)

    uv_object.select_blts(blt_inds)
    nt.assert_equal(len(blt_inds), uv_object.Nblts)

    uv_object2 = UVData()
    uvtest.checkWarnings(uv_object2.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    uv_object2.select(blt_inds=blt_inds)
    nt.assert_equal(uv_object, uv_object2)

    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    nt.assert_raises(ValueError, uv_object.select_blts, np.arange(-10, -5))
    nt.assert_raises(ValueError, uv_object.select_blts, np.arange(uv_object.Nblts + 1, uv_object.Nblts + 10))


def test_select_antennas():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uvtest.checkWarnings(uv_object.read_miriad, [testfile],
                         known_warning='miriad')
    unique_ants = np.unique(uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist())
    ants_to_keep = np.random.choice(unique_ants, len(unique_ants) / 2, replace=False)

    uv_object.select_antennas(ants_to_keep)

    nt.assert_equal(len(ants_to_keep), uv_object.Nants_data)
    for ant in ants_to_keep:
        nt.assert_true(ant in uv_object.ant_1_array or ant in uv_object.ant_2_array)
    for ant in np.unique(uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist()):
        nt.assert_true(ant in ants_to_keep)

    uv_object2 = UVData()
    uvtest.checkWarnings(uv_object2.read_miriad, [testfile],
                         known_warning='miriad')
    uv_object2.select(antennas=ants_to_keep)
    nt.assert_equal(uv_object, uv_object2)

    uvtest.checkWarnings(uv_object.read_miriad, [testfile],
                         known_warning='miriad')
    nt.assert_raises(ValueError, uv_object.select_antennas, np.max(unique_ants) + np.arange(1, 3))


def test_select_times():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    unique_times = np.unique(uv_object.time_array)
    times_to_keep = np.random.choice(unique_times, uv_object.Ntimes / 2, replace=False)

    uv_object.select_times(times_to_keep)

    nt.assert_equal(len(times_to_keep), uv_object.Ntimes)
    for t in times_to_keep:
        nt.assert_true(t in uv_object.time_array)
    for t in np.unique(uv_object.time_array):
        nt.assert_true(t in times_to_keep)

    uv_object2 = UVData()
    uvtest.checkWarnings(uv_object2.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    uv_object2.select(times=times_to_keep)
    nt.assert_equal(uv_object, uv_object2)

    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    nt.assert_raises(ValueError, uv_object.select_times, [np.min(unique_times) - uv_object.integration_time])


def test_select_frequencies():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    freqs_to_keep = np.random.choice(uv_object.freq_array[0, :], uv_object.Nfreqs / 10, replace=False)
    uv_object.select_frequencies(freqs_to_keep)

    nt.assert_equal(len(freqs_to_keep), uv_object.Nfreqs)
    for t in freqs_to_keep:
        nt.assert_true(t in uv_object.freq_array)
    for t in np.unique(uv_object.freq_array):
        nt.assert_true(t in freqs_to_keep)

    uv_object2 = UVData()
    uvtest.checkWarnings(uv_object2.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    uv_object2.select(frequencies=freqs_to_keep)
    nt.assert_equal(uv_object, uv_object2)

    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    nt.assert_raises(ValueError, uv_object.select_frequencies, [np.max(uv_object.freq_array) + uv_object.channel_width])


def test_select_polarizations():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    pols_to_keep = np.random.choice(uv_object.polarization_array, uv_object.Npols / 2, replace=False)
    pols_dropped = [p for p in uv_object.polarization_array if p not in pols_to_keep]
    uv_object.select_polarizations(pols_to_keep)

    nt.assert_equal(len(pols_to_keep), uv_object.Npols)
    for t in pols_to_keep:
        nt.assert_true(t in uv_object.polarization_array)
    for t in np.unique(uv_object.polarization_array):
        nt.assert_true(t in pols_to_keep)

    uv_object2 = UVData()
    uvtest.checkWarnings(uv_object2.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    uv_object2.select(polarizations=pols_to_keep)
    nt.assert_equal(uv_object, uv_object2)

    nt.assert_raises(ValueError, uv_object.select_polarizations, pols_dropped)


def test_select():
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    # have to be careful with selecting on blt_inds before antennas or times
    # because removing blt_inds can remove entire times or antennas.
    # With this test file, times are particularly sensitive. So test in 2 stages,
    # one with everything but blts, and one with just blts & antennas for coverage.
    unique_ants = np.unique(uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist())
    ants_to_keep = np.random.choice(unique_ants, len(unique_ants) - 1, replace=False)

    freqs_to_keep = np.random.choice(uv_object.freq_array[0, :], uv_object.Nfreqs / 10, replace=False)

    unique_times = np.unique(uv_object.time_array)
    times_to_keep = np.random.choice(unique_times, uv_object.Ntimes / 2, replace=False)

    pols_to_keep = np.random.choice(uv_object.polarization_array, uv_object.Npols / 2, replace=False)

    uv_object.select(antennas=ants_to_keep, frequencies=freqs_to_keep,
                     times=times_to_keep, polarizations=pols_to_keep)

    nt.assert_equal(len(ants_to_keep), uv_object.Nants_data)
    for ant in ants_to_keep:
        nt.assert_true(ant in uv_object.ant_1_array or ant in uv_object.ant_2_array)
    for ant in np.unique(uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist()):
        nt.assert_true(ant in ants_to_keep)

    nt.assert_equal(len(freqs_to_keep), uv_object.Nfreqs)
    for t in freqs_to_keep:
        nt.assert_true(t in uv_object.freq_array)
    for t in np.unique(uv_object.freq_array):
        nt.assert_true(t in freqs_to_keep)

    nt.assert_equal(old_history + '  Downselected to specific antennas, '
                    'frequencies, times, polarizations using pyuvdata.',
                    uv_object.history)

    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history

    blt_inds = np.random.choice(uv_object.Nblts, int(uv_object.Nblts * 0.9), replace=False)

    uv_object.select(blt_inds=blt_inds, antennas=ants_to_keep)

    nt.assert_equal(old_history + '  Downselected to specific baseline-times, '
                    'antennas using pyuvdata.',
                    uv_object.history)
