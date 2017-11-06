"""Tests for uvdata object."""
import nose.tools as nt
import os
import numpy as np
import copy
import ephem
from pyuvdata import UVData
import pyuvdata.utils as uvutils
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
                                 '_x_orientation', '_antenna_diameters',
                                 '_gst0', '_rdate', '_earth_omega', '_dut1',
                                 '_timesys', '_uvplane_reference_time',
                                 '_phase_center_ra', '_phase_center_dec',
                                 '_phase_center_epoch',
                                 '_zenith_ra', '_zenith_dec']

        self.extra_properties = ['extra_keywords', 'antenna_positions',
                                 'x_orientation', 'antenna_diameters', 'gst0',
                                 'rdate', 'earth_omega', 'dut1', 'timesys',
                                 'uvplane_reference_time',
                                 'phase_center_ra', 'phase_center_dec',
                                 'phase_center_epoch',
                                 'zenith_ra', 'zenith_dec']

        self.other_properties = ['telescope_location_lat_lon_alt',
                                 'telescope_location_lat_lon_alt_degrees',
                                 'phase_center_ra_degrees', 'phase_center_dec_degrees',
                                 'zenith_ra_degrees', 'zenith_dec_degrees',
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
        prop_dict = dict(zip(self.required_properties + self.extra_properties,
                             self.required_parameters + self.extra_parameters))
        for k, v in prop_dict.iteritems():
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
        self.uv_object.read_fhd([testdir + '1061316296_flags.sav',
                                 testdir + '1061316296_vis_XX.sav',
                                 testdir + '1061316296_params.sav',
                                 testdir + '1061316296_settings.txt'])
        self.uv_object.select(blt_inds=np.where(self.uv_object.ant_1_array ==
                                                self.uv_object.ant_2_array)[0])
        nt.assert_true(self.uv_object.check())

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
            StandardError, self.uv_object2.baseline_to_antnums, 67585)

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
            StandardError, self.uv_object2.antnums_to_baseline, 0, 0)


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
    UV_phase.phase(0., 0., ephem.J2000)
    UV_phase.unphase_to_drift()

    nt.assert_equal(UV_raw, UV_phase)

    # check errors when trying to unphase drift or unknown data
    nt.assert_raises(ValueError, UV_raw.unphase_to_drift)
    UV_raw.set_unknown_phase_type()
    nt.assert_raises(ValueError, UV_raw.unphase_to_drift)

    # check errors when trying to phase phased or unknown data
    UV_phase.phase(0., 0., ephem.J2000)
    nt.assert_raises(ValueError, UV_phase.phase, 0., 0., ephem.J2000)
    nt.assert_raises(ValueError, UV_phase.phase_to_time,
                     UV_phase.time_array[0])

    UV_phase.set_unknown_phase_type()
    nt.assert_raises(ValueError, UV_phase.phase, 0., 0., ephem.J2000)
    nt.assert_raises(ValueError, UV_phase.phase_to_time,
                     UV_phase.time_array[0])

    del(UV_phase)
    del(UV_raw)


def test_set_phase_unknown():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [
                         testfile], message='Telescope EVLA is not')

    uv_object.set_unknown_phase_type()
    nt.assert_equal(uv_object.phase_type, 'unknown')
    nt.assert_false(uv_object._zenith_ra.required)
    nt.assert_false(uv_object._zenith_dec.required)
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
    nt.assert_false(uvutils.check_histories(old_history, uv_object2.history))

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
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

    # check for errors associated with antennas not included in data, bad names or providing numbers and names
    nt.assert_raises(ValueError, uv_object.select,
                     antenna_nums=np.max(unique_ants) + np.arange(1, 3))
    nt.assert_raises(ValueError, uv_object.select, antenna_names='test1')
    nt.assert_raises(ValueError, uv_object.select,
                     antenna_nums=ants_to_keep, antenna_names=ant_names)


def test_select_ant_pairs():
    uv_object = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_object.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    old_history = uv_object.history
    first_ants = [6, 2, 7, 2, 21, 27, 8]
    second_ants = [0, 20, 8, 1, 2, 3, 22]
    new_unique_ants = np.unique(first_ants + second_ants)
    ant_pairs_to_keep = zip(first_ants, second_ants)
    sorted_pairs_to_keep = [tuple(sorted(p)) for p in ant_pairs_to_keep]

    sorted_pairs_object = [tuple(sorted(p)) for p in zip(
        uv_object.ant_1_array, uv_object.ant_2_array)]

    blts_select = [tuple(sorted((a1, a2))) in sorted_pairs_to_keep for (a1, a2) in
                   zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(ant_pairs_nums=ant_pairs_to_keep)
    sorted_pairs_object2 = [tuple(sorted(p)) for p in zip(
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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific antenna pairs using pyuvdata.',
                                           uv_object2.history))

    # check that you can use numpy integers with out errors:
    first_ants = map(np.int32, [6, 2, 7, 2, 21, 27, 8])
    second_ants = map(np.int32, [0, 20, 8, 1, 2, 3, 22])
    ant_pairs_to_keep = zip(first_ants, second_ants)

    uv_object2 = uv_object.select(ant_pairs_nums=ant_pairs_to_keep, inplace=False)
    sorted_pairs_object2 = [tuple(sorted(p)) for p in zip(
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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific antenna pairs using pyuvdata.',
                                           uv_object2.history))

    # check that you can specify a single pair without errors
    uv_object2.select(ant_pairs_nums=(0, 6))
    sorted_pairs_object2 = [tuple(sorted(p)) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]
    nt.assert_equal(list(set(sorted_pairs_object2)), [(0, 6)])

    # check for errors associated with antenna pairs not included in data and bad inputs
    nt.assert_raises(ValueError, uv_object.select,
                     ant_pairs_nums=zip(first_ants, second_ants) + [0, 6])
    nt.assert_raises(ValueError, uv_object.select,
                     ant_pairs_nums=[(uv_object.antenna_names[0], uv_object.antenna_names[1])])
    nt.assert_raises(ValueError, uv_object.select, ant_pairs_nums=(5, 1))
    nt.assert_raises(ValueError, uv_object.select, ant_pairs_nums=(0, 5))
    nt.assert_raises(ValueError, uv_object.select, ant_pairs_nums=(27, 27))


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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific times using pyuvdata.',
                                           uv_object2.history))

    # check for errors associated with times not included in data
    nt.assert_raises(ValueError, uv_object.select, times=[
                     np.min(unique_times) - uv_object.integration_time])


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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific frequencies using pyuvdata.',
                                           uv_object2.history))

    # check that selecting one frequency works
    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(frequencies=freqs_to_keep[0])
    nt.assert_equal(1, uv_object2.Nfreqs)
    nt.assert_true(freqs_to_keep[0] in uv_object2.freq_array)
    for f in uv_object2.freq_array:
        nt.assert_true(f in [freqs_to_keep[0]])

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
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
    sorted_pairs_to_keep = [tuple(sorted(p)) for p in ant_pairs_to_keep]

    freqs_to_keep = uv_object.freq_array[0, np.arange(31, 39)]

    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 2, 6, 8, 10, 13, 14]]

    pols_to_keep = [-1, -3]

    # Independently count blts that should be selected
    blts_blt_select = [i in blt_inds for i in np.arange(uv_object.Nblts)]
    blts_ant_select = [(a1 in ants_to_keep) & (a2 in ants_to_keep) for (a1, a2) in
                       zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    blts_pair_select = [tuple(sorted((a1, a2))) in sorted_pairs_to_keep for (a1, a2) in
                        zip(uv_object.ant_1_array, uv_object.ant_2_array)]
    blts_time_select = [t in times_to_keep for t in uv_object.time_array]
    Nblts_select = np.sum([bi & ai & pi & ti for (bi, ai, pi, ti) in
                           zip(blts_blt_select, blts_ant_select, blts_pair_select,
                               blts_time_select)])

    uv_object2 = copy.deepcopy(uv_object)
    uv_object2.select(blt_inds=blt_inds, antenna_nums=ants_to_keep,
                      ant_pairs_nums=ant_pairs_to_keep, frequencies=freqs_to_keep,
                      times=times_to_keep, polarizations=pols_to_keep)

    nt.assert_equal(Nblts_select, uv_object2.Nblts)
    for ant in np.unique(uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()):
        nt.assert_true(ant in ants_to_keep)

    sorted_pairs_object2 = [tuple(sorted(p)) for p in zip(
        uv_object2.ant_1_array, uv_object2.ant_2_array)]
    for pair in sorted_pairs_object2:
        nt.assert_true(pair in sorted_pairs_to_keep)

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

    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific baseline-times, antennas, '
                                           'antenna pairs, times, frequencies, '
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
    nt.assert_true(uvutils.check_histories(old_history + '  Downselected to '
                                           'specific frequencies using pyuvdata. '
                                           'Combined data along frequency axis '
                                           'using pyuvdata.', uv1.history))

    uv1.history = old_history
    nt.assert_equal(uv1, uv_object)


def test_reorder_pols():
    # Test function to fix polarization order
    uv1 = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(
        uv1.read_uvfits, [testfile], message='Telescope EVLA is not')
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
    uvtest.checkWarnings(
        uv1.read_uvfits, [testfile], message='Telescope EVLA is not')
    uv2.reorder_pols()
    nt.assert_equal(uv1, uv2)


def test_add():
    uv_full = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    # Add frequencies
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific frequencies using pyuvdata. '
                                           'Combined data along frequency axis '
                                           'using pyuvdata.', uv1.history))

    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add polarizations
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific polarizations using pyuvdata. '
                                           'Combined data along polarization axis '
                                           'using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add times
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) / 2])
    uv2.select(times=times[len(times) / 2:])
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific times using pyuvdata. '
                                           'Combined data along baseline-time axis '
                                           'using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add baselines
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    ant_list = range(15)  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
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
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
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
    uv1.select(times=times[0:len(times) / 2],
               polarizations=uv1.polarization_array[0:2])
    uv2.select(times=times[len(times) / 2:],
               polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific times, polarizations using '
                                           'pyuvdata. Combined data along '
                                           'baseline-time, polarization axis '
                                           'using pyuvdata.', uv1.history))
    blt_ind1 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) / 2]])
    blt_ind2 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) / 2:]])
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
    uv1.select(times=times[0:len(times) / 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) / 2:], freq_chans=np.arange(32, 64))
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific times, frequencies using '
                                           'pyuvdata. Combined data along '
                                           'baseline-time, frequency axis using '
                                           'pyuvdata.', uv1.history))
    blt_ind1 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) / 2]])
    blt_ind2 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) / 2:]])
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
    uv1.select(times=times[0:len(times) / 2])
    uv2.select(times=times[len(times) / 2:])
    uv1 = uv1 + uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
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
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific polarizations using pyuvdata. '
                                           'Combined data along polarization '
                                           'axis using pyuvdata. testing the history.',
                                           uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)


def test_add_drift():
    uv_full = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv_full.read_uvfits, [testfile],
                         message='Telescope EVLA is not')

    uv_full.unphase_to_drift()
    # Add frequencies
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
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
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific polarizations using pyuvdata. '
                                           'Combined data along polarization '
                                           'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add times
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0:len(times) / 2])
    uv2.select(times=times[len(times) / 2:])
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific times using pyuvdata. '
                                           'Combined data along baseline-time '
                                           'axis using pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)

    # Add baselines
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    ant_list = range(15)  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
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
    uv1.select(times=times[0:len(times) / 2],
               polarizations=uv1.polarization_array[0:2])
    uv2.select(times=times[len(times) / 2:],
               polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific times, polarizations using '
                                           'pyuvdata. Combined data along '
                                           'baseline-time, polarization '
                                           'axis using pyuvdata.', uv1.history))
    blt_ind1 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) / 2]])
    blt_ind2 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) / 2:]])
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
    uv1.select(times=times[0:len(times) / 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) / 2:], freq_chans=np.arange(32, 64))
    uv1 += uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific times, frequencies using '
                                           'pyuvdata. Combined data along '
                                           'baseline-time, frequency '
                                           'axis using pyuvdata.', uv1.history))
    blt_ind1 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[0:len(times) / 2]])
    blt_ind2 = np.array([ind for ind in xrange(uv_full.Nblts) if
                         uv_full.time_array[ind] in times[len(times) / 2:]])
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
    uv1.select(times=times[0:len(times) / 2])
    uv2.select(times=times[len(times) / 2:])
    uv1 = uv1 + uv2
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
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
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
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
    nt.assert_raises(ValueError, uv1.__iadd__, np.zeros(5))

    # One phased, one not
    uv2 = copy.deepcopy(uv_full)
    uv2.unphase_to_drift()
    nt.assert_raises(ValueError, uv1.__iadd__, uv2)

    # Different units
    uv2 = copy.deepcopy(uv_full)
    uv2.vis_units = "Jy"
    nt.assert_raises(ValueError, uv1.__iadd__, uv2)

    # Overlapping data
    uv2 = copy.deepcopy(uv_full)
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
    nt.assert_true(np.array_equal([0], indp))
    # Any of these inputs can also be a tuple of a tuple, so need to be checked twice.
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, pol)))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal([0], indp))

    # Combo with pol as string
    ind1, ind2, indp = uv._key2inds((ant1, ant2, uvutils.polnum2str(pol)))
    nt.assert_true(np.array_equal([0], indp))
    ind1, ind2, indp = uv._key2inds(((ant1, ant2, uvutils.polnum2str(pol))))
    nt.assert_true(np.array_equal([0], indp))

    # Check conjugation
    ind1, ind2, indp = uv._key2inds((ant2, ant1, pol))
    nt.assert_true(np.array_equal(bltind, ind2))
    nt.assert_true(np.array_equal(np.array([]), ind1))
    nt.assert_true(np.array_equal([0], indp))

    # Antpair only
    ind1, ind2, indp = uv._key2inds((ant1, ant2))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp))
    ind1, ind2, indp = uv._key2inds(((ant1, ant2)))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp))

    # Baseline number only
    ind1, ind2, indp = uv._key2inds(uv.antnums_to_baseline(ant1, ant2))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp))
    ind1, ind2, indp = uv._key2inds((uv.antnums_to_baseline(ant1, ant2)))
    nt.assert_true(np.array_equal(bltind, ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.arange(uv.Npols), indp))

    # Pol number only
    ind1, ind2, indp = uv._key2inds(pol)
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([0]), indp))
    ind1, ind2, indp = uv._key2inds((pol))
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([0]), indp))

    # Pol string only
    ind1, ind2, indp = uv._key2inds('LL')
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([1]), indp))
    ind1, ind2, indp = uv._key2inds(('LL'))
    nt.assert_true(np.array_equal(np.arange(uv.Nblts), ind1))
    nt.assert_true(np.array_equal(np.array([]), ind2))
    nt.assert_true(np.array_equal(np.array([1]), indp))

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
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))
    nt.assert_false(d.flags.writeable)
    # Ensure a view was returned
    uv.data_array[ind1[1], 0, 0, indp[0]] = 5.43
    nt.assert_equal(d[1, 0, 0], uv.data_array[ind1[1], 0, 0, indp[0]])

    # force copy
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp, force_copy=True)
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
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
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
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
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
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
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
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1 empty, ind2 reg, pol not reg
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1 empty, ind2 not reg, pol reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1 empty, ind2 not reg, pol not reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = uv.data_array[ind2, :, :, :]
    dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    nt.assert_true(np.all(d == dcheck))

    # ind1, ind2 not empty, pol reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = np.append(uv.data_array[ind1, :, :, :],
                       np.conj(uv.data_array[ind2, :, :, :]), axis=0)
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))

    # ind1, ind2 not empty, pol not reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = np.append(uv.data_array[ind1, :, :, :],
                       np.conj(uv.data_array[ind2, :, :, :]), axis=0)
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))

    # test single element
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp], axis=1)
    nt.assert_true(np.all(d == dcheck))

    # test single element
    ind1 = []
    ind2 = [45]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp)
    nt.assert_true(np.all(d == np.conj(dcheck)))

    # Full squeeze
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, indp, squeeze='full')
    dcheck = uv.data_array[ind1, :, :, :]
    dcheck = np.squeeze(dcheck[:, :, :, indp])
    nt.assert_true(np.all(d == dcheck))


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


def test_get_pols():
    # Test function to get unique polarizations in string format
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    pols = uv.get_pols()
    pols_data = ['RR', 'LL', 'LR', 'RL']
    nt.assert_items_equal(pols, pols_data)


def test_get_feedpols():
    # Test function to get unique antenna feed polarizations in data. String format.
    uv = UVData()
    testfile = os.path.join(
        DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    uvtest.checkWarnings(uv.read_uvfits, [testfile],
                         message='Telescope EVLA is not')
    pols = uv.get_feedpols()
    pols_data = ['R', 'L']
    nt.assert_items_equal(pols, pols_data)

    # Test break when stokes visibilities are present
    uv.polarization_array[0] = 1  # Stokes I
    nt.assert_raises(ValueError, uv.get_feedpols)
