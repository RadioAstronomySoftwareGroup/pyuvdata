# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvdata object."""
import pytest
import os
import copy
import itertools
import h5py

import numpy as np
from astropy import units
from astropy.time import Time
from astropy.utils import iers
from astropy.coordinates import Angle, EarthLocation, SkyCoord

from pyuvdata import UVData, UVCal
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

# needed for multifile read error test
from pyuvdata.uvdata.tests.test_mwa_corr_fits import filelist as mwa_corr_files
from pyuvdata.uvdata.tests.test_fhd import testfiles as fhd_files


from collections import Counter


@pytest.fixture(scope="function")
def uvdata_props():
    required_properties = [
        "data_array",
        "nsample_array",
        "flag_array",
        "Ntimes",
        "Nbls",
        "Nblts",
        "Nfreqs",
        "Npols",
        "Nspws",
        "uvw_array",
        "time_array",
        "ant_1_array",
        "ant_2_array",
        "lst_array",
        "baseline_array",
        "freq_array",
        "polarization_array",
        "spw_array",
        "integration_time",
        "channel_width",
        "object_name",
        "telescope_name",
        "instrument",
        "telescope_location",
        "history",
        "vis_units",
        "Nants_data",
        "Nants_telescope",
        "antenna_names",
        "antenna_numbers",
        "antenna_positions",
        "phase_type",
        "flex_spw",
        "future_array_shapes",
        "multi_phase_center",
    ]
    required_parameters = ["_" + prop for prop in required_properties]

    extra_properties = [
        "extra_keywords",
        "x_orientation",
        "antenna_diameters",
        "blt_order",
        "gst0",
        "rdate",
        "earth_omega",
        "dut1",
        "timesys",
        "uvplane_reference_time",
        "phase_center_ra",
        "phase_center_dec",
        "phase_center_app_ra",
        "phase_center_app_dec",
        "phase_center_frame_pa",
        "phase_center_epoch",
        "phase_center_frame",
        "Nphase",
        "phase_center_catalog",
        "phase_center_id_array",
        "scan_number_array",
        "eq_coeffs",
        "eq_coeffs_convention",
        "flex_spw_id_array",
        "filename",
    ]
    extra_parameters = ["_" + prop for prop in extra_properties]

    other_properties = [
        "telescope_location_lat_lon_alt",
        "telescope_location_lat_lon_alt_degrees",
        "phase_center_ra_degrees",
        "phase_center_dec_degrees",
        "pyuvdata_version_str",
    ]

    uv_object = UVData()

    class DataHolder:
        def __init__(
            self,
            uv_object,
            required_parameters,
            required_properties,
            extra_parameters,
            extra_properties,
            other_properties,
        ):
            self.uv_object = uv_object
            self.required_parameters = required_parameters
            self.required_properties = required_properties
            self.extra_parameters = extra_parameters
            self.extra_properties = extra_properties
            self.other_properties = other_properties

    uvdata_props = DataHolder(
        uv_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    )
    # yields the data we need but will continue to the del call after tests
    yield uvdata_props

    # some post-test object cleanup
    del uvdata_props

    return


@pytest.fixture(scope="session")
def hera_uvh5_split_main(hera_uvh5_main):
    # Get some meta info up from
    unique_times = np.unique(hera_uvh5_main.time_array)
    mid_pt = int(len(unique_times) * 0.5)

    # We'll split the data in half here
    uv1 = hera_uvh5_main.select(times=unique_times[:mid_pt], inplace=False)
    uv2 = hera_uvh5_main.select(times=unique_times[mid_pt:], inplace=False)

    yield uv1, uv2, hera_uvh5_main

    # clean up when done
    del uv1, uv2

    return


@pytest.fixture(scope="function")
def hera_uvh5_split(hera_uvh5_split_main):
    uv1, uv2, uvfull = hera_uvh5_split_main
    uv1_copy = uv1.copy()
    uv2_copy = uv2.copy()
    uvfull_copy = uvfull.copy()

    yield uv1_copy, uv2_copy, uvfull_copy

    # clean up when done
    del uv1_copy, uv2_copy, uvfull_copy

    return


@pytest.fixture(scope="session")
def hera_uvh5_xx_main():
    """Read in a HERA uvh5 file."""
    hera_uvh5_xx = UVData()
    hera_uvh5_xx.read_uvh5(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.uvh5"))

    yield hera_uvh5_xx

    # clean up when done
    del hera_uvh5_xx

    return


@pytest.fixture(scope="function")
def hera_uvh5_xx(hera_uvh5_xx_main):
    """Make function level HERA uvh5 file based object."""
    hera_uvh5_xx = hera_uvh5_xx_main.copy()

    yield hera_uvh5_xx

    # clean up when done
    del hera_uvh5_xx

    return


@pytest.fixture(scope="session")
def sma_mir_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_object.read(testfile)

    yield uv_object


@pytest.fixture(scope="function")
def sma_mir(sma_mir_main):
    # read in test file for the resampling in time functions
    uv_object = sma_mir_main.copy()

    yield uv_object


@pytest.fixture(scope="session")
def sma_mir_catalog(sma_mir_main):
    catalog_dict = sma_mir_main.phase_center_catalog

    yield catalog_dict


@pytest.fixture(scope="session")
def carma_miriad_main():
    # read in test file for the resampling in time functions
    pytest.importorskip("pyuvdata._miriad")
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "carma_miriad")
    uv_object.read(testfile, run_check=False, check_extra=False)
    uv_object.extra_keywords = None

    yield uv_object


@pytest.fixture(scope="session")
def carma_miriad(carma_miriad_main):
    pytest.importorskip("pyuvdata._miriad")
    uv_object = carma_miriad_main.copy()

    yield uv_object


@pytest.fixture(scope="session")
def paper_uvh5_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    uvh5_file = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA.uvh5")
    uv_object.read_uvh5(uvh5_file)

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def paper_uvh5(paper_uvh5_main):
    # read in test file for the resampling in time functions
    uv_object = paper_uvh5_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="session")
def bda_test_file_main():
    # read in test file for BDA-like data
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "simulated_bda_file.uvh5")
    uv_object.read(testfile)

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def bda_test_file(bda_test_file_main):
    # read in test file for BDA-like data
    uv_object = bda_test_file_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="session")
def pyuvsim_redundant_main():
    # read in test file for the compress/inflate redundancy functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    uv_object.read(testfile)

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def pyuvsim_redundant(pyuvsim_redundant_main):
    # read in test file for the compress/inflate redundancy functions
    uv_object = pyuvsim_redundant_main.copy()

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def uvdata_baseline():
    uv_object = UVData()
    uv_object.Nants_telescope = 128
    uv_object2 = UVData()
    uv_object2.Nants_telescope = 2049

    class DataHolder:
        def __init__(self, uv_object, uv_object2):
            self.uv_object = uv_object
            self.uv_object2 = uv_object2

    uvdata_baseline = DataHolder(uv_object, uv_object2)

    # yields the data we need but will continue to the del call after tests
    yield uvdata_baseline

    # Post test clean-up
    del uvdata_baseline
    return


@pytest.fixture(scope="session")
def set_uvws_main(hera_uvh5_main):
    uv1 = hera_uvh5_main.copy()
    # uvws in the file are wrong. reset them.
    uv1.set_uvws_from_antenna_positions()

    yield uv1

    del uv1

    return


@pytest.fixture
def uv1_2_set_uvws(set_uvws_main):
    uv1 = set_uvws_main.copy()
    uv2 = set_uvws_main.copy()

    yield uv1, uv2

    del uv1, uv2

    return


@pytest.fixture()
def uv_phase_time_split(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws

    uv_phase.reorder_blts(order="time", minor_order="baseline")
    uv_raw.reorder_blts(order="time", minor_order="baseline")

    uv_phase.phase(ra=0, dec=0, epoch="J2000", use_ant_pos=True)
    times = np.unique(uv_phase.time_array)
    time_set_1, time_set_2 = times[::2], times[1::2]

    uv_phase_1 = uv_phase.select(times=time_set_1, inplace=False)
    uv_phase_2 = uv_phase.select(times=time_set_2, inplace=False)

    uv_raw_1 = uv_raw.select(times=time_set_1, inplace=False)
    uv_raw_2 = uv_raw.select(times=time_set_2, inplace=False)

    yield uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw

    del uv_phase_1, uv_phase_2, uv_raw_1, uv_raw_2, uv_phase, uv_raw


@pytest.fixture(scope="session")
def uv_phase_comp_main():
    file1 = os.path.join(DATA_PATH, "1133866760.uvfits")
    file2 = os.path.join(DATA_PATH, "1133866760_rephase.uvfits")
    uvd1 = UVData()
    uvd2 = UVData()
    uvd1.read_uvfits(file1, fix_old_proj=True)
    uvd2.read_uvfits(file2, fix_old_proj=True)

    yield uvd1, uvd2


@pytest.fixture(scope="function")
def uv_phase_comp(uv_phase_comp_main):
    uvd1, uvd2 = uv_phase_comp_main
    uvd1_copy = uvd1.copy()
    uvd2_copy = uvd2.copy()

    yield uvd1_copy, uvd2_copy


@pytest.fixture()
def dummy_phase_dict():
    dummy_dict = {
        "cat_name": "z1",
        "cat_type": "sidereal",
        "cat_lon": 0.0,
        "cat_lat": 1.0,
        "cat_frame": "fk5",
        "cat_epoch": 2000.0,
        "cat_times": None,
        "cat_pm_ra": 0.0,
        "cat_pm_dec": 0.0,
        "cat_dist": 0.0,
        "cat_vrad": 0.0,
        "info_source": "user",
        "cat_id": None,
    }

    return dummy_dict


def test_parameter_iter(uvdata_props):
    """Test expected parameters."""
    all_params = []
    for prop in uvdata_props.uv_object:
        all_params.append(prop)
    for a in uvdata_props.required_parameters + uvdata_props.extra_parameters:
        assert a in all_params, (
            "expected attribute " + a + " not returned in object iterator"
        )


def test_required_parameter_iter(uvdata_props):
    """Test expected required parameters."""
    # at first it's a metadata_only object, so need to modify required_parameters
    required = []
    for prop in uvdata_props.uv_object.required():
        required.append(prop)
    expected_required = copy.copy(uvdata_props.required_parameters)
    expected_required.remove("_data_array")
    expected_required.remove("_nsample_array")
    expected_required.remove("_flag_array")
    for a in expected_required:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )

    uvdata_props.uv_object.data_array = 1
    uvdata_props.uv_object.nsample_array = 1
    uvdata_props.uv_object.flag_array = 1
    required = []
    for prop in uvdata_props.uv_object.required():
        required.append(prop)
    for a in uvdata_props.required_parameters:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )


def test_extra_parameter_iter(uvdata_props):
    """Test expected optional parameters."""
    extra = []
    for prop in uvdata_props.uv_object.extra():
        extra.append(prop)
    for a in uvdata_props.extra_parameters:
        assert a in extra, "expected attribute " + a + " not returned in extra iterator"


def test_unexpected_parameters(uvdata_props):
    """Test for extra parameters."""
    expected_parameters = (
        uvdata_props.required_parameters + uvdata_props.extra_parameters
    )
    attributes = [i for i in uvdata_props.uv_object.__dict__.keys() if i[0] == "_"]
    for a in attributes:
        assert a in expected_parameters, (
            "unexpected parameter " + a + " found in UVData"
        )


def test_unexpected_attributes(uvdata_props):
    """Test for extra attributes."""
    expected_attributes = (
        uvdata_props.required_properties
        + uvdata_props.extra_properties
        + uvdata_props.other_properties
    )
    attributes = [i for i in uvdata_props.uv_object.__dict__.keys() if i[0] != "_"]
    for a in attributes:
        assert a in expected_attributes, (
            "unexpected attribute " + a + " found in UVData"
        )


def test_properties(uvdata_props):
    """Test that properties can be get and set properly."""
    prop_dict = dict(
        list(
            zip(
                uvdata_props.required_properties + uvdata_props.extra_properties,
                uvdata_props.required_parameters + uvdata_props.extra_parameters,
            )
        )
    )
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(uvdata_props.uv_object, k, rand_num)
        this_param = getattr(uvdata_props.uv_object, v)
        try:
            assert rand_num == this_param.value
        except AssertionError:
            print("setting {prop_name} to a random number failed".format(prop_name=k))
            raise


def test_metadata_only_property(casa_uvfits):
    uvobj = casa_uvfits
    uvobj.data_array = None
    assert uvobj.metadata_only is False
    pytest.raises(ValueError, uvobj.check)
    uvobj.flag_array = None
    assert uvobj.metadata_only is False
    pytest.raises(ValueError, uvobj.check)
    uvobj.nsample_array = None
    assert uvobj.metadata_only is True


@pytest.mark.parametrize("filetype", ["miriad", "mir", "ms", "uvfits", "uvh5"])
def test_error_metadata_only_write(casa_uvfits, filetype, tmp_path):
    uvobj = casa_uvfits
    uvobj.data_array = None
    uvobj.flag_array = None
    uvobj.nsample_array = None
    assert uvobj.metadata_only is True

    out_file = os.path.join(tmp_path, "outtest." + filetype)
    with pytest.raises(ValueError, match="Cannot write out metadata only objects to a"):
        getattr(uvobj, "write_" + filetype)(out_file)


def test_equality(casa_uvfits):
    """Basic equality test."""
    uvobj = casa_uvfits
    assert uvobj == uvobj


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_nants_data(casa_uvfits):
    """Test check function."""
    uvobj = casa_uvfits

    assert uvobj.check()
    # Check variety of special cases
    uvobj.Nants_data += 1
    with pytest.raises(
        ValueError,
        match=(
            "Nants_data must be equal to the number of unique values in "
            "ant_1_array and ant_2_array"
        ),
    ):
        uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_nbls(casa_uvfits):
    uvobj = casa_uvfits
    uvobj.Nbls += 1
    with pytest.raises(
        ValueError,
        match=(
            "Nbls must be equal to the number of unique baselines in the data_array"
        ),
    ):
        uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_ntimes(casa_uvfits):
    uvobj = casa_uvfits
    uvobj.Ntimes += 1
    with pytest.raises(
        ValueError,
        match=("Ntimes must be equal to the number of unique times in the time_array"),
    ):
        uvobj.check()
    uvobj.Ntimes -= 1


def test_check_strict_uvw(casa_uvfits):
    uvobj = casa_uvfits
    with pytest.raises(
        ValueError,
        match=(
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ),
    ):
        uvobj.check(strict_uvw_antpos_check=True)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_autos_only(hera_uvh5_xx):
    """
    Check case where all data is autocorrelations
   """
    uvobj = hera_uvh5_xx

    uvobj.select(blt_inds=np.where(uvobj.ant_1_array == uvobj.ant_2_array)[0])
    assert uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_uvw_array(hera_uvh5_xx):
    # test auto and cross corr uvw_array
    uvd = hera_uvh5_xx.copy()
    autos = np.isclose(uvd.ant_1_array - uvd.ant_2_array, 0.0)
    auto_inds = np.where(autos)[0]
    cross_inds = np.where(~autos)[0]

    # make auto have non-zero uvw coords, assert ValueError
    uvd.uvw_array[auto_inds[0], 0] = 0.1
    with pytest.raises(
        ValueError,
        match=("Some auto-correlations have non-zero uvw_array coordinates."),
    ):
        uvd.check()

    # make cross have |uvw| zero, assert ValueError
    uvd = hera_uvh5_xx.copy()
    uvd.uvw_array[cross_inds[0]][:] = 0.0
    with pytest.raises(
        ValueError,
        match=("Some cross-correlations have near-zero uvw_array magnitudes."),
    ):
        uvd.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_flag_array(casa_uvfits):
    uvobj = casa_uvfits

    uvobj.flag_array = np.ones((uvobj.flag_array.shape), dtype=int)

    with pytest.raises(
        ValueError, match="UVParameter _flag_array is not the appropriate type.",
    ):
        uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_future_array_shape(casa_uvfits):
    """Convert to future shapes and check. Convert back and test for equality."""
    uvobj = casa_uvfits
    uvobj2 = casa_uvfits.copy()
    uvobj.use_future_array_shapes()
    uvobj.check()

    uvobj.use_current_array_shapes()
    uvobj.check()

    assert uvobj == uvobj2

    uvobj2.data_array = None
    uvobj2.flag_array = None
    uvobj2.nsample_array = None
    assert uvobj2.metadata_only

    uvobj2.use_future_array_shapes()
    assert uvobj2._data_array.expected_shape(uvobj2) == (
        uvobj2.Nblts,
        uvobj2.Nfreqs,
        uvobj2.Npols,
    )

    uvobj.use_future_array_shapes()
    uvobj.channel_width[-1] = uvobj.channel_width[0] * 2.0
    uvobj.check()

    with pytest.raises(
        ValueError, match="channel_width parameter contains multiple unique values"
    ):
        uvobj.use_current_array_shapes()

    with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
        uvobj._check_freq_spacing()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_nants_data_telescope_larger(casa_uvfits):
    uvobj = casa_uvfits
    # make sure it's okay for Nants_telescope to be strictly greater than Nants_data
    uvobj.Nants_telescope += 1
    # add dummy information for "new antenna" to pass object check
    uvobj.antenna_names = np.concatenate((uvobj.antenna_names, ["dummy_ant"]))
    uvobj.antenna_numbers = np.concatenate((uvobj.antenna_numbers, [20]))
    uvobj.antenna_positions = np.concatenate(
        (uvobj.antenna_positions, np.zeros((1, 3))), axis=0
    )
    assert uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_ant1_array_not_in_antnums(casa_uvfits):
    uvobj = casa_uvfits
    # make sure an error is raised if antennas in ant_1_array not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    uvobj.antenna_names = uvobj.antenna_names[1:]
    uvobj.antenna_numbers = uvobj.antenna_numbers[1:]
    uvobj.antenna_positions = uvobj.antenna_positions[1:, :]
    uvobj.Nants_telescope = uvobj.antenna_numbers.size
    with pytest.raises(
        ValueError, match="All antennas in ant_1_array must be in antenna_numbers"
    ):
        uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_ant2_array_not_in_antnums(casa_uvfits):
    uvobj = casa_uvfits
    # make sure an error is raised if antennas in ant_2_array not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    uvobj = uvobj
    uvobj.antenna_names = uvobj.antenna_names[:-1]
    uvobj.antenna_numbers = uvobj.antenna_numbers[:-1]
    uvobj.antenna_positions = uvobj.antenna_positions[:-1]
    uvobj.Nants_telescope = uvobj.antenna_numbers.size
    with pytest.raises(
        ValueError, match="All antennas in ant_2_array must be in antenna_numbers"
    ):
        uvobj.check()


def test_converttofiletype(casa_uvfits):
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()
    fhd_obj = uvobj._convert_to_filetype("fhd")
    uvobj._convert_from_filetype(fhd_obj)
    assert uvobj == uvobj2

    with pytest.raises(ValueError) as cm:
        uvobj._convert_to_filetype("foo")
    assert str(cm.value).startswith(
        "filetype must be uvfits, mir, miriad, ms, fhd, or uvh5"
    )


def test_baseline_to_antnums(uvdata_baseline):
    """Test baseline to antnum conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.baseline_to_antnums(67585) == (0, 0)
    with pytest.raises(Exception) as cm:
        uvdata_baseline.uv_object2.baseline_to_antnums(67585)
    assert str(cm.value).startswith(
        "error Nants={Nants}>2048"
        " not supported".format(Nants=uvdata_baseline.uv_object2.Nants_telescope)
    )

    ant_pairs = [(10, 20), (280, 310)]
    for pair in ant_pairs:
        if np.max(np.array(pair)) < 255:
            bl = uvdata_baseline.uv_object.antnums_to_baseline(
                pair[0], pair[1], attempt256=True
            )
            ant_pair_out = uvdata_baseline.uv_object.baseline_to_antnums(bl)
            assert pair == ant_pair_out

        bl = uvdata_baseline.uv_object.antnums_to_baseline(
            pair[0], pair[1], attempt256=False
        )
        ant_pair_out = uvdata_baseline.uv_object.baseline_to_antnums(bl)
        assert pair == ant_pair_out


def test_baseline_to_antnums_vectorized(uvdata_baseline):
    """Test vectorized antnum to baseline conversion."""
    ant_1 = [10, 280]
    ant_2 = [20, 310]
    baseline_array = uvdata_baseline.uv_object.antnums_to_baseline(ant_1, ant_2)
    assert np.array_equal(baseline_array, [88085, 641335])
    ant_1_out, ant_2_out = uvdata_baseline.uv_object.baseline_to_antnums(
        baseline_array.tolist()
    )
    assert np.array_equal(ant_1, ant_1_out)
    assert np.array_equal(ant_2, ant_2_out)


def test_antnums_to_baselines(uvdata_baseline):
    """Test antums to baseline conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0) == 67585
    assert uvdata_baseline.uv_object.antnums_to_baseline(257, 256) == 594177
    assert uvdata_baseline.uv_object.baseline_to_antnums(594177) == (257, 256)
    # Check attempt256
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0, attempt256=True) == 257
    assert uvdata_baseline.uv_object.antnums_to_baseline(257, 256) == 594177
    with uvtest.check_warnings(UserWarning, "found antenna numbers > 256"):
        uvdata_baseline.uv_object.antnums_to_baseline(257, 256, attempt256=True)
    pytest.raises(Exception, uvdata_baseline.uv_object2.antnums_to_baseline, 0, 0)
    # check a len-1 array returns as an array
    ant1 = np.array([1])
    ant2 = np.array([2])
    assert isinstance(
        uvdata_baseline.uv_object.antnums_to_baseline(ant1, ant2), np.ndarray
    )


def test_known_telescopes():
    """Test known_telescopes method returns expected results."""
    uv_object = UVData()
    astropy_sites = EarthLocation.get_site_names()
    while "" in astropy_sites:
        astropy_sites.remove("")

    # Using set to drop duplicate entries
    known_telescopes = list(
        set(astropy_sites + ["PAPER", "HERA", "SMA", "SZA", "OVRO-LWA"])
    )
    # calling np.sort().tolist() because [].sort() acts inplace and returns None
    # Before test had None == None
    assert (
        np.sort(known_telescopes).tolist()
        == np.sort(uv_object.known_telescopes()).tolist()
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_hera_diameters(paper_uvh5):
    uv_in = paper_uvh5

    uv_in.telescope_name = "HERA"
    with uvtest.check_warnings(
        UserWarning, "antenna_diameters is not set. Using known values for HERA."
    ):
        uv_in.set_telescope_params()

    assert uv_in.telescope_name == "HERA"
    assert uv_in.antenna_diameters is not None

    uv_in.check()


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_generic_read():
    uv_in = UVData()
    uvfits_file = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uv_in.read(uvfits_file, read_data=False)
    unique_times = np.unique(uv_in.time_array)

    with pytest.raises(
        ValueError, match="Only one of times and time_range can be provided."
    ):
        uv_in.read(
            uvfits_file,
            times=unique_times[0:2],
            time_range=[unique_times[0], unique_times[1]],
        )

    with pytest.raises(
        ValueError, match="Only one of antenna_nums and antenna_names can be provided."
    ):
        uv_in.read(
            uvfits_file,
            antenna_nums=uv_in.antenna_numbers[0],
            antenna_names=uv_in.antenna_names[1],
        )

    with pytest.raises(ValueError, match="File type could not be determined"):
        uv_in.read("foo")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    "phase_kwargs",
    [
        {"ra": 0.0, "dec": 0.0, "epoch": "J2000"},
        {"ra": Angle("5d").rad, "dec": Angle("30d").rad, "phase_frame": "gcrs"},
        {
            "ra": Angle("180d").rad,
            "dec": Angle("90d").rad,
            "epoch": Time("2010-01-01T00:00:00", format="isot", scale="utc"),
        },
    ],
)
def test_phase_unphase_hera(uv1_2_set_uvws, future_shapes, phase_kwargs):
    """
    Read in drift data, phase to an RA/DEC, unphase and check for object equality.
    """
    uv1, uv_raw = uv1_2_set_uvws

    if future_shapes:
        uv1.use_future_array_shapes()
        uv_raw.use_future_array_shapes()

    uv1.phase(**phase_kwargs)
    uv1.unphase_to_drift()
    # check that phase + unphase gets back to raw
    assert uv_raw == uv1


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_unphase_hera_one_bl(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check that phase + unphase work with one baseline
    uv_raw_small = uv_raw.select(blt_inds=[0], inplace=False)
    uv_phase_small = uv_raw_small.copy()
    uv_phase_small.phase(Angle("23h").rad, Angle("15d").rad)
    uv_phase_small.unphase_to_drift()
    assert uv_raw_small == uv_phase_small


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_unphase_hera_antpos(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check that they match if you phase & unphase using antenna locations
    # first replace the uvws with the right values
    antenna_enu = uvutils.ENU_from_ECEF(
        (uv_raw.antenna_positions + uv_raw.telescope_location),
        *uv_raw.telescope_location_lat_lon_alt,
    )
    uvw_calc = np.zeros_like(uv_raw.uvw_array)
    unique_times, unique_inds = np.unique(uv_raw.time_array, return_index=True)
    for ind, jd in enumerate(unique_times):
        inds = np.where(uv_raw.time_array == jd)[0]
        for bl_ind in inds:
            wh_ant1 = np.where(uv_raw.antenna_numbers == uv_raw.ant_1_array[bl_ind])
            ant1_index = wh_ant1[0][0]
            wh_ant2 = np.where(uv_raw.antenna_numbers == uv_raw.ant_2_array[bl_ind])
            ant2_index = wh_ant2[0][0]
            uvw_calc[bl_ind, :] = (
                antenna_enu[ant2_index, :] - antenna_enu[ant1_index, :]
            )

    uv_raw_new = uv_raw.copy()
    uv_raw_new.uvw_array = uvw_calc
    uv_phase.phase(0.0, 0.0, epoch="J2000", use_ant_pos=True)
    uv_phase2 = uv_raw_new.copy()
    uv_phase2.phase(0.0, 0.0, epoch="J2000")

    # The uvw's only agree to ~1mm. should they be better?
    assert np.allclose(uv_phase2.uvw_array, uv_phase.uvw_array, atol=1e-3)
    # the data array are just multiplied by the w's for phasing, so a difference
    # at the 1e-3 level makes the data array different at that level too.
    # -> change the tolerance on data_array for this test
    uv_phase2._data_array.tols = (0, 1e-3 * np.amax(np.abs(uv_phase2.data_array)))
    assert uv_phase2 == uv_phase

    # check that phase + unphase gets back to raw using antpos
    uv_phase.unphase_to_drift(use_ant_pos=True)
    assert uv_raw_new == uv_phase


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_hera_zenith_timestamp_minimal_changes(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check that phasing to zenith with one timestamp has small changes
    # (it won't be identical because of precession/nutation changing the
    # coordinate axes)
    # use gcrs rather than icrs to reduce differences (don't include abberation)
    uv_raw_small = uv_raw.select(times=uv_raw.time_array[0], inplace=False)
    uv_phase_simple_small = uv_raw_small.copy()
    uv_phase_simple_small.phase_to_time(
        time=Time(uv_raw.time_array[0], format="jd"), phase_frame="gcrs"
    )

    # it's unclear to me how close this should be...
    assert np.allclose(
        uv_phase_simple_small.uvw_array, uv_raw_small.uvw_array, atol=1e-1
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_to_time_jd_input(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    uv_phase.phase_to_time(uv_raw.time_array[0])
    uv_phase.unphase_to_drift()
    assert uv_phase == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_to_time_error(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check error if not passing a Time object to phase_to_time
    with pytest.raises(TypeError) as cm:
        uv_phase.phase_to_time("foo")
    assert str(cm.value).startswith("time must be an astropy.time.Time object")


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_unphase_drift_data_error(uv1_2_set_uvws, sma_mir, future_shapes):
    uv_phase, uv_drift = uv1_2_set_uvws
    # check error unphasing an unphased object

    if future_shapes:
        uv_phase.use_future_array_shapes()
        uv_drift.use_future_array_shapes()
        sma_mir.use_future_array_shapes()

    with pytest.raises(ValueError) as cm:
        uv_drift.unphase_to_drift()
    assert str(cm.value).startswith("The data is already drift scanning;")

    # Test to make sure we get the right errors when usng the old proj method
    uv_phase.phase(0.0, 0.0, use_old_proj=True)
    with pytest.raises(AttributeError) as cm:
        uv_phase.unphase_to_drift()
    assert str(cm.value).startswith("Object missing phase_center_ra_app or")

    # Now make sure the old proj method works with unphasing
    uv_phase.unphase_to_drift(use_old_proj=True)
    assert uv_drift == uv_phase

    with pytest.raises(ValueError) as cm:
        sma_mir.unphase_to_drift(use_old_proj=True)
    assert str(cm.value).startswith("Multi phase center data sets are not compatible")

    # Check to make sure that wa can unphase w/o an error getting thrown. The new
    # unphase method does not throw an error when being called twice
    sma_mir.unphase_to_drift()
    sma_mir.unphase_to_drift()


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "phase_func,phase_kwargs,err_msg",
    [
        (
            "phase",
            {"ra": 0, "dec": 0, "epoch": "J2000", "allow_rephase": False},
            "The data is already phased;",
        ),
        (
            "phase_to_time",
            {"time": 0, "allow_rephase": False},
            "The data is already phased;",
        ),
    ],
)
def test_phase_rephase_hera_errors(uv1_2_set_uvws, phase_func, phase_kwargs, err_msg):
    uv_phase, uv_raw = uv1_2_set_uvws

    uv_phase.phase(0.0, 0.0, epoch="J2000")
    # if this is phase_to_time, use this index set in the dictionary and
    # assign the value of the time_array associated with that index
    # this is a little hacky, but we cannot acces uv_phase.time_array in the
    # parametrize
    if phase_func == "phase_to_time":
        phase_kwargs["time"] = uv_phase.time_array[int(phase_kwargs["time"])]

    with pytest.raises(ValueError) as cm:
        getattr(uv_phase, phase_func)(**phase_kwargs)
    assert str(cm.value).startswith(err_msg)


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_unphase_hera_bad_frame(uv1_2_set_uvws):
    uv_phase, uv_raw = uv1_2_set_uvws
    # check errors when trying to phase to an unsupported frame
    with pytest.raises(ValueError) as cm:
        uv_phase.phase(0.0, 0.0, epoch="J2000", phase_frame="cirs", use_old_proj=True)
    assert str(cm.value).startswith("phase_frame can only be set to icrs or gcrs.")


@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("use_ant_pos1", [True, False])
@pytest.mark.parametrize("use_ant_pos2", [True, False])
def test_unphasing(uv_phase_comp, future_shapes, use_ant_pos1, use_ant_pos2):
    uvd1, uvd2 = uv_phase_comp

    if future_shapes:
        uvd1.use_future_array_shapes()
        uvd2.use_future_array_shapes()

    uvd1.unphase_to_drift(phase_frame="fk5", use_ant_pos=use_ant_pos1)
    uvd2.unphase_to_drift(phase_frame="fk5", use_ant_pos=use_ant_pos2)

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1.uvw_array, uvd2.uvw_array, atol=1e-12)


@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("use_ant_pos", [True, False])
@pytest.mark.parametrize("unphase_first", [True, False])
def test_phasing(uv_phase_comp, future_shapes, unphase_first, use_ant_pos):
    uvd1, uvd2 = uv_phase_comp
    if future_shapes:
        uvd1.use_future_array_shapes()
        uvd2.use_future_array_shapes()

    if unphase_first:
        uvd2.unphase_to_drift(phase_frame="fk5", use_ant_pos=use_ant_pos)

    uvd2.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        orig_phase_frame="fk5",
        phase_frame=uvd1.phase_center_frame,
        use_ant_pos=use_ant_pos,
    )

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1.uvw_array, uvd2.uvw_array, atol=1e-12)


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.parametrize(
    "arg_dict,set_phased,err_type,msg",
    [
        [{"mask": [True]}, False, ValueError, "Cannot apply a selection mask"],
        [{"pm_ra": 1}, False, ValueError, "Non-zero values of pm_ra not supported"],
        [{"pm_dec": 1}, False, ValueError, "Non-zero values of pm_dec not supported"],
        [{"dist": 1}, False, ValueError, "Non-zero values of dist not supported"],
        [{"vrad": 1}, False, ValueError, "Non-zero values of vrad not supported"],
        [{"type": "ephem"}, False, ValueError, "Only sidereal sources are supported"],
        [{"name": "abc", "lookup": True}, False, ValueError, "Object name lookup is"],
        [{"fix": False, "usepos": False}, True, AttributeError, "Data missing phase_"],
    ],
)
def test_phasing_non_multi_phase_errs(hera_uvh5, arg_dict, set_phased, err_type, msg):
    """
    Test expected phasing errors related to mutli-phase-ctr data sets
    """
    if set_phased:
        hera_uvh5._set_phased()
    with pytest.raises(err_type) as cm:
        hera_uvh5.phase(
            0,
            0,
            select_mask=arg_dict.get("mask"),
            pm_ra=arg_dict.get("pm_ra"),
            pm_dec=arg_dict.get("pm_dec"),
            dist=arg_dict.get("dist"),
            vrad=arg_dict.get("vrad"),
            cat_type=arg_dict.get("type"),
            cat_name=arg_dict.get("name"),
            lookup_name=arg_dict.get("lookup"),
            fix_old_proj=arg_dict.get("fix"),
            use_ant_pos=arg_dict.get("usepos"),
        )
    assert str(cm.value).startswith(msg)


@pytest.mark.parametrize(
    "arg_dict,err_type,msg",
    [
        [{}, ValueError, "Must supply a unique name for cat_name"],
        [
            {"name": "abc", "oldproj": True},
            NotImplementedError,
            "Multi phase center data sets are not",
        ],
        [{"name": "abc", "mask": [True] * 2}, IndexError, "Selection mask must be of"],
    ],
)
def test_phasing_multi_phase_errs(sma_mir, arg_dict, err_type, msg):
    # Now do a few things that aren't allowed w/ a mutli-phase-ctr data set
    with pytest.raises(err_type) as cm:
        sma_mir.phase(
            0,
            0,
            cat_name=arg_dict.get("name"),
            use_old_proj=arg_dict.get("oldproj"),
            select_mask=arg_dict.get("mask"),
        )
    assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_phasing_fix_old_proj(hera_uvh5, future_shapes):
    if future_shapes:
        hera_uvh5.use_future_array_shapes()
    # Finally, make sure that the fix_old_proj switch works correctly
    hera_copy = hera_uvh5.copy()
    hera_uvh5.phase(0, 0, use_old_proj=True, use_ant_pos=False)
    hera_uvh5.phase(0, 0, use_ant_pos=False)
    hera_copy.phase(0, 0)

    # The fix introduces small errors on the order of 0.1 deg, when not using antenna
    # positions, hence the special handling here
    assert np.allclose(hera_copy.data_array, hera_uvh5.data_array, rtol=3e-4)

    # Once data are verified, make sure that everything else looks okay
    hera_uvh5.data_array = hera_copy.data_array
    assert hera_uvh5 == hera_copy


# We're using the old phase method here since these values were all derived using that
# method, so we'll just filter out those warnings now.
@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_old_phasing(future_shapes):
    """Use MWA files phased to 2 different places to test phasing."""
    file1 = os.path.join(DATA_PATH, "1133866760.uvfits")
    file2 = os.path.join(DATA_PATH, "1133866760_rephase.uvfits")
    uvd1 = UVData()
    uvd2 = UVData()
    uvd1.read_uvfits(file1)
    uvd2.read_uvfits(file2)

    if future_shapes:
        uvd1.use_future_array_shapes()
        uvd2.use_future_array_shapes()

    uvd1_drift = uvd1.copy()
    uvd1_drift.unphase_to_drift(
        phase_frame="gcrs", use_old_proj=True, use_ant_pos=False
    )
    uvd1_drift_antpos = uvd1.copy()
    uvd1_drift_antpos.unphase_to_drift(
        phase_frame="gcrs", use_ant_pos=True, use_old_proj=True,
    )

    uvd2_drift = uvd2.copy()
    uvd2_drift.unphase_to_drift(
        phase_frame="gcrs", use_old_proj=True, use_ant_pos=False
    )
    uvd2_drift_antpos = uvd2.copy()
    uvd2_drift_antpos.unphase_to_drift(
        phase_frame="gcrs", use_ant_pos=True, use_old_proj=True,
    )

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1_drift.uvw_array, uvd2_drift.uvw_array, atol=2e-2)
    assert np.allclose(uvd1_drift_antpos.uvw_array, uvd2_drift_antpos.uvw_array)

    uvd2_rephase = uvd2.copy()
    uvd2_rephase.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        orig_phase_frame="gcrs",
        phase_frame="gcrs",
        use_ant_pos=False,
        use_old_proj=True,
    )
    uvd2_rephase_antpos = uvd2.copy()
    uvd2_rephase_antpos.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        orig_phase_frame="gcrs",
        phase_frame="gcrs",
        use_ant_pos=True,
        use_old_proj=True,
    )

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1.uvw_array, uvd2_rephase.uvw_array, atol=2e-2)
    assert np.allclose(uvd1.uvw_array, uvd2_rephase_antpos.uvw_array, atol=5e-3)

    # rephase the drift objects to the original pointing and verify that they
    # match
    uvd1_drift.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        phase_frame="gcrs",
        use_ant_pos=False,
        use_old_proj=True,
    )
    uvd1_drift_antpos.phase(
        uvd1.phase_center_ra,
        uvd1.phase_center_dec,
        uvd1.phase_center_epoch,
        phase_frame="gcrs",
        use_ant_pos=True,
        use_old_proj=True,
    )

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd1.uvw_array, uvd1_drift.uvw_array, atol=1e-4)
    assert np.allclose(uvd1.uvw_array, uvd1_drift_antpos.uvw_array, atol=5e-3)

    uvd2_drift.phase(
        uvd2.phase_center_ra,
        uvd2.phase_center_dec,
        uvd2.phase_center_epoch,
        phase_frame="gcrs",
        use_ant_pos=False,
        use_old_proj=True,
    )
    uvd2_drift_antpos.phase(
        uvd2.phase_center_ra,
        uvd2.phase_center_dec,
        uvd2.phase_center_epoch,
        phase_frame="gcrs",
        use_ant_pos=True,
        use_old_proj=True,
    )

    # the tolerances here are empirical -- caused by one unphase/phase cycle.
    # the antpos-based phasing differences are based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    assert np.allclose(uvd2.uvw_array, uvd2_drift.uvw_array, atol=1e-4)
    assert np.allclose(uvd2.uvw_array, uvd2_drift_antpos.uvw_array, atol=2e-2)

    # Check to make sure that the old errors work
    with pytest.raises(ValueError, match="The data is already phased;"):
        uvd1_drift.phase(0, 0, use_old_proj=True, allow_rephase=False)

    uvd1_drift.phase_type = "unk"
    with pytest.raises(ValueError, match="The phasing type of the data is unknown"):
        uvd1_drift.phase(0, 0, use_old_proj=True)

    uvd1_drift = uvd1.copy()
    # Move the time ~1 µsec off from J2000
    epoch_val = Time(Time(2000, format="jyear").mjd - 1e-11, format="mjd")
    # Unlike in the new phasing system, this should produce different results (since one
    # is FK5, and the other is ICRS)
    uvd1_drift.phase(0, 0, epoch=epoch_val, use_old_proj=True)
    uvd1.phase(0, 0, epoch="J2000", use_old_proj=True)
    assert uvd1_drift != uvd1
    uvd1_drift = uvd1.copy()
    uvd1_drift.phase_center_frame = None

    # Make sure the old default works for reverting to ICRS if no coord frame is found
    uvd1.unphase_to_drift(use_old_proj=True)
    uvd1_drift.unphase_to_drift(use_old_proj=True)
    assert uvd1 == uvd1_drift


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_blts(paper_uvh5, future_shapes):
    uv_object = paper_uvh5
    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history
    # fmt: off
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
    # fmt: on
    selected_data = uv_object.data_array[np.sort(blt_inds)]

    uv_object2 = uv_object.copy()
    uv_object2.select(blt_inds=blt_inds)
    assert len(blt_inds) == uv_object2.Nblts

    # verify that histories are different
    assert not uvutils._check_histories(old_history, uv_object2.history)

    assert uvutils._check_histories(
        old_history + "  Downselected to specific baseline-times using pyuvdata.",
        uv_object2.history,
    )

    assert np.all(selected_data == uv_object2.data_array)

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(blt_inds=blt_inds[np.newaxis, :])
    assert len(blt_inds) == uv_object2.Nblts

    assert uvutils._check_histories(
        old_history + "  Downselected to specific baseline-times using pyuvdata.",
        uv_object2.history,
    )
    assert np.all(selected_data == uv_object2.data_array)

    # check that just doing the metadata works properly
    uv_object3 = uv_object.copy()
    uv_object3.data_array = None
    uv_object3.flag_array = None
    uv_object3.nsample_array = None
    assert uv_object3.metadata_only is True
    uv_object4 = uv_object3.select(blt_inds=blt_inds, inplace=False)
    for param in uv_object4:
        param_name = getattr(uv_object4, param).name
        if param_name not in ["data_array", "flag_array", "nsample_array"]:
            assert getattr(uv_object4, param) == getattr(uv_object2, param)
        else:
            assert getattr(uv_object4, param_name) is None

    # also check with inplace=True
    uv_object3.select(blt_inds=blt_inds)
    assert uv_object3 == uv_object4

    # check for errors associated with out of bounds indices
    pytest.raises(ValueError, uv_object.select, blt_inds=np.arange(-10, -5))
    pytest.raises(
        ValueError,
        uv_object.select,
        blt_inds=np.arange(uv_object.Nblts + 1, uv_object.Nblts + 10),
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_antennas(casa_uvfits):
    uv_object = casa_uvfits

    old_history = uv_object.history
    unique_ants = np.unique(
        uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist()
    )
    ants_to_keep = np.array([0, 19, 11, 24, 3, 23, 1, 20, 21])

    blts_select = [
        (a1 in ants_to_keep) & (a2 in ants_to_keep)
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = uv_object.copy()
    uv_object2.select(antenna_nums=ants_to_keep)

    assert len(ants_to_keep) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in ants_to_keep:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in ants_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific antennas using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(antenna_nums=ants_to_keep[np.newaxis, :])

    assert len(ants_to_keep) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in ants_to_keep:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in ants_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific antennas using pyuvdata.",
        uv_object2.history,
    )

    # now test using antenna_names to specify antennas to keep
    uv_object3 = uv_object.copy()
    ants_to_keep = np.array(sorted(ants_to_keep))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=ant_names)

    assert uv_object2 == uv_object3

    # check that it also works with higher dimension array
    uv_object3 = uv_object.copy()
    ants_to_keep = np.array(sorted(ants_to_keep))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(uv_object3.antenna_numbers == a)[0][0]
        ant_names.append(uv_object3.antenna_names[ind])

    uv_object3.select(antenna_names=[ant_names])

    assert uv_object2 == uv_object3

    # test removing metadata associated with antennas that are no longer present
    # also add (different) antenna_diameters to test downselection
    uv_object.antenna_diameters = 1.0 * np.ones(
        (uv_object.Nants_telescope,), dtype=np.float64
    )
    for i in range(uv_object.Nants_telescope):
        uv_object.antenna_diameters += i
    uv_object4 = uv_object.copy()
    uv_object4.select(antenna_nums=ants_to_keep, keep_all_metadata=False)
    assert uv_object4.Nants_telescope == 9
    assert set(uv_object4.antenna_numbers) == set(ants_to_keep)
    for a in ants_to_keep:
        idx1 = uv_object.antenna_numbers.tolist().index(a)
        idx2 = uv_object4.antenna_numbers.tolist().index(a)
        assert uv_object.antenna_names[idx1] == uv_object4.antenna_names[idx2]
        assert np.allclose(
            uv_object.antenna_positions[idx1, :], uv_object4.antenna_positions[idx2, :]
        )
        assert uv_object.antenna_diameters[idx1], uv_object4.antenna_diameters[idx2]

    # remove antenna_diameters from object
    uv_object.antenna_diameters = None

    # check for errors associated with antennas not included in data, bad names
    # or providing numbers and names
    pytest.raises(
        ValueError, uv_object.select, antenna_nums=np.max(unique_ants) + np.arange(1, 3)
    )
    pytest.raises(ValueError, uv_object.select, antenna_names="test1")
    pytest.raises(
        ValueError, uv_object.select, antenna_nums=ants_to_keep, antenna_names=ant_names
    )


def sort_bl(p):
    """Sort a tuple that starts with a pair of antennas, and may have stuff after."""
    if p[1] >= p[0]:
        return p
    return (p[1], p[0]) + p[2:]


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_bls(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    first_ants = [6, 2, 7, 2, 21, 27, 8]
    second_ants = [0, 20, 8, 1, 2, 3, 22]
    new_unique_ants = np.unique(first_ants + second_ants)
    ant_pairs_to_keep = list(zip(first_ants, second_ants))
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    blts_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = uv_object.copy()
    uv_object2.select(bls=ant_pairs_to_keep)
    sorted_pairs_object2 = [
        sort_bl(p) for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object2
    for pair in sorted_pairs_object2:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific baselines using pyuvdata.",
        uv_object2.history,
    )

    # check using baseline number parameter
    uv_object3 = uv_object.copy()
    bls_nums_to_keep = [
        uv_object.antnums_to_baseline(ant1, ant2) for ant1, ant2 in sorted_pairs_to_keep
    ]

    uv_object3.select(bls=bls_nums_to_keep)
    sorted_pairs_object3 = [
        sort_bl(p) for p in zip(uv_object3.ant_1_array, uv_object3.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object3.Nants_data
    assert Nblts_selected == uv_object3.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object3.ant_1_array or ant in uv_object3.ant_2_array
    for ant in np.unique(
        uv_object3.ant_1_array.tolist() + uv_object3.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object3
    for pair in sorted_pairs_object3:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific baselines using pyuvdata.",
        uv_object3.history,
    )

    # check select with polarizations
    first_ants = [6, 2, 7, 2, 21, 27, 8]
    second_ants = [0, 20, 8, 1, 2, 3, 22]
    pols = ["RR", "RR", "RR", "RR", "RR", "RR", "RR"]
    new_unique_ants = np.unique(first_ants + second_ants)
    bls_to_keep = list(zip(first_ants, second_ants, pols))
    sorted_bls_to_keep = [sort_bl(p) for p in bls_to_keep]

    blts_select = [
        sort_bl((a1, a2, "RR")) in sorted_bls_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    Nblts_selected = np.sum(blts_select)

    uv_object2 = uv_object.copy()
    uv_object2.select(bls=bls_to_keep)
    sorted_pairs_object2 = [
        sort_bl(p) + ("RR",)
        for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for bl in sorted_bls_to_keep:
        assert bl in sorted_pairs_object2
    for bl in sorted_pairs_object2:
        assert bl in sorted_bls_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific baselines, polarizations using pyuvdata.",
        uv_object2.history,
    )

    # check that you can use numpy integers with out errors:
    first_ants = list(map(np.int32, [6, 2, 7, 2, 21, 27, 8]))
    second_ants = list(map(np.int32, [0, 20, 8, 1, 2, 3, 22]))
    ant_pairs_to_keep = list(zip(first_ants, second_ants))

    uv_object2 = uv_object.select(bls=ant_pairs_to_keep, inplace=False)
    sorted_pairs_object2 = [
        sort_bl(p) for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]

    assert len(new_unique_ants) == uv_object2.Nants_data
    assert Nblts_selected == uv_object2.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object2.ant_1_array or ant in uv_object2.ant_2_array
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object2
    for pair in sorted_pairs_object2:
        assert pair in sorted_pairs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific baselines using pyuvdata.",
        uv_object2.history,
    )

    # check that you can specify a single pair without errors
    uv_object2.select(bls=(0, 6))
    sorted_pairs_object2 = [
        sort_bl(p) for p in zip(uv_object2.ant_1_array, uv_object2.ant_2_array)
    ]
    assert list(set(sorted_pairs_object2)) == [(0, 6)]

    # check for errors associated with antenna pairs not included in data and bad inputs
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=list(zip(first_ants, second_ants)) + [0, 6])
    assert str(cm.value).startswith("bls must be a list of tuples of antenna numbers")

    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=[(uv_object.antenna_names[0], uv_object.antenna_names[1])])
    assert str(cm.value).startswith("bls must be a list of tuples of antenna numbers")

    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(5, 1))
    assert str(cm.value).startswith(
        "Antenna number 5 is not present in the " "ant_1_array or ant_2_array"
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(0, 5))
    assert str(cm.value).startswith(
        "Antenna number 5 is not present in the " "ant_1_array or ant_2_array"
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(27, 27))
    assert str(cm.value).startswith("Antenna pair (27, 27) does not have any data")
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(6, 0, "RR"), polarizations="RR")
    assert str(cm.value).startswith(
        "Cannot provide length-3 tuples and also " "specify polarizations."
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=(6, 0, 8))
    assert str(cm.value).startswith(
        "The third element in each bl must be a " "polarization string"
    )
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=[])
    assert str(cm.value).startswith("bls must be a list of tuples of antenna numbers")
    with pytest.raises(ValueError) as cm:
        uv_object.select(bls=[100])
    assert str(cm.value).startswith("Baseline number 100 is not present in the")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_times(casa_uvfits, future_shapes):
    uv_object = casa_uvfits

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history
    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 3, 5, 6, 7, 10, 14]]

    Nblts_selected = np.sum([t in times_to_keep for t in uv_object.time_array])

    uv_object2 = uv_object.copy()
    uv_object2.select(times=times_to_keep)

    assert len(times_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific times using pyuvdata.",
        uv_object2.history,
    )
    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(times=times_to_keep[np.newaxis, :])

    assert len(times_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific times using pyuvdata.",
        uv_object2.history,
    )

    # check for errors associated with times not included in data
    pytest.raises(
        ValueError,
        uv_object.select,
        times=[np.min(unique_times) - uv_object.integration_time[0]],
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_time_range(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    times_to_keep = unique_times[
        np.nonzero((unique_times <= time_range[1]) & (unique_times >= time_range[0]))
    ]

    Nblts_selected = np.nonzero(
        (uv_object.time_array <= time_range[1])
        & (uv_object.time_array >= time_range[0])
    )[0].size

    uv_object2 = uv_object.copy()
    uv_object2.select(time_range=time_range)

    assert times_to_keep.size == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for t in times_to_keep:
        assert t in uv_object2.time_array
    for t in np.unique(uv_object2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific times using pyuvdata.",
        uv_object2.history,
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_lsts(casa_uvfits, future_shapes):
    uv_object = casa_uvfits

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history
    unique_lsts = np.unique(uv_object.lst_array)
    lsts_to_keep = unique_lsts[[0, 3, 5, 6, 7, 10, 14]]

    Nblts_selected = np.sum([lst in lsts_to_keep for lst in uv_object.lst_array])

    uv_object2 = uv_object.copy()
    uv_object2.select(lsts=lsts_to_keep)

    assert len(lsts_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )
    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(lsts=lsts_to_keep[np.newaxis, :])

    assert len(lsts_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )

    return


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_lsts_multi_day(casa_uvfits, future_shapes):
    uv_object = casa_uvfits
    # check that times come from a single JD
    assert len(np.unique(np.asarray(uv_object.time_array, dtype=np.int_))) == 1
    # artificially make a "double object" with times on 2 different days
    # the addition of 0.9973 days is cleverly chosen so LSTs will roughly line up
    uv_object2 = uv_object.copy()
    uv_object2.time_array += 0.9973
    assert len(np.unique(np.asarray(uv_object2.time_array, dtype=np.int_))) == 1
    uv_object2.set_lsts_from_time_array()
    uv_object += uv_object2
    # check we have times from 2 days
    assert len(np.unique(np.asarray(uv_object.time_array, dtype=np.int_))) == 2

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history
    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    lst_range = [np.min(unique_lsts), mean_lst]
    lsts_to_keep = unique_lsts[
        np.nonzero((unique_lsts <= lst_range[1]) & (unique_lsts >= lst_range[0]))
    ]

    Nblts_selected = np.nonzero(
        (uv_object.lst_array <= lst_range[1]) & (uv_object.lst_array >= lst_range[0])
    )[0].size

    uv_object2 = uv_object.copy()
    uv_object2.select(lst_range=lst_range)

    assert lsts_to_keep.size == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep
    unique_jds = np.unique(np.asarray(uv_object2.time_array, dtype=np.int_))
    assert len(unique_jds) == 2

    assert uvutils._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )

    return


def test_select_lsts_out_of_range_error(casa_uvfits):
    uv_object = casa_uvfits
    unique_lsts = np.unique(uv_object.lst_array)
    target_lst = np.min(unique_lsts) - 0.1
    with pytest.raises(ValueError) as cm:
        uv_object.select(lsts=[target_lst])
    assert str(cm.value).startswith(f"LST {target_lst} is not present in the lst_array")

    return


def test_select_lsts_too_big(casa_uvfits):
    uv_object = casa_uvfits
    # replace one LST with bogus value larger than 2*pi; otherwise we'll get an
    # error that the value isn't actually in the LST array
    lst0 = uv_object.lst_array[0]
    uv_object.lst_array = np.where(
        uv_object.lst_array == lst0, 7.0, uv_object.lst_array
    )
    unique_lsts = np.unique(uv_object.lst_array)
    lsts_to_keep = unique_lsts[[0, 3, 5, 6, 7, 10, 14]]
    assert 7.0 in lsts_to_keep

    Nblts_selected = np.sum([lst in lsts_to_keep for lst in uv_object.lst_array])

    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "The lsts parameter contained a value greater than 2*pi",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_object2.select(lsts=lsts_to_keep)

    assert len(lsts_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_lst_range(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    lst_range = [np.min(unique_lsts), mean_lst]
    lsts_to_keep = unique_lsts[
        np.nonzero((unique_lsts <= lst_range[1]) & (unique_lsts >= lst_range[0]))
    ]

    Nblts_selected = np.nonzero(
        (uv_object.lst_array <= lst_range[1]) & (uv_object.lst_array >= lst_range[0])
    )[0].size

    uv_object2 = uv_object.copy()
    uv_object2.select(lst_range=lst_range)

    assert lsts_to_keep.size == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )

    return


def test_select_lst_range_too_big(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    lst_range = [mean_lst, np.max(unique_lsts)]
    lsts_to_keep = unique_lsts[
        np.nonzero((unique_lsts <= lst_range[1]) & (unique_lsts >= lst_range[0]))
    ]

    Nblts_selected = np.nonzero(
        (uv_object.lst_array <= lst_range[1]) & (uv_object.lst_array >= lst_range[0])
    )[0].size

    # make max value larger than 2*pi
    lst_range[1] = 7.0
    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "The lst_range contained a value greater than 2*pi",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_object2.select(lst_range=lst_range)

    assert lsts_to_keep.size == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_lst_range_wrap_around(casa_uvfits):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    min_lst = np.min(unique_lsts)
    max_lst = np.max(unique_lsts)
    lst_range = [max_lst + 0.1, mean_lst]
    lsts_to_keep = unique_lsts[
        np.nonzero((unique_lsts <= mean_lst) & (unique_lsts >= min_lst))
    ]

    Nblts_selected = np.nonzero(
        (uv_object.lst_array <= mean_lst) & (uv_object.lst_array >= min_lst)
    )[0].size

    uv_object2 = uv_object.copy()
    uv_object2.select(lst_range=lst_range)

    assert lsts_to_keep.size == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )

    return


def test_select_time_range_no_data(casa_uvfits):
    """Check for error associated with times not included in data."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    with pytest.raises(ValueError) as cm:
        uv_object.select(
            time_range=[
                np.min(unique_times) - uv_object.integration_time[0] * 2,
                np.min(unique_times) - uv_object.integration_time[0],
            ]
        )
    assert str(cm.value).startswith("No elements in time range")


def test_select_lst_range_no_data(casa_uvfits):
    """Check for error associated with LSTS not included in data."""
    uv_object = casa_uvfits
    unique_lsts = np.unique(uv_object.lst_array)
    with pytest.raises(ValueError) as cm:
        uv_object.select(
            lst_range=[np.min(unique_lsts) - 0.2, np.min(unique_lsts) - 0.1]
        )
    assert str(cm.value).startswith("No elements in LST range")


def test_select_time_and_time_range(casa_uvfits):
    """Check for error setting times and time_range."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    times_to_keep = unique_times[[0, 3, 5, 6, 7, 10, 14]]
    with pytest.raises(ValueError) as cm:
        uv_object.select(time_range=time_range, times=times_to_keep)
    assert str(cm.value).startswith(
        "Only one of [times, time_range, lsts, lst_range] may be specified"
    )


def test_select_time_range_one_elem(casa_uvfits):
    """Check for error if time_range not length 2."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    with pytest.raises(ValueError) as cm:
        uv_object.select(time_range=time_range[0])
    assert str(cm.value).startswith("time_range must be length 2")


def test_select_lst_range_one_elem(casa_uvfits):
    """Check for error if time_range not length 2."""
    uv_object = casa_uvfits
    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    lst_range = [np.min(unique_lsts), mean_lst]
    with pytest.raises(ValueError) as cm:
        uv_object.select(lst_range=lst_range[0])
    assert str(cm.value).startswith("lst_range must be length 2")

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_frequencies_writeerrors(casa_uvfits, future_shapes, tmp_path):
    uv_object = casa_uvfits

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history

    if future_shapes:
        freqs_to_keep = uv_object.freq_array[np.arange(12, 22)]
    else:
        freqs_to_keep = uv_object.freq_array[0, np.arange(12, 22)]

    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep[np.newaxis, :])

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that selecting one frequency works
    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep[0])
    assert 1 == uv_object2.Nfreqs
    assert freqs_to_keep[0] in uv_object2.freq_array
    for f in uv_object2.freq_array:
        assert f in [freqs_to_keep[0]]

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check for errors associated with frequencies not included in data
    with pytest.raises(ValueError, match="Frequency "):
        uv_object.select(
            frequencies=[np.max(uv_object.freq_array) + uv_object.channel_width],
        )
    write_file_miriad = str(tmp_path / "select_test")
    write_file_uvfits = str(tmp_path / "select_test.uvfits")

    # check for warnings and errors associated with unevenly spaced or
    # non-contiguous frequencies
    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        if future_shapes:
            uv_object2.select(frequencies=uv_object2.freq_array[[0, 5, 6]])
        else:
            uv_object2.select(frequencies=uv_object2.freq_array[0, [0, 5, 6]])

    with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
        uv_object2.write_uvfits(write_file_uvfits)

    try:
        import pyuvdata._miriad

        with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
            uv_object2.write_miriad(write_file_miriad)
    except ImportError:
        pass

    uv_object2 = uv_object.copy()
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected frequencies are not contiguous",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        if future_shapes:
            uv_object2.select(frequencies=uv_object2.freq_array[[0, 2, 4]])
        else:
            uv_object2.select(frequencies=uv_object2.freq_array[0, [0, 2, 4]])

    with pytest.raises(
        ValueError,
        match="The frequencies are separated by more than their channel width",
    ):
        uv_object2.write_uvfits(write_file_uvfits)

    try:
        import pyuvdata._miriad  # noqa

        with pytest.raises(
            ValueError,
            match="The frequencies are separated by more than their channel width",
        ):
            uv_object2.write_miriad(write_file_miriad)
    except ImportError:
        pass


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_freq_chans(casa_uvfits, future_shapes):
    uv_object = casa_uvfits

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history
    chans_to_keep = np.arange(12, 22)

    uv_object2 = uv_object.copy()
    uv_object2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == uv_object2.Nfreqs
    for chan in chans_to_keep:
        if future_shapes:
            assert uv_object.freq_array[chan] in uv_object2.freq_array
        else:
            assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        if future_shapes:
            assert f in uv_object.freq_array[chans_to_keep]
        else:
            assert f in uv_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that it also works with higher dimension array
    uv_object2 = uv_object.copy()
    uv_object2.select(freq_chans=chans_to_keep[np.newaxis, :])

    assert len(chans_to_keep) == uv_object2.Nfreqs
    for chan in chans_to_keep:
        if future_shapes:
            assert uv_object.freq_array[chan] in uv_object2.freq_array
        else:
            assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        if future_shapes:
            assert f in uv_object.freq_array[chans_to_keep]
        else:
            assert f in uv_object.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # Test selecting both channels and frequencies
    if future_shapes:
        freqs_to_keep = uv_object.freq_array[np.arange(20, 30)]  # Overlaps with chans
    else:
        freqs_to_keep = uv_object.freq_array[
            0, np.arange(20, 30)
        ]  # Overlaps with chans
    all_chans_to_keep = np.arange(12, 30)

    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == uv_object2.Nfreqs
    for chan in all_chans_to_keep:
        if future_shapes:
            assert uv_object.freq_array[chan] in uv_object2.freq_array
        else:
            assert uv_object.freq_array[0, chan] in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        if future_shapes:
            assert f in uv_object.freq_array[all_chans_to_keep]
        else:
            assert f in uv_object.freq_array[0, all_chans_to_keep]


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    "pols_to_keep", ([-5, -6], ["xx", "yy"], ["nn", "ee"], [[-5, -6]])
)
def test_select_polarizations(hera_uvh5, future_shapes, pols_to_keep):
    uv_object = hera_uvh5

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history

    uv_object2 = uv_object.copy()
    uv_object2.select(polarizations=pols_to_keep)

    if isinstance(pols_to_keep[0], list):
        pols_to_keep = pols_to_keep[0]

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        if isinstance(p, int):
            assert p in uv_object2.polarization_array
        else:
            assert (
                uvutils.polstr2num(p, x_orientation=uv_object2.x_orientation)
                in uv_object2.polarization_array
            )
    for p in np.unique(uv_object2.polarization_array):
        if isinstance(pols_to_keep[0], int):
            assert p in pols_to_keep
        else:
            assert p in uvutils.polstr2num(
                pols_to_keep, x_orientation=uv_object2.x_orientation
            )

    assert uvutils._check_histories(
        old_history + "  Downselected to specific polarizations using pyuvdata.",
        uv_object2.history,
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_polarizations_errors(casa_uvfits, tmp_path):
    uv_object = casa_uvfits
    uv_object2 = uv_object.copy()
    uv_object2.select(polarizations=[-1, -2])

    # check for errors associated with polarizations not included in data
    with pytest.raises(
        ValueError, match="Polarization -3 is not present in the polarization_array"
    ):
        uv_object2.select(polarizations=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    with uvtest.check_warnings(
        UserWarning,
        [
            "Selected polarization values are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_object.select(polarizations=uv_object.polarization_array[[0, 1, 3]])
    write_file_uvfits = str(tmp_path / "select_test.uvfits")
    with pytest.raises(
        ValueError, match="The polarization values are not evenly spaced"
    ):
        uv_object.write_uvfits(write_file_uvfits)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select(casa_uvfits, future_shapes):
    # now test selecting along all axes at once
    uv_object = casa_uvfits

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history
    # fmt: off
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
    # fmt: on
    ants_to_keep = np.array([11, 6, 20, 26, 2, 27, 7, 14])

    ant_pairs_to_keep = [(2, 11), (20, 26), (6, 7), (3, 27), (14, 6)]
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    if future_shapes:
        freqs_to_keep = uv_object.freq_array[np.arange(31, 39)]
    else:
        freqs_to_keep = uv_object.freq_array[0, np.arange(31, 39)]

    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 2, 6, 8, 10, 13, 14]]

    pols_to_keep = [-1, -3]

    # Independently count blts that should be selected
    blts_blt_select = [i in blt_inds for i in np.arange(uv_object.Nblts)]
    blts_ant_select = [
        (a1 in ants_to_keep) & (a2 in ants_to_keep)
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    blts_pair_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    blts_time_select = [t in times_to_keep for t in uv_object.time_array]
    Nblts_select = np.sum(
        [
            bi & (ai & pi) & ti
            for (bi, ai, pi, ti) in zip(
                blts_blt_select, blts_ant_select, blts_pair_select, blts_time_select
            )
        ]
    )

    uv_object2 = uv_object.copy()
    uv_object2.select(
        blt_inds=blt_inds,
        antenna_nums=ants_to_keep,
        bls=ant_pairs_to_keep,
        frequencies=freqs_to_keep,
        times=times_to_keep,
        polarizations=pols_to_keep,
    )

    assert Nblts_select == uv_object2.Nblts
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
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

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific baseline-times, antennas, "
        "baselines, times, frequencies, "
        "polarizations using pyuvdata.",
        uv_object2.history,
    )

    # test that a ValueError is raised if the selection eliminates all blts
    pytest.raises(ValueError, uv_object.select, times=unique_times[0], antenna_nums=1)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_with_lst(casa_uvfits, future_shapes):
    # now test selecting along all axes at once, but with LST instead of times
    uv_object = casa_uvfits

    if future_shapes:
        uv_object.use_future_array_shapes()

    old_history = uv_object.history
    # fmt: off
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
    # fmt: on
    ants_to_keep = np.array([11, 6, 20, 26, 2, 27, 7, 14])

    ant_pairs_to_keep = [(2, 11), (20, 26), (6, 7), (3, 27), (14, 6)]
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    if future_shapes:
        freqs_to_keep = uv_object.freq_array[np.arange(31, 39)]
    else:
        freqs_to_keep = uv_object.freq_array[0, np.arange(31, 39)]

    unique_lsts = np.unique(uv_object.lst_array)
    lsts_to_keep = unique_lsts[[0, 2, 6, 8, 10, 13, 14]]

    pols_to_keep = [-1, -3]

    # Independently count blts that should be selected
    blts_blt_select = [i in blt_inds for i in np.arange(uv_object.Nblts)]
    blts_ant_select = [
        (a1 in ants_to_keep) & (a2 in ants_to_keep)
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    blts_pair_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array)
    ]
    blts_lst_select = [lst in lsts_to_keep for lst in uv_object.lst_array]
    Nblts_select = np.sum(
        [
            bi & (ai & pi) & li
            for (bi, ai, pi, li) in zip(
                blts_blt_select, blts_ant_select, blts_pair_select, blts_lst_select
            )
        ]
    )

    uv_object2 = uv_object.copy()
    uv_object2.select(
        blt_inds=blt_inds,
        antenna_nums=ants_to_keep,
        bls=ant_pairs_to_keep,
        frequencies=freqs_to_keep,
        lsts=lsts_to_keep,
        polarizations=pols_to_keep,
    )

    assert Nblts_select == uv_object2.Nblts
    for ant in np.unique(
        uv_object2.ant_1_array.tolist() + uv_object2.ant_2_array.tolist()
    ):
        assert ant in ants_to_keep

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    assert len(pols_to_keep) == uv_object2.Npols
    for p in pols_to_keep:
        assert p in uv_object2.polarization_array
    for p in np.unique(uv_object2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific baseline-times, antennas, "
        "baselines, lsts, frequencies, "
        "polarizations using pyuvdata.",
        uv_object2.history,
    )

    # test that a ValueError is raised if the selection eliminates all blts
    pytest.raises(ValueError, uv_object.select, lsts=unique_lsts[0], antenna_nums=1)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_not_inplace(casa_uvfits):
    # Test non-inplace select
    uv_object = casa_uvfits
    old_history = uv_object.history
    uv1 = uv_object.select(freq_chans=np.arange(32), inplace=False)
    uv1 += uv_object.select(freq_chans=np.arange(32, 64), inplace=False)
    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = old_history
    assert uv1 == uv_object


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("metadata_only", [True, False])
@pytest.mark.parametrize("future_shapes", [True, False])
def test_conjugate_bls(casa_uvfits, metadata_only, future_shapes):
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")

    if not metadata_only:
        uv1 = casa_uvfits
    else:
        uv1 = UVData()
        uv1.read_uvfits(testfile, read_data=False)
    if metadata_only:
        assert uv1.metadata_only

    if future_shapes:
        uv1.use_future_array_shapes()

    # file comes in with ant1<ant2
    assert np.min(uv1.ant_2_array - uv1.ant_1_array) >= 0

    # check everything swapped & conjugated when go to ant2<ant1
    uv2 = uv1.copy()
    uv2.conjugate_bls(convention="ant2<ant1")
    assert np.min(uv2.ant_1_array - uv2.ant_2_array) >= 0

    assert np.allclose(uv1.ant_1_array, uv2.ant_2_array)
    assert np.allclose(uv1.ant_2_array, uv2.ant_1_array)
    assert np.allclose(
        uv1.uvw_array,
        -1 * uv2.uvw_array,
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )

    if not metadata_only:
        # complicated because of the polarization swaps
        # polarization_array = [-1 -2 -3 -4]
        if future_shapes:
            assert np.allclose(
                uv1.data_array[:, :, :2],
                np.conj(uv2.data_array[:, :, :2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[:, :, 2],
                np.conj(uv2.data_array[:, :, 3]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[:, :, 3],
                np.conj(uv2.data_array[:, :, 2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
        else:
            assert np.allclose(
                uv1.data_array[:, :, :, :2],
                np.conj(uv2.data_array[:, :, :, :2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[:, :, :, 2],
                np.conj(uv2.data_array[:, :, :, 3]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[:, :, :, 3],
                np.conj(uv2.data_array[:, :, :, 2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

    # check everything returned to original values with original convention
    uv2.conjugate_bls(convention="ant1<ant2")
    assert uv1 == uv2

    # conjugate a particular set of blts
    blts_to_conjugate = np.arange(uv2.Nblts // 2)
    blts_not_conjugated = np.arange(uv2.Nblts // 2, uv2.Nblts)
    uv2.conjugate_bls(convention=blts_to_conjugate)

    assert np.allclose(
        uv1.ant_1_array[blts_to_conjugate], uv2.ant_2_array[blts_to_conjugate]
    )
    assert np.allclose(
        uv1.ant_2_array[blts_to_conjugate], uv2.ant_1_array[blts_to_conjugate]
    )
    assert np.allclose(
        uv1.ant_1_array[blts_not_conjugated], uv2.ant_1_array[blts_not_conjugated]
    )
    assert np.allclose(
        uv1.ant_2_array[blts_not_conjugated], uv2.ant_2_array[blts_not_conjugated]
    )

    assert np.allclose(
        uv1.uvw_array[blts_to_conjugate],
        -1 * uv2.uvw_array[blts_to_conjugate],
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )
    assert np.allclose(
        uv1.uvw_array[blts_not_conjugated],
        uv2.uvw_array[blts_not_conjugated],
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )
    if not metadata_only:
        # complicated because of the polarization swaps
        # polarization_array = [-1 -2 -3 -4]
        if future_shapes:
            assert np.allclose(
                uv1.data_array[blts_to_conjugate, :, :2],
                np.conj(uv2.data_array[blts_to_conjugate, :, :2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
            assert np.allclose(
                uv1.data_array[blts_not_conjugated, :, :2],
                uv2.data_array[blts_not_conjugated, :, :2],
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[blts_to_conjugate, :, 2],
                np.conj(uv2.data_array[blts_to_conjugate, :, 3]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
            assert np.allclose(
                uv1.data_array[blts_not_conjugated, :, 2],
                uv2.data_array[blts_not_conjugated, :, 2],
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[blts_to_conjugate, :, 3],
                np.conj(uv2.data_array[blts_to_conjugate, :, 2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
            assert np.allclose(
                uv1.data_array[blts_not_conjugated, :, 3],
                uv2.data_array[blts_not_conjugated, :, 3],
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
        else:
            assert np.allclose(
                uv1.data_array[blts_to_conjugate, :, :, :2],
                np.conj(uv2.data_array[blts_to_conjugate, :, :, :2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
            assert np.allclose(
                uv1.data_array[blts_not_conjugated, :, :, :2],
                uv2.data_array[blts_not_conjugated, :, :, :2],
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[blts_to_conjugate, :, :, 2],
                np.conj(uv2.data_array[blts_to_conjugate, :, :, 3]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
            assert np.allclose(
                uv1.data_array[blts_not_conjugated, :, :, 2],
                uv2.data_array[blts_not_conjugated, :, :, 2],
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

            assert np.allclose(
                uv1.data_array[blts_to_conjugate, :, :, 3],
                np.conj(uv2.data_array[blts_to_conjugate, :, :, 2]),
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )
            assert np.allclose(
                uv1.data_array[blts_not_conjugated, :, :, 3],
                uv2.data_array[blts_not_conjugated, :, :, 3],
                rtol=uv1._data_array.tols[0],
                atol=uv1._data_array.tols[1],
            )

    # check uv half plane conventions
    uv2.conjugate_bls(convention="u<0", use_enu=False)
    assert np.max(uv2.uvw_array[:, 0]) <= 0

    uv2.conjugate_bls(convention="u>0", use_enu=False)
    assert np.min(uv2.uvw_array[:, 0]) >= 0

    uv2.conjugate_bls(convention="v<0", use_enu=False)
    assert np.max(uv2.uvw_array[:, 1]) <= 0

    uv2.conjugate_bls(convention="v>0", use_enu=False)
    assert np.min(uv2.uvw_array[:, 1]) >= 0

    # unphase to drift to test using ENU positions
    uv2.unphase_to_drift(use_ant_pos=True)
    uv2.conjugate_bls(convention="u<0")
    assert np.max(uv2.uvw_array[:, 0]) <= 0

    uv2.conjugate_bls(convention="u>0")
    assert np.min(uv2.uvw_array[:, 0]) >= 0

    uv2.conjugate_bls(convention="v<0")
    assert np.max(uv2.uvw_array[:, 1]) <= 0

    uv2.conjugate_bls(convention="v>0")
    assert np.min(uv2.uvw_array[:, 1]) >= 0

    # test errors
    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention="foo")
    assert str(cm.value).startswith("convention must be one of")

    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention=np.arange(5) - 1)
    assert str(cm.value).startswith("If convention is an index array")

    with pytest.raises(ValueError) as cm:
        uv2.conjugate_bls(convention=[uv2.Nblts])

    assert str(cm.value).startswith("If convention is an index array")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_reorder_pols(casa_uvfits, future_shapes):
    # Test function to fix polarization order
    uv1 = casa_uvfits

    if future_shapes:
        uv1.use_future_array_shapes()

    uv2 = uv1.copy()
    uv3 = uv1.copy()
    # reorder uv2 manually
    order = [1, 3, 2, 0]
    uv2.polarization_array = uv2.polarization_array[order]
    if future_shapes:
        uv2.data_array = uv2.data_array[:, :, order]
        uv2.nsample_array = uv2.nsample_array[:, :, order]
        uv2.flag_array = uv2.flag_array[:, :, order]
    else:
        uv2.data_array = uv2.data_array[:, :, :, order]
        uv2.nsample_array = uv2.nsample_array[:, :, :, order]
        uv2.flag_array = uv2.flag_array[:, :, :, order]
    uv1.reorder_pols(order=order)
    assert uv1 == uv2

    # Restore original order
    uv1 = uv3.copy()
    uv2.reorder_pols()
    assert uv1 == uv2

    uv1.reorder_pols(order="AIPS")
    # check that we have aips ordering
    aips_pols = np.array([-1, -2, -3, -4]).astype(int)
    assert np.all(uv1.polarization_array == aips_pols)

    uv2 = uv1.copy()
    uv2.reorder_pols(order="CASA")
    # check that we have casa ordering
    casa_pols = np.array([-1, -3, -4, -2]).astype(int)
    assert np.all(uv2.polarization_array == casa_pols)
    order = np.array([0, 2, 3, 1])
    if future_shapes:
        assert np.all(uv2.data_array == uv1.data_array[:, :, order])
        assert np.all(uv2.flag_array == uv1.flag_array[:, :, order])
    else:
        assert np.all(uv2.data_array == uv1.data_array[:, :, :, order])
        assert np.all(uv2.flag_array == uv1.flag_array[:, :, :, order])

    uv2.reorder_pols(order="AIPS")
    # check that we have aips ordering again
    assert uv1 == uv2

    # check error on unknown order
    pytest.raises(ValueError, uv2.reorder_pols, {"order": "foo"})

    # check error if order is an array of the wrong length
    with pytest.raises(ValueError) as cm:
        uv2.reorder_pols(order=[3, 2, 1])
    assert str(cm.value).startswith("If order is an index array, it must")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "order,minor_order,msg",
    [
        ["foo", None, "order must be one of"],
        [np.arange(5), None, "If order is an index array, it must"],
        [np.arange(5, dtype=np.float64), None, "If order is an index array, it must"],
        [np.arange(1360), "time", "Minor order cannot be set if order is an index"],
        ["bda", "time", "minor_order cannot be specified if order is"],
        ["baseline", "ant1", "minor_order conflicts with order"],
        ["time", "foo", "minor_order can only be one of"],
    ],
)
def test_reorder_blts_errs(casa_uvfits, order, minor_order, msg):
    """
    Verify that reorder_blts throws expected errors when supplied with bad args
    """
    print(casa_uvfits.Nblts)
    with pytest.raises(ValueError) as cm:
        casa_uvfits.reorder_blts(order=order, minor_order=minor_order)
    assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("multi_phase_center", [True, False])
def test_reorder_blts(casa_uvfits, future_shapes, multi_phase_center):
    uv1 = casa_uvfits

    if future_shapes:
        uv1.use_future_array_shapes()
    if multi_phase_center:
        uv1._set_multi_phase_center(preserve_phase_center_info=True)

    # test default reordering in detail
    uv2 = uv1.copy()
    uv2.reorder_blts()
    assert uv2.blt_order == ("time", "baseline")
    assert np.min(np.diff(uv2.time_array)) >= 0
    for this_time in np.unique(uv2.time_array):
        bls_2 = uv2.baseline_array[np.where(uv2.time_array == this_time)]
        bls_1 = uv1.baseline_array[np.where(uv2.time_array == this_time)]
        assert bls_1.shape == bls_2.shape
        assert np.min(np.diff(bls_2)) >= 0
        bl_inds = [np.where(bls_1 == bl)[0][0] for bl in bls_2]
        assert np.allclose(bls_1[bl_inds], bls_2)

        uvw_1 = uv1.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        uvw_2 = uv2.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        assert uvw_1.shape == uvw_2.shape
        assert np.allclose(uvw_1[bl_inds, :], uvw_2)

        data_1 = uv1.data_array[np.where(uv2.time_array == this_time)[0]]
        data_2 = uv2.data_array[np.where(uv2.time_array == this_time)[0]]
        assert data_1.shape == data_2.shape
        assert np.allclose(data_1[bl_inds], data_2)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("multi_phase", [True, False])
@pytest.mark.parametrize(
    "args1,args2",
    [
        [{"order": "time"}, {"order": "time", "minor_order": "time"}],
        [{"order": "time"}, {"order": "time", "minor_order": "ant1"}],
        [{"order": "time"}, {"order": "time", "minor_order": "baseline"}],
        [{}, {"order": np.arange(1360)}],  # casa_uvfits already in default order
        [{"order": "time"}, {"order": "time", "conj_convention": "ant1<ant2"}],
    ],
)
def test_reorder_blts_equiv(casa_uvfits, args1, args2, future_shapes, multi_phase):
    """
    Test that sorting orders that _should_ be equivalent actually are
    """
    uv1 = casa_uvfits
    if future_shapes:
        uv1.use_future_array_shapes()
    if multi_phase:
        uv1._set_multi_phase_center(preserve_phase_center_info=True)
    uv2 = uv1.copy()

    uv1.reorder_blts(**args1)
    uv2.reorder_blts(**args2)
    # Ignore the blt_order for now, since we check this elsewhere and we know its not
    # going to be consistent between the two different sorts
    uv1.blt_order = None
    uv2.blt_order = None
    assert uv1 == uv2

    # check that loopback works
    uv1.reorder_blts()
    uv2.reorder_blts()
    assert uv1 == uv2


@pytest.mark.parametrize(
    "order,m_order,check_tuple,check_attr",
    [
        ["time", "ant1", ("time", "ant1"), ("time_array",)],
        ["time", "ant2", ("time", "ant2"), ("time_array",)],
        ["time", "baseline", ("time", "baseline"), ("time_array",)],
        ["baseline", None, ("baseline", "time"), ("baseline_array",)],
        ["ant1", None, ("ant1", "ant2"), ("ant_1_array",)],
        ["ant1", "time", ("ant1", "time"), ("ant_1_array",)],
        ["ant1", "baseline", ("ant1", "baseline"), ("ant_1_array",)],
        ["ant2", None, ("ant2", "ant1"), ("ant_2_array",)],
        ["ant2", "time", ("ant2", "time"), ("ant_2_array",)],
        ["ant2", "baseline", ("ant2", "baseline"), ("ant_2_array",)],
        ["bda", None, ("bda",), ("integration_time", "baseline_array")],
        # Below is the ant1 order in the hera_uvh5 file for one integration, for
        # testing the case of providing an index array
        [
            np.argsort(np.tile([0, 0, 2, 0, 2, 1, 0, 2, 1, 11], 20)),
            None,
            None,
            ("ant_1_array",),
        ],
    ],
)
def test_reorder_blts_sort_order(hera_uvh5, order, m_order, check_tuple, check_attr):
    hera_uvh5.reorder_blts(order=order, minor_order=m_order)
    assert hera_uvh5.blt_order == check_tuple
    for item in check_attr:
        assert np.all(np.diff(getattr(hera_uvh5, item)) >= 0)


@pytest.mark.parametrize(
    "arg_dict,msg",
    [
        [{"spord": [1]}, "Index array for spw_order must contain all indicies for"],
        [{"spord": "karto"}, "spw_order can only be one of 'number', '-number',"],
        [{"chord": [1]}, "Index array for channel_order must contain all indicies"],
        [{"chord": "karto"}, "channel_order can only be one of 'freq' or '-freq'"],
    ],
)
def test_reorder_freqs_errs(sma_mir, arg_dict, msg):
    """
    Verify that appropriate errors are thrown when providing bad arguments to
    reorder_freqs.
    """
    with pytest.raises(ValueError) as cm:
        sma_mir.reorder_freqs(
            spw_order=arg_dict.get("spord"), channel_order=arg_dict.get("chord"),
        )
    assert str(cm.value).startswith(msg)


@pytest.mark.parametrize(
    "arg_dict,msg",
    [
        [{}, "Not specifying either spw_order or channel_order"],
        [
            {"spword": "number", "selspw": [1, 3]},
            [
                "The spw_order argument is ignored when providing an argument for",
                "Specifying select_spw without providing channel_order causes",
            ],
        ],
        [
            {"spword": "number", "selspw": [1, 4], "chord": np.arange(131072)},
            [
                "The select_spw argument is ignored when providing an array_like",
                "The spw_order argument is ignored when providing an array_like",
            ],
        ],
    ],
)
def test_reorder_freqs_warnings(sma_mir, sma_mir_main, arg_dict, msg):
    """
    Verify that reorder_freqs throws appropriate warnings, all of which effectively
    warn of no-ops (so verify that the UVData object is unchanged).
    """
    with uvtest.check_warnings(UserWarning, msg):
        sma_mir.reorder_freqs(
            select_spw=arg_dict.get("selspw"),
            spw_order=arg_dict.get("spword"),
            channel_order=arg_dict.get("chord"),
        )

    assert sma_mir == sma_mir_main


@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    "sel_spw,spord,chord",
    [
        [[1, [1]], [None] * 2, ["freq"] * 2],
        [[None] * 2, ["number", "freq"], ["freq"] * 2],
        [[None] * 2, ["-number", "-freq"], ["-freq"] * 2],
    ],
)
def test_reorder_freqs_equal(sma_mir, future_shapes, sel_spw, spord, chord):
    if future_shapes:
        sma_mir.use_future_array_shapes()

    # Create a dummy copy that we can muck with at will
    sma_mir_copy = sma_mir.copy()

    # Make sure that arrays and ints work for select_spw
    sma_mir.reorder_freqs(
        select_spw=sel_spw[0], spw_order=spord[0], channel_order=chord[0],
    )
    sma_mir_copy.reorder_freqs(
        select_spw=sel_spw[1], spw_order=spord[1], channel_order=chord[1],
    )
    assert sma_mir == sma_mir_copy


@pytest.mark.parametrize("future_shapes", [True, False])
def test_reorder_freqs_flipped(sma_mir, future_shapes):
    """
    Test that when sorting the data in ways that _should_ flip the frequency
    axis, that it actually does so.
    """
    if future_shapes:
        sma_mir.use_future_array_shapes()

    # Make a copy
    sma_mir_copy = sma_mir.copy()

    # Order the data in opposite orientations -- note that for SMA, SPW numbers are
    # ordered in frequency order, so spw_order="number" _should_ be the same as
    # spw_order="freq"
    sma_mir.reorder_freqs(spw_order="number", channel_order="freq")
    sma_mir_copy.reorder_freqs(spw_order="-freq", channel_order="-freq")

    # This is kind of a sneaky test, but basically, there are 8 SPWs in the MIR
    # data file, with 6 pairs of windows partially overlapping. If all is correct,
    # then across the freq axis we should only see 6 instances of the freq
    # incrementing downwards/upwards for the two data sets.
    assert np.sum(np.diff(sma_mir.freq_array) < 0) == 6
    assert np.sum(np.diff(sma_mir_copy.freq_array) > 0) == 6

    # Check that the ordering of the spw_array makes sense
    assert np.all(sma_mir.spw_array == np.sort(sma_mir.spw_array))
    assert np.all(sma_mir_copy.spw_array == np.flip(np.sort(sma_mir.spw_array)))

    # Finally, lets make sure that the major freq arrays are flipped when sorting in
    # opposite directions
    assert np.all(np.flip(sma_mir.freq_array, axis=-1) == sma_mir_copy.freq_array)
    assert np.all(np.flip(sma_mir.data_array, axis=-2) == sma_mir_copy.data_array)


def test_reorder_freqs_eq_coeffs(sma_mir):
    # No test datasets to examine this with, so let's generate some mock data,
    # with a pre-determined order that we can flip
    sma_mir.reorder_freqs(spw_order="-number", channel_order="-freq")
    sma_mir.eq_coeffs = np.tile(
        np.arange(sma_mir.Nfreqs, dtype=float), (sma_mir.Nants_telescope, 1)
    )
    sma_mir.reorder_freqs(spw_order="number", channel_order="freq")
    assert np.all(np.diff(sma_mir.eq_coeffs, axis=1) == -1)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_sum_vis(casa_uvfits, future_shapes):
    # check sum_vis
    uv_full = casa_uvfits

    if future_shapes:
        uv_full.use_future_array_shapes()

    uv_half = uv_full.copy()
    uv_half.data_array = uv_full.data_array / 2
    uv_half_mod = uv_half.copy()
    uv_half_mod.history += " testing the history. "
    uv_summed = uv_half.sum_vis(uv_half_mod)

    assert np.array_equal(uv_summed.data_array, uv_full.data_array)
    assert uvutils._check_histories(
        uv_half.history + " Visibilities summed using pyuvdata. Unique part of second "
        "object history follows.  testing the history.",
        uv_summed.history,
    )
    # add a test for full coverage of _combine_history_addition function
    assert (
        uvutils._combine_history_addition(
            uv_half.history
            + " Visibilities summed using pyuvdata. Unique part of second "
            "object history follows.  testing the history.",
            uv_summed.history,
        )
        is None
    )

    uv_summed = uv_half.sum_vis(uv_half_mod, verbose_history=True)

    assert np.array_equal(uv_summed.data_array, uv_full.data_array)
    assert uvutils._check_histories(
        uv_half.history + " Visibilities summed using pyuvdata. Second object history "
        "follows. " + uv_half_mod.history,
        uv_summed.history,
    )

    # check diff_vis
    uv_diffed = uv_full.diff_vis(uv_half)

    assert np.array_equal(uv_diffed.data_array, uv_half.data_array)
    assert uvutils._check_histories(
        uv_full.history + " Visibilities differenced using pyuvdata.",
        uv_diffed.history,
    )

    # check in place
    uv_summed.diff_vis(uv_half, inplace=True)
    assert np.array_equal(uv_summed.data_array, uv_half.data_array)

    # check object_name merge
    uv_zenith = uv_full.copy()
    uv_zenith.object_name = "zenith"
    uv_merged = uv_zenith.sum_vis(uv_full)
    assert uv_merged.object_name == "zenith-J1008+0730"

    # check extra_keywords handling
    uv_keys = uv_full.copy()
    uv_keys.extra_keywords["test_key"] = "test_value"
    uv_keys.extra_keywords["SPECSYS"] = "altered_value"
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Keyword SPECSYS in _extra_keywords is different in the two objects. "
            "Taking the first object's entry.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv_merged_keys = uv_keys.sum_vis(uv_full)
    assert uv_merged_keys.extra_keywords["test_key"] == "test_value"
    assert uv_merged_keys.extra_keywords["SPECSYS"] == "altered_value"

    # check override_params
    uv_overrides = uv_full.copy()
    uv_overrides.instrument = "test_telescope"
    uv_overrides.telescope_location = [
        -1601183.15377712,
        -5042003.74810822,
        3554841.17192104,
    ]
    uv_overrides_2 = uv_overrides.sum_vis(
        uv_full, override_params=["instrument", "telescope_location"]
    )

    assert uv_overrides_2.instrument == "test_telescope"
    assert uv_overrides_2.telescope_location == [
        -1601183.15377712,
        -5042003.74810822,
        3554841.17192104,
    ]


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "attr_to_get,attr_to_set,arg_dict,msg",
    [
        [["use_future_array_shapes"], {}, {}, "Both objects must have the same `futu"],
        [[], {}, {"override": ["fake"]}, "Provided parameter fake is not a recogniza"],
        [[], {"__class__": UVCal}, {}, "Only UVData (or subclass) objects can be"],
        [[], {"instrument": "foo"}, {"inplace": True}, "UVParameter instrument does"],
    ],
)
def test_sum_vis_errors(uv1_2_set_uvws, attr_to_get, attr_to_set, arg_dict, msg):
    uv1, uv2 = uv1_2_set_uvws
    for method in attr_to_get:
        getattr(uv2, method)()
    for attr in attr_to_set.keys():
        setattr(uv2, attr, attr_to_set[attr])

    with pytest.raises(ValueError) as cm:
        uv1.sum_vis(
            uv2,
            override_params=arg_dict.get("override"),
            inplace=arg_dict.get("inplace"),
        )
    assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_add(casa_uvfits, hera_uvh5_xx, future_shapes):
    uv_full = casa_uvfits

    if future_shapes:
        uv_full.use_future_array_shapes()

    # Add frequencies
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add frequencies - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add polarizations
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add times
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
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
    uv3.data_array = uv3.data_array[-1::-1]
    uv3.nsample_array = uv3.nsample_array[-1::-1]
    uv3.flag_array = uv3.flag_array[-1::-1]
    uv3.uvw_array = uv3.uvw_array[-1::-1, :]
    uv3.time_array = uv3.time_array[-1::-1]
    uv3.lst_array = uv3.lst_array[-1::-1]
    uv3.ant_1_array = uv3.ant_1_array[-1::-1]
    uv3.ant_2_array = uv3.ant_2_array[-1::-1]
    uv3.baseline_array = uv3.baseline_array[-1::-1]
    uv1 += uv3
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata. Combined data along "
        "baseline-time axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add multiple axes
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(
        times=times[0 : len(times) // 2], polarizations=uv1.polarization_array[0:2]
    )
    uv2.select(
        times=times[len(times) // 2 :], polarizations=uv2.polarization_array[2:4]
    )
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, polarizations using "
        "pyuvdata. Combined data along "
        "baseline-time, polarization axis "
        "using pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
    # Zero out missing data in reference object
    if future_shapes:
        uv_ref.data_array[blt_ind1, :, 2:] = 0.0
        uv_ref.nsample_array[blt_ind1, :, 2:] = 0.0
        uv_ref.flag_array[blt_ind1, :, 2:] = True
        uv_ref.data_array[blt_ind2, :, 0:2] = 0.0
        uv_ref.nsample_array[blt_ind2, :, 0:2] = 0.0
        uv_ref.flag_array[blt_ind2, :, 0:2] = True
    else:
        uv_ref.data_array[blt_ind1, :, :, 2:] = 0.0
        uv_ref.nsample_array[blt_ind1, :, :, 2:] = 0.0
        uv_ref.flag_array[blt_ind1, :, :, 2:] = True
        uv_ref.data_array[blt_ind2, :, :, 0:2] = 0.0
        uv_ref.nsample_array[blt_ind2, :, :, 0:2] = 0.0
        uv_ref.flag_array[blt_ind2, :, :, 0:2] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Another combo
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2 :], freq_chans=np.arange(32, 64))
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, frequencies using "
        "pyuvdata. Combined data along "
        "baseline-time, frequency axis using "
        "pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
    # Zero out missing data in reference object
    if future_shapes:
        uv_ref.data_array[blt_ind1, 32:, :] = 0.0
        uv_ref.nsample_array[blt_ind1, 32:, :] = 0.0
        uv_ref.flag_array[blt_ind1, 32:, :] = True
        uv_ref.data_array[blt_ind2, 0:32, :] = 0.0
        uv_ref.nsample_array[blt_ind2, 0:32, :] = 0.0
        uv_ref.flag_array[blt_ind2, 0:32, :] = True
    else:
        uv_ref.data_array[blt_ind1, :, 32:, :] = 0.0
        uv_ref.nsample_array[blt_ind1, :, 32:, :] = 0.0
        uv_ref.flag_array[blt_ind1, :, 32:, :] = True
        uv_ref.data_array[blt_ind2, :, 0:32, :] = 0.0
        uv_ref.nsample_array[blt_ind2, :, 0:32, :] = 0.0
        uv_ref.flag_array[blt_ind2, :, 0:32, :] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Add without inplace
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 = uv1 + uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__add__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are separated by more than their channel width.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[1])
    uv2.freq_array += uv2._channel_width.tols[1] / 2.0
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.__iadd__(uv2)

    # Combining histories
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv_new = uv1 + uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Unique part of next "
        "object history follows.  testing the history.",
        uv_new.history,
    )
    uv_new.history = uv_full.history
    assert uv_new == uv_full

    uv_new = uv1.__add__(uv2, verbose_history=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Next object history "
        "follows.  " + uv2.history,
        uv_new.history,
    )

    # test add of autocorr-only and crosscorr-only objects
    uv_full = hera_uvh5_xx
    bls = uv_full.get_antpairs()
    autos = [bl for bl in bls if bl[0] == bl[1]]
    cross = sorted(set(bls) - set(autos))
    uv_auto = uv_full.select(bls=autos, inplace=False)
    uv_cross = uv_full.select(bls=cross, inplace=False)
    uv1 = uv_auto + uv_cross
    assert uv1.Nbls == uv_auto.Nbls + uv_cross.Nbls
    uv2 = uv_cross + uv_auto
    assert uv2.Nbls == uv_auto.Nbls + uv_cross.Nbls


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_drift(casa_uvfits):
    uv_full = casa_uvfits
    uv_full.unphase_to_drift()

    # Add frequencies
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add times
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add multiple axes
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(
        times=times[0 : len(times) // 2], polarizations=uv1.polarization_array[0:2]
    )
    uv2.select(
        times=times[len(times) // 2 :], polarizations=uv2.polarization_array[2:4]
    )
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, polarizations using "
        "pyuvdata. Combined data along "
        "baseline-time, polarization "
        "axis using pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
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
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2 :], freq_chans=np.arange(32, 64))
    uv1 += uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times, frequencies using "
        "pyuvdata. Combined data along "
        "baseline-time, frequency "
        "axis using pyuvdata.",
        uv1.history,
    )
    blt_ind1 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[0 : len(times) // 2]
        ]
    )
    blt_ind2 = np.array(
        [
            ind
            for ind in range(uv_full.Nblts)
            if uv_full.time_array[ind] in times[len(times) // 2 :]
        ]
    )
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
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 = uv1 + uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    with uvtest.check_warnings(
        UserWarning, "Combined frequencies are not evenly spaced"
    ):
        uv1.__add__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    with uvtest.check_warnings(
        UserWarning,
        ["Combined frequencies are separated by more than their channel width"],
    ):
        uv1.__iadd__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    with uvtest.check_warnings(None, None):
        uv1.__iadd__(uv2)

    # Combining histories
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv_new = uv1 + uv2
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata.  Unique part of next "
        "object history follows.  testing the history.",
        uv_new.history,
    )
    uv_new.history = uv_full.history
    assert uv_new == uv_full

    uv_new = uv1.__add__(uv2, verbose_history=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Next object history "
        "follows." + uv2.history,
        uv_new.history,
    )


def test_flex_spw_freq_avg(sma_mir):
    """
    Test that freq averaging works correctly when called on a flex_spw data set
    (currently not implented).
    """
    with pytest.raises(NotImplementedError, match="Frequency averaging not"):
        sma_mir.frequency_average(2)


def test_check_flex_spw_contiguous(sma_mir):
    """
    Verify that check_flex_spw_contiguous works as expected (throws an error if
    windows are not contiguous, otherwise no error raised).
    """
    sma_mir._check_flex_spw_contiguous()
    sma_mir.flex_spw_id_array[0] = 1
    with pytest.raises(ValueError, match="Channels from different spectral windows"):
        sma_mir._check_flex_spw_contiguous()


@pytest.mark.parametrize(
    "chan_width,msg",
    [
        [np.arange(131072), "The frequencies are not evenly spaced"],
        [np.zeros(131072), "The frequencies are separated by more"],
    ],
)
def test_check_freq_spacing_flex_spw(sma_mir, chan_width, msg):
    """
    Verify that _check_freq_spacing works as expected with flex_spw data sets (throws
    an error if windows are not contiguous, otherwise no error raised).
    """
    sma_mir.channel_width = chan_width
    with pytest.raises(ValueError, match=msg):
        sma_mir._check_freq_spacing()


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    "add_method,screen1,screen2",
    # All of these tests are passing bool arrays as selection screens for grabbing out
    # different frequency indexes.
    [
        [  # First-half of all windows vs last-half of all windows
            ["__add__", {}],
            np.arange(8 * 16384) < 4 * 16384,
            np.arange(8 * 16384) >= 4 * 16384,
        ],
        [  # First-half of all windows vs last 3/4ths of all windows (overlap flagged)
            ["__add__", {}],
            np.arange(8 * 16384) < 4 * 16384,
            np.arange(8 * 16384) >= 2 * 16384,
        ],
        [  # Last vs first half of channels in each window, across all windows
            ["__add__", {}],
            np.mod(np.arange(8 * 16384), 16384) >= 8192,
            np.mod(np.arange(8 * 16384), 16384) < 8192,
        ],
        [  # fast-concat w/ first vs last half of all windows
            ["fast_concat", {"axis": "freq"}],
            np.arange(8 * 16384) < 4 * 16384,
            np.arange(8 * 16384) >= 4 * 16384,
        ],
        [  # First window, first half vs last half of all channels
            ["__add__", {}],
            np.arange(8 * 16384) < 8192,
            np.logical_and(np.arange(8 * 16384) >= 8192, np.arange(8 * 16384) < 16384),
        ],
        [  # First half of first spw vs full first window (overlap flagged)
            ["__add__", {}],
            np.arange(8 * 16384) < 8192,
            np.arange(8 * 16384) < 16384,
        ],
        [  # Fast concat with first window, first half vs last half of all channels
            ["fast_concat", {"axis": "freq"}],
            np.arange(8 * 16384) < 8192,
            np.logical_and(np.arange(8 * 16384) >= 8192, np.arange(8 * 16384) < 16384),
        ],
    ],
)
def test_flex_spw_add_concat(sma_mir, future_shapes, add_method, screen1, screen2):
    """
    Test add & fast concat with flexible spws using Mir file.

    Read in Mir files using flexible spectral windows, all of the same nchan
    """
    if future_shapes:
        sma_mir.use_future_array_shapes()

    uv1 = sma_mir.select(freq_chans=np.where(screen1), inplace=False)
    uv2 = sma_mir.select(freq_chans=np.where(screen2), inplace=False)

    if np.any(np.logical_and(screen1, screen2)):
        flag_screen = screen2[screen1]
        if future_shapes:
            uv1.data_array[:, flag_screen] = 0.0
            uv1.flag_array[:, flag_screen] = True
        else:
            uv1.data_array[:, :, flag_screen] = 0.0
            uv1.flag_array[:, :, flag_screen] = True

    uv_recomb = getattr(uv1, add_method[0])(uv2, **add_method[1])

    if np.any(~np.logical_or(screen1, screen2)):
        sma_mir.select(freq_chans=np.where(np.logical_or(screen1, screen2)))

    # Make sure the two datasets are in the same frequency order
    uv_recomb.reorder_freqs(spw_order=sma_mir.spw_array, channel_order="freq")
    sma_mir.reorder_freqs(spw_order=sma_mir.spw_array, channel_order="freq")

    # Check the history first
    assert uv_recomb.history.startswith(sma_mir.history)
    sma_mir.history = uv_recomb.history
    assert uv_recomb == sma_mir


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "attr_to_set,attr_to_get,msg",
    [
        [[], [], "These objects have overlapping data and cannot be combined."],
        [[["__class__", UVCal]], [], "Only UVData"],
        [[], [["use_future_array_shapes", {}]], "Both objects must have the same `fu"],
        [
            [],
            [["unphase_to_drift", {}], ["select", {"freq_chans": np.arange(32, 64)}]],
            "UVParameter phase_type does not match. Cannot combine objects.",
        ],
        [
            [["vis_units", "Jy"]],
            [["select", {"freq_chans": np.arange(32, 64)}]],
            "UVParameter vis_units does not match. Cannot combine objects.",
        ],
        [
            [["integration_time", np.zeros(1360)]],
            [["select", {"freq_chans": np.arange(32, 64)}]],
            "UVParameter integration_time does not match.",
        ],
        [
            [["channel_width", np.ones(64)], ["flex_spw_id_array", np.array([0] * 64)]],
            [["_set_flex_spw", {}], ["select", {"freq_chans": np.arange(32, 64)}]],
            "To combine these data, flex_spw must be set",
        ],
    ],
)
def test_break_add(casa_uvfits, attr_to_set, attr_to_get, msg):
    """
    Verify that the add function throws errors appropriately when trying to combine
    objects that cannot be combined.
    """
    # Test failure modes of add function
    uv1 = casa_uvfits
    uv2 = uv1.copy()
    uv1.select(freq_chans=np.arange(0, 32))

    for item in attr_to_set:
        setattr(uv2, item[0], item[1])

    for item in attr_to_get:
        getattr(uv2, item[0])(**item[1])
    with pytest.raises(ValueError) as cm:
        uv1 += uv2
    assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_error_drift_and_rephase(casa_uvfits, test_func, extra_kwargs):
    uv_full = casa_uvfits

    with pytest.raises(ValueError) as cm:
        getattr(uv_full, test_func)(
            uv_full, phase_center_radec=(0, 45), unphase_to_drift=True, **extra_kwargs
        )
    assert str(cm.value).startswith(
        "phase_center_radec cannot be set if unphase_to_drift is True."
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_this_phased_unphase_to_drift(uv_phase_time_split, test_func, extra_kwargs):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split
    func_kwargs = {
        "unphase_to_drift": True,
        "inplace": False,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(UserWarning, "Unphasing this UVData object to drift"):
        uv_out = getattr(uv_phase_1, test_func)(uv_raw_2, **func_kwargs)
    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)
    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert uv_out.phase_type == "drift"
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_other_phased_unphase_to_drift(
    uv_phase_time_split, test_func, extra_kwargs
):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split

    func_kwargs = {
        "unphase_to_drift": True,
        "inplace": False,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(UserWarning, "Unphasing other UVData object to drift"):
        uv_out = getattr(uv_raw_1, test_func)(uv_phase_2, **func_kwargs)

    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)
    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert uv_out.phase_type == "drift"
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_this_rephase_new_phase_center(
    uv_phase_time_split, test_func, extra_kwargs
):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split

    phase_center_radec = (Angle("0d").rad, Angle("-30d").rad)

    # phase each half to different spots
    uv_raw_1.phase(
        ra=0, dec=0, use_ant_pos=True,
    )
    uv_raw_2.phase(
        ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True
    )
    # phase original to phase_center_radec
    uv_raw.phase(ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True)

    func_kwargs = {
        "inplace": False,
        "phase_center_radec": phase_center_radec,
        "use_ant_pos": True,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(
        UserWarning, "Phasing this UVData object to phase_center_radec",
    ):
        uv_out = getattr(uv_raw_1, test_func)(uv_raw_2, **func_kwargs)

    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)
    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert (uv_out.phase_center_ra, uv_out.phase_center_dec) == phase_center_radec
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_other_rephase_new_phase_center(
    uv_phase_time_split, test_func, extra_kwargs
):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split

    phase_center_radec = (Angle("0d").rad, Angle("-30d").rad)

    # phase each half to different spots
    uv_raw_1.phase(
        ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True,
    )
    uv_raw_2.phase(
        ra=0, dec=0, use_ant_pos=True,
    )
    # phase original to phase_center_radec
    uv_raw.phase(
        ra=phase_center_radec[0], dec=phase_center_radec[1], use_ant_pos=True,
    )

    func_kwargs = {
        "inplace": False,
        "phase_center_radec": phase_center_radec,
        "use_ant_pos": True,
    }
    func_kwargs.update(extra_kwargs)
    with uvtest.check_warnings(
        UserWarning, "Phasing other UVData object to phase_center_radec"
    ):
        uv_out = getattr(uv_raw_1, test_func)(uv_raw_2, **func_kwargs)

    # the histories will be different here
    # but everything else should match.
    uv_out.history = copy.deepcopy(uv_raw.history)

    # ensure baseline time order is the same
    # because fast_concat will not order for us
    uv_out.reorder_blts(order="time", minor_order="baseline")
    assert uv_out.phase_type == "phased"
    assert (uv_out.phase_center_ra, uv_out.phase_center_dec) == phase_center_radec
    assert uv_out == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "test_func,extra_kwargs", [("__add__", {}), ("fast_concat", {"axis": "blt"})]
)
def test_add_error_too_long_phase_center(uv_phase_time_split, test_func, extra_kwargs):
    (uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw) = uv_phase_time_split
    phase_center_radec = (Angle("0d").rad, Angle("-30d").rad, 7)
    func_kwargs = {
        "inplace": False,
        "phase_center_radec": phase_center_radec,
    }
    func_kwargs.update(extra_kwargs)
    with pytest.raises(ValueError) as cm:
        getattr(uv_phase_1, test_func)(uv_phase_2, **func_kwargs)
    assert str(cm.value).startswith("phase_center_radec should have length 2.")


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_fast_concat(casa_uvfits, hera_uvh5_xx, future_shapes):
    uv_full = casa_uvfits

    if future_shapes:
        uv_full.use_future_array_shapes()

    # Add frequencies
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 20))
    uv2.select(freq_chans=np.arange(20, 40))
    uv3.select(freq_chans=np.arange(40, 64))
    uv1.fast_concat([uv2, uv3], "freq", inplace=True)
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add frequencies - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 20))
    uv2.select(freq_chans=np.arange(20, 40))
    uv3.select(freq_chans=np.arange(40, 64))
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ]
        * 4
        + ["Combined frequencies are not evenly spaced"],
    ):
        uv2.fast_concat([uv1, uv3], "freq", inplace=True)

    assert uv2.Nfreqs == uv_full.Nfreqs
    assert uv2._freq_array != uv_full._freq_array
    assert uv2._data_array != uv_full._data_array

    # reorder frequencies and test that they are equal
    if future_shapes:
        index_array = np.argsort(uv2.freq_array)
        uv2.freq_array = uv2.freq_array[index_array]
        uv2.data_array = uv2.data_array[:, index_array, :]
        uv2.nsample_array = uv2.nsample_array[:, index_array, :]
        uv2.flag_array = uv2.flag_array[:, index_array, :]
    else:
        index_array = np.argsort(uv2.freq_array[0, :])
        uv2.freq_array = uv2.freq_array[:, index_array]
        uv2.data_array = uv2.data_array[:, :, index_array, :]
        uv2.nsample_array = uv2.nsample_array[:, :, index_array, :]
        uv2.flag_array = uv2.flag_array[:, :, index_array, :]
    uv2.history = uv_full.history
    assert uv2._freq_array == uv_full._freq_array
    assert uv2 == uv_full

    # Add polarizations
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:1])
    uv2.select(polarizations=uv2.polarization_array[1:3])
    uv3.select(polarizations=uv3.polarization_array[3:4])
    uv1.fast_concat([uv2, uv3], "polarization", inplace=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific polarizations using pyuvdata. "
        "Combined data along polarization axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add polarizations - out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:1])
    uv2.select(polarizations=uv2.polarization_array[1:3])
    uv3.select(polarizations=uv3.polarization_array[3:4])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ]
        * 4
        + ["Combined polarizations are not evenly spaced"],
    ):
        uv2.fast_concat([uv1, uv3], "polarization", inplace=True)

    assert uv2._polarization_array != uv_full._polarization_array
    assert uv2._data_array != uv_full._data_array

    # reorder pols
    uv2.reorder_pols()
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add times
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv3 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 3])
    uv2.select(times=times[len(times) // 3 : (len(times) // 3) * 2])
    uv3.select(times=times[(len(times) // 3) * 2 :])
    uv1.fast_concat([uv2, uv3], "blt", inplace=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    # divide in half to keep in order
    ind1 = np.arange(uv1.Nblts // 2)
    ind2 = np.arange(uv1.Nblts // 2, uv1.Nblts)
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv1.fast_concat(uv2, "blt", inplace=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1, uv_full

    # Add baselines out of order
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv2.fast_concat(uv1, "blt", inplace=True)
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
    assert uv2.blt_order == ("time", "baseline")
    uv2.blt_order = None
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # add baselines such that Nants_data needs to change
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv1.Nblts) if uv1.ant_1_array[i] not in ant_list]
    uv1.select(blt_inds=ind1)
    uv2.select(blt_inds=ind2)
    uv2.fast_concat(uv1, "blt", inplace=True)

    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv2.history,
    )

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
    assert uv2.blt_order == ("time", "baseline")
    uv2.blt_order = None
    uv2.history = uv_full.history
    assert uv2 == uv_full

    # Add multiple axes
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(
        times=times[0 : len(times) // 2], polarizations=uv1.polarization_array[0:2]
    )
    uv2.select(
        times=times[len(times) // 2 :], polarizations=uv2.polarization_array[2:4]
    )
    pytest.raises(ValueError, uv1.fast_concat, uv2, "blt", inplace=True)

    # Another combo
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32))
    uv2.select(times=times[len(times) // 2 :], freq_chans=np.arange(32, 64))
    pytest.raises(ValueError, uv1.fast_concat, uv2, "blt", inplace=True)

    # Add without inplace
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 = uv1.fast_concat(uv2, "blt", inplace=False)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Check warnings
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(33, 64))
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.fast_concat(uv1, "freq", inplace=True)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined frequencies are separated by more than their channel width",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.fast_concat(uv2, "freq")

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[1])
    uv2.freq_array += uv2._channel_width.tols[1] / 2.0
    with uvtest.check_warnings(
        UserWarning,
        "The uvw_array does not match the expected values given the antenna "
        "positions.",
        nwarnings=3,
    ):
        uv1.fast_concat(uv2, "freq")

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Combined polarizations are not evenly spaced",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv1.fast_concat(uv2, "polarization")

    # Combining histories
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv_new = uv1.fast_concat(uv2, "polarization")
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Unique part of next "
        "object history follows. testing the history.",
        uv_new.history,
    )
    uv_new.history = uv_full.history
    assert uv_new == uv_full

    uv_new = uv1.fast_concat(uv2, "polarization", verbose_history=True)
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Next object history "
        "follows." + uv2.history,
        uv_new.history,
    )

    # test add of autocorr-only and crosscorr-only objects
    uv_full = hera_uvh5_xx
    bls = uv_full.get_antpairs()
    autos = [bl for bl in bls if bl[0] == bl[1]]
    cross = sorted(set(bls) - set(autos))
    uv_auto = uv_full.select(bls=autos, inplace=False)
    uv_cross = uv_full.select(bls=cross, inplace=False)
    uv1 = uv_auto.fast_concat(uv_cross, "blt")
    assert uv1.Nbls == uv_auto.Nbls + uv_cross.Nbls
    uv2 = uv_cross.fast_concat(uv_auto, "blt")
    assert uv2.Nbls == uv_auto.Nbls + uv_cross.Nbls


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat_errors(casa_uvfits):
    uv_full = casa_uvfits

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    with pytest.raises(ValueError, match="If axis is specifed it must be one of"):
        uv1.fast_concat(uv2, "foo", inplace=True)

    uv2.use_future_array_shapes()
    with pytest.raises(
        ValueError,
        match="All objects must have the same `future_array_shapes` parameter.",
    ):
        uv1.fast_concat(uv2, "freq", inplace=True)

    cal = UVCal()
    with pytest.raises(
        ValueError, match="Only UVData \\(or subclass\\) objects can be added"
    ):
        uv1.fast_concat(cal, "freq", inplace=True)


def test_key2inds(casa_uvfits):
    # Test function to interpret key as antpair, pol
    uv = casa_uvfits

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
    ind1, ind2, indp = uv._key2inds("LL")
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([1]), indp[0])
    ind1, ind2, indp = uv._key2inds(("LL"))
    assert np.array_equal(np.arange(uv.Nblts), ind1)
    assert np.array_equal(np.array([]), ind2)
    assert np.array_equal(np.array([1]), indp[0])

    # Test invalid keys
    pytest.raises(KeyError, uv._key2inds, "I")  # pol str not in data
    pytest.raises(KeyError, uv._key2inds, -8)  # pol num not in data
    pytest.raises(KeyError, uv._key2inds, 6)  # bl num not in data
    pytest.raises(KeyError, uv._key2inds, (1, 1))  # ant pair not in data
    pytest.raises(KeyError, uv._key2inds, (1, 1, "rr"))  # ant pair not in data
    pytest.raises(KeyError, uv._key2inds, (0, 1, "xx"))  # pol not in data

    # Test autos are handled correctly
    uv.ant_2_array[0] = uv.ant_1_array[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant1, pol))
    assert np.array_equal(ind1, [0])
    assert np.array_equal(ind2, [])


def test_key2inds_conj_all_pols(casa_uvfits):
    uv = casa_uvfits

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


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_fringe(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
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


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_bl_fringe(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
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


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_missing_data(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]

    pytest.raises(KeyError, uv._key2inds, (ant2, ant1))


def test_key2inds_conj_all_pols_bls(casa_uvfits):
    uv = casa_uvfits

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


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_missing_data_bls(casa_uvfits):
    uv = casa_uvfits
    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bl = uvutils.antnums_to_baseline(ant2, ant1, uv.Nants_telescope)

    pytest.raises(KeyError, uv._key2inds, bl)


def test_smart_slicing_err(casa_uvfits):
    """
    Test that smart_slicing throws an error when using an invald squeeze
    """
    # Test invalid squeeze
    pytest.raises(
        ValueError,
        casa_uvfits._smart_slicing,
        casa_uvfits.data_array,
        [0, 4, 5],
        [],
        ([0, 1], []),
        squeeze="notasqueeze",
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_smart_slicing(casa_uvfits, future_shapes):
    # Test function to slice data
    uv = casa_uvfits

    if future_shapes:
        uv.use_future_array_shapes()

    # ind1 reg, ind2 empty, pol reg
    ind1 = 10 * np.arange(9)
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1]
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a view was returned
    if future_shapes:
        uv.data_array[ind1[1], 0, indp[0]] = 5.43
        assert d[1, 0, 0] == uv.data_array[ind1[1], 0, indp[0]]
    else:
        uv.data_array[ind1[1], 0, 0, indp[0]] = 5.43
        assert d[1, 0, 0] == uv.data_array[ind1[1], 0, 0, indp[0]]

    # force copy
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), force_copy=True)
    dcheck = uv.data_array[ind1]
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert d.flags.writeable
    # Ensure a copy was returned
    if future_shapes:
        uv.data_array[ind1[1], 0, indp[0]] = 4.3
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, indp[0]]
    else:
        uv.data_array[ind1[1], 0, 0, indp[0]] = 4.3
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 reg, ind2 empty, pol not reg
    ind1 = 10 * np.arange(9)
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1]
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    if future_shapes:
        uv.data_array[ind1[1], 0, indp[0]] = 1.2
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, indp[0]]
    else:
        uv.data_array[ind1[1], 0, 0, indp[0]] = 1.2
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 not reg, ind2 empty, pol reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1]
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    if future_shapes:
        uv.data_array[ind1[1], 0, indp[0]] = 8.2
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, indp[0]]
    else:
        uv.data_array[ind1[1], 0, 0, indp[0]] = 8.2
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 not reg, ind2 empty, pol not reg
    ind1 = [0, 4, 5]
    ind2 = []
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1]
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)
    assert not d.flags.writeable
    # Ensure a copy was returned
    if future_shapes:
        uv.data_array[ind1[1], 0, indp[0]] = 3.4
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, indp[0]]
    else:
        uv.data_array[ind1[1], 0, 0, indp[0]] = 3.4
        assert d[1, 0, 0] != uv.data_array[ind1[1], 0, 0, indp[0]]

    # ind1 empty, ind2 reg, pol reg
    # Note conjugation test ensures the result is a copy, not a view.
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2]
    if future_shapes:
        dcheck = np.squeeze(np.conj(dcheck[:, :, indp]))
    else:
        dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 reg, pol not reg
    ind1 = []
    ind2 = 10 * np.arange(9)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2]
    if future_shapes:
        dcheck = np.squeeze(np.conj(dcheck[:, :, indp]))
    else:
        dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 not reg, pol reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2]
    if future_shapes:
        dcheck = np.squeeze(np.conj(dcheck[:, :, indp]))
    else:
        dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1 empty, ind2 not reg, pol not reg
    ind1 = []
    ind2 = [1, 4, 5, 10]
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, ([], indp))
    dcheck = uv.data_array[ind2]
    if future_shapes:
        dcheck = np.squeeze(np.conj(dcheck[:, :, indp]))
    else:
        dcheck = np.squeeze(np.conj(dcheck[:, :, :, indp]))
    assert np.all(d == dcheck)

    # ind1, ind2 not empty, pol reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(uv.data_array[ind1], np.conj(uv.data_array[ind2]), axis=0)
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # ind1, ind2 not empty, pol not reg
    ind1 = np.arange(20)
    ind2 = np.arange(30, 40)
    indp = [0, 1, 3]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, indp))
    dcheck = np.append(uv.data_array[ind1], np.conj(uv.data_array[ind2]), axis=0)
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)

    # test single element
    ind1 = [45]
    ind2 = []
    indp = [0, 1]
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []))
    dcheck = uv.data_array[ind1]
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
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
    d = uv._smart_slicing(uv.data_array, ind1, ind2, (indp, []), squeeze="full")
    dcheck = uv.data_array[ind1]
    if future_shapes:
        dcheck = np.squeeze(dcheck[:, :, indp])
    else:
        dcheck = np.squeeze(dcheck[:, :, :, indp])
    assert np.all(d == dcheck)


def test_get_data(casa_uvfits):
    # Test get_data function for easy access to data
    uv = casa_uvfits

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
    assert str(cm.value).startswith("no more than 3 key values can be passed")

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


def test_get_flags(casa_uvfits):
    # Test function for easy access to flags
    uv = casa_uvfits

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
    assert str(cm.value).startswith("no more than 3 key values can be passed")

    # Check conjugation
    d = uv.get_flags(ant2, ant1, pol)
    assert np.all(dcheck == d)
    assert d.dtype == np.bool_

    # Antpair only
    dcheck = np.squeeze(uv.flag_array[bltind, :, :, :])
    d = uv.get_flags(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    dcheck = np.squeeze(uv.flag_array[:, :, :, 0])
    d = uv.get_flags(pol)
    assert np.all(dcheck == d)


def test_get_nsamples(casa_uvfits):
    # Test function for easy access to nsample array
    uv = casa_uvfits

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
    assert str(cm.value).startswith("no more than 3 key values can be passed")

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


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind(paper_uvh5):
    # Test for baseline-time axis indexer
    uv = paper_uvh5

    # get indices
    inds = uv.antpair2ind(0, 1, ordered=False)
    # fmt: off
    np.testing.assert_array_equal(
        inds,
        np.array(
            [
                1, 22, 43, 64, 85, 106, 127, 148, 169,
                190, 211, 232, 253, 274, 295, 316, 337,
                358, 379
            ]
        )
    )
    # fmt: on
    assert np.issubdtype(inds.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_conj(paper_uvh5):
    # conjugate (and use key rather than arg expansion)
    uv = paper_uvh5
    inds = uv.antpair2ind(0, 1, ordered=False)
    inds2 = uv.antpair2ind((1, 0), ordered=False)
    np.testing.assert_array_equal(inds, inds2)
    assert np.issubdtype(inds2.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_ordered(paper_uvh5):
    # test ordered
    uv = paper_uvh5
    inds = uv.antpair2ind(0, 1, ordered=False)

    # make sure conjugated baseline returns nothing
    inds2 = uv.antpair2ind(1, 0, ordered=True)
    assert inds2.size == 0

    # now use baseline actually in data
    inds2 = uv.antpair2ind(0, 1, ordered=True)
    np.testing.assert_array_equal(inds, inds2)
    assert np.issubdtype(inds2.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_autos(paper_uvh5):
    # test autos w/ and w/o ordered
    uv = paper_uvh5

    inds = uv.antpair2ind(0, 0, ordered=True)
    inds2 = uv.antpair2ind(0, 0, ordered=False)
    np.testing.assert_array_equal(inds, inds2)
    assert np.issubdtype(inds.dtype, np.integer)
    assert np.issubdtype(inds2.dtype, np.integer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_exceptions(paper_uvh5):
    # test exceptions
    uv = paper_uvh5

    with pytest.raises(ValueError, match="antpair2ind must be fed an antpair tuple"):
        uv.antpair2ind(1)
    with pytest.raises(ValueError, match="antpair2ind must be fed an antpair tuple"):
        uv.antpair2ind("bar", "foo")
    with pytest.raises(ValueError, match="ordered must be a boolean"):
        uv.antpair2ind(0, 1, "foo")

    return


def test_get_times(casa_uvfits):
    # Test function for easy access to times, to work in conjunction with get_data
    uv = casa_uvfits
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
    assert str(cm.value).startswith("no more than 3 key values can be passed")

    # Check conjugation
    d = uv.get_times(ant2, ant1, pol)
    assert np.all(dcheck == d)

    # Antpair only
    d = uv.get_times(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    d = uv.get_times(pol)
    assert np.all(d == uv.time_array)


def test_get_lsts(casa_uvfits):
    # Test function for easy access to LSTs, to work in conjunction with get_data
    uv = casa_uvfits
    # Get an antpair/pol combo (pol shouldn't actually effect result)
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    dcheck = uv.lst_array[bltind]
    d = uv.get_lsts(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = uv.get_lsts(ant1, ant2, uvutils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = uv.get_lsts((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError) as cm:
        uv.get_lsts((ant1, ant2, pol), (ant1, ant2, pol))
    assert str(cm.value).startswith("no more than 3 key values can be passed")

    # Check conjugation
    d = uv.get_lsts(ant2, ant1, pol)
    assert np.all(dcheck == d)

    # Antpair only
    d = uv.get_lsts(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    d = uv.get_lsts(pol)
    assert np.all(d == uv.lst_array)


def test_antpairpol_iter(casa_uvfits):
    # Test generator
    uv = casa_uvfits
    pol_dict = {
        uvutils.polnum2str(uv.polarization_array[i]): i for i in range(uv.Npols)
    }
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


def test_get_ants(casa_uvfits):
    # Test function to get unique antennas in data
    uv = casa_uvfits

    ants = uv.get_ants()
    for ant in ants:
        assert (ant in uv.ant_1_array) or (ant in uv.ant_2_array)
    for ant in uv.ant_1_array:
        assert ant in ants
    for ant in uv.ant_2_array:
        assert ant in ants


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_enu_antpos(hera_uvh5_xx):
    uvd = hera_uvh5_xx
    # no center, no pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=False, pick_data_ants=False)
    assert len(ants) == 113
    assert np.isclose(antpos[0, 0], 19.340211050751535)
    assert ants[0] == 0
    # test default behavior
    antpos2, ants = uvd.get_ENU_antpos()

    assert np.all(antpos == antpos2)
    # center
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=False)
    assert np.isclose(antpos[0, 0], 22.472442651767714)
    # pick data ants
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    assert ants[0] == 9
    assert np.isclose(antpos[0, 0], -0.0026981323386223721)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_telescope_loc_xyz_check(paper_uvh5, tmp_path):
    # test that improper telescope locations can still be read
    uv = paper_uvh5
    uv.telescope_location = uvutils.XYZ_from_LatLonAlt(*uv.telescope_location)
    # fix LST values
    uv.set_lsts_from_time_array()
    fname = str(tmp_path / "test.uvh5")
    uv.write_uvh5(fname, run_check=False, check_extra=False, clobber=True)

    # try to read file without checks (passing is implicit)
    uv.read(fname, run_check=False)

    # try to read without checks: assert it fails
    pytest.raises(ValueError, uv.read, fname)


def test_get_pols(casa_uvfits):
    # Test function to get unique polarizations in string format
    uv = casa_uvfits
    pols = uv.get_pols()
    pols_data = ["rr", "ll", "lr", "rl"]
    assert sorted(pols) == sorted(pols_data)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_pols_x_orientation(paper_uvh5):
    uv_in = paper_uvh5

    uv_in.x_orientation = "east"

    pols = uv_in.get_pols()
    pols_data = ["en"]
    assert pols == pols_data

    uv_in.x_orientation = "north"

    pols = uv_in.get_pols()
    pols_data = ["ne"]
    assert pols == pols_data


def test_get_feedpols(casa_uvfits):
    # Test function to get unique antenna feed polarizations in data. String format.
    uv = casa_uvfits
    pols = uv.get_feedpols()
    pols_data = ["r", "l"]
    assert sorted(pols) == sorted(pols_data)

    # Test break when pseudo-Stokes visibilities are present
    uv.polarization_array[0] = 1  # pseudo-Stokes I
    pytest.raises(ValueError, uv.get_feedpols)


def test_parse_ants(casa_uvfits, hera_uvh5_xx):
    # Test function to get correct antenna pairs and polarizations
    uv = casa_uvfits

    # All baselines
    ant_str = "all"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Auto correlations
    ant_str = "auto"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(ant_pairs_nums) == Counter([])
    assert isinstance(polarizations, type(None))

    # Cross correlations
    ant_str = "cross"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(uv.get_antpairs()) == Counter(ant_pairs_nums)
    assert isinstance(polarizations, type(None))

    # pseudo-Stokes params
    ant_str = "pI,pq,pU,pv"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    pols_expected = [4, 3, 2, 1]
    assert isinstance(ant_pairs_nums, type(None))
    assert Counter(polarizations) == Counter(pols_expected)

    # Unparsible string
    ant_str = "none"
    pytest.raises(ValueError, uv.parse_ants, ant_str)

    # Single antenna number
    ant_str = "0"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    # fmt: off
    ant_pairs_expected = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8),
                          (0, 11), (0, 14), (0, 18), (0, 19), (0, 20),
                          (0, 21), (0, 22), (0, 23), (0, 24), (0, 26),
                          (0, 27)]
    # fmt: on
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single antenna number not in the data
    ant_str = "10"
    with uvtest.check_warnings(
        UserWarning,
        "Warning: Antenna number 10 passed, but not present in the ant_1_array "
        "or ant_2_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)

    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Single antenna number with polarization, both not in the data
    ant_str = "10x"
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Antenna number 10 passed, but not present in the ant_1_array or "
            "ant_2_array",
            "Warning: Polarization XX,XY is not present in the polarization_array",
        ],
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Multiple antenna numbers as list
    ant_str = "22,26"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    # fmt: off
    ant_pairs_expected = [(0, 22), (0, 26), (1, 22), (1, 26), (2, 22), (2, 26),
                          (3, 22), (3, 26), (6, 22), (6, 26), (7, 22),
                          (7, 26), (8, 22), (8, 26), (11, 22), (11, 26),
                          (14, 22), (14, 26), (18, 22), (18, 26),
                          (19, 22), (19, 26), (20, 22), (20, 26),
                          (21, 22), (21, 26), (22, 23), (22, 24),
                          (22, 26), (22, 27), (23, 26), (24, 26),
                          (26, 27)]
    # fmt: on
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single baseline
    ant_str = "1_3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single baseline with polarization
    ant_str = "1l_3r"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3)]
    pols_expected = [-4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Single baseline with single polarization in first entry
    ant_str = "1l_3,2x_3"
    with uvtest.check_warnings(
        UserWarning,
        "Warning: Polarization XX,XY is not present in the polarization_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)

    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Single baseline with single polarization in last entry
    ant_str = "1_3l,2_3x"
    with uvtest.check_warnings(
        UserWarning,
        "Warning: Polarization XX,YX is not present in the polarization_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines as list
    ant_str = "1_2,1_3,1_11"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Multiples baselines with polarizations as list
    ant_str = "1r_2l,1l_3l,1r_11r"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 11)]
    pols_expected = [-1, -2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Specific baselines with parenthesis
    ant_str = "(1,3)_11"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11), (3, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Specific baselines with parenthesis
    ant_str = "1_(3,11)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations
    ant_str = "(1l,2r)_(3l,6r)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [-1, -2, -3, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers with - for avoidance
    ant_str = "1_(-3,11)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific antenna number
    ant_str = "1,-3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [
        (0, 1),
        (1, 2),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 11),
        (1, 14),
        (1, 18),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 26),
        (1, 27),
    ]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific baseline (same expected antenna pairs as above example)
    ant_str = "1,-1_3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations and - for avoidance
    ant_str = "1l_(-3r,11l)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 11)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers and pseudo-Stokes parameters
    ant_str = "(1l,2r)_(3l,6r),pI,pq"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols_expected = [2, 1, -1, -2, -3, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines with multiple polarizations, one pol to be removed
    ant_str = "1l_2,1l_3,-1l_3r"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines with multiple polarizations, one pol (not in data)
    # to be removed
    ant_str = "1l_2,1l_3,-1x_3y"
    with uvtest.check_warnings(
        UserWarning, "Warning: Polarization XY is not present in the polarization_array"
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3)]
    pols_expected = [-2, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Test print toggle on single baseline with polarization
    ant_str = "1l_2l"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str, print_toggle=True)
    ant_pairs_expected = [(1, 2)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Test ant_str='auto' on file with auto correlations
    uv = hera_uvh5_xx

    ant_str = "auto"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_nums = [
        9,
        10,
        20,
        22,
        31,
        43,
        53,
        64,
        65,
        72,
        80,
        81,
        88,
        89,
        96,
        97,
        104,
        105,
        112,
    ]
    ant_pairs_autos = [(ant_i, ant_i) for ant_i in ant_nums]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_autos)
    assert isinstance(polarizations, type(None))

    # Test cross correlation extraction on data with auto + cross
    ant_str = "cross"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_cross = list(itertools.combinations(ant_nums, 2))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_cross)
    assert isinstance(polarizations, type(None))

    # Remove only polarization of single baseline
    ant_str = "all,-9x_10x"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = ant_pairs_autos + ant_pairs_cross
    ant_pairs_expected.remove((9, 10))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Test appending all to beginning of strings that start with -
    ant_str = "-9"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = ant_pairs_autos + ant_pairs_cross
    for ant_i in ant_nums:
        ant_pairs_expected.remove((9, ant_i))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_with_ant_str(casa_uvfits, hera_uvh5_xx):
    # Test select function with ant_str argument
    uv = casa_uvfits
    inplace = False

    # All baselines
    ant_str = "all"
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Cross correlations
    ant_str = "cross"
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())
    # All baselines in data are cross correlations

    # Single antenna number
    ant_str = "0"
    ant_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 11),
        (0, 14),
        (0, 18),
        (0, 19),
        (0, 20),
        (0, 21),
        (0, 22),
        (0, 23),
        (0, 24),
        (0, 26),
        (0, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single antenna number not present in data
    ant_str = "10"
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Antenna number 10 passed, but not present in the "
            "ant_1_array or ant_2_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv.select(ant_str=ant_str, inplace=inplace)

    # Multiple antenna numbers as list
    ant_str = "22,26"
    ant_pairs = [
        (0, 22),
        (0, 26),
        (1, 22),
        (1, 26),
        (2, 22),
        (2, 26),
        (3, 22),
        (3, 26),
        (6, 22),
        (6, 26),
        (7, 22),
        (7, 26),
        (8, 22),
        (8, 26),
        (11, 22),
        (11, 26),
        (14, 22),
        (14, 26),
        (18, 22),
        (18, 26),
        (19, 22),
        (19, 26),
        (20, 22),
        (20, 26),
        (21, 22),
        (21, 26),
        (22, 23),
        (22, 24),
        (22, 26),
        (22, 27),
        (23, 26),
        (24, 26),
        (26, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single baseline
    ant_str = "1_3"
    ant_pairs = [(1, 3)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single baseline with polarization
    ant_str = "1l_3r"
    ant_pairs = [(1, 3)]
    pols = ["lr"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Single baseline with single polarization in first entry
    ant_str = "1l_3,2x_3"
    # x,y pols not present in data
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Polarization XX,XY is not present in the polarization_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv.select(ant_str=ant_str, inplace=inplace)
    # with polarizations in data
    ant_str = "1l_3,2_3"
    ant_pairs = [(1, 3), (2, 3)]
    pols = ["ll", "lr"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Single baseline with single polarization in last entry
    ant_str = "1_3l,2_3x"
    # x,y pols not present in data
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Polarization XX,YX is not present in the polarization_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    # with polarizations in data
    ant_str = "1_3l,2_3"
    ant_pairs = [(1, 3), (2, 3)]
    pols = ["ll", "rl"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Multiple baselines as list
    ant_str = "1_2,1_3,1_10"
    # Antenna number 10 not in data
    with uvtest.check_warnings(
        UserWarning,
        [
            "Warning: Antenna number 10 passed, but not present in the "
            "ant_1_array or ant_2_array",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv2 = uv.select(ant_str=ant_str, inplace=inplace)

    ant_pairs = [(1, 2), (1, 3)]
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Multiples baselines with polarizations as list
    ant_str = "1r_2l,1l_3l,1r_11r"
    ant_pairs = [(1, 2), (1, 3), (1, 11)]
    pols = ["rr", "ll", "rl"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Specific baselines with parenthesis
    ant_str = "(1,3)_11"
    ant_pairs = [(1, 11), (3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Specific baselines with parenthesis
    ant_str = "1_(3,11)"
    ant_pairs = [(1, 3), (1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations
    ant_str = "(1l,2r)_(3l,6r)"
    ant_pairs = [(1, 3), (1, 6), (2, 3), (2, 6)]
    pols = ["rr", "ll", "rl", "lr"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Antenna numbers with - for avoidance
    ant_str = "1_(-3,11)"
    ant_pairs = [(1, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    ant_str = "(-1,3)_11"
    ant_pairs = [(3, 11)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific antenna number
    ant_str = "1,-3"
    ant_pairs = [
        (0, 1),
        (1, 2),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 11),
        (1, 14),
        (1, 18),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 26),
        (1, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific baseline
    ant_str = "1,-1_3"
    ant_pairs = [
        (0, 1),
        (1, 2),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 11),
        (1, 14),
        (1, 18),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 26),
        (1, 27),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations and - for avoidance
    ant_str = "1l_(-3r,11l)"
    ant_pairs = [(1, 11)]
    pols = ["ll"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Test pseudo-Stokes params with select
    ant_str = "pi,pQ"
    pols = ["pQ", "pI"]
    uv.polarization_array = np.array([4, 3, 2, 1])
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(uv.get_antpairs())
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Test ant_str = 'auto' on file with auto correlations
    uv = hera_uvh5_xx

    ant_str = "auto"
    ant_nums = [
        9,
        10,
        20,
        22,
        31,
        43,
        53,
        64,
        65,
        72,
        80,
        81,
        88,
        89,
        96,
        97,
        104,
        105,
        112,
    ]
    ant_pairs_autos = [(ant_i, ant_i) for ant_i in ant_nums]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs_autos)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Test cross correlation extraction on data with auto + cross
    ant_str = "cross"
    ant_pairs_cross = list(itertools.combinations(ant_nums, 2))
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs_cross)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove only polarization of single baseline
    ant_str = "all,-9x_10x"
    ant_pairs = ant_pairs_autos + ant_pairs_cross
    ant_pairs.remove((9, 10))
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Test appending all to beginning of strings that start with -
    ant_str = "-9"
    ant_pairs = ant_pairs_autos + ant_pairs_cross
    for ant_i in ant_nums:
        ant_pairs.remove((9, ant_i))
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"ant_str": "", "antenna_nums": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        (
            {"ant_str": "", "antenna_names": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        (
            {"ant_str": "", "bls": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        (
            {"ant_str": "", "polarizations": []},
            "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
            "polarizations.",
        ),
        ({"ant_str": "auto"}, "There is no data matching ant_str=auto in this object."),
        (
            {"ant_str": "pI,pq,pU,pv"},
            "Polarization 4 is not present in the polarization_array",
        ),
        ({"ant_str": "none"}, "Unparsible argument none"),
    ],
)
def test_select_with_ant_str_errors(casa_uvfits, kwargs, message):
    uv = casa_uvfits

    with pytest.raises(ValueError, match=message):
        uv.select(**kwargs)


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.filterwarnings("ignore:Data will be unphased")
@pytest.mark.parametrize(
    "arg_dict,msg",
    [
        [{"out": "icrs", "oldproj": True}, "UVW calculation requires unphased data."],
        [{"out": "icrs", "oldproj": False}, "UVW recalculation requires either"],
        [
            {"allow": True, "orig": "gcrs", "out": "xyz", "oldproj": True},
            "Invalid parameter output_phase_frame.",
        ],
        [
            {"allow": True, "orig": "xyz", "out": "hcrs", "oldproj": True},
            "Invalid parameter orig_phase_frame.",
        ],
    ],
)
def test_set_uvws_from_antenna_pos_errs(casa_uvfits, arg_dict, msg):
    """
    Verify that set_uvws_from_antenna_pos throws appropriate errors when being
    provided with bad combinations are arguments.
    """
    with pytest.raises(ValueError) as cm:
        casa_uvfits.set_uvws_from_antenna_positions(
            allow_phasing=arg_dict.get("allow"),
            orig_phase_frame=arg_dict.get("orig"),
            output_phase_frame=arg_dict.get("out"),
            use_old_proj=arg_dict.get("oldproj"),
        )
    assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
def test_set_uvws_from_antenna_pos_old(uv_phase_comp):
    uv_object, _ = uv_phase_comp
    orig_uvw_array = uv_object.uvw_array.copy()

    with uvtest.check_warnings(
        [UserWarning, DeprecationWarning],
        [
            "Data will be unphased",
            "The original `phase` method is deprecated, and will be removed",
        ],
    ):
        uv_object.set_uvws_from_antenna_positions(
            allow_phasing=True, use_old_proj=True,
        )
    max_diff = np.amax(np.absolute(np.subtract(orig_uvw_array, uv_object.uvw_array)))
    assert np.isclose(max_diff, 0.0, atol=2)


def test_set_uvws_multi_phase_error(sma_mir):
    """
    Verify that we can't use the old proj method with multi-phase-ctr objects when
    calling set_uvws_from_antenna_positions.
    """
    with pytest.raises(
        NotImplementedError, match="Multi phase center data sets are not"
    ):
        sma_mir.set_uvws_from_antenna_positions(use_old_proj=True)


@pytest.mark.parametrize(
    "rephase,warn,msg",
    [[False, UserWarning, "Recalculating uvw_array without"], [True, None, None]],
)
def test_set_uvws_from_antenna_pos(sma_mir, sma_mir_main, rephase, warn, msg):
    # Now do this operation w/ SMA data, whose uvws are known good.
    with uvtest.check_warnings(warn, msg):
        sma_mir.set_uvws_from_antenna_positions(
            allow_phasing=rephase, require_phasing=rephase
        )
    max_diff = np.amax(np.absolute(sma_mir_main.uvw_array - sma_mir.uvw_array))
    assert np.isclose(max_diff, 0.0, atol=1e-5)
    # Verify that the data array is untouched if not rephased, otherwise that
    # the data are in fact different
    assert np.all(sma_mir_main.data_array == sma_mir.data_array) != rephase


def test_get_antenna_redundancies(pyuvsim_redundant):
    uv0 = pyuvsim_redundant

    old_bl_array = np.copy(uv0.baseline_array)
    red_gps, centers, lengths = uv0.get_redundancies(
        use_antpos=True, include_autos=False, conjugate_bls=True
    )
    # new and old baseline Numbers are not the same (different conjugation)
    assert not np.allclose(uv0.baseline_array, old_bl_array)

    # assert all baselines are in the data (because it's conjugated to match)
    for i, gp in enumerate(red_gps):
        for bl in gp:
            assert bl in uv0.baseline_array

    # conjugate data differently
    uv0.conjugate_bls(convention="ant1<ant2")
    new_red_gps, new_centers, new_lengths, conjs = uv0.get_redundancies(
        use_antpos=True, include_autos=False, include_conjugates=True
    )

    assert conjs is None

    apos, anums = uv0.get_ENU_antpos()
    new_red_gps, new_centers, new_lengths = uvutils.get_antenna_redundancies(
        anums, apos, include_autos=False
    )

    # all redundancy info is the same
    assert red_gps == new_red_gps
    assert np.allclose(centers, new_centers)
    assert np.allclose(lengths, new_lengths)


@pytest.mark.parametrize("method", ("select", "average"))
@pytest.mark.parametrize("reconjugate", (True, False))
@pytest.mark.parametrize("flagging_level", ("none", "some", "all"))
@pytest.mark.parametrize("future_shapes", [True, False])
def test_redundancy_contract_expand(
    method, reconjugate, flagging_level, future_shapes, pyuvsim_redundant
):
    # Test that a UVData object can be reduced to one baseline from each redundant group
    # and restored to its original form.

    uv0 = pyuvsim_redundant

    if future_shapes:
        uv0.use_future_array_shapes()

    # Fails at lower precision because some baselines fall into multiple
    # redundant groups
    tol = 0.02

    if reconjugate:
        # the test file has groups that are either all not conjugated or all conjugated.
        # need to conjugate some so we have mixed groups to properly test the average
        # method.
        (
            orig_red_gps,
            orig_centers,
            orig_lengths,
            orig_conjugates,
        ) = uv0.get_redundancies(tol, include_conjugates=True)
        blt_inds_to_conj = []
        for gp_ind, gp in enumerate(orig_red_gps):
            if len(gp) > 1:
                blt_inds_to_conj.extend(
                    list(np.nonzero(uv0.baseline_array == gp[0])[0])
                )
        uv0.conjugate_bls(convention=np.array(blt_inds_to_conj))

    # Assign identical data to each redundant group, set up flagging.
    # This must be done after reconjugation because reconjugation can alter the index
    # baseline
    red_gps, centers, lengths, conjugates = uv0.get_redundancies(
        tol, include_conjugates=True
    )
    index_bls = []
    for gp_ind, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            uv0.data_array[inds] += complex(gp_ind)
        index_bls.append(gp[0])

    if flagging_level == "none":
        assert np.all(~uv0.flag_array)
    elif flagging_level == "some":
        # flag all the index baselines in a redundant group
        for bl in index_bls:
            bl_locs = np.where(uv0.baseline_array == bl)
            uv0.flag_array[bl_locs] = True
    elif flagging_level == "all":
        uv0.flag_array[:] = True
        uv0.check()
        assert np.all(uv0.flag_array)

    uv3 = uv0.copy()
    if reconjugate:
        # undo the conjugations to make uv3 have different conjugations than uv0 to test
        # that we still get the same answer
        uv3.conjugate_bls(convention=np.array(blt_inds_to_conj))

    uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)
    uv2.check()

    if method == "average":
        gp_bl_use = []
        nbls_group = []
        for gp_ind, gp in enumerate(red_gps):
            bls_init = [bl for bl in gp if bl in uv0.baseline_array]
            nbls_group.append(len(bls_init))
            bl_use = [bl for bl in gp if bl in uv2.baseline_array]
            assert len(bl_use) == 1
            gp_bl_use.append(bl_use[0])

        for gp_ind, bl in enumerate(gp_bl_use):
            if flagging_level == "none" or flagging_level == "all":
                assert np.all(uv2.get_nsamples(bl) == nbls_group[gp_ind])
            else:
                assert np.all(uv2.get_nsamples(bl) == max((nbls_group[gp_ind] - 1), 1))
        if flagging_level == "all":
            assert np.all(uv2.flag_array)
        else:
            for gp_ind, bl in enumerate(gp_bl_use):
                if nbls_group[gp_ind] > 1:
                    assert np.all(~uv2.get_flags(bl))
    else:
        assert np.all(uv2.nsample_array == 1)
        if flagging_level == "some" or flagging_level == "all":
            assert np.all(uv2.flag_array)
        else:
            assert np.all(~uv2.flag_array)

    # Compare in-place to separated compression without the conjugation.
    uv3.compress_by_redundancy(method=method, tol=tol)
    if reconjugate:
        assert len(orig_red_gps) == len(red_gps)
        match_ind_list = []
        for gp_ind, gp in enumerate(red_gps):
            for bl in gp:
                match_ind = [
                    ind for ind, orig_gp in enumerate(orig_red_gps) if bl in orig_gp
                ]
                if len(match_ind) > 0:
                    break
            assert len(match_ind) == 1
            match_ind_list.append(match_ind[0])

        # the reconjugation of select baselines causes the set of baselines on the
        # two objects to differ. Need to match up baselines again
        unique_bls_2 = np.unique(uv2.baseline_array)
        unique_bls_3 = np.unique(uv3.baseline_array)
        unmatched_bls = list(
            set(unique_bls_2) - set(unique_bls_2).intersection(unique_bls_3)
        )

        # first find the ones that will be fixed by simple conjugation
        ant1, ant2 = uv2.baseline_to_antnums(unmatched_bls)
        conj_bls = uv2.antnums_to_baseline(ant2, ant1)
        bls_to_conj = list(set(unique_bls_3).intersection(conj_bls))
        if len(bls_to_conj) > 0:
            blts_to_conj = []
            for bl in bls_to_conj:
                blts_to_conj.extend(list(np.nonzero(uv3.baseline_array == bl)[0]))
            uv3.conjugate_bls(convention=blts_to_conj)

        # now check for ones that are still not matching
        unique_bls_3 = np.unique(uv3.baseline_array)
        unmatched_bls = list(
            set(unique_bls_2) - set(unique_bls_2).intersection(unique_bls_3)
        )
        for bl in unmatched_bls:
            assert bl in uv0.baseline_array
            for gp_ind, gp in enumerate(red_gps):
                if bl in gp:
                    bl_match = [
                        bl3
                        for bl3 in orig_red_gps[match_ind_list[gp_ind]]
                        if bl3 in unique_bls_3
                    ]
                    assert len(bl_match) == 1
                    blts = np.nonzero(uv3.baseline_array == bl_match[0])[0]
                    uv3.baseline_array[blts] = bl
                    # use the uvw values from the original for this baseline
                    orig_blts = np.nonzero(uv0.baseline_array == bl)[0]
                    uv3.uvw_array[blts, :] = uv0.uvw_array[orig_blts, :]
                    if method == "select":
                        # use the data values from the original for this baseline
                        # TODO: Spw axis to be collapsed in future release
                        uv2_blts = np.nonzero(uv2.baseline_array == bl)[0]
                        assert np.allclose(
                            uv2.data_array[uv2_blts], uv0.data_array[orig_blts],
                        )
                        uv3.data_array[blts] = uv2.data_array[uv2_blts]
                        if flagging_level == "some":
                            uv3.flag_array[:] = True

        uv3.ant_1_array, uv3.ant_2_array = uv3.baseline_to_antnums(uv3.baseline_array)
        uv3.Nants_data = uv3._calc_nants_data()
        unique_bls_3 = np.unique(uv3.baseline_array)
        unmatched_bls = list(
            set(unique_bls_2) - set(unique_bls_2).intersection(unique_bls_3)
        )
        assert set(unique_bls_2) == set(unique_bls_3)

    uv4 = uv2.copy()
    uv4.reorder_blts()
    uv3.reorder_blts()
    assert uv4 == uv3

    # check inflating gets back to the original
    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv2.inflate_by_redundancy(tol=tol)

    # Confirm that we get the same result looping inflate -> compress -> inflate.
    uv3 = uv2.compress_by_redundancy(method=method, tol=tol, inplace=False)
    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv3.inflate_by_redundancy(tol=tol)

    if method == "average":
        # with average, the nsample_array goes up by the number of baselines
        # averaged together.
        assert not np.allclose(uv3.nsample_array, uv2.nsample_array)
        # reset it to test other parameters
        uv3.nsample_array = uv2.nsample_array
    uv3.history = uv2.history
    assert uv2 == uv3

    uv2.history = uv0.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv0.reorder_blts(conj_convention="u>0")
    uv2.reorder_blts(conj_convention="u>0")
    uv2._uvw_array.tols = [0, tol]

    if method == "average":
        # with average, the nsample_array goes up by the number of baselines
        # averaged together.
        assert not np.allclose(uv2.nsample_array, uv0.nsample_array)
        # reset it to test other parameters
        uv2.nsample_array = uv0.nsample_array
    if flagging_level == "some":
        if method == "select":
            # inflated array will be entirely flagged
            assert np.all(uv2.flag_array)
            assert not np.allclose(uv0.flag_array, uv2.flag_array)
            uv2.flag_array = uv0.flag_array
        else:
            # flag arrays will not match -- inflated array will mostly be unflagged
            # it will only be flagged if only one in group
            assert not np.allclose(uv0.flag_array, uv2.flag_array)
            uv2.flag_array = uv0.flag_array

    assert uv2 == uv0


@pytest.mark.parametrize("method", ("select", "average"))
@pytest.mark.parametrize("flagging_level", ("none", "some", "all"))
def test_redundancy_contract_expand_variable_data(
    method, flagging_level, pyuvsim_redundant
):
    # Test that a UVData object can be reduced to one baseline from each redundant group
    # and restored to its original form.

    uv0 = pyuvsim_redundant

    # Fails at lower precision because some baselines fall into multiple
    # redundant groups
    tol = 0.02
    # Assign identical data to each redundant group in comparison object
    # Assign data to the index baseline and zeros elsewhere in the one to compress
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    index_bls = [gp[0] for gp in red_gps]
    uv0.data_array *= 0
    uv1 = uv0.copy()
    for gp_ind, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv1.data_array[inds] += complex(gp_ind)
            if bl in index_bls:
                uv0.data_array[inds] += complex(gp_ind)

    if flagging_level == "none":
        assert np.all(~uv0.flag_array)
    elif flagging_level == "some":
        # flag all the non index baselines in a redundant group
        uv0.flag_array[:, :, :, :] = True
        for bl in index_bls:
            bl_locs = np.where(uv0.baseline_array == bl)
            uv0.flag_array[bl_locs, :, :, :] = False
    elif flagging_level == "all":
        uv0.flag_array[:] = True
        uv0.check()
        assert np.all(uv0.flag_array)

    uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)

    # inflate to get back to the original size
    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv2.inflate_by_redundancy(tol=tol)

    uv2.history = uv1.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv1.reorder_blts(conj_convention="u>0")
    uv2.reorder_blts(conj_convention="u>0")
    uv2._uvw_array.tols = [0, tol]

    if method == "select":
        if flagging_level == "all":
            assert uv2._flag_array != uv1._flag_array
            uv2.flag_array = uv1.flag_array
        assert uv2 == uv1
    else:
        if flagging_level == "some":
            for gp in red_gps:
                bls_init = [bl for bl in gp if bl in uv1.baseline_array]
                for bl in bls_init:
                    assert np.all(uv2.get_data(bl) == uv1.get_data(bl))
                    assert np.all(uv2.get_nsamples(bl) == uv1.get_nsamples(bl))
        else:
            assert uv2.data_array.min() < uv1.data_array.min()
            assert np.all(uv2.data_array <= uv1.data_array)
            for gp in red_gps:
                bls_init = [bl for bl in gp if bl in uv1.baseline_array]
                for bl in bls_init:
                    assert np.all(
                        uv2.get_data(bl) == (uv1.get_data(bl) / len(bls_init))
                    )
                    assert np.all(uv2.get_nsamples(bl) == len(bls_init))


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("method", ("select", "average"))
def test_redundancy_contract_expand_nblts_not_nbls_times_ntimes(method, casa_uvfits):
    uv0 = casa_uvfits

    # check that Nblts != Nbls * Ntimes
    assert uv0.Nblts != uv0.Nbls * uv0.Ntimes

    tol = 1.0

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    for i, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds, ...] *= 0
            uv0.data_array[inds, ...] += complex(i)

    if method == "average":
        with uvtest.check_warnings(
            UserWarning,
            [
                "Index baseline in the redundant group does not have all the "
                "times, compressed object will be missing those times."
            ]
            * 4
            + [
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ],
        ):
            uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)
    else:
        uv2 = uv0.compress_by_redundancy(method=method, tol=tol, inplace=False)

    # check inflating gets back to the original
    with uvtest.check_warnings(
        UserWarning,
        [
            "Missing some redundant groups. Filling in available data.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    ):
        uv2.inflate_by_redundancy(tol=tol)

    uv2.history = uv0.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv0.reorder_blts()
    uv2.reorder_blts()
    uv2._uvw_array.tols = [0, tol]

    blt_inds = []
    missing_inds = []
    for bl, t in zip(uv0.baseline_array, uv0.time_array):
        if (bl, t) in zip(uv2.baseline_array, uv2.time_array):
            this_ind = np.where((uv2.baseline_array == bl) & (uv2.time_array == t))[0]
            blt_inds.append(this_ind[0])
        else:
            # this is missing because of the compress_by_redundancy step
            missing_inds.append(
                np.where((uv0.baseline_array == bl) & (uv0.time_array == t))[0]
            )

    uv3 = uv2.select(blt_inds=blt_inds, inplace=False)

    orig_inds_keep = list(np.arange(uv0.Nblts))
    for ind in missing_inds:
        orig_inds_keep.remove(ind)
    uv1 = uv0.select(blt_inds=orig_inds_keep, inplace=False)

    if method == "average":
        # the nsample array in the original object varies, so they
        # don't come out the same
        assert not np.allclose(uv3.nsample_array, uv1.nsample_array)
        uv3.nsample_array = uv1.nsample_array

    assert uv3 == uv1


def test_compress_redundancy_variable_inttime():
    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    tol = 0.05
    ntimes_in = uv0.Ntimes

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    index_bls = [gp[0] for gp in red_gps]
    uv0.data_array *= 0
    # set different int time for index baseline in object to compress
    uv1 = uv0.copy()
    ave_int_time = np.average(uv0.integration_time)
    nbls_group = np.zeros(len(red_gps))
    for gp_ind, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            if inds[0].size > 0:
                nbls_group[gp_ind] += 1
            uv1.data_array[inds] += complex(gp_ind)
            uv0.data_array[inds] += complex(gp_ind)
            if bl not in index_bls:
                uv0.integration_time[inds] = ave_int_time / 2

    assert uv0._integration_time != uv1._integration_time

    with uvtest.check_warnings(
        UserWarning,
        "Integrations times are not identical in a redundant "
        "group. Averaging anyway but this may cause unexpected "
        "behavior.",
        nwarnings=56,
    ) as warn_record:
        uv0.compress_by_redundancy(method="average", tol=tol)
    assert len(warn_record) == np.sum(nbls_group > 1) * ntimes_in

    uv1.compress_by_redundancy(method="average", tol=tol)

    assert uv0 == uv1


@pytest.mark.parametrize("method", ("select", "average"))
def test_compress_redundancy_metadata_only(method, pyuvsim_redundant):
    uv0 = pyuvsim_redundant

    tol = 0.05

    # Assign identical data to each redundant group
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True
    )
    for i, gp in enumerate(red_gps):
        for bl_ind, bl in enumerate(gp):
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            uv0.data_array[inds] += complex(i)

    uv2 = uv0.copy(metadata_only=True)
    uv2.compress_by_redundancy(method=method, tol=tol, inplace=True)

    uv0.compress_by_redundancy(method=method, tol=tol)
    uv0.data_array = None
    uv0.flag_array = None
    uv0.nsample_array = None
    assert uv0 == uv2


def test_compress_redundancy_wrong_method(pyuvsim_redundant):
    uv0 = pyuvsim_redundant

    tol = 0.05
    with pytest.raises(ValueError, match="method must be one of"):
        uv0.compress_by_redundancy(method="foo", tol=tol, inplace=True)


@pytest.mark.parametrize("method", ("select", "average"))
def test_redundancy_missing_groups(method, pyuvsim_redundant, tmp_path):
    # Check that if I try to inflate a compressed UVData that is missing
    # redundant groups, it will raise the right warnings and fill only what
    # data are available.

    uv0 = pyuvsim_redundant

    tol = 0.02
    num_select = 19

    uv0.compress_by_redundancy(method=method, tol=tol)
    fname = str(tmp_path / "temp_hera19_missingreds.uvfits")

    bls = np.unique(uv0.baseline_array)[:num_select]  # First twenty baseline groups
    uv0.select(bls=[uv0.baseline_to_antnums(bl) for bl in bls])
    uv0.write_uvfits(fname)
    uv1 = UVData()
    uv1.read_uvfits(fname)

    # check that filenames are what we expect
    assert uv0.filename == ["fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits"]
    assert uv1.filename == ["temp_hera19_missingreds.uvfits"]

    assert uv0 == uv1  # Check that writing compressed files causes no issues.

    with uvtest.check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv1.inflate_by_redundancy(tol=tol)

    uv2 = uv1.compress_by_redundancy(method=method, tol=tol, inplace=False)

    assert np.unique(uv2.baseline_array).size == num_select


def test_quick_redundant_vs_redundant_test_array(pyuvsim_redundant):
    """Verify the quick redundancy calc returns the same groups as a known array."""
    uv = pyuvsim_redundant

    uv.select(times=uv.time_array[0])
    uv.unphase_to_drift()
    uv.conjugate_bls(convention="u>0", use_enu=True)
    tol = 0.05
    # a quick and dirty redundancy calculation
    unique_bls, baseline_inds = np.unique(uv.baseline_array, return_index=True)
    uvw_vectors = np.take(uv.uvw_array, baseline_inds, axis=0)
    uvw_diffs = np.expand_dims(uvw_vectors, axis=0) - np.expand_dims(
        uvw_vectors, axis=1
    )
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
    groups = [sorted(bl for bl in grp if bl != -1) for grp in groups]
    groups.sort()

    redundant_groups, centers, lengths, conj_inds = uv.get_redundancies(
        tol=tol, include_conjugates=True
    )
    redundant_groups.sort()
    assert groups == redundant_groups


def test_redundancy_finder_when_nblts_not_nbls_times_ntimes(casa_uvfits):
    """Test the redundancy finder functions when Nblts != Nbls * Ntimes."""
    tol = 1  # meter
    uv = casa_uvfits
    uv.conjugate_bls(convention="u>0", use_enu=True)
    # check that Nblts != Nbls * Ntimes
    assert uv.Nblts != uv.Nbls * uv.Ntimes

    # a quick and dirty redundancy calculation
    unique_bls, baseline_inds = np.unique(uv.baseline_array, return_index=True)
    uvw_vectors = np.take(uv.uvw_array, baseline_inds, axis=0)
    uvw_diffs = np.expand_dims(uvw_vectors, axis=0) - np.expand_dims(
        uvw_vectors, axis=1
    )
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
    groups = [sorted(bl for bl in grp if bl != -1) for grp in groups]
    groups.sort()

    redundant_groups, centers, lengths, conj_inds = uv.get_redundancies(
        tol=tol, include_conjugates=True
    )
    redundant_groups.sort()
    assert groups == redundant_groups


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_overlapping_data_add(casa_uvfits, tmp_path, future_shapes):
    # read in test data
    uv = casa_uvfits

    if future_shapes:
        uv.use_future_array_shapes()

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
    extra_history = (
        "Downselected to specific baseline-times, polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Combined data along "
        "baseline-time axis using pyuvdata. Overwrote invalid data using pyuvdata."
    )
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history)
    uvfull.history = uv.history  # make histories match
    assert uv == uvfull

    # combine in a different order and check for equality
    uvfull = uv1.copy()
    uvfull += uv2
    uvfull2 = uv3.copy()
    uvfull2 += uv4
    uvfull += uvfull2
    extra_history2 = (
        "Downselected to specific baseline-times, polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Combined data along "
        "baseline-time axis using pyuvdata."
    )
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history2)
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
    uv1_out = str(tmp_path / "uv1.uvfits")
    uv1.write_uvfits(uv1_out)
    uv2_out = str(tmp_path / "uv2.uvfits")
    uv2.write_uvfits(uv2_out)
    uv3_out = str(tmp_path / "uv3.uvfits")
    uv3.write_uvfits(uv3_out)
    uv4_out = str(tmp_path / "uv4.uvfits")
    uv4.write_uvfits(uv4_out)

    uvfull = UVData()
    uvfull.read(np.array([uv1_out, uv2_out, uv3_out, uv4_out]))
    uvfull.reorder_blts()
    if future_shapes:
        uvfull.use_future_array_shapes()
    uv.reorder_blts()
    assert uvutils._check_histories(uvfull.history, uv.history + extra_history2)
    uvfull.history = uv.history  # make histories match

    # make sure filenames are what we expect
    assert set(uvfull.filename) == {
        "uv1.uvfits",
        "uv2.uvfits",
        "uv3.uvfits",
        "uv4.uvfits",
    }
    assert uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uvfull.filename = uv.filename
    uvfull._filename.form = (1,)

    assert uvfull == uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_lsts_from_time_with_only_unique(paper_uvh5):
    """
    Test `set_lsts_from_time_array` with only unique values is identical to full array.
    """
    uv = paper_uvh5
    lat, lon, alt = uv.telescope_location_lat_lon_alt_degrees
    # calculate the lsts for all elements in time array
    full_lsts = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    # use `set_lst_from_time_array` to set the uv.lst_array using only unique values
    uv.set_lsts_from_time_array()
    assert np.array_equal(full_lsts, uv.lst_array)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_lsts_from_time_with_only_unique_background(paper_uvh5):
    """
    Test `set_lsts_from_time_array` with only unique values is identical to full array.
    """
    uv = paper_uvh5
    lat, lon, alt = uv.telescope_location_lat_lon_alt_degrees
    # calculate the lsts for all elements in time array
    full_lsts = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    # use `set_lst_from_time_array` to set the uv.lst_array using only unique values
    proc = uv.set_lsts_from_time_array(background=True)
    proc.join()
    assert np.array_equal(full_lsts, uv.lst_array)


def test_copy(casa_uvfits):
    """Test the copy method"""
    uv_object = casa_uvfits

    uv_object_copy = uv_object.copy()
    assert uv_object_copy == uv_object

    uv_object_copy = uv_object.copy(metadata_only=True)
    assert uv_object_copy.metadata_only

    for name in uv_object._data_params:
        setattr(uv_object, name, None)
    assert uv_object_copy == uv_object

    uv_object_copy = uv_object.copy()
    assert uv_object_copy == uv_object

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_upsample_in_time(hera_uvh5, future_shapes):
    """Test the upsample_in_time method"""
    uv_object = hera_uvh5

    if future_shapes:
        uv_object.use_future_array_shapes()

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the same
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_with_flags(hera_uvh5):
    """Test the upsample_in_time method with flags"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0

    # add flags and upsample again
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    # data and nsamples should be changed as normal, but flagged
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0])
    out_flags = uv_object.get_flags(0, 1)
    assert np.all(out_flags[:2, 0, 0])
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_noninteger_resampling(hera_uvh5):
    """Test the upsample_in_time method with a non-integer resampling factor"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) * 0.75
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    assert np.allclose(uv_object.integration_time, max_integration_time * 0.5 / 0.75)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be different by a factor of 2
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_errors(hera_uvh5):
    """Test errors and warnings raised by upsample_in_time"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # test using a too-small integration time
    max_integration_time = 1e-3 * np.amin(uv_object.integration_time)
    with pytest.raises(ValueError) as cm:
        uv_object.upsample_in_time(max_integration_time)
    assert str(cm.value).startswith("Decreasing the integration time by more than")

    # catch a warning for doing no work
    uv_object2 = uv_object.copy()
    max_integration_time = 2 * np.amax(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(max_integration_time)
    assert uv_object == uv_object2

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_summing_correlator_mode(hera_uvh5):
    """Test the upsample_in_time method with summing correlator mode"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", summing_correlator_mode=True
    )

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the half the input
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0] / 2, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_summing_correlator_mode_with_flags(hera_uvh5):
    """Test the upsample_in_time method with summing correlator mode and flags"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # add flags and upsample again
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", summing_correlator_mode=True
    )

    # data and nsamples should be changed as normal, but flagged
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0] / 2, out_wf[0, 0, 0])
    out_flags = uv_object.get_flags(0, 1)
    assert np.all(out_flags[:2, 0, 0])
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_summing_correlator_mode_nonint_resampling(hera_uvh5):
    """Test the upsample_in_time method with summing correlator mode
    and non-integer resampling
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # try again with a non-integer resampling factor
    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) * 0.75
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", summing_correlator_mode=True
    )

    assert np.allclose(uv_object.integration_time, max_integration_time * 0.5 / 0.75)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be half the input
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[0, 0, 0] / 2, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_partial_upsample_in_time(hera_uvh5):
    """Test the upsample_in_time method with non-uniform upsampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # change a whole baseline's integration time
    bl_inds = uv_object.antpair2ind(0, 1)
    uv_object.integration_time[bl_inds] = uv_object.integration_time[0] / 2.0

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf_01 = uv_object.get_data(0, 1)
    init_wf_02 = uv_object.get_data(0, 2)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns_01 = uv_object.get_nsamples(0, 1)
    init_ns_02 = uv_object.get_nsamples(0, 2)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time)
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # output data should be the same
    out_wf_01 = uv_object.get_data(0, 1)
    out_wf_02 = uv_object.get_data(0, 2)
    assert np.all(init_wf_01 == out_wf_01)
    assert np.isclose(init_wf_02[0, 0, 0], out_wf_02[0, 0, 0])
    assert init_wf_02.size * 2 == out_wf_02.size

    # this should be true because there are no flags
    out_ns_01 = uv_object.get_nsamples(0, 1)
    out_ns_02 = uv_object.get_nsamples(0, 2)
    assert np.allclose(out_ns_01, init_ns_01)
    assert np.isclose(init_ns_02[0, 0, 0], out_ns_02[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_drift(hera_uvh5):
    """Test the upsample_in_time method on drift mode data"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", allow_drift=True
    )

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the same
    out_wf = uv_object.get_data(0, 1)
    # we need a "large" tolerance given the "large" data
    new_tol = 1e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0], atol=new_tol)

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_drift_no_phasing(hera_uvh5):
    """Test the upsample_in_time method on drift mode data without phasing"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    # upsample with allow_drift=False
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", allow_drift=False
    )

    assert np.allclose(uv_object.integration_time, max_integration_time)
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be similar, but somewhat different because of the phasing
    out_wf = uv_object.get_data(0, 1)
    # we need a "large" tolerance given the "large" data
    new_tol = 1e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0], atol=new_tol)

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(init_ns[0, 0, 0], out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_downsample_in_time(hera_uvh5, future_shapes):
    """Test the downsample_in_time method"""
    uv_object = hera_uvh5

    if future_shapes:
        uv_object.use_future_array_shapes()

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    # histories are different when n_times_to_avg is set vs min_int_time
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2

    assert not isinstance(uv_object.data_array, np.ma.MaskedArray)
    assert not isinstance(uv_object.nsample_array, np.ma.MaskedArray)


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_partial_flags(hera_uvh5):
    """Test the downsample_in_time method with partial flagging"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add flags and try again. With one of the 2 inputs flagged, the data should
    # just be the unflagged value and nsample should be half the unflagged one
    # and the output should not be flagged.
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object2 = uv_object.copy()

    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[1, 0, 0], out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are still no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_downsample_in_time_totally_flagged(hera_uvh5, future_shapes):
    """Test the downsample_in_time method with totally flagged integrations"""
    uv_object = hera_uvh5

    if future_shapes:
        uv_object.use_future_array_shapes()

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add more flags and try again. When all the input points are flagged,
    # data and nsample should have the same results as no flags but the output
    # should be flagged
    inds01 = uv_object.antpair2ind(0, 1)
    if future_shapes:
        uv_object.flag_array[inds01[:2], 0, 0] = True
    else:
        uv_object.flag_array[inds01[:2], 0, 0, 0] = True
    uv_object2 = uv_object.copy()

    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that the new sample is flagged
    out_flag = uv_object.get_flags(0, 1)
    assert out_flag[0, 0, 0]

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_uneven_samples(hera_uvh5):
    """Test the downsample_in_time method with uneven downsampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # test again with a downsample factor that doesn't go evenly into the
    # number of samples
    min_integration_time = original_int_time * 3.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=False,
    )

    # Only some baselines have an even number of times, so the output integration time
    # is not uniformly the same. For the test case, we'll have *either* the original
    # integration time or twice that.
    assert np.all(
        np.logical_or(
            np.isclose(uv_object.integration_time, original_int_time),
            np.isclose(uv_object.integration_time, min_integration_time),
        )
    )

    # make sure integration time is correct
    # in this case, all integration times should be the target one
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # as usual, the new data should be the average of the input data (3 points now)
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.mean(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=3, blt_order="baseline", minor_order="time", keep_ragged=False
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_uneven_samples_keep_ragged(hera_uvh5):
    """Test downsample_in_time with uneven downsampling and keep_ragged=True."""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # test again with a downsample factor that doesn't go evenly into the
    # number of samples
    min_integration_time = original_int_time * 3.0

    # test again with keep_ragged=False
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=True,
    )

    # as usual, the new data should be the average of the input data
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.mean(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=3, blt_order="baseline", minor_order="time", keep_ragged=True
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the sum
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]), out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_partial_flags(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode and
    partial flags
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add flags and try again. With one of the 2 inputs flagged, the data should
    # just be the unflagged value and nsample should be half the unflagged one
    # and the output should not be flagged.
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[1, 0, 0], out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are still no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_totally_flagged(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode and
    totally flagged integrations.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add more flags and try again. When all the input points are flagged,
    # data and nsample should have the same results as no flags but the output
    # should be flagged
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[:2], 0, 0, 0] = True
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]), out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that the new sample is flagged
    out_flag = uv_object.get_flags(0, 1)
    assert out_flag[0, 0, 0]

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_uneven_samples(hera_uvh5):
    """Test the downsample_in_time method with summing correlator mode and
    uneven samples.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # test again with a downsample factor that doesn't go evenly into the
    # number of samples
    min_integration_time = original_int_time * 3.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=False,
        summing_correlator_mode=True,
    )

    # Only some baselines have an even number of times, so the output integration time
    # is not uniformly the same. For the test case, we'll have *either* the original
    # integration time or twice that.
    assert np.all(
        np.logical_or(
            np.isclose(uv_object.integration_time, original_int_time),
            np.isclose(uv_object.integration_time, min_integration_time),
        )
    )

    # as usual, the new data should be the average of the input data (3 points now)
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.sum(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(np.mean(init_ns[0:3, 0, 0]), out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_summing_correlator_mode_uneven_samples_drop_ragged(
    hera_uvh5,
):
    """Test the downsample_in_time method with summing correlator mode and
    uneven samples, dropping ragged ones.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # test again with keep_ragged=False
    min_integration_time = original_int_time * 3.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        keep_ragged=False,
        summing_correlator_mode=True,
    )

    # make sure integration time is correct
    # in this case, all integration times should be the target one
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # as usual, the new data should be the average of the input data
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(np.sum(init_wf[0:3, 0, 0]), out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(np.mean(init_ns[0:3, 0, 0]), out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_partial_downsample_in_time(hera_uvh5):
    """Test the downsample_in_time method without uniform downsampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # change a whole baseline's integration time
    bl_inds = uv_object.antpair2ind(0, 1)
    uv_object.integration_time[bl_inds] = uv_object.integration_time[0] * 2.0

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline")

    # save some values for later
    init_wf_01 = uv_object.get_data(0, 1)
    init_wf_02 = uv_object.get_data(0, 2)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns_01 = uv_object.get_nsamples(0, 1)
    init_ns_02 = uv_object.get_nsamples(0, 2)

    # change the target integration time
    min_integration_time = np.amax(uv_object.integration_time)
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline"
    )

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # output data should be the same
    out_wf_01 = uv_object.get_data(0, 1)
    out_wf_02 = uv_object.get_data(0, 2)
    assert np.all(init_wf_01 == out_wf_01)
    assert np.isclose(
        (init_wf_02[0, 0, 0] + init_wf_02[1, 0, 0]) / 2.0, out_wf_02[0, 0, 0]
    )

    # this should be true because there are no flags
    out_ns_01 = uv_object.get_nsamples(0, 1)
    out_ns_02 = uv_object.get_nsamples(0, 2)
    assert np.allclose(out_ns_01, init_ns_01)
    assert np.isclose(
        (init_ns_02[0, 0, 0] + init_ns_02[1, 0, 0]) / 2.0, out_ns_02[0, 0, 0]
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_drift(hera_uvh5):
    """Test the downsample_in_time method on drift mode data"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", allow_drift=True
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", allow_drift=True
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_drift_no_phasing(hera_uvh5):
    """Test the downsample_in_time method on drift mode data without phasing"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # try again with allow_drift=False
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", allow_drift=False,
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be similar to the average, but somewhat different
    # because of the phasing
    out_wf = uv_object.get_data(0, 1)
    new_tol = 5e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0], atol=new_tol
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_nsample_precision(hera_uvh5):
    """Test the downsample_in_time method with a half-precision nsample_array"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    original_int_time = np.amax(uv_object.integration_time)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    min_integration_time = original_int_time * 2.0

    # add flags and try again. With one of the 2 inputs flagged, the data should
    # just be the unflagged value and nsample should be half the unflagged one
    # and the output should not be flagged.
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01[0], 0, 0, 0] = True
    uv_object2 = uv_object.copy()

    # change precision of nsample array
    uv_object.nsample_array = uv_object.nsample_array.astype(np.float16)
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(init_wf[1, 0, 0], out_wf[0, 0, 0])

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    # make sure nsamples has the right dtype
    assert uv_object.nsample_array.dtype.type is np.float16

    # check that there are still no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # Compare doing it with n_times_to_avg
    uv_object2.nsample_array = uv_object2.nsample_array.astype(np.float16)
    uv_object2.downsample_in_time(
        n_times_to_avg=2, blt_order="baseline", minor_order="time"
    )
    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history
    assert uv_object == uv_object2

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_errors(hera_uvh5):
    """Test various errors and warnings are raised"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # raise an error if set neither min_int_time and n_times_to_avg
    with pytest.raises(
        ValueError, match="Either min_int_time or n_times_to_avg must be set."
    ):
        uv_object.downsample_in_time()

    # raise an error if set both min_int_time and n_times_to_avg
    with pytest.raises(
        ValueError, match="Only one of min_int_time or n_times_to_avg can be set."
    ):
        uv_object.downsample_in_time(
            min_int_time=2 * np.amin(uv_object.integration_time), n_times_to_avg=2
        )
    # raise an error if only one time
    uv_object2 = uv_object.copy()
    uv_object2.select(times=uv_object2.time_array[0])
    with pytest.raises(
        ValueError, match="Only one time in this object, cannot downsample."
    ):
        uv_object2.downsample_in_time(n_times_to_avg=2)

    # raise an error for a too-large integration time
    max_integration_time = 1e3 * np.amax(uv_object.integration_time)
    with pytest.raises(
        ValueError, match="Increasing the integration time by more than"
    ):
        uv_object.downsample_in_time(min_int_time=max_integration_time)

    # catch a warning for doing no work
    uv_object2 = uv_object.copy()
    max_integration_time = 0.5 * np.amin(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning, match="All values in the integration_time array are already longer"
    ):
        uv_object.downsample_in_time(min_int_time=max_integration_time)

    assert uv_object == uv_object2
    del uv_object2

    # raise an error if n_times_to_avg is not an integer
    with pytest.raises(ValueError, match="n_times_to_avg must be an integer."):
        uv_object.downsample_in_time(n_times_to_avg=2.5)

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # make a gap in the times to check a warning about that
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01[-1]] += initial_int_time / (24 * 3600)
    uv_object.Ntimes += 1
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    times_01 = uv_object.get_times(0, 1)
    assert np.unique(np.diff(times_01)).size > 1
    with uvtest.check_warnings(
        UserWarning, "There is a gap in the times of baseline",
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_int_time_mismatch_warning(hera_uvh5):
    """Test warning in downsample_in_time about mismatch between integration
    times and the time between integrations.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the integration times to catch a warning about integration times
    # not matching the time delta between integrations
    uv_object.integration_time *= 0.5
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning,
        match="The time difference between integrations is not the same",
        nwarnings=10,
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_integration_time(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # test handling (& warnings) with varying integration time in a baseline
    # First, change both integration time & time array to match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01[-2]] += (initial_int_time / 2) / (24 * 3600)
    uv_object.time_array[inds01[-1]] += (3 * initial_int_time / 2) / (24 * 3600)
    uv_object.integration_time[inds01[-2:]] += initial_int_time
    uv_object.Ntimes = np.unique(uv_object.time_array).size
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    # check that there are no warnings about inconsistencies between
    # integration_time & time_array
    with uvtest.check_warnings(None):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    out_wf = uv_object.get_data(0, 1)

    n_times_in = init_wf.shape[0]
    n_times_out = out_wf.shape[0]
    assert n_times_out == (n_times_in - 2) / 2 + 2

    # output data should be the average for the first set
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])
    # last 2 time samples should be identical to initial ones
    assert np.isclose(init_wf[-1, 0, 0], out_wf[-1, 0, 0])
    assert np.isclose(init_wf[-2, 0, 0], out_wf[-2, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])
    assert np.isclose(init_ns[-1, 0, 0], out_ns[-1, 0, 0])
    assert np.isclose(init_ns[-2, 0, 0], out_ns[2, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_int_time_partial_flags(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline and partial flagging.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # downselect to 14 times and one baseline
    uv_object.select(times=np.unique(uv_object.time_array)[:14])

    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # change last 2 integrations to be twice as long
    # (so 12 normal length, 2 double length)
    # change integration time & time array to match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01[-2]] += (initial_int_time / 2) / (24 * 3600)
    uv_object.time_array[inds01[-1]] += (3 * initial_int_time / 2) / (24 * 3600)
    uv_object.integration_time[inds01[-2:]] += initial_int_time
    uv_object.Ntimes = np.unique(uv_object.time_array).size

    # add a flag on last time
    uv_object.flag_array[inds01[-1], :, :, :] = True
    # add a flag on thrid to last time
    uv_object.flag_array[inds01[-3], :, :, :] = True

    uv_object2 = uv_object.copy()

    with uvtest.check_warnings(None):
        uv_object.downsample_in_time(min_int_time=4 * initial_int_time)
    with uvtest.check_warnings(None):
        uv_object.downsample_in_time(min_int_time=8 * initial_int_time)
    with uvtest.check_warnings(None):
        uv_object2.downsample_in_time(min_int_time=8 * initial_int_time)

    assert uv_object.history != uv_object2.history
    uv_object2.history = uv_object.history

    assert uv_object == uv_object2
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_integration_time_warning(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline, but without adjusting the time_array so there is a mismatch.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # Next, change just integration time, so time array doesn't match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01[0]]
    uv_object.integration_time[inds01[-2:]] += initial_int_time
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    with uvtest.check_warnings(
        UserWarning, "The time difference between integrations is different than",
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    assert np.all(np.isclose(uv_object.integration_time, min_integration_time))

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose((init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0, out_wf[0, 0, 0])

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose((init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0, out_ns[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:Data will be unphased and rephased")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_downsample_in_time(hera_uvh5):
    """Test round trip works"""
    uv_object = hera_uvh5

    # Using astropy here (and elsewhere) to match previously calculated values
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time
    new_Nblts = uv_object.Nblts

    # check that calling upsample again with the same max_integration_time
    # gives warning and does nothing
    with uvtest.check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert uv_object.Nblts == new_Nblts

    # check that calling upsample again with the almost the same max_integration_time
    # gives warning and does nothing
    small_number = 0.9 * uv_object._integration_time.tols[1]
    with uvtest.check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(
            max_integration_time - small_number, blt_order="baseline",
        )
    assert uv_object.Nblts == new_Nblts

    uv_object.downsample_in_time(
        min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline",
    )

    # increase tolerance on LST if iers.conf.auto_max_age is set to None, as we
    # do in testing if the iers url is down. See conftest.py for more info.
    if iers.conf.auto_max_age is None:
        uv_object._lst_array.tols = (0, 1e-4)

    # make sure that history is correct
    assert (
        "Upsampled data to 0.939524 second integration time using pyuvdata."
        in uv_object.history
    )
    assert (
        "Downsampled data to 1.879048 second integration time using pyuvdata."
        in uv_object.history
    )

    # overwrite history and check for equality
    uv_object.history = uv_object2.history
    assert uv_object == uv_object2

    # check that calling downsample again with the same min_integration_time
    # gives warning and does nothing
    with uvtest.check_warnings(
        UserWarning, match="All values in the integration_time array are already longer"
    ):
        uv_object.downsample_in_time(
            min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline"
        )
    assert uv_object.Nblts == uv_object2.Nblts

    # check that calling upsample again with the almost the same min_integration_time
    # gives warning and does nothing
    with uvtest.check_warnings(
        UserWarning, match="All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(
            np.amin(uv_object2.integration_time) + small_number, blt_order="baseline"
        )

    assert uv_object.Nblts == uv_object2.Nblts

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:Data will be unphased and rephased")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_upsample_downsample_in_time_odd_resample(hera_uvh5, future_shapes):
    """Test round trip works with odd resampling"""
    uv_object = hera_uvh5

    if future_shapes:
        uv_object.use_future_array_shapes()

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # try again with a resampling factor of 3 (test odd numbers)
    max_integration_time = np.amin(uv_object.integration_time) / 3.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time

    uv_object.downsample_in_time(
        np.amin(uv_object2.integration_time), blt_order="baseline",
    )

    # increase tolerance on LST if iers.conf.auto_max_age is set to None, as we
    # do in testing if the iers url is down. See conftest.py for more info.
    if iers.conf.auto_max_age is None:
        uv_object._lst_array.tols = (0, 1e-4)

    # make sure that history is correct
    assert (
        "Upsampled data to 0.626349 second integration time using pyuvdata."
        in uv_object.history
    )
    assert (
        "Downsampled data to 1.879048 second integration time using pyuvdata."
        in uv_object.history
    )

    # overwrite history and check for equality
    uv_object.history = uv_object2.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_downsample_in_time_metadata_only(hera_uvh5):
    """Test round trip works with metadata-only objects"""
    uv_object = hera_uvh5

    # drop the data arrays
    uv_object.data_array = None
    uv_object.flag_array = None
    uv_object.nsample_array = None

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts(order="baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time

    uv_object.downsample_in_time(
        np.amin(uv_object2.integration_time), blt_order="baseline",
    )

    # increase tolerance on LST if iers.conf.auto_max_age is set to None, as we
    # do in testing if the iers url is down. See conftest.py for more info.
    if iers.conf.auto_max_age is None:
        uv_object._lst_array.tols = (0, 1e-4)

    # make sure that history is correct
    assert (
        "Upsampled data to 0.939524 second integration time using pyuvdata."
        in uv_object.history
    )
    assert (
        "Downsampled data to 1.879048 second integration time using pyuvdata."
        in uv_object.history
    )

    # overwrite history and check for equality
    uv_object.history = uv_object2.history
    assert uv_object == uv_object2


@pytest.mark.filterwarnings("ignore:Unknown phase types are no longer supported,")
@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_resample_in_time(bda_test_file, future_shapes):
    """Test the resample_in_time method"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv_object = bda_test_file

    if future_shapes:
        uv_object.use_future_array_shapes

    # save some initial info
    # 2s integration time
    init_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    init_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    init_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    init_data_136_137 = uv_object.get_data((136, 137))

    uv_object.resample_in_time(8, allow_drift=True)
    # Should have all the target integration time
    assert np.all(np.isclose(uv_object.integration_time, 8))

    # 2s integration time
    out_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    out_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    out_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    out_data_136_137 = uv_object.get_data((136, 137))

    # check array sizes make sense
    assert out_data_1_136.size * 4 == init_data_1_136.size
    assert out_data_1_137.size * 2 == init_data_1_137.size
    assert out_data_1_138.size == init_data_1_138.size
    assert out_data_136_137.size / 2 == init_data_136_137.size

    # check some values
    assert np.isclose(np.mean(init_data_1_136[0:4, 0, 0]), out_data_1_136[0, 0, 0])
    assert np.isclose(np.mean(init_data_1_137[0:2, 0, 0]), out_data_1_137[0, 0, 0])
    assert np.isclose(init_data_1_138[0, 0, 0], out_data_1_138[0, 0, 0])
    assert np.isclose(init_data_136_137[0, 0, 0], out_data_136_137[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_downsample_only(bda_test_file):
    """Test resample_in_time with downsampling only"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv_object = bda_test_file

    # save some initial info
    # 2s integration time
    init_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    init_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    init_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    init_data_136_137 = uv_object.get_data((136, 137))

    # resample again, with only_downsample set
    uv_object.resample_in_time(8, only_downsample=True, allow_drift=True)
    # Should have all less than or equal to the target integration time
    assert np.all(
        np.logical_or(
            np.isclose(uv_object.integration_time, 8),
            np.isclose(uv_object.integration_time, 16),
        )
    )

    # 2s integration time
    out_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    out_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    out_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    out_data_136_137 = uv_object.get_data((136, 137))

    # check array sizes make sense
    assert out_data_1_136.size * 4 == init_data_1_136.size
    assert out_data_1_137.size * 2 == init_data_1_137.size
    assert out_data_1_138.size == init_data_1_138.size
    assert out_data_136_137.size == init_data_136_137.size

    # check some values
    assert np.isclose(np.mean(init_data_1_136[0:4, 0, 0]), out_data_1_136[0, 0, 0])
    assert np.isclose(np.mean(init_data_1_137[0:2, 0, 0]), out_data_1_137[0, 0, 0])
    assert np.isclose(init_data_1_138[0, 0, 0], out_data_1_138[0, 0, 0])
    assert np.isclose(init_data_136_137[0, 0, 0], out_data_136_137[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_only_upsample(bda_test_file):
    """Test resample_in_time with only upsampling"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv_object = bda_test_file

    # save some initial info
    # 2s integration time
    init_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    init_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    init_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    init_data_136_137 = uv_object.get_data((136, 137))

    # again, with only_upsample set
    uv_object.resample_in_time(8, only_upsample=True, allow_drift=True)
    # Should have all greater than or equal to the target integration time
    assert np.all(
        np.logical_or(
            np.logical_or(
                np.isclose(uv_object.integration_time, 2.0),
                np.isclose(uv_object.integration_time, 4.0),
            ),
            np.isclose(uv_object.integration_time, 8.0),
        )
    )

    # 2s integration time
    out_data_1_136 = uv_object.get_data((1, 136))
    # 4s integration time
    out_data_1_137 = uv_object.get_data((1, 137))
    # 8s integration time
    out_data_1_138 = uv_object.get_data((1, 138))
    # 16s integration time
    out_data_136_137 = uv_object.get_data((136, 137))

    # check array sizes make sense
    assert out_data_1_136.size == init_data_1_136.size
    assert out_data_1_137.size == init_data_1_137.size
    assert out_data_1_138.size == init_data_1_138.size
    assert out_data_136_137.size / 2 == init_data_136_137.size

    # check some values
    assert np.isclose(init_data_1_136[0, 0, 0], out_data_1_136[0, 0, 0])
    assert np.isclose(init_data_1_137[0, 0, 0], out_data_1_137[0, 0, 0])
    assert np.isclose(init_data_1_138[0, 0, 0], out_data_1_138[0, 0, 0])
    assert np.isclose(init_data_136_137[0, 0, 0], out_data_136_137[0, 0, 0])

    return


@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_partial_flags(bda_test_file):
    """Test resample_in_time with partial flags"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv = bda_test_file
    # For ease, select a single baseline
    uv.select(bls=[(1, 136)])
    # Flag one time
    uv.flag_array[0, :, :, :] = True
    uv2 = uv.copy()

    # Downsample in two stages
    uv.resample_in_time(4.0, only_downsample=True)
    uv.resample_in_time(8.0, only_downsample=True)
    # Downsample in a single stage
    uv2.resample_in_time(8.0, only_downsample=True)

    assert uv.history != uv2.history
    uv2.history = uv.history
    assert uv == uv2
    return


@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_downsample_in_time_mwa():
    """
    Test resample in time works with numerical weirdnesses.

    In particular, when min_int_time is not quite an integer mulitple of
    integration_time. This text broke with a prior bug (see issue 773).
    """
    filename = os.path.join(DATA_PATH, "mwa_integration_time.uvh5")
    uv = UVData()
    uv.read(filename)
    uv.phase_to_time(np.mean(uv.time_array))
    uv_object2 = uv.copy()

    # all data within 5 milliseconds of 2 second integrations
    assert np.allclose(uv.integration_time, 2, atol=5e-3)
    min_int_time = 4.0
    uv.resample_in_time(min_int_time, only_downsample=True, keep_ragged=False)

    assert np.all(uv.integration_time > (min_int_time - 5e-3))

    # Now do the human expected thing:
    init_data = uv_object2.get_data((61, 58))
    uv_object2.downsample_in_time(n_times_to_avg=2, keep_ragged=False)

    assert uv_object2.Ntimes == 5

    out_data = uv_object2.get_data((61, 58))

    assert np.isclose(np.mean(init_data[0:2, 0, 0]), out_data[0, 0, 0])


@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_warning():
    filename = os.path.join(DATA_PATH, "mwa_integration_time.uvh5")
    uv = UVData()
    uv.read(filename)

    uv2 = uv.copy()

    with uvtest.check_warnings(
        UserWarning, match="No resampling will be done because target time"
    ):
        uv.resample_in_time(3, keep_ragged=False)

    assert uv2 == uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_frequency_average(casa_uvfits, future_shapes):
    """Test averaging in frequency."""
    uvobj = casa_uvfits

    if future_shapes:
        uvobj.use_future_array_shapes()
    uvobj2 = uvobj.copy()

    eq_coeffs = np.tile(
        np.arange(uvobj.Nfreqs, dtype=np.float64), (uvobj.Nants_telescope, 1),
    )
    uvobj.eq_coeffs = eq_coeffs
    uvobj.check()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    with uvtest.check_warnings(UserWarning, "eq_coeffs vary by frequency"):
        uvobj.frequency_average(2),

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    if future_shapes:
        expected_freqs = uvobj2.freq_array.reshape(int(uvobj2.Nfreqs / 2), 2).mean(
            axis=1
        )
    else:
        expected_freqs = uvobj2.freq_array.reshape(1, int(uvobj2.Nfreqs / 2), 2).mean(
            axis=2
        )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_coeffs = eq_coeffs.reshape(
        uvobj2.Nants_telescope, int(uvobj2.Nfreqs / 2), 2,
    ).mean(axis=2)
    assert np.max(np.abs(uvobj.eq_coeffs - expected_coeffs)) == 0

    # no flagging, so the following is true
    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    if future_shapes:
        reshape_tuple = (
            expected_data.shape[0],
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)

    else:
        reshape_tuple = (
            expected_data.shape[0],
            1,
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)

    assert np.nonzero(uvobj.flag_array)[0].size == 0

    assert not isinstance(uvobj.data_array, np.ma.MaskedArray)
    assert not isinstance(uvobj.nsample_array, np.ma.MaskedArray)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_frequency_average_uneven(casa_uvfits, future_shapes):
    """Test averaging in frequency with a number that is not a factor of Nfreqs."""
    uvobj = casa_uvfits

    if future_shapes:
        uvobj.use_future_array_shapes()
    uvobj2 = uvobj.copy()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    with uvtest.check_warnings(
        UserWarning,
        [
            "Nfreqs does not divide by `n_chan_to_avg` evenly. The final 1 "
            "frequencies will be excluded, to control which frequencies to exclude, "
            "use a select to control.",
            "The uvw_array does not match the expected values",
        ],
    ):
        uvobj.frequency_average(7)

    assert uvobj2.Nfreqs % 7 != 0

    assert uvobj.Nfreqs == (uvobj2.Nfreqs // 7)

    if future_shapes:
        expected_freqs = uvobj2.freq_array[np.arange((uvobj2.Nfreqs // 7) * 7)]
        expected_freqs = expected_freqs.reshape(int(uvobj2.Nfreqs // 7), 7).mean(axis=1)
    else:
        expected_freqs = uvobj2.freq_array[:, np.arange((uvobj2.Nfreqs // 7) * 7)]
        expected_freqs = expected_freqs.reshape(1, int(uvobj2.Nfreqs // 7), 7).mean(
            axis=2
        )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    # no flagging, so the following is true
    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    if future_shapes:
        expected_data = expected_data[:, 0 : ((uvobj2.Nfreqs // 7) * 7), :]
        reshape_tuple = (
            expected_data.shape[0],
            int(uvobj2.Nfreqs // 7),
            7,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)
    else:
        expected_data = expected_data[:, :, 0 : ((uvobj2.Nfreqs // 7) * 7), :]
        reshape_tuple = (
            expected_data.shape[0],
            1,
            int(uvobj2.Nfreqs // 7),
            7,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)

    assert np.nonzero(uvobj.flag_array)[0].size == 0


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_frequency_average_flagging(casa_uvfits, future_shapes):
    """Test averaging in frequency with flagging all samples averaged."""
    uvobj = casa_uvfits

    if future_shapes:
        uvobj.use_future_array_shapes()
    uvobj2 = uvobj.copy()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvobj.antpair2ind(0, 1)
    if future_shapes:
        uvobj.flag_array[inds01[0], 0:2, :] = True
    else:
        uvobj.flag_array[inds01[0], :, 0:2, :] = True
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols * 2

    uvobj.frequency_average(2)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    if future_shapes:
        expected_freqs = uvobj2.freq_array.reshape(int(uvobj2.Nfreqs / 2), 2).mean(
            axis=1
        )
    else:
        expected_freqs = uvobj2.freq_array.reshape(1, int(uvobj2.Nfreqs / 2), 2).mean(
            axis=2
        )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    if future_shapes:
        reshape_tuple = (
            expected_data.shape[0],
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)
    else:
        reshape_tuple = (
            expected_data.shape[0],
            1,
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)

    if future_shapes:
        assert np.sum(uvobj.flag_array[inds01[0], 0, :]) == 4
    else:
        assert np.sum(uvobj.flag_array[inds01[0], :, 0, :]) == 4
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols
    if future_shapes:
        assert np.nonzero(uvobj.flag_array[inds01[1:], 0, :])[0].size == 0
    else:
        assert np.nonzero(uvobj.flag_array[inds01[1:], :, 0, :])[0].size == 0


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_flagging_partial(casa_uvfits):
    """Test averaging in frequency with flagging only one sample averaged."""
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()
    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvobj.antpair2ind(0, 1)
    uvobj.flag_array[inds01[0], :, 0, :] = True
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols

    uvobj.frequency_average(2)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvobj2.freq_array.reshape(1, int(uvobj2.Nfreqs / 2), 2).mean(
        axis=2
    )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvobj2.Nfreqs / 2),
        2,
        uvobj2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    expected_data[0, :, 0, :] = uvobj2.data_array[inds01[0], :, 1, :]
    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_flagging_full_and_partial(casa_uvfits):
    """
    Test averaging in frequency with flagging all of one and only one of
    another sample averaged.
    """
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()
    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvobj.antpair2ind(0, 1)
    uvobj.flag_array[inds01[0], :, 0:3, :] = True
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols * 3

    uvobj.frequency_average(2)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvobj2.freq_array.reshape(1, int(uvobj2.Nfreqs / 2), 2).mean(
        axis=2
    )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvobj2.Nfreqs / 2),
        2,
        uvobj2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)

    expected_data[0, :, 1, :] = uvobj2.data_array[inds01[0], :, 3, :]

    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_flagging_partial_twostage(casa_uvfits):
    """
    Test averaging in frequency in two stages with flagging only one sample averaged.
    """
    uvobj = casa_uvfits
    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvobj.antpair2ind(0, 1)
    uvobj.flag_array[inds01[0], :, 0, :] = True
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols

    uv_object3 = uvobj.copy()

    uvobj.frequency_average(2)
    uvobj.frequency_average(2)

    uv_object3.frequency_average(4)

    assert uvobj == uv_object3


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_frequency_average_summing_corr_mode(casa_uvfits, future_shapes):
    """Test averaging in frequency."""
    # check that there's no flagging
    uvobj = casa_uvfits

    if future_shapes:
        uvobj.use_future_array_shapes()
    uvobj2 = uvobj.copy()

    assert np.nonzero(uvobj.flag_array)[0].size == 0

    uvobj.frequency_average(2, summing_correlator_mode=True)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    if future_shapes:
        expected_freqs = uvobj2.freq_array.reshape(int(uvobj2.Nfreqs / 2), 2).mean(
            axis=1
        )
    else:
        expected_freqs = uvobj2.freq_array.reshape(1, int(uvobj2.Nfreqs / 2), 2).mean(
            axis=2
        )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    # no flagging, so the following is true
    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    if future_shapes:
        reshape_tuple = (
            expected_data.shape[0],
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).sum(axis=2)
    else:
        reshape_tuple = (
            expected_data.shape[0],
            1,
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).sum(axis=3)
    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)

    assert np.nonzero(uvobj.flag_array)[0].size == 0
    assert not isinstance(uvobj.data_array, np.ma.MaskedArray)
    assert not isinstance(uvobj.nsample_array, np.ma.MaskedArray)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_frequency_average_propagate_flags(casa_uvfits, future_shapes):
    """
    Test averaging in frequency with flagging all of one and only one of
    another sample averaged, and propagating flags. Data should be identical,
    but flags should be slightly different compared to other test of the same
    name.
    """
    uvobj = casa_uvfits

    if future_shapes:
        uvobj.use_future_array_shapes()
    uvobj2 = uvobj.copy()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    # apply some flagging for testing
    inds01 = uvobj.antpair2ind(0, 1)
    if future_shapes:
        uvobj.flag_array[inds01[0], 0:3, :] = True
    else:
        uvobj.flag_array[inds01[0], :, 0:3, :] = True

    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols * 3

    uvobj.frequency_average(2, propagate_flags=True)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    if future_shapes:
        expected_freqs = uvobj2.freq_array.reshape(int(uvobj2.Nfreqs / 2), 2).mean(
            axis=1
        )
    else:
        expected_freqs = uvobj2.freq_array.reshape(1, int(uvobj2.Nfreqs / 2), 2).mean(
            axis=2
        )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    if future_shapes:
        reshape_tuple = (
            expected_data.shape[0],
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)

        expected_data[0, 1, :] = uvobj2.data_array[inds01[0], 3, :]
    else:
        reshape_tuple = (
            expected_data.shape[0],
            1,
            int(uvobj2.Nfreqs / 2),
            2,
            uvobj2.Npols,
        )
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)

        expected_data[0, :, 1, :] = uvobj2.data_array[inds01[0], :, 3, :]

    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)
    # Twice as many flags should exist compared to test of previous name.
    assert np.nonzero(uvobj.flag_array)[0].size == 2 * uvobj.Npols


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_nsample_precision(casa_uvfits):
    """Test averaging in frequency with a half-precision nsample_array."""
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()
    eq_coeffs = np.tile(
        np.arange(uvobj.Nfreqs, dtype=np.float64), (uvobj.Nants_telescope, 1),
    )
    uvobj.eq_coeffs = eq_coeffs
    uvobj.check()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    # change precision of the nsample array
    uvobj.nsample_array = uvobj.nsample_array.astype(np.float16)

    with uvtest.check_warnings(UserWarning, "eq_coeffs vary by frequency"):
        uvobj.frequency_average(2),

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    # TODO: Spw axis to be collapsed in future release
    expected_freqs = uvobj2.freq_array.reshape(1, int(uvobj2.Nfreqs / 2), 2).mean(
        axis=2
    )
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_coeffs = eq_coeffs.reshape(
        uvobj2.Nants_telescope, int(uvobj2.Nfreqs / 2), 2,
    ).mean(axis=2)
    assert np.max(np.abs(uvobj.eq_coeffs - expected_coeffs)) == 0

    # no flagging, so the following is true
    expected_data = uvobj2.get_data(0, 1, squeeze="none")
    # TODO: Spw axis to be collapsed in future release
    reshape_tuple = (
        expected_data.shape[0],
        1,
        int(uvobj2.Nfreqs / 2),
        2,
        uvobj2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=3)
    assert np.allclose(uvobj.get_data(0, 1, squeeze="none"), expected_data)

    assert np.nonzero(uvobj.flag_array)[0].size == 0

    assert not isinstance(uvobj.data_array, np.ma.MaskedArray)
    assert not isinstance(uvobj.nsample_array, np.ma.MaskedArray)

    # make sure we still have a half-precision nsample_array
    assert uvobj.nsample_array.dtype.type is np.float16


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_remove_eq_coeffs_divide(casa_uvfits, future_shapes):
    """Test using the remove_eq_coeffs method with divide convention."""
    uvobj = casa_uvfits

    if future_shapes:
        uvobj.use_future_array_shapes()
    uvobj2 = uvobj.copy()

    # give eq_coeffs to the object
    eq_coeffs = np.empty((uvobj.Nants_telescope, uvobj.Nfreqs), dtype=np.float64)
    for i, ant in enumerate(uvobj.antenna_numbers):
        eq_coeffs[i, :] = ant + 1
    uvobj.eq_coeffs = eq_coeffs
    uvobj.eq_coeffs_convention = "divide"
    uvobj.remove_eq_coeffs()

    # make sure the right coefficients were removed
    for key in uvobj.get_antpairs():
        eq1 = key[0] + 1
        eq2 = key[1] + 1
        blt_inds = uvobj.antpair2ind(key)
        norm_data = uvobj.data_array[blt_inds]
        unnorm_data = uvobj2.data_array[blt_inds]
        assert np.allclose(norm_data, unnorm_data / (eq1 * eq2))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_remove_eq_coeffs_multiply(casa_uvfits, future_shapes):
    """Test using the remove_eq_coeffs method with multiply convention."""
    uvobj = casa_uvfits

    if future_shapes:
        uvobj.use_future_array_shapes()
    uvobj2 = uvobj.copy()

    # give eq_coeffs to the object
    eq_coeffs = np.empty((uvobj.Nants_telescope, uvobj.Nfreqs), dtype=np.float64)
    for i, ant in enumerate(uvobj.antenna_numbers):
        eq_coeffs[i, :] = ant + 1
    uvobj.eq_coeffs = eq_coeffs
    uvobj.eq_coeffs_convention = "multiply"
    uvobj.remove_eq_coeffs()

    # make sure the right coefficients were removed
    for key in uvobj.get_antpairs():
        eq1 = key[0] + 1
        eq2 = key[1] + 1
        blt_inds = uvobj.antpair2ind(key)
        norm_data = uvobj.data_array[blt_inds]
        unnorm_data = uvobj2.data_array[blt_inds]
        assert np.allclose(norm_data, unnorm_data * (eq1 * eq2))

    return


def test_remove_eq_coeffs_errors(casa_uvfits):
    """Test errors raised by remove_eq_coeffs method."""
    uvobj = casa_uvfits
    # raise error when eq_coeffs are not defined
    with pytest.raises(ValueError) as cm:
        uvobj.remove_eq_coeffs()
    assert str(cm.value).startswith("The eq_coeffs attribute must be defined")

    # raise error when eq_coeffs are defined but not eq_coeffs_convention
    uvobj.eq_coeffs = np.ones((uvobj.Nants_telescope, uvobj.Nfreqs))
    with pytest.raises(ValueError) as cm:
        uvobj.remove_eq_coeffs()
    assert str(cm.value).startswith(
        "The eq_coeffs_convention attribute must be defined"
    )

    # raise error when convention is not a valid choice
    uvobj.eq_coeffs_convention = "foo"
    with pytest.raises(ValueError) as cm:
        uvobj.remove_eq_coeffs()
    assert str(cm.value).startswith("Got unknown convention foo. Must be one of")

    return


@pytest.mark.parametrize(
    "read_func,filelist",
    [
        ("read_miriad", [os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA")] * 2),
        (
            "read_mwa_corr_fits",
            [[mwa_corr_files[0:2], [mwa_corr_files[0], mwa_corr_files[2]]]],
        ),
        ("read_uvh5", [os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")] * 2),
        (
            "read_uvfits",
            [os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")] * 2,
        ),
        (
            "read_ms",
            [
                os.path.join(DATA_PATH, "multi_1.ms"),
                os.path.join(DATA_PATH, "multi_2.ms"),
            ],
        ),
        (
            "read_fhd",
            [
                list(np.array(fhd_files)[[0, 1, 2, 4, 6, 7]]),
                list(np.array(fhd_files)[[0, 2, 3, 5, 6, 7]]),
            ],
        ),
    ],
)
def test_multifile_read_errors(read_func, filelist):
    uv = UVData()
    with pytest.raises(ValueError) as cm:
        getattr(uv, read_func)(filelist)
    assert str(cm.value).startswith(
        "Reading multiple files from class specific read functions is no "
        "longer supported."
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multifile_read_check(hera_uvh5, tmp_path):
    """Test setting skip_bad_files=True when reading in files"""

    uvTrue = hera_uvh5.copy()

    uvh5_file = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")

    # Create a test file and remove header info to 'corrupt' it
    testfile = os.path.join(tmp_path, "zen.2458661.23480.HH.uvh5")

    uvTrue.write_uvh5(testfile)
    with h5py.File(testfile, "r+") as h5f:
        del h5f["Header/ant_1_array"]

    uv = UVData()
    # Test that the expected error arises
    with pytest.raises(KeyError) as cm:
        uv.read(testfile, skip_bad_files=False)
    assert "Unable to open object (object 'ant_1_array' doesn't exist)" in str(cm.value)

    # Test when the corrupted file is at the beggining, skip_bad_files=False
    fileList = [testfile, uvh5_file]
    with pytest.raises(KeyError) as cm:
        with uvtest.check_warnings(UserWarning, match="Failed to read"):
            uv.read(fileList, skip_bad_files=False)
    assert "Unable to open object (object 'ant_1_array' doesn't exist)" in str(cm.value)
    assert uv != uvTrue

    # Test when the corrupted file is at the beggining, skip_bad_files=True
    fileList = [testfile, uvh5_file]
    with uvtest.check_warnings(
        UserWarning, match=["Failed to read"],
    ):
        uv.read(fileList, skip_bad_files=True)
    assert uv == uvTrue

    # Test when the corrupted file is at the end of a list
    fileList = [uvh5_file, testfile]
    with uvtest.check_warnings(
        UserWarning, match=["Failed to read"],
    ):
        uv.read(fileList, skip_bad_files=True)
    # Check that the uncorrupted file was still read in
    assert uv == uvTrue

    os.remove(testfile)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("err_type", ["KeyError", "ValueError"])
def test_multifile_read_check_long_list(hera_uvh5, tmp_path, err_type):
    """
    Test KeyError catching by setting skip_bad_files=True when
    reading in files for a list of length >2
    """
    # Create mini files for testing
    uv = hera_uvh5

    fileList = []
    for i in range(0, 4):
        uv2 = uv.select(
            times=np.unique(uv.time_array)[i * 5 : i * 5 + 4], inplace=False
        )
        fname = str(tmp_path / f"minifile_{i}.uvh5")
        fileList.append(fname)
        uv2.write_uvh5(fname)
    if err_type == "KeyError":
        with h5py.File(fileList[-1], "r+") as h5f:
            del h5f["Header/ant_1_array"]
    elif err_type == "ValueError":
        with h5py.File(fileList[-1], "r+") as h5f:
            h5f["Header/antenna_numbers"][3] = 85
            h5f["Header/ant_1_array"][2] = 1024

    # Test with corrupted file as last file in list, skip_bad_files=True
    uvTest = UVData()
    with uvtest.check_warnings(UserWarning, "Failed to read"):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()
    uvTrue.read(fileList[0:3], skip_bad_files=True)

    assert uvTest == uvTrue

    # Repeat above test, but with corrupted file as first file in list
    os.remove(fileList[3])
    uv2 = uv.select(times=np.unique(uv.time_array)[15:19], inplace=False)
    fname = str(tmp_path / f"minifile_{3}.uvh5")
    uv2.write_uvh5(fname)
    if err_type == "KeyError":
        with h5py.File(fileList[0], "r+") as h5f:
            del h5f["Header/ant_1_array"]
    elif err_type == "ValueError":
        with h5py.File(fileList[0], "r+") as h5f:
            h5f["Header/antenna_numbers"][3] = 85
            h5f["Header/ant_1_array"][2] = 1024
    uvTest = UVData()
    with uvtest.check_warnings(UserWarning, "Failed to read"):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()
    uvTrue.read(fileList[1:4], skip_bad_files=True)

    assert uvTest == uvTrue

    # Test with corrupted file first in list, but with skip_bad_files=False
    uvTest = UVData()
    if err_type == "KeyError":
        with pytest.raises(KeyError, match="Unable to open object"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    elif err_type == "ValueError":
        with pytest.raises(ValueError, match="Nants_data must be equal to"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    uvTrue = UVData()
    uvTrue.read([fileList[1], fileList[2], fileList[3]], skip_bad_files=False)

    assert uvTest != uvTrue

    # Repeat above test, but with corrupted file in the middle of the list
    os.remove(fileList[0])
    uv2 = uv.select(times=np.unique(uv.time_array)[0:4], inplace=False)
    fname = str(tmp_path / f"minifile_{0}.uvh5")
    uv2.write_uvh5(fname)
    if err_type == "KeyError":
        with h5py.File(fileList[1], "r+") as h5f:
            del h5f["Header/ant_1_array"]
    elif err_type == "ValueError":
        with h5py.File(fileList[1], "r+") as h5f:
            h5f["Header/antenna_numbers"][3] = 85
            h5f["Header/ant_1_array"][2] = 1024
    uvTest = UVData()
    with uvtest.check_warnings(UserWarning, "Failed to read"):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()
    uvTrue.read([fileList[0], fileList[2], fileList[3]], skip_bad_files=True)

    assert uvTest == uvTrue

    # Test with corrupted file in middle of list, but with skip_bad_files=False
    uvTest = UVData()
    if err_type == "KeyError":
        with pytest.raises(KeyError, match="Unable to open object"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    elif err_type == "ValueError":
        with pytest.raises(ValueError, match="Nants_data must be equal to"):
            with uvtest.check_warnings(UserWarning, match="Failed to read"):
                uvTest.read(fileList[0:4], skip_bad_files=False)
    uvTrue = UVData()
    uvTrue.read([fileList[0], fileList[2], fileList[3]], skip_bad_files=False)

    assert uvTest != uvTrue

    # Test case where all files in list are corrupted
    os.remove(fileList[1])
    uv2 = uv.select(times=np.unique(uv.time_array)[5:9], inplace=False)
    fname = str(tmp_path / f"minifile_{1}.uvh5")
    uv2.write_uvh5(fname)
    for file in fileList:
        if err_type == "KeyError":
            with h5py.File(file, "r+") as h5f:
                del h5f["Header/ant_1_array"]
        elif err_type == "ValueError":
            with h5py.File(file, "r+") as h5f:
                h5f["Header/antenna_numbers"][3] = 85
                h5f["Header/ant_1_array"][2] = 1024
    uvTest = UVData()
    with uvtest.check_warnings(
        UserWarning,
        match=(
            "########################################################\n"
            "ALL FILES FAILED ON READ - NO READABLE FILES IN FILENAME\n"
            "########################################################"
        ),
    ):
        uvTest.read(fileList[0:4], skip_bad_files=True)
    uvTrue = UVData()

    assert uvTest == uvTrue

    os.remove(fileList[0])
    os.remove(fileList[1])
    os.remove(fileList[2])
    os.remove(fileList[3])

    return


def test_unknown_phase():
    """
    Test to see that unknown phase types now throw an error
    """
    uv = UVData()
    uv.phase_type = "unknown"
    with pytest.raises(
        ValueError, match='Phase type must be either "phased" or "drift"'
    ):
        uv.check()


def test_set_phased():
    """
    Test the deprecation warnings in set_phased et al.
    """
    uv = UVData()
    uv._set_phased()
    assert uv.phase_type == "phased"
    assert uv._phase_center_ra.required
    assert uv._phase_center_dec.required
    assert uv._phase_center_app_ra.required
    assert uv._phase_center_app_dec.required
    assert uv._phase_center_frame_pa.required


def test_set_drift():
    """
    Test parameter settings with _set_drift
    """
    uv = UVData()
    uv._set_drift()
    assert uv.phase_type == "drift"
    assert not uv._phase_center_ra.required
    assert not uv._phase_center_dec.required
    assert not uv._phase_center_app_ra.required
    assert not uv._phase_center_app_dec.required
    assert not uv._phase_center_frame_pa.required


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_background_lsts():
    """Test reading a file with the lst calc in the background."""
    uvd = UVData()
    uvd2 = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uvd.read(testfile, background_lsts=False)
    uvd2.read(testfile, background_lsts=True)
    assert uvd == uvd2


def test_parse_ants_x_orientation_kwarg(hera_uvh5):
    uvd = hera_uvh5
    # call with x_orientation = None to make parse_ants read from the object
    ant_pair, pols = uvutils.parse_ants(uvd, "cross")
    ant_pair2, pols2 = uvd.parse_ants("cross")
    assert np.array_equal(ant_pair, ant_pair2)
    assert np.array_equal(pols, pols2)


def test_rephase_to_time():
    uvfits_file = os.path.join(DATA_PATH, "1061316296.uvfits")
    uvd = UVData()

    uvd.read(uvfits_file)
    phase_time = np.unique(uvd.time_array)[1]
    time = Time(phase_time, format="jd")
    # Generate ra/dec of zenith at time in the phase_frame coordinate
    # system to use for phasing
    telescope_location = EarthLocation.from_geocentric(
        *uvd.telescope_location, unit="m"
    )

    zenith_coord = SkyCoord(
        alt=Angle(90 * units.deg),
        az=Angle(0 * units.deg),
        obstime=time,
        frame="altaz",
        location=telescope_location,
    )

    obs_zenith_coord = zenith_coord.transform_to("icrs")
    zenith_ra = obs_zenith_coord.ra.rad
    zenith_dec = obs_zenith_coord.dec.rad

    uvd.phase_to_time(phase_time)

    assert uvd.phase_center_ra == zenith_ra
    assert uvd.phase_center_dec == zenith_dec


def test_print_object_standard(sma_mir, hera_uvh5):
    """
    Check that the 'standard' mode of print_object works.
    """
    check_str = (
        "   ID     Cat Entry       Type     Az/Lon/RA    El/Lat/Dec  Frame    Epoch \n"
        "    #          Name                    hours           deg                 \n"
        "---------------------------------------------------------------------------\n"
        "    1          3c84   sidereal    3:19:48.16  +41:30:42.11    fk5  J2000.0 \n"
    )

    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str

    # Make sure we can specify the object name and get the same result
    table_str = sma_mir.print_phase_center_info(
        print_table=False, return_str=True, cat_name="3c84",
    )
    assert table_str == check_str

    # Make sure that things still work when we force the HMS format
    table_str = sma_mir.print_phase_center_info(
        print_table=False, return_str=True, hms_format=True
    )
    assert table_str == check_str


def test_print_object_dms(sma_mir):
    """
    Test that forcing DMS format works as expected.
    """
    check_str = (
        "   ID     Cat Entry       Type      Az/Lon/RA    El/Lat/Dec  Frame    Epoch \n"
        "    #          Name                       deg           deg                 \n"
        "----------------------------------------------------------------------------\n"
        "    1          3c84   sidereal    49:57:02.40  +41:30:42.11    fk5  J2000.0 \n"
    )

    # And likewise when forcing the degree format
    table_str = sma_mir.print_phase_center_info(
        print_table=False, return_str=True, hms_format=False
    )
    assert table_str == check_str


def test_print_object_full(sma_mir):
    """
    Test that print object w/ all optional paramters prints as expected.
    """
    # Now check and see what happens if we add the full suite of phase center parameters
    _ = sma_mir._add_phase_center(
        "3c84",
        cat_type="sidereal",
        cat_lat=sma_mir.phase_center_catalog["3c84"]["cat_lat"],
        cat_lon=sma_mir.phase_center_catalog["3c84"]["cat_lon"],
        cat_dist=0.0,
        cat_vrad=0.0,
        cat_pm_ra=0.0,
        cat_pm_dec=0.0,
        cat_frame="fk5",
        cat_epoch=2000.0,
        force_update=True,
    )
    check_str = (
        "   ID     Cat Entry       Type     Az/Lon/RA"
        "    El/Lat/Dec  Frame    Epoch   PM-Ra  PM-Dec     Dist   V_rad \n"
        "    #          Name                    hours"
        "           deg                  mas/yr  mas/yr       pc    km/s \n"
        "--------------------------------------------"
        "----------------------------------------------------------------\n"
        "    1          3c84   sidereal    3:19:48.16"
        "  +41:30:42.11    fk5  J2000.0       0       0  0.0e+00       0 \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


def test_print_object_ephem(sma_mir):
    """
    Test that printing ephem objects works as expected.
    """
    # Now check and see that printing ephems works well
    _ = sma_mir._add_phase_center(
        "3c84",
        cat_type="ephem",
        cat_lat=0.0,
        cat_lon=0.0,
        cat_frame="icrs",
        cat_times=2456789.0,
        force_update=True,
    )
    check_str = (
        "   ID     Cat Entry       Type     Az/Lon/RA"
        "    El/Lat/Dec  Frame        Ephem Range    \n"
        "    #          Name                    hours"
        "           deg         Start-MJD    End-MJD \n"
        "---------------------------------------------"
        "-------------------------------------------\n"
        "    1          3c84      ephem    0:00:00.00"
        "  + 0:00:00.00   icrs   56788.50   56788.50 \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


def test_print_object_driftscan(sma_mir):
    """
    Test that printing driftscan objects works as expected.
    """
    # Check and see that if we force this to be a driftscan, we get the output
    # we expect
    _ = sma_mir._add_phase_center("3c84", cat_type="driftscan", force_update=True)
    check_str = (
        "   ID     Cat Entry       Type      Az/Lon/RA    El/Lat/Dec  Frame \n"
        "    #          Name                       deg           deg        \n"
        "-------------------------------------------------------------------\n"
        "    1          3c84  driftscan     0:00:00.00  +90:00:00.00  altaz \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


def test_print_object_unphased(sma_mir):
    _ = sma_mir._add_phase_center("3c84", cat_type="unphased", force_update=True)
    check_str = (
        "   ID     Cat Entry       Type      Az/Lon/RA    El/Lat/Dec  Frame \n"
        "    #          Name                       deg           deg        \n"
        "-------------------------------------------------------------------\n"
        "    1          3c84   unphased     0:00:00.00  +90:00:00.00  altaz \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


def test_print_object_no_math(sma_mir):
    """
    Test that print_object fails as expected when print
    """
    with pytest.raises(ValueError, match="No entry by the name test in the catalog."):
        sma_mir.print_phase_center_info(cat_name="test")


def test_print_object_no_multi_phase(hera_uvh5):
    """
    Test that print_object throws the expected error when attempting to use it on a
    non-multi-phase-center data set.
    """
    with pytest.raises(ValueError, match="Cannot use print_phase_center_info"):
        hera_uvh5.print_phase_center_info()


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
def test_print_object_multi(carma_miriad):
    """
    Test the print_phase_center_info function when there are multiple objects stored in
    the internal catalog.
    """
    pytest.importorskip("pyuvdata._miriad")

    _ = carma_miriad._add_phase_center("NOISE", cat_type="unphased", force_update=True)
    check_str = (
        "   ID     Cat Entry       Type     Az/Lon/RA    El/Lat/Dec  Frame    Epoch \n"
        "    #          Name                    hours           deg                 \n"
        "---------------------------------------------------------------------------\n"
        "    0         NOISE   unphased    0:00:00.00  +90:00:00.00  altaz          \n"
        "    1         3C273   sidereal   12:29:06.70  + 2:03:08.60    fk5  J2000.0 \n"
        "    2      1159+292   sidereal   11:59:31.83  +29:14:43.83    fk5  J2000.0 \n"
    )
    table_str = carma_miriad.print_phase_center_info(
        print_table=False, return_str=True, hms_format=True
    )
    assert table_str == check_str


@pytest.mark.parametrize(
    "name,stype,arg_dict,exp_id,exp_diffs",
    (
        ["zenith", None, {}, 0, 4],
        ["zenith", "driftscan", {}, 0, 1],
        ["zenith", "unphased", {}, 0, 0],
        ["unphased", "unphased", {}, None, 0],
        ["unphased", "unphased", {"ignore_name": True}, 0, 0],
        ["zenith", "unphased", {"lat": 1.0}, 0, 1],
        ["zenith", "unphased", {"lon": 1.0}, 0, 1],
        ["zenith", "unphased", {"frame": 1.0}, 0, 1],
        ["zenith", "unphased", {"epoch": 1.0}, 0, 1],
        ["zenith", "unphased", {"times": 1.0}, 0, 1],
        ["zenith", "unphased", {"pm_ra": 1.0}, 0, 1],
        ["zenith", "unphased", {"pm_dec": 1.0}, 0, 1],
        ["zenith", "unphased", {"dist": 1.0}, 0, 1],
        ["zenith", "unphased", {"vrad": 1.0}, 0, 1],
    ),
)
def test_look_in_catalog(hera_uvh5, name, stype, arg_dict, exp_id, exp_diffs):
    """
    Test some basic functions of _look_in_catalog and check that it finds the
    appropriate phase center ID and number of differences between the provided
    parameters and that recorded in the UVData object.
    """
    [cat_id, num_diffs] = hera_uvh5._look_in_catalog(
        name,
        cat_type=stype,
        cat_lon=arg_dict.get("lon"),
        cat_lat=arg_dict.get("lat"),
        cat_frame=arg_dict.get("frame"),
        cat_epoch=arg_dict.get("epoch"),
        cat_times=arg_dict.get("times"),
        cat_pm_ra=arg_dict.get("pm_ra"),
        cat_pm_dec=arg_dict.get("pm_dec"),
        cat_dist=arg_dict.get("dist"),
        cat_vrad=arg_dict.get("vrad"),
        ignore_name=arg_dict.get("ignore_name"),
    )
    assert (cat_id is None) == (exp_id is None)
    if cat_id is not None:
        assert cat_id == exp_id
    assert num_diffs == exp_diffs


def test_look_in_catalog_phase_dict(sma_mir):
    """
    Use the phase_dict argument for _look_in_catalog and make sure that things
    behave as expected
    """
    # Now try lookup using a dictionary of properties
    assert sma_mir._look_in_catalog("3c84") == (1, 5)
    phase_dict = sma_mir.phase_center_catalog["3c84"]
    assert sma_mir._look_in_catalog("3c84", phase_dict=phase_dict) == (1, 0)

    # Make sure that if we set ignore_name, we still get a match
    assert sma_mir._look_in_catalog(
        "3c84", phase_dict=phase_dict, ignore_name=True
    ) == (1, 0)

    # Match w/ a mis-capitalization
    assert sma_mir._look_in_catalog(
        "3C84", phase_dict=phase_dict, ignore_name=True
    ) == (1, 0)


def test_add_phase_center_no_multi_phase(hera_uvh5):
    """
    Check that _add_phase_center throws an approrpriate error when called on an
    object where multi_phase_center=False.
    """
    with pytest.raises(
        ValueError, match="Cannot add a source if multi_phase_center != True.",
    ):
        hera_uvh5._add_phase_center("unphased", cat_type="unphased")


def test_remove_phase_center_no_multi_phase(hera_uvh5):
    """
    Check that _remove_phase_center throws an approrpriate error when called on an
    object where multi_phase_center=False.
    """
    with pytest.raises(
        ValueError, match="Cannot remove a phase center if multi_phase_center != True",
    ):
        hera_uvh5._remove_phase_center("zenith")


def test_clear_phase_centers_no_multi_phase(hera_uvh5):
    """
    Check that _clear_unused_phase_centers throws an approrpriate error when called on
    an object where multi_phase_center=False.
    """
    with pytest.raises(
        ValueError, match="Cannot remove a phase center if multi_phase_center != True",
    ):
        hera_uvh5._clear_unused_phase_centers()


def test_split_phase_center_no_multi_phase(hera_uvh5):
    """
    Check that split_phase_center throws an approrpriate error when called on
    an object where multi_phase_center=False.
    """
    with pytest.raises(
        ValueError,
        match="Cannot use split_phase_center on a non-multi phase center data set.",
    ):
        hera_uvh5.split_phase_center("zenith", "zenith", None)


def test_merge_phase_centers_no_multi_phase(hera_uvh5):
    """
    Check that merge_phase_centers throws an approrpriate error when called on
    an object where multi_phase_center=False.
    """
    with pytest.raises(
        ValueError,
        match="Cannot use merge_phase_centers on a non-multi phase center data set.",
    ):
        hera_uvh5.merge_phase_centers("zenith", "zenith")


def test_rename_phase_centers_no_multi_phase(hera_uvh5):
    """
    Check that rename_phase_centers throws an approrpriate error when called on
    an object where multi_phase_center=False.
    """
    with pytest.raises(
        ValueError, match="Cannot rename a phase center if multi_phase_center != True."
    ):
        hera_uvh5.rename_phase_center("zenith", "unphased")


def test_update_id_no_multi_phase(hera_uvh5):
    """
    Check that _update_phase_center_id throws an approrpriate error when called on
    an object where multi_phase_center=False.
    """
    with pytest.raises(
        ValueError,
        match="Cannot use _update_phase_center_id on a non-multi phase center data",
    ):
        hera_uvh5._update_phase_center_id("test")


@pytest.mark.parametrize(
    "name,stype,arg_dict,msg",
    (
        [-1, "drift", {}, "cat_name must be a string."],
        ["unphased", "drift", {}, "The name unphased is reserved."],
        ["unphased", "drift", {}, "The name unphased is reserved."],
        ["zenith", "drift", {}, "Only sidereal, ephem, driftscan or unphased may"],
        ["zenith", "driftscan", {"pm_ra": 0, "pm_dec": 0}, "Non-zero proper motion"],
        ["unphased", "unphased", {"lon": 1}, "Catalog entries that are unphased"],
        ["unphased", "unphased", {"lat": 1}, "Catalog entries that are unphased"],
        ["unphased", "unphased", {"frame": "fk5"}, "cat_frame must be either None"],
        ["test", "ephem", {}, "cat_times cannot be None for ephem object."],
        [
            "test",
            "ephem",
            {"lon": 0, "lat": 0, "frame": "icrs", "times": [0, 1]},
            "Object properties -- lon, lat, pm_ra, pm_dec, dist, vrad",
        ],
        ["test", "sidereal", {"pm_ra": 0}, "Must supply values for either both or"],
        ["test", "sidereal", {"pm_dec": 0}, "Must supply values for either both or"],
        ["test", "sidereal", {"times": 0}, "cat_times cannot be used for non-ephem"],
        ["test", "sidereal", {}, "cat_lon cannot be None for sidereal phase centers."],
        ["test", "sidereal", {"lon": 0}, "cat_lat cannot be None for sidereal"],
        ["test", "sidereal", {"lon": 0, "lat": 0}, "cat_frame cannot be None"],
        [
            "3c84",
            "sidereal",
            {"lat": 0, "lon": 0, "frame": "fk4", "epoch": "B1950.0"},
            "Cannot add different source with an non-unique name.",
        ],
        ["unphased", "unphased", {"id": 1}, "Provided cat_id belongs to another"],
    ),
)
def test_add_phase_center_arg_errs(sma_mir, name, stype, arg_dict, msg):
    with pytest.raises(ValueError) as cm:
        sma_mir._add_phase_center(
            name,
            cat_type=stype,
            cat_lon=arg_dict.get("lon"),
            cat_lat=arg_dict.get("lat"),
            cat_frame=arg_dict.get("frame"),
            cat_epoch=arg_dict.get("epoch"),
            cat_times=arg_dict.get("times"),
            cat_pm_ra=arg_dict.get("pm_ra"),
            cat_pm_dec=arg_dict.get("pm_dec"),
            cat_dist=arg_dict.get("dist"),
            cat_vrad=arg_dict.get("vrad"),
            force_update=arg_dict.get("force"),
            cat_id=arg_dict.get("id"),
        )
    assert str(cm.value).startswith(msg)


def test_add_phase_center_known_source(sma_mir):
    """
    Verify that if we attempt to add a source already in the catalog, we don't return
    an error but instead the call completes normally.
    """
    return_id = sma_mir._add_phase_center(
        "3c84",
        cat_type="sidereal",
        cat_lon=0.8718035968995141,
        cat_lat=0.7245157752262148,
        cat_frame="fk5",
        cat_epoch="j2000",
    )

    assert return_id == 1


def test_remove_phase_center_arg_errs(sma_mir):
    """
    Verify that _remove_phase_center throws errors appropriately when supplied with
    bad arguments.
    """
    # Only one bad argument to check, so no use parametrizing it
    with pytest.raises(
        IndexError, match="No source by that name contained in the catalog."
    ):
        sma_mir._remove_phase_center("zenith")


def test_clear_unused_phase_centers_no_op(sma_mir):
    """
    Verify that _clear_unused_phase_centers does nothing if no unused phase
    centers exist
    """
    check_dict = sma_mir.phase_center_catalog.copy()
    # Check and see that clearing out the unused objects doesn't actually change the
    # phase_center_catalog (because all objects are being "used").
    sma_mir._clear_unused_phase_centers()
    assert sma_mir.phase_center_catalog == check_dict


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.parametrize(
    "name1,name2,err_type,msg",
    (
        ["abc", "xyz", ValueError, "No entry by the name abc in the catalog."],
        ["3C273", -2, TypeError, "Value provided to new_name must be a string"],
        ["3C273", "unphased", ValueError, "The name unphased is reserved."],
        ["3C273", "NOISE", ValueError, "Must include a unique name for new_name"],
    ),
)
def test_rename_phase_center_bad_args(carma_miriad, name1, name2, err_type, msg):
    """
    Verify that rename_phase_center will throw appropriate errors when supplying
    bad arguments to the method.
    """
    pytest.importorskip("pyuvdata._miriad")
    with pytest.raises(err_type) as cm:
        carma_miriad.rename_phase_center(name1, name2)
    assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.parametrize(
    "name1,name2,mask,err_type,msg",
    (
        ["abc", "xyz", 1, ValueError, "No entry by the name abc in the catalog."],
        ["3C273", -2, 1, TypeError, "Value provided to new_name must be a string"],
        ["3C273", "unphased", 1, ValueError, "The name unphased is reserved."],
        ["3C273", "3C273", 1, ValueError, "The name 3C273 is already found"],
        ["3C273", "3c273", 1.5, IndexError, "select_mask must be an array-like,"],
        ["3C273", "3c273", 1, ValueError, "Data selected with select_mask includes"],
    ),
)
def test_split_phase_center_bad_args(carma_miriad, name1, name2, mask, err_type, msg):
    """
    Verify that split_phase_center will throw an error if supplied with bad args
    """
    with pytest.raises(err_type) as cm:
        carma_miriad.split_phase_center(name1, name2, mask)
    assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.parametrize(
    "name1,name2,err_type,msg",
    (
        ["3C273", "dummy1", ValueError, "No entry by the name dummy1 in the catalog"],
        ["dummy2", "3C273", ValueError, "No entry by the name dummy2 in the catalog"],
        ["3C273", "NOISE", ValueError, "Attributes of 3C273 and NOISE differ"],
    ),
)
def test_merge_phase_centers_bad_args(carma_miriad, name1, name2, err_type, msg):
    """
    Verify that merge_phase_centers will throw an error if supplied with bad args
    """
    pytest.importorskip("pyuvdata._miriad")
    with pytest.raises(err_type) as cm:
        carma_miriad.merge_phase_centers(name1, name2)
    assert str(cm.value).startswith(msg)


@pytest.mark.parametrize(
    "name,cat_id,res_id,err_type,msg",
    (
        ["abc", 0, 0, ValueError, "Cannot run _update_phase_center_id: no entry"],
        ["3c84", 0, [0], ValueError, "Catalog ID supplied already taken by another"],
    ),
)
def test_update_id_bad_args(sma_mir, name, cat_id, res_id, err_type, msg):
    """
    Verify that _update_phase_center_id throws errors when supplied with bad args
    """
    with pytest.raises(err_type) as cm:
        sma_mir._update_phase_center_id(name, new_cat_id=cat_id, reserved_ids=res_id)
    assert str(cm.value).startswith(msg)


def test_add_clear_phase_center(sma_mir):
    """
    Test that we can add a phase center, verify that we can find it correctly in the
    catalog, and then clear it as its unused.
    """
    check_dict = sma_mir.phase_center_catalog.copy()
    check_id = sma_mir._add_phase_center(
        "Mars",
        cat_type="ephem",
        cat_lon=[0.0, 1.0],
        cat_lat=[0, 1],
        cat_dist=(0, 1),
        cat_vrad=np.array([0, 1], dtype=np.float32),
        cat_times=np.array([0.0, 1.0]),
        cat_frame="icrs",
    )

    # Make sure the catalog ID returns as expected, and that the catalog changed
    assert check_id == 0
    # Check to see that the catalog actually changed
    assert sma_mir.phase_center_catalog != check_dict
    # And ake sure we can ID by name, but find diffs if attributes dont match
    assert sma_mir._look_in_catalog("Mars", cat_lon=[0], cat_lat=[0]) == (0, 7)

    # Finally, clear out the unused entries and check for equivalency w/ the old catalog
    sma_mir._clear_unused_phase_centers()
    assert sma_mir.phase_center_catalog == check_dict


def test_rename_object_capitalization(sma_mir, sma_mir_catalog):
    """
    Verify that renaming works, and that its case sensitive
    """
    # Check and see what happens if we attempt to rename the source
    sma_mir.rename_phase_center("3c84", "3C84")
    assert sma_mir.phase_center_catalog["3C84"] == sma_mir_catalog["3c84"]
    assert list(sma_mir.phase_center_catalog.keys()) == ["3C84"]

    sma_mir.rename_phase_center("3C84", "3c84")
    assert sma_mir.phase_center_catalog == sma_mir_catalog
    assert list(sma_mir.phase_center_catalog.keys()) == ["3c84"]


def test_rename_no_ops(sma_mir, sma_mir_catalog):
    """
    Verify that renaming the phase center with the same name results in no changes
    """
    # Check to make sure that setting the same name doesn't harm anything
    sma_mir.rename_phase_center("3c84", "3c84")
    assert sma_mir.phase_center_catalog == sma_mir_catalog
    assert list(sma_mir.phase_center_catalog.keys()) == ["3c84"]


def test_update_id_no_op(sma_mir, sma_mir_catalog):
    """
    Verify that updating the ID of a source without any ID conflicts results in no
    changes to the catalog
    """
    # This should effectively be a no-op, since the catalog ID of the source isn't
    # being taken up by anything else
    sma_mir._update_phase_center_id("3c84")
    assert sma_mir.phase_center_catalog == sma_mir_catalog


def test_update_id(sma_mir):
    """
    Verify that calling _update_phase_center_id will produce the lowest available
    positive int as the new ID for the source being updated.
    """
    # If all goes well, this operation should assign the lowest possible integer to the
    # catalog ID of 3c84 -- in this case, 4.
    sma_mir._update_phase_center_id("3c84", reserved_ids=[0, 1, 2, 3])
    assert sma_mir.phase_center_catalog["3c84"]["cat_id"] == 4


@pytest.mark.parametrize(
    "name1,name2,select_mask,msg",
    (
        ["3c84", "3C84", False, "No relevant data selected"],
        ["3c84", "3C84", True, "All data for 3c84 selected"],
    ),
)
def test_split_phase_center_warnings(sma_mir, name1, name2, select_mask, msg):
    # Now let's select no data at all
    with uvtest.check_warnings(UserWarning, msg):
        sma_mir.split_phase_center(name1, name2, select_mask)


def test_split_phase_center(hera_uvh5):
    # Set the HERA file as multi phase center so that we can play around with it a bit
    hera_uvh5._set_multi_phase_center(preserve_phase_center_info=True)

    # Alright, now let's actually try to split the sources -- let's say every other
    # integration?
    select_mask = np.isin(hera_uvh5.time_array, np.unique(hera_uvh5.time_array)[::2])

    hera_uvh5.split_phase_center("zenith", "zenith2", select_mask)

    # Check that the catalog IDs also line up w/ what we expect
    obj_id_check = hera_uvh5.phase_center_catalog["zenith"]["cat_id"]
    assert np.all(hera_uvh5.phase_center_id_array[~select_mask] == obj_id_check)
    obj_id_check = hera_uvh5.phase_center_catalog["zenith2"]["cat_id"]
    assert np.all(hera_uvh5.phase_center_id_array[select_mask] == obj_id_check)

    # Make sure the catalog makes sense -- entries should be identical sans cat_id
    temp_cat = hera_uvh5.phase_center_catalog.copy()
    assert temp_cat["zenith"]["cat_id"] != temp_cat["zenith2"]["cat_id"]
    temp_cat["zenith"]["cat_id"] = temp_cat["zenith2"]["cat_id"]
    assert temp_cat["zenith"] == temp_cat["zenith2"]

    # Finally, verify the phase center names
    sorted_names = sorted(hera_uvh5.phase_center_catalog.keys())
    assert sorted_names == sorted(["zenith", "zenith2"])
    assert hera_uvh5.Nphase == 2


def test_split_phase_center_downselect(hera_uvh5):
    # Set the HERA file as multi phase center so that we can play around with it a bit
    hera_uvh5._set_multi_phase_center(preserve_phase_center_info=True)
    catalog_copy = hera_uvh5.phase_center_catalog.copy()

    # Again, only select the first half of the data
    select_mask = np.isin(hera_uvh5.time_array, np.unique(hera_uvh5.time_array)[::2])
    hera_uvh5.split_phase_center("zenith", "zenith2", select_mask)

    # Now effectively rename zenith2 as zenith3 by selecting all data and using
    # the downselect switch
    with uvtest.check_warnings(UserWarning, "All data for zenith2 selected"):
        hera_uvh5.split_phase_center(
            "zenith2", "zenith3", np.arange(hera_uvh5.Nblts), downselect=True
        )

    obj_id_check = hera_uvh5.phase_center_catalog["zenith"]["cat_id"]
    assert np.all(hera_uvh5.phase_center_id_array[~select_mask] == obj_id_check)
    obj_id_check = hera_uvh5.phase_center_catalog["zenith3"]["cat_id"]
    assert np.all(hera_uvh5.phase_center_id_array[select_mask] == obj_id_check)

    sorted_names = sorted(hera_uvh5.phase_center_catalog.keys())
    assert sorted_names == sorted(["zenith", "zenith3"])

    # Make sure the dicts make sense
    temp_dict = hera_uvh5.phase_center_catalog["zenith"].copy()
    temp_dict2 = hera_uvh5.phase_center_catalog["zenith3"].copy()
    assert temp_dict["cat_id"] != temp_dict2["cat_id"]
    temp_dict["cat_id"] = temp_dict2["cat_id"]
    assert temp_dict == temp_dict2

    # Finally, force the two objects back to being one, despite the fact that we've
    # contaminated the dict of one (which will be overwritten by the other)
    hera_uvh5.phase_center_catalog["zenith3"]["cat_epoch"] = 2000.0
    with uvtest.check_warnings(UserWarning, "Forcing zenith and zenith3 together"):
        hera_uvh5.merge_phase_centers("zenith", "zenith3", force_merge=True)

    # We merged everything back together, so we _should_  get back the same
    # thing that we started with.
    assert hera_uvh5.phase_center_catalog == catalog_copy
    obj_id_check = hera_uvh5.phase_center_catalog["zenith"]["cat_id"]
    assert np.all(hera_uvh5.phase_center_id_array == obj_id_check)


@pytest.mark.parametrize(
    "val1,val2,val3,err_type,msg",
    [
        [0.0, 0.0, 1.5, IndexError, "select_mask must be an array-like, either of"],
        [[0.0, 0.0], 0.0, [0], IndexError, "The length of new_w_vals is wrong"],
        [0.0, [0.0, 0.0], [0], IndexError, "The length of old_w_vals is wrong"],
    ],
)
def test_apply_w_arg_errs(hera_uvh5, val1, val2, val3, err_type, msg):
    with pytest.raises(err_type) as cm:
        hera_uvh5._apply_w_proj(val1, val2, val3)
    assert str(cm.value).startswith(msg)


@pytest.mark.parametrize("future_shapes", [True, False])
def test_apply_w_no_ops(hera_uvh5, future_shapes):
    """
    Test to make sure that the _apply_w method throws  expected errors
    """
    if future_shapes:
        hera_uvh5.use_future_array_shapes()

    hera_copy = hera_uvh5.copy()

    # Test to make sure that the following gives us back the same results,
    # first without a selection mask
    hera_uvh5._apply_w_proj(0.0, 0.0)
    assert hera_uvh5 == hera_copy

    # And now with a selection mask applied
    hera_uvh5._apply_w_proj([0.0, 1.0], [0.0, 1.0], [0, 1])
    assert hera_uvh5 == hera_copy


def test_phase_dict_helper_simple(hera_uvh5, sma_mir, dummy_phase_dict):
    """
    Verify that _phase_dict_helper behaves appropriately when being handed a "typical"
    set of parameters, for a single-phase-ctr (hera_uvh5) and multi-phase-ctr (sma_mir)
    objects alike.
    """
    # If we could parameterize fixtures, I'd do that, but for now I'll use a simple
    # for loop to move through two different datasets.
    for uv_object in [hera_uvh5, sma_mir]:
        phase_dict = uv_object._phase_dict_helper(
            dummy_phase_dict["cat_lon"],
            dummy_phase_dict["cat_lat"],
            dummy_phase_dict["cat_epoch"],
            dummy_phase_dict["cat_frame"],
            dummy_phase_dict["cat_times"],
            dummy_phase_dict["cat_type"],
            dummy_phase_dict["cat_pm_ra"],
            dummy_phase_dict["cat_pm_dec"],
            dummy_phase_dict["cat_dist"],
            dummy_phase_dict["cat_vrad"],
            dummy_phase_dict["cat_name"],
            False,  # Don't lookup source
            None,  # Apply no mask
            None,  # Don't supply a time_array
        )
        assert phase_dict == dummy_phase_dict


@pytest.mark.parametrize(
    "arg_dict, msg",
    [
        [{"lookup": True}, "Unable to find z1 in among the existing sources recorded"],
        [
            {"cat_type": "ephem", "cat_epoch": None, "cat_times": 1, "time_arr": 0},
            "Ephemeris data does not cover",
        ],
        [
            {"cat_type": "ephem", "time_arr": 2456789, "lookup": True},
            "Target ID is not recognized in either the small or major",
        ],
        [{"cat_name": "3c84", "sel_mask": np.array([False])}, "The entry name 3c84 is"],
    ],
)
def test_phase_dict_helper_errs(sma_mir, arg_dict, dummy_phase_dict, msg):
    """
    Test the `_phase_dict_helper` method.

    Test the helper function that the `phase` method uses for looking up astronomical
    source information.
    """
    pytest.importorskip("astroquery")
    for key in dummy_phase_dict.keys():
        if key not in arg_dict.keys():
            arg_dict[key] = dummy_phase_dict[key]
    with pytest.raises(ValueError) as cm:
        sma_mir._phase_dict_helper(
            arg_dict["cat_lon"],
            arg_dict["cat_lat"],
            arg_dict["cat_epoch"],
            arg_dict["cat_frame"],
            arg_dict["cat_times"],
            arg_dict["cat_type"],
            arg_dict["cat_pm_ra"],
            arg_dict["cat_pm_dec"],
            arg_dict["cat_dist"],
            arg_dict["cat_vrad"],
            arg_dict["cat_name"],
            arg_dict.get("lookup"),
            arg_dict.get("sel_mask"),
            arg_dict.get("time_arr"),
        )
    assert str(cm.value).startswith(msg)


@pytest.mark.parametrize("sel_mask", [None, np.array([True])])
def test_phase_dict_helper_sidereal_no_lookup(sma_mir, dummy_phase_dict, sel_mask):
    """
    Verify that _phase_dict_helper will accept name collisions where all of the data
    phased to that named phase center is being selected (note that select_mask=None
    selects all of the data in the UVData object).
    """
    # Try looking up a name, where the properties are different but where we've selected
    # all of the data (via None for the select mask)
    phase_dict = sma_mir._phase_dict_helper(
        dummy_phase_dict["cat_lon"],
        dummy_phase_dict["cat_lat"],
        dummy_phase_dict["cat_epoch"],
        dummy_phase_dict["cat_frame"],
        dummy_phase_dict["cat_times"],
        dummy_phase_dict["cat_type"],
        dummy_phase_dict["cat_pm_ra"],
        dummy_phase_dict["cat_pm_dec"],
        dummy_phase_dict["cat_dist"],
        dummy_phase_dict["cat_vrad"],
        "3c84",
        False,  # Do lookup source!
        sel_mask,  # Apply no mask
        None,  # Don't supply a time_array
    )

    assert phase_dict["cat_name"] == "3c84"
    phase_dict["cat_name"] = "z1"
    phase_dict["cat_id"] = None
    assert phase_dict == dummy_phase_dict


def test_phase_dict_helper_sidereal_lookup(sma_mir, dummy_phase_dict):
    """
    Check that we can use the lookup option to find a sidereal source properties in
    a multi-phase-ctr dataset.
    """
    phase_dict = sma_mir._phase_dict_helper(
        dummy_phase_dict["cat_lon"],
        dummy_phase_dict["cat_lat"],
        dummy_phase_dict["cat_epoch"],
        dummy_phase_dict["cat_frame"],
        dummy_phase_dict["cat_times"],
        dummy_phase_dict["cat_type"],
        dummy_phase_dict["cat_pm_ra"],
        dummy_phase_dict["cat_pm_dec"],
        dummy_phase_dict["cat_dist"],
        dummy_phase_dict["cat_vrad"],
        "3c84",
        True,  # Do lookup source!
        None,  # Apply no mask
        None,  # Don't supply a time_array
    )
    assert phase_dict.pop("cat_name") == "3c84"
    assert phase_dict == sma_mir.phase_center_catalog["3c84"]
    # Check that even if we force the names to match, the catalogs are different, i.e.
    # the dummy dict was ignored upon lookup.
    phase_dict["cat_name"] = dummy_phase_dict["cat_name"]
    assert phase_dict != dummy_phase_dict


def test_phase_dict_helper_jpl_lookup_existing(sma_mir):
    """
    Verify that the _phase_dict_helper function correctly hands back a dict that
    matches that in the catalog, provided the source properties match.
    """
    # Finally, check that we get a good result if feeding the same values, even if not
    # actually performing a lookup
    phase_dict = sma_mir._phase_dict_helper(
        sma_mir.phase_center_catalog["3c84"].get("cat_lon"),
        sma_mir.phase_center_catalog["3c84"].get("cat_lat"),
        sma_mir.phase_center_catalog["3c84"].get("cat_epoch"),
        sma_mir.phase_center_catalog["3c84"].get("cat_frame"),
        sma_mir.phase_center_catalog["3c84"].get("cat_times"),
        sma_mir.phase_center_catalog["3c84"].get("cat_type"),
        sma_mir.phase_center_catalog["3c84"].get("cat_pm_ra"),
        sma_mir.phase_center_catalog["3c84"].get("cat_pm_dec"),
        sma_mir.phase_center_catalog["3c84"].get("cat_dist"),
        sma_mir.phase_center_catalog["3c84"].get("cat_vrad"),
        "3c84",
        False,
        None,
        sma_mir.time_array,
    )
    assert phase_dict.pop("cat_name") == "3c84"
    assert phase_dict == sma_mir.phase_center_catalog["3c84"]


def test_phase_dict_helper_jpl_lookup_append(sma_mir):
    """
    Test _phase_dict_helper to see if it will correctly call the JPL lookup when
    an old ephem does not cover the newly requested time range
    """
    pytest.importorskip("astroquery")
    # Now see what happens if we attempt to lookup something that JPL actually knows
    obs_time = np.array(2456789.0)
    phase_dict = sma_mir._phase_dict_helper(
        0, 0, None, None, None, None, 0, 0, 0, 0, "Mars", True, None, obs_time,
    )

    cat_id = sma_mir._add_phase_center(
        phase_dict["cat_name"],
        phase_dict["cat_type"],
        cat_lon=phase_dict["cat_lon"],
        cat_lat=phase_dict["cat_lat"],
        cat_frame=phase_dict["cat_frame"],
        cat_epoch=phase_dict["cat_epoch"],
        cat_times=phase_dict["cat_times"],
        cat_pm_ra=phase_dict["cat_pm_ra"],
        cat_pm_dec=phase_dict["cat_pm_dec"],
        cat_dist=phase_dict["cat_dist"],
        cat_vrad=phase_dict["cat_vrad"],
        info_source=phase_dict["info_source"],
        force_update=True,
    )

    # By default, the catalog ID here should be zero (lowest available ID)
    assert cat_id == 0

    # Tick the obs_time up by a day, see if the software will fetch additional
    # coordinates and expand the existing ephem
    obs_time += 1
    phase_dict = sma_mir._phase_dict_helper(
        0, 0, None, None, None, None, 0, 0, 0, 0, "Mars", True, None, obs_time,
    )

    # Previously, everything else will have had a single point, but the new ephem (which
    # covers 36 hours at 3 hour intervals) should have a lucky total of 13 points.
    keycheck = ["cat_lon", "cat_lat", "cat_vrad", "cat_dist", "cat_times"]
    for key in keycheck:
        assert len(phase_dict[key]) == 13


def test_fix_phase_multi_phase_err(sma_mir):
    """
    Verify that running fix_phase with the baselines-only method on multi-phase-ctr
    datasets throws an appropriate error
    """
    # Check the one error condition that fix_phase raises
    with pytest.raises(ValueError, match="Cannot run fix_phase on a mutli-phase-ctr"):
        sma_mir.fix_phase(use_ant_pos=False)


@pytest.mark.filterwarnings("ignore:The original `phase` method is deprecated")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("use_ant_pos", [True, False])
def test_fix_phase(hera_uvh5, future_shapes, use_ant_pos):
    """
    Test the phase fixing method fix_phase
    """
    if future_shapes:
        hera_uvh5.use_future_array_shapes()

    # Make some copies of the data
    uv_in = hera_uvh5.copy()
    uv_in_bad = hera_uvh5

    # These values could be anything -- we're just picking something that we know should
    # be visible from the telescope at the time of obs (ignoring horizon limits).
    phase_ra = uv_in.lst_array[-1]
    phase_dec = uv_in.telescope_location_lat_lon_alt[0] * 0.333

    # Do the improved phasing on the dat set.
    uv_in.phase(phase_ra, phase_dec)

    # First test the case where we are using the old phase method with the uvws
    # calculated from the antenna positions. Using fix phase here should be "perfect",
    # since the uvws are completely recalculated from scratch.
    uv_in_bad.phase(phase_ra, phase_dec, use_old_proj=True, use_ant_pos=use_ant_pos)
    uv_in_bad.fix_phase(use_ant_pos=use_ant_pos)

    # We have to handle this case a little carefully, because since the old
    # unphase_to_drift was _mostly_ accurate, although it does seem to intoduce errors
    # on the order of a part in 1e5, which translates to about a tenth of a degree phase
    # error in the test data set used here. Check that first, make sure it's good
    assert np.allclose(uv_in.data_array, uv_in_bad.data_array, rtol=3e-4)

    # Once we know the data are okay, copy over data array and check for equality btw
    # the other attributes of the two objects.
    uv_in_bad.data_array = uv_in.data_array
    assert uv_in == uv_in_bad


@pytest.mark.parametrize("future_shapes", [True, False])
def test_multi_file_ignore_name(hera_uvh5_split, future_shapes):
    """
    Verify that if phased two objects to the same position with different names, we
    can successfully use the "ignore_name" switch in the add operation to allow
    the two objects to be combined.
    """
    uv1, uv2, uvfull = hera_uvh5_split
    if future_shapes:
        uv1.use_future_array_shapes()
        uv2.use_future_array_shapes()
        uvfull.use_future_array_shapes()

    # Phase both targets to the same position with different names
    uv1.phase(3.6, -0.5, cat_name="target1")
    uv2.phase(3.6, -0.5, cat_name="target2")
    uvfull.phase(3.6, -0.5, cat_name="target1")

    # Catch the obvious error
    with pytest.raises(ValueError, match="UVParameter object_name does not match."):
        _ = uv1 + uv2

    # Now ignore the obvious error!
    uv3 = uv1.__add__(uv2, ignore_name=True, inplace=False)
    # The reorders here are neccessary after the add to make sure that the baseline
    # ordering is consistent between the objects
    uv3.reorder_blts()
    uvfull.reorder_blts()

    # Make sure that after the add, everything agrees
    assert uvfull.history in uv3.history
    uvfull.history = uv3.history
    assert uvfull == uv3


@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    "add_method", [["__add__", {}], ["fast_concat", {"axis": "blt"}]],
)
def test_multi_phase_add_ops(hera_uvh5_split, future_shapes, add_method):
    """
    Verify that both add operations work correctly with multi-phase-ctr objects.
    """
    uv1, uv2, uvfull = hera_uvh5_split

    uv1._set_multi_phase_center(preserve_phase_center_info=True)
    uv2._set_multi_phase_center(preserve_phase_center_info=True)
    uvfull._set_multi_phase_center(preserve_phase_center_info=True)

    if future_shapes:
        uv1.use_future_array_shapes()
        uv2.use_future_array_shapes()
        uvfull.use_future_array_shapes()

    # Phase both targets to the same position with different names
    uv1.phase(3.6, -0.5, cat_name="target1")
    uv2.phase(3.6, -0.5, cat_name="target1")
    uvfull.phase(3.6, -0.5, cat_name="target1")

    uv3 = getattr(uv1, add_method[0])(uv2, **add_method[1])

    # The reorders here are neccessary after the add to make sure that the baseline
    # ordering is consistent between the objects
    uv3.reorder_blts()
    uvfull.reorder_blts()

    # Make sure that after the add, everything agrees
    assert uvfull.history in uv3.history
    uvfull.history = uv3.history
    assert uvfull == uv3


@pytest.mark.parametrize(
    "mpc1,mpc2,msg",
    [
        [False, True, "To combine these data, please run the add operation with"],
        [True, False, "There exists a target named target1 in"],
        [True, True, "There exists a target named target1 in"],
    ],
)
def test_multi_phase_add_errs(hera_uvh5_split, mpc1, mpc2, msg):
    """
    """
    uv1, uv2, _ = hera_uvh5_split
    if mpc1:
        uv1._set_multi_phase_center(preserve_phase_center_info=True)
    if mpc2:
        uv2._set_multi_phase_center(preserve_phase_center_info=True)

    uv1.phase(3.6, -0.5, cat_name="target1")
    uv2.phase(-0.5, 3.6, cat_name="target1")

    with pytest.raises(ValueError, match=msg):
        _ = uv1 + uv2


@pytest.mark.parametrize("test_op", [None, "split", "rename", "merge"])
def test_multi_phase_split_merge_rename(hera_uvh5_split, test_op):
    """
    Test the split, merge, and rename operations, and make sure their operations
    are internally consistent.
    """
    uv1, uv2, uvfull = hera_uvh5_split
    half_mask = np.arange(uvfull.Nblts) < (uvfull.Nblts * 0.5)

    uv1._set_multi_phase_center(preserve_phase_center_info=True)
    uv2._set_multi_phase_center(preserve_phase_center_info=True)
    uvfull._set_multi_phase_center(preserve_phase_center_info=True)

    uv1.phase(3.6, -0.5, cat_name="target1")
    uv2.phase(3.6, -0.5, cat_name="target1" if (test_op is None) else "target2")
    uv3 = uv1 + uv2
    uv3.reorder_blts()

    uvfull.reorder_blts()
    uvfull.phase(3.6, -0.5, cat_name="target1")
    uvfull._update_phase_center_id("target1", 1 if (test_op is None) else 0)

    # Any of these operations should allow for the objects to become equal to the
    # other -- they're basically the inverse action taken on two different objects.
    if test_op is None:
        # Nothing to do here -- this should be an apples-to-apples comparison without
        # any renaming operations.
        pass
    if test_op == "split":
        uvfull.split_phase_center("target1", "target2", ~half_mask)
    elif test_op == "rename":
        uv3.merge_phase_centers("target2", "target1")
        uvfull.rename_phase_center("target1", "target2")
        uvfull._update_phase_center_id("target2", 1)
    elif test_op == "merge":
        uv3.merge_phase_centers("target1", "target2")

    assert uvfull.history in uv3.history
    uvfull.history = uv3.history
    assert uvfull == uv3


@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    "mpc1,mpc2,catid",
    [[False, False, 0], [True, False, 0], [True, True, 0], [True, True, 1]],
)
def test_multi_phase_add(hera_uvh5_split, mpc1, mpc2, catid, future_shapes):
    uv1, uv2, uvfull = hera_uvh5_split
    if future_shapes:
        uv1.use_future_array_shapes()
        uv2.use_future_array_shapes()
        uvfull.use_future_array_shapes()

    if mpc1:
        uv1._set_multi_phase_center(preserve_phase_center_info=True)
    if mpc2:
        uv2._set_multi_phase_center(preserve_phase_center_info=True)
    uvfull._set_multi_phase_center(preserve_phase_center_info=True)

    # Give it a new name, and then rephase half of the "full" object
    uv1.phase(3.6, -0.5, cat_name="target1")
    uv2.phase(-0.5, 3.6, cat_name="target2")

    # Test that addition handles cat ID collisions correctly
    if mpc2:
        uv2._update_phase_center_id("target2", catid)

    # Add the objects together
    uv3 = uv1.__add__(uv2, make_multi_phase=True)
    uv3.reorder_blts()

    # Separately phase both halves of the full data set
    half_mask = np.arange(uvfull.Nblts) < (uvfull.Nblts * 0.5)
    uvfull.phase(-0.5, 3.6, cat_name="target2", select_mask=~half_mask)
    uvfull.phase(3.6, -0.5, cat_name="target1", select_mask=half_mask)
    uvfull.reorder_blts()

    # Check that the histories line up
    assert uvfull.history in uv3.history
    uvfull.history = uv3.history

    # By construct, we've made it so that the cat IDs don't line up, but everything
    # else should. Make sure the IDs and catalogs are different, but contain the
    # same names for the phase centers
    assert np.any(uv3.phase_center_id_array != uvfull.phase_center_id_array)
    assert uv3.phase_center_catalog != uvfull.phase_center_catalog
    uvfull_names = sorted(uvfull.phase_center_catalog.keys())
    uv3_names = sorted(uv3.phase_center_catalog.keys())
    assert uvfull_names == uv3_names

    # Update the Obs IDs, and make sure that _now_ the objects are equal
    uv3._update_phase_center_id("target2", 99)
    uv3._update_phase_center_id("target1", 2)
    uv3._update_phase_center_id("target2", 1)
    assert uv3 == uvfull


def test_multi_phase_on_read(hera_uvh5):
    """"
    Verify that we can create a multi-phase-ctr object on read that matches what
    one would expect when running some of the various helper functions for converting
    a single phase center object into a multi-phase one.
    """
    # The file below is the same that's used for the hera_uvh5 fixture
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
    uv_object.read(testfile, make_multi_phase=True)

    # Marking the file as multi-phase and calculating the apparent coords are the
    # two major things required for replicating the "make_multi_phase" operation above
    hera_uvh5._set_multi_phase_center(preserve_phase_center_info=True)
    hera_uvh5._set_app_coords_helper()

    # These two objects should be identical
    assert uv_object == hera_uvh5


@pytest.mark.parametrize("future_shapes", [True, False])
def test_multi_phase_downselect(hera_uvh5_split, future_shapes):
    """
    Verify that we can create the same UVData object if we phase then downselect
    vs downselect and phase when working with a multi-phase-ctr object.
    """
    uv1, uv2, uvfull = hera_uvh5_split
    if future_shapes:
        uv1.use_future_array_shapes()
        uv2.use_future_array_shapes()
        uvfull.use_future_array_shapes()

    uv1._set_multi_phase_center(preserve_phase_center_info=True)
    uv2._set_multi_phase_center(preserve_phase_center_info=True)
    uvfull._set_multi_phase_center(preserve_phase_center_info=True)

    # Give it a new name, and then rephase half of the "full" object
    uv1.phase(3.6, -0.5, cat_name="target1")
    uv1.reorder_blts()
    uv2.phase(-0.5, 3.6, cat_name="target2")
    uv2.reorder_blts()

    # Separately phase both halves of the full data set
    half_mask = np.arange(uvfull.Nblts) < (uvfull.Nblts * 0.5)
    unique_times = np.unique(uvfull.time_array)

    uvfull.phase(-0.5, 3.6, cat_name="target2", select_mask=~half_mask)
    uvfull.phase(3.6, -0.5, cat_name="target1", select_mask=half_mask)
    for mask, uvdata in zip([np.arange(10), np.arange(10, 20)], [uv1, uv2]):
        uvtemp = uvfull.select(times=unique_times[mask], inplace=False)
        uvtemp.reorder_blts()
        # Select does not clear the catalog, so clear the unused source and
        # update the cat ID so that it matches with the indv datasets
        uvtemp._clear_unused_phase_centers()
        uvtemp._update_phase_center_id(list(uvtemp.phase_center_catalog.keys())[0], 1)
        assert uvtemp.history in uvdata.history
        uvtemp.history = uvdata.history
        assert uvtemp == uvdata


@pytest.mark.filterwarnings("ignore:Unknown phase types are no longer supported")
@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
def test_eq_allowed_failures(bda_test_file, capsys):
    """
    Test that the allowed_failures keyword on the __eq__ method works as intended.
    """
    uv1 = bda_test_file
    uv2 = uv1.copy()

    # adjust optional parameters to be different
    uv1.x_orientation = "NORTH"
    uv2.x_orientation = "EAST"
    assert uv1.__eq__(uv2, check_extra=True, allowed_failures=["x_orientation"])
    captured = capsys.readouterr()
    assert captured.out == (
        "x_orientation parameter value is a string, values are different\n"
        "parameter _x_orientation does not match, but is not required to for equality. "
        "Left is NORTH, right is EAST.\n"
    )

    # make sure that objects are not equal without specifying allowed_failures
    assert uv1 != uv2

    return


@pytest.mark.filterwarnings("ignore:Unknown phase types are no longer supported")
@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
def test_eq_allowed_failures_filename(bda_test_file, capsys):
    """
    Test that the `filename` parameter does not trip up the __eq__ method.
    """
    uv1 = bda_test_file
    uv2 = uv1.copy()

    uv1.filename = ["foo.uvh5"]
    uv2.filename = ["bar.uvh5"]
    assert uv1 == uv2
    captured = capsys.readouterr()
    assert captured.out == (
        "filename parameter value is a list of strings, values are different\n"
        "parameter _filename does not match, but is not required to for equality. "
        "Left is ['foo.uvh5'], right is ['bar.uvh5'].\n"
    )

    return


@pytest.mark.filterwarnings("ignore:Unknown phase types are no longer supported")
@pytest.mark.filterwarnings("ignore:Telescope mock-HERA is not in known_telescopes")
def test_eq_allowed_failures_filename_string(bda_test_file, capsys):
    """
    Try passing a string to the __eq__ method instead of an iterable.
    """
    uv1 = bda_test_file
    uv2 = uv1.copy()

    uv1.filename = ["foo.uvh5"]
    uv2.filename = ["bar.uvh5"]
    assert uv1.__eq__(uv2, allowed_failures="filename")
    captured = capsys.readouterr()
    assert captured.out == (
        "filename parameter value is a list of strings, values are different\n"
        "parameter _filename does not match, but is not required to for equality. "
        "Left is ['foo.uvh5'], right is ['bar.uvh5'].\n"
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_set_data(hera_uvh5, future_shapes):
    """
    Test setting data for a given baseline.
    """
    uv = hera_uvh5

    if future_shapes:
        uv.use_future_array_shapes()

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    data = 2 * uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    inds1, inds2, indp = uv._key2inds((ant1, ant2))
    uv.set_data(data, ant1, ant2)
    data2 = uv.get_data(ant1, ant2, squeeze="none")

    assert np.allclose(data, data2)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_set_data_evla(future_shapes):
    """
    Test setting data for a given baseline on a different test file.
    """
    filename = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uv = UVData()
    uv.read(filename)

    if future_shapes:
        uv.use_future_array_shapes()

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    data = 2 * uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    inds1, inds2, indp = uv._key2inds((ant1, ant2))
    uv.set_data(data, ant1, ant2)
    data2 = uv.get_data(ant1, ant2, squeeze="none")

    assert np.allclose(data, data2)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_set_data_polkey(hera_uvh5, future_shapes):
    """
    Test setting data for a given baseline with a specific polarization.
    """
    uv = hera_uvh5

    if future_shapes:
        uv.use_future_array_shapes()

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    pol = "xx"
    data = 2 * uv.get_data(ant1, ant2, pol, squeeze="none", force_copy=True)
    inds1, inds2, indp = uv._key2inds((ant1, ant2, pol))
    uv.set_data(data, ant1, ant2, pol)
    data2 = uv.get_data(ant1, ant2, pol, squeeze="none")

    assert np.allclose(data, data2)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_set_flags(hera_uvh5, future_shapes):
    """
    Test setting flags for a given baseline.
    """
    uv = hera_uvh5
    if future_shapes:
        uv.use_future_array_shapes()

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    flags = uv.get_flags(ant1, ant2, squeeze="none", force_copy=True)
    if future_shapes:
        flags[:, :, :] = True
    else:
        flags[:, :, :, :] = True
    inds1, inds2, indp = uv._key2inds((ant1, ant2))
    uv.set_flags(flags, ant1, ant2)
    flags2 = uv.get_flags(ant1, ant2, squeeze="none")

    assert np.allclose(flags, flags2)
    assert not np.allclose(uv.flag_array, True)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_set_flags_polkey(hera_uvh5, future_shapes):
    """
    Test setting flags for a given baseline with a specific polarization.
    """
    uv = hera_uvh5
    if future_shapes:
        uv.use_future_array_shapes()

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    pol = "xx"
    flags = uv.get_flags(ant1, ant2, pol, squeeze="none", force_copy=True)
    if future_shapes:
        flags[:, :, :] = True
    else:
        flags[:, :, :, :] = True
    inds1, inds2, indp = uv._key2inds((ant1, ant2, pol))
    uv.set_flags(flags, ant1, ant2, pol)
    flags2 = uv.get_flags(ant1, ant2, pol, squeeze="none")

    assert np.allclose(flags, flags2)
    assert not np.allclose(uv.flag_array, True)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_set_nsamples(hera_uvh5, future_shapes):
    """
    Test setting nsamples for a given baseline.
    """
    uv = hera_uvh5
    if future_shapes:
        uv.use_future_array_shapes()

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    nsamples = uv.get_nsamples(ant1, ant2, squeeze="none", force_copy=True)
    if future_shapes:
        nsamples[:, :, :] = np.pi
    else:
        nsamples[:, :, :, :] = np.pi
    inds1, inds2, indp = uv._key2inds((ant1, ant2))
    uv.set_nsamples(nsamples, ant1, ant2)
    nsamples2 = uv.get_nsamples(ant1, ant2, squeeze="none")

    assert np.allclose(nsamples, nsamples2)
    assert not np.allclose(uv.nsample_array, np.pi)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_set_nsamples_polkey(hera_uvh5, future_shapes):
    """
    Test setting nsamples for a given baseline with a specific polarization.
    """
    uv = hera_uvh5
    if future_shapes:
        uv.use_future_array_shapes()

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    pol = "xx"
    nsamples = uv.get_nsamples(ant1, ant2, pol, squeeze="none", force_copy=True)
    if future_shapes:
        nsamples[:, :, :] = np.pi
    else:
        nsamples[:, :, :, :] = np.pi
    inds1, inds2, indp = uv._key2inds((ant1, ant2, pol))
    uv.set_nsamples(nsamples, ant1, ant2, pol)
    nsamples2 = uv.get_nsamples(ant1, ant2, pol, squeeze="none")

    assert np.allclose(nsamples, nsamples2)
    assert not np.allclose(uv.nsample_array, np.pi)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_data_bad_key_error(hera_uvh5):
    """
    Test an error is raised when a key has too many values.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    data = uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    match = "no more than 3 key values can be passed"
    with pytest.raises(ValueError, match=match):
        uv.set_data(data, (ant1, ant2, "xx", "foo"))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_data_conj_data_error(hera_uvh5):
    """
    Test an error is raised when a conjugated baseline is specified.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    data = uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    match = "the requested key is present on the object, but conjugated"
    with pytest.raises(ValueError, match=match):
        uv.set_data(data, (ant2, ant1))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_data_wrong_shape_error(hera_uvh5):
    """
    Test an error is raised when the data are the wrong shape.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    data = uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    # make data the wrong rank
    data = data[0]
    match = "the input array is not compatible with the shape of the destination"
    with pytest.raises(ValueError, match=match):
        uv.set_data(data, (ant1, ant2))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_flags_bad_key_error(hera_uvh5):
    """
    Test an error is raised when a key has too many values.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    flags = uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    match = "no more than 3 key values can be passed"
    with pytest.raises(ValueError, match=match):
        uv.set_flags(flags, (ant1, ant2, "xx", "foo"))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_flags_conj_data_error(hera_uvh5):
    """
    Test an error is raised when a conjugated baseline is specified.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    flags = uv.get_flags(ant1, ant2, squeeze="none", force_copy=True)
    match = "the requested key is present on the object, but conjugated"
    with pytest.raises(ValueError, match=match):
        uv.set_flags(flags, (ant2, ant1))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_flags_wrong_shape_error(hera_uvh5):
    """
    Test an error is raised when the flags are the wrong shape.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    flags = uv.get_flags(ant1, ant2, squeeze="none", force_copy=True)
    # make data the wrong rank
    flags = flags[0]
    match = "the input array is not compatible with the shape of the destination"
    with pytest.raises(ValueError, match=match):
        uv.set_flags(flags, (ant1, ant2))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_nsamples_bad_key_error(hera_uvh5):
    """
    Test an error is raised when a key has too many values.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    nsamples = uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    match = "no more than 3 key values can be passed"
    with pytest.raises(ValueError, match=match):
        uv.set_nsamples(nsamples, (ant1, ant2, "xx", "foo"))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_nsamples_conj_data_error(hera_uvh5):
    """
    Test an error is raised when a conjugated baseline is specified.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    nsamples = uv.get_nsamples(ant1, ant2, squeeze="none", force_copy=True)
    match = "the requested key is present on the object, but conjugated"
    with pytest.raises(ValueError, match=match):
        uv.set_nsamples(nsamples, (ant2, ant1))

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_nsamples_wrong_shape_error(hera_uvh5):
    """
    Test an error is raised when the nsamples are the wrong shape.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.antenna_numbers)[0]
    ant2 = np.unique(uv.antenna_numbers)[1]
    nsamples = uv.get_nsamples(ant1, ant2, squeeze="none", force_copy=True)
    # make data the wrong rank
    nsamples = nsamples[0]
    match = "the input array is not compatible with the shape of the destination"
    with pytest.raises(ValueError, match=match):
        uv.set_nsamples(nsamples, (ant1, ant2))

    return


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.filterwarnings("ignore:using known location values for SZA.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected")
@pytest.mark.filterwarnings("ignore:pamatten in extra_keywords is a list, array")
@pytest.mark.filterwarnings("ignore:psys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:psysattn in extra_keywords is a list, array or")
@pytest.mark.filterwarnings("ignore:ambpsys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:bfmask in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
@pytest.mark.parametrize(
    "filename",
    [
        "zen.2458661.23480.HH.uvh5",
        "sma_test.mir",
        "carma_miriad",
        "1133866760.uvfits",
        fhd_files,
    ],
)
def test_from_file(filename):

    if "miriad" in filename:
        pytest.importorskip("pyuvdata._miriad")
    if isinstance(filename, str):
        testfile = os.path.join(DATA_PATH, filename)
    else:
        testfile = filename
    uvd = UVData()
    uvd.read(testfile)
    uvd2 = UVData.from_file(testfile)
    assert uvd == uvd2


@pytest.mark.parametrize("add_type", ["blt", "freq", "pol"])
@pytest.mark.parametrize("sort_type", ["blt", "freq", "pol"])
@pytest.mark.parametrize("future_array_shapes", [True, False])
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_pol_sorting_bl(casa_uvfits, add_type, sort_type, future_array_shapes):
    if future_array_shapes:
        casa_uvfits.use_future_array_shapes()

    if add_type == "pol":
        uv1 = casa_uvfits.select(polarizations=["ll", "lr"], inplace=False)
        uv2 = casa_uvfits.select(polarizations=["rr", "rl"], inplace=False)
    elif add_type == "blt":
        uv1 = casa_uvfits.select(
            blt_inds=np.arange(0, casa_uvfits.Nblts // 2), inplace=False
        )
        uv2 = casa_uvfits.select(
            blt_inds=np.arange(casa_uvfits.Nblts // 2, casa_uvfits.Nblts), inplace=False
        )
    elif add_type == "freq":
        uv1 = casa_uvfits.select(
            freq_chans=np.arange(0, casa_uvfits.Nfreqs // 2), inplace=False
        )
        uv2 = casa_uvfits.select(
            freq_chans=np.arange(casa_uvfits.Nfreqs // 2, casa_uvfits.Nfreqs),
            inplace=False,
        )

    if sort_type == "blt":
        uv1.reorder_blts("time", "ant1")
        uv2.reorder_blts("time", "ant2")
        casa_uvfits.reorder_blts("bda")
        order_check = uv1.ant_1_array == uv2.ant_1_array
    elif sort_type == "freq":
        uv1.reorder_freqs(channel_order="freq")
        uv2.reorder_freqs(channel_order="-freq")
        casa_uvfits.reorder_freqs("freq")
        order_check = uv1.freq_array == uv2.freq_array
    elif sort_type == "pol":
        uv1.reorder_pols("AIPS")
        uv2.reorder_pols("CASA")
        casa_uvfits.reorder_pols("CASA")
        order_check = uv1.polarization_array == uv2.polarization_array

    # Make sure that the order has actually been scrambled
    assert not np.all(order_check)

    # Combine the objects
    uv3 = uv1 + uv2

    if sort_type == "blt":
        uv3.reorder_blts("bda")
    elif sort_type == "freq":
        uv3.reorder_freqs(channel_order="freq")
    elif sort_type == "pol":
        uv3.reorder_pols("CASA")

    # Deal with the history separately, since it will be different
    assert str.startswith(uv3.history, casa_uvfits.history)
    casa_uvfits.history = ""
    uv3.history = ""

    # Finally, make sure everything else lines up
    assert uv3 == casa_uvfits
