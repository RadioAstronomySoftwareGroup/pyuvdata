# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvdata object."""

from __future__ import annotations

import copy
import itertools
import os
import re
import warnings
from collections import Counter, namedtuple

import h5py
import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Angle, EarthLocation, Latitude, Longitude, SkyCoord
from astropy.time import Time
from astropy.utils import iers

from pyuvdata import UVCal, UVData, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

from ..utils.test_coordinates import frame_selenoid
from .test_mwa_corr_fits import filelist as mwa_corr_files

try:
    import pyuvdata._miriad  # noqa F401

    hasmiriad = True
except ImportError:
    hasmiriad = False


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
        "telescope",
        "history",
        "vis_units",
        "Nants_data",
        "flex_spw_id_array",
        "phase_center_app_ra",
        "phase_center_app_dec",
        "phase_center_frame_pa",
        "Nphase",
        "phase_center_catalog",
        "phase_center_id_array",
    ]
    required_parameters = ["_" + prop for prop in required_properties]

    extra_properties = [
        "extra_keywords",
        "blt_order",
        "gst0",
        "rdate",
        "earth_omega",
        "dut1",
        "timesys",
        "uvplane_reference_time",
        "scan_number_array",
        "eq_coeffs",
        "eq_coeffs_convention",
        "flex_spw_polarization_array",
        "filename",
        "blts_are_rectangular",
        "time_axis_faster_than_bls",
        "pol_convention",
    ]
    extra_parameters = ["_" + prop for prop in extra_properties]

    other_attributes = [
        "telescope_name",
        "telescope_location",
        "instrument",
        "Nants_telescope",
        "antenna_names",
        "antenna_numbers",
        "antenna_positions",
        "x_orientation",
        "antenna_diameters",
        "telescope_location_lat_lon_alt",
        "telescope_location_lat_lon_alt_degrees",
        "pyuvdata_version_str",
    ]

    uv_object = UVData()

    DataHolder = namedtuple(
        "DataHolder",
        [
            "uv_object",
            "required_parameters",
            "required_properties",
            "extra_parameters",
            "extra_properties",
            "other_attributes",
        ],
    )

    uvdata_props = DataHolder(
        uv_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_attributes,
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


@pytest.fixture(scope="function")
def hera_uvh5_split(hera_uvh5_split_main):
    uv1, uv2, uvfull = hera_uvh5_split_main
    uv1_copy = uv1.copy()
    uv2_copy = uv2.copy()
    uvfull_copy = uvfull.copy()

    yield uv1_copy, uv2_copy, uvfull_copy


@pytest.fixture(scope="session")
def sma_mir_catalog(sma_mir_main):
    catalog_dict = sma_mir_main.phase_center_catalog

    yield catalog_dict


@pytest.fixture(scope="session")
def carma_miriad_main():
    # read in test file for the resampling in time functions
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "carma_miriad")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Altitude is not present in Miriad file")
        warnings.filterwarnings(
            "ignore", "mount_type, feed_array, feed_angle, antenna_diameters are not"
        )
        uv_object.read(testfile, run_check=False, check_extra=False)
    uv_object.extra_keywords = {}

    yield uv_object


@pytest.fixture(scope="function")
def carma_miriad(carma_miriad_main):
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    uv_object = carma_miriad_main.copy()

    yield uv_object


@pytest.fixture(scope="session")
def bda_test_file_main():
    # read in test file for BDA-like data
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "simulated_bda_file.uvh5")
    with check_warnings(
        UserWarning, match="Unknown phase type, assuming object is unprojected"
    ):
        uv_object.read(testfile, default_mount_type="fixed")

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


@pytest.fixture(scope="function")
def pyuvsim_redundant(pyuvsim_redundant_main):
    # read in test file for the compress/inflate redundancy functions
    uv_object = pyuvsim_redundant_main.copy()

    yield uv_object


@pytest.fixture(scope="function")
def uvdata_baseline():
    uv_object = UVData()
    uv_object.telescope.Nants = 128
    uv_object2 = UVData()
    uv_object2.telescope.Nants = 2147483649
    uv_object3 = UVData()
    uv_object3.telescope.Nants = 2050

    DataHolder = namedtuple("DataHolder", ["uv_object", "uv_object2", "uv_object3"])

    uvdata_baseline = DataHolder(uv_object, uv_object2, uv_object3)

    # yields the data we need but will continue to the del call after tests
    yield uvdata_baseline

    # Post test clean-up
    del uvdata_baseline
    return


@pytest.fixture()
def uv_phase_time_split(hera_uvh5):
    uv_phase = hera_uvh5.copy
    uv_raw = hera_uvh5.copy

    uv_phase.reorder_blts("time", minor_order="baseline")
    uv_raw.reorder_blts("time", minor_order="baseline")

    uv_phase.phase(ra=0, dec=0, cat_name="npole", epoch="J2000", use_ant_pos=True)
    times = np.unique(uv_phase.time_array)
    time_set_1, time_set_2 = times[::2], times[1::2]

    uv_phase_1 = uv_phase.select(times=time_set_1, inplace=False)
    uv_phase_2 = uv_phase.select(times=time_set_2, inplace=False)

    uv_raw_1 = uv_raw.select(times=time_set_1, inplace=False)
    uv_raw_2 = uv_raw.select(times=time_set_2, inplace=False)

    yield uv_phase_1, uv_phase_2, uv_phase, uv_raw_1, uv_raw_2, uv_raw

    del uv_phase_1, uv_phase_2, uv_raw_1, uv_raw_2, uv_phase, uv_raw


@pytest.fixture
def mwa_integration_time():
    filename = os.path.join(DATA_PATH, "1061316296.uvfits")
    uv_init = UVData.from_file(filename)
    new_int_time = 1.99813843
    new_int_time_jd = new_int_time / 86400.0
    new_times = np.min(uv_init.time_array) + (
        np.arange(11, dtype=float) * new_int_time_jd
    )
    uvd = UVData.new(
        freq_array=uv_init.freq_array,
        channel_width=uv_init.channel_width,
        polarization_array=uv_init.polarization_array,
        times=new_times,
        telescope=uv_init.telescope,
        do_blt_outer=True,
        empty=True,
    )
    uvd.set_rectangularity()

    yield uvd


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
    assert uvdata_props.uv_object.metadata_only is False
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
    attributes = [
        i
        for i in uvdata_props.uv_object.__dict__
        if (i[0] == "_" and not i.startswith("_UVData__"))
    ]
    for a in attributes:
        assert a in expected_parameters, (
            "unexpected parameter " + a + " found in UVData"
        )


def test_unexpected_attributes(uvdata_props):
    """Test for extra attributes."""
    expected_attributes = (
        uvdata_props.required_properties
        + uvdata_props.extra_properties
        + uvdata_props.other_attributes
    )
    attributes = [i for i in uvdata_props.uv_object.__dict__ if i[0] != "_"]
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
                strict=True,
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
            print(f"setting {k} to a random number failed")
            raise


def test_metadata_only_property(casa_uvfits):
    uvobj = casa_uvfits
    uvobj.data_array = None
    assert uvobj.metadata_only is False
    with pytest.raises(
        ValueError, match="Required UVParameter _data_array has not been set."
    ):
        uvobj.check()
    uvobj.flag_array = None
    assert uvobj.metadata_only is False
    with pytest.raises(
        ValueError, match="Required UVParameter _data_array has not been set."
    ):
        uvobj.check()
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
        match="Nbls must be equal to the number of unique baselines in the data_array",
    ):
        uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_ntimes(casa_uvfits):
    uvobj = casa_uvfits
    uvobj.Ntimes += 1
    with pytest.raises(
        ValueError,
        match="Ntimes must be equal to the number of unique times in the time_array",
    ):
        uvobj.check()
    uvobj.Ntimes -= 1


def test_check_phase_center_id_array(casa_uvfits):
    uvobj = casa_uvfits
    uvobj.phase_center_id_array[0] = 4
    with pytest.raises(
        ValueError,
        match="Phase center id 4 is does not have an entry in `phase_center_catalog`",
    ):
        uvobj.check(strict_uvw_antpos_check=True)


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


def test_check_autos_only(hera_uvh5):
    """
    Check case where all data is autocorrelations
    """
    uvobj = hera_uvh5

    uvobj.select(blt_inds=np.where(uvobj.ant_1_array == uvobj.ant_2_array)[0])
    assert uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_uvw_array(hera_uvh5):
    # test auto and cross corr uvw_array
    uvd = hera_uvh5.copy()
    autos = np.isclose(uvd.ant_1_array - uvd.ant_2_array, 0.0)
    auto_inds = np.where(autos)[0]
    cross_inds = np.where(~autos)[0]

    # make auto have non-zero uvw coords, assert ValueError
    uvd.uvw_array[auto_inds[0], 0] = 0.1
    with pytest.raises(
        ValueError, match="Some auto-correlations have non-zero uvw_array coordinates."
    ):
        uvd.check()

    # make cross have |uvw| zero, assert ValueError
    uvd = hera_uvh5.copy()
    uvd.uvw_array[cross_inds[0]][:] = 0.0
    with pytest.raises(
        ValueError, match="Some cross-correlations have near-zero uvw_array magnitudes."
    ):
        uvd.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_check_flag_array(casa_uvfits):
    uvobj = casa_uvfits

    uvobj.flag_array = np.ones((uvobj.flag_array.shape), dtype=int)

    with pytest.raises(
        ValueError, match="UVParameter _flag_array is not the appropriate type."
    ):
        uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_nants_data_telescope_larger(casa_uvfits):
    uvobj = casa_uvfits
    # make sure it's okay for Nants_telescope to be strictly greater than Nants_data
    uvobj.telescope.Nants += 1
    # add dummy information for "new antenna" to pass object check
    uvobj.telescope.antenna_names = np.concatenate(
        (uvobj.telescope.antenna_names, ["dummy_ant"])
    )
    uvobj.telescope.antenna_numbers = np.concatenate(
        (uvobj.telescope.antenna_numbers, [20])
    )
    uvobj.telescope.antenna_positions = np.concatenate(
        (uvobj.telescope.antenna_positions, np.zeros((1, 3))), axis=0
    )
    uvobj.telescope.feed_array = np.concatenate(
        (uvobj.telescope.feed_array, [uvobj.telescope.feed_array[-1]]), axis=0
    )
    uvobj.telescope.feed_angle = np.concatenate(
        (uvobj.telescope.feed_angle, [uvobj.telescope.feed_angle[-1]]), axis=0
    )
    uvobj.telescope.mount_type = np.concatenate(
        (uvobj.telescope.mount_type, [uvobj.telescope.mount_type[-1]]), axis=0
    )
    assert uvobj.check()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_ant1_array_not_in_antnums(casa_uvfits):
    uvobj = casa_uvfits
    # make sure an error is raised if antennas in ant_1_array not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    uvobj.telescope.antenna_names = uvobj.telescope.antenna_names[1:]
    uvobj.telescope.antenna_numbers = uvobj.telescope.antenna_numbers[1:]
    uvobj.telescope.antenna_positions = uvobj.telescope.antenna_positions[1:, :]
    uvobj.telescope.mount_type = uvobj.telescope.mount_type[1:]
    uvobj.telescope.feed_angle = uvobj.telescope.feed_angle[1:, :]
    uvobj.telescope.feed_array = uvobj.telescope.feed_array[1:, :]
    uvobj.telescope.Nants = uvobj.telescope.antenna_numbers.size
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
    uvobj.telescope.antenna_names = uvobj.telescope.antenna_names[:-1]
    uvobj.telescope.antenna_numbers = uvobj.telescope.antenna_numbers[:-1]
    uvobj.telescope.antenna_positions = uvobj.telescope.antenna_positions[:-1]
    uvobj.telescope.mount_type = uvobj.telescope.mount_type[:-1]
    uvobj.telescope.feed_angle = uvobj.telescope.feed_angle[:-1, :]
    uvobj.telescope.feed_array = uvobj.telescope.feed_array[:-1, :]
    uvobj.telescope.Nants = uvobj.telescope.antenna_numbers.size
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

    with pytest.raises(
        ValueError, match="filetype must be uvfits, mir, miriad, ms, fhd, or uvh5"
    ):
        uvobj._convert_to_filetype("foo")


def test_baseline_to_antnums(uvdata_baseline):
    """Test baseline to antnum conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.baseline_to_antnums(65536) == (0, 0)
    assert uvdata_baseline.uv_object.baseline_to_antnums(592128) == (257, 256)
    assert uvdata_baseline.uv_object.baseline_to_antnums(4404493223938) == (2051, 2050)

    with pytest.raises(
        Exception,
        match=(
            f"error Nants={uvdata_baseline.uv_object2.telescope.Nants}>2147483648"
            " not supported"
        ),
    ):
        uvdata_baseline.uv_object2.baseline_to_antnums(65536)
    with pytest.raises(ValueError, match="negative baseline numbers are not supported"):
        uvdata_baseline.uv_object.baseline_to_antnums(-10)
    with pytest.raises(
        ValueError, match="baseline numbers > 4611686018498691072 are not supported"
    ):
        uvdata_baseline.uv_object.baseline_to_antnums(4611686018498691073)
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
    assert np.array_equal(baseline_array, [86036, 639286])
    ant_1_out, ant_2_out = uvdata_baseline.uv_object.baseline_to_antnums(
        baseline_array.tolist()
    )
    assert np.array_equal(ant_1, ant_1_out)
    assert np.array_equal(ant_2, ant_2_out)


def test_antnums_to_baselines(uvdata_baseline):
    """Test antums to baseline conversion for 256 & larger conventions."""
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0) == 65536
    assert uvdata_baseline.uv_object.antnums_to_baseline(257, 256) == 592128
    assert uvdata_baseline.uv_object.antnums_to_baseline(2051, 2050) == 4404493223938
    # Check attempt256
    assert uvdata_baseline.uv_object.antnums_to_baseline(0, 0, attempt256=True) == 0
    with check_warnings(UserWarning, "found antenna numbers > 255"):
        uvdata_baseline.uv_object.antnums_to_baseline(256, 255, attempt256=True)
    with check_warnings(UserWarning, "found antenna numbers > 2047"):
        uvdata_baseline.uv_object.antnums_to_baseline(2051, 2050, attempt256=True)
    with check_warnings(UserWarning, "found antenna numbers > 2047"):
        uvdata_baseline.uv_object3.antnums_to_baseline(1112, 1111, attempt256=True)
    with pytest.raises(
        ValueError,
        match=(
            "cannot convert ant1, ant2 to a baseline index with "
            f"Nants={uvdata_baseline.uv_object2.telescope.Nants}>2147483648."
        ),
    ):
        uvdata_baseline.uv_object2.antnums_to_baseline(0, 0)
    # check for out of range antenna numbers
    with pytest.raises(
        ValueError,
        match=(
            "cannot convert ant1, ant2 to a baseline index "
            "with antenna numbers greater than 2147483647."
        ),
    ):
        uvdata_baseline.uv_object.antnums_to_baseline(2147483649, 2147483648)
    with pytest.raises(
        ValueError,
        match=(
            "cannot convert ant1, ant2 to a baseline index "
            "with antenna numbers less than zero."
        ),
    ):
        uvdata_baseline.uv_object.antnums_to_baseline(-10, 2047)
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
        set(astropy_sites + ["PAPER", "HERA", "SMA", "SZA", "OVRO-LWA", "ATA"])
    )
    # calling np.sort().tolist() because [].sort() acts inplace and returns None
    # Before test had None == None
    assert (
        np.sort(known_telescopes).tolist()
        == np.sort(uv_object.known_telescopes()).tolist()
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_hera_diameters(casa_uvfits):
    uv_in = casa_uvfits

    # change telescope name to HERA
    uv_in.telescope.name = "HERA"
    # check that set_telescope_params sets the diameters properly
    uv_in.set_telescope_params()

    assert uv_in.telescope.name == "HERA"
    assert uv_in.telescope.antenna_diameters is not None

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
            antenna_nums=uv_in.telescope.antenna_numbers[0],
            antenna_names=uv_in.telescope.antenna_names[1],
        )

    with pytest.raises(ValueError, match="File type could not be determined"):
        uv_in.read(os.path.join(DATA_PATH, "mwa_ant_pos.csv"))

    with pytest.raises(FileNotFoundError, match="File not found, check path for: foo"):
        uv_in.read("foo")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    ["phase_kwargs", "partial"],
    [
        ({"cat_name": "loc0", "lon": 0.0, "lat": 0.0, "epoch": "J2000"}, False),
        ({"cat_name": "loc0", "lon": 0.0, "lat": 0.0, "phase_frame": "icrs"}, True),
        (
            {
                "cat_name": "gcrs1",
                "lon": Angle("5d").rad,
                "lat": Angle("30d").rad,
                "phase_frame": "gcrs",
            },
            False,
        ),
        (
            {
                "cat_name": "epoch2010",
                "ra": Angle("180d").rad,
                "dec": Angle("90d").rad,
                "epoch": Time("2010-01-01T00:00:00", format="isot", scale="utc"),
            },
            False,
        ),
        (
            {
                "cat_name": "near_field_test",
                "ra": 0.4,
                "dec": -0.3,
                "cat_type": "near_field",
                "dist": 10 * units.km,
            },
            False,
        ),
    ],
)
def test_phase_unphase_hera(hera_uvh5, phase_kwargs, partial):
    """
    Read in drift data, phase to an RA/DEC, unphase and check for object equality.
    """
    uv1 = hera_uvh5.copy()
    uv_raw = hera_uvh5

    if partial:
        mask = np.full(uv1.Nblts, False)
        mask[: uv1.Nblts // 2] = True
        phase_kwargs["select_mask"] = mask
    else:
        mask = None

    uv1.phase(**phase_kwargs)
    if partial:
        uv1.unproject_phase(select_mask=mask)
    else:
        uv1.unproject_phase()
    if partial:
        uv1.merge_phase_centers(
            catalog_identifier=list(uv1.phase_center_catalog.keys()), ignore_name=True
        )
    else:
        uv1.rename_phase_center(0, new_name="zenith")
    # check that phase + unphase gets back to raw
    assert uv_raw == uv1


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("cat_type", ["sidereal", "near_field"])
def test_phase_unphase_hera_one_bl(hera_uvh5, cat_type):
    uv_raw = hera_uvh5
    # check that phase + unphase work with one baseline
    uv_raw_small = uv_raw.select(blt_inds=[0], inplace=False)
    uv_phase_small = uv_raw_small.copy()
    uv_phase_small.phase(
        lon=Angle("23h").rad,
        lat=Angle("15d").rad,
        cat_name="foo",
        cat_type=cat_type,
        dist=5000,
    )
    uv_phase_small.unproject_phase(cat_name="zenith")
    assert uv_raw_small == uv_phase_small


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("cat_type", ["sidereal", "near_field"])
def test_phase_unphase_hera_antpos(hera_uvh5, cat_type):
    uv_phase = hera_uvh5.copy()
    uv_raw = hera_uvh5
    # check that they match if you phase & unphase using antenna locations
    # first replace the uvws with the right values
    lat, lon, alt = uv_raw.telescope.location_lat_lon_alt
    antenna_enu = utils.ENU_from_ECEF(
        (uv_raw.telescope.antenna_positions + uv_raw.telescope._location.xyz()),
        center_loc=uv_raw.telescope.location,
    )
    uvw_calc = np.zeros_like(uv_raw.uvw_array)
    unique_times = np.unique(uv_raw.time_array)
    for jd in unique_times:
        inds = np.where(uv_raw.time_array == jd)[0]
        for bl_ind in inds:
            wh_ant1 = np.where(
                uv_raw.telescope.antenna_numbers == uv_raw.ant_1_array[bl_ind]
            )
            ant1_index = wh_ant1[0][0]
            wh_ant2 = np.where(
                uv_raw.telescope.antenna_numbers == uv_raw.ant_2_array[bl_ind]
            )
            ant2_index = wh_ant2[0][0]
            uvw_calc[bl_ind, :] = (
                antenna_enu[ant2_index, :] - antenna_enu[ant1_index, :]
            )

    uv_raw_new = uv_raw.copy()
    uv_raw_new.uvw_array = uvw_calc
    uv_phase.phase(
        ra=0.0,
        dec=0.0,
        epoch="J2000",
        cat_name="foo",
        use_ant_pos=True,
        cat_type=cat_type,
        dist=7000,
    )
    uv_phase2 = uv_raw_new.copy()
    uv_phase2.phase(
        ra=0.0, dec=0.0, epoch="J2000", cat_name="foo", cat_type=cat_type, dist=7000
    )

    # The uvw's only agree to ~1mm. should they be better?
    np.testing.assert_allclose(
        uv_phase2.uvw_array, uv_phase.uvw_array, atol=1e-3, rtol=0
    )
    # the data array are just multiplied by the w's for phasing, so a difference
    # at the 1e-3 level makes the data array different at that level too.
    # -> change the tolerance on data_array for this test
    uv_phase2._data_array.tols = (0, 1e-3 * np.amax(np.abs(uv_phase2.data_array)))
    assert uv_phase2 == uv_phase

    # check that phase + unphase gets back to raw using antpos
    uv_phase.unproject_phase(use_ant_pos=True, cat_name="zenith")
    assert uv_raw_new == uv_phase


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_hera_zenith_timestamp_minimal_changes(hera_uvh5):
    uv_raw = hera_uvh5
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
    np.testing.assert_allclose(
        uv_phase_simple_small.uvw_array, uv_raw_small.uvw_array, atol=1e-1, rtol=0
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_to_time_jd_input(hera_uvh5):
    uv_phase = hera_uvh5.copy()
    uv_raw = hera_uvh5
    uv_phase.phase_to_time(uv_raw.time_array[0])
    uv_phase.unproject_phase(cat_name="zenith")
    assert uv_phase == uv_raw


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_to_time_error(hera_uvh5):
    uv_phase = hera_uvh5
    # check error if not passing a Time object to phase_to_time
    with pytest.raises(TypeError, match="time must be an astropy.time.Time object"):
        uv_phase.phase_to_time("foo")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_phase_to_time(casa_uvfits, telescope_frame, selenoid):
    uv_in = casa_uvfits
    phase_time = Time(uv_in.time_array[0], format="jd")

    if telescope_frame == "mcmf":
        pytest.importorskip("lunarsky")
        from lunarsky import MoonLocation, SkyCoord as LunarSkyCoord
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        spicey_err = SpiceUNKNOWNFRAME

        enu_antpos = uv_in.telescope.get_enu_antpos()
        uv_in.telescope.location = MoonLocation.from_selenodetic(
            lat=uv_in.telescope.location.lat,
            lon=uv_in.telescope.location.lon,
            height=uv_in.telescope.location.height,
            ellipsoid=selenoid,
        )
        new_full_antpos = utils.ECEF_from_ENU(
            enu=enu_antpos, center_loc=uv_in.telescope.location
        )
        uv_in.telescope.antenna_positions = (
            new_full_antpos - uv_in.telescope._location.xyz()
        )
        uv_in.set_lsts_from_time_array()
        uv_in.check()

        zenith_coord = LunarSkyCoord(
            alt=Angle(90 * units.deg),
            az=Angle(0 * units.deg),
            obstime=phase_time,
            frame="lunartopo",
            location=uv_in.telescope.location,
        )
    else:
        spicey_err = None
        zenith_coord = SkyCoord(
            alt=Angle(90 * units.deg),
            az=Angle(0 * units.deg),
            obstime=phase_time,
            frame="altaz",
            location=uv_in.telescope.location,
        )
    zen_icrs = zenith_coord.transform_to("icrs")

    try:
        uv_in.phase_to_time(uv_in.time_array[0])
    except spicey_err:
        pytest.skip(reason="Flaky CSPICE issue")

    assert np.isclose(
        uv_in.phase_center_catalog[1]["cat_lat"],
        zen_icrs.dec.rad,
        rtol=0,
        atol=utils.RADIAN_TOL,
    )
    assert np.isclose(
        uv_in.phase_center_catalog[1]["cat_lon"],
        zen_icrs.ra.rad,
        rtol=0,
        atol=utils.RADIAN_TOL,
    )

    assert np.isclose(
        uv_in.phase_center_catalog[1]["cat_lon"], uv_in.lst_array[0], rtol=1e-3
    )

    if telescope_frame == "itrs":
        assert np.isclose(
            uv_in.phase_center_catalog[1]["cat_lat"],
            uv_in.telescope.location.lat.rad,
            1e-2,
        )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_unphase_drift_data_error(hera_uvh5, sma_mir):
    uv_phase = hera_uvh5.copy()
    uv_drift = hera_uvh5
    # check error unphasing an unphased object

    uv_phase.phase(ra=0.0, dec=0.0, cat_name="foo")

    with pytest.raises(IndexError, match="Selection mask must be of length Nblts."):
        uv_phase.unproject_phase(select_mask=np.full(uv_phase.Nblts // 2, True))

    with pytest.raises(ValueError, match="Selection mask must be a boolean array"):
        uv_phase.unproject_phase(select_mask=np.full(uv_phase.Nblts, 5.3))

    uv_phase.unproject_phase(cat_name="zenith")
    assert uv_drift == uv_phase

    # Check to make sure that wa can unphase w/o an error getting thrown. The new
    # unphase method does not throw an error when being called twice, but it does warn
    sma_mir.unproject_phase()
    with check_warnings(
        UserWarning, match="No selected baselines are projected, doing nothing"
    ):
        sma_mir.unproject_phase()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_phase_errors():
    uv_phase = UVData()

    with pytest.raises(
        ValueError, match="lon parameter must be set if cat_type is not 'unprojected'"
    ):
        uv_phase.phase(dec=0.0, epoch="J2000", cat_name="foo")

    with pytest.raises(
        ValueError, match="lat parameter must be set if cat_type is not 'unprojected'"
    ):
        uv_phase.phase(ra=0.0, epoch="J2000", cat_name="foo")


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.parametrize("use_ant_pos1", [True, False])
@pytest.mark.parametrize("use_ant_pos2", [True, False])
def test_unphasing(uv_phase_comp, use_ant_pos1, use_ant_pos2):
    uvd1, uvd2 = uv_phase_comp

    uvd1.unproject_phase(use_ant_pos=use_ant_pos1)
    uvd2.unproject_phase(use_ant_pos=use_ant_pos2)

    if use_ant_pos1 and use_ant_pos2:
        atol = 1e-12
    else:
        atol = 5e-2

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    np.testing.assert_allclose(uvd1.uvw_array, uvd2.uvw_array, atol=atol, rtol=0)


@pytest.mark.parametrize("use_ant_pos", [True, False])
@pytest.mark.parametrize("unphase_first", [True, False])
def test_phasing(uv_phase_comp, unphase_first, use_ant_pos):
    uvd1, uvd2 = uv_phase_comp

    if unphase_first:
        uvd2.unproject_phase(use_ant_pos=use_ant_pos)
        warning_str = ""
        exp_warning = None
    else:
        warning_str = [
            (
                "The entry name UVCeti is not unique inside the phase center catalog, "
                "adding anyways"
            ),
            (
                "The provided name UVCeti is already used but has different parameters."
                " Adding another entry with the same name but a different ID and"
                " parameters."
            ),
        ]
        exp_warning = UserWarning

    if use_ant_pos:
        uvd1.set_uvws_from_antenna_positions()

    uvd1_phase_dict = list(uvd1.phase_center_catalog.values())[0]

    with check_warnings(exp_warning, match=warning_str):
        uvd2.phase(
            lon=uvd1_phase_dict["cat_lon"],
            lat=uvd1_phase_dict["cat_lat"],
            epoch=uvd1_phase_dict["cat_epoch"],
            phase_frame=uvd1_phase_dict["cat_frame"],
            cat_name=uvd1_phase_dict["cat_name"],
            use_ant_pos=use_ant_pos,
        )

    if use_ant_pos:
        atol = 1e-12
    else:
        atol = 5e-2

    # the tolerances here are empirical -- based on what was seen in the
    # external phasing test. See the phasing memo in docs/references for
    # details.
    np.testing.assert_allclose(uvd1.uvw_array, uvd2.uvw_array, atol=atol, rtol=0)


@pytest.mark.parametrize(
    "arg_dict,err_type,msg",
    [
        [
            {"name": "abc", "mask": [True] * 2},
            IndexError,
            "Selection mask must be of length Nblts.",
        ],
        [
            {"name": "abc", "mask": [5.0]},
            ValueError,
            "Selection mask must be a boolean array",
        ],
    ],
)
def test_phasing_multi_phase_errs(sma_mir, arg_dict, err_type, msg):
    # Now do a few things that aren't allowed w/ a multi-phase-ctr data set
    with pytest.raises(err_type, match=msg):
        sma_mir.phase(
            ra=0, dec=0, cat_name=arg_dict.get("name"), select_mask=arg_dict.get("mask")
        )


@pytest.mark.filterwarnings("ignore:The provided name UVCeti is already used")
@pytest.mark.filterwarnings("ignore:The entry name UVCeti is not unique")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
def test_cotter_phasing(uv_phase_comp):
    """Use MWA files phased to 2 different places to test phasing."""
    uvd1, uvd2 = uv_phase_comp

    uvd1_drift = uvd1.copy()
    uvd1_drift.unproject_phase()

    uvd2_drift = uvd2.copy()
    uvd2_drift.unproject_phase()

    np.testing.assert_allclose(
        uvd1_drift.uvw_array,
        uvd2_drift.uvw_array,
        rtol=uvd1_drift._uvw_array.tols[0],
        atol=uvd1_drift._uvw_array.tols[1],
    )
    # the tolerances here are empirical -- this just makes sure they don't get worse.
    # TODO: more investigation here needed!
    np.testing.assert_allclose(
        uvd1_drift.data_array, uvd2_drift.data_array, atol=3e-2, rtol=0
    )

    uvd1_phase_dict = list(uvd1.phase_center_catalog.values())[0]
    uvd2_rephase = uvd2.copy()
    uvd2_rephase.phase(
        lon=uvd1_phase_dict["cat_lon"],
        lat=uvd1_phase_dict["cat_lat"],
        epoch=uvd1_phase_dict["cat_epoch"],
        cat_name=uvd1_phase_dict["cat_name"],
        phase_frame=uvd1_phase_dict["cat_frame"],
        cat_type="sidereal",
        use_ant_pos=True,
    )

    # the tolerances here are empirical -- this just makes sure they don't get worse.
    # TODO: more investigation here needed!
    np.testing.assert_allclose(
        uvd1.uvw_array, uvd2_rephase.uvw_array, atol=5e-2, rtol=0
    )

    # rephase the drift objects to the original pointing and verify that they
    # match
    uvd1_drift.phase(
        lon=uvd1_phase_dict["cat_lon"],
        lat=uvd1_phase_dict["cat_lat"],
        epoch=uvd1_phase_dict["cat_epoch"],
        cat_name=uvd1_phase_dict["cat_name"],
        phase_frame=uvd1_phase_dict["cat_frame"],
        use_ant_pos=True,
    )

    # the tolerances here are empirical -- this just makes sure they don't get worse.
    # TODO: more investigation here needed!
    np.testing.assert_allclose(uvd1.uvw_array, uvd1_drift.uvw_array, atol=5e-2, rtol=0)
    np.testing.assert_allclose(uvd1.data_array, uvd1_drift.data_array, atol=4, rtol=0)

    uvd2_phase_dict = list(uvd2.phase_center_catalog.values())[0]
    uvd2_drift.phase(
        lon=uvd2_phase_dict["cat_lon"],
        lat=uvd2_phase_dict["cat_lat"],
        epoch=uvd2_phase_dict["cat_epoch"],
        cat_name=uvd2_phase_dict["cat_name"],
        use_ant_pos=True,
    )

    # the tolerances here are empirical -- this just makes sure they don't get worse.
    # TODO: more investigation here needed!
    np.testing.assert_allclose(uvd2.uvw_array, uvd2_drift.uvw_array, atol=5e-3, rtol=0)
    np.testing.assert_allclose(uvd2.data_array, uvd2_drift.data_array, atol=5, rtol=0)

    # TODO: not sure the rest of this is useful...
    uvd1_drift = uvd1.copy()
    # Move the time ~1 µsec off from J2000
    epoch_val = Time(Time(2000, format="jyear").mjd - 1e-11, format="mjd")
    # Unlike in the new phasing system, this should produce different results (since one
    # is FK5, and the other is ICRS)
    uvd1_drift.phase(ra=0, dec=0, cat_name="fk50", epoch=epoch_val)
    uvd1.phase(ra=0, dec=0, cat_name="icrs0", epoch="J2000")
    assert uvd1_drift != uvd1
    uvd1_drift = uvd1.copy()

    # Make sure the old default works for reverting to ICRS if no coord frame is found
    uvd1.unproject_phase()
    uvd1_drift.unproject_phase()
    assert uvd1 == uvd1_drift


def test_phasing_unprojected(sma_mir):
    # Make sure that unprojected via phasing and unproject_phase works the same.
    sma_copy = sma_mir.copy()

    sma_mir.unproject_phase()
    sma_copy.phase(
        cat_name="unprojected", cat_type="unprojected", phase_frame=None, epoch=None
    )

    assert sma_mir == sma_copy


def test_set_uvws(hera_uvh5):
    uv1 = hera_uvh5
    # mess up the uvw_array
    uv1.uvw_array *= 1.1
    uv2 = uv1.copy()
    uv1.phase(ra=0.0, dec=0.0, cat_name="foo", use_ant_pos=False)
    uv2.phase(ra=0.0, dec=0.0, cat_name="foo")
    with check_warnings(
        UserWarning,
        match=(
            "Recalculating uvw_array without adjusting visibility "
            "phases -- this can introduce significant errors if used "
            "incorrectly."
        ),
    ):
        uv1.set_uvws_from_antenna_positions(update_vis=False)
    assert uv1._uvw_array == uv2._uvw_array
    assert uv1._data_array != uv2._data_array


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("metadata_only", [True, False])
@pytest.mark.parametrize("invert", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("higher_dims", [True, False])
def test_select_blts(hera_uvh5, metadata_only, invert, inplace, higher_dims):
    uv_object = hera_uvh5

    # fmt: off
    # generated with np.random.choice(200, size=50, replace=False)
    # using the fact that uv_object.Nblts=200
    blt_inds = np.array(
        [142, 111, 168, 155, 101, 58, 124, 122,  44, 102, 77, 174,  88, 76, 20, 161,
         43, 78, 104, 80, 56, 107, 42, 74, 153, 180, 158, 95, 12, 93, 47, 163, 128,
         105, 0,  30, 196, 8, 191, 39, 9, 125, 112, 152, 11, 18, 97, 51, 177, 110]
    )
    # fmt: on
    if not metadata_only:
        selected_data = uv_object.data_array[np.sort(blt_inds)]

    if invert:
        select_inds = np.nonzero(
            np.isin(np.arange(uv_object.Nblts), blt_inds, invert=True)
        )[0]
    else:
        select_inds = blt_inds

    if higher_dims:
        select_inds = select_inds[np.newaxis, :]

    uv_object2 = uv_object.copy(metadata_only=metadata_only)
    uv_object3 = uv_object2.select(blt_inds=select_inds, inplace=inplace, invert=invert)
    if inplace:
        assert uv_object3 is None
        uv_object3 = uv_object2
    else:
        if metadata_only:
            uv_object.data_array = None
            uv_object.flag_array = None
            uv_object.nsample_array = None
            assert uv_object.metadata_only
        assert uv_object2 == uv_object

    assert len(blt_inds) == uv_object3.Nblts

    # verify that histories are different
    assert not utils.history._check_histories(uv_object.history, uv_object3.history)

    assert utils.history._check_histories(
        uv_object.history + "  Downselected to specific baseline-times using pyuvdata.",
        uv_object3.history,
    )

    if not metadata_only:
        assert np.all(selected_data == uv_object3.data_array)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [
            {"blt_inds": [-2, -1], "strict": True},
            "blt_inds contains indices that are negative",
        ],
        [
            {"blt_inds": 10000000, "strict": True},
            "blt_inds contains indices that are too large",
        ],
        [
            {"blt_inds": 10000000, "strict": None},
            "No baseline-times were found that match",
        ],
    ],
)
def test_select_blts_err(casa_uvfits, kwargs, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        casa_uvfits.select(**kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_phase_center_id(tmp_path, carma_miriad):
    uv_obj = carma_miriad
    testfile = os.path.join(tmp_path, "outtest.uvh5")
    assert uv_obj.telescope.instrument is not None

    uv1 = uv_obj.select(phase_center_ids=0, inplace=False)
    uv2 = uv_obj.select(phase_center_ids=[1, 2], inplace=False)

    uv_sum = uv1 + uv2
    assert utils.history._check_histories(
        uv_obj.history + "  Downselected to specific phase center IDs using pyuvdata.  "
        "Combined data along baseline-time axis using pyuvdata.",
        uv_sum.history,
    )
    uv_sum.history = uv_obj.history

    assert uv_sum == uv_obj

    uv_obj.write_uvh5(testfile)

    uv1_read = UVData.from_file(testfile, phase_center_ids=0)
    assert uv1_read == uv1

    uv2_read = UVData.from_file(testfile, phase_center_ids=[1, 2])
    assert uv2_read == uv2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_phase_center_id_blts(carma_miriad):
    uv_obj = carma_miriad
    uv_obj.reorder_blts("baseline")

    uv1 = uv_obj.select(
        phase_center_ids=0, blt_inds=np.arange(uv_obj.Nblts // 2), inplace=False
    )
    uv2 = uv_obj.select(
        phase_center_ids=[1, 2], blt_inds=np.arange(uv_obj.Nblts // 2), inplace=False
    )
    uv3 = uv_obj.select(
        blt_inds=np.arange(uv_obj.Nblts // 2, uv_obj.Nblts), inplace=False
    )

    uv_sum = uv1 + uv2 + uv3
    assert utils.history._check_histories(
        uv_obj.history
        + "  Downselected to specific phase center IDs, baseline-times using pyuvdata. "
        "Combined data along baseline-time axis using pyuvdata.  "
        "Combined data along baseline-time axis using pyuvdata.  ",
        uv_sum.history,
    )
    uv_sum.history = uv_obj.history

    uv_sum.reorder_blts()
    uv_obj.reorder_blts()

    assert uv_sum == uv_obj


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("invert", [True, False])
@pytest.mark.parametrize("use_names", [True, False])
@pytest.mark.parametrize("higher_dim", [True, False])
@pytest.mark.parametrize("keep_meta", [True, False])
def test_select_antennas(casa_uvfits, invert, use_names, higher_dim, keep_meta):
    uv_object = casa_uvfits
    old_history = uv_object.history

    # Plug in ant diameters for this test
    uv_object.telescope.antenna_diameters = np.arange(
        uv_object.telescope.Nants, dtype=np.float64
    )
    orig_telescope = uv_object.telescope.copy()

    ants_to_keep = np.array([1, 20, 12, 25, 4, 24, 2, 21, 22])

    if invert:
        full_ants = np.unique([uv_object.ant_1_array, uv_object.ant_2_array])
        ants_to_discard = full_ants[np.isin(full_ants, ants_to_keep, invert=True)]

    Nblts_selected = np.sum(
        np.logical_and(
            np.isin(uv_object.ant_1_array, ants_to_keep),
            np.isin(uv_object.ant_2_array, ants_to_keep),
        )
    )

    kwargs = {"invert": invert, "keep_all_metadata": keep_meta}
    if use_names:
        key = "antenna_nums"
        value = ants_to_discard if invert else ants_to_keep
    else:
        key = "antenna_names"
        value = [
            name
            for num, name in zip(
                uv_object.telescope.antenna_numbers,
                uv_object.telescope.antenna_names,
                strict=True,
            )
            if num in (ants_to_discard if invert else ants_to_keep)
        ]
    kwargs[key] = [value] if higher_dim else value

    uv_object.select(**kwargs)

    assert len(ants_to_keep) == uv_object.Nants_data
    assert Nblts_selected == uv_object.Nblts
    assert np.all(np.isin(uv_object.ant_1_array, ants_to_keep))
    assert np.all(np.isin(uv_object.ant_2_array, ants_to_keep))
    assert np.all(
        np.logical_or(
            np.isin(ants_to_keep, uv_object.ant_1_array),
            np.isin(ants_to_keep, uv_object.ant_2_array),
        )
    )

    assert utils.history._check_histories(
        old_history + "  Downselected to specific antennas using pyuvdata.",
        uv_object.history,
    )

    if keep_meta:
        assert uv_object.telescope == orig_telescope
    else:
        assert uv_object.telescope.Nants == len(ants_to_keep)
        assert all(np.isin(uv_object.telescope.antenna_numbers, ants_to_keep))
        # Make an array to make comparison easier w/ mask
        mask = np.isin(orig_telescope.antenna_numbers, ants_to_keep)
        uv_object.telescope.antenna_names = np.array(uv_object.telescope.antenna_names)
        orig_telescope.antenna_names = np.array(orig_telescope.antenna_names)
        for param in ["_antenna_names", "_antenna_positions"]:
            assert getattr(uv_object.telescope, param).compare_value(
                getattr(orig_telescope, param[1:])[mask]
            )
        assert np.asarray(uv_object.telescope)


@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [
            {"antenna_nums": [29, 30], "strict": True},
            r"Antenna number \[29 30\] is not present",
        ],
        [
            {"antenna_nums": np.array([29, 30]), "strict": True},
            r"Antenna number \[29 30\] is not present",
        ],
        [
            {"antenna_names": "test1", "strict": True},
            "Antenna name test1 is not present in the antenna_names",
        ],
        [
            {"antenna_names": "test1", "strict": None},
            "No baseline-times were found that match criteria",
        ],
        [
            {"antenna_nums": np.arange(100), "strict": None, "invert": True},
            "No baseline-times were found that match criteria",
        ],
        [
            {"antenna_nums": [], "antenna_names": []},
            "Only one of antenna_nums and antenna_names can be provided.",
        ],
    ],
)
def test_select_antnum_errs(casa_uvfits, kwargs, err_msg):
    uv_object = casa_uvfits
    with pytest.raises(ValueError, match=err_msg):
        uv_object.select(**kwargs)


def sort_bl(p):
    """Sort a tuple that starts with a pair of antennas, and may have stuff after."""
    if p[1] >= p[0]:
        return p
    return (p[1], p[0]) + p[2:]


@pytest.mark.filterwarnings("ignore:Selected bls contain a mixture of different")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "sel_type", ["antpair", "blnum", "antpairpol", "antpair_npint", "single"]
)
def test_select_bls(casa_uvfits, sel_type):
    uv_object = casa_uvfits
    old_history = uv_object.history
    first_ants = [7, 3, 8, 3, 22, 28, 9]
    second_ants = [1, 21, 9, 2, 3, 4, 23]
    pols = ["RR", "RR", "RR", "RR", "RR", "RR", "RR"]

    if sel_type == "antpairpol":
        # Also test that reading different pols at the same time works.
        pols[-1] = "LL"

    new_unique_ants = np.unique(first_ants + second_ants)
    ant_pairs_to_keep = list(zip(first_ants, second_ants, strict=True))
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]
    bls_nums_to_keep = [
        uv_object.antnums_to_baseline(ant1, ant2) for ant1, ant2 in sorted_pairs_to_keep
    ]
    bls_to_keep = list(zip(first_ants, second_ants, pols, strict=True))

    blts_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array, strict=True)
    ]
    Nblts_selected = np.sum(blts_select)
    sel_str = "antenna pairs"

    if sel_type == "antpair":
        bls_select = ant_pairs_to_keep
    elif sel_type == "antpair_npint":
        bls_select = list(
            zip(
                list(map(np.int32, first_ants)),
                list(map(np.int32, second_ants)),
                strict=True,
            )
        )
    elif sel_type == "blnum":
        bls_select = bls_nums_to_keep
    elif sel_type == "antpairpol":
        bls_select = bls_to_keep
        sel_str = "antenna pairs, polarizations"
    elif sel_type == "single":
        bls_select = (1, 7)
        new_unique_ants = [1, 7]
        sorted_pairs_to_keep = [(1, 7)]
        blts_select = [
            sort_bl((a1, a2)) in sorted_pairs_to_keep
            for (a1, a2) in zip(
                uv_object.ant_1_array, uv_object.ant_2_array, strict=True
            )
        ]
        Nblts_selected = np.sum(blts_select)

    uv_object.select(bls=bls_select)
    sorted_pairs_object2 = [
        sort_bl(p)
        for p in zip(uv_object.ant_1_array, uv_object.ant_2_array, strict=True)
    ]

    assert len(new_unique_ants) == uv_object.Nants_data
    assert Nblts_selected == uv_object.Nblts
    for ant in new_unique_ants:
        assert ant in uv_object.ant_1_array or ant in uv_object.ant_2_array
    for ant in np.unique(
        uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist()
    ):
        assert ant in new_unique_ants
    for pair in sorted_pairs_to_keep:
        assert pair in sorted_pairs_object2
    for pair in sorted_pairs_object2:
        assert pair in sorted_pairs_to_keep

    if sel_type == "antpairpol":
        assert uv_object.Npols == 2

    assert utils.history._check_histories(
        old_history + f"  Downselected to specific {sel_str} using pyuvdata.",
        uv_object.history,
    )


@pytest.mark.parametrize(
    ["sel_kwargs", "err_msg"],
    [
        [
            {"bls": list(zip([7, 3, 8], [1, 21, 9], strict=True)) + [1, 7]},
            "bls must be a list of tuples of antenna numbers",
        ],
        [{"bls": ("foo", "bar")}, "bls must be a list of tuples of antenna numbers"],
        [{"bls": (5, 1)}, re.escape("Antenna pair (5, 1) does not have any")],
        [
            {"bls": (5, 1, "RR")},
            re.escape("Antenna pair (5, 1, 'RR') does not have any"),
        ],
        [{"bls": (1, 5)}, re.escape("Antenna pair (1, 5) does not have any")],
        [{"bls": (27, 27)}, re.escape("Antenna pair (27, 27) does not have any")],
        [
            {"bls": (7, 1, "RR"), "polarizations": "RR"},
            "Cannot provide any length-3 tuples and also specify polarizations.",
        ],
        [
            {"bls": (7, 1, 7)},
            "The third element in a bl tuple must be a polarization string",
        ],
        [
            {"bls": [(7, 1, "RR"), (1, 5)]},
            "bls tuples must be all length-2, or all length-3.",
        ],
        [{"bls": []}, "bls must be a list of tuples of antenna numbers"],
        [{"bls": [100]}, "Baseline number 100 is not present in the baseline_array"],
    ],
)
def test_select_bls_errors(casa_uvfits, sel_kwargs, err_msg):
    uv_object = casa_uvfits

    with pytest.raises(ValueError, match=err_msg):
        uv_object.select(**sel_kwargs, strict=True)


def test_select_bls_multipol_warning(casa_uvfits):
    uv_object = casa_uvfits

    with check_warnings(
        UserWarning,
        match=[
            "Selected bls contain a mixture of different baselines with different",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_object.select(bls=[(7, 1, "RR"), (1, 2, "LL")])

    assert all(uv_object.ant_1_array == 1)
    assert all(np.isin(uv_object.ant_2_array, [2, 7]))
    assert uv_object.Npols == 2
    assert all(np.isin(uv_object.polarization_array, [-1, -2]))


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("invert", [True, False])
@pytest.mark.parametrize("higher_dims", [True, False])
def test_select_times(casa_uvfits, invert, higher_dims):
    uv_object = casa_uvfits
    old_history = uv_object.history

    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 3, 5, 6, 7, 10, 14]]
    Nblts_selected = np.sum(np.isin(uv_object.time_array, times_to_keep))

    if invert:
        times = unique_times[np.isin(unique_times, times_to_keep, invert=True)]
    else:
        times = times_to_keep

    if higher_dims:
        times = [times]

    uv_object.select(times=times, invert=invert)

    assert len(times_to_keep) == uv_object.Ntimes
    assert Nblts_selected == uv_object.Nblts
    assert np.all(np.isin(uv_object.time_array, times_to_keep))
    assert np.all(np.isin(times_to_keep, uv_object.time_array))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific times using pyuvdata.",
        uv_object.history,
    )


@pytest.mark.parametrize(
    "kwargs, err_msg",
    [
        [{"times": 0, "strict": True}, "Time 0 is not present in the time_array"],
        [{"times": [0], "strict": True}, "Time 0 is not present in the time_array"],
        [{"times": 0, "strict": None}, "No data matching this time selection present"],
    ],
)
def test_select_times_errs(casa_uvfits, kwargs, err_msg):
    uv_object = casa_uvfits
    with pytest.raises(ValueError, match=err_msg):
        uv_object.select(**kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("invert", [True, False])
def test_select_time_range(casa_uvfits, invert):
    uv_object = casa_uvfits
    old_history = uv_object.history
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    time_mask = (unique_times <= time_range[1]) & (unique_times >= time_range[0])
    blt_mask = (uv_object.time_array <= time_range[1]) & (
        uv_object.time_array >= time_range[0]
    )

    if invert:
        time_mask = np.logical_not(time_mask)
        blt_mask = np.logical_not(blt_mask)

    times_selected = unique_times[time_mask]
    Nblts_selected = sum(blt_mask)

    uv_object.select(time_range=time_range, invert=invert)

    assert times_selected.size == uv_object.Ntimes
    assert Nblts_selected == uv_object.Nblts
    assert np.all(np.isin(times_selected, uv_object.time_array))
    assert np.all(np.isin(uv_object.time_array, times_selected))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific times using pyuvdata.",
        uv_object.history,
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.parametrize("invert", [True, False])
@pytest.mark.parametrize("higher_dims", [True, False])
def test_select_lsts(casa_uvfits, invert, higher_dims):
    uv_object = casa_uvfits
    old_history = uv_object.history

    unique_lsts = np.unique(uv_object.lst_array)
    lsts_to_keep = unique_lsts[[0, 3, 5, 6, 7, 10, 14]]
    lsts_to_discard = unique_lsts[np.isin(unique_lsts, lsts_to_keep, invert=True)]
    Nblts_selected = sum(np.isin(uv_object.lst_array, lsts_to_keep))

    select_lsts = lsts_to_discard if invert else lsts_to_keep
    if higher_dims:
        select_lsts = select_lsts[np.newaxis, :]

    uv_object.select(lsts=select_lsts, invert=invert)

    assert len(lsts_to_keep) == uv_object.Ntimes
    assert Nblts_selected == uv_object.Nblts
    assert np.all(np.isin(uv_object.lst_array, lsts_to_keep))
    assert np.all(np.isin(lsts_to_keep, uv_object.lst_array))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object.history,
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_lsts_on_read(casa_uvfits, tmp_path):
    uv_object = casa_uvfits
    testfile = os.path.join(tmp_path, "outtest.uvh5")
    uv_object.write_uvh5(testfile)

    unique_lsts = np.unique(uv_object.lst_array)
    lsts_to_keep = unique_lsts[[0, 3, 5, 6, 7, 10, 14]]

    uv_object.select(lsts=lsts_to_keep)
    uv_object2 = UVData.from_file(testfile, lsts=lsts_to_keep)

    assert uv_object2 == uv_object


def test_consolidate_phase_centers(casa_uvfits):
    uv1 = casa_uvfits
    uv2 = uv1.copy()
    uv3 = uv1.copy()

    init_phase_dict = uv1.phase_center_catalog[0]

    uv1.phase(ra=0, dec=0, phase_frame="icrs", cat_name="foo")
    assert uv1.Nphase == 1

    phase_dict1 = uv1.phase_center_catalog[1]

    with pytest.raises(
        ValueError,
        match="Either the reference_catalog or the other parameter must be set.",
    ):
        uv1._consolidate_phase_center_catalogs()

    uv2.phase(ra=0.5, dec=0.5, phase_frame="fk4", cat_name="bar")
    assert uv1.Nphase == 1

    phase_dict2 = uv2.phase_center_catalog[1]

    uv1._consolidate_phase_center_catalogs(
        other=uv2, reference_catalog=uv3.phase_center_catalog
    )

    assert uv1.phase_center_catalog[0] == init_phase_dict
    assert uv1.phase_center_catalog[1] == phase_dict2
    assert uv1.phase_center_catalog[2] == phase_dict1

    uv2._consolidate_phase_center_catalogs(other=uv1)
    assert uv1._phase_center_catalog == uv2._phase_center_catalog


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.parametrize("select_type", ["range", "specified"])
def test_select_lsts_casa(casa_uvfits, tmp_path, select_type):
    pytest.importorskip("casacore")

    uv_object = casa_uvfits

    unique_lsts = np.unique(uv_object.lst_array)

    if select_type == "range":
        mean_lst = np.mean(unique_lsts)
        lst_range = [np.min(unique_lsts), mean_lst]
        lsts_to_keep = unique_lsts[
            np.nonzero((unique_lsts <= lst_range[1]) & (unique_lsts >= lst_range[0]))
        ]
    elif select_type == "specified":
        lsts_to_keep = unique_lsts[[0, 3, 5, 6, 7, 10, 14]]

    uv_object.select(lsts=lsts_to_keep)

    testfile = os.path.join(tmp_path, "outtest.ms")
    uv_object.write_ms(testfile)

    with check_warnings(
        UserWarning,
        match=[
            (
                'select on read keyword set, but file_type is "ms" which does not '
                "support select on read"
            ),
            "The uvw_array does not match the expected values",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_in = UVData.from_file(testfile, lsts=lsts_to_keep)

    uv_in.history = uv_object.history
    uv_in._consolidate_phase_center_catalogs(
        reference_catalog=uv_object.phase_center_catalog, ignore_name=True
    )
    params_to_update = [
        "dut1",
        "earth_omega",
        "gst0",
        "rdate",
        "timesys",
        "extra_keywords",
        "scan_number_array",
    ]
    for param in params_to_update:
        setattr(uv_in, param, getattr(uv_object, param))

    assert uv_in == uv_object


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_lsts_multi_day(casa_uvfits):
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

    assert utils.history._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )

    return


def test_select_lsts_out_of_range_error(casa_uvfits):
    uv_object = casa_uvfits
    unique_lsts = np.unique(uv_object.lst_array)
    target_lst = np.min(unique_lsts) - 0.1
    with pytest.raises(
        ValueError, match=f"LST {target_lst} is not present in the lst_array"
    ):
        uv_object.select(lsts=[target_lst], strict=True)

    with pytest.raises(
        ValueError, match="No data matching this lst selection present in object."
    ):
        uv_object.select(lsts=[target_lst], strict=None)


def test_select_lsts_too_big(casa_uvfits, tmp_path):
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
    with check_warnings(
        UserWarning,
        [
            "The lsts parameter contained a value greater than 2*pi",
            "The uvw_array does not match the expected values",
            (
                "The lst_array is not self-consistent with the time_array and telescope"
                " location. Consider recomputing with the `set_lsts_from_time_array`"
                " method"
            ),
        ],
    ):
        uv_object2.select(lsts=lsts_to_keep)

    assert len(lsts_to_keep) == uv_object2.Ntimes
    assert Nblts_selected == uv_object2.Nblts
    for lst in lsts_to_keep:
        assert lst in uv_object2.lst_array
    for lst in np.unique(uv_object2.lst_array):
        assert lst in lsts_to_keep

    # check that it's detected in the uvfits reader
    test_filename = os.path.join(tmp_path, "test_bad_lsts.uvfits")
    warn_msg = [
        (
            "The lst_array is not self-consistent with the time_array and telescope "
            "location. Consider recomputing with the `set_lsts_from_time_array` method"
        ),
        "The uvw_array does not match the expected values given the antenna positions.",
    ]
    with check_warnings(UserWarning, match=warn_msg):
        uv_object2.write_uvfits(test_filename)
    with check_warnings(
        UserWarning,
        match=warn_msg
        + [
            (
                "The lst_array is not self-consistent with the time_array and telescope"
                " location. Consider recomputing with the `set_lsts_from_time_array`"
                " method"
            )
        ],
    ):
        UVData.from_file(test_filename)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("invert", [True, False])
def test_select_lst_range(casa_uvfits, invert):
    uv_object = casa_uvfits
    old_history = uv_object.history

    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    lst_range = [np.min(unique_lsts), mean_lst]

    time_mask = (unique_lsts <= lst_range[1]) & (unique_lsts >= lst_range[0])
    blt_mask = (uv_object.lst_array <= lst_range[1]) & (
        uv_object.lst_array >= lst_range[0]
    )
    if invert:
        time_mask = np.logical_not(time_mask)
        blt_mask = np.logical_not(blt_mask)

    lsts_selected = unique_lsts[time_mask]
    Nblts_selected = sum(blt_mask)

    uv_object.select(lst_range=lst_range, invert=invert)

    assert len(lsts_selected) == uv_object.Ntimes
    assert Nblts_selected == uv_object.Nblts
    assert np.all(np.isin(uv_object.lst_array, lsts_selected))
    assert np.all(np.isin(lsts_selected, uv_object.lst_array))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object.history,
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_lst_range_on_read(casa_uvfits, tmp_path):
    uv_object = casa_uvfits
    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    lst_range = [np.min(unique_lsts), mean_lst]

    testfile = os.path.join(tmp_path, "outtest.uvh5")
    uv_object.write_uvh5(testfile)
    uv_object.select(lst_range=lst_range)

    uv_object2 = UVData.from_file(testfile, lst_range=lst_range)

    assert uv_object == uv_object2


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
    with check_warnings(
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

    assert utils.history._check_histories(
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

    assert utils.history._check_histories(
        old_history + "  Downselected to specific lsts using pyuvdata.",
        uv_object2.history,
    )

    return


def test_select_time_range_no_data(casa_uvfits):
    """Check for error associated with times not included in data."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    time_range = [
        np.min(unique_times) - uv_object.integration_time[0] * 2,
        np.min(unique_times) - uv_object.integration_time[0],
    ]
    with pytest.raises(ValueError, match="No elements in time_array"):
        uv_object.select(time_range=time_range, strict=True)
    with pytest.raises(ValueError, match="No data matching this time selection"):
        uv_object.select(time_range=time_range, strict=None)


def test_select_lst_range_no_data(casa_uvfits):
    """Check for error associated with LSTS not included in data."""
    uv_object = casa_uvfits
    unique_lsts = np.unique(uv_object.lst_array)
    lst_range = [np.min(unique_lsts) - 0.2, np.min(unique_lsts) - 0.1]
    with pytest.raises(ValueError, match="No elements in lst_array"):
        uv_object.select(lst_range=lst_range, strict=True)
    with pytest.raises(ValueError, match="No data matching this lst selection"):
        uv_object.select(lst_range=lst_range, strict=None)


def test_select_time_and_time_range(casa_uvfits):
    """Check for error setting times and time_range."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    times_to_keep = unique_times[[0, 3, 5, 6, 7, 10, 14]]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Only one of [times, time_range, lsts, lst_range] may be specified"
        ),
    ):
        uv_object.select(time_range=time_range, times=times_to_keep)


def test_select_time_range_one_elem(casa_uvfits):
    """Check for error if time_range not length 2."""
    uv_object = casa_uvfits
    unique_times = np.unique(uv_object.time_array)
    mean_time = np.mean(unique_times)
    time_range = [np.min(unique_times), mean_time]
    with pytest.raises(ValueError, match="time_range must be length 2"):
        uv_object.select(time_range=time_range[0])


def test_select_lst_range_one_elem(casa_uvfits):
    """Check for error if time_range not length 2."""
    uv_object = casa_uvfits
    unique_lsts = np.unique(uv_object.lst_array)
    mean_lst = np.mean(unique_lsts)
    lst_range = [np.min(unique_lsts), mean_lst]
    with pytest.raises(ValueError, match="lst_range must be length 2"):
        uv_object.select(lst_range=lst_range[0])


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_frequencies_writeerrors(casa_uvfits, tmp_path):
    uv_object = casa_uvfits
    old_history = uv_object.history
    freqs_to_keep = uv_object.freq_array[np.arange(12, 22)]

    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == uv_object2.Nfreqs
    for f in freqs_to_keep:
        assert f in uv_object2.freq_array
    for f in np.unique(uv_object2.freq_array):
        assert f in freqs_to_keep

    assert utils.history._check_histories(
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

    assert utils.history._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check that selecting one frequency works
    uv_object2 = uv_object.copy()
    uv_object2.select(frequencies=freqs_to_keep[0])
    assert uv_object2.Nfreqs == 1
    assert freqs_to_keep[0] in uv_object2.freq_array
    for f in uv_object2.freq_array:
        assert f in [freqs_to_keep[0]]

    assert utils.history._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object2.history,
    )

    # check for errors associated with frequencies not included in data
    with pytest.raises(ValueError, match="Frequency "):
        uv_object.select(
            frequencies=[np.max(uv_object.freq_array) + uv_object.channel_width],
            strict=True,
        )
    with pytest.raises(ValueError, match="No data matching this frequency selection"):
        uv_object.select(
            frequencies=[np.max(uv_object.freq_array) + uv_object.channel_width],
            strict=None,
        )
    write_file_miriad = str(tmp_path / "select_test")
    write_file_uvfits = str(tmp_path / "select_test.uvfits")

    # check for warnings and errors associated with unevenly spaced or
    # non-contiguous frequencies
    uv_object2 = uv_object.copy()
    with check_warnings(
        UserWarning,
        [
            "Selected frequencies are not evenly spaced",
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv_object2.select(
            frequencies=uv_object2.freq_array[[0, 5, 6]], warn_spacing=True
        )

    with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
        uv_object2.write_uvfits(write_file_uvfits)

    if hasmiriad:
        with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
            uv_object2.write_miriad(write_file_miriad)

    uv_object2 = uv_object.copy()
    with check_warnings(
        UserWarning,
        [
            "Selected frequencies are not contiguous",
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv_object2.select(
            frequencies=uv_object2.freq_array[[0, 2, 4]], warn_spacing=True
        )

    with pytest.raises(
        ValueError,
        match="The frequencies are separated by more than their channel width",
    ):
        uv_object2.write_uvfits(write_file_uvfits)

    if hasmiriad:
        with pytest.raises(
            ValueError,
            match="The frequencies are separated by more than their channel width",
        ):
            uv_object2.write_miriad(write_file_miriad)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("invert", [True, False])
@pytest.mark.parametrize("higher_dims", [True, False])
def test_select_freq_chans(casa_uvfits, invert, higher_dims):
    uv_object = casa_uvfits
    old_history = uv_object.history

    chans_to_keep = np.arange(12, 22)
    chans_to_discard = np.nonzero(
        np.isin(np.arange(uv_object.Nfreqs), chans_to_keep, invert=True)
    )[0]
    freqs_to_keep = uv_object.freq_array[chans_to_keep]

    select_chans = chans_to_discard if invert else chans_to_keep
    if higher_dims:
        select_chans = select_chans[np.newaxis, :]

    uv_object.select(freq_chans=select_chans, invert=invert)

    assert len(chans_to_keep) == uv_object.Nfreqs
    assert np.all(np.isin(uv_object.freq_array, freqs_to_keep))
    assert np.all(np.isin(freqs_to_keep, uv_object.freq_array))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        uv_object.history,
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("invert", [True, False])
def test_select_freqs_and_chans(casa_uvfits, invert):
    uv_object = casa_uvfits
    # Test selecting both channels and frequencies
    freqs = uv_object.freq_array[np.arange(20, 30)]  # Overlaps with chans
    chans = np.arange(12, 25)
    exp_freqs = uv_object.freq_array[12:30]
    if invert:
        exp_freqs = uv_object.freq_array[
            np.isin(uv_object.freq_array, exp_freqs, invert=True)
        ]

    # Strict=None to silence any errors about freq spacing
    uv_object.select(frequencies=freqs, freq_chans=chans, invert=invert, strict=None)

    assert len(exp_freqs) == uv_object.Nfreqs
    assert np.all(np.isin(uv_object.freq_array, exp_freqs))
    assert np.all(np.isin(exp_freqs, uv_object.freq_array))


def test_select_spws(sma_mir):
    # This tests inversion and not all in one go, along with underlying handling
    # of freq_chans behavior.
    sma_copy1 = sma_mir.copy()
    sma_copy2 = sma_mir.copy()
    sma_copy3 = sma_mir.copy()

    sma_mir.select(spws=[-4, -3, -2, -1], inplace=True)
    sma_copy1.select(spws=[1, 2, 3, 4], invert=True, inplace=True)

    assert sma_mir == sma_copy1

    sma_copy2.select(freq_chans=np.arange(4 * 16384))

    # Histories should be different, since one was spws, the other was freqs
    assert sma_mir.history != sma_copy2.history
    assert "Downselected to specific spectral windows" in sma_mir.history
    sma_mir.__eq__(sma_copy2, allowed_failures=["history"])

    # Test list handling
    sma_copy3.spw_array = sma_copy3.spw_array.tolist()
    sma_copy3.select(freq_chans=np.arange(4 * 16384, 8 * 16384), invert=True)
    sma_copy3.spw_array = np.asarray(sma_copy3.spw_array)

    assert sma_mir.history != sma_copy3.history
    assert "Downselected to specific frequencies" in sma_copy3.history
    sma_mir.__eq__(sma_copy3, allowed_failures=["history"])
    assert sma_copy3 == sma_copy2


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "pol_list", ([-5, -6], ["xx", "yy"], ["nn", "ee"], [[-5, -6]], -5, [-6], "xx", "ee")
)
@pytest.mark.parametrize("invert", [True, False])
def test_select_polarizations(hera_uvh5, pol_list, invert):
    uv_object = hera_uvh5
    old_history = uv_object.history

    if invert and np.array(pol_list).size == 2:
        with pytest.raises(ValueError, match="No data matching this polarization"):
            # Deselecting all pols -- catch the error and bail
            uv_object.select(polarizations=pol_list, invert=invert, strict=None)
        return

    uv_object.select(polarizations=pol_list, invert=invert, strict=None)

    if not isinstance(pol_list, list):
        pol_list = [pol_list]
    elif isinstance(pol_list[0], list):
        pol_list = pol_list[0]

    assert len(pol_list) == uv_object.Npols
    pol_int_list = [None] * len(pol_list)
    for idx, p in enumerate(pol_list):
        if isinstance(p, int):
            pol_int_list[idx] = p
        else:
            pol_int_list[idx] = utils.polstr2num(
                p, x_orientation=uv_object.telescope.get_x_orientation_from_feeds()
            )

    assert np.all(np.isin(pol_int_list, uv_object.polarization_array, invert=invert))
    assert np.all(np.isin(uv_object.polarization_array, pol_int_list, invert=invert))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific polarizations using pyuvdata.",
        uv_object.history,
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [
            {"polarizations": [-5, -6], "strict": True},
            "Polarization -5 is not present in the polarization_array",
        ],
        [
            {"polarizations": -7, "strict": None},
            "No data matching this polarization selection exists.",
        ],
    ],
)
def test_select_polarizations_errors(casa_uvfits, kwargs, err_msg):
    uv_object = casa_uvfits
    with pytest.raises(ValueError, match=err_msg):
        uv_object.select(**kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_polarization_uvfits_error(casa_uvfits, tmp_path):
    uv_object = casa_uvfits
    # check for warnings and errors associated with unevenly spaced polarizations
    with check_warnings(
        UserWarning,
        [
            "Selected polarization values are not evenly spaced",
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv_object.select(
            polarizations=uv_object.polarization_array[[0, 1, 3]], warn_spacing=True
        )
    write_file_uvfits = str(tmp_path / "select_test.uvfits")
    with pytest.raises(
        ValueError, match="The polarization values are not evenly spaced"
    ):
        uv_object.write_uvfits(write_file_uvfits)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select(casa_uvfits):
    # now test selecting along all axes at once
    uv_object = casa_uvfits

    # Set the scan numbers so that we can check to make sure they are selected correctly
    uv_object._set_scan_numbers()

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
    ants_to_keep = np.array([12, 7, 21, 27, 3, 28, 8, 15])

    ant_pairs_to_keep = [(3, 12), (21, 27), (7, 8), (4, 28), (15, 7)]
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]

    freqs_to_keep = uv_object.freq_array[np.arange(31, 39)]
    unique_times = np.unique(uv_object.time_array)
    times_to_keep = unique_times[[0, 2, 6, 8, 10, 13, 14]]

    pols_to_keep = [-1, -3]

    # Independently count blts that should be selected
    blts_blt_select = [i in blt_inds for i in np.arange(uv_object.Nblts)]
    blts_ant_select = [
        (a1 in ants_to_keep) & (a2 in ants_to_keep)
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array, strict=True)
    ]
    blts_pair_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array, strict=True)
    ]
    blts_time_select = [t in times_to_keep for t in uv_object.time_array]
    Nblts_select = np.sum(
        [
            bi & (ai & pi) & ti
            for (bi, ai, pi, ti) in zip(
                blts_blt_select,
                blts_ant_select,
                blts_pair_select,
                blts_time_select,
                strict=True,
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

    assert utils.history._check_histories(
        old_history + "  Downselected to "
        "specific baseline-times, antennas, "
        "antenna pairs, times, frequencies, "
        "polarizations using pyuvdata.",
        uv_object2.history,
    )

    # test that a ValueError is raised if the selection eliminates all blts
    with pytest.raises(
        ValueError, match="No baseline-times were found that match criteria"
    ):
        uv_object.select(times=unique_times[0], antenna_nums=1)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_with_lst(casa_uvfits):
    # now test selecting along all axes at once, but with LST instead of times
    uv_object = casa_uvfits

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
    ants_to_keep = np.array([12, 7, 21, 27, 3, 28, 8, 15])
    ant_pairs_to_keep = [(3, 12), (21, 27), (7, 8), (4, 28), (15, 7)]
    sorted_pairs_to_keep = [sort_bl(p) for p in ant_pairs_to_keep]
    freqs_to_keep = uv_object.freq_array[np.arange(31, 39)]
    unique_lsts = np.unique(uv_object.lst_array)
    lsts_to_keep = unique_lsts[[0, 2, 6, 8, 10, 13, 14]]
    pols_to_keep = [-1, -3]

    # Independently count blts that should be selected
    blts_blt_select = [i in blt_inds for i in np.arange(uv_object.Nblts)]
    blts_ant_select = [
        (a1 in ants_to_keep) & (a2 in ants_to_keep)
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array, strict=True)
    ]
    blts_pair_select = [
        sort_bl((a1, a2)) in sorted_pairs_to_keep
        for (a1, a2) in zip(uv_object.ant_1_array, uv_object.ant_2_array, strict=True)
    ]
    blts_lst_select = [lst in lsts_to_keep for lst in uv_object.lst_array]
    Nblts_select = np.sum(
        [
            bi & (ai & pi) & li
            for (bi, ai, pi, li) in zip(
                blts_blt_select,
                blts_ant_select,
                blts_pair_select,
                blts_lst_select,
                strict=True,
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

    assert utils.history._check_histories(
        old_history + "  Downselected to "
        "specific baseline-times, antennas, "
        "antenna pairs, lsts, frequencies, "
        "polarizations using pyuvdata.",
        uv_object2.history,
    )

    # test that a ValueError is raised if the selection eliminates all blts
    with pytest.raises(
        ValueError, match="No baseline-times were found that match criteria"
    ):
        uv_object.select(lsts=unique_lsts[0], antenna_nums=1)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_not_inplace(casa_uvfits):
    # Test non-inplace select
    uv_object = casa_uvfits
    old_history = uv_object.history
    uv1 = uv_object.select(freq_chans=np.arange(32), inplace=False)
    uv1 += uv_object.select(freq_chans=np.arange(32, 64), inplace=False)
    assert utils.history._check_histories(
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
def test_conjugate_bls(casa_uvfits, metadata_only):
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")

    if not metadata_only:
        uv1 = casa_uvfits
    else:
        uv1 = UVData()
        uv1.read_uvfits(testfile, read_data=False)
    if metadata_only:
        assert uv1.metadata_only

    # file comes in with ant1<ant2
    assert np.min(uv1.ant_2_array - uv1.ant_1_array) >= 0

    # check everything swapped & conjugated when go to ant2<ant1
    uv2 = uv1.copy()
    uv2.conjugate_bls("ant2<ant1")
    assert np.min(uv2.ant_1_array - uv2.ant_2_array) >= 0

    np.testing.assert_allclose(uv1.ant_1_array, uv2.ant_2_array)
    np.testing.assert_allclose(uv1.ant_2_array, uv2.ant_1_array)
    np.testing.assert_allclose(
        uv1.uvw_array,
        -1 * uv2.uvw_array,
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )

    if not metadata_only:
        # complicated because of the polarization swaps
        # polarization_array = [-1 -2 -3 -4]
        np.testing.assert_allclose(
            uv1.data_array[:, :, :2],
            np.conj(uv2.data_array[:, :, :2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        np.testing.assert_allclose(
            uv1.data_array[:, :, 2],
            np.conj(uv2.data_array[:, :, 3]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        np.testing.assert_allclose(
            uv1.data_array[:, :, 3],
            np.conj(uv2.data_array[:, :, 2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

    # check everything returned to original values with original convention
    uv2.conjugate_bls("ant1<ant2")
    assert uv1 == uv2

    # conjugate a particular set of blts
    blts_to_conjugate = np.arange(uv2.Nblts // 2)
    blts_not_conjugated = np.arange(uv2.Nblts // 2, uv2.Nblts)
    uv2.conjugate_bls(blts_to_conjugate)

    np.testing.assert_allclose(
        uv1.ant_1_array[blts_to_conjugate], uv2.ant_2_array[blts_to_conjugate]
    )
    np.testing.assert_allclose(
        uv1.ant_2_array[blts_to_conjugate], uv2.ant_1_array[blts_to_conjugate]
    )
    np.testing.assert_allclose(
        uv1.ant_1_array[blts_not_conjugated], uv2.ant_1_array[blts_not_conjugated]
    )
    np.testing.assert_allclose(
        uv1.ant_2_array[blts_not_conjugated], uv2.ant_2_array[blts_not_conjugated]
    )

    np.testing.assert_allclose(
        uv1.uvw_array[blts_to_conjugate],
        -1 * uv2.uvw_array[blts_to_conjugate],
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )
    np.testing.assert_allclose(
        uv1.uvw_array[blts_not_conjugated],
        uv2.uvw_array[blts_not_conjugated],
        rtol=uv1._uvw_array.tols[0],
        atol=uv1._uvw_array.tols[1],
    )
    if not metadata_only:
        # complicated because of the polarization swaps
        # polarization_array = [-1 -2 -3 -4]
        np.testing.assert_allclose(
            uv1.data_array[blts_to_conjugate, :, :2],
            np.conj(uv2.data_array[blts_to_conjugate, :, :2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )
        np.testing.assert_allclose(
            uv1.data_array[blts_not_conjugated, :, :2],
            uv2.data_array[blts_not_conjugated, :, :2],
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        np.testing.assert_allclose(
            uv1.data_array[blts_to_conjugate, :, 2],
            np.conj(uv2.data_array[blts_to_conjugate, :, 3]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )
        np.testing.assert_allclose(
            uv1.data_array[blts_not_conjugated, :, 2],
            uv2.data_array[blts_not_conjugated, :, 2],
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

        np.testing.assert_allclose(
            uv1.data_array[blts_to_conjugate, :, 3],
            np.conj(uv2.data_array[blts_to_conjugate, :, 2]),
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )
        np.testing.assert_allclose(
            uv1.data_array[blts_not_conjugated, :, 3],
            uv2.data_array[blts_not_conjugated, :, 3],
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )

    # check uv half plane conventions
    uv2.conjugate_bls("u<0", use_enu=False)
    assert np.max(uv2.uvw_array[:, 0]) <= 0

    uv2.conjugate_bls("u>0", use_enu=False)
    assert np.min(uv2.uvw_array[:, 0]) >= 0

    uv2.conjugate_bls("v<0", use_enu=False)
    assert np.max(uv2.uvw_array[:, 1]) <= 0

    uv2.conjugate_bls("v>0", use_enu=False)
    assert np.min(uv2.uvw_array[:, 1]) >= 0

    # unphase to drift to test using ENU positions
    uv2.unproject_phase(use_ant_pos=True)
    uv2.conjugate_bls("u<0")
    assert np.max(uv2.uvw_array[:, 0]) <= 0

    uv2.conjugate_bls("u>0")
    assert np.min(uv2.uvw_array[:, 0]) >= 0

    uv2.conjugate_bls("v<0")
    assert np.max(uv2.uvw_array[:, 1]) <= 0

    uv2.conjugate_bls("v>0")
    assert np.min(uv2.uvw_array[:, 1]) >= 0

    # test errors
    with pytest.raises(ValueError, match="convention must be one of"):
        uv2.conjugate_bls("foo")

    with pytest.raises(ValueError, match="If convention is an index array"):
        uv2.conjugate_bls(np.arange(5) - 1)

    with pytest.raises(ValueError, match="If convention is an index array"):
        uv2.conjugate_bls([uv2.Nblts])


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_reorder_pols(casa_uvfits):
    # Test function to fix polarization order
    uv1 = casa_uvfits

    uv2 = uv1.copy()
    uv3 = uv1.copy()
    # reorder uv2 manually
    order = [1, 3, 2, 0]
    uv2.polarization_array = uv2.polarization_array[order]
    uv2.data_array = uv2.data_array[:, :, order]
    uv2.nsample_array = uv2.nsample_array[:, :, order]
    uv2.flag_array = uv2.flag_array[:, :, order]
    uv1.reorder_pols(order)
    assert uv1 == uv2

    # Restore original order
    uv1 = uv3.copy()
    uv2.reorder_pols()
    assert uv1 == uv2

    uv1.reorder_pols("AIPS")
    # check that we have aips ordering
    aips_pols = np.array([-1, -2, -3, -4]).astype(int)
    assert np.all(uv1.polarization_array == aips_pols)

    uv2 = uv1.copy()
    uv2.reorder_pols("CASA")
    # check that we have casa ordering
    casa_pols = np.array([-1, -3, -4, -2]).astype(int)
    assert np.all(uv2.polarization_array == casa_pols)
    order = np.array([0, 2, 3, 1])
    assert np.all(uv2.data_array == uv1.data_array[:, :, order])
    assert np.all(uv2.flag_array == uv1.flag_array[:, :, order])

    uv2.reorder_pols("AIPS")
    # check that we have aips ordering again
    assert uv1 == uv2

    # check error on unknown order
    with pytest.raises(
        ValueError,
        match="order must be one of: 'AIPS', 'CASA', or an index array of length Npols",
    ):
        uv2.reorder_pols({"order": "foo"})

    # check error if order is an array of the wrong length
    with pytest.raises(ValueError, match="If order is an index array, it must"):
        uv2.reorder_pols([3, 2, 1])


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
    with pytest.raises(ValueError, match=msg):
        casa_uvfits.reorder_blts(order, minor_order=minor_order)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_reorder_blts(casa_uvfits):
    uv1 = casa_uvfits

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
        np.testing.assert_allclose(bls_1[bl_inds], bls_2)

        uvw_1 = uv1.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        uvw_2 = uv2.uvw_array[np.where(uv2.time_array == this_time)[0], :]
        assert uvw_1.shape == uvw_2.shape
        np.testing.assert_allclose(
            uvw_1[bl_inds, :],
            uvw_2,
            rtol=uv1._uvw_array.tols[0],
            atol=uv1._uvw_array.tols[1],
        )

        data_1 = uv1.data_array[np.where(uv2.time_array == this_time)[0]]
        data_2 = uv2.data_array[np.where(uv2.time_array == this_time)[0]]
        assert data_1.shape == data_2.shape
        np.testing.assert_allclose(
            data_1[bl_inds],
            data_2,
            rtol=uv1._data_array.tols[0],
            atol=uv1._data_array.tols[1],
        )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "args1,args2",
    [
        [{"order": "time"}, {"order": "time", "minor_order": "time"}],
        [{"order": "time"}, {"order": "time", "minor_order": "ant1"}],
        [{"order": "time"}, {"order": "time", "minor_order": "baseline"}],
        [{}, {"order": np.arange(1360)}],  # casa_uvfits already in default order
        [{"order": "time"}, {"order": "time", "conj_convention": "ant1<ant2"}],
        [{"autos_first": True}, {"autos_first": False}],  # No autos in this file
    ],
)
def test_reorder_blts_equiv(casa_uvfits, args1, args2):
    """
    Test that sorting orders that _should_ be equivalent actually are
    """
    uv1 = casa_uvfits
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
@pytest.mark.parametrize("autos_first", [True, False])
def test_reorder_blts_sort_order(
    hera_uvh5, order, m_order, check_tuple, check_attr, autos_first
):
    hera_uvh5.reorder_blts(order, minor_order=m_order, autos_first=autos_first)
    assert hera_uvh5.blt_order == check_tuple
    if isinstance(order, str) and autos_first:
        auto_inds = np.nonzero(hera_uvh5.ant_1_array == hera_uvh5.ant_2_array)[0]
        cross_inds = np.nonzero(hera_uvh5.ant_1_array != hera_uvh5.ant_2_array)[0]

        assert np.max(auto_inds) < np.min(cross_inds)
        for item in check_attr:
            attr_arr = getattr(hera_uvh5, item)
            attr_auto = attr_arr[: np.min(cross_inds)]
            attr_cross = attr_arr[np.min(cross_inds) :]
            assert np.all(np.diff(attr_auto) >= 0)
            assert np.all(np.diff(attr_cross) >= 0)

    else:
        for item in check_attr:
            assert np.all(np.diff(getattr(hera_uvh5, item)) >= 0)


@pytest.mark.parametrize(
    "arg_dict,msg",
    [
        [{"spord": [1]}, "Index array for spw_order must contain all indices for"],
        [{"spord": "karto"}, "spw_order can only be one of 'number', '-number',"],
        [{"chord": [1]}, "Index array for channel_order must contain all indices"],
        [{"chord": "karto"}, "channel_order can only be one of 'freq' or '-freq'"],
    ],
)
def test_reorder_freqs_errs(sma_mir, arg_dict, msg):
    """
    Verify that appropriate errors are thrown when providing bad arguments to
    reorder_freqs.
    """
    with pytest.raises(ValueError, match=msg):
        sma_mir.reorder_freqs(
            spw_order=arg_dict.get("spord"), channel_order=arg_dict.get("chord")
        )


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
    with check_warnings(UserWarning, msg):
        sma_mir.reorder_freqs(
            select_spw=arg_dict.get("selspw"),
            spw_order=arg_dict.get("spword"),
            channel_order=arg_dict.get("chord"),
        )

    assert sma_mir == sma_mir_main


@pytest.mark.parametrize(
    "sel_spw,spord,chord",
    [
        [[1, [1]], [None] * 2, ["freq"] * 2],
        [[None] * 2, ["number", "freq"], ["freq"] * 2],
        [[None] * 2, ["-number", "-freq"], ["-freq"] * 2],
    ],
)
def test_reorder_freqs_equal(sma_mir, sel_spw, spord, chord):
    # Create a dummy copy that we can muck with at will
    sma_mir_copy = sma_mir.copy()

    # Make sure that arrays and ints work for select_spw
    sma_mir.reorder_freqs(
        select_spw=sel_spw[0], spw_order=spord[0], channel_order=chord[0]
    )
    sma_mir_copy.reorder_freqs(
        select_spw=sel_spw[1], spw_order=spord[1], channel_order=chord[1]
    )
    assert sma_mir == sma_mir_copy


def test_reorder_freqs_flipped(sma_mir):
    """
    Test that when sorting the data in ways that _should_ flip the frequency
    axis, that it actually does so.
    """
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


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_reorder_freqs_eq_coeffs(casa_uvfits):
    # No test datasets to examine this with, so let's generate some mock data,
    # with a pre-determined order that we can flip
    casa_uvfits.reorder_freqs(channel_order="-freq")
    casa_uvfits.eq_coeffs = np.tile(
        np.arange(casa_uvfits.Nfreqs, dtype=float), (casa_uvfits.telescope.Nants, 1)
    )
    # modify the channel widths so we can check them too
    casa_uvfits.channel_width += np.arange(casa_uvfits.Nfreqs, dtype=float)
    casa_uvfits.reorder_freqs(channel_order="freq")
    assert np.all(np.diff(casa_uvfits.eq_coeffs, axis=1) == -1)
    assert np.all(np.diff(casa_uvfits.channel_width) == -1)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_sum_vis(casa_uvfits):
    # check sum_vis
    uv_full = casa_uvfits

    uv_half = uv_full.copy()
    uv_half.data_array = uv_full.data_array / 2
    uv_half_mod = uv_half.copy()
    uv_half_mod.history += " testing the history. "
    uv_half_mod.filename = ["foo.uvfits"]
    uv_summed = uv_half.sum_vis(uv_half_mod)

    assert np.array_equal(uv_summed.data_array, uv_full.data_array)
    assert utils.history._check_histories(
        uv_half.history + " Visibilities summed using pyuvdata. Unique part of second "
        "object history follows.  testing the history.",
        uv_summed.history,
    )
    # add a test for full coverage of _combine_history_addition function
    assert (
        utils.history._combine_history_addition(
            uv_half.history
            + " Visibilities summed using pyuvdata. Unique part of second "
            "object history follows.  testing the history.",
            uv_summed.history,
        )
        is None
    )

    uv_summed = uv_half.sum_vis(uv_half_mod, verbose_history=True)

    assert np.array_equal(uv_summed.data_array, uv_full.data_array)
    assert utils.history._check_histories(
        uv_half.history
        + " Visibilities summed using pyuvdata. Second object history follows. "
        + uv_half_mod.history,
        uv_summed.history,
    )

    # check diff_vis
    uv_diffed = uv_full.diff_vis(uv_half)

    assert np.array_equal(uv_diffed.data_array, uv_half.data_array)
    assert utils.history._check_histories(
        uv_full.history + " Visibilities differenced using pyuvdata.", uv_diffed.history
    )

    # check in place
    uv_summed.diff_vis(uv_half, inplace=True)
    assert np.array_equal(uv_summed.data_array, uv_half.data_array)

    # check extra_keywords handling
    uv_keys = uv_full.copy()
    uv_keys.extra_keywords["test_key"] = "test_value"
    uv_keys.extra_keywords["SPECSYS"] = "altered_value"
    with check_warnings(
        UserWarning,
        [
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
            (
                "Keyword SPECSYS in _extra_keywords is different in the two objects. "
                "Taking the first object's entry."
            ),
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv_merged_keys = uv_keys.sum_vis(uv_full)
    assert uv_merged_keys.extra_keywords["test_key"] == "test_value"
    assert uv_merged_keys.extra_keywords["SPECSYS"] == "altered_value"

    # check override_params
    uv_overrides = uv_full.copy()
    uv_overrides.timesys = "foo"
    uv_overrides.telescope.location = EarthLocation.from_geocentric(
        -1601183.15377712, -5042003.74810822, 3554841.17192104, unit="m"
    )
    uv_overrides_2 = uv_overrides.sum_vis(
        uv_full, override_params=["timesys", "telescope"]
    )

    assert uv_overrides_2.timesys == "foo"
    np.testing.assert_allclose(
        uv_overrides_2.telescope._location.xyz(),
        np.array([-1601183.15377712, -5042003.74810822, 3554841.17192104]),
        rtol=uv_overrides.telescope._location.tols[0],
        atol=uv_overrides.telescope._location.tols[1],
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "attr_to_get,attr_to_set,arg_dict,msg",
    [
        [[], {}, {"override": ["fake"]}, "Provided parameter fake is not a recogniza"],
        [[], {"__class__": UVCal}, {}, "Only UVData (or subclass) objects can be"],
        [[], {"timesys": "foo"}, {"inplace": True}, "UVParameter timesys does"],
    ],
)
def test_sum_vis_errors(hera_uvh5, attr_to_get, attr_to_set, arg_dict, msg):
    uv1 = hera_uvh5.copy()
    uv2 = hera_uvh5
    for method in attr_to_get:
        getattr(uv2, method)()
    for attr in attr_to_set:
        setattr(uv2, attr, attr_to_set[attr])

    with pytest.raises(ValueError, match=re.escape(msg)):
        uv1.sum_vis(
            uv2,
            override_params=arg_dict.get("override"),
            inplace=arg_dict.get("inplace"),
        )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_freq(casa_uvfits):
    uv_full = casa_uvfits

    uv1 = uv_full.select(freq_chans=np.arange(0, 32), inplace=False)
    uv2 = uv_full.select(freq_chans=np.arange(32, 64), inplace=False)
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Other order
    uv1 = uv_full.select(freq_chans=np.arange(0, 32), inplace=False)
    with check_warnings(
        UserWarning, match=["The uvw_array does not match the expected values"] * 3
    ):
        uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_pols(casa_uvfits):
    uv_full = casa_uvfits

    uv1 = uv_full.select(polarizations=uv_full.polarization_array[0:2], inplace=False)
    uv2 = uv_full.select(polarizations=uv_full.polarization_array[2:4], inplace=False)
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv_new = uv1 + uv2
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Unique part of next "
        "object history follows.  testing the history.",
        uv_new.history,
    )
    uv_new.history = uv_full.history
    assert uv_new == uv_full

    uv_new = uv1.__add__(uv2, verbose_history=True)
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Next object history "
        "follows.  " + uv2.history,
        uv_new.history,
    )

    # Other order
    uv2 += uv1
    uv2.history = uv_full.history
    assert uv2 == uv_full

    uv2 = uv_full.select(polarizations=uv_full.polarization_array[3], inplace=False)
    uv1.__iadd__(uv2)
    uv_ref = uv_full.select(polarizations=uv1.polarization_array[0:3], inplace=False)
    uv1.history = uv_ref.history
    assert uv1 == uv_ref


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_times(casa_uvfits):
    uv_full = casa_uvfits

    times = np.unique(uv_full.time_array)
    uv1 = uv_full.select(times=times[0 : len(times) // 2], inplace=False)
    uv2 = uv_full.select(times=times[len(times) // 2 :], inplace=False)
    # Add without inplace
    uv1 = uv1 + uv2
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_bls(casa_uvfits):
    uv_full = casa_uvfits

    ant_list = list(range(15))  # Roughly half the antennas in the data
    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv_full.Nblts) if uv_full.ant_1_array[i] in ant_list]
    ind2 = [i for i in range(uv_full.Nblts) if uv_full.ant_1_array[i] not in ant_list]
    uv1 = uv_full.select(blt_inds=ind1, inplace=False)
    uv2 = uv_full.select(blt_inds=ind2, inplace=False)
    uv1 += uv2
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add baselines - out of order
    ants = uv_full.get_ants()
    ants1 = ants[0:6]
    ants2 = ants[6:12]
    ants3 = ants[12:]

    # All blts where ant_1 is in list
    ind1 = [i for i in range(uv_full.Nblts) if uv_full.ant_1_array[i] in ants1]
    ind2 = [i for i in range(uv_full.Nblts) if uv_full.ant_1_array[i] in ants2]
    ind3 = [i for i in range(uv_full.Nblts) if uv_full.ant_1_array[i] in ants3]
    uv1 = uv_full.select(blt_inds=ind1, inplace=False)
    uv2 = uv_full.select(blt_inds=ind2, inplace=False)
    uv3 = uv_full.select(blt_inds=ind3, inplace=False)
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
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata. Combined data along "
        "baseline-time axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_multi_axis(casa_uvfits):
    uv_full = casa_uvfits

    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1 = uv_full.select(
        times=times[0 : len(times) // 2],
        polarizations=uv_full.polarization_array[0:2],
        inplace=False,
    )
    uv2 = uv_full.select(
        times=times[len(times) // 2 :],
        polarizations=uv_full.polarization_array[2:4],
        inplace=False,
    )
    uv1 += uv2
    assert utils.history._check_histories(
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
    uv_ref.data_array[blt_ind1, :, 2:] = 0.0
    uv_ref.nsample_array[blt_ind1, :, 2:] = 0.0
    uv_ref.flag_array[blt_ind1, :, 2:] = True
    uv_ref.data_array[blt_ind2, :, 0:2] = 0.0
    uv_ref.nsample_array[blt_ind2, :, 0:2] = 0.0
    uv_ref.flag_array[blt_ind2, :, 0:2] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Another combo
    uv_ref = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1 = uv_full.select(
        times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32), inplace=False
    )
    uv2 = uv_full.select(
        times=times[len(times) // 2 :], freq_chans=np.arange(32, 64), inplace=False
    )
    uv1 += uv2
    assert utils.history._check_histories(
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
    uv_ref.data_array[blt_ind1, 32:, :] = 0.0
    uv_ref.nsample_array[blt_ind1, 32:, :] = 0.0
    uv_ref.flag_array[blt_ind1, 32:, :] = True
    uv_ref.data_array[blt_ind2, 0:32, :] = 0.0
    uv_ref.nsample_array[blt_ind2, 0:32, :] = 0.0
    uv_ref.flag_array[blt_ind2, 0:32, :] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_auto_cross(hera_uvh5):
    # test add of autocorr-only and crosscorr-only objects
    uv_full = hera_uvh5
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
def test_add_unprojected(casa_uvfits):
    uv_full = casa_uvfits
    uv_full.unproject_phase()

    # Add frequencies
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1 += uv2
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
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
    assert utils.history._check_histories(
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
    assert utils.history._check_histories(
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
    assert utils.history._check_histories(
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
    assert utils.history._check_histories(
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
    uv_ref.data_array[blt_ind1, :, 2:] = 0.0
    uv_ref.nsample_array[blt_ind1, :, 2:] = 0.0
    uv_ref.flag_array[blt_ind1, :, 2:] = True
    uv_ref.data_array[blt_ind2, :, 0:2] = 0.0
    uv_ref.nsample_array[blt_ind2, :, 0:2] = 0.0
    uv_ref.flag_array[blt_ind2, :, 0:2] = True
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
    assert utils.history._check_histories(
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
    uv_ref.data_array[blt_ind1, 32:, :] = 0.0
    uv_ref.nsample_array[blt_ind1, 32:, :] = 0.0
    uv_ref.flag_array[blt_ind1, 32:, :] = True
    uv_ref.data_array[blt_ind2, 0:32, :] = 0.0
    uv_ref.nsample_array[blt_ind2, 0:32, :] = 0.0
    uv_ref.flag_array[blt_ind2, 0:32, :] = True
    uv1.history = uv_full.history
    assert uv1 == uv_ref

    # Add without inplace
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    times = np.unique(uv_full.time_array)
    uv1.select(times=times[0 : len(times) // 2])
    uv2.select(times=times[len(times) // 2 :])
    uv1 = uv1 + uv2
    assert utils.history._check_histories(
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
    uv1.__add__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=[0])
    uv2.select(freq_chans=[3])
    uv1.__iadd__(uv2)

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[3])
    with check_warnings(None, None):
        uv1.__iadd__(uv2)

    # Combining histories
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(polarizations=uv1.polarization_array[0:2])
    uv2.select(polarizations=uv2.polarization_array[2:4])
    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv_new = uv1 + uv2
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata.  Unique part of next "
        "object history follows.  testing the history.",
        uv_new.history,
    )
    uv_new.history = uv_full.history
    assert uv_new == uv_full

    uv_new = uv1.__add__(uv2, verbose_history=True)
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Next object history "
        "follows." + uv2.history,
        uv_new.history,
    )


def test_check_flex_spw_contiguous(sma_mir, casa_uvfits):
    """
    Verify that check_flex_spw_contiguous passes silently as expected when windows are
    contiguous.
    """
    sma_mir._check_flex_spw_contiguous()

    casa_uvfits._check_flex_spw_contiguous()


def test_check_flex_spw_contiguous_error(sma_mir):
    """
    Verify that check_flex_spw_contiguous errors as expected when windows are
    not contiguous.
    """
    sma_mir.flex_spw_id_array[0] = 1
    with pytest.raises(
        ValueError,
        match=(
            "Channels from different spectral windows are interspersed with "
            "one another, rather than being grouped together along the "
            "frequency axis. Most file formats do not support such "
            "non-grouping of data."
        ),
    ):
        sma_mir._check_flex_spw_contiguous()


def test_check_flex_spw_contiguous_no_flex_spw(hera_uvh5):
    """
    Verify that with a non-flex-spw dataset, the _check_flex_spw_contiguous
    check returns as True.
    """
    hera_uvh5._check_flex_spw_contiguous()


@pytest.mark.parametrize(
    "chan_width,msg",
    [
        [np.arange(131072), "The frequencies are not evenly spaced"],
        [np.zeros(131072), "The frequencies are separated by more"],
    ],
)
def test_check_freq_spacing_flex_spw(sma_mir, chan_width, msg):
    """
    Verify that _check_freq_spacing works as expected with data sets (throws
    an error if windows are not contiguous, otherwise no error raised).
    """
    sma_mir.channel_width = chan_width
    with pytest.raises(ValueError, match=msg):
        sma_mir._check_freq_spacing()


def test_check_freq_spacing_single_chan_spw(sma_mir):
    sma_mir.flex_spw_id_array[-1] = 5
    spw_list = sma_mir.spw_array.tolist()
    spw_list.append(5)
    sma_mir.spw_array = np.asarray(spw_list)
    spacing_error, chanwidth_error = sma_mir._check_freq_spacing()
    assert not spacing_error
    assert not chanwidth_error


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
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
def test_flex_spw_add_concat(sma_mir, add_method, screen1, screen2):
    """
    Test add & fast concat with flexible spws using Mir file.

    Read in Mir files using flexible spectral windows, all of the same nchan
    """
    uv1 = sma_mir.select(freq_chans=np.where(screen1), inplace=False)
    uv2 = sma_mir.select(freq_chans=np.where(screen2), inplace=False)

    if np.any(np.logical_and(screen1, screen2)):
        flag_screen = screen2[screen1]
        uv1.data_array[:, flag_screen] = 0.0
        uv1.flag_array[:, flag_screen] = True

    uv_recomb = getattr(uv1, add_method[0])(uv2, **add_method[1])

    if np.any(~np.logical_or(screen1, screen2)):
        sma_mir.select(freq_chans=np.where(np.logical_or(screen1, screen2)))

    # Make sure the two datasets are in the same frequency order
    uv_recomb.reorder_freqs(
        spw_order=np.argsort(uv_recomb.spw_array), channel_order="freq"
    )
    sma_mir.reorder_freqs(spw_order=np.argsort(sma_mir.spw_array), channel_order="freq")

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
        [
            [],
            [["unproject_phase", {}], ["select", {"freq_chans": np.arange(32, 64)}]],
            "UVParameter phase_center_catalog does not match. Cannot combine objects.",
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
    with pytest.raises(ValueError, match=msg):
        uv1 += uv2


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat_freq(casa_uvfits):
    uv_full = casa_uvfits
    uv1 = uv_full.select(freq_chans=np.arange(0, 20), inplace=False)
    uv2 = uv_full.select(freq_chans=np.arange(20, 40), inplace=False)
    uv3 = uv_full.select(freq_chans=np.arange(40, 64), inplace=False)
    with check_warnings(
        UserWarning,
        match=[
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ],
        nwarnings=4,
    ):
        uv1.fast_concat([uv2, uv3], "freq", inplace=True)
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history
    assert uv1 == uv_full

    # Add frequencies - out of order
    uv1 = uv_full.select(freq_chans=np.arange(0, 20), inplace=False)
    with check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ]
        * 4,
    ):
        uv2.fast_concat([uv1, uv3], "freq", inplace=True)

    assert uv2.Nfreqs == uv_full.Nfreqs
    assert uv2._freq_array != uv_full._freq_array
    assert uv2._data_array != uv_full._data_array

    # reorder frequencies and test that they are equal
    index_array = np.argsort(uv2.freq_array)
    uv2.freq_array = uv2.freq_array[index_array]
    uv2.data_array = uv2.data_array[:, index_array, :]
    uv2.nsample_array = uv2.nsample_array[:, index_array, :]
    uv2.flag_array = uv2.flag_array[:, index_array, :]
    uv2.history = uv_full.history
    assert uv2._freq_array == uv_full._freq_array
    assert uv2 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat_pols(casa_uvfits):
    uv_full = casa_uvfits

    uv1 = uv_full.select(polarizations=uv_full.polarization_array[0:1], inplace=False)
    uv2 = uv_full.select(polarizations=uv_full.polarization_array[1:3], inplace=False)
    uv3 = uv_full.select(polarizations=uv_full.polarization_array[3:4], inplace=False)

    uv2.history += " testing the history. AIPS WTSCAL = 1.0"
    uv_new = uv1.fast_concat([uv2, uv3], "polarization")
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Unique part of next "
        "object history follows. testing the history.",
        uv_new.history,
    )
    uv_new.history = uv_full.history
    assert uv_new == uv_full

    uv_new = uv1.fast_concat([uv2, uv3], "polarization", verbose_history=True)
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Next object history "
        "follows." + uv2.history,
        uv_new.history,
    )

    # Add polarizations - out of order
    uv1 = uv_full.select(polarizations=uv_full.polarization_array[0:1], inplace=False)
    with check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions."
        ]
        * 4,
    ):
        uv2.fast_concat([uv1, uv3], "polarization", inplace=True)

    assert uv2._polarization_array != uv_full._polarization_array
    assert uv2._data_array != uv_full._data_array

    # reorder pols
    uv2.reorder_pols()
    uv2.history = uv_full.history
    assert uv2 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat_times(casa_uvfits):
    uv_full = casa_uvfits

    times = np.unique(uv_full.time_array)
    uv1 = uv_full.select(times=times[0 : len(times) // 3], inplace=False)
    uv2 = uv_full.select(
        times=times[len(times) // 3 : (len(times) // 3) * 2], inplace=False
    )
    uv3 = uv_full.select(times=times[(len(times) // 3) * 2 :], inplace=False)
    uv1 = uv1.fast_concat([uv2, uv3], "blt", inplace=False)
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific times using pyuvdata. "
        "Combined data along baseline-time "
        "axis using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("in_order", [True, False])
def test_fast_concat_bls(casa_uvfits, in_order):
    uv_full = casa_uvfits

    if in_order:
        # divide in half to keep in order
        ind1 = np.arange(uv_full.Nblts // 2)
        ind2 = np.arange(uv_full.Nblts // 2, uv_full.Nblts)
    else:
        # # add baselines such that Nants_data needs to change
        ant_list = list(range(15))  # Roughly half the antennas in the data
        # # All blts where ant_1 is in list
        ind1 = [i for i in range(uv_full.Nblts) if uv_full.ant_1_array[i] in ant_list]
        ind2 = [
            i for i in range(uv_full.Nblts) if uv_full.ant_1_array[i] not in ant_list
        ]
    uv1 = uv_full.select(blt_inds=ind1, inplace=False)
    uv2 = uv_full.select(blt_inds=ind2, inplace=False)
    uv1.fast_concat(uv2, "blt", inplace=True)
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific baseline-times using pyuvdata. "
        "Combined data along baseline-time axis "
        "using pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history
    assert uv1, uv_full

    # Add baselines out of order
    uv1 = uv_full.select(blt_inds=ind1, inplace=False)
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


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat_multi_axis_errors(casa_uvfits):
    uv_full = casa_uvfits

    # Add multiple axes
    times = np.unique(uv_full.time_array)
    uv1 = uv_full.select(
        times=times[0 : len(times) // 2],
        polarizations=uv_full.polarization_array[0:2],
        inplace=False,
    )
    uv2 = uv_full.select(
        times=times[len(times) // 2 :],
        polarizations=uv_full.polarization_array[2:4],
        inplace=False,
    )
    with pytest.raises(
        ValueError,
        match="UVParameter polarization_array does not match. Cannot combine objects.",
    ):
        uv1.fast_concat(uv2, "blt", inplace=True)

    # Another combo
    times = np.unique(uv_full.time_array)
    uv1 = uv_full.select(
        times=times[0 : len(times) // 2], freq_chans=np.arange(0, 32), inplace=False
    )
    uv2 = uv_full.select(
        times=times[len(times) // 2 :], freq_chans=np.arange(32, 64), inplace=False
    )
    with pytest.raises(
        ValueError,
        match="UVParameter freq_array does not match. Cannot combine objects.",
    ):
        uv1.fast_concat(uv2, "blt", inplace=True)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_fast_concat_auto_cross(hera_uvh5):
    # test add of autocorr-only and crosscorr-only objects
    uv_full = hera_uvh5
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
    with pytest.raises(ValueError, match="Axis must be one of"):
        uv1.fast_concat(uv2, "foo", inplace=True)

    cal = UVCal()
    with pytest.raises(
        ValueError, match="Only UVData \\(or subclass\\) objects can be added"
    ):
        uv1.fast_concat(cal, "freq", inplace=True)


@pytest.mark.parametrize("tuplify", [False, True])
def test_key2inds(casa_uvfits, tuplify):
    # Test function to interpret key as antpair, pol
    uv = casa_uvfits

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    key = (ant1, ant2, pol)
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)

    assert np.array_equal(bltind, ind1)
    assert ind2 is None
    assert indp[0] == slice(0, 1, 1)

    # Combo with pol as string
    key = (ant1, ant2, utils.polnum2str(pol))
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)
    assert indp[0] == slice(0, 1, 1)

    # Check conjugation
    key = (ant2, ant1, pol)
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)
    assert np.array_equal(bltind, ind2)
    assert ind1 is None
    assert indp[1] == slice(0, 1, 1)

    # Conjugation with pol as string
    key = (ant2, ant1, utils.polnum2str(pol))
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)
    assert np.array_equal(bltind, ind2)
    assert ind1 is None
    assert indp[1] == slice(0, 1, 1)
    assert indp[0] is None

    # Antpair only
    key = (ant1, ant2)
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)
    assert np.array_equal(bltind, ind1)
    assert ind2 is None
    assert indp[0] == slice(None)

    # Baseline number only
    key = uv.antnums_to_baseline(ant1, ant2)
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)
    assert np.array_equal(bltind, ind1)
    assert ind2 is None
    assert indp[0] == slice(None)

    # Pol number only
    key = pol
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)
    assert ind1 == slice(None)
    assert ind2 is None
    assert indp[0] == slice(0, 1, 1)

    # Pol string only
    key = "LL"
    if tuplify:
        key = (key,)
    ind1, ind2, indp = uv._key2inds(key)
    assert ind1 == slice(None)
    assert ind2 is None
    assert indp[0] == slice(1, 2, 1)

    # Test invalid keys
    with pytest.raises(KeyError, match="Polarization I not found in data."):
        uv._key2inds("I")  # pol str not in data
    with pytest.raises(KeyError, match="Polarization -8 not found in data."):
        uv._key2inds(-8)  # pol num not in data
    with pytest.raises(KeyError, match="Baseline 6 not found in data."):
        uv._key2inds(6)  # bl num not in data
    with pytest.raises(KeyError, match=r"Antenna pair \(1, 1\) not found in data"):
        uv._key2inds((1, 1))  # ant pair not in data
    with pytest.raises(KeyError, match=r"Antenna pair \(1, 1\) not found in data"):
        uv._key2inds((1, 1, "rr"))  # ant pair not in data
    with pytest.raises(KeyError, match="Polarization xx not found in data."):
        uv._key2inds((1, 2, "xx"))  # pol not in data

    # Test autos are handled correctly
    uv.ant_2_array[0] = uv.ant_1_array[0]
    ind1, ind2, indp = uv._key2inds((ant1, ant1, pol))
    assert ind1 == slice(0, 1, 1)
    assert ind2 is None


def test_key2inds_conj_all_pols(casa_uvfits):
    uv = casa_uvfits

    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds((ant2, ant1))

    # Pols in data are 'rr', 'll', 'rl', 'lr'
    # So conjugated order should be [0, 1, 3, 2]
    assert np.array_equal(bltind, ind2)
    assert ind1 is None
    assert indp[0] is None
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
    assert ind2 is None
    assert indp[0] == slice(None)
    assert indp[1] is None


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_bl_fringe(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    # Mix one instance of this baseline.
    uv.ant_1_array[0] = ant2
    uv.ant_2_array[0] = ant1
    uv.baseline_array[0] = utils.antnums_to_baseline(
        ant2, ant1, Nants_telescope=uv.telescope.Nants
    )
    bl = utils.antnums_to_baseline(ant1, ant2, Nants_telescope=uv.telescope.Nants)
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds(bl)

    assert np.array_equal(bltind, ind1)
    assert ind2 is None
    assert indp[0] == slice(None)
    assert indp[1] is None


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_missing_data(casa_uvfits):
    uv = casa_uvfits

    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]

    with pytest.raises(
        KeyError, match=r"Baseline \(8, 4\) not found for polarization array in data."
    ):
        uv._key2inds((ant2, ant1))


def test_key2inds_conj_all_pols_bls(casa_uvfits):
    uv = casa_uvfits

    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bl = utils.antnums_to_baseline(ant2, ant1, Nants_telescope=uv.telescope.Nants)
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    ind1, ind2, indp = uv._key2inds(bl)

    # Pols in data are 'rr', 'll', 'rl', 'lr'
    # So conjugated order should be [0, 1, 3, 2]
    assert np.array_equal(bltind, ind2)
    assert ind1 is None
    assert indp[0] is None
    assert np.array_equal([0, 1, 3, 2], indp[1])


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_key2inds_conj_all_pols_missing_data_bls(casa_uvfits):
    uv = casa_uvfits
    uv.select(polarizations=["rl"])
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    bl = utils.antnums_to_baseline(ant2, ant1, Nants_telescope=uv.telescope.Nants)

    with pytest.raises(
        KeyError, match="Baseline 81924 not found for polarization array in data."
    ):
        uv._key2inds(bl)


def test_smart_slicing_err(casa_uvfits):
    """
    Test that smart_slicing throws an error when using an invald squeeze
    """
    # Test invalid squeeze
    with pytest.raises(
        ValueError,
        match=(
            '"notasqueeze" is not a valid option for squeeze.Only "default", "none", '
            'or "full" are allowed.'
        ),
    ):
        casa_uvfits._smart_slicing(
            casa_uvfits.data_array,
            [0, 4, 5],
            None,
            ([0, 1], None),
            squeeze="notasqueeze",
        )


REG_IND = slice(0, 90, 10)
SINGLE_IND = [45]
REG_POL = slice(0, 2)
IRREG_IND = [0, 4, 5]
IRREG_POL = [0, 1, 3]


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "ind1, ind2, indp, squeeze, force_copy",
    [
        (REG_IND, None, REG_POL, "default", False),  # ind1 reg, ind2 empty, pol reg
        (REG_IND, None, REG_POL, "default", True),
        (REG_IND, None, IRREG_POL, "default", False),
        (IRREG_IND, None, REG_POL, "default", False),
        (IRREG_IND, None, IRREG_POL, "default", False),
        (None, REG_IND, REG_POL, "default", False),
        (None, REG_IND, IRREG_POL, "default", False),
        (None, IRREG_IND, REG_POL, "default", False),
        (slice(20), slice(20, 30), REG_POL, "default", False),
        (slice(20), slice(20, 30), IRREG_POL, "default", False),
        (SINGLE_IND, None, REG_POL, "default", False),
        (None, SINGLE_IND, REG_POL, "default", False),
        (SINGLE_IND, None, REG_POL, "full", False),
    ],
)
def test_smart_slicing(casa_uvfits, ind1, ind2, indp, squeeze, force_copy):
    # Test function to slice data
    uv = casa_uvfits

    if ind1 is None:
        polind = (None, indp)
        ind = ind2
        # Get actual arrays of integers for indexing
        bltinds = np.arange(uv.Nblts)[ind]
        polidx = np.arange(uv.Npols)[indp]

    elif ind2 is None:
        polind = (indp, None)
        ind = ind1
        # Get actual arrays of integers for indexing
        bltinds = np.arange(uv.Nblts)[ind]
        polidx = np.arange(uv.Npols)[indp]

    else:
        polind = (indp, indp)

    copy_made = force_copy or not all(
        (
            isinstance(ind1, slice) or ind1 is None,
            isinstance(ind2, slice) or ind2 is None,
            isinstance(indp, slice),
        )
    )

    d = uv._smart_slicing(
        uv.data_array, ind1, ind2, polind, force_copy=force_copy, squeeze=squeeze
    )
    if ind1 is None:
        dcheck = np.conj(uv.data_array[ind])
    elif ind2 is None:
        dcheck = uv.data_array[ind]
    else:
        dcheck = np.append(uv.data_array[ind1], np.conj(uv.data_array[ind2]), axis=0)

    dcheck = dcheck[..., indp]
    # if squeeze == "default" and ind1 != SINGLE_IND:
    dcheck = np.squeeze(dcheck)

    assert np.all(d == dcheck)  # don't care about shape.

    # Ensure a view/copy was returned
    if ind1 in (
        REG_IND,
        IRREG_IND,
    ):  # Note conjugation test ensures the result is a copy, not a view.
        if force_copy:
            assert d.flags.writeable
        else:
            assert not d.flags.writeable

        uv.data_array[bltinds[1], 0, polidx[0]] = 5.43
        if copy_made:
            assert d[1, 0, 0] != uv.data_array[bltinds[1], 0, polidx[0]]
        else:
            assert d[1, 0, 0] == uv.data_array[bltinds[1], 0, polidx[0]]


@pytest.mark.parametrize("kind", ["data", "flags", "nsamples", "times", "lsts"])
def test_get_data(casa_uvfits, kind):
    # Test get_data function for easy access to data
    uv = casa_uvfits

    fnc = getattr(uv, "get_" + kind)
    if kind.endswith("s"):
        kind = kind[:-1]
    thing = getattr(uv, kind + "_array")

    # Get an antpair/pol combo
    ant1 = uv.ant_1_array[0]
    ant2 = uv.ant_2_array[0]
    pol = uv.polarization_array[0]
    bltind = np.where((uv.ant_1_array == ant1) & (uv.ant_2_array == ant2))[0]
    if kind in ["time", "lst"]:
        dcheck = thing[bltind]
    else:
        dcheck = np.squeeze(thing[bltind, :, 0])

    d = fnc(ant1, ant2, pol)
    assert np.all(dcheck == d)

    d = fnc(ant1, ant2, utils.polnum2str(pol))
    assert np.all(dcheck == d)

    d = fnc((ant1, ant2, pol))
    assert np.all(dcheck == d)

    with pytest.raises(ValueError, match="no more than 3 key values can be passed"):
        fnc((ant1, ant2, pol), (ant1, ant2, pol))

    # Check conjugation
    d = fnc(ant2, ant1, pol)
    assert np.all(dcheck == np.conj(d))
    assert d.dtype == dcheck.dtype

    # Check cross pol conjugation
    d = fnc(ant2, ant1, uv.polarization_array[2])
    d1 = fnc(ant1, ant2, uv.polarization_array[3])
    assert np.all(d == np.conj(d1))

    # Antpair only
    if kind in ["time", "lst"]:
        dcheck = thing[bltind]
    else:
        dcheck = np.squeeze(thing[bltind, :, :])

    d = fnc(ant1, ant2)
    assert np.all(dcheck == d)

    # Pol number only
    if kind in ["time", "lst"]:
        dcheck = thing
    else:
        dcheck = np.squeeze(thing[..., 0])

    d = fnc(pol)
    assert np.all(dcheck == d)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind(hera_uvh5):
    hera_uvh5.set_rectangularity()

    # Test for baseline-time axis indexer
    uv = hera_uvh5

    # get indices
    inds = uv.antpair2ind(0, 1, ordered=False)
    assert inds == slice(3, None, 10)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_conj(hera_uvh5):
    # conjugate (and use key rather than arg expansion)
    uv = hera_uvh5
    uv.set_rectangularity()
    inds = uv.antpair2ind(0, 1, ordered=False)
    inds2 = uv.antpair2ind((1, 0), ordered=False)
    assert inds == inds2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_ordered(hera_uvh5):
    # test ordered
    uv = hera_uvh5
    inds = uv.antpair2ind(0, 1, ordered=False)

    # make sure conjugated baseline returns nothing
    inds2 = uv.antpair2ind(1, 0, ordered=True)
    assert inds2 is None

    # now use baseline actually in data
    inds2 = uv.antpair2ind(0, 1, ordered=True)
    assert inds == inds2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_autos(hera_uvh5):
    # test autos w/ and w/o ordered
    uv = hera_uvh5

    inds = uv.antpair2ind(0, 0, ordered=True)
    inds2 = uv.antpair2ind(0, 0, ordered=False)
    assert inds == inds2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpair2ind_exceptions(hera_uvh5):
    # test exceptions
    uv = hera_uvh5

    with pytest.raises(ValueError, match="antpair2ind must be fed an antpair tuple"):
        uv.antpair2ind(1)
    with pytest.raises(ValueError, match="antpair2ind must be fed an antpair tuple"):
        uv.antpair2ind("bar", "foo")
    with pytest.raises(ValueError, match="ordered must be a boolean"):
        uv.antpair2ind(0, 1, ordered="foo")


def test_antpairpol_iter(casa_uvfits):
    # Test generator
    uv = casa_uvfits
    pol_dict = {utils.polnum2str(uv.polarization_array[i]): i for i in range(uv.Npols)}
    keys = []
    pols = set()
    bls = set()
    for key, d in uv.antpairpol_iter():
        keys += key
        bl = uv.antnums_to_baseline(key[0], key[1])
        blind = np.where(uv.baseline_array == bl)[0]
        bls.add(bl)
        pols.add(key[2])
        dcheck = np.squeeze(uv.data_array[blind, :, pol_dict[key[2]]])
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
def test_get_enu_antpos(hera_uvh5):
    uvd = hera_uvh5
    # no center, no pick data ants
    antpos = uvd.telescope.get_enu_antpos()
    assert uvd.telescope.Nants == 9
    assert np.isclose(antpos[-1, 0], -105.13193283147963, rtol=0, atol=1e-3)
    assert uvd.telescope.antenna_numbers[0] == 0

    # pick data ants
    antpos2, ants = uvd.get_enu_data_ants()
    assert uvd.Nants_data == 4
    assert ants[0] == 0
    assert np.isclose(antpos2[-1, 0], -112.3875190893361, rtol=0, atol=1e-3)

    data_ant_inds = np.isin(uvd.telescope.antenna_numbers, ants)
    assert np.all(antpos[data_ant_inds] == antpos2)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_telescope_loc_xyz_check(hera_uvh5, tmp_path):
    # test that improper telescope locations can still be read
    uv = hera_uvh5
    uv.telescope.location = EarthLocation.from_geocentric(
        *utils.XYZ_from_LatLonAlt(*uv.telescope._location.xyz()), unit="m"
    )
    # fix LST values
    uv.set_lsts_from_time_array()
    fname = str(tmp_path / "test.uvh5")
    uv.write_uvh5(fname, run_check=False, check_extra=False, clobber=True)

    # try to read file without checks, should be no warnings
    uv.read(fname, run_check=False)

    # try to read without checks: should be warnings
    with check_warnings(
        UserWarning,
        match=["The uvw_array does not match the expected"]
        + [
            "itrs position vector magnitudes must be on the order "
            "of the radius of Earth -- they appear to lie well below this."
        ]
        * 4,
    ):
        uv.read(fname)


def test_get_pols(casa_uvfits):
    # Test function to get unique polarizations in string format
    uv = casa_uvfits
    pols = uv.get_pols()
    pols_data = ["rr", "ll", "lr", "rl"]
    assert sorted(pols) == sorted(pols_data)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_get_pols_x_orientation(hera_uvh5):
    uv_in = hera_uvh5

    uv_in.telescope.set_feeds_from_x_orientation(
        "east", polarization_array=uv_in.polarization_array
    )

    pols = uv_in.get_pols()
    pols_data = ["ee", "nn"]
    assert pols == pols_data

    uv_in.telescope.set_feeds_from_x_orientation(
        "north", polarization_array=uv_in.polarization_array
    )
    pols = uv_in.get_pols()
    pols_data = ["nn", "ee"]
    assert pols == pols_data


def test_get_feedpols(casa_uvfits):
    # Test function to get unique antenna feed polarizations in data. String format.
    uv = casa_uvfits
    pols = uv.get_feedpols()
    pols_data = ["r", "l"]
    assert sorted(pols) == sorted(pols_data)

    # Test break when pseudo-Stokes visibilities are present
    uv.polarization_array[0] = 1  # pseudo-Stokes I
    with pytest.raises(
        ValueError,
        match="Pseudo-Stokes visibilities cannot be interpreted as feed polarizations",
    ):
        uv.get_feedpols()


def test_parse_ants(casa_uvfits, hera_uvh5):
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

    # Unparsable string
    ant_str = "none"
    with pytest.raises(ValueError, match="Unparsable argument none"):
        uv.parse_ants(ant_str)

    # Single antenna number
    ant_str = "1"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    # fmt: off
    ant_pairs_expected = [(1, 2), (1, 3), (1, 4), (1, 7), (1, 8), (1, 9),
                          (1, 12), (1, 15), (1, 19), (1, 20), (1, 21),
                          (1, 22), (1, 23), (1, 24), (1, 25), (1, 27),
                          (1, 28)]
    # fmt: on
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Single antenna number not in the data
    ant_str = "10"
    with check_warnings(
        UserWarning,
        "Warning: Antenna number 10 passed, but not present in the ant_1_array "
        "or ant_2_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)

    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Single antenna number with polarization, both not in the data
    ant_str = "10x"
    with check_warnings(
        UserWarning,
        [
            (
                "Warning: Antenna number 10 passed, but not present in the ant_1_array"
                " or ant_2_array"
            ),
            "Warning: Polarization XX,XY is not present in the polarization_array",
        ],
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert isinstance(ant_pairs_nums, type(None))
    assert isinstance(polarizations, type(None))

    # Multiple antenna numbers as list
    ant_str = "22,27"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    # fmt: off
    ant_pairs_expected = [(1, 22), (2, 22), (3, 22), (4, 22), (7, 22), (8, 22),
                          (9, 22), (12, 22), (15, 22), (19, 22), (20, 22),
                          (21, 22), (22, 23), (22, 24), (22, 25), (22, 27),
                          (22, 28), (1, 27), (2, 27), (3, 27), (4, 27),
                          (7, 27), (8, 27), (9, 27), (12, 27), (15, 27),
                          (19, 27), (20, 27), (21, 27), (23, 27),
                          (24, 27), (25, 27), (27, 28)]
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
    with check_warnings(
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
    with check_warnings(
        UserWarning,
        "Warning: Polarization XX,YX is not present in the polarization_array",
    ):
        ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (2, 3)]
    pols_expected = [-2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Multiple baselines as list
    ant_str = "1_2,1_3,1_12"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 12)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Multiples baselines with polarizations as list
    ant_str = "1r_2l,1l_3l,1r_12r"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 2), (1, 3), (1, 12)]
    pols_expected = [-1, -2, -3]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Specific baselines with parenthesis
    ant_str = "(1,3)_12"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 12), (3, 12)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Specific baselines with parenthesis
    ant_str = "1_(3,12)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 12)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations
    ant_str = "(1l,2r)_(3l,7r)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 7), (2, 3), (2, 7)]
    pols_expected = [-1, -2, -3, -4]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers with - for avoidance
    ant_str = "1_(-3,12)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 12)]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific antenna number
    ant_str = "1,-3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [
        (1, 2),
        (1, 4),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 12),
        (1, 15),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 25),
        (1, 27),
        (1, 28),
    ]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Remove specific baseline (same expected antenna pairs as above example)
    ant_str = "1,-1_3"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Antenna numbers with polarizations and - for avoidance
    ant_str = "1l_(-3r,12l)"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 12)]
    pols_expected = [-2]
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert Counter(polarizations) == Counter(pols_expected)

    # Antenna numbers and pseudo-Stokes parameters
    ant_str = "(1l,2r)_(3l,7r),pI,pq"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = [(1, 3), (1, 7), (2, 3), (2, 7)]
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
    with check_warnings(
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

    # Test ant_str='auto' on file with auto correlations, want single pol
    uv = hera_uvh5
    uv.select(polarizations="xx")
    uv.conjugate_bls(convention="ant1<ant2")

    ant_str = "auto"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_nums = [0, 1, 2, 11]
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
    ant_str = "all,-1x_11x"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = ant_pairs_autos + ant_pairs_cross
    ant_pairs_expected.remove((1, 11))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))

    # Test appending all to beginning of strings that start with -
    ant_str = "-11"
    ant_pairs_nums, polarizations = uv.parse_ants(ant_str)
    ant_pairs_expected = ant_pairs_autos + ant_pairs_cross
    for ant_i in ant_nums:
        ant_pairs_expected.remove((ant_i, 11))
    assert Counter(ant_pairs_nums) == Counter(ant_pairs_expected)
    assert isinstance(polarizations, type(None))


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_with_ant_str(casa_uvfits, hera_uvh5):
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
    ant_str = "1"
    ant_pairs = [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 12),
        (1, 15),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 25),
        (1, 27),
        (1, 28),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Single antenna number not present in data
    ant_str = "10"
    with check_warnings(
        UserWarning,
        [
            (
                "Warning: Antenna number 10 passed, but not present in the "
                "ant_1_array or ant_2_array"
            ),
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv.select(ant_str=ant_str, inplace=inplace)

    # Multiple antenna numbers as list
    ant_str = "23,27"
    ant_pairs = [
        (1, 23),
        (1, 27),
        (2, 23),
        (2, 27),
        (3, 23),
        (3, 27),
        (4, 23),
        (4, 27),
        (7, 23),
        (7, 27),
        (8, 23),
        (8, 27),
        (9, 23),
        (9, 27),
        (12, 23),
        (12, 27),
        (15, 23),
        (15, 27),
        (19, 23),
        (19, 27),
        (20, 23),
        (20, 27),
        (21, 23),
        (21, 27),
        (22, 23),
        (22, 27),
        (23, 24),
        (23, 25),
        (23, 27),
        (23, 28),
        (24, 27),
        (25, 27),
        (27, 28),
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
    with check_warnings(
        UserWarning,
        [
            "Warning: Polarization XX,XY is not present in the polarization_array",
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
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
    with check_warnings(
        UserWarning,
        [
            "Warning: Polarization XX,YX is not present in the polarization_array",
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
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
    with check_warnings(
        UserWarning,
        [
            (
                "Warning: Antenna number 10 passed, but not present in the "
                "ant_1_array or ant_2_array"
            ),
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv2 = uv.select(ant_str=ant_str, inplace=inplace)

    ant_pairs = [(1, 2), (1, 3)]
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Multiples baselines with polarizations as list
    ant_str = "1r_2l,1l_3l,1r_12r"
    ant_pairs = [(1, 2), (1, 3), (1, 12)]
    pols = ["rr", "ll", "rl"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Specific baselines with parenthesis
    ant_str = "(1,3)_12"
    ant_pairs = [(1, 12), (3, 12)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Specific baselines with parenthesis
    ant_str = "1_(3,12)"
    ant_pairs = [(1, 3), (1, 12)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations
    ant_str = "(1l,2r)_(3l,7r)"
    ant_pairs = [(1, 3), (1, 7), (2, 3), (2, 7)]
    pols = ["rr", "ll", "rl", "lr"]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(pols)

    # Antenna numbers with - for avoidance
    ant_str = "1_(-3,12)"
    ant_pairs = [(1, 12)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    ant_str = "(-1,3)_12"
    ant_pairs = [(3, 12)]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific antenna number
    ant_str = "1,-3"
    ant_pairs = [
        (1, 2),
        (1, 4),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 12),
        (1, 15),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 25),
        (1, 27),
        (1, 28),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Remove specific baseline
    ant_str = "1,-1_3"
    ant_pairs = [
        (1, 2),
        (1, 4),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 12),
        (1, 15),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (1, 23),
        (1, 24),
        (1, 25),
        (1, 27),
        (1, 28),
    ]
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Antenna numbers with polarizations and - for avoidance
    ant_str = "1l_(-3r,12l)"
    ant_pairs = [(1, 12)]
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

    # Test ant_str='auto' on file with auto correlations, want single pol
    uv = hera_uvh5
    uv.select(polarizations="xx")
    uv.conjugate_bls(convention="ant1<ant2")

    ant_str = "auto"
    ant_nums = [0, 1, 2, 11]
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
    ant_str = "all,-1x_11x"
    ant_pairs = ant_pairs_autos + ant_pairs_cross
    ant_pairs.remove((1, 11))
    uv2 = uv.select(ant_str=ant_str, inplace=inplace)
    assert Counter(uv2.get_antpairs()) == Counter(ant_pairs)
    assert Counter(uv2.get_pols()) == Counter(uv.get_pols())

    # Test appending all to beginning of strings that start with -
    ant_str = "-11"
    ant_pairs = ant_pairs_autos + ant_pairs_cross
    for ant_i in ant_nums:
        ant_pairs.remove((ant_i, 11))
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
            (
                "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
                "polarizations."
            ),
        ),
        (
            {"ant_str": "", "antenna_names": []},
            (
                "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
                "polarizations."
            ),
        ),
        (
            {"ant_str": "", "bls": []},
            (
                "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
                "polarizations."
            ),
        ),
        (
            {"ant_str": "", "polarizations": []},
            (
                "Cannot provide ant_str with antenna_nums, antenna_names, bls, or "
                "polarizations."
            ),
        ),
        ({"ant_str": "auto"}, "There is no data matching ant_str=auto in this object."),
        (
            {"ant_str": "pI,pq,pU,pv"},
            "Polarization 4 is not present in the polarization_array",
        ),
        ({"ant_str": "none"}, "Unparsable argument none"),
        (
            {"ant_str": "4l_8l", "invert": True},
            "Cannot set invert=True if using ant_str with polarizations.",
        ),
    ],
)
def test_select_with_ant_str_errors(casa_uvfits, kwargs, message):
    uv = casa_uvfits

    with pytest.raises(ValueError, match=message):
        uv.select(**kwargs, strict=True)


@pytest.mark.parametrize("grid_alg", [True, False])
def test_get_antenna_redundancies(pyuvsim_redundant, grid_alg):
    uv0 = pyuvsim_redundant

    old_bl_array = np.copy(uv0.baseline_array)
    red_gps, centers, lengths = uv0.get_redundancies(
        use_antpos=True, include_autos=False, conjugate_bls=True, use_grid_alg=grid_alg
    )
    # new and old baseline Numbers are not the same (different conjugation)
    assert not np.allclose(uv0.baseline_array, old_bl_array)

    # assert all baselines are in the data (because it's conjugated to match)
    for gp in red_gps:
        for bl in gp:
            assert bl in uv0.baseline_array

    # conjugate data differently
    uv0.conjugate_bls(convention="ant1<ant2")
    new_red_gps, new_centers, new_lengths, conjs = uv0.get_redundancies(
        use_antpos=True,
        include_autos=False,
        include_conjugates=True,
        use_grid_alg=grid_alg,
    )

    assert conjs is None

    apos = uv0.telescope.get_enu_antpos()
    new_red_gps, new_centers, new_lengths = utils.redundancy.get_antenna_redundancies(
        uv0.telescope.antenna_numbers, apos, include_autos=False, use_grid_alg=grid_alg
    )

    # all redundancy info is the same
    assert red_gps == new_red_gps
    np.testing.assert_allclose(
        centers, new_centers, rtol=uv0._uvw_array.tols[0], atol=uv0._uvw_array.tols[1]
    )
    np.testing.assert_allclose(
        lengths, new_lengths, rtol=uv0._uvw_array.tols[0], atol=uv0._uvw_array.tols[1]
    )


@pytest.mark.parametrize("grid_alg", [True, False])
@pytest.mark.parametrize("method", ("select", "average"))
@pytest.mark.parametrize("reconjugate", (True, False))
@pytest.mark.parametrize("flagging_level", ("none", "some", "all"))
def test_redundancy_contract_expand(
    method, reconjugate, flagging_level, pyuvsim_redundant, grid_alg
):
    # Test that a UVData object can be reduced to one baseline from each redundant group
    # and restored to its original form.

    uv0 = pyuvsim_redundant

    # Fails at lower precision because some baselines fall into multiple
    # redundant groups
    tol = 0.02

    if reconjugate:
        # the test file has groups that are either all not conjugated or all conjugated.
        # need to conjugate some so we have mixed groups to properly test the average
        # method.
        (orig_red_gps, _, _, _) = uv0.get_redundancies(
            tol=tol, include_conjugates=True, use_grid_alg=grid_alg
        )
        blt_inds_to_conj = []
        for gp in orig_red_gps:
            if len(gp) > 1:
                blt_inds_to_conj.extend(
                    list(np.nonzero(uv0.baseline_array == gp[0])[0])
                )
        uv0.conjugate_bls(np.array(blt_inds_to_conj))

    # Assign identical data to each redundant group, set up flagging.
    # This must be done after reconjugation because reconjugation can alter the index
    # baseline
    red_gps, _, _, conjugates = uv0.get_redundancies(
        tol=tol, include_conjugates=True, use_grid_alg=grid_alg
    )
    index_bls = []
    for gp_ind, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            # Make this data complex so that we can track phase issues
            uv0.data_array[inds] += complex(1 + gp_ind, 1 - gp_ind)
        index_bls.append(gp[0])

    # Data in the conjugated list, make sure that we manually conjugate here
    conj_mask = np.isin(uv0.baseline_array, conjugates)
    uv0.data_array[conj_mask] = np.conj(uv0.data_array[conj_mask])

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
        uv3.conjugate_bls(np.array(blt_inds_to_conj))

    uv2 = uv0.compress_by_redundancy(
        method=method, tol=tol, inplace=False, use_grid_alg=grid_alg
    )
    uv2.check()
    if grid_alg:
        with check_warnings(
            UserWarning,
            match="The use_grid_alg parameter is not set. Defaulting to True to "
            "use the new gridding based algorithm (developed by the HERA team) "
            "rather than the older clustering based algorithm. This is change "
            "to the default, to use the clustering algorithm set "
            "use_grid_alg=False.",
        ):
            uv4 = uv0.compress_by_redundancy(
                method=method, tol=tol, inplace=False, use_grid_alg=None
            )
        assert uv4 == uv2

    if method == "average":
        gp_bl_use = []
        nbls_group = []
        for gp in red_gps:
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
    uv3.compress_by_redundancy(method=method, tol=tol, use_grid_alg=grid_alg)
    if reconjugate:
        assert len(orig_red_gps) == len(red_gps)
        match_ind_list = []
        for gp in red_gps:
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
            uv3.conjugate_bls(blts_to_conj)

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
                        uv2_blts = np.nonzero(uv2.baseline_array == bl)[0]
                        np.testing.assert_allclose(
                            uv2.data_array[uv2_blts],
                            uv0.data_array[orig_blts],
                            rtol=uv0._data_array.tols[0],
                            atol=uv0._data_array.tols[1],
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
    with check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv2.inflate_by_redundancy(tol=tol, use_grid_alg=grid_alg)

    # Confirm that we get the same result looping inflate -> compress -> inflate.
    uv3 = uv2.compress_by_redundancy(
        method=method, tol=tol, inplace=False, use_grid_alg=grid_alg
    )
    with check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv3.inflate_by_redundancy(tol=tol, use_grid_alg=grid_alg)

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


@pytest.mark.parametrize("grid_alg", [True, False])
@pytest.mark.parametrize("method", ("select", "average"))
@pytest.mark.parametrize("flagging_level", ("none", "some", "all"))
def test_redundancy_contract_expand_variable_data(
    method, flagging_level, grid_alg, pyuvsim_redundant
):
    # Test that a UVData object can be reduced to one baseline from each redundant group
    # and restored to its original form.

    uv0 = pyuvsim_redundant

    # Fails at lower precision because some baselines fall into multiple
    # redundant groups
    tol = 0.02
    # Assign identical data to each redundant group in comparison object
    # Assign data to the index baseline and zeros elsewhere in the one to compress
    red_gps, _, _ = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True, use_grid_alg=grid_alg
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
        uv0.flag_array[:, :, :] = True
        for bl in index_bls:
            bl_locs = np.where(uv0.baseline_array == bl)
            uv0.flag_array[bl_locs, :, :] = False
    elif flagging_level == "all":
        uv0.flag_array[:] = True
        uv0.check()
        assert np.all(uv0.flag_array)

    uv2 = uv0.compress_by_redundancy(
        method=method, tol=tol, inplace=False, use_grid_alg=grid_alg
    )

    # inflate to get back to the original size
    with check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv2.inflate_by_redundancy(tol=tol, use_grid_alg=grid_alg)

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
@pytest.mark.parametrize("grid_alg", [True, False])
@pytest.mark.parametrize("method", ("select", "average"))
def test_redundancy_contract_expand_nblts_not_nbls_times_ntimes(
    method, casa_uvfits, grid_alg
):
    uv0 = casa_uvfits

    # check that Nblts != Nbls * Ntimes
    assert uv0.Nblts != uv0.Nbls * uv0.Ntimes

    tol = 1.0

    # Assign identical data to each redundant group:
    red_gps, _, _ = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True, use_grid_alg=grid_alg
    )
    for i, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds, ...] *= 0
            uv0.data_array[inds, ...] += complex(i)

    msg = [
        "The uvw_array does not match the expected values given the antenna positions."
    ]
    if method == "average":
        msg *= 2
        msg += [
            "Index baseline in the redundant group does not have all the "
            "times, compressed object will be missing those times."
        ] * 4
    with check_warnings(UserWarning, match=msg):
        uv2 = uv0.compress_by_redundancy(
            method=method, tol=tol, inplace=False, use_grid_alg=grid_alg
        )

    # check inflating gets back to the original
    with check_warnings(
        UserWarning,
        [
            "Missing some redundant groups. Filling in available data.",
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv2.inflate_by_redundancy(tol=tol, use_grid_alg=grid_alg)

    uv2.history = uv0.history
    # Inflation changes the baseline ordering into the order of the redundant groups.
    # reorder bls for comparison
    uv0.reorder_blts()
    uv2.reorder_blts()
    uv2._uvw_array.tols = [0, tol]

    blt_inds = []
    missing_inds = []
    for bl, t in zip(uv0.baseline_array, uv0.time_array, strict=True):
        if (bl, t) in zip(uv2.baseline_array, uv2.time_array, strict=True):
            # get inds for inflated blts that exist on the original object
            this_ind = np.where((uv2.baseline_array == bl) & (uv2.time_array == t))[0]
            blt_inds.append(this_ind[0])
        else:
            # get inds on original object that are missing in uv2
            # (because of the compress_by_redundancy step)
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


@pytest.mark.parametrize("grid_alg", [True, False])
def test_compress_redundancy_variable_inttime(grid_alg):
    uv0 = UVData()
    uv0.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    tol = 0.05
    ntimes_in = uv0.Ntimes

    # Assign identical data to each redundant group:
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True, use_grid_alg=grid_alg
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

    with check_warnings(
        UserWarning,
        "Integrations times are not identical in a redundant "
        "group. Averaging anyway but this may cause unexpected "
        "behavior.",
        nwarnings=56,
    ) as warn_record:
        uv0.compress_by_redundancy(method="average", tol=tol, use_grid_alg=grid_alg)
    assert len(warn_record) == np.sum(nbls_group > 1) * ntimes_in

    uv1.compress_by_redundancy(method="average", tol=tol, use_grid_alg=grid_alg)

    assert uv0 == uv1


@pytest.mark.parametrize("grid_alg", [True, False])
@pytest.mark.parametrize("method", ("select", "average"))
def test_compress_redundancy_metadata_only(method, grid_alg, pyuvsim_redundant):
    uv0 = pyuvsim_redundant

    tol = 0.05

    # Assign identical data to each redundant group
    red_gps, centers, lengths = uv0.get_redundancies(
        tol=tol, use_antpos=True, conjugate_bls=True, use_grid_alg=grid_alg
    )
    for i, gp in enumerate(red_gps):
        for bl in gp:
            inds = np.where(bl == uv0.baseline_array)
            uv0.data_array[inds] *= 0
            uv0.data_array[inds] += complex(i)

    uv2 = uv0.copy(metadata_only=True)
    uv2.compress_by_redundancy(
        method=method, tol=tol, inplace=True, use_grid_alg=grid_alg
    )

    uv0.compress_by_redundancy(method=method, tol=tol, use_grid_alg=grid_alg)
    uv0.data_array = None
    uv0.flag_array = None
    uv0.nsample_array = None
    assert uv0 == uv2


def test_compress_redundancy_wrong_method(pyuvsim_redundant):
    uv0 = pyuvsim_redundant

    tol = 0.05
    with pytest.raises(ValueError, match="method must be one of"):
        uv0.compress_by_redundancy(method="foo", tol=tol, inplace=True)


@pytest.mark.parametrize("grid_alg", [True, False])
@pytest.mark.parametrize("method", ("select", "average"))
def test_redundancy_missing_groups(method, grid_alg, pyuvsim_redundant, tmp_path):
    # Check that if I try to inflate a compressed UVData that is missing
    # redundant groups, it will raise the right warnings and fill only what
    # data are available.

    uv0 = pyuvsim_redundant

    tol = 0.02
    num_select = 19

    uv0.compress_by_redundancy(method=method, tol=tol, use_grid_alg=grid_alg)
    fname = str(tmp_path / "temp_hera19_missingreds.uvfits")

    bls = np.unique(uv0.baseline_array)[:num_select]  # First twenty baseline groups
    uv0.select(bls=[uv0.baseline_to_antnums(bl) for bl in bls])
    uv0.write_uvfits(fname)
    uv1 = UVData()
    uv1.read_uvfits(fname)

    # The UVFITS writer fills in the rdate parameter automatically if not present on
    # the main object, so check it and set the two equal to one another.
    assert uv1.rdate == "2017-12-22"
    uv0.rdate = uv1.rdate

    # check that filenames are what we expect
    assert uv0.filename == ["fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits"]
    assert uv1.filename == ["temp_hera19_missingreds.uvfits"]

    # update phase center catalog to make objects match
    uv1._consolidate_phase_center_catalogs(
        reference_catalog=uv0.phase_center_catalog, ignore_name=True
    )
    assert uv0 == uv1  # Check that writing compressed files causes no issues.

    with check_warnings(
        UserWarning, match="Missing some redundant groups. Filling in available data."
    ):
        uv1.inflate_by_redundancy(tol=tol, use_grid_alg=grid_alg)

    uv2 = uv1.compress_by_redundancy(
        method=method, tol=tol, inplace=False, use_grid_alg=grid_alg
    )

    assert np.unique(uv2.baseline_array).size == num_select


@pytest.mark.parametrize("grid_alg", [True, False])
def test_quick_redundant_vs_redundant_test_array(grid_alg, pyuvsim_redundant):
    """Verify the quick redundancy calc returns the same groups as a known array."""
    uv = pyuvsim_redundant

    uv.select(times=uv.time_array[0])
    uv.unproject_phase()
    uv.conjugate_bls("u>0", use_enu=True)
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

    redundant_groups, _, _, _ = uv.get_redundancies(
        tol=tol, include_conjugates=True, use_grid_alg=grid_alg
    )
    redundant_groups.sort()
    assert groups == redundant_groups


@pytest.mark.parametrize("grid_alg", [True, False])
def test_redundancy_finder_when_nblts_not_nbls_times_ntimes(grid_alg, casa_uvfits):
    """Test the redundancy finder functions when Nblts != Nbls * Ntimes."""
    tol = 1  # meter
    uv = casa_uvfits
    uv.conjugate_bls("u>0", use_enu=True)
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

    redundant_groups, _, _, _ = uv.get_redundancies(
        tol=tol, include_conjugates=True, use_grid_alg=grid_alg
    )
    redundant_groups.sort()

    assert groups == redundant_groups


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_overlapping_data_add(casa_uvfits, tmp_path):
    # read in test data
    uv = casa_uvfits

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
    assert utils.history._check_histories(uvfull.history, uv.history + extra_history)
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
    assert utils.history._check_histories(uvfull.history, uv.history + extra_history2)
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
    with pytest.raises(
        ValueError,
        match=(
            "To combine these data, please run the add operation again, but with "
            "the object whose data is to be overwritten as the first object"
        ),
    ):
        uv4.__iadd__(uvfull)
    with pytest.raises(
        ValueError, match="These objects have overlapping data and cannot be combined."
    ):
        uv4.__add__(uv4)

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
    uv.reorder_blts()
    assert utils.history._check_histories(uvfull.history, uv.history + extra_history2)
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

    uvfull._update_phase_center_id(1, new_id=0)
    uvfull.phase_center_catalog[0]["info_source"] = uv.phase_center_catalog[0][
        "info_source"
    ]

    assert uvfull == uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_lsts_from_time_with_only_unique(hera_uvh5):
    """
    Test `set_lsts_from_time_array` with only unique values is identical to full array.
    """
    uv = hera_uvh5
    # calculate the lsts for all elements in time array
    full_lsts = utils.get_lst_for_time(
        uv.time_array, telescope_loc=uv.telescope.location
    )
    # use `set_lst_from_time_array` to set the uv.lst_array using only unique values
    uv.set_lsts_from_time_array()
    assert np.array_equal(full_lsts, uv.lst_array)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_lsts_from_time_with_only_unique_background(hera_uvh5):
    """
    Test `set_lsts_from_time_array` with only unique values is identical to full array.
    """
    uv = hera_uvh5
    # calculate the lsts for all elements in time array
    full_lsts = utils.get_lst_for_time(
        uv.time_array, telescope_loc=uv.telescope.location
    )
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
def test_upsample_in_time(hera_uvh5):
    """Test the upsample_in_time method"""
    uv_object = hera_uvh5

    init_phase_dict = uv_object.phase_center_catalog[0]
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # test that setting the app coords works with extra unused phase centers
    assert 0 not in uv_object.phase_center_catalog
    uv_object.phase_center_catalog[0] = init_phase_dict

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    np.testing.assert_allclose(
        uv_object.integration_time,
        max_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the same
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[0, 0, 0],
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_with_flags(hera_uvh5):
    """Test the upsample_in_time method with flags"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) / 2.0

    # add flags and upsample again
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01.start or 0, 0, 0] = True
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    # data and nsamples should be changed as normal, but flagged
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[0, 0, 0],
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    out_flags = uv_object.get_flags(0, 1)
    assert np.all(out_flags[:2, 0, 0])
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_noninteger_resampling(hera_uvh5):
    """Test the upsample_in_time method with a non-integer resampling factor"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline")

    # save some values for later
    init_data_size = uv_object.data_array.size
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # change the target integration time
    max_integration_time = np.amin(uv_object.integration_time) * 0.75
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")

    np.testing.assert_allclose(
        uv_object.integration_time,
        max_integration_time * 0.5 / 0.75,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be different by a factor of 2
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[0, 0, 0],
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_errors(hera_uvh5):
    """Test errors and warnings raised by upsample_in_time"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # test using a too-small integration time
    max_integration_time = 1e-3 * np.amin(uv_object.integration_time)
    with pytest.raises(
        ValueError, match="Decreasing the integration time by more than"
    ):
        uv_object.upsample_in_time(max_integration_time)

    # catch a warning for doing no work
    uv_object2 = uv_object.copy()
    max_integration_time = 2 * np.amax(uv_object.integration_time)
    with check_warnings(
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
    uv_object.reorder_blts("baseline")

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

    np.testing.assert_allclose(
        uv_object.integration_time,
        max_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the half the input
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[0, 0, 0] / 2,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_summing_correlator_mode_with_flags(hera_uvh5):
    """Test the upsample_in_time method with summing correlator mode and flags"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # add flags and upsample again
    inds01 = uv_object.antpair2ind(0, 1)
    uv_object.flag_array[inds01, 0, 0] = True
    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(
        max_integration_time, blt_order="baseline", summing_correlator_mode=True
    )

    # data and nsamples should be changed as normal, but flagged
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[0, 0, 0] / 2,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    out_flags = uv_object.get_flags(0, 1)
    assert np.all(out_flags[:2, 0, 0])
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline")

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

    np.testing.assert_allclose(
        uv_object.integration_time,
        max_integration_time * 0.5 / 0.75,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be half the input
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[0, 0, 0] / 2,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline")

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

    np.testing.assert_allclose(
        uv_object.integration_time,
        max_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    # output data should be the same
    out_wf_01 = uv_object.get_data(0, 1)
    out_wf_02 = uv_object.get_data(0, 2)
    assert np.all(init_wf_01 == out_wf_01)
    assert np.isclose(
        init_wf_02[0, 0, 0],
        out_wf_02[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert init_wf_02.size * 2 == out_wf_02.size

    # this should be true because there are no flags
    out_ns_01 = uv_object.get_nsamples(0, 1)
    out_ns_02 = uv_object.get_nsamples(0, 2)
    np.testing.assert_allclose(
        out_ns_01,
        init_ns_01,
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )
    assert np.isclose(
        init_ns_02[0, 0, 0],
        out_ns_02[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_upsample_in_time_drift(hera_uvh5):
    """Test the upsample_in_time method on drift mode data"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline")

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

    np.testing.assert_allclose(
        uv_object.integration_time,
        max_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be the same
    out_wf = uv_object.get_data(0, 1)
    # we need a "large" tolerance given the "large" data
    new_tol = 1e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0], atol=new_tol, rtol=0)

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("driftscan", [True, False])
@pytest.mark.parametrize("partial_phase", [True, False])
def test_upsample_in_time_drift_no_phasing(hera_uvh5, driftscan, partial_phase):
    """Test the upsample_in_time method on drift mode data without phasing"""
    uv_object = hera_uvh5

    if driftscan:
        uv_object.phase(
            lon=0,
            lat=np.pi / 2,
            phase_frame="altaz",
            cat_type="driftscan",
            cat_name="foo",
        )

    if partial_phase:
        mask = np.full(uv_object.Nblts, False)
        mask[: uv_object.Nblts // 2] = True
        uv_object.phase(
            ra=0, dec=0, phase_frame="icrs", select_mask=mask, cat_name="bar"
        )

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline")

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

    np.testing.assert_allclose(
        uv_object.integration_time,
        max_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    # we should double the size of the data arrays
    assert uv_object.data_array.size == 2 * init_data_size
    # output data should be similar, but somewhat different because of the phasing
    out_wf = uv_object.get_data(0, 1)
    # we need a "large" tolerance given the "large" data
    new_tol = 1e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(init_wf[0, 0, 0], out_wf[0, 0, 0], atol=new_tol, rtol=0)

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        init_ns[0, 0, 0],
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time(hera_uvh5):
    """Test the downsample_in_time method"""
    uv_object = hera_uvh5

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")
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
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[1],
    )
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")
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
    uv_object.flag_array[inds01, 0, 0][0] = True
    uv_object2 = uv_object.copy()

    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[1, 0, 0],
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
def test_downsample_in_time_totally_flagged(hera_uvh5):
    """Test the downsample_in_time method with totally flagged integrations"""
    uv_object = hera_uvh5

    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")
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
    uv_object.flag_array[inds01, 0, 0][:2] = True
    uv_object2 = uv_object.copy()

    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")
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
            np.isclose(
                uv_object.integration_time,
                original_int_time,
                rtol=uv_object._integration_time.tols[0],
                atol=uv_object._integration_time.tols[0],
            ),
            np.isclose(
                uv_object.integration_time,
                min_integration_time,
                rtol=uv_object._integration_time.tols[0],
                atol=uv_object._integration_time.tols[0],
            ),
        )
    )

    # make sure integration time is correct
    # in this case, all integration times should be the target one
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )

    # as usual, the new data should be the average of the input data (3 points now)
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        np.mean(init_wf[0:3, 0, 0]),
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
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
    uv_object.reorder_blts("baseline", minor_order="time")
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
    assert np.isclose(
        np.mean(init_wf[0:3, 0, 0]),
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")

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
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the sum
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]),
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")

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
    uv_object.flag_array[inds01, 0, 0][0] = True
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[1, 0, 0],
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")

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
    uv_object.flag_array[inds01, 0, 0][:2] = True
    uv_object.downsample_in_time(
        min_int_time=min_integration_time,
        blt_order="baseline",
        minor_order="time",
        summing_correlator_mode=True,
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]),
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")

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
            np.isclose(
                uv_object.integration_time,
                original_int_time,
                rtol=uv_object._integration_time.tols[0],
                atol=uv_object._integration_time.tols[0],
            ),
            np.isclose(
                uv_object.integration_time,
                min_integration_time,
                rtol=uv_object._integration_time.tols[0],
                atol=uv_object._integration_time.tols[0],
            ),
        )
    )

    # as usual, the new data should be the average of the input data (3 points now)
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        np.sum(init_wf[0:3, 0, 0]),
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        np.mean(init_ns[0:3, 0, 0]),
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")

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
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )

    # as usual, the new data should be the average of the input data
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        np.sum(init_wf[0:3, 0, 0]),
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        np.mean(init_ns[0:3, 0, 0]),
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline")

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
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )

    # output data should be the same
    out_wf_01 = uv_object.get_data(0, 1)
    out_wf_02 = uv_object.get_data(0, 2)
    assert np.all(init_wf_01 == out_wf_01)
    assert np.isclose(
        (init_wf_02[0, 0, 0] + init_wf_02[1, 0, 0]) / 2.0,
        out_wf_02[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns_01 = uv_object.get_nsamples(0, 1)
    out_ns_02 = uv_object.get_nsamples(0, 2)
    np.testing.assert_allclose(
        out_ns_01,
        init_ns_01,
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )
    assert np.isclose(
        (init_ns_02[0, 0, 0] + init_ns_02[1, 0, 0]) / 2.0,
        out_ns_02[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_drift(hera_uvh5):
    """Test the downsample_in_time method on drift mode data"""
    uv_object = hera_uvh5

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")
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
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )

    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
@pytest.mark.parametrize("driftscan", [True, False])
@pytest.mark.parametrize("partial_phase", [True, False])
def test_downsample_in_time_drift_no_phasing(hera_uvh5, driftscan, partial_phase):
    """Test the downsample_in_time method on drift mode data without phasing"""
    uv_object = hera_uvh5

    if driftscan:
        uv_object.phase(
            lon=0,
            lat=np.pi / 2,
            phase_frame="altaz",
            cat_type="driftscan",
            cat_name="foo",
        )

    if partial_phase:
        mask = np.full(uv_object.Nblts, False)
        mask[uv_object.Nblts // 2 :] = True
        uv_object.phase(
            ra=0, dec=0, phase_frame="icrs", select_mask=mask, cat_name="bar"
        )

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")
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
        min_int_time=min_integration_time, blt_order="baseline", allow_drift=False
    )

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be similar to the average, but somewhat different
    # because of the phasing
    out_wf = uv_object.get_data(0, 1)
    new_tol = 5e-2 * np.amax(np.abs(uv_object.data_array))
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        atol=new_tol,
        rtol=0,
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")
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
    uv_object.flag_array[inds01, 0, 0][0] = True
    uv_object2 = uv_object.copy()

    # change precision of nsample array
    uv_object.nsample_array = uv_object.nsample_array.astype(np.float16)
    uv_object.downsample_in_time(
        min_int_time=min_integration_time, blt_order="baseline", minor_order="time"
    )
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        init_wf[1, 0, 0],
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # make sure nsamples is correct
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")

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

    # raise an error if phase centers change within an downsampling window
    uv_object2 = uv_object.copy()
    uv_object2.reorder_blts("time")
    mask = np.full(uv_object2.Nblts, False)
    mask[: uv_object2.Nblts // 3] = True
    uv_object2.phase(ra=0, dec=0, phase_frame="icrs", select_mask=mask, cat_name="foo")
    with pytest.raises(
        ValueError,
        match=(
            "Multiple phase centers included in a downsampling window. Use `phase` to"
            " phase to a single phase center or decrease the `min_int_time` or"
            " `n_times_to_avg` parameter to avoid multiple phase centers being included"
            " in a downsampling window."
        ),
    ):
        uv_object2.downsample_in_time(n_times_to_avg=2)

    # catch a warning for doing no work
    uv_object2 = uv_object.copy()
    max_integration_time = 0.5 * np.amin(uv_object.integration_time)
    with check_warnings(
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
    initial_int_time = uv_object.integration_time[inds01][0]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01][-1] += initial_int_time / (24 * 3600)
    uv_object.Ntimes += 1
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    times_01 = uv_object.get_times(0, 1)
    assert np.unique(np.diff(times_01)).size > 1
    with check_warnings(
        UserWarning,
        match=[
            "There is a gap in the times of baseline",
            (
                "The lst_array is not self-consistent with the time_array and telescope"
                " location. Consider recomputing with the `set_lsts_from_time_array`"
                " method"
            ),
        ],
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_int_time_mismatch_warning(hera_uvh5):
    """Test warning in downsample_in_time about mismatch between integration
    times and the time between integrations.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")

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
    with check_warnings(
        UserWarning,
        match="The time difference between integrations is not the same",
        nwarnings=10,
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have half the size of the data array and all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )
    assert uv_object.data_array.size * 2 == init_data_size

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_integration_time(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # test handling (& warnings) with varying integration time in a baseline
    # First, change both integration time & time array to match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01][0]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01][-2] += (initial_int_time / 2) / (24 * 3600)
    uv_object.time_array[inds01][-1] += (3 * initial_int_time / 2) / (24 * 3600)
    uv_object.set_lsts_from_time_array()
    uv_object.integration_time[inds01][-2:] += initial_int_time
    uv_object.Ntimes = np.unique(uv_object.time_array).size
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    # check that there are no warnings about inconsistencies between
    # integration_time & time_array
    uv_object.check()
    uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )

    out_wf = uv_object.get_data(0, 1)

    n_times_in = init_wf.shape[0]
    n_times_out = out_wf.shape[0]
    assert n_times_out == (n_times_in - 2) / 2 + 2

    # output data should be the average for the first set
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    # last 2 time samples should be identical to initial ones
    assert np.isclose(
        init_wf[-1, 0, 0],
        out_wf[-1, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_wf[-2, 0, 0],
        out_wf[-2, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )
    assert np.isclose(
        init_ns[-1, 0, 0],
        out_ns[-1, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )
    assert np.isclose(
        init_ns[-2, 0, 0],
        out_ns[2, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_downsample_in_time_varying_int_time_partial_flags(hera_uvh5):
    """Test downsample_in_time handling of file with integration time changing
    within a baseline and partial flagging.
    """
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))
    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")

    # downselect to 14 times and one baseline
    uv_object.select(times=np.unique(uv_object.time_array)[:14])

    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0

    # change last 2 integrations to be twice as long
    # (so 12 normal length, 2 double length)
    # change integration time & time array to match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01][0]
    # time array is in jd, integration time is in sec
    uv_object.time_array[inds01][-2] += (initial_int_time / 2) / (24 * 3600)
    uv_object.time_array[inds01][-1] += (3 * initial_int_time / 2) / (24 * 3600)
    uv_object.set_lsts_from_time_array()
    uv_object.integration_time[inds01][-2:] += initial_int_time
    uv_object.Ntimes = np.unique(uv_object.time_array).size

    # add a flag on last time
    uv_object.flag_array[inds01][-1] = True
    # add a flag on thrid to last time
    uv_object.flag_array[inds01][-3] = True

    uv_object2 = uv_object.copy()

    with check_warnings(None):
        uv_object.downsample_in_time(min_int_time=4 * initial_int_time)
    with check_warnings(None):
        uv_object.downsample_in_time(min_int_time=8 * initial_int_time)
    with check_warnings(None):
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
    uv_object.reorder_blts("baseline", minor_order="time")

    # save some values for later
    init_wf = uv_object.get_data(0, 1)
    # check that there are no flags
    assert np.nonzero(uv_object.flag_array)[0].size == 0
    init_ns = uv_object.get_nsamples(0, 1)

    # Next, change just integration time, so time array doesn't match
    inds01 = uv_object.antpair2ind(0, 1)
    initial_int_time = uv_object.integration_time[inds01][0]
    uv_object.integration_time[inds01][-2:] += initial_int_time
    min_integration_time = 2 * np.amin(uv_object.integration_time)
    with check_warnings(
        UserWarning, "The time difference between integrations is different than"
    ):
        uv_object.downsample_in_time(min_int_time=min_integration_time)

    # Should have all the new integration time
    # (for this file with 20 integrations and a factor of 2 downsampling)
    np.testing.assert_allclose(
        uv_object.integration_time,
        min_integration_time,
        rtol=uv_object._integration_time.tols[0],
        atol=uv_object._integration_time.tols[0],
    )

    # output data should be the average
    out_wf = uv_object.get_data(0, 1)
    assert np.isclose(
        (init_wf[0, 0, 0] + init_wf[1, 0, 0]) / 2.0,
        out_wf[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    # this should be true because there are no flags
    out_ns = uv_object.get_nsamples(0, 1)
    assert np.isclose(
        (init_ns[0, 0, 0] + init_ns[1, 0, 0]) / 2.0,
        out_ns[0, 0, 0],
        rtol=uv_object._nsample_array.tols[0],
        atol=uv_object._nsample_array.tols[1],
    )

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
    uv_object.reorder_blts("baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time
    new_Nblts = uv_object.Nblts

    # check that calling upsample again with the same max_integration_time
    # gives warning and does nothing
    with check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert uv_object.Nblts == new_Nblts

    # check that calling upsample again with the almost the same max_integration_time
    # gives warning and does nothing
    small_number = 0.9 * uv_object._integration_time.tols[1]
    with check_warnings(
        UserWarning, "All values in the integration_time array are already longer"
    ):
        uv_object.upsample_in_time(
            max_integration_time - small_number, blt_order="baseline"
        )
    assert uv_object.Nblts == new_Nblts

    uv_object.downsample_in_time(
        min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline"
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
    with check_warnings(
        UserWarning, match="All values in the integration_time array are already longer"
    ):
        uv_object.downsample_in_time(
            min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline"
        )
    assert uv_object.Nblts == uv_object2.Nblts

    # check that calling upsample again with the almost the same min_integration_time
    # gives warning and does nothing
    with check_warnings(
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
def test_upsample_downsample_in_time_odd_resample(hera_uvh5):
    """Test round trip works with odd resampling"""
    uv_object = hera_uvh5
    uv_object.phase_to_time(Time(uv_object.time_array[0], format="jd"))

    # reorder to make sure we get the right value later
    uv_object.reorder_blts("baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    # try again with a resampling factor of 3 (test odd numbers)
    max_integration_time = np.amin(uv_object.integration_time) / 3.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time

    uv_object.downsample_in_time(
        min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline"
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
    uv_object.reorder_blts("baseline", minor_order="time")
    uv_object2 = uv_object.copy()

    max_integration_time = np.amin(uv_object.integration_time) / 2.0
    uv_object.upsample_in_time(max_integration_time, blt_order="baseline")
    assert np.amax(uv_object.integration_time) <= max_integration_time

    uv_object.downsample_in_time(
        min_int_time=np.amin(uv_object2.integration_time), blt_order="baseline"
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
@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
@pytest.mark.parametrize("driftscan", [True, False])
@pytest.mark.parametrize("partial_phase", [True, False])
def test_resample_in_time(bda_test_file, driftscan, partial_phase):
    """Test the resample_in_time method"""
    # Note this file has slight variations in the delta t between integrations
    # that causes our gap test to issue a warning, but the variations are small
    # We aren't worried about them, so we filter those warnings
    uv_object = bda_test_file

    if driftscan:
        uv_object.phase(
            lon=0,
            lat=np.pi / 2,
            phase_frame="altaz",
            cat_type="driftscan",
            cat_name="zenith",
        )

    if partial_phase:
        mask = np.full(uv_object.Nblts, False)
        mask[: uv_object.Nblts // 2] = True
        uv_object.phase(
            ra=0, dec=0, phase_frame="icrs", select_mask=mask, cat_name="foo"
        )

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
    np.testing.assert_allclose(uv_object.integration_time, 8)

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
    assert np.isclose(
        np.mean(init_data_1_136[0:4, 0, 0]),
        out_data_1_136[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        np.mean(init_data_1_137[0:2, 0, 0]),
        out_data_1_137[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_data_1_138[0, 0, 0],
        out_data_1_138[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_data_136_137[0, 0, 0],
        out_data_136_137[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    return


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
            np.isclose(
                uv_object.integration_time,
                8,
                rtol=uv_object._integration_time.tols[0],
                atol=uv_object._integration_time.tols[0],
            ),
            np.isclose(
                uv_object.integration_time,
                16,
                rtol=uv_object._integration_time.tols[0],
                atol=uv_object._integration_time.tols[0],
            ),
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
    assert np.isclose(
        np.mean(init_data_1_136[0:4, 0, 0]),
        out_data_1_136[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        np.mean(init_data_1_137[0:2, 0, 0]),
        out_data_1_137[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_data_1_138[0, 0, 0],
        out_data_1_138[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_data_136_137[0, 0, 0],
        out_data_136_137[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    return


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
        np.isclose(
            uv_object.integration_time,
            2,
            rtol=uv_object._integration_time.tols[0],
            atol=uv_object._integration_time.tols[0],
        )
        | np.isclose(
            uv_object.integration_time,
            4,
            rtol=uv_object._integration_time.tols[0],
            atol=uv_object._integration_time.tols[0],
        )
        | np.isclose(
            uv_object.integration_time,
            8,
            rtol=uv_object._integration_time.tols[0],
            atol=uv_object._integration_time.tols[0],
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
    assert np.isclose(
        init_data_1_136[0, 0, 0],
        out_data_1_136[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_data_1_137[0, 0, 0],
        out_data_1_137[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_data_1_138[0, 0, 0],
        out_data_1_138[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )
    assert np.isclose(
        init_data_136_137[0, 0, 0],
        out_data_136_137[0, 0, 0],
        rtol=uv_object._data_array.tols[0],
        atol=uv_object._data_array.tols[1],
    )

    return


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
    uv.flag_array[0] = True
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
def test_downsample_in_time_mwa(mwa_integration_time):
    """
    Test resample in time works with numerical weirdnesses.

    In particular, when min_int_time is not quite an integer mulitple of
    integration_time. This text broke with a prior bug (see issue 773).
    """
    uv = mwa_integration_time
    uv.phase_to_time(np.mean(uv.time_array))
    uv_object2 = uv.copy()

    # all data within 5 milliseconds of 2 second integrations
    np.testing.assert_allclose(uv.integration_time, 2, atol=5e-3, rtol=0)
    min_int_time = 4.0
    uv.resample_in_time(min_int_time, only_downsample=True, keep_ragged=False)

    assert np.all(uv.integration_time > (min_int_time - 5e-3))

    # Now do the human expected thing:
    init_data = uv_object2.get_data((61, 58))
    uv_object2.downsample_in_time(n_times_to_avg=2, keep_ragged=False)

    assert uv_object2.Ntimes == 5

    out_data = uv_object2.get_data((61, 58))

    assert np.isclose(
        np.mean(init_data[0:2, 0, 0]),
        out_data[0, 0, 0],
        rtol=uv._data_array.tols[0],
        atol=uv._data_array.tols[1],
    )


@pytest.mark.filterwarnings("ignore:There is a gap in the times of baseline")
def test_resample_in_time_warning(mwa_integration_time):
    uv = mwa_integration_time
    uv2 = uv.copy()

    with check_warnings(
        UserWarning, match="No resampling will be done because target time"
    ):
        uv.resample_in_time(3, keep_ragged=False)

    assert uv2 == uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("multi_spw", [True, False])
@pytest.mark.parametrize("sum_corr", [True, False])
def test_frequency_average(casa_uvfits, multi_spw, sum_corr):
    """Test averaging in frequency."""
    uvobj = casa_uvfits

    if multi_spw:
        # Make multiple spws
        spw_nchan = int(uvobj.Nfreqs / 4)
        uvobj.flex_spw_id_array = np.concatenate(
            (
                np.full(spw_nchan, 0, dtype=int),
                np.full(spw_nchan, 1, dtype=int),
                np.full(spw_nchan, 2, dtype=int),
                np.full(spw_nchan, 3, dtype=int),
            )
        )
        uvobj.spw_array = np.arange(4)
        uvobj.Nspws = 4

    uvobj2 = uvobj.copy()

    eq_coeffs = np.tile(
        np.arange(uvobj.Nfreqs, dtype=np.float64), (uvobj.telescope.Nants, 1)
    )
    uvobj.eq_coeffs = eq_coeffs

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    n_chan_to_avg = 2
    with check_warnings(UserWarning, "eq_coeffs vary by frequency"):
        uvobj.frequency_average(2, keep_ragged=True, summing_correlator_mode=sum_corr)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / n_chan_to_avg)

    input_freqs = np.squeeze(uvobj2.freq_array)

    expected_freqs = input_freqs[
        np.arange((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
    ]
    expected_freqs = expected_freqs.reshape(
        int(uvobj2.Nfreqs // n_chan_to_avg), n_chan_to_avg
    ).mean(axis=1)

    input_chan_width = uvobj2.channel_width

    expected_chan_widths = np.full(
        int(uvobj2.Nfreqs // n_chan_to_avg),
        input_chan_width[0] * n_chan_to_avg,
        dtype=float,
    )

    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0
    assert np.max(np.abs(uvobj.channel_width - expected_chan_widths)) == 0

    expected_coeffs = eq_coeffs.reshape(
        uvobj2.telescope.Nants, int(uvobj2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvobj.eq_coeffs - expected_coeffs)) == 0

    # no flagging, so the following is true
    expected_data = uvobj2.get_data(1, 2)
    reshape_tuple = (expected_data.shape[0], int(uvobj2.Nfreqs / 2), 2, uvobj2.Npols)
    if sum_corr:
        expected_data = expected_data.reshape(reshape_tuple).sum(axis=2)
    else:
        expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)

    np.testing.assert_allclose(
        uvobj.get_data(1, 2, squeeze="none"),
        expected_data,
        rtol=uvobj._data_array.tols[0],
        atol=uvobj._data_array.tols[1],
    )

    assert np.nonzero(uvobj.flag_array)[0].size == 0

    assert not isinstance(uvobj.data_array, np.ma.MaskedArray)
    assert not isinstance(uvobj.nsample_array, np.ma.MaskedArray)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("sum_corr", [True, True])
@pytest.mark.parametrize(
    ["multi_spw", "respect_spws"], [[True, True], [True, False], [False, False]]
)
@pytest.mark.parametrize("keep_ragged", [True, False])
def test_frequency_average_uneven(
    casa_uvfits, keep_ragged, sum_corr, multi_spw, respect_spws
):
    """Test averaging in frequency with a number that is not a factor of Nfreqs."""
    uvobj = casa_uvfits

    eq_coeffs = np.tile(
        np.arange(uvobj.Nfreqs, dtype=np.float64), (uvobj.telescope.Nants, 1)
    )
    uvobj.eq_coeffs = eq_coeffs

    if multi_spw:
        # Make multiple spws
        spw_nchan = int(uvobj.Nfreqs / 4)
        uvobj.flex_spw_id_array = np.concatenate(
            (
                np.full(spw_nchan, 0, dtype=int),
                np.full(spw_nchan, 1, dtype=int),
                np.full(spw_nchan, 2, dtype=int),
                np.full(spw_nchan, 3, dtype=int),
            )
        )
        uvobj.spw_array = np.arange(4)
        uvobj.Nspws = 4

    uvobj2 = uvobj.copy()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0
    n_chan_to_avg = 7

    warn = [UserWarning]
    msg = [
        re.escape(
            "eq_coeffs vary by frequency. They should be applied to the data using "
            "`remove_eq_coeffs` before frequency averaging"
        )
    ]

    with check_warnings(warn, match=msg):
        uvobj.frequency_average(
            n_chan_to_avg=n_chan_to_avg,
            keep_ragged=keep_ragged,
            summing_correlator_mode=sum_corr,
            respect_spws=respect_spws,
        )

    assert uvobj2.Nfreqs % n_chan_to_avg != 0

    input_freqs = np.squeeze(uvobj2.freq_array)

    input_chan_width = uvobj2.channel_width

    if multi_spw and respect_spws:
        expected_freqs = np.array([])
        expected_chan_widths = np.array([])
        for spw_ind in range(uvobj2.Nspws):
            start_chan = spw_ind * spw_nchan
            n_reg_chan = (spw_nchan // n_chan_to_avg) * n_chan_to_avg
            this_expected = input_freqs[start_chan : (start_chan + n_reg_chan)]
            this_expected = this_expected.reshape(
                int(spw_nchan // n_chan_to_avg), n_chan_to_avg
            ).mean(axis=1)
            this_expected_cw = input_chan_width[start_chan : (start_chan + n_reg_chan)]
            this_expected_cw = this_expected_cw.reshape(
                int(spw_nchan // n_chan_to_avg), n_chan_to_avg
            ).sum(axis=1)
            if keep_ragged:
                this_expected = np.append(
                    this_expected,
                    np.mean(
                        input_freqs[
                            (start_chan + n_reg_chan) : (spw_ind + 1) * spw_nchan
                        ]
                    ),
                )
                this_expected_cw = np.append(
                    this_expected_cw,
                    np.sum(
                        input_chan_width[
                            (start_chan + n_reg_chan) : (spw_ind + 1) * spw_nchan
                        ]
                    ),
                )
            expected_freqs = np.append(expected_freqs, this_expected)
            expected_chan_widths = np.append(expected_chan_widths, this_expected_cw)
        if keep_ragged:
            assert uvobj.Nfreqs == (spw_nchan // n_chan_to_avg + 1) * uvobj2.Nspws
        else:
            assert uvobj.Nfreqs == (spw_nchan // n_chan_to_avg) * uvobj2.Nspws
    else:
        expected_freqs = input_freqs[
            np.arange((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
        ]
        expected_freqs = expected_freqs.reshape(
            int(uvobj2.Nfreqs // n_chan_to_avg), n_chan_to_avg
        ).mean(axis=1)

        expected_chan_widths = input_chan_width[
            np.arange((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
        ]
        expected_chan_widths = expected_chan_widths.reshape(
            int(uvobj2.Nfreqs // n_chan_to_avg), n_chan_to_avg
        ).sum(axis=1)

        if keep_ragged:
            assert uvobj.Nfreqs == (uvobj2.Nfreqs // n_chan_to_avg + 1)
            expected_freqs = np.append(
                expected_freqs,
                np.mean(
                    input_freqs[(uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg :]
                ),
            )
            expected_chan_widths = np.append(
                expected_chan_widths,
                np.sum(
                    input_chan_width[(uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg :]
                ),
            )
        else:
            assert uvobj.Nfreqs == (uvobj2.Nfreqs // n_chan_to_avg)

    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0
    assert np.max(np.abs(uvobj.channel_width - expected_chan_widths)) == 0

    # no flagging, so the following is true
    initial_data = uvobj2.get_data(1, 2)
    if multi_spw and respect_spws:
        reshape_tuple = (
            initial_data.shape[0],
            int(spw_nchan // n_chan_to_avg),
            n_chan_to_avg,
            uvobj2.Npols,
        )
        for spw_ind in range(uvobj2.Nspws):
            start_chan = spw_ind * spw_nchan
            n_reg_chan = (spw_nchan // n_chan_to_avg) * n_chan_to_avg

            this_expected = initial_data[:, start_chan : (start_chan + n_reg_chan)]
            if sum_corr:
                this_expected = this_expected.reshape(reshape_tuple).sum(axis=2)
            else:
                this_expected = this_expected.reshape(reshape_tuple).mean(axis=2)

            if keep_ragged:
                if sum_corr:
                    this_expected = np.append(
                        this_expected,
                        initial_data[
                            :, (start_chan + n_reg_chan) : (spw_ind + 1) * spw_nchan
                        ].sum(axis=1, keepdims=True),
                        axis=1,
                    )
                else:
                    this_expected = np.append(
                        this_expected,
                        initial_data[
                            :, (start_chan + n_reg_chan) : (spw_ind + 1) * spw_nchan
                        ].mean(axis=1, keepdims=True),
                        axis=1,
                    )
            if spw_ind == 0:
                expected_data = this_expected
            else:
                expected_data = np.append(expected_data, this_expected, axis=1)
    else:
        expected_data = initial_data[
            :, 0 : ((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
        ]
        reshape_tuple = (
            initial_data.shape[0],
            int(uvobj2.Nfreqs // n_chan_to_avg),
            n_chan_to_avg,
            uvobj2.Npols,
        )
        if sum_corr:
            expected_data = expected_data.reshape(reshape_tuple).sum(axis=2)
        else:
            expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)

        if keep_ragged:
            if sum_corr:
                expected_data = np.append(
                    expected_data,
                    initial_data[
                        :, ((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg) :
                    ].sum(axis=1, keepdims=True),
                    axis=1,
                )
            else:
                expected_data = np.append(
                    expected_data,
                    initial_data[
                        :, ((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg) :
                    ].mean(axis=1, keepdims=True),
                    axis=1,
                )

    np.testing.assert_allclose(
        uvobj.get_data(1, 2, squeeze="none"),
        expected_data,
        rtol=uvobj._data_array.tols[0],
        atol=uvobj._data_array.tols[1],
    )

    assert np.nonzero(uvobj.flag_array)[0].size == 0


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("keep_ragged", [True, False])
@pytest.mark.parametrize("fully_flagged", [True, False])
def test_frequency_average_flagging(casa_uvfits, keep_ragged, fully_flagged):
    """Test averaging in frequency with flagging all samples averaged."""
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    n_chan_to_avg = 3

    # apply some flagging for testing
    inds01 = uvobj.antpair2ind(1, 2)
    if fully_flagged:
        uvobj.flag_array[inds01[0], 0:n_chan_to_avg, :] = True
        assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols * n_chan_to_avg
    else:
        uvobj.flag_array[inds01[0], 1:n_chan_to_avg, :] = True
        assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols * (n_chan_to_avg - 1)

    with check_warnings(None):
        uvobj.frequency_average(n_chan_to_avg=n_chan_to_avg, keep_ragged=keep_ragged)

    input_freqs = np.squeeze(uvobj2.freq_array)

    expected_freqs = input_freqs[
        np.arange((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
    ]
    expected_freqs = expected_freqs.reshape(
        int(uvobj2.Nfreqs // n_chan_to_avg), n_chan_to_avg
    ).mean(axis=1)

    if keep_ragged:
        assert uvobj.Nfreqs == (uvobj2.Nfreqs // n_chan_to_avg + 1)
        expected_freqs = np.append(
            expected_freqs,
            np.mean(input_freqs[(uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg :]),
        )
    else:
        assert uvobj.Nfreqs == (uvobj2.Nfreqs // n_chan_to_avg)

    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    initial_data = uvobj2.get_data(1, 2)
    expected_data = initial_data[
        :, 0 : ((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
    ]
    reshape_tuple = (
        initial_data.shape[0],
        int(uvobj2.Nfreqs // n_chan_to_avg),
        n_chan_to_avg,
        uvobj2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)

    if keep_ragged:
        expected_data = np.append(
            expected_data,
            initial_data[:, ((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg) :].mean(
                axis=1, keepdims=True
            ),
            axis=1,
        )
    if not fully_flagged:
        expected_data[0, 0, :] = uvobj2.data_array[inds01[0], 0, :]

    np.testing.assert_allclose(
        uvobj.get_data(1, 2, squeeze="none"),
        expected_data,
        rtol=uvobj._data_array.tols[0],
        atol=uvobj._data_array.tols[1],
    )

    if fully_flagged:
        assert np.sum(uvobj.flag_array[inds01[0], 0, :]) == 4
        assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols
        assert np.nonzero(uvobj.flag_array[inds01[1:], 0, :])[0].size == 0
    else:
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
    inds01 = uvobj.antpair2ind(1, 2)
    uvobj.flag_array[inds01[0], 0:3, :] = True
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols * 3

    uvobj.frequency_average(n_chan_to_avg=2)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    expected_freqs = uvobj2.freq_array.reshape((-1, 2)).mean(axis=1)
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_data = uvobj2.get_data(1, 2, squeeze="none")
    reshape_tuple = (expected_data.shape[0], int(uvobj2.Nfreqs / 2), 2, uvobj2.Npols)
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)

    expected_data[0, 1, :] = uvobj2.data_array[inds01[0], 3, :]

    np.testing.assert_allclose(
        uvobj.get_data(1, 2, squeeze="none"),
        expected_data,
        rtol=uvobj._data_array.tols[0],
        atol=uvobj._data_array.tols[1],
    )
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
    inds01 = uvobj.antpair2ind(1, 2)
    uvobj.flag_array[inds01[0], 0, :] = True
    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols

    uv_object3 = uvobj.copy()

    uvobj.frequency_average(n_chan_to_avg=2)
    uvobj.frequency_average(n_chan_to_avg=2)

    uv_object3.frequency_average(n_chan_to_avg=4)

    assert uvobj == uv_object3


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("keep_ragged", [True, False])
def test_frequency_average_propagate_flags(casa_uvfits, keep_ragged):
    """
    Test averaging in frequency with flagging all of one and only one of
    another sample averaged, and propagating flags. Data should be identical,
    but flags should be slightly different compared to other test of the same
    name.
    """
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    n_chan_to_avg = 3

    # apply some flagging for testing
    inds01 = uvobj.antpair2ind(1, 2)
    uvobj.flag_array[inds01[0], 0 : (n_chan_to_avg * 2 - 1), :] = True

    assert np.nonzero(uvobj.flag_array)[0].size == uvobj.Npols * (n_chan_to_avg * 2 - 1)

    with check_warnings(None):
        uvobj.frequency_average(
            n_chan_to_avg=n_chan_to_avg, propagate_flags=True, keep_ragged=keep_ragged
        )

    input_freqs = np.squeeze(uvobj2.freq_array)

    expected_freqs = input_freqs[
        np.arange((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
    ]
    expected_freqs = expected_freqs.reshape(
        int(uvobj2.Nfreqs // n_chan_to_avg), n_chan_to_avg
    ).mean(axis=1)

    if keep_ragged:
        assert uvobj.Nfreqs == (uvobj2.Nfreqs // n_chan_to_avg + 1)
        expected_freqs = np.append(
            expected_freqs,
            np.mean(input_freqs[(uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg :]),
        )
    else:
        assert uvobj.Nfreqs == (uvobj2.Nfreqs // n_chan_to_avg)

    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    initial_data = uvobj2.get_data(1, 2)
    expected_data = initial_data[
        :, 0 : ((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg)
    ]
    reshape_tuple = (
        initial_data.shape[0],
        int(uvobj2.Nfreqs // n_chan_to_avg),
        n_chan_to_avg,
        uvobj2.Npols,
    )
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)
    expected_data[0, 1, :] = np.take(
        uvobj2.data_array[inds01[0]], n_chan_to_avg * 2 - 1, axis=0
    )

    if keep_ragged:
        expected_data = np.append(
            expected_data,
            initial_data[:, ((uvobj2.Nfreqs // n_chan_to_avg) * n_chan_to_avg) :].mean(
                axis=1, keepdims=True
            ),
            axis=1,
        )

    np.testing.assert_allclose(
        uvobj.get_data(1, 2, squeeze="none"),
        expected_data,
        rtol=uvobj._data_array.tols[0],
        atol=uvobj._data_array.tols[1],
    )
    # Twice as many flags should exist compared to test of previous name.
    assert np.nonzero(uvobj.flag_array)[0].size == 2 * uvobj.Npols


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_frequency_average_nsample_precision(casa_uvfits):
    """Test averaging in frequency with a half-precision nsample_array."""
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()
    eq_coeffs = np.tile(
        np.arange(uvobj.Nfreqs, dtype=np.float64), (uvobj.telescope.Nants, 1)
    )
    uvobj.eq_coeffs = eq_coeffs
    uvobj.check()

    # check that there's no flagging
    assert np.nonzero(uvobj.flag_array)[0].size == 0

    # change precision of the nsample array
    uvobj.nsample_array = uvobj.nsample_array.astype(np.float16)

    with check_warnings(UserWarning, "eq_coeffs vary by frequency"):
        uvobj.frequency_average(n_chan_to_avg=2)

    assert uvobj.Nfreqs == (uvobj2.Nfreqs / 2)

    expected_freqs = uvobj2.freq_array.reshape(-1, 2).mean(axis=1)
    assert np.max(np.abs(uvobj.freq_array - expected_freqs)) == 0

    expected_coeffs = eq_coeffs.reshape(
        uvobj2.telescope.Nants, int(uvobj2.Nfreqs / 2), 2
    ).mean(axis=2)
    assert np.max(np.abs(uvobj.eq_coeffs - expected_coeffs)) == 0

    # no flagging, so the following is true
    expected_data = uvobj2.get_data(1, 2, squeeze="none")
    reshape_tuple = (expected_data.shape[0], int(uvobj2.Nfreqs / 2), 2, uvobj2.Npols)
    expected_data = expected_data.reshape(reshape_tuple).mean(axis=2)
    np.testing.assert_allclose(
        uvobj.get_data(1, 2, squeeze="none"),
        expected_data,
        rtol=uvobj._data_array.tols[0],
        atol=uvobj._data_array.tols[1],
    )

    assert np.nonzero(uvobj.flag_array)[0].size == 0

    assert not isinstance(uvobj.data_array, np.ma.MaskedArray)
    assert not isinstance(uvobj.nsample_array, np.ma.MaskedArray)

    # make sure we still have a half-precision nsample_array
    assert uvobj.nsample_array.dtype.type is np.float16


def test_frequency_average_warnings(casa_uvfits):
    # test errors with varying freq spacing but with one spw
    uvd = casa_uvfits.copy()
    uvd.freq_array[-1] += uvd.channel_width[0]
    with check_warnings(
        UserWarning,
        match=[
            re.escape(
                "The frequency spacing and/or channel widths vary, so after averaging "
                "they will also vary."
            )
        ],
    ):
        uvd.frequency_average(n_chan_to_avg=3)

    # also test errors with varying channel widths but with one spw
    uvd = casa_uvfits.copy()
    uvd.channel_width[-1] *= 2
    with check_warnings(
        UserWarning,
        match=re.escape(
            "The frequency spacing and/or channel widths vary, so after averaging "
            "they will also vary."
        ),
    ):
        uvd.frequency_average(n_chan_to_avg=2)

    # test warning with freq spacing not equal to channel width
    uvd = casa_uvfits.copy()
    uvd.channel_width *= 0.5
    with check_warnings(
        UserWarning,
        match=re.escape(
            "The frequency spacing is even but not equal to the channel width, so "
            "after averaging the channel_width will also not match the frequency "
            "spacing."
        ),
    ):
        uvd.frequency_average(n_chan_to_avg=2)

    spacing_error, chanwidth_error = uvd._check_freq_spacing(raise_errors=None)
    assert not spacing_error
    assert chanwidth_error


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_remove_eq_coeffs_divide(casa_uvfits):
    """Test using the remove_eq_coeffs method with divide convention."""
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()

    # give eq_coeffs to the object
    eq_coeffs = np.empty((uvobj.telescope.Nants, uvobj.Nfreqs), dtype=np.float64)
    for i, ant in enumerate(uvobj.telescope.antenna_numbers):
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
        np.testing.assert_allclose(
            norm_data,
            unnorm_data / (eq1 * eq2),
            rtol=uvobj._data_array.tols[0],
            atol=uvobj._data_array.tols[1],
        )

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_remove_eq_coeffs_multiply(casa_uvfits):
    """Test using the remove_eq_coeffs method with multiply convention."""
    uvobj = casa_uvfits
    uvobj2 = uvobj.copy()

    # give eq_coeffs to the object
    eq_coeffs = np.empty((uvobj.telescope.Nants, uvobj.Nfreqs), dtype=np.float64)
    for i, ant in enumerate(uvobj.telescope.antenna_numbers):
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
        np.testing.assert_allclose(
            norm_data,
            unnorm_data * (eq1 * eq2),
            rtol=uvobj._data_array.tols[0],
            atol=uvobj._data_array.tols[1],
        )

    return


def test_remove_eq_coeffs_errors(casa_uvfits):
    """Test errors raised by remove_eq_coeffs method."""
    uvobj = casa_uvfits
    # raise error when eq_coeffs are not defined
    with pytest.raises(ValueError, match="The eq_coeffs attribute must be defined"):
        uvobj.remove_eq_coeffs()

    # raise error when eq_coeffs are defined but not eq_coeffs_convention
    uvobj.eq_coeffs = np.ones((uvobj.telescope.Nants, uvobj.Nfreqs))
    with pytest.raises(
        ValueError, match="The eq_coeffs_convention attribute must be defined"
    ):
        uvobj.remove_eq_coeffs()

    # raise error when convention is not a valid choice
    uvobj.eq_coeffs_convention = "foo"
    with pytest.raises(ValueError, match="Got unknown convention foo. Must be one of"):
        uvobj.remove_eq_coeffs()

    return


@pytest.mark.parametrize(
    "read_func,filelist",
    [
        ("read_miriad", [os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA")] * 2),
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
            [os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")] * 2,
        ),
        ("read_fhd", []),
    ],
)
def test_multifile_read_errors(read_func, filelist, fhd_data_files):
    uv = UVData()
    kwargs = {}
    if "fhd" in read_func:
        filelist = [[fhd_data_files["filename"][0]], [fhd_data_files["filename"][1]]]
        kwargs = fhd_data_files
        del kwargs["filename"]
    with pytest.raises(
        ValueError,
        match=(
            "Reading multiple files from class specific read functions is no "
            "longer supported."
        ),
    ):
        getattr(uv, read_func)(filelist, **kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multifile_read_check(hera_uvh5, tmp_path):
    """Test setting skip_bad_files=True when reading in files"""

    uv_true = hera_uvh5.copy()

    uvh5_file = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")

    # Create a test file and remove header info to 'corrupt' it
    testfile = os.path.join(tmp_path, "zen.2458661.23480.HH.uvh5")

    uv_true.write_uvh5(testfile)
    with h5py.File(testfile, "r+") as h5f:
        del h5f["Header/ant_1_array"]

    err_msg = "ant_1_array not found in"
    uv = UVData()
    # Test that the expected error arises
    with pytest.raises(KeyError, match=err_msg):
        uv.read(testfile, skip_bad_files=False)

    # Test when the corrupted file is at the beggining, skip_bad_files=False
    fileList = [testfile, uvh5_file]
    with (
        pytest.raises(KeyError, match=err_msg),
        check_warnings(UserWarning, match="Failed to read"),
    ):
        uv.read(fileList, skip_bad_files=False)
    assert uv != uv_true

    # Test when the corrupted file is at the beggining, skip_bad_files=True
    fileList = [testfile, uvh5_file]
    with check_warnings(UserWarning, match="Failed to read"):
        uv.read(fileList, skip_bad_files=True)
    assert uv == uv_true

    # Test when the corrupted file is at the end of a list
    fileList = [uvh5_file, testfile]
    with check_warnings(UserWarning, match="Failed to read"):
        uv.read(fileList, skip_bad_files=True)
    # Check that the uncorrupted file was still read in
    assert uv == uv_true

    # Test that selection happens when there's only one good file in a list
    uv_true2 = uv_true.copy()
    uv_true2.select(freq_chans=np.arange(uv_true.Nfreqs // 2))
    with check_warnings(UserWarning, match="Failed to read"):
        uv.read(
            fileList, skip_bad_files=True, freq_chans=np.arange(uv_true.Nfreqs // 2)
        )
    # Check that the uncorrupted file was still read in and selection is applied
    assert uv == uv_true2

    # Check that the uncorrupted file was still read in and phased properly
    uv._consolidate_phase_center_catalogs(
        reference_catalog=uv_true2.phase_center_catalog, ignore_name=True
    )
    assert uv == uv_true2
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
    uv_test = UVData()
    with check_warnings(UserWarning, "Failed to read"):
        uv_test.read(fileList[0:4], skip_bad_files=True)
    uv_true = UVData()
    uv_true.read(fileList[0:3], skip_bad_files=True)
    assert uv_test == uv_true

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
    uv_test = UVData()
    with check_warnings(UserWarning, "Failed to read"):
        uv_test.read(fileList[0:4], skip_bad_files=True)
    uv_true = UVData()
    uv_true.read(fileList[1:4], skip_bad_files=True)

    assert uv_test == uv_true

    err_msg = "ant_1_array not found"

    # Test with corrupted file first in list, but with skip_bad_files=False
    uv_test = UVData()
    if err_type == "KeyError":
        with (
            pytest.raises(KeyError, match=err_msg),
            check_warnings(UserWarning, match="Failed to read"),
        ):
            uv_test.read(fileList[0:4], skip_bad_files=False)
    elif err_type == "ValueError":
        with (
            pytest.raises(ValueError, match="Nants_data must be equal to"),
            check_warnings(UserWarning, match="Failed to read"),
        ):
            uv_test.read(fileList[0:4], skip_bad_files=False)
    uv_true = UVData()
    uv_true.read([fileList[1], fileList[2], fileList[3]], skip_bad_files=False)

    assert uv_test != uv_true

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
    uv_test = UVData()
    with check_warnings(UserWarning, "Failed to read"):
        uv_test.read(fileList[0:4], skip_bad_files=True)
    uv_true = UVData()
    uv_true.read([fileList[0], fileList[2], fileList[3]], skip_bad_files=True)

    assert uv_test == uv_true

    # Test with corrupted file in middle of list, but with skip_bad_files=False
    uv_test = UVData()
    if err_type == "KeyError":
        with (
            pytest.raises(KeyError, match=err_msg),
            check_warnings(UserWarning, match="Failed to read"),
        ):
            uv_test.read(fileList[0:4], skip_bad_files=False)
    elif err_type == "ValueError":
        with (
            pytest.raises(ValueError, match="Nants_data must be equal to"),
            check_warnings(UserWarning, match="Failed to read"),
        ):
            uv_test.read(fileList[0:4], skip_bad_files=False)
    uv_true = UVData()
    uv_true.read([fileList[0], fileList[2], fileList[3]], skip_bad_files=False)

    assert uv_test != uv_true

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
    uv_test = UVData()
    with check_warnings(
        UserWarning,
        match=(
            "########################################################\n"
            "ALL FILES FAILED ON READ - NO READABLE FILES IN FILENAME\n"
            "########################################################"
        ),
    ):
        uv_test.read(fileList[0:4], skip_bad_files=True)
    uv_true = UVData()

    assert uv_test == uv_true

    os.remove(fileList[0])
    os.remove(fileList[1])
    os.remove(fileList[2])
    os.remove(fileList[3])

    return


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
    ant_pair, pols = utils.bls.parse_ants(uvd, "cross")
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
    zenith_coord = SkyCoord(
        alt=Angle(90 * units.deg),
        az=Angle(0 * units.deg),
        obstime=time,
        frame="altaz",
        location=uvd.telescope.location,
    )

    obs_zenith_coord = zenith_coord.transform_to("icrs")
    zenith_ra = obs_zenith_coord.ra.rad
    zenith_dec = obs_zenith_coord.dec.rad

    uvd.phase_to_time(phase_time)
    cat_id = uvd.phase_center_id_array[0]

    assert uvd.phase_center_catalog[cat_id]["cat_lon"] == zenith_ra
    assert uvd.phase_center_catalog[cat_id]["cat_lat"] == zenith_dec


@pytest.mark.parametrize(
    ["identifier", "errtype", "msg"],
    [
        (-1, ValueError, "No entry with the ID -1 in the catalog."),
        (5.0, TypeError, "catalog_identifier must be a string, an integer or a list"),
        ("test", ValueError, "No entry by the name test in the catalog."),
    ],
)
def test_print_object_err(sma_mir, identifier, errtype, msg):
    with pytest.raises(errtype, match=msg):
        sma_mir.print_phase_center_info(catalog_identifier=identifier)


@pytest.mark.parametrize(
    "kwargs", [{}, {"catalog_identifier": "3c84"}, {"catalog_identifier": 1}]
)
def test_print_object_standard(sma_mir, kwargs):
    """
    Check that the 'standard' mode of print_object works.
    """
    check_str = (
        "   ID     Cat Entry          Type     Az/Lon/RA    El/Lat/Dec  Frame    Epoch \n"  # noqa
        "    #          Name                       hours           deg                 \n"  # noqa
        "------------------------------------------------------------------------------\n"  # noqa
        "    1          3c84      sidereal    3:19:48.16  +41:30:42.11   icrs  J2000.0 \n"  # noqa
    )

    table_str = sma_mir.print_phase_center_info(
        print_table=False, return_str=True, **kwargs
    )
    assert table_str == check_str

    # Make sure we can specify the object name and get the same result
    table_str = sma_mir.print_phase_center_info(
        print_table=False, return_str=True, catalog_identifier="3c84"
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
        "   ID     Cat Entry          Type      Az/Lon/RA    El/Lat/Dec  Frame    Epoch \n"  # noqa
        "    #          Name                          deg           deg                 \n"  # noqa
        "-------------------------------------------------------------------------------\n"  # noqa
        "    1          3c84      sidereal    49:57:02.40  +41:30:42.11   icrs  J2000.0 \n"  # noqa
    )

    # And likewise when forcing the degree format
    table_str = sma_mir.print_phase_center_info(
        print_table=False, return_str=True, hms_format=False
    )
    assert table_str == check_str


@pytest.mark.filterwarnings("ignore:The provided name")
@pytest.mark.parametrize(
    ["frame", "epoch"],
    [["fk5", 2000.0], ["fk5", "J2000.0"], ["fk4", "B1950.0"], ["fk4", 1950.0]],
)
def test_print_object_full(sma_mir, frame, epoch):
    """
    Test that print object w/ all optional paramters prints as expected.
    """
    # Now check and see what happens if we add the full suite of phase center parameters
    _ = sma_mir._add_phase_center(
        "3c84",
        cat_type="sidereal",
        cat_lat=-1.0,
        cat_lon=-1.0,
        cat_dist=0.0,
        cat_vrad=0.0,
        cat_pm_ra=0.0,
        cat_pm_dec=0.0,
        cat_frame=frame,
        cat_epoch=epoch,
        cat_id=list(sma_mir.phase_center_catalog)[0],
        force_update=True,
    )
    frame_str = str(frame)
    if frame == "fk5":
        if isinstance(epoch, str):
            epoch_str = epoch
        else:
            epoch_str = "J" + str(epoch)
    elif frame == "fk4":
        if isinstance(epoch, str):
            epoch_str = epoch
        else:
            epoch_str = "B" + str(epoch)
    check_str = (
        "   ID     Cat Entry          Type     Az/Lon/RA"
        "    El/Lat/Dec  Frame    Epoch   PM-Ra  PM-Dec     Dist   V_rad \n"
        "    #          Name                       hours"
        "           deg                  mas/yr  mas/yr       pc    km/s \n"
        "-----------------------------------------------"
        "----------------------------------------------------------------\n"
        "    1          3c84      sidereal   20:10:49.01"
        f"  -57:17:44.81    {frame_str:5s}{epoch_str:7s}       0       0  0.0e+00"
        "       0 \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


@pytest.mark.filterwarnings("ignore:The provided name")
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
        cat_id=list(sma_mir.phase_center_catalog)[0],
        force_update=True,
    )
    check_str = (
        "   ID     Cat Entry          Type     Az/Lon/RA"
        "    El/Lat/Dec  Frame        Ephem Range    \n"
        "    #          Name                       hours"
        "           deg         Start-MJD    End-MJD \n"
        "------------------------------------------------"
        "-------------------------------------------\n"
        "    1          3c84         ephem    0:00:00.00"
        "  + 0:00:00.00   icrs   56788.50   56788.50 \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


@pytest.mark.filterwarnings("ignore:The provided name")
def test_print_object_driftscan(sma_mir):
    """
    Test that printing driftscan objects works as expected.
    """
    # Check and see that if we force this to be a driftscan, we get the output
    # we expect
    _ = sma_mir._add_phase_center(
        "3c84",
        cat_type="driftscan",
        force_update=True,
        cat_id=list(sma_mir.phase_center_catalog)[0],
    )
    check_str = (
        "   ID     Cat Entry          Type      Az/Lon/RA    El/Lat/Dec  Frame \n"
        "    #          Name                          deg           deg        \n"
        "----------------------------------------------------------------------\n"
        "    1          3c84     driftscan     0:00:00.00  +90:00:00.00  altaz \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


@pytest.mark.filterwarnings("ignore:The provided name")
def test_print_object_unprojected(sma_mir):
    _ = sma_mir._add_phase_center(
        "3c84",
        cat_type="unprojected",
        force_update=True,
        cat_id=list(sma_mir.phase_center_catalog)[0],
    )
    check_str = (
        "   ID     Cat Entry          Type      Az/Lon/RA    El/Lat/Dec  Frame \n"
        "    #          Name                          deg           deg        \n"
        "----------------------------------------------------------------------\n"
        "    1          3c84   unprojected     0:00:00.00  +90:00:00.00  altaz \n"
    )
    table_str = sma_mir.print_phase_center_info(print_table=False, return_str=True)
    assert table_str == check_str


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.filterwarnings("ignore:The provided name")
def test_print_object_multi(carma_miriad):
    """
    Test the print_phase_center_info function when there are multiple objects stored in
    the internal catalog.
    """
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)

    _ = carma_miriad._add_phase_center(
        "NOISE", cat_type="unprojected", force_update=True, cat_id=0
    )
    check_str = (
        "   ID     Cat Entry          Type     Az/Lon/RA    El/Lat/Dec  Frame    Epoch \n"  # noqa
        "    #          Name                       hours           deg                 \n"  # noqa
        "------------------------------------------------------------------------------\n"  # noqa
        "    0         NOISE   unprojected    0:00:00.00  +90:00:00.00  altaz          \n"  # noqa
        "    1         3C273      sidereal   12:29:06.70  + 2:03:08.60    fk5  J2000.0 \n"  # noqa
        "    2      1159+292      sidereal   11:59:31.83  +29:14:43.83    fk5  J2000.0 \n"  # noqa
    )
    table_str = carma_miriad.print_phase_center_info(
        print_table=False, return_str=True, hms_format=True
    )
    assert table_str == check_str

    # check that it works if you specify a list of ids
    table_str = carma_miriad.print_phase_center_info(
        catalog_identifier=[0, 1, 2],
        print_table=False,
        return_str=True,
        hms_format=True,
    )
    assert table_str == check_str


@pytest.mark.parametrize(
    "kwargs,err_type,err_msg",
    [
        [{}, ValueError, "Must specify either phase_dict or cat_name"],
        [{"cat_name": "3c84", "target_cat_id": -1}, ValueError, "No phase center with"],
        [
            {"cat_name": "3c84", "cat_type": "foo"},
            ValueError,
            re.escape(
                "If set, cat_type must be one of ['sidereal', 'ephem', 'unprojected', "
                "'driftscan', 'near_field']"
            ),
        ],
    ],
)
def test_look_in_catalog_err(sma_mir, kwargs, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        utils.phase_center_catalog.look_in_catalog(
            sma_mir.phase_center_catalog, **kwargs
        )


@pytest.mark.parametrize(
    "name,stype,arg_dict,exp_id,exp_diffs",
    (
        ["zenith", None, {}, 0, 4],
        ["zenith", "driftscan", {}, 0, 1],
        ["zenith", "unprojected", {}, 0, 0],
        ["unprojected", "unprojected", {}, None, 99999],
        ["unprojected", "unprojected", {"ignore_name": True}, 0, 0],
        ["zenith", "unprojected", {"lat": 1.0}, 0, 1],
        ["zenith", "unprojected", {"lon": 1.0}, 0, 1],
        ["zenith", "unprojected", {"frame": 1.0}, 0, 1],
        ["zenith", "unprojected", {"epoch": 1.0}, 0, 1],
        ["zenith", "unprojected", {"times": 1.0}, 0, 1],
        ["zenith", "unprojected", {"pm_ra": 1.0}, 0, 1],
        ["zenith", "unprojected", {"pm_dec": 1.0}, 0, 1],
        ["zenith", "unprojected", {"dist": 1.0}, 0, 1],
        ["zenith", "unprojected", {"vrad": 1.0}, 0, 1],
    ),
)
def test_look_in_catalog(hera_uvh5, name, stype, arg_dict, exp_id, exp_diffs):
    """
    Test some basic functions of _look_in_catalog and check that it finds the
    appropriate phase center ID and number of differences between the provided
    parameters and that recorded in the UVData object.
    """
    hera_uvh5.print_phase_center_info()
    [cat_id, num_diffs] = utils.phase_center_catalog.look_in_catalog(
        hera_uvh5.phase_center_catalog,
        cat_name=name,
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
    if cat_id is not None:
        assert cat_id == exp_id
    else:
        assert exp_id is None
    assert num_diffs == exp_diffs


def test_look_in_catalog_phase_dict(sma_mir):
    """
    Use the phase_dict argument for _look_in_catalog and make sure that things
    behave as expected
    """
    # Now try lookup using a dictionary of properties
    assert utils.phase_center_catalog.look_in_catalog(
        sma_mir.phase_center_catalog, cat_name="3c84"
    ) == (1, 5)
    phase_dict = sma_mir.phase_center_catalog[1]
    assert utils.phase_center_catalog.look_in_catalog(
        sma_mir.phase_center_catalog, cat_name="3c84", phase_dict=phase_dict
    ) == (1, 0)

    # Make sure that if we set ignore_name, we still get a match
    assert utils.phase_center_catalog.look_in_catalog(
        sma_mir.phase_center_catalog,
        cat_name="3c84",
        phase_dict=phase_dict,
        ignore_name=True,
    ) == (1, 0)

    # Match w/ a mis-capitalization
    assert utils.phase_center_catalog.look_in_catalog(
        sma_mir.phase_center_catalog,
        cat_name="3C84",
        phase_dict=phase_dict,
        ignore_name=True,
    ) == (1, 0)


@pytest.mark.parametrize(
    "name,stype,arg_dict,msg",
    (
        [-1, "drift", {}, "cat_name must be a string."],
        ["zenith", "drift", {}, "cat_type must be one of"],
        ["zenith", "driftscan", {"pm_ra": 0, "pm_dec": 0}, "Non-zero proper motion"],
        [
            "unprojected",
            "unprojected",
            {"lon": 1},
            "Catalog entries that are unprojected",
        ],
        [
            "unprojected",
            "unprojected",
            {"lat": 1},
            "Catalog entries that are unprojected",
        ],
        [
            "unprojected",
            "unprojected",
            {"frame": "fk5"},
            "cat_frame must be either None",
        ],
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
        [
            "test",
            "sidereal",
            {},
            "cat_lon cannot be None for sidereal or ephem phase centers.",
        ],
        [
            "test",
            "sidereal",
            {"lon": 0},
            "cat_lat cannot be None for sidereal or ephem phase centers.",
        ],
        [
            "test",
            "sidereal",
            {"lon": 0, "lat": 0},
            "cat_frame cannot be None for sidereal or ephem phase centers.",
        ],
        ["unprojected", "unprojected", {"id": 1}, "Provided cat_id belongs to another"],
    ),
)
def test_add_phase_center_arg_errs(sma_mir, name, stype, arg_dict, msg):
    with pytest.raises(ValueError, match=msg):
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
        cat_frame="icrs",
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
        IndexError, match="No source by that ID contained in the catalog."
    ):
        sma_mir._remove_phase_center(-1)


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
@pytest.mark.parametrize("cat_id,new_name", [(1, "foo"), ([1, 2], "foo")])
def test_rename_phase_center_ints(carma_miriad, cat_id, new_name):
    uvd = carma_miriad.copy()
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    uvd.rename_phase_center(cat_id, new_name)


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.parametrize(
    "args,err_type,msg",
    (
        [["abc", "xyz"], ValueError, "No entry by the name abc in the catalog."],
        [["3C273", -2], TypeError, "Value provided to new_name must be a string"],
        [
            [3.1415, "abc"],
            TypeError,
            "catalog_identifier must be a string, an integer or a list of integers.",
        ],
        [
            [["3C273", "unprojected"], "abc"],
            TypeError,
            "catalog_identifier must be a string, an integer or a list of integers.",
        ],
        [
            [["3C273", 1], "abc"],
            TypeError,
            "catalog_identifier must be a string, an integer or a list of integers.",
        ],
        [[-1, None], ValueError, "No entry with the ID -1 in the catalog."],
    ),
)
def test_rename_phase_center_bad_args(carma_miriad, args, err_type, msg):
    """
    Verify that rename_phase_center will throw appropriate errors when supplying
    bad arguments to the method.
    """
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    with pytest.raises(err_type, match=msg):
        carma_miriad.rename_phase_center(*args)


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.parametrize(
    "kwargs,err_type,msg",
    (
        [
            {"catalog_identifier": "abc", "new_name": "xyz", "select_mask": 1},
            ValueError,
            "No catalog entries matching the name abc.",
        ],
        [
            {"catalog_identifier": "3C273", "new_name": -2, "select_mask": 1},
            TypeError,
            "Value provided to new_name must be a string",
        ],
        [
            {"catalog_identifier": "3C273", "new_name": "3c273", "select_mask": 1.5},
            IndexError,
            "select_mask must be an array-like,",
        ],
        [
            {"catalog_identifier": "3C273", "new_name": "3c273", "select_mask": 1},
            ValueError,
            "Data selected with select_mask includes",
        ],
        [
            {"catalog_identifier": -1},
            ValueError,
            "No entry with the ID -1 found in the catalog",
        ],
        [
            {
                "catalog_identifier": 1,
                "new_name": None,
                "select_mask": None,
                "new_id": "hi",
            },
            TypeError,
            "Value provided to new_id must be an int",
        ],
        [
            {
                "catalog_identifier": 1,
                "new_name": None,
                "select_mask": None,
                "new_id": 2,
            },
            ValueError,
            "The ID 2 is already in the catalog",
        ],
        [
            {"catalog_identifier": 35.5},
            TypeError,
            "catalog_identifier must be a string or an integer.",
        ],
    ),
)
def test_split_phase_center_bad_args(carma_miriad, kwargs, err_type, msg):
    """
    Verify that split_phase_center will throw an error if supplied with bad args
    """
    with pytest.raises(err_type, match=msg):
        carma_miriad.split_phase_center(**kwargs)


def test_split_phase_center_err_multiname(carma_miriad):
    """
    Verify that split_phase_center will throw an error if multiple fields
    have the same name in the dataset.
    """
    for key in carma_miriad.phase_center_catalog:
        carma_miriad.phase_center_catalog[key]["cat_name"] = "NOISE"
    with pytest.raises(ValueError, match="The cat_name NOISE has multiple entries in"):
        carma_miriad.split_phase_center("NOISE")


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
@pytest.mark.parametrize(
    "cat_iden,err_type,msg",
    (
        ["dummy1", ValueError, "No entry by the name dummy1 in"],
        [[0, 1, 2], ValueError, "Attributes of phase centers differ"],
        [[-1, -2], ValueError, "No entry with the ID -1 in the catalog."],
        [
            [0, 1.5],
            TypeError,
            (
                "catalog_identifier must be a string, an integer or a list of strings"
                " or integers."
            ),
        ],
    ),
)
def test_merge_phase_centers_bad_args(carma_miriad, cat_iden, err_type, msg):
    """
    Verify that merge_phase_centers will throw an error if supplied with bad args
    """
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    with pytest.raises(err_type, match=msg):
        carma_miriad.merge_phase_centers(cat_iden)


def test_merge_phase_centers_bad_warn(sma_mir):
    with check_warnings(UserWarning, "Selection matches less than two phase"):
        sma_mir.merge_phase_centers("3c84")


@pytest.mark.parametrize(
    "cat_id,new_id,res_id,err_type,msg",
    (
        [-1, -1, 0, ValueError, "No match in catalog to an entry with id -1."],
        [1, 1, [1], ValueError, "Provided cat_id belongs to another source"],
    ),
)
def test_update_id_bad_args(sma_mir, cat_id, new_id, res_id, err_type, msg):
    """
    Verify that _update_phase_center_id throws errors when supplied with bad args
    """
    with pytest.raises(err_type, match=msg):
        sma_mir._update_phase_center_id(cat_id, new_id=new_id, reserved_ids=res_id)


def test_add_clear_phase_center(sma_mir):
    """
    Test that we can add a phase center, verify that we can find it correctly in the
    catalog, and then clear it as its unused.
    """
    check_dict = sma_mir.phase_center_catalog.copy()
    check_id = sma_mir._add_phase_center(
        cat_name="Mars",
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
    assert utils.phase_center_catalog.look_in_catalog(
        sma_mir.phase_center_catalog, cat_name="Mars", cat_lon=[0], cat_lat=[0]
    ) == (0, 7)

    # Finally, clear out the unused entries and check for equivalency w/ the old catalog
    sma_mir._clear_unused_phase_centers()
    assert sma_mir.phase_center_catalog == check_dict


def test_rename_object_capitalization(sma_mir, sma_mir_catalog):
    """
    Verify that renaming works, and that its case sensitive
    """
    # Check and see what happens if we attempt to rename the source
    sma_mir.rename_phase_center("3c84", "3C84")
    assert sma_mir.phase_center_catalog != sma_mir_catalog

    sma_mir.rename_phase_center("3C84", "3c84")
    assert sma_mir.phase_center_catalog == sma_mir_catalog


def test_rename_no_ops(sma_mir, sma_mir_catalog):
    """
    Verify that renaming the phase center with the same name results in no changes
    """
    # Check to make sure that setting the same name doesn't harm anything
    sma_mir.rename_phase_center("3c84", "3c84")
    assert sma_mir.phase_center_catalog == sma_mir_catalog


def test_update_id_no_op(sma_mir, sma_mir_catalog):
    """
    Verify that updating the ID of a source without any ID conflicts results in no
    changes to the catalog
    """
    # This should effectively be a no-op, since the catalog ID of the source isn't
    # being taken up by anything else
    sma_mir._update_phase_center_id(1)
    assert sma_mir.phase_center_catalog == sma_mir_catalog


def test_update_id(sma_mir):
    """
    Verify that calling _update_phase_center_id will produce the lowest available
    positive int as the new ID for the source being updated.
    """
    # If all goes well, this operation should assign the lowest possible integer to the
    # catalog ID of 3c84 -- in this case, 4.
    sma_mir._update_phase_center_id(1, reserved_ids=[0, 1, 2, 3])
    assert list(sma_mir.phase_center_catalog)[0] == 4


@pytest.mark.parametrize(
    "name1,name2,select_mask,msg",
    (
        ["3c84", "3C84", False, "No relevant data selected"],
        ["3c84", "3C84", True, "All data for the source selected"],
    ),
)
def test_split_phase_center_warnings(sma_mir, name1, name2, select_mask, msg):
    # Now let's select no data at all
    with check_warnings(UserWarning, match=msg):
        sma_mir.split_phase_center(name1, new_name=name2, select_mask=select_mask)


def test_split_phase_center(hera_uvh5):
    # Phase the HERA file so that we can play around with it a bit
    hera_uvh5.phase(
        cat_name="3c84",
        lon=Longitude("3:19:48.16", unit="hourangle").rad,
        lat=Latitude("+41:30:42.11", unit="deg").rad,
        phase_frame="fk5",
        epoch="J2000",
    )
    # Alright, now let's actually try to split the sources -- let's say every other
    # integration?
    select_mask = np.isin(hera_uvh5.time_array, np.unique(hera_uvh5.time_array)[::2])

    hera_uvh5.split_phase_center("3c84", new_name="3c84_2", select_mask=select_mask)
    cat_id1 = utils.phase_center_catalog.look_for_name(
        hera_uvh5.phase_center_catalog, "3c84"
    )
    cat_id2 = utils.phase_center_catalog.look_for_name(
        hera_uvh5.phase_center_catalog, "3c84_2"
    )
    # Check that the catalog IDs also line up w/ what we expect
    assert np.all(hera_uvh5.phase_center_id_array[~select_mask] == cat_id1)
    assert np.all(hera_uvh5.phase_center_id_array[select_mask] == cat_id2)
    assert hera_uvh5.Nphase == 2

    cat_id_all = utils.phase_center_catalog.look_for_name(
        hera_uvh5.phase_center_catalog, ["3c84", "3c84_2"]
    )
    assert np.all(np.isin(hera_uvh5.phase_center_id_array, cat_id_all))

    # Make sure the catalog makes sense -- entries should be identical sans cat_id
    temp_cat = hera_uvh5.phase_center_catalog.copy()
    temp_cat[cat_id1[0]]["cat_name"] = "3c84_2"
    assert temp_cat[cat_id1[0]] == temp_cat[cat_id2[0]]


def test_split_phase_center_downselect(hera_uvh5):
    # Phase the HERA file so that we can play around with it a bit
    hera_uvh5.phase(
        cat_name="3c84",
        lon=Longitude("3:19:48.16", unit="hourangle").rad,
        lat=Latitude("+41:30:42.11", unit="deg").rad,
        phase_frame="fk5",
        epoch="J2000",
    )
    catalog_copy = hera_uvh5.phase_center_catalog.copy()

    # Again, only select the first half of the data
    select_mask = np.isin(hera_uvh5.time_array, np.unique(hera_uvh5.time_array)[::2])
    hera_uvh5.split_phase_center("3c84", new_name="3c84_2", select_mask=select_mask)

    # Now effectively rename zenith2 as zenith3 by selecting all data and using
    # the downselect switch
    with check_warnings(UserWarning, "All data for the source selected"):
        hera_uvh5.split_phase_center(
            "3c84_2",
            new_name="3c84_3",
            select_mask=np.arange(hera_uvh5.Nblts),
            downselect=True,
        )

    cat_id1 = utils.phase_center_catalog.look_for_name(
        hera_uvh5.phase_center_catalog, "3c84"
    )
    cat_id3 = utils.phase_center_catalog.look_for_name(
        hera_uvh5.phase_center_catalog, "3c84_3"
    )
    assert np.all(hera_uvh5.phase_center_id_array[~select_mask] == cat_id1)
    assert np.all(hera_uvh5.phase_center_id_array[select_mask] == cat_id3)

    # Make sure the dicts make sense
    temp_dict = hera_uvh5.phase_center_catalog[cat_id1[0]].copy()
    temp_dict["cat_name"] = "3c84_3"
    assert temp_dict == hera_uvh5.phase_center_catalog[cat_id3[0]]

    # Finally, force the two objects back to being one, despite the fact that we've
    # contaminated the dict of one (which will be overwritten by the other)
    hera_uvh5.phase_center_catalog[cat_id3[0]]["cat_epoch"] = 2000.0
    with check_warnings(UserWarning, "Forcing fields together, even though"):
        hera_uvh5.merge_phase_centers(cat_id1 + cat_id3, force_merge=True)

    # We merged everything back together, so we _should_  get back the same
    # thing that we started with.
    assert hera_uvh5.phase_center_catalog == catalog_copy
    assert np.all(
        hera_uvh5.phase_center_id_array
        == utils.phase_center_catalog.look_for_name(
            hera_uvh5.phase_center_catalog, "3c84"
        )
    )


@pytest.mark.parametrize(
    "new_w_vals,old_w_vals,select_mask,err_type,msg",
    [
        [0.0, 0.0, 1.5, IndexError, "select_mask must be an array-like, either of"],
        [[0.0, 0.0], 0.0, [0], IndexError, "The length of new_w_vals is wrong"],
        [0.0, [0.0, 0.0], [0], IndexError, "The length of old_w_vals is wrong"],
    ],
)
def test_apply_w_arg_errs(
    hera_uvh5, new_w_vals, old_w_vals, select_mask, err_type, msg
):
    with pytest.raises(err_type, match=msg):
        hera_uvh5._apply_w_proj(
            new_w_vals=new_w_vals, old_w_vals=old_w_vals, select_mask=select_mask
        )


def test_apply_w_no_ops(hera_uvh5):
    """
    Test to make sure that the _apply_w method throws  expected errors
    """
    hera_copy = hera_uvh5.copy()

    # Test to make sure that the following gives us back the same results,
    # first without a selection mask
    hera_uvh5._apply_w_proj(new_w_vals=0.0, old_w_vals=0.0)
    assert hera_uvh5 == hera_copy

    # And now with a selection mask applied
    hera_uvh5._apply_w_proj(
        new_w_vals=np.arange(hera_uvh5.Nblts),
        old_w_vals=np.arange(hera_uvh5.Nblts),
        select_mask=[0, 1],
    )
    assert hera_uvh5 == hera_copy


@pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad file,")
def test_phase_dict_helper_err_multi_match(carma_miriad):
    """
    Verify that _phase_dict_helper will throw an error if multiple fields
    have the same name in the dataset when using lookup.
    """
    for key in carma_miriad.phase_center_catalog:
        carma_miriad.phase_center_catalog[key]["cat_name"] = "NOISE"

    with pytest.raises(ValueError, match="Name of object has multiple matches in "):
        carma_miriad._phase_dict_helper(
            lon=None,
            lat=None,
            epoch=None,
            phase_frame=None,
            ephem_times=None,
            cat_type=None,
            pm_ra=None,
            pm_dec=None,
            dist=None,
            vrad=None,
            cat_name="NOISE",
            lookup_name=True,
            time_array=None,
        )


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
            lon=dummy_phase_dict["cat_lon"],
            lat=dummy_phase_dict["cat_lat"],
            epoch=dummy_phase_dict["cat_epoch"],
            phase_frame=dummy_phase_dict["cat_frame"],
            ephem_times=dummy_phase_dict["cat_times"],
            cat_type=dummy_phase_dict["cat_type"],
            pm_ra=dummy_phase_dict["cat_pm_ra"],
            pm_dec=dummy_phase_dict["cat_pm_dec"],
            dist=dummy_phase_dict["cat_dist"],
            vrad=dummy_phase_dict["cat_vrad"],
            cat_name=dummy_phase_dict["cat_name"],
            lookup_name=False,
            time_array=None,
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
    ],
)
def test_phase_dict_helper_errs(sma_mir, arg_dict, dummy_phase_dict, msg):
    """
    Test the `_phase_dict_helper` method.

    Test the helper function that the `phase` method uses for looking up astronomical
    source information.
    """
    pytest.importorskip("astroquery")

    from ssl import SSLError

    from requests import RequestException

    for key in dummy_phase_dict:
        if key not in arg_dict:
            arg_dict[key] = dummy_phase_dict[key]

    # We have to handle this piece a bit carefully, since some queries fail due to
    # intermittent failures connecting to the JPL-Horizons service.
    with pytest.raises(Exception) as cm:
        sma_mir._phase_dict_helper(
            lon=arg_dict["cat_lon"],
            lat=arg_dict["cat_lat"],
            epoch=arg_dict["cat_epoch"],
            phase_frame=arg_dict["cat_frame"],
            ephem_times=arg_dict["cat_times"],
            cat_type=arg_dict["cat_type"],
            pm_ra=arg_dict["cat_pm_ra"],
            pm_dec=arg_dict["cat_pm_dec"],
            dist=arg_dict["cat_dist"],
            vrad=arg_dict["cat_vrad"],
            cat_name=arg_dict["cat_name"],
            lookup_name=arg_dict.get("lookup"),
            time_array=arg_dict.get("time_arr"),
        )

    if issubclass(cm.type, RequestException) or issubclass(cm.type, SSLError):
        pytest.skip("SSL/Connection error w/ JPL Horizons")

    assert issubclass(cm.type, ValueError)
    assert str(cm.value).startswith(msg)


def test_phase_dict_helper_sidereal_lookup(sma_mir, dummy_phase_dict):
    """
    Check that we can use the lookup option to find a sidereal source properties in
    a multi-phase-ctr dataset.
    """
    phase_dict = sma_mir._phase_dict_helper(
        lon=dummy_phase_dict["cat_lon"],
        lat=dummy_phase_dict["cat_lat"],
        epoch=dummy_phase_dict["cat_epoch"],
        phase_frame=dummy_phase_dict["cat_frame"],
        ephem_times=dummy_phase_dict["cat_times"],
        cat_type=dummy_phase_dict["cat_type"],
        pm_ra=dummy_phase_dict["cat_pm_ra"],
        pm_dec=dummy_phase_dict["cat_pm_dec"],
        dist=dummy_phase_dict["cat_dist"],
        vrad=dummy_phase_dict["cat_vrad"],
        cat_name="3c84",
        lookup_name=True,
        time_array=None,
    )
    assert (
        phase_dict.pop("cat_id")
        == utils.phase_center_catalog.look_for_name(
            sma_mir.phase_center_catalog, "3c84"
        )[0]
    )
    assert (
        phase_dict
        == sma_mir.phase_center_catalog[
            utils.phase_center_catalog.look_for_name(
                sma_mir.phase_center_catalog, "3c84"
            )[0]
        ]
    )


def test_phase_dict_helper_jpl_lookup_existing(sma_mir):
    """
    Verify that the _phase_dict_helper function correctly hands back a dict that
    matches that in the catalog, provided the source properties match.
    """
    # Finally, check that we get a good result if feeding the same values, even if not
    # actually performing a lookup
    cat_id = utils.phase_center_catalog.look_for_name(
        sma_mir.phase_center_catalog, "3c84"
    )[0]
    phase_dict = sma_mir._phase_dict_helper(
        lon=sma_mir.phase_center_catalog[cat_id].get("cat_lon"),
        lat=sma_mir.phase_center_catalog[cat_id].get("cat_lat"),
        epoch=sma_mir.phase_center_catalog[cat_id].get("cat_epoch"),
        phase_frame=sma_mir.phase_center_catalog[cat_id].get("cat_frame"),
        ephem_times=sma_mir.phase_center_catalog[cat_id].get("cat_times"),
        cat_type=sma_mir.phase_center_catalog[cat_id].get("cat_type"),
        pm_ra=sma_mir.phase_center_catalog[cat_id].get("cat_pm_ra"),
        pm_dec=sma_mir.phase_center_catalog[cat_id].get("cat_pm_dec"),
        dist=sma_mir.phase_center_catalog[cat_id].get("cat_dist"),
        vrad=sma_mir.phase_center_catalog[cat_id].get("cat_vrad"),
        cat_name="3c84",
        lookup_name=False,
        time_array=sma_mir.time_array,
    )
    assert phase_dict.pop("cat_id") == cat_id
    assert (
        phase_dict
        == sma_mir.phase_center_catalog[
            utils.phase_center_catalog.look_for_name(
                sma_mir.phase_center_catalog, "3c84"
            )[0]
        ]
    )


def test_phase_dict_helper_jpl_lookup_append(sma_mir):
    """
    Test _phase_dict_helper to see if it will correctly call the JPL lookup when
    an old ephem does not cover the newly requested time range
    """
    pytest.importorskip("astroquery")

    from ssl import SSLError

    from requests import RequestException

    # Now see what happens if we attempt to lookup something that JPL actually knows
    obs_time = np.array(2456789.0)

    # Handle this part with care, since we don't want the test to fail if we are unable
    # to reach the JPL-Horizons service.
    try:
        phase_dict = sma_mir._phase_dict_helper(
            lon=0,
            lat=0,
            epoch=None,
            phase_frame=None,
            ephem_times=None,
            cat_type=None,
            pm_ra=0,
            pm_dec=0,
            dist=0,
            vrad=0,
            cat_name="Mars",
            lookup_name=True,
            time_array=obs_time,
        )
    except (SSLError, RequestException) as err:
        pytest.skip("SSL/Connection error w/ JPL Horizons: " + str(err))

    cat_id = sma_mir._add_phase_center(
        phase_dict["cat_name"],
        cat_type=phase_dict["cat_type"],
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

    # Again, just skip if we are unable to reach the JPL-Horizons
    try:
        phase_dict = sma_mir._phase_dict_helper(
            lon=0,
            lat=0,
            epoch=None,
            phase_frame=None,
            ephem_times=None,
            cat_type=None,
            pm_ra=0,
            pm_dec=0,
            dist=0,
            vrad=0,
            cat_name="Mars",
            lookup_name=True,
            time_array=obs_time,
        )
    except (SSLError, RequestException) as err:
        pytest.skip("SSL/Connection error w/ JPL Horizons: " + str(err))

    # Previously, everything else will have had a single point, but the new ephem (which
    # covers 36 hours at 3 hour intervals) should have a lucky total of 13 points.
    keycheck = ["cat_lon", "cat_lat", "cat_vrad", "cat_dist", "cat_times"]
    for key in keycheck:
        assert len(phase_dict[key]) == 13


@pytest.mark.parametrize("use_ant_pos", [True, False])
@pytest.mark.parametrize("phase_frame", ["icrs", "gcrs"])
@pytest.mark.parametrize("file_type", ["uvh5", "uvfits", "miriad"])
def test_fix_phase(hera_uvh5, tmp_path, use_ant_pos, phase_frame, file_type):
    """
    Test the phase fixing method fix_phase
    """
    if file_type == "miriad" and not hasmiriad:
        pytest.skip("MIRIAD not installed.")

    # Make some copies of the data
    uv_in = hera_uvh5
    uv_in_bad = uv_in.copy()

    del uv_in_bad.phase_center_catalog[0]
    uv_in_bad.phase_center_catalog[1] = {
        "cat_dist": None,
        "cat_epoch": np.float64(2000.0),
        "cat_frame": phase_frame,
        "cat_lat": np.float64(-0.17855186342047605),
        "cat_lon": np.float64(3.502185879515176),
        "cat_name": "foo",
        "cat_pm_dec": None,
        "cat_pm_ra": None,
        "cat_times": None,
        "cat_type": "sidereal",
        "cat_vrad": None,
        "info_source": "user",
    }
    uv_in_bad.phase_center_id_array[:] = 1
    uv_in_bad._set_app_coords_helper()

    if use_ant_pos:
        uvw_path = f"oldproj_antpos_{phase_frame}_uvw.npy"
    else:
        uvw_path = f"oldproj_{phase_frame}_uvw.npy"

    uv_in_bad.uvw_array = np.load(os.path.join(DATA_PATH, uvw_path))
    uv_in_bad._apply_w_proj(
        new_w_vals=uv_in_bad.uvw_array[:, -1],
        old_w_vals=0.0,
        select_mask=(uv_in_bad.ant_1_array != uv_in_bad.ant_2_array),
    )

    # These values could be anything -- we're just picking something that we know should
    # be visible from the telescope at the time of obs (ignoring horizon limits).
    phase_ra = uv_in.lst_array[-1]
    phase_dec = uv_in.telescope.location.lat.rad * 0.333

    # Do the improved phasing on the data set.
    uv_in.phase(lon=phase_ra, lat=phase_dec, phase_frame=phase_frame, cat_name="foo")

    if use_ant_pos:
        warn_msg = ["Fixing phases using antenna positions."]
    else:
        warn_msg = ["Attempting to fix residual phasing errors from the old `phase`"]

    read_warn_msg = copy.deepcopy(warn_msg)
    read_warn_type = [UserWarning]

    uv_in_bad_copy = uv_in_bad.copy()
    if file_type == "uvh5":
        outfile = os.path.join(tmp_path, "test_bad_phase.uvh5")
        uv_in_bad_copy.write_uvh5(outfile)
    elif file_type == "uvfits":
        outfile = os.path.join(tmp_path, "test_bad_phase.uvfits")
        uv_in_bad_copy.write_uvfits(outfile)
    elif file_type == "miriad":
        outfile = os.path.join(tmp_path, "test_bad_phase.uv")
        with check_warnings(
            UserWarning,
            "writing default values for restfreq, vsource, veldop, jyperk, and systemp",
        ):
            uv_in_bad_copy.write_miriad(outfile, clobber=True)

    with check_warnings(read_warn_type, match=read_warn_msg), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Fixing auto-correlations to be be real-only")
        uv_in_bad2 = UVData.from_file(
            outfile, fix_old_proj=True, fix_use_ant_pos=use_ant_pos
        )

    with check_warnings(UserWarning, match=warn_msg):
        uv_in_bad.fix_phase(use_ant_pos=use_ant_pos)

    uv_in_bad_copy = uv_in_bad.copy()
    if file_type == "miriad":
        uv_in_bad_copy.conjugate_bls("ant1<ant2")
        uv_in_bad_copy.reorder_blts()
        uv_in_bad2.reorder_blts()
        uv_in_bad2._update_phase_center_id(0, new_id=1)
    elif file_type == "uvfits":
        uv_in_bad2.dut1 = None
        uv_in_bad2.earth_omega = None
        uv_in_bad2.gst0 = None
        uv_in_bad2.rdate = None
        uv_in_bad2.timesys = None
    uv_in_bad2.phase_center_catalog[1]["info_source"] = (
        uv_in_bad_copy.phase_center_catalog[1]["info_source"]
    )
    uv_in_bad2.extra_keywords = uv_in_bad_copy.extra_keywords

    # We have to handle this case a little carefully, because since the old
    # unproject_phase was _mostly_ accurate, although it does seem to intoduce errors
    # on the order of a part in 1e5, which translates to about a tenth of a degree phase
    # error in the test data set used here. Check that first, make sure it's good
    np.testing.assert_allclose(uv_in.data_array, uv_in_bad.data_array, rtol=3e-4)
    np.testing.assert_allclose(
        uv_in_bad2.data_array, uv_in_bad_copy.data_array, rtol=3e-4
    )

    # Once we know the data are okay, copy over data array and check for equality btw
    # the other attributes of the two objects.
    uv_in_bad.data_array = uv_in.data_array
    uv_in_bad2.data_array = uv_in_bad_copy.data_array
    uv_in_bad.history = uv_in.history
    assert uv_in == uv_in_bad
    assert uv_in_bad2 == uv_in_bad_copy


def test_fix_phase_error(hera_uvh5):
    uv_in = hera_uvh5
    with pytest.raises(
        ValueError, match="Data are unprojected, no phase fixing required."
    ):
        uv_in.fix_phase()
    uv_in.phase(
        lon=0, lat=np.pi / 2, cat_type="driftscan", phase_frame="altaz", cat_name="foo"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Objects with driftscan phase centers were not phased with the old "
            "method, so no fixing is required."
        ),
    ):
        uv_in.fix_phase()


def test_multi_file_ignore_name(hera_uvh5_split):
    """
    Verify that if phased two objects to the same position with different names, we
    can successfully use the "ignore_name" switch in the add operation to allow
    the two objects to be combined.
    """
    uv1, uv2, uvfull = hera_uvh5_split

    # Phase both targets to the same position with different names
    uv1.phase(lon=3.6, lat=-0.5, cat_name="target1")
    uv2.phase(lon=3.6, lat=-0.5, cat_name="target2")
    uvfull.phase(lon=3.6, lat=-0.5, cat_name="target1")

    # Check that you end up with two phase centers if you don't ignore the name
    uv3 = uv1 + uv2
    assert uv3.Nphase == 2

    # Check that you only end up with one phase centers if you ignore the name
    uv3 = uv1.__add__(uv2, ignore_name=True, inplace=False)
    assert uv3.Nphase == 1
    # The reorders here are neccessary after the add to make sure that the baseline
    # ordering is consistent between the objects
    uv3.reorder_blts()
    uvfull.reorder_blts()

    # Make sure that after the add, everything agrees
    assert uvfull.history in uv3.history
    uvfull.history = uv3.history

    uv3._consolidate_phase_center_catalogs(
        reference_catalog=uvfull.phase_center_catalog, ignore_name=True
    )


@pytest.mark.parametrize("test_op", [None, "split", "rename", "merge", "r+m"])
def test_multi_phase_split_merge_rename(hera_uvh5_split, test_op):
    """
    Test the split, merge, and rename operations, and make sure their operations
    are internally consistent.
    """
    uv1, uv2, uvfull = hera_uvh5_split
    half_mask = np.arange(uvfull.Nblts) < (uvfull.Nblts * 0.5)

    uv1.phase(lon=3.6, lat=-0.5, cat_name="target1")
    uv2.phase(lon=3.6, lat=-0.5, cat_name="target1" if (test_op is None) else "target2")
    uv3 = uv1 + uv2
    uv3.reorder_blts()

    uvfull.reorder_blts()
    uvfull.phase(lon=3.6, lat=-0.5, cat_name="target1")
    uvfull._update_phase_center_id(
        list(uvfull.phase_center_catalog)[0], new_id=1 if (test_op is None) else 0
    )

    # Any of these operations should allow for the objects to become equal to the
    # other -- they're basically the inverse action taken on two different objects.
    if test_op is None:
        # Nothing to do here -- this should be an apples-to-apples comparison without
        # any renaming operations.
        pass
    if test_op == "split":
        uvfull.split_phase_center(
            select_mask=~half_mask, catalog_identifier=0, new_id=1, new_name="target2"
        )
    elif test_op == "rename":
        uv3.merge_phase_centers(list(uv3.phase_center_catalog)[::-1], ignore_name=True)
        uvfull.rename_phase_center("target1", "target2")
        uvfull._update_phase_center_id(0, new_id=1)
    elif test_op == "merge":
        uv3.merge_phase_centers(["target1", "target2"], ignore_name=True)
    elif test_op == "r+m":
        uv3.rename_phase_center("target2", "target1")
        uv3.merge_phase_centers("target1")

    assert uvfull.history in uv3.history
    uvfull.history = uv3.history
    assert uvfull == uv3


@pytest.mark.parametrize("catid", [0, 1])
def test_multi_phase_add(hera_uvh5_split, catid):
    uv1, uv2, uvfull = hera_uvh5_split

    # Give it a new name, and then rephase half of the "full" object
    uv1.phase(lon=3.6, lat=-0.5, cat_name="target1")
    uv2.phase(lon=-0.5, lat=3.6, cat_name="target2")

    # Test that addition handles cat ID collisions correctly
    for pc_id in list(uv2.phase_center_catalog):
        if uv2.phase_center_catalog[pc_id]["cat_name"] == "target2":
            uv2._update_phase_center_id(pc_id, new_id=catid)

    # Add the objects together
    uv3 = uv1.__add__(uv2)
    uv3.reorder_blts()

    # Separately phase both halves of the full data set
    half_mask = np.arange(uvfull.Nblts) < (uvfull.Nblts * 0.5)
    uvfull.phase(lon=-0.5, lat=3.6, cat_name="target2", select_mask=~half_mask)
    uvfull.phase(lon=3.6, lat=-0.5, cat_name="target1", select_mask=half_mask)
    uvfull.reorder_blts()

    # Check that the histories line up
    assert uvfull.history in uv3.history
    uvfull.history = uv3.history

    # By construct, we've made it so that the cat IDs don't line up, but everything
    # else should. Make sure the IDs are different, but contain the same entries/names
    # for the phase centers
    assert np.any(uv3.phase_center_id_array != uvfull.phase_center_id_array)
    assert uv3.phase_center_catalog != uvfull.phase_center_catalog

    # Update the Obs IDs, and make sure that _now_ the objects are equal
    name_map1 = {
        pc_dict["cat_name"]: pc_id
        for pc_id, pc_dict in uv3.phase_center_catalog.items()
    }
    name_map2 = {
        pc_dict["cat_name"]: pc_id
        for pc_id, pc_dict in uvfull.phase_center_catalog.items()
    }

    uv3._update_phase_center_id(name_map1["target1"], new_id=100)
    uv3._update_phase_center_id(name_map1["target2"], new_id=101)
    uv3._update_phase_center_id(100, new_id=name_map2["target1"])
    uv3._update_phase_center_id(101, new_id=name_map2["target2"])

    assert uv3 == uvfull


@pytest.mark.parametrize("cat_type", ["sidereal", "ephem", "driftscan"])
def test_multi_phase_downselect(hera_uvh5_split, cat_type):
    """
    Verify that we can create the same UVData object if we phase then downselect
    vs downselect and phase when working with a multi-phase-ctr object.
    """
    if cat_type == "ephem":
        pytest.importorskip("astroquery")

    uv1, uv2, uvfull = hera_uvh5_split

    # get the halves of the full data set
    half_mask = np.arange(uvfull.Nblts) < (uvfull.Nblts * 0.5)
    unique_times = np.unique(uvfull.time_array)

    # Give it a new name, and then rephase half of the "full" object
    if cat_type == "sidereal":
        uv1.phase(lon=3.6, lat=-0.5, cat_name="target1")
        uv2.phase(lon=-0.5, lat=3.6, cat_name="target2")
        uvfull.phase(lon=-0.5, lat=3.6, cat_name="target2", select_mask=~half_mask)
        uvfull.phase(lon=3.6, lat=-0.5, cat_name="target1", select_mask=half_mask)
    elif cat_type == "ephem":
        from ssl import SSLError

        from requests import RequestException

        try:
            uv1.phase(ra=0, dec=0, epoch="J2000", lookup_name="Mars", cat_name="Mars")
            uv2.phase(
                ra=0, dec=0, epoch="J2000", lookup_name="Jupiter", cat_name="Jupiter"
            )
            uvfull.phase(
                lon=0,
                lat=0,
                lookup_name="Jupiter",
                cat_name="Jupiter",
                select_mask=~half_mask,
            )
            uvfull.phase(
                lon=0, lat=0, lookup_name="Mars", cat_name="Mars", select_mask=half_mask
            )
        except (SSLError, RequestException) as err:
            pytest.skip("SSL/Connection error w/ JPL Horizons: " + str(err))
    elif cat_type == "driftscan":
        uv1.phase(
            lon=3.6, lat=-0.5, cat_type=cat_type, phase_frame=None, cat_name="drift1"
        )
        uv2.phase(
            lon=-0.5, lat=3.6, cat_type=cat_type, phase_frame="altaz", cat_name="drift2"
        )
        uvfull.phase(
            lon=-0.5,
            lat=3.6,
            cat_type=cat_type,
            cat_name="drift2",
            phase_frame=None,
            select_mask=~half_mask,
        )
        uvfull.phase(
            lon=3.6,
            lat=-0.5,
            cat_type=cat_type,
            cat_name="drift1",
            phase_frame="altaz",
            select_mask=half_mask,
        )

    uv1.reorder_blts()
    uv2.reorder_blts()

    for mask, uvdata in zip(
        [np.arange(10), np.arange(10, 20)], [uv1, uv2], strict=True
    ):
        uvtemp = uvfull.select(times=unique_times[mask], inplace=False)
        uvtemp.reorder_blts()
        # Select does not clear the catalog, so clear the unused source and
        # update the cat ID so that it matches with the indv datasets
        uvtemp._clear_unused_phase_centers()
        uvtemp._update_phase_center_id(
            list(uvtemp.phase_center_catalog.keys())[0], new_id=1
        )
        assert uvtemp.history in uvdata.history
        uvtemp.history = uvdata.history
        assert uvtemp == uvdata


@pytest.mark.filterwarnings("ignore:Unknown phase types are no longer supported")
def test_eq_allowed_failures(bda_test_file, capsys):
    """
    Test that the allowed_failures keyword on the __eq__ method works as intended.
    """
    uv1 = bda_test_file
    uv2 = uv1.copy()

    # adjust optional parameters to be different
    uv1.extra_keywords = {"foo": 2}
    uv2.extra_keywords = {"foo": 4}
    assert uv1.__eq__(uv2, check_extra=True, allowed_failures=["extra_keywords"])
    captured = capsys.readouterr()
    assert (
        captured.out == "extra_keywords parameter is a dict, key foo is not equal\n"
        "parameter _extra_keywords does not match, but is not required to for "
        "equality. Left is {'foo': 2}, right is {'foo': 4}.\n"
    )

    # make sure that objects are not equal without specifying allowed_failures
    assert uv1 != uv2

    return


@pytest.mark.filterwarnings("ignore:Unknown phase types are no longer supported")
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
    assert (
        captured.out
        == "filename parameter value is a list of strings, values are different\n"
        "parameter _filename does not match, but is not required to for equality. "
        "Left is ['foo.uvh5'], right is ['bar.uvh5'].\n"
    )

    return


@pytest.mark.filterwarnings("ignore:Unknown phase types are no longer supported")
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
    assert (
        captured.out
        == "filename parameter value is a list of strings, values are different\n"
        "parameter _filename does not match, but is not required to for equality. "
        "Left is ['foo.uvh5'], right is ['bar.uvh5'].\n"
    )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_data(hera_uvh5):
    """
    Test setting data for a given baseline.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]

    data = 2 * uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    uv.set_data(data, ant1, ant2)
    data2 = uv.get_data(ant1, ant2, squeeze="none")

    np.testing.assert_allclose(
        data, data2, rtol=uv._data_array.tols[0], atol=uv._data_array.tols[1]
    )
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_data_evla():
    """
    Test setting data for a given baseline on a different test file.
    """
    filename = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")
    uv = UVData()
    uv.read(filename)

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
    data = 2 * uv.get_data(ant1, ant2, squeeze="none", force_copy=True)
    inds1, inds2, indp = uv._key2inds((ant1, ant2))
    uv.set_data(data, ant1, ant2)
    data2 = uv.get_data(ant1, ant2, squeeze="none")

    np.testing.assert_allclose(
        data, data2, rtol=uv._data_array.tols[0], atol=uv._data_array.tols[1]
    )
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_data_polkey(hera_uvh5):
    """
    Test setting data for a given baseline with a specific polarization.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
    pol = "xx"
    data = 2 * uv.get_data(ant1, ant2, pol, squeeze="none", force_copy=True)
    inds1, inds2, indp = uv._key2inds((ant1, ant2, pol))
    uv.set_data(data, ant1, ant2, pol)
    data2 = uv.get_data(ant1, ant2, pol, squeeze="none")

    np.testing.assert_allclose(
        data, data2, rtol=uv._data_array.tols[0], atol=uv._data_array.tols[1]
    )
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_flags(hera_uvh5):
    """
    Test setting flags for a given baseline.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
    flags = uv.get_flags(ant1, ant2, squeeze="none", force_copy=True)
    flags[:] = True
    _ = uv._key2inds((ant1, ant2))
    uv.set_flags(flags, ant1, ant2)
    flags2 = uv.get_flags(ant1, ant2, squeeze="none")

    np.testing.assert_allclose(flags, flags2)
    assert not np.allclose(uv.flag_array, True)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_flags_polkey(hera_uvh5):
    """
    Test setting flags for a given baseline with a specific polarization.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
    pol = "xx"
    flags = uv.get_flags(ant1, ant2, pol, squeeze="none", force_copy=True)
    flags[:] = True
    _ = uv._key2inds((ant1, ant2, pol))
    uv.set_flags(flags, ant1, ant2, pol)
    flags2 = uv.get_flags(ant1, ant2, pol, squeeze="none")

    np.testing.assert_allclose(flags, flags2)
    assert not np.allclose(uv.flag_array, True)
    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_nsamples(hera_uvh5):
    """
    Test setting nsamples for a given baseline.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
    nsamples = uv.get_nsamples(ant1, ant2, squeeze="none", force_copy=True)
    nsamples[:] = np.pi
    _ = uv._key2inds((ant1, ant2))
    uv.set_nsamples(nsamples, ant1, ant2)
    nsamples2 = uv.get_nsamples(ant1, ant2, squeeze="none")

    np.testing.assert_allclose(
        nsamples,
        nsamples2,
        rtol=uv._nsample_array.tols[0],
        atol=uv._nsample_array.tols[1],
    )
    assert not np.allclose(uv.nsample_array, np.pi)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_nsamples_polkey(hera_uvh5):
    """
    Test setting nsamples for a given baseline with a specific polarization.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
    pol = "xx"
    nsamples = uv.get_nsamples(ant1, ant2, pol, squeeze="none", force_copy=True)
    nsamples[:] = np.pi
    _ = uv._key2inds((ant1, ant2, pol))
    uv.set_nsamples(nsamples, ant1, ant2, pol)
    nsamples2 = uv.get_nsamples(ant1, ant2, pol, squeeze="none")

    np.testing.assert_allclose(
        nsamples,
        nsamples2,
        rtol=uv._nsample_array.tols[0],
        atol=uv._nsample_array.tols[1],
    )
    assert not np.allclose(uv.nsample_array, np.pi)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_set_data_bad_key_error(hera_uvh5):
    """
    Test an error is raised when a key has too many values.
    """
    uv = hera_uvh5

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
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

    ant1 = np.unique(uv.telescope.antenna_numbers)[0]
    ant2 = np.unique(uv.telescope.antenna_numbers)[1]
    nsamples = uv.get_nsamples(ant1, ant2, squeeze="none", force_copy=True)
    # make data the wrong rank
    nsamples = nsamples[0]
    match = "the input array is not compatible with the shape of the destination"
    with pytest.raises(ValueError, match=match):
        uv.set_nsamples(nsamples, (ant1, ant2))

    return


@pytest.mark.parametrize(
    ["filename", "msg"],
    [
        ["zen.2458661.23480.HH.uvh5", ""],
        [
            "sma_test.mir",
            [
                (
                    "The lst_array is not self-consistent with the time_array and "
                    "telescope location. Consider recomputing with the "
                    "`set_lsts_from_time_array` method"
                ),
                (
                    "> 25 ms errors detected reading in LST values from MIR data. "
                    "This typically signifies a minor metadata recording error (which "
                    "can be mitigated by calling the `set_lsts_from_time_array` method "
                    "with `update_vis=False`), though additional errors about "
                    "uvw-position accuracy may signal more significant issues with "
                    "metadata accuracy that could have substantial impact on "
                    "downstream analysis."
                ),
            ],
        ],
        [
            "carma_miriad",
            [
                (
                    "Altitude is not present in Miriad file, using known location"
                    " values for SZA."
                ),
                (
                    "The uvw_array does not match the expected values given the antenna"
                    " positions."
                ),
            ],
        ],
        [
            "1133866760.uvfits",
            [
                (
                    "Fixing auto-correlations to be be real-only, after some imaginary"
                    " values were detected in data_array."
                )
            ],
        ],
        [
            "fhd",
            [
                "Telescope location derived from obs lat/lon/alt values does not match "
                "the location in the layout file."
            ],
        ],
    ],
)
def test_from_file(filename, msg, fhd_data_files):
    kwargs = {}
    if "miriad" in filename:
        pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    elif "fhd" in filename:
        filename = fhd_data_files["filename"]
        kwargs = fhd_data_files
        del kwargs["filename"]

    if isinstance(filename, str):
        testfile = os.path.join(DATA_PATH, filename)
    else:
        testfile = filename
    uvd = UVData()

    if len(msg) == 0:
        warn = None
    else:
        warn = UserWarning

    with check_warnings(warn, match=msg):
        uvd.read(testfile, **kwargs)
    with check_warnings(warn, match=msg):
        uvd2 = UVData.from_file(testfile, **kwargs)
    assert uvd == uvd2


@pytest.mark.parametrize("add_type", ["blt", "freq", "pol"])
@pytest.mark.parametrize("sort_type", ["blt", "freq", "pol"])
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_add_pol_sorting_bl(casa_uvfits, add_type, sort_type):
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
        uv1.reorder_blts("time", minor_order="ant1")
        uv2.reorder_blts("time", minor_order="ant2")
        casa_uvfits.reorder_blts("bda")
        order_check = uv1.ant_1_array == uv2.ant_1_array
    elif sort_type == "freq":
        uv1.reorder_freqs(channel_order="freq")
        uv2.reorder_freqs(channel_order="-freq")
        casa_uvfits.reorder_freqs(spw_order="freq")
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


@pytest.mark.parametrize(
    "pol_sel,flex_err,err_msg",
    [
        [None, True, "Npols must be equal to 1 if flex_spw_polarization_array is set."],
        [-5, True, "polarization_array must all be equal to 0 if flex_spw_pol"],
        [None, False, "polarization_array may not be equal to 0 if flex_spw_pol"],
    ],
)
def test_flex_pol_check_errs(sma_mir, pol_sel, flex_err, err_msg):
    """
    Test that appropriate errors are thrown when flex_spw_polarization_array is set
    incorrectly.
    """
    sma_mir.select(polarizations=pol_sel)
    if flex_err:
        sma_mir.flex_spw_polarization_array = sma_mir.spw_array
    else:
        sma_mir.polarization_array[0] = 0

    with pytest.raises(ValueError, match=err_msg):
        sma_mir.check()


def test_convert_remove_flex_pol_error(uv_phase_comp):
    uvd, _ = uv_phase_comp

    uvd2 = uvd.copy()
    uvd2.convert_to_flex_pol()

    with pytest.raises(ValueError, match="This is already a flex-pol object"):
        uvd2.convert_to_flex_pol()

    uvd2.flex_spw_polarization_array[1] = uvd2.flex_spw_polarization_array[0]
    with pytest.raises(
        ValueError,
        match=(
            "Some spectral windows have identical frequencies, "
            "channel widths and polarizations, so spws cannot be "
            "combined. Set combine_spws=False to avoid this error."
        ),
    ):
        uvd2.remove_flex_pol()


def test_sma_make_remove_flex_pol_no_op(sma_mir):
    """
    Test shortcircuits of _make_flex_pol & remove_flex_pol method
    """
    uvd = sma_mir.copy()

    # remove_flex_pol no op
    uvd.select(polarizations=["xx"])
    sma_copy = uvd.copy()
    uvd.remove_flex_pol()

    assert uvd == sma_copy

    # _make_flex_pol no op
    uvd.flag_array[:, : sma_mir.Nfreqs // 2, 0] = True
    uvd._make_flex_pol()
    sma_copy = uvd.copy()

    uvd._make_flex_pol()

    assert uvd == sma_copy


def test_remove_flex_pol_no_op_multiple_spws(uv_phase_comp):
    """
    Test shortcircuits of flex_pol method
    """
    # remove_flex_pol with multiple spws but only one pol
    uvd, _ = uv_phase_comp

    uvd2 = uvd.copy()
    uvd2.channel_width = np.full(uvd2.Nfreqs, uvd2.channel_width)
    uvd2.spw_array = np.array([1, 4, 5])
    uvd2.Nspws = 3
    uvd2.flex_spw_id_array = np.zeros(uvd2.Nfreqs, dtype=int)
    uvd2.flex_spw_id_array[: uvd2.Nfreqs // 3] = 1
    uvd2.flex_spw_id_array[uvd2.Nfreqs // 3 : 2 * (uvd2.Nfreqs // 3)] = 4
    uvd2.flex_spw_id_array[2 * (uvd2.Nfreqs // 3) :] = 5
    uvd2.check()
    uvd3 = uvd2.copy()

    uvd2.convert_to_flex_pol()
    uvd2.select(polarizations=["xx"])
    uvd2.remove_flex_pol()

    uvd3.select(polarizations=["xx"])
    assert uvd2 == uvd3


@pytest.mark.parametrize(
    "multi_spw,sorting",
    [
        [False, None],
        [True, None],
        [False, "channel"],
        [True, "channel"],
        [True, "spw1"],
        [True, "spw2"],
    ],
)
def test_flex_pol_uvh5(multi_spw, sorting, uv_phase_comp, tmp_path):
    """
    Check that we can write out uvh5 files with flex pol data sets.

    This exercises `convert_to_flex_pol` and `remove_flex_pol` and the check_autos for
    flex_pol.
    """

    uvd, _ = uv_phase_comp

    assert uvd.Npols > 1

    if multi_spw:
        # split data into multiple spws
        uvd.spw_array = np.array([1, 4, 5])
        uvd.Nspws = 3
        uvd.flex_spw_id_array = np.zeros(uvd.Nfreqs, dtype=int)
        uvd.flex_spw_id_array[: uvd.Nfreqs // 3] = 1
        uvd.flex_spw_id_array[uvd.Nfreqs // 3 : 2 * (uvd.Nfreqs // 3)] = 4
        with pytest.raises(
            ValueError,
            match="All values in the flex_spw_id_array must exist in the spw_array.",
        ):
            uvd.check()
        uvd.flex_spw_id_array[2 * (uvd.Nfreqs // 3) :] = 5

    uvd_orig = uvd.copy()

    # make a copy and reshape improperly to trigger the check_autos code
    uvd2 = uvd.copy(metadata_only=True)
    uvd2.convert_to_flex_pol()
    uvd2.check(check_autos=True)
    uvd2.data_array = uvd.data_array.reshape(uvd.Nblts, uvd.Nfreqs * uvd.Npols, 1)
    uvd2.flag_array = uvd.flag_array.reshape(uvd.Nblts, uvd.Nfreqs * uvd.Npols, 1)
    uvd2.nsample_array = uvd.nsample_array.reshape(uvd.Nblts, uvd.Nfreqs * uvd.Npols, 1)
    with pytest.raises(ValueError, match="Some auto-correlations have non-real values"):
        uvd2.check(check_autos=True)

    uvd.convert_to_flex_pol()

    if sorting == "channel":
        spw_reorder = 2
        uvd.reorder_freqs(select_spw=spw_reorder, channel_order="-freq")
    elif sorting == "spw1":
        uvd.reorder_freqs(spw_order="number")
    elif sorting == "spw2":
        if multi_spw:
            spw_final_order = [1, 4, 5, 0, 3, 2, 6, 7, 8, 9, 10, 11]
            spw_order = np.zeros_like(uvd.spw_array)
            for idx, spw in enumerate(spw_final_order):
                spw_order[idx] = np.nonzero(uvd.spw_array == spw)[0][0]
            uvd.reorder_freqs(spw_order=spw_order)

    uvd.check(check_autos=True)

    outfile = os.path.join(tmp_path, "test.uvh5")
    uvd.write_uvh5(outfile)
    uvd2 = UVData.from_file(outfile, remove_flex_pol=False)

    assert uvd2 == uvd

    uvd3 = UVData.from_file(outfile)
    uvd2.remove_flex_pol()

    assert uvd3 == uvd2

    uvd2.check()

    if multi_spw and sorting == "spw1":
        # This changes which spw numbers are kept, so need to renumber for equality
        spw_renumber_dict = {0: 1, 2: 4, 3: 5}
        new_spw_array = np.zeros_like(uvd2.spw_array)
        for idx, spw in enumerate(uvd2.spw_array):
            new_spw = spw_renumber_dict[spw]
            new_spw_array[idx] = new_spw
            uvd2.flex_spw_id_array[np.nonzero(uvd2.flex_spw_id_array == spw)[0]] = (
                new_spw
            )
        uvd2.spw_array = new_spw_array
        uvd2.check()

    assert uvd_orig == uvd2


@pytest.mark.parametrize(
    "err_msg,param,param_val",
    [["Cannot make a flex-pol UVData object, as some windows have", None, None]],
)
def test_make_flex_pol_errs(sma_mir, err_msg, param, param_val):
    """Check to make sure that _make_flex_pol throws correct errors"""

    if param is not None:
        setattr(sma_mir, param, param_val)

    sma_copy = sma_mir.copy()

    with pytest.raises(ValueError, match=err_msg):
        sma_mir._make_flex_pol(raise_error=True, raise_warning=True)

    with check_warnings(UserWarning, err_msg):
        sma_mir._make_flex_pol(raise_error=False, raise_warning=True)
    assert sma_copy == sma_mir

    with check_warnings(None):
        sma_mir._make_flex_pol(raise_error=False, raise_warning=False)
    assert sma_copy == sma_mir


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("dataset", ["hera", "mwa"])
def test_auto_check(hera_uvh5, uv_phase_comp, dataset, tmp_path):
    """
    Checks that checking/fixing the autos works correctly, both with dual-pol data
    (supplied by hera_uvh5) and full-pol data (supplied by casa_uvfits).
    """
    if dataset == "hera":
        uv = hera_uvh5
    elif dataset == "mwa":
        uv, _ = uv_phase_comp

    out_file = os.path.join(tmp_path, "auto_check.uvh5")

    # Corrupt the auto data
    auto_screen = uv.ant_1_array == uv.ant_2_array
    uv.data_array[auto_screen] *= 1 + 0.5j

    with pytest.raises(
        ValueError, match="Some auto-correlations have non-real values in data_array."
    ):
        uv.write_uvh5(out_file, clobber=True)

    uv.write_uvh5(out_file, check_autos=False, clobber=True)

    with pytest.raises(
        ValueError, match="Some auto-correlations have non-real values in data_array."
    ):
        uv1 = uv.from_file(out_file, fix_autos=False)
    warn_types = [UserWarning]
    msg = ["Fixing auto-correlations to be be real"]
    with check_warnings(warn_types, match=msg):
        uv1 = UVData.from_file(out_file)

    with check_warnings(UserWarning, match="Fixing auto-correlations to be be real"):
        uv.write_uvh5(out_file, fix_autos=True, clobber=True)

    uv2 = uv.from_file(out_file)

    assert uv1 == uv2

    assert uv == uv1


@pytest.mark.parametrize(
    "select_kwargs,err_type,err_msg",
    [
        [{"bls": (1, 2)}, ValueError, "No autos available in this data set to do"],
        [{"polarizations": -7}, ValueError, "Cannot normalize xy, matching pols"],
    ],
)
def test_normalize_by_autos_errs(uv_phase_comp, select_kwargs, err_type, err_msg):
    uv, _ = uv_phase_comp
    uv.select(**select_kwargs)
    with pytest.raises(err_type, match=err_msg):
        uv.normalize_by_autos()


@pytest.mark.parametrize("muck_data", ["pol", "time", None])
def test_normalize_by_autos_roundtrip(hera_uvh5, muck_data):
    """
    Check that we can roundtrip autocorrelation normalization under various
    different circumstances.
    """
    if muck_data == "pol":
        # Stick in pseudo-Stokes pols and verify that these roundtrip correctly
        hera_uvh5.polarization_array[:] = [1, 2]
    elif muck_data == "time":
        # Flip the times so that they are now out of order.
        hera_uvh5.time_array[:] = hera_uvh5.time_array[::-1]

    hera_copy = hera_uvh5.copy()
    # Figure out where all the crosses live
    cross_mask = hera_uvh5.ant_1_array != hera_uvh5.ant_2_array

    # Make sure that all the values have high amps
    assert np.all(np.abs(hera_copy.data_array[cross_mask]) > 1)

    # Normalize and verify that things look as expected.
    hera_copy.normalize_by_autos()
    assert np.all(np.abs(hera_copy.data_array[cross_mask]) < 1)
    assert hera_copy != hera_uvh5

    # Complete the roundtrip, and verify that things look identical to how
    # we started.
    hera_copy.normalize_by_autos(invert=True)
    assert hera_copy == hera_uvh5


def test_normalize_by_autos_flag_noautos(hera_uvh5):
    """
    Verify that the crosses are correctly flagged as expected when a given antennas
    autos are missing from the data set.
    """
    hera_uvh5.select(
        bls=[(0, 1), (2, 1), (0, 0), (1, 11), (1, 1), (0, 2), (2, 2), (2, 11), (0, 11)]
    )
    cross_mask = (hera_uvh5.ant_1_array == 11) | (hera_uvh5.ant_2_array == 11)
    assert not np.any(hera_uvh5.flag_array[cross_mask])

    hera_uvh5.normalize_by_autos()
    assert np.all(hera_uvh5.flag_array[cross_mask])


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("multi_phase", [True, False])
def test_split_write_comb_read(tmp_path, multi_phase):
    """Pulled from a failed tutorial example."""
    uvd = UVData()
    filename = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits")

    uvd.read(filename)

    if multi_phase:
        mask = np.full(uvd.Nblts, False)
        mask[: uvd.Nblts // 2] = True
        uvd.phase(ra=0, dec=0, phase_frame="fk5", select_mask=mask, cat_name="foo")

    uvd1 = uvd.select(freq_chans=np.arange(0, 20), inplace=False)
    uvd2 = uvd.select(freq_chans=np.arange(20, 40), inplace=False)
    uvd3 = uvd.select(freq_chans=np.arange(40, 64), inplace=False)

    uvd1.write_uvfits(os.path.join(tmp_path, "select1.uvfits"))
    uvd2.write_uvfits(os.path.join(tmp_path, "select2.uvfits"))
    uvd3.write_uvfits(os.path.join(tmp_path, "select3.uvfits"))
    filenames = [
        os.path.join(tmp_path, f)
        for f in ["select1.uvfits", "select2.uvfits", "select3.uvfits"]
    ]
    uvd2 = UVData()

    uvd2.read(filenames)

    uvd2._consolidate_phase_center_catalogs(other=uvd)

    uvd2.history = uvd.history
    assert uvd2 == uvd


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("projected", [True, False])
@pytest.mark.parametrize("check_before_write", [True, False])
def test_init_like_hera_cal(hera_uvh5, tmp_path, projected, check_before_write):
    """Pulled from an error in hera_cal."""

    params = [
        "Nants_data",
        "Nbls",
        "Nblts",
        "Nfreqs",
        "Npols",
        "Nspws",
        "Ntimes",
        "Nphase",
        "ant_1_array",
        "ant_2_array",
        "baseline_array",
        "channel_width",
        "data_array",
        "extra_keywords",
        "flag_array",
        "flex_spw_id_array",
        "freq_array",
        "history",
        "integration_time",
        "lst_array",
        "nsample_array",
        "phase_center_catalog",
        "phase_center_id_array",
        "phase_center_app_ra",
        "phase_center_app_dec",
        "phase_center_frame_pa",
        "polarization_array",
        "spw_array",
        "time_array",
        "uvw_array",
        "vis_units",
    ]

    tel_params = [
        "name",
        "location",
        "instrument",
        "Nfeeds",
        "feed_array",
        "feed_angle",
        "mount_type",
        "Nants",
        "antenna_names",
        "antenna_numbers",
        "antenna_positions",
        "antenna_diameters",
    ]

    uvd = UVData()
    for par in tel_params:
        setattr(uvd.telescope, par, getattr(hera_uvh5.telescope, par))

    if projected:
        hera_uvh5.phase_center_catalog[0]["cat_type"] = "sidereal"
        hera_uvh5.phase_center_catalog[0]["cat_lon"] = 0.0
        hera_uvh5.phase_center_catalog[0]["cat_lat"] = 0.0
        hera_uvh5.phase_center_catalog[0]["cat_frame"] = "icrs"
        hera_uvh5.phase_center_catalog[0]["cat_epoch"] = 2000.0
        hera_uvh5._set_app_coords_helper()
        warn_type = UserWarning
        msg = "The uvw_array does not match the expected values"
    else:
        warn_type = None
        msg = None

    param_dict = {}
    for par in params:
        param_dict[par] = getattr(hera_uvh5, par)

    # set parameters in uvd
    for par in params:
        if par not in param_dict:
            continue
        uvd.__setattr__(par, param_dict[par])

    if check_before_write:
        with check_warnings(warn_type, match=msg):
            uvd.check()

        if projected:
            assert uvd.phase_center_catalog[0]["cat_type"] == "sidereal"
        else:
            assert uvd.phase_center_catalog[0]["cat_name"] == "zenith"
            uvd.phase_center_catalog[0]["cat_name"] = hera_uvh5.phase_center_catalog[0][
                "cat_name"
            ]

        assert uvd == hera_uvh5

    testfile = os.path.join(tmp_path, "outtest.uvh5")
    uvd.write_uvh5(testfile)

    uvd2 = UVData.from_file(testfile)

    if not check_before_write:
        if projected:
            assert uvd2.phase_center_catalog[0]["cat_type"] == "sidereal"
        else:
            assert uvd2.phase_center_catalog[0]["cat_name"] == "zenith"
            uvd2.phase_center_catalog[0]["cat_name"] = hera_uvh5.phase_center_catalog[
                0
            ]["cat_name"]

    assert uvd2 == hera_uvh5


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_setting_time_axis_wrongly(casa_uvfits):
    casa_uvfits.time_axis_faster_than_bls = True
    with pytest.raises(ValueError, match="time_axis_faster_than_bls is True but"):
        casa_uvfits.check()

    casa_uvfits.reorder_blts("time", minor_order="baseline")
    casa_uvfits.blts_are_rectangular = True
    casa_uvfits.time_axis_faster_than_bls = True
    with pytest.raises(
        ValueError, match="time_axis_faster_than_bls is True but time_array"
    ):
        casa_uvfits.check()

    casa_uvfits.reorder_blts("baseline", minor_order="time")
    assert not casa_uvfits.time_axis_faster_than_bls
    casa_uvfits.blts_are_rectangular = True
    assert not casa_uvfits.time_axis_faster_than_bls


def test_set_rectangularity(casa_uvfits, hera_uvh5):
    # setting force=True will set the rectangularity attributes only if obvious
    casa_uvfits.set_rectangularity(force=True)
    assert casa_uvfits.blts_are_rectangular is False
    assert casa_uvfits.time_axis_faster_than_bls is False

    hera_uvh5.reorder_blts("time", minor_order="baseline")
    hera_uvh5.set_rectangularity(force=True)
    assert hera_uvh5.blts_are_rectangular is True
    assert hera_uvh5.time_axis_faster_than_bls is False

    hera_uvh5.reorder_blts(np.random.permutation(hera_uvh5.Nblts))
    hera_uvh5.set_rectangularity(force=True)
    assert hera_uvh5.blts_are_rectangular is False
    assert hera_uvh5.time_axis_faster_than_bls is False


def test_determine_blt_order(hera_uvh5):
    # test that the blt order is determined correctly
    hera_uvh5.reorder_blts(order="time", minor_order="baseline")
    assert hera_uvh5.blt_order == ("time", "baseline")

    # Run determine with order already set -- should short-circuit.
    order = hera_uvh5.determine_blt_order()
    assert order == ("time", "baseline")

    # woops, forgot the blt_order!
    hera_uvh5.blt_order = None

    # lets find it again....
    order = hera_uvh5.determine_blt_order()
    assert order == ("time", "baseline")


def test_select_catalog_name_errs(hera_uvh5):
    with pytest.raises(
        ValueError, match="Cannot set both phase_center_ids and catalog_names."
    ):
        hera_uvh5.select(phase_center_ids=[1, 2, 3], catalog_names=[1, 2, 3])


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("invert", [True, False])
def test_select_catalog_name(carma_miriad, invert):
    # Select out the source
    for cat_id, cat_dict in carma_miriad.phase_center_catalog.items():
        uv_name = carma_miriad.select(
            catalog_names=cat_dict["cat_name"], inplace=False, invert=invert
        )
        uv_id = carma_miriad.select(
            phase_center_ids=cat_id, inplace=False, invert=invert
        )

        assert uv_id.history != uv_name.history
        uv_id.history = uv_name.history = None

        assert uv_name == uv_id

        assert np.any(np.isin(cat_id, uv_id.phase_center_id_array)) != invert


def test_update_antenna_positions_no_op(sma_mir, sma_mir_main):
    """Test no-op condition for updating antenna positions."""
    with check_warnings(UserWarning, "No antenna positions appear to have chan"):
        sma_mir.update_antenna_positions({-1: [0, 0, 0]})

    assert sma_mir == sma_mir_main


@pytest.mark.parametrize("flip_antpos", [False, True])
@pytest.mark.parametrize("delta_antpos", [False, True])
def test_update_antenna_positions(sma_mir, delta_antpos, flip_antpos):
    # Call this now to mitigate minor metadata recording errors.
    sma_mir.set_uvws_from_antenna_positions()

    sma_copy = sma_mir.copy()

    if flip_antpos:
        # Flip the coords and antpos to see if we get back to where we are supposed to
        # be in the uvws (though we'll ignore the data in this case).
        sma_mir.uvw_array *= -1
        sma_mir.telescope.antenna_positions *= -1
    else:
        # Introduce a small delta to all ants so that the positions are different
        sma_mir.telescope.antenna_positions += 1

    new_positions = dict(
        zip(
            sma_copy.telescope.antenna_numbers,
            sma_copy.telescope.antenna_positions,
            strict=True,
        )
    )

    sma_mir.update_antenna_positions(
        new_positions=new_positions, delta_antpos=delta_antpos
    )

    if flip_antpos:
        sma_mir.data_array = sma_copy.data_array

    assert sma_mir == sma_copy


def test_antpair2ind_rect_not_ordered(hera_uvh5):
    hera_uvh5.reorder_blts(order="baseline", minor_order="time")

    assert hera_uvh5.blts_are_rectangular

    inds = hera_uvh5.antpair2ind((0, 1), ordered=True)

    assert np.all(inds == hera_uvh5.antpair2ind((1, 0), ordered=False))


def test_antpair2ind_not_rect_not_ordered(hera_uvh5):
    hera_uvh5.reorder_blts(order=np.random.permutation(hera_uvh5.Nblts))
    assert not hera_uvh5.blts_are_rectangular
    inds = hera_uvh5.antpair2ind((1, 0), ordered=False)

    assert np.all(inds == hera_uvh5.antpair2ind((0, 1), ordered=True))


def test_antpair2ind_unordered_both_exist(hera_uvh5):
    # Make a new object that has conjugated baselines in it as well.
    hera_uvh5.set_rectangularity()
    hera_uvh5.conjugate_bls("ant1<ant2")
    full = hera_uvh5.copy()

    full.conjugate_bls("ant2<ant1")
    full.fast_concat(hera_uvh5, axis="blt", inplace=True)
    full.select(bls=[(0, 1), (1, 0)], inplace=True)
    full.reorder_blts(order="time", minor_order="baseline")
    inds_ordered = full.antpair2ind((0, 1), ordered=True)
    inds_unordered = full.antpair2ind((0, 1), ordered=False)

    idxs = np.arange(full.Nblts)
    assert len(idxs[inds_ordered]) < len(idxs[inds_unordered])


def test_key2inds_nonexistent_pol(hera_uvh5):
    with pytest.raises(KeyError, match="Polarization -7 not found in data"):
        hera_uvh5._key2inds((1, 0, -7))


def test_get_ants_rectangular(hera_uvh5):
    hera_uvh5.blts_are_rectangular = False  # even if its true...
    ants = np.sort(hera_uvh5.get_ants())

    hera_uvh5.reorder_blts(order="baseline", minor_order="time")
    ants1 = np.sort(hera_uvh5.get_ants())
    assert np.all(ants1 == ants)

    hera_uvh5.reorder_blts(order="time", minor_order="baseline")
    ants1 = np.sort(hera_uvh5.get_ants())
    assert np.all(ants1 == ants)


def test_pol_convention_warnings(hera_uvh5):
    hera_uvh5.vis_units = "Jy"

    hera_uvh5.pol_convention = "badconvention"
    with pytest.raises(ValueError):
        hera_uvh5.check()

    hera_uvh5.vis_units = "uncalib"
    hera_uvh5.pol_convention = "sum"
    with pytest.raises(
        ValueError, match="pol_convention is set but the data is uncalibrated"
    ):
        hera_uvh5.check()


def test_select_no_bls_match(sma_mir):
    # Test a particular corner-case with the bls to make sure that the resultant
    # error is on baselines and _not_ polarizations
    with (
        pytest.raises(ValueError, match="No baseline-times were found that match"),
        check_warnings(UserWarning, match=re.escape("Antenna pair (10, 20, 'xx')")),
    ):
        sma_mir.select(bls=(10, 20, "xx"), strict=False)


@pytest.mark.parametrize("invert", [True, False])
def test_select_partial_spw_match(sma_mir, invert):
    with check_warnings(UserWarning, match="SPW number 5 is not present"):
        sma_mir.select(spws=[1, 2, 3, 4, 5], invert=invert)

    assert np.all(np.isin(sma_mir.spw_array, [1, 2, 3, 4], invert=invert))


@pytest.mark.parametrize("invert", [True, False])
def test_select_partial_pol_match(sma_mir, invert):
    with check_warnings(UserWarning, match="Polarization xy is not present"):
        sma_mir.select(polarizations=["xx", "xy"], invert=invert)

    assert np.all(np.isin(sma_mir.polarization_array, [-5], invert=invert))


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_near_field_corrections():
    uvfits_raw = os.path.join(DATA_PATH, "1061316296.uvfits")
    corr_w_path = os.path.join(DATA_PATH, "1061316296_nearfield_w.npy")

    uvd_raw = UVData()
    uvd_raw.read(uvfits_raw)
    uvd_raw.phase(
        ra=np.radians(30),
        dec=np.radians(-20),
        cat_name="foo",
        dist=10000,
        cat_type="near_field",
    )

    # Only compare the w's
    corr_w = np.load(corr_w_path)

    assert np.allclose(uvd_raw.uvw_array[:, -1], corr_w)


def test_near_field_err():
    uvfits_sample = os.path.join(DATA_PATH, "1061316296.uvfits")

    uvd = UVData()
    uvd.read(uvfits_sample)

    with pytest.raises(
        ValueError, match="dist parameter must be specified for cat_type 'near_field'"
    ):
        uvd.phase(
            ra=np.radians(30),
            dec=np.radians(-20),
            cat_name="foo",
            cat_type="near_field",
        )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "file_format, import_check, error_message",
    [
        (
            "miriad",
            lambda: pytest.importorskip("pyuvdata.uvdata._miriad"),
            "Writing near-field phased data to miriad format is not yet supported.",
        ),
        (
            "ms",
            lambda: pytest.importorskip("casacore"),
            "Writing near-field phased data to Measurement Set format "
            + "is not yet supported.",
        ),
        (
            "uvfits",
            None,
            "Writing near-field phased data to uvfits format is not yet supported.",
        ),
    ],
)
def test_write_near_field_err(file_format, import_check, error_message):
    uvfits_sample = os.path.join(DATA_PATH, "1061316296.uvfits")

    uvd = UVData()
    uvd.read(uvfits_sample)
    uvd.phase(
        ra=np.radians(30),
        dec=np.radians(-20),
        cat_name="foo",
        dist=10000,
        cat_type="near_field",
    )

    if import_check:
        import_check()

    with pytest.raises(NotImplementedError, match=error_message):
        if file_format == "miriad":
            uvd.write_miriad("test_path")
        elif file_format == "ms":
            uvd.write_ms("test_path")
        elif file_format == "uvfits":
            uvd.write_uvfits("test_path")
