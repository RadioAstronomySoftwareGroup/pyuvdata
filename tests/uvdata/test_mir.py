# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Mir class.

Performs a series of test for the Mir class, which inherits from UVData. Note that
there is a separate test module for the MirParser class (mir_parser.py), which is
what is used to read the raw binary data into something that the Mir class can
manipulate into a UVData object.
"""

import os

import numpy as np
import pytest

from pyuvdata import UVData, utils
from pyuvdata.datasets import fetch_data
from pyuvdata.testing import check_warnings
from pyuvdata.uvdata.mir import (
    Mir,
    calc_delta_uvw_from_offset_illum,
    generate_sma_antpos_dict,
)
from pyuvdata.uvdata.mir_parser import MirParser


@pytest.fixture(scope="session")
def sma_mir_filt_main():
    uv_object = UVData()
    with check_warnings(
        UserWarning,
        match=[
            "> 25 ms errors detected reading in LST values from MIR data. ",
            "The lst_array is not self-consistent with the time_array and telescope ",
        ],
    ):
        uv_object.read(fetch_data("sma_mir"), pseudo_cont=True, corrchunk=0)

    uv_object.flag_array[:, : uv_object.Nfreqs // 2, 0] = True
    uv_object.flag_array[:, uv_object.Nfreqs // 2 :, 1] = True
    uv_object.set_lsts_from_time_array()
    uv_object._set_app_coords_helper()

    yield uv_object


@pytest.fixture(scope="function")
def sma_mir_filt(sma_mir_filt_main):
    uv_object = sma_mir_filt_main.copy()

    yield uv_object


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
def test_read_mir_write_uvfits(sma_mir, tmp_path):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    testfile = os.path.join(tmp_path, "outtest_mir.uvfits")
    uvfits_uv = UVData()

    sma_mir.write_uvfits(testfile)
    uvfits_uv.read_uvfits(testfile)
    assert sma_mir.telescope.instrument == uvfits_uv.telescope.instrument
    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific parameters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(sma_mir, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    # UVFITS doesn't allow for numbering of spectral windows like MIR does, so
    # we need an extra bit of handling here
    assert len(np.unique(sma_mir.spw_array)) == len(np.unique(uvfits_uv.spw_array))

    spw_dict = dict(zip(uvfits_uv.spw_array, sma_mir.spw_array, strict=True))

    assert np.all(
        [
            idx == spw_dict[jdx]
            for idx, jdx in zip(
                sma_mir.flex_spw_id_array, uvfits_uv.flex_spw_id_array, strict=True
            )
        ]
    )

    # Now that we've checked, set this things as equivalent
    uvfits_uv.spw_array = sma_mir.spw_array
    uvfits_uv.flex_spw_id_array = sma_mir.flex_spw_id_array

    # Check the history first via find
    assert (
        uvfits_uv.history.find(
            sma_mir.history + "  Read/written with pyuvdata version:"
        )
        == 0
    )
    sma_mir.history = uvfits_uv.history

    # We have to do a bit of special handling for the phase_center_catalog, because
    # _very_ small errors (like last bit in the mantissa) creep in when passing through
    # the util function transform_sidereal_coords (for mutli-phase-ctr datasets). Verify
    # the two match up in terms of their coordinates
    for cat_name in sma_mir.phase_center_catalog:
        assert np.isclose(
            sma_mir.phase_center_catalog[cat_name]["cat_lat"],
            uvfits_uv.phase_center_catalog[cat_name]["cat_lat"],
            rtol=0,
            atol=utils.RADIAN_TOL,
        )
        assert np.isclose(
            sma_mir.phase_center_catalog[cat_name]["cat_lon"],
            uvfits_uv.phase_center_catalog[cat_name]["cat_lon"],
            rtol=0,
            atol=utils.RADIAN_TOL,
        )
    uvfits_uv.phase_center_catalog = sma_mir.phase_center_catalog

    # There's a minor difference between what SMA calculates online for app coords
    # and what pyuvdata calculates, to the tune of ~1 arcsec. Check those values here,
    # then set them equal to one another.
    assert np.all(
        np.abs(sma_mir.phase_center_app_ra - uvfits_uv.phase_center_app_ra) < 1e-5
    )

    assert np.all(
        np.abs(sma_mir.phase_center_app_dec - uvfits_uv.phase_center_app_dec) < 1e-5
    )

    sma_mir._set_app_coords_helper()
    uvfits_uv._set_app_coords_helper()

    # make sure filenames are what we expect
    assert sma_mir.filename == ["sma_test.mir"]
    assert uvfits_uv.filename == ["outtest_mir.uvfits"]
    sma_mir.filename = uvfits_uv.filename

    assert sma_mir == uvfits_uv


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.parametrize("use_feeds", [True, False])
def test_read_mir_write_ms(sma_mir, tmp_path, use_feeds):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as ms, read back in and check for
    object equality.
    """
    pytest.importorskip("casacore")
    testfile = os.path.join(tmp_path, "outtest_mir.ms")
    ms_uv = UVData()

    if not use_feeds:
        sma_mir.telescope.feed_array = None
        sma_mir.telescope.feed_angle = None
        sma_mir.telescope.Nfeeds = None

    sma_mir.write_ms(testfile, clobber=True)
    ms_uv.read(testfile)

    # fix up the phase center info to match the mir dataset
    cat_id = list(sma_mir.phase_center_catalog.keys())[0]
    cat_name = sma_mir.phase_center_catalog[cat_id]["cat_name"]
    ms_uv._update_phase_center_id(
        list(ms_uv.phase_center_catalog.keys())[0], new_id=cat_id
    )
    ms_uv.phase_center_catalog[cat_id]["cat_name"] = cat_name
    ms_uv.phase_center_catalog[cat_id]["info_source"] = "file"

    # Single integration with 1 phase center = single scan number
    # output in the MS
    assert ms_uv.scan_number_array == np.array([1])

    # There are some minor differences between the values stored by MIR and that
    # calculated by UVData. Since MS format requires these to be calculated on the fly,
    # we calculate them here just to verify that everything is looking okay.
    sma_mir.set_lsts_from_time_array()
    sma_mir._set_app_coords_helper()

    # These reorderings just make sure that data from the two formats are lined up
    # correctly.
    sma_mir.reorder_freqs(spw_order="number")
    ms_uv.reorder_blts()

    # MS doesn't have the concept of an "instrument" name like FITS does, and instead
    # defaults to the telescope name. Make sure that checks out here.
    assert sma_mir.telescope.instrument == "SWARM"
    assert ms_uv.telescope.instrument == "SMA"
    sma_mir.telescope.instrument = ms_uv.telescope.instrument

    # Quick check for history here
    assert ms_uv.history != sma_mir.history
    ms_uv.history = sma_mir.history

    # Only MS has extra keywords, verify those look as expected.
    assert ms_uv.extra_keywords == {"DATA_COL": "DATA", "observer": "SMA"}
    assert sma_mir.extra_keywords == {}
    sma_mir.extra_keywords = ms_uv.extra_keywords

    # Make sure the filenames line up as expected.
    assert sma_mir.filename == ["sma_test.mir"]
    assert ms_uv.filename == ["outtest_mir.ms"]
    sma_mir.filename = ms_uv.filename = None

    # Finally, with all exceptions handled, check for equality.
    assert ms_uv.__eq__(sma_mir, allowed_failures=["filename"])


@pytest.mark.filterwarnings("ignore:LST values stored ")
def test_read_mir_write_uvh5(sma_mir, tmp_path):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvh5, read back in and check for
    object equality.
    """
    testfile = os.path.join(tmp_path, "outtest_mir.uvh5")
    uvh5_uv = UVData()

    sma_mir.write_uvh5(testfile)
    uvh5_uv.read_uvh5(testfile)

    # Check the history first via find
    assert (
        uvh5_uv.history.find(sma_mir.history + "  Read/written with pyuvdata version:")
        == 0
    )

    # test fails because of updated history, so this is our workaround for now.
    sma_mir.history = uvh5_uv.history

    # make sure filenames are what we expect
    assert sma_mir.filename == ["sma_test.mir"]
    assert uvh5_uv.filename == ["outtest_mir.uvh5"]
    sma_mir.filename = uvh5_uv.filename

    assert sma_mir == uvh5_uv


def test_mir_partial_read(sma_mir):
    """Check that select is done after the read for select on read with mir files."""
    uv = sma_mir

    uv2 = uv.copy()
    freq_chans_to_keep = np.arange(uv.Nfreqs // 2)
    uv2.select(freq_chans=freq_chans_to_keep)

    with check_warnings(
        UserWarning,
        match=[
            "Warning: a select on read keyword is set that is not supported by "
            "read_mir. This select will be done after reading the file.",
            "> 25 ms errors detected reading in LST values from MIR data. ",
            "The lst_array is not self-consistent with the time_array and telescope ",
            "The lst_array is not self-consistent with the time_array and telescope ",
        ],
    ):
        uv3 = UVData.from_file(fetch_data("sma_mir"), freq_chans=freq_chans_to_keep)
    uv3.set_lsts_from_time_array()
    assert uv3 == uv2


def test_write_mir(hera_uvh5, err_type=NotImplementedError):
    """
    Mir writer test

    Check and make sure that attempts to use the writer return a
    'not implemented' error.
    """

    # Check and see if the correct error is raised
    with pytest.raises(err_type):
        hera_uvh5.write_mir("dummy.mir")


def test_multi_nchan_spw_read(tmp_path):
    """
    Mir to uvfits error test for spws of different sizes.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    uv_in = UVData()
    with check_warnings(
        UserWarning,
        match=[
            "> 25 ms errors detected reading in LST values from MIR data. ",
            "The lst_array is not self-consistent with the time_array and telescope ",
        ],
    ):
        uv_in.read_mir(fetch_data("sma_mir"), corrchunk=[0, 1, 2, 3, 4])
    uv_in.set_lsts_from_time_array()

    dummyfile = os.path.join(tmp_path, "dummy.mirtest.uvfits")
    with pytest.raises(IndexError):
        uv_in.write_uvfits(dummyfile)


@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent with the.")
@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
@pytest.mark.parametrize("use_feeds", [True, False])
def test_read_mir_write_ms_flex_pol(mir_data, tmp_path, use_feeds):
    """
    Mir to MS loopback test with flex-pol.

    Read in Mir files, write out as ms, read back in and check for
    object equality.
    """
    pytest.importorskip("casacore")
    testfile = os.path.join(tmp_path, "read_mir_write_ms_flex_pol.ms")

    # Read in the raw data so that we can manipulate it, and make it look like the
    # test data set was recorded with split-tuning
    mir_data.sp_data._data["gunnLO"][np.isin(mir_data.sp_data["blhid"], [1, 3])] += 30.0
    mir_data.sp_data._data["fsky"][np.isin(mir_data.sp_data["blhid"], [1, 3])] += 30.0

    # Spin up a Mir object, which can be converted into a UVData object,
    # with flex-pol enabled.
    mir_uv = UVData()
    mir_obj = Mir()
    mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)

    if not use_feeds:
        mir_uv.telescope.feed_array = None
        mir_uv.telescope.feed_angle = None
        mir_uv.telescope.Nfeeds = None

    # Write out our modified data set
    mir_uv.write_ms(testfile, clobber=True)
    ms_uv = UVData.from_file(testfile)

    # fix up the phase center info to match the mir dataset
    cat_id = list(mir_uv.phase_center_catalog.keys())[0]
    cat_name = mir_uv.phase_center_catalog[cat_id]["cat_name"]
    ms_uv._update_phase_center_id(
        list(ms_uv.phase_center_catalog.keys())[0], new_id=cat_id
    )
    ms_uv.phase_center_catalog[cat_id]["cat_name"] = cat_name
    ms_uv.phase_center_catalog[cat_id]["info_source"] = "file"

    # There are some minor differences between the values stored by MIR and that
    # calculated by UVData. Since MS format requires these to be calculated on the
    # fly, we calculate them here just to verify that everything is looking okay.
    mir_uv.set_lsts_from_time_array()
    mir_uv._set_app_coords_helper()

    # These reorderings just make sure that data from the two formats
    # are lined up correctly.
    mir_uv.reorder_freqs(spw_order="number")
    ms_uv.reorder_blts()

    # MS doesn't have the concept of an "instrument" name like FITS does, and instead
    # defaults to the telescope name. Make sure that checks out here.
    assert mir_uv.telescope.instrument == "SWARM"
    assert ms_uv.telescope.instrument == "SMA"
    mir_uv.telescope.instrument = ms_uv.telescope.instrument

    # Quick check for history here
    assert ms_uv.history != mir_uv.history
    ms_uv.history = mir_uv.history

    # Only MS has extra keywords, verify those look as expected.
    assert ms_uv.extra_keywords == {"DATA_COL": "DATA", "observer": "SMA"}
    assert mir_uv.extra_keywords == {}
    mir_uv.extra_keywords = ms_uv.extra_keywords

    # Make sure the filenames line up as expected.
    assert mir_uv.filename == ["sma_test.mir"]
    assert ms_uv.filename == ["read_mir_write_ms_flex_pol.ms"]

    # Finally, with all exceptions handled, check for equality.
    assert ms_uv.__eq__(mir_uv, allowed_failures=["filename"])


def test_inconsistent_sp_records(mir_data, sma_mir):
    """
    Test that the MIR object does the right thing w/ inconsistent meta-data.
    """
    mir_data.select(where=("iband", "ne", 0))
    mir_data.sp_data._data["ipq"][1] = 0
    mir_data.load_data()

    mir_uv = UVData()
    mir_obj = Mir()
    with check_warnings(
        UserWarning,
        match=[
            "Per-spectral window metadata differ.",
            "> 25 ms errors detected reading in LST values",
        ],
    ):
        mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)
    mir_uv.set_lsts_from_time_array()

    assert mir_uv == sma_mir


def test_inconsistent_bl_records(mir_data, sma_mir):
    """
    Test that the MIR object does the right thing w/ inconsistent meta-data.
    """
    mir_data.select(where=("iband", "ne", 0))
    mir_data.bl_data._data["u"][0] = 0.0
    mir_data.load_data()
    mir_uv = UVData()
    mir_obj = Mir()
    with check_warnings(
        UserWarning,
        match=[
            "> 25 ms errors detected reading in LST values",
            "Per-baseline metadata differ.",
        ],
    ):
        mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)

    mir_uv.set_lsts_from_time_array()
    assert mir_uv == sma_mir


@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
def test_multi_ipol(mir_data, sma_mir):
    """
    Test that the MIR object does the right thing when different polarization types
    are recorded in the pol code.
    """
    mir_data.select(where=("iband", "ne", 0))
    mir_data.bl_data._data["ipol"][:] = mir_data.bl_data["ant1rx"]
    mir_data.load_data()

    mir_uv = UVData()
    mir_obj = Mir()
    mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)
    mir_uv.set_lsts_from_time_array()
    assert mir_uv == sma_mir


@pytest.mark.parametrize("filetype", ["uvh5", "miriad", "ms", "uvfits"])
def test_flex_pol_roundtrip(sma_mir_filt, filetype, tmp_path):
    """Test that we can round-trip flex-pol data sets"""
    testfile = os.path.join(tmp_path, "flex_pol_roundtrip." + filetype)
    if filetype == "ms":
        pytest.importorskip("casacore")
    elif filetype == "miriad":
        pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)

    uvd2 = sma_mir_filt.copy(metadata_only=True)

    sma_mir_filt._make_flex_pol(raise_error=True)
    with pytest.raises(
        ValueError,
        match="Cannot make a metadata_only UVData object flex-pol because flagging "
        "info is required.",
    ):
        uvd2._make_flex_pol(raise_error=True)
    with check_warnings(
        UserWarning,
        match="Cannot make a metadata_only UVData object flex-pol because flagging "
        "info is required.",
    ):
        uvd2._make_flex_pol()
    uvd2._make_flex_pol(raise_warning=False)
    uvd3 = sma_mir_filt.copy(metadata_only=True)
    assert uvd2 != uvd3

    uvd3.remove_flex_pol(combine_spws=False)
    assert uvd2 != uvd3

    exp_warning = None
    warn_str = ""

    if filetype in ["uvfits", "miriad"]:
        warn_str = [
            (
                "combine_spws is True but there are not matched spws for all "
                "polarizations, so spws will not be combined."
            )
        ]
        exp_warning = UserWarning

    if filetype == "miriad":
        warn_str.append(
            "writing default values for restfreq, vsource, veldop, jyperk, and systemp"
        )

    with check_warnings(exp_warning, warn_str):
        if filetype == "uvfits":
            sma_mir_filt.write_uvfits(testfile)
        else:
            getattr(sma_mir_filt, "write_" + filetype)(testfile)

    test_uv = UVData.from_file(testfile, remove_flex_pol=False)

    if filetype in ["uvfits", "miriad"]:
        test_uv._make_flex_pol(raise_error=True)

    assert np.all(
        sma_mir_filt.flex_spw_polarization_array == test_uv.flex_spw_polarization_array
    )
    assert np.all(sma_mir_filt.polarization_array == test_uv.polarization_array)
    assert np.all(sma_mir_filt.flag_array == test_uv.flag_array)
    assert np.all(sma_mir_filt.data_array == test_uv.data_array)
    assert np.all(sma_mir_filt.nsample_array == test_uv.nsample_array)


def test_flex_pol_select(sma_mir_filt):
    """Test select operations on flex-pol UVData objects"""
    sma_mir_filt._make_flex_pol(raise_error=True)
    with pytest.raises(
        ValueError, match="No data matching this polarization and frequency"
    ):
        sma_mir_filt.select(polarizations=["xx"], freq_chans=[0, 1, 2, 3])

    sma_mir_filt2 = sma_mir_filt.copy()
    sma_mir_filt.select(polarizations=["xx"])

    assert sma_mir_filt.flex_spw_polarization_array is None
    assert np.all(sma_mir_filt.polarization_array == -5)

    sma_mir_filt._make_flex_pol()

    assert np.all(sma_mir_filt.flex_spw_polarization_array == -5)
    assert np.all(sma_mir_filt.polarization_array == 0)

    # Need to add more polarizations to test uneven pol spacing
    sma_mir_filt3 = sma_mir_filt2.copy()
    sma_mir_filt3.flex_spw_polarization_array = np.asarray([-7, -8])
    sma_mir_filt3.flex_spw_id_array = np.asarray([0] * 4 + [1] * 4)
    sma_mir_filt3.spw_array = [0, 1]
    sma_mir_filt2 += sma_mir_filt3

    with check_warnings(
        UserWarning, match="Selected polarizations are not evenly spaced"
    ):
        sma_mir_filt2.select(polarizations=["xx", "xy", "yx"], warn_spacing=True)


def test_flex_pol_select_warning(sma_mir_filt):
    """
    Check that selecting flex-pol datasets with uneven pol spacing throws a warning.
    """
    sma_mir_filt._make_flex_pol(raise_error=True)
    sma_mir_filt.flex_spw_id_array = np.arange(8) // 2
    sma_mir_filt.spw_array = np.arange(4)
    sma_mir_filt.flex_spw_polarization_array = np.array([-8, -8, -6, -5])
    sma_mir_filt.Nspws = 4

    with check_warnings(UserWarning, "Selected polarizations are not evenly spaced."):
        sma_mir_filt.select(polarizations=[-8, -6, -5], warn_spacing=True)


@pytest.mark.parametrize(
    "make_flex,err_msg",
    [
        [True, "Cannot add a flex-pol UVData objects where the same"],
        [False, "Cannot add a flex-pol and non-flex-pol UVData objects."],
    ],
)
def test_flex_pol_add_errs(sma_mir_filt, make_flex, err_msg):
    """Test that the add error throws appropriate errors"""
    sma_copy = sma_mir_filt.copy()
    sma_mir_filt._make_flex_pol()

    if make_flex:
        sma_copy.select(polarizations=["xx"])
        sma_copy.flag_array[:] = True
        sma_copy.data_array[:] = 0.0
        sma_copy._make_flex_pol()

    with pytest.raises(ValueError) as cm:
        sma_copy + sma_mir_filt
    assert str(cm.value).startswith(err_msg)


def test_flex_pol_add(sma_mir_filt):
    """Test that the add method works correctly with flex-pol data"""
    # Grab two copies of the data before we start to manipulate it
    sma_xx_copy = sma_mir_filt.copy()
    sma_yy_copy = sma_mir_filt.copy()
    sma_mir_filt._make_flex_pol()

    # In both copies, isolate out the relevant polarization data.
    sma_xx_copy.select(polarizations=["xx"], freq_chans=[4, 5, 6, 7])
    sma_xx_copy._make_flex_pol()
    sma_yy_copy.select(polarizations=["yy"], freq_chans=[0, 1, 2, 3])
    sma_yy_copy._make_flex_pol()

    # Add the two back together here, and make sure we can the same value out,
    # modulo the history.
    sma_check = sma_yy_copy + sma_xx_copy

    assert sma_check.history != sma_mir_filt.history
    sma_check.history = sma_mir_filt.history = None

    assert sma_check == sma_mir_filt


def test_flex_pol_spw_all_flag(sma_mir_filt):
    """
    Test that if one spw is totally flagged, the polarization gets filled correctly.
    """
    # Flag all of the data where we have y-pol data
    sma_mir_filt.flag_array[:, : sma_mir_filt.Nfreqs // 2, :] = True
    sma_mir_filt._make_flex_pol()

    # Since all of the y-pol data is flagged, the flex-pol data should only contain
    # xx visibilities (as recorded in flex_spw_polarization_array).t
    assert np.all(sma_mir_filt.flex_spw_polarization_array == -5)


@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
def test_bad_sphid(mir_data):
    """
    Test what bad values for sphid in sp_data result in an error.
    """
    mir_obj = Mir()
    with check_warnings(UserWarning, "Changing fields that tie to header keys can"):
        mir_data.sp_data["sphid"] = -1
        mir_data.sp_data._set_header_key_index_dict()

    with pytest.raises(KeyError) as err:
        mir_obj._init_from_mir_parser(mir_data)
    assert str(err.value).startswith(
        "'Something went wrong in Mir._prep_and_insert_data. Please file an issue "
        "in our GitHub issue log so that we can help: "
        "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues."
        " Developer info: Mismatch between keys in vis_data and sphid in sp_data."
    )


@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
def test_bad_pol_code(mir_data):
    """
    Test that an extra (unused) pol code doesn't produce an error. Note that we want
    this check because the "Unknown" pol code is something present in some data sets.
    """
    mir_obj = Mir()
    mir_data.codes_data._data = np.resize(
        mir_data.codes_data._data, mir_data.codes_data._size + 1
    )
    mir_data.codes_data._data[-1] = ("pol", -999, "Unknown", 0)
    mir_data.codes_data._mask = np.ones(mir_data.codes_data._size, dtype=bool)

    mir_obj._init_from_mir_parser(mir_data)


@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
def test_rechunk_on_read():
    """Test that rechunking on read works as expected."""
    uv_data = UVData.from_file(fetch_data("sma_mir"), rechunk=16384)

    # Do some basic checks to make sure that this loaded correctly.
    assert uv_data.freq_array.size == 8
    assert np.all(uv_data.channel_width == 2.288e09)


@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"antenna_nums": [1, 4]},
        {"antenna_names": ["1", "4"]},
        {"bls": [(1, 4)]},
        {"time_range": [2459055, 2459056]},
        {"lst_range": [2, 2.5]},
        {"polarizations": ["hh", "vv"]},
        {"catalog_names": ["3c84"]},
        {"corrchunk": [1, 2, 3, 4]},
        {"receivers": ["230", "240"]},
        {"sidebands": ["l", "u"]},
    ],
)
def test_select_on_read(select_kwargs, sma_mir):
    uv_data = UVData.from_file(fetch_data("sma_mir"), **select_kwargs)
    uv_data.history = sma_mir.history
    uv_data.set_lsts_from_time_array()
    assert sma_mir == uv_data


@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
def test_non_icrs_coord_read(mir_data):
    # When fed a non-J2000 coordinate, we want to convert that so that it can easily
    mir_uv = UVData()
    mir_obj = Mir()

    # Plug in a dummy epoch value
    mir_data.in_data["epoch"] = 2020.0
    mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)

    cat_entry = list(mir_uv.phase_center_catalog.values())[0]

    assert cat_entry["cat_frame"] == "icrs"
    assert cat_entry["cat_epoch"] is None
    assert np.isclose(cat_entry["cat_lon"], 0.8718033763283803, atol=4e-9, rtol=0)
    assert np.isclose(cat_entry["cat_lat"], 0.724518442710549, atol=4e-9, rtol=0)

    # Note that if the test dataset were written perfectly, the apparent coords would
    # perfectly translate back to the original values in the dataset, but there's a
    # defect of ~0.5 arcsec in how the positions were recorded, which we tolerate
    np.testing.assert_allclose(
        cat_entry["cat_lon"], mir_data.in_data["rar"], atol=4e-6, rtol=0
    )
    np.testing.assert_allclose(
        cat_entry["cat_lat"], mir_data.in_data["decr"], atol=4e-6, rtol=0
    )


@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
def test_dedoppler_data(mir_data, sma_mir):
    mir_uv = UVData()
    mir_obj = Mir()

    # Need to unload data to redoppler
    mir_data.unload_data()

    # Deselect the pseudo-cont channel
    mir_data.select(where=("corrchunk", "ne", 0))

    # Now spoof a v4-style file
    mir_data.codes_data.set_value("code", "4", index=0)
    mir_data.sp_data["fDDS"] = 1.0

    # Complete the initialization of the UVData object
    mir_obj._init_from_mir_parser(mir_data, apply_dedoppler=True)
    mir_uv._convert_from_filetype(mir_obj)
    mir_uv.set_lsts_from_time_array()

    # Make sure things differ
    assert sma_mir != mir_uv

    # The only two things affected should be data_array, flag_array, and n_samples in
    # the object, everything else should be identical.
    mir_uv.data_array = sma_mir.data_array
    mir_uv.flag_array = sma_mir.flag_array
    mir_uv.nsample_array = sma_mir.nsample_array
    assert sma_mir == mir_uv


def test_source_pos_change_warning(mir_data, tmp_path):
    # We need to spoof a new file to synthetically generate a two-integration dataset
    filepath = os.path.join(tmp_path, "source_pos_change_warning")
    with check_warnings(UserWarning, "Writing out raw data with tsys applied."):
        mir_data.write(filepath)

    mir_copy = MirParser(filepath)

    # Now combine the data
    with check_warnings(
        UserWarning,
        [
            "Duplicate metadata found for the following attributes",
            "These two objects contain data taken at the exact same time",
            "Both objects do not have auto-correlation data.",
        ],
    ):
        mir_copy.__iadd__(mir_data, force=True, merge=False)

    # Muck the ra coord
    mir_copy.in_data["rar"] = [0, 1]
    mir_obj = Mir()

    with check_warnings(
        UserWarning,
        [
            "> 25 ms errors detected reading in LST values",
            "Position for 3c84 changes by more than an arcminute.",
        ],
    ):
        mir_obj._init_from_mir_parser(mir_copy)


def test_generate_sma_antpos_dict_errs():
    with pytest.raises(ValueError, match="No such file or folder exists"):
        generate_sma_antpos_dict("abcdefg")


@pytest.mark.parametrize("use_file", [True, False])
def test_generate_sma_antpos_dict(use_file, sma_mir):
    filepath = fetch_data("sma_mir")
    if use_file:
        filepath = os.path.join(filepath, "antennas")

    ant_dict = generate_sma_antpos_dict(filepath)
    for ant_num, xyz_pos in zip(
        sma_mir.telescope.antenna_numbers,
        sma_mir.telescope.antenna_positions,
        strict=True,
    ):
        np.testing.assert_allclose(ant_dict[ant_num], xyz_pos, rtol=0, atol=1e-3)


def test_spw_consistency_warning(mir_data):
    mir_data.sp_data._data["fres"][1] *= 2
    mir_data.bl_data._data["ant1rx"][:] = 0
    mir_data.bl_data._data["ant2rx"][:] = 0

    mir_uv = Mir()
    with check_warnings(
        UserWarning,
        match=["Discrepancy in fres", "> 25 ms errors detected reading in LST values"],
    ):
        mir_uv._init_from_mir_parser(mir_data)


SMA_LAT = np.deg2rad(19.8242)


def _illum_offsets(*, ant_1_array, ant_2_array, ha, dec, illum_dict, frame_pa=None):
    """Convenience wrapper: app_ra = 0 so that lst = hour angle."""
    n_blts = len(ant_1_array)
    return calc_delta_uvw_from_offset_illum(
        illum_dict=illum_dict,
        ant_1_array=np.asarray(ant_1_array),
        ant_2_array=np.asarray(ant_2_array),
        app_ra=np.zeros(n_blts),
        app_dec=np.full(n_blts, dec),
        frame_pa=frame_pa,
        lst_array=np.full(n_blts, ha),
        telescope_lat=SMA_LAT,
    )


def test_illum_offset_transit_geometry():
    """
    At transit for a source south of the zenith, +El points north and +Az points west,
    so a y-offset on ant 2 gives +v (north) and an x-offset gives -u (west).
    """
    dec = SMA_LAT - np.deg2rad(30.0)
    d_y = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=0.0,
        dec=dec,
        illum_dict={2: {"x0": 0.0, "y0": 1.0, "x1": 0.0, "y1": 0.0}},
    )
    np.testing.assert_allclose(d_y, [[0.0, 1.0, 0.0]], atol=1e-12)
    d_x = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=0.0,
        dec=dec,
        illum_dict={2: {"x0": 1.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}},
    )
    np.testing.assert_allclose(d_x, [[-1.0, 0.0, 0.0]], atol=1e-12)

    # For a source transiting north of the zenith the whole frame is flipped
    dec = SMA_LAT + np.deg2rad(30.0)
    d_y = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=0.0,
        dec=dec,
        illum_dict={2: {"x0": 0.0, "y0": 1.0, "x1": 0.0, "y1": 0.0}},
    )
    np.testing.assert_allclose(d_y, [[0.0, -1.0, 0.0]], atol=1e-12)
    d_x = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=0.0,
        dec=dec,
        illum_dict={2: {"x0": 1.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}},
    )
    np.testing.assert_allclose(d_x, [[1.0, 0.0, 0.0]], atol=1e-12)


def test_illum_offset_parallactic_rotation():
    """
    Away from transit the dish frame is rotated by the parallactic angle: check the
    reflection + rotation against an independent evaluation, and the sign of the
    rotation using a setting source (zenith due east of the source -> +El maps to +E).
    """
    import erfa

    dec = np.deg2rad(-5.0)
    ha = np.deg2rad(45.0)
    x_off, y_off = 0.3, -0.7
    d = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=ha,
        dec=dec,
        illum_dict={2: {"x0": x_off, "y0": y_off, "x1": 0.0, "y1": 0.0}},
    )
    q = erfa.pas(0.0, dec, ha, SMA_LAT)
    expected = [
        -x_off * np.cos(q) + y_off * np.sin(q),
        x_off * np.sin(q) + y_off * np.cos(q),
        0.0,
    ]
    np.testing.assert_allclose(d[0], expected, atol=1e-12)
    assert d[0, 2] == 0.0

    # A source that crosses the prime vertical (dec > latitude) reaches a parallactic
    # angle of +90 deg, where the zenith is due east of the source and an offset toward
    # +El must therefore map to +East (a pure rotation by +q would give -East).
    from scipy.optimize import brentq

    dec = SMA_LAT + np.deg2rad(20.0)
    ha_grid = np.linspace(0.05, np.pi / 2, 200)
    q_grid = np.array([erfa.pas(0.0, dec, h, SMA_LAT) for h in ha_grid])
    idx = np.argmax((q_grid[:-1] - np.pi / 2) * (q_grid[1:] - np.pi / 2) < 0)
    ha_90 = brentq(
        lambda h: erfa.pas(0.0, dec, h, SMA_LAT) - np.pi / 2,
        ha_grid[idx],
        ha_grid[idx + 1],
    )
    d = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=ha_90,
        dec=dec,
        illum_dict={2: {"x0": 0.0, "y0": 1.0, "x1": 0.0, "y1": 0.0}},
    )
    np.testing.assert_allclose(d[0], [1.0, 0.0, 0.0], atol=1e-9)


def test_illum_offset_antisymmetry_and_autos():
    dec = np.deg2rad(10.0)
    ha = np.deg2rad(-20.0)
    illum_dict = {
        1: {"x0": 0.1, "y0": 0.2, "x1": -0.3, "y1": 0.4},
        2: {"x0": -0.5, "y0": 0.6, "x1": 0.7, "y1": -0.8},
    }
    d12 = _illum_offsets(
        ant_1_array=[1], ant_2_array=[2], ha=ha, dec=dec, illum_dict=illum_dict
    )
    d21 = _illum_offsets(
        ant_1_array=[2], ant_2_array=[1], ha=ha, dec=dec, illum_dict=illum_dict
    )
    d11 = _illum_offsets(
        ant_1_array=[1], ant_2_array=[1], ha=ha, dec=dec, illum_dict=illum_dict
    )
    np.testing.assert_allclose(d12, -d21, atol=1e-12)
    np.testing.assert_allclose(d11, 0.0, atol=1e-12)
    # Antennas absent from the dict have no offset
    d13 = _illum_offsets(
        ant_1_array=[1], ant_2_array=[3], ha=ha, dec=dec, illum_dict=illum_dict
    )
    d10 = _illum_offsets(
        ant_1_array=[1], ant_2_array=[3], ha=ha, dec=dec, illum_dict={1: illum_dict[1]}
    )
    np.testing.assert_allclose(d13, d10, atol=1e-12)


def test_illum_offset_elevation_dependence():
    """Cabin-fixed (x1, y1) terms rotate with elevation, primary-fixed ones do not."""
    dec = SMA_LAT  # transits through the zenith, so el = 90 deg at ha = 0
    illum_dict = {2: {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 0.0}}
    # At el = 90 deg: X = x0 + y1 = 0, Y = y0 - x1 = -1 (in the dish frame)
    d = _illum_offsets(
        ant_1_array=[1], ant_2_array=[2], ha=0.0, dec=dec, illum_dict=illum_dict
    )
    np.testing.assert_allclose(np.linalg.norm(d[0]), 1.0, atol=1e-9)
    # Near the horizon (el -> 0): X = x0 + x1 = 1, Y = y0 + y1 = 0
    import erfa

    for ha in np.deg2rad([-89.0, 89.0]):
        _, el = erfa.hd2ae(ha, dec, SMA_LAT)
        d = _illum_offsets(
            ant_1_array=[1], ant_2_array=[2], ha=ha, dec=dec, illum_dict=illum_dict
        )
        q = erfa.pas(0.0, dec, ha, SMA_LAT)
        X, Y = np.cos(el), -np.sin(el)
        np.testing.assert_allclose(
            d[0],
            [-X * np.cos(q) + Y * np.sin(q), X * np.sin(q) + Y * np.cos(q), 0.0],
            atol=1e-12,
        )


def test_illum_offset_frame_pa():
    """The frame rotation follows calc_uvw (a rotation of (u, v) by +frame_pa)."""
    dec = np.deg2rad(-5.0)
    ha = np.deg2rad(30.0)
    illum_dict = {2: {"x0": 0.4, "y0": 0.3, "x1": 0.0, "y1": 0.0}}
    d0 = _illum_offsets(
        ant_1_array=[1], ant_2_array=[2], ha=ha, dec=dec, illum_dict=illum_dict
    )
    d1 = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=ha,
        dec=dec,
        illum_dict=illum_dict,
        frame_pa=np.array([np.pi / 2]),
    )
    np.testing.assert_allclose(d1[0], [-d0[0, 1], d0[0, 0], 0.0], atol=1e-12)
    # Compare directly against calc_uvw's own frame rotation of a uvw vector
    d_ref = utils.phasing.calc_uvw(
        uvw_array=d0,
        old_frame_pa=np.zeros(1),
        frame_pa=np.array([0.3]),
        use_ant_pos=False,
    )
    d2 = _illum_offsets(
        ant_1_array=[1],
        ant_2_array=[2],
        ha=ha,
        dec=dec,
        illum_dict=illum_dict,
        frame_pa=np.array([0.3]),
    )
    np.testing.assert_allclose(d2, d_ref, atol=1e-12)


def test_illum_offset_errors():
    with pytest.raises(KeyError, match="Invalid keys in illum_dict."):
        _illum_offsets(
            ant_1_array=[1],
            ant_2_array=[2],
            ha=0.0,
            dec=0.0,
            illum_dict={2: {"x0": 1.0}},
        )
    with pytest.raises(ValueError, match="Antenna numbers in illum_dict must be in"):
        _illum_offsets(
            ant_1_array=[1],
            ant_2_array=[2],
            ha=0.0,
            dec=0.0,
            illum_dict={42: {"x0": 1.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}},
        )
    with pytest.raises(ValueError, match="Values in ant_2_array out of range"):
        _illum_offsets(
            ant_1_array=[1],
            ant_2_array=[42],
            ha=0.0,
            dec=0.0,
            illum_dict={2: {"x0": 1.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}},
        )


def test_read_mir_illum_dict():
    """Reading with illum_dict shifts the uvws as expected and notes it in history."""
    warn_list = [
        "> 25 ms errors detected reading in LST values from MIR data. ",
        "The lst_array is not self-consistent with the time_array and telescope ",
    ]
    uv_plain = UVData()
    with check_warnings(UserWarning, match=warn_list):
        uv_plain.read(fetch_data("sma_mir"))

    illum_dict = {
        ant: {"x0": 0.05 * ant, "y0": -0.02 * ant, "x1": 0.1, "y1": -0.3}
        for ant in uv_plain.telescope.antenna_numbers
    }
    uv_corr = UVData()
    # The test file contains both receivers, so a single set of offsets being applied
    # to both should be flagged.
    with check_warnings(
        UserWarning,
        match=warn_list + ["Data from more than one receiver are present, but the uvw"],
    ):
        uv_corr.read(fetch_data("sma_mir"), illum_dict=illum_dict)

    expected = calc_delta_uvw_from_offset_illum(
        illum_dict=illum_dict,
        ant_1_array=uv_plain.ant_1_array,
        ant_2_array=uv_plain.ant_2_array,
        app_ra=uv_plain.phase_center_app_ra,
        app_dec=uv_plain.phase_center_app_dec,
        frame_pa=uv_plain.phase_center_frame_pa,
        lst_array=uv_plain.lst_array,
        telescope_lat=uv_plain.telescope.location.lat.rad,
    )
    np.testing.assert_allclose(
        uv_corr.uvw_array - uv_plain.uvw_array, expected, atol=1e-9
    )
    assert np.any(np.abs(expected[:, :2]) > 0.1)
    assert np.all(expected[:, 2] == 0.0)
    assert "illumination offset corrections to uvw coordinates" in uv_corr.history

    # Everything other than the uvws and history should be untouched
    uv_corr.uvw_array = uv_plain.uvw_array
    uv_corr.history = uv_plain.history
    assert uv_corr == uv_plain


def test_read_mir_illum_dict_single_receiver():
    """No receiver-mix warning when a single receiver is read."""
    illum_dict = {1: {"x0": 0.1, "y0": 0.0, "x1": 0.0, "y1": 0.0}}
    uv_corr = UVData()
    with check_warnings(
        UserWarning,
        match=[
            "> 25 ms errors detected reading in LST values from MIR data. ",
            "The lst_array is not self-consistent with the time_array and telescope ",
        ],
    ):
        uv_corr.read(fetch_data("sma_mir"), illum_dict=illum_dict, receivers=["230"])
    assert "illumination offset corrections to uvw coordinates" in uv_corr.history
