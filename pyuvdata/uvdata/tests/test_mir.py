# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Mir class.

Performs a series of test for the Mir class, which inherits from UVData. Note that
there is a separate test module for the MirParser class (mir_parser.py), which is
what is used to read the raw binary data into something that the Mir class can
manipulate into a UVData object.
"""
import os

import pytest
import numpy as np

from ... import tests as uvtest
from ... import UVData
from ...data import DATA_PATH
from ...uvdata.mir_parser import MirParser
from ...uvdata.mir import Mir


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
def sma_mir_filt_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_object.read(testfile, pseudo_cont=True, corrchunk=0)

    uv_object.flag_array[:, :, : uv_object.Nfreqs // 2, 0] = True
    uv_object.flag_array[:, :, uv_object.Nfreqs // 2 :, 1] = True
    uv_object.set_lsts_from_time_array()
    uv_object._set_app_coords_helper()

    yield uv_object


@pytest.fixture(scope="function")
def sma_mir_filt(sma_mir_filt_main):
    # read in test file for the resampling in time functions
    uv_object = sma_mir_filt_main.copy()

    yield uv_object


@pytest.fixture
def uv_in_ms(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = os.path.join(tmp_path, "outtest_mir.ms")

    # Currently only one source is supported.
    uv_in.read(testfile)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.fixture
def uv_in_uvfits(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir/")
    write_file = os.path.join(tmp_path, "outtest_mir.uvfits")

    # Currently only one source is supported.
    uv_in.read(testfile, pseudo_cont=False)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.fixture
def uv_in_uvh5(tmp_path):
    uv_in = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    write_file = os.path.join(tmp_path, "outtest_mir.uvh5")

    # Currently only one source is supported.
    uv_in.read(testfile)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_read_mir_write_uvfits(uv_in_uvfits, future_shapes):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    mir_uv, uvfits_uv, testfile = uv_in_uvfits

    if future_shapes:
        mir_uv.use_future_array_shapes()

    mir_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvfits_uv.read_uvfits(testfile)

    if future_shapes:
        uvfits_uv.use_future_array_shapes()

    # UVFITS doesn't allow for numbering of spectral windows like MIR does, so
    # we need an extra bit of handling here
    assert len(np.unique(mir_uv.spw_array)) == len(np.unique(uvfits_uv.spw_array))

    spw_dict = {idx: jdx for idx, jdx in zip(uvfits_uv.spw_array, mir_uv.spw_array)}

    assert np.all(
        [
            idx == spw_dict[jdx]
            for idx, jdx in zip(mir_uv.flex_spw_id_array, uvfits_uv.flex_spw_id_array,)
        ]
    )

    # Now that we've checked, set this things as equivalent
    uvfits_uv.spw_array = mir_uv.spw_array
    uvfits_uv.flex_spw_id_array = mir_uv.flex_spw_id_array

    # Check the history first via find
    assert 0 == uvfits_uv.history.find(
        mir_uv.history + "  Read/written with pyuvdata version:"
    )
    mir_uv.history = uvfits_uv.history

    # We have to do a bit of special handling for the phase_center_catalog, because
    # _very_ small errors (like last bit in the mantissa) creep in when passing through
    # the util function transform_sidereal_coords (for mutli-phase-ctr datasets). Verify
    # the two match up in terms of their coordinates
    for cat_name in mir_uv.phase_center_catalog.keys():
        assert np.isclose(
            mir_uv.phase_center_catalog[cat_name]["cat_lat"],
            uvfits_uv.phase_center_catalog[cat_name]["cat_lat"],
        )
        assert np.isclose(
            mir_uv.phase_center_catalog[cat_name]["cat_lon"],
            uvfits_uv.phase_center_catalog[cat_name]["cat_lon"],
        )
    uvfits_uv.phase_center_catalog = mir_uv.phase_center_catalog

    # There's a minor difference between what SMA calculates online for app coords
    # and what pyuvdata calculates, to the tune of ~1 arcsec. Check those values here,
    # then set them equal to one another.
    assert np.all(
        np.abs(mir_uv.phase_center_app_ra - uvfits_uv.phase_center_app_ra) < 1e-5
    )

    assert np.all(
        np.abs(mir_uv.phase_center_app_dec - uvfits_uv.phase_center_app_dec) < 1e-5
    )

    mir_uv._set_app_coords_helper()
    uvfits_uv._set_app_coords_helper()

    # make sure filenames are what we expect
    assert mir_uv.filename == ["sma_test.mir"]
    assert uvfits_uv.filename == ["outtest_mir.uvfits"]
    mir_uv.filename = uvfits_uv.filename
    assert mir_uv == uvfits_uv

    # Since mir is mutli-phase-ctr by default, this should effectively be a no-op
    mir_uv._set_multi_phase_center()

    assert mir_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_read_mir_write_ms(uv_in_ms, future_shapes):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as ms, read back in and check for
    object equality.
    """
    pytest.importorskip("casacore")
    mir_uv, ms_uv, testfile = uv_in_ms

    if future_shapes:
        mir_uv.use_future_array_shapes()

    mir_uv.write_ms(testfile, clobber=True)
    ms_uv.read(testfile)

    # Single integration with 1 phase center = single scan number
    # output in the MS
    assert ms_uv.scan_number_array == np.array([1])

    if future_shapes:
        ms_uv.use_future_array_shapes()

    # There are some minor differences between the values stored by MIR and that
    # calculated by UVData. Since MS format requires these to be calculated on the fly,
    # we calculate them here just to verify that everything is looking okay.
    mir_uv.set_lsts_from_time_array()
    mir_uv._set_app_coords_helper()

    # These reorderings just make sure that data from the two formats are lined up
    # correctly.
    mir_uv.reorder_freqs(spw_order="number")
    ms_uv.reorder_blts()

    # MS doesn't have the concept of an "instrument" name like FITS does, and instead
    # defaults to the telescope name. Make sure that checks out here.
    assert mir_uv.instrument == "SWARM"
    assert ms_uv.instrument == "SMA"
    mir_uv.instrument = ms_uv.instrument

    # Quick check for history here
    assert ms_uv.history != mir_uv.history
    ms_uv.history = mir_uv.history

    # Only MS has extra keywords, verify those look as expected.
    assert ms_uv.extra_keywords == {"DATA_COL": "DATA", "observer": "SMA"}
    assert mir_uv.extra_keywords == {}
    mir_uv.extra_keywords = ms_uv.extra_keywords

    # Make sure the filenames line up as expected.
    assert mir_uv.filename == ["sma_test.mir"]
    assert ms_uv.filename == ["outtest_mir.ms"]
    mir_uv.filename = ms_uv.filename = None

    # Finally, with all exceptions handled, check for equality.
    assert ms_uv.__eq__(mir_uv, allowed_failures=["filename"])


@pytest.mark.filterwarnings("ignore:LST values stored ")
def test_read_mir_write_uvh5(uv_in_uvh5):
    """
    Mir to uvfits loopback test.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    mir_uv, uvh5_uv, testfile = uv_in_uvh5

    mir_uv.write_uvh5(testfile)
    uvh5_uv.read_uvh5(testfile)

    # Check the history first via find
    assert 0 == uvh5_uv.history.find(
        mir_uv.history + "  Read/written with pyuvdata version:"
    )

    # test fails because of updated history, so this is our workaround for now.
    mir_uv.history = uvh5_uv.history

    # make sure filenames are what we expect
    assert mir_uv.filename == ["sma_test.mir"]
    assert uvh5_uv.filename == ["outtest_mir.uvh5"]
    mir_uv.filename = uvh5_uv.filename

    assert mir_uv == uvh5_uv


def test_write_mir(uv_in_uvfits, err_type=NotImplementedError):
    """
    Mir writer test

    Check and make sure that attempts to use the writer return a
    'not implemented' error.
    """
    mir_uv, uvfits_uv, testfile = uv_in_uvfits

    # Check and see if the correct error is raised
    with pytest.raises(err_type):
        mir_uv.write_mir("dummy.mir")


def test_multi_nchan_spw_read(tmp_path):
    """
    Mir to uvfits error test for spws of different sizes.

    Read in Mir files, write out as uvfits, read back in and check for
    object equality.
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_in = UVData()
    uv_in.read_mir(testfile, corrchunk=[0, 1, 2, 3, 4])

    dummyfile = os.path.join(tmp_path, "dummy.mirtest.uvfits")
    with pytest.raises(IndexError):
        uv_in.write_uvfits(dummyfile, spoof_nonessential=True)


def test_read_mir_no_records():
    """
    Mir no-records check

    Make sure that mir correctly handles the case where no matching records are found
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    uv_in = UVData()
    with pytest.raises(ValueError, match="No valid sources selected!"):
        uv_in.read_mir(testfile, isource=-1)

    with pytest.raises(ValueError, match="No valid receivers selected!"):
        uv_in.read_mir(testfile, irec=-1)

    with pytest.raises(ValueError, match="No valid sidebands selected!"):
        uv_in.read_mir(testfile, isb=-156)

    with pytest.raises(ValueError, match="No valid spectral bands selected!"):
        uv_in.read_mir(testfile, corrchunk=999)


def test_read_mir_sideband_select():
    """
    Mir sideband read check

    Make sure that we can read the individual sidebands out of MIR correctly, and then
    stitch them back together as though they were read together from the start.
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_dsb = UVData()
    mir_dsb.read(testfile)
    # Re-order here so that we can more easily compare the two
    mir_dsb.reorder_freqs(channel_order="freq", spw_order="freq")
    # Drop the history
    mir_dsb.history = ""

    mir_lsb = UVData()
    mir_lsb.read(testfile, isb=[0])

    mir_usb = UVData()
    mir_usb.read(testfile, isb=[1])

    mir_recomb = mir_lsb + mir_usb
    # Re-order here so that we can more easily compare the two
    mir_recomb.reorder_freqs(spw_order="freq", channel_order="freq")
    # Drop the history
    mir_recomb.history = ""

    assert mir_dsb == mir_recomb


def test_mir_auto_read(
    err_type=IndexError, err_msg="Could not determine auto-correlation record size!"
):
    """
    Mir read tester

    Make sure that Mir autocorrelations are read correctly
    """
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_data = MirParser(testfile, has_auto=True)
    with pytest.raises(err_type, match=err_msg):
        ac_data = mir_data.scan_auto_data(testfile, nchunks=999)

    ac_data = mir_data.scan_auto_data(testfile)
    assert np.all(ac_data["nchunks"] == 8)

    mir_data.load_data(load_vis=False, load_auto=True)

    # Select the relevant auto records, which should be for spwin 0-3
    auto_data = mir_data.read_auto_data(testfile, ac_data)[:, 0:4, :, :]
    assert np.all(
        np.logical_or(
            auto_data == mir_data.auto_data,
            np.logical_and(np.isnan(auto_data), np.isnan(mir_data.auto_data)),
        )
    )
    mir_data.unload_data()


def test_read_mir_write_ms_flex_pol(mir_data, tmp_path):
    """
    Mir to MS loopback test with flex-pol.

    Read in Mir files, write out as ms, read back in and check for
    object equality.
    """
    pytest.importorskip("casacore")
    testfile = os.path.join(tmp_path, "read_mir_write_ms_flex_pol.ms")

    # Read in the raw data so that we can manipulate it, and make it look like the
    # test data set was recorded with split-tuning
    mir_data.sp_data["gunnLO"][np.isin(mir_data.sp_data["blhid"], [1, 3])] += 30.0
    mir_data.sp_data["gunnLO"][np.isin(mir_data.sp_data["fsky"], [1, 3])] += 30.0

    # Spin up a Mir object, which can be covered into a UVData object,
    # with flex-pol enabled.
    mir_uv = UVData()
    mir_obj = Mir()
    mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)

    # Write out our modified data set
    mir_uv.write_ms(testfile, clobber=True)
    ms_uv = UVData.from_file(testfile)

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
    assert mir_uv.instrument == "SWARM"
    assert ms_uv.instrument == "SMA"
    mir_uv.instrument = ms_uv.instrument

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


def test_inconsistent_sp_records(mir_data, uv_in_ms):
    """
    Test that the MIR object does the right thing w/ inconsistent meta-data.
    """
    sma_mir, _, _ = uv_in_ms

    mir_data.use_sp = mir_data.sp_read["iband"] != 0
    mir_data.sp_read["ipq"][1] = 0
    mir_data.load_data()

    with uvtest.check_warnings(UserWarning, "Per-spectral window metadata differ."):
        mir_uv = UVData()
        mir_obj = Mir()
        mir_obj._init_from_mir_parser(mir_data)
        mir_uv._convert_from_filetype(mir_obj)

    assert mir_uv == sma_mir


def test_inconsistent_bl_records(mir_data, uv_in_ms):
    """
    Test that the MIR object does the right thing w/ inconsistent meta-data.
    """
    sma_mir, _, _ = uv_in_ms

    mir_data.use_sp = mir_data.sp_read["iband"] != 0
    mir_data.bl_read["u"][0] = 0.0
    mir_data.load_data()
    with uvtest.check_warnings(UserWarning, "Per-baseline metadata differ."):
        mir_uv = UVData()
        mir_obj = Mir()
        mir_obj._init_from_mir_parser(mir_data)
        mir_uv._convert_from_filetype(mir_obj)

    assert mir_uv == sma_mir


def test_multi_ipol(mir_data, sma_mir):
    """
    Test that the MIR object does the right thing when different polarization types
    are recorded in the pol code.
    """
    mir_data.use_sp = mir_data.sp_read["iband"] != 0
    mir_data.bl_read["ipol"][:] = mir_data.bl_read["ant1rx"]
    mir_data.load_data()

    mir_uv = UVData()
    mir_obj = Mir()
    mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)

    assert mir_uv == sma_mir


@pytest.mark.parametrize("filetype", ["uvh5", "miriad", "ms", "uvfits"])
@pytest.mark.parametrize("future_shapes", [True, False])
def test_flex_pol_roundtrip(sma_mir_filt, filetype, future_shapes, tmp_path):
    """Test that we can round-trip flex-pol data sets"""
    testfile = os.path.join(tmp_path, "flex_pol_roundtrip." + filetype)
    if filetype == "ms":
        pytest.importorskip("casacore")
    elif filetype == "miriad":
        pytest.importorskip("pyuvdata._miriad")

    if future_shapes:
        sma_mir_filt.use_future_array_shapes()

    sma_mir_filt._make_flex_pol(raise_error=True)

    # sma_mir_filtered._make_flex_pol()
    if filetype == "uvfits":
        sma_mir_filt.write_uvfits(testfile, spoof_nonessential=True)
    else:
        getattr(sma_mir_filt, "write_" + filetype)(testfile)

    test_uv = UVData.from_file(testfile)

    if future_shapes:
        test_uv.use_future_array_shapes()

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
    with pytest.raises(ValueError) as cm:
        sma_mir_filt.select(polarizations=["xx"], freq_chans=[0, 1, 2, 3])

    assert str(cm.value).startswith("No data matching this polarization and frequency")

    sma_mir_filt.select(polarizations=["xx"])

    assert sma_mir_filt.flex_spw_polarization_array is None
    assert np.all(sma_mir_filt.polarization_array == -5)

    sma_mir_filt._make_flex_pol()

    assert np.all(sma_mir_filt.flex_spw_polarization_array == -5)
    assert np.all(sma_mir_filt.polarization_array == 0)


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
    sma_mir_filt.flag_array[:, :, : sma_mir_filt.Nfreqs // 2, :] = True
    sma_mir_filt._make_flex_pol()

    # Since all of the y-pol data is flagged, the flex-pol data should only contain
    # xx visibilities (as recorded in flex_spw_polarization_array).t
    assert np.all(sma_mir_filt.flex_spw_polarization_array == -5)
