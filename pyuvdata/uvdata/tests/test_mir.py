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

from ... import UVData
from ...data import DATA_PATH
from ...uvdata.mir_parser import MirParser
from ...uvdata.mir import Mir


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
    mir_uv = UVData()
    ms_uv = UVData()

    # Read in the raw data so that we can manipulate it, and make it look like the
    # test data set was recorded with split-tuning
    mir_data.sp_data["gunnLO"][np.isin(mir_data.sp_data["blhid"], [1, 3])] += 30.0
    mir_data.sp_data["gunnLO"][np.isin(mir_data.sp_data["fsky"], [1, 3])] += 30.0

    # Spin up a Mir object, which can be covered into a UVData object,
    # with flex-pol enabled.
    mir_obj = Mir()
    mir_obj._init_from_mir_parser(mir_data)
    mir_uv._convert_from_filetype(mir_obj)

    # Write out our modified data set
    mir_uv.write_ms(testfile, clobber=True)
    ms_uv.read(testfile)

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
