# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MS object.

"""
import os
import shutil

import numpy as np
import pytest
from astropy.time import Time

import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.uvdata.ms import MS
from pyuvdata.uvdata.uvdata import _future_array_shapes_warning

pytest.importorskip("casacore")

allowed_failures = "filename"


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def check_members(tar, path):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    return tar.getmembers()


def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    # this is factored this way (splitting out the `check_members` function)
    # to appease bandit.
    tar.extractall(path, members=check_members(tar, path), numeric_owner=numeric_owner)


@pytest.fixture(scope="session")
def nrao_uv_main():
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    uvobj.read(testfile, use_future_array_shapes=True)

    yield uvobj

    del uvobj


@pytest.fixture(scope="function")
def nrao_uv(nrao_uv_main):
    """Make function level NRAO ms object."""
    nrao_ms = nrao_uv_main.copy()
    yield nrao_ms

    # clean up when done
    del nrao_ms

    return


@pytest.fixture(scope="function")
def nrao_uv_legacy():
    """Make function level NRAO ms object, legacy array shapes."""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    uvobj.read(testfile, use_future_array_shapes=False)

    yield uvobj

    del uvobj


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:UVW orientation appears to be flipped,")
@pytest.mark.filterwarnings("ignore:Nants_telescope, antenna_names")
def test_cotter_ms():
    """Test reading in an ms made from MWA data with cotter (no dysco compression)"""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "1102865728_small.ms/")
    uvobj.read(testfile)

    # check that a select on read works
    uvobj2 = UVData()

    with uvtest.check_warnings(
        [UserWarning] * 3 + [DeprecationWarning],
        match=[
            "Warning: select on read keyword set",
            (
                "telescope_location are not set or are being overwritten. Using known "
                "values for MWA."
            ),
            "UVW orientation appears to be flipped,",
            _future_array_shapes_warning,
        ],
    ):
        uvobj2.read(testfile, freq_chans=np.arange(2))
    uvobj.select(freq_chans=np.arange(2))
    assert uvobj == uvobj2
    del uvobj


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("telescope_frame", ["itrs", "mcmf"])
def test_read_nrao_loopback(tmp_path, nrao_uv, telescope_frame):
    """Test reading in a CASA tutorial ms file and looping it through write_ms."""
    uvobj = nrao_uv

    if telescope_frame == "mcmf":
        pytest.importorskip("lunarsky")
        enu_antpos, _ = uvobj.get_ENU_antpos()
        latitude, longitude, altitude = uvobj.telescope_location_lat_lon_alt
        uvobj._telescope_location.frame = "mcmf"
        uvobj.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
        new_full_antpos = uvutils.ECEF_from_ENU(
            enu=enu_antpos,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            frame="mcmf",
        )
        uvobj.antenna_positions = new_full_antpos - uvobj.telescope_location
        uvobj.set_lsts_from_time_array()
        uvobj.set_uvws_from_antenna_positions()
        uvobj._set_app_coords_helper()
        uvobj.check()

    expected_extra_keywords = ["DATA_COL", "observer"]

    assert sorted(expected_extra_keywords) == sorted(uvobj.extra_keywords.keys())

    testfile = os.path.join(tmp_path, "ms_testfile.ms")

    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Writing in the MS file that the units of the data are",
            "The uvw_array does not match the expected values",
        ],
    ):
        uvobj.write_ms(testfile)

    uvobj2 = UVData()
    uvobj2.read_ms(testfile, use_future_array_shapes=True)

    # also update filenames
    assert uvobj.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    assert uvobj2.filename == ["ms_testfile.ms"]
    uvobj.filename = uvobj2.filename

    assert uvobj._telescope_location.frame == uvobj2._telescope_location.frame

    # Test that the scan numbers are equal
    assert (uvobj.scan_number_array == uvobj2.scan_number_array).all()

    assert uvobj == uvobj2

    # test that the clobber keyword works by rewriting
    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Writing in the MS file that the units of the data are",
            "The uvw_array does not match the expected values",
        ],
    ):
        uvobj.write_ms(testfile, clobber=True)


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_lwa(tmp_path):
    """Test reading in an LWA ms file."""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "lwasv_cor_58342_05_00_14.ms.tar.gz")
    expected_extra_keywords = ["DATA_COL", "observer"]

    import tarfile

    with tarfile.open(testfile) as tf:
        new_filename = os.path.join(tmp_path, tf.getnames()[0])

        safe_extract(tf, path=tmp_path)

    uvobj.read(new_filename, file_type="ms", use_future_array_shapes=True)
    assert sorted(expected_extra_keywords) == sorted(uvobj.extra_keywords.keys())

    assert uvobj.history == uvobj.pyuvdata_version_str

    # delete the untarred folder
    shutil.rmtree(new_filename)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:telescope_location are not set")
def test_no_spw():
    """Test reading in a PAPER ms converted by CASA from a uvfits with no spw axis."""
    uvobj = UVData()
    testfile_no_spw = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAAM.ms")
    uvobj.read(testfile_no_spw, use_future_array_shapes=True)
    del uvobj


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_spwsupported():
    """Test reading in an ms file with multiple spws."""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1scan.ms")
    uvobj.read(testfile, use_future_array_shapes=True)

    assert uvobj.Nspws == 2


@pytest.mark.filterwarnings("ignore:Coordinate reference frame not detected,")
@pytest.mark.filterwarnings("ignore:UVW orientation appears to be flipped,")
@pytest.mark.filterwarnings("ignore:telescope_location are not set")
def test_extra_pol_setup(tmp_path):
    """Test reading in an ms file with extra polarization setups (not used in data)."""
    uvobj = UVData()
    testfile = os.path.join(
        DATA_PATH, "X5707_1spw_1scan_10chan_1time_1bl_noatm.ms.tar.gz"
    )

    import tarfile

    with tarfile.open(testfile) as tf:
        new_filename = os.path.join(tmp_path, tf.getnames()[0])

        safe_extract(tf, path=tmp_path)

    uvobj.read(new_filename, file_type="ms", use_future_array_shapes=True)

    # delete the untarred folder
    shutil.rmtree(new_filename)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:The older phase attributes")
def test_read_ms_read_uvfits(nrao_uv, casa_uvfits):
    """
    Test that a uvdata object instantiated from an ms file created with CASA's
    importuvfits is equal to a uvdata object instantiated from the original
    uvfits file (tests equivalence with importuvfits in uvdata up to issues around the
    direction of the uvw array).
    Since the history is missing from the ms file, this test sets both uvdata
    histories to identical empty strings before comparing them.
    """
    ms_uv = nrao_uv.copy()
    uvfits_uv = casa_uvfits.copy()
    # set histories to identical blank strings since we do not expect
    # them to be the same anyways.
    ms_uv.history = ""
    uvfits_uv.history = ""

    # the objects won't be equal because uvfits adds some optional parameters
    # and the ms sets default antenna diameters even though the uvfits file
    # doesn't have them
    assert uvfits_uv != ms_uv
    uvfits_uv.integration_time = ms_uv.integration_time

    # The uvfits was written by CASA, which adds one to all the antenna numbers relative
    # to the measurement set. Adjust those:
    uvfits_uv.antenna_numbers = uvfits_uv.antenna_numbers - 1
    uvfits_uv.ant_1_array = uvfits_uv.ant_1_array - 1
    uvfits_uv.ant_2_array = uvfits_uv.ant_2_array - 1
    uvfits_uv.baseline_array = uvfits_uv.antnums_to_baseline(
        uvfits_uv.ant_1_array, uvfits_uv.ant_2_array
    )

    # also need to adjust phase_center_catalogs
    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=ms_uv.phase_center_catalog
    )

    # they are equal if only required parameters are checked:
    # scan numbers only defined for the MS
    assert uvfits_uv.__eq__(ms_uv, check_extra=False, allowed_failures=allowed_failures)

    # set those parameters to none to check that the rest of the objects match
    ms_uv.antenna_diameters = None
    uvfits_required_extra = ["dut1", "earth_omega", "gst0", "rdate", "timesys"]
    for p in uvfits_uv.extra():
        fits_param = getattr(uvfits_uv, p)
        ms_param = getattr(ms_uv, p)
        if fits_param.name in uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uvfits_uv, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uvfits_uv.extra_keywords = {}
    ms_uv.extra_keywords = {}

    # also update filenames
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    uvfits_uv.filename = ms_uv.filename

    # propagate scan numbers to the uvfits, ONLY for comparison
    uvfits_uv.scan_number_array = ms_uv.scan_number_array

    assert uvfits_uv == ms_uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_ms_write_uvfits(nrao_uv, tmp_path):
    """
    read ms, write uvfits test.
    Read in ms file, write out as uvfits, read back in and check for
    object equality.
    """
    ms_uv = nrao_uv
    uvfits_uv = UVData()
    testfile = os.path.join(tmp_path, "outtest.uvfits")
    ms_uv.write_uvfits(testfile)
    uvfits_uv.read(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uvfits_uv.filename == ["outtest.uvfits"]
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    uvfits_uv.filename = ms_uv.filename

    # propagate scan numbers to the uvfits, ONLY for comparison
    uvfits_uv.scan_number_array = ms_uv.scan_number_array

    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific paramters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(ms_uv, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=ms_uv.phase_center_catalog
    )
    assert uvfits_uv == ms_uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_ms_write_miriad(nrao_uv, tmp_path):
    """
    read ms, write miriad test.
    Read in ms file, write out as miriad, read back in and check for
    object equality.
    """
    pytest.importorskip("pyuvdata._miriad")
    ms_uv = nrao_uv
    miriad_uv = UVData()
    testfile = os.path.join(tmp_path, "outtest_miriad")
    with uvtest.check_warnings(
        UserWarning,
        [
            (
                "The uvw_array does not match the expected values given the antenna"
                " positions."
            ),
            "writing default values for restfreq, vsource, veldop, jyperk, and systemp",
        ],
    ):
        ms_uv.write_miriad(testfile)
    miriad_uv.read(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    miriad_uv.filename = ms_uv.filename

    # propagate scan numbers to the miriad uvdata, ONLY for comparison
    miriad_uv.scan_number_array = ms_uv.scan_number_array

    assert miriad_uv == ms_uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:The older phase attributes")
@pytest.mark.parametrize("axis", [None, "freq"])
def test_multi_files(casa_uvfits, axis):
    """
    Reading multiple files at once.
    """
    uv_full = casa_uvfits.copy()

    # Ensure the scan numbers are defined for the comparison
    uv_full._set_scan_numbers()

    uv_multi = UVData()
    testfile1 = os.path.join(DATA_PATH, "multi_1.ms")
    testfile2 = os.path.join(DATA_PATH, "multi_2.ms")

    # It seems that these two files were made using the CASA importuvfits task, but the
    # history tables are missing, so we can't infer that from the history. This means
    # that the uvws are not flipped/data is not conjugated as they should be. Fix that.

    filesread = [testfile1, testfile2]
    # test once as list and once as an array
    if axis is None:
        filesread = np.array(filesread)

    uv_multi.read(filesread, axis=axis, use_future_array_shapes=True)

    # histories are different because of combining along freq. axis
    # replace the history
    uv_multi.history = uv_full.history

    # the objects won't be equal because uvfits adds some optional parameters
    # and the ms sets default antenna diameters even though the uvfits file
    # doesn't have them
    assert uv_multi != uv_full
    # The uvfits was written by CASA, which adds one to all the antenna numbers relative
    # to the measurement set. Adjust those:
    uv_full.antenna_numbers = uv_full.antenna_numbers - 1
    uv_full.ant_1_array = uv_full.ant_1_array - 1
    uv_full.ant_2_array = uv_full.ant_2_array - 1
    uv_full.baseline_array = uv_full.antnums_to_baseline(
        uv_full.ant_1_array, uv_full.ant_2_array
    )

    uv_full._consolidate_phase_center_catalogs(
        reference_catalog=uv_multi.phase_center_catalog
    )

    # now they are equal if only required parameters are checked:
    assert uv_multi.__eq__(uv_full, check_extra=False)

    # set those parameters to none to check that the rest of the objects match
    uv_multi.antenna_diameters = None

    uvfits_required_extra = ["dut1", "earth_omega", "gst0", "rdate", "timesys"]
    for p in uv_full.extra():
        fits_param = getattr(uv_full, p)
        ms_param = getattr(uv_multi, p)
        if fits_param.name in uvfits_required_extra and ms_param.value is None:
            fits_param.value = None
            setattr(uv_full, p, fits_param)

    # extra keywords are also different, set both to empty dicts
    uv_full.extra_keywords = {}
    uv_multi.extra_keywords = {}

    # make sure filenames are what we expect
    assert set(uv_multi.filename) == {"multi_1.ms", "multi_2.ms"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv_multi.filename = uv_full.filename
    uv_multi._filename.form = (1,)

    assert uv_multi.__eq__(uv_full, allowed_failures=allowed_failures)
    del uv_full
    del uv_multi


def test_bad_col_name():
    """
    Test error with invalid column name.
    """
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")

    with pytest.raises(ValueError, match="Invalid data_column value supplied"):
        uvobj.read(testfile, data_column="FOO", use_future_array_shapes=True)


@pytest.mark.parametrize("check_warning", [True, False])
@pytest.mark.parametrize(
    "frame,errtype,msg",
    (
        ["JNAT", NotImplementedError, "Support for the JNAT frame is not yet"],
        ["AZEL", NotImplementedError, "Support for the AZEL frame is not yet"],
        ["GALACTIC", NotImplementedError, "Support for the GALACTIC frame is not yet"],
        ["ABC", ValueError, "The coordinate frame ABC is not one of the supported"],
        ["123", ValueError, "The coordinate frame 123 is not one of the supported"],
    ),
)
def test_parse_casa_frame_ref_errors(check_warning, frame, errtype, msg):
    """
    Test errors with matching CASA frames to astropy frame/epochs
    """
    uvobj = MS()
    if check_warning:
        with uvtest.check_warnings(UserWarning, match=msg):
            uvobj._parse_casa_frame_ref(frame, raise_error=False)
    else:
        with pytest.raises(errtype) as cm:
            uvobj._parse_casa_frame_ref(frame)
        assert str(cm.value).startswith(msg)


@pytest.mark.parametrize("check_warning", [True, False])
@pytest.mark.parametrize(
    "frame,epoch,msg",
    (
        ["fk5", 1991.1, "Frame fk5 (epoch 1991.1) does not have a corresponding match"],
        ["fk4", 1991.1, "Frame fk4 (epoch 1991.1) does not have a corresponding match"],
        ["icrs", 2021.0, "Frame icrs (epoch 2021) does not have a corresponding"],
    ),
)
def test_parse_pyuvdata_frame_ref_errors(check_warning, frame, epoch, msg):
    """
    Test errors with matching CASA frames to astropy frame/epochs
    """
    uvobj = MS()
    if check_warning:
        with uvtest.check_warnings(UserWarning, match=msg):
            uvobj._parse_pyuvdata_frame_ref(frame, epoch, raise_error=False)
    else:
        with pytest.raises(ValueError) as cm:
            uvobj._parse_pyuvdata_frame_ref(frame, epoch)
        assert str(cm.value).startswith(msg)


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_history_lesson(sma_mir, tmp_path):
    """
    Test that the MS reader/writer can parse complex history
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_history_lesson.ms")
    sma_mir.history = (
        "Line 1.\nBegin measurement set history\nAPP_PARAMS;CLI_COMMAND;"
        "APPLICATION;MESSAGE;OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
        "End measurement set history.\nLine 2.\n"
    )
    sma_mir.write_ms(testfile)

    tb_hist = tables.table(testfile + "/HISTORY", readonly=False, ack=False)
    tb_hist.addrows()
    for col in tb_hist.colnames():
        tb_hist.putcell(col, tb_hist.nrows() - 1, tb_hist.getcell(col, 0))
    tb_hist.putcell("ORIGIN", 1, "DUMMY")
    tb_hist.putcell("APPLICATION", 1, "DUMMY")
    tb_hist.putcell("TIME", 1, 0.0)
    tb_hist.putcell("MESSAGE", 2, "Line 3.")
    tb_hist.close()

    ms_uv.read(testfile, use_future_array_shapes=True)

    assert ms_uv.history.startswith(
        "Line 1.\nBegin measurement set history\nAPP_PARAMS;CLI_COMMAND;APPLICATION;"
        "MESSAGE;OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n;;DUMMY;Line 2.;0;-1;"
        "DUMMY;INFO;0.0\nEnd measurement set history.\nLine 3.\n  Read/written with "
        "pyuvdata version:"
    )

    tb_hist = tables.table(os.path.join(testfile, "HISTORY"), ack=False, readonly=False)
    tb_hist.rename(os.path.join(testfile, "FORGOTTEN"))
    tb_hist.close()

    ms_uv.read(testfile, use_future_array_shapes=True)
    assert ms_uv.history.startswith("  Read/written with pyuvdata version:")


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_multi_spw_data_variation(sma_mir, tmp_path):
    """
    Test that the MS writer/reader appropriately reads in a single-source data set
    as non-multi-phase if it can be, even if the original data set was.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_multi_spw_data_variation.ms")

    sma_mir.write_ms(testfile)

    tb_main = tables.table(testfile, readonly=False, ack=False)
    tb_main.putcol("EXPOSURE", np.arange(sma_mir.Nblts * sma_mir.Nspws) + 1.0)
    tb_main.close()

    with pytest.raises(ValueError, match="Column EXPOSURE appears to vary on between"):
        ms_uv.read(testfile)

    with uvtest.check_warnings(
        UserWarning, match="Column EXPOSURE appears to vary on between windows, "
    ):
        ms_uv.read(testfile, raise_error=False, use_future_array_shapes=True)

    # Check that the values do indeed match the first entry in the catalog
    assert np.all(ms_uv.integration_time == np.array([1.0]))


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_ms_phasing(sma_mir, future_shapes, tmp_path):
    """
    Test that the MS writer can appropriately handle unphased data sets.
    """
    if not future_shapes:
        sma_mir.use_current_array_shapes()

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_phasing.ms")

    sma_mir.unproject_phase()

    with pytest.raises(ValueError, match="The data are unprojected."):
        sma_mir.write_ms(testfile)

    sma_mir.write_ms(testfile, force_phase=True)

    ms_uv.read(testfile, use_future_array_shapes=True)

    assert np.allclose(ms_uv.phase_center_app_ra, ms_uv.lst_array)
    assert np.allclose(
        ms_uv.phase_center_app_dec, ms_uv.telescope_location_lat_lon_alt[0]
    )


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_ms_single_chan(sma_mir, future_shapes, tmp_path):
    """
    Make sure that single channel writing/reading work as expected
    """
    if not future_shapes:
        sma_mir.use_current_array_shapes()

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_single_chan.ms")

    sma_mir.select(freq_chans=0)
    sma_mir.write_ms(testfile)
    sma_mir.set_lsts_from_time_array()
    sma_mir._set_app_coords_helper()

    with pytest.raises(ValueError, match="No valid data available in the MS file."):
        ms_uv.read(testfile)

    ms_uv.read(
        testfile, ignore_single_chan=False, use_future_array_shapes=future_shapes
    )

    # Easiest way to check that everything worked is to just check for equality, but
    # the MS file is single-spw, single-field, so we have a few things we need to fix

    cat_id = list(sma_mir.phase_center_catalog.keys())[0]
    cat_name = sma_mir.phase_center_catalog[cat_id]["cat_name"]
    ms_uv._update_phase_center_id(list(ms_uv.phase_center_catalog.keys())[0], cat_id)
    ms_uv.phase_center_catalog[cat_id]["cat_name"] = cat_name
    ms_uv.phase_center_catalog[cat_id]["info_source"] = "file"

    # Next, turn on flex-spw
    ms_uv._set_flex_spw()
    if not future_shapes:
        ms_uv.channel_width = np.array([ms_uv.channel_width])
    ms_uv.flex_spw_id_array = ms_uv.spw_array.copy()

    # Finally, take care of the odds and ends
    ms_uv.extra_keywords = {}
    ms_uv.history = sma_mir.history
    ms_uv.filename = sma_mir.filename
    ms_uv.instrument = sma_mir.instrument
    ms_uv.reorder_blts()

    # propagate scan numbers to the uvfits, ONLY for comparison
    sma_mir.scan_number_array = ms_uv.scan_number_array

    assert ms_uv == sma_mir


@pytest.mark.filterwarnings("ignore:pamatten in extra_keywords is a list, array")
@pytest.mark.filterwarnings("ignore:psys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:psysattn in extra_keywords is a list, array or")
@pytest.mark.filterwarnings("ignore:ambpsys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:bfmask in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.parametrize("multi_frame", [True, False])
def test_ms_scannumber_multiphasecenter(tmp_path, multi_frame):
    """
    Make sure that single channel writing/reading work as expected
    """
    carma_file = os.path.join(DATA_PATH, "carma_miriad")
    testfile = os.path.join(tmp_path, "carma_out.ms")

    miriad_uv = UVData()

    # Copied in from test_miriad.py::test_read_carma_miriad_write_ms
    with uvtest.check_warnings(
        UserWarning,
        [
            (
                "Altitude is not present in Miriad file, "
                "using known location values for SZA."
            ),
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
            "pamatten in extra_keywords is a list, array or dict",
            "psys in extra_keywords is a list, array or dict",
            "psysattn in extra_keywords is a list, array or dict",
            "ambpsys in extra_keywords is a list, array or dict",
            "bfmask in extra_keywords is a list, array or dict",
        ],
    ):
        miriad_uv.read(carma_file, use_future_array_shapes=True)

    # MIRIAD is missing these in the file, so we'll fill it in here.
    miriad_uv.antenna_diameters = np.zeros(miriad_uv.Nants_telescope)
    miriad_uv.antenna_diameters[:6] = 10.0
    miriad_uv.antenna_diameters[15:] = 3.5

    # We need to recalculate app coords here for one source ("NOISE"), which was
    # not actually correctly calculated in the online CARMA system (long story). Since
    # the MS format requires recalculating apparent coords after read in, we'll
    # calculate them here just to verify that everything matches.
    miriad_uv._set_app_coords_helper()

    if multi_frame:
        cat_id = miriad_uv._look_for_name("NOISE")
        ra_use = miriad_uv.phase_center_catalog[cat_id[0]]["cat_lon"][0]
        dec_use = miriad_uv.phase_center_catalog[cat_id[0]]["cat_lat"][0]
        with pytest.raises(
            ValueError,
            match="lon parameter must be a single value for cat_type sidereal",
        ):
            miriad_uv.phase(
                miriad_uv.phase_center_catalog[cat_id[0]]["cat_lon"],
                dec_use,
                cat_name="foo",
                phase_frame="icrs",
                select_mask=miriad_uv.phase_center_id_array == cat_id[0],
            )

        with pytest.raises(
            ValueError,
            match="lat parameter must be a single value for cat_type sidereal",
        ):
            miriad_uv.phase(
                ra_use,
                miriad_uv.phase_center_catalog[cat_id[0]]["cat_lat"],
                cat_name="foo",
                phase_frame="icrs",
                select_mask=miriad_uv.phase_center_id_array == cat_id[0],
            )

        with uvtest.check_warnings(
            UserWarning,
            match=[
                "The entry name NOISE is not unique",
                "The provided name NOISE is already used",
            ],
        ):
            miriad_uv.phase(
                ra_use,
                dec_use,
                cat_name="NOISE",
                phase_frame="icrs",
                select_mask=miriad_uv.phase_center_id_array == cat_id[0],
            )
    miriad_uv.write_ms(testfile)

    # Check on the scan number grouping based on consecutive integrations per phase
    # center

    # Read back in as MS. Should have 3 scan numbers defined.
    ms_uv = UVData()
    ms_uv.read(testfile, use_future_array_shapes=True)

    assert np.unique(ms_uv.scan_number_array).size == 3
    assert (np.unique(ms_uv.scan_number_array) == np.array([1, 2, 3])).all()

    # The scan numbers should match the phase center IDs, offset by 1
    # so that the scan numbers start with 1, not 0.
    assert ((miriad_uv.phase_center_id_array == (ms_uv.scan_number_array - 1))).all()


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_extra_data_descrip(sma_mir, tmp_path):
    """
    Make sure that data sets can be read even if the main table doesn't have data
    for a particular listed spectral window in the DATA_DESCRIPTION table.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_extra_data_descrip.ms")

    sma_mir.write_ms(testfile)

    tb_dd = tables.table(
        os.path.join(testfile, "DATA_DESCRIPTION"), ack=False, readonly=False
    )
    tb_dd.addrows()
    for col in tb_dd.colnames():
        tb_dd.putcell(col, tb_dd.nrows() - 1, tb_dd.getcell(col, 0))
    tb_dd.close()

    ms_uv.read(testfile, ignore_single_chan=False, use_future_array_shapes=True)
    cat_id = list(sma_mir.phase_center_catalog.keys())[0]
    cat_name = sma_mir.phase_center_catalog[cat_id]["cat_name"]
    ms_uv._update_phase_center_id(list(ms_uv.phase_center_catalog.keys())[0], cat_id)
    ms_uv.phase_center_catalog[cat_id]["cat_name"] = cat_name
    ms_uv.phase_center_catalog[cat_id]["info_source"] = "file"

    # There are some minor differences between the values stored by MIR and that
    # calculated by UVData. Since MS format requires these to be calculated on the fly,
    # we calculate them here just to verify that everything is looking okay.
    sma_mir.set_lsts_from_time_array()
    sma_mir._set_app_coords_helper()

    # These reorderings just make sure that data from the two formats are lined up
    # correctly.
    sma_mir.reorder_freqs(spw_order="number")
    ms_uv.reorder_blts()

    # Fix the remaining differences between the two objects, all of which are expected
    sma_mir.instrument = sma_mir.telescope_name
    ms_uv.history = sma_mir.history
    sma_mir.extra_keywords = ms_uv.extra_keywords
    sma_mir.filename = ms_uv.filename = None

    # propagate scan numbers to the miriad uvdata, ONLY for comparison
    sma_mir.scan_number_array = ms_uv.scan_number_array

    # Finally, with all exceptions handled, check for equality.
    assert ms_uv == sma_mir


@pytest.mark.parametrize("onewin", [True, False])
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_weights(sma_mir, tmp_path, onewin):
    """
    Test that the MS writer/reader appropriately handles data when the
    WEIGHT_SPECTRUM column is missing or bypassed.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_weights.ms")
    if onewin:
        sma_mir.select(freq_chans=np.arange(16384))

    sma_mir.nsample_array[0, :, :] = np.tile(
        np.arange(sma_mir.Nfreqs / sma_mir.Nspws), (sma_mir.Npols, sma_mir.Nspws)
    ).T
    sma_mir.write_ms(testfile)

    tb_main = tables.table(testfile, readonly=False, ack=False)
    tb_main.removecols("WEIGHT_SPECTRUM")
    tb_main.close()

    ms_uv.read(testfile, use_future_array_shapes=True)

    # Check that the values do indeed match expected (median) value
    assert np.all(ms_uv.nsample_array == np.median(sma_mir.nsample_array))

    ms_uv.read(testfile, read_weights=False, use_future_array_shapes=True)
    # Check that the values do indeed match expected (median) value
    assert np.all(ms_uv.nsample_array == 1.0)


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.parametrize(
    "badcol,badval,errtype,msg",
    (
        [None, None, IOError, "Thisisnofile.ms not found"],
        ["DATA_DESC_ID", [1000] * 8, ValueError, "No valid data available in the MS"],
        ["ARRAY_ID", np.arange(8), ValueError, "This file appears to have multiple"],
        ["DATA_COL", None, ValueError, "Invalid data_column value supplied."],
        [
            "TEL_LOC",
            None,
            ValueError,
            (
                "Telescope frame in file is abc. Only 'itrs' and 'mcmf' are currently "
                "supported."
            ),
        ],
        [
            "timescale",
            None,
            ValueError,
            "This file has a timescale that is not supported by astropy.",
        ],
    ),
)
def test_ms_reader_errs(sma_mir, tmp_path, badcol, badval, errtype, msg):
    """
    Test whether the reader throws an appripropriate errors on read.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_reader_errs.ms")
    sma_mir.write_ms(testfile)

    data_col = "DATA"

    if badcol is None:
        testfile = "Thisisnofile.ms"
    elif badcol == "DATA_COL":
        data_col = badval
    elif badcol == "TEL_LOC":
        tb_obs = tables.table(
            os.path.join(testfile, "OBSERVATION"), ack=False, readonly=False
        )
        tb_obs.removecols("TELESCOPE_LOCATION")
        tb_obs.putcol("TELESCOPE_NAME", "ABC")
        tb_obs.close()
        tb_ant = tables.table(
            os.path.join(testfile, "ANTENNA"), ack=False, readonly=False
        )
        tb_ant.putcolkeyword("POSITION", "MEASINFO", {"type": "position", "Ref": "ABC"})
        tb_ant.close()
    elif badcol == "timescale":
        tb_main = tables.table(testfile, ack=False, readonly=False)
        tb_main.putcolkeyword("TIME", "MEASINFO", {"Ref": "GMST"})
    else:
        tb_main = tables.table(testfile, ack=False, readonly=False)
        tb_main.putcol(badcol, badval)
        tb_main.close()

    with pytest.raises(errtype, match=msg):
        ms_uv.read(testfile, data_column=data_col, file_type="ms")

    if badcol == "timescale":
        with uvtest.check_warnings(UserWarning, match=msg):
            ms_uv.read(
                testfile,
                data_column=data_col,
                file_type="ms",
                raise_error=False,
                use_future_array_shapes=True,
            )
        assert ms_uv._time_array == sma_mir._time_array


def test_antenna_diameter_handling(hera_uvh5, tmp_path):
    uv_obj = hera_uvh5

    uv_obj.antenna_diameters = np.asarray(uv_obj.antenna_diameters, dtype=">f4")

    test_file = os.path.join(tmp_path, "dish_diameter_out.ms")
    with uvtest.check_warnings(
        UserWarning, match="Writing in the MS file that the units of the data are"
    ):
        uv_obj.write_ms(test_file, force_phase=True)

    uv_obj2 = UVData.from_file(test_file, use_future_array_shapes=True)

    # MS write/read adds some stuff to history & extra keywords
    uv_obj2.history = uv_obj.history
    uv_obj2.extra_keywords = uv_obj.extra_keywords

    uv_obj2._consolidate_phase_center_catalogs(
        reference_catalog=uv_obj.phase_center_catalog
    )
    assert uv_obj2.__eq__(uv_obj, allowed_failures=allowed_failures)


def test_no_source(sma_mir, tmp_path):
    uv = UVData()
    uv2 = UVData()
    filename = os.path.join(tmp_path, "no_source.ms")

    sma_mir.write_ms(filename)

    uv.read(filename, use_future_array_shapes=True)

    shutil.rmtree(os.path.join(filename, "SOURCE"))
    uv2.read(filename, use_future_array_shapes=True)

    assert uv == uv2


@pytest.mark.filterwarnings("ignore:Nants_telescope, antenna_names")
@pytest.mark.filterwarnings("ignore:UVW orientation appears to be flipped")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
def test_timescale_handling():
    ut1_file = os.path.join(DATA_PATH, "1090008640_birli_pyuvdata.ms")

    uvobj = UVData.from_file(ut1_file, use_future_array_shapes=True)
    assert (
        np.round(Time(uvobj.time_array[0], format="jd").gps, decimals=5) == 1090008642.0
    )


def test_ms_bad_history(sma_mir, tmp_path):
    # Adding history string that rlbryne (Issue 1324) reported as throwing a particular
    # bug on write.
    sma_mir.history = (
        "Begin measurement set history\nAPP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE;"
        "OBJECT_ID;OBSERVATION_ID;ORIG\nIN;PRIORITY;TIME\nEnd measurement set history."
        "\n  Read/written with pyuvdata version: 2.1.2. Combined data along baselin\ne-"
        "time axis using pyuvdata. Combined data along baseline-time axis using\n "
        "pyuvdata. Combined data along baseline-time axis using pyuvdata. Combin\ned "
        "data along baseline-time axis using pyuvdata. Combined data along bas\neline-"
        "time axis using pyuvdata. Combined data along baseline-time axis u\nsing "
        "pyuvdata. Combined data along baseline-time axis using pyuvdata. Co\nmbined "
        "data along baseline-time axis using pyuvdata. Combined data along\n baseline-"
        "time axis using pyuvdata. Combined data along baseline-time ax\nis using "
        "pyuvdata. Combined data along baseline-time axis using pyuvdata\n. Combined "
        "data along baseline-time axis using pyuvdata. Combined data a\nlong baseline-"
        "time axis using pyuvdata. Combined data along baseline-tim\ne axis using "
        "pyuvdata. Combined data along baseline-time axis using pyuv\ndata. Combined "
        "data along baseline-time axis using pyuvdata. Combined da\nta along baseline-"
        "time axis using pyuvdata. Combined data along baseline\n-time axis using "
        "pyuvdata. Combined data along baseline-time axis using\npyuvdata. Combined "
        "data along baseline-time axis using pyuvdata. Combine\nd data along baseline-"
        "time axis using pyuvdata. Combined data along base\nline-time axis using "
        "pyuvdata. Combined data along baseline-time axis us\ning pyuvdata. Combined "
        "data along baseline-time axis using pyuvdata. Com\nbined data along baseline-"
        "time axis using pyuvdata. Combined data along\nbaseline-time axis using "
        "pyuvdata. Combined data along baseline-time axi\ns using pyuvdata. Combined "
        "data along baseline-time axis using pyuvdata.\n Combined data along baseline-"
        "time axis using pyuvdata.\nFlagged with pyuvdata.utils.apply_uvflags.  "
        "Downselected to specific tim\nes using pyuvdata.\n  Read/written with pyuvdata"
        "version: 2.3.2.\nCalibrated with pyuvdata.utils.uvcalibrate."
    )

    filename = os.path.join(tmp_path, "bad_history.ms")
    with uvtest.check_warnings(
        UserWarning, match="Failed to parse prior history of MS file,"
    ):
        sma_mir.write_ms(filename)

    # Make sure the history is actually preserved correctly.
    sma_ms = UVData.from_file(filename, use_future_array_shapes=True)
    assert sma_mir.history in sma_ms.history


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_flip_conj(nrao_uv, tmp_path):
    filename = os.path.join(tmp_path, "flip_conj.ms")
    nrao_uv.set_uvws_from_antenna_positions()

    with uvtest.check_warnings(
        UserWarning, match="Writing in the MS file that the units of the data are unca"
    ):
        nrao_uv.write_ms(filename, flip_conj=True)

    with uvtest.check_warnings(
        UserWarning, match="UVW orientation appears to be flipped,"
    ):
        uv = UVData.from_file(filename, use_future_array_shapes=True)

    assert nrao_uv == uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_flip_conj_multispw(sma_mir, tmp_path):
    sma_mir._set_app_coords_helper()
    filename = os.path.join(tmp_path, "flip_conj_multispw.ms")

    sma_mir.write_ms(filename, flip_conj=True)
    with uvtest.check_warnings(
        UserWarning, match="UVW orientation appears to be flipped,"
    ):
        ms_uv = UVData.from_file(filename, use_future_array_shapes=True)

    # MS doesn't have the concept of an "instrument" name like FITS does, and instead
    # defaults to the telescope name. Make sure that checks out here.
    assert sma_mir.instrument == "SWARM"
    assert ms_uv.instrument == "SMA"
    sma_mir.instrument = ms_uv.instrument

    # Quick check for history here
    assert ms_uv.history != sma_mir.history
    ms_uv.history = sma_mir.history

    # Only MS has extra keywords, verify those look as expected.
    assert ms_uv.extra_keywords == {"DATA_COL": "DATA", "observer": "SMA"}
    assert sma_mir.extra_keywords == {}
    sma_mir.extra_keywords = ms_uv.extra_keywords

    # Make sure the filenames line up as expected.
    assert sma_mir.filename == ["sma_test.mir"]
    assert ms_uv.filename == ["flip_conj_multispw.ms"]
    sma_mir.filename = ms_uv.filename = None

    assert sma_mir == ms_uv


@pytest.mark.filterwarnings(
    "ignore:Writing in the MS file that the units of the data are"
)
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
def test_read_ms_write_ms_legacy(nrao_uv, nrao_uv_legacy, tmp_path):
    """
    write ms from future and legacy array shapes.
    """
    testfile_l = os.path.join(tmp_path, "outtest_legacy.ms")
    testfile_f = os.path.join(tmp_path, "outtest_future.ms")

    nrao_uv.write_ms(testfile_l)
    nrao_uv_legacy.write_ms(testfile_f)
