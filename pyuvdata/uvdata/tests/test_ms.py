# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MS object.

"""
import pytest
import os
import shutil
import numpy as np

from pyuvdata import UVData
from pyuvdata.uvdata.ms import MS
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
from ..uvfits import UVFITS

pytest.importorskip("casacore")

allowed_failures = "filename"


@pytest.fixture(scope="session")
def nrao_uv_main():
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    uvobj.read(testfile)

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


@pytest.fixture(scope="session")
def mir_uv_main():
    mir_uv = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    mir_uv.read(testfile)

    yield mir_uv


@pytest.fixture(scope="function")
def mir_uv(mir_uv_main):
    """Make a function level copy of a MIR data object"""
    mir_uv = mir_uv_main.copy()

    yield mir_uv


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:UVW orientation appears to be flipped,")
@pytest.mark.filterwarnings("ignore:telescope_location is not set")
def test_cotter_ms():
    """Test reading in an ms made from MWA data with cotter (no dysco compression)"""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "1102865728_small.ms/")
    uvobj.read(testfile)

    # check that a select on read works
    uvobj2 = UVData()

    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Warning: select on read keyword set",
            "telescope_location is not set. Using known values for MWA.",
            "UVW orientation appears to be flipped,",
        ],
    ):
        uvobj2.read(testfile, freq_chans=np.arange(2))
    uvobj.select(freq_chans=np.arange(2))
    assert uvobj == uvobj2
    del uvobj


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_nrao_loopback(tmp_path, nrao_uv):
    """Test reading in a CASA tutorial ms file and looping it through write_ms."""
    uvobj = nrao_uv
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
    uvobj2.read_ms(testfile)

    # also update filenames
    assert uvobj.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    assert uvobj2.filename == ["ms_testfile.ms"]
    uvobj.filename = uvobj2.filename

    # Test that the scan numbers are equal
    assert (uvobj.scan_number_array == uvobj2.scan_number_array).all()

    assert uvobj == uvobj2


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
        tf.extractall(path=tmp_path)

    uvobj.read(new_filename, file_type="ms")
    assert sorted(expected_extra_keywords) == sorted(uvobj.extra_keywords.keys())

    assert uvobj.history == uvobj.pyuvdata_version_str

    # delete the untarred folder
    shutil.rmtree(new_filename)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:telescope_location is not set")
def test_no_spw():
    """Test reading in a PAPER ms converted by CASA from a uvfits with no spw axis."""
    uvobj = UVData()
    testfile_no_spw = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAAM.ms")
    uvobj.read(testfile_no_spw)
    del uvobj


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_spwsupported():
    """Test reading in an ms file with multiple spws."""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1scan.ms")
    uvobj.read(testfile)

    assert uvobj.Nspws == 2


@pytest.mark.filterwarnings("ignore:Coordinate reference frame not detected,")
@pytest.mark.filterwarnings("ignore:UVW orientation appears to be flipped,")
@pytest.mark.filterwarnings("ignore:telescope_location is not set")
def test_extra_pol_setup(tmp_path):
    """Test reading in an ms file with extra polarization setups (not used in data)."""
    uvobj = UVData()
    testfile = os.path.join(
        DATA_PATH, "X5707_1spw_1scan_10chan_1time_1bl_noatm.ms.tar.gz"
    )

    import tarfile

    with tarfile.open(testfile) as tf:
        new_filename = os.path.join(tmp_path, tf.getnames()[0])
        tf.extractall(path=tmp_path)

    uvobj.read(new_filename, file_type="ms")

    # delete the untarred folder
    shutil.rmtree(new_filename)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
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
    # they are equal if only required parameters are checked:
    # scan numbers only defined for the MS
    assert uvfits_uv.__eq__(ms_uv, check_extra=False, allowed_failures=allowed_failures)

    # set those parameters to none to check that the rest of the objects match
    ms_uv.antenna_diameters = None

    for p in uvfits_uv.extra():
        fits_param = getattr(uvfits_uv, p)
        ms_param = getattr(ms_uv, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
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
    ms_uv.write_uvfits(testfile, spoof_nonessential=True)
    uvfits_uv.read(testfile)

    # make sure filenames are what we expect
    assert uvfits_uv.filename == ["outtest.uvfits"]
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    uvfits_uv.filename = ms_uv.filename

    # propagate scan numbers to the uvfits, ONLY for comparison
    uvfits_uv.scan_number_array = ms_uv.scan_number_array

    assert uvfits_uv == ms_uv
    del ms_uv
    del uvfits_uv


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
    ms_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read(testfile)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    miriad_uv.filename = ms_uv.filename

    # propagate scan numbers to the miriad uvdata, ONLY for comparison
    miriad_uv.scan_number_array = ms_uv.scan_number_array

    assert miriad_uv == ms_uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
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

    uv_multi.read(filesread, axis=axis)

    # histories are different because of combining along freq. axis
    # replace the history
    uv_multi.history = uv_full.history

    # the objects won't be equal because uvfits adds some optional parameters
    # and the ms sets default antenna diameters even though the uvfits file
    # doesn't have them
    assert uv_multi != uv_full
    # they are equal if only required parameters are checked:
    assert uv_multi.__eq__(uv_full, check_extra=False)

    # set those parameters to none to check that the rest of the objects match
    uv_multi.antenna_diameters = None

    for p in uv_full.extra():
        fits_param = getattr(uv_full, p)
        ms_param = getattr(uv_multi, p)
        if fits_param.name in UVFITS.uvfits_required_extra and ms_param.value is None:
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
        uvobj.read_ms(testfile, data_column="FOO")


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
def test_ms_history_lesson(mir_uv, tmp_path):
    """
    Test that the MS reader/writer can parse complex history
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_history_lesson.ms")
    mir_uv.history = (
        "Line 1.\nBegin measurement set history\nAPP_PARAMS;CLI_COMMAND;"
        "APPLICATION;MESSAGE;OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
        "End measurement set history.\nLine 2.\n"
    )
    mir_uv.write_ms(testfile, clobber=True)

    tb_hist = tables.table(testfile + "/HISTORY", readonly=False, ack=False)
    tb_hist.addrows()
    for col in tb_hist.colnames():
        tb_hist.putcell(col, tb_hist.nrows() - 1, tb_hist.getcell(col, 0))
    tb_hist.putcell("ORIGIN", 1, "DUMMY")
    tb_hist.putcell("APPLICATION", 1, "DUMMY")
    tb_hist.putcell("TIME", 1, 0.0)
    tb_hist.putcell("MESSAGE", 2, "Line 3.")
    tb_hist.close()

    ms_uv.read(testfile)

    assert ms_uv.history.startswith(
        "Line 1.\nBegin measurement set history\nAPP_PARAMS;CLI_COMMAND;APPLICATION;"
        "MESSAGE;OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n;;DUMMY;Line 2.;0;-1;"
        "DUMMY;INFO;0.0\nEnd measurement set history.\nLine 3.\n  Read/written with "
        "pyuvdata version:"
    )

    tb_hist = tables.table(os.path.join(testfile, "HISTORY"), ack=False, readonly=False)
    tb_hist.rename(os.path.join(testfile, "FORGOTTEN"))
    tb_hist.close()

    ms_uv.read(testfile)
    assert ms_uv.history.startswith("  Read/written with pyuvdata version:")


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_no_ref_dir_source(mir_uv, tmp_path):
    """
    Test that the MS writer/reader appropriately reads in a single-source data set
    as non-multi-phase if it can be, even if the original data set was.
    """
    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_no_ref_dir_source.ms")

    mir_uv.phase_center_frame = "fk5"
    mir_uv._set_app_coords_helper()
    mir_uv.write_ms(testfile, clobber=True)
    ms_uv.read(testfile)

    assert ms_uv.multi_phase_center is False

    ms_uv._set_multi_phase_center(preserve_phase_center_info=True)
    ms_uv._update_phase_center_id("3c84", 1)
    ms_uv.phase_center_catalog["3c84"]["info_source"] = "file"

    assert ms_uv.phase_center_catalog == mir_uv.phase_center_catalog


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_multi_spw_data_variation(mir_uv, tmp_path):
    """
    Test that the MS writer/reader appropriately reads in a single-source data set
    as non-multi-phase if it can be, even if the original data set was.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_multi_spw_data_variation.ms")

    mir_uv.write_ms(testfile, clobber=True)

    tb_main = tables.table(testfile, readonly=False, ack=False)
    tb_main.putcol("EXPOSURE", np.arange(mir_uv.Nblts * mir_uv.Nspws) + 1.0)
    tb_main.close()

    with pytest.raises(ValueError) as cm:
        ms_uv.read_ms(testfile)
    assert str(cm.value).startswith("Column EXPOSURE appears to vary on between")

    with uvtest.check_warnings(
        UserWarning, match="Column EXPOSURE appears to vary on between windows, ",
    ):
        ms_uv.read_ms(testfile, raise_error=False)

    # Check that the values do indeed match the first entry in the catalog
    assert np.all(ms_uv.integration_time == np.array([1.0]))


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_ms_phasing(mir_uv, future_shapes, tmp_path):
    """
    Test that the MS writer can appropriately handle unphased data sets.
    """
    if future_shapes:
        mir_uv.use_future_array_shapes()

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_phasing.ms")

    mir_uv.unphase_to_drift()

    with pytest.raises(ValueError) as cm:
        mir_uv.write_ms(testfile, clobber=True)
    assert str(cm.value).startswith("The data are in drift mode.")

    mir_uv.write_ms(testfile, clobber=True, force_phase=True)

    ms_uv.read(testfile)

    assert np.allclose(ms_uv.phase_center_app_ra, ms_uv.lst_array)
    assert np.allclose(
        ms_uv.phase_center_app_dec, ms_uv.telescope_location_lat_lon_alt[0]
    )


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_ms_single_chan(mir_uv, future_shapes, tmp_path):
    """
    Make sure that single channel writing/reading work as expected
    """
    if future_shapes:
        mir_uv.use_future_array_shapes()

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_single_chan.ms")

    mir_uv.select(freq_chans=0)
    mir_uv.write_ms(testfile, clobber=True)
    mir_uv.set_lsts_from_time_array()
    mir_uv._set_app_coords_helper()

    with pytest.raises(ValueError) as cm:
        ms_uv.read_ms(testfile)
    assert str(cm.value).startswith("No valid data available in the MS file.")

    ms_uv.read_ms(testfile, ignore_single_chan=False, read_weights=False)

    # Easiest way to check that everything worked is to just check for equality, but
    # the MS file is single-spw, single-field, so we have a few things we need to fix

    # First, make the date multi-phase-ctr
    ms_uv._set_multi_phase_center(preserve_phase_center_info=True)
    ms_uv._update_phase_center_id("3c84", 1)

    # Next, turn on flex-spw
    ms_uv._set_flex_spw()
    ms_uv.channel_width = np.array([ms_uv.channel_width])
    ms_uv.flex_spw_id_array = ms_uv.spw_array.copy()

    if future_shapes:
        ms_uv.use_future_array_shapes()

    # Finally, take care of the odds and ends
    ms_uv.extra_keywords = {}
    ms_uv.history = mir_uv.history
    ms_uv.filename = mir_uv.filename
    ms_uv.instrument = mir_uv.instrument
    ms_uv.reorder_blts()

    # propagate scan numbers to the uvfits, ONLY for comparison
    mir_uv.scan_number_array = ms_uv.scan_number_array

    assert ms_uv == mir_uv


@pytest.mark.filterwarnings("ignore:pamatten in extra_keywords is a list, array")
@pytest.mark.filterwarnings("ignore:psys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:psysattn in extra_keywords is a list, array or")
@pytest.mark.filterwarnings("ignore:ambpsys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:bfmask in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_scannumber_multiphasecenter(tmp_path):
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
            "Altitude is not present in Miriad file, "
            "using known location values for SZA.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "pamatten in extra_keywords is a list, array or dict",
            "psys in extra_keywords is a list, array or dict",
            "psysattn in extra_keywords is a list, array or dict",
            "ambpsys in extra_keywords is a list, array or dict",
            "bfmask in extra_keywords is a list, array or dict",
            "Cannot fix the phases of multi phase center datasets, as they were not "
            "supported when the old phasing method was used, and thus, there "
            "is no need to correct the data.",
        ],
    ):
        miriad_uv.read(carma_file, fix_old_proj=True)

    # MIRIAD is missing these in the file, so we'll fill it in here.
    miriad_uv.antenna_diameters = np.zeros(miriad_uv.Nants_telescope)
    miriad_uv.antenna_diameters[:6] = 10.0
    miriad_uv.antenna_diameters[15:] = 3.5

    # We need to recalculate app coords here for one source ("NOISE"), which was
    # not actually correctly calculated in the online CARMA system (long story). Since
    # the MS format requires recalculating apparent coords after read in, we'll
    # calculate them here just to verify that everything matches.
    miriad_uv._set_app_coords_helper()
    miriad_uv.write_ms(testfile, clobber=True)

    # Check on the scan number grouping based on consecutive integrations per phase
    # center

    # Double-check multi-phase center is True.
    assert miriad_uv.multi_phase_center

    # Read back in as MS. Should have 3 scan numbers defined.
    ms_uv = UVData()
    ms_uv.read(testfile)

    assert np.unique(ms_uv.scan_number_array).size == 3
    assert (np.unique(ms_uv.scan_number_array) == np.array([1, 2, 3])).all()

    # The scan numbers should match the phase center IDs, offset by 1
    # so that the scan numbers start with 1, not 0.
    assert ((miriad_uv.phase_center_id_array == (ms_uv.scan_number_array - 1))).all()


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_extra_data_descrip(mir_uv, tmp_path):
    """
    Make sure that data sets can be read even if the main table doesn't have data
    for a particular listed spectral window in the DATA_DESCRIPTION table.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_extra_data_descrip.ms")

    mir_uv.write_ms(testfile, clobber=True)

    tb_dd = tables.table(
        os.path.join(testfile, "DATA_DESCRIPTION"), ack=False, readonly=False
    )
    tb_dd.addrows()
    for col in tb_dd.colnames():
        tb_dd.putcell(col, tb_dd.nrows() - 1, tb_dd.getcell(col, 0))
    tb_dd.close()

    ms_uv.read_ms(testfile, ignore_single_chan=False, read_weights=False)

    # There are some minor differences between the values stored by MIR and that
    # calculated by UVData. Since MS format requires these to be calculated on the fly,
    # we calculate them here just to verify that everything is looking okay.
    mir_uv.set_lsts_from_time_array()
    mir_uv._set_app_coords_helper()

    # These reorderings just make sure that data from the two formats are lined up
    # correctly.
    mir_uv.reorder_freqs(spw_order="number")
    ms_uv.reorder_blts()

    # Fix the remaining differences between the two objects, all of which are expected
    mir_uv.instrument = mir_uv.telescope_name
    ms_uv.history = mir_uv.history
    mir_uv.extra_keywords = ms_uv.extra_keywords
    mir_uv.filename = ms_uv.filename = None

    # propagate scan numbers to the miriad uvdata, ONLY for comparison
    mir_uv.scan_number_array = ms_uv.scan_number_array

    # Finally, with all exceptions handled, check for equality.
    assert ms_uv == mir_uv


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_weights(mir_uv, tmp_path):
    """
    Test that the MS writer/reader appropriately handles data when the
    WEIGHT_SPECTRUM column is missing or bypassed.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_weights.ms")

    mir_uv.nsample_array[0, 0, :, 0] = np.tile(
        np.arange(mir_uv.Nfreqs / mir_uv.Nspws), mir_uv.Nspws,
    )
    mir_uv.write_ms(testfile, clobber=True)

    tb_main = tables.table(testfile, readonly=False, ack=False)
    tb_main.removecols("WEIGHT_SPECTRUM")
    tb_main.close()

    ms_uv.read_ms(testfile)

    # Check that the values do indeed match expected (median) value
    assert np.all(ms_uv.nsample_array == np.median(mir_uv.nsample_array))

    ms_uv.read_ms(testfile, read_weights=False)
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
        ["TEL_LOC", None, ValueError, "Telescope frame is not ITRF and telescope is"],
    ),
)
def test_ms_reader_errs(mir_uv, tmp_path, badcol, badval, errtype, msg):
    """
    Test whether the reader throws an appripropriate errors on read.
    """
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_reader_errs.ms")
    mir_uv.write_ms(testfile, clobber=True)

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
    else:
        tb_main = tables.table(testfile, ack=False, readonly=False)
        tb_main.putcol(badcol, badval)
        tb_main.close()

    with pytest.raises(errtype) as cm:
        ms_uv.read_ms(testfile, data_column=data_col)
    assert str(cm.value).startswith(msg)


def test_antenna_diameter_handling(hera_uvh5, tmp_path):
    uv_obj = hera_uvh5

    uv_obj.antenna_diameters = np.asarray(uv_obj.antenna_diameters, dtype=">f4")

    test_file = os.path.join(tmp_path, "dish_diameter_out.ms")
    with uvtest.check_warnings(
        UserWarning, match="Writing in the MS file that the units of the data are"
    ):
        uv_obj.write_ms(test_file, force_phase=True)

    uv_obj2 = UVData.from_file(test_file)

    # MS write/read adds some stuff to history & extra keywords
    uv_obj2.history = uv_obj.history
    uv_obj2.extra_keywords = uv_obj.extra_keywords

    assert uv_obj2.__eq__(uv_obj, allowed_failures=allowed_failures)
