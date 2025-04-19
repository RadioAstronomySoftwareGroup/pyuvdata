# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MS object."""

import os
import shutil

import numpy as np
import pytest
from astropy.time import Time

from pyuvdata import UVData, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

from ..utils.test_coordinates import frame_selenoid

pytest.importorskip("casacore")


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


def safe_extract(
    tar, path=".", members=None, *, numeric_owner=False, use_filter="data"
):
    # this is factored this way (splitting out the `check_members` function)
    # to appease bandit.
    try:
        tar.extractall(
            path,
            members=check_members(tar, path),
            numeric_owner=numeric_owner,
            filter=use_filter,
        )
    except TypeError:
        # older versions of python don't have the filter argument
        tar.extractall(
            path, members=check_members(tar, path), numeric_owner=numeric_owner
        )


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


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:Setting telescope_location to value")
def test_cotter_ms():
    """Test reading in an ms made from MWA data with cotter (no dysco compression)"""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "1102865728_small.ms/")
    uvobj.read(testfile)

    # check that a select on read works
    uvobj2 = UVData()

    with check_warnings(
        UserWarning,
        match=[
            "Warning: select on read keyword set",
            "Setting telescope_location to value in known_telescopes for MWA.",
        ],
    ):
        uvobj2.read(testfile, freq_chans=np.arange(2))
    uvobj.select(freq_chans=np.arange(2))
    assert uvobj == uvobj2
    del uvobj


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
@pytest.mark.parametrize("del_tel_loc", [True, False])
def test_read_nrao_loopback(tmp_path, nrao_uv, telescope_frame, selenoid, del_tel_loc):
    """Test reading in a CASA tutorial ms file and looping it through write_ms."""
    uvobj = nrao_uv

    if telescope_frame == "mcmf":
        pytest.importorskip("lunarsky")
        from lunarsky import MoonLocation
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        try:
            enu_antpos = uvobj.telescope.get_enu_antpos()
            uvobj.telescope.location = MoonLocation.from_selenodetic(
                lat=uvobj.telescope.location.lat,
                lon=uvobj.telescope.location.lon,
                height=uvobj.telescope.location.height,
                ellipsoid=selenoid,
            )
            new_full_antpos = utils.ECEF_from_ENU(
                enu=enu_antpos, center_loc=uvobj.telescope.location
            )
            uvobj.telescope.antenna_positions = (
                new_full_antpos - uvobj.telescope._location.xyz()
            )
            uvobj.set_lsts_from_time_array()
            uvobj.set_uvws_from_antenna_positions()
            uvobj._set_app_coords_helper()
            uvobj.check()
        except SpiceUNKNOWNFRAME as err:
            pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))

    expected_extra_keywords = ["DATA_COL", "observer"]

    assert sorted(expected_extra_keywords) == sorted(uvobj.extra_keywords.keys())

    testfile = os.path.join(tmp_path, "ms_testfile.ms")

    with check_warnings(
        UserWarning,
        match=[
            "Writing in the MS file that the units of the data are",
            "The uvw_array does not match the expected values",
        ],
    ):
        uvobj.write_ms(testfile)

    # check handling of default ellipsoid: remove the ellipsoid and check that
    # it is properly defaulted to SPHERE
    if telescope_frame == "mcmf" and selenoid == "SPHERE":
        from casacore import tables

        tb_ant = tables.table(
            os.path.join(testfile, "ANTENNA"), ack=False, readonly=False
        )
        meas_info_dict = tb_ant.getcolkeyword("POSITION", "MEASINFO")
        del meas_info_dict["RefEllipsoid"]
        tb_ant.putcolkeyword("POSITION", "MEASINFO", meas_info_dict)
        tb_ant.close()

    if del_tel_loc:
        # This doesn't lead to test errors because the original data set didn't
        # have a location, so we were already using the center of the antenna positions
        from casacore import tables

        tb_obs = tables.table(
            os.path.join(testfile, "OBSERVATION"), ack=False, readonly=False
        )
        tb_obs.removecols("TELESCOPE_LOCATION")
        tb_obs.close()

    uvobj2 = UVData()
    uvobj2.read_ms(testfile)

    # also update filenames
    assert uvobj.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    assert uvobj2.filename == ["ms_testfile.ms"]
    uvobj.filename = uvobj2.filename

    assert uvobj.telescope._location.frame == uvobj2.telescope._location.frame

    # Test that the scan numbers are equal
    assert (uvobj.scan_number_array == uvobj2.scan_number_array).all()

    assert uvobj == uvobj2

    # test that the clobber keyword works by rewriting
    with check_warnings(
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

    uvobj.read(new_filename, file_type="ms")
    assert sorted(expected_extra_keywords) == sorted(uvobj.extra_keywords.keys())

    assert uvobj.history == uvobj.pyuvdata_version_str

    # delete the untarred folder
    shutil.rmtree(new_filename)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Setting telescope_location to value")
def test_no_spw():
    """Test reading in a PAPER ms converted by CASA from a uvfits with no spw axis."""
    uvobj = UVData()
    testfile_no_spw = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAAM.ms")
    uvobj.read(testfile_no_spw)
    del uvobj


@pytest.mark.filterwarnings("ignore:Coordinate reference frame not detected,")
@pytest.mark.filterwarnings("ignore:UVW orientation appears to be flipped,")
@pytest.mark.filterwarnings("ignore:Setting telescope_location to value")
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

    uvobj.read(new_filename, file_type="ms")

    # delete the untarred folder
    shutil.rmtree(new_filename)


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
    uvfits_uv.telescope.antenna_numbers = uvfits_uv.telescope.antenna_numbers - 1
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
    assert uvfits_uv.__eq__(ms_uv, check_extra=False)

    # set those parameters to none to check that the rest of the objects match
    ms_uv.telescope.antenna_diameters = None
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

    # Set the feed information the same, ONLY for comparison
    uvfits_uv.telescope.feed_array = ms_uv.telescope.feed_array
    uvfits_uv.telescope.feed_angle = ms_uv.telescope.feed_angle
    uvfits_uv.telescope.Nfeeds = ms_uv.telescope.Nfeeds

    assert uvfits_uv == ms_uv


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
    uvfits_uv.read(testfile)

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


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_ms_write_miriad(nrao_uv, tmp_path):
    """
    read ms, write miriad test.
    Read in ms file, write out as miriad, read back in and check for
    object equality.
    """
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    ms_uv = nrao_uv
    miriad_uv = UVData()
    testfile = os.path.join(tmp_path, "outtest_miriad")
    with check_warnings(
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
    miriad_uv.read(testfile)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    miriad_uv.filename = ms_uv.filename

    # propagate scan numbers to the miriad uvdata, ONLY for comparison
    miriad_uv.scan_number_array = ms_uv.scan_number_array

    assert miriad_uv == ms_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:The older phase attributes")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.parametrize("axis", [None, "freq"])
def test_multi_files(casa_uvfits, axis, tmp_path):
    """
    Reading multiple files at once.
    """
    uv_full = casa_uvfits.copy()

    # Ensure the scan numbers are defined for the comparison
    uv_full._set_scan_numbers()

    uv_part1 = uv_full.select(
        freq_chans=np.arange(0, uv_full.Nfreqs // 2), inplace=False
    )
    uv_part2 = uv_full.select(
        freq_chans=np.arange(uv_full.Nfreqs // 2, uv_full.Nfreqs), inplace=False
    )

    uv_multi = UVData()
    testfile1 = os.path.join(tmp_path, "multi_1.ms")
    testfile2 = os.path.join(tmp_path, "multi_2.ms")
    uv_part1.write_ms(testfile1, clobber=True)
    uv_part2.write_ms(testfile2, clobber=True)

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
    # The uvfits was written by CASA, which adds one to all the antenna numbers relative
    # to the measurement set. Adjust those:
    uv_full.baseline_array = uv_full.antnums_to_baseline(
        uv_full.ant_1_array, uv_full.ant_2_array
    )

    uv_full._consolidate_phase_center_catalogs(
        reference_catalog=uv_multi.phase_center_catalog
    )

    # now they are equal if only required parameters are checked:
    assert uv_multi.__eq__(uv_full, check_extra=False)

    # set those parameters to none to check that the rest of the objects match
    uv_multi.telescope.antenna_diameters = None

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

    assert uv_multi == uv_full
    del uv_full
    del uv_multi


def test_bad_col_name():
    """
    Test error with invalid column name.
    """
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")

    with pytest.raises(ValueError, match="Invalid data_column value supplied"):
        uvobj.read(testfile, data_column="FOO")


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

    with check_warnings(
        UserWarning, match="Column EXPOSURE appears to vary on between windows, "
    ):
        ms_uv.read(testfile, raise_error=False)

    # Check that the values do indeed match the first entry in the catalog
    assert np.all(ms_uv.integration_time == np.array([1.0]))


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_ms_phasing(sma_mir, tmp_path):
    """
    Test that the MS writer can appropriately handle unphased data sets.
    """
    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_phasing.ms")

    sma_mir.unproject_phase()

    with pytest.raises(ValueError, match="The data are unprojected."):
        sma_mir.write_ms(testfile)

    sma_mir.write_ms(testfile, force_phase=True)

    ms_uv.read(testfile)

    np.testing.assert_allclose(
        ms_uv.phase_center_app_ra, ms_uv.lst_array, rtol=0, atol=utils.RADIAN_TOL
    )
    np.testing.assert_allclose(
        ms_uv.phase_center_app_dec,
        ms_uv.telescope.location.lat.rad,
        rtol=0,
        atol=utils.RADIAN_TOL,
    )


@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_ms_single_chan(sma_mir, tmp_path):
    """
    Make sure that single channel writing/reading work as expected
    """
    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_single_chan.ms")

    sma_mir.select(freq_chans=0)
    sma_mir.write_ms(testfile)
    sma_mir.set_lsts_from_time_array()
    sma_mir._set_app_coords_helper()

    with pytest.raises(ValueError, match="No valid data available in the MS file."):
        ms_uv.read(testfile)

    ms_uv.read(testfile, ignore_single_chan=False)

    # Easiest way to check that everything worked is to just check for equality, but
    # the MS file is single-spw, single-field, so we have a few things we need to fix

    cat_id = list(sma_mir.phase_center_catalog.keys())[0]
    cat_name = sma_mir.phase_center_catalog[cat_id]["cat_name"]
    ms_uv._update_phase_center_id(
        list(ms_uv.phase_center_catalog.keys())[0], new_id=cat_id
    )
    ms_uv.phase_center_catalog[cat_id]["cat_name"] = cat_name
    ms_uv.phase_center_catalog[cat_id]["info_source"] = "file"

    # Finally, take care of the odds and ends
    ms_uv.extra_keywords = {}
    ms_uv.history = sma_mir.history
    ms_uv.filename = sma_mir.filename
    ms_uv.telescope.instrument = sma_mir.telescope.instrument
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
    with check_warnings(
        UserWarning,
        [
            (
                "Altitude is not present in Miriad file, "
                "using known location values for SZA."
            ),
            "The uvw_array does not match the expected values",
        ],
    ):
        miriad_uv.read(carma_file)

    # MIRIAD is missing these in the file, so we'll fill it in here.
    miriad_uv.telescope.antenna_diameters = np.zeros(miriad_uv.telescope.Nants)
    miriad_uv.telescope.antenna_diameters[:6] = 10.0
    miriad_uv.telescope.antenna_diameters[15:] = 3.5

    # We need to recalculate app coords here for one source ("NOISE"), which was
    # not actually correctly calculated in the online CARMA system (long story). Since
    # the MS format requires recalculating apparent coords after read in, we'll
    # calculate them here just to verify that everything matches.
    miriad_uv._set_app_coords_helper()

    if multi_frame:
        cat_id = utils.phase_center_catalog.look_for_name(
            miriad_uv.phase_center_catalog, "NOISE"
        )
        ra_use = miriad_uv.phase_center_catalog[cat_id[0]]["cat_lon"][0]
        dec_use = miriad_uv.phase_center_catalog[cat_id[0]]["cat_lat"][0]
        with pytest.raises(
            ValueError,
            match="lon parameter must be a single value for cat_type sidereal",
        ):
            miriad_uv.phase(
                lon=miriad_uv.phase_center_catalog[cat_id[0]]["cat_lon"],
                lat=dec_use,
                cat_name="foo",
                phase_frame="icrs",
                select_mask=miriad_uv.phase_center_id_array == cat_id[0],
            )

        with pytest.raises(
            ValueError,
            match="lat parameter must be a single value for cat_type sidereal",
        ):
            miriad_uv.phase(
                lon=ra_use,
                lat=miriad_uv.phase_center_catalog[cat_id[0]]["cat_lat"],
                cat_name="foo",
                phase_frame="icrs",
                select_mask=miriad_uv.phase_center_id_array == cat_id[0],
            )

        with check_warnings(
            UserWarning,
            match=[
                "The entry name NOISE is not unique",
                "The provided name NOISE is already used",
            ],
        ):
            miriad_uv.phase(
                lon=ra_use,
                lat=dec_use,
                cat_name="NOISE",
                phase_frame="icrs",
                select_mask=miriad_uv.phase_center_id_array == cat_id[0],
            )
    miriad_uv.write_ms(testfile)

    # Check on the scan number grouping based on consecutive integrations per phase
    # center

    # Read back in as MS. Should have 3 scan numbers defined.
    ms_uv = UVData()
    ms_uv.read(testfile)

    assert np.unique(ms_uv.scan_number_array).size == 3
    assert (np.unique(ms_uv.scan_number_array) == np.array([1, 2, 3])).all()

    # The scan numbers should match the phase center IDs, offset by 1
    # so that the scan numbers start with 1, not 0.
    assert (miriad_uv.phase_center_id_array == (ms_uv.scan_number_array - 1)).all()


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

    ms_uv.read(testfile, ignore_single_chan=False)
    cat_id = list(sma_mir.phase_center_catalog.keys())[0]
    cat_name = sma_mir.phase_center_catalog[cat_id]["cat_name"]
    ms_uv._update_phase_center_id(
        list(ms_uv.phase_center_catalog.keys())[0], new_id=cat_id
    )
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
    sma_mir.telescope.instrument = sma_mir.telescope.name
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

    ms_uv.read(testfile)

    # Check that the values do indeed match expected (median) value
    assert np.all(ms_uv.nsample_array == np.median(sma_mir.nsample_array))

    ms_uv.read(testfile, read_weights=False)
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
        with check_warnings(UserWarning, match=msg):
            ms_uv.read(
                testfile, data_column=data_col, file_type="ms", raise_error=False
            )
        assert ms_uv._time_array == sma_mir._time_array


@pytest.mark.skipif(
    len(frame_selenoid) > 1, reason="Test only when lunarsky not installed."
)
def test_ms_no_moon(sma_mir, tmp_path):
    """Check errors when calling read_ms with MCMF without lunarsky."""
    from casacore import tables

    ms_uv = UVData()
    testfile = os.path.join(tmp_path, "out_ms_reader_errs.ms")
    sma_mir.write_ms(testfile)

    tb_obs = tables.table(
        os.path.join(testfile, "OBSERVATION"), ack=False, readonly=False
    )
    tb_obs.removecols("TELESCOPE_LOCATION")
    tb_obs.putcol("TELESCOPE_NAME", "ABC")
    tb_obs.close()
    tb_ant = tables.table(os.path.join(testfile, "ANTENNA"), ack=False, readonly=False)
    tb_ant.putcolkeyword("POSITION", "MEASINFO", {"type": "position", "Ref": "MCMF"})
    tb_ant.close()

    msg = "Need to install `lunarsky` package to work with MCMF frame."
    with pytest.raises(ValueError, match=msg):
        ms_uv.read(testfile, data_column="DATA", file_type="ms")


def test_antenna_diameter_handling(hera_uvh5, tmp_path):
    uv_obj = hera_uvh5

    uv_obj.telescope.antenna_diameters = np.asarray(
        uv_obj.telescope.antenna_diameters, dtype=">f4"
    )

    test_file = os.path.join(tmp_path, "dish_diameter_out.ms")
    with check_warnings(
        UserWarning,
        match=[
            "Writing in the MS file that the units of the data are",
            "UVData object contains a mix of baseline conjugation states",
        ],
    ):
        uv_obj.write_ms(test_file, force_phase=True)

    uv_obj2 = UVData.from_file(test_file)

    # MS write/read adds some stuff to history & extra keywords
    uv_obj2.history = uv_obj.history
    uv_obj2.extra_keywords = uv_obj.extra_keywords

    uv_obj2._consolidate_phase_center_catalogs(
        reference_catalog=uv_obj.phase_center_catalog
    )
    assert uv_obj2 == uv_obj


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_ms_optional_parameters(nrao_uv, tmp_path):
    uv_obj = nrao_uv

    uv_obj.telescope.set_feeds_from_x_orientation(
        "east", polarization_array=uv_obj.polarization_array
    )
    uv_obj.pol_convention = "sum"
    uv_obj.vis_units = "Jy"
    # Update the order so as to be UVFITS compliant
    uv_obj.telescope.reorder_feeds("AIPS")

    test_file = os.path.join(tmp_path, "dish_diameter_out.ms")
    uv_obj.write_ms(test_file, force_phase=True)

    uv_obj2 = UVData.from_file(test_file)

    uv_obj2._consolidate_phase_center_catalogs(
        reference_catalog=uv_obj.phase_center_catalog
    )
    assert uv_obj2 == uv_obj


def test_no_source(sma_mir, tmp_path):
    uv = UVData()
    uv2 = UVData()
    filename = os.path.join(tmp_path, "no_source.ms")

    sma_mir.write_ms(filename)

    uv.read(filename)

    shutil.rmtree(os.path.join(filename, "SOURCE"))
    uv2.read(filename)

    assert uv == uv2


@pytest.mark.filterwarnings("ignore:Setting telescope_location to value")
@pytest.mark.filterwarnings("ignore:UVW orientation appears to be flipped")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
def test_timescale_handling():
    ut1_file = os.path.join(DATA_PATH, "1090008640_birli_pyuvdata.ms")

    uvobj = UVData.from_file(ut1_file)
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
    with check_warnings(UserWarning, match="Failed to parse prior history of MS file,"):
        sma_mir.write_ms(filename)

    # Make sure the history is actually preserved correctly.
    sma_ms = UVData.from_file(filename)
    assert sma_mir.history in sma_ms.history


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_flip_conj(nrao_uv, tmp_path):
    filename = os.path.join(tmp_path, "flip_conj.ms")
    nrao_uv.set_uvws_from_antenna_positions()
    nrao_uv.uvw_array *= -1
    nrao_uv.data_array = np.conj(nrao_uv.data_array)

    with check_warnings(
        UserWarning, match="Writing in the MS file that the units of the data are unca"
    ):
        nrao_uv.write_ms(filename, flip_conj=True, run_check=False, clobber=True)

    with check_warnings(UserWarning, match=["UVW orientation appears to be flip"] * 2):
        uv = UVData.from_file(filename)
        nrao_uv.check(allow_flip_conj=True)

    assert nrao_uv == uv


def test_no_flip(sma_mir, tmp_path):
    filename = os.path.join(tmp_path, "no_flip_conj.ms")
    sma_mir._set_app_coords_helper()

    # Now test that turning off the flip passes through nominally.
    sma_mir.write_ms(filename, flip_conj=False, clobber=True)
    uv = UVData.from_file(filename)

    assert sma_mir.__eq__(
        uv, allowed_failures=["history", "extra_keywords", "instrument", "filename"]
    )


def test_importuvfits_flip_conj(sma_mir, tmp_path):
    from casacore import tables

    uv = UVData()
    sma_mir._set_app_coords_helper()

    filename = os.path.join(tmp_path, "importuvfits_flip_conj.ms")
    sma_mir.write_ms(filename, flip_conj=True)

    # Overwrite history info to make it look like written as importfits
    with tables.table(filename + "/HISTORY", readonly=False, ack=False) as tb_hist:
        tb_hist.putcell("ORIGIN", 0, "DUMMY")
        tb_hist.putcell("APPLICATION", 0, "DUMMY")
        tb_hist.putcell("MESSAGE", 0, "importuvfits")

    # Test flip_conj on read
    uv.read(filename)
    uv.history = sma_mir.history
    uv.extra_keywords = sma_mir.extra_keywords
    uv.telescope.instrument = sma_mir.telescope.instrument

    assert sma_mir == uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_flip_conj_multispw(sma_mir, tmp_path):
    sma_mir._set_app_coords_helper()
    filename = os.path.join(tmp_path, "flip_conj_multispw.ms")

    sma_mir.write_ms(filename, flip_conj=True)
    ms_uv = UVData.from_file(filename)

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
    assert ms_uv.filename == ["flip_conj_multispw.ms"]
    sma_mir.filename = ms_uv.filename = None

    assert sma_mir == ms_uv


@pytest.mark.parametrize("data_column", ["MODEL_DATA", "CORRECTED_DATA"])
def test_read_ms_write_ms_alt_data_colums(sma_mir, tmp_path, data_column):
    # Fix the app coords since CASA reader calculates them on the fly
    sma_mir._set_app_coords_helper()

    testfile = os.path.join(tmp_path, "alt_data_columns.ms")
    model_data = corrected_data = None
    if data_column == "MODEL_DATA":
        model_data = np.full(sma_mir.data_array.shape, 2.0 + 3.0j)
        data_test = model_data
    if data_column == "CORRECTED_DATA":
        corrected_data = np.full(sma_mir.data_array.shape, 4.0 - 5.0j)
        data_test = corrected_data
    sma_mir.write_ms(testfile, model_data=model_data, corrected_data=corrected_data)

    uvd = UVData()
    uvd.read(testfile, data_column=data_column)
    assert np.array_equal(uvd.data_array, data_test)
    uvd.data_array = sma_mir.data_array

    assert uvd.extra_keywords["DATA_COL"] == data_column
    uvd.extra_keywords = sma_mir.extra_keywords
    uvd.telescope.instrument = sma_mir.telescope.instrument
    assert sma_mir.history in uvd.history
    uvd.history = sma_mir.history
    assert uvd == sma_mir


def test_write_ms_feed_sort(sma_mir, tmp_path):
    # Fix the app coords since CASA reader calculates them on the fly
    sma_mir._set_app_coords_helper()

    uvd = UVData()
    testfile = os.path.join(tmp_path, "feed_order.ms")
    sma_mir.telescope.reorder_feeds(order=["y", "x", "l", "r"])
    sma_mir.write_ms(testfile, clobber=True)
    uvd.read(testfile)

    # Just set this up front
    uvd.history = sma_mir.history
    uvd.telescope.instrument = sma_mir.telescope.instrument
    uvd.extra_keywords = sma_mir.extra_keywords

    assert uvd != sma_mir
    uvd.telescope.reorder_feeds(order=["y", "x", "l", "r"])
    assert uvd == sma_mir


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_write_ms_baseline_conj_warning(nrao_uv, tmp_path):
    testfile = os.path.join(tmp_path, "mix_bl_conj.ms")

    uvd = nrao_uv
    uvd.vis_units = "Jy"
    uvd.pol_convention = "sum"

    # Mix up the conj for some baselines
    uvd.conjugate_bls(convention="u>0")

    with check_warnings(
        UserWarning,
        match=[
            "The uvw_array does not match",
            "UVData object contains a mix of baseline conjugation states",
        ],
    ):
        uvd.write_ms(testfile, clobber=True)

    uvd2 = UVData.from_file(testfile)
    assert uvd == uvd2
    assert all(uvd.ant_1_array >= uvd.ant_2_array)
