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
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
from ..uvfits import UVFITS

pytest.importorskip("casacore")


@pytest.fixture(scope="session")
def nrao_uv_main():
    # This file is known to be made with CASA's importuvfits task, but apparently the
    # history table is missing, so we can't infer that from the history. This means that
    # the uvws are not flipped/data is not conjugated as they should be. Fix that.

    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    uvobj.read(testfile)
    # uvobj.uvw_array *= -1
    # uvobj.data_array = np.conj(uvobj.data_array)

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
            "ITRF coordinate frame detected, although this is often ",
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
    expected_extra_keywords = ["DATA_COL"]

    assert sorted(expected_extra_keywords) == sorted(uvobj.extra_keywords.keys())

    testfile = os.path.join(tmp_path, "ms_testfile.ms")

    uvobj.write_ms(testfile)
    uvobj2 = UVData()
    uvobj2.read_ms(testfile)

    # also update filenames
    assert uvobj.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    assert uvobj2.filename == ["ms_testfile.ms"]
    uvobj.filename = uvobj2.filename

    assert uvobj == uvobj2


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_lwa(tmp_path):
    """Test reading in an LWA ms file."""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "lwasv_cor_58342_05_00_14.ms.tar.gz")
    expected_extra_keywords = ["DATA_COL"]

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


@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected,")
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
    assert uvfits_uv.__eq__(ms_uv, check_extra=False)

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

    assert miriad_uv == ms_uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("axis", [None, "freq"])
def test_multi_files(casa_uvfits, axis):
    """
    Reading multiple files at once.
    """
    uv_full = casa_uvfits.copy()

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
    # uv_multi.uvw_array *= -1
    # uv_multi.data_array = np.conj(uv_multi.data_array)
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
        uvobj.read_ms(testfile, data_column="FOO")
