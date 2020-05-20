# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD object.

"""
import pytest
import os
import glob
import numpy as np
from shutil import copyfile

from pyuvdata import UVData
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

# set up FHD file list
testdir = os.path.join(DATA_PATH, "fhd_vis_data/")
testfile_prefix = "1061316296_"
# note: 1061316296_obs.sav isn't used -- it's there to test handling of unneeded files
testfile_suffix = [
    "flags.sav",
    "vis_XX.sav",
    "params.sav",
    "vis_YY.sav",
    "vis_model_XX.sav",
    "vis_model_YY.sav",
    "layout.sav",
    "settings.txt",
    "obs.sav",
]
testfiles = []
for s in testfile_suffix:
    testfiles.append(testdir + testfile_prefix + s)


def test_read_fhd_write_read_uvfits(tmp_path):
    """
    FHD to uvfits loopback test.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    uvtest.checkWarnings(
        fhd_uv.read_fhd,
        func_args=[testfiles],
        nwarnings=2,
        message=[
            "Telescope location derived from obs lat/lon/alt values does not "
            "match the location in the layout file. Value from obs lat/lon/alt "
            "matches the known_telescopes values, using them.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    )

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_metadata_only():
    fhd_uv = UVData()
    fhd_uv.read_fhd(testfiles, read_data=False)

    assert fhd_uv.metadata_only

    fhd_uv2 = UVData()
    fhd_uv2.read_fhd(testfiles)
    fhd_uv3 = fhd_uv2.copy(metadata_only=True)

    assert fhd_uv == fhd_uv3


def test_read_fhd_metadata_only_error():
    fhd_uv = UVData()
    with pytest.raises(
        ValueError, match="No obs file included in file list and read_data is False."
    ):
        fhd_uv.read_fhd(testfiles[:7], read_data=False)


def test_read_fhd_select():
    """
    test select on read with FHD files.

    Read in FHD files with generic read & select on read, compare to read fhd
    files then do select
    """
    fhd_uv = UVData()
    fhd_uv2 = UVData()
    uvtest.checkWarnings(
        fhd_uv2.read,
        func_args=[testfiles],
        func_kwargs={"freq_chans": np.arange(2)},
        message=[
            'Warning: select on read keyword set, but file_type is "fhd" which '
            "does not support select on read. Entire file will be read and then "
            "select will be performed",
            "Telescope location derived from obs lat/lon/alt values does not "
            "match the location in the layout file. Value from obs lat/lon/alt "
            "matches the known_telescopes values, using them.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
        nwarnings=3,
    )

    uvtest.checkWarnings(
        fhd_uv.read,
        func_args=[testfiles],
        message=[
            "Telescope location derived from obs lat/lon/alt values does not "
            "match the location in the layout file. Value from obs lat/lon/alt "
            "matches the known_telescopes values, using them.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
        nwarnings=2,
    )

    fhd_uv.select(freq_chans=np.arange(2))
    assert fhd_uv == fhd_uv2


def test_read_fhd_write_read_uvfits_no_layout():
    """
    Test errors/warnings with with no layout file.
    """
    fhd_uv = UVData()
    files_use = testfiles[:-3] + [testfiles[-2]]

    # check warning raised
    uvtest.checkWarnings(
        fhd_uv.read,
        func_args=[files_use],
        func_kwargs={"run_check": False},
        message="No layout file",
        nwarnings=1,
        category=UserWarning,
    )

    with pytest.raises(ValueError) as cm:
        uvtest.checkWarnings(
            fhd_uv.read,
            func_args=[files_use],
            message="No layout file",
            nwarnings=1,
            category=UserWarning,
        )
    assert str(cm.value).startswith(
        "Required UVParameter _antenna_positions has not been set"
    )


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_fhd_write_read_uvfits_variant_flag(tmp_path):
    """
    FHD to uvfits loopback test with variant flag file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    variant_flag_file = testdir + testfile_prefix + "variant_flags.sav"
    files_use = testfiles[1:] + [variant_flag_file]
    fhd_uv.read(files_use)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_fix_layout(tmp_path):
    """
    FHD to uvfits loopback test with fixed array center layout file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    layout_fixed_file = testdir + testfile_prefix + "fixed_arr_center_layout.sav"
    files_use = testfiles[0:6] + [layout_fixed_file, testfiles[7]]
    fhd_uv.read(files_use)
    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")

    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_fix_layout_bad_obs_loc(tmp_path):
    """
    FHD to uvfits loopback test with fixed array center layout file, bad obs location.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    bad_obs_loc_file = testdir + testfile_prefix + "bad_obs_loc_vis_XX.sav"
    layout_fixed_file = testdir + testfile_prefix + "fixed_arr_center_layout.sav"
    files_use = [
        testfiles[0],
        testfiles[2],
        bad_obs_loc_file,
        layout_fixed_file,
        testfiles[7],
    ]
    messages = [
        "Telescope location derived from obs",
        "tile_names from obs structure does not match",
        "The uvw_array does not match the expected values given the antenna "
        "positions.",
    ]
    uvtest.checkWarnings(fhd_uv.read, [files_use], message=messages, nwarnings=3)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_bad_obs_loc(tmp_path):
    """
    FHD to uvfits loopback test with bad obs location (and bad layout location).

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    bad_obs_loc_file = testdir + testfile_prefix + "bad_obs_loc_vis_XX.sav"
    files_use = [
        testfiles[0],
        testfiles[2],
        bad_obs_loc_file,
        testfiles[6],
        testfiles[7],
    ]
    messages = [
        "Telescope location derived from obs",
        "tile_names from obs structure does not match",
        "The uvw_array does not match the expected values given the antenna "
        "positions.",
    ]
    uvtest.checkWarnings(fhd_uv.read, [files_use], message=messages, nwarnings=3)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_altered_layout(tmp_path):
    """
    FHD to uvfits loopback test with altered layout file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()

    # bad layout structure values
    altered_layout_file = testdir + testfile_prefix + "broken_layout.sav"
    files_use = testfiles[0:6] + [altered_layout_file, testfiles[7]]
    fhd_uv.read(files_use)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_no_settings(tmp_path):
    """
    FHD to uvfits loopback test with no settings file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    messages = [
        "No settings file included in file list",
        "Telescope location derived from obs lat/lon/alt values does "
        "not match the location in the layout file. Value from obs "
        "lat/lon/alt matches the known_telescopes values, using them.",
        "The uvw_array does not match the expected values given the antenna positions.",
    ]
    uvtest.checkWarnings(fhd_uv.read, [testfiles[:-2]], message=messages, nwarnings=3)

    # Check only pyuvdata history with no settings file
    assert fhd_uv.history == fhd_uv.pyuvdata_version_str

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


def test_break_read_fhd():
    """Try various cases of incomplete file lists."""
    fhd_uv = UVData()
    # missing flags
    with pytest.raises(ValueError, match="No flags file included in file list"):
        fhd_uv.read(testfiles[1:])

    # Missing params
    subfiles = [item for sublist in [testfiles[0:2], testfiles[3:]] for item in sublist]
    with pytest.raises(ValueError, match="No params file included in file list"):
        fhd_uv.read(subfiles)

    # No data files
    with pytest.raises(
        ValueError, match="No data files included in file list and read_data is True.",
    ):
        fhd_uv.read(["foo.sav"])


def test_read_fhd_warnings():
    """Test warnings with various broken inputs."""
    # bad obs structure values
    broken_data_file = testdir + testfile_prefix + "broken_vis_XX.sav"
    bad_filelist = [
        testfiles[0],
        testfiles[2],
        broken_data_file,
        testfiles[6],
        testfiles[7],
    ]
    warn_messages = [
        "Ntimes does not match",
        "Telescope location derived from obs",
        "Telescope foo is not in known_telescopes.",
        "These visibilities may have been phased improperly",
        "Nbls does not match",
    ]
    fhd_uv = UVData()
    uvtest.checkWarnings(
        fhd_uv.read,
        [bad_filelist],
        {"run_check": False},
        nwarnings=5,
        message=warn_messages,
    )

    # bad flag file
    broken_flag_file = testdir + testfile_prefix + "broken_flags.sav"
    bad_filelist = testfiles[1:] + [broken_flag_file]
    fhd_uv = UVData()
    with pytest.raises(
        ValueError, match="No recognized key for visibility weights in flags_file."
    ):
        fhd_uv.read(bad_filelist)


@pytest.mark.parametrize(
    "new_file_end,file_copy,message",
    [
        (["extra_vis_XX.sav"], testfiles[1], "multiple xx datafiles in filelist"),
        (["extra_vis_YY.sav"], testfiles[3], "multiple yy datafiles in filelist"),
        (
            ["vis_XY.sav", "extra_vis_XY.sav"],
            testfiles[1],
            "multiple xy datafiles in filelist",
        ),
        (
            ["vis_YX.sav", "extra_vis_YX.sav"],
            testfiles[1],
            "multiple yx datafiles in filelist",
        ),
        (["extra_params.sav"], testfiles[2], "multiple params files in filelist"),
        (["extra_obs.sav"], testfiles[8], "multiple obs files in filelist"),
        (["extra_flags.sav"], testfiles[0], "multiple flags files in filelist"),
        (["extra_layout.sav"], testfiles[6], "multiple layout files in filelist"),
        (["extra_settings.txt"], testfiles[7], "multiple settings files in filelist"),
    ],
)
def test_read_fhd_extra_files(new_file_end, file_copy, message):
    # try cases with extra files of each type
    new_files = []
    for file_end in new_file_end:
        extra_file = testdir + testfile_prefix + file_end
        new_files.append(extra_file)
        copyfile(file_copy, extra_file)
    fhd_uv = UVData()
    with pytest.raises(ValueError, match=message):
        fhd_uv.read(testfiles + new_files)
    for extra_file in new_files:
        os.remove(extra_file)


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_fhd_model(tmp_path):
    """FHD to uvfits loopback test with model visibilities."""
    fhd_uv = UVData()
    uvfits_uv = UVData()
    fhd_uv.read(testfiles, use_model=True)

    outfile = str(tmp_path / "outtest_FHD_1061316296_model.uvfits")
    fhd_uv.write_uvfits(
        outfile, spoof_nonessential=True,
    )
    uvfits_uv.read_uvfits(outfile)
    assert fhd_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multi_files():
    """Read multiple files at once."""
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    test1 = list(np.array(testfiles)[[0, 1, 2, 4, 6, 7]])
    test2 = list(np.array(testfiles)[[0, 2, 3, 5, 6, 7]])
    fhd_uv1.read(np.array([test1, test2]), use_model=True, file_type="fhd")

    fhd_uv2.read(testfiles, use_model=True)

    assert uvutils._check_histories(
        fhd_uv2.history + " Combined data " "along polarization axis using pyuvdata.",
        fhd_uv1.history,
    )

    fhd_uv1.history = fhd_uv2.history
    assert fhd_uv1 == fhd_uv2


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multi_files_axis():
    """Read multiple files at once with axis keyword."""
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    test1 = list(np.array(testfiles)[[0, 1, 2, 4, 6, 7]])
    test2 = list(np.array(testfiles)[[0, 2, 3, 5, 6, 7]])
    fhd_uv1.read(
        np.array([test1, test2]), use_model=True, file_type="fhd", axis="polarization"
    )

    fhd_uv2.read(testfiles, use_model=True)

    assert uvutils._check_histories(
        fhd_uv2.history + " Combined data " "along polarization axis using pyuvdata.",
        fhd_uv1.history,
    )

    fhd_uv1.history = fhd_uv2.history
    assert fhd_uv1 == fhd_uv2


def test_single_time():
    """
    test reading in a file with a single time.
    """
    single_time_filelist = glob.glob(os.path.join(DATA_PATH, "refsim1.1_fhd/*"))

    fhd_uv = UVData()
    uvtest.checkWarnings(
        fhd_uv.read,
        func_args=[single_time_filelist],
        nwarnings=2,
        message=[
            "Telescope gaussian is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    )

    assert np.unique(fhd_uv.time_array).size == 1


def test_conjugation():
    """ test uvfits vs fhd conjugation """
    uvfits_file = os.path.join(DATA_PATH, "ref_1.1_uniform.uvfits")
    fhd_filelist = glob.glob(os.path.join(DATA_PATH, "refsim1.1_fhd/*"))

    uvfits_uv = UVData()
    uvfits_uv.read(uvfits_file)

    fhd_uv = UVData()
    uvtest.checkWarnings(
        fhd_uv.read,
        func_args=[fhd_filelist],
        nwarnings=2,
        message=[
            "Telescope gaussian is not in known_telescopes.",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
        ],
    )

    uvfits_uv.select(polarizations=fhd_uv.polarization_array)

    assert uvfits_uv._uvw_array == fhd_uv._uvw_array
    assert uvfits_uv._data_array == fhd_uv._data_array
