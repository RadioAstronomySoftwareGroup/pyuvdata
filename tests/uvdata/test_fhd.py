# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD object."""

import copy
import glob
import os
from shutil import copyfile

import numpy as np
import pytest

from pyuvdata import Telescope, UVData, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings


def get_fhd_files(filelist):
    data_files = []
    model_files = []
    params_file = None
    obs_file = None
    flags_file = None
    layout_file = None
    settings_file = None
    for fname in filelist:
        basename = os.path.basename(fname)
        if "vis_model" in basename:
            model_files.append(fname)
        elif "vis" in basename:
            data_files.append(fname)
        elif "params" in basename:
            params_file = fname
        elif "obs" in basename:
            obs_file = fname
        elif "flag" in basename:
            flags_file = fname
        elif "layout" in basename:
            layout_file = fname
        elif "settings" in basename:
            settings_file = fname

    return {
        "data_files": data_files,
        "model_files": model_files,
        "params_file": params_file,
        "obs_file": obs_file,
        "flags_file": flags_file,
        "layout_file": layout_file,
        "settings_file": settings_file,
    }


@pytest.fixture(scope="function")
def fhd_data(fhd_data_files):
    fhd_uv = UVData()
    fhd_uv.read(**fhd_data_files)

    return fhd_uv


@pytest.fixture(scope="function")
def fhd_model(fhd_model_files):
    fhd_uv = UVData()
    fhd_uv.read(**fhd_model_files)

    return fhd_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_write_read_uvfits(fhd_data, tmp_path, fhd_data_files):
    """
    FHD to uvfits loopback test.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = fhd_data
    uvfits_uv = UVData()

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    all_testfiles = []
    for fname in fhd_data_files.values():
        if isinstance(fname, list):
            for temp in fname:
                all_testfiles.append(temp)
        else:
            all_testfiles.append(fname)

    # make sure filename attributes are correct
    assert set(fhd_uv.filename) == {os.path.basename(fn) for fn in all_testfiles}
    assert uvfits_uv.filename == [os.path.basename(outfile)]
    fhd_uv.filename = uvfits_uv.filename
    fhd_uv._filename.form = (1,)

    # fix up the phase_center_catalogs
    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_metadata_only(fhd_data, fhd_data_files):
    fhd_uv = UVData()
    fhd_uv.read(**fhd_data_files, read_data=False)

    assert fhd_uv.metadata_only

    fhd_uv2 = fhd_data
    fhd_uv3 = fhd_uv2.copy(metadata_only=True)

    assert fhd_uv == fhd_uv3


@pytest.mark.parametrize("multi", [True, False])
def test_read_fhd_metadata_only_error(fhd_data_files, multi):
    fhd_uv = UVData()

    del fhd_data_files["obs_file"]

    if multi:
        for ftype, fnames in fhd_data_files.items():
            if isinstance(fnames, list):
                fhd_data_files[ftype] = [[fnames[0]], [fnames[1]]]
            else:
                fhd_data_files[ftype] = [fnames] * 2

    with pytest.raises(
        ValueError, match="The obs_file parameter must be passed if read_data is False."
    ):
        fhd_uv.read(**fhd_data_files, read_data=False)


def test_read_fhd_select(fhd_data_files):
    """
    test select on read with FHD files.

    Read in FHD files with generic read & select on read, compare to read fhd
    files then do select
    """
    fhd_uv = UVData()
    fhd_uv2 = UVData()

    with check_warnings(
        UserWarning,
        [
            'Warning: select on read keyword set, but file_type is "fhd" which '
            "does not support select on read. Entire file will be read and then "
            "select will be performed",
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file. Using the value from known_telescopes.",
        ],
    ):
        fhd_uv2.read(**fhd_data_files, freq_chans=np.arange(2))

    with check_warnings(
        UserWarning,
        (
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file. Using the value from known_telescopes."
        ),
    ):
        fhd_uv.read(**fhd_data_files)

    fhd_uv.select(freq_chans=np.arange(2))
    assert fhd_uv == fhd_uv2


@pytest.mark.parametrize("multi", [True, False])
def test_read_fhd_write_read_uvfits_no_layout(fhd_data_files, multi):
    """
    Test errors/warnings with with no layout file.
    """
    fhd_uv = UVData()

    del fhd_data_files["layout_file"]

    if multi:
        for ftype, fnames in fhd_data_files.items():
            if isinstance(fnames, list):
                fhd_data_files[ftype] = [[fnames[0]], [fnames[1]]]
            else:
                fhd_data_files[ftype] = [fnames] * 2

    warn_msg = [
        "The layout_file parameter was not passed, so antenna_postions will "
        "not be defined and antenna names and numbers might be incorrect."
    ]
    if multi:
        warn_msg = warn_msg * 2
    with check_warnings(UserWarning, match=warn_msg):
        fhd_uv.read(**fhd_data_files)


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.filterwarnings("ignore:telescope_location is not set. ")
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_fhd_antenna_pos(fhd_data):
    """
    Check that FHD antenna positions are handled as rotated ECEF coords, like uvfits.
    """
    mwa_corr_dir = os.path.join(DATA_PATH, "mwa_corr_fits_testfiles/")

    mwa_corr_files = [
        "1131733552.metafits",
        "1131733552_20151116182537_mini_gpubox01_00.fits",
    ]
    mwa_corr_file_list = [os.path.join(mwa_corr_dir, fname) for fname in mwa_corr_files]

    mwa_corr_obj = UVData()
    mwa_corr_obj.read(
        mwa_corr_file_list,
        correct_cable_len=True,
        phase_to_pointing_center=True,
        read_data=False,
    )

    assert fhd_data.telescope._antenna_names == mwa_corr_obj.telescope._antenna_names
    assert (
        fhd_data.telescope._antenna_positions
        == mwa_corr_obj.telescope._antenna_positions
    )

    cotter_file = os.path.join(DATA_PATH, "1061316296.uvfits")
    cotter_obj = UVData()
    cotter_obj.read(cotter_file)

    # don't test antenna_numbers, they will not match.
    # mwa_corr_fits now uses antenna_numbers that correspond to antenna_names
    # instead of following the cotter convention of using 0-127.
    assert fhd_data.telescope._antenna_names == cotter_obj.telescope._antenna_names
    assert (
        fhd_data.telescope._antenna_positions == cotter_obj.telescope._antenna_positions
    )

    assert (
        mwa_corr_obj.telescope._antenna_positions
        == cotter_obj.telescope._antenna_positions
    )


def test_read_fhd_write_read_uvfits_variant_flag(tmp_path, fhd_data_files):
    """
    FHD to uvfits loopback test with variant flag file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()

    variant_flags_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_variant_flags.sav"
    )
    fhd_data_files["flags_file"] = variant_flags_file

    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['flags']",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file. Using the value from known_telescopes.",
        ],
    ):
        fhd_uv.read(**fhd_data_files)

    os.makedirs(os.path.join(tmp_path, "vis_data"))
    temp_flag_file = copyfile(
        variant_flags_file, os.path.join(tmp_path, "vis_data", "foo.sav")
    )
    fhd_data_files["flags_file"] = temp_flag_file
    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected suffix so prefix matching "
            "could not be done. The affected file types are: ['flags']",
            "The FHD input files do not all have the same parent folder, so they may "
            "not be for the same FHD run.",
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file. Using the value from known_telescopes.",
        ],
    ):
        fhd_uv2 = UVData.from_file(**fhd_data_files)

    assert fhd_uv == fhd_uv2

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )

    assert fhd_uv == uvfits_uv


def test_read_fhd_latlonalt_match_xyz(fhd_data_files):
    fhd_data_files["layout_file"] = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_known_match_xyz_layout.sav"
    )

    fhd_data_files["obs_file"] = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_known_match_latlonalt_obs.sav"
    )

    with check_warnings(
        UserWarning,
        match=[
            "The FHD input files do not all have matching prefixes, so they "
            "may not be for the same data.",
            "Some FHD input files do not have the expected subfolder so FHD "
            "folder matching could not be done. The affected file types are: "
            "['layout', 'obs']",
        ],
    ):
        fhd_uv = UVData.from_file(**fhd_data_files, read_data=False)

    mwa_tel = Telescope.from_known_telescopes("mwa")

    # TODO: why don't these match better?
    np.testing.assert_allclose(
        mwa_tel._location.xyz(), fhd_uv.telescope._location.xyz(), rtol=0, atol=1e-1
    )


def test_read_fhd_write_read_uvfits_fix_layout(tmp_path, fhd_data_files):
    """
    FHD to uvfits loopback test with fixed array center layout file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()

    layout_fixed_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_fixed_arr_center_layout.sav"
    )

    fhd_data_files["layout_file"] = layout_fixed_file
    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['layout']",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
        ],
    ):
        fhd_uv.read(**fhd_data_files)

    os.makedirs(os.path.join(tmp_path, "fhd_vis_data2", "metadata"))
    temp_layout_file = copyfile(
        layout_fixed_file,
        os.path.join(
            tmp_path,
            "fhd_vis_data2",
            "metadata",
            "1061316296_fixed_arr_center_layout.sav",
        ),
    )
    fhd_data_files["layout_file"] = temp_layout_file
    with check_warnings(
        UserWarning,
        match=[
            "The FHD input files do not all have the same parent folder, so they may "
            "not be for the same FHD run.",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
        ],
    ):
        fhd_uv.read(**fhd_data_files)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")

    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )

    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_fix_layout_bad_obs_loc(tmp_path, fhd_data_files):
    """
    FHD to uvfits loopback test with fixed array center layout file, bad obs location.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    bad_obs_loc_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_bad_obs_loc_vis_XX.sav"
    )
    layout_fixed_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_fixed_arr_center_layout.sav"
    )
    messages = [
        "Telescope location derived from obs",
        "tile_names from obs structure does not match",
        "Some FHD input files do not have the expected subfolder so FHD folder "
        "matching could not be done. The affected file types are: ['vis', 'layout']",
        "The FHD input files do not all have matching prefixes, so they may not be "
        "for the same data.",
    ]

    fhd_data_files["filename"] = [bad_obs_loc_file]
    fhd_data_files["layout_file"] = layout_fixed_file

    with check_warnings(UserWarning, match=messages):
        fhd_uv.read(**fhd_data_files)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_bad_obs_loc(tmp_path, fhd_data_files):
    """
    FHD to uvfits loopback test with bad obs location (and bad layout location).

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    bad_obs_loc_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_bad_obs_loc_vis_XX.sav"
    )
    messages = [
        "Telescope location derived from obs",
        "tile_names from obs structure does not match",
        "Some FHD input files do not have the expected subfolder so FHD folder "
        "matching could not be done. The affected file types are: ['vis']",
        "The FHD input files do not all have matching prefixes, so they may not be "
        "for the same data.",
    ]

    fhd_data_files["filename"] = [bad_obs_loc_file]

    with check_warnings(UserWarning, match=messages):
        fhd_uv.read(**fhd_data_files)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_altered_layout(tmp_path, fhd_data_files):
    """
    FHD to uvfits loopback test with altered layout file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()

    # bad layout structure values
    altered_layout_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_broken_layout.sav"
    )

    fhd_data_files["layout_file"] = altered_layout_file

    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['layout']",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
        ],
    ):
        fhd_uv.read(**fhd_data_files)

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


@pytest.mark.parametrize("multi", [True, False])
def test_read_fhd_write_read_uvfits_no_settings(tmp_path, fhd_data_files, multi):
    """
    FHD to uvfits loopback test with no settings file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    messages = [
        "The settings_file parameter was not passed, so some history information will "
        "be missing.",
        "Telescope location derived from obs lat/lon/alt values does not match the "
        "location in the layout file. Using the value from known_telescopes.",
    ]
    del fhd_data_files["settings_file"]
    if multi:
        messages *= 2
        for ftype, fnames in fhd_data_files.items():
            if isinstance(fnames, list):
                fhd_data_files[ftype] = [[fnames[0]], [fnames[1]]]
            else:
                fhd_data_files[ftype] = [fnames] * 2

    with check_warnings(UserWarning, match=messages):
        fhd_uv.read(**fhd_data_files)

    if not multi:
        # Check only pyuvdata history with no settings file
        assert fhd_uv.history == fhd_uv.pyuvdata_version_str

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


def test_break_read_fhd(fhd_data_files, fhd_model_files):
    """Try various cases of incomplete file lists."""
    fhd_uv = UVData()
    # missing flags

    file_dict = copy.deepcopy(fhd_data_files)
    del file_dict["flags_file"]
    with pytest.raises(
        ValueError, match="The flags_file parameter must be passed if read_data is True"
    ):
        fhd_uv.read(**file_dict)

    for ftype, fnames in file_dict.items():
        if isinstance(fnames, list):
            file_dict[ftype] = [[fnames[0]], [fnames[1]]]
        else:
            file_dict[ftype] = [fnames] * 2

    with pytest.raises(
        ValueError, match="The flags_file parameter must be passed if read_data is True"
    ):
        fhd_uv.read(**file_dict)

    file_dict = copy.deepcopy(fhd_data_files)
    del file_dict["params_file"]
    # Missing params
    with pytest.raises(
        ValueError, match="The params_file must be passed for FHD files."
    ):
        fhd_uv.read(**file_dict)

    file_dict = copy.deepcopy(fhd_data_files)
    file_dict["filename"] = ["foo.sav"]
    # No data files
    with pytest.raises(ValueError, match="unrecognized file in vis_files"):
        fhd_uv.read(**file_dict)

    file_dict["filename"] = [None]
    # No data files
    with pytest.raises(
        ValueError, match="The vis_files parameter must be passed if read_data is True"
    ):
        fhd_uv.read(**file_dict, file_type="fhd")

    # mix of model & data files

    file_dict["filename"] = [
        fhd_data_files["filename"][0],
        fhd_model_files["filename"][1],
    ]
    with pytest.raises(
        ValueError, match="The vis_files parameter has a mix of model and data files."
    ):
        fhd_uv.read(**file_dict, file_type="fhd")


def test_read_fhd_warnings(fhd_data_files):
    """Test warnings with various broken inputs."""
    # bad obs structure values
    broken_data_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_broken_vis_XX.sav"
    )

    warn_messages = [
        "Ntimes does not match",
        "Telescope location derived from obs",
        "These visibilities may have been phased improperly",
        "Nbls does not match",
        "Some FHD input files do not have the expected subfolder so FHD folder "
        "matching could not be done. The affected file types are: ['vis']",
        "The FHD input files do not all have matching prefixes, so they may not be for "
        "the same data.",
    ]
    fhd_uv = UVData()

    fhd_data_files["filename"] = broken_data_file

    with check_warnings(UserWarning, match=warn_messages):
        fhd_uv.read(**fhd_data_files, run_check=False)

    # bad flag file
    broken_flags_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_broken_flags.sav"
    )

    fhd_data_files["flags_file"] = broken_flags_file
    fhd_uv = UVData()
    with (
        check_warnings(
            UserWarning,
            match=[
                "Some FHD input files do not have the expected subfolder so FHD "
                "folder matching could not be done. The affected file types are: "
                "['vis', 'flags']",
                "The FHD input files do not all have matching prefixes, so they "
                "may not be for the same data.",
            ],
        ),
        pytest.raises(
            ValueError, match="No recognized key for visibility weights in flags_file."
        ),
    ):
        fhd_uv.read(**fhd_data_files)


@pytest.mark.parametrize(
    "new_file_end,file_copy_ind,message",
    [
        (["extra_vis_XX.sav"], 0, "multiple xx datafiles in vis_files"),
        (["extra_vis_YY.sav"], 1, "multiple yy datafiles in vis_files"),
        (["vis_XY.sav", "extra_vis_XY.sav"], 0, "multiple xy datafiles in vis_files"),
        (["vis_YX.sav", "extra_vis_YX.sav"], 0, "multiple yx datafiles in vis_files"),
    ],
)
def test_read_fhd_extra_files(
    tmp_path, new_file_end, file_copy_ind, message, fhd_data_files
):
    # try cases with extra files of each type
    new_files = []
    for file_end in new_file_end:
        extra_file = os.path.join(tmp_path, "1061316296_" + file_end)
        new_files.append(extra_file)
        copyfile(fhd_data_files["filename"][file_copy_ind], extra_file)

    fhd_data_files["filename"].extend(new_files)
    fhd_uv = UVData()
    with pytest.raises(ValueError, match=message):
        fhd_uv.read(**fhd_data_files)


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_model(tmp_path, fhd_model):
    """FHD to uvfits loopback test with model visibilities."""
    fhd_uv = fhd_model
    uvfits_uv = UVData()

    outfile = str(tmp_path / "outtest_FHD_1061316296_model.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.parametrize("axis", [None, "polarization"])
def test_multi_files(fhd_model, axis, fhd_model_files):
    """Read multiple files at once."""
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    for ftype, fnames in fhd_model_files.items():
        if isinstance(fnames, list):
            fhd_model_files[ftype] = [[fnames[0]], [fnames[1]]]
        else:
            fhd_model_files[ftype] = [fnames] * 2
    fhd_model_files["filename"] = np.array(
        [fhd_model_files["filename"][0], fhd_model_files["filename"][1]]
    )
    fhd_uv1.read(**fhd_model_files, file_type="fhd", axis=axis)

    fhd_uv2 = fhd_model

    assert utils.history._check_histories(
        fhd_uv2.history + " Combined data along polarization axis using pyuvdata.",
        fhd_uv1.history,
    )

    fhd_uv1.history = fhd_uv2.history

    assert fhd_uv1 == fhd_uv2


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.parametrize(
    "ftype_err",
    ["params_file", "obs_file", "flags_file", "layout_file", "settings_file"],
)
def test_multi_files_errors(fhd_model, fhd_model_files, ftype_err):
    fhd_uv1 = UVData()

    for ftype, fnames in fhd_model_files.items():
        if isinstance(fnames, list):
            fhd_model_files[ftype] = [[fnames[0]], [fnames[1]]]
        else:
            fhd_model_files[ftype] = [fnames] * 2
    fhd_model_files["filename"] = np.array(
        [fhd_model_files["filename"][0], fhd_model_files["filename"][1]]
    )

    fhd_model_files[ftype_err] = [fhd_model_files[ftype_err]] * 3

    msg = "For multiple FHD files, "
    if ftype_err == "params_file":
        msg += "the number of params_file"
    else:
        ftype_name = ftype_err
        msg += "if " + ftype_name + " is passed, the number of " + ftype_name

    msg += " values must match the number of data file sets."
    with pytest.raises(ValueError, match=msg):
        fhd_uv1.read(**fhd_model_files, file_type="fhd", axis="polarization")


def test_single_time():
    """
    test reading in a file with a single time.
    """
    single_time_filelist = glob.glob(os.path.join(DATA_PATH, "refsim1.1_fhd/*"))
    file_dict = get_fhd_files(single_time_filelist)
    file_dict["filename"] = file_dict["data_files"]
    del file_dict["data_files"]
    del file_dict["model_files"]

    fhd_uv = UVData()
    with check_warnings(
        UserWarning,
        [
            "tile_names from obs structure does not match",
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['vis', 'vis', "
            "'flags', 'layout', 'params', 'settings']",
        ],
    ):
        fhd_uv.read(**file_dict)

    assert np.unique(fhd_uv.time_array).size == 1


def test_conjugation():
    """test uvfits vs fhd conjugation"""
    uvfits_file = os.path.join(DATA_PATH, "ref_1.1_uniform.uvfits")
    single_time_filelist = glob.glob(os.path.join(DATA_PATH, "refsim1.1_fhd/*"))
    file_dict = get_fhd_files(single_time_filelist)
    file_dict["filename"] = file_dict["data_files"]
    del file_dict["data_files"]
    del file_dict["model_files"]

    uvfits_uv = UVData()
    uvfits_uv.read(uvfits_file)

    fhd_uv = UVData()
    with check_warnings(
        UserWarning,
        [
            "tile_names from obs structure does not match",
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['vis', 'vis', "
            "'flags', 'layout', 'params', 'settings']",
        ],
    ):
        fhd_uv.read(**file_dict)

    uvfits_uv.select(polarizations=fhd_uv.polarization_array)

    assert uvfits_uv._uvw_array == fhd_uv._uvw_array
    assert uvfits_uv._data_array == fhd_uv._data_array
