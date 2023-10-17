# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD object.

"""
import glob
import os
from shutil import copyfile

import numpy as np
import pytest

import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.uvdata.uvdata import _future_array_shapes_warning


def get_fhd_files(filelist):
    data_files = []
    model_files = []
    params_file = None
    obs_file = None
    flag_file = None
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
            flag_file = fname
        elif "layout" in basename:
            layout_file = fname
        elif "settings" in basename:
            settings_file = fname

    return (
        data_files,
        model_files,
        params_file,
        obs_file,
        flag_file,
        layout_file,
        settings_file,
    )


@pytest.fixture(scope="function")
def fhd_data(fhd_test_files):
    fhd_uv = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    fhd_uv.read(
        tf_data,
        params_file=tf_params,
        obs_file=tf_obs,
        flag_file=tf_flag,
        layout_file=tf_layout,
        settings_file=tf_stngs,
        use_future_array_shapes=True,
    )

    return fhd_uv


@pytest.fixture(scope="function")
def fhd_model(fhd_test_files):
    fhd_uv = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    fhd_uv.read(
        tf_model,
        params_file=tf_params,
        obs_file=tf_obs,
        flag_file=tf_flag,
        layout_file=tf_layout,
        settings_file=tf_stngs,
        use_future_array_shapes=True,
    )

    return fhd_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_write_read_uvfits(fhd_data, tmp_path, fhd_test_files):
    """
    FHD to uvfits loopback test.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = fhd_data
    uvfits_uv = UVData()

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    all_testfiles = list(fhd_test_files)
    for fname in fhd_test_files:
        if isinstance(fname, list):
            all_testfiles.remove(fname)
            all_testfiles.extend(fname)

    all_testfiles_data = []
    for fname in all_testfiles:
        temp = os.path.basename(fname)
        if "vis_model" not in temp:
            all_testfiles_data.append(temp)

    # make sure filename attributes are correct
    assert set(fhd_uv.filename) == {os.path.basename(fn) for fn in all_testfiles_data}
    assert uvfits_uv.filename == [os.path.basename(outfile)]
    fhd_uv.filename = uvfits_uv.filename
    fhd_uv._filename.form = (1,)

    # fix up the phase_center_catalogs
    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_metadata_only(fhd_data, fhd_test_files):
    fhd_uv = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    fhd_uv.read(
        tf_data,
        params_file=tf_params,
        obs_file=tf_obs,
        flag_file=tf_flag,
        layout_file=tf_layout,
        settings_file=tf_stngs,
        read_data=False,
        use_future_array_shapes=True,
    )

    assert fhd_uv.metadata_only

    fhd_uv2 = fhd_data
    fhd_uv3 = fhd_uv2.copy(metadata_only=True)

    assert fhd_uv == fhd_uv3


def test_read_fhd_metadata_only_error(fhd_test_files):
    fhd_uv = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    with pytest.raises(
        ValueError, match="The obs_file parameter must be passed if read_data is False."
    ):
        fhd_uv.read(
            tf_data,
            params_file=tf_params,
            flag_file=tf_flag,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            read_data=False,
            use_future_array_shapes=True,
        )


def test_read_fhd_select(fhd_test_files):
    """
    test select on read with FHD files.

    Read in FHD files with generic read & select on read, compare to read fhd
    files then do select
    """
    fhd_uv = UVData()
    fhd_uv2 = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    with uvtest.check_warnings(
        [UserWarning, UserWarning, DeprecationWarning],
        [
            'Warning: select on read keyword set, but file_type is "fhd" which '
            "does not support select on read. Entire file will be read and then "
            "select will be performed",
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file. Using the value from known_telescopes.",
            _future_array_shapes_warning,
        ],
    ):
        fhd_uv2.read(
            tf_data,
            params_file=tf_params,
            obs_file=tf_obs,
            flag_file=tf_flag,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            freq_chans=np.arange(2),
        )

    with uvtest.check_warnings(
        [UserWarning, DeprecationWarning],
        [
            "Telescope location derived from obs lat/lon/alt values does not match the "
            "location in the layout file. Using the value from known_telescopes.",
            _future_array_shapes_warning,
        ],
    ):
        fhd_uv.read(
            tf_data,
            params_file=tf_params,
            obs_file=tf_obs,
            flag_file=tf_flag,
            layout_file=tf_layout,
            settings_file=tf_stngs,
        )

    fhd_uv.select(freq_chans=np.arange(2))
    assert fhd_uv == fhd_uv2


def test_read_fhd_write_read_uvfits_no_layout(fhd_test_files):
    """
    Test errors/warnings with with no layout file.
    """
    fhd_uv = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files

    # check warning raised
    with uvtest.check_warnings(
        UserWarning,
        match="The layout_file parameter was not passed, so antenna_postions will not "
        "be defined and antenna names and numbers might be incorrect.",
    ):
        fhd_uv.read(
            tf_data,
            params_file=tf_params,
            obs_file=tf_obs,
            flag_file=tf_flag,
            settings_file=tf_stngs,
            run_check=False,
            use_future_array_shapes=True,
        )

    with pytest.raises(
        ValueError, match="Required UVParameter _antenna_positions has not been set"
    ):
        with uvtest.check_warnings(UserWarning, "No layout file"):
            fhd_uv.read(
                tf_data,
                params_file=tf_params,
                obs_file=tf_obs,
                flag_file=tf_flag,
                settings_file=tf_stngs,
                use_future_array_shapes=True,
            )


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
        use_future_array_shapes=True,
    )

    assert fhd_data._antenna_names == mwa_corr_obj._antenna_names
    assert fhd_data._antenna_positions == mwa_corr_obj._antenna_positions

    cotter_file = os.path.join(DATA_PATH, "1061316296.uvfits")
    cotter_obj = UVData()
    cotter_obj.read(cotter_file, use_future_array_shapes=True)

    # don't test antenna_numbers, they will not match.
    # mwa_corr_fits now uses antenna_numbers that correspond to antenna_names
    # instead of following the cotter convention of using 0-127.
    assert fhd_data._antenna_names == cotter_obj._antenna_names
    assert fhd_data._antenna_positions == cotter_obj._antenna_positions

    assert mwa_corr_obj._antenna_positions == cotter_obj._antenna_positions


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_write_read_uvfits_variant_flag(tmp_path, fhd_test_files):
    """
    FHD to uvfits loopback test with variant flag file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    variant_flag_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_variant_flags.sav"
    )
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    fhd_uv.read(
        tf_data,
        params_file=tf_params,
        obs_file=tf_obs,
        flag_file=variant_flag_file,
        layout_file=tf_layout,
        settings_file=tf_stngs,
        use_future_array_shapes=True,
    )

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )

    assert fhd_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
def test_read_fhd_write_read_uvfits_fix_layout(tmp_path, fhd_test_files):
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

    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    fhd_uv.read(
        tf_data,
        params_file=tf_params,
        obs_file=tf_obs,
        flag_file=tf_flag,
        layout_file=layout_fixed_file,
        settings_file=tf_stngs,
        use_future_array_shapes=True,
    )

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")

    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )

    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_fix_layout_bad_obs_loc(tmp_path, fhd_test_files):
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
    ]

    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    with uvtest.check_warnings(UserWarning, messages):
        fhd_uv.read(
            [bad_obs_loc_file],
            params_file=tf_params,
            obs_file=tf_obs,
            flag_file=tf_flag,
            layout_file=layout_fixed_file,
            settings_file=tf_stngs,
            use_future_array_shapes=True,
        )

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_bad_obs_loc(tmp_path, fhd_test_files):
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
    ]
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    with uvtest.check_warnings(UserWarning, messages):
        fhd_uv.read(
            [bad_obs_loc_file],
            params_file=tf_params,
            obs_file=tf_obs,
            flag_file=tf_flag,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            use_future_array_shapes=True,
        )

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_altered_layout(tmp_path, fhd_test_files):
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
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    fhd_uv.read(
        tf_data,
        params_file=tf_params,
        obs_file=tf_obs,
        flag_file=tf_flag,
        layout_file=altered_layout_file,
        settings_file=tf_stngs,
        use_future_array_shapes=True,
    )

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


def test_read_fhd_write_read_uvfits_no_settings(tmp_path, fhd_test_files):
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
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    with uvtest.check_warnings(UserWarning, messages):
        fhd_uv.read(
            tf_data,
            params_file=tf_params,
            obs_file=tf_obs,
            flag_file=tf_flag,
            layout_file=tf_layout,
            use_future_array_shapes=True,
        )

    # Check only pyuvdata history with no settings file
    assert fhd_uv.history == fhd_uv.pyuvdata_version_str

    outfile = str(tmp_path / "outtest_FHD_1061316296.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


def test_break_read_fhd(fhd_test_files):
    """Try various cases of incomplete file lists."""
    fhd_uv = UVData()
    # missing flags
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    with pytest.raises(
        ValueError, match="The flag_file parameter must be passed if read_data is True"
    ):
        fhd_uv.read(
            tf_data,
            params_file=tf_params,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            use_future_array_shapes=True,
        )

    # Missing params
    with pytest.raises(
        ValueError, match="The params_file must be passed for FHD files."
    ):
        fhd_uv.read(
            tf_data,
            flag_file=tf_flag,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            use_future_array_shapes=True,
        )

    # No data files
    with pytest.raises(ValueError, match="unrecognized file in vis_files"):
        fhd_uv.read(
            ["foo.sav"],
            params_file=tf_params,
            flag_file=tf_flag,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            use_future_array_shapes=True,
        )

    # No data files
    with pytest.raises(
        ValueError, match="The vis_files parameter must be passed if read_data is True"
    ):
        fhd_uv.read(
            [None],
            params_file=tf_params,
            flag_file=tf_flag,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            file_type="fhd",
            use_future_array_shapes=True,
        )

    # mix of model & data files
    with pytest.raises(
        ValueError,
        match="The vis_files parameter has a mix of model and in data files.",
    ):
        fhd_uv.read(
            [tf_data[0], tf_model[1]],
            params_file=tf_params,
            flag_file=tf_flag,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            file_type="fhd",
            use_future_array_shapes=True,
        )


def test_read_fhd_warnings(fhd_test_files):
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
    ]
    fhd_uv = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    with uvtest.check_warnings(UserWarning, warn_messages):
        fhd_uv.read(
            broken_data_file,
            params_file=tf_params,
            flag_file=tf_flag,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            run_check=False,
            use_future_array_shapes=True,
        )

    # bad flag file
    broken_flag_file = os.path.join(
        DATA_PATH, "fhd_vis_data/", "1061316296_broken_flags.sav"
    )
    fhd_uv = UVData()
    with pytest.raises(
        ValueError, match="No recognized key for visibility weights in flag_file."
    ):
        fhd_uv.read(
            tf_data,
            params_file=tf_params,
            flag_file=broken_flag_file,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            use_future_array_shapes=True,
        )


@pytest.mark.parametrize(
    "new_file_end,file_copy_ind,message",
    [
        (["extra_vis_XX.sav"], 0, "multiple xx datafiles in vis_files"),
        (["extra_vis_YY.sav"], 1, "multiple yy datafiles in vis_files"),
        (["vis_XY.sav", "extra_vis_XY.sav"], 0, "multiple xy datafiles in vis_files"),
        (["vis_YX.sav", "extra_vis_YX.sav"], 0, "multiple yx datafiles in vis_files"),
    ],
)
def test_read_fhd_extra_files(new_file_end, file_copy_ind, message, fhd_test_files):
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    # try cases with extra files of each type
    new_files = []
    for file_end in new_file_end:
        extra_file = os.path.join(DATA_PATH, "fhd_vis_data/", "1061316296_" + file_end)
        new_files.append(extra_file)
        copyfile(tf_data[file_copy_ind], extra_file)
    fhd_uv = UVData()
    with pytest.raises(ValueError, match=message):
        fhd_uv.read(
            tf_data + new_files,
            params_file=tf_params,
            flag_file=tf_flag,
            obs_file=tf_obs,
            layout_file=tf_layout,
            settings_file=tf_stngs,
            use_future_array_shapes=True,
        )
    for extra_file in new_files:
        os.remove(extra_file)


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
def test_read_fhd_model(tmp_path, fhd_model):
    """FHD to uvfits loopback test with model visibilities."""
    fhd_uv = fhd_model
    uvfits_uv = UVData()

    outfile = str(tmp_path / "outtest_FHD_1061316296_model.uvfits")
    fhd_uv.write_uvfits(outfile)
    uvfits_uv.read_uvfits(outfile, use_future_array_shapes=True)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=fhd_uv.phase_center_catalog
    )
    assert fhd_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs")
@pytest.mark.parametrize("axis", [None, "polarization"])
def test_multi_files(fhd_model, axis, fhd_test_files):
    """Read multiple files at once."""
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    tf_data, tf_model, tf_params, tf_obs, tf_flag, tf_layout, tf_stngs = fhd_test_files
    test1 = [tf_model[0]]
    test2 = [tf_model[1]]
    fhd_uv1.read(
        np.array([test1, test2]),
        params_file=[tf_params, tf_params],
        flag_file=[tf_flag, tf_flag],
        obs_file=[tf_obs, tf_obs],
        layout_file=[tf_layout, tf_layout],
        settings_file=[tf_stngs, tf_stngs],
        file_type="fhd",
        axis=axis,
        use_future_array_shapes=True,
    )

    fhd_uv2 = fhd_model

    assert uvutils._check_histories(
        fhd_uv2.history + " Combined data along polarization axis using pyuvdata.",
        fhd_uv1.history,
    )

    fhd_uv1.history = fhd_uv2.history

    assert fhd_uv1 == fhd_uv2


def test_single_time():
    """
    test reading in a file with a single time.
    """
    single_time_filelist = glob.glob(os.path.join(DATA_PATH, "refsim1.1_fhd/*"))
    (
        data_files,
        model_files,
        params_file,
        obs_file,
        flag_file,
        layout_file,
        settings_file,
    ) = get_fhd_files(single_time_filelist)

    fhd_uv = UVData()
    with uvtest.check_warnings(
        UserWarning,
        [
            "tile_names from obs structure does not match",
            "Telescope location derived from obs lat/lon/alt",
        ],
    ):
        fhd_uv.read(
            data_files,
            params_file=params_file,
            flag_file=flag_file,
            obs_file=obs_file,
            layout_file=layout_file,
            settings_file=settings_file,
            use_future_array_shapes=True,
        )

    assert np.unique(fhd_uv.time_array).size == 1


def test_conjugation():
    """test uvfits vs fhd conjugation"""
    uvfits_file = os.path.join(DATA_PATH, "ref_1.1_uniform.uvfits")
    single_time_filelist = glob.glob(os.path.join(DATA_PATH, "refsim1.1_fhd/*"))
    (
        data_files,
        model_files,
        params_file,
        obs_file,
        flag_file,
        layout_file,
        settings_file,
    ) = get_fhd_files(single_time_filelist)
    uvfits_uv = UVData()
    uvfits_uv.read(uvfits_file, use_future_array_shapes=True)

    fhd_uv = UVData()
    with uvtest.check_warnings(
        UserWarning,
        [
            "tile_names from obs structure does not match",
            "Telescope location derived from obs lat/lon/alt",
        ],
    ):
        fhd_uv.read(
            data_files,
            params_file=params_file,
            flag_file=flag_file,
            obs_file=obs_file,
            layout_file=layout_file,
            settings_file=settings_file,
            use_future_array_shapes=True,
        )

    uvfits_uv.select(polarizations=fhd_uv.polarization_array)

    assert uvfits_uv._uvw_array == fhd_uv._uvw_array
    assert uvfits_uv._data_array == fhd_uv._data_array
