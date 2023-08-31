# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD_cal object."""
import os

import numpy as np
import pytest

import pyuvdata.tests as uvtest
from pyuvdata import UVCal
from pyuvdata.data import DATA_PATH
from pyuvdata.uvcal.uvcal import _future_array_shapes_warning

# set up FHD files
testdir = os.path.join(DATA_PATH, "fhd_cal_data/")
testfile_prefix = "1061316296_"
obs_testfile = os.path.join(testdir, testfile_prefix + "obs.sav")
cal_testfile = os.path.join(testdir, testfile_prefix + "cal.sav")
settings_testfile = os.path.join(testdir, testfile_prefix + "settings.txt")
settings_testfile_nodiffuse = os.path.join(
    testdir, testfile_prefix + "nodiffuse_settings.txt"
)
layout_testfile = os.path.join(testdir, testfile_prefix + "layout.sav")

testdir2 = os.path.join(DATA_PATH, "fhd_cal_data/set2")
obs_file_multi = [obs_testfile, os.path.join(testdir2, testfile_prefix + "obs.sav")]
cal_file_multi = [cal_testfile, os.path.join(testdir2, testfile_prefix + "cal.sav")]
layout_file_multi = [layout_testfile, layout_testfile]
settings_file_multi = [
    settings_testfile,
    os.path.join(testdir2, testfile_prefix + "settings.txt"),
]


@pytest.mark.parametrize("raw", [True, False])
def test_read_fhdcal_write_read_calfits(raw, fhd_cal_raw, fhd_cal_fit, tmp_path):
    """
    FHD cal to calfits loopback test.

    Read in FHD cal files, write out as calfits, read back in and check for
    object equality.
    """
    if raw:
        fhd_cal = fhd_cal_raw
    else:
        fhd_cal = fhd_cal_fit

    filelist = [cal_testfile, obs_testfile, layout_testfile, settings_testfile]

    assert fhd_cal.filename == sorted(os.path.basename(file) for file in filelist)
    assert np.max(fhd_cal.gain_array) < 2.0

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal = UVCal.from_file(outfile, use_future_array_shapes=True)
    assert fhd_cal == calfits_cal


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
@pytest.mark.parametrize("raw", [True, False])
def test_read_fhdcal_metadata(raw, fhd_cal_raw, fhd_cal_fit):
    """
    Test FHD cal metadata only read.
    """
    if raw:
        fhd_cal_full = fhd_cal_raw
    else:
        fhd_cal_full = fhd_cal_fit

    with uvtest.check_warnings(
        [DeprecationWarning, UserWarning],
        match=[
            _future_array_shapes_warning,
            "Telescope location derived from obs lat/lon/alt",
        ],
    ):
        fhd_cal = UVCal.from_file(
            cal_testfile,
            obs_file=obs_testfile,
            layout_file=layout_testfile,
            settings_file=settings_testfile,
            raw=raw,
            read_data=False,
        )
    fhd_cal.use_future_array_shapes()

    fhd_cal2 = fhd_cal_full.copy(metadata_only=True)

    # this file set has a mismatch in Nsources between the cal file & settings
    # file for some reason. I think it's just an issue with the files chosen
    assert fhd_cal.Nsources != fhd_cal2.Nsources
    fhd_cal.Nsources = fhd_cal2.Nsources

    # there is a loss in precision for float auto scale values in the
    # settings file vs the cal file
    # first check that they are similar (extract from the string they are packed in)
    assert np.allclose(
        np.asarray(fhd_cal.extra_keywords["AUTOSCAL"][1:-1].split(", "), dtype=float),
        np.asarray(fhd_cal2.extra_keywords["AUTOSCAL"][1:-1].split(", "), dtype=float),
    )
    # replace the strings to prevent errors
    fhd_cal.extra_keywords["autoscal".upper()] = fhd_cal2.extra_keywords[
        "autoscal".upper()
    ]

    assert fhd_cal == fhd_cal2

    # test that no diffuse is properly picked up from the settings file when
    # read_data is False
    fhd_cal = UVCal.from_file(
        cal_testfile,
        obs_file=obs_testfile,
        layout_file=layout_testfile,
        settings_file=settings_testfile_nodiffuse,
        raw=raw,
        read_data=False,
        use_future_array_shapes=True,
    )

    assert fhd_cal.diffuse_model is None

    return


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
def test_read_fhdcal_multimode():
    """
    Read cal with multiple mode_fit values.
    """
    fhd_cal = UVCal.from_file(
        os.path.join(testdir, testfile_prefix + "multimode_cal.sav"),
        obs_file=obs_testfile,
        layout_file=layout_testfile,
        settings_file=os.path.join(testdir, testfile_prefix + "multimode_settings.txt"),
        raw=False,
        use_future_array_shapes=True,
    )
    assert fhd_cal.extra_keywords["MODE_FIT"] == "[90, 150, 230, 320, 400, 524]"

    fhd_cal2 = fhd_cal.copy(metadata_only=True)

    # check metadata only read
    fhd_cal = UVCal.from_file(
        os.path.join(testdir, testfile_prefix + "multimode_cal.sav"),
        obs_file=obs_testfile,
        layout_file=layout_testfile,
        settings_file=os.path.join(testdir, testfile_prefix + "multimode_settings.txt"),
        raw=False,
        read_data=False,
        use_future_array_shapes=True,
    )
    # this file set has a mismatch in Nsources between the cal file & settings
    # file for some reason. I think it's just an issue with the files chosen
    assert fhd_cal.Nsources != fhd_cal2.Nsources
    fhd_cal.Nsources = fhd_cal2.Nsources

    # there is a loss in precision for float auto scale values in the
    # settings file vs the cal file
    assert (
        fhd_cal.extra_keywords["autoscal".upper()]
        != fhd_cal2.extra_keywords["autoscal".upper()]
    )
    fhd_cal.extra_keywords["autoscal".upper()] = fhd_cal2.extra_keywords[
        "autoscal".upper()
    ]
    assert fhd_cal == fhd_cal2

    return


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
@pytest.mark.parametrize(
    "extra_history",
    [
        "Some extra history for testing\n",
        ["Some extra history for testing", "And some more history as well"],
    ],
)
def test_extra_history(extra_history, tmp_path):
    """Test that setting the extra_history keyword works."""
    fhd_cal = UVCal.from_file(
        cal_testfile,
        obs_file=obs_testfile,
        layout_file=layout_testfile,
        settings_file=settings_testfile,
        extra_history=extra_history,
        use_future_array_shapes=True,
    )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal = UVCal.from_file(outfile, use_future_array_shapes=True)
    assert fhd_cal == calfits_cal
    for line in extra_history:
        assert line in fhd_cal.history

    return


@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
def test_flags_galaxy(tmp_path):
    """Test files with time, freq and tile flags and galaxy models behave."""
    testdir = os.path.join(DATA_PATH, "fhd_cal_data/flag_set")
    obs_testfile_flag = os.path.join(testdir, testfile_prefix + "obs.sav")
    cal_testfile_flag = os.path.join(testdir, testfile_prefix + "cal.sav")
    settings_testfile_flag = os.path.join(testdir, testfile_prefix + "settings.txt")
    layout_testfile_flag = os.path.join(testdir, testfile_prefix + "layout.sav")

    with uvtest.check_warnings(
        UserWarning,
        match=[
            "tile_names from obs structure does not match",
            "Telescope location derived from obs lat/lon/alt",
        ],
    ):
        fhd_cal = UVCal.from_file(
            cal_testfile_flag,
            obs_file=obs_testfile_flag,
            layout_file=layout_testfile_flag,
            settings_file=settings_testfile_flag,
            use_future_array_shapes=True,
        )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal = UVCal.from_file(outfile, use_future_array_shapes=True)
    assert fhd_cal == calfits_cal


def test_unknown_telescope():
    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Telescope foo is not in known_telescopes.",
            "Telescope location derived from obs lat/lon/alt",
        ],
    ):
        fhd_cal = UVCal.from_file(
            cal_testfile,
            obs_file=os.path.join(testdir, testfile_prefix + "telescopefoo_obs.sav"),
            layout_file=layout_testfile,
            settings_file=settings_testfile,
            use_future_array_shapes=True,
        )
    assert fhd_cal.telescope_name == "foo"


@pytest.mark.parametrize(
    "cal_file,obs_file,layout_file,settings_file,nfiles",
    [
        [cal_testfile, obs_testfile, layout_testfile, settings_testfile, 1],
        [cal_file_multi, obs_file_multi, layout_file_multi, settings_file_multi, 2],
    ],
)
def test_break_read_fhdcal(cal_file, obs_file, layout_file, settings_file, nfiles):
    """Try various cases of missing files."""
    # check useful error message for metadata only read with no settings file
    with pytest.raises(
        ValueError, match="A settings_file must be provided if read_data is False."
    ):
        fhd_cal = UVCal.from_file(
            cal_file,
            obs_file=obs_file,
            layout_file=layout_file,
            read_data=False,
            use_future_array_shapes=True,
        )

    message_list = [
        "No settings file",
        "Telescope location derived from obs lat/lon/alt",
    ]
    if nfiles > 1:
        message_list *= 2
        message_list.append("UVParameter diffuse_model does not match")

    with uvtest.check_warnings(UserWarning, message_list):
        fhd_cal = UVCal.from_file(
            cal_file,
            obs_file=obs_file,
            layout_file=layout_file,
            use_future_array_shapes=True,
        )

    # Check only pyuvdata version history with no settings file
    expected_history = "\n" + fhd_cal.pyuvdata_version_str
    if nfiles > 1:
        expected_history += " Combined data along time axis using pyuvdata."
    assert fhd_cal.history == expected_history

    message_list = [
        "No layout file, antenna_postions will not be defined.",
        "antenna_positions are not set or are being overwritten. Using known values "
        "for mwa.",
    ] * nfiles + ["UVParameter diffuse_model does not match"] * (nfiles - 1)

    warning_list = [UserWarning] * (3 * nfiles - 1)

    if nfiles > 1:
        warning_list += [DeprecationWarning]
        message_list += [
            "Reading multiple files from file specific read methods is deprecated. Use "
            "the generic `UVCal.read` method instead. This will become an error in "
            "version 2.5"
        ]

    with uvtest.check_warnings(warning_list, match=message_list):
        fhd_cal.read_fhd_cal(
            cal_file,
            obs_file,
            settings_file=settings_file,
            use_future_array_shapes=True,
        )

    with pytest.raises(
        ValueError, match="A settings_file must be provided if read_data is False."
    ):
        with uvtest.check_warnings(
            [DeprecationWarning],
            match="Reading multiple files from file specific read methods is "
            "deprecated. Use the generic `UVCal.read` method instead. This will become "
            "an error in version 2.5",
        ):
            fhd_cal.read_fhd_cal(
                cal_file,
                obs_file,
                layout_file=layout_file,
                read_data=False,
                use_future_array_shapes=True,
            )


@pytest.mark.parametrize(
    ["concat_method", "read_method"],
    [["__add__", "read"], ["__add__", "read_fhd_cal"], ["fast_concat", "read"]],
)
def test_read_multi(tmp_path, concat_method, read_method):
    """Test reading in multiple files."""
    warn_type = [UserWarning] * 3
    msg = [
        "UVParameter diffuse_model does not match",
        "Telescope location derived from obs lat/lon/alt values does not match the "
        "location in the layout file.",
        "Telescope location derived from obs lat/lon/alt values does not match the "
        "location in the layout file.",
    ]

    if concat_method == "fast_concat":
        with uvtest.check_warnings(warn_type, match=msg):
            fhd_cal = UVCal.from_file(
                cal_file_multi,
                axis="time",
                obs_file=obs_file_multi,
                settings_file=settings_file_multi,
                layout_file=layout_file_multi,
                use_future_array_shapes=True,
            )
    else:
        if read_method == "read_fhd_cal":
            warn_type += [DeprecationWarning]
            msg += [
                "Reading multiple files from file specific read methods is deprecated. "
                "Use the generic `UVCal.read` method instead."
            ]
        fhd_cal = UVCal()
        with uvtest.check_warnings(warn_type, match=msg):
            getattr(fhd_cal, read_method)(
                cal_file_multi,
                obs_file=obs_file_multi,
                settings_file=settings_file_multi,
                layout_file=layout_file_multi,
                use_future_array_shapes=True,
            )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal = UVCal.from_file(outfile, use_future_array_shapes=True)
    assert fhd_cal == calfits_cal


@pytest.mark.parametrize(
    "cal_file,obs_file,layout_file,settings_file,message",
    [
        [
            cal_file_multi[0],
            obs_file_multi,
            layout_file_multi,
            settings_file_multi,
            "Number of obs_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi[0],
            layout_file_multi,
            settings_file_multi,
            "Number of obs_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi[0],
            settings_file_multi,
            "Number of layout_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi,
            settings_file_multi[0],
            "Number of settings_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi + obs_file_multi,
            layout_file_multi,
            settings_file_multi,
            "Number of obs_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi + layout_file_multi,
            settings_file_multi,
            "Number of layout_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            obs_file_multi,
            layout_file_multi,
            settings_file_multi + settings_file_multi,
            "Number of settings_files must match number of cal_files",
        ],
        [
            cal_file_multi[0],
            obs_file_multi[0],
            layout_file_multi,
            settings_file_multi,
            "Number of layout_files must match number of cal_files",
        ],
        [
            cal_file_multi[0],
            obs_file_multi[0],
            layout_file_multi[0],
            settings_file_multi,
            "Number of settings_files must match number of cal_files",
        ],
        [
            cal_file_multi,
            None,
            layout_file_multi,
            settings_file_multi,
            "obs_file parameter must be set for FHD files.",
        ],
    ],
)
@pytest.mark.parametrize("read_method", ["read", "read_fhd_cal"])
def test_break_read_multi(
    cal_file, obs_file, layout_file, settings_file, message, read_method
):
    """Test errors for different numbers of files."""
    cal = UVCal()
    if read_method == "read_fhd_cal":
        warn_type = DeprecationWarning
        msg = [
            "Reading multiple files from file specific read methods is deprecated. "
            "Use the generic `UVCal.read` method instead."
        ]
    else:
        warn_type = None
        msg = ""
    if obs_file is None and read_method == "read_fhd_cal":
        message = "Number of obs_files must match number of cal_files"
    with pytest.raises(ValueError, match=message):
        with uvtest.check_warnings(warn_type, match=msg):
            getattr(cal, read_method)(
                cal_file,
                obs_file=obs_file,
                layout_file=layout_file,
                settings_file=settings_file,
                use_future_array_shapes=True,
            )
