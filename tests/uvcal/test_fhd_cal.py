# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD_cal object."""

import os

import numpy as np
import pytest

from pyuvdata import UVCal
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

# set up FHD files
testdir = os.path.join(DATA_PATH, "fhd_cal_data/")
testfile_prefix = "1061316296_"
obs_testfile = os.path.join(testdir, "metadata", testfile_prefix + "obs.sav")
cal_testfile = os.path.join(testdir, "calibration", testfile_prefix + "cal.sav")
settings_testfile = os.path.join(testdir, "metadata", testfile_prefix + "settings.txt")
settings_testfile_nodiffuse = os.path.join(
    testdir, testfile_prefix + "nodiffuse_settings.txt"
)
layout_testfile = os.path.join(testdir, "metadata", testfile_prefix + "layout.sav")

testdir2 = os.path.join(DATA_PATH, "fhd_cal_data/set2")
obs_file_multi = [obs_testfile, os.path.join(testdir2, testfile_prefix + "obs.sav")]
cal_file_multi = [cal_testfile, os.path.join(testdir2, testfile_prefix + "cal.sav")]
layout_file_multi = [layout_testfile, layout_testfile]
settings_file_multi = [
    settings_testfile,
    os.path.join(testdir2, testfile_prefix + "settings.txt"),
]


@pytest.mark.filterwarnings("ignore:The calfits format does not support")
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("file_type", ["calfits", "calh5"])
def test_read_fhdcal_write_read_calfits_h5(
    raw, fhd_cal_raw, fhd_cal_fit, tmp_path, file_type
):
    """
    FHD cal to calfits/calh5 loopback test.

    Read in FHD cal files, write out as calfits, read back in and check for
    object equality.
    """
    if raw:
        fhd_cal = fhd_cal_raw
    else:
        fhd_cal = fhd_cal_fit

    filelist = [cal_testfile, obs_testfile, layout_testfile, settings_testfile]
    assert set(fhd_cal.filename) == {os.path.basename(fn) for fn in filelist}

    assert np.max(fhd_cal.gain_array) < 2.0

    outfile = str(tmp_path / ("outtest_FHDcal_1061311664." + file_type))
    write_method = "write_" + file_type
    getattr(fhd_cal, write_method)(outfile)

    cal_out = UVCal.from_file(outfile)
    if file_type == "calfits":
        # the phase center catalog does not round trip through calfits files
        cal_out.phase_center_catalog = fhd_cal.phase_center_catalog
        cal_out.phase_center_id_array = fhd_cal.phase_center_id_array
        cal_out.Nphase = fhd_cal.Nphase
    assert fhd_cal == cal_out


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

    fhd_cal = UVCal.from_file(
        cal_testfile,
        obs_file=obs_testfile,
        layout_file=layout_testfile,
        settings_file=settings_testfile,
        raw=raw,
        read_data=False,
    )

    fhd_cal2 = fhd_cal_full.copy(metadata_only=True)

    # this file set has a mismatch in Nsources between the cal file & settings
    # file for some reason. I think it's just an issue with the files chosen
    assert fhd_cal.Nsources != fhd_cal2.Nsources
    fhd_cal.Nsources = fhd_cal2.Nsources

    # there is a loss in precision for float auto scale values in the
    # settings file vs the cal file
    # first check that they are similar (extract from the string they are packed in)
    # only test to single precision because that's how it's stored.
    np.testing.assert_allclose(
        np.asarray(fhd_cal.extra_keywords["AUTOSCAL"][1:-1].split(", "), dtype=float),
        np.asarray(fhd_cal2.extra_keywords["AUTOSCAL"][1:-1].split(", "), dtype=float),
        atol=0,
        rtol=2e-6,
    )
    # replace the strings to prevent errors
    fhd_cal.extra_keywords["autoscal".upper()] = fhd_cal2.extra_keywords[
        "autoscal".upper()
    ]

    assert fhd_cal == fhd_cal2

    # test that no diffuse is properly picked up from the settings file when
    # read_data is False
    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['settings']",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
        ],
    ):
        fhd_cal = UVCal.from_file(
            cal_testfile,
            obs_file=obs_testfile,
            layout_file=layout_testfile,
            settings_file=settings_testfile_nodiffuse,
            raw=raw,
            read_data=False,
        )

    assert fhd_cal.diffuse_model is None

    return


def test_read_fhdcal_multimode():
    """
    Read cal with multiple mode_fit values.
    """
    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: "
            "['cal', 'settings']",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
        ],
    ):
        fhd_cal = UVCal.from_file(
            os.path.join(testdir, testfile_prefix + "multimode_cal.sav"),
            obs_file=obs_testfile,
            layout_file=layout_testfile,
            settings_file=os.path.join(
                testdir, testfile_prefix + "multimode_settings.txt"
            ),
            raw=False,
        )
    assert fhd_cal.extra_keywords["MODE_FIT"] == "[90, 150, 230, 320, 400, 524]"

    fhd_cal2 = fhd_cal.copy(metadata_only=True)

    # check metadata only read
    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: "
            "['cal', 'settings']",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
        ],
    ):
        fhd_cal = UVCal.from_file(
            os.path.join(testdir, testfile_prefix + "multimode_cal.sav"),
            obs_file=obs_testfile,
            layout_file=layout_testfile,
            settings_file=os.path.join(
                testdir, testfile_prefix + "multimode_settings.txt"
            ),
            raw=False,
            read_data=False,
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


@pytest.mark.filterwarnings("ignore:The calfits format does not support")
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
    )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal = UVCal.from_file(outfile)
    # the phase center catalog does not round trip through calfits files
    assert fhd_cal != calfits_cal
    calfits_cal.phase_center_catalog = fhd_cal.phase_center_catalog
    calfits_cal.phase_center_id_array = fhd_cal.phase_center_id_array
    calfits_cal.Nphase = fhd_cal.Nphase
    assert fhd_cal == calfits_cal
    for line in extra_history:
        assert line in fhd_cal.history

    return


@pytest.mark.filterwarnings("ignore:The calfits format does not support")
@pytest.mark.filterwarnings("ignore:Telescope location derived from obs lat/lon/alt")
def test_flags_galaxy(tmp_path):
    """Test files with time, freq and tile flags and galaxy models behave."""
    testdir = os.path.join(DATA_PATH, "fhd_cal_data/flag_set")
    obs_testfile_flag = os.path.join(testdir, testfile_prefix + "obs.sav")
    cal_testfile_flag = os.path.join(testdir, testfile_prefix + "cal.sav")
    settings_testfile_flag = os.path.join(testdir, testfile_prefix + "settings.txt")
    layout_testfile_flag = os.path.join(testdir, testfile_prefix + "layout.sav")

    with check_warnings(
        UserWarning,
        match=[
            "tile_names from obs structure does not match",
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['cal', "
            "'layout', 'obs', 'settings']",
        ],
    ):
        fhd_cal = UVCal.from_file(
            cal_testfile_flag,
            obs_file=obs_testfile_flag,
            layout_file=layout_testfile_flag,
            settings_file=settings_testfile_flag,
        )

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    fhd_cal.write_calfits(outfile, clobber=True)
    calfits_cal = UVCal.from_file(outfile)

    # the phase center catalog does not round trip through calfits files
    calfits_cal.phase_center_catalog = fhd_cal.phase_center_catalog
    calfits_cal.phase_center_id_array = fhd_cal.phase_center_id_array
    calfits_cal.Nphase = fhd_cal.Nphase
    assert fhd_cal == calfits_cal


def test_unknown_telescope():
    with check_warnings(
        UserWarning,
        match=[
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['obs']",
            "The FHD input files do not all have matching prefixes, so they may not be "
            "for the same data.",
        ],
    ):
        fhd_cal = UVCal.from_file(
            cal_testfile,
            obs_file=os.path.join(testdir, testfile_prefix + "telescopefoo_obs.sav"),
            layout_file=layout_testfile,
            settings_file=settings_testfile,
            default_mount_type="fixed",
        )
    assert fhd_cal.telescope.name == "foo"


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
            cal_file, obs_file=obs_file, layout_file=layout_file, read_data=False
        )

    message_list = ["No settings file"]
    if nfiles > 1:
        message_list *= 2
        message_list.append("UVParameter diffuse_model does not match")
        message_list.append(
            "Some FHD input files do not have the expected subfolder so FHD folder "
            "matching could not be done. The affected file types are: ['cal', 'obs']"
        )

    with check_warnings(UserWarning, message_list):
        fhd_cal = UVCal.from_file(cal_file, obs_file=obs_file, layout_file=layout_file)

    # Check only pyuvdata version history with no settings file
    expected_history = "\n" + fhd_cal.pyuvdata_version_str
    if nfiles > 1:
        expected_history += " Combined data along time axis using pyuvdata."
    assert fhd_cal.history == expected_history

    message_list = [
        "No layout file, antenna_postions will not be defined."
    ] * nfiles + ["UVParameter diffuse_model does not match"] * (nfiles - 1)

    if nfiles == 1:
        with check_warnings(UserWarning, match=message_list):
            fhd_cal.read_fhd_cal(
                cal_file=cal_file, obs_file=obs_file, settings_file=settings_file
            )

        with pytest.raises(
            ValueError, match="A settings_file must be provided if read_data is False."
        ):
            fhd_cal.read_fhd_cal(
                cal_file=cal_file,
                obs_file=obs_file,
                layout_file=layout_file,
                read_data=False,
            )


@pytest.mark.filterwarnings("ignore:The calfits format does not support")
@pytest.mark.parametrize(
    ["concat_method", "read_method"], [["__add__", "read"], ["fast_concat", "read"]]
)
def test_read_multi(tmp_path, concat_method, read_method):
    """Test reading in multiple files."""
    warn_type = [UserWarning] * 2
    msg = [
        "UVParameter diffuse_model does not match",
        "Some FHD input files do not have the expected subfolder so FHD folder "
        "matching could not be done. The affected file types are: "
        "['cal', 'obs', 'settings']",
    ]

    if concat_method == "fast_concat":
        with check_warnings(warn_type, match=msg):
            fhd_cal = UVCal.from_file(
                cal_file_multi,
                axis="time",
                obs_file=obs_file_multi,
                settings_file=settings_file_multi,
                layout_file=layout_file_multi,
            )
    else:
        if read_method == "read_fhd_cal":
            warn_type += [DeprecationWarning]
            msg += [
                "Reading multiple files from file specific read methods is deprecated. "
                "Use the generic `UVCal.read` method instead."
            ]
        fhd_cal = UVCal()
        with check_warnings(warn_type, match=msg):
            getattr(fhd_cal, read_method)(
                cal_file_multi,
                obs_file=obs_file_multi,
                settings_file=settings_file_multi,
                layout_file=layout_file_multi,
            )

    calfits_outfile = str(tmp_path / "outtest_FHDcal_1061311664.calfits")
    with pytest.raises(ValueError, match="Object contains multiple time ranges."):
        fhd_cal.write_calfits(calfits_outfile, clobber=True)

    outfile = str(tmp_path / "outtest_FHDcal_1061311664.calh5")
    fhd_cal.write_calh5(outfile, clobber=True)

    calh5_cal = UVCal.from_file(outfile)
    assert fhd_cal == calh5_cal


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
def test_break_read_multi(cal_file, obs_file, layout_file, settings_file, message):
    """Test errors for different numbers of files."""
    cal = UVCal()
    with pytest.raises(ValueError, match=message):
        cal.read(
            cal_file,
            obs_file=obs_file,
            layout_file=layout_file,
            settings_file=settings_file,
        )
