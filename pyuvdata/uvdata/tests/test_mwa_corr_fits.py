# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MWACorrFITS object."""

import itertools
import os

import h5py
import numpy as np
import pytest
from astropy.io import fits

import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.uvdata.mwa_corr_fits import input_output_mapping
from pyuvdata.uvdata.uvdata import _future_array_shapes_warning

# set up MWA correlator file list
testdir = os.path.join(DATA_PATH, "mwa_corr_fits_testfiles/")

testfiles = [
    "1131733552.metafits",
    "1131733552_20151116182537_mini_gpubox01_00.fits",
    "1131733552_20151116182637_mini_gpubox06_01.fits",
    "1131733552_mini_01.mwaf",
    "1131733552_mini_06.mwaf",
    "1131733552_mod.metafits",
    "1131733552_mini_cotter.uvfits",
    "1131733552_metafits_ppds.fits",
    "1061315448_20130823175130_mini_gpubox07_01.fits",
    "1061315448.metafits",
    "1061315448_20130823175130_mini_vv_07_01.uvh5",
    "1320409688.metafits",
    "1320409688_20211108122750_mini_ch137_000.fits",
]
filelist = [os.path.join(testdir, filei) for filei in testfiles]


@pytest.fixture(scope="module")
def flag_file_init(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pyuvdata_corr_fits", numbered=True)
    spoof_file1 = str(tmp_path / "spoof_01_00.fits")
    spoof_file6 = str(tmp_path / "spoof_06_00.fits")
    # spoof box files of the appropriate size
    with fits.open(filelist[1]) as mini1:
        mini1[1].data = np.repeat(mini1[1].data, 8, axis=0)
        extra_dat = np.copy(mini1[1].data)
        for _ in range(2):
            mini1.append(fits.ImageHDU(extra_dat))
        mini1[2].header["MILLITIM"] = 500
        mini1[2].header["TIME"] = mini1[1].header["TIME"]
        mini1[3].header["MILLITIM"] = 0
        mini1[3].header["TIME"] = mini1[1].header["TIME"] + 1
        mini1.writeto(spoof_file1)

    with fits.open(filelist[2]) as mini6:
        mini6[1].data = np.repeat(mini6[1].data, 8, axis=0)
        extra_dat = np.copy(mini6[1].data)
        for _ in range(2):
            mini6.append(fits.ImageHDU(extra_dat))
        mini6[2].header["MILLITIM"] = 500
        mini6[2].header["TIME"] = mini6[1].header["TIME"]
        mini6[3].header["MILLITIM"] = 0
        mini6[3].header["TIME"] = mini6[1].header["TIME"] + 1
        mini6.writeto(spoof_file6)

    flag_testfiles = [spoof_file1, spoof_file6, filelist[0]]

    yield flag_testfiles


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
def test_read_mwa_write_uvfits(tmp_path):
    """
    MWA correlator fits to uvfits loopback test.

    Read in MWA correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    mwa_uv = UVData()
    uvfits_uv = UVData()
    with uvtest.check_warnings(
        [UserWarning, DeprecationWarning],
        match=[
            "some coarse channel files were not submitted",
            _future_array_shapes_warning,
        ],
    ):
        mwa_uv.read(
            filelist[0:2], correct_cable_len=True, phase_to_pointing_center=True
        )

    testfile = str(tmp_path / "outtest_MWAcorr.uvfits")
    mwa_uv.write_uvfits(testfile)
    uvfits_uv.read_uvfits(testfile)

    # make sure filenames are what we expect
    assert set(mwa_uv.filename) == {
        "1131733552.metafits",
        "1131733552_20151116182537_mini_gpubox01_00.fits",
    }
    assert uvfits_uv.filename == ["outtest_MWAcorr.uvfits"]
    mwa_uv.filename = uvfits_uv.filename
    mwa_uv._filename.form = (1,)

    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific paramters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(mwa_uv, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=mwa_uv.phase_center_catalog
    )
    assert mwa_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:antnums_to_baseline")
@pytest.mark.filterwarnings("ignore:Found antenna numbers > 255 in this data")
def test_read_mwax_write_uvfits(tmp_path):
    """
    MWAX correlator fits to uvfits loopback test.

    Read in MWAX correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    # spoof testfile to contain 2 times and 2 freqs
    spoof_file = str(tmp_path / "mwax_spoof_ch137_000.fits")
    with fits.open(filelist[12]) as mini1:
        mini1[1].data = np.repeat(mini1[1].data, 2, axis=1)
        mini1[2].data = np.repeat(mini1[2].data, 2, axis=1)
        extra_dat = np.copy(mini1[1].data)
        extra_samps = np.copy(mini1[2].data)
        mini1.append(fits.ImageHDU(extra_dat))
        mini1.append(fits.ImageHDU(extra_samps))
        mini1[3].header["TIME"] = 1636374472
        mini1[4].header["TIME"] = 1636374472
        mini1[3].header["MILLITIM"] = 0
        mini1[4].header["MILLITIM"] = 0
        mini1.writeto(spoof_file)
    mwax_uv = UVData()
    uvfits_uv = UVData()
    messages = [
        "Fixing auto-correlations to be be real-only, after some imaginary values",
        "some coarse channel files were not submitted",
    ]
    with uvtest.check_warnings(UserWarning, messages):
        mwax_uv.read(
            [spoof_file, filelist[11]],
            correct_cable_len=True,
            phase_to_pointing_center=True,
            use_future_array_shapes=True,
        )
    testfile = str(tmp_path / "outtest_MWAXcorr.uvfits")
    mwax_uv.write_uvfits(testfile)
    uvfits_uv.read_uvfits(testfile, use_future_array_shapes=True)

    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific paramters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(mwax_uv, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=mwax_uv.phase_center_catalog, ignore_name=True
    )
    assert mwax_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_mwax_metafits_keys(tmp_path):
    """Check that mwax keywords are removed from extra_keywords for legacy files"""
    meta_spoof_file = str(tmp_path / "spoof_1131733552.metafits")
    with fits.open(filelist[0]) as meta:
        meta[0].header["DELAYMOD"] = None
        meta.writeto(meta_spoof_file)
    uv = UVData()
    uv.read([meta_spoof_file, filelist[1]], use_future_array_shapes=True)

    assert "DELAYMOD" not in uv.extra_keywords.keys()


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_select_on_read():
    mwa_uv = UVData()
    mwa_uv2 = UVData()
    mwa_uv.read(filelist[0:2], correct_cable_len=True, use_future_array_shapes=True)
    unique_times = np.unique(mwa_uv.time_array)
    select_times = unique_times[
        np.where(
            (unique_times >= np.min(mwa_uv.time_array))
            & (unique_times <= np.mean(mwa_uv.time_array))
        )
    ]
    mwa_uv.select(times=select_times)
    with uvtest.check_warnings(
        UserWarning,
        [
            'Warning: select on read keyword set, but file_type is "mwa_corr_fits"',
            "some coarse channel files were not submitted",
        ],
    ):
        mwa_uv2.read(
            filelist[0:2],
            correct_cable_len=True,
            time_range=[np.min(mwa_uv.time_array), np.mean(mwa_uv.time_array)],
            use_future_array_shapes=True,
        )
    assert mwa_uv == mwa_uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_read_mwa_read_cotter():
    """
    Pyuvdata and cotter equality test.

    Read in MWA correlator files and the corresponding cotter file and check
    for data array equality.
    """
    mwa_uv = UVData()
    cotter_uv = UVData()
    # cotter data has cable correction and is unphased
    mwa_uv.read(
        filelist[0:2],
        correct_cable_len=True,
        remove_dig_gains=False,
        remove_coarse_band=False,
        remove_flagged_ants=False,
        use_future_array_shapes=True,
    )
    cotter_uv.read(filelist[6], use_future_array_shapes=True)
    # cotter doesn't record the auto xy polarizations
    # due to a possible bug in cotter, the auto yx polarizations are conjugated
    # fix these before testing data_array
    autos = np.isclose(mwa_uv.ant_1_array - mwa_uv.ant_2_array, 0.0)
    cotter_uv.data_array[autos, :, 2] = cotter_uv.data_array[autos, :, 3]
    cotter_uv.data_array[autos, :, 3] = np.conj(cotter_uv.data_array[autos, :, 3])
    assert np.allclose(mwa_uv.data_array, cotter_uv.data_array, atol=1e-4, rtol=0)
    assert mwa_uv.freq_array == cotter_uv.freq_array


def test_read_mwa_write_uvfits_meta_mod(tmp_path):
    """
    MWA correlator fits to uvfits loopback test with a modified metafits file.

    Read in MWA correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    # The metafits file has been modified to contain some coarse channels < 129,
    # and to have an uncorrected cable length.
    mwa_uv = UVData()
    uvfits_uv = UVData()
    messages = ["some coarse channel files were not submitted"]
    files = [filelist[1], filelist[5]]
    with uvtest.check_warnings(UserWarning, messages):
        mwa_uv.read(
            files,
            correct_cable_len=True,
            phase_to_pointing_center=True,
            use_future_array_shapes=True,
        )
    testfile = str(tmp_path / "outtest_MWAcorr.uvfits")
    mwa_uv.write_uvfits(testfile)
    uvfits_uv.read_uvfits(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert set(mwa_uv.filename) == {
        "1131733552_20151116182537_mini_gpubox01_00.fits",
        "1131733552_mod.metafits",
    }
    assert uvfits_uv.filename == ["outtest_MWAcorr.uvfits"]
    mwa_uv.filename = uvfits_uv.filename
    mwa_uv._filename.form = (1,)

    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific paramters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(mwa_uv, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=mwa_uv.phase_center_catalog
    )
    assert mwa_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Combined frequencies are separated by more than")
def test_read_mwa_multi():
    """Test reading in two sets of files."""
    set1 = filelist[0:2]
    set2 = [filelist[0], filelist[2]]
    mwa_uv = UVData()

    mwa_uv.read([set1, set2], use_future_array_shapes=True)

    mwa_uv2 = UVData()
    messages = [
        "some coarse channel files were not submitted",
        "some coarse channel files were not submitted",
        "Combined frequencies are separated by more than their channel width",
    ]
    with uvtest.check_warnings(UserWarning, messages):
        mwa_uv2.read(
            [set1, set2], file_type="mwa_corr_fits", use_future_array_shapes=True
        )

    assert mwa_uv == mwa_uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Combined frequencies are separated by more than")
def test_read_mwa_multi_concat(tmp_path):
    """Test reading in two sets of files with fast concatenation."""
    # modify file so that time arrays are matching
    mod_mini_6 = str(tmp_path / "mini_gpubox06_01.fits")
    with fits.open(filelist[2]) as mini6:
        mini6[1].header["time"] = 1447698337
        mini6.writeto(mod_mini_6)
    set1 = filelist[0:2]
    set2 = [filelist[0], mod_mini_6]
    mwa_uv = UVData()
    mwa_uv.read([set1, set2], axis="freq", use_future_array_shapes=True)

    mwa_uv2 = UVData()
    messages = [
        "some coarse channel files were not submitted",
        "some coarse channel files were not submitted",
        "Combined frequencies are separated by more than their channel width",
    ]
    with uvtest.check_warnings(UserWarning, messages):
        mwa_uv2.read(
            [set1, set2],
            axis="freq",
            file_type="mwa_corr_fits",
            use_future_array_shapes=True,
        )

    assert mwa_uv == mwa_uv2
    os.remove(mod_mini_6)


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_read_mwa_flags():
    """Test handling of flag files."""
    mwa_uv = UVData()
    subfiles = [filelist[0], filelist[1], filelist[3], filelist[4]]
    messages = [
        "mwaf files submitted with use_aoflagger_flags=False",
        "some coarse channel files were not submitted",
    ]
    with uvtest.check_warnings(UserWarning, messages):
        mwa_uv.read(subfiles, use_aoflagger_flags=False, use_future_array_shapes=True)

    del mwa_uv

    mwa_uv = UVData()
    with pytest.raises(ValueError, match="no flag files submitted"):
        mwa_uv.read(
            subfiles[0:2], use_aoflagger_flags=True, use_future_array_shapes=True
        )
    del mwa_uv


def test_multiple_coarse():
    """
    Test two coarse channel files.

    Read in MWA correlator files with two different orderings of the files
    and check for object equality.
    """
    order1 = filelist[0:3]
    order2 = [filelist[0], filelist[2], filelist[1]]
    mwa_uv1 = UVData()
    mwa_uv2 = UVData()
    messages = [
        "coarse channels are not contiguous for this observation",
        "some coarse channel files were not submitted",
    ]
    with uvtest.check_warnings(UserWarning, messages):
        mwa_uv1.read(order1, use_future_array_shapes=True)
    with uvtest.check_warnings(UserWarning, messages):
        mwa_uv2.read(order2, use_future_array_shapes=True)

    assert mwa_uv1 == mwa_uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_ppds(tmp_path):
    """Test handling of ppds files"""
    # turnaround test with just ppds file given
    mwa_uv = UVData()
    mwa_uv.read(
        [filelist[1], filelist[7]],
        phase_to_pointing_center=True,
        flag_init=False,
        use_future_array_shapes=True,
    )
    testfile = str(tmp_path / "outtest_MWAcorr.uvfits")
    mwa_uv.write_uvfits(testfile)
    uvfits_uv = UVData()
    uvfits_uv.read_uvfits(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert set(mwa_uv.filename) == {
        "1131733552_20151116182537_mini_gpubox01_00.fits",
        "1131733552_metafits_ppds.fits",
    }
    assert uvfits_uv.filename == ["outtest_MWAcorr.uvfits"]
    mwa_uv.filename = uvfits_uv.filename
    mwa_uv._filename.form = (1,)

    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific paramters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(mwa_uv, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=mwa_uv.phase_center_catalog
    )
    assert mwa_uv == uvfits_uv

    del mwa_uv
    del uvfits_uv

    # check that extra keywords are added when both ppds file and metafits file given
    mwa_uv = UVData()
    mwa_uv.read([filelist[0], filelist[1], filelist[7]], use_future_array_shapes=True)
    assert "MWAVER" in mwa_uv.extra_keywords and "MWADATE" in mwa_uv.extra_keywords


def test_fine_channels(tmp_path):
    """
    Break read_mwa_corr_fits by submitting files with different fine channels.

    Test that error is raised if files with different numbers of fine channels
    are submitted.
    """
    mwa_uv = UVData()
    bad_fine = str(tmp_path / "bad_gpubox06_01.fits")
    with fits.open(filelist[2]) as mini6:
        mini6[1].data = np.concatenate((mini6[1].data, mini6[1].data))
        mini6.writeto(bad_fine)
    with pytest.raises(ValueError, match="files submitted have different numbers"):
        mwa_uv.read([bad_fine, filelist[1]])
    del mwa_uv


def test_fine_channels_mwax(tmp_path):
    """
    Break read_mwa_corr_fits by submitting mwax files with different fine channels.

    Test that error is raised if files with different numbers of fine channels
    are submitted.
    """
    mwax_uv = UVData()
    bad_fine = str(tmp_path / "bad_ch137_000.fits.fits")
    with fits.open(filelist[12]) as mini:
        mini[1].data = np.repeat(mini[1].data, 2, axis=1)
        mini[2].data = np.repeat(mini[2].data, 2, axis=1)
        mini.writeto(bad_fine)
    with pytest.raises(ValueError, match="files submitted have different numbers"):
        mwax_uv.read([bad_fine, filelist[12]])
    del mwax_uv


@pytest.mark.parametrize(
    "files,err_msg",
    [
        ([filelist[0]], "no data files submitted"),
        ([filelist[1]], "no metafits file submitted"),
        (
            [filelist[0], filelist[1], filelist[5]],
            "multiple metafits files in filelist",
        ),
    ],
)
def test_break_read_mwacorrfits(files, err_msg):
    """Break read_mwa_corr_fits by submitting files incorrectly."""
    mwa_uv = UVData()
    with pytest.raises(ValueError, match=err_msg):
        mwa_uv.read(files)
    del mwa_uv


def test_file_extension(tmp_path):
    """
    Break read_mwa_corr_fits by submitting file with the wrong extension.

    Test that error is raised if a file with an extension that is not fits,
    metafits, or mwaf is submitted.
    """
    mwa_uv = UVData()
    bad_ext = str(tmp_path / "1131733552.meta")
    with fits.open(filelist[0]) as meta:
        meta.writeto(bad_ext)
    with pytest.raises(
        ValueError, match="only fits, metafits, and mwaf files supported"
    ):
        mwa_uv.read(bad_ext, file_type="mwa_corr_fits")
    del mwa_uv


def test_diff_obs(tmp_path):
    """
    Break read_mwa_corr_fits by submitting files from different observations.

    Test that error is raised if files from different observations are
    submitted in the same file list.
    """
    mwa_uv = UVData()
    bad_obs = str(tmp_path / "bad2_gpubox06_01.fits")
    with fits.open(filelist[2]) as mini6:
        mini6[0].header["OBSID"] = "1131733555"
        mini6.writeto(bad_obs)
    with pytest.raises(ValueError, match="files from different observations"):
        mwa_uv.read([bad_obs, filelist[0], filelist[1]])
    del mwa_uv


def test_misaligned_times(tmp_path):
    """
    Break read_mwa_corr_fits by submitting files with misaligned times.

    Test that error is raised if file start times are different by an amount
    that is not an integer multiiple of the integration time.
    """
    mwa_uv = UVData()
    bad_obs = str(tmp_path / "bad3_gpubox06_01.fits")
    with fits.open(filelist[2]) as mini6:
        mini6[1].header["MILLITIM"] = 250
        mini6.writeto(bad_obs)
    with pytest.raises(ValueError, match="coarse channel start times are misaligned"):
        mwa_uv.read([bad_obs, filelist[0], filelist[1]])
    del mwa_uv


@pytest.mark.filterwarnings(
    "ignore:coarse channels are not contiguous for this observation"
)
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_flag_nsample_basic():
    """
    Test that the flag(without flag_int) and nsample arrays correctly reflect data.
    """
    uv = UVData()
    uv.read(
        filelist[0:3],
        flag_init=False,
        propagate_coarse_flags=False,
        remove_flagged_ants=False,
        use_future_array_shapes=True,
    )
    # check that only bad antennas are flagged for all times, freqs, pols
    bad_ants = [59, 114]
    good_ants = np.delete(np.unique(uv.ant_1_array), bad_ants)
    bad = uv.select(antenna_nums=np.unique(uv.ant_1_array)[bad_ants], inplace=False)
    good = uv.select(antenna_nums=good_ants, inplace=False)
    assert np.all(bad.flag_array)
    good.flag_array = good.flag_array.reshape(
        (good.Ntimes, good.Nbls, good.Nfreqs, good.Npols)
    )
    # good ants should be flagged except for the first time and second freq,
    # and for the second time and first freq
    assert np.all(good.flag_array[1:-1, :, :, :])
    assert np.all(good.flag_array[0, :, 1, :] == 0)
    assert np.all(good.flag_array[-1, :, 0, :] == 0)
    assert np.all(good.flag_array[0, :, 0, :])
    assert np.all(good.flag_array[-1, :, 1, :])
    # check that nsample array is filled properly
    uv.nsample_array = uv.nsample_array.reshape(
        (uv.Ntimes, uv.Nbls, uv.Nfreqs, uv.Npols)
    )
    assert np.all(uv.nsample_array[1:-1, :, :, :] == 0.0)
    assert np.all(uv.nsample_array[0, :, 1, :] == 1.0)
    assert np.all(uv.nsample_array[-1, :, 0, :] == 1.0)
    assert np.all(uv.nsample_array[0, :, 0, :] == 0.0)
    assert np.all(uv.nsample_array[-1, :, 1, :] == 0.0)


@pytest.mark.filterwarnings(
    "ignore:coarse channels are not contiguous for this observation"
)
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_flag_init(flag_file_init):
    """
    Test that routine MWA flagging works as intended.
    """
    uv = UVData()
    uv.read(
        flag_file_init,
        flag_init=True,
        start_flag=0,
        end_flag=0,
        propagate_coarse_flags=False,
        use_future_array_shapes=True,
    )

    freq_inds = [0, 1, 4, 6, 7, 8, 9, 12, 14, 15]
    freq_inds_complement = [ind for ind in range(16) if ind not in freq_inds]

    assert np.all(
        uv.flag_array[:, freq_inds, :]
    ), "Not all of edge and center channels are flagged!"
    assert not np.any(
        np.all(uv.flag_array[:, freq_inds_complement, :], axis=(0, 1, -1))
    ), "Some non-edge/center channels are entirely flagged!"


@pytest.mark.filterwarnings(
    "ignore:coarse channels are not contiguous for this observation"
)
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_flag_start_flag(flag_file_init):
    uv = UVData()
    uv.read(
        flag_file_init,
        flag_init=True,
        start_flag=1.0,
        end_flag=1.0,
        edge_width=0,
        flag_dc_offset=False,
        propagate_coarse_flags=False,
        use_future_array_shapes=True,
    )

    reshape = [uv.Ntimes, uv.Nbls, uv.Nfreqs, uv.Npols]
    time_inds = [0, 1, -1, -2]
    assert np.all(
        uv.flag_array.reshape(reshape)[time_inds, :, :, :]
    ), "Not all of start and end times are flagged."
    # Check that it didn't just flag everything
    # Should have unflagged data for time inds [2, -3]
    assert not np.any(
        np.all(uv.flag_array.reshape(reshape)[[2, -3], :, :, :], axis=(1, 2, 3))
    ), "All the data is flagged for some intermediate times!"


@pytest.mark.parametrize(
    "err_type,read_kwargs,err_msg",
    [
        (
            ValueError,
            {"flag_init": True, "start_flag": 0, "end_flag": 0, "edge_width": 90e3},
            "The edge_width must be an integer multiple of the channel_width",
        ),
        (
            ValueError,
            {"flag_init": True, "start_flag": 1.2, "end_flag": 0},
            "The start_flag must be an integer multiple of the integration_time",
        ),
        (
            ValueError,
            {"flag_init": True, "start_flag": 0, "end_flag": 1.2},
            "The end_flag must be an integer multiple of the integration_time",
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:coarse channels are not contiguous for this observation"
)
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_flag_init_errors(flag_file_init, err_type, read_kwargs, err_msg):
    uv = UVData()
    # give noninteger multiple inputs
    with pytest.raises(err_type, match=err_msg):
        uv.read(flag_file_init, **read_kwargs)


def test_read_metadata_only(tmp_path):
    """Test reading an MWA corr fits file as metadata only."""
    uvd = UVData()
    messages = ["some coarse channel files were not submitted"]
    with uvtest.check_warnings(UserWarning, messages):
        uvd.read(
            filelist[0:2],
            correct_cable_len=True,
            phase_to_pointing_center=True,
            read_data=False,
            use_future_array_shapes=True,
        )

    assert uvd.metadata_only


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_data_array_precision():
    uv = UVData()
    uv2 = UVData()
    # read in data array as single precision
    uv.read(filelist[0:2], data_array_dtype=np.complex64, use_future_array_shapes=True)
    # now read as double precision
    uv2.read(
        filelist[0:2], data_array_dtype=np.complex128, use_future_array_shapes=True
    )

    assert uv == uv2
    assert uv.data_array.dtype.type is np.complex64
    assert uv2.data_array.dtype.type is np.complex128

    return


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_nsample_array_precision():
    uv = UVData()
    uv2 = UVData()
    uv3 = UVData()
    # read in nsample array at different precisions
    uv.read(filelist[0:2], nsample_array_dtype=np.float32, use_future_array_shapes=True)
    uv2.read(
        filelist[0:2], nsample_array_dtype=np.float64, use_future_array_shapes=True
    )
    uv3.read(
        filelist[0:2], nsample_array_dtype=np.float16, use_future_array_shapes=True
    )

    assert uv == uv2
    assert uv == uv3
    assert uv.nsample_array.dtype.type is np.float32
    assert uv2.nsample_array.dtype.type is np.float64
    assert uv3.nsample_array.dtype.type is np.float16

    return


def test_invalid_precision_errors():
    uv = UVData()

    # raise errors by passing bogus precision values
    with pytest.raises(ValueError, match="data_array_dtype must be np.complex64"):
        uv.read(filelist[0:2], data_array_dtype=np.float64)

    with pytest.raises(
        ValueError, match="nsample_array_dtype must be one of: np.float64"
    ):
        uv.read(filelist[0:2], nsample_array_dtype=np.complex128)

    return


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_remove_dig_gains():
    """Test digital gain removal."""
    uv1 = UVData()
    uv1.read(filelist[0:2], data_array_dtype=np.complex64, use_future_array_shapes=True)

    uv2 = UVData()
    uv2.read(filelist[0:2], remove_dig_gains=False, use_future_array_shapes=True)

    with fits.open(filelist[0]) as meta:
        meta_tbl = meta[1].data
        antenna_inds = meta_tbl["Antenna"][1::2]
        dig_gains = meta_tbl["Gains"][1::2, :].astype(np.float64) / 64
    reordered_inds = antenna_inds.argsort()
    bad_ant_inds = [59, 114]
    good_ants = np.delete(reordered_inds, bad_ant_inds)
    ant_1_inds, ant_2_inds = np.transpose(
        list(itertools.combinations_with_replacement(np.arange(uv2.Nants_data), 2))
    )
    ant_1_inds = np.tile(np.array(ant_1_inds), uv2.Ntimes).astype(np.int_)
    ant_2_inds = np.tile(np.array(ant_2_inds), uv2.Ntimes).astype(np.int_)
    dig_gains = dig_gains[good_ants, 23]
    uv2.data_array = uv2.data_array / (
        dig_gains[ant_1_inds, np.newaxis, np.newaxis]
        * dig_gains[ant_2_inds, np.newaxis, np.newaxis]
    )
    uv2.history = uv1.history

    # make sure correction doesn't change data_array type
    assert uv1.data_array.dtype == np.complex64
    assert "Divided out digital gains" in uv1.history
    assert uv1 == uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_remove_coarse_band(tmp_path):
    """Test coarse band removal."""
    # generate a spoof file with 32 channels
    cb_spoof = str(tmp_path / "cb_spoof_01_00.fits")
    with fits.open(filelist[1]) as mini1:
        mini1[1].data = np.repeat(mini1[1].data, 32, axis=0)
        mini1.writeto(cb_spoof)

    uv1 = UVData()
    uv1.read(
        [filelist[0], cb_spoof],
        data_array_dtype=np.complex64,
        use_future_array_shapes=True,
    )

    uv2 = UVData()
    uv2.read(
        [filelist[0], cb_spoof], remove_coarse_band=False, use_future_array_shapes=True
    )

    with h5py.File(
        DATA_PATH + "/mwa_config_data/MWA_rev_cb_10khz_doubles.h5", "r"
    ) as f:
        cb = f["coarse_band"][:]
    cb_array = cb.reshape(32, 4)
    cb_array = np.average(cb_array, axis=1)
    uv2.data_array /= cb_array[:, np.newaxis]

    uv2.history = uv1.history

    # make sure correction doesn't change data_array type
    assert uv1.data_array.dtype == np.complex64
    assert "Divided out pfb coarse channel bandpass" in uv1.history
    assert uv1 == uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
def test_remove_coarse_band_mwax_40(tmp_path):
    """Test coarse band removal for a 40 kHz mwax file."""
    # generate a spoof file with 32 channels
    cb_spoof = str(tmp_path / "mwax_cb_spoof40_ch137_000.fits")
    meta_spoof = str(tmp_path / "mwax_cb_spoof40.metafits")

    with fits.open(filelist[12]) as mini1:
        mini1[1].data = np.repeat(mini1[1].data, 32, axis=1)
        mini1.writeto(cb_spoof)

    with fits.open(filelist[11]) as meta:
        meta[0].header["FINECHAN"] = 80
        meta.writeto(meta_spoof)

    uv1 = UVData()
    uv1.read([filelist[11], cb_spoof], use_future_array_shapes=True)

    uv2 = UVData()
    uv2.read(
        [filelist[11], cb_spoof], remove_coarse_band=False, use_future_array_shapes=True
    )

    with h5py.File(DATA_PATH + "/mwa_config_data/mwax_pfb_bandpass_40kHz.h5", "r") as f:
        cb_array = f["coarse_band"][:]

    uv2.data_array /= cb_array[:, np.newaxis]

    uv2.history = uv1.history

    assert "Divided out pfb coarse channel bandpass" in uv1.history
    assert uv1 == uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
def test_remove_coarse_band_mwax_80(tmp_path):
    """Test coarse band removal for an 80 kHz mwax file."""
    # generate a spoof file with 16 channels
    cb_spoof = str(tmp_path / "mwax_cb_spoof80_ch137_000.fits")
    meta_spoof = str(tmp_path / "mwax_cb_spoof80.metafits")

    with fits.open(filelist[12]) as mini1:
        mini1[1].data = np.repeat(mini1[1].data, 16, axis=1)
        mini1.writeto(cb_spoof)

    with fits.open(filelist[11]) as meta:
        meta[0].header["FINECHAN"] = 80
        meta.writeto(meta_spoof)

    uv1 = UVData()
    uv1.read([meta_spoof, cb_spoof], use_future_array_shapes=True)

    uv2 = UVData()
    uv2.read(
        [meta_spoof, cb_spoof], remove_coarse_band=False, use_future_array_shapes=True
    )

    with h5py.File(DATA_PATH + "/mwa_config_data/mwax_pfb_bandpass_80kHz.h5", "r") as f:
        cb_array = f["coarse_band"][:]

    uv2.data_array /= cb_array[:, np.newaxis]

    uv2.history = uv1.history

    assert "Divided out pfb coarse channel bandpass" in uv1.history
    assert uv1 == uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
def test_remove_coarse_band_mwax_warning(tmp_path):
    """Test coarse band removal for a file we don't have a passband for."""
    # generate a spoof file with 16 channels
    cb_spoof = str(tmp_path / "mwax_cb_spoof160_ch137_000.fits")
    meta_spoof = str(tmp_path / "mwax_cb_spoof160.metafits")

    with fits.open(filelist[12]) as mini1:
        mini1[1].data = np.repeat(mini1[1].data, 8, axis=1)
        mini1.writeto(cb_spoof)

    with fits.open(filelist[11]) as meta:
        meta[0].header["FINECHAN"] = 160
        meta.writeto(meta_spoof)

    uv = UVData()
    with pytest.raises(ValueError, match="mwax passband shapes are only available"):
        uv.read([meta_spoof, cb_spoof], flag_init=False, use_future_array_shapes=True)


def test_aoflagger_flags():
    """Test using aoflagger flags"""
    uv = UVData()
    files = filelist[0:2]
    files.append(filelist[3])
    messages = [
        "some coarse channel files were not submitted",
        "coarse channel, start time, and end time flagging will default",
    ]
    with uvtest.check_warnings(UserWarning, messages):
        uv.read(
            files,
            flag_init=False,
            remove_flagged_ants=False,
            correct_cable_len=False,
            use_future_array_shapes=True,
        )

    with fits.open(filelist[3]) as aoflags:
        flags = aoflags[1].data.field("FLAGS")
    flags = flags[:, :, np.newaxis]
    flags = np.repeat(flags, 4, axis=2)

    assert np.all(uv.flag_array == flags)


@pytest.mark.filterwarnings(
    "ignore:coarse channels are not contiguous for this observation"
)
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:coarse channel, start time, and end time flagging")
def test_aoflagger_flags_multiple(tmp_path):
    """Test aoflagger flags with multiple coarse bands"""
    mod_mini_6 = str(tmp_path / "mini_gpubox06_01.fits")
    with fits.open(filelist[2]) as mini6:
        mini6[1].header["time"] = 1447698337
        mini6.writeto(mod_mini_6)
    files = filelist[0:2] + filelist[3:5]
    files.append(mod_mini_6)

    uv = UVData()
    uv.read(
        files, flag_init=False, remove_flagged_ants=False, use_future_array_shapes=True
    )

    with fits.open(filelist[3]) as aoflags:
        flags1 = aoflags[1].data.field("FLAGS")
    with fits.open(filelist[4]) as aoflags:
        flags2 = aoflags[1].data.field("FLAGS")
    flags = np.array([flags2[:, 0], flags1[:, 0]])
    flags = np.transpose(flags)
    flags = flags[:, :, np.newaxis]
    flags = np.repeat(flags, 4, axis=2)

    assert np.all(uv.flag_array == flags)


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:coarse channel, start time, and end time flagging")
def test_mismatch_flags():
    """Break by submitting flag and gpubox files from different coarse bands."""
    uv = UVData()
    files = filelist[0:2]
    files.append(filelist[4])
    with pytest.raises(ValueError, match="flag file coarse bands do not match"):
        uv.read(files)


@pytest.mark.filterwarnings(
    "ignore:coarse channels are not contiguous for this observation"
)
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_propagate_coarse_flags():
    """
    Test that the flag(without flag_int) and nsample arrays correctly reflect data.
    """
    uv = UVData()
    uv.read(
        filelist[0:3],
        flag_init=False,
        propagate_coarse_flags=True,
        use_future_array_shapes=True,
    )
    assert np.all(uv.flag_array)


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_start_flag(tmp_path):
    """Test the default value of start_flag."""
    uv1 = UVData()
    uv1.read(
        filelist[0:2],
        flag_init=True,
        start_flag="goodtime",
        end_flag=0,
        edge_width=0,
        flag_dc_offset=False,
        use_future_array_shapes=True,
    )
    good_ants = np.delete(np.unique(uv1.ant_1_array), [59, 114])
    uv1.select(antenna_nums=good_ants)
    # start_time is after goodtime, so data for good antennas should be unflagged
    assert np.all(~uv1.flag_array)
    mod_mini = str(tmp_path / "starttime_gpubox01_00.fits")
    with fits.open(filelist[1]) as mini:
        mini[1].header["time"] = 1447698334
        mini.writeto(mod_mini)
    uv2 = UVData()
    uv2.read(
        [filelist[0], mod_mini],
        flag_init=True,
        start_flag="goodtime",
        end_flag=0,
        edge_width=0,
        flag_dc_offset=False,
        use_future_array_shapes=True,
    )
    # start_time is before goodtime, so all data should be flagged
    assert np.all(uv2.flag_array)


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_start_flag_goodtime_ppds():
    """Test that error is thrown using 'goodtime' with only ppds file."""
    uv = UVData()
    with pytest.raises(ValueError, match="To use start_flag='goodtime',"):
        uv.read([filelist[1], filelist[7]], flag_init=True, start_flag="goodtime")


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_start_flag_bad_string():
    """Test that error is thrown if start_flag is given string other than 'goodtime'"""
    uv = UVData()
    with pytest.raises(
        ValueError, match="start_flag must be int or float or 'goodtime'"
    ):
        uv.read(filelist[0:2], flag_init=True, start_flag="badstring")


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_start_flag_int_time(tmp_path):
    """Test goodtime returning a start_flag smaller than integration time."""
    uv = UVData()
    new_meta = str(tmp_path / "1131733552_goodtime.metafits")
    with fits.open(filelist[0]) as meta:
        meta[0].header["GOODTIME"] = 1447698337.25
        meta.writeto(new_meta)
    uv.read(
        [new_meta, filelist[1]],
        flag_init=True,
        start_flag="goodtime",
        use_future_array_shapes=True,
    )
    # first integration time should be flagged
    # data only has one integration time, so all data should be flagged
    assert np.all(uv.flag_array)


def test_input_output_mapping():
    """Test the input_output_mapping function."""
    mapping_dict = {}
    # fmt: off
    pfb_mapper = [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63,
    ]
    # fmt: on
    for p in range(4):
        for i in range(64):
            mapping_dict[pfb_mapper[i] + p * 64] = p * 64 + i

    # compare with output from function
    function_output = input_output_mapping()
    assert function_output == mapping_dict

    return


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_van_vleck_int():
    """Test van vleck correction integral implementation."""
    uv1 = UVData()
    uv1.read(
        filelist[8:10],
        flag_init=False,
        data_array_dtype=np.complex64,
        correct_van_vleck=True,
        cheby_approx=False,
        remove_coarse_band=False,
        remove_dig_gains=False,
        remove_flagged_ants=False,
        correct_cable_len=False,
        use_future_array_shapes=True,
    )
    # read in file corrected using integrate.quad with 1e-10 precision
    uv2 = UVData()
    uv2.read(filelist[10], use_future_array_shapes=True)
    assert np.allclose(uv1.data_array, uv2.data_array)


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_van_vleck_cheby():
    """Test van vleck correction chebyshev implementation."""
    uv1 = UVData()
    uv1.read(
        filelist[8:10],
        flag_init=False,
        correct_van_vleck=True,
        cheby_approx=True,
        remove_coarse_band=False,
        remove_dig_gains=False,
        correct_cable_len=False,
        use_future_array_shapes=True,
    )
    # read in file corrected using integrate.quad with 1e-10 precision
    uv2 = UVData()
    uv2.read(filelist[10], use_future_array_shapes=True)

    # select only good ants
    good_ants = np.delete(np.unique(uv2.ant_1_array), 76)
    uv2.select(antenna_nums=good_ants)

    assert np.allclose(uv1.data_array, uv2.data_array)


def test_van_vleck_interp(tmp_path):
    """Test van vleck correction with sigmas out of cheby interpolation range."""
    small_sigs = str(tmp_path / "small_sigs07_01.fits")
    with fits.open(filelist[8]) as mini:
        mini[1].data = np.full((1, 66048), 7744)
        mini.writeto(small_sigs)
    messages = ["values are being corrected with the van vleck integral"]
    messages = messages * 10
    messages.append("some coarse channel files were not submitted")
    messages.append("Fixing auto-correlations to be be real-only,")
    uv = UVData()
    with uvtest.check_warnings(UserWarning, messages):
        uv.read(
            [small_sigs, filelist[9]],
            flag_init=False,
            correct_van_vleck=True,
            cheby_approx=True,
            remove_coarse_band=False,
            remove_dig_gains=False,
            correct_cable_len=False,
            use_future_array_shapes=True,
        )


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_remove_flagged_ants(tmp_path):
    """Test remove_flagged_ants."""
    uv1 = UVData()
    uv1.read(filelist[8:10], remove_flagged_ants=True, use_future_array_shapes=True)
    uv2 = UVData()
    uv2.read(filelist[8:10], remove_flagged_ants=False, use_future_array_shapes=True)
    good_ants = np.delete(np.unique(uv2.ant_1_array), 76)

    uv2.select(antenna_nums=good_ants)

    assert uv1 == uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:.*values are being corrected with the van vleck")
def test_small_sigs(tmp_path):
    """Test flag_small_auto_ants."""
    small_sigs = str(tmp_path / "small_sigs07_02.fits")
    with fits.open(filelist[8]) as mini:
        mini[1].data[0, 0] = 1000
        mini.writeto(small_sigs)
    uv1 = UVData()
    uv1.read(
        [small_sigs, filelist[9]],
        correct_van_vleck=True,
        flag_small_auto_ants=True,
        use_future_array_shapes=True,
    )
    messages = ["values are being corrected with the van vleck integral"]
    messages = messages * 8
    messages.append("some coarse channel files were not submitted")
    uv2 = UVData()
    with uvtest.check_warnings(UserWarning, messages):
        uv2.read(
            [small_sigs, filelist[9]],
            correct_van_vleck=True,
            flag_small_auto_ants=False,
            use_future_array_shapes=True,
        )

    assert "flagged by the Van Vleck" in uv1.history
    assert uv2.Nants_data - uv1.Nants_data == 1


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
def test_bscale(tmp_path):
    """Test that bscale is saved correctly"""
    # some data does not have bscale in the zeroth hdu
    bscale = str(tmp_path / "bscale_01_00.fits")
    with fits.open(filelist[1], do_not_scale_image_data=True) as mini:
        mini[0].header.remove("BSCALE")
        mini.writeto(bscale)
    uv1 = UVData()
    uv2 = UVData()
    uv3 = UVData()
    uv4 = UVData()
    # check when bscale is not in zeroth hdu but is in first
    uv1.read([filelist[0], bscale], use_future_array_shapes=True)
    assert uv1.extra_keywords["SCALEFAC"] == 0.5
    # check when bscale is in both zeroth and first hdu
    uv2.read(filelist[0:2], use_future_array_shapes=True)
    assert uv2.extra_keywords["SCALEFAC"] == 0.5
    # check pre-October 2014 data
    uv3.read(filelist[8:10], use_future_array_shapes=True)
    assert uv3.extra_keywords["SCALEFAC"] == 0.25
    # check mwax data
    uv4.read(filelist[11:13], use_future_array_shapes=True)
    assert "SCALEFAC" not in uv4.extra_keywords.keys()
