# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for MWACorrFITS object."""

import copy
import importlib
import itertools
import os
import shutil
import warnings

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings
from pyuvdata.uvdata.mwa_corr_fits import input_output_mapping

hasbench = importlib.util.find_spec("pytest_benchmark") is not None

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


def spoof_mwax(tmp_path, nfreq=16, ntimes=2, ncoarse=1):
    fine_res = 1280 // nfreq

    cb_spoof = str(tmp_path / f"mwax_cb_spoof{fine_res}_ch137_000.fits")
    meta_spoof = str(tmp_path / f"mwax_cb_spoof{fine_res}.metafits")

    outfiles = []
    with fits.open(filelist[12]) as mini1:
        mini1[1].data = np.repeat(mini1[1].data, nfreq, axis=1)  # data
        mini1[2].data = np.repeat(mini1[2].data, nfreq, axis=1)  # weights

        for tind in range(1, ntimes):
            new_time = mini1[1].header["TIME"] + 2 * tind
            extra_dat = np.copy(mini1[1].data)
            extra_samps = np.copy(mini1[2].data)
            mini1.append(fits.ImageHDU(extra_dat))
            mini1.append(fits.ImageHDU(extra_samps))
            data_hdu_ind = tind * 2 + 1
            samp_hdu_ind = tind * 2 + 2
            mini1[data_hdu_ind].header["TIME"] = new_time
            mini1[samp_hdu_ind].header["TIME"] = new_time
            mini1[data_hdu_ind].header["MILLITIM"] = 0
            mini1[samp_hdu_ind].header["MILLITIM"] = 0

        mini1.writeto(cb_spoof)
        outfiles.append(cb_spoof)
    for nfile in range(1, ncoarse):
        filename = str(tmp_path / f"mwax_cb_spoof{fine_res}_ch{137 + nfile}_000.fits")
        shutil.copy(cb_spoof, filename)
        outfiles.append(filename)

    with fits.open(filelist[11]) as meta:
        meta[0].header["FINECHAN"] = fine_res
        meta[0].header["NCHANS"] = len(meta[0].header["CHANNELS"].split(",")) * nfreq
        meta.writeto(meta_spoof)
        outfiles.append(meta_spoof)

    return outfiles


def spoof_legacy(tmp_path, nfreq=16, ntimes=2, ncoarse=2):
    input_files = filelist[1:3]
    spoof_files = [
        str(tmp_path / "spoof_01_00.fits"),
        str(tmp_path / "spoof_06_00.fits"),
    ]

    files_use = [filelist[0]]
    for f_ind in range(ncoarse):
        with fits.open(input_files[f_ind]) as mini:
            mini[1].data = np.repeat(mini[1].data, nfreq, axis=0)
            extra_dat = np.copy(mini[1].data)
            for tind in range(1, ntimes):
                new_time = mini[1].header["TIME"] + tind // 2
                new_millitime = (tind % 2) * 500
                mini.append(fits.ImageHDU(extra_dat))
                hdu_ind = tind + 1
                mini[hdu_ind].header["MILLITIM"] = new_millitime
                mini[hdu_ind].header["TIME"] = new_time
            mini.writeto(spoof_files[f_ind])
            files_use.append(spoof_files[f_ind])

    return files_use


@pytest.fixture(scope="module")
def flag_file_init(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pyuvdata_corr_fits", numbered=True)
    flag_testfiles = spoof_legacy(tmp_path, nfreq=8, ntimes=3)

    yield flag_testfiles


def test_read_mwa_write_uvfits(tmp_path):
    """
    MWA correlator fits to uvfits loopback test.

    Read in MWA correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    mwa_uv = UVData()
    uvfits_uv = UVData()
    with check_warnings(
        UserWarning, match="some coarse channel files were not submitted"
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
    spoof_files = spoof_mwax(tmp_path)
    mwax_uv = UVData()
    uvfits_uv = UVData()
    messages = [
        "Fixing auto-correlations to be be real-only, after some imaginary values",
        "some coarse channel files were not submitted",
    ]
    with check_warnings(UserWarning, messages):
        mwax_uv.read(spoof_files, correct_cable_len=True, phase_to_pointing_center=True)
    testfile = str(tmp_path / "outtest_MWAXcorr.uvfits")
    mwax_uv.write_uvfits(testfile)
    uvfits_uv.read_uvfits(testfile)

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
    uv.read([meta_spoof_file, filelist[1]])

    assert "DELAYMOD" not in uv.extra_keywords


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
    )
    cotter_uv.read(filelist[6])
    # cotter doesn't record the auto xy polarizations
    # due to a possible bug in cotter, the auto yx polarizations are conjugated
    # fix these before testing data_array
    autos = np.isclose(mwa_uv.ant_1_array - mwa_uv.ant_2_array, 0.0)
    cotter_uv.data_array[autos, :, 2] = cotter_uv.data_array[autos, :, 3]
    cotter_uv.data_array[autos, :, 3] = np.conj(cotter_uv.data_array[autos, :, 3])
    np.testing.assert_allclose(
        mwa_uv.data_array, cotter_uv.data_array, atol=1e-4, rtol=0
    )
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
    with check_warnings(UserWarning, messages):
        mwa_uv.read(files, correct_cable_len=True, phase_to_pointing_center=True)
    testfile = str(tmp_path / "outtest_MWAcorr.uvfits")
    mwa_uv.write_uvfits(testfile)
    uvfits_uv.read_uvfits(testfile)

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

    mwa_uv.read([set1, set2])

    mwa_uv2 = UVData()
    messages = [
        "some coarse channel files were not submitted",
        "some coarse channel files were not submitted",
    ]
    with check_warnings(UserWarning, messages):
        mwa_uv2.read([set1, set2], file_type="mwa_corr_fits")

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
    mwa_uv.read([set1, set2], axis="freq")

    mwa_uv2 = UVData()
    messages = [
        "some coarse channel files were not submitted",
        "some coarse channel files were not submitted",
    ]
    with check_warnings(UserWarning, messages):
        mwa_uv2.read([set1, set2], axis="freq", file_type="mwa_corr_fits")

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
    with check_warnings(UserWarning, messages):
        mwa_uv.read(subfiles, use_aoflagger_flags=False)

    del mwa_uv

    mwa_uv = UVData()
    with pytest.raises(ValueError, match="no flag files submitted"):
        mwa_uv.read(subfiles[0:2], use_aoflagger_flags=True)
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
    with check_warnings(UserWarning, messages):
        mwa_uv1.read(order1)
    with check_warnings(UserWarning, messages):
        mwa_uv2.read(order2)

    assert mwa_uv1 == mwa_uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_ppds(tmp_path):
    """Test handling of ppds files"""
    # turnaround test with just ppds file given
    mwa_uv = UVData()
    mwa_uv.read(
        [filelist[1], filelist[7]], phase_to_pointing_center=True, flag_init=False
    )
    testfile = str(tmp_path / "outtest_MWAcorr.uvfits")
    mwa_uv.write_uvfits(testfile)
    uvfits_uv = UVData()
    uvfits_uv.read_uvfits(testfile)

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
    mwa_uv.read([filelist[0], filelist[1], filelist[7]])
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
    with pytest.raises(
        ValueError, match="files submitted have different numbers of fine channels"
    ):
        mwa_uv.read([bad_fine, filelist[1]])
    del mwa_uv


def test_fine_channels_mwax(tmp_path):
    """
    Break read_mwa_corr_fits by submitting mwax files with different fine channels.

    Test that error is raised if files with different numbers of fine channels
    are submitted.
    """
    mwax_uv = UVData()
    spoof_files = spoof_mwax(tmp_path)
    with pytest.raises(
        ValueError, match="files submitted have different numbers of fine channels"
    ):
        mwax_uv.read(spoof_files + [filelist[12]])
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
    )

    freq_inds = [0, 1, 4, 6, 7, 8, 9, 12, 14, 15]
    freq_inds_complement = [ind for ind in range(16) if ind not in freq_inds]

    assert np.all(uv.flag_array[:, freq_inds, :]), (
        "Not all of edge and center channels are flagged!"
    )
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
    )

    reshape = [uv.Ntimes, uv.Nbls, uv.Nfreqs, uv.Npols]
    time_inds = [0, 1, -1, -2]
    assert np.all(uv.flag_array.reshape(reshape)[time_inds, :, :, :]), (
        "Not all of start and end times are flagged."
    )
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


@pytest.mark.filterwarnings(
    "ignore:coarse channels are not contiguous for this observation"
)
@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_flag_init_error_freq_sel(flag_file_init):
    from pyuvdata.uvdata.mwa_corr_fits import MWACorrFITS

    mwa_obj = MWACorrFITS()
    mwa_obj.read_mwa_corr_fits(flag_file_init, flag_init=False)

    with pytest.raises(
        AssertionError, match="If freq_inds is not None, n_orig_freq must be passed."
    ):
        mwa_obj.flag_init(32, freq_inds=np.arange(10), n_orig_freq=None)


def test_read_metadata_only(tmp_path):
    """Test reading an MWA corr fits file as metadata only."""
    uvd = UVData()
    messages = ["some coarse channel files were not submitted"]
    with check_warnings(UserWarning, messages):
        uvd.read(
            filelist[0:2],
            correct_cable_len=True,
            phase_to_pointing_center=True,
            read_data=False,
        )

    assert uvd.metadata_only


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_data_array_precision():
    uv = UVData()
    uv2 = UVData()
    # read in data array as single precision
    uv.read(filelist[0:2], data_array_dtype=np.complex64)
    # now read as double precision
    uv2.read(filelist[0:2], data_array_dtype=np.complex128)

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
    uv.read(filelist[0:2], nsample_array_dtype=np.float32)
    uv2.read(filelist[0:2], nsample_array_dtype=np.float64)
    uv3.read(filelist[0:2], nsample_array_dtype=np.float16)

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
    uv1.read(filelist[0:2], data_array_dtype=np.complex64)

    uv2 = UVData()
    uv2.read(filelist[0:2], remove_dig_gains=False)

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
    spoof_files = spoof_legacy(tmp_path, nfreq=32, ncoarse=1)

    uv1 = UVData()
    uv1.read(spoof_files, data_array_dtype=np.complex64)

    uv2 = UVData()
    uv2.read(spoof_files, remove_coarse_band=False)

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

    spoof_files = spoof_mwax(tmp_path, nfreq=32)

    uv1 = UVData()
    uv1.read(spoof_files)

    uv2 = UVData()
    uv2.read(spoof_files, remove_coarse_band=False)

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
    spoof_files = spoof_mwax(tmp_path, nfreq=16)

    uv1 = UVData()
    uv1.read(spoof_files)

    uv2 = UVData()
    uv2.read(spoof_files, remove_coarse_band=False)

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
    spoof_files = spoof_mwax(tmp_path, nfreq=8)

    uv = UVData()
    with pytest.raises(ValueError, match="mwax passband shapes are only available"):
        uv.read(spoof_files, flag_init=False)


def test_aoflagger_flags():
    """Test using aoflagger flags"""
    uv = UVData()
    files = filelist[0:2]
    files.append(filelist[3])
    messages = [
        "some coarse channel files were not submitted",
        "coarse channel, start time, and end time flagging will default",
    ]
    with check_warnings(UserWarning, messages):
        uv.read(
            files, flag_init=False, remove_flagged_ants=False, correct_cable_len=False
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
    uv.read(files, flag_init=False, remove_flagged_ants=False)

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
    uv.read(filelist[0:3], flag_init=False, propagate_coarse_flags=True)
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
    uv.read([new_meta, filelist[1]], flag_init=True, start_flag="goodtime")
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
    )
    # read in file corrected using integrate.quad with 1e-10 precision
    uv2 = UVData()
    uv2.read(filelist[10])

    np.testing.assert_allclose(
        uv1.data_array,
        uv2.data_array,
        rtol=uv1._data_array.tols[0],
        atol=uv1._data_array.tols[1],
    )


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
    )
    # read in file corrected using integrate.quad with 1e-10 precision
    uv2 = UVData()
    uv2.read(filelist[10])

    # select only good ants
    good_ants = np.delete(np.unique(uv2.ant_1_array), 76)
    uv2.select(antenna_nums=good_ants)

    np.testing.assert_allclose(
        uv1.data_array,
        uv2.data_array,
        rtol=uv1._data_array.tols[0],
        atol=uv1._data_array.tols[1],
    )


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
    with check_warnings(UserWarning, messages):
        uv.read(
            [small_sigs, filelist[9]],
            flag_init=False,
            correct_van_vleck=True,
            cheby_approx=True,
            remove_coarse_band=False,
            remove_dig_gains=False,
            correct_cable_len=False,
        )


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
def test_remove_flagged_ants(tmp_path):
    """Test remove_flagged_ants."""
    uv1 = UVData()
    uv1.read(filelist[8:10], remove_flagged_ants=True)
    uv2 = UVData()
    uv2.read(filelist[8:10], remove_flagged_ants=False)
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
        [small_sigs, filelist[9]], correct_van_vleck=True, flag_small_auto_ants=True
    )
    messages = ["values are being corrected with the van vleck integral"]
    messages = messages * 8
    messages.append("some coarse channel files were not submitted")
    uv2 = UVData()
    with check_warnings(UserWarning, messages):
        uv2.read(
            [small_sigs, filelist[9]],
            correct_van_vleck=True,
            flag_small_auto_ants=False,
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
    uv1.read([filelist[0], bscale])
    assert uv1.extra_keywords["SCALEFAC"] == 0.5
    # check when bscale is in both zeroth and first hdu
    uv2.read(filelist[0:2])
    assert uv2.extra_keywords["SCALEFAC"] == 0.5
    # check pre-October 2014 data
    uv3.read(filelist[8:10])
    assert uv3.extra_keywords["SCALEFAC"] == 0.25
    # check mwax data
    uv4.read(filelist[11:13])
    assert "SCALEFAC" not in uv4.extra_keywords


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
def test_default_corrections(tmp_path):
    """Test that default corrections are applied"""
    # mwa_corr_fits defaults to applying corrections for cable reflections,
    # digital gains, and the polyphase filter bank bandpass
    uv1 = UVData()
    uv2 = UVData()
    uv1.read(filelist[0:2])
    uv2.read(filelist[11:13])

    assert "Divided out digital gains" in uv1.history
    assert "Divided out digital gains" in uv2.history
    assert "Divided out pfb coarse channel bandpass" in uv1.history
    assert "Divided out pfb coarse channel bandpass" in uv2.history
    assert "Applied cable length correction" in uv1.history
    assert "Applied cable length correction" in uv2.history


@pytest.mark.skipif(not hasbench, reason="benchmark utility not installed")
def test_read_mwa(benchmark, tmp_path):
    """
    MWA correlator fits to uvfits loopback test.

    Read in MWA correlator files, write out as uvfits, read back in and check
    for object equality.
    """
    mwa_uv = UVData()
    uvfits_uv = UVData()
    # we check warnings earlier here we care about performance.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        benchmark(
            mwa_uv.read,
            filelist[0:2],
            correct_cable_len=True,
            phase_to_pointing_center=True,
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
@pytest.mark.skipif(not hasbench, reason="benchmark utility not installed")
def test_read_mwax(benchmark, tmp_path):
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

    # we check warnings earlier here we care about performance.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        benchmark(
            mwax_uv.read,
            [spoof_file, filelist[11]],
            correct_cable_len=True,
            phase_to_pointing_center=True,
        )
    testfile = str(tmp_path / "outtest_MWAXcorr.uvfits")
    mwax_uv.write_uvfits(testfile)
    uvfits_uv.read_uvfits(testfile)

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
@pytest.mark.skipif(not hasbench, reason="benchmark utility not installed")
def test_corr_fits_select_on_read(benchmark):
    mwa_uv = UVData()
    mwa_uv2 = UVData()
    mwa_uv.read(filelist[0:2], correct_cable_len=True)
    unique_times = np.unique(mwa_uv.time_array)
    select_times = unique_times[
        np.where(
            (unique_times >= np.min(mwa_uv.time_array))
            & (unique_times <= np.mean(mwa_uv.time_array))
        )
    ]
    mwa_uv.select(times=select_times)
    # we check warnings earlier here we care about performance.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        benchmark(
            mwa_uv2.read,
            filelist[0:2],
            correct_cable_len=True,
            time_range=[np.min(mwa_uv.time_array), np.mean(mwa_uv.time_array)],
        )

    # histories are slightly different in the ordering of content. Not a concern
    mwa_uv2.history = mwa_uv.history
    assert mwa_uv == mwa_uv2


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.skipif(not hasbench, reason="benchmark utility not installed")
@pytest.mark.parametrize("cheby", [True, False], ids=lambda x: f"cheby={x:}")
def test_van_vleck(benchmark, cheby):
    uv1 = UVData()
    benchmark(
        uv1.read,
        filelist[8:10],
        flag_init=False,
        correct_van_vleck=True,
        cheby_approx=cheby,
        remove_coarse_band=False,
        remove_dig_gains=False,
        remove_flagged_ants=cheby,
        correct_cable_len=False,
    )
    # read in file corrected using integrate.quad with 1e-10 precision
    uv2 = UVData()
    uv2.read(filelist[10])

    if cheby:
        # select only good ants
        good_ants = np.delete(np.unique(uv2.ant_1_array), 76)
        uv2.select(antenna_nums=good_ants)

    np.testing.assert_allclose(
        uv1.data_array,
        uv2.data_array,
        rtol=uv1._data_array.tols[0],
        atol=uv1._data_array.tols[1],
    )


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.parametrize(
    ["select_kwargs", "warn_msg"],
    [
        [{"antenna_nums": [18, 31, 66, 95]}, ""],
        [{"antenna_names": [f"Tile{ant:03d}" for ant in [18, 31, 66, 95]]}, ""],
        [{"bls": [(48, 34), (96, 11), (22, 87)]}, ""],
        [
            {"bls": [(48, 34, "xx"), (96, 11, "xx"), (22, 87, "xx")]},
            "a select on read keyword is set that is not supported by "
            "read_mwa_corr_fits. This select will be done after reading the file.",
        ],
        [
            {"ant_str": "48_34,96_11,22_87"},
            "a select on read keyword is set that is not supported by "
            "read_mwa_corr_fits. This select will be done after reading the file.",
        ],
    ],
)
@pytest.mark.parametrize("mwax", [False, True])
def test_partial_read_bl_axis(tmp_path, mwax, select_kwargs, warn_msg):
    if mwax:
        files_use = spoof_mwax(tmp_path, nfreq=16)
    else:
        files_use = spoof_legacy(tmp_path, nfreq=16, ncoarse=1)

    uv_full = UVData.from_file(files_use)

    warn_msg_list = ["some coarse channel files were not submitted"]
    if warn_msg != "":
        warn_msg_list.append(warn_msg)

    if mwax and ("bls" not in select_kwargs or warn_msg != ""):
        # The bls selection has no autos
        warn_msg_list.append("Fixing auto-correlations to be be real-only")

    with check_warnings(UserWarning, match=warn_msg_list):
        uv_partial = UVData.from_file(files_use, **select_kwargs)
    exp_uv = uv_full.select(**select_kwargs, inplace=False)

    if "bls" in select_kwargs:
        sel_type = "antenna pairs"
    else:
        sel_type = "antennas"

    if warn_msg == "":
        # history doesn't match because of different order of operations.
        # fix order of operations in history
        loc_divided = uv_full.history.find("Divided")
        if not mwax:
            loc_downsel = uv_full.history.find("  Downselected")
            hist_end = uv_full.history[loc_divided:loc_downsel]
        else:
            hist_end = uv_full.history[loc_divided:]
        exp_uv.history = (
            uv_full.history[:loc_divided]
            + f" Downselected to specific {sel_type} using pyuvdata. "
            + hist_end
        )
    assert uv_partial == exp_uv


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.parametrize("select", ["times", "time_range", "lsts", "lst_range"])
@pytest.mark.parametrize("mwax", [False, True])
def test_partial_read_time_axis(tmp_path, mwax, select):
    if mwax:
        files_use = spoof_mwax(tmp_path, nfreq=16, ntimes=6)
    else:
        files_use = spoof_legacy(tmp_path, nfreq=16, ntimes=6, ncoarse=1)

    uv_full = UVData.from_file(files_use)
    unique_times = np.unique(uv_full.time_array)
    assert uv_full.Ntimes == 6
    unique_lsts = np.unique(uv_full.lst_array)

    if select == "times":
        select_kwargs = {"times": unique_times[[0, 2]]}
        sel_type = "times"
    elif select == "lsts":
        select_kwargs = {"lsts": unique_lsts[[0, 2]]}
        sel_type = "lsts"
    elif select == "time_range":
        select_kwargs = {"time_range": [unique_times[0], unique_times[3]]}
        sel_type = "times"
    else:
        select_kwargs = {"lst_range": [unique_lsts[0], unique_lsts[3]]}
        sel_type = "lsts"

    uv_partial = UVData.from_file(files_use, **select_kwargs)
    exp_uv = uv_full.select(**select_kwargs, inplace=False)
    # fix order of operations in history
    loc_divided = uv_full.history.find("Divided")
    exp_uv.history = (
        uv_full.history[:loc_divided]
        + f" Downselected to specific {sel_type} using pyuvdata. "
        + uv_full.history[loc_divided:]
    )

    assert uv_partial == exp_uv


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:coarse channels are not contiguous for this")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.filterwarnings("ignore:Selected frequencies are not evenly spaced.")
@pytest.mark.parametrize(
    ["mwax", "select_kwargs", "read_kwargs", "nspw_exp"],
    [
        [True, {"frequencies": np.arange(10)}, {"propagate_coarse_flags": False}, 1],
        [True, {"frequencies": np.arange(20)}, {}, 1],
        [True, {"frequencies": np.arange(10, 20)}, {}, 1],
        [
            True,
            {"freq_chans": np.concatenate((np.arange(0, 10), np.arange(20, 30)))},
            {},
            2,
        ],
        [False, {"frequencies": np.arange(5, 10)}, {}, 2],
    ],
)
def test_partial_read_freq_axis(tmp_path, mwax, select_kwargs, read_kwargs, nspw_exp):
    if mwax:
        files_use = spoof_mwax(tmp_path, nfreq=16, ntimes=1, ncoarse=2)
    else:
        files_use = spoof_legacy(tmp_path, nfreq=8, ntimes=1, ncoarse=2)

    uv_full = UVData.from_file(files_use, **read_kwargs)

    kwargs_use = copy.deepcopy(select_kwargs)
    if "frequencies" in select_kwargs:
        kwargs_use["frequencies"] = uv_full.freq_array[select_kwargs["frequencies"]]

    uv_partial = UVData.from_file(files_use, **read_kwargs, **kwargs_use)
    assert uv_partial.Nspws == nspw_exp

    exp_uv = uv_full.select(**kwargs_use, inplace=False)
    # fix up spws
    if nspw_exp == 1 and (exp_uv.Nspws != 1 or exp_uv.spw_array[0] != 0):
        exp_uv.Nspws = 1
        exp_uv.spw_array = np.array([0])
        exp_uv.flex_spw_id_array = np.full(
            exp_uv.Nfreqs, exp_uv.spw_array[0], dtype=int
        )
    elif exp_uv.Nspws != nspw_exp:
        # this only happens for MWAX where selecting discontinuous sets
        exp_uv.Nspws = 2
        exp_uv.spw_array = np.array([137, 138])
        n_137 = (np.nonzero(select_kwargs["freq_chans"] < 16)[0]).size
        n_138 = (np.nonzero(select_kwargs["freq_chans"] >= 16)[0]).size
        exp_uv.flex_spw_id_array = np.concatenate(
            (np.full(n_137, 137, dtype=int), np.full(n_138, 138, dtype=int))
        )

    # fix order of operations in history
    loc_divided = uv_full.history.find("Divided")
    exp_uv.history = (
        uv_full.history[:loc_divided]
        + " Downselected to specific frequencies using pyuvdata. "
        + uv_full.history[loc_divided:]
    )
    assert uv_partial == exp_uv


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"polarizations": ["xx"]},
        {"polarizations": np.atleast_3d(["xx", "yy"])},
        {"polarizations": ["xx", "xy"]},
        {"polarizations": [-7, -8]},
    ],
)
@pytest.mark.parametrize("mwax", [False, True])
def test_partial_read_pol_axis(tmp_path, mwax, select_kwargs):
    if mwax:
        files_use = spoof_mwax(tmp_path, nfreq=16)
    else:
        files_use = spoof_legacy(tmp_path, nfreq=16, ncoarse=1)

    uv_full = UVData.from_file(files_use)

    uv_partial = UVData.from_file(files_use, **select_kwargs)
    exp_uv = uv_full.select(**select_kwargs, inplace=False)
    # fix order of operations in history
    loc_divided = uv_full.history.find("Divided")
    exp_uv.history = (
        uv_full.history[:loc_divided]
        + " Downselected to specific polarizations using pyuvdata. "
        + uv_full.history[loc_divided:]
    )

    assert uv_partial == exp_uv


@pytest.mark.filterwarnings("ignore:some coarse channel files were not submitted")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.parametrize(
    ["select_kwargs", "selections"],
    [
        [
            {"polarizations": ["xx", "yy"], "antenna_nums": [18, 31, 66, 95]},
            ["antennas", "polarizations"],
        ],
        [
            {"polarizations": ["xx"], "antenna_nums": "most"},
            ["antennas", "polarizations"],
        ],
        [{"antenna_nums": [18, 31, 66, 95], "times": [0, 2]}, ["antennas", "times"]],
        [{"polarizations": ["xx", "yy"], "times": [0, 2]}, ["times", "polarizations"]],
        [
            {"polarizations": ["xx", "yy"], "freq_chans": np.arange(2)},
            ["frequencies", "polarizations"],
        ],
        [
            {"antenna_nums": [18, 31, 66, 95], "freq_chans": np.arange(10)},
            ["antennas", "frequencies"],
        ],
        [{"times": [0, 2], "freq_chans": np.arange(10)}, ["times", "frequencies"]],
        [
            {
                "antenna_nums": [18, 31, 66, 95],
                "times": [0, 2],
                "freq_chans": np.arange(10),
                "polarizations": ["xx", "yy"],
            },
            ["antennas", "times", "frequencies", "polarizations"],
        ],
    ],
)
@pytest.mark.parametrize("mwax", [False, True])
def test_partial_read_multi(tmp_path, mwax, select_kwargs, selections):
    if mwax:
        files_use = spoof_mwax(tmp_path, nfreq=16, ntimes=3)
    else:
        files_use = spoof_legacy(tmp_path, nfreq=16, ntimes=3, ncoarse=1)

    uv_full = UVData.from_file(files_use)
    all_ants = uv_full.get_ants()

    kwargs_use = copy.deepcopy(select_kwargs)
    if "antenna_nums" in select_kwargs and isinstance(
        select_kwargs["antenna_nums"], str
    ):
        kwargs_use["antenna_nums"] = all_ants[0 : all_ants.size // 2]
    if "times" in select_kwargs:
        unique_times = np.unique(uv_full.time_array)
        kwargs_use["times"] = unique_times[select_kwargs["times"]]

    uv_partial = UVData.from_file(files_use, **kwargs_use)
    exp_uv = uv_full.select(**kwargs_use, inplace=False)
    # fix order of operations in history
    loc_divided = uv_full.history.find("Divided")
    if not mwax and "antenna_nums" in select_kwargs:
        loc_downsel = uv_full.history.find("  Downselected")
        hist_end = uv_full.history[loc_divided:loc_downsel]
    else:
        hist_end = uv_full.history[loc_divided:]
    exp_uv.history = (
        uv_full.history[:loc_divided]
        + f" Downselected to specific {', '.join(selections)} using pyuvdata. "
        + hist_end
    )

    assert uv_partial == exp_uv


def test_partial_read_errors(tmp_path):
    files_use = spoof_mwax(tmp_path, nfreq=16, ntimes=1)
    uvd = UVData()
    with pytest.raises(
        ValueError, match="bls must be a list of 2-tuples giving antenna number pairs"
    ):
        uvd.read_mwa_corr_fits(files_use, bls=[(18, 31, "xx")])
