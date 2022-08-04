# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for UVFITS object.

"""
import os

import erfa
import numpy as np
import pytest
from astropy.io import fits

import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from pyuvdata.data import DATA_PATH

casa_tutorial_uvfits = os.path.join(
    DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits"
)

paper_uvfits = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAAM.uvfits")


@pytest.fixture(scope="session")
def uvfits_nospw_main():
    uv_in = UVData()
    # This file has a crazy epoch (2291.34057617) which breaks the uvw_antpos check
    # Since it's a PAPER file, I think this is a bug in the file, not in the check.
    uv_in.read(paper_uvfits, run_check_acceptability=False)

    return uv_in


@pytest.fixture(scope="function")
def uvfits_nospw(uvfits_nospw_main):
    return uvfits_nospw_main.copy()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_read_nrao(casa_uvfits):
    """Test reading in a CASA tutorial uvfits file."""
    uvobj = casa_uvfits
    expected_extra_keywords = ["OBSERVER", "SORTORD", "SPECSYS", "RESTFREQ", "ORIGIN"]
    assert expected_extra_keywords.sort() == list(uvobj.extra_keywords.keys()).sort()

    # test reading metadata only
    uvobj2 = UVData()
    uvobj2.read(casa_tutorial_uvfits, read_data=False)

    assert expected_extra_keywords.sort() == list(uvobj2.extra_keywords.keys()).sort()
    assert uvobj2.check()

    uvobj3 = uvobj.copy(metadata_only=True)
    assert uvobj2 == uvobj3


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:ITRF coordinate frame detected")
@pytest.mark.filterwarnings("ignore:Telescope OVRO_MMA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_time_precision(tmp_path):
    """
    This tests that the times are round-tripped through write/read uvfits to sufficient
    precision.
    """
    pytest.importorskip("casacore")
    lwa_file = os.path.join(
        DATA_PATH, "2018-03-21-01_26_33_0004384620257280_000000_downselected.ms"
    )
    uvd = UVData()
    uvd.read(lwa_file)

    testfile = os.path.join(tmp_path, "lwa_testfile.uvfits")
    uvd.write_uvfits(testfile)

    uvd2 = UVData()
    uvd2.read(testfile)

    latitude, longitude, altitude = uvd2.telescope_location_lat_lon_alt_degrees
    unique_times, inverse_inds = np.unique(uvd2.time_array, return_inverse=True)
    unique_lst_array = uvutils.get_lst_for_time(
        unique_times,
        latitude,
        longitude,
        altitude,
    )

    calc_lst_array = unique_lst_array[inverse_inds]

    assert np.allclose(
        calc_lst_array,
        uvd2.lst_array,
        rtol=uvd2._lst_array.tols[0],
        atol=uvd2._lst_array.tols[1],
    )

    assert uvd2.__eq__(
        uvd,
        allowed_failures=[
            "filename",
            "scan_number_array",
            "dut1",
            "earth_omega",
            "gst0",
            "rdate",
            "timesys",
        ],
    )


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_break_read_uvfits():
    """Test errors on reading in a uvfits file with subarrays and other problems."""
    uvobj = UVData()
    multi_subarray_file = os.path.join(DATA_PATH, "multi_subarray.uvfits")
    with pytest.raises(ValueError, match="This file appears to have multiple subarray"):
        uvobj.read(multi_subarray_file)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:UVFITS file is missing AIPS SU table")
def test_source_group_params(casa_uvfits, tmp_path):
    # make a file with a single source to test that it works
    uv_in = casa_uvfits
    # Writing a source table to UVFITS makes pyuvdata think that the data are
    # mutli-phase-ctr, so we'll force that in the original file as well
    write_file = os.path.join(tmp_path, "outtest_casa.uvfits")
    write_file2 = os.path.join(tmp_path, "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = vis_hdu.data.parnames
        group_parameter_list = []

        time_ind = 0
        lst_ind = 0
        for index, name in enumerate(par_names):
            par_value = vis_hdu.data.par(name)
            # time_array and lst_array need to be split in 2 parts to get high enough
            # accuracy
            if name.lower() == "date":
                if time_ind == 0:
                    # first time entry, par_value has full time value
                    # (astropy adds the 2 values)
                    time_array1 = np.float32(par_value)
                    time_array2 = np.float32(par_value - np.float64(time_array1))
                    par_value = time_array1
                    time_ind = 1
                else:
                    par_value = time_array2
            elif name.lower() == "lst":
                if lst_ind == 0:
                    # first lst entry, par_value has full lst value
                    # (astropy adds the 2 values)
                    lst_array_1 = np.float32(par_value)
                    lst_array_2 = np.float32(par_value - np.float64(lst_array_1))
                    par_value = lst_array_1
                    lst_ind = 1
                else:
                    par_value = lst_array_2

            # need to account for PZERO values
            group_parameter_list.append(par_value - vis_hdr["PZERO" + str(index + 1)])

        par_names.append("SOURCE")
        source_array = np.ones_like(vis_hdu.data.par("BASELINE"))
        group_parameter_list.append(source_array)

        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    uv_out = UVData()
    uv_out.read(write_file2)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa2.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_source_frame_defaults(casa_uvfits, tmp_path):
    # make a file with a single source to test that it works
    uv_in = casa_uvfits
    # Writing a source table to UVFITS makes pyuvdata think that the data are
    # mutli-phase-ctr, so we'll force that in the original file as well
    write_file = os.path.join(tmp_path, "outtest_casa.uvfits")
    write_file2 = os.path.join(tmp_path, "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = vis_hdu.data.parnames
        group_parameter_list = []

        time_ind = 0
        lst_ind = 0
        for index, name in enumerate(par_names):
            par_value = vis_hdu.data.par(name)
            # lst_array needs to be split in 2 parts to get high enough accuracy
            # time_array and lst_array need to be split in 2 parts to get high enough
            # accuracy
            if name.lower() == "date":
                if time_ind == 0:
                    # first time entry, par_value has full time value
                    # (astropy adds the 2 values)
                    time_array1 = np.float32(par_value)
                    time_array2 = np.float32(par_value - np.float64(time_array1))
                    par_value = time_array1
                    time_ind = 1
                else:
                    par_value = time_array2
            elif name.lower() == "lst":
                if lst_ind == 0:
                    # first lst entry, par_value has full lst value
                    # (astropy adds the 2 values)
                    lst_array_1 = np.float32(par_value)
                    lst_array_2 = np.float32(par_value - np.float64(lst_array_1))
                    par_value = lst_array_1
                    lst_ind = 1
                else:
                    par_value = lst_array_2

            # need to account for PZERO values
            group_parameter_list.append(par_value - vis_hdr["PZERO" + str(index + 1)])

        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        del vis_hdu.header["RADESYS"]
        del vis_hdu.header["EPOCH"]
        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    uv_out = UVData()
    uv_out.read(write_file2)
    assert uv_out.phase_center_frame == "icrs"


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_missing_aips_su_table(casa_uvfits, tmp_path):
    # make a file with multiple sources to test error condition
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = vis_hdu.data.parnames
        group_parameter_list = []

        time_ind = 0
        lst_ind = 0
        for index, name in enumerate(par_names):
            par_value = vis_hdu.data.par(name)
            # time_array and lst_array need to be split in 2 parts to get high enough
            # accuracy
            if name.lower() == "date":
                if time_ind == 0:
                    # first time entry, par_value has full time value
                    # (astropy adds the 2 values)
                    time_array1 = np.float32(par_value)
                    time_array2 = np.float32(par_value - np.float64(time_array1))
                    par_value = time_array1
                    time_ind = 1
                else:
                    par_value = time_array2
            elif name.lower() == "lst":
                if lst_ind == 0:
                    # first lst entry, par_value has full lst value
                    # (astropy adds the 2 values)
                    lst_array_1 = np.float32(par_value)
                    lst_array_2 = np.float32(par_value - np.float64(lst_array_1))
                    par_value = lst_array_1
                    lst_ind = 1
                else:
                    par_value = lst_array_2

            # need to account for PZERO values
            group_parameter_list.append(par_value - vis_hdr["PZERO" + str(index + 1)])

        par_names.append("SOURCE")
        source_array = np.ones_like(vis_hdu.data.par("BASELINE"))
        mid_index = source_array.shape[0] // 2
        source_array[mid_index:] = source_array[mid_index:] * 2
        group_parameter_list.append(source_array)

        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    with uvtest.check_warnings(
        UserWarning,
        [
            "UVFITS file is missing AIPS SU table, which is required when ",
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_in.read(write_file2)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_multispw_supported():
    """Test reading in a uvfits file with multiple spws."""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1scan.uvfits")
    uvobj.read(testfile)

    # We know this file has two spws
    assert uvobj.Nspws == 2

    # Verify that the data array has the right shape
    assert np.size(uvobj.data_array, axis=1) == 1
    assert np.size(uvobj.data_array, axis=2) == uvobj.Nfreqs

    # Verify that the freq array has the right shape
    assert np.size(uvobj.freq_array, axis=0) == 1
    assert np.size(uvobj.freq_array, axis=1) == uvobj.Nfreqs

    # Verift thaat the spw_array is the right length
    assert len(uvobj.spw_array) == uvobj.Nspws


def test_casa_nonascii_bytes_antenna_names():
    """Test that nonascii bytes in antenna names are handled properly."""
    uv1 = UVData()
    testfile = os.path.join(DATA_PATH, "corrected2_zen.2458106.28114.ant012.HH.uvfits")
    # this file has issues with the telescope location so turn checking off
    with uvtest.check_warnings(
        UserWarning, "Telescope mock-HERA is not in known_telescopes."
    ):
        uv1.read(testfile, run_check=False)
    # fmt: off
    expected_ant_names = [
        'HH0', 'HH1', 'HH2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2',
        'H2', 'HH11', 'HH12', 'HH13', 'HH14', 'H14', 'H14', 'H14', 'H14',
        'H14', 'H14', 'H14', 'H14', 'HH23', 'HH24', 'HH25', 'HH26', 'HH27',
        'H27', 'H27', 'H27', 'H27', 'H27', 'H27', 'H27', 'H27', 'HH36',
        'HH37', 'HH38', 'HH39', 'HH40', 'HH41', 'H41', 'H41', 'H41', 'H41',
        'H41', 'H41', 'H41', 'H41', 'HH50', 'HH51', 'HH52', 'HH53', 'HH54',
        'HH55', 'H55', 'H55', 'H55', 'H55', 'H55', 'H55', 'H55', 'H55',
        'H55', 'HH65', 'HH66', 'HH67', 'HH68', 'HH69', 'HH70', 'HH71',
        'H71', 'H71', 'H71', 'H71', 'H71', 'H71', 'H71', 'H71', 'H71',
        'H71', 'HH82', 'HH83', 'HH84', 'HH85', 'HH86', 'HH87', 'HH88',
        'H88', 'H88', 'H88', 'H88', 'H88', 'H88', 'H88', 'H88', 'H88',
        'HH98', 'H98', 'H98', 'H98', 'H98', 'H98', 'H98', 'H98', 'H98',
        'H98', 'H98', 'H98', 'H98', 'H98', 'H98', 'H98', 'H98', 'H98',
        'H98', 'H98', 'H98', 'H98', 'HH120', 'HH121', 'HH122', 'HH123',
        'HH124', 'H124', 'H124', 'H124', 'H124', 'H124', 'H124', 'H124',
        'H124', 'H124', 'H124', 'H124', 'HH136', 'HH137', 'HH138', 'HH139',
        'HH140', 'HH141', 'HH142', 'HH143']
    # fmt: on
    assert uv1.antenna_names == expected_ant_names


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_readwriteread(tmp_path, casa_uvfits, future_shapes):
    """
    CASA tutorial uvfits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    uv_in = casa_uvfits

    if future_shapes:
        uv_in.use_future_array_shapes()

    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)
    if future_shapes:
        uv_out.use_future_array_shapes()

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.parametrize("uvw_suffix", ["---SIN", "---NCP"])
def test_uvw_coordinate_suffixes(casa_uvfits, tmp_path, uvw_suffix):
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = vis_hdu.data.parnames
        group_parameter_list = []

        time_ind = 0
        lst_ind = 0
        for index, name in enumerate(par_names):
            par_value = vis_hdu.data.par(name)
            # time_array and lst_array need to be split in 2 parts to get high enough
            # accuracy
            if name.lower() == "date":
                if time_ind == 0:
                    # first time entry, par_value has full time value
                    # (astropy adds the 2 values)
                    time_array1 = np.float32(par_value)
                    time_array2 = np.float32(par_value - np.float64(time_array1))
                    par_value = time_array1
                    time_ind = 1
                else:
                    par_value = time_array2
            elif name.lower() == "lst":
                if lst_ind == 0:
                    # first lst entry, par_value has full lst value
                    # (astropy adds the 2 values)
                    lst_array_1 = np.float32(par_value)
                    lst_array_2 = np.float32(par_value - np.float64(lst_array_1))
                    par_value = lst_array_1
                    lst_ind = 1
                else:
                    par_value = lst_array_2

            # need to account for PZERO values
            group_parameter_list.append(par_value - vis_hdr["PZERO" + str(index + 1)])

        for name in ["UU", "VV", "WW"]:
            par_names[par_names.index(name)] = name + uvw_suffix
        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    if uvw_suffix == "---NCP":
        with uvtest.check_warnings(
            UserWarning,
            match=[
                "Telescope EVLA is not in known_telescopes.",
                "The baseline coordinates (uvws) in this file are specified in the "
                "---NCP coordinate system",
                "The uvw_array does not match the expected values",
            ],
        ):
            uv2 = UVData.from_file(write_file2)
        uv2.uvw_array = uvutils._rotate_one_axis(
            uv2.uvw_array[:, :, None], -1 * (uv2.phase_center_app_dec - np.pi / 2), 0
        )[:, :, 0]
    else:
        uv2 = UVData.from_file(write_file2)

    assert uv2 == uv_in


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.parametrize(
    "uvw_suffix", [["---SIN", "", ""], ["", "---NCP", ""], ["", "---NCP", "---SIN"]]
)
def test_uvw_coordinate_suffix_errors(casa_uvfits, tmp_path, uvw_suffix):
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = vis_hdu.data.parnames
        group_parameter_list = []

        time_ind = 0
        lst_ind = 0
        for index, name in enumerate(par_names):
            par_value = vis_hdu.data.par(name)
            # time_array and lst_array need to be split in 2 parts to get high enough
            # accuracy
            if name.lower() == "date":
                if time_ind == 0:
                    # first time entry, par_value has full time value
                    # (astropy adds the 2 values)
                    time_array1 = np.float32(par_value)
                    time_array2 = np.float32(par_value - np.float64(time_array1))
                    par_value = time_array1
                    time_ind = 1
                else:
                    par_value = time_array2
            elif name.lower() == "lst":
                if lst_ind == 0:
                    # first lst entry, par_value has full lst value
                    # (astropy adds the 2 values)
                    lst_array_1 = np.float32(par_value)
                    lst_array_2 = np.float32(par_value - np.float64(lst_array_1))
                    par_value = lst_array_1
                    lst_ind = 1
                else:
                    par_value = lst_array_2

            # need to account for PZERO values
            group_parameter_list.append(par_value - vis_hdr["PZERO" + str(index + 1)])

        for ind, name in enumerate(["UU", "VV", "WW"]):
            par_names[par_names.index(name)] = name + uvw_suffix[ind]
        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    with pytest.raises(
        ValueError,
        match="There is no consistent set of baseline coordinates in this file.",
    ):
        UVData.from_file(write_file2)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_no_lst(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # test that it works with write_lst = False
    uv_in.write_uvfits(write_file, write_lst=False)
    uv_out.read(write_file)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_x_orientation(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that if x_orientation is set, it's read back out properly
    uv_in.x_orientation = "east"
    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_antenna_diameters(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that if antenna_diameters is set, it's read back out properly
    uv_in.antenna_diameters = (
        np.zeros((uv_in.Nants_telescope,), dtype=np.float64) + 14.0
    )
    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_large_antnums(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that if antenna_numbers are > 256 everything works
    uv_in.antenna_numbers = uv_in.antenna_numbers + 256
    uv_in.ant_1_array = uv_in.ant_1_array + 256
    uv_in.ant_2_array = uv_in.ant_2_array + 256
    uv_in.baseline_array = uv_in.antnums_to_baseline(
        uv_in.ant_1_array, uv_in.ant_2_array
    )
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions",
            "Found antenna numbers > 255 in this data set. This is permitted by "
            "UVFITS ",
            "antnums_to_baseline: found antenna numbers > 255, using 2048 baseline",
        ],
    ):
        uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_missing_info(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")

    # check missing telescope_name, timesys vs timsys spelling, xyz_telescope_frame=????
    uv_in.write_uvfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()

        vis_hdr.pop("TELESCOP")

        vis_hdu.header = vis_hdr

        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        ant_hdr = ant_hdu.header.copy()

        time_sys = ant_hdr.pop("TIMESYS")
        ant_hdr["TIMSYS"] = time_sys
        ant_hdr["FRAME"] = "????"

        ant_hdu.header = ant_hdr

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)

    uv_out.read(write_file2)
    assert uv_out.telescope_name == "EVLA"
    assert uv_out.timesys == time_sys

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_error_timesys(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check error if timesys is 'IAT'
    uv_in.timesys = "IAT"
    with pytest.raises(ValueError) as cm:
        uv_in.write_uvfits(write_file)
    assert str(cm.value).startswith(
        "This file has a time system IAT. " 'Only "UTC" time system files are supported'
    )

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_error_single_time(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")

    # check error if one time & no inttime specified
    uv_singlet = uv_in.select(times=uv_in.time_array[0], inplace=False)
    uv_singlet.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data

        par_names = np.array(vis_hdu.data.parnames)
        pars_use = np.where(par_names != "INTTIM")[0]
        par_names = par_names[pars_use].tolist()

        group_parameter_list = [vis_hdu.data.par(name) for name in par_names]

        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr

        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)

    with pytest.raises(
        ValueError, match="Required UVParameter _integration_time has not been set"
    ):
        with uvtest.check_warnings(
            [
                UserWarning,
                erfa.core.ErfaWarning,
                erfa.core.ErfaWarning,
                UserWarning,
                UserWarning,
            ],
            [
                "Telescope EVLA is not",
                "ERFA function 'utcut1' yielded 1 of 'dubious year (Note 3)'",
                "ERFA function 'utctai' yielded 1 of 'dubious year (Note 3)'",
                "LST values stored in this file are not self-consistent",
                "The integration time is not specified and only one time",
            ],
        ):
            uv_out.read(write_file2),

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_unflagged_data_warnings(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that unflagged data with nsample = 0 will cause warnings
    uv_in.nsample_array[list(range(11, 22))] = 0
    uv_in.flag_array[list(range(11, 22))] = False
    with uvtest.check_warnings(
        UserWarning,
        [
            "The uvw_array does not match the expected values given the antenna "
            "positions",
            "Some unflagged data has nsample = 0",
        ],
    ):
        uv_in.write_uvfits(write_file)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.parametrize(
    "kwd_name,kwd_value,warnstr,errstr",
    (
        [
            "testdict",
            {"testkey": 23},
            "testdict in extra_keywords is a list, array or dict",
            "Extra keyword testdict is of <class 'dict'>",
        ],
        [
            "testlist",
            [12, 14, 90],
            "testlist in extra_keywords is a list, array or dict",
            "Extra keyword testlist is of <class 'list'>",
        ],
        [
            "testarr",
            np.array([12, 14, 90]),
            "testarr in extra_keywords is a list, array or dict",
            "Extra keyword testarr is of <class 'numpy.ndarray'>",
        ],
        [
            "test_long_key",
            True,
            "key test_long_key in extra_keywords is longer than 8 characters",
            None,
        ],
    ),
)
def test_extra_keywords_errors(
    casa_uvfits, tmp_path, kwd_name, kwd_value, warnstr, errstr
):
    uv_in = casa_uvfits
    testfile = str(tmp_path / "outtest_casa.uvfits")

    uvw_warn_str = "The uvw_array does not match the expected values"
    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    uv_in.extra_keywords[kwd_name] = kwd_value
    if warnstr is None:
        warnstr_list = [uvw_warn_str]
    else:
        warnstr_list = [warnstr, uvw_warn_str]

    with uvtest.check_warnings(UserWarning, match=warnstr_list):
        uv_in.check()

    if errstr is not None:
        with pytest.raises(TypeError, match=errstr):
            uv_in.write_uvfits(testfile, run_check=False)
    else:
        with uvtest.check_warnings(UserWarning, match=warnstr):
            uv_in.write_uvfits(testfile, run_check=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.parametrize(
    "kwd_names,kwd_values",
    (
        [["bool", "bool2"], [True, False]],
        [["int1", "int2"], [np.int64(5), 7]],
        [["float1", "float2"], [np.int64(5.3), 6.9]],
        [["complex1", "complex2"], [np.complex64(5.3 + 1.2j), 6.9 + 4.6j]],
        [
            ["str", "comment"],
            [
                "hello",
                "this is a very long comment that will be broken into several "
                "lines\nif everything works properly.",
            ],
        ],
    ),
)
def test_extra_keywords(casa_uvfits, tmp_path, kwd_names, kwd_values):
    uv_in = casa_uvfits
    uv_out = UVData()
    testfile = str(tmp_path / "outtest_casa.uvfits")

    for name, value in zip(kwd_names, kwd_values):
        uv_in.extra_keywords[name] = value
    uv_in.write_uvfits(testfile)
    uv_out.read(testfile)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.parametrize("order", ["time", "bda"])
def test_roundtrip_blt_order(casa_uvfits, order, tmp_path):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "blt_order_test.uvfits")

    uv_in.reorder_blts(order)

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["blt_order_test.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"antenna_nums": np.array([1, 20, 12, 25, 4, 24, 2, 21, 22])},
        {"freq_chans": np.arange(12, 22)},
        {"freq_chans": [0]},
        {"polarizations": [-1, -2]},
        {"time_inds": np.array([0, 1])},
        {"lst_inds": np.array([0, 1])},
        {
            "antenna_nums": np.array([1, 20, 12, 25, 4, 24, 2, 21, 22]),
            "freq_chans": np.arange(12, 22),
            "polarizations": [-1, -2],
        },
        {
            "antenna_nums": np.array([1, 2]),
            "freq_chans": np.arange(12, 22),
            "polarizations": [-1, -2],
        },
        {
            "antenna_nums": np.array([1, 2, 3, 4, 7, 8, 9, 12, 15, 19, 20, 21, 22, 23]),
            "freq_chans": np.arange(12, 64),
            "polarizations": [-1, -2],
        },
    ],
)
def test_select_read(casa_uvfits, tmp_path, select_kwargs):
    uvfits_uv = UVData()
    uvfits_uv2 = UVData()

    uvfits_uv2 = casa_uvfits
    if "time_inds" in select_kwargs.keys():
        time_inds = select_kwargs.pop("time_inds")
        unique_times = np.unique(uvfits_uv2.time_array)
        select_kwargs["time_range"] = unique_times[time_inds]

    if "lst_inds" in select_kwargs.keys():
        lst_inds = select_kwargs.pop("lst_inds")
        unique_lsts = np.unique(uvfits_uv2.lst_array)
        select_kwargs["lst_range"] = unique_lsts[lst_inds]

    uvfits_uv.read(casa_tutorial_uvfits, **select_kwargs)
    uvfits_uv2.select(**select_kwargs)
    assert uvfits_uv == uvfits_uv2

    testfile = str(tmp_path / "outtest_casa.uvfits")
    uvfits_uv.write_uvfits(testfile)
    uvfits_uv2.read(testfile)

    # make sure filenames are what we expect
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uvfits_uv2.filename == ["outtest_casa.uvfits"]
    uvfits_uv.filename = uvfits_uv2.filename

    assert uvfits_uv == uvfits_uv2


@pytest.mark.filterwarnings("ignore:Required Antenna keyword 'FRAME'")
@pytest.mark.filterwarnings("ignore:telescope_location is not set")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "select_kwargs",
    [{"antenna_nums": np.array([2, 4, 5])}, {"freq_chans": np.arange(4, 8)}],
)
def test_select_read_nospw(uvfits_nospw, tmp_path, select_kwargs):
    uvfits_uv2 = uvfits_nospw

    uvfits_uv = UVData()
    # This file has a crazy epoch (2291.34057617) which breaks the uvw_antpos check
    # Since it's a PAPER file, I think this is a bug in the file, not in the check.
    uvfits_uv.read(paper_uvfits, run_check_acceptability=False, **select_kwargs)

    uvfits_uv2.select(run_check_acceptability=False, **select_kwargs)
    assert uvfits_uv == uvfits_uv2


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_read_nospw_pol(casa_uvfits, tmp_path):
    # this requires writing a new file because the no spw file we have has only 1 pol

    with fits.open(casa_tutorial_uvfits, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data
        raw_data_array = raw_data_array[:, :, :, 0, :, :, :]

        vis_hdr["NAXIS"] = 6

        vis_hdr["NAXIS5"] = vis_hdr["NAXIS6"]
        vis_hdr["CTYPE5"] = vis_hdr["CTYPE6"]
        vis_hdr["CRVAL5"] = vis_hdr["CRVAL6"]
        vis_hdr["CDELT5"] = vis_hdr["CDELT6"]
        vis_hdr["CRPIX5"] = vis_hdr["CRPIX6"]
        vis_hdr["CROTA5"] = vis_hdr["CROTA6"]

        vis_hdr["NAXIS6"] = vis_hdr["NAXIS7"]
        vis_hdr["CTYPE6"] = vis_hdr["CTYPE7"]
        vis_hdr["CRVAL6"] = vis_hdr["CRVAL7"]
        vis_hdr["CDELT6"] = vis_hdr["CDELT7"]
        vis_hdr["CRPIX6"] = vis_hdr["CRPIX7"]
        vis_hdr["CROTA6"] = vis_hdr["CROTA7"]

        vis_hdr.pop("NAXIS7")
        vis_hdr.pop("CTYPE7")
        vis_hdr.pop("CRVAL7")
        vis_hdr.pop("CDELT7")
        vis_hdr.pop("CRPIX7")
        vis_hdr.pop("CROTA7")

        par_names = vis_hdu.data.parnames

        group_parameter_list = [vis_hdu.data.par(ind) for ind in range(len(par_names))]

        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr

        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        write_file = str(tmp_path / "outtest_casa.uvfits")
        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)

    pols_to_keep = [-1, -2]
    uvfits_uv = UVData()
    uvfits_uv.read(write_file, polarizations=pols_to_keep)
    uvfits_uv2 = casa_uvfits
    uvfits_uv2.select(polarizations=pols_to_keep)

    # make sure filenames are what we expect
    assert uvfits_uv.filename == ["outtest_casa.uvfits"]
    assert uvfits_uv2.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uvfits_uv.filename = uvfits_uv2.filename

    assert uvfits_uv == uvfits_uv2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_read_uvfits_write_miriad(casa_uvfits, tmp_path):
    """
    read uvfits, write miriad test.
    Read in uvfits file, write out as miriad, read back in and check for
    object equality.
    """
    pytest.importorskip("pyuvdata._miriad")
    uvfits_uv = casa_uvfits
    miriad_uv = UVData()
    testfile = str(tmp_path / "outtest_miriad")
    uvfits_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read_miriad(testfile)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    miriad_uv.filename = uvfits_uv.filename

    assert miriad_uv == uvfits_uv

    # check that setting the phase_type keyword also works
    miriad_uv.read_miriad(testfile, phase_type="phased")

    # check that setting the phase_type to drift raises an error
    with pytest.raises(
        ValueError, match='phase_type is "drift" but the RA values are constant.'
    ):
        miriad_uv.read_miriad(testfile, phase_type="drift")

    # check that setting it works after selecting a single time
    uvfits_uv.select(times=uvfits_uv.time_array[0])
    uvfits_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read_miriad(testfile)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    miriad_uv.filename = uvfits_uv.filename

    assert miriad_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multi_files(casa_uvfits, tmp_path):
    """
    Reading multiple files at once.
    """
    uv_full = casa_uvfits
    testfile1 = str(tmp_path / "uv1.uvfits")
    testfile2 = str(tmp_path / "uv2.uvfits")
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvfits(testfile1)
    uv2.write_uvfits(testfile2)
    uv1.read(np.array([testfile1, testfile2]), file_type="uvfits")

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history

    # make sure filenames are what we expect
    assert set(uv1.filename) == {"uv1.uvfits", "uv2.uvfits"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv1.filename = uv_full.filename
    uv1._filename.form = (1,)

    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multi_files_axis(casa_uvfits, tmp_path):
    """
    Reading multiple files at once using "axis" keyword.
    """
    uv_full = casa_uvfits
    testfile1 = str(tmp_path / "uv1.uvfits")
    testfile2 = str(tmp_path / "uv2.uvfits")
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvfits(testfile1)
    uv2.write_uvfits(testfile2)

    uv1.read([testfile1, testfile2], axis="freq")
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history

    # make sure filenames are what we expect
    assert set(uv1.filename) == {"uv1.uvfits", "uv2.uvfits"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv1.filename = uv_full.filename
    uv1._filename.form = (1,)

    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multi_files_metadata_only(casa_uvfits, tmp_path):
    """
    Reading multiple files at once with metadata only.
    """
    uv_full = casa_uvfits
    testfile1 = str(tmp_path / "uv1.uvfits")
    testfile2 = str(tmp_path / "uv2.uvfits")
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvfits(testfile1)
    uv2.write_uvfits(testfile2)

    # check with metadata_only
    uv_full = uv_full.copy(metadata_only=True)
    uv1 = UVData()
    uv1.read([testfile1, testfile2], read_data=False)

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv1.history = uv_full.history

    # make sure filenames are what we expect
    assert set(uv1.filename) == {"uv1.uvfits", "uv2.uvfits"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv1.filename = uv_full.filename
    uv1._filename.form = (1,)

    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
def test_multi_unphase_on_read(casa_uvfits, tmp_path):
    uv_full = casa_uvfits
    uv_full2 = UVData()
    testfile1 = str(tmp_path / "uv1.uvfits")
    testfile2 = str(tmp_path / "uv2.uvfits")
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvfits(testfile1)
    uv2.write_uvfits(testfile2)
    with uvtest.check_warnings(
        UserWarning,
        [
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "Unphasing this UVData object to drift",
            "Unphasing other UVData object to drift",
        ],
    ):
        uv1.read(np.array([testfile1, testfile2]), unphase_to_drift=True)

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv_full.unphase_to_drift()

    uv1.history = uv_full.history

    # make sure filenames are what we expect
    assert set(uv1.filename) == {"uv1.uvfits", "uv2.uvfits"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv1.filename = uv_full.filename
    uv1._filename.form = (1,)

    assert uv1 == uv_full

    # check unphasing when reading only one file
    with uvtest.check_warnings(
        UserWarning,
        [
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "Unphasing this UVData object to drift",
        ],
    ):
        uv_full2.read(casa_tutorial_uvfits, unphase_to_drift=True)
    assert uv_full2 == uv_full


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The xyz array in ENU_from_ECEF")
@pytest.mark.filterwarnings("ignore:The enu array in ECEF_from_ENU")
def test_multi_phase_on_read(casa_uvfits, tmp_path):
    uv_full = casa_uvfits
    uv_full2 = UVData()
    testfile1 = str(tmp_path / "uv1.uvfits")
    testfile2 = str(tmp_path / "uv2.uvfits")
    phase_center_radec = [
        uv_full.phase_center_ra + 0.01,
        uv_full.phase_center_dec + 0.01,
    ]
    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_uvfits(testfile1)
    uv2.write_uvfits(testfile2)
    with uvtest.check_warnings(
        UserWarning,
        [
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "Phasing this UVData object to phase_center_radec",
            "The uvw_array does not match the expected values given the "
            "antenna positions.",
            "Phasing this UVData object to phase_center_radec",
        ],
    ):
        uv1.read(
            np.array([testfile1, testfile2]),
            phase_center_radec=phase_center_radec,
        )

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis "
        "using pyuvdata.",
        uv1.history,
    )

    uv_full.phase(*phase_center_radec)
    uv1.history = uv_full.history

    # make sure filenames are what we expect
    assert set(uv1.filename) == {"uv1.uvfits", "uv2.uvfits"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv1.filename = uv_full.filename
    uv1._filename.form = (1,)

    assert uv1 == uv_full

    # check phasing when reading only one file
    with uvtest.check_warnings(
        UserWarning,
        [
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values given the antenna "
            "positions.",
            "Phasing this UVData object to phase_center_radec",
        ],
    ):
        uv_full2.read(casa_tutorial_uvfits, phase_center_radec=phase_center_radec)
    assert uv_full2 == uv_full

    with pytest.raises(ValueError) as cm:
        uv_full2.read(casa_tutorial_uvfits, phase_center_radec=phase_center_radec[0])
    assert str(cm.value).startswith("phase_center_radec should have length 2.")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_read_ms_write_uvfits_casa_history(tmp_path):
    """
    read in .ms file.
    Write to a uvfits file, read back in and check for casa_history parameter
    """
    pytest.importorskip("casacore")
    ms_uv = UVData()
    uvfits_uv = UVData()
    ms_file = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    testfile = str(tmp_path / "outtest.uvfits")
    ms_uv.read_ms(ms_file)
    ms_uv.write_uvfits(testfile)
    uvfits_uv.read(testfile)

    # make sure filenames are what we expect
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    assert uvfits_uv.filename == ["outtest.uvfits"]
    ms_uv.filename = uvfits_uv.filename

    # propagate scan numbers to the uvfits, ONLY for comparison
    uvfits_uv.scan_number_array = ms_uv.scan_number_array

    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific paramters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(ms_uv, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    assert ms_uv == uvfits_uv


def test_cotter_telescope_frame(tmp_path):
    file1 = os.path.join(DATA_PATH, "1061316296.uvfits")
    write_file = os.path.join(tmp_path, "emulate_cotter.uvfits")
    uvd1 = UVData()

    with fits.open(file1, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        ant_hdu.header.pop("FRAME")

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)
        hdulist.close()

    with uvtest.check_warnings(
        UserWarning,
        ["Required Antenna keyword 'FRAME' not set; Assuming frame is 'ITRF'."],
    ):
        uvd1.read_uvfits(write_file, read_data=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_readwriteread_reorder_pols(tmp_path, casa_uvfits, future_shapes):
    """
    CASA tutorial uvfits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality. We check that on-the-fly polarization reordering works.
    """
    uv_in = casa_uvfits

    if future_shapes:
        uv_in.use_future_array_shapes()

    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # reorder polarizations
    polarization_input = uv_in.polarization_array
    uv_in.reorder_pols(order=[3, 0, 2, 1])
    assert not np.allclose(uv_in.polarization_array, polarization_input)

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)
    if future_shapes:
        uv_out.use_future_array_shapes()

    # put polarizations back in order
    uv_in.reorder_pols(order="AIPS")

    # make sure filename is what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out


@pytest.mark.parametrize(
    "freq_val,chan_val,msg",
    [
        [-1, 1, "Frequency values must be > 0 for UVFITS!"],
        [1, 0, "Something is wrong, frequency values not"],
    ],
)
def test_flex_spw_uvfits_write_errs(sma_mir, freq_val, chan_val, msg):
    sma_mir.freq_array[:] = freq_val
    sma_mir.channel_width[:] = chan_val
    with pytest.raises(ValueError, match=msg):
        sma_mir.write_uvfits("dummy")


def test_mwax_birli_frame(tmp_path):
    fits_file = os.path.join(DATA_PATH, "1061316296.uvfits")
    outfile = tmp_path / "mwax_birli.uvfits"
    with fits.open(fits_file, memmap=True) as hdu_list:
        hdu_list[0].header["SOFTWARE"] = "birli"
        # remove the frame keyword
        del hdu_list[1].header["FRAME"]
        hdu_list.writeto(outfile)
    with uvtest.check_warnings(
        UserWarning,
        ["Required Antenna keyword 'FRAME' not set; Assuming frame is 'ITRF'."],
    ):
        UVData.from_file(outfile, read_data=False)


def test_mwax_missing_frame_comment(tmp_path):
    fits_file = os.path.join(DATA_PATH, "1061316296.uvfits")
    outfile = tmp_path / "mwax_birli.uvfits"
    with fits.open(fits_file, memmap=True) as hdu_list:
        del hdu_list[1].header["FRAME"], hdu_list[0].header["COMMENT"]
        hdu_list[0].header["COMMENT"] = "A dummy comment."
        hdu_list.writeto(outfile)
    with uvtest.check_warnings(
        UserWarning,
        ["Required Antenna keyword 'FRAME' not set; Assuming frame is 'ITRF'."],
    ):
        UVData.from_file(outfile, read_data=False)


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.parametrize("spoof", [False, True])
def test_no_spoof(sma_mir, tmp_path, spoof):
    """
    Verify that option UVParameters are correctly generated when writing to a UVFITS
    file if not previously set, and that spoof_nonessential throws the correct warning
    message if used.
    """
    sma_uvfits = UVData()
    sma_mir._set_app_coords_helper()
    filename = os.path.join(tmp_path, "spoof.uvfits" if spoof else "no_spoof.uvfits")

    with uvtest.check_warnings(
        DeprecationWarning if spoof else None,
        "UVFITS-required metadata are now set automatically to " if spoof else None,
    ):
        sma_mir.write_uvfits(filename, spoof_nonessential=spoof)

    sma_uvfits = UVData.from_file(filename)

    # UVFITS has some differences w/ the MIR format that are expected -- handle
    # all of that here, making sure that the returned values are consistent with
    # what we expect. Start w/ spectral windows
    assert len(np.unique(sma_mir.spw_array)) == len(np.unique(sma_uvfits.spw_array))

    spw_dict = {idx: jdx for idx, jdx in zip(sma_uvfits.spw_array, sma_mir.spw_array)}

    assert np.all(
        [
            idx == spw_dict[jdx]
            for idx, jdx in zip(sma_mir.flex_spw_id_array, sma_uvfits.flex_spw_id_array)
        ]
    )
    sma_uvfits.spw_array = sma_mir.spw_array
    sma_uvfits.flex_spw_id_array = sma_mir.flex_spw_id_array

    # Check the history next
    assert sma_uvfits.history.startswith(sma_mir.history)
    sma_mir.history = sma_uvfits.history

    # We have to do a bit of special handling for the phase_center_catalog, because
    # _very_ small floating point errors can creep in.
    for cat_name in sma_mir.phase_center_catalog.keys():
        this_cat = sma_mir.phase_center_catalog[cat_name]
        other_cat = sma_uvfits.phase_center_catalog[cat_name]
        assert np.isclose(this_cat["cat_lat"], other_cat["cat_lat"])
        assert np.isclose(this_cat["cat_lon"], other_cat["cat_lon"])
    sma_uvfits.phase_center_catalog = sma_mir.phase_center_catalog

    # Finally, move on to the "non-spoofed paramters"
    exp_dict = {
        "dut1": -0.2137079,
        "earth_omega": 360.9856438593,
        "gst0": 122.6673828188983,
        "rdate": "2020-07-24",
        "timesys": "UTC",
    }

    for key, value in exp_dict.items():
        if isinstance(value, str):
            assert value == getattr(sma_uvfits, key)
        else:
            assert np.isclose(getattr(sma_uvfits, key), value)
        assert getattr(sma_mir, key) is None
        setattr(sma_mir, key, value)

    assert sma_uvfits == sma_mir


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_uvfits_phasing_errors(hera_uvh5, tmp_path):
    # check error if phase_type is wrong and force_phase not set
    with pytest.raises(ValueError, match="The data are in drift mode. Set force_phase"):
        hera_uvh5.write_uvfits(tmp_path)
