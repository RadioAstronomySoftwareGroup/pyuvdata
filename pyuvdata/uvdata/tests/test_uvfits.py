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
from pyuvdata.tests.test_utils import frame_selenoid
from pyuvdata.uvdata.uvdata import _future_array_shapes_warning

casa_tutorial_uvfits = os.path.join(
    DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits"
)

paper_uvfits = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAAM.uvfits")


def _fix_uvfits_multi_group_params(vis_hdu):
    par_names = vis_hdu.data.parnames
    group_parameter_list = []

    multi_params = {}
    for name in par_names:
        wh_name = np.nonzero(np.asarray(par_names) == name)[0]
        if len(wh_name) > 1:
            assert len(wh_name) == 2
            multi_params[name] = {"ind": 0}

    for index, name in enumerate(par_names):
        par_value = vis_hdu.data.par(name)

        if name in multi_params.keys():
            # these params need to be split in 2 parts to get high enough accuracy
            # (e.g. time and lst arrays)
            if multi_params[name]["ind"] == 0:
                multi_params[name]["arr1"] = np.float32(par_value)
                multi_params[name]["arr2"] = np.float32(
                    par_value - np.float64(multi_params[name]["arr1"])
                )
                par_value = multi_params[name]["arr1"]
                multi_params[name]["ind"] += 1
            else:
                par_value = multi_params[name]["arr2"]

        # need to account for PSCAL and PZERO values
        group_parameter_list.append(
            par_value / np.float64(vis_hdu.header["PSCAL" + str(index + 1)])
            - vis_hdu.header["PZERO" + str(index + 1)]
        )
    return group_parameter_list


def _make_multi_phase_center(uvobj, init_phase_dict, frame_list):
    init_ra = init_phase_dict["cat_lon"]
    init_dec = init_phase_dict["cat_lat"]
    init_epoch = init_phase_dict["cat_epoch"]
    init_name = init_phase_dict["cat_name"]

    nphase = len(frame_list)
    for frame_ind, frame_use in enumerate(frame_list):
        phase_mask = np.full(uvobj.Nblts, False)
        mask_start = frame_ind * uvobj.Nblts // nphase
        if frame_ind == nphase - 1:
            mask_end = uvobj.Nblts
        else:
            mask_end = (frame_ind + 1) * uvobj.Nblts // nphase
        phase_mask[mask_start:mask_end] = True
        if frame_use == "icrs":
            epoch = None
        elif frame_use == "fk5":
            if frame_ind == 0:
                epoch = init_epoch
            else:
                epoch = init_epoch + 5
        elif frame_use == "fk4":
            epoch = 1950
        if frame_ind == 0:
            cat_name = init_name
        else:
            cat_name = "test_" + str(frame_ind)
        uvobj.phase(
            ra=init_ra + 0.05 * (frame_ind + 1),
            dec=init_dec + 0.05 * (frame_ind + 1),
            phase_frame=frame_use,
            epoch=epoch,
            cat_name=cat_name,
            cat_type="sidereal",
            select_mask=phase_mask,
        )
    assert uvobj.Nphase == nphase


@pytest.fixture(scope="session")
def uvfits_nospw_main():
    uv_in = UVData()
    # This file has a crazy epoch (2291.34057617) which breaks the uvw_antpos check
    # Since it's a PAPER file, I think this is a bug in the file, not in the check.
    uv_in.read(
        paper_uvfits, run_check_acceptability=False, use_future_array_shapes=True
    )

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
    uvobj2.read(casa_tutorial_uvfits, read_data=False, use_future_array_shapes=True)

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
    uvd.read(lwa_file, use_future_array_shapes=True)

    testfile = os.path.join(tmp_path, "lwa_testfile.uvfits")
    uvd.write_uvfits(testfile)

    uvd2 = UVData()
    uvd2.read(testfile, use_future_array_shapes=True)

    latitude, longitude, altitude = uvd2.telescope_location_lat_lon_alt_degrees
    unique_times, inverse_inds = np.unique(uvd2.time_array, return_inverse=True)
    unique_lst_array = uvutils.get_lst_for_time(
        unique_times, latitude, longitude, altitude
    )

    calc_lst_array = unique_lst_array[inverse_inds]

    assert np.allclose(
        calc_lst_array,
        uvd2.lst_array,
        rtol=uvd2._lst_array.tols[0],
        atol=uvd2._lst_array.tols[1],
    )

    # The incoming ra is specified as negative, it gets 2pi added to it in the roundtrip
    uvd2.phase_center_catalog[1]["cat_lon"] -= 2 * np.pi

    uvd2._consolidate_phase_center_catalogs(reference_catalog=uvd.phase_center_catalog)
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
def test_break_read_uvfits(tmp_path):
    """Test errors on reading in a uvfits file with subarrays and other problems."""
    uvobj = UVData()
    multi_subarray_file = os.path.join(DATA_PATH, "multi_subarray.uvfits")
    with pytest.raises(ValueError, match="This file appears to have multiple subarray"):
        uvobj.read(multi_subarray_file)

    file1 = os.path.join(DATA_PATH, "1061316296.uvfits")
    write_file = os.path.join(tmp_path, "bad_frame.uvfits")

    with fits.open(file1, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        ant_hdr = ant_hdu.header.copy()

        ant_hdr["FRAME"] = "FOO"
        ant_hdu.header = ant_hdr

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)
        hdulist.close()

    with pytest.raises(
        ValueError,
        match=(
            "Telescope frame in file is foo. Only 'itrs' and 'mcmf' are currently "
            "supported."
        ),
    ):
        uvobj.read(write_file, read_data=False, use_future_array_shapes=True)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:UVFITS file is missing AIPS SU table")
def test_source_group_params(casa_uvfits, tmp_path):
    # make a file with a single source to test that it works
    uv_in = casa_uvfits
    write_file = os.path.join(tmp_path, "outtest_casa.uvfits")
    write_file2 = os.path.join(tmp_path, "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data
        par_names = vis_hdu.data.parnames

        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

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
    uv_out.read(write_file2, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa2.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not in known_telescopes.")
@pytest.mark.filterwarnings("ignore:The entry name")
@pytest.mark.filterwarnings("ignore:The provided name")
@pytest.mark.parametrize("frame", [["icrs"], ["fk5"], ["fk4"], ["fk5", "icrs"]])
@pytest.mark.parametrize("high_precision", [True, False])
def test_read_write_multi_source(casa_uvfits, tmp_path, frame, high_precision):
    uv_in = casa_uvfits

    # generate a multi source object by phasing it in multiple directions
    init_phase_dict = uv_in.phase_center_catalog[0]
    init_frame = init_phase_dict["cat_frame"]
    frame_list = [init_frame] + frame

    _make_multi_phase_center(uv_in, init_phase_dict, frame_list)

    if high_precision:
        # Increase the precision of the data_array because the uvw_array precision is
        # tied to the data_array precision
        uv_in.data_array = uv_in.data_array.astype(np.complex128)

    write_file = os.path.join(tmp_path, "outtest_multisource.uvfits")
    uv_in.write_uvfits(write_file)
    uv_out = UVData.from_file(write_file, use_future_array_shapes=True)

    if frame == "fk5":
        # objects should just be equal because all the phase centers had the same frames
        uv_out._consolidate_phase_center_catalogs(
            reference_catalog=uv_in.phase_center_catalog
        )
        assert uv_in == uv_out

    if not high_precision:
        # replace the uvw_array because of loss of precision roundtripping through
        # uvfits
        uv_out.uvw_array = uv_in.uvw_array

    # Now rephase to the same places as the initial object was phased to. Note that if
    # frame == ["fk5"] this should not change anything (but it does if uvws lose
    # precision and they aren't fixed before here)
    _make_multi_phase_center(uv_out, init_phase_dict, frame_list)

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The older phase attributes")
@pytest.mark.parametrize("frame", ["icrs", "fk4"])
@pytest.mark.filterwarnings("ignore:UVFITS file is missing AIPS SU table")
def test_source_frame_defaults(casa_uvfits, tmp_path, frame):
    # make a file with a single source to test that it works
    uv_in = casa_uvfits
    write_file = os.path.join(tmp_path, "outtest_casa.uvfits")
    write_file2 = os.path.join(tmp_path, "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data
        par_names = vis_hdu.data.parnames

        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        del vis_hdu.header["RADESYS"]
        if frame == "icrs":
            del vis_hdu.header["EPOCH"]
        elif frame == "fk4":
            vis_hdu.header["EPOCH"] = 1950.0
        ant_hdu = hdu_list[hdunames["AIPS AN"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    uv_out = UVData()
    uv_out.read(write_file2, use_future_array_shapes=True)
    assert uv_out.phase_center_catalog[0]["cat_frame"] == frame


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The entry name")
@pytest.mark.filterwarnings("ignore:The provided name")
@pytest.mark.parametrize("frame_list", [["fk5", "fk5"], ["fk4", "fk4"], ["fk4", "fk5"]])
def test_multi_source_frame_defaults(casa_uvfits, tmp_path, frame_list):
    # make a file with a single source to test that it works
    uv_in = casa_uvfits
    write_file = os.path.join(tmp_path, "outtest_casa.uvfits")
    write_file2 = os.path.join(tmp_path, "outtest_casa2.uvfits")

    init_phase_dict = uv_in.phase_center_catalog[0]
    _make_multi_phase_center(uv_in, init_phase_dict, frame_list)
    uv_in._clear_unused_phase_centers()

    in_frame_list = []
    for phase_dict in uv_in.phase_center_catalog.values():
        in_frame_list.append(phase_dict["cat_frame"])
    assert in_frame_list == frame_list

    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        vis_hdr = vis_hdu.header.copy()
        raw_data_array = vis_hdu.data.data
        par_names = vis_hdu.data.parnames

        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        del vis_hdu.header["RADESYS"]

        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        su_hdu = hdu_list[hdunames["AIPS SU"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu, su_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    uv_out = UVData()
    uv_out.read(write_file2, use_future_array_shapes=True)

    out_frame_list = []
    for phase_dict in uv_out.phase_center_catalog.values():
        out_frame_list.append(phase_dict["cat_frame"])
    assert out_frame_list == frame_list

    if frame_list[0] == frame_list[1]:
        # don't check this for different frames because we cannot write different frames
        # so they won't actually match
        uv_out._consolidate_phase_center_catalogs(
            reference_catalog=uv_in.phase_center_catalog
        )

        assert uv_out == uv_in


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
        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

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
        [UserWarning] * 3 + [DeprecationWarning],
        [
            "UVFITS file is missing AIPS SU table, which is required when ",
            "Telescope EVLA is not",
            "The uvw_array does not match the expected values",
            _future_array_shapes_warning,
        ],
    ):
        uv_in.read(write_file2)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_multispw_supported():
    """Test reading in a uvfits file with multiple spws."""
    uvobj = UVData()
    testfile = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1scan.uvfits")
    uvobj.read(testfile, use_future_array_shapes=True)

    # We know this file has two spws
    assert uvobj.Nspws == 2

    # Verify that the data array has the right shape
    assert np.size(uvobj.data_array, axis=1) == uvobj.Nfreqs

    # Verify that the freq array has the right shape
    assert np.size(uvobj.freq_array) == uvobj.Nfreqs

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
        uv1.read(testfile, run_check=False, use_future_array_shapes=True)
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


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_readwriteread(tmp_path, casa_uvfits, future_shapes, telescope_frame, selenoid):
    """
    CASA tutorial uvfits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    uv_in = casa_uvfits

    if not future_shapes:
        uv_in.use_current_array_shapes()

    if telescope_frame == "mcmf":
        pytest.importorskip("lunarsky")
        enu_antpos, _ = uv_in.get_ENU_antpos()
        latitude, longitude, altitude = uv_in.telescope_location_lat_lon_alt
        uv_in._telescope_location.frame = "mcmf"
        uv_in._telescope_location.ellipsoid = selenoid
        uv_in.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
        new_full_antpos = uvutils.ECEF_from_ENU(
            enu=enu_antpos,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            frame="mcmf",
            ellipsoid=selenoid,
        )
        uv_in.antenna_positions = new_full_antpos - uv_in.telescope_location
        uv_in.set_lsts_from_time_array()
        uv_in.set_uvws_from_antenna_positions()
        uv_in._set_app_coords_helper()
        uv_in.check()

    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file, use_future_array_shapes=future_shapes)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    assert uv_in._telescope_location.frame == uv_out._telescope_location.frame
    assert uv_in._telescope_location.ellipsoid == uv_out._telescope_location.ellipsoid

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
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

        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

        for name in ["UU", "VV", "WW"]:
            name_locs = np.nonzero(np.array(par_names) == name)[0]
            for index in name_locs:
                par_names[index] = name + uvw_suffix
        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        source_hdu = hdu_list[hdunames["AIPS SU"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu, source_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    if uvw_suffix == "---NCP":
        with uvtest.check_warnings(
            UserWarning,
            match=[
                "Telescope EVLA is not in known_telescopes.",
                (
                    "The baseline coordinates (uvws) in this file are specified in the "
                    "---NCP coordinate system"
                ),
                "The uvw_array does not match the expected values",
            ],
        ):
            uv2 = UVData.from_file(write_file2, use_future_array_shapes=True)
        uv2.uvw_array = uvutils._rotate_one_axis(
            uv2.uvw_array[:, :, None], -1 * (uv2.phase_center_app_dec - np.pi / 2), 0
        )[:, :, 0]
    else:
        uv2 = UVData.from_file(write_file2, use_future_array_shapes=True)

    uv2._consolidate_phase_center_catalogs(reference_catalog=uv_in.phase_center_catalog)
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

        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

        for ind, name in enumerate(["UU", "VV", "WW"]):
            name_locs = np.nonzero(np.array(par_names) == name)[0]
            for index in name_locs:
                par_names[index] = name + uvw_suffix[ind]
        vis_hdu = fits.GroupData(
            raw_data_array, parnames=par_names, pardata=group_parameter_list, bitpix=-32
        )
        vis_hdu = fits.GroupsHDU(vis_hdu)
        vis_hdu.header = vis_hdr
        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        source_hdu = hdu_list[hdunames["AIPS SU"]]

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu, source_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    with pytest.raises(
        ValueError,
        match="There is no consistent set of baseline coordinates in this file.",
    ):
        UVData.from_file(write_file2, use_future_array_shapes=True)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
def test_readwriteread_no_lst(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # test that it works with write_lst = False
    uv_in.write_uvfits(write_file, write_lst=False)
    uv_out.read(write_file, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
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
    uv_out.read(write_file, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
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
    uv_out.read(write_file, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
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
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions"
            ),
            (
                "Found antenna numbers > 255 in this data set. This is permitted by "
                "UVFITS "
            ),
            "antnums_to_baseline: found antenna numbers > 255, using 2048 baseline",
        ],
    ):
        uv_in.write_uvfits(write_file)
    uv_out.read(write_file, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.parametrize("lat_lon_alt", [True, False])
def test_readwriteread_missing_info(tmp_path, casa_uvfits, lat_lon_alt):
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

        if not lat_lon_alt:
            vis_hdr.pop("LAT")
            vis_hdr.pop("LON")
            vis_hdr.pop("ALT")

        ant_hdu.header = ant_hdr
        source_hdu = hdu_list[hdunames["AIPS SU"]]

        vis_hdu.header = vis_hdr

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu, source_hdu])
        hdulist.writeto(write_file2, overwrite=True)

    with uvtest.check_warnings(
        UserWarning,
        match=[
            (
                "The telescope frame is set to '????', which generally indicates "
                "ignorance. Defaulting the frame to 'itrs', but this may lead to other "
                "warnings or errors."
            ),
            "Telescope EVLA is not in known_telescopes.",
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv_out.read(write_file2, use_future_array_shapes=True)
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
        'This file has a time system IAT. Only "UTC" time system files are supported'
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
        par_names = vis_hdu.data.parnames

        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

        inttime_ind = np.nonzero(np.asarray(par_names) == "INTTIM")[0]
        assert inttime_ind.size == 1
        inttime_ind = inttime_ind[0]
        del par_names[inttime_ind]
        del group_parameter_list[inttime_ind]

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
            uv_out.read(write_file2, use_future_array_shapes=True)

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
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions"
            ),
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
                (
                    "this is a very long comment that will be broken into several "
                    "lines\nif everything works properly."
                ),
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
    uv_out.read(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
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
    uv_out.read(write_file, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["blt_order_test.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
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

    uvfits_uv.read(casa_tutorial_uvfits, use_future_array_shapes=True, **select_kwargs)
    uvfits_uv2.select(**select_kwargs)
    assert uvfits_uv == uvfits_uv2

    testfile = str(tmp_path / "outtest_casa.uvfits")
    uvfits_uv.write_uvfits(testfile)
    uvfits_uv2.read(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uvfits_uv2.filename == ["outtest_casa.uvfits"]
    uvfits_uv.filename = uvfits_uv2.filename

    uvfits_uv2._consolidate_phase_center_catalogs(
        reference_catalog=uvfits_uv.phase_center_catalog
    )
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
    uvfits_uv.read(
        paper_uvfits,
        run_check_acceptability=False,
        use_future_array_shapes=True,
        **select_kwargs
    )

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

        group_parameter_list = _fix_uvfits_multi_group_params(vis_hdu)

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
    uvfits_uv.read(write_file, polarizations=pols_to_keep, use_future_array_shapes=True)
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
    with uvtest.check_warnings(
        UserWarning,
        [
            (
                "The uvw_array does not match the expected values given the antenna"
                " positions."
            ),
            "writing default values for restfreq, vsource, veldop, jyperk, and systemp",
        ],
    ):
        uvfits_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read_miriad(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    miriad_uv.filename = uvfits_uv.filename

    miriad_uv._consolidate_phase_center_catalogs(
        reference_catalog=uvfits_uv.phase_center_catalog
    )
    assert miriad_uv == uvfits_uv

    # check that setting the projected keyword also works
    miriad_uv.read_miriad(testfile, projected=True, use_future_array_shapes=True)

    # check that setting the projected False raises an error
    with pytest.raises(
        ValueError, match="projected is False but the RA values are constant."
    ):
        miriad_uv.read_miriad(testfile, projected=False, use_future_array_shapes=True)

    # check that setting it works after selecting a single time
    uvfits_uv.select(times=uvfits_uv.time_array[0])
    with uvtest.check_warnings(
        UserWarning,
        [
            (
                "The uvw_array does not match the expected values given the antenna"
                " positions."
            ),
            "writing default values for restfreq, vsource, veldop, jyperk, and systemp",
        ],
    ):
        uvfits_uv.write_miriad(testfile, clobber=True)
    miriad_uv.read_miriad(testfile, use_future_array_shapes=True)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    miriad_uv.filename = uvfits_uv.filename

    miriad_uv._consolidate_phase_center_catalogs(
        reference_catalog=uvfits_uv.phase_center_catalog
    )
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
    uv1.read(
        np.array([testfile1, testfile2]),
        file_type="uvfits",
        use_future_array_shapes=True,
    )

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

    uv1._consolidate_phase_center_catalogs(
        reference_catalog=uv_full.phase_center_catalog
    )
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

    uv1.read([testfile1, testfile2], axis="freq", use_future_array_shapes=True)
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

    uv1._consolidate_phase_center_catalogs(
        reference_catalog=uv_full.phase_center_catalog
    )
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
    uv1.read([testfile1, testfile2], read_data=False, use_future_array_shapes=True)

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

    uv1._consolidate_phase_center_catalogs(
        reference_catalog=uv_full.phase_center_catalog
    )
    assert uv1 == uv_full


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
    ms_uv.read_ms(ms_file, use_future_array_shapes=True)
    ms_uv.write_uvfits(testfile)
    uvfits_uv.read(testfile, use_future_array_shapes=True)

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

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=ms_uv.phase_center_catalog
    )
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
        uvd1.read_uvfits(write_file, read_data=False, use_future_array_shapes=True)


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
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

    if not future_shapes:
        uv_in.use_current_array_shapes()

    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # reorder polarizations
    polarization_input = uv_in.polarization_array
    uv_in.reorder_pols(order=[3, 0, 2, 1])
    assert not np.allclose(uv_in.polarization_array, polarization_input)

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file, use_future_array_shapes=future_shapes)

    # put polarizations back in order
    uv_in.reorder_pols(order="AIPS")

    # make sure filename is what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
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
        UVData.from_file(outfile, read_data=False, use_future_array_shapes=True)


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
        UVData.from_file(outfile, read_data=False, use_future_array_shapes=True)


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
def test_uvfits_extra_params(sma_mir, tmp_path):
    """
    Verify that option UVParameters are correctly generated when writing to a UVFITS
    file if not previously set.
    """
    sma_uvfits = UVData()
    sma_mir._set_app_coords_helper()
    filename = os.path.join(tmp_path, "test.uvfits")

    sma_mir.write_uvfits(filename)

    sma_uvfits = UVData.from_file(filename, use_future_array_shapes=True)

    # UVFITS has some differences w/ the MIR format that are expected -- handle
    # all of that here, making sure that the returned values are consistent with
    # what we expect. Start w/ spectral windows
    assert len(np.unique(sma_mir.spw_array)) == len(np.unique(sma_uvfits.spw_array))

    spw_dict = dict(zip(sma_uvfits.spw_array, sma_mir.spw_array))

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

    # Finally, move on to the uvfits extra parameters
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
    with pytest.raises(
        ValueError, match="The data are not all phased to a sidereal source"
    ):
        hera_uvh5.write_uvfits(tmp_path)


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings(
    "ignore:The shapes of several attributes will be changing "
    "in the future to remove the deprecated spectral window axis."
)
def test_miriad_convention(tmp_path):
    """
    Test writing a MIRIAD-compatible UVFITS file
    """
    uv = UVData()
    uv.read(casa_tutorial_uvfits)

    # Change an antenna ID to 512
    old_idx = uv.antenna_numbers[10]  # This is antenna 19
    new_idx = 512

    uv.antenna_numbers[10] = new_idx
    uv.ant_1_array[uv.ant_1_array == old_idx] = new_idx
    uv.ant_2_array[uv.ant_2_array == old_idx] = new_idx
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array)

    testfile1 = str(tmp_path / "uv1.uvfits")
    uv.write_uvfits(testfile1, use_miriad_convention=True)

    # These are based on known values in casa_tutorial_uvfits
    expected_vals = {"ANTENNA1_0": 4, "ANTENNA2_0": 8, "NOSTA_0": 1}

    # Check baselines match MIRIAD convention
    bl_miriad_expected = uvutils.antnums_to_baseline(
        uv.ant_1_array, uv.ant_2_array, 512, use_miriad_convention=True
    )
    with fits.open(testfile1) as hdu:
        assert np.allclose(hdu[0].data["BASELINE"], bl_miriad_expected)

        # Quick check of other antenna values
        assert hdu[0].data["ANTENNA1"][0] == expected_vals["ANTENNA1_0"]
        assert hdu[0].data["ANTENNA2"][0] == expected_vals["ANTENNA2_0"]
        assert hdu[1].data["NOSTA"][0] == expected_vals["NOSTA_0"]

    uv2 = UVData.from_file(testfile1)
    uv2._update_phase_center_id(1, 0)
    uv2.phase_center_catalog[0]["info_source"] = uv.phase_center_catalog[0][
        "info_source"
    ]

    assert uv2._ant_1_array == uv._ant_1_array
    assert uv2._ant_2_array == uv._ant_2_array

    assert uv2 == uv

    # Test that antennas get +1 if there is a 0-indexed antennas
    old_idx = uv.antenna_numbers[0]
    new_idx = 0
    uv.antenna_numbers[0] = new_idx
    uv.ant_1_array[uv.ant_1_array == old_idx] = new_idx
    uv.ant_2_array[uv.ant_2_array == old_idx] = new_idx
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array)
    uv2 = uv.copy()

    testfile1 = str(tmp_path / "uv2.uvfits")
    uv.write_uvfits(testfile1, use_miriad_convention=True)

    # make sure write_uvfits doesn't change the object
    assert uv2 == uv

    with fits.open(testfile1) as hdu:
        assert hdu[0].data["ANTENNA1"][0] == expected_vals["ANTENNA1_0"] + 1
        assert hdu[0].data["ANTENNA2"][0] == expected_vals["ANTENNA2_0"] + 1
        assert hdu[1].data["NOSTA"][0] == 1  # expected_vals["NOSTA_0"]

    uv2 = UVData.from_file(testfile1)
    uv2._update_phase_center_id(1, 0)
    uv2.phase_center_catalog[0]["info_source"] = uv.phase_center_catalog[0][
        "info_source"
    ]

    # adjust for expected antenna number changes:
    uv2.antenna_numbers -= 1
    uv2.ant_1_array -= 1
    uv2.ant_2_array -= 1
    uv2.baseline_array = uv2.antnums_to_baseline(uv2.ant_1_array, uv2.ant_2_array)

    assert uv2 == uv
