# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for UVFITS object."""

import os

import erfa
import numpy as np
import pytest
from astropy.io import fits

import pyuvdata.utils.io.fits as fits_utils
from pyuvdata import UVData, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

from ..utils.test_coordinates import frame_selenoid

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

        if name in multi_params:
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
    uv_in.read(paper_uvfits, run_check_acceptability=False)

    return uv_in


@pytest.fixture(scope="function")
def uvfits_nospw(uvfits_nospw_main):
    return uvfits_nospw_main.copy()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
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
@pytest.mark.parametrize("uvw_double", [True, False])
def test_group_param_precision(tmp_path, uvw_double):
    """
    This tests that the times and uvws are round-tripped through write/read
    uvfits to sufficient precision.
    """
    pytest.importorskip("casacore")
    lwa_file = os.path.join(
        DATA_PATH, "2018-03-21-01_26_33_0004384620257280_000000_downselected.ms"
    )
    uvd = UVData()
    uvd.read(lwa_file, default_mount_type="fixed")

    testfile = os.path.join(tmp_path, "lwa_testfile.uvfits")
    uvd.write_uvfits(testfile, uvw_double=uvw_double)

    uvd2 = UVData()
    uvd2.read(testfile)

    unique_times, inverse_inds = np.unique(uvd2.time_array, return_inverse=True)
    unique_lst_array = utils.get_lst_for_time(
        unique_times, telescope_loc=uvd.telescope.location
    )

    calc_lst_array = unique_lst_array[inverse_inds]

    np.testing.assert_allclose(
        calc_lst_array,
        uvd2.lst_array,
        rtol=uvd2._lst_array.tols[0],
        atol=uvd2._lst_array.tols[1],
    )

    if uvw_double:
        np.testing.assert_allclose(uvd.uvw_array, uvd2.uvw_array, rtol=1e-13)
    else:
        np.testing.assert_allclose(uvd.uvw_array, uvd2.uvw_array, rtol=1e-7)
        assert not np.allclose(uvd.uvw_array, uvd2.uvw_array, rtol=1e-13)

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


def test_break_read_uvfits(tmp_path):
    """Test errors on reading in a uvfits file with subarrays and other problems."""
    uvobj = UVData()

    file1 = os.path.join(DATA_PATH, "1061316296.uvfits")
    write_file = os.path.join(tmp_path, "multi_subarray.uvfits")
    with fits.open(file1, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        vis_data = vis_hdu.data.copy()

        vis_data["SUBARRAY"][:] = np.arange(len(vis_data["SUBARRAY"]))
        vis_hdu.data = vis_data

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)
        hdulist.close()

    with pytest.raises(ValueError, match="This file appears to have multiple subarray"):
        uvobj.read(write_file)

    write_file = os.path.join(tmp_path, "bad_frame.uvfits")

    with fits.open(file1, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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
        uvobj.read(write_file, read_data=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:UVFITS file is missing AIPS SU table")
def test_source_group_params(casa_uvfits, tmp_path):
    # make a file with a single source to test that it works
    uv_in = casa_uvfits
    write_file = os.path.join(tmp_path, "outtest_casa.uvfits")
    write_file2 = os.path.join(tmp_path, "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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
    uv_out.read(write_file2)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa2.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
    assert uv_in == uv_out


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
    uv_out = UVData.from_file(write_file)

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
        hdunames = fits_utils._indexhdus(hdu_list)
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
    uv_out.read(write_file2)
    assert uv_out.phase_center_catalog[0]["cat_frame"] == frame


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
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
        hdunames = fits_utils._indexhdus(hdu_list)
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
    uv_out.read(write_file2)

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
def test_missing_aips_su_table(casa_uvfits, tmp_path):
    # make a file with multiple sources to test error condition
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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

    with check_warnings(
        UserWarning,
        [
            "UVFITS file is missing AIPS SU table, which is required when ",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_in.read(write_file2)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_casa_nonascii_bytes_antenna_names(tmpdir):
    """Test that nonascii bytes in antenna names are handled properly."""
    orig_file = casa_tutorial_uvfits
    testfile = tmpdir + "test_nonascii_antnames.uvfits"

    with open(orig_file, "rb") as f_in, open(testfile, "wb") as f_out:
        f_bytes = f_in.read()
        assert b"W08\x00" in f_bytes
        new_bytes = f_bytes.replace(b"W08\x00", b"W08\xc0")
        f_out.write(new_bytes)

    uv1 = UVData.from_file(orig_file)
    uv2 = UVData.from_file(testfile)

    assert uv1 == uv2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_readwriteread(tmp_path, casa_uvfits, telescope_frame, selenoid):
    """
    CASA tutorial uvfits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality.
    """
    uv_in = casa_uvfits

    if telescope_frame == "mcmf":
        pytest.importorskip("lunarsky")
        from lunarsky import MoonLocation
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        enu_antpos = uv_in.telescope.get_enu_antpos()
        uv_in.telescope.location = MoonLocation.from_selenodetic(
            lat=uv_in.telescope.location.lat,
            lon=uv_in.telescope.location.lon,
            height=uv_in.telescope.location.height,
            ellipsoid=selenoid,
        )
        new_full_antpos = utils.ECEF_from_ENU(
            enu=enu_antpos, center_loc=uv_in.telescope.location
        )
        uv_in.telescope.antenna_positions = (
            new_full_antpos - uv_in.telescope._location.xyz()
        )
        uv_in.set_lsts_from_time_array()
        uv_in.set_uvws_from_antenna_positions()
        try:
            uv_in._set_app_coords_helper()
        except SpiceUNKNOWNFRAME as err:
            pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))
        uv_in.check()

    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")

    uv_in.write_uvfits(write_file)
    file_read = write_file
    # check handling of default ellipsoid: remove the ellipsoid and check that
    # it is properly defaulted to SPHERE
    if telescope_frame == "mcmf" and selenoid == "SPHERE":
        with fits.open(write_file, memmap=True) as hdu_list:
            hdunames = fits_utils._indexhdus(hdu_list)
            ant_hdu = hdu_list[hdunames["AIPS AN"]]
            ant_hdr = ant_hdu.header.copy()

            del ant_hdr["ELLIPSOI"]
            ant_hdu.header = ant_hdr

            vis_hdu = hdu_list[0]
            source_hdu = hdu_list[hdunames["AIPS SU"]]
            hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu, source_hdu])
            hdulist.writeto(write_file2, overwrite=True)
            hdulist.close()
            file_read = write_file2

    uv_out.read(file_read)

    assert uv_in.telescope._location.frame == uv_out.telescope._location.frame
    assert uv_in.telescope._location.ellipsoid == uv_out.telescope._location.ellipsoid

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("uvw_suffix", ["--", "---SIN", "---NCP"])
def test_uvw_coordinate_suffixes(casa_uvfits, tmp_path, uvw_suffix):
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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
        with check_warnings(
            UserWarning,
            match=[
                (
                    "The baseline coordinates (uvws) in this file are specified in the "
                    "---NCP coordinate system"
                ),
                "The uvw_array does not match the expected values",
            ],
        ):
            uv2 = UVData.from_file(write_file2)
        uv2.uvw_array = utils.phasing._rotate_one_axis(
            xyz_array=uv2.uvw_array[:, :, None],
            rot_amount=-1 * (uv2.phase_center_app_dec - np.pi / 2),
            rot_axis=0,
        )[:, :, 0]
    else:
        uv2 = UVData.from_file(write_file2)

    uv2._consolidate_phase_center_catalogs(reference_catalog=uv_in.phase_center_catalog)
    assert uv2 == uv_in


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "uvw_suffix", [["---SIN", "", ""], ["", "---NCP", ""], ["", "---NCP", "---SIN"]]
)
def test_uvw_coordinate_suffix_errors(casa_uvfits, tmp_path, uvw_suffix):
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")
    uv_in.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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
        UVData.from_file(write_file2)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
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

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
    assert uv_in == uv_out

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_uvfits_optional_params(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that if optional params are set, they are read back out properly
    uv_in.telescope.set_feeds_from_x_orientation(
        "east", polarization_array=uv_in.polarization_array
    )
    uv_in.telescope.pol_convention = "sum"
    # Order feeds in AIPS convention for round-tripping
    uv_in.telescope.reorder_feeds("AIPS")
    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

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
def test_readwriteread_antenna_diameters(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that if antenna_diameters is set, it's read back out properly
    uv_in.telescope.antenna_diameters = (
        np.zeros((uv_in.telescope.Nants,), dtype=np.float64) + 14.0
    )
    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

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
def test_readwriteread_large_antnums(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that if antenna_numbers are > 256 everything works
    uv_in.telescope.antenna_numbers = uv_in.telescope.antenna_numbers + 256
    uv_in.ant_1_array = uv_in.ant_1_array + 256
    uv_in.ant_2_array = uv_in.ant_2_array + 256
    uv_in.baseline_array = uv_in.antnums_to_baseline(
        uv_in.ant_1_array, uv_in.ant_2_array
    )
    with check_warnings(
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
    uv_out.read(write_file)

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
@pytest.mark.parametrize("lat_lon_alt", [True, False])
def test_readwriteread_missing_info(tmp_path, casa_uvfits, lat_lon_alt):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")

    # check missing telescope_name, timesys vs timsys spelling, xyz_telescope_frame=????
    uv_in.write_uvfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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

    with check_warnings(
        UserWarning,
        match=[
            (
                "The telescope frame is set to '????', which generally indicates "
                "ignorance. Defaulting the frame to 'itrs', but this may lead to other "
                "warnings or errors."
            ),
            (
                "The uvw_array does not match the expected values given the antenna "
                "positions."
            ),
        ],
    ):
        uv_out.read(write_file2)
    assert uv_out.telescope.name == "EVLA"
    assert uv_out.timesys == time_sys

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
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
def test_readwriteread_error_single_time(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")

    # check error if one time & no inttime specified
    uv_singlet = uv_in.select(times=uv_in.time_array[0], inplace=False)
    uv_singlet.write_uvfits(write_file)

    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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

    with (
        pytest.raises(
            ValueError, match="Required UVParameter _integration_time has not been set"
        ),
        check_warnings(
            [erfa.core.ErfaWarning, erfa.core.ErfaWarning, UserWarning, UserWarning],
            [
                "ERFA function 'utcut1' yielded 1 of 'dubious year (Note 3)'",
                "ERFA function 'utctai' yielded 1 of 'dubious year (Note 3)'",
                "LST values stored in this file are not self-consistent",
                "The integration time is not specified and only one time",
            ],
        ),
    ):
        uv_out.read(write_file2)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.skipif(
    len(frame_selenoid) > 1, reason="Test only when lunarsky not installed."
)
def test_uvfits_no_moon(casa_uvfits, tmp_path):
    """Check errors when reading uvfits with MCMF without lunarsky."""
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")
    write_file2 = str(tmp_path / "outtest_casa2.uvfits")

    uv_in.write_uvfits(write_file)

    uv_out = UVData()
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        ant_hdr = ant_hdu.header.copy()

        ant_hdr["FRAME"] = "mcmf"
        ant_hdu.header = ant_hdr

        vis_hdu = hdu_list[0]
        source_hdu = hdu_list[hdunames["AIPS SU"]]
        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu, source_hdu])
        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    msg = "Need to install `lunarsky` package to work with MCMF frame."
    with pytest.raises(ValueError, match=msg):
        uv_out.read(write_file2)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_readwriteread_unflagged_data_warnings(tmp_path, casa_uvfits):
    uv_in = casa_uvfits
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # check that unflagged data with nsample = 0 will cause warnings
    uv_in.nsample_array[list(range(11, 22))] = 0
    uv_in.flag_array[list(range(11, 22))] = False
    with check_warnings(
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

    with check_warnings(UserWarning, match=uvw_warn_str):
        uv_in.check()

    if errstr is not None:
        with pytest.raises(TypeError, match=errstr):
            uv_in.write_uvfits(testfile, run_check=False)
    else:
        with check_warnings(UserWarning, match=warnstr):
            uv_in.write_uvfits(testfile, run_check=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
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

    for name, value in zip(kwd_names, kwd_values, strict=True):
        uv_in.extra_keywords[name] = value
    uv_in.write_uvfits(testfile)
    uv_out.read(testfile)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["outtest_casa.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("order", ["time", "bda"])
def test_roundtrip_blt_order(casa_uvfits, order, tmp_path):
    uv_in = casa_uvfits
    uv_out = UVData()
    write_file = str(tmp_path / "blt_order_test.uvfits")

    uv_in.reorder_blts(order=order)

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

    # make sure filenames are what we expect
    assert uv_in.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    assert uv_out.filename == ["blt_order_test.uvfits"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(
        reference_catalog=uv_in.phase_center_catalog
    )
    assert uv_in == uv_out


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
    if "time_inds" in select_kwargs:
        time_inds = select_kwargs.pop("time_inds")
        unique_times = np.unique(uvfits_uv2.time_array)
        select_kwargs["time_range"] = unique_times[time_inds]

    if "lst_inds" in select_kwargs:
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
    uvfits_uv.read(paper_uvfits, run_check_acceptability=False, **select_kwargs)

    uvfits_uv2.select(run_check_acceptability=False, **select_kwargs)
    assert uvfits_uv == uvfits_uv2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_select_read_nospw_pol(casa_uvfits, tmp_path):
    # this requires writing a new file because the no spw file we have has only 1 pol

    with fits.open(casa_tutorial_uvfits, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
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
    uvfits_uv.read(write_file, polarizations=pols_to_keep)
    uvfits_uv2 = casa_uvfits
    uvfits_uv2.select(polarizations=pols_to_keep)

    # make sure filenames are what we expect
    assert uvfits_uv.filename == ["outtest_casa.uvfits"]
    assert uvfits_uv2.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uvfits_uv.filename = uvfits_uv2.filename

    assert uvfits_uv == uvfits_uv2


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_uvfits_write_miriad(casa_uvfits, tmp_path):
    """
    read uvfits, write miriad test.
    Read in uvfits file, write out as miriad, read back in and check for
    object equality.
    """
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    uvfits_uv = casa_uvfits
    miriad_uv = UVData()
    testfile = str(tmp_path / "outtest_miriad")
    with check_warnings(
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
    miriad_uv.read_miriad(testfile)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    miriad_uv.filename = uvfits_uv.filename

    miriad_uv._consolidate_phase_center_catalogs(
        reference_catalog=uvfits_uv.phase_center_catalog
    )
    assert miriad_uv == uvfits_uv

    # check that setting the projected keyword also works
    miriad_uv.read_miriad(testfile, projected=True)

    # check that setting the projected False raises an error
    with pytest.raises(
        ValueError, match="projected is False but the RA values are constant."
    ):
        miriad_uv.read_miriad(testfile, projected=False)

    # check that setting it works after selecting a single time
    uvfits_uv.select(times=uvfits_uv.time_array[0])
    with check_warnings(
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
    miriad_uv.read_miriad(testfile)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert uvfits_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    miriad_uv.filename = uvfits_uv.filename

    miriad_uv._consolidate_phase_center_catalogs(
        reference_catalog=uvfits_uv.phase_center_catalog
    )
    assert miriad_uv == uvfits_uv


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
    assert utils.history._check_histories(
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
    assert utils.history._check_histories(
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
    assert utils.history._check_histories(
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

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=ms_uv.phase_center_catalog
    )
    assert ms_uv == uvfits_uv


def test_cotter_telescope_frame(tmp_path):
    file1 = os.path.join(DATA_PATH, "1061316296.uvfits")
    write_file = os.path.join(tmp_path, "emulate_cotter.uvfits")
    uvd1 = UVData()

    with fits.open(file1, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
        vis_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["AIPS AN"]]
        ant_hdu.header.pop("FRAME")

        hdulist = fits.HDUList(hdus=[vis_hdu, ant_hdu])
        hdulist.writeto(write_file, overwrite=True)
        hdulist.close()

    with check_warnings(
        UserWarning,
        "Required Antenna keyword 'FRAME' not set; Assuming frame is 'ITRF'.",
    ):
        uvd1.read_uvfits(write_file, read_data=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_readwriteread_reorder_pols(tmp_path, casa_uvfits):
    """
    CASA tutorial uvfits loopback test.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality. We check that on-the-fly polarization reordering works.
    """
    uv_in = casa_uvfits

    uv_out = UVData()
    write_file = str(tmp_path / "outtest_casa.uvfits")

    # reorder polarizations
    polarization_input = uv_in.polarization_array
    uv_in.reorder_pols(order=[3, 0, 2, 1])
    assert not np.allclose(uv_in.polarization_array, polarization_input)

    uv_in.write_uvfits(write_file)
    uv_out.read(write_file)

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
        [-1, 0, "Frequency values must be > 0 for UVFITS!"],
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
    with check_warnings(
        UserWarning,
        "Required Antenna keyword 'FRAME' not set; Assuming frame is 'ITRF'.",
    ):
        UVData.from_file(outfile, read_data=False)


def test_mwax_missing_frame_comment(tmp_path):
    fits_file = os.path.join(DATA_PATH, "1061316296.uvfits")
    outfile = tmp_path / "mwax_birli.uvfits"
    with fits.open(fits_file, memmap=True) as hdu_list:
        del hdu_list[1].header["FRAME"], hdu_list[0].header["COMMENT"]
        hdu_list[0].header["COMMENT"] = "A dummy comment."
        hdu_list.writeto(outfile)
    with check_warnings(
        UserWarning,
        "Required Antenna keyword 'FRAME' not set; Assuming frame is 'ITRF'.",
    ):
        UVData.from_file(outfile, read_data=False)


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

    sma_uvfits = UVData.from_file(filename)

    # UVFITS has some differences w/ the MIR format that are expected -- handle
    # all of that here, making sure that the returned values are consistent with
    # what we expect. Start w/ spectral windows
    assert len(np.unique(sma_mir.spw_array)) == len(np.unique(sma_uvfits.spw_array))

    spw_dict = dict(zip(sma_uvfits.spw_array, sma_mir.spw_array, strict=True))

    assert np.all(
        [
            idx == spw_dict[jdx]
            for idx, jdx in zip(
                sma_mir.flex_spw_id_array, sma_uvfits.flex_spw_id_array, strict=True
            )
        ]
    )
    sma_uvfits.spw_array = sma_mir.spw_array
    sma_uvfits.flex_spw_id_array = sma_mir.flex_spw_id_array

    # Check the history next
    assert sma_uvfits.history.startswith(sma_mir.history)
    sma_mir.history = sma_uvfits.history

    # We have to do a bit of special handling for the phase_center_catalog, because
    # _very_ small floating point errors can creep in.
    for cat_name in sma_mir.phase_center_catalog:
        this_cat = sma_mir.phase_center_catalog[cat_name]
        other_cat = sma_uvfits.phase_center_catalog[cat_name]

        assert np.isclose(
            this_cat["cat_lat"], other_cat["cat_lat"], rtol=0, atol=utils.RADIAN_TOL
        )
        assert np.isclose(
            this_cat["cat_lon"], other_cat["cat_lon"], rtol=0, atol=utils.RADIAN_TOL
        )
    sma_uvfits.phase_center_catalog = sma_mir.phase_center_catalog

    # Finally, move on to the uvfits extra parameters
    exp_dict = {
        "dut1": -0.2139843,
        "earth_omega": 360.9856438593,
        "gst0": 302.1745672595617,
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


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
def test_uvfits_extra_phase_centers(sma_mir, tmp_path):
    """
    Verify that extra phase centers are correctly handled when reading from and writing
    to UVFITS file format.
    """
    sma_uvfits = UVData()
    sma_mir._set_app_coords_helper()

    # Add a dummy entry
    sma_mir._add_phase_center(
        "dummy",
        cat_type="sidereal",
        cat_lon=2.0,
        cat_lat=1.0,
        cat_frame="icrs",
        cat_epoch=2000.0,
    )
    filename = os.path.join(tmp_path, "test.uvfits")

    sma_mir.write_uvfits(filename)

    sma_uvfits = UVData.from_file(filename)

    this_names = {
        value["cat_name"]: key for key, value in sma_mir.phase_center_catalog.items()
    }
    other_names = {
        value["cat_name"]: key for key, value in sma_uvfits.phase_center_catalog.items()
    }

    # Check the entries of the phase center catalog
    for cat_id in sma_mir.phase_center_catalog:
        name = sma_mir.phase_center_catalog[cat_id]["cat_name"]
        this_cat = sma_mir.phase_center_catalog[this_names[name]]
        other_cat = sma_uvfits.phase_center_catalog[other_names[name]]

        assert np.isclose(
            this_cat["cat_lat"], other_cat["cat_lat"], rtol=0, atol=utils.RADIAN_TOL
        )
        assert np.isclose(
            this_cat["cat_lon"], other_cat["cat_lon"], rtol=0, atol=utils.RADIAN_TOL
        )
        assert this_cat["cat_name"] == other_cat["cat_name"]
        assert this_cat["cat_frame"] == other_cat["cat_frame"]
        assert this_cat["cat_epoch"] == other_cat["cat_epoch"]
        assert np.array_equal(
            sma_mir.phase_center_id_array == this_names[name],
            sma_uvfits.phase_center_id_array == other_names[name],
        )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_uvfits_phasing_errors(hera_uvh5, tmp_path):
    # check error if data are not phase to a sidereal source and force_phase not set
    with pytest.raises(
        ValueError, match="The data are not all phased to a sidereal source"
    ):
        hera_uvh5.write_uvfits(tmp_path)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_convention(tmp_path):
    """
    Test writing a MIRIAD-compatible UVFITS file
    """
    uv = UVData()
    uv.read(casa_tutorial_uvfits)

    # Change an antenna ID to 512
    old_idx = uv.telescope.antenna_numbers[10]  # This is antenna 19
    new_idx = 512

    uv.telescope.antenna_numbers[10] = new_idx
    uv.ant_1_array[uv.ant_1_array == old_idx] = new_idx
    uv.ant_2_array[uv.ant_2_array == old_idx] = new_idx
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, uv.ant_2_array)

    testfile1 = str(tmp_path / "uv1.uvfits")
    uv.write_uvfits(testfile1, use_miriad_convention=True)

    # These are based on known values in casa_tutorial_uvfits
    expected_vals = {"ANTENNA1_0": 4, "ANTENNA2_0": 8, "NOSTA_0": 1}

    # Check baselines match MIRIAD convention
    bl_miriad_expected = utils.antnums_to_baseline(
        uv.ant_1_array, uv.ant_2_array, Nants_telescope=512, use_miriad_convention=True
    )
    with fits.open(testfile1) as hdu:
        np.testing.assert_allclose(hdu[0].data["BASELINE"], bl_miriad_expected)

        # Quick check of other antenna values
        assert hdu[0].data["ANTENNA1"][0] == expected_vals["ANTENNA1_0"]
        assert hdu[0].data["ANTENNA2"][0] == expected_vals["ANTENNA2_0"]
        assert hdu[1].data["NOSTA"][0] == expected_vals["NOSTA_0"]

    uv2 = UVData.from_file(testfile1)
    uv2._update_phase_center_id(1, new_id=0)
    uv2.phase_center_catalog[0]["info_source"] = uv.phase_center_catalog[0][
        "info_source"
    ]

    assert uv2._ant_1_array == uv._ant_1_array
    assert uv2._ant_2_array == uv._ant_2_array

    assert uv2 == uv

    # Test that antennas get +1 if there is a 0-indexed antennas
    old_idx = uv.telescope.antenna_numbers[0]
    new_idx = 0
    uv.telescope.antenna_numbers[0] = new_idx
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
    uv2._update_phase_center_id(1, new_id=0)
    uv2.phase_center_catalog[0]["info_source"] = uv.phase_center_catalog[0][
        "info_source"
    ]

    # adjust for expected antenna number changes:
    uv2.telescope.antenna_numbers -= 1
    uv2.ant_1_array -= 1
    uv2.ant_2_array -= 1
    uv2.baseline_array = uv2.antnums_to_baseline(uv2.ant_1_array, uv2.ant_2_array)

    assert uv2 == uv


def test_feed_err(sma_mir, tmp_path):
    outpath = os.path.join(tmp_path, "feed_err")
    sma_mir.telescope.feed_array.flat[0] = "k"
    with pytest.raises(ValueError, match="UVFITS only supports"):
        sma_mir.write_uvfits(outpath, run_check=False)


def test_vlbi_read():
    with check_warnings(
        UserWarning,
        match=["The uvw_array does not match", "The telescope frame is set to '?????'"],
    ):
        uvd = UVData.from_file(os.path.join(DATA_PATH, "mojave.uvfits"))

    assert uvd.telescope.instrument == "VLBA"
    assert uvd.telescope.name == "VLBA"
