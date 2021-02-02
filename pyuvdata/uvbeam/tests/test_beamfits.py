# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for BeamFits object.

"""
import os

import pytest
import numpy as np
from astropy.io import fits

from pyuvdata import UVBeam
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils

filenames = ["HERA_NicCST_150MHz.txt", "HERA_NicCST_123MHz.txt"]
cst_folder = "NicCSTbeams"
cst_files = [os.path.join(DATA_PATH, cst_folder, f) for f in filenames]


@pytest.fixture(scope="module")
def cst_power_1freq(cst_efield_1freq_main):
    beam_in = cst_efield_1freq_main.copy()
    beam_in.efield_to_power()
    return beam_in.copy()


@pytest.fixture(scope="module")
def cst_power_1freq_cut_healpix(cst_efield_1freq_cut_healpix_main):
    beam_in = cst_efield_1freq_cut_healpix_main.copy()
    beam_in.efield_to_power()
    return beam_in.copy()


@pytest.fixture(scope="function")
def hera_beam_casa():
    beam_in = UVBeam()
    casa_file = os.path.join(DATA_PATH, "HERABEAM.FITS")
    beam_in.read_beamfits(casa_file, run_check=False)

    # fill in missing parameters
    beam_in.data_normalization = "peak"
    beam_in.feed_name = "casa_ideal"
    beam_in.feed_version = "v0"
    beam_in.model_name = "casa_airy"
    beam_in.model_version = "v0"

    # this file is actually in an orthoslant projection RA/DEC at zenith at a
    # particular time.
    # For now pretend it's in a zenith orthoslant projection
    beam_in.pixel_coordinate_system = "orthoslant_zenith"

    return beam_in


def test_read_cst_write_read_fits_efield(cst_efield_1freq, tmp_path):
    beam_in = cst_efield_1freq.copy()
    beam_out = UVBeam()

    # add optional parameters for testing purposes
    beam_in.extra_keywords = {"KEY1": "test_keyword"}
    beam_in.x_orientation = "east"
    beam_in.reference_impedance = 340.0
    beam_in.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(beam_in.Nspws, beam_in.Nfreqs)
    )
    beam_in.loss_array = np.random.normal(50.0, 5, size=(beam_in.Nspws, beam_in.Nfreqs))
    beam_in.mismatch_array = np.random.normal(
        0.0, 1.0, size=(beam_in.Nspws, beam_in.Nfreqs)
    )
    beam_in.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, beam_in.Nspws, beam_in.Nfreqs)
    )
    beam_in.interpolation_function = "az_za_simple"
    beam_in.freq_interp_kind = "linear"

    write_file = str(tmp_path / "outtest_beam.fits")

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    assert beam_in == beam_out

    return


def test_read_cst_write_read_fits_power(cst_power_1freq, tmp_path):
    # redo for power beam
    beam_in = cst_power_1freq
    beam_out = UVBeam()

    # add optional parameters for testing purposes
    beam_in.extra_keywords = {"KEY1": "test_keyword"}
    beam_in.x_orientation = "east"
    beam_in.reference_impedance = 340.0
    beam_in.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(beam_in.Nspws, beam_in.Nfreqs)
    )
    beam_in.loss_array = np.random.normal(50.0, 5, size=(beam_in.Nspws, beam_in.Nfreqs))
    beam_in.mismatch_array = np.random.normal(
        0.0, 1.0, size=(beam_in.Nspws, beam_in.Nfreqs)
    )
    beam_in.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, beam_in.Nspws, beam_in.Nfreqs)
    )

    write_file = str(tmp_path / "outtest_beam.fits")

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)
    assert beam_in == beam_out

    return


def test_read_cst_write_read_fits_intensity(cst_power_1freq, tmp_path):
    # set up power beam
    beam_in = cst_power_1freq
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam.fits")
    write_file2 = str(tmp_path / "outtest_beam2.fits")
    beam_in.write_beamfits(write_file, clobber=True)

    # now replace 'power' with 'intensity' for btype
    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    primary_hdr["BTYPE"] = "Intensity"
    hdunames = uvutils._fits_indexhdus(fname)
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    beam_out.read_beamfits(write_file2)
    assert beam_in == beam_out

    return


def test_read_cst_write_read_fits_no_coordsys(cst_power_1freq, tmp_path):
    # set up power beam
    beam_in = cst_power_1freq
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam.fits")
    write_file2 = str(tmp_path / "outtest_beam2.fits")
    beam_in.write_beamfits(write_file, clobber=True)

    # now remove coordsys but leave ctypes 1 & 2
    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    primary_hdr.pop("COORDSYS")
    hdunames = uvutils._fits_indexhdus(fname)
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    beam_out.read_beamfits(write_file2)
    assert beam_in == beam_out

    return


def test_read_cst_write_read_fits_change_freq_units(cst_power_1freq, tmp_path):
    # set up power beam
    beam_in = cst_power_1freq
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam.fits")
    write_file2 = str(tmp_path / "outtest_beam2.fits")
    beam_in.write_beamfits(write_file, clobber=True)

    # now change frequency units
    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    primary_hdr["CUNIT3"] = "MHz"
    primary_hdr["CRVAL3"] = primary_hdr["CRVAL3"] / 1e6
    primary_hdr["CDELT3"] = primary_hdr["CRVAL3"] / 1e6
    hdunames = uvutils._fits_indexhdus(fname)
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    beam_out.read_beamfits(write_file2)
    assert beam_in == beam_out

    return


def test_writeread_healpix(cst_efield_1freq_cut_healpix, tmp_path):
    beam_in = cst_efield_1freq_cut_healpix.copy()
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam_hpx.fits")

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    assert beam_in == beam_out

    return


def test_writeread_healpix_power(cst_power_1freq_cut_healpix, tmp_path):
    # redo for power beam
    beam_in = cst_power_1freq_cut_healpix
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam_hpx.fits")

    # add optional parameters for testing purposes
    beam_in.extra_keywords = {"KEY1": "test_keyword"}
    beam_in.x_orientation = "east"
    beam_in.reference_impedance = 340.0
    beam_in.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(beam_in.Nspws, beam_in.Nfreqs)
    )
    beam_in.loss_array = np.random.normal(50.0, 5, size=(beam_in.Nspws, beam_in.Nfreqs))
    beam_in.mismatch_array = np.random.normal(
        0.0, 1.0, size=(beam_in.Nspws, beam_in.Nfreqs)
    )
    beam_in.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, beam_in.Nspws, beam_in.Nfreqs)
    )

    # check that data_array is complex
    assert np.iscomplexobj(np.real_if_close(beam_in.data_array, tol=10))

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    assert beam_in == beam_out

    return


def test_writeread_healpix_no_corrdsys(cst_power_1freq_cut_healpix, tmp_path):
    beam_in = cst_power_1freq_cut_healpix
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam.fits")
    write_file2 = str(tmp_path / "outtest_beam2.fits")
    beam_in.write_beamfits(write_file, clobber=True)

    # now remove coordsys but leave ctype 1
    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    primary_hdr.pop("COORDSYS")
    hdunames = uvutils._fits_indexhdus(fname)
    hpx_hdu = fname[hdunames["HPX_INDS"]]
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, hpx_hdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    beam_out.read_beamfits(write_file2)
    assert beam_in == beam_out

    return


def test_error_beam_type(cst_efield_1freq, tmp_path):
    beam_in = cst_efield_1freq
    beam_in.beam_type = "foo"

    write_file = str(tmp_path / "outtest_beam.fits")

    # make sure writing fails
    with pytest.raises(
        ValueError, match="UVParameter _beam_type has unacceptable values"
    ):
        beam_in.write_beamfits(write_file, clobber=True)

    # make sure it fails even if check is off
    with pytest.raises(ValueError, match="Unknown beam_type: foo"):
        beam_in.write_beamfits(write_file, clobber=True, run_check=False)

    return


def test_error_antenna_type(cst_efield_1freq, tmp_path):
    beam_in = cst_efield_1freq
    beam_in.antenna_type = "phased_array"

    write_file = str(tmp_path / "outtest_beam.fits")
    with pytest.raises(
        ValueError, match="This beam fits writer currently only supports"
    ):
        beam_in.write_beamfits(write_file, clobber=True)

    return


@pytest.mark.parametrize(
    "header_dict,error_msg",
    [
        ({"BTYPE": "foo"}, "Unknown beam_type: foo"),
        ({"COORDSYS": "orthoslant_zenith"}, "Coordinate axis list does not match"),
        ({"NAXIS": ""}, "beam_type is efield and data dimensionality"),
        ({"CUNIT1": "foo"}, 'Units of first axis array are not "deg" or "rad"'),
        ({"CUNIT2": "foo"}, 'Units of second axis array are not "deg" or "rad"'),
        ({"CUNIT3": "foo"}, "Frequency units not recognized"),
    ],
)
def test_header_val_errors(cst_efield_1freq, tmp_path, header_dict, error_msg):
    beam_in = cst_efield_1freq
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam.fits")
    write_file2 = str(tmp_path / "outtest_beam2.fits")

    # now change values for various items in primary hdu to test errors
    beam_in.write_beamfits(write_file, clobber=True)

    keyword = list(header_dict.keys())[0]
    new_val = header_dict[keyword]
    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    hdunames = uvutils._fits_indexhdus(fname)
    basisvec_hdu = fname[hdunames["BASISVEC"]]
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    if "NAXIS" in keyword:
        ax_num = keyword.split("NAXIS")[1]
        if ax_num != "":
            ax_num = int(ax_num)
            ax_use = len(data.shape) - ax_num
            new_arrays = np.split(data, primary_hdr[keyword], axis=ax_use)
            data = new_arrays[0]
        else:
            data = np.squeeze(
                np.split(data, primary_hdr["NAXIS1"], axis=len(data.shape) - 1)[0]
            )
    else:
        primary_hdr[keyword] = new_val

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, basisvec_hdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    with pytest.raises(ValueError, match=error_msg):
        beam_out.read_beamfits(write_file2)

    return


@pytest.mark.parametrize(
    "header_dict,error_msg",
    [
        ({"COORDSYS": "foo"}, "Pixel coordinate system in BASISVEC"),
        ({"CTYPE1": "foo"}, "Pixel coordinate list in BASISVEC"),
        ({"CTYPE2": "foo"}, "Pixel coordinate list in BASISVEC"),
        ({"CDELT1": "foo"}, "First image axis in BASISVEC"),
        ({"CDELT2": "foo"}, "Second image axis in BASISVEC"),
        ({"NAXIS4": ""}, "Number of vector coordinate axes in BASISVEC"),
        ({"CUNIT1": "foo"}, "Units of first axis array in BASISVEC"),
        ({"CUNIT2": "foo"}, "Units of second axis array in BASISVEC"),
    ],
)
def test_basisvec_hdu_errors(cst_efield_1freq, tmp_path, header_dict, error_msg):
    beam_in = cst_efield_1freq
    beam_out = UVBeam()

    write_file = str(tmp_path / "outtest_beam.fits")
    write_file2 = str(tmp_path / "outtest_beam2.fits")

    # now change values for various items in basisvec hdu to not match primary hdu
    beam_in.write_beamfits(write_file, clobber=True)

    keyword = list(header_dict.keys())[0]
    # hacky treatment of CDELT b/c we need the object to be defined already
    if keyword == "CDELT1":
        new_val = np.diff(beam_in.axis1_array)[0] * 2
    elif keyword == "CDELT2":
        new_val = np.diff(beam_in.axis2_array)[0] * 2
    else:
        new_val = header_dict[keyword]

    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    hdunames = uvutils._fits_indexhdus(fname)
    basisvec_hdu = fname[hdunames["BASISVEC"]]
    basisvec_hdr = basisvec_hdu.header
    basisvec_data = basisvec_hdu.data
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    if "NAXIS" in keyword:
        ax_num = keyword.split("NAXIS")[1]
        if ax_num != "":
            ax_num = int(ax_num)
            ax_use = len(basisvec_data.shape) - ax_num
            new_arrays = np.split(basisvec_data, basisvec_hdr[keyword], axis=ax_use)
            basisvec_data = new_arrays[0]
        else:
            basisvec_data = np.split(
                basisvec_data,
                basisvec_hdr["NAXIS1"],
                axis=len(basisvec_data.shape) - 1,
            )[0]
    else:
        basisvec_hdr[keyword] = new_val

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    basisvec_hdu = fits.ImageHDU(data=basisvec_data, header=basisvec_hdr)
    hdulist = fits.HDUList([prihdu, basisvec_hdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    with pytest.raises(ValueError, match=error_msg):
        beam_out.read_beamfits(write_file2)

    return


@pytest.mark.parametrize(
    "header_dict,error_msg",
    [
        ({"CTYPE1": "foo"}, "Coordinate axis list does not match"),
        ({"NAXIS1": ""}, "Number of pixels in HPX_IND extension"),
    ],
)
def test_healpix_errors(cst_efield_1freq_cut_healpix, tmp_path, header_dict, error_msg):
    beam_in = cst_efield_1freq_cut_healpix
    beam_out = UVBeam()
    write_file = str(tmp_path / "outtest_beam_hpx.fits")
    write_file2 = str(tmp_path / "outtest_beam_hpx2.fits")

    beam_in.write_beamfits(write_file, clobber=True)

    # now change values for various items in primary hdu to test errors
    keyword = list(header_dict.keys())[0]
    new_val = header_dict[keyword]
    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    hdunames = uvutils._fits_indexhdus(fname)
    basisvec_hdu = fname[hdunames["BASISVEC"]]
    hpx_hdu = fname[hdunames["HPX_INDS"]]
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    if "NAXIS" in keyword:
        ax_num = keyword.split("NAXIS")[1]
        if ax_num != "":
            ax_num = int(ax_num)
            ax_use = len(data.shape) - ax_num
            new_arrays = np.split(data, primary_hdr[keyword], axis=ax_use)
            data = new_arrays[0]
        else:
            data = np.squeeze(
                np.split(data, primary_hdr["NAXIS1"], axis=len(data.shape) - 1)[0]
            )
    else:
        primary_hdr[keyword] = new_val

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, basisvec_hdu, hpx_hdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    with pytest.raises(ValueError, match=error_msg):
        beam_out.read_beamfits(write_file2)

    return


@pytest.mark.parametrize(
    "header_dict,error_msg",
    [
        ({"CTYPE1": "foo"}, "First axis in BASISVEC HDU"),
        ({"NAXIS1": ""}, "Number of pixels in BASISVEC HDU"),
    ],
)
def test_healpix_basisvec_hdu_errors(
    cst_efield_1freq_cut_healpix, tmp_path, header_dict, error_msg
):
    beam_in = cst_efield_1freq_cut_healpix
    beam_out = UVBeam()
    write_file = str(tmp_path / "outtest_beam_hpx.fits")
    write_file2 = str(tmp_path / "outtest_beam_hpx2.fits")

    beam_in.write_beamfits(write_file, clobber=True)

    # now change values for various items in basisvec hdu to not match primary hdu
    keyword = list(header_dict.keys())[0]
    new_val = header_dict[keyword]
    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    hdunames = uvutils._fits_indexhdus(fname)
    basisvec_hdu = fname[hdunames["BASISVEC"]]
    basisvec_hdr = basisvec_hdu.header
    basisvec_data = basisvec_hdu.data
    hpx_hdu = fname[hdunames["HPX_INDS"]]
    bandpass_hdu = fname[hdunames["BANDPARM"]]

    if "NAXIS" in keyword:
        ax_num = keyword.split("NAXIS")[1]
        if ax_num != "":
            ax_num = int(ax_num)
            ax_use = len(basisvec_data.shape) - ax_num
            new_arrays = np.split(basisvec_data, basisvec_hdr[keyword], axis=ax_use)
            basisvec_data = new_arrays[0]
        else:
            basisvec_data = np.split(
                basisvec_data,
                basisvec_hdr["NAXIS1"],
                axis=len(basisvec_data.shape) - 1,
            )[0]
    else:
        basisvec_hdr[keyword] = new_val

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    basisvec_hdu = fits.ImageHDU(data=basisvec_data, header=basisvec_hdr)
    hdulist = fits.HDUList([prihdu, basisvec_hdu, hpx_hdu, bandpass_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    with pytest.raises(ValueError, match=error_msg):
        beam_out.read_beamfits(write_file2)

    return


def test_casa_beam(tmp_path, hera_beam_casa):
    # test reading in CASA power beam. Some header items are missing...
    beam_in = hera_beam_casa
    beam_out = UVBeam()
    write_file = str(tmp_path / "outtest_beam.fits")

    expected_extra_keywords = [
        "OBSERVER",
        "OBSDEC",
        "DATAMIN",
        "OBJECT",
        "INSTRUME",
        "DATAMAX",
        "OBSRA",
        "ORIGIN",
        "DATE-MAP",
        "DATE",
        "EQUINOX",
        "DATE-OBS",
        "COMMENT",
    ]
    assert expected_extra_keywords.sort() == list(beam_in.extra_keywords.keys()).sort()

    beam_in.write_beamfits(write_file, clobber=True)
    beam_out.read_beamfits(write_file)

    assert beam_in == beam_out


@pytest.mark.parametrize(
    "ex_val,error_msg",
    [
        ({"testdict": {"testkey": 23}}, "Extra keyword testdict is of"),
        ({"testlist": [12, 14, 90]}, "Extra keyword testlist is of"),
        ({"testarr": np.array([12, 14, 90])}, "Extra keyword testarr is of"),
    ],
)
def test_extra_keywords_errors(tmp_path, hera_beam_casa, ex_val, error_msg):
    beam_in = hera_beam_casa
    testfile = str(tmp_path / "outtest_beam.fits")

    # fill in missing parameters
    beam_in.data_normalization = "peak"
    beam_in.feed_name = "casa_ideal"
    beam_in.feed_version = "v0"
    beam_in.model_name = "casa_airy"
    beam_in.model_version = "v0"

    # this file is actually in an orthoslant projection RA/DEC at zenith at a
    # particular time.
    # For now pretend it's in a zenith orthoslant projection
    beam_in.pixel_coordinate_system = "orthoslant_zenith"

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    keyword = list(ex_val.keys())[0]
    val = ex_val[keyword]
    beam_in.extra_keywords[keyword] = val
    with uvtest.check_warnings(
        UserWarning, f"{keyword} in extra_keywords is a list, array or dict"
    ):
        beam_in.check()

    with pytest.raises(TypeError, match=error_msg):
        beam_in.write_beamfits(testfile, run_check=False)

    return


def test_extra_keywords_warnings(tmp_path, hera_beam_casa):
    beam_in = hera_beam_casa
    testfile = str(tmp_path / "outtest_beam.fits")

    # check for warnings with extra_keywords keys that are too long
    beam_in.extra_keywords["test_long_key"] = True
    with uvtest.check_warnings(
        UserWarning, "key test_long_key in extra_keywords is longer than 8 characters"
    ):
        beam_in.check()
    with uvtest.check_warnings(
        UserWarning, "key test_long_key in extra_keywords is longer than 8 characters"
    ):
        beam_in.write_beamfits(testfile, run_check=False, clobber=True)

    return


def test_extra_keywords_boolean(tmp_path, hera_beam_casa):
    beam_in = hera_beam_casa
    beam_out = UVBeam()
    testfile = str(tmp_path / "outtest_beam.fits")

    # check handling of boolean keywords
    beam_in.extra_keywords["bool"] = True
    beam_in.extra_keywords["bool2"] = False
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile, run_check=False)

    assert beam_in == beam_out

    return


def test_extra_keywords_int(tmp_path, hera_beam_casa):
    beam_in = hera_beam_casa
    beam_out = UVBeam()
    testfile = str(tmp_path / "outtest_beam.fits")

    # check handling of int-like keywords
    beam_in.extra_keywords["int1"] = np.int64(5)
    beam_in.extra_keywords["int2"] = 7
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile, run_check=False)

    assert beam_in == beam_out

    return


def test_extra_keywords_float(tmp_path, hera_beam_casa):
    beam_in = hera_beam_casa
    beam_out = UVBeam()
    testfile = str(tmp_path / "outtest_beam.fits")

    # check handling of float-like keywords
    beam_in.extra_keywords["float1"] = np.int64(5.3)
    beam_in.extra_keywords["float2"] = 6.9
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile, run_check=False)

    assert beam_in == beam_out

    return


def test_extra_keywords_complex(tmp_path, hera_beam_casa):
    beam_in = hera_beam_casa
    beam_out = UVBeam()
    testfile = str(tmp_path / "outtest_beam.fits")

    # check handling of complex-like keywords
    beam_in.extra_keywords["complex1"] = np.complex64(5.3 + 1.2j)
    beam_in.extra_keywords["complex2"] = 6.9 + 4.6j
    beam_in.write_beamfits(testfile, clobber=True)
    beam_out.read_beamfits(testfile, run_check=False)

    assert beam_in == beam_out

    return


def test_multi_files(cst_efield_2freq, tmp_path):
    """
    Reading multiple files at once.
    """
    beam_full = cst_efield_2freq

    # add optional parameters for testing purposes
    beam_full.extra_keywords = {"KEY1": "test_keyword"}
    beam_full.x_orientation = "east"
    beam_full.reference_impedance = 340.0
    beam_full.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(beam_full.Nspws, beam_full.Nfreqs)
    )
    beam_full.loss_array = np.random.normal(
        50.0, 5, size=(beam_full.Nspws, beam_full.Nfreqs)
    )
    beam_full.mismatch_array = np.random.normal(
        0.0, 1.0, size=(beam_full.Nspws, beam_full.Nfreqs)
    )
    beam_full.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, beam_full.Nspws, beam_full.Nfreqs)
    )

    testfile1 = str(tmp_path / "outtest_beam1.fits")
    testfile2 = str(tmp_path / "outtest_beam2.fits")

    beam1 = beam_full.select(freq_chans=0, inplace=False)
    beam2 = beam_full.select(freq_chans=1, inplace=False)
    beam1.write_beamfits(testfile1, clobber=True)
    beam2.write_beamfits(testfile2, clobber=True)
    beam1.read_beamfits([testfile1, testfile2])
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        beam_full.history + "  Downselected "
        "to specific frequencies using pyuvdata. "
        "Combined data along frequency axis using"
        " pyuvdata.",
        beam1.history,
    )

    beam1.history = beam_full.history
    assert beam1 == beam_full
