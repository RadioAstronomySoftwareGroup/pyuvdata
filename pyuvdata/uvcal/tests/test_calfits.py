# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for calfits object

"""
import pytest
import os
import numpy as np
from astropy.io import fits

from pyuvdata import UVCal
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils


def test_readwriteread(tmp_path):
    """
    Omnical fits loopback test.

    Read in calfits file, write out new calfits file, read back in and check for
    object equality.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(testfile)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out

    return


def test_readwriteread_no_freq_range(tmp_path):
    # test without freq_range parameter
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")

    cal_in.read_calfits(testfile)
    cal_in.freq_range = None
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out

    return


def test_readwriteread_delays(tmp_path):
    """
    Read-Write-Read test with a fits calibration files containing delays.

    Read in uvfits file, write out new uvfits file, read back in and check for
    object equality
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    write_file = str(tmp_path / "outtest_firstcal.fits")
    cal_in.read_calfits(testfile)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del cal_in
    del cal_out


def test_error_unknown_cal_type(tmp_path):
    """
    Test an error is raised when writing an unknown cal type.
    """
    cal_in = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    write_file = str(tmp_path / "outtest_firstcal.fits")
    cal_in.read_calfits(testfile)

    cal_in._set_unknown_cal_type()
    with pytest.raises(ValueError, match="unknown calibration type"):
        cal_in.write_calfits(write_file, run_check=False, clobber=True)

    return


@pytest.mark.parametrize(
    "header_dict,error_msg",
    [
        ({"flag": "CDELT2"}, "Jones values are different in FLAGS"),
        ({"flag": "CDELT3"}, "Time values are different in FLAGS"),
        ({"flag": "CRVAL5"}, "Spectral window values are different in FLAGS"),
        ({"totqual": "CDELT1"}, "Jones values are different in TOTQLTY"),
        ({"totqual": "CDELT2"}, "Time values are different in TOTQLTY"),
        ({"totqual": "CRVAL4"}, "Spectral window values are different in TOTQLTY"),
    ],
)
def test_fits_header_errors_delay(tmp_path, header_dict, error_msg):
    # change values for various axes in flag and total quality hdus to not
    # match primary hdu
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    write_file = str(tmp_path / "outtest_firstcal.fits")
    write_file2 = str(tmp_path / "outtest_firstcal2.fits")

    cal_in.read_calfits(testfile)

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.delay_array = np.ones(
        cal_in._delay_array.expected_shape(cal_in), dtype=np.float64
    )
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    # write file
    cal_in.write_calfits(write_file, clobber=True)

    unit = list(header_dict.keys())[0]
    keyword = header_dict[unit]

    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    hdunames = uvutils._fits_indexhdus(fname)
    ant_hdu = fname[hdunames["ANTENNAS"]]
    flag_hdu = fname[hdunames["FLAGS"]]
    flag_hdr = flag_hdu.header
    totqualhdu = fname[hdunames["TOTQLTY"]]
    totqualhdr = totqualhdu.header

    if unit == "flag":
        flag_hdr[keyword] *= 2
    elif unit == "totqual":
        totqualhdr[keyword] *= 2

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, ant_hdu])
    flag_hdu = fits.ImageHDU(data=flag_hdu.data, header=flag_hdr)
    hdulist.append(flag_hdu)
    totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
    hdulist.append(totqualhdu)

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    with pytest.raises(ValueError, match=error_msg):
        cal_out.read_calfits(write_file2)

    return


@pytest.mark.parametrize(
    "header_dict,error_msg",
    [
        ({"totqual": "CDELT1"}, "Jones values are different in TOTQLTY"),
        ({"totqual": "CDELT2"}, "Time values are different in TOTQLTY"),
        ({"totqual": "CDELT3"}, "Frequency values are different in TOTQLTY"),
        ({"totqual": "CRVAL4"}, "Spectral window values are different in TOTQLTY"),
    ],
)
def test_fits_header_errors_gain(tmp_path, header_dict, error_msg):
    # repeat for gain type file
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")
    write_file2 = str(tmp_path / "outtest_omnical2.fits")
    cal_in.read_calfits(testfile)

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.gain_array = np.ones(
        cal_in._gain_array.expected_shape(cal_in), dtype=np.complex64
    )
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    # write file
    cal_in.write_calfits(write_file, clobber=True)

    unit = list(header_dict.keys())[0]
    keyword = header_dict[unit]

    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    hdunames = uvutils._fits_indexhdus(fname)
    ant_hdu = fname[hdunames["ANTENNAS"]]
    totqualhdu = fname[hdunames["TOTQLTY"]]
    totqualhdr = totqualhdu.header

    if unit == "totqual":
        totqualhdr[keyword] *= 2

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, ant_hdu])
    totqualhdu = fits.ImageHDU(data=totqualhdu.data, header=totqualhdr)
    hdulist.append(totqualhdu)

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    with pytest.raises(ValueError, match=error_msg):
        cal_out.read_calfits(write_file2)

    return


def test_extra_keywords_boolean(tmp_path):
    cal_in = UVCal()
    cal_out = UVCal()
    calfits_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    testfile = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(calfits_file)

    # check handling of boolean keywords
    cal_in.extra_keywords["bool"] = True
    cal_in.extra_keywords["bool2"] = False
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out

    return


def test_extra_keywords_int(tmp_path):
    cal_in = UVCal()
    cal_out = UVCal()
    calfits_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    testfile = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(calfits_file)

    # check handling of int-like keywords
    cal_in.extra_keywords["int1"] = np.int(5)
    cal_in.extra_keywords["int2"] = 7
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out

    return


def test_extra_keywords_float(tmp_path):
    cal_in = UVCal()
    cal_out = UVCal()
    calfits_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    testfile = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(calfits_file)

    # check handling of float-like keywords
    cal_in.extra_keywords["float1"] = np.int64(5.3)
    cal_in.extra_keywords["float2"] = 6.9
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out

    return


def test_extra_keywords_complex(tmp_path):
    cal_in = UVCal()
    cal_out = UVCal()
    calfits_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    testfile = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(calfits_file)

    # check handling of complex-like keywords
    cal_in.extra_keywords["complex1"] = np.complex64(5.3 + 1.2j)
    cal_in.extra_keywords["complex2"] = 6.9 + 4.6j
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out

    return


def test_extra_keywords_comment(tmp_path):
    cal_in = UVCal()
    cal_out = UVCal()
    calfits_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    testfile = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(calfits_file)

    # check handling of comment keywords
    cal_in.extra_keywords["comment"] = (
        "this is a very long comment that will "
        "be broken into several lines\nif "
        "everything works properly."
    )
    cal_in.write_calfits(testfile, clobber=True)
    cal_out.read_calfits(testfile)

    assert cal_in == cal_out

    return


@pytest.mark.parametrize(
    "ex_val,error_msg",
    [
        ({"testdict": {"testkey": 23}}, "Extra keyword testdict is of"),
        ({"testlist": [12, 14, 90]}, "Extra keyword testlist is of"),
        ({"testarr": np.array([12, 14, 90])}, "Extra keyword testarr is of"),
    ],
)
def test_extra_keywords_errors(tmp_path, ex_val, error_msg):
    cal_in = UVCal()
    calfits_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    testfile = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(calfits_file)

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    keyword = list(ex_val.keys())[0]
    val = ex_val[keyword]
    cal_in.extra_keywords[keyword] = val
    with uvtest.check_warnings(
        UserWarning, f"{keyword} in extra_keywords is a list, array or dict"
    ):
        cal_in.check()
    with pytest.raises(TypeError, match=error_msg):
        cal_in.write_calfits(testfile, run_check=False)

    return


def test_extra_keywords_warnings(tmp_path):
    cal_in = UVCal()
    calfits_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    testfile = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(calfits_file)

    # check for warnings with extra_keywords keys that are too long
    cal_in.extra_keywords["test_long_key"] = True
    with uvtest.check_warnings(
        UserWarning, "key test_long_key in extra_keywords is longer than 8 characters"
    ):
        cal_in.check()
    with uvtest.check_warnings(
        UserWarning, "key test_long_key in extra_keywords is longer than 8 characters"
    ):
        cal_in.write_calfits(testfile, run_check=False, clobber=True)

    return


def test_input_flag_array_gain(tmp_path):
    """
    Test when data file has input flag array.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_input_flags.fits")
    cal_in.read_calfits(testfile)
    cal_in.input_flag_array = np.zeros(
        cal_in._input_flag_array.expected_shape(cal_in), dtype=bool
    )
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_input_flag_array_delay(tmp_path):
    """
    Test when data file has input flag array.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    write_file = str(tmp_path / "outtest_input_flags.fits")
    cal_in.read_calfits(testfile)
    cal_in.input_flag_array = np.zeros(
        cal_in._input_flag_array.expected_shape(cal_in), dtype=bool
    )
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_jones_gain(tmp_path):
    """
    Test when data file has more than one element in Jones matrix.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_jones.fits")
    cal_in.read_calfits(testfile)

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.gain_array = np.ones(
        cal_in._gain_array.expected_shape(cal_in), dtype=np.complex64
    )
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_jones_delay(tmp_path):
    """
    Test when data file has more than one element in Jones matrix.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    write_file = str(tmp_path / "outtest_jones.fits")
    cal_in.read_calfits(testfile)

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    cal_in.delay_array = np.ones(
        cal_in._delay_array.expected_shape(cal_in), dtype=np.float64
    )
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_readwriteread_total_quality_array(tmp_path):
    """
    Test when data file has a total quality array.

    Currently we have no such file, so we will artificially create one and
    check for internal consistency.
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_total_quality_array.fits")
    cal_in.read_calfits(testfile)

    # Create filler total quality array
    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del cal_in
    del cal_out

    # also test delay-type calibrations
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    write_file = str(tmp_path / "outtest_total_quality_array_delays.fits")
    cal_in.read_calfits(testfile)

    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
    del cal_in
    del cal_out


def test_total_quality_array_size():
    """
    Test that total quality array defaults to the proper size
    """

    cal_in = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    cal_in.read_calfits(testfile)

    # Create filler total quality array
    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    proper_shape = (cal_in.Nspws, cal_in.Nfreqs, cal_in.Ntimes, cal_in.Njones)
    assert cal_in.total_quality_array.shape == proper_shape
    del cal_in

    # also test delay-type calibrations
    cal_in = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")
    cal_in.read_calfits(testfile)

    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    proper_shape = (cal_in.Nspws, 1, cal_in.Ntimes, cal_in.Njones)
    assert cal_in.total_quality_array.shape == proper_shape
    del cal_in


def test_write_time_precision(tmp_path):
    """
    Test that times are being written to appropriate precision (see issue 311).
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(testfile)
    # overwrite time array to break old code
    dt = cal_in.integration_time / (24.0 * 60.0 * 60.0)
    cal_in.time_array = dt * np.arange(cal_in.Ntimes)
    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out


def test_read_noversion_history(tmp_path):
    """
    Test that version info gets added to the history if it's missing
    """
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")
    write_file2 = str(tmp_path / "outtest_omnical2.fits")
    cal_in.read_calfits(testfile)

    cal_in.write_calfits(write_file, clobber=True)

    fname = fits.open(write_file)
    data = fname[0].data
    primary_hdr = fname[0].header
    hdunames = uvutils._fits_indexhdus(fname)
    ant_hdu = fname[hdunames["ANTENNAS"]]

    primary_hdr["HISTORY"] = ""

    prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
    hdulist = fits.HDUList([prihdu, ant_hdu])

    hdulist.writeto(write_file2, overwrite=True)
    hdulist.close()

    cal_out.read_calfits(write_file2)
    assert cal_in == cal_out


def test_write_freq_spacing_not_channel_width(tmp_path):
    cal_in = UVCal()
    cal_out = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")
    cal_in.read_calfits(testfile)

    # select every other frequency -- then evenly spaced but doesn't match channel width
    cal_in.select(freq_chans=np.arange(0, 10, 2))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out.read_calfits(write_file)
    assert cal_in == cal_out
