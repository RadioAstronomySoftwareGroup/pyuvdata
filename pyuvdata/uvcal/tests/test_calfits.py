# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for calfits object

"""
import os

import numpy as np
import pytest
from astropy.io import fits

import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata import UVCal
from pyuvdata.data import DATA_PATH
from pyuvdata.uvcal.uvcal import _future_array_shapes_warning

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values",
    "ignore:antenna_positions are not set or are being overwritten. Using known values",
)


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("quality", [True, False])
@pytest.mark.parametrize("input_flag_array", [True, False])
def test_readwriteread(
    future_shapes, caltype, quality, input_flag_array, gain_data, delay_data, tmp_path
):
    """
    Omnical/Firstcal fits loopback test.

    Read in calfits file, write out new calfits file, read back in and check for
    object equality.
    """
    if caltype == "gain":
        cal_in = gain_data
    else:
        cal_in = delay_data

    if not future_shapes:
        cal_in.use_current_array_shapes()

    if not quality:
        cal_in.quality_array = None
    if input_flag_array:
        cal_in.input_flag_array = cal_in.flag_array

    write_file = str(tmp_path / "outtest.fits")
    cal_in.write_calfits(write_file, clobber=True)
    if not future_shapes:
        warn_type = [DeprecationWarning]
        warn_msg = [_future_array_shapes_warning]
    elif caltype == "delay":
        warn_type = [UserWarning]
        warn_msg = ["When converting a delay-style cal to future array shapes the"]
    else:
        warn_type = []
        warn_msg = []

    if input_flag_array:
        warn_type.append(DeprecationWarning)
        warn_msg.append(
            "The input_flag_array is deprecated and will be removed in version 2.5"
        )

    if len(warn_type) == 0:
        warn_type = None
        warn_msg = ""

    with uvtest.check_warnings(warn_type, match=warn_msg):
        cal_out = UVCal.from_file(write_file, use_future_array_shapes=future_shapes)

    assert cal_out.filename == [os.path.basename(write_file)]

    assert cal_in == cal_out

    return


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_write_inttime_equal_timediff(future_shapes, gain_data, delay_data, tmp_path):
    """
    test writing out object with integration times close to time diffs
    """
    cal_in = gain_data

    time_diffs = np.diff(cal_in.time_array)

    gain_data.integration_time = np.full(
        cal_in.Ntimes, np.mean(time_diffs) * (24.0 * 60.0**2)
    )

    if not future_shapes:
        cal_in.use_current_array_shapes()

    write_file = str(tmp_path / "outtest.fits")
    cal_in.write_calfits(write_file, clobber=True)

    if not future_shapes:
        warn_type = DeprecationWarning
    else:
        warn_type = None

    with uvtest.check_warnings(warn_type, match=_future_array_shapes_warning):
        cal_out = UVCal.from_file(write_file, use_future_array_shapes=future_shapes)

    assert cal_in == cal_out

    return


@pytest.mark.parametrize(
    "filein,caltype",
    [
        ("zen.2457698.40355.xx.gain.calfits", "gain"),
        ("zen.2457698.40355.xx.delay.calfits", "delay"),
    ],
)
def test_read_metadata_only(filein, caltype, gain_data, delay_data):
    """
    check that metadata only reads work
    """
    if caltype == "gain":
        cal_in = gain_data
    else:
        cal_in = delay_data

    # check that metadata only reads work
    cal2 = cal_in.copy(metadata_only=True)
    testfile = os.path.join(DATA_PATH, filein)
    cal3 = UVCal.from_file(testfile, read_data=False, use_future_array_shapes=True)
    assert cal2 == cal3

    return


def test_readwriteread_no_freq_range(gain_data, tmp_path):
    # test without freq_range parameter
    cal_in = gain_data
    write_file = str(tmp_path / "outtest_omnical.fits")

    cal_in.freq_range = None
    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file, use_future_array_shapes=True)
    assert cal_in == cal_out

    return


def test_readwriteread_no_time_range(tmp_path):
    # test without time_range parameter
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")

    cal_in = UVCal.from_file(testfile, use_future_array_shapes=True)
    cal_in.time_range = None
    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file, use_future_array_shapes=True)
    assert cal_in == cal_out

    return


def test_error_unknown_cal_type(delay_data, tmp_path):
    """
    Test an error is raised when writing an unknown cal type.
    """
    cal_in = delay_data
    write_file = str(tmp_path / "outtest_firstcal.fits")

    with uvtest.check_warnings(
        DeprecationWarning,
        match="Setting the cal_type to 'unknown' is deprecated. This will become an "
        "error in version 2.5",
    ):
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
def test_fits_header_errors_delay(delay_data, tmp_path, header_dict, error_msg):
    # change values for various axes in flag and total quality hdus to not
    # match primary hdu
    cal_in = delay_data
    write_file = str(tmp_path / "outtest_firstcal.fits")
    write_file2 = str(tmp_path / "outtest_firstcal2.fits")

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

    with fits.open(write_file) as fname:
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
        UVCal.from_file(write_file2, use_future_array_shapes=True)

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
def test_fits_header_errors_gain(gain_data, tmp_path, header_dict, error_msg):
    # repeat for gain type file
    cal_in = gain_data
    write_file = str(tmp_path / "outtest_omnical.fits")
    write_file2 = str(tmp_path / "outtest_omnical2.fits")

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

    with fits.open(write_file) as fname:
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
        UVCal.from_file(write_file2, use_future_array_shapes=True)

    return


def test_latlonalt_noxyz(gain_data, tmp_path):
    cal_in = gain_data
    write_file = str(tmp_path / "outtest.fits")
    write_file2 = str(tmp_path / "outtest_noxyz.fits")

    cal_in.write_calfits(write_file)

    with fits.open(write_file) as fname:
        data = fname[0].data
        primary_hdr = fname[0].header
        hdunames = uvutils._fits_indexhdus(fname)
        ant_hdu = fname[hdunames["ANTENNAS"]]

        primary_hdr.pop("ARRAYX")
        primary_hdr.pop("ARRAYY")
        primary_hdr.pop("ARRAYZ")

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])

        hdulist.writeto(write_file2, overwrite=True)

    cal_out = UVCal.from_file(write_file2, use_future_array_shapes=True)
    assert cal_out == cal_in


@pytest.mark.parametrize(
    "kwd1,kwd2,val1,val2",
    [
        ["keyword1", "keyword2", True, False],
        ["keyword1", "keyword2", np.int64(5), 7],
        ["keyword1", "keyword2", np.int64(5.3), 6.9],
        ["keyword1", "keyword2", np.complex64(5.3 + 1.2j), 6.9 + 4.6j],
        [
            "keyword1",
            "comment",
            "short comment",
            "this is a very long comment that will "
            "be broken into several lines\nif "
            "everything works properly.",
        ],
    ],
)
def test_extra_keywords(gain_data, kwd1, kwd2, val1, val2, tmp_path):
    cal_in = gain_data
    testfile = str(tmp_path / "outtest_extrakws.fits")

    # check handling of boolean keywords
    cal_in.extra_keywords[kwd1] = val1
    cal_in.extra_keywords[kwd2] = val2
    cal_in.write_calfits(testfile, clobber=True)
    cal_out = UVCal.from_file(testfile, use_future_array_shapes=True)

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
def test_extra_keywords_errors(gain_data, tmp_path, ex_val, error_msg):
    cal_in = gain_data
    testfile = str(tmp_path / "outtest_extrakwd_err.fits")

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    keyword = list(ex_val.keys())[0]
    val = ex_val[keyword]
    cal_in.extra_keywords[keyword] = val
    with uvtest.check_warnings(
        UserWarning, match=f"{keyword} in extra_keywords is a list, array or dict"
    ):
        cal_in.check()
    with pytest.raises(TypeError, match=error_msg):
        cal_in.write_calfits(testfile, run_check=False)

    return


def test_extra_keywords_warnings(gain_data, tmp_path):
    cal_in = gain_data
    testfile = str(tmp_path / "outtest_extrakwd_warn.fits")

    # check for warnings with extra_keywords keys that are too long
    cal_in.extra_keywords["test_long_key"] = True
    with uvtest.check_warnings(
        UserWarning,
        match="key test_long_key in extra_keywords is longer than 8 characters",
    ):
        cal_in.check()
    with uvtest.check_warnings(
        UserWarning, "key test_long_key in extra_keywords is longer than 8 characters"
    ):
        cal_in.write_calfits(testfile, run_check=False, clobber=True)

    return


@pytest.mark.filterwarnings("ignore:When converting a delay-style cal to future array")
@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_input_flag_array(caltype, gain_data, delay_data, tmp_path):
    """
    Test when data file has input flag array.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    if caltype == "gain":
        cal_in = gain_data
    else:
        cal_in = delay_data

    write_file = str(tmp_path / "outtest_input_flags.fits")
    cal_in.input_flag_array = np.zeros(
        cal_in._input_flag_array.expected_shape(cal_in), dtype=bool
    )
    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file, use_future_array_shapes=True)
    assert cal_in == cal_out


@pytest.mark.filterwarnings("ignore:When converting a delay-style cal to future array")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_jones(caltype, gain_data, delay_data, tmp_path):
    """
    Test when data file has more than one element in Jones matrix.

    Currently we do not have a testfile, so we will artifically create one
    and check for internal consistency.
    """
    if caltype == "gain":
        cal_in = gain_data
    else:
        cal_in = delay_data

    write_file = str(tmp_path / "outtest_jones.fits")

    # Create filler jones info
    cal_in.jones_array = np.array([-5, -6, -7, -8])
    cal_in.Njones = 4
    cal_in.flag_array = np.zeros(cal_in._flag_array.expected_shape(cal_in), dtype=bool)
    if caltype == "gain":
        cal_in.gain_array = np.ones(
            cal_in._gain_array.expected_shape(cal_in), dtype=np.complex64
        )
    else:
        cal_in.delay_array = np.ones(
            cal_in._delay_array.expected_shape(cal_in), dtype=np.float64
        )
    cal_in.quality_array = np.zeros(cal_in._quality_array.expected_shape(cal_in))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file, use_future_array_shapes=True)
    assert cal_in == cal_out


@pytest.mark.filterwarnings("ignore:When converting a delay-style cal to future array")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_readwriteread_total_quality_array(caltype, gain_data, delay_data, tmp_path):
    """
    Test when data file has a total quality array.

    Currently we have no such file, so we will artificially create one and
    check for internal consistency.
    """
    if caltype == "gain":
        cal_in = gain_data
    else:
        cal_in = delay_data

    write_file = str(tmp_path / "outtest_total_quality_array.fits")

    # Create filler total quality array
    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file, use_future_array_shapes=True)
    assert cal_in == cal_out
    del cal_in
    del cal_out


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_total_quality_array_size(caltype, gain_data, delay_data):
    """
    Test that total quality array defaults to the proper size
    """
    if caltype == "gain":
        cal_in = gain_data
    else:
        cal_in = delay_data

    # Create filler total quality array
    cal_in.total_quality_array = np.zeros(
        cal_in._total_quality_array.expected_shape(cal_in)
    )

    if caltype == "gain":
        proper_shape = (cal_in.Nfreqs, cal_in.Ntimes, cal_in.Njones)
    else:
        proper_shape = (1, cal_in.Ntimes, cal_in.Njones)
    assert cal_in.total_quality_array.shape == proper_shape
    del cal_in


def test_write_time_precision(gain_data, tmp_path):
    """
    Test that times are being written to appropriate precision (see issue 311).
    """
    cal_in = gain_data

    write_file = str(tmp_path / "outtest_time_prec.fits")
    # overwrite time array to break old code
    dt = cal_in.integration_time / (24.0 * 60.0 * 60.0)
    t0 = cal_in.time_array[0] + dt * 3
    cal_in.time_array = dt * np.arange(cal_in.Ntimes) + t0
    if cal_in.lst_array is not None:
        cal_in.set_lsts_from_time_array()
    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file, use_future_array_shapes=True)
    assert cal_in == cal_out


def test_read_noversion_history(gain_data, tmp_path):
    """
    Test that version info gets added to the history if it's missing
    """
    cal_in = gain_data

    write_file = str(tmp_path / "outtest_nover.fits")
    write_file2 = str(tmp_path / "outtest_nover2.fits")

    cal_in.write_calfits(write_file, clobber=True)

    with fits.open(write_file) as fname:
        data = fname[0].data
        primary_hdr = fname[0].header
        hdunames = uvutils._fits_indexhdus(fname)
        ant_hdu = fname[hdunames["ANTENNAS"]]

        primary_hdr["HISTORY"] = ""

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])

        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    cal_out = UVCal.from_file(write_file2, use_future_array_shapes=True)
    assert cal_in == cal_out


@pytest.mark.filterwarnings("ignore:Selected frequencies are not contiguous")
def test_write_freq_spacing_not_channel_width(gain_data, tmp_path):
    cal_in = gain_data

    write_file = str(tmp_path / "outtest_freqspace.fits")

    # select every other frequency -- then evenly spaced but doesn't match channel width
    cal_in.select(freq_chans=np.arange(0, 10, 2))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file, use_future_array_shapes=True)
    assert cal_in == cal_out
