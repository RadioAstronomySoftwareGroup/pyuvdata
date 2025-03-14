# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for calfits object"""

import os

import numpy as np
import pytest
from astropy.io import fits

import pyuvdata.utils.io.fits as fits_utils
from pyuvdata import UVCal, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

from ..utils.test_coordinates import frame_selenoid, selenoids
from . import extend_jones_axis, time_array_to_time_range


@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("quality", [True, False])
@pytest.mark.parametrize("time_range", [True, False])
def test_readwriteread(caltype, quality, time_range, gain_data, delay_data, tmp_path):
    """
    Omnical/Firstcal fits loopback test.

    Read in calfits file, write out new calfits file, read back in and check for
    object equality.
    """
    if caltype == "gain":
        cal_in = gain_data
    else:
        cal_in = delay_data

    if time_range:
        # can only have a calfits with time_range for one time
        cal_in.select(times=cal_in.time_array[0], inplace=True)
        cal_in = time_array_to_time_range(cal_in)

    if not quality:
        cal_in.quality_array = None
    # add total_quality_array so that can be tested as well
    cal_in.total_quality_array = np.ones(
        cal_in._total_quality_array.expected_shape(cal_in)
    )
    # add instrument and antenna_diameters
    cal_in.telescope.instrument = cal_in.telescope.name
    cal_in.telescope.antenna_diameters = (
        np.zeros((cal_in.telescope.Nants,), dtype=float) + 5.0
    )

    write_file = str(tmp_path / "outtest.fits")
    cal_in.write_calfits(write_file, clobber=True)
    with check_warnings(None):
        cal_out = UVCal.from_file(write_file)

    assert cal_out.filename == [os.path.basename(write_file)]

    assert cal_in == cal_out

    return


@pytest.mark.parametrize("selenoid", selenoids)
def test_moon_loopback(tmp_path, gain_data, selenoid):
    pytest.importorskip("lunarsky")
    from lunarsky import MoonLocation

    cal_in = gain_data

    enu_antpos = utils.ENU_from_ECEF(
        (cal_in.telescope.antenna_positions + cal_in.telescope._location.xyz()),
        center_loc=cal_in.telescope.location,
    )
    cal_in.telescope.location = MoonLocation.from_selenodetic(
        lat=cal_in.telescope.location.lat,
        lon=cal_in.telescope.location.lon,
        height=cal_in.telescope.location.height,
        ellipsoid=selenoid,
    )

    new_full_antpos = utils.ECEF_from_ENU(
        enu=enu_antpos, center_loc=cal_in.telescope.location
    )
    cal_in.telescope.antenna_positions = (
        new_full_antpos - cal_in.telescope._location.xyz()
    )
    cal_in.set_lsts_from_time_array()
    cal_in.check()

    write_file = str(tmp_path / "outtest.fits")
    cal_in.write_calfits(write_file, clobber=True)

    cal_out = UVCal.from_file(write_file)

    assert cal_in == cal_out

    # check in case xyz is missing
    write_file2 = str(tmp_path / "outtest_noxyz.fits")
    with fits.open(write_file) as fname:
        data = fname[0].data
        primary_hdr = fname[0].header
        hdunames = fits_utils._indexhdus(fname)
        ant_hdu = fname[hdunames["ANTENNAS"]]

        primary_hdr.pop("ARRAYX")
        primary_hdr.pop("ARRAYY")
        primary_hdr.pop("ARRAYZ")

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])

        hdulist.writeto(write_file2, overwrite=True)

    cal_out = UVCal.from_file(write_file2)
    assert cal_out == cal_in


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.skipif(
    len(frame_selenoid) > 1, reason="Test only when lunarsky not installed."
)
def test_calfits_no_moon(gain_data, tmp_path):
    """Check errors when reading uvfits with MCMF without lunarsky."""
    write_file = str(tmp_path / "outtest.calfits")
    write_file2 = str(tmp_path / "outtest2.calfits")

    gain_data.write_calfits(write_file)

    with fits.open(write_file, memmap=True) as fname:
        data = fname[0].data
        primary_hdr = fname[0].header
        hdunames = fits_utils._indexhdus(fname)
        ant_hdu = fname[hdunames["ANTENNAS"]]

        primary_hdr["FRAME"] = "mcmf"
        primary_hdr.pop("ARRAYY")
        primary_hdr.pop("ARRAYZ")

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])

        hdulist.writeto(write_file2, overwrite=True)

    msg = "Need to install `lunarsky` package to work with MCMF frame."
    with pytest.raises(ValueError, match=msg):
        UVCal.from_file(write_file2)


def test_write_inttime_equal_timediff(gain_data, tmp_path):
    """
    test writing out object with integration times close to time diffs
    """
    cal_in = gain_data

    time_diffs = np.diff(cal_in.time_array)

    gain_data.integration_time = np.full(
        cal_in.Ntimes, np.mean(time_diffs) * (24.0 * 60.0**2)
    )

    write_file = str(tmp_path / "outtest.fits")
    cal_in.write_calfits(write_file, clobber=True)

    with check_warnings(None):
        cal_out = UVCal.from_file(write_file)

    assert cal_in == cal_out

    return


@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
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
    cal3 = UVCal.from_file(testfile, read_data=False)
    assert cal2 == cal3

    return


def test_readwriteread_no_freq_range(gain_data, tmp_path):
    # test without freq_range parameter
    cal_in = gain_data
    write_file = str(tmp_path / "outtest_omnical.fits")

    cal_in.freq_range = None
    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file)
    assert cal_in == cal_out

    return


@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
def test_readwriteread_no_time_range(tmp_path):
    # test without time_range parameter
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    write_file = str(tmp_path / "outtest_omnical.fits")

    cal_in = UVCal.from_file(testfile)
    cal_in.time_range = None
    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file)
    assert cal_in == cal_out


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
        hdunames = fits_utils._indexhdus(fname)
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
        UVCal.from_file(write_file2)

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
        hdunames = fits_utils._indexhdus(fname)
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
        UVCal.from_file(write_file2)

    return


def test_latlonalt_noxyz(gain_data, tmp_path):
    cal_in = gain_data
    write_file = str(tmp_path / "outtest.fits")
    write_file2 = str(tmp_path / "outtest_noxyz.fits")

    cal_in.write_calfits(write_file)

    with fits.open(write_file) as fname:
        data = fname[0].data
        primary_hdr = fname[0].header
        hdunames = fits_utils._indexhdus(fname)
        ant_hdu = fname[hdunames["ANTENNAS"]]

        primary_hdr.pop("ARRAYX")
        primary_hdr.pop("ARRAYY")
        primary_hdr.pop("ARRAYZ")

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])

        hdulist.writeto(write_file2, overwrite=True)

    cal_out = UVCal.from_file(write_file2)
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
    cal_out = UVCal.from_file(testfile)

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
    cal_in.check()
    with pytest.raises(TypeError, match=error_msg):
        cal_in.write_calfits(testfile, run_check=False)

    return


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
    cal_out = UVCal.from_file(write_file)
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
    cal_out = UVCal.from_file(write_file)
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
    cal_out = UVCal.from_file(write_file)
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
        hdunames = fits_utils._indexhdus(fname)
        ant_hdu = fname[hdunames["ANTENNAS"]]

        primary_hdr["HISTORY"] = ""

        prihdu = fits.PrimaryHDU(data=data, header=primary_hdr)
        hdulist = fits.HDUList([prihdu, ant_hdu])

        hdulist.writeto(write_file2, overwrite=True)
        hdulist.close()

    cal_out = UVCal.from_file(write_file2)
    assert cal_in == cal_out


@pytest.mark.filterwarnings("ignore:Selected frequencies are not contiguous")
def test_write_freq_spacing_not_channel_width(gain_data, tmp_path):
    cal_in = gain_data

    write_file = str(tmp_path / "outtest_freqspace.fits")

    # select every other frequency -- then evenly spaced but doesn't match channel width
    cal_in.select(freq_chans=np.arange(0, 10, 2))

    cal_in.write_calfits(write_file, clobber=True)
    cal_out = UVCal.from_file(write_file)
    assert cal_in == cal_out


@pytest.mark.parametrize(
    ["caltype", "param_dict"],
    [
        [
            "gain",
            {
                "antenna_nums": np.array([65, 96, 9, 97, 89, 22, 20, 72]),
                "freq_chans": np.arange(2, 9),
                "times": np.arange(2, 5),
                "jones": ["xx", "yy"],
            },
        ],
        [
            "delay",
            {
                "antenna_nums": np.array([65, 96, 9, 97, 89, 22, 20, 72]),
                "times": 0,
                "jones": -5,
            },
        ],
    ],
)
def test_calfits_partial_read(gain_data, delay_data, tmp_path, caltype, param_dict):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    orig_time_array = calobj.time_array

    for par, val in param_dict.items():
        if par == "times":
            param_dict[par] = orig_time_array[val]

    extend_jones_axis(calobj, total_quality=False)

    write_file = str(tmp_path / "outtest.calfits")
    calobj.write_calfits(write_file, clobber=True)

    calobj2 = calobj.copy()

    calobj2.select(**param_dict)

    msg = [
        'Warning: select on read keyword set, but file_type is "calfits" which '
        "does not support select on read. Entire file will be read and then select "
        "will be performed"
    ]

    with check_warnings(UserWarning, match=msg):
        calobj3 = UVCal.from_file(write_file, **param_dict)

    assert calobj2 == calobj3


def test_extra_keywords_warnings(gain_data, tmp_path):
    cal_in = gain_data
    testfile = str(tmp_path / "outtest_extrakwd_warn.fits")

    # check for warnings with extra_keywords keys that are too long
    cal_in.extra_keywords["test_long_key"] = True
    with check_warnings(None):
        cal_in.check()

    with check_warnings(
        UserWarning, "key test_long_key in extra_keywords is longer than 8 characters"
    ):
        cal_in.write_calfits(testfile, run_check=False, clobber=True)
