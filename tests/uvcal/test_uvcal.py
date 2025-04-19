# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvcal object."""

import copy
import itertools
import os
import re

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import pyuvdata.utils.io.fits as fits_utils
from pyuvdata import UVCal, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

from . import extend_jones_axis, time_array_to_time_range

pytestmark = pytest.mark.filterwarnings(
    "ignore:key CASA_Version in extra_keywords is longer than 8 characters",
    "ignore:telescope_location, antenna_positions, mount_type, antenna_diameters are "
    "not set or are being overwritten. telescope_location, antenna_positions, "
    "mount_type, antenna_diameters are set using values from known telescopes "
    "for HERA.",
)


@pytest.fixture(scope="session")
def uvcal_phase_center_main(gain_data_main):
    gain_copy = gain_data_main.copy()
    gain_copy._set_sky()
    gain_copy.ref_antenna_name = gain_copy.telescope.antenna_names[0]
    gain_copy.sky_catalog = "unknown"

    # Copying the catalog from sma_test.mir
    gain_copy.phase_center_catalog = {
        1: {
            "cat_name": "3c84",
            "cat_type": "sidereal",
            "cat_lon": 0.8718035968995141,
            "cat_lat": 0.7245157752262148,
            "cat_frame": "icrs",
            "cat_epoch": 2000.0,
            "cat_times": None,
            "cat_pm_ra": None,
            "cat_pm_dec": None,
            "cat_vrad": None,
            "cat_dist": None,
            "info_source": "file",
        }
    }
    gain_copy.phase_center_id_array = np.ones(gain_copy.Ntimes, dtype=int)
    gain_copy.Nphase = 1

    yield gain_copy


@pytest.fixture(scope="function")
def uvcal_phase_center(uvcal_phase_center_main):
    gain_copy = uvcal_phase_center_main.copy()

    yield gain_copy


@pytest.fixture(scope="function")
def uvcal_data():
    """Set up some uvcal iter tests."""
    required_properties = [
        "Nfreqs",
        "Njones",
        "Ntimes",
        "Nspws",
        "Nants_data",
        "wide_band",
        "ant_array",
        "telescope",
        "freq_array",
        "channel_width",
        "spw_array",
        "jones_array",
        "integration_time",
        "gain_convention",
        "flag_array",
        "cal_type",
        "cal_style",
        "history",
    ]
    required_parameters = ["_" + prop for prop in required_properties]

    extra_properties = [
        "lst_array",
        "lst_range",
        "gain_array",
        "delay_array",
        "sky_catalog",
        "ref_antenna_name",
        "Nsources",
        "baseline_range",
        "diffuse_model",
        "time_range",
        "time_array",
        "freq_range",
        "flex_spw_id_array",
        "flex_jones_array",
        "observer",
        "git_origin_cal",
        "git_hash_cal",
        "quality_array",
        "total_quality_array",
        "extra_keywords",
        "gain_scale",
        "pol_convention",
        "filename",
        "scan_number_array",
        "phase_center_catalog",
        "phase_center_id_array",
        "Nphase",
        "ref_antenna_array",
    ]
    extra_parameters = ["_" + prop for prop in extra_properties]

    other_attributes = [
        "pyuvdata_version_str",
        "telescope_name",
        "telescope_location",
        "instrument",
        "Nants_telescope",
        "antenna_names",
        "antenna_numbers",
        "antenna_positions",
        "x_orientation",
        "antenna_diameters",
        "telescope_location_lat_lon_alt",
        "telescope_location_lat_lon_alt_degrees",
    ]

    uv_cal_object = UVCal()

    # yields the data we need but will continue to the del call after tests
    yield (
        uv_cal_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_attributes,
    )

    # some post-test object cleanup
    del uv_cal_object
    return


def test_parameter_iter(uvcal_data):
    """Test expected parameters."""
    (uv_cal_object, required_parameters, _, extra_parameters, _, _) = uvcal_data
    all_params = []
    for prop in uv_cal_object:
        all_params.append(prop)
    for a in required_parameters + extra_parameters:
        assert a in all_params, (
            "expected attribute " + a + " not returned in object iterator"
        )


def test_required_parameter_iter(uvcal_data):
    """Test expected required parameters."""
    (uv_cal_object, required_parameters, _, _, _, _) = uvcal_data
    # at first it's a metadata_only object, so need to modify required_parameters
    required = []
    for prop in uv_cal_object.required():
        required.append(prop)
    expected_required = copy.copy(required_parameters)
    expected_required.remove("_flag_array")
    for a in expected_required:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )

    uv_cal_object.flag_array = 1
    assert uv_cal_object.metadata_only is False
    required = []
    for prop in uv_cal_object.required():
        required.append(prop)
    for a in required_parameters:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )


def test_unexpected_parameters(uvcal_data):
    """Test for extra parameters."""
    (uv_cal_object, required_parameters, _, extra_parameters, _, _) = uvcal_data
    expected_parameters = required_parameters + extra_parameters
    attributes = [i for i in uv_cal_object.__dict__ if i[0] == "_"]
    for a in attributes:
        assert a in expected_parameters, "unexpected parameter " + a + " found in UVCal"


def test_unexpected_attributes(uvcal_data):
    """Test for extra attributes."""
    (uv_cal_object, _, required_properties, _, extra_properties, other_attributes) = (
        uvcal_data
    )
    expected_attributes = required_properties + extra_properties + other_attributes
    attributes = [i for i in uv_cal_object.__dict__ if i[0] != "_"]
    for a in attributes:
        assert a in expected_attributes, "unexpected attribute " + a + " found in UVCal"


def test_properties(uvcal_data):
    """Test that properties can be get and set properly."""
    (
        uv_cal_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        _,
    ) = uvcal_data
    prop_dict = dict(
        list(
            zip(
                required_properties + extra_properties,
                required_parameters + extra_parameters,
                strict=True,
            )
        )
    )
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(uv_cal_object, k, rand_num)
        this_param = getattr(uv_cal_object, v)
        try:
            assert rand_num == this_param.value
        except AssertionError:
            print(f"setting {k} to a random number failed")
            raise


def test_equality(gain_data):
    """Basic equality test"""
    assert gain_data == gain_data


def test_check(gain_data, delay_data):
    """Test that parameter checks run properly"""
    assert gain_data.check()

    gain_data.freq_range = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="The freq_range attribute should not be set"):
        gain_data.check()

    assert delay_data.check()

    delay_data.flex_spw_id_array = np.array([0])
    with pytest.raises(ValueError, match="The flex_spw_id_array attribute should not"):
        delay_data.check()

    delay_data.channel_width = np.array([1.0])
    with pytest.raises(ValueError, match="The channel_width attribute should not be"):
        delay_data.check()

    delay_data.freq_array = np.array([1.0])
    with pytest.raises(ValueError, match="The freq_array attribute should not be set"):
        delay_data.check()


def test_check_flag_array(gain_data):
    gain_data.flag_array = np.ones((gain_data.flag_array.shape), dtype=int)

    with pytest.raises(
        ValueError, match="UVParameter _flag_array is not the appropriate type."
    ):
        gain_data.check()


def test_check_time_range_errors(gain_data):
    calobj = time_array_to_time_range(gain_data)
    original_range = copy.copy(calobj.time_range)

    calobj.time_range[1, 1] = calobj.time_range[0, 0]
    with pytest.raises(
        ValueError,
        match="The time ranges are not well-formed, some stop times are after start "
        "times.",
    ):
        calobj.check()

    calobj.time_range = original_range
    calobj.time_range[0, 1] = calobj.time_range[1, 1]
    with pytest.raises(ValueError, match="Some time_ranges overlap"):
        calobj.check()


@pytest.mark.parametrize("time_range", [True, False])
def test_check_lst(gain_data, time_range):
    gain_time_range = time_array_to_time_range(gain_data)
    if time_range:
        calobj = gain_time_range
        tparam = "time_range"
        lst_param = "lst_range"
        calobj.lst_range = None
    else:
        calobj = gain_data
        tparam = "time_array"
        lst_param = "lst_array"
        calobj.lst_array = None

    with pytest.raises(ValueError, match="Either lst_array or lst_range must be set."):
        calobj.check()

    if time_range:
        calobj.lst_array = gain_data.lst_array
    else:
        calobj.lst_range = gain_time_range.lst_range

    with pytest.raises(
        ValueError, match=f"If {tparam} is present, {lst_param} must also be present."
    ):
        calobj.check()


@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_array_shape_errors(caltype, gain_data, delay_data, wideband_gain):
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
        calobj_wideband = wideband_gain
        calobj_wideband.select(spws=1)
    else:
        calobj = delay_data

    calobj.integration_time[-1] = calobj.integration_time[0] * 2.0
    if caltype == "delay":
        calobj.Nfreqs = 2
        with pytest.raises(
            ValueError, match="Nfreqs is required to be 1 for wide_band cals"
        ):
            calobj.check()
        calobj.Nfreqs = 1
    else:
        calobj.check()

    with pytest.raises(
        ValueError, match="The integration times are variable. The calfits format"
    ):
        calobj.write_calfits("foo")

    if caltype == "gain":
        calobj2.freq_array[-1] *= 2.0
        calobj2.check()
        with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
            calobj2._check_freq_spacing()


def test_get_time_array(gain_data):
    calobj = gain_data
    orig_time_array = copy.copy(calobj.time_array)

    time_array = calobj.get_time_array()
    np.testing.assert_allclose(
        time_array,
        orig_time_array,
        rtol=calobj._time_array.tols[0],
        atol=calobj._time_array.tols[1],
    )

    calobj = time_array_to_time_range(calobj)
    time_array = calobj.get_time_array()
    np.testing.assert_allclose(
        time_array,
        orig_time_array,
        rtol=calobj._time_array.tols[0],
        atol=calobj._time_array.tols[1],
    )


def test_lst_array(gain_data):
    calobj = gain_data
    orig_lst_array = copy.copy(calobj.lst_array)

    lst_array = calobj.get_lst_array()
    np.testing.assert_allclose(
        lst_array,
        orig_lst_array,
        rtol=calobj._lst_array.tols[0],
        atol=calobj._lst_array.tols[1],
    )

    calobj = time_array_to_time_range(calobj)
    lst_array = calobj.get_lst_array()
    np.testing.assert_allclose(
        lst_array,
        orig_lst_array,
        rtol=calobj._lst_array.tols[0],
        atol=calobj._lst_array.tols[1],
    )


def test_unknown_telescopes(gain_data, tmp_path):
    calobj = gain_data

    write_file = str(tmp_path / "test.calfits")
    write_file2 = str(tmp_path / "test2.calfits")
    calobj.write_calfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
        primary_hdu = hdu_list[0]
        primary_hdr = primary_hdu.header.copy()
        del primary_hdr["ARRAYX"]
        del primary_hdr["ARRAYY"]
        del primary_hdr["ARRAYZ"]
        del primary_hdr["LAT"]
        del primary_hdr["LON"]
        del primary_hdr["ALT"]
        primary_hdr["TELESCOP"] = "foo"
        primary_hdu.header = primary_hdr

        ant_hdu = hdu_list[hdunames["ANTENNAS"]]

        table = Table(ant_hdu.data)
        del table["ANTXYZ"]
        ant_hdu = fits.BinTableHDU(table)
        ant_hdu.header["EXTNAME"] = "ANTENNAS"

        hdulist = fits.HDUList([primary_hdu, ant_hdu])

        hdulist.writeto(write_file2)
        hdulist.close()

    with (
        pytest.raises(
            ValueError,
            match="Required UVParameter _antenna_positions has not been set.",
        ),
        check_warnings(
            UserWarning,
            match="Telescope foo is not in astropy_sites or known_telescopes_dict.",
        ),
    ):
        UVCal.from_file(write_file2)
    with check_warnings(
        UserWarning,
        match="Telescope foo is not in astropy_sites or known_telescopes_dict.",
    ):
        UVCal.from_file(write_file2, run_check=False)


def test_nants_data_telescope_larger(gain_data):
    # make sure it's okay for Nants_telescope to be strictly greater than Nants_data
    gain_data.telescope.Nants += 1
    # add dummy information for "new antenna" to pass object check
    gain_data.telescope.antenna_names = np.concatenate(
        (gain_data.telescope.antenna_names, ["dummy_ant"])
    )
    gain_data.telescope.antenna_numbers = np.concatenate(
        (gain_data.telescope.antenna_numbers, [20])
    )
    gain_data.telescope.antenna_positions = np.concatenate(
        (gain_data.telescope.antenna_positions, np.zeros((1, 3), dtype=float))
    )
    gain_data.telescope.feed_array = np.concatenate(
        (gain_data.telescope.feed_array, np.array([["x"]], dtype=str))
    )
    gain_data.telescope.mount_type = gain_data.telescope.mount_type + ["fixed"]
    gain_data.telescope.feed_angle = np.concatenate(
        (gain_data.telescope.feed_angle, np.full((1, 1), np.pi / 2, dtype=float))
    )
    if gain_data.telescope.antenna_diameters is not None:
        gain_data.telescope.antenna_diameters = np.concatenate(
            (gain_data.telescope.antenna_diameters, np.ones((1,), dtype=float))
        )

    assert gain_data.check()


def test_ant_array_not_in_antnums(gain_data):
    # make sure an error is raised if antennas with data not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    gain_data.telescope.antenna_names = gain_data.telescope.antenna_names[1:]
    gain_data.telescope.antenna_numbers = gain_data.telescope.antenna_numbers[1:]
    gain_data.telescope.antenna_positions = gain_data.telescope.antenna_positions[1:, :]
    gain_data.telescope.feed_array = gain_data.telescope.feed_array[1:, :]
    gain_data.telescope.feed_angle = gain_data.telescope.feed_angle[1:, :]
    gain_data.telescope.mount_type = gain_data.telescope.mount_type[1:]
    if gain_data.telescope.antenna_diameters is not None:
        gain_data.telescope.antenna_diameters = gain_data.telescope.antenna_diameters[
            1:
        ]
    gain_data.telescope.Nants = gain_data.telescope.antenna_numbers.size
    with pytest.raises(ValueError) as cm:
        gain_data.check()
    assert str(cm.value).startswith(
        "All antennas in ant_array must be in antenna_numbers"
    )


def test_set_gain():
    delay_data = UVCal()
    delay_data._set_gain()
    assert delay_data._gain_array.required
    assert not delay_data._delay_array.required
    assert delay_data._gain_array.form == delay_data._flag_array.form
    assert delay_data._gain_array.form == delay_data._quality_array.form


def test_set_delay():
    gain_data = UVCal()
    gain_data._set_delay()
    assert gain_data._delay_array.required
    assert not gain_data._gain_array.required
    assert gain_data._gain_array.form == gain_data._flag_array.form
    assert gain_data._delay_array.form == gain_data._quality_array.form


def test_set_sky():
    gain_data = UVCal()
    gain_data._set_sky()
    assert gain_data._sky_catalog.required
    assert gain_data._ref_antenna_name.required


def test_set_redundant():
    gain_data = UVCal()
    gain_data._set_redundant()
    assert not gain_data._sky_catalog.required
    assert not gain_data._ref_antenna_name.required


def test_convert_filetype():
    gain_data = UVCal()
    # error testing
    with pytest.raises(ValueError, match="filetype must be calh5, calfits, or ms."):
        gain_data._convert_to_filetype("uvfits")


def test_error_metadata_only_write(gain_data, tmp_path):
    calobj = gain_data.copy(metadata_only=True)

    out_file = os.path.join(tmp_path, "outtest.calfits")
    with pytest.raises(ValueError, match="Cannot write out metadata only objects to a"):
        calobj.write_calfits(out_file)


def test_flexible_spw(gain_data):
    calobj = gain_data

    # check that this check passes on single-window objects
    calobj._check_flex_spw_contiguous()

    # check warning if flex_spw_id_array is not set
    calobj.flex_spw_id_array = None
    with pytest.raises(ValueError, match="Required UVParameter _flex_spw_id_array"):
        calobj.check()

    # first just make one spw and check that object still passes check
    calobj.channel_width = (
        np.zeros(calobj.Nfreqs, dtype=np.float64) + calobj.channel_width
    )
    calobj.flex_spw_id_array = np.zeros(calobj.Nfreqs, dtype=int)
    calobj.check()

    # now make two
    calobj.Nspws = 2
    calobj.spw_array = np.array([1, 2])

    calobj.flex_spw_id_array = np.concatenate(
        (
            np.ones(calobj.Nfreqs // 2, dtype=int),
            np.full(calobj.Nfreqs // 2, 5, dtype=int),
        )
    )
    with pytest.raises(
        ValueError,
        match="All values in the flex_spw_id_array must exist in the spw_array.",
    ):
        calobj.check()

    calobj.flex_spw_id_array = np.concatenate(
        (
            np.ones(calobj.Nfreqs // 2, dtype=int),
            np.full(calobj.Nfreqs // 2, 2, dtype=int),
        )
    )
    calobj.check()

    calobj._check_flex_spw_contiguous()

    # now mix them up
    calobj.Nspws = 2
    calobj.flex_spw_id_array = np.concatenate(
        (
            np.ones(2, dtype=int),
            np.full(2, 2, dtype=int),
            np.ones(2, dtype=int),
            np.full(2, 2, dtype=int),
            np.ones(2, dtype=int),
        )
    )
    calobj.spw_array = np.array([1, 2])
    calobj.check()

    with pytest.raises(
        ValueError, match="Channels from different spectral windows are interspersed"
    ):
        calobj._check_flex_spw_contiguous()


@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
@pytest.mark.parametrize("convention", ["minus", "plus"])
@pytest.mark.parametrize("same_freqs", [True, False])
def test_convert_to_gain(convention, same_freqs, delay_data):
    delay_obj = delay_data

    freq_array = np.arange(30) * 1e6 + 1e8
    channel_width = np.full(30, 1e6)
    if not same_freqs:
        # try with different number and same number but different values
        if convention == "minus":
            freq_array = freq_array[:15]
            channel_width = channel_width[:15]
        else:
            freq_array[2] = freq_array[2] + 1e6

    # test passing a 1 element array for channel width
    assert channel_width.size == freq_array.size

    new_gain_obj = delay_obj.copy()
    tqa_size = new_gain_obj.delay_array.shape[1:]
    new_gain_obj.total_quality_array = np.ones(tqa_size)

    new_gain_obj2 = new_gain_obj.copy()

    new_gain_obj.convert_to_gain(
        freq_array=freq_array, channel_width=channel_width, delay_convention=convention
    )
    assert np.isclose(
        np.max(np.absolute(new_gain_obj.gain_array)),
        1.0,
        rtol=new_gain_obj._gain_array.tols[0],
        atol=new_gain_obj._gain_array.tols[1],
    )
    assert np.isclose(
        np.min(np.absolute(new_gain_obj.gain_array)),
        1.0,
        rtol=new_gain_obj._gain_array.tols[0],
        atol=new_gain_obj._gain_array.tols[1],
    )

    if convention == "minus":
        conv = -1
    else:
        conv = 1
    np.testing.assert_allclose(
        np.angle(new_gain_obj.gain_array[:, 10, :, :]) % (2 * np.pi),
        (conv * 2 * np.pi * delay_obj.delay_array[:, 0, :, :] * freq_array[10])
        % (2 * np.pi),
        rtol=new_gain_obj._gain_array.tols[0],
        atol=new_gain_obj._gain_array.tols[1],
    )
    np.testing.assert_allclose(
        delay_obj.quality_array[:, 0],
        new_gain_obj.quality_array[:, 10, :, :],
        rtol=new_gain_obj._quality_array.tols[0],
        atol=new_gain_obj._quality_array.tols[1],
    )

    assert new_gain_obj.history == (
        delay_obj.history + "  Converted from delays to gains using pyuvdata."
    )

    if same_freqs:
        with check_warnings(None):
            new_gain_obj2.convert_to_gain(
                delay_convention=convention,
                freq_array=new_gain_obj.freq_array,
                channel_width=new_gain_obj.channel_width,
            )

        assert new_gain_obj == new_gain_obj2


def test_convert_to_gain_errors(gain_data, delay_data, multi_spw_delay):
    delay_obj = delay_data
    gain_obj = gain_data

    delay_obj.Nfreqs = 1
    delay_obj.freq_array = np.arange(delay_obj.Nfreqs) * 1e6 + 1e8
    delay_obj.channel_width = np.full(delay_obj.Nfreqs, 1e6)

    with pytest.raises(
        ValueError,
        match="convert_to_gain currently does not support multiple spectral windows",
    ):
        multi_spw_delay.convert_to_gain(
            freq_array=delay_obj.freq_array, channel_width=delay_obj.channel_width
        )

    with pytest.raises(ValueError, match="The data is already a gain cal_type."):
        gain_obj.convert_to_gain(
            freq_array=delay_obj.freq_array, channel_width=delay_obj.channel_width
        )

    with pytest.raises(
        ValueError,
        match="The channel_width parameter be an array shaped like the freq_array",
    ):
        delay_obj.convert_to_gain(
            freq_array=delay_obj.freq_array, channel_width=delay_obj.channel_width[0]
        )

    with pytest.raises(
        ValueError, match="delay_convention can only be 'minus' or 'plus'"
    ):
        delay_obj.convert_to_gain(
            delay_convention="foo",
            freq_array=delay_obj.freq_array,
            channel_width=delay_obj.channel_width,
        )

    with pytest.raises(
        ValueError, match="freq_array parameter must be a one dimensional array"
    ):
        delay_obj.convert_to_gain(
            freq_array=delay_obj.freq_array.reshape(-1, 1),
            channel_width=delay_obj.channel_width,
        )

    with pytest.raises(
        ValueError, match="freq_array contains values outside the freq_range."
    ):
        delay_obj.convert_to_gain(
            freq_array=delay_obj.freq_array * 1e6, channel_width=delay_obj.channel_width
        )


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("sel_type", ["names", "numbers"])
@pytest.mark.parametrize("invert", [True, False])
def test_select_antennas(caltype, gain_data, delay_data, sel_type, invert):
    if caltype == "gain":
        calobj = gain_data
        # test list handling
        calobj.ant_array = calobj.ant_array.tolist()
    else:
        calobj = delay_data

    old_history = calobj.history

    ants_to_keep = np.array([65, 96, 9, 97, 89, 22, 20, 72])
    names_dict = dict(
        zip(
            calobj.telescope.antenna_numbers,
            calobj.telescope.antenna_names,
            strict=True,
        )
    )
    ants_to_discard = calobj.telescope.antenna_numbers[
        np.isin(calobj.telescope.antenna_numbers, ants_to_keep, invert=True)
    ]
    sel_ants = ants_to_discard if invert else ants_to_keep
    if sel_type == "names":
        kwargs = {"antenna_names": [names_dict[a] for a in sel_ants], "invert": invert}
    if sel_type == "numbers":
        kwargs = {"antenna_nums": sel_ants, "invert": invert}

    calobj.select(**kwargs)

    assert len(ants_to_keep) == calobj.Nants_data
    assert np.all(np.isin(ants_to_keep, calobj.ant_array))
    assert np.all(np.isin(calobj.ant_array, ants_to_keep))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific antennas using pyuvdata.",
        calobj.history,
    )


@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [
            {"antenna_nums": [500, 600, 700], "strict": True},
            "Antenna number [500 600 700] is not present in the ant_array",
        ],
        [
            {"antenna_names": ["test1"], "strict": True},
            "Antenna name test1 is not present in the antenna_names array",
        ],
        [
            {"antenna_names": ["test1"], "strict": None},
            "No data matching this antenna selection exists.",
        ],
        [
            {"antenna_names": [], "antenna_nums": []},
            "Only one of antenna_nums and antenna_names can be provided.",
        ],
    ],
)
def test_select_antenna_errors(gain_data, kwargs, err_msg):
    # check for errors associated with antennas not included in data, bad names
    # or providing numbers and names

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        gain_data.select(**kwargs)


def test_select_antennas_write_telescope(gain_data, tmp_path):
    calobj = gain_data
    calobj.select(antenna_nums=[65, 96, 9, 97, 89, 22, 20, 72], inplace=True)

    # check that write_calfits works with Nants_data < Nants_telescope
    write_file_calfits = str(tmp_path / "select_test.calfits")
    calobj.write_calfits(write_file_calfits, clobber=True)

    # check that reading it back in works too
    new_calobj = UVCal.from_file(write_file_calfits)

    assert calobj == new_calobj


def test_select_antennas_total_quality(gain_data):
    calobj = gain_data
    # check that total_quality_array is handled properly when present
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    warn_type = [UserWarning]
    msg = [
        "Changing number of antennas, but preserving the total_quality_array, "
        "which may have been defined based in part on antennas which will be "
        "removed."
    ]
    with check_warnings(warn_type, match=msg):
        calobj.select(antenna_nums=[65, 96, 9, 97, 89, 22, 20, 72], inplace=True)
    assert calobj.total_quality_array is not None


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("time_range", [True, False])
@pytest.mark.parametrize("use_range", [True, False])
@pytest.mark.parametrize("invert", [True, False])
def test_select_times(caltype, time_range, use_range, gain_data, delay_data, invert):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    orig_time_array = calobj.time_array
    times_to_keep = orig_time_array[2:5]
    time_range_to_keep = [np.min(times_to_keep), np.max(times_to_keep)]
    old_history = calobj.history
    if use_range:
        kwargs = {"time_range": time_range_to_keep, "invert": invert}
    else:
        kwargs = {"times": times_to_keep, "invert": invert}

    if time_range:
        calobj = time_array_to_time_range(calobj)

    calobj.select(**kwargs)
    if not invert:
        assert len(times_to_keep) == calobj.Ntimes
    else:
        assert (len(orig_time_array) - len(times_to_keep)) == calobj.Ntimes

    if time_range:
        for t in times_to_keep:
            assert (
                np.any((calobj.time_range[:, 0] <= t) & (calobj.time_range[:, 1] >= t))
                != invert
            )
    else:
        assert np.all(np.isin(calobj.time_array, times_to_keep, invert=invert))
        assert np.all(np.isin(times_to_keep, calobj.time_array, invert=invert))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific times using pyuvdata.", calobj.history
    )


@pytest.mark.parametrize("time_range", [True, False])
def test_select_times_single_to_calfits(gain_data, tmp_path, time_range):
    calobj = gain_data
    time_to_keep = calobj.time_array[1]
    time_range_to_keep = [time_to_keep - 0.1, time_to_keep + 0.1]

    if time_range:
        calobj = time_array_to_time_range(calobj)

    write_file_calfits = str(tmp_path / "select_test_single_time.calfits")
    calobj2 = calobj.copy()

    calobj.select(time_range=time_range_to_keep)
    calobj2.select(times=time_to_keep)

    # check same answer with time_range parameter
    assert calobj == calobj2

    # Make sure that writing out to calfits works
    calobj2.write_calfits(write_file_calfits, clobber=True)


@pytest.mark.parametrize(
    "time_range,kwargs,err_msg",
    [
        [True, {"times": 0, "strict": True}, "Time 0 does not fall in any time_range."],
        [False, {"times": 0, "strict": True}, "Time 0 is not present in the time_arr"],
        [False, {"times": [0], "strict": None}, "No data matching this time selection"],
        [False, {"times": 0, "time_range": [0, 0]}, "Only one of [times, time_range"],
    ],
)
def test_select_times_errs(gain_data, time_range, kwargs, err_msg):
    calobj = gain_data
    if time_range:
        calobj = time_array_to_time_range(calobj)

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        calobj.select(**kwargs)


@pytest.mark.parametrize("time_range", [True, False])
def test_select_times_calfits_error(gain_data, time_range, tmp_path):
    write_file_calfits = str(tmp_path / "select_test_error.calfits")
    # check for warnings and errors associated with unevenly spaced times
    calobj = gain_data
    sel_times = calobj.time_array[[0, 2, 3]]
    if time_range:
        calobj = time_array_to_time_range(calobj)
        sel_msg = "Selected times include multiple time ranges."
        msg = "Object contains multiple time ranges."
    else:
        sel_msg = "Selected times are not evenly spaced."
        msg = "The times are not evenly spaced."

    with check_warnings(UserWarning, sel_msg):
        calobj.select(times=sel_times, warn_spacing=True)

    with pytest.raises(ValueError, match=msg):
        calobj.write_calfits(write_file_calfits)


@pytest.mark.parametrize("invert", [True, False])
def test_select_frequencies(gain_data, invert):
    calobj = gain_data
    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    old_history = calobj.history

    freqs_to_keep = calobj.freq_array[np.arange(4, 8)]
    freqs_to_discard = calobj.freq_array[
        np.isin(calobj.freq_array, freqs_to_keep, invert=True)
    ]

    calobj.select(
        frequencies=freqs_to_discard if invert else freqs_to_keep, invert=invert
    )

    assert len(freqs_to_keep) == calobj.Nfreqs
    assert np.all(np.isin(calobj.freq_array, freqs_to_keep))
    assert np.all(np.isin(freqs_to_keep, calobj.freq_array))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        calobj.history,
    )


@pytest.mark.parametrize(
    "freq_chans,warn_msg",
    [[5, None], [[0, 2, 4, 6, 8], "Selected frequencies are not contiguous."]],
)
def test_select_frequency_write_calfits(gain_data, tmp_path, freq_chans, warn_msg):
    calobj = gain_data
    write_file_calfits = str(tmp_path / "select_test.calfits")

    freqs_to_keep = calobj.freq_array[freq_chans]
    with check_warnings(None if warn_msg is None else UserWarning, match=warn_msg):
        calobj.select(frequencies=freqs_to_keep, warn_spacing=True)
    calobj.write_calfits(write_file_calfits, clobber=True)


@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [
            {"frequencies": [1000.5], "strict": True},
            "Frequency 1000.5 is not present in the freq_array",
        ],
        [
            {"frequencies": -100, "strict": None},
            "No data matching this frequency selection exists.",
        ],
        [
            {"spws": -9999, "strict": True},
            "SPW number -9999 is not present in the spw_array",
        ],
        [
            {"spws": [-999], "strict": None},
            "No data matching this spectral window selection exists.",
        ],
    ],
)
def test_select_freq_errors(multi_spw_gain, kwargs, err_msg):
    # check for errors associated with frequencies not included in data
    with pytest.raises(ValueError, match=err_msg):
        multi_spw_gain.select(**kwargs)


def test_select_frequency_spacing_uvfits_error(gain_data, tmp_path):
    write_file_calfits = str(tmp_path / "select_freq_err.calfits")
    calobj = gain_data
    # check for warnings and errors associated with unevenly spaced frequencies
    freqs_to_keep = calobj.freq_array[[0, 5, 6]]
    with check_warnings(
        UserWarning, match="Selected frequencies are not evenly spaced."
    ):
        calobj.select(frequencies=freqs_to_keep, warn_spacing=True)

    with pytest.raises(
        ValueError,
        match="The frequencies are not evenly spaced or have differing values of",
    ):
        calobj.write_calfits(write_file_calfits, clobber=True)


@pytest.mark.filterwarnings("ignore:The freq_range attribute should not be set if")
def test_select_frequencies_multispw(multi_spw_gain, tmp_path):
    calobj = multi_spw_gain

    calobj2 = calobj.copy()
    old_history = calobj.history
    freqs_to_keep = calobj.freq_array[np.arange(4, 8)]

    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    calobj2.total_quality_array = np.zeros(
        calobj2._total_quality_array.expected_shape(calobj2)
    )

    calobj2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == calobj2.Nfreqs
    for f in freqs_to_keep:
        assert f in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in freqs_to_keep

    assert utils.history._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        calobj2.history,
    )

    # test calfits write error
    write_file_calfits = str(tmp_path / "select_test.calfits")
    with pytest.raises(
        ValueError,
        match="The calfits format does not support multiple spectral windows",
    ):
        calobj2.write_calfits(write_file_calfits, clobber=True)

    # test that we can write to calfits if select to only one spw
    calobj2 = calobj.copy()
    freqs_to_keep = calobj.freq_array[np.arange(5)]
    calobj2.select(frequencies=freqs_to_keep)
    calobj2.check()

    assert len(freqs_to_keep) == calobj2.Nfreqs
    for f in freqs_to_keep:
        assert f in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in freqs_to_keep

    assert calobj2.Nspws == 1
    calobj2.write_calfits(write_file_calfits, clobber=True)

    calobj3 = calobj.select(spws=[1], inplace=False)

    # We've used different selection criteria, so the history _will_ be different
    calobj3.history = calobj2.history

    assert calobj3 == calobj2
    with check_warnings(UserWarning, match="Cannot select on spws if Nspws=1."):
        calobj3.select(spws=1)

    assert calobj3 == calobj2

    calobj3 = UVCal.from_file(write_file_calfits)

    calobj2.flex_spw_id_array = np.zeros(calobj2.Nfreqs, dtype=int)
    calobj2.spw_array = np.array([0])
    calobj2.check()

    assert calobj3 == calobj2


def test_select_freq_chans_delay_err(delay_data):
    with pytest.raises(
        ValueError, match="Cannot select on frequencies because this is a wide_band"
    ):
        delay_data.select(freq_chans=[0])


@pytest.mark.parametrize("invert", [True, False])
def test_select_freq_chans(gain_data, invert):
    calobj = gain_data
    old_history = calobj.history
    obj_freqs = calobj.freq_array
    chans_to_keep = np.arange(4, 8)
    chans_to_discard = np.nonzero(
        np.isin(np.arange(calobj.Nfreqs), chans_to_keep, invert=True)
    )

    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )

    calobj.select(
        freq_chans=chans_to_discard if invert else chans_to_keep, invert=invert
    )

    assert len(chans_to_keep) == calobj.Nfreqs
    assert np.all(np.isin(obj_freqs[chans_to_keep], calobj.freq_array))
    assert np.all(np.isin(calobj.freq_array, obj_freqs[chans_to_keep]))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        calobj.history,
    )


@pytest.mark.parametrize("invert", [True, False])
def test_select_freqs_and_chans(gain_data, invert):
    calobj = gain_data
    # Test selecting both channels and frequencies
    obj_freqs = calobj.freq_array
    freqs_to_keep = obj_freqs[np.arange(7, 10)]  # Overlaps with chans
    chans_to_keep = np.arange(4, 8)
    all_freqs = obj_freqs[4:10]

    calobj.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep, invert=invert)

    if invert:
        assert (len(obj_freqs) - len(all_freqs)) == calobj.Nfreqs
    else:
        assert len(all_freqs) == calobj.Nfreqs
    assert np.all(np.isin(calobj.freq_array, all_freqs, invert=invert))
    assert np.all(np.isin(all_freqs, calobj.freq_array, invert=invert))


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("invert", [True, False])
def test_select_spws_wideband(caltype, multi_spw_delay, wideband_gain, invert):
    if caltype == "gain":
        calobj = wideband_gain
    else:
        calobj = multi_spw_delay

    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    old_history = calobj.history

    spws_to_keep = [2, 3]
    spws_to_discard = calobj.spw_array[
        np.isin(calobj.spw_array, spws_to_keep, invert=True)
    ]
    calobj.select(spws=spws_to_discard if invert else spws_to_keep, invert=invert)

    assert len(spws_to_keep) == calobj.Nspws
    assert np.all(np.isin(spws_to_keep, calobj.spw_array))
    assert np.all(np.isin(calobj.spw_array, spws_to_keep))

    assert utils.history._check_histories(
        old_history + "  Downselected to specific spectral windows using pyuvdata.",
        calobj.history,
    )


@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [{"spws": [5], "strict": True}, "SPW number 5 is not present in the spw_array"],
        [{"spws": [5], "strict": None}, "No data matching this spectral window"],
    ],
)
def test_select_spws_wideband_errors(multi_spw_delay, kwargs, err_msg):
    # check for errors associated with spws not included in data
    with pytest.raises(ValueError, match=err_msg):
        multi_spw_delay.select(**kwargs)


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize(
    "sel_jones", ([-5, -6], ["xx", "yy"], ["nn", "ee"], [[-5, -6]])
)
@pytest.mark.parametrize("invert", [True, False])
def test_select_polarizations(caltype, sel_jones, gain_data, delay_data, invert):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    # add more jones terms to allow for better testing of selections
    extend_jones_axis(calobj)
    njones = calobj.Njones
    old_history = calobj.history

    calobj.select(jones=sel_jones, invert=invert)

    if isinstance(sel_jones[0], list):
        sel_jones = sel_jones[0]

    assert calobj.Njones == (njones - len(sel_jones) if invert else len(sel_jones))

    jones_arr = []
    for j in sel_jones:
        if not isinstance(j, int):
            j = utils.jstr2num(
                j, x_orientation=calobj.telescope.get_x_orientation_from_feeds()
            )
        jones_arr.append(j)

    assert np.all(np.isin(jones_arr, calobj.jones_array, invert=invert))
    assert np.all(np.isin(calobj.jones_array, jones_arr, invert=invert))
    assert utils.history._check_histories(
        old_history + "  Downselected to "
        "specific jones polarization terms "
        "using pyuvdata.",
        calobj.history,
    )


@pytest.mark.parametrize(
    "kwargs,err_msg",
    [
        [{"jones": 999, "strict": True}, "Jones term 999 is not present"],
        [{"jones": 999, "strict": None}, "No data matching this jones term selection"],
    ],
)
def test_select_jones_error(gain_data, kwargs, err_msg):
    # check for errors associated with polarizations not included in data
    with pytest.raises(ValueError, match=err_msg):
        gain_data.select(**kwargs)


def test_select_polarization_write_error(gain_data, tmp_path):
    calobj = gain_data
    extend_jones_axis(calobj)

    with check_warnings(
        UserWarning, match="Selected jones polarization terms are not evenly spaced"
    ):
        calobj.select(jones=calobj.jones_array[[0, 1, 3]], warn_spacing=True)
    write_file_calfits = os.path.join(tmp_path, "select_test_pol_err.calfits")

    with pytest.raises(ValueError, match="The jones values are not evenly spaced"):
        calobj.write_calfits(write_file_calfits)


def test_select_phase_center_err():
    uvc = UVCal()

    with pytest.raises(ValueError, match="Cannot set both phase_center_ids and"):
        uvc.select(phase_center_ids=0, catalog_names="dummy")

    with pytest.raises(ValueError, match="Both phase_center_id_array and phase_center"):
        uvc.select(phase_center_ids=0)


def test_select_phase_centers(uvcal_phase_center):
    uvcal_phase_center.phase_center_id_array[::2] = 2
    uvcal_phase_center.phase_center_catalog[2] = dict(
        uvcal_phase_center.phase_center_catalog[1].items()
    )
    uvcal_phase_center.phase_center_catalog[2]["cat_name"] = "mystery"

    uvcopy1 = uvcal_phase_center.select(catalog_names="3c84", inplace=False)
    uvcopy2 = uvcal_phase_center.select(phase_center_ids=1, inplace=False)

    uvcopy2.history = uvcopy1.history
    assert uvcopy1 == uvcopy2
    assert uvcopy2 != uvcal_phase_center

    # Force the selection on the same times
    uvcopy2 = uvcal_phase_center.copy()
    uvcal_phase_center.select(times=uvcal_phase_center.time_array[1::2])
    # Try multi-select
    uvcopy2.select(times=uvcopy2.time_array[1::2], catalog_names="3c84")
    uvcal_phase_center.history = uvcopy1.history
    uvcopy2.history = uvcopy1.history
    assert uvcal_phase_center == uvcopy1
    assert uvcopy2 == uvcopy1


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select(caltype, gain_data, delay_data):
    # now test selecting along all axes at once
    if caltype == "gain":
        calobj = gain_data
        freqs_to_keep = calobj.freq_array[np.arange(2, 5)]
    else:
        calobj = delay_data
        freqs_to_keep = None

    calobj2 = calobj.copy()

    old_history = calobj.history

    ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
    times_to_keep = calobj.time_array[[1, 2]]
    jones_to_keep = [-5]

    calobj2.select(
        antenna_nums=ants_to_keep,
        frequencies=freqs_to_keep,
        times=times_to_keep,
        jones=jones_to_keep,
    )

    assert len(ants_to_keep) == calobj2.Nants_data
    for ant in ants_to_keep:
        assert ant in calobj2.ant_array
    for ant in calobj2.ant_array:
        assert ant in ants_to_keep

    assert len(times_to_keep) == calobj2.Ntimes
    for t in times_to_keep:
        assert t in calobj2.time_array
    for t in np.unique(calobj2.time_array):
        assert t in times_to_keep

    if caltype == "gain":
        assert len(freqs_to_keep) == calobj2.Nfreqs
        for f in freqs_to_keep:
            assert f in calobj2.freq_array
        for f in np.unique(calobj2.freq_array):
            assert f in freqs_to_keep
        expected_history = old_history + (
            "  Downselected to specific antennas, times, frequencies, "
            "jones polarization terms using pyuvdata."
        )
    else:
        expected_history = old_history + (
            "  Downselected to specific antennas, times, "
            "jones polarization terms using pyuvdata."
        )

    assert len(jones_to_keep) == calobj2.Njones
    for j in jones_to_keep:
        assert j in calobj2.jones_array
    for j in np.unique(calobj2.jones_array):
        assert j in jones_to_keep
    assert utils.history._check_histories(expected_history, calobj2.history)


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_wideband(caltype, multi_spw_delay, wideband_gain):
    if caltype == "gain":
        calobj = wideband_gain
    else:
        calobj = multi_spw_delay

    calobj2 = calobj.copy()

    old_history = calobj.history

    ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
    spws_to_keep = [2, 3]
    times_to_keep = calobj.time_array[[1, 2]]
    jones_to_keep = [-5]

    calobj2.select(
        antenna_nums=ants_to_keep,
        spws=spws_to_keep,
        times=times_to_keep,
        jones=jones_to_keep,
    )

    assert len(ants_to_keep) == calobj2.Nants_data
    for ant in ants_to_keep:
        assert ant in calobj2.ant_array
    for ant in calobj2.ant_array:
        assert ant in ants_to_keep

    assert len(times_to_keep) == calobj2.Ntimes
    for t in times_to_keep:
        assert t in calobj2.time_array
    for t in np.unique(calobj2.time_array):
        assert t in times_to_keep

    assert len(spws_to_keep) == calobj2.Nspws
    for spw in spws_to_keep:
        assert spw in calobj2.spw_array
    for spw in np.unique(calobj2.spw_array):
        assert spw in spws_to_keep

    assert len(jones_to_keep) == calobj2.Njones
    for j in jones_to_keep:
        assert j in calobj2.jones_array
    for j in np.unique(calobj2.jones_array):
        assert j in jones_to_keep

    assert utils.history._check_histories(
        old_history + "  Downselected to "
        "specific antennas, times, "
        "spectral windows, jones polarization terms "
        "using pyuvdata.",
        calobj2.history,
    )


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_antennas(caltype, gain_data, method, delay_data):
    """Test adding antennas between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    calobj.select(antenna_nums=ants1)
    calobj2.select(antenna_nums=ants2)
    if method == "fast_concat":
        kwargs = {"axis": "antenna", "inplace": True}
    else:
        kwargs = {}

    getattr(calobj, method)(calobj2, **kwargs)
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_full.history + "  Downselected to specific "
        "antennas using pyuvdata. Combined "
        "data along antenna axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # test for when total_quality_array is present
    calobj.select(antenna_nums=ants1)
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    warn_type = [UserWarning]
    msg = ["Total quality array detected"]
    with check_warnings(warn_type, match=msg):
        getattr(calobj, method)(calobj2, **kwargs)
    assert calobj.total_quality_array is None

    if caltype == "delay":
        # test for when quality array is present in first file but
        # not in second
        calobj.select(antenna_nums=ants1)
        qa = np.ones(calobj._quality_array.expected_shape(calobj))
        qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2))

        tot_qa = np.concatenate([qa, qa2], axis=0)
        calobj.quality_array = qa
        calobj2.quality_array = None
        getattr(calobj, method)(calobj2, **kwargs)
        np.testing.assert_allclose(
            calobj.quality_array,
            tot_qa,
            rtol=calobj._quality_array.tols[0],
            atol=calobj._quality_array.tols[1],
        )

        # test for when quality_array is present in second file but not first
        calobj.select(antenna_nums=ants1)
        qa = np.zeros(calobj._quality_array.expected_shape(calobj))
        qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2))
        tot_qa = np.concatenate([qa, qa2], axis=0)
        calobj.quality_array = None
        calobj2.quality_array = qa2
        getattr(calobj, method)(calobj2, **kwargs)
        np.testing.assert_allclose(
            calobj.quality_array,
            tot_qa,
            rtol=calobj._quality_array.tols[0],
            atol=calobj._quality_array.tols[1],
        )

    # Out of order - antennas
    calobj = calobj_full.copy()
    calobj2 = calobj.copy()
    calobj.select(antenna_nums=ants2)
    calobj2.select(antenna_nums=ants1)
    getattr(calobj, method)(calobj2, **kwargs)
    calobj.history = calobj_full.history
    if method == "fast_concat":
        # need to sort objects before they will be equal
        assert calobj != calobj_full
        calobj.reorder_antennas()
    assert calobj == calobj_full


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_reorder_ants(caltype, metadata_only, gain_data, delay_data):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    calobj2 = calobj.copy(metadata_only=metadata_only)
    if metadata_only:
        calobj = calobj2.copy()

    # this is a no-op because it's already sorted this way
    calobj2.reorder_antennas()
    ant_num_diff = np.diff(calobj2.ant_array)
    assert np.all(ant_num_diff > 0)

    calobj2.reorder_antennas("-number")
    ant_num_diff = np.diff(calobj2.ant_array)
    assert np.all(ant_num_diff < 0)

    sorted_names = np.sort(calobj.telescope.antenna_names)
    calobj.reorder_antennas("name")
    temp = np.asarray(calobj.telescope.antenna_names)
    dtype_use = temp.dtype
    name_array = np.zeros_like(calobj.ant_array, dtype=dtype_use)
    for ind, ant in enumerate(calobj.ant_array):
        name_array[ind] = calobj.telescope.antenna_names[
            np.nonzero(calobj.telescope.antenna_numbers == ant)[0][0]
        ]

    assert np.all(sorted_names == name_array)

    # test sorting with an integer array. First resort back to by number
    calobj2.reorder_antennas("number")
    sorted_nums = [int(name[3:]) for name in sorted_names]
    index_array = [np.nonzero(calobj2.ant_array == ant)[0][0] for ant in sorted_nums]
    calobj2.reorder_antennas(index_array)
    assert calobj2 == calobj


def test_reorder_ants_errors(gain_data):
    with pytest.raises(
        ValueError,
        match="order must be one of 'number', 'name', '-number', '-name' or an "
        "index array of length Nants_data",
    ):
        gain_data.reorder_antennas("foo")

    with pytest.raises(
        ValueError,
        match="If order is an index array, it must contain all indices for the"
        "ant_array, without duplicates.",
    ):
        gain_data.reorder_antennas(gain_data.telescope.antenna_numbers.astype(float))

    with pytest.raises(
        ValueError,
        match="If order is an index array, it must contain all indices for the"
        "ant_array, without duplicates.",
    ):
        gain_data.reorder_antennas(gain_data.telescope.antenna_numbers[:8])


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_reorder_freqs(caltype, metadata_only, gain_data, delay_data):
    if caltype == "gain":
        calobj = gain_data
        # add total_quality_array
        calobj.total_quality_array = np.tile(
            np.arange(calobj.Nfreqs, dtype=float)[:, np.newaxis, np.newaxis],
            (1, calobj.Ntimes, calobj.Njones),
        )
    else:
        calobj = delay_data

    calobj2 = calobj.copy(metadata_only=metadata_only)
    if metadata_only:
        calobj = calobj2.copy()

    if caltype == "delay":
        with check_warnings(
            UserWarning,
            match="Cannot reorder the frequency/spw axis with only one frequency and "
            "spw. Returning the object unchanged.",
        ):
            calobj2.reorder_freqs(channel_order="-freq")
        assert calobj == calobj2
    else:
        calobj2.reorder_freqs(channel_order="-freq")
        freq_diff = np.diff(calobj2.freq_array)
        assert np.all(freq_diff < 0)

        if caltype == "gain" and not metadata_only:
            # check total quality array
            total_quality_diff = np.diff(calobj2.total_quality_array, axis=0)
            assert np.all(total_quality_diff < 0)

        calobj.reorder_freqs(channel_order=np.flip(np.arange(calobj.Nfreqs)))
        assert calobj == calobj2


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_reorder_freqs_multi_spw(caltype, multi_spw_gain, multi_spw_delay):
    if caltype == "gain":
        calobj = multi_spw_gain
    else:
        calobj = multi_spw_delay

    calobj2 = calobj.copy()

    if caltype == "delay":
        with check_warnings(
            UserWarning,
            match=(
                "channel_order and select_spws are ignored for wide-band "
                "calibration solutions"
            ),
        ):
            calobj.reorder_freqs(spw_order="-number", channel_order="freq")
    else:
        # this should be a no-op
        calobj.reorder_freqs(spw_order="number", channel_order="freq")
        assert calobj2 == calobj

        calobj.reorder_freqs(spw_order="-number", channel_order="freq")
        for spw in calobj.spw_array:
            ant_num_diff = np.diff(
                calobj.freq_array[np.nonzero(calobj.flex_spw_id_array == spw)[0]]
            )
            assert np.all(ant_num_diff > 0)

    spw_diff = np.diff(calobj.spw_array)
    assert np.all(spw_diff < 0)

    calobj2.reorder_freqs(spw_order=np.flip(np.arange(calobj2.Nspws)))
    assert calobj2 == calobj

    calobj.reorder_freqs(spw_order="freq")
    spw_diff = np.diff(calobj.spw_array)
    assert np.all(spw_diff > 0)


def test_reorder_freqs_errors(gain_data, multi_spw_delay):
    with pytest.raises(
        ValueError,
        match="spw_order can only be one of 'number', '-number', "
        "'freq', '-freq', None or an index array of length Nspws",
    ):
        gain_data.reorder_freqs(spw_order="foo")

    with pytest.raises(
        ValueError,
        match="spw_order can only be one of 'number', '-number', "
        "'freq', '-freq', None or an index array of length Nspws",
    ):
        multi_spw_delay.reorder_freqs(spw_order="foo")

    with pytest.raises(
        ValueError,
        match="If spw_order is an array, it must contain all indices for "
        "the spw_array, without duplicates.",
    ):
        multi_spw_delay.reorder_freqs(spw_order=[0, 1])

    with pytest.raises(
        ValueError,
        match="channel_order can only be one of 'freq' or '-freq' or an index "
        "array of length Nfreqs",
    ):
        gain_data.reorder_freqs(channel_order="foo")

    with pytest.raises(
        ValueError,
        match="Index array for channel_order must contain all indices for "
        "the frequency axis, without duplicates.",
    ):
        gain_data.reorder_freqs(
            channel_order=np.arange(gain_data.Nfreqs, dtype=float) * 2
        )

    with pytest.raises(
        ValueError,
        match="Index array for channel_order must contain all indices for "
        "the frequency axis, without duplicates.",
    ):
        gain_data.reorder_freqs(channel_order=np.arange(3))


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
@pytest.mark.parametrize("time_range", [True, False])
def test_reorder_times(caltype, metadata_only, time_range, gain_data, delay_data):
    if caltype == "gain":
        calobj = gain_data
        # add total_quality_array
        calobj.total_quality_array = np.tile(
            np.arange(calobj.Ntimes, dtype=float)[np.newaxis, :, np.newaxis],
            (calobj.Nfreqs, 1, calobj.Njones),
        )
        calobj.check()
    else:
        calobj = delay_data

    if time_range:
        calobj = time_array_to_time_range(calobj)

    calobj2 = calobj.copy(metadata_only=metadata_only)
    if metadata_only:
        calobj = calobj2.copy()

    # this should be a no-op
    calobj.reorder_times()
    assert calobj == calobj2

    calobj2.reorder_times("-time")
    if time_range:
        time_diff = np.diff(calobj2.time_range[:, 0])
    else:
        time_diff = np.diff(calobj2.time_array)
    assert np.all(time_diff < 0)

    if caltype == "gain" and not metadata_only:
        # check total quality array
        total_quality_diff = np.diff(calobj2.total_quality_array, axis=1)
        assert np.all(total_quality_diff < 0)

    calobj.reorder_times(np.flip(np.arange(calobj.Ntimes)))
    assert calobj == calobj2


def test_reorder_times_errors(gain_data):
    with pytest.raises(
        ValueError,
        match="order must be one of 'time', '-time' or an index array of length Ntimes",
    ):
        gain_data.reorder_times("foo")

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indices for the time axis, "
        "without duplicates.",
    ):
        gain_data.reorder_times(np.arange(gain_data.Ntimes) * 2)

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indices for the time axis, "
        "without duplicates.",
    ):
        gain_data.reorder_times(np.arange(7))

    gain_data = time_array_to_time_range(gain_data, keep_time_array=True)
    with pytest.raises(
        ValueError, match="The time_array and time_range attributes are both set."
    ):
        gain_data.reorder_times()


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_reorder_jones(caltype, metadata_only, gain_data, delay_data):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    # all the input objects have a Njones=1, extend to get to 4
    calobj2 = calobj.copy(metadata_only=metadata_only)
    extend_jones_axis(calobj2)

    if caltype == "gain" and not metadata_only:
        # add total_quality_array
        calobj2.total_quality_array = np.tile(
            np.arange(calobj2.Njones, dtype=float)[np.newaxis, np.newaxis, :],
            (calobj2.Nfreqs, calobj2.Ntimes, 1),
        )

    calobj = calobj2.copy()

    # this is a no-op because it's already sorted this way
    calobj2.reorder_jones("-number")
    jnum_diff = np.diff(calobj2.jones_array)
    assert np.all(jnum_diff < 0)

    calobj2.reorder_jones("number")
    jnum_diff = np.diff(calobj2.jones_array)
    assert np.all(jnum_diff > 0)

    if caltype == "gain" and not metadata_only:
        assert calobj2.total_quality_array is not None
        # check total quality array
        total_quality_diff = np.diff(calobj2.total_quality_array, axis=2)
        assert np.all(total_quality_diff < 0)

    # the default order is "name"
    calobj2.reorder_jones()
    name_array = np.asarray(
        utils.jnum2str(
            calobj2.jones_array,
            x_orientation=calobj2.telescope.get_x_orientation_from_feeds(),
        )
    )
    sorted_names = np.sort(name_array)
    assert np.all(sorted_names == name_array)

    # test sorting with an index array. Sort back to number first so indexing works
    sorted_nums = utils.jstr2num(
        sorted_names, x_orientation=calobj.telescope.get_x_orientation_from_feeds()
    )
    index_array = [np.nonzero(calobj.jones_array == num)[0][0] for num in sorted_nums]
    calobj.reorder_jones(index_array)
    assert calobj2 == calobj


def test_reorder_jones_errors(gain_data):
    # all the input objects have a Njones=1, extend to get to 4
    calobj = gain_data.copy()
    extend_jones_axis(calobj)

    with pytest.raises(
        ValueError,
        match="order must be one of 'number', 'name', '-number', '-name' or an "
        "index array of length Njones",
    ):
        calobj.reorder_jones("foo")

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indices for "
        "the jones axis, without duplicates.",
    ):
        calobj.reorder_jones(np.arange(gain_data.Njones) * 2)

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indices for "
        "the jones axis, without duplicates.",
    ):
        calobj.reorder_jones(np.arange(2))


@pytest.mark.filterwarnings("ignore:Combined Jones elements are not evenly spaced")
@pytest.mark.filterwarnings("ignore:Combined frequencies are not evenly spaced")
@pytest.mark.filterwarnings("ignore:Cannot reorder the frequency/spw axis with only")
@pytest.mark.parametrize("add_type", ["ant", "time", "freq", "jones"])
@pytest.mark.parametrize("sort_type", ["ant", "time", "freq", "jones"])
@pytest.mark.parametrize("wide_band", [True, False])
def test_add_different_sorting(
    add_type, sort_type, wide_band, gain_data, wideband_gain
):
    if wide_band:
        calobj = wideband_gain.copy()
        calobj.check()
        assert calobj.freq_range is not None
    else:
        calobj = gain_data.copy()
    # add total_quality_array and initial flag array
    if add_type != "ant":
        calobj.total_quality_array = np.random.random(
            calobj._total_quality_array.expected_shape(calobj)
        )

    # all the input objects have a Njones=1, extend to get to 4
    extend_jones_axis(calobj, total_quality=False)

    if add_type == "ant":
        ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 80, 81])
        ants2 = np.array([65, 72, 88, 89, 96, 97, 104, 105, 112])
        cal1 = calobj.select(antenna_nums=ants1, inplace=False)
        cal2 = calobj.select(antenna_nums=ants2, inplace=False)
    elif add_type == "time":
        n_times2 = calobj.Ntimes // 2
        times1 = calobj.time_array[:n_times2]
        times2 = calobj.time_array[n_times2:]
        cal1 = calobj.select(times=times1, inplace=False)
        cal2 = calobj.select(times=times2, inplace=False)
    elif add_type == "freq":
        if wide_band:
            spws1 = calobj.spw_array[: calobj.Nspws // 2]
            spws2 = calobj.spw_array[calobj.Nspws // 2 :]
            cal1 = calobj.select(spws=spws1, inplace=False)
            cal2 = calobj.select(spws=spws2, inplace=False)
        else:
            cal1 = calobj.select(
                freq_chans=np.arange(calobj.Nfreqs // 2), inplace=False
            )
            cal2 = calobj.select(
                freq_chans=np.arange(calobj.Nfreqs // 2, calobj.Nfreqs), inplace=False
            )
    elif add_type == "jones":
        cal1 = calobj.select(jones=np.array([-5, -7]), inplace=False)
        cal2 = calobj.select(jones=np.array([-6, -8]), inplace=False)

    if sort_type == "ant":
        cal1.reorder_antennas("number")
        cal2.reorder_antennas("-number")
        calobj.reorder_antennas("name")
        order_check = cal1._ant_array == cal2._ant_array
    elif sort_type == "time":
        cal1.reorder_times("time")
        cal2.reorder_times("-time")
        calobj.reorder_times("time")
        order_check = cal1._time_array == cal2._time_array
    elif sort_type == "freq":
        if wide_band:
            cal1.reorder_freqs(spw_order="number")
            cal2.reorder_freqs(spw_order="-number")
            calobj.reorder_freqs(spw_order="number")
            order_check = cal1._spw_array == cal2._spw_array
        else:
            cal1.reorder_freqs(channel_order="freq")
            cal2.reorder_freqs(channel_order="-freq")
            calobj.reorder_freqs(channel_order="freq")
            order_check = cal1._freq_array == cal2._freq_array
    elif sort_type == "jones":
        cal1.reorder_jones("name")
        cal2.reorder_jones("-number")
        calobj.reorder_jones("number")
        order_check = cal1._jones_array == cal2._jones_array

    # Make sure that the order has actually been scrambled
    assert not np.all(order_check)

    # Combine the objects in both orders
    cal3 = cal1 + cal2
    cal4 = cal2 + cal1

    if sort_type == "ant":
        cal3.reorder_antennas("name")
        cal4.reorder_antennas("name")
    elif sort_type == "time":
        cal3.reorder_times("time")
        cal4.reorder_times("time")
    elif sort_type == "freq":
        if wide_band:
            cal3.reorder_freqs()
            cal4.reorder_freqs()
        else:
            cal3.reorder_freqs(channel_order="freq")
            cal4.reorder_freqs(channel_order="freq")
    elif sort_type == "jones":
        cal3.reorder_jones("number")
        cal4.reorder_jones("number")

    # Deal with the history separately, since it will be different
    assert str.startswith(cal3.history, calobj.history)
    assert str.startswith(cal4.history, calobj.history)
    calobj.history = ""
    cal3.history = ""
    cal4.history = ""

    # Finally, make sure everything else lines up
    assert cal3 == calobj
    assert cal4._ant_array == calobj._ant_array
    assert cal4._freq_array == calobj._freq_array
    assert cal4 == calobj


@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
@pytest.mark.parametrize("quality", [True, False])
def test_add_antennas_multispw(multi_spw_gain, quality, method):
    """Test adding antennas between two UVCal objects"""
    calobj = multi_spw_gain

    if not quality:
        calobj.quality_array = None

    calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    calobj.select(antenna_nums=ants1)
    calobj2.select(antenna_nums=ants2)
    if method == "fast_concat":
        kwargs = {"axis": "antenna", "inplace": True}
    else:
        kwargs = {}

    getattr(calobj, method)(calobj2, **kwargs)
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_full.history + "  Downselected to specific "
        "antennas using pyuvdata. Combined "
        "data along antenna axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full


@pytest.mark.filterwarnings("ignore:The freq_range attribute should not be set if")
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_frequencies(gain_data, method):
    """Test adding frequencies between two UVCal objects"""
    # don't test on delays because there's no freq axis for the delay array
    calobj = gain_data

    calobj2 = calobj.copy()
    calobj_full = calobj.copy()
    freqs1 = calobj.freq_array[np.arange(0, calobj.Nfreqs // 2)]
    freqs2 = calobj.freq_array[np.arange(calobj.Nfreqs // 2, calobj.Nfreqs)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    if method == "fast_concat":
        kwargs = {"axis": "freq", "inplace": True}
    else:
        kwargs = {}

    getattr(calobj, method)(calobj2, **kwargs)
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_full.history + "  Downselected to specific "
        "frequencies using pyuvdata. Combined "
        "data along frequency axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # test for when total_quality_array is present in first file but not second
    calobj.select(frequencies=freqs1)
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.zeros(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=0)
    calobj.total_quality_array = tqa

    with check_warnings(None):
        getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in second file but not first
    calobj = calobj_full.copy()
    calobj.select(frequencies=freqs1)
    tqa = np.zeros(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=0)
    calobj.total_quality_array = None
    calobj2.total_quality_array = tqa2

    getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in both
    calobj.select(frequencies=freqs1)
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=0)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when quality_array is present in first file but not in second
    calobj = calobj_full.copy()
    calobj2 = calobj.copy()
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    qa = np.ones(calobj._quality_array.expected_shape(calobj))
    qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2))
    tot_qa = np.concatenate([qa, qa2], axis=1)

    calobj.quality_array = qa
    calobj2.quality_array = None
    getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.quality_array,
        tot_qa,
        rtol=calobj._quality_array.tols[0],
        atol=calobj._quality_array.tols[1],
    )

    # test for when quality_array is present in second file but not first
    calobj.select(frequencies=freqs1)
    qa = np.zeros(calobj._quality_array.expected_shape(calobj))
    qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2))
    tot_qa = np.concatenate([qa, qa2], axis=1)

    calobj.quality_array = None
    calobj2.quality_array = qa2
    getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.quality_array,
        tot_qa,
        rtol=calobj._quality_array.tols[0],
        atol=calobj._quality_array.tols[1],
    )

    # Out of order - freqs
    calobj = calobj_full.copy()
    calobj2 = calobj_full.copy()
    calobj.select(frequencies=freqs2)
    calobj2.select(frequencies=freqs1)
    getattr(calobj, method)(calobj2, **kwargs)
    calobj.history = calobj_full.history
    if method == "fast_concat":
        # need to sort object first
        calobj.reorder_freqs(channel_order="freq")
    assert calobj == calobj_full


@pytest.mark.parametrize("split_f_ind", [3, 5])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add_frequencies_multispw(split_f_ind, method, multi_spw_gain):
    """Test adding frequencies between two UVCal objects"""
    # don't test on delays because there's no freq axis for the delay array

    # split_f_ind=5 splits the objects in the same place as the spws split
    # (so each object has only one spw). A different value splits within an spw.

    calobj = multi_spw_gain

    calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    freqs1 = calobj.freq_array[np.arange(0, split_f_ind)]
    freqs2 = calobj.freq_array[np.arange(split_f_ind, calobj.Nfreqs)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    if method == "fast_concat":
        kwargs = {"axis": "freq"}
    else:
        kwargs = {}

    calobj_sum = getattr(calobj, method)(calobj2, **kwargs)

    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_full.history + "  Downselected to specific "
        "frequencies using pyuvdata. Combined "
        "data along frequency axis using pyuvdata.",
        calobj_sum.history,
    )
    calobj_sum.history = calobj_full.history
    assert calobj_sum == calobj_full

    # test adding out of order
    if method == "fast_concat":
        if split_f_ind == 5:
            calobj_sum = calobj2.fast_concat(calobj, axis="freq")
        else:
            with check_warnings(
                UserWarning,
                match=[
                    "Channels from different spectral windows are interspersed "
                    "with one another, rather than being grouped together along the "
                    "frequency axis. Most file formats do not support such "
                    "non-grouping of data.",
                    "The frequencies are not evenly spaced or have differing values",
                ],
            ):
                calobj_sum = calobj2.fast_concat(calobj, axis="freq", warn_spacing=True)
            return
    else:
        calobj_sum = calobj2 + calobj

    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_full.history + "  Downselected to specific "
        "frequencies using pyuvdata. Combined "
        "data along frequency axis using pyuvdata.",
        calobj_sum.history,
    )
    calobj_sum.history = calobj_full.history

    if method == "fast_concat":
        # need to sort object first
        calobj_sum.reorder_freqs(channel_order="freq", spw_order="number")
    assert calobj_sum == calobj_full


@pytest.mark.parametrize(
    ["axis", "method"],
    [
        ["antenna", "__add__"],
        ["spw", "__add__"],
        ["multi", "__add__"],
        ["antenna", "fast_concat"],
        ["spw", "fast_concat"],
    ],
)
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_add_spw_wideband(axis, caltype, method, multi_spw_delay, wideband_gain):
    if caltype == "delay":
        calobj_full = multi_spw_delay
    else:
        calobj_full = wideband_gain
    calobj_full.select(times=calobj_full.time_array[:2])

    spw1 = calobj_full.spw_array[0]
    spw2 = calobj_full.spw_array[1:]
    ants1 = np.array([9, 10])
    ants2 = np.array([80, 81])
    calobj_full.select(antenna_nums=np.concatenate((ants1, ants2)))

    calobj = calobj_full.copy()
    calobj2 = calobj.copy()

    if axis == "antenna":
        calobj.select(antenna_nums=ants1)
        calobj2.select(antenna_nums=ants2)
    elif axis == "spw":
        calobj.select(spws=spw1)
        calobj2.select(spws=spw2)
    elif axis == "multi":
        calobj.select(antenna_nums=ants1, spws=spw1)
        calobj2.select(antenna_nums=ants2, spws=spw2)

        # zero out missing data in reference object
        ant1_inds = np.nonzero(np.isin(calobj_full.ant_array, ants1))[0]
        ant2_inds = np.nonzero(np.isin(calobj_full.ant_array, ants2))[0]
        if caltype == "delay":
            calobj_full.delay_array[ant1_inds, 1:] = 0
            calobj_full.delay_array[ant2_inds, 0] = 0
        else:
            calobj_full.gain_array[ant1_inds, 1:] = 0
            calobj_full.gain_array[ant2_inds, 0] = 0
        calobj_full.quality_array[ant1_inds, 1:] = 0
        calobj_full.quality_array[ant2_inds, 0] = 0
        calobj_full.flag_array[ant1_inds, 1:] = True
        calobj_full.flag_array[ant2_inds, 0] = True

    if method == "fast_concat":
        kwargs = {"axis": axis, "inplace": False}
    else:
        kwargs = {}

    calobj3 = getattr(calobj, method)(calobj2, **kwargs)

    # Check history is correct, before replacing and doing a full object check
    if axis == "multi":
        assert utils.history._check_histories(
            calobj_full.history + "  Downselected to specific antennas, spectral "
            "windows using pyuvdata. Combined data along antenna, spectral window axis "
            "using pyuvdata.",
            calobj3.history,
        )
    elif axis == "spw":
        assert utils.history._check_histories(
            calobj_full.history + "  Downselected to specific spectral windows using "
            "pyuvdata. Combined data along spectral window axis using pyuvdata.",
            calobj3.history,
        )
    elif axis == "antenna":
        assert utils.history._check_histories(
            calobj_full.history + "  Downselected to specific antennas using pyuvdata. "
            "Combined data along antenna axis using pyuvdata.",
            calobj3.history,
        )
    calobj3.history = calobj_full.history
    assert calobj3 == calobj_full

    # test adding out of order
    calobj3 = getattr(calobj2, method)(calobj, **kwargs)
    if method == "fast_concat":
        if axis == "spw":
            calobj3.reorder_freqs(spw_order="number")
        else:
            calobj3.reorder_antennas()

    # Check history is correct, before replacing and doing a full object check
    if axis == "multi":
        assert utils.history._check_histories(
            calobj_full.history + "  Downselected to specific antennas, spectral "
            "windows using pyuvdata. Combined data along antenna, spectral window axis "
            "using pyuvdata.",
            calobj3.history,
        )
    elif axis == "spw":
        assert utils.history._check_histories(
            calobj_full.history + "  Downselected to specific spectral windows using "
            "pyuvdata. Combined data along spectral window axis using pyuvdata.",
            calobj3.history,
        )
    elif axis == "antenna":
        assert utils.history._check_histories(
            calobj_full.history + "  Downselected to specific antennas using pyuvdata. "
            "Combined data along antenna axis using pyuvdata.",
            calobj3.history,
        )
    calobj3.history = calobj_full.history
    assert calobj3 == calobj_full


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("time_range", [True, False])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_times(caltype, time_range, method, gain_data, delay_data):
    """Test adding times between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    n_times2 = calobj.Ntimes // 2
    times1 = calobj.time_array[:n_times2]
    times2 = calobj.time_array[n_times2:]

    if time_range:
        keep_time_array = caltype == "gain"
        calobj = time_array_to_time_range(calobj, keep_time_array=keep_time_array)
        calobj.time_array = calobj.lst_array = None

    calobj2 = calobj.copy()
    calobj_full = calobj.copy()
    if caltype == "delay":
        check_warn = None
        check_msg = ""
        select_warn = None
        select_msg = ""
        add_warn = None
        add_msg = ""
    else:
        check_warn = None
        check_msg = ""
        select_warn = None
        select_msg = ""
        add_warn = None
        add_msg = ""
    with check_warnings(select_warn, match=select_msg):
        calobj.select(time_range=[np.min(times1), np.max(times1)])
    with check_warnings(select_warn, match=select_msg):
        calobj2.select(time_range=[np.min(times2), np.max(times2)])

    if method == "fast_concat":
        kwargs = {"axis": "time", "inplace": True}
    else:
        kwargs = {}

    with check_warnings(add_warn, match=add_msg):
        getattr(calobj, method)(calobj2, **kwargs)
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_full.history + "  Downselected to specific "
        "times using pyuvdata. Combined "
        "data along time axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # test for when total_quality_array is present in first file but not second
    with check_warnings(select_warn, match=select_msg):
        calobj.select(time_range=[np.min(times1), np.max(times1)])
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.zeros(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = tqa
    with check_warnings(add_warn, match=add_msg):
        getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in second file but not first
    with check_warnings(select_warn, match=select_msg):
        calobj.select(time_range=[np.min(times1), np.max(times1)])
    tqa = np.zeros(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = None
    calobj2.total_quality_array = tqa2
    with check_warnings(add_warn, match=add_msg):
        getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in both
    with check_warnings(select_warn, match=select_msg):
        calobj.select(time_range=[np.min(times1), np.max(times1)])
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    with check_warnings(add_warn, match=add_msg):
        getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    if caltype == "delay":
        # test for when quality_array is present in first file but not in second
        with check_warnings(select_warn, match=select_msg):
            calobj.select(time_range=[np.min(times1), np.max(times1)])
        qa = np.ones(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2), dtype=float)
        tot_qa = np.concatenate([qa, qa2], axis=2)

        calobj.quality_array = qa
        calobj2.quality_array = None
        with check_warnings(add_warn, match=add_msg):
            getattr(calobj, method)(calobj2, **kwargs)
        np.testing.assert_allclose(
            calobj.quality_array,
            tot_qa,
            rtol=calobj._quality_array.tols[0],
            atol=calobj._quality_array.tols[1],
        )

        # test for when quality array is present in second file but not first
        with check_warnings(select_warn, match=select_msg):
            calobj.select(time_range=[np.min(times1), np.max(times1)])
        qa = np.zeros(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2), dtype=float)
        tot_qa = np.concatenate([qa, qa2], axis=2)

        calobj.quality_array = None
        calobj2.quality_array = qa2
        with check_warnings(add_warn, match=add_msg):
            getattr(calobj, method)(calobj2, **kwargs)
        np.testing.assert_allclose(
            calobj.quality_array,
            tot_qa,
            rtol=calobj._quality_array.tols[0],
            atol=calobj._quality_array.tols[1],
        )

    # Out of order - times
    calobj = calobj_full.copy()
    calobj2 = calobj.copy()
    with check_warnings(select_warn, match=select_msg):
        calobj.select(time_range=[np.min(times2), np.max(times2)])
    with check_warnings(select_warn, match=select_msg):
        calobj2.select(time_range=[np.min(times1), np.max(times1)])
    with check_warnings(add_warn, match=add_msg):
        getattr(calobj, method)(calobj2, **kwargs)
    calobj.history = calobj_full.history
    if method == "fast_concat":
        # need to sort object first
        warn = check_warn
        warn_msg = check_msg
        with check_warnings(warn, match=warn_msg):
            calobj.reorder_times()
    assert calobj == calobj_full


@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
@pytest.mark.parametrize("quality", [True, False])
def test_add_times_multispw(method, multi_spw_gain, quality):
    """Test adding times between two UVCal objects"""
    calobj = multi_spw_gain

    if not quality:
        calobj.quality_array = None

    calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    n_times2 = calobj.Ntimes // 2
    times1 = calobj.time_array[:n_times2]
    times2 = calobj.time_array[n_times2:]
    calobj.select(times=times1)
    calobj2.select(times=times2)
    if method == "fast_concat":
        kwargs = {"axis": "time", "inplace": True}
    else:
        kwargs = {}
    getattr(calobj, method)(calobj2, **kwargs)
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_full.history + "  Downselected to specific "
        "times using pyuvdata. Combined "
        "data along time axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_jones(caltype, method, gain_data, delay_data):
    """Test adding Jones axes between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    calobj2 = calobj.copy()

    calobj_original = calobj.copy()
    # artificially change the Jones value to permit addition
    calobj2.jones_array[0] = -6
    if method == "fast_concat":
        kwargs = {"axis": "jones", "inplace": True}
    else:
        kwargs = {}
    getattr(calobj, method)(calobj2, **kwargs)

    # check dimensionality of resulting object
    if caltype == "gain":
        assert calobj.gain_array.shape[-1] == 2
    else:
        assert calobj.delay_array.shape[-1] == 2
    assert sorted(calobj.jones_array) == [-6, -5]

    # test for when total_quality_array is present in first file but not second
    calobj = calobj_original.copy()
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.zeros(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = tqa
    getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in second file but not first
    calobj = calobj_original.copy()
    tqa = np.zeros(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in both
    calobj = calobj_original.copy()
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tqa *= 2
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    np.testing.assert_allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    if caltype == "delay":
        # test for when quality array is present in first file but
        # not in second
        calobj = calobj_original.copy()
        qa = np.ones(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2), dtype=float)
        tot_qa = np.concatenate([qa, qa2], axis=3)
        calobj.quality_array = qa
        calobj2.quality_array = None
        getattr(calobj, method)(calobj2, **kwargs)
        np.testing.assert_allclose(
            calobj.quality_array,
            tot_qa,
            rtol=calobj._quality_array.tols[0],
            atol=calobj._quality_array.tols[1],
        )

        # test for when quality array is present in second file but not first
        calobj = calobj_original.copy()
        qa = np.zeros(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2), dtype=float)

        tot_qa = np.concatenate([qa, qa2], axis=3)

        calobj.quality_array = None
        calobj2.quality_array = qa2
        getattr(calobj, method)(calobj2, **kwargs)
        np.testing.assert_allclose(
            calobj.quality_array,
            tot_qa,
            rtol=calobj._quality_array.tols[0],
            atol=calobj._quality_array.tols[1],
        )

    # Out of order - jones
    calobj = calobj_original.copy()
    calobj2 = calobj_original.copy()
    calobj.jones_array[0] = -6
    getattr(calobj, method)(calobj2, **kwargs)
    calobj2 = calobj.copy()
    calobj.select(jones=-5)
    calobj.history = calobj_original.history
    assert calobj == calobj_original
    calobj2.select(jones=-6)
    calobj2.jones_array[:] = -5
    calobj2.history = calobj_original.history
    assert calobj2 == calobj_original


@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
@pytest.mark.parametrize("quality", [True, False])
def test_add_jones_multispw(method, quality, multi_spw_gain):
    """Test adding Jones axes between two UVCal objects"""
    calobj = multi_spw_gain

    if not quality:
        calobj.quality_array = None

    calobj2 = calobj.copy()

    # artificially change the Jones value to permit addition
    calobj2.jones_array[0] = -6
    if method == "fast_concat":
        kwargs = {"axis": "jones", "inplace": True}
    else:
        kwargs = {}
    getattr(calobj, method)(calobj2, **kwargs)

    # check dimensionality of resulting object
    assert calobj.gain_array.shape[-1] == 2

    assert sorted(calobj.jones_array) == [-6, -5]


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add(caltype, method, gain_data, delay_data):
    """Test miscellaneous aspects of add method"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    calobj2 = calobj.copy()

    # test not-in-place addition
    calobj_original = calobj.copy()
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    calobj.select(antenna_nums=ants1)
    calobj2.select(antenna_nums=ants2)

    if method == "fast_concat":
        kwargs = {"axis": "antenna", "inplace": False}
    else:
        kwargs = {}

    calobj_add = getattr(calobj, method)(calobj2, **kwargs)
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        calobj_original.history + "  Downselected to specific "
        "antennas using pyuvdata. Combined "
        "data along antenna axis using pyuvdata.",
        calobj_add.history,
    )
    calobj_add.history = calobj_original.history
    assert calobj_add == calobj_original

    # test history concatenation
    calobj.history = calobj_original.history
    if caltype == "gain":
        calobj2.history = "Some random history string OMNI_RUN:"
    else:
        calobj2.history = "Some random history string firstcal.py"
    new_cal = getattr(calobj, method)(calobj2, **kwargs)

    additional_history = "Some random history string"
    assert utils.history._check_histories(
        calobj_original.history + " Combined data along antenna axis "
        "using pyuvdata. Unique part of next object history follows.  "
        + additional_history,
        new_cal.history,
    )

    kwargs["verbose_history"] = True
    new_cal = getattr(calobj, method)(calobj2, **kwargs)
    assert utils.history._check_histories(
        calobj_original.history + " Combined data along antenna axis "
        "using pyuvdata. Next object history follows.  " + calobj2.history,
        new_cal.history,
    )


@pytest.mark.parametrize(
    ["ant", "freq", "time", "jones"],
    [
        [True, True, False, False],
        [False, False, True, True],
        [True, True, True, False],
        [False, True, True, True],
        [True, True, True, True],
    ],
)
@pytest.mark.parametrize("in_order", [True, False])
def test_add_multiple_axes(gain_data, ant, freq, time, jones, in_order):
    """Test addition along multiple axes"""
    calobj_full = gain_data
    calobj_full.select(
        antenna_nums=calobj_full.ant_array[:4],
        frequencies=calobj_full.freq_array[:4],
        times=calobj_full.time_array[:2],
    )

    # add more jones terms to allow for better testing of selections
    while calobj_full.Njones < 2:
        new_jones = np.min(calobj_full.jones_array) - 1
        calobj_full.jones_array = np.append(calobj_full.jones_array, new_jones)
        calobj_full.Njones += 1
        calobj_full.flag_array = np.concatenate(
            (calobj_full.flag_array, calobj_full.flag_array[:, :, :, [-1]]), axis=3
        )
        calobj_full.gain_array = np.concatenate(
            (calobj_full.gain_array, calobj_full.gain_array[:, :, :, [-1]]), axis=3
        )
        calobj_full.quality_array = np.concatenate(
            (calobj_full.quality_array, calobj_full.quality_array[:, :, :, [-1]]),
            axis=3,
        )
    calobj = calobj_full.copy()
    calobj2 = calobj_full.copy()

    ants1 = None
    ants2 = None
    freqs1 = None
    freqs2 = None
    times1 = None
    times2 = None
    jones1 = None
    jones2 = None

    if ant:
        ants1 = calobj.ant_array[calobj.Nants_data // 2 :]
        ants2 = calobj.ant_array[: calobj.Nants_data // 2]
    if freq:
        freqs1 = calobj.freq_array[: calobj.Nfreqs // 2]
        freqs2 = calobj.freq_array[calobj.Nfreqs // 2 :]
    if time:
        times1 = calobj.time_array[: calobj.Ntimes // 2]
        times2 = calobj.time_array[calobj.Ntimes // 2 :]
    if jones:
        jones1 = calobj.jones_array[: calobj.Njones // 2]
        jones2 = calobj.jones_array[calobj.Njones // 2 :]

    # perform select
    calobj.select(antenna_nums=ants1, frequencies=freqs1, times=times1, jones=jones1)
    calobj2.select(antenna_nums=ants2, frequencies=freqs2, times=times2, jones=jones2)

    if in_order:
        calobj3 = calobj + calobj2
    else:
        calobj3 = calobj2 + calobj

    # remove the missing parts from calobj_full
    if ant:
        ant1_inds = np.nonzero(np.isin(calobj_full.ant_array, ants1))[0]
        ant2_inds = np.nonzero(np.isin(calobj_full.ant_array, ants2))[0]
    else:
        ant1_inds = np.arange(calobj_full.Nants_data)
        ant2_inds = np.arange(calobj_full.Nants_data)
    if freq:
        freq1_inds = np.nonzero(np.isin(calobj_full.freq_array, freqs1))[0]
        freq2_inds = np.nonzero(np.isin(calobj_full.freq_array, freqs2))[0]
    else:
        freq1_inds = np.arange(calobj_full.Nfreqs)
        freq2_inds = np.arange(calobj_full.Nfreqs)
    if time:
        time1_inds = np.nonzero(np.isin(calobj_full.time_array, times1))[0]
        time2_inds = np.nonzero(np.isin(calobj_full.time_array, times2))[0]
    else:
        time1_inds = np.arange(calobj_full.Ntimes)
        time2_inds = np.arange(calobj_full.Ntimes)
    if jones:
        jones1_inds = np.nonzero(np.isin(calobj_full.jones_array, jones1))[0]
        jones2_inds = np.nonzero(np.isin(calobj_full.jones_array, jones2))[0]
    else:
        jones1_inds = np.arange(calobj_full.Njones)
        jones2_inds = np.arange(calobj_full.Njones)
    axis_dict = {
        "ant": {"axis": 0, 1: ant1_inds, 2: ant2_inds},
        "freq": {"axis": 1, 1: freq1_inds, 2: freq2_inds},
        "time": {"axis": 2, 1: time1_inds, 2: time2_inds},
        "jones": {"axis": 3, 1: jones1_inds, 2: jones2_inds},
    }
    axes_used = []
    if ant:
        axes_used.append("ant")
    if freq:
        axes_used.append("freq")
    if time:
        axes_used.append("time")
    if jones:
        axes_used.append("jones")
    axis_list = []
    for n_comb in range(1, len(axes_used)):
        axis_list += list(itertools.combinations(axes_used, n_comb))
    for al in axis_list:
        set_use = [1, 1, 1, 1]
        for axis in al:
            set_use[axis_dict[axis]["axis"]] = 2
        inds = np.ix_(
            axis_dict["ant"][set_use[0]],
            axis_dict["freq"][set_use[1]],
            axis_dict["time"][set_use[2]],
            axis_dict["jones"][set_use[3]],
        )
        calobj_full.gain_array[inds] = 0
        calobj_full.quality_array[inds] = 0
        calobj_full.flag_array[inds] = True

    # reset history to equality passes
    calobj3.history = calobj_full.history

    assert calobj3 == calobj_full


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("time_range", [True, False])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add_errors(caltype, time_range, method, gain_data, delay_data):
    """Test behavior that will raise errors"""
    if caltype == "gain":
        calobj = gain_data.copy()
    else:
        calobj = delay_data.copy()

    if time_range:
        calobj = time_array_to_time_range(calobj)

    calobj2 = calobj.copy()

    if method == "fast_concat":
        kwargs = {"axis": "antenna", "inplace": True}
    else:
        kwargs = {}

    if method == "fast_concat":
        # test unknown axis values
        allowed_axes = ["antenna", "time", "jones"]
        if caltype == "gain":
            allowed_axes.append("freq")
        with pytest.raises(
            ValueError, match="Axis must be one of: " + ", ".join(allowed_axes)
        ):
            calobj.fast_concat(calobj2, axis="foo")

    else:
        # test addition of two identical objects
        if time_range:
            msg = "A time_range overlaps in the two objects."
        else:
            msg = "These objects have overlapping data and cannot be combined."

        with pytest.raises(ValueError, match=msg):
            calobj + calobj2

    # test addition of UVCal and non-UVCal object
    with pytest.raises(ValueError, match="Only UVCal "):
        getattr(calobj, method)("foo", **kwargs)

    # test time_range, time_array mismatch
    calobj2 = calobj.copy()
    if time_range:
        calobj2.time_array = gain_data.time_array
        calobj2.time_range = None
        calobj2.lst_range = None
    else:
        calobj2 = time_array_to_time_range(calobj2)

    calobj2.set_lsts_from_time_array()

    if method == "fast_concat":
        kwargs = {"axis": "time", "inplace": True}
    else:
        kwargs = {}

    with pytest.raises(
        ValueError,
        match="Some objects have a time_array while others do not. All objects must "
        "either have or not have a time_array.",
    ):
        getattr(calobj, method)(calobj2, **kwargs)


@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add_errors_wideband_mismatch(gain_data, wideband_gain, method):
    if method == "fast_concat":
        kwargs = {"axis": "time", "inplace": True}
    else:
        kwargs = {}

    # test wide_band mismatch
    with pytest.raises(
        ValueError,
        match="To combine these data, wide_band must be set to the same value",
    ):
        getattr(gain_data, method)(wideband_gain, **kwargs)


@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add_error_wideband(wideband_gain, delay_data, method):
    kwargs = {"axis": "antenna"} if method == "fast_concat" else {}
    with pytest.raises(
        ValueError, match="UVParameter cal_type does not match. Cannot combine objects."
    ):
        _ = getattr(wideband_gain, method)(delay_data, **kwargs)


@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add_errors_telescope(caltype, method, gain_data, delay_data):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    calobj2 = calobj.copy()

    # Do this to offset the solns so as to be able to add them
    calobj2.time_array += 1 / 1440

    if method == "fast_concat":
        kwargs = {"axis": "antenna", "inplace": True}
    else:
        kwargs = {}

        # test compatibility param mismatch
    calobj2.telescope.name = "PAPER"
    with pytest.raises(ValueError, match="Parameter Telescope.name does not match"):
        getattr(calobj, method)(calobj2, **kwargs)


# write it out this way because cannot combine along the freq axis with old delay types
@pytest.mark.parametrize(
    ["axis", "caltype"],
    [
        ["antenna", "gain"],
        ["antenna", "delay"],
        ["freq", "gain"],
        ["time", "gain"],
        ["time", "delay"],
        ["jones", "gain"],
        ["jones", "delay"],
        ["spw", "gain"],
        ["spw", "delay"],
    ],
)
def test_fast_concat_multiple_files(
    gain_data, delay_data, wideband_gain, multi_spw_delay, axis, caltype
):
    if axis == "spw":
        if caltype == "gain":
            calobj_full = wideband_gain
        else:
            calobj_full = multi_spw_delay
        n_objects = 3
    else:
        if caltype == "gain":
            calobj_full = gain_data
        else:
            calobj_full = delay_data
        n_objects = 4

    # add more jones terms to allow for better testing of selections
    if axis != "antenna":
        total_quality = True
    else:
        total_quality = False
    extend_jones_axis(calobj_full, total_quality=total_quality)

    axis_dict = {
        "antenna": {
            "arr_use": calobj_full.ant_array,
            "axis_len": calobj_full.Nants_data,
            "select_param": "antenna_nums",
        },
        "freq": {
            "arr_use": np.arange(calobj_full.Nfreqs),
            "axis_len": calobj_full.Nfreqs,
            "select_param": "freq_chans",
        },
        "spw": {
            "arr_use": calobj_full.spw_array,
            "axis_len": calobj_full.Nspws,
            "select_param": "spws",
        },
        "time": {
            "arr_use": calobj_full.time_array,
            "axis_len": calobj_full.Ntimes,
            "select_param": "times",
        },
        "jones": {
            "arr_use": calobj_full.jones_array,
            "axis_len": calobj_full.Njones,
            "select_param": "jones",
        },
    }

    select_params = {}
    ind_ranges = {
        0: np.arange(axis_dict[axis]["axis_len"] // n_objects),
        1: np.arange(
            axis_dict[axis]["axis_len"] // n_objects,
            axis_dict[axis]["axis_len"] * 2 // n_objects,
        ),
        2: np.arange(
            axis_dict[axis]["axis_len"] * 2 // n_objects,
            axis_dict[axis]["axis_len"] * 3 // n_objects,
        ),
        3: np.arange(
            axis_dict[axis]["axis_len"] * 3 // n_objects, axis_dict[axis]["axis_len"]
        ),
    }
    for obj_num in range(n_objects):
        select_params[obj_num] = {}
        select_params[obj_num][axis_dict[axis]["select_param"]] = axis_dict[axis][
            "arr_use"
        ][ind_ranges[obj_num]]
    calobj0 = calobj_full.select(**select_params[0], inplace=False)
    calobj1 = calobj_full.select(**select_params[1], inplace=False)
    calobj2 = calobj_full.select(**select_params[2], inplace=False)
    if n_objects == 4:
        calobj3 = calobj_full.select(**select_params[3], inplace=False)

    concat_list = [calobj1, calobj2]
    if n_objects == 4:
        concat_list.append(calobj3)
    calobj_final = calobj0.fast_concat(concat_list, axis=axis, inplace=False)
    calobj_final.history = calobj_full.history
    assert calobj_final == calobj_full


def test_parameter_warnings(gain_data):
    """Test changing a parameter that will raise a warning"""
    # change observer and select frequencies
    calobj = gain_data
    calobj2 = calobj.copy()

    calobj2.observer = "mystery_person"
    freqs1 = calobj.freq_array[np.arange(0, 5)]
    freqs2 = calobj2.freq_array[np.arange(5, 10)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)
    with check_warnings(UserWarning, match="UVParameter observer does not match"):
        calobj.__iadd__(calobj2)

    freqs = np.concatenate([freqs1, freqs2])
    np.testing.assert_allclose(
        calobj.freq_array,
        freqs,
        rtol=calobj._freq_array.tols[0],
        atol=calobj._freq_array.tols[1],
    )


@pytest.mark.parametrize(
    "method", ["read_ms_cal", "read_fhd_cal", "read_calh5", "read_calfits"]
)
def test_multi_files_errors(method):
    uvc = UVCal()
    kwargs = {"obs_file": "dummy"} if method == "read_fhd_cal" else {}
    with pytest.raises(
        ValueError, match="Use the generic `UVCal.read` method to read multiple files."
    ):
        getattr(uvc, method)(["foo", "bar"], **kwargs)


@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
@pytest.mark.parametrize("nfiles", [3, 4])
def test_multi_files(caltype, method, gain_data, delay_data, tmp_path, nfiles):
    """Test read function when multiple files are included"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

    calobj_full = calobj.copy()
    assert calobj.Ntimes >= nfiles
    nt_per_files = calobj.Ntimes // nfiles
    write_files = []
    filenames = []
    # Break up cal object into nfiles objects, divided in time
    for filei in range(nfiles):
        nt_start = filei * nt_per_files
        if filei < nfiles - 1:
            nt_end = (filei + 1) * nt_per_files
        else:
            nt_end = calobj.Ntimes
        this_times = calobj.time_array[nt_start:nt_end]

        this_obj = calobj.select(times=this_times, inplace=False)
        this_fname = f"read_multi{filei}.calh5"
        filenames.append(this_fname)
        this_file = str(tmp_path / this_fname)
        this_obj.write_calh5(this_file)
        write_files.append(this_file)

    # Read all files together
    if method == "fast_concat":
        calobj = UVCal.from_file(write_files, axis="time")
        nrep = 1
    else:
        with check_warnings(None):
            calobj.read(write_files)
        nrep = 2

    assert utils.history._check_histories(
        calobj_full.history
        + "  Downselected to specific times using pyuvdata."
        + " Combined data along time axis using pyuvdata." * nrep,
        calobj.history,
    )
    assert calobj.filename == filenames
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # check metadata only read
    calobj = UVCal.from_file(write_files, read_data=False)
    calobj_full_metadata_only = calobj_full.copy(metadata_only=True)

    calobj.history = calobj_full_metadata_only.history
    assert calobj == calobj_full_metadata_only


def test_uvcal_get_methods(gain_data):
    # load data
    uvc = gain_data

    # test get methods: add in a known value and make sure it is returned
    key = (10, "Jee")
    uvc.gain_array[1] = 0.0
    gain_arr = uvc.get_gains(key)
    flag_arr = uvc.get_flags(key)
    quality_arr = uvc.get_quality(key)

    # test shapes
    np.testing.assert_allclose(gain_arr, 0.0)
    assert gain_arr.shape == (uvc.Nfreqs, uvc.Ntimes)
    assert flag_arr.shape == (uvc.Nfreqs, uvc.Ntimes)
    assert quality_arr.shape == (uvc.Nfreqs, uvc.Ntimes)

    # test against by-hand indexing
    expected_array = uvc.gain_array[
        uvc.ant_array.tolist().index(10), :, :, uvc.jones_array.tolist().index(-5)
    ]
    np.testing.assert_array_almost_equal(gain_arr, expected_array)

    # test variable key input
    gain_arr2 = uvc.get_gains(key[0], key[1])
    np.testing.assert_array_almost_equal(gain_arr, gain_arr2)
    gain_arr2 = uvc.get_gains(key[0])
    np.testing.assert_array_almost_equal(gain_arr, gain_arr2)
    gain_arr2 = uvc.get_gains(key[:1])
    np.testing.assert_array_almost_equal(gain_arr, gain_arr2)
    gain_arr2 = uvc.get_gains(10, -5)
    np.testing.assert_array_almost_equal(gain_arr, gain_arr2)
    gain_arr2 = uvc.get_gains(10, "x")
    np.testing.assert_array_almost_equal(gain_arr, gain_arr2)

    # check has_key
    assert uvc._key_exists(antnum=10)
    assert uvc._key_exists(jpol="Jee")
    assert uvc._key_exists(antnum=10, jpol="Jee")
    assert not uvc._key_exists(antnum=10, jpol="Jnn")
    assert not uvc._key_exists(antnum=101, jpol="Jee")

    # test exceptions
    with pytest.raises(ValueError, match="1 not found in ant_array"):
        uvc.get_gains(1)
    with pytest.raises(ValueError, match="-6 not found in jones_array"):
        uvc.get_gains((10, "Jnn"))
    uvc.cal_type = "delay"
    with pytest.raises(
        ValueError, match=re.escape("cal_type must be 'gain' for get_gains() method")
    ):
        uvc.get_gains(10)


@pytest.mark.parametrize("file_type", ["calfits", "calh5", "ms"])
def test_write_read_optional_attrs(gain_data, tmp_path, file_type):
    if file_type == "ms":
        pytest.importorskip("casacore")

    # read a test file
    cal_in = gain_data

    # set some optional parameters
    cal_in.pol_convention = "sum"
    with check_warnings(
        UserWarning,
        match="gain_scale should be set if pol_convention is set. When "
        "calibrating data with `uvcalibrate`, pol_convention will be ignored if "
        "gain_scale is not set.",
    ):
        cal_in.check()

    cal_in.gain_scale = "Jy"

    # write
    outfile = str(tmp_path / ("test." + file_type))
    write_method = "write_" + file_type
    if file_type == "ms":
        write_method += "_cal"
    getattr(cal_in, write_method)(outfile)

    # read and compare
    # also check that passing a single file in a list works properly
    cal_in2 = UVCal.from_file([outfile], file_type=file_type)

    # some things are different for ms and it's ok. reset those
    if file_type == "ms":
        cal_in2.scan_number_array = None
        cal_in2.scan_number_array = None
        cal_in2.extra_keywords = cal_in.extra_keywords
        cal_in2.history = cal_in.history

    assert cal_in == cal_in2


@pytest.mark.parametrize("caltype", ["gain", "delay", None])
def test_copy(gain_data, delay_data, caltype):
    """Test the copy method"""
    if caltype == "gain":
        uv_object = gain_data
    elif caltype == "delay":
        uv_object = delay_data
    else:
        uv_object = gain_data
        uv_object.cal_type = caltype

    if caltype is not None:
        uv_object_copy = uv_object.copy()
    else:
        with pytest.raises(
            ValueError,
            match="Cannot get required data params because cal_type is not set.",
        ):
            uv_object_copy = uv_object.copy()
        return
    assert uv_object_copy == uv_object

    uv_object_copy = uv_object.copy(metadata_only=True)
    assert uv_object_copy.metadata_only

    for name in uv_object._data_params:
        setattr(uv_object, name, None)
    assert uv_object_copy == uv_object

    uv_object_copy = uv_object.copy()
    assert uv_object_copy == uv_object

    return


@pytest.mark.parametrize("antnamefix", ["all", "partial"])
def test_match_antpos_antname(gain_data, antnamefix, tmp_path):
    # fix the antenna names in the uvcal object to match telescope object
    new_names = np.array(
        [name.replace("ant", "HH") for name in gain_data.telescope.antenna_names]
    )
    if antnamefix == "all":
        gain_data.telescope.antenna_names = new_names
    else:
        gain_data.telescope.antenna_names[0 : gain_data.telescope.Nants // 2] = (
            new_names[0 : gain_data.telescope.Nants // 2]
        )

    # remove the antenna_positions to test matching them on read
    write_file = str(tmp_path / "test.calfits")
    write_file2 = str(tmp_path / "test2.calfits")
    gain_data.write_calfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
        primary_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["ANTENNAS"]]

        table = Table(ant_hdu.data)
        del table["ANTXYZ"]
        ant_hdu = fits.BinTableHDU(table)
        ant_hdu.header["EXTNAME"] = "ANTENNAS"

        hdulist = fits.HDUList([primary_hdu, ant_hdu])

        hdulist.writeto(write_file2)
        hdulist.close()

    gain_data2 = UVCal.from_file(write_file2)

    assert gain_data2.telescope.antenna_positions is not None
    assert gain_data == gain_data2


@pytest.mark.parametrize("modtype", ["rename", "swap"])
def test_set_antpos_from_telescope_errors(gain_data, modtype, tmp_path):
    """Test that setting antenna positions doesn't happen if ants don't match."""
    # fix the antenna names in the uvcal object to match telescope object
    new_names = np.array(
        [name.replace("ant", "HH") for name in gain_data.telescope.antenna_names]
    )
    gain_data.telescope.antenna_names = new_names

    if modtype == "rename":
        # change the name & number of one of the antennas
        orig_num = gain_data.telescope.antenna_numbers[0]
        gain_data.telescope.antenna_names[0] = "HH400"
        gain_data.telescope.antenna_numbers[0] = 400
        gain_data.ant_array[np.where(gain_data.ant_array == orig_num)[0]] = 400
    else:
        # change the name of one antenna and swap the number with a different antenna
        orig_num = gain_data.telescope.antenna_numbers[0]
        gain_data.telescope.antenna_names[0] = "HH400"
        gain_data.telescope.antenna_numbers[0] = gain_data.telescope.antenna_numbers[1]
        gain_data.telescope.antenna_numbers[1] = orig_num

    # remove the antenna_positions to test matching them on read
    write_file = str(tmp_path / "test.calfits")
    write_file2 = str(tmp_path / "test2.calfits")
    gain_data.write_calfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = fits_utils._indexhdus(hdu_list)
        primary_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["ANTENNAS"]]

        table = Table(ant_hdu.data)
        del table["ANTXYZ"]
        ant_hdu = fits.BinTableHDU(table)
        ant_hdu.header["EXTNAME"] = "ANTENNAS"

        hdulist = fits.HDUList([primary_hdu, ant_hdu])

        hdulist.writeto(write_file2)
        hdulist.close()

    with (
        pytest.raises(
            ValueError,
            match="Required UVParameter _antenna_positions has not been set.",
        ),
        check_warnings(
            [UserWarning],
            match=[
                "Not all antennas have metadata in the known_telescope data. Not "
                "setting ['antenna_positions'].",
                "Required UVParameter _antenna_positions has not been set.",
            ],
        ),
    ):
        gain_data2 = UVCal.from_file(write_file2)

    with check_warnings(
        UserWarning,
        match=[
            "Not all antennas have metadata in the known_telescope data. Not "
            "setting ['antenna_positions'].",
            "Required UVParameter _antenna_positions has not been set.",
        ],
    ):
        gain_data2 = UVCal.from_file(write_file2, run_check=False)

    assert gain_data2.telescope.antenna_positions is None


def test_read_errors():
    with pytest.raises(
        ValueError,
        match="File type could not be determined, use the file_type keyword to specify "
        "the type.",
    ):
        UVCal.from_file("foo.blah")

    gainfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    with pytest.raises(
        ValueError,
        match="If filename is a list, tuple or array it cannot be nested or "
        "multi dimensional.",
    ):
        UVCal.from_file([[gainfile]])

    with pytest.raises(
        ValueError,
        match="The only supported file_types are 'calfits', 'calh5', 'fhd', and 'ms'.",
    ):
        UVCal.from_file(gainfile, file_type="foo")


@pytest.mark.parametrize("multi_spw", [True, False])
def test_init_from_uvdata(multi_spw, uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    if multi_spw:
        uvd.flex_spw_id_array = [1] * (uvd.Nfreqs // 2) + [2] * (uvd.Nfreqs // 2)
        uvd.spw_array = np.array([1, 2])
        uvd.Nspws = 2
        uvd.channel_width = np.full(uvd.Nfreqs, uvd.channel_width)
        uvd.check()

        uvc.flex_spw_id_array = [1] * (uvc.Nfreqs // 2) + [2] * (uvc.Nfreqs // 2)
        uvc.spw_array = np.array([1, 2])
        uvc.Nspws = 2
        uvc.channel_width = np.full(uvc.Nfreqs, uvc.channel_width)
        uvc.check()

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd, gain_convention=uvc.gain_convention, cal_style=uvc.cal_style
    )

    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.check()

    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


@pytest.mark.filterwarnings("ignore:Selected frequencies are not contiguous.")
@pytest.mark.parametrize("multi_spw", [True, False])
def test_init_from_uvdata_setfreqs(multi_spw, uvcalibrate_data):
    uvd, uvc = uvcalibrate_data
    channel_width = uvd.channel_width[0]
    freqs_use = uvd.freq_array[0:5]

    uvc2 = uvc.copy(metadata_only=True)

    uvc2.select(frequencies=freqs_use)

    if multi_spw:
        flex_spw_id_array = np.array([1, 1, 2, 2, 2])
        uvc2.flex_spw_id_array = flex_spw_id_array
        uvc2.spw_array = np.array([1, 2])
        uvc2.Nspws = 2
        uvc2.channel_width = np.full(uvc2.Nfreqs, uvc2.channel_width)
        # test passing a list instead of a single value
        channel_width = np.full(freqs_use.size, channel_width).tolist()
    else:
        flex_spw_id_array = np.zeros(5, dtype=int)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        gain_convention=uvc.gain_convention,
        cal_style=uvc.cal_style,
        freq_array=freqs_use,
        channel_width=channel_width,
        flex_spw_id_array=flex_spw_id_array,
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    uvc_new.history = uvc2.history

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


@pytest.mark.parametrize("metadata_only", [True, False])
def test_init_from_uvdata_settimes(metadata_only, uvcalibrate_data):
    uvd, uvc = uvcalibrate_data
    integration_time = np.mean(uvd.integration_time)
    times_use = uvc.time_array[0:3]

    if metadata_only:
        # test passing in a list of integration times
        integration_time = np.full(
            times_use.size, np.mean(uvd.integration_time)
        ).tolist()

    uvc2 = uvc.copy(metadata_only=metadata_only)

    uvc2.select(times=times_use)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        gain_convention=uvc.gain_convention,
        cal_style=uvc.cal_style,
        metadata_only=metadata_only,
        time_array=times_use,
        integration_time=integration_time,
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )

    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    if not metadata_only:
        uvc2.gain_array[:] = 1.0
        uvc2.quality_array = None
        uvc2.flag_array[:] = False
        uvc2.total_quality_array = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


def test_init_from_uvdata_setjones(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        gain_convention=uvc.gain_convention,
        cal_style=uvc.cal_style,
        jones_array=[-5, -6],
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


@pytest.mark.parametrize("pol", ["nn", "ee", "rr", "ll"])
def test_init_single_pol(uvcalibrate_data, pol):
    uvd, uvc = uvcalibrate_data

    if pol in ["ll", "rr"]:
        # convert to circular pol
        uvd.polarization_array = np.array([-1, -2, -3, -4])
        uvc.jones_array = np.array([-1, -2])

    # downselect to one pol
    uvd.select(polarizations=[pol])
    uvc.select(jones=[pol])

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd, gain_convention=uvc.gain_convention, cal_style=uvc.cal_style
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


def test_init_from_uvdata_circular_pol(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # convert to circular pol
    uvd.polarization_array = np.array([-1, -2, -3, -4])
    uvc.jones_array = np.array([-1, -2])

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd, gain_convention=uvc.gain_convention, cal_style=uvc.cal_style
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


def test_init_from_uvdata_sky(uvcalibrate_data, fhd_cal_raw):
    uvd, uvc = uvcalibrate_data
    uvc_sky = fhd_cal_raw

    # make cal object be a sky cal type
    uvc._set_sky()
    params_to_copy = [
        "sky_catalog",
        "ref_antenna_name",
        "diffuse_model",
        "baseline_range",
        "Nsources",
        "observer",
        "gain_scale",
        "git_hash_cal",
        "git_origin_cal",
        "extra_keywords",
    ]
    for param in params_to_copy:
        setattr(uvc, param, getattr(uvc_sky, param))

    # also set gain scale to test that it is set properly
    uvc.gain_scale = "Jy"

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        gain_convention=uvc.gain_convention,
        cal_style=uvc.cal_style,
        sky_catalog=uvc_sky.sky_catalog,
        ref_antenna_name=uvc_sky.ref_antenna_name,
        diffuse_model=uvc_sky.diffuse_model,
        baseline_range=uvc_sky.baseline_range,
        Nsources=uvc_sky.Nsources,
        observer=uvc_sky.observer,
        gain_scale="Jy",
        git_hash_cal=uvc_sky.git_hash_cal,
        git_origin_cal=uvc_sky.git_origin_cal,
        extra_keywords=uvc_sky.extra_keywords,
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention

    assert uvc_new == uvc2


@pytest.mark.parametrize("multi_spw", [True, False])
@pytest.mark.parametrize("set_frange", [True, False])
def test_init_from_uvdata_delay(multi_spw, set_frange, uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # make cal object be a delay cal type
    uvc2 = uvc.copy(metadata_only=True)
    uvc2.Nfreqs = 1
    uvc2.flex_spw_id_array = uvc2.freq_array = uvc2.channel_width = None
    uvc2.freq_range = np.array([[np.min(uvc.freq_array), np.max(uvc.freq_array)]])
    uvc2._set_delay()

    if multi_spw:
        spw_cut = uvd.Nfreqs // 2
        uvd.flex_spw_id_array = [1] * spw_cut + [2] * spw_cut
        uvd.spw_array = np.array([1, 2])
        uvd.Nspws = 2
        uvd.channel_width = np.full(uvd.Nfreqs, uvd.channel_width)
        uvd.check()

        uvc.flex_spw_id_array = np.asarray([1] * spw_cut + [2] * spw_cut)
        uvc.spw_array = np.array([1, 2])
        uvc.Nspws = 2
        uvc.channel_width = np.full(uvc.Nfreqs, uvc.channel_width)
        uvc.check()

    if multi_spw:
        uvc2.spw_array = np.array([1, 2])
        uvc2.Nspws = 2
        uvc2.freq_range = np.asarray(
            [
                [np.min(uvc.freq_array[0:spw_cut]), np.max(uvc.freq_array[0:spw_cut])],
                [np.min(uvc.freq_array[spw_cut:]), np.max(uvc.freq_array[spw_cut:])],
            ]
        )
    uvc2.check()

    if set_frange:
        freq_range = uvc2.freq_range
        if multi_spw:
            spw_array = uvc2.spw_array
        else:
            # check that it works with 1d array for one spw
            freq_range = np.squeeze(uvc2.freq_range)
            spw_array = None
    else:
        freq_range = None
        spw_array = None

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        gain_convention=uvc.gain_convention,
        cal_style=uvc.cal_style,
        cal_type="delay",
        freq_range=freq_range,
        spw_array=spw_array,
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


@pytest.mark.parametrize("multi_spw", [True, False])
@pytest.mark.parametrize("set_frange", [True, False])
def test_init_from_uvdata_wideband(multi_spw, set_frange, uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # make cal object be a wide-band cal
    uvc2 = uvc.copy(metadata_only=True)
    uvc2.flex_spw_id_array = uvc2.freq_array = uvc2.channel_width = None
    uvc2.Nfreqs = 1
    uvc2._set_wide_band()
    uvc2.freq_range = np.asarray([[np.min(uvc.freq_array), np.max(uvc.freq_array)]])

    if multi_spw:
        spw_cut = uvd.Nfreqs // 2
        uvd.flex_spw_id_array = [1] * spw_cut + [2] * spw_cut
        uvd.spw_array = np.array([1, 2])
        uvd.Nspws = 2
        uvd.channel_width = np.full(uvd.Nfreqs, uvd.channel_width)
        uvd.check()

        uvc.flex_spw_id_array = np.asarray([1] * spw_cut + [2] * spw_cut)
        uvc.spw_array = np.array([1, 2])
        uvc.Nspws = 2
        uvc.channel_width = np.full(uvc.Nfreqs, uvc.channel_width)
        uvc.check()

        uvc2.spw_array = np.array([1, 2])
        uvc2.Nspws = 2
        uvc2.freq_range = np.asarray(
            [
                [np.min(uvc.freq_array[0:spw_cut]), np.max(uvc.freq_array[0:spw_cut])],
                [np.min(uvc.freq_array[spw_cut:]), np.max(uvc.freq_array[spw_cut:])],
            ]
        )

    uvc2.check()

    if set_frange:
        freq_range = uvc2.freq_range
        if multi_spw:
            spw_array = uvc2.spw_array
        else:
            spw_array = None
    else:
        freq_range = None
        spw_array = None

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        gain_convention=uvc.gain_convention,
        cal_style=uvc.cal_style,
        wide_band=True,
        freq_range=freq_range,
        spw_array=spw_array,
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    np.testing.assert_allclose(
        uvc2.telescope.antenna_positions,
        uvc_new.telescope.antenna_positions,
        rtol=0,
        atol=0.1,
    )
    uvc_new.telescope.antenna_positions = uvc2.telescope.antenna_positions

    assert utils.history._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # the new one has an instrument set because UVData requires it
    assert uvc_new.telescope.instrument == uvd.telescope.instrument
    # remove it to match uvc2
    uvc_new.telescope.instrument = None

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new.pol_convention != uvc2.pol_convention
    uvc_new.pol_convention = uvc2.pol_convention
    assert uvc_new.gain_scale != uvc2.gain_scale
    uvc_new.gain_scale = uvc2.gain_scale

    assert uvc_new == uvc2


def test_init_from_uvdata_basic_errors(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    with pytest.raises(ValueError, match="uvdata must be a UVData object."):
        UVCal.initialize_from_uvdata(
            uvc, gain_convention=uvc.gain_convention, cal_style=uvc.cal_style
        )

    with pytest.raises(ValueError, match="cal_type must be either 'gain' or 'delay'."):
        UVCal.initialize_from_uvdata(
            uvd,
            gain_convention=uvc.gain_convention,
            cal_style=uvc.cal_style,
            cal_type="unknown",
        )

    with pytest.raises(
        ValueError,
        match="If cal_style is 'sky', ref_antenna_name and sky_catalog must be "
        "provided.",
    ):
        UVCal.initialize_from_uvdata(
            uvd, gain_convention=uvc.gain_convention, cal_style="sky"
        )

    uvd.polarization_array = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="you must set jones_array."):
        UVCal.initialize_from_uvdata(
            uvd, gain_convention=uvc.gain_convention, cal_style=uvc.cal_style
        )


def test_init_from_uvdata_freqrange_errors(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    with pytest.raises(
        ValueError,
        match=re.escape(
            "UVParameter _freq_range is not expected shape. Parameter shape is "
            "(1, 4), expected shape is (1, 2)."
        ),
    ):
        UVCal.initialize_from_uvdata(
            uvd,
            gain_convention=uvc.gain_convention,
            cal_style=uvc.cal_style,
            cal_type="delay",
            freq_range=[1e8, 1.2e8, 1.3e8, 1.5e8],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "UVParameter _spw_array is not expected shape. Parameter shape is "
            "(1,), expected shape is (2,)."
        ),
    ):
        UVCal.initialize_from_uvdata(
            uvd,
            gain_convention=uvc.gain_convention,
            cal_style=uvc.cal_style,
            cal_type="delay",
            freq_range=np.asarray([[1e8, 1.2e8], [1.3e8, 1.5e8]]),
        )


@pytest.mark.parametrize(
    "mode,cat_id", [["noop", None], ["force", 1], ["nocat", 1], ["muck", None]]
)
def test_add_phase_center(uvcal_phase_center, mode, cat_id):
    """
    Verify that if we attempt to add a source already in the catalog, we don't return
    an error but instead the call completes normally.
    """
    warntype = warnmsg = None
    if mode == "nocat":
        uvcal_phase_center.phase_center_catalog = None
    if mode == "muck":
        uvcal_phase_center.phase_center_catalog[1]["cat_lon"] = 0.0
        uvcal_phase_center.phase_center_catalog[0] = (
            uvcal_phase_center.phase_center_catalog.pop(1)
        )
        warntype = UserWarning
        warnmsg = "The provided name 3c84 is already used"

    with check_warnings(warntype, warnmsg):
        return_id = uvcal_phase_center._add_phase_center(
            "3c84",
            cat_type="sidereal",
            cat_lon=0.8718035968995141,
            cat_lat=0.7245157752262148,
            cat_frame="icrs",
            cat_epoch="j2000",
            cat_id=cat_id,
            force_update=(mode == "force"),
        )

    assert return_id == 1


def test_remove_phase_center_err(uvcal_phase_center):
    cat_id = 2
    with pytest.raises(IndexError, match="No source by that ID contained in the cata"):
        uvcal_phase_center._remove_phase_center(cat_id)
    assert cat_id not in uvcal_phase_center.phase_center_catalog


def test_remove_phase_center(uvcal_phase_center):
    cat_id = 1
    uvcal_phase_center._remove_phase_center(cat_id)
    assert cat_id not in uvcal_phase_center.phase_center_catalog


def test_clear_unused_phase_centers(uvcal_phase_center):
    pc_copy = copy.deepcopy(uvcal_phase_center.phase_center_catalog)
    uvcal_phase_center.phase_center_catalog[0] = (
        uvcal_phase_center.phase_center_catalog[1]
    )
    uvcal_phase_center.Nphase = 2
    assert uvcal_phase_center.phase_center_catalog != pc_copy

    uvcal_phase_center._clear_unused_phase_centers()
    assert uvcal_phase_center.phase_center_catalog == pc_copy
    assert uvcal_phase_center.Nphase == 1

    # Test no-op
    uvcal_phase_center._clear_unused_phase_centers()
    assert uvcal_phase_center.phase_center_catalog == pc_copy
    assert uvcal_phase_center.Nphase == 1


@pytest.mark.parametrize(
    "kwargs", [{}, {"catalog_identifier": "3c84"}, {"catalog_identifier": 1}]
)
def test_print_phase_center_catalog(uvcal_phase_center, kwargs):
    """
    Check that the 'standard' mode of print_object works.
    """
    check_str = (
        "   ID     Cat Entry          Type     Az/Lon/RA    El/Lat/Dec  Frame    Epoch \n"  # noqa
        "    #          Name                       hours           deg                 \n"  # noqa
        "------------------------------------------------------------------------------\n"  # noqa
        "    1          3c84      sidereal    3:19:48.16  +41:30:42.11   icrs  J2000.0 \n"  # noqa
    )

    table_str = uvcal_phase_center.print_phase_center_info(
        print_table=False, return_str=True, **kwargs
    )
    assert table_str == check_str

    # Make sure we can specify the object name and get the same result
    table_str = uvcal_phase_center.print_phase_center_info(
        print_table=False, return_str=True, catalog_identifier="3c84"
    )
    assert table_str == check_str

    # Make sure that things still work when we force the HMS format
    table_str = uvcal_phase_center.print_phase_center_info(
        print_table=False, return_str=True, hms_format=True
    )
    assert table_str == check_str


def test_update_phase_center_id(uvcal_phase_center):
    pc_dict = uvcal_phase_center.phase_center_catalog.copy()

    uvcal_phase_center._update_phase_center_id(1, new_id=2)
    assert pc_dict != uvcal_phase_center.phase_center_catalog
    assert pc_dict[1] == uvcal_phase_center.phase_center_catalog[2]


def test_consolidate_phase_center_err(uvcal_phase_center):
    with pytest.raises(ValueError, match="Either the reference_catalog or the other"):
        uvcal_phase_center._consolidate_phase_center_catalogs()


def test_consolidate_phase_center(uvcal_phase_center):
    uvcal1 = uvcal_phase_center.copy()
    uvcal2 = uvcal_phase_center.copy()

    # Test the no-op first
    uvcal1._consolidate_phase_center_catalogs(other=uvcal2)
    assert uvcal1 == uvcal_phase_center
    assert uvcal2 == uvcal_phase_center

    uvcal1._consolidate_phase_center_catalogs(
        other=uvcal2, reference_catalog=uvcal1.phase_center_catalog
    )
    assert uvcal1 == uvcal_phase_center
    assert uvcal2 == uvcal_phase_center

    # Update uvcal2 with a modified entry
    uvcal2.phase_center_catalog[1]["cat_name"] = "WHATISTHISTHING"
    uvcal1._consolidate_phase_center_catalogs(
        other=uvcal2, reference_catalog=uvcal1.phase_center_catalog
    )
    assert uvcal1.phase_center_catalog != uvcal_phase_center.phase_center_catalog
    assert uvcal1.phase_center_catalog[1] == uvcal_phase_center.phase_center_catalog[1]

    uvcal1._clear_unused_phase_centers()
    uvcal1._consolidate_phase_center_catalogs(other=uvcal2, ignore_name=True)

    assert uvcal1 == uvcal_phase_center


@pytest.mark.parametrize(
    "func,suffix", [["write_ms_cal", "ms"], ["write_calh5", "calh5"]]
)
def test_flex_jones_write(multi_spw_gain, func, suffix, tmp_path):
    if suffix == "ms":
        pytest.importorskip("casacore")

    uvc_copy = multi_spw_gain.copy()
    uvc_copy.jones_array[0] = -6
    multi_spw_gain += uvc_copy
    multi_spw_gain.convert_to_flex_jones()
    multi_spw_gain.ref_antenna_name = multi_spw_gain.telescope.antenna_names[0]

    filename = os.path.join(tmp_path, "flex_jones_write." + suffix)
    getattr(multi_spw_gain, func)(filename)

    uvc = UVCal()
    uvc.read(filename)
    uvc.history = multi_spw_gain.history
    if suffix == "ms":
        # Handle some extra bits here for MS-type
        uvc.extra_keywords = multi_spw_gain.extra_keywords
        uvc.scan_number_array = multi_spw_gain.scan_number_array

    assert uvc == multi_spw_gain


@pytest.mark.parametrize("func", ["__add__", "fast_concat"])
@pytest.mark.parametrize("caltype", ["delay", "gain"])
def test_flex_jones_divide_and_add(multi_spw_gain, multi_spw_delay, func, caltype):
    if caltype == "delay":
        uvc = multi_spw_delay
        kwargs = {} if (func == "__add__") else {"axis": "spw"}
    elif caltype == "gain":
        uvc = multi_spw_gain
        kwargs = {} if (func == "__add__") else {"axis": "freq"}
    uvc_copy = uvc.copy()

    uvc_copy.jones_array[0] = -6

    # We know add works across jones, so use this as baseline
    uvcomb1 = uvc + uvc_copy
    uvcomb1.convert_to_flex_jones()

    # Modify the spws so that they align nicely
    uvc_copy.spw_array += uvc_copy.Nspws
    if caltype == "gain":
        uvc_copy.flex_spw_id_array += uvc_copy.Nspws
    uvc.convert_to_flex_jones()
    uvc_copy.convert_to_flex_jones()

    uvcomb2 = getattr(uvc, func)(uvc_copy, **kwargs)

    # Bypass histories since they are different
    uvcomb1.history = uvcomb2.history = None

    assert uvcomb1 == uvcomb2


@pytest.mark.filterwarnings("ignore:combine_spws is True but there are not matched")
@pytest.mark.parametrize("mode", ["make", "convert", "meta", "single"])
@pytest.mark.parametrize("caltype", ["delay", "gain"])
def test_flex_jones_roundtrip(multi_spw_gain, multi_spw_delay, mode, caltype):
    if caltype == "delay":
        uvc = multi_spw_delay
    elif caltype == "gain":
        uvc = multi_spw_gain
    uvc_copy = uvc.copy()

    if mode == "single":
        pass
    elif caltype == "delay":
        uvc_copy.delay_array *= 1.56  # adjust gains to make them pol-unique
        if mode == "make":
            uvc.flag_array[:, 1::2] = True
            uvc.delay_array[:, 1::2] = 0.0
            uvc.quality_array[:, 1::2] = 0.0
            uvc_copy.flag_array[:, ::2] = True
            uvc_copy.delay_array[:, ::2] = 0.0
            uvc_copy.quality_array[:, ::2] = 0.0
    elif caltype == "gain":
        uvc_copy.gain_array *= 2 + 3j  # adjust gains to make them pol-unique
        if mode == "make":
            uvc.flag_array[:, uvc.flex_spw_id_array == 2] = True
            uvc.gain_array[:, uvc.flex_spw_id_array == 2] = 0.0
            uvc.quality_array[:, uvc.flex_spw_id_array == 2] = 0.0
            uvc_copy.flag_array[:, uvc.flex_spw_id_array == 1] = True
            uvc_copy.gain_array[:, uvc.flex_spw_id_array == 1] = 0.0
            uvc_copy.quality_array[:, uvc.flex_spw_id_array == 1] = 0.0

    if mode != "single":
        # Spoof a second polarization
        uvc_copy.jones_array[0] = -6
        uvc += uvc_copy
        uvc_copy = uvc.copy()

    uvc.remove_flex_jones()
    assert uvc_copy == uvc

    if mode == "convert":
        flex_jones_check = ([-5] * uvc.Nspws) + ([-6] * uvc.Nspws)
        with pytest.raises(ValueError, match="Cannot make a flex-pol UVCal object"):
            uvc._make_flex_jones()
        uvc.convert_to_flex_jones()
        with pytest.raises(ValueError, match="This is already a flex-pol object"):
            uvc.convert_to_flex_jones()
    elif mode == "make":
        flex_jones_check = [-5, -6] if caltype == "gain" else [-5, -6, -5]
        uvc._make_flex_jones()
    elif mode == "meta":
        flex_jones_check = ([-5] * uvc.Nspws) + ([-6] * uvc.Nspws)
        for name in uvc._data_params:
            setattr(uvc, name, None)
            setattr(uvc_copy, name, None)
        with pytest.raises(ValueError, match="Cannot make a metadata_only UVCal"):
            uvc._make_flex_jones()
        uvc.convert_to_flex_jones()
    elif mode == "single":
        flex_jones_check = [-5] * uvc.Nspws
        uvc._make_flex_jones()

    assert uvc.Njones == 1
    assert np.array_equal(uvc.flex_jones_array, flex_jones_check)
    assert uvc != uvc_copy

    uvc.remove_flex_jones()

    assert uvc == uvc_copy


def test_make_flex_jones_flagged_window(multi_spw_gain):
    multi_spw_gain.flag_array[:, multi_spw_gain.flex_spw_id_array == 2] = True
    uvc_copy = multi_spw_gain.copy()

    uvc_spoof = multi_spw_gain.copy()
    uvc_spoof.jones_array[0] = -6
    uvc_spoof.flag_array[:] = True
    multi_spw_gain += uvc_spoof

    multi_spw_gain._make_flex_jones()
    assert np.array_equal(multi_spw_gain.flex_jones_array, [-5, -5])
    multi_spw_gain.remove_flex_jones(combine_spws=False)

    assert uvc_copy.history in multi_spw_gain.history
    uvc_copy.history = multi_spw_gain.history
    assert multi_spw_gain == uvc_copy


def test_flex_jones_select_err(multi_spw_gain):
    uvc_spoof = multi_spw_gain.copy()
    uvc_spoof.jones_array[0] = -6
    multi_spw_gain += uvc_spoof
    multi_spw_gain.convert_to_flex_jones()

    with pytest.raises(ValueError, match="No data matching this Jones term"):
        multi_spw_gain.select(jones=[-6], spws=1, strict=True)

    with pytest.raises(ValueError, match="The Jones term -1000"):
        multi_spw_gain.select(jones=-1000, spws=1, strict=True)


def test_remove_flex_jones_dup_err(multi_spw_gain):
    uvc_spoof = multi_spw_gain.copy()
    uvc_spoof.jones_array[0] = -6
    multi_spw_gain += uvc_spoof
    multi_spw_gain.convert_to_flex_jones()

    multi_spw_gain.flex_jones_array[:] = -5
    with pytest.raises(ValueError, match="Some spectral windows have identical"):
        multi_spw_gain.remove_flex_jones()


@pytest.mark.parametrize("mode", ["delay", "gain"])
def test_flex_jones_shuffle(multi_spw_gain, multi_spw_delay, mode):
    if mode == "gain":
        uvc = multi_spw_gain
    elif mode == "delay":
        uvc = multi_spw_delay
    uvc_spoof = uvc.copy()
    uvc_spoof.jones_array[0] = -6
    uvc += uvc_spoof
    uvc.convert_to_flex_jones()

    uvc1 = uvc.select(jones=[-5], inplace=False)
    uvc2 = uvc.select(jones=[-6], inplace=False)

    assert uvc1 != uvc2

    uvc_comb = uvc1 + uvc2
    assert uvc.history in uvc_comb.history
    uvc_comb.history = uvc.history
    assert uvc_comb == uvc

    uvc_comb = uvc2.fast_concat(uvc1, axis=("spw" if (mode == "delay") else "freq"))
    uvc_comb.history = uvc.history
    assert uvc_comb != uvc
    uvc_comb.reorder_freqs(spw_order="number")
    assert uvc_comb == uvc


@pytest.mark.parametrize("func", ["__add__", "__iadd__", "fast_concat"])
def test_flex_jones_add_errs(multi_spw_gain, func):
    uvc_copy = multi_spw_gain.copy()
    multi_spw_gain.convert_to_flex_jones()

    kwargs = {"axis": "spw"} if func == "fast_concat" else {}
    with pytest.raises(ValueError, match="be either set to regular or flex-jones."):
        getattr(multi_spw_gain, func)(uvc_copy, **kwargs)


def test_phase_center_add_err(uvcal_phase_center):
    uvcopy = uvcal_phase_center.copy()
    uvcopy.phase_center_catalog[1]["cat_name"] = "whoami"
    uvcopy.jones_array[0] = -6
    with pytest.raises(ValueError, match="with different phase centers."):
        _ = uvcopy + uvcal_phase_center

    # Test what happens if no phase center catalog period
    uvcopy.phase_center_catalog = None
    with pytest.raises(ValueError, match="To combine these data, phase_center_"):
        _ = uvcopy + uvcal_phase_center


@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.parametrize(
    "func,kwargs", [["__add__", {}], ["fast_concat", {"axis": "time"}]]
)
def test_phase_center_add(uvcal_phase_center, func, kwargs):
    uvcopy = uvcal_phase_center.copy()
    uvcopy.time_array += 1
    uvcopy._update_phase_center_id(1, new_id=2)

    uvcomb = getattr(uvcopy, func)(uvcal_phase_center, **kwargs)
    assert uvcal_phase_center.phase_center_catalog == uvcomb.phase_center_catalog

    uvcopy.phase_center_catalog[2]["cat_name"] = "whoisthis"
    uvcomb = getattr(uvcopy, func)(uvcal_phase_center, **kwargs)
    uvcomb.reorder_times()

    assert all(idx in uvcomb.phase_center_catalog for idx in [1, 2])
    assert np.array_equal(uvcomb.phase_center_id_array, [1, 2] * uvcopy.Ntimes)


def test_ref_ant_array_add_err(uvcal_phase_center):
    uvcopy = uvcal_phase_center.copy()
    uvcopy.ref_antenna_array = np.ones(uvcopy.Ntimes, dtype=int)
    uvcopy.jones_array[0] = -6  # Need this to differentiate the data sets
    with pytest.raises(ValueError, match="To combine these data, both or neither "):
        _ = uvcopy + uvcal_phase_center

    # Test what happens if no phase center catalog period
    uvcal_phase_center.ref_antenna_array = np.zeros(uvcopy.Ntimes, dtype=int)
    with pytest.raises(ValueError, match="with different reference antennas."):
        _ = uvcopy + uvcal_phase_center


@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.parametrize(
    "func,kwargs", [["__add__", {}], ["fast_concat", {"axis": "time"}]]
)
def test_ref_ant_array_add(uvcal_phase_center, func, kwargs):
    uvcopy = uvcal_phase_center.copy()
    uvcopy.time_array += 1
    uvcopy.ref_antenna_array = np.ones(uvcopy.Ntimes, dtype=int)
    uvcal_phase_center.ref_antenna_array = np.zeros(uvcopy.Ntimes, dtype=int)

    uvcomb = getattr(uvcopy, func)(uvcal_phase_center, **kwargs)
    uvcomb.reorder_times()
    assert np.array_equal(uvcomb.ref_antenna_array, [0, 1] * uvcopy.Ntimes)


@pytest.mark.parametrize(
    "func,suffix", [["write_ms_cal", "ms"], ["write_calh5", "calh5"]]
)
def test_phase_center_write_roundtrip(uvcal_phase_center, func, suffix, tmp_path):
    if suffix == "ms":
        pytest.importorskip("casacore")

    filename = os.path.join(tmp_path, "pc_roundtrip." + suffix)
    getattr(uvcal_phase_center, func)(filename)

    uvc = UVCal()
    uvc.read(filename)
    uvc.history = uvcal_phase_center.history
    if suffix == "ms":
        # Handle some extra bits here for MS-type
        uvc.extra_keywords = uvcal_phase_center.extra_keywords
        uvc.scan_number_array = uvcal_phase_center.scan_number_array

    assert uvc == uvcal_phase_center


@pytest.mark.parametrize(
    "func,suffix", [["write_ms_cal", "ms"], ["write_calh5", "calh5"]]
)
def test_refant_array_write_roundtrip(uvcal_phase_center, func, suffix, tmp_path):
    if suffix == "ms":
        pytest.importorskip("casacore")
    uvcal_phase_center.ref_antenna_array = np.full(
        uvcal_phase_center.Ntimes,
        uvcal_phase_center.telescope.antenna_numbers[0],
        dtype=int,
    )
    uvcal_phase_center.ref_antenna_array[-1] = (
        uvcal_phase_center.telescope.antenna_numbers[1]
    )
    uvcal_phase_center.ref_antenna_name = "various"

    filename = os.path.join(tmp_path, "refantarr_roundtrip." + suffix)
    getattr(uvcal_phase_center, func)(filename)

    uvc = UVCal()
    uvc.read(filename)
    uvc.history = uvcal_phase_center.history
    if suffix == "ms":
        # Handle some extra bits here for MS-type
        uvc.extra_keywords = uvcal_phase_center.extra_keywords
        uvc.scan_number_array = uvcal_phase_center.scan_number_array

    assert uvc == uvcal_phase_center


@pytest.mark.filterwarnings("ignore:The calfits format does not support")
@pytest.mark.filterwarnings("ignore:key CASA_Version in extra_keyword")
@pytest.mark.parametrize(
    "func,suffix",
    [["write_ms_cal", "ms"], ["write_calh5", "calh5"], ["write_calfits", "fits"]],
)
def test_antdiam_write_roundtrip(uvcal_phase_center, func, suffix, tmp_path):
    if suffix == "ms":
        pytest.importorskip("casacore")

    filename = os.path.join(tmp_path, "pc_roundtrip." + suffix)
    uvcal_phase_center.telescope.antenna_diameters = np.full(
        uvcal_phase_center.telescope.Nants, 10.0
    )
    getattr(uvcal_phase_center, func)(filename)

    uvc = UVCal()
    uvc.read(filename)
    uvc.history = uvcal_phase_center.history
    if suffix == "ms":
        # Handle some extra bits here for MS-type
        uvc.extra_keywords = uvcal_phase_center.extra_keywords
        uvc.scan_number_array = uvcal_phase_center.scan_number_array
    elif suffix == "fits":
        # Drop phase center info
        uvc.Nphase = uvcal_phase_center.Nphase
        uvc.phase_center_catalog = uvcal_phase_center.phase_center_catalog
        uvc.phase_center_id_array = uvcal_phase_center.phase_center_id_array

    assert uvc == uvcal_phase_center


def test_phase_center_fast_concat(multi_spw_delay):
    multi_spw_delay.phase_center_catalog = {
        1: {
            "cat_name": "3c84",
            "cat_type": "sidereal",
            "cat_lon": 0.8718035968995141,
            "cat_lat": 0.7245157752262148,
            "cat_frame": "icrs",
            "cat_epoch": 2000.0,
            "cat_times": None,
            "cat_pm_ra": None,
            "cat_pm_dec": None,
            "cat_vrad": None,
            "cat_dist": None,
            "info_source": "file",
        }
    }
    multi_spw_delay.phase_center_id_array = np.ones(multi_spw_delay.Ntimes, dtype=int)
    uvc1 = multi_spw_delay.select(spws=1, inplace=False)
    uvc2 = multi_spw_delay.select(spws=2, inplace=False)
    uvc3 = multi_spw_delay.select(spws=3, inplace=False)
    uvc_comp = uvc1.fast_concat([uvc2, uvc3], axis="spw")

    uvc_comp.history = multi_spw_delay.history
    assert uvc_comp == multi_spw_delay


def test_fast_concat_phase_center_missing_err(uvcal_phase_center):
    uvcopy = uvcal_phase_center.copy()
    uvcopy.phase_center_catalog = None
    uvcopy.phase_center_id_array = None

    with pytest.raises(ValueError, match="To combine these data, phase_center_id_"):
        uvcal_phase_center.fast_concat(uvcopy, axis="time")
