# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvcal object.

"""
import copy
import itertools
import os
import re

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata import UVCal
from pyuvdata.data import DATA_PATH
from pyuvdata.uvcal.uvcal import _future_array_shapes_warning

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values",
    "ignore:antenna_positions are not set or are being overwritten. Using known values",
)


@pytest.fixture(scope="function")
def uvcal_data():
    """Set up some uvcal iter tests."""
    required_properties = [
        "Nfreqs",
        "Njones",
        "Ntimes",
        "Nspws",
        "Nants_data",
        "Nants_telescope",
        "wide_band",
        "antenna_names",
        "antenna_numbers",
        "ant_array",
        "telescope_name",
        "freq_array",
        "channel_width",
        "spw_array",
        "flex_spw",
        "jones_array",
        "time_array",
        "integration_time",
        "gain_convention",
        "flag_array",
        "cal_type",
        "cal_style",
        "x_orientation",
        "future_array_shapes",
        "history",
    ]
    required_parameters = ["_" + prop for prop in required_properties]

    extra_properties = [
        "telescope_location",
        "antenna_positions",
        "lst_array",
        "gain_array",
        "delay_array",
        "sky_field",
        "sky_catalog",
        "ref_antenna_name",
        "Nsources",
        "baseline_range",
        "diffuse_model",
        "input_flag_array",
        "time_range",
        "freq_range",
        "flex_spw_id_array",
        "observer",
        "git_origin_cal",
        "git_hash_cal",
        "quality_array",
        "total_quality_array",
        "extra_keywords",
        "gain_scale",
        "filename",
    ]
    extra_parameters = ["_" + prop for prop in extra_properties]

    other_properties = ["pyuvdata_version_str"]

    uv_cal_object = UVCal()

    # yields the data we need but will continue to the del call after tests
    yield (
        uv_cal_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    )

    # some post-test object cleanup
    del uv_cal_object
    return


@pytest.fixture
def multi_spw_gain(gain_data):
    gain_obj = gain_data.copy()
    gain_obj._set_flex_spw()
    gain_obj.channel_width = (
        np.zeros(gain_obj.Nfreqs, dtype=np.float64) + gain_obj.channel_width
    )
    gain_obj.Nspws = 2
    gain_obj.flex_spw_id_array = np.concatenate(
        (
            np.ones(gain_obj.Nfreqs // 2, dtype=int),
            np.full(gain_obj.Nfreqs // 2, 2, dtype=int),
        )
    )
    gain_obj.spw_array = np.array([1, 2])
    spw2_inds = np.nonzero(gain_obj.flex_spw_id_array == 2)[0]
    spw2_chan_width = gain_obj.channel_width[0] * 2
    gain_obj.freq_array[spw2_inds] = gain_obj.freq_array[
        spw2_inds[0]
    ] + spw2_chan_width * np.arange(spw2_inds.size)
    gain_obj.channel_width[spw2_inds] = spw2_chan_width
    gain_obj.check(check_freq_spacing=True)

    yield gain_obj

    del gain_obj


@pytest.fixture
def wideband_gain(gain_data):
    gain_obj = gain_data.copy()
    gain_obj._set_wide_band()

    gain_obj.spw_array = np.array([1, 2, 3])
    gain_obj.Nspws = 3
    gain_obj.gain_array = gain_obj.gain_array[:, 0:3, :, :]
    gain_obj.flag_array = gain_obj.flag_array[:, 0:3, :, :]
    gain_obj.quality_array = gain_obj.quality_array[:, 0:3, :, :]
    gain_obj.input_flag_array = np.zeros(
        gain_obj._input_flag_array.expected_shape(gain_obj)
    ).astype(np.bool_)

    gain_obj.freq_range = np.zeros((gain_obj.Nspws, 2), dtype=gain_obj.freq_array.dtype)
    gain_obj.freq_range[0, :] = gain_obj.freq_array[[0, 2]]
    gain_obj.freq_range[1, :] = gain_obj.freq_array[[2, 4]]
    gain_obj.freq_range[2, :] = gain_obj.freq_array[[4, 6]]

    gain_obj.channel_width = None
    gain_obj.freq_array = None
    gain_obj.flex_spw_id_array = None
    gain_obj.Nfreqs = 1

    with uvtest.check_warnings(
        DeprecationWarning,
        match="The input_flag_array is deprecated and will be removed in version 2.5",
    ):
        gain_obj.check(check_freq_spacing=True)

    yield gain_obj

    del gain_obj


@pytest.fixture
def multi_spw_delay(delay_data_inputflag):
    delay_obj = delay_data_inputflag.copy()
    delay_obj.Nspws = 3
    delay_obj.spw_array = np.array([1, 2, 3])

    # copy the delay array to the second SPW
    delay_obj.delay_array = np.repeat(delay_obj.delay_array, delay_obj.Nspws, axis=1)
    delay_obj.flag_array = np.repeat(delay_obj.flag_array, delay_obj.Nspws, axis=1)
    delay_obj.input_flag_array = np.repeat(
        delay_obj.input_flag_array, delay_obj.Nspws, axis=1
    )
    delay_obj.quality_array = np.repeat(
        delay_obj.quality_array, delay_obj.Nspws, axis=1
    )

    delay_obj.freq_range = np.repeat(delay_obj.freq_range, delay_obj.Nspws, axis=0)
    # Make the second & third SPWs be contiguous with a 10 MHz range
    delay_obj.freq_range[1, 0] = delay_obj.freq_range[0, 1]
    delay_obj.freq_range[1, 1] = delay_obj.freq_range[1, 0] + 10e6
    delay_obj.freq_range[2, 0] = delay_obj.freq_range[1, 1]
    delay_obj.freq_range[2, 1] = delay_obj.freq_range[1, 1] + 10e6

    with uvtest.check_warnings(
        DeprecationWarning,
        match="The input_flag_array is deprecated and will be removed in version 2.5",
    ):
        delay_obj.check()

    yield delay_obj

    del delay_obj


def extend_jones_axis(calobj, input_flag=True, total_quality=True):
    while calobj.Njones < 4:
        new_jones = np.min(calobj.jones_array) - 1
        calobj.jones_array = np.append(calobj.jones_array, new_jones)
        calobj.Njones += 1
        if not calobj.metadata_only:
            if calobj.future_array_shapes:
                calobj.flag_array = np.concatenate(
                    (calobj.flag_array, calobj.flag_array[:, :, :, [-1]]), axis=3
                )
                if calobj.cal_type == "gain":
                    calobj.gain_array = np.concatenate(
                        (calobj.gain_array, calobj.gain_array[:, :, :, [-1]]), axis=3
                    )
                else:
                    calobj.delay_array = np.concatenate(
                        (calobj.delay_array, calobj.delay_array[:, :, :, [-1]]), axis=3
                    )
                if calobj.input_flag_array is not None:
                    calobj.input_flag_array = np.concatenate(
                        (
                            calobj.input_flag_array,
                            calobj.input_flag_array[:, :, :, [-1]],
                        ),
                        axis=3,
                    )
                calobj.quality_array = np.concatenate(
                    (calobj.quality_array, calobj.quality_array[:, :, :, [-1]]), axis=3
                )
                if calobj.total_quality_array is not None:
                    calobj.total_quality_array = np.concatenate(
                        (
                            calobj.total_quality_array,
                            calobj.total_quality_array[:, :, [-1]],
                        ),
                        axis=2,
                    )
            else:
                calobj.flag_array = np.concatenate(
                    (calobj.flag_array, calobj.flag_array[:, :, :, :, [-1]]), axis=4
                )
                if calobj.cal_type == "gain":
                    calobj.gain_array = np.concatenate(
                        (calobj.gain_array, calobj.gain_array[:, :, :, :, [-1]]), axis=4
                    )
                else:
                    calobj.delay_array = np.concatenate(
                        (calobj.delay_array, calobj.delay_array[:, :, :, :, [-1]]),
                        axis=4,
                    )
                if calobj.input_flag_array is not None:
                    calobj.input_flag_array = np.concatenate(
                        (
                            calobj.input_flag_array,
                            calobj.input_flag_array[:, :, :, :, [-1]],
                        ),
                        axis=4,
                    )
                calobj.quality_array = np.concatenate(
                    (calobj.quality_array, calobj.quality_array[:, :, :, :, [-1]]),
                    axis=4,
                )
                if calobj.total_quality_array is not None:
                    calobj.total_quality_array = np.concatenate(
                        (
                            calobj.total_quality_array,
                            calobj.total_quality_array[:, :, :, [-1]],
                        ),
                        axis=3,
                    )
    if not calobj.metadata_only:
        if calobj.input_flag_array is None and input_flag:
            calobj.input_flag_array = calobj.flag_array
        if calobj.total_quality_array is None and total_quality:
            calobj.total_quality_array = np.ones(
                calobj._total_quality_array.expected_shape(calobj)
            )


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
    attributes = [i for i in uv_cal_object.__dict__.keys() if i[0] == "_"]
    for a in attributes:
        assert a in expected_parameters, "unexpected parameter " + a + " found in UVCal"


def test_unexpected_attributes(uvcal_data):
    """Test for extra attributes."""
    (uv_cal_object, _, required_properties, _, extra_properties, other_properties) = (
        uvcal_data
    )
    expected_attributes = required_properties + extra_properties + other_properties
    attributes = [i for i in uv_cal_object.__dict__.keys() if i[0] != "_"]
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
            print("setting {prop_name} to a random number failed".format(prop_name=k))
            raise


def test_equality(gain_data):
    """Basic equality test"""
    assert gain_data == gain_data


def test_check(gain_data):
    """Test that parameter checks run properly"""
    assert gain_data.check()


def test_check_flag_array(gain_data):
    gain_data.flag_array = np.ones((gain_data.flag_array.shape), dtype=int)

    with pytest.raises(
        ValueError, match="UVParameter _flag_array is not the appropriate type."
    ):
        gain_data.check()


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_future_array_shape(caltype, gain_data, delay_data_inputflag):
    """Convert to future shapes and check. Convert back and test for equality."""

    if caltype == "gain":
        calobj = gain_data
        calobj.total_quality_array = np.ones(
            (calobj._total_quality_array.expected_shape(calobj))
        )
        calobj.input_flag_array = np.zeros(
            calobj._input_flag_array.expected_shape(calobj)
        ).astype(np.bool_)
    else:
        calobj = delay_data_inputflag
        calobj.total_quality_array = np.ones(
            (calobj._total_quality_array.expected_shape(calobj))
        )
        calobj.input_flag_array = np.zeros(
            calobj._input_flag_array.expected_shape(calobj)
        ).astype(np.bool_)

    calobj2 = calobj.copy()
    # test the no-op
    calobj2.use_future_array_shapes()
    assert calobj == calobj2

    with uvtest.check_warnings(
        DeprecationWarning,
        match="This method will be removed in version 3.0 when the current array "
        "shapes are no longer supported.",
    ):
        calobj.use_current_array_shapes()
    with uvtest.check_warnings(
        DeprecationWarning,
        match="The input_flag_array is deprecated and will be removed in version 2.5",
    ):
        calobj.check()

    calobj3 = calobj.copy()
    # test the no-op
    calobj.use_current_array_shapes()
    assert calobj3 == calobj

    warn_type = [DeprecationWarning]
    warn_msg = ["The input_flag_array is deprecated and will be removed in version 2.5"]

    if caltype == "delay":
        warn_type += [UserWarning]
        warn_msg += [
            "When converting a delay-style cal to future array shapes the "
            "flag_array (and input_flag_array if it exists) must drop the "
            "frequency axis so that it will be the same shape as the delay_array."
        ]
    with uvtest.check_warnings(warn_type, match=warn_msg):
        calobj.use_future_array_shapes()
        calobj.check()

    assert calobj == calobj2


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_future_array_shape_errors(
    caltype,
    gain_data,
    delay_data_inputflag,
    multi_spw_gain,
    multi_spw_delay,
    wideband_gain,
):
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()

        calobj_multi_spw = multi_spw_gain

        calobj_wideband = wideband_gain
        calobj_wideband.select(spws=1)
        with pytest.raises(
            ValueError,
            match="Cannot use current array shapes if cal_style is not 'delay' and "
            "wide_band is True.",
        ):
            with uvtest.check_warnings(
                DeprecationWarning,
                "This method will be removed in version 3.0 when the current array "
                "shapes are no longer supported.",
            ):
                calobj_wideband.use_current_array_shapes()

    else:
        calobj = delay_data_inputflag

        calobj_multi_spw = multi_spw_delay

        with pytest.raises(
            ValueError, match="Cannot use current array shapes if Nspws > 1."
        ):
            with uvtest.check_warnings(
                DeprecationWarning,
                match="This method will be removed in version 3.0 when the current "
                "array shapes are no longer supported.",
            ):
                calobj_multi_spw.use_current_array_shapes()

    calobj.integration_time[-1] = calobj.integration_time[0] * 2.0
    if caltype == "delay":
        calobj.Nfreqs = 2
        with uvtest.check_warnings(
            DeprecationWarning,
            match=[
                "Nfreqs will be required to be 1 for wide_band cals",
                "The input_flag_array is deprecated and will be removed in version 2.5",
            ],
        ):
            calobj.check()
        calobj.Nfreqs = 1
    else:
        calobj.check()

    with pytest.raises(
        ValueError, match="integration_time parameter contains multiple unique values"
    ):
        with uvtest.check_warnings(
            UserWarning,
            match="When converting a delay-style cal to future array shapes",
        ):
            calobj.use_current_array_shapes()

    with pytest.raises(
        ValueError, match="The integration times are variable. The calfits format"
    ):
        with uvtest.check_warnings(
            DeprecationWarning,
            match="Nfreqs will be required to be 1 for wide_band cals",
        ):
            calobj.write_calfits("foo")

    if caltype == "gain":
        calobj2.channel_width[-1] = calobj2.channel_width[0] * 2.0
        calobj2.check()

        with pytest.raises(
            ValueError, match="channel_width parameter contains multiple unique values"
        ):
            with uvtest.check_warnings(
                UserWarning,
                match="When converting a delay-style cal to future array shapes",
            ):
                calobj2.use_current_array_shapes()

        with pytest.raises(ValueError, match="The frequencies are not evenly spaced"):
            calobj2._check_freq_spacing()


def test_unknown_telescopes(gain_data, tmp_path):
    calobj = gain_data

    write_file = str(tmp_path / "test.calfits")
    write_file2 = str(tmp_path / "test2.calfits")
    calobj.write_calfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
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

    with pytest.raises(
        ValueError, match="Required UVParameter _antenna_positions has not been set."
    ):
        with uvtest.check_warnings(
            [UserWarning], match=["Telescope foo is not in known_telescopes"]
        ):
            UVCal.from_file(write_file2, use_future_array_shapes=True)
    with uvtest.check_warnings(
        [UserWarning], match=["Telescope foo is not in known_telescopes"]
    ):
        UVCal.from_file(write_file2, use_future_array_shapes=True, run_check=False)


def test_nants_data_telescope_larger(gain_data):
    # make sure it's okay for Nants_telescope to be strictly greater than Nants_data
    gain_data.Nants_telescope += 1
    # add dummy information for "new antenna" to pass object check
    gain_data.antenna_names = np.concatenate((gain_data.antenna_names, ["dummy_ant"]))
    gain_data.antenna_numbers = np.concatenate((gain_data.antenna_numbers, [20]))
    gain_data.antenna_positions = np.concatenate(
        (gain_data.antenna_positions, np.zeros((1, 3), dtype=float))
    )
    assert gain_data.check()


def test_ant_array_not_in_antnums(gain_data):
    # make sure an error is raised if antennas with data not in antenna_numbers
    # remove antennas from antenna_names & antenna_numbers by hand
    gain_data.antenna_names = gain_data.antenna_names[1:]
    gain_data.antenna_numbers = gain_data.antenna_numbers[1:]
    gain_data.antenna_positions = gain_data.antenna_positions[1:, :]
    gain_data.Nants_telescope = gain_data.antenna_numbers.size
    with pytest.raises(ValueError) as cm:
        gain_data.check()
    assert str(cm.value).startswith(
        "All antennas in ant_array must be in antenna_numbers"
    )


def test_set_gain(gain_data, delay_data):
    delay_data._set_gain()
    assert delay_data._gain_array.required
    assert not delay_data._delay_array.required
    assert delay_data._gain_array.form == delay_data._flag_array.form
    assert delay_data._gain_array.form == delay_data._quality_array.form


def test_set_delay(gain_data):
    gain_data._set_delay()
    assert gain_data._delay_array.required
    assert not gain_data._gain_array.required
    assert gain_data._gain_array.form == gain_data._flag_array.form
    assert gain_data._delay_array.form == gain_data._quality_array.form


def test_set_unknown(gain_data):
    with uvtest.check_warnings(
        DeprecationWarning,
        match="Setting the cal_type to 'unknown' is deprecated. This will become an "
        "error in version 2.5",
    ):
        gain_data._set_unknown_cal_type()

    with uvtest.check_warnings(
        DeprecationWarning,
        match="The 'unknown' cal_type is deprecated and will be removed in version "
        "2.5",
    ):
        gain_data.check()

    assert not gain_data._delay_array.required
    assert not gain_data._gain_array.required
    assert gain_data._gain_array.form == gain_data._flag_array.form
    assert gain_data._gain_array.form == gain_data._quality_array.form


def test_set_sky(gain_data):
    gain_data._set_sky()
    assert gain_data._sky_catalog.required
    assert gain_data._ref_antenna_name.required


def test_set_redundant(gain_data):
    gain_data._set_redundant()
    assert not gain_data._sky_catalog.required
    assert not gain_data._ref_antenna_name.required


def test_convert_filetype(gain_data):
    # error testing
    with pytest.raises(ValueError, match="filetype must be calfits."):
        gain_data._convert_to_filetype("uvfits")


def test_error_metadata_only_write(gain_data, tmp_path):
    calobj = gain_data.copy(metadata_only=True)

    out_file = os.path.join(tmp_path, "outtest.calfits")
    with pytest.raises(ValueError, match="Cannot write out metadata only objects to a"):
        calobj.write_calfits(out_file)


def test_flexible_spw(gain_data):
    calobj = gain_data

    # check that this check passes on non-flex_spw objects
    calobj._check_flex_spw_contiguous()

    # check warning if flex_spw_id_array is not set
    calobj.flex_spw_id_array = None
    with uvtest.check_warnings(
        DeprecationWarning,
        match="flex_spw_id_array is not set. It will be required starting in version "
        "3.0 for non-wide-band objects",
    ):
        calobj.check()

    # first just make one spw and check that object still passes check
    calobj._set_flex_spw()
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


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("convention", ["minus", "plus"])
@pytest.mark.parametrize("same_freqs", [True, False])
def test_convert_to_gain(future_shapes, convention, same_freqs, delay_data_inputflag):
    delay_obj = delay_data_inputflag
    delay_obj.Nfreqs = 30
    delay_obj.freq_array = np.arange(delay_obj.Nfreqs) * 1e6 + 1e8
    delay_obj.channel_width = np.full(delay_obj.Nfreqs, 1e6)
    with uvtest.check_warnings(
        DeprecationWarning,
        match=[
            "Nfreqs will be required to be 1 for wide_band cals (including all "
            "delay cals) starting in version 3.0",
            "The input_flag_array is deprecated and will be removed in version 2.5",
            "The freq_array attribute should not be set if wide_band=True. This will "
            "become an error in version 3.0.",
            "The channel_width attribute should not be set if wide_band=True. This "
            "will become an error in version 3.0.",
        ],
    ):
        delay_obj.check()
    freq_array = copy.deepcopy(delay_obj.freq_array)
    channel_width = delay_obj.channel_width
    if not same_freqs:
        # try with different number and same number but different values
        if convention == "minus":
            freq_array = freq_array[0 : (delay_obj.Nfreqs // 2)]
            channel_width = channel_width[0 : (delay_obj.Nfreqs // 2)]
        else:
            freq_array[2] = freq_array[2] + 1e6

    # test passing a 1 element array for channel width
    if not future_shapes:
        delay_obj.use_current_array_shapes()
        if convention == "minus":
            channel_width = np.asarray([delay_obj.channel_width])
        else:
            channel_width = delay_obj.channel_width
    else:
        assert channel_width.size == freq_array.size

    new_gain_obj = delay_obj.copy()
    tqa_size = new_gain_obj.delay_array.shape[1:]
    new_gain_obj.total_quality_array = np.ones(tqa_size)

    new_gain_obj2 = new_gain_obj.copy()

    if not future_shapes and not same_freqs:
        with uvtest.check_warnings(
            [UserWarning, DeprecationWarning],
            match=[
                "Existing flag array has a frequency axis of length > 1 but "
                "frequencies do not match freq_array. The existing flag array "
                "(and input_flag_array if it exists) will be collapsed using "
                "the `pyuvdata.utils.and_collapse` function which will only "
                "flag an antpol-time if all of the frequecies are flagged for "
                "that antpol-time. Then it will be broadcast to all the new "
                "frequencies. To preserve the original flag information, "
                "create a UVFlag object from this cal object before this "
                "operation.",
                "The input_flag_array is deprecated and will be removed in version 2.5",
            ],
        ):
            new_gain_obj.convert_to_gain(
                freq_array=freq_array,
                channel_width=channel_width,
                delay_convention=convention,
            )
    else:
        new_gain_obj.convert_to_gain(
            freq_array=freq_array,
            channel_width=channel_width,
            delay_convention=convention,
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
    if future_shapes:
        assert np.allclose(
            np.angle(new_gain_obj.gain_array[:, 10, :, :]) % (2 * np.pi),
            (conv * 2 * np.pi * delay_obj.delay_array[:, 0, :, :] * freq_array[10])
            % (2 * np.pi),
            rtol=new_gain_obj._gain_array.tols[0],
            atol=new_gain_obj._gain_array.tols[1],
        )
        assert np.allclose(
            delay_obj.quality_array,
            new_gain_obj.quality_array[:, 10, :, :],
            rtol=new_gain_obj._quality_array.tols[0],
            atol=new_gain_obj._quality_array.tols[1],
        )

    else:
        assert np.allclose(
            np.angle(new_gain_obj.gain_array[:, :, 10, :, :]) % (2 * np.pi),
            (conv * 2 * np.pi * delay_obj.delay_array[:, :, 0, :, :] * freq_array[10])
            % (2 * np.pi),
            rtol=new_gain_obj._gain_array.tols[0],
            atol=new_gain_obj._gain_array.tols[1],
        )
        assert np.allclose(
            delay_obj.quality_array,
            new_gain_obj.quality_array[:, :, 10, :, :],
            rtol=new_gain_obj._quality_array.tols[0],
            atol=new_gain_obj._quality_array.tols[1],
        )

    assert new_gain_obj.history == (
        delay_obj.history + "  Converted from delays to gains using pyuvdata."
    )

    if same_freqs:
        with uvtest.check_warnings(
            DeprecationWarning,
            match=[
                "In version 3.0 and later freq_array and channel_width will be",
                "The input_flag_array is deprecated and will be removed in version 2.5",
            ],
        ):
            new_gain_obj2.convert_to_gain(delay_convention=convention)

        assert new_gain_obj == new_gain_obj2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
def test_convert_to_gain_errors(gain_data, delay_data_inputflag, multi_spw_delay):
    delay_obj = delay_data_inputflag
    gain_obj = gain_data

    delay_obj.Nfreqs = 30
    delay_obj.freq_array = np.arange(delay_obj.Nfreqs) * 1e6 + 1e8
    delay_obj.channel_width = np.full(delay_obj.Nfreqs, 1e6)
    delay_obj_current = delay_obj.copy()
    delay_obj_current.use_current_array_shapes()

    with pytest.raises(
        ValueError, match="freq_array contains values outside the freq_range."
    ):
        delay_obj_current.convert_to_gain(
            freq_array=np.asarray([50e6, 60e6]),
            channel_width=delay_obj_current.channel_width,
        )

    with pytest.raises(
        ValueError, match="freq_array parameter must be a one dimensional array"
    ):
        delay_obj_current.convert_to_gain(
            freq_array=delay_obj_current.freq_array,
            channel_width=delay_obj_current.channel_width,
        )

    with pytest.raises(
        ValueError,
        match="This object is using the current array shapes, so the "
        "channel_width parameter must be a scalar value",
    ):
        delay_obj_current.convert_to_gain(
            freq_array=delay_obj_current.freq_array[0, :],
            channel_width=(
                np.zeros(delay_obj_current.Nfreqs, dtype=float)
                + delay_obj_current.channel_width
            ),
        )

    with pytest.raises(
        ValueError,
        match="convert_to_gain currently does not support multiple spectral windows",
    ):
        multi_spw_delay.convert_to_gain()

    with pytest.raises(
        ValueError, match="delay_convention can only be 'minus' or 'plus'"
    ):
        delay_obj_current.convert_to_gain(delay_convention="bogus")

    with pytest.raises(ValueError, match="The data is already a gain cal_type."):
        gain_obj.convert_to_gain()

    with uvtest.check_warnings(
        DeprecationWarning,
        match="Setting the cal_type to 'unknown' is deprecated. This will become an "
        "error in version 2.5",
    ):
        gain_obj._set_unknown_cal_type()
    with pytest.raises(ValueError, match="cal_type is unknown, cannot convert to gain"):
        gain_obj.convert_to_gain()

    delay_obj_current.freq_array = None
    delay_obj_current.channel_width = None
    with pytest.raises(
        ValueError, match="freq_array and channel_width must be provided"
    ):
        delay_obj_current.convert_to_gain()

    with pytest.raises(
        ValueError,
        match="This object is using the future array shapes, so the "
        "channel_width parameter be an array shaped like the freq_array",
    ):
        delay_obj.convert_to_gain(
            freq_array=delay_obj.freq_array, channel_width=delay_obj.channel_width[0]
        )


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:When converting a delay-style cal to future array")
@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_antennas(
    caltype, future_shapes, gain_data, delay_data_inputflag, tmp_path
):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()

    old_history = calobj.history
    ants_to_keep = np.array([65, 96, 9, 97, 89, 22, 20, 72])
    calobj2.select(antenna_nums=ants_to_keep)

    assert len(ants_to_keep) == calobj2.Nants_data
    for ant in ants_to_keep:
        assert ant in calobj2.ant_array
    for ant in calobj2.ant_array:
        assert ant in ants_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific antennas using pyuvdata.",
        calobj2.history,
    )

    # now test using antenna_names to specify antennas to keep
    ants_to_keep = np.array(sorted(ants_to_keep))
    ant_names = []
    for a in ants_to_keep:
        ind = np.where(calobj.antenna_numbers == a)[0][0]
        ant_names.append(calobj.antenna_names[ind])

    calobj3 = calobj.select(antenna_names=ant_names, inplace=False)

    assert calobj2 == calobj3

    # check for errors associated with antennas not included in data, bad names
    # or providing numbers and names
    with pytest.raises(
        ValueError,
        match=f"Antenna number {np.max(calobj.ant_array) + 1} "
        "is not present in the array",
    ):
        calobj.select(antenna_nums=np.max(calobj.ant_array) + np.arange(1, 3))

    with pytest.raises(
        ValueError, match="Antenna name test1 is not present in the antenna_names array"
    ):
        calobj.select(antenna_names=["test1"])
    with pytest.raises(
        ValueError, match="Only one of antenna_nums and antenna_names can be provided."
    ):
        calobj.select(antenna_nums=ants_to_keep, antenna_names=ant_names)

    # check that write_calfits works with Nants_data < Nants_telescope
    write_file_calfits = str(tmp_path / "select_test.calfits")
    calobj2.write_calfits(write_file_calfits, clobber=True)

    # check that reading it back in works too
    new_calobj = UVCal.from_file(
        write_file_calfits, use_future_array_shapes=future_shapes
    )
    assert calobj2 == new_calobj

    # check that total_quality_array is handled properly when present
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    warn_type = [UserWarning]
    msg = ["Cannot preserve total_quality_array"]
    if caltype == "delay":
        warn_type += [DeprecationWarning]
        msg += ["The input_flag_array is deprecated and will be removed in version 2.5"]
    with uvtest.check_warnings(warn_type, match=msg):
        calobj.select(antenna_names=ant_names, inplace=True)
    assert calobj.total_quality_array is None


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_times(
    future_shapes, caltype, gain_data, delay_data_inputflag, tmp_path
):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()

    old_history = calobj.history
    times_to_keep = calobj.time_array[2:5]

    calobj2.select(times=times_to_keep)

    assert len(times_to_keep) == calobj2.Ntimes
    for t in times_to_keep:
        assert t in calobj2.time_array
    for t in np.unique(calobj2.time_array):
        assert t in times_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific times using pyuvdata.",
        calobj2.history,
    )

    write_file_calfits = str(tmp_path / "select_test.calfits")
    # test writing calfits with only one time
    calobj2 = calobj.copy()
    times_to_keep = calobj.time_array[[1]]
    calobj2.select(times=times_to_keep)
    calobj2.write_calfits(write_file_calfits, clobber=True)

    # check for errors associated with times not included in data
    pytest.raises(
        ValueError,
        calobj.select,
        times=[np.min(calobj.time_array) - calobj.integration_time],
    )

    # check for warnings and errors associated with unevenly spaced times
    calobj2 = calobj.copy()
    warn_type = [UserWarning]
    msg = ["Selected times are not evenly spaced"]
    if caltype == "delay":
        warn_type += [DeprecationWarning]
        msg += ["The input_flag_array is deprecated and will be removed in version 2.5"]
    with uvtest.check_warnings(warn_type, match=msg):
        calobj2.select(times=calobj2.time_array[[0, 2, 3]])
    pytest.raises(ValueError, calobj2.write_calfits, write_file_calfits)


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:When converting a delay-style cal to future array")
@pytest.mark.filterwarnings("ignore:Nfreqs will be required to be 1 for wide_band")
@pytest.mark.filterwarnings("ignore:The freq_array attribute should not be set if")
@pytest.mark.filterwarnings("ignore:The channel_width attribute should not be set if")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_frequencies(
    future_shapes, caltype, gain_data, delay_data_inputflag, tmp_path
):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag
        calobj.freq_array = gain_data.freq_array
        calobj.channel_width = gain_data.channel_width
        calobj.Nfreqs = gain_data.Nfreqs

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()

    old_history = calobj.history

    if future_shapes:
        freqs_to_keep = calobj.freq_array[np.arange(4, 8)]
    else:
        freqs_to_keep = calobj.freq_array[0, np.arange(4, 8)]

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

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        calobj2.history,
    )

    write_file_calfits = str(tmp_path / "select_test.calfits")
    # test writing calfits with only one frequency
    calobj2 = calobj.copy()
    if future_shapes:
        freqs_to_keep = calobj.freq_array[5]
    else:
        freqs_to_keep = calobj.freq_array[0, 5]
    calobj2.select(frequencies=freqs_to_keep)
    calobj2.write_calfits(write_file_calfits, clobber=True)

    # test writing calfits with frequencies spaced by more than the channel width
    calobj2 = calobj.copy()
    if future_shapes:
        freqs_to_keep = calobj.freq_array[[0, 2, 4, 6, 8]]
    else:
        freqs_to_keep = calobj.freq_array[0, [0, 2, 4, 6, 8]]
    warn_type = [UserWarning]
    msg = ["Selected frequencies are not contiguous."]
    extra_warn_type = []
    extra_msg = []
    if caltype == "delay":
        extra_warn_type += [DeprecationWarning]
        extra_msg += [
            "The input_flag_array is deprecated and will be removed in version 2.5"
        ]
        if future_shapes:
            extra_warn_type += [DeprecationWarning] * 3
            extra_msg += [
                "Nfreqs will be required to be 1 for wide_band cals",
                "The freq_array attribute should not be set if wide_band=True",
                "The channel_width attribute should not be set if wide_band=True",
            ]
    with uvtest.check_warnings(warn_type + extra_warn_type, match=msg + extra_msg):
        calobj2.select(frequencies=freqs_to_keep)
    calobj2.write_calfits(write_file_calfits, clobber=True)

    # check for errors associated with frequencies not included in data
    pytest.raises(
        ValueError,
        calobj.select,
        frequencies=[np.max(calobj.freq_array) + calobj.channel_width],
    )

    # check for warnings and errors associated with unevenly spaced frequencies
    calobj2 = calobj.copy()
    if future_shapes:
        freqs_to_keep = calobj2.freq_array[[0, 5, 6]]
    else:
        freqs_to_keep = calobj2.freq_array[0, [0, 5, 6]]
    warn_type = [UserWarning]
    msg = ["Selected frequencies are not evenly spaced."]
    with uvtest.check_warnings(warn_type + extra_warn_type, match=msg + extra_msg):
        calobj2.select(frequencies=freqs_to_keep)

    with pytest.raises(
        ValueError,
        match="Frequencies are not evenly spaced or have differing values of channel",
    ):
        calobj2.write_calfits(write_file_calfits)


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:The freq_range attribute should not be set if")
@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_frequencies_multispw(future_shapes, multi_spw_gain, tmp_path):
    calobj = multi_spw_gain

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()
    old_history = calobj.history

    if future_shapes:
        freqs_to_keep = calobj.freq_array[np.arange(4, 8)]
    else:
        freqs_to_keep = calobj.freq_array[0, np.arange(4, 8)]

    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    calobj2.total_quality_array = np.zeros(
        calobj2._total_quality_array.expected_shape(calobj2)
    )

    # add dummy input_flag_array
    calobj.input_flag_array = np.zeros(
        calobj._input_flag_array.expected_shape(calobj)
    ).astype(np.bool_)
    calobj2.input_flag_array = np.zeros(
        calobj2._input_flag_array.expected_shape(calobj2)
    ).astype(np.bool_)

    # add freq_range
    if future_shapes:
        calobj2.freq_range = np.zeros(
            (calobj2.Nspws, 2), dtype=calobj2.freq_array.dtype
        )
        for index, spw in enumerate(calobj2.spw_array):
            spw_inds = np.nonzero(calobj2.flex_spw_id_array == spw)[0]
            calobj2.freq_range[index, 0] = np.min(
                np.squeeze(calobj2.freq_array)[spw_inds]
            )
            calobj2.freq_range[index, 1] = np.max(
                np.squeeze(calobj2.freq_array)[spw_inds]
            )
        with uvtest.check_warnings(
            DeprecationWarning,
            match=[
                "The freq_range attribute should not be set if cal_type='gain' and "
                "wide_band=False. This will become an error in version 3.0.",
                "The input_flag_array is deprecated and will be removed in version 2.5",
            ],
        ):
            calobj2.check()

    calobj2.select(frequencies=freqs_to_keep)

    assert len(freqs_to_keep) == calobj2.Nfreqs
    for f in freqs_to_keep:
        assert f in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
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

    if future_shapes:
        freqs_to_keep = calobj.freq_array[np.arange(5)]
    else:
        freqs_to_keep = calobj.freq_array[0, np.arange(5)]

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
    assert calobj3 == calobj2
    with uvtest.check_warnings(
        [UserWarning, DeprecationWarning],
        match=[
            "Cannot select on spws if Nspws=1.",
            "The input_flag_array is deprecated and will be removed in version 2.5",
        ],
    ):
        calobj3.select(spws=1)
    assert calobj3 == calobj2

    calobj3 = UVCal.from_file(write_file_calfits, use_future_array_shapes=future_shapes)

    calobj2.flex_spw = False
    calobj2._flex_spw_id_array.required = False
    calobj2.flex_spw_id_array = np.zeros(calobj2.Nfreqs, dtype=int)
    calobj2.spw_array = np.array([0])
    if not future_shapes:
        calobj2._channel_width.form = ()
        calobj2.channel_width = calobj2.channel_width[0]
    calobj2.check()

    assert calobj3 == calobj2


@pytest.mark.filterwarnings("ignore:The freq_array attribute should not be set if")
@pytest.mark.filterwarnings("ignore:The channel_width attribute should not be set if")
@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:When converting a delay-style cal to future array")
@pytest.mark.filterwarnings("ignore:Nfreqs will be required to be 1 for wide_band cals")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_freq_chans(caltype, future_shapes, gain_data, delay_data_inputflag):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    old_history = calobj.history
    chans_to_keep = np.arange(4, 8)

    if caltype == "delay":
        with pytest.raises(
            ValueError,
            match="Cannot select on frequencies because this is a wide_band object "
            "with no freq_array.",
        ):
            calobj.select(freq_chans=chans_to_keep)

    if caltype == "delay":
        calobj.freq_array = gain_data.freq_array
        calobj.channel_width = gain_data.channel_width
        calobj.Nfreqs = gain_data.Nfreqs

    if not future_shapes:
        calobj.use_current_array_shapes()

    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )

    calobj2 = calobj.copy()

    calobj2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == calobj2.Nfreqs
    if future_shapes:
        obj_freqs = calobj.freq_array
    else:
        obj_freqs = calobj.freq_array[0, :]
    for chan in chans_to_keep:
        assert obj_freqs[chan] in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in obj_freqs

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        calobj2.history,
    )

    # Test selecting both channels and frequencies
    if future_shapes:
        obj_freqs = calobj.freq_array
    else:
        obj_freqs = calobj.freq_array[0, :]
    freqs_to_keep = obj_freqs[np.arange(7, 10)]  # Overlaps with chans
    all_chans_to_keep = np.arange(4, 10)

    calobj2 = calobj.copy()
    calobj2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == calobj2.Nfreqs
    for chan in all_chans_to_keep:
        assert obj_freqs[chan] in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in obj_freqs[all_chans_to_keep]


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_spws_wideband(caltype, multi_spw_delay, wideband_gain, tmp_path):
    if caltype == "gain":
        calobj = wideband_gain
    else:
        calobj = multi_spw_delay

    calobj2 = calobj.copy()

    old_history = calobj.history

    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    calobj2.total_quality_array = np.zeros(
        calobj2._total_quality_array.expected_shape(calobj2)
    )

    spws_to_keep = [2, 3]
    calobj2.select(spws=spws_to_keep)

    assert len(spws_to_keep) == calobj2.Nspws
    for spw in spws_to_keep:
        assert spw in calobj2.spw_array
    for spw in calobj2.spw_array:
        assert spw in spws_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific spectral windows using pyuvdata.",
        calobj2.history,
    )

    # check for errors associated with spws not included in data
    with pytest.raises(
        ValueError, match="SPW number 5 is not present in the spw_array"
    ):
        calobj.select(spws=[5])


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize(
    "jones_to_keep", ([-5, -6], ["xx", "yy"], ["nn", "ee"], [[-5, -6]])
)
def test_select_polarizations(
    future_shapes, caltype, jones_to_keep, gain_data, delay_data_inputflag, tmp_path
):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()

    # add more jones terms to allow for better testing of selections
    extend_jones_axis(calobj)

    assert calobj.check()
    calobj2 = calobj.copy()

    old_history = calobj.history

    calobj2.select(jones=jones_to_keep)

    if isinstance(jones_to_keep[0], list):
        jones_to_keep = jones_to_keep[0]
    assert len(jones_to_keep) == calobj2.Njones
    for j in jones_to_keep:
        if isinstance(j, int):
            assert j in calobj2.jones_array
        else:
            assert (
                uvutils.jstr2num(j, x_orientation=calobj2.x_orientation)
                in calobj2.jones_array
            )
    for j in np.unique(calobj2.jones_array):
        if isinstance(jones_to_keep[0], int):
            assert j in jones_to_keep
        else:
            assert j in uvutils.jstr2num(
                jones_to_keep, x_orientation=calobj2.x_orientation
            )

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific jones polarization terms "
        "using pyuvdata.",
        calobj2.history,
    )

    # check for errors associated with polarizations not included in data
    pytest.raises(ValueError, calobj2.select, jones=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    warn_type = [UserWarning, DeprecationWarning]
    msg = [
        "Selected jones polarization terms are not evenly spaced",
        "The input_flag_array is deprecated and will be removed in version 2.5",
    ]
    with uvtest.check_warnings(warn_type, match=msg):
        calobj.select(jones=calobj.jones_array[[0, 1, 3]])
    write_file_calfits = os.path.join(tmp_path, "select_test.calfits")
    pytest.raises(ValueError, calobj.write_calfits, write_file_calfits)


@pytest.mark.filterwarnings("ignore:The freq_array attribute should not be set if")
@pytest.mark.filterwarnings("ignore:The channel_width attribute should not be set if")
@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:Nfreqs will be required to be 1 for wide_band")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select(future_shapes, caltype, gain_data, delay_data_inputflag):
    # now test selecting along all axes at once
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag
        calobj.freq_array = gain_data.freq_array
        calobj.channel_width = gain_data.channel_width
        calobj.Nfreqs = gain_data.Nfreqs

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()

    old_history = calobj.history

    ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
    if future_shapes:
        freqs_to_keep = calobj.freq_array[np.arange(2, 5)]
    else:
        freqs_to_keep = calobj.freq_array[0, np.arange(2, 5)]
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

    assert len(freqs_to_keep) == calobj2.Nfreqs
    for f in freqs_to_keep:
        assert f in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in freqs_to_keep

    assert len(jones_to_keep) == calobj2.Njones
    for j in jones_to_keep:
        assert j in calobj2.jones_array
    for j in np.unique(calobj2.jones_array):
        assert j in jones_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific antennas, times, "
        "frequencies, jones polarization terms "
        "using pyuvdata.",
        calobj2.history,
    )


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
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

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific antennas, times, "
        "spectral windows, jones polarization terms "
        "using pyuvdata.",
        calobj2.history,
    )


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_antennas(future_shapes, caltype, gain_data, method, delay_data_inputflag):
    """Test adding antennas between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()
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
    assert uvutils._check_histories(
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
    if caltype == "delay":
        warn_type += [DeprecationWarning] * 3
        msg += [
            "The input_flag_array is deprecated and will be removed in version 2.5"
        ] * 3
    with uvtest.check_warnings(warn_type, match=msg):
        getattr(calobj, method)(calobj2, **kwargs)
    assert calobj.total_quality_array is None

    if caltype == "delay":
        # test for when input_flag_array & quality array is present in first file but
        # not in second
        calobj.select(antenna_nums=ants1)
        ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        qa = np.ones(calobj._quality_array.expected_shape(calobj))
        qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2))

        tot_ifa = np.concatenate([ifa, ifa2], axis=0)
        tot_qa = np.concatenate([qa, qa2], axis=0)
        calobj.input_flag_array = ifa
        calobj2.input_flag_array = None
        calobj.quality_array = qa
        calobj2.quality_array = None
        getattr(calobj, method)(calobj2, **kwargs)
        assert np.allclose(calobj.input_flag_array, tot_ifa)
        assert np.allclose(calobj.quality_array, tot_qa)

        # test for when input_flag_array is present in second file but not first
        calobj.select(antenna_nums=ants1)
        ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        qa = np.zeros(calobj._quality_array.expected_shape(calobj))
        qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2))
        tot_ifa = np.concatenate([ifa, ifa2], axis=0)
        tot_qa = np.concatenate([qa, qa2], axis=0)
        calobj.input_flag_array = None
        calobj2.input_flag_array = ifa2
        calobj.quality_array = None
        calobj2.quality_array = qa2
        getattr(calobj, method)(calobj2, **kwargs)
        assert np.allclose(calobj.input_flag_array, tot_ifa)
        assert np.allclose(calobj.quality_array, tot_qa)

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


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_reorder_ants(
    future_shapes, caltype, metadata_only, gain_data, delay_data_inputflag
):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

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

    sorted_names = np.sort(calobj.antenna_names)
    calobj.reorder_antennas("name")
    temp = np.asarray(calobj.antenna_names)
    dtype_use = temp.dtype
    name_array = np.zeros_like(calobj.ant_array, dtype=dtype_use)
    for ind, ant in enumerate(calobj.ant_array):
        name_array[ind] = calobj.antenna_names[
            np.nonzero(calobj.antenna_numbers == ant)[0][0]
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
        gain_data.reorder_antennas(order="foo")

    with pytest.raises(
        ValueError,
        match="If order is an index array, it must contain all indicies for the"
        "ant_array, without duplicates.",
    ):
        gain_data.reorder_antennas(order=gain_data.antenna_numbers.astype(float))

    with pytest.raises(
        ValueError,
        match="If order is an index array, it must contain all indicies for the"
        "ant_array, without duplicates.",
    ):
        gain_data.reorder_antennas(order=gain_data.antenna_numbers[:8])


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_reorder_freqs(
    future_shapes, caltype, metadata_only, gain_data, delay_data_inputflag
):
    if caltype == "gain":
        calobj = gain_data
        # add total_quality_array
        calobj.total_quality_array = np.tile(
            np.arange(calobj.Nfreqs, dtype=float)[:, np.newaxis, np.newaxis],
            (1, calobj.Ntimes, calobj.Njones),
        )
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        if caltype == "delay":
            calobj.freq_array = gain_data.freq_array
            calobj.channel_width = gain_data.channel_width
            calobj.Nfreqs = gain_data.Nfreqs
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy(metadata_only=metadata_only)
    if metadata_only:
        calobj = calobj2.copy()

    if future_shapes and caltype == "delay":
        with uvtest.check_warnings(
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
            if future_shapes:
                total_quality_diff = np.diff(calobj2.total_quality_array, axis=0)
            else:
                total_quality_diff = np.diff(calobj2.total_quality_array, axis=1)
            assert np.all(total_quality_diff < 0)

        calobj.reorder_freqs(channel_order=np.flip(np.arange(calobj.Nfreqs)))
        assert calobj == calobj2


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_reorder_freqs_multi_spw(caltype, multi_spw_gain, multi_spw_delay):
    if caltype == "gain":
        calobj = multi_spw_gain
    else:
        calobj = multi_spw_delay

    if not calobj.future_array_shapes:
        calobj.use_future_array_shapes()

    calobj2 = calobj.copy()

    if caltype == "delay":
        with uvtest.check_warnings(
            [UserWarning, DeprecationWarning],
            match=[
                "channel_order and select_spws are ignored for wide-band "
                "calibration solutions",
                "The input_flag_array is deprecated and will be removed in version 2.5",
            ],
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
        match="If spw_order is an array, it must contain all indicies for "
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
        match="Index array for channel_order must contain all indicies for "
        "the frequency axis, without duplicates.",
    ):
        gain_data.reorder_freqs(
            channel_order=np.arange(gain_data.Nfreqs, dtype=float) * 2
        )

    with pytest.raises(
        ValueError,
        match="Index array for channel_order must contain all indicies for "
        "the frequency axis, without duplicates.",
    ):
        gain_data.reorder_freqs(channel_order=np.arange(3))


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_reorder_times(
    future_shapes, caltype, metadata_only, gain_data, delay_data_inputflag
):
    if caltype == "gain":
        calobj = gain_data
        # add total_quality_array
        calobj.total_quality_array = np.tile(
            np.arange(calobj.Ntimes, dtype=float)[np.newaxis, :, np.newaxis],
            (calobj.Nfreqs, 1, calobj.Njones),
        )
        calobj.check()
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy(metadata_only=metadata_only)
    if metadata_only:
        calobj = calobj2.copy()

    # this should be a no-op
    calobj.reorder_times()
    assert calobj == calobj2

    calobj2.reorder_times(order="-time")
    time_diff = np.diff(calobj2.time_array)
    assert np.all(time_diff < 0)

    if caltype == "gain" and not metadata_only:
        # check total quality array
        if future_shapes:
            total_quality_diff = np.diff(calobj2.total_quality_array, axis=1)
        else:
            total_quality_diff = np.diff(calobj2.total_quality_array, axis=2)
        assert np.all(total_quality_diff < 0)

    calobj.reorder_times(order=np.flip(np.arange(calobj.Ntimes)))
    assert calobj == calobj2


def test_reorder_times_errors(gain_data):
    with pytest.raises(
        ValueError,
        match="order must be one of 'time', '-time' or an index array of length Ntimes",
    ):
        gain_data.reorder_times(order="foo")

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indicies for the time axis, "
        "without duplicates.",
    ):
        gain_data.reorder_times(order=np.arange(gain_data.Ntimes) * 2)

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indicies for the time axis, "
        "without duplicates.",
    ):
        gain_data.reorder_times(order=np.arange(7))


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_reorder_jones(
    future_shapes, caltype, metadata_only, gain_data, delay_data_inputflag
):
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

    # all the input objects have a Njones=1, extend to get to 4
    calobj2 = calobj.copy(metadata_only=metadata_only)
    extend_jones_axis(calobj2)

    if caltype == "gain" and not metadata_only:
        # add total_quality_array
        if future_shapes:
            calobj2.total_quality_array = np.tile(
                np.arange(calobj2.Njones, dtype=float)[np.newaxis, np.newaxis, :],
                (calobj2.Nfreqs, calobj2.Ntimes, 1),
            )
        else:
            calobj2.total_quality_array = np.tile(
                np.arange(calobj2.Njones, dtype=float)[
                    np.newaxis, np.newaxis, np.newaxis, :
                ],
                (1, calobj2.Nfreqs, calobj2.Ntimes, 1),
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
        if future_shapes:
            total_quality_diff = np.diff(calobj2.total_quality_array, axis=2)
        else:
            total_quality_diff = np.diff(calobj2.total_quality_array, axis=3)
        assert np.all(total_quality_diff < 0)

    # the default order is "name"
    calobj2.reorder_jones()
    name_array = np.asarray(
        uvutils.jnum2str(calobj2.jones_array, x_orientation=calobj2.x_orientation)
    )
    sorted_names = np.sort(name_array)
    assert np.all(sorted_names == name_array)

    # test sorting with an index array. Sort back to number first so indexing works
    sorted_nums = uvutils.jstr2num(sorted_names, x_orientation=calobj.x_orientation)
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
        calobj.reorder_jones(order="foo")

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indicies for "
        "the jones axis, without duplicates.",
    ):
        calobj.reorder_jones(order=np.arange(gain_data.Njones) * 2)

    with pytest.raises(
        ValueError,
        match="If order is an array, it must contain all indicies for "
        "the jones axis, without duplicates.",
    ):
        calobj.reorder_jones(order=np.arange(2))


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:Combined Jones elements are not evenly spaced")
@pytest.mark.filterwarnings("ignore:Combined frequencies are not evenly spaced")
@pytest.mark.filterwarnings("ignore:Cannot reorder the frequency/spw axis with only")
@pytest.mark.parametrize("add_type", ["ant", "time", "freq", "jones"])
@pytest.mark.parametrize("sort_type", ["ant", "time", "freq", "jones"])
@pytest.mark.parametrize(
    ["future_shapes", "wide_band"], [[True, False], [False, False], [True, True]]
)
def test_add_different_sorting(
    add_type, sort_type, future_shapes, wide_band, gain_data, wideband_gain
):
    if wide_band:
        calobj = wideband_gain.copy()
        calobj.check()
        assert calobj.freq_range is not None
    else:
        calobj = gain_data.copy()
    # add total_quality_array and initial flag array
    calobj.input_flag_array = copy.copy(calobj.flag_array)
    if add_type != "ant":
        calobj.total_quality_array = np.random.random(
            calobj._total_quality_array.expected_shape(calobj)
        )

    # all the input objects have a Njones=1, extend to get to 4
    extend_jones_axis(calobj, total_quality=False)

    if future_shapes:
        calobj.use_future_array_shapes()

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


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
@pytest.mark.parametrize("quality", [True, False])
def test_add_antennas_multispw(future_shapes, multi_spw_gain, quality, method):
    """Test adding antennas between two UVCal objects"""
    calobj = multi_spw_gain

    if not quality:
        calobj.quality_array = None

    if not future_shapes:
        calobj.use_current_array_shapes()

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
    assert uvutils._check_histories(
        calobj_full.history + "  Downselected to specific "
        "antennas using pyuvdata. Combined "
        "data along antenna axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full


@pytest.mark.filterwarnings("ignore:The freq_range attribute should not be set if")
@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_frequencies(future_shapes, gain_data, method):
    """Test adding frequencies between two UVCal objects"""
    # don't test on delays because there's no freq axis for the delay array
    calobj = gain_data

    if not future_shapes:
        calobj.use_current_array_shapes()
    calobj2 = calobj.copy()
    calobj_full = calobj.copy()
    if future_shapes:
        freqs1 = calobj.freq_array[np.arange(0, calobj.Nfreqs // 2)]
        freqs2 = calobj.freq_array[np.arange(calobj.Nfreqs // 2, calobj.Nfreqs)]
    else:
        freqs1 = calobj.freq_array[0, np.arange(0, calobj.Nfreqs // 2)]
        freqs2 = calobj.freq_array[0, np.arange(calobj.Nfreqs // 2, calobj.Nfreqs)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    if method == "fast_concat":
        kwargs = {"axis": "freq", "inplace": True}
    else:
        kwargs = {}

    getattr(calobj, method)(calobj2, **kwargs)
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        calobj_full.history + "  Downselected to specific "
        "frequencies using pyuvdata. Combined "
        "data along frequency axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # test for when total_quality_array is present in first file but not second
    # also check for missing flex_spw_id_array and freq_range in one object
    calobj.select(frequencies=freqs1)
    calobj.flex_spw_id_array = None
    if future_shapes:
        calobj.freq_range = np.array(
            [np.min(calobj.freq_array), np.max(calobj.freq_array)]
        )[np.newaxis, :]
    else:
        calobj.freq_range = None
        calobj2.freq_range = np.array(
            [np.min(calobj2.freq_array), np.max(calobj2.freq_array)]
        )
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.zeros(calobj2._total_quality_array.expected_shape(calobj2))
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=0)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = tqa
    msg = [
        "flex_spw_id_array is not set. It will be required starting in version 3.0 "
        "for non-wide-band objects",
        "The freq_range attribute should not be set if cal_type='gain' and "
        "wide_band=False. This will become an error in version 3.0.",
    ]
    warn_type = [DeprecationWarning, DeprecationWarning]
    if method == "fast_concat":
        msg.extend(
            [
                "Some objects have the flex_spw_id_array set and some do not. Combined "
                "object will have it set.",
                "Some objects have the freq_range set and and some do not. "
                "Combined object will not have it set.",
            ]
        )
        warn_type.extend([UserWarning, UserWarning])
    else:
        msg.extend(
            [
                "One object has the flex_spw_id_array set and one does not. Combined "
                "object will have it set.",
                "One object has the freq_range set and one does not. Combined "
                "object will not have it set.",
            ]
        )
        warn_type.extend([UserWarning, UserWarning])

    with uvtest.check_warnings(warn_type, match=msg):
        getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
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
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=0)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = None
    calobj2.total_quality_array = tqa2
    calobj.freq_range = np.array([np.min(calobj.freq_array), np.max(calobj.freq_array)])
    calobj2.freq_range = np.array(
        [np.min(calobj2.freq_array), np.max(calobj2.freq_array)]
    )
    if future_shapes:
        calobj.freq_range = calobj.freq_range[np.newaxis, :]
        calobj2.freq_range = calobj2.freq_range[np.newaxis, :]

    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
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
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=0)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when input_flag_array and quality_array is present in first file but not
    # in second
    calobj = calobj_full.copy()
    calobj.input_flag_array = np.zeros(
        calobj._input_flag_array.expected_shape(calobj), dtype=bool
    )
    calobj2 = calobj.copy()
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
    ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(np.bool_)
    qa = np.ones(calobj._quality_array.expected_shape(calobj))
    qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2))
    if future_shapes:
        ax_num = 1
    else:
        ax_num = 2

    tot_ifa = np.concatenate([ifa, ifa2], axis=ax_num)
    tot_qa = np.concatenate([qa, qa2], axis=ax_num)

    calobj.input_flag_array = ifa
    calobj2.input_flag_array = None
    calobj.quality_array = qa
    calobj2.quality_array = None
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(calobj.input_flag_array, tot_ifa)
    print(calobj.quality_array[2:4, :, 0, 0])
    print(tot_qa[2:4, :, 0, 0])
    assert np.allclose(calobj.quality_array, tot_qa)

    # test for when input_flag_array is present in second file but not first
    calobj.select(frequencies=freqs1)
    ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
    ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(np.bool_)
    qa = np.zeros(calobj._quality_array.expected_shape(calobj))
    qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2))

    tot_ifa = np.concatenate([ifa, ifa2], axis=ax_num)
    tot_qa = np.concatenate([qa, qa2], axis=ax_num)

    calobj.input_flag_array = None
    calobj2.input_flag_array = ifa2
    calobj.quality_array = None
    calobj2.quality_array = qa2
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(calobj.input_flag_array, tot_ifa)
    assert np.allclose(calobj.quality_array, tot_qa)

    # Out of order - freqs
    calobj = calobj_full.copy()
    calobj2 = calobj_full.copy()
    calobj.select(frequencies=freqs2)
    calobj2.select(frequencies=freqs1)
    if method == "fast_concat":
        warn_type = UserWarning
        msg = "Combined frequencies are not evenly spaced"
    else:
        warn_type = None
        msg = ""
    with uvtest.check_warnings(warn_type, match=msg):
        getattr(calobj, method)(calobj2, **kwargs)
    calobj.history = calobj_full.history
    if method == "fast_concat":
        # need to sort object first
        calobj.reorder_freqs(channel_order="freq")
    assert calobj == calobj_full


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:One object has the freq_range set and one does not")
@pytest.mark.filterwarnings("ignore:The freq_range attribute should not be set if")
@pytest.mark.filterwarnings("ignore:Some objects have the freq_range set")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    ["split_f_ind", "freq_range1", "freq_range2"],
    [[5, True, True], [3, False, False], [5, True, False]],
)
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add_frequencies_multispw(
    future_shapes, split_f_ind, method, freq_range1, freq_range2, multi_spw_gain
):
    """Test adding frequencies between two UVCal objects"""
    # don't test on delays because there's no freq axis for the delay array

    # split_f_ind=5 splits the objects in the same place as the spws split
    # (so each object has only one spw). A different value splits within an spw.

    calobj = multi_spw_gain

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    if future_shapes:
        freqs1 = calobj.freq_array[np.arange(0, split_f_ind)]
        freqs2 = calobj.freq_array[np.arange(split_f_ind, calobj.Nfreqs)]
    else:
        freqs1 = calobj.freq_array[0, np.arange(0, split_f_ind)]
        freqs2 = calobj.freq_array[0, np.arange(split_f_ind, calobj.Nfreqs)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    warn_type = []
    msg = []
    if freq_range1:
        warn_type.append(DeprecationWarning)
        msg.append(
            "The freq_range attribute should not be set if cal_type='gain' and "
            "wide_band=False. This will become an error in version 3.0."
        )
        if future_shapes:
            calobj.freq_range = np.array(
                [np.min(calobj.freq_array), np.max(calobj.freq_array)]
            )[np.newaxis, :]
        else:
            calobj.freq_range = np.array(
                [np.min(calobj.freq_array), np.max(calobj.freq_array)]
            )
    else:
        calobj.freq_range = None

    if freq_range2:
        warn_type.append(DeprecationWarning)
        msg.append(
            "The freq_range attribute should not be set if cal_type='gain' and "
            "wide_band=False. This will become an error in version 3.0."
        )
        if future_shapes:
            calobj2.freq_range = np.array(
                [np.min(calobj2.freq_array), np.max(calobj2.freq_array)]
            )[np.newaxis, :]
        else:
            calobj2.freq_range = np.array(
                [np.min(calobj2.freq_array), np.max(calobj2.freq_array)]
            )
    else:
        calobj2.freq_range = None

    if freq_range1 != freq_range2:
        warn_type.append(UserWarning)
        if method == "fast_concat":
            msg.append(
                "Some objects have the freq_range set and some do not. "
                "Combined object will not have it set."
            )
        else:
            msg.append(
                "One object has the freq_range set and one does not. Combined "
                "object will not have it set."
            )
    elif freq_range1:
        warn_type.append(DeprecationWarning)
        msg.append(
            "The freq_range attribute should not be set if cal_type='gain' and "
            "wide_band=False. This will become an error in version 3.0."
        )

    if freq_range1 and freq_range2:
        if future_shapes:
            calobj_full.freq_range = np.concatenate(
                [calobj.freq_range, calobj2.freq_range], axis=0
            )
        else:
            calobj_full.freq_range = np.array(
                [np.min(calobj_full.freq_array), np.max(calobj_full.freq_array)]
            )
    else:
        calobj_full.freq_range = None

    if len(warn_type) == 0:
        warn_type = None
        msg = ""

    if method == "fast_concat":
        kwargs = {"axis": "freq"}
    else:
        kwargs = {}

    with uvtest.check_warnings(warn_type, match=msg):
        calobj_sum = getattr(calobj, method)(calobj2, **kwargs)

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
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
            with pytest.raises(
                ValueError,
                match="Channels from different spectral windows are interspersed "
                "with one another, rather than being grouped together along the "
                "frequency axis. Most file formats do not support such "
                "non-grouping of data.",
            ):
                calobj_sum = calobj2.fast_concat(calobj, axis="freq")
            return
    else:
        calobj_sum = calobj2 + calobj

    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
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


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
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
        ant1_inds = np.nonzero(np.in1d(calobj_full.ant_array, ants1))[0]
        ant2_inds = np.nonzero(np.in1d(calobj_full.ant_array, ants2))[0]
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
        calobj_full.input_flag_array[ant1_inds, 1:] = True
        calobj_full.input_flag_array[ant2_inds, 0] = True

    if method == "fast_concat":
        kwargs = {"axis": axis, "inplace": False}
    else:
        kwargs = {}

    calobj3 = getattr(calobj, method)(calobj2, **kwargs)

    # Check history is correct, before replacing and doing a full object check
    if axis == "multi":
        assert uvutils._check_histories(
            calobj_full.history + "  Downselected to specific antennas, spectral "
            "windows using pyuvdata. Combined data along antenna, spectral window axis "
            "using pyuvdata.",
            calobj3.history,
        )
    elif axis == "spw":
        assert uvutils._check_histories(
            calobj_full.history + "  Downselected to specific spectral windows using "
            "pyuvdata. Combined data along spectral window axis using pyuvdata.",
            calobj3.history,
        )
    elif axis == "antenna":
        assert uvutils._check_histories(
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
        assert uvutils._check_histories(
            calobj_full.history + "  Downselected to specific antennas, spectral "
            "windows using pyuvdata. Combined data along antenna, spectral window axis "
            "using pyuvdata.",
            calobj3.history,
        )
    elif axis == "spw":
        assert uvutils._check_histories(
            calobj_full.history + "  Downselected to specific spectral windows using "
            "pyuvdata. Combined data along spectral window axis using pyuvdata.",
            calobj3.history,
        )
    elif axis == "antenna":
        assert uvutils._check_histories(
            calobj_full.history + "  Downselected to specific antennas using pyuvdata. "
            "Combined data along antenna axis using pyuvdata.",
            calobj3.history,
        )
    calobj3.history = calobj_full.history
    assert calobj3 == calobj_full


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_times(future_shapes, caltype, method, gain_data, delay_data_inputflag):
    """Test adding times between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

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
    assert uvutils._check_histories(
        calobj_full.history + "  Downselected to specific "
        "times using pyuvdata. Combined "
        "data along time axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # test for when total_quality_array is present in first file but not second
    calobj.select(times=times1)
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.zeros(calobj2._total_quality_array.expected_shape(calobj2))
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = tqa
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in second file but not first
    calobj.select(times=times1)
    tqa = np.zeros(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = None
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in both
    calobj.select(times=times1)
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tqa *= 2
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    if caltype == "delay":
        # test for when input_flag_array & quality_array is present in first file but
        # not in second
        calobj.select(times=times1)
        ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        qa = np.ones(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2), dtype=float)
        if future_shapes:
            ax_num = 2
        else:
            ax_num = 3
        tot_ifa = np.concatenate([ifa, ifa2], axis=ax_num)
        tot_qa = np.concatenate([qa, qa2], axis=ax_num)

        calobj.input_flag_array = ifa
        calobj2.input_flag_array = None
        calobj.quality_array = qa
        calobj2.quality_array = None
        getattr(calobj, method)(calobj2, **kwargs)
        assert np.allclose(calobj.input_flag_array, tot_ifa)
        assert np.allclose(calobj.quality_array, tot_qa)

        # test for when input_flag_array is present in second file but not first
        calobj.select(times=times1)
        ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        qa = np.zeros(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2), dtype=float)
        tot_ifa = np.concatenate([ifa, ifa2], axis=ax_num)
        tot_qa = np.concatenate([qa, qa2], axis=ax_num)

        calobj.input_flag_array = None
        calobj2.input_flag_array = ifa2
        calobj.quality_array = None
        calobj2.quality_array = qa2
        getattr(calobj, method)(calobj2, **kwargs)
        assert np.allclose(calobj.input_flag_array, tot_ifa)
        assert np.allclose(calobj.quality_array, tot_qa)

    # Out of order - times
    calobj = calobj_full.copy()
    calobj2 = calobj.copy()
    calobj.select(times=times2)
    calobj2.select(times=times1)
    getattr(calobj, method)(calobj2, **kwargs)
    calobj.history = calobj_full.history
    if method == "fast_concat":
        # need to sort object first
        calobj.reorder_times()
    assert calobj == calobj_full


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
@pytest.mark.parametrize("quality", [True, False])
def test_add_times_multispw(future_shapes, method, multi_spw_gain, quality):
    """Test adding times between two UVCal objects"""
    calobj = multi_spw_gain

    if not quality:
        calobj.quality_array = None

    if not future_shapes:
        calobj.use_current_array_shapes()

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
    assert uvutils._check_histories(
        calobj_full.history + "  Downselected to specific "
        "times using pyuvdata. Combined "
        "data along time axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_add_jones(future_shapes, caltype, method, gain_data, delay_data_inputflag):
    """Test adding Jones axes between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    if not future_shapes:
        calobj.use_current_array_shapes()

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
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    calobj.total_quality_array = tqa
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in second file but not first
    calobj = calobj_original.copy()
    tqa = np.zeros(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
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
    if future_shapes:
        tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    else:
        tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    getattr(calobj, method)(calobj2, **kwargs)
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    if caltype == "delay":
        # test for when input_flag_array & quality array is present in first file but
        # not in second
        calobj = calobj_original.copy()
        ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        qa = np.ones(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.zeros(calobj2._quality_array.expected_shape(calobj2), dtype=float)
        if future_shapes:
            ax_num = 3
        else:
            ax_num = 4
        tot_ifa = np.concatenate([ifa, ifa2], axis=ax_num)
        tot_qa = np.concatenate([qa, qa2], axis=ax_num)
        calobj.input_flag_array = ifa
        calobj2.input_flag_array = None
        calobj.quality_array = qa
        calobj2.quality_array = None
        getattr(calobj, method)(calobj2, **kwargs)
        assert np.allclose(calobj.input_flag_array, tot_ifa)
        assert np.allclose(calobj.quality_array, tot_qa)

        # test for when input_flag_array is present in second file but not first
        calobj = calobj_original.copy()
        ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        qa = np.zeros(calobj._quality_array.expected_shape(calobj), dtype=float)
        qa2 = np.ones(calobj2._quality_array.expected_shape(calobj2), dtype=float)

        tot_ifa = np.concatenate([ifa, ifa2], axis=ax_num)
        tot_qa = np.concatenate([qa, qa2], axis=ax_num)

        calobj.input_flag_array = None
        calobj2.input_flag_array = ifa2
        calobj.quality_array = None
        calobj2.quality_array = qa2
        getattr(calobj, method)(calobj2, **kwargs)
        assert np.allclose(calobj.input_flag_array, tot_ifa)
        assert np.allclose(calobj.quality_array, tot_qa)

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


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
@pytest.mark.parametrize("quality", [True, False])
def test_add_jones_multispw(future_shapes, method, quality, multi_spw_gain):
    """Test adding Jones axes between two UVCal objects"""
    calobj = multi_spw_gain

    if not quality:
        calobj.quality_array = None

    if not future_shapes:
        calobj.use_current_array_shapes()

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


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add(caltype, method, gain_data, delay_data_inputflag):
    """Test miscellaneous aspects of add method"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

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
    assert uvutils._check_histories(
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
    assert uvutils._check_histories(
        calobj_original.history + " Combined data along antenna axis "
        "using pyuvdata. Unique part of next object history follows.  "
        + additional_history,
        new_cal.history,
    )

    kwargs["verbose_history"] = True
    new_cal = getattr(calobj, method)(calobj2, **kwargs)
    assert uvutils._check_histories(
        calobj_original.history + " Combined data along antenna axis "
        "using pyuvdata. Next object history follows.  " + calobj2.history,
        new_cal.history,
    )


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
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
        frequencies=calobj_full.freq_array[:2],
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
    # add an input_flag_array
    calobj_full.input_flag_array = calobj_full.flag_array

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
        ant1_inds = np.nonzero(np.in1d(calobj_full.ant_array, ants1))[0]
        ant2_inds = np.nonzero(np.in1d(calobj_full.ant_array, ants2))[0]
    else:
        ant1_inds = np.arange(calobj_full.Nants_data)
        ant2_inds = np.arange(calobj_full.Nants_data)
    if freq:
        freq1_inds = np.nonzero(np.in1d(calobj_full.freq_array, freqs1))[0]
        freq2_inds = np.nonzero(np.in1d(calobj_full.freq_array, freqs2))[0]
    else:
        freq1_inds = np.arange(calobj_full.Nfreqs)
        freq2_inds = np.arange(calobj_full.Nfreqs)
    if time:
        time1_inds = np.nonzero(np.in1d(calobj_full.time_array, times1))[0]
        time2_inds = np.nonzero(np.in1d(calobj_full.time_array, times2))[0]
    else:
        time1_inds = np.arange(calobj_full.Ntimes)
        time2_inds = np.arange(calobj_full.Ntimes)
    if jones:
        jones1_inds = np.nonzero(np.in1d(calobj_full.jones_array, jones1))[0]
        jones2_inds = np.nonzero(np.in1d(calobj_full.jones_array, jones2))[0]
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
        calobj_full.input_flag_array[inds] = True

    # reset history to equality passes
    calobj3.history = calobj_full.history

    assert calobj3 == calobj_full


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_add_errors(
    caltype, method, gain_data, delay_data, multi_spw_gain, wideband_gain
):
    """Test behavior that will raise errors"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data

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
        with pytest.raises(
            ValueError,
            match="These objects have overlapping data and cannot be combined.",
        ):
            calobj + calobj2

    # test addition of UVCal and non-UVCal object
    with pytest.raises(ValueError, match="Only UVCal "):
        getattr(calobj, method)("foo", **kwargs)

    # test compatibility param mismatch
    calobj2.telescope_name = "PAPER"
    with pytest.raises(ValueError, match="Parameter telescope_name does not match"):
        getattr(calobj, method)(calobj2, **kwargs)

    # test array shape mismatch
    calobj2 = calobj.copy()
    calobj2.use_current_array_shapes()
    msg = (
        " objects must have the same `future_array_shapes` parameter. Use the "
        "`use_future_array_shapes` or `use_current_array_shapes` methods to convert "
        "them."
    )
    if method == "fast_concat":
        msg = "All" + msg
    else:
        msg = "Both" + msg

    with pytest.raises(ValueError, match=msg):
        getattr(calobj, method)(calobj2, **kwargs)

    # test flex_spw mismatch
    with pytest.raises(
        ValueError,
        match="To combine these data, flex_spw must be set to the same value",
    ):
        getattr(gain_data, method)(multi_spw_gain, **kwargs)

    # test wide_band mismatch
    with pytest.raises(
        ValueError,
        match="To combine these data, wide_band must be set to the same value",
    ):
        getattr(gain_data, method)(wideband_gain, **kwargs)


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
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


def test_jones_warning(gain_data):
    """Test having non-contiguous Jones elements"""
    calobj = gain_data
    calobj2 = calobj.copy()

    calobj2.jones_array[0] = -6
    calobj += calobj2
    calobj2.jones_array[0] = -8
    with uvtest.check_warnings(UserWarning, match="Combined Jones elements"):
        calobj.__iadd__(calobj2)
    assert sorted(calobj.jones_array) == [-8, -6, -5]


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("method", ["__iadd__", "fast_concat"])
def test_frequency_warnings(future_shapes, gain_data, method):
    """Test having uneven or non-contiguous frequencies"""
    # test having unevenly spaced frequency separations
    calobj = gain_data

    if not future_shapes:
        calobj.use_current_array_shapes()

    calobj2 = calobj.copy()

    go1 = calobj.copy()
    go2 = calobj2.copy()
    if future_shapes:
        freqs1 = calobj.freq_array[np.arange(0, 5)]
        freqs2 = calobj2.freq_array[np.arange(5, 10)]
    else:
        freqs1 = calobj.freq_array[0, np.arange(0, 5)]
        freqs2 = calobj2.freq_array[0, np.arange(5, 10)]

    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    if method == "fast_concat":
        kwargs = {"axis": "freq", "inplace": True}
    else:
        kwargs = {}

    # change the last frequency bin to be smaller than the others
    if future_shapes:
        df = calobj2.freq_array[-1] - calobj2.freq_array[-2]
        calobj2.freq_array[-1] = calobj2.freq_array[-2] + df / 2
    else:
        df = calobj2.freq_array[0, -1] - calobj2.freq_array[0, -2]
        calobj2.freq_array[0, -1] = calobj2.freq_array[0, -2] + df / 2
    with uvtest.check_warnings(
        UserWarning, match="Combined frequencies are not evenly spaced"
    ):
        getattr(calobj, method)(calobj2, **kwargs)

    assert calobj.freq_array.size == freqs1.size + freqs2.size

    # now check having "non-contiguous" frequencies
    calobj = go1.copy()
    calobj2 = go2.copy()
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    # artificially space out frequencies
    calobj.freq_array *= 10
    calobj2.freq_array *= 10
    with uvtest.check_warnings(
        UserWarning,
        match="Combined frequencies are separated by more than their channel width",
    ):
        getattr(calobj, method)(calobj2, **kwargs)

    assert calobj.freq_array.size == freqs1.size + freqs2.size

    freqs1 *= 10
    freqs2 *= 10
    freqs = np.concatenate([freqs1, freqs2])
    if future_shapes:
        freq_arr = calobj.freq_array
    else:
        freq_arr = calobj.freq_array[0, :]
    assert np.allclose(
        freq_arr,
        freqs,
        rtol=calobj._freq_array.tols[0],
        atol=calobj._freq_array.tols[1],
    )


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
    with uvtest.check_warnings(
        UserWarning, match="UVParameter observer does not match"
    ):
        calobj.__iadd__(calobj2)

    freqs = np.concatenate([freqs1, freqs2])
    assert np.allclose(
        calobj.freq_array,
        freqs,
        rtol=calobj._freq_array.tols[0],
        atol=calobj._freq_array.tols[1],
    )


@pytest.mark.filterwarnings("ignore:The input_flag_array is deprecated")
@pytest.mark.filterwarnings("ignore:When converting a delay-style cal to future array")
@pytest.mark.parametrize("caltype", ["gain", "delay"])
@pytest.mark.parametrize("method", ["__add__", "fast_concat"])
def test_multi_files(caltype, method, gain_data, delay_data_inputflag, tmp_path):
    """Test read function when multiple files are included"""
    if caltype == "gain":
        calobj = gain_data
    else:
        calobj = delay_data_inputflag

    calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    n_times2 = calobj.Ntimes // 2
    # Break up delay object into two objects, divided in time
    times1 = calobj.time_array[:n_times2]
    times2 = calobj.time_array[n_times2:]
    calobj.select(times=times1)
    calobj2.select(times=times2)
    # Write those objects to files
    f1 = str(tmp_path / "read_multi1.calfits")
    f2 = str(tmp_path / "read_multi2.calfits")
    calobj.write_calfits(f1, clobber=True)
    calobj2.write_calfits(f2, clobber=True)
    # Read both files together
    if method == "fast_concat":
        calobj = UVCal.from_file([f1, f2], axis="time", use_future_array_shapes=True)
    else:
        warn_type = [DeprecationWarning]
        msg = [
            "Reading multiple files from file specific read methods is deprecated. "
            "Use the generic `UVCal.read` method instead."
        ]
        if caltype == "delay":
            warn_type += 2 * [UserWarning] + [DeprecationWarning] * 5
            msg += 2 * ["When converting a delay-style cal to future array"] + 5 * [
                "The input_flag_array is deprecated and will be removed in version 2.5"
            ]

        with uvtest.check_warnings(warn_type, match=msg):
            calobj.read_calfits([f1, f2], use_future_array_shapes=True)

    assert uvutils._check_histories(
        calobj_full.history + "  Downselected to specific times"
        " using pyuvdata. Combined data "
        "along time axis using pyuvdata.",
        calobj.history,
    )
    assert calobj.filename == ["read_multi1.calfits", "read_multi2.calfits"]
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # check metadata only read
    calobj = UVCal.from_file([f1, f2], read_data=False, use_future_array_shapes=True)
    calobj_full_metadata_only = calobj_full.copy(metadata_only=True)

    calobj.history = calobj_full_metadata_only.history
    assert calobj == calobj_full_metadata_only


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_uvcal_get_methods(future_shapes, gain_data):
    # load data
    uvc = gain_data

    if not future_shapes:
        uvc.use_current_array_shapes()

    # test get methods: add in a known value and make sure it is returned
    key = (10, "Jee")
    uvc.gain_array[1] = 0.0
    gain_arr = uvc.get_gains(key)
    flag_arr = uvc.get_flags(key)
    quality_arr = uvc.get_quality(key)

    # test shapes
    assert np.all(np.isclose(gain_arr, 0.0))
    assert gain_arr.shape == (uvc.Nfreqs, uvc.Ntimes)
    assert flag_arr.shape == (uvc.Nfreqs, uvc.Ntimes)
    assert quality_arr.shape == (uvc.Nfreqs, uvc.Ntimes)

    # test against by-hand indexing
    if future_shapes:
        expected_array = uvc.gain_array[
            uvc.ant_array.tolist().index(10), :, :, uvc.jones_array.tolist().index(-5)
        ]
    else:
        expected_array = uvc.gain_array[
            uvc.ant_array.tolist().index(10),
            0,
            :,
            :,
            uvc.jones_array.tolist().index(-5),
        ]
    np.testing.assert_array_almost_equal(gain_arr, expected_array)

    # test variable key input
    gain_arr2 = uvc.get_gains(*key)
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
    assert uvc._has_key(antnum=10)
    assert uvc._has_key(jpol="Jee")
    assert uvc._has_key(antnum=10, jpol="Jee")
    assert not uvc._has_key(antnum=10, jpol="Jnn")
    assert not uvc._has_key(antnum=101, jpol="Jee")

    # test exceptions
    pytest.raises(ValueError, uvc.get_gains, 1)
    pytest.raises(ValueError, uvc.get_gains, (10, "Jnn"))
    uvc.cal_type = "delay"
    pytest.raises(ValueError, uvc.get_gains, 10)


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
def test_write_read_optional_attrs(gain_data, tmp_path):
    # read a test file
    cal_in = gain_data

    # set some optional parameters
    cal_in.gain_scale = "Jy"
    cal_in.sky_field = "GLEAM"

    # write
    write_file_calfits = str(tmp_path / "test.calfits")
    with uvtest.check_warnings(
        DeprecationWarning,
        match="The sky_field parameter is deprecated and will be removed in version "
        "2.5",
    ):
        cal_in.write_calfits(write_file_calfits, clobber=True)

    # read and compare
    # also check that passing a single file in a list works properly
    with uvtest.check_warnings(
        DeprecationWarning,
        match="The sky_field parameter is deprecated and will be removed in version "
        "2.5",
    ):
        cal_in2 = UVCal.from_file([write_file_calfits], use_future_array_shapes=True)
    assert cal_in == cal_in2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("caltype", ["gain", "delay", "unknown", None])
def test_copy(future_shapes, gain_data, delay_data_inputflag, caltype):
    """Test the copy method"""
    if caltype == "gain":
        uv_object = gain_data
    elif caltype == "delay":
        uv_object = delay_data_inputflag
    else:
        uv_object = gain_data
        with uvtest.check_warnings(
            DeprecationWarning,
            match="Setting the cal_type to 'unknown' is deprecated. This will become "
            "an error in version 2.5",
        ):
            uv_object._set_unknown_cal_type()
        uv_object.cal_type = caltype

    if not future_shapes:
        if caltype is not None:
            uv_object.use_current_array_shapes()
        else:
            with pytest.raises(
                ValueError,
                match="Cannot get required data params because cal_type is not set.",
            ):
                uv_object.use_current_array_shapes()

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
        [name.replace("ant", "HH") for name in gain_data.antenna_names]
    )
    if antnamefix == "all":
        gain_data.antenna_names = new_names
    else:
        gain_data.antenna_names[0 : gain_data.Nants_telescope // 2] = new_names[
            0 : gain_data.Nants_telescope // 2
        ]

    # remove the antenna_positions to test matching them on read
    write_file = str(tmp_path / "test.calfits")
    write_file2 = str(tmp_path / "test2.calfits")
    gain_data.write_calfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        primary_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["ANTENNAS"]]

        table = Table(ant_hdu.data)
        del table["ANTXYZ"]
        ant_hdu = fits.BinTableHDU(table)
        ant_hdu.header["EXTNAME"] = "ANTENNAS"

        hdulist = fits.HDUList([primary_hdu, ant_hdu])

        hdulist.writeto(write_file2)
        hdulist.close()

    with uvtest.check_warnings(
        UserWarning,
        match="antenna_positions are not set or are being overwritten. Using known "
        "values for HERA.",
    ):
        gain_data2 = UVCal.from_file(write_file2, use_future_array_shapes=True)

    assert gain_data2.antenna_positions is not None
    assert gain_data == gain_data2


@pytest.mark.parametrize("modtype", ["rename", "swap"])
def test_set_antpos_from_telescope_errors(gain_data, modtype, tmp_path):
    """Test that setting antenna positions doesn't happen if ants don't match."""
    # fix the antenna names in the uvcal object to match telescope object
    new_names = np.array(
        [name.replace("ant", "HH") for name in gain_data.antenna_names]
    )
    gain_data.antenna_names = new_names

    if modtype == "rename":
        # change the name & number of one of the antennas
        orig_num = gain_data.antenna_numbers[0]
        gain_data.antenna_names[0] = "HH400"
        gain_data.antenna_numbers[0] = 400
        gain_data.ant_array[np.where(gain_data.ant_array == orig_num)[0]] = 400
    else:
        # change the name of one antenna and swap the number with a different antenna
        orig_num = gain_data.antenna_numbers[0]
        gain_data.antenna_names[0] = "HH400"
        gain_data.antenna_numbers[0] = gain_data.antenna_numbers[1]
        gain_data.antenna_numbers[1] = orig_num

    # remove the antenna_positions to test matching them on read
    write_file = str(tmp_path / "test.calfits")
    write_file2 = str(tmp_path / "test2.calfits")
    gain_data.write_calfits(write_file)
    with fits.open(write_file, memmap=True) as hdu_list:
        hdunames = uvutils._fits_indexhdus(hdu_list)
        primary_hdu = hdu_list[0]
        ant_hdu = hdu_list[hdunames["ANTENNAS"]]

        table = Table(ant_hdu.data)
        del table["ANTXYZ"]
        ant_hdu = fits.BinTableHDU(table)
        ant_hdu.header["EXTNAME"] = "ANTENNAS"

        hdulist = fits.HDUList([primary_hdu, ant_hdu])

        hdulist.writeto(write_file2)
        hdulist.close()

    with pytest.raises(
        ValueError, match="Required UVParameter _antenna_positions has not been set."
    ):
        with uvtest.check_warnings(
            [UserWarning],
            match=[
                "Not all antennas have positions in the telescope object. "
                "Not setting antenna_positions."
            ],
        ):
            gain_data2 = UVCal.from_file(write_file2, use_future_array_shapes=True)

    with uvtest.check_warnings(
        UserWarning,
        match=[
            "Not all antennas have positions in the telescope object. "
            "Not setting antenna_positions."
        ],
    ):
        gain_data2 = UVCal.from_file(
            write_file2, use_future_array_shapes=True, run_check=False
        )

    assert gain_data2.antenna_positions is None


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
        ValueError, match="The only supported file_types are 'calfits' and 'fhd'."
    ):
        UVCal.from_file(gainfile, file_type="foo")


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("uvdata_future_shapes", [True, False])
@pytest.mark.parametrize("uvcal_future_shapes", [True, False])
@pytest.mark.parametrize("flex_spw", [True, False])
def test_init_from_uvdata(
    uvdata_future_shapes, uvcal_future_shapes, flex_spw, uvcalibrate_data
):
    uvd, uvc = uvcalibrate_data

    if not uvdata_future_shapes:
        uvd.use_current_array_shapes()
        uvd.flex_spw_id_array = None

    if flex_spw:
        uvd._set_flex_spw()
        uvd.flex_spw_id_array = [1] * (uvd.Nfreqs // 2) + [2] * (uvd.Nfreqs // 2)
        uvd.spw_array = np.array([1, 2])
        uvd.Nspws = 2
        uvd.channel_width = np.full(uvd.Nfreqs, uvd.channel_width)
        uvd.check()

        uvc._set_flex_spw()
        uvc.flex_spw_id_array = [1] * (uvc.Nfreqs // 2) + [2] * (uvc.Nfreqs // 2)
        uvc.spw_array = np.array([1, 2])
        uvc.Nspws = 2
        uvc.channel_width = np.full(uvc.Nfreqs, uvc.channel_width)
        uvc.check()
    else:
        # Always compare to a flex_spw uvcal, because the function always returns
        # a flex_spw uvcal
        uvc._set_flex_spw()

    if not uvcal_future_shapes:
        uvc.use_current_array_shapes()

    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd, uvc.gain_convention, uvc.cal_style, future_array_shapes=uvcal_future_shapes
    )

    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

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

    assert uvc_new == uvc2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.filterwarnings("ignore:Selected frequencies are not contiguous.")
@pytest.mark.parametrize("uvdata_future_shapes", [True, False])
@pytest.mark.parametrize("uvcal_future_shapes", [True, False])
@pytest.mark.parametrize("flex_spw", [True, False])
def test_init_from_uvdata_setfreqs(
    uvdata_future_shapes, uvcal_future_shapes, flex_spw, uvcalibrate_data
):
    uvd, uvc = uvcalibrate_data
    channel_width = uvd.channel_width[0]
    freqs_use = uvd.freq_array[0:5]

    if not uvdata_future_shapes:
        uvd.use_current_array_shapes()

    if not uvcal_future_shapes:
        uvc.use_current_array_shapes()

    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    uvc2 = uvc.copy(metadata_only=True)

    uvc2.select(frequencies=freqs_use)

    if flex_spw:
        flex_spw_id_array = np.array([1, 1, 2, 2, 2])
        uvc2._set_flex_spw()
        uvc2.flex_spw_id_array = flex_spw_id_array
        uvc2.spw_array = np.array([1, 2])
        uvc2.Nspws = 2
        uvc2.channel_width = np.full(uvc2.Nfreqs, uvc2.channel_width)
        # test passing a list instead of a single value
        channel_width = np.full(freqs_use.size, channel_width).tolist()
    else:
        if uvdata_future_shapes:
            flex_spw_id_array = np.zeros(5, dtype=int)
        else:
            flex_spw_id_array = None

        uvc2._set_flex_spw()
        if not uvcal_future_shapes:
            uvc2.channel_width = np.full(uvc2.Nfreqs, uvc2.channel_width)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        uvc.gain_convention,
        uvc.cal_style,
        future_array_shapes=uvcal_future_shapes,
        freq_array=freqs_use,
        channel_width=channel_width,
        flex_spw_id_array=flex_spw_id_array,
    )

    with pytest.warns(
        DeprecationWarning,
        match="The frequencies keyword is deprecated in favor of freq_array",
    ):
        UVCal.initialize_from_uvdata(
            uvd,
            uvc.gain_convention,
            uvc.cal_style,
            future_array_shapes=uvcal_future_shapes,
            frequencies=freqs_use,
            channel_width=channel_width,
            flex_spw_id_array=flex_spw_id_array,
        )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

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

    assert uvc_new == uvc2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("uvdata_future_shapes", [True, False])
@pytest.mark.parametrize("uvcal_future_shapes", [True, False])
@pytest.mark.parametrize("metadata_only", [True, False])
def test_init_from_uvdata_settimes(
    uvdata_future_shapes, uvcal_future_shapes, metadata_only, uvcalibrate_data
):
    uvd, uvc = uvcalibrate_data
    uvc._set_flex_spw()
    integration_time = np.mean(uvd.integration_time)
    times_use = uvc.time_array[0:3]

    if not uvdata_future_shapes:
        uvd.use_current_array_shapes()

    if not uvcal_future_shapes:
        uvc.use_current_array_shapes()
    elif metadata_only:
        # test passing in a list of integration times
        integration_time = np.full(
            times_use.size, np.mean(uvd.integration_time)
        ).tolist()

    uvc2 = uvc.copy(metadata_only=metadata_only)

    uvc2.select(times=times_use)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        uvc.gain_convention,
        uvc.cal_style,
        future_array_shapes=uvcal_future_shapes,
        metadata_only=metadata_only,
        time_array=times_use,
        integration_time=integration_time,
        time_range=uvc.time_range,
    )

    with pytest.warns(
        DeprecationWarning,
        match="The times keyword is deprecated in favor of time_array",
    ):
        UVCal.initialize_from_uvdata(
            uvd,
            uvc.gain_convention,
            uvc.cal_style,
            future_array_shapes=uvcal_future_shapes,
            metadata_only=metadata_only,
            times=times_use,
            integration_time=integration_time,
            time_range=uvc.time_range,
        )
    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

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

    assert uvc_new == uvc2


def test_init_from_uvdata_setjones(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data
    uvc._set_flex_spw()
    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(
        uvd, uvc.gain_convention, uvc.cal_style, jones_array=[-5, -6]
    )

    with pytest.warns(
        DeprecationWarning,
        match="The jones keyword is deprecated in favor of jones_array",
    ):
        UVCal.initialize_from_uvdata(
            uvd, uvc.gain_convention, uvc.cal_style, jones=[-5, -6]
        )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new == uvc2


@pytest.mark.parametrize("pol", ["nn", "ee", "rr", "ll"])
def test_init_single_pol(uvcalibrate_data, pol):
    uvd, uvc = uvcalibrate_data
    uvc._set_flex_spw()

    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    if pol in ["ll", "rr"]:
        # convert to circular pol
        uvd.polarization_array = np.array([-1, -2, -3, -4])
        uvc.jones_array = np.array([-1, -2])

    # downselect to one pol
    uvd.select(polarizations=[pol])
    uvc.select(jones=[pol])

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(uvd, uvc.gain_convention, uvc.cal_style)

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new == uvc2


def test_init_from_uvdata_circular_pol(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data
    uvc._set_flex_spw()

    # convert to circular pol
    uvd.polarization_array = np.array([-1, -2, -3, -4])
    uvc.jones_array = np.array([-1, -2])

    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    uvc2 = uvc.copy(metadata_only=True)

    uvc_new = UVCal.initialize_from_uvdata(uvd, uvc.gain_convention, uvc.cal_style)

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new == uvc2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("uvdata_future_shapes", [True, False])
@pytest.mark.parametrize("uvcal_future_shapes", [True, False])
def test_init_from_uvdata_sky(
    uvdata_future_shapes, uvcal_future_shapes, uvcalibrate_data, fhd_cal_raw
):
    uvd, uvc = uvcalibrate_data
    uvc_sky = fhd_cal_raw
    uvc._set_flex_spw()

    if not uvdata_future_shapes:
        uvd.use_current_array_shapes()

    if not uvcal_future_shapes:
        uvc.use_current_array_shapes()

    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    # make cal object be a sky cal type
    uvc._set_sky()
    params_to_copy = [
        "sky_field",
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
        uvc.gain_convention,
        uvc.cal_style,
        future_array_shapes=uvcal_future_shapes,
        sky_field=uvc_sky.sky_field,
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
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new == uvc2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("uvdata_future_shapes", [True, False])
@pytest.mark.parametrize(
    ["uvcal_future_shapes", "flex_spw"], [[True, False], [True, True], [False, False]]
)
@pytest.mark.parametrize("set_frange", [True, False])
def test_init_from_uvdata_delay(
    uvdata_future_shapes, uvcal_future_shapes, flex_spw, set_frange, uvcalibrate_data
):
    uvd, uvc = uvcalibrate_data

    if not uvdata_future_shapes:
        uvd.use_current_array_shapes()

    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    # make cal object be a delay cal type
    uvc2 = uvc.copy(metadata_only=True)
    uvc2._set_delay()
    uvc2.Nfreqs = 1
    uvc2.flex_spw_id_array = None
    uvc2.freq_array = None
    uvc2.channel_width = None
    uvc2.freq_range = np.array([[np.min(uvc.freq_array), np.max(uvc.freq_array)]])

    if flex_spw:
        spw_cut = uvd.Nfreqs // 2
        uvd._set_flex_spw()
        uvd.flex_spw_id_array = [1] * spw_cut + [2] * spw_cut
        uvd.spw_array = np.array([1, 2])
        uvd.Nspws = 2
        uvd.channel_width = np.full(uvd.Nfreqs, uvd.channel_width)
        uvd.check()

        uvc._set_flex_spw()
        uvc.flex_spw_id_array = np.asarray([1] * spw_cut + [2] * spw_cut)
        uvc.spw_array = np.array([1, 2])
        uvc.Nspws = 2
        uvc.channel_width = np.full(uvc.Nfreqs, uvc.channel_width)
        uvc.check()
    else:
        uvc._set_flex_spw()

    if not uvcal_future_shapes:
        uvc.use_current_array_shapes()
        uvc2.use_current_array_shapes()
    elif flex_spw:
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
        if flex_spw:
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
        uvc.gain_convention,
        uvc.cal_style,
        future_array_shapes=uvcal_future_shapes,
        cal_type="delay",
        freq_range=freq_range,
        spw_array=spw_array,
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new == uvc2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0 when")
@pytest.mark.parametrize("uvdata_future_shapes", [True, False])
@pytest.mark.parametrize("flex_spw", [True, False])
@pytest.mark.parametrize("set_frange", [True, False])
def test_init_from_uvdata_wideband(
    uvdata_future_shapes, flex_spw, set_frange, uvcalibrate_data
):
    uvd, uvc = uvcalibrate_data

    # wide-band gain requires future array shapes

    if not uvdata_future_shapes:
        uvd.use_current_array_shapes()

    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    # make cal object be a wide-band cal
    uvc2 = uvc.copy(metadata_only=True)
    uvc2._set_wide_band()
    uvc2.freq_range = np.asarray([[np.min(uvc.freq_array), np.max(uvc.freq_array)]])
    uvc2.Nfreqs = 1
    uvc2.flex_spw_id_array = None
    uvc2.freq_array = None
    uvc2.channel_width = None

    if flex_spw:
        spw_cut = uvd.Nfreqs // 2
        uvd._set_flex_spw()
        uvd.flex_spw_id_array = [1] * spw_cut + [2] * spw_cut
        uvd.spw_array = np.array([1, 2])
        uvd.Nspws = 2
        uvd.channel_width = np.full(uvd.Nfreqs, uvd.channel_width)
        uvd.check()

        uvc._set_flex_spw()
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
        if flex_spw:
            spw_array = uvc2.spw_array
        else:
            spw_array = None
    else:
        freq_range = None
        spw_array = None

    uvc_new = UVCal.initialize_from_uvdata(
        uvd,
        uvc.gain_convention,
        uvc.cal_style,
        wide_band=True,
        freq_range=freq_range,
        spw_array=spw_array,
    )

    # antenna positions are different by ~6cm or less. The ones in the uvcal file
    # derive from info on our telescope object while the ones in the uvdata file
    # derive from the HERA correlator. I'm not sure why they're different, but it may be
    # because the data are a little old
    assert np.allclose(uvc2.antenna_positions, uvc_new.antenna_positions, atol=0.1)
    uvc_new.antenna_positions = uvc2.antenna_positions

    assert uvutils._check_histories(
        uvc_new.history[:200],
        (
            "Initialized from a UVData object with pyuvdata."
            " UVData history is: " + uvd.history
        )[:200],
    )

    uvc_new.history = uvc2.history

    # The times are different by 9.31322575e-10, which is below than our tolerance on
    # the time array (which is 1ms = 1.1574074074074074e-08) but it leads to differences
    # in the lsts of 5.86770454e-09 which are larger than our tolerance
    # (which is 1mas = 4.84813681109536e-09)
    # I'm not sure why the times are different at all, there must have been some loss
    # of precision in the processing pipeline.
    assert uvc_new._time_array == uvc2._time_array
    uvc_new.time_array = uvc2.time_array
    uvc_new.set_lsts_from_time_array()

    assert uvc_new == uvc2


def test_init_from_uvdata_basic_errors(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data
    uvc._set_flex_spw()
    # uvc has a time_range which it shouldn't really have because Ntimes > 1,
    # but that requirement is not enforced. Set it to None for this test
    uvc.time_range = None

    with pytest.raises(ValueError, match="uvdata must be a UVData object."):
        UVCal.initialize_from_uvdata(uvc, uvc.gain_convention, uvc.cal_style)

    with pytest.raises(ValueError, match="cal_type must be either 'gain' or 'delay'."):
        UVCal.initialize_from_uvdata(
            uvd, uvc.gain_convention, uvc.cal_style, cal_type="unknown"
        )

    with pytest.raises(
        ValueError,
        match="If cal_style is 'sky', ref_antenna_name and sky_catalog must be "
        "provided.",
    ):
        UVCal.initialize_from_uvdata(uvd, uvc.gain_convention, "sky")

    uvd.polarization_array = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError, match="you must set jones_array."):
        UVCal.initialize_from_uvdata(uvd, uvc.gain_convention, uvc.cal_style)


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
            uvc.gain_convention,
            uvc.cal_style,
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
            uvc.gain_convention,
            uvc.cal_style,
            cal_type="delay",
            freq_range=np.asarray([[1e8, 1.2e8], [1.3e8, 1.5e8]]),
        )
