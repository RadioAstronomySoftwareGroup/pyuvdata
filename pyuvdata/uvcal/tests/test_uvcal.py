# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvcal object.

"""
import pytest
import os
import copy
import numpy as np

from pyuvdata import UVCal
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values",
    "ignore:antenna_positions is not set. Using known values",
)


@pytest.fixture(scope="function")
def uvcal_data():
    """Set up some uvcal iter tests."""
    required_parameters = [
        "_Nfreqs",
        "_Njones",
        "_Ntimes",
        "_Nspws",
        "_Nants_data",
        "_Nants_telescope",
        "_antenna_names",
        "_antenna_numbers",
        "_ant_array",
        "_telescope_name",
        "_freq_array",
        "_channel_width",
        "_spw_array",
        "_jones_array",
        "_time_array",
        "_integration_time",
        "_gain_convention",
        "_flag_array",
        "_quality_array",
        "_cal_type",
        "_cal_style",
        "_x_orientation",
        "_history",
    ]

    required_properties = [
        "Nfreqs",
        "Njones",
        "Ntimes",
        "Nspws",
        "Nants_data",
        "Nants_telescope",
        "antenna_names",
        "antenna_numbers",
        "ant_array",
        "telescope_name",
        "freq_array",
        "channel_width",
        "spw_array",
        "jones_array",
        "time_array",
        "integration_time",
        "gain_convention",
        "flag_array",
        "quality_array",
        "cal_type",
        "cal_style",
        "x_orientation",
        "history",
    ]

    extra_parameters = [
        "_telescope_location",
        "_antenna_positions",
        "_lst_array",
        "_gain_array",
        "_delay_array",
        "_sky_field",
        "_sky_catalog",
        "_ref_antenna_name",
        "_Nsources",
        "_baseline_range",
        "_diffuse_model",
        "_input_flag_array",
        "_time_range",
        "_freq_range",
        "_observer",
        "_git_origin_cal",
        "_git_hash_cal",
        "_total_quality_array",
        "_extra_keywords",
        "_gain_scale",
    ]

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
        "observer",
        "git_origin_cal",
        "git_hash_cal",
        "total_quality_array",
        "extra_keywords",
        "gain_scale",
    ]

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


@pytest.fixture(scope="function")
def gain_data():
    """Initialize for some basic uvcal tests."""
    gain_object = UVCal()
    gainfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    gain_object.read_calfits(gainfile)

    yield gain_object

    del gain_object


@pytest.fixture(scope="function")
def delay_data(tmp_path):
    """Initialization for some basic uvcal tests."""

    delay_object = UVCal()
    delayfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits")

    # add an input flag array to the file to test for that.
    write_file = str(tmp_path / "outtest_input_flags.fits")
    uv_in = UVCal()
    uv_in.read_calfits(delayfile)
    uv_in.input_flag_array = np.zeros(
        uv_in._input_flag_array.expected_shape(uv_in), dtype=bool
    )
    uv_in.write_calfits(write_file, clobber=True)
    delay_object.read_calfits(write_file)

    # yield the data for testing, then del after tests finish
    yield delay_object

    del delay_object


def test_parameter_iter(uvcal_data):
    """Test expected parameters."""
    (
        uv_cal_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    ) = uvcal_data
    all_params = []
    for prop in uv_cal_object:
        all_params.append(prop)
    for a in required_parameters + extra_parameters:
        assert a in all_params, (
            "expected attribute " + a + " not returned in object iterator"
        )


def test_required_parameter_iter(uvcal_data):
    """Test expected required parameters."""
    (
        uv_cal_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    ) = uvcal_data
    # at first it's a metadata_only object, so need to modify required_parameters
    required = []
    for prop in uv_cal_object.required():
        required.append(prop)
    expected_required = copy.copy(required_parameters)
    expected_required.remove("_quality_array")
    expected_required.remove("_flag_array")
    for a in expected_required:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )

    uv_cal_object.quality_array = 1
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
    (
        uv_cal_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    ) = uvcal_data
    expected_parameters = required_parameters + extra_parameters
    attributes = [i for i in uv_cal_object.__dict__.keys() if i[0] == "_"]
    for a in attributes:
        assert a in expected_parameters, "unexpected parameter " + a + " found in UVCal"


def test_unexpected_attributes(uvcal_data):
    """Test for extra attributes."""
    (
        uv_cal_object,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    ) = uvcal_data
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
        other_properties,
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


def test_check_warnings(gain_data):
    """Test that parameter checks run properly"""
    gain_data.telescope_location = None
    gain_data.lst_array = None
    gain_data.antenna_positions = None

    with uvtest.check_warnings(
        DeprecationWarning,
        [
            "The telescope_location is not set. It will be a required "
            "parameter starting in pyuvdata version 2.3",
            "The antenna_positions parameter is not set. It will be a required "
            "parameter starting in pyuvdata version 2.3",
            "The lst_array is not set. It will be a required "
            "parameter starting in pyuvdata version 2.3",
        ],
    ):
        assert gain_data.check()


def test_unknown_telescopes(gain_data, tmp_path):
    calobj = gain_data
    calobj.telescope_name = "foo"
    calobj.telescope_location = None
    calobj.lst_array = None
    calobj.antenna_positions = None

    write_file_calfits = str(tmp_path / "test.calfits")
    deprecation_messages = [
        "The telescope_location is not set. It will be a required "
        "parameter starting in pyuvdata version 2.3",
        "The antenna_positions parameter is not set. It will be a required "
        "parameter starting in pyuvdata version 2.3",
        "The lst_array is not set. It will be a required "
        "parameter starting in pyuvdata version 2.3",
    ]

    with uvtest.check_warnings(DeprecationWarning, match=deprecation_messages):
        calobj.write_calfits(write_file_calfits, clobber=True)

    calobj2 = UVCal()
    with uvtest.check_warnings(
        [UserWarning] + [DeprecationWarning] * 3,
        match=["Telescope foo is not in known_telescopes"] + deprecation_messages,
    ):
        calobj2.read_calfits(write_file_calfits)


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
    with uvtest.check_warnings(
        DeprecationWarning,
        match="`set_gain` is deprecated, and will be removed in "
        "pyuvdata version 2.2. Use `_set_gain` instead.",
    ):
        gain_data.set_gain()


def test_set_delay(gain_data):
    gain_data._set_delay()
    assert gain_data._delay_array.required
    assert not gain_data._gain_array.required
    assert gain_data._gain_array.form == gain_data._flag_array.form
    assert gain_data._delay_array.form == gain_data._quality_array.form
    with uvtest.check_warnings(
        DeprecationWarning,
        match="`set_delay` is deprecated, and will be removed in "
        "pyuvdata version 2.2. Use `_set_delay` instead.",
    ):
        gain_data.set_delay()


def test_set_unknown(gain_data):
    gain_data._set_unknown_cal_type()
    assert not gain_data._delay_array.required
    assert not gain_data._gain_array.required
    assert gain_data._gain_array.form == gain_data._flag_array.form
    assert gain_data._gain_array.form == gain_data._quality_array.form

    with uvtest.check_warnings(
        DeprecationWarning,
        match="`set_unknown_cal_type` is deprecated, and will be removed in "
        "pyuvdata version 2.2. Use `_set_unknown_cal_type` instead.",
    ):
        gain_data.set_unknown_cal_type()


def test_set_sky(gain_data):
    gain_data._set_sky()
    assert gain_data._sky_field.required
    assert gain_data._sky_catalog.required
    assert gain_data._ref_antenna_name.required

    with uvtest.check_warnings(
        DeprecationWarning,
        match="`set_sky` is deprecated, and will be removed in "
        "pyuvdata version 2.2. Use `_set_sky` instead.",
    ):
        gain_data.set_sky()


def test_set_redundant(gain_data):
    gain_data._set_redundant()
    assert not gain_data._sky_field.required
    assert not gain_data._sky_catalog.required
    assert not gain_data._ref_antenna_name.required

    with uvtest.check_warnings(
        DeprecationWarning,
        match="`set_redundant` is deprecated, and will be removed in "
        "pyuvdata version 2.2. Use `_set_redundant` instead.",
    ):
        gain_data.set_redundant()


def test_convert_filetype(gain_data):
    # error testing
    with pytest.raises(ValueError, match="filetype must be calfits."):
        gain_data._convert_to_filetype("uvfits")


def test_convert_to_gain(gain_data, delay_data):
    conventions = ["minus", "plus"]
    for c in conventions:
        gain_data.new_object = delay_data.copy()

        gain_data.new_object.convert_to_gain(delay_convention=c)
        assert np.isclose(
            np.max(np.absolute(gain_data.new_object.gain_array)),
            1.0,
            rtol=gain_data.new_object._gain_array.tols[0],
            atol=gain_data.new_object._gain_array.tols[1],
        )
        assert np.isclose(
            np.min(np.absolute(gain_data.new_object.gain_array)),
            1.0,
            rtol=gain_data.new_object._gain_array.tols[0],
            atol=gain_data.new_object._gain_array.tols[1],
        )

        if c == "minus":
            conv = -1
        else:
            conv = 1
        assert np.allclose(
            np.angle(gain_data.new_object.gain_array[:, :, 10, :, :]) % (2 * np.pi),
            (
                conv
                * 2
                * np.pi
                * delay_data.delay_array[:, :, 0, :, :]
                * delay_data.freq_array[0, 10]
            )
            % (2 * np.pi),
            rtol=gain_data.new_object._gain_array.tols[0],
            atol=gain_data.new_object._gain_array.tols[1],
        )
        assert np.allclose(
            delay_data.quality_array,
            gain_data.new_object.quality_array[:, :, 10, :, :],
            rtol=gain_data.new_object._quality_array.tols[0],
            atol=gain_data.new_object._quality_array.tols[1],
        )

        assert gain_data.new_object.history == (
            delay_data.history + "  Converted from delays to gains using pyuvdata."
        )

    # test a file with a total_quality_array
    gain_data.new_object = delay_data.copy()
    tqa_size = gain_data.new_object.delay_array.shape[1:]
    gain_data.new_object.total_quality_array = np.ones(tqa_size)
    gain_data.new_object.convert_to_gain(delay_convention="minus")
    assert np.isclose(
        np.max(np.absolute(gain_data.new_object.gain_array)),
        1.0,
        rtol=gain_data.new_object._gain_array.tols[0],
        atol=gain_data.new_object._gain_array.tols[1],
    )
    assert np.isclose(
        np.min(np.absolute(gain_data.new_object.gain_array)),
        1.0,
        rtol=gain_data.new_object._gain_array.tols[0],
        atol=gain_data.new_object._gain_array.tols[1],
    )
    assert np.allclose(
        np.angle(gain_data.new_object.gain_array[:, :, 10, :, :]) % (2 * np.pi),
        (
            -1
            * 2
            * np.pi
            * delay_data.delay_array[:, :, 0, :, :]
            * delay_data.freq_array[0, 10]
        )
        % (2 * np.pi),
        rtol=gain_data.new_object._gain_array.tols[0],
        atol=gain_data.new_object._gain_array.tols[1],
    )
    assert np.allclose(
        delay_data.quality_array,
        gain_data.new_object.quality_array[:, :, 10, :, :],
        rtol=gain_data.new_object._quality_array.tols[0],
        atol=gain_data.new_object._quality_array.tols[1],
    )

    assert gain_data.new_object.history == (
        delay_data.history + "  Converted from delays to gains using pyuvdata."
    )

    # error testing
    pytest.raises(ValueError, delay_data.convert_to_gain, delay_convention="bogus")
    pytest.raises(ValueError, gain_data.convert_to_gain)
    gain_data._set_unknown_cal_type()
    pytest.raises(ValueError, gain_data.convert_to_gain)


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_antennas(caltype, gain_data, delay_data, tmp_path):
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
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
        old_history + "  Downselected to " "specific antennas using pyuvdata.",
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
        match=f"Antenna number {np.max(calobj.ant_array)+1} "
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
    new_gain_object = UVCal()
    new_gain_object.read_calfits(write_file_calfits)
    assert calobj2 == new_gain_object

    # check that total_quality_array is handled properly when present
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    with uvtest.check_warnings(
        UserWarning, match="Cannot preserve total_quality_array",
    ):
        calobj.select(antenna_names=ant_names, inplace=True)
    assert calobj.total_quality_array is None


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_times(caltype, gain_data, delay_data, tmp_path):
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
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
        old_history + "  Downselected to " "specific times using pyuvdata.",
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
    with uvtest.check_warnings(
        UserWarning, match="Selected times are not evenly spaced",
    ):
        calobj2.select(times=calobj2.time_array[[0, 2, 3]])
    pytest.raises(ValueError, calobj2.write_calfits, write_file_calfits)


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_frequencies(caltype, gain_data, delay_data, tmp_path):
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    old_history = calobj.history
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
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        calobj2.history,
    )

    write_file_calfits = str(tmp_path / "select_test.calfits")
    # test writing calfits with only one frequency
    calobj2 = calobj.copy()
    freqs_to_keep = calobj.freq_array[0, 5]
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
    with uvtest.check_warnings(
        UserWarning, match="Selected frequencies are not evenly spaced",
    ):
        calobj2.select(frequencies=calobj2.freq_array[0, [0, 5, 6]])
    pytest.raises(ValueError, calobj2.write_calfits, write_file_calfits)


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_freq_chans(caltype, gain_data, delay_data):
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    old_history = calobj.history
    chans_to_keep = np.arange(4, 8)

    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )
    calobj2.total_quality_array = np.zeros(
        calobj2._total_quality_array.expected_shape(calobj2)
    )

    calobj2.select(freq_chans=chans_to_keep)

    assert len(chans_to_keep) == calobj2.Nfreqs
    for chan in chans_to_keep:
        assert calobj.freq_array[0, chan] in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in calobj.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        calobj2.history,
    )

    # Test selecting both channels and frequencies
    freqs_to_keep = calobj.freq_array[0, np.arange(7, 10)]  # Overlaps with chans
    all_chans_to_keep = np.arange(4, 10)

    calobj2 = calobj.copy()
    calobj2.select(frequencies=freqs_to_keep, freq_chans=chans_to_keep)

    assert len(all_chans_to_keep) == calobj2.Nfreqs
    for chan in all_chans_to_keep:
        assert calobj.freq_array[0, chan] in calobj2.freq_array
    for f in np.unique(calobj2.freq_array):
        assert f in calobj.freq_array[0, all_chans_to_keep]


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select_polarizations(caltype, gain_data, delay_data):
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    # add more jones terms to allow for better testing of selections
    while calobj.Njones < 4:
        new_jones = np.min(calobj.jones_array) - 1
        calobj.jones_array = np.append(calobj.jones_array, new_jones)
        calobj.Njones += 1
        calobj.flag_array = np.concatenate(
            (calobj.flag_array, calobj.flag_array[:, :, :, :, [-1]],), axis=4,
        )
        if calobj.input_flag_array is not None:
            calobj.input_flag_array = np.concatenate(
                (calobj.input_flag_array, calobj.input_flag_array[:, :, :, :, [-1]],),
                axis=4,
            )
        if caltype == "gain":
            calobj.gain_array = np.concatenate(
                (calobj.gain_array, calobj.gain_array[:, :, :, :, [-1]],), axis=4,
            )
        else:
            delay_data.delay_array = np.concatenate(
                (delay_data.delay_array, delay_data.delay_array[:, :, :, :, [-1]],),
                axis=4,
            )
        calobj.quality_array = np.concatenate(
            (calobj.quality_array, calobj.quality_array[:, :, :, :, [-1]],), axis=4,
        )
    # add dummy total_quality_array
    calobj.total_quality_array = np.zeros(
        calobj._total_quality_array.expected_shape(calobj)
    )

    assert calobj.check()
    calobj2 = calobj.copy()

    old_history = calobj.history
    jones_to_keep = [-5, -6]

    calobj2.select(jones=jones_to_keep)

    assert len(jones_to_keep) == calobj2.Njones
    for j in jones_to_keep:
        assert j in calobj2.jones_array
    for j in np.unique(calobj2.jones_array):
        assert j in jones_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific jones polarization terms "
        "using pyuvdata.",
        calobj2.history,
    )

    # check for errors associated with polarizations not included in data
    pytest.raises(ValueError, calobj2.select, jones=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    with uvtest.check_warnings(
        UserWarning, match="Selected jones polarization terms are not evenly spaced",
    ):
        calobj.select(jones=calobj.jones_array[[0, 1, 3]])
    write_file_calfits = os.path.join(DATA_PATH, "test/select_test.calfits")
    pytest.raises(ValueError, calobj.write_calfits, write_file_calfits)


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_select(caltype, gain_data, delay_data):
    # now test selecting along all axes at once
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    old_history = calobj.history

    ants_to_keep = np.array([10, 89, 43, 9, 80, 96, 64])
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


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_add_antennas(caltype, gain_data, delay_data):
    """Test adding antennas between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    calobj.select(antenna_nums=ants1)
    calobj2.select(antenna_nums=ants2)
    calobj += calobj2
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
    with uvtest.check_warnings(UserWarning, match="Total quality array detected"):
        calobj.__iadd__(calobj2)
    assert calobj.total_quality_array is None

    if caltype == "delay":
        # test for when input_flag_array is present in first file but not second
        calobj.select(antenna_nums=ants1)
        ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        tot_ifa = np.concatenate([ifa, ifa2], axis=0)
        calobj.input_flag_array = ifa
        calobj2.input_flag_array = None
        calobj += calobj2
        assert np.allclose(calobj.input_flag_array, tot_ifa)

        # test for when input_flag_array is present in second file but not first
        calobj.select(antenna_nums=ants1)
        ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        tot_ifa = np.concatenate([ifa, ifa2], axis=0)
        calobj.input_flag_array = None
        calobj2.input_flag_array = ifa2
        calobj += calobj2
        assert np.allclose(calobj.input_flag_array, tot_ifa)

    # Out of order - antennas
    calobj = calobj_full.copy()
    calobj2 = calobj.copy()
    calobj.select(antenna_nums=ants2)
    calobj2.select(antenna_nums=ants1)
    calobj += calobj2
    calobj.history = calobj_full.history
    assert calobj == calobj_full


def test_add_frequencies(gain_data):
    """Test adding frequencies between two UVCal objects"""
    # don't test on delays because there's no freq axis for the delay array
    calobj = gain_data
    calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    freqs1 = calobj.freq_array[0, np.arange(0, calobj.Nfreqs // 2)]
    freqs2 = calobj.freq_array[0, np.arange(calobj.Nfreqs // 2, calobj.Nfreqs)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)
    calobj += calobj2
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
    calobj.select(frequencies=freqs1)
    tqa = np.ones(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.zeros(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = tqa
    calobj += calobj2
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when total_quality_array is present in second file but not first
    calobj.select(frequencies=freqs1)
    tqa = np.zeros(calobj._total_quality_array.expected_shape(calobj))
    tqa2 = np.ones(calobj2._total_quality_array.expected_shape(calobj2))
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = None
    calobj2.total_quality_array = tqa2
    calobj += calobj2
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
    tot_tqa = np.concatenate([tqa, tqa2], axis=1)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    calobj += calobj2
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    # test for when input_flag_array is present in first file but not second
    calobj = calobj_full.copy()
    calobj.input_flag_array = np.zeros(
        calobj._input_flag_array.expected_shape(calobj), dtype=bool
    )
    calobj2 = calobj.copy()
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)
    ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
    ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(np.bool_)
    tot_ifa = np.concatenate([ifa, ifa2], axis=2)
    calobj.input_flag_array = ifa
    calobj2.input_flag_array = None
    calobj += calobj2
    assert np.allclose(calobj.input_flag_array, tot_ifa)

    # test for when input_flag_array is present in second file but not first
    calobj.select(frequencies=freqs1)
    ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
    ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(np.bool_)
    tot_ifa = np.concatenate([ifa, ifa2], axis=2)
    calobj.input_flag_array = None
    calobj2.input_flag_array = ifa2
    calobj += calobj2
    assert np.allclose(calobj.input_flag_array, tot_ifa)

    # Out of order - freqs
    calobj = calobj_full.copy()
    calobj2 = calobj_full.copy()
    calobj.select(frequencies=freqs2)
    calobj2.select(frequencies=freqs1)
    calobj += calobj2
    calobj.history = calobj_full.history
    assert calobj == calobj_full


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_add_times(caltype, gain_data, delay_data):
    """Test adding times between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    calobj_full = calobj.copy()
    n_times2 = calobj.Ntimes // 2
    times1 = calobj.time_array[:n_times2]
    times2 = calobj.time_array[n_times2:]
    calobj.select(times=times1)
    calobj2.select(times=times2)
    calobj += calobj2
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
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = tqa
    calobj += calobj2
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
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = None
    calobj2.total_quality_array = tqa2
    calobj += calobj2
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
    tot_tqa = np.concatenate([tqa, tqa2], axis=2)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    calobj += calobj2
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    if caltype == "delay":
        # test for when input_flag_array is present in first file but not second
        calobj.select(times=times1)
        ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        tot_ifa = np.concatenate([ifa, ifa2], axis=3)
        calobj.input_flag_array = ifa
        calobj2.input_flag_array = None
        calobj += calobj2
        assert np.allclose(calobj.input_flag_array, tot_ifa)

        # test for when input_flag_array is present in second file but not first
        calobj.select(times=times1)
        ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        tot_ifa = np.concatenate([ifa, ifa2], axis=3)
        calobj.input_flag_array = None
        calobj2.input_flag_array = ifa2
        calobj += calobj2
        assert np.allclose(calobj.input_flag_array, tot_ifa)

    # Out of order - times
    calobj = calobj_full.copy()
    calobj2 = calobj.copy()
    calobj.select(times=times2)
    calobj2.select(times=times1)
    calobj += calobj2
    calobj.history = calobj_full.history
    assert calobj == calobj_full


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_add_jones(caltype, gain_data, delay_data):
    """Test adding Jones axes between two UVCal objects"""
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    calobj_original = calobj.copy()
    # artificially change the Jones value to permit addition
    calobj2.jones_array[0] = -6
    calobj += calobj2

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
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    calobj.total_quality_array = tqa
    calobj += calobj2
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
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    calobj2.total_quality_array = tqa2
    calobj += calobj2
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
    tot_tqa = np.concatenate([tqa, tqa2], axis=3)
    calobj.total_quality_array = tqa
    calobj2.total_quality_array = tqa2
    calobj += calobj2
    assert np.allclose(
        calobj.total_quality_array,
        tot_tqa,
        rtol=calobj._total_quality_array.tols[0],
        atol=calobj._total_quality_array.tols[1],
    )

    if caltype == "delay":
        # test for when input_flag_array is present in first file but not second
        calobj = calobj_original.copy()
        ifa = np.zeros(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.ones(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        tot_ifa = np.concatenate([ifa, ifa2], axis=4)
        calobj.input_flag_array = ifa
        calobj2.input_flag_array = None
        calobj += calobj2
        assert np.allclose(calobj.input_flag_array, tot_ifa)

        # test for when input_flag_array is present in second file but not first
        calobj = calobj_original.copy()
        ifa = np.ones(calobj._input_flag_array.expected_shape(calobj)).astype(np.bool_)
        ifa2 = np.zeros(calobj2._input_flag_array.expected_shape(calobj2)).astype(
            np.bool_
        )
        tot_ifa = np.concatenate([ifa, ifa2], axis=4)
        calobj.input_flag_array = None
        calobj2.input_flag_array = ifa2
        calobj += calobj2
        assert np.allclose(calobj.input_flag_array, tot_ifa)

    # Out of order - jones
    calobj = calobj_original.copy()
    calobj2 = calobj_original.copy()
    calobj.jones_array[0] = -6
    calobj += calobj2
    calobj2 = calobj.copy()
    calobj.select(jones=-5)
    calobj.history = calobj_original.history
    assert calobj == calobj_original
    calobj2.select(jones=-6)
    calobj2.jones_array[:] = -5
    calobj2.history = calobj_original.history
    assert calobj2 == calobj_original


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_add(caltype, gain_data, delay_data):
    """Test miscellaneous aspects of add method"""
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    # test not-in-place addition
    calobj_original = calobj.copy()
    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    calobj.select(antenna_nums=ants1)
    calobj2.select(antenna_nums=ants2)
    calobj_add = calobj + calobj2
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
    calobj += calobj2

    additional_history = "Some random history string"
    assert uvutils._check_histories(
        calobj_original.history + " Combined data along antenna axis "
        "using pyuvdata. " + additional_history,
        calobj.history,
    )


def test_add_multiple_axes(gain_data):
    """Test addition along multiple axes"""
    calobj = gain_data
    calobj2 = calobj.copy()

    ants1 = np.array([9, 10, 20, 22, 31, 43, 53, 64, 65, 72])
    ants2 = np.array([80, 81, 88, 89, 96, 97, 104, 105, 112])
    freqs1 = calobj.freq_array[0, np.arange(0, 5)]
    freqs2 = calobj.freq_array[0, np.arange(5, 10)]
    n_times2 = calobj.Ntimes // 2
    times1 = calobj.time_array[:n_times2]
    times2 = calobj.time_array[n_times2:]
    # artificially change the Jones value to permit addition
    calobj2.jones_array[0] = -6

    # perform select
    calobj.select(antenna_nums=ants1, frequencies=freqs1, times=times1)
    calobj2.select(antenna_nums=ants2, frequencies=freqs2, times=times2)

    calobj += calobj2

    # check resulting dimensionality
    assert len(calobj.ant_array) == 19
    assert len(calobj.freq_array[0, :]) == 10
    assert len(calobj.time_array) == calobj.Ntimes
    assert len(calobj.jones_array) == 2


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_add_errors(caltype, gain_data, delay_data):
    """Test behavior that will raise errors"""
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
        calobj2 = calobj.copy()

    # test addition of two identical objects
    with pytest.raises(
        ValueError, match="These objects have overlapping data and cannot be combined."
    ):
        calobj.__add__(calobj2)

    # test addition of UVCal and non-UVCal object (empty list)
    with pytest.raises(
        ValueError, match="Only UVCal ",
    ):
        calobj.__add__([])

    # test compatibility param mismatch
    calobj2.telescope_name = "PAPER"
    with pytest.raises(
        ValueError, match="Parameter telescope_name does not match",
    ):
        calobj.__add__(calobj2)


def test_jones_warning(gain_data):
    """Test having non-contiguous Jones elements"""
    calobj = gain_data
    calobj2 = calobj.copy()

    calobj2.jones_array[0] = -6
    calobj += calobj2
    calobj2.jones_array[0] = -8
    with uvtest.check_warnings(
        UserWarning, match="Combined Jones elements",
    ):
        calobj.__iadd__(calobj2)
    assert sorted(calobj.jones_array) == [-8, -6, -5]


def test_frequency_warnings(gain_data):
    """Test having uneven or non-contiguous frequencies"""
    # test having unevenly spaced frequency separations
    calobj = gain_data
    calobj2 = calobj.copy()

    go1 = calobj.copy()
    go2 = calobj2.copy()
    freqs1 = calobj.freq_array[0, np.arange(0, 5)]
    freqs2 = calobj2.freq_array[0, np.arange(5, 10)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    # change the last frequency bin to be smaller than the others
    df = calobj2.freq_array[0, -1] - calobj2.freq_array[0, -2]
    calobj2.freq_array[0, -1] = calobj2.freq_array[0, -2] + df / 2
    with uvtest.check_warnings(
        UserWarning, match="Combined frequencies are not evenly spaced"
    ):
        calobj.__iadd__(calobj2)

    assert len(calobj.freq_array[0, :]) == calobj.Nfreqs

    # now check having "non-contiguous" frequencies
    calobj = go1.copy()
    calobj2 = go2.copy()
    freqs1 = calobj.freq_array[0, np.arange(0, 5)]
    freqs2 = calobj2.freq_array[0, np.arange(5, 10)]
    calobj.select(frequencies=freqs1)
    calobj2.select(frequencies=freqs2)

    # artificially space out frequencies
    calobj.freq_array[0, :] *= 10
    calobj2.freq_array[0, :] *= 10
    with uvtest.check_warnings(
        UserWarning, match="Combined frequencies are not contiguous"
    ):
        calobj.__iadd__(calobj2)

    freqs1 *= 10
    freqs2 *= 10
    freqs = np.concatenate([freqs1, freqs2])
    assert np.allclose(
        calobj.freq_array[0, :],
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
    freqs1 = calobj.freq_array[0, np.arange(0, 5)]
    freqs2 = calobj2.freq_array[0, np.arange(5, 10)]
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


@pytest.mark.parametrize("caltype", ["gain", "delay"])
def test_multi_files(caltype, gain_data, delay_data, tmp_path):
    """Test read function when multiple files are included"""
    if caltype == "gain":
        calobj = gain_data
        calobj2 = calobj.copy()
    else:
        calobj = delay_data
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
    calobj.read_calfits([f1, f2])
    assert uvutils._check_histories(
        calobj_full.history + "  Downselected to specific times"
        " using pyuvdata. Combined data "
        "along time axis using pyuvdata.",
        calobj.history,
    )
    calobj.history = calobj_full.history
    assert calobj == calobj_full

    # check metadata only read
    calobj.read_calfits([f1, f2], read_data=False)
    calobj_full_metadata_only = calobj_full.copy(metadata_only=True)

    calobj.history = calobj_full_metadata_only.history
    assert calobj == calobj_full_metadata_only


def test_uvcal_get_methods():
    # load data
    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits"))

    # test get methods: add in a known value and make sure it is returned
    key = (10, "Jee")
    uvc.gain_array[1] = 0.0
    d, f, q = uvc.get_gains(key), uvc.get_flags(key), uvc.get_quality(key)

    # test shapes
    assert np.all(np.isclose(d, 0.0))
    assert d.shape == (uvc.Nfreqs, uvc.Ntimes)
    assert f.shape == (uvc.Nfreqs, uvc.Ntimes)
    assert q.shape == (uvc.Nfreqs, uvc.Ntimes)

    # test against by-hand indexing
    np.testing.assert_array_almost_equal(
        d,
        uvc.gain_array[
            uvc.ant_array.tolist().index(10),
            0,
            :,
            :,
            uvc.jones_array.tolist().index(-5),
        ],
    )

    # test variable key input
    d2 = uvc.get_gains(*key)
    np.testing.assert_array_almost_equal(d, d2)
    d2 = uvc.get_gains(key[0])
    np.testing.assert_array_almost_equal(d, d2)
    d2 = uvc.get_gains(key[:1])
    np.testing.assert_array_almost_equal(d, d2)
    d2 = uvc.get_gains(10, -5)
    np.testing.assert_array_almost_equal(d, d2)
    d2 = uvc.get_gains(10, "x")
    np.testing.assert_array_almost_equal(d, d2)

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


def test_write_read_optional_attrs(tmp_path):
    # read a test file
    cal_in = UVCal()
    testfile = os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits")
    cal_in.read_calfits(testfile)

    # set some optional parameters
    cal_in.gain_scale = "Jy"
    cal_in.sky_field = "GLEAM"

    # write
    write_file_calfits = str(tmp_path / "test.calfits")
    cal_in.write_calfits(write_file_calfits, clobber=True)

    # read and compare
    cal_in2 = UVCal()
    cal_in2.read_calfits(write_file_calfits)
    assert cal_in == cal_in2


@pytest.mark.parametrize("caltype", ["gain", "delay", "unknown", None])
def test_copy(gain_data, delay_data, caltype):
    """Test the copy method"""
    if caltype == "gain":
        uv_object = gain_data
    elif caltype == "delay":
        uv_object = delay_data
    else:
        uv_object = gain_data
        uv_object._set_unknown_cal_type()
        uv_object.cal_type = caltype

    uv_object_copy = uv_object.copy()
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
    gain_data2 = gain_data.copy()
    gain_data2.antenna_positions = None

    write_file_calfits = str(tmp_path / "test.calfits")
    with uvtest.check_warnings(
        DeprecationWarning,
        match="The antenna_positions parameter is not set. It will be a required "
        "parameter starting in pyuvdata version 2.3",
    ):
        gain_data2.write_calfits(write_file_calfits)

    with uvtest.check_warnings(
        UserWarning, "antenna_positions is not set. Using known values for HERA."
    ):
        gain_data2.read_calfits(write_file_calfits)

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
        print(gain_data.antenna_names)
        print(gain_data.antenna_numbers)

    # remove the antenna_positions to test matching them on read
    gain_data2 = gain_data.copy()
    gain_data2.antenna_positions = None

    write_file_calfits = str(tmp_path / "test.calfits")
    with uvtest.check_warnings(
        DeprecationWarning,
        match="The antenna_positions parameter is not set. It will be a required "
        "parameter starting in pyuvdata version 2.3",
    ):
        gain_data2.write_calfits(write_file_calfits)

    with uvtest.check_warnings(
        [UserWarning, DeprecationWarning],
        match=[
            "Not all antennas have positions in the telescope object. "
            "Not setting antenna_positions.",
            "The antenna_positions parameter is not set. It will be a required "
            "parameter starting in pyuvdata version 2.3",
        ],
    ):
        gain_data2.read_calfits(write_file_calfits)

    assert gain_data2.antenna_positions is None
