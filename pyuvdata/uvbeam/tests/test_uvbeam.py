# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvbeam object.

"""
import copy
import os
import re
import warnings
from collections import namedtuple

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Angle
from astropy.io import fits

import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata import UVBeam, _uvbeam
from pyuvdata.data import DATA_PATH
from pyuvdata.uvbeam.tests.test_cst_beam import cst_files, cst_yaml_file
from pyuvdata.uvbeam.tests.test_mwa_beam import filename as mwa_beam_file
from pyuvdata.uvbeam.uvbeam import _future_array_shapes_warning

try:
    from astropy_healpix import HEALPix

    healpix_installed = True
except ImportError:
    healpix_installed = False

casa_beamfits = os.path.join(DATA_PATH, "HERABEAM.FITS")


@pytest.fixture(scope="function")
def uvbeam_data():
    """Setup and teardown for basic parameter, property and iterator tests."""
    required_properties = [
        "beam_type",
        "Nfreqs",
        "Naxes_vec",
        "pixel_coordinate_system",
        "freq_array",
        "data_normalization",
        "data_array",
        "bandpass_array",
        "telescope_name",
        "feed_name",
        "feed_version",
        "model_name",
        "model_version",
        "history",
        "antenna_type",
        "future_array_shapes",
    ]
    required_parameters = ["_" + prop for prop in required_properties]

    extra_properties = [
        "Naxes1",
        "Naxes2",
        "Npixels",
        "Nfeeds",
        "Npols",
        "Ncomponents_vec",
        "axis1_array",
        "axis2_array",
        "nside",
        "ordering",
        "pixel_array",
        "feed_array",
        "polarization_array",
        "basis_vector_array",
        "extra_keywords",
        "Nelements",
        "element_coordinate_system",
        "element_location_array",
        "delay_array",
        "x_orientation",
        "gain_array",
        "coupling_matrix",
        "reference_impedance",
        "receiver_temperature_array",
        "loss_array",
        "mismatch_array",
        "s_parameters",
        "filename",
    ]
    extra_parameters = ["_" + prop for prop in extra_properties]

    other_properties = ["pyuvdata_version_str"]

    beam_obj = UVBeam()

    DataHolder = namedtuple(
        "DataHolder",
        [
            "beam_obj",
            "required_parameters",
            "required_properties",
            "extra_parameters",
            "extra_properties",
            "other_properties",
        ],
    )

    uvbeam_data = DataHolder(
        beam_obj,
        required_parameters,
        required_properties,
        extra_parameters,
        extra_properties,
        other_properties,
    )
    # yields the data we need but will continue to the del call after tests
    yield uvbeam_data

    # some post-test object cleanup
    del uvbeam_data

    return


@pytest.fixture(scope="function")
def power_beam_for_adding(cst_power_1freq):
    power_beam = cst_power_1freq

    # generate more frequencies for testing by copying and adding
    new_beam = power_beam.copy()
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    yield power_beam

    del power_beam

    return


@pytest.fixture(scope="function")
def efield_beam_for_adding(cst_efield_1freq):
    # Add feeds
    efield_beam = cst_efield_1freq

    # generate more frequencies for testing by copying and adding
    new_beam = efield_beam.copy()
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    yield efield_beam

    del efield_beam

    return


@pytest.fixture(scope="function")
def cross_power_beam_for_adding(efield_beam_for_adding):
    # generate more polarizations for testing by using efield and keeping cross-pols
    power_beam = efield_beam_for_adding

    # Filter the warning that sometimes happens. Needs to be done this way rather than
    # with uvtest.check_warnings because the warning is not raised on all os types
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Fixing auto polarization power beams"
        )
        power_beam.efield_to_power()

    # generate more frequencies for testing by copying and adding several times
    while power_beam.Nfreqs < 8:
        new_beam = power_beam.copy()
        new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
        power_beam += new_beam

    yield power_beam

    del power_beam

    return


def test_parameter_iter(uvbeam_data):
    """Test expected parameters."""
    all_params = []
    for prop in uvbeam_data.beam_obj:
        all_params.append(prop)
    for a in uvbeam_data.required_parameters + uvbeam_data.extra_parameters:
        assert a in all_params, (
            "expected attribute " + a + " not returned in object iterator"
        )


def test_required_parameter_iter(uvbeam_data):
    """Test expected required parameters."""
    required = []
    for prop in uvbeam_data.beam_obj.required():
        required.append(prop)
    for a in uvbeam_data.required_parameters:
        assert a in required, (
            "expected attribute " + a + " not returned in required iterator"
        )


def test_extra_parameter_iter(uvbeam_data):
    """Test expected optional parameters."""
    extra = []
    for prop in uvbeam_data.beam_obj.extra():
        extra.append(prop)
    for a in uvbeam_data.extra_parameters:
        assert a in extra, "expected attribute " + a + " not returned in extra iterator"


def test_unexpected_parameters(uvbeam_data):
    """Test for extra parameters."""
    expected_parameters = uvbeam_data.required_parameters + uvbeam_data.extra_parameters
    attributes = [i for i in uvbeam_data.beam_obj.__dict__.keys() if i[0] == "_"]
    for a in attributes:
        assert a in expected_parameters, (
            "unexpected parameter " + a + " found in UVBeam"
        )


def test_unexpected_attributes(uvbeam_data):
    """Test for extra attributes."""
    expected_attributes = (
        uvbeam_data.required_properties
        + uvbeam_data.extra_properties
        + uvbeam_data.other_properties
    )
    attributes = [i for i in uvbeam_data.beam_obj.__dict__.keys() if i[0] != "_"]
    for a in attributes:
        assert a in expected_attributes, (
            "unexpected attribute " + a + " found in UVBeam"
        )


def test_properties(uvbeam_data):
    """Test that properties can be get and set properly."""
    prop_dict = dict(
        list(
            zip(
                uvbeam_data.required_properties + uvbeam_data.extra_properties,
                uvbeam_data.required_parameters + uvbeam_data.extra_parameters,
            )
        )
    )
    for k, v in prop_dict.items():
        rand_num = np.random.rand()
        setattr(uvbeam_data.beam_obj, k, rand_num)
        this_param = getattr(uvbeam_data.beam_obj, v)
        try:
            assert rand_num == this_param.value
        except AssertionError:
            print("setting {prop_name} to a random number failed".format(prop_name=k))
            raise


@pytest.mark.parametrize("beam_type", ["efield", "power", "phased_array"])
def test_future_array_shapes(
    beam_type, cst_efield_2freq, cst_power_2freq, phased_array_beam_2freq
):
    if beam_type == "efield":
        beam = cst_efield_2freq
    elif beam_type == "power":
        beam = cst_power_2freq
    elif beam_type == "phased_array":
        beam = phased_array_beam_2freq

    beam2 = beam.copy()

    # test the no-op
    with uvtest.check_warnings(
        DeprecationWarning,
        match="The unset_spw_params parameter is deprecated and has no effect. "
        "This will become an error in version 2.6.",
    ):
        beam.use_future_array_shapes(unset_spw_params=True)

    with uvtest.check_warnings(
        [DeprecationWarning] * 2,
        match=[
            "This method will be removed in version 3.0",
            "The set_spw_params parameter is deprecated and has no effect. "
            "This will become an error in version 2.6.",
        ],
    ):
        beam.use_current_array_shapes(set_spw_params=False)
    beam.check()

    # test the no-op
    with uvtest.check_warnings(
        DeprecationWarning, match="This method will be removed in version 3.0"
    ):
        beam.use_current_array_shapes()

    beam.use_future_array_shapes()
    beam.check()

    assert beam == beam2


@pytest.mark.parametrize("param", ["freq_interp_kind", "spw_array", "Nspws"])
def test_deprecated_params(cst_efield_2freq, param):
    with uvtest.check_warnings(
        DeprecationWarning,
        match=f"The {param} attribute on UVBeam objects is "
        "deprecated and support for it will be removed in version 2.6.",
    ):
        getattr(cst_efield_2freq, param)

    with uvtest.check_warnings(
        DeprecationWarning,
        match=f"The {param} attribute on UVBeam objects is "
        "deprecated and support for it will be removed in version 2.6.",
    ):
        setattr(cst_efield_2freq, param, "foo")

    assert getattr(cst_efield_2freq, param) == "foo"


def test_deprecated_feed_names(cst_efield_2freq):
    cst_efield_2freq.feed_array = np.array(["N", "E"])

    with uvtest.check_warnings(
        DeprecationWarning,
        match="Feed array has values ['N', 'E'] that are deprecated. Values in "
        "feed_array should be lower case. This will become an error in version 2.6",
    ):
        cst_efield_2freq.check()


def test_set_cs_params(cst_efield_2freq):
    """
    Test _set_cs_params.
    """
    efield_beam = cst_efield_2freq
    efield_beam2 = efield_beam.copy()

    efield_beam2._set_cs_params()
    assert efield_beam2 == efield_beam


def test_set_efield(cst_efield_2freq):
    """
    Test _set_efield parameter settings.
    """
    efield_beam = cst_efield_2freq
    efield_beam2 = efield_beam.copy()

    efield_beam2._set_efield()

    assert efield_beam2 == efield_beam


def test_set_power(cst_power_2freq):
    """
    Test _set_power parameter settings.
    """
    power_beam = cst_power_2freq
    power_beam2 = power_beam.copy()

    power_beam2._set_power()

    assert power_beam2 == power_beam


def test_set_antenna_type(cst_efield_2freq):
    """
    Test set_simple and set_phased_array parameter settings.
    """
    efield_beam = cst_efield_2freq
    efield_beam2 = efield_beam.copy()

    efield_beam2._set_simple()

    assert efield_beam2 == efield_beam

    efield_beam._set_phased_array()
    assert efield_beam2 != efield_beam


def test_errors():
    beam_obj = UVBeam()
    with pytest.raises(ValueError, match="filetype must be beamfits"):
        beam_obj._convert_to_filetype("foo")


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_check_auto_power(future_shapes, cst_efield_2freq_cut):
    power_beam = cst_efield_2freq_cut.copy()
    power_beam.efield_to_power()
    if not future_shapes:
        power_beam.use_current_array_shapes()

    power_beam.data_array[..., 0, :, :, :] += power_beam.data_array[..., 2, :, :, :]

    with pytest.raises(
        ValueError,
        match="Some auto polarization power beams have non-real values in "
        "data_array.",
    ):
        power_beam.check(check_auto_power=True)

    with uvtest.check_warnings(
        UserWarning,
        match="Fixing auto polarization power beams to be be real-only, "
        "after some imaginary values were detected in data_array.",
    ):
        power_beam.check(check_auto_power=True, fix_auto_power=True)
    power_beam.check(check_auto_power=True)

    power_beam2 = power_beam.select(polarizations=[-5, -7], inplace=False)
    power_beam2.polarization_array = [-5, -6]
    with uvtest.check_warnings(
        UserWarning,
        match="Fixing auto polarization power beams to be be real-only, "
        "after some imaginary values were detected in data_array.",
    ):
        power_beam2.check(check_auto_power=True, fix_auto_power=True)


def test_check_auto_power_errors(cst_efield_2freq_cut):
    with uvtest.check_warnings(
        UserWarning,
        match="Cannot use _check_auto_power if beam_type is not 'power', or "
        "polarization_array is None.",
    ):
        cst_efield_2freq_cut._check_auto_power()

    with uvtest.check_warnings(
        UserWarning,
        match="Cannot use _fix_autos if beam_type is not 'power', or "
        "polarization_array is None. Leaving data_array untouched.",
    ):
        cst_efield_2freq_cut._fix_auto_power()


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_peak_normalize(future_shapes, beam_type, cst_efield_2freq, cst_power_2freq):
    if beam_type == "efield":
        beam = cst_efield_2freq
    else:
        beam = cst_power_2freq

    if not future_shapes:
        beam.use_current_array_shapes()

    orig_bandpass_array = copy.deepcopy(beam.bandpass_array)
    maxima = np.zeros(beam.Nfreqs)
    for freq_i in range(beam.Nfreqs):
        maxima[freq_i] = np.amax(abs(beam.data_array[..., freq_i, :, :]))

    beam.peak_normalize()
    assert np.amax(abs(beam.data_array)) == 1
    assert np.sum(abs(beam.bandpass_array - orig_bandpass_array * maxima)) == 0
    assert beam.data_normalization == "peak"


def test_peak_normalize_errors(cst_power_2freq):
    cst_power_2freq.data_normalization = "solid_angle"
    with pytest.raises(
        NotImplementedError,
        match="Conversion from solid_angle to peak "
        "normalization is not yet implemented",
    ):
        cst_power_2freq.peak_normalize()


def test_stokes_matrix():
    beam = UVBeam()
    with pytest.raises(ValueError, match="n must be positive integer."):
        beam._stokes_matrix(-2)
    with pytest.raises(ValueError, match="n should lie between 0 and 3."):
        beam._stokes_matrix(5)


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_efield_to_pstokes(
    future_shapes, cst_efield_2freq_cut, cst_efield_2freq_cut_healpix
):
    pstokes_beam = cst_efield_2freq_cut
    pstokes_beam_2 = cst_efield_2freq_cut_healpix

    if not future_shapes:
        pstokes_beam.use_current_array_shapes()
        pstokes_beam_2.use_current_array_shapes()

    # convert to pstokes after interpolating
    beam_return = pstokes_beam_2.efield_to_pstokes(inplace=False)

    # interpolate after converting to pstokes
    pstokes_beam.efield_to_pstokes()
    pstokes_beam.to_healpix()

    pstokes_beam.peak_normalize()
    beam_return.peak_normalize()

    # NOTE:  So far, the following doesn't hold unless the beams are
    # peak_normalized again.
    # This seems to be the fault of interpolation
    assert np.allclose(pstokes_beam.data_array, beam_return.data_array, atol=1e-2)


def test_efield_to_pstokes_error(cst_power_2freq_cut):
    power_beam = cst_power_2freq_cut

    with pytest.raises(ValueError, match="beam_type must be efield."):
        power_beam.efield_to_pstokes()


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize(
    ["future_shapes", "physical_orientation"], [[True, False], [False, True]]
)
def test_efield_to_power(
    future_shapes, physical_orientation, cst_efield_2freq_cut, cst_power_2freq_cut
):
    efield_beam = cst_efield_2freq_cut
    power_beam = cst_power_2freq_cut

    if not future_shapes:
        efield_beam.use_current_array_shapes()
        power_beam.use_current_array_shapes()

    if physical_orientation:
        efield_beam.feed_array = np.array(["e", "n"])
        power_beam.polarization_array = np.array(
            uvutils.polstr2num(["ee", "nn"], x_orientation=power_beam.x_orientation)
        )

    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=False, inplace=False)

    # The values in the beam file only have 4 sig figs, so they don't match precisely
    diff = np.abs(new_power_beam.data_array - power_beam.data_array)
    assert np.max(diff) < 2
    reldiff = diff / power_beam.data_array
    assert np.max(reldiff) < 0.002

    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [0.002, 0]
    power_beam._data_array.tols = tols
    # modify the history to match
    power_beam.history += " Converted from efield to power using pyuvdata."
    assert power_beam == new_power_beam


def test_efield_to_power_1feed(cst_efield_2freq_cut, cst_power_2freq_cut):
    efield_beam = cst_efield_2freq_cut
    efield_beam.select(feeds=["x"])

    power_beam = cst_power_2freq_cut
    power_beam.select(polarizations=["xx"])

    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=True, inplace=False)

    # The values in the beam file only have 4 sig figs, so they don't match precisely
    diff = np.abs(new_power_beam.data_array - power_beam.data_array)
    assert np.max(diff) < 2
    reldiff = diff / power_beam.data_array
    assert np.max(reldiff) < 0.002

    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [0.002, 0]
    power_beam._data_array.tols = tols
    # modify the history to match
    power_beam.history = new_power_beam.history

    assert power_beam == new_power_beam


def test_efield_to_power_nonorthogonal(cst_efield_2freq_cut):
    efield_beam = cst_efield_2freq_cut

    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=False, inplace=False)

    # test with non-orthogonal basis vectors
    # first construct a beam with non-orthogonal basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[0, 1, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, :, :, :] = efield_beam.basis_vector_array[1, :, :, :]
    new_data = np.zeros_like(efield_beam.data_array)
    # drop all the trailing colons in the slicing below
    new_data[0] = np.sqrt(2) * efield_beam.data_array[0]
    new_data[1] = efield_beam.data_array[1] - efield_beam.data_array[0]
    efield_beam2 = efield_beam.copy()
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = efield_beam2.copy()
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2

    if healpix_installed:
        # check that this raises an error if trying to convert to HEALPix:
        with pytest.raises(
            NotImplementedError,
            match="interpolation for input basis vectors that are not aligned to the "
            "native theta/phi coordinate system is not yet supported",
        ):
            efield_beam2.to_healpix(inplace=False)

    # now try a different rotation to non-orthogonal basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, :, :, :] = efield_beam.basis_vector_array[0, :, :, :]
    new_basis_vecs[1, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, 1, :, :] = np.sqrt(0.5)
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, ...] = efield_beam.data_array[0, ...] - efield_beam.data_array[1, ...]
    new_data[1, ...] = np.sqrt(2) * efield_beam.data_array[1, ...]
    efield_beam2 = efield_beam.copy()
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = efield_beam2.copy()
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2


def test_efield_to_power_rotated(cst_efield_2freq_cut):
    efield_beam = cst_efield_2freq_cut

    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=False, inplace=False)

    # now construct a beam with  orthogonal but rotated basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[0, 1, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, 0, :, :] = -1 * np.sqrt(0.5)
    new_basis_vecs[1, 1, :, :] = np.sqrt(0.5)
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, ...] = np.sqrt(0.5) * (
        efield_beam.data_array[0, ...] + efield_beam.data_array[1, ...]
    )
    new_data[1, ...] = np.sqrt(0.5) * (
        -1 * efield_beam.data_array[0, ...] + efield_beam.data_array[1, ...]
    )
    efield_beam2 = efield_beam.copy()
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = efield_beam2.copy()
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.parametrize("future_shapes", [True, False])
def test_efield_to_power_crosspol(future_shapes, cst_efield_2freq_cut, tmp_path):
    efield_beam = cst_efield_2freq_cut

    if not future_shapes:
        efield_beam.use_current_array_shapes()

    # test calculating cross pols
    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=True, inplace=False)
    wh_2pi = np.where(new_power_beam.axis1_array == np.pi / 2.0)[0]
    wh_0 = np.where(new_power_beam.axis1_array == 0)[0]
    assert np.all(
        np.abs(new_power_beam.data_array[..., 0, :, :, wh_0])
        > np.abs(new_power_beam.data_array[..., 2, :, :, wh_0])
    )
    assert np.all(
        np.abs(new_power_beam.data_array[..., 0, :, :, wh_2pi])
        > np.abs(new_power_beam.data_array[..., 2, :, :, wh_2pi])
    )
    # test writing out & reading back in power files (with cross pols which are complex)
    write_file = str(tmp_path / "outtest_beam.fits")
    new_power_beam.write_beamfits(write_file, clobber=True)
    new_power_beam2 = UVBeam()
    new_power_beam2.read_beamfits(write_file, use_future_array_shapes=future_shapes)
    assert new_power_beam == new_power_beam2

    # test keeping basis vectors
    new_power_beam = efield_beam.efield_to_power(
        calc_cross_pols=False, keep_basis_vector=True, inplace=False
    )
    assert np.allclose(new_power_beam.data_array, np.abs(efield_beam.data_array) ** 2)


def test_efield_to_power_errors(cst_efield_2freq_cut, cst_power_2freq_cut):
    efield_beam = cst_efield_2freq_cut
    power_beam = cst_power_2freq_cut

    # test raises error if beam is already a power beam
    with pytest.raises(ValueError, match="beam_type must be efield"):
        power_beam.efield_to_power()

    # test raises error if input efield beam has Naxes_vec=3
    efield_beam.Naxes_vec = 3
    with pytest.raises(
        ValueError,
        match="Conversion to power with 3-vector efields is not currently supported",
    ):
        efield_beam.efield_to_power()


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("antenna_type", ["simple", "phased_array"])
def test_freq_interpolation(
    future_shapes, antenna_type, cst_power_2freq, phased_array_beam_2freq
):
    if antenna_type == "simple":
        beam = cst_power_2freq
    else:
        beam = phased_array_beam_2freq

    if not future_shapes:
        beam.use_current_array_shapes()

    # test frequency interpolation returns data arrays for small and large tolerances
    freq_orig_vals = np.array([123e6, 150e6])
    need_coupling = False
    if antenna_type == "phased_array":
        need_coupling = True
    interp_arrays = beam.interp(
        freq_array=freq_orig_vals,
        freq_interp_tol=0.0,
        freq_interp_kind="linear",
        return_bandpass=True,
        return_coupling=need_coupling,
    )
    if antenna_type == "simple":
        interp_data, interp_basis_vector, interp_bandpass = interp_arrays
    else:
        (interp_data, interp_basis_vector, interp_bandpass, interp_coupling_matrix) = (
            interp_arrays
        )
    assert isinstance(interp_data, np.ndarray)
    assert isinstance(interp_bandpass, np.ndarray)
    np.testing.assert_array_almost_equal(beam.bandpass_array, interp_bandpass)
    np.testing.assert_array_almost_equal(beam.data_array, interp_data)
    if antenna_type == "simple":
        assert interp_basis_vector is None
    else:
        np.testing.assert_array_almost_equal(
            beam.basis_vector_array, interp_basis_vector
        )
        np.testing.assert_array_almost_equal(
            beam.coupling_matrix, interp_coupling_matrix
        )

    interp_arrays = beam.interp(
        freq_array=freq_orig_vals,
        freq_interp_tol=1.0,
        freq_interp_kind="cubic",
        return_bandpass=True,
        return_coupling=need_coupling,
    )
    if antenna_type == "simple":
        interp_data, interp_basis_vector, interp_bandpass = interp_arrays
    else:
        (interp_data, interp_basis_vector, interp_bandpass, interp_coupling_matrix) = (
            interp_arrays
        )
    assert isinstance(interp_data, np.ndarray)
    assert isinstance(interp_bandpass, np.ndarray)
    np.testing.assert_array_almost_equal(beam.bandpass_array, interp_bandpass)
    np.testing.assert_array_almost_equal(beam.data_array, interp_data)
    if antenna_type == "simple":
        assert interp_basis_vector is None
    else:
        np.testing.assert_array_almost_equal(
            beam.basis_vector_array, interp_basis_vector
        )

    # test frequency interpolation returns new UVBeam for small and large tolerances
    beam.saved_interp_functions = {}

    optional_freq_params = [
        "receiver_temperature_array",
        "loss_array",
        "mismatch_array",
        "s_parameters",
    ]
    exp_warnings = []
    for param_name in optional_freq_params:
        exp_warnings.append(
            f"Input object has {param_name} defined but we do not "
            "currently support interpolating it in frequency. Returned "
            "object will have it set to None."
        )
    # check setting freq_interp_kind on object also works
    with uvtest.check_warnings(
        DeprecationWarning,
        match="The freq_interp_kind attribute on UVBeam objects is "
        "deprecated and support for it will be removed in version 2.6. ",
    ):
        beam.freq_interp_kind = "linear"
    with uvtest.check_warnings(UserWarning, match=exp_warnings):
        new_beam_obj = beam.interp(
            freq_array=freq_orig_vals, freq_interp_tol=0.0, new_object=True
        )
    assert isinstance(new_beam_obj, UVBeam)
    if future_shapes:
        np.testing.assert_array_almost_equal(new_beam_obj.freq_array, freq_orig_vals)
    else:
        np.testing.assert_array_almost_equal(new_beam_obj.freq_array[0], freq_orig_vals)
    # test that saved functions are erased in new obj
    assert not hasattr(new_beam_obj, "saved_interp_functions")
    assert "freq_interp_kind = linear" in new_beam_obj.history
    assert beam.history != new_beam_obj.history
    new_beam_obj.history = beam.history
    # add back optional params to get equality:
    for param_name in optional_freq_params:
        setattr(new_beam_obj, param_name, getattr(beam, param_name))
    assert beam == new_beam_obj

    with uvtest.check_warnings(
        [UserWarning] * (len(exp_warnings) + 1),
        match=exp_warnings
        + [
            "The freq_interp_kind parameter was set but it does not "
            "match the freq_interp_kind attribute on the object. "
            "Using the one passed to this method."
        ],
    ):
        new_beam_obj = beam.interp(
            freq_array=freq_orig_vals,
            freq_interp_tol=1.0,
            freq_interp_kind="cubic",
            new_object=True,
        )
    assert isinstance(new_beam_obj, UVBeam)
    if future_shapes:
        np.testing.assert_array_almost_equal(new_beam_obj.freq_array, freq_orig_vals)
    else:
        np.testing.assert_array_almost_equal(new_beam_obj.freq_array[0], freq_orig_vals)
    # assert interp kind is 'nearest' when within tol
    assert "freq_interp_kind = nearest" in new_beam_obj.history
    assert beam.history != new_beam_obj.history
    new_beam_obj.history = beam.history
    # add back optional params to get equality:
    for param_name in optional_freq_params:
        setattr(new_beam_obj, param_name, getattr(beam, param_name))
    assert beam == new_beam_obj

    # test frequency interpolation returns valid new UVBeam for different
    # number of freqs from input
    beam.saved_interp_functions = {}
    with uvtest.check_warnings(UserWarning, match=exp_warnings):
        new_beam_obj = beam.interp(
            freq_array=np.linspace(123e6, 150e6, num=5),
            freq_interp_tol=0.0,
            freq_interp_kind="linear",
            new_object=True,
        )

    assert isinstance(new_beam_obj, UVBeam)
    if future_shapes:
        np.testing.assert_array_almost_equal(
            new_beam_obj.freq_array, np.linspace(123e6, 150e6, num=5)
        )
    else:
        np.testing.assert_array_almost_equal(
            new_beam_obj.freq_array[0], np.linspace(123e6, 150e6, num=5)
        )
    # test that saved functions are erased in new obj
    assert not hasattr(new_beam_obj, "saved_interp_functions")
    assert beam.history != new_beam_obj.history
    new_beam_obj.history = beam.history

    # down select to orig freqs and test equality
    new_beam_obj.select(frequencies=freq_orig_vals)
    assert beam.history != new_beam_obj.history
    new_beam_obj.history = beam.history
    # add back optional params to get equality:
    for param_name in optional_freq_params:
        setattr(new_beam_obj, param_name, getattr(beam, param_name))
    assert beam == new_beam_obj

    # using only one freq chan should trigger a ValueError if interp_bool is True
    # unless requesting the original frequency channel such that interp_bool is False.
    # Therefore, to test that interp_bool is False returns array slice as desired,
    # test that ValueError is not raised in this case.
    # Other ways of testing this (e.g. interp_data_array.flags['OWNDATA']) does not work
    if future_shapes:
        _pb = beam.select(frequencies=beam.freq_array[:1], inplace=False)
        freq_arr_use = _pb.freq_array
    else:
        _pb = beam.select(frequencies=beam.freq_array[0, :1], inplace=False)
        freq_arr_use = _pb.freq_array[0]
    try:
        interp_data, interp_basis_vector = _pb.interp(freq_array=freq_arr_use)
    except ValueError as err:
        raise AssertionError(
            "UVBeam.interp didn't return an array slice as expected"
        ) from err

    # test errors if one frequency
    beam_singlef = beam.select(freq_chans=[0], inplace=False)
    with pytest.raises(
        ValueError, match="Only one frequency in UVBeam so cannot interpolate."
    ):
        beam_singlef.interp(freq_array=np.array([150e6]))


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_freq_interp_real_and_complex(future_shapes, cst_power_2freq):
    # test interpolation of real and complex data are the same
    power_beam = cst_power_2freq

    if not future_shapes:
        power_beam.use_current_array_shapes()

    # make a new object with more frequencies
    freqs = np.linspace(123e6, 150e6, 4)

    optional_freq_params = [
        "receiver_temperature_array",
        "loss_array",
        "mismatch_array",
        "s_parameters",
    ]
    exp_warnings = []
    for param_name in optional_freq_params:
        exp_warnings.append(
            f"Input object has {param_name} defined but we do not "
            "currently support interpolating it in frequency. Returned "
            "object will have it set to None."
        )
    with uvtest.check_warnings(UserWarning, match=exp_warnings):
        pbeam = power_beam.interp(
            freq_array=freqs, freq_interp_kind="linear", new_object=True
        )

    # modulate the data
    pbeam.data_array[..., 1] *= 2
    pbeam.data_array[..., 2] *= 0.5

    # interpolate cubic on real data
    freqs = np.linspace(123e6, 150e6, 10)
    pb_int = pbeam.interp(freq_array=freqs)[0]

    # interpolate cubic on complex data and compare to ensure they are the same
    pbeam.data_array = pbeam.data_array.astype(np.complex128)
    pb_int2 = pbeam.interp(freq_array=freqs)[0]
    assert np.all(np.isclose(np.abs(pb_int - pb_int2), 0))


@pytest.mark.parametrize("beam_type", ["efield", "power", "phased_array"])
def test_spatial_interpolation_samepoints(
    beam_type, cst_power_2freq_cut, cst_efield_2freq_cut, phased_array_beam_2freq
):
    """
    check that interpolating to existing points gives the same answer
    """
    if beam_type == "power":
        uvbeam = cst_power_2freq_cut
    elif beam_type == "efield":
        uvbeam = cst_efield_2freq_cut
    else:
        uvbeam = phased_array_beam_2freq

    za_orig_vals, az_orig_vals = np.meshgrid(uvbeam.axis2_array, uvbeam.axis1_array)
    az_orig_vals = az_orig_vals.ravel(order="C")
    za_orig_vals = za_orig_vals.ravel(order="C")
    freq_orig_vals = np.array([123e6, 150e6])

    # test defaulting works if no interpolation function is set
    interp_data_array, interp_basis_vector = uvbeam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )

    interp_data_array2, interp_basis_vector2 = uvbeam.interp(
        az_array=az_orig_vals,
        za_array=za_orig_vals,
        freq_array=freq_orig_vals,
        interpolation_function="az_za_simple",
    )
    assert np.allclose(interp_data_array, interp_data_array2)
    if beam_type == "efield":
        assert np.allclose(interp_basis_vector, interp_basis_vector2)

    interp_data_array = interp_data_array.reshape(uvbeam.data_array.shape, order="F")
    assert np.allclose(uvbeam.data_array, interp_data_array)
    if beam_type == "efield":
        interp_basis_vector = interp_basis_vector.reshape(
            uvbeam.basis_vector_array.shape, order="F"
        )
        assert np.allclose(uvbeam.basis_vector_array, interp_basis_vector)

    # test error with using an incompatible interpolation function
    with pytest.raises(
        ValueError,
        match=re.escape(
            "pixel_coordinate_system must be 'healpix' to use this interpolation "
            "function"
        ),
    ):
        interp_data_array, interp_basis_vector = uvbeam.interp(
            az_array=az_orig_vals,
            za_array=za_orig_vals,
            freq_array=freq_orig_vals,
            interpolation_function="healpix_simple",
        )

    # test warning if interpolation_function is set differently on object and in
    # function call and error if not set to known function
    with pytest.raises(
        ValueError, match="interpolation_function not recognized, must be one of "
    ):
        interp_data_array, interp_basis_vector = uvbeam.interp(
            az_array=az_orig_vals,
            za_array=za_orig_vals,
            freq_array=freq_orig_vals,
            interpolation_function="foo",
        )

    # test that new object from interpolation is identical
    optional_freq_params = [
        "receiver_temperature_array",
        "loss_array",
        "mismatch_array",
        "s_parameters",
    ]
    exp_warnings = []
    for param_name in optional_freq_params:
        exp_warnings.append(
            f"Input object has {param_name} defined but we do not "
            "currently support interpolating it in frequency. Returned "
            "object will have it set to None."
        )
    with uvtest.check_warnings(UserWarning, match=exp_warnings):
        new_beam = uvbeam.interp(
            az_array=uvbeam.axis1_array,
            za_array=uvbeam.axis2_array,
            az_za_grid=True,
            freq_array=freq_orig_vals,
            new_object=True,
        )
    assert new_beam.history == (
        uvbeam.history + " Interpolated in "
        "frequency and to a new azimuth/zenith "
        "angle grid using pyuvdata with "
        "interpolation_function = az_za_simple "
        "and freq_interp_kind = nearest."
    )
    # make histories & freq_interp_kind equal
    new_beam.history = uvbeam.history
    # add back optional params to get equality:
    for param_name in optional_freq_params:
        setattr(new_beam, param_name, getattr(uvbeam, param_name))
    assert new_beam == uvbeam

    # test error if new_object set without az_za_grid
    with pytest.raises(ValueError, match="A new object can only be returned"):
        uvbeam.interp(
            az_array=az_orig_vals,
            za_array=za_orig_vals,
            freq_array=freq_orig_vals,
            new_object=True,
        )

    if beam_type == "power":
        # test only a single polarization
        interp_data_array, interp_basis_vector = uvbeam.interp(
            az_array=az_orig_vals,
            za_array=za_orig_vals,
            freq_array=freq_orig_vals,
            polarizations=["xx"],
        )

        data_array_compare = uvbeam.data_array[:, :1]
        interp_data_array = interp_data_array.reshape(
            data_array_compare.shape, order="F"
        )
        assert np.allclose(data_array_compare, interp_data_array)


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_spatial_interpolation_everyother(
    future_shapes, beam_type, cst_power_2freq_cut, cst_efield_2freq_cut
):
    """
    test that interp to every other point returns an object that matches a select
    """
    if beam_type == "power":
        uvbeam = cst_power_2freq_cut
    else:
        uvbeam = cst_efield_2freq_cut

    if not future_shapes:
        uvbeam.use_current_array_shapes()

    axis1_inds = np.arange(0, uvbeam.Naxes1, 2)
    axis2_inds = np.arange(0, uvbeam.Naxes2, 2)

    select_beam = uvbeam.select(
        axis1_inds=axis1_inds, axis2_inds=axis2_inds, inplace=False
    )
    interp_beam = uvbeam.interp(
        az_array=uvbeam.axis1_array[axis1_inds],
        za_array=uvbeam.axis2_array[axis2_inds],
        az_za_grid=True,
        new_object=True,
    )
    assert select_beam.history != interp_beam.history
    interp_beam.history = select_beam.history
    assert select_beam == interp_beam

    # test no errors using different points
    az_interp_vals = np.array(
        np.arange(0, 2 * np.pi, np.pi / 9.0).tolist()
        + np.arange(0, 2 * np.pi, np.pi / 9.0).tolist()
    )
    za_interp_vals = np.array(
        (np.zeros((18)) + np.pi / 18).tolist() + (np.zeros((18)) + np.pi / 36).tolist()
    )
    freq_interp_vals = np.arange(125e6, 145e6, 5e6)

    _, _ = uvbeam.interp(
        az_array=az_interp_vals,
        za_array=za_interp_vals,
        freq_array=freq_interp_vals,
        freq_interp_kind="linear",
    )

    if beam_type == "power":
        # Test requesting separate polarizations on different calls
        # while reusing splines.
        _, _ = uvbeam.interp(
            az_array=az_interp_vals[:2],
            za_array=za_interp_vals[:2],
            freq_array=freq_interp_vals,
            freq_interp_kind="linear",
            polarizations=["xx"],
            reuse_spline=True,
        )

        _, _ = uvbeam.interp(
            az_array=az_interp_vals[:2],
            za_array=za_interp_vals[:2],
            freq_array=freq_interp_vals,
            freq_interp_kind="linear",
            polarizations=["yy"],
            reuse_spline=True,
        )

    # test reusing the spline fit.
    orig_data_array, _ = uvbeam.interp(
        az_array=az_interp_vals,
        za_array=za_interp_vals,
        freq_array=freq_interp_vals,
        freq_interp_kind="linear",
        reuse_spline=True,
    )

    reused_data_array, _ = uvbeam.interp(
        az_array=az_interp_vals,
        za_array=za_interp_vals,
        freq_array=freq_interp_vals,
        freq_interp_kind="linear",
        reuse_spline=True,
    )
    assert np.all(reused_data_array == orig_data_array)

    # test passing spline options
    spline_opts = {"kx": 4, "ky": 4}
    quartic_data_array, _ = uvbeam.interp(
        az_array=az_interp_vals,
        za_array=za_interp_vals,
        freq_array=freq_interp_vals,
        freq_interp_kind="linear",
        spline_opts=spline_opts,
    )

    # slightly different interpolation, so not identical.
    assert np.allclose(quartic_data_array, orig_data_array, atol=1e-10)
    assert not np.all(quartic_data_array == orig_data_array)

    select_data_array_orig, _ = uvbeam.interp(
        az_array=az_interp_vals[0:1],
        za_array=za_interp_vals[0:1],
        freq_array=np.array([127e6]),
        freq_interp_kind="linear",
    )

    select_data_array_reused, _ = uvbeam.interp(
        az_array=az_interp_vals[0:1],
        za_array=za_interp_vals[0:1],
        freq_array=np.array([127e6]),
        freq_interp_kind="linear",
        reuse_spline=True,
    )
    assert np.allclose(select_data_array_orig, select_data_array_reused)
    del uvbeam.saved_interp_functions


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_spatial_interp_cutsky(beam_type, cst_power_2freq_cut, cst_efield_2freq_cut):
    """
    Test that when the beam doesn't cover the full sky it still works.
    """
    if beam_type == "power":
        uvbeam = cst_power_2freq_cut
    else:
        uvbeam = cst_efield_2freq_cut

    # limit phi range
    axis1_inds = np.arange(0, np.ceil(uvbeam.Naxes1 / 2), dtype=int)
    axis2_inds = np.arange(0, uvbeam.Naxes2)

    uvbeam.select(axis1_inds=axis1_inds, axis2_inds=axis2_inds)

    # now do every other point test.
    axis1_inds = np.arange(0, uvbeam.Naxes1, 2)
    axis2_inds = np.arange(0, uvbeam.Naxes2, 2)

    select_beam = uvbeam.select(
        axis1_inds=axis1_inds, axis2_inds=axis2_inds, inplace=False
    )
    interp_beam = uvbeam.interp(
        az_array=uvbeam.axis1_array[axis1_inds],
        za_array=uvbeam.axis2_array[axis2_inds],
        az_za_grid=True,
        new_object=True,
    )
    assert select_beam.history != interp_beam.history
    interp_beam.history = select_beam.history
    assert select_beam == interp_beam


def test_spatial_interpolation_errors(cst_power_2freq_cut):
    uvbeam = cst_power_2freq_cut

    az_interp_vals = np.array(
        np.arange(0, 2 * np.pi, np.pi / 9.0).tolist()
        + np.arange(0, 2 * np.pi, np.pi / 9.0).tolist()
    )
    za_interp_vals = np.array(
        (np.zeros((18)) + np.pi / 18).tolist() + (np.zeros((18)) + np.pi / 36).tolist()
    )
    freq_interp_vals = np.arange(125e6, 145e6, 5e6)

    # test errors if frequency interp values outside range
    with pytest.raises(
        ValueError,
        match="at least one interpolation frequency is outside of "
        "the UVBeam freq_array range.",
    ):
        uvbeam.interp(
            az_array=az_interp_vals, za_array=za_interp_vals, freq_array=np.array([100])
        )

    # test errors if frequency interp values outside range
    with pytest.raises(
        ValueError,
        match="If az_za_grid is set to True, az_array and za_array must be provided.",
    ):
        uvbeam.interp(az_za_grid=True, freq_array=freq_interp_vals)
    # test errors if positions outside range
    with pytest.raises(
        ValueError,
        match="at least one interpolation location "
        "is outside of the UVBeam pixel coverage.",
    ):
        uvbeam.interp(az_array=az_interp_vals, za_array=za_interp_vals + np.pi / 2)

    # test no errors only frequency interpolation
    _, _ = uvbeam.interp(freq_array=freq_interp_vals, freq_interp_kind="linear")

    # assert polarization value error
    with pytest.raises(
        ValueError,
        match="Requested polarization 1 not found in self.polarization_array",
    ):
        uvbeam.interp(
            az_array=az_interp_vals, za_array=za_interp_vals, polarizations=["pI"]
        )

    # test error returning coupling matrix for simple antenna_types
    with pytest.raises(
        ValueError,
        match="return_coupling can only be set if antenna_type is phased_array",
    ):
        uvbeam.interp(
            az_array=az_interp_vals, za_array=za_interp_vals, return_coupling=True
        )


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_interp_longitude_branch_cut(beam_type, cst_efield_2freq, cst_power_2freq):
    if beam_type == "power":
        beam = cst_power_2freq
    else:
        beam = cst_efield_2freq

    interp_data_array, _ = beam.interp(
        az_array=np.deg2rad(
            np.repeat(np.array([[-1], [359], [0], [360]]), 181, axis=1).flatten()
        ),
        za_array=np.repeat(beam.axis2_array[np.newaxis, :], 4, axis=0).flatten(),
    )

    if beam_type == "power":
        npol_feed = beam.Npols
    else:
        npol_feed = beam.Nfeeds

    interp_data_array = interp_data_array.reshape(
        beam.Naxes_vec, npol_feed, beam.Nfreqs, 4, beam.Naxes2
    )

    assert np.allclose(
        interp_data_array[:, :, :, 0, :],
        interp_data_array[:, :, :, 1, :],
        rtol=beam._data_array.tols[0],
        atol=beam._data_array.tols[1],
    )

    assert np.allclose(
        interp_data_array[:, :, :, 2, :],
        interp_data_array[:, :, :, 3, :],
        rtol=beam._data_array.tols[0],
        atol=beam._data_array.tols[1],
    )


def test_interp_healpix_nside(cst_efield_2freq, cst_efield_2freq_cut_healpix):
    efield_beam = cst_efield_2freq

    # check nside calculation
    min_res = np.min(
        np.array(
            [np.diff(efield_beam.axis1_array)[0], np.diff(efield_beam.axis2_array)[0]]
        )
    )
    nside_min_res = np.sqrt(3 / np.pi) * np.radians(60.0) / min_res
    nside = int(2 ** np.ceil(np.log2(nside_min_res)))
    assert cst_efield_2freq_cut_healpix.nside == nside

    # check that calling without specifying hpx indices doesn't error
    # select every eighth point to make it smaller
    axis1_inds = np.arange(0, efield_beam.Naxes1, 8)
    axis2_inds = np.arange(0, efield_beam.Naxes2, 8)
    efield_beam.select(axis1_inds=axis1_inds, axis2_inds=axis2_inds)
    hpx_beam = efield_beam.interp(healpix_nside=64, new_object=True)
    assert hpx_beam.Npixels == 12 * hpx_beam.nside**2


def test_interp_healpix_errors(cst_efield_2freq_cut, cst_efield_2freq_cut_healpix):
    efield_beam = cst_efield_2freq_cut

    new_efield_beam = cst_efield_2freq_cut_healpix

    # check error with cut sky
    with pytest.raises(
        ValueError, match="simple healpix interpolation requires full sky healpix maps."
    ):
        new_efield_beam.interp(
            az_array=efield_beam.axis1_array,
            za_array=efield_beam.axis2_array,
            az_za_grid=True,
            new_object=True,
        )


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("antenna_type", ["simple", "phased_array"])
def test_healpix_interpolation(
    future_shapes, antenna_type, cst_efield_2freq, phased_array_beam_2freq
):
    pytest.importorskip("astropy_healpix")
    if antenna_type == "simple":
        efield_beam = cst_efield_2freq
    else:
        efield_beam = phased_array_beam_2freq

    if not future_shapes:
        efield_beam.use_current_array_shapes()

    # select every fourth point to make it smaller
    axis1_inds = np.arange(0, efield_beam.Naxes1, 4)
    axis2_inds = np.arange(0, efield_beam.Naxes2, 4)
    efield_beam.select(axis1_inds=axis1_inds, axis2_inds=axis2_inds)

    hpx_efield_beam = efield_beam.to_healpix(inplace=False)

    # check that interpolating to existing points gives the same answer
    hp_obj = HEALPix(nside=hpx_efield_beam.nside)
    hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(hpx_efield_beam.pixel_array)
    za_orig_vals = (Angle(np.pi / 2, units.radian) - hpx_lat).radian
    az_orig_vals = hpx_lon.radian

    az_orig_vals = az_orig_vals.ravel(order="C")
    za_orig_vals = za_orig_vals.ravel(order="C")
    freq_orig_vals = np.array([123e6, 150e6])

    interp_data_array, _ = hpx_efield_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )
    data_array_compare = hpx_efield_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order="F")
    assert np.allclose(data_array_compare, interp_data_array)

    # test error with using an incompatible interpolation function
    with pytest.raises(
        ValueError,
        match=re.escape(
            "pixel_coordinate_system must be 'az_za' to use this interpolation "
            "function"
        ),
    ):
        interp_data_array, _ = hpx_efield_beam.interp(
            az_array=az_orig_vals,
            za_array=za_orig_vals,
            freq_array=freq_orig_vals,
            interpolation_function="az_za_simple",
        )

    # test that interp to every other point returns an object that matches a select
    pixel_inds = np.arange(0, hpx_efield_beam.Npixels, 2)
    select_beam = hpx_efield_beam.select(pixels=pixel_inds, inplace=False)
    interp_beam = hpx_efield_beam.interp(
        healpix_inds=hpx_efield_beam.pixel_array[pixel_inds],
        healpix_nside=hpx_efield_beam.nside,
        new_object=True,
    )
    assert select_beam.history != interp_beam.history
    interp_beam.history = select_beam.history
    assert select_beam == interp_beam

    # check history with interp healpix & freq
    message = [
        f"Input object has {param_name} defined but we do not "
        "currently support interpolating it in frequency. Returned "
        "object will have it set to None."
        for param_name in [
            "receiver_temperature_array",
            "loss_array",
            "mismatch_array",
            "s_parameters",
        ]
    ]
    with uvtest.check_warnings(UserWarning, match=message):
        interp_beam = hpx_efield_beam.interp(
            healpix_inds=hpx_efield_beam.pixel_array[pixel_inds],
            healpix_nside=hpx_efield_beam.nside,
            freq_array=np.array([np.mean(freq_orig_vals)]),
            freq_interp_kind="linear",
            new_object=True,
        )
    assert "Interpolated in frequency and to a new healpix grid" in interp_beam.history

    # test interp from healpix to regular az/za grid
    new_reg_beam = hpx_efield_beam.interp(
        az_array=efield_beam.axis1_array,
        za_array=efield_beam.axis2_array,
        az_za_grid=True,
        new_object=True,
    )

    # this diff is pretty large. 2 rounds of interpolation is not a good thing.
    # but we can check that the rest of the object makes sense
    diff = new_reg_beam.data_array - efield_beam.data_array
    diff_ratio = diff / efield_beam.data_array
    assert np.all(np.abs(diff_ratio) < 4)
    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [4, 0]
    new_reg_beam._data_array.tols = tols
    assert new_reg_beam.history != efield_beam.history
    new_reg_beam.history = efield_beam.history
    assert new_reg_beam == efield_beam

    # test no inputs equals same answer
    interp_data_array2, _ = hpx_efield_beam.interp()
    assert np.allclose(interp_data_array, interp_data_array2)

    # test errors with specifying healpix_inds without healpix_nside
    hp_obj = HEALPix(nside=hpx_efield_beam.nside)
    with pytest.raises(
        ValueError, match="healpix_nside must be set if healpix_inds is set"
    ):
        hpx_efield_beam.interp(
            healpix_inds=np.arange(hp_obj.npix), freq_array=freq_orig_vals
        )

    # test error setting both healpix_nside and az_array
    with pytest.raises(
        ValueError,
        match="healpix_nside and healpix_inds can not be set if az_array or "
        "za_array is set.",
    ):
        hpx_efield_beam.interp(
            healpix_nside=hpx_efield_beam.nside,
            az_array=az_orig_vals,
            za_array=za_orig_vals,
            freq_array=freq_orig_vals,
        )

    # basis_vector exception
    hpx_efield_beam.basis_vector_array[0, 1, :] = 10.0
    with pytest.raises(
        NotImplementedError,
        match="interpolation for input basis vectors that are not aligned to the "
        "native theta/phi coordinate system is not yet supported",
    ):
        hpx_efield_beam.interp(az_array=az_orig_vals, za_array=za_orig_vals)

    # now convert to power beam
    if antenna_type == "phased_array":
        with pytest.raises(
            NotImplementedError,
            match="Conversion to power is not yet implemented for phased_array",
        ):
            hpx_efield_beam.efield_to_power()
        return

    power_beam = hpx_efield_beam.efield_to_power(inplace=False)
    del hpx_efield_beam
    interp_data_array, _ = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )
    data_array_compare = power_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order="F")
    assert np.allclose(data_array_compare, interp_data_array)

    # test that interp to every other point returns an object that matches a select
    pixel_inds = np.arange(0, power_beam.Npixels, 2)
    select_beam = power_beam.select(pixels=pixel_inds, inplace=False)
    interp_beam = power_beam.interp(
        healpix_inds=power_beam.pixel_array[pixel_inds],
        healpix_nside=power_beam.nside,
        new_object=True,
    )
    assert select_beam.history != interp_beam.history
    interp_beam.history = select_beam.history
    assert select_beam == interp_beam

    # assert not feeding frequencies gives same answer
    interp_data_array2, _ = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals
    )
    assert np.allclose(interp_data_array, interp_data_array2)

    # assert not feeding az_array gives same answer
    interp_data_array2, _ = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals
    )
    assert np.allclose(interp_data_array, interp_data_array2)

    # test requesting polarization gives the same answer
    interp_data_array2, _ = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, polarizations=["yy"]
    )

    assert np.allclose(
        interp_data_array[..., 1:2, :, :], interp_data_array2[..., :1, :, :]
    )

    # change complex data_array to real data_array and test again
    assert power_beam.data_array.dtype == np.complex128
    power_beam.data_array = np.abs(power_beam.data_array)
    interp_data_array, _ = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )
    data_array_compare = power_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order="F")
    assert np.allclose(data_array_compare, interp_data_array)

    # assert polarization value error
    with pytest.raises(
        ValueError,
        match="Requested polarization 1 not found in self.polarization_array",
    ):
        power_beam.interp(
            az_array=az_orig_vals, za_array=za_orig_vals, polarizations=["pI"]
        )

    # check error when pixels out of order
    if future_shapes:
        power_beam.pixel_array = power_beam.pixel_array[
            np.argsort(power_beam.data_array[0, 0, 0, :])
        ]
    else:
        power_beam.pixel_array = power_beam.pixel_array[
            np.argsort(power_beam.data_array[0, 0, 0, 0, :])
        ]
    with pytest.raises(
        ValueError,
        match="simple healpix interpolation requires healpix pixels to be in order.",
    ):
        power_beam.interp(az_array=az_orig_vals, za_array=za_orig_vals)


@pytest.mark.parametrize(
    "start, stop",
    [
        (-3 * np.pi, -2 * np.pi),
        (-np.pi, 0),
        (2 * np.pi, 3 * np.pi),
        (10 * np.pi, 11 * np.pi),
    ],
)
@pytest.mark.parametrize("phi_start, phi_end", [(0, 2 * np.pi), (0, -2 * np.pi)])
def test_find_healpix_indices(start, stop, phi_start, phi_end):
    pytest.importorskip("astropy_healpix")
    hp_obj = HEALPix(nside=2)
    pixels = np.arange(hp_obj.npix)
    hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(pixels)

    hpx_theta = (Angle(np.pi / 2, units.radian) - hpx_lat).radian
    hpx_phi = hpx_lon.radian

    theta_vals1 = np.linspace(0, np.pi, 5, endpoint=True)
    theta_vals2 = np.linspace(start, stop, 5, endpoint=True)

    phi_vals = np.linspace(phi_start, phi_end, 10, endpoint=False)

    inds_to_use1 = _uvbeam.find_healpix_indices(
        np.ascontiguousarray(theta_vals1, dtype=np.float64),
        np.ascontiguousarray(phi_vals, dtype=np.float64),
        np.ascontiguousarray(hpx_theta, dtype=np.float64),
        np.ascontiguousarray(hpx_phi, dtype=np.float64),
        np.float64(hp_obj.pixel_resolution.to_value(units.radian)),
    )

    inds_to_use2 = _uvbeam.find_healpix_indices(
        np.ascontiguousarray(theta_vals2, dtype=np.float64),
        np.ascontiguousarray(phi_vals, dtype=np.float64),
        np.ascontiguousarray(hpx_theta, dtype=np.float64),
        np.ascontiguousarray(hpx_phi, dtype=np.float64),
        np.float64(hp_obj.pixel_resolution.to_value(units.radian)),
    )

    assert np.array_equal(np.sort(pixels[inds_to_use1]), np.sort(pixels[inds_to_use2]))


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_to_healpix_power(
    future_shapes, cst_power_2freq_cut, cst_power_2freq_cut_healpix
):
    power_beam = cst_power_2freq_cut
    power_beam_healpix = cst_power_2freq_cut_healpix

    if not future_shapes:
        power_beam.use_current_array_shapes()
        power_beam_healpix.use_current_array_shapes()

    sky_area_reduction_factor = (1.0 - np.cos(np.deg2rad(10))) / 2.0

    # check that history is updated appropriately
    assert power_beam_healpix.history == (
        power_beam.history
        + " Interpolated from "
        + power_beam.coordinate_system_dict["az_za"]["description"]
        + " to "
        + power_beam.coordinate_system_dict["healpix"]["description"]
        + " using pyuvdata with interpolation_function = az_za_simple."
    )

    hp_obj = HEALPix(nside=power_beam_healpix.nside)
    assert power_beam_healpix.Npixels <= hp_obj.npix * (sky_area_reduction_factor * 1.5)

    # test that Npixels make sense
    n_max_pix = power_beam.Naxes1 * power_beam.Naxes2
    assert power_beam_healpix.Npixels <= n_max_pix

    # Test error if not az_za
    power_beam.pixel_coordinate_system = "sin_zenith"
    with pytest.raises(
        ValueError,
        match="There is no default interpolation function for objects with "
        "pixel_coordinate_system: sin_zenith",
    ):
        power_beam.to_healpix()


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_to_healpix_efield(
    future_shapes, cst_efield_2freq_cut, cst_efield_2freq_cut_healpix
):
    efield_beam = cst_efield_2freq_cut
    interp_then_sq = cst_efield_2freq_cut_healpix

    if not future_shapes:
        efield_beam.use_current_array_shapes()
        interp_then_sq.use_current_array_shapes()

    interp_then_sq.efield_to_power(calc_cross_pols=False)

    # convert to power and then interpolate to compare.
    # Don't use power read from file because it has rounding errors that will
    # dominate this comparison
    sq_then_interp = efield_beam.efield_to_power(calc_cross_pols=False, inplace=False)
    sq_then_interp.to_healpix(
        nside=interp_then_sq.nside, interpolation_function="az_za_simple"
    )

    # square then interpolate is different from interpolate then square at a
    # higher level than normally allowed in the equality.
    # We can live with it for now, may need to improve it later
    diff = np.abs(interp_then_sq.data_array - sq_then_interp.data_array)
    assert np.max(diff) < 0.6
    reldiff = diff * 2 / np.abs(interp_then_sq.data_array + sq_then_interp.data_array)
    assert np.max(reldiff) < 0.005

    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [0.05, 0]
    sq_then_interp._data_array.tols = tols

    # check history changes
    interp_history_add = (
        " Interpolated from "
        + efield_beam.coordinate_system_dict["az_za"]["description"]
        + " to "
        + efield_beam.coordinate_system_dict["healpix"]["description"]
        + " using pyuvdata with interpolation_function = az_za_simple."
    )
    sq_history_add = " Converted from efield to power using pyuvdata."
    assert (
        sq_then_interp.history
        == efield_beam.history + sq_history_add + interp_history_add
    )
    assert (
        interp_then_sq.history
        == efield_beam.history + interp_history_add + sq_history_add
    )

    # now change history on one so we can compare the rest of the object
    sq_then_interp.history = efield_beam.history + interp_history_add + sq_history_add

    # set interpolation function for equality
    assert sq_then_interp == interp_then_sq


@pytest.mark.parametrize("inplace", [True, False])
def test_to_healpix_no_op(cst_power_2freq_cut_healpix, inplace):
    uvbeam = cst_power_2freq_cut_healpix
    if inplace:
        uvbeam2 = uvbeam.copy()
        uvbeam2.to_healpix(inplace=True)
    else:
        uvbeam2 = uvbeam.to_healpix(inplace=False)

    assert uvbeam == uvbeam2


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_select_axis(future_shapes, cst_power_1freq, tmp_path):
    power_beam = cst_power_1freq
    if not future_shapes:
        power_beam.use_current_array_shapes()

    old_history = power_beam.history

    # Test selecting on axis1
    inds1_to_keep = np.arange(14, 63)

    power_beam2 = power_beam.select(axis1_inds=inds1_to_keep, inplace=False)

    assert len(inds1_to_keep) == power_beam2.Naxes1
    for i in inds1_to_keep:
        assert power_beam.axis1_array[i] in power_beam2.axis1_array
    for i in np.unique(power_beam2.axis1_array):
        assert i in power_beam.axis1_array

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific parts of first image axis "
        "using pyuvdata.",
        power_beam2.history,
    )

    write_file_beamfits = str(tmp_path / "select_beam.fits")

    # test writing beamfits with only one element in axis1
    inds_to_keep = [len(inds1_to_keep) + 1]
    power_beam2 = power_beam.select(axis1_inds=inds_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with indices not included in data
    with pytest.raises(ValueError, match="axis1_inds must be > 0 and < Naxes1"):
        power_beam2.select(axis1_inds=[power_beam.Naxes1 - 1])

    # check for warnings and errors associated with unevenly spaced image pixels
    power_beam2 = power_beam.copy()
    with uvtest.check_warnings(
        UserWarning, "Selected values along first image axis are not evenly spaced"
    ):
        power_beam2.select(axis1_inds=[0, 5, 6])
    with pytest.raises(
        ValueError,
        match="The pixels are not evenly spaced along first axis. "
        "The beam fits format does not support unevenly spaced pixels.",
    ):
        power_beam2.write_beamfits(write_file_beamfits)

    # Test selecting on axis2
    inds2_to_keep = np.arange(5, 14)

    power_beam2 = power_beam.select(axis2_inds=inds2_to_keep, inplace=False)

    assert len(inds2_to_keep) == power_beam2.Naxes2
    for i in inds2_to_keep:
        assert power_beam.axis2_array[i] in power_beam2.axis2_array
    for i in np.unique(power_beam2.axis2_array):
        assert i in power_beam.axis2_array

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific parts of second image axis "
        "using pyuvdata.",
        power_beam2.history,
    )

    write_file_beamfits = str(tmp_path / "select_beam.fits")

    # test writing beamfits with only one element in axis2
    inds_to_keep = [len(inds2_to_keep) + 1]
    power_beam2 = power_beam.select(axis2_inds=inds_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with indices not included in data
    with pytest.raises(ValueError, match="axis2_inds must be > 0 and < Naxes2"):
        power_beam2.select(axis2_inds=[power_beam.Naxes2 - 1])

    # check for warnings and errors associated with unevenly spaced image pixels
    power_beam2 = power_beam.copy()
    with uvtest.check_warnings(
        UserWarning, "Selected values along second image axis are not evenly spaced"
    ):
        power_beam2.select(axis2_inds=[0, 5, 6])
    with pytest.raises(
        ValueError,
        match="The pixels are not evenly spaced along second axis. "
        "The beam fits format does not support unevenly spaced pixels.",
    ):
        power_beam2.write_beamfits(write_file_beamfits)


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("antenna_type", ["simple", "phased_array"])
def test_select_frequencies(
    future_shapes, antenna_type, cst_power_1freq, phased_array_beam_1freq, tmp_path
):
    if antenna_type == "simple":
        beam = cst_power_1freq
    else:
        beam = phased_array_beam_1freq

    if not future_shapes:
        beam.use_current_array_shapes()

    # generate more frequencies for testing by copying and adding several times
    while beam.Nfreqs < 8:
        new_beam = beam.copy()
        new_beam.freq_array = beam.freq_array + beam.Nfreqs * 1e6
        beam += new_beam

    old_history = beam.history
    if future_shapes:
        freqs_to_keep = beam.freq_array[np.arange(2, 7)]
    else:
        freqs_to_keep = beam.freq_array[0, np.arange(2, 7)]

    beam2 = beam.select(frequencies=freqs_to_keep, inplace=False)

    assert len(freqs_to_keep) == beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in beam2.freq_array
    for f in np.unique(beam2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        beam2.history,
    )

    write_file_beamfits = str(tmp_path / "select_beam.fits")
    # test writing beamfits with only one frequency

    if future_shapes:
        freqs_to_keep = beam.freq_array[5]
    else:
        freqs_to_keep = beam.freq_array[0, 5]
    beam2 = beam.select(frequencies=freqs_to_keep, inplace=False)

    if antenna_type == "simple":
        beam2.write_beamfits(write_file_beamfits, clobber=True)

    freq_select = np.max(beam.freq_array) + 10
    # check for errors associated with frequencies not included in data
    with pytest.raises(
        ValueError,
        match="Frequency {f} is not present in the freq_array".format(f=freq_select),
    ):
        beam.select(frequencies=[freq_select])

    # check for warnings and errors associated with unevenly spaced frequencies
    if antenna_type == "simple":
        beam2 = beam.copy()
        if future_shapes:
            freqs_to_keep = beam.freq_array[[0, 5, 6]]
        else:
            freqs_to_keep = beam.freq_array[0, [0, 5, 6]]
        with uvtest.check_warnings(
            UserWarning, "Selected frequencies are not evenly spaced"
        ):
            beam2.select(frequencies=freqs_to_keep)
        with pytest.raises(ValueError, match="The frequencies are not evenly spaced "):
            beam2.write_beamfits(write_file_beamfits)

    # Test selecting on freq_chans
    chans_to_keep = np.arange(2, 7)

    beam2 = beam.select(freq_chans=chans_to_keep, inplace=False)

    assert len(chans_to_keep) == beam2.Nfreqs
    if future_shapes:
        for chan in chans_to_keep:
            assert beam.freq_array[chan] in beam2.freq_array
        for f in np.unique(beam2.freq_array):
            assert f in beam.freq_array[chans_to_keep]
    else:
        for chan in chans_to_keep:
            assert beam.freq_array[0, chan] in beam2.freq_array
        for f in np.unique(beam2.freq_array):
            assert f in beam.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to specific frequencies using pyuvdata.",
        beam2.history,
    )

    # Test selecting both channels and frequencies
    if future_shapes:
        freqs_to_keep = beam.freq_array[np.arange(6, 8)]  # Overlaps with chans
    else:
        freqs_to_keep = beam.freq_array[0, np.arange(6, 8)]  # Overlaps with chans
    all_chans_to_keep = np.arange(2, 8)

    beam2 = beam.select(
        frequencies=freqs_to_keep, freq_chans=chans_to_keep, inplace=False
    )

    assert len(all_chans_to_keep) == beam2.Nfreqs
    if future_shapes:
        for chan in all_chans_to_keep:
            assert beam.freq_array[chan] in beam2.freq_array
        for f in np.unique(beam2.freq_array):
            assert f in beam.freq_array[all_chans_to_keep]
    else:
        for chan in all_chans_to_keep:
            assert beam.freq_array[0, chan] in beam2.freq_array
        for f in np.unique(beam2.freq_array):
            assert f in beam.freq_array[0, all_chans_to_keep]


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("antenna_type", ["simple", "phased_array"])
def test_select_feeds(
    future_shapes, antenna_type, cst_efield_1freq, phased_array_beam_2freq
):
    if antenna_type == "simple":
        efield_beam = cst_efield_1freq
        efield_beam.feed_array = np.array(["n", "e"])
        feeds_to_keep = ["e"]
    else:
        efield_beam = phased_array_beam_2freq
        feeds_to_keep = ["x"]

    if not future_shapes:
        efield_beam.use_current_array_shapes()

    old_history = efield_beam.history

    if antenna_type == "phased_array":
        expected_warning = UserWarning
        warn_msg = (
            "Downselecting feeds on phased array beams will lead to loss of information"
        )
    else:
        expected_warning = None
        warn_msg = ""
    with uvtest.check_warnings(expected_warning, match=warn_msg):
        efield_beam2 = efield_beam.select(feeds=feeds_to_keep, inplace=False)

    assert len(feeds_to_keep) == efield_beam2.Nfeeds
    for f in feeds_to_keep:
        assert f in efield_beam2.feed_array
    for f in np.unique(efield_beam2.feed_array):
        assert f in feeds_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific feeds using pyuvdata.",
        efield_beam2.history,
    )

    # check with physical orientation strings:
    with uvtest.check_warnings(expected_warning, match=warn_msg):
        efield_beam3 = efield_beam.select(feeds=["e"], inplace=False)

    assert efield_beam2 == efield_beam3

    # check for errors associated with feeds not included in data
    with pytest.raises(
        ValueError, match="Feed {f} is not present in the feed_array".format(f="p")
    ):
        with uvtest.check_warnings(expected_warning, match=warn_msg):
            efield_beam.select(feeds=["p"])

    # check for error with selecting polarizations on efield beams
    with pytest.raises(
        ValueError, match="polarizations cannot be used with efield beams"
    ):
        with uvtest.check_warnings(expected_warning, match=warn_msg):
            efield_beam.select(polarizations=[-5, -6])

    # Test check basis vectors
    efield_beam.basis_vector_array[0, 1, :, :] = 1.0
    with pytest.raises(
        ValueError, match="basis vectors must have lengths of 1 or less."
    ):
        efield_beam.check()

    efield_beam.basis_vector_array[0, 0, :, :] = np.sqrt(0.5)
    efield_beam.basis_vector_array[0, 1, :, :] = np.sqrt(0.5)
    assert efield_beam.check()

    efield_beam.basis_vector_array = None
    with pytest.raises(
        ValueError, match="Required UVParameter _basis_vector_array has not been set."
    ):
        efield_beam.check()


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.filterwarnings("ignore:Fixing auto polarization power beams")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize(
    "pols_to_keep", ([-5, -6], ["xx", "yy"], ["nn", "ee"], [[-5, -6]])
)
def test_select_polarizations(future_shapes, pols_to_keep, cst_efield_1freq):
    # generate more polarizations for testing by using efield and keeping cross-pols
    power_beam = cst_efield_1freq
    if not future_shapes:
        power_beam.use_current_array_shapes()
    power_beam.efield_to_power()

    old_history = power_beam.history

    power_beam2 = power_beam.select(polarizations=pols_to_keep, inplace=False)

    if isinstance(pols_to_keep[0], list):
        pols_to_keep = pols_to_keep[0]

    assert len(pols_to_keep) == power_beam2.Npols
    for p in pols_to_keep:
        if isinstance(p, int):
            assert p in power_beam2.polarization_array
        else:
            assert (
                uvutils.polstr2num(p, x_orientation=power_beam2.x_orientation)
                in power_beam2.polarization_array
            )
    for p in np.unique(power_beam2.polarization_array):
        if isinstance(pols_to_keep[0], int):
            assert p in pols_to_keep
        else:
            assert p in uvutils.polstr2num(
                pols_to_keep, x_orientation=power_beam2.x_orientation
            )

    assert uvutils._check_histories(
        old_history + "  Downselected to specific polarizations using pyuvdata.",
        power_beam2.history,
    )


@pytest.mark.filterwarnings("ignore:Fixing auto polarization power beams")
def test_select_polarizations_errors(cst_efield_1freq):
    # generate more polarizations for testing by using efield and keeping cross-pols
    power_beam = cst_efield_1freq
    power_beam.efield_to_power()

    # check for errors associated with polarizations not included in data
    with pytest.raises(
        ValueError,
        match="polarization {p} is not present in the polarization_array".format(p=-3),
    ):
        power_beam.select(polarizations=[-3, -4])

    # check for warnings and errors associated with unevenly spaced polarizations
    with uvtest.check_warnings(
        UserWarning, "Selected polarizations are not evenly spaced"
    ):
        power_beam.select(polarizations=power_beam.polarization_array[[0, 1, 3]])
    write_file_beamfits = os.path.join(DATA_PATH, "test/select_beam.fits")
    with pytest.raises(
        ValueError, match="The polarization values are not evenly spaced "
    ):
        power_beam.write_beamfits(write_file_beamfits)

    # check for error with selecting on feeds on power beams
    with pytest.raises(ValueError, match="feeds cannot be used with power beams"):
        power_beam.select(feeds=["x"])

    # check for error with complex auto pols
    power_beam.data_array[:, 0] = power_beam.data_array[:, 2]
    with uvtest.check_warnings(
        UserWarning,
        match="Polarization select should result in a real array but the "
        "imaginary part is not zero.",
    ):
        with pytest.raises(
            ValueError, match="UVParameter _data_array is not the appropriate type"
        ):
            power_beam.select(polarizations=[-5, -6])


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_select(future_shapes, beam_type, cst_power_1freq, cst_efield_1freq):
    if beam_type == "efield":
        beam = cst_efield_1freq
    else:
        beam = cst_power_1freq

    if not future_shapes:
        beam.use_current_array_shapes()

    # generate more frequencies for testing by copying and adding
    new_beam = beam.copy()
    new_beam.freq_array = beam.freq_array + beam.Nfreqs * 1e6
    beam += new_beam

    # now test selecting along all axes at once
    old_history = beam.history

    inds1_to_keep = np.arange(14, 63)
    inds2_to_keep = np.arange(5, 14)
    if future_shapes:
        freqs_to_keep = [beam.freq_array[0]]
    else:
        freqs_to_keep = [beam.freq_array[0, 0]]
    if beam_type == "efield":
        feeds_to_keep = ["x"]
        pols_to_keep = None
    else:
        pols_to_keep = [-5]
        feeds_to_keep = None

    beam2 = beam.select(
        axis1_inds=inds1_to_keep,
        axis2_inds=inds2_to_keep,
        frequencies=freqs_to_keep,
        polarizations=pols_to_keep,
        feeds=feeds_to_keep,
        inplace=False,
    )

    assert len(inds1_to_keep) == beam2.Naxes1
    for i in inds1_to_keep:
        assert beam.axis1_array[i] in beam2.axis1_array
    for i in np.unique(beam2.axis1_array):
        assert i in beam.axis1_array

    assert len(inds2_to_keep) == beam2.Naxes2
    for i in inds2_to_keep:
        assert beam.axis2_array[i] in beam2.axis2_array
    for i in np.unique(beam2.axis2_array):
        assert i in beam.axis2_array

    assert len(freqs_to_keep) == beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in beam2.freq_array
    for f in np.unique(beam2.freq_array):
        assert f in freqs_to_keep

    if beam_type == "efield":
        assert len(feeds_to_keep) == beam2.Nfeeds
        for f in feeds_to_keep:
            assert f in beam2.feed_array
        for f in np.unique(beam2.feed_array):
            assert f in feeds_to_keep

        assert uvutils._check_histories(
            old_history + "  Downselected to "
            "specific parts of first image axis, "
            "parts of second image axis, "
            "frequencies, feeds using pyuvdata.",
            beam2.history,
        )
    else:
        assert len(pols_to_keep) == beam2.Npols
        for p in pols_to_keep:
            assert p in beam2.polarization_array
        for p in np.unique(beam2.polarization_array):
            assert p in pols_to_keep

        assert uvutils._check_histories(
            old_history + "  Downselected to "
            "specific parts of first image axis, "
            "parts of second image axis, "
            "frequencies, polarizations using pyuvdata.",
            beam2.history,
        )


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_add_axis1(future_shapes, power_beam_for_adding):
    power_beam = power_beam_for_adding

    if not future_shapes:
        power_beam.use_current_array_shapes()

    # Add along first image axis
    beam1 = power_beam.select(axis1_inds=np.arange(0, 180), inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(180, 360), inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific parts of "
        "first image axis using pyuvdata. "
        "Combined data along first image axis "
        "using pyuvdata.",
        beam1.history,
    )
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - axis1
    beam1 = power_beam.select(axis1_inds=np.arange(180, 360), inplace=False)
    beam2 = power_beam.select(axis1_inds=np.arange(0, 180), inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_add_axis2(future_shapes, power_beam_for_adding):
    power_beam = power_beam_for_adding

    if not future_shapes:
        power_beam.use_current_array_shapes()

    # Add along second image axis
    beam1 = power_beam.select(axis2_inds=np.arange(0, 90), inplace=False)
    beam2 = power_beam.select(axis2_inds=np.arange(90, 181), inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific parts of "
        "second image axis using pyuvdata. "
        "Combined data along second image axis "
        "using pyuvdata.",
        beam1.history,
    )
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - axis2
    beam1 = power_beam.select(axis2_inds=np.arange(90, 181), inplace=False)
    beam2 = power_beam.select(axis2_inds=np.arange(0, 90), inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_add_frequencies(future_shapes, power_beam_for_adding):
    power_beam = power_beam_for_adding

    if not future_shapes:
        power_beam.use_current_array_shapes()

    # Add frequencies
    beam1 = power_beam.select(freq_chans=0, inplace=False)
    beam2 = power_beam.select(freq_chans=1, inplace=False)
    beam1 += beam2
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific frequencies "
        "using pyuvdata. Combined data along "
        "frequency axis using pyuvdata.",
        beam1.history,
    )
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - freqs
    beam1 = power_beam.select(freq_chans=1, inplace=False)
    beam2 = power_beam.select(freq_chans=0, inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_add_pols(future_shapes, power_beam_for_adding):
    power_beam = power_beam_for_adding

    if not future_shapes:
        power_beam.use_current_array_shapes()

    # Add polarizations
    beam1 = power_beam.select(polarizations=-5, inplace=False)
    beam2 = power_beam.select(polarizations=-6, inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific polarizations "
        "using pyuvdata. Combined data along "
        "polarization axis using pyuvdata.",
        beam1.history,
    )
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # Out of order - pols
    beam1 = power_beam.select(polarizations=-6, inplace=False)
    beam2 = power_beam.select(polarizations=-5, inplace=False)
    beam1 += beam2
    beam1.history = power_beam.history
    assert beam1 == power_beam


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("antenna_type", ["simple", "phased_array"])
def test_add_feeds(
    future_shapes, antenna_type, efield_beam_for_adding, phased_array_beam_2freq
):
    if antenna_type == "simple":
        efield_beam = efield_beam_for_adding
    else:
        efield_beam = phased_array_beam_2freq

    if not future_shapes:
        efield_beam.use_current_array_shapes()

    if antenna_type == "phased_array":
        expected_warning = UserWarning
        warn_msg = (
            "Downselecting feeds on phased array beams will lead to loss of information"
        )
    else:
        expected_warning = None
        warn_msg = ""

    with uvtest.check_warnings(expected_warning, match=warn_msg):
        beam1 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    with uvtest.check_warnings(expected_warning, match=warn_msg):
        beam2 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(
        efield_beam.history + "  Downselected to specific feeds "
        "using pyuvdata. Combined data along "
        "feed axis using pyuvdata.",
        beam1.history,
    )
    beam1.history = efield_beam.history

    if antenna_type == "phased_array":
        # coupling matrix won't match because info is lost on cross-feed coupling
        assert not np.allclose(beam1.coupling_matrix, efield_beam.coupling_matrix)
        beam1.coupling_matrix[:, :, 0, 1] = efield_beam.coupling_matrix[:, :, 0, 1]
        beam1.coupling_matrix[:, :, 1, 0] = efield_beam.coupling_matrix[:, :, 1, 0]
    assert beam1 == efield_beam

    # Out of order - feeds
    with uvtest.check_warnings(expected_warning, match=warn_msg):
        beam1 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    with uvtest.check_warnings(expected_warning, match=warn_msg):
        beam2 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    beam1 += beam2
    beam1.history = efield_beam.history
    if antenna_type == "phased_array":
        # coupling matrix won't match because info is lost on cross-feed coupling
        assert not np.allclose(beam1.coupling_matrix, efield_beam.coupling_matrix)
        beam1.coupling_matrix[:, :, 0, 1] = efield_beam.coupling_matrix[:, :, 0, 1]
        beam1.coupling_matrix[:, :, 1, 0] = efield_beam.coupling_matrix[:, :, 1, 0]
    assert beam1 == efield_beam


def test_add_multi_power(power_beam_for_adding):
    power_beam = power_beam_for_adding

    # Add multiple axes
    beam_ref = power_beam.copy()
    beam1 = power_beam.select(
        axis1_inds=np.arange(0, power_beam.Naxes1 // 2),
        polarizations=power_beam.polarization_array[0],
        inplace=False,
    )
    beam2 = power_beam.select(
        axis1_inds=np.arange(power_beam.Naxes1 // 2, power_beam.Naxes1),
        polarizations=power_beam.polarization_array[1],
        inplace=False,
    )
    beam1 += beam2
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific parts of "
        "first image axis, polarizations using "
        "pyuvdata. Combined data along first "
        "image, polarization axis using pyuvdata.",
        beam1.history,
    )
    # Zero out missing data in reference object
    beam_ref.data_array[:, 0, :, :, power_beam.Naxes1 // 2 :] = 0.0
    beam_ref.data_array[:, 1, :, :, : power_beam.Naxes1 // 2] = 0.0
    beam1.history = power_beam.history
    assert beam1 == beam_ref


def test_add_multi_efield(efield_beam_for_adding):
    efield_beam = efield_beam_for_adding

    # Another combo with efield
    beam_ref = efield_beam.copy()
    beam1 = efield_beam.select(
        axis1_inds=np.arange(0, efield_beam.Naxes1 // 2),
        axis2_inds=np.arange(0, efield_beam.Naxes2 // 2),
        inplace=False,
    )
    beam2 = efield_beam.select(
        axis1_inds=np.arange(efield_beam.Naxes1 // 2, efield_beam.Naxes1),
        axis2_inds=np.arange(efield_beam.Naxes2 // 2, efield_beam.Naxes2),
        inplace=False,
    )
    beam1 += beam2
    assert uvutils._check_histories(
        efield_beam.history + "  Downselected to specific parts of "
        "first image axis, parts of second "
        "image axis using pyuvdata. Combined "
        "data along first image, second image "
        "axis using pyuvdata.",
        beam1.history,
    )

    # Zero out missing data in reference object
    beam_ref.data_array[
        :, :, :, : efield_beam.Naxes2 // 2, efield_beam.Naxes1 // 2 :
    ] = 0.0
    beam_ref.data_array[
        :, :, :, efield_beam.Naxes2 // 2 :, : efield_beam.Naxes1 // 2
    ] = 0.0

    beam_ref.basis_vector_array[
        :, :, : efield_beam.Naxes2 // 2, efield_beam.Naxes1 // 2 :
    ] = 0.0
    beam_ref.basis_vector_array[
        :, :, efield_beam.Naxes2 // 2 :, : efield_beam.Naxes1 // 2
    ] = 0.0
    beam1.history = efield_beam.history
    assert beam1, beam_ref


def test_add_warnings(cross_power_beam_for_adding):
    power_beam = cross_power_beam_for_adding

    beam1 = power_beam.select(freq_chans=np.arange(0, 4), inplace=False)
    beam2 = power_beam.select(freq_chans=np.arange(5, 8), inplace=False)
    with uvtest.check_warnings(
        UserWarning, "Combined frequencies are not evenly spaced"
    ):
        beam1.__add__(beam2)

    power_beam.receiver_temperature_array = np.ones((8))
    beam1 = power_beam.select(
        polarizations=power_beam.polarization_array[0:2], inplace=False
    )
    beam2 = power_beam.select(
        polarizations=power_beam.polarization_array[3], inplace=False
    )
    with uvtest.check_warnings(
        UserWarning, "Combined polarizations are not evenly spaced"
    ):
        beam1.__iadd__(beam2)

    beam1 = power_beam.select(
        polarizations=power_beam.polarization_array[0:2], inplace=False
    )
    beam2 = power_beam.select(
        polarizations=power_beam.polarization_array[2:3], inplace=False
    )
    beam2.receiver_temperature_array = None
    assert beam1.receiver_temperature_array is not None
    with uvtest.check_warnings(
        UserWarning,
        "Only one of the UVBeam objects being combined has optional parameter",
    ):
        beam1 += beam2

    assert beam1.receiver_temperature_array is None


@pytest.mark.parametrize("use_double", [True, False])
def test_add_cross_power(cross_power_beam_for_adding, use_double):
    power_beam = cross_power_beam_for_adding
    beam1 = power_beam.select(
        polarizations=power_beam.polarization_array[0:2], inplace=False
    )
    beam2 = power_beam.select(
        polarizations=power_beam.polarization_array[2:4], inplace=False
    )
    if not use_double:
        beam1.data_array = beam1.data_array.astype(np.float32)
        beam2.data_array = beam2.data_array.astype(np.complex64)

    beam2.history += " testing the history. Read/written with pyuvdata"
    new_beam = beam1 + beam2
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Unique part of next "
        "object history follows.  testing the history.",
        new_beam.history,
    )
    new_beam.history = power_beam.history
    assert new_beam == power_beam

    new_beam = beam1.__add__(beam2, verbose_history=True)
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific polarizations using pyuvdata. "
        "Combined data along polarization axis using pyuvdata. Next object history "
        "follows. " + beam2.history,
        new_beam.history,
    )


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
def test_add_errors(power_beam_for_adding, efield_beam_for_adding):
    power_beam = power_beam_for_adding
    efield_beam = efield_beam_for_adding

    # Wrong class
    beam1 = power_beam.copy()
    with pytest.raises(ValueError, match="Only UVBeam "):
        beam1.__iadd__(np.zeros(5))

    params_to_change = {
        "beam_type": "efield",
        "data_normalization": "solid_angle",
        "telescope_name": "foo",
        "feed_name": "foo",
        "feed_version": "v12",
        "model_name": "foo",
        "model_version": "v12",
        "pixel_coordinate_system": "sin_zenith",
        "Naxes_vec": 3,
        "nside": 16,
        "ordering": "nested",
    }

    beam1 = power_beam.select(freq_chans=0, inplace=False)
    beam2 = power_beam.select(freq_chans=1, inplace=False)
    for param, value in params_to_change.items():
        beam1_copy = beam1.copy()
        if param == "beam_type":
            beam2_copy = efield_beam.select(freq_chans=1, inplace=False)
        elif param == "Naxes_vec":
            beam2_copy = beam2.copy()
            beam2_copy.Naxes_vec = value
            beam2_copy.data_array = np.concatenate(
                (beam2_copy.data_array, beam2_copy.data_array, beam2_copy.data_array)
            )
        else:
            beam2_copy = beam2.copy()
            setattr(beam2_copy, param, value)
        with pytest.raises(
            ValueError,
            match=f"UVParameter {param} does not match. Cannot combine objects.",
        ):
            beam1_copy.__iadd__(beam2_copy)

    # different future shapes
    beam1_copy = beam1.copy()
    beam2_copy = beam2.copy()
    beam2_copy.use_current_array_shapes()

    with pytest.raises(
        ValueError,
        match="Both objects must have the same `future_array_shapes` parameter.",
    ):
        beam1_copy.__iadd__(beam2_copy)

    del beam1_copy
    del beam2_copy

    # Overlapping data
    beam2 = power_beam.copy()
    with pytest.raises(
        ValueError, match="These objects have overlapping data and cannot be combined."
    ):
        beam1.__iadd__(beam2)


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_select_healpix_pixels(
    future_shapes,
    beam_type,
    cst_power_1freq_cut_healpix,
    cst_efield_1freq_cut_healpix,
    tmp_path,
):
    if beam_type == "power":
        beam_healpix = cst_power_1freq_cut_healpix
    else:
        beam_healpix = cst_efield_1freq_cut_healpix

    if not future_shapes:
        beam_healpix.use_current_array_shapes()

    old_history = beam_healpix.history
    pixels_to_keep = np.arange(31, 184)

    beam_healpix2 = beam_healpix.select(pixels=pixels_to_keep, inplace=False)

    assert len(pixels_to_keep) == beam_healpix2.Npixels
    for pi in pixels_to_keep:
        assert pi in beam_healpix2.pixel_array
    for pi in np.unique(beam_healpix2.pixel_array):
        assert pi in pixels_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to specific healpix pixels using pyuvdata.",
        beam_healpix2.history,
    )

    write_file_beamfits = str(tmp_path / "select_beam.fits")

    # test writing beamfits with only one pixel
    pixels_to_keep = [43]
    beam_healpix2 = beam_healpix.select(pixels=pixels_to_keep, inplace=False)
    beam_healpix2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with pixels not included in data
    pixel_select = 12 * beam_healpix.nside**2 + 10
    with pytest.raises(
        ValueError,
        match="Pixel {p} is not present in the pixel_array".format(p=pixel_select),
    ):
        beam_healpix.select(pixels=[pixel_select])

    # test writing beamfits with non-contiguous pixels
    pixels_to_keep = np.arange(2, 150, 4)

    beam_healpix2 = beam_healpix.select(pixels=pixels_to_keep, inplace=False)
    beam_healpix2.write_beamfits(write_file_beamfits, clobber=True)

    # -----------------
    # check for errors selecting axis1_inds on healpix beams
    inds1_to_keep = np.arange(14, 63)
    with pytest.raises(
        ValueError, match="axis1_inds cannot be used with healpix coordinate system"
    ):
        beam_healpix.select(axis1_inds=inds1_to_keep)

    # check for errors selecting axis2_inds on healpix beams
    inds2_to_keep = np.arange(5, 14)
    with pytest.raises(
        ValueError, match="axis2_inds cannot be used with healpix coordinate system"
    ):
        beam_healpix.select(axis2_inds=inds2_to_keep)

    # ------------------------
    # test selecting along all axes at once for healpix beams
    if future_shapes:
        freqs_to_keep = [beam_healpix.freq_array[0]]
    else:
        freqs_to_keep = [beam_healpix.freq_array[0, 0]]

    if beam_type == "efield":
        feeds_to_keep = ["x"]
        pols_to_keep = None
    else:
        pols_to_keep = [-5]
        feeds_to_keep = None

    beam_healpix2 = beam_healpix.select(
        pixels=pixels_to_keep,
        frequencies=freqs_to_keep,
        polarizations=pols_to_keep,
        feeds=feeds_to_keep,
        inplace=False,
    )

    assert len(pixels_to_keep) == beam_healpix2.Npixels
    for pi in pixels_to_keep:
        assert pi in beam_healpix2.pixel_array
    for pi in np.unique(beam_healpix2.pixel_array):
        assert pi in pixels_to_keep

    assert len(freqs_to_keep) == beam_healpix2.Nfreqs
    for f in freqs_to_keep:
        assert f in beam_healpix2.freq_array
    for f in np.unique(beam_healpix2.freq_array):
        assert f in freqs_to_keep

    if beam_type == "efield":
        assert len(feeds_to_keep) == beam_healpix2.Nfeeds
        for f in feeds_to_keep:
            assert f in beam_healpix2.feed_array
        for f in np.unique(beam_healpix2.feed_array):
            assert f in feeds_to_keep
    else:
        assert len(pols_to_keep) == beam_healpix2.Npols
        for p in pols_to_keep:
            assert p in beam_healpix2.polarization_array
        for p in np.unique(beam_healpix2.polarization_array):
            assert p in pols_to_keep

    if beam_type == "efield":
        history_add = "feeds"
    else:
        history_add = "polarizations"

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific healpix pixels, frequencies, "
        f"{history_add} using pyuvdata.",
        beam_healpix2.history,
    )


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_select_healpix_pixels_error(
    beam_type, cst_power_2freq_cut, cst_efield_2freq_cut
):
    if beam_type == "power":
        beam = cst_power_2freq_cut
    else:
        beam = cst_efield_2freq_cut
    # check for errors selecting pixels on non-healpix beams
    with pytest.raises(
        ValueError, match="pixels can only be used with healpix coordinate system"
    ):
        beam.select(pixels=np.arange(31, 184))


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_add_healpix(
    future_shapes, beam_type, cst_power_2freq_cut_healpix, cst_efield_2freq_cut_healpix
):
    if beam_type == "power":
        beam_healpix = cst_power_2freq_cut_healpix
    else:
        beam_healpix = cst_efield_2freq_cut_healpix

    if not future_shapes:
        beam_healpix.use_current_array_shapes()

    # Test adding a different combo with healpix
    beam_ref = beam_healpix.copy()
    beam1 = beam_healpix.select(
        pixels=beam_healpix.pixel_array[0 : beam_healpix.Npixels // 2],
        freq_chans=0,
        inplace=False,
    )
    beam2 = beam_healpix.select(
        pixels=beam_healpix.pixel_array[beam_healpix.Npixels // 2 :],
        freq_chans=1,
        inplace=False,
    )
    beam1 += beam2
    assert uvutils._check_histories(
        beam_healpix.history + "  Downselected to specific healpix "
        "pixels, frequencies using pyuvdata. "
        "Combined data along healpix pixel, "
        "frequency axis using pyuvdata.",
        beam1.history,
    )
    # Zero out missing data in reference object
    beam_ref.data_array[..., 0, beam_healpix.Npixels // 2 :] = 0.0
    beam_ref.data_array[..., 1, : beam_healpix.Npixels // 2] = 0.0
    beam1.history = beam_healpix.history
    assert beam1 == beam_ref

    if beam_type == "efield":
        # Test adding another combo with efield
        beam_ref = beam_healpix.copy()
        beam1 = beam_healpix.select(
            freq_chans=0, feeds=beam_healpix.feed_array[0], inplace=False
        )
        beam2 = beam_healpix.select(
            freq_chans=1, feeds=beam_healpix.feed_array[1], inplace=False
        )
        beam1 += beam2
        assert uvutils._check_histories(
            beam_healpix.history + "  Downselected to specific frequencies, "
            "feeds using pyuvdata. Combined data "
            "along frequency, feed axis using pyuvdata.",
            beam1.history,
        )
        # Zero out missing data in reference object
        beam_ref.data_array[..., 1, 0, :] = 0.0
        beam_ref.data_array[..., 0, 1, :] = 0.0
        beam1.history = beam_healpix.history
        assert beam1 == beam_ref

    # Add without inplace
    beam1 = beam_healpix.select(
        pixels=beam_healpix.pixel_array[0 : beam_healpix.Npixels // 2], inplace=False
    )
    beam2 = beam_healpix.select(
        pixels=beam_healpix.pixel_array[beam_healpix.Npixels // 2 :], inplace=False
    )
    beam1 = beam1 + beam2
    assert uvutils._check_histories(
        beam_healpix.history + "  Downselected to specific healpix pixels "
        "using pyuvdata. Combined data "
        "along healpix pixel axis using pyuvdata.",
        beam1.history,
    )
    beam1.history = beam_healpix.history
    assert beam1 == beam_healpix

    # ---------------
    # Test error: adding overlapping data with healpix
    beam1 = beam_healpix.copy()
    beam2 = beam_healpix.copy()
    with pytest.raises(
        ValueError, match="These objects have overlapping data and cannot be combined."
    ):
        beam1.__iadd__(beam2)


@pytest.mark.filterwarnings("ignore:This method will be removed in version 3.0")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_beam_area_healpix(
    future_shapes, cst_power_1freq_cut_healpix, cst_efield_1freq_cut_healpix
):
    power_beam_healpix = cst_power_1freq_cut_healpix

    if not future_shapes:
        power_beam_healpix.use_current_array_shapes()

    # Test beam area methods
    # Check that non-peak normalizations error
    with pytest.raises(ValueError, match="beam must be peak normalized"):
        power_beam_healpix.get_beam_area()
    with pytest.raises(ValueError, match="beam must be peak normalized"):
        power_beam_healpix.get_beam_sq_area()

    healpix_norm = power_beam_healpix.copy()
    healpix_norm.data_normalization = "solid_angle"
    with pytest.raises(ValueError, match="beam must be peak normalized"):
        healpix_norm.get_beam_area()
    with pytest.raises(ValueError, match="beam must be peak normalized"):
        healpix_norm.get_beam_sq_area()

    # change it back to 'physical'
    healpix_norm.data_normalization = "physical"
    # change it to peak for rest of checks
    healpix_norm.peak_normalize()

    # Check sizes of output
    numfreqs = healpix_norm.freq_array.shape[-1]
    beam_int = healpix_norm.get_beam_area(pol="xx")
    beam_sq_int = healpix_norm.get_beam_sq_area(pol="xx")
    assert beam_int.shape[0] == numfreqs
    assert beam_sq_int.shape[0] == numfreqs

    # Check for the case of a uniform beam over the whole sky
    hp_obj = HEALPix(nside=healpix_norm.nside)
    d_omega = hp_obj.pixel_area.to("steradian").value
    npix = healpix_norm.Npixels
    healpix_norm.data_array = np.ones_like(healpix_norm.data_array)
    assert np.allclose(
        np.sum(healpix_norm.get_beam_area(pol="xx")), numfreqs * npix * d_omega
    )
    healpix_norm.data_array = 2.0 * np.ones_like(healpix_norm.data_array)
    assert np.allclose(
        np.sum(healpix_norm.get_beam_sq_area(pol="xx")), numfreqs * 4.0 * npix * d_omega
    )

    # check XX and YY beam areas work and match to within 5 sigfigs
    xx_area = healpix_norm.get_beam_area("XX")
    xx_area = healpix_norm.get_beam_area("xx")
    assert np.allclose(xx_area, xx_area)
    yy_area = healpix_norm.get_beam_area("YY")
    assert np.allclose(yy_area / xx_area, np.ones(numfreqs))
    xx_area = healpix_norm.get_beam_sq_area("XX")
    yy_area = healpix_norm.get_beam_sq_area("YY")
    assert np.allclose(yy_area / xx_area, np.ones(numfreqs))

    # Check that if pseudo-Stokes I (pI) is in the beam polarization_array it
    # just uses it
    healpix_norm.polarization_array = [1, 2]

    # Check error if desired pol is allowed but isn't in the polarization_array
    with pytest.raises(
        ValueError, match="Do not have the right polarization information"
    ):
        healpix_norm.get_beam_area(pol="xx")
    with pytest.raises(
        ValueError, match="Do not have the right polarization information"
    ):
        healpix_norm.get_beam_sq_area(pol="xx")

    # Check polarization error
    healpix_norm.polarization_array = [9, 18, 27, -4]
    with pytest.raises(
        ValueError, match="Do not have the right polarization information"
    ):
        healpix_norm.get_beam_area(pol="xx")
    with pytest.raises(
        ValueError, match="Do not have the right polarization information"
    ):
        healpix_norm.get_beam_sq_area(pol="xx")

    efield_beam = cst_efield_1freq_cut_healpix.copy()
    if not future_shapes:
        efield_beam.use_current_array_shapes()

    healpix_norm_fullpol = efield_beam.efield_to_power(inplace=False)
    healpix_norm_fullpol.peak_normalize()
    xx_area = healpix_norm_fullpol.get_beam_sq_area("XX")
    yy_area = healpix_norm_fullpol.get_beam_sq_area("YY")
    XY_area = healpix_norm_fullpol.get_beam_sq_area("XY")
    YX_area = healpix_norm_fullpol.get_beam_sq_area("YX")
    # check if XY beam area is equal to beam YX beam area
    assert np.allclose(XY_area, YX_area)
    # check if XY/YX beam area is less than XX/YY beam area
    assert np.all(np.less(XY_area, xx_area))
    assert np.all(np.less(XY_area, yy_area))
    assert np.all(np.less(YX_area, xx_area))
    assert np.all(np.less(YX_area, yy_area))

    # Check if power is scalar
    healpix_vec_norm = efield_beam.efield_to_power(
        keep_basis_vector=True, calc_cross_pols=False, inplace=False
    )
    healpix_vec_norm.peak_normalize()
    with pytest.raises(ValueError, match="Expect scalar for power beam, found vector"):
        healpix_vec_norm.get_beam_area()
    with pytest.raises(ValueError, match="Expect scalar for power beam, found vector"):
        healpix_vec_norm.get_beam_sq_area()

    # Check only power beams accepted
    with pytest.raises(ValueError, match="beam_type must be power"):
        efield_beam.get_beam_area()
    with pytest.raises(ValueError, match="beam_type must be power"):
        efield_beam.get_beam_sq_area()

    # check pseudo-Stokes parameters
    efield_beam = cst_efield_1freq_cut_healpix
    if not future_shapes:
        efield_beam.use_current_array_shapes()

    efield_beam.efield_to_pstokes()
    efield_beam.peak_normalize()
    pI_area = efield_beam.get_beam_sq_area("pI")
    pQ_area = efield_beam.get_beam_sq_area("pQ")
    pU_area = efield_beam.get_beam_sq_area("pU")
    pV_area = efield_beam.get_beam_sq_area("pV")
    assert np.all(np.less(pQ_area, pI_area))
    assert np.all(np.less(pU_area, pI_area))
    assert np.all(np.less(pV_area, pI_area))

    # check backwards compatability with pstokes nomenclature and int polnum
    I_area = efield_beam.get_beam_area("I")
    pI_area = efield_beam.get_beam_area("pI")
    area1 = efield_beam.get_beam_area(1)
    assert np.allclose(I_area, pI_area)
    assert np.allclose(I_area, area1)

    # check efield beam type is accepted for pseudo-stokes and power for
    # linear polarizations
    with pytest.raises(ValueError, match="Expect scalar for power beam, found vector"):
        healpix_vec_norm.get_beam_sq_area("pI")
    with pytest.raises(
        ValueError, match="Do not have the right polarization information"
    ):
        efield_beam.get_beam_sq_area("xx")


def test_get_beam_function_errors(cst_power_1freq_cut):
    power_beam = cst_power_1freq_cut.copy()

    with pytest.raises(AssertionError, match="pixel_coordinate_system must be healpix"):
        power_beam._get_beam("xx")

    # Check only healpix accepted (HEALPix checks are in test_healpix)
    # change data_normalization to peak for rest of checks
    power_beam.peak_normalize()
    with pytest.raises(ValueError, match="Currently only healpix format supported"):
        power_beam.get_beam_area()
    with pytest.raises(ValueError, match="Currently only healpix format supported"):
        power_beam.get_beam_sq_area()


def test_get_beam_functions(cst_power_1freq_cut_healpix):
    healpix_power_beam = cst_power_1freq_cut_healpix
    healpix_power_beam.peak_normalize()
    healpix_power_beam._get_beam("xx")
    with pytest.raises(
        ValueError, match="Do not have the right polarization information"
    ):
        healpix_power_beam._get_beam(4)


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.parametrize("future_shapes", [True, False])
def test_generic_read_cst(future_shapes):
    uvb = UVBeam()
    uvb.read(
        cst_files,
        use_future_array_shapes=future_shapes,
        beam_type="power",
        frequency=np.array([150e6, 123e6]),
        feed_pol="y",
        telescope_name="TEST",
        feed_name="bob",
        feed_version="0.1",
        model_name="E-field pattern - Rigging height 4.9m",
        model_version="1.0",
        run_check=False,
    )
    assert uvb.check()


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("filename", [cst_yaml_file, mwa_beam_file, casa_beamfits])
def test_generic_read(filename, future_shapes):
    """Test generic read can infer the file types correctly."""
    uvb = UVBeam()
    # going to check in a second anyway, no need to double check.
    uvb.read(filename, use_future_array_shapes=future_shapes, run_check=False)
    # hera casa beam is missing some parameters but we just want to check
    # that reading is going okay
    if filename == casa_beamfits:
        # fill in missing parameters
        uvb.data_normalization = "peak"
        uvb.feed_name = "casa_ideal"
        uvb.feed_version = "v0"
        uvb.model_name = "casa_airy"
        uvb.model_version = "v0"

        # this file is actually in an orthoslant projection RA/DEC at zenith at a
        # particular time.
        # For now pretend it's in a zenith orthoslant projection
        uvb.pixel_coordinate_system = "orthoslant_zenith"
    assert uvb.check()


def test_generic_read_bad_filetype():
    uvb = UVBeam()
    with pytest.raises(ValueError, match="File type could not be determined"):
        uvb.read("foo")


def test_generic_read_multi(tmp_path):
    uvb = UVBeam()
    uvb.read(
        mwa_beam_file,
        pixels_per_deg=1,
        freq_range=[100e6, 200e6],
        use_future_array_shapes=True,
    )

    uvb1 = uvb.select(frequencies=uvb.freq_array[::2], inplace=False)
    uvb2 = uvb.select(frequencies=uvb.freq_array[1::2], inplace=False)
    fname1 = str(tmp_path / "test_beam1.beamfits")
    fname2 = str(tmp_path / "test_beam2.beamfits")
    uvb1.write_beamfits(fname1)
    uvb2.write_beamfits(fname2)

    uvb3 = UVBeam()
    uvb3.read([fname1, fname2], use_future_array_shapes=True)
    assert uvb3.filename == ["test_beam1.beamfits", "test_beam2.beamfits"]
    # the histories will be different
    uvb3.history = uvb.history

    assert uvb3 == uvb


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("flip_order", [True, False])
def test_generic_read_multi_bad_files(tmp_path, skip, flip_order):
    uvb = UVBeam()
    uvb = UVBeam()
    uvb.read(
        mwa_beam_file,
        pixels_per_deg=1,
        freq_range=[100e6, 200e6],
        use_future_array_shapes=True,
    )

    uvb1 = uvb.select(frequencies=uvb.freq_array[::2], inplace=False)
    uvb2 = uvb.select(frequencies=uvb.freq_array[1::2], inplace=False)
    fname1 = str(tmp_path / "test_beam1.beamfits")
    fname2 = str(tmp_path / "test_beam2.beamfits")
    uvb1.write_beamfits(fname1)
    uvb2.write_beamfits(fname2)

    # Give file a bad beam type
    fits.setval(fname1, "BTYPE", value="foobar")
    uvb3 = UVBeam()
    filenames = [fname1, fname2]
    if flip_order:
        # reverse the order to trigger other try/catch block
        filenames = list(reversed(filenames))
    if skip:
        with uvtest.check_warnings(
            UserWarning, f"Failed to read {filenames[0]} due to ValueError"
        ):
            uvb3.read(filenames, skip_bad_files=skip, use_future_array_shapes=True)
        assert uvb3 == uvb2

    else:
        with pytest.raises(ValueError, match="Unknown beam_type: foobar, beam_type"):
            uvb3.read(filenames, skip_bad_files=skip, use_future_array_shapes=True)


def test_generic_read_all_bad_files(tmp_path):
    uvb = UVBeam()
    uvb = UVBeam()
    uvb.read(
        mwa_beam_file,
        pixels_per_deg=1,
        freq_range=[100e6, 200e6],
        use_future_array_shapes=True,
    )

    uvb1 = uvb.select(frequencies=uvb.freq_array[::2], inplace=False)
    uvb2 = uvb.select(frequencies=uvb.freq_array[1::2], inplace=False)
    fname1 = str(tmp_path / "test_beam1.beamfits")
    fname2 = str(tmp_path / "test_beam2.beamfits")
    uvb1.write_beamfits(fname1)
    uvb2.write_beamfits(fname2)
    # Give files a bad beam type
    fits.setval(fname1, "BTYPE", value="foobar")
    fits.setval(fname2, "BTYPE", value="foobar")
    uvb3 = UVBeam()
    filenames = [fname1, fname2]
    with uvtest.check_warnings(UserWarning, "ALL FILES FAILED ON READ"):
        uvb3.read(filenames, skip_bad_files=True, use_future_array_shapes=True)


@pytest.mark.filterwarnings("ignore:" + _future_array_shapes_warning)
@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("filename", [cst_yaml_file, mwa_beam_file, casa_beamfits])
def test_from_file(future_shapes, filename):
    """Test from file produces same the results as reading explicitly."""
    uvb = UVBeam()
    # don't run checks because of casa_beamfits, we'll do that later
    uvb2 = UVBeam.from_file(
        filename, use_future_array_shapes=future_shapes, run_check=False
    )
    uvb.read(filename, use_future_array_shapes=future_shapes, run_check=False)
    # hera casa beam is missing some parameters but we just want to check
    # that reading is going okay
    if filename == casa_beamfits:
        # fill in missing parameters
        for _uvb in [uvb, uvb2]:
            _uvb.data_normalization = "peak"
            _uvb.feed_name = "casa_ideal"
            _uvb.feed_version = "v0"
            _uvb.model_name = "casa_airy"
            _uvb.model_version = "v0"

            # this file is actually in an orthoslant projection RA/DEC at zenith at a
            # particular time.
            # For now pretend it's in a zenith orthoslant projection
            _uvb.pixel_coordinate_system = "orthoslant_zenith"
    # double check the files are valid
    assert uvb.check()
    assert uvb2.check()
    assert uvb == uvb2
