# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for uvbeam object.

"""
import os
import copy

import numpy as np
from astropy import units
from astropy.coordinates import Angle
import pytest

from pyuvdata import UVBeam
import pyuvdata.tests as uvtest
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH


try:
    from astropy_healpix import HEALPix

    healpix_installed = True
except (ImportError):
    healpix_installed = False


@pytest.fixture(scope="function")
def uvbeam_data():
    """Setup and teardown for basic parameter, property and iterator tests."""
    required_parameters = [
        "_beam_type",
        "_Nfreqs",
        "_Naxes_vec",
        "_Nspws",
        "_pixel_coordinate_system",
        "_freq_array",
        "_spw_array",
        "_data_normalization",
        "_data_array",
        "_bandpass_array",
        "_telescope_name",
        "_feed_name",
        "_feed_version",
        "_model_name",
        "_model_version",
        "_history",
        "_antenna_type",
    ]

    required_properties = [
        "beam_type",
        "Nfreqs",
        "Naxes_vec",
        "Nspws",
        "pixel_coordinate_system",
        "freq_array",
        "spw_array",
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
    ]

    extra_parameters = [
        "_Naxes1",
        "_Naxes2",
        "_Npixels",
        "_Nfeeds",
        "_Npols",
        "_Ncomponents_vec",
        "_axis1_array",
        "_axis2_array",
        "_nside",
        "_ordering",
        "_pixel_array",
        "_feed_array",
        "_polarization_array",
        "_basis_vector_array",
        "_extra_keywords",
        "_Nelements",
        "_element_coordinate_system",
        "_element_location_array",
        "_delay_array",
        "_x_orientation",
        "_interpolation_function",
        "_freq_interp_kind",
        "_gain_array",
        "_coupling_matrix",
        "_reference_impedance",
        "_receiver_temperature_array",
        "_loss_array",
        "_mismatch_array",
        "_s_parameters",
    ]

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
        "interpolation_function",
        "freq_interp_kind",
        "gain_array",
        "coupling_matrix",
        "reference_impedance",
        "receiver_temperature_array",
        "loss_array",
        "mismatch_array",
        "s_parameters",
    ]

    other_properties = ["pyuvdata_version_str"]

    beam_obj = UVBeam()

    class DataHolder:
        def __init__(
            self,
            beam_obj,
            required_parameters,
            required_properties,
            extra_parameters,
            extra_properties,
            other_properties,
        ):
            self.beam_obj = beam_obj
            self.required_parameters = required_parameters
            self.required_properties = required_properties
            self.extra_parameters = extra_parameters
            self.extra_properties = extra_properties
            self.other_properties = other_properties

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


def test_deprecation_warnings_set_cs_params(cst_efield_2freq):
    """
    Test the deprecation warnings in set_cs_params.
    """
    efield_beam = cst_efield_2freq
    efield_beam2 = efield_beam.copy()

    with uvtest.check_warnings(
        DeprecationWarning, match="`set_cs_params` is deprecated"
    ):
        efield_beam2.set_cs_params()

    assert efield_beam2 == efield_beam


def test_deprecation_warnings_set_efield(cst_efield_2freq):
    """
    Test the deprecation warnings in set_efield.
    """
    efield_beam = cst_efield_2freq
    efield_beam2 = efield_beam.copy()

    with uvtest.check_warnings(DeprecationWarning, match="`set_efield` is deprecated"):
        efield_beam2.set_efield()

    assert efield_beam2 == efield_beam


def test_deprecation_warnings_set_power(cst_power_2freq):
    """
    Test the deprecation warnings in set_power.
    """
    power_beam = cst_power_2freq
    power_beam2 = power_beam.copy()

    with uvtest.check_warnings(DeprecationWarning, match="`set_power` is deprecated"):
        power_beam2.set_power()

    assert power_beam2 == power_beam


def test_deprecation_warnings_set_antenna_type(cst_efield_2freq):
    """
    Test the deprecation warnings in set_simple and set_phased_array.
    """
    efield_beam = cst_efield_2freq
    efield_beam2 = efield_beam.copy()

    with uvtest.check_warnings(DeprecationWarning, match="`set_simple` is deprecated"):
        efield_beam2.set_simple()

    assert efield_beam2 == efield_beam

    efield_beam._set_phased_array()
    with uvtest.check_warnings(
        DeprecationWarning, match="`set_phased_array` is deprecated"
    ):
        efield_beam2.set_phased_array()

    assert efield_beam2 == efield_beam


def test_errors():
    beam_obj = UVBeam()
    with pytest.raises(ValueError, match="filetype must be beamfits"):
        beam_obj._convert_to_filetype("foo")


def test_peak_normalize(cst_efield_2freq, cst_power_2freq):
    efield_beam = cst_efield_2freq

    orig_bandpass_array = copy.deepcopy(efield_beam.bandpass_array)
    maxima = np.zeros(efield_beam.Nfreqs)
    for freq_i in range(efield_beam.Nfreqs):
        maxima[freq_i] = np.amax(abs(efield_beam.data_array[:, :, :, freq_i]))
    efield_beam.peak_normalize()
    assert np.amax(abs(efield_beam.data_array)) == 1
    assert np.sum(abs(efield_beam.bandpass_array - orig_bandpass_array * maxima)) == 0
    assert efield_beam.data_normalization == "peak"

    power_beam = cst_power_2freq

    orig_bandpass_array = copy.deepcopy(power_beam.bandpass_array)
    maxima = np.zeros(efield_beam.Nfreqs)
    for freq_i in range(efield_beam.Nfreqs):
        maxima[freq_i] = np.amax(power_beam.data_array[:, :, :, freq_i])
    power_beam.peak_normalize()
    assert np.amax(abs(power_beam.data_array)) == 1
    assert np.sum(abs(power_beam.bandpass_array - orig_bandpass_array * maxima)) == 0
    assert power_beam.data_normalization == "peak"

    power_beam.data_normalization = "solid_angle"
    with pytest.raises(
        NotImplementedError,
        match="Conversion from solid_angle to peak "
        "normalization is not yet implemented",
    ):
        power_beam.peak_normalize()


def test_stokes_matrix():
    beam = UVBeam()
    with pytest.raises(ValueError, match="n must be positive integer."):
        beam._stokes_matrix(-2)
    with pytest.raises(ValueError, match="n should lie between 0 and 3."):
        beam._stokes_matrix(5)


def test_efield_to_pstokes(cst_efield_2freq_cut, cst_efield_2freq_cut_healpix):
    pstokes_beam_2 = cst_efield_2freq_cut_healpix
    # convert to pstokes after interpolating
    beam_return = pstokes_beam_2.efield_to_pstokes(inplace=False)

    pstokes_beam = cst_efield_2freq_cut

    # interpolate after converting to pstokes
    pstokes_beam.interpolation_function = "az_za_simple"
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


def test_efield_to_power(cst_efield_2freq_cut, cst_power_2freq_cut, tmp_path):
    efield_beam = cst_efield_2freq_cut
    power_beam = cst_power_2freq_cut

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
        efield_beam2.interpolation_function = "az_za_simple"
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
    new_data[0, :, :, :, :, :] = (
        efield_beam.data_array[0, :, :, :, :, :]
        - efield_beam.data_array[1, :, :, :, :, :]
    )
    new_data[1, :, :, :, :, :] = np.sqrt(2) * efield_beam.data_array[1, :, :, :, :, :]
    efield_beam2 = efield_beam.copy()
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = efield_beam2.copy()
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2

    # now construct a beam with  orthogonal but rotated basis vectors
    new_basis_vecs = np.zeros_like(efield_beam.basis_vector_array)
    new_basis_vecs[0, 0, :, :] = np.sqrt(0.5)
    new_basis_vecs[0, 1, :, :] = np.sqrt(0.5)
    new_basis_vecs[1, 0, :, :] = -1 * np.sqrt(0.5)
    new_basis_vecs[1, 1, :, :] = np.sqrt(0.5)
    new_data = np.zeros_like(efield_beam.data_array)
    new_data[0, :, :, :, :, :] = np.sqrt(0.5) * (
        efield_beam.data_array[0, :, :, :, :, :]
        + efield_beam.data_array[1, :, :, :, :, :]
    )
    new_data[1, :, :, :, :, :] = np.sqrt(0.5) * (
        -1 * efield_beam.data_array[0, :, :, :, :, :]
        + efield_beam.data_array[1, :, :, :, :, :]
    )
    efield_beam2 = efield_beam.copy()
    efield_beam2.basis_vector_array = new_basis_vecs
    efield_beam2.data_array = new_data
    efield_beam2.check()
    # now convert to power. Should get the same result
    new_power_beam2 = efield_beam2.copy()
    new_power_beam2.efield_to_power(calc_cross_pols=False)

    assert new_power_beam == new_power_beam2

    # test calculating cross pols
    new_power_beam = efield_beam.efield_to_power(calc_cross_pols=True, inplace=False)
    assert np.all(
        np.abs(
            new_power_beam.data_array[
                :, :, 0, :, :, np.where(new_power_beam.axis1_array == 0)[0]
            ]
        )
        > np.abs(
            new_power_beam.data_array[
                :, :, 2, :, :, np.where(new_power_beam.axis1_array == 0)[0]
            ]
        )
    )
    assert np.all(
        np.abs(
            new_power_beam.data_array[
                :, :, 0, :, :, np.where(new_power_beam.axis1_array == np.pi / 2.0)[0]
            ]
        )
        > np.abs(
            new_power_beam.data_array[
                :, :, 2, :, :, np.where(new_power_beam.axis1_array == np.pi / 2.0)[0]
            ]
        )
    )
    # test writing out & reading back in power files (with cross pols which are complex)
    write_file = str(tmp_path / "outtest_beam.fits")
    new_power_beam.write_beamfits(write_file, clobber=True)
    new_power_beam2 = UVBeam()
    new_power_beam2.read_beamfits(write_file)
    assert new_power_beam == new_power_beam2

    # test keeping basis vectors
    new_power_beam = efield_beam.efield_to_power(
        calc_cross_pols=False, keep_basis_vector=True, inplace=False
    )
    assert np.allclose(new_power_beam.data_array, np.abs(efield_beam.data_array) ** 2)

    # test raises error if beam is already a power beam
    with pytest.raises(ValueError, match="beam_type must be efield"):
        power_beam.efield_to_power()

    # test raises error if input efield beam has Naxes_vec=3
    efield_beam.Naxes_vec = 3
    with pytest.raises(
        ValueError,
        match="Conversion to power with 3-vector efields " "is not currently supported",
    ):
        efield_beam.efield_to_power()


def test_freq_interpolation(cst_power_2freq):
    power_beam = cst_power_2freq

    power_beam.interpolation_function = "az_za_simple"

    # test frequency interpolation returns data arrays for small and large tolerances
    freq_orig_vals = np.array([123e6, 150e6])
    interp_data, interp_basis_vector, interp_bandpass = power_beam.interp(
        freq_array=freq_orig_vals, freq_interp_tol=0.0, return_bandpass=True
    )
    assert isinstance(interp_data, np.ndarray)
    assert isinstance(interp_bandpass, np.ndarray)
    np.testing.assert_array_almost_equal(power_beam.bandpass_array, interp_bandpass)
    np.testing.assert_array_almost_equal(power_beam.data_array, interp_data)
    assert interp_basis_vector is None

    interp_data, interp_basis_vector, interp_bandpass = power_beam.interp(
        freq_array=freq_orig_vals, freq_interp_tol=1.0, return_bandpass=True
    )
    assert isinstance(interp_data, np.ndarray)
    assert isinstance(interp_bandpass, np.ndarray)
    np.testing.assert_array_almost_equal(power_beam.bandpass_array, interp_bandpass)
    np.testing.assert_array_almost_equal(power_beam.data_array, interp_data)
    assert interp_basis_vector is None

    # test frequency interpolation returns new UVBeam for small and large tolerances
    power_beam.saved_interp_functions = {}
    new_beam_obj = power_beam.interp(
        freq_array=freq_orig_vals, freq_interp_tol=0.0, new_object=True
    )
    assert isinstance(new_beam_obj, UVBeam)
    np.testing.assert_array_almost_equal(new_beam_obj.freq_array[0], freq_orig_vals)
    assert new_beam_obj.freq_interp_kind == "linear"
    # test that saved functions are erased in new obj
    assert not hasattr(new_beam_obj, "saved_interp_functions")
    assert power_beam.history != new_beam_obj.history
    new_beam_obj.history = power_beam.history
    assert power_beam == new_beam_obj

    new_beam_obj = power_beam.interp(
        freq_array=freq_orig_vals, freq_interp_tol=1.0, new_object=True
    )
    assert isinstance(new_beam_obj, UVBeam)
    np.testing.assert_array_almost_equal(new_beam_obj.freq_array[0], freq_orig_vals)
    # assert interp kind is 'nearest' when within tol
    assert new_beam_obj.freq_interp_kind == "nearest"
    new_beam_obj.freq_interp_kind = "linear"
    assert power_beam.history != new_beam_obj.history
    new_beam_obj.history = power_beam.history
    assert power_beam == new_beam_obj

    # test frequency interpolation returns valid new UVBeam for different
    # number of freqs from input
    power_beam.saved_interp_functions = {}
    new_beam_obj = power_beam.interp(
        freq_array=np.linspace(123e6, 150e6, num=5),
        freq_interp_tol=0.0,
        new_object=True,
    )

    assert isinstance(new_beam_obj, UVBeam)
    np.testing.assert_array_almost_equal(
        new_beam_obj.freq_array[0], np.linspace(123e6, 150e6, num=5)
    )
    assert new_beam_obj.freq_interp_kind == "linear"
    # test that saved functions are erased in new obj
    assert not hasattr(new_beam_obj, "saved_interp_functions")
    assert power_beam.history != new_beam_obj.history
    new_beam_obj.history = power_beam.history

    # down select to orig freqs and test equality
    new_beam_obj.select(frequencies=freq_orig_vals)
    assert power_beam.history != new_beam_obj.history
    new_beam_obj.history = power_beam.history
    assert power_beam == new_beam_obj

    # using only one freq chan should trigger a ValueError if interp_bool is True
    # unless requesting the original frequency channel such that interp_bool is False.
    # Therefore, to test that interp_bool is False returns array slice as desired,
    # test that ValueError is not raised in this case.
    # Other ways of testing this (e.g. interp_data_array.flags['OWNDATA']) does not work
    _pb = power_beam.select(frequencies=power_beam.freq_array[0, :1], inplace=False)
    try:
        interp_data, interp_basis_vector = _pb.interp(freq_array=_pb.freq_array[0])
    except ValueError:
        raise AssertionError("UVBeam.interp didn't return an array slice as expected")

    # test errors if one frequency
    power_beam_singlef = power_beam.select(freq_chans=[0], inplace=False)
    with pytest.raises(
        ValueError, match="Only one frequency in UVBeam so cannot interpolate."
    ):
        power_beam_singlef.interp(freq_array=np.array([150e6]))

    # assert freq_interp_kind ValueError
    power_beam.interpolation_function = "az_za_simple"
    power_beam.freq_interp_kind = None
    with pytest.raises(
        ValueError, match="freq_interp_kind must be set on object first"
    ):
        power_beam.interp(
            az_array=power_beam.axis1_array,
            za_array=power_beam.axis2_array,
            freq_array=freq_orig_vals,
            polarizations=["xx"],
        )


def test_freq_interp_real_and_complex(cst_power_2freq):
    # test interpolation of real and complex data are the same
    power_beam = cst_power_2freq

    power_beam.interpolation_function = "az_za_simple"

    # make a new object with more frequencies
    freqs = np.linspace(123e6, 150e6, 4)
    power_beam.freq_interp_kind = "linear"
    pbeam = power_beam.interp(freq_array=freqs, new_object=True)

    # modulate the data
    pbeam.data_array[:, :, :, 1] *= 2
    pbeam.data_array[:, :, :, 2] *= 0.5

    # interpolate cubic on real data
    freqs = np.linspace(123e6, 150e6, 10)
    pbeam.freq_interp_kind = "cubic"
    pb_int = pbeam.interp(freq_array=freqs)[0]

    # interpolate cubic on complex data and compare to ensure they are the same
    pbeam.data_array = pbeam.data_array.astype(np.complex128)
    pb_int2 = pbeam.interp(freq_array=freqs)[0]
    assert np.all(np.isclose(np.abs(pb_int - pb_int2), 0))


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_spatial_interpolation_samepoints(
    beam_type, cst_power_2freq_cut, cst_efield_2freq_cut
):
    """
    check that interpolating to existing points gives the same answer
    """
    if beam_type == "power":
        uvbeam = cst_power_2freq_cut
    else:
        uvbeam = cst_efield_2freq_cut

    za_orig_vals, az_orig_vals = np.meshgrid(uvbeam.axis2_array, uvbeam.axis1_array)
    az_orig_vals = az_orig_vals.ravel(order="C")
    za_orig_vals = za_orig_vals.ravel(order="C")
    freq_orig_vals = np.array([123e6, 150e6])

    # test error if no interpolation function is set
    with pytest.raises(
        ValueError, match="interpolation_function must be set on object first"
    ):
        uvbeam.interp(
            az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals,
        )

    uvbeam.interpolation_function = "az_za_simple"
    interp_data_array, interp_basis_vector = uvbeam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )

    interp_data_array = interp_data_array.reshape(uvbeam.data_array.shape, order="F")
    assert np.allclose(uvbeam.data_array, interp_data_array)
    if beam_type == "efield":
        interp_basis_vector = interp_basis_vector.reshape(
            uvbeam.basis_vector_array.shape, order="F"
        )
        assert np.allclose(uvbeam.basis_vector_array, interp_basis_vector)

    # test that new object from interpolation is identical
    new_beam = uvbeam.interp(
        az_array=uvbeam.axis1_array,
        za_array=uvbeam.axis2_array,
        az_za_grid=True,
        freq_array=freq_orig_vals,
        new_object=True,
    )
    assert new_beam.freq_interp_kind == "nearest"
    assert new_beam.history == (
        uvbeam.history + " Interpolated in "
        "frequency and to a new azimuth/zenith "
        "angle grid using pyuvdata with "
        "interpolation_function = az_za_simple "
        "and freq_interp_kind = nearest."
    )
    # make histories & freq_interp_kind equal
    new_beam.history = uvbeam.history
    new_beam.freq_interp_kind = "linear"
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

        data_array_compare = uvbeam.data_array[:, :, :1]
        interp_data_array = interp_data_array.reshape(
            data_array_compare.shape, order="F"
        )
        assert np.allclose(data_array_compare, interp_data_array)


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_spatial_interpolation_everyother(
    beam_type, cst_power_2freq_cut, cst_efield_2freq_cut
):
    """
    test that interp to every other point returns an object that matches a select
    """
    if beam_type == "power":
        uvbeam = cst_power_2freq_cut
    else:
        uvbeam = cst_efield_2freq_cut
    uvbeam.interpolation_function = "az_za_simple"

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

    interp_data_array, interp_basis_vector = uvbeam.interp(
        az_array=az_interp_vals, za_array=za_interp_vals, freq_array=freq_interp_vals
    )

    if beam_type == "power":
        # Test requesting separate polarizations on different calls
        # while reusing splines.
        interp_data_array, interp_basis_vector = uvbeam.interp(
            az_array=az_interp_vals[:2],
            za_array=za_interp_vals[:2],
            freq_array=freq_interp_vals,
            polarizations=["xx"],
            reuse_spline=True,
        )

        interp_data_array, interp_basis_vector = uvbeam.interp(
            az_array=az_interp_vals[:2],
            za_array=za_interp_vals[:2],
            freq_array=freq_interp_vals,
            polarizations=["yy"],
            reuse_spline=True,
        )

    # test reusing the spline fit.
    orig_data_array, interp_basis_vector = uvbeam.interp(
        az_array=az_interp_vals,
        za_array=za_interp_vals,
        freq_array=freq_interp_vals,
        reuse_spline=True,
    )

    reused_data_array, interp_basis_vector = uvbeam.interp(
        az_array=az_interp_vals,
        za_array=za_interp_vals,
        freq_array=freq_interp_vals,
        reuse_spline=True,
    )
    assert np.all(reused_data_array == orig_data_array)

    # test passing spline options
    spline_opts = {"kx": 4, "ky": 4}
    quartic_data_array, interp_basis_vector = uvbeam.interp(
        az_array=az_interp_vals,
        za_array=za_interp_vals,
        freq_array=freq_interp_vals,
        spline_opts=spline_opts,
    )

    # slightly different interpolation, so not identical.
    assert np.allclose(quartic_data_array, orig_data_array, atol=1e-10)
    assert not np.all(quartic_data_array == orig_data_array)

    select_data_array_orig, interp_basis_vector = uvbeam.interp(
        az_array=az_interp_vals[0:1],
        za_array=za_interp_vals[0:1],
        freq_array=np.array([127e6]),
    )

    select_data_array_reused, interp_basis_vector = uvbeam.interp(
        az_array=az_interp_vals[0:1],
        za_array=za_interp_vals[0:1],
        freq_array=np.array([127e6]),
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
    uvbeam.interpolation_function = "az_za_simple"

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
    """
    test that interp to every other point returns an object that matches a select
    """
    uvbeam = cst_power_2freq_cut
    uvbeam.interpolation_function = "az_za_simple"

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
            az_array=az_interp_vals,
            za_array=za_interp_vals,
            freq_array=np.array([100]),
        )

    # test errors if positions outside range
    with pytest.raises(
        ValueError,
        match="at least one interpolation location "
        "is outside of the UVBeam pixel coverage.",
    ):
        uvbeam.interp(
            az_array=az_interp_vals, za_array=za_interp_vals + np.pi / 2,
        )

    # test no errors only frequency interpolation
    interp_data_array, interp_basis_vector = uvbeam.interp(freq_array=freq_interp_vals)

    # assert polarization value error
    with pytest.raises(
        ValueError,
        match="Requested polarization 1 not found in self.polarization_array",
    ):
        uvbeam.interp(
            az_array=az_interp_vals, za_array=za_interp_vals, polarizations=["pI"],
        )


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_interp_longitude_branch_cut(beam_type, cst_efield_2freq, cst_power_2freq):
    if beam_type == "power":
        beam = cst_power_2freq
    else:
        beam = cst_efield_2freq

    beam.interpolation_function = "az_za_simple"
    interp_data_array, interp_basis_vector = beam.interp(
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
        beam.Naxes_vec, beam.Nspws, npol_feed, beam.Nfreqs, 4, beam.Naxes2
    )

    assert np.allclose(
        interp_data_array[:, :, :, :, 0, :],
        interp_data_array[:, :, :, :, 1, :],
        rtol=beam._data_array.tols[0],
        atol=beam._data_array.tols[1],
    )

    assert np.allclose(
        interp_data_array[:, :, :, :, 2, :],
        interp_data_array[:, :, :, :, 3, :],
        rtol=beam._data_array.tols[0],
        atol=beam._data_array.tols[1],
    )


def test_interp_healpix_nside(cst_efield_2freq_cut, cst_efield_2freq_cut_healpix):
    efield_beam = cst_efield_2freq_cut

    efield_beam.interpolation_function = "az_za_simple"

    # test calling interp with healpix parameters directly gives same result
    min_res = np.min(
        np.array(
            [np.diff(efield_beam.axis1_array)[0], np.diff(efield_beam.axis2_array)[0]]
        )
    )
    nside_min_res = np.sqrt(3 / np.pi) * np.radians(60.0) / min_res
    nside = int(2 ** np.ceil(np.log2(nside_min_res)))

    new_efield_beam = cst_efield_2freq_cut_healpix
    assert new_efield_beam.nside == nside

    new_efield_beam.interpolation_function = "healpix_simple"

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


def test_healpix_interpolation(cst_efield_2freq):
    pytest.importorskip("astropy_healpix")
    efield_beam = cst_efield_2freq

    efield_beam.interpolation_function = "az_za_simple"

    # select every fourth point to make it smaller
    axis1_inds = np.arange(0, efield_beam.Naxes1, 4)
    axis2_inds = np.arange(0, efield_beam.Naxes2, 4)
    efield_beam.select(axis1_inds=axis1_inds, axis2_inds=axis2_inds)

    orig_efield_beam = efield_beam.copy()

    efield_beam.to_healpix()

    # check that interpolating to existing points gives the same answer
    efield_beam.interpolation_function = "healpix_simple"
    hp_obj = HEALPix(nside=efield_beam.nside)
    hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(efield_beam.pixel_array)
    za_orig_vals = (Angle(np.pi / 2, units.radian) - hpx_lat).radian
    az_orig_vals = hpx_lon.radian

    az_orig_vals = az_orig_vals.ravel(order="C")
    za_orig_vals = za_orig_vals.ravel(order="C")
    freq_orig_vals = np.array([123e6, 150e6])

    interp_data_array, interp_basis_vector = efield_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )
    data_array_compare = efield_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order="F")
    assert np.allclose(data_array_compare, interp_data_array)

    # test that interp to every other point returns an object that matches a select
    pixel_inds = np.arange(0, efield_beam.Npixels, 2)
    select_beam = efield_beam.select(pixels=pixel_inds, inplace=False)
    interp_beam = efield_beam.interp(
        healpix_inds=efield_beam.pixel_array[pixel_inds],
        healpix_nside=efield_beam.nside,
        new_object=True,
    )
    assert select_beam.history != interp_beam.history
    interp_beam.history = select_beam.history
    assert select_beam == interp_beam

    # test interp from healpix to regular az/za grid
    new_reg_beam = efield_beam.interp(
        az_array=orig_efield_beam.axis1_array,
        za_array=orig_efield_beam.axis2_array,
        az_za_grid=True,
        new_object=True,
    )

    # this diff is pretty large. 2 rounds of interpolation is not a good thing.
    # but we can check that the rest of the object makes sense
    diff = new_reg_beam.data_array - orig_efield_beam.data_array
    diff_ratio = diff / orig_efield_beam.data_array
    assert np.all(np.abs(diff_ratio) < 4)
    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [4, 0]
    new_reg_beam._data_array.tols = tols
    assert new_reg_beam.history != orig_efield_beam.history
    new_reg_beam.history = orig_efield_beam.history
    new_reg_beam.interpolation_function = "az_za_simple"
    assert new_reg_beam == orig_efield_beam

    # test errors with specifying healpix_inds without healpix_nside
    hp_obj = HEALPix(nside=efield_beam.nside)
    with pytest.raises(
        ValueError, match="healpix_nside must be set if healpix_inds is set"
    ):
        efield_beam.interp(
            healpix_inds=np.arange(hp_obj.npix), freq_array=freq_orig_vals
        )

    # test error setting both healpix_nside and az_array
    with pytest.raises(
        ValueError,
        match="healpix_nside and healpix_inds can not be set if az_array or "
        "za_array is set.",
    ):
        efield_beam.interp(
            healpix_nside=efield_beam.nside,
            az_array=az_orig_vals,
            za_array=za_orig_vals,
            freq_array=freq_orig_vals,
        )

    # basis_vector exception
    efield_beam.basis_vector_array[0, 1, :] = 10.0
    with pytest.raises(
        NotImplementedError,
        match="interpolation for input basis vectors that are not aligned to the "
        "native theta/phi coordinate system is not yet supported",
    ):
        efield_beam.interp(
            az_array=az_orig_vals, za_array=za_orig_vals,
        )

    # now convert to power beam
    power_beam = efield_beam.efield_to_power(inplace=False)
    del efield_beam
    interp_data_array, interp_basis_vector = power_beam.interp(
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
    interp_data_array2, interp_basis_vector2 = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals
    )
    assert np.allclose(interp_data_array, interp_data_array2)

    # assert not feeding az_array gives same answer
    interp_data_array2, interp_basis_vector2 = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals
    )
    assert np.allclose(interp_data_array, interp_data_array2)

    # test requesting polarization gives the same answer
    interp_data_array2, interp_basis_vector2 = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, polarizations=["yy"]
    )
    assert np.allclose(interp_data_array[:, :, 1:2], interp_data_array2[:, :, :1])

    # change complex data_array to real data_array and test again
    assert power_beam.data_array.dtype == np.complex128
    power_beam.data_array = np.abs(power_beam.data_array)
    interp_data_array, interp_basis_vector = power_beam.interp(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )
    data_array_compare = power_beam.data_array
    interp_data_array = interp_data_array.reshape(data_array_compare.shape, order="F")
    assert np.allclose(data_array_compare, interp_data_array)

    # test no inputs equals same answer
    interp_data_array2, interp_basis_vector2 = power_beam.interp()
    assert np.allclose(interp_data_array, interp_data_array2)

    # assert polarization value error
    with pytest.raises(
        ValueError,
        match="Requested polarization 1 not found in self.polarization_array",
    ):
        power_beam.interp(
            az_array=az_orig_vals, za_array=za_orig_vals, polarizations=["pI"]
        )

    # check error when pixels out of order
    power_beam.pixel_array = power_beam.pixel_array[
        np.argsort(power_beam.data_array[0, 0, 0, 0, :])
    ]
    with pytest.raises(
        ValueError,
        match="simple healpix interpolation requires healpix pixels to be in order.",
    ):
        power_beam.interp(az_array=az_orig_vals, za_array=za_orig_vals)

    # healpix coord exception
    power_beam.pixel_coordinate_system = "foo"
    with pytest.raises(ValueError, match='pixel_coordinate_system must be "healpix"'):
        power_beam.interp(az_array=az_orig_vals, za_array=za_orig_vals)


def test_to_healpix(
    cst_power_2freq_cut,
    cst_power_2freq_cut_healpix,
    cst_efield_2freq_cut,
    cst_efield_2freq_cut_healpix,
):
    power_beam = cst_power_2freq_cut
    power_beam_healpix = cst_power_2freq_cut_healpix

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
    power_beam.interpolation_function = "az_za_simple"
    power_beam.pixel_coordinate_system = "sin_zenith"
    with pytest.raises(ValueError, match='pixel_coordinate_system must be "az_za"'):
        power_beam.to_healpix()

    # Now check Efield interpolation
    efield_beam = cst_efield_2freq_cut
    interp_then_sq = cst_efield_2freq_cut_healpix
    interp_then_sq.efield_to_power(calc_cross_pols=False)

    # convert to power and then interpolate to compare.
    # Don't use power read from file because it has rounding errors that will
    # dominate this comparison
    efield_beam.interpolation_function = "az_za_simple"
    sq_then_interp = efield_beam.efield_to_power(calc_cross_pols=False, inplace=False)
    sq_then_interp.to_healpix()

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
        + power_beam.coordinate_system_dict["az_za"]["description"]
        + " to "
        + power_beam.coordinate_system_dict["healpix"]["description"]
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

    assert sq_then_interp == interp_then_sq


def test_select_axis(cst_power_1freq, tmp_path):
    power_beam = cst_power_1freq

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {"KEY1": "test_keyword"}
    power_beam.reference_impedance = 340.0
    power_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.loss_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs)
    )

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


def test_select_frequencies(cst_power_1freq, tmp_path):
    power_beam = cst_power_1freq

    # generate more frequencies for testing by copying and adding several times
    while power_beam.Nfreqs < 8:
        new_beam = power_beam.copy()
        new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
        power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {"KEY1": "test_keyword"}
    power_beam.reference_impedance = 340.0
    power_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.loss_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs)
    )

    old_history = power_beam.history
    freqs_to_keep = power_beam.freq_array[0, np.arange(2, 7)]

    power_beam2 = power_beam.select(frequencies=freqs_to_keep, inplace=False)

    assert len(freqs_to_keep) == power_beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in freqs_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        power_beam2.history,
    )

    write_file_beamfits = str(tmp_path / "select_beam.fits")
    # test writing beamfits with only one frequency

    freqs_to_keep = power_beam.freq_array[0, 5]
    power_beam2 = power_beam.select(frequencies=freqs_to_keep, inplace=False)
    power_beam2.write_beamfits(write_file_beamfits, clobber=True)

    freq_select = np.max(power_beam.freq_array) + 10
    # check for errors associated with frequencies not included in data
    with pytest.raises(
        ValueError,
        match="Frequency {f} is not present in the freq_array".format(f=freq_select),
    ):
        power_beam.select(frequencies=[freq_select])

    # check for warnings and errors associated with unevenly spaced frequencies
    power_beam2 = power_beam.copy()
    with uvtest.check_warnings(
        UserWarning, "Selected frequencies are not evenly spaced"
    ):
        power_beam2.select(frequencies=power_beam2.freq_array[0, [0, 5, 6]])
    with pytest.raises(ValueError, match="The frequencies are not evenly spaced "):
        power_beam2.write_beamfits(write_file_beamfits)

    # Test selecting on freq_chans
    chans_to_keep = np.arange(2, 7)

    power_beam2 = power_beam.select(freq_chans=chans_to_keep, inplace=False)

    assert len(chans_to_keep) == power_beam2.Nfreqs
    for chan in chans_to_keep:
        assert power_beam.freq_array[0, chan] in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in power_beam.freq_array[0, chans_to_keep]

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific frequencies using pyuvdata.",
        power_beam2.history,
    )

    # Test selecting both channels and frequencies
    freqs_to_keep = power_beam.freq_array[0, np.arange(6, 8)]  # Overlaps with chans
    all_chans_to_keep = np.arange(2, 8)

    power_beam2 = power_beam.select(
        frequencies=freqs_to_keep, freq_chans=chans_to_keep, inplace=False
    )

    assert len(all_chans_to_keep) == power_beam2.Nfreqs
    for chan in all_chans_to_keep:
        assert power_beam.freq_array[0, chan] in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in power_beam.freq_array[0, all_chans_to_keep]


def test_select_feeds(cst_efield_1freq):
    efield_beam = cst_efield_1freq

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {"KEY1": "test_keyword"}
    efield_beam.reference_impedance = 340.0
    efield_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.loss_array = np.random.normal(
        50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs)
    )

    old_history = efield_beam.history
    feeds_to_keep = ["x"]

    efield_beam2 = efield_beam.select(feeds=feeds_to_keep, inplace=False)

    assert len(feeds_to_keep) == efield_beam2.Nfeeds
    for f in feeds_to_keep:
        assert f in efield_beam2.feed_array
    for f in np.unique(efield_beam2.feed_array):
        assert f in feeds_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific feeds using pyuvdata.",
        efield_beam2.history,
    )

    # check for errors associated with feeds not included in data
    with pytest.raises(
        ValueError, match="Feed {f} is not present in the feed_array".format(f="N")
    ):
        efield_beam.select(feeds=["N"])

    # check for error with selecting polarizations on efield beams
    with pytest.raises(
        ValueError, match="polarizations cannot be used with efield beams"
    ):
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


def test_select_polarizations(cst_power_1freq):
    power_beam = cst_power_1freq

    # generate more polarizations for testing by copying and adding several times
    while power_beam.Npols < 4:
        new_beam = power_beam.copy()
        new_beam.polarization_array = power_beam.polarization_array - power_beam.Npols
        power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {"KEY1": "test_keyword"}
    power_beam.reference_impedance = 340.0
    power_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.loss_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs)
    )

    old_history = power_beam.history
    pols_to_keep = [-5, -6]

    power_beam2 = power_beam.select(polarizations=pols_to_keep, inplace=False)

    assert len(pols_to_keep) == power_beam2.Npols
    for p in pols_to_keep:
        assert p in power_beam2.polarization_array
    for p in np.unique(power_beam2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific polarizations using pyuvdata.",
        power_beam2.history,
    )

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


def test_select(cst_power_1freq, cst_efield_1freq):
    power_beam = cst_power_1freq

    # generate more frequencies for testing by copying and adding
    new_beam = power_beam.copy()
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {"KEY1": "test_keyword"}
    power_beam.reference_impedance = 340.0
    power_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.loss_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs)
    )

    # now test selecting along all axes at once
    old_history = power_beam.history

    inds1_to_keep = np.arange(14, 63)
    inds2_to_keep = np.arange(5, 14)
    freqs_to_keep = [power_beam.freq_array[0, 0]]
    pols_to_keep = [-5]

    power_beam2 = power_beam.select(
        axis1_inds=inds1_to_keep,
        axis2_inds=inds2_to_keep,
        frequencies=freqs_to_keep,
        polarizations=pols_to_keep,
        inplace=False,
    )

    assert len(inds1_to_keep) == power_beam2.Naxes1
    for i in inds1_to_keep:
        assert power_beam.axis1_array[i] in power_beam2.axis1_array
    for i in np.unique(power_beam2.axis1_array):
        assert i in power_beam.axis1_array

    assert len(inds2_to_keep) == power_beam2.Naxes2
    for i in inds2_to_keep:
        assert power_beam.axis2_array[i] in power_beam2.axis2_array
    for i in np.unique(power_beam2.axis2_array):
        assert i in power_beam.axis2_array

    assert len(freqs_to_keep) == power_beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in power_beam2.freq_array
    for f in np.unique(power_beam2.freq_array):
        assert f in freqs_to_keep

    assert len(pols_to_keep) == power_beam2.Npols
    for p in pols_to_keep:
        assert p in power_beam2.polarization_array
    for p in np.unique(power_beam2.polarization_array):
        assert p in pols_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific parts of first image axis, "
        "parts of second image axis, "
        "frequencies, polarizations using pyuvdata.",
        power_beam2.history,
    )

    # repeat for efield beam
    efield_beam = cst_efield_1freq

    # generate more frequencies for testing by copying and adding
    new_beam = efield_beam.copy()
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {"KEY1": "test_keyword"}
    efield_beam.reference_impedance = 340.0
    efield_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.loss_array = np.random.normal(
        50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs)
    )

    feeds_to_keep = ["x"]

    efield_beam2 = efield_beam.select(
        axis1_inds=inds1_to_keep,
        axis2_inds=inds2_to_keep,
        frequencies=freqs_to_keep,
        feeds=feeds_to_keep,
        inplace=False,
    )

    assert len(inds1_to_keep) == efield_beam2.Naxes1
    for i in inds1_to_keep:
        assert efield_beam.axis1_array[i] in efield_beam2.axis1_array
    for i in np.unique(efield_beam2.axis1_array):
        assert i in efield_beam.axis1_array

    assert len(inds2_to_keep) == efield_beam2.Naxes2
    for i in inds2_to_keep:
        assert efield_beam.axis2_array[i] in efield_beam2.axis2_array
    for i in np.unique(efield_beam2.axis2_array):
        assert i in efield_beam.axis2_array

    assert len(freqs_to_keep) == efield_beam2.Nfreqs
    for f in freqs_to_keep:
        assert f in efield_beam2.freq_array
    for f in np.unique(efield_beam2.freq_array):
        assert f in freqs_to_keep

    assert len(feeds_to_keep) == efield_beam2.Nfeeds
    for f in feeds_to_keep:
        assert f in efield_beam2.feed_array
    for f in np.unique(efield_beam2.feed_array):
        assert f in feeds_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to "
        "specific parts of first image axis, "
        "parts of second image axis, "
        "frequencies, feeds using pyuvdata.",
        efield_beam2.history,
    )


def test_add(cst_power_1freq, cst_efield_1freq):
    power_beam = cst_power_1freq

    # generate more frequencies for testing by copying and adding
    new_beam = power_beam.copy()
    new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
    power_beam += new_beam

    # add optional parameters for testing purposes
    power_beam.extra_keywords = {"KEY1": "test_keyword"}
    power_beam.reference_impedance = 340.0
    power_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.loss_array = np.random.normal(
        50.0, 5, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(power_beam.Nspws, power_beam.Nfreqs)
    )
    power_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, power_beam.Nspws, power_beam.Nfreqs)
    )

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

    # Add feeds
    efield_beam = cst_efield_1freq

    # generate more frequencies for testing by copying and adding
    new_beam = efield_beam.copy()
    new_beam.freq_array = efield_beam.freq_array + efield_beam.Nfreqs * 1e6
    efield_beam += new_beam

    # add optional parameters for testing purposes
    efield_beam.extra_keywords = {"KEY1": "test_keyword"}
    efield_beam.reference_impedance = 340.0
    efield_beam.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.loss_array = np.random.normal(
        50.0, 5, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.mismatch_array = np.random.normal(
        0.0, 1.0, size=(efield_beam.Nspws, efield_beam.Nfreqs)
    )
    efield_beam.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, efield_beam.Nspws, efield_beam.Nfreqs)
    )

    beam1 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    beam2 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    beam1 += beam2
    assert uvutils._check_histories(
        efield_beam.history + "  Downselected to specific feeds "
        "using pyuvdata. Combined data along "
        "feed axis using pyuvdata.",
        beam1.history,
    )
    beam1.history = efield_beam.history
    assert beam1 == efield_beam

    # Out of order - feeds
    beam1 = efield_beam.select(feeds=efield_beam.feed_array[1], inplace=False)
    beam2 = efield_beam.select(feeds=efield_beam.feed_array[0], inplace=False)
    beam1 += beam2
    beam1.history = efield_beam.history
    assert beam1, efield_beam

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
    beam_ref.data_array[:, :, 0, :, :, power_beam.Naxes1 // 2 :] = 0.0
    beam_ref.data_array[:, :, 1, :, :, : power_beam.Naxes1 // 2] = 0.0
    beam1.history = power_beam.history
    assert beam1 == beam_ref

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
        :, :, :, :, : efield_beam.Naxes2 // 2, efield_beam.Naxes1 // 2 :
    ] = 0.0
    beam_ref.data_array[
        :, :, :, :, efield_beam.Naxes2 // 2 :, : efield_beam.Naxes1 // 2
    ] = 0.0

    beam_ref.basis_vector_array[
        :, :, : efield_beam.Naxes2 // 2, efield_beam.Naxes1 // 2 :
    ] = 0.0
    beam_ref.basis_vector_array[
        :, :, efield_beam.Naxes2 // 2 :, : efield_beam.Naxes1 // 2
    ] = 0.0
    beam1.history = efield_beam.history
    assert beam1, beam_ref

    # Check warnings
    # generate more frequencies for testing by copying and adding several times
    while power_beam.Nfreqs < 8:
        new_beam = power_beam.copy()
        new_beam.freq_array = power_beam.freq_array + power_beam.Nfreqs * 1e6
        power_beam += new_beam

    beam1 = power_beam.select(freq_chans=np.arange(0, 4), inplace=False)
    beam2 = power_beam.select(freq_chans=np.arange(5, 8), inplace=False)
    with uvtest.check_warnings(
        UserWarning, "Combined frequencies are not evenly spaced"
    ):
        beam1.__add__(beam2)

    # generate more polarizations for testing by copying and adding several times
    while power_beam.Npols < 4:
        new_beam = power_beam.copy()
        new_beam.polarization_array = power_beam.polarization_array - power_beam.Npols
        power_beam += new_beam

    power_beam.receiver_temperature_array = np.ones((1, 8))
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
        beam1.__iadd__(beam2)

    assert beam1.receiver_temperature_array is None

    # Combining histories
    beam1 = power_beam.select(
        polarizations=power_beam.polarization_array[0:2], inplace=False
    )
    beam2 = power_beam.select(
        polarizations=power_beam.polarization_array[2:4], inplace=False
    )
    beam2.history += " testing the history. Read/written with pyuvdata"
    beam1 += beam2
    assert uvutils._check_histories(
        power_beam.history + "  Downselected to specific polarizations "
        "using pyuvdata. Combined data along "
        "polarization axis using pyuvdata. "
        "testing the history.",
        beam1.history,
    )
    beam1.history = power_beam.history
    assert beam1 == power_beam

    # ------------------------
    # Test failure modes of add function

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
    del beam1_copy
    del beam2_copy

    # Overlapping data
    beam2 = power_beam.copy()
    with pytest.raises(
        ValueError, match="These objects have overlapping data and cannot be combined."
    ):
        beam1.__iadd__(beam2)


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_select_healpix_pixels(
    beam_type, cst_power_1freq_cut_healpix, cst_efield_1freq_cut_healpix, tmp_path
):
    if beam_type == "power":
        beam_healpix = cst_power_1freq_cut_healpix
    else:
        beam_healpix = cst_efield_1freq_cut_healpix

    # add optional parameters for testing purposes
    beam_healpix.extra_keywords = {"KEY1": "test_keyword"}
    beam_healpix.reference_impedance = 340.0
    beam_healpix.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(beam_healpix.Nspws, beam_healpix.Nfreqs)
    )
    beam_healpix.loss_array = np.random.normal(
        50.0, 5, size=(beam_healpix.Nspws, beam_healpix.Nfreqs)
    )
    beam_healpix.mismatch_array = np.random.normal(
        0.0, 1.0, size=(beam_healpix.Nspws, beam_healpix.Nfreqs)
    )
    beam_healpix.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, beam_healpix.Nspws, beam_healpix.Nfreqs)
    )

    old_history = beam_healpix.history
    pixels_to_keep = np.arange(31, 184)

    beam_healpix2 = beam_healpix.select(pixels=pixels_to_keep, inplace=False)

    assert len(pixels_to_keep) == beam_healpix2.Npixels
    for pi in pixels_to_keep:
        assert pi in beam_healpix2.pixel_array
    for pi in np.unique(beam_healpix2.pixel_array):
        assert pi in pixels_to_keep

    assert uvutils._check_histories(
        old_history + "  Downselected to " "specific healpix pixels using pyuvdata.",
        beam_healpix2.history,
    )

    write_file_beamfits = str(tmp_path / "select_beam.fits")

    # test writing beamfits with only one pixel
    pixels_to_keep = [43]
    beam_healpix2 = beam_healpix.select(pixels=pixels_to_keep, inplace=False)
    beam_healpix2.write_beamfits(write_file_beamfits, clobber=True)

    # check for errors associated with pixels not included in data
    pixel_select = 12 * beam_healpix.nside ** 2 + 10
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


@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_add_healpix(
    beam_type, cst_power_2freq_cut_healpix, cst_efield_2freq_cut_healpix
):
    if beam_type == "power":
        beam_healpix = cst_power_2freq_cut_healpix
    else:
        beam_healpix = cst_efield_2freq_cut_healpix

    # add optional parameters for testing purposes
    beam_healpix.extra_keywords = {"KEY1": "test_keyword"}
    beam_healpix.reference_impedance = 340.0
    beam_healpix.receiver_temperature_array = np.random.normal(
        50.0, 5, size=(beam_healpix.Nspws, beam_healpix.Nfreqs)
    )
    beam_healpix.loss_array = np.random.normal(
        50.0, 5, size=(beam_healpix.Nspws, beam_healpix.Nfreqs)
    )
    beam_healpix.mismatch_array = np.random.normal(
        0.0, 1.0, size=(beam_healpix.Nspws, beam_healpix.Nfreqs)
    )
    beam_healpix.s_parameters = np.random.normal(
        0.0, 0.3, size=(4, beam_healpix.Nspws, beam_healpix.Nfreqs)
    )

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
    beam_ref.data_array[:, :, :, 0, beam_healpix.Npixels // 2 :] = 0.0
    beam_ref.data_array[:, :, :, 1, : beam_healpix.Npixels // 2] = 0.0
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
        beam_ref.data_array[:, :, 1, 0, :] = 0.0
        beam_ref.data_array[:, :, 0, 1, :] = 0.0
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


def test_beam_area_healpix(cst_power_1freq_cut_healpix, cst_efield_1freq_cut_healpix):
    power_beam_healpix = cst_power_1freq_cut_healpix

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
    # nt.assert_almost_equal(yy_area / xx_area, 1.0, places=5)
    xx_area = healpix_norm.get_beam_sq_area("XX")
    yy_area = healpix_norm.get_beam_sq_area("YY")
    assert np.allclose(yy_area / xx_area, np.ones(numfreqs))
    # nt.assert_almost_equal(yy_area / xx_area, 1.0, places=5)

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

    efield_beam = cst_efield_1freq_cut_healpix
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
