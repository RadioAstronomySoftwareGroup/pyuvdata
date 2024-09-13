# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import re

import numpy as np
import pytest
from astropy.constants import c as speed_of_light
from scipy.special import j1

from pyuvdata import AiryBeam, GaussianBeam, ShortDipoleBeam, UniformBeam, UVBeam
from pyuvdata.uvbeam.analytic_beam import AnalyticBeam


def test_airy_beam_values(az_za_deg_grid):
    diameter_m = 14.0
    beam = AiryBeam(diameter=diameter_m)

    az_vals, za_vals, freqs = az_za_deg_grid

    beam_vals = beam.efield_eval(az_array=az_vals, za_array=za_vals, freq_array=freqs)
    nsrcs = az_vals.size
    n_freqs = freqs.size

    expected_data = np.zeros((2, 2, n_freqs, nsrcs), dtype=float)
    za_grid, f_grid = np.meshgrid(za_vals, freqs)
    c_ms = speed_of_light.to("m/s").value
    xvals = diameter_m / 2.0 * np.sin(za_grid) * 2.0 * np.pi * f_grid / c_ms
    airy_values = np.zeros_like(xvals)
    nz = xvals != 0.0
    ze = xvals == 0.0
    airy_values[nz] = 2.0 * j1(xvals[nz]) / xvals[nz]
    airy_values[ze] = 1.0
    for pol in range(2):
        for feed in range(2):
            expected_data[pol, feed, :, :] = airy_values / np.sqrt(2.0)

    np.testing.assert_allclose(beam_vals, expected_data)

    assert beam.__repr__() == f"AiryBeam(diameter={diameter_m})"


def test_airy_uv_beam_widths(xy_grid):
    # Check that the width of the Airy disk beam in UV space corresponds with
    # the dish diameter.
    diameter_m = 25.0
    beam = AiryBeam(diameter=diameter_m)

    az_array, za_array, freqs = xy_grid

    wavelengths = speed_of_light.to("m/s").value / freqs

    beam_vals = beam.efield_eval(az_array=az_array, za_array=za_array, freq_array=freqs)

    ebeam = beam_vals[0, 0, :, :]
    npix_side = int(np.sqrt(az_array.size))
    ebeam = ebeam.reshape(freqs.size, npix_side, npix_side)
    beam_kern = np.fft.fft2(ebeam, axes=(1, 2))
    beam_kern = np.fft.fftshift(beam_kern, axes=(1, 2))
    for i, bk in enumerate(beam_kern):
        # Cutoff at half a % of the maximum value in Fourier space.
        thresh = np.max(np.abs(bk)) * 0.005
        points = np.sum(np.abs(bk) >= thresh)
        upix = 1 / (2 * np.sin(np.max(za_array)))
        area = np.sum(points) * upix**2
        kern_radius = np.sqrt(area / np.pi)
        assert np.isclose(diameter_m / wavelengths[i], kern_radius, rtol=0.5)


@pytest.mark.parametrize("sigma_type", ["efield", "power"])
def test_achromatic_gaussian_beam(az_za_deg_grid, sigma_type):
    sigma_rad = np.deg2rad(5)
    beam = GaussianBeam(sigma=sigma_rad, sigma_type=sigma_type)

    az_vals, za_vals, freqs = az_za_deg_grid
    nsrcs = az_vals.size
    n_freqs = freqs.size

    beam_vals = beam.efield_eval(
        az_array=az_vals, za_array=za_vals, freq_array=np.array(freqs)
    )

    expected_data = np.zeros((2, 2, n_freqs, nsrcs), dtype=float)

    expand_za = np.repeat(za_vals[np.newaxis], n_freqs, axis=0)
    if sigma_type == "power":
        sigma_use = np.sqrt(2) * sigma_rad
    else:
        sigma_use = sigma_rad

    gaussian_vals = np.exp(-(expand_za**2) / (2 * sigma_use**2))

    for pol in range(2):
        for feed in range(2):
            expected_data[pol, feed, :, :] = gaussian_vals / np.sqrt(2.0)

    np.testing.assert_allclose(beam_vals, expected_data)

    assert (
        beam.__repr__()
        == f"GaussianBeam(sigma={sigma_use.__repr__()}, sigma_type='efield', "
        "diameter=None, spectral_index=0.0, reference_frequency=1.0)"
    )


def test_chromatic_gaussian():
    """
    Defining a gaussian beam with a spectral index and reference frequency.
    Check that beam width follows prescribed power law.
    """
    freqs = np.arange(120e6, 160e6, 4e6)
    Npix = 1000
    alpha = -1.5
    sigma = np.radians(15.0)

    az = np.zeros(Npix)
    za = np.linspace(0, np.pi / 2.0, Npix)

    # Error if trying to define chromatic beam without a reference frequency
    with pytest.raises(
        ValueError,
        match="reference_frequency must be set if `spectral_index` is not zero.",
    ):
        GaussianBeam(sigma=sigma, spectral_index=alpha)

    beam = GaussianBeam(sigma=sigma, reference_frequency=freqs[0], spectral_index=alpha)

    # Get the widths at each frequency.

    vals = beam.efield_eval(az_array=az, za_array=za, freq_array=freqs)
    # pick out a single polarization direction and feed
    vals = vals[0, 0]

    # The beam peaks at 1/sqrt(2) in each pol. Find where it drops by a factor of 2
    half_power_val = 1 / (2.0 * np.sqrt(2.0))
    hwhm = za[np.argmin(np.abs(vals - half_power_val), axis=1)]
    sig_f = sigma * (freqs / freqs[0]) ** alpha
    np.testing.assert_allclose(sig_f, 2 * hwhm / 2.355, atol=1e-3)

    assert (
        beam.__repr__()
        == f"GaussianBeam(sigma={sigma.__repr__()}, sigma_type='efield', "
        f"diameter=None, spectral_index={alpha}, "
        f"reference_frequency={freqs[0].__repr__()})"
    )


def test_diameter_to_sigma(az_za_deg_grid):
    # The integrals of an Airy power beam and a Gaussian power beam, within
    # the first Airy null, should be close if the Gaussian width is set to the
    # Airy width.
    diameter_m = 25.0
    abm = AiryBeam(diameter=diameter_m)
    gbm = GaussianBeam(diameter=diameter_m)

    assert (
        gbm.__repr__()
        == f"GaussianBeam(sigma=None, sigma_type='efield', diameter={diameter_m}, "
        "spectral_index=0.0, reference_frequency=None)"
    )

    az_array, za_array, freqs = az_za_deg_grid

    wavelengths = speed_of_light.to("m/s").value / freqs

    airy_vals = abm._power_eval(
        az_array=az_array.flatten(), za_array=za_array.flatten(), freq_array=freqs
    )

    gauss_vals = gbm._power_eval(
        az_array=az_array.flatten(), za_array=za_array.flatten(), freq_array=freqs
    )

    # Remove pol/spw/feed axes.
    airy_vals = airy_vals[0, 0]
    gauss_vals = gauss_vals[0, 0]

    for fi in range(freqs.size):
        null = 1.22 * wavelengths[fi] / diameter_m
        inds = np.where(np.abs(za_array) < null)

        # Assert integral of power beams within the first Airy null are close
        np.testing.assert_allclose(
            np.sum(airy_vals[fi, inds]), np.sum(gauss_vals[fi, inds]), rtol=1e-2
        )


def test_short_dipole_beam(az_za_deg_grid):
    beam = ShortDipoleBeam()

    az_vals, za_vals, freqs = az_za_deg_grid

    nsrcs = az_vals.size
    n_freqs = freqs.size

    efield_vals = beam.efield_eval(az_array=az_vals, za_array=za_vals, freq_array=freqs)

    expected_data = np.zeros((2, 2, n_freqs, nsrcs), dtype=float)

    expected_data[0, 0] = -np.sin(az_vals)
    expected_data[0, 1] = np.cos(az_vals)
    expected_data[1, 0] = np.cos(za_vals) * np.cos(az_vals)
    expected_data[1, 1] = np.cos(za_vals) * np.sin(az_vals)

    np.testing.assert_allclose(efield_vals, expected_data)

    power_vals = beam.power_eval(az_array=az_vals, za_array=za_vals, freq_array=freqs)
    print(power_vals.shape)
    expected_data = np.zeros((1, 4, n_freqs, nsrcs), dtype=float)
    print(expected_data.shape)

    expected_data[0, 0] = 1 - np.sin(za_vals) ** 2 * np.cos(az_vals) ** 2
    expected_data[0, 1] = 1 - np.sin(za_vals) ** 2 * np.sin(az_vals) ** 2
    expected_data[0, 2] = -(np.sin(za_vals) ** 2) * np.sin(2.0 * az_vals) / 2.0
    expected_data[0, 3] = -(np.sin(za_vals) ** 2) * np.sin(2.0 * az_vals) / 2.0

    np.testing.assert_allclose(power_vals, expected_data)

    assert beam.__repr__() == "ShortDipoleBeam(x_orientation='east')"


def test_uniform_beam(az_za_deg_grid):
    beam = UniformBeam()

    az_vals, za_vals, freqs = az_za_deg_grid

    nsrcs = az_vals.size
    n_freqs = freqs.size

    beam_vals = beam.efield_eval(az_array=az_vals, za_array=za_vals, freq_array=freqs)

    expected_data = np.ones((2, 2, n_freqs, nsrcs), dtype=float) / np.sqrt(2.0)
    np.testing.assert_allclose(beam_vals, expected_data)

    assert beam.__repr__() == "UniformBeam()"


@pytest.mark.parametrize(
    ["beam_obj", "kwargs"],
    [
        [AiryBeam, {"diameter": 14.0}],
        [GaussianBeam, {"diameter": 14.0}],
        [UniformBeam, {}],
        [ShortDipoleBeam, {}],
    ],
)
def test_power_analytic_beam(beam_obj, kwargs, xy_grid):
    # Check that power beam evaluation matches electric field amp**2 for analytic beams.

    az_vals, za_vals, freqs = xy_grid

    beam = beam_obj(**kwargs)

    efield_vals = beam.efield_eval(az_array=az_vals, za_array=za_vals, freq_array=freqs)
    power_vals = beam.power_eval(az_array=az_vals, za_array=za_vals, freq_array=freqs)

    # check power beams are peak normalized
    assert np.max(power_vals) == 1.0

    atol = 2e-15
    np.testing.assert_allclose(
        efield_vals[0, 0] ** 2 + efield_vals[1, 0] ** 2,
        power_vals[0, 0],
        rtol=0,
        atol=atol,
    )

    np.testing.assert_allclose(
        efield_vals[0, 1] ** 2 + efield_vals[1, 1] ** 2,
        power_vals[0, 1],
        rtol=0,
        atol=atol,
    )

    cross_power = (
        efield_vals[0, 0] * efield_vals[0, 1] + efield_vals[1, 0] * efield_vals[1, 1]
    )
    np.testing.assert_allclose(cross_power, power_vals[0, 2], rtol=0, atol=atol)

    np.testing.assert_allclose(cross_power, power_vals[0, 3], rtol=0, atol=atol)


def test_eval_errors(az_za_deg_grid):
    diameter_m = 14.0
    beam = AiryBeam(diameter=diameter_m)

    az_vals, za_vals, freqs = az_za_deg_grid

    az_mesh, za_mesh = np.meshgrid(az_vals, za_vals)

    with pytest.raises(
        ValueError,
        match="az_array, za_array and freq_array must all be one dimensional.",
    ):
        beam.power_eval(az_array=az_mesh, za_array=za_mesh, freq_array=freqs)

    with pytest.raises(
        ValueError, match="az_array and za_array must have the same shape."
    ):
        beam.efield_eval(az_array=az_vals, za_array=za_vals[0:-1], freq_array=freqs)


@pytest.mark.parametrize(
    ["beam_kwargs", "err_msg"],
    [
        [
            {"feed_array": "w", "diameter": 5},
            re.escape("Feeds must be one of: ['n', 'e', 'x', 'y', 'r', 'l']"),
        ],
        [{}, "Either diameter or sigma must be set."],
        [{"diameter": 5, "sigma": 0.2}, "Only one of diameter or sigma can be set."],
    ],
)
def test_beamerrs(beam_kwargs, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        GaussianBeam(**beam_kwargs)


def test_bad_basis_vector_type():
    class BadBeam(AnalyticBeam):
        basis_vector_type = "healpix"
        name = "bad beam"

        def _efield_eval(
            self, az_array: np.ndarray, za_array: np.ndarray, freq_array: np.ndarray
        ):
            """Evaluate the efield at the given coordinates."""
            data_array = self._get_empty_data_array(az_array, za_array, freq_array)
            data_array = data_array + 1.0 / np.sqrt(2.0)
            return data_array

        def _power_eval(
            self, az_array: np.ndarray, za_array: np.ndarray, freq_array: np.ndarray
        ):
            """Evaluate the efield at the given coordinates."""
            data_array = self._get_empty_data_array(
                az_array, za_array, freq_array, beam_type="power"
            )
            data_array = data_array + 1.0
            return data_array

    with pytest.raises(
        ValueError,
        match=re.escape("basis_vector_type is healpix, must be one of ['az_za']"),
    ):
        BadBeam()


def test_to_uvbeam_errors():
    beam = GaussianBeam(sigma=0.02)

    with pytest.raises(ValueError, match="Beam type must be 'efield' or 'power'"):
        beam.to_uvbeam(
            freq_array=np.linspace(100, 200, 5),
            beam_type="foo",
            axis1_array=np.deg2rad(np.linspace(0, 360, 36, endpoint=False)),
            axis2_array=np.deg2rad(np.linspace(0, 90, 10)),
        )

    with pytest.raises(
        NotImplementedError,
        match="Currently this method only supports 'az_za' and 'healpix' "
        "pixel_coordinate_systems.",
    ):
        beam.to_uvbeam(
            freq_array=np.linspace(100, 200, 5),
            beam_type="efield",
            axis1_array=np.deg2rad(np.linspace(0, 360, 36, endpoint=False)),
            axis2_array=np.deg2rad(np.linspace(0, 90, 10)),
            pixel_coordinate_system="orthoslant_zenith",
        )

    allowed_coord_sys = list(UVBeam().coordinate_system_dict.keys())
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unknown coordinate system foo. UVBeam supported coordinate systems "
            f"are: {allowed_coord_sys}."
        ),
    ):
        beam.to_uvbeam(
            freq_array=np.linspace(100, 200, 5),
            beam_type="efield",
            axis1_array=np.deg2rad(np.linspace(0, 360, 36, endpoint=False)),
            axis2_array=np.deg2rad(np.linspace(0, 90, 10)),
            pixel_coordinate_system="foo",
        )
