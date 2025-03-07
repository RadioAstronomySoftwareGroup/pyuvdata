# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
import copy

import numpy as np
import pytest

from pyuvdata import (
    AiryBeam,
    BeamInterface,
    GaussianBeam,
    ShortDipoleBeam,
    UniformBeam,
    UVBeam,
    utils,
)
from pyuvdata.testing import check_warnings


@pytest.fixture(scope="function")
def airy() -> AiryBeam:
    return AiryBeam(diameter=14.0)


@pytest.fixture()
def gaussian() -> GaussianBeam:
    return GaussianBeam(diameter=14.0)


@pytest.fixture()
def xy_grid_coarse():
    nfreqs = 5
    freqs = np.linspace(100e6, 130e6, nfreqs)

    xy_half_n = 6
    zmax = np.radians(60)  # Degrees
    arr = np.arange(-xy_half_n, xy_half_n)
    x_arr, y_arr = np.meshgrid(arr, arr)
    x_arr = x_arr.flatten()
    y_arr = y_arr.flatten()
    radius = np.sqrt(x_arr**2 + y_arr**2) / float(xy_half_n)
    za_array = radius * zmax
    az_array = np.arctan2(y_arr, x_arr) + np.pi  # convert from -180->180 to 0->360

    return az_array, za_array, freqs


@pytest.fixture()
def gaussian_uv(gaussian, az_za_coords) -> UVBeam:
    az, za = az_za_coords
    return gaussian.to_uvbeam(
        axis1_array=az, axis2_array=za, freq_array=np.array([1e8])
    )


@pytest.mark.parametrize(
    ["beam_obj", "kwargs"],
    [
        [AiryBeam, {"diameter": 14.0}],
        [GaussianBeam, {"diameter": 14.0}],
        [UniformBeam, {"include_cross_pols": False}],
        [ShortDipoleBeam, {}],
    ],
)
@pytest.mark.parametrize("init_beam_type", ["efield", "power"])
@pytest.mark.parametrize("final_beam_type", ["efield", "power"])
@pytest.mark.parametrize("coord_sys", ["az_za", "healpix"])
def test_beam_interface(
    beam_obj,
    kwargs,
    init_beam_type,
    final_beam_type,
    az_za_coords,
    xy_grid_coarse,
    coord_sys,
):
    analytic = beam_obj(**kwargs)

    nfreqs = 20
    freq_array = np.linspace(100e6, 150e6, nfreqs)

    if coord_sys == "healpix":
        nside = 64
        if init_beam_type == "efield":
            ordering = "ring"
        else:
            ordering = "nested"
        healpix_pixel_array = np.arange(12 * nside**2, dtype=int)

        to_uvbeam_kwargs = {
            "nside": nside,
            "ordering": ordering,
            "healpix_pixel_array": healpix_pixel_array,
        }

        try:
            from astropy_healpix import HEALPix

        except ImportError:
            with pytest.raises(
                ImportError,
                match="astropy_healpix is not installed but is "
                "required for healpix functionality. ",
            ):
                uvb = analytic.to_uvbeam(
                    beam_type=init_beam_type, freq_array=freq_array, **to_uvbeam_kwargs
                )
            pytest.importorskip("astropy_healpix")

        hp_obj = HEALPix(nside=nside, order=ordering)
        hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(healpix_pixel_array)

        za_array, az_array = utils.coordinates.hpx_latlon_to_zenithangle_azimuth(
            hpx_lat.radian, hpx_lon.radian
        )

        # downselect places to check
        above_horizon = np.nonzero(za_array <= (np.pi / 2.0))
        az_array = az_array[above_horizon]
        za_array = za_array[above_horizon]
        az_array = az_array[::10]
        za_array = za_array[::10]
    else:
        az_array, za_array = az_za_coords
        to_uvbeam_kwargs = {"axis1_array": az_array, "axis2_array": za_array}

    include_cross_pols = kwargs.get("include_cross_pols", True)

    uvb = analytic.to_uvbeam(
        beam_type=init_beam_type, freq_array=freq_array, **to_uvbeam_kwargs
    )
    bi_analytic = BeamInterface(analytic, final_beam_type)

    if final_beam_type != init_beam_type:
        if final_beam_type == "efield":
            with pytest.raises(
                ValueError,
                match="Input beam is a power UVBeam but beam_type is specified as "
                "'efield'. It's not possible to convert a power beam to an "
                "efield beam, either provide an efield UVBeam or do not "
                "specify `beam_type`.",
            ):
                BeamInterface(uvb, final_beam_type)
            return

        warn_type = UserWarning
        msg = (
            "Input beam is an efield UVBeam but beam_type is specified as "
            "'power'. Converting efield beam to power."
        )
    else:
        warn_type = None
        msg = ""

    with check_warnings(warn_type, match=msg):
        bi_uvbeam = BeamInterface(
            uvb, final_beam_type, include_cross_pols=include_cross_pols
        )

    if coord_sys == "az_za":
        az_za_grid = True
    else:
        az_za_grid = False

    analytic_data = bi_analytic.compute_response(
        az_array=az_array,
        za_array=za_array,
        freq_array=freq_array,
        az_za_grid=az_za_grid,
    )

    uvb_data = bi_uvbeam.compute_response(
        az_array=az_array,
        za_array=za_array,
        freq_array=freq_array,
        az_za_grid=az_za_grid,
    )

    np.testing.assert_allclose(analytic_data, uvb_data, rtol=0, atol=1e-14)

    # now on a grid that is not the same as where the beam was evaluated
    # larger differences of course
    az_vals, za_vals, freqs = xy_grid_coarse
    analytic_data = bi_analytic.compute_response(
        az_array=az_vals, za_array=za_vals, freq_array=freqs, az_za_grid=True
    )
    uvb_data = bi_uvbeam.compute_response(
        az_array=az_vals, za_array=za_vals, freq_array=freqs, az_za_grid=True
    )

    if not (coord_sys == "healpix" and isinstance(analytic, ShortDipoleBeam)):
        np.testing.assert_allclose(analytic_data, uvb_data, rtol=0, atol=1e-1)
    else:
        # the comparison falls apart at zenith because there's no healpix
        # pixel right at zenith and the dipole beam changes quickly there.
        az_mesh, za_mesh = np.meshgrid(az_vals, za_vals)
        az_mesh = az_mesh.flatten()
        za_mesh = za_mesh.flatten()
        wh_not_zenith = np.nonzero(za_mesh != 0)
        np.testing.assert_allclose(
            analytic_data[:, :, :, wh_not_zenith],
            uvb_data[:, :, :, wh_not_zenith],
            rtol=0,
            atol=1e-1,
        )


def test_beam_interface_errors():
    with pytest.raises(
        ValueError, match="beam must be a UVBeam or an AnalyticBeam instance."
    ):
        BeamInterface("foo", "power")


@pytest.mark.parametrize(
    ["param", "value"],
    [
        ["az_array", None],
        ["za_array", None],
        ["freq_array", None],
        ["az_array", np.zeros((10, 10), dtype=float)],
        ["za_array", np.zeros((10, 10), dtype=float)],
        ["freq_array", np.zeros((10, 10), dtype=float)],
    ],
)
def test_compute_response_errors(param, value):
    orig_kwargs = {
        "az_array": np.deg2rad(np.linspace(0, 360, 36, endpoint=False)),
        "za_array": np.deg2rad(np.linspace(0, 90, 10)),
        "freq_array": np.deg2rad(np.linspace(100, 200, 5)),
    }

    compute_kwargs = copy.deepcopy(orig_kwargs)
    compute_kwargs["az_za_grid"] = True
    compute_kwargs[param] = value

    analytic = ShortDipoleBeam()
    bi_analytic = BeamInterface(analytic, beam_type="efield")

    with pytest.raises(
        ValueError, match=f"{param} must be a one-dimensional numpy array"
    ):
        bi_analytic.compute_response(**compute_kwargs)

    uvb = analytic.to_uvbeam(
        beam_type="power",
        freq_array=orig_kwargs["freq_array"],
        axis1_array=orig_kwargs["az_array"],
        axis2_array=orig_kwargs["za_array"],
    )
    bi_uvb = BeamInterface(uvb)

    if param != "freq_array":
        with pytest.raises(
            ValueError, match=f"{param} must be a one-dimensional numpy array"
        ):
            bi_uvb.compute_response(**compute_kwargs)
    elif value is not None:
        with pytest.raises(ValueError, match="freq_array must be one-dimensional"):
            bi_uvb.compute_response(**compute_kwargs)

    else:
        # this shouldn't error
        bi_uvb.compute_response(**compute_kwargs)


@pytest.mark.parametrize("beam_obj", ["airy", "gaussian", "gaussian_uv"])
def test_idempotent_instantiation(beam_obj, request):
    beam = BeamInterface(request.getfixturevalue(beam_obj))
    beam2 = BeamInterface(beam)
    assert beam == beam2


def test_properties(airy: AiryBeam):
    intf = BeamInterface(airy)
    assert airy.Npols == intf.Npols
    assert airy.Nfeeds == intf.Nfeeds
    assert np.all(airy.polarization_array == intf.polarization_array)
    assert np.all(airy.feed_array == intf.feed_array)
    assert np.all(airy.feed_angle == intf.feed_angle)


def test_clone(airy):
    intf = BeamInterface(airy)
    intf_clone = intf.clone(beam_type="power")
    assert intf != intf_clone


@pytest.mark.parametrize("uvbeam", [True, False], ids=["uvbeam", "analytic"])
@pytest.mark.parametrize("allow_mutation", [True, False], ids=["mutate", "nomutate"])
@pytest.mark.parametrize(
    "include_cross_pols", [True, False, None], ids=["incx", "nox", "xpolnone"]
)
def test_as_power(
    uvbeam: bool, allow_mutation: bool, include_cross_pols: bool, gaussian, gaussian_uv
):
    beam = gaussian_uv if uvbeam else gaussian
    intf = BeamInterface(beam)
    intf_power = intf.as_power_beam(
        allow_beam_mutation=allow_mutation, include_cross_pols=include_cross_pols
    )
    if include_cross_pols is None:
        include_cross_pols = True

    assert intf_power.beam_type == "power"
    assert intf_power.Npols == 4 if include_cross_pols else 2

    if uvbeam:
        if allow_mutation:
            assert intf.beam.beam_type == "power"
        else:
            assert intf.beam.beam_type == "efield"


def test_as_power_noop(airy):
    """Ensure that calling as_power_beam on a power beam is a no-op."""
    intf = BeamInterface(airy, beam_type="power")
    intf2 = intf.as_power_beam()
    assert intf is intf2

    with pytest.warns(UserWarning, match="as_power_beam does not modify cross pols"):
        intf2 = intf.as_power_beam(include_cross_pols=False)
    assert intf is intf2


@pytest.mark.parametrize("uvbeam", [True, False])
def test_with_feeds(uvbeam: bool, gaussian, gaussian_uv):
    beam = gaussian_uv if uvbeam else gaussian

    intf = BeamInterface(beam)

    intf_feedx = intf.with_feeds(["x"])
    assert intf_feedx.feed_array == ["x"]


def test_with_feeds_ordering(airy):
    intf = BeamInterface(airy)

    intf_feedx = intf.with_feeds(["y", "x"], maintain_ordering=True)
    assert np.all(intf_feedx.feed_array == ["x", "y"])

    intf_feedyx = intf.with_feeds(["y", "x"], maintain_ordering=False)
    assert np.all(intf_feedyx.feed_array == ["y", "x"])


@pytest.mark.filterwarnings("ignore:Input beam is an efield UVBeam")
@pytest.mark.filterwarnings("ignore:Selected polarizations are not evenly spaced")
def test_with_feeds_ordering_power(gaussian_uv):
    # beam = AiryBeam(diameter=14.0).to_uvbeam(freq_array=np.array([1e8]), nside=16)
    intf = BeamInterface(gaussian_uv, beam_type="power")
    intf_feedx = intf.with_feeds(["y", "x"], maintain_ordering=True)
    assert np.all(intf_feedx.polarization_array == [-5, -6, -7, -8])

    intf_feedyx = intf.with_feeds(["y", "x"], maintain_ordering=False)
    # N.b. (Karto), this used to check against [-6, -8, -7, -5], but I _think_ this
    # was actually a bug, in that UVBeam.select was sensitive to the ordering of
    # pol arguments for polarization_array *only*, and not anything else with a
    # polarization axis.
    assert np.all(intf_feedyx.polarization_array == [-5, -6, -7, -8])
