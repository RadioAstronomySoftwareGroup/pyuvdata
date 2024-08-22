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
    utils,
)
from pyuvdata.testing import check_warnings


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
    if coord_sys == "healpix":
        pytest.importorskip("astropy_healpix")
        from astropy_healpix import HEALPix

        nside = 64
        if init_beam_type == "efield":
            ordering = "ring"
        else:
            ordering = "nested"
        healpix_pixel_array = np.arange(12 * nside**2, dtype=int)

        hp_obj = HEALPix(nside=nside, order=ordering)
        hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(healpix_pixel_array)

        za_array, az_array = utils.coordinates.hpx_latlon_to_zenithangle_azimuth(
            hpx_lat.radian, hpx_lon.radian
        )

        to_uvbeam_kwargs = {
            "nside": nside,
            "ordering": ordering,
            "healpix_pixel_array": healpix_pixel_array,
        }

        # downselect places to check
        above_horizon = np.nonzero(za_array <= (np.pi / 2.0))
        az_array = az_array[above_horizon]
        za_array = za_array[above_horizon]
        az_array = az_array[::10]
        za_array = za_array[::10]
    else:
        az_array, za_array = az_za_coords
        to_uvbeam_kwargs = {"axis1_array": az_array, "axis2_array": za_array}

    nfreqs = 20
    freq_array = np.linspace(100e6, 150e6, nfreqs)

    include_cross_pols = kwargs.get("include_cross_pols", True)

    analytic = beam_obj(**kwargs)

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

    if not (coord_sys == "healpix" and "dipole" in beam_obj.name.lower()):
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


@pytest.mark.parametrize(
    ["bi1", "bi2", "equality"],
    [
        [
            BeamInterface(ShortDipoleBeam(), beam_type="efield"),
            BeamInterface(ShortDipoleBeam(), beam_type="efield"),
            True,
        ],
        [
            BeamInterface(ShortDipoleBeam(), beam_type="efield"),
            BeamInterface(ShortDipoleBeam(), beam_type="power"),
            False,
        ],
        [
            BeamInterface(ShortDipoleBeam(), beam_type="efield"),
            BeamInterface(AiryBeam(diameter=14.0), beam_type="efield"),
            False,
        ],
        [
            BeamInterface(AiryBeam(diameter=12.0), beam_type="efield"),
            BeamInterface(AiryBeam(diameter=14.0), beam_type="efield"),
            False,
        ],
        [
            BeamInterface(ShortDipoleBeam(), beam_type="efield"),
            ShortDipoleBeam(),
            False,
        ],
    ],
)
def test_beam_interface_equality(bi1, bi2, equality):
    if equality:
        assert bi1 == bi2
    else:
        assert bi1 != bi2
        assert not bi1 == bi2  # noqa SIM201


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
