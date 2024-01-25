# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import numpy as np
import pytest

import pyuvdata.tests as uvtest
from pyuvdata import AiryBeam, BeamInterface, GaussianBeam, ShortDipoleBeam, UniformBeam


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
def test_beam_interface(
    beam_obj, kwargs, init_beam_type, final_beam_type, az_za_coords, xy_grid_coarse
):
    az_array, za_array = az_za_coords
    nfreqs = 20
    freq_array = np.linspace(100e6, 150e6, nfreqs)

    if "include_cross_pols" in kwargs.keys():
        include_cross_pols = kwargs["include_cross_pols"]
    else:
        include_cross_pols = True

    analytic = beam_obj(**kwargs)
    uvb = analytic.to_uvbeam(
        beam_type=init_beam_type,
        freq_array=freq_array,
        axis1_array=az_array,
        axis2_array=za_array,
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

    with uvtest.check_warnings(warn_type, match=msg):
        bi_uvbeam = BeamInterface(
            uvb, final_beam_type, include_cross_pols=include_cross_pols
        )

    analytic_data = bi_analytic.compute_response(
        az_array, za_array, freq_array, az_za_grid=True
    )
    uvb_data = bi_uvbeam.compute_response(
        az_array, za_array, freq_array, az_za_grid=True
    )

    np.testing.assert_allclose(analytic_data, uvb_data, rtol=0, atol=1e-15)

    # now on a grid that is not the same as where the beam was evaluated
    # larger differences of course
    az_vals, za_vals, freqs = xy_grid_coarse
    analytic_data = bi_analytic.compute_response(
        az_vals, za_vals, freqs, az_za_grid=True
    )
    uvb_data = bi_uvbeam.compute_response(az_vals, za_vals, freqs, az_za_grid=True)

    np.testing.assert_allclose(analytic_data, uvb_data, rtol=0, atol=1e-1)
