# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import numpy as np
import pytest

from pyuvdata import AiryBeam, BeamInterface, GaussianBeam, ShortDipoleBeam, UniformBeam


# This is just copied from test_analytic_beam for now to exercise the interface.
# Should be replaced with a more useful test, maybe using the `to_uvbeam` method.
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

    bi_efield = BeamInterface(beam, "efield")
    bi_power = BeamInterface(beam, "power")

    efield_vals = bi_efield.compute_response(az_vals, za_vals, freqs)
    power_vals = bi_power.compute_response(az_vals, za_vals, freqs)

    # check power beams are peak normalized
    assert np.max(power_vals) == 1.0

    np.testing.assert_allclose(
        efield_vals[0, 0] ** 2 + efield_vals[1, 0] ** 2,
        power_vals[0, 0],
        rtol=0,
        atol=1e-15,
    )

    np.testing.assert_allclose(
        efield_vals[0, 1] ** 2 + efield_vals[1, 1] ** 2,
        power_vals[0, 1],
        rtol=0,
        atol=1e-15,
    )

    cross_power = (
        efield_vals[0, 0] * efield_vals[0, 1] + efield_vals[1, 0] * efield_vals[1, 1]
    )
    np.testing.assert_allclose(cross_power, power_vals[0, 2], rtol=0, atol=1e-15)

    np.testing.assert_allclose(cross_power, power_vals[0, 3], rtol=0, atol=1e-15)


# This is just copied from test_uvbeam for now to exercise the interface.
# Should be replaced with a more useful test, maybe using the `to_uvbeam` method.
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

    bi_obj = BeamInterface(uvbeam)

    za_orig_vals, az_orig_vals = np.meshgrid(uvbeam.axis2_array, uvbeam.axis1_array)
    az_orig_vals = az_orig_vals.ravel(order="C")
    za_orig_vals = za_orig_vals.ravel(order="C")
    freq_orig_vals = np.array([123e6, 150e6])

    interp_data_array = bi_obj.compute_response(
        az_array=az_orig_vals, za_array=za_orig_vals, freq_array=freq_orig_vals
    )

    interp_data_array = interp_data_array.reshape(uvbeam.data_array.shape, order="F")
    assert np.allclose(uvbeam.data_array, interp_data_array)
