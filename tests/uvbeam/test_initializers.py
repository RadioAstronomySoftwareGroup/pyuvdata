# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
import re

import numpy as np
import pytest

from pyuvdata import UVBeam

ph_params = [
    "element_location_array",
    "element_coordinate_system",
    "delay_array",
    "gain_array",
    "coupling_matrix",
]


@pytest.fixture()
def uvb_common_kw():
    return {
        "telescope_name": "mock",
        "data_normalization": "physical",
        "freq_array": np.linspace(100e6, 200e6, 10),
    }


@pytest.fixture()
def uvb_azza_kw():
    return {
        "axis1_array": np.deg2rad(np.linspace(-180, 179, 360)),
        "axis2_array": np.deg2rad(np.linspace(0, 90, 181)),
    }


@pytest.fixture()
def uvb_healpix_kw():
    return {"nside": 64}


@pytest.fixture()
def uvb_efield_kw():
    return {"feed_array": ["x", "y"]}


@pytest.fixture()
def uvb_power_kw():
    return {"polarization_array": ["xx", "yy"]}


@pytest.fixture()
def uvb_azza_efield_kw(uvb_common_kw, uvb_azza_kw, uvb_efield_kw):
    return {**uvb_common_kw, **uvb_azza_kw, **uvb_efield_kw}


@pytest.fixture()
def uvb_healpix_efield_kw(uvb_common_kw, uvb_healpix_kw, uvb_efield_kw):
    return {**uvb_common_kw, **uvb_healpix_kw, **uvb_efield_kw}


@pytest.fixture()
def phased_array_efield(uvb_azza_efield_kw, phased_array_beam_2freq):
    uvb_azza_efield_kw["freq_array"] = (uvb_azza_efield_kw["freq_array"])[0:2]
    for param in ph_params:
        uvb_azza_efield_kw[param] = getattr(phased_array_beam_2freq, param)

    return uvb_azza_efield_kw


@pytest.mark.parametrize("coord_sys", ["az_za", "healpix"])
@pytest.mark.parametrize("beam_type", ["efield", "power"])
def test_new_uvcal_simplest(
    uvb_common_kw,
    uvb_azza_kw,
    uvb_healpix_kw,
    uvb_efield_kw,
    uvb_power_kw,
    coord_sys,
    beam_type,
):
    if coord_sys == "az_za":
        kw_use = {**uvb_common_kw, **uvb_azza_kw}
    else:
        kw_use = {**uvb_common_kw, **uvb_healpix_kw}

    if beam_type == "efield":
        kw_use = {**kw_use, **uvb_efield_kw}
    else:
        kw_use = {**kw_use, **uvb_power_kw}

    uvb = UVBeam.new(**kw_use)
    assert uvb.Nfreqs == 10
    if beam_type == "efield":
        assert uvb.Nfeeds == 2
    else:
        assert uvb.Npols == 2

    if uvb.pixel_coordinate_system == "healpix":
        assert uvb.Npixels == 12 * uvb.nside**2
    else:
        assert uvb.Naxes1 == 360
        assert uvb.Naxes2 == 181


def test_x_orientation(uvb_azza_efield_kw):
    uvb_azza_efield_kw["x_orientation"] = "e"
    uvb = UVBeam.new(**uvb_azza_efield_kw)

    assert uvb.x_orientation == "east"


@pytest.mark.parametrize("pcs", ["az_za", "healpix"])
def test_basis_vec(pcs, uvb_azza_efield_kw, uvb_healpix_efield_kw):
    if pcs == "az_za":
        beam_kw_use = uvb_azza_efield_kw
        pix_shape = (181, 360)
    else:
        beam_kw_use = uvb_healpix_efield_kw
        pix_shape = (12 * beam_kw_use["nside"] ** 2,)

    uvb1 = UVBeam.new(**beam_kw_use)
    full_shape = (2, 2, *pix_shape)
    basis_vector_array = np.zeros(full_shape, dtype=float)
    basis_vector_array[0, 0] = np.ones(pix_shape, dtype=float)
    basis_vector_array[1, 1] = np.ones(pix_shape, dtype=float)
    beam_kw_use["basis_vector_array"] = basis_vector_array
    uvb2 = UVBeam.new(**beam_kw_use)

    # make histories match (time stamp makes them different)
    uvb2.history = uvb1.history

    assert uvb2 == uvb1

    if pcs == "az_za":
        beam_kw_use["basis_vector_array"] = np.zeros((2, 2, 360, 181), dtype=float)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "basis_vector_array shape (2, 2, 360, 181) does not match expected "
                "shape (2, 2, 181, 360)."
            ),
        ):
            UVBeam.new(**uvb_azza_efield_kw)


def test_bandpass(uvb_azza_efield_kw):
    uvb1 = UVBeam.new(**uvb_azza_efield_kw)

    uvb_azza_efield_kw["bandpass_array"] = np.ones(10)
    uvb2 = UVBeam.new(**uvb_azza_efield_kw)

    # make histories match (time stamp makes them different)
    uvb2.history = uvb1.history

    assert uvb2 == uvb1

    uvb_azza_efield_kw["bandpass_array"] = np.ones((1, 10))
    with pytest.raises(
        ValueError,
        match="The bandpass array must have the same shape as the freq_array.",
    ):
        UVBeam.new(**uvb_azza_efield_kw)


@pytest.mark.parametrize("pcs", ["az_za", "healpix"])
def test_data_array(pcs, uvb_azza_efield_kw, uvb_healpix_efield_kw):
    if pcs == "az_za":
        beam_kw_use = uvb_azza_efield_kw
        pix_shape = (181, 360)
    else:
        beam_kw_use = uvb_healpix_efield_kw
        pix_shape = (12 * beam_kw_use["nside"] ** 2,)

    uvb1 = UVBeam.new(**beam_kw_use)
    full_shape = (2, 2, 10, *pix_shape)
    data_array = np.zeros(full_shape, dtype=complex)
    beam_kw_use["data_array"] = data_array
    uvb2 = UVBeam.new(**beam_kw_use)

    # make histories match (time stamp makes them different)
    uvb2.history = uvb1.history

    assert uvb2 == uvb1


@pytest.mark.parametrize(
    "rm_param",
    [None, "element_coordinate_system", "delay_array", "gain_array", "coupling_matrix"],
)
def test_phased_array(phased_array_efield, phased_array_beam_2freq, rm_param):
    if rm_param is not None:
        del phased_array_efield[rm_param]

    uvb = UVBeam.new(**phased_array_efield)

    for param in ph_params:
        assert getattr(uvb, "_" + param) == getattr(
            phased_array_beam_2freq, "_" + param
        )


def test_no_feed_pol_error(uvb_common_kw):
    with pytest.raises(
        ValueError,
        match=re.escape("Provide *either* feed_array *or* polarization_array"),
    ):
        UVBeam.new(**uvb_common_kw)


def test_pcs_params_error(uvb_common_kw, uvb_efield_kw, uvb_azza_kw, uvb_healpix_kw):
    with pytest.raises(
        ValueError,
        match="Either nside or both axis1_array and axis2_array must be provided.",
    ):
        UVBeam.new(**uvb_common_kw, **uvb_efield_kw)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Provide *either* nside (and optionally healpix_pixel_array and "
            "ordering) *or* axis1_array and axis2_array."
        ),
    ):
        UVBeam.new(**uvb_common_kw, **uvb_efield_kw, **uvb_azza_kw, **uvb_healpix_kw)

    hpx_kws = {"healpix_pixel_array": np.arange(12 * 64**2)}
    with pytest.raises(
        ValueError, match="nside must be provided if healpix_pixel_array is given."
    ):
        UVBeam.new(**uvb_common_kw, **uvb_efield_kw, **hpx_kws)

    pcs_kws = {"pixel_coordinate_system": "foo"}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "pixel_coordinate_system must be one of ['az_za', "
            "'orthoslant_zenith', 'healpix']"
        ),
    ):
        UVBeam.new(**uvb_common_kw, **uvb_efield_kw, **uvb_azza_kw, **pcs_kws)


def test_freq_array_errors(uvb_azza_efield_kw):
    uvb_azza_efield_kw["freq_array"] = (uvb_azza_efield_kw["freq_array"])[np.newaxis]
    with pytest.raises(ValueError, match="freq_array must be one dimensional."):
        UVBeam.new(**uvb_azza_efield_kw)

    uvb_azza_efield_kw["freq_array"] = uvb_azza_efield_kw["freq_array"].tolist()
    with pytest.raises(ValueError, match="freq_array must be a numpy ndarray"):
        UVBeam.new(**uvb_azza_efield_kw)


def test_data_array_errors(uvb_azza_efield_kw):
    uvb_azza_efield_kw["data_array"] = (2, 2, 10, 360, 181)
    with pytest.raises(ValueError, match="data_array must be a numpy ndarray"):
        UVBeam.new(**uvb_azza_efield_kw)

    uvb_azza_efield_kw["data_array"] = np.zeros((2, 2, 10, 360, 181))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Data array shape (2, 2, 10, 360, 181) does not match expected "
            "shape (2, 2, 10, 181, 360)."
        ),
    ):
        UVBeam.new(**uvb_azza_efield_kw)


@pytest.mark.parametrize(
    ["key_rm", "new_key", "value", "msg"],
    [
        [
            "feed_array",
            "polarization_array",
            [-1, -2],
            "feed_array must be provided if element_location_array is given.",
        ],
        [
            None,
            "element_location_array",
            np.arange(4),
            "element_location_array must be 2 dimensional",
        ],
        [
            None,
            "element_location_array",
            np.ones((4, 4)),
            "The first dimension of element_location_array must be length 2",
        ],
        [
            None,
            "element_location_array",
            np.ones((2, 1)),
            "The second dimension of element_location_array must be >= 2.",
        ],
        [
            None,
            "delay_array",
            np.arange(6),
            "delay_array must be one dimensional with length "
            "equal to the second dimension of element_location_array",
        ],
        [
            None,
            "gain_array",
            np.arange(6),
            "gain_array must be one dimensional with length "
            "equal to the second dimension of element_location_array",
        ],
        [
            None,
            "coupling_matrix",
            np.zeros((4, 4, 2, 2, 20)),
            re.escape(
                "coupling_matrix shape (4, 4, 2, 2, 20) does not "
                "match expected shape (4, 4, 2, 2, 2)."
            ),
        ],
    ],
)
def test_phased_array_errors(phased_array_efield, key_rm, new_key, value, msg):
    if key_rm is not None:
        del phased_array_efield[key_rm]
    phased_array_efield[new_key] = value

    with pytest.raises(ValueError, match=msg):
        UVBeam.new(**phased_array_efield)
