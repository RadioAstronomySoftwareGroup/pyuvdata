# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for phasing utility functions."""

import os
import re

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time

import pyuvdata.utils.phasing as phs_utils
from pyuvdata import UVData, utils
from pyuvdata.data import DATA_PATH

from .test_coordinates import frame_selenoid


@pytest.fixture
def vector_list():
    x_vecs = np.array([[1, 0, 0], [2, 0, 0]], dtype=float).T
    y_vecs = np.array([[0, 1, 0], [0, 2, 0]], dtype=float).T
    z_vecs = np.array([[0, 0, 1], [0, 0, 2]], dtype=float).T
    test_vecs = np.array([[1, 1, 1], [2, 2, 2]], dtype=float).T

    yield x_vecs, y_vecs, z_vecs, test_vecs


@pytest.fixture
def calc_uvw_args():
    default_args = {
        "app_ra": np.zeros(3),
        "app_dec": np.zeros(3) + 1.0,
        "frame_pa": np.zeros(3) + 1e-3,
        "lst_array": np.zeros(3) + np.pi,
        "use_ant_pos": True,
        "uvw_array": np.array([[1, -1, 0], [0, -1, 1], [-1, 0, 1]], dtype=float),
        "antenna_positions": np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float),
        "antenna_numbers": [1, 2, 3],
        "ant_1_array": np.array([1, 1, 2]),
        "ant_2_array": np.array([2, 3, 3]),
        "old_app_ra": np.zeros(3) + np.pi,
        "old_app_dec": np.zeros(3),
        "old_frame_pa": np.zeros(3),
        "telescope_lat": 1.0,
        "telescope_lon": 0.0,
        "to_enu": False,
        "from_enu": False,
    }
    yield default_args


@pytest.mark.skipif(
    len(frame_selenoid) > 1, reason="Test only when lunarsky not installed."
)
def test_no_moon():
    """Check errors when calling functions with MCMF without lunarsky."""
    msg = "Need to install `lunarsky` package to work with MCMF frame."
    with pytest.raises(ImportError, match=msg):
        phs_utils.transform_icrs_to_app(
            time_array=[2451545.0],
            ra=0,
            dec=0,
            telescope_loc=(0, 0, 0),
            telescope_frame="mcmf",
        )
    with pytest.raises(ImportError, match=msg):
        phs_utils.transform_app_to_icrs(
            time_array=[2451545.0],
            app_ra=0,
            app_dec=0,
            telescope_loc=(0, 0, 0),
            telescope_frame="mcmf",
        )
    with pytest.raises(ImportError, match=msg):
        phs_utils.calc_app_coords(lon_coord=0.0, lat_coord=0.0, telescope_frame="mcmf")


@pytest.mark.parametrize(
    "lon_array,lat_array,msg",
    (
        [0.0, np.array([0.0]), "lon_array and lat_array must either both be floats or"],
        [np.array([0.0, 1.0]), np.array([0.0]), "lon_array and lat_array must have "],
    ),
)
def test_polar2_to_cart3_arg_errs(lon_array, lat_array, msg):
    """
    Test that bad arguments to polar2_to_cart3 throw appropriate errors.
    """
    with pytest.raises(ValueError, match=msg):
        phs_utils.polar2_to_cart3(lon_array=lon_array, lat_array=lat_array)


@pytest.mark.parametrize(
    "input1,msg",
    (
        [0.0, "xyz_array must be an ndarray."],
        [np.array(0.0), "xyz_array must have ndim > 0"],
        [np.array([0.0]), "xyz_array must be length 3"],
    ),
)
def test_cart3_to_polar2_arg_errs(input1, msg):
    """
    Test that bad arguments to cart3_to_polar2 throw appropriate errors.
    """
    with pytest.raises(ValueError, match=msg):
        phs_utils.cart3_to_polar2(input1)


@pytest.mark.parametrize(
    "input1,input2,input3,msg",
    (
        [np.zeros((1, 3, 1)), np.zeros((1, 3, 3)), 2, "rot_matrix must be of shape "],
        [np.zeros((1, 2, 1)), np.zeros((1, 3, 3)), 1, "Misshaped xyz_array - expected"],
        [np.zeros((2, 1)), np.zeros((1, 3, 3)), 1, "Misshaped xyz_array - expected"],
        [np.zeros(2), np.zeros((1, 3, 3)), 1, "Misshaped xyz_array - expected shape"],
    ),
)
def test_rotate_matmul_wrapper_arg_errs(input1, input2, input3, msg):
    """
    Test that bad arguments to _rotate_matmul_wrapper throw appropriate errors.
    """
    with pytest.raises(ValueError, match=msg):
        phs_utils._rotate_matmul_wrapper(
            xyz_array=input1, rot_matrix=input2, n_rot=input3
        )


def test_cart_to_polar_roundtrip():
    """
    Test that polar->cart coord transformation is the inverse of cart->polar.
    """
    # Basic round trip with vectors
    assert phs_utils.cart3_to_polar2(
        phs_utils.polar2_to_cart3(lon_array=0.0, lat_array=0.0)
    ) == (0.0, 0.0)


def test_rotate_one_axis(vector_list):
    """
    Tests some basic vector rotation operations with a single axis rotation.
    """
    # These tests are used to verify the basic functionality of the primary
    # functions used to perform rotations
    x_vecs, y_vecs, z_vecs, test_vecs = vector_list

    # Test no-ops w/ 0 deg rotations
    assert np.all(
        phs_utils._rotate_one_axis(xyz_array=x_vecs, rot_amount=0.0, rot_axis=0)
        == x_vecs
    )
    assert np.all(
        phs_utils._rotate_one_axis(xyz_array=x_vecs[:, 0], rot_amount=0.0, rot_axis=1)
        == x_vecs[np.newaxis, :, 0, np.newaxis]
    )
    assert np.all(
        phs_utils._rotate_one_axis(
            xyz_array=x_vecs[:, :, np.newaxis], rot_amount=0.0, rot_axis=2
        )
        == x_vecs[:, :, np.newaxis]
    )

    # Test no-ops w/ None
    assert np.all(
        phs_utils._rotate_one_axis(xyz_array=test_vecs, rot_amount=None, rot_axis=1)
        == test_vecs
    )
    assert np.all(
        phs_utils._rotate_one_axis(
            xyz_array=test_vecs[:, 0], rot_amount=None, rot_axis=2
        )
        == test_vecs[np.newaxis, :, 0, np.newaxis]
    )
    assert np.all(
        phs_utils._rotate_one_axis(
            xyz_array=test_vecs[:, :, np.newaxis], rot_amount=None, rot_axis=0
        )
        == test_vecs[:, :, np.newaxis]
    )

    # Test some basic equivalencies to make sure rotations are working correctly
    np.testing.assert_allclose(
        x_vecs[np.newaxis],
        phs_utils._rotate_one_axis(xyz_array=x_vecs, rot_amount=1.0, rot_axis=0),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        y_vecs[np.newaxis],
        phs_utils._rotate_one_axis(xyz_array=y_vecs, rot_amount=2.0, rot_axis=1),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        z_vecs[np.newaxis],
        phs_utils._rotate_one_axis(xyz_array=z_vecs, rot_amount=3.0, rot_axis=2),
        rtol=0,
        atol=1e-3,
    )

    np.testing.assert_allclose(
        x_vecs[np.newaxis],
        phs_utils._rotate_one_axis(
            xyz_array=y_vecs, rot_amount=-np.pi / 2.0, rot_axis=2
        ),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        y_vecs[np.newaxis],
        phs_utils._rotate_one_axis(
            xyz_array=x_vecs, rot_amount=np.pi / 2.0, rot_axis=2
        ),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        x_vecs[np.newaxis],
        phs_utils._rotate_one_axis(
            xyz_array=z_vecs, rot_amount=np.pi / 2.0, rot_axis=1
        ),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        z_vecs[np.newaxis],
        phs_utils._rotate_one_axis(
            xyz_array=x_vecs, rot_amount=-np.pi / 2.0, rot_axis=1
        ),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        y_vecs[np.newaxis],
        phs_utils._rotate_one_axis(
            xyz_array=z_vecs, rot_amount=-np.pi / 2.0, rot_axis=0
        ),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        z_vecs[np.newaxis],
        phs_utils._rotate_one_axis(
            xyz_array=y_vecs, rot_amount=np.pi / 2.0, rot_axis=0
        ),
        rtol=0,
        atol=1e-3,
    )

    assert np.all(
        np.equal(
            phs_utils._rotate_one_axis(xyz_array=test_vecs, rot_amount=1.0, rot_axis=2),
            phs_utils._rotate_one_axis(
                xyz_array=test_vecs, rot_amount=1.0, rot_axis=np.array([2])
            ),
        )
    )

    # Testing a special case, where the xyz_array vectors are reshaped if there
    # is only a single rotation matrix used (helps speed things up significantly)
    mod_vec = x_vecs.T.reshape((2, 3, 1))
    assert np.all(
        phs_utils._rotate_one_axis(xyz_array=mod_vec, rot_amount=1.0, rot_axis=0)
        == mod_vec
    )


def test_rotate_two_axis(vector_list):
    """
    Tests some basic vector rotation operations with a double axis rotation.
    """
    x_vecs, y_vecs, z_vecs, test_vecs = vector_list

    # These tests are used to verify the basic functionality of the primary
    # functions used to two-axis rotations
    np.testing.assert_allclose(
        x_vecs[np.newaxis],
        phs_utils._rotate_two_axis(
            xyz_array=x_vecs,
            rot_amount1=2 * np.pi,
            rot_amount2=1.0,
            rot_axis1=1,
            rot_axis2=0,
        ),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        y_vecs[np.newaxis],
        phs_utils._rotate_two_axis(
            xyz_array=y_vecs,
            rot_amount1=2 * np.pi,
            rot_amount2=2.0,
            rot_axis1=2,
            rot_axis2=1,
        ),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        z_vecs[np.newaxis],
        phs_utils._rotate_two_axis(
            xyz_array=z_vecs,
            rot_amount1=2 * np.pi,
            rot_amount2=3.0,
            rot_axis1=0,
            rot_axis2=2,
        ),
        rtol=0,
        atol=1e-3,
    )

    # Do one more test, which verifies that we can filp our (1,1,1) test vector to
    # the postiion at (-1, -1 , -1)
    mod_vec = test_vecs.T.reshape((2, 3, 1))
    np.testing.assert_allclose(
        phs_utils._rotate_two_axis(
            xyz_array=mod_vec,
            rot_amount1=np.pi,
            rot_amount2=np.pi / 2.0,
            rot_axis1=0,
            rot_axis2=1,
        ),
        -mod_vec,
        rtol=0,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "rot1,axis1,rot2,rot3,axis2,axis3",
    (
        [2.0, 0, 1.0, 1.0, 0, 0],
        [2.0, 0, 2.0, 0.0, 0, 1],
        [2.0, 0, None, 2.0, 1, 0],
        [0.0, 0, None, 0.0, 1, 2],
    ),
)
def test_compare_one_to_two_axis(vector_list, rot1, axis1, rot2, rot3, axis2, axis3):
    """
    Check that one-axis and two-axis rotations provide the same values when the
    two-axis rotations are fundamentally rotating around a single axis.
    """
    x_vecs, y_vecs, z_vecs, test_vecs = vector_list
    # If performing two rots on the same axis, that should be identical to using
    # a single rot (with the rot angle equal to the sum of the two rot angles)
    assert np.all(
        np.equal(
            phs_utils._rotate_one_axis(
                xyz_array=test_vecs, rot_amount=rot1, rot_axis=axis1
            ),
            phs_utils._rotate_two_axis(
                xyz_array=test_vecs,
                rot_amount1=rot2,
                rot_amount2=rot3,
                rot_axis1=axis2,
                rot_axis2=axis3,
            ),
        )
    )


@pytest.mark.parametrize(
    "arg_dict,err",
    (
        [
            {"lst_array": None, "to_enu": True, "use_ant_pos": False},
            (ValueError, "Must include lst_array to calculate baselines in ENU"),
        ],
        [
            {"lst_array": None, "to_enu": True, "telescope_lat": None},
            (ValueError, "Must include telescope_lat to calculate baselines"),
        ],
        [
            {"lst_array": None},
            (ValueError, "Must include lst_array if use_ant_pos=True and not"),
        ],
        [
            {"app_ra": None, "frame_pa": None},
            (ValueError, "Must include both app_ra and app_dec, or frame_pa to"),
        ],
        [
            {"app_dec": None, "frame_pa": None},
            (ValueError, "Must include both app_ra and app_dec, or frame_pa to"),
        ],
        [
            {"app_ra": None, "app_dec": None, "frame_pa": None},
            (ValueError, "Must include both app_ra and app_dec, or frame_pa to"),
        ],
        [
            {"antenna_positions": None},
            (ValueError, "Must include antenna_positions if use_ant_pos=True."),
        ],
        [
            {"ant_1_array": None},
            (ValueError, "Must include ant_1_array, ant_2_array, and antenna_numbers"),
        ],
        [
            {"ant_2_array": None},
            (ValueError, "Must include ant_1_array, ant_2_array, and antenna_numbers"),
        ],
        [
            {"antenna_numbers": None},
            (ValueError, "Must include ant_1_array, ant_2_array, and antenna_numbers"),
        ],
        [
            {"telescope_lon": None},
            (ValueError, "Must include telescope_lon if use_ant_pos=True."),
        ],
        [
            {"uvw_array": None, "use_ant_pos": False},
            (ValueError, "Must include uvw_array if use_ant_pos=False."),
        ],
        [
            {"telescope_lat": None, "use_ant_pos": False, "from_enu": True},
            (ValueError, "Must include telescope_lat if moving "),
        ],
        [
            {"lst_array": None, "use_ant_pos": False, "from_enu": True},
            (
                ValueError,
                re.escape("Must include lst_array if moving between ENU (i.e.,"),
            ),
        ],
        [
            {"use_ant_pos": False, "old_app_ra": None},
            (ValueError, "Must include old_app_ra and old_app_dec values when data"),
        ],
        [
            {"use_ant_pos": False, "old_app_dec": None},
            (ValueError, "Must include old_app_ra and old_app_dec values when data"),
        ],
        [
            {"use_ant_pos": False, "old_frame_pa": None},
            (ValueError, "Must include old_frame_pa values if data are phased and "),
        ],
    ),
)
def test_calc_uvw_input_errors(calc_uvw_args, arg_dict, err):
    """
    Check for argument errors with calc_uvw.
    """
    for key in arg_dict:
        calc_uvw_args[key] = arg_dict[key]

    with pytest.raises(err[0], match=err[1]):
        phs_utils.calc_uvw(
            app_ra=calc_uvw_args["app_ra"],
            app_dec=calc_uvw_args["app_dec"],
            frame_pa=calc_uvw_args["frame_pa"],
            lst_array=calc_uvw_args["lst_array"],
            use_ant_pos=calc_uvw_args["use_ant_pos"],
            uvw_array=calc_uvw_args["uvw_array"],
            antenna_positions=calc_uvw_args["antenna_positions"],
            antenna_numbers=calc_uvw_args["antenna_numbers"],
            ant_1_array=calc_uvw_args["ant_1_array"],
            ant_2_array=calc_uvw_args["ant_2_array"],
            old_app_ra=calc_uvw_args["old_app_ra"],
            old_app_dec=calc_uvw_args["old_app_dec"],
            old_frame_pa=calc_uvw_args["old_frame_pa"],
            telescope_lat=calc_uvw_args["telescope_lat"],
            telescope_lon=calc_uvw_args["telescope_lon"],
            from_enu=calc_uvw_args["from_enu"],
            to_enu=calc_uvw_args["to_enu"],
        )


def test_calc_uvw_no_op(calc_uvw_args):
    """
    Test that transfroming ENU -> ENU gives you an output identical to the input.
    """
    # This should be a no-op, check for equality
    uvw_check = phs_utils.calc_uvw(
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=calc_uvw_args["uvw_array"],
        telescope_lat=calc_uvw_args["telescope_lat"],
        telescope_lon=calc_uvw_args["telescope_lon"],
        to_enu=True,
        from_enu=True,
    )
    assert np.all(np.equal(calc_uvw_args["uvw_array"], uvw_check))


def test_calc_uvw_same_place(calc_uvw_args):
    """
    Check and see that the uvw calculator derives the same values derived by hand
    (i.e, that calculating for the same position returns the same answer).
    """
    # Check ant make sure that when we plug in the original values, we recover the
    # exact same values that we calculated above.
    uvw_ant_check = phs_utils.calc_uvw(
        app_ra=calc_uvw_args["old_app_ra"],
        app_dec=calc_uvw_args["old_app_dec"],
        frame_pa=calc_uvw_args["old_frame_pa"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=True,
        antenna_positions=calc_uvw_args["antenna_positions"],
        antenna_numbers=calc_uvw_args["antenna_numbers"],
        ant_1_array=calc_uvw_args["ant_1_array"],
        ant_2_array=calc_uvw_args["ant_2_array"],
        telescope_lat=calc_uvw_args["telescope_lat"],
        telescope_lon=calc_uvw_args["telescope_lon"],
    )

    uvw_base_check = phs_utils.calc_uvw(
        app_ra=calc_uvw_args["old_app_ra"],
        app_dec=calc_uvw_args["old_app_dec"],
        frame_pa=calc_uvw_args["old_frame_pa"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=calc_uvw_args["uvw_array"],
        old_app_ra=calc_uvw_args["old_app_ra"],
        old_app_dec=calc_uvw_args["old_app_dec"],
        old_frame_pa=calc_uvw_args["old_frame_pa"],
    )

    np.testing.assert_allclose(
        uvw_ant_check, calc_uvw_args["uvw_array"], rtol=0, atol=1e-3
    )
    np.testing.assert_allclose(
        uvw_base_check, calc_uvw_args["uvw_array"], rtol=0, atol=1e-3
    )


@pytest.mark.parametrize("to_enu", [False, True])
def test_calc_uvw_base_vs_ants(calc_uvw_args, to_enu):
    """
    Check to see that we get the same values for uvw coordinates whether we calculate
    them using antenna positions or the previously calculated uvw's.
    """

    # Now change position, and make sure that whether we used ant positions of rotated
    # uvw vectors, we derived the same uvw-coordinates at the end
    uvw_ant_check = phs_utils.calc_uvw(
        app_ra=calc_uvw_args["app_ra"],
        app_dec=calc_uvw_args["app_dec"],
        frame_pa=calc_uvw_args["frame_pa"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=True,
        antenna_positions=calc_uvw_args["antenna_positions"],
        antenna_numbers=calc_uvw_args["antenna_numbers"],
        ant_1_array=calc_uvw_args["ant_1_array"],
        ant_2_array=calc_uvw_args["ant_2_array"],
        telescope_lat=calc_uvw_args["telescope_lat"],
        telescope_lon=calc_uvw_args["telescope_lon"],
        to_enu=to_enu,
    )

    uvw_base_check = phs_utils.calc_uvw(
        app_ra=calc_uvw_args["app_ra"],
        app_dec=calc_uvw_args["app_dec"],
        frame_pa=calc_uvw_args["frame_pa"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=calc_uvw_args["uvw_array"],
        old_app_ra=calc_uvw_args["old_app_ra"],
        old_app_dec=calc_uvw_args["old_app_dec"],
        old_frame_pa=calc_uvw_args["old_frame_pa"],
        telescope_lat=calc_uvw_args["telescope_lat"],
        telescope_lon=calc_uvw_args["telescope_lon"],
        to_enu=to_enu,
    )

    np.testing.assert_allclose(uvw_ant_check, uvw_base_check, rtol=0, atol=1e-3)


def test_calc_uvw_enu_roundtrip(calc_uvw_args):
    """
    Check and see that we can go from uvw to ENU and back to uvw using the `uvw_array`
    argument alone (i.e., without antenna positions).
    """
    # Now attempt to round trip from projected to ENU back to projected -- that should
    # give us the original set of uvw-coordinates.
    temp_uvw = phs_utils.calc_uvw(
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=calc_uvw_args["uvw_array"],
        old_app_ra=calc_uvw_args["old_app_ra"],
        old_app_dec=calc_uvw_args["old_app_dec"],
        old_frame_pa=calc_uvw_args["old_frame_pa"],
        telescope_lat=calc_uvw_args["telescope_lat"],
        telescope_lon=calc_uvw_args["telescope_lon"],
        to_enu=True,
    )

    uvw_base_enu_check = phs_utils.calc_uvw(
        app_ra=calc_uvw_args["old_app_ra"],
        app_dec=calc_uvw_args["old_app_dec"],
        frame_pa=calc_uvw_args["old_frame_pa"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=temp_uvw,
        telescope_lat=calc_uvw_args["telescope_lat"],
        telescope_lon=calc_uvw_args["telescope_lon"],
        from_enu=True,
    )

    np.testing.assert_allclose(
        calc_uvw_args["uvw_array"], uvw_base_enu_check, atol=1e-15, rtol=0
    )


def test_calc_uvw_pa_ex_post_facto(calc_uvw_args):
    """
    Check and see that one can apply the frame position angle rotation after-the-fact
    and still get out the same answer you get if you were doing it during the initial
    uvw coordinate calculation.
    """
    # Finally, check and see what happens if you do the PA rotation as part of the
    # first uvw calcuation, and make sure it agrees with what you get if you decide
    # to apply the PA rotation after-the-fact.
    uvw_base_check = phs_utils.calc_uvw(
        app_ra=calc_uvw_args["app_ra"],
        app_dec=calc_uvw_args["app_dec"],
        frame_pa=calc_uvw_args["frame_pa"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=calc_uvw_args["uvw_array"],
        old_app_ra=calc_uvw_args["old_app_ra"],
        old_app_dec=calc_uvw_args["old_app_dec"],
        old_frame_pa=calc_uvw_args["old_frame_pa"],
    )

    temp_uvw = phs_utils.calc_uvw(
        app_ra=calc_uvw_args["app_ra"],
        app_dec=calc_uvw_args["app_dec"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=calc_uvw_args["uvw_array"],
        old_app_ra=calc_uvw_args["old_app_ra"],
        old_app_dec=calc_uvw_args["old_app_dec"],
        old_frame_pa=calc_uvw_args["old_frame_pa"],
    )

    uvw_base_late_pa_check = phs_utils.calc_uvw(
        frame_pa=calc_uvw_args["frame_pa"],
        use_ant_pos=False,
        uvw_array=temp_uvw,
        old_frame_pa=calc_uvw_args["old_frame_pa"],
    )

    np.testing.assert_allclose(
        uvw_base_check, uvw_base_late_pa_check, rtol=0, atol=1e-3
    )


@pytest.mark.filterwarnings('ignore:ERFA function "pmsafe" yielded')
@pytest.mark.filterwarnings('ignore:ERFA function "dtdtf" yielded')
@pytest.mark.filterwarnings('ignore:ERFA function "utcut1" yielded')
@pytest.mark.filterwarnings('ignore:ERFA function "utctai" yielded')
@pytest.mark.parametrize(
    "arg_dict,msg",
    (
        [{"library": "xyz"}, "Requested coordinate transformation library is not"],
        [{"icrs_ra": np.arange(10)}, "ra and dec must be the same shape."],
        [{"icrs_dec": np.arange(10)}, "ra and dec must be the same shape."],
        [{"pm_ra": np.arange(10)}, "pm_ra must be the same shape as ra and dec."],
        [{"pm_dec": np.arange(10)}, "pm_dec must be the same shape as ra and dec."],
        [{"dist": np.arange(10)}, "dist must be the same shape as ra and dec."],
        [{"vrad": np.arange(10)}, "vrad must be the same shape as ra and dec."],
        [
            {
                "icrs_ra": [0, 0],
                "icrs_dec": [0, 0],
                "pm_ra": None,
                "pm_dec": None,
                "dist": None,
                "vrad": None,
            },
            "time_array must be of either of",
        ],
        [{"time_array": 0.0, "library": "novas"}, "No current support for JPL ephems"],
    ),
)
def test_transform_icrs_to_app_arg_errs(astrometry_args, arg_dict, msg):
    """
    Check for argument errors with transform_icrs_to_app
    """
    if "library" in arg_dict and arg_dict["library"] == "novas":
        pytest.importorskip("novas")
    default_args = astrometry_args.copy()
    for key in arg_dict:
        default_args[key] = arg_dict[key]

    # Start w/ the transform_icrs_to_app block
    with pytest.raises(ValueError, match=msg):
        phs_utils.transform_icrs_to_app(
            time_array=default_args["time_array"],
            ra=default_args["icrs_ra"],
            dec=default_args["icrs_dec"],
            telescope_loc=default_args["telescope_loc"],
            telescope_frame=default_args["telescope_frame"],
            pm_ra=default_args["pm_ra"],
            pm_dec=default_args["pm_dec"],
            dist=default_args["dist"],
            vrad=default_args["vrad"],
            epoch=default_args["epoch"],
            astrometry_library=default_args["library"],
        )


def test_transform_icrs_to_app_no_novas_error(astrometry_args):
    try:
        import novas_de405  # noqa
        from novas import compat as novas  # noqa
        from novas.compat import eph_manager  # noqa
    except ImportError:
        with pytest.raises(
            ImportError,
            match="novas and/or novas_de405 are not installed but is required for "
            "NOVAS functionality",
        ):
            phs_utils.transform_icrs_to_app(
                time_array=astrometry_args["time_array"],
                ra=astrometry_args["icrs_ra"],
                dec=astrometry_args["icrs_dec"],
                telescope_loc=astrometry_args["telescope_loc"],
                telescope_frame=astrometry_args["telescope_frame"],
                pm_ra=astrometry_args["pm_ra"],
                pm_dec=astrometry_args["pm_dec"],
                dist=astrometry_args["dist"],
                vrad=astrometry_args["vrad"],
                epoch=astrometry_args["epoch"],
                astrometry_library="novas",
            )


@pytest.mark.parametrize(
    "arg_dict,msg",
    (
        [{"library": "xyz"}, "Requested coordinate transformation library is not"],
        [{"app_ra": np.arange(10)}, "app_ra and app_dec must be the same shape."],
        [{"app_dec": np.arange(10)}, "app_ra and app_dec must be the same shape."],
        [{"time_array": np.arange(10)}, "time_array must be of either of length 1"],
    ),
)
def test_transform_app_to_icrs_arg_errs(astrometry_args, arg_dict, msg):
    """
    Check for argument errors with transform_app_to_icrs
    """
    default_args = astrometry_args.copy()
    for key in arg_dict:
        default_args[key] = arg_dict[key]

    with pytest.raises(ValueError, match=msg):
        phs_utils.transform_app_to_icrs(
            time_array=default_args["time_array"],
            app_ra=default_args["app_ra"],
            app_dec=default_args["app_dec"],
            telescope_loc=default_args["telescope_loc"],
            telescope_frame=default_args["telescope_frame"],
            astrometry_library=default_args["library"],
        )


def test_transform_sidereal_coords_arg_errs():
    """
    Check for argument errors with transform_sidereal_coords
    """
    # Next on to sidereal to sidereal
    with pytest.raises(ValueError, match="lon and lat must be the same shape."):
        phs_utils.transform_sidereal_coords(
            longitude=[0.0],
            latitude=[0.0, 1.0],
            in_coord_frame="fk5",
            out_coord_frame="icrs",
            in_coord_epoch="J2000.0",
            time_array=[0.0, 1.0, 2.0],
        )

    with pytest.raises(ValueError, match="Shape of time_array must be either that of "):
        phs_utils.transform_sidereal_coords(
            longitude=[0.0, 1.0],
            latitude=[0.0, 1.0],
            in_coord_frame="fk4",
            out_coord_frame="fk4",
            in_coord_epoch=1950.0,
            out_coord_epoch=1984.0,
            time_array=[0.0, 1.0, 2.0],
        )


@pytest.mark.filterwarnings('ignore:ERFA function "d2dtf" yielded')
@pytest.mark.parametrize(
    ["arg_dict", "msg"],
    [
        [
            {"force_lookup": True, "time_array": np.arange(100000)},
            "Requesting too many individual ephem points from JPL-Horizons.",
        ],
        [{"force_lookup": False, "high_cadence": True}, "Too many ephem points"],
        [{"time_array": np.arange(10)}, "No current support for JPL ephems outside"],
        [{"targ_name": "whoami"}, "Target ID is not recognized in either the small"],
        [{"telescope_loc": "foo"}, "telescope_loc is not a valid type:"],
    ],
)
def test_lookup_jplhorizons_arg_errs(arg_dict, msg):
    """
    Check for argument errors with lookup_jplhorizons.
    """
    # Don't do this test if we don't have astroquery loaded
    pytest.importorskip("astroquery")

    from ssl import SSLError

    from requests import RequestException

    default_args = {
        "targ_name": "Mars",
        "time_array": np.array([0.0, 1000.0]) + 2456789.0,
        "telescope_loc": EarthLocation.from_geodetic(0, 0, height=0.0),
        "high_cadence": False,
        "force_lookup": None,
    }

    for key in arg_dict:
        default_args[key] = arg_dict[key]

    # We have to handle this piece a bit carefully, since some queries fail due to
    # intermittent failures connecting to the JPL-Horizons service.
    with pytest.raises(Exception) as cm:
        phs_utils.lookup_jplhorizons(
            default_args["targ_name"],
            default_args["time_array"],
            telescope_loc=default_args["telescope_loc"],
            high_cadence=default_args["high_cadence"],
            force_indv_lookup=default_args["force_lookup"],
        )

    if issubclass(cm.type, RequestException) or issubclass(cm.type, SSLError):
        pytest.skip("SSL/Connection error w/ JPL Horizons")

    assert issubclass(cm.type, ValueError)
    assert str(cm.value).startswith(msg)


def test_lookup_jplhorizons_moon_err():
    """
    Check for argument errors with lookup_jplhorizons.
    """
    # Don't do this test if we don't have astroquery loaded
    pytest.importorskip("astroquery")
    pytest.importorskip("lunarsky")

    from ssl import SSLError

    from lunarsky import MoonLocation
    from requests import RequestException

    default_args = {
        "targ_name": "Mars",
        "time_array": np.array([0.0, 1000.0]) + 2456789.0,
        "telescope_loc": MoonLocation.from_selenodetic(0.6875, 24.433, 0),
        "high_cadence": False,
        "force_lookup": None,
    }

    # We have to handle this piece a bit carefully, since some queries fail due to
    # intermittent failures connecting to the JPL-Horizons service.
    with pytest.raises(Exception) as cm:
        phs_utils.lookup_jplhorizons(
            default_args["targ_name"],
            default_args["time_array"],
            telescope_loc=default_args["telescope_loc"],
            high_cadence=default_args["high_cadence"],
            force_indv_lookup=default_args["force_lookup"],
        )

    if issubclass(cm.type, RequestException) or issubclass(cm.type, SSLError):
        pytest.skip("SSL/Connection error w/ JPL Horizons")

    assert issubclass(cm.type, NotImplementedError)
    assert str(cm.value).startswith(
        "Cannot lookup JPL positions for telescopes with a MoonLocation"
    )


def test_lookup_jplhorizons_no_astroquery_err():
    # We have to handle this piece a bit carefully, since some queries fail due to
    # intermittent failures connecting to the JPL-Horizons service.
    try:
        import astroquery  # noqa
    except ImportError:
        with pytest.raises(
            ImportError,
            match="astroquery is not installed but is required for planet "
            "ephemeris functionality",
        ):
            phs_utils.lookup_jplhorizons(
                target_name="Mars",
                time_array=np.array([0.0, 1000.0]) + 2456789.0,
                telescope_loc=EarthLocation.from_geodetic(0, 0, height=0.0),
            )


@pytest.mark.parametrize(
    "bad_arg,msg",
    [
        ["etimes", "ephem_ra must have the same shape as ephem_times."],
        ["ra", "ephem_ra must have the same shape as ephem_times."],
        ["dec", "ephem_dec must have the same shape as ephem_times."],
        ["dist", "ephem_dist must have the same shape as ephem_times."],
        ["vel", "ephem_vel must have the same shape as ephem_times."],
    ],
)
def test_interpolate_ephem_arg_errs(bad_arg, msg):
    """
    Check for argument errors with interpolate_ephem
    """
    # Now moving on to the interpolation scheme
    with pytest.raises(ValueError, match=msg):
        phs_utils.interpolate_ephem(
            time_array=0.0,
            ephem_times=0.0 if (bad_arg == "etimes") else [0.0, 1.0],
            ephem_ra=0.0 if (bad_arg == "ra") else [0.0, 1.0],
            ephem_dec=0.0 if (bad_arg == "dec") else [0.0, 1.0],
            ephem_dist=0.0 if (bad_arg == "dist") else [0.0, 1.0],
            ephem_vel=0.0 if (bad_arg == "vel") else [0.0, 1.0],
        )


def test_calc_app_coords_arg_errs():
    """
    Check for argument errors with calc_app_coords
    """
    # Now on to app_coords
    with pytest.raises(ValueError, match="Object type whoknows is not recognized."):
        phs_utils.calc_app_coords(
            lon_coord=0.0, lat_coord=0.0, telescope_loc=(0, 1, 2), coord_type="whoknows"
        )


def test_transform_multi_sidereal_coords(astrometry_args):
    """
    Perform some basic tests to verify that we can transform between sidereal frames
    with multiple coordinates.
    """
    # Check and make sure that we can deal with non-singleton times or coords with
    # singleton coords and times, respectively.
    check_ra, check_dec = phs_utils.transform_sidereal_coords(
        longitude=astrometry_args["icrs_ra"] * np.ones(2),
        latitude=astrometry_args["icrs_dec"] * np.ones(2),
        in_coord_frame="icrs",
        out_coord_frame="fk5",
        in_coord_epoch=2000.0,
        out_coord_epoch=2000.0,
        time_array=astrometry_args["time_array"][0] * np.ones(2),
    )
    assert np.all(np.equal(astrometry_args["fk5_ra"], check_ra))
    assert np.all(np.equal(astrometry_args["fk5_dec"], check_dec))


def test_transform_fk5_fk4_icrs_loop(astrometry_args):
    """
    Do a roundtrip test between ICRS, FK5, FK4 and back to ICRS to verify that we can
    handle transformation between different sidereal frames correctly.
    """
    # Now do a triangle between ICRS -> FK5 -> FK4 -> ICRS. If all is working well,
    # then we should recover the same position we started with.
    fk5_ra, fk5_dec = phs_utils.transform_sidereal_coords(
        longitude=astrometry_args["icrs_ra"],
        latitude=astrometry_args["icrs_dec"],
        in_coord_frame="icrs",
        out_coord_frame="fk5",
        in_coord_epoch=2000.0,
        out_coord_epoch=2000.0,
        time_array=astrometry_args["time_array"][0],
    )

    fk4_ra, fk4_dec = phs_utils.transform_sidereal_coords(
        longitude=fk5_ra,
        latitude=fk5_dec,
        in_coord_frame="fk5",
        out_coord_frame="fk4",
        in_coord_epoch="J2000.0",
        out_coord_epoch="B1950.0",
    )

    check_ra, check_dec = phs_utils.transform_sidereal_coords(
        longitude=fk4_ra,
        latitude=fk4_dec,
        in_coord_frame="fk4",
        out_coord_frame="icrs",
        in_coord_epoch="B1950.0",
        out_coord_epoch="J2000.0",
    )

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(check_coord.separation(astrometry_args["icrs_coord"]).uarcsec < 0.1)


@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
@pytest.mark.parametrize("in_lib", ["erfa", "astropy"])
@pytest.mark.parametrize("out_lib", ["erfa", "astropy"])
def test_roundtrip_icrs(astrometry_args, telescope_frame, selenoid, in_lib, out_lib):
    """
    Performs a roundtrip test to verify that one can transform between
    ICRS <-> topocentric to the precision limit, without running into
    issues.
    """
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]

    if telescope_frame == "mcmf" and in_lib != "astropy":
        with pytest.raises(
            NotImplementedError,
            match="MoonLocation telescopes are only supported with the 'astropy' "
            "astrometry library",
        ):
            app_ra, app_dec = phs_utils.transform_icrs_to_app(
                time_array=astrometry_args["time_array"],
                ra=astrometry_args["icrs_ra"],
                dec=astrometry_args["icrs_dec"],
                telescope_loc=telescope_loc,
                telescope_frame=telescope_frame,
                ellipsoid=selenoid,
                epoch=astrometry_args["epoch"],
                astrometry_library=in_lib,
            )
        return

    if telescope_frame == "mcmf" and out_lib == "astropy":
        kwargs = {"telescope_frame": telescope_frame, "ellipsoid": selenoid}
    else:
        # don't pass telescope frame here so something still happens if frame and
        # astrometry lib conflict
        kwargs = {}

    app_ra, app_dec = phs_utils.transform_icrs_to_app(
        time_array=astrometry_args["time_array"],
        ra=astrometry_args["icrs_ra"],
        dec=astrometry_args["icrs_dec"],
        telescope_loc=telescope_loc,
        epoch=astrometry_args["epoch"],
        astrometry_library=in_lib,
        **kwargs,
    )

    if telescope_frame == "mcmf" and out_lib != "astropy":
        with pytest.raises(
            NotImplementedError,
            match="MoonLocation telescopes are only supported with the 'astropy' "
            "astrometry library",
        ):
            check_ra, check_dec = phs_utils.transform_app_to_icrs(
                time_array=astrometry_args["time_array"],
                app_ra=app_ra,
                app_dec=app_dec,
                telescope_loc=telescope_loc,
                telescope_frame=telescope_frame,
                ellipsoid=selenoid,
                astrometry_library=out_lib,
            )
        return

    if telescope_frame == "mcmf":
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        try:
            check_ra, check_dec = phs_utils.transform_app_to_icrs(
                time_array=astrometry_args["time_array"],
                app_ra=app_ra,
                app_dec=app_dec,
                telescope_loc=telescope_loc,
                astrometry_library=out_lib,
                **kwargs,
            )
        except SpiceUNKNOWNFRAME as err:
            pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))
    else:
        check_ra, check_dec = phs_utils.transform_app_to_icrs(
            time_array=astrometry_args["time_array"],
            app_ra=app_ra,
            app_dec=app_dec,
            telescope_loc=telescope_loc,
            astrometry_library=out_lib,
            **kwargs,
        )

    check_coord = SkyCoord(check_ra, check_dec, unit="rad", frame="icrs")
    # Verify that everything agrees to better than µas-level accuracy if the
    # libraries are the same, otherwise to 100 µas if cross-comparing libraries
    if in_lib == out_lib:
        assert np.all(
            astrometry_args["icrs_coord"].separation(check_coord).uarcsec < 1.0
        )
    else:
        assert np.all(
            astrometry_args["icrs_coord"].separation(check_coord).uarcsec < 100.0
        )

    if selenoid == "SPHERE":
        # check defaults
        app_ra, app_dec = phs_utils.transform_icrs_to_app(
            time_array=astrometry_args["time_array"],
            ra=astrometry_args["icrs_ra"],
            dec=astrometry_args["icrs_dec"],
            telescope_loc=telescope_loc,
            epoch=astrometry_args["epoch"],
            astrometry_library=in_lib,
            telescope_frame=telescope_frame,
        )
        check_ra, check_dec = phs_utils.transform_app_to_icrs(
            time_array=astrometry_args["time_array"],
            app_ra=app_ra,
            app_dec=app_dec,
            telescope_loc=telescope_loc,
            astrometry_library=out_lib,
            telescope_frame=telescope_frame,
        )
        check_coord = SkyCoord(check_ra, check_dec, unit="rad", frame="icrs")
        # Verify that everything agrees to better than µas-level accuracy if the
        # libraries are the same, otherwise to 100 µas if cross-comparing libraries
        assert np.all(
            astrometry_args["icrs_coord"].separation(check_coord).uarcsec < 1.0
        )


def test_calc_parallactic_angle():
    """
    A relatively straightforward test to verify that we recover the parallactic
    angles we expect given some known inputs
    """
    expected_vals = np.array([1.0754290375762232, 0.0, -0.6518070715011698])
    meas_vals = phs_utils.calc_parallactic_angle(
        app_ra=[0.0, 1.0, 2.0],
        app_dec=[-1.0, 0.0, 1.0],
        lst_array=[2.0, 1.0, 0],
        telescope_lat=1.0,
    )
    # Make sure things agree to better than ~0.1 uas (as it definitely should)
    np.testing.assert_allclose(expected_vals, meas_vals, rtol=0.0, atol=1e-12)


def test_calc_frame_pos_angle():
    """
    Verify that we recover frame position angles correctly
    """
    # First test -- plug in "topo" for the frame, which should always produce an
    # array of all zeros (the topo frame is what the apparent coords are in)
    frame_pa = phs_utils.calc_frame_pos_angle(
        time_array=np.array([2456789.0] * 100),
        app_ra=np.arange(100) * (np.pi / 50),
        app_dec=np.zeros(100),
        telescope_loc=(0, 0, 0),
        ref_frame="topo",
    )
    assert len(frame_pa) == 100
    assert np.all(frame_pa == 0.0)
    # PA of zero degrees (they're always aligned)
    # Next test -- plug in J2000 and see that we actually get back a frame PA
    # of basically 0 degrees.
    j2000_jd = Time(2000.0, format="jyear").utc.jd
    frame_pa = phs_utils.calc_frame_pos_angle(
        time_array=np.array([j2000_jd] * 100),
        app_ra=np.arange(100) * (np.pi / 50),
        app_dec=np.zeros(100),
        telescope_loc=(0, 0, 0),
        ref_frame="fk5",
        ref_epoch=2000.0,
    )
    # At J2000, the only frame PA terms come from aberation, which basically max out
    # at ~< 1e-4 rad. Check to make sure that lines up with what we measure.
    assert np.all(np.abs(frame_pa) < 1e-4)

    # JD 2458849.5 is Jan-01-2020, so 20 years of parallax ought to have accumulated
    # (with about 1 arcmin/yr of precession). Make sure these values are sensible
    frame_pa = phs_utils.calc_frame_pos_angle(
        time_array=np.array([2458849.5] * 100),
        app_ra=np.arange(100) * (np.pi / 50),
        app_dec=np.zeros(100),
        telescope_loc=(0, 0, 0),
        ref_frame="fk5",
        ref_epoch=2000.0,
    )
    assert np.all(np.abs(frame_pa) < 20 * (50.3 / 3600) * (np.pi / 180.0))
    # Check the PA at a couple of chosen points, which just so happen to be very close
    # in magnitude (as they're basically in the same plane as the motion of the Earth)
    assert np.isclose(frame_pa[25], 0.001909957544309159, rtol=0, atol=utils.RADIAN_TOL)
    assert np.isclose(
        frame_pa[-25], -0.0019098101664715339, rtol=0, atol=utils.RADIAN_TOL
    )


def test_jphl_lookup(astrometry_args):
    """
    A very simple lookup query to verify that the astroquery tools for accessing
    JPL-Horizons are working. This test is very limited, on account of not wanting to
    slam JPL w/ coordinate requests.
    """
    pytest.importorskip("astroquery")

    from ssl import SSLError

    from requests import RequestException

    # If we can't connect to JPL-Horizons, then skip this test and don't outright fail.
    try:
        [ephem_times, ephem_ra, ephem_dec, ephem_dist, ephem_vel] = (
            phs_utils.lookup_jplhorizons("Sun", 2456789.0)
        )
    except (SSLError, RequestException) as err:
        pytest.skip("SSL/Connection error w/ JPL Horizons: " + str(err))

    assert np.all(np.equal(ephem_times, 2456789.0))
    np.testing.assert_allclose(
        ephem_ra, 0.8393066751804976, rtol=0, atol=utils.RADIAN_TOL
    )
    np.testing.assert_allclose(
        ephem_dec, 0.3120687480116649, rtol=0, atol=utils.RADIAN_TOL
    )
    np.testing.assert_allclose(ephem_dist, 1.00996185750717, rtol=0, atol=1e-3)
    np.testing.assert_allclose(ephem_vel, 0.386914, rtol=0, atol=1e-3)

    # check calling lookup_jplhorizons with EarthLocation vs lat/lon/alt passed
    try:
        ephem_info_latlon = phs_utils.lookup_jplhorizons(
            "Sun", 2456789.0, telescope_loc=astrometry_args["telescope_loc"]
        )
        ephem_info_el = phs_utils.lookup_jplhorizons(
            "Sun",
            2456789.0,
            telescope_loc=EarthLocation.from_geodetic(
                lat=astrometry_args["telescope_loc"][0] * units.rad,
                lon=astrometry_args["telescope_loc"][1] * units.rad,
                height=astrometry_args["telescope_loc"][2] * units.m,
            ),
        )
    except (SSLError, RequestException) as err:
        pytest.skip("SSL/Connection error w/ JPL Horizons: " + str(err))

    for ind, item in enumerate(ephem_info_latlon):
        assert item == ephem_info_el[ind]


def test_ephem_interp_one_point():
    """
    These tests do some simple checks to verify that the interpolator behaves properly
    when only being provided singleton values.
    """
    # First test the case where there is only one ephem point, and thus everything
    # takes on that value
    time_array = np.arange(100) * 0.01
    ephem_times = np.array([0])
    ephem_ra = np.array([1.0])
    ephem_dec = np.array([2.0])
    ephem_dist = np.array([3.0])
    ephem_vel = np.array([4.0])

    ra_vals0, dec_vals0, dist_vals0, vel_vals0 = phs_utils.interpolate_ephem(
        time_array=time_array,
        ephem_times=ephem_times,
        ephem_ra=ephem_ra,
        ephem_dec=ephem_dec,
        ephem_dist=ephem_dist,
        ephem_vel=ephem_vel,
    )

    assert np.all(ra_vals0 == 1.0)
    assert np.all(dec_vals0 == 2.0)
    assert np.all(dist_vals0 == 3.0)
    assert np.all(vel_vals0 == 4.0)


def test_ephem_interp_multi_point():
    """
    Test that ephem coords are interpolated correctly when supplying more than a
    singleton value for the various arrays.
    """
    # Next test the case where the ephem only has a couple of points, in which case the
    # code will default to using a simple, linear interpolation scheme.
    time_array = np.arange(100) * 0.01
    ephem_times = np.array([0, 1])
    ephem_ra = np.array([0, 1]) + 1.0
    ephem_dec = np.array([0, 1]) + 2.0
    ephem_dist = np.array([0, 1]) + 3.0
    ephem_vel = np.array([0, 1]) + 4.0

    ra_vals1, dec_vals1, dist_vals1, vel_vals1 = phs_utils.interpolate_ephem(
        time_array=time_array,
        ephem_times=ephem_times,
        ephem_ra=ephem_ra,
        ephem_dec=ephem_dec,
        ephem_dist=ephem_dist,
        ephem_vel=ephem_vel,
    )

    # When there are lots more data points, the interpolator will default to using a
    # cubic spline, which _should_ be very close (to numerical precision limits) to what
    # we get with the method above.
    ephem_times = np.arange(11) * 0.1
    ephem_ra = (np.arange(11) * 0.1) + 1.0
    ephem_dec = (np.arange(11) * 0.1) + 2.0
    ephem_dist = (np.arange(11) * 0.1) + 3.0
    ephem_vel = (np.arange(11) * 0.1) + 4.0

    ra_vals2, dec_vals2, dist_vals2, vel_vals2 = phs_utils.interpolate_ephem(
        time_array=time_array,
        ephem_times=ephem_times,
        ephem_ra=ephem_ra,
        ephem_dec=ephem_dec,
        ephem_dist=ephem_dist,
        ephem_vel=ephem_vel,
    )

    # Make sure that everything is consistent to floating point precision
    np.testing.assert_allclose(ra_vals1, ra_vals2, 1e-15, 0.0)
    np.testing.assert_allclose(dec_vals1, dec_vals2, 1e-15, 0.0)
    np.testing.assert_allclose(dist_vals1, dist_vals2, 1e-15, 0.0)
    np.testing.assert_allclose(vel_vals1, vel_vals2, 1e-15, 0.0)
    np.testing.assert_allclose(time_array + 1.0, ra_vals2, 1e-15, 0.0)
    np.testing.assert_allclose(time_array + 2.0, dec_vals2, 1e-15, 0.0)
    np.testing.assert_allclose(time_array + 3.0, dist_vals2, 1e-15, 0.0)
    np.testing.assert_allclose(time_array + 4.0, vel_vals2, 1e-15, 0.0)


@pytest.mark.parametrize("frame", ["icrs", "fk5"])
@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_calc_app_sidereal(astrometry_args, frame, telescope_frame, selenoid):
    """
    Tests that we can calculate app coords for sidereal objects
    """
    # First step is to check and make sure we can do sidereal coords. This is the most
    # basic thing to check, so this really _should work.
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
    else:
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        telescope_loc = astrometry_args["moon_telescope_loc"]

    try:
        check_ra, check_dec = phs_utils.calc_app_coords(
            lon_coord=(
                astrometry_args["fk5_ra"]
                if (frame == "fk5")
                else astrometry_args["icrs_ra"]
            ),
            lat_coord=(
                astrometry_args["fk5_dec"]
                if (frame == "fk5")
                else astrometry_args["icrs_dec"]
            ),
            coord_type="sidereal",
            telescope_loc=telescope_loc,
            telescope_frame=telescope_frame,
            ellipsoid=selenoid,
            time_array=astrometry_args["time_array"],
            coord_frame=frame,
            coord_epoch=astrometry_args["epoch"],
        )
    except SpiceUNKNOWNFRAME as err:
        pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")

    if telescope_frame == "itrs":
        app_coord = astrometry_args["app_coord"]
    else:
        app_coord = astrometry_args["moon_app_coord"][selenoid]

    assert np.all(app_coord.separation(check_coord).uarcsec < 1.0)


@pytest.mark.parametrize("frame", ["icrs", "fk5"])
@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_calc_app_ephem(astrometry_args, frame, telescope_frame, selenoid):
    """
    Tests that we can calculate app coords for ephem objects
    """
    # Next, see what happens when we pass an ephem. Note that this is just a single
    # point ephem, so its not testing any of the fancy interpolation, but we have other
    # tests for poking at that. The two tests here are to check bot the ICRS and FK5
    # paths through the ephem.
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
        err_type = None
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        err_type = SpiceUNKNOWNFRAME

    if frame == "fk5":
        ephem_ra = astrometry_args["fk5_ra"]
        ephem_dec = astrometry_args["fk5_dec"]
    else:
        ephem_ra = np.array([astrometry_args["icrs_ra"]])
        ephem_dec = np.array([astrometry_args["icrs_dec"]])

    ephem_times = np.array([astrometry_args["time_array"][0]])

    try:
        check_ra, check_dec = phs_utils.calc_app_coords(
            lon_coord=ephem_ra,
            lat_coord=ephem_dec,
            coord_times=ephem_times,
            coord_type="ephem",
            telescope_loc=telescope_loc,
            telescope_frame=telescope_frame,
            ellipsoid=selenoid,
            time_array=astrometry_args["time_array"],
            coord_epoch=astrometry_args["epoch"],
            coord_frame=frame,
        )
    except err_type as err:
        pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")

    if telescope_frame == "itrs":
        app_coord = astrometry_args["app_coord"]
    else:
        app_coord = astrometry_args["moon_app_coord"][selenoid]
    assert np.all(app_coord.separation(check_coord).uarcsec < 1.0)


@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_calc_app_driftscan(astrometry_args, telescope_frame, selenoid):
    """
    Tests that we can calculate app coords for driftscan objects
    """
    # Now on to the driftscan, which takes in arguments in terms of az and el (and
    # the values we've given below should also be for zenith)
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]

    check_ra, check_dec = phs_utils.calc_app_coords(
        lon_coord=0.0,
        lat_coord=np.pi / 2.0,
        coord_type="driftscan",
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
        time_array=astrometry_args["time_array"],
    )
    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    if telescope_frame == "itrs":
        drift_coord = astrometry_args["drift_coord"]
    else:
        drift_coord = astrometry_args["moon_drift_coord"][selenoid]

    assert np.all(drift_coord.separation(check_coord).uarcsec < 1.0)


@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_calc_app_unprojected(astrometry_args, telescope_frame, selenoid):
    """
    Tests that we can calculate app coords for unphased objects
    """
    # Finally, check unprojected, which is forced to point toward zenith (unlike
    # driftscan, which is allowed to point at any az/el position)
    # use "unphased" to check for deprecation warning
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
        lst_array = astrometry_args["lst_array"]
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]
        lst_array = astrometry_args["moon_lst_array"][selenoid]

    check_ra, check_dec = phs_utils.calc_app_coords(
        lon_coord=None,
        lat_coord=None,
        coord_type="unprojected",
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
        time_array=astrometry_args["time_array"],
        lst_array=lst_array,
    )

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")

    if telescope_frame == "itrs":
        drift_coord = astrometry_args["drift_coord"]
    else:
        drift_coord = astrometry_args["moon_drift_coord"][selenoid]
    assert np.all(drift_coord.separation(check_coord).uarcsec < 1.0)


@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_calc_app_fk5_roundtrip(astrometry_args, telescope_frame, selenoid):
    # Do a round-trip with the two top-level functions and make sure they agree to
    # better than 1 µas, first in FK5
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]

    app_ra, app_dec = phs_utils.calc_app_coords(
        lon_coord=0.0,
        lat_coord=0.0,
        coord_type="sidereal",
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
        time_array=astrometry_args["time_array"],
        coord_frame="fk5",
        coord_epoch="J2000.0",
    )

    if telescope_frame == "mcmf":
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        try:
            check_ra, check_dec = phs_utils.calc_sidereal_coords(
                time_array=astrometry_args["time_array"],
                app_ra=app_ra,
                app_dec=app_dec,
                telescope_loc=telescope_loc,
                coord_frame="fk5",
                telescope_frame=telescope_frame,
                ellipsoid=selenoid,
                coord_epoch=2000.0,
            )
        except SpiceUNKNOWNFRAME as err:
            pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))
    else:
        check_ra, check_dec = phs_utils.calc_sidereal_coords(
            time_array=astrometry_args["time_array"],
            app_ra=app_ra,
            app_dec=app_dec,
            telescope_loc=telescope_loc,
            coord_frame="fk5",
            telescope_frame=telescope_frame,
            ellipsoid=selenoid,
            coord_epoch=2000.0,
        )

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(SkyCoord(0, 0, unit="rad").separation(check_coord).uarcsec < 1.0)

    if selenoid == "SPHERE":
        # check defaults

        app_ra, app_dec = phs_utils.calc_app_coords(
            lon_coord=0.0,
            lat_coord=0.0,
            coord_type="sidereal",
            telescope_loc=telescope_loc,
            telescope_frame=telescope_frame,
            time_array=astrometry_args["time_array"],
            coord_frame="fk5",
            coord_epoch="J2000.0",
        )

        if telescope_frame == "mcmf":
            from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

            try:
                check_ra, check_dec = phs_utils.calc_sidereal_coords(
                    time_array=astrometry_args["time_array"],
                    app_ra=app_ra,
                    app_dec=app_dec,
                    telescope_loc=telescope_loc,
                    coord_frame="fk5",
                    telescope_frame=telescope_frame,
                    coord_epoch=2000.0,
                )
            except SpiceUNKNOWNFRAME as err:
                pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))
        else:
            check_ra, check_dec = phs_utils.calc_sidereal_coords(
                time_array=astrometry_args["time_array"],
                app_ra=app_ra,
                app_dec=app_dec,
                telescope_loc=telescope_loc,
                coord_frame="fk5",
                telescope_frame=telescope_frame,
                coord_epoch=2000.0,
            )
        check_coord = SkyCoord(check_ra, check_dec, unit="rad")
        assert np.all(SkyCoord(0, 0, unit="rad").separation(check_coord).uarcsec < 1.0)


@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_calc_app_fk4_roundtrip(astrometry_args, telescope_frame, selenoid):
    # Finally, check and make sure that FK4 performs similarly
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]

    app_ra, app_dec = phs_utils.calc_app_coords(
        lon_coord=0.0,
        lat_coord=0.0,
        coord_type="sidereal",
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
        time_array=astrometry_args["time_array"],
        coord_frame="fk4",
        coord_epoch=1950.0,
    )

    if telescope_frame == "mcmf":
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        try:
            check_ra, check_dec = phs_utils.calc_sidereal_coords(
                time_array=astrometry_args["time_array"],
                app_ra=app_ra,
                app_dec=app_dec,
                telescope_loc=telescope_loc,
                coord_frame="fk4",
                telescope_frame=telescope_frame,
                ellipsoid=selenoid,
                coord_epoch=1950.0,
            )
        except SpiceUNKNOWNFRAME as err:
            pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))

    else:
        check_ra, check_dec = phs_utils.calc_sidereal_coords(
            time_array=astrometry_args["time_array"],
            app_ra=app_ra,
            app_dec=app_dec,
            telescope_loc=telescope_loc,
            coord_frame="fk4",
            telescope_frame=telescope_frame,
            ellipsoid=selenoid,
            coord_epoch=1950.0,
        )

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(SkyCoord(0, 0, unit="rad").separation(check_coord).uarcsec < 1.0)


@pytest.mark.filterwarnings('ignore:ERFA function "pmsafe" yielded 4 of')
@pytest.mark.filterwarnings('ignore:ERFA function "utcut1" yielded 2 of')
@pytest.mark.filterwarnings('ignore:ERFA function "d2dtf" yielded 1 of')
@pytest.mark.parametrize("use_extra", [True, False])
def test_astrometry_icrs_to_app(astrometry_args, use_extra):
    """
    Check for consistency beteen astrometry libraries when converting ICRS -> TOPP

    This test checks for consistency in apparent coordinate calculations using the
    three different libraries that are available to pyuvdata, namely: astropy, pyERFA,
    and python-novas. Between these three, we expect agreement within 100 µas in
    most instances, although for pyuvdata we tolerate differences of up to 1 mas since
    we don't expect to need astrometry better than this.
    """
    pytest.importorskip("novas")
    pytest.importorskip("novas_de405")
    # Do some basic cross-checking between the different astrometry libraries
    # to see if they all line up correctly.
    astrometry_list = ["novas", "erfa", "astropy"]
    coord_results = [None, None, None, None]

    # These values were indepedently calculated using erfa v1.7.2, which at the
    # time of coding agreed to < 1 mas with astropy v4.2.1 and novas 3.1.1.5. We
    # use those values here as a sort of history check to make sure that something
    # hasn't changed in the underlying astrometry libraries without being caught
    precalc_ra = np.array(
        [2.4736400623737507, 2.4736352750862760, 2.4736085367439893, 2.4734781687162820]
    )
    precalc_dec = np.array(
        [1.2329576409345270, 1.2329556410623417, 1.2329541289890513, 1.2328577308430242]
    )

    coord_results[3] = (precalc_ra, precalc_dec)

    kwargs = {}
    extra_args = ["pm_ra", "pm_dec", "vrad", "dist"]
    if use_extra:
        for key in extra_args:
            kwargs[key] = astrometry_args[key]
    else:
        # don't compare to precalc if not using extra arguments
        coord_results = coord_results[:-1]

    for idx, name in enumerate(astrometry_list):
        coord_results[idx] = phs_utils.transform_icrs_to_app(
            time_array=astrometry_args["time_array"],
            ra=astrometry_args["icrs_ra"],
            dec=astrometry_args["icrs_dec"],
            telescope_loc=astrometry_args["telescope_loc"],
            epoch=astrometry_args["epoch"],
            astrometry_library=name,
            **kwargs,
        )

    for idx in range(len(coord_results) - 1):
        for jdx in range(idx + 1, len(coord_results)):
            alpha_coord = SkyCoord(
                coord_results[idx][0], coord_results[idx][1], unit="rad"
            )
            beta_coord = SkyCoord(
                coord_results[jdx][0], coord_results[jdx][1], unit="rad"
            )
            assert np.all(alpha_coord.separation(beta_coord).marcsec < 1.0)


def test_astrometry_app_to_icrs(astrometry_args):
    """
    Check for consistency beteen astrometry libraries when converting TOPO -> ICRS

    This test checks for consistency between the pyERFA and astropy libraries for
    converting apparent coords back to ICRS. Between these two, we expect agreement
    within 100 µas in most instances, although for pyuvdata we tolerate differences of
    up to 1 mas since we don't expect to need astrometry better than this.
    """
    astrometry_list = ["erfa", "astropy"]
    coord_results = [None, None, None]

    # These values were indepedently calculated using erfa v1.7.2, which at the
    # time of coding agreed to < 1 mas with astropy v4.2.1. We again are using
    # those values here as a sort of history check to make sure that something
    # hasn't changed in the underlying astrometry libraries without being caught
    precalc_ra = np.array(
        [2.4623360300722170, 2.4623407989706756, 2.4623676572008280, 2.4624965192217900]
    )
    precalc_dec = np.array(
        [1.2350407132378372, 1.2350427272595987, 1.2350443204758008, 1.2351412288987034]
    )
    coord_results[2] = (precalc_ra, precalc_dec)

    for idx, name in enumerate(astrometry_list):
        # Note we're using icrs_ra and icrs_dec instead of app_ra and app_dec keys
        # because the above pre-calculated values were generated using the ICRS
        # coordinate values
        coord_results[idx] = phs_utils.transform_app_to_icrs(
            time_array=astrometry_args["time_array"],
            app_ra=astrometry_args["icrs_ra"],
            app_dec=astrometry_args["icrs_dec"],
            telescope_loc=astrometry_args["telescope_loc"],
            astrometry_library=name,
        )

    for idx in range(len(coord_results) - 1):
        for jdx in range(idx + 1, len(coord_results)):
            alpha_coord = SkyCoord(
                coord_results[idx][0], coord_results[idx][1], unit="rad"
            )
            beta_coord = SkyCoord(
                coord_results[jdx][0], coord_results[jdx][1], unit="rad"
            )
            assert np.all(alpha_coord.separation(beta_coord).marcsec < 1.0)


def test_sidereal_reptime(astrometry_args):
    """
    Check for equality when supplying a singleton time versus an array of identical
    values for transform_sidereal_coords
    """

    gcrs_ra, gcrs_dec = phs_utils.transform_sidereal_coords(
        longitude=astrometry_args["icrs_ra"] * np.ones(2),
        latitude=astrometry_args["icrs_dec"] * np.ones(2),
        in_coord_frame="icrs",
        out_coord_frame="gcrs",
        time_array=Time(astrometry_args["time_array"][0], format="jd"),
    )

    check_ra, check_dec = phs_utils.transform_sidereal_coords(
        longitude=astrometry_args["icrs_ra"] * np.ones(2),
        latitude=astrometry_args["icrs_dec"] * np.ones(2),
        in_coord_frame="icrs",
        out_coord_frame="gcrs",
        time_array=Time(astrometry_args["time_array"][0] * np.ones(2), format="jd"),
    )

    assert np.all(gcrs_ra == check_ra)
    assert np.all(gcrs_dec == check_dec)


@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_transform_icrs_to_app_time_obj(astrometry_args, telescope_frame, selenoid):
    """
    Test that we recover identical values when using a Time objects instead of a floats
    for the various time-related arguments in transform_icrs_to_app.
    """
    if telescope_frame == "itrs":
        telescope_loc = astrometry_args["telescope_loc"]
        err_type = None
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]
        from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

        err_type = SpiceUNKNOWNFRAME

    try:
        check_ra, check_dec = phs_utils.transform_icrs_to_app(
            time_array=Time(astrometry_args["time_array"], format="jd"),
            ra=astrometry_args["icrs_ra"],
            dec=astrometry_args["icrs_dec"],
            telescope_loc=telescope_loc,
            telescope_frame=telescope_frame,
            ellipsoid=selenoid,
            epoch=Time(astrometry_args["epoch"], format="jyear"),
        )
    except err_type as err:
        pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))

    if telescope_frame == "itrs":
        app_ra = astrometry_args["app_ra"]
        app_dec = astrometry_args["app_dec"]
    else:
        app_ra = astrometry_args["moon_app_ra"][selenoid]
        app_dec = astrometry_args["moon_app_dec"][selenoid]

    assert np.all(check_ra == app_ra)
    assert np.all(check_dec == app_dec)


def test_transform_app_to_icrs_objs(astrometry_args):
    """
    Test that we recover identical values when using Time/EarthLocation objects instead
    of floats for time_array and telescope_loc, respectively for transform_app_to_icrs.
    """
    telescope_loc = EarthLocation.from_geodetic(
        astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
        astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
        height=astrometry_args["telescope_loc"][2],
    )

    icrs_ra, icrs_dec = phs_utils.transform_app_to_icrs(
        time_array=astrometry_args["time_array"][0],
        app_ra=astrometry_args["app_ra"][0],
        app_dec=astrometry_args["app_dec"][0],
        telescope_loc=astrometry_args["telescope_loc"],
    )

    check_ra, check_dec = phs_utils.transform_app_to_icrs(
        time_array=Time(astrometry_args["time_array"][0], format="jd"),
        app_ra=astrometry_args["app_ra"][0],
        app_dec=astrometry_args["app_dec"][0],
        telescope_loc=telescope_loc,
    )

    assert np.all(check_ra == icrs_ra)
    assert np.all(check_dec == icrs_dec)


@pytest.mark.parametrize(["telescope_frame", "selenoid"], frame_selenoid)
def test_calc_app_coords_objs(astrometry_args, telescope_frame, selenoid):
    """
    Test that we recover identical values when using Time/EarthLocation objects instead
    of floats for time_array and telescope_loc, respectively for calc_app_coords.
    """
    if telescope_frame == "itrs":
        telescope_loc = EarthLocation.from_geodetic(
            astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
            astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
            height=astrometry_args["telescope_loc"][2],
        )
        TimeClass = Time
    else:
        from lunarsky import MoonLocation, Time as LTime

        telescope_loc = MoonLocation.from_selenodetic(
            astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
            astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
            height=astrometry_args["telescope_loc"][2],
            ellipsoid=selenoid,
        )
        TimeClass = LTime

    app_ra, app_dec = phs_utils.calc_app_coords(
        lon_coord=astrometry_args["icrs_ra"],
        lat_coord=astrometry_args["icrs_dec"],
        time_array=astrometry_args["time_array"][0],
        telescope_loc=astrometry_args["telescope_loc"],
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
    )

    check_ra, check_dec = phs_utils.calc_app_coords(
        lon_coord=astrometry_args["icrs_ra"],
        lat_coord=astrometry_args["icrs_dec"],
        time_array=TimeClass(astrometry_args["time_array"][0], format="jd"),
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
    )

    assert np.all(check_ra == app_ra)
    assert np.all(check_dec == app_dec)


def test_phasing_funcs():
    # these tests are based on a notebook where I tested against the mwa_tools
    # phasing code
    ra_hrs = 12.1
    dec_degs = -42.3
    mjd = 55780.1

    array_center_xyz = np.array([-2559454.08, 5095372.14, -2849057.18])
    lat_lon_alt = utils.LatLonAlt_from_XYZ(array_center_xyz)

    obs_time = Time(mjd, format="mjd", location=(lat_lon_alt[1], lat_lon_alt[0]))

    icrs_coord = SkyCoord(
        ra=Angle(ra_hrs, unit="hr"), dec=Angle(dec_degs, unit="deg"), obstime=obs_time
    )
    gcrs_coord = icrs_coord.transform_to("gcrs")

    # in east/north/up frame (relative to array center) in meters: (Nants, 3)
    ants_enu = np.array([-101.94, 156.41, 1.24])

    ant_xyz_abs = utils.ECEF_from_ENU(
        ants_enu,
        latitude=lat_lon_alt[0],
        longitude=lat_lon_alt[1],
        altitude=lat_lon_alt[2],
    )

    array_center_coord = SkyCoord(
        x=array_center_xyz[0] * units.m,
        y=array_center_xyz[1] * units.m,
        z=array_center_xyz[2] * units.m,
        frame="itrs",
        obstime=obs_time,
    )

    itrs_coord = SkyCoord(
        x=ant_xyz_abs[0] * units.m,
        y=ant_xyz_abs[1] * units.m,
        z=ant_xyz_abs[2] * units.m,
        frame="itrs",
        obstime=obs_time,
    )

    gcrs_array_center = array_center_coord.transform_to("gcrs")
    gcrs_from_itrs_coord = itrs_coord.transform_to("gcrs")

    gcrs_rel = (
        (gcrs_from_itrs_coord.cartesian - gcrs_array_center.cartesian).get_xyz().T
    )

    gcrs_uvw = phs_utils.old_uvw_calc(
        gcrs_coord.ra.rad, gcrs_coord.dec.rad, gcrs_rel.value
    )

    mwa_tools_calcuvw_u = -97.122828
    mwa_tools_calcuvw_v = 50.388281
    mwa_tools_calcuvw_w = -151.27976

    np.testing.assert_allclose(gcrs_uvw[0, 0], mwa_tools_calcuvw_u, rtol=0, atol=1e-3)
    np.testing.assert_allclose(gcrs_uvw[0, 1], mwa_tools_calcuvw_v, rtol=0, atol=1e-3)
    np.testing.assert_allclose(gcrs_uvw[0, 2], mwa_tools_calcuvw_w, rtol=0, atol=1e-3)

    # also test unphasing
    temp2 = phs_utils.undo_old_uvw_calc(
        gcrs_coord.ra.rad, gcrs_coord.dec.rad, np.squeeze(gcrs_uvw)
    )
    np.testing.assert_allclose(gcrs_rel.value, np.squeeze(temp2), rtol=0, atol=1e-3)


def test_calc_app_coords_time_obj():
    # Generate ra/dec of zenith at time in the phase_frame coordinate system
    # to use for phasing
    telescope_location = EarthLocation.from_geodetic(lon=0, lat=1 * units.rad)

    # JD is arbitrary
    jd = 2454600

    zenith_coord = SkyCoord(
        alt=90 * units.deg,
        az=0 * units.deg,
        obstime=Time(jd, format="jd"),
        frame="altaz",
        location=telescope_location,
    )
    zenith_coord = zenith_coord.transform_to("icrs")

    obstime = Time(jd + (np.array([-1, 0, 1]) / 24.0), format="jd")

    ra = zenith_coord.ra.to_value("rad")
    dec = zenith_coord.dec.to_value("rad")
    app_ra_to, app_dec_to = phs_utils.calc_app_coords(
        lon_coord=ra,
        lat_coord=dec,
        time_array=obstime,
        telescope_loc=telescope_location,
    )

    app_ra_nto, app_dec_nto = phs_utils.calc_app_coords(
        lon_coord=ra,
        lat_coord=dec,
        time_array=obstime.utc.jd,
        telescope_loc=telescope_location,
    )

    np.testing.assert_allclose(app_ra_to, app_ra_nto, rtol=0, atol=utils.RADIAN_TOL)
    np.testing.assert_allclose(app_dec_to, app_dec_nto, rtol=0, atol=utils.RADIAN_TOL)


@pytest.mark.skipif(len(frame_selenoid) > 1, reason="lunarsky installed")
def test_uvw_track_generator_errs():
    with pytest.raises(
        ImportError, match="Need to install `lunarsky` package to work with MCMF frame."
    ):
        utils.uvw_track_generator(telescope_loc=(0, 0, 0), telescope_frame="MCMF")


@pytest.mark.parametrize("flip_u", [False, True])
@pytest.mark.parametrize("use_uvw", [False, True])
@pytest.mark.parametrize("use_earthloc", [False, True])
@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.filterwarnings("ignore:> 25 ms errors detected reading in LST values")
def test_uvw_track_generator(flip_u, use_uvw, use_earthloc):
    sma_mir = UVData.from_file(os.path.join(DATA_PATH, "sma_test.mir"))
    sma_mir.set_lsts_from_time_array()
    sma_mir._set_app_coords_helper()
    sma_mir.set_uvws_from_antenna_positions()
    if not use_uvw:
        # Just subselect the antennas in the dataset
        sma_mir.telescope.antenna_positions = sma_mir.telescope.antenna_positions[
            [0, 3], :
        ]

    if use_earthloc:
        telescope_loc = EarthLocation.from_geodetic(
            lon=sma_mir.telescope.location_lat_lon_alt_degrees[1],
            lat=sma_mir.telescope.location_lat_lon_alt_degrees[0],
            height=sma_mir.telescope.location_lat_lon_alt_degrees[2],
        )
    else:
        telescope_loc = sma_mir.telescope.location_lat_lon_alt_degrees

    if use_uvw:
        sma_copy = sma_mir.copy()
        sma_copy.unproject_phase()
        uvw_array = sma_copy.uvw_array
    else:
        uvw_array = None

    cat_dict = sma_mir.phase_center_catalog[1]
    gen_results = utils.uvw_track_generator(
        lon_coord=cat_dict["cat_lon"],
        lat_coord=cat_dict["cat_lat"],
        coord_frame=cat_dict["cat_frame"],
        coord_epoch=cat_dict["cat_epoch"],
        telescope_loc=telescope_loc,
        time_array=sma_mir.time_array if use_uvw else sma_mir.time_array[0],
        antenna_positions=(
            sma_mir.telescope.antenna_positions if uvw_array is None else None
        ),
        force_postive_u=flip_u,
        uvw_array=uvw_array,
    )

    assert sma_mir._phase_center_app_ra.compare_value(gen_results["app_ra"])
    assert sma_mir._phase_center_app_dec.compare_value(gen_results["app_dec"])
    assert sma_mir._phase_center_frame_pa.compare_value(gen_results["frame_pa"])
    assert sma_mir._lst_array.compare_value(gen_results["lst"])
    if flip_u:
        assert sma_mir._uvw_array.compare_value(-gen_results["uvw"])
    else:
        assert sma_mir._uvw_array.compare_value(gen_results["uvw"])


@pytest.mark.parametrize("selenoid", ["SPHERE", "GSFC", "GRAIL23", "CE-1-LAM-GEO"])
def test_uvw_track_generator_moon(selenoid):
    pytest.importorskip("lunarsky")
    # Note this isn't a particularly deep test, but it at least exercises the code.
    from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

    try:
        gen_results = utils.uvw_track_generator(
            lon_coord=0.0,
            lat_coord=0.0,
            coord_frame="icrs",
            telescope_loc=(0, 0, 0),
            time_array=2456789.0,
            antenna_positions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            telescope_frame="mcmf",
            ellipsoid=selenoid,
        )
    except SpiceUNKNOWNFRAME as err:
        pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))

    # Check that the total lengths all match 1
    np.testing.assert_allclose(
        (gen_results["uvw"] ** 2.0).sum(1), 2.0, rtol=0, atol=1e-3
    )

    if selenoid == "SPHERE":
        # check defaults
        try:
            gen_results = utils.uvw_track_generator(
                lon_coord=0.0,
                lat_coord=0.0,
                coord_frame="icrs",
                telescope_loc=(0, 0, 0),
                time_array=2456789.0,
                antenna_positions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                telescope_frame="mcmf",
            )
        except SpiceUNKNOWNFRAME as err:
            pytest.skip("SpiceUNKNOWNFRAME error: " + str(err))

        # Check that the total lengths all match 1
        np.testing.assert_allclose(
            (gen_results["uvw"] ** 2.0).sum(1), 2.0, rtol=0, atol=1e-3
        )


def test_pole_calc_pa():
    # Check a couple of positions up
    pos_angle = utils.phasing.calc_frame_pos_angle(
        time_array=np.array([2456789.0, 2456789.0]),
        app_ra=np.array([np.pi / 4, np.pi]),
        app_dec=np.radians(np.array([-89.7, 89.8])),
        telescope_loc=(0, 30, 0),
        ref_frame="icrs",
        offset_pos=(np.pi / 180) / (3600),
    )

    # Test against directly calculated values
    np.testing.assert_allclose(
        pos_angle, [0.2564204225333429, 0.00920308652161622], atol=utils.RADIAN_TOL
    )
