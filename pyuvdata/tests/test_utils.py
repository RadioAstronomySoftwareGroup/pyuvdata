# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for common utility functions."""
import os
import copy
import re

import pytest
import numpy as np
from astropy import units
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle, EarthLocation

from pyuvdata import UVData, UVFlag, UVCal
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH


ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values",
    "ignore:antenna_positions is not set. Using known values",
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values for HERA.",
    "ignore:antenna_positions is not set. Using known values for HERA.",
)


@pytest.fixture(scope="session")
def astrometry_args():
    default_args = {
        "time_array": 2456789.0 + np.array([0.0, 1.25, 10.5, 100.75]),
        "icrs_ra": 2.468,
        "icrs_dec": 1.234,
        "epoch": 2000.0,
        "telescope_loc": (0.123, -0.456, 4321.0),
        "pm_ra": 12.3,
        "pm_dec": 45.6,
        "vrad": 31.4,
        "dist": 73.31,
        "library": "erfa",
    }

    default_args["lst_array"] = uvutils.get_lst_for_time(
        default_args["time_array"],
        default_args["telescope_loc"][0] * (180.0 / np.pi),
        default_args["telescope_loc"][1] * (180.0 / np.pi),
        default_args["telescope_loc"][2],
    )

    default_args["drift_coord"] = SkyCoord(
        default_args["lst_array"],
        [default_args["telescope_loc"][0]] * len(default_args["lst_array"]),
        unit="rad",
    )

    default_args["icrs_coord"] = SkyCoord(
        default_args["icrs_ra"], default_args["icrs_dec"], unit="rad",
    )

    default_args["fk5_ra"], default_args["fk5_dec"] = uvutils.transform_sidereal_coords(
        default_args["icrs_ra"],
        default_args["icrs_dec"],
        "icrs",
        "fk5",
        in_coord_epoch="J2000.0",
        out_coord_epoch="J2000.0",
    )

    # These are values calculated w/o the optional arguments, e.g. pm, vrad, dist
    default_args["app_ra"], default_args["app_dec"] = uvutils.transform_icrs_to_app(
        default_args["time_array"],
        default_args["icrs_ra"],
        default_args["icrs_dec"],
        default_args["telescope_loc"],
    )

    default_args["app_coord"] = SkyCoord(
        default_args["app_ra"], default_args["app_dec"], unit="rad",
    )

    yield default_args


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


@pytest.fixture
def uvcalibrate_init_data():
    uvdata = UVData()
    uvdata.read(
        os.path.join(DATA_PATH, "zen.2458098.45361.HH.uvh5_downselected"),
        file_type="uvh5",
    )
    uvcal = UVCal()
    uvcal.read_calfits(
        os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni.calfits_downselected")
    )

    yield uvdata, uvcal


@pytest.fixture
def uvcalibrate_data(uvcalibrate_init_data):
    uvdata, uvcal = uvcalibrate_init_data

    # fix the antenna names in the uvcal object to match the uvdata object
    uvcal.antenna_names = np.array(
        [name.replace("ant", "HH") for name in uvcal.antenna_names]
    )

    yield uvdata, uvcal


def test_XYZ_from_LatLonAlt():
    """Test conversion from lat/lon/alt to ECEF xyz with reference values."""
    out_xyz = uvutils.XYZ_from_LatLonAlt(
        ref_latlonalt[0], ref_latlonalt[1], ref_latlonalt[2]
    )
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    assert np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3)

    # test error checking
    with pytest.raises(
        ValueError,
        match="latitude, longitude and altitude must all have the same length",
    ):
        uvutils.XYZ_from_LatLonAlt(
            ref_latlonalt[0],
            ref_latlonalt[1],
            np.array([ref_latlonalt[2], ref_latlonalt[2]]),
        )

    with pytest.raises(
        ValueError,
        match="latitude, longitude and altitude must all have the same length",
    ):
        uvutils.XYZ_from_LatLonAlt(
            ref_latlonalt[0],
            np.array([ref_latlonalt[1], ref_latlonalt[1]]),
            ref_latlonalt[2],
        )


def test_LatLonAlt_from_XYZ():
    """Test conversion from ECEF xyz to lat/lon/alt with reference values."""
    out_latlonalt = uvutils.LatLonAlt_from_XYZ(ref_xyz)
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    assert np.allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3)
    pytest.raises(ValueError, uvutils.LatLonAlt_from_XYZ, ref_latlonalt)

    # test passing multiple values
    xyz_mult = np.stack((np.array(ref_xyz), np.array(ref_xyz)))
    lat_vec, lon_vec, alt_vec = uvutils.LatLonAlt_from_XYZ(xyz_mult)
    assert np.allclose(
        ref_latlonalt, (lat_vec[1], lon_vec[1], alt_vec[1]), rtol=0, atol=1e-3
    )
    # check error if array transposed
    with pytest.raises(ValueError) as cm:
        uvutils.LatLonAlt_from_XYZ(xyz_mult.T)
    assert str(cm.value).startswith(
        "The expected shape of ECEF xyz array is (Npts, 3)."
    )

    # check error if only 2 coordinates
    with pytest.raises(ValueError) as cm:
        uvutils.LatLonAlt_from_XYZ(xyz_mult[:, 0:2])
    assert str(cm.value).startswith(
        "The expected shape of ECEF xyz array is (Npts, 3)."
    )

    # test error checking
    pytest.raises(ValueError, uvutils.LatLonAlt_from_XYZ, ref_xyz[0:1])


def test_lla_xyz_lla_roundtrip():
    """Test roundtripping an array will yield the same values."""
    np.random.seed(0)
    lats = -30.721 + np.random.normal(0, 0.0005, size=30)
    lons = 21.428 + np.random.normal(0, 0.0005, size=30)
    alts = np.random.uniform(1051, 1054, size=30)
    lats *= np.pi / 180.0
    lons *= np.pi / 180.0
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    lats_new, lons_new, alts_new = uvutils.LatLonAlt_from_XYZ(xyz)
    assert np.allclose(lats_new, lats)
    assert np.allclose(lons_new, lons)
    assert np.allclose(alts_new, alts)


@pytest.fixture(scope="module")
def enu_ecef_info():
    """Some setup info for ENU/ECEF calculations."""
    center_lat = -30.7215261207 * np.pi / 180.0
    center_lon = 21.4283038269 * np.pi / 180.0
    center_alt = 1051.7
    # fmt: off
    lats = (np.array([-30.72218216, -30.72138101, -30.7212785, -30.7210011,
                     -30.72159853, -30.72206199, -30.72174614, -30.72188775,
                     -30.72183915, -30.72100138])
            * np.pi / 180.0)
    lons = (np.array([21.42728211, 21.42811727, 21.42814544, 21.42795736,
                     21.42686739, 21.42918772, 21.42785662, 21.4286408,
                     21.42750933, 21.42896567])
            * np.pi / 180.0)
    alts = np.array([1052.25, 1051.35, 1051.2, 1051., 1051.45, 1052.04, 1051.68,
                     1051.87, 1051.77, 1051.06])
    # used pymap3d, which implements matlab code, as a reference.
    x = [5109327.46674067, 5109339.76407785, 5109344.06370947,
         5109365.11297147, 5109372.115673, 5109266.94314734,
         5109329.89620962, 5109295.13656657, 5109337.21810468,
         5109329.85680612]

    y = [2005130.57953031, 2005221.35184577, 2005225.93775268,
         2005214.8436201, 2005105.42364036, 2005302.93158317,
         2005190.65566222, 2005257.71335575, 2005157.78980089,
         2005304.7729239]

    z = [-3239991.24516348, -3239914.4185286, -3239904.57048431,
         -3239878.02656316, -3239935.20415493, -3239979.68381865,
         -3239949.39266985, -3239962.98805772, -3239958.30386264,
         -3239878.08403833]

    east = [-97.87631659, -17.87126443, -15.17316938, -33.19049252, -137.60520964,
            84.67346748, -42.84049408, 32.28083937, -76.1094745, 63.40285935]
    north = [-72.7437482, 16.09066646, 27.45724573, 58.21544651, -8.02964511,
             -59.41961437, -24.39698388, -40.09891961, -34.70965816, 58.18410876]
    up = [0.54883333, -0.35004539, -0.50007736, -0.70035299, -0.25148791, 0.33916067,
          -0.02019057, 0.16979185, 0.06945155, -0.64058124]
    # fmt: on
    yield (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    )


def test_xyz_from_latlonalt(enu_ecef_info):
    """Test calculating xyz from lat lot alt."""
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    assert np.allclose(np.stack((x, y, z), axis=1), xyz, atol=1e-3)


def test_enu_from_ecef(enu_ecef_info):
    """Test calculating ENU from ECEF coordinates."""
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)

    enu = uvutils.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)
    assert np.allclose(np.stack((east, north, up), axis=1), enu, atol=1e-3)


@pytest.mark.parametrize("shape_type", ["transpose", "Nblts,2", "Nblts,1"])
def test_enu_from_ecef_shape_errors(enu_ecef_info, shape_type):
    """Test ENU_from_ECEF input shape errors."""
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    if shape_type == "transpose":
        xyz = xyz.T.copy()
    elif shape_type == "Nblts,2":
        xyz = xyz.copy()[:, 0:2]
    elif shape_type == "Nblts,1":
        xyz = xyz.copy()[:, 0:1]

    # check error if array transposed
    with pytest.raises(ValueError) as cm:
        uvutils.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)
    assert str(cm.value).startswith(
        "The expected shape of ECEF xyz array is (Npts, 3)."
    )


def test_enu_from_ecef_magnitude_error(enu_ecef_info):
    """Test ENU_from_ECEF input magnitude errors."""
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    # error checking
    with pytest.raises(ValueError) as cm:
        uvutils.ENU_from_ECEF(xyz / 2.0, center_lat, center_lon, center_alt)
    assert str(cm.value).startswith(
        "ECEF vector magnitudes must be on the order of the radius of the earth"
    )


def test_ecef_from_enu_roundtrip(enu_ecef_info):
    """Test ECEF_from_ENU values."""
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    enu = uvutils.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)
    # check that a round trip gives the original value.
    xyz_from_enu = uvutils.ECEF_from_ENU(enu, center_lat, center_lon, center_alt)
    assert np.allclose(xyz, xyz_from_enu, atol=1e-3)


@pytest.mark.parametrize("shape_type", ["transpose", "Nblts,2", "Nblts,1"])
def test_ecef_from_enu_shape_errors(enu_ecef_info, shape_type):
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    enu = uvutils.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)
    if shape_type == "transpose":
        enu = enu.copy().T
    elif shape_type == "Nblts,2":
        enu = enu.copy()[:, 0:2]
    elif shape_type == "Nblts,1":
        enu = enu.copy()[:, 0:1]

    # check error if array transposed
    with pytest.raises(ValueError) as cm:
        uvutils.ECEF_from_ENU(enu, center_lat, center_lon, center_alt)
    assert str(cm.value).startswith("The expected shape of the ENU array is (Npts, 3).")


def test_ecef_from_enu_single(enu_ecef_info):
    """Test single coordinate transform."""
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    # check passing a single value
    enu_single = uvutils.ENU_from_ECEF(xyz[0, :], center_lat, center_lon, center_alt)

    assert np.allclose(np.array((east[0], north[0], up[0])), enu_single, atol=1e-3)


def test_ecef_from_enu_single_roundtrip(enu_ecef_info):
    """Test single coordinate roundtrip."""
    (
        center_lat,
        center_lon,
        center_alt,
        lats,
        lons,
        alts,
        x,
        y,
        z,
        east,
        north,
        up,
    ) = enu_ecef_info
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    # check passing a single value
    enu = uvutils.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)

    enu_single = uvutils.ENU_from_ECEF(xyz[0, :], center_lat, center_lon, center_alt)
    assert np.allclose(np.array((east[0], north[0], up[0])), enu[0, :], atol=1e-3)

    xyz_from_enu = uvutils.ECEF_from_ENU(enu_single, center_lat, center_lon, center_alt)
    assert np.allclose(xyz[0, :], xyz_from_enu, atol=1e-3)


def test_mwa_ecef_conversion():
    """
    Test based on comparing the antenna locations in a Cotter uvfits file to
    the antenna locations in MWA_tools.
    """

    test_data_file = os.path.join(DATA_PATH, "mwa128_ant_layouts.npz")
    f = np.load(test_data_file)

    # From the STABXYZ table in a cotter-generated uvfits file, obsid = 1066666832
    xyz = f["stabxyz"]
    # From the East/North/Height columns in a cotter-generated metafits file,
    # obsid = 1066666832
    enh = f["ENH"]
    # From a text file antenna_locations.txt in MWA_Tools/scripts
    txt_topo = f["txt_topo"]

    # From the unphased uvw coordinates of obsid 1066666832, positions relative
    # to antenna 0
    # these aren't used in the current test, but are interesting and might help
    # with phasing diagnosis in the future
    uvw_topo = f["uvw_topo"]
    # Sky coordinates are flipped for uvw derived values
    uvw_topo = -uvw_topo
    uvw_topo += txt_topo[0]

    # transpose these arrays to get them into the right shape
    txt_topo = txt_topo.T
    uvw_topo = uvw_topo.T

    # ARRAYX, ARRAYY, ARRAYZ in ECEF frame from Cotter file
    arrcent = f["arrcent"]
    lat, lon, alt = uvutils.LatLonAlt_from_XYZ(arrcent)

    # The STABXYZ coordinates are defined with X through the local meridian,
    # so rotate back to the prime meridian
    new_xyz = uvutils.ECEF_from_rotECEF(xyz.T, lon)
    # add in array center to get real ECEF
    ecef_xyz = new_xyz + arrcent

    enu = uvutils.ENU_from_ECEF(ecef_xyz, lat, lon, alt)

    assert np.allclose(enu, enh)

    # test other direction of ECEF rotation
    rot_xyz = uvutils.rotECEF_from_ECEF(new_xyz, lon)
    assert np.allclose(rot_xyz.T, xyz)


@pytest.mark.parametrize(
    "input1,input2,msg",
    (
        [0.0, np.array([0.0]), "lon_array and lat_array must either both be floats or"],
        [np.array([0.0, 1.0]), np.array([0.0]), "lon_array and lat_array must have "],
    ),
)
def test_polar2_to_cart3_arg_errs(input1, input2, msg):
    """
    Test that bad arguments to polar2_to_cart3 throw appropriate errors.
    """
    with pytest.raises(ValueError) as cm:
        uvutils.polar2_to_cart3(input1, input2)
    assert str(cm.value).startswith(msg)


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
    with pytest.raises(ValueError) as cm:
        uvutils.cart3_to_polar2(input1)
    assert str(cm.value).startswith(msg)


@pytest.mark.parametrize(
    "input1,input2,input3,msg",
    (
        [np.zeros((1, 3, 1)), np.zeros((1, 3, 3)), 2, "rot_matrix must be of shape "],
        [np.zeros((1, 2, 1)), np.zeros((1, 3, 3)), 1, "Misshaped xyz_array - expected"],
        [np.zeros((2, 1)), np.zeros((1, 3, 3)), 1, "Misshaped xyz_array - expected"],
        [np.zeros((2)), np.zeros((1, 3, 3)), 1, "Misshaped xyz_array - expected shape"],
    ),
)
def test_rotate_matmul_wrapper_arg_errs(input1, input2, input3, msg):
    """
    Test that bad arguments to _rotate_matmul_wrapper throw appropriate errors.
    """
    with pytest.raises(ValueError) as cm:
        uvutils._rotate_matmul_wrapper(input1, input2, input3)
    assert str(cm.value).startswith(msg)


def test_cart_to_polar_roundtrip():
    """
    Test that polar->cart coord transformation is the inverse of cart->polar.
    """
    # Basic round trip with vectors
    assert uvutils.cart3_to_polar2(uvutils.polar2_to_cart3(0.0, 0.0)) == (0.0, 0.0)


def test_rotate_one_axis(vector_list):
    """
    Tests some basic vector rotation operations with a single axis rotation.
    """
    # These tests are used to verify the basic functionality of the primary
    # functions used to perform rotations
    x_vecs, y_vecs, z_vecs, test_vecs = vector_list

    # Test no-ops w/ 0 deg rotations
    assert np.all(uvutils._rotate_one_axis(x_vecs, 0.0, 0) == x_vecs)
    assert np.all(
        uvutils._rotate_one_axis(x_vecs[:, 0], 0.0, 1)
        == x_vecs[np.newaxis, :, 0, np.newaxis],
    )
    assert np.all(
        uvutils._rotate_one_axis(x_vecs[:, :, np.newaxis], 0.0, 2,)
        == x_vecs[:, :, np.newaxis],
    )

    # Test no-ops w/ None
    assert np.all(uvutils._rotate_one_axis(test_vecs, None, 1) == test_vecs)
    assert np.all(
        uvutils._rotate_one_axis(test_vecs[:, 0], None, 2)
        == test_vecs[np.newaxis, :, 0, np.newaxis]
    )
    assert np.all(
        uvutils._rotate_one_axis(test_vecs[:, :, np.newaxis], None, 0,)
        == test_vecs[:, :, np.newaxis]
    )

    # Test some basic equivalencies to make sure rotations are working correctly
    assert np.allclose(x_vecs, uvutils._rotate_one_axis(x_vecs, 1.0, 0))
    assert np.allclose(y_vecs, uvutils._rotate_one_axis(y_vecs, 2.0, 1))
    assert np.allclose(z_vecs, uvutils._rotate_one_axis(z_vecs, 3.0, 2))

    assert np.allclose(x_vecs, uvutils._rotate_one_axis(y_vecs, -np.pi / 2.0, 2))
    assert np.allclose(y_vecs, uvutils._rotate_one_axis(x_vecs, np.pi / 2.0, 2))
    assert np.allclose(x_vecs, uvutils._rotate_one_axis(z_vecs, np.pi / 2.0, 1))
    assert np.allclose(z_vecs, uvutils._rotate_one_axis(x_vecs, -np.pi / 2.0, 1))
    assert np.allclose(y_vecs, uvutils._rotate_one_axis(z_vecs, -np.pi / 2.0, 0))
    assert np.allclose(z_vecs, uvutils._rotate_one_axis(y_vecs, np.pi / 2.0, 0))

    assert np.all(
        np.equal(
            uvutils._rotate_one_axis(test_vecs, 1.0, 2),
            uvutils._rotate_one_axis(test_vecs, 1.0, np.array([2])),
        )
    )

    # Testing a special case, where the xyz_array vectors are reshaped if there
    # is only a single rotation matrix used (helps speed things up significantly)
    mod_vec = x_vecs.T.reshape((2, 3, 1))
    assert np.all(uvutils._rotate_one_axis(mod_vec, 1.0, 0) == mod_vec)


def test_rotate_two_axis(vector_list):
    """
    Tests some basic vector rotation operations with a double axis rotation.
    """
    x_vecs, y_vecs, z_vecs, test_vecs = vector_list

    # These tests are used to verify the basic functionality of the primary
    # functions used to two-axis rotations
    assert np.allclose(x_vecs, uvutils._rotate_two_axis(x_vecs, 2 * np.pi, 1.0, 1, 0))
    assert np.allclose(y_vecs, uvutils._rotate_two_axis(y_vecs, 2 * np.pi, 2.0, 2, 1))
    assert np.allclose(z_vecs, uvutils._rotate_two_axis(z_vecs, 2 * np.pi, 3.0, 0, 2))

    # Do one more test, which verifies that we can filp our (1,1,1) test vector to
    # the postiion at (-1, -1 , -1)
    mod_vec = test_vecs.T.reshape((2, 3, 1))
    assert np.allclose(
        uvutils._rotate_two_axis(mod_vec, np.pi, np.pi / 2.0, 0, 1), -mod_vec
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
            uvutils._rotate_one_axis(test_vecs, rot1, axis1),
            uvutils._rotate_two_axis(test_vecs, rot2, rot3, axis2, axis3),
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
            (ValueError, "Must include lst_array if moving between ENU (i.e.,"),
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
    for key in arg_dict.keys():
        calc_uvw_args[key] = arg_dict[key]

    with pytest.raises(err[0]) as cm:
        uvutils.calc_uvw(
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
    assert str(cm.value).startswith(err[1])


def test_calc_uvw_no_op(calc_uvw_args):
    """
    Test that transfroming ENU -> ENU gives you an output identical to the input.
    """
    # This should be a no-op, check for equality
    uvw_check = uvutils.calc_uvw(
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
    uvw_ant_check = uvutils.calc_uvw(
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

    uvw_base_check = uvutils.calc_uvw(
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

    assert np.allclose(uvw_ant_check, calc_uvw_args["uvw_array"])
    assert np.allclose(uvw_base_check, calc_uvw_args["uvw_array"])


@pytest.mark.parametrize("to_enu", [False, True])
def test_calc_uvw_base_vs_ants(calc_uvw_args, to_enu):
    """
    Check to see that we get the same values for uvw coordinates whether we calculate
    them using antenna positions or the previously calculated uvw's.
    """

    # Now change position, and make sure that whether we used ant positions of rotated
    # uvw vectors, we derived the same uvw-coordinates at the end
    uvw_ant_check = uvutils.calc_uvw(
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

    uvw_base_check = uvutils.calc_uvw(
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

    assert np.allclose(uvw_ant_check, uvw_base_check)


def test_calc_uvw_enu_roundtrip(calc_uvw_args):
    """
    Check and see that we can go from uvw to ENU and back to uvw using the `uvw_array`
    argument alone (i.e., without antenna positions).
    """
    # Now attempt to round trip from projected to ENU back to projected -- that should
    # give us the original set of uvw-coordinates.
    temp_uvw = uvutils.calc_uvw(
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

    uvw_base_enu_check = uvutils.calc_uvw(
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

    assert np.allclose(calc_uvw_args["uvw_array"], uvw_base_enu_check)


def test_calc_uvw_pa_ex_post_facto(calc_uvw_args):
    """
    Check and see that one can apply the frame position angle rotation after-the-fact
    and still get out the same answer you get if you were doing it during the initial
    uvw coordinate calculation.
    """
    # Finally, check and see what happens if you do the PA rotation as part of the
    # first uvw calcuation, and make sure it agrees with what you get if you decide
    # to apply the PA rotation after-the-fact.
    uvw_base_check = uvutils.calc_uvw(
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

    temp_uvw = uvutils.calc_uvw(
        app_ra=calc_uvw_args["app_ra"],
        app_dec=calc_uvw_args["app_dec"],
        lst_array=calc_uvw_args["lst_array"],
        use_ant_pos=False,
        uvw_array=calc_uvw_args["uvw_array"],
        old_app_ra=calc_uvw_args["old_app_ra"],
        old_app_dec=calc_uvw_args["old_app_dec"],
        old_frame_pa=calc_uvw_args["old_frame_pa"],
    )

    uvw_base_late_pa_check = uvutils.calc_uvw(
        frame_pa=calc_uvw_args["frame_pa"],
        use_ant_pos=False,
        uvw_array=temp_uvw,
        old_frame_pa=calc_uvw_args["old_frame_pa"],
    )

    assert np.allclose(uvw_base_check, uvw_base_late_pa_check)


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
    pytest.importorskip("novas")
    default_args = astrometry_args.copy()
    for key in arg_dict.keys():
        default_args[key] = arg_dict[key]

    # Start w/ the transform_icrs_to_app block
    with pytest.raises(ValueError) as cm:
        uvutils.transform_icrs_to_app(
            default_args["time_array"],
            default_args["icrs_ra"],
            default_args["icrs_dec"],
            default_args["telescope_loc"],
            pm_ra=default_args["pm_ra"],
            pm_dec=default_args["pm_dec"],
            dist=default_args["dist"],
            vrad=default_args["vrad"],
            epoch=default_args["epoch"],
            astrometry_library=default_args["library"],
        )
    assert str(cm.value).startswith(msg)


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
    for key in arg_dict.keys():
        default_args[key] = arg_dict[key]

    with pytest.raises(ValueError) as cm:
        uvutils.transform_app_to_icrs(
            default_args["time_array"],
            default_args["app_ra"],
            default_args["app_dec"],
            default_args["telescope_loc"],
            astrometry_library=default_args["library"],
        )
    assert str(cm.value).startswith(msg)


def test_transform_sidereal_coords_arg_errs():
    """
    Check for argument errors with transform_sidereal_coords
    """
    # Next on to sidereal to sidereal
    with pytest.raises(ValueError) as cm:
        uvutils.transform_sidereal_coords(
            [0.0],
            [0.0, 1.0],
            "fk5",
            "icrs",
            in_coord_epoch="J2000.0",
            time_array=[0.0, 1.0, 2.0],
        )
    assert str(cm.value).startswith("lon and lat must be the same shape.")

    with pytest.raises(ValueError) as cm:
        uvutils.transform_sidereal_coords(
            [0.0, 1.0],
            [0.0, 1.0],
            "fk4",
            "fk4",
            in_coord_epoch=1950.0,
            out_coord_epoch=1984.0,
            time_array=[0.0, 1.0, 2.0],
        )
    assert str(cm.value).startswith("Shape of time_array must be either that of ")


@pytest.mark.filterwarnings('ignore:ERFA function "d2dtf" yielded')
@pytest.mark.parametrize(
    "arg_dict,msg",
    (
        [
            {"force_lookup": True, "time_array": np.arange(100000)},
            "Requesting too many individual ephem points from JPL-Horizons.",
        ],
        [{"force_lookup": False, "high_cadence": True}, "Too many ephem points"],
        [{"time_array": np.arange(10)}, "No current support for JPL ephems outside"],
        [{"targ_name": "whoami"}, "Target ID is not recognized in either the small"],
    ),
)
def test_lookup_jplhorizons_arg_errs(arg_dict, msg):
    """
    Check for argument errors with lookup_jplhorizons.
    """
    # Don't do this test if we don't have astroquery loaded
    pytest.importorskip("astroquery")
    default_args = {
        "targ_name": "Mars",
        "time_array": np.array([0.0, 1000.0]) + 2456789.0,
        "telescope_loc": EarthLocation.from_geodetic(0, 0, height=0.0),
        "high_cadence": False,
        "force_lookup": None,
    }

    for key in arg_dict.keys():
        default_args[key] = arg_dict[key]

    with pytest.raises(ValueError) as cm:
        uvutils.lookup_jplhorizons(
            default_args["targ_name"],
            default_args["time_array"],
            telescope_loc=default_args["telescope_loc"],
            high_cadence=default_args["high_cadence"],
            force_indv_lookup=default_args["force_lookup"],
        )
    assert str(cm.value).startswith(msg)


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
    with pytest.raises(ValueError) as cm:
        uvutils.interpolate_ephem(
            0.0,
            0.0 if ("etimes" == bad_arg) else [0.0, 1.0],
            0.0 if ("ra" == bad_arg) else [0.0, 1.0],
            0.0 if ("dec" == bad_arg) else [0.0, 1.0],
            ephem_dist=0.0 if ("dist" == bad_arg) else [0.0, 1.0],
            ephem_vel=0.0 if ("vel" == bad_arg) else [0.0, 1.0],
        )
    assert str(cm.value).startswith(msg)


def test_calc_app_coords_arg_errs():
    """
    Check for argument errors with calc_app_coords
    """
    # Now on to app_coords
    with pytest.raises(ValueError) as cm:
        uvutils.calc_app_coords(
            0.0, 0.0, telescope_loc=(0, 1, 2), coord_type="whoknows"
        )
    assert str(cm.value).startswith("Object type whoknows is not recognized.")


def test_transform_multi_sidereal_coords(astrometry_args):
    """
    Perform some basic tests to verify that we can transform between sidereal frames
    with multiple coordinates.
    """
    # Check and make sure that we can deal with non-singleton times or coords with
    # singleton coords and times, respectively.
    check_ra, check_dec = uvutils.transform_sidereal_coords(
        astrometry_args["icrs_ra"] * np.ones(2),
        astrometry_args["icrs_dec"] * np.ones(2),
        "icrs",
        "fk5",
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
    fk5_ra, fk5_dec = uvutils.transform_sidereal_coords(
        astrometry_args["icrs_ra"],
        astrometry_args["icrs_dec"],
        "icrs",
        "fk5",
        in_coord_epoch=2000.0,
        out_coord_epoch=2000.0,
        time_array=astrometry_args["time_array"][0],
    )

    fk4_ra, fk4_dec = uvutils.transform_sidereal_coords(
        fk5_ra,
        fk5_dec,
        "fk5",
        "fk4",
        in_coord_epoch="J2000.0",
        out_coord_epoch="B1950.0",
    )

    check_ra, check_dec = uvutils.transform_sidereal_coords(
        fk4_ra,
        fk4_dec,
        "fk4",
        "icrs",
        in_coord_epoch="B1950.0",
        out_coord_epoch="J2000.0",
    )

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(check_coord.separation(astrometry_args["icrs_coord"]).uarcsec < 0.1)


def test_roundtrip_icrs(astrometry_args):
    """
    Performs a roundtrip test to verify that one can transform between
    ICRS <-> topocentric to the precision limit, without running into
    issues.
    """
    in_lib_list = ["erfa", "erfa", "astropy", "astropy"]
    out_lib_list = ["erfa", "astropy", "erfa", "astropy"]

    for in_lib, out_lib in zip(in_lib_list, out_lib_list):
        app_ra, app_dec = uvutils.transform_icrs_to_app(
            astrometry_args["time_array"],
            astrometry_args["icrs_ra"],
            astrometry_args["icrs_dec"],
            astrometry_args["telescope_loc"],
            epoch=astrometry_args["epoch"],
            astrometry_library=in_lib,
        )

        check_ra, check_dec = uvutils.transform_app_to_icrs(
            astrometry_args["time_array"],
            app_ra,
            app_dec,
            astrometry_args["telescope_loc"],
            astrometry_library=out_lib,
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


def test_calc_parallactic_angle():
    """
    A relatively straightforward test to verify that we recover the parallactic
    angles we expect given some known inputs
    """
    expected_vals = np.array([1.0754290375762232, 0.0, -0.6518070715011698])
    meas_vals = uvutils.calc_parallactic_angle(
        [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [2.0, 1.0, 0], 1.0,
    )
    # Make sure things agree to better than ~0.1 uas (as it definitely should)
    assert np.allclose(expected_vals, meas_vals, 0.0, 1e-12)


def test_calc_frame_pos_angle():
    """
    Verify that we recover frame position angles correctly
    """
    # First test -- plug in "topo" for the frame, which should always produce an
    # array of all zeros (the topo frame is what the apparent coords are in)
    frame_pa = uvutils.calc_frame_pos_angle(
        np.array([2456789.0] * 100),
        np.arange(100) * (np.pi / 50),
        np.zeros(100),
        (0, 0, 0),
        "topo",
    )
    assert len(frame_pa) == 100
    assert np.all(frame_pa == 0.0)
    # PA of zero degrees (they're always aligned)
    # Next test -- plug in J2000 and see that we actually get back a frame PA
    # of basically 0 degrees.
    j2000_jd = Time(2000.0, format="jyear").utc.jd
    frame_pa = uvutils.calc_frame_pos_angle(
        np.array([j2000_jd] * 100),
        np.arange(100) * (np.pi / 50),
        np.zeros(100),
        (0, 0, 0),
        "fk5",
        ref_epoch=2000.0,
    )
    # At J2000, the only frame PA terms come from aberation, which basically max out
    # at ~< 1e-4 rad. Check to make sure that lines up with what we measure.
    assert np.all(np.abs(frame_pa) < 1e-4)

    # JD 2458849.5 is Jan-01-2020, so 20 years of parallax ought to have accumulated
    # (with about 1 arcmin/yr of precession). Make sure these values are sensible
    frame_pa = uvutils.calc_frame_pos_angle(
        np.array([2458849.5] * 100),
        np.arange(100) * (np.pi / 50),
        np.zeros(100),
        (0, 0, 0),
        "fk5",
        ref_epoch=2000.0,
    )
    assert np.all(np.abs(frame_pa) < 20 * (50.3 / 3600) * (np.pi / 180.0))
    # Check the PA at a couple of chosen points, which just so happen to be very close
    # in magnitude (as they're basically in the same plane as the motion of the Earth)
    assert np.isclose(frame_pa[25], 0.001909957544309159)
    assert np.isclose(frame_pa[-25], -0.0019098101664715339)


def test_jphl_lookup():
    """
    A very simple lookup query to verify that the astroquery tools for accessing
    JPL-Horizons are working. This test is very limited, on account of not wanting to
    slam JPL w/ coordinate requests.
    """
    pytest.importorskip("astroquery")

    [
        ephem_times,
        ephem_ra,
        ephem_dec,
        ephem_dist,
        ephem_vel,
    ] = uvutils.lookup_jplhorizons("Sun", 2456789.0)

    assert np.all(np.equal(ephem_times, 2456789.0))
    assert np.allclose(ephem_ra, 0.8393066751804976)
    assert np.allclose(ephem_dec, 0.3120687480116649)
    assert np.allclose(ephem_dist, 1.00996185750717)
    assert np.allclose(ephem_vel, 0.386914)


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

    ra_vals0, dec_vals0, dist_vals0, vel_vals0 = uvutils.interpolate_ephem(
        time_array,
        ephem_times,
        ephem_ra,
        ephem_dec,
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

    ra_vals1, dec_vals1, dist_vals1, vel_vals1 = uvutils.interpolate_ephem(
        time_array,
        ephem_times,
        ephem_ra,
        ephem_dec,
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

    ra_vals2, dec_vals2, dist_vals2, vel_vals2 = uvutils.interpolate_ephem(
        time_array,
        ephem_times,
        ephem_ra,
        ephem_dec,
        ephem_dist=ephem_dist,
        ephem_vel=ephem_vel,
    )

    # Make sure that everything is consistent to floating point precision
    assert np.allclose(ra_vals1, ra_vals2, 1e-15, 0.0)
    assert np.allclose(dec_vals1, dec_vals2, 1e-15, 0.0)
    assert np.allclose(dist_vals1, dist_vals2, 1e-15, 0.0)
    assert np.allclose(vel_vals1, vel_vals2, 1e-15, 0.0)
    assert np.allclose(time_array + 1.0, ra_vals2, 1e-15, 0.0)
    assert np.allclose(time_array + 2.0, dec_vals2, 1e-15, 0.0)
    assert np.allclose(time_array + 3.0, dist_vals2, 1e-15, 0.0)
    assert np.allclose(time_array + 4.0, vel_vals2, 1e-15, 0.0)


@pytest.mark.parametrize("frame", ["icrs", "fk5"])
def test_calc_app_sidereal(astrometry_args, frame):
    """
    Tests that we can calculate app coords for sidereal objects
    """
    # First step is to check and make sure we can do sidereal coords. This is the most
    # basic thing to check, so this really _should work.
    check_ra, check_dec = uvutils.calc_app_coords(
        astrometry_args["fk5_ra"] if (frame == "fk5") else astrometry_args["icrs_ra"],
        astrometry_args["fk5_dec"] if (frame == "fk5") else astrometry_args["icrs_dec"],
        coord_type="sidereal",
        telescope_loc=astrometry_args["telescope_loc"],
        time_array=astrometry_args["time_array"],
        coord_frame=frame,
        coord_epoch=astrometry_args["epoch"],
    )
    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(astrometry_args["app_coord"].separation(check_coord).uarcsec < 1.0)


@pytest.mark.parametrize("frame", ["icrs", "fk5"])
def test_calc_app_ephem(astrometry_args, frame):
    """
    Tests that we can calculate app coords for ephem objects
    """
    # Next, see what happens when we pass an ephem. Note that this is just a single
    # point ephem, so its not testing any of the fancy interpolation, but we have other
    # tests for poking at that. The two tests here are to check bot the ICRS and FK5
    # paths through the ephem.
    if frame == "fk5":
        ephem_ra = astrometry_args["fk5_ra"]
        ephem_dec = astrometry_args["fk5_dec"]
    else:
        ephem_ra = np.array([astrometry_args["icrs_ra"]])
        ephem_dec = np.array([astrometry_args["icrs_dec"]])

    ephem_times = np.array([astrometry_args["time_array"][0]])
    check_ra, check_dec = uvutils.calc_app_coords(
        ephem_ra,
        ephem_dec,
        coord_times=ephem_times,
        coord_type="ephem",
        telescope_loc=astrometry_args["telescope_loc"],
        time_array=astrometry_args["time_array"],
        coord_epoch=astrometry_args["epoch"],
        coord_frame=frame,
    )
    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(astrometry_args["app_coord"].separation(check_coord).uarcsec < 1.0)


def test_calc_app_driftscan(astrometry_args):
    """
    Tests that we can calculate app coords for driftscan objects
    """
    # Now on to the driftscan, which takes in arguments in terms of az and el (and
    # the values we've given below should also be for zenith)
    check_ra, check_dec = uvutils.calc_app_coords(
        0.0,
        np.pi / 2.0,
        coord_type="driftscan",
        telescope_loc=astrometry_args["telescope_loc"],
        time_array=astrometry_args["time_array"],
    )
    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(astrometry_args["drift_coord"].separation(check_coord).uarcsec < 1.0)


def test_calc_app_unphased(astrometry_args):
    """
    Tests that we can calculate app coords for unphased objects
    """
    # Finally, check unphased, which is forced to point toward zenith (unlike driftscan,
    # which is allowed to point at any az/el position)
    check_ra, check_dec = uvutils.calc_app_coords(
        None,
        None,
        coord_type="unphased",
        telescope_loc=astrometry_args["telescope_loc"],
        time_array=astrometry_args["time_array"],
        lst_array=astrometry_args["lst_array"],
    )
    check_coord = SkyCoord(check_ra, check_dec, unit="rad")

    assert np.all(astrometry_args["drift_coord"].separation(check_coord).uarcsec < 1.0)


def test_calc_app_fk5_roundtrip(astrometry_args):
    # Do a round-trip with the two top-level functions and make sure they agree to
    # better than 1 µas, first in FK5
    app_ra, app_dec = uvutils.calc_app_coords(
        0.0,
        0.0,
        coord_type="sidereal",
        telescope_loc=astrometry_args["telescope_loc"],
        time_array=astrometry_args["time_array"],
        coord_frame="fk5",
        coord_epoch="J2000.0",
    )

    check_ra, check_dec = uvutils.calc_sidereal_coords(
        astrometry_args["time_array"],
        app_ra,
        app_dec,
        astrometry_args["telescope_loc"],
        "fk5",
        coord_epoch=2000.0,
    )
    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(SkyCoord(0, 0, unit="rad").separation(check_coord).uarcsec < 1.0)


def test_calc_app_fk4_roundtrip(astrometry_args):
    # Finally, check and make sure that FK4 performs similarly
    app_ra, app_dec = uvutils.calc_app_coords(
        0.0,
        0.0,
        coord_type="sidereal",
        telescope_loc=astrometry_args["telescope_loc"],
        time_array=astrometry_args["time_array"],
        coord_frame="fk4",
        coord_epoch=1950.0,
    )

    check_ra, check_dec = uvutils.calc_sidereal_coords(
        astrometry_args["time_array"],
        app_ra,
        app_dec,
        astrometry_args["telescope_loc"],
        "fk4",
        coord_epoch=1950.0,
    )

    check_coord = SkyCoord(check_ra, check_dec, unit="rad")
    assert np.all(SkyCoord(0, 0, unit="rad").separation(check_coord).uarcsec < 1.0)


@pytest.mark.filterwarnings('ignore:ERFA function "pmsafe" yielded 4 of')
@pytest.mark.filterwarnings('ignore:ERFA function "utcut1" yielded 2 of')
@pytest.mark.filterwarnings('ignore:ERFA function "d2dtf" yielded 1 of')
def test_astrometry_icrs_to_app(astrometry_args):
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

    for idx, name in enumerate(astrometry_list):
        coord_results[idx] = uvutils.transform_icrs_to_app(
            astrometry_args["time_array"],
            astrometry_args["icrs_ra"],
            astrometry_args["icrs_dec"],
            astrometry_args["telescope_loc"],
            epoch=astrometry_args["epoch"],
            pm_ra=astrometry_args["pm_ra"],
            pm_dec=astrometry_args["pm_dec"],
            vrad=astrometry_args["vrad"],
            dist=astrometry_args["dist"],
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
        coord_results[idx] = uvutils.transform_app_to_icrs(
            astrometry_args["time_array"],
            astrometry_args["icrs_ra"],
            astrometry_args["icrs_dec"],
            astrometry_args["telescope_loc"],
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

    gcrs_ra, gcrs_dec = uvutils.transform_sidereal_coords(
        astrometry_args["icrs_ra"] * np.ones(2),
        astrometry_args["icrs_dec"] * np.ones(2),
        "icrs",
        "gcrs",
        time_array=Time(astrometry_args["time_array"][0], format="jd"),
    )

    check_ra, check_dec = uvutils.transform_sidereal_coords(
        astrometry_args["icrs_ra"] * np.ones(2),
        astrometry_args["icrs_dec"] * np.ones(2),
        "icrs",
        "gcrs",
        time_array=Time(astrometry_args["time_array"][0] * np.ones(2), format="jd"),
    )

    assert np.all(gcrs_ra == check_ra)
    assert np.all(gcrs_dec == check_dec)


def test_transform_icrs_to_app_time_obj(astrometry_args):
    """
    Test that we recover identical values when using a Time objects instead of a floats
    for the various time-related arguments in transform_icrs_to_app.
    """
    check_ra, check_dec = uvutils.transform_icrs_to_app(
        Time(astrometry_args["time_array"], format="jd"),
        astrometry_args["icrs_ra"],
        astrometry_args["icrs_dec"],
        astrometry_args["telescope_loc"],
        epoch=Time(astrometry_args["epoch"], format="jyear"),
    )

    assert np.all(check_ra == astrometry_args["app_ra"])
    assert np.all(check_dec == astrometry_args["app_dec"])


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

    icrs_ra, icrs_dec = uvutils.transform_app_to_icrs(
        astrometry_args["time_array"][0],
        astrometry_args["app_ra"][0],
        astrometry_args["app_dec"][0],
        astrometry_args["telescope_loc"],
    )

    check_ra, check_dec = uvutils.transform_app_to_icrs(
        Time(astrometry_args["time_array"][0], format="jd"),
        astrometry_args["app_ra"][0],
        astrometry_args["app_dec"][0],
        telescope_loc,
    )

    assert np.all(check_ra == icrs_ra)
    assert np.all(check_dec == icrs_dec)


def test_calc_app_coords_objs(astrometry_args):
    """
    Test that we recover identical values when using Time/EarthLocation objects instead
    of floats for time_array and telescope_loc, respectively for calc_app_coords.
    """
    telescope_loc = EarthLocation.from_geodetic(
        astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
        astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
        height=astrometry_args["telescope_loc"][2],
    )

    app_ra, app_dec = uvutils.calc_app_coords(
        astrometry_args["icrs_ra"],
        astrometry_args["icrs_dec"],
        time_array=astrometry_args["time_array"][0],
        telescope_loc=astrometry_args["telescope_loc"],
    )

    check_ra, check_dec = uvutils.calc_app_coords(
        astrometry_args["icrs_ra"],
        astrometry_args["icrs_dec"],
        time_array=Time(astrometry_args["time_array"][0], format="jd"),
        telescope_loc=telescope_loc,
    )

    assert np.all(check_ra == app_ra)
    assert np.all(check_dec == app_dec)


def test_astrometry_lst(astrometry_args):
    """
    Check for consistency beteen astrometry libraries when calculating LAST

    This test evaluates consistency in calculating local apparent sidereal time when
    using the different astrometry libraries available in pyuvdata, namely: astropy,
    pyERFA, and python-novas. Between these three, we expect agreement within 6 µs in
    most instances, although for pyuvdata we tolerate differences of up to ~60 µs
    (which translates to 1 mas in sky position error) since we don't expect to need
    astrometry better than this.
    """
    pytest.importorskip("novas")
    pytest.importorskip("novas_de405")
    astrometry_list = ["erfa", "astropy", "novas"]
    lst_results = [None, None, None, None]
    # These values were indepedently calculated using erfa v1.7.2, which at the
    # time of coding agreed to < 50 µs with astropy v4.2.1 and novas 3.1.1.5. We
    # use those values here as a sort of history check to make sure that something
    # hasn't changed in the underlying astrometry libraries without being caught
    lst_results[3] = np.array(
        [0.8506741803481069, 2.442973468758589, 4.1728965710160555, 1.0130589895999587]
    )

    for idx, name in enumerate(astrometry_list):
        # Note that the units aren't right here (missing a rad-> deg conversion), but
        # the above values were calculated using the arguments below.
        lst_results[idx] = uvutils.get_lst_for_time(
            astrometry_args["time_array"],
            astrometry_args["telescope_loc"][0],
            astrometry_args["telescope_loc"][1],
            astrometry_args["telescope_loc"][2],
            astrometry_library=name,
        )

    for idx in range(len(lst_results) - 1):
        for jdx in range(idx + 1, len(lst_results)):
            alpha_time = lst_results[idx] * units.rad
            beta_time = lst_results[jdx] * units.rad
            assert np.all(np.abs(alpha_time - beta_time).to_value("mas") < 1.0)


def test_lst_for_time_float_vs_array(astrometry_args):
    """
    Test for equality when passing a single float vs an ndarray (of length 1) when
    calling get_lst_for_time.
    """
    lst_array = uvutils.get_lst_for_time(
        np.array(astrometry_args["time_array"][0]),
        astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
        astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
        astrometry_args["telescope_loc"][2],
    )

    check_lst = uvutils.get_lst_for_time(
        astrometry_args["time_array"][0],
        astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
        astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
        astrometry_args["telescope_loc"][2],
    )

    assert np.all(lst_array == check_lst)


def test_phasing_funcs():
    # these tests are based on a notebook where I tested against the mwa_tools
    # phasing code
    ra_hrs = 12.1
    dec_degs = -42.3
    mjd = 55780.1

    array_center_xyz = np.array([-2559454.08, 5095372.14, -2849057.18])
    lat_lon_alt = uvutils.LatLonAlt_from_XYZ(array_center_xyz)

    obs_time = Time(mjd, format="mjd", location=(lat_lon_alt[1], lat_lon_alt[0]))

    icrs_coord = SkyCoord(
        ra=Angle(ra_hrs, unit="hr"), dec=Angle(dec_degs, unit="deg"), obstime=obs_time
    )
    gcrs_coord = icrs_coord.transform_to("gcrs")

    # in east/north/up frame (relative to array center) in meters: (Nants, 3)
    ants_enu = np.array([-101.94, 156.41, 1.24])

    ant_xyz_abs = uvutils.ECEF_from_ENU(
        ants_enu, lat_lon_alt[0], lat_lon_alt[1], lat_lon_alt[2]
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

    gcrs_uvw = uvutils.phase_uvw(gcrs_coord.ra.rad, gcrs_coord.dec.rad, gcrs_rel.value)

    mwa_tools_calcuvw_u = -97.122828
    mwa_tools_calcuvw_v = 50.388281
    mwa_tools_calcuvw_w = -151.27976

    assert np.allclose(gcrs_uvw[0, 0], mwa_tools_calcuvw_u, atol=1e-3)
    assert np.allclose(gcrs_uvw[0, 1], mwa_tools_calcuvw_v, atol=1e-3)
    assert np.allclose(gcrs_uvw[0, 2], mwa_tools_calcuvw_w, atol=1e-3)

    # also test unphasing
    temp2 = uvutils.unphase_uvw(
        gcrs_coord.ra.rad, gcrs_coord.dec.rad, np.squeeze(gcrs_uvw)
    )
    assert np.allclose(gcrs_rel.value, temp2)


def test_pol_funcs():
    """ Test utility functions to convert between polarization strings and numbers """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    pol_str = ["yx", "xy", "yy", "xx", "lr", "rl", "ll", "rr", "pI", "pQ", "pU", "pV"]
    assert pol_nums == uvutils.polstr2num(pol_str)
    assert pol_str == uvutils.polnum2str(pol_nums)
    # Check individuals
    assert -6 == uvutils.polstr2num("YY")
    assert "pV" == uvutils.polnum2str(4)
    # Check errors
    pytest.raises(KeyError, uvutils.polstr2num, "foo")
    pytest.raises(ValueError, uvutils.polstr2num, 1)
    pytest.raises(ValueError, uvutils.polnum2str, 7.3)
    # Check parse
    assert uvutils.parse_polstr("xX") == "xx"
    assert uvutils.parse_polstr("XX") == "xx"
    assert uvutils.parse_polstr("i") == "pI"


def test_pol_funcs_x_orientation():
    """Test functions to convert between pol strings and numbers with x_orientation."""

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]

    x_orient1 = "e"
    pol_str = ["ne", "en", "nn", "ee", "lr", "rl", "ll", "rr", "pI", "pQ", "pU", "pV"]
    assert pol_nums == uvutils.polstr2num(pol_str, x_orientation=x_orient1)
    assert pol_str == uvutils.polnum2str(pol_nums, x_orientation=x_orient1)
    # Check individuals
    assert -6 == uvutils.polstr2num("NN", x_orientation=x_orient1)
    assert "pV" == uvutils.polnum2str(4)
    # Check errors
    pytest.raises(KeyError, uvutils.polstr2num, "foo", x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.polstr2num, 1, x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.polnum2str, 7.3, x_orientation=x_orient1)
    # Check parse
    assert uvutils.parse_polstr("eE", x_orientation=x_orient1) == "ee"
    assert uvutils.parse_polstr("xx", x_orientation=x_orient1) == "ee"
    assert uvutils.parse_polstr("NN", x_orientation=x_orient1) == "nn"
    assert uvutils.parse_polstr("yy", x_orientation=x_orient1) == "nn"
    assert uvutils.parse_polstr("i", x_orientation=x_orient1) == "pI"

    x_orient2 = "n"
    pol_str = ["en", "ne", "ee", "nn", "lr", "rl", "ll", "rr", "pI", "pQ", "pU", "pV"]
    assert pol_nums == uvutils.polstr2num(pol_str, x_orientation=x_orient2)
    assert pol_str == uvutils.polnum2str(pol_nums, x_orientation=x_orient2)
    # Check individuals
    assert -6 == uvutils.polstr2num("EE", x_orientation=x_orient2)
    assert "pV" == uvutils.polnum2str(4)
    # Check errors
    pytest.raises(KeyError, uvutils.polstr2num, "foo", x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.polstr2num, 1, x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.polnum2str, 7.3, x_orientation=x_orient2)
    # Check parse
    assert uvutils.parse_polstr("nN", x_orientation=x_orient2) == "nn"
    assert uvutils.parse_polstr("xx", x_orientation=x_orient2) == "nn"
    assert uvutils.parse_polstr("EE", x_orientation=x_orient2) == "ee"
    assert uvutils.parse_polstr("yy", x_orientation=x_orient2) == "ee"
    assert uvutils.parse_polstr("i", x_orientation=x_orient2) == "pI"

    # check warnings for non-recognized x_orientation
    with uvtest.check_warnings(UserWarning, "x_orientation not recognized"):
        assert uvutils.polstr2num("xx", x_orientation="foo") == -5

    with uvtest.check_warnings(UserWarning, "x_orientation not recognized"):
        assert uvutils.polnum2str(-6, x_orientation="foo") == "yy"


def test_jones_num_funcs():
    """Test functions to convert between jones polarization strings and numbers."""

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    jstr = ["Jyx", "Jxy", "Jyy", "Jxx", "Jlr", "Jrl", "Jll", "Jrr"]
    assert jnums == uvutils.jstr2num(jstr)
    assert jstr, uvutils.jnum2str(jnums)
    # Check shorthands
    jstr = ["yx", "xy", "yy", "y", "xx", "x", "lr", "rl", "ll", "l", "rr", "r"]
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == uvutils.jstr2num(jstr)
    # Check individuals
    assert -6 == uvutils.jstr2num("jyy")
    assert "Jxy" == uvutils.jnum2str(-7)
    # Check errors
    pytest.raises(KeyError, uvutils.jstr2num, "foo")
    pytest.raises(ValueError, uvutils.jstr2num, 1)
    pytest.raises(ValueError, uvutils.jnum2str, 7.3)

    # check parse method
    assert uvutils.parse_jpolstr("x") == "Jxx"
    assert uvutils.parse_jpolstr("xy") == "Jxy"
    assert uvutils.parse_jpolstr("XY") == "Jxy"


def test_jones_num_funcs_x_orientation():
    """Test functions to convert jones pol strings and numbers with x_orientation."""

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    x_orient1 = "east"
    jstr = ["Jne", "Jen", "Jnn", "Jee", "Jlr", "Jrl", "Jll", "Jrr"]
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient1)
    assert jstr == uvutils.jnum2str(jnums, x_orientation=x_orient1)
    # Check shorthands
    jstr = ["ne", "en", "nn", "n", "ee", "e", "lr", "rl", "ll", "l", "rr", "r"]
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient1)
    # Check individuals
    assert -6 == uvutils.jstr2num("jnn", x_orientation=x_orient1)
    assert "Jen" == uvutils.jnum2str(-7, x_orientation=x_orient1)
    # Check errors
    pytest.raises(KeyError, uvutils.jstr2num, "foo", x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.jstr2num, 1, x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.jnum2str, 7.3, x_orientation=x_orient1)

    # check parse method
    assert uvutils.parse_jpolstr("e", x_orientation=x_orient1) == "Jee"
    assert uvutils.parse_jpolstr("x", x_orientation=x_orient1) == "Jee"
    assert uvutils.parse_jpolstr("y", x_orientation=x_orient1) == "Jnn"
    assert uvutils.parse_jpolstr("en", x_orientation=x_orient1) == "Jen"
    assert uvutils.parse_jpolstr("NE", x_orientation=x_orient1) == "Jne"

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    x_orient2 = "north"
    jstr = ["Jen", "Jne", "Jee", "Jnn", "Jlr", "Jrl", "Jll", "Jrr"]
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient2)
    assert jstr == uvutils.jnum2str(jnums, x_orientation=x_orient2)
    # Check shorthands
    jstr = ["en", "ne", "ee", "e", "nn", "n", "lr", "rl", "ll", "l", "rr", "r"]
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient2)
    # Check individuals
    assert -6 == uvutils.jstr2num("jee", x_orientation=x_orient2)
    assert "Jne" == uvutils.jnum2str(-7, x_orientation=x_orient2)
    # Check errors
    pytest.raises(KeyError, uvutils.jstr2num, "foo", x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.jstr2num, 1, x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.jnum2str, 7.3, x_orientation=x_orient2)

    # check parse method
    assert uvutils.parse_jpolstr("e", x_orientation=x_orient2) == "Jee"
    assert uvutils.parse_jpolstr("x", x_orientation=x_orient2) == "Jnn"
    assert uvutils.parse_jpolstr("y", x_orientation=x_orient2) == "Jee"
    assert uvutils.parse_jpolstr("en", x_orientation=x_orient2) == "Jen"
    assert uvutils.parse_jpolstr("NE", x_orientation=x_orient2) == "Jne"

    # check warnings for non-recognized x_orientation
    with uvtest.check_warnings(UserWarning, "x_orientation not recognized"):
        assert uvutils.jstr2num("x", x_orientation="foo") == -5

    with uvtest.check_warnings(UserWarning, "x_orientation not recognized"):
        assert uvutils.jnum2str(-6, x_orientation="foo") == "Jyy"


def test_conj_pol():
    """ Test function to conjugate pols """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    cpol_nums = [-7, -8, -6, -5, -3, -4, -2, -1, 1, 2, 3, 4]
    assert pol_nums == uvutils.conj_pol(cpol_nums)
    assert uvutils.conj_pol(pol_nums) == cpol_nums
    # fmt: off
    pol_str = ['yx', 'xy', 'yy', 'xx', 'ee', 'nn', 'en', 'ne', 'lr', 'rl', 'll',
               'rr', 'pI', 'pQ', 'pU', 'pV']
    cpol_str = ['xy', 'yx', 'yy', 'xx', 'ee', 'nn', 'ne', 'en', 'rl', 'lr', 'll',
                'rr', 'pI', 'pQ', 'pU', 'pV']
    # fmt: on
    assert pol_str == uvutils.conj_pol(cpol_str)
    assert uvutils.conj_pol(pol_str) == cpol_str
    assert [pol_str, pol_nums] == uvutils.conj_pol([cpol_str, cpol_nums])

    # Test error with jones
    cjstr = ["Jxy", "Jyx", "Jyy", "Jxx", "Jrl", "Jlr", "Jll", "Jrr"]
    assert pytest.raises(KeyError, uvutils.conj_pol, cjstr)

    # Test invalid pol
    with pytest.raises(ValueError) as cm:
        uvutils.conj_pol(2.3)
    assert str(cm.value).startswith(
        "Polarization not recognized, cannot be conjugated."
    )


def test_redundancy_finder():
    """
    Check that get_baseline_redundancies and get_antenna_redundancies return consistent
    redundant groups for a test file with the HERA19 layout.
    """
    uvd = UVData()
    uvd.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    uvd.select(times=uvd.time_array[0])
    uvd.unphase_to_drift(use_ant_pos=True)
    # uvw_array is now equivalent to baseline positions
    uvd.conjugate_bls(convention="ant1<ant2", use_enu=True)

    tol = 0.05  # meters

    bl_positions = uvd.uvw_array
    bl_pos_backup = copy.deepcopy(uvd.uvw_array)

    pytest.raises(
        ValueError,
        uvutils.get_baseline_redundancies,
        uvd.baseline_array,
        bl_positions[0:2, 0:1],
    )

    baseline_groups, vec_bin_centers, lens = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol
    )

    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            assert np.allclose(
                np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])), lens[gi], atol=tol
            )

    # Shift the baselines around in a circle. Check that the same baselines are
    # recovered to the corresponding tolerance increase.
    # This moves one baseline at a time by a fixed displacement and checks that
    # the redundant groups are the same.

    hightol = 0.25  # meters. Less than the smallest baseline in the file.
    Nbls = uvd.Nbls
    Nshifts = 5
    shift_angs = np.linspace(0, 2 * np.pi, Nshifts)
    base_shifts = np.stack(
        (
            (hightol - tol) * np.cos(shift_angs),
            (hightol - tol) * np.sin(shift_angs),
            np.zeros(Nshifts),
        )
    ).T
    for sh in base_shifts:
        for bi in range(Nbls):
            # Shift one baseline at a time.
            bl_positions_new = uvd.uvw_array
            bl_positions_new[bi] += sh

            (
                baseline_groups_new,
                vec_bin_centers,
                lens,
            ) = uvutils.get_baseline_redundancies(
                uvd.baseline_array, bl_positions_new, tol=hightol
            )

            for gi, gp in enumerate(baseline_groups_new):
                for bl in gp:
                    bl_ind = np.where(uvd.baseline_array == bl)
                    bl_vec = bl_positions[bl_ind]
                    assert np.allclose(
                        np.sqrt(np.abs(np.dot(bl_vec, vec_bin_centers[gi]))),
                        lens[gi],
                        atol=hightol,
                    )

            # Compare baseline groups:
            a = [tuple(el) for el in baseline_groups]
            b = [tuple(el) for el in baseline_groups_new]
            assert set(a) == set(b)

    tol = 0.05

    antpos, antnums = uvd.get_ENU_antpos()

    baseline_groups_ants, vec_bin_centers, lens = uvutils.get_antenna_redundancies(
        antnums, antpos, tol=tol, include_autos=False
    )
    # Under these conditions, should see 19 redundant groups in the file.
    assert len(baseline_groups_ants) == 19

    # Check with conjugated baseline redundancies returned
    # Ensure at least one baseline has u==0 and v!=0 (for coverage of this case)
    bl_positions[16, 0] = 0
    (
        baseline_groups,
        vec_bin_centers,
        lens,
        conjugates,
    ) = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True
    )

    # restore baseline (16,0) and repeat to get correct groups
    bl_positions = bl_pos_backup
    (
        baseline_groups,
        vec_bin_centers,
        lens,
        conjugates,
    ) = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True
    )

    # Apply flips to compare with get_antenna_redundancies().
    bl_gps_unconj = copy.deepcopy(baseline_groups)
    for gi, gp in enumerate(bl_gps_unconj):
        for bi, bl in enumerate(gp):
            if bl in conjugates:
                bl_gps_unconj[gi][bi] = uvutils.baseline_index_flip(bl, len(antnums))
    bl_gps_unconj = [sorted(bgp) for bgp in bl_gps_unconj]
    bl_gps_ants = [sorted(bgp) for bgp in baseline_groups_ants]
    assert np.all(sorted(bl_gps_ants) == sorted(bl_gps_unconj))
    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            if bl in conjugates:
                bl_vec *= -1
            assert np.isclose(
                np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])), lens[gi], atol=tol
            )


def test_high_tolerance_redundancy_error():
    """
    Confirm that an error is raised if the redundancy tolerance is set too high,
    such that baselines end up in multiple
    """
    uvd = UVData()
    uvd.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    uvd.select(times=uvd.time_array[0])
    uvd.unphase_to_drift(use_ant_pos=True)
    # uvw_array is now equivalent to baseline positions
    uvd.conjugate_bls(convention="ant1<ant2", use_enu=True)
    bl_positions = uvd.uvw_array

    tol = 20.05  # meters

    with pytest.raises(ValueError) as cm:
        (
            baseline_groups,
            vec_bin_centers,
            lens,
            conjugates,
        ) = uvutils.get_baseline_redundancies(
            uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True
        )
    assert "Some baselines are falling into" in str(cm.value)


def test_redundancy_conjugates():
    """
    Check that redundancy finding with conjugation works.

    Check that the correct baselines are flipped.
    """
    Nants = 10
    tol = 0.5
    ant1_arr = np.tile(np.arange(Nants), Nants)
    ant2_arr = np.repeat(np.arange(Nants), Nants)
    Nbls = ant1_arr.size
    bl_inds = uvutils.antnums_to_baseline(ant1_arr, ant2_arr, Nants)

    maxbl = 100.0
    bl_vecs = np.random.uniform(-maxbl, maxbl, (Nbls, 3))
    bl_vecs[0, 0] = 0
    bl_vecs[1, 0:2] = 0

    expected_conjugates = []
    for i, (u, v, w) in enumerate(bl_vecs):
        uneg = u < -tol
        uzer = np.isclose(u, 0.0, atol=tol)
        vneg = v < -tol
        vzer = np.isclose(v, 0.0, atol=tol)
        wneg = w < -tol
        if uneg or (uzer and vneg) or (uzer and vzer and wneg):
            expected_conjugates.append(bl_inds[i])
    bl_gps, vecs, lens, conjugates = uvutils.get_baseline_redundancies(
        bl_inds, bl_vecs, tol=tol, with_conjugates=True
    )

    assert sorted(conjugates) == sorted(expected_conjugates)


def test_redundancy_finder_fully_redundant_array():
    """Test the redundancy finder for a fully redundant array."""
    uvd = UVData()
    uvd.read_uvfits(os.path.join(DATA_PATH, "test_redundant_array.uvfits"))
    uvd.select(times=uvd.time_array[0])

    tol = 1  # meters
    bl_positions = uvd.uvw_array

    (
        baseline_groups,
        vec_bin_centers,
        lens,
        conjugates,
    ) = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True
    )

    # Only 1 set of redundant baselines
    assert len(baseline_groups) == 1
    #  Should return the input baselines
    assert baseline_groups[0].sort() == np.unique(uvd.baseline_array).sort()


@pytest.mark.parametrize("n_blocks", [1, 10])
def test_adjacency_lists(n_blocks):
    """Test the adjacency list method in utils."""
    # n_blocks: in _adj_list, loop over chunks of vectors when computing distances.

    # Make a grid.
    Nx = 5
    Lmax = 50

    xbase = np.linspace(0, Lmax, Nx)
    x, y, z = map(np.ndarray.flatten, np.meshgrid(xbase, xbase, xbase))

    # Make more vectors by shifting by Lmax/Nx/3 in x, y, and z:
    dx = (Lmax / Nx) / 3  # One third of cell size.
    x = np.append(x, x + dx)
    y = np.append(y, y + dx)
    z = np.append(z, z + dx)

    # Construct vectors
    vecs = np.vstack((x, y, z)).T
    Npts = x.size

    # Reorder randomly.
    np.random.shuffle(vecs)

    # Tolerance = half of cell diagonal.
    tol = Lmax / Nx * np.sqrt(2) / 2

    adj = uvutils._adj_list(vecs, tol, n_blocks=n_blocks)

    # Confirm that each adjacency set contains all of the vectors that
    # are within the tolerance distance.
    for vi in range(Npts):
        for vj in range(Npts):
            dist = np.linalg.norm(vecs[vi] - vecs[vj])
            if dist < tol:
                assert vj in adj[vi]
                assert vi in adj[vj]
            else:
                assert vj not in adj[vi]
                assert vi not in adj[vj]

    # The way the grid is set up, every clique should have two elements.
    assert all(len(vi) == 2 for vi in adj)


def test_strict_cliques():
    # Adjacency lists comprising only isolated cliques.
    adj_isol = [
        {0, 1, 2},
        {1, 0, 2},
        {2, 0, 1},
        {3},
        {4},
        {5, 6, 7, 8},
        {5, 6, 7, 8},
        {5, 6, 7, 8},
        {5, 6, 7, 8},
    ]
    adj_isol = [frozenset(st) for st in adj_isol]
    exp_cliques = [[0, 1, 2], [3], [4], [5, 6, 7, 8]]

    res = uvutils._find_cliques(adj_isol, strict=True)
    assert res == exp_cliques

    # Error if two cliques are not isolated
    adj_link = adj_isol
    adj_link[-1] = frozenset({5, 6, 7, 8, 1})

    with pytest.raises(ValueError, match="Non-isolated cliques found in graph."):
        uvutils._find_cliques(adj_link, strict=True),


def test_reorder_conj_pols_non_list():
    pytest.raises(ValueError, uvutils.reorder_conj_pols, 4)


def test_reorder_conj_pols_strings():
    pols = ["xx", "xy", "yx"]
    corder = uvutils.reorder_conj_pols(pols)
    assert np.array_equal(corder, [0, 2, 1])


def test_reorder_conj_pols_ints():
    pols = [-5, -7, -8]  # 'xx', 'xy', 'yx'
    corder = uvutils.reorder_conj_pols(pols)
    assert np.array_equal(corder, [0, 2, 1])


def test_reorder_conj_pols_missing_conj():
    pols = ["xx", "xy"]  # Missing 'yx'
    pytest.raises(ValueError, uvutils.reorder_conj_pols, pols)


def test_collapse_mean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out = uvutils.collapse(data, "mean", axis=0)
    out1 = uvutils.mean_collapse(data, axis=0)
    # Actual values are tested in test_mean_no_weights
    assert np.array_equal(out, out1)


def test_collapse_mean_returned_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out, wo = uvutils.collapse(data, "mean", axis=0, return_weights=True)
    out1, wo1 = uvutils.mean_collapse(data, axis=0, return_weights=True)
    # Actual values are tested in test_mean_no_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_mean_returned_with_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo = uvutils.collapse(data, "mean", weights=w, axis=0, return_weights=True)
    out1, wo1 = uvutils.mean_collapse(data, weights=w, axis=0, return_weights=True)
    # Actual values are tested in test_mean_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_mean_returned_with_weights_and_weights_square():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo, wso = uvutils.collapse(
        data, "mean", weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    out1, wo1, wso1 = uvutils.mean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    # Actual values are tested in test_mean_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)
    assert np.array_equal(wso, wso1)


def test_collapse_mean_returned_with_weights_square_no_return_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wso = uvutils.collapse(
        data,
        "mean",
        weights=w,
        axis=0,
        return_weights=False,
        return_weights_square=True,
    )
    out1, wso1 = uvutils.mean_collapse(
        data, weights=w, axis=0, return_weights=False, return_weights_square=True
    )
    # Actual values are tested in test_mean_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wso, wso1)


def test_collapse_absmean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = (-1) ** i * np.ones_like(data[:, i])
    out = uvutils.collapse(data, "absmean", axis=0)
    out1 = uvutils.absmean_collapse(data, axis=0)
    # Actual values are tested in test_absmean_no_weights
    assert np.array_equal(out, out1)


def test_collapse_quadmean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out = uvutils.collapse(data, "quadmean", axis=0)
    out1 = uvutils.quadmean_collapse(data, axis=0)
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)


def test_collapse_quadmean_returned_with_weights_and_weights_square():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo, wso = uvutils.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=True,
        return_weights_square=True,
    )
    out1, wo1, wso1 = uvutils.quadmean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)
    assert np.array_equal(wso, wso1)


def test_collapse_quadmean_returned_with_weights_square_no_return_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wso = uvutils.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=False,
        return_weights_square=True,
    )
    out1, wso1 = uvutils.quadmean_collapse(
        data, weights=w, axis=0, return_weights=False, return_weights_square=True
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wso, wso1)


def test_collapse_quadmean_returned_without_weights_square_with_return_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo = uvutils.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=True,
        return_weights_square=False,
    )
    out1, wo1 = uvutils.quadmean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=False
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_quadmean_returned_with_weights_square_without_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo = uvutils.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=False,
        return_weights_square=True,
    )
    out1, wo1 = uvutils.quadmean_collapse(
        data, weights=w, axis=0, return_weights=False, return_weights_square=True
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_or_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, 8] = True
    o = uvutils.collapse(data, "or", axis=0)
    o1 = uvutils.or_collapse(data, axis=0)
    assert np.array_equal(o, o1)


def test_collapse_and_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, :] = True
    o = uvutils.collapse(data, "and", axis=0)
    o1 = uvutils.and_collapse(data, axis=0)
    assert np.array_equal(o, o1)


def test_collapse_error():
    pytest.raises(ValueError, uvutils.collapse, np.ones((2, 3)), "fooboo")


def test_mean_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out, wo = uvutils.mean_collapse(data, axis=0, return_weights=True)
    assert np.array_equal(out, np.arange(data.shape[1]))
    assert np.array_equal(wo, data.shape[0] * np.ones(data.shape[1]))
    out, wo = uvutils.mean_collapse(data, axis=1, return_weights=True)
    assert np.all(out == np.mean(np.arange(data.shape[1])))
    assert len(out) == data.shape[0]
    assert np.array_equal(wo, data.shape[1] * np.ones(data.shape[0]))
    out, wo = uvutils.mean_collapse(data, return_weights=True)
    assert out == np.mean(np.arange(data.shape[1]))
    assert wo == data.size
    out = uvutils.mean_collapse(data)
    assert out == np.mean(np.arange(data.shape[1]))


def test_mean_weights_and_weights_square():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo, wso = uvutils.mean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    assert np.allclose(out * wo, data.shape[0])
    assert np.allclose(wo, float(data.shape[0]) / (np.arange(data.shape[1]) + 1))
    assert np.allclose(wso, float(data.shape[0]) / (np.arange(data.shape[1]) + 1) ** 2)
    out, wo, wso = uvutils.mean_collapse(
        data, weights=w, axis=1, return_weights=True, return_weights_square=True
    )
    assert np.allclose(out * wo, data.shape[1])
    assert np.allclose(wo, np.sum(1.0 / (np.arange(data.shape[1]) + 1)))
    assert np.allclose(wso, np.sum(1.0 / (np.arange(data.shape[1]) + 1) ** 2))

    # Zero weights
    w = np.ones_like(w)
    w[0, :] = 0
    w[:, 0] = 0
    out, wo = uvutils.mean_collapse(data, weights=w, axis=0, return_weights=True)
    ans = np.arange(data.shape[1]).astype(np.float64) + 1
    ans[0] = np.inf
    assert np.array_equal(out, ans)
    ans = (data.shape[0] - 1) * np.ones(data.shape[1])
    ans[0] = 0
    assert np.all(wo == ans)
    out, wo = uvutils.mean_collapse(data, weights=w, axis=1, return_weights=True)
    ans = np.mean(np.arange(data.shape[1])[1:] + 1) * np.ones(data.shape[0])
    ans[0] = np.inf
    assert np.all(out == ans)
    ans = (data.shape[1] - 1) * np.ones(data.shape[0])
    ans[0] = 0
    assert np.all(wo == ans)


def test_mean_infs():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    data[:, 0] = np.inf
    data[0, :] = np.inf
    out, wo = uvutils.mean_collapse(data, axis=0, return_weights=True)
    ans = np.arange(data.shape[1]).astype(np.float64)
    ans[0] = np.inf
    assert np.array_equal(out, ans)
    ans = (data.shape[0] - 1) * np.ones(data.shape[1])
    ans[0] = 0
    assert np.all(wo == ans)
    out, wo = uvutils.mean_collapse(data, axis=1, return_weights=True)
    ans = np.mean(np.arange(data.shape[1])[1:]) * np.ones(data.shape[0])
    ans[0] = np.inf
    assert np.all(out == ans)
    ans = (data.shape[1] - 1) * np.ones(data.shape[0])
    ans[0] = 0
    assert np.all(wo == ans)


def test_absmean():
    # Fake data
    data1 = np.zeros((50, 25))
    for i in range(data1.shape[1]):
        data1[:, i] = (-1) ** i * np.ones_like(data1[:, i])
    data2 = np.ones_like(data1)
    out1 = uvutils.absmean_collapse(data1)
    out2 = uvutils.absmean_collapse(data2)
    assert out1 == out2


def test_quadmean():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    o1, w1 = uvutils.quadmean_collapse(data, return_weights=True)
    o2, w2 = uvutils.mean_collapse(np.abs(data) ** 2, return_weights=True)
    o3 = uvutils.quadmean_collapse(data)  # without return_weights
    o2 = np.sqrt(o2)
    assert o1 == o2
    assert w1 == w2
    assert o1 == o3


def test_or_collapse():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, 8] = True
    o = uvutils.or_collapse(data, axis=0)
    ans = np.zeros(25, np.bool_)
    ans[8] = True
    assert np.array_equal(o, ans)
    o = uvutils.or_collapse(data, axis=1)
    ans = np.zeros(50, np.bool_)
    ans[0] = True
    assert np.array_equal(o, ans)
    o = uvutils.or_collapse(data)
    assert o


def test_or_collapse_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, 8] = True
    w = np.ones_like(data, np.float64)
    o, wo = uvutils.or_collapse(data, axis=0, weights=w, return_weights=True)
    ans = np.zeros(25, np.bool_)
    ans[8] = True
    assert np.array_equal(o, ans)
    assert np.array_equal(wo, np.ones_like(o, dtype=np.float64))
    w[0, 8] = 0.3
    with uvtest.check_warnings(UserWarning, "Currently weights are"):
        o = uvutils.or_collapse(data, axis=0, weights=w)
    assert np.array_equal(o, ans)


def test_or_collapse_errors():
    data = np.zeros(5)
    pytest.raises(ValueError, uvutils.or_collapse, data)


def test_and_collapse():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, :] = True
    o = uvutils.and_collapse(data, axis=0)
    ans = np.zeros(25, np.bool_)
    assert np.array_equal(o, ans)
    o = uvutils.and_collapse(data, axis=1)
    ans = np.zeros(50, np.bool_)
    ans[0] = True
    assert np.array_equal(o, ans)
    o = uvutils.and_collapse(data)
    assert not o


def test_and_collapse_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, :] = True
    w = np.ones_like(data, np.float64)
    o, wo = uvutils.and_collapse(data, axis=0, weights=w, return_weights=True)
    ans = np.zeros(25, np.bool_)
    assert np.array_equal(o, ans)
    assert np.array_equal(wo, np.ones_like(o, dtype=np.float64))
    w[0, 8] = 0.3
    with uvtest.check_warnings(UserWarning, "Currently weights are"):
        o = uvutils.and_collapse(data, axis=0, weights=w)
    assert np.array_equal(o, ans)


def test_and_collapse_errors():
    data = np.zeros(5)
    pytest.raises(ValueError, uvutils.and_collapse, data)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_uvcalibrate_apply_gains_oldfiles():
    # read data
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.uvh5"))
    # give it an x_orientation
    uvd.x_orientation = "east"
    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits"))
    # downselect to match each other in shape (but not in actual values!)
    uvd.select(frequencies=uvd.freq_array[0, :10])
    uvc.select(times=uvc.time_array[:3])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All antenna names with data on UVData are missing "
            "on UVCal. To continue with calibration "
            "(and flag all the data), set ant_check=False."
        ),
    ):
        uvutils.uvcalibrate(uvd, uvc, prop_flags=True, ant_check=True, inplace=False)

    ants_expected = [
        "The uvw_array does not match the expected values",
        "All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
    ]
    missing_times = [2457698.4036761867, 2457698.4038004624]

    time_expected = f"Time {missing_times[0]} exists on UVData but not on UVCal."

    freq_expected = (
        f"Frequency {uvd.freq_array[0, 0]} exists on UVData but not on UVCal."
    )

    with uvtest.check_warnings(UserWarning, match=ants_expected):
        with pytest.raises(ValueError, match=time_expected):
            uvutils.uvcalibrate(
                uvd, uvc, prop_flags=True, ant_check=False, inplace=False
            )

    uvc.select(times=uvc.time_array[0])

    with uvtest.check_warnings(UserWarning, match=ants_expected):
        with pytest.raises(ValueError, match=freq_expected):
            uvutils.uvcalibrate(
                uvd, uvc, prop_flags=True, ant_check=False, time_check=False,
            )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_uvcalibrate_delay_oldfiles():
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.uvh5"))

    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits"))
    # downselect to match
    uvc.select(times=uvc.time_array[3], frequencies=uvd.freq_array[0, :])
    uvc.gain_convention = "multiply"
    ant_expected = [
        "The uvw_array does not match the expected values",
        "All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
        r"UVData object does not have `x_orientation` specified but UVCal does",
    ]
    with uvtest.check_warnings(UserWarning, match=ant_expected):
        uvdcal = uvutils.uvcalibrate(
            uvd, uvc, prop_flags=False, ant_check=False, time_check=False, inplace=False
        )

    uvc.convert_to_gain()
    with uvtest.check_warnings(UserWarning, match=ant_expected):
        uvdcal2 = uvutils.uvcalibrate(
            uvd, uvc, prop_flags=False, ant_check=False, time_check=False, inplace=False
        )

    assert uvdcal == uvdcal2


@pytest.mark.parametrize("future_shapes", [True, False])
@pytest.mark.parametrize("flip_gain_conj", [False, True])
@pytest.mark.parametrize("gain_convention", ["divide", "multiply"])
def test_uvcalibrate(uvcalibrate_data, future_shapes, flip_gain_conj, gain_convention):
    uvd, uvc = uvcalibrate_data

    if future_shapes:
        uvd.use_future_array_shapes()

    uvc.gain_convention = gain_convention

    if gain_convention == "divide":
        assert uvc.gain_scale is None
    else:
        # set the gain_scale to "Jy" to test that vis units are set properly
        uvc.gain_scale = "Jy"

    uvdcal = uvutils.uvcalibrate(uvd, uvc, inplace=False, flip_gain_conj=flip_gain_conj)
    if gain_convention == "divide":
        assert uvdcal.vis_units == "uncalib"
    else:
        assert uvdcal.vis_units == "Jy"

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    if flip_gain_conj:
        gain_product = (uvc.get_gains(ant1).conj() * uvc.get_gains(ant2)).T
    else:
        gain_product = (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T

    if gain_convention == "divide":
        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key), uvd.get_data(key) / gain_product,
        )
    else:
        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key), uvd.get_data(key) * gain_product,
        )

    # test undo
    uvdcal = uvutils.uvcalibrate(
        uvdcal,
        uvc,
        prop_flags=True,
        ant_check=False,
        inplace=False,
        undo=True,
        flip_gain_conj=flip_gain_conj,
    )

    np.testing.assert_array_almost_equal(uvd.get_data(key), uvdcal.get_data(key))
    assert uvdcal.vis_units == "uncalib"


@pytest.mark.filterwarnings("ignore:Combined frequencies are not contiguous.")
def test_uvcalibrate_dterm_handling(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # test d-term exception
    with pytest.raises(
        ValueError, match="Cannot apply D-term calibration without -7 or -8"
    ):
        uvutils.uvcalibrate(uvd, uvc, Dterm_cal=True)

    # d-term not implemented error
    uvcDterm = copy.deepcopy(uvc)
    uvcDterm.jones_array = np.array([-7, -8])
    uvcDterm = uvc + uvcDterm
    with pytest.raises(
        NotImplementedError, match="D-term calibration is not yet implemented."
    ):
        uvutils.uvcalibrate(uvd, uvcDterm, Dterm_cal=True)


@pytest.mark.filterwarnings("ignore:Cannot preserve total_quality_array")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_uvcalibrate_flag_propagation(uvcalibrate_data, future_shapes):
    uvd, uvc = uvcalibrate_data

    if future_shapes:
        uvd.use_future_array_shapes()

    # test flag propagation
    uvc.flag_array[0] = True
    uvc.gain_array[1] = 0.0
    uvdcal = uvutils.uvcalibrate(
        uvd, uvc, prop_flags=True, ant_check=False, inplace=False
    )

    assert np.all(uvdcal.get_flags(1, 13, "xx"))  # assert completely flagged
    assert np.all(uvdcal.get_flags(0, 12, "xx"))  # assert completely flagged
    np.testing.assert_array_almost_equal(
        uvd.get_data(1, 13, "xx"), uvdcal.get_data(1, 13, "xx")
    )
    np.testing.assert_array_almost_equal(
        uvd.get_data(0, 12, "xx"), uvdcal.get_data(0, 12, "xx")
    )

    uvd_ant_dict = dict(zip(uvd.antenna_numbers, uvd.antenna_names))
    uvc_ant_dict = dict(zip(uvc.antenna_numbers, uvc.antenna_names))

    for key in uvc_ant_dict.keys():
        if key in uvd_ant_dict.keys():
            uvc_ant_dict[key] = str(uvd_ant_dict[key])
        else:
            uvc_ant_dict[key] = str(uvd_ant_dict[key])

    uvc.antenna_names = np.array(
        [uvc_ant_dict[ant_number] for ant_number in uvc.antenna_numbers]
    )

    uvc_sub = uvc.select(antenna_nums=[1, 12], inplace=False)

    uvdata_unique_nums = np.unique(np.append(uvd.ant_1_array, uvd.ant_2_array))
    uvd.antenna_names = np.array(uvd.antenna_names)
    missing_ants = uvdata_unique_nums.tolist()
    missing_ants.remove(1)
    missing_ants.remove(12)
    missing_ant_names = [
        uvd.antenna_names[np.where(uvd.antenna_numbers == antnum)[0][0]]
        for antnum in missing_ants
    ]

    exp_err = (
        f"Antennas {missing_ant_names} have data on UVData but "
        "are missing on UVCal. To continue calibration and "
        "flag the data from missing antennas, set ant_check=False."
    )

    with pytest.raises(ValueError) as errinfo:
        uvdcal = uvutils.uvcalibrate(
            uvd, uvc_sub, prop_flags=True, ant_check=True, inplace=False
        )

    assert exp_err == str(errinfo.value)

    with pytest.warns(UserWarning) as warninfo:
        uvdcal = uvutils.uvcalibrate(
            uvd, uvc_sub, prop_flags=True, ant_check=False, inplace=False
        )
    warns = {warn.message.args[0] for warn in warninfo}
    ant_expected = {
        f"Antennas {missing_ant_names} have data on UVData but are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed and the data for these antennas will be flagged."
    }

    assert warns == ant_expected
    assert np.all(uvdcal.get_flags(13, 24, "xx"))  # assert completely flagged


@pytest.mark.filterwarnings("ignore:Cannot preserve total_quality_array")
def test_uvcalibrate_flag_propagation_name_mismatch(uvcalibrate_init_data):
    uvd, uvc = uvcalibrate_init_data

    # test flag propagation
    uvc.flag_array[0] = True
    uvc.gain_array[1] = 0.0
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All antenna names with data on UVData are missing "
            "on UVCal. To continue with calibration "
            "(and flag all the data), set ant_check=False."
        ),
    ):
        uvdcal = uvutils.uvcalibrate(
            uvd, uvc, prop_flags=True, ant_check=True, inplace=False
        )

    with uvtest.check_warnings(
        UserWarning,
        match="All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
    ):
        uvdcal = uvutils.uvcalibrate(
            uvd, uvc, prop_flags=True, ant_check=False, inplace=False
        )

    assert np.all(uvdcal.get_flags(1, 13, "xx"))  # assert completely flagged
    assert np.all(uvdcal.get_flags(0, 12, "xx"))  # assert completely flagged
    np.testing.assert_array_almost_equal(
        uvd.get_data(1, 13, "xx"), uvdcal.get_data(1, 13, "xx")
    )
    np.testing.assert_array_almost_equal(
        uvd.get_data(0, 12, "xx"), uvdcal.get_data(0, 12, "xx")
    )


def test_uvcalibrate_extra_cal_antennas(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # remove some antennas from the data
    uvd.select(antenna_nums=[0, 1, 12, 13])

    uvdcal = uvutils.uvcalibrate(uvd, uvc, inplace=False)

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    np.testing.assert_array_almost_equal(
        uvdcal.get_data(key),
        uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
    )


@pytest.mark.parametrize("future_shapes", [True, False])
def test_uvcalibrate_antenna_names_mismatch(uvcalibrate_init_data, future_shapes):
    uvd, uvc = uvcalibrate_init_data

    if future_shapes:
        uvd.use_future_array_shapes()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All antenna names with data on UVData are missing "
            "on UVCal. To continue with calibration "
            "(and flag all the data), set ant_check=False."
        ),
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)

    # now test that they're all flagged if ant_check is False
    with uvtest.check_warnings(
        UserWarning,
        match="All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
    ):
        uvdcal = uvutils.uvcalibrate(uvd, uvc, ant_check=False, inplace=False)

    assert np.all(uvdcal.flag_array)  # assert completely flagged


def test_uvcalibrate_time_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # change times to get warnings
    uvc.time_array = uvc.time_array + 1

    expected_err = {
        f"Time {this_time} exists on UVData but not on UVCal."
        for this_time in np.unique(uvd.time_array)
    }

    with pytest.raises(ValueError) as errinfo:
        uvutils.uvcalibrate(uvd, uvc, inplace=False)
    assert str(errinfo.value) in expected_err


def test_uvcalibrate_time_wrong_size(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # downselect by one time to get error
    uvc.select(times=uvc.time_array[1:])
    with pytest.raises(
        ValueError,
        match="The uvcal object has more than one time but fewer than the "
        "number of unique times on the uvdata object.",
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)


@pytest.mark.parametrize("len_time_range", [0, 1])
def test_uvcalibrate_time_types(uvcalibrate_data, len_time_range):
    uvd, uvc = uvcalibrate_data

    # only one time
    uvc.select(times=uvc.time_array[0])
    if len_time_range == 0:
        uvc.time_range = None
    else:
        # check cal runs fine with a good time range
        uvdcal = uvutils.uvcalibrate(uvd, uvc, inplace=False)

        key = (1, 13, "xx")
        ant1 = (1, "Jxx")
        ant2 = (13, "Jxx")

        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key),
            uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
        )

        # then change time_range to get warnings
        uvc.time_range = np.array(uvc.time_range) + 1

    with pytest.raises(
        ValueError,
        match=(
            "Times do not match between UVData and UVCal. "
            "Set time_check=False to apply calibration anyway."
        ),
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)

    # set time_check=False to test the user warning
    with uvtest.check_warnings(
        UserWarning,
        match=(
            "Times do not match between UVData and UVCal "
            "but time_check is False, so calibration "
            "will be applied anyway."
        ),
    ):
        uvdcal = uvutils.uvcalibrate(uvd, uvc, inplace=False, time_check=False)

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    np.testing.assert_array_almost_equal(
        uvdcal.get_data(key),
        uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
    )


@pytest.mark.filterwarnings("ignore:Combined frequencies are not contiguous.")
def test_uvcalibrate_extra_cal_times(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc2 = copy.deepcopy(uvc)
    uvc2.time_array = uvc.time_array + 1
    uvc_use = uvc + uvc2

    uvdcal = uvutils.uvcalibrate(uvd, uvc_use, inplace=False)

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    np.testing.assert_array_almost_equal(
        uvdcal.get_data(key),
        uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
    )


def test_uvcalibrate_freq_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # change some frequencies to get warnings
    maxf = np.max(uvc.freq_array)
    uvc.freq_array[0, uvc.Nfreqs // 2 :] = uvc.freq_array[0, uvc.Nfreqs // 2 :] + maxf
    expected_err = {
        f"Frequency {this_freq} exists on UVData but not on UVCal."
        for this_freq in uvd.freq_array[0, uvd.Nfreqs // 2 :]
    }
    with pytest.raises(ValueError) as errinfo:
        uvutils.uvcalibrate(uvd, uvc, inplace=False)
    assert str(errinfo.value) in expected_err


@pytest.mark.filterwarnings("ignore:Combined frequencies are not evenly spaced.")
def test_uvcalibrate_extra_cal_freqs(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc2 = copy.deepcopy(uvc)
    uvc2.freq_array = uvc.freq_array + np.max(uvc.freq_array)
    uvc_use = uvc + uvc2

    uvdcal = uvutils.uvcalibrate(uvd, uvc_use, inplace=False)

    key = (1, 13, "xx")
    ant1 = (1, "Jxx")
    ant2 = (13, "Jxx")

    np.testing.assert_array_almost_equal(
        uvdcal.get_data(key),
        uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
    )


def test_uvcalibrate_feedpol_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # downselect the feed polarization to get warnings
    uvc.select(jones=uvutils.jstr2num("Jnn", x_orientation=uvc.x_orientation))
    with pytest.raises(
        ValueError, match=("Feed polarization e exists on UVData but not on UVCal."),
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)


def test_uvcalibrate_x_orientation_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # next check None uvd_x
    uvd.x_orientation = None
    uvc.x_orientation = "east"
    with pytest.warns(
        UserWarning,
        match=r"UVData object does not have `x_orientation` specified but UVCal does",
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("future_shapes", [True, False])
def test_apply_uvflag(future_shapes):
    # load data and insert some flags
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.uvh5"))
    uvd.flag_array[uvd.antpair2ind(9, 20)] = True

    if future_shapes:
        uvd.use_future_array_shapes()

    # load a UVFlag into flag type
    uvf = UVFlag(uvd)
    uvf.to_flag()

    # insert flags for 2 out of 3 times
    uvf.flag_array[uvf.antpair2ind(9, 10)[:2]] = True

    # apply flags and check for basic flag propagation
    uvdf = uvutils.apply_uvflag(uvd, uvf, inplace=False)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])

    # test inplace
    uvdf = copy.deepcopy(uvd)
    uvutils.apply_uvflag(uvdf, uvf, inplace=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])

    # test flag missing
    uvf2 = uvf.select(bls=uvf.get_antpairs()[:-1], inplace=False)
    uvdf = uvutils.apply_uvflag(uvd, uvf2, inplace=False, flag_missing=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(uvf.get_antpairs()[-1])])
    uvdf = uvutils.apply_uvflag(uvd, uvf2, inplace=False, flag_missing=False)
    assert not np.any(uvdf.flag_array[uvdf.antpair2ind(uvf.get_antpairs()[-1])])

    # test force polarization
    uvdf = copy.deepcopy(uvd)
    uvdf2 = copy.deepcopy(uvd)
    uvdf2.polarization_array[0] = -6
    uvdf += uvdf2
    uvdf = uvutils.apply_uvflag(uvdf, uvf, inplace=False, force_pol=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    with pytest.raises(ValueError) as cm:
        uvutils.apply_uvflag(uvdf, uvf, inplace=False, force_pol=False)
    assert "Input uvf and uvd polarizations do not match" in str(cm.value)

    # test unflag first
    uvdf = uvutils.apply_uvflag(uvd, uvf, inplace=False, unflag_first=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    assert not np.any(uvdf.flag_array[uvdf.antpair2ind(9, 20)])

    # convert uvf to waterfall and test
    uvfw = copy.deepcopy(uvf)
    uvfw.to_waterfall(method="or")
    uvdf = uvutils.apply_uvflag(uvd, uvfw, inplace=False)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 20)][:2])
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(20, 22)][:2])

    # test mode exception
    uvfm = copy.deepcopy(uvf)
    uvfm.mode = "metric"
    with pytest.raises(ValueError) as cm:
        uvutils.apply_uvflag(uvd, uvfm)
    assert "UVFlag must be flag mode" in str(cm.value)

    # test polarization exception
    uvd2 = copy.deepcopy(uvd)
    uvd2.polarization_array[0] = -6
    uvf2 = UVFlag(uvd)
    uvf2.to_flag()
    uvd2.polarization_array[0] = -8
    with pytest.raises(ValueError) as cm:
        uvutils.apply_uvflag(uvd2, uvf2, force_pol=False)
    assert "Input uvf and uvd polarizations do not match" in str(cm.value)

    # test time and frequency mismatch exceptions
    uvf2 = uvf.select(frequencies=uvf.freq_array[:, :2], inplace=False)
    with pytest.raises(ValueError) as cm:
        uvutils.apply_uvflag(uvd, uvf2)
    assert "UVFlag and UVData have mismatched frequency arrays" in str(cm.value)

    uvf2 = copy.deepcopy(uvf)
    uvf2.freq_array += 1.0
    with pytest.raises(ValueError) as cm:
        uvutils.apply_uvflag(uvd, uvf2)
    assert "UVFlag and UVData have mismatched frequency arrays" in str(cm.value)

    uvf2 = uvf.select(times=np.unique(uvf.time_array)[:2], inplace=False)
    with pytest.raises(ValueError) as cm:
        uvutils.apply_uvflag(uvd, uvf2)
    assert "UVFlag and UVData have mismatched time arrays" in str(cm.value)

    uvf2 = copy.deepcopy(uvf)
    uvf2.time_array += 1.0
    with pytest.raises(ValueError) as cm:
        uvutils.apply_uvflag(uvd, uvf2)
    assert "UVFlag and UVData have mismatched time arrays" in str(cm.value)

    # assert implicit broadcasting works
    uvf2 = uvf.select(frequencies=uvf.freq_array[:, :1], inplace=False)
    uvd2 = uvutils.apply_uvflag(uvd, uvf2, inplace=False)
    assert np.all(uvd2.get_flags(9, 10)[:2])
    uvf2 = uvf.select(times=np.unique(uvf.time_array)[:1], inplace=False)
    uvd2 = uvutils.apply_uvflag(uvd, uvf2, inplace=False)
    assert np.all(uvd2.get_flags(9, 10))


def test_upos_tol_reds():
    # Checks that the u-positive convention in get_antenna_redundancies
    # is enforced to the specificed tolerance.

    # Make a layout with two NS baselines, one with u ~ -2*eps, and another with u == 0
    # This would previously cause one to be flipped, when they should be redundant.

    eps = 1e-5
    tol = 3 * eps

    ant_pos = np.array(
        [[-eps, 1.0, 0.0], [1.0, 1.0, 0.0], [eps, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    ant_nums = np.arange(4)

    red_grps, _, _ = uvutils.get_antenna_redundancies(ant_nums, ant_pos, tol=tol)

    assert len(red_grps) == 4


class FakeClass:
    def __init__(self):
        pass


def test_parse_ants_error():
    test_obj = FakeClass()
    with pytest.raises(
        ValueError,
        match=(
            "UVBased objects must have all the following attributes in order "
            "to call 'parse_ants': "
        ),
    ):
        uvutils.parse_ants(test_obj, ant_str="")


@pytest.mark.parametrize(
    "filename1,filename2,answer",
    [
        (["foo.uvh5"], ["bar.uvh5"], ["foo.uvh5", "bar.uvh5"]),
        (["foo.uvh5", "bar.uvh5"], ["foo.uvh5"], ["foo.uvh5", "bar.uvh5"]),
        (["foo.uvh5"], None, ["foo.uvh5"]),
        (None, ["bar.uvh5"], ["bar.uvh5"]),
        (None, None, None),
    ],
)
def test_combine_filenames(filename1, filename2, answer):
    combined_filenames = uvutils._combine_filenames(filename1, filename2)
    if answer is None:
        assert combined_filenames is answer
    else:
        # use sets to test equality so that order doesn't matter
        assert set(combined_filenames) == set(answer)

    return


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_slicing():
    """Test HDF5 slicing helper functions"""
    # check trivial slice representations
    slices, _ = uvutils._convert_to_slices([])
    assert slices == [slice(0, 0, 0)]
    slices, _ = uvutils._convert_to_slices(10)
    assert slices == [slice(10, 11, 1)]

    # dataset shape checking
    # check various kinds of indexing give the right answer
    indices = [slice(0, 10), 0, [0, 1, 2], [0]]
    dset = np.empty((100, 1, 1024, 2), dtype=np.float64)
    shape, _ = uvutils._get_dset_shape(dset, indices)
    assert tuple(shape) == (10, 1, 3, 1)

    # dataset indexing
    # check various kinds of indexing give the right answer
    slices = [uvutils._convert_to_slices(ind)[0] for ind in indices]
    slices[1] = 0
    data = uvutils._index_dset(dset, slices)
    assert data.shape == tuple(shape)
