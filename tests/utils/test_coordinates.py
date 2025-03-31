# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for coordinate utility functions."""

import os
import re

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import EarthLocation

from pyuvdata import utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

selenoids = ["SPHERE", "GSFC", "GRAIL23", "CE-1-LAM-GEO"]

try:
    from lunarsky import MoonLocation

    frame_selenoid = [["itrs", None]]
    for snd in selenoids:
        frame_selenoid.append(["mcmf", snd])
except ImportError:
    frame_selenoid = [["itrs", None]]


# Earth
ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)

# Moon
ref_latlonalt_moon = (0.6875 * np.pi / 180.0, 24.433 * np.pi / 180.0, 0.3)
ref_xyz_moon = {
    "SPHERE": (1581421.43506347, 718463.12201783, 20843.2071012),
    "GSFC": (1582332.08831085, 718876.84524219, 20805.18709001),
    "GRAIL23": (1581855.3916402, 718660.27490195, 20836.2107652),
    "CE-1-LAM-GEO": (1581905.99108228, 718683.26297605, 20806.77965693),
}


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


@pytest.fixture(scope="module")
def enu_mcmf_info():
    center_lat, center_lon, center_alt = [
        0.6875 * np.pi / 180.0,
        24.433 * np.pi / 180.0,
        0.3,
    ]

    # Creating a test pattern of a circle of antennas, radius 500 m in ENU coordinates.
    angs = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    enus = 500 * np.array([np.cos(angs), np.sin(angs), [0] * angs.size])
    east = enus[0].tolist()
    north = enus[1].tolist()
    up = enus[2].tolist()

    # fmt: off
    lats = {
        "SPHERE": np.deg2rad(
            [
                0.68749997, 0.69719361, 0.70318462, 0.70318462, 0.69719361,
                0.68749997, 0.67780635, 0.67181538, 0.67181538, 0.67780635
            ]
        ),
        "GSFC": np.deg2rad(
            [
                0.68749997, 0.69721132, 0.70321328, 0.70321328, 0.69721132,
                0.68749997, 0.67778864, 0.67178672, 0.67178672, 0.67778864
            ]
        ),
        "GRAIL23": np.deg2rad(
            [
                0.68749997, 0.69719686, 0.70318988, 0.70318988, 0.69719686,
                0.68749997, 0.6778031 , 0.67181011, 0.67181011, 0.6778031
            ]
        ),
        "CE-1-LAM-GEO": np.deg2rad(
            [
                0.68749997, 0.69721058, 0.70321207, 0.70321207, 0.69721058,
                0.68749997, 0.67778938, 0.67178792, 0.67178792, 0.67778938
            ]
        ),
    }
    lons = {
        "SPHERE": np.deg2rad(
            [
                24.44949297, 24.44634312, 24.43809663, 24.42790337, 24.41965688,
                24.41650703, 24.41965693, 24.42790341, 24.43809659, 24.44634307
            ]
        ),
        "GSFC": np.deg2rad(
            [
                24.44948348, 24.44633544, 24.43809369, 24.42790631, 24.41966456,
                24.41651652, 24.41966461, 24.42790634, 24.43809366, 24.44633539
            ]
        ),
        "GRAIL23": np.deg2rad(
            [
                24.44948845, 24.44633946, 24.43809523, 24.42790477, 24.41966054,
                24.41651155, 24.41966059, 24.42790481, 24.43809519, 24.44633941
            ]
        ),
        "CE-1-LAM-GEO": np.deg2rad(
            [
                24.44948792, 24.44633904, 24.43809507, 24.42790493, 24.41966096,
                24.41651208, 24.41966102, 24.42790497, 24.43809503, 24.44633898
            ]
        ),
    }
    alts = {
        "SPHERE": [
            0.371959, 0.371959, 0.371959, 0.371959, 0.371959, 0.371959,
            0.371959, 0.371959, 0.371959, 0.371959
        ],
        "GSFC": [
            0.37191758, 0.37197732, 0.37207396, 0.37207396, 0.37197732,
            0.37191758, 0.37197732, 0.37207396, 0.37207396, 0.37197732
        ],
        "GRAIL23": [
            0.37193926, 0.37195442, 0.37197896, 0.37197896, 0.37195442,
            0.37193926, 0.37195442, 0.37197896, 0.37197896, 0.37195442
        ],
        "CE-1-LAM-GEO": [
            0.37193696, 0.37198809, 0.37207083, 0.37207083, 0.37198809,
            0.37193696, 0.37198809, 0.37207083, 0.37207083, 0.37198809
        ],
    }
    x = {
        "SPHERE": [
            1581214.62062477, 1581250.9080965 , 1581352.33107362,
            1581480.14942611, 1581585.54088769, 1581628.24950218,
            1581591.96203044, 1581490.53905332, 1581362.72070084,
            1581257.32923925
        ],
        "GSFC": [
            1582125.27387214, 1582161.56134388, 1582262.984321,
            1582390.80267348, 1582496.19413507, 1582538.90274956,
            1582502.61527782, 1582401.1923007 , 1582273.37394822,
            1582167.98248663
        ],
        "GRAIL23": [
            1581648.57720149, 1581684.86467323, 1581786.28765035,
            1581914.10600283, 1582019.49746442, 1582062.2060789 ,
            1582025.91860717, 1581924.49563005, 1581796.67727756,
            1581691.28581598
        ],
        "CE-1-LAM-GEO": [
            1581699.17664357, 1581735.46411531, 1581836.88709243,
            1581964.70544491, 1582070.0969065 , 1582112.80552098,
            1582076.51804925, 1581975.09507213, 1581847.27671964,
            1581741.88525806
        ]
    }

    y = {
        "SPHERE": [
            718918.34480718, 718829.94638063, 718601.4335154 , 718320.09035913,
            718093.38043501, 718007.89922848, 718096.29765503, 718324.81052027,
            718606.15367654, 718832.86360065
        ],
        "GSFC": [
            719332.06803154, 719243.66960499, 719015.15673976, 718733.81358349,
            718507.10365937, 718421.62245284, 718510.02087939, 718738.53374463,
            719019.8769009 , 719246.58682501
        ],
        "GRAIL23": [
            719115.4976913 , 719027.09926475, 718798.58639952, 718517.24324325,
            718290.53331913, 718205.0521126 , 718293.45053915, 718521.96340439,
            718803.30656066, 719030.01648477
        ],
        "CE-1-LAM-GEO": [
            719138.4857654 , 719050.08733885, 718821.57447362, 718540.23131734,
            718313.52139323, 718228.0401867 , 718316.43861325, 718544.95147849,
            718826.29463476, 719053.00455887
        ],
    }
    z = {
        "SPHERE": [
            20843.2071012 , 21137.07857037, 21318.70112664, 21318.70112664,
            21137.07857037, 20843.2071012 , 20549.33563204, 20367.71307577,
            20367.71307577, 20549.33563204
        ],
        "GSFC": [
            20805.18709001, 21099.05855918, 21280.68111545, 21280.68111545,
            21099.05855918, 20805.18709001, 20511.31562084, 20329.69306457,
            20329.69306457, 20511.31562084
        ],
        "GRAIL23": [
            20836.2107652 , 21130.08223437, 21311.70479064, 21311.70479064,
            21130.08223437, 20836.2107652 , 20542.33929603, 20360.71673976,
            20360.71673976, 20542.33929603
        ],
        "CE-1-LAM-GEO": [
            20806.77965693, 21100.6511261 , 21282.27368237, 21282.27368237,
            21100.6511261 , 20806.77965693, 20512.90818776, 20331.28563149,
            20331.28563149, 20512.90818776
        ],
    }

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


def test_XYZ_from_LatLonAlt():
    """Test conversion from lat/lon/alt to ECEF xyz with reference values."""
    out_xyz = utils.XYZ_from_LatLonAlt(
        ref_latlonalt[0], ref_latlonalt[1], ref_latlonalt[2]
    )
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    np.testing.assert_allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3)

    # test error checking
    with pytest.raises(
        ValueError,
        match="latitude, longitude and altitude must all have the same length",
    ):
        utils.XYZ_from_LatLonAlt(
            ref_latlonalt[0],
            ref_latlonalt[1],
            np.array([ref_latlonalt[2], ref_latlonalt[2]]),
        )

    with pytest.raises(
        ValueError,
        match="latitude, longitude and altitude must all have the same length",
    ):
        utils.XYZ_from_LatLonAlt(
            ref_latlonalt[0],
            np.array([ref_latlonalt[1], ref_latlonalt[1]]),
            ref_latlonalt[2],
        )


def test_LatLonAlt_from_XYZ():
    """Test conversion from ECEF xyz to lat/lon/alt with reference values."""
    out_latlonalt = utils.LatLonAlt_from_XYZ(ref_xyz)
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    np.testing.assert_allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3)
    pytest.raises(ValueError, utils.LatLonAlt_from_XYZ, ref_latlonalt)

    # test passing multiple values
    xyz_mult = np.stack((np.array(ref_xyz), np.array(ref_xyz)))
    lat_vec, lon_vec, alt_vec = utils.LatLonAlt_from_XYZ(xyz_mult)
    np.testing.assert_allclose(
        ref_latlonalt, (lat_vec[1], lon_vec[1], alt_vec[1]), rtol=0, atol=1e-3
    )
    # check error if array transposed
    with pytest.raises(
        ValueError,
        match=re.escape("The expected shape of ECEF xyz array is (Npts, 3)."),
    ):
        utils.LatLonAlt_from_XYZ(xyz_mult.T)

    # check error if only 2 coordinates
    with pytest.raises(
        ValueError,
        match=re.escape("The expected shape of ECEF xyz array is (Npts, 3)."),
    ):
        utils.LatLonAlt_from_XYZ(xyz_mult[:, 0:2])

    # test error checking
    pytest.raises(ValueError, utils.LatLonAlt_from_XYZ, ref_xyz[0:1])


@pytest.mark.parametrize("selenoid", selenoids)
def test_XYZ_from_LatLonAlt_mcmf(selenoid):
    """Test MCMF lat/lon/alt to xyz with reference values."""
    pytest.importorskip("lunarsky")
    lat, lon, alt = ref_latlonalt_moon
    out_xyz = utils.XYZ_from_LatLonAlt(lat, lon, alt, frame="mcmf", ellipsoid=selenoid)
    np.testing.assert_allclose(ref_xyz_moon[selenoid], out_xyz, rtol=0, atol=1e-3)

    # test default ellipsoid
    if selenoid == "SPHERE":
        out_xyz = utils.XYZ_from_LatLonAlt(lat, lon, alt, frame="mcmf")
        np.testing.assert_allclose(ref_xyz_moon[selenoid], out_xyz, rtol=0, atol=1e-3)

    # Test errors with invalid frame
    with pytest.raises(
        ValueError, match="No cartesian to spherical transform defined for frame"
    ):
        utils.XYZ_from_LatLonAlt(lat, lon, alt, frame="undef")


@pytest.mark.parametrize("selenoid", selenoids)
def test_LatLonAlt_from_XYZ_mcmf(selenoid):
    """Test MCMF xyz to lat/lon/alt with reference values."""
    pytest.importorskip("lunarsky")
    out_latlonalt = utils.LatLonAlt_from_XYZ(
        ref_xyz_moon[selenoid], frame="mcmf", ellipsoid=selenoid
    )
    np.testing.assert_allclose(ref_latlonalt_moon, out_latlonalt, rtol=0, atol=1e-3)

    # test default ellipsoid
    if selenoid == "SPHERE":
        out_latlonalt = utils.LatLonAlt_from_XYZ(ref_xyz_moon[selenoid], frame="mcmf")
        np.testing.assert_allclose(ref_latlonalt_moon, out_latlonalt, rtol=0, atol=1e-3)

    # Test errors with invalid frame
    with pytest.raises(
        ValueError, match="Cannot check acceptability for unknown frame"
    ):
        out_latlonalt = utils.LatLonAlt_from_XYZ(ref_xyz_moon[selenoid], frame="undef")
    with pytest.raises(
        ValueError, match="No spherical to cartesian transform defined for frame"
    ):
        utils.LatLonAlt_from_XYZ(
            ref_xyz_moon[selenoid], frame="undef", check_acceptability=False
        )


@pytest.mark.skipif(
    len(frame_selenoid) > 1, reason="Test only when lunarsky not installed."
)
def test_no_moon():
    """Check errors when calling functions with MCMF without lunarsky."""

    msg = "Need to install `lunarsky` package to work with MCMF frame."
    with pytest.raises(ImportError, match=msg):
        utils.get_lst_for_time(
            [2451545.0], latitude=0, longitude=0, altitude=0, frame="mcmf"
        )

    msg = "Need to install `lunarsky` package to work with selenoids or MCMF frame."
    with pytest.raises(ImportError, match=msg):
        utils.coordinates.get_selenoids()

    with pytest.raises(ImportError, match=msg):
        utils.LatLonAlt_from_XYZ(ref_xyz_moon["SPHERE"], frame="mcmf")

    lat, lon, alt = ref_latlonalt_moon
    with pytest.raises(ImportError, match=msg):
        utils.XYZ_from_LatLonAlt(lat, lon, alt, frame="mcmf")

    with pytest.raises(ImportError, match=msg):
        utils.ENU_from_ECEF(
            None, latitude=0.0, longitude=1.0, altitude=10.0, frame="mcmf"
        )

    with pytest.raises(ImportError, match=msg):
        utils.ECEF_from_ENU(
            None, latitude=0.0, longitude=1.0, altitude=10.0, frame="mcmf"
        )

    msg = "Need to install `lunarsky` package to work with MoonLocations."
    with pytest.raises(ImportError, match=msg):
        utils.coordinates.check_surface_based_positions(
            telescope_frame="mcmf", telescope_loc="foo"
        )


def test_lla_xyz_lla_roundtrip():
    """Test roundtripping an array will yield the same values."""
    np.random.seed(0)
    lats = -30.721 + np.random.normal(0, 0.0005, size=30)
    lons = 21.428 + np.random.normal(0, 0.0005, size=30)
    alts = np.random.uniform(1051, 1054, size=30)
    lats *= np.pi / 180.0
    lons *= np.pi / 180.0
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)
    lats_new, lons_new, alts_new = utils.LatLonAlt_from_XYZ(xyz)
    np.testing.assert_allclose(lats_new, lats, rtol=0, atol=utils.RADIAN_TOL)
    np.testing.assert_allclose(lons_new, lons, rtol=0, atol=utils.RADIAN_TOL)
    np.testing.assert_allclose(alts_new, alts, rtol=0, atol=1e-3)


def test_xyz_from_latlonalt(enu_ecef_info):
    """Test calculating xyz from lat lot alt."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)
    np.testing.assert_allclose(np.stack((x, y, z), axis=1), xyz, atol=1e-3)


def test_enu_from_ecef(enu_ecef_info):
    """Test calculating ENU from ECEF coordinates."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)

    enu = utils.ENU_from_ECEF(
        xyz, latitude=center_lat, longitude=center_lon, altitude=center_alt
    )
    np.testing.assert_allclose(
        np.stack((east, north, up), axis=1), enu, rtol=0, atol=1e-3
    )

    enu2 = utils.ENU_from_ECEF(
        xyz,
        center_loc=EarthLocation.from_geodetic(
            lat=center_lat * units.rad,
            lon=center_lon * units.rad,
            height=center_alt * units.m,
        ),
    )
    np.testing.assert_allclose(enu, enu2, rtol=0, atol=1e-3)


@pytest.mark.parametrize("selenoid", selenoids)
def test_enu_from_mcmf(enu_mcmf_info, selenoid):
    pytest.importorskip("lunarsky")
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_mcmf_info
    )
    xyz = utils.XYZ_from_LatLonAlt(
        lats[selenoid], lons[selenoid], alts[selenoid], frame="mcmf", ellipsoid=selenoid
    )
    enu = utils.ENU_from_ECEF(
        xyz,
        latitude=center_lat,
        longitude=center_lon,
        altitude=center_alt,
        frame="mcmf",
        ellipsoid=selenoid,
    )

    np.testing.assert_allclose(
        np.stack((east, north, up), axis=1), enu, rtol=0, atol=1e-3
    )

    enu2 = utils.ENU_from_ECEF(
        xyz,
        center_loc=MoonLocation.from_selenodetic(
            lat=center_lat * units.rad,
            lon=center_lon * units.rad,
            height=center_alt * units.m,
            ellipsoid=selenoid,
        ),
    )
    np.testing.assert_allclose(enu, enu2, rtol=0, atol=1e-3)


def test_invalid_frame():
    """Test error is raised when an invalid frame name is passed in."""
    with pytest.raises(
        ValueError, match='No ENU_from_ECEF transform defined for frame "UNDEF".'
    ):
        utils.ENU_from_ECEF(
            np.zeros((2, 3)), latitude=0.0, longitude=0.0, altitude=0.0, frame="undef"
        )
    with pytest.raises(
        ValueError, match='No ECEF_from_ENU transform defined for frame "UNDEF".'
    ):
        utils.ECEF_from_ENU(
            np.zeros((2, 3)), latitude=0.0, longitude=0.0, altitude=0.0, frame="undef"
        )

    with pytest.raises(
        ValueError, match="center_loc is not a supported type. It must be one of "
    ):
        utils.ENU_from_ECEF(
            np.zeros((2, 3)), center_loc=units.Quantity(np.array([0, 0, 0]) * units.m)
        )

    with pytest.raises(
        ValueError, match="center_loc is not a supported type. It must be one of "
    ):
        utils.ECEF_from_ENU(
            np.zeros((2, 3)), center_loc=units.Quantity(np.array([0, 0, 0]) * units.m)
        )


@pytest.mark.parametrize("shape_type", ["transpose", "Nblts,2", "Nblts,1"])
def test_enu_from_ecef_shape_errors(enu_ecef_info, shape_type):
    """Test ENU_from_ECEF input shape errors."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)
    if shape_type == "transpose":
        xyz = xyz.T.copy()
    elif shape_type == "Nblts,2":
        xyz = xyz.copy()[:, 0:2]
    elif shape_type == "Nblts,1":
        xyz = xyz.copy()[:, 0:1]

    # check error if array transposed
    with pytest.raises(
        ValueError,
        match=re.escape("The expected shape of ECEF xyz array is (Npts, 3)."),
    ):
        utils.ENU_from_ECEF(
            xyz, longitude=center_lat, latitude=center_lon, altitude=center_alt
        )


def test_enu_from_ecef_magnitude_error(enu_ecef_info):
    """Test ENU_from_ECEF input magnitude errors."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)
    # error checking
    with pytest.raises(
        ValueError,
        match="itrs position vector magnitudes must be on the order of the "
        "radius of Earth",
    ):
        utils.ENU_from_ECEF(
            xyz / 2.0, latitude=center_lat, longitude=center_lon, altitude=center_alt
        )


def test_enu_from_ecef_error():
    # check error no center location info passed
    with pytest.raises(
        ValueError,
        match="Either center_loc or all of latitude, longitude and altitude "
        "must be passed.",
    ):
        utils.ENU_from_ECEF(np.array([0, 0, 0]))

    with pytest.raises(
        ValueError,
        match="Either center_loc or all of latitude, longitude and altitude "
        "must be passed.",
    ):
        utils.ECEF_from_ENU(np.array([0, 0, 0]))


@pytest.mark.parametrize(["frame", "selenoid"], frame_selenoid)
def test_ecef_from_enu_roundtrip(enu_ecef_info, enu_mcmf_info, frame, selenoid):
    """Test ECEF_from_ENU values."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info if frame == "itrs" else enu_mcmf_info
    )
    if frame == "mcmf":
        lats = lats[selenoid]
        lons = lons[selenoid]
        alts = alts[selenoid]
        loc_obj = MoonLocation.from_selenodetic(
            lat=center_lat * units.rad,
            lon=center_lon * units.rad,
            height=center_alt * units.m,
            ellipsoid=selenoid,
        )
    else:
        loc_obj = EarthLocation.from_geodetic(
            lat=center_lat * units.rad,
            lon=center_lon * units.rad,
            height=center_alt * units.m,
        )

    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts, frame=frame, ellipsoid=selenoid)
    enu = utils.ENU_from_ECEF(
        xyz,
        latitude=center_lat,
        longitude=center_lon,
        altitude=center_alt,
        frame=frame,
        ellipsoid=selenoid,
    )
    # check that a round trip gives the original value.
    xyz_from_enu = utils.ECEF_from_ENU(
        enu,
        latitude=center_lat,
        longitude=center_lon,
        altitude=center_alt,
        frame=frame,
        ellipsoid=selenoid,
    )
    np.testing.assert_allclose(xyz, xyz_from_enu, rtol=0, atol=1e-3)

    xyz_from_enu2 = utils.ECEF_from_ENU(enu, center_loc=loc_obj)
    np.testing.assert_allclose(xyz_from_enu, xyz_from_enu2, rtol=0, atol=1e-3)

    if selenoid == "SPHERE":
        enu = utils.ENU_from_ECEF(
            xyz,
            latitude=center_lat,
            longitude=center_lon,
            altitude=center_alt,
            frame=frame,
        )
        # check that a round trip gives the original value.
        xyz_from_enu = utils.ECEF_from_ENU(
            enu,
            latitude=center_lat,
            longitude=center_lon,
            altitude=center_alt,
            frame=frame,
        )
        np.testing.assert_allclose(xyz, xyz_from_enu, rtol=0, atol=1e-3)


@pytest.mark.parametrize("shape_type", ["transpose", "Nblts,2", "Nblts,1"])
def test_ecef_from_enu_shape_errors(enu_ecef_info, shape_type):
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)
    enu = utils.ENU_from_ECEF(
        xyz, latitude=center_lat, longitude=center_lon, altitude=center_alt
    )
    if shape_type == "transpose":
        enu = enu.copy().T
    elif shape_type == "Nblts,2":
        enu = enu.copy()[:, 0:2]
    elif shape_type == "Nblts,1":
        enu = enu.copy()[:, 0:1]

    # check error if array transposed
    with pytest.raises(
        ValueError, match=re.escape("The expected shape of the ENU array is (Npts, 3).")
    ):
        utils.ECEF_from_ENU(
            enu, latitude=center_lat, longitude=center_lon, altitude=center_alt
        )


def test_ecef_from_enu_single(enu_ecef_info):
    """Test single coordinate transform."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)
    # check passing a single value
    enu_single = utils.ENU_from_ECEF(
        xyz[0, :], latitude=center_lat, longitude=center_lon, altitude=center_alt
    )

    np.testing.assert_allclose(
        np.array((east[0], north[0], up[0])), enu_single, rtol=0, atol=1e-3
    )


def test_ecef_from_enu_single_roundtrip(enu_ecef_info):
    """Test single coordinate roundtrip."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = utils.XYZ_from_LatLonAlt(lats, lons, alts)
    # check passing a single value
    enu = utils.ENU_from_ECEF(
        xyz, latitude=center_lat, longitude=center_lon, altitude=center_alt
    )

    enu_single = utils.ENU_from_ECEF(
        xyz[0, :], latitude=center_lat, longitude=center_lon, altitude=center_alt
    )
    np.testing.assert_allclose(
        np.array((east[0], north[0], up[0])), enu[0, :], rtol=0, atol=1e-3
    )

    xyz_from_enu = utils.ECEF_from_ENU(
        enu_single, latitude=center_lat, longitude=center_lon, altitude=center_alt
    )
    np.testing.assert_allclose(xyz[0, :], xyz_from_enu, rtol=0, atol=1e-3)


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
    lat, lon, alt = utils.LatLonAlt_from_XYZ(arrcent)

    # The STABXYZ coordinates are defined with X through the local meridian,
    # so rotate back to the prime meridian
    new_xyz = utils.ECEF_from_rotECEF(xyz.T, lon)
    # add in array center to get real ECEF
    ecef_xyz = new_xyz + arrcent

    enu = utils.ENU_from_ECEF(ecef_xyz, latitude=lat, longitude=lon, altitude=alt)

    np.testing.assert_allclose(enu, enh, rtol=0, atol=1e-3)

    # test other direction of ECEF rotation
    rot_xyz = utils.rotECEF_from_ECEF(new_xyz, lon)
    np.testing.assert_allclose(rot_xyz.T, xyz, rtol=0, atol=1e-3)


def test_hpx_latlon_az_za():
    zenith_angle = np.deg2rad(np.linspace(0, 90, 10))
    azimuth = np.deg2rad(np.linspace(0, 360, 36, endpoint=False))
    az_mesh, za_mesh = np.meshgrid(azimuth, zenith_angle)

    hpx_lat = np.deg2rad(np.linspace(90, 0, 10))
    hpx_lon = np.deg2rad(np.linspace(0, 360, 36, endpoint=False))
    lon_mesh, lat_mesh = np.meshgrid(hpx_lon, hpx_lat)

    with pytest.raises(
        ValueError, match="shapes of zenith_angle and azimuth values must match."
    ):
        utils.coordinates.zenithangle_azimuth_to_hpx_latlon(zenith_angle, azimuth)

    calc_lat, calc_lon = utils.coordinates.zenithangle_azimuth_to_hpx_latlon(
        za_mesh, az_mesh
    )

    np.testing.assert_allclose(calc_lat, lat_mesh, rtol=0, atol=utils.RADIAN_TOL)
    np.testing.assert_allclose(calc_lon, lon_mesh, rtol=0, atol=utils.RADIAN_TOL)

    with pytest.raises(
        ValueError, match="shapes of hpx_lat and hpx_lon values must match."
    ):
        utils.coordinates.hpx_latlon_to_zenithangle_azimuth(hpx_lat, hpx_lon)

    calc_za, calc_az = utils.coordinates.hpx_latlon_to_zenithangle_azimuth(
        lat_mesh, lon_mesh
    )

    np.testing.assert_allclose(calc_za, za_mesh, rtol=0, atol=utils.RADIAN_TOL)
    np.testing.assert_allclose(calc_az, az_mesh, rtol=0, atol=utils.RADIAN_TOL)


@pytest.mark.parametrize("err_state", ["err", "warn", "none"])
@pytest.mark.parametrize("tel_loc", ["Center", "Moon", "Earth", "Space"])
@pytest.mark.parametrize("check_frame", ["Moon", "Earth"])
@pytest.mark.parametrize("del_tel_loc", [False, None, True])
def test_check_surface_based_positions(err_state, tel_loc, check_frame, del_tel_loc):
    tel_loc_dict = {
        "Center": np.array([0, 0, 0]),
        "Moon": np.array([0, 0, 1.737e6]),
        "Earth": np.array([0, 6.37e6, 0]),
        "Space": np.array([4.22e7, 0, 0]),
    }
    tel_frame_dict = {"Moon": "mcmf", "Earth": "itrs"}

    ant_pos = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    if del_tel_loc:
        ant_pos += tel_loc_dict[tel_loc]

    fail_type = err_msg = err_type = None
    err_check = check_warnings
    if (tel_loc != check_frame) and (err_state != "none"):
        if tel_loc == "Center":
            fail_type = "below"
        elif tel_loc == "Space":
            fail_type = "above"
        else:
            fail_type = "above" if tel_loc == "Earth" else "below"

    if fail_type is not None:
        err_msg = (
            f"{tel_frame_dict[check_frame]} position vector magnitudes must be "
            f"on the order of the radius of {check_frame} -- they appear to lie well "
            f"{fail_type} this."
        )
        if err_state == "err":
            err_type = ValueError
            err_check = pytest.raises
        else:
            err_type = UserWarning

        with err_check(err_type, match=err_msg):
            status = utils.coordinates.check_surface_based_positions(
                telescope_loc=None if (del_tel_loc) else tel_loc_dict[tel_loc],
                antenna_positions=None if (del_tel_loc is None) else ant_pos,
                telescope_frame=tel_frame_dict[check_frame],
                raise_error=err_state == "err",
                raise_warning=err_state == "warn",
            )

        assert (err_state == "err") or (status == (tel_loc == check_frame))


@pytest.mark.parametrize("tel_loc", ["Earth", "Moon"])
@pytest.mark.parametrize("check_frame", ["Earth", "Moon"])
def test_check_surface_based_positions_earthmoonloc(tel_loc, check_frame):
    frame = "mcmf" if (check_frame == "Moon") else "itrs"

    if tel_loc == "Earth":
        loc = EarthLocation.from_geodetic(0, 0, 0)
    else:
        pytest.importorskip("lunarsky")
        loc = MoonLocation.from_selenodetic(0, 0, 0)

    if tel_loc == check_frame:
        assert utils.coordinates.check_surface_based_positions(
            telescope_loc=loc, telescope_frame=frame
        )
    else:
        with pytest.raises(ValueError, match=(f"{frame} position vector")):
            utils.coordinates.check_surface_based_positions(
                telescope_loc=[loc.x.value, loc.y.value, loc.z.value],
                telescope_frame=frame,
            )
