# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for common utility functions."""
import copy
import os
import re

import numpy as np
import pytest
from astropy import units
from astropy import units as un
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time

import pyuvdata.utils as uvutils
from pyuvdata import UVCal, UVData, UVFlag
from pyuvdata.data import DATA_PATH
from pyuvdata.utils import hasmoon

from . import check_warnings

selenoids = ["SPHERE", "GSFC", "GRAIL23", "CE-1-LAM-GEO"]

if hasmoon:
    from pyuvdata.utils import LTime, MoonLocation

    frame_selenoid = [["itrs", None]]
    for snd in selenoids:
        frame_selenoid.append(["mcmf", snd])
else:
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

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values",
    "ignore:antenna_positions are not set or are being overwritten. Using known values",
)


@pytest.fixture(scope="session")
def astrometry_args():
    default_args = {
        "time_array": 2456789.0 + np.array([0.0, 1.25, 10.5, 100.75]),
        "icrs_ra": 2.468,
        "icrs_dec": 1.234,
        "epoch": 2000.0,
        "telescope_loc": (0.123, -0.456, 4321.0),
        "telescope_frame": "itrs",
        "pm_ra": 12.3,
        "pm_dec": 45.6,
        "vrad": 31.4,
        "dist": 73.31,
        "library": "erfa",
    }
    default_args["lst_array"] = uvutils.get_lst_for_time(
        jd_array=default_args["time_array"],
        latitude=default_args["telescope_loc"][0] * (180.0 / np.pi),
        longitude=default_args["telescope_loc"][1] * (180.0 / np.pi),
        altitude=default_args["telescope_loc"][2],
        frame="itrs",
    )

    default_args["drift_coord"] = SkyCoord(
        default_args["lst_array"],
        [default_args["telescope_loc"][0]] * len(default_args["lst_array"]),
        unit="rad",
    )

    if hasmoon:
        default_args["moon_telescope_loc"] = (
            0.6875 * np.pi / 180.0,
            24.433 * np.pi / 180.0,
            0.3,
        )
        default_args["moon_lst_array"] = {}
        default_args["moon_drift_coord"] = {}
        for selenoid in selenoids:
            default_args["moon_lst_array"][selenoid] = uvutils.get_lst_for_time(
                jd_array=default_args["time_array"],
                latitude=default_args["moon_telescope_loc"][0] * (180.0 / np.pi),
                longitude=default_args["moon_telescope_loc"][1] * (180.0 / np.pi),
                altitude=default_args["moon_telescope_loc"][2],
                frame="mcmf",
                ellipsoid=selenoid,
            )
            default_args["moon_drift_coord"][selenoid] = SkyCoord(
                default_args["moon_lst_array"][selenoid],
                [default_args["moon_telescope_loc"][0]]
                * len(default_args["moon_lst_array"][selenoid]),
                unit="rad",
            )

    default_args["icrs_coord"] = SkyCoord(
        default_args["icrs_ra"], default_args["icrs_dec"], unit="rad"
    )

    default_args["fk5_ra"], default_args["fk5_dec"] = uvutils.transform_sidereal_coords(
        longitude=default_args["icrs_ra"],
        latitude=default_args["icrs_dec"],
        in_coord_frame="icrs",
        out_coord_frame="fk5",
        in_coord_epoch="J2000.0",
        out_coord_epoch="J2000.0",
    )

    # These are values calculated w/o the optional arguments, e.g. pm, vrad, dist
    default_args["app_ra"], default_args["app_dec"] = uvutils.transform_icrs_to_app(
        time_array=default_args["time_array"],
        ra=default_args["icrs_ra"],
        dec=default_args["icrs_dec"],
        telescope_loc=default_args["telescope_loc"],
    )

    default_args["app_coord"] = SkyCoord(
        default_args["app_ra"], default_args["app_dec"], unit="rad"
    )

    if hasmoon:
        default_args["moon_app_ra"] = {}
        default_args["moon_app_dec"] = {}
        default_args["moon_app_coord"] = {}
        for selenoid in selenoids:
            (
                default_args["moon_app_ra"][selenoid],
                default_args["moon_app_dec"][selenoid],
            ) = uvutils.transform_icrs_to_app(
                time_array=default_args["time_array"],
                ra=default_args["icrs_ra"],
                dec=default_args["icrs_dec"],
                telescope_loc=default_args["moon_telescope_loc"],
                telescope_frame="mcmf",
                ellipsoid=selenoid,
            )

            default_args["moon_app_coord"][selenoid] = SkyCoord(
                default_args["moon_app_ra"][selenoid],
                default_args["moon_app_dec"][selenoid],
                unit="rad",
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


@pytest.fixture(scope="session")
def utils_uvdata_main():
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.uvh5"))

    yield uvd


@pytest.fixture(scope="function")
def utils_uvdata(utils_uvdata_main):
    uvd = utils_uvdata_main.copy()

    yield uvd


def test_XYZ_from_LatLonAlt():
    """Test conversion from lat/lon/alt to ECEF xyz with reference values."""
    out_xyz = uvutils.XYZ_from_LatLonAlt(
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
    np.testing.assert_allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3)
    pytest.raises(ValueError, uvutils.LatLonAlt_from_XYZ, ref_latlonalt)

    # test passing multiple values
    xyz_mult = np.stack((np.array(ref_xyz), np.array(ref_xyz)))
    lat_vec, lon_vec, alt_vec = uvutils.LatLonAlt_from_XYZ(xyz_mult)
    np.testing.assert_allclose(
        ref_latlonalt, (lat_vec[1], lon_vec[1], alt_vec[1]), rtol=0, atol=1e-3
    )
    # check error if array transposed
    with pytest.raises(
        ValueError,
        match=re.escape("The expected shape of ECEF xyz array is (Npts, 3)."),
    ):
        uvutils.LatLonAlt_from_XYZ(xyz_mult.T)

    # check error if only 2 coordinates
    with pytest.raises(
        ValueError,
        match=re.escape("The expected shape of ECEF xyz array is (Npts, 3)."),
    ):
        uvutils.LatLonAlt_from_XYZ(xyz_mult[:, 0:2])

    # test error checking
    pytest.raises(ValueError, uvutils.LatLonAlt_from_XYZ, ref_xyz[0:1])


@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
@pytest.mark.parametrize("selenoid", selenoids)
def test_XYZ_from_LatLonAlt_mcmf(selenoid):
    """Test MCMF lat/lon/alt to xyz with reference values."""
    lat, lon, alt = ref_latlonalt_moon
    out_xyz = uvutils.XYZ_from_LatLonAlt(
        lat, lon, alt, frame="mcmf", ellipsoid=selenoid
    )
    np.testing.assert_allclose(ref_xyz_moon[selenoid], out_xyz, rtol=0, atol=1e-3)

    # test default ellipsoid
    if selenoid == "SPHERE":
        out_xyz = uvutils.XYZ_from_LatLonAlt(lat, lon, alt, frame="mcmf")
        np.testing.assert_allclose(ref_xyz_moon[selenoid], out_xyz, rtol=0, atol=1e-3)

    # Test errors with invalid frame
    with pytest.raises(
        ValueError, match="No cartesian to spherical transform defined for frame"
    ):
        uvutils.XYZ_from_LatLonAlt(lat, lon, alt, frame="undef")


@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
@pytest.mark.parametrize("selenoid", selenoids)
def test_LatLonAlt_from_XYZ_mcmf(selenoid):
    """Test MCMF xyz to lat/lon/alt with reference values."""
    out_latlonalt = uvutils.LatLonAlt_from_XYZ(
        ref_xyz_moon[selenoid], frame="mcmf", ellipsoid=selenoid
    )
    np.testing.assert_allclose(ref_latlonalt_moon, out_latlonalt, rtol=0, atol=1e-3)

    # test default ellipsoid
    if selenoid == "SPHERE":
        out_latlonalt = uvutils.LatLonAlt_from_XYZ(ref_xyz_moon[selenoid], frame="mcmf")
        np.testing.assert_allclose(ref_latlonalt_moon, out_latlonalt, rtol=0, atol=1e-3)

    # Test errors with invalid frame
    with pytest.raises(
        ValueError, match="Cannot check acceptability for unknown frame"
    ):
        out_latlonalt = uvutils.LatLonAlt_from_XYZ(
            ref_xyz_moon[selenoid], frame="undef"
        )
    with pytest.raises(
        ValueError, match="No spherical to cartesian transform defined for frame"
    ):
        uvutils.LatLonAlt_from_XYZ(
            ref_xyz_moon[selenoid], frame="undef", check_acceptability=False
        )


@pytest.mark.skipif(hasmoon, reason="Test only when lunarsky not installed.")
def test_no_moon():
    """Check errors when calling functions with MCMF without lunarsky."""
    msg = "Need to install `lunarsky` package to work with MCMF frame."
    with pytest.raises(ValueError, match=msg):
        uvutils.LatLonAlt_from_XYZ(ref_xyz_moon["SPHERE"], frame="mcmf")
    lat, lon, alt = ref_latlonalt_moon
    with pytest.raises(ValueError, match=msg):
        uvutils.XYZ_from_LatLonAlt(lat, lon, alt, frame="mcmf")
    with pytest.raises(ValueError, match=msg):
        uvutils.get_lst_for_time(
            [2451545.0], latitude=0, longitude=0, altitude=0, frame="mcmf"
        )
    with pytest.raises(ValueError, match=msg):
        uvutils.ENU_from_ECEF(
            None, latitude=0.0, longitude=1.0, altitude=10.0, frame="mcmf"
        )
    with pytest.raises(ValueError, match=msg):
        uvutils.ECEF_from_ENU(
            None, latitude=0.0, longitude=1.0, altitude=10.0, frame="mcmf"
        )
    with pytest.raises(ValueError, match=msg):
        uvutils.transform_icrs_to_app(
            time_array=[2451545.0],
            ra=0,
            dec=0,
            telescope_loc=(0, 0, 0),
            telescope_frame="mcmf",
        )
    with pytest.raises(ValueError, match=msg):
        uvutils.transform_app_to_icrs(
            time_array=[2451545.0],
            app_ra=0,
            app_dec=0,
            telescope_loc=(0, 0, 0),
            telescope_frame="mcmf",
        )
    with pytest.raises(ValueError, match=msg):
        uvutils.calc_app_coords(lon_coord=0.0, lat_coord=0.0, telescope_frame="mcmf")


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
    np.testing.assert_allclose(lats_new, lats)
    np.testing.assert_allclose(lons_new, lons)
    np.testing.assert_allclose(alts_new, alts)


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


def test_xyz_from_latlonalt(enu_ecef_info):
    """Test calculating xyz from lat lot alt."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    np.testing.assert_allclose(np.stack((x, y, z), axis=1), xyz, atol=1e-3)


def test_enu_from_ecef(enu_ecef_info):
    """Test calculating ENU from ECEF coordinates."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)

    enu = uvutils.ENU_from_ECEF(
        xyz, latitude=center_lat, longitude=center_lon, altitude=center_alt
    )
    np.testing.assert_allclose(np.stack((east, north, up), axis=1), enu, atol=1e-3)

    enu2 = uvutils.ENU_from_ECEF(
        xyz,
        center_loc=EarthLocation.from_geodetic(
            lat=center_lat * units.rad,
            lon=center_lon * units.rad,
            height=center_alt * units.m,
        ),
    )
    np.testing.assert_allclose(enu, enu2)


@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
@pytest.mark.parametrize("selenoid", selenoids)
def test_enu_from_mcmf(enu_mcmf_info, selenoid):
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_mcmf_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(
        lats[selenoid], lons[selenoid], alts[selenoid], frame="mcmf", ellipsoid=selenoid
    )
    enu = uvutils.ENU_from_ECEF(
        xyz,
        latitude=center_lat,
        longitude=center_lon,
        altitude=center_alt,
        frame="mcmf",
        ellipsoid=selenoid,
    )

    np.testing.assert_allclose(np.stack((east, north, up), axis=1), enu, atol=1e-3)

    enu2 = uvutils.ENU_from_ECEF(
        xyz,
        center_loc=MoonLocation.from_selenodetic(
            lat=center_lat * units.rad,
            lon=center_lon * units.rad,
            height=center_alt * units.m,
            ellipsoid=selenoid,
        ),
    )
    np.testing.assert_allclose(enu, enu2, atol=1e-3)


def test_invalid_frame():
    """Test error is raised when an invalid frame name is passed in."""
    with pytest.raises(
        ValueError, match='No ENU_from_ECEF transform defined for frame "UNDEF".'
    ):
        uvutils.ENU_from_ECEF(
            np.zeros((2, 3)), latitude=0.0, longitude=0.0, altitude=0.0, frame="undef"
        )
    with pytest.raises(
        ValueError, match='No ECEF_from_ENU transform defined for frame "UNDEF".'
    ):
        uvutils.ECEF_from_ENU(
            np.zeros((2, 3)), latitude=0.0, longitude=0.0, altitude=0.0, frame="undef"
        )

    with pytest.raises(
        ValueError, match="center_loc is not a supported type. It must be one of "
    ):
        uvutils.ENU_from_ECEF(
            np.zeros((2, 3)), center_loc=units.Quantity(np.array([0, 0, 0]) * units.m)
        )

    with pytest.raises(
        ValueError, match="center_loc is not a supported type. It must be one of "
    ):
        uvutils.ECEF_from_ENU(
            np.zeros((2, 3)), center_loc=units.Quantity(np.array([0, 0, 0]) * units.m)
        )


@pytest.mark.parametrize("shape_type", ["transpose", "Nblts,2", "Nblts,1"])
def test_enu_from_ecef_shape_errors(enu_ecef_info, shape_type):
    """Test ENU_from_ECEF input shape errors."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
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
        uvutils.ENU_from_ECEF(
            xyz, longitude=center_lat, latitude=center_lon, altitude=center_alt
        )


def test_enu_from_ecef_magnitude_error(enu_ecef_info):
    """Test ENU_from_ECEF input magnitude errors."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    # error checking
    with pytest.raises(
        ValueError,
        match="ITRS vector magnitudes must be on the order of the radius of the earth",
    ):
        uvutils.ENU_from_ECEF(
            xyz / 2.0, latitude=center_lat, longitude=center_lon, altitude=center_alt
        )


def test_enu_from_ecef_error():
    # check error no center location info passed
    with pytest.raises(
        ValueError,
        match="Either center_loc or all of latitude, longitude and altitude "
        "must be passed.",
    ):
        uvutils.ENU_from_ECEF(np.array([0, 0, 0]))

    with pytest.raises(
        ValueError,
        match="Either center_loc or all of latitude, longitude and altitude "
        "must be passed.",
    ):
        uvutils.ECEF_from_ENU(np.array([0, 0, 0]))


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

    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts, frame=frame, ellipsoid=selenoid)
    enu = uvutils.ENU_from_ECEF(
        xyz,
        latitude=center_lat,
        longitude=center_lon,
        altitude=center_alt,
        frame=frame,
        ellipsoid=selenoid,
    )
    # check that a round trip gives the original value.
    xyz_from_enu = uvutils.ECEF_from_ENU(
        enu,
        latitude=center_lat,
        longitude=center_lon,
        altitude=center_alt,
        frame=frame,
        ellipsoid=selenoid,
    )
    np.testing.assert_allclose(xyz, xyz_from_enu, atol=1e-3)

    xyz_from_enu2 = uvutils.ECEF_from_ENU(enu, center_loc=loc_obj)
    np.testing.assert_allclose(xyz_from_enu, xyz_from_enu2, atol=1e-3)

    if selenoid == "SPHERE":
        enu = uvutils.ENU_from_ECEF(
            xyz,
            latitude=center_lat,
            longitude=center_lon,
            altitude=center_alt,
            frame=frame,
        )
        # check that a round trip gives the original value.
        xyz_from_enu = uvutils.ECEF_from_ENU(
            enu,
            latitude=center_lat,
            longitude=center_lon,
            altitude=center_alt,
            frame=frame,
        )
        np.testing.assert_allclose(xyz, xyz_from_enu, atol=1e-3)


@pytest.mark.parametrize("shape_type", ["transpose", "Nblts,2", "Nblts,1"])
def test_ecef_from_enu_shape_errors(enu_ecef_info, shape_type):
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    enu = uvutils.ENU_from_ECEF(
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
        uvutils.ECEF_from_ENU(
            enu, latitude=center_lat, longitude=center_lon, altitude=center_alt
        )


def test_ecef_from_enu_single(enu_ecef_info):
    """Test single coordinate transform."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    # check passing a single value
    enu_single = uvutils.ENU_from_ECEF(
        xyz[0, :], latitude=center_lat, longitude=center_lon, altitude=center_alt
    )

    np.testing.assert_allclose(
        np.array((east[0], north[0], up[0])), enu_single, atol=1e-3
    )


def test_ecef_from_enu_single_roundtrip(enu_ecef_info):
    """Test single coordinate roundtrip."""
    (center_lat, center_lon, center_alt, lats, lons, alts, x, y, z, east, north, up) = (
        enu_ecef_info
    )
    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    # check passing a single value
    enu = uvutils.ENU_from_ECEF(
        xyz, latitude=center_lat, longitude=center_lon, altitude=center_alt
    )

    enu_single = uvutils.ENU_from_ECEF(
        xyz[0, :], latitude=center_lat, longitude=center_lon, altitude=center_alt
    )
    np.testing.assert_allclose(
        np.array((east[0], north[0], up[0])), enu[0, :], atol=1e-3
    )

    xyz_from_enu = uvutils.ECEF_from_ENU(
        enu_single, latitude=center_lat, longitude=center_lon, altitude=center_alt
    )
    np.testing.assert_allclose(xyz[0, :], xyz_from_enu, atol=1e-3)


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

    enu = uvutils.ENU_from_ECEF(ecef_xyz, latitude=lat, longitude=lon, altitude=alt)

    np.testing.assert_allclose(enu, enh)

    # test other direction of ECEF rotation
    rot_xyz = uvutils.rotECEF_from_ECEF(new_xyz, lon)
    np.testing.assert_allclose(rot_xyz.T, xyz)


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
        uvutils.polar2_to_cart3(lon_array=lon_array, lat_array=lat_array)


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
        uvutils.cart3_to_polar2(input1)


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
    with pytest.raises(ValueError, match=msg):
        uvutils._rotate_matmul_wrapper(
            xyz_array=input1, rot_matrix=input2, n_rot=input3
        )


def test_cart_to_polar_roundtrip():
    """
    Test that polar->cart coord transformation is the inverse of cart->polar.
    """
    # Basic round trip with vectors
    assert uvutils.cart3_to_polar2(
        uvutils.polar2_to_cart3(lon_array=0.0, lat_array=0.0)
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
        uvutils._rotate_one_axis(xyz_array=x_vecs, rot_amount=0.0, rot_axis=0) == x_vecs
    )
    assert np.all(
        uvutils._rotate_one_axis(xyz_array=x_vecs[:, 0], rot_amount=0.0, rot_axis=1)
        == x_vecs[np.newaxis, :, 0, np.newaxis]
    )
    assert np.all(
        uvutils._rotate_one_axis(
            xyz_array=x_vecs[:, :, np.newaxis], rot_amount=0.0, rot_axis=2
        )
        == x_vecs[:, :, np.newaxis]
    )

    # Test no-ops w/ None
    assert np.all(
        uvutils._rotate_one_axis(xyz_array=test_vecs, rot_amount=None, rot_axis=1)
        == test_vecs
    )
    assert np.all(
        uvutils._rotate_one_axis(xyz_array=test_vecs[:, 0], rot_amount=None, rot_axis=2)
        == test_vecs[np.newaxis, :, 0, np.newaxis]
    )
    assert np.all(
        uvutils._rotate_one_axis(
            xyz_array=test_vecs[:, :, np.newaxis], rot_amount=None, rot_axis=0
        )
        == test_vecs[:, :, np.newaxis]
    )

    # Test some basic equivalencies to make sure rotations are working correctly
    assert np.allclose(
        x_vecs, uvutils._rotate_one_axis(xyz_array=x_vecs, rot_amount=1.0, rot_axis=0)
    )
    assert np.allclose(
        y_vecs, uvutils._rotate_one_axis(xyz_array=y_vecs, rot_amount=2.0, rot_axis=1)
    )
    assert np.allclose(
        z_vecs, uvutils._rotate_one_axis(xyz_array=z_vecs, rot_amount=3.0, rot_axis=2)
    )

    assert np.allclose(
        x_vecs,
        uvutils._rotate_one_axis(xyz_array=y_vecs, rot_amount=-np.pi / 2.0, rot_axis=2),
    )
    assert np.allclose(
        y_vecs,
        uvutils._rotate_one_axis(xyz_array=x_vecs, rot_amount=np.pi / 2.0, rot_axis=2),
    )
    assert np.allclose(
        x_vecs,
        uvutils._rotate_one_axis(xyz_array=z_vecs, rot_amount=np.pi / 2.0, rot_axis=1),
    )
    assert np.allclose(
        z_vecs,
        uvutils._rotate_one_axis(xyz_array=x_vecs, rot_amount=-np.pi / 2.0, rot_axis=1),
    )
    assert np.allclose(
        y_vecs,
        uvutils._rotate_one_axis(xyz_array=z_vecs, rot_amount=-np.pi / 2.0, rot_axis=0),
    )
    assert np.allclose(
        z_vecs,
        uvutils._rotate_one_axis(xyz_array=y_vecs, rot_amount=np.pi / 2.0, rot_axis=0),
    )

    assert np.all(
        np.equal(
            uvutils._rotate_one_axis(xyz_array=test_vecs, rot_amount=1.0, rot_axis=2),
            uvutils._rotate_one_axis(
                xyz_array=test_vecs, rot_amount=1.0, rot_axis=np.array([2])
            ),
        )
    )

    # Testing a special case, where the xyz_array vectors are reshaped if there
    # is only a single rotation matrix used (helps speed things up significantly)
    mod_vec = x_vecs.T.reshape((2, 3, 1))
    assert np.all(
        uvutils._rotate_one_axis(xyz_array=mod_vec, rot_amount=1.0, rot_axis=0)
        == mod_vec
    )


def test_rotate_two_axis(vector_list):
    """
    Tests some basic vector rotation operations with a double axis rotation.
    """
    x_vecs, y_vecs, z_vecs, test_vecs = vector_list

    # These tests are used to verify the basic functionality of the primary
    # functions used to two-axis rotations
    assert np.allclose(
        x_vecs,
        uvutils._rotate_two_axis(
            xyz_array=x_vecs,
            rot_amount1=2 * np.pi,
            rot_amount2=1.0,
            rot_axis1=1,
            rot_axis2=0,
        ),
    )
    assert np.allclose(
        y_vecs,
        uvutils._rotate_two_axis(
            xyz_array=y_vecs,
            rot_amount1=2 * np.pi,
            rot_amount2=2.0,
            rot_axis1=2,
            rot_axis2=1,
        ),
    )
    assert np.allclose(
        z_vecs,
        uvutils._rotate_two_axis(
            xyz_array=z_vecs,
            rot_amount1=2 * np.pi,
            rot_amount2=3.0,
            rot_axis1=0,
            rot_axis2=2,
        ),
    )

    # Do one more test, which verifies that we can filp our (1,1,1) test vector to
    # the postiion at (-1, -1 , -1)
    mod_vec = test_vecs.T.reshape((2, 3, 1))
    assert np.allclose(
        uvutils._rotate_two_axis(
            xyz_array=mod_vec,
            rot_amount1=np.pi,
            rot_amount2=np.pi / 2.0,
            rot_axis1=0,
            rot_axis2=1,
        ),
        -mod_vec,
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
            uvutils._rotate_one_axis(
                xyz_array=test_vecs, rot_amount=rot1, rot_axis=axis1
            ),
            uvutils._rotate_two_axis(
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
    for key in arg_dict.keys():
        calc_uvw_args[key] = arg_dict[key]

    with pytest.raises(err[0], match=err[1]):
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

    np.testing.assert_allclose(uvw_ant_check, calc_uvw_args["uvw_array"])
    np.testing.assert_allclose(uvw_base_check, calc_uvw_args["uvw_array"])


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

    np.testing.assert_allclose(uvw_ant_check, uvw_base_check)


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

    np.testing.assert_allclose(uvw_base_check, uvw_base_late_pa_check)


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
    with pytest.raises(ValueError, match=msg):
        uvutils.transform_icrs_to_app(
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

    with pytest.raises(ValueError, match=msg):
        uvutils.transform_app_to_icrs(
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
        uvutils.transform_sidereal_coords(
            longitude=[0.0],
            latitude=[0.0, 1.0],
            in_coord_frame="fk5",
            out_coord_frame="icrs",
            in_coord_epoch="J2000.0",
            time_array=[0.0, 1.0, 2.0],
        )

    with pytest.raises(ValueError, match="Shape of time_array must be either that of "):
        uvutils.transform_sidereal_coords(
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

    for key in arg_dict.keys():
        default_args[key] = arg_dict[key]

    # We have to handle this piece a bit carefully, since some queries fail due to
    # intermittent failures connecting to the JPL-Horizons service.
    with pytest.raises(Exception) as cm:
        uvutils.lookup_jplhorizons(
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


@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
def test_lookup_jplhorizons_moon_err():
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
        "telescope_loc": MoonLocation.from_selenodetic(0.6875, 24.433, 0),
        "high_cadence": False,
        "force_lookup": None,
    }

    # We have to handle this piece a bit carefully, since some queries fail due to
    # intermittent failures connecting to the JPL-Horizons service.
    with pytest.raises(Exception) as cm:
        uvutils.lookup_jplhorizons(
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
        uvutils.interpolate_ephem(
            time_array=0.0,
            ephem_times=0.0 if ("etimes" == bad_arg) else [0.0, 1.0],
            ephem_ra=0.0 if ("ra" == bad_arg) else [0.0, 1.0],
            ephem_dec=0.0 if ("dec" == bad_arg) else [0.0, 1.0],
            ephem_dist=0.0 if ("dist" == bad_arg) else [0.0, 1.0],
            ephem_vel=0.0 if ("vel" == bad_arg) else [0.0, 1.0],
        )


def test_calc_app_coords_arg_errs():
    """
    Check for argument errors with calc_app_coords
    """
    # Now on to app_coords
    with pytest.raises(ValueError, match="Object type whoknows is not recognized."):
        uvutils.calc_app_coords(
            lon_coord=0.0, lat_coord=0.0, telescope_loc=(0, 1, 2), coord_type="whoknows"
        )


def test_transform_multi_sidereal_coords(astrometry_args):
    """
    Perform some basic tests to verify that we can transform between sidereal frames
    with multiple coordinates.
    """
    # Check and make sure that we can deal with non-singleton times or coords with
    # singleton coords and times, respectively.
    check_ra, check_dec = uvutils.transform_sidereal_coords(
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
    fk5_ra, fk5_dec = uvutils.transform_sidereal_coords(
        longitude=astrometry_args["icrs_ra"],
        latitude=astrometry_args["icrs_dec"],
        in_coord_frame="icrs",
        out_coord_frame="fk5",
        in_coord_epoch=2000.0,
        out_coord_epoch=2000.0,
        time_array=astrometry_args["time_array"][0],
    )

    fk4_ra, fk4_dec = uvutils.transform_sidereal_coords(
        longitude=fk5_ra,
        latitude=fk5_dec,
        in_coord_frame="fk5",
        out_coord_frame="fk4",
        in_coord_epoch="J2000.0",
        out_coord_epoch="B1950.0",
    )

    check_ra, check_dec = uvutils.transform_sidereal_coords(
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
            app_ra, app_dec = uvutils.transform_icrs_to_app(
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

    app_ra, app_dec = uvutils.transform_icrs_to_app(
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
            check_ra, check_dec = uvutils.transform_app_to_icrs(
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
            check_ra, check_dec = uvutils.transform_app_to_icrs(
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
        check_ra, check_dec = uvutils.transform_app_to_icrs(
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
        app_ra, app_dec = uvutils.transform_icrs_to_app(
            time_array=astrometry_args["time_array"],
            ra=astrometry_args["icrs_ra"],
            dec=astrometry_args["icrs_dec"],
            telescope_loc=telescope_loc,
            epoch=astrometry_args["epoch"],
            astrometry_library=in_lib,
            telescope_frame=telescope_frame,
        )
        check_ra, check_dec = uvutils.transform_app_to_icrs(
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
    meas_vals = uvutils.calc_parallactic_angle(
        app_ra=[0.0, 1.0, 2.0],
        app_dec=[-1.0, 0.0, 1.0],
        lst_array=[2.0, 1.0, 0],
        telescope_lat=1.0,
    )
    # Make sure things agree to better than ~0.1 uas (as it definitely should)
    np.testing.assert_allclose(expected_vals, meas_vals, 0.0, 1e-12)


def test_calc_frame_pos_angle():
    """
    Verify that we recover frame position angles correctly
    """
    # First test -- plug in "topo" for the frame, which should always produce an
    # array of all zeros (the topo frame is what the apparent coords are in)
    frame_pa = uvutils.calc_frame_pos_angle(
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
    frame_pa = uvutils.calc_frame_pos_angle(
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
    frame_pa = uvutils.calc_frame_pos_angle(
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
    assert np.isclose(frame_pa[25], 0.001909957544309159)
    assert np.isclose(frame_pa[-25], -0.0019098101664715339)


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
            uvutils.lookup_jplhorizons("Sun", 2456789.0)
        )
    except (SSLError, RequestException) as err:
        pytest.skip("SSL/Connection error w/ JPL Horizons: " + str(err))

    assert np.all(np.equal(ephem_times, 2456789.0))
    np.testing.assert_allclose(ephem_ra, 0.8393066751804976)
    np.testing.assert_allclose(ephem_dec, 0.3120687480116649)
    np.testing.assert_allclose(ephem_dist, 1.00996185750717)
    np.testing.assert_allclose(ephem_vel, 0.386914)

    # check calling lookup_jplhorizons with EarthLocation vs lat/lon/alt passed
    try:
        ephem_info_latlon = uvutils.lookup_jplhorizons(
            "Sun", 2456789.0, telescope_loc=astrometry_args["telescope_loc"]
        )
        ephem_info_el = uvutils.lookup_jplhorizons(
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

    ra_vals0, dec_vals0, dist_vals0, vel_vals0 = uvutils.interpolate_ephem(
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

    ra_vals1, dec_vals1, dist_vals1, vel_vals1 = uvutils.interpolate_ephem(
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

    ra_vals2, dec_vals2, dist_vals2, vel_vals2 = uvutils.interpolate_ephem(
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
        check_ra, check_dec = uvutils.calc_app_coords(
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
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]

    if frame == "fk5":
        ephem_ra = astrometry_args["fk5_ra"]
        ephem_dec = astrometry_args["fk5_dec"]
    else:
        ephem_ra = np.array([astrometry_args["icrs_ra"]])
        ephem_dec = np.array([astrometry_args["icrs_dec"]])

    ephem_times = np.array([astrometry_args["time_array"][0]])
    check_ra, check_dec = uvutils.calc_app_coords(
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

    check_ra, check_dec = uvutils.calc_app_coords(
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

    check_ra, check_dec = uvutils.calc_app_coords(
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

    app_ra, app_dec = uvutils.calc_app_coords(
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

    check_ra, check_dec = uvutils.calc_sidereal_coords(
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

        app_ra, app_dec = uvutils.calc_app_coords(
            lon_coord=0.0,
            lat_coord=0.0,
            coord_type="sidereal",
            telescope_loc=telescope_loc,
            telescope_frame=telescope_frame,
            time_array=astrometry_args["time_array"],
            coord_frame="fk5",
            coord_epoch="J2000.0",
        )

        check_ra, check_dec = uvutils.calc_sidereal_coords(
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

    app_ra, app_dec = uvutils.calc_app_coords(
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

    check_ra, check_dec = uvutils.calc_sidereal_coords(
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
        coord_results[idx] = uvutils.transform_icrs_to_app(
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
        coord_results[idx] = uvutils.transform_app_to_icrs(
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

    gcrs_ra, gcrs_dec = uvutils.transform_sidereal_coords(
        longitude=astrometry_args["icrs_ra"] * np.ones(2),
        latitude=astrometry_args["icrs_dec"] * np.ones(2),
        in_coord_frame="icrs",
        out_coord_frame="gcrs",
        time_array=Time(astrometry_args["time_array"][0], format="jd"),
    )

    check_ra, check_dec = uvutils.transform_sidereal_coords(
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
    else:
        telescope_loc = astrometry_args["moon_telescope_loc"]

    check_ra, check_dec = uvutils.transform_icrs_to_app(
        time_array=Time(astrometry_args["time_array"], format="jd"),
        ra=astrometry_args["icrs_ra"],
        dec=astrometry_args["icrs_dec"],
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
        epoch=Time(astrometry_args["epoch"], format="jyear"),
    )

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

    icrs_ra, icrs_dec = uvutils.transform_app_to_icrs(
        time_array=astrometry_args["time_array"][0],
        app_ra=astrometry_args["app_ra"][0],
        app_dec=astrometry_args["app_dec"][0],
        telescope_loc=astrometry_args["telescope_loc"],
    )

    check_ra, check_dec = uvutils.transform_app_to_icrs(
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
        telescope_loc = MoonLocation.from_selenodetic(
            astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
            astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
            height=astrometry_args["telescope_loc"][2],
            ellipsoid=selenoid,
        )
        TimeClass = LTime

    app_ra, app_dec = uvutils.calc_app_coords(
        lon_coord=astrometry_args["icrs_ra"],
        lat_coord=astrometry_args["icrs_dec"],
        time_array=astrometry_args["time_array"][0],
        telescope_loc=astrometry_args["telescope_loc"],
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
    )

    check_ra, check_dec = uvutils.calc_app_coords(
        lon_coord=astrometry_args["icrs_ra"],
        lat_coord=astrometry_args["icrs_dec"],
        time_array=TimeClass(astrometry_args["time_array"][0], format="jd"),
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=selenoid,
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
            jd_array=astrometry_args["time_array"],
            latitude=astrometry_args["telescope_loc"][0],
            longitude=astrometry_args["telescope_loc"][1],
            altitude=astrometry_args["telescope_loc"][2],
            astrometry_library=name,
        )

    for idx in range(len(lst_results) - 1):
        for jdx in range(idx + 1, len(lst_results)):
            alpha_time = lst_results[idx] * units.rad
            beta_time = lst_results[jdx] * units.rad
            assert np.all(np.abs(alpha_time - beta_time).to_value("mas") < 1.0)


@pytest.mark.parametrize("astrometry_lib", ["astropy", "novas", "erfa"])
def test_lst_for_time_smooth(astrometry_lib):
    """
    Test that LSTs are smooth and do not have large discontinuities.

    Inspired by a bug found by the HERA validation team in our original implemenatation
    using the erfa library.
    """
    if astrometry_lib == "novas":
        pytest.importorskip("novas")
        pytest.importorskip("novas_de405")

    hera_loc = EarthLocation.from_geodetic(
        lat=-30.72152612068957, lon=21.428303826863015, height=1051.6900000218302
    )

    start_time = 2458101.5435486115
    n_times = 28728
    integration_time = 1.0

    daysperhour = 1 / 24.0
    hourspersec = 1 / 60.0**2
    dayspersec = daysperhour * hourspersec
    inttime_days = integration_time * dayspersec
    duration = inttime_days * n_times
    end_time = start_time + duration - inttime_days
    times = np.linspace(start_time, end_time + inttime_days, n_times, endpoint=False)

    uv_lsts = uvutils.get_lst_for_time(
        times,
        latitude=hera_loc.lat.deg,
        longitude=hera_loc.lon.deg,
        altitude=hera_loc.height.value,
        astrometry_library=astrometry_lib,
        frame="itrs",
    )

    dtimes = times - int(times[0])
    poly_fit = np.poly1d(np.polyfit(dtimes, uv_lsts, 2))
    diff_poly = uv_lsts - poly_fit(dtimes)
    assert np.max(np.abs(diff_poly)) < 1e-10


@pytest.mark.parametrize("astrolib", ["novas", "astropy", "erfa"])
def test_lst_for_time_float_vs_array(astrometry_args, astrolib):
    """
    Test for equality when passing a single float vs an ndarray (of length 1) when
    calling get_lst_for_time.
    """
    if astrolib == "novas":
        pytest.importorskip("novas")
        pytest.importorskip("novas_de405")

    r2d = 180.0 / np.pi

    lst_array = uvutils.get_lst_for_time(
        jd_array=np.array(astrometry_args["time_array"][0]),
        latitude=astrometry_args["telescope_loc"][0] * r2d,
        longitude=astrometry_args["telescope_loc"][1] * r2d,
        altitude=astrometry_args["telescope_loc"][2],
        astrometry_library=astrolib,
    )

    check_lst = uvutils.get_lst_for_time(
        jd_array=astrometry_args["time_array"][0],
        telescope_loc=np.multiply(astrometry_args["telescope_loc"], [r2d, r2d, 1]),
        astrometry_library=astrolib,
    )

    assert np.all(lst_array == check_lst)


def test_get_lst_for_time_errors(astrometry_args):
    with pytest.raises(
        ValueError,
        match="Requested coordinate transformation library is not supported, please "
        "select either 'erfa' or 'astropy' for astrometry_library.",
    ):
        uvutils.get_lst_for_time(
            jd_array=np.array(astrometry_args["time_array"][0]),
            latitude=astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
            longitude=astrometry_args["telescope_loc"][1] * (180.0 / np.pi),
            altitude=astrometry_args["telescope_loc"][2],
            astrometry_library="foo",
        )

    with pytest.raises(
        ValueError,
        match="Cannot set both telescope_loc and latitude/longitude/altitude",
    ):
        uvutils.get_lst_for_time(
            np.array(astrometry_args["time_array"][0]),
            latitude=astrometry_args["telescope_loc"][0] * (180.0 / np.pi),
            telescope_loc=astrometry_args["telescope_loc"][2],
        )


@pytest.mark.filterwarnings("ignore:The get_frame_attr_names")
@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
@pytest.mark.parametrize("selenoid", selenoids)
def test_lst_for_time_moon(astrometry_args, selenoid):
    """Test the get_lst_for_time function with MCMF frame"""
    from lunarsky import SkyCoord as LSkyCoord

    lat, lon, alt = (0.6875, 24.433, 0)  # Degrees

    # check error if try to use the wrong astrometry library
    with pytest.raises(
        NotImplementedError,
        match="The MCMF frame is only supported with the 'astropy' astrometry library",
    ):
        lst_array = uvutils.get_lst_for_time(
            jd_array=astrometry_args["time_array"],
            latitude=lat,
            longitude=lon,
            altitude=alt,
            frame="mcmf",
            ellipsoid=selenoid,
            astrometry_library="novas",
        )

    lst_array = uvutils.get_lst_for_time(
        jd_array=astrometry_args["time_array"],
        latitude=lat,
        longitude=lon,
        altitude=alt,
        frame="mcmf",
        ellipsoid=selenoid,
    )

    # Verify that lsts are close to local zenith RA
    loc = MoonLocation.from_selenodetic(lon, lat, alt, ellipsoid=selenoid)
    for ii, tt in enumerate(
        LTime(astrometry_args["time_array"], format="jd", scale="utc", location=loc)
    ):
        src = LSkyCoord(alt="90d", az="0d", frame="lunartopo", obstime=tt, location=loc)
        # TODO: would be nice to get this down to uvutils.RADIAN_TOL
        # seems like maybe the ellipsoid isn't being used properly?
        assert np.isclose(lst_array[ii], src.transform_to("icrs").ra.rad, atol=1e-5)

    # test default ellipsoid
    if selenoid == "SPHERE":
        lst_array_default = uvutils.get_lst_for_time(
            jd_array=astrometry_args["time_array"],
            latitude=lat,
            longitude=lon,
            altitude=alt,
            frame="mcmf",
        )
        np.testing.assert_allclose(lst_array, lst_array_default)


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

    gcrs_uvw = uvutils.old_uvw_calc(
        gcrs_coord.ra.rad, gcrs_coord.dec.rad, gcrs_rel.value
    )

    mwa_tools_calcuvw_u = -97.122828
    mwa_tools_calcuvw_v = 50.388281
    mwa_tools_calcuvw_w = -151.27976

    np.testing.assert_allclose(gcrs_uvw[0, 0], mwa_tools_calcuvw_u, atol=1e-3)
    np.testing.assert_allclose(gcrs_uvw[0, 1], mwa_tools_calcuvw_v, atol=1e-3)
    np.testing.assert_allclose(gcrs_uvw[0, 2], mwa_tools_calcuvw_w, atol=1e-3)

    # also test unphasing
    temp2 = uvutils.undo_old_uvw_calc(
        gcrs_coord.ra.rad, gcrs_coord.dec.rad, np.squeeze(gcrs_uvw)
    )
    np.testing.assert_allclose(gcrs_rel.value, np.squeeze(temp2))


def test_pol_funcs():
    """Test utility functions to convert between polarization strings and numbers"""

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
    with check_warnings(UserWarning, "x_orientation not recognized"):
        assert uvutils.polstr2num("xx", x_orientation="foo") == -5

    with check_warnings(UserWarning, "x_orientation not recognized"):
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
    with check_warnings(UserWarning, "x_orientation not recognized"):
        assert uvutils.jstr2num("x", x_orientation="foo") == -5

    with check_warnings(UserWarning, "x_orientation not recognized"):
        assert uvutils.jnum2str(-6, x_orientation="foo") == "Jyy"


def test_conj_pol():
    """Test function to conjugate pols"""

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
    with pytest.raises(
        ValueError, match="Polarization not recognized, cannot be conjugated."
    ):
        uvutils.conj_pol(2.3)


@pytest.mark.parametrize("grid_alg", [True, False, None])
def test_redundancy_finder(grid_alg):
    """
    Check that get_baseline_redundancies and get_antenna_redundancies return consistent
    redundant groups for a test file with the HERA19 layout.
    """
    uvd = UVData()
    uvd.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    uvd.select(times=uvd.time_array[0])
    uvd.unproject_phase(use_ant_pos=True)
    # uvw_array is now equivalent to baseline positions
    uvd.conjugate_bls("ant1<ant2", use_enu=True)

    tol = 0.05  # meters

    bl_positions = uvd.uvw_array
    bl_pos_backup = copy.deepcopy(uvd.uvw_array)

    if grid_alg is None:
        warn_str = (
            "The use_grid_alg parameter is not set. Defaulting to True to "
            "use the new gridding based algorithm (developed by the HERA team) "
            "rather than the older clustering based algorithm. This is change "
            "to the default, to use the clustering algorithm set "
            "use_grid_alg=False."
        )
        warn_type = UserWarning
    else:
        warn_type = None
        warn_str = ""

    with pytest.raises(
        ValueError, match=re.escape("Baseline vectors must be shape (Nbls, 3)")
    ):
        with check_warnings(warn_type, match=warn_str):
            uvutils.get_baseline_redundancies(
                uvd.baseline_array, bl_positions[0:2, 0:1], use_grid_alg=grid_alg
            )

    with check_warnings(warn_type, match=warn_str):
        baseline_groups, vec_bin_centers, lens = uvutils.get_baseline_redundancies(
            uvd.baseline_array, bl_positions, tol=tol, use_grid_alg=grid_alg
        )

    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            np.testing.assert_allclose(
                np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])), lens[gi], atol=tol
            )

    # Shift the baselines around in a circle. Check that the same baselines are
    # recovered to the corresponding tolerance increase.
    # This moves one baseline at a time by a fixed displacement and checks that
    # the redundant groups are the same.

    hightol = 0.25  # meters. Less than the smallest baseline in the file.
    tol_use = hightol
    if grid_alg in [None, True]:
        tol_use = hightol * 4
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

            with check_warnings(warn_type, match=warn_str):
                (baseline_groups_new, vec_bin_centers, lens) = (
                    uvutils.get_baseline_redundancies(
                        uvd.baseline_array,
                        bl_positions_new,
                        tol=tol_use,
                        use_grid_alg=grid_alg,
                    )
                )

            for gi, gp in enumerate(baseline_groups_new):
                for bl in gp:
                    bl_ind = np.where(uvd.baseline_array == bl)
                    bl_vec = bl_positions[bl_ind]
                    np.testing.assert_allclose(
                        np.sqrt(np.abs(np.dot(bl_vec, vec_bin_centers[gi]))),
                        lens[gi],
                        atol=tol_use,
                    )

            # Compare baseline groups:
            a = [tuple(el) for el in baseline_groups]
            b = [tuple(el) for el in baseline_groups_new]
            assert set(a) == set(b)

    tol = 0.05

    antpos = uvd.telescope.get_enu_antpos()

    with check_warnings(warn_type, match=warn_str):
        baseline_groups_ants, vec_bin_centers, lens = uvutils.get_antenna_redundancies(
            uvd.telescope.antenna_numbers,
            antpos,
            tol=tol,
            include_autos=False,
            use_grid_alg=grid_alg,
        )
    # Under these conditions, should see 19 redundant groups in the file.
    assert len(baseline_groups_ants) == 19

    # Check with conjugated baseline redundancies returned
    # Ensure at least one baseline has u==0 and v!=0 (for coverage of this case)
    bl_positions[16, 0] = 0
    with check_warnings(warn_type, match=warn_str):
        (baseline_groups, vec_bin_centers, lens, conjugates) = (
            uvutils.get_baseline_redundancies(
                uvd.baseline_array,
                bl_positions,
                tol=tol,
                include_conjugates=True,
                use_grid_alg=grid_alg,
            )
        )

    # restore baseline (16,0) and repeat to get correct groups
    bl_positions = bl_pos_backup
    with check_warnings(warn_type, match=warn_str):
        (baseline_groups, vec_bin_centers, lens, conjugates) = (
            uvutils.get_baseline_redundancies(
                uvd.baseline_array,
                bl_positions,
                tol=tol,
                include_conjugates=True,
                use_grid_alg=grid_alg,
            )
        )

    # Apply flips to compare with get_antenna_redundancies().
    bl_gps_unconj = copy.deepcopy(baseline_groups)
    for gi, gp in enumerate(bl_gps_unconj):
        for bi, bl in enumerate(gp):
            if bl in conjugates:
                bl_gps_unconj[gi][bi] = uvutils.baseline_index_flip(
                    bl, Nants_telescope=uvd.telescope.Nants
                )
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
    such that baselines end up in multiple groups
    """
    uvd = UVData()
    uvd.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    uvd.select(times=uvd.time_array[0])
    uvd.unproject_phase(use_ant_pos=True)
    # uvw_array is now equivalent to baseline positions
    uvd.conjugate_bls("ant1<ant2", use_enu=True)
    bl_positions = uvd.uvw_array

    tol = 20.05  # meters

    with pytest.raises(ValueError, match="Some baselines are falling into"):
        uvutils.get_baseline_redundancies(
            uvd.baseline_array,
            bl_positions,
            tol=tol,
            include_conjugates=True,
            use_grid_alg=False,
        )


@pytest.mark.parametrize("grid_alg", [True, False])
def test_redundancy_conjugates(grid_alg):
    """
    Check that redundancy finding with conjugation works.

    Check that the correct baselines are flipped.
    """
    Nants = 10
    tol = 0.5
    ant1_arr = np.tile(np.arange(Nants), Nants)
    ant2_arr = np.repeat(np.arange(Nants), Nants)
    Nbls = ant1_arr.size
    bl_inds = uvutils.antnums_to_baseline(ant1_arr, ant2_arr, Nants_telescope=Nants)

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
    _, _, _, conjugates = uvutils.get_baseline_redundancies(
        bl_inds, bl_vecs, tol=tol, include_conjugates=True, use_grid_alg=grid_alg
    )

    assert sorted(conjugates) == sorted(expected_conjugates)


@pytest.mark.parametrize("grid_alg", [True, False])
def test_redundancy_finder_fully_redundant_array(grid_alg):
    """Test the redundancy finder for a fully redundant array."""
    uvd = UVData()
    uvd.read_uvfits(os.path.join(DATA_PATH, "test_redundant_array.uvfits"))
    uvd.select(times=uvd.time_array[0])

    tol = 1  # meters
    bl_positions = uvd.uvw_array

    baseline_groups, _, _, _ = uvutils.get_baseline_redundancies(
        uvd.baseline_array,
        bl_positions,
        tol=tol,
        include_conjugates=True,
        use_grid_alg=grid_alg,
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
        uvutils._find_cliques(adj_link, strict=True)


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
    np.testing.assert_allclose(out * wo, data.shape[0])
    np.testing.assert_allclose(
        wo, float(data.shape[0]) / (np.arange(data.shape[1]) + 1)
    )
    np.testing.assert_allclose(
        wso, float(data.shape[0]) / (np.arange(data.shape[1]) + 1) ** 2
    )
    out, wo, wso = uvutils.mean_collapse(
        data, weights=w, axis=1, return_weights=True, return_weights_square=True
    )
    np.testing.assert_allclose(out * wo, data.shape[1])
    np.testing.assert_allclose(wo, np.sum(1.0 / (np.arange(data.shape[1]) + 1)))
    np.testing.assert_allclose(wso, np.sum(1.0 / (np.arange(data.shape[1]) + 1) ** 2))

    # Zero weights
    w = np.ones_like(data)
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
    with check_warnings(UserWarning, "Currently weights are"):
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
    with check_warnings(UserWarning, "Currently weights are"):
        o = uvutils.and_collapse(data, axis=0, weights=w)
    assert np.array_equal(o, ans)


def test_and_collapse_errors():
    data = np.zeros(5)
    pytest.raises(ValueError, uvutils.and_collapse, data)


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
def test_uvcalibrate_apply_gains_oldfiles(utils_uvdata):
    # read data
    uvd = utils_uvdata

    # give it an x_orientation
    uvd.telescope.x_orientation = "east"
    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.gain.calfits"))
    # downselect to match each other in shape (but not in actual values!)
    uvd.select(frequencies=uvd.freq_array[:10])
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

    freq_expected = f"Frequency {uvd.freq_array[0]} exists on UVData but not on UVCal."

    with check_warnings(UserWarning, match=ants_expected):
        with pytest.raises(ValueError, match=time_expected):
            uvutils.uvcalibrate(
                uvd, uvc, prop_flags=True, ant_check=False, inplace=False
            )

    uvc.select(times=uvc.time_array[0])

    time_expected = [
        "Times do not match between UVData and UVCal but time_check is False, so "
        "calibration will be applied anyway."
    ]

    with check_warnings(UserWarning, match=ants_expected + time_expected):
        with pytest.raises(ValueError, match=freq_expected):
            uvutils.uvcalibrate(
                uvd, uvc, prop_flags=True, ant_check=False, time_check=False
            )


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
def test_uvcalibrate_delay_oldfiles(utils_uvdata):
    uvd = utils_uvdata

    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits"))
    # downselect to match
    uvc.select(times=uvc.time_array[3])
    uvc.gain_convention = "multiply"

    freq_array_use = np.squeeze(uvd.freq_array)
    chan_with_use = uvd.channel_width

    ant_expected = [
        "The uvw_array does not match the expected values",
        "All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
        "Times do not match between UVData and UVCal but time_check is False, so "
        "calibration will be applied anyway.",
        r"UVData object does not have `x_orientation` specified but UVCal does",
    ]
    with check_warnings(UserWarning, match=ant_expected):
        uvdcal = uvutils.uvcalibrate(
            uvd, uvc, prop_flags=False, ant_check=False, time_check=False, inplace=False
        )

    uvc.convert_to_gain(freq_array=freq_array_use, channel_width=chan_with_use)
    with check_warnings(UserWarning, match=ant_expected):
        uvdcal2 = uvutils.uvcalibrate(
            uvd, uvc, prop_flags=False, ant_check=False, time_check=False, inplace=False
        )

    assert uvdcal == uvdcal2


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.parametrize("flip_gain_conj", [False, True])
@pytest.mark.parametrize("gain_convention", ["divide", "multiply"])
@pytest.mark.parametrize("time_range", [None, "Ntimes", 3])
def test_uvcalibrate(uvcalibrate_data, flip_gain_conj, gain_convention, time_range):
    uvd, uvc = uvcalibrate_data

    if time_range is not None:
        tstarts = uvc.time_array - uvc.integration_time / (86400 * 2)
        tends = uvc.time_array + uvc.integration_time / (86400 * 2)
        if time_range == "Ntimes":
            uvc.time_range = np.stack((tstarts, tends), axis=1)
        else:
            nt_per_range = int(np.ceil(uvc.Ntimes / time_range))
            tstart_inds = np.array(np.arange(time_range) * nt_per_range)
            tstarts_use = tstarts[tstart_inds]
            tend_inds = np.array((np.arange(time_range) + 1) * nt_per_range - 1)
            tend_inds[-1] = -1
            tends_use = tends[tend_inds]
            uvc.select(times=uvc.time_array[0:time_range])
            uvc.time_range = np.stack((tstarts_use, tends_use), axis=1)
        uvc.time_array = None
        uvc.lst_array = None
        uvc.set_lsts_from_time_array()

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

    if time_range is not None and time_range != "Ntimes":
        gain_product = gain_product[:, np.newaxis]
        gain_product = np.repeat(gain_product, nt_per_range, axis=1)
        current_shape = gain_product.shape
        new_shape = (current_shape[0] * current_shape[1], current_shape[-1])
        gain_product = gain_product.reshape(new_shape)
        gain_product = gain_product[: uvd.Ntimes]

    if gain_convention == "divide":
        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key), uvd.get_data(key) / gain_product
        )
    else:
        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key), uvd.get_data(key) * gain_product
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


@pytest.mark.filterwarnings("ignore:Combined frequencies are separated by more than")
def test_uvcalibrate_dterm_handling(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # test d-term exception
    with pytest.raises(
        ValueError, match="Cannot apply D-term calibration without -7 or -8"
    ):
        uvutils.uvcalibrate(uvd, uvc, Dterm_cal=True)

    # d-term not implemented error
    uvcDterm = uvc.copy()
    uvcDterm.jones_array = np.array([-7, -8])
    uvcDterm = uvc + uvcDterm
    with pytest.raises(
        NotImplementedError, match="D-term calibration is not yet implemented."
    ):
        uvutils.uvcalibrate(uvd, uvcDterm, Dterm_cal=True)


@pytest.mark.filterwarnings("ignore:Changing number of antennas, but preserving")
def test_uvcalibrate_flag_propagation(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

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

    uvc_sub = uvc.select(antenna_nums=[1, 12], inplace=False)

    uvdata_unique_nums = np.unique(np.append(uvd.ant_1_array, uvd.ant_2_array))
    uvd.telescope.antenna_names = np.array(uvd.telescope.antenna_names)
    missing_ants = uvdata_unique_nums.tolist()
    missing_ants.remove(1)
    missing_ants.remove(12)
    missing_ant_names = [
        uvd.telescope.antenna_names[
            np.where(uvd.telescope.antenna_numbers == antnum)[0][0]
        ]
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

    with check_warnings(
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


def test_uvcalibrate_antenna_names_mismatch(uvcalibrate_init_data):
    uvd, uvc = uvcalibrate_init_data

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
    with check_warnings(
        UserWarning,
        match="All antenna names with data on UVData are missing "
        "on UVCal. Since ant_check is False, calibration will "
        "proceed but all data will be flagged.",
    ):
        uvdcal = uvutils.uvcalibrate(uvd, uvc, ant_check=False, inplace=False)

    assert np.all(uvdcal.flag_array)  # assert completely flagged


@pytest.mark.parametrize("time_range", [True, False])
def test_uvcalibrate_time_mismatch(uvcalibrate_data, time_range):
    uvd, uvc = uvcalibrate_data

    if time_range:
        tstarts = uvc.time_array - uvc.integration_time / (86400 * 2)
        tends = uvc.time_array + uvc.integration_time / (86400 * 2)
        original_time_range = np.stack((tstarts, tends), axis=1)
        uvc.time_range = original_time_range
        uvc.time_array = None
        uvc.lst_array = None
        uvc.set_lsts_from_time_array()

    # change times to get warnings
    if time_range:
        uvc.time_range = uvc.time_range + 1
        uvc.set_lsts_from_time_array()
        expected_err = "Time_ranges on UVCal do not cover all UVData times."
        with pytest.raises(ValueError, match=expected_err):
            uvutils.uvcalibrate(uvd, uvc, inplace=False)
    else:
        uvc.time_array = uvc.time_array + 1
        uvc.set_lsts_from_time_array()
        expected_err = {
            f"Time {this_time} exists on UVData but not on UVCal."
            for this_time in np.unique(uvd.time_array)
        }

        with pytest.raises(ValueError) as errinfo:
            uvutils.uvcalibrate(uvd, uvc, inplace=False)
        assert str(errinfo.value) in expected_err

    # for time_range, make the time ranges not cover some UVData times
    if time_range:
        uvc.time_range = original_time_range
        uvc.time_range[0, 1] = uvc.time_range[0, 0] + uvc.integration_time[0] / (
            86400 * 4
        )
        uvc.set_lsts_from_time_array()
        with pytest.raises(ValueError, match=expected_err):
            uvutils.uvcalibrate(uvd, uvc, inplace=False)

        uvc.phase_center_id_array = np.arange(uvc.Ntimes)
        uvc.phase_center_catalog = {0: None}
        uvc.select(phase_center_ids=0)
        with check_warnings(
            UserWarning, match="Time_range on UVCal does not cover all UVData times"
        ):
            _ = uvutils.uvcalibrate(uvd, uvc, inplace=False, time_check=False)


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


@pytest.mark.filterwarnings("ignore:The time_array and time_range attributes")
@pytest.mark.filterwarnings("ignore:The lst_array and lst_range attributes")
@pytest.mark.parametrize("time_range", [True, False])
def test_uvcalibrate_single_time_types(uvcalibrate_data, time_range):
    uvd, uvc = uvcalibrate_data

    # only one time
    uvc.select(times=uvc.time_array[0])
    if time_range:
        # check cal runs fine with a good time range
        uvc.time_range = np.reshape(
            np.array([np.min(uvd.time_array), np.max(uvd.time_array)]), (1, 2)
        )
        uvc.set_lsts_from_time_array()
        with pytest.raises(
            ValueError, match="The time_array and time_range attributes are both set"
        ):
            uvdcal = uvutils.uvcalibrate(uvd, uvc, inplace=False, time_check=False)
        uvc.time_array = uvc.lst_array = None
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
        uvc.set_lsts_from_time_array()

    if time_range:
        msg_start = "Time_range on UVCal does not cover all UVData times"
    else:
        msg_start = "Times do not match between UVData and UVCal"
    err_msg = msg_start + ". Set time_check=False to apply calibration anyway."
    warn_msg = [
        msg_start + " but time_check is False, so calibration will be applied anyway."
    ]

    with pytest.raises(ValueError, match=err_msg):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)

    if not time_range:
        with check_warnings(UserWarning, match=warn_msg):
            uvdcal = uvutils.uvcalibrate(uvd, uvc, inplace=False, time_check=False)

        key = (1, 13, "xx")
        ant1 = (1, "Jxx")
        ant2 = (13, "Jxx")

        np.testing.assert_array_almost_equal(
            uvdcal.get_data(key),
            uvd.get_data(key) / (uvc.get_gains(ant1) * uvc.get_gains(ant2).conj()).T,
        )


@pytest.mark.filterwarnings("ignore:Combined frequencies are separated by more than")
def test_uvcalibrate_extra_cal_times(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc2 = uvc.copy()
    uvc2.time_array = uvc.time_array + 1
    uvc2.set_lsts_from_time_array()
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
    uvc.freq_array[uvc.Nfreqs // 2 :] = uvc.freq_array[uvc.Nfreqs // 2 :] + maxf
    expected_err = {
        f"Frequency {this_freq} exists on UVData but not on UVCal."
        for this_freq in uvd.freq_array[uvd.Nfreqs // 2 :]
    }
    # structured this way rather than using the match parameter because expected_err
    # is a set.
    with pytest.raises(ValueError) as errinfo:
        uvutils.uvcalibrate(uvd, uvc, inplace=False)
    assert str(errinfo.value) in expected_err


@pytest.mark.filterwarnings("ignore:Combined frequencies are not evenly spaced.")
@pytest.mark.filterwarnings("ignore:Selected frequencies are not contiguous.")
def test_uvcalibrate_extra_cal_freqs(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc2 = uvc.copy()
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
    uvc.select(jones=uvutils.jstr2num("Jnn", x_orientation=uvc.telescope.x_orientation))
    with pytest.raises(
        ValueError, match=("Feed polarization e exists on UVData but not on UVCal.")
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)


def test_uvcalibrate_x_orientation_mismatch(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    # next check None uvd_x
    uvd.telescope.x_orientation = None
    uvc.telescope.x_orientation = "east"
    with pytest.warns(
        UserWarning,
        match=r"UVData object does not have `x_orientation` specified but UVCal does",
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)


def test_uvcalibrate_wideband_gain(uvcalibrate_data):
    uvd, uvc = uvcalibrate_data

    uvc.flex_spw_id_array = None
    uvc._set_wide_band()
    uvc.spw_array = np.array([1, 2, 3])
    uvc.Nspws = 3
    uvc.gain_array = uvc.gain_array[:, 0:3, :, :]
    uvc.flag_array = uvc.flag_array[:, 0:3, :, :]
    uvc.quality_array = uvc.quality_array[:, 0:3, :, :]
    uvc.total_quality_array = uvc.total_quality_array[0:3, :, :]

    uvc.freq_range = np.zeros((uvc.Nspws, 2), dtype=uvc.freq_array.dtype)
    uvc.freq_range[0, :] = uvc.freq_array[[0, 2]]
    uvc.freq_range[1, :] = uvc.freq_array[[2, 4]]
    uvc.freq_range[2, :] = uvc.freq_array[[4, 6]]

    uvc.channel_width = None
    uvc.freq_array = None
    uvc.Nfreqs = 1

    uvc.check()
    with pytest.raises(
        ValueError,
        match="uvcalibrate currently does not support wide-band calibrations",
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Nfreqs will be required to be 1 for wide_band cals")
@pytest.mark.filterwarnings("ignore:telescope_location, antenna_positions")
def test_uvcalibrate_delay_multispw(utils_uvdata):
    uvd = utils_uvdata

    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, "zen.2457698.40355.xx.delay.calfits"))
    # downselect to match
    uvc.select(times=uvc.time_array[3])
    uvc.gain_convention = "multiply"

    uvc.Nspws = 3
    uvc.spw_array = np.array([1, 2, 3])

    # copy the delay array to the second SPW
    uvc.delay_array = np.repeat(uvc.delay_array, uvc.Nspws, axis=1)
    uvc.flag_array = np.repeat(uvc.flag_array, uvc.Nspws, axis=1)
    uvc.quality_array = np.repeat(uvc.quality_array, uvc.Nspws, axis=1)

    uvc.freq_range = np.repeat(uvc.freq_range, uvc.Nspws, axis=0)
    # Make the second & third SPWs be contiguous with a 10 MHz range
    uvc.freq_range[1, 0] = uvc.freq_range[0, 1]
    uvc.freq_range[1, 1] = uvc.freq_range[1, 0] + 10e6
    uvc.freq_range[2, 0] = uvc.freq_range[1, 1]
    uvc.freq_range[2, 1] = uvc.freq_range[1, 1] + 10e6

    uvc.check()
    with pytest.raises(
        ValueError,
        match="uvcalibrate currently does not support multi spectral window delay "
        "calibrations",
    ):
        uvutils.uvcalibrate(uvd, uvc, inplace=False)


@pytest.mark.filterwarnings("ignore:The shapes of several attributes will be changing")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_apply_uvflag(utils_uvdata):
    # load data and insert some flags
    uvd = utils_uvdata
    uvd.flag_array[uvd.antpair2ind(9, 20)] = True

    # load a UVFlag into flag type
    uvf = UVFlag(uvd)
    uvf.to_flag()

    # insert flags for 2 out of 3 times
    uvf.flag_array[uvf.antpair2ind(9, 10)[:2]] = True

    # apply flags and check for basic flag propagation
    uvdf = uvutils.apply_uvflag(uvd, uvf, inplace=False)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])

    # test inplace
    uvdf = uvd.copy()
    uvutils.apply_uvflag(uvdf, uvf, inplace=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])

    # test flag missing
    uvf2 = uvf.select(bls=uvf.get_antpairs()[:-1], inplace=False)
    uvdf = uvutils.apply_uvflag(uvd, uvf2, inplace=False, flag_missing=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(uvf.get_antpairs()[-1])])
    uvdf = uvutils.apply_uvflag(uvd, uvf2, inplace=False, flag_missing=False)
    assert not np.any(uvdf.flag_array[uvdf.antpair2ind(uvf.get_antpairs()[-1])])

    # test force polarization
    uvdf = uvd.copy()
    uvdf2 = uvd.copy()
    uvdf2.polarization_array[0] = -6
    uvdf += uvdf2
    uvdf = uvutils.apply_uvflag(uvdf, uvf, inplace=False, force_pol=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    with pytest.raises(
        ValueError, match="Input uvf and uvd polarizations do not match"
    ):
        uvutils.apply_uvflag(uvdf, uvf, inplace=False, force_pol=False)

    # test unflag first
    uvdf = uvutils.apply_uvflag(uvd, uvf, inplace=False, unflag_first=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    assert not np.any(uvdf.flag_array[uvdf.antpair2ind(9, 20)])

    # convert uvf to waterfall and test
    uvfw = uvf.copy()
    uvfw.to_waterfall(method="or")
    uvdf = uvutils.apply_uvflag(uvd, uvfw, inplace=False)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 20)][:2])
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(20, 22)][:2])

    # test mode exception
    uvfm = uvf.copy()
    uvfm.mode = "metric"
    with pytest.raises(ValueError, match="UVFlag must be flag mode"):
        uvutils.apply_uvflag(uvd, uvfm)

    # test polarization exception
    uvd2 = uvd.copy()
    uvd2.polarization_array[0] = -6
    uvf2 = UVFlag(uvd)
    uvf2.to_flag()
    uvd2.polarization_array[0] = -8
    with pytest.raises(
        ValueError, match="Input uvf and uvd polarizations do not match"
    ):
        uvutils.apply_uvflag(uvd2, uvf2, force_pol=False)

    # test time and frequency mismatch exceptions
    uvf2 = uvf.select(frequencies=uvf.freq_array[:2], inplace=False)
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched frequency arrays"
    ):
        uvutils.apply_uvflag(uvd, uvf2)

    uvf2 = uvf.copy()
    uvf2.freq_array += 1.0
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched frequency arrays"
    ):
        uvutils.apply_uvflag(uvd, uvf2)

    uvf2 = uvf.select(times=np.unique(uvf.time_array)[:2], inplace=False)
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched time arrays"
    ):
        uvutils.apply_uvflag(uvd, uvf2)

    uvf2 = uvf.copy()
    uvf2.time_array += 1.0
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched time arrays"
    ):
        uvutils.apply_uvflag(uvd, uvf2)

    # assert implicit broadcasting works
    uvf2 = uvf.select(frequencies=uvf.freq_array[:1], inplace=False)
    uvd2 = uvutils.apply_uvflag(uvd, uvf2, inplace=False)
    assert np.all(uvd2.get_flags(9, 10)[:2])
    uvf2 = uvf.select(times=np.unique(uvf.time_array)[:1], inplace=False)
    uvd2 = uvutils.apply_uvflag(uvd, uvf2, inplace=False)
    assert np.all(uvd2.get_flags(9, 10))


@pytest.mark.parametrize("grid_alg", [True, False])
def test_upos_tol_reds(grid_alg):
    # Checks that the u-positive convention in get_antenna_redundancies
    # is enforced to the specificed tolerance.

    # Make a layout with two NS baselines, one with u ~ -2*eps, and another with u == 0
    # This would previously cause one to be flipped, when they should be redundant.

    eps = 1e-5
    tol = 3 * eps
    if grid_alg:
        tol = tol * 2

    ant_pos = np.array(
        [[-eps, 1.0, 0.0], [1.0, 1.0, 0.0], [eps, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    ant_nums = np.arange(4)

    red_grps, _, _ = uvutils.get_antenna_redundancies(
        ant_nums, ant_pos, tol=tol, use_grid_alg=grid_alg
    )

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

    # Handling bool arrays
    bool_arr = np.zeros((10000,), dtype=bool)
    index_arr = np.arange(1, 10000, 2)
    bool_arr[index_arr] = True
    assert uvutils._convert_to_slices(bool_arr) == uvutils._convert_to_slices(index_arr)
    assert uvutils._convert_to_slices(bool_arr, return_index_on_fail=True) == (
        uvutils._convert_to_slices(index_arr, return_index_on_fail=True)
    )

    # Index return on fail with two slices
    index_arr[0] = 0
    bool_arr[0:2] = [True, False]

    for item in [index_arr, bool_arr]:
        result, check = uvutils._convert_to_slices(
            item, max_nslice=1, return_index_on_fail=True
        )
        assert not check
        assert len(result) == 1
        assert result[0] is item

    # Check a more complicated pattern w/ just the max_slice_frac defined
    index_arr = np.arange(0, 100) ** 2
    bool_arr[:] = False
    bool_arr[index_arr] = True

    for item in [index_arr, bool_arr]:
        result, check = uvutils._convert_to_slices(item, return_index_on_fail=True)
        assert not check
        assert len(result) == 1
        assert result[0] is item


@pytest.mark.parametrize(
    "blt_order",
    [
        ("time", "baseline"),
        ("baseline", "time"),
        ("ant1", "time"),
        ("ant2", "time"),
        ("time", "ant1"),
        ("time", "ant2"),
        ("baseline",),
        ("time",),
        ("ant1",),
        ("ant2",),
        (),
        ([0, 2, 6, 4, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11, 13, 15, 17]),
    ],
)
def test_determine_blt_order(blt_order):
    nant = 3
    ntime = 2

    def getbl(ant1, ant2):
        return uvutils.antnums_to_baseline(ant1, ant2, Nants_telescope=nant)

    def getantbls():
        # Arrange them backwards so by default they are NOT sorted
        ant1 = np.arange(nant, dtype=int)[::-1]
        ant2 = ant1.copy()
        ANT1, ANT2 = np.meshgrid(ant1, ant2)

        return ANT1.flatten(), ANT2.flatten()

    def gettimebls(blt_order):
        ant1, ant2 = getantbls()
        time_array = np.linspace(
            2000, 1000, ntime
        )  # backwards so not sorted by default

        TIME = np.tile(time_array, len(ant1))
        ANT1 = np.repeat(ant1, len(time_array))
        ANT2 = np.repeat(ant2, len(time_array))
        BASELINE = getbl(ANT1, ANT2)

        lc = locals()
        if isinstance(blt_order, list):
            inds = np.array(blt_order)
        elif blt_order:
            inds = np.lexsort(tuple(lc[k.upper()] for k in blt_order[::-1]))
        else:
            inds = np.arange(len(TIME))

        return TIME[inds], ANT1[inds], ANT2[inds], BASELINE[inds]

    # time, bl
    TIME, ANT1, ANT2, BL = gettimebls(blt_order)
    order = uvutils.determine_blt_order(
        time_array=TIME,
        ant_1_array=ANT1,
        ant_2_array=ANT2,
        baseline_array=BL,
        Nbls=nant**2,
        Ntimes=ntime,
    )
    if isinstance(blt_order, list):
        assert order is None
    elif blt_order:
        assert order == blt_order
    else:
        assert order is None

    is_rect, time_first = uvutils.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=nant**2, ntimes=ntime
    )
    if blt_order in [("ant1", "time"), ("ant2", "time")]:
        # sorting by ant1/ant2 then time means we split the other ant into a
        # separate group
        assert not is_rect
        assert not time_first
    elif isinstance(blt_order, list):
        assert not is_rect
        assert not time_first
    else:
        assert is_rect
        assert time_first == (
            (len(blt_order) == 2 and blt_order[-1] == "time")
            or (len(blt_order) == 1 and blt_order[0] != "time")
            or not blt_order  # we by default move time first (backwards, but still)
        )


def test_determine_blt_order_size_1():
    times = np.array([2458119.5])
    ant1 = np.array([0])
    ant2 = np.array([1])
    bl = uvutils.antnums_to_baseline(ant1, ant2, Nants_telescope=2)

    order = uvutils.determine_blt_order(
        time_array=times,
        ant_1_array=ant1,
        ant_2_array=ant2,
        baseline_array=bl,
        Nbls=1,
        Ntimes=1,
    )
    assert order == ("baseline", "time")
    is_rect, time_first = uvutils.determine_rectangularity(
        time_array=times, baseline_array=bl, nbls=1, ntimes=1
    )
    assert is_rect
    assert time_first


def test_antnums_to_baseline_miriad_convention():
    ant1 = np.array([1, 2, 3, 1, 1, 1, 255, 256])  # Ant1 array should be 1-based
    ant2 = np.array([1, 2, 3, 254, 255, 256, 1, 2])  # Ant2 array should be 1-based
    bl_gold = np.array([257, 514, 771, 510, 511, 67840, 65281, 65538], dtype="uint64")

    n_ant = 256
    bl = uvutils.antnums_to_baseline(
        ant1, ant2, Nants_telescope=n_ant, use_miriad_convention=True
    )
    np.testing.assert_allclose(bl, bl_gold)


def test_determine_rect_time_first():
    times = np.linspace(2458119.5, 2458120.5, 10)
    ant1 = np.arange(3)
    ant2 = np.arange(3)
    ANT1, ANT2 = np.meshgrid(ant1, ant2)
    bls = uvutils.antnums_to_baseline(ANT1.flatten(), ANT2.flatten(), Nants_telescope=3)

    rng = np.random.default_rng(12345)

    TIME = np.tile(times, len(bls))
    BL = np.concatenate([rng.permuted(bls) for i in range(len(times))])

    is_rect, time_first = uvutils.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=9, ntimes=10
    )
    assert not is_rect

    # now, permute time instead of bls
    TIME = np.concatenate([rng.permuted(times) for i in range(len(bls))])
    BL = np.tile(bls, len(times))
    is_rect, time_first = uvutils.determine_rectangularity(
        time_array=TIME, baseline_array=BL, nbls=9, ntimes=10
    )
    assert not is_rect

    TIME = np.array([1000.0, 1000.0, 2000.0, 1000.0])
    BLS = np.array([0, 0, 1, 0])

    is_rect, time_first = uvutils.determine_rectangularity(
        time_array=TIME, baseline_array=BLS, nbls=2, ntimes=2
    )
    assert not is_rect


def test_calc_app_coords_time_obj():
    # Generate ra/dec of zenith at time in the phase_frame coordinate system
    # to use for phasing
    telescope_location = EarthLocation.from_geodetic(lon=0, lat=1 * un.rad)

    # JD is arbitrary
    jd = 2454600

    zenith_coord = SkyCoord(
        alt=90 * un.deg,
        az=0 * un.deg,
        obstime=Time(jd, format="jd"),
        frame="altaz",
        location=telescope_location,
    )
    zenith_coord = zenith_coord.transform_to("icrs")

    obstime = Time(jd + (np.array([-1, 0, 1]) / 24.0), format="jd")

    ra = zenith_coord.ra.to_value("rad")
    dec = zenith_coord.dec.to_value("rad")
    app_ra_to, app_dec_to = uvutils.calc_app_coords(
        lon_coord=ra,
        lat_coord=dec,
        time_array=obstime,
        telescope_loc=telescope_location,
    )

    app_ra_nto, app_dec_nto = uvutils.calc_app_coords(
        lon_coord=ra,
        lat_coord=dec,
        time_array=obstime.utc.jd,
        telescope_loc=telescope_location,
    )

    np.testing.assert_allclose(app_ra_to, app_ra_nto)
    np.testing.assert_allclose(app_dec_to, app_dec_nto)


@pytest.mark.skipif(hasmoon, reason="lunarsky installed")
def test_uvw_track_generator_errs():
    with pytest.raises(
        ValueError, match="Need to install `lunarsky` package to work with MCMF frame."
    ):
        uvutils.uvw_track_generator(telescope_loc=(0, 0, 0), telescope_frame="MCMF")


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
    gen_results = uvutils.uvw_track_generator(
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


@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
@pytest.mark.parametrize("selenoid", ["SPHERE", "GSFC", "GRAIL23", "CE-1-LAM-GEO"])
def test_uvw_track_generator_moon(selenoid):
    # Note this isn't a particularly deep test, but it at least exercises the code.
    from spiceypy.utils.exceptions import SpiceUNKNOWNFRAME

    try:
        gen_results = uvutils.uvw_track_generator(
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
    np.testing.assert_allclose((gen_results["uvw"] ** 2.0).sum(1), 2.0)

    if selenoid == "SPHERE":
        # check defaults
        gen_results = uvutils.uvw_track_generator(
            lon_coord=0.0,
            lat_coord=0.0,
            coord_frame="icrs",
            telescope_loc=(0, 0, 0),
            time_array=2456789.0,
            antenna_positions=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            telescope_frame="mcmf",
        )

        # Check that the total lengths all match 1
        np.testing.assert_allclose((gen_results["uvw"] ** 2.0).sum(1), 2.0)


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
            status = uvutils.check_surface_based_positions(
                telescope_loc=None if (del_tel_loc) else tel_loc_dict[tel_loc],
                antenna_positions=None if (del_tel_loc is None) else ant_pos,
                telescope_frame=tel_frame_dict[check_frame],
                raise_error=err_state == "err",
                raise_warning=err_state == "warn",
            )

        assert (err_state == "err") or (status == (tel_loc == check_frame))


@pytest.mark.skipif(not hasmoon, reason="lunarsky not installed")
@pytest.mark.parametrize("tel_loc", ["Earth", "Moon"])
@pytest.mark.parametrize("check_frame", ["Earth", "Moon"])
def test_check_surface_based_positions_earthmoonloc(tel_loc, check_frame):
    frame = "mcmf" if (check_frame == "Moon") else "itrs"

    if tel_loc == "Earth":
        loc = EarthLocation.from_geodetic(0, 0, 0)
    else:
        loc = MoonLocation.from_selenodetic(0, 0, 0)

    if tel_loc == check_frame:
        assert uvutils.check_surface_based_positions(
            telescope_loc=loc, telescope_frame=frame
        )
    else:
        with pytest.raises(ValueError, match=(f"{frame} position vector")):
            uvutils.check_surface_based_positions(
                telescope_loc=[loc.x.value, loc.y.value, loc.z.value],
                telescope_frame=frame,
            )


def test_determine_pol_order_err():
    with pytest.raises(ValueError, match='order must be either "AIPS" or "CASA".'):
        uvutils.determine_pol_order([], order="ABC")


@pytest.mark.parametrize(
    "pols,aips_order,casa_order",
    [
        [[-8, -7, -6, -5], [3, 2, 1, 0], [3, 1, 0, 2]],
        [[-5, -6, -7, -8], [0, 1, 2, 3], [0, 2, 3, 1]],
        [[1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 3]],
    ],
)
@pytest.mark.parametrize("order", ["CASA", "AIPS"])
def test_pol_order(pols, aips_order, casa_order, order):
    check = uvutils.determine_pol_order(pols, order=order)

    if order == "CASA":
        assert all(check == casa_order)
    if order == "AIPS":
        assert all(check == aips_order)


def test_slicify():
    assert uvutils.slicify(None) is None
    assert uvutils.slicify(slice(None)) == slice(None)
    assert uvutils.slicify([]) is None
    assert uvutils.slicify([1, 2, 3]) == slice(1, 4, 1)
    assert uvutils.slicify([1]) == slice(1, 2, 1)
    assert uvutils.slicify([0, 2, 4]) == slice(0, 5, 2)
    assert uvutils.slicify([0, 1, 2, 7]) == [0, 1, 2, 7]


@pytest.mark.parametrize(
    "obj1,obj2,union_result,interset_result,diff_result",
    [
        [[1, 2, 3], [3, 4, 5], [1, 2, 3, 4, 5], [3], [1, 2]],  # Partial overlap
        [[1, 2], [1, 2], [1, 2], [1, 2], []],  # Full overlap
        [[1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6], [], [1, 3, 5]],  # No overlap
        [[1, 2], None, [1, 2], [1, 2], [1, 2]],  # Nones
    ],
)
def test_sorted_unique_ops(obj1, obj2, union_result, interset_result, diff_result):
    assert uvutils._sorted_unique_union(obj1, obj2) == union_result
    assert uvutils._sorted_unique_intersection(obj1, obj2) == interset_result
    assert uvutils._sorted_unique_difference(obj1, obj2) == diff_result


def test_generate_new_phase_center_id_errs():
    with pytest.raises(ValueError, match="Cannot specify old_id if no catalog"):
        uvutils.generate_new_phase_center_id(old_id=1)

    with pytest.raises(ValueError, match="Provided cat_id was found in reserved_ids"):
        uvutils.generate_new_phase_center_id(cat_id=1, reserved_ids=[1, 2, 3])