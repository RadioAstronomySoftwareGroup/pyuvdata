"""Tests for common utility functions."""
import os
import nose.tools as nt
import pyuvdata
import numpy as np
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils

ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)


def test_XYZ_from_LatLonAlt():
    """Test conversion from lat/lon/alt to ECEF xyz with reference values."""
    out_xyz = pyuvdata.XYZ_from_LatLonAlt(ref_latlonalt[0], ref_latlonalt[1],
                                          ref_latlonalt[2])
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    nt.assert_true(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))

    # test error checking
    nt.assert_raises(ValueError, pyuvdata.XYZ_from_LatLonAlt, ref_latlonalt[0],
                     ref_latlonalt[1], np.array([ref_latlonalt[2], ref_latlonalt[2]]))
    nt.assert_raises(ValueError, pyuvdata.XYZ_from_LatLonAlt, ref_latlonalt[0],
                     np.array([ref_latlonalt[1], ref_latlonalt[1]]), ref_latlonalt[2])


def test_LatLonAlt_from_XYZ():
    """Test conversion from ECEF xyz to lat/lon/alt with reference values."""
    out_latlonalt = pyuvdata.LatLonAlt_from_XYZ(ref_xyz)
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    nt.assert_true(np.allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3))
    nt.assert_raises(ValueError, pyuvdata.LatLonAlt_from_XYZ, ref_latlonalt)

    # test passing multiple values
    xyz_mult = np.stack((np.array(ref_xyz), np.array(ref_xyz)), axis=1)
    lat_vec, lon_vec, alt_vec = pyuvdata.LatLonAlt_from_XYZ(xyz_mult)
    nt.assert_true(np.allclose(ref_latlonalt, (lat_vec[1], lon_vec[1], alt_vec[1]), rtol=0, atol=1e-3))

    # test error checking
    nt.assert_raises(ValueError, pyuvdata.LatLonAlt_from_XYZ, ref_xyz[0:1])


def test_ENU_tofrom_ECEF():
    center_lat = -30.7215261207 * np.pi / 180.0
    center_lon = 21.4283038269 * np.pi / 180.0
    center_alt = 1051.7
    lats = np.array([-30.72218216, -30.72138101, -30.7212785, -30.7210011,
                     -30.72159853, -30.72206199, -30.72174614, -30.72188775,
                     -30.72183915, -30.72100138]) * np.pi / 180.0
    lons = np.array([21.42728211, 21.42811727, 21.42814544, 21.42795736,
                     21.42686739, 21.42918772, 21.42785662, 21.4286408,
                     21.42750933, 21.42896567]) * np.pi / 180.0
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

    xyz = pyuvdata.XYZ_from_LatLonAlt(lats, lons, alts)
    nt.assert_true(np.allclose(np.stack((x, y, z)), xyz, atol=1e-3))

    enu = pyuvdata.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(np.stack((east, north, up)), enu, atol=1e-3))

    # check that a round trip gives the original value.
    xyz_from_enu = pyuvdata.ECEF_from_ENU(enu, center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(xyz, xyz_from_enu, atol=1e-3))

    # check passing a single value
    enu_single = pyuvdata.ENU_from_ECEF(xyz[:, 0], center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(np.stack((east[0], north[0], up[0])), enu[:, 0], atol=1e-3))

    xyz_from_enu = pyuvdata.ECEF_from_ENU(enu_single, center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(xyz[:, 0], xyz_from_enu, atol=1e-3))

    # error checking
    nt.assert_raises(ValueError, pyuvdata.ENU_from_ECEF, xyz[0:1, :], center_lat, center_lon, center_alt)
    nt.assert_raises(ValueError, pyuvdata.ECEF_from_ENU, enu[0:1, :], center_lat, center_lon, center_alt)
    nt.assert_raises(ValueError, pyuvdata.ENU_from_ECEF, xyz / 2., center_lat, center_lon, center_alt)


def test_mwa_ecef_conversion():
    '''
    Test based on comparing the antenna locations in a Cotter uvfits file to
    the antenna locations in MWA_tools.
    '''

    test_data_file = os.path.join(DATA_PATH, 'mwa128_ant_layouts.npz')
    f = np.load(test_data_file)

    # From the STABXYZ table in a cotter-generated uvfits file, obsid = 1066666832
    xyz = f['stabxyz']
    # From the East/North/Height columns in a cotter-generated metafits file, obsid = 1066666832
    enh = f['ENH']
    # From a text file antenna_locations.txt in MWA_Tools/scripts
    txt_topo = f['txt_topo']

    # From the unphased uvw coordinates of obsid 1066666832, positions relative to antenna 0
    # these aren't used in the current test, but are interesting and might help with phasing diagnosis in the future
    uvw_topo = f['uvw_topo']
    # Sky coordinates are flipped for uvw derived values
    uvw_topo = -uvw_topo
    uvw_topo += txt_topo[0]

    # transpose these arrays to get them into the right shape
    txt_topo = txt_topo.T
    uvw_topo = uvw_topo.T
    enh = enh.T

    # ARRAYX, ARRAYY, ARRAYZ in ECEF frame from Cotter file
    arrcent = f['arrcent']
    lat, lon, alt = uvutils.LatLonAlt_from_XYZ(arrcent)

    # The STABXYZ coordinates are defined with X through the local meridian,
    # so rotate back to the prime meridian
    new_xyz = uvutils.ECEF_from_rotECEF(xyz.T, lon)
    # add in array center to get real ECEF
    ecef_xyz = new_xyz + arrcent

    enu = uvutils.ENU_from_ECEF(ecef_xyz.T, lat, lon, alt)

    nt.assert_true(np.allclose(enu, enh))

    # test other direction of ECEF rotation
    rot_xyz = uvutils.rotECEF_from_ECEF(new_xyz, lon)
    nt.assert_true(np.allclose(rot_xyz.T, xyz))


def test_pol_funcs():
    """ Test utility functions to convert between polarization strings and numbers """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    pol_str = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q', 'U', 'V']
    nt.assert_equal(pol_nums, uvutils.polstr2num(pol_str))
    nt.assert_equal(pol_str, uvutils.polnum2str(pol_nums))
    # Check individuals
    nt.assert_equal(-6, uvutils.polstr2num('YY'))
    nt.assert_equal('V', uvutils.polnum2str(4))
    # Check errors
    nt.assert_raises(KeyError, uvutils.polstr2num, 'foo')
    nt.assert_raises(ValueError, uvutils.polstr2num, 1)
    nt.assert_raises(ValueError, uvutils.polnum2str, 7.3)


def test_jones_num_funcs():
    """ Test utility functions to convert between jones polarization strings and numbers """

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    jstr = ['jyx', 'jxy', 'jyy', 'jxx', 'jlr', 'jrl', 'jll', 'jrr']
    nt.assert_equal(jnums, uvutils.jstr2num(jstr))
    nt.assert_equal(jstr, uvutils.jnum2str(jnums))
    # Check shorthands
    jstr = ['yx', 'xy', 'yy', 'y', 'xx', 'x', 'lr', 'rl', 'll', 'l', 'rr', 'r']
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    nt.assert_equal(jnums, uvutils.jstr2num(jstr))
    # Check individuals
    nt.assert_equal(-6, uvutils.jstr2num('jyy'))
    nt.assert_equal('jxy', uvutils.jnum2str(-7))
    # Check errors
    nt.assert_raises(KeyError, uvutils.jstr2num, 'foo')
    nt.assert_raises(ValueError, uvutils.jstr2num, 1)
    nt.assert_raises(ValueError, uvutils.jnum2str, 7.3)
