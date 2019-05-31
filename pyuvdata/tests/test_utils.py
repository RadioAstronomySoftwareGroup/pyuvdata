# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for common utility functions.

"""
from __future__ import absolute_import, division, print_function

import os
import pytest
import numpy as np
import six
from astropy import units
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits

import pyuvdata
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
import pyuvdata.version as uvversion


ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)


def test_XYZ_from_LatLonAlt():
    """Test conversion from lat/lon/alt to ECEF xyz with reference values."""
    out_xyz = uvutils.XYZ_from_LatLonAlt(ref_latlonalt[0], ref_latlonalt[1],
                                         ref_latlonalt[2])
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    assert np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3)

    # test error checking
    pytest.raises(ValueError, uvutils.XYZ_from_LatLonAlt, ref_latlonalt[0],
                  ref_latlonalt[1], np.array([ref_latlonalt[2], ref_latlonalt[2]]))
    pytest.raises(ValueError, uvutils.XYZ_from_LatLonAlt, ref_latlonalt[0],
                  np.array([ref_latlonalt[1], ref_latlonalt[1]]), ref_latlonalt[2])


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
    assert np.allclose(ref_latlonalt, (lat_vec[1], lon_vec[1], alt_vec[1]), rtol=0, atol=1e-3)
    # check warning if array transposed
    uvtest.checkWarnings(uvutils.LatLonAlt_from_XYZ, [xyz_mult.T],
                         message='The expected shape of ECEF xyz array',
                         category=DeprecationWarning)
    # check warning if  3 x 3 array
    xyz_3 = np.stack((np.array(ref_xyz), np.array(ref_xyz), np.array(ref_xyz)))
    uvtest.checkWarnings(uvutils.LatLonAlt_from_XYZ, [xyz_3],
                         message='The xyz array in LatLonAlt_from_XYZ is',
                         category=DeprecationWarning)
    # check error if only 2 coordinates
    pytest.raises(ValueError, uvutils.LatLonAlt_from_XYZ, xyz_mult[:, 0:2])

    # test error checking
    pytest.raises(ValueError, uvutils.LatLonAlt_from_XYZ, ref_xyz[0:1])


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

    xyz = uvutils.XYZ_from_LatLonAlt(lats, lons, alts)
    assert np.allclose(np.stack((x, y, z), axis=1), xyz, atol=1e-3)

    enu = uvutils.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)
    assert np.allclose(np.stack((east, north, up), axis=1), enu, atol=1e-3)
    # check warning if array transposed
    uvtest.checkWarnings(uvutils.ENU_from_ECEF, [xyz.T, center_lat, center_lon,
                                                 center_alt],
                         message='The expected shape of ECEF xyz array',
                         category=DeprecationWarning)
    # check warning if  3 x 3 array
    uvtest.checkWarnings(uvutils.ENU_from_ECEF, [xyz[0:3], center_lat, center_lon,
                                                 center_alt],
                         message='The xyz array in ENU_from_ECEF is',
                         category=DeprecationWarning)
    # check error if only 2 coordinates
    pytest.raises(ValueError, uvutils.ENU_from_ECEF, xyz[:, 0:2],
                  center_lat, center_lon, center_alt)

    # check that a round trip gives the original value.
    xyz_from_enu = uvutils.ECEF_from_ENU(enu, center_lat, center_lon, center_alt)
    assert np.allclose(xyz, xyz_from_enu, atol=1e-3)
    # check warning if array transposed
    uvtest.checkWarnings(uvutils.ECEF_from_ENU, [enu.T, center_lat, center_lon,
                                                 center_alt],
                         message='The expected shape the ENU array',
                         category=DeprecationWarning)
    # check warning if  3 x 3 array
    uvtest.checkWarnings(uvutils.ECEF_from_ENU, [enu[0:3], center_lat, center_lon,
                                                 center_alt],
                         message='The enu array in ECEF_from_ENU is',
                         category=DeprecationWarning)
    # check error if only 2 coordinates
    pytest.raises(ValueError, uvutils.ENU_from_ECEF, enu[:, 0:2], center_lat,
                  center_lon, center_alt)

    # check passing a single value
    enu_single = uvutils.ENU_from_ECEF(xyz[0, :], center_lat, center_lon, center_alt)
    assert np.allclose(np.array((east[0], north[0], up[0])), enu[0, :], atol=1e-3)

    xyz_from_enu = uvutils.ECEF_from_ENU(enu_single, center_lat, center_lon, center_alt)
    assert np.allclose(xyz[0, :], xyz_from_enu, atol=1e-3)

    # error checking
    pytest.raises(ValueError, uvutils.ENU_from_ECEF, xyz[:, 0:1], center_lat, center_lon, center_alt)
    pytest.raises(ValueError, uvutils.ECEF_from_ENU, enu[:, 0:1], center_lat, center_lon, center_alt)
    pytest.raises(ValueError, uvutils.ENU_from_ECEF, xyz / 2., center_lat, center_lon, center_alt)


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

    # ARRAYX, ARRAYY, ARRAYZ in ECEF frame from Cotter file
    arrcent = f['arrcent']
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


def test_phasing_funcs():
    # these tests are based on a notebook where I tested against the mwa_tools phasing code
    ra_hrs = 12.1
    dec_degs = -42.3
    mjd = 55780.1

    array_center_xyz = np.array([-2559454.08, 5095372.14, -2849057.18])
    lat_lon_alt = uvutils.LatLonAlt_from_XYZ(array_center_xyz)

    obs_time = Time(mjd, format='mjd', location=(lat_lon_alt[1], lat_lon_alt[0]))

    icrs_coord = SkyCoord(ra=Angle(ra_hrs, unit='hr'), dec=Angle(dec_degs, unit='deg'),
                          obstime=obs_time)
    gcrs_coord = icrs_coord.transform_to('gcrs')

    # in east/north/up frame (relative to array center) in meters: (Nants, 3)
    ants_enu = np.array([-101.94, 0156.41, 0001.24])

    ant_xyz_abs = uvutils.ECEF_from_ENU(ants_enu, lat_lon_alt[0], lat_lon_alt[1], lat_lon_alt[2])
    ant_xyz_rel_itrs = ant_xyz_abs - array_center_xyz
    ant_xyz_rel_rot = uvutils.rotECEF_from_ECEF(ant_xyz_rel_itrs, lat_lon_alt[1])

    array_center_coord = SkyCoord(x=array_center_xyz[0] * units.m,
                                  y=array_center_xyz[1] * units.m,
                                  z=array_center_xyz[2] * units.m,
                                  frame='itrs',
                                  obstime=obs_time)

    itrs_coord = SkyCoord(x=ant_xyz_abs[0] * units.m,
                          y=ant_xyz_abs[1] * units.m,
                          z=ant_xyz_abs[2] * units.m,
                          frame='itrs',
                          obstime=obs_time)

    gcrs_array_center = array_center_coord.transform_to('gcrs')
    gcrs_from_itrs_coord = itrs_coord.transform_to('gcrs')

    gcrs_rel = (gcrs_from_itrs_coord.cartesian - gcrs_array_center.cartesian).get_xyz().T

    gcrs_uvw = uvutils.phase_uvw(gcrs_coord.ra.rad, gcrs_coord.dec.rad,
                                 gcrs_rel.value)

    mwa_tools_calcuvw_u = -97.122828
    mwa_tools_calcuvw_v = 50.388281
    mwa_tools_calcuvw_w = -151.27976

    assert np.allclose(gcrs_uvw[0, 0], mwa_tools_calcuvw_u, atol=1e-3)
    assert np.allclose(gcrs_uvw[0, 1], mwa_tools_calcuvw_v, atol=1e-3)
    assert np.allclose(gcrs_uvw[0, 2], mwa_tools_calcuvw_w, atol=1e-3)

    # also test unphasing
    temp2 = uvutils.unphase_uvw(gcrs_coord.ra.rad, gcrs_coord.dec.rad,
                                np.squeeze(gcrs_uvw))
    assert np.allclose(gcrs_rel.value, temp2)


def test_pol_funcs():
    """ Test utility functions to convert between polarization strings and numbers """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    pol_str = ['yx', 'xy', 'yy', 'xx', 'lr', 'rl', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    assert pol_nums == uvutils.polstr2num(pol_str)
    assert pol_str == uvutils.polnum2str(pol_nums)
    # Check individuals
    assert -6 == uvutils.polstr2num('YY')
    assert 'pV' == uvutils.polnum2str(4)
    # Check errors
    pytest.raises(KeyError, uvutils.polstr2num, 'foo')
    pytest.raises(ValueError, uvutils.polstr2num, 1)
    pytest.raises(ValueError, uvutils.polnum2str, 7.3)
    # Check parse
    assert uvutils.parse_polstr("xX") == 'xx'
    assert uvutils.parse_polstr("XX") == 'xx'
    assert uvutils.parse_polstr('i') == 'pI'


def test_pol_funcs_x_orientation():
    """ Test utility functions to convert between polarization strings and numbers with x_orientation """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]

    x_orient1 = 'e'
    pol_str = ['ne', 'en', 'nn', 'ee', 'lr', 'rl', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    assert pol_nums == uvutils.polstr2num(pol_str, x_orientation=x_orient1)
    assert pol_str == uvutils.polnum2str(pol_nums, x_orientation=x_orient1)
    # Check individuals
    assert -6 == uvutils.polstr2num('NN', x_orientation=x_orient1)
    assert 'pV' == uvutils.polnum2str(4)
    # Check errors
    pytest.raises(KeyError, uvutils.polstr2num, 'foo', x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.polstr2num, 1, x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.polnum2str, 7.3, x_orientation=x_orient1)
    # Check parse
    assert uvutils.parse_polstr("eE", x_orientation=x_orient1) == 'ee'
    assert uvutils.parse_polstr("xx", x_orientation=x_orient1) == 'ee'
    assert uvutils.parse_polstr("NN", x_orientation=x_orient1) == 'nn'
    assert uvutils.parse_polstr("yy", x_orientation=x_orient1) == 'nn'
    assert uvutils.parse_polstr('i', x_orientation=x_orient1) == 'pI'

    x_orient2 = 'n'
    pol_str = ['en', 'ne', 'ee', 'nn', 'lr', 'rl', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    assert pol_nums == uvutils.polstr2num(pol_str, x_orientation=x_orient2)
    assert pol_str == uvutils.polnum2str(pol_nums, x_orientation=x_orient2)
    # Check individuals
    assert -6 == uvutils.polstr2num('EE', x_orientation=x_orient2)
    assert 'pV' == uvutils.polnum2str(4)
    # Check errors
    pytest.raises(KeyError, uvutils.polstr2num, 'foo', x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.polstr2num, 1, x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.polnum2str, 7.3, x_orientation=x_orient2)
    # Check parse
    assert uvutils.parse_polstr("nN", x_orientation=x_orient2) == 'nn'
    assert uvutils.parse_polstr("xx", x_orientation=x_orient2) == 'nn'
    assert uvutils.parse_polstr("EE", x_orientation=x_orient2) == 'ee'
    assert uvutils.parse_polstr("yy", x_orientation=x_orient2) == 'ee'
    assert uvutils.parse_polstr('i', x_orientation=x_orient2) == 'pI'

    # check warnings for non-recognized x_orientation
    assert uvtest.checkWarnings(uvutils.polstr2num, ['xx'], {'x_orientation': 'foo'},
                                message='x_orientation not recognized') == -5
    assert uvtest.checkWarnings(uvutils.polnum2str, [-6], {'x_orientation': 'foo'},
                                message='x_orientation not recognized') == 'yy'


def test_jones_num_funcs():
    """ Test utility functions to convert between jones polarization strings and numbers """

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    jstr = ['Jyx', 'Jxy', 'Jyy', 'Jxx', 'Jlr', 'Jrl', 'Jll', 'Jrr']
    assert jnums == uvutils.jstr2num(jstr)
    assert jstr, uvutils.jnum2str(jnums)
    # Check shorthands
    jstr = ['yx', 'xy', 'yy', 'y', 'xx', 'x', 'lr', 'rl', 'll', 'l', 'rr', 'r']
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == uvutils.jstr2num(jstr)
    # Check individuals
    assert -6 == uvutils.jstr2num('jyy')
    assert 'Jxy' == uvutils.jnum2str(-7)
    # Check errors
    pytest.raises(KeyError, uvutils.jstr2num, 'foo')
    pytest.raises(ValueError, uvutils.jstr2num, 1)
    pytest.raises(ValueError, uvutils.jnum2str, 7.3)

    # check parse method
    assert uvutils.parse_jpolstr('x') == 'Jxx'
    assert uvutils.parse_jpolstr('xy') == 'Jxy'
    assert uvutils.parse_jpolstr('XY') == 'Jxy'


def test_jones_num_funcs_x_orientation():
    """ Test utility functions to convert between jones polarization strings and numbers with x_orientation"""

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    x_orient1 = 'east'
    jstr = ['Jne', 'Jen', 'Jnn', 'Jee', 'Jlr', 'Jrl', 'Jll', 'Jrr']
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient1)
    assert jstr == uvutils.jnum2str(jnums, x_orientation=x_orient1)
    # Check shorthands
    jstr = ['ne', 'en', 'nn', 'n', 'ee', 'e', 'lr', 'rl', 'll', 'l', 'rr', 'r']
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient1)
    # Check individuals
    assert -6 == uvutils.jstr2num('jnn', x_orientation=x_orient1)
    assert 'Jen' == uvutils.jnum2str(-7, x_orientation=x_orient1)
    # Check errors
    pytest.raises(KeyError, uvutils.jstr2num, 'foo', x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.jstr2num, 1, x_orientation=x_orient1)
    pytest.raises(ValueError, uvutils.jnum2str, 7.3, x_orientation=x_orient1)

    # check parse method
    assert uvutils.parse_jpolstr('e', x_orientation=x_orient1) == 'Jee'
    assert uvutils.parse_jpolstr('x', x_orientation=x_orient1) == 'Jee'
    assert uvutils.parse_jpolstr('y', x_orientation=x_orient1) == 'Jnn'
    assert uvutils.parse_jpolstr('en', x_orientation=x_orient1) == 'Jen'
    assert uvutils.parse_jpolstr('NE', x_orientation=x_orient1) == 'Jne'

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    x_orient2 = 'north'
    jstr = ['Jen', 'Jne', 'Jee', 'Jnn', 'Jlr', 'Jrl', 'Jll', 'Jrr']
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient2)
    assert jstr == uvutils.jnum2str(jnums, x_orientation=x_orient2)
    # Check shorthands
    jstr = ['en', 'ne', 'ee', 'e', 'nn', 'n', 'lr', 'rl', 'll', 'l', 'rr', 'r']
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    assert jnums == uvutils.jstr2num(jstr, x_orientation=x_orient2)
    # Check individuals
    assert -6 == uvutils.jstr2num('jee', x_orientation=x_orient2)
    assert 'Jne' == uvutils.jnum2str(-7, x_orientation=x_orient2)
    # Check errors
    pytest.raises(KeyError, uvutils.jstr2num, 'foo', x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.jstr2num, 1, x_orientation=x_orient2)
    pytest.raises(ValueError, uvutils.jnum2str, 7.3, x_orientation=x_orient2)

    # check parse method
    assert uvutils.parse_jpolstr('e', x_orientation=x_orient2) == 'Jee'
    assert uvutils.parse_jpolstr('x', x_orientation=x_orient2) == 'Jnn'
    assert uvutils.parse_jpolstr('y', x_orientation=x_orient2) == 'Jee'
    assert uvutils.parse_jpolstr('en', x_orientation=x_orient2) == 'Jen'
    assert uvutils.parse_jpolstr('NE', x_orientation=x_orient2) == 'Jne'

    # check warnings for non-recognized x_orientation
    assert uvtest.checkWarnings(uvutils.jstr2num, ['x'], {'x_orientation': 'foo'},
                                message='x_orientation not recognized') == -5
    assert uvtest.checkWarnings(uvutils.jnum2str, [-6], {'x_orientation': 'foo'},
                                message='x_orientation not recognized') == 'Jyy'


def test_conj_pol():
    """ Test function to conjugate pols """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    cpol_nums = [-7, -8, -6, -5, -3, -4, -2, -1, 1, 2, 3, 4]
    assert pol_nums == uvutils.conj_pol(cpol_nums)
    assert uvutils.conj_pol(pol_nums) == cpol_nums
    pol_str = ['yx', 'xy', 'yy', 'xx', 'lr', 'rl', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    cpol_str = ['xy', 'yx', 'yy', 'xx', 'rl', 'lr', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    assert pol_str == uvutils.conj_pol(cpol_str)
    assert uvutils.conj_pol(pol_str) == cpol_str
    assert [pol_str, pol_nums] == uvutils.conj_pol([cpol_str, cpol_nums])

    jstr = ['Jyx', 'Jxy', 'Jyy', 'Jxx', 'Jlr', 'Jrl', 'Jll', 'Jrr']
    cjstr = ['Jxy', 'Jyx', 'Jyy', 'Jxx', 'Jrl', 'Jlr', 'Jll', 'Jrr']
    conj_cjstr = uvtest.checkWarnings(uvutils.conj_pol, [cjstr], nwarnings=8,
                                      category=DeprecationWarning,
                                      message='conj_pol should not be called with jones')
    assert jstr == conj_cjstr

    # Test invalid pol
    pytest.raises(ValueError, uvutils.conj_pol, 2.3)


def test_deprecated_funcs():
    uvtest.checkWarnings(uvutils.get_iterable, [5], category=DeprecationWarning,
                         message='The get_iterable function is deprecated')

    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    with fits.open(testfile, memmap=True) as hdu_list:
        uvtest.checkWarnings(uvutils.fits_indexhdus, [hdu_list],
                             category=DeprecationWarning,
                             message='The fits_indexhdus function is deprecated')

        vis_hdu = hdu_list[0]
        uvtest.checkWarnings(uvutils.fits_gethduaxis, [vis_hdu, 5],
                             category=DeprecationWarning,
                             message='The fits_gethduaxis function is deprecated')

    uvtest.checkWarnings(uvutils.check_history_version, ['some random history',
                                                         uvversion.version],
                         category=DeprecationWarning,
                         message='The check_history_version function is deprecated')

    uvtest.checkWarnings(uvutils.check_histories, ['some random history',
                                                   'some random history'],
                         category=DeprecationWarning,
                         message='The check_histories function is deprecated')

    uvtest.checkWarnings(uvutils.combine_histories, ['some random history',
                                                     uvversion.version],
                         category=DeprecationWarning,
                         message='The combine_histories function is deprecated')


def test_redundancy_finder():
    """
    Check that get_baseline_redundancies and get_antenna_redundancies return consistent
    redundant groups for a test file with the HERA19 layout.
    """
    uvd = pyuvdata.UVData()
    uvd.read_uvfits(os.path.join(DATA_PATH, 'fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits'))

    uvd.select(times=uvd.time_array[0])
    uvd.unphase_to_drift()   # uvw_array is now equivalent to baseline positions
    uvtest.checkWarnings(uvd.conjugate_bls, func_kwargs={'convention': 'u>0', 'use_enu': True},
                         message=['The default for the `center`'],
                         nwarnings=1, category=DeprecationWarning)

    tol = 0.05  # meters

    bl_positions = uvd.uvw_array

    pytest.raises(ValueError, uvutils.get_baseline_redundancies,
                  uvd.baseline_array, bl_positions[0:2, 0:1])
    baseline_groups, vec_bin_centers, lens = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol)

    baseline_groups, vec_bin_centers, lens = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol)

    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            assert np.allclose(np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])),
                               lens[gi], atol=tol)

    # Shift the baselines around in a circle. Check that the same baselines are
    # recovered to the corresponding tolerance increase.
    # This moves one baseline at a time by a fixed displacement and checks that
    # the redundant groups are the same.

    hightol = 0.25  # meters. Less than the smallest baseline in the file.
    Nbls = uvd.Nbls
    Nshifts = 5
    shift_angs = np.linspace(0, 2 * np.pi, Nshifts)
    base_shifts = np.stack(((hightol - tol) * np.cos(shift_angs),
                            (hightol - tol) * np.sin(shift_angs),
                            np.zeros(Nshifts))).T
    for sh in base_shifts:
        for bi in range(Nbls):
            # Shift one baseline at a time.
            bl_positions_new = uvd.uvw_array
            bl_positions_new[bi] += sh

            baseline_groups_new, vec_bin_centers, lens = uvutils.get_baseline_redundancies(
                uvd.baseline_array, bl_positions_new, tol=hightol)

            for gi, gp in enumerate(baseline_groups_new):
                for bl in gp:
                    bl_ind = np.where(uvd.baseline_array == bl)
                    bl_vec = bl_positions[bl_ind]
                    assert np.allclose(np.sqrt(np.abs(np.dot(bl_vec, vec_bin_centers[gi]))),
                                       lens[gi], atol=hightol)

            # Compare baseline groups:
            a = [tuple(el) for el in baseline_groups]
            b = [tuple(el) for el in baseline_groups_new]
            assert set(a) == set(b)

    tol = 0.05

    antpos, antnums = uvtest.checkWarnings(uvd.get_ENU_antpos,
                                           message=['The default for the `center`'],
                                           category=DeprecationWarning,
                                           nwarnings=1)

    baseline_groups_ants, vec_bin_centers, lens = uvutils.get_antenna_redundancies(
        antnums, antpos, tol=tol, include_autos=False)
    # Under these conditions, should see 19 redundant groups in the file.
    assert len(baseline_groups_ants) == 19

    # Check with conjugated baseline redundancies returned
    u16_0 = bl_positions[16, 0]
    # Ensure at least one baseline has u==0 and v!=0 (for coverage of this case)
    bl_positions[16, 0] = 0
    baseline_groups, vec_bin_centers, lens, conjugates = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True)

    # restore baseline (16,0) and repeat to get correct groups
    bl_positions[16, 0] = u16_0
    baseline_groups, vec_bin_centers, lens, conjugates = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True)

    # Should get the same groups as with the antenna method:
    baseline_groups_flipped = []
    for bgp in baseline_groups:
        bgp_new = []
        for bl in bgp:
            ai, aj = uvutils.baseline_to_antnums(bl, uvd.Nants_telescope)
            if bl in conjugates:
                bgp_new.append(uvutils.antnums_to_baseline(aj, ai, uvd.Nants_telescope))
            else:
                bgp_new.append(uvutils.antnums_to_baseline(ai, aj, uvd.Nants_telescope))
        bgp_new.sort()
        baseline_groups_flipped.append(bgp_new)
    baseline_groups = [sorted(bgp) for bgp in baseline_groups]
    assert np.all(sorted(baseline_groups_ants) == sorted(baseline_groups_flipped))
    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            if bl in conjugates:
                bl_vec *= (-1)
            assert np.isclose(np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])),
                              lens[gi], atol=tol)


def test_redundancy_conjugates():
    # Check that the correct baselines are flipped when returning redundancies with conjugates.

    Nants = 10
    tol = 0.5
    ant1_arr = np.tile(np.arange(Nants), Nants)
    ant2_arr = np.repeat(np.arange(Nants), Nants)
    Nbls = ant1_arr.size
    bl_inds = uvutils.antnums_to_baseline(ant1_arr, ant2_arr, Nants)

    maxbl = 100.
    bl_vecs = np.random.uniform(-maxbl, maxbl, (Nbls, 3))
    bl_vecs[0, 0] = 0
    bl_vecs[1, 0:2] = 0

    expected_conjugates = []
    for i, (u, v, w) in enumerate(bl_vecs):
        if (u < 0) or (v < 0 and u == 0) or (w < 0 and u == v == 0):
            expected_conjugates.append(bl_inds[i])
    bl_gps, vecs, lens, conjugates = uvutils.get_baseline_redundancies(
        bl_inds, bl_vecs, tol=tol, with_conjugates=True)

    assert sorted(conjugates) == sorted(expected_conjugates)


def test_redundancy_finder_fully_redundant_array():
    """Test the redundancy finder only returns one baseline group for fully redundant array."""
    uvd = pyuvdata.UVData()
    uvd.read_uvfits(os.path.join(DATA_PATH, 'test_redundant_array.uvfits'))
    uvd.select(times=uvd.time_array[0])

    tol = 1  # meters
    bl_positions = uvd.uvw_array

    baseline_groups, vec_bin_centers, lens, conjugates = uvutils.get_baseline_redundancies(
        uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True)

    # Only 1 set of redundant baselines
    assert len(baseline_groups) == 1
    #  Should return the input baselines
    assert baseline_groups[0].sort() == np.unique(uvd.baseline_array).sort()


def test_reraise_context():
    with pytest.raises(ValueError) as cm:
        try:
            uvutils.LatLonAlt_from_XYZ(ref_xyz[0:1])
        except ValueError:
            uvutils._reraise_context('Add some info')
    assert 'Add some info: xyz values should be ECEF x, y, z coordinates in meters' in str(cm.value)

    with pytest.raises(ValueError) as cm:
        try:
            uvutils.LatLonAlt_from_XYZ(ref_xyz[0:1])
        except ValueError:
            uvutils._reraise_context('Add some info %s', 'and then more')
    assert 'Add some info and then more: xyz values should be ECEF x, y, z coordinates in meters' in str(cm.value)

    with pytest.raises(EnvironmentError) as cm:
        try:
            raise EnvironmentError(1, 'some bad problem')
        except EnvironmentError:
            uvutils._reraise_context('Add some info')
    assert 'Add some info: some bad problem' in str(cm.value)


def test_str_to_bytes():
    test_str = 'HERA'
    test_bytes = uvutils._str_to_bytes(test_str)
    assert type(test_bytes) == six.binary_type
    assert test_bytes == b'\x48\x45\x52\x41'
    return


def test_bytes_to_str():
    test_bytes = b'\x48\x45\x52\x41'
    test_str = uvutils._bytes_to_str(test_bytes)
    assert type(test_str) == str
    assert test_str == 'HERA'
    return


def test_reorder_conj_pols_non_list():
    pytest.raises(ValueError, uvutils.reorder_conj_pols, 4)


def test_reorder_conj_pols_strings():
    pols = ['xx', 'xy', 'yx']
    corder = uvutils.reorder_conj_pols(pols)
    assert np.array_equal(corder, [0, 2, 1])


def test_reorder_conj_pols_ints():
    pols = [-5, -7, -8]  # 'xx', 'xy', 'yx'
    corder = uvutils.reorder_conj_pols(pols)
    assert np.array_equal(corder, [0, 2, 1])


def test_reorder_conj_pols_missing_conj():
    pols = ['xx', 'xy']  # Missing 'yx'
    pytest.raises(ValueError, uvutils.reorder_conj_pols, pols)


def test_collapse_mean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out = uvutils.collapse(data, 'mean', axis=0)
    out1 = uvutils.mean_collapse(data, axis=0)
    # Actual values are tested in test_mean_no_weights
    assert np.array_equal(out, out1)


def test_collapse_mean_returned_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out, wo = uvutils.collapse(data, 'mean', axis=0, return_weights=True)
    out1, wo1 = uvutils.mean_collapse(data, axis=0, return_weights=True)
    # Actual values are tested in test_mean_no_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_mean_returned_with_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1. / data
    out, wo = uvutils.collapse(data, 'mean', weights=w, axis=0, return_weights=True)
    out1, wo1 = uvutils.mean_collapse(data, weights=w, axis=0, return_weights=True)
    # Actual values are tested in test_mean_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_absmean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = (-1)**i * np.ones_like(data[:, i])
    out = uvutils.collapse(data, 'absmean', axis=0)
    out1 = uvutils.absmean_collapse(data, axis=0)
    # Actual values are tested in test_absmean_no_weights
    assert np.array_equal(out, out1)


def test_collapse_quadmean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out = uvutils.collapse(data, 'quadmean', axis=0)
    out1 = uvutils.quadmean_collapse(data, axis=0)
    # Actual values are tested in test_absmean_no_weights
    assert np.array_equal(out, out1)


def test_collapse_or_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool)
    data[0, 8] = True
    o = uvutils.collapse(data, 'or', axis=0)
    o1 = uvutils.or_collapse(data, axis=0)
    assert np.array_equal(o, o1)


def test_collapse_and_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool)
    data[0, :] = True
    o = uvutils.collapse(data, 'and', axis=0)
    o1 = uvutils.and_collapse(data, axis=0)
    assert np.array_equal(o, o1)


def test_collapse_error():
    pytest.raises(ValueError, uvutils.collapse, np.ones((2, 3)), 'fooboo')


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


def test_mean_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1. / data
    out, wo = uvutils.mean_collapse(data, weights=w, axis=0, return_weights=True)
    assert np.allclose(out * wo, data.shape[0])
    assert np.allclose(wo, float(data.shape[0]) / (np.arange(data.shape[1]) + 1))
    out, wo = uvutils.mean_collapse(data, weights=w, axis=1, return_weights=True)
    assert np.allclose(out * wo, data.shape[1])
    assert np.allclose(wo, np.sum(1. / (np.arange(data.shape[1]) + 1)))

    # Zero weights
    w = np.ones_like(w)
    w[0, :] = 0
    w[:, 0] = 0
    out, wo = uvutils.mean_collapse(data, weights=w, axis=0, return_weights=True)
    ans = np.arange(data.shape[1]).astype(np.float) + 1
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
    ans = np.arange(data.shape[1]).astype(np.float)
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
        data1[:, i] = (-1)**i * np.ones_like(data1[:, i])
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
    o2, w2 = uvutils.mean_collapse(np.abs(data)**2, return_weights=True)
    o3 = uvutils.quadmean_collapse(data)  # without return_weights
    o2 = np.sqrt(o2)
    assert o1 == o2
    assert w1 == w2
    assert o1 == o3


def test_or_collapse():
    # Fake data
    data = np.zeros((50, 25), np.bool)
    data[0, 8] = True
    o = uvutils.or_collapse(data, axis=0)
    ans = np.zeros(25, np.bool)
    ans[8] = True
    assert np.array_equal(o, ans)
    o = uvutils.or_collapse(data, axis=1)
    ans = np.zeros(50, np.bool)
    ans[0] = True
    assert np.array_equal(o, ans)
    o = uvutils.or_collapse(data)
    assert o


def test_or_collapse_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool)
    data[0, 8] = True
    w = np.ones_like(data, np.float)
    o, wo = uvutils.or_collapse(data, axis=0, weights=w, return_weights=True)
    ans = np.zeros(25, np.bool)
    ans[8] = True
    assert np.array_equal(o, ans)
    assert np.array_equal(wo, np.ones_like(o, dtype=np.float))
    w[0, 8] = 0.3
    o = uvtest.checkWarnings(uvutils.or_collapse, [data], {'axis': 0, 'weights': w},
                             nwarnings=1, message='Currently weights are')
    assert np.array_equal(o, ans)


def test_or_collapse_errors():
    data = np.zeros(5)
    pytest.raises(ValueError, uvutils.or_collapse, data)


def test_and_collapse():
    # Fake data
    data = np.zeros((50, 25), np.bool)
    data[0, :] = True
    o = uvutils.and_collapse(data, axis=0)
    ans = np.zeros(25, np.bool)
    assert np.array_equal(o, ans)
    o = uvutils.and_collapse(data, axis=1)
    ans = np.zeros(50, np.bool)
    ans[0] = True
    assert np.array_equal(o, ans)
    o = uvutils.and_collapse(data)
    assert not o


def test_and_collapse_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool)
    data[0, :] = True
    w = np.ones_like(data, np.float)
    o, wo = uvutils.and_collapse(data, axis=0, weights=w, return_weights=True)
    ans = np.zeros(25, np.bool)
    assert np.array_equal(o, ans)
    assert np.array_equal(wo, np.ones_like(o, dtype=np.float))
    w[0, 8] = 0.3
    o = uvtest.checkWarnings(uvutils.and_collapse, [data], {'axis': 0, 'weights': w},
                             nwarnings=1, message='Currently weights are')
    assert np.array_equal(o, ans)


def test_and_collapse_errors():
    data = np.zeros(5)
    pytest.raises(ValueError, uvutils.and_collapse, data)
