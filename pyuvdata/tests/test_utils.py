# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for common utility functions.

"""
from __future__ import absolute_import, division, print_function

import os
import nose.tools as nt
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
    nt.assert_true(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))

    # test error checking
    nt.assert_raises(ValueError, uvutils.XYZ_from_LatLonAlt, ref_latlonalt[0],
                     ref_latlonalt[1], np.array([ref_latlonalt[2], ref_latlonalt[2]]))
    nt.assert_raises(ValueError, uvutils.XYZ_from_LatLonAlt, ref_latlonalt[0],
                     np.array([ref_latlonalt[1], ref_latlonalt[1]]), ref_latlonalt[2])


def test_LatLonAlt_from_XYZ():
    """Test conversion from ECEF xyz to lat/lon/alt with reference values."""
    out_latlonalt = uvutils.LatLonAlt_from_XYZ(ref_xyz)
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    nt.assert_true(np.allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3))
    nt.assert_raises(ValueError, uvutils.LatLonAlt_from_XYZ, ref_latlonalt)

    # test passing multiple values
    xyz_mult = np.stack((np.array(ref_xyz), np.array(ref_xyz)))
    lat_vec, lon_vec, alt_vec = uvutils.LatLonAlt_from_XYZ(xyz_mult)
    nt.assert_true(np.allclose(ref_latlonalt, (lat_vec[1], lon_vec[1], alt_vec[1]), rtol=0, atol=1e-3))
    # check warning if array transposed
    uvtest.checkWarnings(uvutils.LatLonAlt_from_XYZ, [xyz_mult.T],
                         message='The expected shape of ECEF xyz array',
                         category=PendingDeprecationWarning)
    # check warning if  3 x 3 array
    xyz_3 = np.stack((np.array(ref_xyz), np.array(ref_xyz), np.array(ref_xyz)))
    uvtest.checkWarnings(uvutils.LatLonAlt_from_XYZ, [xyz_3],
                         message='The xyz array in LatLonAlt_from_XYZ is',
                         category=PendingDeprecationWarning)
    # check error if only 2 coordinates
    nt.assert_raises(ValueError, uvutils.LatLonAlt_from_XYZ, xyz_mult[:, 0:2])

    # test error checking
    nt.assert_raises(ValueError, uvutils.LatLonAlt_from_XYZ, ref_xyz[0:1])


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
    nt.assert_true(np.allclose(np.stack((x, y, z), axis=1), xyz, atol=1e-3))

    enu = uvutils.ENU_from_ECEF(xyz, center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(np.stack((east, north, up), axis=1), enu, atol=1e-3))
    # check warning if array transposed
    uvtest.checkWarnings(uvutils.ENU_from_ECEF, [xyz.T, center_lat, center_lon,
                                                 center_alt],
                         message='The expected shape of ECEF xyz array',
                         category=PendingDeprecationWarning)
    # check warning if  3 x 3 array
    uvtest.checkWarnings(uvutils.ENU_from_ECEF, [xyz[0:3], center_lat, center_lon,
                                                 center_alt],
                         message='The xyz array in ENU_from_ECEF is',
                         category=PendingDeprecationWarning)
    # check error if only 2 coordinates
    nt.assert_raises(ValueError, uvutils.ENU_from_ECEF, xyz[:, 0:2],
                     center_lat, center_lon, center_alt)

    # check that a round trip gives the original value.
    xyz_from_enu = uvutils.ECEF_from_ENU(enu, center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(xyz, xyz_from_enu, atol=1e-3))
    # check warning if array transposed
    uvtest.checkWarnings(uvutils.ECEF_from_ENU, [enu.T, center_lat, center_lon,
                                                 center_alt],
                         message='The expected shape the ENU array',
                         category=PendingDeprecationWarning)
    # check warning if  3 x 3 array
    uvtest.checkWarnings(uvutils.ECEF_from_ENU, [enu[0:3], center_lat, center_lon,
                                                 center_alt],
                         message='The enu array in ECEF_from_ENU is',
                         category=PendingDeprecationWarning)
    # check error if only 2 coordinates
    nt.assert_raises(ValueError, uvutils.ENU_from_ECEF, enu[:, 0:2], center_lat,
                     center_lon, center_alt)

    # check passing a single value
    enu_single = uvutils.ENU_from_ECEF(xyz[0, :], center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(np.array((east[0], north[0], up[0])), enu[0, :], atol=1e-3))

    xyz_from_enu = uvutils.ECEF_from_ENU(enu_single, center_lat, center_lon, center_alt)
    nt.assert_true(np.allclose(xyz[0, :], xyz_from_enu, atol=1e-3))

    # error checking
    nt.assert_raises(ValueError, uvutils.ENU_from_ECEF, xyz[:, 0:1], center_lat, center_lon, center_alt)
    nt.assert_raises(ValueError, uvutils.ECEF_from_ENU, enu[:, 0:1], center_lat, center_lon, center_alt)
    nt.assert_raises(ValueError, uvutils.ENU_from_ECEF, xyz / 2., center_lat, center_lon, center_alt)


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

    nt.assert_true(np.allclose(enu, enh))

    # test other direction of ECEF rotation
    rot_xyz = uvutils.rotECEF_from_ECEF(new_xyz, lon)
    nt.assert_true(np.allclose(rot_xyz.T, xyz))


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
                                  representation='cartesian', frame='itrs',
                                  obstime=obs_time)

    itrs_coord = SkyCoord(x=ant_xyz_abs[0] * units.m,
                          y=ant_xyz_abs[1] * units.m,
                          z=ant_xyz_abs[2] * units.m,
                          representation='cartesian', frame='itrs',
                          obstime=obs_time)

    gcrs_array_center = array_center_coord.transform_to('gcrs')
    gcrs_from_itrs_coord = itrs_coord.transform_to('gcrs')

    gcrs_rel = (gcrs_from_itrs_coord.cartesian - gcrs_array_center.cartesian).get_xyz().T

    gcrs_uvw = uvutils.phase_uvw(gcrs_coord.ra.rad, gcrs_coord.dec.rad,
                                 gcrs_rel.value)

    mwa_tools_calcuvw_u = -97.122828
    mwa_tools_calcuvw_v = 50.388281
    mwa_tools_calcuvw_w = -151.27976

    nt.assert_true(np.allclose(gcrs_uvw[0, 0], mwa_tools_calcuvw_u, atol=1e-3))
    nt.assert_true(np.allclose(gcrs_uvw[0, 1], mwa_tools_calcuvw_v, atol=1e-3))
    nt.assert_true(np.allclose(gcrs_uvw[0, 2], mwa_tools_calcuvw_w, atol=1e-3))

    # also test unphasing
    temp2 = uvutils.unphase_uvw(gcrs_coord.ra.rad, gcrs_coord.dec.rad,
                                np.squeeze(gcrs_uvw))
    nt.assert_true(np.allclose(gcrs_rel.value, temp2))


def test_pol_funcs():
    """ Test utility functions to convert between polarization strings and numbers """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    pol_str = ['yx', 'xy', 'yy', 'xx', 'lr', 'rl', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    nt.assert_equal(pol_nums, uvutils.polstr2num(pol_str))
    nt.assert_equal(pol_str, uvutils.polnum2str(pol_nums))
    # Check individuals
    nt.assert_equal(-6, uvutils.polstr2num('YY'))
    nt.assert_equal('pV', uvutils.polnum2str(4))
    # Check errors
    nt.assert_raises(KeyError, uvutils.polstr2num, 'foo')
    nt.assert_raises(ValueError, uvutils.polstr2num, 1)
    nt.assert_raises(ValueError, uvutils.polnum2str, 7.3)
    # Check parse
    nt.assert_equal(uvutils.parse_polstr("xX"), 'xx')
    nt.assert_equal(uvutils.parse_polstr("XX"), 'xx')
    nt.assert_equal(uvutils.parse_polstr('i'), 'pI')
    nt.assert_equal(uvutils.parse_jpolstr('x'), 'Jxx')
    nt.assert_equal(uvutils.parse_jpolstr('xy'), 'Jxy')
    nt.assert_equal(uvutils.parse_jpolstr('XY'), 'Jxy')


def test_jones_num_funcs():
    """ Test utility functions to convert between jones polarization strings and numbers """

    jnums = [-8, -7, -6, -5, -4, -3, -2, -1]
    jstr = ['Jyx', 'Jxy', 'Jyy', 'Jxx', 'Jlr', 'Jrl', 'Jll', 'Jrr']
    nt.assert_equal(jnums, uvutils.jstr2num(jstr))
    nt.assert_equal(jstr, uvutils.jnum2str(jnums))
    # Check shorthands
    jstr = ['yx', 'xy', 'yy', 'y', 'xx', 'x', 'lr', 'rl', 'll', 'l', 'rr', 'r']
    jnums = [-8, -7, -6, -6, -5, -5, -4, -3, -2, -2, -1, -1]
    nt.assert_equal(jnums, uvutils.jstr2num(jstr))
    # Check individuals
    nt.assert_equal(-6, uvutils.jstr2num('jyy'))
    nt.assert_equal('Jxy', uvutils.jnum2str(-7))
    # Check errors
    nt.assert_raises(KeyError, uvutils.jstr2num, 'foo')
    nt.assert_raises(ValueError, uvutils.jstr2num, 1)
    nt.assert_raises(ValueError, uvutils.jnum2str, 7.3)


def test_conj_pol():
    """ Test function to conjugate pols """

    pol_nums = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4]
    cpol_nums = [-7, -8, -6, -5, -3, -4, -2, -1, 1, 2, 3, 4]
    nt.assert_equal(pol_nums, uvutils.conj_pol(cpol_nums))
    nt.assert_equal(uvutils.conj_pol(pol_nums), cpol_nums)
    pol_str = ['yx', 'xy', 'yy', 'xx', 'lr', 'rl', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    cpol_str = ['xy', 'yx', 'yy', 'xx', 'rl', 'lr', 'll', 'rr', 'pI', 'pQ', 'pU', 'pV']
    nt.assert_equal(pol_str, uvutils.conj_pol(cpol_str))
    nt.assert_equal(uvutils.conj_pol(pol_str), cpol_str)
    nt.assert_equal([pol_str, pol_nums], uvutils.conj_pol([cpol_str, cpol_nums]))

    jstr = ['Jyx', 'Jxy', 'Jyy', 'Jxx', 'Jlr', 'Jrl', 'Jll', 'Jrr']
    cjstr = ['Jxy', 'Jyx', 'Jyy', 'Jxx', 'Jrl', 'Jlr', 'Jll', 'Jrr']
    conj_cjstr = uvtest.checkWarnings(uvutils.conj_pol, [cjstr], nwarnings=8,
                                      category=PendingDeprecationWarning,
                                      message='conj_pol should not be called with jones')
    nt.assert_equal(jstr, conj_cjstr)

    # Test invalid pol
    nt.assert_raises(ValueError, uvutils.conj_pol, 2.3)


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
    uvd.read_uvfits(os.path.join(DATA_PATH, 'hera19_8hrs_uncomp_10MHz_000_05.003111-05.033750.uvfits'))
    uvd.select(times=uvd.time_array[0])
    uvd.unphase_to_drift(use_ant_pos=True)   # uvw_array is now equivalent to baseline positions

    tol = 0.05  # meters

    bl_positions = uvd.uvw_array

    nt.assert_raises(ValueError, uvutils.get_baseline_redundancies, uvd.baseline_array, bl_positions[0:2, 0:1])
    baseline_groups, vec_bin_centers, lens = uvutils.get_baseline_redundancies(uvd.baseline_array, bl_positions, tol=tol)

    baseline_groups, vec_bin_centers, lens = uvutils.get_baseline_redundancies(uvd.baseline_array, bl_positions, tol=tol)

    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            nt.assert_true(np.allclose(np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])), lens[gi], atol=tol))

    # Now jostle the baselines around by up to 0.25m and see if we can recover the same redundancies to that tolerance.
    tol = 0.25  # meters. Less than the smallest baseline in the file.
    Nbls = uvd.Nbls
    shift_dists = np.random.uniform(low=0.0, high=tol / 2., size=Nbls)
    shift_angs = np.random.uniform(low=0.0, high=2 * np.pi, size=Nbls)
    shift_vecs = np.stack((shift_dists * np.cos(shift_angs), shift_dists * np.sin(shift_angs), np.zeros(Nbls))).T

    bl_positions_new = uvd.uvw_array + shift_vecs

    baseline_groups_new, vec_bin_centers, lens = uvutils.get_baseline_redundancies(uvd.baseline_array, bl_positions_new, tol=tol)

    for gi, gp in enumerate(baseline_groups_new):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            nt.assert_true(np.allclose(np.sqrt(np.abs(np.dot(bl_vec, vec_bin_centers[gi]))), lens[gi], atol=tol))

    # Compare baseline groups:
    for c, blg in enumerate(baseline_groups):
        bl = blg[0]
        ind = np.sum(np.where([bl in gp for gp in baseline_groups_new]))
        nt.assert_equal(baseline_groups_new[ind], blg)

    tol = 0.05

    antpos, antnums = uvd.get_ENU_antpos()
    baseline_groups_ants, vec_bin_centers, lens = uvutils.get_antenna_redundancies(antnums, antpos,
                                                                                   tol=tol, include_autos=True)
    # Under these conditions, should see 31 redundant groups in the file.
    nt.assert_equal(len(baseline_groups_ants), 31)

    # Check with conjugated baseline redundancies returned
    u16_0 = bl_positions[16, 0]
    bl_positions[16, 0] = 0                 # Ensure at least one baseline has u==0 and v!=0 (for coverage of this case)
    baseline_groups, vec_bin_centers, lens, conjugates = uvutils.get_baseline_redundancies(uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True)

    # restore baseline (16,0) and repeat to get correct groups
    bl_positions[16, 0] = u16_0
    baseline_groups, vec_bin_centers, lens, conjugates = uvutils.get_baseline_redundancies(uvd.baseline_array, bl_positions, tol=tol, with_conjugates=True)

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
    nt.assert_true(np.all(sorted(baseline_groups_ants) == sorted(baseline_groups_flipped)))
    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            if bl in conjugates:
                bl_vec *= (-1)
            nt.assert_true(np.isclose(np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])), lens[gi], atol=tol))


def test_reraise_context():
    with nt.assert_raises(ValueError) as cm:
        try:
            uvutils.LatLonAlt_from_XYZ(ref_xyz[0:1])
        except ValueError:
            uvutils._reraise_context('Add some info')
    ex = cm.exception
    nt.assert_equal(ex.args[0], 'Add some info: xyz values should be ECEF x, y, z coordinates in meters')

    with nt.assert_raises(ValueError) as cm:
        try:
            uvutils.LatLonAlt_from_XYZ(ref_xyz[0:1])
        except ValueError:
            uvutils._reraise_context('Add some info %s', 'and then more')
    ex = cm.exception
    nt.assert_equal(ex.args[0], 'Add some info and then more: xyz values should be ECEF x, y, z coordinates in meters')

    with nt.assert_raises(EnvironmentError) as cm:
        try:
            raise EnvironmentError(1, 'some bad problem')
        except EnvironmentError:
            uvutils._reraise_context('Add some info')
    ex = cm.exception
    nt.assert_equal(ex.args[1], 'Add some info: some bad problem')


def test_str_to_bytes():
    test_str = 'HERA'
    test_bytes = uvutils._str_to_bytes(test_str)
    nt.assert_equal(type(test_bytes), six.binary_type)
    nt.assert_equal(test_bytes, b'\x48\x45\x52\x41')
    return


def test_bytes_to_str():
    test_bytes = b'\x48\x45\x52\x41'
    test_str = uvutils._bytes_to_str(test_bytes)
    nt.assert_equal(type(test_str), str)
    nt.assert_equal(test_str, 'HERA')
    return
