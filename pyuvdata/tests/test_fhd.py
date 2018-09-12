# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD object.

"""
from __future__ import absolute_import, division, print_function

import nose.tools as nt
import os
from pyuvdata import UVData
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import numpy as np

# set up FHD file list
testdir = os.path.join(DATA_PATH, 'fhd_vis_data/')
testfile_prefix = '1061316296_'
# note: 1061316296_obs.sav isn't used -- it's there to test handling of unneeded files
testfile_suffix = ['flags.sav', 'vis_XX.sav', 'params.sav', 'vis_YY.sav',
                   'vis_model_XX.sav', 'vis_model_YY.sav', 'layout.sav',
                   'settings.txt', 'obs.sav']
testfiles = []
for s in testfile_suffix:
    testfiles.append(testdir + testfile_prefix + s)


def test_ReadFHDWriteReadUVFits():
    """
    FHD to uvfits loopback test.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read, [testfiles], known_warning='fhd')
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = ['Telescope location derived from obs'] + scipy_warn_list
        category_list = [UserWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read, [testfiles],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))

    diff = fhd_uv.lst_array - uvfits_uv.lst_array
    nt.assert_equal(fhd_uv, uvfits_uv)

    # check that a select on read works
    fhd_uv2 = UVData()
    uvtest.checkWarnings(fhd_uv2.read, [testfiles], {'freq_chans': np.arange(2)},
                         message=['Warning: select on read keyword set',
                                  'Telescope location derived from obs'],
                         nwarnings=2)

    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read, [testfiles], known_warning='fhd')
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = ['Telescope location derived from obs'] + scipy_warn_list
        category_list = [UserWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read, [testfiles],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)
    fhd_uv.select(freq_chans=np.arange(2))
    nt.assert_equal(fhd_uv, fhd_uv2)

    # check loopback (and warning) with no layout file
    files_use = testfiles[:-3] + [testfiles[-2]]
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read_fhd, [files_use],
                             message=['No layout file'], category=DeprecationWarning)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = ['No layout file'] + scipy_warn_list
        category_list = [DeprecationWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read_fhd, [files_use],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    nt.assert_equal(fhd_uv, uvfits_uv)

    # check loopback with variant flag file
    variant_flag_file = testdir + testfile_prefix + 'variant_flags.sav'
    files_use = testfiles[1:] + [variant_flag_file]
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read, [files_use], known_warning='fhd')
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = ['Telescope location derived from obs'] + scipy_warn_list
        category_list = [UserWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read, [files_use],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)

    del(fhd_uv)
    del(uvfits_uv)


def test_breakReadFHD():
    """Try various cases of incomplete file lists."""
    fhd_uv = UVData()
    nt.assert_raises(Exception, fhd_uv.read_fhd, testfiles[1:])  # Missing flags
    del(fhd_uv)
    fhd_uv = UVData()
    subfiles = [item for sublist in [testfiles[0:2], testfiles[3:]] for item in sublist]
    nt.assert_raises(Exception, fhd_uv.read_fhd, subfiles)  # Missing params
    del(fhd_uv)
    fhd_uv = UVData()
    nt.assert_raises(Exception, fhd_uv.read_fhd, ['foo'])  # No data files
    del(fhd_uv)
    fhd_uv = UVData()
    messages = ['No settings', 'Telescope location derived from obs']
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read_fhd, [testfiles[:-2]], message=messages,
                             nwarnings=2)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = messages + scipy_warn_list
        category_list = [UserWarning] * 2 + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read_fhd, [testfiles[:-2]],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 2)
    # Check only pyuvdata history with no settings file
    nt.assert_equal(fhd_uv.history, fhd_uv.pyuvdata_version_str)  # Check empty history with no settings
    del(fhd_uv)

    # test with various broken inputs
    broken_data_file = testdir + testfile_prefix + 'broken_vis_XX.sav'
    bad_filelist = [testfiles[0], testfiles[2],
                    broken_data_file, testfiles[6], testfiles[7]]
    warn_messages = ['Ntimes does not match', 'Nbls does not match',
                     'These visibilities may have been phased improperly',
                     'Telescope location derived from obs',
                     'tile_names from obs structure does not match',
                     'Telescope foo is not in known_telescopes.']
    fhd_uv = UVData()
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read_fhd, [bad_filelist], {'run_check': False},
                             nwarnings=6, message=warn_messages)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = warn_messages + scipy_warn_list
        category_list = [UserWarning] * 6 + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read_fhd, [bad_filelist],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 6)

    broken_layout_file = testdir + testfile_prefix + 'broken_layout.sav'
    bad_filelist = testfiles[0:4] + [broken_layout_file, testfiles[7]]
    warn_messages = ['coordinate_frame keyword in layout file not set']
    fhd_uv = UVData()
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read_fhd, [bad_filelist], {'run_check': False},
                             nwarnings=1, message=warn_messages)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = warn_messages + scipy_warn_list
        category_list = [UserWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read_fhd, [bad_filelist],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)

    broken_flag_file = testdir + testfile_prefix + 'broken_flags.sav'
    bad_filelist = testfiles[1:] + [broken_flag_file]
    fhd_uv = UVData()
    nt.assert_raises(ValueError, fhd_uv.read_fhd, bad_filelist)


def test_ReadFHD_model():
    """FHD to uvfits loopback test with model visibilities."""
    fhd_uv = UVData()
    uvfits_uv = UVData()
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read, [testfiles], {'use_model': True}, known_warning='fhd')
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = ['Telescope location derived from obs'] + scipy_warn_list
        category_list = [UserWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read, [testfiles], {'use_model': True},
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296_model.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296_model.uvfits'))
    nt.assert_equal(fhd_uv, uvfits_uv)
    del(fhd_uv)
    del(uvfits_uv)


def test_multi_files():
    """
    Reading multiple files at once.
    """
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    test1 = list(np.array(testfiles)[[0, 1, 2, 4, 6, 7]])
    test2 = list(np.array(testfiles)[[0, 2, 3, 5, 6, 7]])
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv1.read, [[test1, test2]], {'use_model': True},
                             message=['Telescope location derived from obs'],
                             nwarnings=2)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings(n_scipy_warnings=1100)
        warn_list = ['Telescope location derived from obs'] * 2 + scipy_warn_list
        category_list = [UserWarning] * 2 + scipy_category_list
        uvtest.checkWarnings(fhd_uv1.read, [[test1, test2]],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 2)

    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv2.read, [testfiles], {'use_model': True}, known_warning='fhd')
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = ['Telescope location derived from obs'] + scipy_warn_list
        category_list = [UserWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv2.read, [testfiles],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)

    nt.assert_true(uvutils._check_histories(fhd_uv2.history + ' Combined data '
                                            'along polarization axis using pyuvdata.',
                                            fhd_uv1.history))

    fhd_uv1.history = fhd_uv2.history
    nt.assert_equal(fhd_uv1, fhd_uv2)
