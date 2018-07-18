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
testfile_suffix = ['flags.sav', 'vis_XX.sav', 'params.sav', 'vis_YY.sav',
                   'vis_model_XX.sav', 'vis_model_YY.sav', 'settings.txt']
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
        fhd_uv.read(testfiles)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        uvtest.checkWarnings(fhd_uv.read, [testfiles],
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    nt.assert_equal(fhd_uv, uvfits_uv)

    # check that a select on read works
    fhd_uv2 = UVData()
    uvtest.checkWarnings(fhd_uv2.read, [testfiles], {'freq_chans': np.arange(2)},
                         message='Warning: select on read keyword set')

    if not uvtest.scipy_warnings:
        fhd_uv.read(testfiles)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        uvtest.checkWarnings(fhd_uv.read, [testfiles],
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)
    fhd_uv.select(freq_chans=np.arange(2))
    nt.assert_equal(fhd_uv, fhd_uv2)

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
    if not uvtest.scipy_warnings:
        uvtest.checkWarnings(fhd_uv.read_fhd, [testfiles[:-1]], message=['No settings'])
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        warn_list = ['No settings'] + scipy_warn_list
        category_list = [UserWarning] + scipy_category_list
        uvtest.checkWarnings(fhd_uv.read_fhd, [testfiles[:-1]],
                             message=warn_list, category=category_list,
                             nwarnings=n_scipy_warnings + 1)
    # Check only pyuvdata history with no settings file
    nt.assert_equal(fhd_uv.history, fhd_uv.pyuvdata_version_str)  # Check empty history with no settings
    del(fhd_uv)


def test_ReadFHD_model():
    """FHD to uvfits loopback test with model visibilities."""
    fhd_uv = UVData()
    uvfits_uv = UVData()
    if not uvtest.scipy_warnings:
        fhd_uv.read(testfiles, use_model=True)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        uvtest.checkWarnings(fhd_uv.read, [testfiles], {'use_model': True},
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)

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
    test1 = list(np.array(testfiles)[[0, 1, 2, 4, 6]])
    test2 = list(np.array(testfiles)[[0, 2, 3, 5, 6]])
    if not uvtest.scipy_warnings:
        fhd_uv1.read([test1, test2])
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings(n_scipy_warnings=1100)
        uvtest.checkWarnings(fhd_uv1.read, [[test1, test2]],
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)

    if not uvtest.scipy_warnings:
        fhd_uv2.read(testfiles)
    else:
        # numpy 1.14 introduced a new deprecation warning.
        # Should be fixed when the next scipy version comes out.
        # The number of replications of the warning varies some and must be
        # empirically discovered. It it defaults to the most common number.
        n_scipy_warnings, scipy_warn_list, scipy_category_list = uvtest.get_scipy_warnings()
        uvtest.checkWarnings(fhd_uv2.read, [testfiles],
                             message=scipy_warn_list, category=scipy_category_list,
                             nwarnings=n_scipy_warnings)

    nt.assert_true(uvutils._check_histories(fhd_uv2.history + ' Combined data '
                                            'along polarization axis using pyuvdata.',
                                            fhd_uv1.history))

    fhd_uv1.history = fhd_uv2.history
    nt.assert_equal(fhd_uv1, fhd_uv2)
