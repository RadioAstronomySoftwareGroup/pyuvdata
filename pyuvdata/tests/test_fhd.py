# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for FHD object.

"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import glob
import numpy as np
from shutil import copyfile

from pyuvdata import UVData
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

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
    uvtest.checkWarnings(fhd_uv.read, [testfiles], known_warning='fhd')

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    assert fhd_uv == uvfits_uv


def test_ReadFHD_select():
    """
    test select on read with FHD files.

    Read in FHD files with generic read & select on read, compare to read fhd
    files then do select
    """
    fhd_uv = UVData()
    fhd_uv2 = UVData()
    uvtest.checkWarnings(fhd_uv2.read, [testfiles], {'freq_chans': np.arange(2)},
                         message=['Warning: select on read keyword set',
                                  'Telescope location derived from obs'],
                         nwarnings=2)

    uvtest.checkWarnings(fhd_uv.read, [testfiles], known_warning='fhd')

    fhd_uv.select(freq_chans=np.arange(2))
    assert fhd_uv == fhd_uv2


def test_ReadFHDWriteReadUVFits_no_layout():
    """
    FHD to uvfits loopback test with no layout file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    files_use = testfiles[:-3] + [testfiles[-2]]
    uvtest.checkWarnings(fhd_uv.read_fhd, [files_use], nwarnings=2,
                         message=['No layout file', 'antenna_positions are not defined'],
                         category=DeprecationWarning)

    uvtest.checkWarnings(fhd_uv.write_uvfits,
                         [os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits')],
                         func_kwargs={"spoof_nonessential": True},
                         nwarnings=1,
                         message=['antenna_positions are not defined'],
                         category=DeprecationWarning
                         )
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))

    assert fhd_uv == uvfits_uv


def test_ReadFHDWriteReadUVFits_variant_flag():
    """
    FHD to uvfits loopback test with variant flag file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    variant_flag_file = testdir + testfile_prefix + 'variant_flags.sav'
    files_use = testfiles[1:] + [variant_flag_file]
    uvtest.checkWarnings(fhd_uv.read, [files_use], known_warning='fhd')

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    assert fhd_uv == uvfits_uv


def test_ReadFHDWriteReadUVFits_fix_layout():
    """
    FHD to uvfits loopback test with fixed array center layout file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    layout_fixed_file = testdir + testfile_prefix + 'fixed_arr_center_layout.sav'
    files_use = testfiles[0:6] + [layout_fixed_file, testfiles[7]]
    fhd_uv.read(files_use)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    assert fhd_uv == uvfits_uv


def test_ReadFHDWriteReadUVFits_fix_layout_bad_obs_loc():
    """
    FHD to uvfits loopback test with fixed array center layout file, bad obs location.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    bad_obs_loc_file = testdir + testfile_prefix + 'bad_obs_loc_vis_XX.sav'
    layout_fixed_file = testdir + testfile_prefix + 'fixed_arr_center_layout.sav'
    files_use = [testfiles[0], testfiles[2], bad_obs_loc_file,
                 layout_fixed_file, testfiles[7]]
    messages = ['Telescope location derived from obs',
                'tile_names from obs structure does not match']
    uvtest.checkWarnings(fhd_uv.read_fhd, [files_use], message=messages,
                         nwarnings=2)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    assert fhd_uv == uvfits_uv


def test_ReadFHDWriteReadUVFits_bad_obs_loc():
    """
    FHD to uvfits loopback test with bad obs location (and bad layout location).

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    bad_obs_loc_file = testdir + testfile_prefix + 'bad_obs_loc_vis_XX.sav'
    files_use = [testfiles[0], testfiles[2], bad_obs_loc_file,
                 testfiles[6], testfiles[7]]
    messages = ['Telescope location derived from obs',
                'tile_names from obs structure does not match']
    uvtest.checkWarnings(fhd_uv.read_fhd, [files_use], message=messages,
                         nwarnings=2)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    assert fhd_uv == uvfits_uv


def test_ReadFHDWriteReadUVFits_altered_layout():
    """
    FHD to uvfits loopback test with altered layout file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()

    # bad layout structure values
    altered_layout_file = testdir + testfile_prefix + 'broken_layout.sav'
    files_use = testfiles[0:6] + [altered_layout_file, testfiles[7]]
    fhd_uv.read(files_use)

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    assert fhd_uv == uvfits_uv


def test_ReadFHDWriteReadUVFits_no_settings():
    """
    FHD to uvfits loopback test with no settings file.

    Read in FHD files, write out as uvfits, read back in and check for object
    equality.
    """
    fhd_uv = UVData()
    uvfits_uv = UVData()
    messages = ['No settings', 'Telescope location derived from obs']
    uvtest.checkWarnings(fhd_uv.read_fhd, [testfiles[:-2]], message=messages,
                         nwarnings=2)

    # Check only pyuvdata history with no settings file
    assert fhd_uv.history == fhd_uv.pyuvdata_version_str  # Check empty history with no settings

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296.uvfits'))
    assert fhd_uv == uvfits_uv


def test_breakReadFHD():
    """Try various cases of incomplete file lists."""
    fhd_uv = UVData()
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles[1:])  # Missing flags
    del(fhd_uv)
    fhd_uv = UVData()
    subfiles = [item for sublist in [testfiles[0:2], testfiles[3:]] for item in sublist]
    pytest.raises(Exception, fhd_uv.read_fhd, subfiles)  # Missing params
    del(fhd_uv)
    fhd_uv = UVData()
    pytest.raises(Exception, fhd_uv.read_fhd, ['foo'])  # No data files
    del(fhd_uv)

    # test warnings with various broken inputs
    # bad obs structure values
    broken_data_file = testdir + testfile_prefix + 'broken_vis_XX.sav'
    bad_filelist = [testfiles[0], testfiles[2],
                    broken_data_file, testfiles[6], testfiles[7]]
    warn_messages = ['Ntimes does not match', 'Nbls does not match',
                     'These visibilities may have been phased improperly',
                     'Telescope location derived from obs',
                     'Telescope foo is not in known_telescopes.']
    fhd_uv = UVData()
    uvtest.checkWarnings(fhd_uv.read_fhd, [bad_filelist], {'run_check': False},
                         nwarnings=5, message=warn_messages)

    # bad flag file
    broken_flag_file = testdir + testfile_prefix + 'broken_flags.sav'
    bad_filelist = testfiles[1:] + [broken_flag_file]
    fhd_uv = UVData()
    pytest.raises(ValueError, fhd_uv.read_fhd, bad_filelist)

    # try cases with extra files of each type
    extra_xx_file = testdir + testfile_prefix + 'extra_vis_XX.sav'
    copyfile(testfiles[1], extra_xx_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [extra_xx_file])
    os.remove(extra_xx_file)

    extra_yy_file = testdir + testfile_prefix + 'extra_vis_YY.sav'
    copyfile(testfiles[3], extra_yy_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [extra_yy_file])
    os.remove(extra_yy_file)

    xy_file = testdir + testfile_prefix + 'vis_XY.sav'
    extra_xy_file = testdir + testfile_prefix + 'extra_vis_XY.sav'
    copyfile(testfiles[1], xy_file)
    copyfile(testfiles[1], extra_xy_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [xy_file, extra_xy_file])
    os.remove(xy_file)
    os.remove(extra_xy_file)

    yx_file = testdir + testfile_prefix + 'vis_YX.sav'
    extra_yx_file = testdir + testfile_prefix + 'extra_vis_YX.sav'
    copyfile(testfiles[1], yx_file)
    copyfile(testfiles[1], extra_yx_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [yx_file, extra_yx_file])
    os.remove(yx_file)
    os.remove(extra_yx_file)

    extra_params_file = testdir + testfile_prefix + 'extra_params.sav'
    copyfile(testfiles[2], extra_params_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [extra_params_file])
    os.remove(extra_params_file)

    extra_flags_file = testdir + testfile_prefix + 'extra_flags.sav'
    copyfile(testfiles[0], extra_flags_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [extra_flags_file])
    os.remove(extra_flags_file)

    extra_layout_file = testdir + testfile_prefix + 'extra_layout.sav'
    copyfile(testfiles[6], extra_layout_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [extra_layout_file])
    os.remove(extra_layout_file)

    extra_settings_file = testdir + testfile_prefix + 'extra_settings.txt'
    copyfile(testfiles[7], extra_settings_file)
    pytest.raises(Exception, fhd_uv.read_fhd, testfiles + [extra_settings_file])
    os.remove(extra_settings_file)


def test_ReadFHD_model():
    """FHD to uvfits loopback test with model visibilities."""
    fhd_uv = UVData()
    uvfits_uv = UVData()
    uvtest.checkWarnings(fhd_uv.read, [testfiles], {'use_model': True}, known_warning='fhd')

    fhd_uv.write_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296_model.uvfits'),
                        spoof_nonessential=True)
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'test/outtest_FHD_1061316296_model.uvfits'))
    assert fhd_uv == uvfits_uv


def test_multi_files():
    """
    Reading multiple files at once.
    """
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    test1 = list(np.array(testfiles)[[0, 1, 2, 4, 6, 7]])
    test2 = list(np.array(testfiles)[[0, 2, 3, 5, 6, 7]])
    uvtest.checkWarnings(fhd_uv1.read, [[test1, test2]], {'use_model': True},
                         message=['Telescope location derived from obs'],
                         nwarnings=2)

    uvtest.checkWarnings(fhd_uv2.read, [testfiles], {'use_model': True}, known_warning='fhd')

    assert uvutils._check_histories(fhd_uv2.history + ' Combined data '
                                    'along polarization axis using pyuvdata.',
                                    fhd_uv1.history)

    fhd_uv1.history = fhd_uv2.history
    assert fhd_uv1 == fhd_uv2


def test_multi_files_axis():
    """
    Reading multiple files at once with axis keyword.
    """
    fhd_uv1 = UVData()
    fhd_uv2 = UVData()
    test1 = list(np.array(testfiles)[[0, 1, 2, 4, 6, 7]])
    test2 = list(np.array(testfiles)[[0, 2, 3, 5, 6, 7]])
    uvtest.checkWarnings(fhd_uv1.read, [[test1, test2]],
                         {'use_model': True, 'axis': 'polarization'},
                         message=['Telescope location derived from obs'],
                         nwarnings=2)

    uvtest.checkWarnings(fhd_uv2.read, [testfiles], {'use_model': True},
                         known_warning='fhd')

    assert uvutils._check_histories(fhd_uv2.history + ' Combined data '
                                    'along polarization axis using pyuvdata.',
                                    fhd_uv1.history)

    fhd_uv1.history = fhd_uv2.history
    assert fhd_uv1 == fhd_uv2


def test_single_time():
    """
    test reading in a file with a single time.
    """
    single_time_filelist = glob.glob(os.path.join(DATA_PATH, 'refsim1.1_fhd/*'))

    fhd_uv = UVData()
    uvtest.checkWarnings(fhd_uv.read, [single_time_filelist],
                         message=['Telescope gaussian is not in known_telescopes.'])

    assert np.unique(fhd_uv.time_array).size == 1


def test_conjugation():
    """ test uvfits vs fhd conjugation """
    uvfits_file = os.path.join(DATA_PATH, 'ref_1.1_uniform.uvfits')
    fhd_filelist = glob.glob(os.path.join(DATA_PATH, 'refsim1.1_fhd/*'))

    uvfits_uv = UVData()
    uvfits_uv.read(uvfits_file)

    fhd_uv = UVData()
    uvtest.checkWarnings(fhd_uv.read, [fhd_filelist],
                         message=['Telescope gaussian is not in known_telescopes.'])

    uvfits_uv.select(polarizations=fhd_uv.polarization_array)

    assert uvfits_uv._uvw_array == fhd_uv._uvw_array
    assert uvfits_uv._data_array == fhd_uv._data_array
