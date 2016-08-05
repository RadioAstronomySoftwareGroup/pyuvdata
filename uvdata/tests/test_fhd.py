import nose.tools as nt
import os.path as op
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
import uvdata.tests as uvtest


testdir = '../data/fhd_vis_data/'
testfile_prefix = '1061316296_'
testfile_suffix = ['flags.sav', 'vis_XX.sav', 'params.sav', 'vis_YY.sav',
                   'vis_model_XX.sav', 'vis_model_YY.sav', 'settings.txt']
testfiles = []
for s in testfile_suffix:
    testfiles.append(testdir + testfile_prefix + s)


def test_ReadFHDWriteReadUVFits():
    fhd_uv = UVData()
    uvfits_uv = UVData()
    fhd_uv.read(testfiles, 'fhd')
    fhd_uv.write(op.join('../data/test/outtest_FHD_1061316296.uvfits'),
                 file_type='uvfits', spoof_nonessential=True)
    uvfits_uv.read(op.join('../data/test/outtest_FHD_1061316296.uvfits'), 'uvfits')
    nt.assert_equal(fhd_uv, uvfits_uv)
    del(fhd_uv)
    del(uvfits_uv)


def test_breakReadFHD():
    # Try various cases of incomplete file lists
    fhd_uv = UVData()
    nt.assert_raises(StandardError, fhd_uv.read, testfiles[1:], 'fhd')  # Missing flags
    del(fhd_uv)
    fhd_uv = UVData()
    subfiles = [item for sublist in [testfiles[0:2], testfiles[3:]] for item in sublist]
    nt.assert_raises(StandardError, fhd_uv.read, subfiles, 'fhd')  # Missing params
    del(fhd_uv)
    fhd_uv = UVData()
    nt.assert_raises(StandardError, fhd_uv.read, ['foo'], 'fhd')  # No data files
    del(fhd_uv)
    fhd_uv = UVData()
    nt.assert_true(uvtest.checkWarnings(fhd_uv.read, [testfiles[:-1],
                                        'fhd'], message=['No settings'])[1])
    nt.assert_equal(fhd_uv.history, '')  # Check empty history with no settings
    del(fhd_uv)


def test_ReadFHD_model():
    fhd_uv = UVData()
    uvfits_uv = UVData()
    fhd_uv.read(testfiles, 'fhd')
    fhd_uv.write('../data/test/outtest_FHD_1061316296_model.uvfits',
                 file_type='uvfits', spoof_nonessential=True)
    uvfits_uv.read('../data/test/outtest_FHD_1061316296_model.uvfits', 'uvfits')
    nt.assert_equal(fhd_uv, uvfits_uv)
    del(fhd_uv)
    del(uvfits_uv)
