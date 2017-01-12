"""Tests for MS object."""
import nose.tools as nt
import os
from uvdata import UVData
import uvdata.tests as uvtest
import glob as glob
from uvdata.data import DATA_PATH



def test_ReadMSWriteReadUVFits(msname):
    """
    MS to uvfits loopback test.

    Read in MS files, write out as uvfits, read back in and check for object
    equality.
    """
    ms_uv = UVData()
    uvfits_uv = UVData()
    uvfits_uv_orig=UVData()
    uvfits_uv_orig.read_uvfits(DATA_PATH+'/'+msname+'.uvfits')
    ms_uv.read_ms(DATA_PATH+'/'+msname+'.ms')
    ms_uv.order_pols()
    ms_uv.write_uvfits(os.path.join(DATA_PATH, 'outtest_'+msname+'.uvfits'),
                        spoof_nonessential=True)
    ms_uv.order_pols(order='CASA')
    uvfits_uv.read_uvfits(os.path.join(DATA_PATH, 'outtest_'+msname+'.uvfits'))

    
    #print ms_uv.required()
    #for pnum,p in enumerate(ms_uv.required()):
    #    print p
    #    print uvfits_uv.required()
    #print uvfits_uv_orig.polarization_array
    #print uvfits_uv.polarization_array
    #print uvfits_uv_orig.history
    #print uvfits_uv.history
    #print uvfits_uv.integration_time
    #print uvfits_uv_orig.integration_time
    #print uvfits_uv.instrument
    #print uvfits_uv_orig.instrument
    #print uvfits_uv.object_name
    #print uvfits_uv_orig.object_name
    #print uvfits_uv.lst_array
    #print uvfits_uv_orig.lst_array

    #print uvfits_uv.phase_center_epoch
    #print uvfits_uv_orig.phase_center_epoch
    #print uvfits_uv.Nbls
    #print uvfits_uv_orig.Nbls
    #print ms_uv.Nbls
    #print uvfits_uv.polarization_array
    #print uvfits_uv_orig.polarization_array
    #print ms_uv.polarization_array

    #print uvfits_uv.telescope_location
    #print uvfits_uv_orig.telescope_location

    #print uvfits_uv.vis_units
    #print uvfits_uv_orig.vis_units
    ms_uv.order_pols()

    nt.assert_equal(ms_uv, uvfits_uv)
    nt.assert_equal(uvfits_uv,uvfits_uv_orig)
    del(ms_uv)
    del(uvfits_uv)
