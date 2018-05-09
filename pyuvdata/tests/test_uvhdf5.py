"""Tests for HDF5 object"""
import os
import copy
import numpy as np
import nose.tools as nt
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH
import pyuvdata.tests as uvtest
import warnings


def test_ReadMiriadWriteUVhdf5ReadUVhdf5():
    """
    Read in a miriad file, write it out, read back in, and
    check for object equality.
    """
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test', 'outtest_miriad.hdf5')
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         nwarnings=1, category=[UserWarning],
                         message=['Altitude is not present'])
    uv_in.write_uvhdf5(testfile, clobber=True)
    uv_out.read_uvhdf5(testfile)
    nt.assert_equal(uv_in, uv_out)
