import nose.tools as nt
import os, sys
import numpy as np, healpy as hp
from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam

def test_read():
    beams = UVBeam()
    files_use = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
    file_paths = [os.path.join(DATA_PATH, f) for f in files_use]

    beams.read_cst_power(file_paths, 'peak')

    nt.assert_true(beams.check())
