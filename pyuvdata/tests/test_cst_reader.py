import nose.tools as nt
import os, sys, glob
import numpy as np, healpy as hp
from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam

def test_read():
    beams = UVBeam()
    files_use = glob.glob(os.path.join(DATA_PATH) + 'HERA_NicCST*.txt')

    beams.read_cst_power(files_use, 'peak')

    nt.assert_true(beams.check())
