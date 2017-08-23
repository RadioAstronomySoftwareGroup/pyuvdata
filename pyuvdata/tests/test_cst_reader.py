import nose.tools as nt
import os, sys
import numpy as np
from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam

def test_read():
    beams = UVBeam()
    files_use = ['HERA_NicCST_150MHz.txt', 'HERA_NicCST_123MHz.txt']
    file_paths = [os.path.join(DATA_PATH, f) for f in files_use]

    beams.read_cst_power(file_paths, 'peak')

    nt.assert_true(beams.check())

    del beams

    beams = UVBeam()

    # test the bit about checking if the input is a list/tuple or not
    fuse = file_paths[0]

    beams.read_cst_power(fuse, 'peak')

    nt.assert_true(beams.check())
