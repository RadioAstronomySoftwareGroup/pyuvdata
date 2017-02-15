#!/usr/bin/env python
import aipy as a
import numpy as np
import optparse
import sys
import os
import pyuvdata
from matplotlib.pyplot import *
o = optparse.OptionParser()
o.set_usage('miriad_to_uvfits.py  *.uv')
o.set_description(__doc__)
# a.scripting.add_standard_options(o,cal=True)
opts, args = o.parse_args(sys.argv[1:])

for filename in args:
    UV = pyuvdata.UVData()
    UV.read_miriad(filename, 'miriad')
    outfilename = filename + '.uvfits'
    UV.phase_to_time(UV.time_array[0])
    UV.write_uvfits(outfilename, 'uvfits')
show()
