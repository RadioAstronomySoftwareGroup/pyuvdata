#!/usr/bin/env python
import aipy as a, numpy as np
import optparse, sys, os
import uvdata
from matplotlib.pyplot import *
o = optparse.OptionParser()
o.set_usage('miriad_to_uvfits.py  *.uv')
o.set_description(__doc__)
#a.scripting.add_standard_options(o,cal=True)
opts,args = o.parse_args(sys.argv[1:])

for filename in args:
    UV = uvdata.uv.UVData()
    UV.read_miriad(filename,'miriad')
    outfilename = filename+'.uvfits'
    UV.phase(time=UV.time_array[0])
    UV.write_uvfits(outfilename,'uvfits')
show()

