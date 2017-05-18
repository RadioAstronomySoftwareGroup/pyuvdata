#!/bin/env python

from pyuvdata.utils import ENU_from_ECEF
from pyuvdata.telescopes import get_telescope
import numpy as np, pylab as p

f = np.load('mwa128_ant_layouts.npz')
xyz = f['ecef']
txt_topo = f['txt_topo']
uvw_topo = f['uvw_topo']

txt_topo -= txt_topo[0]

mwa = get_telescope('mwa')
lat, lon, alt = mwa.telescope_location_lat_lon_alt

enu = ENU_from_ECEF(xyz, lat,lon, alt)

p.scatter(enu[0,:],enu[1,:], marker='.', label='ENU')
p.scatter(txt_topo[:,0],txt_topo[:,1], marker='.', label='text_topo')
p.scatter(-uvw_topo[:,0],-uvw_topo[:,1], marker='.',label='uvw_topo')
p.legend()
p.show()
