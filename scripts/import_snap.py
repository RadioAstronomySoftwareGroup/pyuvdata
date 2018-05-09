'''
Script to import a numpy array written by a snap board and convert to a uvdata
set
'''

import numpy
import yaml
import argparse
from pyuvdata import UVData
from pyuvdata import polstr2num
desc=('Convert SNAP outputs into uvdata sets.'
      'Usage: import_snap.py -c config.yaml -z observation.npz'
      'config.yaml is a .yaml file containing instrument meta-data'
      'observation.npz contains numpy array of data from the SNAP')

parser=argparse.ArgumentParser(description=desc)
parser.add_argument('--config','-c',help='configuration .yaml file')
parser.add_argument('--data','-z',help='data .npz file')

args=parser.parse_args()


with open(parser.config) as configfile:
    config=yaml.load(configfile)

data=np.load(parser.data)

#instatiate uvdata object
data_uv=UVData()
data_uv.Ntimes=len(data['times'])
data_uv.Nbls=len(data['antenna_pairs'])
data_uv.Nblts=data_uv.Ntimes*data_uv.Nbls
data_uv.Nfreqs=len(data['frequencies'])
data_uv.Npols=len(data['polarizations'])
data_uv.vis_units='uncalib'
data_uv.Nspws=1 #!!!Assume a single spectral window.!!!
#instantiate flag array with no flags
data_uv.flag_array=np.empty((data_uv.Nblts,data_uv.Nspws,
                             data_uv.Nfreqs,data_uv.Npols,dtype=bool)
data_uv.flag_array[:]=False
#instantiate nsamples array with equal unity weights.
data_uv.nsample_array=np.ones(data_uv.Nblts,data_uv.Nspws,
                              data_uv.Nfreqs,data_uv.Npols,dtype=float)

#create frequency array
data_uv.freq_array=np.zeros((self.Nspws,self.Nfreqs))
data_uv.freq_array[0,:]=data['frequencies']
#channel width
data_uv.channel_width=data_uv.freq_array[0,1]-data_uv.freq_array[0,0]
#convert to Nblt ordering
data_uv.data_array
#create ant_1_array
#create ant_2_array
#create baseline_array
#create polarization array
data_uv.polarization_array=\
np.array([polstr2num(config['POLARIZATIONS'][p])\
 for p in len(config['POLARIZATIONS'])]).astype(int)
#set telescope location
data_uv.telescope_location=np.array(config['TELESCOPE_LOCATION_ITRF'])

#create time array
#convert times to julian days
