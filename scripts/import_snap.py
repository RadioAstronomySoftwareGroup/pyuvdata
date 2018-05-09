'''
Script to import a numpy array written by a snap board and convert to a uvdata
set
'''

import numpy
import yaml
import argparse
from pyuvdata import UVData
from pyuvdata import polstr2num
from astropy.time import Time
import copy
from astropy.coordinates import SkyCoord,EarthLocation,AltAz
import astropy.units as u
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
#!!!Does the snap board have a 100% duty cycle? Probably not quite...
data_uv.integration_time=data['times'][1]-data['times'][0]
#convert to Nblt ordering
data_uv.data_array=np.zeros((data_uv.Nblts,data_uv.Nspws,data_uv.Nfreqs,data_uv.Npols),
                             dtype=complex)
for tnum in range(data_uv.Ntimes):
    data_uv.data_array[tnum*data_uv.Ntimes:(tnum+1)*data_uv.Ntimes,1,:,:]\
    =data['data'][tnum,:,:,:]
#create ant_1_array
data_uv.ant_1_array=\
np.hstack([data['antenna_pairs'][:,0] for p in range(data_uv.Ntimes)]).flatten()\
.astype(int)
#create ant_2_array
data_uv.ant_2_array=\
np.hstack([data['anetenna_pairs'][:,1] for p in range(data_uv.Ntimes)]).flatten()\
.astype(int)
data_uv.baseline_array=\
(2048*(data_uv.ant_2_array+1)+(data_uv.ant_1_array+1)+2**16).astype(int)
#create baseline_array
#create polarization array
data_uv.polarization_array=\
np.array([polstr2num(config['POLARIZATIONS'][p])\
 for p in len(config['POLARIZATIONS'])]).astype(int)
#set telescope location
data_uv.telescope_location=np.array(config['TELESCOPE_LOCATION_ITRF'])
#create time array, convert to julian days
jd_times=Time(data['times'],format='unix').jd
data_uv.time_array=\
np.hstack([[jd_times[p] for m in range(data_uv.Nbls)]\
            for p in range(data_uv.Ntimes)])
data_uv.object_name=config['OBJECT_NAME']
data_uv.history='Imported data from Snap correlation file.'
data_uv.phase_type='drift'
data_uv.set_lsts_from_time_array()
radec_obs=ephem.Observer()
lat,lon,alt = data_uv.telescope_location_lat_long_alt_degrees
observatory=EarthLocation(lat=lat*u.degrees,lon=long*u.degrees,height=alt*u.m)
observation_times=Time(data_uv.time_array,format='jd')
zenith_altaz=SkyCoord(location=observatory,obstime=observation_times,
             alt=90.*u.degrees,az=0.*u.degrees,frame='altaz').icrs
data_uv.zenith_ra=zenith_altaz.ra.degree
data_uv.zenith_dec=zenith_altaz.dec.degree
data_uv.instrument=config['INSTRUMENT_NAME']
