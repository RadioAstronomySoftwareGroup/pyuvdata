'''
Script to import a numpy array written by a snap board and convert to a uvdata
set
'''

import numpy as np
import yaml
import argparse
from pyuvdata import UVData
from pyuvdata import polstr2num
import pyuvdata.utils as utils
from astropy.time import Time
import copy
from astropy.coordinates import SkyCoord,EarthLocation,AltAz
import astropy.units as u
from hera_mc.sys_handling import Handling
from hera_mc import cm_utils
desc=('Convert SNAP outputs into uvdata sets.'
      'Usage: import_snap.py -c config.yaml -z observation.npz'
      'config.yaml is a .yaml file containing instrument meta-data'
      'observation.npz contains numpy array of data from the SNAP')

parser=argparse.ArgumentParser(description=desc)
parser.add_argument('--config','-c',dest='config',help='configuration yaml file')
parser.add_argument('--data','-z',dest='data',help='data npz file')

args=parser.parse_args()


with open(args.config) as configfile:
    config=yaml.load(configfile)

data=np.load(args.data)

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
                             data_uv.Nfreqs,data_uv.Npols),dtype=bool)
data_uv.flag_array[:]=False
#instantiate nsamples array with equal unity weights.
data_uv.nsample_array=np.ones((data_uv.Nblts,data_uv.Nspws,
                              data_uv.Nfreqs,data_uv.Npols),dtype=float)

#create frequency array
data_uv.freq_array=np.zeros((data_uv.Nspws,data_uv.Nfreqs))
data_uv.freq_array[0,:]=data['frequencies']
#channel width
data_uv.channel_width=data_uv.freq_array[0,1]-data_uv.freq_array[0,0]
#!!!Does the snap board have a 100% duty cycle? Probably not quite...
data_uv.integration_time=data['times'][1]-data['times'][0]
#convert to Nblt ordering
data_uv.data_array=np.zeros((data_uv.Nblts,data_uv.Nspws,data_uv.Nfreqs,data_uv.Npols),
                             dtype=complex)
for tnum in range(data_uv.Ntimes):
    #print data['data'].shape
    #print data_uv.data_array.shape
    #print tnum
    data_uv.data_array[tnum*data_uv.Nbls:(tnum+1)*data_uv.Nbls,0,:,:]\
    =data['data'][tnum,:,:,:]

#Translate antenna locations
my_handle=Handling()
begintime=Time(data['times'][0],format='unix')
antenna_configuration=my_handle.get_all_fully_connected_at_date(begintime)
#get antenna numbers
all_antnums=[]
all_antnames=[]
data_antnums=[]
data_antnames=[]
data_antnums=[]
data_antnames=[]
data_antennas_lla=[]
data_antennas_enu=[]
data_antennas_xyz=[]
all_antennas_lla={}
all_antennas_enu={}
all_antennas_xyz=[]

enu_datum=['WGS84']
for connection in antenna_configuration:
    antnum=connection['antenna_number']
    antname=connection['station_name']
    ant_lla=(np.radians(connection['latitude']),
             np.radians(connection['longitude']),
              connection['elevation'])
    ant_xyz=utils.XYZ_from_LatLonAlt(ant_lla[0],ant_lla[1],ant_lla[2])
    ant_enu=(connection['easting'],
             connection['northing'],
             connection['elevation'])
    all_antnums.append(antnum)
    all_antnames.append(antname)
    all_antennas_lla[antnum]=np.array(ant_lla)
    all_antennas_enu[antnum]=np.array(ant_enu)
    all_antennas_xyz.append(np.array(ant_xyz))
    if connection['antenna_number'] in config['ANTENNA_NUMBERS']:
        data_antnums.append(antnum)
        data_antnames.append(antname)
        data_antennas_lla.append(ant_lla)
        data_antennas_enu.append(ant_enu)
        data_antennas_xyz.append(ant_xyz)
#Convert antenna ENU into ITRF
#Create baseline_array
#create polarization array
data_uv.polarization_array=\
np.array([polstr2num(config['POLARIZATIONS'][p])\
 for p in range(len(config['POLARIZATIONS']))]).astype(int)
#set telescope location
data_uv.telescope_location=np.array(all_antennas_xyz).mean(axis=0)
data_uv.antenna_positions=np.array(all_antennas_xyz)-data_uv.telescope_location
data_uv.antenna_numbers=np.array(all_antnums).astype(int)
data_uv.antenna_names=all_antnames
data_uv.Nants_telescope=len(data_uv.antenna_numbers)
#create ant_1_array
data_uv.ant_1_array=\
np.hstack([data['antenna_pairs'][:,0] for p in range(data_uv.Ntimes)]).flatten()\
.astype(int)
#create ant_2_array
data_uv.ant_2_array=\
np.hstack([data['antenna_pairs'][:,1] for p in range(data_uv.Ntimes)]).flatten()\
.astype(int)

#print(np.unique(data_uv.ant_1_array))
#print(np.unique(data_uv.ant_2_array))
#print config['ANTENNA_NUMBERS']
data_uv.Nants_data=len(data_antnums)
data_uv.antenna_names=all_antnames
#convert from input index to antenna number and compute uvw
data_uv.uvw_array=np.zeros((data_uv.Nblts,3))
for ai,ant1,ant2 in zip(range(data_uv.Nblts),
                        data_uv.ant_1_array,data_uv.ant_2_array):
    data_uv.ant_1_array[ai]=config['ANTENNA_NUMBERS'][ant1]
    data_uv.ant_2_array[ai]=config['ANTENNA_NUMBERS'][ant2]
    ant1,ant2=data_uv.ant_1_array[ai],data_uv.ant_2_array[ai]
    data_uv.uvw_array[ai]=all_antennas_enu[ant2]-all_antennas_enu[ant1]
data_uv.baseline_array=\
(2048*(data_uv.ant_2_array+1)+(data_uv.ant_1_array+1)+2**16).astype(np.int64)
#print(data_uv.Nants_data)
#print(np.unique(data_uv.ant_1_array))
#print(np.unique(data_uv.ant_2_array))

#create time array, convert to julian days
jd_times=Time(data['times'],format='unix').jd
data_uv.time_array=\
np.hstack([[jd_times[p] for m in range(data_uv.Nbls)]\
            for p in range(data_uv.Ntimes)])
data_uv.object_name=config['OBJECT_NAME']
data_uv.history='Imported data from Snap correlation file.'
data_uv.phase_type='drift'
data_uv.set_lsts_from_time_array()
lat,lon,alt = data_uv.telescope_location_lat_lon_alt_degrees
observatory=EarthLocation(lat=lat * u.degree,lon=lon * u.degree,height=alt * u.m)
data_uv.zenith_ra=np.zeros_like(data_uv.time_array)
data_uv.zenith_dec=np.zeros_like(data_uv.time_array)
for tnum,t in enumerate(data_uv.time_array):
    observation_time=Time(t,format='jd')
    zenith_altaz=SkyCoord(location=observatory,obstime=observation_time,
                          alt=90. * u.degree,az=0. * u.degree,frame='altaz').icrs
    data_uv.zenith_ra[tnum]=zenith_altaz.ra.degree
    data_uv.zenith_dec[tnum]=zenith_altaz.dec.degree

data_uv.instrument=config['INSTRUMENT_NAME']
<<<<<<< HEAD
data_uv.spw_array=np.zeros(data_uv.Nspws).astype(int)
data_uv.telescope_name=config['TELESCOPE_NAME']
#print(data_uv.Nbls)
#print(data_uv.Nblts)
#print(data_uv.Ntimes)
=======

>>>>>>> file writing added.
if config['FORMAT']=='MIRIAD':
    data_uv.write_miriad(config['OUTPUTNAME'])
elif config['FORMAT']=='UVFITS':
    data_uv.write_uvfits(config['OUTPUTNAME'])
