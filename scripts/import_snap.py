'''
Script to import a numpy array written by a snap board and convert to a uvdata
set
'''

import numpy
import yaml
import argparse
from pyuvdata import UVData

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

#instantiate a uvdata object.
