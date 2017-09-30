import os
import copy
import numpy as np
import nose.tools as nt
import pyuvdata.utils as uvutils
from pyuvdata.data import DATA_PATH
import subprocess


def test_pyuvdata_inspect():
    """
    test pyuvdata_inspect.py command line tool
    """
    devnull = open(os.devnull, 'w')
    # run on miriad file
    out = subprocess.call("pyuvdata_inspect.py -a=Ntimes {0}".format(os.path.join(DATA_PATH, 'new.uvA')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_in(out, [0, -6])
    # run on uvfits
    out = subprocess.call("pyuvdata_inspect.py -a=Ntimes {0}".format(os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAAM.uvfits')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_in(out, [0, -6])
    # check double file run
    out = subprocess.call("pyuvdata_inspect.py -a=Ntimes {0}".format(os.path.join(DATA_PATH, 'new.uvA')+' '+os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAAM.uvfits')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_in(out, [0, -6])
    # check double attr run
    out = subprocess.call("pyuvdata_inspect.py -a=Ntimes,Nfreqs {0}".format(os.path.join(DATA_PATH, 'new.uvA')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_in(out, [0, -6])
    # check exceptions
    out = subprocess.call("pyuvdata_inspect.py -a {0}".format(os.path.join(DATA_PATH, 'new.uvA')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_equal(out, 1)
    out = subprocess.call("pyuvdata_inspect.py -a='' {0}".format(os.path.join(DATA_PATH, 'new.uvA')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_equal(out, 1)
    out = subprocess.call("pyuvdata_inspect.py -a=Ntimes {0}".format(os.path.join(DATA_PATH, 'rando')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_equal(out, 1)
    out = subprocess.call("pyuvdata_inspect.py -a=foo {0}".format(os.path.join(DATA_PATH, 'new.uvA')), shell=True, stdout=devnull, stderr=devnull)
    nt.assert_equal(out, 1)
    devnull.close()
    
