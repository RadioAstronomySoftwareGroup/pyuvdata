import unittest
import inspect
import os
import os.path as op
import shutil
import astropy.time  # necessary for Jonnie's workflow help us all
from uvdata.uv import UVData
import numpy as np
import copy
import ephem
import warnings
import collections
from astropy.io import fits
import sys


test_file_directory = '../data/test/'
if not os.path.exists(test_file_directory):
    print('making test directory')
    os.mkdir(test_file_directory)

if __name__ == '__main__':
    unittest.main()
