#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Read in a uvfits file and write a new one out."""

import argparse
import os.path as op
from pyuvdata import UVData

parser = argparse.ArgumentParser()
parser.add_argument("uvfits_read", help="name of a uvfits file to read in")
parser.add_argument("uvfits_write", help="name of a uvfits file to write out")

args = parser.parse_args()

uvfits_file_in = args.uvfits_read
if not op.isfile(uvfits_file_in):
    raise IOError("There is no file named {}".format(args.uvfits_file_in))

uvfits_file_out = args.uvfits_write

this_uv = UVData()
this_uv.read_uvfits(uvfits_file_in)

this_uv.write_uvfits(uvfits_file_out)

del this_uv
