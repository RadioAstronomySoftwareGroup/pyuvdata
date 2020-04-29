#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Test if two UVData compatible files are equal."""

import argparse
import os.path as op
from pyuvdata import UVData

parser = argparse.ArgumentParser()
parser.add_argument("uvfits1", help="name of first uvfits file.")
parser.add_argument("uvfits2", help="name of second uvfits file to compare to first.")

args = parser.parse_args()

uvfits_file1 = args.uvfits1
if not op.isfile(uvfits_file1):
    raise IOError("There is no file named {}".format(args.uvfits_file1))

uvfits_file2 = args.uvfits2
if not op.isfile(uvfits_file2):
    raise IOError("There is no file named {}".format(args.uvfits_file2))

uv1 = UVData()
uv1.read_uvfits(uvfits_file1)

uv2 = UVData()
uv2.read_uvfits(uvfits_file2)

if uv1 == uv2:
    print("UVData objects from files are equal")
else:
    print("UVData objects from files are not equal")

del uv1
del uv2
