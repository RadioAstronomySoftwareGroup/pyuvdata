#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
A command-line script for renumbering antenna numbers > 254 if possible.

This is necessary for CASA because CASA cannot read in uvfits files with
antenna numbers > 254 (apparently 255 isn't ok because 0-based antenna 255 is
1-based 256 and that gets turned into 0 in some 8-bit code path in CASA).

This only works if the number of antennas (Nants_telescope) is less than 255.

Antenna names are not changed, so they reflect the original names of the antennas.

"""
import numpy as np
import os
import argparse
from pyuvdata import UVData
import sys

# setup argparse
a = argparse.ArgumentParser(
    description="A command-line script for renumbering "
    "antenna numbers > 254 if possible."
)
a.add_argument("file_in", type=str, help="input uvfits file.")
a.add_argument("file_out", type=str, help="output uvfits file.")
a.add_argument(
    "--overwrite",
    default=False,
    action="store_true",
    help="overwrite output file if it already exists.",
)
a.add_argument(
    "--verbose", default=False, action="store_true", help="report feedback to stdout."
)
a.add_argument(
    "--filetype",
    default="uvfits",
    type=str,
    help="filetype, options=['uvfits', 'miriad']",
)

# get args
args = a.parse_args()

if os.path.exists(args.file_out) and args.overwrite is False:
    print("{} exists. Use --overwrite to overwrite the file.".format(args.file_out))
    sys.exit(0)

uv_obj = UVData()
if args.filetype == "uvfits":
    uv_obj.read_uvfits(args.file_in)
elif args.filetype == "miriad":
    uv_obj.read_miriad(args.file_in)
else:
    raise IOError("didn't recognize filetype {}".format(args.filetype))

large_ant_nums = sorted(
    uv_obj.antenna_numbers[np.where(uv_obj.antenna_numbers > 254)[0]]
)

new_nums = sorted(set(range(255)) - set(uv_obj.antenna_numbers))
if len(new_nums) < len(large_ant_nums):
    raise ValueError("too many antennas in dataset, cannot renumber all below 255")
new_nums = new_nums[-1 * len(large_ant_nums) :]
renumber_dict = dict(list(zip(large_ant_nums, new_nums)))

for ant_in, ant_out in renumber_dict.items():
    if args.verbose:
        print("renumbering {a1} to {a2}".format(a1=ant_in, a2=ant_out))

    wh_ant_num = np.where(uv_obj.antenna_numbers == ant_in)[0]
    wh_ant1_arr = np.where(uv_obj.ant_1_array == ant_in)[0]
    wh_ant2_arr = np.where(uv_obj.ant_2_array == ant_in)[0]

    uv_obj.antenna_numbers[wh_ant_num] = ant_out
    uv_obj.ant_1_array[wh_ant1_arr] = ant_out
    uv_obj.ant_2_array[wh_ant2_arr] = ant_out

uv_obj.baseline_array = uv_obj.antnums_to_baseline(
    uv_obj.ant_1_array, uv_obj.ant_2_array
)

uv_obj.check()

if args.filetype == "uvfits":
    uv_obj.write_uvfits(args.file_out)
elif args.filetype == "miriad":
    uv_obj.write_miriad(args.file_out, clobber=True)
