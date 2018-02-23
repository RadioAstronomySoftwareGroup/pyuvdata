#!/usr/bin/env python
"""
A command-line script for renumbering antenna numbers > 255 if possible in uvfits files.
"""
import numpy as np
import os
import argparse
from pyuvdata import UVData

# setup argparse
a = argparse.ArgumentParser(description="A command-line script for renumbering "
                            "antenna numbers > 255 if possible in uvfits files.")
a.add_argument("file_in", type=str, help="input uvfits file.")
a.add_argument("file_out", type=str, help="output uvfits file.")
a.add_argument("--overwrite", default=False, action='store_true',
               help="overwrite output file if it already exists.")
a.add_argument("--verbose", default=False, action='store_true',
               help="report feedback to stdout.")

# get args
args = a.parse_args()

if os.path.exists(args.file_out) and args.overwrite is False:
    print("{} exists, not overwriting...".format(args.file_out))
    continue

uv_obj = UVData()
uv_obj.read_uvfits(args.file_in)

large_ant_nums = sorted(list(uv_obj.antenna_numbers[np.where(uv_obj.antenna_numbers > 254)[0]]))

new_nums = sorted(list(set(range(255)) - set(uv_obj.antenna_numbers)))
if len(new_nums) < len(large_ant_nums):
    raise ValueError('too many antennas in dataset, cannot renumber all below 255')
new_nums = new_nums[-1 * len(large_ant_nums):]
renumber_dict = dict(zip(large_ant_nums, new_nums))

for ant_in, ant_out in renumber_dict.iteritems():
    if args.verbose:
        print "renumbering {a1} to {a2}".format(a1=ant_in, a2=ant_out)

    wh_ant_num = np.where(uv_obj.antenna_numbers == ant_in)[0]
    wh_ant1_arr = np.where(uv_obj.ant_1_array == ant_in)[0]
    wh_ant2_arr = np.where(uv_obj.ant_2_array == ant_in)[0]

    uv_obj.antenna_numbers[wh_ant_num] = ant_out
    uv_obj.ant_1_array[wh_ant1_arr] = ant_out
    uv_obj.ant_2_array[wh_ant2_arr] = ant_out
    blt_inds = np.array(sorted(list(set(wh_ant1_arr.tolist() + wh_ant2_arr.tolist()))))
    uv_obj.baseline_array[blt_inds] = \
        uv_obj.antnums_to_baseline(uv_obj.ant_1_array[blt_inds], uv_obj.ant_2_array[blt_inds])

uv_obj.check()

uv_obj.write_uvfits(args.file_out)
