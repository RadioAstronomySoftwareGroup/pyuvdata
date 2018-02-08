#!/usr/bin/env python
"""
A command-line script for converting
a Miriad file to UVFITS format
"""
import aipy as a
import numpy as np
import argparse
import sys
import os
import pyuvdata

# setup argparse
a = argparse.ArgumentParser(description="A command-line script for converting a Miriad file to UVFITS format.")
a.add_argument("files", type=str, nargs='*', help="Miriad files to convert to uvfits.")
a.add_argument("--phase_time", type=float, default=None, help="Julian Date to phase data to. Default is the first integration of the file.")
a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output file if it already exists.")
a.add_argument("--verbose", default=False, action='store_true', help="report feedback to stdout.")

# get args
args = a.parse_args()
history = ' '.join(sys.argv)

# iterate over files
for filename in args.files:

    # check output
    outfilename = filename + '.uvfits'
    if os.path.exists(outfilename) and args.overwrite is False:
        print("{} exists, not overwriting...".format(outfilename))
        continue

    # read in file
    UV = pyuvdata.UVData()
    UV.read_miriad(filename, 'miriad')

    # phase data
    if args.phase_time is not None:
        UV.phase_to_time(args.phase_time)
        if args.verbose:
            print "phasing {} to time {}".format(filename, args.phase_time)

    else:
        UV.phase_to_time(UV.time_array[0])
        if args.verbose:
            print "phasing {} to time {}".format(filename, UV.time_array[0])

    # write data
    UV.history += history
    if args.verbose:
        print "saving {}".format(outfilename)
    UV.write_uvfits(outfilename, spoof_nonessential=True)
