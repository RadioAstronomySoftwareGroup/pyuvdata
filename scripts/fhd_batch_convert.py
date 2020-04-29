#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Convert multiple FHD datasets to UVFITS format."""

import argparse
import os
import os.path as op
import re
from pyuvdata import UVData


def parse_range(string):
    """Parse numbers from a range string."""
    m = re.match(r"(\d+)(?:-(\d+))?$", string)
    if not m:
        raise argparse.ArgumentTypeError(
            "'{}' is not a range of numbers. Expected forms like '0-5' or '2'.".format(
                string
            )
        )
    start = int(m.group(1))
    end = int(m.group(2)) or start

    return start, end


parser = argparse.ArgumentParser()
parser.add_argument(
    "fhd_run_folder",
    help="name of an FHD output folder that contains a "
    "vis_data folder and a metadata folder",
)
parser.add_argument(
    "--obsid_range",
    type=parse_range,
    help="range of obsids to use, can be a single value or "
    "a min and max with a dash between",
)
parser.add_argument(
    "--no-dirty",
    dest="dirty",
    action="store_false",
    help="do not convert dirty visibilities",
)
parser.set_defaults(dirty=True)
parser.add_argument(
    "--no-model",
    dest="model",
    action="store_false",
    help="do not convert model visibilities",
)
parser.set_defaults(model=True)
args = parser.parse_args()

vis_folder = op.join(args.fhd_run_folder, "vis_data")
if not op.isdir(vis_folder):
    raise IOError("There is no vis_data folder in {}".format(args.fhd_run_folder))

metadata_folder = op.join(args.fhd_run_folder, "metadata")
if not op.isdir(vis_folder):
    raise IOError("There is no metadata folder in {}".format(args.fhd_run_folder))

output_folder = op.join(args.fhd_run_folder, "uvfits")
if not op.exists(output_folder):
    os.mkdir(output_folder)

files = []
obsids = []
for f in os.listdir(vis_folder):
    files.append(op.join(vis_folder, f))

for f in os.listdir(metadata_folder):
    files.append(op.join(metadata_folder, f))

file_dict = {}
for f in files:
    dirname, fname = op.split(f)
    fparts = fname.split("_")
    try:
        obsid = int(fparts[0])
        if obsid in file_dict:
            file_dict[obsid].append(f)
        else:
            file_dict[obsid] = [f]
    except ValueError:
        continue

try:
    obs_min = args.obsid_range[0]
    obs_max = args.obsid_range[1]
except TypeError:
    obs_min = min(file_dict.keys())
    obs_max = max(file_dict.keys())

for k in list(file_dict.keys()):
    if k > obs_max or k < obs_min:
        file_dict.pop(k)

for i, (k, v) in enumerate(file_dict.items()):
    if args.dirty:
        print(
            "converting dirty vis for obsid {}, ({} of {})".format(k, i, len(file_dict))
        )
        uvfits_file = op.join(output_folder, str(k) + ".uvfits")
        this_uv = UVData()
        this_uv.read_fhd(v)

        this_uv.write_uvfits(uvfits_file, spoof_nonessential=True)

        del this_uv

    if args.model:
        print(
            "converting model vis for obsid {}, ({} of {})".format(k, i, len(file_dict))
        )
        uvfits_file = op.join(output_folder, str(k) + "_model.uvfits")
        this_uv = UVData()
        this_uv.read_fhd(v, use_model=True)

        this_uv.write_uvfits(uvfits_file, spoof_nonessential=True)

        del this_uv
