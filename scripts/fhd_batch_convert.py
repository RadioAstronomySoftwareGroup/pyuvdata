#! /usr/bin/env python

import argparse
import os
import os.path as op
import re
from uvdata.uv import UVData


def parse_range(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    if not m:
        raise ArgumentTypeError("'" + string + "' is not a range of numbers." +
                                " Expected forms like '0-5' or '2'.")
    start = int(m.group(1))
    end = int(m.group(2)) or start

    return start, end

parser = argparse.ArgumentParser()
parser.add_argument('fhd_run_folder',
                    help='name of an FHD output folder that contains a ' +
                         'vis_data folder')
parser.add_argument('--obsid_range', type=parse_range,
                    help='range of obsids to use, can be a single value or ' +
                         'a min and max with a dash between')
args = parser.parse_args()

vis_folder = op.join(args.fhd_run_folder, 'vis_data')
if not os.path.isdir(vis_folder):
    raise IOError('There is no vis_data folder in {}'.format(args.fhd_run_folder))

metadata_folder = op.join(args.fhd_run_folder, 'metadata')
if not os.path.isdir(vis_folder):
    raise IOError('There is no metadata folder in {}'.format(args.fhd_run_folder))

output_folder = op.join(args.fhd_run_folder, 'uvfits')
if not os.path.exists(output_folder):
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
    fparts = fname.split('_')
    try:
        obsid = int(fparts[0])
        if obsid in file_dict:
            file_dict[obsid].append(f)
        else:
            file_dict[obsid] = [f]
    except:
        continue

try:
    obs_min = args.obsid_range[0]
    obs_max = args.obsid_range[1]
except:
    obs_min = min(file_dict.keys())
    obs_max = max(file_dict.keys())

for k in file_dict.keys():
    if k > obs_max or k < obs_min:
        file_dict.pop(k)

for i, (k, v) in enumerate(file_dict.iteritems()):
    print('converting dirty vis for obsid {}, ({} of {})'.format(k, i, len(file_dict)))

    uvfits_file = op.join(output_folder, str(k) + '.uvfits')
    this_uv = UVData()
    this_uv.read_fhd(v)

    this_uv.write_uvfits(uvfits_file, spoof_nonessential=True)

    del(this_uv)

    print('converting model vis for obsid {}, ({} of {})'.format(k, i, len(file_dict)))
    uvfits_file = op.join(output_folder, str(k) + '_model.uvfits')
    this_uv = UVData()
    this_uv.read_fhd(v, use_model=True)

    this_uv.write_uvfits(uvfits_file, spoof_nonessential=True)

    del(this_uv)
