#! /usr/bin/env python

import argparse
import os
import os.path as op
from uvdata.uv import UVData


def parse_range(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    if not m:
        raise ArgumentTypeError("'" + string + "' is not a range of numbers." +
                                " Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start

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

output_folder = op.join(args.fhd_run_folder, 'uvfits')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files = []
obsids = []
for (dirpath, dirnames, filenames) in os.walk(vis_folder):
    files.extend(filenames)
    fparts = filenames.split('_')
    try:
        obsid = int(fparts[0])
        obsids.extend(obsid)
    except:
        continue
    break

obsids = list(set(obsids.sort()))

try:
    obs_min = arg.obsid_range(0)
    obs_max = arg.obsid_range(1)
except:
    obs_min = min(obsids)
    obs_max = max(obsids)

obsids_use = [obs for obs in obsids if obs >= obs_min and obs <= obs_max]

for obs in obsids_use:
    obs_files = []
    for f in (f for f in files if f.startswith(str(obs))):
        obs_files.extend(f)

    uvfits_file = op.join(output_folder, str(obs) + '.uvfits')
    this_uv = UVData()
    this_uv.read_fhd(obs_files)

    this_uv.write_uvfits(uvfits_file, spoof_nonessential=True)
