#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Inspect attributes of pyuvdata objects."""

import argparse
from pyuvdata import UVData, UVBeam, UVCal
import os

# setup argparse
a = argparse.ArgumentParser(
    description=(
        "Inspect attributes of pyuvdata objects.\n"
        "Example: pyuvdata_inspect.py -a=ant_array.shape,"
        "Ntimes zen.xx.HH.omni.calfits zen.yy.HH.uvc"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

a.add_argument(
    "-a",
    "--attrs",
    dest="attrs",
    type=str,
    default="",
    help="attribute(s) of object to print. Ex: ant_array.shape,Ntimes",
)
a.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="Send feedback to stdout.",
)
a.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    default=False,
    help="Exit into a python interpretor with objects in memory as 'uv'.",
)
a.add_argument(
    "files",
    metavar="files",
    type=str,
    nargs="*",
    default=[],
    help="pyuvdata object files to run on",
)

# parse arguments
args = a.parse_args()

# check for empty attributes
if len(args.attrs) == 0 and args.interactive is False:
    raise Exception("no attributes fed...")
if len(args.files) == 0:
    raise Exception("no files fed...")

# pack data objects, their names, and read functions
objs = [UVData, UVCal, UVBeam]
ob_names = ["UVData", "UVCal", "UVBeam"]
ob_reads = [
    ["read", "read_miriad", "read_fhd", "read_ms", "read_uvfits", "read_uvh5"],
    ["read_calfits", "read_fhd_cal"],
    ["read_beamfits", "read_cst_beam"],
]

# iterate through files
Nfiles = len(args.files)
uv = []
exit_clean = True
for i, f in enumerate(args.files):
    # check file exists
    if os.path.exists(f) is False:
        print("{0} doesn't exist".format(f))
        if i == (Nfiles - 1):
            exit(1)
        else:
            continue

    opened = False
    filetype = None
    # try to open object
    for j, ob in enumerate(objs):
        for k, r in enumerate(ob_reads[j]):
            try:
                # instantiate data class and try to read file
                UV = ob()
                getattr(UV, r)(f)
                opened = True
                uv.append(UV)
                filetype = r.split("_")[-1]
                if args.verbose is True:
                    print(
                        "opened {0} as a {1} file with the {2} pyuvdata object".format(
                            f, filetype, ob_names[j]
                        )
                    )
            except (IOError, KeyError, ValueError, RuntimeError):
                continue
            # exit loop if opened
            if opened is True:
                break
        if opened is True:
            break

    # if object isn't opened continue
    if opened is False:
        print(
            "couldn't open {0} with any of the pyuvdata objects {1}".format(f, ob_names)
        )
        continue

    # print out desired attribute(s) of data object
    attrs = [x.split(".") for x in args.attrs.split(",")]
    for j, attr in enumerate(attrs):
        # try to get attribute
        try:
            Nnest = len(attr)
            this_attr = getattr(UV, attr[0])
            for k in range(Nnest - 1):
                this_attr = getattr(this_attr, attr[k + 1])
            # print to stdout
            print("{0} of {1} is: {2}".format(".".join(attr), f, this_attr))
            exit_clean = True
        except AttributeError:
            print("Couldn't access '{0}' from {1}".format(".".join(attr), f))
            exit_clean = False

if args.interactive:
    if len(uv) == 1:
        uv = uv[0]
    try:
        from IPython import embed

        embed()
    except ImportError:
        import code

        code.interact(local=dict(globals(), **locals()))

else:
    if exit_clean is True:
        exit(0)
    else:
        exit(1)
