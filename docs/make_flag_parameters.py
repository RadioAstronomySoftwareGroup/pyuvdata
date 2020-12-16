# -*- mode: python; coding: utf-8 -*-

"""Format the UVData object parameters into a sphinx rst file."""
import os
import inspect
from pyuvdata import UVFlag
from pyuvdata.data import DATA_PATH
from astropy.time import Time


def write_flagparams_rst(write_file=None):
    test_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5")
    UV = UVFlag(test_file)
    out = "UVFlag Parameters\n=================\n"
    out += (
        "These are the standard attributes of UVFlag objects.\n\nUnder the hood "
        "they are actually properties based on UVParameter objects.\n\n"
    )
    out += "Required\n----------------\n"
    out += (
        "These parameters are required to have a sensible UVFlag object and \n"
        "are required for most kinds of uv data files."
    )
    out += "\n\n"
    for thing in UV.required():
        obj = getattr(UV, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Optional\n----------------\n"
    out += (
        "These parameters are defined by one or more  type but are not "
        "always required.\nSome of them are required depending on the "
        "type (as noted below)."
    )
    out += "\n\n"
    for thing in UV.extra():
        obj = getattr(UV, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"
    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += "last updated: {date}".format(date=t.iso)
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "uvflag_parameters.rst")
    F = open(write_file, "w")
    F.write(out)
    print("wrote " + write_file)
