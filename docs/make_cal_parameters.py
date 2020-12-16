# -*- mode: python; coding: utf-8 -*-

"""
Format the UVCal object parameters into a sphinx rst file.

"""
import os
import inspect
from pyuvdata import UVCal
from astropy.time import Time


def write_calparams_rst(write_file=None):
    cal = UVCal()
    out = "UVCal Parameters\n==========================\n"
    out += (
        "These are the standard attributes of UVCal objects.\n\nUnder the hood "
        "they are actually properties based on UVParameter objects.\n\n"
    )
    out += "Required\n----------------\n"
    out += (
        "These parameters are required to have a sensible UVCal object and \n"
        "are required for most kinds of uv cal files."
    )
    out += "\n\n"
    for thing in cal.required():
        obj = getattr(cal, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Optional\n----------------\n"
    out += (
        "These parameters are defined by one or more file standard but are not "
        "always required.\nSome of them are required depending on the "
        "cal_type (as noted below)."
    )
    out += "\n\n"
    for thing in cal.extra():
        obj = getattr(cal, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"
    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += "last updated: {date}".format(date=t.iso)
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "uvcal_parameters.rst")
    F = open(write_file, "w")
    F.write(out)
    print("wrote " + write_file)
