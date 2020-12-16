# -*- mode: python; coding: utf-8 -*-

"""
Format the UVData object parameters into a sphinx rst file.

"""
import os
import inspect
from pyuvdata import UVData
from astropy.time import Time


def write_dataparams_rst(write_file=None):
    UV = UVData()
    out = "UVData Parameters\n==========================\n"
    out += (
        "These are the standard attributes of UVData objects.\n\nUnder the hood "
        "they are actually properties based on UVParameter objects.\n\nAngle type "
        "attributes also have convenience properties named the same thing \nwith "
        "'_degrees' appended through which you can get or set the value in "
        "degrees.\n\nSimilarly location type attributes (which are given in "
        "topocentric xyz coordinates) \nhave convenience properties named the "
        "same thing with '_lat_lon_alt' and \n'_lat_lon_alt_degrees' appended "
        "through which you can get or set the values using \nlatitude, longitude and "
        "altitude values in radians or degrees and meters.\n\n"
    )
    out += "Required\n----------------\n"
    out += (
        "These parameters are required to have a sensible UVData object and \n"
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
        "These parameters are defined by one or more file standard but are not "
        "always required.\nSome of them are required depending on the "
        "phase_type (as noted below)."
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
        write_file = os.path.join(write_path, "uvdata_parameters.rst")
    F = open(write_file, "w")
    F.write(out)
    print("wrote " + write_file)
