# -*- mode: python; coding: utf-8 -*-

"""
Format the UVData object parameters into a sphinx rst file.

"""
import os
import inspect
from pyuvdata import UVData
from astropy.time import Time


def write_uvdata_rst(write_file=None):
    UV = UVData()
    out = "UVData\n======\n\n"
    out += (
        "UVData is the main user class for intereferometric data (visibilities).\n"
        "It provides import and export functionality to and from the supported file\n"
        "formats (UVFITS, MeasurementSets, Miriad, uvh5, FHD, MIR) and can be\n"
        "interacted with directly.\n\n"
        "Attributes\n----------\n"
        "The attributes on UVData hold all of the metadata and data required to"
        "analyze interferometric data sets. They are implemented as properties based\n"
        "on :class:`pyuvdata.UVParameter` objects.\n\n"
        "Note that angle type attributes also have convenience properties named the\n"
        "same thing with '_degrees' appended through which you can get or set the\n"
        "value in degrees. Similarly location type attributes (which are given in\n"
        "topocentric xyz coordinates) have convenience properties named the\n"
        "same thing with '_lat_lon_alt' and '_lat_lon_alt_degrees' appended\n"
        "through which you can get or set the values using latitude, longitude and\n"
        "altitude values in radians or degrees and meters.\n\n"
    )
    out += "Required\n********\n"
    out += (
        "These parameters are required to have a sensible UVData object and\n"
        "are required for most kinds of interferometric data files."
    )
    out += "\n\n"
    for thing in UV.required():
        obj = getattr(UV, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Optional\n********\n"
    out += (
        "These parameters are defined by one or more file standard but are not\n"
        "always required. Some of them are required depending on the\n"
        "phase_type (as noted below)."
    )
    out += "\n\n"

    for thing in UV.extra():
        obj = getattr(UV, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.UVData\n  :members:\n\n"

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += "last updated: {date}".format(date=t.iso)
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "uvdata_desc.rst")
    F = open(write_file, "w")
    F.write(out)
    print("wrote " + write_file)
