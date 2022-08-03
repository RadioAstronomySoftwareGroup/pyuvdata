# -*- mode: python; coding: utf-8 -*-

"""Format the UVData object parameters into a sphinx rst file."""
import inspect
import os

from astropy.time import Time

from pyuvdata import UVFlag
from pyuvdata.data import DATA_PATH


def write_uvflag_rst(write_file=None):
    test_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5")
    UV = UVFlag(test_file)
    out = "UVFlag\n======\n"
    out += (
        "UVFlag is a main user class that is still in the development and\n"
        "beta-testing phase, which holds metadata and data related to flagging and\n"
        "metric information for interferometric data sets. It provides import and\n"
        "export functionality to and from all file formats supported by\n"
        ":class:`pyuvdata.UVData` and :class:`pyuvdata.UVCal` objects as well as an\n"
        "HDF5 file format specified by this object.\n"
        'It supports three different "shapes" of data (all with time and frequency\n'
        "axes): visibility based, antenna based, and waterfall (a single value for\n"
        "the entire array at each time and frequency).\n"
        "It has methods for transforming the data between different shapes and \n"
        "converting metrics to flags, and can be interacted with directly.\n\n"
        "Attributes\n----------\n"
        "The attributes on UVFlag hold all of the metadata and data required to\n"
        "specify flagging and metric information for interferometric data sets.\n"
        "Under the hood, the attributes are implemented as properties based on\n"
        ":class:`pyuvdata.parameter.UVParameter` objects but this is fairly\n"
        "transparent to users.\n\n"
        "UVFlag objects can be initialized from a file or a :class:`pyuvdata.UVData`\n"
        "or :class:`pyuvdata.UVCal` object\n"
        "(as ``flag = UVFlag(<filename or object>)``). Some of these attributes\n"
        "are `required`_ to be set to have a fully defined data set while others are\n"
        "`optional`_. The :meth:`pyuvdata.UVFlag.check` method can be called on the\n"
        "object to verify that all of the required attributes have been set in a\n"
        "consistent way.\n\n"
    )
    out += "Required\n********\n"
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

    out += "Optional\n********\n"
    out += (
        "These parameters are defined by one or more type but are not "
        "always required.\nSome of them are required depending on the "
        "type (as noted below)."
    )
    out += "\n\n"
    for thing in UV.extra():
        obj = getattr(UV, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.UVFlag\n  :members:\n\n"

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
