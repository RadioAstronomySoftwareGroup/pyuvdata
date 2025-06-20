"""Format the UVData object parameters into a sphinx rst file."""

import inspect
import os

from astropy.time import Time

from pyuvdata import UVFlag


def write_uvflag_rst(write_file=None):
    UV = UVFlag()
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
        "transparent to users.\n"
        "Starting in version 3.0, metadata that is associated with the telescope\n"
        "(as opposed to the data set) is stored in a :class:`pyuvdata.Telescope`\n"
        "object (see :ref:`Telescope`) as the ``telescope`` attribute on a UVFlag\n"
        "object. This includes metadata related to the telescope location,\n"
        "antenna names, numbers and positions as well as other telescope metadata.\n\n"
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
        "These parameters are required to have a well-defined UVFlag object and \n"
        "are required for most kinds of uv data files."
    )
    out += "\n\n"
    for thing in UV.required():
        obj = getattr(UV, thing)
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
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
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.UVFlag\n  :members:\n\n"

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += f"last updated: {t.iso}"
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "uvflag_parameters.rst")
    with open(write_file, "w") as F:
        F.write(out)
    print("wrote " + write_file)
