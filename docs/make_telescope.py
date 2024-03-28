# -*- mode: python; coding: utf-8 -*-

"""
Format the Telescope object parameters into a sphinx rst file.

"""
import inspect
import json
import os

from astropy.time import Time

from pyuvdata import Telescope
from pyuvdata.telescopes import KNOWN_TELESCOPES


def write_telescope_rst(write_file=None):
    tel = Telescope()
    out = "Telescope\n=========\n\n"
    out += (
        "Telescope is a helper class for telescope-related metadata.\n"
        "Several of the primary user classes need telescope metdata, so they "
        "have a Telescope object as an attribute.\n\n"
        "Attributes\n----------\n"
        "The attributes on Telescope hold all of the metadata required to\n"
        "describe interferometric telescopes. Under the hood, the attributes are\n"
        "implemented as properties based on :class:`pyuvdata.parameter.UVParameter`\n"
        "objects but this is fairly transparent to users.\n\n"
        "When a new Telescope object is initialized, it has all of these \n"
        "attributes defined but set to ``None``. The attributes\n"
        "can be set directly on the object. Some of these attributes\n"
        "are `required`_ to be set to have a fully defined object while others are\n"
        "`optional`_. The :meth:`pyuvdata.Telescope.check` method can be called\n"
        "on the object to verify that all of the required attributes have been\n"
        "set in a consistent way.\n\n"
        "Note that angle type attributes also have convenience properties named the\n"
        "same thing with ``_degrees`` appended through which you can get or set the\n"
        "value in degrees. Similarly location type attributes (which are given in\n"
        "geocentric xyz coordinates) have convenience properties named the\n"
        "same thing with ``_lat_lon_alt`` and ``_lat_lon_alt_degrees`` appended\n"
        "through which you can get or set the values using latitude, longitude and\n"
        "altitude values in radians or degrees and meters.\n\n"
    )
    out += "Required\n********\n"
    out += (
        "These parameters are required to have a basic well-defined Telescope object.\n"
    )
    out += "\n\n"
    for thing in tel.required():
        obj = getattr(tel, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Optional\n********\n"
    out += (
        "These parameters are needed by by one or of the primary user classes\n"
        "but are not always required. Some of them are required when attached to\n"
        "the primary classes."
    )
    out += "\n\n"

    for thing in tel.extra():
        obj = getattr(tel, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.Telescope\n  :members:\n\n"

    out += (
        "Known Telescopes\n================\n\n"
        "Known Telescope Data\n--------------------\n"
        "pyuvdata uses `Astropy sites\n"
        "<https://docs.astropy.org/en/stable/api/astropy.coordinates."
        "EarthLocation.html#astropy.coordinates.EarthLocation.get_site_names>`_\n"
        "for telescope location information, in addition to the following\n"
        "telescope information that is tracked within pyuvdata. Note that for\n"
        "some telescopes we store csv files giving antenna layout information\n"
        "which can be used if data files are missing that information.\n\n"
    )

    json_obj = json.dumps(KNOWN_TELESCOPES, sort_keys=True, indent=4)
    json_obj = json_obj[:-1] + " }"
    out += ".. code-block:: JavaScript\n\n {json_str}\n\n".format(json_str=json_obj)

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += "last updated: {date}".format(date=t.iso)
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "telescope.rst")
    F = open(write_file, "w")
    F.write(out)
    print("wrote " + write_file)
