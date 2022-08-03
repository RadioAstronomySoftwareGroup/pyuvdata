# -*- mode: python; coding: utf-8 -*-

"""
Format the UVCal object parameters into a sphinx rst file.

"""
import inspect
import os

from astropy.time import Time

from pyuvdata import UVCal


def write_uvcal_rst(write_file=None):
    cal = UVCal()
    out = "UVCal\n=====\n"
    out += (
        "UVCal is the main user class for calibration solutions for interferometric\n"
        "data sets. It provides import and export functionality to and from the\n"
        "supported file formats (calfits, FHD) as well as methods for transforming\n"
        "the data (converting types, selecting, sorting) and can be interacted with\n"
        "directly.\n\n"
        "Attributes\n----------\n"
        "The attributes on UVCal hold all of the metadata and data required to\n"
        "work with calibration solutions for interferometric data sets. Under the\n"
        "hood, the attributes are implemented as properties based on\n"
        ":class:`pyuvdata.parameter.UVParameter` objects but this is fairly\n"
        "transparent to users.\n\n"
        "UVCal objects can be initialized as an empty object (as ``cal = UVCal()``).\n"
        "When an empty UVCal object is initialized, it has all of these attributes\n"
        "defined but set to ``None``. The attributes can be set by reading in a data\n"
        "file using the :meth:`pyuvdata.UVCal.read_calfits` or\n"
        ":meth:`pyuvdata.UVCal.read_fhd_cal` methods or by setting them directly on\n"
        "the object. Some of these attributes are `required`_ to be set to have a\n"
        "fully defined calibration data set while others are `optional`_. The\n"
        ":meth:`pyuvdata.UVCal.check` method can be called on the object to verify\n"
        "that all of the required attributes have been set in a consistent way.\n\n"
        'Note that objects can be in a "metadata only" state where\n'
        "all of the metadata is defined but the data-like attributes (``gain_array``,\n"
        "``delay_array``, ``flag_array``, ``quality_array``) are not. The\n"
        ":meth:`pyuvdata.UVCal.check` method will still pass for metadata only\n"
        "objects.\n\n"
        "Note location type attributes (which are given in topocentric xyz\n"
        "coordinates) have convenience properties named the same thing with\n"
        "``_lat_lon_alt`` and ``_lat_lon_alt_degrees`` appended through which you can\n"
        "get or set the values using latitude, longitude and altitude values in\n"
        "radians or degrees and meters.\n\n"
    )
    out += "Required\n********\n"
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

    out += "Optional\n********\n"
    out += (
        "These parameters are defined by one or more file standard but are not "
        "always required.\nSome of them are required depending on the "
        "cal_type or cal_style (as noted below)."
    )
    out += "\n\n"
    for thing in cal.extra():
        obj = getattr(cal, thing)
        out += "**{name}**\n".format(name=obj.name)
        out += "     {desc}\n".format(desc=obj.description)
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.UVCal\n  :members:\n\n"

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += "last updated: {date}".format(date=t.iso)
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "uvcal.rst")
    F = open(write_file, "w")
    F.write(out)
    print("wrote " + write_file)
