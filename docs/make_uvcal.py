"""
Format the UVCal object parameters into a sphinx rst file.

"""

import inspect
import os

from astropy.time import Time

from pyuvdata import Telescope, UVCal


def write_uvcal_rst(write_file=None):
    cal = UVCal()
    cal.telescope = Telescope()
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
        "transparent to users.\n"
        "Starting in version 3.0, metadata that is associated with the telescope\n"
        "(as opposed to the data set) is stored in a :class:`pyuvdata.Telescope`\n"
        "object (see :ref:`Telescope`) as the ``telescope`` attribute on a\n"
        "UVCal object. This includes metadata related to the telescope location,\n"
        "antenna names, numbers and positions as well as other telescope metadata.\n\n"
        "UVCal objects can be initialized in many ways: from a file using the\n"
        ":meth:`pyuvdata.UVCal.from_file` class method\n"
        "(as ``uvc = UVCal.from_file(<filename>)``), from arrays in memory using\n"
        "the :meth:`pyuvdata.UVCal.new` class method, from a\n"
        ":class:`pyvdata.UVData` object using the\n"
        ":meth:`pyuvdata.UVCal.initialize_from_uvdata` class method, or as an\n"
        "empty object (as ``cal = UVCal()``).\n"
        "When an empty UVCal object is initialized, it has all of these attributes\n"
        "defined but set to ``None``. The attributes can be set by reading in a data\n"
        "file using the :meth:`pyuvdata.UVCal.read` method or by setting them\n"
        "directly on the object. Some of these attributes are `required`_ to be\n"
        "set to have a fully defined calibration data set while others are\n"
        "`optional`_. The :meth:`pyuvdata.UVCal.check` method can be called on\n"
        "the object to verify that all of the required attributes have been set\n"
        "in a consistent way.\n\n"
        'Note that objects can be in a "metadata only" state where\n'
        "all of the metadata is defined but the data-like attributes (``gain_array``,\n"
        "``delay_array``, ``flag_array``, ``quality_array``) are not. The\n"
        ":meth:`pyuvdata.UVCal.check` method will still pass for metadata only\n"
        "objects.\n\n"
    )
    out += "Required\n********\n"
    out += (
        "These parameters are required to have a well-defined UVCal object and \n"
        "are required for most kinds of uv cal files."
    )
    out += "\n\n"
    for thing in cal.required():
        obj = getattr(cal, thing)
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
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
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.UVCal\n  :members:\n\n"

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += f"last updated: {t.iso}"
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "uvcal.rst")
    with open(write_file, "w") as F:
        F.write(out)
    print("wrote " + write_file)
