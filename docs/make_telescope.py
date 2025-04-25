"""
Format the Telescope object parameters into a sphinx rst file.

"""

import copy
import inspect
import json
import os

from astropy.time import Time

from pyuvdata import Telescope
from pyuvdata.telescopes import KNOWN_TELESCOPES


def write_telescope_rst(write_file=None):
    tel = Telescope()
    out = ".. _Telescope:\n\nTelescope\n=========\n\n"
    out += (
        "Telescope is a helper class for telescope-related metadata.\n"
        "Several of the primary user classes need telescope metdata, so they "
        "have a Telescope object as an attribute.\n\n"
        "Attributes\n----------\n"
        "The attributes on Telescope hold all of the metadata required to\n"
        "describe interferometric telescopes. Under the hood, the attributes are\n"
        "implemented as properties based on :class:`pyuvdata.parameter.UVParameter`\n"
        "objects but this is fairly transparent to users.\n\n"
        "Most commonly, Telescope objects are found as the ``telescope`` attribute\n"
        "on UVData, UVCal and UVFlag objects and they are typically initialized\n"
        "when those objects are initialized.\n\n"
        "Stand-alone Telescope objects can be initialized in many ways: from\n"
        "arrays in memory using the :meth:`pyuvdata.Telescope.from_params`\n"
        "class method, from our known telescope information using the\n"
        ":meth:`pyuvdata.Telescope.from_known_telescopes` class method,\n"
        "from uvh5, calh5 or uvflag HDF5 files using the "
        ":meth:`pyuvdata.Telescope.from_hdf5` class method,\n"
        "or as an empty object (as ``tel = Telescope()``).\n"
        "When an empty Telescope object is initialized, it has all of these \n"
        "attributes defined but set to ``None``. The attributes\n"
        "can be set directly on the object. Some of these attributes\n"
        "are `required`_ to be set to have a fully defined object while others are\n"
        "`optional`_. The :meth:`pyuvdata.Telescope.check` method can be called\n"
        "on the object to verify that all of the required attributes have been\n"
        "set in a consistent way.\n\n"
    )
    out += "Required\n********\n"
    out += (
        "These parameters are required to have a basic well-defined Telescope object.\n"
    )
    out += "\n\n"
    for thing in tel.required():
        obj = getattr(tel, thing)
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
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
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.Telescope\n  :members:\n\n"

    out += (
        "Known Telescopes\n----------------\n\n"
        "pyuvdata uses `Astropy sites\n"
        "<https://docs.astropy.org/en/stable/api/astropy.coordinates."
        "EarthLocation.html#astropy.coordinates.EarthLocation.get_site_names>`_\n"
        "for telescope location information, in addition to the following\n"
        "telescope information that is tracked within pyuvdata. Note that the\n"
        "location entry is actually stored as an\n"
        ":class:`astropy.coordinates.EarthLocation` object, which\n"
        "is shown here using the Geodetic representation. Also note that for\n"
        "some telescopes we store csv files giving antenna layout information\n"
        "which can be used if data files are missing that information.\n"
        "We also provide a convenience function to get known telescope locations.\n\n"
    )

    known_tel_use = copy.deepcopy(KNOWN_TELESCOPES)
    for tel, tel_dict in KNOWN_TELESCOPES.items():
        if "location" in tel_dict:
            known_tel_use[tel]["location"] = (
                tel_dict["location"].to_geodetic().__repr__()
            )

    json_obj = json.dumps(known_tel_use, sort_keys=True, indent=4)
    json_obj = json_obj[:-1] + " }"
    out += f".. code-block:: JavaScript\n\n {json_obj}\n\n"

    out += ".. autofunction:: pyuvdata.telescopes.known_telescope_location\n\n"

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += f"last updated: {t.iso}"
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "telescope.rst")
    with open(write_file, "w") as F:
        F.write(out)
    print("wrote " + write_file)
