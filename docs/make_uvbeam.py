"""
Format the UVBeam object parameters into a sphinx rst file.

"""

import inspect
import os

from astropy.time import Time

from pyuvdata import UVBeam


def write_uvbeam_rst(write_file=None):
    beam = UVBeam()
    out = "UVBeam\n======\n"
    out += (
        "UVBeam is the main user class for primary beam models for radio telescopes.\n"
        "It provides import and export functionality to and from the supported file\n"
        "formats (beamFITS, CST, MWA primary beam) as well as methods for\n"
        "transforming the data (interpolating/regridding, selecting, converting\n"
        "types) and can be interacted with directly.\n\n"
        "Note that there are some tricks that can help with reading in CST beam\n"
        "simulation files in `CST Settings Files`_.\n"
        "UVBeam also provides a yaml constructor that can enable beams to be\n"
        "instantiated directly from yaml files (see: `UVbeam yaml constructor`_)\n\n"
        "Attributes\n----------\n"
        "The attributes on UVBeam hold all of the metadata and data required to\n"
        "describe primary beam models. Under the hood, the attributes are implemented\n"
        "as properties based on :class:`pyuvdata.parameter.UVParameter` objects but\n"
        "this is fairly transparent to users.\n\n"
        "UVBeam objects can be initialized from a file using the\n"
        ":meth:`pyuvdata.UVBeam.from_file` class method\n"
        "(as ``beam = UVBeam.from_file(<filename>)``) or be initialized from arrays\n"
        "in memory using the :meth:`pyuvdata.UVBeam.new` class method or as an empty\n"
        "object (as ``beam = UVBeam()``). When an empty UVBeam object is initialized,\n"
        "it has all of these attributes defined but set to ``None``. The attributes\n"
        "can be set by reading in a data file using the :meth:`pyuvdata.UVBeam.read`\n"
        "method or by setting them directly on the object. Some of these attributes\n"
        "are `required`_ to be set to have a fully defined data set while others are\n"
        "`optional`_. The :meth:`pyuvdata.UVBeam.check` method can be called on the\n"
        "object to verify that all of the required attributes have been set in a\n"
        "consistent way.\n\n"
    )
    out += "Required\n********\n"
    out += (
        "These parameters are required to have a well-defined UVBeam object and \n"
        "are required for most kinds of beam files."
    )
    out += "\n\n"
    for thing in beam.required():
        obj = getattr(beam, thing)
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
        out += "\n"

    out += "Optional\n********\n"
    out += (
        "These parameters are defined by one or more file standard but are not "
        "always required.\nSome of them are required depending on the "
        "beam_type, antenna_type and pixel_coordinate_systems (as noted below)."
    )
    out += "\n\n"
    for thing in beam.extra():
        obj = getattr(beam, thing)
        out += f"**{obj.name}**\n"
        out += f"     {obj.description}\n"
        out += "\n"

    out += "Methods\n-------\n.. autoclass:: pyuvdata.UVBeam\n  :members:\n\n"

    out += """
UVBeam yaml constructor
-----------------------

UVbeams can be instantiated directly from yaml files using the
``!UVBeam`` tag. The ``filename`` parameter must be specified and
any other parameter that can be passed to the :meth:`pyuvdata.UVBeam.read`
method can also be specified.

Some examples are provided below, note that the node key can be anything,
it does not need to be ``beam``:

A simple UVBeam specification::

    beam: !UVBeam
        filename: hera.beamfits

An UVbeam specification with some keywords to pass to ``UVBeam.read``::

    beam: !UVBeam
        filename: mwa_full_EE_test.h5
        pixels_per_deg: 1
        freq_range: [100.e+6, 200.e+6]

\n\n
"""

    with open("cst_settings_yaml.rst", encoding="utf-8") as cst_settings_file:
        cst_setting_text = cst_settings_file.read()

    out += cst_setting_text + "\n\n"

    t = Time.now()
    t.format = "iso"
    t.out_subfmt = "date"
    out += f"last updated: {t.iso}"
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, "uvbeam_parameters.rst")
    with open(write_file, "w") as F:
        F.write(out)
    print("wrote " + write_file)
