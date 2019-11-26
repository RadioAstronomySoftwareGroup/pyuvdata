# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""init file for pyuvdata.

"""
from __future__ import absolute_import, division, print_function

# Filter annoying Cython warnings that serve no good purpose. see numpy#432
# needs to be done before the imports to work properly
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .uvdata.uvdata import UVData  # noqa
from .uvcal.uvcal import UVCal  # noqa
from .uvbeam.uvbeam import UVBeam  # noqa
from .telescopes import known_telescopes, Telescope, get_telescope  # noqa
from . import version  # noqa
from .uvflag.uvflag import UVFlag  # noqa


__version__ = version.version
__all__ = [
    "UVData",
    "UVCal",
    "UVFlag",
    "UVBeam",
    "version",
    "Telescope",
    "known_telescopes",
    "get_telescope"
]


# adapted from https://github.com/astropy/astropy/__init__.py
# please consult astropy/__init__.py for clarification on logic details

# Cleanup the top-level namespace.
# Delete everything that is not in __all__, a magic function,
# or is a submodule of this package
from types import ModuleType as __module_type__  # noqa
for varname in dir():
    if not (
        varname in __all__
        or varname == "version"
        or (varname.startswith("__") and varname.endswith("__"))
        or (
            varname[0] != "_"
            and isinstance(locals()[varname], __module_type__)
            and locals()[varname].__name__.startswith(__name__ + '.')
        )
    ):
        del locals()[varname]

del varname, __module_type__
