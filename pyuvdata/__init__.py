# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Init file for pyuvdata."""
import warnings
from setuptools_scm import get_version
from pathlib import Path
from pkg_resources import get_distribution, DistributionNotFound

from .branch_scheme import branch_scheme


try:  # pragma: nocover
    # get accurate version for developer installs
    version_str = get_version(Path(__file__).parent.parent, local_scheme=branch_scheme)

    __version__ = version_str

except (LookupError, ImportError):
    try:
        # Set the version automatically from the package details.
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:  # pragma: nocover
        # package is not installed
        pass

# Filter annoying Cython warnings that serve no good purpose. see numpy#432
# needs to be done before the imports to work properly
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .uvdata import UVData  # noqa
from .telescopes import known_telescopes, Telescope, get_telescope  # noqa
from .uvcal import UVCal  # noqa
from .uvbeam import UVBeam  # noqa
from .uvflag import UVFlag  # noqa

__all__ = [
    "UVData",
    "UVCal",
    "UVFlag",
    "UVBeam",
    "Telescope",
    "known_telescopes",
    "get_telescope",
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
        or (varname.startswith("__") and varname.endswith("__"))
        or (
            varname[0] != "_"
            and isinstance(locals()[varname], __module_type__)
            and locals()[varname].__name__.startswith(__name__ + ".")
        )
    ):
        del locals()[varname]

del varname, __module_type__
