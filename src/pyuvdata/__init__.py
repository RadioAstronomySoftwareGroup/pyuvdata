# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Init file for pyuvdata."""

import contextlib
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from setuptools_scm import get_version


# copy this function here from setup.py.
# Copying code is terrible, but it's better than altering the python path in setup.py.
def branch_scheme(version):  # pragma: nocover
    """
    Local version scheme that adds the branch name for absolute reproducibility.

    If and when this is added to setuptools_scm this function can be removed.
    """
    if version.exact or version.node is None:
        return version.format_choice("", "+d{time:{time_format}}", time_format="%Y%m%d")
    else:
        if version.branch == "main":
            return version.format_choice("+{node}", "+{node}.dirty")
        else:
            return version.format_choice("+{node}.{branch}", "+{node}.{branch}.dirty")


try:
    # get accurate version for developer installs
    # must point to folder that contains the .git file!
    version_str = get_version(
        Path(__file__).parent.parent.parent, local_scheme=branch_scheme
    )

    __version__ = version_str

except (LookupError, ImportError):  # pragma: no cover
    # Set the version automatically from the package details.
    # don't set anything if the package is not installed
    with contextlib.suppress(PackageNotFoundError):
        __version__ = version("pyuvdata")

# Filter annoying Cython warnings that serve no good purpose. see numpy#432
# needs to be done before the imports to work properly
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .analytic_beam import AiryBeam, GaussianBeam, ShortDipoleBeam, UniformBeam  # noqa
from .beam_interface import BeamInterface  # noqa
from .telescopes import Telescope  # noqa
from .uvbeam import UVBeam  # noqa
from .uvcal import UVCal  # noqa
from .uvdata import FastUVH5Meta  # noqa
from .uvdata import UVData  # noqa
from .uvflag import UVFlag  # noqa

__all__ = [
    "UVData",
    "FastUVH5Meta",
    "UVCal",
    "UVFlag",
    "UVBeam",
    "BeamInterface",
    "AiryBeam",
    "GaussianBeam",
    "ShortDipoleBeam",
    "UniformBeam",
    "Telescope",
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
