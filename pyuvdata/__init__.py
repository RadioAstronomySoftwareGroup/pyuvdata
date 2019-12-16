# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""init file for pyuvdata.

"""
from __future__ import absolute_import, division, print_function
import warnings
from pkg_resources import get_distribution, DistributionNotFound

# Set the version automatically from the package details.
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

# Filter annoying Cython warnings that serve no good purpose. see numpy#432
# needs to be done before the imports to work properly
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .uvdata import *  # noqa
from .telescopes import *  # noqa
from .uvcal import *  # noqa
from .uvbeam import *  # noqa
from .uvflag import UVFlag  # noqa
