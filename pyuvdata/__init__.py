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

from .uvdata import *  # noqa
from .telescopes import *  # noqa
from .uvcal import *  # noqa
from .uvbeam import *  # noqa
from . import version  # noqa
from .uvflag import UVFlag  # noqa

__version__ = version.version
