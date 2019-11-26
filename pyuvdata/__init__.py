# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Init file for pyuvdata."""
from __future__ import absolute_import, division, print_function

# Filter annoying Cython warnings that serve no good purpose. see numpy#432
# needs to be done before the imports to work properly
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from .uvdata.uvdata import UVData  # noqa
from .telescopes import *  # noqa
from .uvcal.uvcal import UVCal  # noqa
from .uvbeam.uvbeam import UVBeam  # noqa
from . import version  # noqa
from .uvflag.uvflag import UVFlag  # noqa

__version__ = version.version
