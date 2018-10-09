# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""init file for pyuvdata.

"""
from __future__ import absolute_import, division, print_function

from .uvbase import *
from .parameter import *
from .uvdata import *
from .utils import *
from .telescopes import *
from .uvfits import *
from .fhd import *
from .miriad import *
from .uvcal import *
from .calfits import *
from .uvbeam import *
from .uvh5 import *
from . import version

__version__ = version.version


def reraise_context(fmt, *args):
    """Reraise an exception with its message modified to specify additional
    context.
    This function tries to help provide context when a piece of code
    encounters an exception while trying to get something done, and it wishes
    to propagate contextual information farther up the call stack. It is a
    consistent way to do it for both Python 2 and 3, since Python 2 does not
    provide Python 3’s `exception chaining <https://www.python.org/dev/peps/pep-3134/>`_ functionality.
    Instead of that more sophisticated infrastructure, this function just
    modifies the textual message associated with the exception being raised.
    If only a single argument is supplied, the exception text prepended with
    the stringification of that argument. If multiple arguments are supplied,
    the first argument is treated as an old-fashioned ``printf``-type
    (``%``-based) format string, and the remaining arguments are the formatted
    values.
    Borrowed from pwkit (https://github.com/pkgw/pwkit/blob/master/pwkit/__init__.py)
    Example usage::
      from pwkit import reraise_context
      from pwkit.io import Path
      filename = 'my-filename.txt'
      try:
        f = Path(filename).open('rt')
        for line in f.readlines():
          # do stuff ...
      except Exception as e:
        reraise_context('while reading "%r"', filename)
        # The exception is reraised and so control leaves this function.
    If an exception with text ``"bad value"`` were to be raised inside the
    ``try`` block in the above example, its text would be modified to read
    ``"while reading \"my-filename.txt\": bad value"``.
    """
    import sys

    if len(args):
        cstr = fmt % args
    else:
        cstr = text_type(fmt)

    ex = sys.exc_info()[1]

    if isinstance(ex, EnvironmentError):
        ex.strerror = '%s: %s' % (cstr, ex.strerror)
        ex.args = (ex.errno, ex.strerror)
    else:
        if len(ex.args):
            cstr = '%s: %s' % (cstr, ex.args[0])
        ex.args = (cstr, ) + ex.args[1:]

    raise
