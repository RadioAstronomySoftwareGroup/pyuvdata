# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2023 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading MS calibration tables."""

import warnings

import numpy as np
from docstring_parser import DocstringStyle

from ..docstrings import copy_replace_short_description
from .uvcal import UVCal, _future_array_shapes_warning

__all__ = ["MSCal"]

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
try:
    import casacore.tables as tables
except ImportError as error:  # pragma: no cover
    casa_present = False
    casa_error = error


class MSCal(UVCal):
    """
    Defines an MS-specific subclass of UVCal for reading MS calibration tables.

    This class should not be interacted with directly, instead use the read_ms_cal
    method on the UVCal class.
    """

    @copy_replace_short_description(UVCal.read_ms_cal, style=DocstringStyle.NUMPYDOC)
    def read_ms_cal(self, use_future_array_shapes=True):
        """Read gains from an MS calibration table."""
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        self.gain_array = np.zeros(1)

        if use_future_array_shapes:
            self.use_future_array_shapes()
        else:
            warnings.warn(_future_array_shapes_warning, DeprecationWarning)

    def write_ms_cal(self, filename):
        """Write out a MS calibration table."""
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        tables.table(filename, {})
