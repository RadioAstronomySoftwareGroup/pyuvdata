# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Commonly used utility functions."""
from __future__ import annotations

import warnings

import numpy as np

# standard angle tolerance: 1 mas in radians.
RADIAN_TOL = 1 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)
# standard lst time tolerance: 5 ms (75 mas in radians), based on an expected RMS
# accuracy of 1 ms at 7 days out from issuance of Bulletin A (which are issued once a
# week with rapidly determined parameters and forecasted values of DUT1), the exact
# formula for which is t_err = 0.00025 (MJD-<Bulletin A Release Data>)**0.75 (in secs).
LST_RAD_TOL = 2 * np.pi * 5e-3 / (86400.0)

# these seem to be necessary for the installed package to access these submodules
from . import apply_uvflag  # noqa
from . import array_collapse  # noqa
from . import bls  # noqa
from . import bltaxis  # noqa
from . import coordinates  # noqa
from . import frequency  # noqa
from . import history  # noqa
from . import io  # noqa
from . import phase_center_catalog  # noqa
from . import phasing  # noqa
from . import pol  # noqa
from . import redundancy  # noqa
from . import times  # noqa
from . import tools  # noqa
from . import uvcalibrate  # noqa

# Add things to the utils namespace used by outside packages
from .apply_uvflag import apply_uvflag  # noqa
from .array_collapse import collapse  # noqa
from .bls import *  # noqa
from .coordinates import *  # noqa
from .phasing import uvw_track_generator  # noqa
from .pol import *  # noqa
from .times import get_lst_for_time  # noqa
from .uvcalibrate import uvcalibrate  # noqa

# deprecated imports


def _check_histories(history1, history2):
    """Check if two histories are the same.

    Deprecated. Use pyuvdata.utils.history._check_histories
    """
    from .history import _check_histories

    warnings.warn(
        "The _check_histories function has moved, please import it from "
        "pyuvdata.utils.history. This warnings will become an error in version 3.2",
        DeprecationWarning,
    )

    return _check_histories(history1, history2)


def _fits_gethduaxis(hdu, axis):
    """
    Make axis arrays for fits files.

    Deprecated. Use pyuvdata.utils.io.fits._gethduaxis.

    Parameters
    ----------
    hdu : astropy.io.fits HDU object
        The HDU to make an axis array for.
    axis : int
        The axis number of interest (1-based).

    Returns
    -------
    ndarray of float
        Array of values for the specified axis.

    """
    from .io.fits import _gethduaxis

    warnings.warn(
        "The _fits_gethduaxis function has moved, please import it as "
        "pyuvdata.utils.io.fits._gethduaxis. This warnings will become an "
        "error in version 3.2",
        DeprecationWarning,
    )

    return _gethduaxis(hdu, axis)


def _fits_indexhdus(hdulist):
    """
    Get a dict of table names and HDU numbers from a FITS HDU list.

    Deprecated. Use pyuvdata.utils.io.fits._indexhdus.

    Parameters
    ----------
    hdulist : list of astropy.io.fits HDU objects
        List of HDUs to get names for

    Returns
    -------
    dict
        dictionary with table names as keys and HDU number as values.

    """
    from .io.fits import _indexhdus

    warnings.warn(
        "The _fits_indexhdus function has moved, please import it as "
        "pyuvdata.utils.io.fits._indexhdus. This warnings will become an "
        "error in version 3.2",
        DeprecationWarning,
    )

    return _indexhdus(hdulist)
