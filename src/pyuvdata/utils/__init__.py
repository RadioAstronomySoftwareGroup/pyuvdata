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

# Add things to the utils namespace used by outside packages
from .array_collapse import collapse  # noqa
from .bls import *  # noqa
from .coordinates import *  # noqa
from .phasing import uvw_track_generator  # noqa
from .pol import *  # noqa
from .times import get_lst_for_time  # noqa

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


def uvcalibrate(uvdata, uvcal, **kwargs):
    """
    Calibrate a UVData object with a UVCal object.

    Deprecated, use pyuvdata.uvcalibrate

    Parameters
    ----------
    uvdata : UVData object
        UVData object to calibrate.
    uvcal : UVCal object
        UVCal object containing the calibration.
    inplace : bool, optional
        if True edit uvdata in place, else return a calibrated copy
    prop_flags : bool, optional
        if True, propagate calibration flags to data flags
        and doesn't use flagged gains. Otherwise, uses flagged gains and
        does not propagate calibration flags to data flags.
    Dterm_cal : bool, optional
        Calibrate the off-diagonal terms in the Jones matrix if present
        in uvcal. Default is False. Currently not implemented.
    flip_gain_conj : bool, optional
        This function uses the UVData ant_1_array and ant_2_array to specify the
        antennas in the UVCal object. By default, the conjugation convention, which
        follows the UVData convention (i.e. ant2 - ant1), is that the applied
        gain = ant1_gain * conjugate(ant2_gain). If the other convention is required,
        set flip_gain_conj=True.
    delay_convention : str, optional
        Exponent sign to use in conversion of 'delay' to 'gain' cal_type
        if the input uvcal is not inherently 'gain' cal_type. Default to 'minus'.
    undo : bool, optional
        If True, undo the provided calibration. i.e. apply the calibration with
        flipped gain_convention. Flag propagation rules apply the same.
    time_check : bool
        Option to check that times match between the UVCal and UVData
        objects if UVCal has a single time or time range. Times are always
        checked if UVCal has multiple times.
    ant_check : bool
        Option to check that all antennas with data on the UVData
        object have calibration solutions in the UVCal object. If this option is
        set to False, uvcalibrate will proceed without erroring and data for
        antennas without calibrations will be flagged.

    Returns
    -------
    UVData, optional
        Returns if not inplace

    """
    from ..uvcalibrate import uvcalibrate

    warnings.warn(
        "uvcalibrate has moved, please import it as 'from pyuvdata import "
        "uvcalibrate'. This warnings will become an error in version 3.2",
        DeprecationWarning,
    )

    return uvcalibrate(uvdata, uvcal, **kwargs)


def apply_uvflag(uvd, uvf, **kwargs):
    """
    Apply flags from a UVFlag to a UVData instantiation.

    Deprecated, use pyuvdata.apply_uvflag

    Note that if uvf.Nfreqs or uvf.Ntimes is 1, it will broadcast flags across
    that axis.

    Parameters
    ----------
    uvd : UVData object
        UVData object to add flags to.
    uvf : UVFlag object
        A UVFlag object in flag mode.
    inplace : bool
        If True overwrite flags in uvd, otherwise return new object
    unflag_first : bool
        If True, completely unflag the UVData before applying flags.
        Else, OR the inherent uvd flags with uvf flags.
    flag_missing : bool
        If input uvf is a baseline type and antpairs in uvd do not exist in uvf,
        flag them in uvd. Otherwise leave them untouched.
    force_pol : bool
        If True, broadcast flags to all polarizations if they do not match.
        Only works if uvf.Npols == 1.

    Returns
    -------
    UVData
        If not inplace, returns new UVData object with flags applied

    """
    from ..apply_uvflag import apply_uvflag

    warnings.warn(
        "uvcalibrate has moved, please import it as 'from pyuvdata import "
        "uvcalibrate'. This warnings will become an error in version 3.2",
        DeprecationWarning,
    )

    return apply_uvflag(uvd, uvf, **kwargs)
