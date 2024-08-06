# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with FITS files."""

import numpy as np


def _gethduaxis(hdu, axis):
    """
    Make axis arrays for fits files.

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
    ax = str(axis)
    axis_num = hdu.header["NAXIS" + ax]
    val = hdu.header["CRVAL" + ax]
    delta = hdu.header["CDELT" + ax]
    index = hdu.header["CRPIX" + ax] - 1

    return delta * (np.arange(axis_num) - index) + val


def _indexhdus(hdulist):
    """
    Get a dict of table names and HDU numbers from a FITS HDU list.

    Parameters
    ----------
    hdulist : list of astropy.io.fits HDU objects
        List of HDUs to get names for

    Returns
    -------
    dict
        dictionary with table names as keys and HDU number as values.

    """
    tablenames = {}
    for i in range(len(hdulist)):
        try:
            tablenames[hdulist[i].header["EXTNAME"]] = i
        except KeyError:
            continue
    return tablenames


def _get_extra_keywords(header, *, keywords_to_skip=None):
    """
    Get any extra keywords and return as dict.

    Parameters
    ----------
    header : FITS header object
        header object to get extra_keywords from.
    keywords_to_skip : list of str
        list of keywords to not include in extra keywords in addition to standard
        FITS keywords.

    Returns
    -------
    dict
        dict of extra keywords.
    """
    # List standard FITS header items that are still should not be included in
    # extra_keywords
    # These are the beginnings of FITS keywords to ignore, the actual keywords
    # often include integers following these names (e.g. NAXIS1, CTYPE3)
    std_fits_substrings = [
        "HISTORY",
        "SIMPLE",
        "BITPIX",
        "EXTEND",
        "BLOCKED",
        "GROUPS",
        "PCOUNT",
        "GCOUNT",
        "BSCALE",
        "BZERO",
        "NAXIS",
        "PTYPE",
        "PSCAL",
        "PZERO",
        "CTYPE",
        "CRVAL",
        "CRPIX",
        "CDELT",
        "CROTA",
        "CUNIT",
    ]

    if keywords_to_skip is not None:
        std_fits_substrings.extend(keywords_to_skip)

    extra_keywords = {}
    # find all the other header items and keep them as extra_keywords
    for key in header:
        # check if key contains any of the standard FITS substrings
        if np.any([sub in key for sub in std_fits_substrings]):
            continue
        if key == "COMMENT":
            extra_keywords[key] = str(header.get(key))
        elif key != "":
            extra_keywords[key] = header.get(key)

    return extra_keywords
