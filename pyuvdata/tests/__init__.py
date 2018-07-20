# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Setup testing environment, define useful testing functions.

"""
from __future__ import absolute_import, division, print_function

import os
import warnings
import sys
import numpy as np
import scipy
from pyuvdata.data import DATA_PATH
import pyuvdata.utils as uvutils


def setup_package():
    """Make data/test directory to put test output files in."""
    testdir = os.path.join(DATA_PATH, 'test/')
    if not os.path.exists(testdir):
        print('making test directory')
        os.mkdir(testdir)


# Functions that are useful for testing:
def clearWarnings():
    """Quick code to make warnings reproducible."""
    for name, mod in list(sys.modules.items()):
        try:
            reg = getattr(mod, "__warningregistry__", None)
        except ImportError:
            continue
        if reg:
            reg.clear()


# things we need to figure out expected warnings when reading FHD files
scipy_warnings = (float(np.__version__[0:4]) >= 1.14 and (scipy.__version__ < '1.0.1'))


def get_scipy_warnings(n_scipy_warnings=1093):
    """
    Function to generate warning information for reading IDL save files.

    numpy 1.14 introduced a new deprecation warning.
    Should be fixed when the next scipy version comes out.
    The number of replications of the warning varies some and must be
    empirically discovered. It it defaults to the most common number.
    """
    scipy_warn_str = 'The binary mode of fromstring is deprecated'
    scipy_warn_list = []
    scipy_category_list = []
    for i in range(n_scipy_warnings):
        scipy_warn_list.append(scipy_warn_str)
        scipy_category_list.append(DeprecationWarning)
    return n_scipy_warnings, scipy_warn_list, scipy_category_list


def checkWarnings(func, func_args=[], func_kwargs={},
                  category=UserWarning,
                  nwarnings=1, message=None, known_warning=None):
    """Function to check expected warnings."""

    if (not isinstance(category, list) or len(category) == 1) and nwarnings > 1:
        if isinstance(category, list):
            category = category * nwarnings
        else:
            category = [category] * nwarnings

    if (not isinstance(message, list) or len(message) == 1) and nwarnings > 1:
        if isinstance(message, list):
            message = message * nwarnings
        else:
            message = [message] * nwarnings

    if known_warning == 'miriad':
        # The default warnings for known telescopes when reading miriad files
        category = [UserWarning]
        message = ['Altitude is not present in Miriad file, using known '
                   'location values for PAPER.']
        nwarnings = 1
    elif known_warning == 'paper_uvfits':
        # The default warnings for known telescopes when reading uvfits files
        category = [UserWarning] * 2
        message = ['Required Antenna frame keyword', 'telescope_location is not set']
        nwarnings = 2

    category = uvutils._get_iterable(category)
    message = uvutils._get_iterable(message)

    clearWarnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # All warnings triggered
        retval = func(*func_args, **func_kwargs)  # Run function
        # Verify
        if len(w) != nwarnings:
            print('wrong number of warnings. Expected number was {nexp}, '
                  'actual number was {nact}.'.format(nexp=nwarnings, nact=len(w)))
            for idx, wi in enumerate(w):
                print('warning {i} is: {w}'.format(i=idx, w=wi))
            assert(False)
        else:
            for i, w_i in enumerate(w):
                if w_i.category is not category[i]:
                    print('expected category ' + str(i) + ' was: ', category[i])
                    print('category ' + str(i) + ' was: ', str(w_i.category))
                    assert(False)
                if message[i] is not None:
                    if message[i] not in str(w_i.message):
                        print('expected message ' + str(i) + ' was: ', message[i])
                        print('message ' + str(i) + ' was: ', str(w_i.message))
                        assert(False)
        return retval
