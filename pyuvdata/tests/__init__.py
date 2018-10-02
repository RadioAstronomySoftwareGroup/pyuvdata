# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Setup testing environment, define useful testing functions.

"""
from __future__ import absolute_import, division, print_function

import os
import warnings
import sys
from unittest import SkipTest, TestCase
import functools
import types
import six

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
    elif known_warning == 'fhd':
        category = [UserWarning]
        message = ['Telescope location derived from obs']
        nwarnings = 1

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


def _id(obj):
    return obj


def skip(reason):
    """
    Defines a decorator to unconditionally skip a test. Called by conditional
    skip wrappers to skip tests that require optional dependencies.

    This is needed because nose doesn't respect unittest skip_if decorators.
    Based on: https://stackoverflow.com/questions/21936292/conditional-skip-testcase-decorator-in-nosetests
    Args:
        reason: String describing the reason for skipping a test.
    """
    def decorator(test_item):
        if six.PY2:
            class_types = (type, types.ClassType)
        else:
            class_types = (type)
        if not isinstance(test_item, class_types):
            @functools.wraps(test_item)
            def skip_wrapper(*args, **kwargs):
                raise SkipTest(reason)
            test_item = skip_wrapper
        elif issubclass(test_item, TestCase):
            @classmethod
            @functools.wraps(test_item.setUpClass)
            def skip_wrapper(*args, **kwargs):
                raise SkipTest(reason)
            test_item.setUpClass = skip_wrapper
        test_item.__unittest_skip__ = True
        test_item.__unittest_skip_why__ = reason
        return test_item
    return decorator


def skipIf_no_casa(test_func):
    """defines a decorator to skip tests that require casacore."""
    reason = 'casacore is not installed, skipping tests that require it.'
    try:
        import casacore
    except(ImportError):
            return skip(reason)(test_func)
    return test_func


def skipIf_no_healpy(test_func):
    """defines a decorator to skip tests that require healpy."""
    reason = 'healpy is not installed, skipping tests that require it.'
    try:
        import healpy
    except(ImportError):
            return skip(reason)(test_func)
    return test_func


def skipIf_no_h5py(test_func):
    """defines a decorator to skip tests that require h5py."""
    reason = 'h5py is not installed, skipping tests that require it.'
    try:
        import h5py
    except(ImportError):
            return skip(reason)(test_func)
    return test_func
