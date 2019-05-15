# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Setup testing environment, define useful testing functions.

"""
from __future__ import absolute_import, division, print_function

import warnings
import sys
import pytest
from unittest import SkipTest, TestCase

import functools
import types
import six

from astropy.utils import iers

import pyuvdata.utils as uvutils

# define a pytest marker for skipping casacore tests
try:
    import casacore

    casa_installed = True
except ImportError:
    casa_installed = False
reason = 'casacore is not installed, skipping tests that require it.'
skipIf_no_casa = pytest.mark.skipif(not casa_installed, reason=reason)


# define a pytest marker to skip healpy tests
try:
    import healpy

    healpy_installed = True
except(ImportError):
    healpy_installed = False
reason = 'healpy is not installed, skipping tests that require it.'
skipIf_no_healpy = pytest.mark.skipif(not healpy_installed, reason=reason)


# defines a decorator to skip tests that require h5py.
try:
    import h5py

    h5py_installed = True
except(ImportError):
    h5py_installed = False
reason = 'h5py is not installed, skipping tests that require it.'
skipIf_no_h5py = pytest.mark.skipif(not h5py_installed, reason=reason)


# defines a decorator to skip tests that require yaml.
try:
    import yaml

    yaml_installed = True
except(ImportError):
    yaml_installed = False
reason = 'yaml is not installed, skipping tests that require it.'
skipIf_no_yaml = pytest.mark.skipif(not yaml_installed, reason=reason)


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
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

        # filter iers warnings if iers.conf.auto_max_age is set to None, as we do in testing if the iers url is down
        if iers.conf.auto_max_age is None:
            warnings.filterwarnings("ignore", message="failed to download")
            warnings.filterwarnings("ignore", message="time is out of IERS range")

            if isinstance(message, six.string_types):
                test_message = [message.startswith("LST values stored in ")]
            else:
                test_message = []
                for m in message:
                    if m is None:
                        test_message.append(False)
                    else:
                        test_message.append(m.startswith("LST values stored in "))
            if not any(test_message):
                warnings.filterwarnings("ignore", message="LST values stored in ")

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
                if message[i] is None or message[i] == '':
                    print('Expected message ' + str(i) + ' was None or an empty string')
                    print('message ' + str(i) + ' was: ', str(w_i.message))
                    assert(False)
                else:
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
    warnings.simplefilter('always', DeprecationWarning)  # turn off filter
    warnings.warn("The skip function was built to work with the `nose` test "
                  "suite and is deprecated. It will be removed version 1.5.",
                  DeprecationWarning)
    warnings.simplefilter('default', DeprecationWarning)  # reset filter
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
