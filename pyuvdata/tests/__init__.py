# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Setup testing environment, define useful testing functions.

"""
import warnings
import sys

from astropy.utils import iers

import pyuvdata.utils as uvutils

__all__ = [
    "clearWarnings",
    "checkWarnings",
]


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


def checkWarnings(
    func,
    func_args=[],
    func_kwargs={},
    nwarnings=1,
    category=UserWarning,
    message=None,
    known_warning=None,
):
    """Function to check expected warnings in tests.

    Useful for checking that appropriate warnings are raised and to capture
    (and silence) warnings in tests.

    Parameters
    ----------
    func : function
        Function or method to check warnings for.
    func_args : list, optional
        List of positional parameters to pass `func`
    func_kwargs : dict, optional
        Dict of keyword parameter to pass func. Keys are the parameter names,
        values are the values to pass to the parameters.
    nwarnings : int
        Number of expected warnings.
    category : warning type or list of warning types
        Expected warning type(s). If a scalar is passed and `nwarnings` is
        greater than one, the same category will be expected for all warnings.
    message : str or list of str
        Expected warning string(s). If a scalar is passed and `nwarnings` is
        greater than one, the same warning string will be expected for all warnings.
    known_warning : {'miriad', 'paper_uvfits', 'fhd'}, optional
        Shorthand way to specify one of a standard set of warnings.

    Returns
    -------
    Value returned by `func`

    Raises
    ------
    AssertionError
        If the warning(s) raised by func do not match the expected values.
    """

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

    if known_warning == "miriad":
        # The default warnings for known telescopes when reading miriad files
        category = [UserWarning]
        message = [
            "Altitude is not present in Miriad file, using known "
            "location values for PAPER."
        ]
        nwarnings = 1
    elif known_warning == "paper_uvfits":
        # The default warnings for known telescopes when reading uvfits files
        category = [UserWarning] * 2
        message = ["Required Antenna frame keyword", "telescope_location is not set"]
        nwarnings = 2
    elif known_warning == "fhd":
        category = [UserWarning]
        message = ["Telescope location derived from obs"]
        nwarnings = 1

    category = uvutils._get_iterable(category)
    message = uvutils._get_iterable(message)

    clearWarnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # All warnings triggered
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

        # Filter iers warnings if iers.conf.auto_max_age is set to None, as we
        # do in testing if the iers url is down. See conftest.py for more info.
        if iers.conf.auto_max_age is None:
            warnings.filterwarnings("ignore", message="failed to download")
            warnings.filterwarnings("ignore", message="time is out of IERS range")

            if isinstance(message, str):
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
            print(
                "wrong number of warnings. Expected number was {nexp}, "
                "actual number was {nact}.".format(nexp=nwarnings, nact=len(w))
            )
            for idx, wi in enumerate(w):
                print("warning {i} is: {w}".format(i=idx, w=wi))
            assert False
        else:
            for i, w_i in enumerate(w):
                if w_i.category is not category[i]:
                    print("expected category " + str(i) + " was: ", category[i])
                    print("category " + str(i) + " was: ", str(w_i.category))
                    assert False
                if message[i] is None or message[i] == "":
                    print("Expected message " + str(i) + " was None or an empty string")
                    print("message " + str(i) + " was: ", str(w_i.message))
                    assert False
                else:
                    if message[i] not in str(w_i.message):
                        print("expected message " + str(i) + " was: ", message[i])
                        print("message " + str(i) + " was: ", str(w_i.message))
                        assert False
        return retval
