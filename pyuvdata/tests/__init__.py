# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Setup testing environment, define useful testing functions.

"""
import re
import sys
import warnings

from astropy.utils import iers

import pyuvdata.utils as uvutils

__all__ = [
    "check_warnings",
    "clearWarnings",
    "checkWarnings",
]


class WarningsChecker(warnings.catch_warnings):
    """
    A context manager to check raised warnings.

    Adapted from pytest WarningsRecorder and WarningsChecker.

    Parameters
    ----------
    expected_warning : list of Warnings
        List of expected warnings.
    match : list of str or regex
        List of strings to match warnings to.

    """

    def __init__(self, expected_warning, match):
        """Check inputs and initalize object."""
        super().__init__(record=True)
        self._entered = False
        self._list = []

        msg = "exceptions must be derived from Warning, not %s"
        if expected_warning is None:
            expected_warning_list = expected_warning
        elif isinstance(expected_warning, list):
            for exc in expected_warning:
                if not issubclass(exc, Warning):
                    raise TypeError(msg % type(exc))
            expected_warning_list = expected_warning
        elif issubclass(expected_warning, Warning):
            expected_warning_list = [expected_warning]
        else:
            raise TypeError(msg % type(expected_warning))

        msg = "match must be a str, not %s"
        if match is None:
            match_list = None
        elif isinstance(match, list):
            for exc in match:
                if not isinstance(exc, str):
                    raise TypeError(msg % type(exc))
            match_list = match
        elif isinstance(match, str):
            match_list = [match]
        else:
            raise TypeError(msg % type(match))

        self.expected_warning = expected_warning_list
        self.match = match_list

    @property
    def warnlist(self):
        """The list of recorded warnings."""
        return self._list

    def __getitem__(self, i: int):
        """Get a recorded warning by index."""
        return self._list[i]

    def __iter__(self):
        """Iterate through the recorded warnings."""
        return iter(self._list)

    def __len__(self):
        """The number of recorded warnings."""
        return len(self._list)

    def pop(self, cls):
        """
        Pop the first recorded warning, raise exception if not exists.

        Parameters
        ----------
        cls : Warning
            Warning class to check for.
        """
        for i, w in enumerate(self._list):
            if issubclass(w.category, cls):
                return self._list.pop(i)
        __tracebackhide__ = True
        raise AssertionError("%r not found in warning list" % cls)

    def clear(self) -> None:
        """Clear the list of recorded warnings."""
        self._list[:] = []

    def __enter__(self):
        if self._entered:
            __tracebackhide__ = True
            raise RuntimeError("Cannot enter %r twice" % self)
        _list = super().__enter__()
        # record=True means it's None.
        assert _list is not None
        self._list = _list
        warnings.simplefilter("always")
        # Filter annoying Cython warnings that serve no good purpose. see numpy#432
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
        # PLP: Temporarily add this filter as well.
        # numpy v1.20 causes h5py to issue warnings. Must keep as is for now to
        # avoid test failures, should change back later.
        warnings.filterwarnings("ignore", message="Passing None into shape arguments")

        # Filter iers warnings if iers.conf.auto_max_age is set to None, as we
        # do in testing if the iers url is down. See conftest.py for more info.
        if iers.conf.auto_max_age is None:
            warnings.filterwarnings("ignore", message="failed to download")
            warnings.filterwarnings("ignore", message="time is out of IERS range")

            test_message = []
            for message in self.match:
                if message is None:
                    test_message.append(False)
                else:
                    test_message.append(message.startswith("LST values stored in "))
            if not any(test_message):
                warnings.filterwarnings("ignore", message="LST values stored in ")

        return self

    def __exit__(
        self, exc_type=None, exc_val=None, exc_tb=None,
    ):
        if not self._entered:
            __tracebackhide__ = True
            raise RuntimeError("Cannot exit %r without entering first" % self)

        super().__exit__(exc_type, exc_val, exc_tb)

        # Built-in catch_warnings does not reset entered state so we do it
        # manually here for this context manager to become reusable.
        self._entered = False

        # only check if we're not currently handling an exception
        if exc_type is None and exc_val is None and exc_tb is None:
            if self.expected_warning is None:
                assert len(self) == 0
            else:
                assert len(self) == len(self.expected_warning), (
                    f"{len(self.expected_warning)} warnings expected, "
                    f"{len(self)} warnings issued. The list of emitted warnings is: "
                    f"{[each.message for each in self]}."
                )

                for warn_i, exp_warn in enumerate(self.expected_warning):
                    if not any(issubclass(r.category, exp_warn) for r in self):
                        __tracebackhide__ = True
                        raise AssertionError(
                            "DID NOT WARN. No warnings of type {} was emitted. "
                            "The list of emitted warnings is: {}.".format(
                                self.expected_warning, [each.message for each in self]
                            )
                        )
                    elif self.match is not None:
                        for record in self:
                            if str(record.message).startswith(
                                self.match[warn_i]
                            ) or re.compile(self.match[warn_i]).search(
                                str(record.message)
                            ):
                                if issubclass(
                                    record.category, self.expected_warning[warn_i]
                                ):
                                    break
                        else:
                            raise AssertionError(
                                f"No warnings of type {self.expected_warning[warn_i]} "
                                f"matching ('{self.match[warn_i]}') was "
                                "emitted. The list of emitted warnings is: "
                                f"{[each.message for each in self]}."
                            )


def check_warnings(expected_warning, match=None, nwarnings=None, *args, **kwargs):
    """
    Assert that code raises a particular set of warnings, used as a context manager.

    Similar to ``pytest.warns``, but allows for specifying multiple warnings.
    It also better matches warning strings when the warning uses f-strings or
    formating. Can be used as a drop-in replacement for ``pytest.warns`` if
    only one warning is issued (if more are issued they will need to be added
    to the input lists for this to pass).

    Note that unlike the older checkWarnings function, the warnings can be passed
    in any order, they do not have to match the order the warnings are raised
    in the code.

    To assert that there are no warnings raised by some code, set `expected_warning`
    to None (i.e. `with check_warnings(None):`)

    Parameters
    ----------
    expected_warning : list of Warnings or Warning
        List of expected warnings. If a single warning type or a length 1 list,
        will be used for the type of all warnings.
    match : str or regex or list of str or regex
        List of strings or regexes to match warnings to. If a str or a length 1
        list, will be used for all warnings.
    nwarnings : int, optional
        Option to specify that multiple of a single type of warning is expected.
        Only used if category and match both only have one element.

    """
    __tracebackhide__ = True

    if not (
        expected_warning is None
        or isinstance(expected_warning, list)
        or issubclass(expected_warning, Warning)
    ):
        raise TypeError("expected_warning must be a list or be derived from Warning")
    if match is not None and not isinstance(match, (list, str)):
        raise TypeError("match must be a list or a string.")

    if expected_warning is not None and not isinstance(expected_warning, list):
        expected_warning_list = [expected_warning]
    else:
        expected_warning_list = expected_warning
    if match is not None and not isinstance(match, list):
        match_list = [match]
    else:
        match_list = match

    if expected_warning is not None:
        if (
            len(expected_warning_list) > 1
            and len(match_list) > 1
            and len(expected_warning_list) != len(match_list)
        ):
            raise ValueError(
                "If expected_warning and match both have more than one element, "
                "they must be the same length."
            )

        if len(expected_warning_list) > 1 or len(match_list) > 1:
            nwarnings = max(len(expected_warning_list), len(match_list))
        elif nwarnings is None:
            nwarnings = 1

        if len(expected_warning_list) < nwarnings:
            expected_warning_list = expected_warning_list * nwarnings
        if len(match_list) < nwarnings:
            match_list = match_list * nwarnings

    if not args:
        if kwargs:
            msg = "Unexpected keyword arguments passed to check_warnings: "
            msg += ", ".join(sorted(kwargs))
            msg += "\nUse context-manager form instead?"
            raise TypeError(msg)
        return WarningsChecker(expected_warning_list, match_list)
    else:
        func = args[0]
        if not callable(func):
            raise TypeError(
                "{!r} object (type: {}) must be callable".format(func, type(func))
            )
        with WarningsChecker(expected_warning_list, match_list):
            return func(*args[1:], **kwargs)


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
    func_args=None,
    func_kwargs=None,
    nwarnings=1,
    category=UserWarning,
    message=None,
    known_warning=None,
):
    """
    Function to check expected warnings in tests.

    Deprecated. Use check_warnings instead.

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
    warnings.warn(
        "`checkWarnings` is deprecated, and will be removed in pyuvdata version "
        "2.3. Use `check_warnings` instead.",
        DeprecationWarning,
    )
    if func_args is None:
        func_args = []
    if func_kwargs is None:
        func_kwargs = {}

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
