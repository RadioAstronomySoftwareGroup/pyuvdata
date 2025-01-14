# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Commonly utility functions for testing."""

import inspect
import re
import warnings

from astropy.utils import iers

__all__ = ["check_warnings"]


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
                if not (inspect.isclass(exc) and issubclass(exc, Warning)):
                    raise TypeError(msg % type(exc))
            expected_warning_list = expected_warning
        elif inspect.isclass(expected_warning) and issubclass(
            expected_warning, Warning
        ):
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
        """Return the number of recorded warnings."""
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
        raise AssertionError(f"{cls!r} not found in warning list")

    def clear(self) -> None:
        """Clear the list of recorded warnings."""
        self._list[:] = []

    def __enter__(self):
        if self._entered:
            __tracebackhide__ = True
            raise RuntimeError(f"Cannot enter {self!r} twice")
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
        if iers.conf.auto_max_age is None:  # pragma: no cover
            warnings.filterwarnings("ignore", message="failed to download")
            warnings.filterwarnings("ignore", message="time is out of IERS range")

            test_message = []
            if self.match is not None:
                for message in self.match:
                    if message is None:
                        test_message.append(False)
                    else:
                        test_message.append(message.startswith("LST values stored in "))
            if not any(test_message):
                warnings.filterwarnings("ignore", message="LST values stored in ")

        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if not self._entered:
            __tracebackhide__ = True
            raise RuntimeError(f"Cannot exit {self!r} without entering first")

        super().__exit__(exc_type, exc_val, exc_tb)

        # Built-in catch_warnings does not reset entered state so we do it
        # manually here for this context manager to become reusable.
        self._entered = False

        # only check if we're not currently handling an exception
        if exc_type is None and exc_val is None and exc_tb is None:
            if self.expected_warning is None:
                expected_length = 0
            else:
                expected_length = len(self.expected_warning)

            if len(self) != expected_length:
                warn_file_line = []
                msg_list = []
                for each in self:
                    warn_file_line.append(f"{each.filename}: {each.lineno}")
                    msg_list.append(each.message)
                if self.expected_warning is None:
                    err_msg = "No warnings expected, "
                else:
                    err_msg = f"{len(self.expected_warning)} warnings expected, "
                err_msg += (
                    f"{len(self)} warnings issued. The list of emitted warnings is: "
                    f"{msg_list}. The filenames and line numbers are: {warn_file_line}"
                )
                raise AssertionError(err_msg)

            if expected_length > 0:
                for warn_i, exp_warn in enumerate(self.expected_warning):
                    if not any(issubclass(r.category, exp_warn) for r in self):
                        __tracebackhide__ = True
                        raise AssertionError(
                            "DID NOT WARN. No warnings of type "
                            f"{self.expected_warning} was emitted. The list of "
                            f"emitted warnings is: {[each.message for each in self]}."
                        )
                    elif self.match is not None:
                        for record in self:
                            if (
                                str(record.message).startswith(self.match[warn_i])
                                or re.compile(self.match[warn_i]).search(
                                    str(record.message)
                                )
                            ) and issubclass(
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
        or (inspect.isclass(expected_warning) and issubclass(expected_warning, Warning))
    ):
        raise TypeError("expected_warning must be a list or be derived from Warning")
    if match is not None and not isinstance(match, list | str):
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
            raise TypeError(f"{func!r} object (type: {type(func)}) must be callable")
        with WarningsChecker(expected_warning_list, match_list):
            return func(*args[1:], **kwargs)
