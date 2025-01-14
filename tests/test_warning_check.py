# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for testing_utils module."""

import re
import warnings

import pytest

from pyuvdata.testing.warning_check import WarningsChecker, check_warnings


@pytest.mark.parametrize(
    [
        "true_warn_type",
        "true_warn_msg",
        "exp_warn_type",
        "exp_warn_msg",
        "err_type",
        "err_msg",
    ],
    [
        [
            DeprecationWarning,
            "the warning message",
            UserWarning,
            "the warning message",
            AssertionError,
            re.escape(
                "DID NOT WARN. No warnings of type [<class 'UserWarning'>] was "
                "emitted. The list of emitted warnings is: "
                "[DeprecationWarning('the warning message')]."
            ),
        ],
        [
            UserWarning,
            "different message",
            UserWarning,
            "the warning message",
            AssertionError,
            re.escape(
                "No warnings of type <class 'UserWarning'> matching ('the warning "
                "message') was emitted. The list of emitted warnings is: "
                "[UserWarning('different message')]"
            ),
        ],
        [
            None,
            "",
            UserWarning,
            "the warning message",
            AssertionError,
            re.escape(
                "No warnings of type <class 'UserWarning'> matching ('the warning "
                "message') was emitted. The list of emitted warnings is: "
                "[UserWarning('')]."
            ),
        ],
        [
            UserWarning,
            "the warning message",
            None,
            "the warning message",
            AssertionError,
            re.escape(
                "No warnings expected, 1 warnings issued. The list of emitted "
                "warnings is: [UserWarning('the warning message')]. The filenames "
                "and line numbers are"
            ),
        ],
        [
            UserWarning,
            "the warning message",
            None,
            12.1,
            TypeError,
            "match must be a list or a string.",
        ],
        [
            [UserWarning, UserWarning],
            ["the warning message", "a different message"],
            UserWarning,
            "the warning message",
            AssertionError,
            re.escape(
                "1 warnings expected, 2 warnings issued. The list of emitted "
                "warnings is: [UserWarning('the warning message'), "
                "UserWarning('a different message')]. The filenames and line "
                "numbers are:"
            ),
        ],
        [
            UserWarning,
            "the warning message",
            [UserWarning, UserWarning],
            ["the warning message", "a different message"],
            AssertionError,
            re.escape(
                "2 warnings expected, 1 warnings issued. The list of emitted "
                "warnings is: [UserWarning('the warning message')]. The filenames "
                "and line numbers are:"
            ),
        ],
        [
            UserWarning,
            "the warning message",
            (UserWarning, UserWarning),
            ["the warning message", "a different message"],
            TypeError,
            "expected_warning must be a list or be derived from Warning",
        ],
        [
            [UserWarning, UserWarning],
            ["the warning message", "a different message"],
            [UserWarning, UserWarning],
            ["the warning message", "a different message", "a third message"],
            ValueError,
            "If expected_warning and match both have more than one element, "
            "they must be the same length.",
        ],
    ],
)
def test_check_warnings_errors(
    true_warn_type, true_warn_msg, exp_warn_type, exp_warn_msg, err_type, err_msg
):
    with (
        pytest.raises(err_type, match=err_msg),
        check_warnings(exp_warn_type, match=exp_warn_msg),
    ):
        if true_warn_msg is not None:
            if isinstance(true_warn_type, list):
                for warn_type, warn_msg in zip(
                    true_warn_type, true_warn_msg, strict=True
                ):
                    warnings.warn(warn_msg, warn_type)
            else:
                warnings.warn(true_warn_msg, true_warn_type)
        else:
            pass


def test_check_warnings_errors_no_cm():
    with pytest.raises(
        TypeError, match="Unexpected keyword arguments passed to check_warnings: foo"
    ):
        check_warnings(UserWarning, "a message", foo=5)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "'this is not a function' object (type: <class 'str'>) must be callable"
        ),
    ):
        check_warnings(UserWarning, "a message", 2, "this is not a function")

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 warnings expected, 0 warnings issued. The list of emitted "
            "warnings is: []. The filenames and line numbers are: []"
        ),
    ):
        check_warnings(UserWarning, "a message", 1, print, "foo")


@pytest.mark.parametrize(
    ["exp_warn", "exp_msg", "err_type", "err_msg"],
    [
        [
            "foo",
            "a message",
            TypeError,
            "exceptions must be derived from Warning, not <class 'str'>",
        ],
        [
            ["foo"],
            "a message",
            TypeError,
            "exceptions must be derived from Warning, not <class 'str'>",
        ],
        [UserWarning, 5, TypeError, "match must be a str, not <class 'int'>"],
        [UserWarning, [5], TypeError, "match must be a str, not <class 'int'>"],
    ],
)
def test_warnings_checker_errors(exp_warn, exp_msg, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        WarningsChecker(expected_warning=exp_warn, match=exp_msg)


@pytest.mark.parametrize(
    ["warn_type", "warn_msg", "exp_warn", "exp_msg"],
    [
        [UserWarning, "a message", [UserWarning], ["a message"]],
        [UserWarning, None, [UserWarning], None],
        [UserWarning, ["a message"], [UserWarning], ["a message"]],
    ],
)
def test_warning_checker(warn_type, warn_msg, exp_warn, exp_msg):
    wc_obj = WarningsChecker(warn_type, match=warn_msg)
    assert wc_obj.expected_warning == exp_warn
    assert wc_obj.match == exp_msg


def test_warning_checker_methods():
    with WarningsChecker(expected_warning=None, match="") as wc_obj:
        warnings.warn("this message")
        warnings.warn("another message")

        assert str(wc_obj.warnlist[0].message) == "this message"
        assert str(wc_obj[1].message) == "another message"

        assert str(wc_obj.pop(UserWarning).message) == "this message"

        with pytest.raises(
            AssertionError,
            match="<class 'DeprecationWarning'> not found in warning list",
        ):
            wc_obj.pop(DeprecationWarning)

        with pytest.raises(
            RuntimeError,
            match=re.escape("Cannot enter WarningsChecker(record=True) twice"),
        ):
            wc_obj.__enter__()
        wc_obj.clear()

    wc_obj = WarningsChecker(expected_warning=None, match="")
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Cannot exit WarningsChecker(record=True) without entering first"
        ),
    ):
        wc_obj.__exit__()
