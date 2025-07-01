# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Define UVParameters: data and metadata objects for interferometric data sets.

UVParameters are objects to hold specific data and metadata associated with
interferometric data sets. They are used as attributes for classes based on
UVBase. This module also includes specialized subclasses for particular types
of metadata.

"""

from __future__ import annotations

import builtins
import warnings

import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity

from .utils.tools import _multidim_ind2sub, _strict_raise

__all__ = ["UVParameter", "AngleParameter", "LocationParameter", "SkyCoordParameter"]


def _get_generic_type(expected_type, strict_type_check=False):
    """Return tuple of more generic types.

    Allows for more flexible type checking in the case when a Parameter's value
    changes precision or to/from a numpy dtype but still is the desired generic type.
    If a generic type cannot be found, the expected_type is returned

    Parameters
    ----------
    expected_type : Type or string or list of types or strings
        The expected type of a Parameter object or a string of the name of a type. Lists
        are only for recarray parameters and in that case the input expected_type is
        returned exactly.
    strict_type_check : bool
        If True, the input expected_type is returned exactly.

    Returns
    -------
    Tuple of types based on input expected_type

    """
    if isinstance(expected_type, str):
        try:
            expected_type = getattr(builtins, expected_type)
        except AttributeError as err:
            raise ValueError(
                f"Input expected_type is a string with value: '{expected_type}'. "
                "When the expected_type is a string, it must be a Python builtin type."
            ) from err
    if strict_type_check or isinstance(expected_type, list | tuple):
        return expected_type

    for types in [
        (bool, np.bool_),
        (float, np.floating),
        (np.unsignedinteger),  # unexpected but just in case
        (int, np.integer),
        (complex, np.complexfloating),
    ]:
        if issubclass(expected_type, types):
            return types

    return expected_type


def _param_dict_equal(this_dict, other_dict):
    """
    Test if dicts are equal for parameter equality.

    Helper function pulled out to allow recursion for nested dicts
    """
    try:
        # Try a naive comparison first
        # this will fail if keys are the same
        # but cases differ.
        # so only look for exact equality
        # then default to the long test below.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "elementwise comparison failed")
            if this_dict == other_dict:
                return True, ""
    except (ValueError, TypeError):
        pass
        # this dict may contain arrays or Nones
        # we will need to check each item individually

    # check to see if they are equal other than
    # upper/lower case keys
    this_lower = {
        (k.lower() if isinstance(k, str) else k): v for k, v in this_dict.items()
    }
    other_lower = {
        (k.lower() if isinstance(k, str) else k): v for k, v in other_dict.items()
    }
    if set(this_lower.keys()) != set(other_lower.keys()):
        message_str = ", keys are not the same."
        return False, message_str
    else:
        # need to check if values are close,
        # not just equal
        for key in this_lower:
            if isinstance(this_lower[key], dict):
                # nested dict, use recursion
                subdict_equal, subdict_message = _param_dict_equal(
                    this_lower[key], other_lower[key]
                )
                if subdict_equal:
                    continue
                else:
                    message_str = f", key {key} is a dict" + subdict_message
                    return False, message_str

            # this is not a dict, use other methods
            if this_lower[key] is None or other_lower[key] is None:
                if this_lower[key] is None and other_lower[key] is None:
                    continue
                else:
                    message_str = f", key {key} is not equal"
                    return False, message_str

            if isinstance(this_lower[key], list | np.ndarray | tuple) and isinstance(
                other_lower[key], list | np.ndarray | tuple
            ):
                this_array = np.asarray(this_lower[key])
                other_array = np.asarray(other_lower[key])
                if this_array.shape != other_array.shape:
                    message_str = f", key {key} is not equal"
                    return False, message_str
                if np.allclose(this_array, other_array):
                    continue
                else:
                    message_str = f", key {key} is not equal"
                    return False, message_str
            else:
                # this isn't a list, array or tuple
                try:
                    if np.isclose(this_lower[key], other_lower[key]):
                        continue
                    else:
                        message_str = f", key {key} is not equal"
                        return False, message_str
                except TypeError:
                    # this isn't a type that can be
                    # handled by np.isclose,
                    # test for equality
                    if this_lower[key] == other_lower[key]:
                        continue
                    else:
                        message_str = f", key {key} is not equal"
                        return False, message_str

    return True, ""


class UVParameter:
    """
    Data and metadata objects for interferometric data sets.

    Parameters
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    spoof_val
        A fake value that can be assigned to a non-required UVParameter if the
        metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.
    form : 'str', int or tuple
        Either 'str' or an int (if a single value) or tuple giving information about the
        expected shape of the value. Elements of the tuple may be the name of other
        UVParameters that indicate data shapes.

        Form examples:
            - 'str': a string value
            - ('Nblts', 3): the value should be an array of shape:
               Nblts (another UVParameter name), 3
            - (): a single numeric value
            - 3: the value should be an array of shape (3, )

    description : str
        Description of the data or metadata in the object.
    expected_type : str or list or tuple of str
        The type that the data or metadata should be. Default is int or str if
        form is 'str'.
    acceptable_vals : list, optional
        List giving allowed values for elements of value.
    acceptable_range: 2-tuple, optional
        Tuple giving a range of allowed magnitudes for elements of value.
    tols : float or 2-tuple of float
        Tolerances for testing the equality of UVParameters. Either a single
        absolute value or a tuple of relative and absolute values to be used by
        np.isclose()
    strict_type_check : bool
        When True, the input expected_type is used exactly, otherwise a more
        generic type is found to allow changes in precisions or to/from numpy
        dtypes to not break checks.
    setter : function, optional
        An optional function to be called after the parameter is set.
        The function should take a single argument, the instance of the class
        to which the parameter is attached.

    Attributes
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    spoof_val
        A fake value that can be assigned to a non-required UVParameter if the
        metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.
    form : 'str', int or tuple
        Either 'str' or an int (if a single value) or tuple giving information about the
        expected shape of the value. Elements of the tuple may be the name of other
        UVParameters that indicate data shapes.

        Form examples:
            - 'str': a string value
            - ('Nblts', 3): the value should be an array of shape:
               Nblts (another UVParameter name), 3
            - (): a single numeric value
            - 3: the value should be an array of shape (3, )

    description : str
        Description of the data or metadata in the object.
    expected_type : str or list or tuple of str
        The type that the data or metadata should be. Default is int or str if
        form is 'str'.
    acceptable_vals : list, optional
        List giving allowed values for elements of value.
    acceptable_range: 2-tuple, optional
        Tuple giving a range of allowed magnitudes for elements of value.
    tols : 2-tuple of float
        Relative and absolute tolerances for testing the equality of UVParameters, to be
        used by np.isclose(). Default is (1e-5, 1e-8) if expected_type is either float
        or complex, otherwise (0, 0).
    strict_type_check : bool
        When True, the input expected_type is used exactly, otherwise a more
        generic type is found to allow changes in precisions or to/from numpy
        dtypes to not break checks.

    """

    def __init__(
        self,
        name,
        *,
        required=True,
        value=None,
        spoof_val=None,
        form=(),
        description="",
        expected_type=int,
        acceptable_vals=None,
        acceptable_range=None,
        tols=None,
        strict_type_check=False,
        ignore_eq_none: bool = False,
        setter: callable | None = None,
    ):
        """Init UVParameter object."""
        self.name = name
        self.required = required
        # cannot set a spoof_val for required parameters
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description
        self.form = form
        if self.form == "str":
            self.expected_type = str
            self.strict_type = True
        else:
            self.expected_type = _get_generic_type(
                expected_type, strict_type_check=strict_type_check
            )
            self.strict_type = strict_type_check
        self.acceptable_vals = acceptable_vals
        self.acceptable_range = acceptable_range
        if tols is None:
            excl_list = (str, np.bool_, np.integer, np.unsignedinteger, bool, int)

            if isinstance(self.expected_type, list):
                no_tol = all(issubclass(item, excl_list) for item in self.expected_type)
            else:
                no_tol = any(issubclass(item, self.expected_type) for item in excl_list)

            self.tols = (0, 0) if no_tol else (1e-5, 1e-8)
        elif np.size(tols) == 1:
            # Only one tolerance given, assume absolute, set relative to zero
            self.tols = (0, tols)
        else:
            # relative and absolute tolerances to be used in np.isclose
            self.tols = tols

        self.ignore_eq_none = ignore_eq_none and not required
        self._setter = setter

    def __eq__(self, other, *, silent=False):
        """
        Test if classes match and values are within tolerances.

        Parameters
        ----------
        other : UVParameter or subclass
            The other UVParameter to compare with this one.
        silent : bool
            When set to False (default), descriptive text is printed out when parameters
            do not match. If set to True, this text is not printed.
        """
        if not (
            isinstance(other, self.__class__) and isinstance(self, other.__class__)
        ):
            if not silent:
                print(
                    f"{self.name} parameter classes are different. Left is "
                    f"{type(self.__class__)}, right is {type(other.__class__)}."
                )
            return False

        # if a parameter should be considered equal if one of them is None, exit here.
        if self.ignore_eq_none and (self.value is None or other.value is None):
            return True

        if self.value is None:
            if other.value is not None:
                if not silent:
                    print(f"{self.name} is None on left, but is not None on right.")
                return False
            else:
                return True
        if other.value is None and self.value is not None:
            if not silent:
                print(f"{self.name} is None on right, but is not None on left.")
            return False

        if isinstance(self.value, EarthLocation) and not isinstance(
            other.value, EarthLocation
        ):
            if not silent:
                print(
                    f"{self.name} parameter value is an EarthLocation on left, "
                    f"but is {type(other.value)} on right."
                )
            return False

        if isinstance(self.value, SkyCoord) and not isinstance(other.value, SkyCoord):
            if not silent:
                print(
                    f"{self.name} parameter value is a SkyCoord on left, "
                    f"but is {type(other.value)} on right."
                )
            return False

        if isinstance(self.value, np.recarray):
            # check both recarrays and field names match (order doesn't have to)
            # then iterate through field names and check that each matches
            if not isinstance(other.value, np.recarray):
                if not silent:
                    print(
                        f"{self.name} parameter value is a recarray on left, "
                        f"but is {type(other.value)} on right."
                    )
                return False
            this_names = self.value.dtype.names
            other_names = other.value.dtype.names
            if np.setxor1d(this_names, other_names).size != 0:
                if not silent:
                    print(
                        f"{self.name} parameter value is a recarray, field names "
                        f"are different. Left has names {this_names}, right has "
                        f"names {other_names}."
                    )
                return False
            for name in this_names:
                this_arr = self.value[name]
                other_arr = other.value[name]
                if isinstance(this_arr.item(0), str | np.str_):
                    if not np.all(this_arr == other_arr):
                        if not silent:
                            print(
                                f"{self.name} parameter value is a recarray, values in "
                                f"field {name} are not close. Left has values "
                                f"{this_arr}, right has values {other_arr}."
                            )
                        return False
                else:
                    if not np.allclose(
                        this_arr,
                        other_arr,
                        rtol=self.tols[0],
                        atol=self.tols[1],
                        equal_nan=True,
                    ):
                        if not silent:
                            print(
                                f"{self.name} parameter value is a recarray, values in "
                                f"field {name} are not close.  Left has values "
                                f"{this_arr}, right has values {other_arr}."
                            )
                        return False
        elif isinstance(self.value, np.ndarray) and not isinstance(
            self.value.item(0), str | np.str_
        ):
            if not isinstance(other.value, np.ndarray):
                if not silent:
                    print(
                        f"{self.name} parameter value is an array on left, "
                        f"but is {type(other.value)} on right."
                    )
                return False
            if self.value.shape != other.value.shape:
                if not silent:
                    print(
                        f"{self.name} parameter value is an array, shapes are "
                        f"different. Left has shape {self.value.shape}, right "
                        f"has shape {other.value.shape}."
                    )
                return False

            if isinstance(self.value, Quantity):
                if not isinstance(other.value, Quantity):
                    if not silent:
                        print(
                            f"{self.name} parameter value is a Quantity on left, "
                            f"but is {type(other.value)} on right."
                        )
                    return False

                if not self.value.unit.is_equivalent(other.value.unit):
                    if not silent:
                        print(
                            f"{self.name} parameter value is an astropy Quantity, "
                            "units are not equivalent. Left has units "
                            f"{self.value.unit}, right has units {other.value.unit}."
                        )
                    return False
                if not isinstance(self.tols[1], Quantity):
                    atol_use = self.tols[1] * self.value.unit
                else:
                    atol_use = self.tols[1]
                if not units.allclose(
                    self.value,
                    other.value,
                    rtol=self.tols[0],
                    atol=atol_use,
                    equal_nan=True,
                ):
                    if not silent:
                        print(
                            f"{self.name} parameter value is an astropy Quantity, "
                            "units are equivalent but values are not close."
                        )
                    return False
            else:
                # check to see if strict types are used
                if self.strict_type:
                    # types must match
                    if other.strict_type:
                        # both strict, expected_type must match
                        if self.expected_type != other.expected_type:
                            if not silent:
                                print(
                                    f"{self.name} parameter has incompatible "
                                    f"types. Left is {self.expected_type}, right "
                                    f"is {other.expected_type}"
                                )
                            return False
                    elif not isinstance(self.value.item(0), other.expected_type):
                        if not silent:
                            print(
                                f"{self.name} parameter has incompatible dtypes. "
                                f"Left requires {self.expected_type}, right is "
                                f"{other.value.dtype}"
                            )
                        return False
                elif other.strict_type and not isinstance(
                    other.value.item(0), self.expected_type
                ):
                    # types must match in the other direction
                    if not silent:
                        print(
                            f"{self.name} parameter has incompatible dtypes. "
                            f"Left is {self.value.dtype}, right requires "
                            f"{other.expected_type}"
                        )
                    return False
                if not np.allclose(
                    self.value,
                    other.value,
                    rtol=self.tols[0],
                    atol=self.tols[1],
                    equal_nan=True,
                ):
                    if not silent:
                        print(
                            f"{self.name} parameter value is an array with matching "
                            "shapes, values are not close."
                        )
                    return False
        else:
            # check to see if strict types are used
            if self.strict_type and not isinstance(self.value, other.expected_type):
                # types must match
                if not silent:
                    print(
                        f"{self.name} parameter has incompatible types. Left "
                        f"requires {type(self.value)}, right is "
                        f"{other.expected_type}"
                    )
                return False
            if other.strict_type and not isinstance(other.value, self.expected_type):
                # types must match in the other direction
                if not silent:
                    print(
                        f"{self.name} parameter has incompatible types. Left "
                        f"is {self.expected_type}, right requires "
                        f"{type(other.value)}"
                    )
                return False

            str_type = False
            if isinstance(self.value, str):
                str_type = True
            if isinstance(self.value, list | tuple) and isinstance(self.value[0], str):
                str_type = True
            if isinstance(self.value, np.ndarray) and isinstance(
                self.value.flat[0], str
            ):
                str_type = True

            if not str_type:
                if isinstance(other.value, np.ndarray):
                    if not silent:
                        print(
                            f"{self.name} parameter value is not an array on left "
                            "but right is an array."
                        )
                    return False
                try:
                    if not np.allclose(
                        np.array(self.value),
                        np.array(other.value),
                        rtol=self.tols[0],
                        atol=self.tols[1],
                        equal_nan=True,
                    ):
                        if not silent:
                            print(
                                f"{self.name} parameter value can be cast to an "
                                "array and tested with np.allclose. The values are "
                                "not close"
                            )
                        return False
                except TypeError:
                    if isinstance(self.value, dict):
                        message_str = f"{self.name} parameter is a dict"
                        dict_equal, dict_message_str = _param_dict_equal(
                            self.value, other.value
                        )

                        if dict_equal:
                            return True
                        else:
                            message_str += dict_message_str
                            if not silent:
                                print(message_str)
                            return False
                    else:
                        if self.value != other.value:
                            if not silent:
                                print(
                                    f"{self.name} parameter value has type "
                                    f"{type(self.value)} on left and "
                                    f"{type(other.value)} on right. The values "
                                    "are not equal."
                                )
                            return False

            else:
                if isinstance(self.value, list | np.ndarray | tuple):
                    this_val = self.value
                    other_val = other.value

                    # Catch the case that we have a multidim array, use the flat view
                    if isinstance(this_val, np.ndarray):
                        this_val = this_val.flat
                    if isinstance(other_val, np.ndarray):
                        other_val = other_val.flat
                    if [s.strip() for s in this_val] != [s.strip() for s in other_val]:
                        if not silent:
                            print(
                                f"{self.name} parameter value is a list of "
                                "strings, values are different"
                            )
                        return False
                else:
                    if (self.value.strip() != other.value.strip()) and (
                        self.value.replace("\n", "").replace(" ", "")
                        != other.value.replace("\n", "").replace(" ", "")
                    ):
                        if not silent:
                            print(
                                f"{self.name} parameter value is a string, "
                                "values are different"
                            )
                        return False

        return True

    def __ne__(self, other, *, silent=True):
        """
        Test if classes do not match or values are not within tolerances.

        Parameters
        ----------
        other : UVParameter or subclass
            The other UVParameter to compare with this one.
        silent : bool
            When set to False (default), descriptive text is printed out when parameters
            do not match. If set to True, this text is not printed.
        """
        return not self.__eq__(other, silent=silent)

    def apply_spoof(self):
        """Set value to spoof_val for non-required UVParameters."""
        self.value = self.spoof_val

    def expected_shape(self, uvbase):
        """
        Get the expected shape of the value based on the form.

        Parameters
        ----------
        uvbase : object
            Object with this UVParameter as an attribute. Needed
            because the form can refer to other UVParameters on this object.

        Returns
        -------
        tuple
            The expected shape of the value.
        """
        if self.form == "str":
            return self.form
        elif isinstance(self.form, int | np.integer):
            # Fixed shape, just return the form
            return (self.form,)
        else:
            # Given by other attributes, look up values
            eshape = ()
            for p in self.form:
                if isinstance(p, int | np.integer):
                    eshape = eshape + (p,)
                else:
                    val = getattr(uvbase, p)
                    if val is None:
                        raise ValueError(
                            f"Missing UVBase parameter {p} needed to "
                            f"calculate expected shape of parameter {self.name}"
                        )
                    eshape = eshape + (val,)
            return eshape

    def check_acceptability(self):
        """Check that values are acceptable."""
        if self.acceptable_vals is None and self.acceptable_range is None:
            return True, "No acceptability check"
        else:
            # either acceptable_vals or acceptable_range is set. Prefer acceptable_vals
            if self.acceptable_vals is not None:
                # acceptable_vals are a list of allowed values
                if self.expected_type is str:
                    # strings need to be converted to lower case
                    if isinstance(self.value, str):
                        value_set = {self.value.lower()}
                    elif isinstance(self.value, np.ndarray):
                        # this is an ndarray of unknown dimensions, make it flat
                        value_set = {x.lower() for x in self.value.flat}
                    else:
                        # this is a list or array of strings, make them all lower case
                        value_set = {x.lower() for x in self.value}
                    acceptable_vals = [x.lower() for x in self.acceptable_vals]
                else:
                    if isinstance(self.value, list | np.ndarray):
                        value_set = set(self.value)
                    else:
                        value_set = {self.value}
                    acceptable_vals = self.acceptable_vals
                for elem in value_set:
                    if elem not in acceptable_vals:
                        message = (
                            f"Value {elem}, is not in allowed values: {acceptable_vals}"
                        )
                        return False, message
                return True, "Value is acceptable"
            else:
                # acceptable_range is a tuple giving a range of allowed magnitudes
                testval = np.mean(np.abs(self.value))
                if (testval >= self.acceptable_range[0]) and (
                    testval <= self.acceptable_range[1]
                ):
                    return True, "Value is acceptable"
                else:
                    message = (
                        f"Mean of abs values, {testval}, is not in allowed range: "
                        f"{self.acceptable_range}"
                    )
                    return False, message

    def compare_value(self, value):
        """
        Compare UVParameter value to a supplied value.

        Parameters
        ----------
        value
            The value to compare against that stored in the UVParameter object. Must
            be the same type.

        Returns
        -------
        same : bool
            True if the values are equivalent (or within specified tolerances),
            otherwise false.
        """
        # Catch the case when the values are different types
        if not (
            isinstance(value, self.value.__class__)
            and isinstance(self.value, value.__class__)
        ):
            raise ValueError(
                "UVParameter value and supplied values are of different types."
            )

        # If these are numeric types, handle them via allclose
        if isinstance(value, np.ndarray | int | float | complex):
            # Check that we either have a number or an ndarray
            # Array equal here is a) _much_ faster if identical, and b) will
            # handle things like strings (where tols are not defined).
            check = np.array_equal(value, self.value)
            if check or (
                isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.str_)
            ):
                return check

            return bool(
                not isinstance(value, np.ndarray)
                or value.shape == self.value.shape
                and np.allclose(
                    value,
                    self.value,
                    rtol=self.tols[0],
                    atol=self.tols[1],
                    equal_nan=True,
                )
            )
        else:
            # Otherwise just default to checking equality
            return value == self.value

    def setter(self, inst):
        """Extra operations to perform on the parent instance when setting the param.

        This can be used to do things like cache invalidation, for example,
        if there is a cache of antpairs to blt indices, when the ant_1_array
        or ant_2_array are set, this cache should be cleared here.
        """
        if self._setter is not None:
            self._setter(inst)

    def get_from_form(self, form_dict: dict, strict: bool | None = None):
        """
        Get values along a set of parameterized axes.

        This method should not be called directly by users; instead, it is called by
        various selection-related functions for selecting a subset of values along
        a given axis, whose expected shape is given (at least in part) by a named
        parameter within the object (e.g., "Nblts", "Ntimes", "Nants").

        Parameters
        ----------
        form_dict : dict
            Dictionary which maps axes to index arrays, with keys that are matched
            against entries within UVParameter.form, and values which demark which
            indices should be selected (must be 1D). Values can also be given as None,
            in which case no selection is performed along that axis.
        strict : bool or None
            Only used if no entries in form_dict match what is present in the form_dict
            parameter. If set to True, then an error is raised. If set to False, a
            warning is raised, and the entirety of UVParameter.value is returned. If
            set to None (default), neither a warning nor error is raised, and the
            entirety of UVParameter.value is returned.

        Returns
        -------
        values : list or ndarray or obj
            Values based on the positions selected, as specified in form_dict.
        """
        # Check the no-op case first
        if self.value is None or not (
            isinstance(self.form, tuple) and any(key in form_dict for key in self.form)
        ):
            msg = "form_dict does not match anything in UVParameter.form" + (
                "." if strict else ", returning whole value."
            )
            _strict_raise(err_msg=msg, strict=strict)
            return self.value

        val_slice = [form_dict.get(key, slice(None)) for key in self.form]
        # Which axes are index lists versus slices?
        ind_axes = [i for i, itm in enumerate(val_slice) if not isinstance(itm, slice)]
        extra_axes = {}
        if len(ind_axes) > 1:
            for axis in ind_axes:
                # Allow the first list to be handled with fancy indexing. After
                # that, it's time to use np.take!
                extra_axes[axis] = val_slice[axis]
                val_slice[axis] = slice(None)

        if isinstance(
            self.value, np.ndarray | np.ma.MaskedArray | SkyCoord | Time | Quantity
        ):
            # If we're working with an ndarray, use take to slice along
            # the axis that we want to grab from.
            value = self.value[tuple(val_slice)]
            if extra_axes:
                ind_arr, new_shape = _multidim_ind2sub(extra_axes, value.shape)
                if isinstance(self.value, SkyCoord | Time):
                    value = (value.ravel())[ind_arr].reshape(new_shape)
                else:
                    value = value.flat[ind_arr].reshape(new_shape)
        elif isinstance(self.value, list):
            # If this is a list, it _should_ always have 1-dimension.
            assert len(val_slice) == 1, (
                "Something is wrong, len(UVParameter.form) != 1 when selecting on a "
                "list, which should not be possible. Please file an "
                "issue in our GitHub issue log so that we can fix it."
            )
            if isinstance(val_slice[0], slice):
                value = self.value[val_slice[0]]
            else:
                value = [self.value[idx] for idx in val_slice[0]]
        return value

    def set_from_form(
        self, form_dict: dict, values: list | np.ndarray, strict: bool | None = False
    ):
        """
        Set values along a set of parameterized axes.

        This method should not be called directly by users; instead, it is called by
        various selection-related functions for setting a subset of values along
        a given axis, whose expected shape is given (at least in part) by a named
        parameter within the object (e.g., "Nblts", "Ntimes", "Nants").

        Parameters
        ----------
        form_dict : dict
            Dictionary which maps axes to index arrays, with keys that are matched
            against entries within UVParameter.form, and values which demark which
            indices should be selected (must be 1D). Values can also be given as None,
            in which case no selection is performed along that axis.
        values : list or ndarray
            Values based on the positions selected, as specified in form_dict.
        strict : bool or None
            Only used if no entries in form_dict match what is present in the form_dict
            parameter. If set to True, then an error is raised. If set to False, a
            warning is raised, and the entirety of UVParameter.value set to be equal to
            the values argument. If set to None (default), neither a warning nor error
            is raised, and the entirety of UVParameter.value is set.
        """
        # Check the no-op case first
        if not (
            isinstance(self.form, tuple) and any(key in form_dict for key in self.form)
        ):
            msg = "form_dict does not match anything in UVParameter.form" + (
                "." if strict else ", setting the whole value."
            )
            _strict_raise(err_msg=msg, strict=strict)

            self.value = values
            return

        # Validate that we actually have something to plug into
        if self.value is None:
            raise ValueError("Cannot call set_from_form if UVParameter.value is None.")

        val_slice = [form_dict.get(key, slice(None)) for key in self.form]
        # Which axes are index lists versus slices?
        ind_axes = [i for i, itm in enumerate(val_slice) if not isinstance(itm, slice)]
        extra_axes = {}
        if len(ind_axes) > 1:
            for axis in ind_axes:
                # Allow the first list to be handled with fancy indexing. After
                # that, it's time to use np.take!
                extra_axes[axis] = val_slice[axis]
                val_slice[axis] = slice(None)

        if isinstance(self.value, Quantity | SkyCoord | Time):
            if extra_axes:
                # Quantity, SkyCoord, and TIme have some extra caching that causes
                # some headaches, so we effectively get a copy of the subarray to
                # plug individual values into before going back to the "master" array
                temp_arr = self.value[tuple(val_slice)]
                temp_shape = temp_arr.shape
                ind_arr, _ = _multidim_ind2sub(extra_axes, temp_arr.shape)
                temp_arr = temp_arr.ravel()
                temp_arr[ind_arr] = values.ravel()
                self.value[tuple(val_slice)] = temp_arr.reshape(temp_shape)
            else:
                # If regular indexing solves this, then that's how we'll roll
                self.value[tuple(val_slice)] = values
        elif isinstance(self.value, np.ndarray | np.ma.MaskedArray):
            if extra_axes:
                # If we have multiple list-based selections, then we slice first and
                # then use multi_index_ravel to generate indices that we can plug in.
                # That way we only need to create one extra array whose size is
                # the same as values (for the indices).
                temp_arr = self.value[tuple(val_slice)]

                # Make sure this is a view that is connected in memory
                assert temp_arr.base is (
                    self.value if self.value.base is None else self.value.base
                ), (
                    "Something is wrong, slicing self.value does not return a view "
                    "on the original array. Please file an issue in our GitHub "
                    "issue log so that we can fix it."
                )
                ind_arr, _ = _multidim_ind2sub(extra_axes, temp_arr.shape)

                # N.b., later if needed we can add mask handling here (e.g., for data
                # parameters).
                temp_arr.flat[ind_arr] = values.flat
            else:
                # If regular indexing solves this, then that's how we'll roll
                self.value[tuple(val_slice)] = values
        elif isinstance(self.value, list):
            # If this is a list, it _should_ always have 1-dimension.
            assert len(val_slice) == 1, (
                "Something is wrong, len(UVParameter.form) != 1 when setting a "
                "list, which should not be possible. Please file an "
                "issue in our GitHub issue log so that we can fix it."
            )
            if isinstance(val_slice[0], slice):
                self.value[val_slice[0]] = values
            else:
                for idx, ind_val in zip(val_slice[0], values, strict=True):
                    self.value[idx] = ind_val


class AngleParameter(UVParameter):
    """
    Subclass of UVParameter for Angle type parameters.

    Adds extra methods for conversion to & from degrees (used by UVBase objects
    for _degrees properties associated with these parameters).

    Parameters
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    spoof_val
        A fake value that can be assigned to a non-required UVParameter if the
        metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.
    form : 'str', int or tuple
        Either 'str' or an int (if a single value) or tuple giving information about the
        expected shape of the value. Elements of the tuple may be the name of other
        UVParameters that indicate data shapes.

        Form examples:
            - 'str': a string value
            - ('Nblts', 3): the value should be an array of shape:
               Nblts (another UVParameter name), 3
            - (): a single numeric value
            - 3: the value should be an array of shape (3, )

    description : str
        Description of the data or metadata in the object.
    expected_type
        The type that the data or metadata should be. Default is int or str if
        form is 'str'.
    acceptable_vals : list, optional
        List giving allowed values for elements of value.
    acceptable_range: 2-tuple, optional
        Tuple giving a range of allowed magnitudes for elements of value.
    tols : float or 2-tuple of float
        Tolerances for testing the equality of UVParameters. Either a single
        absolute value or a tuple of relative and absolute values to be used by
        np.isclose()
    strict_type_check : bool
        When True, the input expected_type is used exactly, otherwise a more
        generic type is found to allow changes in precicions or to/from numpy
        dtypes to not break checks.

    Attributes
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    spoof_val
        A fake value that can be assigned to a non-required UVParameter if the
        metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.
    form : 'str', int or tuple
        Either 'str' or an int (if a single value) or tuple giving information about the
        expected shape of the value. Elements of the tuple may be the name of other
        UVParameters that indicate data shapes.

        Form examples:
            - 'str': a string value
            - ('Nblts', 3): the value should be an array of shape:
               Nblts (another UVParameter name), 3
            - (): a single numeric value
            - 3: the value should be an array of shape (3, )

    description : str
        Description of the data or metadata in the object.
    expected_type
        The type that the data or metadata should be. Default is int or str if
        form is 'str'.
    acceptable_vals : list, optional
        List giving allowed values for elements of value.
    acceptable_range: 2-tuple, optional
        Tuple giving a range of allowed magnitudes for elements of value.
    tols : 2-tuple of float
        Relative and absolute tolerances for testing the equality of UVParameters, to be
        used by np.isclose()
    strict_type_check : bool
        When True, the input expected_type is used exactly, otherwise a more
        generic type is found to allow changes in precicions or to/from numpy
        dtypes to not break checks.

    """

    def degrees(self):
        """Get value in degrees."""
        if self.value is None:
            return None
        else:
            return self.value * 180.0 / np.pi

    def set_degrees(self, degree_val):
        """
        Set value in degrees.

        Parameters
        ----------
        degree_val : float
            Value in degrees to use to set the value attribute.
        """
        if degree_val is None:
            self.value = None
        else:
            self.value = degree_val * np.pi / 180.0


class LocationParameter(UVParameter):
    """
    Subclass of UVParameter for location type parameters.

    Supports either :class:`astropy.coordinates.EarthLocation` objects or
    :class:`lunarsky.MoonLocation` objects.
    Adds extra methods for conversion to & from lat/lon/alt in radians or
    degrees (used by UVBase objects for _lat_lon_alt and _lat_lon_alt_degrees
    properties associated with these parameters).

    Parameters
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    description : str
        Description of the data or metadata in the object.
    tols : float or 2-tuple of float
        Tolerances for testing the equality of UVParameters. Either a single
        absolute value or a tuple of relative and absolute values to be used by
        np.isclose()

    Attributes
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    spoof_val
        Always set to None.
    form : int
       Always set to None.
    description : str
        Description of the data or metadata in the object.
    expected_type
        Set to EarthLocation or MoonLocation
    acceptable_vals : list, optional
        Always set to None.
    acceptable_range: 2-tuple, optional
        Always set to None.
    tols : 2-tuple of float
        Relative and absolute tolerances for testing the equality of UVParameters, to be
        used by np.isclose()
    strict_type_check : bool
        When True, the input expected_type is used exactly, otherwise a more
        generic type is found to allow changes in precicions or to/from numpy
        dtypes to not break checks.

    """

    def __init__(
        self,
        name,
        *,
        required=True,
        value=None,
        spoof_val=None,
        description="",
        tols=1e-3,
    ):
        super().__init__(
            name,
            required=required,
            value=value,
            spoof_val=spoof_val,
            description=description,
            expected_type=(EarthLocation,),
            strict_type_check=True,
            acceptable_range=None,
            tols=tols,
        )

        # call self.on_moon to set expected types properly
        _ = self.on_moon

    @property
    def on_moon(self):
        """Detect if this location is on the moon."""
        # figure out if this is on the moon. Doing it this way limits
        # attempted imports of lunarsky
        if isinstance(self.value, EarthLocation):
            return False
        else:
            try:
                from lunarsky import MoonLocation

                if isinstance(self.value, MoonLocation):
                    self.expected_type = (EarthLocation, MoonLocation)
                    return True
            except ImportError:
                return False

    @property
    def frame(self):
        """Get the frame."""
        if self.on_moon:
            return "mcmf"
        else:
            return "itrs"

    @property
    def ellipsoid(self):
        """Get the ellipsoid."""
        if self.on_moon:
            return self.value.ellipsoid
        else:
            return None

    def xyz(self):
        """Get the body centric coordinates in meters."""
        if self.value is None:
            return None
        elif self.on_moon:
            centric_coords = self.value.selenocentric
        else:
            centric_coords = self.value.geocentric
        return Quantity(centric_coords).to_value("m")

    def set_xyz(self, xyz, *, frame=None, ellipsoid=None):
        """Set the body centric coordinates in meters."""
        if frame is None:
            if self.on_moon:
                frame = "mcmf"
            else:
                frame = "itrs"

        allowed_frames = ["itrs"]
        if self.on_moon:
            allowed_frames += ["mcmf"]
        if frame not in allowed_frames:
            raise ValueError(f"frame must be one of {allowed_frames}")

        if xyz is None:
            self.value = None
        else:
            if frame == "itrs":
                self.value = EarthLocation.from_geocentric(*(xyz * units.m))
            else:
                from lunarsky import MoonLocation

                if ellipsoid is None and isinstance(self.value, MoonLocation):
                    ellipsoid = self.value.ellipsoid
                moonloc = MoonLocation.from_selenocentric(*(xyz * units.m))
                if ellipsoid is not None:
                    moonloc.ellipsoid = ellipsoid
                self.value = moonloc

    def lat_lon_alt(self):
        """Get value in (latitude, longitude, altitude) tuple in radians."""
        if self.value is None:
            return None
        else:
            lat = self.value.lat.rad
            lon = self.value.lon.rad
            alt = self.value.height.to_value("m")
            return lat, lon, alt

    def set_lat_lon_alt(self, lat_lon_alt, ellipsoid=None):
        """
        Set value from (latitude, longitude, altitude) tuple in radians.

        Parameters
        ----------
        lat_lon_alt : 3-tuple of float
            Tuple with the latitude (radians), longitude (radians)
            and altitude (meters) to use to set the value attribute.
        ellipsoid : str or None
            Ellipsoid to use to convert between lat/lon/alt and xyz. Only used
            for MoonLocation objects.
        """
        if lat_lon_alt is None:
            self.value = None
        else:
            if self.on_moon:
                from lunarsky import MoonLocation

                if ellipsoid is None:
                    ellipsoid = self.value.ellipsoid
                self.value = MoonLocation.from_selenodetic(
                    lon=lat_lon_alt[1] * units.rad,
                    lat=lat_lon_alt[0] * units.rad,
                    height=lat_lon_alt[2] * units.m,
                    ellipsoid=ellipsoid,
                )
            else:
                self.value = EarthLocation.from_geodetic(
                    lon=lat_lon_alt[1] * units.rad,
                    lat=lat_lon_alt[0] * units.rad,
                    height=lat_lon_alt[2] * units.m,
                )

    def lat_lon_alt_degrees(self):
        """Get value in (latitude, longitude, altitude) tuple in degrees."""
        if self.value is None:
            return None
        else:
            lat = self.value.lat.deg
            lon = self.value.lon.deg
            alt = self.value.height.to_value("m")
            return lat, lon, alt

    def set_lat_lon_alt_degrees(self, lat_lon_alt_degree, ellipsoid=None):
        """
        Set value from (latitude, longitude, altitude) tuple in degrees.

        Parameters
        ----------
        lat_lon_alt : 3-tuple of float
            Tuple with the latitude (degrees), longitude (degrees)
            and altitude (meters) to use to set the value attribute.
        ellipsoid : str or None
            Ellipsoid to use to convert between lat/lon/alt and xyz. Only used
            for MoonLocation objects.

        """
        if lat_lon_alt_degree is None:
            self.value = None
        else:
            lat_lon_alt = (
                lat_lon_alt_degree[0] * np.pi / 180.0,
                lat_lon_alt_degree[1] * np.pi / 180.0,
                lat_lon_alt_degree[2],
            )
            self.set_lat_lon_alt(lat_lon_alt, ellipsoid=ellipsoid)

    def check_acceptability(self):
        """Check that value is an allowed object type."""
        if not isinstance(self.value, self.expected_type):
            return (False, f"Location must be an object of type: {self.expected_type}")
        else:
            return True, ""

    def __eq__(self, other, *, silent=False):
        """Handle equality properly for Earth/Moon Location objects."""
        # First check that other is a LocationParameter
        if not (
            isinstance(other, self.__class__) and isinstance(self, other.__class__)
        ):
            if not silent:
                print(
                    f"{self.name} parameter classes are different. Left is "
                    f"{type(self.__class__)}, right is {type(other.__class__)}."
                )
            return False

        if not isinstance(self.value, self.expected_type) and not isinstance(
            other.value, other.expected_type
        ):
            # these are not proper location types (e.g. None), call the super
            return super().__eq__(other, silent=silent)

        if not isinstance(self.value, self.expected_type) or not isinstance(
            other.value, other.expected_type
        ):
            # one has a location type, the other does not
            if not silent:
                print(
                    f"{self.name} parameter values are locations types in one "
                    f"object and not in the other. Left is type {type(self.value)}, "
                    f"right is type {type(other.value)}."
                )
            return False

        if not isinstance(other.value, self.value.__class__):
            if not silent:
                print(
                    f"{self.name} parameter value classes do not match. Left is "
                    f"type {type(self.value)}, right is type {type(other.value)}."
                )
            return False

        if self.on_moon and self.value.ellipsoid != other.value.ellipsoid:
            if not silent:
                print(
                    f"{self.name} parameter value ellipsoid is not the same. "
                    f"Left is {self.value.ellipsoid}, right is {other.value.ellipsoid}."
                )
            return False

        if not np.allclose(
            self.xyz(), other.xyz(), rtol=self.tols[0], atol=self.tols[1]
        ):
            if not silent:
                print(
                    f"{self.name} parameter values have the same class but "
                    "the values are not close."
                )
            return False

        return True


class SkyCoordParameter(UVParameter):
    """
    Subclass of UVParameter for SkyCoord parameters.

    Needed for handling tolerances properly. The `tols` attribute is interpreted as the
    tolerance of the sky separation in radians.

    Parameters
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    spoof_val
        A fake value that can be assigned to a non-required UVParameter if the
        metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.
    form : 'str', int or tuple
        Either 'str' or an int (if a single value) or tuple giving information about the
        expected shape of the value. Elements of the tuple may be the name of other
        UVParameters that indicate data shapes.

        Form examples:
            - 'str': a string value
            - ('Nblts', 3): the value should be an array of shape:
               Nblts (another UVParameter name), 3
            - (): a single numeric value
            - 3: the value should be an array of shape (3, )

    description : str
        Description of the data or metadata in the object.
    acceptable_range: 2-tuple, optional
        Tuple giving a range of allowed magnitudes for elements of value.
    radian_tol : float
        Tolerance of the sky separation in radians.

    Attributes
    ----------
    name : str
        The name of the attribute. Used as the associated property name in
        classes based on UVBase.
    required : bool
        Flag indicating whether this is required metadata for
        the class with this UVParameter as an attribute. Default is True.
    value
        The value of the data or metadata.
    spoof_val
        A fake value that can be assigned to a non-required UVParameter if the
        metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.
    form : 'str', int or tuple
        Either 'str' or an int (if a single value) or tuple giving information about the
        expected shape of the value. Elements of the tuple may be the name of other
        UVParameters that indicate data shapes.

        Form examples:
            - 'str': a string value
            - ('Nblts', 3): the value should be an array of shape:
               Nblts (another UVParameter name), 3
            - (): a single numeric value
            - 3: the value should be an array of shape (3, )

    description : str
        Description of the data or metadata in the object.
    expected_type
        Always set to SkyCoord.
    acceptable_range: 2-tuple, optional
        Tuple giving a range of allowed magnitudes for elements of value.
    tols : 2-tuple of float
        Set to (0, `radian_tol`).
    strict_type_check : bool
        When True, the input expected_type is used exactly, otherwise a more
        generic type is found to allow changes in precicions or to/from numpy
        dtypes to not break checks.

    """

    def __init__(
        self,
        name,
        *,
        required=True,
        value=None,
        spoof_val=None,
        form=(),
        description="",
        acceptable_range=None,
        # standard angle tolerance: 1 mas in radians.
        radian_tol=1 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0),
    ):
        super().__init__(
            name,
            required=required,
            value=value,
            spoof_val=spoof_val,
            form=form,
            description=description,
            expected_type=SkyCoord,
            acceptable_range=acceptable_range,
            tols=(0, radian_tol),
        )

    def __eq__(self, other, *, silent=False):
        """
        Test if classes match and values are within tolerances.

        Parameters
        ----------
        other : UVParameter or subclass
            The other UVParameter to compare with this one.
        silent : bool
            When set to False (default), descriptive text is printed out when parameters
            do not match. If set to True, this text is not printed.
        """
        if not issubclass(self.value.__class__, SkyCoord) or not issubclass(
            other.value.__class__, SkyCoord
        ):
            return super().__eq__(other, silent=silent)

        if self.value.shape != other.value.shape:
            if not silent:
                print(f"{self.name} parameter shapes are different")
            return False

        this_frame = self.value.frame.name
        other_frame = other.value.frame.name
        if this_frame != other_frame:
            if not silent:
                print(
                    f"{self.name} parameter has different frames, {this_frame} vs "
                    f"{other_frame}."
                )
            return False

        this_rep_type = self.value.representation_type
        other_rep_type = other.value.representation_type
        if this_rep_type != other_rep_type:
            if not silent:
                print(
                    f"{self.name} parameter has different representation_types, "
                    f"{this_rep_type} vs {other_rep_type}."
                )
            return False

        # finally calculate on sky separations
        sky_separation = self.value.separation(other.value).rad
        if np.any(sky_separation > self.tols[1]):
            if not silent:
                print(f"{self.name} parameter is not close. ")
            return False

        return True
