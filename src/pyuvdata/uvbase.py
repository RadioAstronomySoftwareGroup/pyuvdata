# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Base class for objects with UVParameter attributes.

Subclassed by UVData and Telescope.
"""

import copy
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from . import __version__, parameter as uvp
from .utils.tools import _get_iterable, slicify

__all__ = ["UVBase"]


def _warning(msg, *a, **kwargs):
    """
    Improve the printing of user warnings.

    Parameters
    ----------
    msg : str
        Input warning message.
    a
        postional parameters not used by this formatting method.
    kwargs
        named parameters not used by this formatting method.

    Returns
    -------
    str
        Input warning message with new line character appended to improve warning
        formatting.

    """
    return str(msg) + "\n"


class UVBase:
    """
    Base class for objects with UVParameter attributes.

    This class is intended to be subclassed and its init method should be
    called in the subclass init after all associated UVParameter attributes are
    defined. The init method of this base class creates properties
    (named using UVParameter.name) from all the UVParameter attributes on the subclass.
    AngleParameter and LocationParameter attributes also have extra convenience
    properties defined:

    AngleParameter:

        UVParameter.name+'_degrees'

    LocationParameter:

        UVParameter.name+'_lat_lon_alt'
        UVParameter.name+'_lat_lon_alt_degrees'
    """

    def _setup_parameters(self):
        """Set up parameter objects to be able to be referenced by their names."""
        # set any UVParameter attributes to be properties
        for p in self:
            this_param = getattr(self, p)
            attr_name = this_param.name
            setattr(
                self.__class__,
                attr_name,
                property(self.prop_fget(p), self.prop_fset(p)),
            )
            if isinstance(this_param, uvp.AngleParameter):
                setattr(
                    self.__class__,
                    attr_name + "_degrees",
                    property(self.degree_prop_fget(p), self.degree_prop_fset(p)),
                )
            elif isinstance(this_param, uvp.LocationParameter):
                setattr(
                    self.__class__,
                    attr_name + "_lat_lon_alt",
                    property(
                        self.lat_lon_alt_prop_fget(p), self.lat_lon_alt_prop_fset(p)
                    ),
                )
                setattr(
                    self.__class__,
                    attr_name + "_lat_lon_alt_degrees",
                    property(
                        self.lat_lon_alt_degrees_prop_fget(p),
                        self.lat_lon_alt_degrees_prop_fset(p),
                    ),
                )
        return

    def __init__(self):
        """Create properties from UVParameter attributes."""
        warnings.formatwarning = _warning

        self._setup_parameters()

        # String to add to history of any files written with this version of pyuvdata
        self.pyuvdata_version_str = (
            f"  Read/written with pyuvdata version: {__version__}."
        )

    def __setstate__(self, state):
        """
        Set the state of the object from given input state.

        This is useful for pickling.

        Parameters
        ----------
        state
            input state to assign to the __dict__.

        """
        self.__dict__ = state
        self._setup_parameters()

    def prop_fget(self, param_name):
        """
        Getter method for UVParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to get, corresponds the the UVParameter.name.

        Returns
        -------
        fget
            getter method to use for the property definition.

        """

        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.value

        return fget

    def prop_fset(self, param_name):
        """
        Setter method for UVParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to set, corresponds the the UVParameter.name.

        Returns
        -------
        fset
            setter method to use for the property definition.

        """

        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.value = value
            this_param.setter(self)
            setattr(self, param_name, this_param)
            if isinstance(this_param, uvp.LocationParameter):
                # call self.on_moon to set expected types properly
                _ = this_param.on_moon

        return fset

    def degree_prop_fget(self, param_name):
        """
        Degree getter method for AngleParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to get, corresponds the the UVParameter.name with "_degrees"
            appended.

        Returns
        -------
        fget
            getter method to use for the property definition.

        """

        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.degrees()

        return fget

    def degree_prop_fset(self, param_name):
        """
        Degree setter method for AngleParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to set, corresponds the the UVParameter.name with "_degrees"
            appended.

        Returns
        -------
        fset
            setter method to use for the property definition.

        """

        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_degrees(value)
            this_param.setter(self)
            setattr(self, param_name, this_param)

        return fset

    def lat_lon_alt_prop_fget(self, param_name):
        """
        Lat/lon/alt getter method for LocationParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to get, corresponds the the UVParameter.name with
            "_lat_lon_alt" appended.

        Returns
        -------
        fget
            getter method to use for the property definition.

        """

        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt()

        return fget

    def lat_lon_alt_prop_fset(self, param_name):
        """
        Lat/lon/alt setter method for LocationParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to set, corresponds the the UVParameter.name with
            "_lat_lon_alt" appended.

        Returns
        -------
        fset
            setter method to use for the property definition.

        """

        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt(value)
            this_param.setter(self)
            setattr(self, param_name, this_param)

        return fset

    def lat_lon_alt_degrees_prop_fget(self, param_name):
        """
        Lat/lon/alt degree getter method for LocationParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to get, corresponds the the UVParameter.name with
            "_lat_lon_alt_degrees" appended.

        Returns
        -------
        fget
            getter method to use for the property definition.

        """

        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt_degrees()

        return fget

    def lat_lon_alt_degrees_prop_fset(self, param_name):
        """
        Lat/lon/alt degree setter method for LocationParameter properties.

        Parameters
        ----------
        param_name : str
            Property name to set, corresponds the the UVParameter.name with
            "_lat_lon_alt_degrees" appended.

        Returns
        -------
        fset
            setter method to use for the property definition.

        """

        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt_degrees(value)
            this_param.setter(self)
            setattr(self, param_name, this_param)

        return fset

    def __iter__(self, uvparams_only=True):
        """
        Iterate over all (UVParameter) attributes.

        Parameters
        ----------
        uvparams_only : bool
            Option to only iterate over UVParameter attributes.

        Yields
        ------
        attribute : UVParameter or any type
            Object attributes, exclusively UVParameter objects if uvparams_only is True.

        """
        if uvparams_only:
            attribute_list = [
                a
                for a in dir(self)
                if a.startswith("_") and isinstance(getattr(self, a), uvp.UVParameter)
            ]
        else:
            attribute_list = [
                a
                for a in dir(self)
                if not a.startswith("__") and not callable(getattr(self, a))
            ]
        param_list = []
        for a in attribute_list:
            if uvparams_only:
                attr = getattr(self, a)
                if isinstance(attr, uvp.UVParameter):
                    param_list.append(a)
            else:
                param_list.append(a)
        for a in param_list:
            yield a

    def required(self):
        """
        Iterate over all required UVParameter attributes.

        Yields
        ------
        UVParameter
            required UVParameters on this object.

        """
        attribute_list = list(self.__iter__(uvparams_only=True))
        required_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if attr.required:
                required_list.append(a)
        for a in required_list:
            yield a

    def extra(self):
        """
        Iterate over all non-required UVParameter attributes.

        Yields
        ------
        UVParameter
            optional (non-required) UVParameters on this object.

        """
        attribute_list = list(self.__iter__(uvparams_only=True))
        extra_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if not attr.required:
                extra_list.append(a)
        for a in extra_list:
            yield a

    def __eq__(
        self, other, *, check_extra=True, allowed_failures=("filename",), silent=False
    ):
        """
        Test if classes match and parameters are equal.

        Parameters
        ----------
        other : class
            Other class instance to check
        check_extra : bool
            Option to specify whether to include all parameters, or just the
            required ones. Default is True.
        allowed_failures : iterable of str, optional
            List or tuple of parameter names that are allowed to fail while
            still passing an overall equality check. These should only include
            optional parameters. By default, the `filename` parameter will be
            ignored. 'blt_order' is also ignored, because currently it is ascertained
            directly, if not provided, by UVH5 files, but not by other file types.
            In any case, if it was to fail, other parameters would fail as well.
        silent : bool
            Option to turn off printing explanations of why equality fails. Useful to
            prevent __ne__ from printing lots of messages.

        Returns
        -------
        bool
            True if the two instances are equivalent.

        """
        if isinstance(other, self.__class__):
            # only check that required parameters are identical
            if hasattr(self, "metadata_only"):
                self.metadata_only  # noqa B018
                other.metadata_only  # noqa B018

            self_required = set(self.required())
            other_required = set(other.required())
            if self_required != other_required:
                if not silent:
                    print(
                        "Sets of required parameters do not match. \n"
                        f"Left is {self_required},\n"
                        f" right is {other_required}.\n\n Left has "
                        f"{self_required.difference(other_required)} extra."
                        f" Right has {other_required.difference(self_required)} extra."
                    )
                return False

            if check_extra:
                self_extra = []
                other_extra = []
                for param in self.extra():
                    self_extra.append(param)
                for param in other.extra():
                    other_extra.append(param)
                if set(self_extra) != set(other_extra):
                    if not silent:
                        print(
                            "Sets of extra parameters do not match. "
                            f"Left is {self_extra},"
                            f" right is {other_extra}."
                        )
                    return False
                p_check = list(self_required) + self_extra
            else:
                p_check = list(self_required)

            if allowed_failures is not None:
                if isinstance(allowed_failures, str):
                    # convert a single string into a length-1 list
                    allowed_failures = [allowed_failures]
                if isinstance(allowed_failures, tuple):
                    # convert a tuple into a list
                    allowed_failures = list(allowed_failures)

                for i, param in enumerate(allowed_failures):
                    if not param.startswith("_"):
                        param = "_" + param
                        allowed_failures[i] = param
                    if param in p_check:
                        p_check.remove(param)

            p_equal = True
            for param in p_check:
                self_param = getattr(self, param)
                other_param = getattr(other, param)
                if isinstance(self_param.value, UVBase):
                    if self_param.value.__ne__(
                        other_param.value,
                        check_extra=check_extra,
                        silent=True,
                        allowed_failures=allowed_failures,
                    ):
                        if not silent:
                            print(f"parameter {param} does not match.")
                            # call again with silent passed to get the details
                            # about what is different on the UVBase object
                            self_param.value.__ne__(
                                other_param.value,
                                check_extra=check_extra,
                                silent=silent,
                            )
                        p_equal = False
                else:
                    if self_param.__ne__(other_param, silent=silent):
                        if not silent:
                            print(
                                f"parameter {param} does not match. Left is "
                                f"{self_param.value}, right is {other_param.value}."
                            )
                        p_equal = False

            if allowed_failures is not None:
                for param in allowed_failures:
                    if hasattr(self, param):
                        self_param = getattr(self, param)
                        other_param = getattr(other, param)
                        if self_param.__ne__(other_param, silent=silent) and not silent:
                            print(
                                f"parameter {param} does not match, but is not "
                                "required to for equality. Left is "
                                f"{self_param.value}, right is {other_param.value}."
                            )

            return p_equal
        else:
            if not silent:
                print("Classes do not match")
            return False

    def __ne__(
        self, other, *, check_extra=True, allowed_failures=("filename",), silent=True
    ):
        """
        Test if classes match and parameters are not equal.

        Parameters
        ----------
        other : class
            Other class instance to check
        check_extra : bool
            Option to specify whether to include all parameters, or just the
            required ones. Default is True.
        allowed_failures : iterable of str, optional
            List or tuple of parameter names that are allowed to fail while
            still passing an overall equality check. These should only include
            optional parameters. By default, the `filename` parameter will be
            ignored.
        silent : bool
            Option to turn off printing explanations of why equality fails. Useful to
            prevent __ne__ from printing lots of messages.

        Returns
        -------
        bool
            True if the two instances are equivalent.

        """
        return not self.__eq__(
            other,
            check_extra=check_extra,
            allowed_failures=allowed_failures,
            silent=silent,
        )

    def check(
        self,
        *,
        check_extra=True,
        run_check_acceptability=True,
        ignore_requirements=False,
    ):
        """
        Check that required parameters exist and have the correct shapes.

        Optionally, check that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check shapes and values on all parameters,
            otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.
        ignore_requirements : bool
            Do not error if a required parameter isn't set.
            This allows the user to run the shape/acceptability checks
            on parameters in a partially-defined UVData object.

        Returns
        -------
        bool
            True if the checks pass.

        Raises
        ------
        ValueError
            If required UVParameter values have not been set or if set UVParameters
            values do not have the expected names, shapes, types or values.

        """
        if check_extra:
            p_check = list(self.required()) + list(self.extra())
        else:
            p_check = list(self.required())

        for p in p_check:
            param = getattr(self, p)
            if p != ("_" + param.name):
                raise ValueError(
                    f"UVParameter {p} does not follow the required naming convention"
                    f"(expected be {'_' + param.name})."
                )

            # Check required parameter exists
            if param.value is None:
                if ignore_requirements:
                    continue
                if param.required is True:
                    raise ValueError(f"Required UVParameter {p} has not been set.")
            else:
                # Check parameter shape
                eshape = param.expected_shape(self)
                # default value of eshape is ()
                if eshape == "str" or (eshape == () and param.expected_type == "str"):
                    # Check that it's a string
                    if not isinstance(param.value, str):
                        raise ValueError(
                            f"UVParameter {p} expected to be string, but is not."
                        )
                else:
                    # Check the shape of the parameter value. Note that np.shape
                    # returns an empty tuple for single numbers.
                    # eshape should do the same.
                    if not np.shape(param.value) == eshape:
                        raise ValueError(
                            f"UVParameter {p} is not expected shape. Parameter "
                            f"shape is {np.shape(param.value)}, expected shape "
                            f"is {eshape}."
                        )
                    # Handle UVBase objects (e.g. Telescope) separately
                    if isinstance(param.value, UVBase):
                        param.value.check()

                    # Handle SkyCoord objects separately
                    if isinstance(param, uvp.SkyCoordParameter):
                        if not issubclass(param.value.__class__, SkyCoord):
                            raise ValueError(
                                f"UVParameter {p} should be a subclass of a "
                                f"SkyCoord object but it is {type(param.value)}."
                            )
                        else:
                            # matches expected type. Don't need to iterate through it.
                            continue  # pragma: no cover

                    # Handle recarrays separately
                    if isinstance(param.value, np.recarray):
                        rec_names = param.value.dtype.names
                        if not isinstance(param.expected_type, list) or len(
                            param.expected_type
                        ) != len(rec_names):
                            raise ValueError(
                                f"Parameter {p} is a recarray, but the expected type "
                                "is not a list with a length equal to the number of "
                                "columns in the recarray. The expected type is: "
                                f"{param.expected_type}, the recarray dtype is "
                                f"{param.value.dtype}."
                            )

                        for ind, name in enumerate(rec_names):
                            if isinstance(
                                param.value[name].item(0), param.expected_type[ind]
                            ):
                                raise ValueError(
                                    f"Parameter {p} is a recarray, the columns do not "
                                    "all have the expected types. The expected type is:"
                                    f" {param.expected_type}, the recarray dtype is "
                                    f"{param.value.dtype}."
                                )
                        continue  # pragma: no cover

                    # Quantity objects complicate things slightly
                    # Do a separate check with warnings until a quantity based
                    # parameter value is created
                    if isinstance(param.value, Quantity):
                        # check if user put expected type as a type of quantity
                        # not a more generic type of number.
                        if any(
                            issubclass(param_type, Quantity)
                            for param_type in _get_iterable(param.expected_type)
                        ):
                            # Verify the param is an instance
                            # of the specific Quantity type
                            if not isinstance(param.value, param.expected_type):
                                raise ValueError(
                                    f"UVParameter {p} is a Quantity object "
                                    "but not the appropriate type. "
                                    f"Is {type(param.value)} but "
                                    f"expected {param.expected_type}."
                                )
                            else:
                                # matches expected type
                                continue  # pragma: no cover
                        else:
                            # Expected type is not a Quantity subclass
                            # Assuming it is a data type like float, int, etc
                            # continuing with check below
                            warnings.warn(
                                f"Parameter {p} is a Quantity object, "
                                "but the expected type is a precision identifier: "
                                f"{param.expected_type}. "
                                "Testing the precision of the value, but this "
                                "check will fail in a future version."
                            )
                            check_vals = [param.value.item(0).value]
                    elif eshape == ():
                        # Single element
                        check_vals = [param.value]
                    else:
                        if isinstance(param.value, list | tuple):
                            # List & tuples needs to be handled differently than array
                            # list values may be different types, so they all
                            # need to be checked
                            check_vals = list(param.value)
                        else:
                            # numpy array
                            # the code below ensures that the check value is the type
                            # given by the dtype
                            check_vals = [param.value.dtype.type(param.value.item(0))]

                    for val in check_vals:
                        if not isinstance(val, param.expected_type):
                            raise ValueError(
                                f"UVParameter {p} is not the appropriate"
                                f" type. Is:  {type(val)}. "
                                f"Should be: {param.expected_type}."
                            )

                if run_check_acceptability:
                    accept, message = param.check_acceptability()
                    if not accept:
                        raise ValueError(
                            f"UVParameter {p} has unacceptable values. {message}"
                        )

        return True

    def copy(self):
        """
        Make and return a copy of the object.

        Returns
        -------
        UVBase
            A deep copy of this object.

        """
        return copy.deepcopy(self)

    def _select_along_param_axis(self, param_dict: dict):
        """
        Downselect values along a parameterized axis.

        This method should not be called directly by users; instead, it is called by
        various selection-related functions for selecting a subset of values along
        a given axis, whose expected shape is given (at least in part) by a named
        parameter within the object (e.g., "Nblts", "Ntimes", "Nants"). Additionally,
        this method will recalculate the value for the named parameter given the input
        indexing array.

        Parameters
        ----------
        param_dict : dict
            Dictionary which maps axes to index arrays, with keys that are matched
            against entries within UVParameter.form, and values which demark which
            indices should be selected (must be 1D). Values can also be given as None,
            in which case no selection is performed along that axis.
        """
        # This is a minor optimization -- see if the ind_arr can be expressed as slices,
        # and if so, use them where we can!
        slice_dict = {}
        for key, value in param_dict.items():
            if value is not None:
                slice_entry = slicify(value, allow_empty=True)
                if not (
                    isinstance(slice_entry, slice)
                    and slice_entry == slice(0, getattr(self, key), 1)
                ):
                    # Check that the slice isn't effectively a no-op
                    slice_dict[key] = slicify(value, allow_empty=True)

        if slice_dict:
            # If slice_dict isn't empty, proceed forward with the select
            for param in self:
                # For each attribute, if the value is None, then bail, otherwise
                # attempt to figure out along which axis ind_arr will apply.

                attr = getattr(self, param)
                if attr.name in slice_dict:
                    # This is the length argument itself -- set it accordingly. Look at
                    # param_dict since it has the lists instead of the slices
                    attr.value = len(param_dict[attr.name])
                    attr.setter(self)
                elif (
                    attr.value is not None
                    and isinstance(attr.form, tuple)
                    and any(key in slice_dict for key in attr.form)
                ):
                    # Only look at where form is a tuple, since that's the only case we
                    # can have a dynamically defined shape. Note that index doesn't work
                    # here in the case of a repeated param_name in the form.
                    attr.value = attr.get_from_form(slice_dict)
                    attr.setter(self)
