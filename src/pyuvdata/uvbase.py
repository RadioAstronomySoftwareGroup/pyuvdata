# -*- mode: python; coding: utf-8 -*-
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

from . import __version__
from . import parameter as uvp
from .utils.tools import _get_iterable

__all__ = ["UVBase"]

# the old names of attributes as keys, values are the names on the telescope object
old_telescope_metadata_attrs = {
    "telescope_name": "name",
    "telescope_location": None,
    "telescope_location_lat_lon_alt": None,
    "telescope_location_lat_lon_alt_degrees": None,
    "instrument": "instrument",
    "Nants_telescope": "Nants",
    "antenna_names": "antenna_names",
    "antenna_numbers": "antenna_numbers",
    "antenna_positions": "antenna_positions",
    "x_orientation": "x_orientation",
    "antenna_diameters": "antenna_diameters",
}


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


class UVBase(object):
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

    def __getattr__(self, __name):
        """Handle old names for telescope metadata."""
        if __name in old_telescope_metadata_attrs:
            if hasattr(self, "telescope"):
                if old_telescope_metadata_attrs[__name] is not None:
                    tel_param = old_telescope_metadata_attrs[__name]
                else:
                    tel_param = "location"
                warnings.warn(
                    f"The UVData.{__name} attribute now just points to the "
                    f"{tel_param} attribute on the telescope object (at "
                    "UVData.telescope). Accessing it this way is deprecated, "
                    "please access it via the telescope object. This will "
                    "become an error in version 3.2.",
                    DeprecationWarning,
                )

                tel_name = old_telescope_metadata_attrs[__name]
                if tel_name is not None:
                    # if it's a simple remapping, just return the value
                    ret_val = getattr(self.telescope, tel_name)
                else:
                    # handle location related stuff
                    if __name == "telescope_location":
                        ret_val = self.telescope._location.xyz()
                    elif __name == "telescope_location_lat_lon_alt":
                        ret_val = self.telescope._location.lat_lon_alt()
                    elif __name == "telescope_location_lat_lon_alt_degrees":
                        ret_val = self.telescope._location.lat_lon_alt_degrees()
                return ret_val
        elif __name == "future_array_shapes":
            warnings.warn(
                f"The {__name} attribute is now deprecated, as all UVBase "
                "objects use future array shapes. This will become an error in "
                "version 3.2.",
                DeprecationWarning,
            )
            # Always true as of v3.0
            return True

        return super().__getattribute__(__name)

    def __setattr__(self, __name, __value):
        """Handle old names for telescope metadata."""
        if __name in old_telescope_metadata_attrs:
            if hasattr(self, "telescope"):
                if old_telescope_metadata_attrs[__name] is not None:
                    tel_param = old_telescope_metadata_attrs[__name]
                else:
                    tel_param = "location"
                warnings.warn(
                    f"The UVData.{__name} attribute now just points to the "
                    f"{tel_param} attribute on the telescope object (at "
                    "UVData.telescope). Accessing it this way is deprecated, "
                    "please access it via the telescope object. This will "
                    "become an error in version 3.2.",
                    DeprecationWarning,
                )

                tel_name = old_telescope_metadata_attrs[__name]
                if tel_name is not None:
                    # if it's a simple remapping, just set the value
                    setattr(self.telescope, tel_name, __value)
                else:
                    # handle location related stuff
                    if __name == "telescope_location":
                        self.telescope._location.set_xyz(__value)
                    elif __name == "telescope_location_lat_lon_alt":
                        self.telescope._location.set_lat_lon_alt(__value)
                    elif __name == "telescope_location_lat_lon_alt_degrees":
                        self.telescope._location.set_lat_lon_alt_degrees(__value)
                return

        return super().__setattr__(__name, __value)

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
                self.metadata_only
                other.metadata_only

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
                        other_param.value, check_extra=check_extra, silent=True
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
                        if self_param.__ne__(other_param, silent=silent):
                            if not silent:
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
                            "UVParameter {param} is not expected shape. "
                            "Parameter shape is {pshape}, expected shape is "
                            "{eshape}.".format(
                                param=p, pshape=np.shape(param.value), eshape=eshape
                            )
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
                        if isinstance(param.value, (list, tuple)):
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

    def _set_future_array_shapes(self, use_future_array_shapes=None):
        """
        Set future_array_shapes to True and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by file-reading methods and `use_future_array_shapes` to indicate the
        `future_array_shapes` is True and define expected parameter shapes.
        This function has been deprecated, and will result in an error in version 3.2.
        """
        if use_future_array_shapes is None:
            # This basically wraps no-ops when no argument is passed.
            return

        if not use_future_array_shapes:
            raise ValueError(
                'The future is now! So-called "current" array shapes no longer '
                'supported, must use "future" array shapes (spw-axis dropped).'
            )
        warnings.warn(
            (
                "Future array shapes are now always used, setting/calling "
                "use_future_array_shapes will result in an error in version 3.2."
            ),
            DeprecationWarning,
        )

    def use_future_array_shapes(self):
        """
        Change the array shapes of this object to match the planned future shapes.

        This function has been deprecated, and will result in an error in version 3.2.
        """
        self._set_future_array_shapes(True)
