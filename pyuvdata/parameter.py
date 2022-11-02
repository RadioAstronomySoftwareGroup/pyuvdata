# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Define UVParameters: data and metadata objects for interferometric data sets.

UVParameters are objects to hold specific data and metadata associated with
interferometric data sets. They are used as attributes for classes based on
UVBase. This module also includes specialized sublasses for particular types
of metadata.

"""
import builtins

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord

from . import utils

__all__ = ["UVParameter", "AngleParameter", "LocationParameter"]


def _get_generic_type(expected_type, strict_type_check=False):
    """Return tuple of more generic types.

    Allows for more flexible type checking in the case when a Parameter's value
    changes precison or to/from a numpy dtype but still is the desired generic type.
    If a generic type cannot be found, the expected_type is returned

    Parameters
    ----------
    expected_type : Type or string
        The expected type of a Parameter object or a string of the name of a type.
    strict_type_check : bool
        If True the input expected_type is return exactly
    if strict_type_check:
        return expected_type exactly

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
    if strict_type_check:
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


class UVParameter(object):
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

    def __init__(
        self,
        name,
        required=True,
        value=None,
        spoof_val=None,
        form=(),
        description="",
        expected_type=int,
        acceptable_vals=None,
        acceptable_range=None,
        tols=(1e-05, 1e-08),
        strict_type_check=False,
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
                expected_type,
                strict_type_check=strict_type_check,
            )
            self.strict_type = strict_type_check
        self.acceptable_vals = acceptable_vals
        self.acceptable_range = acceptable_range
        if np.size(tols) == 1:
            # Only one tolerance given, assume absolute, set relative to zero
            self.tols = (0, tols)
        else:
            # relative and absolute tolerances to be used in np.isclose
            self.tols = tols

    def __eq__(self, other):
        """
        Test if classes match and values are within tolerances.

        Parameters
        ----------
        other : UVParameter or subclass
            The other UVParameter to compare with this one.

        """
        if isinstance(other, self.__class__):
            if self.value is None:
                if other.value is not None:
                    print(f"{self.name} is None on left, but not right")
                    return False
                else:
                    return True
            if other.value is None:
                if self.value is not None:
                    print(f"{self.name} is None on right, but not left")
                    return False

            if isinstance(self.value, np.ndarray) and not isinstance(
                self.value.item(0), (str, np.str_)
            ):
                if not isinstance(other.value, np.ndarray):
                    print(f"{self.name} parameter value is array, but other is not")
                    return False
                if self.value.shape != other.value.shape:
                    print(f"{self.name} parameter value is array, shapes are different")
                    return False

                # check to see if strict types are used
                if self.strict_type:
                    # types must match
                    if other.strict_type:
                        # both strict, expected_type must match
                        if self.expected_type != other.expected_type:
                            print(
                                f"{self.name} parameter has incompatible types. "
                                f"Left is {self.expected_type}, right is "
                                f"{other.expected_type}"
                            )
                            return False
                    elif not isinstance(self.value.item(0), other.expected_type):
                        print(
                            f"{self.name} parameter has incompatible dtypes. Left "
                            f"requires {self.expected_type}, right is "
                            f"{other.value.dtype}"
                        )
                        return False
                elif other.strict_type:
                    # types must match in the other direction
                    if not isinstance(other.value.item(0), self.expected_type):
                        print(
                            f"{self.name} parameter has incompatible dtypes. Left is "
                            f"{self.value.dtype}, right requires {other.expected_type}"
                        )
                        return False

                elif isinstance(self.value, units.Quantity):
                    if not self.value.unit.is_equivalent(other.value.unit):
                        print(
                            f"{self.name} parameter value is an astropy Quantity, "
                            "units are not equivalent"
                        )
                        return False
                    if not isinstance(self.tols[1], units.Quantity):
                        atol_use = self.tols[1] * self.value.unit
                    else:
                        atol_use = self.tols[1]
                    if not units.quantity.allclose(
                        self.value,
                        other.value,
                        rtol=self.tols[0],
                        atol=atol_use,
                        equal_nan=True,
                    ):
                        print(
                            f"{self.name} parameter value is an astropy Quantity, "
                            "values are not close"
                        )
                        return False
                elif not np.allclose(
                    self.value,
                    other.value,
                    rtol=self.tols[0],
                    atol=self.tols[1],
                    equal_nan=True,
                ):
                    print(f"{self.name} parameter value is array, values are not close")
                    return False
            else:
                # check to see if strict types are used
                if self.strict_type:
                    # types must match
                    if not isinstance(self.value, other.expected_type):
                        print(
                            f"{self.name} parameter has incompatible types. Left "
                            f"requires {type(self.value)}, right is "
                            f"{other.expected_type}"
                        )
                        return False
                if other.strict_type:
                    # types must match in the other direction
                    if not isinstance(other.value, self.expected_type):
                        print(
                            f"{self.name} parameter has incompatible types. Left is "
                            f"{self.expected_type}, right requires {type(other.value)}"
                        )
                        return False

                str_type = False
                if isinstance(self.value, str):
                    str_type = True
                if isinstance(self.value, (list, np.ndarray, tuple)):
                    if isinstance(self.value[0], str):
                        str_type = True

                if not str_type:
                    if isinstance(other.value, np.ndarray):
                        print(
                            f"{self.name} parameter value is not an array, "
                            "but other is not"
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
                            print(
                                f"{self.name} parameter value can be cast to an array"
                                " and tested with np.allclose. The values are "
                                "not close"
                            )
                            return False
                    except (TypeError):
                        if isinstance(self.value, dict):
                            try:
                                # Try a naive comparison first
                                # this will fail if keys are the same
                                # but cases differ.
                                # so only look for exact equality
                                # then default to the long test below.
                                if self.value == other.value:
                                    return True
                            except ValueError:
                                pass
                                # this dict probably contains arrays
                                # we will need to check each item individually

                            # check to see if they are equal other than
                            # upper/lower case keys
                            self_lower = {
                                (k.lower() if isinstance(k, str) else k): v
                                for k, v in self.value.items()
                            }
                            other_lower = {
                                (k.lower() if isinstance(k, str) else k): v
                                for k, v in other.value.items()
                            }
                            message_str = f"{self.name} parameter is a dict"
                            self_key_set = set(self_lower.keys())
                            other_key_set = set(other_lower.keys())
                            if self_key_set != other_key_set:
                                message_str += ", keys are not the same.\n"
                                self_only = self_key_set - other_key_set
                                other_only = other_key_set - self_key_set
                                if len(self_only) > 0:
                                    message_str += (
                                        f" keys only on left are: {self_only}\n"
                                    )
                                if len(other_only) > 0:
                                    message_str += (
                                        f" keys only on right are: {self_only}\n"
                                    )
                                print(message_str)
                                return False
                            else:
                                # need to check if values are close,
                                # not just equal
                                values_close = True
                                for key in self_lower.keys():
                                    try:
                                        if not np.allclose(
                                            self_lower[key], other_lower[key]
                                        ):
                                            message_str += f", key {key} is not equal"
                                            values_close = False
                                    except (TypeError):
                                        # this isn't a type that can be
                                        # handled by np.isclose,
                                        # test for equality
                                        if self_lower[key] != other_lower[key]:
                                            message_str += f", key {key} is not equal"
                                            values_close = False
                                if values_close is False:
                                    print(message_str)
                                    return False
                                else:
                                    return True
                        else:
                            if self.value != other.value:
                                print(
                                    f"{self.name} parameter value is not a string "
                                    "or a dict and cannot be cast as a numpy "
                                    "array. The values are not equal."
                                )

                                return False

                else:
                    if isinstance(self.value, (list, np.ndarray, tuple)):
                        if [s.strip() for s in self.value] != [
                            s.strip() for s in other.value
                        ]:
                            print(
                                f"{self.name} parameter value is a list of strings, "
                                "values are different"
                            )
                            return False
                    else:
                        if self.value.strip() != other.value.strip():
                            if self.value.replace("\n", "").replace(
                                " ", ""
                            ) != other.value.replace("\n", "").replace(" ", ""):
                                print(
                                    f"{self.name} parameter value is a string, "
                                    "values are different"
                                )
                                return False

            return True
        else:
            print(f"{self.name} parameter classes are different")
            return False

    def __ne__(self, other):
        """
        Test if classes do not match or values are not within tolerances.

        Parameters
        ----------
        other : UVParameter or subclass
            The other UVParameter to compare with this one.

        """
        return not self.__eq__(other)

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
        elif isinstance(self.form, (int, np.integer)):
            # Fixed shape, just return the form
            return (self.form,)
        else:
            # Given by other attributes, look up values
            eshape = ()
            for p in self.form:
                if isinstance(p, (int, np.integer)):
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
                    else:
                        # this is a list or array of strings, make them all lower case
                        value_set = {x.lower() for x in self.value}
                    acceptable_vals = [x.lower() for x in self.acceptable_vals]
                else:
                    if isinstance(self.value, (list, np.ndarray)):
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
    spoof_val
        A fake value that can be assigned to a non-required UVParameter if the
        metadata is required for a particular file-type.
        This is not an attribute of required UVParameters.
    description : str
        Description of the data or metadata in the object.
    frame : str, optional
        Coordinate frame. Valid options are "itrs" (default) or "mcmf".
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
    form : int
       Always set to 3.
    description : str
        Description of the data or metadata in the object.
    frame : str, optional
        Coordinate frame. Valid options are "itrs" (default) or "mcmf".
    expected_type
        Always set to float.
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

    def __init__(
        self,
        name,
        required=True,
        value=None,
        spoof_val=None,
        description="",
        frame="itrs",
        acceptable_range=-1,
        tols=1e-3,
    ):
        if acceptable_range == -1:
            if frame == "itrs":
                acceptable_range = (6.35e6, 6.39e6)
            elif frame == "mcmf":
                acceptable_range = (1717100.0, 1757100.0)
            else:
                acceptable_range = None

        super(LocationParameter, self).__init__(
            name,
            required=required,
            value=value,
            spoof_val=spoof_val,
            form=3,
            description=description,
            expected_type=float,
            acceptable_range=acceptable_range,
            tols=tols,
        )
        self.frame = frame

    def lat_lon_alt(self):
        """Get value in (latitude, longitude, altitude) tuple in radians."""
        if self.value is None:
            return None
        else:
            # check defaults to False b/c exposed check kwarg exists in UVData
            return utils.LatLonAlt_from_XYZ(
                self.value, check_acceptability=False, frame=self.frame
            )

    def set_lat_lon_alt(self, lat_lon_alt):
        """
        Set value from (latitude, longitude, altitude) tuple in radians.

        Parameters
        ----------
        lat_lon_alt : 3-tuple of float
            Tuple with the latitude (radians), longitude (radians)
            and altitude (meters) to use to set the value attribute.
        """
        if lat_lon_alt is None:
            self.value = None
        else:
            self.value = utils.XYZ_from_LatLonAlt(
                lat_lon_alt[0],
                lat_lon_alt[1],
                lat_lon_alt[2],
                frame=self.frame,
            )

    def lat_lon_alt_degrees(self):
        """Get value in (latitude, longitude, altitude) tuple in degrees."""
        if self.value is None:
            return None
        else:
            latitude, longitude, altitude = self.lat_lon_alt()
            return latitude * 180.0 / np.pi, longitude * 180.0 / np.pi, altitude

    def set_lat_lon_alt_degrees(self, lat_lon_alt_degree):
        """
        Set value from (latitude, longitude, altitude) tuple in degrees.

        Parameters
        ----------
        lat_lon_alt : 3-tuple of float
            Tuple with the latitude (degrees), longitude (degrees)
            and altitude (meters) to use to set the value attribute.

        """
        if lat_lon_alt_degree is None:
            self.value = None
        else:
            latitude, longitude, altitude = lat_lon_alt_degree
            self.value = utils.XYZ_from_LatLonAlt(
                latitude * np.pi / 180.0,
                longitude * np.pi / 180.0,
                altitude,
                frame=self.frame,
            )

    def check_acceptability(self):
        """Check that vector magnitudes are in range."""
        if self.acceptable_range is None:
            return True, "No acceptability check"
        else:
            # acceptable_range is a tuple giving a range of allowed vector magnitudes
            testval = np.sqrt(np.sum(np.abs(self.value) ** 2))
            if (testval >= self.acceptable_range[0]) and (
                testval <= self.acceptable_range[1]
            ):
                return True, "Value is acceptable"
            else:
                message = (
                    f"Value {testval}, is not in allowed range: {self.acceptable_range}"
                )
                return False, message


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
    expected_type
        The type that the data or metadata should be. Default is int or str if
        form is 'str'.
    acceptable_vals : list, optional
        List giving allowed values for elements of value.
    acceptable_range: 2-tuple, optional
        Tuple giving a range of allowed magnitudes for elements of value.
    radian_tol : float
        Tolerance of the sky separation in radians.
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
        Always set to SkyCoord.
    acceptable_vals : list, optional
        List giving allowed values for elements of value.
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
        required=True,
        value=None,
        spoof_val=None,
        form=(),
        description="",
        acceptable_range=None,
        # standard angle tolerance: 1 mas in radians.
        radian_tol=1 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0),
    ):
        super(SkyCoordParameter, self).__init__(
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

    def __eq__(self, other):
        if not issubclass(self.value.__class__, SkyCoord) or not issubclass(
            other.value.__class__, SkyCoord
        ):
            return super(SkyCoordParameter, self).__eq__(other)

        if self.value.shape != other.value.shape:
            print(f"{self.name} parameter shapes are different")
            return False

        this_frame = self.value.frame.name
        other_frame = other.value.frame.name
        if this_frame != other_frame:
            print(
                f"{self.name} parameter has different frames, {this_frame} vs "
                f"{other_frame}."
            )
            return False

        this_rep_type = self.value.representation_type
        other_rep_type = other.value.representation_type
        if this_rep_type != other_rep_type:
            print(
                f"{self.name} parameter has different representation_types, "
                f"{this_rep_type} vs {other_rep_type}."
            )
            return False

        # finally calculate on sky separations
        sky_separation = self.value.separation(other.value).rad
        if np.any(sky_separation > self.tols[1]):
            print(f"{self.name} parameter is not close. ")
            return False

        return True
