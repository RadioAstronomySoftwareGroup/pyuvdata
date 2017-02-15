"""
Base class for objects with UVParameter attributes.

Subclassed by UVData and Telescope.
"""
import numpy as np
import parameter as uvp
import warnings


def _warning(msg, *a):
    """Improve the printing of user warnings."""
    return str(msg) + '\n'


class UVBase(object):
    """
    Base class for objects with UVParameter attributes.

    This class is intended to be subclassed and its init method should be
    called in the subclass init after all associated UVParameter attributes are
    defined. The init method of this base class creates properties
    (named using UVParameter.name) from all the UVParameter attributes on the subclass.
    AngleParameter and LocationParameter attributes also have extra convenience
    properties defined:\n
        AngleParameter:\n
            UVParameter.name+'_degrees'\n
        LocationParameter:\n
            UVParameter.name+'_lat_lon_alt'\n
            UVParameter.name+'_lat_lon_alt_degrees'
    """

    def __init__(self):
        """Create properties from UVParameter attributes."""

        warnings.formatwarning = _warning

        # set any UVParameter attributes to be properties
        for p in self:
            this_param = getattr(self, p)
            attr_name = this_param.name
            setattr(self.__class__, attr_name, property(self.prop_fget(p),
                                                        self.prop_fset(p)))
            if isinstance(this_param, uvp.AngleParameter):
                setattr(self.__class__, attr_name + '_degrees',
                        property(self.degree_prop_fget(p), self.degree_prop_fset(p)))
            elif isinstance(this_param, uvp.LocationParameter):
                setattr(self.__class__, attr_name + '_lat_lon_alt',
                        property(self.lat_lon_alt_prop_fget(p),
                                 self.lat_lon_alt_prop_fset(p)))
                setattr(self.__class__, attr_name + '_lat_lon_alt_degrees',
                        property(self.lat_lon_alt_degrees_prop_fget(p),
                                 self.lat_lon_alt_degrees_prop_fset(p)))

    def prop_fget(self, param_name):
        """Getter method for UVParameter properties."""
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.value
        return fget

    def prop_fset(self, param_name):
        """Setter method for UVParameter properties."""
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.value = value
            setattr(self, param_name, this_param)
        return fset

    def degree_prop_fget(self, param_name):
        """Degree getter method for AngleParameter properties."""
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.degrees()
        return fget

    def degree_prop_fset(self, param_name):
        """Degree setter method for AngleParameter properties."""
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_degrees(value)
            setattr(self, param_name, this_param)
        return fset

    def lat_lon_alt_prop_fget(self, param_name):
        """Lat/lon/alt getter method for LocationParameter properties."""
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt()
        return fget

    def lat_lon_alt_prop_fset(self, param_name):
        """Lat/lon/alt setter method for LocationParameter properties."""
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt(value)
            setattr(self, param_name, this_param)
        return fset

    def lat_lon_alt_degrees_prop_fget(self, param_name):
        """Lat/lon/alt degree getter method for LocationParameter properties."""
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt_degrees()
        return fget

    def lat_lon_alt_degrees_prop_fset(self, param_name):
        """Lat/lon/alt degree setter method for LocationParameter properties."""
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt_degrees(value)
            setattr(self, param_name, this_param)
        return fset

    def __iter__(self):
        """Iterator for all UVParameter attributes."""
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        param_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, uvp.UVParameter):
                param_list.append(a)
        for a in param_list:
            yield a

    def required(self):
        """Iterator for all required UVParameter attributes."""
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        required_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, uvp.UVParameter):
                if attr.required:
                    required_list.append(a)
        for a in required_list:
            yield a

    def extra(self):
        """Iterator for all non-required UVParameter attributes."""
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        extra_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, uvp.UVParameter):
                if not attr.required:
                    extra_list.append(a)
        for a in extra_list:
            yield a

    def __eq__(self, other):
        """Equal if classes match and required parameters are equal."""
        if isinstance(other, self.__class__):
            # only check that required parameters are identical
            self_required = []
            other_required = []
            for p in self.required():
                self_required.append(p)
            for p in other.required():
                other_required.append(p)
            if set(self_required) != set(other_required):
                print('Sets of required parameters do not match. Left is {lset},'
                      ' right is {rset}'.format(lset=self_required,
                                                rset=other_required))
                return False

            p_equal = True
            for p in self.required():
                self_param = getattr(self, p)
                other_param = getattr(other, p)
                if self_param != other_param:
                    p_equal = False
            return p_equal
        else:
            print('Classes do not match')
            return False

    def __ne__(self, other):
        """Not equal."""
        return not self.__eq__(other)

    def check(self, run_check_acceptability=True):
        """
        Check that all required parameters are set reasonably.

        Check that required parameters exist and have appropriate shapes.
        Optionally check if the values are acceptable.

        Args:
            run_check_acceptability: Option to check if values in required parameters
                are acceptable. Default is True.
        """
        for p in self.required():
            param = getattr(self, p)
            # Check required parameter exists
            if param.value is None:
                raise ValueError('Required UVParameter ' + p +
                                 ' has not been set.')

            # Check required parameter shape
            eshape = param.expected_shape(self)
            if eshape is None:
                raise ValueError('Required UVParameter ' + p +
                                 ' expected shape is not defined.')
            elif eshape == 'str':
                # Check that it's a string
                if not isinstance(param.value, str):
                    raise ValueError('UVParameter ' + p + 'expected to be '
                                     'string, but is not')
            else:
                # Check the shape of the parameter value. Note that np.shape
                # returns an empty tuple for single numbers. eshape should do the same.
                if not np.shape(param.value) == eshape:
                    raise ValueError('UVParameter {param} is not expected shape. '
                                     'Parameter shape is {pshape}, expected shape is '
                                     '{eshape}.'.format(param=p, pshape=np.shape(param.value),
                                                        eshape=eshape))
                if eshape == ():
                    # Single element
                    if not isinstance(param.value, param.expected_type):
                        raise ValueError('UVParameter ' + p + ' is not the appropriate'
                                         ' type. Is: ' + str(type(param.value)) +
                                         '. Should be: ' + str(param.expected_type))
                else:
                    if isinstance(param.value, list):
                        # List needs to be handled differently than array
                        # list values may be different types, so they all need to be checked
                        for item in param.value:
                            if not isinstance(item, param.expected_type):
                                raise ValueError('UVParameter ' + p + ' is not the'
                                                 ' appropriate type. Is: ' +
                                                 str(type(param.value[0])) + '. Should'
                                                 ' be: ' + str(param.expected_type))
                    else:
                        # Array
                        if not isinstance(param.value.item(0), param.expected_type):
                            raise ValueError('UVParameter ' + p + ' is not the appropriate'
                                             ' type. Is: ' + str(param.value.dtype) +
                                             '. Should be: ' + str(param.expected_type))

            if run_check_acceptability:
                accept, message = param.check_acceptability()
                if not accept:
                    raise ValueError('UVParameter ' + p + ' has unacceptable values. ' +
                                     message)

        return True
