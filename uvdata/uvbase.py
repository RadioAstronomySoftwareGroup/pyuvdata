import numpy as np
import uvdata.parameter as uvp


class UVBase(object):

    def __init__(self):
        # set any UVParameter attributes to be properties
        for p in self.parameter_iter():
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
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.value
        return fget

    def prop_fset(self, param_name):
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.value = value
            setattr(self, param_name, this_param)
        return fset

    def degree_prop_fget(self, param_name):
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.degrees()
        return fget

    def degree_prop_fset(self, param_name):
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_degrees(value)
            setattr(self, param_name, this_param)
        return fset

    def lat_lon_alt_prop_fget(self, param_name):
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt()
        return fget

    def lat_lon_alt_prop_fset(self, param_name):
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt(value)
            setattr(self, param_name, this_param)
        return fset

    def lat_lon_alt_degrees_prop_fget(self, param_name):
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt_degrees()
        return fget

    def lat_lon_alt_degrees_prop_fset(self, param_name):
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt_degrees(value)
            setattr(self, param_name, this_param)
        return fset

    def parameter_iter(self):
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        param_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, uvp.UVParameter):
                param_list.append(a)
        for a in param_list:
            yield a

    def required_parameter_iter(self):
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

    def extra_parameter_iter(self):
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
        if isinstance(other, self.__class__):
            # only check that required parameters are identical
            for p in self.required_parameter_iter():
                self_param = getattr(self, p)
                other_param = getattr(other, p)
                if self_param != other_param:
                    # print('parameter {pname} does not match. Left is {lval} '
                    #       'and right is {rval}'.
                    #       format(pname=p, lval=str(self_param.value),
                    #              rval=str(other_param.value)))
                    return False
            return True
        else:
            print('Classes do not match')
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def check(self, run_sanity_check=True):
        # loop through all required parameters, make sure that they are filled
        for p in self.required_parameter_iter():
            param = getattr(self, p)
            # Check required parameter exists
            if param.value is None:
                raise ValueError('Required UVParameter ' + p +
                                 ' has not been set.')

            # Check required parameter size
            esize = param.expected_size(self)
            if esize is None:
                raise ValueError('Required UVParameter ' + p +
                                 ' expected size is not defined.')
            elif esize == 'str':
                # Check that it's a string
                if not isinstance(param.value, str):
                    raise ValueError('UVParameter ' + p + 'expected to be '
                                     'string, but is not')
            else:
                # Check the size of the parameter value. Note that np.shape
                # returns an empty tuple for single numbers. esize should do the same.
                if not np.shape(param.value) == esize:
                    raise ValueError('UVParameter {param} is not expected size. '
                                     'Parameter size is {psize}, expected size is '
                                     '{esize}.'.format(param=p, psize=np.shape(param.value),
                                                       esize=esize))
                if esize == ():
                    # Single element
                    if not isinstance(param.value, param.expected_type):
                        raise ValueError('UVParameter ' + p + ' is not the appropriate'
                                         ' type. Is: ' + str(type(param.value)) +
                                         '. Should be: ' + str(param.expected_type))
                else:
                    if isinstance(param.value, list):
                        # List needs to be handled differently than array (I think)
                        if not isinstance(param.value[0], param.expected_type):
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

            if run_sanity_check:
                if not param.sanity_check():
                    raise ValueError('UVParameter ' + p + ' has insane values.')

        return True
