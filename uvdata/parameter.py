import numpy as np
import uvdata.utils as utils


class UVParameter(object):
    def __init__(self, name, required=True, value=None, spoof_val=None,
                 form=(), description='', expected_type=np.int, sane_vals=None,
                 tols=(1e-05, 1e-08)):
        self.name = name
        self.required = required
        # cannot set a spoof_val for required parameters
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description
        self.form = form
        if self.form == 'str':
            self.expected_type = str
        else:
            self.expected_type = expected_type
        self.sane_vals = sane_vals
        if np.size(tols) == 1:
            # Only one tolerance given, assume absolute, set relative to zero
            self.tols = (0, tols)
        else:
            self.tols = tols  # relative and absolute tolerances to be used in np.isclose

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # only check that value is identical
            if not isinstance(self.value, other.value.__class__):
                print('parameter value classes are different')
                return False
            if isinstance(self.value, np.ndarray):
                if self.value.shape != other.value.shape:
                    print('parameter value is array, shapes are different')
                    return False
                elif not np.allclose(self.value, other.value,
                                     rtol=self.tols[0], atol=self.tols[1]):
                    print('parameter value is array, values are not close')
                    return False
            else:
                str_type = False
                if isinstance(self.value, (str, unicode)):
                    str_type = True
                if isinstance(self.value, list):
                    if isinstance(self.value[0], str):
                        str_type = True

                if not str_type:
                    try:
                        if not np.isclose(np.array(self.value),
                                          np.array(other.value),
                                          rtol=self.tols[0], atol=self.tols[1]):
                            print('parameter value is not a string, values are not close')
                            return False
                    except:
                        # print(self.value, other.value)
                        print('parameter value is not a string, cannot be cast as numpy array')
                        return False
                else:
                    if self.value != other.value:
                        if not isinstance(self.value, list):
                            if self.value.replace('\n', '') != other.value.replace('\n', ''):
                                print('parameter value is a string (not a list), values are different')
                                return False
                        else:
                            print('parameter value is a list of strings, values are different')
                            return False

            return True
        else:
            print('parameter classes are different')
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_spoof(self, *args):
        self.value = self.spoof_val

    def expected_size(self, dataobj):
        # Takes the form of the parameter and returns the size
        # expected, given values in the UVBase object
        if self.form == 'str':
            return self.form
        elif isinstance(self.form, np.int):
            # Fixed size, just return the form
            return self.form
        else:
            # Given by other attributes, look up values
            esize = ()
            for p in self.form:
                if isinstance(p, np.int):
                    esize = esize + (p,)
                else:
                    val = getattr(dataobj, p)
                    if val is None:
                        raise ValueError('Missing UVBase parameter {p} needed to '
                                         'calculate expected size of parameter'.format(p=p))
                    esize = esize + (val,)
            return esize

    def sanity_check(self):
        # A quick method for checking that values are sane
        # This needs development
        if self.sane_vals is None:
            return True
        else:
            testval = np.mean(np.abs(self.value))
            if (testval >= self.sane_vals[0]) and (testval <= self.sane_vals[1]):
                return True
        return False


class AntPositionParameter(UVParameter):
    def apply_spoof(self, uvbase):
        self.value = np.zeros((len(uvbase.antenna_numbers), 3))


class ExtraKeywordParameter(UVParameter):
    def __init__(self, name, required=False, value={}, spoof_val={},
                 description=''):
        self.name = name
        self.required = required
        # cannot set a spoof_val for required parameters
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description


class AngleParameter(UVParameter):
    def degrees(self):
        if self.value is None:
            return None
        else:
            return self.value * 180. / np.pi

    def set_degrees(self, degree_val):
        if degree_val is None:
            self.value = None
        else:
            self.value = degree_val * np.pi / 180.


class LocationParameter(UVParameter):
    def lat_lon_alt(self):
        if self.value is None:
            return None
        else:
            return utils.LatLonAlt_from_XYZ(self.value)

    def set_lat_lon_alt(self, lat_lon_alt):
        if lat_lon_alt is None:
            self.value = None
        else:
            self.value = utils.XYZ_from_LatLonAlt(lat_lon_alt[0],
                                                  lat_lon_alt[1],
                                                  lat_lon_alt[2])

    def lat_lon_alt_degrees(self):
        if self.value is None:
            return None
        else:
            latitude, longitude, altitude = utils.LatLonAlt_from_XYZ(self.value)
            return latitude * 180. / np.pi, longitude * 180. / np.pi, altitude

    def set_lat_lon_alt_degrees(self, lat_lon_alt_degree):
        if lat_lon_alt_degree is None:
            self.value = None
        else:
            latitude, longitude, altitude = lat_lon_alt_degree
            self.value = utils.XYZ_from_LatLonAlt(latitude * np.pi / 180.,
                                                  longitude * np.pi / 180.,
                                                  altitude)
