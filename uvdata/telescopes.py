import uvdata.parameter as uvp
import numpy as np


class telescope(object):

    def __init__(self):
        # add the UVParameters to the class
        self._telescope_name = uvp.UVParameter('telescope_name', description='name of telescope '
                                               '(string)', form='str')
        desc = ('coordinate frame for antenna positions '
                '(eg "ITRF" -also google ECEF). NB: ECEF has x running '
                'through long=0 and z through the north pole')
        self._xyz_telescope_frame = uvp.UVParameter('xyz_telescope_frame', description=desc,
                                                    form='str')

        self._x_telescope = uvp.UVParameter('x_telescope',
                                            description='x coordinates of array '
                                                        'center in meters in coordinate frame',
                                            tols=1e-3)  # 1 mm
        self._y_telescope = uvp.UVParameter('y_telescope',
                                            description='y coordinates of array '
                                                        'center in meters in coordinate frame',
                                            tols=1e-3)  # 1 mm
        self._z_telescope = uvp.UVParameter('z_telescope',
                                            description='z coordinates of array '
                                                        'center in meters in coordinate frame',
                                            tols=1e-3)  # 1 mm

        self._latitude = uvp.AngleParameter('latitude',
                                            description='latitude of telescope, units radians',
                                            expected_type=np.float,
                                            tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians
        self._longitude = uvp.AngleParameter('longitude',
                                             description='longitude of telescope, units radians',
                                             expected_type=np.float,
                                             tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians
        self._altitude = uvp.UVParameter('altitude',
                                         description='altitude of telescope, units meters',
                                         expected_type=np.float,
                                         tols=1e-3)  # 1 mm

        # possibly add in future versions:
        # Antenna positions (but what about reconfigurable/growing telescopes?)


class PAPER_SA(telescope):

    def init(self):
        self.telescope_name = 'PAPER_SA'


class HERA(telescope):

    def init(self):
        self.telescope_name = 'HERA'


class MWA(telescope):

    def init(self):
        self.telescope_name = 'MWA'
