import numpy as np
from astropy.coordinates import Angle
from uvdata.uvbase import UVBase
import uvdata.parameter as uvp
import uvdata.utils as utils


telescopes = {'PAPER': {'frame': None, 'center_xyz': None,
                        'latitude': Angle('-30d43m17.5s').radian,
                        'longitude': Angle('21d25m41.9s').radian,
                        'altitude': 1073.,
                        'citation': 'value taken from capo/cals/hsa7458_v000.py, '
                                    'comment reads KAT/SA  (GPS), altitude from elevationmap.net'},
              'HERA': {'frame': None, 'center_xyz': None,
                       'latitude': Angle('-30d43m17.5s').radian,
                       'longitude': Angle('21d25m41.9s').radian,
                       'altitude': 1073.,
                       'citation': 'value taken from capo/cals/hsa7458_v000.py, '
                                   'comment reads KAT/SA  (GPS), altitude from elevationmap.net'},
              'MWA': {'frame': None, 'center_xyz': None,
                      'latitude': Angle('-26d42m11.94986s').radian,
                      'longitude': Angle('116d40m14.93485s').radian,
                      'altitude': 377.827,
                      'citation': 'Tingay et al., 2013'}}


class Telescope(UVBase):

    def __init__(self):
        # add the UVParameters to the class
        # use the same names as in UVData so they can be automatically set
        self.citation = None

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

        super(UVBase, self).__init__()


def known_telescopes():
    return telescopes.keys()


def get_telescope(telescope_name):

    if telescope_name.upper() in (name.upper() for name in telescopes.keys()):
        uc_telescope_list = [item.upper() for item in telescopes.keys()]
        telescope_index = uc_telescope_list.index(telescope_name.upper())
        telescope_dict = telescopes[uc_telescope_list[telescope_index]]
        obj = Telescope()
        obj.citation = telescope_dict['citation']
        obj.telescope_name = uc_telescope_list[telescope_index]
        if (telescope_dict['center_xyz'] is not None and telescope_dict['frame'] is not None):
            obj.xyz_telescope_frame = telescope_dict['frame']
            obj.x_telescope = telescope_dict['center_xyz'][0]
            obj.y_telescope = telescope_dict['center_xyz'][1]
            obj.z_telescope = telescope_dict['center_xyz'][2]

            if (telescope_dict['latitude'] is not None and telescope_dict['longitude'] is not
                    None and telescope_dict['altitude'] is not None):
                obj.latitude = telescope_dict['latitude']
                obj.longitude = telescope_dict['longitude']
                obj.altitude = telescope_dict['altitude']
            else:
                if telescope_dict['frame'] == 'ITRF':
                    latitude, longitude, altitude = utils.LatLonAlt_from_XYZ(telescope_dict['center_xyz'])
                    obj.latitude = latitude
                    obj.longitude = longitude
                    obj.altitude = altitude
                else:
                    raise ValueError('latitude, longitude or altitude not'
                                     'specified and frame is not "ITRF"')
        else:
            if (telescope_dict['latitude'] is None or telescope_dict['longitude'] is
                    None or telescope_dict['altitude'] is None):
                raise ValueError('either the center_xyz and frame or the '
                                 'latitude, longitude and altitude of the '
                                 'telescope must be specified')
            obj.latitude = telescope_dict['latitude']
            obj.longitude = telescope_dict['longitude']
            obj.altitude = telescope_dict['altitude']

            xyz = utils.XYZ_from_LatLonAlt(telescope_dict['latitude'], telescope_dict['longitude'],
                                           telescope_dict['altitude'])
            obj.xyz_telescope_frame = 'ITRF'
            obj.x_telescope = xyz[0]
            obj.y_telescope = xyz[1]
            obj.z_telescope = xyz[2]
    else:
        # no telescope matching this name
        return False

    return obj
