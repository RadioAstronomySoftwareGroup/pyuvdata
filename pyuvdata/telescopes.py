"""Telescope information and known telescope list."""
import numpy as np
from astropy.coordinates import Angle
import uvbase
import parameter as uvp

# center_xyz is the location of the telescope in ITRF (earth-centered frame)
telescopes = {'PAPER': {'center_xyz': None,
                        'latitude': Angle('-30d43m17.5s').radian,
                        'longitude': Angle('21d25m41.9s').radian,
                        'altitude': 1073.,
                        'citation': 'value taken from capo/cals/hsa7458_v000.py, '
                                    'comment reads KAT/SA  (GPS), altitude from elevationmap.net'},
              'HERA': {'center_xyz': None,
                       'latitude': Angle('-30d43m17.5s').radian,
                       'longitude': Angle('21d25m41.9s').radian,
                       'altitude': 1073.,
                       'citation': 'value taken from capo/cals/hsa7458_v000.py, '
                                   'comment reads KAT/SA  (GPS), altitude from elevationmap.net'},
              'MWA': {'center_xyz': None,
                      'latitude': Angle('-26d42m11.94986s').radian,
                      'longitude': Angle('116d40m14.93485s').radian,
                      'altitude': 377.827,
                      'citation': 'Tingay et al., 2013'}}


class Telescope(uvbase.UVBase):
    """
    A class for defining a telescope for use with UVData objects.

    Attributes:
        citation (str): text giving source of telescope information
        telescope_name (string, UVParameter): name of the telescope
        telescope_location (array_like, UVParameter): telescope location xyz coordinates in ITRF
            (earth-centered frame).
    """

    def __init__(self):
        """Create a new Telescope object."""
        # add the UVParameters to the class
        # use the same names as in UVData so they can be automatically set
        self.citation = None

        self._telescope_name = uvp.UVParameter('telescope_name', description='name of telescope '
                                               '(string)', form='str')
        desc = ('telescope location: xyz in ITRF (earth-centered frame). '
                'Can also be set using telescope_location_lat_lon_alt or '
                'telescope_location_lat_lon_alt_degrees properties')
        self._telescope_location = uvp.LocationParameter('telescope_location',
                                                         description=desc,
                                                         acceptable_range=(6.35e6, 6.39e6),
                                                         tols=1e-3)
        # possibly add in future versions:
        # Antenna positions (but what about reconfigurable/growing telescopes?)

        super(Telescope, self).__init__()


def known_telescopes():
    """Get list of known telescopes."""
    return telescopes.keys()


def get_telescope(telescope_name):
    """
    Get Telescope object for a telescope in known_telescopes().

    Args:
        telescope_name: string name of a telescope, must be in known_telescopes().

    Returns:
        The Telescope object associated with telescope_name.
    """
    if telescope_name.upper() in (name.upper() for name in telescopes.keys()):
        uc_telescope_list = [item.upper() for item in telescopes.keys()]
        telescope_index = uc_telescope_list.index(telescope_name.upper())
        telescope_dict = telescopes[uc_telescope_list[telescope_index]]
        obj = Telescope()
        obj.citation = telescope_dict['citation']
        obj.telescope_name = uc_telescope_list[telescope_index]
        if telescope_dict['center_xyz'] is not None:
            obj.telescope_location = center_xyz
        else:
            if (telescope_dict['latitude'] is None or telescope_dict['longitude'] is
                    None or telescope_dict['altitude'] is None):
                raise ValueError('either the center_xyz or the '
                                 'latitude, longitude and altitude of the '
                                 'telescope must be specified')
            obj.telescope_location_lat_lon_alt = (telescope_dict['latitude'],
                                                  telescope_dict['longitude'],
                                                  telescope_dict['altitude'])

            obj.check(run_check_acceptability=True)
    else:
        # no telescope matching this name
        return False

    return obj
