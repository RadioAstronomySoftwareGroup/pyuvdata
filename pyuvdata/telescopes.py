# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Telescope information and known telescope list.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.coordinates import Angle

from . import uvbase
from . import parameter as uvp

# center_xyz is the location of the telescope in ITRF (earth-centered frame)
KNOWN_TELESCOPES = {
    'PAPER': {'center_xyz': None,
              'latitude': Angle('-30d43m17.5s').radian,
              'longitude': Angle('21d25m41.9s').radian,
              'altitude': 1073.,
              'citation': ('value taken from capo/cals/hsa7458_v000.py, '
                           'comment reads KAT/SA  (GPS), altitude from elevationmap.net')},
    'HERA': {'center_xyz': None,
             'latitude': Angle('-30d43m17.5s').radian,
             'longitude': Angle('21d25m41.9s').radian,
             'altitude': 1073.,
             'diameters': 14.0,
             'citation': ('value taken from capo/cals/hsa7458_v000.py, '
                          'comment reads KAT/SA  (GPS), altitude from elevationmap.net')},
    'MWA': {'center_xyz': None,
            'latitude': Angle('-26d42m11.94986s').radian,
            'longitude': Angle('116d40m14.93485s').radian,
            'altitude': 377.827,
            'citation': 'Tingay et al., 2013'}}


class Telescope(uvbase.UVBase):
    """
    A class for defining a telescope for use with UVData objects.

    Attributes
    ----------
    citation : str
        text giving source of telescope information
    telescope_name : UVParameter of str
        name of the telescope
    telescope_location : UVParameter of array_like
        telescope location xyz coordinates in ITRF (earth-centered frame).
    antenna_diameters : UVParameter of float
        Optional, antenna diameters in meters. Used by CASA to construct a
        default beam if no beam is supplied.
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
        desc = ('Antenna diameters in meters. Used by CASA to '
                'construct a default beam if no beam is supplied.')
        self._antenna_diameters = uvp.UVParameter('antenna_diameters',
                                                  required=False, description=desc,
                                                  expected_type=np.float,
                                                  tols=1e-3)  # 1 mm
        # possibly add in future versions:
        # Antenna positions (but what about reconfigurable/growing telescopes?)

        super(Telescope, self).__init__()


def known_telescopes():
    """
    Get list of known telescopes.

    Returns
    -------
    list of str
        List of known telescope names.
    """
    return list(KNOWN_TELESCOPES.keys())


def get_telescope(telescope_name, telescope_dict_in=None):
    """
    Get Telescope object for a telescope in telescope_dict.

    Parameters
    ----------
    telescope_name : str
        Name of a telescope
    telescope_dict_in: dict
        telescope info dict. Default is None, meaning use KNOWN_TELESCOPES
        (other values are only used for testing)

    Returns
    -------
    Telescope object
        The Telescope object associated with telescope_name.
    """
    if telescope_dict_in is None:
        telescope_dict_in = KNOWN_TELESCOPES

    telescope_list = list(telescope_dict_in.keys())
    uc_telescope_list = [item.upper() for item in telescope_list]
    if telescope_name.upper() in uc_telescope_list:
        telescope_index = uc_telescope_list.index(telescope_name.upper())
        telescope_dict = telescope_dict_in[telescope_list[telescope_index]]
        obj = Telescope()
        obj.citation = telescope_dict['citation']
        obj.telescope_name = telescope_list[telescope_index]
        if telescope_dict['center_xyz'] is not None:
            obj.telescope_location = telescope_dict['center_xyz']
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
        if 'diameters' in telescope_dict.keys():
            obj.antenna_diameters = telescope_dict['diameters']
    else:
        # no telescope matching this name
        return False

    return obj
