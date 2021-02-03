# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Telescope information and known telescope list."""
import os
import numpy as np
from astropy.coordinates import Angle, EarthLocation

from . import uvbase
from . import parameter as uvp
from pyuvdata.data import DATA_PATH

__all__ = ["Telescope", "known_telescopes", "get_telescope"]

# We use astropy sites for telescope locations. The dict below is for
# telescopes not in astropy sites, or to include extra information for a telescope.

# The center_xyz is the location of the telescope in ITRF (earth-centered frame)

# Antenna positions can be specified via a csv file with the following columns:
# "name" -- antenna name, "number" -- antenna number, "x", "y", "z" -- ECEF coordinates
# relative to the telescope location.
KNOWN_TELESCOPES = {
    "PAPER": {
        "center_xyz": None,
        "latitude": Angle("-30d43m17.5s").radian,
        "longitude": Angle("21d25m41.9s").radian,
        "altitude": 1073.0,
        "citation": (
            "value taken from capo/cals/hsa7458_v000.py, "
            "comment reads KAT/SA  (GPS), altitude from elevationmap.net"
        ),
    },
    "HERA": {
        "center_xyz": None,
        "latitude": Angle("-30.72152612068925d").radian,
        "longitude": Angle("21.42830382686301d").radian,
        "altitude": 1051.69,
        "diameters": 14.0,
        "antenna_positions_file": "hera_ant_pos.csv",
        "citation": (
            "value taken from hera_mc geo.py script "
            "(using hera_cm_db_updates under the hood.)"
        ),
    },
    "SMA": {
        "center_xyz": None,
        "latitude": Angle("19d49m27.13895s").radian,
        "longitude": Angle("-155d28m39.08279s").radian,
        "altitude": 4083.948144,
        "citation": "Ho, P. T. P., Moran, J. M., & Lo, K. Y. 2004, ApJL, 616, L1",
    },
}


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

        self._telescope_name = uvp.UVParameter(
            "telescope_name", description="name of telescope " "(string)", form="str"
        )
        desc = (
            "telescope location: xyz in ITRF (earth-centered frame). "
            "Can also be set using telescope_location_lat_lon_alt or "
            "telescope_location_lat_lon_alt_degrees properties"
        )
        self._telescope_location = uvp.LocationParameter(
            "telescope_location",
            description=desc,
            acceptable_range=(6.35e6, 6.39e6),
            tols=1e-3,
        )
        desc = (
            "Antenna diameters in meters. Used by CASA to "
            "construct a default beam if no beam is supplied."
        )
        self._antenna_diameters = uvp.UVParameter(
            "antenna_diameters",
            required=False,
            description=desc,
            expected_type=float,
            tols=1e-3,  # 1 mm
        )

        desc = "Number of antennas in the array."
        self._Nants_telescope = uvp.UVParameter(
            "Nants_telescope", required=False, description=desc, expected_type=int
        )

        desc = (
            "List of antenna names, shape (Nants_telescope), "
            "with numbers given by antenna_numbers."
        )
        self._antenna_names = uvp.UVParameter(
            "antenna_names",
            required=False,
            description=desc,
            form=("Nants_telescope",),
            expected_type=str,
        )

        desc = (
            "List of integer antenna numbers corresponding to antenna_names, "
            "shape (Nants_telescope)."
        )
        self._antenna_numbers = uvp.UVParameter(
            "antenna_numbers",
            required=False,
            description=desc,
            form=("Nants_telescope",),
            expected_type=int,
        )

        desc = (
            "Array giving coordinates of antennas relative to "
            "telescope_location (ITRF frame), shape (Nants_telescope, 3), "
            "units meters. See the tutorial page in the documentation "
            "for an example of how to convert this to topocentric frame."
        )
        self._antenna_positions = uvp.UVParameter(
            "antenna_positions",
            required=False,
            description=desc,
            form=("Nants_telescope", 3),
            expected_type=float,
            tols=1e-3,  # 1 mm
        )

        super(Telescope, self).__init__()


def known_telescopes():
    """
    Get list of known telescopes.

    Returns
    -------
    list of str
        List of known telescope names.
    """
    astropy_sites = EarthLocation.get_site_names()
    while "" in astropy_sites:
        astropy_sites.remove("")
    known_telescopes = list(set(astropy_sites + list(KNOWN_TELESCOPES.keys())))
    return known_telescopes


def _parse_antpos_file(antenna_positions_file):
    """
    Interpret the antenna positions file.

    Parameters
    ----------
    antenna_positions_file : str
        Name of the antenna_positions_file, which is assumed to be in DATA_PATH.
        Should contain antenna names, numbers and ECEF positions relative to the
        telescope location.

    Returns
    -------
    antenna_names : array of str
        Antenna names.
    antenna_names : array of int
        Antenna numbers.
    antenna_positions : array of float
        Antenna positions in ECEF relative to the telescope location.

    """
    columns = ["name", "number", "x", "y", "z"]
    formats = ["U10", "i8", np.longdouble, np.longdouble, np.longdouble]

    dt = np.format_parser(formats, columns, [])
    ant_array = np.genfromtxt(
        antenna_positions_file,
        delimiter=",",
        autostrip=True,
        skip_header=1,
        dtype=dt.dtype,
    )
    antenna_names = ant_array["name"]
    antenna_numbers = ant_array["number"]
    antenna_positions = np.stack((ant_array["x"], ant_array["y"], ant_array["z"])).T

    return antenna_names, antenna_numbers, antenna_positions.astype("float")


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

    astropy_sites = EarthLocation.get_site_names()
    telescope_keys = list(telescope_dict_in.keys())
    telescope_list = [tel.lower() for tel in telescope_keys]
    if telescope_name in astropy_sites:
        telescope_loc = EarthLocation.of_site(telescope_name)

        obj = Telescope()
        obj.telescope_name = telescope_name
        obj.citation = "astropy sites"
        obj.telescope_location = np.array(
            [telescope_loc.x.value, telescope_loc.y.value, telescope_loc.z.value]
        )

    elif telescope_name.lower() in telescope_list:
        telescope_index = telescope_list.index(telescope_name.lower())
        telescope_dict = telescope_dict_in[telescope_keys[telescope_index]]
        obj = Telescope()
        obj.citation = telescope_dict["citation"]
        obj.telescope_name = telescope_keys[telescope_index]
        if telescope_dict["center_xyz"] is not None:
            obj.telescope_location = telescope_dict["center_xyz"]
        else:
            if (
                telescope_dict["latitude"] is None
                or telescope_dict["longitude"] is None
                or telescope_dict["altitude"] is None
            ):
                raise ValueError(
                    "either the center_xyz or the "
                    "latitude, longitude and altitude of the "
                    "telescope must be specified"
                )
            obj.telescope_location_lat_lon_alt = (
                telescope_dict["latitude"],
                telescope_dict["longitude"],
                telescope_dict["altitude"],
            )
    else:
        # no telescope matching this name
        return False

    # check for extra info
    if telescope_name.lower() in telescope_list:
        telescope_index = telescope_list.index(telescope_name.lower())
        telescope_dict = telescope_dict_in[telescope_keys[telescope_index]]
        if "diameters" in telescope_dict.keys():
            obj.antenna_diameters = telescope_dict["diameters"]

        if "antenna_positions_file" in telescope_dict.keys():
            antpos_file = os.path.join(
                DATA_PATH, telescope_dict["antenna_positions_file"]
            )
            antenna_names, antenna_numbers, antenna_positions = _parse_antpos_file(
                antpos_file
            )
            obj.Nants_telescope = antenna_names.size
            obj.antenna_names = antenna_names
            obj.antenna_numbers = antenna_numbers
            obj.antenna_positions = antenna_positions

    obj.check(run_check_acceptability=True)

    return obj
