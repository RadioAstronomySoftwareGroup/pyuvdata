# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Telescope information and known telescope list."""
import os
import warnings
from typing import Literal, Union

import numpy as np
from astropy import units
from astropy.coordinates import Angle, EarthLocation

from pyuvdata.data import DATA_PATH

from . import parameter as uvp
from . import utils as uvutils
from . import uvbase

__all__ = ["Telescope", "known_telescopes"]

Locations = Union[uvutils.allowed_location_types]

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
        "antenna_diameters": 14.0,
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
    "SZA": {
        "center_xyz": None,
        "latitude": Angle("37d16m49.3698s").radian,
        "longitude": Angle("-118d08m29.9126s").radian,
        "altitude": 2400.0,
        "citation": "Unknown",
    },
    "OVRO-LWA": {
        "center_xyz": None,
        "latitude": Angle("37.239777271d").radian,
        "longitude": Angle("-118.281666695d").radian,
        "altitude": 1183.48,
        "citation": "OVRO Sharepoint Documentation",
    },
    "MWA": {"antenna_positions_file": "mwa_ant_pos.csv"},
}


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


def known_telescopes():
    """
    Get list of known telescopes.

    Returns
    -------
    list of str
        List of known telescope names.
    """
    astropy_sites = [site for site in EarthLocation.get_site_names() if site != ""]
    known_telescopes = list(set(astropy_sites + list(KNOWN_TELESCOPES.keys())))
    return known_telescopes


def known_telescope_location(
    name: str,
    return_citation: bool = False,
    known_telescope_dict: dict = KNOWN_TELESCOPES,
):
    """
    Get the location for a known telescope.

    Parameters
    ----------
    name : str
        Name of the telescope
    return_citation : bool
        Option to return the citation.
    known_telescope_dict: dict
        This should only be used for testing. This allows passing in a
        different dict to use in place of the KNOWN_TELESCOPES dict.

    Returns
    -------
    location : EarthLocation
        Telescope location as an EarthLocation object.
    citation : str, optional
        Citation string.

    """
    astropy_sites = EarthLocation.get_site_names()
    telescope_keys = list(known_telescope_dict.keys())
    telescope_list = [tel.lower() for tel in telescope_keys]

    # first deal with location.
    if name in astropy_sites:
        location = EarthLocation.of_site(name)

        citation = "astropy sites"
    elif name.lower() in telescope_list:
        telescope_index = telescope_list.index(name.lower())
        telescope_dict = known_telescope_dict[telescope_keys[telescope_index]]
        citation = telescope_dict["citation"]

        if telescope_dict["center_xyz"] is not None:
            location = EarthLocation.from_geocentric(
                *telescope_dict["center_xyz"], unit="m"
            )
        else:
            if (
                telescope_dict["latitude"] is None
                or telescope_dict["longitude"] is None
                or telescope_dict["altitude"] is None
            ):
                raise ValueError(
                    "Bad location information in known_telescopes_dict "
                    f"for telescope {name}. Either the center_xyz "
                    "or the latitude, longitude and altitude of the "
                    "telescope must be specified."
                )
            location = EarthLocation.from_geodetic(
                lat=telescope_dict["latitude"] * units.rad,
                lon=telescope_dict["longitude"] * units.rad,
                height=telescope_dict["altitude"] * units.m,
            )
    else:
        # no telescope matching this name
        raise ValueError(
            f"Telescope {name} is not in astropy_sites or " "known_telescopes_dict."
        )

    if not return_citation:
        return location
    else:
        return location, citation


def get_antenna_params(
    *,
    antenna_positions: np.ndarray | dict[str | int, np.ndarray],
    antenna_names: list[str] | None = None,
    antenna_numbers: list[int] | None = None,
    antname_format: str = "{0:03d}",
) -> tuple[np.ndarray, list[str], list[int]]:
    """Configure antenna parameters for new UVData object."""
    # Get Antenna Parameters

    if isinstance(antenna_positions, dict):
        keys = list(antenna_positions.keys())
        if all(isinstance(key, int) for key in keys):
            antenna_numbers = list(antenna_positions.keys())
        elif all(isinstance(key, str) for key in keys):
            antenna_names = list(antenna_positions.keys())
        else:
            raise ValueError(
                "antenna_positions must be a dictionary with keys that are all type "
                "int or all type str."
            )
        antenna_positions = np.array(list(antenna_positions.values()))

    if antenna_numbers is None and antenna_names is None:
        raise ValueError(
            "Either antenna_numbers or antenna_names must be provided unless "
            "antenna_positions is a dict."
        )

    if antenna_names is None:
        antenna_names = [antname_format.format(i) for i in antenna_numbers]
    elif antenna_numbers is None:
        try:
            antenna_numbers = [int(name) for name in antenna_names]
        except ValueError as e:
            raise ValueError(
                "Antenna names must be integers if antenna_numbers is not provided."
            ) from e

    if not isinstance(antenna_positions, np.ndarray):
        raise ValueError("antenna_positions must be a numpy array or a dictionary.")

    if antenna_positions.shape != (len(antenna_numbers), 3):
        raise ValueError(
            "antenna_positions must be a 2D array with shape (N_antennas, 3), "
            f"got {antenna_positions.shape}"
        )

    if len(antenna_names) != len(set(antenna_names)):
        raise ValueError("Duplicate antenna names found.")

    if len(antenna_numbers) != len(set(antenna_numbers)):
        raise ValueError("Duplicate antenna numbers found.")

    if len(antenna_numbers) != len(antenna_names):
        raise ValueError("antenna_numbers and antenna_names must have the same length.")

    return antenna_positions, np.asarray(antenna_names), np.asarray(antenna_numbers)


class Telescope(uvbase.UVBase):
    """
    A class for telescope metadata, used on UVData, UVCal and UVFlag objects.

    Attributes
    ----------
    UVParameter objects :
        For full list see the documentation on ReadTheDocs:
        http://pyuvdata.readthedocs.io/en/latest/.

    """

    def __init__(self):
        """Create a new Telescope object."""
        # add the UVParameters to the class
        # use the same names as in UVData so they can be automatically set
        self.citation = None

        self._name = uvp.UVParameter(
            "name", description="name of telescope (string)", form="str"
        )
        desc = (
            "telescope location: Either an astropy.EarthLocation oject or a "
            "lunarsky MoonLocation object."
        )
        self._location = uvp.LocationParameter("location", description=desc, tols=1e-3)

        desc = "Number of antennas in the array."
        self._Nants = uvp.UVParameter("Nants", description=desc, expected_type=int)

        desc = (
            "Array of antenna names, shape (Nants), "
            "with numbers given by antenna_numbers."
        )
        self._antenna_names = uvp.UVParameter(
            "antenna_names", description=desc, form=("Nants",), expected_type=str
        )

        desc = (
            "Array of integer antenna numbers corresponding to antenna_names, "
            "shape (Nants)."
        )
        self._antenna_numbers = uvp.UVParameter(
            "antenna_numbers", description=desc, form=("Nants",), expected_type=int
        )

        desc = (
            "Array giving coordinates of antennas relative to "
            "location (ITRF frame), shape (Nants, 3), "
            "units meters. See the tutorial page in the documentation "
            "for an example of how to convert this to topocentric frame."
        )
        self._antenna_positions = uvp.UVParameter(
            "antenna_positions",
            description=desc,
            form=("Nants", 3),
            expected_type=float,
            tols=1e-3,  # 1 mm
        )

        self._instrument = uvp.UVParameter(
            "instrument",
            description="Receiver or backend. Sometimes identical to name.",
            required=False,
            form="str",
            expected_type=str,
        )

        desc = (
            "Orientation of the physical dipole corresponding to what is "
            "labelled as the x polarization. Options are 'east' "
            "(indicating east/west orientation) and 'north (indicating "
            "north/south orientation)."
        )
        self._x_orientation = uvp.UVParameter(
            "x_orientation",
            description=desc,
            required=False,
            expected_type=str,
            acceptable_vals=["east", "north"],
        )

        desc = (
            "Antenna diameters in meters. Used by CASA to "
            "construct a default beam if no beam is supplied."
        )
        self._antenna_diameters = uvp.UVParameter(
            "antenna_diameters",
            description=desc,
            required=False,
            form=("Nants",),
            expected_type=float,
            tols=1e-3,  # 1 mm
        )

        super(Telescope, self).__init__()

    def check(self, *, check_extra=True, run_check_acceptability=True):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check all parameters, otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.

        Returns
        -------
        bool
            True if check passes

        Raises
        ------
        ValueError
            if parameter shapes or types are wrong or do not have acceptable
            values (if run_check_acceptability is True)

        """
        # first run the basic check from UVBase

        super(Telescope, self).check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        if run_check_acceptability:
            # Check antenna positions
            uvutils.check_surface_based_positions(
                antenna_positions=self.antenna_positions,
                telescope_loc=self.location,
                raise_error=False,
            )

        return True

    def update_params_from_known_telescopes(
        self,
        *,
        overwrite: bool = False,
        warn: bool = True,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
        known_telescope_dict: dict = KNOWN_TELESCOPES,
    ):
        """
        Update the parameters based on telescope in known_telescopes.

        This fills in any missing parameters (or to overwrite parameters that
        have inaccurate values) on self that are available for a telescope from
        either Astropy sites and/or from the KNOWN_TELESCOPES dict. This is
        primarily used on UVData, UVCal and UVFlag to fill in information that
        is missing, especially in older files.

        Parameters
        ----------
        overwrite : bool
            If set, overwrite parameters with information from Astropy sites
            and/or from the KNOWN_TELESCOPES dict. Defaults to False.
        warn : bool
            Option to issue a warning listing all modified parameters.
            Defaults to True.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after updating.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            updating.
        known_telescope_dict: dict
            This should only be used for testing. This allows passing in a
            different dict to use in place of the KNOWN_TELESCOPES dict.

        """
        if self.name is None:
            raise ValueError(
                "The telescope name attribute must be set to update from "
                "known_telescopes."
            )
        telescope_keys = list(known_telescope_dict.keys())
        telescope_list = [tel.lower() for tel in telescope_keys]

        astropy_sites_list = []
        known_telescope_list = []
        # first deal with location.
        if overwrite or self.location is None:
            location, citation = known_telescope_location(
                self.name,
                return_citation=True,
                known_telescope_dict=known_telescope_dict,
            )
            self.location = location
            if "astropy sites" in citation:
                astropy_sites_list.append("telescope_location")
            else:
                known_telescope_list.append("telescope_location")

        # check for extra info
        if self.name.lower() in telescope_list:
            telescope_index = telescope_list.index(self.name.lower())
            telescope_dict = known_telescope_dict[telescope_keys[telescope_index]]

            if "antenna_positions_file" in telescope_dict.keys() and (
                overwrite
                or self.antenna_names is None
                or self.antenna_numbers is None
                or self.antenna_positions is None
                or self.Nants is None
            ):
                antpos_file = os.path.join(
                    DATA_PATH, telescope_dict["antenna_positions_file"]
                )
                antenna_names, antenna_numbers, antenna_positions = _parse_antpos_file(
                    antpos_file
                )
                ant_info = {
                    "Nants": antenna_names.size,
                    "antenna_names": antenna_names,
                    "antenna_numbers": antenna_numbers,
                    "antenna_positions": antenna_positions,
                }
                ant_params_missing = []
                for key in ant_info.keys():
                    if getattr(self, key) is None:
                        ant_params_missing.append(key)
                if overwrite or len(ant_params_missing) == len(ant_info.keys()):
                    for key, value in ant_info.items():
                        known_telescope_list.append(key)
                        setattr(self, key, value)
                elif self.antenna_names is not None or self.antenna_numbers is not None:
                    ant_inds = []
                    telescope_ant_inds = []
                    # first try to match using names only
                    if self.antenna_names is not None:
                        for index, antname in enumerate(self.antenna_names):
                            if antname in ant_info["antenna_names"]:
                                ant_inds.append(index)
                                telescope_ant_inds.append(
                                    np.where(ant_info["antenna_names"] == antname)[0][0]
                                )
                    # next try using numbers
                    if self.antenna_numbers is not None:
                        if len(ant_inds) != self.Nants:
                            for index, antnum in enumerate(self.antenna_numbers):
                                # only update if not already found
                                if (
                                    index not in ant_inds
                                    and antnum in ant_info["antenna_numbers"]
                                ):
                                    this_ant_ind = np.where(
                                        ant_info["antenna_numbers"] == antnum
                                    )[0][0]
                                    # make sure we don't already have this antenna
                                    #  associated with another antenna
                                    if this_ant_ind not in telescope_ant_inds:
                                        ant_inds.append(index)
                                        telescope_ant_inds.append(this_ant_ind)
                    if len(ant_inds) != self.Nants:
                        warnings.warn(
                            "Not all antennas have metadata in the "
                            f"known_telescope data. Not setting {ant_params_missing}."
                        )
                    else:
                        known_telescope_list.extend(ant_params_missing)
                        if "antenna_positions" in ant_params_missing:
                            self.antenna_positions = ant_info["antenna_positions"][
                                telescope_ant_inds, :
                            ]
                        if "antenna_names" in ant_params_missing:
                            self.antenna_names = ant_info["antenna_names"][
                                telescope_ant_inds
                            ]

                        if "antenna_numbers" in ant_params_missing:
                            self.antenna_numbers = ant_info["antenna_numbers"][
                                telescope_ant_inds
                            ]

            if "antenna_diameters" in telescope_dict.keys() and (
                overwrite or self.antenna_diameters is None
            ):
                antenna_diameters = np.atleast_1d(telescope_dict["antenna_diameters"])
                if antenna_diameters.size == 1:
                    known_telescope_list.append("antenna_diameters")
                    self.antenna_diameters = (
                        np.zeros(self.Nants, dtype=float) + antenna_diameters[0]
                    )
                elif antenna_diameters.size == self.Nants:
                    known_telescope_list.append("antenna_diameters")
                    self.antenna_diameters = antenna_diameters
                else:
                    if warn:
                        warnings.warn(
                            "antenna_diameters are not set because the number "
                            "of antenna_diameters on known_telescopes_dict is "
                            "more than one and does not match Nants for "
                            f"telescope {self.name}."
                        )

            if "x_orientation" in telescope_dict.keys() and (
                overwrite or self.x_orientation is None
            ):
                known_telescope_list.append("x_orientation")
                self.x_orientation = telescope_dict["x_orientation"]

        full_list = astropy_sites_list + known_telescope_list
        if warn and len(full_list) > 0:
            warn_str = ", ".join(full_list) + " are not set or are being overwritten. "
            specific_str = []
            if len(astropy_sites_list) > 0:
                specific_str.append(
                    ", ".join(astropy_sites_list) + " are set using values from "
                    f"astropy sites for {self.name}."
                )
            if len(known_telescope_list) > 0:
                specific_str.append(
                    ", ".join(known_telescope_list) + " are set using values "
                    f"from known telescopes for {self.name}."
                )
            warn_str += " ".join(specific_str)
            warnings.warn(warn_str)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    @classmethod
    def from_known_telescopes(
        cls,
        name: str,
        *,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
        known_telescope_dict: dict = KNOWN_TELESCOPES,
    ):
        """
        Create a new Telescope object using information from known_telescopes.

        Parameters
        ----------
        name : str
            Name of the telescope.
        run_check : bool
            Option to check for the existence and proper shapes of parameters.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters.
        known_telescope_dict: dict
            This should only be used for testing. This allows passing in a
            different dict to use in place of the KNOWN_TELESCOPES dict.

        Returns
        -------
        Telescope
            A new Telescope object populated with information from
            known_telescopes.

        """
        tel_obj = cls()
        tel_obj.name = name
        tel_obj.update_params_from_known_telescopes(
            warn=False,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            known_telescope_dict=known_telescope_dict,
        )
        return tel_obj

    @classmethod
    def from_params(
        cls,
        name: str,
        location: Locations,
        antenna_positions: np.ndarray | dict[str | int, np.ndarray] | None = None,
        antenna_names: list[str] | np.ndarray | None = None,
        antenna_numbers: list[int] | np.ndarray | None = None,
        antname_format: str = "{0:03d}",
        instrument: str | None = None,
        x_orientation: Literal["east", "north", "e", "n", "ew", "ns"] | None = None,
        antenna_diameters: list[float] | np.ndarray | None = None,
    ):
        """
        Initialize a new Telescope object from keyword arguments.

        Parameters
        ----------
        name : str
            Telescope name.
        location : EarthLocation or MoonLocation object
            Telescope location as an astropy EarthLocation object or MoonLocation
            object.
        antenna_positions : ndarray of float or dict of ndarray of float
            Array of antenna positions in ECEF coordinates in meters.
            If a dict, keys can either be antenna numbers or antenna names, and
            values are position arrays. Keys are interpreted as antenna numbers
            if they are integers, otherwise they are interpreted as antenna names
            if strings. You cannot provide a mix of different types of keys.
        antenna_names : list or np.ndarray of str, optional
            List or array of antenna names. Not used if antenna_positions is a
            dict with string keys. Otherwise, if not provided, antenna numbers
            will be used to form the antenna_names, according to the antname_format.
        antenna_numbers : list or np.ndarray of int, optional
            List or array of antenna numbers. Not used if antenna_positions is a
            dict with integer keys. Otherwise, if not provided, antenna names
            will be used to form the antenna_numbers, but in this case the
            antenna_names must be strings that can be converted to integers.
        antname_format : str, optional
            Format string for antenna names. Default is '{0:03d}'.
        instrument : str, optional
            Instrument name.
        x_orientation : str
            Orientation of the x-axis. Options are 'east', 'north', 'e', 'n',
            'ew', 'ns'.
        antenna_diameters :  list or np.ndarray of float, optional
            List or array of antenna diameters.

        Returns
        -------
        Telescope object
            A Telescope object with the specified metadata.

        """
        tel_obj = cls()

        if not isinstance(location, tuple(uvutils.allowed_location_types)):
            raise ValueError(
                "telescope_location has an unsupported type, it must be one of "
                f"{uvutils.allowed_location_types}"
            )

        tel_obj.name = name
        tel_obj.location = location

        antenna_positions, antenna_names, antenna_numbers = get_antenna_params(
            antenna_positions=antenna_positions,
            antenna_names=antenna_names,
            antenna_numbers=antenna_numbers,
            antname_format=antname_format,
        )

        tel_obj.antenna_positions = antenna_positions
        tel_obj.antenna_names = antenna_names
        tel_obj.antenna_numbers = antenna_numbers
        tel_obj.Nants = len(antenna_numbers)

        if instrument is not None:
            tel_obj.instrument = instrument

        if x_orientation is not None:
            x_orientation = uvutils.XORIENTMAP[x_orientation.lower()]
            tel_obj.x_orientation = x_orientation

        if antenna_diameters is not None:
            tel_obj.antenna_diameters = np.asarray(antenna_diameters)

        tel_obj.check()

        return tel_obj
