# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Telescope information and known telescope list."""
from __future__ import annotations

import copy
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np
from astropy import units
from astropy.coordinates import Angle, EarthLocation

from . import parameter as uvp
from . import utils
from .data import DATA_PATH
from .utils.io import antpos
from .utils.io import hdf5 as hdf5_utils
from .uvbase import UVBase

__all__ = ["Telescope", "known_telescopes", "known_telescope_location", "get_telescope"]

try:
    from lunarsky import MoonLocation

    # This can be built from utils.allowed_location_types in python >= 3.11
    # but in 3.10 Union has to be declare with types
    Locations = Union[EarthLocation, MoonLocation]
except ImportError:
    Locations = EarthLocation

# We use astropy sites for telescope locations. The dict below is for
# telescopes not in astropy sites, or to include extra information for a telescope.

# The center_xyz is the location of the telescope in ITRF (earth-centered frame)

# Antenna positions can be specified via a csv file with the following columns:
# "name" -- antenna name, "number" -- antenna number, "x", "y", "z" -- ECEF coordinates
# relative to the telescope location.
_KNOWN_TELESCOPES = {
    "PAPER": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("-30d43m17.5s"), lon=Angle("21d25m41.9s"), height=1073.0 * units.m
        ),
        "citation": (
            "value taken from capo/cals/hsa7458_v000.py, "
            "comment reads KAT/SA  (GPS), altitude from elevationmap.net"
        ),
    },
    "HERA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("-30.72152612068925d"),
            lon=Angle("21.42830382686301d"),
            height=1051.69 * units.m,
        ),
        "antenna_diameters": 14.0,
        "antenna_positions_file": "hera_ant_pos.csv",
        "citation": (
            "value taken from hera_mc geo.py script "
            "(using hera_cm_db_updates under the hood.)"
        ),
    },
    "SMA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("19d49m27.13895s"),
            lon=Angle("-155d28m39.08279s"),
            height=4083.948144 * units.m,
        ),
        "citation": "Ho, P. T. P., Moran, J. M., & Lo, K. Y. 2004, ApJL, 616, L1",
    },
    "SZA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("37d16m49.3698s"),
            lon=Angle("-118d08m29.9126s"),
            height=2400.0 * units.m,
        ),
        "citation": "Unknown",
    },
    "OVRO-LWA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("37.239777271d"),
            lon=Angle("-118.281666695d"),
            height=1183.48 * units.m,
        ),
        "citation": "OVRO Sharepoint Documentation",
    },
    "MWA": {"antenna_positions_file": "mwa_ant_pos.csv"},
}


# Deprecation to handle accessing old keys of KNOWN_TELESCOPES
class TelMapping(Mapping):
    def __init__(self, mapping=()):
        self._mapping = dict(mapping)

    def __getitem__(self, key):
        warnings.warn(
            "Directly accessing the KNOWN_TELESCOPES dict is deprecated. If you "
            "need a telescope location, use the known_telescope_location function. "
            "For a full Telescope object use the classmethod "
            "Telescope.from_known_telescopes.",
            DeprecationWarning,
        )
        if key in ["latitude", "longitude", "altitude", "center_xyz"]:
            if key == "latitude":
                return self._mapping["location"].lat.rad
            if key == "longitude":
                return self._mapping["location"].lon.rad
            if key == "altitude":
                return self._mapping["location"].height.to_value("m")
            if key == "center_xyz":
                return units.Quantity(self._mapping["location"].geocentric).to_value(
                    "m"
                )

        return self._mapping[key]

    def __len__(self):
        return len(self._mapping)

    def __iter__(self):
        return iter(self._mapping)


KNOWN_TELESCOPES = TelMapping(
    (name, TelMapping(tel_dict)) for name, tel_dict in _KNOWN_TELESCOPES.items()
)


def known_telescopes():
    """
    Get list of known telescopes.

    Returns
    -------
    list of str
        List of known telescope names.
    """
    astropy_sites = [site for site in EarthLocation.get_site_names() if site != ""]
    known_telescopes = list(set(astropy_sites + list(_KNOWN_TELESCOPES.keys())))
    return known_telescopes


def get_telescope(telescope_name, telescope_dict_in=_KNOWN_TELESCOPES):
    """
    Get Telescope object for a telescope in telescope_dict. Deprecated.

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
    warnings.warn(
        "This method is deprecated and will be removed in version 3.2. If you "
        "just need a telescope location, use the known_telescope_location function. "
        "For a full Telescope object use the classmethod "
        "Telescope.from_known_telescopes.",
        DeprecationWarning,
    )
    return Telescope.from_known_telescopes(
        telescope_name, known_telescope_dict=telescope_dict_in, run_check=False
    )


def known_telescope_location(
    name: str,
    return_citation: bool = False,
    known_telescope_dict: dict = _KNOWN_TELESCOPES,
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
    known_telescopes = {k.lower(): v for k, v in known_telescope_dict.items()}

    # first deal with location.
    if name in astropy_sites:
        location = EarthLocation.of_site(name)

        citation = "astropy sites"
    elif name.lower() in known_telescopes:
        telescope_dict = known_telescopes[name.lower()]
        citation = telescope_dict["citation"]

        try:
            location = telescope_dict["location"]
        except KeyError as ke:
            raise KeyError(
                "Missing location information in known_telescopes_dict "
                f"for telescope {name}."
            ) from ke
    else:
        # no telescope matching this name
        raise ValueError(
            f"Telescope {name} is not in astropy_sites or known_telescopes_dict."
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


class Telescope(UVBase):
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

    def __getattr__(self, __name):
        """Handle old names attributes."""
        if __name == "telescope_location":
            warnings.warn(
                "The Telescope.telescope_location attribute is deprecated, use "
                "Telescope.location instead (which contains an astropy "
                "EarthLocation object). This will become an error in version 3.2.",
                DeprecationWarning,
            )
            return self._location.xyz()
        elif __name == "telescope_name":
            warnings.warn(
                "The Telescope.telescope_name attribute is deprecated, use "
                "Telescope.name instead. This will become an error in version 3.2.",
                DeprecationWarning,
            )
            return self.name

        return super().__getattribute__(__name)

    def __setattr__(self, __name, __value):
        """Handle old names for telescope metadata."""
        if __name == "telescope_location":
            warnings.warn(
                "The Telescope.telescope_location attribute is deprecated, use "
                "Telescope.location instead (which should be set to an astropy "
                "EarthLocation object). This will become an error in version 3.2.",
                DeprecationWarning,
            )
            self._location.set_xyz(__value)
            return
        elif __name == "telescope_name":
            warnings.warn(
                "The Telescope.telescope_name attribute is deprecated, use "
                "Telescope.name instead. This will become an error in version 3.2.",
                DeprecationWarning,
            )
            self.name = __value
            return

        return super().__setattr__(__name, __value)

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
            utils.coordinates.check_surface_based_positions(
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
        known_telescope_dict: dict = _KNOWN_TELESCOPES,
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

        Raises
        ------
        ValueError
            If self.name is not set or if ((location is missing or overwrite is
            set) and self.name is not found either astropy sites our our
            known_telescopes dict)

        """
        if self.name is None:
            raise ValueError(
                "The telescope name attribute must be set to update from "
                "known_telescopes."
            )
        known_telescopes = {k.lower(): v for k, v in known_telescope_dict.items()}

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
            self.citation = citation
            if "astropy sites" in citation:
                astropy_sites_list.append("telescope_location")
            else:
                known_telescope_list.append("telescope_location")

        # check for extra info
        if self.name.lower() in known_telescopes:
            telescope_dict = known_telescopes[self.name.lower()]

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
                antenna_names, antenna_numbers, antenna_positions = (
                    antpos.read_antpos_csv(antpos_file)
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
        known_telescope_dict: dict = _KNOWN_TELESCOPES,
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
    def new(
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

        if not isinstance(location, tuple(utils.coordinates.allowed_location_types)):
            raise ValueError(
                "telescope_location has an unsupported type, it must be one of "
                f"{utils.coordinates.allowed_location_types}"
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
            x_orientation = utils.XORIENTMAP[x_orientation.lower()]
            tel_obj.x_orientation = x_orientation

        if antenna_diameters is not None:
            tel_obj.antenna_diameters = np.asarray(antenna_diameters)

        tel_obj.check()

        return tel_obj

    @classmethod
    def from_hdf5(
        cls,
        filename: str | Path | hdf5_utils.HDF5Meta,
        required_keys: list | None = None,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
    ):
        """
        Initialize a new Telescope object from an HDF5 file.

        The file must have a Header dataset that has the appropriate header
        items. UVH5, CalH5 and UVFlag HDF5 files have these.

        Parameters
        ----------
        path : str or Path or subclass of hdf5_utils.HDF5Meta
            The filename to read from.

        """
        if required_keys is None:
            required_keys = ["telescope_name", "latitude", "longitude", "altitude"]
        tel_obj = cls()

        if not isinstance(filename, hdf5_utils.HDF5Meta):
            if isinstance(filename, h5py.File):
                path = Path(filename.filename).resolve()
            elif isinstance(filename, h5py.Group):
                path = Path(filename.file.filename).resolve()
            else:
                path = Path(filename).resolve()
            meta = hdf5_utils.HDF5Meta(path)

        else:
            meta = copy.deepcopy(filename)

        tel_obj.location = meta.telescope_location_obj

        telescope_attrs = {
            "telescope_name": "name",
            "Nants_telescope": "Nants",
            "antenna_names": "antenna_names",
            "antenna_numbers": "antenna_numbers",
            "antenna_positions": "antenna_positions",
            "instrument": "instrument",
            "x_orientation": "x_orientation",
            "antenna_diameters": "antenna_diameters",
        }
        for attr, tel_attr in telescope_attrs.items():
            try:
                setattr(tel_obj, tel_attr, getattr(meta, attr))
            except (AttributeError, KeyError) as e:
                if attr in required_keys:
                    raise KeyError(str(e)) from e
                else:
                    pass

        if run_check:
            tel_obj.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return tel_obj

    def write_hdf5_header(self, header):
        """Write the telescope metadata to an hdf5 dataset.

        This is assumed to be writing to a general header (e.g. for uvh5),
        so the header names include 'telescope'.

        Parameters
        ----------
        header : HDF5 dataset
            Dataset to write the telescope metadata to.

        """
        header["telescope_frame"] = np.bytes_(self._location.frame)
        if self._location.frame == "mcmf":
            header["ellipsoid"] = self._location.ellipsoid
        lat, lon, alt = self.location_lat_lon_alt_degrees
        header["latitude"] = lat
        header["longitude"] = lon
        header["altitude"] = alt
        header["telescope_name"] = np.bytes_(self.name)
        header["Nants_telescope"] = self.Nants
        header["antenna_numbers"] = self.antenna_numbers
        header["antenna_positions"] = self.antenna_positions
        header["antenna_names"] = np.asarray(self.antenna_names, dtype="bytes")

        if self.instrument is not None:
            header["instrument"] = np.bytes_(self.instrument)
        if self.x_orientation is not None:
            header["x_orientation"] = np.bytes_(self.x_orientation)
        if self.antenna_diameters is not None:
            header["antenna_diameters"] = self.antenna_diameters

    def get_enu_antpos(self):
        """
        Get antenna positions in East, North, Up coordinates in units of meters.

        Returns
        -------
        antpos : ndarray
            Antenna positions in East, North, Up coordinates in units of
            meters, shape=(Nants, 3)

        """
        antenna_xyz = self.antenna_positions + self._location.xyz()
        antpos = utils.ENU_from_ECEF(antenna_xyz, center_loc=self.location)

        return antpos
