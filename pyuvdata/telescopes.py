# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Telescope information and known telescope list."""

import os
import warnings

import numpy as np
from astropy.coordinates import Angle, EarthLocation

from pyuvdata.data import DATA_PATH

from . import parameter as uvp
from . import utils as uvutils
from . import uvbase

__all__ = ["Telescope", "known_telescopes"]

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

    dt = np.rec.format_parser(formats, columns, [])
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

        self._name = uvp.UVParameter(
            "name", description="name of telescope (string)", form="str"
        )
        desc = (
            "telescope location: xyz in ITRF (earth-centered frame). "
            "Can also be set using telescope_location_lat_lon_alt or "
            "telescope_location_lat_lon_alt_degrees properties"
        )
        self._location = uvp.LocationParameter("location", description=desc, tols=1e-3)

        self._instrument = uvp.UVParameter(
            "instrument",
            description="Receiver or backend. Sometimes identical to telescope_name.",
            required=False,
            form="str",
            expected_type=str,
        )

        desc = "Number of antennas in the array."
        self._Nants = uvp.UVParameter(
            "Nants", required=False, description=desc, expected_type=int
        )

        desc = (
            "Array of antenna names, shape (Nants), "
            "with numbers given by antenna_numbers."
        )
        self._antenna_names = uvp.UVParameter(
            "antenna_names",
            description=desc,
            required=False,
            form=("Nants",),
            expected_type=str,
        )

        desc = (
            "Array of integer antenna numbers corresponding to antenna_names, "
            "shape (Nants)."
        )
        self._antenna_numbers = uvp.UVParameter(
            "antenna_numbers",
            description=desc,
            required=False,
            form=("Nants",),
            expected_type=int,
        )

        desc = (
            "Array giving coordinates of antennas relative to "
            "telescope_location (ITRF frame), shape (Nants, 3), "
            "units meters. See the tutorial page in the documentation "
            "for an example of how to convert this to topocentric frame."
        )
        self._antenna_positions = uvp.UVParameter(
            "antenna_positions",
            description=desc,
            required=False,
            form=("Nants", 3),
            expected_type=float,
            tols=1e-3,  # 1 mm
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
                telescope_frame=self._location.frame,
                raise_error=False,
            )

        return True

    @property
    def location_obj(self):
        """The location as an EarthLocation or MoonLocation object."""
        if self.location is None:
            return None

        if self._location.frame == "itrs":
            return EarthLocation.from_geocentric(*self.location, unit="m")
        elif uvutils.hasmoon and self._location.frame == "mcmf":
            moon_loc = uvutils.MoonLocation.from_selenocentric(*self.location, unit="m")
            moon_loc.ellipsoid = self._location.ellipsoid
            return moon_loc

    @location_obj.setter
    def location_obj(self, val):
        if isinstance(val, EarthLocation):
            self._location.frame = "itrs"
        elif isinstance(val, uvutils.MoonLocation):
            self._location.frame = "mcmf"
            self._location.ellipsoid = val.ellipsoid
        else:
            raise ValueError(
                "location_obj is not a recognized location object. Must be an "
                "EarthLocation or MoonLocation object."
            )
        self.location = np.array(
            [val.x.to("m").value, val.y.to("m").value, val.z.to("m").value]
        )

    def _set_uvcal_requirements(self):
        """Set the UVParameter required fields appropriately for UVCal."""
        self._name.required = True
        self._location.required = True
        self._instrument.required = False
        self._Nants.required = True
        self._antenna_names.required = True
        self._antenna_numbers.required = True
        self._antenna_positions.required = True
        self._antenna_diameters.required = False
        self._x_orientation.required = True

    def _set_uvdata_requirements(self):
        """Set the UVParameter required fields appropriately for UVData."""
        self._name.required = True
        self._location.required = True
        self._instrument.required = True
        self._Nants.required = True
        self._antenna_names.required = True
        self._antenna_numbers.required = True
        self._antenna_positions.required = True
        self._x_orientation.required = False
        self._antenna_diameters.required = False

    def update_params_from_known_telescopes(
        self,
        *,
        known_telescope_dict=KNOWN_TELESCOPES,
        overwrite=False,
        warn=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Set the parameters based on telescope in known_telescopes.

        Parameters
        ----------
        known_telescope_dict: dict
            telescope info dict. Default is KNOWN_TELESCOPES
            (other values are only used for testing)

        """
        astropy_sites = EarthLocation.get_site_names()
        telescope_keys = list(known_telescope_dict.keys())
        telescope_list = [tel.lower() for tel in telescope_keys]

        astropy_sites_list = []
        known_telescope_list = []
        # first deal with location.
        if overwrite or self.location is None:
            if self.name in astropy_sites:
                tel_loc = EarthLocation.of_site(self.name)

                self.citation = "astropy sites"
                self.location = np.array(
                    [tel_loc.x.value, tel_loc.y.value, tel_loc.z.value]
                )
                astropy_sites_list.append("telescope_location")

            elif self.name.lower() in telescope_list:
                telescope_index = telescope_list.index(self.name.lower())
                telescope_dict = known_telescope_dict[telescope_keys[telescope_index]]
                self.citation = telescope_dict["citation"]

                known_telescope_list.append("telescope_location")
                if telescope_dict["center_xyz"] is not None:
                    self.location = telescope_dict["center_xyz"]
                else:
                    if (
                        telescope_dict["latitude"] is None
                        or telescope_dict["longitude"] is None
                        or telescope_dict["altitude"] is None
                    ):
                        raise ValueError(
                            "Bad location information in known_telescopes_dict "
                            f"for telescope {self.name}. Either the center_xyz "
                            "or the latitude, longitude and altitude of the "
                            "telescope must be specified."
                        )
                    self.location_lat_lon_alt = (
                        telescope_dict["latitude"],
                        telescope_dict["longitude"],
                        telescope_dict["altitude"],
                    )
            else:
                # no telescope matching this name
                raise ValueError(
                    f"Telescope {self.name} is not in astropy_sites or "
                    "known_telescopes_dict."
                )

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
                if overwrite or all(
                    getattr(self, key) is None for key in ant_info.keys()
                ):
                    for key, value in ant_info.items():
                        known_telescope_list.append(key)
                        setattr(self, key, value)
                elif (
                    self.antenna_positions is None
                    and self.antenna_names is not None
                    and self.antenna_numbers is not None
                ):
                    ant_inds = []
                    telescope_ant_inds = []
                    # first try to match using names only
                    for index, antname in enumerate(self.antenna_names):
                        if antname in ant_info["antenna_names"]:
                            ant_inds.append(index)
                            telescope_ant_inds.append(
                                np.where(ant_info["antenna_names"] == antname)[0][0]
                            )
                    # next try using numbers
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
                            "Not all antennas have positions in the "
                            "known_telescope data. Not setting antenna_positions."
                        )
                    else:
                        known_telescope_list.append("antenna_positions")
                        self.antenna_positions = ant_info["antenna_positions"][
                            telescope_ant_inds, :
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
    def get_telescope_from_known_telescopes(
        cls, name: str, *, known_telescope_dict: dict = KNOWN_TELESCOPES
    ):
        """
        Create a new Telescope object using information from known_telescopes.

        Parameters
        ----------
        name : str
            Name of the telescope.

        Returns
        -------
        Telescope
            A new Telescope object populated with information from
            known_telescopes.

        """
        tel_obj = cls()
        tel_obj.name = name
        tel_obj.update_params_from_known_telescopes(
            warn=False, known_telescope_dict=known_telescope_dict
        )
        return tel_obj
