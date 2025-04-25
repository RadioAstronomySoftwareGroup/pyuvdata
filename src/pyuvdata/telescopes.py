# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Telescope information and known telescope list."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from astropy import units
from astropy.coordinates import Angle, EarthLocation

from . import parameter as uvp, utils
from .data import DATA_PATH
from .utils.io import antpos, hdf5 as hdf5_utils
from .utils.tools import _strict_raise, slicify
from .uvbase import UVBase

__all__ = ["Telescope", "known_telescopes", "known_telescope_location"]

# We use astropy sites for telescope locations. The dict below is for
# telescopes not in astropy sites, or to include extra information for a telescope.

# The center_xyz is the location of the telescope in ITRF (earth-centered frame)

# Antenna positions can be specified via a csv file with the following columns:
# "name" -- antenna name, "number" -- antenna number, "x", "y", "z" -- ECEF coordinates
# relative to the telescope location.
KNOWN_TELESCOPES = {
    "PAPER": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("-30d43m17.5s"), lon=Angle("21d25m41.9s"), height=1073.0 * units.m
        ),
        "citation": (
            "value taken from capo/cals/hsa7458_v000.py, "
            "comment reads KAT/SA  (GPS), altitude from elevationmap.net"
        ),
        "mount_type": "fixed",
        "feed_array": ["x", "y"],
        "feed_angle": [np.pi / 2, 0.0],
    },
    "HERA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("-30.72152612068925d"),
            lon=Angle("21.42830382686301d"),
            height=1051.69 * units.m,
        ),
        "antenna_diameters": 14.0,
        "antenna_positions_file": "hera_ant_pos.csv",
        "mount_type": "fixed",
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
        "Nants": 8,
        "antenna_diameters": 6.0,
        "mount_type": "alt-az+nasmyth-l",
        "citation": "Ho, P. T. P., Moran, J. M., & Lo, K. Y. 2004, ApJL, 616, L1",
    },
    "SZA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("37d16m49.3698s"),
            lon=Angle("-118d08m29.9126s"),
            height=2400.0 * units.m,
        ),
        "Nants": 8,
        "antenna_diameters": 3.5,
        "mount_type": "alt-az+nasmyth-l",
        "feed_array": ["x", "y"],
        "feed_angle": [np.pi / 2, 0.0],
        "citation": "Unknown",
    },
    "OVRO-LWA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("37.239777271d"),
            lon=Angle("-118.281666695d"),
            height=1183.48 * units.m,
        ),
        "mount_type": "fixed",
        "feed_array": ["x", "y"],
        "feed_angle": [np.pi / 2, 0.0],
        "citation": "OVRO Sharepoint Documentation",
    },
    "MWA": {
        "antenna_positions_file": "mwa_ant_pos.csv",
        "mount_type": "phased",
        "feed_array": ["x", "y"],
        "feed_angle": [np.pi / 2, 0.0],
    },
    "ATA": {
        "location": EarthLocation.from_geodetic(
            lat=Angle("40d49m02.75s"),
            lon=Angle("-121d28m14.65s"),
            height=1019.222 * units.m,
        ),
        "antenna_diameters": 6.1,
        "mount_type": "alt-az",
        "feed_array": ["x", "y"],
        "feed_angle": [np.pi / 2, 0.0],
        "citation": "Private communication (D. DeBoer to G. Keating; 2024)",
    },
}


# Define a (private) dictionary that tracks whether the user wants warnings
# to be raised on updating known telescopes from params.
_WARN_STATUS = {k.lower(): True for k in KNOWN_TELESCOPES}


def ignore_telescope_param_update_warnings_for(tel: str):
    """Globally ignore update warnings for a given telescope.

    This affects the :meth:`Telescope.update_params_from_known_telescopes` method,
    which updates unspecified telescope information with known information from
    a KNOWN_TELESCOPE. In some cases, you will know that many files have
    unspecified information and that it is OK to supply this information from the
    known info in pyuvdata. Ignoring warnings can be achieved by setting `warn=False`
    in that method, but this is sometimes difficult because it is called further
    up in the stack. This simple convenience method allows all such warnings for a
    given telescope to be ignored.
    """
    if tel.lower() not in _WARN_STATUS:
        raise ValueError(f"'{tel}' is not a known telescope")
    _WARN_STATUS[tel.lower()] = False


def unignore_telescope_param_update_warnings_for(tel: str):
    """Globally un-ignore update warnings for a given telescope.

    See :func:`ignore_telescope_param_update_warnings`
    """
    if tel not in _WARN_STATUS:
        raise ValueError(f"'{tel}' is not a known telescope")
    _WARN_STATUS[tel] = True


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


def known_telescope_location(name: str, return_citation: bool = False, **kwargs):
    """
    Get the location for a known telescope.

    Parameters
    ----------
    name : str
        Name of the telescope
    return_citation : bool
        Option to return the citation.
    location : EarthLocation or MoonLocation
        Location of the telescope, can be used to overwrite or supplement information in
        the KNOWN_TELESCOPES dictionary.
    citation : str
        Source of location information, can be used to overwrite or supplement
        information in the KNOWN_TELESCOPES dictionary.

    Returns
    -------
    location : EarthLocation
        Telescope location as an EarthLocation object.
    citation : str, optional
        Citation string.

    """
    # first deal with location.
    if name in EarthLocation.get_site_names():
        location = EarthLocation.of_site(name)
        citation = "astropy sites"
    else:
        telescope_dict = {}
        tel_name = name.lower()
        for key in KNOWN_TELESCOPES:
            if key.lower() == tel_name:
                telescope_dict = KNOWN_TELESCOPES[key]
        for key in kwargs:
            if (key == "citation") or (key == "location"):
                telescope_dict[key] = kwargs[key]

        if telescope_dict:
            citation = telescope_dict.get("citation")
            location = telescope_dict.get("location")
            if location is None:
                raise KeyError(
                    "Missing location information in known_telescopes_dict "
                    f"for telescope {name}."
                )
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
            "telescope location: Either an astropy.EarthLocation object or a "
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

        desc = (
            "Antenna mount type, which describes the optics of the antenna in question."
            'Supported options include: "alt-az" (primary rotates in azimuth and'
            'elevation), "equatorial" (primary rotates in hour angle and declination), '
            '"orbiting" (antenna is in motion, and its orientation depends on orbital'
            'parameters), "x-y" (primary rotates first in the plane connecting east, '
            "west, and zenith, and then perpendicular to that plane), "
            '"alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary '
            'mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree '
            'tertiary mirror), "phased" (antenna is "electronically steered" by '
            'summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna '
            'beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other" '
            '(also referred to in some formats as "bizarre"). See the "Conventions" '
            "page of the documentation for further details. shape (Nants,)."
        )
        self._mount_type = uvp.UVParameter(
            "mount_type",
            description=desc,
            form=("Nants",),
            required=False,
            expected_type=str,
            acceptable_vals=[
                "alt-az",
                "equatorial",
                "orbiting",
                "x-y",
                "alt-az+nasmyth-r",
                "alt-az+nasmyth-l",
                "phased",
                "fixed",
                "other",
            ],
        )

        self._Nfeeds = uvp.UVParameter(
            "Nfeeds",
            description="Number of feeds.",
            expected_type=int,
            acceptable_vals=[1, 2],
            required=False,
        )

        desc = (
            "Array of feed orientations. shape (Nants, Nfeeds), type str. Options are: "
            "x/y or r/l. Optional parameter, required if feed_angle is set."
        )
        self._feed_array = uvp.UVParameter(
            "feed_array",
            description=desc,
            required=False,
            expected_type=str,
            form=("Nants", "Nfeeds"),
            acceptable_vals=["x", "y", "r", "l"],
        )

        desc = (
            "Position angle of a given feed, shape (Nants, Nfeeds), units of radians. "
            "A feed angle of 0 is typically oriented toward zenith for steerable "
            "antennas, otherwise toward north for fixed antennas (e.g., HERA, LWA)."
            'More details on this can be found on the "Conventions" page of the docs.'
            "Optional parameter, required if feed_array is set."
        )
        self._feed_angle = uvp.UVParameter(
            "feed_angle",
            description=desc,
            form=("Nants", "Nfeeds"),
            required=False,
            expected_type=float,
            tols=1e-6,  # 10x (~2 pi) single precision limit
        )

        super().__init__()

    def __getattr__(self, __name):
        """Handle old names attributes."""
        if __name == "x_orientation":
            warnings.warn(
                "The Telescope.x_orientation attribute is deprecated, and has "
                "been superseded by Telescope.feed_angle and Telescope.feed_array. "
                "This will become an error in version 3.4. To set the equivalent "
                "value in the future, you can substitute accessing this parameter "
                "with a call to Telescope.get_x_orientation_from_feeds().",
                DeprecationWarning,
            )
            return self.get_x_orientation_from_feeds()

        return super().__getattribute__(__name)

    def __setattr__(self, __name, __value):
        """Handle old names for telescope metadata."""
        if __name == "x_orientation":
            warnings.warn(
                "The Telescope.x_orientation attribute is deprecated, and has "
                "been superseded by Telescope.feed_angle and Telescope.feed_array. "
                "This will become an error in version 3.4. To get the equivalent "
                "value in the future, you can substitute accessing this parameter "
                "with a call to Telescope.set_feeds_from_x_orientation().",
                DeprecationWarning,
            )
            if __value is not None:
                self.set_feeds_from_x_orientation(__value)
        else:
            return super().__setattr__(__name, __value)

    def get_x_orientation_from_feeds(self) -> Literal["east", "north", None]:
        """
        Get x-orientation equivalent value based on feed information.

        Returns
        -------
        x_orientation : str
            One of "east", "north", or None, based on values present in
            Telescope.feed_array and Telescope.feed_angle.
        """
        return utils.pol.get_x_orientation_from_feeds(
            feed_array=self.feed_array,
            feed_angle=self.feed_angle,
            tols=self._feed_angle.tols,
        )

    def set_feeds_from_x_orientation(
        self,
        x_orientation,
        feeds=None,
        polarization_array=None,
        flex_polarization_array=None,
    ):
        """
        Set feed information based on x-orientation value.

        Populates newer parameters describing feed-orientation (`Telescope.feed_array`
        and `Telescope.feed_angle`) based on the "older" x-orientation string. Note that
        this method will overwrite any previously populated values, and if x-orientation
        is set to None, feed information will be removed from the object.

        Parameters
        ----------
        x_orientation : str
            String describing how the x-orientation is oriented. Must be either "north"/
            "n"/"ns" (x-polarization of antenna has a position angle of 0 degrees with
            respect to zenith/north) or "east"/"e"/"ew" (x-polarization of antenna has a
            position angle of 90 degrees with respect to zenith/north).
        feeds : list of str or None
            List of strings denoting feed orientations/polarizations. Must be one of
            "x", "y", "l", "r" (the former two for linearly polarized feeds, the latter
            for circularly polarized feeds). Default uses what is already present in
            feed_array if set, otherwise assumes a pair of linearly polarized feeds
            (["x", "y"]).
        polarization_array : array-like of int or None
            Array listing the polarization codes present, based on the UVFITS numbering
            scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
            polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
            feeds present if not supplied, ignored if flex_polarization_array is set
            to anything but None.
        flex_polarization_array : array-like of int or None
            Array listing the polarization codes present per spectral window (used with
            certain "flexible-polarization" objects), based on the UVFITS numbering
            scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
            polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
            feeds present if not supplied.
        """
        self.Nfeeds, self.feed_array, self.feed_angle = (
            utils.pol.get_feeds_from_x_orientation(
                x_orientation=x_orientation,
                feeds=feeds,
                feed_array=self.feed_array if feeds is None else None,
                polarization_array=polarization_array,
                flex_polarization_array=flex_polarization_array,
                nants=self.Nants,
            )
        )

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

        super().check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # If using feed_angle, make sure feed_array is set (and visa-versa)
        if (self.feed_array is not None or self.feed_angle is not None) and (
            self.feed_array is None
            or self.feed_angle is None
            or self.mount_type is None
        ):
            strict = (
                self._feed_array.required
                or self._feed_angle.required
                or self._mount_type.required
            )
            msg = (
                "Parameter feed_array, feed_angle, and mount_type must be set together."
            ) + ("" if strict else " Unsetting optional parameters.")
            _strict_raise(err_msg=msg, strict=strict)
            # If we got to this point, only a warning raised, so unset the offending
            # parameter(s) and move on along.
            self.feed_array = self.feed_angle = self.mount_type = None

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
        override_known_params: bool = True,
        feeds: str | list[str] | None = None,
        polarization_array: np.ndarray | None = None,
        flex_polarization_array: np.ndarray | None = None,
        **kwargs,
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
        override_known_params : bool
            Normally, when passing arguments for individual parameters for the Telescope
            object, they will be used over what is present in the KNOWN_TELESCOPES dict.
            However, if set to False, the information in the KNOWN_TELESCOPES dict is
            used if/when present, even if supplied as an argument when calling this
            method. Default is True.
        x_orientation : str
            String describing how the x-orientation is oriented. Must be either "north"/
            "n"/"ns" (x-polarization of antenna has a position angle of 0 degrees with
            respect to zenith/north) or "east"/"e"/"ew" (x-polarization of antenna has a
            position angle of 90 degrees with respect to zenith/north). Ignored if
            "x_orientation" is relevant entry for the KNOWN_TELESCOPES dict, or if
            feed_array and feed_angle supplied (either by passing an argument or in the
            KNOWN_TELESCOPES dict).
        feeds : list of str or None
            List of strings denoting feed orientations/polarizations. Must be one of
            "x", "y", "l", "r" (the former two for linearly polarized feeds, the latter
            for circularly polarized feeds). Default assumes a pair of linearly
            polarized feeds (["x", "y"]).
        polarization_array : array-like of int or None
            Array listing the polarization codes present, based on the UVFITS numbering
            scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
            polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
            feeds present if not supplied, ignored if flex_polarization_array is set
            to anything but None.
        flex_polarization_array : array-like of int or None
            Array listing the polarization codes present per spectral window (used with
            certain "flexible-polarization" objects), based on the UVFITS numbering
            scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
            polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
            feeds present if not supplied.
        location : EarthLocation or MoonLocation
            Location of the telescope (phase reference position).
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        citation : str
            Source of telescope information.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_positions_file : str
            Path to a csv-formatted file containing a total of five columns, which
            describe (on a per-antenna basis) the name, number, and x/y/z location
            (relative to the phase center, units of meters) of each antenna.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        Nants : int
            Number of antennas present in the telescope.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_numbers : array-like of int
            Antenna numbers that correspond to entries in antenna_names, array-like
            of shape (Nants,), dtype int.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_names : array-like of str
            Names of the antennas present in the telescope object, array-like of shape
            (Nants,), dtype str.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_positions : array of float
            Coordinates of antennas relative to the telescope location, shape
            (Nants, 3), dtype float, units meters.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        mount_type : str or array-like of str
            Antenna mount type, which describes the optics for each antenna. Can either
            be supplied on a per-antenna basis, in which case an array of shape (Nants,)
            with dtype str is expected, or if all antennas share the same optics type, a
            single string can be provided (see supported types in the documentation for
            `Telescope.mount_type`).
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        feed_array : array-like of str
            Feed types present within each telescope in terms of polarization response,
            must be one of "r", "l", "x", or "y". Can be supplied on a per-antenna
            basis, in which case an array of shape (Nants, Nfeeds) is expected, or if
            the same for all antennas, an array of shape (Nfeeds,) can be provided (
            note that Nfeeds cannot be greater than 2).
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        feed_angle : array-like of float
            Relative receptor angle for each feed (see more information about his in the
            documentation for `Telescope.feed_angle`). Can be supplied on a per-antenna
            basis, in which case an array of shape (Nants, Nfeeds) is expected, or if
            the same for all antennas, an array of shape (Nfeeds,) can be provided (
            note that Nfeeds cannot be greater than 2). Expected dtype is float.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_diameters : float or array-like of float
            Diameter of the antennas within the telescope, dtype float, units of meters.
            Can be supplied on a per-antenna basis, in which case an array of shape
            (Nants,) is expected, or if the same for all antennas, a single float can
            be provided.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.

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
        telescope_dict = {}
        tel_name = self.name.lower()
        for key in KNOWN_TELESCOPES:
            if key.lower() == tel_name:
                telescope_dict.update(KNOWN_TELESCOPES[key])

        astropy_sites_list = []
        known_telescope_list = []
        direct_set_list = []
        for key in kwargs:
            if override_known_params or key not in telescope_dict:
                value = kwargs[key]
                telescope_dict[key] = value
                if value is None:
                    del telescope_dict[key]
                else:
                    direct_set_list.append(key)

        # first deal with location.
        if overwrite or self.location is None:
            self.location, self.citation = known_telescope_location(
                self.name, return_citation=True, **kwargs
            )
            if self.citation is not None and "astropy sites" in self.citation:
                astropy_sites_list.append("telescope_location")
            elif self.location is not None:
                known_telescope_list.append("telescope_location")

        # check for extra info
        if telescope_dict:
            x_orientation = telescope_dict.get("x_orientation")

            if "antenna_positions_file" in telescope_dict and (
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
                for key in ant_info:
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
                    if self.antenna_numbers is not None and len(ant_inds) != self.Nants:
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
            else:
                for param in [
                    "antenna_positions",
                    "antenna_names",
                    "antenna_numbers",
                    "Nants",
                ]:
                    write_check = overwrite or getattr(self, param) is None
                    if param in telescope_dict and write_check:
                        setattr(self, param, telescope_dict[param])
                        known_telescope_list.append(param)
                        if "Nants" not in telescope_dict and param != "Nants":
                            telescope_dict["Nants"] = len(telescope_dict[param])

            if "mount_type" in telescope_dict and (
                overwrite or self.mount_type is None
            ):
                mount_type = telescope_dict["mount_type"]
                if isinstance(mount_type, str):
                    mount_type = [mount_type]

                if len(mount_type) == 1 and self.Nants is not None:
                    mount_type *= self.Nants

                if len(mount_type) == self.Nants:
                    known_telescope_list.append("mount_type")
                    self.mount_type = mount_type
                elif warn:
                    warnings.warn(
                        "mount_type is not set because the number of mount_type on "
                        "known_telescopes_dict is more than one and does not match "
                        f"Nants for telescope {self.name}."
                    )

            feed_params = ["feed_array", "feed_angle"]
            if x_orientation is not None and not overwrite:
                feed_params.remove("feed_angle")
                if not all(
                    item is None
                    for item in [polarization_array, flex_polarization_array, feeds]
                ):
                    feed_params.remove("feed_array")

            for param in feed_params:
                if param not in telescope_dict or not (
                    overwrite or getattr(self, param) is None
                ):
                    continue
                new_val = np.atleast_1d(telescope_dict[param])
                if new_val.ndim == 1 and self.Nants is not None:
                    new_val = np.tile(new_val, (self.Nants, 1))
                if new_val.shape[0] == self.Nants:
                    known_telescope_list.append(param)
                    setattr(self, param, new_val)
                    self.Nfeeds = new_val.shape[1]
                elif warn:
                    warnings.warn(
                        f"{param} is not set because the number of {param} on "
                        "known_telescopes_dict is more than one and does not match "
                        f"Nants for telescope {self.name}."
                    )

            if "antenna_diameters" in telescope_dict and (
                overwrite or self.antenna_diameters is None
            ):
                antenna_diameters = np.atleast_1d(telescope_dict["antenna_diameters"])
                if antenna_diameters.size == 1 and self.Nants is not None:
                    known_telescope_list.append("antenna_diameters")
                    self.antenna_diameters = np.full(
                        self.Nants, antenna_diameters[0], dtype=float
                    )
                elif antenna_diameters.size == self.Nants:
                    known_telescope_list.append("antenna_diameters")
                    self.antenna_diameters = antenna_diameters
                elif warn:
                    warnings.warn(
                        "antenna_diameters is not set because the number "
                        "of antenna_diameters on known_telescopes_dict is "
                        "more than one and does not match Nants for "
                        f"telescope {self.name}."
                    )

            # Set this separately, since if we've specified x-orientation we want to
            # propagate that information to the relevant parameters.
            if x_orientation is not None and (
                (
                    (overwrite or "Nants" in known_telescope_list)
                    and not (
                        "feed_array" in known_telescope_list
                        and "feed_angle" in known_telescope_list
                    )
                )
                or (self.feed_array is None or self.feed_angle is None)
            ):
                self.set_feeds_from_x_orientation(
                    x_orientation=x_orientation,
                    feeds=feeds,
                    polarization_array=polarization_array,
                    flex_polarization_array=flex_polarization_array,
                )

        # Don't warn on things we've set directly, because that seems overly verbose
        # and not actually helpful to tell the user that they've provided and argument
        for item in direct_set_list:
            if item in known_telescope_list:
                known_telescope_list.remove(item)

        full_list = astropy_sites_list + known_telescope_list
        if warn and _WARN_STATUS.get(self.name.lower(), True) and len(full_list) > 0:
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

        if "Nants" in known_telescope_list:
            for param_name in self:
                param = getattr(self, param_name)
                if (
                    param.value is None
                    or param.name in known_telescope_list
                    or param.name in direct_set_list
                    or not (isinstance(param.form, tuple) and "Nants" in param.form)
                ):
                    continue

                new_val = None
                if isinstance(param.value, list):
                    if all(item == param.value[0] for item in param.value):
                        new_val = [param.value[0]] * self.Nants
                else:
                    slice_grp = [slice(None)] * len(param.form)
                    tile_grp = [1] * len(param.form)
                    for idx, axis in enumerate(param.form):
                        if axis == "Nants":
                            slice_grp[idx] = slice(0, 1)
                            tile_grp[idx] = self.Nants
                    if np.all(param.value == param.value[tuple(slice_grp)]):
                        new_val = np.tile(param.value[tuple(slice_grp)], tile_grp)

                param.value = new_val
                param.setter(self)
                if warn and new_val is None:
                    warnings.warn(
                        "Nants has changed, but no information present in the known "
                        f"telescopes to set {param.name}, and entries of {param.name} "
                        "vary on a per-antenna basis -- setting to None."
                    )

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
        **kwargs,
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
        x_orientation : str
            String describing how the x-orientation is oriented. Must be either "north"/
            "n"/"ns" (x-polarization of antenna has a position angle of 0 degrees with
            respect to zenith/north) or "east"/"e"/"ew" (x-polarization of antenna has a
            position angle of 90 degrees with respect to zenith/north). Ignored if
            "x_orientation" is relevant entry for the KNOWN_TELESCOPES dict, or if
            feed_array and feed_angle supplied (either by passing an argument or in the
            KNOWN_TELESCOPES dict).
        feeds : list of str or None
            List of strings denoting feed orientations/polarizations. Must be one of
            "x", "y", "l", "r" (the former two for linearly polarized feeds, the latter
            for circularly polarized feeds). Default assumes a pair of linearly
            polarized feeds (["x", "y"]).
        polarization_array : array-like of int or None
            Array listing the polarization codes present, based on the UVFITS numbering
            scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
            polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
            feeds present if not supplied, ignored if flex_polarization_array is set
            to anything but None.
        flex_polarization_array : array-like of int or None
            Array listing the polarization codes present per spectral window (used with
            certain "flexible-polarization" objects), based on the UVFITS numbering
            scheme. See utils.POL_NUM2STR_DICT for a mapping between codes and
            polarization types. Used with `utils.pol.get_feeds_from_pols` to determine
            feeds present if not supplied.
        override_known_params : bool
            Normally, when passing arguments for individual parameters for the Telescope
            object, they will be used over what is present in the KNOWN_TELESCOPES dict.
            However, if set to False, the information in the KNOWN_TELESCOPES dict is
            used if/when present, even if supplied as an argument when calling this
            method.
        location : EarthLocation or MoonLocation
            Location of the telescope (phase reference position).
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        citation : str
            Source of telescope information.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_positions_file : str
            Path to a csv-formatted file containing a total of five columns, which
            describe (on a per-antenna basis) the name, number, and x/y/z location
            (relative to the phase center, units of meters) of each antenna.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        Nants : int
            Number of antennas present in the telescope.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_numbers : array-like of int
            Antenna numbers that correspond to entries in antenna_names, array-like
            of shape (Nants,), dtype int.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_names : array-like of str
            Names of the antennas present in the telescope object, array-like of shape
            (Nants,), dtype str.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_positions : array of float
            Coordinates of antennas relative to the telescope location, shape
            (Nants, 3), dtype float, units meters.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        mount_type : str or array-like of str
            Antenna mount type, which describes the optics for each antenna. Can either
            be supplied on a per-antenna basis, in which case an array of shape (Nants,)
            with dtype str is expected, or if all antennas share the same optics type, a
            single string can be provided (see supported types in the documentation for
            `Telescope.mount_type`).
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        feed_array : array-like of str
            Feed types present within each telescope in terms of polarization response,
            must be one of "r", "l", "x", or "y". Can be supplied on a per-antenna
            basis, in which case an array of shape (Nants, Nfeeds) is expected, or if
            the same for all antennas, an array of shape (Nfeeds,) can be provided (
            note that Nfeeds cannot be greater than 2).
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        feed_angle : array-like of float
            Relative receptor angle for each feed (see more information about his in the
            documentation for `Telescope.feed_angle`). Can be supplied on a per-antenna
            basis, in which case an array of shape (Nants, Nfeeds) is expected, or if
            the same for all antennas, an array of shape (Nfeeds,) can be provided (
            note that Nfeeds cannot be greater than 2). Expected dtype is float.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.
        antenna_diameters : float or array-like of float
            Diameter of the antennas within the telescope, dtype float, units of meters.
            Can be supplied on a per-antenna basis, in which case an array of shape
            (Nants,) is expected, or if the same for all antennas, a single float can
            be provided.
            Ignored if information is present in KNOWN_TELESCOPES dict and
            override_known_params=False.

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
            **kwargs,
        )
        return tel_obj

    @classmethod
    def new(
        cls,
        name: str,
        # do not type hint here because MoonLocations are allowed but we don't
        # want to import them just for this.
        location,
        antenna_positions: np.ndarray | dict[str | int, np.ndarray] | None = None,
        antenna_names: list[str] | np.ndarray | None = None,
        antenna_numbers: list[int] | np.ndarray | None = None,
        antname_format: str = "{0:03d}",
        instrument: str | None = None,
        x_orientation: Literal["east", "north", "e", "n", "ew", "ns"] | None = None,
        antenna_diameters: list[float] | np.ndarray | None = None,
        feeds: Literal["x", "y", "l", "r"] | list[str] | None = None,
        feed_array: list[str] | np.ndarray | None = None,
        feed_angle: list[float] | np.ndarray | None = None,
        mount_type: Literal[
            "alt-az",
            "equatorial",
            "orbiting",
            "x-y",
            "alt-az+nasmyth-r",
            "alt-az+nasmyth-l",
            "phased",
            "fixed",
            "other",
        ]
        | list[str]
        | None = None,
        update_from_known: bool = True,
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
            'ew', 'ns'. Ignored if feed_array and feed_angle are provided.
        antenna_diameters :  list or np.ndarray of float, optional
            List or array of antenna diameters.
        feeds : list of str or None:
            List of feeds present in the Telescope, which must be one of "x", "y", "l",
            "r". Length of the list must be either 1 or 2. Used to populate feed_array
            and feed_angle parameters if only supplying x_orientation, default is
            ["x", "y"].
        feed_array : array-like of str or None
            List of feeds for each antenna in the Telescope object, must be one of
            "x", "y", "l", "r". Shape (Nants, Nfeeds), dtype str. Can also be shape
            (Nfeeds,), in which case the same values for feed_array are used for
            all antennas in the object.
        feed_angle : array-like of float or None
            Orientation of the feed with respect to zenith (or with respect to north if
            pointed at zenith). Units is in rads, x-polarization is nominally pi / 2,
            and x-polarization (as well as r- and l-polarizations) is nominally pi / 2.
            Shape (Nants, Nfeeds), dtype float.  Can also be shape (Nfeeds,), in which
            case the same values for feed_angle are used for all antennas in the object.
        mount_type : str or array-like of str
            Antenna mount type, which describes the optics of the antenna in question.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details. Shape (Nants,), dtype str.
            Can also provide a single string, in which case the same mount_type is
            used for all antennas in the object.
        update_from_known : bool
            If set to True and the telescope name is found in the KNOWN_TELESCOPES
            dictionary, then any missing parameters will be filled in using the
            information from said dictionary if available. Otherwise if set to False,
            any missing parameters are left set to None. Default is True.

        Returns
        -------
        Telescope object
            A Telescope object with the specified metadata.

        """
        tel_obj = cls()
        _ = utils.coordinates.get_frame_ellipsoid_loc_obj(
            location, "telescope_location"
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

        if feed_angle is not None and feed_array is not None:
            tel_obj.feed_array = np.asarray(feed_array)
            tel_obj.feed_angle = np.asarray(feed_angle)
            if (tel_obj.feed_array.ndim == 1) and (tel_obj.feed_angle.ndim == 1):
                tel_obj.feed_array = np.tile(tel_obj.feed_array, (tel_obj.Nants, 1))
                tel_obj.feed_angle = np.tile(tel_obj.feed_angle, (tel_obj.Nants, 1))

            tel_obj.Nfeeds = tel_obj.feed_angle.shape[1]
        elif x_orientation is not None:
            tel_obj.set_feeds_from_x_orientation(x_orientation.lower(), feeds=feeds)

        if mount_type is not None:
            if isinstance(mount_type, str):
                tel_obj.mount_type = [mount_type] * tel_obj.Nants
            else:
                tel_obj.mount_type = mount_type

        if antenna_diameters is not None:
            tel_obj.antenna_diameters = np.asarray(antenna_diameters)

        if update_from_known:
            tel_obj.update_params_from_known_telescopes()

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
            meta = filename  # no copy required because its read-only

        tel_obj.location = meta.telescope_location_obj

        telescope_attrs = {
            "telescope_name": "name",
            "Nants_telescope": "Nants",
            "antenna_names": "antenna_names",
            "antenna_numbers": "antenna_numbers",
            "antenna_positions": "antenna_positions",
            "instrument": "instrument",
            "antenna_diameters": "antenna_diameters",
            "mount_type": "mount_type",
            "Nfeeds": "Nfeeds",
            "feed_array": "feed_array",
            "feed_angle": "feed_angle",
        }
        for attr, tel_attr in telescope_attrs.items():
            try:
                setattr(tel_obj, tel_attr, getattr(meta, attr))
            except (AttributeError, KeyError) as e:
                if attr in required_keys:
                    raise KeyError(str(e)) from e
                else:
                    pass

        # Handle the retired x-orientation parameter
        if (tel_obj.feed_array is None) or (tel_obj.feed_angle is None):
            tel_obj.set_feeds_from_x_orientation(meta.x_orientation, feeds=["x", "y"])

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
        if self.antenna_diameters is not None:
            header["antenna_diameters"] = self.antenna_diameters
        if self.Nfeeds is not None:
            header["Nfeeds"] = self.Nfeeds
        if self.feed_array is not None:
            header["feed_array"] = np.asarray(self.feed_array, dtype="bytes")
        if self.feed_angle is not None:
            header["feed_angle"] = self.feed_angle
        if self.mount_type is not None:
            header["mount_type"] = np.asarray(self.mount_type, dtype="bytes")

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

    def reorder_feeds(
        self,
        order="AIPS",
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Arrange feed axis according to desired order.

        Parameters
        ----------
        order : str
            Either a string specifying a canonical ordering ('AIPS' or 'CASA')
            or list of strings specifying the preferred ordering of the four
            feed types ("x", "y", "l", and "r").
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reordering.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reordering.

        Raises
        ------
        ValueError
            If the order is not one of the allowed values.

        """
        if self.Nfeeds is None or self.Nfeeds == 1:
            # Nothing to do but bail!
            return

        if (order == "AIPS") or (order == "CASA"):
            order = {"x": 1, "y": 2, "r": 3, "l": 4}
        elif isinstance(order, list) and all(f in ["x", "y", "l", "r"] for f in order):
            order = {item: idx for idx, item in enumerate(order)}
        else:
            raise ValueError(
                "order must be one of: 'AIPS', 'CASA', or a "
                'list of length 4 containing only "x", "y", "r", or "l".'
            )

        for idx in range(self.Nants):
            feed_a, feed_b = self.feed_array[idx]
            if order.get(feed_a, 999999) > order.get(feed_b, 999999):
                self.feed_array[idx] = self.feed_array[idx, ::-1]
                self.feed_angle[idx] = self.feed_angle[idx, ::-1]

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def reorder_antennas(
        self,
        order="number",
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Arrange the antenna axis according to desired order.

        Parameters
        ----------
        order: str or array like of int
            If a string, allowed values are "name" and "number" to sort on the antenna
            name or number respectively. A '-' can be prepended to signify descending
            order instead of the default ascending order (e.g. "-number"). An array of
            integers of length Nants representing indexes along the existing `ant_array`
            can also be supplied to sort in any desired order (note these are indices
            into the `ant_array` not antenna numbers).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if order is not an allowed string or is an array that does not
            contain all the required numbers.

        """
        if isinstance(order, np.ndarray | list | tuple):
            if not np.array_equal(np.sort(order), np.arange(self.Nants)):
                raise ValueError(
                    "If order is an index array, it must contain all indices for the"
                    "ant_array, without duplicates."
                )
            index_array = order
        else:
            if order not in ["number", "name", "-number", "-name"]:
                raise ValueError(
                    "order must be one of 'number', 'name', '-number', '-name' or an "
                    "index array of length Nants_data"
                )

            index_array = np.argsort(
                self.antenna_names if "name" in order else self.antenna_numbers
            )

            if order[0] == "-":
                index_array = np.flip(index_array)

        self._select_along_param_axis({"Nants": index_array})

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def __add__(
        self,
        other: Telescope,
        *,
        warning_params: list | None = None,
        inplace: bool = False,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
    ):
        """
        Combine two Telescope objects along antennas or feeds.

        Parameters
        ----------
        other : Telescope object
            Another Telescope object which will be added to self.
        warning_params : list of str
            By default, when parameters are from the two objects are incompatible, an
            error is thrown if the parameter is required, otherwise the parameter on
            the returned object is set to None and a warning is raised. However, one can
            pass a list of strings to explicitly specify which parameters to allow to
            raise a warning versus an error.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.

        Raises
        ------
        ValueError
            If other is not a Telescope object, self and other are not compatible.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that both objects are Telescope and valid
        this.check(check_extra=check_extra, run_check_acceptability=False)
        if not issubclass(other.__class__, this.__class__) and not issubclass(
            this.__class__, other.__class__
        ):
            raise ValueError(
                "Only Telescope (or subclass) objects can be added "
                "to a Telescope (or subclass) object"
            )
        other.check(check_extra=check_extra, run_check_acceptability=False)

        if warning_params is None:
            warning_params = []

        for param in this:
            this_param = getattr(this, param)
            other_param = getattr(other, param)
            if not (this_param.required or other_param.required):
                warning_params.append(this_param.name)

        # Begin doing some addition magic
        axis_list = [("Nants", "_antenna_numbers"), ("Nfeeds", "_feed_array")]
        this_overlap = {}
        other_overlap = {}

        tget_map = {}
        tset_map = {}
        oget_map = {}
        oset_map = {}
        aget_map = {}
        aset_map = {}

        nind_dict = {}
        excepted_list = []
        diff_axis = []
        for axis_name, axis_attr in axis_list:
            this_param = getattr(this, axis_attr)
            other_param = getattr(other, axis_attr)
            this_val = this_param.value
            other_val = other_param.value
            n_this = getattr(this, axis_name)
            n_other = getattr(this, axis_name)
            if (
                this_val is not None
                and other_val is not None
                and (this_param != other_param)
            ):
                excepted_list.append(axis_name)
                if axis_attr == "_feed_array":
                    # Invoke some special handling here for feed_array, since it's
                    # Nants by Nfeeds, and we're only trying to evaluate the latter
                    if any(np.isin(this.antenna_numbers, other.antenna_numbers)):
                        # We have some overlap, so establish keys based on matching
                        # values where we have them (and otherwise ignore). Note
                        # This will create an array of strings of length Nfeeds.
                        t_ants = this.antenna_numbers
                        o_ants = other.antenna_numbers

                        ind = np.argsort(t_ants)
                        this_val = this_val[ind[np.isin(t_ants, o_ants)[ind]], :]
                        this_val = np.array(["".join(item) for item in this_val.T])

                        ind = np.argsort(o_ants)
                        other_val = other_val[ind[np.isin(o_ants, t_ants)[ind]], :]
                        other_val = np.array(["".join(item) for item in other_val.T])

                        if np.array_equal(this_val, other_val):
                            # If everything matches after antenna down-selection, then
                            # there's no extra checking to do on this axis.
                            continue
                    else:
                        # Otherwise if no antenna overlap is present but the feeds
                        # don't match, see if every entry is identical
                        this_val = np.unique(this_val, axis=0)
                        other_val = np.unique(other_val, axis=0)
                        if (len(this_val) == 1) and (len(other_val) == 1):
                            # If every entry is the same, then we can index effectively
                            # using the first entry.
                            this_val = this_val[0]
                            other_val = other_val[0]
                        else:
                            # At this point, throw out hands up -- just map current
                            # index positions to the new positions.
                            this_val = np.arange(this.Nfeeds)
                            other_val = np.arange(other.Nfeeds)

                # Figure out first which indices contain overlap, and make sure they
                # are ordered correctly so that we can compare apples-to-apples.
                # Note if there is no overlap, this will spit out trivial slices
                # that will produce arrays of zero-length along this relevant axis.
                _, this_olap_ind, other_olap_ind = np.intersect1d(
                    this_val, other_val, return_indices=True
                )
                this_overlap[axis_name] = slicify(this_olap_ind, allow_empty=True)
                other_overlap[axis_name] = slicify(other_olap_ind, allow_empty=True)

                # Next, figure out how these things plug in to the "big" array, using
                # unique (which automatically sorts the output).
                ind_order = {key: None for arr in [this_val, other_val] for key in arr}
                # TODO: Can insert ordering here if desired as a future feature
                ind_order = {key: idx for idx, key in enumerate(ind_order)}

                # Record the length of the new axis.
                nind_dict[axis_name] = len(ind_order)

                # Figure out how indices map from other array to new array.
                this_put = [ind_order[key] for key in this_val]
                other_put = [ind_order[key] for key in other_val]

                # Create some slices, ordering the arrays accordingly.
                tget_map[axis_name] = slice(0, n_this, 1)
                tset_map[axis_name] = slicify(this_put, allow_empty=True)
                oget_map[axis_name] = slice(0, n_other, 1)
                oset_map[axis_name] = slicify(other_put, allow_empty=True)

                if tset_map[axis_name] != oset_map[axis_name]:
                    diff_axis.append(axis_name)

                if len(this_olap_ind) or len(other_olap_ind):
                    # If there is overlap, we can speed up processing if we don't need
                    # to copy the overlapping bits (provided they match). Use that to
                    # construct an alternate version of the indexing.
                    mask = np.ones_like(other_put, dtype=bool)
                    mask[other_olap_ind] = False
                    alt_put = np.asarray(other_put)[mask].tolist()
                    aget_map[axis_name] = slicify(np.nonzero(mask)[0], allow_empty=True)
                    aset_map[axis_name] = slicify(alt_put, allow_empty=True)

        # Now go through and verify that parameters match where we need them
        for param in this:
            this_param = getattr(this, param)
            other_param = getattr(other, param)
            if this_param.name in excepted_list:
                continue
            elif not (this_param.value is None or other_param.value is None) and (
                isinstance(this_param.form, tuple) and this_param.form != ()
            ):
                # If we have a tuple, that means we have a multi-dim array/list
                # that we need to handle appropriately.
                this_value = this_param.get_from_form(this_overlap)
                other_value = other_param.get_from_form(other_overlap)
                atol, rtol = this_param.tols

                # Use lazy comparison to do direct comparison first, then fall back
                # to isclose if not comparing strings to see if that passes.
                if np.array_equal(this_value, other_value) or (
                    (not issubclass(str, this_param.expected_type))
                    and np.allclose(this_value, other_value, rtol=rtol, atol=atol)
                ):
                    continue
            elif this_param == other_param:
                # If not a tuple, just let UVParameter.__eq__ handle it
                continue

            # If we got here, then no match was achieved. Time to successfully fail!
            strict = this_param.name not in warning_params
            err_msg = f"UVParameter Telescope.{this_param.name} does not match." + (
                " Continuing anyways." if not strict else ""
            )
            utils.tools._strict_raise(err_msg=err_msg, strict=strict)

        # We've checked everything, time to start the merge
        for param in this:
            # Grab the params to make life easier
            this_param = getattr(this, param)
            other_param = getattr(other, param)
            if this_param.name in nind_dict:
                # If one of the index lengths, grab that here and now.
                this_param.value = nind_dict[this_param.name]
                this_param.setter(this)
            elif isinstance(this_param.form, tuple) and any(
                key in nind_dict for key in this_param.form
            ):
                if (this_param.value is None) or other_param.value is None:
                    # This is an attribute that _should_ be combined, but one or the
                    # other values is set to None, and therefore we are just unsetting
                    # the whole thing.
                    this_param.value = None
                    this_param.setter(this)
                    continue

                # Only need to do the combine if there are multiple elements to worry
                # about, otherwise we've checked/forced compatibility
                temp_val = this_param.get_from_form(tget_map)
                old_shape = (
                    (len(temp_val),) if isinstance(temp_val, list) else temp_val.shape
                )
                new_shape = tuple(
                    nind_dict.get(key, old_shape[idx])
                    for idx, key in enumerate(this_param.form)
                )
                if old_shape == new_shape:
                    # temp_val contains all we need, nothing further to do here, just
                    # assign the value and continue
                    this_param.value = temp_val
                    this_param.setter(this)
                    continue

                # Otherwise if the shapes don't match, make a larger array
                # that we can plug values into
                this_param.value = (
                    [None] * new_shape[0]
                    if isinstance(temp_val, list)
                    else np.zeros_like(temp_val, shape=new_shape)
                )
                this_param.set_from_form(tset_map, temp_val)

                # Now update based on other vals
                alt_map = [item for item in diff_axis if item in this_param.form]
                if (len(alt_map) == 1) and alt_map[0] in aget_map:
                    # We only need to do this if there is _exactly_ one overlapping
                    # axis, but otherwise the axes agree. Do the copies to avoid
                    # updating the original dict
                    temp_get_map = oget_map.copy()
                    temp_set_map = oset_map.copy()
                    temp_get_map[alt_map[0]] = aget_map[alt_map[0]]
                    temp_set_map[alt_map[0]] = aset_map[alt_map[0]]
                else:
                    temp_get_map = oget_map
                    temp_set_map = oset_map

                temp_val = other_param.get_from_form(temp_get_map)
                this_param.set_from_form(temp_set_map, temp_val)
            elif this_param.value is None:
                # If this is None other is not, carry over the value from other
                # Note we're only working w/ optional values at this point.
                this_param.value = other_param.value
                this_param.setter(this)

        # Final check
        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return this

    def __iadd__(
        self,
        other,
        *,
        warning_params=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        In place add.

        Parameters
        ----------
        other : Telescope object
            Another Telescope object which will be added to self.
        warning_params : list of str
            By default, when parameters are from the two objects are incompatible, an
            error is thrown if the parameter is required, otherwise the parameter on
            the returned object is set to None and a warning is raised. However, one can
            pass a list of strings to explicitly specify which parameters to allow to
            raise a warning versus an error.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.

        Raises
        ------
        ValueError
            If other is not a Telescope object, self and other are not compatible.

        """
        self.__add__(
            other,
            warning_params=warning_params,
            inplace=True,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )
        return self
