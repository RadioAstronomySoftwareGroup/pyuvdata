# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with phase center catalogs."""

import numpy as np
from astropy.time import Time

from . import RADIAN_TOL

allowed_cat_types = ["sidereal", "ephem", "unprojected", "driftscan", "near_field"]


def look_in_catalog(
    phase_center_catalog,
    *,
    cat_name=None,
    cat_type=None,
    cat_lon=None,
    cat_lat=None,
    cat_frame=None,
    cat_epoch=None,
    cat_times=None,
    cat_pm_ra=None,
    cat_pm_dec=None,
    cat_dist=None,
    cat_vrad=None,
    ignore_name=False,
    target_cat_id=None,
    phase_dict=None,
):
    """
    Check the catalog to see if an existing entry matches provided data.

    This is a helper function for verifying if an entry already exists within
    the catalog, contained within the supplied phase center catalog.

    Parameters
    ----------
    phase_center_catalog : dict
        Dictionary containing the entries to check.
    cat_name : str
        Name of the phase center, which should match a the value of "cat_name"
        inside an entry of `phase_center_catalog`.
    cat_type : str
        Type of phase center of the entry. Must be one of:
            "sidereal" (fixed RA/Dec),
            "ephem" (RA/Dec that moves with time),
            "driftscan" (fixed az/el position),
            "unprojected" (no w-projection, equivalent to the old
            `phase_type` == "drift").
    cat_lon : float or ndarray
        Value of the longitudinal coordinate (e.g., RA, Az, l) in radians of the
        phase center. No default unless `cat_type="unprojected"`, in which case the
        default is zero. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    cat_lat : float or ndarray
        Value of the latitudinal coordinate (e.g., Dec, El, b) in radians of the
        phase center. No default unless `cat_type="unprojected"`, in which case the
        default is pi/2. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    cat_frame : str
        Coordinate frame that cat_lon and cat_lat are given in. Only used for
        sidereal and ephem phase centers. Can be any of the several supported frames
        in astropy (a limited list: fk4, fk5, icrs, gcrs, cirs, galactic).
    cat_epoch : str or float
        Epoch of the coordinates, only used when cat_frame = fk4 or fk5. Given
        in units of fractional years, either as a float or as a string with the
        epoch abbreviation (e.g, Julian epoch 2000.0 would be J2000.0).
    cat_times : ndarray of floats
        Only used when `cat_type="ephem"`. Describes the time for which the values
        of `cat_lon` and `cat_lat` are caclulated, in units of JD. Shape is (Npts,).
    cat_pm_ra : float
        Proper motion in RA, in units of mas/year. Only used for sidereal phase
        centers.
    cat_pm_dec : float
        Proper motion in Dec, in units of mas/year. Only used for sidereal phase
        centers.
    cat_dist : float or ndarray of float
        Distance of the source, in units of pc. Only used for sidereal and ephem
        phase centers. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    cat_vrad : float or ndarray of float
        Radial velocity of the source, in units of km/s. Only used for sidereal and
        ephem phase centers. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    ignore_name : bool
        Nominally, this method will only look at entries where `cat_name`
        matches the name of an entry in the catalog. However, by setting this to
        True, the method will search all entries in the catalog and see if any
        match all of the provided data (excluding `cat_name`).
    target_cat_id : int
        Optional argument to specify a particular cat_id to check against.
    phase_dict : dict
        Instead of providing individual parameters, one may provide a dict which
        matches that format used within `phase_center_catalog` for checking for
        existing entries. If used, all other parameters (save for `ignore_name`
        and `cat_name`) are disregarded.

    Returns
    -------
    cat_id : int or None
        The unique ID number for the phase center added to the internal catalog.
        This value is used in the `phase_center_id_array` attribute to denote which
        source a given baseline-time corresponds to. If no catalog entry matches,
        then None is returned.
    cat_diffs : int
        The number of differences between the information provided and the catalog
        entry contained within `phase_center_catalog`. If everything matches, then
        `cat_diffs=0`.
    """
    # 1 marcsec tols
    radian_tols = (0, RADIAN_TOL)
    default_tols = (1e-5, 1e-8)
    match_id = None
    match_diffs = 99999

    if (cat_name is None) and (not ignore_name):
        if phase_dict is None:
            raise ValueError(
                "Must specify either phase_dict or cat_name if ignore_name=False."
            )
        cat_name = phase_dict["cat_name"]

    if cat_type is not None and cat_type not in allowed_cat_types:
        raise ValueError(f"If set, cat_type must be one of {allowed_cat_types}")

    # Emulate the defaults that are set if None is detected for
    # unprojected and driftscan types.
    if cat_type in ["unprojected", "driftscan"]:
        if cat_lon is None:
            cat_lon = 0.0
        if cat_lat is None:
            cat_lat = np.pi / 2
        if cat_frame is None:
            cat_frame = "altaz"

    if phase_dict is None:
        phase_dict = {
            "cat_type": cat_type,
            "cat_lon": cat_lon,
            "cat_lat": cat_lat,
            "cat_frame": cat_frame,
            "cat_epoch": cat_epoch,
            "cat_times": cat_times,
            "cat_pm_ra": cat_pm_ra,
            "cat_pm_dec": cat_pm_dec,
            "cat_dist": cat_dist,
            "cat_vrad": cat_vrad,
        }

    tol_dict = {
        "cat_type": None,
        "cat_lon": radian_tols,
        "cat_lat": radian_tols,
        "cat_frame": None,
        "cat_epoch": None,
        "cat_times": default_tols,
        "cat_pm_ra": default_tols,
        "cat_pm_dec": default_tols,
        "cat_dist": default_tols,
        "cat_vrad": default_tols,
    }

    if target_cat_id is not None:
        if target_cat_id not in phase_center_catalog:
            raise ValueError(f"No phase center with ID number {target_cat_id}.")
        name_dict = {target_cat_id: phase_center_catalog[target_cat_id]["cat_name"]}
    else:
        name_dict = {
            key: cat_dict["cat_name"] for key, cat_dict in phase_center_catalog.items()
        }

    for cat_id, name in name_dict.items():
        cat_diffs = 0
        if (cat_name != name) and (not ignore_name):
            continue
        check_dict = phase_center_catalog[cat_id]
        for key in tol_dict:
            if phase_dict.get(key) is not None:
                if check_dict.get(key) is None:
                    cat_diffs += 1
                elif tol_dict[key] is None:
                    # If no tolerance specified, expect attributes to be identical
                    cat_diffs += phase_dict.get(key) != check_dict.get(key)
                else:
                    # allclose will throw a Value error if you have two arrays
                    # of different shape, which we can catch to flag that
                    # the two arrays are actually not within tolerance.
                    if np.shape(phase_dict[key]) != np.shape(check_dict[key]):
                        cat_diffs += 1
                    else:
                        cat_diffs += not np.allclose(
                            phase_dict[key],
                            check_dict[key],
                            tol_dict[key][0],
                            tol_dict[key][1],
                        )
            else:
                cat_diffs += check_dict.get(key) is not None

        if (cat_diffs == 0) or (cat_name == name):
            if cat_diffs < match_diffs:
                # If our current match is an improvement on any previous matches,
                # then record it as the best match.
                match_id = cat_id
                match_diffs = cat_diffs
            if match_diffs == 0:
                # If we have a total match, we can bail at this point
                break

    return match_id, match_diffs


def look_for_name(phase_center_catalog, cat_name):
    """
    Look up catalog IDs which match a given name.

    Parameters
    ----------
    phase_center_catalog : dict
        Catalog to look for matching names in.
    cat_name : str or list of str
        Name to match against entries in phase_center_catalog.

    Returns
    -------
    cat_id_list : list
        List of all catalog IDs which match the given name.
    """
    if isinstance(cat_name, str):
        return [
            pc_id
            for pc_id, pc_dict in phase_center_catalog.items()
            if pc_dict["cat_name"] == cat_name
        ]
    else:
        return [
            pc_id
            for pc_id, pc_dict in phase_center_catalog.items()
            if pc_dict["cat_name"] in cat_name
        ]


def print_phase_center_info(
    phase_center_catalog,
    catalog_identifier=None,
    *,
    hms_format=None,
    return_str=False,
    print_table=True,
):
    """
    Print out the details of the phase centers.

    Prints out an ASCII table that contains the details of the supploed phase center
    catalog, which typically acts as the internal source catalog for various UV objects.

    Parameters
    ----------
    phase_center_catalog : dict
        Dict containing the list of phase centers (and corresponding data) to be
        printed out.
    catalog_identifier : str or int or list of str or int
        Optional parameter which, if provided, will cause the method to only return
        information on the phase center(s) with the matching name(s) or catalog ID
        number(s). Default is to print out information on all catalog entries.
    hms_format : bool
        Optional parameter, which if selected, can be used to force coordinates to
        be printed out in Hours-Min-Sec (if set to True) or Deg-Min-Sec (if set to
        False) format. Default is to print out in HMS if all the objects have
        coordinate frames of icrs, gcrs, fk5, fk4, and top; otherwise, DMS format
        is used.
    return_str: bool
        If set to True, the method returns an ASCII string which contains all the
        table infrmation. Default is False.
    print_table : bool
        If set to True, prints the table to the terminal window. Default is True.

    Returns
    -------
    table_str : bool
        If return_str=True, an ASCII string containing the entire table text

    Raises
    ------
    ValueError
        If `cat_name` matches no keys in `phase_center_catalog`.
    """
    r2d = 180.0 / np.pi
    r2m = 60.0 * 180.0 / np.pi
    r2s = 3600.0 * 180.0 / np.pi
    ra_frames = ["icrs", "gcrs", "fk5", "fk4", "topo"]

    if catalog_identifier is not None:
        if (
            isinstance(catalog_identifier, str | int)
            or isinstance(catalog_identifier, list)
            and all(isinstance(cat, str | int) for cat in catalog_identifier)
        ):
            pass
        else:
            raise TypeError(
                "catalog_identifier must be a string, an integer or a list of "
                "strings or integers."
            )

        if not isinstance(catalog_identifier, list):
            catalog_identifier = [catalog_identifier]

        cat_id_list = []
        for cat in catalog_identifier:
            if isinstance(cat, str):
                this_list = []
                for key, ps_dict in phase_center_catalog.items():
                    if ps_dict["cat_name"] == cat:
                        this_list.append(key)
                if len(this_list) == 0:
                    raise ValueError(f"No entry by the name {cat} in the catalog.")
                cat_id_list.extend(this_list)
            else:
                # Force cat_id to be a list to make downstream code simpler.
                # If cat_id is an int, it will throw a TypeError on casting to
                # list, which we can catch.
                if cat not in phase_center_catalog:
                    raise ValueError(f"No entry with the ID {cat} in the catalog.")
                cat_id_list.append(cat)
    else:
        cat_id_list = list(phase_center_catalog)

    dict_list = [phase_center_catalog[cat_id] for cat_id in cat_id_list]

    # We want to check and actually see which fields we need to print
    any_lon = any_lat = any_frame = any_epoch = any_times = False
    any_pm_ra = any_pm_dec = any_dist = any_vrad = False

    for indv_dict in dict_list:
        any_lon = any_lon or indv_dict.get("cat_lon") is not None
        any_lat = any_lat or indv_dict.get("cat_lat") is not None
        any_frame = any_frame or indv_dict.get("cat_frame") is not None
        any_epoch = any_epoch or indv_dict.get("cat_epoch") is not None
        any_times = any_times or indv_dict.get("cat_times") is not None
        any_pm_ra = any_pm_ra or indv_dict.get("cat_pm_ra") is not None
        any_pm_dec = any_pm_dec or indv_dict.get("cat_pm_dec") is not None
        any_dist = any_dist or indv_dict.get("cat_dist") is not None
        any_vrad = any_vrad or indv_dict.get("cat_vrad") is not None

        if any_lon and (hms_format is None):
            cat_frame = indv_dict.get("cat_frame")
            cat_type = indv_dict["cat_type"]
            if (cat_frame not in ra_frames) or (cat_type == "driftscan"):
                hms_format = False

    if hms_format is None:
        hms_format = True

    col_list = []
    col_list.append(
        {"hdr": ("ID", "#"), "fmt": "% 4i", "field": " %4s ", "name": "cat_id"}
    )
    col_list.append(
        {
            "hdr": ("Cat Entry", "Name"),
            "fmt": "%12s",
            "field": " %12s ",
            "name": "cat_name",
        }
    )
    col_list.append(
        {"hdr": ("Type", ""), "fmt": "%12s", "field": " %12s ", "name": "cat_type"}
    )

    if any_lon:
        col_list.append(
            {
                "hdr": ("Az/Lon/RA", "hours" if hms_format else "deg"),
                "fmt": "% 3i:%02i:%05.2f",
                "field": " %12s " if hms_format else " %13s ",
                "name": "cat_lon",
            }
        )
    if any_lat:
        col_list.append(
            {
                "hdr": ("El/Lat/Dec", "deg"),
                "fmt": "%1s%2i:%02i:%05.2f",
                "field": " %12s ",
                "name": "cat_lat",
            }
        )
    if any_frame:
        col_list.append(
            {"hdr": ("Frame", ""), "fmt": "%5s", "field": " %5s ", "name": "cat_frame"}
        )
    if any_epoch:
        col_list.append(
            {"hdr": ("Epoch", ""), "fmt": "%7s", "field": " %7s ", "name": "cat_epoch"}
        )
    if any_times:
        col_list.append(
            {
                "hdr": ("   Ephem Range   ", "Start-MJD    End-MJD"),
                "fmt": " %8.2f  % 8.2f",
                "field": " %20s ",
                "name": "cat_times",
            }
        )
    if any_pm_ra:
        col_list.append(
            {
                "hdr": ("PM-Ra", "mas/yr"),
                "fmt": "%.4g",
                "field": " %6s ",
                "name": "cat_pm_ra",
            }
        )
    if any_pm_dec:
        col_list.append(
            {
                "hdr": ("PM-Dec", "mas/yr"),
                "fmt": "%.4g",
                "field": " %6s ",
                "name": "cat_pm_dec",
            }
        )
    if any_dist:
        col_list.append(
            {"hdr": ("Dist", "pc"), "fmt": "%.1e", "field": " %7s ", "name": "cat_dist"}
        )
    if any_vrad:
        col_list.append(
            {
                "hdr": ("V_rad", "km/s"),
                "fmt": "%.4g",
                "field": " %6s ",
                "name": "cat_vrad",
            }
        )

    top_str = ""
    bot_str = ""
    for col in col_list:
        top_str += col["field"] % col["hdr"][0]
        bot_str += col["field"] % col["hdr"][1]

    info_str = ""

    info_str += top_str + "\n"
    info_str += bot_str + "\n"
    info_str += ("-" * len(bot_str)) + "\n"
    # We want to print in the order of cat_id
    for idx in np.argsort(cat_id_list):
        tbl_str = ""
        for col in col_list:
            # If we have a "special" field that needs extra handling,
            # take care of that up front
            if col["name"] == "cat_id":
                temp_val = cat_id_list[idx]
            else:
                temp_val = dict_list[idx][col["name"]]
            if temp_val is None:
                temp_str = ""
            elif col["name"] == "cat_lon":
                # Force the longitude component to be a positive value
                temp_val = np.mod(np.median(temp_val), 2 * np.pi)
                temp_val /= 15.0 if hms_format else 1.0
                coord_tuple = (
                    np.mod(temp_val * r2d, 360.0),
                    np.mod(temp_val * r2m, 60.0),
                    np.mod(temp_val * r2s, 60.0),
                )
                temp_str = col["fmt"] % coord_tuple
            elif col["name"] == "cat_lat":
                temp_val = np.median(temp_val)
                coord_tuple = (
                    "-" if temp_val < 0.0 else "+",
                    np.mod(np.abs(temp_val) * r2d, 360.0),
                    np.mod(np.abs(temp_val) * r2m, 60.0),
                    np.mod(np.abs(temp_val) * r2s, 60.0),
                )
                temp_str = col["fmt"] % coord_tuple
            elif col["name"] == "cat_epoch":
                use_byrs = dict_list[idx]["cat_frame"] in ["fk4", "fk4noeterms"]
                temp_val = ("B%6.1f" if use_byrs else "J%6.1f") % temp_val
                temp_str = col["fmt"] % temp_val
            elif col["name"] == "cat_times":
                time_tuple = (
                    np.min(temp_val) - 2400000.5,
                    np.max(temp_val) - 2400000.5,
                )
                temp_str = col["fmt"] % time_tuple
            elif (col["name"] == "cat_dist") or (col["name"] == "cat_vrad"):
                temp_val = np.median(temp_val)
                temp_str = col["fmt"] % temp_val
            else:
                temp_str = col["fmt"] % temp_val
            tbl_str += col["field"] % temp_str
        info_str += tbl_str + "\n"

    if print_table:
        # We need this extra bit of code to handle trailing whitespace, since
        # otherwise some checks (e.g., doc check on tutorials) will balk
        print(
            "\n".join([line.rstrip() for line in info_str.split("\n")]), end=""
        )  # pragma: nocover
    if return_str:
        return info_str


def generate_new_phase_center_id(
    phase_center_catalog=None, *, cat_id=None, old_id=None, reserved_ids=None
):
    """
    Update a phase center with a new catalog ID number.

    Parameters
    ----------
    phase_center_catalog : dict
        Catalog to be updated. Note that the supplied catalog will be modified in situ.
    cat_id : int
        Optional argument. If supplied, then the method will check to see that the
        supplied ID is not in either the supplied catalog or in the reserved IDs.
        provided value as the new catalog ID, provided that an existing catalog
        If not supplied, then the method will automatically assign a value, defaulting
        to the value in `cat_id` if supplied (and assuming that ID value has no
        conflicts with the reserved IDs).
    old_id : int
        Optional argument, current catalog ID of the phase center, which corresponds to
        a key in `phase_center_catalog`.
    reserved_ids : array-like in int
        Optional argument. An array-like of ints that denotes which ID numbers
        are already reserved. Useful for when combining two separate catalogs.

    Returns
    -------
    new_id : int
        New phase center ID.

    Raises
    ------
    ValueError
        If there's no entry that matches `cat_id`, or of the value `new_id`
        is already taken.
    """
    used_cat_ids = set()
    if phase_center_catalog is None:
        if old_id is not None:
            raise ValueError("Cannot specify old_id if no catalog is supplied.")
    else:
        used_cat_ids = set(phase_center_catalog)
        if old_id is not None:
            if old_id not in phase_center_catalog:
                raise ValueError(f"No match in catalog to an entry with id {cat_id}.")
            used_cat_ids.remove(old_id)

    if reserved_ids is not None:
        used_cat_ids = used_cat_ids.union(reserved_ids)

    if cat_id is None:
        # Default to using the old ID if available.
        cat_id = old_id

        # If the old ID is in the reserved list, then we'll need to update it
        if (old_id is None) or (old_id in used_cat_ids):
            cat_id = set(range(len(used_cat_ids) + 1)).difference(used_cat_ids).pop()
    elif cat_id in used_cat_ids:
        if phase_center_catalog is not None and cat_id in phase_center_catalog:
            raise ValueError(
                "Provided cat_id belongs to another source ({}).".format(
                    phase_center_catalog[cat_id]["cat_name"]
                )
            )
        else:
            raise ValueError("Provided cat_id was found in reserved_ids.")

    return cat_id


def generate_phase_center_cat_entry(
    cat_name=None,
    *,
    cat_type=None,
    cat_lon=None,
    cat_lat=None,
    cat_frame=None,
    cat_epoch=None,
    cat_times=None,
    cat_pm_ra=None,
    cat_pm_dec=None,
    cat_dist=None,
    cat_vrad=None,
    info_source="user",
    force_update=False,
    cat_id=None,
):
    """
    Add an entry to a object/source catalog or find a matching one.

    This is a helper function for identifying and adding a phase center to a catalog,
    typically contained within the attribute `phase_center_catalog`. If a matching
    phase center is found, the catalog ID associated with that phase center is returned.

    Parameters
    ----------
    cat_name : str
        Name of the phase center to be added.
    cat_type : str
        Type of phase center to be added. Must be one of:
            "sidereal" (fixed RA/Dec),
            "ephem" (RA/Dec that moves with time),
            "driftscan" (fixed az/el position),
            "unprojected" (no w-projection, equivalent to the old
            `phase_type` == "drift").
            "near-field" (equivalent to sidereal with the addition
            of near-field corrections)
    cat_lon : float or ndarray
        Value of the longitudinal coordinate (e.g., RA, Az, l) in radians of the
        phase center. No default unless `cat_type="unprojected"`, in which case the
        default is zero. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    cat_lat : float or ndarray
        Value of the latitudinal coordinate (e.g., Dec, El, b) in radians of the
        phase center. No default unless `cat_type="unprojected"`, in which case the
        default is pi/2. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    cat_frame : str
        Coordinate frame that cat_lon and cat_lat are given in. Only used
        for sidereal and ephem targets. Can be any of the several supported frames
        in astropy (a limited list: fk4, fk5, icrs, gcrs, cirs, galactic).
    cat_epoch : str or float
        Epoch of the coordinates, only used when cat_frame = fk4 or fk5. Given
        in units of fractional years, either as a float or as a string with the
        epoch abbreviation (e.g, Julian epoch 2000.0 would be J2000.0).
    cat_times : ndarray of floats
        Only used when `cat_type="ephem"`. Describes the time for which the values
        of `cat_lon` and `cat_lat` are caclulated, in units of JD. Shape is (Npts,).
    cat_pm_ra : float
        Proper motion in RA, in units of mas/year. Only used for sidereal phase
        centers.
    cat_pm_dec : float
        Proper motion in Dec, in units of mas/year. Only used for sidereal phase
        centers.
    cat_dist : float or ndarray of float
        Distance of the source, in units of pc. Only used for sidereal and ephem
        phase centers. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    cat_vrad : float or ndarray of float
        Radial velocity of the source, in units of km/s. Only used for sidereal and
        ephem phase centers. Expected to be a float for sidereal and driftscan phase
        centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
    info_source : str
        Optional string describing the source of the information provided. Used
        primarily in UVData to denote when an ephemeris has been supplied by the
        JPL-Horizons system, user-supplied, or read in by one of the various file
        interpreters. Default is 'user'.
    force_update : bool
        Normally, `_add_phase_center` will throw an error if there already exists a
        phase_center with the given cat_id. However, if one sets
        `force_update=True`, the method will overwrite the existing entry in
        `phase_center_catalog` with the parameters supplied. Note that doing this
        will _not_ update other attributes of the `UVData` object. Default is False.
    cat_id : int
        An integer signifying the ID number for the phase center, used in the
        `phase_center_id_array` attribute. If a matching phase center entry exists
        already, that phase center ID will be returned, which may be different than
        the value specified to this parameter. The default is for the method to
        assign this value automatically.

    Returns
    -------
    phase_center_entry : dict
        Catalog containing the phase centers.
    cat_id : int
        The unique ID number for the phase center that either matches the specified
        parameters or was added to the internal catalog. If a matching entry was
        found, this may not be the value passed to the `cat_id` parameter. This
        value is used in the `phase_center_id_array` attribute to denote which
        source a given baseline-time corresponds to.

    Raises
    ------
    ValueError
        If attempting to add a non-unique source name or if adding a sidereal
        source without coordinates.

    """
    if not isinstance(cat_name, str):
        raise ValueError("cat_name must be a string.")

    # We currently only have 5 supported types -- make sure the user supplied
    # one of those
    if cat_type not in allowed_cat_types:
        raise ValueError(f"cat_type must be one of {allowed_cat_types}.")

    # Both proper motion parameters need to be set together
    if (cat_pm_ra is None) != (cat_pm_dec is None):
        raise ValueError(
            "Must supply values for either both or neither of cat_pm_ra and cat_pm_dec."
        )

    # If left unset, unprojected and driftscan defaulted to Az, El = (0 deg, 90 deg)
    if cat_type in ["unprojected", "driftscan"]:
        if cat_lon is None:
            cat_lon = 0.0
        if cat_lat is None:
            cat_lat = np.pi / 2
        if cat_frame is None:
            cat_frame = "altaz"

    # check some case-specific things and make sure all the entries are acceptable
    if (cat_times is None) and (cat_type == "ephem"):
        raise ValueError("cat_times cannot be None for ephem object.")
    elif (cat_times is not None) and (cat_type != "ephem"):
        raise ValueError("cat_times cannot be used for non-ephem phase centers.")

    if (cat_lon is None) and (cat_type in ["sidereal", "ephem"]):
        raise ValueError("cat_lon cannot be None for sidereal or ephem phase centers.")

    if (cat_lat is None) and (cat_type in ["sidereal", "ephem"]):
        raise ValueError("cat_lat cannot be None for sidereal or ephem phase centers.")

    if (cat_frame is None) and (cat_type in ["sidereal", "ephem"]):
        raise ValueError(
            "cat_frame cannot be None for sidereal or ephem phase centers."
        )
    elif (cat_frame != "altaz") and (cat_type in ["driftscan", "unprojected"]):
        raise ValueError(
            "cat_frame must be either None or 'altaz' when the cat type "
            "is either driftscan or unprojected."
        )

    if (cat_type == "unprojected") and (cat_lon != 0.0):
        raise ValueError(
            "Catalog entries that are unprojected must have cat_lon set to either "
            "0 or None."
        )
    if (cat_type == "unprojected") and (cat_lat != (np.pi / 2)):
        raise ValueError(
            "Catalog entries that are unprojected must have cat_lat set to either "
            "pi/2 or None."
        )

    if (cat_type != "sidereal") and (
        (cat_pm_ra is not None) or (cat_pm_dec is not None)
    ):
        raise ValueError(
            "Non-zero proper motion values (cat_pm_ra, cat_pm_dec) "
            "for cat types other than sidereal are not supported."
        )

    if isinstance(cat_epoch, Time | str):
        if cat_frame in ["fk4", "fk4noeterms"]:
            cat_epoch = Time(cat_epoch).byear
        else:
            cat_epoch = Time(cat_epoch).jyear
    elif cat_epoch is not None:
        cat_epoch = float(cat_epoch)

    if cat_type == "ephem":
        cat_times = np.array(cat_times, dtype=float).reshape(-1)
        cshape = cat_times.shape
        try:
            cat_lon = np.array(cat_lon, dtype=float).reshape(cshape)
            cat_lat = np.array(cat_lat, dtype=float).reshape(cshape)
            if cat_dist is not None:
                cat_dist = np.array(cat_dist, dtype=float).reshape(cshape)
            if cat_vrad is not None:
                cat_vrad = np.array(cat_vrad, dtype=float).reshape(cshape)
        except ValueError as err:
            raise ValueError(
                "Object properties -- lon, lat, pm_ra, pm_dec, dist, vrad -- must "
                "be of the same size as cat_times for ephem phase centers."
            ) from err
    else:
        if cat_lon is not None:
            cat_lon = float(cat_lon)
        cat_lon = None if cat_lon is None else float(cat_lon)
        cat_lat = None if cat_lat is None else float(cat_lat)
        cat_pm_ra = None if cat_pm_ra is None else float(cat_pm_ra)
        cat_pm_dec = None if cat_pm_dec is None else float(cat_pm_dec)
        cat_dist = None if cat_dist is None else float(cat_dist)
        cat_vrad = None if cat_vrad is None else float(cat_vrad)

    cat_entry = {
        "cat_name": cat_name,
        "cat_type": cat_type,
        "cat_lon": cat_lon,
        "cat_lat": cat_lat,
        "cat_frame": cat_frame,
        "cat_epoch": cat_epoch,
        "cat_times": cat_times,
        "cat_pm_ra": cat_pm_ra,
        "cat_pm_dec": cat_pm_dec,
        "cat_vrad": cat_vrad,
        "cat_dist": cat_dist,
        "info_source": info_source,
    }

    return cat_entry
