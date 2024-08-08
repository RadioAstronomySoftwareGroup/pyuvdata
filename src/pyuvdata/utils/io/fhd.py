# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with FHD files."""

import os
import warnings

import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation
from scipy.io import readsav

from ... import Telescope
from .. import coordinates


def fhd_filenames(
    *,
    vis_files: list[str] | np.ndarray | str | None = None,
    params_file: str | None = None,
    obs_file: str | None = None,
    flags_file: str | None = None,
    layout_file: str | None = None,
    settings_file: str | None = None,
    cal_file: str | None = None,
):
    """
    Check the FHD input files for matching prefixes and folders.

    Parameters
    ----------
    vis_files : str or array-like of str, optional
        FHD visibility save file names, can be data or model visibilities.
    params_file : str
        FHD params save file name.
    obs_file : str
        FHD obs save file name.
    flags_file : str
        FHD flag save file name.
    layout_file : str
        FHD layout save file name.
    layout_file : str
        FHD layout save file name.
    settings_file : str
        FHD settings text file name.
    cal_file : str
        FHD cal save file name.

    Returns
    -------
    A list of file basenames to be used in the object `filename` attribute.

    """
    file_types = {
        "vis": {"files": vis_files, "suffix": "_vis", "sub_folder": "vis_data"},
        "cal": {"files": cal_file, "suffix": "_cal", "sub_folder": "calibration"},
        "flags": {"files": flags_file, "suffix": "_flags", "sub_folder": "vis_data"},
        "layout": {"files": layout_file, "suffix": "_layout", "sub_folder": "metadata"},
        "obs": {"files": obs_file, "suffix": "_obs", "sub_folder": "metadata"},
        "params": {"files": params_file, "suffix": "_params", "sub_folder": "metadata"},
        "settings": {
            "files": settings_file,
            "suffix": "_settings",
            "sub_folder": "metadata",
        },
    }

    basename_list = []
    prefix_list = []
    folder_list = []
    missing_suffix = []
    missing_subfolder = []
    for ftype, fdict in file_types.items():
        if fdict["files"] is None:
            continue
        if isinstance(fdict["files"], list | np.ndarray):
            these_files = fdict["files"]
        else:
            these_files = [fdict["files"]]

        for fname in these_files:
            dirname, basename = os.path.split(fname)
            basename_list.append(basename)
            if fdict["suffix"] in basename:
                suffix_loc = basename.find(fdict["suffix"])
                prefix_list.append(basename[:suffix_loc])
            else:
                missing_suffix.append(ftype)
            fhd_folder, subfolder = os.path.split(dirname)
            if subfolder == fdict["sub_folder"]:
                folder_list.append(fhd_folder)
            else:
                missing_subfolder.append(ftype)

    if len(missing_suffix) > 0:
        warnings.warn(
            "Some FHD input files do not have the expected suffix so prefix "
            f"matching could not be done. The affected file types are: {missing_suffix}"
        )
    if len(missing_subfolder) > 0:
        warnings.warn(
            "Some FHD input files do not have the expected subfolder so FHD "
            "folder matching could not be done. The affected file types are: "
            f"{missing_subfolder}"
        )

    if np.unique(prefix_list).size > 1:
        warnings.warn(
            "The FHD input files do not all have matching prefixes, so they "
            "may not be for the same data."
        )
    if np.unique(folder_list).size > 1:
        warnings.warn(
            "The FHD input files do not all have the same parent folder, so "
            "they may not be for the same FHD run."
        )

    return basename_list


def get_fhd_history(settings_file, *, return_user=False):
    """
    Small function to get the important history from an FHD settings text file.

    Includes information about the command line call, the user, machine name and date

    Parameters
    ----------
    settings_file : str
        FHD settings file name
    return_user : bool
        optionally return the username who ran FHD

    Returns
    -------
    history : str
        string of history extracted from the settings file
    user : str
        Only returned if return_user is True

    """
    with open(settings_file) as f:
        settings_lines = f.readlines()
    main_loc = None
    command_loc = None
    obs_loc = None
    user_line = None
    for ind, line in enumerate(settings_lines):
        if line.startswith("##MAIN"):
            main_loc = ind
        if line.startswith("##COMMAND_LINE"):
            command_loc = ind
        if line.startswith("##OBS"):
            obs_loc = ind
        if line.startswith("User"):
            user_line = ind
        if (
            main_loc is not None
            and command_loc is not None
            and obs_loc is not None
            and user_line is not None
        ):
            break

    main_lines = settings_lines[main_loc + 1 : command_loc]
    command_lines = settings_lines[command_loc + 1 : obs_loc]
    history_lines = ["FHD history\n"] + main_lines + command_lines
    for ind, line in enumerate(history_lines):
        history_lines[ind] = line.rstrip().replace("\t", " ")
    history = "\n".join(history_lines)
    user = settings_lines[user_line].split()[1]

    if return_user:
        return history, user
    else:
        return history


def _xyz_close(xyz1, xyz2, loc_tols):
    return np.allclose(xyz1, xyz2, rtol=loc_tols[0], atol=loc_tols[1])


def _latlonalt_close(latlonalt1, latlonalt2, radian_tol, loc_tols):
    latlon_close = np.allclose(
        np.array(latlonalt1[0:2]), np.array(latlonalt2[0:2]), rtol=0, atol=radian_tol
    )
    alt_close = np.isclose(
        latlonalt1[2], latlonalt2[2], rtol=loc_tols[0], atol=loc_tols[1]
    )
    return latlon_close and alt_close


def get_fhd_layout_info(
    *,
    layout_file,
    telescope_name,
    latitude,
    longitude,
    altitude,
    obs_tile_names,
    run_check_acceptability=True,
):
    """
    Get the telescope and antenna positions from an FHD layout file.

    Parameters
    ----------
    layout_file : str
        FHD layout file name
    telescope_name : str
        Telescope name
    latitude : float
        telescope latitude in radians
    longitude : float
        telescope longitude in radians
    altitude : float
        telescope altitude in meters
    obs_tile_names : array-like of str
        Tile names from the bl_info structure inside the obs structure.
        Only used if telescope_name is "mwa".
    run_check_acceptability : bool
        Option to check acceptable range of the telescope locations.

    Returns
    -------
    dict
        A dictionary of parameters from the layout file to assign to the object. The
        keys are:

        * telescope_xyz : Telescope location in ECEF, shape (3,) (float)
        * Nants_telescope : Number of antennas in the telescope (int)
        * antenna_postions : Antenna positions in relative ECEF,
            shape (Nants_telescope, 3) (float)
        * antenna_names : Antenna names, length Nants_telescope (list of str)
        * antenna_numbers : Antenna numbers, shape (Nants_telescope,) (array of int)
        * gst0 : Greenwich sidereal time at midnight on reference date. (float)
        * earth_omega : Earth's rotation rate in degrees per day. (float)
        * dut1 : DUT1 (google it) AIPS 117 calls it UT1UTC. (float)
        * timesys : Time system (should only ever be UTC). (str)
        * diameters : Antenna diameters in meters. shape (Nants_telescope,) (float)
        * extra_keywords : Dictionary of extra keywords to preserve on the object.

    """
    layout_dict = readsav(layout_file, python_dict=True)
    layout = layout_dict["layout"]

    layout_fields = [name.lower() for name in layout.dtype.names]
    # Try to get the telescope location from the layout file &
    # compare it to the position from the obs structure.
    arr_center = layout["array_center"][0]
    layout_fields.remove("array_center")

    xyz_telescope_frame = layout["coordinate_frame"][0].decode("utf8").lower()
    layout_fields.remove("coordinate_frame")

    if xyz_telescope_frame.strip() == "itrf":
        # compare to lat/lon/alt
        location_latlonalt = coordinates.XYZ_from_LatLonAlt(
            latitude, longitude, altitude
        )
        latlonalt_arr_center = coordinates.LatLonAlt_from_XYZ(
            arr_center, check_acceptability=run_check_acceptability
        )
        # tolerances are limited by the fact that lat/lon/alt are only saved
        # as floats in the obs structure
        loc_tols = (0, 0.1)  # in meters
        radian_tol = 10.0 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)  # 10mas
        # check both lat/lon/alt and xyz because of subtle differences
        # in tolerances
        if _xyz_close(location_latlonalt, arr_center, loc_tols) or _latlonalt_close(
            (latitude, longitude, altitude), latlonalt_arr_center, radian_tol, loc_tols
        ):
            telescope_location = EarthLocation.from_geocentric(
                *location_latlonalt, unit="m"
            )
        else:
            # values do not agree with each other to within the tolerances.
            # this is a known issue with FHD runs on cotter uvfits
            # files for the MWA
            # compare with the known_telescopes values
            try:
                telescope_obj = Telescope.from_known_telescopes(telescope_name)
            except ValueError:
                telescope_obj = None
            # start warning message
            message = (
                "Telescope location derived from obs lat/lon/alt "
                "values does not match the location in the layout file."
            )

            if telescope_obj is not None:
                message += " Using the value from known_telescopes."
                telescope_location = telescope_obj.location
            else:
                message += (
                    " Telescope is not in known_telescopes. "
                    "Defaulting to using the obs derived values."
                )
                telescope_location = EarthLocation.from_geocentric(
                    *location_latlonalt, unit="m"
                )
            # issue warning
            warnings.warn(message)
    else:
        telescope_location = EarthLocation.from_geodetic(
            lat=latitude * units.rad,
            lon=longitude * units.rad,
            height=altitude * units.m,
        )

    # The FHD positions derive directly from uvfits, so they are in the rotated
    # ECEF frame and must be converted to ECEF
    rot_ecef_positions = layout["antenna_coords"][0]
    layout_fields.remove("antenna_coords")
    # use the longitude from the layout file because that's how the antenna
    # positions were calculated
    latitude, longitude, altitude = coordinates.LatLonAlt_from_XYZ(
        arr_center, check_acceptability=run_check_acceptability
    )
    antenna_positions = coordinates.ECEF_from_rotECEF(rot_ecef_positions, longitude)

    antenna_names = [ant.decode("utf8") for ant in layout["antenna_names"][0].tolist()]
    layout_fields.remove("antenna_names")

    # make these 0-indexed (rather than one indexed)
    antenna_numbers = layout["antenna_numbers"][0]
    layout_fields.remove("antenna_numbers")

    Nants_telescope = int(layout["n_antenna"][0])
    layout_fields.remove("n_antenna")

    if telescope_name.lower() == "mwa" and [ant.strip() for ant in obs_tile_names] != [
        ant.strip() for ant in antenna_names
    ]:
        # check that obs.baseline_info.tile_names match the antenna names
        # (accounting for possible differences in white space)
        # this only applies for MWA because the tile_names come from
        # metafits files. layout["antenna_names"] comes from the antenna table
        # in the uvfits file and will be used if no metafits was submitted
        warnings.warn(
            "tile_names from obs structure does not match antenna_names from layout"
        )

    gst0 = float(layout["gst0"][0])
    layout_fields.remove("gst0")

    if layout["ref_date"][0] != "":
        rdate = layout["ref_date"][0].decode("utf8").lower()
    else:
        rdate = None
    layout_fields.remove("ref_date")

    earth_omega = float(layout["earth_degpd"][0])
    layout_fields.remove("earth_degpd")

    dut1 = float(layout["dut1"][0])
    layout_fields.remove("dut1")

    timesys = layout["time_system"][0].decode("utf8").upper().strip()
    layout_fields.remove("time_system")

    if "diameters" in layout_fields:
        diameters = np.asarray(layout["diameters"][0])
        layout_fields.remove("diameters")
    else:
        diameters = None

    extra_keywords = {}
    # ignore some fields, put everything else in extra_keywords
    layout_fields_ignore = [
        "diff_utc",
        "pol_type",
        "n_pol_cal_params",
        "mount_type",
        "axis_offset",
        "pola",
        "pola_orientation",
        "pola_cal_params",
        "polb",
        "polb_orientation",
        "polb_cal_params",
        "beam_fwhm",
    ]
    for field in layout_fields_ignore:
        if field in layout_fields:
            layout_fields.remove(field)
    for field in layout_fields:
        keyword = field
        if len(keyword) > 8:
            keyword = field.replace("_", "")

        value = layout[field][0]
        if isinstance(value, bytes):
            value = value.decode("utf8")

        extra_keywords[keyword.upper()] = value

    layout_param_dict = {
        "telescope_location": telescope_location,
        "Nants_telescope": Nants_telescope,
        "antenna_positions": antenna_positions,
        "antenna_names": antenna_names,
        "antenna_numbers": antenna_numbers,
        "gst0": gst0,
        "rdate": rdate,
        "earth_omega": earth_omega,
        "dut1": dut1,
        "timesys": timesys,
        "diameters": diameters,
        "extra_keywords": extra_keywords,
    }

    return layout_param_dict
