# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with MeasurementSet files."""

import contextlib
import os
import warnings

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ... import __version__, utils
from ...telescopes import known_telescope_location, known_telescopes
from ...uvdata.uvdata import reporting_request

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
casa_error = None
try:
    from casacore import tables
    from casacore.tables import tableutil
except ImportError as error:
    casa_present = False
    casa_error = error

"""
This dictionary defines the mapping between CASA polarization numbers and
AIPS polarization numbers
"""
# convert from casa polarization integers to pyuvdata
POL_CASA2AIPS_DICT = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: -1,
    6: -3,
    7: -4,
    8: -2,
    9: -5,
    10: -7,
    11: -8,
    12: -6,
}

POL_AIPS2CASA_DICT = {
    aipspol: casapol for casapol, aipspol in POL_CASA2AIPS_DICT.items()
}

VEL_DICT = {
    "REST": 0,
    "LSRK": 1,
    "LSRD": 2,
    "BARY": 3,
    "GEO": 4,
    "TOPO": 5,
    "GALACTO": 6,
    "LGROUP": 7,
    "CMB": 8,
    "Undefined": 64,
}


# In CASA 'J2000' refers to a specific frame -- FK5 w/ an epoch of
# J2000. We'll plug that in here directly, noting that CASA has an
# explicit list of supported reference frames, located here:
# casa.nrao.edu/casadocs/casa-5.0.0/reference-material/coordinate-frames

COORD_PYUVDATA2CASA_DICT = {
    "J2000": ("fk5", 2000.0),  # mean equator and equinox at J2000.0 (FK5)
    "JNAT": None,  # geocentric natural frame
    "JMEAN": None,  # mean equator and equinox at frame epoch
    "JTRUE": None,  # true equator and equinox at frame epoch
    "APP": ("gcrs", 2000.0),  # apparent geocentric position
    "B1950": ("fk4", 1950.0),  # mean epoch and ecliptic at B1950.0.
    "B1950_VLA": ("fk4", 1979.0),  # mean epoch (1979.9) and ecliptic at B1950.0
    "BMEAN": None,  # mean equator and equinox at frame epoch
    "BTRUE": None,  # true equator and equinox at frame epoch
    "GALACTIC": None,  # Galactic coordinates
    "HADEC": None,  # topocentric HA and declination
    "AZEL": None,  # topocentric Azimuth and Elevation (N through E)
    "AZELSW": None,  # topocentric Azimuth and Elevation (S through W)
    "AZELNE": None,  # topocentric Azimuth and Elevation (N through E)
    "AZELGEO": None,  # geodetic Azimuth and Elevation (N through E)
    "AZELSWGEO": None,  # geodetic Azimuth and Elevation (S through W)
    "AZELNEGEO": None,  # geodetic Azimuth and Elevation (N through E)
    "ECLIPTIC": None,  # ecliptic for J2000 equator and equinox
    "MECLIPTIC": None,  # ecliptic for mean equator of date
    "TECLIPTIC": None,  # ecliptic for true equator of date
    "SUPERGAL": None,  # supergalactic coordinates
    "ITRF": None,  # coordinates wrt ITRF Earth frame
    "TOPO": None,  # apparent topocentric position
    "ICRS": ("icrs", 2000.0),  # International Celestial reference system
}


def _ms_utils_call_checks(filepath, invert_check=False):
    # Check for casa.
    if not casa_present:
        raise ImportError(no_casa_message) from casa_error
    if invert_check:
        if os.path.exists(filepath):
            raise FileExistsError(filepath + " already exists.")
    elif not tables.tableexists(filepath):
        raise FileNotFoundError(
            filepath + " not found or not recognized as an MS table."
        )


def _parse_pyuvdata_frame_ref(frame_name, epoch_val, *, raise_error=True):
    """
    Interpret a UVData pair of frame + epoch into a CASA frame name.

    Parameters
    ----------
    frame_name : str
        Name of the frame. Typically matched to the UVData attribute
        `phase_center_frame`.
    epoch_val : float
        Epoch value for the given frame, in Julian years unless `frame_name="FK4"`,
        in which case the value is in Besselian years. Typically matched to the
        UVData attribute `phase_center_epoch`.
    raise_error : bool
        Whether to raise an error if the name has no match. Default is True, if set
        to false will raise a warning instead.

    Returns
    -------
    ref_name : str
        Name of the CASA-based spatial coordinate reference frame.

    Raises
    ------
    ValueError
        If the provided coordinate frame and/or epoch value has no matching
        counterpart to those supported in CASA.

    """
    # N.B. -- this is something of a stub for a more sophisticated function to
    # handle this. For now, this just does a reverse lookup on the limited frames
    # supported by UVData objects, although eventually it can be expanded to support
    # more native MS frames.
    reverse_dict = {ref: key for key, ref in COORD_PYUVDATA2CASA_DICT.items()}

    ref_name = None
    try:
        ref_name = reverse_dict[
            (str(frame_name), 2000.0 if (epoch_val is None) else float(epoch_val))
        ]
    except KeyError as err:
        epoch_msg = (
            "no epoch" if epoch_val is None else f"epoch {format(epoch_val, 'g')}"
        )
        message = (
            f"Frame {frame_name} ({epoch_msg}) does not have a "
            "corresponding match to supported frames in the MS file format."
        )
        if raise_error:
            raise ValueError(message) from err
        else:
            warnings.warn(message)

    return ref_name


def _get_time_scale(ms_table, *, raise_error=False):
    """
    Read time scale from TIME column in an MS table.

    Parameters
    ----------
    ms_table : table
        Handle for the MeasurementSet table that contains a "TIME" column.
    raise_error : bool
        Whether to raise an error if the name has no match. Default is True, if set
        to false will raise a warning instead.

    Returns
    -------
    time_scale_name : str
        Name of the time scale.

    Raises
    ------
    ValueError
        If the CASA time scale frame does not match the known supported list of
        time scales for astropy.
    """
    timescale = ms_table.getcolkeyword("TIME", "MEASINFO")["Ref"]
    if timescale.lower() not in Time.SCALES:
        msg = (
            "This file has a timescale that is not supported by astropy. "
            "If you need support for this timescale please make an issue on our "
            "GitHub repo."
        )
        if raise_error:
            raise ValueError(
                msg + " To bypass this error, you can set raise_error=False, which "
                "will raise a warning instead and treat the time as being in UTC."
            )
        else:
            warnings.warn(msg + " Defaulting to treating it as being in UTC.")
            timescale = "utc"

    return timescale.lower()


def _parse_casa_frame_ref(ref_name, *, raise_error=True):
    """
    Interpret a CASA frame into an astropy-friendly frame and epoch.

    Parameters
    ----------
    ref_name : str
        Name of the CASA-based spatial coordinate reference frame.
    raise_error : bool
        Whether to raise an error if the name has no match. Default is True, if set
        to false will raise a warning instead.

    Returns
    -------
    frame_name : str
        Name of the frame. Typically matched to the UVData attribute
        `phase_center_frame`.
    epoch_val : float
        Epoch value for the given frame, in Julian years unless `frame_name="FK4"`,
        in which case the value is in Besselian years. Typically matched to the
        UVData attribute `phase_center_epoch`.

    Raises
    ------
    ValueError
        If the CASA coordinate frame does not match the known supported list of
        frames for CASA.
    NotImplementedError
        If using a CASA coordinate frame that does not yet have a corresponding
        astropy frame that is supported by pyuvdata.
    """
    frame_name = None
    epoch_val = None
    try:
        frame_tuple = COORD_PYUVDATA2CASA_DICT[ref_name]
        if frame_tuple is None:
            message = f"Support for the {ref_name} frame is not yet supported."
            if raise_error:
                raise NotImplementedError(message)
            else:
                warnings.warn(message)
        else:
            frame_name = frame_tuple[0]
            epoch_val = frame_tuple[1]
    except KeyError as err:
        message = (
            f"The coordinate frame {ref_name} is not one of the supported frames "
            "for measurement sets."
        )
        if raise_error:
            raise ValueError(message) from err
        else:
            warnings.warn(message)

    return frame_name, epoch_val


def read_ms_antenna(filepath, check_frame=True):
    """
    Read Measurement Set ANTENNA table.

    Parameters
    ----------
    filepath : str
        path to MS (without ANTENNA suffix)
    check_frame : bool
        If set to True and the "telescope_frame" keyword is found within the Measurement
        Set, check that the frame is one supported by pyuvdata (and if not, an error is
        raised). Currently supported frames include ITRS/ITRF and MCMF.

    Returns
    -------
    ant_dict : dict
        A dictionary with keys that map to columns of the MS, including "antenna_names"
        (list of type string, shape (Nants_telescope,)), "station_names" (list of type
        string, shape (Nants_telescope,)), "antenna_numbers" (ndarray of type int,
        shape (Nants_telescope,)), "antenna_diameters" (ndarray of type float, shape
        (Nants_telescope,)), "telescope_frame" (str), "telescope_ellipsoid" (str),
        and "antenna_positions" (ndarray of type float, shape (Nants_telescope, 3)).

    Raises
    ------
    ValueError
        If `check_frame=True` and "telescope_frame" does not match to a supported type.
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/ANTENNA")
    # open table with antenna location information
    with tables.table(filepath + "/ANTENNA", ack=False) as tb_ant:
        antenna_positions = tb_ant.getcol("POSITION")
        meas_info_dict = tb_ant.getcolkeyword("POSITION", "MEASINFO")
        telescope_frame = meas_info_dict["Ref"].lower()
        try:
            telescope_ellipsoid = str(meas_info_dict["RefEllipsoid"])
        except KeyError:
            # No keyword means go to defaults
            telescope_ellipsoid = "SPHERE" if telescope_frame == "mcmf" else None

        if check_frame:
            # Check the telescope frame to make sure it's supported
            if telescope_frame not in ["itrs", "mcmf", "itrf"]:
                raise ValueError(
                    f"Telescope frame in file is {telescope_frame}. "
                    "Only 'itrs' and 'mcmf' are currently supported."
                )
                # MS uses "ITRF" while astropy uses "itrs". They are the same.
            elif telescope_frame == "itrf":
                telescope_frame = "itrs"

        # Note: measurement sets use the antenna number as an index into the antenna
        # table. This means that if the antenna numbers do not start from 0 and/or are
        # not contiguous, empty rows are inserted into the antenna table (similar to
        # miriad)).  These 'dummy' rows have positions of zero and need to be removed.
        n_ants_table = antenna_positions.shape[0]
        good_mask = np.any(antenna_positions, axis=1)
        antenna_positions = antenna_positions[good_mask, :]
        antenna_numbers = np.arange(n_ants_table)[good_mask]

        # antenna names
        antenna_names = np.asarray(tb_ant.getcol("NAME"))[good_mask].tolist()
        station_names = np.asarray(tb_ant.getcol("STATION"))[good_mask].tolist()
        ant_diameters = np.asarray(tb_ant.getcol("DISH_DIAMETER"))[good_mask]
        antenna_mount = np.asarray(np.char.lower(tb_ant.getcol("MOUNT")))[
            good_mask
        ].tolist()

        if all(mount == "" for mount in antenna_mount):
            # If no mounts recorded, discard the information.
            antenna_mount = None

        if not np.any(ant_diameters > 0):
            ant_diameters = None

    # Build a dict with all the relevant entries we need.
    ant_dict = {
        "antenna_positions": antenna_positions,
        "antenna_numbers": antenna_numbers,
        "telescope_frame": telescope_frame,
        "telescope_ellipsoid": telescope_ellipsoid,
        "antenna_names": antenna_names,
        "station_names": station_names,
        "antenna_mount": antenna_mount,
        "antenna_diameters": ant_diameters,
    }

    # Return the dict
    return ant_dict


def write_ms_antenna(
    filepath,
    uvobj=None,
    *,
    antenna_numbers=None,
    antenna_names=None,
    antenna_positions=None,
    antenna_diameters=None,
    antenna_mount=None,
    telescope_location=None,
    telescope_frame=None,
    telescope_ellipsoid=None,
):
    """
    Write out the antenna information into a CASA table.

    Parameters
    ----------
    filepath : str
        Path to MS (without ANTENNA suffix).
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have a telescope
        parameter with parameters that match by name to the other keywords
        required here (with the exception of telescope_frame and telescope_ellipsoid,
        which are derived from the telescope.location UVParameter).
    antenna_numbers : ndarray
        Required if uvobj not provided, antenna numbers for all antennas of the
        telescope, dtype int and shape (Nants_telescope,).
    antenna_names : list
        Required if uvobj not provided, antenna names for all antennas of the telescope.
        List should be length Nants_telescope, and contain elements of type str.
    antenna_positions : ndarray
        Required if uvobj not provided, ITRF/MCMF 3D position (in meters) of antennas
        relative to the array center, of dtype float with shape (Nants_telescope, 3).
    antenna_diameters : ndarray
        Required if uvobj not provided, diameter (in meters) of each antenna, dtype
        float with shape (Nants_telescope,).
    telescope_location : ndarray
        Required if uvobj not provided, ITRF/MCMF 3D location of the array center (in
        meters), dtype float with shape (3,).
    telescope_frame : str
        Required if uvobj not provided, name of the frame in which the telescope
        positions are provided (typically "MCMF", "ITRS", or "ITRF").
    telescope_ellipsoid : str
        Required if uvobj not provided and telescope frame is "MCMF", ellipsoid to use
        for lunar coordinates. Must be one of "SPHERE", "GSFC", "GRAIL23",
        "CE-1-LAM-GEO" (see lunarsky package for details).

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::ANTENNA"

    if uvobj is not None:
        antenna_numbers = uvobj.telescope.antenna_numbers
        antenna_names = uvobj.telescope.antenna_names
        antenna_positions = uvobj.telescope.antenna_positions
        antenna_diameters = uvobj.telescope.antenna_diameters
        antenna_mount = uvobj.telescope.mount_type
        telescope_location = uvobj.telescope._location.xyz()
        telescope_frame = uvobj.telescope._location.frame
        telescope_ellipsoid = uvobj.telescope._location.ellipsoid

    tabledesc = tables.required_ms_desc("ANTENNA")
    dminfo = tables.makedminfo(tabledesc)

    with tables.table(
        filepath, tabledesc=tabledesc, dminfo=dminfo, ack=False, readonly=False
    ) as antenna_table:
        # Note: measurement sets use the antenna number as an index into the antenna
        # table. This means that if the antenna numbers do not start from 0 and/or are
        # not contiguous, empty rows need to be inserted into the antenna table
        # (this is somewhat similar to miriad)
        nants_table = np.max(antenna_numbers) + 1
        antenna_table.addrows(nants_table)

        ant_names_table = [""] * nants_table
        ant_mount_table = [""] * nants_table
        for ai, num in enumerate(antenna_numbers):
            ant_names_table[num] = antenna_names[ai]
            ant_mount_table[num] = "" if antenna_mount is None else antenna_mount[ai]

        # There seem to be some variation on whether the antenna names are stored
        # in the NAME or STATION column (importuvfits puts them in the STATION column
        # while Cotter and the MS definition doc puts them in the NAME column).
        # The MS definition doc suggests that antenna names belong in the NAME column
        # and the telescope name belongs in the STATION column (it gives the example of
        # GREENBANK for this column.) so we follow that here. For a reconfigurable
        # array, the STATION can be though of as the "pad" name, which is distinct from
        # the antenna name/number, and nominally fixed in position.
        antenna_table.putcol("NAME", ant_names_table)
        antenna_table.putcol("STATION", ant_names_table)
        antenna_table.putcol("MOUNT", ant_mount_table)

        # Antenna positions in measurement sets appear to be in absolute ECEF
        ant_pos_absolute = antenna_positions + telescope_location.reshape(1, 3)
        ant_pos_table = np.zeros((nants_table, 3), dtype=np.float64)
        for ai, num in enumerate(antenna_numbers):
            ant_pos_table[num, :] = ant_pos_absolute[ai, :]

        antenna_table.putcol("POSITION", ant_pos_table)
        if antenna_diameters is not None:
            ant_diam_table = np.zeros((nants_table), dtype=np.float64)
            # This is here is suppress an error that arises when one has antennas of
            # different diameters (which CASA can't handle), since otherwise the
            # "padded" antennas have zero diameter (as opposed to any real telescope).
            if len(np.unique(antenna_diameters)) == 1:
                ant_diam_table[:] = antenna_diameters[0]
            else:
                for ai, num in enumerate(antenna_numbers):
                    ant_diam_table[num] = antenna_diameters[ai]
            antenna_table.putcol("DISH_DIAMETER", ant_diam_table)

        # Add telescope frame
        telescope_frame = telescope_frame.upper()
        telescope_frame = "ITRF" if (telescope_frame == "ITRS") else telescope_frame
        meas_info_dict = antenna_table.getcolkeyword("POSITION", "MEASINFO")
        meas_info_dict["Ref"] = telescope_frame
        if telescope_frame == "MCMF" and telescope_ellipsoid is not None:
            meas_info_dict["RefEllipsoid"] = telescope_ellipsoid
        antenna_table.putcolkeyword("POSITION", "MEASINFO", meas_info_dict)


def read_ms_data_description(filepath):
    """Read Measurement Set DATA_DESCRIPTION table.

    Parameters
    ----------
    filepath : str
        path to MS (without DATA_DESCRIPTION suffix)

    Returns
    -------
    data_desc_dict : dict
        A dictionary with keys that map to columns of the MS, used to index against
        other tables. These include "SPECTRAL_WINDOW_ID", which match to given rows
        from the SPECTRAL_WINDOW table, "POLARIZATION_ID", which match to given rows
        in the POLARIZATION table, and "FLAG_ROW", which denotes of the ID in question
        has been flagged.

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/DATA_DESCRIPTION")

    # open table with the general data description
    with tables.table(filepath + "/DATA_DESCRIPTION", ack=False) as tb_desc:
        data_desc_dict = {}
        for idx in range(tb_desc.nrows()):
            data_desc_dict[idx] = {
                "SPECTRAL_WINDOW_ID": tb_desc.getcell("SPECTRAL_WINDOW_ID", idx),
                "POLARIZATION_ID": tb_desc.getcell("POLARIZATION_ID", idx),
                "FLAG_ROW": tb_desc.getcell("FLAG_ROW", idx),
            }

    return data_desc_dict


def write_ms_data_description(
    filepath, uvobj=None, nspws=None, flex_spw_polarization_array=None
):
    """
    Write out the data description information into a CASA table.

    Parameters
    ----------
    filepath : str
        Path to MS (without DATA_DESCRIPTION suffix).
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here.
    nspws : int
        Required if uvobj is not supplied, the total number of spectral windows to be
        recorded.
    flex_spw_polarization_array : list of int
        Optional argument, required if trying to record a flex-pol data set, which
        denotes the polarization per spectral window, with length equal to nspws.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::DATA_DESCRIPTION"

    if uvobj is not None:
        nspws = uvobj.Nspws
        flex_spw_polarization_array = uvobj.flex_spw_polarization_array

    with tables.table(filepath, ack=False, readonly=False) as data_descrip_table:
        data_descrip_table.addrows(nspws)
        data_descrip_table.putcol("SPECTRAL_WINDOW_ID", np.arange(nspws))

        if flex_spw_polarization_array is not None:
            pol_dict = {
                pol: idx
                for idx, pol in enumerate(np.unique(flex_spw_polarization_array))
            }
            data_descrip_table.putcol(
                "POLARIZATION_ID",
                np.array([pol_dict[key] for key in flex_spw_polarization_array]),
            )


def read_ms_field(filepath, return_phase_center_catalog=False):
    """
    Read Measurement Set FIELD table.

    Parameters
    ----------
    filepath : str
        path to MS (without FIELD suffix)
    return_phase_center_catalog : bool
        Nominally this function will return a dict containing the columns of the table,
        but if set to True, instead a catalog will be supplied in the style of the
        UVData/UVCal `phase_center_catalog` parameters (further description of this
        parameter can be found in the class documentation).

    Returns
    -------
    field_dict : dict
        A dictionary with keys that map to columns of the MS, including "name" (field
        names, list of len Nfield with elements of type str), "ra" (RA in radians,
        ndarray of floats with shape (Nfield,)), "dec" (Dec in radians, ndarray of
        floats with shape (Nfield,)), "source_id" (row number in MS file, ndarray of
        ints with shape (Nfield,)), and "alt_id" (matching source ID number, ndarray of
        ints with shape (Nfield,)). Only given if return_phase_center_catalog=False.
    phase_center_catalog : dict
        A catalog stylized like the UVData/UVCal `phase_center_catalog` parameters, with
        keys matching to the "alt_id"/source ID entries. Useful for plugging directly
        into the `phase_center_catalog` attribute of a UVBase object. Only supplied if
        return_phase_center_catalog=True.
    field_map : dict
        A dict between row number (keys) and the source ID (values), which can be used
        to map the phase center ID array from the main MS table. Only supplied if
        return_phase_center_catalog=True.

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/FIELD")

    with tables.table(filepath + "/FIELD", ack=False) as tb_field:
        n_rows = tb_field.nrows()

        field_dict = {
            "name": tb_field.getcol("NAME"),
            "ra": [None] * n_rows,
            "dec": [None] * n_rows,
            "source_id": [None] * n_rows,
            "alt_id": [None] * n_rows,
        }

        # MSv2.0 appears to assume J2000. Not sure how to specifiy otherwise
        measinfo_keyword = tb_field.getcolkeyword("PHASE_DIR", "MEASINFO")
        var_frame = "VarRefCol" in measinfo_keyword

        if var_frame:
            # This seems to be a yet-undocumented feature for CASA, which allows one
            # to specify an additional (optional?) column that defines the reference
            # frame on a per-source basis.
            ref_dir_dict = dict(
                zip(
                    measinfo_keyword["TabRefCodes"],
                    measinfo_keyword["TabRefTypes"],
                    strict=True,
                )
            )
            frame_list = []
            epoch_list = []
            for key in tb_field.getcol(measinfo_keyword["VarRefCol"]):
                frame, epoch = _parse_casa_frame_ref(ref_dir_dict[key])
                frame_list.append(frame)
                epoch_list.append(epoch)
            field_dict["frame"] = frame_list
            field_dict["epoch"] = epoch_list
        elif "Ref" in measinfo_keyword:
            field_dict["frame"], field_dict["epoch"] = _parse_casa_frame_ref(
                measinfo_keyword["Ref"]
            )
        else:
            warnings.warn("Coordinate reference frame not detected, defaulting to ICRS")
            field_dict["frame"] = "icrs"
            field_dict["epoch"] = 2000.0
        message = (
            "PHASE_DIR is expressed as a polynomial. "
            "We do not currently support this mode, please make an issue."
        )

        for idx in range(n_rows):
            phase_dir = tb_field.getcell("PHASE_DIR", idx)
            # Error if the phase_dir has a polynomial term because we don't know
            # how to handle that
            assert phase_dir.shape[0] == 1, message

            field_dict["ra"][idx] = float(phase_dir[0, 0])
            field_dict["dec"][idx] = float(phase_dir[0, 1])
            field_dict["source_id"][idx] = idx
            # If no column named SOURCE_ID exists, or if it does exist but is
            # completely unfilled, just move on.
            with contextlib.suppress(RuntimeError):
                field_dict["alt_id"][idx] = int(tb_field.getcell("SOURCE_ID", idx))

    if not return_phase_center_catalog:
        return field_dict

    phase_center_catalog = {}
    for idx in range(n_rows):
        phase_center_catalog[field_dict["source_id"][idx]] = {
            "cat_name": field_dict["name"][idx],
            "cat_type": "sidereal",
            "cat_lon": field_dict["ra"][idx],
            "cat_lat": field_dict["dec"][idx],
            "cat_frame": field_dict["frame"][idx] if var_frame else field_dict["frame"],
            "cat_epoch": field_dict["epoch"][idx] if var_frame else field_dict["epoch"],
            "cat_times": None,
            "cat_pm_ra": None,
            "cat_pm_dec": None,
            "cat_vrad": None,
            "cat_dist": None,
            "info_source": "file",
        }

    if any(item is None or item < 0 for item in field_dict["alt_id"]):
        # If there's no alt id, return a blank mapping for the field IDs
        return phase_center_catalog, {}

    # Construct the mappings of row number to preferred ID, map back to catalog
    field_id_map = dict(zip(field_dict["source_id"], field_dict["alt_id"], strict=True))
    phase_center_catalog = {
        newkey: phase_center_catalog[oldkey] for oldkey, newkey in field_id_map.items()
    }
    return phase_center_catalog, field_id_map


def write_ms_field(filepath, uvobj=None, phase_center_catalog=None, time_val=None):
    """
    Write out the field information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without FIELD suffix)
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here (with the exception of
        time_val, which is calculated from the time_array UVParameter).
    phase_center_catalog : dict
        A catalog stylized like the UVData/UVCal `phase_center_catalog` parameters (see
        documentation of those classes for more details on expected structure). Required
        if uvobj is not supplied.
    time_val : float
        Required if uvobj is not supplied, representative JD date for the catalog to
        be recorded into the MS file.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::FIELD"

    if uvobj is not None:
        phase_center_catalog = uvobj.phase_center_catalog
        time_val = (
            Time(np.median(uvobj.time_array), format="jd", scale="utc").mjd * 86400.0
        )

    tabledesc = tables.required_ms_desc("FIELD")
    dminfo = tables.makedminfo(tabledesc)

    with tables.table(
        filepath, tabledesc=tabledesc, dminfo=dminfo, ack=False, readonly=False
    ) as field_table:
        n_poly = 0

        var_ref = False
        for ind, phase_dict in enumerate(phase_center_catalog.values()):
            if ind == 0:
                sou_frame = phase_dict["cat_frame"]
                sou_epoch = phase_dict["cat_epoch"]
                continue

            if (sou_frame != phase_dict["cat_frame"]) or (
                sou_epoch != phase_dict["cat_epoch"]
            ):
                var_ref = True
                break

        if var_ref:
            var_ref_dict = {
                key: val for val, key in enumerate(COORD_PYUVDATA2CASA_DICT)
            }
            col_ref_dict = {
                "PHASE_DIR": "PhaseDir_Ref",
                "DELAY_DIR": "DelayDir_Ref",
                "REFERENCE_DIR": "RefDir_Ref",
            }
            for key in col_ref_dict:
                fieldcoldesc = tables.makearrcoldesc(
                    col_ref_dict[key],
                    0,
                    valuetype="int",
                    datamanagertype="StandardStMan",
                    datamanagergroup="field standard manager",
                )
                del fieldcoldesc["desc"]["shape"]
                del fieldcoldesc["desc"]["ndim"]
                del fieldcoldesc["desc"]["_c_order"]

                field_table.addcols(fieldcoldesc)
                field_table.getcolkeyword(key, "MEASINFO")

        ref_frame = _parse_pyuvdata_frame_ref(sou_frame, sou_epoch)
        for col in ["PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"]:
            meas_info_dict = field_table.getcolkeyword(col, "MEASINFO")
            meas_info_dict["Ref"] = ref_frame
            if var_ref:
                rev_ref_dict = {value: key for key, value in var_ref_dict.items()}
                meas_info_dict["TabRefTypes"] = [
                    rev_ref_dict[key] for key in sorted(rev_ref_dict.keys())
                ]
                meas_info_dict["TabRefCodes"] = np.arange(
                    len(rev_ref_dict.keys()), dtype=np.int32
                )
                meas_info_dict["VarRefCol"] = col_ref_dict[col]

            field_table.putcolkeyword(col, "MEASINFO", meas_info_dict)

        sou_id_list = list(phase_center_catalog)

        for idx, sou_id in enumerate(sou_id_list):
            cat_dict = phase_center_catalog[sou_id]

            phase_dir = np.array([[cat_dict["cat_lon"], cat_dict["cat_lat"]]])
            if (cat_dict["cat_type"] == "ephem") and (phase_dir.ndim == 3):
                phase_dir = np.median(phase_dir, axis=2)

            sou_name = cat_dict["cat_name"]
            ref_dir = _parse_pyuvdata_frame_ref(
                cat_dict["cat_frame"], cat_dict["cat_epoch"], raise_error=var_ref
            )

            field_table.addrows()

            field_table.putcell("DELAY_DIR", idx, phase_dir)
            field_table.putcell("PHASE_DIR", idx, phase_dir)
            field_table.putcell("REFERENCE_DIR", idx, phase_dir)
            field_table.putcell("NAME", idx, sou_name)
            field_table.putcell("NUM_POLY", idx, n_poly)
            field_table.putcell("TIME", idx, time_val)
            field_table.putcell("SOURCE_ID", idx, sou_id)
            if var_ref:
                for key in col_ref_dict:
                    field_table.putcell(col_ref_dict[key], idx, var_ref_dict[ref_dir])


def read_ms_history(filepath, pyuvdata_version_str, check_origin=False, raise_err=True):
    """
    Read a CASA history table into a string for the uvdata history parameter.

    Also stores messages column as a list for consistency with other uvdata types

    Parameters
    ----------
    filepath : str
        Path to CASA table with history information.
    pyuvdata_version_str : str
        String containing the version of pyuvdata used to read in the MS file, which is
        appended to the history if not previously encoded into the history table.
    check_origin : bool
        If set to True, check whether the MS in question was created by pyuvdata, as
        determined by the history table. Default is False.
    raise_err : bool
        Normally an error is raised if a HISTORY table cannot be found. However, if
        set to False, the history string containing just the pyuvdata version will
        be returned instead.

    Returns
    -------
    str
        string encoding complete casa history table converted with a new
        line denoting rows and a ';' denoting column breaks.
    pyuvdata_written :  bool
        boolean indicating whether or not this MS was written by pyuvdata. Only returned
        of `check_origin=True`.

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    try:
        _ms_utils_call_checks(filepath + "/HISTORY")
    except FileNotFoundError as err:
        if raise_err:
            raise err
        # Just return the defaults, since no history file was found.
        return (pyuvdata_version_str, False) if check_origin else pyuvdata_version_str

    # Set up the history string and pyuvdata check
    history_str = ""
    pyuvdata_written = False

    # Skip reading the history table if it has no information
    with tables.table(filepath + "/HISTORY", ack=False) as tb_hist:
        nrows = tb_hist.nrows()

        if nrows > 0:
            history_str = "Begin measurement set history\n"

            # Grab the standard history columns to stitch together
            application = tb_hist.getcol("APPLICATION")
            message = tb_hist.getcol("MESSAGE")
            obj_id = tb_hist.getcol("OBJECT_ID")
            obs_id = tb_hist.getcol("OBSERVATION_ID")
            origin = tb_hist.getcol("ORIGIN")
            priority = tb_hist.getcol("PRIORITY")
            times = tb_hist.getcol("TIME")

            cols = []
            # APP_PARAMS and CLI_COMMAND are not consistently filled, even though they
            # appear to be required columns. Fill them in with zero-length strings
            # if they don't conform to what we expect.
            default_col = [""] * len(times)
            for field in ["APP_PARAMS", "CLI_COMMAND"]:
                try:
                    check_val = tb_hist.getcol(field)["array"]
                    cols.append(check_val if (len(check_val) > 0) else default_col)
                except RuntimeError:
                    cols.append(default_col)

            # Now add the rest of the columns and generate history string
            cols += [application, message, obj_id, obs_id, origin, priority, times]

            history_str += (
                "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE;OBJECT_ID;"
                "OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
            )

            # if this MS was written by pyuvdata, some history that originated in
            # pyuvdata is in the MS history table. We separate that out since it doesn't
            # really belong to the MS history block (and so round tripping works)
            pyuvdata_line_idx = [
                idx for idx, line in enumerate(application) if "pyuvdata" in line
            ]

            for row_idx in range(nrows):
                # first check to see if this row was put in by pyuvdata.
                # If so, don't mix them in with the MS stuff
                if row_idx in pyuvdata_line_idx:
                    continue

                newline = ";".join([str(col[row_idx]) for col in cols]) + "\n"
                history_str += newline

            history_str += "End measurement set history.\n"

            # make a list of lines that are MS specific (i.e. not pyuvdata lines)
            ms_line_idx = list(np.arange(nrows))
            for drop_idx in reversed(pyuvdata_line_idx):
                # Drop the pyuvdata-related lines, since we handle them separately.
                # We do this in reverse to keep from messing up the indexing of the
                # earlier entries.
                ms_line_idx.pop(drop_idx)

            # Handle the case where there is no history but pyuvdata
            if len(ms_line_idx) == 0:
                ms_line_idx = [-1]

            if len(pyuvdata_line_idx) > 0:
                pyuvdata_written = True
                for idx in pyuvdata_line_idx:
                    if idx < min(ms_line_idx):
                        # prepend these lines to the history
                        history_str = message[idx] + "\n" + history_str
                    else:
                        # append these lines to the history
                        history_str += message[idx] + "\n"

    # Check and make sure the pyuvdata version is in the history if it's not already
    if not utils.history._check_history_version(history_str, pyuvdata_version_str):
        history_str += pyuvdata_version_str

    # Finally, return the completed string
    if check_origin:
        return history_str, pyuvdata_written
    else:
        return history_str


def write_ms_history(filepath, history=None, uvobj=None):
    """
    Parse the history into an MS history table.

    If the history string contains output from `_ms_hist_to_string`, parse that back
    into the MS history table.

    Parameters
    ----------
    filepath : str
        path to MS (without HISTORY suffix)
    history : str
        Required if uvobj is not given, a string containing the history to be written.
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the history string.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::HISTORY"

    if uvobj is not None:
        history = uvobj.history

    app_params = []
    cli_command = []
    application = []
    message = []
    obj_id = []
    obs_id = []
    origin = []
    priority = []
    times = []
    ms_history = "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE" in history

    if ms_history:
        # this history contains info from an MS history table. Need to parse it.

        ms_header_line_no = None
        ms_end_line_no = None
        pre_ms_history_lines = []
        post_ms_history_lines = []
        for line_no, line in enumerate(history.splitlines()):
            if not ms_history:
                continue

            if "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE" in line:
                ms_header_line_no = line_no
                # we don't need this line anywhere below so continue
                continue

            if "End measurement set history" in line:
                ms_end_line_no = line_no
                # we don't need this line anywhere below so continue
                continue

            if ms_header_line_no is not None and ms_end_line_no is None:
                # this is part of the MS history block. Parse it.
                line_parts = line.split(";")
                if len(line_parts) != 9:
                    # If the line has the wrong number of elements, then the history
                    # is mangled and we shouldn't try to parse it -- just record
                    # line-by-line as we do with any other pyuvdata history.
                    warnings.warn(
                        "Failed to parse prior history of MS file, "
                        "switching to standard recording method."
                    )
                    pre_ms_history_lines = post_ms_history_lines = []
                    ms_history = False
                    continue

                app_params.append(line_parts[0])
                cli_command.append(line_parts[1])
                application.append(line_parts[2])
                message.append(line_parts[3])
                obj_id.append(int(line_parts[4]))
                obs_id.append(int(line_parts[5]))
                origin.append(line_parts[6])
                priority.append(line_parts[7])
                times.append(np.float64(line_parts[8]))
            elif ms_header_line_no is None:
                # this is before the MS block
                if "Begin measurement set history" not in line:
                    pre_ms_history_lines.append(line)
            else:
                # this is after the MS block
                post_ms_history_lines.append(line)

        for line_no, line in enumerate(pre_ms_history_lines):
            app_params.insert(line_no, "")
            cli_command.insert(line_no, "")
            application.insert(line_no, "pyuvdata")
            message.insert(line_no, line)
            obj_id.insert(line_no, 0)
            obs_id.insert(line_no, -1)
            origin.insert(line_no, "pyuvdata")
            priority.insert(line_no, "INFO")
            times.insert(line_no, Time.now().mjd * 3600.0 * 24.0)

        for line in post_ms_history_lines:
            app_params.append("")
            cli_command.append("")
            application.append("pyuvdata")
            message.append(line)
            obj_id.append(0)
            obs_id.append(-1)
            origin.append("pyuvdata")
            priority.append("INFO")
            times.append(Time.now().mjd * 3600.0 * 24.0)

    if not ms_history:
        # no prior MS history detected in the history. Put all of our history in
        # the message column
        for line in history.splitlines():
            app_params.append("")
            cli_command.append("")
            application.append("pyuvdata")
            message.append(line)
            obj_id.append(0)
            obs_id.append(-1)
            origin.append("pyuvdata")
            priority.append("INFO")
            times.append(Time.now().mjd * 3600.0 * 24.0)

    tabledesc = tables.required_ms_desc("HISTORY")
    dminfo = tables.makedminfo(tabledesc)

    with tables.table(
        filepath, tabledesc=tabledesc, dminfo=dminfo, ack=False, readonly=False
    ) as history_table:
        nrows = len(message)
        history_table.addrows(nrows)

        # the first two lines below break on python-casacore < 3.1.0
        history_table.putcol("APP_PARAMS", np.asarray(app_params)[:, np.newaxis])
        history_table.putcol("CLI_COMMAND", np.asarray(cli_command)[:, np.newaxis])
        history_table.putcol("APPLICATION", application)
        history_table.putcol("MESSAGE", message)
        history_table.putcol("OBJECT_ID", obj_id)
        history_table.putcol("OBSERVATION_ID", obs_id)
        history_table.putcol("ORIGIN", origin)
        history_table.putcol("PRIORITY", priority)
        history_table.putcol("TIME", times)


def read_ms_observation(filepath):
    """
    Read Measurement Set OBSERVATION table.

    Parameters
    ----------
    filepath : str
        path to MS (without OBSERVATION suffix)

    Returns
    -------
    obs_dict : dict
        A dictionary containing observation information, including "telescope_name"
        (str with the name of the telescope), "observer" (str of observer name), and
        if present, "telescope_location" (ndarray of floats and shape (3,) describing
        the ITRF/MCMF 3D location in meters of the array center).

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/OBSERVATION")

    obs_dict = {}
    with tables.table(filepath + "/OBSERVATION", ack=False) as tb_obs:
        obs_dict["telescope_name"] = tb_obs.getcol("TELESCOPE_NAME")[0]
        obs_dict["observer"] = tb_obs.getcol("OBSERVER")[0]

        # check to see if a TELESCOPE_LOCATION column is present in the observation
        # table. This is non-standard, but inserted by pyuvdata
        if "TELESCOPE_LOCATION" in tb_obs.colnames():
            telescope_location = np.squeeze(tb_obs.getcol("TELESCOPE_LOCATION"))
            obs_dict["telescope_location"] = telescope_location

    return obs_dict


def write_ms_observation(
    filepath, uvobj=None, *, telescope_name=None, telescope_location=None, observer=None
):
    """
    Write out the observation information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without OBSERVATION suffix)
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here (with the exception of
        telescope_frame, which is pulled from the telescope_location UVParameter).
    telescope_name :str
        Required if uvobj not provided, name of the telescope.
    telescope_location : ndarray
        Required if uvobj not provided, 3D location (in ITRF/MCMF coordinates in meters)
        of the array center.
    observer : str
        Required if uvobj not provided, name of the observer.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::OBSERVATION"

    if uvobj is not None:
        telescope_name = uvobj.telescope.name
        telescope_location = uvobj.telescope._location.xyz()
        observer = telescope_name
        for key in uvobj.extra_keywords:
            if key.upper() == "OBSERVER":
                observer = uvobj.extra_keywords[key]

    tabledesc = tables.required_ms_desc("OBSERVATION")
    dminfo = tables.makedminfo(tabledesc)

    with tables.table(
        filepath, tabledesc=tabledesc, dminfo=dminfo, ack=False, readonly=False
    ) as observation_table:
        observation_table.addrows()
        observation_table.putcell("TELESCOPE_NAME", 0, telescope_name)

        # It appears that measurement sets do not have a concept of a telescope location
        # We add it here as a non-standard column in order to round trip it properly
        name_col_desc = tableutil.makearrcoldesc(
            "TELESCOPE_LOCATION", telescope_location[0], shape=[3], valuetype="double"
        )
        observation_table.addcols(name_col_desc)
        observation_table.putcell("TELESCOPE_LOCATION", 0, telescope_location)
        observation_table.putcell("OBSERVER", 0, observer)


def read_ms_spectral_window(filepath):
    """
    Read Measurement Set SPECTRAL_WINDOW table.

    Parameters
    ----------
    filepath : str
        path to MS (without SPECTRAL_WINDOW suffix)

    Returns
    -------
    spw_dict : dict
        Dictionary containing spectral window information, including "chan_freq"
        (ndarray of float and shape (Nspws,), center frequencies of the channels),
        "chan_width" (ndarray of float and shape (Nspws,), channel bandwidths),
        "num_chan" (ndarray of int and shape (Nspws,), number of channels per window),
        and "row_idx" (list of length Nspws with int elements, containing the table row
        number for each window).

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/SPECTRAL_WINDOW")

    with tables.table(filepath + "/SPECTRAL_WINDOW", ack=False) as tb_spws:
        n_rows = tb_spws.nrows()
        # The SPECTRAL_WINDOW table is a little special, in that some rows can
        # contain arrays of different shapes. For that reason, we'll pre-populate lists
        # for each element that we're interested in plugging things into.

        spw_dict = {
            "chan_freq": [None] * n_rows,
            "chan_width": [None] * n_rows,
            "num_chan": tb_spws.getcol("NUM_CHAN"),
            "row_idx": list(range(n_rows)),
        }

        try:
            spw_dict["assoc_spw_id"] = [
                int(idx[0]) for idx in tb_spws.getcol("ASSOC_SPW_ID")
            ]
            spw_dict["spoof_spw_id"] = False
        except RuntimeError:
            spw_dict["assoc_spw_id"] = np.arange(n_rows)
            spw_dict["spoof_spw"] = True

        for idx in range(n_rows):
            spw_dict["chan_freq"][idx] = tb_spws.getcell("CHAN_FREQ", idx)
            spw_dict["chan_width"][idx] = tb_spws.getcell("CHAN_WIDTH", idx)

    return spw_dict


def write_ms_spectral_window(
    filepath=None,
    uvobj=None,
    *,
    freq_array=None,
    channel_width=None,
    spw_array=None,
    id_array=None,
):
    """
    Write out the spectral information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without SPECTRAL_WINDOW suffix)
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here (with the exception of
        id_array, which is pulled from spw_array or flex_spw_id_array depending on the
        context).
    freq_array : ndarray
        Required if uvobj not provided, frequency centers for each channel. Expected
        shape is (Nfreqs,), type float.
    channel_width : ndarray or float
        Required if uvobj not provided, frequency centers for each channel. Expected
        to be of shape (Nfreqs,), type float.
    spw_array : ndarray
        Required if uvobj not provided, ID numbers for spectral windows.
    id_array : ndarray
        Map of how each entry in `freq_array` matches to the spectral windows in
        `spw_array`. Required if uvobj not provided.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::SPECTRAL_WINDOW"

    if uvobj is not None:
        if "_freq_range" in uvobj and uvobj.freq_range is not None:
            freq_array = np.mean(uvobj.freq_range, axis=-1)
            channel_width = np.squeeze(np.diff(uvobj.freq_range, axis=-1), axis=-1)
            id_array = np.array(uvobj.spw_array)
        else:
            freq_array = uvobj.freq_array
            channel_width = uvobj.channel_width
            id_array = uvobj.flex_spw_id_array

        spw_array = uvobj.spw_array

    # Construct a couple of columns we're going to use that are not part of
    # the MS v2.0 baseline format (though are useful for pyuvdata objects).
    tabledesc = tables.required_ms_desc("SPECTRAL_WINDOW")
    extended_desc = tables.complete_ms_desc("SPECTRAL_WINDOW")
    tabledesc["ASSOC_SPW_ID"] = extended_desc["ASSOC_SPW_ID"]
    tabledesc["ASSOC_NATURE"] = extended_desc["ASSOC_NATURE"]
    dminfo = tables.makedminfo(tabledesc)

    with tables.table(
        filepath, tabledesc=tabledesc, dminfo=dminfo, ack=False, readonly=False
    ) as sw_table:
        for idx, spw_id in enumerate(spw_array):
            ch_mask = ... if id_array is None else id_array == spw_id
            sw_table.addrows()
            sw_table.putcell("NUM_CHAN", idx, np.sum(ch_mask))
            sw_table.putcell("NAME", idx, f"SPW{spw_id}")
            sw_table.putcell("ASSOC_SPW_ID", idx, spw_id)
            sw_table.putcell("ASSOC_NATURE", idx, "")  # Blank for now
            sw_table.putcell("CHAN_FREQ", idx, freq_array[ch_mask])
            sw_table.putcell("CHAN_WIDTH", idx, channel_width[ch_mask])
            sw_table.putcell("EFFECTIVE_BW", idx, channel_width[ch_mask])
            sw_table.putcell("TOTAL_BANDWIDTH", idx, np.sum(channel_width[ch_mask]))
            sw_table.putcell("RESOLUTION", idx, channel_width[ch_mask])
            # TODO: These are placeholders for now, but should be replaced with
            # actual frequency reference info (once pyuvdata handles that)
            sw_table.putcell("MEAS_FREQ_REF", idx, VEL_DICT["TOPO"])
            sw_table.putcell("REF_FREQUENCY", idx, freq_array[0])


def read_ms_feed(filepath, select_ants=None):
    """
    Read Measurement Set FEED table.

    Note that this method is not yet implemented, and is a placeholder for future
    development.

    Parameters
    ----------
    filepath : str
        path to MS (without FEED suffix)

    Returns
    -------
    feed_dict : dict
        Dictionary containing feed information.

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/FEED")
    filepath += "::FEED"

    with tables.table(filepath, ack=False) as feed_table:
        if "pyuvdata_has_feed" in feed_table.getkeywords() and not (
            feed_table.getkeyword("pyuvdata_has_feed")
        ):
            feed_array = feed_angle = Nfeeds = antenna_numbers = None
        else:
            Nfeeds = feed_table.getcol("NUM_RECEPTORS")
            if not all(max(Nfeeds) == Nfeeds):
                # This seems like a rare possibility, but screen for it here.
                raise ValueError(  # pragma: no cover
                    "Support for differing numbers of feeds is not supported. Please "
                    "file an issue in our GitHub issue log so that we can add support "
                    "for it."
                )
            Nfeeds = int(Nfeeds[0])
            ant_arr = feed_table.getcol("ANTENNA_ID")
            pol_arr = np.char.lower(feed_table.getcol("POLARIZATION_TYPE")["array"])
            pol_arr = pol_arr.reshape(feed_table.getcol("POLARIZATION_TYPE")["shape"])
            rang_arr = feed_table.getcol("RECEPTOR_ANGLE")
            max_ant = max(ant_arr) + 1
            feed_array = np.full((max_ant, Nfeeds), "", dtype=np.object_)
            feed_angle = np.zeros((max_ant, Nfeeds), dtype=float)
            antenna_numbers = np.arange(max_ant)

            feed_array[ant_arr] = pol_arr
            feed_angle[ant_arr] = rang_arr

            # Set default case to be lower to be consistent w/ pyuvdata standards
            feed_array.flat[:] = [item.lower() for item in feed_array.flat]

    if select_ants is not None and Nfeeds is not None:
        mask = np.isin(np.arange(max_ant), select_ants)
        feed_array = feed_array[mask]
        feed_angle = feed_angle[mask]
        antenna_numbers = select_ants

    tb_feed_dict = {
        "feed_array": feed_array,
        "feed_angle": feed_angle,
        "Nfeeds": Nfeeds,
        "antenna_numbers": antenna_numbers,
    }

    return tb_feed_dict


def write_ms_feed(
    filepath,
    uvobj=None,
    nfeeds=None,
    nspws=None,
    feed_array=None,
    feed_angle=None,
    antenna_numbers=None,
    time_val=None,
):
    """
    Write out the feed information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without FEED suffix)
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here.
    nfeeds : int
        Required if uvobj not provided, number of feeds in the telescope for the object.
    feed_array : ndarray of str
        Required if uvobj not provided, describes the polarization of each receiver,
        should be one of "X", "Y", "L", or "R". Shape (nants, nfeeds).
    feed_angle : ndarray of float
        Required if uvobj not provided, orientation of the receiver with respect to the
        great circle at fixed azimuth, shape (nants, nfeeds).
    antenna_numbers : array-like of int
        Required if uvobj not provided, antenna numbers for all antennas of the
        telescope, dtype int and shape (nants,).
    time_val : float
        Required if uvobj is not supplied, representative JD date for the catalog to
        be recorded into the MS file.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::FEED"

    has_feed = feed_array is None
    if uvobj is not None:
        if uvobj.telescope.feed_array is not None:
            feed_array = uvobj.telescope.feed_array
            feed_angle = uvobj.telescope.feed_angle
            nfeeds = int(uvobj.telescope.Nfeeds)
            has_feed = True
        else:
            if uvobj.flex_spw_polarization_array is None:
                pols = uvobj.polarization_array
            else:
                pols = uvobj.flex_spw_polarization_array

            feed_pols = utils.pol.get_feeds_from_pols(pols)
            nfeeds = len(feed_pols)
            feed_array = np.tile(sorted(feed_pols), (uvobj.telescope.Nants, 1))
            feed_angle = np.zeros((uvobj.telescope.Nants, nfeeds))
            has_feed = False

        antenna_numbers = uvobj.telescope.antenna_numbers
        nspws = uvobj.Nspws
        time_val = (
            Time(np.median(uvobj.time_array), format="jd", scale="utc").mjd * 86400.0
        )

    nrows = np.max(antenna_numbers) + 1
    antenna_id_table = np.arange(nrows, dtype=np.int32)

    with tables.table(filepath, ack=False, readonly=False) as feed_table:
        # Record whether or not we actually have the feed information specified, versus
        # deriving it from a polarization table
        feed_table.putkeyword("pyuvdata_has_feed", has_feed)

        # Plug in what we need here. Tile based on the first element to plug in valid
        # entries for all antennas so that CASA doesn't complain.
        pol_type_table = np.tile(feed_array[0], (nrows, 1)).astype("<U1")
        pol_type_table[antenna_numbers] = feed_array
        pol_type_table = np.char.upper(pol_type_table)
        receptor_angle_table = np.zeros((nrows, nfeeds), dtype=np.float64)
        receptor_angle_table[antenna_numbers] = feed_angle

        for idx in range(nrows):
            if pol_type_table[idx].tolist() in [["Y", "X"], ["L", "R"]]:
                pol_type_table[idx, :] = pol_type_table[idx, ::-1]
                receptor_angle_table[idx, :] = receptor_angle_table[idx, ::-1]

        pol_type_table = np.repeat(pol_type_table, nspws, axis=0)
        receptor_angle_table = np.repeat(receptor_angle_table, nspws, axis=0)
        antenna_id_table = np.repeat(antenna_id_table, nspws, axis=0)
        spectral_window_id_table = np.tile(np.arange(nspws, dtype=np.int32), nrows)
        num_receptors_table = np.full(nrows * nspws, nfeeds, dtype=np.int32)

        # These are all "sensible defaults" for now.
        beam_id_table = -1 * np.ones(nrows * nspws, dtype=np.int32)
        beam_offset_table = np.zeros((nrows * nspws, 2, 2), dtype=np.float64)
        feed_id_table = np.zeros(nrows * nspws, dtype=np.int32)
        time_table = np.full(nrows * nspws, time_val, dtype=np.float64)
        interval_table = np.full(nrows * nspws, np.finfo(float).max, dtype=np.float64)
        position_table = np.zeros((nrows * nspws, 3), dtype=np.float64)

        # TODO: Check and see if this needs additional info for polcal...
        pol_response_table = np.zeros((nspws * nrows, 2, 2), dtype=np.complex64)

        feed_table.addrows(nrows * nspws)
        feed_table.putcol("ANTENNA_ID", antenna_id_table)
        feed_table.putcol("BEAM_ID", beam_id_table)
        feed_table.putcol("BEAM_OFFSET", beam_offset_table)
        feed_table.putcol("FEED_ID", feed_id_table)
        feed_table.putcol("TIME", time_table)
        feed_table.putcol("INTERVAL", interval_table)
        feed_table.putcol("NUM_RECEPTORS", num_receptors_table)
        feed_table.putcol("POLARIZATION_TYPE", pol_type_table)
        feed_table.putcol("POL_RESPONSE", pol_response_table)
        feed_table.putcol("POSITION", position_table)
        feed_table.putcol("RECEPTOR_ANGLE", receptor_angle_table)
        feed_table.putcol("SPECTRAL_WINDOW_ID", spectral_window_id_table)


def read_ms_source(filepath):
    """
    Read Measurement Set SOURCE table.

    Parameters
    ----------
    filepath : str
        path to MS (without SOURCE suffix)

    Returns
    -------
    source_dict : dict
        A dictionary with keys matched to the source ID, and up to six values which
        map to specific keys in entries of the `phase_center_catalog` parameter
        found in UVData/UVCal objects. They are "cat_lon" (longitudinal position in
        radians), "cat_lat" (latitudinal position in radians), "cat_times"
        (JD dates of ephemeris entries), "cat_type" (type of catalog entry), "cat_pm_ra"
        (proper motion in RA), and "cat_pm_dec" (proper motion in Dec). These entries
        can be used to update keys in `phase_center_catalog` as derived from the FIELD
        table, which only has space to record a single RA/Dec coordinate plus reference
        frame information.

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/SOURCE")

    tb_sou_dict = {}
    with tables.table(filepath + "/SOURCE", ack=False) as tb_source:
        for idx in range(tb_source.nrows()):
            sou_id = tb_source.getcell("SOURCE_ID", idx)
            pm_vec = tb_source.getcell("PROPER_MOTION", idx)
            time_stamp = tb_source.getcell("TIME", idx)
            sou_vec = tb_source.getcell("DIRECTION", idx)
            try:
                for idx in np.where(
                    np.isclose(
                        tb_sou_dict[sou_id]["cat_times"], time_stamp, rtol=0, atol=1e-3
                    )
                )[0]:
                    if not (
                        (tb_sou_dict[sou_id]["cat_lon"][idx] == sou_vec[0])
                        and (tb_sou_dict[sou_id]["cat_lat"][idx] == sou_vec[1])
                        and (tb_sou_dict[sou_id]["cat_pm_ra"][idx] == pm_vec[0])
                        and (tb_sou_dict[sou_id]["cat_pm_dec"][idx] == pm_vec[1])
                    ):
                        warnings.warn(
                            "Different windows in this MS file contain different "
                            "metadata for the same integration. Be aware that "
                            "UVData objects do not allow for this, and thus will "
                            "default to using the metadata from the last row read "
                            "from the SOURCE table." + reporting_request
                        )
                    _ = tb_sou_dict[sou_id]["cat_times"].pop(idx)
                    _ = tb_sou_dict[sou_id]["cat_lon"].pop(idx)
                    _ = tb_sou_dict[sou_id]["cat_lat"].pop(idx)
                    _ = tb_sou_dict[sou_id]["cat_pm_ra"].pop(idx)
                    _ = tb_sou_dict[sou_id]["cat_pm_dec"].pop(idx)
                tb_sou_dict[sou_id]["cat_times"].append(time_stamp)
                tb_sou_dict[sou_id]["cat_lon"].append(sou_vec[0])
                tb_sou_dict[sou_id]["cat_lat"].append(sou_vec[1])
                tb_sou_dict[sou_id]["cat_pm_ra"].append(pm_vec[0])
                tb_sou_dict[sou_id]["cat_pm_dec"].append(pm_vec[1])
            except KeyError:
                tb_sou_dict[sou_id] = {
                    "cat_times": [time_stamp],
                    "cat_lon": [sou_vec[0]],
                    "cat_lat": [sou_vec[1]],
                    "cat_pm_ra": [pm_vec[0]],
                    "cat_pm_dec": [pm_vec[1]],
                }

    for cat_dict in tb_sou_dict.values():
        make_arr = len(cat_dict["cat_times"]) != 1
        if make_arr:
            # Convert CASA time to JD (pyuvdata standard)
            cat_dict["cat_times"] = Time(
                np.array(cat_dict["cat_times"]) / 86400.0, format="mjd", scale="utc"
            ).jd
            cat_dict["cat_type"] = "ephem"
        else:
            del cat_dict["cat_times"]

        for key in cat_dict:
            if make_arr:
                cat_dict[key] = np.array(cat_dict[key])
            else:
                cat_dict[key] = cat_dict[key][0]
        if np.allclose(cat_dict["cat_pm_ra"], 0) and np.allclose(
            cat_dict["cat_pm_dec"], 0
        ):
            cat_dict["cat_pm_ra"] = cat_dict["cat_pm_dec"] = None

    return tb_sou_dict


def write_ms_source(filepath, uvobj=None, time_default=None, phase_center_catalog=None):
    """
    Write out the source information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without SOURCE suffix)
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here (with the exception of
        time_default, which is derived from the time_array UVParameter).
    time_default : float
        Default time (in MJD seconds) to use for the catalog entries.
    phase_center_catalog : dict
        A catalog stylized like the UVData/UVCal `phase_center_catalog` parameters (see
        documentation of those classes for more details on expected structure). Required
        if uvobj is not supplied.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::SOURCE"

    if uvobj is not None:
        time_default = (
            Time(np.median(uvobj.time_array), format="jd", scale="utc").mjd * 86400.0
        )
        phase_center_catalog = uvobj.phase_center_catalog

    # Handle this table a bit specially, since it's not a strictly required table
    source_desc = tables.complete_ms_desc("SOURCE")
    dminfo = tables.makedminfo(source_desc)

    with tables.table(
        filepath, tabledesc=source_desc, dminfo=dminfo, ack=False, readonly=False
    ) as source_table:
        # Make the default time be the midpoint of the obs
        # Default interval should be several times a Hubble time. Gotta make this code
        # future-proofed until the eventual heat death of the Universe.
        int_default = np.finfo(float).max

        row_count = 0
        for sou_id, pc_dict in phase_center_catalog.items():
            # Get some pieces of info that should not depend on the cat type, like name,
            # proper motions, (others possible)
            int_val = int_default
            sou_name = pc_dict["cat_name"]
            pm_dir = np.array(
                [pc_dict.get("cat_pm_ra"), pc_dict.get("cat_pm_ra")], dtype=np.double
            )
            if not np.all(np.isfinite(pm_dir)):
                pm_dir[:] = 0.0

            if pc_dict["cat_type"] == "sidereal":
                # If this is just a single set of points, set up values to have shape
                # (1, X) so that we can iterate through them later.
                sou_dir = np.array([[pc_dict["cat_lon"], pc_dict["cat_lat"]]])
                time_val = np.array([time_default])
            elif pc_dict["cat_type"] == "ephem":
                # Otherwise for ephem, make time the outer-most axis so that we
                # can easily iterate through.
                sou_dir = np.vstack((pc_dict["cat_lon"], pc_dict["cat_lat"])).T
                time_val = (
                    Time(pc_dict["cat_times"], format="jd", scale="utc").mjd * 86400.0
                ).flatten()
                # If there are multiple time entries, then approximate a value for the
                # interval (needed for CASA, not UVData) by taking the range divided
                # by the number of points in the ephem.
                if len(time_val) > 1:
                    int_val = (time_val.max() - time_val.min()) / (len(time_val) - 1)

            for idx in range(len(sou_dir)):
                source_table.addrows()
                source_table.putcell("NAME", row_count, sou_name)
                source_table.putcell("SOURCE_ID", row_count, sou_id)
                source_table.putcell("INTERVAL", row_count, int_val)
                source_table.putcell("SPECTRAL_WINDOW_ID", row_count, -1)
                source_table.putcell("NUM_LINES", row_count, 0)
                source_table.putcell("TIME", row_count, time_val[idx])
                source_table.putcell("DIRECTION", row_count, sou_dir[idx])
                source_table.putcell("PROPER_MOTION", row_count, pm_dir)
                row_count += 1


def read_ms_pointing(filepath):
    """
    Read Measurement Set POINTING table.

    Note that this method is not yet implemented, and is a placeholder for future
    development.

    Parameters
    ----------
    filepath : str
        path to MS (without POINTING suffix)

    Returns
    -------
    pointing_dict : dict
        Dictionary containing pointing information.

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/POINTING")

    raise NotImplementedError("Reading of MS POINTING tables not available yet.")


def write_ms_pointing(
    filepath, uvobj=None, max_ant=None, integration_time=None, time_array=None
):
    """
    Write out the pointing information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without POINTING suffix)
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here (with the exception of
        max_ant, which is derived from from the antenna_numbers UVParameter).
    max_ant : int
        Required if uvobj not provided, the highest-number antenna for the telescope.
    integration_time : ndarray
        Required if uvobj not provided, integration time per entry. Ndarray of float,
        should match in shape to `time_array`.
    time_array : ndarray
        Required if uvobj not provided, JD date of each entry. Ndarray of float, should
        match in shape to `integration_time`.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::POINTING"

    if uvobj is not None:
        max_ant = np.max(uvobj.telescope.antenna_numbers)
        integration_time = uvobj.integration_time
        time_array = uvobj.time_array

    with tables.table(filepath, ack=False, readonly=False) as pointing_table:
        nants_casa = max_ant + 1
        antennas = np.arange(nants_casa, dtype=np.int32)

        # We want the number of unique timestamps here, since CASA wants a
        # per-timestamp, per-antenna entry
        unique_times = np.unique(time_array)
        times = Time(unique_times, format="jd", scale="utc").mjd * 86400.0
        ntimes = len(times)

        # Same for the number of intervals
        intervals = np.zeros_like(times)
        for idx, ind_time in enumerate(unique_times):
            intervals[idx] = np.median(integration_time[time_array == ind_time])

        nrows = nants_casa * ntimes

        # This extra step of adding a single row and plugging in values for DIRECTION
        # and TARGET are a workaround for a weird issue that pops up where, because the
        # values are not a fixed size (they are shape(2, Npoly), where Npoly is allowed
        # to vary from integration to integration), casacore seems to be very slow
        # filling in the data after the fact. By "pre-filling" the first row, the
        # addrows operation will automatically fill in an appropriately shaped array.
        # TODO: This works okay for steerable dishes, but less well for non-tracking
        # arrays (i.e., where primary beam center != phase center). Fix later.

        pointing_table.addrows(1)
        pointing_table.putcell("DIRECTION", 0, np.zeros((2, 1), dtype=np.float64))
        pointing_table.putcell("TARGET", 0, np.zeros((2, 1), dtype=np.float64))
        pointing_table.addrows(nrows - 1)

        pointing_table.putcol("ANTENNA_ID", np.tile(antennas, ntimes))
        pointing_table.putcol("TIME", np.repeat(times, nants_casa))
        pointing_table.putcol("INTERVAL", np.repeat(intervals, nants_casa))

        name_col = np.asarray(["ZENITH"] * nrows, dtype=np.bytes_)
        pointing_table.putcol("NAME", name_col, nrow=nrows)

        pointing_table.putcol("NUM_POLY", np.zeros(nrows, dtype=np.int32))
        pointing_table.putcol("TIME_ORIGIN", np.repeat(times, nants_casa))

        # we always "point" at zenith
        # TODO: Fix this for steerable arrays
        direction_col = np.zeros((nrows, 2, 1), dtype=np.float64)
        direction_col[:, 1, :] = np.pi / 2
        pointing_table.putcol("DIRECTION", direction_col)

        # just reuse "direction" for "target"
        pointing_table.putcol("TARGET", direction_col)

        # set tracking info to `False`
        pointing_table.putcol("TRACKING", np.zeros(nrows, dtype=bool))


def read_ms_polarization(filepath):
    """
    Read Measurement Set POLARIZATION table.

    Parameters
    ----------
    filepath : str
        path to MS (without POLARIZATION suffix)

    Returns
    -------
    pol_dict : dict
        A dictionary with keys matched to the polarization ID, with values "corr_type"
        (ndarray of int, shape (Npols,), matched to polarization code) and "num_corr"
        (ndarray of int, shape (Npols,), tallying the total number of polarization
        entries per record).

    Raises
    ------
    FileNotFoundError
        If no MS file is found with the provided name.
    """
    _ms_utils_call_checks(filepath + "/POLARIZATION")

    with tables.table(filepath + "/POLARIZATION", ack=False) as tb_pol:
        pol_dict = {}
        for pol_id in range(tb_pol.nrows()):
            pol_dict[pol_id] = {
                "corr_type": tb_pol.getcell("CORR_TYPE", pol_id).astype(int),
                "num_corr": tb_pol.getcell("NUM_CORR", pol_id),
            }
    return pol_dict


def write_ms_polarization(
    filepath, *, pol_order=..., uvobj=None, polarization_array=None, flex_pol=False
):
    """
    Write out the polarization information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without POLARIZATION suffix)
    pol_order : slice or list of int
        Ordering of the polarization axis on write, only used if not writing a
        flex-pol dataset.
    uvobj : UVBase (with matching parameters)
        Optional parameter, can be used to automatically fill the other required
        keywords for this function. Note that the UVBase object must have parameters
        that match by name to the other keywords required here (with the exception of
        flex_pol, which is derived from the presence of the flex_spw_polarization array
        UVParameter).
    polarization_array : ndarray
        Required if uvobj not provided, array containing polarization ID codes (ndarray
        of dtype int and shape (Npols,) if flex_pol=False, otherwise shape (Nspws,)).
    flex_pol : bool
        Required if uvobj not provided, whether or not the supplied polarization_array
        is derived for a flex-pol object (can contain duplicate entries of codes, with
        positions indexed against entries in the spectral window table). Default is
        False.

    Raises
    ------
    FileNotFoundError
        If no main MS table is found when looking at filepath.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::POLARIZATION"

    if uvobj is not None:
        flex_pol = uvobj.flex_spw_polarization_array is not None
        if flex_pol:
            polarization_array = uvobj.flex_spw_polarization_array
        else:
            polarization_array = uvobj.polarization_array

    pol_arr = polarization_array if flex_pol else polarization_array[pol_order]

    with tables.table(filepath, ack=False, readonly=False) as pol_table:
        if flex_pol:
            for idx, spw_pol in enumerate(np.unique(pol_arr)):
                pol_str = utils.polnum2str([spw_pol])
                feed_pols = {
                    feed for pol in pol_str for feed in utils.pol.POL_TO_FEED_DICT[pol]
                }
                pol_types = [pol.lower() for pol in sorted(feed_pols)]
                pol_tuples = np.asarray(
                    [(pol_types.index(i), pol_types.index(j)) for i, j in pol_str],
                    dtype=np.int32,
                )

                pol_table.addrows()
                pol_table.putcell(
                    "CORR_TYPE", idx, np.array([POL_AIPS2CASA_DICT[spw_pol]])
                )
                pol_table.putcell("CORR_PRODUCT", idx, pol_tuples)
                pol_table.putcell("NUM_CORR", idx, 1)
        else:
            pol_str = utils.polnum2str(pol_arr)
            feed_pols = {
                feed for pol in pol_str for feed in utils.pol.POL_TO_FEED_DICT[pol]
            }
            pol_types = [pol.lower() for pol in sorted(feed_pols)]
            pol_tuples = np.asarray(
                [(pol_types.index(i), pol_types.index(j)) for i, j in pol_str],
                dtype=np.int32,
            )

            pol_table.addrows()
            pol_table.putcell(
                "CORR_TYPE", 0, np.array([POL_AIPS2CASA_DICT[pol] for pol in pol_arr])
            )
            pol_table.putcell("CORR_PRODUCT", 0, pol_tuples)
            pol_table.putcell("NUM_CORR", 0, len(polarization_array))


def init_ms_file(filepath, make_model_col=False, make_corr_col=False):
    """
    Create a skeleton MS dataset to fill.

    Parameters
    ----------
    filepath : str
        Path to MS to be created.
    make_model_col : bool
        If set to True, will construct a measurement set that contains a MODEL_DATA
        column in addition to the DATA column. Default is False.
    make_model_col : bool
        If set to True, will construct a measurement set that contains a CORRECTED_DATA
        column in addition to the DATA column. Default is False.

    Returns
    -------
    ms_table : casacore Table
        Table object linked to the newly created MS file.
    """
    # The required_ms_desc returns the defaults for a CASA MS table
    ms_desc = tables.required_ms_desc("MAIN")

    # The tables have a different choice of dataManagerType and dataManagerGroup
    # based on a test ALMA dataset and comparison with what gets generated with
    # a dataset that comes through importuvfits.
    ms_desc["FLAG"].update(
        dataManagerType="TiledShapeStMan", dataManagerGroup="TiledFlag"
    )

    ms_desc["UVW"].update(
        dataManagerType="TiledColumnStMan", dataManagerGroup="TiledUVW"
    )
    # TODO: Can stuff UVFLAG objects into this
    ms_desc["FLAG_CATEGORY"].update(
        dataManagerType="TiledShapeStMan",
        dataManagerGroup="TiledFlagCategory",
        keywords={"CATEGORY": np.array("baddata")},
    )
    ms_desc["WEIGHT"].update(
        dataManagerType="TiledShapeStMan", dataManagerGroup="TiledWgt"
    )
    ms_desc["SIGMA"].update(
        dataManagerType="TiledShapeStMan", dataManagerGroup="TiledSigma"
    )

    # The ALMA default for the next set of columns from the MAIN table use the
    # name of the column as the dataManagerGroup, so we update the casacore
    # defaults accordingly.
    for key in ["ANTENNA1", "ANTENNA2", "DATA_DESC_ID", "FLAG_ROW"]:
        ms_desc[key].update(dataManagerGroup=key)

    # The ALMA default for he next set of columns from the MAIN table use the
    # IncrementalStMan dataMangerType, and so we update the casacore defaults
    # (along with the name dataManagerGroup name to the column name, which is
    # also the apparent default for ALMA).
    incremental_list = [
        "ARRAY_ID",
        "EXPOSURE",
        "FEED1",
        "FEED2",
        "FIELD_ID",
        "INTERVAL",
        "OBSERVATION_ID",
        "PROCESSOR_ID",
        "SCAN_NUMBER",
        "STATE_ID",
        "TIME",
        "TIME_CENTROID",
    ]
    for key in incremental_list:
        ms_desc[key].update(dataManagerType="IncrementalStMan", dataManagerGroup=key)

    # TODO: Verify that the casacore defaults for coldesc are satisfactory for
    # the tables and columns below (note that these were columns with apparent
    # discrepancies between a test ALMA dataset and what casacore generated).
    # FEED:FOCUS_LENGTH
    # FIELD
    # POINTING:TARGET
    # POINTING:POINTING_OFFSET
    # POINTING:ENCODER
    # POINTING:ON_SOURCE
    # POINTING:OVER_THE_TOP
    # SPECTRAL_WINDOW:BBC_NO
    # SPECTRAL_WINDOW:ASSOC_SPW_ID
    # SPECTRAL_WINDOW:ASSOC_NATURE
    # SPECTRAL_WINDOW:SDM_WINDOW_FUNCTION
    # SPECTRAL_WINDOW:SDM_NUM_BIN

    # Create a column for the data, which is amusingly enough not actually
    # creaed by default.
    datacoldesc = tables.makearrcoldesc(
        "DATA",
        0.0 + 0.0j,
        valuetype="complex",
        ndim=2,
        datamanagertype="TiledShapeStMan",
        datamanagergroup="TiledData",
        comment="The data column",
    )
    del datacoldesc["desc"]["shape"]
    ms_desc.update(tables.maketabdesc(datacoldesc))

    if make_model_col:
        datacoldesc = tables.makearrcoldesc(
            "MODEL_DATA",
            0.0 + 0.0j,
            valuetype="complex",
            ndim=2,
            datamanagertype="TiledShapeStMan",
            datamanagergroup="TiledData",
            comment="The data column",
        )
        del datacoldesc["desc"]["shape"]
        ms_desc.update(tables.maketabdesc(datacoldesc))

    if make_corr_col:
        datacoldesc = tables.makearrcoldesc(
            "CORRECTED_DATA",
            0.0 + 0.0j,
            valuetype="complex",
            ndim=2,
            datamanagertype="TiledShapeStMan",
            datamanagergroup="TiledData",
            comment="The data column",
        )
        del datacoldesc["desc"]["shape"]
        ms_desc.update(tables.maketabdesc(datacoldesc))

    # Now create a column for the weight spectrum, which we plug nsample_array into
    weightspeccoldesc = tables.makearrcoldesc(
        "WEIGHT_SPECTRUM",
        0.0,
        valuetype="float",
        ndim=2,
        datamanagertype="TiledShapeStMan",
        datamanagergroup="TiledWgtSpectrum",
        comment="Weight for each data point",
    )
    del weightspeccoldesc["desc"]["shape"]

    ms_desc.update(tables.maketabdesc(weightspeccoldesc))

    # Finally, create the dataset, and return a handle to the freshly created
    # skeleton measurement set.
    return tables.default_ms(filepath, ms_desc, tables.makedminfo(ms_desc))


def init_ms_cal_file(filename, delay_table=False):
    """
    Create a skeleton MS calibration table to fill.

    Parameters
    ----------
    filepath : str
        Path to MS table to be created.
    delay_table : bool
        Set to False by default, which will create a MS table capable of storing
        complex gains. However, if set to True, the method will instead construct a
        table which can store delay information.
    """
    standard_desc = tables.required_ms_desc()
    tabledesc = {}
    tabledesc["TIME"] = standard_desc["TIME"]
    tabledesc["FIELD_ID"] = standard_desc["FIELD_ID"]
    tabledesc["ANTENNA1"] = standard_desc["ANTENNA1"]
    tabledesc["ANTENNA2"] = standard_desc["ANTENNA2"]
    tabledesc["INTERVAL"] = standard_desc["INTERVAL"]
    tabledesc["EXPOSURE"] = standard_desc["EXPOSURE"]  # Used to track int time
    tabledesc["SCAN_NUMBER"] = standard_desc["SCAN_NUMBER"]
    tabledesc["OBSERVATION_ID"] = standard_desc["OBSERVATION_ID"]
    # This is kind of a weird aliasing that's done for tables -- may not be always true,
    # but this seems to be needed as of now (circa 2024).
    tabledesc["SPECTRAL_WINDOW_ID"] = standard_desc["DATA_DESC_ID"]

    for field in tabledesc:
        # Option seems to be set to 5 for the above fields, based on CASA testing
        tabledesc[field]["option"] = 5

    # FLAG and weight are _mostly_ standard, just needs ndim modified
    tabledesc["FLAG"] = standard_desc["FLAG"]
    tabledesc["WEIGHT"] = standard_desc["WEIGHT"]
    tabledesc["FLAG"]["ndim"] = tabledesc["WEIGHT"]["ndim"] = -1

    # PARAMERR and SNR are very similar to SIGMA, so we'll boot-strap from it, with
    # the comments just being updated
    tabledesc["PARAMERR"] = standard_desc["SIGMA"]
    tabledesc["SNR"] = standard_desc["SIGMA"]
    tabledesc["SNR"]["ndim"] = tabledesc["PARAMERR"]["ndim"] = -1
    tabledesc["SNR"]["comment"] = "Signal-to-noise of the gain solution."
    tabledesc["PARAMERR"]["comment"] = "Uncertainty in the gains."

    tabledesc["FPARAM" if delay_table else "CPARAM"] = tables.makearrcoldesc(
        None,
        None,
        valuetype="float" if delay_table else "complex",
        ndim=-1,
        datamanagertype="StandardStMan",
        comment="Delay values." if delay_table else "Complex gain values.",
    )["desc"]
    del tabledesc["FPARAM" if delay_table else "CPARAM"]["shape"]

    for field in tabledesc:
        tabledesc[field]["dataManagerGroup"] = "MSMTAB"

    dminfo = tables.makedminfo(tabledesc)

    with tables.table(
        filename, tabledesc=tabledesc, dminfo=dminfo, ack=False, readonly=False
    ) as ms:
        # Put some general stuff into the top level dict, default to wideband gains.
        ms.putinfo(
            {
                "type": "Calibration",
                "subType": "K Jones" if delay_table else "G Jones",
                "readme": f"Written with pyuvdata version: {__version__}.",
            }
        )
        # Finally, set up some de
        ms.putkeyword("ParType", "Float" if delay_table else "Complex")
        ms.putkeyword("MSName", "unknown")
        ms.putkeyword("VisCal", "unknown")
        ms.putkeyword("PolBasis", "unknown")
        ms.putkeyword("CASA_Version", "unknown")


def get_ms_telescope_location(*, tb_ant_dict, obs_dict):
    """
    Get the telescope location object.

    Parameters
    ----------
    tb_ant_dict : dict
        dict returned by `read_ms_antenna`
    obs_dict : dict
        dict returned by `read_ms_observation`

    """
    xyz_telescope_frame = tb_ant_dict["telescope_frame"]
    xyz_telescope_ellipsoid = tb_ant_dict["telescope_ellipsoid"]

    # check to see if a TELESCOPE_LOCATION column is present in the observation
    # table. This is non-standard, but inserted by pyuvdata
    if "telescope_location" not in obs_dict and (
        obs_dict["telescope_name"] in known_telescopes()
        or obs_dict["telescope_name"].upper() in known_telescopes()
    ):
        # get it from known telescopes
        telname = obs_dict["telescope_name"]
        if telname.upper() in known_telescopes():
            telname = telname.upper()

        telescope_loc = known_telescope_location(telname)
        warnings.warn(
            f"Setting telescope_location to value in known_telescopes for {telname}."
        )
        return telescope_loc
    else:
        if xyz_telescope_frame == "mcmf":
            try:
                from lunarsky import MoonLocation
            except ImportError as ie:  # pragma: no cover
                # There is a test for this, but it is always skipped with our
                # current CI setup because it requires that python-casacore is
                # installed but lunarsky isn't. Doesn't seem worth setting up a
                # whole separate CI for this.
                raise ValueError(  # pragma: no cover
                    "Need to install `lunarsky` package to work with MCMF frames."
                ) from ie

        if "telescope_location" in obs_dict:
            if xyz_telescope_frame == "itrs":
                return EarthLocation.from_geocentric(
                    *np.squeeze(obs_dict["telescope_location"]), unit="m"
                )
            else:
                loc = MoonLocation.from_selenocentric(
                    *np.squeeze(obs_dict["telescope_location"]), unit="m"
                )
                loc.ellipsoid = xyz_telescope_ellipsoid
                return loc
        else:
            # Set it to be the mean of the antenna positions (this is not ideal!)
            if xyz_telescope_frame == "itrs":
                return EarthLocation.from_geocentric(
                    *np.array(np.mean(tb_ant_dict["antenna_positions"], axis=0)),
                    unit="m",
                )
            else:
                loc = MoonLocation.from_selenocentric(
                    *np.array(np.mean(tb_ant_dict["antenna_positions"], axis=0)),
                    unit="m",
                )
                loc.ellipsoid = xyz_telescope_ellipsoid
                return loc
