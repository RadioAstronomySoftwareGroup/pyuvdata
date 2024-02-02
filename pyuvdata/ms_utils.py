# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with MeasurementSet files."""

import warnings

import numpy as np
from astropy.time import Time

from . import utils as uvutils

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
casa_error = None
try:
    import casacore.tables as tables
except ImportError as error:  # pragma: no cover
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


def _ms_utils_call_checks(filepath):
    # Check for casa.
    if not casa_present:
        raise ImportError(no_casa_message) from casa_error

    if not tables.tableexists(filepath):
        raise FileNotFoundError(
            filepath + " not found or not recognized as an MS table."
        )


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


def read_ms_hist(filepath, pyuvdata_version_str, check_origin=False):
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

    Returns
    -------
    str
        string encoding complete casa history table converted with a new
        line denoting rows and a ';' denoting column breaks.
    pyuvdata_written :  bool
        boolean indicating whether or not this MS was written by pyuvdata. Only returned
        of `check_origin=True`.
    """
    _ms_utils_call_checks(filepath + "/HISTORY")

    # Set up the history string and pyuvdata check
    history_str = ""
    pyuvdata_written = False

    # Skip reading the history table if it has no information
    with tables.table(filepath + "/HISTORY", ack=False) as tb_hist:
        nrows = tb_hist.nrows()

        if nrows > 0:
            history_str = "Begin measurement set history\n"

            # Grab the standard history columns to stitch together
            try:
                app_params = tb_hist.getcol("APP_PARAMS")["array"]
                history_str += "APP_PARAMS;"
            except RuntimeError:
                app_params = None
            try:
                cli_command = tb_hist.getcol("CLI_COMMAND")["array"]
                history_str += "CLI_COMMAND;"
            except RuntimeError:
                cli_command = None
            application = tb_hist.getcol("APPLICATION")
            message = tb_hist.getcol("MESSAGE")
            obj_id = tb_hist.getcol("OBJECT_ID")
            obs_id = tb_hist.getcol("OBSERVATION_ID")
            origin = tb_hist.getcol("ORIGIN")
            priority = tb_hist.getcol("PRIORITY")
            times = tb_hist.getcol("TIME")
            history_str += (
                "APPLICATION;MESSAGE;OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
            )

            # Now loop through columns and generate history string
            cols = [application, message, obj_id, obs_id, origin, priority, times]
            if cli_command is not None:
                cols.insert(0, cli_command)
            if app_params is not None:
                cols.insert(0, app_params)

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
    if not uvutils._check_history_version(history_str, pyuvdata_version_str):
        history_str += pyuvdata_version_str

    # Finally, return the completed string
    if check_origin:
        return history_str, pyuvdata_written
    else:
        return history_str


def read_ms_ant(filepath, check_frame=True):
    """Read Measurement Set ANTENNA table."""
    _ms_utils_call_checks(filepath + "/ANTENNA")
    # open table with antenna location information
    with tables.table(filepath + "/ANTENNA", ack=False) as tb_ant:
        antenna_positions = tb_ant.getcol("POSITION")
        telescope_frame = tb_ant.getcolkeyword("POSITION", "MEASINFO")["Ref"].lower()

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

    # Build a dict with all the relevant entries we need.
    ant_dict = {
        "antenna_positions": antenna_positions,
        "antenna_numbers": antenna_numbers,
        "telescope_frame": telescope_frame,
        "antenna_names": antenna_names,
        "station_names": station_names,
    }

    # Return the dict
    return ant_dict


def read_ms_obs(filepath):
    """Read Measurement Set OBSERVATION table."""
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


def read_ms_spw(filepath):
    """Read Measurement Set SPECTRAL_WINDOW table."""
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
                int(idx) for idx in tb_spws.getcol("ASSOC_SPW_ID")
            ]
            spw_dict["spoof_spw_id"] = False
        except RuntimeError:
            spw_dict["assoc_spw_id"] = list(range(n_rows))
            spw_dict["spoof_spw"] = True

        for idx in range(n_rows):
            spw_dict["chan_freq"][idx] = tb_spws.getcell("CHAN_FREQ", idx)
            spw_dict["chan_width"][idx] = tb_spws.getcell("CHAN_WIDTH", idx)

    return spw_dict


def read_ms_field(filepath, return_phase_center_catalog=False):
    """Read Measurement Set FIELD table."""
    _ms_utils_call_checks(filepath + "/FIELD")

    tb_field = tables.table(filepath + "/FIELD", ack=False)
    n_rows = tb_field.nrows()

    field_dict = {
        "name": tb_field.getcol("NAME"),
        "ra": [None] * n_rows,
        "dec": [None] * n_rows,
        "source_id": [None] * n_rows,
    }

    frame_keyword = tb_field.getcolkeyword("PHASE_DIR", "MEASINFO")["Ref"]
    field_dict["frame"], field_dict["epoch"] = COORD_PYUVDATA2CASA_DICT[frame_keyword]

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
        field_dict["source_id"] = int(tb_field.getcell("SOURCE_ID", idx))

    if not return_phase_center_catalog:
        return field_dict

    phase_center_catalog = []
    for idx in range(n_rows):
        phase_center_catalog[field_dict["source_id"][idx]] = {
            "cat_name": field_dict["name"][idx],
            "cat_type": "sidereal",
            "cat_lon": field_dict["ra"][idx],
            "cat_lat": field_dict["dec"][idx],
            "cat_frame": field_dict["frame"],
            "cat_epoch": field_dict["epoch"],
            "cat_times": None,
            "cat_pm_ra": None,
            "cat_pm_dec": None,
            "cat_vrad": None,
            "cat_dist": None,
            "info_source": None,
        }

    return phase_center_catalog


def read_time_scale(ms_table, *, raise_error=False):
    """Read time scale from TIME column in an MS table."""
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
