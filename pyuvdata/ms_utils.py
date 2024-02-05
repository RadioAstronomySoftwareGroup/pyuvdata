# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with MeasurementSet files."""

import os
import warnings

import numpy as np
from astropy.time import Time

from . import __version__
from . import utils as uvutils

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
casa_error = None
try:
    from casacore import tables
    from casacore.tables import tableutil
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
        ant_diameters = np.asarray(tb_ant.getcol("DISH_DIAMETER"))[good_mask].tolist()

    # Build a dict with all the relevant entries we need.
    ant_dict = {
        "antenna_positions": antenna_positions,
        "antenna_numbers": antenna_numbers,
        "telescope_frame": telescope_frame,
        "antenna_names": antenna_names,
        "station_names": station_names,
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
    telescope_location=None,
    telescope_frame=None,
):
    """
    Write out the antenna information into a CASA table.

    Parameters
    ----------
    filepath : str
        Path to MS (without ANTENNA suffix).
    """
    _ms_utils_call_checks(filepath)
    filepath += "::ANTENNA"

    if uvobj is not None:
        antenna_numbers = uvobj.antenna_numbers
        antenna_names = uvobj.antenna_names
        antenna_positions = uvobj.antenna_positions
        antenna_diameters = uvobj.antenna_diameters
        telescope_location = uvobj.telescope_location
        telescope_frame = uvobj._telescope_location.frame

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
        for ai, num in enumerate(antenna_numbers):
            ant_names_table[num] = antenna_names[ai]

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
        # TODO: ask Karto what the best way is to put in the lunar ellipsoid
        telescope_frame = telescope_frame.upper()
        telescope_frame = "ITRF" if (telescope_frame == "ITRS") else telescope_frame
        meas_info_dict = antenna_table.getcolkeyword("POSITION", "MEASINFO")
        meas_info_dict["Ref"] = telescope_frame
        antenna_table.putcolkeyword("POSITION", "MEASINFO", meas_info_dict)


def read_ms_data_description(filepath):
    """Read Measurement Set DATA_DESCRIPTION table."""
    raise NotImplementedError("Sorry...")


def write_ms_data_description(
    filepath, uvobj=None, nspws=None, flex_spw_polarization_array=None
):
    """
    Write out the data description information into a CASA table.

    Parameters
    ----------
    filepath : str
        Path to MS (without DATA_DESCRIPTION suffix).
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
        field_dict["source_id"][idx] = int(tb_field.getcell("SOURCE_ID", idx))

    if not return_phase_center_catalog:
        return field_dict

    phase_center_catalog = {}
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


def write_ms_field(filepath, uvobj=None, phase_center_catalog=None, time_val=None):
    """
    Write out the field information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without FIELD suffix)

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
            for key in col_ref_dict.keys():
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
                for key in col_ref_dict.keys():
                    field_table.putcell(col_ref_dict[key], idx, var_ref_dict[ref_dir])


def read_ms_history(filepath, pyuvdata_version_str, check_origin=False):
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
        A history string that may or may not contain output from
        `_ms_hist_to_string`.

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


def write_ms_observation(filepath, uvobj):
    """
    Write out the observation information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without OBSERVATION suffix)

    """
    _ms_utils_call_checks(filepath)
    filepath += "::OBSERVATION"

    if uvobj is not None:
        telescope_name = uvobj.telescope_name
        telescope_location = uvobj.telescope_location
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


def write_ms_spectral_window(
    filepath=None,
    uvobj=None,
    *,
    freq_array=None,
    channel_width=None,
    spw_array=None,
    flex_spw_id_array=None,
    use_future_array_shapes=True,
):
    """
    Write out the spectral information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without SPECTRAL_WINDOW suffix)

    """
    _ms_utils_call_checks(filepath)
    filepath += "::SPECTRAL_WINDOW"

    if uvobj is not None:
        freq_array = uvobj.freq_array
        channel_width = uvobj.channel_width
        spw_array = uvobj.spw_array
        flex_spw_id_array = uvobj.flex_spw_id_array
        use_future_array_shapes = uvobj.future_array_shapes

    if not use_future_array_shapes:
        freq_array = freq_array[0]
        channel_width = np.full_like(freq_array, channel_width)

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
            if flex_spw_id_array is None:
                ch_mask = np.ones(freq_array.shape, dtype=bool)
            else:
                ch_mask = flex_spw_id_array == spw_id
            sw_table.addrows()
            sw_table.putcell("NUM_CHAN", idx, np.sum(ch_mask))
            sw_table.putcell("NAME", idx, "SPW%d" % spw_id)
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


def read_ms_feed(filepath):
    """Read Measurement Set FEED table."""
    _ms_utils_call_checks(filepath)

    raise NotImplementedError("More to do...")


def write_ms_feed(
    filepath, pol_order=..., uvobj=None, spectral_window_id_table=None, pol_str=None
):
    """
    Write out the feed information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without FEED suffix)
    pol_order : slice or list of int
        Ordering of the polarization axis on write, only used if not writing a
        flex-pol dataset.
    """
    _ms_utils_call_checks(filepath)
    filepath += "::FEED"

    if uvobj is not None:
        nfeeds_table = np.max(uvobj.antenna_numbers) + 1
        antenna_id_table = np.arange(nfeeds_table, dtype=np.int32)
        if uvobj.flex_spw_polarization_array is None:
            spectral_window_id_table = -1 * np.ones(nfeeds_table, dtype=np.int32)

            # we want "x" or "y", *not* "e" or "n", so as not to confuse CASA
            pol_str = uvutils.polnum2str(uvobj.polarization_array[pol_order])
        else:
            spectral_window_id_table = np.repeat(np.arange(uvobj.Nspws), nfeeds_table)
            nfeeds_table *= uvobj.Nspws
            antenna_id_table = np.tile(antenna_id_table, uvobj.Nspws)
            # we want "x" or "y", *not* "e" or "n", so as not to confuse CASA
            pol_str = uvutils.polnum2str(uvobj.flex_spw_polarization_array)

    with tables.table(filepath, ack=False, readonly=False) as feed_table:
        feed_pols = {feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]}
        nfeed_pols = len(feed_pols)
        pol_types = [pol.upper() for pol in sorted(feed_pols)]
        pol_type_table = np.tile(pol_types, (nfeeds_table, 1))

        num_receptors_table = np.tile(np.int32(nfeed_pols), nfeeds_table)
        beam_id_table = -1 * np.ones(nfeeds_table, dtype=np.int32)
        beam_offset_table = np.zeros((nfeeds_table, 2, 2), dtype=np.float64)
        feed_id_table = np.zeros(nfeeds_table, dtype=np.int32)
        interval_table = np.zeros(nfeeds_table, dtype=np.float64)
        pol_response_table = np.dstack(
            [np.eye(2, dtype=np.complex64) for n in range(nfeeds_table)]
        ).transpose()
        position_table = np.zeros((nfeeds_table, 3), dtype=np.float64)
        receptor_angle_table = np.zeros((nfeeds_table, nfeed_pols), dtype=np.float64)

        feed_table.addrows(nfeeds_table)
        feed_table.putcol("ANTENNA_ID", antenna_id_table)
        feed_table.putcol("BEAM_ID", beam_id_table)
        feed_table.putcol("BEAM_OFFSET", beam_offset_table)
        feed_table.putcol("FEED_ID", feed_id_table)
        feed_table.putcol("INTERVAL", interval_table)
        feed_table.putcol("NUM_RECEPTORS", num_receptors_table)
        feed_table.putcol("POLARIZATION_TYPE", pol_type_table)
        feed_table.putcol("POL_RESPONSE", pol_response_table)
        feed_table.putcol("POSITION", position_table)
        feed_table.putcol("RECEPTOR_ANGLE", receptor_angle_table)
        feed_table.putcol("SPECTRAL_WINDOW_ID", spectral_window_id_table)


def read_ms_source(filepath):
    """Read Measurement Set SOURCE table."""
    _ms_utils_call_checks(filepath)

    raise NotImplementedError("Even more to do...")


def write_ms_source(filepath, uvobj=None, time_default=None, phase_center_catalog=None):
    """
    Write out the source information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without SOURCE suffix)

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
                sou_dir = np.vstack(((pc_dict["cat_lon"], pc_dict["cat_lat"]))).T
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
    """Read Measurement Set POINTING table."""
    _ms_utils_call_checks(filepath)

    raise NotImplementedError("More to do...")


def write_ms_pointing(
    filepath, uvobj=None, max_ant=None, integration_time=None, time_array=None
):
    """
    Write out the pointing information into a CASA table.

    Parameters
    ----------
    filepath : str
        path to MS (without POINTING suffix)

    """
    _ms_utils_call_checks(filepath)
    filepath += "::POINTING"

    if uvobj is not None:
        max_ant = np.max(uvobj.antenna_numbers)
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
    """Read Measurement Set POLARIZATION table."""
    _ms_utils_call_checks(filepath)

    raise NotImplementedError("More, more, more...")


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
                pol_str = uvutils.polnum2str([spw_pol])
                feed_pols = {
                    feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]
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
            pol_str = uvutils.polnum2str(pol_arr)
            feed_pols = {
                feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]
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


def init_ms_file(filepath):
    """
    Create a skeleton MS dataset to fill.

    Parameters
    ----------
    filepath : str
        Path to MS to be created.
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


def init_ms_cal_file(filename):
    """Initialize an MS calibration file."""
    standard_desc = tables.required_ms_desc()
    tabledesc = {}
    tabledesc["TIME"] = standard_desc["TIME"]
    tabledesc["FIELD_ID"] = standard_desc["FIELD_ID"]
    tabledesc["ANTENNA1"] = standard_desc["ANTENNA1"]
    tabledesc["ANTENNA2"] = standard_desc["ANTENNA2"]
    tabledesc["INTERVAL"] = standard_desc["INTERVAL"]
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

    tabledesc["CPARAM"] = tables.makearrcoldesc(
        None,
        None,
        valuetype="complex",
        ndim=-1,
        datamanagertype="StandardStMan",
        comment="Complex gain data.",
    )["desc"]
    del tabledesc["CPARAM"]["shape"]

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
                "subType": "G Jones",
                "readme": f"Written with pyuvdata version: {__version__}.",
            }
        )
        # Finally, set up some de
        ms.putkeyword("ParType", "Complex")
        ms.putkeyword("MSName", "unknown")
        ms.putkeyword("VisCal", "unknown")
        ms.putkeyword("PolBasis", "unknown")
        ms.putkeyword("CASA_Version", "unknown")
