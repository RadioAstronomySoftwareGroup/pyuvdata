# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Class for reading and writing casa measurement sets.

Requires casacore.
"""
import numpy as np
import os
import warnings
from astropy.time import Time

from .uvdata import UVData
from .. import utils as uvutils

__all__ = ["MS"]

no_casa_message = (
    "casacore is not installed but is required for " "measurement set functionality"
)

casa_present = True
try:
    import casacore.tables as tables
    import casacore.tables.tableutil as tableutil
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


class MS(UVData):
    """
    Defines a class for reading and writing casa measurement sets.

    This class should not be interacted with directly, instead use the read_ms
    method on the UVData class.

    Attributes
    ----------
    ms_required_extra : list of str
        Names of optional UVParameters that are required for MS
    """

    ms_required_extra = ["datacolumn", "antenna_positions"]

    def _write_ms_antenna(self, filepath):
        """
        Write out the antenna information into a CASA table.

        Parameters
        ----------
        filepath : str
            Path to MS (without ANTENNA suffix).
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        antenna_table = tables.table(filepath + "::ANTENNA", ack=False, readonly=False)

        # Note: measurement sets use the antenna number as an index into the antenna
        # table. This means that if the antenna numbers do not start from 0 and/or are
        # not contiguous, empty rows need to be inserted into the antenna table
        # (this is somewhat similar to miriad)
        nants_table = np.max(self.antenna_numbers) + 1
        antenna_table.addrows(nants_table)

        ant_names_table = [""] * nants_table
        for ai, num in enumerate(self.antenna_numbers):
            ant_names_table[num] = self.antenna_names[ai]

        # There seem to be some variation on whether the antenna names are stored
        # in the NAME or STATION column (importuvfits puts them in the STATION column
        # while Cotter and the MS definition doc puts them in the NAME column).
        # The MS definition doc suggests that antenna names belong in the NAME column
        # and the telescope name belongs in the STATION column (it gives the example of
        # GREENBANK for this column.) so we follow that here.
        antenna_table.putcol("NAME", ant_names_table)
        antenna_table.putcol("STATION", [self.telescope_name] * nants_table)

        # Antenna positions in measurement sets appear to be in absolute ECEF
        ant_pos_absolute = self.antenna_positions + self.telescope_location.reshape(
            1, 3
        )
        ant_pos_table = np.zeros((nants_table, 3), dtype=np.float64)
        for ai, num in enumerate(self.antenna_numbers):
            ant_pos_table[num, :] = ant_pos_absolute[ai, :]

        antenna_table.putcol("POSITION", ant_pos_table)
        if self.antenna_diameters is not None:
            ant_diam_table = np.zeros((nants_table), dtype=self.antenna_diameters.dtype)
            for ai, num in enumerate(self.antenna_numbers):
                ant_diam_table[num] = self.antenna_diameters[ai]
            antenna_table.putcol("DISH_DIAMETER", ant_diam_table)
        antenna_table.done()

    def _write_ms_data_description(self, filepath):
        """
        Write out the data description information into a CASA table.

        Parameters
        ----------
        filepath : str
            Path to MS (without DATA_DESCRIPTION suffix).
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        data_descrip_table = tables.table(
            filepath + "::DATA_DESCRIPTION", ack=False, readonly=False
        )

        data_descrip_table.addrows(self.Nspws)

        data_descrip_table.putcol("SPECTRAL_WINDOW_ID", np.arange(self.Nspws))

        data_descrip_table.done()

    def _write_ms_feed(self, filepath):
        """
        Write out the feed information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without FEED suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        feed_table = tables.table(filepath + "::FEED", ack=False, readonly=False)

        nfeeds_table = np.max(self.antenna_numbers) + 1
        feed_table.addrows(nfeeds_table)

        antenna_id_table = np.arange(nfeeds_table, dtype=np.int32)
        feed_table.putcol("ANTENNA_ID", antenna_id_table)

        beam_id_table = -1 * np.ones(nfeeds_table, dtype=np.int32)
        feed_table.putcol("BEAM_ID", beam_id_table)

        beam_offset_table = np.zeros((nfeeds_table, 2, 2), dtype=np.float64)
        feed_table.putcol("BEAM_OFFSET", beam_offset_table)

        feed_id_table = np.zeros(nfeeds_table, dtype=np.int32)
        feed_table.putcol("FEED_ID", feed_id_table)

        interval_table = np.zeros(nfeeds_table, dtype=np.float64)
        feed_table.putcol("INTERVAL", interval_table)

        # we want "x" or "y", *not* "e" or "n", so as not to confuse CASA
        pol_str = uvutils.polnum2str(self.polarization_array)
        feed_pols = {feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]}
        nfeed_pols = len(feed_pols)
        num_receptors_table = np.tile(np.int32(nfeed_pols), nfeeds_table)
        feed_table.putcol("NUM_RECEPTORS", num_receptors_table)

        pol_types = [pol.upper() for pol in sorted(feed_pols)]
        pol_type_table = np.tile(pol_types, (nfeeds_table, 1))
        feed_table.putcol("POLARIZATION_TYPE", pol_type_table)

        pol_response_table = np.dstack(
            [np.eye(2, dtype=np.complex64) for n in range(nfeeds_table)]
        ).transpose()
        feed_table.putcol("POL_RESPONSE", pol_response_table)

        position_table = np.zeros((nfeeds_table, 3), dtype=np.float64)
        feed_table.putcol("POSITION", position_table)

        receptor_angle_table = np.zeros((nfeeds_table, nfeed_pols), dtype=np.float64)
        feed_table.putcol("RECEPTOR_ANGLE", receptor_angle_table)

        spectral_window_id_table = -1 * np.ones(nfeeds_table, dtype=np.int32)
        feed_table.putcol("SPECTRAL_WINDOW_ID", spectral_window_id_table)

        feed_table.done()

    def _write_ms_field(self, filepath):
        """
        Write out the field information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without FIELD suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        # TODO revisit for multi object!
        field_table = tables.table(filepath + "::FIELD", ack=False, readonly=False)
        field_table.addrows()
        phasedir = np.array([[self.phase_center_ra, self.phase_center_dec]])
        assert self.phase_center_epoch == 2000.0
        field_table.putcell("DELAY_DIR", 0, phasedir)
        field_table.putcell("PHASE_DIR", 0, phasedir)
        field_table.putcell("REFERENCE_DIR", 0, phasedir)
        field_table.putcell("NAME", 0, self.object_name)

    def _write_ms_pointing(self, filepath):
        """
        Write out the pointing information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without POINTING suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        pointing_table = tables.table(
            filepath + "::POINTING", ack=False, readonly=False
        )
        # TODO: we probably have to fix this for steerable dishes
        nants = np.max(self.antenna_numbers) + 1
        antennas = np.arange(nants, dtype=np.int32)

        # We want the number of unique timestamps here, since CASA wants a
        # per-timestamp, per-antenna entry
        times = np.asarray(
            [
                Time(t, format="jd", scale="utc").mjd * 3600.0 * 24.0
                for t in np.unique(self.time_array)
            ]
        )

        # Same for the number of intervals
        intervals = np.array(
            [
                np.median(self.integration_time[self.time_array == timestamp])
                for timestamp in np.unique(self.time_array)
            ]
        )

        ntimes = len(times)

        antennas_full = np.tile(antennas, ntimes)
        times_full = np.repeat(times, nants)
        intervals_full = np.repeat(intervals, nants)

        nrows = len(antennas_full)
        assert len(times_full) == nrows
        assert len(intervals_full) == nrows

        # This extra step of adding a single row and plugging in values for DIRECTION
        # and TARGET are a workaround for a weird issue that pops up where, because the
        # values are not a fixed size (they are shape(2, Npoly), where Npoly is allowed
        # to vary from integration to integration), casacore seems to be very slow
        # filling in the data after the fact. By "pre-filling" the first row, the
        # addrows operation will automatically fill in an appropriately shaped array.
        pointing_table.addrows(1)
        pointing_table.putcell("DIRECTION", 0, np.zeros((2, 1), dtype=np.float64))
        pointing_table.putcell("TARGET", 0, np.zeros((2, 1), dtype=np.float64))
        pointing_table.addrows(nrows - 1)
        pointing_table.done()

        pointing_table = tables.table(
            filepath + "::POINTING", ack=False, readonly=False
        )

        pointing_table.putcol("ANTENNA_ID", antennas_full, nrow=nrows)
        pointing_table.putcol("TIME", times_full, nrow=nrows)
        pointing_table.putcol("INTERVAL", intervals_full, nrow=nrows)

        name_col = np.asarray(["ZENITH"] * nrows, dtype=np.bytes_)
        pointing_table.putcol("NAME", name_col, nrow=nrows)

        num_poly_col = np.zeros(nrows, dtype=np.int32)
        pointing_table.putcol("NUM_POLY", num_poly_col, nrow=nrows)

        time_origin_col = times_full
        pointing_table.putcol("TIME_ORIGIN", time_origin_col, nrow=nrows)

        # we always "point" at zenith
        direction_col = np.zeros((nrows, 2, 1), dtype=np.float64)
        direction_col[:, 1, :] = np.pi / 2
        pointing_table.putcol("DIRECTION", direction_col, nrow=nrows)

        # just reuse "direction" for "target"
        pointing_table.putcol("TARGET", direction_col, nrow=nrows)

        # set tracking info to `False`
        tracking_col = np.zeros(nrows, dtype=np.bool_)
        pointing_table.putcol("TRACKING", tracking_col, nrow=nrows)

        pointing_table.done()

    def _write_ms_spectralwindow(self, filepath):
        """
        Write out the spectral information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without SPECTRAL_WINDOW suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        sw_table = tables.table(
            filepath + "::SPECTRAL_WINDOW", ack=False, readonly=False
        )

        if self.future_array_shapes:
            freq_array = self.freq_array
            ch_width = self.channel_width
        else:
            freq_array = self.freq_array[0]
            ch_width = np.zeros_like(freq_array) + self.channel_width

        for idx in self.spw_array:
            if self.flex_spw:
                ch_mask = self.flex_spw_id_array == idx
            else:
                ch_mask = np.ones(freq_array.shape, dtype=bool)

            sw_table.addrows()
            sw_table.putcell("NUM_CHAN", 0, np.sum(ch_mask))
            sw_table.putcell("NAME", 0, "WINDOW%d" % idx)
            sw_table.putcell("CHAN_FREQ", 0, freq_array[ch_mask])
            sw_table.putcell("CHAN_WIDTH", 0, ch_width[ch_mask])
            sw_table.putcell("EFFECTIVE_BW", 0, ch_width[ch_mask])
            sw_table.putcell("RESOLUTION", 0, ch_width[ch_mask])
            # TODO: These are placeholders for now, but should be replaced with
            # actual frequency reference info (once UVData handles that)
            sw_table.putcell("MEAS_FREQ_REF", 0, VEL_DICT["TOPO"])
            sw_table.putcell("REF_FREQUENCY", 0, freq_array[0])

    def _write_ms_observation(self, filepath):
        """
        Write out the observation information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without OBSERVATION suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        observation_table = tables.table(
            filepath + "::OBSERVATION", ack=False, readonly=False
        )
        observation_table.addrows()
        observation_table.putcell("TELESCOPE_NAME", 0, self.telescope_name)

        # It appears that measurement sets do not have a concept of a telescope location
        # We add it here as a non-standard column in order to round trip it properly
        name_col_desc = tableutil.makearrcoldesc(
            "TELESCOPE_LOCATION",
            self.telescope_location[0],
            shape=[3],
            valuetype="double",
        )
        observation_table.addcols(name_col_desc)
        observation_table.putcell("TELESCOPE_LOCATION", 0, self.telescope_location)

        # should we test for observer in extra keywords?
        observation_table.putcell("OBSERVER", 0, self.telescope_name)

    def _write_ms_polarization(self, filepath):
        """
        Write out the polarization information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without POLARIZATION suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        pol_str = uvutils.polnum2str(self.polarization_array)
        feed_pols = {feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]}
        pol_types = [pol.lower() for pol in sorted(feed_pols)]
        pol_tuples = np.asarray(
            [(pol_types.index(i), pol_types.index(j)) for i, j in pol_str],
            dtype=np.int32,
        )

        pol_table = tables.table(filepath + "::POLARIZATION", ack=False, readonly=False)
        pol_table.addrows()
        pol_table.putcell(
            "CORR_TYPE",
            0,
            np.array(
                [POL_AIPS2CASA_DICT[aipspol] for aipspol in self.polarization_array]
            ),
        )
        pol_table.putcell("CORR_PRODUCT", 0, pol_tuples)
        pol_table.putcell("NUM_CORR", 0, self.Npols)

    def _ms_hist_to_string(self, history_table):
        """
        Convert a CASA history table into a string for the uvdata history parameter.

        Also stores messages column as a list for consitency with other uvdata types

        Parameters
        ----------
        history_table : a casa table object
            CASA table with history information.

        Returns
        -------
        str
            string enconding complete casa history table converted with a new
            line denoting rows and a ';' denoting column breaks.
        pyuvdata_written :  bool
            boolean indicating whether or not this MS was written by pyuvdata.

        """
        # string to store all the casa history info
        history_str = ""

        # Do not touch the history table if it has no information
        if history_table.nrows() > 0:
            history_str = (
                "Begin measurement set history\n"
                "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE;"
                "OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
            )

            app_params = history_table.getcol("APP_PARAMS")["array"]
            # TODO: might need to handle the case where cli_command is empty
            cli_command = history_table.getcol("CLI_COMMAND")["array"]
            application = history_table.getcol("APPLICATION")
            message = history_table.getcol("MESSAGE")
            obj_id = history_table.getcol("OBJECT_ID")
            obs_id = history_table.getcol("OBSERVATION_ID")
            origin = history_table.getcol("ORIGIN")
            priority = history_table.getcol("PRIORITY")
            times = history_table.getcol("TIME")
            # Now loop through columns and generate history string
            ntimes = len(times)
            cols = [
                app_params,
                cli_command,
                application,
                message,
                obj_id,
                obs_id,
                origin,
                priority,
                times,
            ]

            # if this MS was written by pyuvdata, some history that originated in
            # pyuvdata is in the MS history table. We separate that out since it doesn't
            # really belong to the MS history block (and so round tripping works)
            pyuvdata_line_nos = [
                line_no
                for line_no, line in enumerate(application)
                if "pyuvdata" in line
            ]

            for tbrow in range(ntimes):
                # first check to see if this row was put in by pyuvdata.
                # If so, don't mix them in with the MS stuff
                if tbrow in pyuvdata_line_nos:
                    continue

                newline = ";".join([str(col[tbrow]) for col in cols]) + "\n"
                history_str += newline

            history_str += "End measurement set history.\n"

            # make a list of lines that are MS specific (i.e. not pyuvdata lines)
            ms_line_nos = np.arange(ntimes).tolist()
            for line_no in pyuvdata_line_nos:
                ms_line_nos.pop(line_no)

            pyuvdata_written = False
            if len(pyuvdata_line_nos) > 0:
                pyuvdata_written = True
                for line_no in pyuvdata_line_nos:
                    if line_no < min(ms_line_nos):
                        # prepend these lines to the history
                        history_str = message[line_no] + history_str
                    else:
                        # append these lines to the history
                        history_str += message[line_no]

        def is_not_ascii(s):
            return any(ord(c) >= 128 for c in s)

        def find_not_ascii(s):
            output = []
            for c in s:
                if ord(c) >= 128:
                    output += c
            return output

        return history_str, pyuvdata_written

    def _write_ms_history(self, filepath):
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
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        app_params = []
        cli_command = []
        application = []
        message = []
        obj_id = []
        obs_id = []
        origin = []
        priority = []
        times = []
        if "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE" in self.history:
            # this history contains info from an MS history table. Need to parse it.

            ms_header_line_no = None
            ms_end_line_no = None
            pre_ms_history_lines = []
            post_ms_history_lines = []
            for line_no, line in enumerate(self.history.splitlines()):
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
                message.append(line_no, line)
                obj_id.insert(line_no, 0)
                obs_id.insert(line_no, -1)
                origin.insert(line_no, "pyuvdata")
                priority.insert(line_no, "INFO")
                times.insert(line_no, Time.now().mjd * 3600.0 * 24.0)

            for line_no, line in enumerate(post_ms_history_lines):
                app_params.append("")
                cli_command.append("")
                application.append("pyuvdata")
                message.append(line)
                obj_id.append(0)
                obs_id.append(-1)
                origin.append("pyuvdata")
                priority.append("INFO")
                times.append(Time.now().mjd * 3600.0 * 24.0)

        else:
            # no prior MS history detected in the history. Put all of our history in
            # the message column
            for line in self.history.splitlines():
                app_params.append("")
                cli_command.append("")
                application.append("pyuvdata")
                message.append(line)
                obj_id.append(0)
                obs_id.append(-1)
                origin.append("pyuvdata")
                priority.append("INFO")
                times.append(Time.now().mjd * 3600.0 * 24.0)

        history_table = tables.table(filepath + "::HISTORY", ack=False, readonly=False)

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

        history_table.done()

    def write_ms(
        self,
        filepath,
        force_phase=False,
        clobber=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Write a CASA measurement set (MS).

        Parameters
        ----------
        filepath : str
            The MS file path to write to.
        force_phase : bool
            Option to automatically phase drift scan data to zenith of the first
            timestamp.
        clobber : bool
            Option to overwrite the file if it already exists.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if os.path.exists(filepath):
            if clobber:
                print("File exists; clobbering")
            else:
                raise IOError("File exists; skipping")

        # TODO: the phasing requirement can probably go away once we have proper
        # multi-object support.
        if self.phase_type == "phased":
            pass
        elif self.phase_type == "drift":
            if force_phase:
                print(
                    "The data are in drift mode and do not have a "
                    "defined phase center. Phasing to zenith of the first "
                    "timestamp."
                )
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time)
            else:
                raise ValueError(
                    "The data are in drift mode. "
                    "Set force_phase to true to phase the data "
                    "to zenith of the first timestamp before "
                    "writing a uvfits file."
                )
        else:
            raise ValueError(
                "The phasing type of the data is unknown. "
                "Set the phase_type to drift or phased to "
                "reflect the phasing status of the data"
            )

        nchan = self.freq_array.shape[1]
        npol = len(self.polarization_array)
        nrow = len(self.data_array)

        datacoldesc = tables.makearrcoldesc(
            "DATA",
            0.0 + 0.0j,
            valuetype="complex",
            shape=[nchan, npol],
            datamanagertype="TiledColumnStMan",
            datamanagergroup="TiledData",
        )
        weightcoldesc = tables.makearrcoldesc(
            "WEIGHT_SPECTRUM",
            0.0,
            valuetype="float",
            shape=[nchan, npol],
            datamanagertype="TiledColumnStMan",
            datamanagergroup="TiledData",
        )

        # It'd be good to understand what it is that "option=4" sets
        ms_desc = tables.required_ms_desc("MAIN")
        ms_desc["FLAG"].update(
            dataManagerType="TiledColumnStMan",
            shape=[nchan, npol],
            dataManagerGroup="TiledFlag",
            cellShape=[nchan, npol],
            option=4,
        )
        ms_desc.update(tables.maketabdesc(datacoldesc))
        ms_desc.update(tables.maketabdesc(weightcoldesc))

        ms = tables.default_ms(filepath, ms_desc, tables.makedminfo(ms_desc))
        ms.addrows(nrow)

        # TODO: remove squeeze for future array shapes
        # FITS uvw direction convention is opposite ours and Miriad's.
        # CASA's convention is unclear: the docs contradict themselves,
        # but after a question to the helpdesk we got a clear response that
        # the convention is antenna2_pos - antenna1_pos, so the convention is the
        # same as ours & Miriad's
        ms.putcol("DATA", np.squeeze(self.data_array, axis=1))
        ms.putcol("WEIGHT_SPECTRUM", np.squeeze(self.nsample_array, axis=1))
        ms.putcol("FLAG", np.squeeze(self.flag_array, axis=1))

        ms.putcol("ANTENNA1", self.ant_1_array)
        ms.putcol("ANTENNA2", self.ant_2_array)
        ms.putcol("INTERVAL", self.integration_time)

        ms.putcol("TIME", Time(self.time_array, format="jd").mjd * 3600.0 * 24.0)
        ms.putcol("UVW", self.uvw_array)
        ms.done()

        # The above will only add the default subtables, specifically ANTENNA,
        # DATA_DESCRIPTION, FEED, FIELD, FLAG_CMD, HISTORY, OBSERVATION, POINTING,
        # POLARIZTION, PROCESSOR, SPECTRAL_WINDOW, and STATE, but if needed (say for
        # example the SOURCE ta), one can use the following call to add the subtable
        # so that it can subsequently be edited:
        #   subtable = tables.default_ms_subltable(NAME, filepath + NAME)
        # Done that this has to be done after the MS is closed (via done()), and one
        # needs to close subtable (again, via done()) to finish creating the stub.

        self._write_ms_antenna(filepath)
        self._write_ms_data_description(filepath)
        self._write_ms_feed(filepath)
        self._write_ms_field(filepath)
        self._write_ms_spectralwindow(filepath)
        self._write_ms_pointing(filepath)
        self._write_ms_polarization(filepath)
        self._write_ms_observation(filepath)
        self._write_ms_history(filepath)

    def read_ms(
        self,
        filepath,
        data_column="DATA",
        pol_order="AIPS",
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        use_old_phase=False,
        fix_phase=False,
    ):
        """
        Read in a casa measurement set.

        Parameters
        ----------
        filepath : str
            The measurement set root directory to read from.
        data_column : str
            name of CASA data column to read into data_array. Options are:
            'DATA', 'MODEL', or 'CORRECTED_DATA'
        pol_order : str
            Option to specify polarizations order convention, options are
            'CASA' or 'AIPS'.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        IOError
            If root file directory doesn't exist.
        ValueError
            If the `data_column` is not set to an allowed value.
            If the data are have multiple subarrays or are multi source or have
            multiple spectral windows.
            If the data have multiple data description ID values.

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        # make sure user requests a valid data_column
        if data_column not in ["DATA", "CORRECTED_DATA", "MODEL"]:
            raise ValueError(
                "Invalid data_column value supplied. Use 'Data','MODEL' or"
                " 'CORRECTED_DATA'"
            )
        if not os.path.exists(filepath):
            raise IOError(filepath + " not found")
        # set filename variable
        basename = filepath.rstrip("/")
        self.filename = [os.path.basename(basename)]
        self._filename.form = (1,)
        # set visibility units
        if data_column == "DATA":
            self.vis_units = "UNCALIB"
        elif data_column == "CORRECTED_DATA":
            self.vis_units = "JY"
        elif data_column == "MODEL":
            self.vis_units = "JY"
        # limit length of extra_keywords keys to 8 characters to match uvfits & miriad
        self.extra_keywords["DATA_COL"] = data_column

        # get the history info
        if tables.tableexists("HISTORY"):
            self.history, pyuvdata_written = self._ms_hist_to_string(
                tables.table(filepath + "/HISTORY", ack=False)
            )
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str
        else:
            self.history = self.pyuvdata_version_str

        # get frequency information from spectral window table
        tb_spws = tables.table(filepath + "/SPECTRAL_WINDOW", ack=False)
        spw_names = tb_spws.getcol("NAME")
        self.Nspws = len(spw_names)
        if self.Nspws > 1:
            raise ValueError(
                "Sorry.  Files with more than one spectral"
                "window (spw) are not yet supported. A "
                "great project for the interested student!"
            )
        freqs = tb_spws.getcol("CHAN_FREQ")
        self.freq_array = freqs
        self.Nfreqs = int(freqs.shape[1])
        # beware! There are possibly 3 columns here that might be the correct one to use
        # CHAN_WIDTH, EFFECTIVE_BW, RESOLUTION
        self.channel_width = float(tb_spws.getcol("CHAN_WIDTH")[0, 0])
        self.Nspws = int(freqs.shape[0])

        self.spw_array = np.arange(self.Nspws)
        tb_spws.close()
        # now get the data
        tb = tables.table(filepath, ack=False)
        # check for multiple subarrays. importuvfits does not appear to
        # preserve subarray information!
        subarray = np.unique(np.int32(tb.getcol("ARRAY_ID")) - 1)
        if len(set(subarray)) > 1:
            raise ValueError(
                "This file appears to have multiple subarray "
                "values; only files with one subarray are "
                "supported."
            )
        times_unique = Time(
            np.unique(tb.getcol("TIME") / (3600.0 * 24.0)), format="mjd"
        ).jd

        # check for multiple data description ids (combination of spw id & pol id)
        data_desc_id = np.unique(np.int32(tb.getcol("DATA_DESC_ID") - 1))
        if len(set(data_desc_id)) > 1:
            raise ValueError(
                "This file appears to have multiple data description ID "
                "values; only files with one data description ID are "
                "supported."
            )

        self.Ntimes = int(len(times_unique))
        # FITS uvw direction convention is opposite ours and Miriad's.
        # CASA's convention is unclear: the docs contradict themselves,
        # but after a question to the helpdesk we got a clear response that
        # the convention is antenna2_pos - antenna1_pos, so the convention is the
        # same as ours & Miriad's
        # HOWEVER, it appears that CASA's importuvfits task does not make this
        # convention change. So if the data in the MS came via that task and was not
        # written by pyuvdata, we do need to flip the uvws & conjugate the data
        if "importuvfits" in self.history and not pyuvdata_written:
            data_array = np.conj(tb.getcol(data_column))
        else:
            data_array = tb.getcol(data_column)
        self.Nblts = int(data_array.shape[0])
        flag_array = tb.getcol("FLAG")
        # CASA stores data in complex array with dimension NbltsxNfreqsxNpols
        if len(data_array.shape) == 3:
            data_array = np.expand_dims(data_array, axis=1)
            flag_array = np.expand_dims(flag_array, axis=1)
        self.data_array = data_array
        self.flag_array = flag_array
        self.Npols = int(data_array.shape[-1])

        # see note above the data_array extraction
        if "importuvfits" in self.history:
            self.uvw_array = -1 * tb.getcol("UVW")
        else:
            self.uvw_array = tb.getcol("UVW")

        self.ant_1_array = tb.getcol("ANTENNA1").astype(np.int32)
        self.ant_2_array = tb.getcol("ANTENNA2").astype(np.int32)
        self.Nants_data = len(
            np.unique(
                np.concatenate(
                    (np.unique(self.ant_1_array), np.unique(self.ant_2_array))
                )
            )
        )
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        self.Nbls = len(np.unique(self.baseline_array))
        # Get times. MS from cotter are modified Julian dates in seconds
        # (thanks to Danny Jacobs for figuring out the proper conversion)
        self.time_array = Time(tb.getcol("TIME") / (3600.0 * 24.0), format="mjd").jd

        # Polarization array
        tb_pol = tables.table(filepath + "/POLARIZATION", ack=False)
        num_pols = tb_pol.getcol("NUM_CORR")
        # get pol setup ID from data description table
        tb_data_desc = tables.table(filepath + "/DATA_DESCRIPTION", ack=False)
        pol_id = tb_data_desc.getcol("POLARIZATION_ID")[0]
        # use an assert because I don't think it's possible to make a file
        # that fails this, but I'm not sure
        assert num_pols[pol_id] == self.Npols

        if np.unique(num_pols).size > 1:
            # use getvarcol method, which returns a dict
            pol_list = tb_pol.getvarcol("CORR_TYPE")["r" + str(pol_id + 1)][0].tolist()
        else:
            # list of lists, probably with each list corresponding to SPW.
            pol_list = tb_pol.getcol("CORR_TYPE")[pol_id]
        self.polarization_array = np.zeros(len(pol_list), dtype=np.int32)
        for polnum, casapol in enumerate(pol_list):
            self.polarization_array[polnum] = POL_CASA2AIPS_DICT[casapol]
        tb_pol.close()

        # Integration time
        # use first interval and assume rest are constant (though measurement
        # set has all integration times for each Nblt )
        # self.integration_time=tb.getcol('INTERVAL')[0]
        # for some reason, interval ends up larger than the difference between times...
        if len(times_unique) == 1:
            self.integration_time = np.ones_like(self.time_array, dtype=np.float64)
        else:
            # assume that all times in the file are the same size
            int_time = self._calc_single_integration_time()
            self.integration_time = (
                np.ones_like(self.time_array, dtype=np.float64) * int_time
            )
        # open table with antenna location information
        tb_ant = tables.table(filepath + "/ANTENNA", ack=False)
        tb_obs = tables.table(filepath + "/OBSERVATION", ack=False)
        self.telescope_name = tb_obs.getcol("TELESCOPE_NAME")[0]
        self.instrument = tb_obs.getcol("TELESCOPE_NAME")[0]
        full_antenna_positions = tb_ant.getcol("POSITION")
        n_ants_table = full_antenna_positions.shape[0]
        xyz_telescope_frame = tb_ant.getcolkeyword("POSITION", "MEASINFO")["Ref"]

        # Note: measurement sets use the antenna number as an index into the antenna
        # table. This means that if the antenna numbers do not start from 0 and/or are
        # not contiguous, empty rows are inserted into the antenna table
        # these 'dummy' rows have positions of zero and need to be removed.
        # (this is somewhat similar to miriad)
        ant_good_position = np.nonzero(np.linalg.norm(full_antenna_positions, axis=1))[
            0
        ]
        full_antenna_positions = full_antenna_positions[ant_good_position, :]
        self.antenna_numbers = np.arange(n_ants_table)[ant_good_position]

        # check to see if a TELESCOPE_LOCATION column is present in the observation
        # table. This is non-standard, but inserted by pyuvdata
        if "TELESCOPE_LOCATION" in tb_obs.colnames():
            self.telescope_location = np.squeeze(tb_obs.getcol("TELESCOPE_LOCATION"))
        else:
            if self.telescope_name in self.known_telescopes():
                # Try to get it from the known telescopes
                self.set_telescope_params()
            elif xyz_telescope_frame == "ITRF":
                # Set it to be the mean of the antenna positions (this is not ideal!)
                self.telescope_location = np.array(
                    np.mean(full_antenna_positions, axis=0)
                )
            else:
                warnings.warn(
                    "Telescope frame is not ITRF and telescope is not "
                    "in known_telescopes, so telescope_location is not set."
                )
        tb_obs.close()

        # antenna names
        ant_names = tb_ant.getcol("NAME")
        station_names = tb_ant.getcol("STATION")
        ant_diams = tb_ant.getcol("DISH_DIAMETER")

        self.antenna_diameters = ant_diams[ant_diams > 0]

        station_names
        station_names_same = False
        for name in station_names[1:]:
            if name == station_names[0]:
                station_names_same = True
        # importuvfits measurement sets store antenna names in the STATION column.
        # cotter measurement sets store antenna names in the NAME column, which is
        # inline with the MS definition doc. In that case all the station names are
        # the same.
        if station_names_same:
            self.antenna_names = ant_names
        else:
            self.antenna_names = station_names

        ant_names = np.asarray(self.antenna_names)[ant_good_position].tolist()
        self.antenna_names = ant_names

        self.Nants_telescope = len(self.antenna_names)

        relative_positions = np.zeros_like(full_antenna_positions)
        relative_positions = full_antenna_positions - self.telescope_location.reshape(
            1, 3
        )
        self.antenna_positions = relative_positions

        tb_ant.close()
        tb_field = tables.table(filepath + "/FIELD", ack=False)

        # Error if the phase_dir has a polynomial term because we don't know
        # how to handle that
        message = (
            "PHASE_DIR is expressed as a polynomial. "
            "We do not currently support this mode, please make an issue."
        )
        assert tb_field.getcol("PHASE_DIR").shape[1] == 1, message

        self.phase_type = "phased"
        # MSv2.0 appears to assume J2000. Not sure how to specifiy otherwise
        epoch_string = tb.getcolkeyword("UVW", "MEASINFO")["Ref"]
        # for measurement sets made with COTTER, this keyword is ITRF
        # instead of the epoch
        if epoch_string == "ITRF":
            warnings.warn(
                "ITRF coordinate frame detected, although within cotter this is "
                "synonymous with J2000. Assuming J2000 coordinate frame."
            )
            self.phase_center_frame = "fk5"
            self.phase_center_epoch = 2000.0
        elif epoch_string == "J2000":
            # In CASA 'J2000' refers to a specific frame -- FK5 w/ an epoch of
            # J2000. We'll plug that in here directly, noting that CASA has an
            # explicit list of supported reference frames, located here:
            # casa.nrao.edu/casadocs/casa-5.0.0/reference-material/coordinate-frames
            self.phase_center_frame = "fk5"
            self.phase_center_epoch = 2000.0
        self.phase_center_ra = float(tb_field.getcol("PHASE_DIR")[0][0][0])
        self.phase_center_dec = float(tb_field.getcol("PHASE_DIR")[0][0][1])
        self._set_phased()

        # set LST array from times and itrf
        proc = self.set_lsts_from_time_array(background=background_lsts)

        # CASA weights column keeps track of number of data points averaged.

        # 'WEIGHT_SPECTRUM' is optional - some files may not have per-channel values
        if "WEIGHT_SPECTRUM" in tb.colnames():
            self.nsample_array = tb.getcol("WEIGHT_SPECTRUM")
        else:
            self.nsample_array = tb.getcol("WEIGHT")
            # Propagate the weights in frequency
            self.nsample_array = np.stack(
                [self.nsample_array for chan in range(self.Nfreqs)], axis=1
            )
        if len(self.nsample_array.shape) == 3:
            self.nsample_array = np.expand_dims(self.nsample_array, axis=1)
        self.object_name = tb_field.getcol("NAME")[0]
        tb_field.close()
        tb.close()

        if proc is not None:
            proc.join()
        # Fill in the apparent coordinates here
        self._set_app_coords_helper()

        # order polarizations
        self.reorder_pols(order=pol_order)
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )
