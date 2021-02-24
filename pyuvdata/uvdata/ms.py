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
import astropy.time as time

from .uvdata import UVData
from .. import utils as uvutils

__all__ = ["MS"]

no_casa_message = (
    "casacore is not installed but is required for " "measurement set functionality"
)

casa_present = True
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

        filepath: path to MS (without ANTENNA suffix)
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        antenna_table = tables.table(filepath + "::ANTENNA", ack=False, readonly=False)
        antenna_table.addrows(self.Nants_data)
        antenna_table.putcol("NAME", self.antenna_names)
        # TODO check that this works out!
        ant_pos_absolute = self.antenna_positions + self.telescope_location.reshape(
            1, 3
        )
        ant_pos_rot_ecef = uvutils.rotECEF_from_ECEF(
            ant_pos_absolute, self.telescope_location_latlonalt[1]
        )

        antenna_table.putcol("POSITION", ant_pos_rot_ecef)
        if self.antenna_diameters is not None:
            antenna_table.putcol("DISH_DIAMETER", self.antenna_diameters)
        antenna_table.done()

    def _write_ms_field(self, filepath):
        """
        Write out the field information into a CASA table.

        filepath: path to MS (without FIELD suffix)
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

    def _write_ms_spectralwindow(self, filepath):
        """
        Write out the spectral information into a CASA table.

        filepath: path to MS (without SPECTRAL_WINDOW suffix)
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        # TODO fix for flexible spws
        tables.taql(
            "insert into {}::DATA_DESCRIPTION SET FLAG_ROW=False, "
            "POLARIZATION_ID=0, SPECTRAL_WINDOW_ID=0".format(filepath)
        )

        sw_table = tables.table(
            filepath + "::SPECTRAL_WINDOW", ack=False, readonly=False
        )

        sw_table.addrows()
        sw_table.putcell("CHAN_FREQ", 0, self.freq_array[0])
        # TODO fix for future array shapes
        chanwidths = np.ones_like(self.freq_array[0]) * self.channel_width
        sw_table.putcell("CHAN_WIDTH", 0, chanwidths)
        sw_table.putcell("EFFECTIVE_BW", 0, chanwidths)
        sw_table.putcell("RESOLUTION", 0, chanwidths)
        sw_table.putcell("NUM_CHAN", 0, self.Nfreqs)

    def _write_ms_observation(self, filepath):
        """
        Write out the observation information into a CASA table.

        filepath: path to MS (without OBSERVATION suffix)
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        antenna_table = tables.table(
            filepath + "::OBSERVATION", ack=False, readonly=False
        )
        antenna_table.addrows()
        antenna_table.putcell("TELESCOPE_NAME", 0, self.telescope_name)
        # should we test for observer in extra keywords?
        antenna_table.putcell("OBSERVER", 0, self.telescope_name)

    def _write_ms_polarization(self, filepath):
        """
        Write out the polarization information into a CASA table.

        filepath: path to MS (without POLARIZATION suffix)
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        pol_table = tables.table(filepath + "::POLARIZATION", ack=False, readonly=False)
        pol_table.addrows()
        pol_table.putcell(
            "CORR_TYPE",
            0,
            np.array(
                [POL_AIPS2CASA_DICT[aipspol] for aipspol in self.polarization_array]
            ),
        )
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
            string containing only message column (consistent with other UVDATA
            history strings)
        str
            string enconding complete casa history table converted with a new
            line denoting rows and a ';' denoting column breaks.
        """
        # string to store just the usual uvdata history
        message_str = ""
        # string to store all the casa history info
        history_str = ""

        # Do not touch the history table if it has no information
        if history_table.nrows() > 0:
            history_str = (
                "APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE;"
                "OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n"
            )

            app_params = history_table.getcol("APP_PARAMS")["array"]
            # might need to handle the case where cli_command is empty
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
            tables = [
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
            for tbrow in range(ntimes):
                message_str += str(message[tbrow])
                newline = ";".join([str(table[tbrow]) for table in tables]) + "\n"
                history_str += newline
                if tbrow < ntimes - 1:
                    message_str += "\n"

        def is_not_ascii(s):
            return any(ord(c) >= 128 for c in s)

        def find_not_ascii(s):
            output = []
            for c in s:
                if ord(c) >= 128:
                    output += c
            return output

        return message_str, history_str

    def write_ms(
        self,
        filepath,
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

        ms.putcol("DATA", np.squeeze(self.data_array, axis=1))
        ms.putcol("WEIGHT_SPECTRUM", np.squeeze(self.nsample_array, axis=1))
        ms.putcol("ANTENNA1", self.ant_1_array)
        ms.putcol("ANTENNA2", self.ant_2_array)
        ms.putcol("INTERVAL", self.integration_time)

        ms.putcol("TIME", time.Time(self.time_array, format="jd").mjd * 3600.0 * 24.0)
        ms.putcol("UVW", -self.uvw_array)
        ms.putcol("FLAG", np.squeeze(self.flag_array, axis=1))
        ms.done()

        self._write_ms_antenna(filepath)
        self._write_ms_field(filepath)
        self._write_ms_spectralwindow(filepath)
        self._write_ms_polarization(filepath)
        self._write_ms_observation(filepath)

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
        times_unique = time.Time(
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
        # but empirically it appears to match uvfits
        # So conjugate the visibilities and flip the uvws:
        data_array = np.conj(tb.getcol(data_column))
        self.Nblts = int(data_array.shape[0])
        flag_array = tb.getcol("FLAG")
        # CASA stores data in complex array with dimension NbltsxNfreqsxNpols
        if len(data_array.shape) == 3:
            data_array = np.expand_dims(data_array, axis=1)
            flag_array = np.expand_dims(flag_array, axis=1)
        self.data_array = data_array
        self.flag_array = flag_array
        self.Npols = int(data_array.shape[-1])
        # FITS uvw direction convention is opposite ours and Miriad's.
        # CASA's convention is unclear: the docs contradict themselves,
        # but empirically it appears to match uvfits
        # So conjugate the visibilities and flip the uvws:
        self.uvw_array = -1 * tb.getcol("UVW")
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
        self.time_array = time.Time(
            tb.getcol("TIME") / (3600.0 * 24.0), format="mjd"
        ).jd

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
        tb_obs.close()
        # Use Telescopes.py dictionary to set array position
        full_antenna_positions = tb_ant.getcol("POSITION")
        xyz_telescope_frame = tb_ant.getcolkeyword("POSITION", "MEASINFO")["Ref"]
        ant_flags = np.empty(len(full_antenna_positions), dtype=bool)
        ant_flags[:] = False
        for antnum in range(len(ant_flags)):
            ant_flags[antnum] = np.all(full_antenna_positions[antnum, :] == 0)
        if xyz_telescope_frame == "ITRF":
            self.telescope_location = np.array(
                np.mean(full_antenna_positions[np.invert(ant_flags), :], axis=0)
            )
        if self.telescope_location is None:
            try:
                self.set_telescope_params()
            except ValueError:
                warnings.warn(
                    "Telescope frame is not ITRF and telescope is not "
                    "in known_telescopes, so telescope_location is not set."
                )

        # antenna names
        ant_names = tb_ant.getcol("STATION")
        ant_diams = tb_ant.getcol("DISH_DIAMETER")

        self.antenna_diameters = ant_diams[ant_diams > 0]

        self.Nants_telescope = len(ant_flags[np.invert(ant_flags)])
        test_name = ant_names[0]
        names_same = True
        for antnum in range(len(ant_names)):
            if not (ant_names[antnum] == test_name):
                names_same = False
        if not (names_same):
            # cotter measurement sets store antenna names in the NAMES column.
            self.antenna_names = ant_names
        else:
            # importuvfits measurement sets store antenna names in the STATION column.
            self.antenna_names = tb_ant.getcol("NAME")
        self.antenna_numbers = np.arange(len(self.antenna_names)).astype(int)
        ant_names = []
        for ant_num in range(len(self.antenna_names)):
            if not (ant_flags[ant_num]):
                ant_names.append(self.antenna_names[ant_num])
        self.antenna_names = ant_names
        self.antenna_numbers = self.antenna_numbers[np.invert(ant_flags)]

        relative_positions = np.zeros_like(full_antenna_positions)
        relative_positions = full_antenna_positions - self.telescope_location.reshape(
            1, 3
        )
        self.antenna_positions = relative_positions[np.invert(ant_flags), :]

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

        # set the history parameter
        _, self.history = self._ms_hist_to_string(
            tables.table(filepath + "/HISTORY", ack=False)
        )
        # CASA weights column keeps track of number of data points averaged.

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str
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
