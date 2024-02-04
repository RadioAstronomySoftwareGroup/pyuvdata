# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Class for reading and writing casa measurement sets.

Requires casacore.
"""
import os
import warnings

import numpy as np
from astropy.time import Time
from docstring_parser import DocstringStyle

from .. import ms_utils
from .. import utils as uvutils
from ..docstrings import copy_replace_short_description
from .uvdata import UVData, _future_array_shapes_warning, reporting_request

__all__ = ["MS"]

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
try:
    import casacore.tables as tables
except ImportError as error:  # pragma: no cover
    casa_present = False
    casa_error = error


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

    def _init_ms_file(self, filepath):
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
            ms_desc[key].update(
                dataManagerType="IncrementalStMan", dataManagerGroup=key
            )

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

        if self.flex_spw_polarization_array is not None:
            pol_dict = {
                pol: idx
                for idx, pol in enumerate(np.unique(self.flex_spw_polarization_array))
            }
            data_descrip_table.putcol(
                "POLARIZATION_ID",
                np.array([pol_dict[key] for key in self.flex_spw_polarization_array]),
            )

        data_descrip_table.done()

    def _write_ms_feed(self, filepath, pol_order):
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
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        feed_table = tables.table(filepath + "::FEED", ack=False, readonly=False)

        nfeeds_table = np.max(self.antenna_numbers) + 1
        antenna_id_table = np.arange(nfeeds_table, dtype=np.int32)

        if self.flex_spw_polarization_array is None:
            spectral_window_id_table = -1 * np.ones(nfeeds_table, dtype=np.int32)

            # we want "x" or "y", *not* "e" or "n", so as not to confuse CASA
            pol_str = uvutils.polnum2str(self.polarization_array[pol_order])
        else:
            nfeeds_table *= self.Nspws
            spectral_window_id_table = np.repeat(
                np.arange(self.Nspws), np.max(self.antenna_numbers) + 1
            )
            antenna_id_table = np.tile(antenna_id_table, self.Nspws)
            # we want "x" or "y", *not* "e" or "n", so as not to confuse CASA
            pol_str = uvutils.polnum2str(self.flex_spw_polarization_array)

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
        feed_table.done()

    def _write_ms_source(self, filepath):
        """
        Write out the source information into a CASA table.

        Parameters
        ----------
        filepath : str
            path to MS (without SOURCE suffix)

        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        source_desc = tables.complete_ms_desc("SOURCE")
        source_table = tables.table(
            filepath + "::SOURCE",
            tabledesc=source_desc,
            dminfo=tables.makedminfo(source_desc),
            ack=False,
            readonly=False,
        )

        # Make the default time be the midpoint of the obs
        time_default = (
            Time(np.median(self.time_array), format="jd", scale="utc").mjd * 86400.0
        )
        # Default interval should be several times a Hubble time. Gotta make this code
        # future-proofed until the eventual heat death of the Universe.
        int_default = np.finfo(float).max

        row_count = 0
        for sou_id, pc_dict in self.phase_center_catalog.items():
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
                # If there are multile time entries, then approximate a value for the
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

        source_table.done()

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
        nants = np.max(self.antenna_numbers) + 1
        antennas = np.arange(nants, dtype=np.int32)

        # We want the number of unique timestamps here, since CASA wants a
        # per-timestamp, per-antenna entry
        times = np.asarray(
            [
                Time(t, format="jd", scale="utc").mjd * 86400.0
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
        # TODO: This works okay for steerable dishes, but less well for non-tracking
        # arrays (i.e., where primary beam center != phase center). Fix later.

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

    def _write_ms_polarization(self, filepath, pol_order):
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
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        pol_table = tables.table(filepath + "::POLARIZATION", ack=False, readonly=False)

        if self.flex_spw_polarization_array is None:
            pol_str = uvutils.polnum2str(self.polarization_array[pol_order])
            feed_pols = {
                feed for pol in pol_str for feed in uvutils.POL_TO_FEED_DICT[pol]
            }
            pol_types = [pol.lower() for pol in sorted(feed_pols)]
            pol_tuples = np.asarray(
                [
                    (pol_types.index(i.lower()), pol_types.index(j.lower()))
                    for pol in self.polarization_array
                    for i, j in [uvutils.POL_TO_FEED_DICT[uvutils.polnum2str(pol)]]
                ],
                dtype=np.int32,
            )

            pol_table.addrows()
            pol_table.putcell(
                "CORR_TYPE",
                0,
                np.array(
                    [
                        ms_utils.POL_AIPS2CASA_DICT[pol]
                        for pol in self.polarization_array[pol_order]
                    ]
                ),
            )
            pol_table.putcell("CORR_PRODUCT", 0, pol_tuples)
            pol_table.putcell("NUM_CORR", 0, self.Npols)
        else:
            for idx, spw_pol in enumerate(np.unique(self.flex_spw_polarization_array)):
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
                    "CORR_TYPE", idx, np.array([ms_utils.POL_AIPS2CASA_DICT[spw_pol]])
                )
                pol_table.putcell("CORR_PRODUCT", idx, pol_tuples)
                pol_table.putcell("NUM_CORR", idx, self.Npols)

        pol_table.done()

    def write_ms(
        self,
        filepath,
        *,
        force_phase=False,
        flip_conj=False,
        clobber=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=False,
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
        flip_conj : bool
            If set to True, and the UVW coordinates are flipped (i.e., multiplied by
            -1) and the visibilities are complex conjugated prior to write, such that
            the data are written with the "opposite" conjugation scheme to what UVData
            normally uses.  Note that this is only needed for specific subset of
            applications that read MS-formated data, and should only be used by expert
            users. Default is False.
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
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is False.
        """
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )

        if os.path.exists(filepath):
            if clobber:
                print("File exists; clobbering")
            else:
                raise IOError("File exists; skipping")

        # Determine polarization order for writing out in CASA standard order, check
        # if this order can be represented by a single slice.
        pol_order = uvutils.determine_pol_order(self.polarization_array, order="CASA")
        [pol_order], _ = uvutils._convert_to_slices(
            pol_order, max_nslice=1, return_index_on_fail=True
        )

        # CASA does not have a way to handle "unprojected" data in the way that UVData
        # objects can, so we need to check here whether or not any such data exists
        # (and if need be, fix it).
        # TODO: I thought CASA could handle driftscan data. Are we sure it can't handle
        # unprojected data?
        unprojected_blts = self._check_for_cat_type("unprojected")
        if np.any(unprojected_blts):
            if force_phase:
                print(
                    "The data are unprojected. Phasing to zenith of the first "
                    "timestamp."
                )
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time, select_mask=unprojected_blts)
            else:
                raise ValueError(
                    "The data are unprojected. "
                    "Set force_phase to true to phase the data "
                    "to zenith of the first timestamp before "
                    "writing a measurement set file."
                )

        # If scan numbers are not already defined from reading an MS,
        # group integrations (rows) into scan numbers.
        if self.scan_number_array is None:
            self._set_scan_numbers()

        # Initialize a skelton measurement set
        ms = self._init_ms_file(filepath)

        attr_list = ["data_array", "nsample_array", "flag_array"]
        col_list = ["DATA", "WEIGHT_SPECTRUM", "FLAG"]

        # Some tasks in CASA require a band-representative (band-averaged?) value for
        # the weights and noise for all channels in each row in the MAIN table, which
        # we will roughly calculate in temp_weights below.
        temp_weights = np.zeros((self.Nblts * self.Nspws, self.Npols), dtype=float)
        data_desc_array = np.zeros((self.Nblts * self.Nspws))

        # astropys Time has some overheads associated with it, so use unique to run
        # this date conversion as few times as possible. Note that the default for MS
        # is MJD UTC seconds, versus JD UTC days for UVData.
        time_array, time_ind = np.unique(self.time_array, return_inverse=True)
        # TODO: Verify this should actually be UTC, and not some other scale
        time_array = (Time(time_array, format="jd", scale="utc").mjd * 86400.0)[
            time_ind
        ]

        # Add all the rows we need up front, which will allow us to fill the
        # columns all in one shot.
        ms.addrows(self.Nblts * self.Nspws)
        if self.Nspws == 1:
            # If we only have one spectral window, there is nothing we need to worry
            # about ordering, so just write the data-related arrays as is to disk
            for attr, col in zip(attr_list, col_list):
                if self.future_array_shapes:
                    temp_vals = getattr(self, attr)[:, :, pol_order]
                else:
                    temp_vals = np.squeeze(getattr(self, attr), axis=1)[..., pol_order]

                if flip_conj and (attr == "data_array"):
                    temp_vals = np.conj(temp_vals)

                ms.putcol(col, temp_vals)

            # Band-averaged weights are used for some things in CASA - calculate them
            # here using median nsamples.
            if self.future_array_shapes:
                temp_weights = np.median(self.nsample_array, axis=1)
            else:
                temp_weights = np.squeeze(np.median(self.nsample_array, axis=2), axis=1)

            # Grab pointers for the per-blt record arrays
            ant_1_array = self.ant_1_array
            ant_2_array = self.ant_2_array
            integration_time = self.integration_time
            uvw_array = self.uvw_array * (-1 if flip_conj else 1)
            scan_number_array = self.scan_number_array
        else:
            # If we have _more_ than one spectral window, then we need to handle each
            # window separately, since they can have differing numbers of channels.
            # (n.b., tables.putvarcol can write complex tables like these, but its
            # slower and more memory-intensive than putcol).

            # Since muliple records trace back to a single baseline-time, we use this
            # array to map from arrays that store on a per-record basis to positions
            # within arrays that record metadata on a per-blt basis.
            blt_map_array = np.zeros((self.Nblts * self.Nspws), dtype=int)

            # we will select out individual spectral windows several times, so create
            # these masks once up front before we enter the loop.
            spw_sel_dict = {}
            for spw_id in self.spw_array:
                spw_sel_dict[spw_id] = self.flex_spw_id_array == spw_id

            # Based on some analysis of ALMA/ACA data, various routines in CASA appear
            # to prefer data be grouped together on a "per-scan" basis, then per-spw,
            # and then the more usual selections of per-time, per-ant1, etc.
            last_row = 0
            for scan_num in sorted(np.unique(self.scan_number_array)):
                # Select all data from the scan
                scan_screen = np.where(self.scan_number_array == scan_num)[0]

                # See if we can represent scan_screen with a single slice, which
                # reduces overhead of copying a new array.
                [scan_slice], _ = uvutils._convert_to_slices(
                    scan_screen, max_nslice=1, return_index_on_fail=True
                )

                # Get the number of records inside the scan, where 1 record = 1 spw in
                # 1 baseline at 1 time
                Nrecs = len(scan_screen)

                # Record which SPW/"Data Description" this data is matched to
                data_desc_array[last_row : last_row + (Nrecs * self.Nspws)] = np.repeat(
                    np.arange(self.Nspws), Nrecs
                )

                # Record index positions
                blt_map_array[last_row : last_row + (Nrecs * self.Nspws)] = np.tile(
                    scan_screen, self.Nspws
                )

                # Extract out the relevant data out of our data-like arrays that
                # belong to this scan number.
                val_dict = {}
                for attr, col in zip(attr_list, col_list):
                    if self.future_array_shapes:
                        val_dict[col] = getattr(self, attr)[scan_slice]
                    else:
                        val_dict[col] = np.squeeze(
                            getattr(self, attr)[scan_slice], axis=1
                        )

                    # Have to do this separately since uou can't supply multiple index
                    # arrays at once.
                    val_dict[col] = val_dict[col][:, :, pol_order]

                if flip_conj:
                    val_dict["DATA"] = np.conj(val_dict["DATA"])

                # This is where the bulk of the heavy lifting is - use the per-spw
                # channel masks to record one spectral window at a time.
                for spw_num in self.spw_array:
                    for col in col_list:
                        ms.putcol(
                            col,
                            val_dict[col][:, spw_sel_dict[spw_num]],
                            last_row,
                            Nrecs,
                        )

                    # Tally here the "wideband" weights for the whole spectral window,
                    # which is used in some CASA routines.
                    temp_weights[last_row : last_row + Nrecs] = np.median(
                        val_dict["WEIGHT_SPECTRUM"][:, spw_sel_dict[spw_num]], axis=1
                    )
                    last_row += Nrecs

            # Now that we have an array to map baselime-time to individual records,
            # use our indexing array to map various metadata.
            ant_1_array = self.ant_1_array[blt_map_array]
            ant_2_array = self.ant_2_array[blt_map_array]
            integration_time = self.integration_time[blt_map_array]
            time_array = time_array[blt_map_array]
            uvw_array = self.uvw_array[blt_map_array] * (-1 if flip_conj else 1)
            scan_number_array = self.scan_number_array[blt_map_array]

        # Write out the units of the visibilities, post a warning if its not in Jy since
        # we don't know how every CASA program may react
        ms.putcolkeyword("DATA", "QuantumUnits", self.vis_units)
        if self.vis_units != "Jy":
            warnings.warn(
                "Writing in the MS file that the units of the data are %s, although "
                "some CASA process will ignore this and assume the units are all in Jy "
                "(or may not know how to handle data in these units)." % self.vis_units
            )

        # TODO: If/when UVData objects can store visibility noise estimates, update
        # the code below to capture those.
        ms.putcol("WEIGHT", temp_weights)
        ms.putcol("SIGMA", np.power(temp_weights, -0.5, where=temp_weights != 0))

        ms.putcol("ANTENNA1", ant_1_array)
        ms.putcol("ANTENNA2", ant_2_array)

        # "INVERVAL" refers to "width" of the window of time time over which data was
        # collected, while "EXPOSURE" is the sum total of integration time.  UVData
        # does not differentiate between these concepts, hence why one array is used
        # for both values.
        ms.putcol("INTERVAL", integration_time)
        ms.putcol("EXPOSURE", integration_time)

        ms.putcol("DATA_DESC_ID", data_desc_array)
        ms.putcol("SCAN_NUMBER", scan_number_array)
        ms.putcol("TIME", time_array)
        ms.putcol("TIME_CENTROID", time_array)

        # FITS uvw direction convention is opposite ours and Miriad's.
        # CASA's convention is unclear: the docs contradict themselves,
        # but after a question to the helpdesk we got a clear response that
        # the convention is antenna2_pos - antenna1_pos, so the convention is the
        # same as ours & Miriad's
        ms.putcol("UVW", uvw_array)

        # We have to do an extra bit of work here, as CASA won't accept arbitrary
        # values for field ID (rather, the ID number matches to the row number in
        # the FIELD subtable). When we write out the fields, we use sort so that
        # we can reproduce the same ordering here.
        field_ids = np.empty_like(self.phase_center_id_array)
        for idx, cat_id in enumerate(self.phase_center_catalog):
            field_ids[self.phase_center_id_array == cat_id] = idx

        ms.putcol("FIELD_ID", field_ids[blt_map_array] if self.Nspws > 1 else field_ids)

        # Finally, record extra keywords and x_orientation, both of which the MS format
        # doesn't quite have equivalent fields to stuff data into (and instead is put
        # into the main header as a keyword).
        if len(self.extra_keywords) != 0:
            ms.putkeyword("pyuvdata_extra", self.extra_keywords)

        if self.x_orientation is not None:
            ms.putkeyword("pyuvdata_xorient", self.x_orientation)

        ms.done()

        ms_utils.write_ms_antenna(filepath, self)
        self._write_ms_data_description(filepath)
        self._write_ms_feed(filepath, pol_order=pol_order)
        ms_utils.write_ms_field(filepath, self)
        ms_utils.write_ms_history(filepath, self.history)
        ms_utils.write_ms_observation(filepath, self)
        self._write_ms_pointing(filepath)
        self._write_ms_polarization(filepath, pol_order=pol_order)
        self._write_ms_source(filepath)
        ms_utils.write_ms_spectral_window(filepath, self)

    def _read_ms_main(
        self,
        filepath,
        *,
        data_column,
        data_desc_dict,
        read_weights=True,
        flip_conj=False,
        raise_error=True,
        allow_flex_pol=False,
    ):
        """
        Read data from the main table of a MS file.

        This method is not meant to be called by users, and is instead a utility
        function for the `read_ms` method (which users should call instead).

        Parameters
        ----------
        filepath : str
            The measurement set root directory to read from.
        data_column : str
            name of CASA data column to read into data_array. Options are:
            'DATA', 'MODEL', or 'CORRECTED_DATA'
        data_desc_dict : dict
            Dictionary describing the various rows in the DATA_DESCRIPTION table of
            an MS file. Keys match to the individual rows, and the values are themselves
            dicts containing several keys (including "CORR_TYPE", "SPW_ID", "NUM_CORR",
            "NUM_CHAN").
        read_weights : bool
            Read in the weights from the MS file, default is True. If false, the method
            will set the `nsamples_array` to the same uniform value (namely 1.0).
        flip_conj : bool
            On read, whether to flip the conjugation of the baselines. Normally the MS
            format is the same as what is used for pyuvdata (ant2 - ant1), hence the
            default is False.
        raise_error : bool
            On read, whether to raise an error if different records (i.e.,
            different spectral windows) report different metadata for the same
            time-baseline combination (which CASA allows but UVData does not) or if the
            timescale is not supported by astropy. Default is True, if set to False will
            raise a warning instead.
        allow_flex_pol : bool
            If only one polarization per spectral window is read (and the polarization
            differs from window to window), compress down the polarization-axis of
            various attributes (e.g, `data_array`, `flag_array`) to be of length 1.
            Default is True.

        Returns
        -------
        spw_list : list of int
            List of SPW numbers present in the data set, equivalent to the attribute
            `spw_array` in a UVData object.
        field_list : list of int
            List of field IDs present in the data set. Matched to rows in the FIELD
            table for the measurement set.
        pol_list : list of int
            List of polarization IDs (in the AIPS convention) present in the data set.
            Equivalent to the attribute `polarization_array` in a UVData object.
        flex_pol : list of int
            If `allow_flex_pol=True`, and only one polarization per spectral window is
            read (differing window-to-window), list of the polarization IDs present
            for each window. Equivalent to the attribute `flex_spw_polarization_array`
            in a UVData object.

        Raises
        ------
        ValueError
            If the `data_column` is not set to an allowed value.
            If the MS file contains data from multiple subarrays.
        """
        tb_main = tables.table(filepath, ack=False)

        main_keywords = tb_main.getkeywords()
        if "pyuvdata_extra" in main_keywords.keys():
            self.extra_keywords = main_keywords["pyuvdata_extra"]

        if "pyuvdata_xorient" in main_keywords.keys():
            self.x_orientation = main_keywords["pyuvdata_xorient"]

        default_vis_units = {"DATA": "uncalib", "CORRECTED_DATA": "Jy", "MODEL": "Jy"}

        # make sure user requests a valid data_column
        if data_column not in default_vis_units.keys():
            raise ValueError(
                "Invalid data_column value supplied. Use 'Data','MODEL' or"
                " 'CORRECTED_DATA'"
            )

        # set visibility units
        try:
            self.vis_units = tb_main.getcolkeywords(data_column)["QuantumUnits"]
        except KeyError:
            self.vis_units = default_vis_units[data_column]

        # limit length of extra_keywords keys to 8 characters to match uvfits & miriad
        self.extra_keywords["DATA_COL"] = data_column

        time_arr = tb_main.getcol("TIME")
        timescale = tb_main.getcolkeyword("TIME", "MEASINFO")["Ref"]
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
        # N.b., EXPOSURE is what's needed for noise calculation, but INTERVAL defines
        # the time period over which the data are collected
        int_arr = tb_main.getcol("EXPOSURE")
        ant_1_arr = tb_main.getcol("ANTENNA1")
        ant_2_arr = tb_main.getcol("ANTENNA2")
        field_arr = tb_main.getcol("FIELD_ID")
        scan_number_arr = tb_main.getcol("SCAN_NUMBER")
        uvw_arr = tb_main.getcol("UVW")
        data_desc_arr = tb_main.getcol("DATA_DESC_ID")
        subarr_arr = tb_main.getcol("ARRAY_ID")
        unique_data_desc = np.unique(data_desc_arr)

        if len(np.unique(subarr_arr)) > 1:
            raise ValueError(
                "This file appears to have multiple subarray "
                "values; only files with one subarray are "
                "supported."
            )

        data_desc_count = np.sum(np.isin(list(data_desc_dict.keys()), unique_data_desc))

        if data_desc_count == 0:
            # If there are no records selected, then there isnt a whole lot to do
            return None, None, None, None
        elif data_desc_count == 1:
            # If we only have a single spectral window, then we can bypass a whole lot
            # of slicing and dicing on account of there being a one-to-one releationship
            # in rows of the MS to the per-blt records of UVData objects.
            self.time_array = Time(
                time_arr / 86400.0, format="mjd", scale=timescale.lower()
            ).utc.jd
            self.integration_time = int_arr
            self.ant_1_array = ant_1_arr
            self.ant_2_array = ant_2_arr
            self.uvw_array = uvw_arr * ((-1) ** flip_conj)
            self.phase_center_id_array = field_arr
            self.scan_number_array = scan_number_arr

            self.flag_array = tb_main.getcol("FLAG")

            if flip_conj:
                self.data_array = np.conj(tb_main.getcol(data_column))
            else:
                self.data_array = tb_main.getcol(data_column)

            if read_weights:
                try:
                    self.nsample_array = tb_main.getcol("WEIGHT_SPECTRUM")
                except RuntimeError:
                    self.nsample_array = np.repeat(
                        np.expand_dims(tb_main.getcol("WEIGHT"), axis=1),
                        self.data_array.shape[1],
                        axis=1,
                    )
            else:
                self.nsample_array = np.ones_like(self.data_array, dtype=float)

            data_desc_key = np.intersect1d(
                unique_data_desc, list(data_desc_dict.keys())
            )[0]
            spw_list = [data_desc_dict[data_desc_key]["SPW_ID"]]
            self.flex_spw_id_array = spw_list[0] + np.zeros(
                data_desc_dict[data_desc_key]["NUM_CHAN"], dtype=int
            )

            field_list = np.unique(field_arr)
            pol_list = [
                ms_utils.POL_CASA2AIPS_DICT[key]
                for key in data_desc_dict[data_desc_key]["CORR_TYPE"]
            ]

            tb_main.close()
            return spw_list, field_list, pol_list, None

        tb_main.close()

        # If you are at this point, it means that we potentially have multiple spectral
        # windows to deal with, and so some additional care is required since MS does
        # NOT require data from all windows to be present simultaneously.

        use_row = np.zeros_like(time_arr, dtype=bool)
        data_dict = {}
        for key in data_desc_dict.keys():
            sel_mask = data_desc_arr == key

            if not np.any(sel_mask):
                continue

            use_row[sel_mask] = True
            data_dict[key] = dict(data_desc_dict[key])
            data_dict[key]["TIME"] = time_arr[sel_mask]  # Midpoint time in mjd seconds
            data_dict[key]["EXPOSURE"] = int_arr[sel_mask]  # Int time in sec
            data_dict[key]["ANTENNA1"] = ant_1_arr[sel_mask]  # First antenna
            data_dict[key]["ANTENNA2"] = ant_2_arr[sel_mask]  # Second antenna
            data_dict[key]["FIELD_ID"] = field_arr[sel_mask]  # Source ID
            data_dict[key]["SCAN_NUMBER"] = scan_number_arr[sel_mask]  # Scan number
            data_dict[key]["UVW"] = uvw_arr[sel_mask]  # UVW coords

        time_arr = time_arr[use_row]
        ant_1_arr = ant_1_arr[use_row]
        ant_2_arr = ant_2_arr[use_row]

        unique_blts = sorted(
            {
                (time, ant1, ant2)
                for time, ant1, ant2 in zip(time_arr, ant_1_arr, ant_2_arr)
            }
        )

        blt_dict = {}
        for idx, blt_tuple in enumerate(unique_blts):
            blt_dict[blt_tuple] = idx

        nblts = len(unique_blts)
        pol_list = np.unique([data_dict[key]["CORR_TYPE"] for key in data_dict.keys()])
        npols = len(pol_list)

        spw_dict = {
            data_dict[key]["SPW_ID"]: {
                "DATA_DICT_KEY": key,
                "NUM_CHAN": data_dict[key]["NUM_CHAN"],
            }
            for key in data_dict.keys()
        }
        spw_list = sorted(spw_dict.keys())

        # Here we sort out where the various spectral windows are starting and stoping
        # in our flex_spw spectrum, if applicable. By default, data are sorted in
        # spw-number order.
        nfreqs = 0
        spw_id_array = np.array([], dtype=int)
        for key in sorted(spw_dict.keys()):
            assert len(data_dict) == len(
                spw_dict
            ), "This is a bug, please make an issue in our issue log."
            data_dict_key = spw_dict[key]["DATA_DICT_KEY"]
            nchan = spw_dict[key]["NUM_CHAN"]
            data_dict[data_dict_key]["STARTCHAN"] = nfreqs
            data_dict[data_dict_key]["STOPCHAN"] = nfreqs + nchan
            data_dict[data_dict_key]["NUM_CHAN"] = nchan
            spw_id_array = np.append(spw_id_array, [key] * nchan)
            nfreqs += nchan

        all_single_pol = True
        for key in sorted(data_dict.keys()):
            blt_idx = [
                blt_dict[(time, ant1, ant2)]
                for time, ant1, ant2 in zip(
                    data_dict[key]["TIME"],
                    data_dict[key]["ANTENNA1"],
                    data_dict[key]["ANTENNA2"],
                )
            ]

            data_dict[key]["BLT_IDX"] = np.array(blt_idx, dtype=int)
            data_dict[key]["NBLTS"] = len(blt_idx)

            pol_idx = np.intersect1d(
                pol_list,
                data_desc_dict[key]["CORR_TYPE"],
                assume_unique=True,
                return_indices=True,
            )[1]
            data_dict[key]["POL_IDX"] = pol_idx.astype(int)
            all_single_pol = all_single_pol and (len(pol_idx) == 1)

        pol_list = [ms_utils.POL_CASA2AIPS_DICT[key] for key in pol_list]
        flex_pol = None

        # Check to see if we want to allow flex pol, in which case each data_desc will
        # get assigned it's own spectral window with a potentially different
        # polarization per window (which we separately record).
        if (
            allow_flex_pol
            and all_single_pol
            and ((len(pol_list) > 1) and (len(data_desc_dict) == len(spw_dict)))
        ):
            for key in data_dict.keys():
                spw_dict[data_dict[key]["SPW_ID"]]["POL"] = pol_list[
                    data_dict[key]["POL_IDX"][0]
                ]
                data_dict[key]["POL_IDX"] = np.array([0])
            pol_list = np.array([0])
            npols = 1
            flex_pol = np.array(
                [spw_dict[key]["POL"] for key in sorted(spw_dict.keys())], dtype=int
            )

        # We have all of the meta-information linked the various data desc IDs,
        # so now we can finally get to the business of filling in the actual data.
        data_array = np.zeros((nblts, nfreqs, npols), dtype=complex)
        nsample_array = np.ones((nblts, nfreqs, npols))
        flag_array = np.ones((nblts, nfreqs, npols), dtype=bool)

        # We will also fill in our own metadata on a per-blt basis here
        time_arr = np.zeros(nblts)
        int_arr = np.zeros(nblts)
        ant_1_arr = np.zeros(nblts, dtype=int)
        ant_2_arr = np.zeros(nblts, dtype=int)
        field_arr = np.zeros(nblts, dtype=int)
        scan_number_arr = np.zeros(nblts, dtype=int)
        uvw_arr = np.zeros((nblts, 3))

        # Since each data description (i.e., spectral window) record can technically
        # have its own values for time, int-time, etc, we want to check and verify that
        # the values are consistent on a per-blt basis (since that's the most granular
        # pyuvdata can store that information).
        has_data = np.zeros(nblts, dtype=bool)

        arr_tuple = (
            time_arr,
            int_arr,
            ant_1_arr,
            ant_2_arr,
            field_arr,
            scan_number_arr,
            uvw_arr,
        )
        name_tuple = (
            "TIME",
            "EXPOSURE",
            "ANTENNA1",
            "ANTENNA2",
            "FIELD_ID",
            "SCAN_NUMBER",
            "UVW",
        )
        vary_list = []
        for key in data_dict.keys():
            # Get the indexing information for the data array
            blt_idx = data_dict[key]["BLT_IDX"]
            startchan = data_dict[key]["STARTCHAN"]
            stopchan = data_dict[key]["STOPCHAN"]
            pol_idx = data_dict[key]["POL_IDX"]

            # Identify which values have already been populated with data, so we know
            # which values to check.
            check_mask = has_data[blt_idx]
            check_idx = blt_idx[check_mask]

            # Loop through the metadata fields we intend to populate
            for arr, name in zip(arr_tuple, name_tuple):
                if not np.allclose(data_dict[key][name][check_mask], arr[check_idx]):
                    if raise_error:
                        raise ValueError(
                            "Column %s appears to vary on between windows, which is "
                            "not permitted for UVData objects. To bypass this error, "
                            "you can set raise_error=False, which will raise a warning "
                            "instead and use the first recorded value." % name
                        )
                    elif name not in vary_list:
                        # If not raising an error, then at least warn the user that
                        # discrepant data were detected.
                        warnings.warn(
                            "Column %s appears to vary on between windows, defaulting "
                            "to first recorded value." % name
                        )
                        # Add to a list so we don't spew multiple warnings for one
                        # column (which could just flood the terminal).
                        vary_list.append(name)

                arr[blt_idx[~check_mask]] = data_dict[key][name][~check_mask]

            # Can has data now please?
            has_data[blt_idx] = True

            # Remove a slice out of the larger arrays for us to populate with an MS read
            # operation. This has the advantage that if different data descrips contain
            # different polarizations (which is allowed), it will populate the arrays
            # correctly, although for most files (where all pols are written in one
            # data descrip), this shouldn't matter.
            temp_data = data_array[blt_idx, startchan:stopchan]
            temp_flags = flag_array[blt_idx, startchan:stopchan]
            if read_weights:
                temp_weights = nsample_array[blt_idx, startchan:stopchan]

            # This TaQL call allows the low-level C++ routines to handle mapping data
            # access, and returns a table object that _only_ has records matching our
            # request. This allows one to do a simple and fast getcol for reading the
            # data, flags, and weights, since they should all be the same shape on a
            # per-row basis for the same data description. Alternative read methods
            # w/ getcell, getvarcol, and per-row getcols produced way slower code.
            tb_main_sel = tables.taql(
                f"select from {filepath} where DATA_DESC_ID == {key}"  # nosec
            )

            # Fill in the temp arrays, and then plug them back into the main array.
            # Note that this operation has to be split in two because you can only use
            # advanced slicing on one axis (which both blt_idx and pol_idx require).
            if flip_conj:
                temp_data[:, :, pol_idx] = np.conj(tb_main_sel.getcol("DATA"))
            else:
                temp_data[:, :, pol_idx] = tb_main_sel.getcol("DATA")

            temp_flags[:, :, pol_idx] = tb_main_sel.getcol("FLAG")

            data_array[blt_idx, startchan:stopchan] = temp_data
            flag_array[blt_idx, startchan:stopchan] = temp_flags

            if read_weights:
                # The weights can be stored in a couple of different columns, but we
                # use a try/except here to capture two separate cases (that both will
                # produce runtime errors) -- when WEIGHT_SPECTRUM isn't a column, and
                # when it is BUT its unfilled (which causes getcol to throw an error).
                try:
                    temp_weights[:, :, pol_idx] = tb_main_sel.getcol("WEIGHT_SPECTRUM")
                except RuntimeError:
                    temp_weights[:, :, pol_idx] = np.repeat(
                        np.expand_dims(tb_main_sel.getcol("WEIGHT"), axis=1),
                        data_desc_dict[key]["NUM_CHAN"],
                        axis=1,
                    )
                nsample_array[blt_idx, startchan:stopchan, :] = temp_weights

            # Close the table, get ready for the next loop
            tb_main_sel.close()

        self.data_array = data_array
        self.flag_array = flag_array
        self.nsample_array = nsample_array

        self.ant_1_array = ant_1_arr
        self.ant_2_array = ant_2_arr

        self.time_array = Time(
            time_arr / 86400.0, format="mjd", scale=timescale.lower()
        ).utc.jd
        self.integration_time = int_arr
        self.uvw_array = uvw_arr * ((-1) ** flip_conj)
        self.phase_center_id_array = field_arr
        self.phase_center_id_array = field_arr
        self.scan_number_array = scan_number_arr
        self.flex_spw_id_array = spw_id_array

        field_list = np.unique(field_arr).astype(int).tolist()

        return spw_list, field_list, pol_list, flex_pol

    @copy_replace_short_description(UVData.read_ms, style=DocstringStyle.NUMPYDOC)
    def read_ms(
        self,
        filepath,
        *,
        data_column="DATA",
        pol_order="AIPS",
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        ignore_single_chan=True,
        raise_error=True,
        read_weights=True,
        allow_flex_pol=False,
        check_autos=True,
        fix_autos=True,
        use_future_array_shapes=False,
        astrometry_library=None,
    ):
        """Read in a casa measurement set."""
        if not casa_present:  # pragma: no cover
            raise ImportError(no_casa_message) from casa_error

        if not os.path.exists(filepath):
            raise IOError(filepath + " not found")
        # set filename variable
        basename = filepath.rstrip("/")
        self.filename = [os.path.basename(basename)]
        self._filename.form = (1,)

        # get the history info
        try:
            self.history, pyuvdata_written = ms_utils.read_ms_hist(
                filepath, self.pyuvdata_version_str, check_origin=True
            )
        except FileNotFoundError:
            self.history = self.pyuvdata_version_str
            pyuvdata_written = False

        data_desc_dict = {}
        tb_desc = tables.table(filepath + "/DATA_DESCRIPTION", ack=False)
        for idx in range(tb_desc.nrows()):
            data_desc_dict[idx] = {
                "SPECTRAL_WINDOW_ID": tb_desc.getcell("SPECTRAL_WINDOW_ID", idx),
                "POLARIZATION_ID": tb_desc.getcell("POLARIZATION_ID", idx),
                "FLAG_ROW": tb_desc.getcell("FLAG_ROW", idx),
            }
        tb_desc.close()

        # Polarization array
        tb_pol = tables.table(filepath + "/POLARIZATION", ack=False)
        for key in data_desc_dict.keys():
            pol_id = data_desc_dict[key]["POLARIZATION_ID"]
            data_desc_dict[key]["CORR_TYPE"] = tb_pol.getcell(
                "CORR_TYPE", pol_id
            ).astype(int)
            data_desc_dict[key]["NUM_CORR"] = tb_pol.getcell("NUM_CORR", pol_id)
        tb_pol.close()

        tb_spws = tables.table(filepath + "/SPECTRAL_WINDOW", ack=False)

        try:
            spw_id = tb_spws.getcol("ASSOC_SPW_ID")
            use_assoc_id = True
        except RuntimeError:
            use_assoc_id = False

        single_chan_list = []
        for key in data_desc_dict.keys():
            spw_id = data_desc_dict[key]["SPECTRAL_WINDOW_ID"]
            data_desc_dict[key]["CHAN_FREQ"] = tb_spws.getcell("CHAN_FREQ", spw_id)
            # beware! There are possibly 3 columns here that might be the correct one
            # to use: CHAN_WIDTH, EFFECTIVE_BW, RESOLUTION
            data_desc_dict[key]["CHAN_WIDTH"] = tb_spws.getcell("CHAN_WIDTH", spw_id)
            data_desc_dict[key]["NUM_CHAN"] = tb_spws.getcell("NUM_CHAN", spw_id)
            if data_desc_dict[key]["NUM_CHAN"] == 1:
                single_chan_list.append(key)
            if use_assoc_id:
                data_desc_dict[key]["SPW_ID"] = int(
                    tb_spws.getcell("ASSOC_SPW_ID", spw_id)[0]
                )
            else:
                data_desc_dict[key]["SPW_ID"] = spw_id

        if ignore_single_chan:
            for key in single_chan_list:
                del data_desc_dict[key]

        # FITS uvw direction convention is opposite ours and Miriad's.
        # CASA's convention is unclear: the docs contradict themselves,
        # but after a question to the helpdesk we got a clear response that
        # the convention is antenna2_pos - antenna1_pos, so the convention is the
        # same as ours & Miriad's
        # HOWEVER, it appears that CASA's importuvfits task does not make this
        # convention change. So if the data in the MS came via that task and was not
        # written by pyuvdata, we do need to flip the uvws & conjugate the data
        flip_conj = ("importuvfits" in self.history) and (not pyuvdata_written)
        spw_list, field_list, pol_list, flex_pol = self._read_ms_main(
            filepath,
            data_column=data_column,
            data_desc_dict=data_desc_dict,
            read_weights=read_weights,
            flip_conj=flip_conj,
            raise_error=raise_error,
            allow_flex_pol=allow_flex_pol,
        )

        if (spw_list is None) and (field_list is None) and (pol_list is None):
            raise ValueError(
                "No valid data available in the MS file. If this file contains "
                "single channel data, set ignore_single_chan=False when calling "
                "read_ms."
            )

        self.Npols = len(pol_list)
        self.polarization_array = np.array(pol_list, dtype=np.int64)
        self.Nspws = len(spw_list)
        self.spw_array = np.array(spw_list, dtype=np.int64)
        self.flex_spw_polarization_array = flex_pol

        self.Nfreqs = len(self.flex_spw_id_array)
        self.freq_array = np.zeros(self.Nfreqs)
        self.channel_width = np.zeros(self.Nfreqs)

        for key in data_desc_dict.keys():
            sel_mask = self.flex_spw_id_array == data_desc_dict[key]["SPW_ID"]
            self.freq_array[sel_mask] = data_desc_dict[key]["CHAN_FREQ"]
            self.channel_width[sel_mask] = data_desc_dict[key]["CHAN_WIDTH"]

        # This if/else can go away in version 3.0 when channel width will always be an
        # array and the flex_spw_id_array will always be required.
        if (np.unique(self.channel_width).size == 1) and (len(spw_list) == 1):
            self.channel_width = float(self.channel_width[0])
        else:
            self._set_flex_spw()

        self.Ntimes = int(np.unique(self.time_array).size)
        self.Nblts = int(self.data_array.shape[0])
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

        # open table with antenna location information
        tb_ant = tables.table(filepath + "/ANTENNA", ack=False)
        tb_obs = tables.table(filepath + "/OBSERVATION", ack=False)
        self.telescope_name = tb_obs.getcol("TELESCOPE_NAME")[0]
        self.instrument = tb_obs.getcol("TELESCOPE_NAME")[0]
        self.extra_keywords["observer"] = tb_obs.getcol("OBSERVER")[0]
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
        if (
            "TELESCOPE_LOCATION" not in tb_obs.colnames()
            and self.telescope_name in self.known_telescopes()
        ):
            # get it from known telescopes
            self.set_telescope_params()
        else:
            if xyz_telescope_frame == "ITRF":
                # MS uses "ITRF" while astropy uses "itrs". They are the same.
                self._telescope_location.frame = "itrs"
            else:
                if xyz_telescope_frame.lower() not in ["itrs", "mcmf"]:
                    raise ValueError(
                        f"Telescope frame in file is {xyz_telescope_frame.lower()}. "
                        "Only 'itrs' and 'mcmf' are currently supported."
                    )
                self._telescope_location.frame = xyz_telescope_frame.lower()

            if "TELESCOPE_LOCATION" in tb_obs.colnames():
                self.telescope_location = np.squeeze(
                    tb_obs.getcol("TELESCOPE_LOCATION")
                )
            else:
                # Set it to be the mean of the antenna positions (this is not ideal!)
                self.telescope_location = np.array(
                    np.mean(full_antenna_positions, axis=0)
                )
        tb_obs.close()

        # antenna names
        ant_names = np.asarray(tb_ant.getcol("NAME"))[ant_good_position].tolist()
        station_names = np.asarray(tb_ant.getcol("STATION"))[ant_good_position].tolist()
        antenna_diameters = tb_ant.getcol("DISH_DIAMETER")[ant_good_position]
        if np.any(antenna_diameters > 0):
            self.antenna_diameters = antenna_diameters

        # importuvfits measurement sets store antenna names in the STATION column.
        # cotter measurement sets store antenna names in the NAME column, which is
        # inline with the MS definition doc. In that case all the station names are
        # the same. Default to using what the MS definition doc specifies, unless
        # we read importuvfits in the history, or if the antenna column is not filled.
        if ("importuvfits" not in self.history) and (
            len(ant_names) == len(np.unique(ant_names)) and ("" not in ant_names)
        ):
            self.antenna_names = ant_names
        else:
            self.antenna_names = station_names

        self.Nants_telescope = len(self.antenna_names)

        relative_positions = np.zeros_like(full_antenna_positions)
        relative_positions = full_antenna_positions - self.telescope_location.reshape(
            1, 3
        )
        self.antenna_positions = relative_positions

        tb_ant.close()

        # set LST array from times and itrf
        proc = self.set_lsts_from_time_array(
            background=background_lsts, astrometry_library=astrometry_library
        )

        tb_field = tables.table(filepath + "/FIELD", ack=False)

        # Error if the phase_dir has a polynomial term because we don't know
        # how to handle that
        message = (
            "PHASE_DIR is expressed as a polynomial. "
            "We do not currently support this mode, please make an issue."
        )
        assert tb_field.getcol("PHASE_DIR").shape[1] == 1, message

        # The SOURCE table is optional in MS, but can contain some relevant metadata
        # about
        tb_sou_dict = {}
        try:
            tb_source = tables.table(filepath + "/SOURCE", ack=False)
        except RuntimeError:
            # The SOURCE table is optional, so if not found a RuntimeError will be
            # thrown, and we should forgo trying to associate SOURCE table entries with
            # the FIELD table.
            pass
        else:
            for idx in range(tb_source.nrows()):
                sou_id = tb_source.getcell("SOURCE_ID", idx)
                pm_vec = tb_source.getcell("PROPER_MOTION", idx)
                time_stamp = tb_source.getcell("TIME", idx)
                sou_vec = tb_source.getcell("DIRECTION", idx)
                try:
                    for idx in np.where(
                        np.isclose(
                            tb_sou_dict[sou_id]["cat_times"],
                            time_stamp,
                            rtol=0,
                            atol=1e-3,
                        )
                    )[0]:
                        if not (
                            (tb_sou_dict[sou_id]["cat_ra"][idx] == sou_vec[0])
                            and (tb_sou_dict[sou_id]["cat_dec"][idx] == sou_vec[1])
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
                        _ = tb_sou_dict[sou_id]["cat_ra"].pop(idx)
                        _ = tb_sou_dict[sou_id]["cat_dec"].pop(idx)
                        _ = tb_sou_dict[sou_id]["cat_pm_ra"].pop(idx)
                        _ = tb_sou_dict[sou_id]["cat_pm_dec"].pop(idx)
                    tb_sou_dict[sou_id]["cat_times"].append(time_stamp)
                    tb_sou_dict[sou_id]["cat_ra"].append(sou_vec[0])
                    tb_sou_dict[sou_id]["cat_dec"].append(sou_vec[1])
                    tb_sou_dict[sou_id]["cat_pm_ra"].append(pm_vec[0])
                    tb_sou_dict[sou_id]["cat_pm_dec"].append(pm_vec[1])
                except KeyError:
                    tb_sou_dict[sou_id] = {
                        "cat_times": [time_stamp],
                        "cat_ra": [sou_vec[0]],
                        "cat_dec": [sou_vec[1]],
                        "cat_pm_ra": [pm_vec[0]],
                        "cat_pm_dec": [pm_vec[1]],
                    }

            for cat_dict in tb_sou_dict.values():
                make_arr = len(cat_dict["cat_times"]) != 1
                if not make_arr:
                    del cat_dict["cat_times"]

                for key in cat_dict:
                    if make_arr:
                        cat_dict[key] = np.array(cat_dict[key])
                    else:
                        cat_dict[key] = cat_dict[key][0]
                if np.allclose(cat_dict["cat_pm_ra"], 0):
                    if np.allclose(cat_dict["cat_pm_dec"], 0):
                        cat_dict["cat_pm_ra"] = cat_dict["cat_pm_dec"] = None

        # MSv2.0 appears to assume J2000. Not sure how to specifiy otherwise
        measinfo_keyword = tb_field.getcolkeyword("PHASE_DIR", "MEASINFO")

        ref_dir_colname = None
        ref_dir_dict = None
        if "VarRefCol" in measinfo_keyword.keys():
            # This seems to be a yet-undocumented feature for CASA, which allows one
            # to specify an additional (optional?) column that defines the reference
            # frame on a per-source basis.
            ref_dir_colname = measinfo_keyword["VarRefCol"]
            ref_dir_dict = dict(
                zip(measinfo_keyword["TabRefCodes"], measinfo_keyword["TabRefTypes"])
            )

        if "Ref" in measinfo_keyword.keys():
            frame_tuple = ms_utils._parse_casa_frame_ref(measinfo_keyword["Ref"])
            phase_center_frame = frame_tuple[0]
            phase_center_epoch = frame_tuple[1]
        else:
            warnings.warn("Coordinate reference frame not detected, defaulting to ICRS")
            phase_center_frame = "icrs"
            phase_center_epoch = 2000.0

        field_id_dict = {field_idx: field_idx for field_idx in field_list}
        try:
            id_arr = tb_field.getcol("SOURCE_ID")
            if np.all(id_arr >= 0) and len(np.unique(id_arr)) == len(id_arr):
                for idx, sou_id in enumerate(id_arr):
                    if idx in field_list:
                        field_id_dict[idx] = sou_id
        except RuntimeError:
            # Reach here if no column named SOURCE_ID exists, or if it does exist
            # but is completely unfilled. Nothing to do at this point but move on.
            tb_sou_dict = {}
            pass
        # Field names are allowed to be the same in CASA, so if we detect
        # conflicting names here we use the FIELD row numbers to try and
        # differentiate between them.
        field_name_list = tb_field.getcol("NAME")
        uniq_names, uniq_count = np.unique(field_name_list, return_counts=True)
        rep_name_list = uniq_names[uniq_count > 1]
        for rep_name in rep_name_list:
            rep_count = 0
            for idx in range(len(field_name_list)):
                if field_name_list[idx] == rep_name:
                    field_name_list[idx] = "%s-%03i" % (field_name_list[idx], rep_count)
                    rep_count += 1

        for field_idx in field_list:
            ra_val, dec_val = tb_field.getcell("PHASE_DIR", field_idx)[0]
            field_name = field_name_list[field_idx]
            if ref_dir_colname is not None:
                frame_tuple = ms_utils._parse_casa_frame_ref(
                    ref_dir_dict[tb_field.getcell(ref_dir_colname, field_list[0])]
                )
            else:
                frame_tuple = (phase_center_frame, phase_center_epoch)

            pm_ra = pm_dec = ephem_times = None
            if field_id_dict[field_idx] in tb_sou_dict:
                cat_dict = tb_sou_dict[field_id_dict[field_idx]]
                ra_val, dec_val = cat_dict["cat_ra"], cat_dict["cat_dec"]
                pm_ra, pm_dec = cat_dict["cat_pm_ra"], cat_dict["cat_pm_dec"]
                if "cat_times" in cat_dict:
                    ephem_times = Time(
                        cat_dict["cat_times"] / 86400, format="mjd", scale="utc"
                    ).jd

            self._add_phase_center(
                field_name,
                cat_type="sidereal" if (ephem_times is None) else "ephem",
                cat_lon=ra_val,
                cat_lat=dec_val,
                cat_times=ephem_times,  # Convert Julian secs to JD
                cat_frame=frame_tuple[0],
                cat_epoch=frame_tuple[1],
                cat_pm_ra=pm_ra,
                cat_pm_dec=pm_dec,
                info_source="file",
                cat_id=field_id_dict[field_idx],
            )
        # Only thing left to do is to update the IDs for the per-BLT records
        self.phase_center_id_array = np.array(
            [field_id_dict[sou_id] for sou_id in self.phase_center_id_array], dtype=int
        )

        tb_field.close()

        if proc is not None:
            proc.join()
        # Fill in the apparent coordinates here
        self._set_app_coords_helper()

        self.data_array = np.expand_dims(self.data_array, 1)
        self.nsample_array = np.expand_dims(self.nsample_array, 1)
        self.flag_array = np.expand_dims(self.flag_array, 1)
        self.freq_array = np.expand_dims(self.freq_array, 0)

        # order polarizations
        if pol_order is not None:
            self.reorder_pols(order=pol_order, run_check=False)

        if use_future_array_shapes:
            self.use_future_array_shapes()
        else:
            warnings.warn(_future_array_shapes_warning, DeprecationWarning)

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )
