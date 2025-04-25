# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Class for reading and writing casa measurement sets.

Requires casacore.
"""

import contextlib
import os
import warnings

import numpy as np
from astropy.time import Time
from docstring_parser import DocstringStyle

from .. import utils
from ..docstrings import copy_replace_short_description
from ..utils.io import ms as ms_utils
from . import UVData

__all__ = ["MS"]

no_casa_message = (
    "casacore is not installed but is required for measurement set functionality"
)

casa_present = True
try:
    import casacore.tables as tables
except ImportError as error:
    casa_present = False
    casa_error = error


class MS(UVData):
    """
    Defines a class for reading and writing casa measurement sets.

    This class should not be interacted with directly, instead use the read_ms
    method on the UVData class.

    """

    @copy_replace_short_description(UVData.write_ms, style=DocstringStyle.NUMPYDOC)
    def write_ms(
        self,
        filepath,
        *,
        force_phase=False,
        model_data=None,
        corrected_data=None,
        flip_conj=None,
        clobber=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=False,
    ):
        """Write a CASA measurement set (MS)."""
        if not casa_present:
            raise ImportError(no_casa_message) from casa_error

        if any(
            entry.get("cat_type") == "near_field"
            for entry in self.phase_center_catalog.values()
        ):
            raise NotImplementedError(
                "Writing near-field phased data to Measurement Set format "
                + "is not yet supported."
            )

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
                raise OSError("File exists; skipping")

        # Determine polarization order for writing out in CASA standard order, check
        # if this order can be represented by a single slice.
        pol_order = utils.pol.determine_pol_order(self.polarization_array, order="CASA")
        pol_order = utils.tools.slicify(pol_order, allow_empty=True)

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

        if flip_conj is None:
            flip_conj = np.all(self.ant_1_array <= self.ant_2_array)
            if np.any(self.ant_1_array < self.ant_2_array) != flip_conj:
                warnings.warn(
                    "UVData object contains a mix of baseline conjugation states, "
                    "which is not uniformly supported in CASA -- forcing conjugation "
                    'to be "ant2<ant1" on object.'
                )
                self.conjugate_bls("ant2<ant1")

        # Initialize a skelton measurement set
        ms = ms_utils.init_ms_file(
            filepath,
            make_model_col=model_data is not None,
            make_corr_col=corrected_data is not None,
        )

        arr_list = [self.data_array, self.nsample_array, self.flag_array]
        col_list = ["DATA", "WEIGHT_SPECTRUM", "FLAG"]

        if model_data is not None:
            assert model_data.shape == self.data_array.shape, (
                "model_data must have the same shape as data_array."
            )
            arr_list.append(model_data)
            col_list.append("MODEL_DATA")
        if corrected_data is not None:
            assert corrected_data.shape == self.data_array.shape, (
                "corrected_data must have the same shape as data_array."
            )
            arr_list.append(corrected_data)
            col_list.append("CORRECTED_DATA")

        # Some tasks in CASA require a band-representative (band-averaged?) value for
        # the weights and noise for all channels in each row in the MAIN table, which
        # we will roughly calculate in temp_weights below.
        temp_weights = np.zeros((self.Nblts * self.Nspws, self.Npols), dtype=float)
        data_desc_array = np.zeros(self.Nblts * self.Nspws)

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
            for arr, col in zip(arr_list, col_list, strict=True):
                temp_vals = arr[:, :, pol_order]

                if flip_conj and ("DATA" in col):
                    temp_vals = np.conj(temp_vals)

                ms.putcol(col, temp_vals)

            # Band-averaged weights are used for some things in CASA - calculate them
            # here using median nsamples.
            temp_weights = np.median(self.nsample_array, axis=1)

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

            # Since multiple records trace back to a single baseline-time, we use this
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
                scan_slice = utils.tools.slicify(scan_screen, allow_empty=True)

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
                for arr, col in zip(arr_list, col_list, strict=True):
                    temp_arr = arr[scan_slice]

                    if flip_conj and ("DATA" in col):
                        temp_arr = np.conjugate(temp_arr)

                    # Have to do this separately since uou can't supply multiple index
                    # arrays at once.
                    val_dict[col] = temp_arr[:, :, pol_order]

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

            # Now that we have an array to map baseline-time to individual records,
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
                "Writing in the MS file that the units of the data are "
                f"{self.vis_units}, although some CASA process will ignore this "
                "and assume the units are all in Jy (or may not know how to "
                "handle data in these units)."
            )

        # TODO: If/when UVData objects can store visibility noise estimates, update
        # the code below to capture those.
        ms.putcol("WEIGHT", temp_weights)
        ms.putcol("SIGMA", np.power(temp_weights, -0.5, where=temp_weights != 0))

        ms.putcol("ANTENNA1", ant_1_array)
        ms.putcol("ANTENNA2", ant_2_array)

        # "INTERVAL" refers to "width" of the window of time time over which data was
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

        # Finally, record extra keywords and pol_convention, both of which the MS format
        # doesn't quite have equivalent fields to stuff data into (and instead is put
        # into the main header as a keyword).
        if len(self.extra_keywords) != 0:
            ms.putkeyword("pyuvdata_extra", self.extra_keywords)

        if self.pol_convention is not None:
            ms.putkeyword("pyuvdata_polconv", self.pol_convention)

        ms.putkeyword("pyuvdata_flip_conj", flip_conj)

        ms.done()

        ms_utils.write_ms_antenna(filepath, uvobj=self)
        ms_utils.write_ms_data_description(filepath, uvobj=self)
        ms_utils.write_ms_feed(filepath, uvobj=self)
        ms_utils.write_ms_field(filepath, uvobj=self)
        ms_utils.write_ms_history(filepath, uvobj=self)
        ms_utils.write_ms_observation(filepath, uvobj=self)
        ms_utils.write_ms_pointing(filepath, uvobj=self)
        ms_utils.write_ms_polarization(filepath, pol_order=pol_order, uvobj=self)
        ms_utils.write_ms_source(filepath, uvobj=self)
        ms_utils.write_ms_spectral_window(filepath, uvobj=self)

    def _read_ms_main(
        self,
        filepath,
        *,
        data_column,
        data_desc_dict,
        read_weights=True,
        pyuvdata_written=False,
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
            'DATA', 'MODEL_DATA', or 'CORRECTED_DATA'
        data_desc_dict : dict
            Dictionary describing the various rows in the DATA_DESCRIPTION table of
            an MS file. Keys match to the individual rows, and the values are themselves
            dicts containing several keys (including "CORR_TYPE", "SPW_ID", "NUM_CORR",
            "NUM_CHAN").
        read_weights : bool
            Read in the weights from the MS file, default is True. If false, the method
            will set the `nsamples_array` to the same uniform value (namely 1.0).
        pyuvdata_written : bool
            Whether the file in question was written by pyuvdata. Used in part to
            determine conjugation of the baselines.
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
        self.extra_keywords = main_keywords.get("pyuvdata_extra", {})
        self.pol_convention = main_keywords.get("pyuvdata_polconv", None)
        x_orientation = main_keywords.get("pyuvdata_xorient", None)
        flip_conj = main_keywords.get("pyuvdata_flip_conj", None)

        default_vis_units = {
            "DATA": "uncalib",
            "CORRECTED_DATA": "Jy",
            "MODEL": "Jy",
            "MODEL_DATA": "Jy",
        }

        # make sure user requests a valid data_column
        if data_column not in default_vis_units:
            raise ValueError(
                "Invalid data_column value supplied. Use 'DATA','MODEL_DATA', or"
                "'CORRECTED_DATA'."
            )

        # set visibility units
        try:
            self.vis_units = tb_main.getcolkeywords(data_column)["QuantumUnits"]
        except KeyError:
            self.vis_units = default_vis_units[data_column]

        # limit length of extra_keywords keys to 8 characters to match uvfits & miriad
        self.extra_keywords["DATA_COL"] = data_column

        time_arr = tb_main.getcol("TIME")
        timescale = ms_utils._get_time_scale(tb_main, raise_error=raise_error)

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

        if flip_conj is None:
            # if we got to this point, it means that the conjugation scheme has not
            # been encoded into the dataset, which is _either_ and old pyuvdata written
            # file or written external to pyuvdata. CASA's convention is not 100% clear,
            # but testing of the code base reveals that CASA supports both conventions.
            # Which convention is used is dependent on antenna numbering, i.e. whether
            # ant1 >= ant2 or ant1 <= ant2 (flip_conj=False for the former and True for
            # the latter). This seems to explain the apparent contradictions in the
            # documentation, and the inconsistent results we have seen w/ importuvfits.
            flip_conj = (not pyuvdata_written) and np.all(ant_1_arr <= ant_2_arr)

        data_desc_count = np.sum(np.isin(list(data_desc_dict.keys()), unique_data_desc))

        if data_desc_count == 0:
            # If there are no records selected, then there isn't a whole lot to do
            return None, None, None, None, None
        elif data_desc_count == 1:
            # If we only have a single spectral window, then we can bypass a whole lot
            # of slicing and dicing on account of there being a one-to-one relationship
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
            return spw_list, field_list, pol_list, None, x_orientation

        tb_main.close()

        # If you are at this point, it means that we potentially have multiple spectral
        # windows to deal with, and so some additional care is required since MS does
        # NOT require data from all windows to be present simultaneously.

        use_row = np.zeros_like(time_arr, dtype=bool)
        data_dict = {}
        for key in data_desc_dict:
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

        unique_blts = sorted(set(zip(time_arr, ant_1_arr, ant_2_arr, strict=True)))

        blt_dict = {}
        for idx, blt_tuple in enumerate(unique_blts):
            blt_dict[blt_tuple] = idx

        nblts = len(unique_blts)

        # Iterate through this list to handle singleton elements (which numpy doesn't
        # like to combine w/ 1-D arrays).
        pol_list = np.unique(
            [item for key in data_dict for item in data_dict[key]["CORR_TYPE"]]
        )
        npols = len(pol_list)

        spw_dict = {
            data_dict[key]["SPW_ID"]: {
                "DATA_DICT_KEY": key,
                "NUM_CHAN": data_dict[key]["NUM_CHAN"],
            }
            for key in data_dict
        }
        spw_list = sorted(spw_dict.keys())

        # Here we sort out where the various spectral windows are starting and stopping.
        # By default, data are sorted in spw-number order.
        nfreqs = 0
        spw_id_array = np.array([], dtype=int)
        for key in sorted(spw_dict.keys()):
            assert len(data_dict) == len(spw_dict), (
                "This is a bug, please make an issue in our issue log."
            )
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
                    strict=True,
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
            for key in data_dict:
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
        for key in data_dict:
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
            for arr, name in zip(arr_tuple, name_tuple, strict=True):
                if not np.allclose(data_dict[key][name][check_mask], arr[check_idx]):
                    if raise_error:
                        raise ValueError(
                            f"Column {name} appears to vary on between windows, "
                            "which is not permitted for UVData objects. To "
                            "bypass this error, you can set raise_error=False, "
                            "which will raise a warning instead and use the "
                            "first recorded value."
                        )
                    elif name not in vary_list:
                        # If not raising an error, then at least warn the user that
                        # discrepant data were detected.
                        warnings.warn(
                            f"Column {name} appears to vary on between windows, "
                            "defaulting to first recorded value."
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
                temp_data[:, :, pol_idx] = np.conj(tb_main_sel.getcol(data_column))
            else:
                temp_data[:, :, pol_idx] = tb_main_sel.getcol(data_column)

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
        self.scan_number_array = scan_number_arr
        self.flex_spw_id_array = spw_id_array

        field_list = np.unique(field_arr).astype(int).tolist()

        return spw_list, field_list, pol_list, flex_pol, x_orientation

    @copy_replace_short_description(UVData.read_ms, style=DocstringStyle.NUMPYDOC)
    def read_ms(
        self,
        filepath,
        *,
        data_column="DATA",
        pol_order="AIPS",
        background_lsts=True,
        default_mount_type="other",
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
        astrometry_library=None,
    ):
        """Read in a CASA measurement set."""
        if not casa_present:
            raise ImportError(no_casa_message) from casa_error

        if not os.path.exists(filepath):
            raise OSError(filepath + " not found")
        # set filename variable
        basename = filepath.rstrip("/")
        self.filename = [os.path.basename(basename)]
        self._filename.form = (1,)

        # get the history info
        self.history, pyuvdata_written = ms_utils.read_ms_history(
            filepath,
            pyuvdata_version_str=self.pyuvdata_version_str,
            check_origin=True,
            raise_err=False,
        )

        data_desc_dict = ms_utils.read_ms_data_description(filepath)

        # Polarization array
        pol_dict = ms_utils.read_ms_polarization(filepath)
        for key in data_desc_dict:
            pol_id = data_desc_dict[key]["POLARIZATION_ID"]
            data_desc_dict[key]["CORR_TYPE"] = pol_dict[pol_id]["corr_type"]
            data_desc_dict[key]["NUM_CORR"] = pol_dict[pol_id]["num_corr"]

        spw_dict = ms_utils.read_ms_spectral_window(filepath)
        single_chan_list = []
        for key in data_desc_dict:
            spw_id = data_desc_dict[key]["SPECTRAL_WINDOW_ID"]
            data_desc_dict[key]["CHAN_FREQ"] = spw_dict["chan_freq"][spw_id]
            # beware! There are possibly 3 columns here that might be the correct one
            # to use: CHAN_WIDTH, EFFECTIVE_BW, RESOLUTION
            data_desc_dict[key]["CHAN_WIDTH"] = spw_dict["chan_width"][spw_id]
            data_desc_dict[key]["NUM_CHAN"] = spw_dict["num_chan"][spw_id]
            data_desc_dict[key]["SPW_ID"] = int(spw_dict["assoc_spw_id"][spw_id])
            if data_desc_dict[key]["NUM_CHAN"] == 1:
                single_chan_list.append(key)

        if ignore_single_chan:
            for key in single_chan_list:
                del data_desc_dict[key]

        spw_list, field_list, pol_list, flex_pol, x_orientation = self._read_ms_main(
            filepath,
            data_column=data_column,
            data_desc_dict=data_desc_dict,
            read_weights=read_weights,
            pyuvdata_written=pyuvdata_written,
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

        for key in data_desc_dict:
            sel_mask = self.flex_spw_id_array == data_desc_dict[key]["SPW_ID"]
            self.freq_array[sel_mask] = data_desc_dict[key]["CHAN_FREQ"]
            self.channel_width[sel_mask] = data_desc_dict[key]["CHAN_WIDTH"]

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

        # open up the observation information
        obs_dict = ms_utils.read_ms_observation(filepath)
        self.telescope.name = obs_dict["telescope_name"]
        self.telescope.instrument = obs_dict["telescope_name"]
        self.extra_keywords["observer"] = obs_dict["observer"]

        # open table with antenna location information
        tb_ant_dict = ms_utils.read_ms_antenna(filepath)
        self.telescope.antenna_numbers = tb_ant_dict["antenna_numbers"]
        self.telescope.antenna_diameters = tb_ant_dict["antenna_diameters"]
        self.telescope.mount_type = tb_ant_dict["antenna_mount"]

        tb_feed_dict = ms_utils.read_ms_feed(
            filepath, select_ants=self.telescope.antenna_numbers
        )
        self.telescope.feed_array = tb_feed_dict["feed_array"]
        self.telescope.feed_angle = tb_feed_dict["feed_angle"]
        self.telescope.Nfeeds = tb_feed_dict["Nfeeds"]

        self.telescope.location = ms_utils.get_ms_telescope_location(
            tb_ant_dict=tb_ant_dict, obs_dict=obs_dict
        )
        full_antenna_positions = tb_ant_dict["antenna_positions"]

        # antenna names
        ant_names = tb_ant_dict["antenna_names"]
        station_names = tb_ant_dict["station_names"]

        # importuvfits measurement sets store antenna names in the STATION column.
        # cotter measurement sets store antenna names in the NAME column, which is
        # inline with the MS definition doc. In that case all the station names are
        # the same. Default to using what the MS definition doc specifies, unless
        # we read importuvfits in the history, or if the antenna column is not filled.
        if ("importuvfits" not in self.history) and (
            len(ant_names) == len(np.unique(ant_names)) and ("" not in ant_names)
        ):
            self.telescope.antenna_names = ant_names
        else:
            self.telescope.antenna_names = station_names

        self.telescope.Nants = len(self.telescope.antenna_names)

        relative_positions = np.zeros_like(full_antenna_positions)
        relative_positions = (
            full_antenna_positions - self.telescope._location.xyz().reshape(1, 3)
        )
        self.telescope.antenna_positions = relative_positions

        # set LST array from times and itrf
        proc = self.set_lsts_from_time_array(
            background=background_lsts, astrometry_library=astrometry_library
        )

        phase_center_catalog, field_id_dict = ms_utils.read_ms_field(
            filepath, return_phase_center_catalog=True
        )

        tb_sou_dict = {}
        # The SOURCE table is optional, so if not found a RuntimeError will be
        # thrown, and we should forgo trying to associate SOURCE table entries with
        # the FIELD table.
        with contextlib.suppress(FileNotFoundError):
            tb_sou_dict = ms_utils.read_ms_source(filepath)

        if len(field_id_dict) != 0:
            # Update the catalog if entries are in the SOURCE table dict
            for key in phase_center_catalog:
                if key in tb_sou_dict:
                    phase_center_catalog[key].update(tb_sou_dict[key])

            self.phase_center_id_array = np.array(
                [field_id_dict[key] for key in self.phase_center_id_array], dtype=int
            )

        self.phase_center_catalog = phase_center_catalog
        self.Nphase = len(phase_center_catalog)

        if proc is not None:
            proc.join()
        # Fill in the apparent coordinates here
        self._set_app_coords_helper()

        # order polarizations
        if pol_order is not None:
            self.reorder_pols(order=pol_order, run_check=False)

        self.set_telescope_params(
            x_orientation=x_orientation,
            mount_type=default_mount_type,
            check_extra=check_extra,
            run_check=run_check,
            run_check_acceptability=run_check_acceptability,
        )

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )
