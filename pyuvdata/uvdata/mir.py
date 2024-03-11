# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing Mir files."""
import os
import warnings

import numpy as np
from astropy.coordinates import angular_separation
from astropy.time import Time
from docstring_parser import DocstringStyle

from .. import get_telescope
from .. import utils as uvutils
from ..docstrings import copy_replace_short_description
from . import mir_parser
from .uvdata import UVData

__all__ = ["generate_sma_antpos_dict", "Mir"]


def generate_sma_antpos_dict(filepath):
    """
    Create a dictionary of antenna positions.

    This is a convenience function for reading in a MIR-styled antennas file, and
    converting that into a dictionary which can elsewhere be used to, for example,
    update baseline positions.

    Parameters
    ----------
    filepath : str
        Path to the file or folder (if the file containing the information is named the
        MIR-typical "antennas") containing the antenna positions.

    Returns
    -------
    ant_dict : dict
        Dictionary of antenna positions, with antenna number used as key, and a
        3-element array with the positions of the telescope (in XYZ coordinates relative
        to array center, as is typical for pyuvdata).
    """
    from .mir_meta_data import MirAntposData

    if not os.path.exists(filepath):
        raise ValueError("No such file or folder exists")

    mir_antpos = MirAntposData()

    if os.path.isfile(filepath):
        # If this is a file, we want to load that in directly rather than looking at
        # the default filename (used when importing solns rather than a full data set)
        mir_antpos._filetype = os.path.basename(filepath)
        filepath = os.path.dirname(filepath)

    # Load in the data
    mir_antpos.read(filepath)

    # We need the antenna positions in ECEF, rather than the native rotECEF format that
    # they are stored in. Get the longitude info, and use the appropriate function in
    # utils to get these values the way that we want them.
    _, lon, _ = get_telescope("SMA")._telescope_location.lat_lon_alt()
    mir_antpos["xyz_pos"] = uvutils.ECEF_from_rotECEF(mir_antpos["xyz_pos"], lon)

    # Create a dictionary that can be used for updates.
    return {item["antenna"]: item["xyz_pos"] for item in mir_antpos}


class Mir(UVData):
    """
    A class for Mir file objects.

    This class defines an Mir-specific subclass of UVData for reading and
    writing Mir files. This class should not be interacted with directly,
    instead use the read_mir and write_mir methods on the UVData class.
    """

    @copy_replace_short_description(UVData.read_mir, style=DocstringStyle.NUMPYDOC)
    def read_mir(
        self,
        filepath,
        antenna_nums=None,
        antenna_names=None,
        bls=None,
        time_range=None,
        lst_range=None,
        polarizations=None,
        catalog_names=None,
        corrchunk=None,
        receivers=None,
        sidebands=None,
        select_where=None,
        apply_tsys=True,
        apply_flags=True,
        apply_dedoppler=False,
        pseudo_cont=False,
        rechunk=None,
        compass_soln=None,
        swarm_only=True,
        codes_check=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        allow_flex_pol=True,
        check_autos=True,
        fix_autos=True,
    ):
        """Read in data from an SMA MIR file, and map to a UVData object."""
        # Use the mir_parser to read in metadata, which can be used to select data.
        # We want to sure that the mir file is v3 compliant, since correctly filling
        # values into a UVData object depends on that.
        mir_data = mir_parser.MirParser(filepath=filepath, compass_soln=compass_soln)

        if codes_check:
            where_list = []
            for code in mir_data.codes_data._mutable_codes:
                if code in mir_data.codes_data.get_code_names():
                    where_list.append((code, "eq", list(mir_data.codes_data[code])))
            mir_data.select(where=where_list)

        if select_where is None:
            select_where = []

            if antenna_nums is not None:
                select_where += [("ant", "eq", antenna_nums)]
            if antenna_names is not None:
                select_where += [("tel1", "eq", antenna_names)]
                select_where += [("tel2", "eq", antenna_names)]
            if bls is not None:
                select_where += [
                    ("blcd", "eq", ["%i-%i" % (tup[0], tup[1]) for tup in bls])
                ]
            if time_range is not None:
                # Have to convert times from UTC JD -> TT MJD for mIR
                select_where += [
                    (
                        "mjd",
                        "between",
                        Time(time_range, format="jd", scale="utc").tt.mjd,
                    )
                ]
            if lst_range is not None:
                select_where += [("lst_range", "between", lst_range)]
            if polarizations is not None:
                select_where += [("pol", "eq", polarizations)]
            if catalog_names is not None:
                select_where += [("source", "eq", catalog_names)]
            if receivers is not None:
                select_where += [("rec", "eq", receivers)]
            if sidebands is not None:
                select_where += [("sb", "eq", sidebands)]

            if corrchunk is not None:
                select_where += [("corrchunk", "eq", corrchunk)]
            elif not pseudo_cont:
                select_where += [("corrchunk", "ne", 0)]
            if swarm_only:
                select_where += [("correlator", "eq", 1)]

        if select_where:
            mir_data.select(where=select_where)

        if rechunk is not None:
            mir_data.rechunk(rechunk)

        self._init_from_mir_parser(
            mir_data,
            allow_flex_pol=allow_flex_pol,
            apply_flags=apply_flags,
            apply_tsys=apply_tsys,
            apply_dedoppler=apply_dedoppler,
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

    def _prep_and_insert_data(
        self,
        mir_data: mir_parser.MirParser,
        sphid_dict,
        spdx_dict,
        blhid_blt_order,
        apply_flags=True,
        apply_tsys=True,
        apply_dedoppler=False,
    ):
        """
        Load and prep data for import into UVData object.

        This is a helper function, not meant for users to call. It is used by the read
        method to fill in the data and flag arrays of a UVData object based on the
        records stored within the MIR dataset.

        Parameters
        ----------
        mir_data : MirParser object
            Object from which to plug data into the data arrays.
        sphid_dict : dict
            Map between MIR spectral record and position in the UVData array, as
            dictated by blt-index, pol-index, and slice corresponding to the range
            in frequencies.
        spdx_dict : dict
            Map between MIR sideband, receiver/polarization, and corrchunk to
            UVData-oriented blt-index, pol-index, and slice corresponding to the range
            in frequencies.
        blhid_blt_order : dict
            Map between MIR blhid to blt-index in the UVData object
        apply_flags : bool
            If set to True, apply "wideband" flags to the visibilities, which are
            recorded by the realtime system to denote when data are expected to be bad
            (e.g., antennas not on source, dewar warm). Default it true.
        apply_tsys : bool
            If set to False, data are returned as correlation coefficients (normalized
            by the auto-correlations). Default is True, which instead scales the raw
            visibilities and forward-gain of the antenna to produce values in Jy
            (uncalibrated).
        apply_dedoppler : bool
            If set to True, data will be corrected for any doppler-tracking performed
            during observations, and brought into the topocentric rest frame (default
            for UVData objects). Default is False.
        """
        if mir_data.vis_data is None:
            mir_data.load_data(load_cross=True, apply_tsys=apply_tsys)
            if apply_flags:
                mir_data.apply_flags()
            if apply_dedoppler:
                mir_data.redoppler_data()

        if not np.all(
            np.isin(list(mir_data.vis_data.keys()), mir_data.sp_data.get_header_keys())
        ):
            raise KeyError(
                "Mismatch between keys in vis_data and sphid in sp_data, which "
                "should not happen. Please file an issue in our GitHub issue log "
                "so that we can fix it."
            )

        # Go through the data record-by-record to fill things in
        for sp_rec, sphid in zip(mir_data.sp_data, mir_data.sp_data.get_header_keys()):
            # sphid_dict is a 3-element tuple, containing the sideband, receiver, and
            # corrchunk for the spectral record (whose header key is sphid)
            window = sphid_dict[sphid]
            vis_rec = mir_data.vis_data[sphid]  # Visibilities + flags
            blt_idx = blhid_blt_order[sp_rec["blhid"]]  # Blt index to load data into
            pol_idx = spdx_dict[window]["pol_idx"]  # Polarization to load data into
            ch_slice = spdx_dict[window]["ch_slice"]  # Channel range to load data into

            # Now populate the fields with the relevant data from the object
            self.data_array[blt_idx, pol_idx, ch_slice] = np.conj(vis_rec["data"])
            self.flag_array[blt_idx, pol_idx, ch_slice] = vis_rec["flags"]
            self.nsample_array[blt_idx, pol_idx, ch_slice] = vis_rec["weights"]

        # Drop the data from the MirParser object once we have it loaded up.
        mir_data.unload_data()

    def _init_from_mir_parser(
        self,
        mir_data: mir_parser.MirParser,
        allow_flex_pol=True,
        apply_tsys=True,
        apply_flags=True,
        apply_dedoppler=False,
    ):
        """
        Convert a MirParser object into a UVData object.

        Parameters
        ----------
        mir_data : MirParser object
            MIR dataset to be converted into a UVData object.
        allow_flex_pol : bool
            If only one polarization per spectral window is read (and the polarization
            differs from window to window), allow for the `UVData` object to use
            "flexible polarization", which compresses the polarization-axis of various
            attributes to be of length 1, sets the `flex_spw_polarization_array`
            attribute to define the polarization per spectral window. Default is True.
        """
        # By default, we will want to assume that MIR datasets are multi-spw.
        # At present, there is no advantage to allowing this
        # not to be true on read-in, particularly as in the long-term, this setting
        # will hopefully become the default for all data sets.
        self._set_flex_spw()

        # Also set future array shapes here
        self._set_future_array_shapes()

        # Create a simple list for broadcasting values stored on a
        # per-integration basis in MIR into the (tasty) per-blt records in UVDATA.
        bl_in_idx = mir_data.in_data._index_query(header_key=mir_data.bl_data["inhid"])

        # Create a simple array/list for broadcasting values stored on a
        # per-blt basis into per-spw records, and per-time into per-blt records
        sp_bl_idx = mir_data.bl_data._index_query(header_key=mir_data.sp_data["blhid"])

        ant1_rxa_mask = mir_data.bl_data.get_value("ant1rx", index=sp_bl_idx) == 0
        ant2_rxa_mask = mir_data.bl_data.get_value("ant2rx", index=sp_bl_idx) == 0

        if len(np.unique(mir_data.bl_data["ipol"])) == 1 and (
            len(mir_data.codes_data["pol"]) == (2 * 4)
        ):
            # If only one pol is found, and the polarization dictionary has only four
            # codes + four unique index codes, then we actually need to verify this is
            # a single pol observation, since the current system has a quirk that it
            # marks both X- and Y-pol receivers as the same polarization.
            pol_arr = np.zeros_like(sp_bl_idx)

            pol_arr[np.logical_and(ant1_rxa_mask, ant2_rxa_mask)] = 0
            pol_arr[np.logical_and(~ant1_rxa_mask, ~ant2_rxa_mask)] = 1
            pol_arr[np.logical_and(ant1_rxa_mask, ~ant2_rxa_mask)] = 2
            pol_arr[np.logical_and(~ant1_rxa_mask, ant2_rxa_mask)] = 3
        else:
            # If this has multiple ipol codes, then we don't need to worry about the
            # single-code ambiguity.
            pol_arr = mir_data.bl_data.get_value("ipol", index=sp_bl_idx)

        # Construct an indexing list, that we'll use later to figure out what data
        # goes where, based on spw, sideband, and pol code.
        spdx_list = [
            (winid, sbid, polid)
            for winid, sbid, polid in zip(
                mir_data.sp_data["corrchunk"],
                mir_data.bl_data.get_value("isb", index=sp_bl_idx),
                pol_arr,
            )
        ]

        # We'll also use this later
        sphid_dict = dict(zip(mir_data.sp_data.get_header_keys(), spdx_list))

        # Create a dict with the ordering of the pols
        pol_dict = {key: idx for idx, key in enumerate(np.unique(pol_arr))}

        pol_split_tuning = False
        if len(pol_dict) == 2:
            # If dual-pol, then we need to check if the tunings are split, because
            # the two polarizations will effectively be concat'd across the freq
            # axis instead of the pol axis. First, see if we have two diff receivers
            rxa_mask = ant1_rxa_mask & ant2_rxa_mask
            rxb_mask = ~(ant1_rxa_mask | ant2_rxa_mask)

            if np.any(rxa_mask) and np.any(rxb_mask):
                # If we have both VV and HH data, check to see that the frequencies of
                # each of the spectral chunks match. If they do, then we can concat
                # across the polarization axis, but if _not_, we should treat this as
                # a pol-split data set.
                fsky_vals = mir_data.sp_data["fsky"]
                chunk_vals = mir_data.sp_data["corrchunk"]
                loa_chunks = set(zip(fsky_vals[rxa_mask], chunk_vals[rxa_mask]))
                lob_chunks = set(zip(fsky_vals[rxb_mask], chunk_vals[rxb_mask]))
                pol_split_tuning = not (
                    loa_chunks.issubset(lob_chunks) or lob_chunks.issubset(loa_chunks)
                )

        # Map MIR pol code to pyuvdata/AIPS polarization number
        pol_code_dict = {}
        icode_dict = mir_data.codes_data["pol"]
        for code in mir_data.codes_data.get_codes("pol", return_dict=False):
            # There are pol modes/codes that are support in MIR that are not in AIPS
            # or CASA, although they are rarely used, so we can skip over translating
            # them in the try/except loop here (if present in the data, it will throw
            # an error further downstream).
            try:
                pol_code_dict[icode_dict[code]] = uvutils.POL_STR2NUM_DICT[code.lower()]
            except KeyError:
                pass
        if pol_split_tuning and allow_flex_pol:
            # If we have a split tuning that, that means we can take advantage of
            # the flex-pol feature within UVData
            Npols = 1
            polarization_array = np.array([0])
        else:
            # Otherwise, calculate dimensions and arrays for the polarization axis
            # like normal.
            Npols = len(set(pol_dict.values()))
            polarization_array = np.zeros(Npols, dtype=int)

            for key in pol_dict.keys():
                polarization_array[pol_dict[key]] = pol_code_dict[key]

        # Create a list of baseline-time combinations in the data
        blt_list = [
            (inhid, ant1, ant2)
            for inhid, ant1, ant2 in zip(*mir_data.bl_data[["inhid", "iant1", "iant2"]])
        ]

        # Use the list above to create a dict that maps baseline-time combo
        # to position along the mouthwatering blt-axis.
        blt_dict = {
            blt_tuple: idx
            for idx, blt_tuple in enumerate(
                sorted(set(blt_list), key=lambda x: (x[0], x[1], x[2]))
            )
        }

        # Match blhid in MIR to blt position in UVData
        blhid_blt_order = {
            key: blt_dict[value]
            for key, value in zip(mir_data.bl_data["blhid"], blt_list)
        }

        # The more blts, the better
        Nblts = len(blt_dict)

        # Here we need to do a little fancy footwork in order to map spectral windows
        # to ranges along the freq-axis, and calculate some values that will eventually
        # populate arrays related to this axis (e.g., freq_array, chan_width).
        spdx_dict = {}
        spw_dict = {}
        for spdx in set(spdx_list):
            # We need to do a some extra handling here, because a single correlator
            # can produce multiple spectral windows (e.g., LSB/USB). The scheme below
            # will negate the corr band number if LSB, will set the corr band number to
            # 255 if the values arise from the pseudo-wideband values, and will add 512
            # if the pols are split-tuned. This scheme, while a little funky, guarantees
            # that each unique freq range has its own spectral window number.
            spw_id = 255 if (spdx[0] == 0) else spdx[0]
            spw_id *= (-1) ** (1 + spdx[1])
            spw_id += 512 if (pol_split_tuning and spdx[2] == 1) else 0

            data_mask = np.array([spdx == item for item in spdx_list])

            # Grab values, get them into appropriate types
            spw_fsky = np.median(mir_data.sp_data["fsky"][data_mask])
            spw_fres = np.median(mir_data.sp_data["fres"][data_mask])
            spw_nchan = np.median(mir_data.sp_data["nch"][data_mask])

            # Make sure that something weird hasn't happened with the metadata (this
            # really should never happen, only one value should exist per window).
            for val, item in zip(
                [spw_fsky, spw_fres, spw_nchan], ["fsky", "fres", "nch"]
            ):
                if not np.allclose(val, mir_data.sp_data[item][data_mask]):
                    warnings.warn(
                        "Discrepancy in %s for win %i sb %i pol %i. Values of "
                        "`freq_array` and `channel_width` should be checked for "
                        "channels corresponding to spw_id %i." % (item, *spdx, spw_id)
                    )

            # Get the data in the right units and dtype
            spw_fsky = float(spw_fsky * 1e9)  # GHz -> Hz
            spw_fres = float(spw_fres * 1e6)  # MHz -> Hz
            spw_nchan = int(spw_nchan)

            # Populate the channel width array
            channel_width = abs(spw_fres) + np.zeros(spw_nchan, dtype=np.float64)

            # Populate the spw_id_array
            spw_id_array = spw_id + np.zeros(spw_nchan, dtype=np.int64)

            # So the freq array here is a little weird, because the current fsky
            # refers to the point between the nch/2 and nch/2 + 1 channel in the
            # raw (unaveraged) spectrum. This was done for the sake of some
            # convenience, at the cost of clarity. In some future format of the
            # data, we expect to be able to drop seemingly random offset here.
            freq_array = (
                spw_fsky
                - (np.sign(spw_fres) * 139648.4375)
                + (spw_fres * (np.arange(spw_nchan) + 0.5 - (spw_nchan / 2)))
            )

            # Note here that if we're using flex-pol, then the polarization axis of
            # the UVData object is 1, hence why pol_idx is forced to be 0 here.
            spdx_dict[spdx] = {
                "spw_id": spw_id,
                "pol_idx": (
                    0 if (pol_split_tuning and allow_flex_pol) else pol_dict[spdx[2]]
                ),
            }

            # Stuff this dictionary full of the per-spw metadata
            spw_dict[spw_id] = {
                "nchan": spw_nchan,
                "freqs": spw_fres,
                "fsky": spw_fsky,
                "channel_width": channel_width,
                "spw_id_array": spw_id_array,
                "freq_array": freq_array,
                "pol_state": pol_code_dict[spdx[2]],
            }

        spw_array = sorted(spw_dict.keys())
        Nspws = len(spw_array)
        Nfreqs = 0
        for key in spw_array:
            spw_dict[key]["ch_slice"] = slice(Nfreqs, Nfreqs + spw_dict[key]["nchan"])
            Nfreqs += spw_dict[key]["nchan"]

        # Initialize some arrays that we'll be appending to
        flex_spw_id_array = np.zeros(Nfreqs, dtype=int)
        channel_width = np.zeros(Nfreqs, dtype=np.float64)
        freq_array = np.zeros(Nfreqs, dtype=np.float64)
        flex_pol = np.zeros(Nspws, dtype=int)
        for idx, key in enumerate(spw_array):
            flex_spw_id_array[spw_dict[key]["ch_slice"]] = spw_dict[key]["spw_id_array"]
            channel_width[spw_dict[key]["ch_slice"]] = spw_dict[key]["channel_width"]
            freq_array[spw_dict[key]["ch_slice"]] = spw_dict[key]["freq_array"]
            flex_pol[idx] = spw_dict[key]["pol_state"]

        if not pol_split_tuning:
            flex_pol = None

        for key in spdx_dict:
            spdx_dict[key]["ch_slice"] = spw_dict[spdx_dict[key]["spw_id"]]["ch_slice"]

        # Now assign our flexible arrays to the object itself
        self.freq_array = freq_array
        self.Nfreqs = Nfreqs
        self.Nspws = Nspws
        self.channel_width = channel_width
        self.flex_spw_id_array = flex_spw_id_array

        # Derive Nants_data from baselines.
        self.Nants_data = len(
            np.unique(
                np.concatenate((mir_data.bl_data["iant1"], mir_data.bl_data["iant2"]))
            )
        )
        self.Nants_telescope = 8
        self.Nbls = int(self.Nants_data * (self.Nants_data - 1) / 2)
        self.Nblts = Nblts
        self.Npols = Npols
        self.Ntimes = len(mir_data.in_data)
        self.antenna_names = ["Ant%i" % idx for idx in range(1, 9)]

        self.antenna_numbers = np.arange(1, 9)

        # Prepare the XYZ coordinates of the antenna positions.
        antXYZ = np.zeros([self.Nants_telescope, 3])
        for idx in range(self.Nants_telescope):
            if (idx + 1) in mir_data.antpos_data["antenna"]:
                antXYZ[idx] = mir_data.antpos_data["xyz_pos"][
                    mir_data.antpos_data["antenna"] == (idx + 1)
                ]

        # Get the coordinates from the entry in telescope.py
        lat, lon, alt = get_telescope("SMA")._telescope_location.lat_lon_alt()
        self.telescope_location_lat_lon_alt = (lat, lon, alt)

        # Calculate antenna positions in ECEF frame. Note that since both
        # coordinate systems are in relative units, no subtraction from
        # telescope geocentric position is required , i.e we are going from
        # relRotECEF -> relECEF
        self.antenna_positions = uvutils.ECEF_from_rotECEF(antXYZ, lon)

        self.history = "Raw Data"
        self.instrument = "SWARM"

        # Before proceeding, we want to check that information that's stored on a
        # per-spectral record basis (sphid) is consistent across a given baseline-time
        # (n.b., blhid changes on sideband and polarization, so multiple blhids
        # correspond to a single baseline-time). We need to do this check here since
        # pyuvdata handles this metadata on a per-baseline-time basis (and there's no
        # good reason it should vary on a per-sphid basis).
        sp_to_blt = ["igq", "ipq", "vradcat"]
        sp_temp_dict = {}
        suppress_warning = False
        for sp_rec in mir_data.sp_data:
            # Evaluate each spectral records metadata individually, and create a simple
            # dict that contains the metadata we want to check.
            temp_dict = {item: sp_rec[item] for item in sp_to_blt}
            try:
                # If we have already captured metadata about this baseline-time, check
                # to make sure that it agrees with the previous entires.
                if sp_temp_dict[blhid_blt_order[sp_rec["blhid"]]] != temp_dict:
                    # If the entry does NOT agree with what was previously given,
                    # warn the user, and update the record (we only want to warn once
                    # to avoid flooding the user with error messages).
                    if not suppress_warning:
                        warnings.warn(
                            "Per-spectral window metadata differ. Defaulting to using "
                            "the last value in the data set."
                        )
                        suppress_warning = True
                    sp_temp_dict[blhid_blt_order[sp_rec["blhid"]]] = temp_dict
            except KeyError:
                # If we get a key error, it means this is the first record to be
                # evaluated for this given baseline-time, so create a new dict
                # entry so that subsequent sphid records can be evaluated.
                sp_temp_dict[blhid_blt_order[sp_rec["blhid"]]] = temp_dict

        # Next step: we want to check that information that's stored on a per-baseline
        # record basis (blhid) is consistent across a given baseline-time (n.b., again,
        # blhid changes on sideband and polarization, so multiple blhids correspond to
        # a single baseline-time). We need to do this check here since pyuvdata handles
        # this metadata on a per-baseline-time basis.
        bl_to_blt = ["u", "v", "w", "iant1", "iant2"]

        # Note that these are entries that vary per-integration record (inhid), but
        # we include them here so that we can easily expand the per-integration data
        # to the per-baseline-time length arrays that UVData expects.
        in_to_blt = ["lst", "mjd", "ara", "adec", "isource", "rinteg"]
        blt_temp_dict = {}
        suppress_warning = False
        for idx, bl_rec in enumerate(mir_data.bl_data):
            # Evaluate each spectral records metadata individually, and create a simple
            # dict that contains the metadata we want to check.
            temp_dict = {item: bl_rec[item] for item in bl_to_blt}
            in_rec = mir_data.in_data._data[bl_in_idx[idx]]

            # Update the dict with the per-inhid data as well.
            temp_dict.update({item: in_rec[item] for item in in_to_blt})

            # Finally, fold in the originally per-sphid data that we checked above.
            temp_dict.update(sp_temp_dict[blhid_blt_order[bl_rec["blhid"]]])
            try:
                # If we have already captured metadata about this baseline-time, check
                # to make sure that it agrees with the previous entires.
                if blt_temp_dict[blhid_blt_order[bl_rec["blhid"]]] != temp_dict:
                    # Again, if we reach this point, only raise a warning one time
                    if not suppress_warning:
                        warnings.warn(
                            "Per-baseline metadata differ. Defaulting to using "
                            "the last value in the data set."
                        )
                        suppress_warning = True
                    blt_temp_dict[blhid_blt_order[bl_rec["blhid"]]] = temp_dict
            except KeyError:
                # If we get a key error, it means this is the first record to be
                # evaluated for this given baseline-time, so create a new dict
                # entry so that subsequent blhid records can be evaluated.
                blt_temp_dict[blhid_blt_order[bl_rec["blhid"]]] = temp_dict

        # Initialize the metadata arrays.
        integration_time = np.zeros(Nblts, dtype=float)
        lst_array = np.zeros(Nblts, dtype=float)
        mjd_array = np.zeros(Nblts, dtype=float)
        ant_1_array = np.zeros(Nblts, dtype=int)
        ant_2_array = np.zeros(Nblts, dtype=int)
        uvw_array = np.zeros((Nblts, 3), dtype=float)
        phase_center_id_array = np.zeros(Nblts, dtype=int)
        app_ra = np.zeros(Nblts, dtype=float)
        app_dec = np.zeros(Nblts, dtype=float)

        # Use the per-blt dict that we constructed above to populate the various
        # metadata arrays that we need for constructing the UVData object.
        for blt_key in blt_temp_dict.keys():
            temp_dict = blt_temp_dict[blt_key]
            integration_time[blt_key] = temp_dict["rinteg"]
            lst_array[blt_key] = temp_dict["lst"] * (np.pi / 12.0)  # Hours -> rad
            mjd_array[blt_key] = temp_dict["mjd"]
            ant_1_array[blt_key] = temp_dict["iant1"]
            ant_2_array[blt_key] = temp_dict["iant2"]
            uvw_array[blt_key] = np.array(
                [temp_dict["u"], temp_dict["v"], temp_dict["w"]]
            )
            phase_center_id_array[blt_key] = temp_dict["isource"]
            app_ra[blt_key] = temp_dict["ara"]
            app_dec[blt_key] = temp_dict["adec"]

        # Finally, assign arrays to attributed
        self.ant_1_array = ant_1_array
        self.ant_2_array = ant_2_array
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array, attempt256=False
        )
        self.time_array = Time(mjd_array, scale="tt", format="mjd").utc.jd
        self.integration_time = integration_time

        # There is a minor issue w/ MIR datasets where the LSTs are calculated via
        # a polled average rather than calculated for the mid-point, which results
        # in some imprecision (and some nuisance warnings).  Fix this by calculating the
        # LSTs here, checking to make sure that they agree to within the expected
        # precision (sampling rate is 20 Hz, and the max error to worry about is half a
        # sample: 25 ms, or in radians, 2*pi/(40 * 86400)) = pi / 1728000.
        # TODO: Re-evaluate this if/when MIR data writer stops calculating LST this way
        self.set_lsts_from_time_array()
        if not np.allclose(lst_array, self.lst_array, rtol=0, atol=np.pi / 1728000.0):
            # If this check fails, it means that there's something off w/ the lst values
            # (to a larger degree than expected), and we'll pass them back to the user,
            # who can inspect them directly and decide what to do.
            warnings.warn(
                "> 25 ms errors detected reading in LST values from MIR data. "
                "This typically signifies a minor metadata recording error (which can "
                "be mitigated by calling the `set_lsts_from_time_array` method with "
                "`update_vis=False`), though additional errors about uvw-position "
                "accuracy may signal more significant issues with metadata accuracy "
                "that could have substantial impact on downstream analysis."
            )
            self.lst_array = lst_array

        self.polarization_array = polarization_array
        self.flex_spw_polarization_array = flex_pol
        self.spw_array = np.array(spw_array, dtype=int)
        self.telescope_name = "SMA"

        # Need to flip the sign convention here on uvw, since we use a1-a2 versus the
        # standard a2-a1 that uvdata expects
        self.uvw_array = (-1.0) * uvw_array

        self.vis_units = "Jy"

        isource = np.unique(mir_data.in_data["isource"])
        for sou_id in isource:
            source_mask = mir_data.in_data["isource"] == sou_id
            source_ra = mir_data.in_data["rar"][source_mask].astype(float)
            source_dec = mir_data.in_data["decr"][source_mask].astype(float)
            source_epoch = np.mean(mir_data.in_data["epoch"][source_mask]).astype(float)
            source_name = mir_data.codes_data["source"][sou_id]
            if source_epoch != 2000.0:
                # When fed a non-J2000 coordinate, we want to convert that so that it
                # can easily be written into CASA MS format. In this case, we take the
                # median apparent position and translate that to ICRS
                time_arr = Time(
                    mir_data.in_data["mjd"][source_mask], scale="tt", format="mjd"
                ).utc.jd
                source_ra, source_dec = uvutils.transform_app_to_icrs(
                    time_arr,
                    mir_data.in_data["ara"][source_mask],
                    mir_data.in_data["adec"][source_mask],
                    self.telescope_location_lat_lon_alt,
                )
            self._add_phase_center(
                source_name,
                cat_type="sidereal",
                cat_lon=np.median(source_ra),
                cat_lat=np.median(source_dec),
                cat_epoch=None if (source_epoch != 2000.0) else source_epoch,
                cat_frame="icrs",
                info_source="file",
                cat_id=int(sou_id),
            )

            # See if the ra/dec positions change by more than an arcminute, and if so,
            # raise a warning to the user since this isn't common.
            dist_check = angular_separation(
                source_ra[0], source_dec[0], source_ra, source_dec
            )

            if any(dist_check > ((np.pi / 180.0) / 60)):
                warnings.warn(
                    "Position for %s changes by more than an arcminute." % source_name
                )

        # Regenerate the sou_id_array thats native to MIR into a zero-indexed per-blt
        # entry for UVData, then grab ra/dec/position data.
        self.phase_center_id_array = phase_center_id_array
        for val in np.unique(self.phase_center_id_array):
            assert val in self.phase_center_catalog.keys()

        # Fill in the apparent coord calculations
        self.phase_center_app_ra = app_ra
        self.phase_center_app_dec = app_dec

        # For MIR, uvw coords are always calculated in the "apparent" position. We can
        # adjust this by calculating the position angle with our preferred coordinate
        # frame (ICRS) and applying the rotation below (via `calc_uvw`).
        self._set_app_coords_helper(pa_only=True)

        self.uvw_array = uvutils.calc_uvw(
            uvw_array=self.uvw_array,
            old_frame_pa=0.0,
            frame_pa=self.phase_center_frame_pa,
            use_ant_pos=False,
        )

        self.antenna_diameters = np.zeros(self.Nants_telescope) + 6.0
        self.blt_order = ("time", "baseline")

        # set filename attribute
        basename = mir_data.filepath.rstrip("/")
        self.filename = [os.path.basename(basename)]
        self._filename.form = (1,)

        # Finally, start the heavy lifting of loading the full data. We start this by
        # creating arrays to plug visibilities and flags into. The array is shaped this
        # way since when reading in a MIR file, we scan through the blt-axis the
        # slowest and the freq-axis the fastest (i.e., the data is roughly ordered by
        # blt, pol, freq).
        self.data_array = np.zeros((Nblts, Npols, Nfreqs), dtype=np.complex64)
        self.flag_array = np.ones((Nblts, Npols, Nfreqs), dtype=bool)
        self.nsample_array = np.zeros((Nblts, Npols, Nfreqs), dtype=np.float32)

        # Get a list of the current inhid values for later
        inhid_list = mir_data.in_data["inhid"].copy()

        # Store a backup of the selection masks
        mir_data.save_mask("pre-select")

        # If no data is loaded, we want to load subsets of data to start rather than
        # the whole block in one go, since this will save us 2x in memory.
        inhid_step = len(inhid_list)
        if (mir_data.vis_data is None) and (mir_data.auto_data is None):
            inhid_step = (inhid_step // 8) + 1

        for start in range(0, len(inhid_list), inhid_step):
            # If no data is loaded, load up a quarter of the data at a time. This
            # keeps the "extra" memory usage below that for the nsample_array attr,
            # which is generated and filled _after_ this step (thus no extra memory
            # should be required)
            if (mir_data.vis_data is None) and (mir_data.auto_data is None):
                # Note that the masks are combined via and operation.
                mir_data.select(
                    where=("inhid", "eq", inhid_list[start : start + inhid_step])
                )

            # Call this convenience function in case we need to run the data filling
            # multiple times (if we are loading up subsets of data)
            self._prep_and_insert_data(
                mir_data,
                sphid_dict,
                spdx_dict,
                blhid_blt_order,
                apply_flags=apply_flags,
                apply_tsys=apply_tsys,
                apply_dedoppler=apply_dedoppler,
            )

            # Because the select operation messes with the masks, we want to restore
            # those in case we mucked with them earlier (so that subsequent selects
            # behave as expected).
            mir_data.restore_mask("pre-select")

        # We call transpose here since vis_data above is shape (Nblts, Npols, Nfreqs),
        # and we need to get it to (Nblts,Nfreqs, Npols) to match what UVData expects.
        self.data_array = np.transpose(self.data_array, (0, 2, 1))
        self.flag_array = np.transpose(self.flag_array, (0, 2, 1))
        self.nsample_array = np.transpose(self.nsample_array, (0, 2, 1))

    def write_mir(self, filename):
        """
        Write out the SMA MIR files.

        Parameters
        ----------
        filename : str
            The path to the folder on disk to write data to.
        """
        raise NotImplementedError
