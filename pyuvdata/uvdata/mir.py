# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing Mir files."""
import os
import numpy as np
from astropy.time import Time
import warnings

from .uvdata import UVData
from . import mir_parser
from .. import utils as uvutils
from .. import get_telescope

__all__ = ["Mir"]


class Mir(UVData):
    """
    A class for Mir file objects.

    This class defines an Mir-specific subclass of UVData for reading and
    writing Mir files. This class should not be interacted with directly,
    instead use the read_mir and write_mir methods on the UVData class.
    """

    def read_mir(
        self,
        filepath,
        isource=None,
        irec=None,
        isb=None,
        corrchunk=None,
        pseudo_cont=False,
        flex_spw=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Read in data from an SMA MIR file, and map to the UVData model.

        Note that with the exception of filename, the rest of the parameters are
        used to sub-select a range of data that matches the limitations of the current
        instantiation of pyuvdata  -- namely 1 source. This could be dropped in the
        future, as pyuvdata capabilities grow.

        Parameters
        ----------
        filepath : str
            The file path to the MIR folder to read from.
        isource : list of int
            Source code(s) for MIR dataset
        irec : array-like of int
            Receiver code for MIR dataset
        isb : array-like of int
            Sideband codes for MIR dataset (0 = LSB, 1 = USB). Default is both sb.
        corrchunk : array-like of int
            Correlator chunk codes for MIR dataset
        pseudo_cont : boolean
            Read in only pseudo-continuuum values. Default is false.
        flex_spw : boolean
            Allow for support of multiple spectral windows. Default is true.
        """
        # Use the mir_parser to read in metadata, which can be used to select data.
        mir_data = mir_parser.MirParser(filepath)

        if isource is not None:
            mir_data.use_in *= np.isin(mir_data.in_read["isource"], isource)
            if not np.any(mir_data.use_in):
                raise ValueError("No valid sources selected!")

        if irec is not None:
            mir_data.use_bl *= np.isin(mir_data.bl_read["irec"], irec)
            if not np.any(mir_data.use_bl):
                raise ValueError("No valid receivers selected!")

        if isb is not None:
            mir_data.use_bl *= np.isin(mir_data.bl_read["isb"], isb)
            if not np.any(mir_data.use_bl):
                raise ValueError("No valid sidebands selected!")

        if corrchunk is not None:
            mir_data.use_sp *= np.isin(mir_data.sp_read["corrchunk"], corrchunk)
            if not np.any(mir_data.use_sp):
                raise ValueError("No valid spectral bands selected!")
        elif not pseudo_cont:
            mir_data.use_sp *= mir_data.sp_read["corrchunk"] != 0

        mir_data._update_filter()

        self._init_from_mir_parser(mir_data)

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
            )

    def _init_from_mir_parser(self, mir_data):
        """
        Convert a MirParser object into a UVData object.

        Parameters
        ----------
        mir_data : MirParser object
            MIR dataset to be converted into a UVData object.
        """
        # By default, we will want to assume that MIR datasets are phased, multi-spw,
        # and multi phase center. At present, there is no advantage to allowing these
        # not to be true on read-in, particularly as in the long-term, these settings
        # will hopefully become the default for all data sets.
        self._set_phased()
        self._set_multi_phase_center()
        self._set_flex_spw()

        # Create a simple list for broadcasting values stored on a
        # per-intergration basis in MIR into the (tasty) per-blt records in UVDATA.
        bl_in_maparr = np.array(
            [mir_data.inhid_dict[idx] for idx in mir_data.bl_data["inhid"]]
        )

        # Create a simple array/list for broadcasting values stored on a
        # per-blt basis into per-spw records, and per-time into per-blt records
        sp_bl_maparr = np.array(
            [mir_data.blhid_dict[idx] for idx in mir_data.sp_data["blhid"]]
        )

        pol_split_tuning = False
        if len(np.unique(mir_data.bl_data["ipol"])) == 4:
            # This is a full-polarization observation, and so we can take a few
            # things for granted here about the spectral tunings
            spdx_list = [
                (winid, sbid, polid)
                for winid, sbid, polid in zip(
                    mir_data.sp_data["corrchunk"],
                    mir_data.bl_data["isb"][sp_bl_maparr],
                    mir_data.bl_data["ipol"][sp_bl_maparr],
                )
            ]
            pol_dict = {
                key: idx for idx, key in enumerate(np.unique(mir_data.bl_data["ipol"]))
            }
        else:
            spdx_list = [
                (winid, sbid, polid)
                for winid, sbid, polid in zip(
                    mir_data.sp_data["corrchunk"],
                    mir_data.bl_data["isb"][sp_bl_maparr],
                    mir_data.bl_data["ant1rx"][sp_bl_maparr],
                )
            ]

            # If not full-pol, then we need to check if the tunings are spilt, because
            # the two polarizations will effectively be concat'd across the freq
            # axis instead of the pol axis.
            sel_mask = mir_data.bl_data["ant1rx"][sp_bl_maparr] == 0

            # The data all belong to one receiver
            if np.all(sel_mask) or np.all(sel_mask):
                pol_dict = {int(np.all(~sel_mask)): 0}
            else:
                loa_freq = np.median(mir_data.sp_data["gunnLO"][sel_mask])
                lob_freq = np.median(mir_data.sp_data["gunnLO"][~sel_mask])
                pol_split_tuning = not np.isclose(loa_freq, lob_freq)
                pol_dict = {0: 0, 1: int(not pol_split_tuning)}

        pol_code_dict = {}
        for code in mir_data.codes_read[mir_data.codes_read["v_name"] == b"pol"]:
            pol_code_dict[code["icode"]] = code["code"].decode("UTF-8").lower()

        Npols = len(set(pol_dict.values()))
        polarization_array = np.zeros(Npols, dtype=int)

        for key in pol_dict.keys():
            polarization_array[pol_dict[key]] = uvutils.POL_STR2NUM_DICT[
                pol_code_dict[key]
            ]

        blt_list = [
            (intid, ant1, ant2)
            for intid, ant1, ant2 in zip(
                mir_data.bl_data["inhid"],
                mir_data.bl_data["iant1"],
                mir_data.bl_data["iant2"],
            )
        ]

        blt_dict = {
            blt_tuple: idx
            for idx, blt_tuple in enumerate(
                sorted(set(blt_list), key=lambda x: (x[0], x[1], x[2]))
            )
        }

        blhid_blt_order = {
            key: blt_dict[value]
            for key, value in zip(mir_data.bl_data["blhid"], blt_list)
        }

        # By Grabthar's Hammer, what a savings!
        Nblts = len(blt_dict)

        spdx_dict = {}
        spw_dict = {}
        for spdx in set(spdx_list):
            data_mask = np.array([spdx == item for item in spdx_list])

            # Grab values, get them into appropriate types
            spw_fsky = np.unique(mir_data.sp_data["fsky"][data_mask])
            spw_fres = np.unique(mir_data.sp_data["fres"][data_mask])
            spw_nchan = np.unique(mir_data.sp_data["nch"][data_mask])

            # Make sure that something weird hasn't happend with the metadata (this
            # really should never happen)
            assert len(spw_fsky) == 1
            assert len(spw_fres) == 1
            assert len(spw_nchan) == 1

            #  Get the data in the right units and dtype
            spw_fsky = float(spw_fsky * 1e9)  # GHz -> Hz
            spw_fres = float(spw_fres * 1e6)  # MHz -> Hz
            spw_nchan = int(spw_nchan)

            spw_id = 255 if (spdx[0] == 0) else spdx[0]
            spw_id *= (-1) ** (1 + spdx[1])
            spw_id += 512 if pol_split_tuning else 0

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

            spdx_dict[spdx] = {
                "spw_id": spw_id,
                "pol_idx": pol_dict[spdx[2]],
            }

            spw_dict[spw_id] = {
                "nchan": spw_nchan,
                "freqs": spw_fres,
                "fsky": spw_fsky,
                "channel_width": channel_width,
                "spw_id_array": spw_id_array,
                "freq_array": freq_array,
                "pol_state": spdx[2],
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
            flex_pol[idx] = spw_dict[key]["pol_state"] if pol_split_tuning else 0.0

        if pol_split_tuning:
            flex_pol = None

        for key in spdx_dict:
            spdx_dict[key]["ch_slice"] = spw_dict[spdx_dict[key]["spw_id"]]["ch_slice"]

        # Load up the visibilities into the MirParser object.
        vis_data = np.zeros((Nblts, Npols, Nfreqs), dtype=np.complex64)
        mir_data.load_data(load_vis=True)
        mir_data._apply_tsys()

        for sp_rec, window, vis_rec in zip(
            mir_data.sp_data, spdx_list, mir_data.vis_data
        ):
            blt_idx = blhid_blt_order[sp_rec["blhid"]]
            spdx = spdx_dict[window]
            vis_data[(blt_idx, spdx["pol_idx"], spdx["ch_slice"])] = vis_rec

        # Drop the data from the MirParser object once we have it loaded up.
        mir_data.unload_data()

        # Now assign our flexible arrays to the object itself
        # TODO: Spw axis to be collapsed in future release
        self.freq_array = freq_array[np.newaxis, :]
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
        self.antenna_names = [
            "Ant 1",
            "Ant 2",
            "Ant 3",
            "Ant 4",
            "Ant 5",
            "Ant 6",
            "Ant 7",
            "Ant 8",
        ]
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

        # Calculate antenna postions in EFEF frame. Note that since both
        # coordinate systems are in relative units, no subtraction from
        # telescope geocentric position is required , i.e we are going from
        # relRotECEF -> relECEF
        self.antenna_positions = uvutils.ECEF_from_rotECEF(antXYZ, lon)

        self.history = "Raw Data"
        self.instrument = "SWARM"

        # Let's try to chalk up all the per-sp stuff here, eh?
        sp_to_blt = ["igq", "ipq", "flags", "vradcat"]
        sp_temp_dict = {}
        suppress_warning = False
        for sp_rec in mir_data.sp_data:
            temp_dict = {item: sp_rec[item] for item in sp_to_blt}
            try:
                # We only want to throw this warning once -- no sense in flooding the
                # user with errors that can't do much about.
                if not suppress_warning:
                    if sp_temp_dict[blhid_blt_order[sp_rec["blhid"]]] != temp_dict:
                        warnings.warn(
                            "Per-spectral window metadata differ. Defaulting to using "
                            "the last value in the data set."
                        )
            except KeyError:
                sp_temp_dict[blhid_blt_order[sp_rec["blhid"]]] = temp_dict

        # Let's try to chalk up all the per-sp stuff here, eh?
        bl_to_blt = ["u", "v", "w", "iant1", "iant2"]
        in_to_blt = ["lst", "mjd", "ara", "adec", "isource", "rinteg"]

        blt_temp_dict = {}
        suppress_warning = False
        for idx, bl_rec in enumerate(mir_data.bl_data):
            temp_dict = {item: bl_rec[item] for item in bl_to_blt}
            in_rec = mir_data.in_data[bl_in_maparr[idx]]
            temp_dict.update({item: in_rec[item] for item in in_to_blt})
            temp_dict.update(sp_temp_dict[blhid_blt_order[bl_rec["blhid"]]])
            try:
                if not suppress_warning:
                    if blt_temp_dict[blhid_blt_order[bl_rec["blhid"]]] != temp_dict:
                        warnings.warn(
                            "Per-baseline metadata differ. Defaulting to using "
                            "the last value in the data set."
                        )
            except KeyError:
                blt_temp_dict[blhid_blt_order[bl_rec["blhid"]]] = temp_dict

        integration_time = np.zeros(Nblts, dtype=float)
        lst_array = np.zeros(Nblts, dtype=float)
        mjd_array = np.zeros(Nblts, dtype=float)
        ant_1_array = np.zeros(Nblts, dtype=int)
        ant_2_array = np.zeros(Nblts, dtype=int)
        uvw_array = np.zeros((Nblts, 3), dtype=float)
        phase_center_id_array = np.zeros(Nblts, dtype=int)
        app_ra = np.zeros(Nblts, dtype=float)
        app_dec = np.zeros(Nblts, dtype=float)

        for blt_key in blt_temp_dict.keys():
            temp_dict = blt_temp_dict[blt_key]
            integration_time[blt_key] = temp_dict["rinteg"]
            lst_array[blt_key] = temp_dict["lst"] * (np.pi / 12.0)
            mjd_array[blt_key] = temp_dict["mjd"]
            ant_1_array[blt_key] = temp_dict["iant1"]
            ant_2_array[blt_key] = temp_dict["iant2"]
            uvw_array[blt_key] = np.array(
                [temp_dict["u"], temp_dict["v"], temp_dict["w"]]
            )
            phase_center_id_array[blt_key] = temp_dict["isource"]
            app_ra[blt_key] = temp_dict["ara"]
            app_dec[blt_key] = temp_dict["adec"]

        self.ant_1_array = ant_1_array
        self.ant_2_array = ant_2_array
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array, attempt256=False
        )

        # We can just skip an appropriate number of records
        self.integration_time = integration_time

        # TODO: Using MIR V3 convention, will need to be V2 compatible eventually.
        self.lst_array = lst_array
        self.time_array = Time(mjd_array, scale="tt", format="mjd").utc.jd

        self.polarization_array = polarization_array
        self.spw_array = np.array(spw_array, dtype=int)
        self.telescope_name = "SMA"

        # Need to flip the sign convention here on uvw, since we use a1-a2 versus the
        # standard a2-a1 that uvdata expects
        self.uvw_array = (-1.0) * uvw_array

        self.vis_units = "Jy"

        sou_list = mir_data.codes_data[mir_data.codes_data["v_name"] == b"source"]
        isource = np.unique(mir_data.in_data["isource"])

        name_list = [
            sou_list[sou_list["icode"] == idx]["code"][0].decode("utf-8")
            for idx in isource
        ]

        for idx, sou_id in enumerate(isource):
            source_mask = mir_data.in_data["isource"] == sou_id
            source_ra = np.mean(mir_data.in_data["rar"][source_mask]).astype(float)
            source_dec = np.mean(mir_data.in_data["decr"][source_mask]).astype(float)
            source_epoch = np.mean(mir_data.in_data["epoch"][source_mask]).astype(float)
            self._add_phase_center(
                name_list[idx],
                cat_type="sidereal",
                cat_lon=source_ra,
                cat_lat=source_dec,
                cat_epoch=source_epoch,
                cat_frame="fk5",
                info_source="file",
                cat_id=int(sou_id),
            )

        # Regenerate the sou_id_array thats native to MIR into a zero-indexed per-blt
        # entry for UVData, then grab ra/dec/position data.
        self.phase_center_id_array = phase_center_id_array

        self.phase_center_ra = 0.0  # This is ignored w/ mutli-phase-ctr data sets
        self.phase_center_dec = 0.0  # This is ignored w/ mutli-phase-ctr data sets
        self.phase_center_epoch = 2000.0  # This is ignored w/ mutli-phase-ctr data sets
        self.phase_center_frame = "icrs"

        # Fill in the apparent coord calculations
        self.phase_center_app_ra = app_ra
        self.phase_center_app_dec = app_dec

        # For MIR, uvws are always calculated in the "apparent" position. We can adjust
        # this by calculating the position angle with our preferred coordinate frame
        # (ICRS) and applying the rotation below (via `calc_uvw`).
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

        self.data_array = np.transpose(vis_data, (0, 2, 1))[:, np.newaxis, :, :]
        self.flag_array = np.zeros(self.data_array.shape, dtype=bool)
        self.nsample_array = np.ones(self.data_array.shape, dtype=np.float32)

    def write_mir(self, filename):
        """
        Write out the SMA MIR files.

        Parameters
        ----------
        filename : str
            The path to the folder on disk to write data to.
        """
        raise NotImplementedError
