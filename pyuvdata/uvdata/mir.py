# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing Mir files."""
import numpy as np
import copy

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
    ):
        """
        Read in data from an SMA MIR file, and map to the UVData model.

        Note that with the exception of filename, the rest of the parameters are
        used to sub-select a range of data that matches the limitations of the current
        instantiation of pyuvdata  -- namely 1 spectral window, 1 source. These could
        be dropped in the future, as pyuvdata capabilities grow.

        Parameters
        ----------
        filepath : str
            The file path to the MIR folder to read from.
        isource : int
            Source code for MIR dataset
        irec : int
            Receiver code for MIR dataset
        isb : list of int
            Sideband codes for MIR dataset (0 = LSB, 1 = USB). Default is both sb.
        corrchunk : list of int
            Correlator chunk codes for MIR dataset
        pseudo_cont : boolean
            Read in pseudo-continuuum values. Default is false.
        """
        # Use the mir_parser to read in metadata, which can be used to select data.
        mir_data = mir_parser.MirParser(filepath)

        # Select out data that we want to work with.
        if isource is None:
            isource = mir_data.in_read["isource"][0]
        if irec is None:
            irec = mir_data.bl_read["irec"][0]

        isb_full_list = np.unique(mir_data.bl_read["isb"]).tolist()
        if isb is None:
            isb = isb_full_list.copy()

        isb_dict = {key: key in isb for key in isb_full_list}
        if len(isb) == 0:
            raise IndexError("No valid sidebands selected!")
        elif not (set(isb).issubset(set(isb_full_list))):
            raise IndexError("isb values contain invalid entries")

        dsb_spws = True if len(isb) == 2 else False

        corrchunk_full_list = np.unique(mir_data.sp_read["corrchunk"]).tolist()
        if corrchunk is None:
            if pseudo_cont:
                corrchunk = [0]
            else:
                corrchunk = copy.deepcopy(corrchunk_full_list)
                corrchunk.remove(0) if 0 in corrchunk else None

        if type(corrchunk) is not list:
            if type(corrchunk) is int:
                corrchunk = [corrchunk]
            else:
                raise IndexError("corrchunk must be int or list of ints!")

        if (0 in corrchunk) and ([0] != corrchunk):
            raise NotImplementedError(
                "cannot load pseudo-cont and spectral windows together!"
            )

        corrchunk_dict = {key: key in corrchunk for key in corrchunk_full_list}

        mir_data.use_in = mir_data.in_read["isource"] == isource
        mir_data.use_bl = np.logical_and(
            np.logical_and(
                mir_data.bl_read["irec"] == irec, mir_data.bl_read["ipol"] == 0
            ),
            np.array([isb_dict[key] for key in mir_data.bl_read["isb"]]),
        )

        mir_data.use_sp = np.array(
            [corrchunk_dict[key] for key in mir_data.sp_read["corrchunk"]]
        )

        # Different sidebands in MIR are (strangely enough) recorded as being
        # different baseline records. To be compatible w/ UVData, we just splice
        # the sidebands together.
        corrchunk_sb = [idx for jdx in sorted(isb) for idx in [jdx] * len(corrchunk)]
        corrchunk *= 1 + dsb_spws

        # Load up the visibilities into the MirParser object. This will also update the
        # filters, and will make sure we're looking at the right metadata.
        mir_data._update_filter()
        if len(mir_data.in_data) == 0:
            raise IndexError("No valid records matching those selections!")

        mir_data.load_data(load_vis=True, load_raw=True)

        # Create a simple list for broadcasting values stored on a
        # per-intergration basis in MIR into the (tasty) per-blt records in UVDATA.
        bl_in_maparr = [
            mir_data.inhid_dict[idx]
            for idx in mir_data.bl_data["inhid"][mir_data.bl_data["isb"] == isb[0]]
        ]

        # Create a simple array/list for broadcasting values stored on a
        # per-blt basis into per-spw records.
        sp_bl_maparr = [mir_data.blhid_dict[idx] for idx in mir_data.sp_data["blhid"]]

        # Derive Nants_data from baselines.
        self.Nants_data = len(
            np.unique(
                np.concatenate((mir_data.bl_data["iant1"], mir_data.bl_data["iant2"]))
            )
        )

        self.Nants_telescope = 8
        self.Nbls = int(self.Nants_data * (self.Nants_data - 1) / 2)
        self.Nblts = len(mir_data.bl_data) // (1 + dsb_spws)
        self.Npols = 1  # todo: We will need to go back and expand this.
        self.Nspws = len(corrchunk)
        self.Ntimes = len(mir_data.in_data)
        self.ant_1_array = mir_data.bl_data["iant1"][:: 1 + dsb_spws] - 1
        self.ant_2_array = mir_data.bl_data["iant2"][:: 1 + dsb_spws] - 1
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
        self.antenna_numbers = np.arange(8)

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
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array, attempt256=False
        )

        if len(np.unique(mir_data.sp_data["nch"])) != 1:
            raise NotImplementedError(
                "Cannot handle spectral windows of different sizes (yet)..."
            )
        self.Nfreqs = int(mir_data.sp_data["nch"][0])

        if len(np.unique(np.abs(mir_data.sp_data["fres"]))) != 1:
            raise NotImplementedError(
                "Cannot handle spectral windows with different channel sizes (yet)..."
            )
        self.channel_width = float(abs(mir_data.sp_data["fres"][0])) * 1e6  # MHz->Hz

        self.history = "Raw Data"
        self.instrument = "SWARM"

        self.integration_time = mir_data.sp_data["integ"][
            mir_data.sp_data["corrchunk"] == corrchunk[0]
        ]

        # If using DSB data, then keep in mind that you'll have two records per
        # Nblt with int time array -- split-filter this
        if dsb_spws:
            self.integration_time = self.integration_time[::2]

        # TODO: Using MIR V3 convention, will need to be V2 compatible eventually.
        self.lst_array = (
            mir_data.in_data["lst"][bl_in_maparr].astype(float) + (0.0 / 3600.0)
        ) * (np.pi / 12.0)

        # TODO: We change between xx yy and rr ll, so we will need to update this.
        self.polarization_array = np.asarray([-5])

        self.spw_array = np.arange(len(corrchunk))

        self.telescope_name = "SMA"
        time_array_mjd = mir_data.in_read["mjd"][bl_in_maparr]
        self.time_array = time_array_mjd + 2400000.5

        # Need to flip the sign convention here on uvw, since we use a1-a2 versus the
        # standard a2-a1 that uvdata expects
        self.uvw_array = (-1.0) * np.transpose(
            np.vstack(
                (
                    mir_data.bl_data["u"][:: 1 + dsb_spws],
                    mir_data.bl_data["v"][:: 1 + dsb_spws],
                    mir_data.bl_data["w"][:: 1 + dsb_spws],
                )
            )
        )

        # todo: Raw data is in correlation coefficients, we may want to convert to Jy.
        self.vis_units = "uncalib"

        self._set_phased()

        sou_list = mir_data.codes_data[mir_data.codes_data["v_name"] == b"source"]

        self.object_name = sou_list[sou_list["icode"] == isource]["code"][0].decode(
            "utf-8"
        )

        self.phase_center_ra = mir_data.in_data["rar"][0]
        self.phase_center_dec = mir_data.in_data["decr"][0]
        self.phase_center_epoch = mir_data.in_data["epoch"][0]

        self.phase_center_epoch = float(self.phase_center_epoch)
        self.antenna_diameters = np.zeros(self.Nants_telescope) + 6
        self.blt_order = ("time", "baseline")

        data_array = np.reshape(
            np.array(mir_data.vis_data),
            (self.Nblts, self.Nspws, self.Nfreqs, self.Npols),
        )
        # Don't need the data anymore, so drop it
        mir_data.unload_data()

        freq_array = np.empty((self.Nspws, self.Nfreqs))
        for idx in range(len(corrchunk)):
            data_mask = np.logical_and(
                mir_data.sp_data["corrchunk"] == corrchunk[idx],
                mir_data.bl_data["isb"][sp_bl_maparr] == corrchunk_sb[idx],
            )

            spw_fsky = np.unique(mir_data.sp_data["fsky"][data_mask])
            spw_fres = np.unique(mir_data.sp_data["fres"][data_mask])
            if (len(spw_fsky) != 1) or (len(spw_fres) != 1):
                raise ValueError(
                    "Spectral window must have the same fsky and fres for whole obs!"
                )
            spw_fsky *= 1e9  # GHz -> Hz
            spw_fres *= 1e6  # MHz -> Hz
            # So the freq array here is a little weird, because the current fsky
            # refers to the point between the nch/2 and nch/2 + 1 channel in the
            # raw (unaveraged) spectrum. This was done for the sake of some
            # convenience, at the cost of clarity. In some future format of the
            # data, we expect to be able to drop seemingly random offset here.
            freq_array[idx] = (
                spw_fsky
                - (np.sign(spw_fres) * 139648437.5)
                + (spw_fres * (np.arange(self.Nfreqs) + 0.5 - (self.Nfreqs / 2)))
            )
            if spw_fres < 0:
                freq_array[idx] = np.flip(freq_array[idx])
                data_array[:, idx, :, :] = np.flip(data_array[:, idx, :, :], axis=1)

        self.freq_array = freq_array
        self.data_array = data_array
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
