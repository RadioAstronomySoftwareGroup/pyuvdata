# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing Mir files."""
import numpy as np

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

    def read_mir(self, filepath, isource=None, irec=None, isb=None, corrchunk=None):
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
        isb : int
            Sideband code for MIR dataset
        corrchunk : int
            Correlator chunk code for MIR dataset
        """
        # Use the mir_parser to read in metadata, which can be used to select data.
        mir_data = mir_parser.MirParser(filepath)

        # Select out data that we want to work with.
        if isource is None:
            isource = mir_data.in_read["isource"][0]
        if irec is None:
            irec = mir_data.bl_read["irec"][0]
        if isb is None:
            isb = mir_data.bl_read["isb"][0]
        if corrchunk is None:
            corrchunk = mir_data.sp_read["corrchunk"][0]

        mir_data.use_in = mir_data.in_read["isource"] == isource
        mir_data.use_bl = np.logical_and(
            np.logical_and(
                mir_data.bl_read["isb"] == isb, mir_data.bl_read["ipol"] == 0
            ),
            mir_data.bl_read["irec"] == irec,
        )
        mir_data.use_sp = mir_data.sp_read["corrchunk"] == corrchunk

        # Load up the visibilities into the MirParser object. This will also update the
        # filters, and will make sure we're looking at the right metadata.
        mir_data._update_filter()
        if len(mir_data.in_data) == 0:
            raise IndexError("No valid records matching those selections!")

        mir_data.load_data(load_vis=True, load_raw=True)

        # Create a simple array/list for broadcasting values stored on a
        # per-intergration basis in MIR into the (tasty) per-blt records in UVDATA.
        bl_in_maparr = [mir_data.inhid_dict[idx] for idx in mir_data.bl_data["inhid"]]

        # Derive Nants_data from baselines.
        self.Nants_data = len(
            np.unique(
                np.concatenate((mir_data.bl_data["iant1"], mir_data.bl_data["iant2"]))
            )
        )

        self.Nants_telescope = 8
        self.Nbls = int(self.Nants_data * (self.Nants_data - 1) / 2)
        self.Nblts = len(mir_data.bl_data)
        self.Nfreqs = int(mir_data.sp_data["nch"][0])
        self.Npols = 1  # todo: We will need to go back and expand this.
        self.Nspws = 1  # todo: We will need to go back and expand this.
        self.Ntimes = len(mir_data.in_data)
        self.ant_1_array = mir_data.bl_data["iant1"] - 1
        self.ant_2_array = mir_data.bl_data["iant2"] - 1
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

        fsky = mir_data.sp_data["fsky"][0] * 1e9  # GHz -> Hz
        fres = mir_data.sp_data["fres"][0] * 1e6  # MHz -> Hz
        nch = mir_data.sp_data["nch"][0]

        self.channel_width = fres
        # Need the half-channel offset below because of the weird way
        # in which MIR identifies the "center" of the band
        self.freq_array = fsky + fres * (np.arange(nch) - (nch / 2 - 0.5))

        # TODO: This will need to be fixed when spw > 1
        self.freq_array = np.reshape(self.freq_array, (1, -1))
        self.history = "Raw Data"
        self.instrument = "SWARM"

        # todo: This won't work when we have multiple spectral windows.
        self.integration_time = mir_data.sp_data["integ"]

        # todo: Using MIR V3 convention, will need to be V2 compatible eventually.
        self.lst_array = (
            mir_data.in_data["lst"][bl_in_maparr].astype(float) + (0.0 / 3600.0)
        ) * (np.pi / 12.0)

        # todo: We change between xx yy and rr ll, so we will need to update this.
        self.polarization_array = np.asarray([-5])

        self.spw_array = np.asarray([0])

        self.telescope_name = "SMA"
        time_array_mjd = mir_data.in_read["mjd"][bl_in_maparr]
        self.time_array = time_array_mjd + 2400000.5

        # Need to flip the sign convention here on uvw, since we use a1-a2 versus the
        # standard a2-a1 that uvdata expects
        self.uvw_array = (-1.0) * np.transpose(
            np.vstack(
                (mir_data.bl_data["u"], mir_data.bl_data["v"], mir_data.bl_data["w"])
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
        self.data_array = np.reshape(
            np.array(mir_data.vis_data),
            (self.Nblts, self.Nspws, self.Nfreqs, self.Npols),
        )
        # Don't need the data anymore, so drop it
        mir_data.unload_data()
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
