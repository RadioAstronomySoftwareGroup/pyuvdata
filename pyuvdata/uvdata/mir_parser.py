# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Module for low-level interface to MIR files.

This module extracts data types associated with MIR files.

"""
import numpy as np
import os
import copy

__all__ = ["MirParser"]

# MIR structure definitions.
in_dtype = np.dtype(
    [
        ("traid", np.int32),
        ("inhid", np.int32),
        ("ints", np.int32),
        ("az", np.float32),
        ("el", np.float32),
        ("ha", np.float32),
        ("iut", np.int16),
        ("iref_time", np.int16),
        ("dhrs", np.float64),
        ("vc", np.float32),
        ("sx", np.float64),
        ("sy", np.float64),
        ("sz", np.float64),
        ("rinteg", np.float32),
        ("proid", np.int32),
        ("souid", np.int32),
        ("isource", np.int16),
        ("ivrad", np.int16),
        ("offx", np.float32),
        ("offy", np.float32),
        ("ira", np.int16),
        ("idec", np.int16),
        ("rar", np.float64),
        ("decr", np.float64),
        ("epoch", np.float32),
        ("size", np.float32),
        ("vrra", np.float32),
        ("vrdec", np.float32),
        ("lst", np.float32),
        ("iproject", np.int16),
        ("tile", np.int16),
        ("obsmode", np.uint8),
        ("obsflag", np.uint8),
        ("spareshort", np.int16),
        ("spareint6", np.int32),
        ("yIGFreq1", np.float64),
        ("yIGFreq2", np.float64),
        ("sflux", np.float64),
        ("ara", np.float64),
        ("adec", np.float64),
        ("mjd", np.float64),
    ]
)

eng_dtype = np.dtype(
    [
        ("antennaNumber", np.int32),
        ("padNumber", np.int32),
        ("antennaStatus", np.int32),
        ("trackStatus", np.int32),
        ("commStatus", np.int32),
        ("inhid", np.int32),
        ("ints", np.int32),
        ("dhrs", np.float64),
        ("ha", np.float64),
        ("lst", np.float64),
        ("pmdaz", np.float64),
        ("pmdel", np.float64),
        ("tiltx", np.float64),
        ("tilty", np.float64),
        ("actual_az", np.float64),
        ("actual_el", np.float64),
        ("azoff", np.float64),
        ("eloff", np.float64),
        ("az_tracking_error", np.float64),
        ("el_tracking_error", np.float64),
        ("refraction", np.float64),
        ("chopper_x", np.float64),
        ("chopper_y", np.float64),
        ("chopper_z", np.float64),
        ("chopper_angle", np.float64),
        ("tsys", np.float64),
        ("tsys_rx2", np.float64),
        ("ambient_load_temperature", np.float64),
    ]
)

bl_dtype = np.dtype(
    [
        ("blhid", np.int32),
        ("inhid", np.int32),
        ("isb", np.int16),
        ("ipol", np.int16),
        ("ant1rx", np.int16),
        ("ant2rx", np.int16),
        ("pointing", np.int16),
        ("irec", np.int16),
        ("u", np.float32),
        ("v", np.float32),
        ("w", np.float32),
        ("prbl", np.float32),
        ("coh", np.float32),
        ("avedhrs", np.float64),
        ("ampave", np.float32),
        ("phaave", np.float32),
        ("blsid", np.int32),
        ("iant1", np.int16),
        ("iant2", np.int16),
        ("ant1TsysOff", np.int32),
        ("ant2TsysOff", np.int32),
        ("iblcd", np.int16),
        ("ble", np.float32),
        ("bln", np.float32),
        ("blu", np.float32),
        ("spareint1", np.int32),
        ("spareint2", np.int32),
        ("spareint3", np.int32),
        ("spareint4", np.int32),
        ("spareint5", np.int32),
        ("spareint6", np.int32),
        ("fave", np.float64),
        ("bwave", np.float64),
        ("sparedbl3", np.float64),
        ("sparedbl4", np.float64),
        ("sparedbl5", np.float64),
        ("sparedbl6", np.float64),
    ]
)

sp_dtype = np.dtype(
    [
        ("sphid", np.int32),
        ("blhid", np.int32),
        ("inhid", np.int32),
        ("igq", np.int16),
        ("ipq", np.int16),
        ("iband", np.int16),
        ("ipstate", np.int16),
        ("tau0", np.float32),
        ("vel", np.float64),
        ("vres", np.float32),
        ("fsky", np.float64),
        ("fres", np.float32),
        ("gunnLO", np.float64),
        ("cabinLO", np.float64),
        ("corrLO1", np.float64),
        ("corrLO2", np.float64),
        ("integ", np.float32),
        ("wt", np.float32),
        ("flags", np.int32),
        ("vradcat", np.float32),
        ("nch", np.int16),
        ("nrec", np.int16),
        ("dataoff", np.int32),
        ("rfreq", np.float64),
        ("corrblock", np.int16),
        ("corrchunk", np.int16),
        ("correlator", np.int32),
        ("spareint2", np.int32),
        ("spareint3", np.int32),
        ("spareint4", np.int32),
        ("spareint5", np.int32),
        ("spareint6", np.int32),
        ("sparedbl1", np.float64),
        ("sparedbl2", np.float64),
        ("sparedbl3", np.float64),
        ("sparedbl4", np.float64),
        ("sparedbl5", np.float64),
        ("sparedbl6", np.float64),
    ]
)

codes_dtype = np.dtype(
    [("v_name", "S12"), ("icode", np.int16), ("code", "S26"), ("ncode", np.int16)]
)

we_dtype = np.dtype(
    [
        ("scanNumber", np.int32),
        ("flags", np.int32, 11),
        ("N", np.float32, 11),
        ("Tamb", np.float32, 11),
        ("pressure", np.float32, 11),
        ("humid", np.float32, 11),
        ("windSpeed", np.float32, 11),
        ("windDir", np.float32, 11),
        ("h2o", np.float32, 11),
    ]
)

ac_read_dtype = np.dtype(
    [
        ("inhid", np.int32),
        ("achid", np.int32),
        ("antenna", np.int32),
        ("nchunks", np.int32),
        ("datasize", np.int32),
        ("dataoff", np.int64),
        ("dhrs", np.float64),
    ]
)

antpos_dtype = np.dtype([("antenna", np.int16), ("xyz_pos", np.float64, 3)])


class MirParser(object):
    """
    General class for reading Mir datasets.

    Does lots of cool things! There are static functions that allow you low level
    access to mir files without needing to create an object.  You can also
    instantiate a MirParser object with the constructor of this class which will only
    read the metadata into memory by default. Read in the raw data through the
    use of the load_vis, load_raw, load_auto flags, or by using the load_data() function
    once the object is created. This allows for the flexible case of quickly loading
    metadata first to check whether or not to load additional data into memory.
    """

    def __init__(self, filepath, load_vis=False, load_raw=False, load_auto=False):
        """
        Read in all files from a mir data set into predefined numpy datatypes.

        The full dataset can be quite large, as such the default behavior of
        this function is only to load the metadata. Use the keyword params to
        load other data into memory.

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        load_vis : bool
            flag to load visibilities into memory, default is False.
        load_raw : bool
            flag to load raw data into memory, default is False.
        load_auto : bool
            flag to load auto-correlations into memory, default is False.
        """
        self.filepath = filepath
        self.in_read = self.read_in_data(filepath)
        self.eng_read = self.read_eng_data(filepath)
        self.bl_read = self.read_bl_data(filepath)
        self.sp_read = self.read_sp_data(filepath)
        self.codes_read = self.read_codes_data(filepath)
        self.we_read = self.read_we_data(filepath)
        self.ac_read = self.scan_auto_data(filepath)
        self.in_start_dict = self.scan_int_start(filepath)
        self.antpos_data = self.read_antennas(filepath)

        self.use_in = np.ones(self.in_read.shape, dtype=np.bool_)
        self.use_bl = np.ones(self.bl_read.shape, dtype=np.bool_)
        self.use_sp = np.ones(self.sp_read.shape, dtype=np.bool_)

        self.in_filter = np.ones(self.in_read.shape, dtype=np.bool_)
        self.eng_filter = np.ones(self.eng_read.shape, dtype=np.bool_)
        self.bl_filter = np.ones(self.bl_read.shape, dtype=np.bool_)
        self.sp_filter = np.ones(self.sp_read.shape, dtype=np.bool_)
        self.we_filter = np.ones(self.we_read.shape, dtype=np.bool_)
        self.ac_filter = np.ones(self.ac_read.shape, dtype=np.bool_)

        self.in_data = self.in_read
        self.eng_data = self.eng_read
        self.bl_data = self.bl_read
        self.sp_data = self.sp_read
        self.codes_data = self.codes_read
        self.we_data = self.we_read
        self.ac_data = self.ac_read

        # Raw data aren't loaded on start, because the datasets can be huge
        # You can force this after creating the object with load_data().
        self.vis_data = None
        self.raw_data = None
        self.auto_data = None
        self.raw_scale_fac = None

        self.inhid_dict = {}
        self.blhid_dict = {}
        self.sphid_dict = {}

        self.load_data(load_vis=load_vis, load_raw=load_raw, load_auto=load_auto)

    def _update_filter(self):
        """
        Update MirClass internal filters for the data.

        Expands the internal 'use_in', 'use_bl', and 'use_sp' arrays to
        construct filters for the individual structures/data

        Returns
        -------
        filter_changed : bool
            Indicates whether the underlying selected records changed.
        """
        old_in_filter = self.in_filter
        old_bl_filter = self.bl_filter
        old_sp_filter = self.sp_filter

        # Start with in-level records
        inhid_filter_dict = {
            key: value for key, value in zip(self.in_read["inhid"], self.use_in)
        }

        # Filter out de-selected bl records
        self.bl_filter = np.logical_and(
            self.use_bl, [inhid_filter_dict[key] for key in self.bl_read["inhid"]]
        )

        # Build a dict of bl-level records
        blhid_filter_dict = {
            key: value for key, value in zip(self.bl_read["blhid"], self.bl_filter)
        }

        # Filter out de-selected sp records
        self.sp_filter = np.logical_and(
            self.use_sp, [blhid_filter_dict[key] for key in self.sp_read["blhid"]]
        )

        # Check for bl records that have no good sp records
        sp_bl_check = {key: False for key in self.bl_read["blhid"]}
        sp_bl_check.update(
            {key: True for key in np.unique(self.sp_read["blhid"][self.sp_filter])}
        )

        # Filter out de-selected bl records
        self.bl_filter = np.logical_and(
            self.bl_filter, [sp_bl_check[key] for key in self.bl_read["blhid"]]
        )

        # Check for in records that have no good bl records
        bl_in_check = {key: False for key in self.in_read["inhid"]}
        bl_in_check.update(
            {key: True for key in np.unique(self.bl_read["inhid"][self.bl_filter])}
        )

        # Filter out de-selected in records
        self.in_filter = np.logical_and(
            self.use_in, [bl_in_check[key] for key in self.in_read["inhid"]]
        )

        # Create a temporary dictionary for filtering the remaining dtypes
        inhid_filter_dict = {
            key: value for key, value in zip(self.in_read["inhid"], self.in_filter)
        }

        # Filter out the last two data products, based on the above
        self.eng_filter = np.array(
            [inhid_filter_dict[key] for key in self.eng_read["inhid"]]
        )
        self.we_filter = np.array(
            [inhid_filter_dict[key] for key in self.we_read["scanNumber"]]
        )
        self.ac_filter = np.array(
            [inhid_filter_dict[key] for key in self.ac_read["inhid"]]
        )

        filter_changed = not (
            np.all(np.array_equal(old_sp_filter, self.sp_filter))
            and np.all(np.array_equal(old_bl_filter, self.bl_filter))
            and np.all(np.array_equal(old_in_filter, self.in_filter))
        )

        if filter_changed:
            self.in_data = self.in_read[self.in_filter]
            self.bl_data = self.bl_read[self.bl_filter]
            self.sp_data = self.sp_read[self.sp_filter]
            self.eng_data = self.eng_read[self.eng_filter]
            self.we_data = self.we_read[self.we_filter]
            self.ac_data = self.ac_read[self.ac_filter]

        # Craft some dictionaries so you know what list position matches
        # to each index entry. This helps avoid ordering issues.
        self.inhid_dict = {
            self.in_data["inhid"][idx]: idx for idx in range(len(self.in_data))
        }
        self.blhid_dict = {
            self.bl_data["blhid"][idx]: idx for idx in range(len(self.bl_data))
        }
        self.sphid_dict = {
            self.sp_data["sphid"][idx]: idx for idx in range(len(self.sp_data))
        }

        return filter_changed

    def load_data(self, load_vis=True, load_raw=False, load_auto=False):
        """
        Load visibility data into MirParser class.

        Parameters
        ----------
        load_vis : bool, optional
            Load the visibility data (floats) into object (deault is True)
        load_raw : bool, optional
            Load the raw visibility data (ints) into object (default is False)
        load_auto: bool, optional
            Load the autos (floats) into object (default is False)
        """
        self._update_filter()

        if load_vis:
            self.vis_data = self.parse_vis_data(
                self.filepath, self.in_start_dict, self.sp_data
            )
            self.vis_data_loaded = True
        if load_raw:
            self.raw_data, self.raw_scale_fac = self.parse_raw_data(
                self.filepath, self.in_start_dict, self.sp_data
            )
            self.raw_data_loaded = True
        if load_auto:
            # Have to do this because of a strange bug in data recording where
            # we record more autos worth of spectral windows than we actually
            # have online.
            winsel = np.unique(self.sp_data["corrchunk"])
            winsel = winsel[winsel != 0].astype(int) - 1
            self.auto_data = self.read_auto_data(
                self.filepath, self.ac_data, winsel=winsel
            )

    def unload_data(self):
        """Unload data from the MirParser object."""
        if self.vis_data is not None:
            self.vis_data = None
        if self.raw_data is not None:
            self.raw_data = None
        if self.auto_data is not None:
            self.auto_data = None

    @staticmethod
    def read_in_data(filepath):
        """
        Read "in_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        ndarray
            Numpy ndarray of custom dtype of in_dtype.
        """
        return np.fromfile(os.path.join(filepath, "in_read"), dtype=in_dtype)

    @staticmethod
    def read_eng_data(filepath):
        """
        Read "eng_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        ndarray
            Numpy ndarray of custom dtype of eng_dtype.
        """
        return np.fromfile(os.path.join(filepath, "eng_read"), dtype=eng_dtype)

    @staticmethod
    def read_bl_data(filepath):
        """
        Read "bl_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        ndarray
            Numpy ndarray of custom dtype of bl_dtype.
        """
        return np.fromfile(os.path.join(filepath, "bl_read"), dtype=bl_dtype)

    @staticmethod
    def read_sp_data(filepath):
        """
        Read "sp_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        ndarray
            Numpy ndarray of custom dtype of sp_dtype.
        """
        return np.fromfile(os.path.join(filepath, "sp_read"), dtype=sp_dtype)

    @staticmethod
    def read_codes_data(filepath):
        """
        Read "codes_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        ndarray
            Numpy ndarray of custom dtype of codes_dtype.
        """
        return np.fromfile(os.path.join(filepath, "codes_read"), dtype=codes_dtype)

    @staticmethod
    def read_we_data(filepath):
        """
        Read "we_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        ndarray
            Numpy ndarray of custom dtype of we_dtype.
        """
        return np.fromfile(os.path.join(filepath, "we_read"), dtype=we_dtype)

    @staticmethod
    def read_antennas(filepath):
        """
        Read "antennas" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        antpos_data : ndarray
            Numpy ndarray of custom dtype of antpos_dtype.
        """
        with open(os.path.join(filepath, "antennas"), "r") as antennas_file:
            temp_list = [
                item for line in antennas_file.readlines() for item in line.split()
            ]
        antpos_data = np.empty(len(temp_list) // 4, dtype=antpos_dtype)
        antpos_data["antenna"] = np.int16(temp_list[0::4])
        antpos_data["xyz_pos"] = np.array(
            [temp_list[1::4], temp_list[2::4], temp_list[3::4]], dtype=np.float64
        ).T

        return antpos_data

    @staticmethod
    def scan_int_start(filepath):
        """
        Read "sch_read" mir file into a python dictionary (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        dict
            Dictionary containing the indexes from sch_read.
        """
        full_filepath = os.path.join(filepath, "sch_read")
        file_size = os.path.getsize(full_filepath)
        with open(full_filepath, "rb") as visibilities_file:
            data_offset = 0
            last_offset = 0
            in_offset_dict = {}
            while data_offset < file_size:
                int_vals = np.fromfile(
                    visibilities_file,
                    dtype=np.dtype([("inhid", np.int32), ("insize", np.int32)]),
                    count=1,
                    offset=last_offset,
                )
                in_offset_dict[int_vals["inhid"][0]] = (
                    int_vals["insize"][0] + 8,
                    data_offset,
                )
                last_offset = int_vals["insize"][0]
                data_offset += last_offset + 8

        return in_offset_dict

    @staticmethod
    def scan_auto_data(filepath, nchunks=8):
        """
        Read "autoCorrelations" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        nchunks : int, optional
            Specify the number of chunks recorded into the autocorrelations
            (default is 8)

        Returns
        -------
        ac_data : ndarray
            Numpy ndarray of custom type ac_read_dtype.
        """
        full_filepath = os.path.join(filepath, "autoCorrelations")
        file_size = os.path.getsize(full_filepath)
        with open(full_filepath, "rb") as auto_file:
            data_offset = 0
            last_offset = 0
            acfile_dtype = np.dtype(
                [
                    ("antenna", np.int32),
                    ("nChunks", np.int32),
                    ("scan", np.int32),
                    ("dhrs", np.float64),
                ]
            )
            marker = 0
            while data_offset < file_size:
                auto_vals = np.fromfile(
                    auto_file, dtype=acfile_dtype, count=1, offset=last_offset
                )
                # This bit of code is to trap an unfortunately
                # common problem with metadata of MIR autos not
                # being correctly recorded.
                if data_offset == 0:
                    if (file_size % (4 * (2 ** 14) * nchunks * 2 + 20)) != 0:
                        nchunks = int(auto_vals["nChunks"][0])
                        if (file_size % (4 * (2 ** 14) * nchunks * 2 + 20)) != 0:
                            raise IndexError(
                                "Could not determine auto-correlation record size!"
                            )
                    # How big the record is for each data set
                    last_offset = 4 * (2 ** 14) * int(nchunks) * 2
                    ac_data = np.zeros(
                        file_size // ((4 * (2 ** 14) * int(nchunks) * 2 + 20)),
                        dtype=ac_read_dtype,
                    )
                ac_data[marker] = (
                    auto_vals["scan"][0],
                    marker + 1,
                    auto_vals["antenna"][0],
                    nchunks,
                    last_offset + 20,
                    data_offset,
                    auto_vals["dhrs"][0],
                )
                data_offset += last_offset + 20
                marker += 1
        return ac_data

    @staticmethod
    def parse_vis_data(filepath, in_start_dict, sp_data):
        """
        Read "sch_read" mir file into a list of ndarrays. (@staticmethod).

        Parameters
        ----------
        filepath : str
            Path to the folder containing the mir data set.
        in_start_dict: dict of tuples
            Dictionary returned from scan_int_start, which records position and
            record size for each integration
        sp_data : ndarray of sp_data_type
            Array from "sp_read", returned by "read_sp_read".

        Returns
        -------
        vis_list : list of ndarrays
            List of ndarrays (dtype=csingle/complex64), with indices equal to sphid
            and values being the floating-point visibilities for the spectrum
        """
        # Gather the needed metadata
        inhid_arr = sp_data["inhid"]
        nch_arr = sp_data["nch"]
        dataoff_arr = sp_data["dataoff"] // 2
        sp_pos = np.arange(len(dataoff_arr))

        unique_inhid = np.unique(inhid_arr)
        unique_nch = np.unique(nch_arr)
        vis_list = []
        sp_pos_list = []
        for inhid in unique_inhid:
            packdata = MirParser.read_vis_data(filepath, {inhid: in_start_dict[inhid]})[
                inhid
            ]["packdata"]
            data_mask = inhid_arr == inhid
            dataoff_subarr = dataoff_arr[data_mask]
            nch_subarr = nch_arr[data_mask]
            sp_pos_subarr = sp_pos[data_mask]
            scale_fac = np.power(2.0, packdata[dataoff_subarr], dtype=np.float32)
            for spec_size in unique_nch:
                start_list = dataoff_subarr[spec_size == nch_subarr] + 1
                end_list = start_list + (spec_size * 2)
                temp_data = np.multiply(
                    scale_fac[spec_size == nch_subarr, None],
                    np.array(
                        [packdata[idx:jdx] for idx, jdx in zip(start_list, end_list)]
                    ),
                    dtype=np.float32,
                ).view(dtype=np.complex64)
                vis_list.extend([temp_data[idx, :] for idx in range(len(start_list))])
                # Record where the data _should_ go in the list
                sp_pos_list.extend(sp_pos_subarr[spec_size == nch_subarr])
        # Do a bit of order swapping so that things match with sp_read
        sp_pos_dict = {sp_pos_list[idx]: idx for idx in range(len(sp_pos_list))}
        vis_list = [vis_list[sp_pos_dict[idx]] for idx in range(len(sp_pos_list))]
        return vis_list

    @staticmethod
    def parse_raw_data(filepath, in_start_dict, sp_data):
        """
        Read "sch_read" mir file into a python dictionary (@staticmethod).

        Note that this returns a list rather than a dict to help expedite some
        processing tasks where order of the arrays matters.

        Parameters
        ----------
        filepath : str
            Path to the folder containing the mir data set.
        in_start_dict: dict of tuples
            Dictionary returned from scan_int_start, which records position and
            record size for each integration
        sp_data : ndarray of sp_data_type
            Array from "sp_read", returned by "read_sp_read".

        Returns
        -------
        vis_list : list of ndarray
            List of ndarrays (dtype=int16), each containing the 'raw' set
            of values in "sch_read", ordered in the same way that as sp_data.
        scale_fac_list : list of int16
            List of scale factors (in log2 units) for each item in vis_list,
            also orded identically to sp_data.
        """
        # Gather the needed metadata
        inhid_arr = sp_data["inhid"]
        nch_arr = sp_data["nch"].astype(np.int64)
        dataoff_arr = sp_data["dataoff"] // 2

        unique_inhid = np.unique(inhid_arr)
        vis_list = []
        scale_fac_list = []
        for inhid in unique_inhid:
            packdata = MirParser.read_vis_data(filepath, {inhid: in_start_dict[inhid]})[
                inhid
            ]["packdata"]
            data_mask = inhid_arr == inhid
            dataoff_subarr = dataoff_arr[data_mask]
            nch_subarr = nch_arr[data_mask]
            scale_fac_list += copy.deepcopy(packdata[dataoff_subarr]).tolist()
            start_list = dataoff_subarr + 1
            end_list = start_list + (nch_subarr * 2)
            vis_list += [
                copy.deepcopy(packdata[idx:jdx])
                for idx, jdx in zip(start_list, end_list)
            ]
        return vis_list, scale_fac_list

    @staticmethod
    def read_auto_data(filepath, ac_data, winsel=None):
        """
        Read "autoCorrelations" mir file into memory (@staticmethod).

        Note that this returns as an array, since there isn't any unique index for the
        autocorrelations file.

        Parameters
        ----------
        filepath : str
            Path to the folder containing the mir data set.
        ac_data : arr of dtype ac_read_dtype
            Structure from returned from scan_auto_data.
        winsel : list of int (optional)
            List of spectral windows to include.

        Returns
        -------
        auto_data : arr of single
            An array of shape (n_ch, n_chunk, n_rec), which containts the auto spectra,
            where n_ch is number of channels (currently always 16384 per chunk),
            n_chunk is the number of spectral "chunks" (i.e., Nspws), and n_rec is the
            number of receivers per antenna recorded (always 2 -- 1 per polarization).
        """
        if winsel is None:
            winsel = np.arange(0, ac_data["nchunks"][0])

        winsel = np.array(winsel)
        # The current generation correlator always produces 2**14 == 16384 channels per
        # spectral window.
        # TODO: Allow this to be flexible if dealing w/ spectrally averaged data
        # (although it's only use currently is in its unaveraged formal for
        # normaliztion of the crosses)
        auto_data = np.empty((len(ac_data), len(winsel), 2, 2 ** 14), dtype=np.float32)
        dataoff = ac_data["dataoff"]
        datasize = ac_data["datasize"]
        del_offset = np.insert(np.diff(dataoff) - datasize[0:-1], 0, dataoff[0])
        nvals = ac_data["nchunks"] * 2 * (2 ** 14)
        nchunks = ac_data["nchunks"]

        with open(os.path.join(filepath, "autoCorrelations"), "rb") as auto_file:
            for idx in range(len(dataoff)):
                auto_data[idx] = np.fromfile(
                    auto_file,
                    dtype=np.float32,
                    count=nvals[idx],
                    offset=20 + del_offset[idx],
                ).reshape((nchunks[idx], 2, 2 ** 14))[winsel, :, :]

        return auto_data

    @staticmethod
    def read_vis_data(filepath, in_start_dict):
        """
        Read "sch_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        in_start_dict : dict
            indexes to the visibility locations within the file.

        Returns
        -------
        in_data_dict : dict
            Dictionary of the data, where the keys are inhid and the values are
            the 'raw' block of values recorded in "sch_read" for that inhid.
        """
        in_data_dict = {}
        in_dtype_dict = {}
        size_list = np.unique(
            [in_start_dict[ind_key][0] for ind_key in in_start_dict.keys()]
        )

        for in_size in size_list:
            in_dtype_dict[in_size] = np.dtype(
                [
                    ("inhid", np.int32),
                    ("nbyt", np.int32),
                    ("packdata", np.int16, (in_size - 8) // 2),
                ]
            )

        inhid_list = []
        with open(os.path.join(filepath, "sch_read"), "rb") as visibilities_file:
            last_offset = last_size = num_vals = del_offset = 0
            key_list = sorted(in_start_dict.keys())
            # We add an extra key here, None, which cannot match any of the values in
            # in_start_dict (since inhid is type int). This basically tricks the loop
            # below into spitting out the last integration
            key_list.append(None)
            for ind_key in key_list:
                if ind_key is None:
                    in_size = in_start = 0
                else:
                    (in_size, in_start) = in_start_dict[ind_key]
                if (in_size != last_size) or (
                    last_offset + last_size * num_vals != in_start
                ):
                    # Numpy's fromfile works fastest when reading multiple instances
                    # of the same dtype. As long as the record sizes are the same, we
                    # can tie mutiple file reads together into one call. The dtype
                    # depends on the record size, which is why we have to dump the
                    # data when last_size changes.
                    if num_vals != 0 and last_size != 0:
                        in_data_dict.update(
                            zip(
                                inhid_list,
                                np.fromfile(
                                    visibilities_file,
                                    dtype=in_dtype_dict[last_size],
                                    count=num_vals,
                                    offset=del_offset,
                                ).copy(),
                            )
                        )
                    del_offset = in_start - (last_offset + (num_vals * last_size))
                    last_offset = in_start
                    last_size = in_size
                    num_vals = 0
                    inhid_list = []
                num_vals += 1
                inhid_list.append(ind_key)
        return in_data_dict
