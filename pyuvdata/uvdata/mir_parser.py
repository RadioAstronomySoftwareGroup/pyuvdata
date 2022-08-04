# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Module for low-level interface to Mir datasets.

This module provides a python interface with Mir datasets, including both metadata
and the visibility data itself.
"""
import copy
import os
import warnings
from functools import partial

import h5py
import numpy as np

from .mir_meta_data import (
    MirAcData,
    MirAntposData,
    MirBlData,
    MirCodesData,
    MirEngData,
    MirInData,
    MirMetaData,
    MirMetaError,
    MirSpData,
    MirWeData,
)

__all__ = ["MirParser"]


class MirParser(object):
    """
    General class for reading Mir datasets.

    Does lots of cool things! There are static functions that allow you low level
    access to mir files without needing to create an object.  You can also
    instantiate a MirParser object with the constructor of this class which will only
    read the metadata into memory by default. Read in the raw data through the
    use of the load_cross and load_auto flags, or by using the load_data() function
    once the object is created. This allows for the flexible case of quickly loading
    metadata first to check whether or not to load additional data into memory.
    """

    def __init__(
        self,
        filepath=None,
        has_auto=False,
        has_cross=True,
        load_auto=False,
        load_cross=False,
        load_raw=False,
    ):
        """
        Initialize a MirParser object.

        The full dataset can be quite large, as such the default behavior of
        this function is only to load the metadata. Use the keyword params to
        load other data into memory.

        Parameters
        ----------
        filepath : str
            Filepath is the path to the folder containing the Mir data set.
        has_auto : bool
            Flag to read auto-correlation data. Default is False.
        has_cross : bool
            Flag to read cross-correlation data. Default is True.
        load_auto : bool
            Flag to load auto-correlations into memory. Default is False.
        load_cross : bool
            Flag to load visibilities into memory. Default is False.
        load_raw : bool
            Flag to load raw data into memory. Default is False.
        """
        self.in_data = MirInData()
        self.bl_data = MirBlData()
        self.sp_data = MirSpData()
        self.we_data = MirWeData()
        self.eng_data = MirEngData()
        self.codes_data = MirCodesData()
        self.antpos_data = MirAntposData()
        self.ac_data = MirAcData()

        self._metadata_attrs = {
            "in_data": self.in_data,
            "bl_data": self.bl_data,
            "sp_data": self.sp_data,
            "eng_data": self.eng_data,
            "we_data": self.we_data,
            "codes_data": self.codes_data,
            "antpos_data": self.antpos_data,
        }

        self.filepath = ""
        self._file_dict = {}
        self._sp_dict = {}
        self._ac_dict = {}

        self.raw_data = None
        self.vis_data = None
        self.auto_data = None
        self._has_auto = False
        self._has_cross = False
        self._tsys_applied = False

        # This value is the forward gain of the antenna (in units of Jy/K), which is
        # multiplied against the system temperatures in order to produce values in units
        # of Jy (technically this is the SEFD, which when multiplied against correlator
        # coefficients produces visibilities in units of Jy). Default is 130.0, which
        # is the estimated value for SMA.
        self.jypk = 130.0

        # On init, if a filepath is provided, then fill in the object
        if filepath is not None:
            self.read(
                filepath,
                has_auto=has_auto,
                has_cross=has_cross,
                load_auto=load_auto,
                load_cross=load_cross,
                load_raw=load_raw,
            )

    def __eq__(self, other, verbose=True, metadata_only=False):
        """
        Compare MirParser attributes for equality.

        Parameters
        ----------
        other : MirParser
            MirParser object to compare with.
        verbose : bool
            If True, will print out all of the differences between the two objects.
            Default is True.
        metadata_only : bool
            If True, the attributes `auto_data`, `raw_data`, and `vis_data` will not
            be compared between objects. Default is False.

        Returns
        -------
        check : bool
            Whether or not the two objects are equal.
        """
        if not isinstance(other, self.__class__):
            raise ValueError("Cannot compare MirParser with %s." % type(other).__name__)

        data_comp_dict = {
            "raw_data": ["data", "scale_fac"],
            "vis_data": ["data", "flags"],
            "auto_data": ["data", "flags"],
        }

        # I say these objects are the same -- prove me wrong!
        is_eq = True

        # First up, check the list of attributes between the two objects
        this_attr_set = set(vars(self))
        other_attr_set = set(vars(other))

        verbose_print = print if verbose else lambda *a, **k: None

        # Go through and drop any attributes that both objects do not have (and set
        # is_eq to False if any such attributes found).
        for item in this_attr_set.union(other_attr_set):
            target = None
            if item not in this_attr_set:
                other_attr_set.remove(item)
                target = "right"
            elif item not in other_attr_set:
                this_attr_set.remove(item)
                target = "left"
            if target is not None:
                is_eq = False
                verbose_print("%s does not exist in %s." % (item, target))

        if metadata_only:
            for item in ["vis_data", "raw_data", "auto_data"]:
                this_attr_set.remove(item)

        # At this point we _only_ have attributes present in both lists
        for item in this_attr_set:
            this_attr = getattr(self, item)
            other_attr = getattr(other, item)

            # Make sure the attributes are of the same type to help ensure
            # we can actually compare the two without error.
            if not isinstance(this_attr, type(other_attr)):
                is_eq = False
                verbose_print(
                    "%s is of different types, left is %s, right is %s."
                    % (item, type(this_attr), type(other_attr))
                )
                continue
            elif this_attr is None:
                # If both are NoneType, we actually have nothing to do here
                pass
            elif item in ["auto_data", "raw_data", "vis_data"]:
                # Data-related attributes are a bit special, in that they are dicts
                # of dicts (note this may change at some point).
                if this_attr.keys() != other_attr.keys():
                    is_eq = False
                    verbose_print(
                        f"{item} has different keys, left is {this_attr.keys()}, "
                        f"right is {other_attr.keys()}."
                    )
                    continue

                comp_list = data_comp_dict[item]

                # For the attributes with multiple fields to check, list them
                # here for convenience.
                for key in this_attr:
                    this_item = this_attr[key]
                    other_item = other_attr[key]

                    is_same = True
                    for subkey in comp_list:
                        if subkey == "scale_fac":
                            is_same &= this_item[subkey] == other_item[subkey]
                        elif not np.array_equal(this_item[subkey], other_item[subkey]):
                            if this_item[subkey].shape == other_item[subkey].shape:
                                # The atol here is set by the max value in the spectrum
                                # times 2^-10. That turns out to be _about_ the worst
                                # case scenario for moving to and from the raw data
                                # format, which compresses the data down from floats to
                                # ints.
                                atol = 1e-3
                                if np.any(np.isfinite(this_item[subkey])):
                                    atol *= np.nanmax(np.abs(this_item[subkey]))

                                is_same &= np.allclose(
                                    this_item[subkey],
                                    other_item[subkey],
                                    atol=atol,
                                    equal_nan=True,
                                )
                            else:
                                is_same = False
                    if not is_same:
                        is_eq = False
                        verbose_print(
                            "%s has the same keys, but different values." % item
                        )
                        break
                # We are done processing the data dicts at this point, so we can skip
                # the item_same evaluation below.
            elif issubclass(type(this_attr), MirMetaData):
                is_eq &= this_attr.__eq__(other_attr, verbose=verbose)
            elif item == "_metadata_attrs":
                if this_attr.keys() != other_attr.keys():
                    is_eq = False
                    verbose_print(
                        f"{item} has different keys, left is {this_attr.keys()}, "
                        f"right is {other_attr.keys()}."
                    )
            else:
                # We don't have special handling for this attribute at this point, so
                # we just use the generic __ne__ method.
                if this_attr != other_attr:
                    is_eq = False
                    verbose_print(
                        f"{item} has different values, left is {this_attr}, "
                        f"right is {other_attr}."
                    )

        return is_eq

    def __ne__(self, other, verbose=False, metadata_only=False):
        """
        Compare two MirParser objects for inequality.

        Parameters
        ----------
        other : MirParser
            MirParser object to compare with.
        verbose : bool
            If True, will print out all of the differences between the two objects.
            Default is False.
        metadata_only : bool
            If True, the attributes `auto_data`, `raw_data`, and `vis_data` will not
            be compared between objects. Default is False.

        Returns
        -------
        check : bool
            Whether or not the two objects are different.
        """
        return not self.__eq__(other, verbose=verbose, metadata_only=metadata_only)

    def copy(self, metadata_only=False):
        """
        Make and return a copy of the MirParser object.

        Parameters
        ----------
        metadata_only : bool
            If set to True, will forgo copying the attributes `vis_data`, raw_data`,
            and `auto_data`.

        Returns
        -------
        MirParser
            Copy of self.
        """
        new_obj = MirParser()

        # include all attributes, not just UVParameter ones.
        for attr in vars(self):
            if not (metadata_only and attr in ["vis_data", "raw_data", "auto_data"]):
                setattr(new_obj, attr, copy.deepcopy(getattr(self, attr)))

        for item in self._metadata_attrs:
            new_obj._metadata_attrs[item] = getattr(new_obj, item)

        return new_obj

    @staticmethod
    def _scan_int_start(filepath, allowed_inhid=None):
        """
        Read "sch_read" or "ach_read" mir file into a python dictionary (@staticmethod).

        Parameters
        ----------
        filepath : str
            Filepath is the path to the folder containing the Mir data set.
        allowed_inhid : list of int
            List of allowed integration header key numbers ("inhid") that should be in
            this dataset. If a header key is not found in this list, then the method
            will exit with an error. No default value (all values allowed).

        Returns
        -------
        int_dict : dict
            Dictionary containing the indexes from sch_read, where keys match to the
            inhid indexes, and the values contain a two-element tuple, with the length
            of the packdata array (in bytes) the relative offset (also in bytes) of
            the record within the sch_read file.

        Raises
        ------
        ValueError
            If a value on "inhid" is read from the file that does not match a value
            given in `allowed_inhid` (if set).
        """
        file_size = os.path.getsize(filepath)
        data_offset = 0
        last_offset = 0
        int_dict = {}
        with open(filepath, "rb") as visibilities_file:
            while data_offset < file_size:
                int_vals = np.fromfile(
                    visibilities_file,
                    dtype=np.dtype([("inhid", "<i4"), ("nbytes", "<i4")]),
                    count=1,
                    offset=last_offset,
                )[0]

                if allowed_inhid is not None:
                    if not int_vals["inhid"] in allowed_inhid:
                        raise ValueError(
                            "Index value inhid in sch_read does not match list of "
                            "allowed indices. The list of allowed values for inhid may "
                            "be incomplete, or sch_read may have become corrupted in "
                            "some way."
                        )
                int_dict[int_vals["inhid"]] = {
                    "inhid": int_vals["inhid"],
                    "record_size": int_vals["nbytes"],
                    "record_start": data_offset,
                }
                last_offset = int_vals["nbytes"].astype(int)
                data_offset += last_offset + 8

        return int_dict

    def _fix_int_dict(self, data_type):
        """
        Fix an integration position dictionary.

        Note that this function is not meant to be called by users, but is instead an
        internal helper method for handling error correction when reading in data.
        This method will fix potential errors in an internal dictionary used to mark
        where in the main visibility file an individual spectral record is located.
        Under normal conditions, this routine does not need to be run, unless another
        method reported a specific error on read calling for the user to run this code.

        Parameters
        ----------
        data_type : str
            Type of data to fix , must either be "cross" (cross-correlations) or "auto"
            (auto-correlations).
        """
        for ifile, idict in self._file_dict.items():
            if not idict[data_type]["ignore_header"]:
                int_dict = copy.deepcopy(idict[data_type]["int_dict"])

            # Each file's inhid is allowed to be different than the objects inhid --
            # this is used in cases when combining multiple files together (via
            # concat). Here, we make a mapping of "file-based" inhid values to that
            # stored in the object.
            imap = {val["inhid"]: inhid for inhid, val in int_dict.items()}

            # Make the new dict by scanning the sch_read file.
            new_dict = self._scan_int_start(
                os.path.join(ifile, idict[data_type]["filetype"]), list(imap)
            )

            # Go through the individual entries in each dict, and update them
            # with the "correct" values as determined by scanning through sch_read
            for key in new_dict:
                int_dict[imap[key]] = new_dict[key]

            idict[data_type]["int_dict"] = int_dict

    @staticmethod
    def _read_packdata(file_dict, inhid_arr, data_type="cross", use_mmap=False):
        """
        Read "sch_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        file_dict : dict
            Dictionary which maps individual integrations to a specific packed data
            record on disk. Keys are the path(s) to the data, with values that are
            themselves dicts with keys of "auto" and/or "cross", which map to per-file
            indexing information ("filetype": the name of the file in the Mir folder;
            "int_dict": per-integration information that is typically generated by the
            `_generate_recpos_dict` method of `MirParser.sp_data` and/or
            `MirParser.ac_data`; "ignore_header": if set to True, disables checking
            for header metadata consistency).
        inhid_arr : sequence of int
            Integration header keys to read the packed data of.
        data_type : str
            Type of data to read, must either be "cross" (cross-correlations) or "auto"
            (auto-correlations). Default is "cross".
        use_mmap : bool
            By default, the method will read all of the data into memory. However,
            if set to True, then the method will return mmap-based objects instead,
            which can be substantially faster on sparser reads.

        Returns
        -------
        int_data_dict : dict
            Dictionary of the data, where the keys are inhid and the values are
            the 'raw' block of values recorded in "sch_read" for that inhid.
        """
        int_data_dict = {}
        int_dtype_dict = {}

        # We want to create a unique dtype for records of different sizes. This will
        # make it easier/faster to read in a sequence of integrations of the same size.
        size_set = {
            rec_dict["record_size"]
            for idict in file_dict.values()
            for rec_dict in idict[data_type]["int_dict"].values()
        }

        for int_size in size_set:
            int_dtype_dict[int_size] = np.dtype(
                [("inhid", "<i4"), ("nbytes", "<i4"), ("packdata", "B", int_size)]
            )

        key_set = list(inhid_arr)
        key_check = key_set.copy()
        # We add an extra key here, None, which cannot match any of the values in
        # int_start_dict (since inhid is type int). This basically tricks the loop
        # below into spitting out the last integration
        key_set.append(None)
        for filepath, indv_file_dict in file_dict.items():
            # Initialize a few values before we start running through the data.
            int_dict = indv_file_dict[data_type]["int_dict"]
            inhid_list = []
            last_offset = last_size = num_vals = del_offset = 0

            # Read list is basically storing all of the individual reads that we need to
            # execute in order to grab all of the data we need. Note that each entry
            # here is going to correspond to a call to either np.fromfile or np.memmap.
            read_list = []

            for ind_key in key_set:
                if ind_key is None:
                    # This helps flush out the last read/integration in this loop
                    int_size = int_start = 0
                else:
                    try:
                        rec_dict = int_dict[ind_key]
                        key_check.remove(ind_key)
                    except KeyError:
                        continue
                    int_size = rec_dict["record_size"]
                    int_start = rec_dict["record_start"]
                if (int_size != last_size) or (
                    last_offset + (8 + last_size) * num_vals != int_start
                ):
                    # Numpy's fromfile works fastest when reading multiple instances
                    # of the same dtype. As long as the record sizes are the same, we
                    # can tie multiple file reads together into one call. The dtype
                    # depends on the record size, which is why we have to dump the
                    # data when last_size changes.
                    if num_vals != 0 and last_size != 0:
                        read_list.append(
                            {
                                "inhid_list": inhid_list,
                                "int_dtype_dict": int_dtype_dict[last_size],
                                "num_vals": num_vals,
                                "del_offset": del_offset,
                                "start_offset": last_offset,
                            }
                        )
                    # Capture the difference between the last integration and this
                    # integration that we're going to drop into the next read.
                    del_offset = int_start - (last_offset + (num_vals * last_size))
                    # Starting position for a sequence of integrations
                    last_offset = int_start
                    # Size of record (make sure all records are the same size in 1 read)
                    last_size = int_size
                    # Number of integrations in the read
                    num_vals = 0
                    # Tally all the inhid values in the read
                    inhid_list = []
                num_vals += 1
                inhid_list.append(ind_key)
            filename = os.path.join(filepath, indv_file_dict[data_type]["filetype"])
            # Time to actually read in the data
            if use_mmap:
                # memmap is a little special, in that it wants the _absolute_ offset
                # rather than the relative offset that np.fromfile uses (if passing a
                # file object rather than a string with the path toward the file).
                for read_dict in read_list:
                    int_data_dict.update(
                        zip(
                            read_dict["inhid_list"],
                            np.memmap(
                                filename=filename,
                                dtype=read_dict["int_dtype_dict"],
                                mode="r",
                                offset=read_dict["start_offset"],
                                shape=(read_dict["num_vals"],),
                            ),
                        )
                    )
            else:
                with open(filename, "rb") as visibilities_file:
                    # Note that we do one open here to avoid the overheads associated
                    # with opening and closing the file each integration.
                    for read_dict in read_list:
                        int_data_dict.update(
                            zip(
                                read_dict["inhid_list"],
                                np.fromfile(
                                    visibilities_file,
                                    dtype=read_dict["int_dtype_dict"],
                                    count=read_dict["num_vals"],
                                    offset=read_dict["del_offset"],
                                ),
                            )
                        )

            if not indv_file_dict[data_type]["ignore_header"]:
                good_check = True
                for inhid, idict in int_data_dict.items():
                    # There is very little to check in the packdata records, so make
                    # sure that this entry corresponds to the inhid and size we expect.
                    good_check &= idict["inhid"] == int_dict[inhid]["inhid"]
                    good_check &= idict["nbytes"] == int_dict[inhid]["record_size"]

                if not good_check:
                    raise MirMetaError(
                        "File indexing information differs from that found in in "
                        "file_dict. Cannot read in %s data." % data_type
                    )

        if len(key_check) != 0:
            raise ValueError("inhid_arr contains keys not found in file_dict.")

        return int_data_dict

    @staticmethod
    def _make_packdata(int_dict, recpos_dict, data_dict, data_type):
        """
        Write packdata from raw_data or auto_data.

        This method will convert regular data into "packed data", ready to be written to
        disk (i.e., MIR-formatted). This method is typically called by file writing
        utilities.

        Parameters
        ----------
        int_dict : dict
            Dictionary describing the data on disk, with keys matched to individual
            integration header numbers (`inhid`) and values themselves dicts containing
            meta-information about the spectral record. This dict is generally produced
            by `MirMetaData._generate_recpos_dict`, where further documentation can be
            found.
        recpos_dict : dict
            Dictionary containing the spectral record indexing information, where keys
            are unique values in the integration record number (`inhid`), and the values
            are themselves dicts, with keys matched to spectral record numbers (`sphid`
            or `achid`). This dict is produced by `MirMetaData._generate_recpos_dict`,
            the documentation of which contains further information.
        data_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual spectral record numbers, and each entry contains a dict with two
            items. If `data_type="cross"`, those two items are "scale_fac", an np.int16
            which describes the common exponent for the spectrum, and "data", an array
            of np.int16.  Note that entries equal to -32768 aren't possible with the
            compression scheme used for MIR, and so this value is used to mark flags.
            If `data_type="auto"`, then those two items are "data", an array of
            np.float32, and "flags", an array of np.bool.
        data_type : str
            Type of data to read, must either be "cross" (cross-correlations) or "auto"
            (auto-correlations).

        Returns
        -------
        int_data_dict : dict
            A dict whose keys correspond to the unique values of "inhid" in `sp_data`,
            and values correspond to the packed data arrays -- an ndarray with a
            custom dtype. Each packed data element contains three fields: "inhid" (
            nominally matching the keys mentioned above), "nbytes" describing the size
            of the packed data (in bytes), and "packdata", which is the packed raw
            data.
        """
        if data_type == "cross":
            val_dtype = "<i2"
            is_cross = True
        elif data_type == "auto":
            val_dtype = "<f4"
            is_cross = False
        else:
            raise ValueError(
                'Argument for data_type not recognized, must be "cross" or "auto".'
            )
        # Figure out all of the unique dtypes we need for constructing the individual
        # packed datasets (where we need a different dtype based on the number of
        # individual visibilities we're packing in).
        int_dtype_dict = {}
        for int_size in {idict["record_size"] for idict in int_dict.values()}:
            int_dtype_dict[int_size] = np.dtype(
                [("inhid", "<i4"), ("nbytes", "<i4"), ("packdata", "B", int_size)]
            )

        # Now we start the heavy lifting -- start looping through the individual
        # integrations and pack them together.
        int_data_dict = {}
        for inhid, int_subdict in int_dict.items():
            # Make an empty packdata dtype, which we will fill with new values
            int_data = np.empty((), dtype=int_dtype_dict[int_subdict["record_size"]])

            # Convenience dict which contains the sphid as keys and start/stop of
            # the slice for each spectral record as values for each integration.
            recpos_subdict = recpos_dict[inhid]

            # Plug in the "easy" parts of packdata
            int_data["inhid"] = inhid
            int_data["nbytes"] = int_subdict["record_size"]

            # Now step through all of the spectral records and plug it in to the
            # main packdata array. In testing, this worked out to be a good degree
            # faster than running np.concat.
            packdata = int_data["packdata"].view(val_dtype)
            for hid, recinfo in recpos_subdict.items():
                data_record = data_dict[hid]
                start_idx = recinfo["start_idx"]
                end_idx = recinfo["end_idx"]
                if is_cross:
                    packdata[start_idx] = data_record["scale_fac"]
                    packdata[(start_idx + 1) : end_idx] = data_record["data"]
                else:
                    packdata[start_idx:end_idx] = data_record["data"]

            int_data_dict[inhid] = int_data

        return int_data_dict

    @staticmethod
    def _convert_raw_to_vis(raw_dict):
        """
        Create a dict with visibility data via a raw data dict.

        Parameters
        ----------
        raw_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry contains a dict
            with two items: "scale_fac", and np.int16 which describes the common
            exponent for the spectrum, and "data", an array of np.int16 (of length
            equal to twice that found in `sp_data["nch"]` for the corresponding value
            of sphid) containing the compressed visibilities.  Note that entries equal
            to -32768 aren't possible with the compression scheme used for MIR, and so
            this value is used to mark flags.

        Returns
        -------
        vis_dict : dict
            A dictionary in the format of `vis_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry contains a dict
            with two items: "data", an array of np.complex64 containing the
            visibilities, and "flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).
        """
        # The code here was derived after a bunch of testing, trying to find the fastest
        # way to covert the compressed data into the "normal" format. Some notes:
        #   1) The setup below is actually faster than ldexp, probably because of
        #      the specific dtype we are using.
        #   2) Casting 2 as float32 will appropriately cast sp_raw values into float32
        #      as well.
        #   3) I only check for the "special" value for flags in the real component. A
        #      little less robust (both real and imag are marked), but faster and
        #      barring data corruption, this shouldn't be an issue (and a single bad
        #      channel sneaking through is okay).
        #   4) pairs of float32 -> complex64 is super fast and efficient.
        vis_dict = {
            sphid: {
                "data": ((np.float32(2) ** sp_raw["scale_fac"]) * sp_raw["data"]).view(
                    dtype=np.complex64
                ),
                "flags": sp_raw["data"][::2] == -32768,
            }
            for sphid, sp_raw in raw_dict.items()
        }

        # In testing, flagging the bad channels out after-the-fact was significantly
        # faster than trying to modify in situ w/ the call above.
        for item in vis_dict.values():
            item["data"][item["flags"]] = 0.0

        return vis_dict

    @staticmethod
    def _convert_vis_to_raw(vis_dict):
        """
        Create a dict with visibility data via a raw data dict.

        Parameters
        ----------
        vis_dict : dict
            A dictionary in the format of `vis_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry contains a dict
            with two items: "data", an array of np.complex64 containing the
            visibilities, and "flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).

        Returns
        -------
        raw_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry contains a dict
            with two items: "scale_fac", and np.int16 which describes the common
            exponent for the spectrum, and "data", an array of np.int16 (of length
            equal to twice that found in `sp_data["nch"]` for the corresponding value
            of sphid) containing the compressed visibilities.  Note that entries equal
            to -32768 aren't possible with the compression scheme used for MIR, and so
            this value is used to mark flags.
        """
        # Similar to _convert_raw_to_vis, fair bit of testing went into making this as
        # fast as possible. Strangely enough, frexp is _way_ faster than ldexp.
        # Note that we only want to calculate a common exponent based on the unflagged
        # spectral channels.
        # Note that the minimum max value here is set to the relative double precision
        # limit, in part because it's the limit of what one might trust for the
        # correlator coefficients, and it's well below the nominal sensitivity limit
        # of any real telescope (femtoJanskys FTW!).
        scale_fac = np.frexp(
            [
                np.abs(sp_vis["data"].view(dtype=np.float32)).max(
                    initial=2.220446049250313e-16
                )
                for sp_vis in vis_dict.values()
            ]
        )[1].astype(np.int16) - np.int16(15)
        # Note we need the -15 above because the range of raw_data goes from - 2**15
        # to 2**15 (being an int16 value).

        # Again, the 10 lines below were the product of lots of speed testing.
        #   1) The setup below is actually faster than ldexp, probably because of
        #      the specific dtype we are using.
        #   2) Casting 2 as float32 saves on complex multiplies, and makes this run
        #      about 2x faster.
        #   3) The call to where here ensures that we plug in the "special" flag value
        #      where flags are detected.
        #   4) pairs of complex64 -> float32 via view, then float32 -> int16 via
        #      astype was the fastest way to do the required rounding.
        raw_dict = {
            sphid: {
                "scale_fac": sfac,
                "data": np.where(
                    sp_vis["flags"],
                    np.complex64(-32768 - 32768j),
                    sp_vis["data"] * (np.float32(2) ** (-sfac)),
                )
                .view(dtype=np.float32)
                .astype(np.int16),
            }
            for sfac, (sphid, sp_vis) in zip(scale_fac, vis_dict.items())
        }

        return raw_dict

    def _read_data(self, data_type, return_vis=True, use_mmap=True, read_only=False):
        """
        Read "sch_read" mir file into a list of ndarrays.

        Parameters
        ----------
        data_type : str
            Type of data to read, must either be "cross" (cross-correlations) or "auto"
            (auto-correlations).
        return_vis : bool
            If set to True, will return a dictionary containing the visibilities read
            in the "normal" format. If set to False, will return a dictionary containing
            the visibilities read in the "raw" format. Default is True.
        use_mmap : bool
            If False, then each integration record needs to be read in before it can
            be parsed on a per-spectral record basis (which can be slow if only reading
            a small subset of the data). Default is True, which will leverage mmap to
            access data on disk (that does not require reading in the whole record).
            There is usually no performance penalty to doing this, although reading in
            data is slow, you may try seeing this to False and seeing if performance
            improves.
        read_only : bool
            Only applicable if `return_vis=False` and `use_mmap=True`. If set to True,
            will return back data arrays which are read-only. Default is False.

        Returns
        -------
        data_dict : dict
            A dictionary, whose the keys are matched to individual values of sphid in
            `sp_data`, and each entry contains a dict with two items. If
            `return_vis=False` then a "raw data" dict is passed, with keys "scale_fac",
            an np.int16 which describes the common exponent for the spectrum, and
            "data", an array of np.int16 (of length equal to twice that found in
            `sp_data["nch"]` for the corresponding value of sphid) containing the
            compressed visibilities.  Note that entries equal to -32768 aren't possible
            with the compression scheme used for MIR, and so this value is used to mark
            flags. If `return_vis=True`, then a "vis data" dict is passed, with keys
            "data", an array of np.complex64 containing the visibilities, and
            "flags", an array of bool containing the per-channel flags of the
            spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).
        """
        if data_type not in ["auto", "cross"]:
            raise ValueError(
                'Argument for data_type not recognized, must be "cross" or "auto".'
            )

        is_cross = data_type == "cross"
        if is_cross:
            chavg_call = partial(self._rechunk_raw, inplace=True, return_vis=return_vis)
            data_map = self._sp_dict
            data_metadata = "sp_data"
            val_type = "<i2"
        else:
            chavg_call = partial(self._rechunk_data, inplace=True)
            data_map = self._ac_dict
            data_metadata = "ac_data"
            val_type = "<f4"
            return_vis = False

        group_dict = getattr(self, data_metadata).group_by("inhid")
        unique_inhid = list(group_dict)

        try:
            # Begin the process of reading the data in, stuffing the "packdata" arrays
            # (to be converted into "raw" data) into the dict below.
            packdata_dict = self._read_packdata(
                self._file_dict, unique_inhid, data_type, use_mmap
            )
        except MirMetaError:
            # Catch an error that indicates that the metadata inside the vis file does
            # not match that in _file_dict, and attempt to fix the problem.
            warnings.warn(
                "Values in int_dict do not match that recorded inside the "
                "file for %s data. Attempting to fix this automatically." % data_type
            )
            self._fix_int_dict(data_type)
            packdata_dict = self._read_packdata(
                self._file_dict, unique_inhid, data_type, use_mmap
            )

        # With the packdata in hand, start parsing the individual spectral records.
        data_dict = {}
        for inhid in unique_inhid:
            # Pop here let's us delete this at the end (and hopefully let garbage
            # collection do it's job correctly).
            packdata = packdata_dict.pop(inhid)["packdata"].view(val_type)
            hid_subarr = group_dict[inhid]
            dataoff_subdict = data_map[inhid]

            # We copy here if we want the raw values AND we've used memmap, since
            # otherwise the resultant entries in raw_data will be memmap arrays, which
            # will be read only (and we want attributes to be modifiable.)
            chan_avg_arr = np.zeros(len(hid_subarr), dtype=int)
            temp_dict = {}
            for idx, hid in enumerate(hid_subarr):
                dataoff = dataoff_subdict[hid]
                start_idx = dataoff["start_idx"]
                end_idx = dataoff["end_idx"]
                chan_avg_arr[idx] = dataoff["chan_avg"]

                if is_cross:
                    temp_dict[hid] = {
                        "scale_fac": packdata[start_idx],
                        "data": packdata[(start_idx + 1) : end_idx],
                    }
                else:
                    data_arr = packdata[start_idx:end_idx]
                    temp_dict[hid] = {"data": data_arr, "flags": np.isnan(data_arr)}

            if np.all(chan_avg_arr == 1):
                if return_vis:
                    temp_dict = self._convert_raw_to_vis(temp_dict)
                elif not read_only:
                    for idict in temp_dict.values():
                        idict["data"] = idict["data"].copy()
            else:
                chavg_call(temp_dict, chan_avg_arr)

            data_dict.update(temp_dict)

            # Do the del here to break the reference to the "old" data so that
            # subsequent assignments don't cause issues for raw_dict.
            del packdata

        # Figure out which results we need to pass back
        return data_dict

    def _write_cross_data(self, filepath, append_data=False, raise_err=True):
        """
        Write cross-correlation data to disk.

        Parameters
        ----------
        filepath : str
            String  describing the folder in which the data should be written.
        append_data : bool
            Option whether to append to an existing file, if one already exists, or to
            overwrite any existing data. Default is False (overwrite existing data).
        raise_err : bool
            If set to True and data are not loaded, raise a ValueError. If False, raise
            a warning instead. Default is True.

        Raises
        ------
        ValueError
            If cross data are not loaded, and `raise_err=True`.
        UserWarning
            If cross data are not loaded, and `raise_err=False`. Also raised if tsys
            corrections have been applied to the data prior to being written out.
        """
        if (self.raw_data is None) and (self.vis_data is None):
            if raise_err:
                raise ValueError("Cannot write data if not already loaded.")
            else:
                warnings.warn("No cross data loaded, skipping writing it to disk.")
                return

        if (self.vis_data is not None) and self._tsys_applied:
            # If using vis_data, we want to alert the user if tsys corrections are
            # applied, so that we mitigate the chance of a double-correction.
            if self._tsys_applied:
                warnings.warn(
                    "Writing out raw data with tsys applied. Be aware that you will "
                    "need to use set apply_tsys=True when calling load_data."
                    "Otherwise, call apply_tsys(invert=True) prior to writing out "
                    "the data set."
                )

        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # In order to write to disk, we need to create an intermediate product known
        # as "packed data", which is written on a per-integration basis. These create
        # duplicate copies of the data which can cause the memory footprint to balloon,
        # So we want to just create one packdata entry at a time. To do that, we
        # actually need to segment sp_data by the integration ID.
        int_dict, sp_dict = self.sp_data._generate_recpos_dict(reindex=True)

        # We can now open the file once, and write each array upon construction
        with open(
            os.path.join(filepath, "sch_read"), "ab+" if append_data else "wb+"
        ) as file:
            for inhid in int_dict:
                if self.vis_data is None:
                    raw_dict = self.raw_data
                else:
                    raw_dict = self._convert_vis_to_raw(
                        {sphid: self.vis_data[sphid] for sphid in sp_dict[inhid]}
                    )

                packdata = self._make_packdata(
                    {inhid: int_dict[inhid]}, {inhid: sp_dict[inhid]}, raw_dict, "cross"
                )
                packdata[inhid].tofile(file)

    def _write_auto_data(self, filepath, append_data=False, raise_err=True):
        """
        Write auto-correlation data to disk.

        Parameters
        ----------
        filepath : str
            String  describing the folder in which the data should be written.
        append_data : bool
            Option whether to append to an existing file, if one already exists, or to
            overwrite any existing data. Default is False (overwrite existing data).
        raise_err : bool
            If set to True and data are not loaded, raise a ValueError. If False, raise
            a warning instead. Default is True.

        Raises
        ------
        ValueError
            If auto data are not loaded, and `raise_err=True`.
        UserWarning
            If auto data are not loaded, and `raise_err=False`. Also raised if tsys
            corrections have been applied to the data prior to being written out.
        """
        if self.auto_data is None:
            if raise_err:
                raise ValueError("Cannot write data if not already loaded.")
            else:
                warnings.warn("No auto data loaded, skipping writing it to disk.")
                return

        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # In order to write to disk, we need to create an intermediate product known
        # as "packed data", which is written on a per-integration basis. These create
        # duplicate copies of the data which can cause the memory footprint to balloon,
        # So we want to just create one packdata entry at a time. To do that, we
        # actually need to segment sp_data by the integration ID.
        int_dict, ac_dict = self.ac_data._generate_recpos_dict(reindex=True)

        # We can now open the file once, and write each array upon construction
        with open(
            os.path.join(filepath, "ach_read"), "ab+" if append_data else "wb+"
        ) as file:
            for inhid in int_dict:
                packdata = self._make_packdata(
                    {inhid: int_dict[inhid]},
                    {inhid: ac_dict[inhid]},
                    self.auto_data,
                    "auto",
                )
                packdata[inhid].tofile(file)

    def apply_tsys(self, invert=False, force=False):
        """
        Apply Tsys calibration to the visibilities.

        SMA MIR data are recorded as correlation coefficients. This allows one to apply
        system temperature information to the data to get values in units of Jy.

        Parameters
        ----------
        invert : bool
            If set to True, this will effectively undo the Tsys correction that has
            been applied. Default is False (convert uncalibrated visibilities to units
            of Jy).
        force : bool
            Normally the method will check if tsys has already been applied (or not
            applied yet, if `invert=True`), and will throw an error if that is the case.
            If set to True, this check will be bypassed. Default is False.
        """
        if self.vis_data is None:
            raise ValueError(
                "Must call load_data first before applying tsys normalization."
            )

        if (self._tsys_applied and not invert) and (not force):
            raise ValueError(
                "Cannot apply tsys again if it has been applied already. Run "
                "apply_tsys with invert=True to undo the prior correction first."
            )

        if (not self._tsys_applied and invert) and (not force):
            raise ValueError(
                "Cannot undo tsys application if it was never applied. Set "
                "invert=True to apply the correction first."
            )

        # Create a dictionary here to map antenna pair + integration time step with
        # a sqrt(tsys) value. Note that the last index here is the receiver number,
        # which technically has a different keyword under which the system temperatures
        # are stored.
        tsys_dict = {
            (idx, jdx, 0): tsys**0.5
            for idx, jdx, tsys in zip(
                self.eng_data["inhid"],
                self.eng_data["antenna"],
                self.eng_data["tsys"],
            )
        }
        tsys_dict.update(
            {
                (idx, jdx, 1): tsys**0.5
                for idx, jdx, tsys in zip(
                    self.eng_data["inhid"],
                    self.eng_data["antenna"],
                    self.eng_data["tsys_rx2"],
                )
            }
        )

        # now create a per-blhid SEFD dictionary based on antenna pair, integration
        # time step, and receiver pairing.
        normal_dict = {}
        for blhid, idx, jdx, kdx, ldx, mdx in zip(
            self.bl_data["blhid"],
            self.bl_data["inhid"],
            self.bl_data["iant1"],
            self.bl_data["ant1rx"],
            self.bl_data["iant2"],
            self.bl_data["ant2rx"],
        ):
            try:
                normal_dict[blhid] = (2.0 * self.jypk) * (
                    tsys_dict[(idx, jdx, kdx)] * tsys_dict[(idx, ldx, mdx)]
                )
            except KeyError:
                warnings.warn(
                    "No tsys for blhid %i found (%i-%i baseline, inhid %i). "
                    "Baseline record will be flagged."
                    % (
                        blhid,
                        jdx,
                        ldx,
                        idx,
                    )
                )

        if invert:
            for key, value in normal_dict.items():
                normal_dict[key] = 1.0 / value

        # Finally, multiply the individual spectral records by the SEFD values
        # that are in the dictionary.
        for sphid, blhid in zip(self.sp_data["sphid"], self.sp_data["blhid"]):
            try:
                self.vis_data[sphid]["data"] *= normal_dict[blhid]
            except KeyError:
                self.vis_data[sphid]["flags"][:] = True

        self._tsys_applied = not invert

    def apply_flags(self):
        """
        Apply online flags to the visibilities.

        Applies flagging as recorded by the online system, which are applied on
        a per-spectral record basis. Users should be aware that this method will only
        modify data in vis_data, and not the "raw" values stored in raw_data.

        Raises
        ------
        ValueError
            If vis_data are not loaded.
        """
        if self.vis_data is None:
            raise ValueError("Cannot apply flags if vis_data are not loaded.")

        for sphid, flagval in zip(self.sp_data["sphid"], self.sp_data["flags"]):
            self.vis_data[sphid]["flags"][:] = bool(flagval)

    def _check_data_index(self):
        """
        Check that data attribute indexes match metadata.

        This is a simple check to make sure that index values stored in the attribute
        sp_data (namely "sphid") and ac_data (if loaded; "achid") provide a one-to-one
        match with the keys of raw_data/vis_data and auto_data, respectively.

        Returns
        -------
        check : bool
            If True, the index values all match up with keys in data data attributes.
        """
        # Set this list up here to make the code that follows a bit more generic (so
        # that we can have 1 loop rather than 3 if statements).
        check_list = [
            (self.sp_data["sphid"], self.vis_data),
            (self.sp_data["sphid"], self.raw_data),
        ]

        if self._has_auto:
            check_list.append((self.ac_data["achid"], self.auto_data))

        # Run our check
        for (idx_arr, data_arr) in check_list:
            if data_arr is None:
                # If not loaded, move along
                continue
            if sorted(idx_arr) != sorted(data_arr):
                # If we have a mismatch, we can leave ASAP
                return False

        # If you got to this point, it means that we've got agreement!
        return True

    def _downselect_data(self, select_vis=None, select_raw=None, select_auto=None):
        """
        Downselect data attributes based on metadata..

        This method will set entries in the data attributes (e.g., `vis_data`,
        `raw_data`, and `auto_data`) based on metadata header values present
        in `sp_data` ("sphid"; for vis/raw data) and `ac_data` ("achid"; for autos).
        It is meant to be an alternative to running `load_data`, in situation when
        the desired data have already been loaded from disk.

        Parameters
        ----------
        select_vis : bool
            If True, modify `vis_data` to contain only records where the key is matched
            to a value of "sphid" in `sp_data`. Default is True if data are loaded,
            otherwise False.
        select_raw : bool
            If True, modify `raw_data` to contain only records where the key is matched
            to a value of "sphid" in `sp_data`. Default is True if data are loaded,
            otherwise False.
        select_auto : bool
            If True, modify `auto_data` to contain only records where the key is matched
            to a value of "achid" in `ac_data`. Default is True if data are loaded,
            otherwise False.

        Raises
        ------
        KeyError
            If a spectral record header ID (either "schid" or "achid") does not have
            a corresponding key in the relevant data attribute, indicating that there
            are records requested that are not loaded into memory.
        """
        select_vis = (self.vis_data is not None) if (select_vis is None) else select_vis
        select_raw = (self.raw_data is not None) if (select_raw is None) else select_raw
        select_auto = (
            (self.auto_data is not None) if (select_auto is None) else select_auto
        )

        try:
            # Check that vis_data has all entries we need for processing the data
            if select_vis:
                vis_data = {
                    sphid: self.vis_data[sphid] for sphid in self.sp_data["sphid"]
                }

            # Now check raw_data
            if select_raw:
                raw_data = {
                    sphid: self.raw_data[sphid] for sphid in self.sp_data["sphid"]
                }

            # Now check auto_data
            if select_auto:
                auto_data = {
                    achid: self.auto_data[achid] for achid in self.ac_data["achid"]
                }
        except (TypeError, KeyError) as err:
            raise MirMetaError(
                "Missing spectral records in data attributes. Run load_data instead."
            ) from err

        # At this point, we can actually plug our values in, since we know that the
        # operation above succeeded.
        if select_vis and self.vis_data is not None:
            self.vis_data = vis_data
        if select_raw and self.raw_data is not None:
            self.raw_data = raw_data
        if select_auto and self.auto_data is not None:
            self.auto_data = auto_data

    def load_data(
        self,
        load_auto=None,
        load_cross=None,
        load_raw=False,
        apply_tsys=True,
        allow_downselect=None,
        allow_conversion=None,
        use_mmap=True,
        read_only=False,
    ):
        """
        Load visibility data into MirParser class.

        Method for loading the visibility data into the object. Can either load the
        compressed "raw data" (which is about half the size in memory, but carries
        overheads in various function calls _except_ for writing to disk) or the
        uncompressed "vis data" (floating point values).

        Note that this function will only load either  vis data or raw data, not
        both. If told to load both, then the method will default to loading only the
        vis data.

        Parameters
        ----------
        load_auto: bool
            Load the autos (floats) into object. Default is True if the object has
            auto-correlation data, otherwise False.
        load_cross : bool
            Load the cross-correlation visibility data (floats) into object. Default is
            True if if the object has cross-correlation data, otherwise False.
        load_raw : bool
            Nominally when loading cross-correlation data, the data are uncompressed
            when being loaded into the object, and put into the attribute `vis_data`.
            However, if set to True, the "raw" (compressed) data are put into the
            attribute `raw_data`. Default is False.
        apply_tsys : bool
            Apply tsys corrections to the data. Only applicable if loading
            cross-correlation vis data (where `load_raw=False`). Default is True.
        allow_downselect : bool
            If data has been previously loaded, and all spectral records are currently
            contained in `vis_data`, `raw_data`, and/or `auto_data`, then down-select
            from the currently loaded data rather than reading the data from disk.
            Default is True if all spectral records have been previously loaded,
            otherwise False.
        allow_conversion : bool
            Allow the method to convert previously loaded uncompressed ("raw") data into
            "normal" data. Only applicable if loading cross-correlation data (and
            `load_raw=False`). Default is True if all of the required spectral records
            have been loaded into the `raw_data` attribute.
        use_mmap : bool
            If False, then each integration record needs to be read in before it can
            be parsed on a per-spectral record basis (which can be slow if only reading
            a small subset of the data). Default is True, which will leverage mmap to
            access data on disk (that does not require reading in the whole record).
            There is usually no performance penalty to doing this, although reading in
            data is slow, you may try seeing this to False and seeing if performance
            improves.
        read_only : bool
            Only applicable if `load_cross=True`, `use_mmap=True`, and `load_raw=True`.
            If set to True, will return back data arrays which are read-only. Default is
            False.

        Raises
        ------
        UserWarning
            If attempting to set both `load_vis` and `load_raw` to True. Also if the
            method is about to attempt to convert previously loaded data.
        """
        # Figure out what exactly we're going to load here.
        if load_cross is None:
            load_cross = self._has_cross
        if load_auto is None:
            load_auto = self._has_auto

        # If there is no auto data to actually load, raise an error now.
        if load_auto and not self._has_auto:
            raise ValueError("This object has no auto-correlation data to load.")

        load_vis = load_cross and not load_raw
        load_raw = load_cross and load_raw

        # Last chance before we load data -- see if we already have raw_data in hand,
        # and just need to convert it. If allow_conversion is None, we should decide
        # first whether or not to attempt this.
        if allow_conversion or (allow_conversion is None):
            if load_vis and (self.raw_data is not None):
                allow_conversion = np.all(
                    np.isin(self.sp_data["sphid"], list(self.raw_data))
                )
            else:
                allow_conversion = False

        if allow_conversion:
            warnings.warn(
                "Converting previously loaded data since allow_conversion=True."
            )
            self.vis_data = self._convert_raw_to_vis(self.raw_data)
            self._tsys_applied = False
            # If we need to apply tsys, do that now.
            if apply_tsys:
                self.apply_tsys()

        # Unload anything that we don't want to load at this point.
        self.unload_data(
            unload_vis=not load_vis,
            unload_raw=not load_raw,
            unload_auto=not load_auto,
        )

        # If we are potentially downselecting data (typically used when calling select),
        # make sure that we actually have all the data we need loaded.
        if allow_downselect or (allow_downselect is None):
            if load_cross:
                try:
                    self._downselect_data(
                        select_vis=load_vis, select_raw=load_raw, select_auto=False
                    )
                    load_cross = False
                except MirMetaError:
                    if allow_downselect:
                        warnings.warn("Cannot downselect cross-correlation data.")

            if load_auto:
                try:
                    self._downselect_data(
                        select_vis=False, select_raw=False, select_auto=True
                    )
                    load_auto = False
                except MirMetaError:
                    if allow_downselect:
                        warnings.warn("Cannot downselect auto-correlation data.")

        # Finally, if we didn't downselect or convert, load the data from disk now.
        if load_cross:
            data_dict = self._read_data(
                "cross",
                return_vis=load_vis,
                use_mmap=use_mmap,
                read_only=read_only,
            )

            setattr(self, "vis_data" if load_vis else "raw_data", data_dict)

            if load_vis:
                # Since we've loaded in "fresh" data, we mark that tsys has
                # not yet been applied (otherwise apply_tsys can thrown an error).
                self._tsys_applied = False

                # Apply tsys if needed.
                if apply_tsys and load_vis:
                    self.apply_tsys()

        # We wrap the auto data here in a somewhat special way because of some issues
        # with the existing online code and how it writes out data. At some point
        # we will fix this, but for now, we triage the autos here. Note that if we
        # already have the auto_data loaded, we can bypass this step.
        if load_auto:
            self.auto_data = self._read_data(
                "auto", return_vis=False, use_mmap=use_mmap, read_only=read_only
            )

    def unload_data(self, unload_vis=True, unload_raw=True, unload_auto=True):
        """
        Unload data from the MirParser object.

        Unloads the data-related attributes from memory, if they are loaded. Because
        these attributes can be formidable in size, this operation will substantially
        reduce the memory footprint of the MirParser object.

        Parameters
        ----------
        unload_vis : bool
            Unload the visibilities stored in the `vis_data` attribute, if loaded.
            Default is True.
        unload_raw : bool
            Unload the raw visibilities stored in the `raw_data` attribute, if loaded.
            Default is True.
        unload_auto : bool
            Unload the auto-correlations stored in the `auto_data` attribute, if loaded.
            Default is True.
        """
        if unload_vis:
            self.vis_data = None
            self._tsys_applied = False
        if unload_raw:
            self.raw_data = None
        if unload_auto:
            self.auto_data = None

    def _update_filter(self, update_data=None):
        """
        Update MirClass internal filters for the data.

        Note that this is an internal helper function which is not for general user use,
        but instead an internal helper function for the MirParser object. Updates
        the masks of the various MirMetaData objects so that only records with entries
        across _all_ the metadata objects are included. Typically used after issuing
        a `select` command to propagate masks to objects that did not immediately have
        matching selection criteria based on what the user provided.

        update_data : bool
            If set to True, will read in data from disk after selecting records. If
            set to False, data attributes (e.g., `vis_data`, `raw_data`, `auto_data`)
            will be unloaded. If set to True, data attributes will be reloaded, based
            on what had been previously.  Default is to downselect the data from that
            previously unloaded if possible, otherwise unload the data.
        """
        mask_update = False
        # Start by cascading the filters up -- from largest metadata tables to the
        # smallest. First up, spec win -> baseline
        if not np.all(self.sp_data.get_mask()):
            mask_update |= self.bl_data.set_mask(header_key=set(self.sp_data["blhid"]))

        # Now do baseline -> antennas. Special handling required because of the
        # lack of a unique index key for this table.
        if not (np.all(self.bl_data.get_mask()) and not self._has_auto):
            key_list = set(
                self.bl_data.get_value(["iant1", "inhid"], return_tuples=True)
                + self.bl_data.get_value(["iant2", "inhid"], return_tuples=True)
            )

            if self._has_auto:
                key_list.union(
                    self.ac_data.get_value(["antenna", "inhid"], return_tuples=True)
                )

            mask_update |= self.eng_data.set_mask(header_key=key_list)

        # Now antennas -> int
        if not np.all(self.eng_data.get_mask()):
            mask_update |= self.in_data.set_mask(header_key=set(self.eng_data["inhid"]))

        # And weather scan -> int
        if not np.all(self.we_data.get_mask()):
            mask_update |= self.in_data.set_mask(
                where=("ints", "eq", self.we_data["ints"])
            )

        # We now cascade the masks downward. First up, int -> weather scan
        mask_update |= self.we_data.set_mask(header_key=self.in_data["ints"])

        # Next, ant -> baseline. Again this requires a little extra special
        # handling, since eng_data doesn't have a unique header key.
        bl_eng_mask = np.logical_and(
            self.eng_data.get_mask(
                header_key=self.bl_data.get_value(
                    ["iant1", "inhid"], return_tuples=True, use_mask=False
                )
            ),
            self.eng_data.get_mask(
                header_key=self.bl_data.get_value(
                    ["iant2", "inhid"], return_tuples=True, use_mask=False
                )
            ),
        )
        mask_update |= self.bl_data.set_mask(mask=bl_eng_mask)

        # Next, do int -> baseline
        mask_update |= self.bl_data.set_mask(
            where=("inhid", "eq", self.in_data["inhid"])
        )

        # Finally, do baseline -> spec win for the crosses...
        mask_update |= self.sp_data.set_mask(
            where=("blhid", "eq", self.bl_data["blhid"])
        )
        # ...and the autos.
        if self._has_auto:
            mask_update |= self.ac_data.set_mask(
                mask=self.eng_data.get_mask(
                    header_key=self.ac_data.get_value(
                        ["antenna", "inhid"],
                        return_tuples=True,
                        use_mask=False,
                    )
                ),
            )

        if update_data or (update_data is None):
            try:
                self._downselect_data()
            except MirMetaError:
                if not update_data:
                    self.unload_data()
                    warnings.warn(
                        "Unable to update data attributes, unloading them now."
                    )
                    return
                self.load_data(
                    load_cross=not (self.vis_data is None and self.raw_data is None),
                    load_raw=self.raw_data is not None,
                    load_auto=self.auto_data is not None,
                    apply_tsys=self._tsys_applied,
                    allow_downselect=False,
                    allow_conversion=False,
                )

    def _clear_auto(self):
        """
        Remove attributes related to autos.

        This method is an internal helper function, and not meant for general users. It
        will clear out attributes related to the auto-correlations.
        """
        self._has_auto = False
        self.auto_data = None
        self.ac_data = MirAcData()
        self._ac_dict = None
        try:
            del self._metadata_attrs["ac_data"]
            for key in self._file_dict:
                del self._file_dict[key]["auto"]
        except KeyError:
            pass

    def reset(self):
        """
        Reset a MirParser object to its original state.

        This method will in effect revert the object to a "pristine" state. Visibility
        data will be unloaded, changed metadata will be restored, and any rechunking
        settings will be removed (so that data will be loaded at full spectral
        resolution).
        """
        for item in self._metadata_attrs:
            self._metadata_attrs[item].reset()

        update_list = []
        if self._has_cross:
            update_list.append(self._sp_dict)
        if self._has_auto:
            update_list.append(self._ac_dict)

        for recpos_dict in update_list:
            for int_dict in recpos_dict.values():
                for idict in int_dict.values():
                    idict["chan_avg"] = 1

        self.unload_data()

    def _fix_acdata(self):
        """
        Fill in missing auto-correlation metadata.

        This method is an internal helper function, and not meant to be called by users.
        Its purpose is to reconstruct auto-correlation metadata based on that available
        in other metadata attributes. This is needed because presently, the online
        system records no such information.
        """
        # First up, we want to down-select any extra records belonging to correlator
        # chunks that are completely blank.
        unique_bands = np.unique(self.sp_data._data["iband"])
        sel_mask = np.isin(self.ac_data._data["iband"], unique_bands)

        self.ac_data._data = self.ac_data._data[sel_mask]
        self.ac_data._mask = np.ones(len(self.ac_data), dtype=bool)

        # Set up the header index for the object, and then construct the header key dict
        self.ac_data._data["achid"] = np.arange(1, len(self.ac_data) + 1)
        self.ac_data._set_header_key_index_dict()

        # Now that we've got vitals, we want to import in metadata from some of the
        # other metadata objects. List out those fields now.
        bl_imports = ["irec", "ipol"]
        sp_imports = ["fsky", "gunnLO", "corrLO1", "corrLO2", "fDDS", "fres"]

        # Handle the special case that is the ipol values -- a common error is that they
        # are all set to 0 when operating in dual-pol mode (when they should be 0 & 1).
        if len(np.unique(self.bl_data["ipol"])) == 1 and (
            len(self.codes_data["pol"]) == (2 * 4)
        ):
            bl_imports.remove("ipol")
            self.ac_data["ipol"] = self.ac_data["antrx"]

        # First, handle the cases where all values should be the same on a per-inhid
        # basis, but pulled from bl_data.
        ac_groups = self.ac_data.group_by(
            ["inhid", "antrx", "antrx", "isb"], return_index=True, use_mask=False
        )
        bl_groups = self.bl_data.group_by(
            ["inhid", "ant1rx", "ant2rx", "isb"], return_index=True, use_mask=False
        )

        # For each field, plug in the entries on a per-group basis. Note we use median
        # here to try to add some robustness (in case of recording error in one record).
        for field in bl_imports:
            data_arr = self.ac_data._data[field]
            export_data_arr = self.bl_data._data[field]
            for key, idx_arr in ac_groups.items():
                data_arr[idx_arr] = np.median(export_data_arr[bl_groups[key]])

        # Now get set up for tackling the items that can change on a per spectral
        # window basis. Note that here we are assuming that the values should be the
        # same for at least the same inhid and spectral band number.
        ac_groups = self.ac_data.group_by(
            ["inhid", "antrx", "isb", "iband"], return_index=True, use_mask=False
        )
        bl_groups = self.bl_data.group_by(
            ["inhid", "ant1rx", "ant2rx", "isb"], use_mask=False
        )
        sp_groups = self.sp_data.group_by(
            ["blhid", "iband"],
            use_mask=False,
            assume_unique=True,
            return_index=True,
        )

        # We have to do a bit of extra legwork here to figure out which blocks o
        # entries map to which values.
        metablock_dict = {}
        for (inhid, ant1rx, ant2rx, isb), blhid_arr in bl_groups.items():
            # Ignore cross-hand pol, since we don't have that for autos.
            if ant1rx == ant2rx:
                for iband in unique_bands:
                    try:
                        ac_idx = ac_groups[(inhid, ant1rx, isb, iband)]
                    except KeyError:
                        # If we have a key error, it means there are no entries with
                        # this receiver, sideband, band at this inhid, so just skip it.
                        continue
                    # Pull together all the sphid entries that are in this group, then
                    # add it to a list for handling later.
                    sp_idx = [sp_groups[(blhid, iband)] for blhid in blhid_arr]
                    try:
                        metablock_dict[(ant1rx, isb, iband)]["ac_idx"].append(ac_idx)
                        metablock_dict[(ant1rx, isb, iband)]["sp_idx"].append(sp_idx)
                    except KeyError:
                        metablock_dict[(ant1rx, isb, iband)] = {
                            "ac_idx": [ac_idx],
                            "sp_idx": [sp_idx],
                        }

        # Now get into plugging in values.
        for field in sp_imports:
            for idx_groups in metablock_dict.values():
                # For several values, they will not change over the course of the obs.
                # If that's the case, then it is _much_ faster to find the one value
                # and plug it in once.
                val_arr = self.sp_data.get_value(
                    field, index=np.concatenate(idx_groups["sp_idx"])
                )
                median_val = np.median(val_arr)
                if np.allclose(val_arr, median_val):
                    self.ac_data.set_value(
                        field, median_val, index=np.concatenate(idx_groups["ac_idx"])
                    )
                    continue

                # Otherwise, if the value varies, then handle it on a per-inhid basis.
                for ac_idx, sp_idx in zip(idx_groups["ac_idx"], idx_groups["sp_idx"]):
                    self.ac_data.set_value(
                        field,
                        np.median(self.sp_data.get_value(field, index=sp_idx)),
                        index=ac_idx,
                    )

        # Finally, clear out any stored values we may have accumulated from the above
        # operations, since we don't want reset to clear them out.
        self.ac_data._stored_values = {}

    def read(
        self,
        filepath,
        has_auto=False,
        has_cross=True,
        load_auto=False,
        load_cross=False,
        load_raw=False,
    ):
        """
        Read in all files from a mir data set into predefined numpy datatypes.

        The full dataset can be quite large, as such the default behavior of
        this function is only to load the metadata. Use the keyword params to
        load other data into memory.

        Parameters
        ----------
        filepath : str
            Filepath is the path to the folder containing the Mir data set.
        has_auto : bool
            Flag to read auto-correlation data. Default is False.
        has_cross : bool
            Flag to read cross-correlation data. Default is True.
        load_auto : bool
            Flag to load auto-correlations into memory. Default is False.
        load_cross : bool
            Flag to load cross-correlations into memory. Default is False.
        load_raw : bool
            Flag to load raw data into memory. Default is False.
        """
        # These functions will read in the major blocks of metadata that get plugged
        # in to the various attributes of the MirParser object. Note that "_read"
        # objects contain the whole data set, while "_data" contains that after
        # filtering (more on that below).

        if has_auto:
            self._metadata_attrs["ac_data"] = self.ac_data
            self._has_auto = True
        else:
            self._clear_auto()
            load_auto = False

        if has_cross:
            self._has_cross = True

        filepath = os.path.abspath(filepath)

        for attr in self._metadata_attrs.values():
            attr.read(filepath)

        # This indexes the "main" file that contains all the visibilities, to make
        # it faster to read in the data.
        file_dict = {}
        if self._has_cross:
            int_dict, self._sp_dict = self.sp_data._generate_recpos_dict()
            file_dict["cross"] = {
                "int_dict": int_dict,
                "filetype": "sch_read",
                "ignore_header": False,
            }

        if self._has_auto:
            filetype = "ach_read"
            old_fmt = self.ac_data._old_fmt
            if old_fmt:
                # If we have the old-style file we are working with, then we need to
                # do two things: first, clean up entries that don't actually have any
                # data in them (the old format recorded lots of blank data to disk),
                # and plug in some missing metadata.
                self._fix_acdata()
                filetype = "autoCorrelations"
            int_dict, self._ac_dict = self.ac_data._generate_recpos_dict()

            file_dict["auto"] = {
                "int_dict": self.ac_data._old_fmt_int_dict if old_fmt else int_dict,
                "filetype": filetype,
                "ignore_header": old_fmt,
            }

        self._file_dict = {filepath: file_dict}
        self.filepath = filepath

        # Set/clear these to start
        self.vis_data = self.raw_data = self.auto_data = None
        self._tsys_applied = False

        # If requested, now we load up the visibilities.
        self.load_data(load_cross=load_cross, load_raw=load_raw, load_auto=load_auto)

    def write(
        self,
        filepath,
        overwrite=True,
        load_data=False,
        append_data=False,
        check_index=True,
    ):
        """
        Write a MirParser object to disk in Mir format.

        Writes out a MirParser object to disk, in the binary Mir format. This method
        can worth with either a full dataset, or partial datasets appended together
        multiple times.

        Parameters
        ----------
        filepath : str
            Path of the directory to write out the data set.
            found in `filepath`. If no previously created data set exists, then a new
            data set is created on disk, and this parameter is ignored. Default is
            False.
        overwrite : bool
            If set to True, any previously written data in `filepath` will be
            overwritten. Default is False. This argument is ignored if `append_data` is
            set to True.
        load_data : bool
            If set to True, load the raw visibility data. Default is False, which will
            forgo loading data. Note that if no data are loaded, then the method
            will then write out a metadata-only object.
        append_data : bool
            If set to True, this will allow the method to append data to an existing
            file on disk. If no such file exists in `filepath`, then a new file is
            created (i.e., no appends are performed).
        check_index : bool
            Only applicable if `append_data=True`. If set to True and data are being
            appended to an existing file, the method will check to make sure that there
            are no header key conflicts with the data being being written to disk, since
            this can cause corrupted the metadata. Default is True, users should
            use this argument with caution, since it can cause the data set on disk
            to become unusable.

        Raises
        ------
        UserWarning
            If only metadata is loaded in the MirParser object.
        FileExistsError
            If a file already exists and cannot append or overwrite.
        ValueError
            If attempting to append data, but conflicting header keys are detected
            between the data on disk and the data in the object.
        """
        # If no directory exists, create one to write the data to
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # Check that the data are loaded
        if load_data:
            self.load_data(load_raw=True)

        # Write out the various metadata fields
        for attr in self._metadata_attrs:
            if attr in ["sp_data", "ac_data"]:
                mir_meta_obj = self._metadata_attrs[attr].copy()
                mir_meta_obj._recalc_dataoff()
            else:
                mir_meta_obj = self._metadata_attrs[attr]

            mir_meta_obj.write(
                filepath,
                overwrite=overwrite,
                append_data=append_data,
                check_index=check_index,
            )

        # Finally, we can package up the data in order to write it to disk.
        if self._has_cross:
            self._write_cross_data(filepath, append_data=append_data, raise_err=False)

        if self._has_auto:
            self._write_auto_data(filepath, append_data=append_data, raise_err=False)

    @staticmethod
    def _rechunk_data(data_dict, chan_avg_arr, inplace=False):
        """
        Rechunk regular cross- and auto-correlation spectra.

        Note this routine is not intended to be called by users, but instead is a
        low-level call from the `rechunk` method of MirParser to spectrally average
        data.

        Parameters
        ----------
        data_dict : dict
            A dict containing auto or cross data, where the keys match to values of
            of "sphid" in `sp_data` for cross, or "achid" in `ac_data` for autos, with
            each value being its own dict, with keys "data" (dtype=np.complex64 for
            cross, dtype=np.float32 for auto) and "flags" (the flagging information,
            dtype=bool).
        chan_avg_arr : sequence of int
            A list, array, or tuple of integers, specifying how many channels to
            average over within each spectral record.
        inplace : bool
            If True, entries in `vis_dict` will be updated with spectrally averaged
            data. If False (default), then the method will construct a new dict that
            will contain the spectrally averaged data.

        Returns
        -------
        new_vis_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `vis_dict`.
        """
        if data_dict is None:
            return

        new_data_dict = data_dict if inplace else {}

        for chan_avg, (hkey, sp_data) in zip(chan_avg_arr, data_dict.items()):
            # If there isn't anything to average, we can skip the heavy lifting
            # and just proceed on to the next record.
            if chan_avg == 1:
                if not inplace:
                    new_data_dict[hkey] = copy.deepcopy(sp_data)
                continue

            # Otherwise, we need to first get a handle on which data is "good"
            # for spectrally averaging over.
            good_mask = ~sp_data["flags"].reshape((-1, chan_avg))

            # We need to count the number of valid visibilities that goes into each
            # new channel, so that we can normalize appropriately later. Note we cast
            # to float32 here, since the data are complex64 (and so there's no extra
            # casting required, but we get the benefit of only multiplying real-only
            # and complex data).
            temp_count = good_mask.sum(axis=-1, dtype=np.float32)

            # Need to mask out when we have no counts, since it'll produce a divide
            # by zero error. As an added bonus, this will let us zero out any channels
            # without any valid visibilities.
            temp_count = np.reciprocal(
                temp_count, where=(temp_count != 0), out=temp_count
            )

            # Now take the sum of all valid visibilities, multiplied by the
            # normalization factor.
            temp_vis = (
                sp_data["data"].reshape((-1, chan_avg)).sum(where=good_mask, axis=-1)
                * temp_count
            )

            # Finally, plug the spectrally averaged data back into the dict, flagging
            # channels with no valid data.
            new_data_dict[hkey] = {"data": temp_vis, "flags": temp_count == 0}

        return new_data_dict

    @staticmethod
    def _rechunk_raw(raw_dict, chan_avg_arr, inplace=False, return_vis=False):
        """
        Rechunk a raw visibility spectrum.

        Note this routine is not intended to be called by users, but instead is a
        low-level call from the `rechunk` method of MirParser to spectrally average
        data.

        Parameters
        ----------
        raw_dict : dict
            A dict containing raw visibility data, where the keys match to individual
            values of "sphid" in `sp_data`, with each value being its own dict, with
            keys "data" (the raw visibility data, dtype=np.int16) and "scale_fac"
            (scale factor to multiply raw data by , dtype=np.int16).
        chan_avg_arr : sequence of int
            A list, array, or tuple of integers, specifying how many channels to
            average over within each spectral record.
        inplace : bool
            If True, entries in `raw_dict` will be updated with spectrally averaged
            data. If False (default), then the method will construct a new dict that
            will contain the spectrally averaged data.
        return_vis : bool
            If True, return data in the "normal" visibility format, where each
            spectral record has a key of "sphid" and a value being a dict of
            "data" (the visibility data, dtype=np.complex64) and "flags"
            (the flagging information, dtype=bool). This option is ignored if `inplace`
            is set to True.

        Returns
        -------
        data_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `raw_dict` (unless `return_vis=True`).
        """
        if raw_dict is None:
            return

        # If inplace, point our new dict to the old one, otherwise create
        # an empty dict to plug values into.
        data_dict = raw_dict if inplace else {}

        for chan_avg, (sphid, sp_raw) in zip(chan_avg_arr, raw_dict.items()):
            # If the number of channels to average is 1, then we just need to make
            # a deep copy of the old data and plug it in to the new dict.
            if chan_avg == 1:
                if (not inplace) or return_vis:
                    data_dict[sphid] = (
                        MirParser._convert_raw_to_vis({0: sp_raw})[0]
                        if return_vis
                        else copy.deepcopy(sp_raw)
                    )
                continue

            # If we are _not_ skipping the spectral averaging, then it turns out to
            # be faster to convert the raw data to "regular" data, spectrally average
            # it, and then convert it back to the raw format. Note that we set up a
            # "dummy" dict here with an sphid of 0 to make it easy to retrieve that
            # entry after the sequence of calls.
            if return_vis:
                data_dict[sphid] = MirParser._rechunk_data(
                    MirParser._convert_raw_to_vis({0: sp_raw}),
                    [chan_avg],
                    inplace=True,
                )[0]
            else:
                data_dict[sphid] = MirParser._convert_vis_to_raw(
                    MirParser._rechunk_data(
                        MirParser._convert_raw_to_vis({0: sp_raw}),
                        [chan_avg],
                        inplace=True,
                    )
                )[0]

        # Finally, return the dict containing the raw data.
        return data_dict

    def rechunk(self, chan_avg):
        """
        Rechunk a MirParser object.

        Spectrally average a Mir dataset. This command attempts to emulate the old
        "SMARechunker" program within the MirParser object. Users should be aware
        that running this operation modifies the metadata in such a way that all data
        loaded will be rechunked in the same manner, until you run the `reset` method.

        Note that this command will only process data from the "normal" spectral
        windows, and not the pseudo-continuum data (which will remain untouched).

        Parameters
        ----------
        chan_avg : int
            Number of contiguous spectral channels to average over.
        """
        # Start of by doing some argument checking.
        arg_dict = {"chan_avg": chan_avg}
        for key, value in arg_dict.items():
            if not (isinstance(value, int) or isinstance(value, np.int_)):
                raise ValueError("%s must be of type int." % key)
            elif value < 1:
                raise ValueError("%s cannot be a number less than one." % key)

        if chan_avg == 1:
            # This is a no-op, so we can actually bail at this point.
            return

        if not self._check_data_index():
            # If the above returns False, then we have a problem, and can't
            # actually run this operation (have to reload the data).
            raise ValueError(
                "Index values do not match data keys. Data will need to be "
                "reloaded before continuing using `select(reset=True)`."
            )

        # Eventually, we can make this handling more sophisticated, but for now, just
        # make it so that we average every window aside from the pseudo continuum
        # (win #0) by the same value.
        chanavg_dict = {}
        for band_name in self.codes_data.get_codes("band", return_dict=False):
            iband_value = self.codes_data["band"][band_name]
            chanavg_dict[iband_value] = 1 if "c" in band_name else chan_avg

        update_dict = {}
        if self._has_cross:
            update_dict["sp_data"] = [self._sp_dict, ["fres", "vres"], None, None]
        if self._has_auto:
            update_dict["ac_data"] = [self._ac_dict, ["fres"], None, None]

        for attr in update_dict:
            band_arr = getattr(self, attr).get_value("iband", use_mask=False)
            nch_arr = getattr(self, attr).get_value("nch", use_mask=False)
            chavg_arr = [chanavg_dict[band] for band in band_arr]
            if np.any(np.mod(nch_arr, chavg_arr) != 0):
                raise ValueError(
                    "chan_avg does not go evenly into the number of channels in each "
                    "spectral window (typically chan_avg should be a power of 2)."
                )
            update_dict[attr][2] = chavg_arr
            update_dict[attr][3] = nch_arr // chavg_arr

        for attr, (recpos_dict, update_list, chavg_arr, nch_arr) in update_dict.items():
            for item in update_list:
                getattr(self, attr).set_value(
                    item,
                    getattr(self, attr).get_value(item, use_mask=False) * chavg_arr,
                    use_mask=False,
                )
            getattr(self, attr).set_value("nch", nch_arr, use_mask=False)

            for int_dict in recpos_dict.values():
                for hid in int_dict:
                    int_dict[hid]["chan_avg"] *= chanavg_dict[
                        getattr(self, attr).get_value("iband", header_key=hid)
                    ]

            chan_avg_arr = [chanavg_dict[band] for band in getattr(self, attr)["iband"]]
            if attr == "sp_data":
                self._rechunk_data(self.vis_data, chan_avg_arr, inplace=True)
                self._rechunk_raw(self.raw_data, chan_avg_arr, inplace=True)
            else:
                self._rechunk_data(self.auto_data, chan_avg_arr, inplace=True)

    def __add__(self, other, merge=None, overwrite=None, force=False, inplace=False):
        """
        Add two MirParser objects.

        This method allows for combining MirParser objects under two different
        scenarios. In the first, which we call a "merge", two objects are instantiated
        from the same file, but may have different data loaded due to, for example,
        different calls to `select` being run. In the second scenario, which we call
        a "concatenation", objects are instantiated from different files, which need
        to be recombined (e.g., a single track is broken in half due to an
        intervening observation/system hiccup/etc.).

        Note that while some checking is performed to check if the metadata objects
        look identical, no checking is done on the `vis_data`, `raw_data`, and
        `auto_data` attributes (other than that they exist). If either object does not
        have data loaded, then the resultant object will also have no data loaded.

        Parameters
        ----------
        other : MirParser object
            Other MirParser object to combine with this data set.
        overwrite : bool
            If set to True, metadata from `other` will overwrite that present in
            this object, even if they differ. Default is False.
        merge : bool
            If set to True, assume that the objects originate from the amd file, and
            combine them together accordingly. If set to False, assume the two objects
            originate from _different_ files, and concatenate them together. By default,
            the method will check the internal file dictionary to see if it appears the
            two objects come from the same file(s), automatically choosing between
            merging or concatenating.
        force : bool
            If set to True, bypass certain checks to force the method to combine the
            two objects. Note that this option should be used with care, as it can
            result in objects with duplicate data (which may affect downstream
            processing), or loss of support for handling auto-correlations within
            the object. Default is False.
        inplace : bool
            If set to True, replace this object with the one resulting from the
            addition operation. Default is False.

        Returns
        -------
        new_obj : MirParser object
            A new object that contains the combined data of the two objects. Only
            returned if `inplace=False`.

        Raises
        ------
        TypeError
            If attempting to add a MirParser object with any other type of object.
        ValueError
            If the objects cannot be combined, either because of differing metadata (if
            `overwrite=False`), or because the two objects appear to be loaded from
            the same files but have different internal mappings (usually caused by the
            MirParser object being added to another object, prior to the addition).
            Also raised if metadata differ between objects and overwrite=False.
        UserWarning
            If duplicate metadata was found but force=True, or if identical timestamps
            were found between the two datasets when concatenating data.
        """
        if not isinstance(other, MirParser):
            raise TypeError(
                "Cannot add a MirParser object an object of a different type."
            )

        if (self._has_auto != other._has_auto) and not force:
            raise ValueError(
                "Cannot combine two MirParser objects if one has auto-correlation data "
                "and the other does not. You can bypass this error by setting "
                "force=True."
            )
        if (self.jypk != other.jypk) and not overwrite:
            raise ValueError(
                "Cannot combine objects where the jypk value is different. Set "
                "overwrite=True to bypass this error, which will use the value "
                "of the right object in the add sequence."
            )

        same_files = self._file_dict.keys() == other._file_dict.keys()
        if np.any([file in self._file_dict for file in other._file_dict]):
            if same_files:
                if not (merge or (merge is None)):
                    raise ValueError(
                        "Must set merge=True in order to combine objects created from "
                        "the same set of files."
                    )
            else:
                raise ValueError(
                    "These two objects were created from a partially overlapping "
                    "set of files. Cannot combine."
                )
        elif merge:
            raise ValueError(
                "Cannot merge objects that originate from different files, you must "
                "set merge=False."
            )

        if merge or (same_files and (merge is None)):
            # First up, check to see that the metadata appear identical, modulo data
            # that has been flagged/deselected.
            metadata_dict = self._metadata_attrs.copy()
            bad_attr = []
            for item in metadata_dict:
                try:
                    metadata_dict[item] = getattr(self, item).__add__(
                        getattr(other, item),
                        merge=merge,
                        overwrite=overwrite,
                    )
                except MirMetaError:
                    # MirMetaError is a unique error thrown when there are conflicting
                    # header keys that do not contain identical metadata.
                    bad_attr.append(item)

            # If we failed to add any objects, raise an error now.
            if len(bad_attr) != 0:
                raise ValueError(
                    "Cannot merge objects due to conflicts in %s. This can be bypassed "
                    "by setting overwrite=True, which will force the metadata from the "
                    "right-hand object in the add sequence to overwrite that from the "
                    "left." % bad_attr
                )

            # At this point, we know that we can merge the two objects, so begin the
            # heavy lifting of combining the two objects. Overwrite the new objects
            # with those from other wherever appropriate.
            new_obj = self if inplace else self.copy()
            new_obj.filepath = other.filepath
            new_obj.jypk = other.jypk

            new_obj._metadata_attrs = metadata_dict
            for item in metadata_dict:
                setattr(new_obj, item, metadata_dict[item])

            other_vis = other.vis_data
            other_raw = other.raw_data
            other_auto = other.auto_data

        else:
            # What if we are NOT going to merge the two files? Then we want to verify
            # that we actually have two unique datasets. We do that by checking the
            # metadata objects and checking for any matches.
            bad_attr = []
            update_dict = {}
            for item in self._metadata_attrs:
                this_attr = self._metadata_attrs[item]
                other_attr = other._metadata_attrs[item]
                if this_attr == other_attr:
                    bad_attr.append(item)
                    if not force:
                        continue
                update_dict.update(other_attr._generate_new_header_keys(this_attr))

            if "antpos_data" in bad_attr:
                bad_attr.remove("antpos_data")
            elif not overwrite:
                raise ValueError(
                    "Antenna positions differ between objects, cannot combine. You can "
                    "bypass this error by setting overwrite=True."
                )
            if len(bad_attr) != 0:
                if not force:
                    raise ValueError(
                        "Duplicate metadata found for the following attributes: "
                        "%s. You can bypass this error by setting force=True, though "
                        " be advised that doing so may result in duplicate data being "
                        "exported downstream." % ", ".join(bad_attr)
                    )
                warnings.warn(
                    "Duplicate metadata found for the following attributes: "
                    "%s. Proceeding anyways since force=True." % ", ".join(bad_attr)
                )

            # Final check - see if the MJD lines up exactly, since that _shouldn't_
            # happen if these are unique sets of data.
            if np.any(
                np.isin(
                    self.in_data.get_value("mjd", use_mask=False),
                    other.in_data.get_value("mjd", use_mask=False),
                )
            ):
                warnings.warn(
                    "These two objects contain data taken at the exact same time, "
                    "which could mean that combining the two will result in duplicate "
                    "data being potentially exported."
                )

            # If you have arrived here, you are at the point of no return. Start by
            # creating a copy of the other object, that we can make updates to.
            new_obj = self if inplace else self.copy()
            new_obj.jypk = other.jypk
            new_obj.filepath += ";" + other.filepath

            # Start combining the metadata
            for item in other._metadata_attrs:
                if (item == "ac_data") and (self._has_auto != other._has_auto):
                    # If we've reached this point, it's because force=True, so just
                    # skip setting this attribute.
                    continue
                # Make a copy of the metadata from other so that we can update the
                # individual fields. This will generally force the header key values
                # for the other object to come _after_ this object. This is useful in
                # case of sequential adds.
                attr = other._metadata_attrs[item].copy()
                attr._update_fields(update_dict)
                new_obj._metadata_attrs[item].__iadd__(attr, overwrite=overwrite)

            # The metadata is done, now we need to update the dicts that contain the
            # actual data itself, since their indexed to particular header keys. For
            # each attribute, we want to grab both the relevant dict AND the header
            # key field that is uses, so we know which update dict to use.
            other_vis = {} if (other.vis_data is None) else other.vis_data.copy()
            other_raw = {} if (other.raw_data is None) else other.raw_data.copy()
            other_auto = {} if (other.auto_data is None) else other.auto_data.copy()

            for hid, data_dict in zip(
                ["sphid", "sphid", "achid"], [other_vis, other_raw, other_auto]
            ):
                try:
                    key_dict = update_dict[hid]
                except KeyError:
                    continue

                data_dict.update(
                    {
                        key_dict[key]: data_dict.pop(key)
                        for key in set(data_dict).intersection(key_dict)
                    }
                )

            # From the primary update dict, grab the three that we need for indexing
            inhid_dict = update_dict.get("inhid", {})
            sphid_dict = update_dict.get("sphid", {})
            achid_dict = update_dict.get("achid", {})

            # Now deal with packdata integration dict
            for filename, file_dict in other._file_dict.items():
                new_obj._file_dict[filename] = {}
                for datatype, datatype_dict in file_dict.items():
                    new_obj._file_dict[filename][datatype] = {
                        "filetype": datatype_dict["filetype"],
                        "ignore_header": datatype_dict["ignore_header"],
                    }
                    new_obj._file_dict[filename][datatype]["int_dict"] = {
                        inhid_dict.get(inhid, inhid): idict.copy()
                        for inhid, idict in datatype_dict["int_dict"].items()
                    }

            # Finally, deal with the recpos_dicts
            recpos_list = []
            if new_obj._has_auto:
                recpos_list.append(("_ac_dict", achid_dict))
            if new_obj._has_cross:
                recpos_list.append(("_sp_dict", sphid_dict))

            for (attr, idict) in recpos_list:
                for inhid, jdict in getattr(other, attr).items():
                    getattr(new_obj, attr)[inhid_dict.get(inhid, inhid)] = {
                        idict.get(sphid, sphid): kdict.copy()
                        for sphid, kdict in jdict.items()
                    }

        # If the data are in a mixed state, we just want to unloaded it all.
        # Otherwise merge the two. Note that deepcopy for dicts is not particularly
        # fast, although most of the overhead here is trapped up in copying the
        # multitude of ndarrays.
        if (
            (self._tsys_applied != other._tsys_applied)
            or (self.jypk != other.jypk)
            or (self.vis_data is None or other.vis_data is None)
        ):
            new_obj.vis_data = None
            new_obj._tsys_applied = False
        else:
            new_obj.vis_data.update(copy.deepcopy(other_vis))

        if self.raw_data is None or other.raw_data is None:
            new_obj.raw_data = None
        else:
            new_obj.raw_data.update(copy.deepcopy(other_raw))

        new_obj._sp_dict.update(copy.deepcopy(other._sp_dict))

        # Finally, if we have discrepant _has_auto states, we force the resultant object
        # to unload any potential auto metadata.
        if self._has_auto != other._has_auto:
            warnings.warn(
                "Both objects do not have auto-correlation data. Since force=True, "
                "dropping auto-correlation data and metadata from the combined object."
            )
            new_obj._clear_auto()
        elif self.auto_data is None or other.auto_data is None:
            new_obj.auto_data = None
        else:
            new_obj.auto_data.update(copy.deepcopy(other_auto))

        return new_obj

    def __iadd__(self, other, merge=None, overwrite=False, force=False):
        """
        Add two MirMetaData objects in place.

        Combine two MirMetaData objects together, nominally under the assumption that
        they have been read in by the same file. If two objects are read in from
        _different_ files, then users may find the `concat` method more appropriate
        to use.

        Note that while some metadata checking is performed to verify that the objects
        look identical, no checking is done on the `vis_data`, `raw_data`, and
        `auto_data` attributes (other than that they exist).

        Parameters
        ----------
        other : MirParser object
            Other MirParser object to combine with this data set.
        merge : bool
            If set to True, assume that the objects originate from the amd file, and
            combine them together accordingly. If set to False, assume the two objects
            originate from _different_ files, and concatenate them together. By default,
            the method will check the internal file dictionary to see if it appears the
            two objects come from the same file(s), automatically choosing between
            merging or concatenating.
        overwrite : bool
            If set to True, metadata from `other` will overwrite that present in
            this object, even if they differ. Default is False.
        force : bool
            If set to True, bypass certain checks to force the method to combine the
            two objects. Note that this option should be used with care, as it can
            result in objects with duplicate data (which may affect downstream
            processing), or loss of support for handling auto-correlations within
            the object. Default is False.

        Raises
        ------
        TypeError
            If attempting to add a MirParser object with any other type of object.
        ValueError
            If the objects cannot be combined, either because of differing metadata (if
            `overwrite=False`), different data being loaded (raw vs vis vs auto), or
            because the two objects appear to be loaded from different files (and
            `force=False`).
        """
        return self.__add__(
            other, merge=merge, overwrite=overwrite, force=force, inplace=True
        )

    def select(
        self,
        where=None,
        and_where_args=True,
        and_mask=True,
        update_data=None,
        reset=False,
    ):
        """
        Select a subset of data inside a Mir-formatted file.

        This routine allows for one to select a subset of data within a Mir dataset,
        based on various metadata. The select command is designed to be flexible,
        allowing for both multiple simultaneous selections and serial calls. Users
        should be aware that the command is case sensitive, and uses "strict" agreement
        (i.e., it does not mimic the behavior is `np.isclose`) with metadata values.
        By default, multiple selection criteria are combined together via an "and"
        operation; for example, if you wanted to select only data from Antenna 1 while
        on 3c84, you would set `where=(("source", "eq", "3c84"), ("ant", "eq", 1))`.

        The select command will automatically translate information in `codes_data`, the
        and the various indexing keys, e.g., it will convert an allowed value for
        "source" (found in`codes_read`) into allowed values for "isource" (found in
        `in_read`).

        Parameters
        ----------
        where : tuple or list of tuples
            Optional argument, where tuple is used to identify a matching subset of
            data. Each tuple must be 3 elements in length, consisting of the "selection
            field", the "comparison operator", and the "comparison value". The selection
            field match one of the field names inside one of the metadata attributes
            (e.g., "ant1", "mjd", "source", "fsky"). The comparison operator specifies
            how the metadata are compared against the selection field. Allowed
            comparisons include:
            "eq" or "==" (equal to);
            "ne" or "!=" (not equal to);
            "lt" or "<" (less than);
            "le" or "<=" (less than or equal to);
            "gt" or ">" (greater than);
            "ge" or ">=" (greater than or equal to);
            "between" (between a range of values);
            "outside" (outside of a range of values).
            The selection value are the value or sequence of values to compare against
            that present in the selection field. Note that in most cases, this should be
            a single number, unless the comparison operator is "between" our "outside"
            (which requires a this element to be a sequence of length 2), of "eq" or
            "ne", where either a single value (string or number), or multiple values can
            be supplied in a single where statement. Multiple selections can be made by
            supplying a sequence of 3-element tuples, where the results of each
            selection are combined based on the value of `and_where_args`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criteria are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        and_mask : bool
            If set to True, then the mask generated by the selection criteria above will
            be combined with the existing mask values using an element-wise "and"
            operation. If set to False, the two will instead be combined with an
            element-wise "or" operation. Default is True.
        update_data : bool
            Whether or not to update the visibility values (as recorded in the
            attributes `vis_data` and `raw_data`). If set to True, it will force data
            to be loaded from disk, based on what had been previously loaded. If False,
            it will unload those attributes. The default is to do nothing if data are
            not loaded, otherwise to downselect from the existing data in the object
            if all records are present (and otherwise unload the data).
        reset : bool
            If set to True, undoes any previous filters, so that all records are once
            again visible.Default is False.
        """
        # This dict records some common field aliases that users might specify, that
        # map to specific fields in the metadata.
        alias_dict = {
            "ant": "antenna",
            "ant1": "tel1",
            "ant2": "tel2",
        }

        # Make sure that where is a list, to make arg parsing more uniform downstream
        if not isinstance(where, list) and where is not None:
            where = [where]

        # If supplying where arguments, we want to condition them properly so that
        # they will result in a successful search, if at all possible.
        if where is not None:
            for idx in range(len(where)):
                query = where[idx]
                # Substitute alias names
                if query[0] in alias_dict:
                    query = (alias_dict[query[0]], *query[1:])
                    where[idx] = query

                # The codes data is different than other metadata, in that it maps
                # arbitrary strings to integer values under specific header names. If
                # we have an argument that matches once of these, we want to substitute
                # the string and field name for the appropriate integer (and associated
                # indexing field name).
                try:
                    index_vals = self.codes_data.where(*query)
                    if query[0] in self.codes_data._codes_index_dict:
                        where[idx] = (
                            self.codes_data._codes_index_dict[query[0]],
                            "eq",
                            index_vals,
                        )
                except MirMetaError:
                    # This error gets thrown if no variable name matching the first arg
                    # in the where tuple matches. In this case, we just trust that the
                    # field name belongs one of the other metadata tables.
                    pass

        # We have 5-6 objects to perform a search across, which link to each other in
        # different ways. We create this dict to start rather than populating the
        # objects since we don't want to change anything until we know that the where
        # statements all parse successfully.
        search_dict = {
            "in_data": None,
            "bl_data": None,
            "sp_data": None,
            "eng_data": None,
            "we_data": None,
        }

        if self._has_auto:
            search_dict["ac_data"] = None

        for attr in search_dict:
            try:
                # Attempt to generate a mask based on the supplied search criteria
                search_dict[attr] = self._metadata_attrs[attr]._generate_mask(
                    where=where, and_where_args=and_where_args
                )
            except MirMetaError:
                # If no field listed in the sequence of tuples is identified
                # in the attribute, it'll throw the above error. That just means we
                # aren't searching on anything relevant to this attr, so move along.
                pass

        for attr, mask in search_dict.items():
            self._metadata_attrs[attr].set_mask(
                mask=mask, reset=reset, and_mask=and_mask
            )

        # Now that we've screened the data that we want, update the object appropriately
        self._update_filter(update_data=update_data)

    def _read_compass_solns(self, filename):
        """
        Read COMPASS-formatted gains and flags.

        This is an internal helper function, not designed to be called by users. Reads
        in an HDF5 file containing the COMPASS-derived flags and gains tables, that
        can later be applied to the data.

        Parameters
        ----------
        filename : str
            Name of the file containing the COMPASS flags and gains solutions.

        Returns
        -------
        compass_soln_dict : dict
            Dictionary containing the flags and gains tables for the dataset. The dict
            contains multiple entries, including "wide_flags", "sphid_flags", and
            "bandpass_gains", which each correspond to their own dicts for flags and
            gains solutions.

        Raises
        ------
        UserWarning
            If the COMPASS solutions do not appear to overlap in time with that in
            the MirParser object.
        """
        # TODO _read_compass_solns: Verify this works.
        # When we read in the COMPASS solutions, we will need to map some per-blhid
        # values to per-sphid values, so create an indexing array that we can do this
        # with conveniently.
        sp_bl_map = self.bl_data._index_query(header_key=self.sp_data["blhid"])

        # COMPASS stores its solns in a multi-dimensional array that needs to be
        # split apart in order to match for MirParser format. We can match each sphid
        # to a particular paring of antennas and polarizations/receivers, sideband,
        # and spectral chunk, so we use the dict below to map that sequence to a
        # particular sphid, for later use.
        sphid_dict = {}
        for sphid, inhid, ant1, rx1, ant2, rx2, sb, chunk in zip(
            self.sp_data["sphid"],
            self.sp_data["inhid"],
            self.bl_data.get_value("iant1", index=sp_bl_map),
            self.bl_data.get_value("ant1rx", index=sp_bl_map),
            self.bl_data.get_value("iant2", index=sp_bl_map),
            self.bl_data.get_value("ant2rx", index=sp_bl_map),
            self.bl_data.get_value("isb", index=sp_bl_map),
            self.sp_data["corrchunk"],
        ):
            sphid_dict[(inhid, ant1, rx1, ant2, rx2, sb, chunk)] = sphid

        # Create an empty dict, that'll be what we hand back to the user.
        compass_soln_dict = {}

        # This dict will be what we stuff bandpass solns into, as an entry in the "main"
        # COMPASS solutions dict (just above).
        bandpass_gains = {}

        # MATLAB v7.3 format uses HDF5 format, so h5py here ftw!
        with h5py.File(filename, "r") as file:
            # First, pull out the bandpass solns, and the associated metadata
            ant_arr = np.array(file["antArr"][0])  # Antenna number
            rx_arr = np.array(file["rx1Arr"][0])  # Receiver (0=RxA, 1=RxB)
            sb_arr = np.array(file["sbArr"][0])  # Sideband (0=LSB, 1=USB)
            chunk_arr = np.array(file["winArr"][0])  # Spectral win #
            bp_arr = np.array(file["bandpassArr"])  # BP gains (3D array)

            # Parse out the bandpass solutions for each antenna, pol/receiver, and
            # sideband-chunk combination.
            for idx, ant in enumerate(ant_arr):
                for jdx, (rx, sb, chunk) in enumerate(zip(rx_arr, sb_arr, chunk_arr)):
                    cal_data = bp_arr[idx, jdx]
                    cal_flags = (cal_data == 0.0) | ~np.isfinite(cal_data)
                    cal_data[cal_flags] = 1.0
                    bandpass_gains[(ant, rx, sb, chunk)] = {
                        "cal_data": cal_data,
                        "cal_flags": cal_flags,
                    }

            # Once we divvy-up the solutions, plug them back into the dict that
            # we will pass back to the user.
            compass_soln_dict["bandpass_gains"] = bandpass_gains

            # Now, we can move on to flags. Note that COMPASS doesn't have access to
            # the integration header IDs, so we have to do a little bit of matching
            # based on the timestamp of the data in COMPASS vs MIR (via the MJD).
            mjd_compass = np.array(file["mjdArr"][0])
            mjd_mir = self.in_data["mjd"]
            inhid_arr = self.in_data["inhid"]

            # Match each index to an inhid entry
            index_dict = {}

            # On occasion, there are some minor rounding issues with the time stamps
            # than can affect things on the order of up to half a second, so we use
            # isclose + an absolute tolerance of 0.5 seconds (in units of Julian days)
            # to try and match COMPASS to Mir timestamp. This is shorter than the
            # shortest possible integration time (as of 2022), so this should be
            # specific enough for our purposes here.
            atol = 0.5 / 86400
            for idx, mjd in enumerate(mjd_compass):
                check = np.where(np.isclose(mjd, mjd_mir, atol=atol))[0]
                index_dict[idx] = None if (len(check) == 0) else inhid_arr[check[0]]

            # Pull out some metadata here for parsing the individual solutions
            flags_arr = np.array(file["flagArr"])  # Per-sphid flags
            wflags_arr = np.array(file["wideFlagArr"])  # "For all time" flags
            ant1_arr = np.array(file["ant1Arr"][0])  # First ant in baseline
            rx1_arr = np.array(file["rx1Arr"][0])  # Receiver/pol of first ant
            ant2_arr = np.array(file["ant2Arr"][0])  # Second ant in baseline
            rx2_arr = np.array(file["rx2Arr"][0])  # Receiver/pol of second ant
            sb_arr = np.array(file["sbArr"][0])  # Sideband (0=LSB, 1=USB)
            chunk_arr = np.array(file["winArr"][0])  # Chunk/spectral window number

            # The wide flags record when some set of channels was bad throughout an
            # entire track, and are calculated on a per-baseline basis. Make a dict
            # with the keys mapped to ant-receivers/pols and sideband-chunk.
            wide_flags = {}

            # Note that the two loops here are used to match the indexing scheme of the
            # flags (so the slowest loop iterates on the outer-most axis of the array).
            for idx, (ant1, ant2) in enumerate(zip(ant1_arr, ant2_arr)):
                for jdx, (rx1, rx2, sb, chunk) in enumerate(
                    zip(rx1_arr, rx2_arr, sb_arr, chunk_arr)
                ):
                    wide_flags[(ant1, rx1, ant2, rx2, sb, chunk)] = wflags_arr[idx, jdx]

            # Once the wide flags dict is built, plug it back into the main dict.
            compass_soln_dict["wide_flags"] = wide_flags

            # Now we need to handle the per-sphid flags.
            sphid_flags = {}

            # Note that the three loops here match the indexing scheme of the spectral
            # flags (so the slowest loop iterates on the outer-most axis of the array).
            for idx, inhid in index_dict.items():
                if inhid is None:
                    # If there is no matching inhid, it means that the COMPASS soln
                    # has no flags for this integration. Skip it (on apply, it will
                    # use the wide flags instead).
                    continue
                for jdx, (ant1, ant2) in enumerate(zip(ant1_arr, ant2_arr)):
                    for kdx, (rx1, rx2, sb, chunk) in enumerate(
                        zip(rx1_arr, rx2_arr, sb_arr, chunk_arr)
                    ):
                        try:
                            sphid = sphid_dict[(inhid, ant1, rx1, ant2, rx2, sb, chunk)]
                            sphid_flags[sphid] = flags_arr[idx, jdx, kdx]
                        except KeyError:
                            # If we don't have a match for the entry, that's okay,
                            # since this may just be a subset of the data that COMPASS
                            # processed for the track. In this case, discard the flags.
                            pass

            if len(sphid_flags) == 0:
                # If we don't have _any_ flags recorded, raise a warning, since this
                # might be an indicator that we've selected the wrong set of solns.
                warnings.warn(
                    "No metadata from COMPASS matches that in this data set. Verify "
                    "that the COMPASS solutions are in fact for this set of data."
                )

            # Finally, plug this set of flags back into the solns dict.
            compass_soln_dict["sphid_flags"] = sphid_flags

        return compass_soln_dict

    def _apply_compass_solns(self, compass_soln_dict, apply_flags=True, apply_bp=True):
        """
        Apply COMPASS-derived gains and flagging.

        Note that this is an internal helper function, not designed to be called by
        users. This routine will apply flagging and gains read in by the COMPASS
        pipeline (as returned by the `_read_compass_solns` method). Presently, the
        method will only attempt to apply spectral flagging and bandpass solutions
        for un-averaged data. Be aware that this routine will modify values stored
        in the `vis_data` attribute.

        Parameters
        ----------
        compass_soln_dict : dict
            A dict containing the the various flagging and gains tables from COMPASS,
            as returned by `_read_compass_solns`.
        apply_flags : bool
            If True (default), apply COMPASS flags to the data set.
        apply_bp : bool
            If True (default), apply COMPASS bandpass solutions.

        Raises
        ------
        ValueError
            If visibility data are not loaded (not that its not enough to have raw data
            loaded -- that needs to be converted to "normal" vis data).
        """
        # TODO _apply_compass_solns: Actually test that this works.

        # If the data isn't loaded, there really isn't anything to do.
        if self.vis_data is None:
            raise ValueError(
                "Visibility data must be loaded in order to apply COMPASS solns. Run "
                "`load_data(load_cross=True)` to fix this issue."
            )

        # Use this to map certain per-blhid values to individual sphid entries.
        sp_bl_map = self.bl_data._index_query(header_key=self.sp_data["blhid"])

        # Now grab all of the metadata we want for processing the spectral records
        sphid_arr = self.sp_data["sphid"]  # Spectral window header ID
        ant1_arr = self.bl_data.get_value("iant1", index=sp_bl_map)  # Ant 1 Number
        rx1_arr = self.bl_data.get_value("ant1rx", index=sp_bl_map)  # Pol | 0:X/L 1:Y/R
        ant2_arr = self.bl_data.get_value("iant2", index=sp_bl_map)  # Ant 2 Number
        rx2_arr = self.bl_data.get_value("ant2rx", index=sp_bl_map)  # Pol | 0:X/L 1:Y/R
        chunk_arr = self.sp_data["corrchunk"]  # Correlator window number
        sb_arr = self.bl_data.get_value("isb", index=sp_bl_map)  # Sideband| 0:LSB 1:USB

        # In case we need it for "dummy" gains solutions, tabulate how many channels
        # there are in each spectral window, remembering that spectral windows can
        # vary depending on which polarization we are looking at (determined by the
        # values in rx1 and rx2).
        chunk_size_dict = {
            (sb, chunk, rx1, rx2): nch
            for sb, chunk, rx1, rx2, nch in zip(
                sb_arr, chunk_arr, rx1_arr, rx2_arr, self.sp_data["nch"]
            )
        }

        if apply_bp:
            # Let's grab the bandpass solns upfront before we iterate through
            # all of the individual spectral records.
            bp_compass = compass_soln_dict["bandpass_gains"]

            for sphid, sb, ant1, rx1, ant2, rx2, chunk in zip(
                sphid_arr, sb_arr, ant1_arr, rx1_arr, ant2_arr, rx2_arr, chunk_arr
            ):
                # Create an empty dictionary here for calculating the solutions for
                # individual receiver pairs within different spectral windows. We'll
                # be basically calculating the gains solns on an "as needed" basis.
                bp_soln = {}

                try:
                    # If we have calculated the bandpass soln before, grab it now.
                    cal_soln = bp_soln[(ant1, rx1, ant2, rx2, chunk, sb)]
                except KeyError:
                    # If we haven't calculated the bandpass soln for this particular
                    # pairing before, then we need to calculate it!
                    try:
                        # Attempt to lookup the solns for the first antenna in the pair
                        ant1soln = bp_compass[(ant1, rx1, sb, chunk)]["cal_data"]

                        # If we can't find a soln for the first antenna, then make
                        # all the gains values equal to one and mark all the channels
                        # as being flagged.
                        ant1flags = bp_compass[(ant1, rx1, sb, chunk)]["cal_flags"]
                    except KeyError:
                        # If we can't find a soln for the first antenna, then make
                        # all the gains values equal to one and mark all the channels
                        # as being flagged.
                        ant1soln = np.ones(
                            chunk_size_dict[(sb, chunk, rx1, rx2)], dtype=np.complex64
                        )
                        ant1flags = np.ones(ant1soln.shape, dtype=bool)
                    try:
                        # Attempt to lookup the solns for the second antenna in the pair
                        ant2soln = bp_compass[(ant2, rx2, sb, chunk)]["cal_data"]
                        ant2flags = bp_compass[(ant2, rx2, sb, chunk)]["cal_flags"]
                    except KeyError:
                        # If we can't find a soln for the second antenna, then make
                        # all the gains values equal to one and mark all the channels
                        # as being flagged.
                        ant2soln = np.ones(
                            chunk_size_dict[(sb, chunk, rx1, rx2)], dtype=np.complex64
                        )
                        ant2flags = np.ones(ant1soln.shape, dtype=bool)

                    # For each baseline, we can calculate the correction needed by
                    # multiplying the gains for ant1 by the complex conj of the gains
                    # for antenna 2. Note that the convention for the gains solns in
                    # COMPASS are set such that they need to be divided out. Division
                    # is a more computationally expensive operation than multiplication,
                    # so we take the reciprocal such that we can just multiply the
                    # visibilities that gains solns we calculate here.
                    cal_data = np.reciprocal(ant1soln * np.conj(ant2soln))

                    # Flag the soln if either ant1 or ant2 solns are bad.
                    cal_flags = ant1flags | ant2flags

                    # Finally, construct our simple dict, and plug it back in to our
                    # bookkeeping dict that we are using to record the solns.
                    cal_soln = {"cal_data": cal_data, "cal_flags": cal_flags}
                    bp_soln[(ant1, rx1, ant2, rx2, chunk, sb)] = cal_soln
                finally:
                    # One way or another, we should have a set of gains solutions that
                    # we can apply now (flagging the data where appropriate).
                    self.vis_data[sphid]["data"] *= cal_soln["cal_data"]
                    self.vis_data[sphid]["flags"] += cal_soln["cal_flags"]

        if apply_flags:
            # For the sake of reading/coding, let's assign the two catalogs of flags
            # to their own variables, so that we can easily call them later.
            sphid_flags = compass_soln_dict["sphid_flags"]
            wide_flags = compass_soln_dict["wide_flags"]

            for idx, sphid in enumerate(sphid_arr):
                # Now we'll step through each spectral record that we have to process.
                # Note that we use unpackbits because MATLAB/HDF5 doesn't seem to have
                # a way to store single-bit values, and so the data are effectively
                # compressed into uint8, which can be reinflated via unpackbits.
                try:
                    # If we have a flags entry for this sphid, then go ahead and apply
                    # them to the flags table for that spectral record.
                    self.vis_data[sphid]["flags"] += np.unpackbits(
                        sphid_flags[sphid], bitorder="little"
                    ).astype(bool)
                except KeyError:
                    # If no key is found, then we want to try and use the "broader"
                    # flags to mask out the data that's associated with the given
                    # antenna-receiver combination (for that sideband and spec window).
                    # Note that if we do not have an entry here, something is amiss.
                    try:
                        self.vis_data[sphid]["flags"] += np.unpackbits(
                            wide_flags[
                                (
                                    ant1_arr[idx],
                                    rx1_arr[idx],
                                    ant2_arr[idx],
                                    rx2_arr[idx],
                                    sb_arr[idx],
                                    chunk_arr[idx],
                                )
                            ],
                            bitorder="little",
                        ).astype(bool)
                    except KeyError:
                        # If we _still_ have no key, that means that this data was
                        # not evaluated by COMPASS, and for now we will default to
                        # not touching the flags.
                        pass

    @staticmethod
    def _generate_chanshift_kernel(chan_shift, kernel_type, alpha_fac=-0.5, tol=1e-3):
        """
        Calculate the kernel for shifting a spectrum a given number of channels.

        This function will calculate the parameters required for shifting a given
        frequency number by an arbitrary amount (i.e., not necessarily an integer
        number of channels).

        chan_shift : float
            Number of channels that the spectrum is to be shifted by, where positive
            values indicate that channels are moving "up" to higher index positions.
            No default.
        kernel_type : str
            There are several supported interpolation schemes that can be used for
            shifting the spectrum a given number of channels. The three supported
            schemes are "nearest" (nearest-neighbor; choose the closest channel to the
            one desired), "linear" (linear interpolation; interpolate between the two
            closest channel), and "cubic" (cubic convolution; see "Cubic Convolution
            Interpolation for Digital Image Processing" by Robert Keys for more details
            ). Nearest neighbor is the fastest, although cubic convolution generally
            provides the best spectral PSF.
        alpha_fac : float
            Only used when `kernel_type="cubic"`, adjusts the alpha parameter for
            the cubic convolution kernel. Typical values are usually in the range of
            -1 to -0.5, the latter of which is the default value due to the compactness
            of the PSF using this kernel.
        tol : float
            If the desired frequency shift is close enough to an integer number of
            channels, then the method will forgo any attempt and interpolation and
            will simply return the nearest channel desired. The tolerance for choosing
            this behavior is given by this parameter, in units of number of channels.
            The default is 1e-3, which means that if the desired shift is within one
            one-thousandth of an existing channel, the method will (for the frequency
            window in question) the nearest-neighbor interpolation scheme. Must be
            in the range [0, 0.5].

        Returns
        -------
        coarse_shift : int
            The "coarse" interpolation, which is the number of whole channels to shift
            the spectrum by (in addition to the "fine" interpolation).
        kernel_size : int
            For the "fine" (i.e., subsample) interpolation, the size of the smoothing
            kernel, which depends on `kernel_type` (0 for "nearest", 2 for "linear" and
            4 for "cubic").
        shift_kernel : ndarray
            For the "fine" (i.e., subsample) interpolation, the smoothing kernel used
            convolve with the array to produce the interpolated samples. Shape is
            `(kernel_size)`, of dtype float32.

        Raises
        ------
        ValueError
            If tol is outside of the range [0, 0.5], or if kernel_type does not match
            the list of supported values.
        """
        if (tol < 0) or (tol > 0.5):
            raise ValueError("tol must be between 0 and 0.5.")

        coarse_shift = np.floor(chan_shift).astype(int)
        fine_shift = chan_shift - coarse_shift

        if (fine_shift < tol) or (fine_shift > (1 - tol)):
            coarse_shift += fine_shift > tol
            fine_shift = 0

        if kernel_type == "nearest" or (fine_shift == 0):
            # If only doing a coarse shift, or if otherwise specified, we can default
            # to the nearest neighbor, which does not actually require convolution
            # to complete (hence why the kernel size is zero and the kernel itself
            # is just None).
            shift_tuple = (coarse_shift + (fine_shift >= 0.5), 0, None)
        elif kernel_type == "linear":
            # Linear operation is pretty easy.
            shift_tuple = (
                coarse_shift,
                2,
                np.array(
                    [1 - fine_shift, fine_shift],
                    dtype=np.float32,
                ),
            )
        elif kernel_type == "cubic":
            # Cubic convolution is a bit more complicated, and the exact value
            # depends on this tuning parameter alpha, which from the literature
            # is optimal in the range [-1, -0.5] (although other values can be used).
            # Note that this formula comes from "Cubic Convolution Interpolation for
            # Digital Image Processing" by Robert Keys, IEEE Trans. VOL ASSP-29 #6
            # (Dec 1981).
            shift_tuple = (
                coarse_shift,
                4,
                np.array(
                    [
                        (alpha_fac * ((2 - fine_shift) ** 3))  # 2-left entry
                        - (5 * alpha_fac * ((2 - fine_shift) ** 2))
                        + (8 * alpha_fac * (2 - fine_shift))
                        - (4 * alpha_fac),
                        ((alpha_fac + 2) * ((1 - fine_shift) ** 3))  # 1-left entry
                        - ((alpha_fac + 3) * ((1 - fine_shift) ** 2))
                        + 1,
                        ((alpha_fac + 2) * (fine_shift**3))  # 1-right entry
                        - ((alpha_fac + 3) * (fine_shift**2))
                        + 1,
                        (alpha_fac * ((1 + fine_shift) ** 3))  # 2-right entry
                        - (5 * alpha_fac * ((1 + fine_shift) ** 2))
                        + (8 * alpha_fac * (1 + fine_shift))
                        - (4 * alpha_fac),
                    ],
                    dtype=np.float32,
                ),
            )
        else:
            raise ValueError(
                'Kernel type of "%s" not recognized, must be either "nearest", '
                '"linear" or "cubic"' % kernel_type
            )

        return shift_tuple

    @staticmethod
    def _chanshift_vis(vis_dict, shift_tuple_list, flag_adj=True, inplace=False):
        """
        Frequency shift (i.e., "redoppler") visibility data.

        Parameters
        ----------
        vis_dict : dict
            A dictionary in the format of `vis_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry contains a dict
            with two items: "data", an array of np.complex64 containing the
            visibilities, and "flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).
        shift_tuple_list : list of tuples
            List of the same length as `vis_dict`, each entry of which contains a three
            element tuple matching the output of `_generate_chanshift_kernel`. The first
            entry is the whole number of channels the spectrum must be shifted by, the
            second entry is the size of the smoothing kernel for performing the "fine"
            (i.e., subsample) interpolation, and the third element is the smoothing
            kernel itself.
        flag_adj : bool
            Option to flag channels adjacent to those that are flagged, the number of
            which depends on the interpolation scheme (1 additional channel with linear
            interpolation, and 3 additional channels with cubic convolution). Set to
            True by default, which prevents the window from having an inconsistent
            spectral PSF across the band.
        inplace : bool
            If True, entries in `vis_dict` will be updated with spectrally averaged
            data. If False (default), then the method will construct a new dict that
            will contain the spectrally averaged data.

        Returns
        -------
        new_vis_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `vis_dict`.
        """
        new_vis_dict = vis_dict if inplace else {}

        for (coarse_shift, kernel_size, shift_kernel), (sphid, sp_vis) in zip(
            shift_tuple_list, vis_dict.items()
        ):
            # If there is no channel shift, and no convolution kernel, then there is
            # literally nothing else left to do.
            if (coarse_shift, kernel_size, shift_kernel) == (0, 0, None):
                # There is literally nothing to do here
                if not inplace:
                    new_vis_dict[sphid] = copy.deepcopy(sp_vis)
                continue

            new_vis = np.empty_like(sp_vis["data"])

            if shift_kernel is None:
                # If the shift kernel is None, it means that we only have a coarse
                # channel shift to worry about, which means we can bypass the whole
                # convolution step (and save on a fair bit of processing time).
                new_flags = np.empty_like(sp_vis["flags"])

                # The indexing is a little different depending on the direction of
                # the shift, hence the if statement here.
                if coarse_shift < 0:
                    new_vis[:coarse_shift] = sp_vis["data"][-coarse_shift:]
                    new_flags[:coarse_shift] = sp_vis["flags"][-coarse_shift:]
                    new_vis[coarse_shift:] = 0.0
                    new_flags[coarse_shift:] = True
                else:
                    new_vis[coarse_shift:] = sp_vis["data"][:-coarse_shift]
                    new_flags[coarse_shift:] = sp_vis["flags"][:-coarse_shift]
                    new_vis[:coarse_shift] = 0.0
                    new_flags[:coarse_shift] = True
            else:
                # If we have to execute a convolution, then the indexing is a bit more
                # complicated. We use the "valid" option for convolve below, which will
                # drop (kernel_size - 1) elements from the array, where the number of
                # elements dropped on the left side is 1 more than it is on the right.
                l_edge = (kernel_size // 2) + coarse_shift
                r_edge = (1 - (kernel_size // 2)) + coarse_shift

                # These clip values here are used to clip the original array to both
                # make sure that the size matches, and to avoid doing any unnecessary
                # work during the convolve for entries that will never get used.
                l_clip = r_clip = None

                # If l_edge falls past the "leftmost" index (i.e., 0), then we "cut"
                # the main array to make it fit.
                if l_edge < 0:
                    l_clip = -l_edge
                    l_edge = 0
                # Same thing on the right side. Note we have to use len(new_vis) here
                # because the slice won't work correctly if this value is 0.
                if r_edge >= 0:
                    r_clip = len(new_vis) - r_edge
                    r_edge = len(new_vis)

                # Grab a copy of the array to manipulate, and plug flagging values into
                temp_vis = sp_vis["data"][l_clip:r_clip].copy()
                temp_vis[sp_vis["flags"][l_clip:r_clip]] = (
                    np.complex64(np.nan) if flag_adj else np.complex64(0.0)
                )

                # For some reason, it's about 5x faster to split this up into real
                # and imaginary operations. The use of "valid" also speeds this up
                # another 10-20% (no need to pad the arrays with extra zeros).
                new_vis.real[l_edge:r_edge] = np.convolve(
                    temp_vis.real, shift_kernel, "valid"
                )
                new_vis.imag[l_edge:r_edge] = np.convolve(
                    temp_vis.imag, shift_kernel, "valid"
                )

                # Flag out the values beyond the outer bounds
                new_vis[:l_edge] = new_vis[r_edge:] = (
                    np.complex64(np.nan) if flag_adj else np.complex64(0.0)
                )

                # Finally, regenerate the flags array for the dict entry.
                if flag_adj:
                    new_flags = np.isnan(new_vis)
                    new_vis[new_flags] = 0.0
                else:
                    new_flags = np.zeros_like(sp_vis["flags"])
                    new_flags[:l_edge] = new_flags[r_edge:] = True
            # Update our dict with the new values for this sphid
            new_vis_dict[sphid] = {"data": new_vis, "flags": new_flags}

        return new_vis_dict

    @staticmethod
    def _chanshift_raw(
        raw_dict, shift_tuple_list, flag_adj=True, inplace=False, return_vis=False
    ):
        """
        Frequency shift (i.e., "redoppler") raw data.

        Parameters
        ----------
        raw_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry contains a dict
            with two items: "scale_fac", and np.int16 which describes the common
            exponent for the spectrum, and "data", an array of np.int16 (of length
            equal to twice that found in `sp_data["nch"]` for the corresponding value
            of sphid) containing the compressed visibilities.  Note that entries equal
            to -32768 aren't possible with the compression scheme used for MIR, and so
            this value is used to mark flags.
        shift_tuple_list : list of tuples
            List of the same length as `vis_dict`, each entry of which contains a three
            element tuple matching the output of `_generate_doppler_kernel`. The first
            entry is the whole number of channels the spectrum must be shifted by, the
            second entry is the size of the smoothing kernel for performing the "fine"
            (i.e., subsample) interpolation, and the third element is the smoothing
            kernel itself.
        flag_adj : bool
            Option to flag channels adjacent to those that are flagged, the number of
            which depends on the interpolation scheme (1 additional channel with linear
            interpolation, and 3 additional channels with cubic convolution). Set to
            True by default, which prevents the window from having an inconsistent
            spectral PSF across the band.
        inplace : bool
            If True, entries in `raw_dict` will be updated with spectrally averaged
            data. If False (default), then the method will construct a new dict that
            will contain the spectrally averaged data.
        return_vis : bool
            If True, return data in the "normal" visibility format, where each
            spectral record has a key of "sphid" and a value being a dict of
            "data" (the visibility data, dtype=np.complex64) and "flags"
            (the flagging information, dtype=bool). This option is ignored if
            `inplace=True`.

        Returns
        -------
        data_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `raw_dict` (unless `return_vis=True`).
        """
        # If inplace, point our new dict to the old one, otherwise create
        # an empty dict to plug values into.
        data_dict = raw_dict if inplace else {}
        return_vis = (not inplace) and return_vis

        for shift_tuple, (sphid, sp_raw) in zip(shift_tuple_list, raw_dict.items()):
            # If we are not actually shifting the data (which is what the tuple
            # (0,0,0,None) signifies), then we can bypass most of the rest of the
            # code and simply return a copy of the data if needed.
            if shift_tuple == (0, 0, None):
                if not inplace:
                    data_dict[sphid] = (
                        MirParser._convert_raw_to_vis({0: sp_raw})[0]
                        if return_vis
                        else copy.deepcopy(sp_raw)
                    )
                continue

            # If we are _not_ skipping the spectral averaging, then it turns out to
            # be faster to convert the raw data to "regular" data, doppler-shift it,
            # and then convert it back to the raw format. Note that we set up a
            # "dummy" dict here with an sphid of 0 to make it easy to retrieve that
            # entry after the sequence of calls.
            if return_vis:
                data_dict[sphid] = MirParser._chanshift_vis(
                    MirParser._convert_raw_to_vis({0: sp_raw}),
                    [shift_tuple],
                    flag_adj=flag_adj,
                    inplace=False,
                )[0]
            else:
                data_dict[sphid] = MirParser._convert_vis_to_raw(
                    MirParser._chanshift_vis(
                        MirParser._convert_raw_to_vis({0: sp_raw}),
                        [shift_tuple],
                        flag_adj=flag_adj,
                        inplace=False,
                    )
                )[0]

        # Finally, return the dict containing the raw data.
        return data_dict

    def redoppler_data(
        self,
        freq_shift=None,
        kernel_type="cubic",
        tol=1e-3,
        flag_adj=True,
        fix_freq=None,
    ):
        """
        Re-doppler the data.

        Note that this function may be moved out into utils module once UVData objects
        are capable of carrying Doppler tracking-related information.

        Parameters
        ----------
        freq_shift : ndarray
            Amount to shift each spectral window by in frequency space. Shape is the
            same as the attribute `sp_data`, of dtype float32, in units of GHz. If no
            argument is provided (or if set to None), then the method will assume you
            want to redoppler to the topocentric rest frame, using the information
            stored in the MirParser object. Note that if supplying a value `delta_nu`,
            the center of the spectra will be shifted to `old_center + delta_nu`.
        kernel_type : str
            The `redoppler_data` method allows for several interpolation schemes for
            adjusting the frequencies of the individual channels. The three supported
            schemes are "nearest" (nearest-neighbor; choose the closest channel to the
            one desired), "linear" (linear interpolation; interpolate between the two
            closest channel), and "cubic" (cubic convolution; see "Cubic Convolution
            Interpolation for Digital Image Processing" by Robert Keys for more details
            ). Nearest neighbor is the fastest, although cubic convolution generally
            provides the best spectral PSF.
        tol : float
            If the desired frequency shift is close enough to an integer number of
            channels, then the method will forgo any attempt and interpolation and
            will simply return the nearest channel desired. The tolerance for choosing
            this behavior is given by this parameter, in units of number of channels.
            The default is 1e-3, which means that if the desired shift is within one
            one-thousandth of an existing channel, the method will (for the frequency
            window in question) the nearest-neighbor interpolation scheme.
        flag_adj : bool
            Option to flag channels adjacent to those that are flagged, the number of
            which depends on the interpolation scheme (1 additional channel with linear
            interpolation, and 3 additional channels with cubic convolution). Set to
            True by default, which prevents the window from having an inconsistent
            spectral PSF across the band.
        fix_freq : bool
            Only used if `freq_shift` is left unset (or otherwise set to None). Some
            versions of MIR data have frequency information incorrectly stored. If set
            to True, this metadata will be fixed before doppler-shifting the data. By
            default, the method will apply this correction if the version of the MIR
            data format is known to have the defective metadata.

        Raises
        ------
        ValueError
            If tol is outside of the range [0, 0.5], or if kernel_type does not match
            the list of supported values. Also if providing no argument to freq_shift,
            but doppler-tracking information cannot be determined (either because
            it's the wrong file version or because the receiver code isn't recognized).
        """
        if freq_shift is None:
            if self.codes_data["filever"] in [["2"], ["3"]]:
                raise ValueError(
                    "MIR file format < v4.0 detected, no doppler tracking information "
                    "is stored in the file headers."
                )
            # Grab the metadata from the sp data structure, flipping the sign since
            # we want to shift the spectrum back to the listed sky frequency.
            freq_shift = -self.sp_data["fDDS"]

            # If we need to "fix" the values, do it now.
            if (fix_freq is None and (self.codes_data["filever"] == ["4"])) or fix_freq:
                # Figure out which receiver this is.
                rx_code = np.median(self.bl_data["irec"][self.bl_data["ant1rx"] == 0])
                rx_name = self.codes_data["rec"][rx_code]
                if rx_name not in ("230", "345"):
                    raise ValueError("Receiver code %i not recognized." % rx_code)

                freq_shift *= 2 if (rx_name == "230") else 3
                # We have to do a bit of special handling for the so-called "RxB"
                # data, which doesn't actually have the fDDS values stored. The correct
                # value though just turns out to be the the RxA value multiplied by
                # the ratio of the two gunn frequencies.
                rxa_blhids = self.bl_data["blhid"][
                    (self.bl_data["ant1rx"] == 0) & (self.bl_data["ant2rx"] == 0)
                ]
                rxb_blhids = self.bl_data["blhid"][
                    (self.bl_data["ant1rx"] == 1) & (self.bl_data["ant2rx"] == 1)
                ]
                sp_rxa = np.isin(self.sp_data["blhid"], rxa_blhids)
                sp_rxb = np.isin(self.sp_data["blhid"], rxb_blhids)
                freq_scale = np.median(self.sp_data["gunnLO"][sp_rxb]) / np.median(
                    self.sp_data["gunnLO"][sp_rxa]
                )
                freq_shift[sp_rxb] *= freq_scale

                # Finally, we want to just ignore the pseudo-cont values
                freq_shift[self.sp_data["corrchunk"] == 0] = 0.0

        # Convert frequency shift into number of channels to shift. Note that the
        # negative sign here is to flip conventions (i.e., shifting "up" the center of
        # the band requires shifting "down" by a certain number of frequency channels).
        chan_shift_arr = -freq_shift / (self.sp_data["fres"] / 1000)

        # We need to generate a set of tuples, which will be used by the lower level
        # re-doppler routines for figuring out
        shift_dict = {}
        shift_tuple_list = []
        for chan_shift in chan_shift_arr:
            try:
                shift_tuple_list.append(shift_dict[chan_shift])
            except KeyError:
                shift_tuple = self._generate_chanshift_kernel(
                    chan_shift, kernel_type, tol=tol
                )
                shift_dict[chan_shift] = shift_tuple
                shift_tuple_list.append(shift_tuple)

        # Okay, now we have all of the metadata, so do the thing.
        if self.raw_data is not None:
            self.raw_data = self._chanshift_raw(
                self.raw_data, shift_tuple_list, inplace=True, flag_adj=flag_adj
            )
        if self.vis_data is not None:
            self.vis_data = self._chanshift_vis(
                self.vis_data, shift_tuple_list, inplace=True, flag_adj=flag_adj
            )
