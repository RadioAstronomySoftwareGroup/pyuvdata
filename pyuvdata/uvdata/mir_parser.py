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
    NEW_AUTO_DTYPE,
    NEW_AUTO_HEADER,
    NEW_VIS_DTYPE,
    NEW_VIS_HEADER,
    NEW_VIS_PAD,
    OLD_AUTO_DTYPE,
    OLD_AUTO_HEADER,
    OLD_VIS_DTYPE,
    OLD_VIS_HEADER,
    OLD_VIS_PAD,
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

__all__ = ["MirParser", "MirPackdataError"]


class MirPackdataError(Exception):
    """
    Class for errors when ingesting packed MIR data.

    This class is used to raise error when attempting to read "packed" data sets, which
    are long byte arrays, within which data has been packed, intermixed with some basic
    header information. Because the data can be irregularly shaped, file integrity
    issues (e.g., flipped bits) can make the header information unparsable. This class
    is used in practice to catch such errors.
    """

    def __init__(self, message="There was an error parsing a packed data file."):
        super().__init__(message)


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
        *,
        compass_soln=None,
        make_v3_compliant=None,
        old_vis_format=None,
        nchunks=None,
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
        filepath : str or Path
            Filepath is the path to the folder containing the Mir data set.
        compass_soln : str
            Optional argument, specifying the path of COMPASS-derived flagging and
            bandpass gains solutions, which are loaded into the object.
        make_v3_compliant : bool
            Convert the metadata required for filling a UVData object into a
            v3-compliant format. Only applicable if MIR file version is v1 or v2.
            Default behavior is to set this to True if the file version is v1 or v2.
        old_vis_format : bool
            Prior to the v1 data format, older visibility data was recorded with
            different header fields and endianness (big-endian versus the current
            little-endian format). If set to True, data are assumed to be in this "old"
            format, if set to False, data are assumed to be in the current format.
            Default behavior is to attempt to automatically determine old versus new
            based on metadata, and set this value accordingly.
        nchunks : int
            Only used if `has_auto=True` is set, this value is used to set the number
            of spectral windows in a single auto-correlation set. The default behavior
            is to set this value automatically depending on file version. This setting
            should only be used by advanced users in specialized circumstances.
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
        self._stored_masks = {}

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
        self._tsys_use_cont_det = True
        self._has_compass_soln = False
        self._compass_bp_soln = None
        self._compass_sphid_flags = None
        self._compass_static_flags = None

        # This value is the forward gain of the antenna (in units of Jy/K), which is
        # multiplied against the system temperatures in order to produce values in units
        # of Jy (technically this is the SEFD, which when multiplied against correlator
        # coefficients produces visibilities in units of Jy). Default is 130.0, which
        # is the estimated value for SMA.
        self.jypk = 130.0

        # On init, if a filepath is provided, then fill in the object
        if filepath is not None:
            self.read(
                filepath=filepath,
                compass_soln=compass_soln,
                make_v3_compliant=make_v3_compliant,
                old_vis_format=old_vis_format,
                nchunks=nchunks,
                has_auto=has_auto,
                has_cross=has_cross,
                load_auto=load_auto,
                load_cross=load_cross,
                load_raw=load_raw,
            )

    def _load_test_data(self, **kwargs):
        """
        Load the pyuvdata test SMA data set.

        This is a very simple convenience function for grabbing the test SMA data set in
        the MirParser class. It's here mostly to make interactive testing (i.e., outside
        of pytest) easier and faster, without having to worry about extra imports.
        """
        from ..data import DATA_PATH

        self.__init__(os.path.join(DATA_PATH, "sma_test.mir"), **kwargs)

        return self

    def __eq__(self, other, *, verbose=True, metadata_only=False):
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
            "_compass_bp_soln": [
                "cal_soln",
                "cal_flags",
                "weight_soln",
                "weight_flags",
            ],
            "_compass_sphid_flags": [...],
            "_compass_static_flags": [...],
            "_stored_masks": [
                "in_data",
                "bl_data",
                "sp_data",
                "eng_data",
                "we_data",
                "codes_data",
                "antpos_data",
                "ac_data",
            ],
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
            elif item in data_comp_dict:
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
                                atol = rtol = 0
                                if subkey == "data":
                                    # The atol here is set by the max value in the
                                    # spectrum times 2^-10. That turns out to be _about_
                                    # the worst case scenario for moving to and from the
                                    # raw data format, which compresses the data down
                                    # from floats to ints.
                                    atol = 1e-3
                                    if np.any(np.isfinite(this_item[subkey])):
                                        atol = 1e-3 * np.nanmax(
                                            np.abs(this_item[subkey])
                                        )
                                else:
                                    # Otherwise if not looking at data, use something
                                    # close to the single precision floating point.
                                    rtol = 1e-6

                                is_same &= np.allclose(
                                    this_item[subkey],
                                    other_item[subkey],
                                    atol=atol,
                                    rtol=rtol,
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

    def __ne__(self, other, *, verbose=False, metadata_only=False):
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
            if issubclass(getattr(self, attr).__class__, MirMetaData):
                setattr(new_obj, attr, getattr(self, attr).copy())
            elif not (metadata_only and attr in ["vis_data", "raw_data", "auto_data"]):
                if attr not in ["_metadata_attrs", "_sp_dict", "_ac_dict"]:
                    setattr(new_obj, attr, copy.deepcopy(getattr(self, attr)))

        rec_dict_list = []
        if self._has_auto:
            rec_dict_list.append("_ac_dict")
        if self._has_cross:
            rec_dict_list.append("_sp_dict")

        for item in rec_dict_list:
            new_dict = {}
            setattr(new_obj, item, new_dict)
            for inhid, in_dict in getattr(self, item).items():
                new_in_dict = {}
                new_dict[inhid] = new_in_dict
                for key, rec_dict in in_dict.items():
                    new_in_dict[key] = {
                        "start_idx": rec_dict["start_idx"],
                        "end_idx": rec_dict["end_idx"],
                        "chan_avg": rec_dict["chan_avg"],
                    }

        for item in self._metadata_attrs:
            new_obj._metadata_attrs[item] = getattr(new_obj, item)

        return new_obj

    @staticmethod
    def _scan_int_headers(
        filepath=None, hdr_fmt=None, *, old_int_dict=None, return_headers=False
    ):
        """
        Read "sch_read" or "ach_read" mir file into a python dictionary (@staticmethod).

        This is an internal helper function not meant to be called by users, but
        instead is a low-level helper function. This function is used to create or
        otherwise fix a previously generated integration header dictionary (typically
        first created from the metadata). This function will scan through the data file
        and recover the header information.

        Parameters
        ----------
        filepath : str
            Filepath is the path to the folder containing the Mir data set.
        hdr_fmt : list
            List describing header fields for each integration record, given as a list
            of 2-element tuples (as appropriate for a structured array, see the docs
            of numpy.dtype for further details). No default.
        old_int_dict : dict
            Previously build int_dict, typically constructed by _generate_recpos_dict.
            If supplied, the method will use the information in this dictionary to map
            old header values to new ones, and will cross-check information to make sure
            that header information matches where expected.
        return_headers : bool
            If set to True, the method will return the header values scanned from the
            file instead. Sometimes useful for debugging. Default is False.

        Returns
        -------
        int_dict : dict
            Dictionary containing the indexes from data file, where keys match to the
            inhid indexes, and the values contain a two-element tuple, with the length
            of the packdata array (in bytes) the relative offset (also in bytes) of
            the record within the file. Only returned if `return_headers=False`.
        int_list : list
            List containing the scanned header values from the file, sometimes used
            in debugging. Only returned if `return_headers=True`.

        Raises
        ------
        ValueError
            If a value on "inhid" is read from the file that does not match a value
            given in `allowed_inhid` (if set).
        """
        # Check a few things up front to set behavior for this method
        file_size = os.path.getsize(filepath)
        no_rs = not any("record_size" in item for item in hdr_fmt)
        int_list = []

        if old_int_dict is None:
            # If operating blindly, make sure headers have the needed information.
            if no_rs:
                raise MirPackdataError(
                    "Cannot read scan start if headers do not contain record size info."
                )
            record_start = delta_offset = 0
        else:
            # Re-draft the old int-dict based on record start, since that should be
            # unique and also once ordered properly will allow us to zip through.
            old_int_dict = {val["record_start"]: val for val in old_int_dict.values()}
            rs_list = sorted(old_int_dict)
            old_int_dict = {key: old_int_dict[key] for key in rs_list}

            # Set the first record start/delta offset based on the first entry
            record_start = delta_offset = rs_list.pop(0)

        hdr_dtype = np.dtype(hdr_fmt)
        int_dict = {}

        with open(filepath, "rb") as packdata_file:
            while record_start < (file_size - hdr_dtype.itemsize):
                int_vals = np.fromfile(
                    packdata_file, dtype=hdr_dtype, count=1, offset=delta_offset
                )[0]
                int_list.append(int_vals)

                if old_int_dict is None:
                    # Grab some information from the header
                    inhid = int_vals["inhid"]
                    record_size = int(int_vals["record_size"])
                    delta_offset = record_size
                else:
                    inhid = old_int_dict[record_start]["inhid"]
                    record_size = old_int_dict[record_start]["record_size"]
                    # Check if this is a good record or not
                    good_record = no_rs or record_size == int_vals["record_size"]
                    good_record &= inhid == int_vals["inhid"]
                    if not good_record:
                        # If this isn't a good record, we can short-circuit recording
                        # it by increasing it's size more than the file size, which will
                        # fail the record_end check below.
                        record_size = file_size + 1

                record_end = record_start + hdr_dtype.itemsize + record_size

                # If record_size is negative, this file is definitely corrupted, and
                # there's not more we can do at this point from the file alone.
                if record_size < 0:
                    raise MirPackdataError("record_size was negative/invalid.")
                if record_end <= file_size:
                    # If the record end is within the file size, then add to the dict
                    int_dict[inhid] = {
                        "inhid": inhid,
                        "record_size": record_size,
                        "record_start": record_start,
                    }

                if old_int_dict is None:
                    record_start = record_end
                else:
                    # If rs_list is empty, force the while loop to end by making the
                    # record start at the end of the file. Otherwise calculate the
                    # delta since the records _could_ have gaps if an integration has
                    # otherwise been dropped.
                    delta_offset = -(record_start + hdr_dtype.itemsize)
                    record_start = rs_list.pop(0) if len(rs_list) else file_size
                    delta_offset += record_start

        return int_list if return_headers else int_dict

    def _fix_int_dict(self, data_type=None):
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
            int_dict = idict[data_type]["int_dict"]

            try:
                new_dict = self._scan_int_headers(
                    os.path.join(ifile, idict[data_type]["filetype"]),
                    hdr_fmt=idict[data_type]["read_hdr_fmt"],
                )
                # If we got to this point, it means that we successfully read in the
                # full set of headers from the file, which we treat as trusted.
            except MirPackdataError:
                # If we reached this point, either record_size does not exist or is
                # otherwise unreliable, such that we can't use _scan_int_headers
                # blindly. Our only remaining option then is to work with the int_dict
                # that we have in-hand and try to validate that.
                new_dict = self._scan_int_headers(
                    os.path.join(ifile, idict[data_type]["filetype"]),
                    hdr_fmt=idict[data_type]["read_hdr_fmt"],
                    old_int_dict=int_dict,
                )

            # Each file's inhid is allowed to be different than the objects inhid --
            # this is used in cases when combining multiple files together (via
            # concat). Here, we make a mapping of "file-based" inhid values to that
            # stored in the object.
            inhid_map = {val["inhid"]: inhid for inhid, val in int_dict.items()}

            # First, clear out any entries that don't have a match above
            for key in set(inhid_map) - set(new_dict):
                del int_dict[inhid_map[key]]
                del inhid_map[key]

            # Go through the individual entries in each dict, and update them
            # with the "correct" values as determined by scanning through sch_read
            for key in inhid_map:
                int_dict[inhid_map[key]] = new_dict[key]

    @staticmethod
    def _read_packdata(
        file_dict=None,
        inhid_arr=None,
        data_type="cross",
        *,
        use_mmap=False,
        raise_err=None,
    ):
        """
        Read packed data mir file into memory (@staticmethod).

        Parameters
        ----------
        file_dict : dict
            Dictionary which maps individual integrations to a specific packed data
            record on disk. Keys are the path(s) to the data, with values that are
            themselves dicts with keys of "auto" and/or "cross", which map to per-file
            indexing information ("filetype": the name of the file in the Mir folder;
            "int_dict": per-integration information that is typically generated by the
            `_generate_recpos_dict` method of `MirParser.sp_data` and/or
            `MirParser.ac_data`; "read_hdr_fmt": format of packed data headers;
            "read_data_fmt": dtype specifying the packed data type; "common_scale":
            whether the data share a common exponent, typically for int-stored values).
        inhid_arr : sequence of int
            Integration header keys to read the packed data of.
        data_type : str
            Type of data to read, must either be "cross" (cross-correlations) or "auto"
            (auto-correlations). Default is "cross".
        use_mmap : bool
            By default, the method will read all of the data into memory. However,
            if set to True, then the method will return mmap-based objects instead,
            which can be substantially faster on sparser reads.
        raise_err : bool
            By default, the method will raise a warning if something is internally
            inconsistent with the headers in the file, which can happen if the data
            are corrupted in some way. Setting this to True will cause an error to be
            raised instead, and setting to False will cause no message to be raised.

        Returns
        -------
        int_data_dict : dict
            Dictionary of the data, where the keys are inhid and the values are
            the 'raw' block of values recorded in binary format for that inhid.
        """
        # Create a blank dict to plug values into
        int_data_dict = {}

        # We want to create a unique dtype for records of different sizes. This will
        # make it easier/faster to read in a sequence of integrations of the same size.
        size_set = {
            rec_dict["record_size"]
            for idict in file_dict.values()
            for rec_dict in idict[data_type]["int_dict"].values()
        }

        key_set = list(inhid_arr)
        key_check = key_set.copy()
        # We add an extra key here, None, which cannot match any of the values in
        # int_start_dict (since inhid is type int). This basically tricks the loop
        # below into spitting out the last integration
        key_set.append(None)
        for filepath, indv_file_dict in file_dict.items():
            # Sort out the header format and details
            hdr_fmt = indv_file_dict[data_type]["read_hdr_fmt"]
            hdr_size = np.dtype(hdr_fmt).itemsize
            data_dtype = indv_file_dict[data_type]["read_data_fmt"]
            common_scale = indv_file_dict[data_type]["common_scale"]

            # Populate a list of packdata dtypes for easy use
            int_dtype_dict = {}
            for int_size in size_set:
                int_dtype_dict[int_size] = np.dtype(
                    hdr_fmt + [("packdata", "B", int_size)]
                )

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
                    last_offset + (hdr_size + last_size) * num_vals != int_start
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
                                "data_dtype": [data_dtype] * len(inhid_list),
                                "common_scale": [common_scale] * len(inhid_list),
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
                            zip(
                                np.memmap(
                                    filename=filename,
                                    dtype=read_dict["int_dtype_dict"],
                                    mode="r",
                                    offset=read_dict["start_offset"],
                                    shape=(read_dict["num_vals"],),
                                ),
                                read_dict["data_dtype"],
                                read_dict["common_scale"],
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
                                zip(
                                    np.fromfile(
                                        visibilities_file,
                                        dtype=read_dict["int_dtype_dict"],
                                        count=read_dict["num_vals"],
                                        offset=read_dict["del_offset"],
                                    ),
                                    read_dict["data_dtype"],
                                    read_dict["common_scale"],
                                ),
                            )
                        )

            has_inhid = any("inhid" in item for item in hdr_fmt)
            has_rs = any("record_size" in item for item in hdr_fmt)

            # Skip the checks if there's nothing we need to check
            if has_inhid or has_rs:
                good_check = True
                # Otherwise, zip through and check all items.
                for inhid, (packdata, _, _) in int_data_dict.items():
                    if inhid in int_dict:
                        idict = int_dict[inhid]

                        # There is very little to check in the packdata records, so make
                        # sure this entry has the inhid and size we expect.
                        if has_inhid and (idict["inhid"] != packdata["inhid"]):
                            good_check = False
                        if has_rs and (idict["record_size"] != packdata["record_size"]):
                            good_check = False

                if not good_check:
                    if raise_err:
                        raise MirPackdataError(
                            "File indexing information differs from that found in in "
                            "file_dict. Cannot read in %s data." % data_type
                        )
                    elif raise_err is None:
                        warnings.warn(
                            "File indexing information differs from that found in in "
                            "file_dict. The %s data may be corrupted." % data_type
                        )
        if len(key_check) != 0:
            if raise_err:
                raise ValueError("inhid_arr contains keys not found in file_dict.")
            elif raise_err is None:
                warnings.warn("inhid_arr contains keys not found in file_dict.")

        return int_data_dict

    @staticmethod
    def _make_packdata(int_dict=None, recpos_dict=None, data_dict=None, data_type=None):
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
            nominally matching the keys mentioned above), "record_size" describing the
            size of the packed data (in bytes), and "packdata", which is the packed raw
            data.
        """
        if data_type == "cross":
            hdr_fmt = NEW_VIS_HEADER
            data_dtype = NEW_VIS_DTYPE
            scale_data = True
        elif data_type == "auto":
            hdr_fmt = NEW_AUTO_HEADER
            data_dtype = NEW_AUTO_DTYPE
            scale_data = False
        else:
            raise ValueError(
                'Argument for data_type not recognized, must be "cross" or "auto".'
            )
        # Figure out all of the unique dtypes we need for constructing the individual
        # packed datasets (where we need a different dtype based on the number of
        # individual visibilities we're packing in).
        int_dtype_dict = {}
        for int_size in {idict["record_size"] for idict in int_dict.values()}:
            int_dtype_dict[int_size] = np.dtype(hdr_fmt + [("packdata", "B", int_size)])

        has_inhid = any("inhid" in item for item in hdr_fmt)
        has_record_size = any("record_size" in item for item in hdr_fmt)

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
            if has_inhid:
                int_data["inhid"] = inhid
            if has_record_size:
                int_data["record_size"] = int_subdict["record_size"]

            # Now step through all of the spectral records and plug it in to the
            # main packdata array. In testing, this worked out to be a good degree
            # faster than running np.concat.
            packdata = int_data["packdata"].view(data_dtype)
            for hid, recinfo in recpos_subdict.items():
                data_record = data_dict[hid]
                start_idx = recinfo["start_idx"]
                if scale_data:
                    packdata[start_idx] = data_record["scale_fac"]
                    start_idx += 1

                packdata[start_idx : recinfo["end_idx"]] = data_record["data"]

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
        #   2) Casting scale_fac as float32 will appropriately cast sp_raw values into
        #      float32 as well.
        #   3) I only check for the "special" value for flags in the real component. A
        #      little less robust (both real and imag are marked), but faster and
        #      barring data corruption, this shouldn't be an issue (and a single bad
        #      channel sneaking through is okay).
        #   4) pairs of float32 -> complex64 is super fast and efficient.
        vis_dict = {}
        for sphid, sp_raw in raw_dict.items():
            raw_data = sp_raw["data"]
            data = (np.exp2(np.float32(sp_raw["scale_fac"])) * raw_data).view(
                np.complex64
            )
            flags = raw_data[::2] == -32768
            data[flags] = 0.0
            weights = (~flags).astype(np.float32)

            vis_dict[sphid] = {"data": data, "flags": flags, "weights": weights}

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

    def _read_data(
        self,
        data_type=None,
        *,
        scale_data=True,
        use_mmap=True,
        read_only=False,
        apply_cal=None,
    ):
        """
        Read "sch_read" mir file into a list of ndarrays.

        Parameters
        ----------
        data_type : str
            Type of data to read, must either be "cross" (cross-correlations) or "auto"
            (auto-correlations).
        scale_data : bool
            If set to True and data are stored in a "commonly scaled"/compact format (as
            the visibilities typically are), return a dictionary containing the data in
            a scaled/floating point format. If set to False, will return a dictionary
            containing the data read in the compact format. Default is True. This
            argument is ignored if the data are not scaled.
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
        apply_cal : bool
            If True, COMPASS-based bandpass and flags solutions will be applied upon
            reading in of data. By default, the solutions will be applied if they have
            been previously loaded into the object.

        Returns
        -------
        data_dict : dict
            A dictionary, whose keys are matched to individual header keys -- if
            `data_type="auto"`, then values of achid in `ac_data` are used; if
            `data_type="cross"`, then  values of sphid in `sp_data`, are used. Matched
            to each of these keys is a dict with two items. If the data are "compressed"
            (like is typical for visibilities) and `scale_data=True`,  then a "raw data"
            dict is passed, with keys "scale_fac", an np.int16 which describes the
            common exponent for the spectrum, and "data", an array of typically np.int16
            (of length equal to twice that found in "nch" for the corresponding metadata
            container (n.b, the maximum negative integer value, -32768 for int16,
            aren't possible with the compression scheme used for MIR, so this value is
            used to mark flags). Otherwise this dict contains three keys: "data", an
            array of np.complex64 containing the visibilities, "flags", an array of bool
            containing the per-channel flags of the spectrum, and "weights", which
            contains the per-channel weights for the spectrum (all of length equal to
            `"nch"` in the relevant metadata container).
        """
        if data_type not in ["auto", "cross"]:
            raise ValueError(
                'Argument for data_type not recognized, must be "cross" or "auto".'
            )
        if apply_cal:
            if not self._has_compass_soln:
                raise ValueError("Cannot apply calibration if no tables loaded.")
            if not scale_data:
                raise ValueError("Cannot return raw data if setting apply_cal=True")
        elif apply_cal is None and scale_data:
            apply_cal = self._has_compass_soln

        if data_type == "cross":
            if apply_cal:
                chavg_call = partial(self._rechunk_data, inplace=True)
            else:
                chavg_call = partial(
                    self._rechunk_raw, inplace=True, return_vis=scale_data
                )
            data_map = self._sp_dict
            data_metadata = "sp_data"
        else:
            chavg_call = partial(self._rechunk_data, inplace=True)
            data_map = self._ac_dict
            data_metadata = "ac_data"

        group_dict = getattr(self, data_metadata).group_by("inhid")
        unique_inhid = list(group_dict)

        try:
            # Begin the process of reading the data in, stuffing the "packdata" arrays
            # (to be converted into "raw" data) into the dict below.
            packdata_dict = self._read_packdata(
                file_dict=self._file_dict,
                inhid_arr=unique_inhid,
                data_type=data_type,
                use_mmap=use_mmap,
                raise_err=True,
            )
        except MirPackdataError:
            # Catch an error that indicates that the metadata inside the vis file does
            # not match that in _file_dict, and attempt to fix the problem.
            warnings.warn(
                "Values in int_dict do not match that recorded inside the "
                "file for %s data. Attempting to fix this automatically." % data_type
            )
            self._fix_int_dict(data_type)
            packdata_dict = self._read_packdata(
                file_dict=self._file_dict,
                inhid_arr=unique_inhid,
                data_type=data_type,
                use_mmap=use_mmap,
            )

        # With the packdata in hand, start parsing the individual spectral records.
        data_dict = {}
        for inhid in unique_inhid:
            # Pop here lets us delete this at the end (and hopefully let garbage
            # collection do it's job correctly).
            packdata, data_dtype, common_scale = packdata_dict.pop(inhid)
            packdata = packdata["packdata"].view(data_dtype)
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

                if common_scale:
                    temp_dict[hid] = {
                        "scale_fac": packdata[start_idx],
                        "data": packdata[(start_idx + 1) : end_idx],
                    }
                else:
                    data_arr = packdata[start_idx:end_idx]
                    temp_dict[hid] = {
                        "data": data_arr,
                        "flags": np.isnan(data_arr),
                        "weights": np.ones_like(data_arr),
                    }

            if apply_cal and common_scale:
                temp_dict = self._convert_raw_to_vis(temp_dict)
                temp_dict = self._apply_compass_solns(temp_dict)

            if np.all(chan_avg_arr == 1):
                if apply_cal:
                    pass
                elif scale_data and common_scale:
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

    def _write_cross_data(self, filepath=None, *, append_data=False, raise_err=True):
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
        int_dict, sp_dict = self.sp_data._generate_recpos_dict(
            data_dtype=NEW_VIS_DTYPE,
            data_nvals=2,
            pad_nvals=NEW_VIS_PAD,
            scale_data=True,
            hdr_fmt=NEW_VIS_HEADER,
            reindex=True,
        )

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
                    int_dict={inhid: int_dict[inhid]},
                    recpos_dict={inhid: sp_dict[inhid]},
                    data_dict=raw_dict,
                    data_type="cross",
                )
                packdata[inhid].tofile(file)

    def _write_auto_data(self, filepath=None, *, append_data=False, raise_err=True):
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
        int_dict, ac_dict = self.ac_data._generate_recpos_dict(
            data_dtype=NEW_AUTO_DTYPE,
            data_nvals=1,
            pad_nvals=0,
            scale_data=False,
            hdr_fmt=NEW_AUTO_HEADER,
            reindex=True,
        )

        # We can now open the file once, and write each array upon construction
        with open(
            os.path.join(filepath, "ach_read"), "ab+" if append_data else "wb+"
        ) as file:
            for inhid in int_dict:
                packdata = self._make_packdata(
                    int_dict={inhid: int_dict[inhid]},
                    recpos_dict={inhid: ac_dict[inhid]},
                    data_dict=self.auto_data,
                    data_type="auto",
                )
                packdata[inhid].tofile(file)

    def apply_tsys(self, *, invert=False, force=False, use_cont_det=None):
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
        use_cont_det : bool
            If set to True, use the continuum tsys data for calculating system
            temperatures. If set to False, the spectral tsys data will be applied if
            available. Default is True.
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

        if use_cont_det or (use_cont_det is None and self._tsys_use_cont_det):
            # Create a dictionary here to map antenna pair + integration time step with
            # a sqrt(tsys) value. Note that the last index here is the receiver number,
            # which technically has a different keyword under which the system
            # temperatures are stored.
            tsys_dict = {
                (idx, jdx, 0): tsys**0.5 if (tsys > 0 and tsys < 1e5) else 0.0
                for idx, jdx, tsys in zip(*self.eng_data[("inhid", "antenna", "tsys")])
            }
            tsys_dict.update(
                {
                    (idx, jdx, 1): tsys**0.5 if (tsys > 0 and tsys < 1e5) else 0.0
                    for idx, jdx, tsys in zip(
                        *self.eng_data[("inhid", "antenna", "tsys_rx2")]
                    )
                }
            )

            # now create a per-blhid SEFD dictionary based on antenna pair, integration
            # time step, and receiver pairing.
            normal_dict = {}
            for blhid, idx, jdx, kdx, ldx, mdx in zip(
                *self.bl_data[("blhid", "inhid", "iant1", "ant1rx", "iant2", "ant2rx")]
            ):
                try:
                    normal_dict[blhid] = (2.0 * self.jypk) * (
                        tsys_dict[(idx, jdx, kdx)] * tsys_dict[(idx, ldx, mdx)]
                    )
                except KeyError:
                    warnings.warn(
                        "No tsys for blhid %i found (%i-%i baseline, inhid %i). "
                        "Baseline record will be flagged." % (blhid, jdx, ldx, idx)
                    )

            if invert:
                for key, value in normal_dict.items():
                    if value != 0:
                        normal_dict[key] = 1.0 / value

            # Finally, multiply the individual spectral records by the SEFD values
            # that are in the dictionary.
            int_time_dict = dict(self.in_data.get_value(("inhid", "rinteg")))
            for sp_rec in self.sp_data:
                vis_dict = self.vis_data[sp_rec["sphid"]]
                n_sample = abs(sp_rec["fres"] * 1e6) * int_time_dict[sp_rec["inhid"]]
                try:
                    norm_val = normal_dict[sp_rec["blhid"]]
                    if norm_val == 0.0:
                        vis_dict["flags"][:] = True
                    else:
                        vis_dict["data"] *= norm_val
                        if invert:
                            vis_dict["weights"] *= (norm_val**2.0) / n_sample
                        else:
                            vis_dict["weights"] *= n_sample / (norm_val**2.0)
                except KeyError:
                    self.vis_data[sp_rec["sphid"]]["flags"][:] = True
        else:
            # The "wt" column is calculated as (integ time)/(T_DSB ** 2), but we want
            # units of Jy**-2. To do this, we just need to multiply by one of the
            # forward gain of the antenna (130 Jy/K for SMA) squared and the channel
            # width. The factor of 2**2 (4) arises because we need to convert T_DSB**2
            # to T_SSB**2. Note the 1e6 is there to convert fres from MHz to Hz.
            wt_arr = (
                self.sp_data["wt"]
                * abs(self.sp_data["fres"])
                * (1e6 * ((self.jypk * 2.0) ** (-2.0)))
            )

            # For data normalization, we used the "wt" but strip out the integration
            # time and take the inverse sqrt to get T_DSB, and then use the forward
            # gain (plus 2x for DSB -> SSB) to get values of Jy.
            norm_arr = np.zeros_like(wt_arr)
            norm_arr = np.reciprocal(
                self.sp_data["wt"], where=(wt_arr != 0), out=norm_arr
            )
            norm_arr = (
                self.jypk
                * 2.0
                * np.sqrt(
                    norm_arr
                    * self.in_data.get_value("rinteg", header_key=self.sp_data["inhid"])
                )
            )

            if invert:
                for arr in [norm_arr, wt_arr]:
                    arr = np.reciprocal(arr, where=(arr != 0), out=arr)

            for sphid, norm_val, wt_val in zip(self.sp_data["sphid"], norm_arr, wt_arr):
                vis_dict = self.vis_data[sphid]
                if norm_val == 0.0:
                    vis_dict["flags"][:] = True
                else:
                    vis_dict["data"] *= norm_val
                    vis_dict["weights"] *= wt_val

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
            if bool(flagval):
                self.vis_data[sphid]["flags"][:] = True

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
        for idx_arr, data_arr in check_list:
            if data_arr is None:
                # If not loaded, move along
                continue
            if sorted(idx_arr) != sorted(data_arr):
                # If we have a mismatch, we can leave ASAP
                return False

        # If you got to this point, it means that we've got agreement!
        return True

    def _downselect_data(self, *, select_vis=None, select_raw=None, select_auto=None):
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
        *,
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
            unload_vis=not load_vis, unload_raw=not load_raw, unload_auto=not load_auto
        )

        # If we are potentially downselecting data (typically used when calling select),
        # make sure that we actually have all the data we need loaded.
        if allow_downselect or (allow_downselect is None):
            if load_cross and not (self.vis_data is None and self.raw_data is None):
                try:
                    self._downselect_data(
                        select_vis=load_vis, select_raw=load_raw, select_auto=False
                    )
                    load_cross = False
                except MirMetaError:
                    if allow_downselect:
                        warnings.warn("Cannot downselect cross-correlation data.")

            if load_auto and self.auto_data is not None:
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
                "cross", scale_data=load_vis, use_mmap=use_mmap, read_only=read_only
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
                "auto", use_mmap=use_mmap, read_only=read_only
            )

    def unload_data(self, *, unload_vis=True, unload_raw=True, unload_auto=True):
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
            if self.vis_data is not None:
                for item in self.vis_data.values():
                    del item["data"]
                    del item["flags"]
                    del item["weights"]
                self.vis_data = None
            self._tsys_applied = False
        if unload_raw and self.raw_data is not None:
            for item in self.raw_data.values():
                del item["data"]
                del item["scale_fac"]
            self.raw_data = None
        if unload_auto and self.auto_data is not None:
            for item in self.auto_data.values():
                del item["data"]
                del item["flags"]
                del item["weights"]
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

        # There's one check to do up front, which is to make sure that there's data
        # that actually matches the integration headers (to avoid loading data that
        # does not exist on disk). We only run this on bl_data, since it's the smallest
        # metadata block that's specific to the visibilities
        # TODO: Add a similar check for the autos and ac_data
        if self._has_cross:
            where_list = [
                ("inhid", "eq", list(idict["cross"]["int_dict"]))
                for idict in self._file_dict.values()
            ]
            mask_update |= self.bl_data.set_mask(where=where_list, and_where_args=False)

        # Now start by cascading the filters up -- from largest metadata tables to the
        # smallest. First up, spec win -> baseline
        if not np.all(self.sp_data.get_mask()):
            mask_update |= self.bl_data._make_key_mask(self.sp_data)

        # Now do baseline -> antennas. Special handling required because of the
        # lack of a unique index key for this table.
        if self._has_auto or not np.all(self.bl_data.get_mask()):
            mask = self.eng_data._make_key_mask(
                self.bl_data,
                check_field=("iant1", "inhid"),
                set_mask=False,
                use_cipher=True,
            )
            mask |= self.eng_data._make_key_mask(
                self.bl_data,
                check_field=("iant2", "inhid"),
                set_mask=False,
                use_cipher=True,
            )

            if self._has_auto:
                mask |= self.eng_data._make_key_mask(
                    self.ac_data, set_mask=False, use_cipher=True
                )

            mask_update |= self.eng_data.set_mask(mask=mask)

        # Now antennas -> int
        if not np.all(self.eng_data.get_mask()):
            mask_update |= self.in_data._make_key_mask(self.eng_data)

        # And weather scan -> int
        if not np.all(self.we_data.get_mask()):
            mask_update |= self.in_data._make_key_mask(self.we_data, reverse=True)

        # We now cascade the masks downward. First up, int -> weather scan
        mask_update |= self.we_data._make_key_mask(self.in_data)

        # Next, do int -> baseline
        mask_update |= self.bl_data._make_key_mask(self.in_data, reverse=True)

        # Next, ant -> baseline. Again this requires a little extra special
        # handling, since eng_data doesn't have a unique header key.
        mask = self.bl_data._make_key_mask(
            self.eng_data,
            check_field=("iant1", "inhid"),
            set_mask=False,
            use_cipher=True,
            reverse=True,
        )
        mask &= self.bl_data._make_key_mask(
            self.eng_data,
            check_field=("iant2", "inhid"),
            set_mask=False,
            use_cipher=True,
            reverse=True,
        )
        mask_update |= self.bl_data.set_mask(mask=mask)

        # Finally, do baseline -> spec win for the crosses...
        mask_update |= self.sp_data._make_key_mask(self.bl_data, reverse=True)

        # ...and the autos.
        if self._has_auto:
            mask_update |= self.ac_data._make_key_mask(
                self.eng_data, use_cipher=True, reverse=True
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
        will clear out attributes related to the auto-correlations. It will also clear
        and stored masks, if present.
        """
        self._has_auto = False
        self.auto_data = None
        self.ac_data = MirAcData()
        self._ac_dict = None
        try:
            del self._metadata_attrs["ac_data"]
            for key in self._file_dict:
                del self._file_dict[key]["auto"]
            self._stored_masks = {}
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

    def save_mask(self, mask_name: str = None, overwrite=False):
        """
        Save masks for later use.

        This method will store selection masks, that can later be reloaded. This can
        be useful for temporarily down-selecting data for various operations (allowing
        one to restore the old selection mask after that is complete).

        Parameters
        ----------
        mask_name : str
            Name under which to store the mask. No default.
        overwrite : bool
            If set to False and attempting to save a mask whose name is already stored,
            an error is thrown. If set to True, and existing masks stored under the name
            given will be overwritten. Default is False.
        """
        if (mask_name in self._stored_masks) and not overwrite:
            raise ValueError(
                "There already exists a stored set of masks with the name %s, "
                "either change the name or set overwrite=True" % mask_name
            )

        if not isinstance(mask_name, str):
            raise ValueError("mask_name must be a string.")

        mask_dict = {}
        for name, attr in self._metadata_attrs.items():
            mask_dict[name] = attr._mask.copy()

        self._stored_masks[mask_name] = mask_dict

    def restore_mask(self, mask_name: str, run_update=True):
        """
        Restore stored masks.

        This method will restore selection masks that have been previously saved.

        Parameters
        ----------
        mask_name : str
            Name of the previously stored mask to restore.
        run_update : bool
            By default, the method will cross-check internal masks and, if needed, will
            unload or downselect the visibility/autocorrelation data to match the
            records selected. Setting this to False will bypass this check. This should
            only be used by advanced users, as this can produce unexpected behaviors.
        """
        if len(self._stored_masks) == 0:
            raise ValueError("No stored masks for this object.")

        if mask_name not in self._stored_masks:
            raise ValueError(
                "No stored set of masks with the name %s (options: %s)."
                % (mask_name, list(self._stored_masks))
            )

        mask_dict = self._stored_masks[mask_name]
        for name, attr in self._metadata_attrs.items():
            attr._mask = mask_dict[name].copy()

        if run_update:
            self._update_filter()

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
        self.ac_data._mask = np.ones(self.ac_data._size, dtype=bool)

        # Set up the header index for the object, and then construct the header key dict
        self.ac_data._data["achid"] = np.arange(1, self.ac_data._size + 1)
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
            ["blhid", "iband"], use_mask=False, assume_unique=True, return_index=True
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
        filepath=None,
        *,
        compass_soln=None,
        make_v3_compliant=None,
        old_vis_format=None,
        nchunks=None,
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
        compass_soln : str
            Optional argument, specifying the path of COMPASS-derived flagging and
            bandpass gains solutions, which are loaded into the object.
        make_v3_compliant : bool
            Convert the metadata required for filling a UVData object into a
            v3-compliant format. Only applicable if MIR file version is v1 or v2.
            Default behavior is to set this to True if the file version is v1 or v2.
        old_vis_format : bool
            Prior to the v1 data format, older visibility data was recorded with
            different header fields and endianness (big-endian versus the current
            little-endian format). If set to True, data are assumed to be in this "old"
            format, if set to False, data are assumed to be in the current format.
            Default behavior is to attempt to automatically determine old versus new
            based on metadata, and set this value accordingly.
        nchunks : int
            Only used if `has_auto=True` is set, this value is used to set the number
            of spectral windows in a single auto-correlation set. The default behavior
            is to set this value automatically depending on file version. This setting
            should only be used by advanced users in specialized circumstances.
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
        # If auto-defaults are turned on, we can use the codes information within the
        # file to determine a few things. Use this to automatically handle a few
        # different things that change between the major file versions. Note that this
        # is meant to be a stopgap solution.
        filever = None
        if None in [make_v3_compliant, old_vis_format, nchunks]:
            try:
                temp_codes = MirCodesData(filepath)
                filever = int(temp_codes["filever"][0])
            except MirMetaError:  # pragma: nocover
                # v1 is the only version without this particular code. Don't cover this
                # since it's a particular file version that we don't want to have to
                # add to the testing data depot.
                filever = 1

        if make_v3_compliant is None:
            # If v0, v1, or v2, we need to plug in the necessary metadata for making
            # the conversion to UVData feasible.
            make_v3_compliant = filever < 3

        if old_vis_format is None:
            # If this is a converted v0 file, from the ASIC era, with the old vis format
            old_vis_format = filever == 0

        if nchunks is None:
            # Newer datasets have a problem with the recorded nchunks value, so force
            # that to be 8 here for all more modern data.
            nchunks = 8 if (filever > 2) else None

        # These functions will read in the major blocks of metadata that get plugged
        # in to the various attributes of the MirParser object. Note that "_read"
        # objects contain the whole data set, while "_data" contains that after
        # filtering (more on that below).
        if has_auto:
            self._metadata_attrs["ac_data"] = self.ac_data
            self._has_auto = True
            self.ac_data._nchunks = nchunks
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
            int_dict, self._sp_dict = self.sp_data._generate_recpos_dict(
                data_dtype=OLD_VIS_DTYPE if old_vis_format else NEW_VIS_DTYPE,
                data_nvals=2,  # Real + imag values
                pad_nvals=OLD_VIS_PAD if old_vis_format else NEW_VIS_PAD,
                scale_data=True,  # Crosses are packed w/ a common exponent
                hdr_fmt=OLD_VIS_HEADER if old_vis_format else NEW_VIS_HEADER,
            )
            file_dict["cross"] = {
                "int_dict": int_dict,
                "filetype": "sch_read",
                "read_hdr_fmt": OLD_VIS_HEADER if old_vis_format else NEW_VIS_HEADER,
                "read_data_fmt": OLD_VIS_DTYPE if old_vis_format else NEW_VIS_DTYPE,
                "common_scale": True,
            }

        # If we need to update metadata for V3 compliance, do that now, since we fill
        # the acdata struct with fields added in V3.
        if make_v3_compliant:
            self._make_v3_compliant()

        if self._has_auto:
            filetype = "ach_read"
            old_auto_format = self.ac_data._old_format
            if old_auto_format:
                # If we have the old-style file we are working with, then we need to
                # do two things: first, clean up entries that don't actually have any
                # data in them (the old format recorded lots of blank data to disk),
                # and plug in some missing metadata.
                self._fix_acdata()
                filetype = "autoCorrelations"
            int_dict, self._ac_dict = self.ac_data._generate_recpos_dict(
                data_dtype=OLD_AUTO_DTYPE if old_auto_format else NEW_AUTO_DTYPE,
                data_nvals=1,  # Real-only vals
                pad_nvals=0,  # Autos have no padding, at least not currently.
                scale_data=False,  # Auto data has no common scaling
                hdr_fmt=OLD_AUTO_HEADER if old_vis_format else NEW_AUTO_HEADER,
            )

            file_dict["auto"] = {
                "int_dict": (
                    self.ac_data._old_format_int_dict if old_auto_format else int_dict
                ),
                "filetype": filetype,
                "read_hdr_fmt": OLD_AUTO_HEADER if old_auto_format else NEW_AUTO_HEADER,
                "read_data_fmt": OLD_AUTO_DTYPE if old_auto_format else NEW_AUTO_DTYPE,
                "common_scale": False,
            }

        self._file_dict = {filepath: file_dict}
        self.filepath = filepath

        # Finally, if we've specified a COMPASS solution, load that now as well.
        if compass_soln is not None:
            self.read_compass_solns(compass_soln)

        # Set/clear these to start
        self.vis_data = self.raw_data = self.auto_data = None
        self._tsys_applied = False

        # If requested, now we load up the visibilities.
        self.load_data(load_cross=load_cross, load_raw=load_raw, load_auto=load_auto)

    def write(
        self,
        filepath=None,
        *,
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
                data_nvals = 2 if (attr == "sp_data") else 1
                data_dtype = NEW_VIS_DTYPE if (attr == "sp_data") else NEW_AUTO_DTYPE
                scale_data = attr == "sp_data"

                mir_meta_obj = self._metadata_attrs[attr].copy()
                mir_meta_obj._recalc_dataoff(
                    data_dtype=data_dtype, data_nvals=data_nvals, scale_data=scale_data
                )
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
    def _rechunk_data(
        data_dict=None,
        chan_avg_arr=None,
        *,
        inplace=False,
        weight_data=True,
        norm_weights=True,
    ):
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
        weight_data : bool
            If True, data are weighted prior to averaging. Default is True.
        norm_weights : bool
            If True, will normalize the rechunked weights to account for the increased
            bandwidth of the individual channels -- needed for accurately calculating
            the "absolute" weights. Default is True, as most calls to this function
            are done before absolute calibration is applied (e.g., via `apply_tsys`).

        Returns
        -------
        new_vis_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `vis_dict`.
        """
        if data_dict is None:
            return

        new_data_dict = data_dict if inplace else {}

        for chan_avg, hkey in zip(chan_avg_arr, data_dict):
            # Pull out the dict that we need.
            vis_data = data_dict[hkey]

            # If there isn't anything to average, we can skip the heavy lifting
            # and just proceed on to the next record.
            if chan_avg == 1:
                if not inplace:
                    new_data_dict[hkey] = copy.deepcopy(vis_data)
                continue

            # Figure out which entries have values we want to used based on flags
            good_mask = ~vis_data["flags"].reshape((-1, chan_avg))
            data_arr = vis_data["data"].reshape(good_mask.shape)
            weight_arr = vis_data["weights"].reshape(good_mask.shape)

            # Sum across all of the channels now, tabulating the sum of all of the
            # weights (either all ones or whatever is in the weights spectrum).
            if weight_data:
                # Tabulate the weights, which are just summed across the channels
                temp_weights = np.sum(weight_arr, axis=1, where=good_mask, initial=0)
                temp_vis = np.sum(
                    (data_arr * weight_arr), axis=1, where=good_mask, initial=0
                )
                norm_vals = temp_weights
            else:
                temp_vis = np.sum(data_arr, axis=1, where=good_mask, initial=0)
                norm_vals = np.sum(good_mask, axis=1, dtype=np.float32)

                # The weights here are in Jy**-2, so take the reciprocal, sum,
                # reciprocal, and normalize (by the num of channels ** 2) to get what
                # the weights "should" be in the nominal Jy**-2 units.
                # variance of each channel (without accounting for)
                temp_weights = np.sum(
                    np.reciprocal(weight_arr, where=good_mask),
                    where=good_mask,
                    axis=1,
                    initial=0,
                )
                temp_weights = np.reciprocal(
                    temp_weights, where=(temp_weights != 0), out=temp_weights
                ) * (norm_vals**2)

            # Now take the sum of all valid visibilities, multiplied by the
            # normalization factor.
            temp_vis = np.divide(
                temp_vis, norm_vals, where=(norm_vals != 0), out=temp_vis
            )

            # If weighting has not already been applied (i.e., just "nsamples") then
            # we need to do a bit extra accounting here to track the fact that we've
            # upped the bandwidth in this particular channel (and therefore the max
            # value this should take on is 1.0).
            if norm_weights:
                temp_weights *= 1.0 / chan_avg

            # Finally, plug the spectrally averaged data back into the dict, flagging
            # channels with no valid data.
            new_data_dict[hkey] = {
                "data": temp_vis,
                "flags": temp_weights == 0,
                "weights": temp_weights,
            }

        return new_data_dict

    @staticmethod
    def _rechunk_raw(
        raw_dict=None, chan_avg_arr=None, *, inplace=False, return_vis=False
    ):
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
                    MirParser._convert_raw_to_vis({0: sp_raw}), [chan_avg], inplace=True
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

    def rechunk(self, chan_avg=None):
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
                self._rechunk_data(
                    self.vis_data,
                    chan_avg_arr,
                    inplace=True,
                    norm_weights=(not self._tsys_applied),
                )
                self._rechunk_raw(self.raw_data, chan_avg_arr, inplace=True)
            else:
                self._rechunk_data(self.auto_data, chan_avg_arr, inplace=True)

    def __add__(self, other, *, merge=None, overwrite=None, force=False, inplace=False):
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
                        getattr(other, item), merge=merge, overwrite=overwrite
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
            # metadata objects and checking for any matches (ignoring masks).
            bad_attr = []
            update_dict = {}
            for item in self._metadata_attrs:
                this_attr = self._metadata_attrs[item]
                other_attr = other._metadata_attrs[item]
                if this_attr.__eq__(other_attr, comp_mask=False):
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
            new_obj._stored_masks = {}

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
                        "read_hdr_fmt": copy.deepcopy(datatype_dict["read_hdr_fmt"]),
                        "read_data_fmt": copy.deepcopy(datatype_dict["read_data_fmt"]),
                        "common_scale": datatype_dict["common_scale"],
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

            for attr, idict in recpos_list:
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
                "dropping auto-correlation data and metadata from the combined object. "
                "Note that this will clear any stored masks."
            )
            new_obj._clear_auto()
        elif self.auto_data is None or other.auto_data is None:
            new_obj.auto_data = None
        else:
            new_obj.auto_data.update(copy.deepcopy(other_auto))

        return new_obj

    def __iadd__(self, other, *, merge=None, overwrite=False, force=False):
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
        *,
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
        alias_dict = {"ant": "antenna", "ant1": "tel1", "ant2": "tel2"}

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
                mask=mask, reset=reset, and_mask=and_mask, use_mask=False
            )

        # Now that we've screened the data that we want, update the object appropriately
        self._update_filter(update_data=update_data)

    def _read_compass_solns(self, filename=None):
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

        has_cross_pol = np.all(
            np.not_equal(
                self.bl_data.get_value("ant1rx", index=sp_bl_map),
                self.bl_data.get_value("ant2rx", index=sp_bl_map),
            )
        )

        # MATLAB v7.3 format uses HDF5 format, so h5py here ftw!
        with h5py.File(filename, "r") as file:
            # First, pull out the bandpass solns, and the associated metadata. Note that
            # the real/imag values here are split to make it easier for grabbing the
            # data cleanly w/o worries about endianness/recasting.
            ant_arr = np.array(file["antArr"][0]).astype(int)  # Antenna number
            rx_arr = np.array(file["rx1Arr"][0]).astype(int)  # Receiver (0=RxA, 1=RxB)
            sb_arr = np.array(file["sbArr"][0]).astype(int)  # Sideband (0=LSB, 1=USB)
            chunk_arr = np.array(file["winArr"][0]).astype(int)  # Spectral win #
            bp_arr = (
                np.array(file["reBandpassArr"]) + (1j * np.array(file["imBandpassArr"]))
            ).astype(
                np.complex64
            )  # BP gains (3D array)
            sefd_arr = np.array(file["sefdArr"]) ** 2.0

            # Parse out the bandpass solutions for each antenna, pol/receiver, and
            # sideband-chunk combination.
            for idx, ant in enumerate(ant_arr):
                for jdx, (rx, sb, chunk) in enumerate(zip(rx_arr, sb_arr, chunk_arr)):
                    cal_data = bp_arr[idx, jdx]
                    cal_flags = (cal_data == 0.0) | ~np.isfinite(cal_data)
                    sefd_data = sefd_arr[idx, jdx]
                    sefd_flags = (sefd_data == 0.0) | ~np.isfinite(sefd_data)

                    cal_data[cal_flags] = 1.0
                    sefd_data[sefd_flags] = 0.0

                    bandpass_gains[(ant, rx, sb, chunk)] = {
                        "cal_data": cal_data,
                        "cal_flags": cal_flags,
                        "sefd_data": sefd_data,
                        "sefd_flags": sefd_flags,
                    }

            # Once we divvy-up the solutions, plug them back into the dict that
            # we will pass back to the user.
            compass_soln_dict["bandpass_gains"] = bandpass_gains
            idx_compare = 2 if has_cross_pol else 1
            bp_gains_corr = {}
            for key1, dict1 in bandpass_gains.items():
                for key2, dict2 in bandpass_gains.items():
                    if key1[idx_compare:] != key2[idx_compare:]:
                        continue

                    # Put together bandpass gains for the visibilities
                    cal_soln = np.zeros(
                        len(dict1["cal_data"]),
                        dtype=np.float32 if key1 == key2 else np.complex64,
                    )
                    cal_flags = dict1["cal_flags"] | dict2["cal_flags"]

                    # Split the processing here based on autos vs crosses
                    if key1 == key2:
                        cal_soln = np.reciprocal(
                            abs(dict1["cal_data"] * dict2["cal_data"]),
                            where=~cal_flags,
                            out=cal_soln,
                        )
                    else:
                        cal_soln = np.reciprocal(
                            dict1["cal_data"] * np.conj(dict2["cal_data"]),
                            where=~cal_flags,
                            out=cal_soln,
                        )

                    # Now generate re-weighting solns based on per-chanel SEFD
                    # measurements calculated by COMPASS.
                    weight_soln = np.zeros(dict1["sefd_data"].shape, dtype=np.float32)
                    weight_flags = dict1["sefd_flags"] | dict2["sefd_flags"]
                    weight_soln = np.reciprocal(
                        dict1["sefd_data"] * dict2["sefd_data"],
                        where=~weight_flags,
                        out=weight_soln,
                    )

                    new_key = key1[:2] + key2
                    bp_gains_corr[new_key] = {
                        "cal_soln": cal_soln,
                        "cal_flags": cal_flags,
                        "weight_soln": weight_soln,
                        "weight_flags": weight_flags,
                    }

            compass_soln_dict["bp_gains_corr"] = bp_gains_corr

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
                check = np.where(np.isclose(mjd, mjd_mir, atol=atol, rtol=0))[0]
                index_dict[idx] = None if (len(check) == 0) else inhid_arr[check[0]]

            # Pull out some metadata here for parsing the individual solutions
            flags_arr = np.array(file["flagArr"])  # Per-sphid flags
            sflags_arr = np.array(file["staticFlagArr"])  # "For all time" flags
            ant1_arr = np.array(file["ant1Arr"][0]).astype(int)  # 1st ant in baseline
            rx1_arr = np.array(file["rx1Arr"][0]).astype(int)  # Receiver/pol of 1st ant
            ant2_arr = np.array(file["ant2Arr"][0]).astype(int)  # 2nd ant in baseline
            rx2_arr = np.array(file["rx2Arr"][0]).astype(int)  # Receiver/pol of 2nd ant
            sb_arr = np.array(file["sbArr"][0]).astype(int)  # Sideband (0=LSB, 1=USB)
            chunk_arr = np.array(file["winArr"][0]).astype(int)  # Spectral win number

            # Begin unpacking the "static" flags, which are antenna based and are
            # persistent across the entire track. Note that the two loops here are used
            # to match the indexing scheme of the flags (so the slowest loop iterates
            # on the outer-most axis of the array).
            temp_flags = {}
            for idx, ant in enumerate(ant_arr):
                for jdx, (rx, sb, chunk) in enumerate(zip(rx1_arr, sb_arr, chunk_arr)):
                    temp_flags[(ant, rx, sb, chunk)] = sflags_arr[idx, jdx]

            # Expand out the flags to produce baseline-based masks, which will be what
            # is actually used with the visibility data.
            static_flags = {}
            idx_compare = 2 if has_cross_pol else 1
            for key1, flags1 in temp_flags.items():
                for key2, flags2 in temp_flags.items():
                    if key1[idx_compare:] != key2[idx_compare:]:
                        continue

                    new_key = key1[:2] + key2
                    static_flags[new_key] = flags1 | flags2

            # Once the wide flags dict is built, plug it back into the main dict.
            compass_soln_dict["static_flags"] = static_flags

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

    def read_compass_solns(self, filename=None, load_flags=True, load_bandpass=True):
        """
        Read in COMPASS-formatted bandpass and flagging solutions.

        Reads in an HDF5 file containing the COMPASS-derived flags and gains tables.
        These solutions are applied as the data are read in (when calling `load_data`).

        Parameters
        ----------
        filename : str
            Name of the file containing the COMPASS flags and gains solutions.
        load_flags : bool
            If set to True, the COMPASS flags will be read into the MirParser object.
            Default is True.
        load_gains : bool
            If set to True, the COMPASS gains will be read into the MirParser object.
            Default is True.

        Raises
        ------
        UserWarning
            If the COMPASS solutions do not appear to overlap in time with that in
            the MirParser object.
        """
        if not load_flags and not load_bandpass:
            # You say you want solutions, but you don't want any solutions?
            # You're tearing me apart, Lisa! (no-op return)
            return
        if not (self.vis_data is None and self.raw_data is None):
            raise ValueError(
                "Cannot call read_compass_solns when data have already been loaded, "
                "call unload_data first in order to resolve this error."
            )

        compass_soln_dict = self._read_compass_solns(filename)
        if load_flags:
            self._compass_sphid_flags = compass_soln_dict["sphid_flags"]
            self._compass_static_flags = compass_soln_dict["static_flags"]

        if load_bandpass:
            self._compass_bp_soln = compass_soln_dict["bp_gains_corr"]

        self._has_compass_soln = True

    def _apply_compass_solns(self, vis_data=None):
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
        vis_data : dict
            Dictionary containing visibility data, to which flags/bandpass solutions
            will be applied. Note that the format of this dict should match that of
            the `vis_data` attribute (see `MirParser._convert_raw_to_vis` for a more
            complete description.).

        Returns
        -------
        vis_data : dict
            Dictionary containing the flagged/calibrated data. Note that this will be
            the same object as the input parameter by the same name.

        Raises
        ------
        ValueError
            If visibility data are not loaded (not that its not enough to have raw data
            loaded -- that needs to be converted to "normal" vis data).
        """
        # We can grab the keys of the vis_data dict to match to metadata records
        sphid_arr = list(vis_data)

        # Use this to map certain per-blhid values to individual sphid entries.
        blhid_arr = self.sp_data.get_value("blhid", header_key=sphid_arr)

        # Now grab all of the metadata we want for processing the spectral records
        ant1_arr = self.bl_data.get_value("iant1", header_key=blhid_arr)  # Ant 1 Number
        rx1_arr = self.bl_data.get_value("ant1rx", header_key=blhid_arr)  # 0:X|L 1:Y|R
        ant2_arr = self.bl_data.get_value("iant2", header_key=blhid_arr)  # Ant 2 Number
        rx2_arr = self.bl_data.get_value("ant2rx", header_key=blhid_arr)  # 0:X|L 1:Y|R
        chunk_arr = self.sp_data.get_value("corrchunk", header_key=sphid_arr)  # SPW#
        sb_arr = self.bl_data.get_value("isb", header_key=blhid_arr)  # SB| 0:LSB 1:USB

        if self._compass_bp_soln is not None:
            # Let's grab the bandpass solns upfront before we iterate through
            # all of the individual spectral records.
            bp_soln = self._compass_bp_soln

            for sphid, sb, ant1, rx1, ant2, rx2, chunk in zip(
                sphid_arr, sb_arr, ant1_arr, rx1_arr, ant2_arr, rx2_arr, chunk_arr
            ):
                try:
                    # If we have calculated the bandpass soln before, grab it now.
                    cal_soln = bp_soln[(ant1, rx1, ant2, rx2, sb, chunk)]
                except KeyError:
                    # Flag the soln if either ant1 or ant2 solns are bad.
                    cal_soln = {
                        "cal_soln": 1.0,
                        "cal_flags": True,
                        "weight_soln": 0.0,
                        "weight_flags": True,
                    }
                finally:
                    # One way or another, we should have a set of gains solutions that
                    # we can apply now (flagging the data where appropriate).
                    vis_data[sphid]["data"] *= cal_soln["cal_soln"]
                    vis_data[sphid]["weights"] *= cal_soln["weight_soln"]
                    vis_data[sphid]["flags"] |= (
                        cal_soln["cal_flags"] | cal_soln["weight_flags"]
                    )

        if not (
            self._compass_sphid_flags is None or self._compass_static_flags is None
        ):
            # For the sake of reading/coding, let's assign the two catalogs of flags
            # to their own variables, so that we can easily call them later.
            sphid_flags = self._compass_sphid_flags
            static_flags = self._compass_static_flags

            for idx, sphid in enumerate(sphid_arr):
                # Now we'll step through each spectral record that we have to process.
                # Note that we use unpackbits because MATLAB/HDF5 doesn't seem to have
                # a way to store single-bit values, and so the data are effectively
                # compressed into uint8, which can be reinflated via unpackbits.
                try:
                    # If we have a flags entry for this sphid, then go ahead and apply
                    # them to the flags table for that spectral record.
                    vis_data[sphid]["flags"] |= np.unpackbits(sphid_flags[sphid]).view(
                        bool
                    )
                except KeyError:
                    # If no key is found, then we want to try and use the "broader"
                    # flags to mask out the data that's associated with the given
                    # antenna-receiver combination (for that sideband and spec window).
                    # Note that if we do not have an entry here, something is amiss.
                    try:
                        vis_data[sphid]["flags"] |= np.unpackbits(
                            static_flags[
                                (
                                    ant1_arr[idx],
                                    rx1_arr[idx],
                                    ant2_arr[idx],
                                    rx2_arr[idx],
                                    sb_arr[idx],
                                    chunk_arr[idx],
                                )
                            ]
                        ).view(bool)
                    except KeyError:
                        # If we _still_ have no key, that means that this data was
                        # not evaluated by COMPASS, and for now we will default to
                        # not touching the flags.
                        pass
        return vis_data

    @staticmethod
    def _generate_chanshift_kernel(
        chan_shift=None, kernel_type=None, *, alpha_fac=-0.5, tol=1e-3
    ):
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
                np.array([1 - fine_shift, fine_shift], dtype=np.float32),
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
    def _chanshift_vis(
        vis_dict=None, shift_tuple_list=None, *, flag_adj=True, inplace=False
    ):
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
            new_weights = np.empty_like(sp_vis["weights"])

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
                    new_weights[:coarse_shift] = sp_vis["weights"][-coarse_shift:]
                    new_vis[coarse_shift:] = 0.0
                    new_flags[coarse_shift:] = True
                    new_weights[coarse_shift:] = 0.0
                else:
                    new_vis[coarse_shift:] = sp_vis["data"][:-coarse_shift]
                    new_flags[coarse_shift:] = sp_vis["flags"][:-coarse_shift]
                    new_weights[coarse_shift:] = sp_vis["weights"][:-coarse_shift]
                    new_vis[:coarse_shift] = 0.0
                    new_flags[:coarse_shift] = True
                    new_weights[:coarse_shift] = 0.0
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
                temp_weights = sp_vis["weights"][l_clip:r_clip].copy()
                temp_weights[sp_vis["flags"][l_clip:r_clip]] = (
                    np.float32(np.nan) if flag_adj else np.float32(0.0)
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
                new_weights[l_edge:r_edge] = np.convolve(
                    temp_weights, shift_kernel, "valid"
                )

                # Flag out the values beyond the outer bounds
                new_vis[:l_edge] = new_vis[r_edge:] = (
                    np.complex64(np.nan) if flag_adj else np.complex64(0.0)
                )
                new_weights[:l_edge] = new_weights[r_edge:] = (
                    np.float32(np.nan) if flag_adj else np.float32(0.0)
                )

                # Finally, regenerate the flags array for the dict entry.
                if flag_adj:
                    new_flags = np.isnan(new_vis)
                    new_vis[new_flags] = new_weights[new_flags] = 0.0
                else:
                    new_flags = np.zeros_like(sp_vis["flags"])
                    new_flags[:l_edge] = new_flags[r_edge:] = True
            # Update our dict with the new values for this sphid
            new_vis_dict[sphid] = {
                "data": new_vis,
                "flags": new_flags,
                "weights": new_weights,
            }

        return new_vis_dict

    @staticmethod
    def _chanshift_raw(
        raw_dict=None,
        shift_tuple_list=None,
        *,
        flag_adj=True,
        inplace=False,
        return_vis=False,
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
        *,
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

    def _make_v3_compliant(self):
        """
        Update MIR metadata for export to UVData.

        This is an internal helper function, not meant to be called by users. This
        function will modify or otherwise fill metadata in fields that were not
        populated in MIR file versions < 3, in order to make in minimally compliant
        with what the Mir.read method needs for populating a UVData object. Only data
        sets recorded prior to 2020 need these modifications.
        """
        if "filever" in self.codes_data.get_code_names():
            if self.codes_data["filever"][0] != "2":
                # If the file version is already >= 3.0, then this is basically a no-op
                return

        warnings.warn(
            "Pre v.3 MIR file format detected, modifying metadata to make in minimally "
            "compliant for reading in with pyuvdata. Note that this may cause spurious "
            "warnings about per-baseline records varying when filling a UVData object."
        )

        from datetime import datetime

        from astropy.time import Time

        from .. import get_telescope
        from .. import utils as uvutils

        # First thing -- we only want modern (i.e., SWARM) data, since the older (ASIC)
        # data is not currently supported by the data handling tools, due to changes
        # in the underlying file format.
        # if swarm_only:
        #     self.select(where=("correlator", "eq", 1))
        # Get SMA coordinates for various data-filling stuff
        sma_lat, sma_lon, sma_alt = get_telescope("SMA").telescope_location_lat_lon_alt

        # in_data updates: mjd, lst, ara, adec
        # First sort out the time stamps using the day reference inside codes_data, and
        # then adding the delta from midnight (dhrs)
        mjd_day_dict = self.codes_data["ref_time"]
        for key, value in mjd_day_dict.items():
            if isinstance(value, str):
                mjd_day_dict[key] = Time(
                    datetime.strptime(value, "%b %d, %Y"), scale="utc"
                ).mjd

        mjd_arr = (self.in_data["dhrs"] / 24.0) + np.array(
            [mjd_day_dict[idx] for idx in self.in_data["iref_time"]]
        )

        # Tally the JD dates, since that's used for various helper functions
        jd_arr = Time(mjd_arr, format="mjd", scale="utc").jd

        # Also, convert MJD back into the expected TT timescale
        mjd_arr = Time(mjd_arr, format="mjd", scale="utc").tt.mjd

        # Calculate the LST at the time of obs
        lst_arr = (12.0 / np.pi) * uvutils.get_lst_for_time(
            jd_array=jd_arr,
            latitude=np.rad2deg(sma_lat),
            longitude=np.rad2deg(sma_lon),
            altitude=sma_alt,
        )

        # Finally, calculate the apparent coordinates based on what we have in the data
        app_ra, app_dec = uvutils.calc_app_coords(
            lon_coord=self.in_data["rar"],
            lat_coord=self.in_data["decr"],
            time_array=jd_arr,
            telescope_loc=(sma_lat, sma_lon, sma_alt),
        )

        # Update the fields accordingly
        self.in_data["mjd"] = mjd_arr
        self.in_data["lst"] = lst_arr
        self.in_data["ara"] = app_ra
        self.in_data["adec"] = app_dec

        # bl_data updates: ant1rx, ant2rx, u, v, w
        # First, update the antenna receiver if these values are unfilled (true in some
        # earlier tracks, no version demarcation notes it).
        if np.all(self.bl_data["ant1rx"] == 0) or np.all(self.bl_data["ant2rx"] == 0):
            ipol = self.bl_data["ipol"]
            irec = self.bl_data["irec"]

            ant1list = [1, 2] if any(ipol) else [2, 3]
            ant2list = [1, 3] if any(ipol) else [2, 3]
            checklist = ipol if any(ipol) else irec
            self.bl_data["ant1rx"] = np.isin(checklist, ant1list).astype("<i2")
            self.bl_data["ant2rx"] = np.isin(checklist, ant2list).astype("<i2")

        # Next, the old data had uvw values calculated in wavelengths by _sideband_, so
        # we need to update those. Note we only have to check ant1rx here because if
        # ant1rx != ant1rx2, this is a pol track, and the tunings (and scaling) are
        # identical anyways.
        rx_idx = self.bl_data["ant1rx"]
        sb_idx = self.bl_data["isb"]

        # These isb and ant1rx should _only_ be either 0 or 1. This should never happen
        # unless the data are corrupt in some substantial way.
        assert np.all(np.isin(sb_idx, [0, 1])), "Bad SB index values detected."
        assert np.all(np.isin(rx_idx, [0, 1])), "Bad RX index values detected."

        u_vals = self.bl_data["u"]
        v_vals = self.bl_data["v"]
        w_vals = self.bl_data["w"]

        # Antenna positions are always stored in meters, so we can use this to get the
        # total distance between antennas. Note that we have to do this b/c of some
        # ambiguity of which frequency exactly the uvw values are calculated at.
        ant_dict = {}
        for ant1, pos1 in zip(*self.antpos_data[("antenna", "xyz_pos")]):
            temp_dict = {}
            for ant2, pos2 in zip(*self.antpos_data[("antenna", "xyz_pos")]):
                temp_dict[ant2] = ((pos1 - pos2) ** 2.0).sum() ** 0.5
            ant_dict[ant1] = temp_dict

        exp_bl_length = np.array(
            [ant_dict[idx][jdx] for idx, jdx in zip(*self.bl_data[["iant1", "iant2"]])]
        )
        meas_bl_length = ((u_vals**2) + (v_vals**2) + (w_vals**2)) ** 0.5

        # Update the uvws by a scale-factor determined by the ratio of the known total
        # lengths between antennas versus their calculated uvws. Note that the scale
        # factor should be the same for all baselines in each receiver-sideband
        # combination, so for the sake of robustness we use the median across all
        # baselines to calculate said scale factor.
        for idx in range(2):
            for jdx in range(2):
                mask = (rx_idx == idx) & (sb_idx == jdx)
                if np.any(mask):
                    scale_fac = np.median(exp_bl_length[mask] / meas_bl_length[mask])
                    u_vals[mask] *= scale_fac
                    v_vals[mask] *= scale_fac
                    w_vals[mask] *= scale_fac

        self.bl_data["u"] = u_vals
        self.bl_data["v"] = v_vals
        self.bl_data["w"] = w_vals
