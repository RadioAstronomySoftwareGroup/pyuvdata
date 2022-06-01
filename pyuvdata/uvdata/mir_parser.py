# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Module for low-level interface to MIR files.

This module extracts data types associated with MIR files.

"""
import numpy as np
import os
import copy
import warnings
import h5py
from functools import partial

__all__ = ["MirParser"]

# MIR structure definitions. Note that because these are all binaries, we need to
# specify the endianness so that we don't potentially muck that on different machines

# in_read is the per-integration header meta information.
in_dtype = np.dtype(
    [
        # Track id number, set to the Project ID (different from SMA Project Code)
        ("traid", np.int32),
        # Integration header ID number
        ("inhid", np.int32),
        # Scan number (usually same as inhid)
        ("ints", np.int32),
        # Azimuth of target from array center (deg)
        ("az", np.float32),
        # Elevation of target from array center (deg)
        ("el", np.float32),
        # Hour angle of target at array center (hours)
        ("ha", np.float32),
        # Time code (matched to ut in codes_read)
        ("iut", np.int16),
        # Date code (matched to ref_time in codes_read)
        ("iref_time", np.int16),
        # Avg time (in TT) at scan midpoint (hours)
        ("dhrs", np.float64),
        # Radial velocity of target (catalog value, km/s)
        ("vc", np.float32),
        # X-component of unit vector pointing towards source (from array center).
        ("sx", np.float64),
        # Y-component of unit vector pointing towards source (from array center).
        ("sy", np.float64),
        # Z-component of unit vector pointing towards source (from array center).
        ("sz", np.float64),
        # Integration time of scan (seconds)
        ("rinteg", np.float32),
        # Project ID Number
        ("proid", np.int32),
        # Source ID number
        ("souid", np.int32),
        # Source ID code (matched to source in codes_read)
        ("isource", np.int16),
        # Radial velocity code (matched to vrad in codes_read)
        ("ivrad", np.int16),
        # Offset in RA/Cross-Dec, used in mosaics (arcsec)
        ("offx", np.float32),
        # Offset in Dec, used in mosaics (arcsec)
        ("offy", np.float32),
        # Source RA Code (matched to ra in codes_read)
        ("ira", np.int16),
        # Source Dec Code (matched to dec in codes_read)
        ("idec", np.int16),
        # Catalog RA (radians)
        ("rar", np.float64),
        # Catalog Dec (radians)
        ("decr", np.float64),
        # Epoch value (Julian years, typically 2000.0)
        ("epoch", np.float32),
        # Angular diameter of source (arcsec)
        ("size", np.float32),
        # RA of velocity reference position (rad)
        ("vrra", np.float32),
        # Dec of velocity reference position (rad)
        ("vrdec", np.float32),
        # LAST at array center (hours)
        ("lst", np.float32),
        # Project ID code (matched to projectid in codes_read)
        ("iproject", np.int16),
        # Tile position of pointing (used sometimes in mosaics)
        ("tile", np.int16),
        # Obs mode (bitwise flag, not used yet)
        ("obsmode", np.uint8),
        # Obs flags (bitwise flag, not used yet)
        ("obsflag", np.uint8),
        # Spare value, always 0
        ("spareshort", np.int16),
        # Spare value, always 0
        ("spareint6", np.int32),
        # RxA YIG Frequency, sometimes used for flagging (GHz)
        ("yIGFreq1", np.float64),
        # RxB YIG Frequency, sometimes used for flagging (GHz)
        ("yIGFreq2", np.float64),
        # Source flux, in known (Jy)
        ("sflux", np.float64),
        # Apparent RA at array center (rad)
        ("ara", np.float64),
        # Apparent Dec at array center (rad)
        ("adec", np.float64),
        # Modified Julian Date (TT scale; days)
        ("mjd", np.float64),
    ]
).newbyteorder("little")

# eng_read records the per-antenna, per-integration metadata. These data are not
# typically used during data processing, but can be helpful in identifying bad or
# otherwise suspect data.
eng_dtype = np.dtype(
    [
        # Antenna number, should match iant in bl_read
        ("antenna", np.int32),
        # Pad number that the antenna is sitting on
        ("padNumber", np.int32),
        # Whether or not antenna was in the project (0 = offline, 1 = online)
        ("antennaStatus", np.int32),
        # Whether or not antenna was tracking (0 = offline, 1 = online)
        ("trackStatus", np.int32),
        # Whether or not antenna was online (0 = offline, 1 = online)
        ("commStatus", np.int32),
        # Integration header ID
        ("inhid", np.int32),
        # Scan number (usually same as inhid)
        ("ints", np.int32),
        # Avg time (in TT) at scan midpoint (hours)
        ("dhrs", np.float64),
        # Hour angle of target at antenna position (hours)
        ("ha", np.float64),
        # LAST at antenna position (hours)
        ("lst", np.float64),
        # Pointing model correction in Az for antenna (arcsec)
        ("pmdaz", np.float64),
        # Pointing model correction in El for antenna (arcsec)
        ("pmdel", np.float64),
        # Tilt measurement of antenna in the direction of antenna Azimuth (arcsec)
        ("tiltx", np.float64),
        # Tilt measurement of antenna in the cross-direction of antenna Azimuth (arcsec)
        ("tilty", np.float64),
        # Actual azimuth of the antenna (deg)
        ("actual_az", np.float64),
        # Actual elevation of the antenna (deg)
        ("actual_el", np.float64),
        # Pointing offset of the antenna in Az (arcsec)
        ("azoff", np.float64),
        # Pointing offset of the antenna in El (arcsec)
        ("eloff", np.float64),
        # RMS tracking error of the antenna in azimuth (arcsec)
        ("az_tracking_error", np.float64),
        # RMS tracking error of the antenna in elevation (arcsec)
        ("el_tracking_error", np.float64),
        # Estimated refraction for the antenna given weather conditions (arcsec)
        ("refraction", np.float64),
        # Secondary x-position (left-right) relative to mount (mm)
        ("chopper_x", np.float64),
        # Secondary y-position (up-down) relative to mount (mm)
        ("chopper_y", np.float64),
        # Secondary z-position (toward-away) relative to mount (mm)
        ("chopper_z", np.float64),
        # Secondary tilt angle relative to mount (arcsec)
        ("chopper_angle", np.float64),
        # System temperature for RxA/rx1 for the antenna
        ("tsys", np.float64),
        # System temperature for RxB/rx2 for the antenna
        ("tsys_rx2", np.float64),
        # Ambient load temperature of the antenna
        ("ambient_load_temperature", np.float64),
    ]
).newbyteorder("little")

# sp_read records the per-baseline, per-time record metadata.
bl_dtype = np.dtype(
    [
        # Baseline header ID
        ("blhid", np.int32),
        # Integration header ID, matched in in_read
        ("inhid", np.int32),
        # Sidebad code (matched to sb in codes_read; usually 0=LSB, 1=USB)
        ("isb", np.int16),
        # Polariztion code (matched to pol in codes_read)
        ("ipol", np.int16),
        # Ant1 receiver number (0 = RxA, 1 = RxB)
        ("ant1rx", np.int16),
        # Ant2 receiver number (0 = RxA, 1 = RxB)
        ("ant2rx", np.int16),
        # Pointing status (1 = offset pointing, 0 = target at pri beam center)
        ("pointing", np.int16),
        # Reciever code (matched to rec in codes_read)
        ("irec", np.int16),
        # u coordinate for the baseline (meters)
        ("u", np.float32),
        # v coordinate for the baseline (meters)
        ("v", np.float32),
        # w coordinate for the baseline (meters)
        ("w", np.float32),
        # uv-distance for the baseline (meters)
        ("prbl", np.float32),
        # Coherence of the baseline (not used, between 0 and 1)
        ("coh", np.float32),
        # Avg time (in TT) at baseline-scan midpoint (hours)
        ("avedhrs", np.float64),
        # Average amplitude across the baseline-sideband
        ("ampave", np.float32),
        # Average phase across the baseline-sideband
        ("phaave", np.float32),
        # Baseline number
        ("blsid", np.int32),
        # Antenna number of the first ant in the baseline pair
        ("iant1", np.int16),
        # Antenna number of the second ant in the baseline pair
        ("iant2", np.int16),
        # Index position for tsys data of ant1 in tsys_read (not used)
        ("ant1TsysOff", np.int32),
        # Index position for tsys data of ant2 in tsys_read (not used)
        ("ant2TsysOff", np.int32),
        # Baseline code (matched to blcd in codes_read)
        ("iblcd", np.int16),
        # East-west baseline length (meters)
        ("ble", np.float32),
        # North-south baseline length (meters)
        ("bln", np.float32),
        # Up-down baseline length (meters)
        ("blu", np.float32),
        # Spare value, always 0
        ("spareint1", np.int32),
        # Spare value, always 0
        ("spareint2", np.int32),
        # Spare value, always 0
        ("spareint3", np.int32),
        # Spare value, always 0
        ("spareint4", np.int32),
        # Spare value, always 0
        ("spareint5", np.int32),
        # Spare value, always 0
        ("spareint6", np.int32),
        # Center freq for ampave, phaave (GHz)
        ("fave", np.float64),
        # Bandwidth for ampave, phaave (MHz)
        ("bwave", np.float64),
        # Average weight for ampave, phaave (Jy**-2)
        ("wtave", np.float64),
        # Spare value, always 0
        ("sparedbl4", np.float64),
        # Spare value, always 0
        ("sparedbl5", np.float64),
        # Spare value, always 0
        ("sparedbl6", np.float64),
    ]
).newbyteorder("little")

# sp_read records the per-spectral record (per-band, per-time, per-baseline) metadata.
sp_dtype = np.dtype(
    [
        # Spectral header ID number
        ("sphid", np.int32),
        # Baseline header ID number, matched to entry in bl_read
        ("blhid", np.int32),
        # Integration header ID number, matched to entry in in_read
        ("inhid", np.int32),
        # Gain code (matched to gq in codes_read)
        ("igq", np.int16),
        # Passband code (matched to pq in codes_read)
        ("ipq", np.int16),
        # Band code (matched to band in codes_read, ususally equal to corrchunk)
        ("iband", np.int16),
        # Polarization state code (matched to pstate in codes_read, not used)
        ("ipstate", np.int16),
        # Opacity at 225 GHz (nepers)
        ("tau0", np.float32),
        # Velocity at source restframe at band center (km/s)
        ("vel", np.float64),
        # Velocity resolution at source restframe at band center (km/s)
        ("vres", np.float32),
        # Sky frequency at band center (GHz)
        ("fsky", np.float64),
        # Channel resolution (MHz)
        ("fres", np.float32),
        # Gunn frequency (GHz)
        ("gunnLO", np.float64),
        # Cabin BDA LO frequency (GHz, not used anymore)
        ("cabinLO", np.float64),
        # Second downconverter frequency (GHz, not used anymore)
        ("corrLO1", np.float64),
        # Final downconverter frequency (GHz)
        ("corrLO2", np.float64),
        # Integration time for spw-baseline-scan (s)
        ("integ", np.float32),
        # Weights for the spectral record (sec / tssb ** 2)
        ("wt", np.float32),
        # Bitwise flagging for the spectral record (0 = good data, 0 != bad data)
        ("flags", np.int32),
        # Catalog radial velocity of the source in the given frame (m/s)
        ("vradcat", np.float32),
        # Number of channels in the spectral record
        ("nch", np.int16),
        # Number of spectral records for this record -- always 1 (why does this exist?)
        ("nrec", np.int16),
        # Index offset of the spectral record in sch_read
        ("dataoff", np.int32),
        # Rest frequency of the source (GHz)
        ("rfreq", np.float64),
        # Correlator block (0 = synthetic continuum; 1 = SWARM)
        ("corrblock", np.int16),
        # Correlator chunk (spectral window number)
        ("corrchunk", np.int16),
        # Correlator number (0 = ASIC; 1 = SWARM)
        ("correlator", np.int32),
        # Spare value, always 0
        ("iddsmode", np.int16),
        # Spare value, always 0
        ("spareshort", np.int16),
        # Spare value, always 0
        ("spareint3", np.int32),
        # Spare value, always 0
        ("spareint4", np.int32),
        # Spare value, always 0
        ("spareint5", np.int32),
        # Spare value, always 0
        ("spareint6", np.int32),
        # SSB tsys (K)
        ("tssb", np.float64),
        # DDS frequency offset on nominal Gunn LO (GHz)
        ("fDDS", np.float64),
        # Spare value, always 0
        ("sparedbl3", np.float64),
        # Spare value, always 0
        ("sparedbl4", np.float64),
        # Spare value, always 0
        ("sparedbl5", np.float64),
        # Spare value, always 0
        ("sparedbl6", np.float64),
    ]
).newbyteorder("little")

# codes_read is a special set of metadata, basically used for storing "everything else".
# It is typically used for storing information that does not change over the course of
# the track, although a few commonly used codes due vary integration by integration.

codes_dtype = np.dtype(
    [("v_name", "U12"), ("icode", np.int16), ("code", "U26"), ("ncode", np.int16)]
).newbyteorder("little")

codes_binary_dtype = np.dtype(
    [("v_name", "S12"), ("icode", np.int16), ("code", "S26"), ("ncode", np.int16)]
).newbyteorder("little")

# we_read records various weather data collected at the antennas, which is typically
# used for refraction correction by online fringe tracking.
we_dtype = np.dtype(
    [
        # Scan number (should be equal to inhid)
        ("ints", np.int32),
        # Per-antenna flags, w/ bitwise flagging conditions
        ("flags", np.int32, 11),
        # Refractivity (N = (n - 1) * 1e6)
        ("N", np.float32, 11),
        # Ambient temperature measured at each antenna (C)
        ("Tamb", np.float32, 11),
        # Air pressure measured at each antenna (mbar)
        ("pressure", np.float32, 11),
        # Relative humidity measured at head antenna (%)
        ("humid", np.float32, 11),
        # Wind speed measured at each antenna (m/s, -1 if no hardware)
        ("windSpeed", np.float32, 11),
        # Wind direction measured at each antenna (rad, -1 if no hardward)
        ("windDir", np.float32, 11),
        # Boresite PWV measured at each antenna (mm, -1 if no hardware)
        ("h2o", np.float32, 11),
    ]
).newbyteorder("little")

# ac_read is _not_ something that is typically read in, but is instead a "helper"
# data structure for recording some of the metadata associated with the auto
# correlations. Because of this, the dtype below may change.
ac_dtype = np.dtype(
    [
        # Auto-correlation header ID
        ("achid", np.int32),
        # Integration header ID
        ("inhid", np.int32),
        # Antenna number
        ("antenna", np.int32),
        # Time at midpoint of scan (in TT UT hours)
        ("dhrs", np.float64),
        # Ant receiver number (0 = RxA, 1 = RxB)
        ("antrx", np.int16),
        # Reciever code (matched to rec in codes_read)
        ("irec", np.int16),
        # Polarization code (matched to pol in codes_read)
        ("ipol", np.int16),
        # Sidebad code (matched to sb in codes_read; usually 0=LSB, 1=USB)
        ("isb", np.int16),
        # Band code (matched to band in codes_read, ususally equal to corrchunk)
        ("iband", np.int16),
        # Correlator chunk (spectral window number)
        ("corrchunk", np.int16),
        # Correlator number (0 = ASIC; 1 = SWARM)
        ("correlator", np.int32),
        # Sky frequency at band center (GHz)
        ("fsky", np.float64),
        # Gunn frequency (GHz)
        ("gunnLO", np.float64),
        # Second downconverter frequency (GHz, not used currently)
        ("corrLO1", np.float64),
        # Final downconverter frequency (GHz)
        ("corrLO2", np.float64),
        # DDS frequency offset on nominal Gunn LO (GHz)
        ("fDDS", np.float64),
        # Channel resolution (MHz)
        ("fres", np.float32),
        # Number of channels
        ("nch", np.int32),
        # Offset from the start of the spectral record of the packed data.
        ("dataoff", np.int64),
        # Spare value, always 0
        ("sparedbl1", np.float64),
        # Spare value, always 0
        ("sparedbl2", np.float64),
        # Spare value, always 0
        ("sparedbl3", np.float64),
        # Spare value, always 0
        ("sparedbl4", np.float64),
        # Spare value, always 0
        ("sparedbl5", np.float64),
        # Spare value, always 0
        ("sparedbl6", np.float64),
        # Spare value, always 0
        ("sparedbl7", np.float64),
        # Spare value, always 0
        ("sparedbl8", np.float64),
        # Spare value, always 0
        ("sparedbl9", np.float64),
        # Spare value, always 0
        ("sparedbl10", np.float64),
        # Spare value, always 0
        ("sparedbl11", np.float64),
        # Spare value, always 0
        ("sparedbl12", np.float64),
        # Spare value, always 0
        ("sparedbl13", np.float64),
        # Spare value, always 0
        ("sparedbl14", np.float64),
        # Spare value, always 0
        ("sparedbl15", np.float64),
        # Spare value, always 0
        ("sparedbl16", np.float64),
    ]
).newbyteorder("little")

# antennas is actually a text file rather than a binary one, which stores the physical
# positions of the antennas at the time of observation. We use a fairly simple dtype
# here for handling its values.
antpos_dtype = np.dtype([("antenna", np.int16), ("xyz_pos", np.float64, 3)])


class MirMetaError(Exception):
    """
    Class for particular errors within MirMetaData objects.

    This class is used to flag errors within MirMetaData objects, usually relating to
    particular indexing fields not being found, or clashes between the indexes of two
    objects. It is used in practice as a unique identifier for these errors, so that
    they can be caught and handled within methods of the MirParser class.
    """

    pass


class MirMetaData(object):
    """
    Class for metadata within Mir datasets.

    This class is used as the parent class for the different types of metadata tables
    that exist within a Mir dataset. The object is built around a complex ndarray, which
    typically contains dozens of fields with different metadata stored. The object also
    contains a mask, which is useful for marking specific header entries as being in
    use (particularly when one has multiple MirMetaData objects together, like in the
    MirParser object).
    """

    def __init__(
        self,
        filetype,
        dtype,
        header_key_name,
        binary_dtype=None,
        pseudo_header_key_names=None,
        filepath=None,
    ):
        """
        Initialize a MirMetaData object.

        Parameters
        ----------
        filetype : str
            Name corresponding to a filetype in a Mir data set that the object is
            populated by (where the full path is filepath + "/" + filetype).
        dtype : dtype
            Numpy-based description of the binary data stored in the file.
        header_key : str or None
            Field inside of `dtype` which contains a unique indexing key for the
            metadata in the file. Typically used to reference values between MirMetaData
            objects. If set to `None`, no field is used for indexing.
        pseudo_index : list of str or None
            Required if `index_name` is `None`, used to identify a group of fields,
            which when taken in combination, can be used as a unique identifier.
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        self._filetype = filetype
        self.dtype = dtype
        self._binary_dtype = binary_dtype
        self._header_key = header_key_name
        self._pseudo_header_key = pseudo_header_key_names

        self._data = None
        self._mask = None
        self._header_key_index_dict = None
        self._stored_values = {}

        if filepath is not None:
            self.fromfile(filepath)

    def __iter__(self):
        """
        Iterate over MirMetaData attributes.

        Yields
        ------
        data_slice : ndarray
            Value(s) at a given position in the .
        """
        for idx in np.where(self._mask)[0]:
            yield self._data[idx]

    def copy(self, skip_data=False):
        """
        Make and return a copy of the MirMetaData object.

        Parameters
        ----------
        skip_data : bool
            If set to True, forgo copying the data-related attributes. Default is False.

        Returns
        -------
        new_obj : MirMetaData object
            Copy of the originial object.
        """
        # Initialize a new object of the given type
        copy_obj = type(self)()

        deepcopy_list = ["_stored_values"]
        data_list = ["_stored_values", "_data", "_mask", "_header_key_index_dict"]

        for attr in vars(self):
            if skip_data and attr in data_list:
                continue

            if attr in deepcopy_list:
                copy_attr = copy.deepcopy(getattr(self, attr))
            else:
                try:
                    copy_attr = getattr(self, attr).copy()
                except AttributeError:
                    copy_attr = copy.deepcopy(getattr(self, attr))

            setattr(copy_obj, attr, copy_attr)

        return copy_obj

    def __len__(self):
        """
        Calculate the number entries in the data table.

        Returns
        -------
        len : int
            Number of unique entries contained within the meta data.
        """
        return self._data.size

    def __eq__(self, other, verbose=False, ignore_params=None, use_mask=False):
        """
        Compare MirMetaData objects for equality.

        Parameters
        ----------
        other : MirMetaData object
            Object of the same type to compare to.
        verbose : bool
            Whether to print out the differences between the two objects, if any are
            found. Default is False.
        ignore_params : list of str or None
            Optional argument, which can be used to specify whether to ignore certain
            attributes when comparing objects. By default, no attributes are ignored.
        use_mask : bool
            Whether or not to ignore the internal mask when performing the comparison.
            If set to True, will only compare those entries where the mask is set to
            True. Default is False.

        Returns
        -------
        is_eq : bool
            Value describing whether or not the two objects contain the same data.
        """
        # Grab the name of the class to make the output a bit more human parsable
        name = type(self).__name__

        if not issubclass(other.__class__, MirMetaData):
            raise ValueError("Both objects must be MirMetaData (sub-) types.")

        # This _should_ be impossible unless the user mucked with the dtype, but
        # for safety sake, check now.
        if self.dtype != other.dtype:
            raise ValueError("Cannot compare %s with different dtypes." % name)

        if (self._data is None or self._mask is None) or (
            other._data is None or other._mask is None
        ):
            is_eq = (self._data is None) == (other._data is None)
            is_eq &= (self._mask is None) == (other._mask is None)
            if verbose and not is_eq:
                print("%s objects are not both initialized (one is empty)." % name)
            return is_eq

        this_keys = self.get_header_keys(use_mask=use_mask)
        other_keys = other.get_header_keys(use_mask=use_mask)

        if set(this_keys) != set(other_keys):
            if verbose:
                print("%s object header key lists are different." % name)
            return False

        this_idx = np.array([self._header_key_index_dict[key] for key in this_keys])
        other_idx = np.array([other._header_key_index_dict[key] for key in this_keys])

        # Figure out which fields inside the data array we need to compare.
        comp_fields = list(self.dtype.fields)
        if ignore_params is not None:
            for item in ignore_params:
                try:
                    comp_fields.remove(item)
                except ValueError:
                    pass

        # At this point we are ready to do our field-by-field comparison.
        # I say these objects are the same -- prove me wrong!
        is_eq = True
        for item in comp_fields:
            left_vals = self.get_value(item, index=this_idx)
            right_vals = other.get_value(item, index=other_idx)

            if not np.array_equal(left_vals, right_vals):
                is_eq = False
                if verbose:
                    print(
                        "%s of %s is different, left is %s, right is %s."
                        % (item, name, left_vals, right_vals)
                    )

        return is_eq

    def __ne__(self, other, verbose=False, ignore_params=None, use_mask=False):
        """
        Compare MirMetaData objects for inequality.

        Parameters
        ----------
        other : MirMetaData object
            Object of the same type to compare to.
        verbose : bool
            Whether to print out the differences between the two objects, if any are
            found. Default is False.
        ignore_params : list of str
            Optional argument, which can be used to specify whether to ignore certain
            attributes when comparing objects. By default, no attributes are ignored.

        Returns
        -------
        is_ne : bool
            Value describing whether the two objects do not contain the same data.
        """
        return not self.__eq__(
            other, verbose=verbose, ignore_params=ignore_params, use_mask=use_mask
        )

    def where(
        self,
        select_field,
        select_comp,
        select_val,
        mask=None,
        return_header_keys=False,
    ):
        """
        Find where metadata match a given set of selection criteria.

        This method will produce a masking screen based on the arguments provided to
        determine which entries matche a given set of conditions.

        Parameters
        ----------
        select_field : str
            Field in the metadata to evaluate.
        select_comp : str
            Specifies the type of comparison to do between the value supplied in
            `select_val` and the metadata. No default, allowed values include:
            "eq" (equal to, matching any in `select_val`),
            "ne" (not equal to, not matching any in `select_val`),
            "lt" (less than `select_val`),
            "le" (less than or equal to `select_val`),
            "gt" (greater than `select_val`),
            "ge" (greater than or equal to `select_val`),
            "btw" (between the range given by two values in `select_val`),
            "out" (outside of the range give by two values in `select_val`).
        select_val : number of str, or sequence of number or str
            Value(s) to compare data in `select_field` against. If `select_comp` is
            "lt", "le", "gt", "ge", then this must be either a single number
            or string. If `select_comp` is "btw" or "out", then this must be a list
            of length 2. If `select_comp` is "eq" or "ne", then this can be either a
            single value or a sequence of values.
        mask : ndarray of bool
            Optional argument, of the same length as the MirMetaData object, which is
            applied to the output of the selection parsing through an elemenent-wise
            "and" operation. Useful for combining multiple calls to `where` together.
        return_header_keys : bool
            If set to True, return a list of the header key values where matching
            entries are found. Default is False, which will return an ndarray of type
            bool, and length equal to that of the MirMetaData object.

        Returns
        -------
        return_arr : ndarray of bool or list
            If `return_header_keys=False`, boolean array marking whether `select_field`
            meets the condition set by `select_comp` and `select_val`. If
            `return_header_keys=True`, then instead of a boolean array, a list of ints
            (or tuples of ints if the MetaDataObject has only a pseudo header key)
            corresponding to the header key values.

        Raises
        ------
        ValueError
            If `select_comp` is not one of the permitted strings, or if `select_field`
            is not one of the fields within `data_arr`.
        """
        if select_field not in self._data.dtype.names:
            raise MirMetaError(
                "select_field %s not found in structured array." % select_field
            )

        # Create a simple dict to match operation keywords to a function.
        op_dict = {
            "eq": np.equal,
            "ne": np.not_equal,
            "lt": np.less,
            "le": np.less_equal,
            "gt": np.greater,
            "ge": np.greater_equal,
            "btw": lambda val, lims: ((val >= lims[0]) & (val <= lims[1])),
            "out": lambda val, lims: ((val < lims[0]) | (val > lims[1])),
        }

        if isinstance(select_val, (list, set, tuple, str, np.ndarray, np.str_)):
            op_dict["eq"] = lambda val, comp: np.isin(val, comp)
            op_dict["ne"] = lambda val, comp: np.isin(val, comp, invert=True)

        # Make sure the inputs look valid
        if select_comp not in op_dict:
            raise ValueError('select_comp must be one of: "%s"' % '", "'.join(op_dict))

        # Evaluate data_arr now
        data_mask = op_dict[select_comp](self._data[select_field], select_val)

        # Trap a corner-case here (most commonly w/ we-read), where some attributes
        # are multi-dim arrays rather than singleton values per index position.
        while data_mask.ndim > 1:
            data_mask = np.any(data_mask, axis=-1)

        # Apply the mask now if an argument has been supplied for it.
        if mask is not None:
            data_mask &= mask

        if return_header_keys and self._header_key is None:
            return list(
                zip(*[self._data[key][data_mask] for key in self._pseudo_header_key])
            )
        elif return_header_keys:
            return self._data[self._header_key][data_mask]

        return data_mask

    def _index_query(
        self,
        use_mask=None,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
    ):
        """
        Find array index positions where selection criteria are met.

        This is an internal helper function used by several methods of the MirMetaData
        class, and is not designed for users, but instead is part of the developer API.
        This function will report back the index positions in the `_data` attribute
        where the given selection criteria are met.

        Parameters
        ----------
        use_mask : bool
            If True, consider only data where the internal mask is marked True. Default
            is True, unless an argument is supplied to `index` or `header_key`, in
            which case the default is False.
        where : tuple or sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        index : sequence of ints
            Index positions of the array. Note that this is typicaly what you are
            calling this method for, but is included as an argument to simplify
            argument processing for various calls.
        header_key : sequence of ints or tuples
            Header key values to get the index position for, which are always recorded
            as ints. If the object has no header key, but instead a pseudo header key,
            a sequence of tuples (matching the pseudo keys) should be supplied.

        Returns
        -------
        index_arr : ndarray
            Array that can be used to access specific index positions, supplied as an
            ndarray of dtype int of variable length if supplying arguments to either
            `header_key` or `index`, otherwise of dtype bool and length matching that
            of the object.

        Raises
        ------
        ValueError
            If attempting to supply arguments to two or more of `index`, `header_key`,
            or `where`; or when attempting to supply a mask when supplying an argument
            to either `index` or `header_key`; or when the length of the mask does not
            match that of the object itself. Also raised if the argument supplied to
            `where` is not a 3-element tuple or seqeuence of 3-element tuples.
        MirMetaError
            If supplying an argument to `where`, and the selected field matches does
            not match for any of the supplied arguments.
        """
        # Check to make sure we aren't providing too many arguments here.
        arg_check = (index is None) + (header_key is None) + (where is None)

        if arg_check < 2:
            raise ValueError(
                "Only one of index, header_key, and where arguments can be set."
            )
        elif arg_check == 3:
            return self._mask.copy() if (use_mask or (use_mask is None)) else ...
        elif where is not None:
            use_mask = True if (use_mask is None) else use_mask
        elif use_mask:
            raise ValueError(
                "Cannot set use_mask=True when setting index or header_key."
            )

        if index is not None:
            # This is the dead-simple case - return just the input.
            return index
        elif header_key is not None:
            # This is a little trickier - use the pos dict to determine which entries
            # it is that we are trying to grab.
            if isinstance(header_key, int) or issubclass(type(header_key), np.integer):
                return self._header_key_index_dict[header_key]
            else:
                return np.array(
                    [self._header_key_index_dict[key] for key in header_key], dtype=int
                )

        # At this point, we expect to hand back a boolean mask, so either instantiate
        # it or make a copy of the supplied mask argument.
        mask = (
            self._mask.copy() if use_mask else np.full(len(self), bool(and_where_args))
        )

        if where is not None:
            # Otherwise, if we are going through where statements, then use the where
            # method to build a mask that we can use to select the data on. Check to
            # make sure that where matches what we expect - want to both accept a tuple
            # and sequence of tuples, so force it to be the latter.
            try:
                if not (isinstance(where[0], (tuple, list))):
                    # If where is not indexable, it'll raise a TypeError here.
                    where = [where]
                for item in where:
                    # Note we raise a TypeError in this loop to trap an identical bug,
                    # namely that the user has not provided a valid argument for where.
                    if len(item) != 3:
                        raise TypeError
            except TypeError:
                raise ValueError(
                    "Argument for where must be either a 3-element tuple, or sequence "
                    "of 3-element tuples."
                )

            # Now actually start going through the where statements.
            where_success = False
            for item in where:
                try:
                    if and_where_args:
                        mask &= self.where(*item)
                    else:
                        mask |= self.where(*item)
                    where_success = True
                except MirMetaError:
                    pass

        # If we had NO success with where, then we should raise an error now.
        if not where_success:
            raise MirMetaError(
                "Argument for where has no match(es) for select_field for this "
                "MirMetaData object. Must be one of %s." % ", ".join(self.dtype.fields)
            )

        return mask

    def get_value(
        self,
        field_name,
        use_mask=None,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
        return_tuples=False,
    ):
        """
        Get values from a particular field or set of fields of the metadata.

        This function allows one to get the values for a particular field or set of
        fields within the metadata. Selection criteria can optionally be specified for
        gathering only a subset of the metadata for the field(s).

        Parameters
        ----------
        field_name : str or list of strs
            Fields from which to extract data. Can either be given as either an str or
            list of strs. Each str must match a field name, as list in the `dtype`
            attribute of the object.
        use_mask : bool
            If True, consider only data where the internal mask is marked True. Default
            is True, unless an argument is supplied to `index` or `header_key`, in
            which case the default is False.
        where : tuple of sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details. Cannot be specified with
            `index` or `header_key`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        index : sequence of int
            Optional argument, specifies the index positions at which to extract data
            from the meta data. Cannot be specified with `header_key` or `where`.
        header_key : sequence of ints or tuples
            Optional argument, values to match against the header key field, in order to
            determine which entries of the array to extract. For example, if the header
            key field "a" has the values [2, 4, 6, 8], setting this argument to [2, 8]
            will grab the values of `field_name` in the metadata array at the index
            positions [0, 3]. Cannot be specified with `index` or `where`.
        return_tuples : bool
            If set to True, return a list of tuples containing the value of each field
            (in the order provided in `field_name`). If False, return an ndarray or
            list of ndarrays, where each array contains the set of values matching the
            specified selection criteria.

        Returns
        -------
        value_arr : ndarray or list of ndarrays or tuples
            Values for the specified field name where the selection criteria match.
            If `return_tuples=False`, then this will be an ndarray (of varying dtype) if
            a single field name was supplied, otherwise a list of ndarrays will be
            returned. If `return_tuples=True`, then a tuple containing the set of all
            fields at each index position will be provided.

        Raises
        ------
        ValueError
            If field_name is not a list, set, tuple, or str.
        """
        idx_arr = self._index_query(use_mask, where, and_where_args, header_key, index)

        if isinstance(field_name, (list, set, tuple)):
            metadata = []
            for item in field_name:
                if isinstance(item, str):
                    metadata.append(self._data[item][idx_arr])
                else:
                    raise ValueError("field_name must either be a str or list of str.")
        else:
            metadata = self._data[field_name][idx_arr]

        if return_tuples:
            return list(zip(*metadata) if isinstance(metadata, list) else zip(metadata))
        else:
            return metadata

    def __getitem__(self, item):
        """
        Get values for a particular field using get_value.

        Parameters
        ----------
        field_name : str
            Fields from which to extract data. Must match a field name, as list in the
            `dtype` attribute of the object.

        Returns
        -------
        value_arr : ndarray or list of ndarrays or tuples
            Values for the specified field name where the selection criteria match.
            If `return_tuples=False`, then this will be an ndarray (of varying dtype) if
            a single field name was supplied, otherwise a list of ndarrays will be
            returned. If `return_tuples=True`, then a tuple containing the set of all
            fields at each index position will be provided.
        """
        return self.get_value(item)

    def set_value(
        self,
        field_name,
        value,
        use_mask=None,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
    ):
        """
        Set values from a particular field of the metadata.

        Allows one to set the values of specific field within the metadata, optionally
        based on a set of selection criteria.

        Parameters
        ----------
        field_name : str
            Fields from which to extract data. Must match a field name, as list in the
            `dtype` attribute of the object.
        value : ndarray
            Values to set the field in question to, where the provided selection
            criteria match. Shape of the array must be broadcastable to either the shape
            of the internal mask or to the shape of the `index` or `header_key`
            arguments.
        use_mask : bool
            If True, consider only data where the internal mask is marked True. Default
            is True, unless an argument is supplied to `index` or `header_key`, in
            which case the default is False.
        where : tuple of sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details. Cannot be specified with
            `index` or `header_key`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        header_key : sequence of ints or tuples
            Optional argument, values to match against the header key field, in order to
            determine which entries of the array to extract. For example, if the header
            key field "a" has the values [2, 4, 6, 8], setting this argument to [2, 8]
            will set the values of `field_name` in the metadata array at the index
            positions [0, 3]. Cannot be specified with `index` or `where`.
        index : sequence of int
            Optional argument, specifies the index positions at which to set the values
            of the metadata. Cannot be specified with `header_key` or `where`.

        Raises
        ------
        UserWarning
            If attempting to set the field "dataoff", which is typically only used for
            internal indexing purposes, and is generally not modified. Also raised if
            modifying one of the known (pseudo) header keys.
        """
        if field_name == "dataoff":
            warnings.warn(
                'Values in "dataoff" are typically only used for internal indexing, '
                "and should generally not be set by users. If you have set this in "
                "error, you can undo this by using the reset method."
            )
        elif field_name in [
            "inhid",
            "blhid",
            "sphid",
            "ints",
            "v_name",
            "icode",
            "antenna",
            "achid",
        ]:
            warnings.warn(
                "Changing fields that tie to header keys can result in unpredicable "
                "behavior, and extreme care should be taken in directly modifying "
                "them. If you have set this in error, you can undo this by using the "
                "reset method."
            )

        idx_arr = self._index_query(use_mask, where, and_where_args, header_key, index)

        # Make a copy of any changed variables, that we can revert if we ever
        # happen to need to do so.
        if field_name not in self._stored_values:
            self._stored_values[field_name] = self._data[field_name].copy()

        self._data[field_name][idx_arr] = value

    def __setitem__(self, item, value):
        """
        Set values for a particular field using set_value.

        field_name : str
            Fields from which to extract data. Must match a field name, as list in the
            `dtype` attribute of the object.
        value : ndarray
            Values to set the field in question to, where the provided selection
            criteria match. Shape of the array must be broadcastable to either the shape
            of the internal mask or to the shape of the `index` or `header_key`
            arguments.
        """
        self.set_value(item, value)

    def _generate_mask(
        self,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
    ):
        """
        Generate a boolean mask based on selection criteria.

        Note that this is an internal helper function which is not for general user use,
        but instead is part of the low-level API for the MirMetaData object. Generates
        a boolean mask to based on the selection criteria (where the array is set to
        True when the selection criteria are met).

        Parameters
        ----------
        where : tuple of sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details. Cannot be specified with
            `index` or `header_key`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        header_key : sequence of ints or tuples
            Optional argument, values to match against the header key field, in order to
            determine which entries of the array to extract. For example, if the header
            key field "a" has the values [2, 4, 6, 8], setting this argument to [2, 8]
            will mask at the index positions [0, 3] to True. Cannot be specified with
            `index` or `where`.
        index : sequence of int
            Optional argument, specifies the index positions at which to set the mask to
            True. Cannot be specified with `header_key` or `where`.

        Returns
        -------
        mask_arr : ndarray of bool
            Array of boolean values, with length equal to that of the object itself.
        """
        idx_arr = self._index_query(False, where, and_where_args, header_key, index)
        new_mask = np.zeros(len(self), dtype=bool)

        new_mask[idx_arr] = True
        return new_mask

    def get_mask(
        self,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
    ):
        """
        Get value of the mask at a set of locations..

        This function allows one to get the value(s) of the internal mask. Selection
        criteria can optionally be specified for accessing the mask at a specific set
        of positions.

        Parameters
        ----------
        where : tuple of sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details. Cannot be specified with
            `index` or `header_key`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        header_key : sequence of ints or tuples
            Optional argument, values to match against the header key field, in order to
            determine which entries of the array to extract. For example, if the header
            key field "a" has the values [2, 4, 6, 8], setting this argument to [2, 8]
            will grab the values internal mask at the index positions [0, 3]. Cannot be
            specified with `index` or `where`.
        index : sequence of int
            Optional argument, specifies the index positions at which to extract data
            from the meta data. Cannot be specified with `header_key` or `where`.

        Returns
        -------
        mask_arr : ndarray of bool
            Values for mask where the selection criteria match.
        """
        idx_arr = self._index_query(False, where, and_where_args, header_key, index)
        return self._mask[idx_arr]

    def set_mask(
        self,
        mask=None,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
        reset=False,
        and_mask=True,
    ):
        """
        Set the internal object mask.

        This function updates the internal mask based on the supplied selection
        criteria. This internal mask is primarily used to identify which rows of data
        are "active", and will affect what some methods return to the user.

        Parameters
        ----------
        mask : ndarray of bool
            Optional argument, of the same length as the MirMetaData object, where True
            marks which index postions to set. Setting this will cause any arguments
            passed to `where`, `header_key`, and `index` to be ignored.
        where : tuple of sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details. Cannot be specified with
            `index` or `header_key`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        header_key : sequence of ints or tuples
            Optional argument, values to match against the header key field, in order to
            determine which entries of the array to extract. For example, if the header
            key field "hid" has the values [2, 4, 6, 8], setting this argument to [2, 8]
            will set the mask at index positions [0, 3] to True. Cannot be specified
            with `index` or `where`.
        index : sequence of int
            Optional argument, specifies the index positions at which to set the mask to
            True. Cannot be specified with `header_key` or `where`.
        reset : bool
            If set to True, reset all values of the mask to True before updating the
            mask. Default is False.
        and_mask : bool
            If set to True, then the mask generated by the selection criteria above will
            be combined with the existing internal mask using an element-wise "and"
            operation. If set to False, the two will instead be combined with an
            element-wise "or" operation. Default is True (i.e., and the masks together).
        """
        if mask is None:
            mask = self._generate_mask(
                where=where,
                and_where_args=and_where_args,
                header_key=header_key,
                index=index,
            )

        if reset:
            self._mask[:] = True

        mask = (self._mask & mask) if and_mask else (self._mask | mask)

        if np.array_equal(self._mask, mask):
            return False
        else:
            self._mask = mask
            return True

    def get_header_keys(
        self,
        use_mask=None,
        where=None,
        and_where_args=True,
        index=None,
        force_list=False,
    ):
        """
        Get the header keys based on selection criteria.

        This function allows one to lookup (pseudo) header keys that match a given
        set of criteria. Header keys are most commonly used to cross-link various
        metadata objects.

        Parameters
        ----------
        use_mask : bool
            If True, consider only data where the internal mask is marked True. Default
            is True, unless an argument is supplied to `index` or `header_key`, in
            which case the default is False.
        where : tuple of sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details. Cannot be specified with
            `index` or `header_key`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        index : sequence of int
            Optional argument, specifies the index positions at which to extract data
            from the meta data. Cannot be specified with `header_key` or `where`.
        force_list : False
            Normally the header keys are returned as an interable (ndarray in there
            is a header key, otherwise a list of tuples), but if set to True, the output
            will instead be a list of ndarray for each one of the fields within the set
            of (pseudo) header keys.

        Return
        ------
        header_key : ndarray or list
            If `force_list=False`, then if the object has a normal header key, an
            ndarray is returned with all keys that match the selection criteria,
            otherwise  a list of tuples is returned. If `force_list=True` or list of
            ndarrays is returned -- one for each field in the (pseudo) header keys.
        """
        if self._header_key is None:
            key = self._pseudo_header_key
        else:
            key = [self._header_key] if force_list else self._header_key

        return self.get_value(
            key,
            use_mask=use_mask,
            where=where,
            and_where_args=and_where_args,
            index=index,
            return_tuples=(self._header_key is None) and (not force_list),
        )

    def _set_header_key_index_dict(self):
        """
        Set internal header key to index position dictionary attribute.

        Note that this is an internal helper function, not intended for general users
        but instead is part of the developer API. Generates a dictionary that can be
        used for mapping header key values to index positions inside the data array.
        """
        self._header_key_index_dict = self.group_by(
            self._pseudo_header_key if self._header_key is None else self._header_key,
            use_mask=False,
            return_index=True,
            assume_unique=True,
        )

    def _generate_new_header_keys(self, other):
        """
        Create an updated set of header keys for a MirMetaData object.

        Note that this function is not meant to be called by users, but instead is
        part of the low-level API for the object. This function allows for one to
        create an updated set of header keys, such that their values do not conflict
        with another MirMetaData object -- useful for situations where you would like
        to combine the two objects together in some fashion.

        Parameters
        ----------
        other : MirMetaData object
            Object of identical type, whose header key values are used for calculating
            what the new header key values should be.

        Returns
        -------
        update_dict : dict
            Dictionary of header key values, where the keys are the old set and the
            values are the new set to be implemented. Note that if the object does not
            have a header key, this will return an empty dict.

        Raises
        ------
        ValueError
            If the two objects are not of the same type.
        """
        # First up, make sure we have two objects of the same dtype
        if type(self) != type(other):
            raise ValueError("Both objects must be of the same type.")

        # If no data are loaded, or if there is no header key, then this is basically
        # a no-op -- hand back an empty dictionary in this case.
        if (self._header_key is None) or (self._data is None):
            return {}

        idx_start = np.max(other._data[other._header_key]) + 1
        idx_stop = idx_start + len(self)

        index_dict = {
            self._header_key: {
                old_key: new_key
                for old_key, new_key in zip(
                    self.get_header_keys(use_mask=False),
                    range(idx_start, idx_stop),
                )
            }
        }

        return index_dict

    def _sort_by_header_key(self):
        """
        Sort data array by header key values.

        Note that this function is not designed to be called by users, but instead is
        part of the low-level API for the object. Calling this function will sort the
        metadaata in the `_data` attribute by the (pseudo) header key, and will
        regenerate the header key index dict accordingly. This function is most
        most commonly used after combining two objects to guarantee that the data are
        in the expected order.
        """
        sort_idx = np.lexsort(self.get_header_keys(use_mask=False, force_list=True))

        # Check and see if the data are already sorted (and skip the work if so).
        if not np.all(sort_idx[1:] > sort_idx[:-1]):
            self._data = self._data[sort_idx]
            self._mask = self._mask[sort_idx]

        self._set_header_key_index_dict()

    def group_by(
        self,
        group_fields,
        use_mask=True,
        return_index=False,
        assume_unique=False,
    ):
        """
        Create groups of index positions based on particular field(s) in the metadata.

        This method is a convenience function for creating groups of data based on a
        particular set of metadata.

        Parameters
        ----------
        group_fields : str or list of str
            Field or list of fields to group the data by. Must be one of the fields
            within the dtype of this object.
        use_mask : bool
            If True, consider only data where the internal mask is marked True.
        return_index : bool
            If False, return the header key values (or pseudo-key tuples) for each
            element of the group. If True, return instead the index position of the
            grouped data (if applicable). Note that the index positions are reported
            after the mask is applied, such that the highest index position will be
            equal to the sum of the mask values minus 1. Default is False.
        assume_unique : bool
            If set to True, assume that the value(s) of `group_field` are unique per
            index position, and return the results without making any attempts at
            grouping, which can produce a moderate increase in speed. Default is
            False.

        Returns
        -------
        group_dict : dict
            A dictionary containing the unique groupings of data depending on the input
            to `group_fields`. If a single str is provided, then the keys of the dict
            will be the unique values of the field, otherwise the keys will be tuples
            of the unique sets of metadata values for the grouping fields. The values
            are is either an ndarray of index positions (if `return_index=True`), an
            ndarray of header key values (if `return_index=True` and the objet has a
            valid  header key), or a list of tuples (if `return_index=True` and the
            object only has a pseudo-index), which correspond to the metadata entries
            that match the unique key.
        """
        # Make this a list just to make it easier to program against.
        if isinstance(group_fields, str):
            group_fields = [group_fields]

        # Get the data we want to group by and then use lexsort to arrange the data
        # in order. This turns out to make extracting the index positions much faster.
        group_data = self.get_value(group_fields, use_mask=use_mask)
        index_arr = np.lexsort(group_data)
        if not np.all(index_arr[1:] > index_arr[:-1]):
            group_data = [data[index_arr] for data in group_data]

        # If we have no data, then bail.
        if len(index_arr) == 0:
            return {}

        # Otherwise, if we don't want the index array, fill in the header keys now.
        if not return_index:
            if use_mask and not np.all(self._mask):
                index_arr = np.where(self._mask)[0][index_arr]

            index_arr = self.get_header_keys(index=index_arr)

        if assume_unique:
            if len(group_fields) == 1:
                return {key: value for key, value in zip(group_data[0], index_arr)}
            else:
                return {tup[:-1]: tup[-1] for tup in zip(*group_data, index_arr)}

        # Otherwise, check element-wise for differences, since that will tell us the
        # boundaries for each "group" of data.
        diff_idx = group_data[0][1:] != group_data[0][:-1]
        for data in group_data[1:]:
            diff_idx |= data[1:] != data[:-1]

        # Need the start position for the first group, and we add 1 to the rest of the
        # index values since the start positions are all offset by 1 thanks to the way
        # that we slicd things above.
        diff_idx = [0] + list(np.where(diff_idx)[0] + 1)

        # Figure out how to "name" the groups, based on how many fields we considered.
        if len(group_fields) == 1:
            group_names = list(group_data[0][diff_idx])
        else:
            group_names = list(zip(*[data[diff_idx] for data in group_data]))

        # In order to cleanly slice the data, we record the last good index position,
        # which will mark the beginning of the slice, with each subsequent list value
        # marking the end of the slice (and in the next iteration, the start).
        last_idx = diff_idx.pop(0)
        diff_idx.append(len(group_data[0]))

        # Finally, group together the data.
        group_dict = {}
        for idx, group in zip(diff_idx, group_names):
            group_dict[group] = index_arr[last_idx:idx]
            last_idx = idx

        return group_dict

    def reset_values(self, field_name=None):
        """
        Reset metadata fields to their original values.

        Restores the original values for metadata that has been changed, when it has
        been modified by set_value or __set_item__.

        Parameters
        ----------
        field_name : str or list of str
            Optional argument, specifies which fields should be restored. Can be either
            a single field (str) or multiple fields (list of str). Default is to restore
            all values which have been changed.

        Raises
        ------
        ValueError
            If the specified field name(s) do not have a backup copy found in the
            internal backup dictionary.
        """
        if field_name is None:
            field_name = list(self._stored_values)
        else:
            if isinstance(field_name, str):
                field_name = [field_name]
            for item in field_name:
                if item not in self._stored_values:
                    raise ValueError("No stored values for field %s." % item)

        for item in field_name:
            self._data[item] = self._stored_values.pop(item)

    def reset(self):
        """
        Reset a MirMeteData object.

        Restores the object to a "pristine" state, similar to when it was first loaded.
        Any changed fields are restored, and the mask is reset (selection criteria are
        unapplied).
        """
        self.reset_values()
        self.set_mask(reset=True)
        self._set_header_key_index_dict()

    def _update_fields(self, update_dict, raise_err=False):
        """
        Update fields within a MirMetaData object.

        Note that this is not a function designed to be called by users, but instead is
        part of the low-level API for the object. This function will take a so-called
        "update dictionary", which provides a way to map an existing set of values
        for a given field to an updated one. This function is most typically called
        when adding two different MirParser objects together, where multiple fields used
        as header keys (or other types of indexes) generally need to be updated prior
        to the add operation.

        Parameters
        ----------
        update_dict : dict
            Dictionary containing the set of updates to be applied. The keys specify
            the field name to be updated, and can either be given as a str (if a single
            field is to be updated) or as a tuple of str (if a series of fields are to
            be updated). The values of this dict are themselves dict, which map the
            existing values (keys) to the updated values (value). Note that if multiple
            fields were selected, both key and value for this lower-level dict should
            be tuples themselves of the same length.
        raise_err : bool
            If set to True, then if the field names in `update_dict` have no match
            in this object, an error is raised. Default is False, which means that
            if no match is found for a particular entry, the method quitely moves on
            to the next item.

        Raises
        ------
        ValueError
            If the keys of `update_dict` are not str or tuples of str, or if no matching
            field names are found and `raise_err` is set to True.
        """
        rebuild_index = False
        for field, data_dict in update_dict.items():
            if not isinstance(field, (str, tuple)):
                raise ValueError(
                    "update_dict must have keys that are type str or tuples of str."
                )

            # Check if we have a tuple, since it changes the logic a bit
            is_tuple = isinstance(field, tuple)
            # Check if we have a match for the field name
            has_match = True
            for item in field if is_tuple else [field]:
                has_match &= item in self.dtype.fields

            # If no match, and we want to raise an error, do so now.
            if not has_match:
                if raise_err:
                    raise ValueError("Field group %s not found in this object." % field)
                # Otherwise, just move along.
                continue

            # Check if we are modifying an index field, which will force us to reindex
            rebuild_index |= np.any(np.isin(field, self._header_key))
            rebuild_index |= np.any(np.isin(field, self._pseudo_header_key))

            # Get the existing metadata now, returned as tuples to make it easy
            # to use the update_dict
            iter_data = self.get_value(field, return_tuples=is_tuple, use_mask=False)

            # Note that with a complex dtype, passing _data a str will return a
            # reference to the array we want, so we can update in situ.
            arr_data = (
                [self._data[item] for item in field] if is_tuple else self._data[field]
            )

            # Now go through each value (or tuple of values) and plug in updates.
            for idx, old_vals in enumerate(iter_data):
                try:
                    if not is_tuple:
                        arr_data[idx] = data_dict[old_vals]
                    else:
                        for subarr, new_val in zip(arr_data, data_dict[old_vals]):
                            subarr[idx] = new_val
                except KeyError:
                    # If no matching key, then there is no update to perform
                    continue

        # If we have messed with an indexing field, then rebuild the header key index.
        if rebuild_index:
            self._set_header_key_index_dict()

    def _add_check(self, other, merge=None, overwrite=None, discard_flagged=False):
        """
        Check if two MirMetaData objects conflicting header key values.

        This method is an internal helper function not meant to be called by users.
        It checks if the header keys for two objects have overlapping values, and if so,
        what subset of each object's data to use when potentially combining the two.

        Parameter
        ---------
        other : MirMetaData
            MirMetaData object to be compared to this object.
        merge : bool
            If set to True, assumes that the two objects are to be "merged", which in
            this context means that they contain identical metadata, with just different
            selection masks applied.If set to False, assume that the objects contain
            unique data sets with unique header keys. By default, the method assumes
            that each object could contain a subset of the other (i.e., a partial
            merge).
        overwrite : bool
            If set to True, then when merging two objects (partial or whole), where
            the two objects have identical header keys, the method will assume metadata
            from `other` will be used to overwrite the metadata of this object,
            bypassing certain checks. If set to False, the method will assume no changes
            in metadata are allowed. The default is to assume that entries where the
            internal mask are set to False are allowed to be overwritten.
        discard_flagged : bool
            If set to True, exclude from consideration entries where the internal mask
            has been set to False. Default is False. Note that this cannot be used if
            setting `merge=True`.

        Returns
        -------
        this_idx : list of int
            Index positions denote which indicies of metadata would be utilized from
            this object if an __add__ operation were to be performed. Note that the
            header keys for this set of index positions will be completely disjoint
            from that of `other_idx` and `other`.
        other_idx : list of int
            Index positions denote which indicies of metadata would be utilized from
            `other` if an __add__ operation were to be performed. Note that the
            header keys for this set of index positions will be completely disjoint
            from that of `this_idx` and this object.
        this_mask : ndarray of bool
            Mask values for the index values in `this_idx`. Note that in the case of
            overlappping keys between this object and `other`, the maskes are "or'd"
            together.
        other_mask : ndarraay of bool
            Mask values for the index values in `other_idx`. Note that in the case of
            overlappping keys between this object and `other`, the maskes are "or'd"
            together.

        Raises
        ------
        MirMetaError
            If there is overlap between header keys, but merging is not permitted, or
            if merging fails because of differences between metadata values.
        ValueError
            If `other` is a different class than this object, or if attempting to set
            both `merge` and `discard_flagged` to True, or if setting `merge=True`,
            but the header keys (and their respective index positions) are different.
        """
        # First up, make sure we have two objects of the same dtype
        if type(self) != type(other):
            raise ValueError("Both objects must be of the same type.")

        if merge and discard_flagged:
            raise ValueError("Cannot both merge and discard flagged data.")

        # Grab copies of the metadata we need for various operations
        index_dict1 = self._header_key_index_dict.copy()
        index_dict2 = other._header_key_index_dict.copy()
        this_mask = self._mask.copy()
        other_mask = other._mask.copy()

        # Do a quick check here if the dicts are the same. If so, there's a fair bit of
        # optimization that we can leverage further down.
        same_dict = index_dict1 == index_dict2

        if merge and not same_dict:
            raise ValueError("Cannot merge if header keys for the objects differ.")

        # Deal w/ flagged data first, if need be
        if discard_flagged and not (np.all(self._mask) and np.all(other._mask)):
            # If nothing is flagged, then we can skip this, otherwise we need to
            # go through entry by entry for the two dicts. Make same_dict as False
            # now since they're no longer equal to the original dicts.
            same_dict = False
            # Note we call list here to instantiate a separate copy
            for key, value in list(index_dict1.items()):
                if not self._mask[value]:
                    _ = index_dict1.pop(key)
            for key, value in list(index_dict2.items()):
                if not self._mask[value]:
                    _ = index_dict2.pop(key)

        # See if we have any overlapping keys
        if same_dict:
            key_overlap = list(index_dict1)
        else:
            key_overlap = [key for key in index_dict1 if key in index_dict2]

        # If we can't merge, then error now
        if len(key_overlap) and not (merge or merge is None):
            raise MirMetaError(
                "Cannot add objects together if merge=False, since the two "
                "contain overlapping header keys."
            )

        # Go through the overlaping keys and see if we have any mismatches in mask
        # state. If we do, then we "or" the mask elements together, which always
        # results in a return value of True.
        if len(key_overlap):
            idx1 = [index_dict1[key] for key in key_overlap]
            idx2 = [index_dict2[key] for key in key_overlap]
            this_mask[idx1] |= other_mask[idx2]
            other_mask[idx2] = this_mask[idx1]

        # Count the sum total number of entries we have
        idx_count = len(index_dict1) + len(index_dict2)

        # Assume that if key_overlap has entries, we are allowed to merge
        if overwrite:
            # If we can overwrite, then nothing else matters -- drop the index
            # positions from this object and move on.
            _ = [index_dict1.pop(key) for key in key_overlap]
        elif len(key_overlap):
            # Check array index positions for arr1 first, see if everything is flagged
            arr1_idx = ... if same_dict else [index_dict1[key] for key in key_overlap]
            arr2_idx = ... if same_dict else [index_dict2[key] for key in key_overlap]
            arr1_mask = self._mask[arr1_idx]
            arr2_mask = other._mask[arr2_idx]

            if (overwrite is None) and not np.any(arr1_mask & arr2_mask):
                # If at each position at least one object is flagged, then drop the key
                # flagged from that object (dropping it from self if both objects have
                # that index flagged).
                for key, arr1_good in zip(key_overlap, arr1_mask):
                    _ = index_dict2.pop(key) if arr1_good else index_dict1.pop(key)
            else:
                # If the previous check fails, we have to do some heavier lifting.
                # Check all of the entries to see if the values are identical for
                # the overlapping keys.
                comp_mask = self._data[arr1_idx] == other._data[arr2_idx]

                if np.all(comp_mask):
                    # If all values are the same, then we can just delete all the
                    # overlapping keys from this object.
                    _ = [index_dict1.pop(key) for key in key_overlap]
                elif overwrite is not None:
                    # If you can't overwrite, then we have a problem -- this will
                    # trigger a fail down below, since there are unremoved keys
                    # not dealt with in key_overlap.
                    pass
                else:
                    # Finally, we are in a mixed state where we have to evaluate
                    # the entries on a case-by-case basis, and pass forward _some_
                    # keys from this object, and _some_ keys from the other object
                    # from the conflicted list.
                    for key, comp, mask1, mask2 in zip(
                        key_overlap, comp_mask, arr1_mask, arr2_mask
                    ):
                        if comp or (not mask1):
                            # If equal values OR this obj's record is flagged
                            del index_dict1[key]
                        elif not mask2:
                            # elif the other obj's record is flag
                            del index_dict2[key]
                        else:
                            # If neither of the above, break the loop, which will
                            # result in an error below.
                            break

        # If you've gotten to this point and you still have unresolved overlap
        # entries, then we have a problem -- time to raise an error.
        if (idx_count - (len(index_dict1) + len(index_dict2))) != len(key_overlap):
            raise MirMetaError(
                "Cannot combine objects, as both contain overlapping index markers "
                "with different metadata. You can bypass this error by setting "
                "overwrite=True."
            )

        this_idx = sorted(index_dict1.values())
        other_idx = sorted(index_dict2.values())

        return this_idx, other_idx, this_mask[this_idx], other_mask[other_idx]

    def __add__(
        self,
        other,
        inplace=False,
        merge=None,
        overwrite=None,
        discard_flagged=False,
    ):
        """
        Combine two MirMetaData objects.

        Note that when overlapping keys are detected (and are able to be reconciled),
        the method will "or" the two internal masks together, such that the sum of the
        two objects will contain the combination of any selection criteria that went
        into each object individually. This is particularly useful for when subsets of
        data have been split off from one another, and you wish to recombine them
        further downstream.

        Parameters
        ----------
        other : MirMetaData object
            Object to combine with this. Must be of the same type.
        inplace : bool
            If set to True, replace this object with the one resulting from the
            addition operation. Default is False.
        merge : bool
            If set to True, then the two objects are assumed to have identical metadata,
            with potentially different selection masks applied. If the underlying data
            or header key differences are detected, an error is raised. If set to False,
            the objects are contain unique data sets with unique header keys. If
            overlapping header keys are detected, an error is raised. By default, the
            method assumes that each object could contain a subset of the other, and
            will allow a partial merge where header keys overlap.
        overwrite : bool
            If set to True, then when merging two objects (partial or whole), where
            the two objects have identical header keys, data from `other` will overwrite
            that from this object. If set to False, no overwriting is allowed, and an
            error will be thrown if differing metadata are detected. The default is to
            allow metadata to be overwritten only where internal mask are set to False.
        discard_flagged : bool
            If set to True, exclude from metadata where the internal mask has been set
            to False. Default is False. Note that this cannot be used if setting
            `merge=True`.

        Returns
        -------
        new_obj : MirMetaData object
            The resultant combination of the two objects.

        Raises
        ------
        ValueError
            If attempting to combine this object with another of a different type.
        """
        # First up, make sure we have two objects of the same dtype
        if type(self) != type(other):
            raise ValueError("Both objects must be of the same type.")

        if other._data is None:
            # If no data is loaded, then this is just a no-op
            return self if inplace else self.copy()
        elif self._data is not None:
            idx1, idx2, mask1, mask2 = self._add_check(
                other, merge=merge, overwrite=overwrite, discard_flagged=discard_flagged
            )

        # At this point, we should be able to combine the two objects
        new_obj = self if inplace else self.copy()

        if self._data is None:
            new_obj._data = other._data.copy()
            new_obj._mask = other._mask.copy()
        else:
            new_obj._data = np.concatenate((new_obj._data[idx1], other._data[idx2]))
            new_obj._mask = np.concatenate((mask1, mask2))

        # Make sure the data is sorted corectly, generate the header key -> index
        # position dictionary.
        new_obj._set_header_key_index_dict()

        # Finally, clear out any sorted values, since there's no longer a good way to
        # carry them forward.
        new_obj._stored_values = {}

        return new_obj

    def __iadd__(self, other, merge=None, overwrite=None, discard_flagged=False):
        """
        In-place addition of two MirMetaData objects.

        Parameters
        ----------
        other : MirMetaData object
            Object to combine with this. Must be of the same type.
        merge : bool
            If set to True, then the two objects are assumed to have identical metadata,
            with potentially different selection masks applied. If the underlying data
            or header key differences are detected, an error is raised. If set to False,
            the objects are contain unique data sets with unique header keys. If
            overlapping header keys are detected, an error is raised. By default, the
            method assumes that each object could contain a subset of the other, and
            will allow a partial merge where header keys overlap.
        overwrite : bool
            If set to True, then when merging two objects (partial or whole), where
            the two objects have identical header keys, data from `other` will overwrite
            that from this object. If set to False, no overwriting is allowed, and an
            error will be thrown if differing metadata are detected. The default is to
            allow metadata to be overwritten only where internal mask are set to False.
        discard_flagged : bool
            If set to True, exclude from metadata where the internal mask has been set
            to False. Default is False. Note that this cannot be used if setting
            `merge=True`.

        Returns
        -------
        new_obj : MirMetaData object
            The resultant combination of the two objects.
        """
        return self.__add__(
            other,
            inplace=True,
            merge=merge,
            overwrite=overwrite,
            discard_flagged=discard_flagged,
        )

    def fromfile(self, filepath):
        """
        Read in data for a MirMetaData object from disk.

        Parameters
        ----------
        filepath : str
            Path of the folder containing the metadata in question.
        """
        if self._binary_dtype is None:
            self._data = np.fromfile(
                os.path.join(filepath, self._filetype),
                dtype=self.dtype,
            )
        else:
            self._data = np.fromfile(
                os.path.join(filepath, self._filetype), dtype=self._binary_dtype
            ).astype(self.dtype)

        self._mask = np.ones(len(self), dtype=bool)
        self._set_header_key_index_dict()

    def _writefile(self, filepath, append_data, datamask=...):
        """
        Write _data attribute to disk.

        This function is part of the low-level API, which is called when calling the
        `tofile` method. It is broken out separately here to enable subclasses to
        differently specify how data are written out (namely binary vs text).

        Parameters
        ----------
        filepath : str
            Path of the folder to write the metadata into.
        append_data : bool
            If set to True, will append to an existing file, otherwise the method will
            overwrite any previously written data.
        datamask : ndarray of bool
            Mask for selecting which data to write to disk. Default is the entire
            array of metadata.
        """
        with open(filepath, "ab" if append_data else "wb+") as file:
            if self._binary_dtype is None:
                self._data[datamask].tofile(file)
            else:
                self._data[datamask].astype(self._binary_dtype).tofile(file)

    def tofile(
        self,
        filepath,
        overwrite=False,
        append_data=False,
        check_index=False,
    ):
        """
        Write a metadata file to disk.

        This function is part of the low-level API, which is called when calling the
        `tofile` method. It is broken out separately here to enable subclasses to
        differently specify how data are written out (namely binary vs text).

        Parameters
        ----------
        filepath : str
            Path of the folder to write the metadata into.
        overwrite : bool
            If set to True, will allow the method to overwrite a previously written
            dataset. If set to False and said file exists, the method will throw an
            error. Default is False.
        append_data : bool
            If set to True, will append data to an existing file. Default is False.
        check_index : bool
            Only applicable if `append_data=True`. If set to True and data are being
            appended to an existing file, the method will check to make sure that there
            are no header key conflicts with the data being being written to disk, since
            this can cause the file to become unusable. Default is False.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        writepath = os.path.join(os.path.abspath(filepath), self._filetype)

        if os.path.exists(writepath):
            if not (append_data or overwrite):
                raise FileExistsError(
                    "File already exists, must set overwrite or append_data to True, "
                    "or delete the file %s in order to proceed." % filepath
                )
        else:
            # Just set these now to forgo the potential check below.
            append_data = False

        if append_data and check_index:
            copy_obj = self.copy(skip_data=True)
            copy_obj.fromfile(filepath)
            try:
                idx_arr = self._add_check(
                    copy_obj,
                    discard_flagged=True,
                    overwrite=False,
                )[0]
            except MirMetaError:
                # If we get this error, it means our (partial) merge has failed.
                # Time to bail.
                raise ValueError(
                    "Conflicting header keys detected with data on disk. Cannot "
                    "append data from this object to specified file."
                )

            if len(idx_arr) == 0:
                # There's literally nothing to do here, so bail.
                return

            # Generate a mask based on the unique data entries.
            datamask = self._generate_mask(index=idx_arr)
        else:
            # If we haven't done so yet, create the data mask now.
            datamask = ... if np.all(self._mask) else self._mask

        self._writefile(writepath, append_data, datamask)

    def _get_record_size_info(self, use_mask=True):
        if isinstance(self, MirSpData):
            # Each channel is 2 bytes in length
            val_size = 2
            # Each channel has two values (real + imag)
            n_val = 2
            # Each vis record has an extra int16 that acts as a common exponent
            n_pad = 1
        elif isinstance(self, MirAcData):
            # Each channel is 4 bytes in length (float32)
            val_size = 4
            # Each channel has one values (real-only)
            n_val = 1
            # There are no "extra" components of auto records
            n_pad = 0
        else:
            raise TypeError(
                "Cannot use this method on objects other than MirSpData"
                "and MirAcData types."
            )

        # Calculate the individual record sizes here. Each record contains 1 int16
        # (common exponent) + 2 * nch values.
        rec_size_arr = val_size * (
            n_pad + (n_val * self.get_value("nch", use_mask=use_mask).astype(int))
        )

        return rec_size_arr, val_size

    def _recalc_dataoff(self, use_mask=True):
        """
        Calculate the offsets of each spectral record for packed data.

        This is an internal helper function not meant to be called by users, but
        instead is part of the low-level API. This function is used to calculate the
        relative offset of the spectral record inside of a per-integration "packed
        data array", which is what is recorded to disk. This method is primarily used
        when writing visibility to disk, since the packing of the data (and by
        extension, it's indexing) depends heavily on what records have been recorded to
        disk. Note that operation _will_ modify the "dataoff" field inside of the
        metadata, so care should be taken when calling it.

        Parameters
        ----------
        use_mask : bool
            If set to True, evaluate/calculate for only those records where the internal
            mask is set to True. If set to False, use all records in the object,
            regardless of mask status. Default is True.
        """
        rec_size_arr, _ = self._get_record_size_info(use_mask=use_mask)

        # Create an array to plug values into
        offset_arr = np.zeros_like(rec_size_arr)
        for index_arr in self.group_by(
            "inhid", use_mask=use_mask, return_index=True
        ).values():
            temp_recsize = rec_size_arr[index_arr]
            offset_arr[index_arr] = np.cumsum(temp_recsize) - temp_recsize

        # Finally, update the attribute with the newly calculated values. Filter out
        # the warning since we don't need to raise this if using the internal method.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message='Values in "dataoff" are typically only used for'
            )
            self.set_value("dataoff", offset_arr, use_mask=use_mask)

    def _generate_recpos_dict(self, use_mask=True, reindex=False):
        """
        Generate a set of dicts for indexing of data.

        This is an internal helper function not meant to be called by users, but
        instead is part of the low-level API. This function is used to calculate
        internal indexing values for use in unpacking the raw data on disk, recorded
        under the filename "sch_read".

        Parameters
        ----------
        use_mask : bool
            If set to True, evaluate/calculate for only those records where the internal
            mask is set to True. If set to False, use all records in the object,
            regardless of mask status. Default is True.
        reindex : bool
            If set to True, evaluate/calculate ignoring the current indexing info,
            instead relying upon record order and size for calculating the results.
            Typically used for generating dicts for writing records to disk.

        Returns
        -------
        int_start_dict : dict
            Dictionary with information about individual integrations, where the key
            is matched to a integration header key ("inhid"), and the value is itself
            a 3-element dictionary containing the keys "inhid" (header key as recorded
            on disk, which is not _neccessarily_ the same as in the object),
            "record_size" (in bytes), and "record_start" (start position of the packed
            data relative to the start of the file, in bytes), with all values recorded
            as ints.
        recpos_dict : dict
            Dictionary containing per-spectral record indexing information. The keys
            are values of "inhid", and the values are themselves dicts whose keys are
            values of the spectral window header key ("sphid" for cross-correlations, or
            "achid" for auto-correlations), the whole group of which are matched to a
            particular "inhid" value. The values of this "lower" dict is yet another
            dict, containing three keys: "start_idx" (starting position of the spectral
            record in the packed data, in number of 2-byte ints), "end_idx" (ending
            position of the spectral record), and "chan_avg" (number of channels ones
            needs to average the spectrum over; default is 1).
        """
        rec_size_arr, val_size = self._get_record_size_info(use_mask=use_mask)

        int_dict = {}
        recpos_dict = {}

        # Group together the spectral records by inhid to begin the process of
        # building out sp_dict.
        inhid_groups = self.group_by("inhid", use_mask=use_mask, return_index=True)
        hkey_arr = self.get_header_keys(use_mask=use_mask)

        # Divide by val_size here since we're going from bytes to number of vals
        if not reindex:
            dataoff_arr = self.get_value("dataoff", use_mask=use_mask) // val_size

        # Begin putting together dicts now.
        record_start = 0
        for inhid in inhid_groups:
            # Extract out the relevant spectral record group.
            rec_idx = inhid_groups[inhid]

            # We captured index values above, so now we need to grab header keys
            # and record start/size information at each index position.
            hkey_subarr = hkey_arr[rec_idx]
            rec_size_subarr = rec_size_arr[rec_idx] // val_size
            if reindex:
                eidx_arr = np.cumsum(rec_size_subarr)
                sidx_arr = eidx_arr - rec_size_subarr
            else:
                sidx_arr = dataoff_arr[rec_idx]
                eidx_arr = sidx_arr + rec_size_subarr

            # Plug in the start/end index positions for each spectral record.
            recpos_dict[inhid] = {
                hkey: {"start_idx": sidx, "end_idx": eidx, "chan_avg": 1}
                for hkey, sidx, eidx in zip(hkey_subarr, sidx_arr, eidx_arr)
            }

            # Record size for int_dict is recorded in bytes, hence the * chan_size here
            record_size = eidx_arr.max() * val_size

            int_dict[inhid] = {
                "inhid": inhid,
                "record_size": record_size,
                "record_start": record_start,
            }
            # Note the +8 here accounts for 2 int32s that are used to mark the inhid
            # and record size within the sch_read file itself.
            record_start += record_size + 8
        return int_dict, recpos_dict


class MirInData(MirMetaData):
    """
    Class for per-integration metadata in Mir datasets.

    This class is a container for per-integration metadata, using the header key
    "inhid". When reading from/writing to disk, the object looks for a file named
    "in_read", which is where the online system records this information.
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirInData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        super().__init__("in_read", in_dtype, "inhid", None, None, filepath)


class MirBlData(MirMetaData):
    """
    Class for per-baseline metadata in Mir datasets.

    This class is a container for per-baseline metadata, using the header key
    "blhid". When reading from/writing to disk, the object looks for a file named
    "bl_read", which is where the online system records this information. Note that
    "per-baseine" here means per-integration, per-sideband, per-receiver/polarization.
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirBlData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        super().__init__("bl_read", bl_dtype, "blhid", None, None, filepath)


class MirSpData(MirMetaData):
    """
    Class for per-spectral window metadata in Mir datasets.

    This class is a container for per-spectral window metadata, using the header key
    "sphid". When reading from/writing to disk, the object looks for a file named
    "sp_read", which is where the online system records this information. Note that
    "per-spectral window" here means per-integration, per-baseline, per-spectral
    band number.
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirSpData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        super().__init__("sp_read", sp_dtype, "sphid", None, None, filepath)


class MirWeData(MirMetaData):
    """
    Class for per-integration weather metadata in Mir datasets.

    This class is a container for per-integration weather metadata, using the header key
    "ints". When reading from/writing to disk, the object looks for a file named
    "we_read", which is where the online system records this information.
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirWeData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        super().__init__("we_read", we_dtype, "ints", None, None, filepath)


class MirEngData(MirMetaData):
    """
    Class for per-antenna metadata in Mir datasets.

    This class is a container for per-antenna, per-integration metadata. When reading
    from/writing to disk, the object looks for a file named "eng_read", which is where
    the online system records this information. This object does not have a unique
    header key, but instead has a pseudo key made up of the integration header ID
    number ("inhid") and the antenna number ("antenna"), which should be unique for
    each entry.
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirEngData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        super().__init__(
            "eng_read", eng_dtype, None, None, ("antenna", "inhid"), filepath
        )


class MirAntposData(MirMetaData):
    """
    Class for antenna position information in Mir datasets.

    This class is a container for antenna positions, which are recorded as a text file
    within a Mir dataset named "antennas". It has a header key of "antenna", which is
    paired to the antenna number in other metadata objects (e.g., "antenna",
    "iant1", "iant2").
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirAntposData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        super().__init__("antennas", antpos_dtype, "antenna", None, None, None)

        if filepath is not None:
            self.fromfile(filepath)

    def fromfile(self, filepath):
        """
        Read in data for a MirAntposData object from disk.

        Parameters
        ----------
        filepath : str
            Path of the folder containing the metadata in question.
        """
        with open(os.path.join(filepath, "antennas"), "r") as antennas_file:
            temp_list = [
                item for line in antennas_file.readlines() for item in line.split()
            ]
        self._data = np.empty(len(temp_list) // 4, dtype=antpos_dtype)
        self._data["antenna"] = np.int16(temp_list[0::4])
        self._data["xyz_pos"] = np.array(
            [temp_list[1::4], temp_list[2::4], temp_list[3::4]], dtype=np.float64
        ).T

        self._mask = np.ones(len(self), dtype=bool)
        self._set_header_key_index_dict()

    def _writefile(self, filepath, append_data, datamask=...):
        """
        Write _data attribute to disk.

        This function is part of the low-level API, which is called when calling the
        `tofile` method. It is broken out separately here to enable subclasses to
        differently specify how data are written out (namely binary vs text).

        Parameters
        ----------
        filepath : str
            Path of the folder to write the metadata into.
        append_data : bool
            If set to True, will append to an existing file, otherwise the method will
            overwrite any previously written data.
        datamask : ndarray of bool
            Mask for selecting which data to write to disk. Default is the entire
            array of metadata.
        """
        # We need a special version of this for the antenna positions file since that's
        # the only one that's a text file vs a binary file.
        with open(filepath, "a" if append_data else "w+") as file:
            for antpos in self._data[datamask]:
                file.write(
                    "%i %.17e %.17e %.17e\n"
                    % (
                        antpos["antenna"],
                        antpos["xyz_pos"][0],
                        antpos["xyz_pos"][1],
                        antpos["xyz_pos"][2],
                    )
                )


class MirCodesData(MirMetaData):
    """
    Class for per-track metadata in Mir datasets.

    This class is a container for various metadata, which typically vary per-integration
    or not at all. When reading from/writing to disk, the object looks for a file named
    "codes_read", which is where the online system records this information. This object
    does not have a unique header key, but instead has a pseudo key made up of the
    variable name ("v_name") and the indexing code ("icode").

    The main feature of block of metadata is two-fold. First, it enables one to match
    strings (for example, like that used for source names) to indexing values that are
    used by other metadata types (e.g., isource in "in_read"). Second, it enables one
    to record arbitrary strings that can be used to describe various properties of the
    whole dataset (e.g., "filever", which denotes the version).

    This object has several methods that are partially inherited from the MirMetaData
    class, but are modified accordingly to enable better flexibility when attempting to
    process these string "codes".
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirCodesData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        super().__init__(
            "codes_read",
            codes_dtype,
            None,
            codes_binary_dtype,
            ("icode", "v_name"),
            filepath,
        )

        self._mutable_codes = [
            "project",
            "ref_time",
            "ut",
            "vrad",
            "source",
            "stype",
            "svtype",
            "ra",
            "dec",
            "ddsmode",
            "sb",
            "tel1",
            "tel2",
            "band",
        ]

        # These are codes that _cannot_ change between objects, otherwise it breaks
        # some of the underlying logic of some code, and could mean that the files
        # may have different metadata fields populated.
        self._immutable_codes = [
            "filever",
            "pol",
        ]

        # These are v_names that match to particular indexing fields in other metadata
        # files (with the values matching said fields).
        self._codes_index_dict = {
            "project": "iproject",
            "ref_time": "iref_time",
            "ut": "iut",
            "source": "isource",
            "ra": "ira",
            "dec": "idec",
            "stype": "isource",
            "svtype": "isource",
            "vrad": "ivrad",
            "gq": "igq",
            "pq": "ipq",
            "tel1": "iant1",
            "tel2": "iant2",
            "pol": "ipol",
            "pstate": "ipstate",
            "sb": "isb",
            "band": "iband",
            "ddsmode": "iddsmode",
        }

    def get_code_names(self):
        """
        Produce a list of code types (v_names) found in the metadata.

        Returns
        -------
        code_list : list of str
            A list of all the unique code types, as recorded in the "v_name" field
            of the metadata.
        """
        return sorted(set(self.get_value("v_name")))

    def where(
        self,
        select_field,
        select_comp,
        select_val,
        mask=None,
        return_header_keys=None,
    ):
        """
        Find where metadata match a given set of selection criteria.

        This method will produce a masking screen based on the arguments provided to
        determine which entries matche a given set of conditions.

        Parameters
        ----------
        select_field : str
            Field or code type ("v_name") in the metadata to evaluate.
        select_comp : str
            Specifies the type of comparison to do between the value supplied in
            `select_val` and the metadata. No default, allowed values include:
            "eq" (equal to, matching any in `select_val`),
            "ne" (not equal to, not matching any in `select_val`),
            "lt" (less than `select_val`),
            "le" (less than or equal to `select_val`),
            "gt" (greater than `select_val`),
            "ge" (greater than or equal to `select_val`),
            "btw" (between the range given by two values in `select_val`),
            "out" (outside of the range give by two values in `select_val`).
        select_val : number of str, or sequence of number or str
            Value(s) to compare data in `select_field` against. If `select_comp` is
            "lt", "le", "gt", "ge", then this must be either a single number
            or string. If `select_comp` is "btw" or "out", then this must be a list
            of length 2. If `select_comp` is "eq" or "ne", then this can be either a
            single value or a sequence of values.
        mask : ndarray of bool
            Optional argument, of the same length as the MirMetaData object, which is
            applied to the output of the selection parsing through an elemenent-wise
            "and" operation. Useful for combining multiple calls to `where` together.
        return_header_keys : bool
            If set to True, return a list of the header key values where matching
            entries are found. Default is False if supplying a field name for
            `select_field`, and True if supplying a code type for `select_field`.

        Returns
        -------
        return_arr : ndarray of bool or list
            If `return_header_keys=False`, boolean array marking whether `select_field`
            meets the condition set by `select_comp` and `select_val`. If
            `return_header_keys=True`, then instead of a boolean array, a list of ints
            (or tuples of ints if the MetaDataObject has only a pseudo header key)
            corresponding to the header key values. Note that if a code type was
            supplied for `select_field` and `return_header_keys` was not set to False,
            then the function will return a list of the matching index codes ("icode")
            for the given code type.

        Raises
        ------
        ValueError
            If `select_comp` is not one of the permitted strings, or if `select_field`
            is not one of the fields within the metadata, or a valid code type. Also
            raised if setting `select_comp` to anything but "eq" or "ne" when selecting
            on a code type (other operations not allowed since they are nonsensical for
            strings).
        MirMetaError
            If `select_field` does not match the metadata field types or any of the
            indexing codes.
        """
        if select_field in self.dtype.fields:
            return super().where(
                select_field, select_comp, select_val, mask, return_header_keys
            )

        if select_field not in self._codes_index_dict:
            raise MirMetaError(
                "select_field must either be one of the native fields inside of the "
                'codes_read array ("v_name", "code", "icode", "ncode") or one of the '
                "indexing codes (%s)." % ", ".join(list(self._codes_index_dict))
            )

        if select_comp not in ["eq", "ne"]:
            raise ValueError(
                'select_comp must be "eq" or "ne" when select_field is a code type.'
            )

        # Convert select_val into a bytes object or sequence of bytes objects, since
        # that's how they are read from disk.
        if not isinstance(select_val, str):
            try:
                select_val = list(select_val)
                for idx, item in enumerate(select_val):
                    if not isinstance(item, str):
                        select_val[idx] = str(item)
            except TypeError:
                # Assume at this point that we are working with a single non-string
                # entry (that we need to convert into)
                select_val = str(select_val)

        data_mask = np.logical_and(
            super().where("v_name", "eq", select_field, mask, False),
            super().where("code", select_comp, select_val, mask, False),
        )

        if return_header_keys or (return_header_keys is None):
            return list(self.get_value("icode")[data_mask])
        else:
            return data_mask

    def get_codes(self, code_name, return_dict=None):
        if code_name not in self.get_code_names():
            raise MirMetaError(
                "%s does not match any code or field in the metadata." % code_name
            )

        mask = self.where("v_name", "eq", code_name, return_header_keys=False)
        codes = list(self.get_value("code", use_mask=False)[mask])
        index = list(self.get_value("icode", use_mask=False)[mask])
        if return_dict is None:
            return_dict = (np.sum(mask) != 1) or (code_name in self._codes_index_dict)

        if return_dict:
            return {key: value for key, value in zip(codes + index, index + codes)}
        else:
            return codes

    def __getitem__(self, item):
        """
        Get values for a particular field using get_value.

        Parameters
        ----------
        field_name : str
            Fields from which to extract data. Must match a field name in the data, or
            a value for "v_name" within the metadata.

        Returns
        -------
        value_arr : ndarray or list of ndarrays or str or dict.
            If `field_name` is one or more of "v_name", "code", "icode", or "ncode",
            then this will be an ndarray if a single field name was selected, or list
            of ndarray if multiple fields were selected. If giving a string which
            matches an entry for "v_name", then the behavior is slightly different:
            if a single entry is found (and "v_name" is not attached to a code that
            is indexed in other metadata), then a str is returned that is the code
            value for that entry. Otherwise, a dictionary mapping the indexing codes
            (type int) and code string (type str) to one another.
        """
        if item in self.dtype.fields:
            return super().__getitem__(item)
        else:
            return self.get_codes(item)

    def _generate_new_header_keys(self, other):
        """
        Create an updated set of pseudo header keys for a MirCodesData object.

        Note that this function is not meant to be called by users, but instead is
        part of the low-level API for the object. This function allows for one to
        create an updated set of pseudo header keys via an updae to the indexing codes,
        such that their values do not conflict with another MirCodesData object --
        useful for situations where you would like to combine the two datasets together.

        Parameters
        ----------
        other : MirCodesData object
            Object of identical type, whose header key values are used for calculating
            what the new header key values should be.

        Returns
        -------
        update_dict : dict
            Dictionary of pseudo header key tuples index code values, where the keys
            are the old set and the values are the new set to be implemented. Note that,
            if applicable, this dict will also contain entries that match to other
            indexing fields (e.g., if the "source" codes were updated, the update
            dictionary will also contain an entry for "isource", which can be used to
            update values in the per-integration record metadata).

        Raises
        ------
        ValueError
            If the two objects are not of the same type.
        """
        # First up, make sure we have two objects of the same dtype
        if type(self) != type(other):
            raise ValueError("Both objects must be of the same type.")

        index_dict = {}

        this_vnames = self.get_code_names()
        other_vnames = other.get_code_names()

        # These are codes that are (annoyingly) tied together, where one index value
        # is used to reference multiple code types.
        skip_codes = {
            "stype": (self["stype"], other["stype"]),
            "svtype": (self["svtype"], other["svtype"]),
        }

        # If the two lists form the "skipped" codes are the same, then we can save
        # ourselves a bit of work later on, so check this now.
        same_skip = np.all([item == jtem for item, jtem in skip_codes.values()])

        for vname in this_vnames:
            # Don't worry about the "skipped" codes.
            if vname in skip_codes:
                continue

            # If the codes are identical, then also skip processing
            if vname in other_vnames:
                if (self[vname] == other[vname]) and (same_skip or (vname != "source")):
                    continue

            if vname in self._immutable_codes:
                # If the codes are supposed to be identical, then we should have bailed
                # by this point. Raise an error.
                raise ValueError(
                    "The codes for %s in codes_read cannot change between "
                    "objects if they are to be combined." % vname
                )
            elif vname in other_vnames:
                if not (vname in self._mutable_codes):
                    # If the code is not recognized as a mutable code, but not forbidden
                    # from changing, then just raise a warning and proceed.
                    warnings.warn(
                        "Codes for %s not in the recognized list of mutable codes. "
                        "Moving ahead anyways since it is not forbidden." % vname
                    )
                temp_dict = {}

                # This will return a dict that maps code string -> indexing value and
                # indexing value -> code string for a given code type.
                this_dict = self[vname]
                other_dict = other[vname]

                # Start the process of reindexing the "icode" values
                last_idx = 1
                for key, value in this_dict.items():
                    if not isinstance(key, str):
                        # The dict contains both strings and ints, but we just want
                        # to deal with the strings in this process.
                        continue
                    try:
                        # See if we can find this code string in the other dict.
                        other_value = other_dict[key]

                        # We need to handle a special case here, due to the source
                        # index being applied across multiple codes.
                        if vname == "source":
                            for dict1, dict2 in skip_codes.values():
                                if dict1[value] != dict2[other_value]:
                                    raise KeyError()

                        if value != other_value:
                            # Here we have to handle to case that the code string _is_
                            # found in the other dict, but not with the same index.
                            temp_dict[value] = other_dict[key]
                    except KeyError:
                        # If the code is _not_ found in the other dict, then we just
                        # want to pick and indexing code that won't cause a conflict.
                        # Loop through and pick the first positive unassigned value.
                        if value in other_dict:
                            while last_idx in other_dict:
                                last_idx += 1
                            temp_dict[value] = last_idx
                            last_idx += 1

                # Store the results in our update dictionary.
                if len(temp_dict):
                    index_dict[vname] = temp_dict
                    if vname == "source":
                        for item in skip_codes:
                            index_dict[item] = temp_dict

        # We now have a list of updates we want to make, but we need to parse the dict
        # in such a way that it can be used by _update_fields. The icode_dict will
        # record entries for this object, while index_dict entries will be modified
        # to match what we want for the other metadata objects.
        icode_dict = {}
        for key in list(index_dict):
            # Remove the item from the dict temporarily.
            temp_dict = index_dict.pop(key)
            if key in self._codes_index_dict:
                # If used as an indexing field, then put the name of the aliased field
                # in as the key, and match it to our existing dict.
                index_dict[self._codes_index_dict[key]] = temp_dict

            for old_idx, new_idx in temp_dict.items():
                # Use the tuple of (v_name, old_icode) to map to the new tuple
                # (v_name, new_icode), which _update_fields will handle properly.
                icode_dict[(key, old_idx)] = (key, new_idx)

        # If there are any codes to update, then merge it into the main dict.
        if len(icode_dict) > 0:
            index_dict[("v_name", "icode")] = icode_dict

        return index_dict


class MirAcData(MirMetaData):
    """
    Class for per-track metadata in Mir datasets.

    This class is a container for per-auto correlation metadata using the header key
    "achid". At present, this class is a "synthetic" metadata object, in that it does
    not match to a natively written file on disk, as recorded by the online system
    (although it will read and write to the filename "ac_read"). As such, this class
    should be considered a "work in progress", whose functionality may evolve
    considerably in future releases.
    """

    def __init__(self, filepath=None):
        """
        Initialize a MirAcData object.

        Parameters
        ----------
        filepath : str
            Optional argument specifying the path to the Mir data folder.
        """
        self._old_type = False
        super().__init__("ac_read", ac_dtype, "achid", None, None, filepath)

    def fromfile(self, filepath, nchunks=8):
        """
        Read in data for a MirAcData object from disk.

        Parameters
        ----------
        filepath : str
            Path of the folder containing the metadata in question.
        """
        old_ac_file = os.path.join(filepath, "autoCorrelations")
        new_ac_file = os.path.join(filepath, self._filetype)
        if not (os.path.exists(old_ac_file) and not os.path.exists(new_ac_file)):
            self._old_type = False
            super().fromfile(filepath)
            return

        self._old_type = True
        file_size = os.path.getsize(old_ac_file)
        hdr_dtype = np.dtype(
            [("antenna", "<i4"), ("nChunks", "<i4"), ("inhid", "<i4"), ("dhrs", "<f8")]
        )
        # Cast this here just to avoid any potential overflow issues w/ shorter ints
        nchunks = int(nchunks)
        rec_size = 4 * 16384 * nchunks * 2

        # This bit of code is to trap an unfortunately common problem with metadata
        # of MIR autos not being correctly recorded.
        if (file_size % (rec_size + hdr_dtype.itemsize)) != 0:
            # If the file size doesn't go in evenly, then read in just the first
            # record and try to figure it out.
            nchunks = int(np.fromfile(old_ac_file, dtype=hdr_dtype, count=1)["nChunks"])
            rec_size = 4 * 16384 * nchunks * 2
            assert (
                file_size % (rec_size + hdr_dtype.itemsize)
            ) == 0, "Could not determine auto-correlation record size."

        # Pre-allocate the metadata array,
        n_rec = file_size // (rec_size + hdr_dtype.itemsize)
        ac_data = np.zeros(2 * nchunks * n_rec, dtype=ac_dtype)

        # Set values that we know apriori
        ac_data["nch"] = 16384
        ac_data["isb"] = 1
        ac_data["correlator"] = 1

        # Grab some references to the values we need to plug in to.
        dataoff_arr = ac_data["dataoff"]
        antenna_arr = ac_data["antenna"]
        chunk_arr = ac_data["corrchunk"]
        antrx_arr = ac_data["antrx"]
        inhid_arr = ac_data["inhid"]
        dhrs_arr = ac_data["dhrs"]
        last_inhid = None

        with open(old_ac_file, "rb") as auto_file:
            for idx in range(n_rec):
                auto_vals = np.fromfile(
                    auto_file,
                    dtype=hdr_dtype,
                    count=1,
                    offset=rec_size if idx else 0,  # Skip offset on first iteration
                )
                if auto_vals["inhid"] != last_inhid:
                    rec_count = 0
                    last_inhid = auto_vals["inhid"]

                # Setup some slices that we'll use for plugging in values
                rxa_slice = slice(idx * 2 * nchunks, ((idx * 2) + 1) * nchunks)
                rxb_slice = slice(rxa_slice.stop, (idx + 1) * 2 * nchunks)
                ac_slice = slice(rxa_slice.start, rxb_slice.stop)

                # Plug in the entries that are changing on a per-record basis
                dhrs_arr[ac_slice] = auto_vals["dhrs"]
                antenna_arr[ac_slice] = auto_vals["antenna"]
                chunk_arr[rxa_slice] = chunk_arr[rxb_slice] = np.arange(1, nchunks + 1)
                antrx_arr[rxa_slice] = 0
                antrx_arr[rxb_slice] = 1
                inhid_arr[ac_slice] = last_inhid

                # Each auto record contains nchunks * nrec (always 2 here) spectra, each
                # one 16384 values of 4-bytes a piece. The offset position is then the
                # sum of the size of the precious records, plus the header size.
                dataoff_rec = (np.arange(2 * nchunks) * 4 * 16384) + hdr_dtype.itemsize
                # Also add the previous offset for the integration, and subtract 8
                # to account for the packdata header size (which we are spoofing).
                dataoff_rec += (rec_count * (rec_size + hdr_dtype.itemsize)) - 8

                # Now plug in the dataoff values
                dataoff_arr[ac_slice] = dataoff_rec
                rec_count += 1

        # Copy the corrchunk values to iband, since they should be the same here.
        ac_data["iband"] = ac_data["corrchunk"]
        self._data = ac_data
        self._mask = np.ones(len(self), dtype=bool)


########################################################################################


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

    def __init__(
        self,
        filepath=None,
        has_auto=False,
        load_vis=False,
        load_raw=False,
        load_auto=False,
    ):
        """
        Initialize a MirParser object.

        The full dataset can be quite large, as such the default behavior of
        this function is only to load the metadata. Use the keyword params to
        load other data into memory.

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        has_auto : bool
            flag to read auto-correlation data, default is False.
        load_vis : bool
            flag to load visibilities into memory, default is False.
        load_raw : bool
            flag to load raw data into memory, default is False.
        load_auto : bool
            flag to load auto-correlations into memory, default is False.
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

        self.raw_data = None
        self.vis_data = None
        self.auto_data = None
        self._has_auto = False
        self._tsys_applied = False

        # This value is the forward gain of the antenna (in units of Jy/K), which is
        # multiplied against the system temperatures in order to produce values in units
        # of Jy (technically this is the SEFD, which when multiplied against correlator
        # coefficients produces visibilities in units of Jy). Default is 130.0, which
        # is the estiamted value for SMA.
        self.jypk = 130.0

        # On init, if a filepath is provided, then fill in the object
        if filepath is not None:
            self.fromfile(
                filepath,
                has_auto=has_auto,
                load_vis=load_vis,
                load_raw=load_raw,
                load_auto=load_auto,
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
                if verbose:
                    print("%s does not exist in %s." % (item, target))

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
                if verbose:
                    print(
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
                    if verbose:
                        print(
                            f"{item} has different keys, left is {this_attr.keys()}, "
                            f"right is {other_attr.keys()}."
                        )
                    continue

                comp_list = data_comp_dict[item]

                # For the attributes with multiple fields to check, list them
                # here for convenience.
                for key in this_attr:
                    this_item = this_attr[key]
                    othr_item = other_attr[key]

                    is_same = True
                    for subkey in comp_list:
                        if subkey == "scale_fac":
                            is_same &= this_item[subkey] == othr_item[subkey]
                        elif not np.array_equal(this_item[subkey], othr_item[subkey]):
                            if this_item[subkey].shape == othr_item[subkey].shape:
                                # The atol here is set by the max value in the spectrum
                                # times 2^-10. That turns out to be _about_ the worst
                                # case scenario for moving to and from the raw data
                                # format, which compresses the data down from floats to
                                # ints.
                                is_same &= np.allclose(
                                    this_item[subkey],
                                    othr_item[subkey],
                                    atol=9.765625e-4
                                    * np.nanmax(np.abs(this_item[subkey]), initial=0),
                                    equal_nan=True,
                                )
                            else:
                                is_same = False
                    if not is_same:
                        is_eq = False
                        if verbose:
                            print("%s has the same keys, but different values." % item)
                        break
                # We are done processing the data dicts at this point, so we can skip
                # the item_same evauation below.
            elif issubclass(type(this_attr), MirMetaData):
                is_eq &= this_attr.__eq__(other_attr, verbose=verbose)
            elif item == "_metadata_attrs":
                if this_attr.keys() != other_attr.keys():
                    is_eq = False
                    if verbose:
                        print(
                            f"{item} has different keys, left is {this_attr.keys()}, "
                            f"right is {other_attr.keys()}."
                        )
            else:
                # We don't have special handling for this attribute at this point, so
                # we just use the generic __ne__ method.
                if this_attr != other_attr:
                    is_eq = False
                    if verbose:
                        print(
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
    def scan_int_start(filepath, allowed_inhid=None):
        """
        Read "sch_read" mir file into a python dictionary (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.

        Returns
        -------
        int_start_dict : dict
            Dictionary containing the indexes from sch_read, where keys match to the
            inhid indexes, and the values contain a two-element tuple, with the length
            of the packdata array (in bytes) the relative offset (also in bytes) of
            the record within the sch_read file.
        """
        file_size = os.path.getsize(filepath)
        data_offset = 0
        last_offset = 0
        int_dict = {}
        with open(filepath, "rb") as visibilities_file:
            while data_offset < file_size:
                int_vals = np.fromfile(
                    visibilities_file,
                    dtype=np.dtype([("inhid", "<i4"), ("nbyt", "<i4")]),
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
                    "record_size": int_vals["nbyt"],
                    "record_start": data_offset,
                }
                last_offset = int_vals["nbyt"].astype(int)
                data_offset += last_offset + 8

        return int_dict

    def _fix_int_start(self, datatype):
        """
        Fix an integration postion dictionary.

        Note that this function is not meant to be called by users, but is instead
        part of the low-level API for handling error correction when reading in data.
        This method will fix potential errors in an internal dictionary used to mark
        where in the main visibility file an individual spectral record is located.
        Under normal conditions, this routine does not need to be run, unless another
        method reported a specific error on read calling for the user to run this code.
        """
        for ifile, idict in self._file_dict.items():
            if not idict[datatype]["ignore_header"]:
                int_dict = copy.deepcopy(idict[datatype]["int_dict"])

            # Each file's inhid is allowed to be different than the objects inhid --
            # this is used in cases when combining multiple files together (via
            # concat). Here, we make a mapping of "file-based" inhid values to that
            # stored in the object.
            imap = {val["inhid"]: inhid for inhid, val in int_dict.items()}

            # Make the new dict by scaning the sch_read file.
            new_dict = self.scan_int_start(
                os.path.join(ifile, idict[datatype]["filetype"]), list(imap)
            )

            # Go through the individual entries in each dict, and update them
            # with the "correct" values as determined by scanning through sch_read
            for key in new_dict:
                int_dict[imap[key]] = new_dict[key]

            idict[datatype]["int_dict"] = int_dict

    @staticmethod
    def read_packdata(file_dict, inhid_arr, data_type="cross", use_mmap=False):
        """
        Read "sch_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        int_start_dict : dict
            indexes to the visibility locations within the file.
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
                [("inhid", "<i4"), ("nbyt", "<i4"), ("packdata", "B", int_size)]
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
                    # can tie mutiple file reads together into one call. The dtype
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
                    # Starting positino for a sequence of integrations
                    last_offset = int_start
                    # Size of record (make sure all records are the same size in 1 read)
                    last_size = int_size
                    # Number of integraions in the read
                    num_vals = 0
                    # Tally all the inhids in the read
                    inhid_list = []
                num_vals += 1
                inhid_list.append(ind_key)
            filename = os.path.join(filepath, indv_file_dict[data_type]["filetype"])
            # Time to actually read in the data
            if use_mmap:
                # kwargs = {
                #     "mode": "r",
                #     "offset": read_dict["start_offset"],
                #     "shape": (read_dict["num_vals"],),
                # }

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
                    good_check &= idict["nbyt"] == int_dict[inhid]["record_size"]

                if not good_check:
                    raise MirMetaError(
                        "File indexing information differs from that found in in "
                        "file_dict. Cannot read in %s data." % data_type
                    )

        if len(key_check) != 0:
            raise ValueError("inhid_arr contains keys not found in file_dict.")

        return int_data_dict

    @staticmethod
    def make_packdata(int_dict, recpos_dict, data_dict, data_type):
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
            metainformation about the secptral record. This dict is generally produced
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
            individual spectral record numbers, and each entry comtains a dict with two
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
            nominally matching the keys mentioned above), "nbyt" describing the size
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
                [("inhid", "<i4"), ("nbyt", "<i4"), ("packdata", "B", int_size)]
            )

        # Now we start the heavy lifting -- start looping through the individual
        # integrations and pack them together.
        int_data_dict = {}
        for inhid, int_subdict in int_dict.items():
            # Make an empty packdata dtype, which we will fill with new values
            int_data = np.empty((), dtype=int_dtype_dict[int_subdict["record_size"]])

            # Convenience dict which contains the sphids as keys and start/stop of
            # the slice for each spectral record as values for each integrtation.
            recpos_subdict = recpos_dict[inhid]

            # Plug in the "easy" parts of packdata
            int_data["inhid"] = inhid
            int_data["nbyt"] = int_subdict["record_size"]

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
    def convert_raw_to_vis(raw_dict):
        """
        Create a dict with visibilitity data via a raw data dict.

        Parameters
        ----------
        raw_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry comtains a dict
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
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "data", an array of np.complex64 containing the
            visibilities, and "flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).
        """
        # The code here was derived after a bunch of testing, trying to find the fastest
        # way to covert the compressed data into the "normal" format. Some notes:
        #   1) The setup below is actually faster than ldexp, probably because of
        #      the specific dtpye we are using.
        #   2) Casting 2 as float32 will appropriately cast sp_raw valuse into float32
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
        # faster than trying to much w/ the data above.
        for item in vis_dict.values():
            item["data"][item["flags"]] = 0.0

        return vis_dict

    @staticmethod
    def convert_vis_to_raw(vis_dict):
        """
        Create a dict with visibilitity data via a raw data dict.

        Parameters
        ----------
        vis_dict : dict
            A dictionary in the format of `vis_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "data", an array of np.complex64 containing the
            visibilities, and "flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).

        Returns
        -------
        raw_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "scale_fac", and np.int16 which describes the common
            exponent for the spectrum, and "data", an array of np.int16 (of length
            equal to twice that found in `sp_data["nch"]` for the corresponding value
            of sphid) containing the compressed visibilities.  Note that entries equal
            to -32768 aren't possible with the compression scheme used for MIR, and so
            this value is used to mark flags.
        """
        # Similar to convert_raw_to_vis, fair bit of testing went into making this as
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
        #      the specific dtpye we are using.
        #   2) Casting 2 as float32 saves on complex multiplies, and makes this run
        #      about 2x faster.
        #   3) The call to where here ensures that we plug in the "special" flag value
        #      whereever flags are detected.
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

    def read_data(self, data_type, return_vis=True, use_mmap=True, read_only=False):
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
            `sp_data`, and each entry comtains a dict with two items. If
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
            packdata_dict = self.read_packdata(
                self._file_dict, unique_inhid, data_type, use_mmap
            )
        except MirMetaError:
            # Catch an error that indicates that the metadata inside the vis file does
            # not match that in _file_dict, and attempt to fix the problem.
            warnings.warn(
                "Values in int_dict do not match that recorded inside the "
                "file for %s data. Attempting to fix this automatically." % data_type
            )
            self._fix_int_start(data_type)
            packdata_dict = self.read_packdata(
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
                        "data": packdata[start_idx + 1 : end_idx],
                    }
                else:
                    data_arr = packdata[start_idx:end_idx]
                    temp_dict[hid] = {"data": data_arr, "flags": np.isnan(data_arr)}

            if np.all(chan_avg_arr == 1):
                if return_vis:
                    temp_dict = self.convert_raw_to_vis(temp_dict)
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

    def write_cross_data(self, filepath, append_data=False, raise_err=True):
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
            If data are not loaded, and `raise_err=True`.
        UserWarning
            If data are not laoded, and `raise_err=False`. Also raised if tsys
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
            # applied, so that we mitigat ethe chance of a double-correction.
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
        # actually need to sgement sp_data by the integration ID.
        int_dict, sp_dict = self.sp_data._generate_recpos_dict(reindex=True)

        # We can now open the file once, and write each array upon construction
        with open(
            os.path.join(filepath, "sch_read"), "ab+" if append_data else "wb+"
        ) as file:
            for inhid in int_dict:
                if self.vis_data is None:
                    raw_dict = self.raw_data
                else:
                    raw_dict = self.convert_vis_to_raw(
                        {sphid: self.vis_data[sphid] for sphid in sp_dict[inhid]}
                    )

                packdata = self.make_packdata(
                    {inhid: int_dict[inhid]}, {inhid: sp_dict[inhid]}, raw_dict, "cross"
                )
                packdata[inhid].tofile(file)

    def write_auto_data(self, filepath, append_data=False, raise_err=True):
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
            If data are not loaded, and `raise_err=True`.
        UserWarning
            If data are not laoded, and `raise_err=False`. Also raised if tsys
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
        # actually need to sgement sp_data by the integration ID.
        int_dict, ac_dict = self.ac_data._generate_recpos_dict(reindex=True)

        # We can now open the file once, and write each array upon construction
        with open(
            os.path.join(filepath, "ach_read"), "ab+" if append_data else "wb+"
        ) as file:
            for inhid in int_dict:
                packdata = self.make_packdata(
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

        Parameteres
        -----------
        invert : bool
            If set to True, this will effectively undo the Tsys correction that has
            been applied. Default is False (convert uncalibrated visilibities to units
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
        # which techically has a different keyword under which the system temperatures
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
        # timestep, and receiver pairing.
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
        except (TypeError, KeyError):
            raise MirMetaError(
                "Missing spectral records in data attributes. Run load_data instead."
            )

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
        load_vis=None,
        load_raw=None,
        load_auto=None,
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
        load_vis : bool
            Load the visibility data (floats) into object. Default is True if
            `load_raw` is unset or otherwise set to False.
        load_raw : bool
            Load the raw visibility data (ints) into object. Default is False, unless
            `load_vis` is set to False, in which case it defaults to True.
        load_auto: bool
            Load the autos (floats) into object. Default is False.
        apply_tsys : bool
            Apply tsys corrections to the data. Only applicable if loading vis data.
            Default is True.
        allow_downselect : bool
            If data has been previously loaded, and all spectral records are currently
            contained in `vis_data`, `raw_data`, and/or `auto_data` (if `load_vis`,
            `load_raw`, and/or `load_auto` are True, respectively), then down-select
            from the currently loaded data rather than reading the data from disk.
        allow_conversion : bool
            Allow the method to convert previously loaded raw_data into "normal"
            visibility data. Default is True if the raw data is loaded and
            `load_vis=True`.
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

        Raises
        ------
        UserWarning
            If attempting to set both `load_vis` and `load_raw` to True. Also if the
            method is about to attempt to convert previously loaded data.
        """
        # Figure out what exactly we're going to load here.
        if load_vis is None and not load_raw:
            load_vis = True
        if load_raw is None:
            load_raw = not load_vis
        if load_auto is None:
            load_auto = self._has_auto

        if load_raw and load_vis:
            warnings.warn(
                "Cannot load raw and vis data simultaneously, loading vis data only."
            )
            load_raw = False

        # If there is no auto data to actually load, raise an error now.
        if load_auto and not self._has_auto:
            raise ValueError("This object has no auto-correlation data to load.")

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
            self.vis_data = self.convert_raw_to_vis(self.raw_data)
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
            if load_vis or load_raw:
                try:
                    self._downselect_data(
                        select_vis=load_vis, select_raw=load_raw, select_auto=False
                    )
                    load_vis = False
                    load_raw = False
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
        if load_vis or load_raw:
            data_dict = self.read_data(
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
            self.auto_data = self.read_data(
                "auto", return_vis=False, use_mmap=use_mmap, read_only=read_only
            )

    def unload_data(self, unload_vis=True, unload_raw=True, unload_auto=True):
        """
        Unload data from the MirParser object.

        Unloads the data-related attributes from memory, if they are loaded. Because
        these attributes can be formidible in size, this operation will substantially
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
        but instead is part of the low-level API for the MirParser object. Updates
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
        if not np.all(self.bl_data.get_mask()):
            key_list = set(
                self.bl_data.get_value(["iant1", "inhid"], return_tuples=True)
                + self.bl_data.get_value(["iant2", "inhid"], return_tuples=True)
            )

            if self._has_auto:
                key_list.union(
                    self.ac_data.get_value(["antenna", "inhid"], return_tuples=True)
                )

            mask_update |= self.eng_data.set_mask(header_key=key_list)

        # Now baseline -> int
        if not (np.all(self.bl_data.get_mask()) and not self._has_auto):
            good_inhid = set(self.bl_data["inhid"])
            if self._has_auto:
                good_inhid.union(self.ac_data["inhid"])
            mask_update |= self.in_data.set_mask(header_key=good_inhid)

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
                if update_data:
                    self.load_data(
                        load_vis=self.vis_data is not None,
                        load_raw=self.raw_data is not None,
                        load_auto=self.auto_data is not None,
                        apply_tsys=self._tsys_applied,
                        allow_downselect=False,
                        allow_conversion=False,
                    )
                else:
                    self.unload_data()
                    warnings.warn(
                        "Unable to update data attributes, unloading them now."
                    )

    def _clear_auto(self):
        """
        Remove attributes related to autos.

        This method is part of the internal API, and not meant for general users. It
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

        This method is part of the internal API, and not meant to be called by users.
        It's purpose is to reconstruct auto-correlation metadata based on that available
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

    def fromfile(
        self,
        filepath,
        has_auto=False,
        has_cross=True,
        load_vis=False,
        load_raw=False,
        load_auto=False,
    ):
        """
        Read in all files from a mir data set into predefined numpy datatypes.

        The full dataset can be quite large, as such the default behavior of
        this function is only to load the metadata. Use the keyword params to
        load other data into memory.

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        has_auto : bool
            flag to read auto-correlation data, default is False.
        load_vis : bool
            flag to load visibilities into memory, default is False.
        load_raw : bool
            flag to load raw data into memory, default is False.
        load_auto : bool
            flag to load auto-correlations into memory, default is False.
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
            attr.fromfile(filepath)

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
            if self.ac_data._old_type:
                # If we have the old-style file we are working with, then we need to
                # do two things: first, clean up entries that don't actually have any
                # data in them (the old format recorded lots of blank data to disk),
                # and plug in some missing metadata.
                self._fix_acdata()
                filetype = "autoCorrelations"
            int_dict, self._ac_dict = self.ac_data._generate_recpos_dict()

            file_dict["auto"] = {
                "int_dict": int_dict,
                "filetype": filetype,
                "ignore_header": self.ac_data._old_type,
            }

        self._file_dict = {filepath: file_dict}
        self.filepath = filepath

        # Set/clear these to start
        self.vis_data = self.raw_data = self.auto_data = None
        self._tsys_applied = False

        # If requested, now we load up the visibilities.
        self.load_data(load_vis=load_vis, load_raw=load_raw, load_auto=load_auto)

    def tofile(
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
        load_data : bool
            If set to True, load the raw visibility data. Default is False, which will
            forgo loading data. Note that if no data are loaded, then the method
            will then write out a metadata-only object.
        append_data : bool
            When called, this method will generally overwrite any prior MIR data
            located in the target directory. If set to True, this will allow the method
            to append data instead. Note that appending will only work correctly if not
            attempting to combine datasets with overlapping integration (inhid values).
        append_codes : bool
            Generally the `codes_data` information remains static over the course of
            a track, and thus only needs to be written out once. However, if set to
            True, then the information in `codes_data` will be appended to the
            "codes_read" file. Default is False, and users are recommended to exercise
            caution using this switch (and leave it alone unless certain it is
            required), as it can corrupt. Only used if `append_data=True`.
        bypass_append_check : bool
            Normally, if seting `append_data=True`, the method will check to see that
            there are no clashes in terms of header ID numbers. However, if set to
            True, this option will bypass this check. Default is False, and users
            should exercise caution using this switch (and leave it alone unless
            certain it is required) as it can corrupt a dataset. Only used if
            `append_data=True`.

        Raises
        ------
        UserWarning
            If only metadata is loaded in the MirParser object.
        """
        # If no directory exists, create one to write the data to
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # Check that the data are loaded
        if load_data:
            self.load_data(
                load_vis=False, load_raw=self._has_cross, load_auto=self._has_auto
            )

        # Write out the various metadata fields
        for attr in self._metadata_attrs:
            if attr in ["sp_data", "ac_data"]:
                mir_meta_obj = self._metadata_attrs[attr].copy()
                mir_meta_obj._recalc_dataoff()
            else:
                mir_meta_obj = self._metadata_attrs[attr]

            mir_meta_obj.tofile(
                filepath,
                overwrite=overwrite,
                append_data=append_data,
                check_index=check_index,
            )

        # Finally, we can package up the data in order to write it to disk.
        if self._has_cross:
            self.write_cross_data(filepath, append_data=append_data, raise_err=False)

        if self._has_auto:
            self.write_auto_data(filepath, append_data=append_data, raise_err=False)

    @staticmethod
    def _rechunk_data(data_dict, chan_avg_arr, inplace=False):
        """
        Rechunk regular visibility spectra.

        Note this routine is not intended to be called by users, but instead is a
        low-level call from the `rechunk` method of MirParser to spectrally average
        data.

        Parameters
        ----------
        vis_dict : dict
            A dict containing visibility data, where the keys match to individual values
            of `sphid` in `sp_data`, with each value being its own dict, with keys
            "data" (the visibility data, dtype=np.complex64) and "flags"
            (the flagging inforformation, dtype=bool).
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
            # new channel, so that we can normalize apporpriately later. Note we cast
            # to float32 here, since the data are complex64 (and so there's no extra
            # casting required, but we get the benefit of only mutiplyng real-only and
            # complex data).
            temp_count = good_mask.sum(axis=-1, dtype=np.float32)

            # Need to mask out when we have no counts, since it'll produce a divide
            # by zero error. As an added bonus, this will let us zero out any channels
            # without any valid visilibies.
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
            (the flagging inforformation, dtype=bool).

        Returns
        -------
        data_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `raw_dict` (unless `return_vis=True`).
        """
        if raw_dict is None:
            return

        # If inplace, point our new dict to the old one, otherwise create
        # an ampty dict to plug values into.
        data_dict = raw_dict if inplace else {}

        for chan_avg, (sphid, sp_raw) in zip(chan_avg_arr, raw_dict.items()):
            # If the number of channels to average is 1, then we just need to make
            # a deep copy of the old data and plug it in to the new dict.
            if chan_avg == 1:
                if (not inplace) or return_vis:
                    data_dict[sphid] = (
                        MirParser.convert_raw_to_vis({0: sp_raw})[0]
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
                    MirParser.convert_raw_to_vis({0: sp_raw}),
                    [chan_avg],
                    inplace=True,
                )[0]
            else:
                data_dict[sphid] = MirParser.convert_vis_to_raw(
                    MirParser._rechunk_data(
                        MirParser.convert_raw_to_vis({0: sp_raw}),
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
        band_dict = self.codes_data["band"]
        chanavg_dict = {
            key: 1 if value[0] == "c" else chan_avg
            for key, value in band_dict.items()
            if isinstance(value, str)
        }

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
        different calls to `select` being run. In the second scenarion, which we call
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
            If attemting to add a MirParser object with any other type of object.
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
                    "righthand object in the add sequence to overwrite that from the "
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

            # Final check - see if the MJDs line up exactly, since that _shouldn't_
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
            # creating a copy of the other object, that we can make udpates to.
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
            # key field that is uses, so we know which upadte dict to use.
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

        # Finaly, if we have discrepant _has_auto states, we force the resultant object
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
        other_obj : MirParser object
            Other MirParser object to combine with this data set.
        overwrite : bool
            If set to True, metadata from `other_obj` will overwrite that present in
            this object, even if they differ. Default is False.
        merge : bool
            TODO

        Raises
        ------
        TypeError
            If attemting to add a MirParser object with any other type of object.
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
        Select a subset of data inside a Mir-formated file.

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
            data. Each tuple must be 3 elements in length. The first element should
            match one of the field names inside one of the metadata attributes (e.g.,
            "ant1", "mjd", "source", "fsky"). The second element specifies the
            comparison operator, which is used to compare the metadata field against
            the third element in the tuple. Allowed comparisons include:
            "eq" (equal to, matching any in the third element),
            "ne" (not equal to, not matching any in third element),
            "lt" (less than the third element),
            "le" (less than or equal to the third element),
            "gt" (greater than the third element),
            "ge" (greater than or equal to the third element),
            "btw" (between the range given by two values in the third element),
            "out" (outside of the range give by two values in the third element).
            Multiple tuples to where can be supplied, where the results of each
            selection are combined based on the value of `and_where_args`.
        and_where_args : bool
            If set to True, then the individual calls to the `where` method will be
            combined via an element-wise "and" operator, such that the returned array
            will report the positions where all criterea are met. If False, results
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

        # Make sure that where is a list, to make arg parsing more uniform downsteam
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
                # arbitary strings to integer values under specific header names. If
                # we have an argument that matches once of these, we want to substitute
                # the string and field name for the appropriate integrer (and associated
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
                # If not of the field indentified in the sequence of tuples were found
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
            gains tavles.

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

            # Once we've divied up the solutions, plug this dict back into the main one.
            compass_soln_dict["bandpass_gains"] = bandpass_gains

            # Now, we can move on to flags. Note that COMPASS doesn't have access to
            # the integration header IDs, so we have to do a little bit of matching
            # based on the timestamp of the data in COMPASS vs MIR (via the MJD).
            mjd_compass = np.array(file["mjdArr"][0])
            mjd_mir = self.in_data["mjd"]
            inhid_arr = self.in_data["inhid"]

            # Match each index to an inhid entry
            index_dict = {}
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
        for unaveraged data. Be aware that this routine will modify values stored
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
                "`load_data(load_vis=True)` to fix this issue."
            )

        # Before we do anything else, we want to be able to map certain entires that
        # are per-blhid to be per-sphid.
        sp_bl_map = self.bl_data._index_query(header_key=self.sp_data["blhid"])

        # Now grab all of the metadata we want for processing the spectral records
        sphid_arr = self.sp_data["sphid"]  # Spectral window header ID
        ant1_arr = self.bl_data.get_value("iant1", index=sp_bl_map)  # Ant 1 Number
        rx1_arr = self.bl_data.get_value("ant1rx", index=sp_bl_map)  # Pol | 0:X/L 1:Y/R
        ant2_arr = self.bl_data.get_value("iant2", index=sp_bl_map)  # Ant 2 Number
        rx2_arr = self.bl_data.get_value("ant2rx", index=sp_bl_map)  # Pol | 0:X/L 1:Y/R
        chunk_arr = self.sp_data["corrchunk"]  # Correlator window number
        sb_arr = self.bl_data.get_value("isb", index=sp_bl_map)  # Sidebad | 0:LSB 1:USB

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
                # be basically calcuating the gains solns on an "as needed" basis.
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
                    # multipling the gains for ant1 by the complex conj of the gains
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
            # For sake of reading/coding, let's assign assign the two catalogs of flags
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
        frequency number by an arbitary amount (i.e., not necessarily an integer
        number of channels).

        chan_shift : float
            Number of channels that the spectrum is to be shifted by, where postive
            values indiciate that channels are moving "up" to higher index positions.
            No default.
        kernel_type : str
            There are several supported interpolation schemes that can be used for
            shifting the spectrum a given number of channels. The three supported
            schemes are "nearest" (nearest-neighbor; choose the closest channel to the
            one desired), "linear" (linear interpolation; interpolate between the two
            closest channel), and "cubic" (cubic convulution; see "Cubic Convolution
            Interpolation for Digital Image Processing" by Robert Keys for more details
            ). Nearest neighbor is the fastest, although cubic convolution generally
            provides the best spectral PSF.
        alpha_fac : float
            Only used when `kernel_type="cubic"`, adjusts the alpha parameter for
            the cubic convolution kernel. Typical values are usually in the range of
            -1 to -0.5, the latter of which is the default value due to the compactness
            of the PSF using this kernel.
        tol : float
            If the desired frequency shift is close enough to an interger number of
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
            raise ValueError("tol must be in the range [0, 0.5].")

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
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "data", an array of np.complex64 containing the
            visibilities, and "flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).
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
                # If the shift kernal is None, it means that we only have a coarse
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
                # drop (kernal_size - 1) elements from the array, where the number of
                # elements dropped on the left side is 1 more than it is on the right.
                l_edge = (kernel_size // 2) + coarse_shift
                r_edge = (1 - (kernel_size // 2)) + coarse_shift

                # These clip values here are used to clip the original array to both
                # make sure that the size matches, and to avoid doing any unneccessary
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
            individual values of sphid in `sp_data`, and each entry comtains a dict
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
            (the flagging inforformation, dtype=bool). This option is ignored if
            `inplace=True`.

        Returns
        -------
        data_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `raw_dict` (unless `return_vis=True`).
        """
        # If inplace, point our new dict to the old one, otherwise create
        # an ampty dict to plug values into.
        data_dict = raw_dict if inplace else {}
        return_vis = (not inplace) and return_vis

        for shift_tuple, (sphid, sp_raw) in zip(shift_tuple_list, raw_dict.items()):
            # If we are not actually shifting the data (which is what the tuple
            # (0,0,0,None) signifies), then we can bypass most of the rest of the
            # code and simply return a copy of the data if needed.
            if shift_tuple == (0, 0, None):
                if not inplace:
                    data_dict[sphid] = (
                        MirParser.convert_raw_to_vis({0: sp_raw})[0]
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
                    MirParser.convert_raw_to_vis({0: sp_raw}),
                    [shift_tuple],
                    flag_adj=flag_adj,
                    inplace=False,
                )[0]
            else:
                data_dict[sphid] = MirParser.convert_vis_to_raw(
                    MirParser._chanshift_vis(
                        MirParser.convert_raw_to_vis({0: sp_raw}),
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
            closest channel), and "cubic" (cubic convulution; see "Cubic Convolution
            Interpolation for Digital Image Processing" by Robert Keys for more details
            ). Nearest neighbor is the fastest, although cubic convolution generally
            provides the best spectral PSF.
        tol : float
            If the desired frequency shift is close enough to an interger number of
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
                # value though just turns out to be the the RxA value multipled by
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
