# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Module for low-level interface to Mir metadata files.

This module provides a python interface for individual Mir metadata files, e.g.
"in_read", "bl_read", "sp_read", "we_read", "eng_read", "antennas", and "codes_read".
"""
import copy
import os
import warnings
from pathlib import Path

import numpy as np

__all__ = [
    "MirMetaData",
    "MirInData",
    "MirBlData",
    "MirSpData",
    "MirEngData",
    "MirWeData",
    "MirCodesData",
    "MirAntposData",
    "MirAcData",
]

# Define the packed data header fields for cross-correlations (should be static)
OLD_VIS_HEADER = [
    ("inhid", ">i4"),
    ("form", "S4"),
    ("nbytes", ">i4"),
    ("record_size", ">i4"),
]
NEW_VIS_HEADER = [("inhid", "<i4"), ("record_size", "<i4")]

# Define packed data vis format, namely that they flip endianness
OLD_VIS_DTYPE = np.dtype(">i2")
NEW_VIS_DTYPE = np.dtype("<i2")

# Define padded entries per spectra -- used to be a feature in older vis data
OLD_VIS_PAD = 4
NEW_VIS_PAD = 0

# Define the packed data header fields for autocorrelations (should be static)
OLD_AUTO_HEADER = [
    ("antenna", "<i4"),
    ("nchunks", "<i4"),
    ("inhid", "<i4"),
    ("dhrs", "<f8"),
]
NEW_AUTO_HEADER = NEW_VIS_HEADER  # Creating this in case we want to change later

# Define packed data vis format, which should be small endian floats
OLD_AUTO_DTYPE = np.dtype("<f4")
NEW_AUTO_DTYPE = OLD_AUTO_DTYPE  # Creating this in case we want to change later

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
        # Sideband code (matched to sb in codes_read; usually 0=LSB, 1=USB)
        ("isb", np.int16),
        # Polarization code (matched to pol in codes_read)
        ("ipol", np.int16),
        # Ant1 receiver number (0 = RxA, 1 = RxB)
        ("ant1rx", np.int16),
        # Ant2 receiver number (0 = RxA, 1 = RxB)
        ("ant2rx", np.int16),
        # Pointing status (1 = offset pointing, 0 = target at pri beam center)
        ("pointing", np.int16),
        # Receiver code (matched to rec in codes_read)
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
        # Band code (matched to band in codes_read, usually equal to corrchunk)
        ("iband", np.int16),
        # Polarization state code (matched to pstate in codes_read, not used)
        ("ipstate", np.int16),
        # Opacity at 225 GHz (nepers)
        ("tau0", np.float32),
        # Velocity at source rest frame at band center (km/s)
        ("vel", np.float64),
        # Velocity resolution at source rest frame at band center (km/s)
        ("vres", np.float32),
        # Sky frequency at band center (GHz)
        ("fsky", np.float64),
        # Channel resolution (MHz)
        ("fres", np.float32),
        # Gunn frequency (GHz)
        ("gunnLO", np.float64),
        # Cabin BDA LO frequency (GHz, not used anymore)
        ("cabinLO", np.float64),
        # Second down-converter frequency (GHz, not used anymore)
        ("corrLO1", np.float64),
        # Final down-converter frequency (GHz)
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
        # DDS operating mode, matched to 'ddsmode' in codes_read
        ("iddsmode", np.int16),
        # Gunn multiplier for the receiver
        ("gunnMult", np.int16),
        # Spectral band continuum amplitude (corr co-eff)
        ("amp", np.float32),
        # Spectral band continuum phase (rad)
        ("phase", np.float32),
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
# the track, although a few commonly used codes do vary integration by integration.

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
        # Wind direction measured at each antenna (rad, -1 if no hardware)
        ("windDir", np.float32, 11),
        # Bore sight PWV measured at each antenna (mm, -1 if no hardware)
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
        # Receiver code (matched to rec in codes_read)
        ("irec", np.int16),
        # Polarization code (matched to pol in codes_read)
        ("ipol", np.int16),
        # Sideband code (matched to sb in codes_read; usually 0=LSB, 1=USB)
        ("isb", np.int16),
        # Band code (matched to band in codes_read, usually equal to corrchunk)
        ("iband", np.int16),
        # Correlator chunk (spectral window number)
        ("corrchunk", np.int16),
        # Correlator number (0 = ASIC; 1 = SWARM)
        ("correlator", np.int32),
        # Sky frequency at band center (GHz)
        ("fsky", np.float64),
        # Gunn frequency (GHz)
        ("gunnLO", np.float64),
        # Second down-converter frequency (GHz, not used currently)
        ("corrLO1", np.float64),
        # Final down-converter frequency (GHz)
        ("corrLO2", np.float64),
        # DDS frequency offset on nominal Gunn LO (GHz)
        ("fDDS", np.float64),
        # Channel resolution (MHz)
        ("fres", np.float32),
        # Number of channels
        ("nch", np.int32),
        # Offset from the start of the spectral record of the packed data.
        ("dataoff", np.int32),
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

    def __init__(self, message="A MirMetaData object has a problem."):
        super().__init__(message)


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
        obj=None,
        *,
        filetype=None,
        dtype=None,
        header_key_name=None,
        binary_dtype=None,
        pseudo_header_key_names=None,
    ):
        """
        Initialize a MirMetaData object.

        Parameters
        ----------
        obj : str or Path or ndarray or int
            Optional argument used to specify how to initialize the object. If a str or
            Path is supplied, then it is treated as the path to the Mir data folder
            containing the metadata. If an int is supplied, a "blank" (zero-filled)
            array of metadata is generated, with length of `obj`. If an ndarray is
            supplied, then the supplied array is used as the underlying data set for the
            object (where dtype of the array must match that appropriate for the
            object).
        filetype : str
            Name of the type MirMetaData object, which is then used as the file name
            that is read from/written to within the folder specified by the path.
        dtype : dtype
            Numpy-based description of the binary data stored in the file.
        header_key_name : str or None
            Field inside of `dtype` which contains a unique indexing key for the
            metadata in the file. Typically used to reference values between MirMetaData
            objects. If set to `None`, no field is used for indexing.
        pseudo_header_key_names : list of str or None
            Required if `header_key_name` is `None`. Used to identify a group of fields,
            which when taken in combination, can be used as a unique identifier. Can
            be set either to `None` (if not used) or otherwise a tuple of strings.
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

        if obj is None:
            return
        if isinstance(obj, (str, Path)):
            self.read(filepath=obj)
            return

        if isinstance(obj, int):
            self._data = np.zeros(obj, dtype=dtype)
        elif isinstance(obj, np.ndarray):
            self._data = obj
            if self.dtype is None:
                self.dtype = obj.dtype
            assert self.dtype == obj.dtype, "ndarray dtype must match object dtype."
        self._mask = np.ones(self._size, dtype=bool)
        self._set_header_key_index_dict()

    @property
    def _size(self):
        """Return length of full data array."""
        return 0 if self._data is None else self._data.size

    @property
    def _identifier(self):
        """Return unique identifier field name(s)."""
        return self._pseudo_header_key if self._header_key is None else self._header_key

    def __iter__(self):
        """
        Iterate over MirMetaData attributes.

        Yields
        ------
        data_slice : ndarray
            Value(s) at a given position in the data array with dtype equal to that in
            the `dtype` attribute of the object.
        """
        for idx in np.where(self._mask)[0]:
            yield self._data[idx]

    def __len__(self):
        """
        Calculate the number of unmasked entries in the data table.

        Returns
        -------
        len : int
            Number of unique entries contained within the meta data.
        """
        # TODO: Consider caching value as usage expands
        return np.sum(self._mask)

    def __eq__(
        self,
        other=None,
        *,
        verbose=False,
        ignore_params=None,
        use_mask=False,
        comp_mask=True,
    ):
        """
        Compare MirMetaData objects for equality.

        Parameters
        ----------
        other : MirMetaData object
            Object of the same type to compare to.
        verbose : bool
            Whether to print out the differences between the two objects, if any are
            found. Default is True.
        ignore_params : list of str or None
            Optional argument, which can be used to specify whether to ignore certain
            attributes when comparing objects. By default, no attributes are ignored.
        use_mask : bool
            Whether or not to ignore the internal mask when performing the comparison.
            If set to True, will only compare those values where the mask is set to
            True. Default is False.
        comp_mask : bool
            Whether or not to compare the masks themselves when comparing the individual
            objects. Default is True.

        Returns
        -------
        is_eq : bool
            Value describing whether or not the two objects contain the same data.
        """
        # Grab the name of the class to make the output a bit more human parsable
        name = type(self).__name__

        if not isinstance(other, self.__class__):
            raise ValueError(
                "Cannot compare {this_type} with {other_type}.".format(
                    this_type=name, other_type=type(other).__name__
                )
            )

        verbose_print = print if verbose else lambda *a, **k: None

        if (self._data is None or self._mask is None) or (
            other._data is None or other._mask is None
        ):
            is_eq = (self._data is None) == (other._data is None)
            is_eq &= (self._mask is None) == (other._mask is None)
            if not is_eq:
                verbose_print(
                    "%s objects are not both initialized (one is empty)." % name
                )
            return is_eq

        this_keys = self.get_header_keys(use_mask=use_mask)
        other_keys = other.get_header_keys(use_mask=use_mask)

        if set(this_keys) != set(other_keys):
            verbose_print("%s object header key lists are different." % name)
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

        # Start with the mask comparison first.
        if comp_mask and not np.array_equal(self._mask, other._mask):
            verbose_print("%s masks are different." % name)
            is_eq = False

        # Move on to the data comparisons
        for item in comp_fields:
            left_vals = self.get_value(item, index=this_idx)
            right_vals = other.get_value(item, index=other_idx)

            if not np.array_equal(left_vals, right_vals):
                is_eq = False
                verbose_print(
                    "%s of %s is different, left is %s, right is %s."
                    % (item, name, left_vals, right_vals)
                )
                if not verbose:
                    break

        return is_eq

    def __ne__(
        self,
        other=None,
        *,
        verbose=False,
        ignore_params=None,
        use_mask=False,
        comp_mask=True,
    ):
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
        use_mask : bool
            Whether or not to ignore the internal mask when performing the comparison.
            If set to True, will only compare those values where the mask is set to
            True. Default is False.
        comp_mask : bool
            Whether or not to compare the masks themselves when comparing the individual
            objects. Default is True.

        Returns
        -------
        is_ne : bool
            Value describing whether the two objects do not contain the same data.
        """
        return not self.__eq__(
            other,
            verbose=verbose,
            ignore_params=ignore_params,
            use_mask=use_mask,
            comp_mask=comp_mask,
        )

    def __add__(
        self, other, *, inplace=False, merge=None, overwrite=None, discard_flagged=False
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
        if not isinstance(other, self.__class__):
            raise ValueError("Both objects must be of the same type.")

        if other._data is None:
            # If no data is loaded, then this is just a no-op
            return self if inplace else self.copy()

        # At this point, we should be able to combine the two objects
        new_obj = self if inplace else self.copy()

        if self._data is None:
            new_obj._data = other._data.copy()
            new_obj._mask = other._mask.copy()
        else:
            idx1, idx2, mask1, mask2 = self._add_check(
                other, merge=merge, overwrite=overwrite, discard_flagged=discard_flagged
            )
            new_obj._data = np.concatenate((new_obj._data[idx1], other._data[idx2]))
            new_obj._mask = np.concatenate((mask1, mask2))

        # Make sure the data is sorted correctly, generate the header key -> index
        # position dictionary.
        new_obj._set_header_key_index_dict()

        # Finally, clear out any sorted values, since there's no longer a good way to
        # carry them forward.
        new_obj._stored_values = {}

        return new_obj

    def __iadd__(self, other, *, merge=None, overwrite=None, discard_flagged=False):
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
            Copy of the original object.
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

    def where(
        self,
        select_field=None,
        select_comp=None,
        select_val=None,
        *,
        mask=None,
        return_header_keys=False,
    ):
        """
        Find where metadata match a given set of selection criteria.

        This method will produce a masking screen based on the arguments provided to
        determine which entries match a given set of conditions.

        Parameters
        ----------
        select_field : str
            Field in the metadata to evaluate.
        select_comp : str
            Specifies the type of comparison to do between the value supplied in
            `select_val` and the metadata. No default, allowed values include:
            "eq" or "==" (equal to);
            "ne" or "!=" (not equal to);
            "lt" or "<" (less than);
            "le" or "<=" (less than or equal to);
            "gt" or ">" (greater than);
            "ge" or ">=" (greater than or equal to);
            "between" (between a range of values);
            "outside" (outside of a range of values).
        select_val : number of str, or sequence of number or str
            Value(s) to compare data in `select_field` against. If `select_comp` is
            "lt", "le", "gt", "ge", then this must be a single number. If `select_comp`
            is "between" or "outside", then this must be a list 2 numbers. If
            `select_comp` is "eq"/"==" or "ne"/"!=", then this can be either a single
            value (number or string) or a sequence of numbers.
        mask : ndarray of bool
            Optional argument, of the same length as the MirMetaData object, which is
            applied to the output of the selection parsing through an element-wise
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
            "between": lambda val, lims: ((val >= lims[0]) & (val <= lims[1])),
            "outside": lambda val, lims: ((val < lims[0]) | (val > lims[1])),
        }

        if isinstance(select_val, (list, set, tuple, str, np.ndarray, np.str_)):
            op_dict["eq"] = lambda val, comp: np.isin(val, comp)
            op_dict["ne"] = lambda val, comp: np.isin(val, comp, invert=True)

        op_dict["=="] = op_dict["eq"]
        op_dict["!="] = op_dict["ne"]
        op_dict["<"] = op_dict["lt"]
        op_dict["<="] = op_dict["le"]
        op_dict[">"] = op_dict["gt"]
        op_dict[">="] = op_dict["ge"]

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
        class, and is not designed for users. This function will report back the index
        positions in the `_data` attribute where the given selection criteria are met.

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
            will report the positions where all criteria are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        header_key : sequence of ints or tuples
            Header key values to get the index position for, which are always recorded
            as ints. If the object has no header key, but instead a pseudo header key,
            a sequence of tuples (matching the pseudo keys) should be supplied.
        index : sequence of ints
            Index positions of the array. Note that this is typically what you are
            calling this method for, but is included as an argument to simplify
            argument processing for various calls.

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
            `where` is not a 3-element tuple or sequence of 3-element tuples.
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
            self._mask.copy() if use_mask else np.full(self._size, bool(and_where_args))
        )

        # To reach this point, we must have supplied an argument to where. Use that
        # method to build a mask that we can use to select the data on.  First check
        # that the where argument matches what we expect - either a tuple or a sequence
        # of tuples.
        try:
            if not (isinstance(where[0], (tuple, list))):
                # If where is not indexable, it'll raise a TypeError here.
                # Force this to be a sequence of tuples here so that the logic below is
                # simplified.
                where = [where]
            for item in where:
                # Note we raise a TypeError in this loop to trap an identical bug,
                # namely that the user has not provided a valid argument for where.
                if len(item) != 3:
                    raise TypeError
        except TypeError as err:
            raise ValueError(
                "Argument for where must be either a 3-element tuple, or sequence "
                "of 3-element tuples."
            ) from err

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
        *,
        use_mask=None,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
        return_tuples=None,
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
            will report the positions where all criteria are met. If False, results
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
            specified selection criteria. Default is to return tuples if multiple fields
            are being extracted, otherwise to return an ndarray of values.

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
            if return_tuples is None:
                return_tuples = True
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

    def __getitem__(self, field_name):
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
            Values for the specified field name where the selection criteria match. If
            a list of fields is supplied, then a list of arrays is returned.
        """
        return self.get_value(field_name=field_name, return_tuples=False)

    def set_value(
        self,
        field_name=None,
        value=None,
        *,
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
            will report the positions where all criteria are met. If False, results
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
                "Changing fields that tie to header keys can result in unpredictable "
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

    def __setitem__(self, field_name, value):
        """
        Set values for a particular field using set_value.

        field_name : str
            Fields from which to extract data. Must match a field name, as listed in the
            `dtype` attribute of the object.
        value : ndarray
            Values to set the field in question to, where the provided selection
            criteria match. Shape of the array must be broadcastable to either the shape
            of the internal mask or to the shape of the `index` or `header_key`
            arguments.
        """
        self.set_value(field_name=field_name, value=value)

    def _generate_mask(
        self,
        *,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
        inverse=False,
    ):
        """
        Generate a boolean mask based on selection criteria.

        Note that this is an internal helper function for other methods, which is not
        intended for general user use. Generates a boolean mask to based on the
        selection criteria (where the array is set to True when the selection criteria
        are met).

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
            will report the positions where all criteria are met. If False, results
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
        inverse : bool
            Optional argument to invert the selection mask (i.e., set mask to False
            where selection conditions are met rather than True). Default is False.

        Returns
        -------
        mask_arr : ndarray of bool
            Array of boolean values, with length equal to that of the object itself.
        """
        idx_arr = self._index_query(False, where, and_where_args, header_key, index)

        # If selecting all array elements, just flip the operation
        if idx_arr is Ellipsis:
            idx_arr = []
            inverse = not inverse

        gen_func = np.ones if inverse else np.zeros
        new_mask = gen_func(self._size, dtype=bool)

        new_mask[idx_arr] = not inverse
        return new_mask

    def get_mask(self, *, where=None, and_where_args=True, header_key=None, index=None):
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
            will report the positions where all criteria are met. If False, results
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
        *,
        mask=None,
        where=None,
        and_where_args=True,
        header_key=None,
        index=None,
        reset=False,
        and_mask=True,
        use_mask=True,
    ):
        """
        Set the internal object mask.

        This function updates the internal mask based on the supplied selection
        criteria. This internal mask is primarily used to identify which rows of data
        are "active", and will affect what some methods return to the user.

        Parameters
        ----------
        mask : ndarray of bool
            Optional argument, typically of the same length as the MirMetaData object,
            where True marks which index positions to set. Setting this will cause any
            arguments passed to `where`, `header_key`, and `index` to be ignored.
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
            will report the positions where all criteria are met. If False, results
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
        use_mask : bool
            Only used if an argument for `mask` is supplied. If set to False, the
            supplied mask is applied to the mask attribute directly, rather than only
            to elements where the underlying mask is already set to True. This means
            that the length of `mask` will need to be equal to the `_mask` attribute,
            rather than the MirMetaData object. Default is True, which covers most
            typical use cases.
        """
        if mask is None:
            mask = self._generate_mask(
                where=where,
                and_where_args=and_where_args,
                header_key=header_key,
                index=index,
            )
        elif use_mask and not np.all(self._mask):
            temp_mask = np.zeros_like(self._mask)
            temp_mask[self._mask] = mask
            mask = temp_mask

        if not (reset or np.all(self._mask)):
            mask = (self._mask & mask) if and_mask else (self._mask | mask)

        if np.array_equal(self._mask, mask):
            return False
        else:
            self._mask = mask
            return True

    def get_header_keys(
        self,
        *,
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
            will report the positions where all criteria are met. If False, results
            are instead combined via an element-wise "or" operator. Default is True.
            If supplied, the argument for `mask` will be combined with the output from
            the calls to `where` with the same logic.
        index : sequence of int
            Optional argument, specifies the index positions at which to extract data
            from the meta data. Cannot be specified with `header_key` or `where`.
        force_list : False
            Normally the header keys are returned as an iterable (ndarray in there
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
        keys = self.get_value(
            self._identifier,
            use_mask=use_mask,
            where=where,
            and_where_args=and_where_args,
            index=index,
            return_tuples=False if force_list else None,
        )

        if force_list and not isinstance(keys, list):
            keys = [keys]

        return keys

    def _set_header_key_index_dict(self):
        """
        Set internal header key to index position dictionary attribute.

        Note that this is an internal helper function, not intended for general users.
        Generates a dictionary that can be used for mapping header key values to index
        positions inside the data array.
        """
        self._header_key_index_dict = self.group_by(
            self._identifier, use_mask=False, return_index=True, assume_unique=True
        )

    def _generate_new_header_keys(self, other):
        """
        Create an updated set of header keys for a MirMetaData object.

        Note that this function is not meant to be called by users, but instead is
        low-level helper function for the object. This function allows for one to
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
        if not isinstance(other, self.__class__):
            raise ValueError("Both objects must be of the same type.")

        # If no data are loaded, or if there is no header key, then this is basically
        # a no-op -- hand back an empty dictionary in this case.
        if (self._header_key is None) or (self._data is None):
            return {}

        idx_start = np.max(other._data[other._header_key]) + 1
        idx_stop = idx_start + self._size

        index_dict = {
            self._header_key: dict(
                zip(self.get_header_keys(use_mask=False), range(idx_start, idx_stop))
            )
        }

        return index_dict

    def _sort_by_header_key(self):
        """
        Sort data array by header key values.

        Note that this function is not designed to be called by users, but instead is
        a low-level helper function for the object. Calling this function will sort the
        metadata in the `_data` attribute by the (pseudo) header key, and will
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
        group_fields=None,
        *,
        use_mask=True,
        where=None,
        header_key=None,
        return_index=None,
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
            If True, consider only data where the internal mask is marked True. Default
            is True.
        where : tuple of sequence of tuples
            Optional argument, each tuple is used to call the `where` method to identify
            which index positions match the given criteria. Can be supplied as a
            sequence of tuples rather than a single tuple, but each much be of length
            3, where the first argument is the `select_field` argument, the second is
            the `select_comp` argument, and the last is the `select_val` argument. See
            the documentation of `where` for more details. Cannot be specified with
            `index` or `header_key`.
        header_key : sequence of ints or tuples
            Optional argument, values to match against the header key field, in order to
            determine which entries of the array to extract. For example, if the header
            key field "hid" has the values [2, 4, 6, 8], setting this argument to [2, 8]
            will set the mask at index positions [0, 3] to True. Cannot be specified
            with `index` or `where`.
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
            ndarray of header key values (if `return_index=True` and the object has a
            valid  header key), or a list of tuples (if `return_index=True` and the
            object only has a pseudo-index), which correspond to the metadata entries
            that match the unique key.
        """
        # Check to make sure arguments are compatible
        if (header_key is not None) and return_index:
            raise ValueError("Cannot specify header_key and set return_index=True.")

        # Make this a list just to make it easier to program against.
        if isinstance(group_fields, str):
            group_fields = [group_fields]

        # Get the data we want to group by and then use lexsort to arrange the data
        # in order. This turns out to make extracting the index positions much faster.
        group_data = self.get_value(
            group_fields,
            use_mask=use_mask,
            where=where,
            header_key=header_key,
            return_tuples=False,
        )
        index_arr = np.lexsort(group_data)
        if not np.all(index_arr[1:] > index_arr[:-1]):
            group_data = [data[index_arr] for data in group_data]

        # If we have no data, then bail.
        if len(index_arr) == 0:
            return {}

        # Otherwise, if we don't want the index array, fill in the header keys now.
        if header_key is not None:
            index_arr = header_key[index_arr]
        elif not return_index:
            index_arr = self.get_header_keys(use_mask=use_mask, where=where)[index_arr]

        if assume_unique:
            if len(group_fields) == 1:
                return dict(zip(group_data[0], index_arr))
            else:
                return {tup[:-1]: tup[-1] for tup in zip(*group_data, index_arr)}

        # Otherwise, check element-wise for differences, since that will tell us the
        # boundaries for each "group" of data.
        diff_idx = group_data[0][1:] != group_data[0][:-1]
        for data in group_data[1:]:
            diff_idx |= data[1:] != data[:-1]

        # Need the start position for the first group, and we add 1 to the rest of the
        # index values since the start positions are all offset by 1 thanks to the way
        # that we sliced things above.
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

    def _update_fields(self, update_dict=None, raise_err=False):
        """
        Update fields within a MirMetaData object.

        Note that this is not a function designed to be called by users, but instead is
        a helper function for other methods. This function will take a so-called
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
            if no match is found for a particular entry, the method quietly moves on
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
            iter_data = self.get_value(field, use_mask=False)

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

    def _add_check(
        self, other=None, *, merge=None, overwrite=None, discard_flagged=False
    ):
        """
        Check if two MirMetaData objects contain conflicting header key values.

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
            Index positions denote which indices of metadata would be utilized from
            this object if an __add__ operation were to be performed. Note that the
            header keys for this set of index positions will be completely disjoint
            from that of `other_idx` and `other`.
        other_idx : list of int
            Index positions denote which indices of metadata would be utilized from
            `other` if an __add__ operation were to be performed. Note that the
            header keys for this set of index positions will be completely disjoint
            from that of `this_idx` and this object.
        this_mask : ndarray of bool
            Mask values for the index values in `this_idx`. Note that in the case of
            overlapping keys between this object and `other`, the masks are "or'd"
            together.
        other_mask : ndarray of bool
            Mask values for the index values in `other_idx`. Note that in the case of
            overlapping keys between this object and `other`, the masks are "or'd"
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
        if not isinstance(other, self.__class__):
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

        # Count the sum total number of entries we have
        idx_count = len(index_dict1) + len(index_dict2)

        # Assume that if key_overlap has entries, we are allowed to merge
        if len(key_overlap):
            # Go through the overlapping keys and see if we have any mismatches in mask
            # state. If we do, then we "or" the mask elements together, which always
            # results in a return value of True. Generate these indexing arrays once
            # up front, so that we don't need build them redundantly for this_mask and
            # other_mask.
            idx1 = [index_dict1[key] for key in key_overlap]
            idx2 = [index_dict2[key] for key in key_overlap]
            this_mask[idx1] |= other_mask[idx2]
            other_mask[idx2] = this_mask[idx1]
            if overwrite:
                # If we can overwrite, then nothing else matters -- drop the index
                # positions from this object and move on.
                _ = [index_dict1.pop(key) for key in key_overlap]
            else:
                # Check array index positions for arr1 first, see if all are flagged
                arr1_idx = (
                    ... if same_dict else [index_dict1[key] for key in key_overlap]
                )
                arr2_idx = (
                    ... if same_dict else [index_dict2[key] for key in key_overlap]
                )
                arr1_mask = self._mask[arr1_idx]
                arr2_mask = other._mask[arr2_idx]

                if (overwrite is None) and not np.any(arr1_mask & arr2_mask):
                    # If at each position at least one object is flagged, then drop the
                    # key flagged from that object (dropping it from self if both
                    # objects have that index flagged).
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

    def _gen_filepath(self, filepath=None, *, check=True, invert_check=False):
        """
        Supply the path to the file for read/write operations.

        Parameters
        ----------
        filepath : str or Path
            Either the file to write to, or if providing the name of an existing folder,
            the name of the folder to write in (with the file name set by the _filetype
            attribute, which is automatically set for various subclasses of
            MirMetaData). No default.
        check : bool
            If set to True, checks if the file exists, and if so, raises a warning.
            Default is True.
        invert_check : bool
            Only applicable if `check=True`. If set to True, changes the check so that
            an error is raised if the file exists.

        Returns
        -------
        fullpath : str
            The full path to the file to be read/written.

        Raises
        ------
        ValueError
            If filepath is not an str.
        FileExistsError
            If running the check and a file is found (and `invert_check=True`).
        FileNotFoundError
            If running the check and no file is found (and `invert_check=False`).
        """
        if not isinstance(filepath, (str, Path)):
            raise ValueError("filepath must be of type str or Path.")

        if os.path.isdir(filepath):
            filepath = os.path.join(os.path.abspath(filepath), self._filetype)

        if check:
            if os.path.exists(filepath) and invert_check:
                raise FileExistsError(
                    "File already exists, must set overwrite or append_data to True, "
                    "or delete the file %s in order to proceed." % filepath
                )
            elif not (os.path.exists(filepath) or invert_check):
                raise FileNotFoundError("No file found with the path %s." % filepath)

        return filepath

    def read(self, filepath=None):
        """
        Read in data for a MirMetaData object from disk.

        Parameters
        ----------
        filepath : str or Path
            Path of the folder containing the metadata in question.
        """
        self._data = np.fromfile(
            self._gen_filepath(filepath),
            dtype=self.dtype if self._binary_dtype is None else self._binary_dtype,
        )

        if self._binary_dtype is not None:
            try:
                self._data = self._data.astype(self.dtype)
            except UnicodeDecodeError:
                # If we get a unicode error, that means that we have chars that are not
                # in the 'normal' ascii range. This is a not uncommon occurrence in the
                # codes_read, where char arrays contained uninitialized memory values.
                # We only need to do this check for string-like fields.
                check_fields = [
                    idx
                    for idx in range(len(codes_dtype))
                    if np.issubdtype(codes_dtype[idx], np.str_)
                ]
                for idx in range(len(self._data)):
                    for jdx in check_fields:
                        marker = self._data[idx][jdx].find(b"\x00")
                        if marker > 0:
                            self._data[idx][jdx] = self._data[idx][jdx][:marker]
                self._data = self._data.astype(self.dtype)

        self._mask = np.ones(self._size, dtype=bool)
        self._set_header_key_index_dict()

    def _writefile(self, filepath=None, *, append_data=False, datamask=...):
        """
        Write _data attribute to disk.

        This function is a low-level helper function, which is called when calling the
        `write` method. It is broken out separately here to enable subclasses to
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

    def write(
        self, filepath=None, *, overwrite=False, append_data=False, check_index=False
    ):
        """
        Write a metadata object to disk.

        Parameters
        ----------
        filepath : str
            Path of the folder to write the metadata into.
        overwrite : bool
            If set to True, allow the file writer to overwrite a previously written
            data set. Default is False. This argument is ignored if `append_data` is
            set to True.
        append_data : bool
            If set to True, will append data to an existing file. Default is False.
        check_index : bool
            Only applicable if `append_data=True`. If set to True and data are being
            appended to an existing file, the method will check to make sure that there
            are no header key conflicts with the data being written to disk, since
            this can cause the file to become unusable. Default is False.

        Raises
        ------
        FileExistsError
            If a file already exists and cannot append or overwrite.
        ValueError
            If attempting to append data, but conflicting header keys are detected
            between the data on disk and the data in the object.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        writepath = self._gen_filepath(
            filepath, check=not (overwrite or append_data), invert_check=True
        )

        if os.path.exists(writepath) and append_data and check_index:
            copy_obj = self.copy(skip_data=True)
            copy_obj.read(filepath)
            try:
                idx_arr = self._add_check(
                    copy_obj, discard_flagged=True, overwrite=False
                )[0]
            except MirMetaError as err:
                # If we get this error, it means our (partial) merge has failed.
                # Time to bail.
                raise ValueError(
                    "Conflicting header keys detected with data on disk. Cannot "
                    "append data from this object to specified file."
                ) from err

            if len(idx_arr) == 0:
                # There's literally nothing to do here, so bail.
                return

            # Generate a mask based on the unique data entries.
            datamask = self._generate_mask(index=idx_arr)
        else:
            # If we haven't done so yet, create the data mask now.
            datamask = ... if np.all(self._mask) else self._mask

        self._writefile(writepath, append_data=append_data, datamask=datamask)

    def _get_record_size_info(self, *, val_size=None, pad_size=0, use_mask=True):
        """
        Calculate the size of each spectral record in number of entries.

        Calculates the size of each auto/cross correlation, used for parsing and
        creating packed data arrays for read and write operations.

        Parameters
        ----------
        val_size : int
            Size of each channel in number of bytes. E.g., visibilities are commonly
            stored as complex int16, which would correspond to 4 bytes.
        pad_size : int
            Spectral records sometimes contain "subheader" information that's not part
            of the raw data. The length of this subheader is specified here, in number
            of bytes. Default is 0.
        use_mask : bool
            If True, consider only data where the internal mask is marked True. Default
            is True.

        Returns
        -------
        rec_size_arr : ndarray of int
            Array that contains the length (in bytes) of each spectral record for the
            corresponding entry in the MirMetaData object.

        Raises
        ------
        TypeError
            If the object is not a MirSpData or MirAcData type.
        """
        if not isinstance(self, (MirSpData, MirAcData)):
            raise TypeError(
                "Cannot use this method on objects other than MirSpData"
                "and MirAcData types."
            )

        rec_size_arr = pad_size + (
            val_size * self.get_value("nch", use_mask=use_mask).astype(int)
        )

        return rec_size_arr

    def _recalc_dataoff(
        self, *, data_dtype=None, data_nvals=None, scale_data=None, use_mask=True
    ):
        """
        Calculate the offsets of each spectral record for packed data.

        This is an internal helper function not meant to be called by users, but
        instead is a low-level helper function. This function is used to calculate the
        relative offset of the spectral record inside of a per-integration "packed
        data array", which is what is recorded to disk. This method is primarily used
        when writing visibility to disk, since the packing of the data (and by
        extension, it's indexing) depends heavily on what records have been recorded to
        disk. Note that operation _will_ modify the "dataoff" field inside of the
        metadata, so care should be taken when calling it.

        Parameters
        ----------
        data_dtype : numpy dtype
            "Simple" dtype (i.e.., not a structured array) that describes the individual
            data values, typically np.int16 or np.float32. No default.
        data_nvals : int
            Number of values per channel entry. Typically, for real-only data this is 1,
            and for complex data this is 2. No default.
        scale_data : bool
            Whether or not the data are packed with a common exponent (typically done
            for the visibilities but not the autos). No default.
        use_mask : bool
            If set to True, evaluate/calculate for only those records where the internal
            mask is set to True. If set to False, use all records in the object,
            regardless of mask status. Default is True.
        """
        rec_size_arr = self._get_record_size_info(
            val_size=data_dtype.itemsize * data_nvals,
            pad_size=data_dtype.itemsize if scale_data else 0,
            use_mask=use_mask,
        )

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

    def _generate_recpos_dict(
        self,
        *,
        data_dtype=None,
        data_nvals=None,
        scale_data=None,
        pad_nvals=None,
        hdr_fmt=None,
        use_mask=True,
        reindex=False,
    ):
        """
        Generate a set of dicts for indexing of data.

        This is an internal helper function not meant to be called by users, but
        instead is a low-level helper function. This function is used to calculate
        internal indexing values for use in unpacking the raw data on disk, recorded
        under the filename "sch_read".

        Parameters
        ----------
        data_dtype : numpy dtype
            "Simple" dtype (i.e.., not a structured array) that describes the individual
            data values, typically np.int16 or np.float32. No default.
        data_nvals : int
            Number of values per channel entry. Typically, for real-only data this is 1,
            and for complex data this is 2. No default.
        scale_data : bool
            Whether or not the data are packed with a common exponent (typically done
            for the visibilities but not the autos). No default.
        pad_nvals :  int
            Number of added values in the subheader for each spectral record, separate
            from the common scale factor value. No default.
        hdr_fmt : list
            List describing header fields for each integration record, given as a list
            of 2-element tuples (as appropriate for a structured array, see the docs
            of numpy.dtype for further details). No default.
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
            on disk, which is not _necessarily_ the same as in the object),
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
        val_size = data_dtype.itemsize
        header_size = np.dtype(hdr_fmt).itemsize
        rec_size_arr = self._get_record_size_info(
            val_size=val_size * data_nvals,
            pad_size=val_size if scale_data else 0,
            use_mask=use_mask,
        )
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
                eidx_arr = np.cumsum(rec_size_subarr) + pad_nvals
                sidx_arr = eidx_arr - rec_size_subarr
            else:
                sidx_arr = dataoff_arr[rec_idx] + pad_nvals
                eidx_arr = sidx_arr + rec_size_subarr

            # Plug in the start/end index positions for each spectral record.
            recpos_dict[inhid] = {
                hkey: {"start_idx": sidx, "end_idx": eidx, "chan_avg": 1}
                for hkey, sidx, eidx in zip(hkey_subarr, sidx_arr, eidx_arr)
            }

            # Record size for int_dict is recorded in bytes, hence the * chan_size here
            record_size = int(eidx_arr.max() * val_size)

            # There's no way these should ever happen unless the metadata are bad,
            # just check and make sure this isn't the case.
            assert record_size > 0, "record_size not gtr than zero, metadata corrupted"
            assert record_start >= 0, "record_start less than zero, metadata corrupted"

            int_dict[inhid] = {
                "inhid": inhid,
                "record_size": record_size,
                "record_start": record_start,
            }
            # Now that we're at the end of the record, add the header size.
            record_start += record_size + header_size
        return int_dict, recpos_dict

    def _make_key_mask(
        self,
        other=None,
        *,
        reverse=False,
        use_mask=True,
        check_field=None,
        set_mask=True,
        use_cipher=True,
    ):
        """
        Generate a key mask by field-matching between MirMetaData objects.

        This creates a (meta)data mask based on when the header key of one object is
        found in another. Useful for propagating flag information between different
        types of objects (e.g., between per-integration and per-baseline records).

        Parameters
        ----------
        other : MirMetaData object
            MirMetaData object to use to compare against this object. Must either
            contain the header key values as fields, or have the field(s) specified by
            the `check_field` keyword.
        reverse : bool
            When set to False, the default, the normal pattern of behavior is to check
            and see if the header key values in this object are found in the
            corresponding field in the `other` object. If set to True, then the header
            keys from the `other` object are sought in the corresponding fields in
            this object.
        use_mask : bool
            If set to False, ignore the underlying data mask (as set by e.g., previous
            select commands). Default is True.
        check_field : str or list of str
            Field in `other` (or this object, if `reverse=True`) to check against header
            key values. Default is to use the same name(s) as the header key field(s).
        set_mask : bool
            If set to True, the default, then after the comparison is complete, the
            method will set the underlying data mask based on matches found (keeping
            only records where matching keys are found).
        use_cipher : bool
            Normally if header key values encompass two fields, pairs of values are
            compared, which can be slow. If set to True, the method will try to combine
            the information of the two fields via bit-shifting and addition, which can
            significantly speed up the operation of the matching. Default is True if
            not supplying an str-type for `check_field`.

        Returns
        -------
        return_val : bool or ndarray of bool
            If `set_mask=True`, then a bool is returned which specifies whether or not
            any underlying mask values were changed. If `set_mask=False`, then the mask
            is returned based on key-matching results.
        """
        # Figure out what check_field should be if undefined
        if check_field is None:
            check_field = (other if reverse else self)._identifier

        # Default to no cipher if there's only one field.
        use_cipher = use_cipher and not isinstance(check_field, str)

        # Get header keys for this if normal, otherwise get corresponding field
        this_data = self.get_value(
            check_field if reverse else self._identifier,
            use_mask=use_mask,
            return_tuples=(not (use_cipher or isinstance(check_field, str))),
        )

        # Get header keys for other if reverse, otherwise get corresponding field
        other_data = other.get_value(
            other._identifier if reverse else check_field,
            use_mask=use_mask,
            return_tuples=(not (use_cipher or isinstance(check_field, str))),
        )

        # If using a cipher, we want to combine two fields together via bitshift
        # and addition. Note that this is really only used for eng_data, since it's
        # the only regularly indexed MirMetaData object that needs this.
        if use_cipher:
            assert len(check_field) == 2, "Cannot use cipher with more than two fields."

            # Make sure that we have unsigned ints
            for idx in range(len(check_field)):
                if "u" not in this_data[idx].dtype.name:
                    this_data[idx] = this_data[idx].view(
                        "u" + this_data[idx].dtype.name
                    )
                if "u" not in other_data[idx].dtype.name:
                    other_data[idx] = other_data[idx].view(
                        "u" + other_data[idx].dtype.name
                    )
            # Cast the first field to int64, bitshift up by 32 bits, then add with
            # the other data field to produce a unique (compact) key to search on.
            this_data = (this_data[0].astype("u8") << 32) + this_data[1]
            other_data = (other_data[0].astype("u8") << 32) + other_data[1]

        # We have two different methods to use here, the former being faster since
        # numpy's isin is pretty fast. Note that use of unique here is to cut down
        # on redundant keys (which helps to speed this whole thing up) -- it's generally
        # only useful for reverse searches, since otherwise other_data contains all
        # unique header keys.
        if isinstance(check_field, str) or use_cipher:
            other_data = other_data if reverse else np.unique(other_data)
            mask = np.isin(this_data, other_data, assume_unique=not reverse)
        else:
            other_data = other_data if reverse else set(other_data)
            mask = np.array([item in other_data for item in this_data])

        return self.set_mask(mask=mask, use_mask=use_mask) if set_mask else mask


class MirInData(MirMetaData):
    """
    Class for per-integration metadata in Mir datasets.

    This class is a container for per-integration metadata, using the header key
    "inhid". When reading from/writing to disk, the object looks for a file named
    "in_read", which is where the online system records this information.
    """

    def __init__(self, obj=None):
        """
        Initialize a MirInData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        """
        super().__init__(
            obj=obj, filetype="in_read", dtype=in_dtype, header_key_name="inhid"
        )


class MirBlData(MirMetaData):
    """
    Class for per-baseline metadata in Mir datasets.

    This class is a container for per-baseline metadata, using the header key
    "blhid". When reading from/writing to disk, the object looks for a file named
    "bl_read", which is where the online system records this information. Note that
    "per-baseline" here means per-integration, per-sideband, per-receiver/polarization.
    """

    def __init__(self, obj=None):
        """
        Initialize a MirBlData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        """
        super().__init__(
            obj=obj, filetype="bl_read", dtype=bl_dtype, header_key_name="blhid"
        )


class MirSpData(MirMetaData):
    """
    Class for per-spectral window metadata in Mir datasets.

    This class is a container for per-spectral window metadata, using the header key
    "sphid". When reading from/writing to disk, the object looks for a file named
    "sp_read", which is where the online system records this information. Note that
    "per-spectral window" here means per-integration, per-baseline, per-spectral
    band number.
    """

    def __init__(self, obj=None):
        """
        Initialize a MirSpData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        """
        super().__init__(
            obj=obj, filetype="sp_read", dtype=sp_dtype, header_key_name="sphid"
        )


class MirWeData(MirMetaData):
    """
    Class for per-integration weather metadata in Mir datasets.

    This class is a container for per-integration weather metadata, using the header key
    "ints". When reading from/writing to disk, the object looks for a file named
    "we_read", which is where the online system records this information.
    """

    def __init__(self, obj=None):
        """
        Initialize a MirWeData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        """
        super().__init__(
            obj=obj, filetype="we_read", dtype=we_dtype, header_key_name="ints"
        )


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

    def __init__(self, obj=None):
        """
        Initialize a MirEngData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        """
        super().__init__(
            obj=obj,
            filetype="eng_read",
            dtype=eng_dtype,
            pseudo_header_key_names=("antenna", "inhid"),
        )


class MirAntposData(MirMetaData):
    """
    Class for antenna position information in Mir datasets.

    This class is a container for antenna positions, which are recorded as a text file
    within a Mir dataset named "antennas". It has a header key of "antenna", which is
    paired to the antenna number in other metadata objects (e.g., "antenna",
    "iant1", "iant2").
    """

    def __init__(self, obj=None):
        """
        Initialize a MirAntposData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        """
        super().__init__(
            obj=obj, filetype="antennas", dtype=antpos_dtype, header_key_name="antenna"
        )

    def read(self, filepath=None):
        """
        Read in data for a MirAntposData object from disk.

        Parameters
        ----------
        filepath : str or Path
            Path of the folder containing the metadata in question.
        """
        with open(self._gen_filepath(filepath), "r") as antennas_file:
            temp_list = [
                item for line in antennas_file.readlines() for item in line.split()
            ]
        self._data = np.empty(len(temp_list) // 4, dtype=antpos_dtype)
        self._data["antenna"] = np.int16(temp_list[0::4])
        self._data["xyz_pos"] = np.array(
            [temp_list[1::4], temp_list[2::4], temp_list[3::4]], dtype=np.float64
        ).T

        self._mask = np.ones(self._size, dtype=bool)
        self._set_header_key_index_dict()

    def _writefile(self, filepath=None, *, append_data=False, datamask=...):
        """
        Write _data attribute to disk.

        This method is an internal function which is called when calling the
        `write` method. It is broken out separately here to enable subclasses to
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

    def __init__(self, obj=None):
        """
        Initialize a MirCodesData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        """
        # Define some things up front. These are codes that should have multiple entries
        # that are different every time, usually changing once an integration. They
        # map to various "iXXX" indexing fields contained in other metadata tables.
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
        self._immutable_codes = ["filever", "pol"]

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
            "rec": "irec",
            "pstate": "ipstate",
            "sb": "isb",
            "band": "iband",
            "ddsmode": "iddsmode",
        }

        super().__init__(
            obj=obj,
            filetype="codes_read",
            dtype=codes_dtype,
            binary_dtype=codes_binary_dtype,
            pseudo_header_key_names=("icode", "v_name"),
        )

    def __getitem__(self, field_name):
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
        if field_name in self.dtype.fields:
            return super().__getitem__(field_name)
        else:
            return self.get_codes(field_name)

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
        select_field=None,
        select_comp=None,
        select_val=None,
        *,
        mask=None,
        return_header_keys=None,
    ):
        """
        Find where metadata match a given set of selection criteria.

        This method will produce a masking screen based on the arguments provided to
        determine which entries match a given set of conditions.

        Parameters
        ----------
        select_field : str
            Field or code type ("v_name") in the metadata to evaluate.
        select_comp : str
            Specifies the type of comparison to do between the value supplied in
            `select_val` and the metadata. No default, allowed values include:
            "eq" or "==" (equal to);
            "ne" or "!=" (not equal to);
            "lt" or "<" (less than);
            "le" or "<=" (less than or equal to);
            "gt" or ">" (greater than);
            "ge" or ">=" (greater than or equal to);
            "between" (between a range of values);
            "outside" (outside of a range of values).
        select_val : number of str, or sequence of number or str
            Value(s) to compare data in `select_field` against. If `select_comp` is
            "lt", "le", "gt", "ge", then this must be a single number. If `select_comp`
            is "between" or "outside", then this must be a list 2 numbers. If
            `select_comp` is "eq"/"==" or "ne"/"!=", then this can be either a single
            value (number or string) or a sequence of numbers.
        mask : ndarray of bool
            Optional argument, of the same length as the MirMetaData object, which is
            applied to the output of the selection parsing through an element-wise
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
                select_field=select_field,
                select_comp=select_comp,
                select_val=select_val,
                mask=mask,
                return_header_keys=return_header_keys,
            )

        if select_field not in self._codes_index_dict:
            raise MirMetaError(
                "select_field must either be one of the native fields inside of the "
                'codes_read array ("v_name", "code", "icode", "ncode") or one of the '
                "indexing codes (%s)." % ", ".join(list(self._codes_index_dict))
            )

        if select_comp not in ["eq", "==", "ne", "!="]:
            raise ValueError(
                'select_comp must be "eq", "==", "ne", or "!=" when '
                "select_field is a code type."
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
            super().where(
                "v_name", "eq", select_field, mask=mask, return_header_keys=False
            ),
            super().where(
                "code", select_comp, select_val, mask=mask, return_header_keys=False
            ),
        )

        if return_header_keys or (return_header_keys is None):
            return list(self.get_value("icode")[data_mask])
        else:
            return data_mask

    def get_codes(self, code_name=None, return_dict=None):
        """
        Get code strings for a given variable name in the metadata.

        Look up the code strings for a given variable name (`v_name`), which typically
        contain information about the data set as a whole, or information for mapping
        indexing data from other MirMetaData objects to more easily understood strings
        of text.

        Parameters
        ----------
        code_name : str
            Name of the codes, a full listing of which can be provided by the method
            `get_code_name`.
        return_dict : bool
            If set to True, return a dict with keys and values that map code strings
            to indexing values, and visa-versa. Useful for mapping values between
            other MirCodesData and other MirMetaData object types. Default is None,
            which will return a dict only if `code_name` has more than one entry or
            has a known counterpart field in one of the other MirMetaData object types
            (e.g., "source" maps to "isource" in MirInData).

        Returns
        -------
        codes : list or dict
            If `return_dict=False`, then a list for all code strings is returned.
            Otherwise, a dict is returned which maps both indexing codes to code strings
            and visa-versa.
        """
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
            return dict(zip(codes + index, index + codes))
        else:
            return codes

    def _generate_new_header_keys(self, other):
        """
        Create an updated set of pseudo header keys for a MirCodesData object.

        Note that this function is not meant to be called by users, but instead is
        is a low-level helper function for the object. This function allows for one to
        create an updated set of pseudo header keys via an update to the indexing codes,
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
        if not isinstance(other, self.__class__):
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

                # Start the process of re-indexing the "icode" values
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

    def __init__(self, obj=None, nchunks=None):
        """
        Initialize a MirAcData object.

        Parameters
        ----------
        obj : str or ndarray or int
            Optional argument used to specify how to initialize the object. If a str is
            supplied, then it is treated as the path to the Mir data folder containing
            the metadata. If an int is supplied, a "blank" (zero-filled) array of
            metadata is generated, with length of `obj`. If an ndarray is supplied, then
            the supplied array is used as the underlying data set for the object (where
            dtype of the array must match that appropriate for the object).
        nchunks : int
            Number of chunks to assume are recorded in the auto-correlation data. Note
            that this parameter is only used with the "old-style" files (i.e., where
            "ac_read" and "ach_read" are not present in the Mir file folder).
        """
        self._old_format = False
        self._old_format_int_dict = None
        self._nchunks = nchunks
        super().__init__(
            obj=obj, filetype="ac_read", dtype=ac_dtype, header_key_name="achid"
        )

    def read(self, filepath=None):
        """
        Read in data for a MirAcData object from disk.

        Parameters
        ----------
        filepath : str or Path
            Path of the folder containing the metadata in question.
        """
        try:
            # Start from the assumption that the 'new' style file is used, which will
            # error with a FileNotFoundError if the new file doesn't exist.
            self._old_format = False
            return super().read(filepath)
        except FileNotFoundError:
            # If FileNotFoundError is thrown, try loading the "old" stype file.
            # Note that this step below will also fail if the file does not exists.
            filepath = self._gen_filepath(os.path.join(filepath, "autoCorrelations"))

        # Assume that we have the old file format at this point, and proceed accordingly
        self._old_format = True
        file_size = os.path.getsize(filepath)
        hdr_dtype = np.dtype(OLD_AUTO_HEADER)
        nrx = 2
        nchan = 16384

        # If not set, try to capture he number of chunks from the first header
        # from inside the file, and see if that parses the records evenly.
        nchunks = self._nchunks
        if nchunks is None:
            nchunks = np.fromfile(filepath, dtype=hdr_dtype, count=1)["nchunks"][0]
        nchunks = int(nchunks)

        # Tabulate the total number of spectra
        nspec = nrx * nchunks

        # Determine the size of the packed data and of the full record (data + headers)
        pack_size = OLD_AUTO_DTYPE.itemsize * nspec * nchan
        rec_size = pack_size + hdr_dtype.itemsize

        # This bit of code is to warn of an unfortunately common problem with metadata
        # of MIR autos not being correctly recorded.
        if (file_size % rec_size) != 0:
            # If the file size doesn't go in evenly, raise a warning
            warnings.warn(
                "Auto-correlation records appear to be the incorrect size, be aware "
                "that the file may be corrupted or nchunks may be incorrectly set."
            )

        # Pre-allocate the metadata array,
        nrec = file_size // rec_size
        ac_data = np.zeros(nrx * nchunks * nrec, dtype=ac_dtype)

        # Set values that we know a priori
        ac_data["nch"] = nchan
        ac_data["isb"] = 1
        ac_data["correlator"] = 1

        # Grab some references (views) to the values we need to plug in to. This saves
        # us a little work later of needing to rearrange this data into a single
        # complex dtype which can be handled in one ndarray.
        dataoff_arr = ac_data["dataoff"]
        antenna_arr = ac_data["antenna"]
        chunk_arr = ac_data["corrchunk"]
        antrx_arr = ac_data["antrx"]
        inhid_arr = ac_data["inhid"]
        dhrs_arr = ac_data["dhrs"]
        int_dict = {}
        last_inhid = None
        last_pos = 0
        rec_count = 0
        dataoff_spec = np.arange(nspec) * OLD_AUTO_DTYPE.itemsize * nchan

        with open(filepath, "rb") as auto_file:
            for idx in range(nrec):
                auto_vals = np.fromfile(
                    auto_file,
                    dtype=hdr_dtype,
                    count=1,
                    offset=pack_size if idx else 0,  # Skip offset on first iteration
                )[0]
                if auto_vals["inhid"] != last_inhid:
                    # If this is a new inhid, start record keeping for it
                    last_pos += rec_count * rec_size
                    rec_count = 0
                    last_inhid = auto_vals["inhid"]

                    # Create a sub-dict for this particular record
                    rec_dict = {
                        "inhid": auto_vals["inhid"],
                        "record_size": 0,
                        "record_start": last_pos,
                    }
                    # Plug the subdict into the main dict
                    int_dict[last_inhid] = rec_dict

                # Setup some slices that we'll use for plugging in values
                rxa_slice = slice(idx * nspec, (idx + 1) * nspec, 2)
                rxb_slice = slice(1 + (idx * nspec), (idx + 1) * nspec, 2)
                ac_slice = slice(rxa_slice.start, rxb_slice.stop)

                # Plug in the entries that are changing on a per-record basis
                dhrs_arr[ac_slice] = auto_vals["dhrs"]
                antenna_arr[ac_slice] = auto_vals["antenna"]
                chunk_arr[rxa_slice] = chunk_arr[rxb_slice] = np.arange(1, nchunks + 1)
                antrx_arr[rxa_slice] = 0
                antrx_arr[rxb_slice] = 1
                inhid_arr[ac_slice] = last_inhid

                # Now plug in the dataoff values
                dataoff_arr[ac_slice] = dataoff_spec + (rec_count * rec_size)
                # Tally one more record
                rec_count += 1
                # Update the record size
                rec_dict["record_size"] = (rec_count * rec_size) - hdr_dtype.itemsize

        # Copy the corrchunk values to iband, since they should be the same here.
        ac_data["iband"] = ac_data["corrchunk"]
        # Plug in the data into the object
        self._data = ac_data
        # Generate a blank (full) mask
        self._mask = np.ones(self._size, dtype=bool)
        # Create header keys
        self._set_header_key_index_dict()
        # Store the int_dict into a special attribute for MirAcRead
        self._old_format_int_dict = int_dict
