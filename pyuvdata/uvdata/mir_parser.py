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
        ("antennaNumber", np.int32),
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
        ("spareint2", np.int32),
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
    [("v_name", "S12"), ("icode", np.int16), ("code", "S26"), ("ncode", np.int16)]
).newbyteorder("little")

# we_read records various weather data collected at the antennas, which is typically
# used for refraction correction by online fringe tracking.
we_dtype = np.dtype(
    [
        # Scan number (should be equal to inhid)
        ("scanNumber", np.int32),
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

# ac_read is _not_ something that is actually read in, but is instead a "helper"
# data structure for recording some of the metadata associated with the auto
# correlations. Because of this, the dtype below may change.
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
).newbyteorder("little")

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

    def __iter__(self, metadata_only=False):
        """
        Iterate over all MirParser attributes.

        Parameters
        ----------
        metadata_only : bool
            If true, data-related attributes ("vis_data","raw_data", and "auto_data")
            will be excluded from the yielded iterable.

        Yields
        ------
        attr : MirParser attribute
            Attribute of the MirParser object.
        """
        attribute_list = [
            a
            for a in dir(self)
            if not a.startswith("__")
            and not callable(getattr(self, a))
            and not (metadata_only and a in ["vis_data", "raw_data", "auto_data"])
        ]

        for attribute in attribute_list:
            yield attribute

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
            raise ValueError(
                "Cannot compare MirParser with non-MirParser class objects."
            )

        # I say these objects are the same -- prove me wrong!
        is_eq = True

        # First up, check the list of attributes between the two objects
        this_attr = set(self.__iter__(metadata_only=metadata_only))
        other_attr = set(other.__iter__(metadata_only=metadata_only))

        # Go through and drop any attributes that both objects do not have (and set
        # is_eq to False if any such attributes found).
        for item in this_attr.union(other_attr):
            target = None
            if item not in this_attr:
                other_attr.remove(item)
                target = "right"
            elif item not in other_attr:
                this_attr.remove(item)
                target = "left"
            if target is not None:
                is_eq = False
                if verbose:
                    print("%s does not exist in %s." % (item, target))

        # At this point we _only_ have attributes present in both lists
        for item in this_attr:
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

            # If both are NoneType, we actually have nothing to do here
            if this_attr is None:
                continue

            item_diff = False
            # Now go through and compare the attribute values
            if item in ["auto_data", "raw_data", "vis_data"]:
                # Data-related attributes are a bit special, in that they are dicts
                # of dicts (note this may change at some point).
                if this_attr.keys() != other_attr.keys():
                    is_eq = False
                    if verbose:
                        print(
                            f"{item} has different keys, left is {this_attr.keys()}, "
                            f"right is %{other_attr.keys()}."
                        )
                    continue
                # For the attributes with multiple fields to check, list them
                # here for convenience.
                comp_dict = {
                    "raw_data": ["raw_data", "scale_fac"],
                    "vis_data": ["vis_data", "vis_flags"],
                }
                for key in this_attr.keys():
                    # auto_data entries are just ndarrays, which we can compare directly
                    if item == "auto_data":
                        if not np.array_equal(this_attr[key], other_attr[key]):
                            item_diff = True
                    else:
                        # If cross-correlation data, then there are multiple dict
                        # entries that we need to compare (defined above).
                        for subkey in comp_dict[item]:
                            if not np.array_equal(
                                this_attr[key][subkey], other_attr[key][subkey]
                            ):
                                item_diff = True
                    if item_diff:
                        is_eq = False
                        if verbose:
                            print(f"{item} has the same keys, but different values")
                        break
                # Nothing to do further here with dicts
                continue

            if isinstance(getattr(self, item), np.ndarray):
                # Need to handle ndarrays a bit special here
                if not np.array_equal(getattr(self, item), getattr(other, item)):
                    item_diff = True
            elif getattr(self, item) != getattr(other, item):
                item_diff = True

            if item_diff:
                is_eq = False
                if verbose:
                    print(
                        f"{item} has different values, left is {getattr(self, item)}, "
                        f"right is %{getattr(other, item)}."
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
        mp = MirParser()

        # include all attributes, not just UVParameter ones.
        for attr in self.__iter__(metadata_only=metadata_only):
            # skip properties
            if not isinstance(getattr(type(self), attr, None), property):
                setattr(mp, attr, copy.deepcopy(getattr(self, attr)))

        return mp

    @staticmethod
    def segment_by_index(read_arr, index_field):
        """
        Break an array into multiple subarrays based on an index value.

        This function will break an array into subarrays, based on the indexing field
        provided. Useful in functions that require grouping of data. Note that the
        ordering of subarray will match that of read_arr (in terms of first appearance
        of each value of the index field), but ordering within subarrays may be
        different.

        Parameters
        ----------
        read_arr : ndarray
            Structured ndarray to be broken into segments.
        index_field : str
            Field within `read_arr` to group data by.

        Returns
        -------
        subarr_dict : dict
            Dict with keys of unique entries of `index_field`, and values which are
            ndarrays containing all elements of `read_arr` where the index field is
            equal to the key.
        pos_dict : dict
            Similar to `subarr_dict`, unique entries of `index_field` make up the keys
            of this dict, with values corresponding to and ndarray of dtype=int which
            reports the position within `read_arr` where each subarray value comes from.

        Raises
        ------
        TypeError
            If `read_arr` is not an ndarray, or if `index_field` is not a string.
        ValueError
            If `index_field` is not a field within `read_arr`.
        """
        if not isinstance(read_arr, np.ndarray):
            raise TypeError("read_arr must be of type ndarray.")
        if not isinstance(index_field, str):
            raise TypeError("read_arr must be of type ndarray.")
        if index_field not in read_arr.dtype.names:
            raise ValueError(
                "index_field %s is not a recognized field in read_arr" % index_field
            )

        subarr_dict = {}
        pos_dict = {}
        for pos, (indv_rec, idx) in enumerate(zip(read_arr, read_arr[index_field])):
            try:
                subarr_dict[idx].append(indv_rec)
                pos_dict[idx].append(pos)
            except KeyError:
                subarr_dict[idx] = [indv_rec]
                pos_dict[idx] = [pos]

        # We want structured arrays rather than a list of "singleton" structures, so
        # use concat here to restore our broken-up entries into arrays again.
        subarr_dict = {
            key: np.concatenate([value]) for key, value in subarr_dict.items()
        }

        # Same thing with position arrays
        pos_dict = {key: np.concatenate([value]) for key, value in pos_dict.items()}

        return subarr_dict, pos_dict

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
    def write_in_data(filepath, in_data, append_data=False):
        """
        Write "in_read" mir file to disk (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        in_data : ndarray of type in_dtype
            Array of entries in the format matching what is returned by read_in_data.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with open(
            os.path.join(filepath, "in_read"), "ab" if append_data else "wb+"
        ) as file:
            in_data.tofile(file)

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
    def write_eng_data(filepath, eng_data, append_data=False):
        """
        Write "in_read" mir file to disk (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        eng_data : ndarray of type eng_dtype
            Array of entries in the format matching what is returned by read_eng_data.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with open(
            os.path.join(filepath, "eng_read"), "ab" if append_data else "wb+"
        ) as file:
            eng_data.tofile(file)

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
    def write_bl_data(filepath, bl_data, append_data=False):
        """
        Write "bl_read" mir file to disk (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        bl_data : ndarray of type bl_dtype
            Array of entries in the format matching what is returned by read_bl_data.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with open(
            os.path.join(filepath, "bl_read"), "ab" if append_data else "wb+"
        ) as file:
            bl_data.tofile(file)

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
    def write_sp_data(filepath, sp_data, append_data=False, recalc_dataoff=True):
        """
        Write "sp_read" mir file to disk (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        sp_data : ndarray of type bl_dtype
            Array of entries in the format matching what is returned by read_sp_data.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        if recalc_dataoff:
            # Make a copy of sp_data so as to not affect the original data
            sp_data = sp_data.copy()
            offset_dict = {}
            inhid_arr = sp_data["inhid"]

            # Each channel is 4 bytes in length (int16 real + int16 imag), plus
            # each spectral record has an int16 up front as the common exponent
            record_size_arr = (4 * sp_data["nch"].astype(int)) + 2

            # Note that this will pass a reference to the array in the structured
            # array sp_data, so updating offset_arr will also update sp_data.
            offset_arr = sp_data["dataoff"]
            for idx, (inhid, record_size) in enumerate(zip(inhid_arr, record_size_arr)):
                try:
                    temp_val = offset_dict[inhid]
                    offset_arr[idx] = temp_val
                    offset_dict[inhid] = record_size + temp_val
                except KeyError:
                    offset_arr[idx] = 0
                    offset_dict[inhid] = record_size

        with open(
            os.path.join(filepath, "sp_read"), "ab" if append_data else "wb+"
        ) as file:
            sp_data.tofile(file)

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
    def write_codes_data(filepath, codes_data, append_data=False):
        """
        Write "codes_read" mir file to disk (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        codes_data : ndarray of type codes_dtype
            Array of entries in the format matching what is returned by read_codes_data.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with open(
            os.path.join(filepath, "codes_read"), "ab" if append_data else "wb+"
        ) as file:
            codes_data.tofile(file)

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
    def write_we_data(filepath, we_data, append_data=False):
        """
        Write "we_read" mir file to disk (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        we_data : ndarray of type we_dtype
            Array of entries in the format matching what is returned by read_we_data.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with open(
            os.path.join(filepath, "we_read"), "ab" if append_data else "wb+"
        ) as file:
            we_data.tofile(file)

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
    def write_antennas(filepath, antpos_data):
        """
        Write "codes_read" mir file to disk (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        codes_data : ndarray of type codes_dtype
            Array of entries in the format matching what is returned by read_codes_data.
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with open(os.path.join(filepath, "antennas"), "w+") as file:
            for antpos in antpos_data:
                file.write(
                    "%i %.17e %.17e %.17e\n"
                    % (
                        antpos["antenna"],
                        antpos["xyz_pos"][0],
                        antpos["xyz_pos"][1],
                        antpos["xyz_pos"][2],
                    )
                )

    @staticmethod
    def calc_int_start(sp_data):
        """
        Calculate the integration size records based on spectral record metadata.

        This is the "faster" equivalent to scan_int_start, which simply reads through
        spectral records to estimate where in the file each integration record starts.
        It can be significantly faster (by factors of 1000) than scan_int_start,
        although in some corner cases, if the data are not ordered by inhid in sch_read,
        can produce inaccurate results. This should never happen with data recorded
        from the telescope.

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        sp_data : ndarray
            Metadata from the individual spectral records of the MIR data set, of the
            custom dtype `sp_dtype`.

        Returns
        -------
        int_start_dict : dict
            Dictionary containing the indexes from sch_read, where keys match to the
            inhid indexes, and the values contain a two-element tuple, with the length
            of the packdata array (in bytes) the relative offset (also in bytes) of
            the record within the sch_read file.
        """
        # Grab the individual inhid values out of the spectral records.
        inhid_arr = sp_data["inhid"]

        # We need to calculate the size (in bytes) of each spectral record here.
        # Each channel contains two int16 values (real + imag), with each record
        # containing one additional int16 value (the common scale factor/exponent),
        # and each int16 value takes up 2 bytes of space. Note cast to int here is
        # to avoid overflow issues (since nch is int16).
        size_arr = ((2 * sp_data["nch"].astype(int)) + 1) * 2

        # If the data are NOT in inhid order, then we need to actually get the data
        # sorted now. Note that this check takes about 0.1% of the sort time, so better
        # to do this check here rather than forcing a sort regardless of order.
        if np.any(inhid_arr[:-1] > inhid_arr[1:]):
            sort_idx = np.argsort(inhid_arr)
            inhid_arr = inhid_arr[sort_idx]
            size_arr = size_arr[sort_idx]

        # Calculate where the inhid changes, since it marks the "boundaries" of the
        # inhid record. Note that this records where the block ends, but the +1 will
        # shift this to where the block _begins_ (and also is more compatible with
        # how slices operate). We concat the first and last index positions to complete
        # the list of all start/stop positions of each block.
        arr_bounds = np.concatenate(
            ([0], 1 + np.where(inhid_arr[:-1] != inhid_arr[1:])[0], [len(inhid_arr)])
        )

        # Use the above to just grab the unique inhid values
        inhid_arr = inhid_arr[arr_bounds[:-1]]

        # And we can use cumsum to quickly calculate how big each record is. Note that
        # this ends up being _much_faster than summing on a per-inhid basis, so long
        # as there is >1 unique inhid value (most datasets have O(10^3)).
        size_arr = np.cumsum(size_arr)[arr_bounds[1:] - 1]

        # We need to get a per-record size, so subtract off the "starting point" of
        # the prior integration record.
        size_arr[1:] -= size_arr[:-1]

        # Create a dict to plug values into
        int_start_dict = {}
        # Record where in the file we expect to be (in bytes from start)
        marker = 0
        for inhid, record_size in zip(inhid_arr, size_arr):
            # Plug in the entry into the dict
            int_start_dict[inhid] = (inhid, record_size, marker)

            # Shift the marker by the record size. Note the extra 8 bytes come from
            # the fact that sch_read has the inhid and number of bytes recorded as
            # int32 values, which are each 4 bytes in size.
            marker += record_size + 8

        return int_start_dict

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
        full_filepath = os.path.join(filepath, "sch_read")

        file_size = os.path.getsize(full_filepath)
        data_offset = 0
        last_offset = 0
        int_start_dict = {}
        with open(full_filepath, "rb") as visibilities_file:
            while data_offset < file_size:
                int_vals = np.fromfile(
                    visibilities_file,
                    dtype=np.dtype([("inhid", np.int32), ("nbyt", np.int32)]),
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
                int_start_dict[int_vals["inhid"]] = (
                    int_vals["inhid"],
                    int_vals["nbyt"],
                    data_offset,
                )
                last_offset = int_vals["nbyt"].astype(int)
                data_offset += last_offset + 8

        return int_start_dict

    def fix_int_start(self, filepath=None, int_start_dict=None):
        """
        Fix an integration postion dictionary.

        This method will fix potential errors in an internal dictionary used to mark
        where in the main visibility file an individual spectral record is located.
        Under normal conditions, this routine does not need to be run, unless another
        method reported a specific error on read calling for the user to run this code.

        Parameter
        ---------
        filepath : list of str
            Optional argument, specifying the path of the individual files. Default
            is to pull this information from the MirParser object itself.
        int_start_dict : list of dict
            Optional argument, specifying the starting positions in the file for each
            record. Default is to pull this information from the MirParser object
            itself.

        Returns
        -------
        file_dict : dict
            Only returned if filepath and int_start_dict are provided, a dictionary
            whose keys are the individual values of filepath, matched with the
            updated integration position dictionary (i.e., int_start_dict).

        Raises
        ------
        ValueError
            If either (but not both) filepath and int_start_dict are set, or if the
            length of the list in filepath does not match that of int_start_dict.
        """
        # Make sure both arguments are set
        if (filepath is None) != (int_start_dict is None):
            raise ValueError(
                "Must either set both or neither of filepath and "
                "int_start_dict arguments."
            )

        # Determine whether or not to return the file_dict
        return_dict = filepath is not None

        # If arguments aren't provided, then grab the relevant data from the object
        if filepath is None:
            filepath = list(self._file_dict.keys())

            # Note that we want the deep copy here to avoid potentially corrupting
            # the object-based values _before_ we finish readign in the new values
            # (which could raise an error on read).
            int_start_dict = copy.deepcopy(list(self._file_dict.values()))
        else:
            if len(filepath) != len(int_start_dict):
                raise ValueError(
                    "Both filepath and int_start_dict must be lists of the same length."
                )

        for ifile, idict in zip(filepath, int_start_dict):
            # Each file's inhid is allowed to be different than the objects inhid --
            # this is used in cases when combining multiple files together (via
            # concat). Here, we make a mapping of "file-based" inhid values to that
            # stored in the object.
            idict_map = {inhid: finhid for inhid, (finhid, _, _) in idict.items()}

            # Make the new dict by scaning the sch_read file.
            new_dict = self.scan_int_start(
                filepath=ifile, allowed_inhid=list(idict_map.keys())
            )

            # Go through the individual entries in each dict, and update them
            # with the "correct" values as determined by scanning through sch_read
            for key in idict.keys():
                idict[key] = new_dict[idict_map[key]]

        # Finally, create the internal file dict by zipping together filepaths and
        # the integration position dicts.
        file_dict = {
            os.path.abspath(ifile): idict
            for ifile, idict in zip(filepath, int_start_dict)
        }

        # If we are supposed to return a dict, do that now, otherwise update
        # the attribute in the object.
        if return_dict:
            return file_dict
        else:
            self._file_dict = file_dict

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

        if not os.path.exists(full_filepath):
            raise FileNotFoundError(
                f"Cannot find file {full_filepath}."  # pragma: no cover
                " Set `has_auto=False` to avoid reading autocorrelations."
            )

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
                    if (file_size % (4 * (2**14) * nchunks * 2 + 20)) != 0:
                        nchunks = int(auto_vals["nChunks"][0])
                        if (file_size % (4 * (2**14) * nchunks * 2 + 20)) != 0:
                            raise IndexError(
                                "Could not determine auto-correlation record size!"
                            )
                    # How big the record is for each data set
                    last_offset = 4 * (2**14) * int(nchunks) * 2
                    ac_data = np.zeros(
                        file_size // ((4 * (2**14) * int(nchunks) * 2 + 20)),
                        dtype=ac_read_dtype,
                    )
                ac_data[marker] = (
                    auto_vals["scan"][0],
                    marker + 1,
                    auto_vals["antenna"][0],
                    nchunks,
                    last_offset,
                    data_offset,
                    auto_vals["dhrs"][0],
                )
                data_offset += last_offset + 20
                marker += 1
        return ac_data

    @staticmethod
    def read_packdata(filepath, int_start_dict, use_mmap=False):
        """
        Read "sch_read" mir file into memory (@staticmethod).

        Parameters
        ----------
        filepath : str
            filepath is the path to the folder containing the mir data set.
        int_start_dict : dict
            indexes to the visibility locations within the file.
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
        full_filepath = os.path.join(filepath, "sch_read")
        int_data_dict = {}
        int_dtype_dict = {}

        # We want to create a unique dtype for records of different sizes. This will
        # make it easier/faster to read in a sequence of integrations of the same size.
        size_list = np.unique(
            [int_start_dict[ind_key][1] for ind_key in int_start_dict.keys()]
        )

        for int_size in size_list:
            int_dtype_dict[int_size] = np.dtype(
                [
                    ("inhid", np.int32),
                    ("nbyt", np.int32),
                    ("packdata", np.int16, int_size // 2),
                ]
            ).newbyteorder("little")

        # Initialize a few values before we start running through the data.
        inhid_list = []
        last_offset = last_size = num_vals = del_offset = 0
        key_list = sorted(int_start_dict.keys())

        # We add an extra key here, None, which cannot match any of the values in
        # int_start_dict (since inhid is type int). This basically tricks the loop
        # below into spitting out the last integration
        key_list.append(None)

        # Read list is basically storing all of the individual reads that we need to
        # execute in order to grab all of the data that we need. Note that each entry
        # here is going to correspond to a call to either np.fromfile or np.memmap.
        read_list = []

        for ind_key in key_list:
            if ind_key is None:
                # This basically helps flush out the last read/integration in this loop
                int_size = int_start = 0
            else:
                (_, int_size, int_start) = int_start_dict[ind_key]
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

        # Time to actually read in the data
        if use_mmap:
            # memmap is a little special, in that it wants the _absolute_ offset rather
            # than the relative offset that np.fromfile uses (if passing a file object
            # rather than a string with the path toward the file).
            for read_dict in read_list:
                int_data_dict.update(
                    zip(
                        read_dict["inhid_list"],
                        np.memmap(
                            filename=full_filepath,
                            dtype=read_dict["int_dtype_dict"],
                            mode="r",
                            offset=read_dict["start_offset"],
                            shape=(read_dict["num_vals"],),
                        ),
                    )
                )
        else:
            with open(full_filepath, "rb") as visibilities_file:
                # Note that we do one open here to avoid the overheads associated with
                # opening and closing the file each integration.
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

        return int_data_dict

    @staticmethod
    def make_packdata(sp_data, raw_data):
        """
        Write packdata from raw_data.

        This method will convert raw data into packed data, ready to be written to
        disk (i.e., MIR-formatted). This method is typically called by file writing
        utilities.

        Parameters
        ----------
        sp_data : ndarray of sp_data_type
            Array from the file "sp_read", returned by `read_sp_data`.
        raw_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "scale_fac", and np.int16 which describes the common
            exponent for the spectrum, and "raw_data", an array of np.int16 (of length
            equal to twice that found in `sp_data["nch"]` for the corresponding value
            of sphid) containing the compressed visibilities.  Note that entries equal
            to -32768 aren't possible with the compression scheme used for MIR, and so
            this value is used to mark flags.

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
        # Before we start, grab some relevant metadata
        inhid_arr = sp_data["inhid"]
        sphid_arr = sp_data["sphid"]

        # Each channel containts two int16 values (real + imag), plus each
        # spectral record has an int16 up front as the common exponent
        record_size_arr = (2 * sp_data["nch"].astype(int)) + 1

        # We need to run through the spectral records and see:
        # a) Which sphids match to which inhids,
        # b) Where each spectral record fits inside the "packed" data array, and
        # c) The total size of the packed data array for each integration
        inhid_offset_dict = {}
        inhid_sphid_dict = {}
        for inhid, sphid, record_size in zip(inhid_arr, sphid_arr, record_size_arr):
            try:
                # We assign here to avoid having to do this lookup twice
                temp_val = inhid_offset_dict[inhid]
                # By keeping a running tally of the integration size, we can also
                # figure out where each spectral record fits into the paced data
                inhid_offset_dict[inhid] = record_size + temp_val
                inhid_sphid_dict[inhid][sphid] = (temp_val, (record_size + temp_val))
            except KeyError:
                # If no key is found, it means we're processing this inhid for the
                # first time, so plug in a record accordingly.
                inhid_offset_dict[inhid] = record_size
                inhid_sphid_dict[inhid] = {sphid: (0, record_size)}

        # Figure out all of the unique dtypes we need for constructing the individual
        # packed datasets (where we need a different dtype based on the number of
        # individual visibilities we're packing in).
        int_dtype_dict = {}

        for int_size in np.unique(list(inhid_offset_dict.values())):
            int_dtype_dict[int_size] = np.dtype(
                [
                    ("inhid", np.int32),
                    ("nbyt", np.int32),
                    ("packdata", np.int16, int_size),
                ]
            ).newbyteorder("little")

        # Now we start the heavy lifting -- start looping through the individual
        # integrations and pack them together.
        int_data_dict = {}
        for inhid, int_size in inhid_offset_dict.items():
            # Make an empty packdata dtype, which we will fill with new values
            packdata = np.empty((), dtype=int_dtype_dict[int_size])

            # Convenience dict which contains the sphids as keys and start/stop of
            # the slice for each spectral record as values for each integrtation.
            datapos_dict = inhid_sphid_dict[inhid]

            # Plug in the "easy" parts of packdata
            packdata["inhid"] = inhid
            packdata["nbyt"] = int_size * 2

            # Now step through all of the spectral records and plug it in to the
            # main packdata array. In testing, this worked out to be a good degree
            # faster than running np.concat.
            for sphid, (ch_start, ch_stop) in datapos_dict.items():
                sp_raw = raw_data[sphid]
                packdata["packdata"][ch_start] = sp_raw["scale_fac"]
                packdata["packdata"][(ch_start + 1) : ch_stop] = sp_raw["raw_data"]

            int_data_dict[inhid] = packdata

        return int_data_dict

    @staticmethod
    def write_rawdata(filepath, raw_dict, sp_data, append_data=False):
        """
        Write packed data to disk.

        Parameters
        ----------
        filepath : str
            String  describing the folder in which the data should be written.
        int_data_dict : dict
            Dict containing packed data
        append_data : bool
            Option whether to append to an existing file, if one already exists, or to
            overwrite any existing data. Default is False (overwrite existing data).
        """
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # In order to write to disk, we need to create an intermediate product known
        # as "packed data", which is written on a per-integration basis. These create
        # duplicate copies of the data which can cause the memory footprint to balloon,
        # So we want to just create one packdata entry at a time. To do that, we
        # actually need to sgement sp_data by the integration ID.
        sp_in_dict, _ = MirParser.segment_by_index(sp_data, "inhid")

        # We can now open the file once, and write each array upon construction
        with open(
            os.path.join(filepath, "sch_read"), "ab+" if append_data else "wb+"
        ) as file:
            for key in sorted(sp_in_dict.keys()):
                packdata = MirParser.make_packdata(sp_in_dict[key], raw_dict)[key]
                packdata.tofile(file)

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
            exponent for the spectrum, and "raw_data", an array of np.int16 (of length
            equal to twice that found in `sp_data["nch"]` for the corresponding value
            of sphid) containing the compressed visibilities.  Note that entries equal
            to -32768 aren't possible with the compression scheme used for MIR, and so
            this value is used to mark flags.

        Returns
        -------
        vis_dict : dict
            A dictionary in the format of `vis_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "vis_data", an array of np.complex64 containing the
            visibilities, and "vis_flags", an array of bool containing the per-channel
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
                "vis_data": (
                    (np.float32(2) ** sp_raw["scale_fac"]) * sp_raw["raw_data"]
                ).view(dtype=np.complex64),
                "vis_flags": sp_raw["raw_data"][::2] == -32768,
            }
            for sphid, sp_raw in raw_dict.items()
        }

        # In testing, flagging the bad channels out after-the-fact was significantly
        # faster than trying to much w/ the data above.
        for item in vis_dict.values():
            item["vis_data"][item["vis_flags"]] = 0.0

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
            with two items: "vis_data", an array of np.complex64 containing the
            visibilities, and "vis_flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid).

        Returns
        -------
        raw_dict : dict
            A dictionary in the format of `raw_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "scale_fac", and np.int16 which describes the common
            exponent for the spectrum, and "raw_data", an array of np.int16 (of length
            equal to twice that found in `sp_data["nch"]` for the corresponding value
            of sphid) containing the compressed visibilities.  Note that entries equal
            to -32768 aren't possible with the compression scheme used for MIR, and so
            this value is used to mark flags.
        """
        # Similar to convert_raw_to_vis, fair bit of testing went into making this as
        # fast as possible. Strangely enough, frexp is _way_ faster than ldexp.
        # Note that we only want to calculate a common exponent based on the unflagged
        # spectral channels.
        scale_fac = np.frexp(
            [
                np.abs(sp_vis["vis_data"].view(dtype=np.float32)).max(initial=1e-45)
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
                "raw_data": np.where(
                    sp_vis["vis_flags"],
                    np.complex64(-32768 - 32768j),
                    sp_vis["vis_data"] * (np.float32(2) ** (-sfac)),
                )
                .view(dtype=np.float32)
                .astype(np.int16),
            }
            for sfac, (sphid, sp_vis) in zip(scale_fac, vis_dict.items())
        }

        return raw_dict

    @staticmethod
    def read_vis_data(
        filepath,
        int_start_dict,
        sp_data,
        return_vis=True,
        return_raw=False,
        use_mmap=True,
        read_only=False,
    ):
        """
        Read "sch_read" mir file into a list of ndarrays. (@staticmethod).

        Parameters
        ----------
        filepath : str or sequence of str
            Path to the folder(s) containing the mir data set.
        int_start_dict: dict or sequence of dict
            Dictionary returned from scan_int_start, which records position and
            record size for each integration
        sp_data : ndarray of sp_data_type
            Array from the file "sp_read", returned by `read_sp_data`.
        return_vis : bool
            If set to True, will return a dictionary containing the visibilities read
            in the "normal" format. Default is True.
        return_vis : bool
            If set to True, will return a dictionary containing the visibilities read
            in the "raw" format. Default is False.
        use_mmap : bool
            If False, then each integration record needs to be read in before it can
            be parsed on a per-spectral record basis (which can be slow if only reading
            a small subset of the data). Default is True, which will leverage mmap to
            access data on disk (that does not require reading in the whole record).
            There is usually no performance penalty to doing this, although reading in
            data is slow, you may try seeing this to False and seeing if performance
            improves.

        Returns
        -------
        raw_data : dict
            A dictionary, whose the keys are matched to individual values of sphid in
            `sp_data`, and each entry comtains a dict with two items: "scale_fac",
            and np.int16 which describes the common exponent for the spectrum, and
            "raw_data", an array of np.int16 (of length equal to twice that found in
            `sp_data["nch"]` for the corresponding value of sphid) containing the
            compressed visibilities.  Note that entries equal to -32768 aren't possible
            with the compression scheme used for MIR, and so this value is used to mark
            flags. Only returned in return_raw=True.
        vis_dict : dict
            A dictionary in the format of `vis_data`, where the keys are matched to
            individual values of sphid in `sp_data`, and each entry comtains a dict
            with two items: "vis_data", an array of np.complex64 containing the
            visibilities, and "vis_flags", an array of bool containing the per-channel
            flags of the spectrum (both are of length equal to `sp_data["nch"]` for the
            corresponding value of sphid). Only returned in return_vis=True.
        """
        # If filepath is just a str, make it a list so we can iterate over it
        if isinstance(filepath, str):
            filepath = [filepath]

        # Same thing for int_start_dict (if it's just a single dict)
        if isinstance(int_start_dict, dict):
            int_start_dict = [int_start_dict]

        if len(filepath) != len(int_start_dict):
            raise ValueError(
                "Must provide a sequence of the same length for "
                "filepath and int_start_dict"
            )

        # Gather the needed metadata that we'll need in order to read in the data.
        inhid_arr = sp_data["inhid"]
        sphid_arr = sp_data["sphid"]
        # Cast to int64 here to avoid overflow issues (since nch is int16)
        nch_arr = sp_data["nch"].astype(np.int64)
        # The divide by two here is because each value is int16 (2 bytes), whereas
        # dataoff records of "offset" in number of bytes.
        dataoff_arr = sp_data["dataoff"] // 2

        unique_inhid = np.unique(inhid_arr)

        # Begin the process of reading the data in, stuffing the "packdata" arrays
        # (to be converted into "raw" data) into the dict below.
        int_data_dict = {}
        check_dict = {}
        for file, startdict in zip(filepath, int_start_dict):
            int_data_dict.update(
                MirParser.read_packdata(
                    file,
                    {
                        idx: startdict[idx]
                        for idx in np.intersect1d(unique_inhid, list(startdict.keys()))
                    },
                    use_mmap=use_mmap,
                )
            )
            check_dict.update(startdict)

        # With the packdata in hand, start parsing the individual spectral records.
        raw_dict = {}
        for inhid in unique_inhid:
            # There is very little to check in the packdata records, so make sure
            # that this entry corresponds to the inhid and size we expect.
            if (int_data_dict[inhid]["inhid"] != check_dict[inhid][0]) or (
                int_data_dict[inhid]["nbyt"] != check_dict[inhid][1]
            ):
                raise ValueError(
                    "Values in int_start_dict do not match that recorded inside "
                    "the data file. Run `fix_int_start` to regenerate the integration "
                    "position information."
                )

            # Pop here let's us delete this at the end (and hopefully let garbage
            # collection do it's job correctly).
            packdata = int_data_dict.pop(inhid)["packdata"]

            # Select out the right subset of metadata
            data_mask = inhid_arr == inhid
            dataoff_subarr = dataoff_arr[data_mask]
            nch_subarr = nch_arr[data_mask]
            sphid_subarr = sphid_arr[data_mask]

            # Dataoff marks the starting position of the record
            start_idx = dataoff_subarr
            # Each record has a real & imag value per channel, plus one common exponent
            end_idx = start_idx + (nch_subarr * 2) + 1

            # We copy here if we want the raw values AND we've used memmap, since
            # otherwise the resultant entries in raw_data will be memmap arrays, which
            # will be read only (and we want attributes to be modifiable.)
            raw_dict.update(
                {
                    shpid: {
                        "scale_fac": (
                            packdata[idx].copy()
                            if ((return_raw and use_mmap) and not read_only)
                            else packdata[idx]
                        ),
                        "raw_data": (
                            packdata[idx + 1 : jdx].copy()
                            if ((return_raw and use_mmap) and not read_only)
                            else packdata[idx + 1 : jdx]
                        ),
                    }
                    for shpid, idx, jdx in zip(sphid_subarr, start_idx, end_idx)
                }
            )
            # Do the del here to break the reference to the "old" data so that
            # subsequent assignments don't cause issues for raw_dict.
            del packdata

        # Figure out which results we need to pass back
        results = ()

        if return_raw:
            results += (raw_dict,)

        if return_vis:
            results += (MirParser.convert_raw_to_vis(raw_dict),)

        return results

    @staticmethod
    def read_auto_data(filepath, int_start_dict, ac_data, winsel=None):
        """
        Read "autoCorrelations" mir file into memory (@staticmethod).

        Note that this returns as an array, since there isn't any unique index for the
        autocorrelations file.

        Parameters
        ----------
        filepath : str or sequence of str
            Path to the folder containing the mir data set.
        int_start_dict : dict or sequence of dict
            Dictionary (or sequence of dictonaries) which stores the integration file
            positions, the keys of which are used for determining which values on the
            integration header number (inhid) match which file.
        ac_data : arr of dtype ac_read_dtype
            Structure from returned from scan_auto_data.
        winsel : list of int (optional)
            List of spectral windows to include.

        Returns
        -------
        auto_data : dict
            A dictionary, whose keys are matched to values to achid in ac_data, and
            values contain the spectrum of individual auto-correlation records, of
            type np.float32.
        """
        if isinstance(filepath, str):
            filepath = [filepath]

        if isinstance(int_start_dict, dict):
            int_start_dict = [int_start_dict]

        if len(filepath) != len(int_start_dict):
            raise ValueError(
                "Must provide a sequence of the same length for "
                "filepath and int_start_dict"
            )

        if winsel is None:
            winsel = np.arange(0, ac_data["nchunks"][0])

        # The current generation correlator always produces 2**14 == 16384 channels per
        # spectral window.
        # TODO read_auto_data: Allow this work w/ spectrally averaged data
        # (although this is presently blocked by the way the _old_ rechunker behaves)
        auto_data = {}

        for file, startdict in zip(filepath, int_start_dict):
            # Select out the appropriate records for this file
            ac_mask = np.isin(ac_data["inhid"], list(startdict.keys()))

            dataoff_arr = ac_data["dataoff"][ac_mask]
            nvals_arr = ac_data["datasize"][ac_mask].astype(np.int64) // 4
            achid_arr = ac_data["achid"][ac_mask]
            with open(os.path.join(file, "autoCorrelations"), "rb") as auto_file:
                # Start scanning through the file now.
                lastpos = 0
                for achid, dataoff, nvals in zip(achid_arr, dataoff_arr, nvals_arr):
                    deloff = dataoff - lastpos
                    auto_data[achid] = np.fromfile(
                        auto_file, dtype=np.float32, count=nvals, offset=deloff,
                    ).reshape((-1, 2, 2 ** 14))[winsel]
                    lastpos = dataoff + (4 * nvals)

        return auto_data

    @staticmethod
    def make_codes_dict(codes_read):
        """
        Make a dictionary from codes_read.

        Generates a dictionary based on a codes_read array.

        Parameters
        ----------
        codes_read : ndarray of dtype codes_dtype
            Array from the file "codes_read", returned by `read_codes_data`.

        Returns
        -------
        codes_dict : dict
            Dictionary whose keys match the value of "v_name" in `codes_read`, with
            values made up of a dictionary with keys corresponding to "icode" and values
            of a tuple made of of ("code", "ncode") from each element of `codes_read`.
        """
        codes_dict = {}

        # These are entries that are supposed to change over the course of the track,
        # but _may_ not if there is only a single entry. For sake of consistency, we
        # force these to behave like v_names that have multiple entries
        code_diff_allowed = [
            "ref_time",
            "ut",
            "vrad",
            "source",
            "stype",
            "svtype",
            "project",
            "ra",
            "dec",
        ]

        for item in codes_read:
            v_name = item["v_name"].decode("UTF-8")
            try:
                codes_dict[v_name][0].append(item["code"].decode("UTF-8"))
                codes_dict[v_name][1].append(int(item["icode"]))
                codes_dict[v_name][2].append(int(item["ncode"]))
            except KeyError:
                codes_dict[v_name] = [
                    [item["code"].decode("UTF-8")],
                    [int(item["icode"])],
                    [int(item["ncode"])],
                ]

        for key in codes_dict.keys():
            if (
                len(codes_dict[key][0]) == 1
                and (key not in code_diff_allowed)
                and (codes_dict[key][1] == [0] and codes_dict[key][2] == [1])
            ):
                codes_dict[key] = codes_dict[key][0][0]
            else:
                codes_dict[key] = {
                    idx: (code, ncode)
                    for code, idx, ncode in zip(
                        codes_dict[key][0], codes_dict[key][1], codes_dict[key][2]
                    )
                }

        return codes_dict

    @staticmethod
    def make_codes_read(codes_dict):
        """
        Make from codes_read array from a dictionary.

        Generates an array of dtype codes_dtype, based on a dictionary. Note that the
        format should match that output by `make_codes_dict`.

        Parameters
        ----------
        codes_dict : dict
            Dictionary whose keys match the value of "v_name" in `codes_read`, with
            values made up of a dictionary with keys corresponding to "icode" and values
            of a tuple made of of ("code", "ncode") from each element of `codes_read`.

        Returns
        -------
        codes_read : ndarray of dtype codes_dtype
            Array of dtype codes_dtype, with an entry corresponding to each unique
            entry in `codes_dict`.
        """
        codes_tuples = []
        for key, value in codes_dict.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    codes_tuples.append((key, subkey, subval[0], subval[1]))
            else:
                codes_tuples.append((key, 0, value, 1))

        codes_read = np.array(codes_tuples, dtype=codes_dtype)

        return codes_read

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
        if not self._vis_data_loaded:
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
            (idx, jdx, 0): tsys ** 0.5
            for idx, jdx, tsys in zip(
                self.eng_data["inhid"],
                self.eng_data["antennaNumber"],
                self.eng_data["tsys"],
            )
        }
        tsys_dict.update(
            {
                (idx, jdx, 1): tsys ** 0.5
                for idx, jdx, tsys in zip(
                    self.eng_data["inhid"],
                    self.eng_data["antennaNumber"],
                    self.eng_data["tsys_rx2"],
                )
            }
        )

        # now create a per-blhid SEFD dictionary based on antenna pair, integration
        # timestep, and receiver pairing.
        normal_dict = {
            blhid: (2.0 * self.jypk)
            * (tsys_dict[(idx, jdx, kdx)] * tsys_dict[(idx, ldx, mdx)])
            for blhid, idx, jdx, kdx, ldx, mdx in zip(
                self.bl_data["blhid"],
                self.bl_data["inhid"],
                self.bl_data["iant1"],
                self.bl_data["ant1rx"],
                self.bl_data["iant2"],
                self.bl_data["ant2rx"],
            )
        }

        if invert:
            for key, value in normal_dict.items():
                normal_dict[key] = 1.0 / value

        # Finally, multiply the individual spectral records by the SEFD values
        # that are in the dictionary.
        for sphid, blhid in zip(self.sp_data["sphid"], self.sp_data["blhid"]):
            self.vis_data[sphid]["vis_data"] *= normal_dict[blhid]

        self._tsys_applied = not invert

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
        # Note that we have to modify out checking here if _file_dict is empty, since
        # it means that all spectral records have to be loaded already, even if not
        # selected.
        arr_fmt = "_%s_read" if (self._file_dict == {}) else "%s_data"

        # Set this list up here to make the code that follows a bit more generic (so
        # that we can have 1 loop rather than 3 if statements).
        check_list = [
            (self._vis_data_loaded, arr_fmt % "sp", "sphid", self.vis_data),
            (self._raw_data_loaded, arr_fmt % "sp", "sphid", self.vis_data),
            (self._auto_data_loaded, arr_fmt % "ac", "achid", self.auto_data),
        ]

        # Run our check
        for (load_state, idx_attr, idx_field, data_arr) in check_list:
            if not load_state:
                # If not loaded, move along
                continue
            if sorted(getattr(self, idx_attr)[idx_field]) != sorted(data_arr.keys()):
                # If we have a mismatch, we can leave ASAP
                return False

        # If you got to this point, it means that we've got agreement!
        return True

    def _downselect_data(self, select_vis=True, select_raw=True, select_auto=True):
        """Do a thing."""
        # TODO _downselect_data: Needs a docstring
        if self._file_dict == {}:
            # There isn't anything to do here, because we can't downselect if there
            # isn't an associated file object. Bail at this point.
            return

        try:
            # Check that vis_data has all entries we need for processing the data
            if self._vis_data_loaded and select_vis:
                vis_data = {
                    sphid: self.vis_data[sphid] for sphid in self.sp_data["sphid"]
                }

            # Now check raw_data
            if self._raw_data_loaded and select_raw:
                raw_data = {
                    sphid: self.raw_data[sphid] for sphid in self.sp_data["sphid"]
                }

            # Now check auto_data
            if self._auto_data_loaded and select_auto:
                auto_data = {
                    achid: self.auto_data[achid] for achid in self.ac_data["achid"]
                }
        except KeyError:
            raise KeyError(
                "Missing spectral records in data attributes. Run load_data instead."
            )

        # At this point, we can actually plug our values in, since we know that the
        # operation above succeeded.
        if self._vis_data_loaded and select_vis:
            self.vis_data = vis_data
        if self._raw_data_loaded and select_raw:
            self.raw_data = raw_data
        if self._auto_data_loaded and select_auto:
            self.auto_data = auto_data

    def load_data(
        self,
        load_vis=None,
        load_raw=None,
        load_auto=None,
        apply_tsys=True,
        use_mmap=True,
        read_only=False,
        allow_downselect=None,
        allow_conversion=None,
    ):
        """
        Load visibility data into MirParser class.

        Parameters
        ----------
        load_vis : bool
            Load the visibility data (floats) into object (deault is True).
        load_raw : bool
            Load the raw visibility data (ints) into object (default is False).
        load_auto: bool
            Load the autos (floats) into object (default is False).
        apply_tsys : bool
            If load_vis is set to true, apply tsys corrections to the data (default
            is True).
        use_mmap : bool
            If False, then each integration record needs to be read in before it can
            be parsed on a per-spectral record basis (which can be slow if only reading
            a small subset of the data). Default is True, which will leverage mmap to
            access data on disk (that does not require reading in the whole record).
            There is usually no performance penalty to doing this, although reading in
            data is slow, you may try seeing this to False and seeing if performance
            improves.
        allow_downselect : bool
            If data has been previously loaded, and all spectral records are currently
            contained in `vis_data`, `raw_data`, and/or `auto_data` (if `load_vis`,
            `load_raw`, and/or `load_auto` are True, respectively), then down-select
            from the currently loaded data rather than reading the data from disk.
        allow_conversion : bool
            Allow the method to convert previously loaded raw_data into "normal"
            visibility data.

        Raises
        ------
        UserWarning
            If there is no file to load data from.
        """
        if self._file_dict == {}:
            # If there is no file_dict, that _usually_ means that we added two
            # MirParser objects together that did not belong to the same file
            # with force=True, which breaks the underlying connection to the
            # data on disk. If this is the case, we want to tread a bit carefully,
            # and not actually unload any data if at all possible.
            if load_raw or (load_raw is None and not (load_vis or load_vis is None)):
                if self._raw_data_loaded:
                    warnings.warn(
                        "No file to load from, and raw data is already loaded, "
                        "skipping raw data load."
                    )
                else:
                    raise ValueError(
                        "Cannot load raw data from disk without a file to load from. "
                        "Set load_raw=False to continue."
                    )
            elif load_vis or (load_vis is None):
                if self._vis_data_loaded:
                    warnings.warn(
                        "No file to load from, and vis data is already loaded, "
                        "skipping vis data load."
                    )
                elif not self._raw_data_loaded:
                    raise ValueError(
                        "No file to load vis_data from, cannot run load_data."
                    )
                elif allow_conversion:
                    # Use update here to prevent wiping out other data stored in memory
                    self.vis_data = self.convert_raw_to_vis(
                        {
                            sphid: self.raw_data[sphid]
                            for sphid in self._sp_read["sphid"]
                        }
                    )
                    self._vis_data_loaded = True
                    if apply_tsys:
                        # Note that this online looks at sp_data["sphid"] to determine
                        # which records to update, which by the logic above are the
                        # only ones that we have converted from raw to vis.
                        self.apply_tsys()
                else:
                    raise ValueError(
                        "No file to load vis_data from, but raw_data is loaded. "
                        "Set allow_conversion=True to load in vis_data."
                    )
            if allow_downselect:
                warnings.warn(
                    "allow_downselect argument ignored because no file to load from."
                )
            return

        # Last chance before we load data -- see if we already have raw_data in hand,
        # and just need to convert it. If allow_conversion is None, we should decide
        # first whether or not to attempt this.
        if allow_conversion or (allow_conversion is None):
            if not (load_vis or (load_vis is None)) or (
                load_raw and (load_vis is None)
            ):
                # Literally nothing to do here because we don't want the vis_data
                allow_conversion = False
            elif load_raw:
                # We can't do conversion if we need to load up the autos/raw from disk.
                if allow_conversion:
                    warnings.warn("Cannot load raw data AND convert, moving on.")
                allow_conversion = False
            elif not self._raw_data_loaded:
                # Also can't convert data if no raw data was loaded to begin with.
                if allow_conversion:
                    warnings.warn(
                        "Raw data not loaded, cannot convert, attempting alternatives."
                    )
                allow_conversion = False
            elif not np.all(np.isin(self.sp_data["sphid"], list(self.raw_data.keys()))):
                if allow_conversion:
                    warnings.warn(
                        "Loaded raw data does not contain all spectral records, "
                        "skipping conversion."
                    )
                allow_conversion = False
            else:
                allow_conversion = True

        if allow_conversion:
            self.vis_data = self.convert_raw_to_vis(self.raw_data)
            self._vis_data_loaded = True
            self._tsys_applied = False
            # If we need to apply tsys, do that now.
            if apply_tsys:
                self.apply_tsys()

        # If we are potentially downselecting data (typically used when calling select),
        # make sure that we actually have all the data we need loaded.
        if allow_downselect or (allow_downselect is None):
            data_list = []
            # We group both types of cross-correlation data here (raw + normal). Mostly
            # we do this because the heavy
            if (load_vis or (load_vis is None)) or (load_raw or (load_raw is None)):
                data_list.append("cross")

            if load_auto or (load_auto is None):
                data_list.append("auto")

            # Anything we aren't downselecting we should be unloading at this point
            self.unload_data(
                unload_vis=(not (load_vis or (load_vis is None))),
                unload_raw=(not (load_raw or (load_raw is None))),
                unload_auto=(not (load_auto or (load_auto is None))),
            )

            for data in data_list:
                try:
                    self._downselect_data(
                        select_vis=data == "cross",
                        select_raw=data == "cross",
                        select_auto=data == "auto",
                    )
                except KeyError:
                    # If we can't downselect, then we have to unload the data.
                    self.unload_data(
                        unload_vis=data == "cross",
                        unload_raw=data == "cross",
                        unload_auto=data == "auto",
                    )
                    if allow_downselect is not None:
                        warnings.warn(
                            "Cannot downselect %s data, attempting alternatives." % data
                        )
        else:
            # If we can't downselect, then we don't trust that the data attributes
            # have only the spectral records that we want. Unload them now, except
            # for vis_data if we just generated that from raw_data.
            self.unload_data(
                unload_vis=(not allow_conversion), unload_raw=True, unload_auto=True,
            )

        # At this point, we've unloaded any data we can't carry forward, so if we still
        # need to load stuff from the file, we can handle this now.
        if (load_vis is None) and (load_raw is None):
            # If letting us choose between vis_data and raw_data, give preference
            # to vis_data. Note we only have to load the data if it hasn't been loaded.
            load_vis = not self._vis_data_loaded

        if load_raw is None:
            # Load raw_data only if won't have loaded vis_data, or otherwise
            # if we already have the raw_data loaded into memory.
            load_raw = not (
                (load_vis or self._vis_data_loaded) or self._raw_data_loaded
            )
        elif load_vis is None:
            # Load vis_data only if won't have loaded raw_data, or otherwise
            # if we already have the vis_data loaded into memory.
            load_vis = not (
                (load_raw or self._raw_data_loaded) or self._vis_data_loaded
            )

        # Finally, if we can't downselect, load the data in now using the information
        # stored in _file_dict.
        if load_vis or load_raw:
            vis_tuple = self.read_vis_data(
                list(self._file_dict.keys()),
                list(self._file_dict.values()),
                self.sp_data,
                return_vis=load_vis,
                return_raw=load_raw,
                use_mmap=use_mmap,
                read_only=read_only,
            )

            # Because the read_vis_data returns a tuple of varying length depending on
            # return_vis and return_raw, we want to parse that here.
            if load_vis and load_raw:
                self.raw_data, self.vis_data = vis_tuple
            elif load_vis:
                (self.vis_data,) = vis_tuple
            elif load_raw:
                (self.raw_data,) = vis_tuple

            # Finally, mark whether or not we loaded these asttributes
            if load_vis:
                self._vis_data_loaded = True
                # Since we've loaded in "fresh" data, we mark that tsys has
                # not yet been applied (otherwise apply_tsys can thrown an error).
                self._tsys_applied = False

                # Apply tsys if needed.
                if apply_tsys and load_vis:
                    self.apply_tsys()
            if load_raw:
                self._raw_data_loaded = True

        # We wrap the auto data here in a somewhat special way because of some issues
        # with the existing online code and how it writes out data. At some point
        # we will fix this, but for now, we triage the autos here. Note that if we
        # already have the auto_data loaded, we can bypass this step.
        if load_auto and self._has_auto and (not self._auto_data_loaded):
            # Have to do this because of a strange bug in data recording where
            # we record more autos worth of spectral windows than we actually
            # have online.
            winsel = np.unique(self.sp_data["corrchunk"])
            winsel = winsel[winsel != 0].astype(int) - 1
            self.auto_data = self.read_auto_data(
                list(self._file_dict.keys()),
                list(self._file_dict.values()),
                self.ac_data,
                winsel=winsel,
            )
            self._auto_data_loaded = True

    def unload_data(self, unload_vis=True, unload_raw=True, unload_auto=True):
        """
        Unload data from the MirParser object.

        Unloads the data-related attributes from memory, if they are loaded. Because
        these attributes can be formidible in size, this operation will substantially
        reduce the memory footprint of the MirParser object.

        Note that you cannot use this operation if adding together to MirParser
        ojbects with force=True.

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

        Raises
        ------
        ValueError
            If attempting to unload data if there is no file to load visibility data
            from.
        """
        if self._file_dict == {}:
            raise ValueError(
                "Cannot unload data as there is no file to load data from."
            )

        if unload_vis:
            self.vis_data = None
            self._vis_data_loaded = False
            self._tsys_applied = False
        if unload_raw:
            self.raw_data = None
            self._raw_data_loaded = False
        if unload_auto:
            self.auto_data = None
            self._auto_data_loaded = False

    def _update_filter(
        self,
        use_in=None,
        use_bl=None,
        use_sp=None,
        update_data=None,
        allow_downselect=False,
    ):
        """
        Update MirClass internal filters for the data.

        Expands the internal 'use_in', 'use_bl', and 'use_sp' arrays to
        construct filters for the individual structures/data.

        use_in : bool
            Boolean array of shape (N_in, ), where `N_in = len(self.in_data)`, which
            marks with integration records to include.
        use_bl : bool
            Boolean array of shape (N_bl, ), where `N_bl = len(self.bl_data)`, which
            marks with baseline records to include.
        use_sp : bool
            Boolean array of shape (N_sp, ), where `N_bl = len(self.sp_data)`, which
            marks with baseline records to include.
        update_data : bool
            If set to True, will read in data from disk after selecting records. If
            set to False, data attributes (e.g., `vis_data`, `raw_data`, `auto_data`)
            will be unloaded. If set to True, data attributes will be reloaded, based
            on what had been previously.  Default is to downselect the data from that
            previously unloaded if possible, otherwise unload the data.
        """
        in_filter = np.zeros(len(self._in_read), dtype=bool)
        bl_filter = np.zeros(len(self._bl_read), dtype=bool)
        sp_filter = np.zeros(len(self._sp_read), dtype=bool)

        in_filter[use_in] = True
        bl_filter[use_bl] = True
        sp_filter[use_sp] = True

        in_inhid = self._in_read["inhid"]
        bl_inhid = self._bl_read["inhid"]
        bl_blhid = self._bl_read["blhid"]
        sp_blhid = self._sp_read["blhid"]
        sp_sphid = self._sp_read["sphid"]

        # Filter out de-selected bl records
        bl_filter[bl_filter] = np.isin(bl_inhid[bl_filter], in_inhid[in_filter])

        # Filter out de-selected sp records
        sp_filter[sp_filter] = np.isin(sp_blhid[sp_filter], bl_blhid[bl_filter])

        # Check for bl records that have no good sp records
        # Filter out de-selected bl records
        bl_filter[bl_filter] = np.isin(
            bl_blhid[bl_filter], np.unique(sp_blhid[sp_filter]), assume_unique=True,
        )

        # Check for in records that have no good bl records
        # Filter out de-selected in records
        in_filter[in_filter] = np.isin(
            in_inhid[in_filter], np.unique(bl_inhid[bl_filter]), assume_unique=True,
        )

        in_inhid = in_inhid[in_filter]
        bl_blhid = bl_blhid[bl_filter]
        sp_sphid = sp_sphid[sp_filter]

        # Filter out the last three data products, based on the above
        eng_filter = np.isin(self._eng_read["inhid"], in_inhid)
        we_filter = np.isin(self._we_read["scanNumber"], in_inhid)
        ac_filter = (
            np.isin(self._ac_read["inhid"], in_inhid) if self._has_auto else None
        )

        if allow_downselect:
            try:
                # If we are downselecting, we want to make sure that we are definitely
                # selecting a subset of the prior data, which we can check by
                # looking at the filters that already exist in the data.
                assert np.all(self._in_filter[in_filter])
                assert np.all(self._eng_filter[eng_filter])
                assert np.all(self._bl_filter[bl_filter])
                assert np.all(self._sp_filter[sp_filter])
                assert np.all(self._we_filter[we_filter])
                if self._has_auto:
                    assert np.all(self._ac_filter[ac_filter])

                # Note that these final set of checks are for making sure that we can
                # downselect the data itself, assuming that it's been loaded. This
                # should ensure that _downselect_data works below without error.
                if self._vis_data_loaded:
                    assert np.all(
                        np.isin(list(self.vis_data.keys(), self.sp_data["sphid"]))
                    )
                if self._raw_data_loaded:
                    assert np.all(
                        np.isin(list(self.raw_data.keys(), self.sp_data["sphid"]))
                    )
                if self._auto_data_loaded:
                    assert np.all(
                        np.isin(list(self.auto_data.keys(), self.ac_data["achid"]))
                    )
            except AssertionError:
                allow_downselect = False

        if allow_downselect:
            # If we are allowing the data to be downselected, then we just want to
            # select from the already loaded data vales. Note that we skip codes
            # and ant positions since those are assumed to be constant over the track.
            self.in_data = self.in_data[in_filter[self._in_filter]]
            self.eng_data = self.in_data[eng_filter[self._eng_filter]]
            self.bl_data = self.in_data[bl_filter[self._bl_filter]]
            self.sp_data = self.in_data[sp_filter[self._sp_filter]]
            self.we_data = self.in_data[we_filter[self._we_filter]]
            if self._has_auto:
                self.ac_data = self.ac_data[ac_filter[self._ac_filter]]
            else:
                self.ac_data = None
        else:
            self.in_data = self._in_read[in_filter]
            self.eng_data = self._eng_read[eng_filter]
            self.bl_data = self._bl_read[bl_filter]
            self.sp_data = self._sp_read[sp_filter]
            self.we_data = self._we_read[we_filter]
            if self._has_auto:
                self.ac_data = self._ac_read[ac_filter]
            else:
                self.ac_data = None

            # Also "refresh" the codes_dict and antpos values, just in case
            # the user changed them under the hood.
            self.codes_dict = self.make_codes_dict(self._codes_read)
            self.antpos_data = self._antpos_read.copy()

        # Now go through and update the filters
        self._in_filter = in_filter
        self._eng_filter = eng_filter
        self._bl_filter = bl_filter
        self._sp_filter = sp_filter
        self._we_filter = we_filter
        self._ac_filter = ac_filter

        # Craft some dictionaries so you know what list position matches
        # to each index entry. This helps avoid ordering issues.
        self._inhid_dict = {inhid: idx for idx, inhid in enumerate(in_inhid)}
        self._blhid_dict = {blhid: idx for idx, blhid in enumerate(bl_blhid)}
        self._sphid_dict = {sphid: idx for idx, sphid in enumerate(sp_sphid)}

        if (update_data is None) or (update_data and allow_downselect):
            try:
                self._downselect_data()
            except KeyError:
                self.unload_data()
        elif update_data and not self._data_mucked:
            self.load_data(
                load_vis=self._vis_data_loaded,
                load_raw=self._raw_data_loaded,
                load_auto=self._auto_data_loaded,
                apply_tsys=self._tsys_applied,
                allow_downselect=False,
                allow_conversion=False,
            )
        else:
            if update_data:
                warnings.warn("Unable to update data attributes, unloading them now.")
            self.unload_data()
            self._data_mucked = False

    def fromfile(
        self, filepath, has_auto=False, load_vis=False, load_raw=False, load_auto=False,
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
        self._in_read = self.read_in_data(filepath)  # Per integration records
        self._eng_read = self.read_eng_data(filepath)  # Per antenna-int records
        self._bl_read = self.read_bl_data(filepath)  # Per baaseline-int records
        self._sp_read = self.read_sp_data(filepath)  # Per spectral win-bl-int records
        self._codes_read = self.read_codes_data(filepath)  # Metadata for the track
        self._we_read = self.read_we_data(filepath)  # Per-int weather data
        self._antpos_read = self.read_antennas(filepath)  # Antenna positions

        # This indexes the "main" file that contains all the visibilities, to make
        # it faster to read in the data.
        self._file_dict = {
            os.path.abspath(filepath): self.calc_int_start(self._sp_read)
        }
        self.filepath = filepath

        self._has_auto = has_auto
        if self._has_auto:
            # If the data has auto correlations, then scan the auto file, pull out
            # the metadata, and get the data index locatinos for faster reads.
            self._ac_read = self.scan_auto_data(filepath)
        else:
            self._ac_read = None

        # Raw data aren't loaded on start, because the datasets can be huge
        # You can force this after creating the object with load_data().
        # Calling to unload_data will set all the relevant fields  we need.
        self.unload_data(unload_vis=True, unload_raw=True, unload_auto=True)

        # _data_mucked records if we've done something where the loaded data may not
        # match the metadata sorted in the *_read attributes, and thus some care is
        # required before loading up data from disk. This is normally unset by
        # running select(reset=True).
        self._data_mucked = False

        # This value is the forward gain of the antenna (in units of Jy/K), which is
        # multiplied against the system temperatures in order to produce values in units
        # of Jy (technically this is the SEFD, which when multiplied against correlator
        # coefficients produces visibilities in units of Jy). Default is 130.0, which
        # is the estiamted value for SMA.
        self.jypk = 130.0

        # _update_filter will assign all of the *_filter attributes, as well as the
        # user-facing *_data attributes on call, in addition to the various *hid_dict's
        # that map ID number to array index position.
        self._update_filter()

        # If requested, now we load out the visibilities.
        self.load_data(
            load_vis=load_vis,
            load_raw=load_raw,
            load_auto=(load_auto and self._has_auto),
        )

    def __init__(
        self,
        filepath=None,
        has_auto=False,
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
        # On init, if a filepath is provided, then fill in the object
        if filepath is not None:
            self.fromfile(
                filepath,
                has_auto=has_auto,
                load_vis=load_vis,
                load_raw=load_raw,
                load_auto=load_auto,
            )

    def tofile(
        self,
        filepath,
        write_raw=True,
        load_data=False,
        append_data=False,
        append_codes=False,
        bypass_append_check=False,
    ):
        """
        Write a MirParser object to disk in MIR format.

        Writes out a MirParser object to disk, in the binary MIR format. This method
        can worth with either a full dataset, or partial datasets appended together
        multiple times.

        Parameters
        ----------
        filepath : str
            Path of the directory to write out the data set.
        write_raw : bool
            If set to True (default), the method will attempt to write out the data
            stored in the `raw_data` attribute, and if this attribute is unpopulated,
            will then revert to using `vis_data` instead. If set to False, the
            preference order is swapped -- `vis_data` will be used first, but if not
            populated, `raw_data` will be used instead. This option has no effect if
            using a metadata-only MirParser object.
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
            self.load_data(load_vis=False, load_raw=True, load_auto=False)

        # If appending, we want to make sure that there are no clashes between the
        # header IDs we are about to write to disk, and the ones that already exist
        # on disk.
        if append_data and (not bypass_append_check):
            try:
                inhid_check = self.read_in_data(filepath)["inhid"]
                if np.any(self.in_data["inhid"], inhid_check):
                    raise ValueError(
                        "Cannot append data when integration header IDs overlap."
                    )
                blhid_check = self.read_bl_data(filepath)["blhid"]
                if np.any(self.bl_data["blhid"], blhid_check):
                    raise ValueError(
                        "Cannot append data when baseline header IDs overlap."
                    )
                sphid_check = self.read_sp_data(filepath)["sphid"]
                if np.any(self.bl_data["sphid"], sphid_check):
                    raise ValueError(
                        "Cannot append data when spectral record header IDs overlap."
                    )
            except FileNotFoundError:
                # If there's no file, then we have nothing to worry about.
                pass

        # Start out by writing the metadata out to file the various files
        self.write_in_data(filepath, self.in_data, append_data=append_data)
        self.write_eng_data(filepath, self.eng_data, append_data=append_data)
        self.write_bl_data(filepath, self.bl_data, append_data=append_data)
        self.write_sp_data(filepath, self.sp_data, append_data=append_data)
        # Note that the handling of codes a bit special, on account of the fact that
        # they should not change over the course of a single track.
        self.write_codes_data(
            filepath,
            self.make_codes_read(self.codes_dict),
            append_data=(append_data and append_codes),
        )
        self.write_we_data(filepath, self.we_data, append_data=append_data)
        self.write_antennas(filepath, self.antpos_data)

        # Now handle the data -- if no data has been loaded, then it's time to bail
        if not (self._vis_data_loaded or self._raw_data_loaded):
            warnings.warn("No data loaded, writing metadata only to disk")
            return
        elif (self._raw_data_loaded and write_raw) or (not self._vis_data_loaded):
            # If we have raw_data and we prefer to use that, or if vis_data is not
            # loaded, then we can just grab the raw data dict directly from the object.
            raw_dict = self.raw_data
        else:
            # Otherwise, if using vis_data, we need to convert that to the raw format
            # before we write the data to disk.
            if self._tsys_applied:
                warnings.warn(
                    "Writing out raw data with tsys applied. Be aware that you will "
                    "need to use set apply_tsys=True when calling load_data."
                    "Otherwise, call apply_tsys(invert=True) prior to writing out "
                    "the data set."
                )
            raw_dict = self.convert_vis_to_raw(self.vis_data)

        # Finally, we can package up the raw data (using make_packdata) in order to
        # write the raw-format data to disk.
        self.write_rawdata(
            filepath, raw_dict, self.sp_data, append_data=append_data,
        )

    @staticmethod
    def _rechunk_vis(vis_dict, chan_avg_arr, inplace=False):
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
            "vis_data" (the visibility data, dtype=np.complex64) and "vis_flags"
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
        new_vis_dict = vis_dict if inplace else {}

        for chan_avg, (sphid, sp_vis) in zip(chan_avg_arr, vis_dict.items()):
            # If there isn't anything to average, we can skip the heavy lifting
            # and just proceed on to the next record.
            if chan_avg == 1:
                if not inplace:
                    new_vis_dict[sphid] = copy.deepcopy(sp_vis)
                continue

            # Otherwise, we need to first get a handle on which data is "good"
            # for spectrally averaging over.
            good_mask = ~sp_vis["vis_flags"].reshape((-1, chan_avg))

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
                sp_vis["vis_data"].reshape((-1, chan_avg)).sum(where=good_mask, axis=-1)
                * temp_count
            )

            # Finally, plug the spectrally averaged data back into the dict
            new_vis_dict[sphid] = {
                "vis_data": temp_vis,
                "vis_flags": temp_count == 0,  # Flag channels with no valid data
            }

        return vis_dict

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
            keys "raw_data" (the raw visibility data, dtype=np.int16) and "scale_fac"
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
            "vis_data" (the visibility data, dtype=np.complex64) and "vis_flags"
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

        for chan_avg, (sphid, sp_raw) in zip(chan_avg_arr, raw_dict.items()):
            # If the number of channels to average is 1, then we just need to make
            # a deep copy of the old data and plug it in to the new dict.
            if chan_avg == 1:
                if not inplace:
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
                data_dict[sphid] = MirParser.rechunk_vis(
                    MirParser.convert_raw_to_vis({0: sp_raw}),
                    [chan_avg],
                    inplace=False,
                )[0]
            else:
                data_dict[sphid] = MirParser.convert_vis_to_raw(
                    MirParser.rechunk_vis(
                        MirParser.convert_raw_to_vis({0: sp_raw}),
                        [chan_avg],
                        inplace=False,
                    )
                )[0]

        # Finally, return the dict containing the raw data.
        return data_dict

    @staticmethod
    def _rechunk_auto(auto_dict, chan_avg_arr, inplace=False):
        """
        Rechunk auto-correlation spectra.

        Note this routine is not intended to be called by users, but instead is a
        low-level call from the `rechunk` method of MirParser to spectrally average
        data.

        Parameters
        ----------
        auto_dict : dict
            A dict containing visibility data, where the keys match to individual values
            of `achid` in `ac_data`, with each value being an auto-correlation spectrum
            (dtype=np.float32).
        chan_avg_arr : sequence of int
            A list, array, or tuple of integers, specifying how many channels to
            average over within each spectral record.
        inplace : bool
            If True, entries in `auto_dict` will be updated with spectrally averaged
            data. If False (default), then the method will construct a new dict that
            will contain the spectrally averaged data.

        Returns
        -------
        new_auto_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `auto_dict`.
        """
        new_auto_dict = auto_dict if inplace else {}

        for chan_avg, (achid, sp_auto) in zip(chan_avg_arr, auto_dict.items()):
            # If there isn't anything to average, we can skip the heavy lifting
            # and just proceed on to the next record.
            if chan_avg == 1:
                if not inplace:
                    new_auto_dict[achid] = copy.deepcopy(sp_auto)
                continue

            # The autos are a bit simpler than the vis/raw data -- there are no
            # flags (yet), so all we need to do is to average the data.
            new_auto_dict[achid] = (
                sp_auto["vis_data"].reshape((-1, chan_avg)).mean(axis=-1)
            )

        return new_auto_dict

    def rechunk(
        self, chan_avg, load_vis=False, load_raw=False, use_mmap=True,
    ):
        """
        Rechunk a MirParser object.

        Spectrally average a MIR dataset. This command attempts to emulate the old
        "SMARechunker" program within the MirParser object. Users should be aware
        that running this operation modifies the metadata in such a way that new data
        will not be able to be loaded until running `select(reset=True)`.

        Note that this command will only process data from the "normal" spectral
        windows, and not the pseudo-continuum data (which will remain untouched).

        Parameters
        ----------
        chan_avg : int
            Number of contiguous spectral channels to average over.
        load_vis : bool
            If set to True, "normal" visibility data will be loaded and specrally
            averaged on the fly. Useful for when attempting to read in large datasets.
            Default is False, which results in only previously loaded data being
            processed.
        load_raw : bool
            Similar to `load_raw`, if set to True, the raw visibility data will be
            loaded from disk and spectrally averaged on the fly. Default is False,
            which results in only previously loaded data being processed.
        use_mmap : bool
            If False, then each integration record needs to be read in before it can
            be parsed on a per-spectral record basis (which can be slow if only reading
            a small subset of the data). Default is True, which will leverage mmap to
            access data on disk (that does not require reading in the whole record).
            There is usually no performance penalty to doing this, although reading in
            data is slow, you may try seeing this to False and seeing if performance
            improves.
        """
        # Start of by doing some argument checking.
        arg_dict = {"chan_avg": chan_avg}
        for key, value in arg_dict.items():
            if not (isinstance(value, int) or isinstance(value, np.int_)):
                raise ValueError("%s must be of type int." % key)
            elif value < 1:
                raise ValueError("%s cannot be a number less than one." % key)

        if load_vis or load_raw:
            if self._data_mucked:
                raise ValueError(
                    "Cannot load data due to modifications of metadata records. "
                    "Run select(reset=True) in order to clear this issue, or "
                    "set load_data and load_raw to False."
                )
            elif self._file_dict == {}:
                raise ValueError(
                    "Cannot unload data as there is no file to load data from."
                )
        else:
            if chan_avg == 1:
                # This is a no-op, so we can actually bail at this point.
                return
            if (not (self._vis_data_loaded or self._raw_data_loaded)) and (
                not self._auto_data_loaded
            ):
                warnings.warn("No data loaded to average, returning.")
                return
            if not self._check_data_index():
                # If the above returns False, then we have a problem, and can't
                # actually run this operation (have to reload the data).
                raise ValueError(
                    "Index values do not match data keys. Data will need to be "
                    "reloaded before continuing (with select(reset=True)."
                )

        sp_data = self._sp_read if (self._file_dict == {}) else self.sp_data
        chan_avg_arr = np.where(sp_data["corrchunk"] == 0, 1, chan_avg)

        if np.any(np.mod(sp_data["nch"], chan_avg_arr) != 0):
            raise ValueError(
                "chan_avg does not go evenly into the number of channels in each "
                "spectral window (typically chan_avg should be set to a power of 2)."
            )

        # Note we are about to modify the data AND metadata, so mark this
        # object as mucked to prevent us from trusting the metadata.
        self._data_mucked = True

        if not (load_vis or load_raw):
            # If no associated file dict, then we need to deal with _all_ the spectral
            # records in _sp_read, not just the ones in sp_data.
            if self._vis_data_loaded:
                self._rechunk_vis(self.vis_data, chan_avg_arr, inplace=True)
            if self._raw_data_loaded:
                self._rechunk_raw(self.raw_data, chan_avg_arr, inplace=True)
            if self._auto_data_loaded:
                self._rechunk_auto(self.auto_data, chan_avg_arr, inplace=True)
        else:
            if self._raw_data_loaded or (
                self._vis_data_loaded or self._auto_data_loaded
            ):
                warnings.warn(
                    "Setting load_data or load_raw to true will unload "
                    "previously loaded data."
                )
            self.unload_data()

            # If we're going to load the data, then we only want to load a block of
            # data at a time, otherwise we run the risk of overtaxing memory by loading
            # in the data at a resolution we can't handle. To do this, we need to break
            # up the sp_data by integration ID, which segment_by_index will do for us.
            sp_in_dict, pos_dict = self.segment_by_index(sp_data, "inhid")
            chan_avg_arr = []

            for subset_sp_data, pos_arr in zip(sp_in_dict.values(), pos_dict.values()):
                # Pluck out the relevant entries from the channel averaging array we
                # calculated earlier.
                temp_chan_avg = chan_avg_arr[pos_arr]

                # We only want to load one set of data, and for the moment, we can
                # save some work by _just_ loading the raw data.
                (data_dict,) = self.read_vis_data(
                    list(self._file_dict.keys()),
                    list(self._file_dict.values()),
                    subset_sp_data,
                    return_vis=False,
                    return_raw=True,
                    use_mmap=use_mmap,
                )

                if load_raw and not load_vis:
                    # If we _only_ want the raw data, we can save a bit on memory
                    # by calling _rechunk_raw and replacing the records in-situ within
                    # data_dict, when can then be used to update raw_data.
                    self._rechunk_raw(data_dict, temp_chan_avg, in_place=True)
                    self.raw_data.update(data_dict)
                else:
                    # Otherwise, if we ever want the visibility data, it's faster to
                    # allow _rechunk_raw to return "normal" data (which it converts
                    # records to for the rechunk), and then deal with the conversion
                    # to raw data later if needed.
                    self.vis_data.update(
                        self._rechunk_raw(data_dict, temp_chan_avg, return_vis=True)
                    )

                # Delete data_dict, just to break any potential references and allow
                # cache to clear.
                del data_dict

            if load_vis and load_raw:
                # If we want vis and raw, make the raw copy now based on vis.
                self.raw_data = self.convert_vis_to_raw(self.vis_data)

            self._vis_data_loaded = load_vis
            self._raw_data_loaded = load_raw

        # Last task - update the metadata about num of channels accordingly,
        # and clobber the dataoff values.
        sp_data["nch"] = sp_data["nch"] // chan_avg_arr
        sp_data["dataoff"] = 0
        if self._file_dict == {}:
            self.sp_data["nch"] = self._sp_read["nch"][self._sp_filter]

    @staticmethod
    def _combine_read_arr_check(arr1, arr2, index_name=None, any_match=False):
        """
        Check if two MirParser metadata arrays have conflicting index values.

        This method is an internal helper function not meant to be called by users.
        It checks two arrays of metadata of one of the custom dtypes used by MirParser
        (for the attributes `in_data`, `bl_data`, `sp_data`, `eng_data`, `we_data`,
        and `_codes_read`) to make sure that if they contain identical index values,
        both arrays contain identical metadata. Used in checking if two arrays can
        be combined without conflict (via the method `_combine_read_arr`).

        Parameter
        ---------
        arr1 : ndarray
            Array of metadata, to be compared to `arr2`.
        arr2 : ndarray
            Array of metadata, to be compared to `arr1`.
        index_name : str
            Name of the field which contains the unique index information (e.g., inhid
            for `_in_read`). No default, not requird if `arr1` and `arr2` have a dtype
            of codes_dtype. Typically, "inhid" is matched to elements in in_read and
            eng_read, "blhid" to bl_read, "sphid" to sp_read, "achid" to ac_read, and
            "scanNumber" to we_read.
        any_match : bool
            Nominally the method checks to see if all fields in each array match when
            overlapping indicies exist. However, if this is set to True, it will check
            instead if any elements with overlapping indicies have metadata that agree.

        Returns
        -------
        check_status : bool
            If True, and `any_match=False`, any entries between the two arrays where
            the index value matches has metadata which is indentical, and thus should
            be safe to merge. If True and `any=True`, then the two arrays have at
            least one entry where the index value matches and the metadata agrees.
        """
        # First up, make sure we have two ndarrays of the same dtype
        if arr1.dtype != arr2.dtype:
            raise ValueError("Both arrays must be of the same dtype.")

        # For a codes_read array, the indexing procedure is a bit funky,
        # so we handle this as a special case.
        if arr1.dtype == codes_dtype:
            # Entries should be uniquely indexed by the combination of v_name, icode,
            # and ncode (the latter of which is _usually_ ignored anyways). The code
            # is allowed to take on whatever value it wants (duplicate or not).
            arr1_dict = {
                (item["v_name"], item["icode"], item["ncode"]): item["code"]
                for item in arr1
            }
            arr2_dict = {
                (item["v_name"], item["icode"], item["ncode"]): item["code"]
                for item in arr2
            }

            for key in arr1_dict.keys():
                try:
                    # Check to see that if a key (v_name, icode, ncode) exists in both
                    # sets of codes_read, that the value (code) is the same in both.
                    if not (arr1_dict[key] == arr2_dict[key]):
                        return False
                    elif any_match:
                        return True
                except KeyError:
                    # If the keys don't confict, then there's no potential clash, so
                    # just move on to the next entry.
                    pass
            return True

        # If not using codes_read, the logic here is a bit simpler. Make sure that
        # index_name is actually a string, since other types can produce spurious
        # results.
        if not isinstance(index_name, str):
            raise ValueError("index_name must be a string.")

        # Make sure the field name actually exists in the array
        if index_name not in arr1.dtype.names:
            raise ValueError("index_name not a recognized field in either array.")

        # Finally, figure out where we have overlapping entries, based on the field
        # used for indexing entries.
        _, idx1, idx2 = np.intersect1d(
            arr1[index_name], arr2[index_name], return_indices=True
        )

        # Finally, compare where we have coverlap and make sure the two arrays
        # are the same.
        if any_match:
            check_status = np.any(arr1[idx1] == arr2[idx2])
        else:
            check_status = np.array_equal(arr1[idx1], arr2[idx2])

        return check_status

    @staticmethod
    def _combine_read_arr(
        arr1, arr2, index_name=None, return_indices=False, overwrite=False
    ):
        """
        Combine two MirParser metadata arrays.

        This method is an internal helper function not meant to be called by users.
        It combines two arrays of metadata of one of the custom dtypes used by MirParser
        into a single array.

        Parameters
        ----------
        arr1 : ndarray
            Array of metadata, to be combined with `arr2`.
        arr2 : ndarray
            Array of metadata, to be combined with `arr1`. Note that if `arr1` and
            `arr2` have overlapping index values, entries in `arr2` will overwrite
            those in `arr1`.
        index_name : str
            Name of the field which contains the unique index information (e.g., inhid
            for `_in_read`). No default, not requird if `arr1` and `arr2` have a dtype
            of codes_dtype. Typically, "inhid" is matched to elements in in_read and
            eng_read, "blhid" to bl_read, "sphid" to sp_read, "achid" to ac_read, and
            "scanNumber" to we_read.
        return_indices : bool
            If set to True, return which index values of `arr1` were merged into the
            final metadata array.
        overwrite : bool
            If `arr1` and `arr2` have overlapping index values, then an error will be
            thrown in the metadata for those entries is not the same. However, if
            set to True, then these differences will be ignored, and values of `arr2`
            will overwrite conflicting entries in `arr1` in the final array.

        Returns
        -------
        comb_arr : ndarray
            Array of the combined values of `arr1` and `arr2`, of the same dtype as
            both of these arrays.
        arr1_index : ndarray
            Array of index positions of `arr1` which were used in `comb_arr` (i.e., not
            overwritten or otherwise dropped). Returned only if `return_indices=True`.

        Raises
        ------
        ValueError
            If `arr1` and `arr2` are of different dtypes, or if entries in `index_name`
            overlap in the two arrays but `overwrite=False`. Also if `index_name` is
            not a string, or is not a field in `arr1` or `arr2`.
        """
        if overwrite:
            # If we are overwriting, then make sure that the two arrays are of the
            # same dtype before continuing.
            if arr1.dtype != arr2.dtype:
                raise ValueError("Both arrays must be of the same dtype.")
        else:
            # If not overwriting, check and make sure that there are no overlapping
            # entries which might clobber differing metadata.
            if not MirParser._combine_read_arr_check(arr1, arr2, index_name=index_name):
                raise ValueError(
                    "Arrays have overlapping indicies with different data, "
                    "cannot combine the two safely."
                )

        # At this point, we are assuming we are safe to overwrite entries (either
        # because its safe to do so or because the user told us that we could). We want
        # to then pluck out the entries in arr1 that have non-overlapping index values
        # versus arr2. To do so, first we find the overlapping values.
        if arr1.dtype == codes_dtype:
            # For a codes_read array, the indexing procedure is a bit funky,
            # so we handle this as a special case. Make a set of unique indexing
            # values to compare against from arr2.
            arr2_set = {(item["v_name"], item["icode"], item["ncode"]) for item in arr2}

            # Loop through each entry in arr1 -- if we have a matching index from arr2,
            # then grab the position within the array.
            idx1 = np.array(
                [
                    idx
                    for idx, item in enumerate(arr1)
                    if (item["v_name"], item["icode"], item["ncode"]) in arr2_set
                ]
            )
        else:
            # If not a codes_read type array, the logic here is a bit simpler.
            if not isinstance(index_name, str):
                raise ValueError("index_name must be a string.")

            if index_name not in arr1.dtype.names:
                raise ValueError("index_name not a recognized field in either array.")

            # Use intersect1d to find where we have overlap from arr1, and grab the
            # array positions where said overlap occurs
            _, idx1, _ = np.intersect1d(
                arr1[index_name], arr2[index_name], return_indices=True
            )

        # We basically want to invert our selection here, so use arange to make the
        # full list of possible position indexes to generate a list of non-overlapping
        # metadata entries.
        arr_sel = np.isin(np.arange(len(arr1)), idx1, invert=True)
        result = np.concatenate(arr1[arr_sel], arr2)

        # If we are returning the index positions, stuff that into the result now
        if return_indices:
            result = (result, arr_sel)

        return result

    def __add__(self, other_obj, overwrite=False, force=False, inplace=False):
        """
        Add two MirParser objects.

        Combine two MirParser objects together, nominally under the assumption that
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
        force : bool
            Normally, if attempting to combine MirParser objects that were created from
            different files, the method will throw an error. This is done because it
            partially breaks the normal operating mode of MirParser, where data can
            be loaded up from disk on-the-fly on an as-needed basis. If set to True,
            different objects can be combined, although the ability to read on demand
            will be disabled for the resultant object (i.e., only data loaded into
            memory will be available). Default is False.
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
            `overwrite=False`), different data being loaded (raw vs vis vs auto), or
            because the two objects appear to be loaded from different files (and
            `force=False`).
        """
        if not isinstance(other_obj, MirParser):
            raise TypeError(
                "Cannot add a MirParser object an object of a different type."
            )

        # First check that the metadata are compatible. First up, if we are _not_
        # permitted to adjust HID numbers in the file, make sure that we don't have
        # conflicting indicies.
        attr_index_dict = {
            "in_data": "inhid",
            "eng_data": "inhid",
            "bl_data": "blhid",
            "sp_data": "sphid",
            "we_data": "scanNumber",
            "antpos_data": "antenna",
        }

        comp_list = [
            "_in_read",
            "_eng_read",
            "_bl_read",
            "_we_read",
            "_codes_read",
            "_antpos_read",
            "_file_dict",
        ]

        # Check and see if both objects have autos, and if so, add that to the list of
        # metadata to check.
        if self._has_auto != other_obj._has_auto:
            # Deal with this after we check other metadata, but warn the user now.
            warnings.warn(
                "Both objects do not have auto-correlation data. Will unload from "
                "the final object."
            )
        elif self._has_auto:
            comp_list.append("_ac_read")
            attr_index_dict["ac_read"] = "achid"

        # First thing -- check that everything here appears to belong to the same file
        force_list = []
        for item in comp_list:
            if not np.array_equal(getattr(self, item), getattr(self, item)):
                force_list.append(item)

        # If we have evidence that these two objects belong to different files, then
        # we can proceed in a few different ways.
        if force_list != []:
            force_list = ", ".join(force_list)
            if force:
                # If we are forcing the two objects together, it'll basically sever our
                # link to the underlying file, so we need to make sure that the data
                # are actually loaded here, and if using vis_data, that both have
                # tsys actually applied.
                if not (
                    (self._vis_data_loaded and other_obj._vis_data_loaded)
                    or (self._raw_data_loaded and other_obj._raw_data_loaded)
                ):
                    raise ValueError(
                        "Cannot combine objects with force=True when no vis or raw "
                        "data gave been loaded. Run the `load_data` method on both "
                        "objects (with the same arguments) to clear this error."
                    )
                elif self._tsys_applied != other_obj._tsys_applied:
                    raise ValueError(
                        "Cannot combine objects with force=True where one object "
                        "has tsys correction applied and the other does not. Run "
                        "load_data(apply_tsys=True) on both objects to correct this."
                    )
                else:
                    warnings.warn(
                        "Objects here do not appear to be from the same file, but "
                        "proceeding ahead since force=True (%s clashes)." % force_list
                    )
            else:
                # If these are different files, then the user should be using concat
                # instead. You can of course bypass this with force=True.
                raise ValueError(
                    "Objects appear to come from different files, based on "
                    "differences in %s. You can use the `concat` method to combine "
                    "data from different files, or if you can set force=True to "
                    "forgo this check." % force_list
                )

        # Okay, so now we know that either the files are the same or otherwise we don't
        # care. Now more on to the data arrays.
        overwrite_list = []
        for item, index in attr_index_dict.items():
            if not MirParser._combine_read_arr_check(
                getattr(self, item), getattr(other_obj, item), index_name=index
            ):
                overwrite_list.append(item)

        # Since codes_dict is a dict (as the name implies), we handle it specially
        # outside of the above loop.
        if self.codes_dict != other_obj.codes_dict:
            overwrite_list.append("codes_dict")

        # Alert the user if we are about to overwrite any data, or raise an error if
        # we aren't allowed to do so.
        if overwrite_list != []:
            overwrite_list = ", ".join(overwrite_list)
            if overwrite:
                warnings.warn(
                    "Data in objects appears to overlap, but with differing metadata. "
                    "Proceeding since overwrite=True (%s overlaps)." % overwrite_list
                )
            else:
                raise ValueError(
                    "Objects appear to contain overlapping data, where the metadata "
                    "differs in %s. This can be corrected by calling `load_data` on "
                    "the individual objects to reload the metadata, or by setting "
                    "overwrite=True, which will pull metadata from the second object "
                    "in the add sequence." % overwrite_list
                )

        # One final check - see if we have the smae type of data loaded or not.
        if self._vis_data_loaded != other_obj._vis_data_loaded:
            raise ValueError(
                "Cannot combine objects where one has vis data loaded and the other "
                "does not. Run the `load_data` method on both objects (with the same "
                "arguments) to clear this error."
            )
        elif self._vis_data_loaded:
            # Does the data have the same normalization? If not, raise an error.
            if self._tsys_applied != other_obj._tsys_applied:
                raise ValueError(
                    "Cannot combine objects where one has tsys normalization applied "
                    "and the other does not. Run the `load_data` method on both "
                    "objects (with the same arguments) to clear this error."
                )

        # Check if both have the same state for the raw data.
        if self._raw_data_loaded != other_obj._raw_data_loaded:
            raise ValueError(
                "Cannot combine objects where one has raw data loaded and the other "
                "does not. Run the `load_data` method on both objects (with the same "
                "arguments) to clear this error."
            )

        # Finally, check if both have the same state for the raw data.
        if self._auto_data_loaded != other_obj._auto_data_loaded:
            raise ValueError(
                "Cannot combine objects where one has auto data loaded and the other "
                "does not. Run the `load_data` method on both objects (with the same "
                "arguments) to clear this error."
            )

        # Now that we know we are good to go, begin by either making a copy of
        # or pointing to the original object in question.
        new_obj = self if inplace else self.copy()

        # Next merge the metadata
        for item, index in attr_index_dict.items():
            setattr(
                new_obj,
                item,
                MirParser._combine_read_arr(
                    getattr(self, item), getattr(other_obj, item), index_name=index
                ),
            )

        # Again. handle the codes dict special here.
        new_obj.codes_dict.update(other_obj.codes_dict)

        # Finally, since the various data arrays are stored as dicts, we can just
        # update them here.
        if new_obj.raw_data is not None:
            new_obj.raw_data.update(other_obj.raw_data)

        if new_obj.vis_data is not None:
            new_obj.raw_data.update(other_obj.raw_data)

        if self._has_auto != other_obj._has_auto:
            # Remember this check earlier? Now is the time to dump the auto data
            # if both objects didn't have it.
            new_obj.auto_data = None
            new_obj.ac_data = new_obj._ac_read = None
            new_obj._ac_filter = None
            new_obj._has_auto = False
        elif new_obj.auto_data is not None:
            # Otherwise fold in the auto data
            new_obj.auto_data.update(other_obj.auto_data)

        # Finally, we need to do a special bit of handling if we "forced" the two
        # objects together. If we did, then we need to update the core attributes
        # so that the *_data and *_read arrays all agree.
        if comp_list != []:
            new_obj._file_dict = {}

            new_obj._in_read = new_obj.in_data.copy()
            new_obj._in_filter = np.ones(new_obj._in_read, dtype=bool)

            new_obj._eng_read = new_obj.eng_data.copy()
            new_obj._eng_filter = np.ones(new_obj._eng_read, dtype=bool)

            new_obj._bl_read = new_obj.bl_data.copy()
            new_obj._bl_filter = np.ones(new_obj._bl_read, dtype=bool)

            new_obj._sp_read = new_obj.sp_data.copy()
            new_obj._sp_filter = np.ones(new_obj._sp_read, dtype=bool)

            new_obj._codes_read = MirParser.make_codes_read(new_obj.codes_dict)

            new_obj._we_read = new_obj.we_data.copy()
            new_obj._we_filter = np.ones(new_obj._we_read, dtype=bool)

            new_obj._antpos_read = new_obj.antpos_data.copy()

        if not inplace:
            return new_obj

    def __iadd__(self, other_obj, overwrite=False, force=False):
        """
        Add two MirParser objects in place.

        Combine two MirParser objects together, nominally under the assumption that
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
        force : bool
            Normally, if attempting to combine MirParser objects that were created from
            different files, the method will throw an error. This is done because it
            partially breaks the normal operating mode of MirParser, where data can
            be loaded up from disk on-the-fly on an as-needed basis. If set to True,
            different objects can be combined, although the ability to read on demand
            will be disabled for the resultant object (i.e., only data loaded into
            memory will be available). Default is False.

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
        self.__add__(other_obj, overwrite=overwrite, force=force, inplace=True)

    @staticmethod
    def concat(obj_list, force=True):
        """
        Concat multiple MirParser objects together.

        Concatenates together multiple MirParser objects (loaded from different files
        on disk) into a single object. Note that this method will only work if they are
        compatible, which means that the following criterion are met: data are taken in
        the same polarization mode, data are taken with the same correlator
        configuration (i.e., number of bands), and data are recorded with the same MIR
        file verison, that the antenna postiions record in each are the same, and the
        `_has_auto` flag has been set the same for all objects.

        Note that this method is intended for combining _different_ files together, and
        will normally throw an error if it appears as though two objects contain the
        same data. Users desiring to add together two datasets loaded from the same
        file (with, for example, different `select` commands run on each) should look
        to the `__add__` method instead.

        Users should be aware that this method will return an object where no data
        are loaded and any selection criterion have been reset (i.e., what you get
        when running `select(reset=True)`).

        Parameters
        ----------
        obj_list : list of MirParser objects
            List of MirParser objects to be combined.
        force : bool
            Normally, if one of the objects looks like it has some of the same data as
            another in the list to be concatenated, an error will be thrown. This is
            to prevent the same visibilities from being loaded into the same data file
            twice, which can have undesirable effects downstream. If set to True, this
            check will result in a warning instead of an error (if at all possible).
            Users should use this option with caution. Default is False.

        Returns
        -------
        new_obj : MirParser object
            A concatenated data set in a single MirParser object.

        Raises
        ------
        ValueError
            If objects appear to contain the same data or have differing values for the
            antenna positions or the `_has_data` flag (if `force=False`). Also if two
            objects point to the same file on disk from which data are loaded, of if
            two objects are otherwise not compatible for being concatenated (different
            file verison, polarization state, or correlator config).
        """
        # These are items that should be different in at least _some_ way, which
        # we can use np.array_equal to check between objects that this is the case.
        diff_list = [
            "_in_read",
            "_eng_read",
            "_bl_read",
            "_sp_read" "_we_read",
            "_codes_read",
        ]

        # Store some check values here, so that we can take appropriate action later.
        auto_check = obj_list[0]._has_auto
        antpos_check = False
        # raise_warning just makes it so that we sound the alarm once on a given
        # problem. No sense in flooding the channel w/ warning messages.
        raise_warning = True
        for idx, obj1 in enumerate(obj_list):
            # Nice try, buddy - can't concat non-MirParser objects.
            if not isinstance(obj1, MirParser):
                raise ValueError("Can only concat MirParser objects.")
            # Check that the autos are either all loaded or unloaded
            if obj1._has_auto != auto_check:
                if force:
                    auto_check = False
                else:
                    raise ValueError(
                        "Cannot combine objects both with and without auto-correlation "
                        "data. When reading data from file, must keep _has_auto set "
                        "consistently between file reads. You can bypass this error "
                        "(and unload all auto-correlation data) by setting force=True."
                    )
            # Make sure that this object actually points to a file on disk (otherwise
            # it's gonna break a lot of functionality in MirParser).
            if obj1._file_dict == {}:
                raise ValueError(
                    "Cannot concat objects without an associated file (this is caused "
                    "by adding objects together with force=True)."
                )
            # Time to do some inter-object comparisons.
            for obj2 in obj_list[:idx]:
                # Make sure that objects don't point to the same file on disk.
                # No double-loading of data allowed!
                for key in obj1._file_dict.keys():
                    if key in obj2._file_dict.keys():
                        raise ValueError(
                            "At least one object to be concatenated has been loaded "
                            "from the same file as another (index position %d). "
                            "Remove it from the list to continue." % idx
                        )
                # Check that the antenna positions agree
                if not np.array_equal(obj1._antpos_read, obj2._antpos_read):
                    if not force:
                        raise ValueError(
                            "Two of the objects provided do not have the same antenna "
                            "positions. You can bypass this error by setting "
                            "force=True (use with caution!)."
                        )
                    else:
                        antpos_check = True
                # Check that our diff_list actually looks different between objects
                for item in diff_list:
                    if np.array_equal(getattr(obj1, item), getattr(obj2, item)):
                        if not force:
                            raise ValueError(
                                "Two of the objects provided appear to hold identical "
                                "data. Verify that you don't have two objects that are "
                                "loaded from the same file. You can bypass this error "
                                "by setting force=True (use with caution!)."
                            )
                        elif raise_warning:
                            warnings.warn(
                                "Objects may contain the same data, pushing forward "
                                "anyways since force=False."
                            )
                            raise_warning = False

        # So at this point, we've checked to see that the list of objects either aren't
        # totally identical (or have forced the user to set force=True). Check real
        # quick that this isn't a single object (the path for which is much simpler)
        if len(obj_list) == 1:
            new_obj = obj_list[0].copy()
            new_obj.unload_data()
            return new_obj

        # Check if we have to effectively dump the auto data (and warn the user)
        for idx, obj1 in enumerate(obj_list):
            if auto_check != obj1._has_auto:
                warnings.warn(
                    "Some (but not all) objects have auto-correlation data -- ignoring "
                    "this data and setting _has_data=False on the returned object."
                )
                break

        # Also if forced, warn the user that we're about to adopt a single set of
        # antenna positions, even though the list has different ones.
        if antpos_check:
            warnings.warn(
                "Some objects have different antenna positions than others. Taking "
                "antenna positions from first object in the list and discarding the "
                "rest."
            )

        # Next item -- move on and check that the codes_read entries actually look
        # sensible. The list below entails the codes that are allowed to be different,
        # otherwise the entries need to be identical (note that this is a hard failure
        # at the moment since differences can signal truly incompatible data).
        code_diff_allowed = [
            "ref_time",
            "ut",
            "vrad",
            "source",
            "stype",
            "svtype",
            "project",
            "ra",
            "dec",
        ]
        # We're going to convert codes_read to a codes_dict for both convenience here
        # and also for later use when we combine the objects.
        codes_dict_list = []
        for idx, obj1 in enumerate(obj_list):
            obj1_codes = MirParser.make_codes_dict(obj1._codes_read)
            codes_dict_list.append(obj1_codes)
            for jdx, obj2 in enumerate(obj_list[:idx]):
                obj2_codes = codes_dict_list[jdx]

                # Check to make sure they objects have the same set of keys
                if list(obj1_codes.keys()) != list(obj2_codes.keys()):
                    raise ValueError(
                        "codes_dict contains different keys between objects, "
                        "data are not compatible."
                    )
                for key in obj1_codes.keys():
                    if key in code_diff_allowed:
                        continue
                    if obj1_codes[key] != obj2_codes[key]:
                        # Okay, so now we have a problem - something that shouldn't
                        # differ between the files does. Most probable explanations are
                        # filever, polarization, and band, so check those first in
                        # order to give a move informative error message (otherwise
                        # just report the offending key)
                        err_msg = "Cannot concat objects, "
                        if key == "filever":
                            err_msg += "differing file versions detected."
                        elif key == "pol":
                            err_msg += (
                                "differing polarization states detected (polarized"
                                "and non-polarized records present)."
                            )
                        elif key == "band":
                            err_msg += "differing correlator configurations detected "
                            "(different number of spectral bands per object)."
                        else:
                            err_msg += "%s key in codes_dict differs." % key
                        raise ValueError(err_msg)

        # Finally, if we haven't raised a warning yet, check and see if there are _any_
        # identical entries in in_read, since it contains information that should always
        # be unique to every observation.
        for idx, obj1 in enumerate(obj_list):
            # If we have already disabled the warning, it means that we have already
            # detected a problem but the user set force=True.
            if not raise_warning:
                break
            else:
                for obj2 in obj_list[:idx]:
                    if MirParser._combine_read_arr_check(
                        obj1._in_read, obj2._in_read, "inhid", any_match=True,
                    ):
                        if force:
                            warnings.warn(
                                "Objects may contain overlapping data, pushing "
                                "forward  anyways since force=False."
                            )
                            raise_warning = False
                            break
                        else:
                            raise ValueError(
                                "Two of the objects appear to have overlapping data. "
                                "Verify that you don't have two objects that are "
                                "loaded from the same file. You can bypass this error "
                                "by setting force=True (use with caution!)."
                            )

        # Alright, at this point we have checked everything that we needed to check,
        # and so now we want to actually start the process of merging all the data.
        # Start with in_read, since we need to cascade the inhid changes downward.
        in_read_list = []
        inhid_dict_list = []
        inhid_count = 1
        for obj in obj_list:
            # Make a copy of the array, for updating and later concat
            temp_in_read = obj._in_read.copy()

            # Map out the old index values to the new one
            old_inhid = temp_in_read["inhid"]
            new_inhid = np.arange(inhid_count, inhid_count + len(temp_in_read))

            # Up the count so that the next set of index values are unique.
            inhid_count += len(temp_in_read)

            # Make a dict so that we can map inhid values later
            inhid_dict_list.append(
                {oldid: newid for oldid, newid in zip(old_inhid, new_inhid)}
            )
            # Plug in the new index values
            temp_in_read["inhid"] = new_inhid
            temp_in_read["ints"] = new_inhid

            # Save this array for later when we construct the final object.
            in_read_list.append(temp_in_read)

        bl_read_list = []
        blhid_dict_list = []
        blhid_count = 1
        for idx, obj in enumerate(obj_list):
            # Make a copy of the array, for updating and later concat
            temp_bl_read = obj._bl_read.copy()

            # Map out the old index values to the new one
            old_blhid = temp_bl_read["blhid"]
            new_blhid = np.arange(blhid_count, blhid_count + len(temp_bl_read))

            # Up the count so that the next set of index values are unique.
            blhid_count += len(temp_bl_read)

            # Make a dict so that we can map blhid values later
            blhid_dict_list.append(
                {oldid: newid for oldid, newid in zip(old_blhid, new_blhid)}
            )
            # Plug in the new index values
            temp_bl_read["blhid"] = new_blhid

            # We've got the blhid index remade, now propagate changes to inhid.
            inhid_dict = inhid_dict_list[idx]
            temp_bl_read["inhid"] = [inhid_dict[idx] for idx in temp_bl_read["inhid"]]

            # Finally, add the updated array to the list.
            bl_read_list.append(temp_bl_read)

        sp_read_list = []
        sphid_dict_list = []
        sphid_count = 1
        for obj in enumerate(obj_list):
            # Make a copy of the array, for updating and later concat
            temp_sp_read = obj._sp_read.copy()

            # Map out the old index values to the new one
            old_sphid = temp_sp_read["sphid"]
            new_sphid = np.arange(sphid_count, sphid_count + len(temp_sp_read))

            # Up the count so that the next set of index values are unique.
            sphid_count += len(temp_sp_read)

            # Make a dict so that we can map sphid values later
            sphid_dict_list.append(
                {oldid: newid for oldid, newid in zip(old_sphid, new_sphid)}
            )
            # Plug in the new index values
            temp_sp_read["sphid"] = new_sphid

            # We've got the sphid index remade, now propagate changes to inhid/blhid.
            inhid_dict = inhid_dict_list[idx]
            blhid_dict = blhid_dict_list[idx]
            temp_sp_read["inhid"] = [inhid_dict[idx] for idx in temp_sp_read["inhid"]]
            temp_sp_read["blhid"] = [blhid_dict[idx] for idx in temp_sp_read["blhid"]]

            # Finally, add the updated array to the list.
            sp_read_list.append(temp_sp_read)

        # If we have autos, now is the time to update the spectral records
        if auto_check:
            ac_read_list = []
            achid_dict_list = []
            achid_count = 1
            for idx, obj in enumerate(obj_list):
                # Make a copy of the array, for updating and later concat
                temp_ac_read = obj._ac_read.copy()

                # Map out the old index values to the new one
                old_achid = temp_ac_read["achid"]
                new_achid = np.arange(achid_count, achid_count + len(temp_ac_read))

                # Up the count so that the next set of index values are unique.
                achid_count += len(temp_ac_read)

                # Make a dict so that we can map achid values later
                achid_dict_list.append(
                    {oldid: newid for oldid, newid in zip(old_achid, new_achid)}
                )
                # Plug in the new index values
                temp_ac_read["achid"] = new_achid

                # We've got the achid index remade, now propagate changes to inhid.
                inhid_dict = inhid_dict_list[idx]
                temp_ac_read["inhid"] = [
                    inhid_dict[idx] for idx in temp_ac_read["inhid"]
                ]

                # Finally, add the updated array to the list.
                ac_read_list.append(temp_ac_read)

        # Now that the "primary" data structures are fixed, we'll move on to the
        # data that only requires updating inhids, namely eng_read and we_read ()
        eng_read_list = []
        for idx, obj in enumerate(obj_list):
            # Make a copy of the array for update
            temp_eng_read = obj._eng_read.copy()

            # Update index values
            inhid_dict = inhid_dict_list[idx]
            temp_eng_read["inhid"] = [inhid_dict[idx] for idx in temp_eng_read["inhid"]]
            temp_eng_read["inhid"] = temp_eng_read["ints"]
            # Hold on to array for later concat
            eng_read_list.append(temp_eng_read)

        we_read_list = []
        for idx, obj in enumerate(obj_list):
            # Make a copy of the array for update
            temp_we_read = obj._we_read.copy()

            # Update index values
            inhid_dict = inhid_dict_list[idx]
            temp_we_read["scanNumber"] = [
                inhid_dict[idx] for idx in temp_we_read["scanNumber"]
            ]
            # Hold on to array for later concat
            we_read_list.append(temp_we_read)

        # Last, but not least, we need to deal with codes_read dicts and combine them
        # together, specifically the entries in code_diff_allowed.
        project_count = ref_time_count = source_count = 0
        ref_time_dict = {}
        source_dict = {}
        for idx, temp_codes in enumerate(codes_dict_list):
            # Codes should all match to entries in in_read, so grab that and the
            # inhid dict in case we need to make some updates.
            inhid_dict = inhid_dict_list[idx]
            in_read = in_read_list[idx]
            # First up, deal with the codes that should match to inhid, so handling
            # them is very easy, and doesn't require any updates to in_read.
            for item in ["ut", "ra", "dec", "vrad"]:
                sub_codes = temp_codes[item]
                sub_codes = {
                    inhid_dict[key]: sub_codes[key] for key in sub_codes.keys()
                }
                temp_codes[item] = sub_codes

            # Next up, deal with the project codes, which just requires entering them
            # in squence.
            project_map = {}
            for key in temp_codes["project"].keys():
                project_count += 1
                project_map[key] = project_count

            temp_codes["project"] = {
                project_map[key]: value for key, value in temp_codes["project"].items()
            }
            # Update in_read with the new index codes for project
            in_read["iproject"] = [project_map[key] for key in in_read["iproject"]]

            # Next up, go through ref_times. These _can_ be identical across files,
            # so we want to check for uniqueness here
            ref_time_map = {}
            for key, value in temp_codes["ref_time"].items():
                try:
                    ref_time_map[key] = ref_time_dict[value]
                except KeyError:
                    ref_time_count += 1
                    ref_time_dict[value] = ref_time_map[key] = ref_time_count

            temp_codes["ref_time"] = {
                ref_time_map[key]: value
                for key, value in temp_codes["ref_time"].items()
            }
            # Update in_read with the new index codes for ref_time
            in_read["iref_time"] = [ref_time_map[key] for key in in_read["iref_time"]]

            # Okay, the final tricky bit -- deal with the source information, which
            # is _technically_ shared across three codes. If source is the same, all
            # three _should_ be the same, but we want to make sure. Note that all
            # three codes here by design share the same icode for the same source,
            # and therefore the same key in codes_dict.
            source_map = {}
            for key in temp_codes["source"].keys():
                temp_key = (
                    temp_codes["source"][key],
                    temp_codes["stype"][key],
                    temp_codes["svtype"][key],
                )
                try:
                    source_map[key] = source_dict[temp_key]
                except KeyError:
                    source_count += 1
                    source_map[key] = source_dict[temp_key] = source_count

            for item in ["source", "stype", "svtype"]:
                temp_codes[item] = {
                    source_map[key]: value for key, value in temp_codes[item].items()
                }

            # Update in_read with the new index codes for source
            in_read["isource"] = [source_map[key] for key in in_read["isource"]]

            if idx == 0:
                codes_dict = temp_codes
            else:
                for key in code_diff_allowed:
                    codes_dict[key].update(temp_codes[key])

        # And with that, we now have everything we need -- all that's left is some
        # concats and variable assignments. Note that "_read" objects contain the
        # whole data set, while "_data" contains that after filtering/selecting.

        # Spin up a new object to start plugging things into.
        new_obj = MirParser()
        new_obj._in_read = np.concatenate(in_read_list)  # Per integration records
        new_obj._eng_read = np.concatenate(eng_read_list)  # Per antenna-int records
        new_obj._bl_read = np.concatenate(bl_read_list)  # Per baaseline-int records
        new_obj._sp_read = np.concatenate(sp_read_list)  # Per spectral win-bl-int recs
        new_obj._we_read = np.concatenate(we_read_list)  # Per-int weather data

        # We just spent a lot of time making the codes_dict correct, so here we now
        # have to convert that back into the codes_read format.
        new_obj._codes_read = MirParser.make_codes_read(codes_dict)

        # Antenna positions we handle special, since there can be only one. Adopt the
        # values found in the first object in the list.
        new_obj._antpos_read = obj_list[0]._antpos_read

        # Pull together all of the _file_dicts that were contained here
        new_obj._file_dict = {
            key: value for obj in obj_list for key, value in obj._file_dict.items()
        }

        # Keep a "user-facing" record of which files this object contains
        new_obj.filepath = ";".join([obj.filepath for obj in obj_list])

        # Now we finally get to use the auto-check value we set earlier!
        new_obj._has_auto = auto_check
        if auto_check:
            new_obj._ac_read = np.concatenate(ac_read_list)
        else:
            new_obj._ac_read = None

        # Raw data aren't loaded with a concat operation, and metadata should
        # not be mucked (since there's no metadata in *_data yet).
        new_obj.unload_data(unload_vis=True, unload_raw=True, unload_auto=True)
        new_obj._data_mucked = False

        # This value is the forward gain of the antenna (in units of Jy/K).
        # Default is 130.0, which is the estimated value for SMA.
        new_obj.jypk = 130.0

        # _update_filter will assign all of the *_filter attributes, as well as the
        # user-facing *_data attributes on call, in addition to the various *hid_dict's
        # that map ID number to array index position. Note that we can do this slightly
        # quicker here if we wanted to, but it's nice not to have to duplicate a bunch
        # of code further on down.
        new_obj._update_filter()

        # And viola! New object ready for the user to interface with!
        return new_obj

    @staticmethod
    def _parse_select_compare(select_field, select_comp, select_val, data_arr):
        """
        Parse a select command into a set of boolean masks.

        This is an internal helper function built as part of the low-level API, and not
        meant to be used by general users. This method will produce a masking screen
        based on the arguments provided to determine which data should be selected,
        and is called by the method `MirParser.select`.

        Parameters
        ----------
        select_field : str
            Field in the `data_arr` to use in evaluating whether to select data.
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
        data_arr : ndarray
            Structured array to evaluate, which should have the field `select_field`
            within it. Must simply contain named fields.

        Returns
        -------
        data_mask : ndarray of bool
            Boolean array marking whether `select_field` in `data_arr` meets the
            condition set by `select_comp` and `select_val`.

        Raises
        ------
        ValueError
            If `select_comp` is not one of the permitted strings, or if `select_field`
            is not one of the fields within `data_arr`.
        """
        # Create a simple dict to match operation keywords to a function.
        op_dict = {
            "eq": lambda val, comp: np.isin(val, comp),
            "ne": lambda val, comp: np.isin(val, comp, invert=True),
            "lt": np.less,
            "le": np.less_equal,
            "gt": np.greater,
            "ge": np.greater_equal,
            "btw": lambda val, lims: ((val >= lims[0]) and (val <= lims[1])),
            "out": lambda val, lims: ((val < lims[0]) or (val > lims[1])),
        }

        # Make sure the inputs look valid
        if select_comp not in op_dict.keys():
            raise ValueError(
                "select_comp must be one of: %s" % ", ".join(op_dict.keys())
            )
        if select_field not in data_arr.dtype.names:
            raise ValueError(
                "select_field %s not found in structured array." % select_field
            )

        # Evaluate data_arr now
        return op_dict["select_comp"](data_arr[select_field], select_val)

    def _parse_select(
        self, select_field, select_comp, select_val, use_in, use_bl, use_sp
    ):
        """
        Parse a select command into a set of boolean masks.

        This is an internal helper function built as part of the low-level API, and not
        meant to be used by general users. This method will produce a masking screen
        based on the arguments provided to determine which data should be selected,
        and is called by the method `MirParser.select`.

        Parameters
        ----------
        select_field : str
            Field in the MirParser metadata to use in evaluating whether to select
            data. This must match one of the dtype fields given in the the attributes
            `in_data`, `bl_data`, `sp_data`, `we_data`, `eng_data`, or the keys of
            `codes_dict`.
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
        use_in : ndarray of bool
            Boolean array of the same length as the attribute `_in_read`, which marks
            which data is "good" (i.e., should be selected). Note that this array
            is modified in-situ (rather than being returned).
        use_bl : ndarray of bool
            Boolean array of the same length as the attribute `_bl_read`, which marks
            which data is "good" (i.e., should be selected). Note that this array
            is modified in-situ (rather than being returned).
        use_sp : ndarray of bool
            Boolean array of the same length as the attribute `_sp_read`, which marks
            which data is "good" (i.e., should be selected). Note that this array
            is modified in-situ (rather than being returned).

        Raises
        ------
        ValueError
            When `select_field` matches a key in the attribute `codes_dict`, if either
            `select_comp` is anything but "eq" or "ne", or `select_val` is not either
            a string or sequence of strings. Also if `select_field` matches not fields
            in any of the dtypes for the attributes `in_data`, `bl_data`, `sp_data`,
            `eng_data`, or `we_data`.
        NotImplementedError
            When `select_field` matches a field in `we_data` (cannot currently select
            on this field due to the fact that it contains both per-antenna and
            per-integration data inter-mingled).
        """
        # We are going to convert select_val to an ndarray here, because it can handle
        # all of the accepted data types, and provides some nice convenience funcs.
        select_val = np.array(select_val)

        # We want select_val to be potentially iterable, so if it's a singleton
        # then take care of this now.
        select_val.shape += (1,) if (select_val.shape == 0) else ()

        if select_field in self.codes_dict.keys():
            if select_comp not in ["eq", "ne"]:
                raise ValueError(
                    'select_comp must be either "eq" or "ne" if matching a field'
                    "found in the attribute codes_dict."
                )

            # Create a temporary dict to map strings to index codes (i.e, icodes)
            temp_codes = {
                code_str: icode
                for icode, (code_str, _) in self.codes_dict[select_field].items()
            }

            # Create a dummy array to plug our icodes into
            temp_vals = []

            for item in select_val:
                if not np.isinstance(item, str):
                    raise ValueError(
                        "If select_field matches a key in codes_dict, then select_val "
                        "must either be a string or a sequence of strings."
                    )
                try:
                    # Find which icode matches the string that's been provided.
                    temp_vals.append(temp_codes[item])
                except KeyError:
                    # If we don't find any enties, that's fine, just skip this one.
                    pass

            # Now that we've converted all of the strings into a list of codes, convert
            # select_val back into an ndarray so that it behaves like we expect.
            select_val = np.array(temp_vals)

            # All of the fields connected to codes_dict just have an "i" up front.
            select_field = "i" + select_field

        elif select_field == "ant":
            # Ant is a funny (and annoying) keyword, because it can refer to either
            # ant1 or ant2, and we want to capture the case for both. Rather than
            # handing this with a special call to _parse_select_compare, we can just
            # change the select field to antennaNumber, which will match what is in
            # eng_data (which is already handled correctly).
            select_field = "antennaNumber"
        elif select_field in ["ant1", "ant2"]:
            # Final special case -- these technically have an "i" in front of then,
            # for iant1 and iant2, which _usually_ means its supposed to match an
            # entry in codes_read, but in this case is not. In any case, just
            # manually plug in the i here.
            select_field = "i" + select_field

        # Now is the time to get to the business of actually figuring out which data
        # we need to grab. The exact action will depend on which data structure
        # we need to select on.
        if select_field in self.in_data.dtype.names:
            # If this is a field in in_data, then we just need to update use_in
            use_in[self._in_filter] = self._parse_select_compare(
                select_field, select_comp, select_val, self.in_data,
            )
        elif select_field in self.bl_data.dtype.names:
            # Same story for bl_data and use_bl
            use_bl[self._bl_filter] = self._parse_select_compare(
                select_field, select_comp, select_val, self.bl_data,
            )
        elif select_field in self.sp_data.dtype.names:
            # And ditto for sp_data and use_sp
            use_sp[self._sp_filter] = self._parse_select_compare(
                select_field, select_comp, select_val, self.sp_data,
            )
        elif select_field in self.eng_data.dtype.names:
            # We have to handle the engineering data a little differently, because most
            # of the other metadata is per-baseline, but the eng data is all recorded
            # per antenna. So first up, we need to check and see which data is about
            # to be discarded by our selection
            data_mask = self._parse_select_compare(
                select_field, select_comp, select_val, self.eng_data,
            )

            # Need to run through the data tuple-by-tuple to see if a given ant-time
            # pairing is good or not. We can either check if a given tuple is in the
            # "allowed" set, or in the "disallowed" set -- pick which based on which
            # results in the shortest list of tuples we need to compare to.
            flip_mask = np.mean(data_mask) > 0.5
            if flip_mask:
                data_mask = ~data_mask

            # Create the list of allowed/disallowed tuples to check against
            check_items = [
                (inhid, ant)
                for inhid, ant in zip(
                    self.eng_data["inhid"][data_mask],
                    self.eng_data["antennaNumber"][data_mask],
                )
            ]

            # Finally, evaluate entry-by-entry that each baseline is "allowed" or
            # "disallowed" based on the antennas in the baseline pairing and the
            # value in the integration index header number.
            use_bl[self._bl_filter] = [
                ((inhid, ant1) in check_items or (inhid, ant2) in check_items)
                for inhid, ant1, ant2 in zip(
                    self.bl_data["inhid"], self.bl_data["iant1"], self.bl_data["iant2"],
                )
            ]

            # We want use_bl to return a boolean mask where True means that the data
            # are good, so if we "flipped" the mask earlier, we have the inverse of
            # that -- fix this now.
            if flip_mask:
                use_bl[self._bl_filter] = ~use_bl[self._bl_filter]
        elif select_field in self.we_data.dtype.names:
            # We currently are not parsing on the weather data, since it contains
            # both array-wide and per-antenna information. Pass on this for now.
            raise NotImplementedError(
                "Selecting based on we_read data not yet supported."
            )
        else:
            # If we didn't recognize the field in any dtype, throw an error now.
            raise ValueError(
                "Field name %s not recognized in any data structure." % select_field
            )

    def select(
        self,
        select_field=None,
        select_comp=None,
        select_val=None,
        use_in=None,
        use_bl=None,
        use_sp=None,
        update_data=None,
        reset=False,
    ):
        """
        Select a subset of data inside a MIR-formated file.

        This routine allows for one to select a subset of data within a MIR file, base
        on various metadata. The select command is designed to be _reasonably_ flexible,
        although is limited to running a single evaluation at a time. Users should be
        aware that the command is case sensitive, and uses "strict" agreement (i.e.,
        it does not mimic the behavior is `np.isclose`) with metadata values.

        The select command will automatically translate between the `codes_read` file
        and the various indexing keys, e.g., it will convert an allowed value for
        "source" (found in`codes_read`) into allowed values for "isource" (found in
        `in_read`). Multiple calls to select work by effectingly "AND"-ing the flags,
        e.g., running select for Antenna 1 and then Antenna 2 will result in only the
        1-2 baseline to appear.

        Parameters
        ----------
        select_field : str
            Field in the MirParser metadata to use in evaluating whether to select
            data. This must match one of the dtype fields given in the the attributes
            `in_data`, `bl_data`, `sp_data`, `we_data`, `eng_data`, or the keys of
            `codes_dict`.
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
            Note that this argument is ignored if resetting flags or supplying
            arguments to `use_in`, `use_bl`, or `use_sp`.
        select_val : str or number, or list of str or number.
            Value(s) to compare data in `select_field` against. If `select_comp` is
            "lt", "le", "gt", "ge", then this must be either a single number
            or string. If `select_comp` is "btw" or "out", then this must be a list
            of length 2. If `select_comp` is "eq" or "ne", then this can be either a
            single value or a sequence of values.
        use_in : slice-like
            Specify which elements of the attribute `in_data` to keep by supplying
            either a simple slice, or an array-like of bools or ints that can be used
            as a complex slice (i.e., the argument can be used as an index to get
            specific entries in `in_data` array). Good for performing complex selections
            on a per-integration basis.
        use_bl : slice-like
            Specify which elements of the attribute `bl_data` to keep by supplying
            either a simple slice, or an array-like of bools or ints that can be used
            as a complex slice (i.e., the argument can be used as an index to get
            specific entries in `bl_data` array). Good for performing complex selections
            on a per-baseline basis.
        use_sp : slice-like
            Specify which elements of the attribute `sp_data` to keep by supplying
            either a simple slice, or an array-like of bools or ints that can be used
            as a complex slice (i.e., the argument can be used as an index to get
            specific entries in `sp_data` array). Good for performing complex selections
            on a per-spectral record basis.
        update_data : bool
            Whether or not to update the visibility values (as recorded in the
            attributes `vis_data` and `raw_data`). If set to True, it will force data
            to be loaded from disk, based on what had been previously loaded. If False,
            it will unload those attributes. The default is to do nothing if data are
            not loaded, otherwise to downselect from the existing data in the object
            if all records are present (and otherwise unload the data).
        reset : bool
            If set to true, undoes any previous filters, so that all records are once
            again visible. Note that this will also unload the data. Default is False.

        Raises
        ------
        UserWarning
            If an argument is supplied for `select_field`, `select_comp`, or
            `select_val` of `reset=True`, or `use_in`, `use_bl`, or `use_sp` are set.
        ValueError
            If `select_field` is not a string, if `select_comp` is not one of the
            permitted values, or if `select_val` does not match the expected size
            given the argument supplied to `select_comp`.
        IndexError
            If the arguments supplied to `use_in`, `use_bl`, or `use_sp` are not able
            to correctly index the attributes `in_data`, `bl_data`, or `sp_data`,
            respectively.
        """
        # Make sure that the input arguments look alright before proceeding to
        # the heavy lifting of select.
        if select_field is None or select_comp is None or select_val is None:
            # Make sure the arguments below are either all filled or none are filled
            select_args = [select_field, select_comp, select_val]
            select_names = ["select_field", "select_comp", "select_val"]
            for arg, name in zip(select_args, select_names):
                if arg is not None:
                    raise ValueError(
                        "%s must be set if seeting other selection arguments based "
                        "on field names in the data." % name
                    )
        else:
            if reset:
                # If doing a reset, then warn that the other arguments are going to
                # get ignored.
                raise warnings.warn(
                    "Resetting data selection, all other arguments are ignored."
                )
            elif not (use_in is None and use_bl is None and use_sp is None):
                # Same case for the use_in, use_bl, and use_sp arguments -- we ignore
                # the "normal" selection commands if these are provided.
                raise warnings.warn(
                    "Selecting data using use_in, use_bl and/or use_sp; "
                    "all other arguments are ignored."
                )
            elif not isinstance(select_field, str):
                # Make sure select_field is a string (since all dtype fields are one)
                raise ValueError("select_field must be a string.")
            elif select_comp not in ["eq", "ne", "lt", "le", "gt", "ge", "btw", "out"]:
                # Make sure we can interpret select_comp
                raise ValueError(
                    'select_comp must be one of "eq", "ne", "lt", '
                    '"le", "gt", "ge", "btw", or "out".'
                )
            elif select_comp in ["lt", "le", "gt", "ge"]:
                # If one of the less/greater than (or equal) select comparisons, make
                # sure that there is only a single number.
                if np.array(select_val).size != 1 or not isinstance(
                    np.array(select_val)[0], np.number
                ):
                    raise ValueError(
                        "select_val must be a single number if select_comp "
                        'is either "lt", "le", "gt", or "ge".'
                    )
            elif select_comp in ["btw", "out"]:
                # Make sure is using between or outside, that we have two numbers
                if (len(select_val) != 2) or not isinstance(
                    np.array(select_val)[0], np.number
                ):
                    raise ValueError(
                        'If select_comp is "btw" or "out", select_val must be a '
                        "sequence if length two."
                    )
                # If the range is flipped, fix that now.
                if select_val[0] > select_val[1]:
                    select_val = [select_val[1], select_val[0]]

        # If all we need to do is reset, then do that now and bail.
        if reset:
            self._update_filter(update_data=False)
            return

        # If directly supplying slices upon which to select the data, deal with
        # those now. Note that these are slices on *_data instead of *_read
        # (the latter of which contains all of the metadata, not just that selected).
        if not ((use_in is None) and (use_bl is None) and (use_sp is None)):
            # We need to pass to update_filter a set of boolean masks of the same
            # length as the *_read attributes, so we need to convert use_in, use_bl,
            # and use_sp (which don't need to be bool arrays) into the right form.
            # We use this dict just to make bookkeeping simpler.
            filt_dict = {
                "use_in": (use_in, self._in_filter),
                "use_bl": (use_bl, self._bl_filter),
                "use_sp": (use_sp, self._sp_filter),
            }
            try:
                # Note arg_filter come from args, obj_filter come from MirParser attrs
                for key, (arg_filter, obj_filter) in filt_dict.items():
                    if arg_filter is None:
                        use_filt = None
                    else:
                        temp_filt = np.zeros(np.sum(obj_filter), dtype=bool)
                        temp_filt[arg_filter] = True
                        use_filt = obj_filter.copy()
                        use_filt[use_filt] = temp_filt
                    filt_dict[key] = use_filt
            except IndexError:
                raise IndexError(
                    "use_in, use_bl, and use_sp must be set such that they can be "
                    "used to index in_data, bl_data, and sp_data, respectively."
                )

            # Now that we have made our masks, update the data accordingly
            self._update_filter(
                use_in=filt_dict["use_in"],
                use_bl=filt_dict["use_bl"],
                use_sp=filt_dict["use_sp"],
                update_data=update_data,
            )
            return

        # If select_field is None at this point, then we have a great big no-op.
        # Just let the user know before we exit.
        if select_field is None:
            warnings.warn(
                "No arguments supplied to select_field to filter on. Returning "
                "the MirParser object unchanged."
            )
            return

        # If we are at this point, use_* arguments are all none, so fill them in based
        # on what the filter attributes say they should be.
        use_in = self._in_filter.copy()
        use_bl = self._bl_filter.copy()
        use_sp = self._sp_filter.copy()

        # Note that the call to _parse_select here will modify use_in, use_bl, and
        # use_sp in situ.
        self._parse_select(
            select_field, select_comp, select_val, use_in, use_bl, use_sp
        )

        # Now that we've screened the data that we want, update the object appropriately
        self._update_filter(
            use_in=use_in, use_bl=use_bl, use_sp=use_sp, update_data=update_data,
        )

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
        sp_bl_map = [self._blhid_dict[blhid] for blhid in self.sp_data["blhid"]]

        # COMPASS stores its solns in a multi-dimensional array that needs to be
        # split apart in order to match for MirParser format. We can match each sphid
        # to a particular paring of antennas and polarizations/receivers, sideband,
        # and spectral chunk, so we use the dict below to map that sequence to a
        # particular sphid, for later use.
        sphid_dict = {}
        for sphid, inhid, ant1, rx1, ant2, rx2, sb, chunk in zip(
            self.sp_data["sphid"],
            self.sp_data["inhid"],
            self.bl_data["iant1"][sp_bl_map],
            self.bl_data["ant1rx"][sp_bl_map],
            self.bl_data["iant2"][sp_bl_map],
            self.bl_data["ant2rx"][sp_bl_map],
            self.bl_data["isb"][sp_bl_map],
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
                for jdx, rx, sb, chunk in enumerate(zip(rx_arr, sb_arr, chunk_arr)):
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
            for idx, rx1, rx2, sb, chunk in enumerate(
                zip(rx1_arr, rx2_arr, sb_arr, chunk_arr)
            ):
                for jdx, ant1, ant2 in enumerate(zip(ant1_arr, ant2_arr)):
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
                for jdx, rx1, rx2, sb, chunk in enumerate(
                    zip(rx1_arr, rx2_arr, sb_arr, chunk_arr)
                ):
                    for kdx, ant1, ant2 in enumerate(zip(ant1_arr, ant2_arr)):
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
        if not self._vis_data_loaded:
            raise ValueError(
                "Visibility data must be loaded in order to apply COMPASS solns. Run "
                "`load_data(load_vis=True)` to fix this issue."
            )

        # Before we do anything else, we want to be able to map certain entires that
        # are per-blhid to be per-sphid.
        sp_bl_map = [self._blhid_dict[blhid] for blhid in self.sp_data["blhid"]]

        # Now grab all of the metadata we want for processing the spectral records
        sphid_arr = self.sp_data["sphid"]  # Spectral window header ID
        ant1_arr = self.bl_data["iant1"][sp_bl_map]  # Ant 1 Number
        rx1_arr = self.bl_data["ant1rx"][sp_bl_map]  # Pol for Ant 1 | 0 : X/L , 1: Y/R
        ant2_arr = self.bl_data["iant2"][sp_bl_map]  # Ant 2 Number
        rx2_arr = self.bl_data["ant2rx"][sp_bl_map]  # Pol for Ant 2 | 0 : X/L , 1: Y/R
        chunk_arr = self.bl_data["corrchunk"]  # Correlator window number
        sb_arr = self.bl_data["isb"][sp_bl_map]  # Sidebad ID | 0 : LSB, 1 : USB

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
                        ant1soln = bp_compass[(ant1, rx1, sb, chunk)]["cal_data"]
                        ant1flags = bp_compass[(ant1, rx1, sb, chunk)]["cal_flags"]
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
                    self.vis_data[sphid]["vis_data"] *= cal_soln["cal_data"]
                    self.vis_data[sphid]["vis_flags"] += cal_soln["cal_flags"]

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
                    self.vis_data[sphid]["vis_flags"] += np.unpackbits(
                        sphid_flags[sphid], bitorder="little"
                    )
                except KeyError:
                    # If no key is found, then we want to try and use the "broader"
                    # flags to mask out the data that's associated with the given
                    # antenna-receiver combination (for that sideband and spec window).
                    # Note that if we do not have an entry here, something is amiss.
                    self.vis_data[sphid]["vis_flags"] += np.unpackbits(
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
                    )

    def redoppler_data(self):
        """
        Re-doppler the data.

        Note that this function may be moved out into utils module once UVData objects
        are capable of carrying Doppler tracking-related information. Also, this
        function is stubbed out for now awaiting an upstream fix to the MIR data
        structure.
        """
        # TODO redoppler_data: Make this work
        raise NotImplementedError("redoppler_data has not yet been implemented.")
