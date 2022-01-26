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
        """Iterate over all MirParser attributes."""
        attribute_list = [
            a
            for a in dir(self)
            if not a.startswith("__")
            and not callable(getattr(self, a))
            and not (metadata_only and a in ["vis_data", "raw_data", "auto_data"])
        ]

        for attribute in attribute_list:
            yield attribute

    def copy(self, metadata_only=False):
        """
        Make and return a copy of the MirParser object.

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
            nch_arr = sp_data["nch"].astype(np.int64)
            new_dataoff = np.zeros_like(sp_data["dataoff"])
            for inhid in np.unique(sp_data["inhid"]):
                data_mask = sp_data["inhid"] == inhid
                nch_subarr = nch_arr[data_mask]
                new_dataoff[data_mask] = 2 * np.cumsum(1 + (2 * nch_subarr)) - (
                    1 + (2 * nch_subarr)
                )

            # With the new values calculated, put the updated values back into sp_data
            sp_data["dataoff"] = new_dataoff

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
    def scan_int_start(filepath, allowed_inhid=None):
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
                    int_vals["nbyt"],
                    data_offset,
                )
                last_offset = int_vals["nbyt"]
                data_offset += last_offset + 8

        return int_start_dict

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

        Returns
        -------
        int_data_dict : dict
            Dictionary of the data, where the keys are inhid and the values are
            the 'raw' block of values recorded in "sch_read" for that inhid.
        """
        full_filepath = os.path.join(filepath, "sch_read")
        int_data_dict = {}
        int_dtype_dict = {}
        size_list = np.unique(
            [int_start_dict[ind_key][0] for ind_key in int_start_dict.keys()]
        )

        for int_size in size_list:
            int_dtype_dict[int_size] = np.dtype(
                [
                    ("inhid", np.int32),
                    ("nbyt", np.int32),
                    ("packdata", np.int16, int_size // 2),
                ]
            ).newbyteorder("little")

        inhid_list = []
        last_offset = last_size = num_vals = del_offset = 0
        key_list = sorted(int_start_dict.keys())

        # We add an extra key here, None, which cannot match any of the values in
        # int_start_dict (since inhid is type int). This basically tricks the loop
        # below into spitting out the last integration
        key_list.append(None)

        read_list = []

        for ind_key in key_list:
            if ind_key is None:
                int_size = int_start = 0
            else:
                (int_size, int_start) = int_start_dict[ind_key]
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
                del_offset = int_start - (last_offset + (num_vals * last_size))
                last_offset = int_start
                last_size = int_size
                num_vals = 0
                inhid_list = []
            num_vals += 1
            inhid_list.append(ind_key)

        if use_mmap:
            for read_dict in read_list:
                int_data_dict.update(
                    zip(
                        read_dict["inhid_list"],
                        np.memmap(
                            filename=full_filepath,
                            dtype=read_dict["int_dtype_dict"],
                            mode="r",
                            offset=read_dict["del_offset"],
                            shape=(read_dict["num_vals"],),
                        ),
                    )
                )
        else:
            with open(full_filepath, "rb") as visibilities_file:
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
        """Write packdata to disk."""
        inhid_arr = sp_data["inhid"]
        sphid_arr = sp_data["sphid"]
        nch_arr = sp_data["nch"].astype(np.int64)

        unique_inhid = np.unique(inhid_arr)
        int_start_dict = {}
        offset_val = 0
        for inhid in unique_inhid:
            data_mask = inhid_arr == inhid
            int_size = (2 * np.sum(nch_arr[data_mask]) + np.sum(data_mask)) * 2
            int_start_dict[inhid] = (int_size, offset_val)
            offset_val += int_size + 8

        int_dtype_dict = {}
        size_list = np.unique(
            [int_start_dict[ind_key][0] for ind_key in int_start_dict.keys()]
        )

        for int_size in size_list:
            int_dtype_dict[int_size] = np.dtype(
                [
                    ("inhid", np.int32),
                    ("nbyt", np.int32),
                    ("packdata", np.int16, int_size // 2),
                ]
            ).newbyteorder("little")

        int_data_dict = {}
        for inhid, (int_size, _) in int_start_dict.items():
            packdata = np.zeros((), dtype=int_dtype_dict[int_size])

            data_mask = inhid_arr == inhid
            nch_subarr = nch_arr[data_mask]
            sphid_subarr = sphid_arr[data_mask]
            dataoff_subarr = np.cumsum((2 * nch_subarr) + 1) - ((2 * nch_subarr) + 1)

            raw_subdata = {sphid: raw_data[sphid] for sphid in sphid_subarr}

            slice_list = [
                slice(idx, jdx)
                for idx, jdx in zip(
                    dataoff_subarr + 1, dataoff_subarr + 1 + (2 * nch_subarr)
                )
            ]

            if np.max([ch_slice.stop for ch_slice in slice_list]) > (int_size // 2):
                raise ValueError(
                    "Mismatch between values of dataoff in sp_data and the size "
                    "of the spectral records in the data."
                )

            packdata["inhid"] = inhid
            packdata["nbyt"] = int_size

            for idx, ch_slice, sp_raw in zip(
                dataoff_subarr, slice_list, raw_subdata.values()
            ):
                packdata["packdata"][idx] = sp_raw["scale_fac"]
                packdata["packdata"][ch_slice] = sp_raw["raw_data"]

            int_data_dict[inhid] = packdata

        return int_data_dict

    @staticmethod
    def write_packdata(filepath, int_data_dict, append_data=False):
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

        with open(
            os.path.join(filepath, "sch_read"), "ab+" if append_data else "wb+"
        ) as file:
            for item in int_data_dict.values():
                item.tofile(file)

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
        vis_dict = {
            sphid: {
                "vis_data": (
                    (np.float32(2) ** sp_raw["scale_fac"]) * sp_raw["raw_data"]
                ).view(dtype=np.complex64),
                "vis_flags": sp_raw["raw_data"][::2] == -32768,
            }
            for sphid, sp_raw in raw_dict.items()
        }

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
        scale_fac = np.frexp(
            [
                np.abs(sp_vis["vis_data"].view(dtype=np.float32)).max(initial=1e-45)
                for sp_vis in vis_dict.values()
            ]
        )[1].astype(np.int16) - np.int16(15)

        raw_dict = {
            sphid: {
                "scale_fac": sfac,
                "raw_data": np.where(
                    sp_vis["vis_flags"],
                    np.complex64(-32768 - 32768j),
                    sp_vis["vis_data"] * np.complex64(2) ** (-sfac),
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
            Array from "sp_read", returned by "read_sp_read".

        Returns
        -------
        vis_dict : list of ndarrays
            List of ndarrays (dtype=csingle/complex64), with indices equal to sphid
            and values being the floating-point visibilities for the spectrum
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

        # Gather the needed metadata
        inhid_arr = sp_data["inhid"]
        sphid_arr = sp_data["sphid"]
        nch_arr = sp_data["nch"].astype(np.int64)
        dataoff_arr = sp_data["dataoff"] // 2

        unique_inhid = np.unique(inhid_arr)

        int_data_dict = {}
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

        raw_dict = {}
        for inhid in unique_inhid:
            packdata = int_data_dict.pop(inhid)["packdata"]

            data_mask = inhid_arr == inhid
            dataoff_subarr = dataoff_arr[data_mask]
            nch_subarr = nch_arr[data_mask]
            sphid_subarr = sphid_arr[data_mask]

            start_idx = dataoff_subarr
            end_idx = start_idx + (nch_subarr * 2) + 1
            raw_list = [packdata[idx:jdx] for idx, jdx in zip(start_idx, end_idx)]

            raw_dict.update(
                {
                    shpid: {"scale_fac": raw_data[0], "raw_data": raw_data[1:]}
                    for shpid, raw_data in zip(sphid_subarr, raw_list)
                }
            )
            # Do the del here to break the reference to the "old" data so that
            # subsequent assignments don't cause issues for raw_dict.
            del packdata

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
            Fill this in later.
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

        winsel = np.array(winsel)
        # The current generation correlator always produces 2**14 == 16384 channels per
        # spectral window.
        # TODO: Allow this to be flexible if dealing w/ spectrally averaged data
        # (although it's only use currently is in its unaveraged formal for
        # normaliztion of the crosses)
        auto_data = {}
        for file, startdict in zip(filepath, int_start_dict):
            unique_inhid = np.unique(ac_data["inhid"])
            ac_mask = np.intersect1d(unique_inhid, list(startdict.keys()))

            dataoff_arr = ac_data["dataoff"][ac_mask]
            nvals_arr = ac_data["datasize"][ac_mask].astype(np.int64) // 4
            achid_arr = ac_data["achid"][ac_mask]
            with open(os.path.join(file, "autoCorrelations"), "rb") as auto_file:
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
        """Make a codes dict."""
        codes_dict = {}
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
            if len(codes_dict[key][0]) == 1 and (
                codes_dict[key][1] == [0] and codes_dict[key][2] == [1]
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
        """Makea codes_read array."""
        codes_tuples = []
        for key, value in codes_dict.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    codes_tuples.append((key, subkey, subval[0], subval[1]))
            else:
                codes_tuples.append((key, 0, value, 1))

        codes_read = np.array(codes_tuples, dtype=codes_dtype)

        return codes_read

    def _update_filter(self, use_in=None, use_bl=None, use_sp=None, update_data=True):
        """
        Update MirClass internal filters for the data.

        Expands the internal 'use_in', 'use_bl', and 'use_sp' arrays to
        construct filters for the individual structures/data
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

        self._in_filter = in_filter
        self._eng_filter = eng_filter
        self._bl_filter = bl_filter
        self._sp_filter = sp_filter
        self._we_filter = we_filter
        self._ac_filter = ac_filter

        self.in_data = self._in_read[in_filter]
        self.eng_data = self._eng_read[eng_filter]
        self.bl_data = self._bl_read[bl_filter]
        self.sp_data = self._sp_read[sp_filter]
        self.codes_data = self._codes_read.copy()
        self.codes_dict = self.make_codes_dict(self._codes_read)
        self.we_data = self._we_read[we_filter]
        self.antpos_data = self._antpos_read.copy()

        # Handle the autos specially, since they are not always scanned/loaded
        self.ac_data = self._ac_read[self._ac_filter] if self._has_auto else None

        # Craft some dictionaries so you know what list position matches
        # to each index entry. This helps avoid ordering issues.
        self._inhid_dict = {inhid: idx for idx, inhid in enumerate(in_inhid)}
        self._blhid_dict = {blhid: idx for idx, blhid in enumerate(bl_blhid)}
        self._sphid_dict = {sphid: idx for idx, sphid in enumerate(sp_sphid)}

        if update_data:
            self.load_data(
                load_vis=self._vis_data_loaded,
                load_raw=self._raw_data_loaded,
                load_auto=self._auto_data_loaded,
                apply_tsys=self._tsys_applied,
            )

    def load_data(
        self,
        load_vis=None,
        load_raw=None,
        load_auto=False,
        apply_tsys=True,
        use_mmap=True,
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
        """
        if (load_vis is None) and (load_raw is None):
            load_vis = True

        if load_raw is None:
            load_raw = not load_vis
        elif load_vis is None:
            load_vis = not load_raw

        if self._file_dict == {}:
            if load_raw:
                if self._raw_data_loaded:
                    warnings.warn(
                        "No file to load from, and raw data is already loaded, "
                        "skipping raw data load."
                    )
                else:
                    self.raw_data = self.convert_vis_to_raw(self.vis_data)
                    self._raw_data_loaded = True
            if load_vis:
                if self._vis_data_loaded:
                    warnings.warn(
                        "No file to load from, and vis data is already loaded, "
                        "skipping vis data load."
                    )
                else:
                    self.raw_data = self.convert_raw_to_vis(self.raw_data)
                    self._vis_data_loaded = True
                    if apply_tsys:
                        self._apply_tsys()
            return

        if load_vis or load_raw:
            vis_tuple = self.read_vis_data(
                list(self._file_dict.keys()),
                list(self._file_dict.values()),
                self.sp_data,
                return_vis=load_vis,
                return_raw=load_raw,
                use_mmap=use_mmap,
            )

        if load_vis and load_raw:
            self.raw_data, self.vis_data = vis_tuple
        elif load_vis:
            (self.vis_data,) = vis_tuple
        elif load_raw:
            (self.raw_data,) = vis_tuple

        if load_vis:
            self._vis_data_loaded = True
        if load_raw:
            self._raw_data_loaded = True

        if apply_tsys and load_vis:
            self._apply_tsys()

        if load_auto and self._has_auto:
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
        """Unload data from the MirParser object."""
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

    def _apply_tsys(self, jypk=130.0):
        """
        Apply Tsys calibration to the visibilities.

        SMA MIR data are recorded as correlation coefficients. This allows one to apply
        system temperature information to the data to get values in units of Jy. This
        method is not meant to be called by users, but is instead meant to be called
        by data read methods.

        Parameteres
        -----------
        jypk : float
            Forward gain of the antenna (in units of Jy/K), which is multiplied against
            the system temperatures in order to produce values in units of Jy
            (technically this is the SEFD, which when multiplied against correlator
            coefficients produces visibilities in units of Jy). Default is 130.0, which
            is the estiamted value for SMA.
        """
        if not self._vis_data_loaded:
            raise ValueError(
                "Must call load_data first before applying tsys normalization."
            )

        # Create a dictionary here to map antenna pair + integration time step with
        # a sqrt(tsys) value. Note that the last index here is the receiver number,
        # which techically has a different keyword under which the system temperatures
        # are stored.
        tsys_dict = {
            (idx, jdx, 0): tsys**0.5
            for idx, jdx, tsys in zip(
                self.eng_data["inhid"],
                self.eng_data["antennaNumber"],
                self.eng_data["tsys"],
            )
        }
        tsys_dict.update(
            {
                (idx, jdx, 1): tsys**0.5
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
            blhid: (2.0 * jypk)
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

        # Finally, multiply the individual spectral records by the SEFD values
        # that are in the dictionary.
        for sphid, blhid in zip(self.sp_data["sphid"], self.sp_data["blhid"]):
            self.vis_data[sphid]["vis_data"] *= normal_dict[blhid]

        self._tsys_applied = True

    def fromfile(
        self,
        filepath,
        has_auto=False,
        load_vis=False,
        load_raw=False,
        load_auto=False,
        metadata_only=False,
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
        # it faster to read in the data. Only need this if we want to read in more
        # than just the metadata.
        if not metadata_only:
            self._file_dict = {
                os.path.abspath(filepath): self.scan_int_start(
                    filepath, allowed_inhid=self._in_read["inhid"],
                )
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
        self.vis_data = self.raw_data = self.auto_data = None

        self._vis_data_loaded = False
        self._tsys_applied = False
        self._raw_data_loaded = False
        self._auto_data_loaded = False

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
        metadata_only=False,
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
                metadata_only=metadata_only,
            )

    def tofile(self, filepath, append_data=False, append_codes=False, load_data=False):
        """
        Write a MirParser object to disk in MIR format.

        Parameters
        ----------
        filepath : str
        append_data : bool
        append_codes : bool
        """
        # TODO: Add comments
        # TODO: Add docstring
        # If no directory exists, create one to write the data to
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # Check that the data are loaded
        if load_data:
            self.load_data(load_vis=False, load_raw=True, load_auto=False)

        # Start out by writing the metadata out to file
        self.write_in_data(filepath, self.in_data, append_data=append_data)
        self.write_eng_data(filepath, self.eng_data, append_data=append_data)
        self.write_bl_data(filepath, self.bl_data, append_data=append_data)
        self.write_sp_data(filepath, self.sp_data, append_data=append_data)
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
        elif self._raw_data_loaded:
            raw_dict = self.raw_data
        else:
            raw_dict = self.convert_vis_to_raw(self.vis_data)

        self.write_packdata(
            filepath,
            self.make_packdata(self.sp_data, raw_dict),
            append_data=append_data,
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
    def _rechunk_raw(raw_dict, chan_avg_arr, inplace=False):
        """
        Rechunk a raw visibility spectrum.

        Note this routine is not intended to be called by users, but instead is a
        low-level call from the `rechunk` method of MirParser to spectrally average
        data.

        Parameters
        ----------
        raw_dict : dict
            A dict containing raw visibility data, where the keys match to individual
            values of `sphid` in `sp_data`, with each value being its own dict, with
            keys "raw_data" (the raw visibility data, dtype=np.int16) and "scale_fac"
            (scale factor to multiply raw data by , dtype=np.int16).
        chan_avg_arr : sequence of int
            A list, array, or tuple of integers, specifying how many channels to
            average over within each spectral record.
        inplace : bool
            If True, entries in `raw_dict` will be updated with spectrally averaged
            data. If False (default), then the method will construct a new dict that
            will contain the spectrally averaged data.

        Returns
        -------
        new_raw_dict : dict
            A dict containing the spectrally averaged data, in the same format as
            that provided in `raw_dict`.

        """
        # If inplace, point our new dict to the old one, otherwise create
        # an ampty dict to plug values into.
        new_raw_dict = raw_dict if inplace else {}

        for chan_avg, (sphid, sp_raw) in zip(chan_avg_arr, raw_dict.items()):
            # If the number of channels to average is 1, then we just need to make
            # a deep copy of the old data and plug it in to the new dict.
            if chan_avg == 1:
                if not inplace:
                    new_raw_dict[sphid] = copy.deepcopy(sp_raw)
                continue

            # If we are _not_ skipping the spectral averaging, then it turns out to
            # be faster to convert the raw data to "regular" data, spectrally average
            # it, and then convert it back to the raw format. Note that we set up a
            # "dummy" dict here with an sphid of 0 to make it easy to retrieve that
            # entry after the sequence of calls.
            new_raw_dict[sphid] = MirParser.convert_vis_to_raw(
                MirParser.rechunk_vis(
                    MirParser.convert_raw_to_vis({0: sp_raw}),
                    [chan_avg],
                    inplace=False,
                )
            )[0]

        # Finally, return the dict containing the raw data.
        return new_raw_dict

    def rechunk(self, chan_avg):
        """Rechunk a MirParser object."""
        # TODO: Add comments
        # TODO: Add docstring
        # TODO: Flesh out this command, which should be used by users.

    @staticmethod
    def _combine_read_arr_check(arr1, arr2, index_name=None):
        """Do a thing."""
        # TODO: Add comments
        # TODO: Add docstring
        if arr1.dtype != arr2.dtype:
            raise ValueError("Both arrays must be of the same dtype.")

        # For a codes_read array, the indexing procedure is a bit funky,
        # so we handle this as a special case.
        if arr1.dtype == codes_dtype:
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
                    if not (arr1_dict[key] == arr2_dict[key]):
                        return False
                except KeyError:
                    # If the keys don't confict, then there's no potential clash, so
                    # just move on to the next entry.
                    pass
            return True

        if not isinstance(index_name, str):
            raise ValueError("index_name must be a string.")

        if index_name not in arr1.dtype.names:
            raise ValueError("index_name not a recognized field in either array.")

        _, idx1, idx2 = np.intersect1d(
            arr1[index_name], arr2[index_name], return_indices=True
        )

        return np.array_equal(arr1[idx1], arr2[idx2])

    @staticmethod
    def _combine_read_arr(
        arr1, arr2, index_name=None, return_indices=False, overwrite=False
    ):
        """Do a thing."""
        # TODO: Add comments
        # TODO: Add docstring
        if overwrite:
            # If we are overwriting, then make sure that the two arrays are of the
            # same dtype before continuing.
            if arr1.dtype != arr2.dtype:
                raise ValueError("Both arrays must be of the same dtype.")
        else:
            # If not overwriting, check and make sure
            if not MirParser._combine_read_arr_check(arr1, arr2, index_name=index_name):
                raise ValueError(
                    "Arrays have overlapping indicies with different data, "
                    "cannot combine the two safely."
                )

        # For a codes_read array, the indexing procedure is a bit funky,
        # so we handle this as a special case.
        if arr1.dtype == codes_dtype:
            arr1_set = {(item["v_name"], item["icode"], item["ncode"]) for item in arr1}

            idx2 = np.array(
                [
                    idx
                    for idx, item in enumerate(arr2)
                    if (item["v_name"], item["icode"], item["ncode"]) in arr1_set
                ]
            )
        else:
            if not isinstance(index_name, str):
                raise ValueError("index_name must be a string.")

            if index_name not in arr1.dtype.names:
                raise ValueError("index_name not a recognized field in either array.")

            _, idx1, idx2 = np.intersect1d(
                arr1[index_name], arr2[index_name], return_indices=True
            )

        arr_sel = np.isin(np.arange(len(arr1)), idx1, invert=True)
        result = np.concatenate(arr1[arr_sel], arr2)

        if return_indices:
            result = (result, arr_sel)

        return result

    def __add__(self, other_obj, overwrite=False, force=False, inplace=False):
        """Add two MirParser objects."""
        # TODO: Put in docstring.
        # TODO: Add a few more comments here.
        if not isinstance(other_obj, MirParser):
            raise ValueError(
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

        if self._has_auto != other_obj._has_auto:
            warnings.warn("")
        elif self._has_auto:
            comp_list.append("_ac_read")
            attr_index_dict["ac_read"] = "achid"

        # First thing -- check that everything here appears to belong to the same file
        force_list = []
        for item in comp_list:
            if not np.array_equal(getattr(self, item), getattr(self, item)):
                force_list.append(item)

        if force_list != []:
            force_list = ", ".join(force_list)
            if force:
                if not (self._vis_data_loaded or self._raw_data_loaded):
                    raise ValueError(
                        "Cannot combine objects with force=True when no vis or raw "
                        "data gave been loaded. Run the `load_data` method on both "
                        "objects (with the same arguments) to clear this error."
                    )
                else:
                    warnings.warn(
                        "Objects here do not appear to be from the same file, but "
                        "proceeding ahead since force=True (%s clashes)." % force_list
                    )
            else:
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

        if new_obj.auto_data is not None:
            new_obj.auto_data.update(other_obj.auto_data)

        # Finally, we need to do a special bit of handling if we "forced" the two
        # objects together. If we did, then we need to update the core attributes
        # so that the *data and *_read arrays all agree.
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
        """Add two MirParser objects in place."""
        self.__add__(other_obj, overwrite=overwrite, force=force, inplace=True)

    @staticmethod
    def concat(obj_list, force=True):
        """Concat multiple MirParser objects together."""
        comp_list = [
            "_in_read",
            "_eng_read",
            "_bl_read",
            "_we_read",
            "_codes_read",
            "_antpos_read",
            "_file_dict",
        ]

        auto_check = obj_list[0]._has_auto
        raise_warning = True
        for idx, obj1 in enumerate(obj_list):
            if not isinstance(obj1, MirParser):
                raise ValueError("Can only concat MirParser objects.")
            if obj1._has_auto != auto_check:
                raise ValueError(
                    "Cannot combine objects both with and without auto-correlation "
                    "data. When reading data from file, must keep _has_auto set "
                    "consistently between file reads."
                )
            if obj1._file_dict == {}:
                raise ValueError(
                    "Cannot concat objects without an associated file (this is caused "
                    "by adding objects together with force=True)."
                )
            for obj2 in obj_list[:idx]:
                for item in comp_list:
                    if not np.array_equal(getattr(obj1, item), getattr(obj2, item)):
                        if not force:
                            raise ValueError(
                                "Two objects in the list appear to hold the same data. "
                                "Verify that you don't have two objects that are "
                                "loaded from the same file."
                            )
                        elif raise_warning:
                            warnings.warn(
                                "Objects may contain the same data, pushing forward "
                                "anyways since force=False."
                            )
                            raise_warning = False

    @staticmethod
    def _parse_select_compare(select_field, select_comp, select_val, data_arr):
        """Parse a select argument."""
        op_dict = {
            "eq": lambda val, comp: np.isin(val, comp),
            "ne": lambda val, comp: np.isin(val, comp, invert=True),
            "btw": lambda val, lims: ((val >= lims[0]) and (val <= lims[1])),
            "lt": np.less,
            "le": np.less_equal,
            "gt": np.greater,
            "ge": np.greater_equal,
        }

        if select_comp not in op_dict.keys():
            raise ValueError(
                "select_comp must be one of: %s" % ", ".join(op_dict.keys())
            )

        return op_dict["select_comp"](data_arr[select_field], select_val)

    def _parse_select(
        self, select_field, select_comp, select_val, use_in, use_bl, use_sp
    ):
        # TODO: Check comments here
        # TODO: Fill out case where codes dict needs to be parsed.
        # TODO: Put in doc string.
        if select_field in self.codes_dict.keys():
            pass
        elif select_field == "ant":
            pass

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
        sel_in=None,
        sel_bl=None,
        sel_sp=None,
        loaddata=None,
        reset=False,
    ):
        """Select data."""
        # TODO: Add docstring.
        # TODO: Add more comments.
        # If all we need to do is reset, then do that now
        if select_field is None or select_comp is None or select_val is None:
            select_args = [select_field, select_comp, select_val]
            select_names = ["select_field", "select_comp", "select_val"]
            for arg, name in zip(select_args, select_names):
                if arg is not None:
                    raise ValueError(
                        "%s must be set if seeting other selection arguments based "
                        "on field names in the data." % name
                    )
        else:
            if not isinstance(select_field, str):
                raise ValueError("select_field must be a string.")
            if select_comp not in ["eq", "ne", "lt", "le", "gt", "ge", "btw"]:
                raise ValueError(
                    "select_comp must be one of 'eq', 'ne', 'lt', "
                    "'le', 'gt', 'ge', or 'btw'."
                )
            if reset:
                raise warnings.warn(
                    "Resetting data selection, all other arguments are ignored."
                )

        if reset:
            self.unload_data()
            self._update_filter()
            return

        use_in = self._in_filter.copy()
        use_bl = self._bl_filter.copy()
        use_sp = self._sp_filter.copy()

        if not ((sel_in is None) and (sel_bl is None) and (sel_sp is None)):
            for item, jtem in zip([use_in, use_bl, use_sp], [sel_in, sel_bl, sel_sp]):
                if jtem is not None:
                    submask = np.zeros(sum(item), dtype=bool)
                    submask[jtem] = True
                    item[item] = submask
            self._update_filter(
                use_in=use_in, use_bl=use_bl, use_sp=use_sp, update_data=loaddata,
            )
            return

        if select_field is None:
            raise ValueError(
                "select_field cannot be set to None if reset=False and sel_in, sel_bl, "
                "and sel_sp are unset."
            )

        # Note that the call to _parse_select here will modify use_in, use_bl, and
        # use_sp in situ.
        self._parse_select(
            select_field, select_comp, select_val, use_in, use_bl, use_sp
        )

        # Now that we've screened the data that we want, update the object appropriately
        self._update_filter(
            use_in=use_in, use_bl=use_bl, use_sp=use_sp, update_data=loaddata,
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

        """
        # TODO: Add a docstring
        # TODO: Add comments here.
        # TODO: Make sure that this works.
        blhid_dict = {blhid: idx for idx, blhid in enumerate(self._bl_read["blhid"])}
        sp_bl_map = [blhid_dict[blhid] for blhid in self._sp_read["blhid"]]

        sphid_dict = {}
        for sphid, inhid, ant1, rx1, ant2, rx2, sb, chunk in zip(
            self._sp_read["sphid"],
            self._sp_read["inhid"],
            self._bl_read["iant1"][sp_bl_map],
            self._bl_read["ant1rx"][sp_bl_map],
            self._bl_read["iant2"][sp_bl_map],
            self._bl_read["ant2rx"][sp_bl_map],
            self._bl_read["isb"][sp_bl_map],
            self._sp_read["corrchunk"],
        ):
            sphid_dict[(inhid, ant1, rx1, ant2, rx2, sb, chunk)] = sphid

        compass_soln_dict = {}
        bandpass_gains = {}
        with h5py.File(filename, "r") as file:
            # First, pull out the bandpass solns
            ant_arr = np.array(file["antArr"][0])
            rx_arr = np.array(file["rx1Arr"][0])
            sb_arr = np.array(file["sbArr"][0])
            chunk_arr = np.array(file["winArr"][0])
            bp_arr = np.array(file["bandpassArr"])

            for idx, ant in enumerate(ant_arr):
                for jdx, rx, sb, chunk in enumerate(zip(rx_arr, sb_arr, chunk_arr)):
                    cal_data = bp_arr[idx, jdx]
                    cal_flags = (cal_data == 0.0) | ~np.isfinite(cal_data)
                    cal_data[cal_flags] = 1.0
                    bandpass_gains[(ant, rx, sb, chunk)] = {
                        "cal_data": cal_data,
                        "cal_flags": cal_flags,
                    }

            compass_soln_dict["bandpass_gains"] = bandpass_gains

            # Now, we can move on to flags. Note that COMPASS doesn't have access to
            # the integration header IDs, so we have to do a little bit of matching
            # based on the timestamp of the data in COMPASS vs MIR (via the MJD).
            mjd_compass = np.array(file["mjdArr"][0])
            mjd_mir = self._in_read["mjd"]
            inhid_arr = self._in_read["inhid"]

            index_dict = {}
            atol = 0.5 / 86400
            for idx, mjd in enumerate(mjd_compass):
                check = np.where(np.isclose(mjd, mjd_mir, atol=atol))[0]
                index_dict[idx] = None if (len(check) == 0) else inhid_arr[check[0]]

            flags_arr = np.array(file["flagArr"])
            wflags_arr = np.array(file["wideFlagArr"])
            ant1_arr = np.array(file["ant1Arr"][0])
            rx1_arr = np.array(file["rx1Arr"][0])
            ant2_arr = np.array(file["ant2Arr"][0])
            rx2_arr = np.array(file["rx2Arr"][0])
            sb_arr = np.array(file["sbArr"][0])
            chunk_arr = np.array(file["winArr"][0])

            wide_flags = {}
            for idx, rx1, rx2, sb, chunk in enumerate(
                zip(rx1_arr, rx2_arr, sb_arr, chunk_arr)
            ):
                for jdx, ant1, ant2 in enumerate(zip(ant1_arr, ant2_arr)):
                    wide_flags[(ant1, rx1, ant2, rx2, sb, chunk)] = wflags_arr[idx, jdx]

            compass_soln_dict["wide_flags"] = wide_flags

            # Now we
            sphid_flags = {}
            for idx, inhid in index_dict.items():
                if inhid is None:
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
        # TODO: Actually test that this works.

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
                try:
                    # If we have a flags entry for this sphid, then go ahead and apply
                    # them to the flags table for that spectral record.
                    self.vis_data[sphid]["vis_flags"] += sphid_flags[sphid]
                except KeyError:
                    # If no key is found, then we want to try and use the "broader"
                    # flags to mask out the data that's associated with the given
                    # antenna-receiver combination (for that sideband and spec window).
                    # Note that if we do not have an entry here, something is amiss.
                    self.vis_data[sphid]["vis_flags"] += wide_flags[
                        (
                            ant1_arr[idx],
                            rx1_arr[idx],
                            ant2_arr[idx],
                            rx2_arr[idx],
                            sb_arr[idx],
                            chunk_arr[idx],
                        )
                    ]

    def redoppler_data(self):
        """
        Re-doppler the data.

        Note that this function may be moved out into utils module once UVData objects
        are capable of carrying Doppler tracking-related information. Also, this
        function is stubbed out for now awaiting an upstream fix to the MIR data
        structure.
        """
        # TODO: Make this work
        raise NotImplementedError("redoppler_data has not yet been implemented.")
