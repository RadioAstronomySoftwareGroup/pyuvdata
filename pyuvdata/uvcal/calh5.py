# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2023 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading and writing calibration HDF5 files."""
from __future__ import annotations

import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .. import utils as uvutils
from ..uvdata.uvh5 import (
    _check_uvh5_dtype,
    _get_compression,
    _read_complex_astype,
    _write_complex_astype,
)
from .uvcal import UVCal, _future_array_shapes_warning, radian_tol

hdf5plugin_present = True
try:
    import hdf5plugin  # noqa: F401
except ImportError as error:
    hdf5plugin_present = False
    hdf5plugin_error = error


class FastCalH5Meta:
    """
    A fast read-only interface to CalH5 file metadata that makes some assumptions.

    This class is just a really thin wrapper over a CalH5 file that makes it easier
    to read in parts of the metadata at a time. This makes it much faster to perform
    small tasks where simple metadata is required, rather than reading in the whole
    header.

    All metadata is available as attributes, through ``__getattr__`` magic. Thus,
    accessing eg. ``obj.freq_array`` will go and get the frequencies directly from the
    file, and store them in memory.

    Anything that is read in is stored in memory so the second access is much faster.
    However, the memory can be released simply by deleting the attribute (it can be
    accessed again, and the data will be re-read).

    Parameters
    ----------
    filename : str or Path
        The filename to read from.

    Notes
    -----
    To check if a particular attribute is available, use ``hasattr(obj, attr)``.
    Many attributes will not show up dynamically in an interpreter, because they are
    gotten dynamically from the file.
    """

    _string_attrs = frozenset(
        {
            "history",
            "x_orientation",
            "telescope_name",
            "cal_type",
            "cal_style",
            "gain_convention",
            "diffuse_model",
            "gain_scale",
            "git_hash_cal",
            "git_origin_cal",
            "observer",
            "ref_antenna_name",
            "sky_catalog",
            "sky_field",
            "version",
        }
    )

    _defaults = {"x_orientation": None, "flex_spw": False}

    _int_attrs = frozenset(
        {
            "Ntimes",
            "Njones",
            "Nspws",
            "Nfreqs",
            "uvplane_reference_time",
            "Nphase",
            "Nants_data",
            "Nants_telescope",
            "Nsources",
        }
    )

    _bool_attrs = frozenset(("wide_band",))

    def __init__(self, path: str | Path | h5py.File | h5py.Group):
        self.__file = None

        if isinstance(path, h5py.File):
            self.path = Path(path.filename)
            self.__file = path
            self.__header = path["/Header"]
            self.__datagrp = path["/Data"]
        elif isinstance(path, h5py.Group):
            self.path = Path(path.file.filename)
            self.__file = path.file
            self.__header = path
            self.__datagrp = self.__file["/Data"]
        elif isinstance(path, (str, Path)):
            self.path = Path(path)

    def is_open(self) -> bool:
        """Whether the file is open."""
        return bool(self.__file)

    def __del__(self):
        """Close the file when the object is deleted."""
        if self.__file:
            self.__file.close()

    def __getstate__(self):
        """Get the state of the object."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in (
                "_FastCalH5Meta__file",
                "_FastCalH5Meta__header",
                "_FastCalH5Meta__datagrp",
                "header",
                "datagrp",
            )
        }

    def __setstate__(self, state):
        """Set the state of the object."""
        self.__dict__.update(state)
        self.__file = None

    def __eq__(self, other):
        """Check equality of two FastCalH5Meta objects."""
        if not isinstance(other, FastCalH5Meta):
            return False

        return self.path == other.path

    def __hash__(self):
        """Get a unique hash for the object."""
        return hash(self.path)

    def close(self):
        """Close the file."""
        self.__header = None
        self.__datagrp = None

        try:
            del self.header  # need to refresh these
        except AttributeError:
            pass

        try:
            del self.datagrp
        except AttributeError:
            pass

        if self.__file:
            self.__file.close()
        self.__file = None

    def open(self):  # noqa: A003
        """Open the file."""
        if not self.__file:
            self.__file = h5py.File(self.path, "r")
            self.__header = self.__file["/Header"]
            self.__datagrp = self.__file["/Data"]

    @cached_property
    def header(self) -> h5py.Group:
        """Get the header group."""
        if not self.__file:
            self.open()
        return self.__header

    @cached_property
    def datagrp(self) -> h5py.Group:
        """Get the header group."""
        if not self.__file:
            self.open()
        return self.__datagrp

    def get_transactional(self, item: str, cache: bool = True) -> Any:
        """Get an attribute from the metadata but close the file object afterwards.

        Using this method is safer than direct attribute access when dealing with
        many files.

        Parameters
        ----------
        item
            The attribute to get.
        cache
            Whether to cache the attribute in the object so that the next access is
            faster.
        """
        try:
            val = getattr(self, item)
        finally:
            self.close()

            if not cache:
                if item in self.__dict__:
                    del self.__dict__[item]

        return val

    def __getattr__(self, name: str) -> Any:
        """Get attribute directly from header group."""
        try:
            x = self.header[name][()]
            if name in self._string_attrs:
                x = bytes(x).decode("utf8")
            elif name in self._int_attrs:
                x = int(x)

            self.__dict__[name] = x
            return x
        except KeyError:
            try:
                return self._defaults[name]
            except KeyError as e:
                raise AttributeError(f"{name} not found in {self.path}") from e

    @cached_property
    def extra_keywords(self) -> dict:
        """The extra_keywords from the file."""
        header = self.header
        if "extra_keywords" not in header:
            return {}

        extra_keywords = {}
        for key in header["extra_keywords"].keys():
            if header["extra_keywords"][key].dtype.type in (np.string_, np.object_):
                extra_keywords[key] = bytes(header["extra_keywords"][key][()]).decode(
                    "utf8"
                )
            else:
                # special handling for empty datasets == python `None` type
                if header["extra_keywords"][key].shape is None:
                    extra_keywords[key] = None
                else:
                    extra_keywords[key] = header["extra_keywords"][key][()]
        return extra_keywords

    def check_lsts_against_times(self):
        """Check that LSTs consistent with the time_array and telescope location."""
        lsts = uvutils.get_lst_for_time(
            self.time_array, *self.telescope_location_lat_lon_alt_degrees
        )

        if not np.all(np.isclose(self.lsts, lsts, rtol=0, atol=radian_tol)):
            warnings.warn(
                f"LST values stored in {self.path} are not self-consistent "
                "with time_array and telescope location. Consider "
                "recomputing with utils.get_lst_for_time."
            )

    @cached_property
    def antenna_names(self) -> list[str]:
        """The antenna names in the file."""
        return [bytes(name).decode("utf8") for name in self.header["antenna_names"][:]]

    def has_key(self, antnum: int | None = None, jpol: str | int | None = None) -> bool:
        """Check if the file has a given antpair or antpair-pol key."""
        if antnum is not None:
            if antnum not in self.ant_array:
                return False
        if jpol is not None:
            if isinstance(jpol, (str, np.str_)):
                jpol = uvutils.jstr2num(jpol, x_orientation=self.x_orientation)
            if jpol not in self.jones_array:
                return False

        return True

    @cached_property
    def pols(self) -> list[str]:
        """The polarizations in the file, as standardized strings, eg. 'xx' or 'ee'."""
        return np.asarray(
            uvutils.jnum2str(self.jones_array, x_orientation=self.x_orientation)
        )

    @cached_property
    def telescope_location(self):
        """The telescope location in ECEF coordinates, in meters."""
        return uvutils.XYZ_from_LatLonAlt(*self.telescope_location_lat_lon_alt)

    @property
    def telescope_location_lat_lon_alt(self) -> tuple[float, float, float]:
        """The telescope location in latitude, longitude, and altitude, in degrees."""
        return self.latitude * np.pi / 180, self.longitude * np.pi / 180, self.altitude

    @property
    def telescope_location_lat_lon_alt_degrees(self) -> tuple[float, float, float]:
        """The telescope location in latitude, longitude, and altitude, in degrees."""
        return self.latitude, self.longitude, self.altitude

    def to_uvcal(self, check_lsts: bool = False) -> UVCal:
        """Convert the file to a UVData object.

        The object will be metadata-only.
        """
        uvc = CalH5()
        uvc.read_calh5(
            self,
            read_data=False,
            run_check_acceptability=check_lsts,
            use_future_array_shapes=True,
        )
        return uvc


class CalH5(UVCal):
    """
    A class for CalH5 file objects.

    This class defines an HDF5-specific subclass of UVCal for reading and
    writing CalH5 files. This class should not be interacted with directly,
    instead use the read_calh5 and write_calh5 methods on the UVCal class.
    """

    def _read_header(
        self,
        filename: str | Path | FastCalH5Meta,
        background_lsts: bool = True,
        run_check_acceptability: bool = True,
    ):
        """
        Read header information from a UVH5 file.

        This is an internal function called by the user-space methods.
        Properties of the UVData object are updated as the file is processed.

        Parameters
        ----------
        filename : string, path, FastCalH5Meta, h5py.File or h5py.Group
            A file name or path or a FastCalH5Meta or h5py File or Group object that
            contains the header information. Should be called "/Header" for CalH5 files
            conforming to spec.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check_acceptability : bool
            Option to check that LSTs match the times given the telescope_location.

        Returns
        -------
        None
        """
        if not isinstance(filename, FastCalH5Meta):
            obj = FastCalH5Meta(filename)
        else:
            obj = filename

        # First, get the things relevant for setting LSTs, so that can be run in the
        # background if desired.
        self.telescope_location_lat_lon_alt_degrees = (
            obj.telescope_location_lat_lon_alt_degrees
        )
        if "time_array" in obj.header:
            self.time_array = obj.time_array
        if "time_range" in obj.header:
            self.time_range = obj.time_range

        if "lst_array" in obj.header:
            self.lst_array = obj.header["lst_array"][:]
            proc = None
        elif "time_array" in obj.header:
            proc = self.set_lsts_from_time_array(background=background_lsts)

        if "lst_range" in obj.header:
            self.lst_range = obj.header["lst_range"][:]
            proc = None
        elif "time_range" in obj.header:
            proc = self.set_lsts_from_time_array(background=background_lsts)

        # Required parameters
        for attr in [
            "telescope_name",
            "history",
            "Nfreqs",
            "Njones",
            "Nspws",
            "Ntimes",
            "Nants_data",
            "Nants_telescope",
            "antenna_names",
            "antenna_numbers",
            "antenna_positions",
            "ant_array",
            "integration_time",
            "spw_array",
            "jones_array",
            "channel_width",
            "cal_style",
            "cal_type",
            "gain_convention",
            "wide_band",
            "x_orientation",
            "flex_spw_id_array",
        ]:
            try:
                setattr(self, attr, getattr(obj, attr))
            except AttributeError as e:
                raise KeyError(str(e)) from e

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        # Optional parameters
        for attr in [
            "Nsources",
            "baseline_range",
            "extra_keywords",
            "freq_array",
            "freq_range",
            "gain_scale",
            "git_hash_cal",
            "git_origin_cal",
            "observer",
            "ref_antenna_name",
            "sky_catalog",
            "sky_field",
        ]:
            try:
                setattr(self, attr, getattr(obj, attr))
            except AttributeError:
                pass

        if self.blt_order is not None:
            self._blt_order.form = (len(self.blt_order),)

        # set telescope params
        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        if run_check_acceptability:
            obj.check_lsts_against_times()

        self._set_future_array_shapes()

        if proc is not None:
            proc.join()

    def _get_data(
        self,
        dgrp,
        *,
        antenna_nums,
        antenna_names,
        frequencies,
        freq_chans,
        spws,
        times,
        time_range,
        lsts,
        lst_range,
        jones,
        gain_array_dtype,
    ):
        """
        Read the data-size arrays (gain/delay arrays, flags, qualities) from a file.

        This is an internal function to read just the calibration solutions, flags, and
        qualities from the CalH5 file. This is separated from full read so that
        header/metadata and data can be read independently. See the
        documentation of `read_calh5` for a full description of most of the
        descriptions of parameters. Below we only include a description of args
        unique to this function.

        Parameters
        ----------
        dgrp : h5py datagroup
            The HDF5 datagroup containing the datasets. Should be "/Data" for
            UVH5 files conforming to spec.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            This is raised if the data array read from the file is not a complex
            datatype (np.complex64 or np.complex128).
        """
        # check for bitshuffle data; bitshuffle filter number is 32008
        # TODO should we check for any other filters?
        if "32008" in dgrp["visdata"]._filters:
            if not hdf5plugin_present:  # pragma: no cover
                raise ImportError(
                    "hdf5plugin is not installed but is required to read this dataset"
                ) from hdf5plugin_error

        # figure out what data to read in
        (
            ant_inds,
            time_inds,
            freq_inds,
            jones_inds,
            history_update_string,
        ) = self._select_preprocess(
            antenna_nums,
            antenna_names,
            frequencies,
            freq_chans,
            times,
            time_range,
            lsts,
            lst_range,
            jones,
        )

        # figure out which axis is the most selective
        if ant_inds is not None:
            ant_frac = len(ant_inds) / float(self.Nblts)
        else:
            ant_frac = 1

        if time_inds is not None:
            time_frac = len(time_inds) / float(self.Nblts)
        else:
            time_frac = 1

        if freq_inds is not None:
            freq_frac = len(freq_inds) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if jones_inds is not None:
            jones_frac = len(jones_inds) / float(self.Npols)
        else:
            jones_frac = 1

        min_frac = np.min([ant_frac, time_frac, freq_frac, jones_frac])

        if self.cal_type == "gain":
            # get the fundamental datatype of the cal data; if integers, we need to
            # cast to floats
            caldata_dtype = dgrp["gains"].dtype
            if caldata_dtype not in ("complex64", "complex128"):
                _check_uvh5_dtype(caldata_dtype)
                if gain_array_dtype not in (np.complex64, np.complex128):
                    raise ValueError(
                        "gain_array_dtype must be np.complex64 or np.complex128"
                    )
                custom_dtype = True
            else:
                custom_dtype = False
        else:
            custom_dtype = False

        quality_present = False
        if "qualities" in dgrp:
            quality_present = True
        total_quality_present = False
        if "total_qualities" in dgrp:
            total_quality_present = True

        if min_frac == 1:
            # no select, read in all the data
            inds = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
            if self.cal_type == "gain":
                if custom_dtype:
                    self.gain_array = _read_complex_astype(
                        dgrp["gains"], inds, gain_array_dtype
                    )
                else:
                    self.gain_array = uvutils._index_dset(dgrp["gains"], inds)
            else:
                self.delay_array = uvutils._index_dset(dgrp["delays"], inds)
            self.flag_array = uvutils._index_dset(dgrp["flags"], inds)
            if quality_present:
                self.quality_array = uvutils._index_dset(dgrp["qualities"], inds)
            if total_quality_present:
                tq_inds = (np.s_[:], np.s_[:], np.s_[:])
                self.total_quality_array = uvutils._index_dset(
                    dgrp["total_qualities"], tq_inds
                )
        else:
            # do select operations on everything except data_array, flag_array
            # and nsample_array
            self._select_by_index(
                ant_inds, time_inds, freq_inds, jones_inds, history_update_string
            )

            # determine which axes can be sliced, rather than fancy indexed
            # max_nslice_frac of 0.1 yields slice speedup over fancy index for HERA data
            # See pyuvdata PR #805
            if ant_inds is not None:
                ant_slices, ant_sliceable = uvutils._convert_to_slices(
                    ant_inds, max_nslice_frac=0.1
                )
            else:
                ant_inds, ant_slices = np.s_[:], np.s_[:]
                ant_sliceable = True

            if time_inds is not None:
                time_slices, time_sliceable = uvutils._convert_to_slices(
                    time_inds, max_nslice_frac=0.1
                )
            else:
                time_inds, time_slices = np.s_[:], np.s_[:]
                time_sliceable = True

            if freq_inds is not None:
                freq_slices, freq_sliceable = uvutils._convert_to_slices(
                    freq_inds, max_nslice_frac=0.1
                )
            else:
                freq_inds, freq_slices = np.s_[:], np.s_[:]
                freq_sliceable = True

            if jones_inds is not None:
                jones_slices, jones_sliceable = uvutils._convert_to_slices(
                    jones_inds, max_nslice_frac=0.5
                )
            else:
                jones_inds, jones_slices = np.s_[:], np.s_[:]
                jones_sliceable = True

            # open references to datasets
            if self.cal_type == "gain":
                caldata_dset = dgrp["gains"]
            else:
                caldata_dset = dgrp["delays"]
            flags_dset = dgrp["flags"]
            if quality_present:
                qualities_dset = dgrp["qualities"]
            if total_quality_present:
                total_qualities_dset = dgrp["total_qualities"]

            # just read in the right portions of the data and flag arrays
            if ant_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [ant_inds, np.s_[:], np.s_[:], np.s_[:]]
                if ant_sliceable:
                    inds[0] = ant_slices

                inds = tuple(inds)
                # change ant_frac so no more selects are done
                ant_frac = 1

            elif time_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], time_inds, np.s_[:], np.s_[:]]
                if time_sliceable:
                    inds[1] = time_slices

                inds = tuple(inds)

                # change time_frac so no more selects are done
                time_frac = 1

            elif freq_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], np.s_[:], freq_inds, np.s_[:]]
                if freq_sliceable:
                    inds[1] = freq_slices

                inds = tuple(inds)

                # change freq_frac so no more selects are done
                freq_frac = 1

            else:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], np.s_[:], np.s_[:], jones_inds]
                if jones_sliceable:
                    inds[2] = jones_slices

                inds = tuple(inds)

                # change jones_frac so no more selects are done
                jones_frac = 1

            # index datasets
            if custom_dtype:
                cal_data = _read_complex_astype(caldata_dset, inds, gain_array_dtype)
            else:
                cal_data = uvutils._index_dset(caldata_dset, inds)
            flags = uvutils._index_dset(flags_dset, inds)
            if quality_present:
                qualities = uvutils._index_dset(qualities_dset, inds)
            if total_quality_present:
                total_qualities = uvutils._index_dset(total_qualities_dset, inds)
            # down select on other dimensions if necessary
            # use indices not slices here: generally not the bottleneck
            if ant_frac < 1:
                cal_data = cal_data[ant_inds]
                flags = flags[ant_inds]
                if quality_present:
                    qualities = qualities[ant_inds]
            if time_frac < 1:
                cal_data = cal_data[:, time_inds]
                flags = flags[:, time_inds]
                if quality_present:
                    qualities = qualities[:, time_inds]
                if total_quality_present:
                    total_qualities = total_qualities[time_inds]
            if freq_frac < 1:
                cal_data = cal_data[:, :, freq_inds]
                flags = flags[:, :, freq_inds]
                if quality_present:
                    qualities = qualities[:, :, freq_inds]
                if total_quality_present:
                    total_qualities = total_qualities[:, freq_inds]
            if jones_frac < 1:
                cal_data = cal_data[:, :, :, jones_inds]
                flags = flags[:, :, :, jones_inds]
                if quality_present:
                    qualities = qualities[:, :, :, jones_inds]
                if total_quality_present:
                    total_qualities = total_qualities[:, :, jones_inds]

            # save arrays in object
            if self.cal_type == "gain":
                self.gain_array = cal_data
            else:
                self.gain_array = cal_data
            self.flag_array = flags
            if quality_present:
                self.quality_array = qualities
            if total_quality_present:
                self.total_quality_array = total_qualities

        return

    def read_calh5(
        self,
        filename,
        *,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        frequencies=None,
        freq_chans=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        jones=None,
        read_data=True,
        gain_array_dtype=np.complex128,
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        use_future_array_shapes=False,
    ):
        """
        Read in data from a CalH5 file.

        Parameters
        ----------
        filename : str
             The UVH5 file to read from.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided. Ignored if read_data is False.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided. Ignored if read_data is False.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array. Ignored if
            read_data is False.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the
            object. Ignored if read_data is False.
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        jones : array_like of int, optional
            The jones polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
            Ignored if read_data is False.
        read_data : bool
            Read in the data-like arrays (gains/delays, flags, qualities). If set to
            False, only the metadata will be read in. Setting read_data to False
            results in a metadata only object.
        gain_array_dtype : numpy dtype
            Datatype to store the output gain_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128 (double-
            precision real and imaginary). Only used for gain type files and if the
            datatype of the gain data on-disk is not 'c8' or 'c16'.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If filename doesn't exist.
        ValueError
            If the data_array_dtype is not a complex dtype.
            If incompatible select keywords are set (e.g. `times` and `time_range`) or
            select keywords exclude all data or if keywords are set to the wrong type.

        """
        if isinstance(filename, FastCalH5Meta):
            meta = filename
            filename = str(meta.path)
        else:
            meta = FastCalH5Meta(filename)

        # update filename attribute
        basename = os.path.basename(filename)
        self.filename = [basename]
        self._filename.form = (1,)

        # open hdf5 file for reading
        self._read_header(
            meta,
            run_check_acceptability=run_check_acceptability,
            background_lsts=background_lsts,
        )

        if read_data:
            # Now read in the data
            self._get_data(
                meta.datagrp,
                antenna_nums,
                antenna_names,
                frequencies,
                freq_chans,
                times,
                time_range,
                lsts,
                lst_range,
                jones,
                gain_array_dtype,
            )

        if not use_future_array_shapes:
            warnings.warn(_future_array_shapes_warning, DeprecationWarning)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This method will be removed in version 3.0 when the "
                    "current array shapes are no longer supported.",
                )
                self.use_current_array_shapes()

        # check if object has all required UVParameters set
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        return

    def _write_header(self, header):
        """
        Write data to the header datagroup of a CalH5 file.

        Parameters
        ----------
        header : h5py datagroup
            The datagroup to write the header information to. For a UVH5 file
            conforming to the spec, it should be "/Header"

        Returns
        -------
        None
        """
        # write out UVH5 version information
        assert_err_msg = (
            "This is a bug, please make an issue in our issue log at "
            "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues"
        )
        assert self.future_array_shapes, assert_err_msg
        header["version"] = np.string_("0.1")

        # write out telescope and source information
        header["latitude"] = self.telescope_location_lat_lon_alt_degrees[0]
        header["longitude"] = self.telescope_location_lat_lon_alt_degrees[1]
        header["altitude"] = self.telescope_location_lat_lon_alt_degrees[2]
        header["telescope_name"] = np.string_(self.telescope_name)

        # write out required UVParameters
        header["Nants_data"] = self.Nants_data
        header["Nants_telescope"] = self.Nants_telescope
        header["Nfreqs"] = self.Nfreqs
        header["Njones"] = self.Njones
        header["Nspws"] = self.Nspws
        header["Ntimes"] = self.Ntimes
        header["antenna_numbers"] = self.antenna_numbers
        header["channel_width"] = self.channel_width
        header["time_array"] = self.time_array
        header["freq_array"] = self.freq_array
        header["integration_time"] = self.integration_time
        header["lst_array"] = self.lst_array
        header["jones_array"] = self.polarization_array
        header["spw_array"] = self.spw_array
        header["ant_array"] = self.ant_1_array
        header["antenna_positions"] = self.antenna_positions
        header["flex_spw_id_array"] = self.flex_spw_id_array
        # handle antenna_names; works for lists or arrays
        header["antenna_names"] = np.asarray(self.antenna_names, dtype="bytes")
        header["x_orientation"] = np.string_(self.x_orientation)
        header["cal_type"] = np.string_(self.cal_type)
        header["cal_style"] = np.string_(self.cal_style)
        header["gain_convention"] = np.string_(self.gain_convention)
        header["wide_band"] = self.wide_band

        # write out optional parameters
        if self.Nsources is not None:
            header["Nsources"] = self.Nsources
        if self.baseline_range is not None:
            header["baseline_range"] = self.baseline_range
        if self.freq_array is not None:
            header["freq_array"] = self.freq_array
        if self.freq_range is not None:
            header["freq_range"] = self.freq_range
        if self.gain_scale is not None:
            header["gain_scale"] = np.string_(self.gain_scale)
        if self.git_hash_cal is not None:
            header["git_hash_cal"] = np.string_(self.git_hash_cal)
        if self.git_origin_cal is not None:
            header["git_origin_cal"] = np.string_(self.git_origin_cal)
        if self.observer is not None:
            header["observer"] = np.string_(self.observer)
        if self.ref_antenna_name is not None:
            header["ref_antenna_name"] = np.string_(self.ref_antenna_name)
        if self.sky_catalog is not None:
            header["sky_catalog"] = np.string_(self.sky_catalog)
        if self.sky_field is not None:
            header["sky_field"] = np.string_(self.sky_field)

        # write out extra keywords if it exists and has elements
        if self.extra_keywords:
            extra_keywords = header.create_group("extra_keywords")
            for k in self.extra_keywords.keys():
                if isinstance(self.extra_keywords[k], str):
                    extra_keywords[k] = np.string_(self.extra_keywords[k])
                elif self.extra_keywords[k] is None:
                    # save as empty/null dataset
                    extra_keywords[k] = h5py.Empty("f")
                else:
                    extra_keywords[k] = self.extra_keywords[k]

        # write out history
        header["history"] = np.string_(self.history)

        return

    def write_calh5(
        self,
        filename,
        *,
        overwrite=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        quality_compression="lzf",
        gain_write_dtype=None,
        add_to_history=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Write an in-memory UVCal object to a CalH5 file.

        Parameters
        ----------
        filename : str
            The CalH5 file to write to.
        overwrite : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the gain_array or delay. Default is None
            (no filter/compression). In addition to the normal HDF5 filter values, the
            user may specify "bitshuffle" which will set the compression to `32008` for
            bitshuffle and will set the `compression_opts` to `(0, 2)` to allow
            bitshuffle to automatically determine the block size and to use the LZF
            filter after bitshuffle. Using `bitshuffle` requires having the
            `hdf5plugin` package installed.  Dataset must be chunked to use compression.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is the
            LZF filter. Dataset must be chunked.
        quality_compression : str
            HDF5 filter to apply when writing the quality_array and/or
            total_quality_array if they are defined. Default is the LZF filter. Dataset
            must be chunked.
        gain_write_dtype : numpy dtype
            The datatype of output gain data (only applies if cal_type="gain"). If
            'None', then the same datatype as gain_array will be used. The user may
            specify 'c8' for single-precision floats or 'c16' for double-presicion.
            Otherwise, a numpy dtype object must be specified with an 'r' field and an
            'i' field for real and imaginary parts, respectively. See uvh5.py for
            an example of defining such a datatype.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the file located at `filename` already exists and overwrite=False,
            an IOError is raised.

        Notes
        -----
        The HDF5 library allows for the application of "filters" when writing
        data, which can provide moderate to significant levels of compression
        for the datasets in question.  Testing has shown that for some typical
        cases of UVData objects (empty/sparse flag_array objects, and/or uniform
        nsample_arrays), the built-in LZF filter provides significant
        compression for minimal computational overhead.
        """
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if os.path.exists(filename):
            if overwrite:
                print("File exists; overwrite")
            else:
                raise IOError("File exists; skipping")

        revert_fas = False
        if not self.future_array_shapes:
            # We force using future array shapes here.
            # We capture the current state so that it can be reverted later if needed.
            revert_fas = True
            self.use_future_array_shapes()

        data_compression, data_compression_opts = _get_compression(data_compression)

        # open file for writing
        with h5py.File(filename, "w") as f:
            # write header
            header = f.create_group("Header")
            self._write_header(header)

            # write out data, flags, and nsample arrays
            dgrp = f.create_group("Data")
            if self.cal_type == "gain":
                if gain_write_dtype is None:
                    if self.gain_array.dtype == "complex64":
                        gain_write_dtype = "c8"
                    else:
                        gain_write_dtype = "c16"
                if gain_write_dtype not in ("c8", "c16"):
                    _check_uvh5_dtype(gain_write_dtype)
                    gaindata = dgrp.create_dataset(
                        "gains",
                        self.gain_array.shape,
                        chunks=chunks,
                        compression=data_compression,
                        compression_opts=data_compression_opts,
                        dtype=gain_write_dtype,
                    )
                    indices = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
                    _write_complex_astype(self.gain_array, gaindata, indices)
                else:
                    gaindata = dgrp.create_dataset(
                        "gains",
                        chunks=chunks,
                        data=self.gain_array,
                        compression=data_compression,
                        compression_opts=data_compression_opts,
                        dtype=gain_write_dtype,
                    )
            else:
                dgrp.create_dataset(
                    "delays",
                    chunks=chunks,
                    data=self.delay_array,
                    compression=data_compression,
                    compression_opts=data_compression_opts,
                )

            dgrp.create_dataset(
                "flags",
                chunks=chunks,
                data=self.flag_array,
                compression=flags_compression,
            )
            if self.quality_array is not None:
                dgrp.create_dataset(
                    "qualities",
                    chunks=chunks,
                    data=self.quality_array.astype(np.float32),
                    compression=quality_compression,
                )
            if self.total_quality_array is not None:
                dgrp.create_dataset(
                    "total_qualities",
                    chunks=chunks,
                    data=self.total_quality_array.astype(np.float32),
                    compression=quality_compression,
                )

        if revert_fas:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This method will be removed in version 3.0 when the "
                    "current array shapes are no longer supported.",
                )
                self.use_current_array_shapes()

        return
