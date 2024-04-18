# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2023 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with HDF5 files."""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import utils as uvutils

hdf5plugin_present = True
try:
    import hdf5plugin  # noqa: F401
except ImportError as error:
    hdf5plugin_present = False
    hdf5plugin_error = error


def _check_complex_dtype(dtype):
    """
    Check that a specified custom datatype conforms to UVH5 standards.

    According to the UVH5 spec, the data type for the data array must be a
    compound datatype with an "r" field and an "i" field. Additionally, both
    datatypes must be the same (e.g., "<i4", "<r8", etc.).

    Parameters
    ----------
    dtype : numpy dtype
        A numpy dtype object with an "r" field and an "i" field.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        This is raised if dtype is not a numpy dtype, if the dtype does not have
        an "r" field and an "i" field, or if the "r" field and "i" fields have
        different types.
    """
    if not isinstance(dtype, np.dtype):
        raise ValueError("dtype in a uvh5 file must be a numpy dtype")
    if "r" not in dtype.names or "i" not in dtype.names:
        raise ValueError(
            "dtype must be a compound datatype with an 'r' field and an 'i' field"
        )
    rkind = dtype["r"].kind + str(dtype["r"].itemsize)
    ikind = dtype["i"].kind + str(dtype["i"].itemsize)
    if rkind != ikind:
        raise ValueError(
            "dtype must have the same kind ('i4', 'f8', etc.) for both real "
            "and imaginary fields"
        )
    return


def _read_complex_astype(dset, indices, dtype_out=np.complex64):
    """
    Read the given data set of a specified type to floating point complex data.

    Parameters
    ----------
    dset : h5py dataset
        A reference to an HDF5 dataset on disk.
    indices : tuple
        The indices to extract. Should be either lists of indices or numpy
        slice objects.
    dtype_out : str or numpy dtype
        The datatype of the output array. One of (complex, np.complex64,
        np.complex128). Default is np.complex64 (single-precision real and
        imaginary floats).

    Returns
    -------
    output_array : ndarray
        The array referenced in the dataset cast to complex values.

    Raises
    ------
    ValueError
        This is raised if dtype_out is not an acceptable value.
    """
    if dtype_out not in (complex, np.complex64, np.complex128):
        raise ValueError(
            "output datatype must be one of (complex, np.complex64, np.complex128)"
        )
    dset_shape, indices = uvutils._get_dset_shape(dset, indices)
    output_array = np.empty(dset_shape, dtype=dtype_out)
    # dset is indexed in native dtype, but is upcast upon assignment

    if dtype_out == np.complex64:
        compound_dtype = [("r", "f4"), ("i", "f4")]
    else:
        compound_dtype = [("r", "f8"), ("i", "f8")]

    output_array.view(compound_dtype)[:, :] = uvutils._index_dset(dset, indices)[:, :]

    return output_array


def _write_complex_astype(data, dset, indices):
    """
    Write floating point complex data as a specified type.

    Parameters
    ----------
    data : ndarray
        The data array to write out. Should be a complex-valued array that
        supports the .real and .imag attributes for accessing real and imaginary
        components.
    dset : h5py dataset
        A reference to an HDF5 dataset on disk.
    indices : tuple
        A 3-tuple representing indices to write data to. Should be either lists
        of indices or numpy slice objects. For data arrays with 4 dimensions, the second
        dimension (the old spw axis) should not be included because it can only be
        length one.

    Returns
    -------
    None
    """
    # get datatype from dataset
    dtype_out = dset.dtype

    if data.dtype == np.complex64:
        compound_dtype = [("r", "f4"), ("i", "f4")]
    else:
        compound_dtype = [("r", "f8"), ("i", "f8")]

    # make doubly sure dtype is valid; should be unless user is pathological
    _check_complex_dtype(dtype_out)
    if len(dset.shape) == 3:
        # this is the future array shape
        dset[indices[0], indices[1], indices[2]] = data.view(compound_dtype).astype(
            dtype_out, copy=False
        )
    else:
        dset[indices[0], np.s_[:], indices[1], indices[2]] = data.view(
            compound_dtype
        ).astype(dtype_out, copy=False)

    return


def _get_compression(compression):
    """
    Get the HDF5 compression and compression options to use.

    Parameters
    ----------
    compression : str
        HDF5 compression specification or "bitshuffle".

    Returns
    -------
    compression_use : str
        HDF5 compression specification
    compression_opts : tuple
        HDF5 compression options
    """
    if compression == "bitshuffle":
        if not hdf5plugin_present:  # pragma: no cover
            raise ImportError(
                "The hdf5plugin package is not installed but is required to use "
                "bitshuffle compression."
            ) from hdf5plugin_error

        compression_use = 32008
        compression_opts = (0, 2)
    else:
        compression_use = compression
        compression_opts = None

    return compression_use, compression_opts


class HDF5Meta:
    """
    A base class for fast read-only interface to our HDF5 file metadata.

    This class is just a really thin wrapper over our HDF5 files that makes it easier
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

    _defaults = {}
    _string_attrs = frozenset({})
    _int_attrs = frozenset({})
    _float_attrs = frozenset({})
    _bool_attrs = frozenset({})

    def __init__(self, path: str | Path | h5py.File | h5py.Group):
        self.__file = None

        if isinstance(path, h5py.File):
            self.path = Path(path.filename).resolve()
            self.__file = path
            self.__header = path["/Header"]
            self.__datagrp = path["/Data"]
        elif isinstance(path, h5py.Group):
            self.path = Path(path.file.filename).resolve()
            self.__file = path.file
            self.__header = path
            self.__datagrp = self.__file["/Data"]
        elif isinstance(path, (str, Path)):
            self.path = Path(path).resolve()

    def is_open(self) -> bool:
        """Whether the file is open."""
        return bool(self.__file)

    def __del__(self):
        """Close the file when the object is deleted."""
        if self.__file:
            self.__file.close()

    def __getstate__(self):
        """Get the state of the object."""
        print(self.__dict__.keys())
        print(self.__class__.__name__)
        return {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in (
                "_HDF5Meta__file",
                "_HDF5Meta__header",
                "_HDF5Meta__datagrp",
                "header",
                "datagrp",
            )
        }

    def __setstate__(self, state):
        """Set the state of the object."""
        self.__dict__.update(state)
        self.__file = None

    def __eq__(self, other):
        """Check equality of two HDF5Meta objects."""
        if not isinstance(other, self.__class__):
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
            elif name in self._float_attrs:
                x = float(x)

            self.__dict__[name] = x
            return x
        except KeyError:
            try:
                return self._defaults[name]
            except KeyError as e:
                raise AttributeError(f"{name} not found in {self.path}") from e

    @cached_property
    def antpos_enu(self) -> np.ndarray:
        """The antenna positions in ENU coordinates, in meters."""
        lat, lon, alt = self.telescope_location_lat_lon_alt
        return uvutils.ENU_from_ECEF(
            self.antenna_positions + self.telescope_location,
            latitude=lat,
            longitude=lon,
            altitude=alt,
            frame=self.telescope_frame,
            ellipsoid=self.ellipsoid,
        )

    @cached_property
    def telescope_location(self):
        """The telescope location in ECEF coordinates, in meters."""
        return uvutils.XYZ_from_LatLonAlt(
            *self.telescope_location_lat_lon_alt,
            frame=self.telescope_frame,
            ellipsoid=self.ellipsoid,
        )

    @property
    def telescope_location_lat_lon_alt(self) -> tuple[float, float, float]:
        """The telescope location in latitude, longitude, and altitude, in degrees."""
        return self.latitude * np.pi / 180, self.longitude * np.pi / 180, self.altitude

    @property
    def telescope_location_lat_lon_alt_degrees(self) -> tuple[float, float, float]:
        """The telescope location in latitude, longitude, and altitude, in degrees."""
        return self.latitude, self.longitude, self.altitude

    @property
    def telescope_frame(self) -> str:
        """The telescope frame."""
        h = self.header
        if "telescope_frame" in h:
            telescope_frame = bytes(h["telescope_frame"][()]).decode("utf8")
            if telescope_frame not in ["itrs", "mcmf"]:
                raise ValueError(
                    f"Telescope frame in file is {telescope_frame}. "
                    "Only 'itrs' and 'mcmf' are currently supported."
                )
            return telescope_frame
        else:
            # default to ITRS
            return "itrs"

    @property
    def ellipsoid(self) -> str:
        """The reference ellipsoid to use for lunar coordinates."""
        h = self.header
        if self.telescope_frame == "mcmf":
            if "ellipsoid" in h:
                return bytes(h["ellipsoid"][()]).decode("utf8")
            else:
                return "SPHERE"
        else:
            return None

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

    @cached_property
    def phase_center_catalog(self) -> dict | None:
        """Dictionary of phase centers."""
        header = self.header
        if "phase_center_catalog" not in header:
            return None
        phase_center_catalog = {}
        key0 = next(iter(header["phase_center_catalog"].keys()))
        if isinstance(header["phase_center_catalog"][key0], h5py.Group):
            # This is the new, correct way
            for pc, pc_dict in header["phase_center_catalog"].items():
                pc_id = int(pc)
                phase_center_catalog[pc_id] = {}
                for key, dset in pc_dict.items():
                    if issubclass(dset.dtype.type, np.bytes_):
                        phase_center_catalog[pc_id][key] = bytes(dset[()]).decode(
                            "utf8"
                        )
                    elif dset.shape is None:
                        phase_center_catalog[pc_id][key] = None
                    else:
                        phase_center_catalog[pc_id][key] = dset[()]
        else:
            # This is the old way this was written
            for key in header["phase_center_catalog"].keys():
                pc_dict = json.loads(
                    bytes(header["phase_center_catalog"][key][()]).decode("utf8")
                )
                pc_dict["cat_name"] = key
                pc_id = pc_dict.pop("cat_id")
                phase_center_catalog[pc_id] = pc_dict

        return phase_center_catalog
