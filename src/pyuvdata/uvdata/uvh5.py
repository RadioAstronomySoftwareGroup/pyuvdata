# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing UVH5 files."""

from __future__ import annotations

import contextlib
import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from docstring_parser import DocstringStyle

from .. import Telescope, utils
from ..docstrings import copy_replace_short_description
from ..utils.io import hdf5 as hdf5_utils
from . import UVData

__all__ = ["UVH5", "FastUVH5Meta"]


# define HDF5 type for interpreting HERA correlator outputs (integers) as
# complex numbers
_hera_corr_dtype = np.dtype([("r", "<i4"), ("i", "<i4")])

hdf5plugin_present = True
try:
    import hdf5plugin  # noqa: F401
except ImportError as error:
    hdf5plugin_present = False
    hdf5plugin_error = error


class FastUVH5Meta(hdf5_utils.HDF5Meta):
    """
    A fast read-only interface to UVH5 file metadata that makes some assumptions.

    This class is just a really thin wrapper over a UVH5 file that makes it easier
    to read in parts of the metadata at a time. This makes it much faster to perform
    small tasks where simple metadata is required, rather than reading in the whole
    header.

    All metadata is available as attributes, through ``__getattr__`` magic. Thus,
    accessing eg. ``obj.freq_array`` will go and get the frequencies directly from the
    file, and store them in memory. However, some attributes are made faster than the
    default, by assumptions on the data shape -- in particular, times and baselines.

    Anything that is read in is stored in memory so the second access is much faster.
    However, the memory can be released simply by deleting the attribute (it can be
    accessed again, and the data will be re-read).

    Parameters
    ----------
    filename : str or Path
        The filename to read from.
    blt_order : tuple of str or "determine", optional
        The order of the baseline-time axis. This can be determined, or read
        directly from file, however since it has been optional in the past, many
        existing files do not contain it in the metadata.
        Some reading operations are significantly faster if this is known, so providing
        it here can provide a speedup. Default is to try and read it from file,
        and if not there, just leave it as None. Set to "determine" to auto-detect
        the blt_order from the metadata (takes extra time to do so).
    blts_are_rectangular : bool, optional
        Whether the baseline-time axis is rectangular. This can be read from metadata
        in new files, but many old files do not contain it. If not provided, the
        rectangularity will be determined from the data. This is a non-negligible
        operation, so if you know it, it can be provided here to speed up reading.
    time_axis_faster_than_bls : bool, optional
        If blts are rectangular, this variable specifies whether the time axis is
        the fastest-moving virtual axis. Various reading functions benefit from knowing
        this, so if it is known, it can be provided here to speed up reading. It will
        be determined from the data if not provided.
    recompute_nbls : bool, optional
        Whether to recompute the number of unique baselines from the data. Before v1.2
        of the UVH5 spec, it was possible to have an incorrect number of baselines in
        the header without error, so this provides an opportunity to rectify it. Old
        HERA files (< March 2023) may have this issue, but in this case the correct
        number of baselines can be computed more quickly than by fully re=computing,
        and so we do this.
    astrometry_library : str
        Library used for calculating the LSTs. Allowed options are
        'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas
        library), and 'astropy' (which uses the astropy utilities). Default is erfa
        unless the telescope location frame is MCMF (on the moon), in which case the
        default is astropy.

    Notes
    -----
    To check if a particular attribute is available, use ``hasattr(obj, attr)``.
    Many attributes will not show up dynamically in an interpreter, because they are
    gotten dynamically from the file.
    """

    _string_attrs = frozenset(
        {
            "history",
            "instrument",
            "object_name",
            "x_orientation",
            "telescope_name",
            "rdate",
            "timesys",
            "eq_coeffs_convention",
            "phase_center_frame",
            "version",
            "pol_convention",
        }
    )

    _defaults = {"x_orientation": None}

    _int_attrs = frozenset(
        {
            "Nblts",
            "Ntimes",
            "Npols",
            "Nspws",
            "Nfreqs",
            "uvplane_reference_time",
            "Nphase",
            "Nants_data",
            "Nants_telescope",
        }
    )
    _float_attrs = frozenset(
        {
            "dut1",
            "earth_omega",
            "gst0",
            "phase_center_ra",
            "phase_center_dec",
            "phase_center_epoch",
        }
    )

    def __init__(
        self,
        path: str | Path | h5py.File | h5py.Group,
        *,
        blt_order: Literal["determine"] | tuple[str] | None = None,
        blts_are_rectangular: bool | None = None,
        time_axis_faster_than_bls: bool | None = None,
        recompute_nbls: bool | None = None,
        astrometry_library: str | None = None,
    ):
        super().__init__(path)

        self.__blts_are_rectangular = blts_are_rectangular
        self.__time_first = time_axis_faster_than_bls
        self.__blt_order = blt_order
        self._recompute_nbls = recompute_nbls
        self._astrometry_library = astrometry_library

    def __eq__(self, other):
        """Check equality of two FastUVH5Meta objects."""
        if not isinstance(other, FastUVH5Meta):
            return False

        return (
            self.path == other.path
            and self.__blts_are_rectangular == other.__blts_are_rectangular
            and (
                self.__time_first == other.__time_first
                or self.__time_first is None
                or other.__time_first is None
            )
            and self.__blt_order == other.__blt_order
            and self.Nbls == other.Nbls
        )

    def __hash__(self):
        """Get a unique hash for the object."""
        return super().__hash__()

    @cached_property
    def Nbls(self) -> int:  # noqa: N802
        """The number of unique baselines."""
        if self._recompute_nbls:
            if self.__blts_are_rectangular:
                return self.Nblts // self.Ntimes
            else:
                return len(np.unique(self.baseline_array))
        else:
            nbls = int(self.header["Nbls"][()])

            if self._recompute_nbls is None:
                # Test if this is an "old" HERA file. If so, the Nbls is wrong
                # in the header, and equal to Nblts == Nbls*Ntimes.
                if (
                    self.telescope_name == "HERA"
                    and nbls == self.Nblts
                    and (self.__blts_are_rectangular or self.Nblts % self.Ntimes == 0)
                ):
                    return self.Nblts // self.Ntimes
                else:
                    return nbls
            else:
                return nbls

    def get_blt_order(self) -> tuple[str]:
        """Get the blt order from analysing metadata."""
        return utils.bltaxis.determine_blt_order(
            time_array=self.time_array,
            ant_1_array=self.ant_1_array,
            ant_2_array=self.ant_2_array,
            baseline_array=self.baseline_array,
            Nbls=self.Nbls,
            Ntimes=self.Ntimes,
        )

    @cached_property
    def blt_order(self) -> tuple[str]:
        """Tuple defining order of blts axis."""
        if self.__blt_order not in (None, "determine"):
            return self.__blt_order

        h = self.header
        if "blt_order" in h:
            blt_order_str = bytes(h["blt_order"][()]).decode("utf8")
            return tuple(blt_order_str.split(", "))
        else:
            if self.__blt_order == "determine":
                return self.get_blt_order()
            else:
                return None

    @cached_property
    def blts_are_rectangular(self) -> bool:
        """Whether blts axis is rectangular.

        That is, this is true if the blts can be reshaped to ``(Ntimes, Nbls)`` OR
        ``(Nbls, Ntimes)``. This requires each baseline to have the same number of times
        associated, AND the blt ordering to either be (time, baseline) or
        (baseline,time).
        """
        if self.__blts_are_rectangular is not None:
            return self.__blts_are_rectangular

        h = self.header
        if "blts_are_rectangular" in h:
            return bool(h["blts_are_rectangular"][()])

        if self.Nblts == self.Ntimes * self.Nbls and self.blt_order in (
            ("time", "baseline"),
            ("baseline", "time"),
        ):
            return True

        is_rect, self.__time_first = utils.bltaxis.determine_rectangularity(
            time_array=self.time_array,
            baseline_array=self.baseline_array,
            nbls=self.Nbls,
            ntimes=self.Ntimes,
        )
        return is_rect

    @cached_property
    def time_axis_faster_than_bls(self) -> bool:
        """Whether times move first in the blt axis."""
        # first hit the blts_are_rectangular property to set the time_first property
        if not self.blts_are_rectangular:
            return False
        if self.__time_first is not None:
            return self.__time_first
        if "time_axis_faster_than_bls" in self.header:
            return bool(self.header["time_axis_faster_than_bls"][()])
        if self.Ntimes == 1:
            return False
        if self.Ntimes == self.Nblts:
            return True
        return self.header["time_array"][1] != self.header["time_array"][0]

    @cached_property
    def times(self) -> np.ndarray:
        """The unique times in the file."""
        h = self.header
        if self.blts_are_rectangular:
            if self.time_axis_faster_than_bls:
                return h["time_array"][: self.Ntimes]
            else:
                return h["time_array"][:: self.Nbls]
        else:
            return np.unique(self.time_array)

    @cached_property
    def lsts(self) -> np.ndarray:
        """The unique LSTs in the file."""
        h = self.header
        if "lst_array" in h and self.blts_are_rectangular:
            if self.time_axis_faster_than_bls:
                return h["lst_array"][: self.Ntimes]
            else:
                return h["lst_array"][:: self.Nbls]
        else:
            return np.unique(self.lst_array)

    @cached_property
    def lst_array(self) -> np.ndarray:
        """The LSTs corresponding to each baseline-time."""
        h = self.header
        if "lst_array" in h:
            return h["lst_array"][:]
        else:
            lst_array = utils.get_lst_for_time(
                jd_array=self.time_array,
                telescope_loc=self.telescope_location_obj,
                astrometry_library=self._astrometry_library,
            )
            return lst_array

    @cached_property
    def channel_width(self) -> float:
        """The width of each frequency channel in Hz."""
        # Pull in the channel_width parameter as either an array or as a single float,
        # depending on whether or not the data is stored with a flexible spw.
        h = self.header
        if np.asarray(h["channel_width"]).ndim == 1:
            return h["channel_width"][:]
        else:
            return float(h["channel_width"][()])

    @cached_property
    def unique_antpair_1_array(self) -> np.ndarray:
        """The unique antenna 1 indices in the file."""
        h = self.header
        if self.blts_are_rectangular:
            if self.time_axis_faster_than_bls:
                return h["ant_1_array"][:: self.Ntimes]
            else:
                return h["ant_1_array"][: self.Nbls]
        else:
            return np.array([x for x, y in self.antpairs])

    @cached_property
    def unique_antpair_2_array(self) -> np.ndarray:
        """The unique antenna 2 indices in the file."""
        h = self.header
        if self.blts_are_rectangular:
            if self.time_axis_faster_than_bls:
                return h["ant_2_array"][:: self.Ntimes]
            else:
                return h["ant_2_array"][: self.Nbls]
        else:
            return np.array([y for x, y in self.antpairs])

    @cached_property
    def unique_ants(self) -> set:
        """The unique antennas in the file."""
        return set(
            np.unique(
                np.concatenate(
                    (self.unique_antpair_1_array, self.unique_antpair_2_array)
                )
            )
        )

    @cached_property
    def baseline_array(self) -> np.ndarray:
        """The baselines in the file, as unique integers."""
        return utils.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array, Nants_telescope=self.Nants_telescope
        )

    @cached_property
    def unique_baseline_array(self) -> np.ndarray:
        """The unique baselines in the file, as unique integers."""
        return utils.antnums_to_baseline(
            self.unique_antpair_1_array,
            self.unique_antpair_2_array,
            Nants_telescope=self.Nants_telescope,
        )

    @cached_property
    def antpairs(self) -> list[tuple[int, int]]:
        """Get the unique antenna pairs in the file."""
        if self.blts_are_rectangular:
            return list(
                zip(
                    self.unique_antpair_1_array,
                    self.unique_antpair_2_array,
                    strict=True,
                )
            )
        else:
            return list(set(zip(self.ant_1_array, self.ant_2_array, strict=True)))

    def has_key(self, key: tuple[int, int] | tuple[int, int, str]) -> bool:
        """Check if the file has a given antpair or antpair-pol key."""
        return bool(
            len(key) == 2
            and key in self.antpairs
            or (key[1], key[0]) in self.antpairs
            or len(key) == 3
            and (
                key[:2] in self.antpairs
                and key[2] in self.pols
                or (key[1], key[0]) in self.antpairs
                and key[2][::-1] in self.pols
            )
        )

    @cached_property
    def pols(self) -> list[str]:
        """The polarizations in the file, as standardized strings, eg. 'xx' or 'ee'."""
        return [
            utils.polnum2str(p, x_orientation=self.x_orientation)
            for p in self.polarization_array
        ]

    @cached_property
    def vis_units(self) -> str:
        """The visibility units in the file, as a string."""
        # check for vis_units
        h = self.header
        if "vis_units" in h:
            vis_units = bytes(h["vis_units"][()]).decode("utf8")
            # Added here because older files allowed for both upper and lowercase
            # formats, although since the attribute is case sensitive, we want to
            # correct for this here.
            if vis_units == "UNCALIB":
                vis_units = "uncalib"
        else:
            # default to uncalibrated data
            vis_units = "uncalib"
        return vis_units

    def to_uvdata(
        self, *, check_lsts: bool = False, astrometry_library: str | None = None
    ) -> UVData:
        """Convert the file to a UVData object.

        The object will be metadata-only.

        Parameters
        ----------
        check_lsts : bool
            Option to check that the LSTs match the expected values for the telescope
            location and times.
        astrometry_library : str
            Library used for calculating the LSTs. Allowed options are
            'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas
            library), and 'astropy' (which uses the astropy utilities). Default is erfa
            unless the telescope location frame is MCMF (on the moon), in which case the
            default is astropy.

        """
        uvd = UVData()
        uvd.read_uvh5(
            self,
            read_data=False,
            run_check_acceptability=check_lsts,
            astrometry_library=astrometry_library,
        )
        return uvd


class UVH5(UVData):
    """
    A class for UVH5 file objects.

    This class defines an HDF5-specific subclass of UVData for reading and
    writing UVH5 files. This class should not be interacted with directly,
    instead use the read_uvh5 and write_uvh5 methods on the UVData class.
    """

    def _read_header_with_fast_meta(
        self,
        filename: str | Path | FastUVH5Meta,
        *,
        default_mount_type: str | None = None,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
        blt_order: tuple[str] | None | Literal["determine"] = None,
        blts_are_rectangular: bool | None = None,
        time_axis_faster_than_bls: bool | None = None,
        background_lsts: bool = True,
        recompute_nbls: bool | None = None,
        astrometry_library: str | None = None,
    ):
        if not isinstance(filename, FastUVH5Meta):
            meta = FastUVH5Meta(
                filename,
                blt_order=blt_order,
                blts_are_rectangular=blts_are_rectangular,
                time_axis_faster_than_bls=time_axis_faster_than_bls,
                recompute_nbls=recompute_nbls,
                astrometry_library=astrometry_library,
            )
        else:
            meta = filename

        # First, get the things relevant for setting LSTs, so that can be run in the
        # background if desired.
        self.time_array = meta.time_array
        required_telescope_keys = [
            "telescope_name",
            "latitude",
            "longitude",
            "altitude",
            "instrument",
            "Nants_telescope",
            "antenna_names",
            "antenna_numbers",
            "antenna_positions",
        ]
        # Skip this check for now since we're repeating it below
        self.telescope = Telescope.from_hdf5(
            filename, required_keys=required_telescope_keys, run_check=False
        )
        self._set_telescope_requirements()

        if "lst_array" in meta.header:
            self.lst_array = meta.header["lst_array"][:]
            proc = None

            if run_check_acceptability:
                utils.times.check_lsts_against_times(
                    jd_array=self.time_array,
                    lst_array=self.lst_array,
                    telescope_loc=self.telescope.location,
                    lst_tols=(0, utils.LST_RAD_TOL),
                )
        else:
            proc = self.set_lsts_from_time_array(
                background=background_lsts, astrometry_library=astrometry_library
            )

        req_params = [
            "history",
            "vis_units",
            "Nfreqs",
            "Npols",
            "Nspws",
            "Ntimes",
            "Nblts",
            "Nbls",
            "Nants_data",
            "ant_1_array",
            "ant_2_array",
            "baseline_array",
            "integration_time",
            "freq_array",
            "spw_array",
            "polarization_array",
            "uvw_array",
            "channel_width",
        ]

        # Take care of the phase center attributes separately, in order to support
        # older versions of the format.
        if meta.phase_center_catalog is not None:
            req_params += [
                "phase_center_catalog",
                "phase_center_app_ra",
                "phase_center_app_dec",
                "phase_center_frame_pa",
                "phase_center_id_array",
                "Nphase",
            ]

        # Required parameters
        for attr in req_params:
            try:
                setattr(self, attr, getattr(meta, attr))
            except AttributeError as e:
                raise KeyError(str(e)) from e

        # check this as soon as we have the inputs
        if self.freq_array.ndim == 2:
            # This is an old-shaped data set, remove the old spw axis
            self.freq_array = self.freq_array[0]
            self.channel_width = np.full(self.freq_array.size, self.channel_width)

        arr_shape_msg = (
            "The size of arrays in this file are not internally consistent, "
            "which should not happen. Please file an issue in our GitHub issue "
            "log so that we can fix it."
        )
        assert np.asarray(self.channel_width).size == self.freq_array.size, (
            arr_shape_msg
        )

        # For now, only set the rectangularity parameters if they exist in the header of
        # the file. These could be set automatically later on, but for now we'll leave
        # that up to the user dealing with the UVData object.
        if "blts_are_rectangular" in meta.header:
            self.blts_are_rectangular = meta.blts_are_rectangular
        if "time_axis_faster_than_bls" in meta.header:
            self.time_axis_faster_than_bls = meta.time_axis_faster_than_bls

        if not utils.history._check_history_version(
            self.history, self.pyuvdata_version_str
        ):
            self.history += self.pyuvdata_version_str

        # Optional parameters
        for attr in [
            "dut1",
            "earth_omega",
            "gst0",
            "rdate",
            "timesys",
            "blt_order",
            "uvplane_reference_time",
            "eq_coeffs",
            "eq_coeffs_convention",
            "flex_spw_id_array",
            "flex_spw_polarization_array",
            "extra_keywords",
            "pol_convention",
        ]:
            with contextlib.suppress(AttributeError):
                setattr(self, attr, getattr(meta, attr))

        if self.blt_order is not None:
            self._blt_order.form = (len(self.blt_order),)

        if self.flex_spw_id_array is None:
            self.flex_spw_id_array = np.full(self.Nfreqs, self.spw_array[0], dtype=int)

        # Here is where we start handling phase center information.  If we have a
        # multi phase center dataset, we need to get different header items
        if meta.phase_center_catalog is None:
            phase_type = bytes(meta.header["phase_type"][()]).decode("utf8")
            if phase_type == "phased":
                cat_id = self._add_phase_center(
                    getattr(meta, "object_name", "unknown"),
                    cat_type="sidereal",
                    cat_lon=meta.phase_center_ra,
                    cat_lat=meta.phase_center_dec,
                    cat_frame=meta.phase_center_frame,
                    cat_epoch=meta.phase_center_epoch,
                )
            else:
                if phase_type != "drift":
                    warnings.warn(
                        "Unknown phase type, assuming object is unprojected "
                        "(unphased) by default."
                    )
                cat_id = self._add_phase_center(
                    getattr(meta, "object_name", "unprojected"), cat_type="unprojected"
                )

            self.phase_center_id_array = np.full(meta.Nblts, cat_id, dtype=int)
        # set any extra telescope params
        self.set_telescope_params(
            x_orientation=meta.x_orientation,
            mount_type=default_mount_type,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        # wait for the LST computation if needed
        if proc is not None:
            proc.join()

    def _read_header(
        self, filename: str | Path | FastUVH5Meta | h5py.File | h5py.Group, **kwargs
    ):
        """
        Read header information from a UVH5 file.

        This is an internal function called by the user-space methods.
        Properties of the UVData object are updated as the file is processed.

        Parameters
        ----------
        filename : string, path, FastUVH5Meta, h5py.File or h5py.Group
            A file name or path or a FastUVH5Meta or h5py File or Group object that
            contains the header information. Should be called "/Header" for UVH5 files
            conforming to spec.

        Other Parameters
        ----------------
        All other parameters passed through to :func:`_read_header_with_fast_meta`.

        Returns
        -------
        None
        """
        self._read_header_with_fast_meta(filename, **kwargs)

    def _get_data(
        self,
        dgrp,
        *,
        antenna_nums,
        antenna_names,
        ant_str,
        bls,
        frequencies,
        freq_chans,
        spws,
        times,
        time_range,
        lsts,
        lst_range,
        polarizations,
        blt_inds,
        phase_center_ids,
        catalog_names,
        data_array_dtype,
        keep_all_metadata,
        multidim_index,
    ):
        """
        Read the data-size arrays (data, flags, nsamples) from a file.

        This is an internal function to read just the visibility, flag, and
        nsample data of the UVH5 file. This is separated from full read so that
        header/metadata and data can be read independently. See the
        documentation of `read_uvh5` for a full description of most of the
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
        if (
            "32008" in dgrp["visdata"]._filters and not hdf5plugin_present
        ):  # pragma: no cover
            raise ImportError(
                "hdf5plugin is not installed but is required to read this dataset"
            ) from hdf5plugin_error

        # figure out what data to read in
        blt_inds, freq_inds, spw_inds, pol_inds, history_update_string = (
            self._select_preprocess(
                antenna_nums=antenna_nums,
                antenna_names=antenna_names,
                ant_str=ant_str,
                bls=bls,
                frequencies=frequencies,
                freq_chans=freq_chans,
                spws=spws,
                times=times,
                time_range=time_range,
                lsts=lsts,
                lst_range=lst_range,
                polarizations=polarizations,
                blt_inds=blt_inds,
                phase_center_ids=phase_center_ids,
                catalog_names=catalog_names,
                invert=False,
            )
        )

        # figure out which axis is the most selective
        if blt_inds is not None:
            blt_frac = len(blt_inds) / float(self.Nblts)
        else:
            blt_frac = 1

        if freq_inds is not None:
            freq_frac = len(freq_inds) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if pol_inds is not None:
            pol_frac = len(pol_inds) / float(self.Npols)
        else:
            pol_frac = 1

        min_frac = np.min([blt_frac, freq_frac, pol_frac])

        arr_shape_msg = (
            "The size of arrays in this file are not internally consistent, "
            "which should not happen. Please file an issue in our GitHub issue "
            "log so that we can fix it."
        )

        if dgrp["visdata"].ndim == 3:
            assert self.freq_array.ndim == 1, arr_shape_msg
            assert self.channel_width.size == self.freq_array.size, arr_shape_msg

        # get the fundamental datatype of the visdata; if integers, we need to
        # cast to floats
        visdata_dtype = dgrp["visdata"].dtype
        if visdata_dtype not in ("complex64", "complex128"):
            hdf5_utils._check_complex_dtype(visdata_dtype)
            if data_array_dtype not in (np.complex64, np.complex128):
                raise ValueError(
                    "data_array_dtype must be np.complex64 or np.complex128"
                )
            custom_dtype = True
        else:
            custom_dtype = False

        if min_frac == 1:
            # no select, read in all the data
            inds = (np.s_[:], np.s_[:], np.s_[:])
            if custom_dtype:
                self.data_array = hdf5_utils._read_complex_astype(
                    dgrp["visdata"], inds, data_array_dtype
                )
            else:
                self.data_array = hdf5_utils._index_dset(dgrp["visdata"], inds)
            self.flag_array = hdf5_utils._index_dset(dgrp["flags"], inds)
            self.nsample_array = hdf5_utils._index_dset(dgrp["nsamples"], inds)
        else:
            # do select operations on everything except data_array, flag_array
            # and nsample_array
            self._select_by_index(
                blt_inds=blt_inds,
                freq_inds=freq_inds,
                spw_inds=spw_inds,
                pol_inds=pol_inds,
                history_update_string=history_update_string,
                keep_all_metadata=keep_all_metadata,
            )

            # determine which axes can be sliced, rather than fancy indexed
            # max_nslice_frac of 0.1 yields slice speedup over fancy index for HERA data
            # See pyuvdata PR #805
            if blt_inds is not None:
                blt_slices, blt_sliceable = utils.tools._convert_to_slices(
                    blt_inds, max_nslice_frac=0.1
                )
            else:
                blt_inds, blt_slices = np.s_[:], np.s_[:]
                blt_sliceable = True

            if freq_inds is not None:
                freq_slices, freq_sliceable = utils.tools._convert_to_slices(
                    freq_inds, max_nslice_frac=0.1
                )
            else:
                freq_inds, freq_slices = np.s_[:], np.s_[:]
                freq_sliceable = True

            if pol_inds is not None:
                pol_slices, pol_sliceable = utils.tools._convert_to_slices(
                    pol_inds, max_nslice_frac=0.5
                )
            else:
                pol_inds, pol_slices = np.s_[:], np.s_[:]
                pol_sliceable = True

            # open references to datasets
            visdata_dset = dgrp["visdata"]
            flags_dset = dgrp["flags"]
            nsamples_dset = dgrp["nsamples"]

            # check that multidim_index is appropriate
            # if more than one dim is not sliceable, then not appropriate
            if (
                multidim_index
                and sum([blt_sliceable, freq_sliceable, pol_sliceable]) < 2
            ):
                multidim_index = False

            # just read in the right portions of the data and flag arrays
            if blt_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [blt_inds, np.s_[:], np.s_[:]]
                if blt_sliceable:
                    inds[0] = blt_slices
                if multidim_index:
                    if freq_sliceable:
                        inds[1] = freq_slices
                    else:
                        inds[1] = freq_inds
                if multidim_index:
                    if pol_sliceable:
                        inds[2] = pol_slices
                    else:
                        inds[2] = pol_inds

                inds = tuple(inds)

                # index datasets
                if custom_dtype:
                    visdata = hdf5_utils._read_complex_astype(
                        visdata_dset, inds, data_array_dtype
                    )
                else:
                    visdata = hdf5_utils._index_dset(visdata_dset, inds)
                flags = hdf5_utils._index_dset(flags_dset, inds)
                nsamples = hdf5_utils._index_dset(nsamples_dset, inds)
                # down select on other dimensions if necessary
                # use indices not slices here: generally not the bottleneck
                if not multidim_index and freq_frac < 1:
                    visdata = visdata[..., freq_inds, :]
                    flags = flags[..., freq_inds, :]
                    nsamples = nsamples[..., freq_inds, :]
                if not multidim_index and pol_frac < 1:
                    visdata = visdata[..., :, pol_inds]
                    flags = flags[..., :, pol_inds]
                    nsamples = nsamples[..., :, pol_inds]

            elif freq_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], freq_inds, np.s_[:]]
                if freq_sliceable:
                    inds[1] = freq_slices
                if multidim_index:
                    if blt_sliceable:
                        inds[0] = blt_slices
                    else:
                        inds[0] = blt_inds
                if multidim_index:
                    if pol_sliceable:
                        inds[2] = pol_slices
                    else:
                        inds[2] = pol_inds

                inds = tuple(inds)

                # index datasets
                if custom_dtype:
                    visdata = hdf5_utils._read_complex_astype(
                        visdata_dset, inds, data_array_dtype
                    )
                else:
                    visdata = hdf5_utils._index_dset(visdata_dset, inds)
                flags = hdf5_utils._index_dset(flags_dset, inds)
                nsamples = hdf5_utils._index_dset(nsamples_dset, inds)

                # down select on other dimensions if necessary
                # use indices not slices here: generally not the bottleneck
                if not multidim_index and blt_frac < 1:
                    visdata = visdata[blt_inds]
                    flags = flags[blt_inds]
                    nsamples = nsamples[blt_inds]
                if not multidim_index and pol_frac < 1:
                    visdata = visdata[..., pol_inds]
                    flags = flags[..., pol_inds]
                    nsamples = nsamples[..., pol_inds]
            else:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], np.s_[:], pol_inds]
                if pol_sliceable:
                    inds[2] = pol_slices
                if multidim_index:
                    if blt_sliceable:
                        inds[0] = blt_slices
                    else:
                        inds[0] = blt_inds
                if multidim_index:
                    if freq_sliceable:
                        inds[1] = freq_slices
                    else:
                        inds[1] = freq_inds

                inds = tuple(inds)

                # index datasets
                if custom_dtype:
                    visdata = hdf5_utils._read_complex_astype(
                        visdata_dset, inds, data_array_dtype
                    )
                else:
                    visdata = hdf5_utils._index_dset(visdata_dset, inds)
                flags = hdf5_utils._index_dset(flags_dset, inds)
                nsamples = hdf5_utils._index_dset(nsamples_dset, inds)

                # down select on other dimensions if necessary
                # use indices not slices here: generally not the bottleneck
                if not multidim_index and blt_frac < 1:
                    visdata = visdata[blt_inds]
                    flags = flags[blt_inds]
                    nsamples = nsamples[blt_inds]
                if not multidim_index and freq_frac < 1:
                    visdata = visdata[..., freq_inds, :]
                    flags = flags[..., freq_inds, :]
                    nsamples = nsamples[..., freq_inds, :]

            # save arrays in object
            self.data_array = visdata
            self.flag_array = flags
            self.nsample_array = nsamples

        if self.data_array.ndim == 4:
            # This is a dataset with old shapes -- squeeze out the former spw axis
            self.data_array = np.squeeze(self.data_array, axis=1)
            self.flag_array = np.squeeze(self.flag_array, axis=1)
            self.nsample_array = np.squeeze(self.nsample_array, axis=1)

        assert self.freq_array.ndim == 1, arr_shape_msg
        assert self.channel_width.size == self.freq_array.size, arr_shape_msg

    @copy_replace_short_description(UVData.read_uvh5, style=DocstringStyle.NUMPYDOC)
    def read_uvh5(
        self,
        filename,
        *,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        blt_inds=None,
        phase_center_ids=None,
        catalog_names=None,
        keep_all_metadata=True,
        read_data=True,
        data_array_dtype=np.complex128,
        multidim_index=False,
        remove_flex_pol=True,
        background_lsts=True,
        default_mount_type="other",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        fix_old_proj=None,
        fix_use_ant_pos=True,
        check_autos=True,
        fix_autos=True,
        blt_order: tuple[str] | Literal["determine"] | None = None,
        blts_are_rectangular: bool | None = None,
        time_axis_faster_than_bls: bool | None = None,
        recompute_nbls: bool | None = None,
        astrometry_library: str | None = None,
    ):
        """Read in data from a UVH5 file."""
        if isinstance(filename, FastUVH5Meta):
            meta = filename
            filename = str(meta.path)
            close_meta = False
        else:
            close_meta = True
            meta = FastUVH5Meta(
                filename,
                blt_order=blt_order,
                blts_are_rectangular=blts_are_rectangular,
                time_axis_faster_than_bls=time_axis_faster_than_bls,
                recompute_nbls=recompute_nbls,
            )

        # update filename attribute
        basename = os.path.basename(filename)
        self.filename = [basename]
        self._filename.form = (1,)

        # open hdf5 file for reading
        self._read_header(
            meta,
            default_mount_type=default_mount_type,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            background_lsts=background_lsts,
            astrometry_library=astrometry_library,
        )

        if read_data:
            # Now read in the data
            self._get_data(
                meta.datagrp,
                antenna_nums=antenna_nums,
                antenna_names=antenna_names,
                ant_str=ant_str,
                bls=bls,
                frequencies=frequencies,
                freq_chans=freq_chans,
                spws=spws,
                times=times,
                time_range=time_range,
                lsts=lsts,
                lst_range=lst_range,
                polarizations=polarizations,
                blt_inds=blt_inds,
                phase_center_ids=phase_center_ids,
                catalog_names=catalog_names,
                data_array_dtype=data_array_dtype,
                keep_all_metadata=keep_all_metadata,
                multidim_index=multidim_index,
            )
        if close_meta:
            meta.close()

        # Finally, backfill the apparent coords if they aren't in the original datafile
        add_app_coords = (
            self.phase_center_app_ra is None
            or (self.phase_center_app_dec is None)
            or (self.phase_center_frame_pa is None)
        )
        if add_app_coords:
            self._set_app_coords_helper()

        # Default behavior for UVH5 is to fix phasing if the problem is
        # detected, since the absence of the app coord attributes is the most
        # clear indicator of the old phasing algorithm being used. Double-check
        # the multi-phase-ctr attribute just to be extra safe.
        old_phase_compatible, _ = self._old_phase_attributes_compatible()
        if np.any(~self._check_for_cat_type("unprojected")) and old_phase_compatible:
            if (fix_old_proj) or (fix_old_proj is None and add_app_coords):
                self.fix_phase(use_ant_pos=fix_use_ant_pos)
            elif add_app_coords:
                warnings.warn(
                    "This data appears to have been phased-up using the old "
                    "`phase` method, which is incompatible with the current set of "
                    "methods. Please run the `fix_phase` method (or set "
                    "`fix_old_proj=True` when loading the dataset) to address this "
                    "issue."
                )

        if remove_flex_pol:
            self.remove_flex_pol()

        # check if object has all required UVParameters set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )

        return

    def _write_header(self, header):
        """
        Write data to the header datagroup of a UVH5 file.

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
        header["version"] = np.bytes_("1.2")

        # write out telescope and source information
        self.telescope.write_hdf5_header(header)

        # write out required UVParameters
        header["Nants_data"] = self.Nants_data
        header["Nbls"] = self.Nbls
        header["Nblts"] = self.Nblts
        header["Nfreqs"] = self.Nfreqs
        header["Npols"] = self.Npols
        header["Nspws"] = self.Nspws
        header["Ntimes"] = self.Ntimes
        header["uvw_array"] = self.uvw_array
        header["vis_units"] = np.bytes_(self.vis_units)
        header["channel_width"] = self.channel_width
        header["time_array"] = self.time_array
        header["freq_array"] = self.freq_array
        header["integration_time"] = self.integration_time
        header["lst_array"] = self.lst_array
        header["polarization_array"] = self.polarization_array
        header["spw_array"] = self.spw_array
        header["ant_1_array"] = self.ant_1_array
        header["ant_2_array"] = self.ant_2_array
        # handle antenna_names; works for lists or arrays

        # write out phasing information
        # Write out the catalog, if available
        header["phase_center_id_array"] = self.phase_center_id_array
        header["Nphase"] = self.Nphase
        # this is a dict of dicts. Top level key is the phase_center_id,
        # next level keys give details for each phase center.
        pc_group = header.create_group("phase_center_catalog")
        for pc, pc_dict in self.phase_center_catalog.items():
            this_group = pc_group.create_group(str(pc))
            for key, value in pc_dict.items():
                if isinstance(value, str):
                    this_group[key] = np.bytes_(value)
                elif value is None:
                    this_group[key] = h5py.Empty("f")
                else:
                    this_group[key] = value
        header["phase_center_app_ra"] = self.phase_center_app_ra
        header["phase_center_app_dec"] = self.phase_center_app_dec
        header["phase_center_frame_pa"] = self.phase_center_frame_pa

        # write out optional parameters
        if self.dut1 is not None:
            header["dut1"] = self.dut1
        if self.earth_omega is not None:
            header["earth_omega"] = self.earth_omega
        if self.gst0 is not None:
            header["gst0"] = self.gst0
        if self.rdate is not None:
            header["rdate"] = np.bytes_(self.rdate)
        if self.timesys is not None:
            header["timesys"] = np.bytes_(self.timesys)
        if self.blt_order is not None:
            header["blt_order"] = np.bytes_(", ".join(self.blt_order))
        if self.uvplane_reference_time is not None:
            header["uvplane_reference_time"] = self.uvplane_reference_time
        if self.eq_coeffs is not None:
            header["eq_coeffs"] = self.eq_coeffs
        if self.eq_coeffs_convention is not None:
            header["eq_coeffs_convention"] = np.bytes_(self.eq_coeffs_convention)
        if self.flex_spw_id_array is not None:
            header["flex_spw_id_array"] = self.flex_spw_id_array
        if self.flex_spw_polarization_array is not None:
            header["flex_spw_polarization_array"] = self.flex_spw_polarization_array

        if self.blts_are_rectangular is not None:
            header["blts_are_rectangular"] = self.blts_are_rectangular
        if self.time_axis_faster_than_bls is not None:
            header["time_axis_faster_than_bls"] = self.time_axis_faster_than_bls
        if self.pol_convention is not None:
            header["pol_convention"] = np.bytes_(self.pol_convention)

        # write out extra keywords if it exists and has elements
        if self.extra_keywords:
            extra_keywords = header.create_group("extra_keywords")
            for k in self.extra_keywords:
                if isinstance(self.extra_keywords[k], str):
                    extra_keywords[k] = np.bytes_(self.extra_keywords[k])
                elif self.extra_keywords[k] is None:
                    # save as empty/null dataset
                    extra_keywords[k] = h5py.Empty("f")
                else:
                    extra_keywords[k] = self.extra_keywords[k]

        # write out history
        header["history"] = np.bytes_(self.history)

        return

    def write_uvh5(
        self,
        filename,
        *,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
        data_write_dtype=None,
        add_to_history=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=False,
    ):
        """
        Write an in-memory UVData object to a UVH5 file.

        Parameters
        ----------
        filename : str
            The UVH5 file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression). In addition to the normal HDF5 filter values, the
            user may specify "bitshuffle" which will set the compression to `32008` for
            bitshuffle and will set the `compression_opts` to `(0, 2)` to allow
            bitshuffle to automatically determine the block size and to use the LZF
            filter after bitshuffle. Using `bitshuffle` requires having the
            `hdf5plugin` package installed.  Dataset must be chunked to use compression.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is the
            LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is the
            LZF filter. Dataset must be chunked.
        data_write_dtype : numpy dtype
            The datatype of output visibility data. If 'None', then the same
            datatype as data_array will be used. The user may specify 'c8' for
            single-precision floats or 'c16' for double-presicion. Otherwise, a
            numpy dtype object must be specified with an 'r' field and an 'i'
            field for real and imaginary parts, respectively. See uvh5.py for
            an example of defining such a datatype.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is False.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the file located at `filename` already exists and clobber=False,
            an IOError is raised.

        Notes
        -----
        The HDF5 library allows for the application of "filters" when writing
        data, which can provide moderate to significant levels of compression
        for the datasets in question.  Testing has shown that for some typical
        cases of UVData objects (empty/sparse flag_array objects, and/or uniform
        nsample_arrays), the built-in LZF filter provides significant
        compression for minimal computational overhead.

        Note that for typical HERA data files written after mid-2020, the
        bitshuffle filter was applied to the data_array. Because of the lack of
        portability, it is not included as an option here; in the future, it may
        be added. Note that as long as bitshuffle is installed on the system in
        a way that h5py can find it, no action needs to be taken to _read_ a
        data_array encoded with bitshuffle (or an error will be raised).
        """
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )

        if os.path.exists(filename):
            if clobber:
                print("File exists; clobbering")
            else:
                raise OSError("File exists; skipping")

        data_compression, data_compression_opts = hdf5_utils._get_compression(
            data_compression
        )

        # open file for writing
        with h5py.File(filename, "w") as f:
            # write header
            header = f.create_group("Header")
            self._write_header(header)

            # write out data, flags, and nsample arrays
            dgrp = f.create_group("Data")
            if data_write_dtype is None:
                if self.data_array.dtype == "complex64":
                    data_write_dtype = "c8"
                else:
                    data_write_dtype = "c16"
            if data_write_dtype not in ("c8", "c16"):
                hdf5_utils._check_complex_dtype(data_write_dtype)
                visdata = dgrp.create_dataset(
                    "visdata",
                    self.data_array.shape,
                    chunks=chunks,
                    compression=data_compression,
                    compression_opts=data_compression_opts,
                    dtype=data_write_dtype,
                )
                indices = (np.s_[:], np.s_[:], np.s_[:])
                hdf5_utils._write_complex_astype(self.data_array, visdata, indices)
            else:
                visdata = dgrp.create_dataset(
                    "visdata",
                    chunks=chunks,
                    data=self.data_array,
                    compression=data_compression,
                    compression_opts=data_compression_opts,
                    dtype=data_write_dtype,
                )
            dgrp.create_dataset(
                "flags",
                chunks=chunks,
                data=self.flag_array,
                compression=flags_compression,
            )
            dgrp.create_dataset(
                "nsamples",
                chunks=chunks,
                data=self.nsample_array.astype(np.float32),
                compression=nsample_compression,
            )

        return

    def initialize_uvh5_file(
        self,
        filename,
        *,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
        data_write_dtype=None,
    ):
        """
        Initialize a UVH5 file on disk to be written to in parts.

        Parameters
        ----------
        filename : str
            The UVH5 file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression). In addition to the normal HDF5 filter values, the
            user may specify "bitshuffle" which will set the compression to `32008` for
            bitshuffle and will set the `compression_opts` to `(0, 2)` to allow
            bitshuffle to automatically determine the block size and to use the LZF
            filter after bitshuffle. Using `bitshuffle` requires having the
            `hdf5plugin` package installed.  Dataset must be chunked to use compression.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is the
            LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is the
            LZF filter. Dataset must be chunked.
        data_write_dtype : str or numpy dtype
            The datatype of output visibility data. If 'None', then double-
            precision floats will be used. The user may specify 'c8' for
            single-precision floats or 'c16' for double-presicion. Otherwise, a
            numpy dtype object must be specified with an 'r' field and an 'i'
            field for real and imaginary parts, respectively. See uvh5.py for
            an example of defining such a datatype.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the file located at `filename` already exists and clobber=False,
            an IOError is raised.

        Notes
        -----
        When partially writing out data, this function should be called first to
        initialize the file on disk. The data is then actually written by
        calling the write_uvh5_part method, with the same filename as the one
        specified in this function. See the tutorial for a worked example.

        The HDF5 library allows for the application of "filters" when writing
        data, which can provide moderate to significant levels of compression
        for the datasets in question.  Testing has shown that for some typical
        cases of UVData objects (empty/sparse flag_array objects, and/or uniform
        nsample_arrays), the built-in LZF filter provides significant
        compression for minimal computational overhead.

        Note that for typical HERA data files written after mid-2018, the
        bitshuffle filter was applied to the data_array. Because of the lack of
        portability, it is not included as an option here; in the future, it may
        be added. Note that as long as bitshuffle is installed on the system in
        a way that h5py can find it, no action needs to be taken to _read_ a
        data_array encoded with bitshuffle (or an error will be raised).
        """
        if os.path.exists(filename):
            if clobber:
                print("File exists; clobbering")
            else:
                raise OSError("File exists; skipping")

        data_compression, data_compression_opts = hdf5_utils._get_compression(
            data_compression
        )

        # write header and empty arrays to file
        with h5py.File(filename, "w") as f:
            # write header
            header = f.create_group("Header")
            self._write_header(header)

            # initialize the data groups on disk
            data_size = (self.Nblts, self.Nfreqs, self.Npols)
            dgrp = f.create_group("Data")
            if data_write_dtype is None:
                # we don't know what kind of data we'll get--default to double-precision
                data_write_dtype = "c16"
            if data_write_dtype not in ("c8", "c16"):
                # make sure the data type is correct
                hdf5_utils._check_complex_dtype(data_write_dtype)
            dgrp.create_dataset(
                "visdata",
                data_size,
                chunks=chunks,
                dtype=data_write_dtype,
                compression=data_compression,
                compression_opts=data_compression_opts,
            )
            dgrp.create_dataset(
                "flags",
                data_size,
                chunks=chunks,
                dtype="b1",
                compression=flags_compression,
            )
            dgrp.create_dataset(
                "nsamples",
                data_size,
                chunks=chunks,
                dtype="f4",
                compression=nsample_compression,
            )

        return

    def _check_header(
        self, filename, *, run_check_acceptability=True, background_lsts=True
    ):
        """
        Check that the metadata in a file header matches the object's metadata.

        Parameters
        ----------
        header : h5py datagroup
            A reference to an h5py data group that contains the header
            information. For UVH5 files conforming to the spec, this should be
            "/Header".
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.

        Returns
        -------
        None

        Notes
        -----
        This function creates a new UVData object an reads in the header
        information saved on disk to compare with the object in memory. Note
        that this adds some small memory overhead, but this amount is typically
        much smaller than the size of the data.
        """
        uvd_file = UVH5()
        with h5py.File(filename, "r") as f:
            header = f["/Header"]
            uvd_file._read_header(
                header,
                run_check_acceptability=run_check_acceptability,
                background_lsts=background_lsts,
            )

        # temporarily remove data, flag, and nsample arrays, so we only check metadata
        if self.data_array is not None:
            data_array = self.data_array
            self.data_array = None
            replace_data = True
        else:
            replace_data = False
        if self.flag_array is not None:
            flag_array = self.flag_array
            self.flag_array = None
            replace_flags = True
        else:
            replace_flags = False
        if self.nsample_array is not None:
            nsample_array = self.nsample_array
            self.nsample_array = None
            replace_nsamples = True
        else:
            replace_nsamples = False

        # also ignore filename attribute
        uvd_file.filename = self.filename
        uvd_file._filename.form = self._filename.form

        if self != uvd_file:
            raise AssertionError(
                "The object metadata in memory and metadata on disk are different"
            )
        else:
            # clean up after ourselves
            if replace_data:
                self.data_array = data_array
            if replace_flags:
                self.flag_array = flag_array
            if replace_nsamples:
                self.nsample_array = nsample_array
            del uvd_file
        return

    def write_uvh5_part(
        self,
        filename,
        *,
        data_array,
        flag_array,
        nsample_array,
        check_header=True,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        blt_inds=None,
        phase_center_ids=None,
        catalog_names=None,
        run_check_acceptability=True,
        add_to_history=None,
    ):
        """
        Write out a part of a UVH5 file that has been previously initialized.

        Parameters
        ----------
        filename : str
            The file on disk to write data to. It must already exist,
            and is assumed to have been initialized with initialize_uvh5_file.
        data_array : array of float
            The data to write to disk. A check is done to ensure that
            the dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        flag_array : array of bool
            The flags array to write to disk. A check is done to ensure
            that the dimensions of the data passed in conform to the ones specified
            by the "selection" arguments.
        nsample_array : array of float
            The nsample array to write to disk. A check is done to ensure
            that the dimensions fo the data passed in conform to the ones specified
            by the "selection" arguments.
        check_header : bool
            Option to check that the metadata present in the header
            on disk matches that in the object.
        run_check_acceptability : bool
            If check_header, additional option to check
            acceptable range of the values of parameters after reading in the file.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when writing data into
            the object (antenna positions and names for the excluded antennas
            will be retained). This cannot be provided if antenna_names is
            also provided.
        antenna_names : array_like of str, optional
            The antennas names to include when writing data into
            the object (antenna positions and names for the excluded antennas
            will be retained). This cannot be provided if antenna_nums is
            also provided.
        bls : list of tuples, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to write to the file. For length-2 tuples, the ordering of the numbers
            within the tuple does not matter. For length-3 tuples, the polarization
            string is in the order of the two antennas. If length-3 tuples are provided,
            the polarizations argument below must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include when writing data into the object.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').
            See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be written for both baselines (1, 2) and (2, 3) to reflect a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of the above antenna
            args or the polarizations arg.
        frequencies : array_like of float, optional
            The frequencies to include when writing data to the file.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when writing data to the file.
        spws : array_like of int, optional
            The spectral window numbers to keep in the file.
        times : array_like of float, optional
            The times in Julian Day to include when writing data to the file.
        time_range : array_like of float, optional
            The time range in Julian Date to include when writing data to the
            file, must be length 2. Some of the times in the object should fall
            between the first and last elements. Cannot be used with `times`.
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
        polarizations : array_like of int, optional
            The polarizations to include when writing data to the file.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when writing data to the file.
            This is not commonly used.
        phase_center_ids : array_like of int, optional
            Phase center IDs to include when reading data into the object (effectively
            a selection on baseline-times). Cannot be used with catalog_names.
        catalog_names : str or array-like of str
            The names of the phase centers (sources) to include when reading data into
            the object, which should match exactly in spelling and capitalization.
            Cannot be used with phase_center_ids.
        add_to_history : str
            String to append to history before write out. Default is no appending.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            An AsserionError is raised if: (1) the location specified by
            `filename` does not exist; (2) the data_array, flag_array, and
            nsample_array do not all have the same shape; (3) the shape of the
            data arrays do not correspond to the sizes specified by the
            properties to write out.

        Notes
        -----
        When partially writing out data, this function should be called after
        calling initialize_uvh5_file. The same filename is passed in, with an
        optional check to ensure that the object's metadata in-memory matches
        the header on-disk. See the tutorial for a worked example.
        """
        # check that the file already exists
        if not os.path.exists(filename):
            raise AssertionError(
                f"{filename} does not exists; please first initialize it with "
                "initialize_uvh5_file"
            )

        if check_header:
            self._check_header(
                filename, run_check_acceptability=run_check_acceptability
            )

        # figure out which "full file" indices to write data to
        blt_inds, freq_inds, _, pol_inds, _ = self._select_preprocess(
            antenna_nums=antenna_nums,
            antenna_names=antenna_names,
            ant_str=ant_str,
            bls=bls,
            frequencies=frequencies,
            freq_chans=freq_chans,
            spws=spws,
            times=times,
            time_range=time_range,
            lsts=lsts,
            lst_range=lst_range,
            polarizations=polarizations,
            blt_inds=blt_inds,
            phase_center_ids=phase_center_ids,
            catalog_names=catalog_names,
        )

        # make sure that the dimensions of the data to write are correct
        if data_array.shape != flag_array.shape:
            raise AssertionError("data_array and flag_array must have the same shape")
        if data_array.shape != nsample_array.shape:
            raise AssertionError(
                "data_array and nsample_array must have the same shape"
            )

        # check what part of each dimension to grab
        # we can use numpy slice objects to index the h5py indices
        if blt_inds is not None:
            Nblts = len(blt_inds)

            # test if blts are regularly spaced
            if len(set(np.ediff1d(blt_inds))) <= 1:
                blt_reg_spaced = True
                blt_start = blt_inds[0]
                blt_end = blt_inds[-1] + 1
                if len(blt_inds) == 1:
                    d_blt = 1
                else:
                    d_blt = blt_inds[1] - blt_inds[0]
                blt_inds = np.s_[blt_start:blt_end:d_blt]
            else:
                blt_reg_spaced = False
        else:
            Nblts = self.Nblts
            blt_reg_spaced = True
            blt_inds = np.s_[:]
        if freq_inds is not None:
            Nfreqs = len(freq_inds)

            # test if frequencies are regularly spaced
            if len(set(np.ediff1d(freq_inds))) <= 1:
                freq_reg_spaced = True
                freq_start = freq_inds[0]
                freq_end = freq_inds[-1] + 1
                if len(freq_inds) == 1:
                    d_freq = 1
                else:
                    d_freq = freq_inds[1] - freq_inds[0]
                freq_inds = np.s_[freq_start:freq_end:d_freq]
            else:
                freq_reg_spaced = False
        else:
            Nfreqs = self.Nfreqs
            freq_reg_spaced = True
            freq_inds = np.s_[:]
        if pol_inds is not None:
            Npols = len(pol_inds)

            # test if pols are regularly spaced
            if len(set(np.ediff1d(pol_inds))) <= 1:
                pol_reg_spaced = True
                pol_start = pol_inds[0]
                pol_end = pol_inds[-1] + 1
                if len(pol_inds) == 1:
                    d_pol = 1
                else:
                    d_pol = pol_inds[1] - pol_inds[0]
                pol_inds = np.s_[pol_start:pol_end:d_pol]
            else:
                pol_reg_spaced = False
        else:
            Npols = self.Npols
            pol_reg_spaced = True
            pol_inds = np.s_[:]

        # check for proper size of input arrays
        proper_shape = (Nblts, Nfreqs, Npols)
        if data_array.shape != proper_shape:
            raise AssertionError(
                f"data_array has shape {data_array.shape}; was expecting {proper_shape}"
            )

        # actually write the data
        with h5py.File(filename, "r+") as f:
            dgrp = f["/Data"]
            visdata_dset = dgrp["visdata"]
            flags_dset = dgrp["flags"]
            nsamples_dset = dgrp["nsamples"]
            visdata_dtype = visdata_dset.dtype
            if visdata_dtype not in ("complex64", "complex128"):
                custom_dtype = True
            else:
                custom_dtype = False

            # check if we can do fancy indexing
            # as long as at least 2 out of 3 axes can be written as slices,
            # we can be fancy
            n_reg_spaced = np.count_nonzero(
                [blt_reg_spaced, freq_reg_spaced, pol_reg_spaced]
            )
            if n_reg_spaced >= 2:
                if custom_dtype:
                    indices = (blt_inds, freq_inds, pol_inds)
                    hdf5_utils._write_complex_astype(data_array, visdata_dset, indices)
                else:
                    visdata_dset[blt_inds, freq_inds, pol_inds] = data_array
                flags_dset[blt_inds, freq_inds, pol_inds] = flag_array
                nsamples_dset[blt_inds, freq_inds, pol_inds] = nsample_array
            elif n_reg_spaced == 1:
                # figure out which axis is regularly spaced
                if blt_reg_spaced:
                    for ifreq, freq_idx in enumerate(freq_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_inds, freq_idx, pol_idx)
                                hdf5_utils._write_complex_astype(
                                    data_array[:, ifreq, ipol], visdata_dset, indices
                                )
                            else:
                                visdata_dset[blt_inds, freq_idx, pol_idx] = data_array[
                                    :, ifreq, ipol
                                ]
                            flags_dset[blt_inds, freq_idx, pol_idx] = flag_array[
                                :, ifreq, ipol
                            ]
                            nsamples_dset[blt_inds, freq_idx, pol_idx] = nsample_array[
                                :, ifreq, ipol
                            ]
                elif freq_reg_spaced:
                    for iblt, blt_idx in enumerate(blt_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_idx, freq_inds, pol_idx)
                                hdf5_utils._write_complex_astype(
                                    data_array[iblt, :, ipol], visdata_dset, indices
                                )
                            else:
                                visdata_dset[blt_idx, freq_inds, pol_idx] = data_array[
                                    iblt, :, ipol
                                ]
                            flags_dset[blt_idx, freq_inds, pol_idx] = flag_array[
                                iblt, :, ipol
                            ]
                            nsamples_dset[blt_idx, freq_inds, pol_idx] = nsample_array[
                                iblt, :, ipol
                            ]
                else:  # pol_reg_spaced
                    for iblt, blt_idx in enumerate(blt_inds):
                        for ifreq, freq_idx in enumerate(freq_inds):
                            if custom_dtype:
                                indices = (blt_idx, freq_idx, pol_inds)
                                hdf5_utils._write_complex_astype(
                                    data_array[iblt, ifreq, :], visdata_dset, indices
                                )
                            else:
                                visdata_dset[blt_idx, freq_idx, pol_inds] = data_array[
                                    iblt, ifreq, :
                                ]
                            flags_dset[blt_idx, freq_idx, pol_inds] = flag_array[
                                iblt, ifreq, :
                            ]
                            nsamples_dset[blt_idx, freq_idx, pol_inds] = nsample_array[
                                iblt, ifreq, :
                            ]
            else:
                # all axes irregularly spaced
                # perform a triple loop -- probably very slow!
                for iblt, blt_idx in enumerate(blt_inds):
                    for ifreq, freq_idx in enumerate(freq_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_idx, freq_idx, pol_idx)
                                hdf5_utils._write_complex_astype(
                                    data_array[iblt, ifreq, ipol], visdata_dset, indices
                                )
                            else:
                                visdata_dset[blt_idx, freq_idx, pol_idx] = data_array[
                                    iblt, ifreq, ipol
                                ]
                            flags_dset[blt_idx, freq_idx, pol_idx] = flag_array[
                                iblt, ifreq, ipol
                            ]
                            nsamples_dset[blt_idx, freq_idx, pol_idx] = nsample_array[
                                iblt, ifreq, ipol
                            ]

            # append to history if desired
            if add_to_history is not None:
                history = np.bytes_(self.history) + np.bytes_(add_to_history)
                if "history" in f["Header"]:
                    # erase dataset first b/c it has fixed-length string datatype
                    del f["Header"]["history"]
                f["Header"]["history"] = np.bytes_(history)

        return
