# Copyright (c) 2023 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading and writing calibration HDF5 files."""

from __future__ import annotations

import contextlib
import os
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
from docstring_parser import DocstringStyle

from .. import utils
from ..docstrings import copy_replace_short_description
from ..telescopes import Telescope
from ..utils.io import hdf5 as hdf5_utils
from .uvcal import UVCal

hdf5plugin_present = True
try:
    import hdf5plugin  # noqa: F401
except ImportError as error:
    hdf5plugin_present = False
    hdf5plugin_error = error


class FastCalH5Meta(hdf5_utils.HDF5Meta):
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
            "instrument",
            "version",
            "pol_convention",
        }
    )

    _defaults = {"x_orientation": None}

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

    def has_key(self, antnum: int | None = None, jpol: str | int | None = None) -> bool:
        """Check if the file has a given antenna number or antenna number-pol key."""
        if antnum is not None and antnum not in self.ant_array:
            return False
        if jpol is not None:
            if isinstance(jpol, str | np.str_):
                jpol = utils.jstr2num(jpol, x_orientation=self.x_orientation)
            if jpol not in self.jones_array:
                return False

        return True

    @cached_property
    def pols(self) -> list[str]:
        """The polarizations in the file, as standardized strings, eg. 'xx' or 'ee'."""
        return np.asarray(
            utils.jnum2str(self.jones_array, x_orientation=self.x_orientation)
        )

    def to_uvcal(
        self, *, check_lsts: bool = False, astrometry_library: str | None = None
    ) -> UVCal:
        """Convert the file to a UVData object.

        The object will be metadata-only.
        """
        uvc = UVCal()
        uvc.read_calh5(self, read_data=False, run_check_acceptability=check_lsts)
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
        meta: FastCalH5Meta,
        *,
        default_mount_type: str | None = None,
        background_lsts: bool = True,
        astrometry_library: str | None = None,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability=True,
    ):
        """
        Read header information from a UVH5 file.

        This is an internal function called by the user-space methods.
        Properties of the UVData object are updated as the file is processed.

        Parameters
        ----------
        meta : FastCalH5Meta, h5py.File or h5py.Group
            A FastCalH5Meta or object that contains the header information.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check_acceptability : bool
            Option to check that LSTs match the times given the telescope_location.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Returns
        -------
        None
        """
        # First, get the things relevant for setting LSTs, so that can be run in the
        # background if desired.
        required_telescope_keys = [
            "telescope_name",
            "latitude",
            "longitude",
            "altitude",
            "Nants_telescope",
            "antenna_names",
            "antenna_numbers",
            "antenna_positions",
        ]

        # Antenna diameters is an optional parameter _inside_ of the telescope attr,
        # so let's just check if it's there now and add it to the required keys if it is
        if "antenna_diameters" in meta.header:
            required_telescope_keys.append("antenna_diameters")

        self.telescope = Telescope.from_hdf5(
            meta,
            required_keys=required_telescope_keys,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        if "time_array" in meta.header:
            self.time_array = meta.time_array
            if "lst_array" in meta.header:
                self.lst_array = meta.header["lst_array"][:]
                proc = None
            else:
                proc = self.set_lsts_from_time_array(
                    background=background_lsts, astrometry_library=astrometry_library
                )
        if "time_range" in meta.header:
            self.time_range = meta.time_range
            if "lst_range" in meta.header:
                self.lst_range = meta.header["lst_range"][:]
                proc = None
            else:
                proc = self.set_lsts_from_time_array(
                    background=background_lsts, astrometry_library=astrometry_library
                )

        required_parameters = [
            "history",
            "Nfreqs",
            "Njones",
            "Nspws",
            "Ntimes",
            "Nants_data",
            "ant_array",
            "integration_time",
            "spw_array",
            "jones_array",
            "cal_style",
            "cal_type",
            "gain_convention",
            "wide_band",
        ]

        optional_parameters = [
            "channel_width",
            "flex_spw_id_array",
            "flex_jones_array",
            "Nphase",
            "Nsources",
            "baseline_range",
            "diffuse_model",
            "extra_keywords",
            "freq_array",
            "freq_range",
            "gain_scale",
            "git_hash_cal",
            "git_origin_cal",
            "observer",
            "phase_center_catalog",
            "phase_center_id_array",
            "ref_antenna_name",
            "ref_antenna_array",
            "scan_number_array",
            "sky_catalog",
            "pol_convention",
        ]

        for attr in required_parameters:
            try:
                setattr(self, attr, getattr(meta, attr))
            except AttributeError as e:
                raise KeyError(str(e)) from e

        if self.wide_band:
            self._set_wide_band()
            # Skip the flex_spw_id_array if we have a wide_band object, since older
            # versions allowed one to store this even if it wasn't actually being used
            optional_parameters.remove("flex_spw_id_array")

        if not utils.history._check_history_version(
            self.history, self.pyuvdata_version_str
        ):
            self.history += self.pyuvdata_version_str

        # Optional parameters
        for attr in optional_parameters:
            with contextlib.suppress(AttributeError):
                setattr(self, attr, getattr(meta, attr))

        # set any extra telescope params (and handle x-orientation)
        self.set_telescope_params(
            x_orientation=meta.x_orientation,
            mount_type=default_mount_type,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )

        # ensure LSTs are set before checking them.
        if proc is not None:
            proc.join()

        if run_check_acceptability:
            if self.time_array is not None:
                utils.times.check_lsts_against_times(
                    jd_array=self.time_array,
                    lst_array=self.lst_array,
                    telescope_loc=self.telescope.location,
                    lst_tols=(0, utils.LST_RAD_TOL),
                )
            if self.time_range is not None:
                utils.times.check_lsts_against_times(
                    jd_array=self.time_range,
                    lst_array=self.lst_range,
                    telescope_loc=self.telescope.location,
                    lst_tols=(0, utils.LST_RAD_TOL),
                )

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
        phase_center_ids,
        catalog_names,
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
        if self.cal_type == "gain":
            data_name = "gains"
        else:
            data_name = "delays"
        if (
            "32008" in dgrp[data_name]._filters and not hdf5plugin_present
        ):  # pragma: no cover
            raise ImportError(
                "hdf5plugin is not installed but is required to read this dataset"
            ) from hdf5plugin_error

        # figure out what data to read in
        (
            ant_inds,
            time_inds,
            spw_inds,
            freq_inds,
            jones_inds,
            history_update_string,
        ) = self._select_preprocess(
            antenna_nums=antenna_nums,
            antenna_names=antenna_names,
            frequencies=frequencies,
            freq_chans=freq_chans,
            spws=spws,
            times=times,
            time_range=time_range,
            lsts=lsts,
            lst_range=lst_range,
            jones=jones,
            phase_center_ids=phase_center_ids,
            catalog_names=catalog_names,
        )
        # figure out which axis is the most selective
        if ant_inds is not None:
            ant_frac = len(ant_inds) / float(self.Nants_data)
        else:
            ant_frac = 1

        if time_inds is not None:
            time_frac = len(time_inds) / float(self.Ntimes)
        else:
            time_frac = 1

        if freq_inds is not None and not self.wide_band:
            freq_frac = len(freq_inds) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if spw_inds is not None and self.wide_band:
            spw_frac = len(spw_inds) / float(self.Nspws)
        else:
            spw_frac = 1

        if jones_inds is not None:
            jones_frac = len(jones_inds) / float(self.Njones)
        else:
            jones_frac = 1

        min_frac = np.min([ant_frac, time_frac, freq_frac, jones_frac, spw_frac])

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
                self.gain_array = hdf5_utils._index_dset(dgrp["gains"], inds)
            else:
                self.delay_array = hdf5_utils._index_dset(dgrp["delays"], inds)
            self.flag_array = hdf5_utils._index_dset(dgrp["flags"], inds)
            if quality_present:
                self.quality_array = hdf5_utils._index_dset(dgrp["qualities"], inds)
            if total_quality_present:
                tq_inds = (np.s_[:], np.s_[:], np.s_[:])
                self.total_quality_array = hdf5_utils._index_dset(
                    dgrp["total_qualities"], tq_inds
                )
        else:
            # do select operations on everything except data_array, flag_array
            # and nsample_array
            self._select_by_index(
                ant_inds=ant_inds,
                time_inds=time_inds,
                spw_inds=spw_inds,
                freq_inds=freq_inds,
                jones_inds=jones_inds,
                history_update_string=history_update_string,
            )

            # determine which axes can be sliced, rather than fancy indexed
            # max_nslice_frac of 0.1 is just copied from uvh5, not validated
            # TODO: this logic is similar to what is in uvh5. See if an abstracted
            # version can be pulled out into a util function.
            if ant_inds is not None:
                ant_slices, ant_sliceable = utils.tools._convert_to_slices(
                    ant_inds, max_nslice_frac=0.1
                )
            else:
                ant_inds, ant_slices = np.s_[:], np.s_[:]
                ant_sliceable = True

            if time_inds is not None:
                time_slices, time_sliceable = utils.tools._convert_to_slices(
                    time_inds, max_nslice_frac=0.1
                )
            else:
                time_inds, time_slices = np.s_[:], np.s_[:]
                time_sliceable = True

            if freq_inds is not None:
                freq_slices, freq_sliceable = utils.tools._convert_to_slices(
                    freq_inds, max_nslice_frac=0.1
                )
            else:
                freq_inds, freq_slices = np.s_[:], np.s_[:]
                freq_sliceable = True

            if spw_inds is not None:
                spw_slices, spw_sliceable = utils.tools._convert_to_slices(
                    spw_inds, max_nslice_frac=0.1
                )
            else:
                spw_inds, spw_slices = np.s_[:], np.s_[:]
                spw_sliceable = True

            if jones_inds is not None:
                jones_slices, jones_sliceable = utils.tools._convert_to_slices(
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

            elif freq_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], freq_inds, np.s_[:], np.s_[:]]
                if freq_sliceable:
                    inds[1] = freq_slices

                inds = tuple(inds)

                # change freq_frac so no more selects are done
                freq_frac = 1

            elif spw_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], spw_inds, np.s_[:], np.s_[:]]
                if spw_sliceable:
                    inds[1] = spw_slices

                inds = tuple(inds)

                # change freq_frac so no more selects are done
                spw_frac = 1

            elif time_frac == min_frac:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], np.s_[:], time_inds, np.s_[:]]
                if time_sliceable:
                    inds[2] = time_slices

                inds = tuple(inds)

                # change time_frac so no more selects are done
                time_frac = 1
            else:
                # construct inds list given simultaneous sliceability
                inds = [np.s_[:], np.s_[:], np.s_[:], jones_inds]
                if jones_sliceable:
                    inds[3] = jones_slices

                inds = tuple(inds)

                # change jones_frac so no more selects are done
                jones_frac = 1

            # index datasets
            cal_data = hdf5_utils._index_dset(caldata_dset, inds)
            flags = hdf5_utils._index_dset(flags_dset, inds)
            if quality_present:
                qualities = hdf5_utils._index_dset(qualities_dset, inds)
            if total_quality_present:
                tq_inds = inds[1:]
                total_qualities = hdf5_utils._index_dset(total_qualities_dset, tq_inds)
            # down select on other dimensions if necessary
            # use indices not slices here: generally not the bottleneck
            if ant_frac < 1:
                cal_data = cal_data[ant_inds]
                flags = flags[ant_inds]
                if quality_present:
                    qualities = qualities[ant_inds]
            if freq_frac < 1:
                cal_data = cal_data[:, freq_inds]
                flags = flags[:, freq_inds]
                if quality_present:
                    qualities = qualities[:, freq_inds]
                if total_quality_present:
                    total_qualities = total_qualities[freq_inds]
            if spw_frac < 1:
                cal_data = cal_data[:, spw_inds]
                flags = flags[:, spw_inds]
                if quality_present:
                    qualities = qualities[:, spw_inds]
                if total_quality_present:
                    total_qualities = total_qualities[spw_inds]
            if time_frac < 1:
                cal_data = cal_data[:, :, time_inds]
                flags = flags[:, :, time_inds]
                if quality_present:
                    qualities = qualities[:, :, time_inds]
                if total_quality_present:
                    total_qualities = total_qualities[:, time_inds]
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
                self.delay_array = cal_data
            self.flag_array = flags
            if quality_present:
                self.quality_array = qualities
            if total_quality_present:
                self.total_quality_array = total_qualities

        return

    @copy_replace_short_description(UVCal.read_calh5, style=DocstringStyle.NUMPYDOC)
    def read_calh5(
        self,
        filename: str | Path | FastCalH5Meta,
        *,
        antenna_nums=None,
        antenna_names=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        jones=None,
        phase_center_ids=None,
        catalog_names=None,
        read_data=True,
        gain_array_dtype=np.complex128,
        default_mount_type="other",
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        astrometry_library=None,
    ):
        """Read in data from a CalH5 file."""
        if isinstance(filename, FastCalH5Meta):
            meta = filename
            filename = str(meta.path)
            close_meta = False
        else:
            close_meta = True
            meta = FastCalH5Meta(filename)

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
                frequencies=frequencies,
                freq_chans=freq_chans,
                spws=spws,
                times=times,
                time_range=time_range,
                lsts=lsts,
                lst_range=lst_range,
                jones=jones,
                gain_array_dtype=gain_array_dtype,
                phase_center_ids=phase_center_ids,
                catalog_names=catalog_names,
            )

        if close_meta:
            meta.close()

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
        header["version"] = np.bytes_("0.1")

        # write out telescope and source information
        self.telescope.write_hdf5_header(header)

        # write out required UVParameters
        header["Nants_data"] = self.Nants_data
        header["Nfreqs"] = self.Nfreqs
        header["Njones"] = self.Njones
        header["Nspws"] = self.Nspws
        header["Ntimes"] = self.Ntimes
        header["integration_time"] = self.integration_time
        header["jones_array"] = self.jones_array
        header["spw_array"] = self.spw_array
        header["ant_array"] = self.ant_array
        header["cal_type"] = np.bytes_(self.cal_type)
        header["cal_style"] = np.bytes_(self.cal_style)
        header["gain_convention"] = np.bytes_(self.gain_convention)
        header["wide_band"] = self.wide_band

        # write out optional parameters
        if self.channel_width is not None:
            header["channel_width"] = self.channel_width
        if self.flex_spw_id_array is not None:
            header["flex_spw_id_array"] = self.flex_spw_id_array
        if self.flex_jones_array is not None:
            header["flex_jones_array"] = self.flex_jones_array

        if self.time_array is not None:
            header["time_array"] = self.time_array
        if self.time_range is not None:
            header["time_range"] = self.time_range
        if self.lst_array is not None:
            header["lst_array"] = self.lst_array
        if self.lst_range is not None:
            header["lst_range"] = self.lst_range
        if self.scan_number_array is not None:
            header["scan_number_array"] = self.scan_number_array

        if self.ref_antenna_array is not None:
            header["ref_antenna_array"] = self.ref_antenna_array

        if self.Nsources is not None:
            header["Nsources"] = self.Nsources
        if self.baseline_range is not None:
            header["baseline_range"] = self.baseline_range
        if self.diffuse_model is not None:
            header["diffuse_model"] = np.bytes_(self.diffuse_model)
        if self.freq_array is not None:
            header["freq_array"] = self.freq_array
        if self.freq_range is not None:
            header["freq_range"] = self.freq_range
        if self.gain_scale is not None:
            header["gain_scale"] = np.bytes_(self.gain_scale)
        if self.git_hash_cal is not None:
            header["git_hash_cal"] = np.bytes_(self.git_hash_cal)
        if self.git_origin_cal is not None:
            header["git_origin_cal"] = np.bytes_(self.git_origin_cal)
        if self.observer is not None:
            header["observer"] = np.bytes_(self.observer)
        if self.ref_antenna_name is not None:
            header["ref_antenna_name"] = np.bytes_(self.ref_antenna_name)
        if self.sky_catalog is not None:
            header["sky_catalog"] = np.bytes_(self.sky_catalog)
        if self.phase_center_id_array is not None:
            header["phase_center_id_array"] = self.phase_center_id_array
        if self.Nphase is not None:
            header["Nphase"] = self.Nphase
        if self.pol_convention is not None:
            header["pol_convention"] = np.bytes_(self.pol_convention)

        if self.phase_center_catalog is not None:
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

    @copy_replace_short_description(UVCal.write_calh5, style=DocstringStyle.NUMPYDOC)
    def write_calh5(
        self,
        filename,
        *,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        quality_compression="lzf",
        add_to_history=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Write an in-memory UVCal object to a CalH5 file."""
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
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
            if self.cal_type == "gain":
                dgrp.create_dataset(
                    "gains",
                    chunks=chunks,
                    data=self.gain_array,
                    compression=data_compression,
                    compression_opts=data_compression_opts,
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

        return
