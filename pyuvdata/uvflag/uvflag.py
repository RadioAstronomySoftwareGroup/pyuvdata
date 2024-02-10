# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Primary container for radio interferometer flag manipulation."""
import copy
import os
import pathlib
import threading
import warnings

import h5py
import numpy as np

from .. import parameter as uvp
from .. import telescopes as uvtel
from .. import utils as uvutils
from ..uvbase import UVBase
from ..uvcal import UVCal
from ..uvdata import UVData

__all__ = ["UVFlag", "flags2waterfall", "and_rows_cols"]


_future_array_shapes_warning = (
    "The shapes of several attributes will be changing in the future to remove the "
    "deprecated spectral window axis. You can call the `use_future_array_shapes` "
    "method to convert to the future array shapes now or set the parameter of the same "
    "name on this method to both convert to the future array shapes and silence this "
    "warning."
)


def and_rows_cols(waterfall):
    """Perform logical and over rows and cols of a waterfall.

    For a 2D flag waterfall, flag pixels only if fully flagged along
    time and/or frequency

    Parameters
    ----------
    waterfall : 2D boolean array of shape (Ntimes, Nfreqs)

    Returns
    -------
    wf : 2D array
        A 2D array (size same as input) where only times/integrations
        that were fully flagged are flagged.

    """
    wf = np.zeros_like(waterfall, dtype=np.bool_)
    Ntimes, Nfreqs = waterfall.shape
    wf[:, (np.sum(waterfall, axis=0) / Ntimes) == 1] = True
    wf[(np.sum(waterfall, axis=1) / Nfreqs) == 1] = True
    return wf


def flags2waterfall(uv, flag_array=None, keep_pol=False):
    """Convert a flag array to a 2D waterfall of dimensions (Ntimes, Nfreqs).

    Averages over baselines and polarizations (in the case of visibility data),
    or antennas and jones parameters (in case of calibrationd data).

    Parameters
    ----------
    uv : A UVData or UVCal object
        Object defines the times and frequencies, and supplies the
        flag_array to convert (if flag_array not specified)
    flag_array :  Optional,
        flag array to convert instead of uv.flag_array.
        Must have same dimensions as uv.flag_array.
    keep_pol : bool
        Option to keep the polarization axis intact.

    Returns
    -------
    waterfall : 2D array or 3D array
        Waterfall of averaged flags, for example fraction of baselines
        which are flagged for every time and frequency (in case of UVData input)
        Size is (Ntimes, Nfreqs) or (Ntimes, Nfreqs, Npols).

    """
    if not isinstance(uv, (UVData, UVCal)):
        raise ValueError(
            "flags2waterfall() requires a UVData or UVCal object as "
            "the first argument."
        )
    if flag_array is None:
        flag_array = uv.flag_array
    if uv.flag_array.shape != flag_array.shape:
        raise ValueError("Flag array must align with UVData or UVCal object.")

    if isinstance(uv, UVCal):
        mean_axis = [0]
        if not uv.future_array_shapes:
            mean_axis.append(1)
        if not keep_pol:
            if uv.future_array_shapes:
                mean_axis.append(3)
            else:
                mean_axis.append(4)

        mean_axis = tuple(mean_axis)
        if keep_pol:
            waterfall = np.swapaxes(np.mean(flag_array, axis=mean_axis), 0, 1)
        else:
            waterfall = np.mean(flag_array, axis=mean_axis).T
    else:
        mean_axis = [0]
        if not uv.future_array_shapes:
            mean_axis.append(1)
        if not keep_pol:
            if uv.future_array_shapes:
                mean_axis.append(2)
            else:
                mean_axis.append(3)

        mean_axis = tuple(mean_axis)
        if keep_pol:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs, uv.Npols))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(
                    flag_array[uv.time_array == t], axis=mean_axis
                )
        else:
            waterfall = np.zeros((uv.Ntimes, uv.Nfreqs))
            for i, t in enumerate(np.unique(uv.time_array)):
                waterfall[i, :] = np.mean(
                    flag_array[uv.time_array == t], axis=mean_axis
                )

    return waterfall


class UVFlag(UVBase):
    """Object to handle flag arrays and waterfalls for interferometric datasets.

    Supports reading/writing, and stores all relevant information to combine
    flags and apply to data.
    Initialization of the UVFlag object requires some parameters. Metadata is
    copied from indata object. If indata is subclass of UVData or UVCal,
    the weights_array will be set to all ones.
    Lists or tuples are iterated through, treating each entry with an
    individual UVFlag init.

    Parameters
    ----------
    indata : UVData, UVCal, str, pathlib.Path, list of compatible combination
        Input to initialize UVFlag object. If str, assumed to be path to previously
        saved UVFlag object. UVData and UVCal objects cannot be directly combined,
        unless waterfall is True.
    mode : {"metric", "flag"}, optional
        The mode determines whether the object has a floating point metric_array
        or a boolean flag_array.
    copy_flags : bool, optional
        Whether to copy flags from indata to new UVFlag object
    waterfall : bool, optional
        Whether to immediately initialize as a waterfall object, with flag/metric
        axes: time, frequency, polarization.
    history : str, optional
        History string to attach to object.
    extra_keywords : dict, optional
        A dictionary of metadata values not explicitly specified by another
        parameter.
    label: str, optional
        String used for labeling the object (e.g. 'FM').
    telescope_name : str, optional
        Name of the telescope. Only used if `indata` is a string or pathlib.Path object.
        This should only be set when reading in old uvflag files
        that do not have the telescope name in them. Setting this parameter for old
        files allows for other telescope metadata to be set from the known
        telescopes. Setting this parameter overrides any telescope name in the file.
    use_future_array_shapes : bool
        Option to convert to the future planned array shapes before the changes go
        into effect by removing the spectral window axis.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        after creating UVFlag object.
    check_extra : bool
        Option to check optional parameters as well as required ones (the
        default is True, meaning the optional parameters will be checked).
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters after
        creating UVFlag object.

    Attributes
    ----------
    UVParameter objects :
        For full list see the UVFlag Parameters Documentation.
        (https://pyuvdata.readthedocs.io/en/latest/uvflag_parameters.html)
        Some are always required, some are required for certain phase_types
        and others are always optional.


    """

    def __init__(
        self,
        indata=None,
        mode="metric",
        copy_flags=False,
        waterfall=False,
        history="",
        label="",
        telescope_name=None,
        mwa_metafits_file=None,
        use_future_array_shapes=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Initialize the object."""
        desc = (
            "The mode determines whether the object has a "
            "floating point metric_array or a boolean flag_array. "
            'Options: {"metric", "flag"}. Default is "metric".'
        )
        self._mode = uvp.UVParameter(
            "mode",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["metric", "flag"],
        )

        desc = (
            "String used for labeling the object (e.g. 'FM'). "
            "Default is empty string."
        )
        self._label = uvp.UVParameter(
            "label", description=desc, form="str", expected_type=str
        )

        desc = (
            "The type of object defines the form of some arrays "
            "and also how metrics/flags are combined. "
            "Accepted types:'waterfall', 'baseline', 'antenna'"
        )
        self._type = uvp.UVParameter(
            "type",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["antenna", "baseline", "waterfall"],
        )

        self._Ntimes = uvp.UVParameter(
            "Ntimes", description="Number of times", expected_type=int
        )
        desc = "Number of baselines. Only Required for 'baseline' type objects."
        self._Nbls = uvp.UVParameter(
            "Nbls", description=desc, expected_type=int, required=False
        )
        self._Nblts = uvp.UVParameter(
            "Nblts",
            description="Number of baseline-times "
            "(i.e. number of spectra). Not necessarily "
            "equal to Nbls * Ntimes",
            expected_type=int,
        )
        self._Nspws = uvp.UVParameter(
            "Nspws",
            description="Number of spectral windows "
            "(ie non-contiguous spectral chunks).",
            expected_type=int,
        )
        self._Nfreqs = uvp.UVParameter(
            "Nfreqs", description="Number of frequency channels", expected_type=int
        )
        self._Npols = uvp.UVParameter(
            "Npols", description="Number of polarizations", expected_type=int
        )

        desc = (
            "Floating point metric information, only available in metric mode. "
            "The shape depends on the `type` parameter and on the "
            "`future_array_shapes` parameter. For 'baseline' type objects, "
            "the shape is (Nblts, 1, Nfreq, Npols) or (Nblts, Nfreqs, Npols) if "
            "future_array_shapes=True. For 'antenna' type objects, the shape is "
            "(Nants_data, 1, Nfreqs, Ntimes, Npols) or "
            "(Nants_data, Nfreqs, Ntimes, Npols) if future_array_shapes=True. "
            "For 'waterfall' type objects, the shape is (Ntimes, Nfreq, Npols)."
        )
        self._metric_array = uvp.UVParameter(
            "metric_array",
            description=desc,
            form=("Nblts", 1, "Nfreqs", "Npols"),
            expected_type=float,
            required=False,
        )

        desc = (
            "Boolean flag, True is flagged, only available in flag mode. "
            "The shape depends on the `type` parameter and on the "
            "`future_array_shapes` parameter. For 'baseline' type objects, "
            "the shape is (Nblts, 1, Nfreq, Npols) or (Nblts, Nfreqs, Npols) if "
            "future_array_shapes=True. For 'antenna' type objects, the shape is "
            "(Nants_data, 1, Nfreqs, Ntimes, Npols) or "
            "(Nants_data, Nfreqs, Ntimes, Npols) if future_array_shapes=True. "
            "For 'waterfall' type objects, the shape is (Ntimes, Nfreq, Npols)."
        )
        self._flag_array = uvp.UVParameter(
            "flag_array",
            description=desc,
            form=("Nblts", 1, "Nfreqs", "Npols"),
            expected_type=bool,
            required=False,
        )

        desc = (
            "Floating point weight information, only available in metric mode."
            "The shape depends on the `type` parameter and on the "
            "`future_array_shapes` parameter. For 'baseline' type objects, "
            "the shape is (Nblts, 1, Nfreq, Npols) or (Nblts, Nfreqs, Npols) if "
            "future_array_shapes=True. For 'antenna' type objects, the shape is "
            "(Nants_data, 1, Nfreqs, Ntimes, Npols) or "
            "(Nants_data, Nfreqs, Ntimes, Npols) if future_array_shapes=True. "
            "For 'waterfall' type objects, the shape is (Ntimes, Nfreq, Npols)."
        )
        self._weights_array = uvp.UVParameter(
            "weights_array",
            description=desc,
            form=("Nblts", 1, "Nfreqs", "Npols"),
            expected_type=float,
        )

        desc = (
            "Floating point weight information about sum of squares of weights "
            "when weighted data is converted from baseline to waterfall type."
            "Only available in metric mode, the shape is (Nfreq, Ntimes, Npols)."
        )
        # TODO: should this be set to None when converting back to baseline or antenna?
        # If not, should the shape be adjusted?
        self._weights_square_array = uvp.UVParameter(
            "weights_square_array",
            description=desc,
            form=("Ntimes", "Nfreqs", "Npols"),
            expected_type=float,
            required=False,
        )

        desc = (
            "Array of times in Julian Date, center of integration. The shape depends "
            "on the `type` parameter. For 'baseline' type object, shape is (Nblts), "
            "for 'antenna' and 'waterfall' type objects, shape is (Ntimes)."
        )
        self._time_array = uvp.UVParameter(
            "time_array",
            description=desc,
            form=("Nblts",),
            expected_type=float,
            tols=1e-3 / (60.0 * 60.0 * 24.0),
        )  # 1 ms in days

        desc = (
            "Array of lsts radians, center of integration. The shape depends "
            "on the `type` parameter. For 'baseline' type object, shape is (Nblts), "
            "for 'antenna' and 'waterfall' type objects, shape is (Ntimes)."
        )
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Nblts",),
            expected_type=float,
            tols=uvutils.RADIAN_TOL,
        )

        desc = (
            "Array of first antenna numbers, shape (Nblts). Only available for "
            "'baseline' type objects."
        )
        self._ant_1_array = uvp.UVParameter(
            "ant_1_array",
            description=desc,
            expected_type=int,
            form=("Nblts",),
            required=False,
        )
        desc = (
            "Array of second antenna numbers, shape (Nblts). Only available for "
            "'baseline' type objects."
        )
        self._ant_2_array = uvp.UVParameter(
            "ant_2_array",
            description=desc,
            expected_type=int,
            form=("Nblts",),
            required=False,
        )

        desc = (
            "Array of antenna numbers, shape (Nants_data), only available for "
            "'antenna' type objects. "
        )
        self._ant_array = uvp.UVParameter(
            "ant_array",
            description=desc,
            expected_type=int,
            form=("Nants_data",),
            required=False,
        )

        desc = (
            "Array of baseline indices, shape (Nblts). "
            "Only available for 'baseline' type objects. "
            "type = int; baseline = 2048 * ant1 + ant2 + 2^16"
        )
        self._baseline_array = uvp.UVParameter(
            "baseline_array",
            description=desc,
            expected_type=int,
            form=("Nblts",),
            required=False,
        )

        desc = (
            "Array of frequencies in Hz, center of the channel. Shape (1, Nfreqs) or "
            "(Nfreqs,) if type is 'waterfall' or if future_array_shapes=True."
        )
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=(1, "Nfreqs"),
            expected_type=float,
            tols=1e-3,
        )  # mHz

        desc = "Width of frequency channels (Hz). Shape (Nfreqs,), type = float."
        self._channel_width = uvp.UVParameter(
            "channel_width",
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,
        )  # 1 mHz

        self._spw_array = uvp.UVParameter(
            "spw_array",
            description="Array of spectral window numbers, shape (Nspws).",
            form=("Nspws",),
            expected_type=int,
        )

        desc = (
            "Required if Nspws > 1 and will always be required starting in "
            "version 3.0. Maps individual channels along the "
            "frequency axis to individual spectral windows, as listed in the "
            "spw_array. Shape (Nfreqs), type = int."
        )
        self._flex_spw_id_array = uvp.UVParameter(
            "flex_spw_id_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=int,
            required=False,
        )

        desc = (
            "Array of polarization integers, shape (Npols). "
            "AIPS Memo 117 says: pseudo-stokes 1:4 (pI, pQ, pU, pV);  "
            "circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX). "
            "NOTE: AIPS Memo 117 actually calls the pseudo-Stokes polarizations "
            '"Stokes", but this is inaccurate as visibilities cannot be in '
            "true Stokes polarizations for physical antennas. We adopt the "
            "term pseudo-Stokes to refer to linear combinations of instrumental "
            "visibility polarizations (e.g. pI = xx + yy)."
        )
        self._polarization_array = uvp.UVParameter(
            "polarization_array",
            description=desc,
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
            form=("Npols",),
        )

        self._telescope_name = uvp.UVParameter(
            "telescope_name",
            description="Name of telescope or array (string).",
            form="str",
            expected_type=str,
        )

        self._telescope_location = uvp.LocationParameter(
            "telescope_location", description=desc, tols=1e-3
        )

        self._history = uvp.UVParameter(
            "history",
            description="String of history, units English",
            form="str",
            expected_type=str,
        )

        desc = (
            "Any user supplied extra keywords, type=dict."
            "Use the special key 'comment' for long multi-line string comments."
            "Default is an empty dictionary."
        )
        self._extra_keywords = uvp.UVParameter(
            "extra_keywords",
            required=False,
            description=desc,
            value={},
            spoof_val={},
            expected_type=dict,
        )

        desc = "Flag indicating that this object is using the future array shapes."
        self._future_array_shapes = uvp.UVParameter(
            "future_array_shapes", description=desc, expected_type=bool, value=False
        )

        # ---antenna information ---
        desc = (
            "Number of antennas in the array.  Only available for 'baseline' type "
            "objects, used for calculating baseline numbers. "
            "May be larger than the number of antennas with data."
        )
        self._Nants_telescope = uvp.UVParameter(
            "Nants_telescope", description=desc, expected_type=int, required=False
        )
        desc = (
            "Number of antennas with data present. "
            "Only available for 'baseline' or 'antenna' type objects."
            "May be smaller than the number of antennas in the array"
        )
        self._Nants_data = uvp.UVParameter(
            "Nants_data", description=desc, expected_type=int, required=False
        )

        desc = (
            "List of antenna names, shape (Nants_telescope), with numbers given by "
            "antenna_numbers (which can be matched to ant_1_array and ant_2_array for "
            "baseline type or ant_array for antenna type objects). Required for "
            "baseline or antenna type objects. There must be one entry here for each "
            "unique entry in ant_1_array and ant_2_array (for baseline type) or "
            "ant_array (for antenna type), but there may be extras as well. "
        )
        self._antenna_names = uvp.UVParameter(
            "antenna_names",
            description=desc,
            form=("Nants_telescope",),
            expected_type=str,
        )

        desc = (
            "List of integer antenna numbers corresponding to antenna_names, "
            "shape (Nants_telescope). Required for baseline or antenna type objects. "
            "There must be one entry here for each unique entry in ant_1_array and "
            "ant_2_array (for baseline type) or ant_array (for antenna type), but "
            "there may be extras as well. Note that these are not indices -- they do "
            "not need to start at zero or be continuous."
        )
        self._antenna_numbers = uvp.UVParameter(
            "antenna_numbers",
            description=desc,
            form=("Nants_telescope",),
            expected_type=int,
        )

        desc = (
            "Array giving coordinates of antennas relative to "
            "telescope_location (ITRF frame), shape (Nants_telescope, 3), "
            "units meters. See the tutorial page in the documentation "
            "for an example of how to convert this to topocentric frame."
        )
        self._antenna_positions = uvp.UVParameter(
            "antenna_positions",
            description=desc,
            form=("Nants_telescope", 3),
            expected_type=float,
            tols=1e-3,  # 1 mm
        )

        #  --extra information ---
        desc = (
            "Orientation of the physical dipole corresponding to what is "
            'labelled as the x polarization. Options are "east" '
            '(indicating east/west orientation) and "north" (indicating '
            "north/south orientation)"
        )
        self._x_orientation = uvp.UVParameter(
            "x_orientation",
            description=desc,
            required=False,
            expected_type=str,
            acceptable_vals=["east", "north"],
        )

        desc = (
            "List of strings containing the unique basenames (not the full path) of "
            "input files."
        )
        self._filename = uvp.UVParameter(
            "filename", required=False, description=desc, expected_type=str
        )

        # initialize the underlying UVBase properties
        super(UVFlag, self).__init__()

        self.history = ""  # Added to at the end

        self.label = ""  # Added to at the end
        if isinstance(indata, (list, tuple)):
            self.__init__(
                indata[0],
                mode=mode,
                copy_flags=copy_flags,
                waterfall=waterfall,
                history=history,
                label=label,
                use_future_array_shapes=use_future_array_shapes,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
            if len(indata) > 1:
                for i in indata[1:]:
                    fobj = UVFlag(
                        i,
                        mode=mode,
                        copy_flags=copy_flags,
                        waterfall=waterfall,
                        history=history,
                        use_future_array_shapes=use_future_array_shapes,
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                    )
                    self.__add__(
                        fobj,
                        run_check=run_check,
                        inplace=True,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                    )
                del fobj

        elif issubclass(indata.__class__, (str, pathlib.Path)):
            # Given a path, read indata
            self.read(
                indata,
                history,
                telescope_name=telescope_name,
                mwa_metafits_file=mwa_metafits_file,
                use_future_array_shapes=use_future_array_shapes,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )
        elif issubclass(indata.__class__, UVData):
            self.from_uvdata(
                indata,
                mode=mode,
                copy_flags=copy_flags,
                waterfall=waterfall,
                history=history,
                label=label,
                use_future_array_shapes=use_future_array_shapes,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )

        elif issubclass(indata.__class__, UVCal):
            self.from_uvcal(
                indata,
                mode=mode,
                copy_flags=copy_flags,
                waterfall=waterfall,
                history=history,
                label=label,
                use_future_array_shapes=use_future_array_shapes,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
            )

        elif indata is not None:
            raise ValueError(
                "input to UVFlag.__init__ must be one of: "
                "list, tuple, string, pathlib.Path, UVData, or UVCal."
            )

    @property
    def _data_params(self):
        """List of strings giving the data-like parameters."""
        if not hasattr(self, "mode") or self.mode is None:
            return None
        elif self.mode == "flag":
            return ["flag_array"]
        elif self.mode == "metric":
            if self.weights_square_array is None:
                return ["metric_array", "weights_array"]
            else:
                return ["metric_array", "weights_array", "weights_square_array"]
        else:
            raise ValueError(
                "Invalid mode. Mode must be one of "
                + ", ".join(["{}"] * len(self._mode.acceptable_vals)).format(
                    *self._mode.acceptable_vals
                )
            )

    @property
    def data_like_parameters(self):
        """Return iterator of defined parameters which are data-like."""
        for key in self._data_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def pol_collapsed(self):
        """Determine if this object has had pols collapsed."""
        if not hasattr(self, "polarization_array") or self.polarization_array is None:
            return False
        elif isinstance(self.polarization_array.item(0), str):
            return True
        else:
            return False

    def _check_pol_state(self):
        if self.pol_collapsed:
            # collapsed pol objects have a different type for
            # the polarization array.
            self._polarization_array.expected_type = str
            self._polarization_array.acceptable_vals = None
        else:
            self._polarization_array.expected_type = uvp._get_generic_type(int)
            self._polarization_array.acceptable_vals = list(np.arange(-8, 0)) + list(
                np.arange(1, 5)
            )

    def _set_future_array_shapes(self):
        """
        Set future_array_shapes to True and adjust required parameters.

        This method should not be called directly by users; instead it is called
        by file-reading methods and `use_future_array_shapes` to indicate the
        `future_array_shapes` is True and define expected parameter shapes.

        """
        self.future_array_shapes = True
        self._freq_array.form = ("Nfreqs",)
        data_like_params = ["metric_array", "weights_array", "flag_array"]

        if self.type == "baseline":
            for param_name in data_like_params:
                getattr(self, "_" + param_name).form = ("Nblts", "Nfreqs", "Npols")
        elif self.type == "antenna":
            for param_name in data_like_params:
                getattr(self, "_" + param_name).form = (
                    "Nants_data",
                    "Nfreqs",
                    "Ntimes",
                    "Npols",
                )

    def use_future_array_shapes(self):
        """
        Change the array shapes of this object to match the planned future shapes.

        This method sets allows users to convert to the planned array shapes changes
        before the changes go into effect. This method sets the `future_array_shapes`
        parameter on this object to True.

        """
        if self.future_array_shapes:
            return

        self._set_future_array_shapes()
        if not self.type == "waterfall":
            # remove the length-1 spw axis for all data-like parameters
            for param_name in self._data_params:
                if param_name == "weights_square_array":
                    continue
                setattr(self, param_name, (getattr(self, param_name))[:, 0])

            # remove the length-1 spw axis for the freq_array
            self.freq_array = self.freq_array[0, :]

    def use_current_array_shapes(self):
        """
        Change the array shapes of this object to match the current future shapes.

        This method sets allows users to convert back to the current array shapes.
        This method sets the `future_array_shapes` parameter on this object to False.
        """
        warnings.warn(
            "This method will be removed in version 3.0 when the current array shapes "
            "are no longer supported.",
            DeprecationWarning,
        )

        if not self.future_array_shapes:
            return

        data_like_params = ["metric_array", "weights_array", "flag_array"]

        self.future_array_shapes = False
        if not self.type == "waterfall":
            if self.type == "baseline":
                for param_name in data_like_params:
                    getattr(self, "_" + param_name).form = (
                        "Nblts",
                        1,
                        "Nfreqs",
                        "Npols",
                    )
            elif self.type == "antenna":
                for param_name in data_like_params:
                    getattr(self, "_" + param_name).form = (
                        "Nants_data",
                        1,
                        "Nfreqs",
                        "Ntimes",
                        "Npols",
                    )

            for param_name in self._data_params:
                if param_name == "weights_square_array":
                    continue
                setattr(
                    self, param_name, (getattr(self, param_name))[:, np.newaxis, :, :]
                )

            self._freq_array.form = (1, "Nfreqs")
            self.freq_array = self.freq_array[np.newaxis, :]

    def _set_mode_flag(self):
        """Set the mode and required parameters consistent with a flag object."""
        self.mode = "flag"
        self._flag_array.required = True
        self._metric_array.required = False
        self._weights_array.required = False
        if self.weights_square_array is not None:
            self.weights_square_array = None

        return

    def _set_mode_metric(self):
        """Set the mode and required parameters consistent with a metric object."""
        self.mode = "metric"
        self._flag_array.required = False
        self._metric_array.required = True
        self._weights_array.required = True

        if self.weights_array is None and self.metric_array is not None:
            self.weights_array = np.ones_like(self.metric_array, dtype=float)

        return

    def _set_type_antenna(self):
        """Set the type and required propertis consistent with an antenna object."""
        self.type = "antenna"
        self._ant_array.required = True
        self._baseline_array.required = False
        self._ant_1_array.required = False
        self._ant_2_array.required = False
        self._Nants_telescope.required = False
        self._Nants_data.required = True
        self._Nbls.required = False
        self._Nblts.required = False

        if self.future_array_shapes:
            self._metric_array.form = ("Nants_data", "Nfreqs", "Ntimes", "Npols")
            self._flag_array.form = ("Nants_data", "Nfreqs", "Ntimes", "Npols")
            self._weights_array.form = ("Nants_data", "Nfreqs", "Ntimes", "Npols")
        else:
            self._metric_array.form = ("Nants_data", 1, "Nfreqs", "Ntimes", "Npols")
            self._flag_array.form = ("Nants_data", 1, "Nfreqs", "Ntimes", "Npols")
            self._weights_array.form = ("Nants_data", 1, "Nfreqs", "Ntimes", "Npols")
            self._freq_array.form = (1, "Nfreqs")

        self._time_array.form = ("Ntimes",)
        self._lst_array.form = ("Ntimes",)

    def _set_type_baseline(self):
        """Set the type and required propertis consistent with a baseline object."""
        self.type = "baseline"
        self._ant_array.required = False
        self._baseline_array.required = True
        self._ant_1_array.required = True
        self._ant_2_array.required = True
        self._Nants_telescope.required = True
        self._Nants_data.required = True
        self._Nbls.required = True
        self._Nblts.required = True

        if self.time_array is not None:
            self.Nblts = len(self.time_array)

        if self.future_array_shapes:
            self._metric_array.form = ("Nblts", "Nfreqs", "Npols")
            self._flag_array.form = ("Nblts", "Nfreqs", "Npols")
            self._weights_array.form = ("Nblts", "Nfreqs", "Npols")

        else:
            self._metric_array.form = ("Nblts", 1, "Nfreqs", "Npols")
            self._flag_array.form = ("Nblts", 1, "Nfreqs", "Npols")
            self._weights_array.form = ("Nblts", 1, "Nfreqs", "Npols")
            self._freq_array.form = (1, "Nfreqs")

        self._time_array.form = ("Nblts",)
        self._lst_array.form = ("Nblts",)

    def _set_type_waterfall(self):
        """Set the type and required propertis consistent with a waterfall object."""
        self.type = "waterfall"
        self._ant_array.required = False
        self._baseline_array.required = False
        self._ant_1_array.required = False
        self._ant_2_array.required = False
        self._Nants_telescope.required = False
        self._Nants_data.required = False
        self._Nbls.required = False
        self._Nblts.required = False

        self._metric_array.form = ("Ntimes", "Nfreqs", "Npols")
        self._flag_array.form = ("Ntimes", "Nfreqs", "Npols")
        self._weights_array.form = ("Ntimes", "Nfreqs", "Npols")

        self._time_array.form = ("Ntimes",)
        self._lst_array.form = ("Ntimes",)

        if not self.future_array_shapes:
            self._freq_array.form = ("Nfreqs",)

    def check(
        self,
        check_extra=True,
        run_check_acceptability=True,
        lst_tol=uvutils.LST_RAD_TOL,
    ):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check all parameters, otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.
        lst_tol : float or None
            Tolerance level at which to test LSTs against their expected values. If
            provided as a float, must be in units of radians. If set to None, the
            default precision tolerance from the `lst_array` parameter is used (1 mas).
            Default value is 75 mas,  which is set by the predictive uncertainty in IERS
            calculations of DUT1 (RMS is of order 1 ms, with with a 5-sigma threshold
            for detection is used to prevent false issues from being reported), which
            for some observatories sets the precision with which these values are
            written.

        Returns
        -------
        bool
            True if check passes

        Raises
        ------
        ValueError
            if parameter shapes or types are wrong or do not have acceptable
            values (if run_check_acceptability is True)

        """
        # set the flex_spw_id_array to required if Nspws > 1
        if self.Nspws is not None and self.Nspws > 1:
            self._flex_spw_id_array.required = True
        else:
            self._flex_spw_id_array.required = False

        # first run the basic check from UVBase
        super().check(check_extra, run_check_acceptability)

        # Check internal consistency of numbers which don't explicitly correspond
        # to the shape of another array.
        if self.type == "baseline":
            if self.Nants_data != int(
                np.union1d(self.ant_1_array, self.ant_2_array).size
            ):
                raise ValueError(
                    "Nants_data must be equal to the number of unique "
                    "values in ant_1_array and ant_2_array"
                )

            if self.Nbls != len(np.unique(self.baseline_array)):
                raise ValueError(
                    "Nbls must be equal to the number of unique "
                    "baselines in the baseline_array"
                )

            if self.Ntimes != len(np.unique(self.time_array)):
                raise ValueError(
                    "Ntimes must be equal to the number of unique "
                    "times in the time_array"
                )

            if self.antenna_numbers is not None:
                if not set(np.unique(self.ant_1_array)).issubset(self.antenna_numbers):
                    raise ValueError(
                        "All antennas in ant_1_array must be in antenna_numbers."
                    )
                if not set(np.unique(self.ant_2_array)).issubset(self.antenna_numbers):
                    raise ValueError(
                        "All antennas in ant_2_array must be in antenna_numbers."
                    )
        elif self.type == "antenna":
            if self.antenna_numbers is not None:
                missing_ants = self.ant_array[
                    ~np.isin(self.ant_array, self.antenna_numbers)
                ]
                if missing_ants.size > 0:
                    raise ValueError(
                        "All antennas in ant_array must be in antenna_numbers. "
                        "The antennas in ant_array that are missing in antenna_numbers "
                        f"are: {missing_ants}"
                    )

        if self.flex_spw_id_array is None:
            warnings.warn(
                "flex_spw_id_array is not set. It will be required starting in version "
                "3.0",
                DeprecationWarning,
            )
        else:
            # Check that all values in flex_spw_id_array are entries in the spw_array
            if not np.all(np.isin(self.flex_spw_id_array, self.spw_array)):
                raise ValueError(
                    "All values in the flex_spw_id_array must exist in the spw_array."
                )

        if run_check_acceptability:
            # Check antenna positions
            uvutils.check_surface_based_positions(
                antenna_positions=self.antenna_positions,
                telescope_loc=self.telescope_location,
                telescope_frame=self._telescope_location.frame,
                raise_error=False,
            )

            lat, lon, alt = self.telescope_location_lat_lon_alt_degrees
            uvutils.check_lsts_against_times(
                jd_array=self.time_array,
                lst_array=self.lst_array,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                lst_tols=self._lst_array.tols if lst_tol is None else [0, lst_tol],
                frame=self._telescope_location.frame,
            )

        return True

    def clear_unused_attributes(self):
        """Remove unused attributes.

        Useful when changing type or mode or to save memory.
        Will set all non-required attributes to None, except x_orientation,
        extra_keywords, weights_square_array and filename.

        """
        optional_attrs_to_keep = [
            "telescope_name",
            "telescope_location",
            "channel_width",
            "spw_array",
            "Nspws",
            "flex_spw_id_array",  # TODO remove this from this list in version 3.0
            "antenna_names",
            "antenna_numbers",
            "antenna_positions",
            "Nants_telescope",
            "x_orientation",
            "weights_square_array",
            "extra_keywords",
            "filename",
        ]
        for p in self:
            attr = getattr(self, p)
            if (
                not attr.required
                and attr.value is not None
                and attr.name not in optional_attrs_to_keep
            ):
                attr.value = None
                setattr(self, p, attr)

    def __eq__(self, other, check_history=True, check_extra=True):
        """Check Equality of two UVFlag objects.

        Parameters
        ----------
        other: UVFlag
            object to check against
        check_history : bool
            Include the history keyword when comparing UVFlag objects.
        check_extra : bool
            Include non-required parameters when comparing UVFlag objects.

        """
        if check_history:
            return super(UVFlag, self).__eq__(other, check_extra=check_extra)

        else:
            # initial check that the classes are the same
            # then strip the histories
            if isinstance(other, self.__class__):
                _h1 = self.history
                self.history = None

                _h2 = other.history
                other.history = None

                truth = super(UVFlag, self).__eq__(other, check_extra=check_extra)

                self.history = _h1
                other.history = _h2

                return truth
            else:
                print("Classes do not match")
                return False

    def __ne__(self, other, check_history=True, check_extra=True):
        """Not Equal."""
        return not self.__eq__(
            other, check_history=check_history, check_extra=check_extra
        )

    def _set_lsts_helper(self, astrometry_library=None):
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
        self.lst_array = uvutils.get_lst_for_time(
            jd_array=self.time_array,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            astrometry_library=astrometry_library,
            frame=self._telescope_location.frame,
        )
        return

    def set_lsts_from_time_array(self, background=False, astrometry_library=None):
        """Set the lst_array based from the time_array.

        Parameters
        ----------
        background : bool, False
            When set to True, start the calculation on a threading.Thread in the
            background and return the thread to the user.

        Returns
        -------
        proc : None or threading.Thread instance
            When background is set to True, a thread is returned which must be
            joined before the lst_array exists on the UVData object.

        """
        if not background:
            self._set_lsts_helper(astrometry_library=astrometry_library)
            return
        else:
            proc = threading.Thread(
                target=self._set_lsts_helper,
                kwargs={"astrometry_library": astrometry_library},
            )
            proc.start()
            return proc

    def set_telescope_params(self, overwrite=False, warn=True):
        """
        Set telescope related parameters.

        If the telescope_name is in the known_telescopes, set any missing
        telescope-associated parameters (e.g. telescope location) to the value
        for the known telescope.

        Parameters
        ----------
        overwrite : bool
            Option to overwrite existing telescope-associated parameters with
            the values from the known telescope.

        Raises
        ------
        ValueError
            if the telescope_name is not in known telescopes
        """
        telescope_obj = uvtel.get_telescope(self.telescope_name)
        if telescope_obj is not False:
            params_set = []
            telescope_params = list(telescope_obj.__iter__())
            # ensure that the Nants_telescope comes first so shapes work out below
            telescope_params.remove("_Nants_telescope")
            telescope_params.insert(0, "_Nants_telescope")

            set_ant_metadata = True
            if self.type != "waterfall" and "_antenna_numbers" in telescope_params:
                # need to check that all antennas on the object are in the telescope's
                # antenna_numbers
                if self.type == "antenna":
                    ants_to_check = self.ant_array
                else:
                    ants_to_check = np.union1d(self.ant_1_array, self.ant_2_array)

                if not all(
                    ant in telescope_obj.antenna_numbers for ant in ants_to_check
                ):
                    warnings.warn(
                        "Not all antennas with data have metadata in the telescope "
                        "object. Not setting antenna metadata."
                    )
                    set_ant_metadata = False

            if not set_ant_metadata:
                ant_params_to_remove = []
                for p in telescope_params:
                    if "ant" in p:
                        ant_params_to_remove.append(p)

                for p in ant_params_to_remove:
                    telescope_params.remove(p)

            for p in telescope_params:
                telescope_param = getattr(telescope_obj, p)
                if p in self:
                    self_param = getattr(self, p)
                else:
                    continue
                if telescope_param.value is not None and (
                    overwrite is True or self_param.value is None
                ):
                    telescope_shape = telescope_param.expected_shape(telescope_obj)
                    self_shape = self_param.expected_shape(self)
                    if telescope_shape == self_shape:
                        params_set.append(self_param.name)
                        prop_name = self_param.name
                        setattr(self, prop_name, getattr(telescope_obj, prop_name))
                    else:
                        # Note dropped handling for antenna diameters that appears in
                        # UVData because they don't exist on UVFlag.
                        warnings.warn(
                            f"{self_param.name} is not set but cannot be set using "
                            f"known values for {telescope_obj.telescope_name} "
                            "because the expected shapes don't match."
                        )

            if len(params_set) > 0:
                if warn:
                    params_set_str = ", ".join(params_set)
                    warnings.warn(
                        f"{params_set_str} are not set or are being "
                        "overwritten. Using known values for "
                        f"{telescope_obj.telescope_name}."
                    )
        else:
            raise ValueError(
                f"Telescope {self.telescope_name} is not in known_telescopes."
            )

    def antpair2ind(self, ant1, ant2):
        """Get blt indices for given (ordered) antenna pair.

        Parameters
        ----------
        ant1 : int or array_like of int
            Number of the first antenna
        ant2 : int or array_like of int
            Number of the second antenna

        Returns
        -------
        int or array_like of int
            baseline number(s) corresponding to the input antenna number

        """
        if self.type != "baseline":
            raise ValueError(
                "UVFlag object of type " + self.type + " does not "
                "contain antenna pairs to index."
            )
        return np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]

    def baseline_to_antnums(self, baseline):
        """Get the antenna numbers corresponding to a given baseline number.

        Parameters
        ----------
        baseline : int
            baseline number

        Returns
        -------
        tuple
            Antenna numbers corresponding to baseline.

        """
        assert self.type == "baseline", "Must be 'baseline' type UVFlag object."
        return uvutils.baseline_to_antnums(baseline, self.Nants_telescope)

    def antnums_to_baseline(self, ant1, ant2, attempt256=False):
        """
        Get the baseline number corresponding to two given antenna numbers.

        Parameters
        ----------
        ant1 : int or array_like of int
            first antenna number
        ant2 : int or array_like of int
            second antenna number
        attempt256 : bool
            Option to try to use the older 256 standard used in many uvfits files
            (will use 2048 standard if there are more than 256 antennas).

        Returns
        -------
        int or array of int
            baseline number corresponding to the two antenna numbers.
        """
        assert self.type == "baseline", "Must be 'baseline' type UVFlag object."
        return uvutils.antnums_to_baseline(
            ant1, ant2, self.Nants_telescope, attempt256=attempt256
        )

    def get_baseline_nums(self):
        """Return numpy array of unique baseline numbers in data."""
        assert self.type == "baseline", "Must be 'baseline' type UVFlag object."
        return np.unique(self.baseline_array)

    def get_antpairs(self):
        """Return list of unique antpair tuples (ant1, ant2) in data."""
        assert self.type == "baseline", "Must be 'baseline' type UVFlag object."
        return list(zip(*self.baseline_to_antnums(self.get_baseline_nums())))

    def get_ants(self):
        """
        Get the unique antennas that have data associated with them.

        Returns
        -------
        ndarray of int
            Array of unique antennas with data associated with them.
        """
        if self.type == "baseline":
            return np.unique(np.append(self.ant_1_array, self.ant_2_array))
        elif self.type == "antenna":
            return np.unique(self.ant_array)
        elif self.type == "waterfall":
            raise ValueError("A waterfall type UVFlag object has no sense of antennas.")

    def get_pols(self):
        """
        Get the polarizations in the data.

        Returns
        -------
        list of str
            list of polarizations (as strings) in the data.
        """
        return uvutils.polnum2str(
            self.polarization_array, x_orientation=self.x_orientation
        )

    def parse_ants(self, ant_str, print_toggle=False):
        """
        Get antpair and polarization from parsing an aipy-style ant string.

        Used to support the select function. This function is only useable when
        the UVFlag type is 'baseline'. Generates two lists of antenna pair tuples
        and polarization indices based on parsing of the string ant_str. If no
        valid polarizations (pseudo-Stokes params, or combinations of [lr] or
        [xy]) or antenna numbers are found in ant_str, ant_pairs_nums and
        polarizations are returned as None.

        Parameters
        ----------
        ant_str : str
            String containing antenna information to parse. Can be 'all',
            'auto', 'cross', or combinations of antenna numbers and polarization
            indicators 'l' and 'r' or 'x' and 'y'.  Minus signs can also be used
            in front of an antenna number or baseline to exclude it from being
            output in ant_pairs_nums. If ant_str has a minus sign as the first
            character, 'all,' will be added to the beginning of the string.
            See the tutorial for examples of valid strings and their behavior.
        print_toggle : bool
            Boolean for printing parsed baselines for a visual user check.

        Returns
        -------
        ant_pairs_nums : list of tuples of int or None
            List of tuples containing the parsed pairs of antenna numbers, or
            None if ant_str is 'all' or a pseudo-Stokes polarizations.
        polarizations : list of int or None
            List of desired polarizations or None if ant_str does not contain a
            polarization specification.

        """
        if self.type != "baseline":
            raise ValueError(
                "UVFlag objects can only call 'parse_ants' function "
                "if type is 'baseline'."
            )
        return uvutils.parse_ants(
            self,
            ant_str=ant_str,
            print_toggle=print_toggle,
            x_orientation=self.x_orientation,
        )

    def collapse_pol(
        self,
        method="quadmean",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Collapse the polarization axis using a given method.

        If the original UVFlag object has more than one polarization,
        the resulting polarization_array will be a single element array with a
        comma separated string encoding the original polarizations.

        Parameters
        ----------
        method : str, {"quadmean", "absmean", "mean", "or", "and"}
            How to collapse the dimension(s).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after collapsing polarizations.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            collapsing polarizations.

        """
        method = method.lower()
        if self.mode == "flag":
            darr = self.flag_array
        else:
            darr = self.metric_array
        if len(self.polarization_array) > 1:
            if self.mode == "metric":
                _weights = self.weights_array
            else:
                _weights = np.ones_like(darr)
            # Collapse pol dimension. But note we retain a polarization axis.
            d, w = uvutils.collapse(
                darr, method, axis=-1, weights=_weights, return_weights=True
            )
            darr = np.expand_dims(d, axis=d.ndim)

            if self.mode == "metric":
                self.weights_array = np.expand_dims(w, axis=w.ndim)

            self.polarization_array = np.array(
                [",".join(map(str, self.polarization_array))], dtype=np.str_
            )

            self.Npols = len(self.polarization_array)
            self._check_pol_state()
        else:
            warnings.warn(
                "Cannot collapse polarization axis when only one pol present."
            )
            return
        if ((method == "or") or (method == "and")) and (self.mode == "flag"):
            self.flag_array = darr
        else:
            self.metric_array = darr
            self._set_mode_metric()
        self.clear_unused_attributes()
        self.history += "Pol axis collapse. "

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_waterfall(
        self,
        method="quadmean",
        keep_pol=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        return_weights_square=False,
    ):
        """Convert an 'antenna' or 'baseline' type object to waterfall.

        Parameters
        ----------
        method : str, {"quadmean", "absmean", "mean", "or", "and"}
            How to collapse the dimension(s).
        keep_pol : bool
            Whether to also collapse the polarization dimension
            If keep_pol is False, and the original UVFlag object has more
            than one polarization, the resulting polarization_array
            will be a single element array with a comma separated string
            encoding the original polarizations.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to waterfall type.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to waterfall type.
        return_weights_square: bool
            Option to compute the sum of the squares of the weights when
            collapsing baseline object to waterfall. Not used if type is not
            baseline to begin with. Fills an optional parameter if so.

        """
        method = method.lower()
        if self.type == "waterfall" and (
            keep_pol or (len(self.polarization_array) == 1)
        ):
            warnings.warn("This object is already a waterfall. Nothing to change.")
            return
        if (not keep_pol) and (len(self.polarization_array) > 1):
            self.collapse_pol(method)

        if self.mode == "flag":
            darr = self.flag_array
        else:
            darr = self.metric_array

        if self.type == "antenna":
            if self.future_array_shapes:
                collapse_axes = (0,)
            else:
                collapse_axes = (0, 1)
            d, w = uvutils.collapse(
                darr,
                method,
                axis=collapse_axes,
                weights=self.weights_array,
                return_weights=True,
            )
            darr = np.swapaxes(d, 0, 1)
            if self.mode == "metric":
                self.weights_array = np.swapaxes(w, 0, 1)
        elif self.type == "baseline":
            Nt = len(np.unique(self.time_array))
            Nf = self.freq_array.size
            Np = len(self.polarization_array)
            d = np.zeros((Nt, Nf, Np))
            w = np.zeros((Nt, Nf, Np))
            if return_weights_square:
                ws = np.zeros((Nt, Nf, Np))
            for i, t in enumerate(np.unique(self.time_array)):
                ind = self.time_array == t
                if self.mode == "metric":
                    _weights = self.weights_array[ind, :, :]
                else:
                    _weights = np.ones_like(darr[ind, :, :], dtype=float)
                if return_weights_square:
                    d[i, :, :], w[i, :, :], ws[i, :, :] = uvutils.collapse(
                        darr[ind, :, :],
                        method,
                        axis=0,
                        weights=_weights,
                        return_weights=True,
                        return_weights_square=return_weights_square,
                    )
                else:
                    d[i, :, :], w[i, :, :] = uvutils.collapse(
                        darr[ind, :, :],
                        method,
                        axis=0,
                        weights=_weights,
                        return_weights=True,
                        return_weights_square=return_weights_square,
                    )
            darr = d
            if self.mode == "metric":
                self.weights_array = w
                if return_weights_square:
                    self.weights_square_array = ws
            self.time_array, ri = np.unique(self.time_array, return_index=True)
            self.lst_array = self.lst_array[ri]
        if ((method == "or") or (method == "and")) and (self.mode == "flag"):
            # If using a boolean operation (AND/OR) and in flag mode, stay in flag
            # flags should be bool, but somehow it is cast as float64
            # is reacasting to bool like this best?
            self.flag_array = darr.astype(bool)
        else:
            # Otherwise change to (or stay in) metric
            self.metric_array = darr
            self._set_mode_metric()
        self.freq_array = self.freq_array.flatten()
        self._set_type_waterfall()
        self.history += 'Collapsed to type "waterfall". '  # + self.pyuvdata_version_str

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        self.clear_unused_attributes()
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def sort_ant_metadata_like(self, uv):
        """
        Sort the antenna metadata arrays like an input object.

        This only does something if both objects have antenna_numbers defined
        and they contain the same set of antenna_numbers and they are
        differently sorted.

        Parameters
        ----------
        uv : UVFlag, UVCal or UVData object
            Object to match the antenna metadata sorting to

        """
        if (
            self.antenna_numbers is not None
            and uv.antenna_numbers is not None
            and np.intersect1d(self.antenna_numbers, uv.antenna_numbers).size
            == self.Nants_telescope
            and not np.allclose(self.antenna_numbers, uv.antenna_numbers)
        ):
            # first get sort order for each
            this_order = np.argsort(self.antenna_numbers)
            uv_order = np.argsort(uv.antenna_numbers)

            # now get array to invert the uv sort
            inv_uv_order = np.empty_like(uv_order)
            inv_uv_order[uv_order] = np.arange(uv.Nants_telescope)

            # generate the array to sort self like uv
            this_uv_sort = this_order[inv_uv_order]

            # do the sorting
            self.antenna_numbers = self.antenna_numbers[this_uv_sort]
            if self.antenna_names is not None:
                self.antenna_names = self.antenna_names[this_uv_sort]
            if self.antenna_positions is not None:
                self.antenna_positions = self.antenna_positions[this_uv_sort]

    def to_baseline(
        self,
        uv,
        force_pol=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert a UVFlag object of type "waterfall" or "antenna" to type "baseline".

        Broadcasts the flag array to all baselines.
        This function does NOT apply flags to uv (see utils.apply_uvflag for that).
        Note that the antenna metadata arrays (`antenna_names`, `antenna_numbers`
        and `antenna_positions`) may be reordered to match the ordering on `uv`.

        Parameters
        ----------
        uv : UVData or UVFlag object
            Object with type baseline to match.
        force_pol : bool
            If True, will use 1 pol to broadcast to any other pol.
            Otherwise, will require polarizations match.
            For example, this keyword is useful if one flags on all
            pols combined, and wants to broadcast back to individual pols.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to baseline type.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to baseline type.

        """
        if self.type == "baseline":
            return
        if not (
            issubclass(uv.__class__, UVData)
            or (isinstance(uv, UVFlag) and uv.type == "baseline")
        ):
            raise ValueError(
                "Must pass in UVData object or UVFlag object of type "
                '"baseline" to match.'
            )

        # write it out this rather than comparing the UVParameters because
        # future_array_shapes might be different. In the future, when shapes are not
        # variable, this can be done by comparing the UVParameters.
        if self.Nfreqs != uv.Nfreqs or not np.allclose(
            np.squeeze(self.freq_array),
            np.squeeze(uv.freq_array),
            rtol=self._freq_array.tols[0],
            atol=self._freq_array.tols[1],
        ):
            raise ValueError(
                "The freq_array on uv is not the same as the freq_array on this "
                f"object. The value on this object is {self.freq_array}; the value "
                f"on uv is {uv.freq_array}"
            )

        warn_compatibility_params = [
            "telescope_name",
            "telescope_location",
            "antenna_names",
            "antenna_numbers",
            "antenna_positions",
            "channel_width",
            "spw_array",
        ]
        if self.Nspws is not None and self.Nspws > 1:
            # TODO: make this always be in the compatibility list in version 3.0
            warn_compatibility_params.append("flex_spw_id_array")

        # sometimes the antenna sorting for the antenna names/numbers/positions
        # is different. If the sets are the same, re-sort self to match uv
        self.sort_ant_metadata_like(uv)

        for param in warn_compatibility_params:
            if (
                issubclass(uv.__class__, UVData)
                and param == "channel_width"
                and not (uv.future_array_shapes or uv.flex_spw)
            ):
                if not np.allclose(
                    self.channel_width,
                    np.full(uv.Nfreqs, uv.channel_width),
                    rtol=self._channel_width.tols[0],
                    atol=self._channel_width.tols[1],
                ):
                    raise ValueError(
                        "channel_width is not the same this object and on uv. The "
                        f"value on this object is {self.channel_width}; the value on "
                        f"uv is {uv.channel_width}."
                    )
            else:
                # compare the UVParameter objects to properly handle tolerances
                this_param = getattr(self, "_" + param)
                uv_param = getattr(uv, "_" + param)
                if this_param.value is not None and this_param != uv_param:
                    raise ValueError(
                        f"{param} is not the same this object and on uv. The value on "
                        f"this object is {this_param.value}; the value on uv is "
                        f"{uv_param.value}."
                    )

        # Deal with polarization
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = uv.polarization_array
            # Broadcast arrays
            if self.mode == "flag":
                self.flag_array = self.flag_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            else:
                self.metric_array = self.metric_array.repeat(
                    self.polarization_array.size, axis=-1
                )
                self.weights_array = self.weights_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            self.Npols = len(self.polarization_array)
            self._check_pol_state()

        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(uv.polarization_array, self.polarization_array):
            if self.polarization_array.size == 1:
                raise ValueError(
                    "Polarizations do not match. Try keyword force_pol"
                    + " if you wish to broadcast to all polarizations."
                )
            else:
                raise ValueError("Polarizations could not be made to match.")
        if self.type == "waterfall":
            # Populate arrays
            if self.mode == "flag":
                if (
                    issubclass(uv.__class__, UVData)
                    and uv.future_array_shapes != self.future_array_shapes
                ):
                    if uv.future_array_shapes:
                        arr = np.zeros_like(uv.flag_array[:, np.newaxis, :, :])
                    else:
                        arr = np.zeros_like(uv.flag_array[:, 0, :, :])
                else:
                    arr = np.zeros_like(uv.flag_array)
                sarr = self.flag_array
            elif self.mode == "metric":
                if (
                    issubclass(uv.__class__, UVData)
                    and uv.future_array_shapes != self.future_array_shapes
                ):
                    if uv.future_array_shapes:
                        arr = np.zeros_like(
                            uv.flag_array[:, np.newaxis, :, :], dtype=np.float64
                        )
                        warr = np.zeros_like(
                            uv.flag_array[:, np.newaxis, :, :], dtype=np.float64
                        )
                    else:
                        arr = np.zeros_like(uv.flag_array[:, 0, :, :], dtype=np.float64)
                        warr = np.zeros_like(
                            uv.flag_array[:, 0, :, :], dtype=np.float64
                        )
                else:
                    arr = np.zeros_like(uv.flag_array, dtype=np.float64)
                    warr = np.zeros_like(uv.flag_array, dtype=np.float64)
                sarr = self.metric_array
            for i, t in enumerate(np.unique(self.time_array)):
                ti = np.where(
                    np.isclose(
                        uv.time_array,
                        t,
                        rtol=max(self._time_array.tols[0], uv._time_array.tols[0]),
                        atol=max(self._time_array.tols[1], uv._time_array.tols[1]),
                    )
                )
                if self.future_array_shapes:
                    arr[ti] = sarr[i][np.newaxis, :, :]
                    if self.mode == "metric":
                        warr[ti] = self.weights_array[i][np.newaxis, :, :]
                else:
                    arr[ti] = sarr[i][np.newaxis, np.newaxis, :, :]
                    if self.mode == "metric":
                        warr[ti] = self.weights_array[i][np.newaxis, np.newaxis, :, :]
            if self.mode == "flag":
                self.flag_array = arr
            elif self.mode == "metric":
                self.metric_array = arr
                self.weights_array = warr
        elif self.type == "antenna":
            if self.mode == "metric":
                raise NotImplementedError(
                    "Cannot currently convert from antenna type, metric mode to "
                    "baseline type UVFlag object."
                )
            ants_data = np.unique(uv.ant_1_array.tolist() + uv.ant_2_array.tolist())
            new_ants = np.setdiff1d(ants_data, self.ant_array)
            if new_ants.size > 0:
                self.ant_array = np.append(self.ant_array, new_ants).tolist()
                # make new flags of the same shape but with first axis the
                # size of the new ants
                flag_shape = list(self.flag_array.shape)
                flag_shape[0] = new_ants.size
                new_flags = np.full(flag_shape, True, dtype=bool)
                self.flag_array = np.append(self.flag_array, new_flags, axis=0)

            if self.future_array_shapes:
                baseline_flags = np.full(
                    (uv.Nblts, self.Nfreqs, self.Npols), True, dtype=bool
                )
            else:
                baseline_flags = np.full(
                    (uv.Nblts, 1, self.Nfreqs, self.Npols), True, dtype=bool
                )

            for blt_index, bl in enumerate(uv.baseline_array):
                uvf_t_index = np.nonzero(
                    np.isclose(
                        uv.time_array[blt_index],
                        self.time_array,
                        rtol=max(self._time_array.tols[0], uv._time_array.tols[0]),
                        atol=max(self._time_array.tols[1], uv._time_array.tols[1]),
                    )
                )[0]
                if uvf_t_index.size > 0:
                    # if the time is found in the uvflag object time_array
                    # input the or'ed data from each antenna
                    ant1, ant2 = uv.baseline_to_antnums(bl)
                    ant1_index = np.nonzero(np.array(self.ant_array) == ant1)
                    ant2_index = np.nonzero(np.array(self.ant_array) == ant2)
                    if self.future_array_shapes:
                        or_flag = np.logical_or(
                            self.flag_array[ant1_index, :, uvf_t_index, :],
                            self.flag_array[ant2_index, :, uvf_t_index, :],
                        )
                    else:
                        or_flag = np.logical_or(
                            self.flag_array[ant1_index, :, :, uvf_t_index, :],
                            self.flag_array[ant2_index, :, :, uvf_t_index, :],
                        )

                    baseline_flags[blt_index] = or_flag.copy()

            self.flag_array = baseline_flags

        # Check the frequency array for shape, broadcast to (1, Nfreqs) if needed
        if not self.future_array_shapes:
            self.freq_array = np.atleast_2d(self.freq_array)

        if self.Nspws is None:
            self.Nspws = uv.Nspws
            self.spw_array = uv.spw_array
            if uv.flex_spw_id_array is not None:
                self.flex_spw_id_array = uv.flex_spw_id_array

        self.baseline_array = uv.baseline_array
        self.Nbls = np.unique(self.baseline_array).size
        self.ant_1_array = uv.ant_1_array
        self.ant_2_array = uv.ant_2_array
        self.Nants_data = int(np.union1d(self.ant_1_array, self.ant_2_array).size)

        self.time_array = uv.time_array
        self.lst_array = uv.lst_array
        self.Nblts = self.time_array.size

        if self.telescope_name is None and self.telescope_location is None:
            self.telescope_name = uv.telescope_name
            self.telescope_location = uv.telescope_location

        if (
            self.antenna_numbers is None
            and self.antenna_names is None
            and self.antenna_positions is None
        ):
            self.antenna_numbers = uv.antenna_numbers
            self.antenna_names = uv.antenna_names
            self.antenna_positions = uv.antenna_positions
            self.Nants_telescope = uv.Nants_telescope

        self._set_type_baseline()
        self.clear_unused_attributes()

        self.history += 'Broadcast to type "baseline". '

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_antenna(
        self,
        uv,
        force_pol=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert a UVFlag object of type "waterfall" to type "antenna".

        Broadcasts the flag array to all antennas.
        This function does NOT apply flags to uv (see utils.apply_uvflag for that).
        Note that the antenna metadata arrays (`antenna_names`, `antenna_numbers`
        and `antenna_positions`) may be reordered to match the ordering on `uv`.

        Parameters
        ----------
        uv : UVCal or UVFlag object
            object of type antenna to match.
        force_pol : bool
            If True, will use 1 pol to broadcast to any other pol.
            Otherwise, will require polarizations match.
            For example, this keyword is useful if one flags on all
            pols combined, and wants to broadcast back to individual pols.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to antenna type.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to antenna type.

        """
        if self.type == "antenna":
            return
        if not (
            issubclass(uv.__class__, UVCal)
            or (isinstance(uv, UVFlag) and uv.type == "antenna")
        ):
            raise ValueError(
                "Must pass in UVCal object or UVFlag object of type "
                '"antenna" to match.'
            )
        if self.type != "waterfall":
            raise ValueError(
                'Cannot convert from type "' + self.type + '" to "antenna".'
            )

        # write it out this rather than comparing the UVParameters because
        # future_array_shapes might be different. In the future, when shapes are not
        # variable, this can be done by comparing the UVParameters.
        if self.Nfreqs != uv.Nfreqs or not np.allclose(
            np.squeeze(self.freq_array),
            np.squeeze(uv.freq_array),
            rtol=self._freq_array.tols[0],
            atol=self._freq_array.tols[1],
        ):
            raise ValueError(
                "The freq_array on uv is not the same as the freq_array on this "
                f"object. The value on this object is {self.freq_array}; the value "
                f"on uv is {uv.freq_array}"
            )

        warn_compatibility_params = [
            "telescope_name",
            "telescope_location",
            "antenna_names",
            "antenna_numbers",
            "antenna_positions",
            "channel_width",
            "spw_array",
        ]
        if self.Nspws is not None and self.Nspws > 1:
            # TODO: make this always be in the compatibility list in version 3.0
            warn_compatibility_params.append("flex_spw_id_array")

        # sometimes the antenna sorting for the antenna names/numbers/positions
        # is different. If the sets are the same, re-sort self to match uv
        self.sort_ant_metadata_like(uv)

        for param in warn_compatibility_params:
            if (
                issubclass(uv.__class__, UVCal)
                and param == "channel_width"
                and not (uv.future_array_shapes or uv.flex_spw)
            ):
                if not np.allclose(
                    self.channel_width,
                    np.full(uv.Nfreqs, uv.channel_width),
                    rtol=self._channel_width.tols[0],
                    atol=self._channel_width.tols[1],
                ):
                    raise ValueError(
                        "channel_width is not the same this object and on uv. The "
                        f"value on this object is {self.channel_width}; the value on "
                        f"uv is {uv.channel_width}."
                    )
            else:
                # compare the UVParameter objects to properly handle tolerances
                this_param = getattr(self, "_" + param)
                uv_param = getattr(uv, "_" + param)
                if this_param.value is not None and this_param != uv_param:
                    raise ValueError(
                        f"{param} is not the same this object and on uv. The value on "
                        f"this object is {this_param.value}; the value on uv is "
                        f"{uv_param.value}."
                    )

        # Deal with polarization
        if issubclass(uv.__class__, UVCal):
            polarr = uv.jones_array
        else:
            polarr = uv.polarization_array
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = polarr
            # Broadcast arrays
            if self.mode == "flag":
                self.flag_array = self.flag_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            else:
                self.metric_array = self.metric_array.repeat(
                    self.polarization_array.size, axis=-1
                )
                self.weights_array = self.weights_array.repeat(
                    self.polarization_array.size, axis=-1
                )
            self.Npols = len(self.polarization_array)
            self._check_pol_state()

        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(polarr, self.polarization_array):
            if self.polarization_array.size == 1:
                raise ValueError(
                    "Polarizations do not match. Try keyword force_pol"
                    + "if you wish to broadcast to all polarizations."
                )
            else:
                raise ValueError("Polarizations could not be made to match.")
        # Populate arrays
        if self.mode == "flag":
            if self.future_array_shapes:
                self.flag_array = np.swapaxes(self.flag_array, 0, 1)[
                    np.newaxis, :, :, :
                ]
            else:
                self.flag_array = np.swapaxes(self.flag_array, 0, 1)[
                    np.newaxis, np.newaxis, :, :, :
                ]
            self.flag_array = self.flag_array.repeat(len(uv.ant_array), axis=0)
        elif self.mode == "metric":
            if self.future_array_shapes:
                self.metric_array = np.swapaxes(self.metric_array, 0, 1)[
                    np.newaxis, :, :, :
                ]
                self.weights_array = np.swapaxes(self.weights_array, 0, 1)[
                    np.newaxis, :, :, :
                ]
            else:
                self.metric_array = np.swapaxes(self.metric_array, 0, 1)[
                    np.newaxis, np.newaxis, :, :, :
                ]
                self.weights_array = np.swapaxes(self.weights_array, 0, 1)[
                    np.newaxis, np.newaxis, :, :, :
                ]
            self.metric_array = self.metric_array.repeat(len(uv.ant_array), axis=0)
            self.weights_array = self.weights_array.repeat(len(uv.ant_array), axis=0)
        self.ant_array = uv.ant_array
        self.Nants_data = len(uv.ant_array)
        # Check the frequency array for Nspws, otherwise broadcast to 1,Nfreqs
        if not self.future_array_shapes:
            self.freq_array = np.atleast_2d(self.freq_array)

        if self.telescope_name is None and self.telescope_location is None:
            self.telescope_name = uv.telescope_name
            self.telescope_location = uv.telescope_location

        if (
            self.antenna_numbers is None
            and self.antenna_names is None
            and self.antenna_positions is None
        ):
            self.antenna_numbers = uv.antenna_numbers
            self.antenna_names = uv.antenna_names
            self.antenna_positions = uv.antenna_positions
            self.Nants_telescope = uv.Nants_telescope

        if self.Nspws is None:
            self.Nspws = uv.Nspws
            self.spw_array = uv.spw_array
            if uv.flex_spw_id_array is not None:
                self.flex_spw_id_array = uv.flex_spw_id_array

        self._set_type_antenna()
        self.history += 'Broadcast to type "antenna". '

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_flag(
        self,
        threshold=np.inf,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert to flag mode.

        This function is NOT SMART. Removes metric_array and creates a
        flag_array from a simple threshold on the metric values.

        Parameters
        ----------
        threshold : float
            Metric value over which the corresponding flag is
            set to True. Default is np.inf, which results in flags of all False.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to flag mode.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to flag mode.

        """
        if self.mode == "flag":
            return
        elif self.mode == "metric":
            self.flag_array = np.where(self.metric_array >= threshold, True, False)
            self._set_mode_flag()
        else:
            raise ValueError(
                "Unknown UVFlag mode: " + self.mode + ". Cannot convert to flag."
            )
        self.history += 'Converted to mode "flag". '
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str
        self.clear_unused_attributes()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def to_metric(
        self,
        convert_wgts=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Convert to metric mode.

        This function is NOT SMART. Simply recasts flag_array as float
        and uses this as the metric array.

        Parameters
        ----------
        convert_wgts : bool
            if True convert self.weights_array to ones
            unless a column or row is completely flagged, in which case
            convert those pixels to zero. This is used when reinterpretting
            flags as metrics to calculate flag fraction. Zero weighting
            completely flagged rows/columns prevents those from counting
            against a threshold along the other dimension.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after converting to metric mode.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            converting to metric mode.

        """
        if self.mode == "metric":
            return
        elif self.mode == "flag":
            self.metric_array = self.flag_array.astype(np.float64)
            self._set_mode_metric()

            if convert_wgts:
                self.weights_array = np.ones_like(self.weights_array)
                if self.type == "waterfall":
                    for i in range(self.Npols):
                        self.weights_array[:, :, i] *= ~and_rows_cols(
                            self.flag_array[:, :, i]
                        )
                elif self.type == "baseline":
                    for i in range(self.Npols):
                        for ap in self.get_antpairs():
                            inds = self.antpair2ind(*ap)
                            if self.future_array_shapes:
                                self.weights_array[inds, :, i] *= ~and_rows_cols(
                                    self.flag_array[inds, :, i]
                                )
                            else:
                                self.weights_array[inds, 0, :, i] *= ~and_rows_cols(
                                    self.flag_array[inds, 0, :, i]
                                )
                elif self.type == "antenna":
                    for i in range(self.Npols):
                        for j in range(self.weights_array.shape[0]):
                            if self.future_array_shapes:
                                self.weights_array[j, :, :, i] *= ~and_rows_cols(
                                    self.flag_array[j, :, :, i]
                                )
                            else:
                                self.weights_array[j, 0, :, :, i] *= ~and_rows_cols(
                                    self.flag_array[j, 0, :, :, i]
                                )
        else:
            raise ValueError(
                "Unknown UVFlag mode: " + self.mode + ". Cannot convert to metric."
            )
        self.history += 'Converted to mode "metric". '

        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str
        self.clear_unused_attributes()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

    def __add__(
        self,
        other,
        inplace=False,
        axis="time",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Add two UVFlag objects together along a given axis.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        axis : str
            Axis along which to combine UVFlag objects.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data.

        Returns
        -------
        uvf : UVFlag
            If inplace==False, return new UVFlag object.

        """
        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that objects are compatible
        if not isinstance(other, this.__class__):
            raise ValueError("Only UVFlag objects can be added to a UVFlag object")
        if this.type != other.type:
            raise ValueError(
                "UVFlag object of type " + other.type + " cannot be "
                "added to object of type " + this.type + "."
            )
        if this.mode != other.mode:
            raise ValueError(
                "UVFlag object of mode " + other.mode + " cannot be "
                "added to object of mode " + this.type + "."
            )

        # check that both objects have the same array shapes
        if this.future_array_shapes != other.future_array_shapes:
            raise ValueError(
                "Both objects must have the same `future_array_shapes` parameter. "
                "Use the `use_future_array_shapes` or `use_current_array_shapes` "
                "methods to convert them."
            )

        this_has_spw_id = this.flex_spw_id_array is not None
        other_has_spw_id = other.flex_spw_id_array is not None
        if this_has_spw_id != other_has_spw_id:
            warnings.warn(
                "One object has the flex_spw_id_array set and one does not. Combined "
                "object will have it set."
            )

        # Update filename parameter
        this.filename = uvutils._combine_filenames(this.filename, other.filename)
        if this.filename is not None:
            this._filename.form = (len(this.filename),)

        # Simplify axis referencing
        axis = axis.lower()
        type_nums = {"waterfall": 0, "baseline": 1, "antenna": 2}
        if self.future_array_shapes:
            axis_nums = {
                "time": [0, 0, 2],
                "baseline": [None, 0, None],
                "antenna": [None, None, 0],
                "frequency": [1, 1, 1],
                "polarization": [2, 2, 3],
                "pol": [2, 2, 3],
                "jones": [2, 2, 3],
            }
        else:
            axis_nums = {
                "time": [0, 0, 3],
                "baseline": [None, 0, None],
                "antenna": [None, None, 0],
                "frequency": [1, 2, 2],
                "polarization": [2, 3, 4],
                "pol": [2, 3, 4],
                "jones": [2, 3, 4],
            }
        if axis not in axis_nums.keys():
            raise ValueError(f"Axis not recognized, must be one of {axis_nums.keys()}")

        ax = axis_nums[axis][type_nums[self.type]]

        warn_compatibility_params = ["telescope_name", "telescope_location"]

        if axis != "frequency":
            warn_compatibility_params.extend(
                ["freq_array", "channel_width", "spw_array"]
            )
            if self.flex_spw_id_array is not None:
                # TODO: make this always be in the compatibility list in version 3.0
                warn_compatibility_params.append("flex_spw_id_array")
        if axis not in ["polarization", "pol", "jones"]:
            warn_compatibility_params.extend(["polarization_array"])
        if axis != "time":
            warn_compatibility_params.extend(["time_array", "lst_array"])
        if axis != "antenna" and self.type == "antenna":
            warn_compatibility_params.extend(
                ["ant_array", "antenna_names", "antenna_numbers", "antenna_positions"]
            )
        if axis != "baseline" and self.type == "baseline":
            warn_compatibility_params.extend(
                [
                    "baseline_array",
                    "ant_1_array",
                    "ant_2_array",
                    "antenna_names",
                    "antenna_numbers",
                    "antenna_positions",
                ]
            )

        for param in warn_compatibility_params:
            # compare the UVParameter objects to properly handle tolerances
            this_param = getattr(self, "_" + param)
            other_param = getattr(other, "_" + param)
            if this_param.value is not None and this_param != other_param:
                raise ValueError(
                    f"{param} is not the same the two objects. The value on this "
                    f"object is {this_param.value}; the value on the other object is "
                    f"{other_param.value}."
                )

        if axis == "time":
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            if this.type == "baseline":
                this.baseline_array = np.concatenate(
                    [this.baseline_array, other.baseline_array]
                )
                this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
                this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
                this.Nants_data = int(
                    np.union1d(this.ant_1_array, this.ant_2_array).size
                )

            this.Ntimes = np.unique(this.time_array).size
            this.Nblts = len(this.time_array)

        elif axis == "baseline":
            if self.type != "baseline":
                raise ValueError(
                    "Flag object of type " + self.type + " cannot be "
                    "concatenated along baseline axis."
                )
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            this.baseline_array = np.concatenate(
                [this.baseline_array, other.baseline_array]
            )
            this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
            this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
            this.Nants_data = int(np.union1d(this.ant_1_array, this.ant_2_array).size)

            this.Nbls = np.unique(this.baseline_array).size
            this.Nblts = len(this.baseline_array)

        elif axis == "antenna":
            if self.type != "antenna":
                raise ValueError(
                    "Flag object of type " + self.type + " cannot be "
                    "concatenated along antenna axis."
                )
            this.ant_array = np.concatenate([this.ant_array, other.ant_array])
            this.Nants_data = len(this.ant_array)
            temp_ant_nums = np.concatenate(
                [this.antenna_numbers, other.antenna_numbers]
            )
            temp_ant_names = np.concatenate([this.antenna_names, other.antenna_names])
            temp_ant_pos = np.concatenate(
                [this.antenna_positions, other.antenna_positions], axis=0
            )
            this.antenna_numbers, unique_inds = np.unique(
                temp_ant_nums, return_index=True
            )
            this.antenna_names = temp_ant_names[unique_inds]
            this.antenna_positions = temp_ant_pos[unique_inds]
            this.Nants_telescope = len(this.antenna_numbers)

        elif axis == "frequency":
            this.freq_array = np.concatenate(
                [this.freq_array, other.freq_array], axis=-1
            )
            this.channel_width = np.concatenate(
                [this.channel_width, other.channel_width]
            )

            # handle multiple spws
            if this.Nspws > 1 or other.Nspws > 1 or this._spw_array != other._spw_array:
                if this.flex_spw_id_array is None:
                    this.flex_spw_id_array = np.full(
                        this.Nfreqs, this.spw_array[0], dtype=int
                    )
                if other.flex_spw_id_array is None:
                    other.flex_spw_id_array = np.full(
                        other.Nfreqs, other.spw_array[0], dtype=int
                    )

                this.flex_spw_id_array = np.concatenate(
                    [this.flex_spw_id_array, other.flex_spw_id_array]
                )
                this.spw_array = np.concatenate([this.spw_array, other.spw_array])
                # We want to preserve per-spw information based on first appearance
                # in the concatenated array.
                unique_index = np.sort(
                    np.unique(this.flex_spw_id_array, return_index=True)[1]
                )
                this.spw_array = this.flex_spw_id_array[unique_index]
                this.Nspws = len(this.spw_array)
            else:
                if this_has_spw_id or other_has_spw_id:
                    this.flex_spw_id_array = np.full(
                        this.freq_array.size, this.spw_array[0], dtype=int
                    )

            this.Nfreqs = np.unique(this.freq_array.flatten()).size

        elif axis in ["polarization", "pol", "jones"]:
            if this.pol_collapsed:
                raise NotImplementedError(
                    "Two UVFlag objects with their "
                    "polarizations collapsed cannot be "
                    "added along the polarization axis "
                    "at this time."
                )
            this.polarization_array = np.concatenate(
                [this.polarization_array, other.polarization_array]
            )
            this.Npols = len(this.polarization_array)

        for attr in this._data_params:
            # Check that 'other' also has the attribute filled
            if getattr(other, attr) is not None:
                setattr(
                    this,
                    attr,
                    np.concatenate(
                        [getattr(this, attr), getattr(other, attr)], axis=ax
                    ),
                )
            # May 21, 2020 - should only happen for weights_square_array attr
            else:
                raise ValueError(
                    f"{attr} optional parameter is missing from second UVFlag"
                    f" object. To concatenate two {this.mode} objects, they"
                    " must both contain the same optional parameters set."
                )

        this.history += "Data combined along " + axis + " axis. "
        if not uvutils._check_history_version(this.history, this.pyuvdata_version_str):
            this.history += this.pyuvdata_version_str

        this.Ntimes = np.unique(this.time_array).size

        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return this

    def __iadd__(
        self,
        other,
        axis="time",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """In place add.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        axis : str
            Axis along which to combine UVFlag objects.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.

        """
        self.__add__(
            other,
            inplace=True,
            axis=axis,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
        )
        return self

    def __or__(
        self,
        other,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Combine two UVFlag objects in "flag" mode by "OR"-ing their flags.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data.

        Returns
        -------
        uvf : UVFlag
            If inplace==False, return new UVFlag object.

        """
        if (self.mode != "flag") or (other.mode != "flag"):
            raise ValueError(
                'UVFlag object must be in "flag" mode to use "or" function.'
            )

        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()
        this.flag_array += other.flag_array
        if other.history not in this.history:
            this.history += "Flags OR'd with: " + other.history

        if not uvutils._check_history_version(this.history, this.pyuvdata_version_str):
            this.history += this.pyuvdata_version_str

        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return this

    def __ior__(
        self, other, run_check=True, check_extra=True, run_check_acceptability=True
    ):
        """Perform an inplace logical or.

        Parameters
        ----------
        other : UVFlag
            object to combine with self.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining two objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining two objects.

        """
        self.__or__(
            other,
            inplace=True,
            run_check=True,
            check_extra=True,
            run_check_acceptability=True,
        )
        return self

    def combine_metrics(
        self,
        others,
        method="quadmean",
        inplace=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Combine metric arrays between different UVFlag objects together.

        Parameters
        ----------
        others : UVFlag or list of UVFlags
            Other UVFlag objects to combine metrics with this one.
        method : str, {"quadmean", "absmean", "mean", "or", "and"}
            Method to combine metrics.
        inplace : bool, optional
            Perform combination in place.

        Returns
        -------
        uvf : UVFlag
            If inplace==False, return new UVFlag object with combined metrics.

        """
        # Ensure others is iterable (in case of single UVFlag object)
        # cannot use uvutils._get_iterable because the object itself is iterable
        if not isinstance(others, (list, tuple, np.ndarray)):
            others = [others]

        if np.any([not isinstance(other, UVFlag) for other in others]):
            raise ValueError('"others" must be UVFlag or list of UVFlag objects')
        if (self.mode != "metric") or np.any(
            [other.mode != "metric" for other in others]
        ):
            raise ValueError(
                'UVFlag object and "others" must be in "metric" mode '
                'to use "add_metrics" function.'
            )
        if inplace:
            this = self
        else:
            this = self.copy()
        method = method.lower()
        darray = np.expand_dims(this.metric_array, 0)
        warray = np.expand_dims(this.weights_array, 0)
        for other in others:
            if this.metric_array.shape != other.metric_array.shape:
                raise ValueError("UVFlag metric array shapes do not match.")
            darray = np.vstack([darray, np.expand_dims(other.metric_array, 0)])
            warray = np.vstack([warray, np.expand_dims(other.weights_array, 0)])
        darray, warray = uvutils.collapse(
            darray, method, weights=warray, axis=0, return_weights=True
        )
        this.metric_array = darray
        this.weights_array = warray
        this.history += "Combined metric arrays. "

        if not uvutils._check_history_version(this.history, this.pyuvdata_version_str):
            this.history += this.pyuvdata_version_str

        if run_check:
            this.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        if not inplace:
            return this

    def _select_preprocess(
        self,
        antenna_nums,
        ant_str,
        bls,
        frequencies,
        freq_chans,
        times,
        polarizations,
        blt_inds,
        ant_inds,
    ):
        """Build up blt_inds, freq_inds, pol_inds and history_update_string for select.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False).
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
            baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
            to keep in the object. For length-2 tuples, the ordering of the numbers
            within the tuple does not matter. For length-3 tuples, the polarization
            string is in the order of the two antennas. If length-3 tuples are
            provided, `polarizations` must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to keep in the object.  Can be 'auto', 'cross', 'all',
            or combinations of antenna numbers and polarizations (e.g. '1',
            '1_2', '1x_2y').  See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to keep in the object, each value passed here should
            exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        times : array_like of float, optional
            The times to keep in the object, each value passed here should
            exist in the time_array.
        polarizations : array_like of int or str, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array. If passing strings, the
            canonical polarization strings (e.g. "xx", "rr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "nn", "ee") are also supported.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.
        ant_inds : array_like of int, optional
            The antenna indices to keep in the object. This is
            not commonly used.

        Returns
        -------
        blt_inds : list of int
            list of baseline-time indices to keep. Can be None (to keep everything).
        ant_inds : list of int
            list of antenna number indices to keep. Can be None
            (keep all; only valid for "antenna" mode).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        pol_inds : list of int
            list of polarization indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.

        """
        # build up history string as we go
        history_update_string = "  Downselected to specific "
        n_selects = 0

        if self.type == "waterfall":
            if antenna_nums is not None:
                raise ValueError(
                    "Cannot select on antenna_nums with waterfall type "
                    "UVFlag objects."
                )
            if bls is not None:
                raise ValueError(
                    "Cannot select on bls with waterfall type UVFlag objects."
                )

        if ant_str is not None:
            if not (antenna_nums is None and bls is None and polarizations is None):
                raise ValueError(
                    "Cannot provide ant_str with antenna_nums, bls, or polarizations."
                )
            else:
                bls, polarizations = self.parse_ants(ant_str)
                if bls is not None and len(bls) == 0:
                    raise ValueError(
                        f"There is no data matching ant_str={ant_str} in this object."
                    )

        # Antennas, times and blt_inds all need to be combined into a set of
        # blts indices to keep.

        # test for blt_inds presence before adding inds from antennas & times
        if blt_inds is not None:
            blt_inds = uvutils._get_iterable(blt_inds)
            if np.array(blt_inds).ndim > 1:
                blt_inds = np.array(blt_inds).flatten()
            if self.type == "baseline":
                history_update_string += "baseline-times"
            else:
                history_update_string += "times"
            n_selects += 1

        if antenna_nums is not None:
            antenna_nums = uvutils._get_iterable(antenna_nums)
            if np.array(antenna_nums).ndim > 1:
                antenna_nums = np.array(antenna_nums).flatten()
            if n_selects > 0:
                history_update_string += ", antennas"
            else:
                history_update_string += "antennas"
            n_selects += 1

            if self.type == "baseline":
                inds1 = np.zeros(0, dtype=np.int64)
                inds2 = np.zeros(0, dtype=np.int64)
                for ant in antenna_nums:
                    if ant in self.ant_1_array or ant in self.ant_2_array:
                        wh1 = np.where(self.ant_1_array == ant)[0]
                        wh2 = np.where(self.ant_2_array == ant)[0]
                        if len(wh1) > 0:
                            inds1 = np.append(inds1, list(wh1))
                        if len(wh2) > 0:
                            inds2 = np.append(inds2, list(wh2))
                    else:
                        raise ValueError(
                            "Antenna number {a} is not present in the "
                            "ant_1_array or ant_2_array".format(a=ant)
                        )
                ant_blt_inds = set(inds1).intersection(inds2)

            if self.type == "antenna":
                ant_blt_inds = None
                ant_inds = np.zeros(0, dtype=np.int64)
                for ant in antenna_nums:
                    if ant in self.ant_array:
                        wh = np.nonzero(self.ant_array == ant)[0]
                        if len(wh) > 0:
                            ant_inds = np.append(ant_inds, list(wh))
                    else:
                        raise ValueError(
                            "Antenna number {a} is not present in the "
                            "ant_array".format(a=ant)
                        )

        else:
            ant_blt_inds = None

        if bls is not None:
            if self.type != "baseline":
                raise ValueError(
                    'Only "baseline" mode UVFlag objects may select'
                    " along the baseline axis"
                )
            if isinstance(bls, tuple) and (len(bls) == 2 or len(bls) == 3):
                bls = [bls]
            if not all(isinstance(item, tuple) for item in bls):
                raise ValueError(
                    "bls must be a list of tuples of antenna numbers "
                    "(optionally with polarization)."
                )
            if not all(
                [isinstance(item[0], (int, np.integer)) for item in bls]
                + [isinstance(item[1], (int, np.integer)) for item in bls]
            ):
                raise ValueError(
                    "bls must be a list of tuples of integer antenna numbers "
                    "(optionally with polarization)."
                )
            if all(len(item) == 3 for item in bls):
                if polarizations is not None:
                    raise ValueError(
                        "Cannot provide length-3 tuples and also specify polarizations."
                    )
                if not all(isinstance(item[2], str) for item in bls):
                    raise ValueError(
                        "The third element in each bl must be a polarization string"
                    )

            if n_selects > 0:
                history_update_string += ", baselines"
            else:
                history_update_string += "baselines"

            n_selects += 1
            bls_blt_inds = np.zeros(0, dtype=np.int64)
            bl_pols = set()
            for bl in bls:
                if not (bl[0] in self.ant_1_array or bl[0] in self.ant_2_array):
                    raise ValueError(
                        "Antenna number {a} is not present in the "
                        "ant_1_array or ant_2_array".format(a=bl[0])
                    )
                if not (bl[1] in self.ant_1_array or bl[1] in self.ant_2_array):
                    raise ValueError(
                        "Antenna number {a} is not present in the "
                        "ant_1_array or ant_2_array".format(a=bl[1])
                    )
                wh1 = np.where(
                    np.logical_and(self.ant_1_array == bl[0], self.ant_2_array == bl[1])
                )[0]
                wh2 = np.where(
                    np.logical_and(self.ant_1_array == bl[1], self.ant_2_array == bl[0])
                )[0]
                if len(wh1) > 0:
                    bls_blt_inds = np.append(bls_blt_inds, list(wh1))
                    if len(bl) == 3:
                        bl_pols.add(bl[2])
                elif len(wh2) > 0:
                    bls_blt_inds = np.append(bls_blt_inds, list(wh2))
                    if len(bl) == 3:
                        bl_pols.add(uvutils.conj_pol(bl[2]))
                else:
                    raise ValueError(
                        "Antenna pair {p} does not have any data "
                        "associated with it.".format(p=bl)
                    )
            if len(bl_pols) > 0:
                polarizations = list(bl_pols)

            if ant_blt_inds is not None:
                # Use intersection (and) to join antenna_names/nums & ant_pairs_nums
                ant_blt_inds = set(ant_blt_inds).intersection(bls_blt_inds)
            else:
                ant_blt_inds = bls_blt_inds

        if ant_blt_inds is not None:
            if blt_inds is not None:
                # Use intersection (and) to join
                # antenna_names/nums/ant_pairs_nums with blt_inds
                blt_inds = set(blt_inds).intersection(ant_blt_inds)
            else:
                blt_inds = ant_blt_inds

        if times is not None:
            times = uvutils._get_iterable(times)
            if np.array(times).ndim > 1:
                times = np.array(times).flatten()

            if n_selects > 0:
                if (
                    self.type != "baseline" and "times" not in history_update_string
                ) or self.type == "baseline":
                    history_update_string += ", times"
            else:
                history_update_string += "times"

            n_selects += 1

            time_blt_inds = np.zeros(0, dtype=np.int64)
            for jd in times:
                if jd in self.time_array:
                    time_blt_inds = np.append(
                        time_blt_inds, np.where(self.time_array == jd)[0]
                    )
                else:
                    raise ValueError(
                        "Time {t} is not present in the time_array".format(t=jd)
                    )

            if blt_inds is not None:
                # Use intesection (and) to join
                # antenna_names/nums/ant_pairs_nums/blt_inds with times
                blt_inds = set(blt_inds).intersection(time_blt_inds)
            else:
                blt_inds = time_blt_inds

        if blt_inds is not None:
            if len(blt_inds) == 0:
                raise ValueError("No baseline-times were found that match criteria")

            if self.type == "baseline":
                compare_length = self.Nblts
            else:
                compare_length = self.Ntimes

            if max(blt_inds) >= compare_length:
                raise ValueError("blt_inds contains indices that are too large")
            if min(blt_inds) < 0:
                raise ValueError("blt_inds contains indices that are negative")

            blt_inds = sorted(set(blt_inds))

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if np.array(freq_chans).ndim > 1:
                freq_chans = np.array(freq_chans).flatten()
            if frequencies is None:
                frequencies = np.squeeze(self.freq_array)[freq_chans]

            else:
                frequencies = uvutils._get_iterable(frequencies)
                frequencies = np.sort(
                    list(
                        set(frequencies) | set(np.squeeze(self.freq_array)[freq_chans])
                    )
                )

        if frequencies is not None:
            frequencies = uvutils._get_iterable(frequencies)
            if np.array(frequencies).ndim > 1:
                frequencies = np.array(frequencies).flatten()
            if n_selects > 0:
                history_update_string += ", frequencies"
            else:
                history_update_string += "frequencies"
            n_selects += 1

            freq_inds = np.zeros(0, dtype=np.int64)
            # this works because we only allow one SPW. This will have to be
            # reworked when we support more.
            if self.type != "waterfall" and not self.future_array_shapes:
                freq_arr_use = self.freq_array[0, :]
            else:
                freq_arr_use = self.freq_array
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError(
                        "Frequency {f} is not present in the freq_array".format(f=f)
                    )

            freq_inds = sorted(set(freq_inds))
        else:
            freq_inds = None

        if polarizations is not None:
            polarizations = uvutils._get_iterable(polarizations)
            if np.array(polarizations).ndim > 1:
                polarizations = np.array(polarizations).flatten()
            if n_selects > 0:
                history_update_string += ", polarizations"
            else:
                history_update_string += "polarizations"
            n_selects += 1

            pol_inds = np.zeros(0, dtype=np.int64)
            for p in polarizations:
                if isinstance(p, str):
                    p_num = uvutils.polstr2num(p, x_orientation=self.x_orientation)
                else:
                    p_num = p
                if p_num in self.polarization_array:
                    pol_inds = np.append(
                        pol_inds, np.where(self.polarization_array == p_num)[0]
                    )
                else:
                    raise ValueError(
                        "Polarization {p} is not present in the "
                        "polarization_array".format(p=p)
                    )

            pol_inds = sorted(set(pol_inds))
        else:
            pol_inds = None

        history_update_string += " using pyuvdata."

        return blt_inds, ant_inds, freq_inds, pol_inds, history_update_string

    def _select_metadata(
        self, blt_inds, ant_inds, freq_inds, pol_inds, history_update_string
    ):
        """Perform select on everything except the data-sized arrays.

        Parameters
        ----------
        blt_inds : list of int
            list of baseline-time indices to keep. Can be None (to keep everything).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        pol_inds : list of int
            list of polarization indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.
        keep_all_metadata : bool
            Option to keep metadata for antennas that are no longer in the dataset.

        """
        if blt_inds is not None:
            if self.type == "baseline":
                self.Nblts = len(blt_inds)
                self.baseline_array = self.baseline_array[blt_inds]
                self.Nbls = len(np.unique(self.baseline_array))
                self.ant_1_array = self.ant_1_array[blt_inds]
                self.ant_2_array = self.ant_2_array[blt_inds]
                self.Nants_data = int(
                    np.union1d(self.ant_1_array, self.ant_2_array).size
                )

            self.time_array = self.time_array[blt_inds]
            self.lst_array = self.lst_array[blt_inds]
            self.Ntimes = len(np.unique(self.time_array))

        if self.type == "antenna":
            if ant_inds is not None:
                self.ant_array = self.ant_array[ant_inds]
                self.Nants_data = int(len(self.ant_array))

        if freq_inds is not None:
            self.Nfreqs = len(freq_inds)
            if self.type != "waterfall" and not self.future_array_shapes:
                self.freq_array = self.freq_array[:, freq_inds]
            else:
                self.freq_array = self.freq_array[freq_inds]
            if self.channel_width is not None:
                self.channel_width = self.channel_width[freq_inds]
            if self.flex_spw_id_array is not None:
                self.flex_spw_id_array = self.flex_spw_id_array[freq_inds]

            if self.Nspws > 1:
                self.spw_array = self.spw_array[
                    np.where(np.isin(self.spw_array, self.flex_spw_id_array))[0]
                ]
                self.Nspws = self.spw_array.size

        if pol_inds is not None:
            self.Npols = len(pol_inds)
            self.polarization_array = self.polarization_array[pol_inds]

        self.history = self.history + history_update_string

    def select(
        self,
        antenna_nums=None,
        ant_inds=None,
        bls=None,
        ant_str=None,
        frequencies=None,
        freq_chans=None,
        times=None,
        polarizations=None,
        blt_inds=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        inplace=True,
    ):
        """
        Downselect data to keep on the object along various axes.

        Axes that can be selected along depend on the current type of the object.
        However some axis may always be selected upon, these include frequencies,
        times and polarizations.
        In "baseline" and "antenna" modes, antenna numbers may be selected.
        In "baseline" mode, antenna pairs may be selected.
        Specific baseline-time indices can also be selected in "baseline" mode,
        but this is not commonly used.
        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
            baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
            to keep in the object. For length-2 tuples, the ordering of the numbers
            within the tuple does not matter. For length-3 tuples, the polarization
            string is in the order of the two antennas. If length-3 tuples are
            provided, `polarizations` must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to keep in the object.  Can be 'auto', 'cross', 'all',
            or combinations of antenna numbers and polarizations (e.g. '1',
            '1_2', '1x_2y').  See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to keep in the object, each value passed here should
            exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        times : array_like of float, optional
            The times to keep in the object, each value passed here should
            exist in the time_array.
        polarizations : array_like of int or str, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array. If passing strings, the
            canonical polarization strings (e.g. "xx", "rr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "nn", "ee") are also supported.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.
        ant_inds : array_like of int, optional
            The antenna indices to keep in the object. This is
            not commonly used.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data.

        Returns
        -------
        UVData object or None
            None is returned if inplace is True, otherwise a new UVData object
            with just the selected data is returned

        Raises
        ------
        ValueError
            If any of the parameters are set to inappropriate values.

        """
        if inplace:
            uv_object = self
        else:
            uv_object = self.copy()

        (blt_inds, ant_inds, freq_inds, pol_inds, history_update_string) = (
            uv_object._select_preprocess(
                antenna_nums=antenna_nums,
                ant_str=ant_str,
                bls=bls,
                frequencies=frequencies,
                freq_chans=freq_chans,
                times=times,
                polarizations=polarizations,
                blt_inds=blt_inds,
                ant_inds=ant_inds,
            )
        )

        # do select operations on everything except data_array, flag_array
        # and nsample_array
        uv_object._select_metadata(
            blt_inds, ant_inds, freq_inds, pol_inds, history_update_string
        )

        if blt_inds is not None:
            if self.type == "baseline":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[blt_inds])
            if self.type == "waterfall":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[blt_inds])
            if self.type == "antenna":
                if self.future_array_shapes:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, blt_inds, :])
                else:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, :, blt_inds, :])

        if ant_inds is not None and self.type == "antenna":
            for param_name, param in zip(
                self._data_params, uv_object.data_like_parameters
            ):
                setattr(uv_object, param_name, param[ant_inds])

        if freq_inds is not None:
            if self.type == "baseline":
                if self.future_array_shapes:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, freq_inds, :])
                else:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, freq_inds, :])
            if self.type == "waterfall":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, freq_inds, :])
            if self.type == "antenna":
                if self.future_array_shapes:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, freq_inds, :, :])
                else:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, freq_inds, :, :])

        if pol_inds is not None:
            if self.type == "baseline":
                if self.future_array_shapes:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, pol_inds])
                else:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, :, pol_inds])
            if self.type == "waterfall":
                for param_name, param in zip(
                    self._data_params, uv_object.data_like_parameters
                ):
                    setattr(uv_object, param_name, param[:, :, pol_inds])
            if self.type == "antenna":
                if self.future_array_shapes:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, :, pol_inds])
                else:
                    for param_name, param in zip(
                        self._data_params, uv_object.data_like_parameters
                    ):
                        setattr(uv_object, param_name, param[:, :, :, :, pol_inds])

        # check if object is uv_object-consistent
        if run_check:
            uv_object.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if not inplace:
            return uv_object

    def read(
        self,
        filename,
        history="",
        mwa_metafits_file=None,
        telescope_name=None,
        use_future_array_shapes=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Read in flag/metric data from a HDF5 file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The file name to read.
        history : str
            History string to append to UVFlag history attribute.
        mwa_metafits_file : str, optional
            For MWA data only, the metafits file corresponding to the data in filename.
            This should only be set when reading in old files that do not have telescope
            metadata in them. Passing in the metafits file for old files allows for
            all the telescope metadata (e.g. connected antennas and positions) to be
            set. Setting this parameter overrides any telescope metadata in the file.
        telescope_name : str, optional
            Name of the telescope. This should only be set when reading in old files
            that do not have the telescope name in them. Setting this parameter for old
            files allows for other telescope metadata to be set from the known
            telescopes. Setting this parameter overrides any telescope name in the file.
            This should not be set if `mwa_metafits_file` is passed.
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reading data.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading data.

        """
        # make sure we have an empty object.
        self.__init__()
        if isinstance(filename, (tuple, list)):
            self.read(filename[0], use_future_array_shapes=use_future_array_shapes)
            if len(filename) > 1:
                for f in filename[1:]:
                    f2 = UVFlag(
                        f,
                        history=history,
                        use_future_array_shapes=use_future_array_shapes,
                    )
                    self += f2
                del f2

        else:
            if not os.path.exists(filename):
                raise IOError(filename + " not found.")

            # update filename attribute
            basename = os.path.basename(filename)
            self.filename = [basename]
            self._filename.form = (1,)

            # Open file for reading
            with h5py.File(filename, "r") as f:
                header = f["/Header"]

                self.type = header["type"][()].decode("utf8")
                if self.type == "antenna":
                    self._set_type_antenna()
                elif self.type == "baseline":
                    self._set_type_baseline()
                elif self.type == "waterfall":
                    self._set_type_waterfall()
                else:
                    raise ValueError(
                        "File cannot be read. Received type "
                        "parameter: {receive} but "
                        "must be within acceptable values: "
                        "{expect}".format(
                            receive=self.type,
                            expect=(", ").join(self._type.acceptable_vals),
                        )
                    )

                self.mode = header["mode"][()].decode("utf8")

                if self.mode == "metric":
                    self._set_mode_metric()
                elif self.mode == "flag":
                    self._set_mode_flag()
                else:
                    raise ValueError(
                        "File cannot be read. Received mode "
                        "parameter: {receive} but "
                        "must be within acceptable values: "
                        "{expect}".format(
                            receive=self.mode,
                            expect=(", ").join(self._mode.acceptable_vals),
                        )
                    )

                if "x_orientation" in header.keys():
                    self.x_orientation = header["x_orientation"][()].decode("utf8")

                self.time_array = header["time_array"][()]
                if "Ntimes" in header.keys():
                    self.Ntimes = int(header["Ntimes"][()])
                else:
                    self.Ntimes = np.unique(self.time_array).size

                self.lst_array = header["lst_array"][()]

                # read data arrays to figure out if the file has future shapes or not
                future_shapes_ndim = {"antenna": 4, "baseline": 3}
                dgrp = f["/Data"]
                if self.mode == "metric":
                    self.metric_array = dgrp["metric_array"][()]
                    if self.type != "waterfall":
                        if self.metric_array.ndim == future_shapes_ndim[self.type]:
                            self._set_future_array_shapes()

                    self.weights_array = dgrp["weights_array"][()]
                    if "weights_square_array" in dgrp:
                        self.weights_square_array = dgrp["weights_square_array"][()]

                elif self.mode == "flag":
                    self.flag_array = dgrp["flag_array"][()]
                    if self.type != "waterfall":
                        if self.flag_array.ndim == future_shapes_ndim[self.type]:
                            self._set_future_array_shapes()

                self.freq_array = header["freq_array"][()]
                # older save files will not have this spws axis
                # at least_2d will preserve shape of 2d arrays and
                # promote 1D to (1, Nfreqs)
                if self.type != "waterfall" and not self.future_array_shapes:
                    self.freq_array = np.atleast_2d(self.freq_array)
                elif self.freq_array.ndim > 1:
                    self.freq_array = np.squeeze(self.freq_array)

                if "Nfreqs" in header.keys():
                    self.Nfreqs = int(header["Nfreqs"][()])
                else:
                    self.Nfreqs = np.unique(self.freq_array).size

                if "channel_width" in header.keys():
                    self.channel_width = header["channel_width"][()]
                else:
                    # older files do not have the channel_width parameter. Guess it from
                    # the freq array spacing.
                    msg = (
                        "channel_width not available in file, computing it from the "
                        "freq_array spacing."
                    )
                    freq_delta = np.diff(np.squeeze(self.freq_array))
                    if uvutils._test_array_constant_spacing(
                        self.freq_array, tols=self._freq_array.tols
                    ):
                        self.channel_width = np.full(self.Nfreqs, freq_delta[0])
                    else:
                        msg += (
                            " The freq_array does not have equal spacing, so the last "
                            "channel_width is set equal to the channel width below it."
                        )
                        self.channel_width = np.concatenate(
                            (freq_delta, np.array([freq_delta[-1]]))
                        )
                    warnings.warn(msg)

                if "spw_array" in header.keys():
                    self.spw_array = header["spw_array"][()]
                else:
                    self.spw_array = np.array([0])

                if "Nspws" in header.keys():
                    self.Nspws = int(header["Nspws"][()])
                else:
                    self.Nspws = self.spw_array.size

                if "flex_spw_id_array" in header.keys():
                    self.flex_spw_id_array = header["flex_spw_id_array"][()]
                elif self.Nspws == 1:
                    # set it by default
                    self.flex_spw_id_array = np.full(
                        self.Nfreqs, self.spw_array[0], dtype=int
                    )

                if mwa_metafits_file is not None:
                    from ..uvdata.mwa_corr_fits import read_metafits

                    meta_dict = read_metafits(
                        mwa_metafits_file, telescope_info_only=True
                    )

                    self.telescope_name = meta_dict["telescope_name"]
                    self.telescope_location = meta_dict["telescope_location"]
                    self.antenna_numbers = meta_dict["antenna_numbers"]
                    self.antenna_names = meta_dict["antenna_names"]
                    self.antenna_positions = meta_dict["antenna_positions"]

                    override_params = []
                    params_to_check = [
                        "telescope_name",
                        "telescope_location",
                        "antenna_numbers",
                        "antenna_names",
                        "antenna_positions",
                    ]
                    for param in params_to_check:
                        if param in header.keys():
                            override_params.append(param)

                    if len(override_params) > 0:
                        warnings.warn(
                            "An mwa_metafits_file was passed. The metadata from the "
                            "metafits file are overriding the following parameters in "
                            f"the UVFlag file: {override_params}"
                        )
                else:
                    if telescope_name is not None:
                        self.telescope_name = telescope_name

                    if "telescope_name" in header.keys():
                        file_telescope_name = header["telescope_name"][()].decode(
                            "utf8"
                        )
                        if telescope_name is not None:
                            if telescope_name.lower() != file_telescope_name.lower():
                                warnings.warn(
                                    f"Telescope_name parameter is set to "
                                    f"{telescope_name}, which overrides the telescope "
                                    f"name in the file ({file_telescope_name})."
                                )
                        else:
                            self.telescope_name = file_telescope_name

                    if "telescope_location" in header.keys():
                        self.telescope_location = header["telescope_location"][()]

                    if "antenna_numbers" in header.keys():
                        self.antenna_numbers = header["antenna_numbers"][()]

                    if "antenna_names" in header.keys():
                        self.antenna_names = np.array(
                            [
                                bytes(n).decode("utf8")
                                for n in header["antenna_names"][:]
                            ]
                        )

                    if "antenna_positions" in header.keys():
                        self.antenna_positions = header["antenna_positions"][()]

                self.history = header["history"][()].decode("utf8")

                self.history += history

                if not uvutils._check_history_version(
                    self.history, self.pyuvdata_version_str
                ):
                    self.history += self.pyuvdata_version_str

                # get extra_keywords
                if "extra_keywords" in header.keys():
                    self.extra_keywords = {}
                    for key in header["extra_keywords"].keys():
                        if header["extra_keywords"][key].dtype.type in (
                            np.string_,
                            np.object_,
                        ):
                            self.extra_keywords[key] = bytes(
                                header["extra_keywords"][key][()]
                            ).decode("utf8")
                        else:
                            self.extra_keywords[key] = header["extra_keywords"][key][()]
                else:
                    self.extra_keywords = {}

                if "label" in header.keys():
                    self.label = header["label"][()].decode("utf8")

                polarization_array = header["polarization_array"][()]
                if isinstance(polarization_array[0], np.string_):
                    polarization_array = np.asarray(polarization_array, dtype=np.str_)
                self.polarization_array = polarization_array
                self._check_pol_state()

                if "Npols" in header.keys():
                    self.Npols = int(header["Npols"][()])
                else:
                    self.Npols = len(self.polarization_array)

                if self.type == "baseline":
                    self.ant_1_array = header["ant_1_array"][()]
                    self.ant_2_array = header["ant_2_array"][()]

                    self.baseline_array = self.antnums_to_baseline(
                        self.ant_1_array, self.ant_2_array
                    )

                    if "Nblts" in header.keys():
                        self.Nblts = int(header["Nblts"][()])
                    else:
                        self.Nblts = len(self.baseline_array)

                    self.Nbls = np.unique(self.baseline_array).size

                    if "Nants_data" in header.keys():
                        self.Nants_data = int(header["Nants_data"][()])

                        n_ants_detected = int(
                            np.union1d(self.ant_1_array, self.ant_2_array).size
                        )
                        if self.Nants_data != n_ants_detected:
                            warnings.warn(
                                "Nants_data in file does not match number of antennas "
                                "with data. Resetting Nants_data."
                            )
                            self.Nants_data = n_ants_detected
                    else:
                        self.Nants_data = int(
                            np.union1d(self.ant_1_array, self.ant_2_array).size
                        )

                elif self.type == "antenna":
                    self.ant_array = header["ant_array"][()]
                    if "Nants_data" in header.keys():
                        self.Nants_data = int(header["Nants_data"][()])
                    else:
                        self.Nants_data = len(self.ant_array)

                if "Nants_telescope" in header.keys():
                    self.Nants_telescope = int(header["Nants_telescope"][()])

                if self.telescope_name is None:
                    warnings.warn(
                        "telescope_name not available in file, so telescope related "
                        "parameters cannot be set. This will result in errors when the "
                        "object is checked. To avoid the errors, either set the "
                        "`telescope_name` parameter or use `run_check=False` "
                        "to turn off the check."
                    )
                elif (
                    self.telescope_location is None
                    or self.antenna_numbers is None
                    or self.antenna_names is None
                    or self.antenna_positions is None
                ):
                    if (
                        self.antenna_numbers is None
                        and self.antenna_names is None
                        and self.antenna_positions is None
                    ):
                        self.Nants_telescope = None

                    if "mwa" in self.telescope_name.lower() and (
                        self.antenna_numbers is None
                        or self.antenna_names is None
                        or self.antenna_positions is None
                    ):
                        warnings.warn(
                            "Antenna metadata are missing for this file. Since this "
                            "is MWA data, the best way to fill in these metadata is to "
                            "pass in an mwa_metafits_file which contains information "
                            "about which antennas were connected when the data were "
                            "taken. Since that was not passed, the antenna metadata "
                            "will be filled in from a static csv file containing all "
                            "the antennas that could have been connected."
                        )
                    self.set_telescope_params()

                if self.antenna_numbers is None and self.type in [
                    "baseline",
                    "antenna",
                ]:
                    msg = "antenna_numbers not in file"
                    if (
                        self.Nants_telescope is None
                        or self.Nants_telescope == self.Nants_data
                    ):
                        if self.type == "baseline":
                            msg += ", setting based on ant_1_array and ant_2_array."
                            self.antenna_numbers = np.unique(
                                np.union1d(self.ant_1_array, self.ant_2_array)
                            )
                        else:
                            msg += ", setting based on ant_array."
                            self.antenna_numbers = np.unique(self.ant_array)
                    else:
                        msg += ", cannot be set based on "
                        if self.type == "baseline":
                            msg += "ant_1_array and ant_2_array"
                        else:
                            msg += "ant_array"
                        msg += (
                            " because Nants_telescope is greater than Nants_data. This "
                            "will result in errors when the object is checked. To "
                            "avoid the errors, use `run_check=False` to turn off the "
                            "check."
                        )
                    warnings.warn(msg)

                if self.antenna_names is None and self.antenna_numbers is not None:
                    warnings.warn(
                        "antenna_names not in file, setting based on antenna_numbers"
                    )
                    self.antenna_names = self.antenna_numbers.astype(str)

                if self.Nants_telescope is None:
                    if self.antenna_numbers is not None:
                        self.Nants_telescope = self.antenna_numbers.size
                    elif self.antenna_names is not None:
                        self.Nants_telescope = self.antenna_names.size
                    elif self.antenna_positions is not None:
                        self.Nants_telescope = (self.antenna_positions.shape)[0]

            self.clear_unused_attributes()

            if use_future_array_shapes != self.future_array_shapes:
                if use_future_array_shapes:
                    self.use_future_array_shapes()
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="This method will be removed in version 3.0 when "
                            "the current array shapes are no longer supported.",
                        )
                        self.use_current_array_shapes()

            if not use_future_array_shapes:
                warnings.warn(_future_array_shapes_warning, DeprecationWarning)

            if run_check:
                self.check(
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                )

    def write(self, filename, clobber=False, data_compression="lzf"):
        """Write a UVFlag object to a hdf5 file.

        Parameters
        ----------
        filename : str
            The file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        data_compression : str
            HDF5 filter to apply when writing the data_array.
            If no compression is wanted, set to None.

        """
        if os.path.exists(filename):
            if clobber:
                print("File " + filename + " exists; clobbering")
            else:
                raise ValueError("File " + filename + " exists; skipping")

        with h5py.File(filename, "w") as f:
            header = f.create_group("Header")

            # write out metadata
            if self.future_array_shapes:
                # this is Version 1.0
                header["version"] = np.string_("1.0")
            else:
                header["version"] = np.string_("0.1")

            header["type"] = np.string_(self.type)
            header["mode"] = np.string_(self.mode)

            if self.telescope_name is not None:
                header["telescope_name"] = np.string_(self.telescope_name)
            if self.telescope_location is not None:
                header["telescope_location"] = self.telescope_location

            header["Ntimes"] = self.Ntimes
            header["time_array"] = self.time_array
            header["lst_array"] = self.lst_array

            header["freq_array"] = self.freq_array
            header["Nfreqs"] = self.Nfreqs
            header["channel_width"] = self.channel_width

            header["Nspws"] = self.Nspws
            header["spw_array"] = self.spw_array
            if self.flex_spw_id_array is not None:
                header["flex_spw_id_array"] = self.flex_spw_id_array

            header["Npols"] = self.Npols

            if self.x_orientation is not None:
                header["x_orientation"] = np.string_(self.x_orientation)

            if isinstance(self.polarization_array.item(0), str):
                polarization_array = np.asarray(
                    self.polarization_array, dtype=np.string_
                )
            else:
                polarization_array = self.polarization_array
            header["polarization_array"] = polarization_array

            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            # write out extra keywords if it exists and has elements
            if self.extra_keywords:
                extra_keywords = header.create_group(
                    "extra_keywords"
                )  # create spot in header
                for k in self.extra_keywords.keys():
                    if isinstance(self.extra_keywords[k], str):
                        extra_keywords[k] = np.string_(self.extra_keywords[k])
                    else:
                        extra_keywords[k] = self.extra_keywords[k]

            header["history"] = np.string_(self.history)

            header["label"] = np.string_(self.label)

            if self.type == "baseline":
                header["Nblts"] = self.Nblts
                header["ant_1_array"] = self.ant_1_array
                header["ant_2_array"] = self.ant_2_array
                header["Nants_data"] = self.Nants_data

            elif self.type == "antenna":
                header["ant_array"] = self.ant_array
                header["Nants_data"] = self.Nants_data

            header["Nants_telescope"] = self.Nants_telescope
            if self.antenna_names is not None:
                header["antenna_names"] = np.asarray(self.antenna_names, dtype="bytes")
            if self.antenna_numbers is not None:
                header["antenna_numbers"] = self.antenna_numbers
            if self.antenna_positions is not None:
                header["antenna_positions"] = self.antenna_positions

            dgrp = f.create_group("Data")
            if self.mode == "metric":
                dgrp.create_dataset(
                    "metric_array",
                    chunks=True,
                    data=self.metric_array,
                    compression=data_compression,
                )
                dgrp.create_dataset(
                    "weights_array",
                    chunks=True,
                    data=self.weights_array,
                    compression=data_compression,
                )
                if self.weights_square_array is not None:
                    dgrp.create_dataset(
                        "weights_square_array",
                        chunks=True,
                        data=self.weights_square_array,
                        compression=data_compression,
                    )
            elif self.mode == "flag":
                dgrp.create_dataset(
                    "flag_array",
                    chunks=True,
                    data=self.flag_array,
                    compression=data_compression,
                )

    def from_uvdata(
        self,
        indata,
        mode="metric",
        copy_flags=False,
        waterfall=False,
        history="",
        label="",
        use_future_array_shapes=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Construct a UVFlag object from a UVData object.

        Parameters
        ----------
        indata : UVData
            Input to initialize UVFlag object.
        mode : {"metric", "flag"}, optional
            The mode determines whether the object has a floating point metric_array
            or a boolean flag_array.
        copy_flags : bool, optional
            Whether to copy flags from indata to new UVFlag object
        waterfall : bool, optional
            Whether to immediately initialize as a waterfall object, with flag/metric
            axes: time, frequency, polarization.
        history : str, optional
            History string to attach to object.
        label: str, optional
            String used for labeling the object (e.g. 'FM').
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after creating UVFlag object.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            creating UVFlag object.

        """
        if not issubclass(indata.__class__, UVData):
            raise ValueError(
                "from_uvdata can only initialize a UVFlag object from an input "
                "UVData object or a subclass of a UVData object."
            )

        if mode.lower() == "metric":
            self._set_mode_metric()
        elif mode.lower() == "flag":
            self._set_mode_flag()
        else:
            raise ValueError(
                "Input mode must be within acceptable values: "
                "{}".format((", ").join(self._mode.acceptable_vals))
            )

        if use_future_array_shapes:
            self._set_future_array_shapes()

        self.Nfreqs = indata.Nfreqs
        self.polarization_array = copy.deepcopy(indata.polarization_array)
        self.Npols = indata.Npols
        self.Nants_telescope = indata.Nants_telescope
        self.Ntimes = indata.Ntimes

        if indata.future_array_shapes or indata.flex_spw:
            self.channel_width = copy.deepcopy(indata.channel_width)
        else:
            self.channel_width = np.full(self.Nfreqs, indata.channel_width)

        self.telescope_name = indata.telescope_name
        self.telescope_location = indata.telescope_location
        self.antenna_names = copy.deepcopy(indata.antenna_names)
        self.antenna_numbers = copy.deepcopy(indata.antenna_numbers)
        self.antenna_positions = copy.deepcopy(indata.antenna_positions)
        self.Nspws = indata.Nspws
        self.spw_array = copy.deepcopy(indata.spw_array)
        if indata.flex_spw_id_array is not None:
            self.flex_spw_id_array = copy.deepcopy(indata.flex_spw_id_array)

        if waterfall:
            self._set_type_waterfall()
            self.history += 'Flag object with type "waterfall" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.time_array, ri = np.unique(indata.time_array, return_index=True)
            if indata.future_array_shapes:
                self.freq_array = copy.deepcopy(indata.freq_array)
            else:
                self.freq_array = indata.freq_array[0, :]
            self.lst_array = indata.lst_array[ri]
            if copy_flags:
                raise NotImplementedError(
                    "Cannot copy flags when initializing waterfall UVFlag from "
                    "UVData or UVCal."
                )
            else:
                if self.mode == "flag":
                    self.flag_array = np.zeros(
                        (self.Ntimes, self.Nfreqs, self.Npols), np.bool_
                    )
                elif self.mode == "metric":
                    self.metric_array = np.zeros((self.Ntimes, self.Nfreqs, self.Npols))

        else:
            self._set_type_baseline()
            self.history += 'Flag object with type "baseline" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.baseline_array = copy.deepcopy(indata.baseline_array)
            self.Nbls = indata.Nbls
            self.Nblts = indata.Nblts
            self.ant_1_array = copy.deepcopy(indata.ant_1_array)
            self.ant_2_array = copy.deepcopy(indata.ant_2_array)
            self.Nants_data = indata.Nants_data

            self.time_array = copy.deepcopy(indata.time_array)
            self.lst_array = copy.deepcopy(indata.lst_array)

            if self.future_array_shapes == indata.future_array_shapes:
                # match on future shape
                self.freq_array = copy.deepcopy(indata.freq_array)
            elif indata.future_array_shapes:
                # input is future shaped, self is not
                self.freq_array = indata.freq_array[np.newaxis, :]
            else:
                # input is not future shaped, self is
                self.freq_array = indata.freq_array[0, :]

            if copy_flags:
                if self.future_array_shapes == indata.future_array_shapes:
                    self.flag_array = copy.deepcopy(indata.flag_array)
                elif indata.future_array_shapes:
                    self.flag_array = indata.flag_array[:, np.newaxis, :, :]
                else:
                    self.flag_array = indata.flag_array[:, 0, :, :]
                self.history += (
                    " Flags copied from " + str(indata.__class__) + " object."
                )
                if self.mode == "metric":
                    warnings.warn(
                        'Copying flags to type=="baseline" results in mode=="flag".'
                    )
                    self._set_mode_flag()
            else:
                if self.future_array_shapes:
                    array_shape = (self.Nblts, self.Nfreqs, self.Npols)
                else:
                    array_shape = (self.Nblts, 1, self.Nfreqs, self.Npols)
                if self.mode == "flag":
                    self.flag_array = np.zeros(array_shape, dtype=np.bool_)
                elif self.mode == "metric":
                    self.metric_array = np.zeros(array_shape, dtype=np.float64)

        self.filename = indata.filename
        self._filename.form = indata._filename.form

        if indata.x_orientation is not None:
            self.x_orientation = indata.x_orientation

        if self.mode == "metric":
            self.weights_array = np.ones(self.metric_array.shape)

        if indata.extra_keywords is not None:
            self.extra_keywords = copy.deepcopy(indata.extra_keywords)

        if history not in self.history:
            self.history += history
        self.label += label

        self.clear_unused_attributes()

        if not use_future_array_shapes:
            warnings.warn(_future_array_shapes_warning, DeprecationWarning)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        return

    def from_uvcal(
        self,
        indata,
        mode="metric",
        copy_flags=False,
        waterfall=False,
        history="",
        label="",
        use_future_array_shapes=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """Construct a UVFlag object from a UVCal object.

        Parameters
        ----------
        indata : UVData
            Input to initialize UVFlag object.
        mode : {"metric", "flag"}, optional
            The mode determines whether the object has a floating point metric_array
            or a boolean flag_array.
        copy_flags : bool, optional
            Whether to copy flags from indata to new UVFlag object
        waterfall : bool, optional
            Whether to immediately initialize as a waterfall object, with flag/metric
            axes: time, frequency, polarization.
        history : str, optional
            History string to attach to object.
        label: str, optional
            String used for labeling the object (e.g. 'FM').
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after creating UVFlag object.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            creating UVFlag object.

        """
        if not issubclass(indata.__class__, UVCal):
            raise ValueError(
                "from_uvcal can only initialize a UVFlag object from an input "
                "UVCal object or a subclass of a UVCal object."
            )

        if indata.wide_band:
            raise ValueError(
                "from_uvcal can only initialize a UVFlag object from a non-wide-band "
                "UVCal object."
            )

        if mode.lower() == "metric":
            self._set_mode_metric()
        elif mode.lower() == "flag":
            self._set_mode_flag()
        else:
            raise ValueError(
                "Input mode must be within acceptable values: "
                "{}".format((", ").join(self._mode.acceptable_vals))
            )

        if use_future_array_shapes:
            self._set_future_array_shapes()

        self.Nfreqs = indata.Nfreqs
        self.polarization_array = copy.deepcopy(indata.jones_array)
        self.Npols = indata.Njones
        self.Nants_telescope = indata.Nants_telescope
        self.Ntimes = indata.Ntimes
        self.time_array = copy.deepcopy(indata.time_array)
        self.lst_array = copy.deepcopy(indata.lst_array)

        if indata.future_array_shapes or indata.flex_spw:
            self.channel_width = copy.deepcopy(indata.channel_width)
        else:
            self.channel_width = np.full(self.Nfreqs, indata.channel_width)

        self.telescope_name = indata.telescope_name
        self.telescope_location = indata.telescope_location
        self.antenna_names = copy.deepcopy(indata.antenna_names)
        self.antenna_numbers = copy.deepcopy(indata.antenna_numbers)
        self.antenna_positions = copy.deepcopy(indata.antenna_positions)
        self.Nspws = indata.Nspws
        self.spw_array = copy.deepcopy(indata.spw_array)
        if indata.flex_spw_id_array is not None:
            self.flex_spw_id_array = copy.deepcopy(indata.flex_spw_id_array)

        if waterfall:
            self._set_type_waterfall()
            self.history += 'Flag object with type "waterfall" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            if indata.future_array_shapes:
                self.freq_array = copy.deepcopy(indata.freq_array)
            else:
                self.freq_array = indata.freq_array[0, :]
            if copy_flags:
                raise NotImplementedError(
                    "Cannot copy flags when "
                    "initializing waterfall UVFlag "
                    "from UVData or UVCal."
                )
            else:
                if self.mode == "flag":
                    self.flag_array = np.zeros(
                        (self.Ntimes, self.Nfreqs, self.Npols), np.bool_
                    )
                elif self.mode == "metric":
                    self.metric_array = np.zeros((self.Ntimes, self.Nfreqs, self.Npols))

        else:
            self._set_type_antenna()
            self.history += 'Flag object with type "antenna" created. '
            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str
            self.ant_array = copy.deepcopy(indata.ant_array)
            self.Nants_data = len(self.ant_array)

            if self.future_array_shapes == indata.future_array_shapes:
                # match on future shape
                self.freq_array = copy.deepcopy(indata.freq_array)
            elif indata.future_array_shapes:
                # input is future shaped, self is not
                self.freq_array = indata.freq_array[np.newaxis, :]
            else:
                # input is not future shaped, self is
                self.freq_array = indata.freq_array[0, :]

            if copy_flags:
                if self.future_array_shapes == indata.future_array_shapes:
                    self.flag_array = copy.deepcopy(indata.flag_array)
                elif indata.future_array_shapes:
                    self.flag_array = indata.flag_array[:, np.newaxis, :, :]
                else:
                    self.flag_array = indata.flag_array[:, 0, :, :]
                self.history += (
                    " Flags copied from " + str(indata.__class__) + " object."
                )
                if self.mode == "metric":
                    warnings.warn(
                        'Copying flags to type=="antenna" results in mode=="flag".'
                    )
                    self._set_mode_flag()
            else:
                if self.future_array_shapes:
                    array_shape = (
                        self.Nants_data,
                        self.Nfreqs,
                        self.Ntimes,
                        self.Npols,
                    )
                else:
                    array_shape = (
                        self.Nants_data,
                        1,
                        self.Nfreqs,
                        self.Ntimes,
                        self.Npols,
                    )
                if self.mode == "flag":
                    self.flag_array = np.zeros(array_shape, dtype=np.bool_)
                elif self.mode == "metric":
                    self.metric_array = np.zeros(array_shape, dtype=np.float64)

        if self.mode == "metric":
            self.weights_array = np.ones(self.metric_array.shape)

        self.filename = indata.filename
        self._filename.form = indata._filename.form

        if indata.x_orientation is not None:
            self.x_orientation = indata.x_orientation

        if history not in self.history:
            self.history += history
        self.label += label

        self.clear_unused_attributes()

        if not use_future_array_shapes:
            warnings.warn(_future_array_shapes_warning, DeprecationWarning)

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
        return
