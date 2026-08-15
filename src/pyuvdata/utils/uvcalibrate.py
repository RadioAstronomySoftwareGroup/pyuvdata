# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Code to apply calibration solutions to visibility data."""

import warnings
from typing import Literal

import numpy as np

from .pol import POL_TO_FEED_DICT, jnum2str, parse_jpolstr, polnum2str, polstr2num


def _get_pol_conventions(
    uvdata,
    uvcal,
    undo: bool,
    uvc_pol_convention: Literal["sum", "avg"] | None,
    uvd_pol_convention: Literal["sum", "avg"] | None,
):
    if uvc_pol_convention is None and uvcal.pol_convention is None:
        warnings.warn(
            message=(
                "pol_convention is not specified on the UVCal object, and "
                "uvc_pol_convention was not specified. Tentatively assuming "
                "that the UVCal and UVData objects (implicitly) have the same "
                "convention."
            ),
            stacklevel=2,
        )
        uvc_pol_convention = uvd_pol_convention or uvdata.pol_convention
    elif uvc_pol_convention is None:
        uvc_pol_convention = uvcal.pol_convention
    elif (
        uvcal.pol_convention is not None and uvc_pol_convention != uvcal.pol_convention
    ):
        raise ValueError(
            "uvc_pol_convention is set, and different than uvcal.pol_convention. "
            f"Got {uvc_pol_convention} and {uvcal.pol_convention}."
        )

    if undo:
        if uvd_pol_convention is None and uvdata.pol_convention is None:
            warnings.warn(
                message=(
                    "pol_convention is not specified on the UVData object, and "
                    "uvd_pol_convention was not specified. Tentatively assuming "
                    "that the UVCal and UVData objects (implicitly) have the same "
                    "convention."
                ),
                stacklevel=2,
            )
            uvd_pol_convention = uvc_pol_convention
        elif uvd_pol_convention is None:
            uvd_pol_convention = uvdata.pol_convention
        elif (
            uvdata.pol_convention is not None
            and uvd_pol_convention != uvdata.pol_convention
        ):
            raise ValueError(
                "Both uvd_pol_convention and uvdata.pol_convention were specified with "
                f"different values: {uvd_pol_convention} and {uvdata.pol_convention}."
            )
    else:
        if uvdata.pol_convention is not None and (
            uvc_pol_convention != uvdata.pol_convention
        ):
            raise ValueError(
                "UVData already has a pol_convention applied that does not match UVCal "
                f"convention: {uvdata.pol_convention} and {uvc_pol_convention}."
            )
        if uvd_pol_convention is None:
            uvd_pol_convention = uvc_pol_convention

        if uvd_pol_convention is None:
            # Both uvc and uvd have no pol_convention specified
            warnings.warn(
                message=(
                    "Neither uvd_pol_convention nor uvc_pol_convention are specified, "
                    "so the resulting UVData object will have ambiguous convention. "
                ),
                stacklevel=2,
            )
    if uvd_pol_convention not in ["sum", "avg", None]:
        raise ValueError(
            f"uvd_pol_convention must be 'sum' or 'avg'. Got {uvd_pol_convention}"
        )
    if uvc_pol_convention not in ["sum", "avg", None]:
        raise ValueError(
            f"uvc_pol_convention must be 'sum' or 'avg'. Got {uvc_pol_convention}"
        )

    return uvc_pol_convention, uvd_pol_convention


def _apply_pol_convention_corrections(
    uvdata,
    undo: bool,
    uvc_pol_convention: Literal["sum", "avg"] | None,
    uvd_pol_convention: Literal["sum", "avg"] | None,
):
    r"""
    Apply corrections to calibration/de-calibration from differences in convention.

    This function corrects the UVData ``data_array`` in-place, when the polarization
    convention desired for the UVData is different from the convention that was used
    for the calibration. It also sets the corresponding ``pol_convention`` attribute
    on the UVData object.

    The logic is as follows. If the convention of the calibration and UVData object
    are the same, no correction is applied. If they are different, the correction
    applied is either to multiply or divide by two. Let's start with a default case:
    let's say that the calibration solutions assume that instrumental polarizations
    are related to the stokes-I sky by the ``avg`` convention, in which case
    :math:`XX \sim I`, and we are calibrating data where we want the result to have
    the ``sum`` convention, i.e. :math:`XX \sim I/2`. Then, for data that is in
    instrumental polarizations (i.e. XX) we would need to *divide* the result by 2.
    This is flipped (i.e. flips between multiply and divide) for every difference from
    the above scenario, i.e.

    * If we are de-calibrated rather than calibrating
    * If the UVData is in stokes polarizations rather than instrumental (note that this
      is not currently possible anyway, so we do not provide the ability here).
    * If the conventions are swapped between the calibration solutions and the UVData.

    To be clear, if two of these are true, the resulting correction will be "flipped
    twice" (i.e. remain as *divide* by two), but if only one or all three are true,
    then the correction will be flipped to be multiply by two.
    """
    if uvd_pol_convention != uvc_pol_convention:
        correction = np.ones(uvdata.Npols) / 2

        if undo:
            # We are de-calibrating
            correction = 1 / correction

        if uvc_pol_convention == "sum":
            # pol convention difference is the other way around
            correction = 1 / correction

        uvdata.data_array *= correction

    uvdata.pol_convention = None if undo else uvd_pol_convention


def _select_uvd_records(uvdata, uvd_select_kwargs):
    """
    Work out which records a calibration should be applied to.

    Note that this is an internal helper function, not meant to be called by users.
    This function grabs the indicies that match a given set of `select` keywords, for
    use during `uvcalibrate`. Under the hood, it uses `UVData._select_preprocess`, to
    identify these index positions, allowing for a subset of visibilities to be modified
    in place (rather than selecting out a subset, applying calibration, and then
    merging the modified visibilities back in).

    Note this chooses what to calibrate; nothing is removed from `uvdata` either way.

    Parameters
    ----------
    uvdata : UVData object
        Object being calibrated, used to resolve the selection and size the masks.
    uvd_select_kwargs : dict or None
        Keywords describing the selection, as accepted by `UVData.select`.

    Returns
    -------
    blt_ind : ndarray of int or None
        Indices of the baseline-time records being calibrated, or None for all of them.
    freq_ind : ndarray of int or None
        Indices of the channels being calibrated, or None for all of them.
    pol_ind : ndarray of int or None
        Indices of the polarizations being calibrated, or None for all of them.

    Raises
    ------
    ValueError
        If a keyword is not one `UVData.select` accepts, or if the selection leaves
        nothing to calibrate.
    """
    if not uvd_select_kwargs:
        return None, None, None

    # `_select_preprocess` takes every selection keyword positionally-by-name with no
    # defaults, so fill in the ones the caller did not ask about.
    allowed = {
        "antenna_nums": None,
        "antenna_names": None,
        "ant_str": None,
        "bls": None,
        "frequencies": None,
        "freq_chans": None,
        "spws": None,
        "times": None,
        "time_range": None,
        "lsts": None,
        "lst_range": None,
        "polarizations": None,
        "blt_inds": None,
        "phase_center_ids": None,
        "catalog_names": None,
    }
    passthrough = {"invert", "strict", "warn_spacing"}
    unknown = set(uvd_select_kwargs) - set(allowed) - passthrough
    if unknown:
        raise ValueError(
            f"Unrecognized keyword(s) in uvd_select_kwargs: {sorted(unknown)}. "
            f"Allowed: {sorted(set(allowed) | passthrough)}."
        )
    kwargs = dict(allowed)
    kwargs.update(uvd_select_kwargs)

    blt_ind, freq_ind, _, pol_ind, _ = uvdata._select_preprocess(**kwargs)

    # Make sure the returned indices are arrays of integers (if not None)
    blt_ind = blt_ind if blt_ind is None else np.asarray(blt_ind, dtype=np.int64)
    freq_ind = freq_ind if freq_ind is None else np.asarray(freq_ind, dtype=np.int64)
    pol_ind = pol_ind if pol_ind is None else np.asarray(pol_ind, dtype=np.int64)

    return blt_ind, freq_ind, pol_ind


def uvcalibrate(
    uvdata,
    uvcal,
    *,
    inplace: bool = True,
    prop_flags: bool = True,
    d_term_cal: bool = False,
    flip_gain_conj: bool = False,
    delay_convention: Literal["minus", "plus"] = "minus",
    undo: bool = False,
    time_check: bool = True,
    ant_check: bool = True,
    freq_range_check: bool = True,
    uvc_pol_convention: Literal["sum", "avg"] | None = None,
    uvd_pol_convention: Literal["sum", "avg"] | None = None,
    apply_to_weights: bool = False,
    uvd_select_kwargs=None,
    uvc_select_kwargs=None,
    flag_unselected: bool | None = None,
    interpolate: bool = False,
    interp_kwargs=None,
):
    """
    Calibrate a UVData object with a UVCal object.

    Parameters
    ----------
    uvdata : UVData object
        UVData object to calibrate.
    uvcal : UVCal object
        UVCal object containing the calibration.
    inplace : bool, optional
        if True edit uvdata in place, else return a calibrated copy
    prop_flags : bool, optional
        if True, propagate calibration flags to data flags
        and doesn't use flagged gains. Otherwise, uses flagged gains and
        does not propagate calibration flags to data flags.
    Dterm_cal : bool, optional
        Calibrate the off-diagonal terms in the Jones matrix if present
        in uvcal. Default is False. Currently not implemented.
    flip_gain_conj : bool, optional
        This function uses the UVData ant_1_array and ant_2_array to specify the
        antennas in the UVCal object. By default, the conjugation convention, which
        follows the UVData convention (i.e. ant2 - ant1), is that the applied
        gain = ant1_gain * conjugate(ant2_gain). If the other convention is required,
        set flip_gain_conj=True.
    delay_convention : str, optional
        Exponent sign to use in conversion of 'delay' to 'gain' cal_type
        if the input uvcal is not inherently 'gain' cal_type. Default to 'minus'.
    undo : bool, optional
        If True, undo the provided calibration. i.e. apply the calibration with
        flipped gain_convention. Flag propagation rules apply the same.
    time_check : bool
        Option to check that times match between the UVCal and UVData
        objects if UVCal has a single time or time range. Times are always
        checked if UVCal has multiple times.
    ant_check : bool
        Option to check that all antennas with data on the UVData
        object have calibration solutions in the UVCal object. If this option is
        set to False, uvcalibrate will proceed without erroring and data for
        antennas without calibrations will be flagged.
    freq_range_check : bool
        Option to check that frequency ranges on the UVCal object matches the channel
        frequencies given in the UVData object for a given spectral window. Only
        applicable for wide-band UVCal objects, default is True.
    uvc_pol_convention : str, {"sum", "avg"}, optional
        The convention for how instrumental polarizations (e.g. XX and YY) are assumed
        to have been converted to Stokes parameters in ``uvcal``. Options are 'sum' and
        'avg', corresponding to I=XX+YY and I=(XX+YY)/2 (for linear instrumental
        polarizations) respectively. Only required if ``pol_convention`` is not set on
        ``uvcal`` itself. If it is not specified and is not set on the UVCal
        object, a deprecation warning is raised (will be an error in the future).
    uvd_pol_convention : str, {"sum", "avg"}, optional
        The same polarization convention as ``uvc_pol_convention``, except that this
        represents either the convention that *has* been adopted in ``uvdata`` (in the
        case that ``undo=True``), or the convention that is *desired* for the resulting
        ``UVData`` object (if ``undo=False``).
    apply_to_weights : bool
        Option to apply gains corrections to `UVData.nsample_array`, effectively
        reweighting the data. Only applicable for gains calibration, `nsample_array`
        array will be multiplied by the inverse-square of the absolute value of the
        gains correction applied to the data (similar to the behavior of CASA's
        applycal function with `calwt=True`). Default is False.
    uvd_select_kwargs : dict, optional
        Keywords describing which data to calibrate, using the same vocabulary
        `UVData.select` accepts -- e.g. `{"catalog_names": ["3c279"]}` to calibrate one
        source, or `{"antenna_nums": [1, 2]}`, `{"time_range": [...]}`,
        `{"blt_inds": [...]}`. Default is None, which calibrates everything. See the
        `UVData.select` docstring for further details. Note that if supplied, the
        calibration is applied to **only** the selected records, so the returned object
        may hold a mix of calibrated and uncalibrated data, with nothing marking which
        is which. The uncalibrated records are left unflagged unless `flag_unselected`
        is set to True.
    uvc_select_kwargs : dict, optional
        Keywords passed to `UVCal.select` to choose which solutions to calibrate
        with, e.g. `{"catalog_names": ["3c279"]}` to use the solutions derived from
        one source. Default is None, which uses all of the solutions. See the
        `UVCal.select` docstring for further details. Note that unlike
        `uvd_select_kwargs`, this narrows the solutions themselves.
    flag_unselected : bool or None
        If True, flag every record that `uvd_select_kwargs` left out, so that the
        uncalibrated data cannot be mistaken for calibrated data. Note that a record is
        flagged if it falls outside the selection on any axis, so only the records that
        were actually calibrated are left unflagged. If False, those records are left
        alone. Default is None, which behaves as False but throws a warning if only
        a subset of the data is being calibrated.
    interpolate : bool
        If True, interpolate the calibration onto the times of the data being
        calibrated (via `UVCal.interpolate_in_time`) before applying it, rather than
        requiring the two to already agree. Default is False.
    interp_kwargs : dict, optional
        Keywords passed through to `UVCal.interpolate_in_time` when
        `interpolate=True`, e.g. `{"kind": "linear"}`. The times to interpolate onto
        are supplied by this method, and `inplace` may not be set. Default is None,
        which uses the defaults of `interpolate_in_time` (see that methods docstring
        for further details).

    Returns
    -------
    UVData, optional
        Returns if not inplace

    """
    if uvcal.cal_type == "delay" and uvcal.Nspws > 1:
        # To fix this, need to make UVCal.convert_to_gain support multiple spws
        raise ValueError(
            "uvcalibrate currently does not support multi spectral window delay "
            "calibrations"
        )

    if np.any(uvdata.polarization_array > 0):
        raise NotImplementedError(
            "It is currently not possible to calibrate or de-calibrate data with "
            "stokes polarizations, since it is impossible to define UVCal objects with "
            "these polarizations. If you require this functionality, please submit an "
            "issue at "
            "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues/new"
        )

    if uvcal.gain_scale is None:
        warnings.warn(
            "gain_scale is not set, so there is no way to know what the resulting units"
            " are. For now, we assume that `gain_scale` matches whatever is on the "
            "UVData object (i.e. we do not change its units). Furthermore, all "
            "corrections concerning the pol_convention will be ignored.",
            category=UserWarning,
            stacklevel=2,
        )
    elif undo and uvcal.gain_scale != uvdata.vis_units:
        raise ValueError(
            "Cannot undo calibration if gain_scale is not the same as the units on "
            "the UVData object."
        )

    uvc_pol_convention, uvd_pol_convention = _get_pol_conventions(
        uvdata, uvcal, undo, uvc_pol_convention, uvd_pol_convention
    )
    if not inplace:
        uvdata = uvdata.copy()

    # check both objects
    uvdata.check()
    uvcal.check()

    # Narrow the solutions before anything else looks at them, so that the time
    # checks and any interpolation below see only the ones being applied.
    if uvc_select_kwargs is not None:
        if "inplace" in uvc_select_kwargs:
            raise ValueError(
                "Cannot set inplace via uvc_select_kwargs, since uvcalibrate needs "
                "to leave the UVCal object it was handed alone."
            )
        uvcal = uvcal.select(**uvc_select_kwargs, inplace=False)

    # Check whether the UVData antennas *that have data associated with them*
    # have associated data in the UVCal object
    uvdata_unique_nums = np.unique(np.append(uvdata.ant_1_array, uvdata.ant_2_array))
    uvdata.telescope.antenna_names = np.asarray(uvdata.telescope.antenna_names)
    uvdata_used_antnames = np.array(
        [
            uvdata.telescope.antenna_names[
                np.where(uvdata.telescope.antenna_numbers == antnum)
            ][0]
            for antnum in uvdata_unique_nums
        ]
    )
    uvcal_unique_nums = np.unique(uvcal.ant_array)
    uvcal.telescope.antenna_names = np.asarray(uvcal.telescope.antenna_names)
    uvcal_used_antnames = np.array(
        [
            uvcal.telescope.antenna_names[
                np.where(uvcal.telescope.antenna_numbers == antnum)
            ][0]
            for antnum in uvcal_unique_nums
        ]
    )

    ant_arr_match = uvcal_used_antnames.tolist() == uvdata_used_antnames.tolist()

    if not ant_arr_match:
        # check more carefully
        name_missing = []
        for this_ant_name in uvdata_used_antnames:
            wh_ant_match = np.nonzero(uvcal_used_antnames == this_ant_name)
            if wh_ant_match[0].size == 0:
                name_missing.append(this_ant_name)

        if len(name_missing) > 0:
            if len(name_missing) == uvdata_used_antnames.size:
                # all antenna_names with data on UVData are missing on UVCal.
                if not ant_check:
                    warnings.warn(
                        "All antenna names with data on UVData are missing "
                        "on UVCal. Since ant_check is False, calibration will "
                        "proceed but all data will be flagged."
                    )
                else:
                    raise ValueError(
                        "All antenna names with data on UVData are missing "
                        "on UVCal. To continue with calibration "
                        "(and flag all the data), set ant_check=False."
                    )
            else:
                # Only some antenna_names with data on UVData are missing on UVCal
                if not ant_check:
                    warnings.warn(
                        f"Antennas {name_missing} have data on UVData but are missing "
                        "on UVCal. Since ant_check is False, calibration will "
                        "proceed and the data for these antennas will be flagged."
                    )
                else:
                    raise ValueError(
                        f"Antennas {name_missing} have data on UVData but "
                        "are missing on UVCal. To continue calibration and "
                        "flag the data from missing antennas, set ant_check=False."
                    )

    # Work out which baseline-time records are being calibrated, and find those entries
    blt_ind_arr, freq_ind_arr, pol_ind_arr = _select_uvd_records(
        uvdata, uvd_select_kwargs
    )

    # A plain Ellipsis when nothing was selected, so the untouched path stays a view.
    blt_mask = freq_mask = pol_mask = ...
    if blt_ind_arr is not None:
        blt_mask = np.zeros(uvdata.Nblts, dtype=bool)
        blt_mask[blt_ind_arr] = True

    if freq_ind_arr is not None:
        freq_mask = np.zeros(uvdata.Nfreqs, dtype=bool)
        freq_mask[freq_ind_arr] = True

    if pol_ind_arr is not None:
        pol_mask = np.zeros(uvdata.Npols, dtype=bool)
        pol_mask[pol_ind_arr] = True

    if uvd_select_kwargs is not None and flag_unselected is None:
        warnings.warn(
            "Calibrating a subset of the data leaves the object holding a mix of "
            "calibrated and uncalibrated records, which nothing on the object marks "
            "apart -- only the history records that it happened. Set "
            "flag_unselected=True to flag the records that are left uncalibrated, "
            "or flag_unselected=False to leave them alone and silence this warning."
        )

    if interpolate:
        interp_kwargs = {} if interp_kwargs is None else dict(interp_kwargs)
        if "inplace" in interp_kwargs:
            raise ValueError(
                "Cannot set inplace via interp_kwargs, since uvcalibrate needs to "
                "leave the UVCal object it was handed alone."
            )
        # Interpolate onto every time in the data. Note that we do this also for the
        # times left out of the selection to keep the time-axis "locking", since that
        # is used by the checks below. The extra solutions simply go unused, and the
        # overhead here is low enough to justify the extra arithmetic.
        preinterp_history_len = len(uvcal.history)
        uvcal = uvcal.interpolate_in_time(
            np.unique(uvdata.time_array), inplace=False, **interp_kwargs
        )
        # Grab just what the interpolation recorded about itself.
        interp_history = uvcal.history[preinterp_history_len:]

    uvdata_times, uvd_time_ri = np.unique(uvdata.time_array, return_inverse=True)
    uvcal_times_to_keep = None
    # time_range supersedes time_array.
    if uvcal.time_range is not None:
        if np.min(uvdata_times) < np.min(uvcal.time_range[:, 0]) or np.max(
            uvdata_times
        ) > np.max(uvcal.time_range[:, 1]):
            if not time_check and uvcal.Ntimes == 1:
                warnings.warn(
                    "Time_range on UVCal does not cover all UVData times "
                    "but time_check is False, so calibration "
                    "will be applied anyway."
                )
            else:
                msg = "Time_ranges on UVCal do not cover all UVData times."
                if uvcal.Ntimes == 1:
                    msg = (
                        "Time_range on UVCal does not cover all UVData times. "
                        "Set time_check=False to apply calibration anyway."
                    )
                else:
                    msg = "Time_ranges on UVCal do not cover all UVData times."
                raise ValueError(msg)

        # now check in detail that all UVData times fall in a UVCal time range.
        # also create the indexing array to match UVData blts to UVCal time inds
        if uvcal.Ntimes > 1:
            trange_ind_arr = np.full_like(uvdata.time_array, -1, dtype=int)
            for tr_ind, trange in enumerate(uvcal.time_range):
                time_inds = np.nonzero(
                    (uvdata_times >= trange[0]) & (uvdata_times <= trange[1])
                )[0]
                for tind in time_inds:
                    trange_ind_arr[np.nonzero(uvd_time_ri == tind)[0]] = tr_ind
            if np.any(trange_ind_arr < 0):
                raise ValueError("Time_ranges on UVCal do not cover all UVData times.")
    else:
        if uvcal.Ntimes > 1 and uvcal.Ntimes < uvdata.Ntimes:
            raise ValueError(
                "The uvcal object has more than one time but fewer than the "
                "number of unique times on the uvdata object."
            )
        uvcal_times = np.unique(uvcal.time_array)
        try:
            time_arr_match = np.allclose(
                uvcal_times,
                uvdata_times,
                atol=uvdata._time_array.tols[1],
                rtol=uvdata._time_array.tols[0],
            )
        except ValueError:
            time_arr_match = False

        if not time_arr_match:
            if uvcal.Ntimes == 1:
                if not time_check:
                    warnings.warn(
                        "Times do not match between UVData and UVCal "
                        "but time_check is False, so calibration "
                        "will be applied anyway."
                    )
                else:
                    raise ValueError(
                        "Times do not match between UVData and UVCal. "
                        "Set time_check=False to apply calibration anyway. "
                    )
            else:
                # check more carefully
                uvcal_times_to_keep = []
                for this_time in uvdata_times:
                    wh_time_match = np.nonzero(
                        np.isclose(
                            uvcal.time_array - this_time,
                            0,
                            atol=uvdata._time_array.tols[1],
                            rtol=uvdata._time_array.tols[0],
                        )
                    )
                    if wh_time_match[0].size > 0:
                        uvcal_times_to_keep.append(uvcal.time_array[wh_time_match][0])
                    else:
                        raise ValueError(
                            f"Time {this_time} exists on UVData but not on UVCal."
                        )

    uvcal_chans_to_keep = None if uvcal.wide_band else []
    uvcal_spws_to_keep = [] if uvcal.wide_band else None
    for spw in uvdata.spw_array:
        try:
            uvcal_spw_idx = np.where(uvcal.spw_array == spw)[0][0]
            if uvcal.wide_band:
                if freq_range_check:
                    freq_array_spw = uvdata.freq_array[uvdata.flex_spw_id_array == spw]
                    min_freq = min(uvcal.freq_range[uvcal_spw_idx])
                    max_freq = max(uvcal.freq_range[uvcal_spw_idx])
                    if any((freq_array_spw < min_freq) | (freq_array_spw > max_freq)):
                        raise ValueError(
                            f"SPW {spw} exists on UVData and UVCal, but the channel "
                            "frequencies are inconsistent with frequency ranges. "
                            "To continue with calibration, set freq_range_check=False."
                        )
                uvcal_spws_to_keep.append(spw)
            else:
                uvcal_chans = np.where(uvcal.flex_spw_id_array == spw)[0]
                uvdata_chans = np.where(uvdata.flex_spw_id_array == spw)[0]
                uvcal_freqs = uvcal.freq_array[uvcal_chans]
                uvdata_freqs = uvdata.freq_array[uvdata_chans]
                for indv_freq in uvdata_freqs:
                    freq_match = np.isclose(
                        uvcal_freqs,
                        indv_freq,
                        atol=uvdata._freq_array.tols[1],
                        rtol=uvdata._freq_array.tols[0],
                    )
                    if any(freq_match):
                        uvcal_chans_to_keep.append(uvcal_chans[freq_match][0])
                    else:
                        raise ValueError(
                            f"Frequency {indv_freq} exists on UVData but not on UVCal."
                        )
        except IndexError as err:
            raise ValueError(f"SPW {spw} exists on UVData but not on UVCal.") from err

    if np.array_equal(uvcal_chans_to_keep, np.arange(uvcal.Nfreqs)):
        uvcal_chans_to_keep = None
    if np.array_equal(uvcal_spws_to_keep, uvcal.spw_array):
        uvcal_spws_to_keep = None

    # check if x_orientation-equivalent in uvdata isn't set (it's required for uvcal)
    uvd_x = uvdata.telescope.get_x_orientation_from_feeds()
    if uvd_x is None and uvdata.telescope.feed_array is None:
        # use the uvcal x_orientation throughout
        uvd_x = uvcal.telescope.get_x_orientation_from_feeds()
        warnings.warn(
            "UVData object does not have `x_orientation` specified but UVCal does. "
            "Matching based on `x` and `y` only "
        )

    # Only the polarizations actually being calibrated need solutions to match against.
    uvdata_pol_strs = polnum2str(
        uvdata.polarization_array[pol_mask], x_orientation=uvd_x
    )
    uvcal_pol_strs = jnum2str(
        uvcal.jones_array, x_orientation=uvcal.telescope.get_x_orientation_from_feeds()
    )
    uvdata_feed_pols = {
        feed for pol in uvdata_pol_strs for feed in POL_TO_FEED_DICT[pol]
    }
    for feed in uvdata_feed_pols:
        # get diagonal jones str
        jones_str = parse_jpolstr(
            feed, x_orientation=uvcal.telescope.get_x_orientation_from_feeds()
        )
        if jones_str not in uvcal_pol_strs:
            raise ValueError(
                f"Feed polarization {feed} exists on UVData but not on UVCal. "
            )

    # downselect UVCal times, frequencies
    new_uvcal = not (
        uvcal_times_to_keep is None
        and uvcal_chans_to_keep is None
        and uvcal_spws_to_keep is None
    )
    uvcal_use = uvcal
    if new_uvcal:
        uvcal_use = uvcal.select(
            times=uvcal_times_to_keep,
            freq_chans=uvcal_chans_to_keep,
            spws=uvcal_spws_to_keep,
            inplace=False,
        )

    # input checks
    if uvcal_use.cal_type == "delay":
        if not new_uvcal:
            # make a copy to convert to gain
            uvcal_use = uvcal_use.copy()
            new_uvcal = True
        freq_array_use = uvdata.freq_array
        channel_width = uvdata.channel_width
        uvcal_use.convert_to_gain(
            delay_convention=delay_convention,
            freq_array=freq_array_use,
            channel_width=channel_width,
        )
        # Force this to false if we've got a delay soln, since it has no amplitude terms
        apply_to_weights = False

    # D-term calibration
    if d_term_cal:
        # check for D-terms
        if -7 not in uvcal_use.jones_array and -8 not in uvcal_use.jones_array:
            raise ValueError(
                "Cannot apply D-term calibration without -7 or -8"
                "Jones polarization in uvcal object."
            )
        raise NotImplementedError("D-term calibration is not yet implemented.")

    # No D-term calibration
    else:
        # key is number, value is name
        uvdata_ant_dict = dict(
            zip(
                uvdata.telescope.antenna_numbers,
                uvdata.telescope.antenna_names,
                strict=False,
            )
        )
        # opposite: key is name, value is number
        uvcal_ant_dict = dict(
            zip(
                uvcal.telescope.antenna_names,
                uvcal.telescope.antenna_numbers,
                strict=False,
            )
        )
        uvc_spw_map = {spw: idx for idx, spw in enumerate(uvcal_use.spw_array)}

        # iterate over keys
        for key in uvdata.get_antpairpols():
            # get indices for this key
            blt_subinds = uvdata.antpair2ind(key)
            keep_pos = None
            if (blt_ind_arr is not None or freq_ind_arr is not None) and isinstance(
                blt_subinds, slice
            ):
                # `antpair2ind` hands back a slice when the records happen to be
                # contiguous, but a slice won't work with the select operations,
                # so make the indices.
                blt_subinds = np.arange(uvdata.Nblts)[blt_subinds]
            if blt_ind_arr is not None:
                # If we need are just performing the apply on a subset of baseline pairs
                # then we want to grab those now.
                keep_pos = blt_mask[blt_subinds]
                blt_subinds = blt_subinds[keep_pos]
                if not blt_subinds.size:
                    # Nothing selected on this baseline, skip!
                    continue
            pol_ind = np.argmin(
                np.abs(uvdata.polarization_array - polstr2num(key[2], uvd_x))
            )
            if pol_ind_arr is not None and not pol_mask[pol_ind]:
                # This polarization was not selected, so skip it
                continue

            # try to get gains for each antenna
            ant1_num = key[0]
            ant2_num = key[1]

            feed1, feed2 = POL_TO_FEED_DICT[key[2]]
            try:
                uvcal_ant1_num = uvcal_ant_dict[uvdata_ant_dict[ant1_num]]
            except KeyError:
                uvcal_ant1_num = None
            try:
                uvcal_ant2_num = uvcal_ant_dict[uvdata_ant_dict[ant2_num]]
            except KeyError:
                uvcal_ant2_num = None

            if (uvcal_ant1_num is None or uvcal_ant2_num is None) or not (
                uvcal_use._key_exists(antnum=uvcal_ant1_num, jpol=feed1)
                and uvcal_use._key_exists(antnum=uvcal_ant2_num, jpol=feed2)
            ):
                uvdata.flag_array[blt_subinds, :, pol_ind] = True
                continue

            uvcal_key1 = (uvcal_ant1_num, feed1)
            uvcal_key2 = (uvcal_ant2_num, feed2)
            if flip_gain_conj:
                gain = (
                    np.conj(uvcal_use.get_gains(uvcal_key1))
                    * uvcal_use.get_gains(uvcal_key2)
                ).T  # transpose to match uvdata shape
            else:
                gain = (
                    uvcal_use.get_gains(uvcal_key1)
                    * np.conj(uvcal_use.get_gains(uvcal_key2))
                ).T  # tranpose to match uvdata shape
            flag = (uvcal_use.get_flags(uvcal_key1) | uvcal_use.get_flags(uvcal_key2)).T

            if uvcal.time_range is not None and uvcal.Ntimes > 1:
                gain = gain[trange_ind_arr[blt_subinds], :]
                flag = flag[trange_ind_arr[blt_subinds], :]
            elif keep_pos is not None and gain.shape[0] == keep_pos.size:
                # If we only need a subset of the interpolated gains, grab them now
                # for the apply step.
                gain = gain[keep_pos]
                flag = flag[keep_pos]

            # Use a slice operator to expand out the flags and gains with a wide_band
            # calibration solution, otherwise use the Ellipsis to select the whole
            # array when using a channel-based soln.
            gain_slice = ...
            if uvcal_use.wide_band:
                gain_slice = np.s_[
                    :, [uvc_spw_map[spw] for spw in uvdata.flex_spw_id_array[freq_mask]]
                ]
            elif freq_ind_arr is not None:
                gain_slice = np.s_[:, freq_ind_arr]

            # Do the same on the data side
            if freq_ind_arr is None:
                data_slice = np.s_[blt_subinds, :, pol_ind]
            else:
                data_slice = np.s_[
                    blt_subinds[:, np.newaxis], freq_ind_arr[np.newaxis, :], pol_ind
                ]

            # propagate flags
            if prop_flags:
                mask = np.isclose(gain, 0.0) | flag
                gain[mask] = 1.0
                uvdata.flag_array[data_slice] |= mask[gain_slice]

            # apply to data
            mult_gains = uvcal_use.gain_convention == "multiply"
            if undo:
                mult_gains = not mult_gains
            if mult_gains:
                uvdata.data_array[data_slice] *= gain[gain_slice]
                if apply_to_weights:
                    uvdata.nsample_array[data_slice] /= abs(gain[gain_slice]) ** 2
            else:
                uvdata.data_array[data_slice] /= gain[gain_slice]
                if apply_to_weights:
                    uvdata.nsample_array[data_slice] *= abs(gain[gain_slice]) ** 2

    # update attributes
    uvdata.history += "\nCalibrated with pyuvdata.utils.uvcalibrate."

    # Record which records were touched, so that it's clear from the history where the
    # calibration was applied and where it was not.
    if uvd_select_kwargs is not None:
        counts = [
            f"{len(ind_arr)} of {total} {name}"
            for name, ind_arr, total in [
                ("baseline-times", blt_ind_arr, uvdata.Nblts),
                ("frequencies", freq_ind_arr, uvdata.Nfreqs),
                ("polarizations", pol_ind_arr, uvdata.Npols),
            ]
            if ind_arr is not None
        ]
        uvdata.history += (
            " Applied to a subset of the data, selected on "
            f"{', '.join(sorted(uvd_select_kwargs))}"
            + (f" ({'; '.join(counts)})" if counts else "")
            + ". Records outside that selection were left uncalibrated, and have "
            + ("been flagged." if flag_unselected else "NOT been flagged.")
        )

    if uvc_select_kwargs is not None:
        uvdata.history += (
            " Calibration solutions were selected on "
            f"{', '.join(sorted(uvc_select_kwargs))} before being applied."
        )

    # Carry over how the solutions were interpolated (if they were)
    if interpolate:
        uvdata.history += " Solutions were interpolated before use:" + interp_history

    if undo:
        uvdata.vis_units = "uncalib"
    else:
        if uvcal_use.gain_scale is not None:
            uvdata.vis_units = uvcal_use.gain_scale

    # Set pol convention properly
    if uvcal.gain_scale is not None:
        _apply_pol_convention_corrections(
            uvdata, undo, uvc_pol_convention, uvd_pol_convention
        )

    if flag_unselected:
        if blt_ind_arr is not None:
            uvdata.flag_array[~blt_mask, :, :] = True
        if freq_ind_arr is not None:
            uvdata.flag_array[:, ~freq_mask, :] = True
        if pol_ind_arr is not None:
            uvdata.flag_array[:, :, ~pol_mask] = True

    if not inplace:
        return uvdata
