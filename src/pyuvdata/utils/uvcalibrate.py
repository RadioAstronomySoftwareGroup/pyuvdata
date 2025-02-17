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
        if uvdata.pol_convention is not None:
            raise ValueError("You are trying to calibrate already-calibrated data.")
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
    uvc_pol_convention: Literal["sum", "avg"] | None = None,
    uvd_pol_convention: Literal["sum", "avg"] | None = None,
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

    Returns
    -------
    UVData, optional
        Returns if not inplace

    """
    if uvcal.cal_type == "gain" and uvcal.wide_band:
        raise ValueError(
            "uvcalibrate currently does not support wide-band calibrations"
        )
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

    uvdata_times, uvd_time_ri = np.unique(uvdata.time_array, return_inverse=True)
    downselect_cal_times = False
    # time_range supercedes time_array.
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
                if len(uvcal_times_to_keep) < uvcal.Ntimes:
                    downselect_cal_times = True

    downselect_cal_freq = False
    if uvcal.freq_array is not None:
        uvdata_freq_arr_use = uvdata.freq_array
        uvcal_freq_arr_use = uvcal.freq_array
        try:
            freq_arr_match = np.allclose(
                np.sort(uvcal_freq_arr_use),
                np.sort(uvdata_freq_arr_use),
                atol=uvdata._freq_array.tols[1],
                rtol=uvdata._freq_array.tols[0],
            )
        except ValueError:
            freq_arr_match = False

        if freq_arr_match is False:
            # check more carefully
            uvcal_freqs_to_keep = []
            for this_freq in uvdata_freq_arr_use:
                wh_freq_match = np.nonzero(
                    np.isclose(
                        uvcal.freq_array - this_freq,
                        0,
                        atol=uvdata._freq_array.tols[1],
                        rtol=uvdata._freq_array.tols[0],
                    )
                )
                if wh_freq_match[0].size > 0:
                    uvcal_freqs_to_keep.append(uvcal.freq_array[wh_freq_match][0])
                else:
                    raise ValueError(
                        f"Frequency {this_freq} exists on UVData but not on UVCal."
                    )
            if len(uvcal_freqs_to_keep) < uvcal.Nfreqs:
                downselect_cal_freq = True

    # check if x_orientation-equivalent in uvdata isn't set (it's required for uvcal)
    uvd_x = uvdata.telescope.get_x_orientation_from_feeds()
    if uvd_x is None and uvdata.telescope.feed_array is None:
        # use the uvcal x_orientation throughout
        uvd_x = uvcal.telescope.get_x_orientation_from_feeds()
        warnings.warn(
            "UVData object does not have `x_orientation` specified but UVCal does. "
            "Matching based on `x` and `y` only "
        )

    uvdata_pol_strs = polnum2str(uvdata.polarization_array, x_orientation=uvd_x)
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
    if downselect_cal_freq or downselect_cal_times:
        if not downselect_cal_times:
            uvcal_times_to_keep = None
        elif not downselect_cal_freq:
            uvcal_freqs_to_keep = None

        uvcal_use = uvcal.select(
            times=uvcal_times_to_keep, frequencies=uvcal_freqs_to_keep, inplace=False
        )

        new_uvcal = True
    else:
        uvcal_use = uvcal
        new_uvcal = False

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

        # iterate over keys
        for key in uvdata.get_antpairpols():
            # get indices for this key
            blt_inds = uvdata.antpair2ind(key)
            pol_ind = np.argmin(
                np.abs(uvdata.polarization_array - polstr2num(key[2], uvd_x))
            )

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
                uvdata.flag_array[blt_inds, :, pol_ind] = True
                continue

            uvcal_key1 = (uvcal_ant1_num, feed1)
            uvcal_key2 = (uvcal_ant2_num, feed2)
            if flip_gain_conj:
                gain = (
                    np.conj(uvcal_use.get_gains(uvcal_key1))
                    * uvcal_use.get_gains(uvcal_key2)
                ).T  # tranpose to match uvdata shape
            else:
                gain = (
                    uvcal_use.get_gains(uvcal_key1)
                    * np.conj(uvcal_use.get_gains(uvcal_key2))
                ).T  # tranpose to match uvdata shape
            flag = (uvcal_use.get_flags(uvcal_key1) | uvcal_use.get_flags(uvcal_key2)).T

            if uvcal.time_range is not None and uvcal.Ntimes > 1:
                gain = gain[trange_ind_arr[blt_inds], :]
                flag = flag[trange_ind_arr[blt_inds], :]

            # propagate flags
            if prop_flags:
                mask = np.isclose(gain, 0.0) | flag
                gain[mask] = 1.0
                uvdata.flag_array[blt_inds, :, pol_ind] += mask

            # apply to data
            mult_gains = uvcal_use.gain_convention == "multiply"
            if undo:
                mult_gains = not mult_gains
            if mult_gains:
                uvdata.data_array[blt_inds, :, pol_ind] *= gain
            else:
                uvdata.data_array[blt_inds, :, pol_ind] /= gain

    # update attributes
    uvdata.history += "\nCalibrated with pyuvdata.utils.uvcalibrate."
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

    if not inplace:
        return uvdata
