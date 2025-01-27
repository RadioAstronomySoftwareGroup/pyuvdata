# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for antennas."""

import numpy as np

from . import tools


def _select_antenna_helper(
    *,
    antenna_names,
    antenna_nums,
    tel_ant_names,
    tel_ant_nums,
    obj_ant_array,
    invert=False,
):
    """
    Get antenna indices in a select.

    Parameters
    ----------
    antenna_names : array_like of str
        List of antennas to be selected based on name.
    antenna_nums : array_like of int
        List of antennas to be selected based on number.
    tel_ant_names : array_like of str
        List that contains the full set of antenna names for the telescope, which is
        matched to the list provided in `tel_ant_nums`. Used to map antenna name to
        number.
    tel_ant_nums : array_like of int
        List that contains the full set of antenna numbers for the telescope, which is
        matched to the list provided in `tel_ant_names`. Used to map antenna name to
        number.
    obj_ant_array : array_like of int
        The antenna numbers present in the object.
    invert : bool, optional
        Normally indices matching given criteria are what are included in the
        subsequent list. However, if set to True, these indices are excluded
        instead. Default is False.

    Returns
    -------
    pol_inds : list of int
        Indices of polarization to keep on the object.
    selections : list of str
        list of selections done.

    """
    ant_inds = None
    selections = []
    if antenna_names is not None:
        if antenna_nums is not None:
            raise ValueError(
                "Only one of antenna_nums and antenna_names can be provided."
            )

        antenna_names = tools._get_iterable(antenna_names)
        antenna_nums = []
        for s in antenna_names:
            if s not in tel_ant_names and not invert:
                raise ValueError(
                    f"Antenna name {s} is not present in the antenna_names array"
                )
            ind = np.where(np.array(tel_ant_names) == s)[0][0]
            antenna_nums.append(tel_ant_nums[ind])

    if antenna_nums is not None:
        antenna_nums = np.asarray(tools._get_iterable(antenna_nums)).flatten()
        selections.append("antennas")

        ant_check = np.isin(antenna_nums, obj_ant_array, invert=True)
        if not invert and any(ant_check):
            raise ValueError(
                f"Antenna number {antenna_nums[ant_check]} is not present "
                "in the ant_array"
            )

        ant_inds = np.nonzero(np.isin(obj_ant_array, antenna_nums, invert=invert))[0]
        ant_inds = sorted(set(ant_inds.tolist()))

    return ant_inds, selections
