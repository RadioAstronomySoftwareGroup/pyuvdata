# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for antennas."""

import numpy as np

from . import tools

MOUNT_STR2NUM_DICT = {
    "alt-az": 0,
    "equatorial": 1,
    "orbiting": 2,
    "x-y": 3,
    "alt-az+nasmyth-r": 4,
    "alt-az+nasmyth-l": 5,
    "phased": 6,
    "fixed": 7,
    "other": 8,
    "bizarre": 8,  # Semi-common code in UVFITS/CASA for "unknown" types
}

MOUNT_NUM2STR_DICT = {
    0: "alt-az",
    1: "equatorial",
    2: "orbiting",
    3: "x-y",
    4: "alt-az+nasmyth-r",
    5: "alt-az+nasmyth-l",  # here and above, UVFITS-defined
    6: "phased",  # <- pyuvdata defined, but not uncommon in UVFITS
    7: "fixed",  # <- pyuvdata defined
    8: "other",  # <- pyuvdata defined
}


def _select_antenna_helper(
    *,
    antenna_names,
    antenna_nums,
    tel_ant_names,
    tel_ant_nums,
    obj_ant_array,
    invert=False,
    strict=False,
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
    strict : bool or None
        Normally, select will warn when an element of the selection criteria does not
        match any element for the parameter, as long as the selection criteria results
        in *at least one* element being selected. However, if set to True, an error is
        thrown if any selection criteria does not match what is given for the object
        parameters element. If set to None, then neither errors nor warnings are raised,
        unless no records are selected. Default is False.

    Returns
    -------
    ant_inds : list of int
        Indices of antennas to keep on the object.
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
        tel_ant_names = np.asarray(tel_ant_names).flatten()
        antenna_nums = []
        for s in antenna_names:
            if s in tel_ant_names:
                ind = np.where(tel_ant_names == s)[0][0]
                antenna_nums.append(tel_ant_nums[ind])
            else:
                err_msg = f"Antenna name {s} is not present in the antenna_names array"
                tools._strict_raise(err_msg, strict=strict)

    if antenna_nums is not None:
        antenna_nums = np.asarray(antenna_nums).flatten()
        selections.append("antennas")

        check = np.isin(antenna_nums, obj_ant_array, invert=True)
        if any(check):
            tools._strict_raise(
                f"Antenna number {antenna_nums[check]} is not present in the ant_array",
                strict=strict,
            )

        ant_inds = np.nonzero(np.isin(obj_ant_array, antenna_nums, invert=invert))[0]
        ant_inds = ant_inds.tolist()
        if len(ant_inds) == 0:
            raise ValueError("No data matching this antenna selection exists.")

    return ant_inds, selections
