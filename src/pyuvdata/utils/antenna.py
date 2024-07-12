# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for antennas."""

import numpy as np

from . import tools


def _select_antenna_helper(
    *, antenna_names, antenna_nums, tel_ant_names, tel_ant_nums, obj_ant_array
):
    if antenna_names is not None:
        if antenna_nums is not None:
            raise ValueError(
                "Only one of antenna_nums and antenna_names can be provided."
            )

        antenna_names = tools._get_iterable(antenna_names)
        antenna_nums = []
        for s in antenna_names:
            if s not in tel_ant_names:
                raise ValueError(
                    f"Antenna name {s} is not present in the antenna_names array"
                )
            ind = np.where(np.array(tel_ant_names) == s)[0][0]
            antenna_nums.append(tel_ant_nums[ind])

    if antenna_nums is not None:
        antenna_nums = tools._get_iterable(antenna_nums)
        antenna_nums = np.asarray(antenna_nums)
        selections = ["antennas"]

        ant_inds = np.zeros(0, dtype=np.int64)
        ant_check = np.isin(antenna_nums, obj_ant_array)
        if not np.all(ant_check):
            raise ValueError(
                f"Antenna number {antenna_nums[~ant_check]} is not present "
                "in the ant_array"
            )

        for ant in antenna_nums:
            ant_inds = np.append(ant_inds, np.where(obj_ant_array == ant)[0])

        ant_inds = sorted(set(ant_inds))
    else:
        ant_inds = None
        selections = None

    return ant_inds, selections
