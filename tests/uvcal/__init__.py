# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import numpy as np


def time_array_to_time_range(calobj_in, keep_time_array=False):
    calobj = calobj_in.copy()
    tstarts = calobj.time_array - calobj.integration_time / (86400 * 2)
    tends = calobj.time_array + calobj.integration_time / (86400 * 2)
    calobj.time_range = np.stack((tstarts, tends), axis=1)
    if not keep_time_array:
        calobj.time_array = None
        calobj.lst_array = None
    calobj.set_lsts_from_time_array()

    return calobj


def extend_jones_axis(calobj, total_quality=True):
    while calobj.Njones < 4:
        new_jones = np.min(calobj.jones_array) - 1
        calobj.jones_array = np.append(calobj.jones_array, new_jones)
        calobj.Njones += 1
        if not calobj.metadata_only:
            attrs_to_extend = [
                "gain_array",
                "delay_array",
                "flag_array",
                "quality_array",
                "total_quality_array",
            ]
            for attr in attrs_to_extend:
                attr_value = getattr(calobj, attr)
                if attr_value is not None:
                    attr_value = np.concatenate(
                        (attr_value, attr_value[..., [-1]]), axis=-1
                    )
                    setattr(calobj, attr, attr_value)

    if (
        not calobj.metadata_only
        and calobj.total_quality_array is None
        and total_quality
    ):
        calobj.total_quality_array = np.ones(
            calobj._total_quality_array.expected_shape(calobj)
        )
