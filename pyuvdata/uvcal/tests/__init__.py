# -*- mode: python; coding: utf-8 -*-
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


def extend_jones_axis(calobj, input_flag=True, total_quality=True):
    while calobj.Njones < 4:
        new_jones = np.min(calobj.jones_array) - 1
        calobj.jones_array = np.append(calobj.jones_array, new_jones)
        calobj.Njones += 1
        if not calobj.metadata_only:
            if calobj.future_array_shapes:
                calobj.flag_array = np.concatenate(
                    (calobj.flag_array, calobj.flag_array[:, :, :, [-1]]), axis=3
                )
                if calobj.cal_type == "gain":
                    calobj.gain_array = np.concatenate(
                        (calobj.gain_array, calobj.gain_array[:, :, :, [-1]]), axis=3
                    )
                else:
                    calobj.delay_array = np.concatenate(
                        (calobj.delay_array, calobj.delay_array[:, :, :, [-1]]), axis=3
                    )
                if calobj.input_flag_array is not None:
                    calobj.input_flag_array = np.concatenate(
                        (
                            calobj.input_flag_array,
                            calobj.input_flag_array[:, :, :, [-1]],
                        ),
                        axis=3,
                    )
                calobj.quality_array = np.concatenate(
                    (calobj.quality_array, calobj.quality_array[:, :, :, [-1]]), axis=3
                )
                if calobj.total_quality_array is not None:
                    calobj.total_quality_array = np.concatenate(
                        (
                            calobj.total_quality_array,
                            calobj.total_quality_array[:, :, [-1]],
                        ),
                        axis=2,
                    )
            else:
                calobj.flag_array = np.concatenate(
                    (calobj.flag_array, calobj.flag_array[:, :, :, :, [-1]]), axis=4
                )
                if calobj.cal_type == "gain":
                    calobj.gain_array = np.concatenate(
                        (calobj.gain_array, calobj.gain_array[:, :, :, :, [-1]]), axis=4
                    )
                else:
                    calobj.delay_array = np.concatenate(
                        (calobj.delay_array, calobj.delay_array[:, :, :, :, [-1]]),
                        axis=4,
                    )
                if calobj.input_flag_array is not None:
                    calobj.input_flag_array = np.concatenate(
                        (
                            calobj.input_flag_array,
                            calobj.input_flag_array[:, :, :, :, [-1]],
                        ),
                        axis=4,
                    )
                calobj.quality_array = np.concatenate(
                    (calobj.quality_array, calobj.quality_array[:, :, :, :, [-1]]),
                    axis=4,
                )
                if calobj.total_quality_array is not None:
                    calobj.total_quality_array = np.concatenate(
                        (
                            calobj.total_quality_array,
                            calobj.total_quality_array[:, :, :, [-1]],
                        ),
                        axis=3,
                    )
    if not calobj.metadata_only:
        if calobj.input_flag_array is None and input_flag:
            calobj.input_flag_array = calobj.flag_array
        if calobj.total_quality_array is None and total_quality:
            calobj.total_quality_array = np.ones(
                calobj._total_quality_array.expected_shape(calobj)
            )
