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
