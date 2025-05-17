# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for hdf5 utility functions."""

import numpy as np
import pytest

import pyuvdata.utils.io.hdf5 as hdf5_utils
from pyuvdata import data, utils


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_slicing():
    """Test HDF5 slicing helper functions"""
    # check trivial slice representations
    slices, _ = utils.tools._convert_to_slices([])
    assert slices == [slice(0, 0, None)]
    slices, _ = utils.tools._convert_to_slices(10)
    assert slices == [slice(10, 11, 1)]

    # dataset shape checking
    # check various kinds of indexing give the right answer
    indices = [slice(0, 10), 0, [0, 1, 2], [0]]
    dset = np.empty((100, 1, 1024, 2), dtype=np.float64)
    shape, _ = hdf5_utils._get_dset_shape(dset, indices)
    assert tuple(shape) == (10, 1, 3, 1)

    # dataset indexing
    # check various kinds of indexing give the right answer
    slices = [utils.tools._convert_to_slices(ind)[0] for ind in indices]
    slices[1] = 0
    data = hdf5_utils._index_dset(dset, slices)
    assert data.shape == tuple(shape)

    # Handling bool arrays
    bool_arr = np.zeros((10000,), dtype=bool)
    index_arr = np.arange(1, 10000, 2)
    bool_arr[index_arr] = True
    assert utils.tools._convert_to_slices(bool_arr) == utils.tools._convert_to_slices(
        index_arr
    )
    assert utils.tools._convert_to_slices(bool_arr, return_index_on_fail=True) == (
        utils.tools._convert_to_slices(index_arr, return_index_on_fail=True)
    )

    # Index return on fail with two slices
    index_arr[0] = 0
    bool_arr[0:2] = [True, False]

    for item in [index_arr, bool_arr]:
        result, check = utils.tools._convert_to_slices(
            item, max_nslice=1, return_index_on_fail=True
        )
        assert not check
        assert len(result) == 1
        assert result[0] is item

    # Check a more complicated pattern w/ just the max_slice_frac defined
    index_arr = np.arange(0, 100) ** 2
    bool_arr[:] = False
    bool_arr[index_arr] = True

    for item in [index_arr, bool_arr]:
        result, check = utils.tools._convert_to_slices(item, return_index_on_fail=True)
        assert not check
        assert len(result) == 1
        assert result[0] is item


def test_telescope_attr():
    """Test Telescope attribute handling"""
    meta = hdf5_utils.HDF5Meta(f"{data.DATA_PATH}/zen.2458661.23480.HH.uvh5")

    assert meta.telescope.location == meta.telescope_location_obj
