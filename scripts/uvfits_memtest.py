#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Test memory usage of read_uvfits."""

from memory_profiler import profile
import numpy as np
from astropy import constants as const
from astropy.io import fits
from pyuvdata import UVData


@profile
def read_uvfits():
    """Test memory usage of read_uvfits."""
    filename = "/Volumes/Data1/mwa_uvfits/1066571272.uvfits"

    # first test uvdata.read_uvfits. First read metadata then full data
    uv_obj = UVData()
    uv_obj.read_uvfits(filename, read_data=False, read_metadata=False)
    uv_obj.read_uvfits_metadata(filename)
    uv_obj.read_uvfits_data(filename)
    del uv_obj

    # now test uvdata.read_uvfits with select on read.
    uv_obj = UVData()
    uv_obj.read_uvfits(filename, read_data=False, read_metadata=False)
    uv_obj.read_uvfits_metadata(filename)
    uv_obj.read_uvfits_data(filename, freq_chans=np.arange(196))
    del uv_obj

    # now test details with astropy
    hdu_list = fits.open(filename, memmap=True)
    vis_hdu = hdu_list[0]

    # only read in times, then uvws, then visibilities
    time0_array = vis_hdu.data.par("date")
    uvw_array = (
        np.array(
            np.stack(
                (vis_hdu.data.par("UU"), vis_hdu.data.par("VV"), vis_hdu.data.par("WW"))
            )
        )
        * const.c.to("m/s").value
    ).T

    if vis_hdu.header["NAXIS"] == 7:

        data_array = (
            vis_hdu.data.data[:, 0, 0, :, :, :, 0]
            + 1j * vis_hdu.data.data[:, 0, 0, :, :, :, 1]
        )
    else:
        data_array = (
            vis_hdu.data.data[:, 0, 0, :, :, 0]
            + 1j * vis_hdu.data.data[:, 0, 0, :, :, 1]
        )
        data_array = data_array[:, np.newaxis, :, :]

    # test for releasing resources
    del time0_array
    del uvw_array
    del data_array

    # release file handles
    del vis_hdu
    del hdu_list

    # now test reading a slice of the data
    hdu_list = fits.open(filename, memmap=True)
    vis_hdu = hdu_list[0]
    Nfreqs = vis_hdu.header["NAXIS4"]
    freq_index = int(Nfreqs // 2)

    if vis_hdu.header["NAXIS"] == 7:

        data_slice = (
            vis_hdu.data.data[:, 0, 0, :, 0:freq_index, :, 0]
            + 1j * vis_hdu.data.data[:, 0, 0, :, 0:freq_index, :, 1]
        )
    else:
        data_slice = (
            vis_hdu.data.data[:, 0, 0, 0:freq_index, :, 0]
            + 1j * vis_hdu.data.data[:, 0, 0, 0:freq_index, :, 1]
        )
        data_slice = data_slice[:, np.newaxis, :, :]

    del data_slice

    del vis_hdu
    del hdu_list
    del filename

    return


if __name__ == "__main__":
    read_uvfits()
