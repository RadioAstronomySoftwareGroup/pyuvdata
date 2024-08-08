# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for apply_uvflag function."""

import numpy as np
import pytest

from pyuvdata import UVFlag
from pyuvdata.utils import apply_uvflag


@pytest.mark.filterwarnings("ignore:The shapes of several attributes will be changing")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_apply_uvflag(uvcalibrate_uvdata_oldfiles):
    # load data and insert some flags
    uvd = uvcalibrate_uvdata_oldfiles
    uvd.flag_array[uvd.antpair2ind(9, 20)] = True

    # load a UVFlag into flag type
    uvf = UVFlag(uvd)
    uvf.to_flag()

    # insert flags for 2 out of 3 times
    uvf.flag_array[uvf.antpair2ind(9, 10)[:2]] = True

    # apply flags and check for basic flag propagation
    uvdf = apply_uvflag(uvd, uvf, inplace=False)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])

    # test inplace
    uvdf = uvd.copy()
    apply_uvflag(uvdf, uvf, inplace=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])

    # test flag missing
    uvf2 = uvf.select(bls=uvf.get_antpairs()[:-1], inplace=False)
    uvdf = apply_uvflag(uvd, uvf2, inplace=False, flag_missing=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(uvf.get_antpairs()[-1])])
    uvdf = apply_uvflag(uvd, uvf2, inplace=False, flag_missing=False)
    assert not np.any(uvdf.flag_array[uvdf.antpair2ind(uvf.get_antpairs()[-1])])

    # test force polarization
    uvdf = uvd.copy()
    uvdf2 = uvd.copy()
    uvdf2.polarization_array[0] = -6
    uvdf += uvdf2
    uvdf = apply_uvflag(uvdf, uvf, inplace=False, force_pol=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    with pytest.raises(
        ValueError, match="Input uvf and uvd polarizations do not match"
    ):
        apply_uvflag(uvdf, uvf, inplace=False, force_pol=False)

    # test unflag first
    uvdf = apply_uvflag(uvd, uvf, inplace=False, unflag_first=True)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    assert not np.any(uvdf.flag_array[uvdf.antpair2ind(9, 20)])

    # convert uvf to waterfall and test
    uvfw = uvf.copy()
    uvfw.to_waterfall(method="or")
    uvdf = apply_uvflag(uvd, uvfw, inplace=False)
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 10)][:2])
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(9, 20)][:2])
    assert np.all(uvdf.flag_array[uvdf.antpair2ind(20, 22)][:2])

    # test mode exception
    uvfm = uvf.copy()
    uvfm.mode = "metric"
    with pytest.raises(ValueError, match="UVFlag must be flag mode"):
        apply_uvflag(uvd, uvfm)

    # test polarization exception
    uvd2 = uvd.copy()
    uvd2.polarization_array[0] = -6
    uvf2 = UVFlag(uvd)
    uvf2.to_flag()
    uvd2.polarization_array[0] = -8
    with pytest.raises(
        ValueError, match="Input uvf and uvd polarizations do not match"
    ):
        apply_uvflag(uvd2, uvf2, force_pol=False)

    # test time and frequency mismatch exceptions
    uvf2 = uvf.select(frequencies=uvf.freq_array[:2], inplace=False)
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched frequency arrays"
    ):
        apply_uvflag(uvd, uvf2)

    uvf2 = uvf.copy()
    uvf2.freq_array += 1.0
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched frequency arrays"
    ):
        apply_uvflag(uvd, uvf2)

    uvf2 = uvf.select(times=np.unique(uvf.time_array)[:2], inplace=False)
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched time arrays"
    ):
        apply_uvflag(uvd, uvf2)

    uvf2 = uvf.copy()
    uvf2.time_array += 1.0
    with pytest.raises(
        ValueError, match="UVFlag and UVData have mismatched time arrays"
    ):
        apply_uvflag(uvd, uvf2)

    # assert implicit broadcasting works
    uvf2 = uvf.select(frequencies=uvf.freq_array[:1], inplace=False)
    uvd2 = apply_uvflag(uvd, uvf2, inplace=False)
    assert np.all(uvd2.get_flags(9, 10)[:2])
    uvf2 = uvf.select(times=np.unique(uvf.time_array)[:1], inplace=False)
    uvd2 = apply_uvflag(uvd, uvf2, inplace=False)
    assert np.all(uvd2.get_flags(9, 10))
