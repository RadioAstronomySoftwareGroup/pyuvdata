# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for helper utility functions."""
import os

from astropy.io import fits

from pyuvdata import utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings

casa_tutorial_uvfits = os.path.join(
    DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits"
)


def test_deprecated_utils_import():

    with fits.open(casa_tutorial_uvfits, memmap=True) as hdu_list:
        vis_hdu = hdu_list[0]

        with check_warnings(
            DeprecationWarning,
            match="The _fits_indexhdus function has moved, please import it as "
            "pyuvdata.utils.io.fits._indexhdus. This warnings will become an "
            "error in version 3.2",
        ):
            utils._fits_indexhdus(hdu_list)

        with check_warnings(
            DeprecationWarning,
            match="The _fits_gethduaxis function has moved, please import it as "
            "pyuvdata.utils.io.fits._gethduaxis. This warnings will become an "
            "error in version 3.2",
        ):
            utils._fits_gethduaxis(vis_hdu, 5)
