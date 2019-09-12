# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np

from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam
import pyuvdata.tests as uvtest

filename = os.path.join(DATA_PATH, 'mwa_full_EE_test.h5')


def test_read_write_mwa():
    """Basic read/write test."""
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam1.read_mwa_beam(filename, pixels_per_deg=1)

    assert beam1.pixel_coordinate_system == 'az_za'
    assert beam1.beam_type == 'efield'
    assert beam1.data_array.shape == (2, 1, 2, 3, 91, 360)
    assert np.isclose(np.max(np.abs(beam1.data_array)), 0.6823676193472403)

    assert 'x' in beam1.feed_array
    assert 'y' in beam1.feed_array
    assert beam1.x_orientation == 'east'

    outfile_name = os.path.join(DATA_PATH, 'test', 'mwa_beam_out.fits')
    beam1.write_beamfits(outfile_name, clobber=True)

    beam2.read_beamfits(outfile_name)

    assert beam1 == beam2
