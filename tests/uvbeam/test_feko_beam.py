import os

import numpy as np
import pytest

from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH

filename = "OVRO_LWA_x.ffe"
feko_folder = "OVRO_LWA_FEKOBeams"
feko_filename = os.path.join(DATA_PATH, feko_folder, filename)
feko_filename2 = feko_filename.replace("x", "y")


@pytest.mark.parametrize(("btype"), [("power"), ("efield")])
def test_read_beam(btype):
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam_feko1 = beam1.from_file(
        feko_filename,
        beam_type=btype,
        frequency=None,
        feed_pol="x",
        telescope_name="LWA",
        feed_name="LWA",
        feed_version="1",
        model_name="FEKO_MROsoil",
        model_version="1.0",
        mount_type="fixed",
        feed_angle=90.0,  # E/W
    )

    beam_feko2 = beam2.from_file(
        feko_filename2,
        beam_type=btype,
        frequency=None,
        feed_pol="y",
        telescope_name="LWA",
        feed_name="LWA",
        feed_version="1",
        model_name="FEKO_MROsoil",
        model_version="1.0",
        mount_type="fixed",
        feed_angle=0.0,  # N/S
    )
    if btype == "power":
        assert beam_feko1.beam_type == "power"
        assert beam_feko2.beam_type == "power"
        assert np.allclose(
            beam_feko1.data_array[0, :, :, 0, np.where(beam_feko1.axis1_array == 0)[0]],
            beam_feko1.data_array[
                0, :, :, 0, np.where(beam_feko1.axis1_array == np.pi / 2.0)[0]
            ],
        )
        assert beam_feko1.data_array.shape == (1, 1, 3, 181, 181)
    else:
        assert beam_feko1.beam_type == "efield"
        assert beam_feko2.beam_type == "efield"
        assert beam_feko1.data_array.shape == (2, 1, 3, 181, 181)
        assert beam_feko2.data_array.shape == beam_feko1.data_array.shape

    assert len(beam_feko1.freq_array) == 3
    assert len(beam_feko1.freq_array) == len(beam_feko2.freq_array)
